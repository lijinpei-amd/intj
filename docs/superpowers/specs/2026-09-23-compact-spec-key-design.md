# Compact spec key

## Problem

The spec key is built once per launch, then hashed and compared once per launch.
Its size is therefore on the hot path twice, and today it is far larger than the
information it carries.

`INTJ_WORD` spends a full 64-bit word per non-constexpr parameter:

```c
#define INTJ_WORD(tag, dtype, flags)                                           \
  (((uint64_t)(tag) << 56) | ((uint64_t)(uint32_t)(dtype) << 8) |              \
   (uint64_t)(flags))
```

That word holds a tag in `0..13`, a torch dtype code in `0..63`, and two flag
bits. Nine bits of payload in sixty-four. Constexpr parameters take two words,
the second a real 64-bit value and the first a 4-bit tag.

On a representative kernel -- six tensors, three int arguments, three
`tl.constexpr` -- that is `1 + 9 + 6 = 16` words, 128 bytes. The cost lands in
three places:

- `intj_hash` runs a serial chain of sixteen `intj_mix` calls, each a
  64x64->128 multiply, between the last argument decode and the first table
  probe.
- `intj_key_eq` memcmps 128 bytes.
- `intj_slot` embeds the key inline, so a slot is `8 + 8 + 128 = 144` bytes and
  a single probe touches three cache lines.

The third matters most: a probe is a cache miss, and the slot size decides how
many lines that miss costs.

## Goals

- Shrink the key to what it actually distinguishes, without making it coarser.
- Keep the key's length a compile-time constant, so `memcmp` and the hash stay
  constant-sized.
- Specialize the hash and the slot on the key size the renderer computes.
- Keep the failure mode of any render-time offset mistake loud.

## Non-goals

- Variable-length encoding (msgpack-style varints, or eliding `None`/`bool`
  constexpr payloads). Considered and rejected: `intj_slot` embeds the key
  inline and is sized by the *maximum* key, so a smaller typical key does not
  shrink the slot or the probe's cache footprint. It would only shrink the bytes
  fed to `intj_hash`, and it pays for that by turning `INTJ_NWORDS` from a
  compile-time constant into a runtime trip count. Making the slot genuinely
  variable means storing the key behind a pointer -- one more dependent cache
  miss on every compare, before you know whether the entry is even the right
  one.
- Narrowing constexpr payloads below 64 bits. The payload width is a runtime
  property: the same `tl.constexpr` parameter can be `None` on one call, `128`
  on the next and `2**40` after that. The renderer knows only that the parameter
  *is* constexpr.
- A key bit for pointer constness. See below: it cannot vary between two keys
  that could share a slot.

## Design

### Constness needs no bits

Triton spells a pointer-to-const `const *fp32`, normalizing to `*kfp32`
(`jit.py:278-283`). `KernelParam.is_const` reads `self.annotation`
(`jit.py:338-341`), never the runtime value, so constness is a property of the
kernel signature -- fixed for a parameter position across every call to a
rendered module. Two keys that could land in the same slot always agree on it.

It is also narrower than it looks. `const` is rejected on anything but a
pointer (`_normalize_ty` asserts the normalized type starts with `*`, so
`const i32` raises), and `pointer_type.to_ir` (`core.py:678`) drops constness
entirely -- it calls `builder.get_ptr_ty(element_ty, address_space)`, which
takes no const argument. Constness never reaches the MLIR type. Its only effect
is `semantic.py:1210` and `:1270`, raising `ValueError: Cannot store to a
constant pointer`, plus a distinct mangled name (`core.py:676`) that gives
const and non-const variants separate compile-cache entries holding identical
machine code.

`launcher.py:611` refuses any annotated non-constexpr parameter today, so
nothing can reach intj's key regardless.

### Compact dtype index

The key encodes a 5-bit index over the dtypes triton can actually take, not
torch's raw `ScalarType` code.

Measured on torch 2.9.1 and triton 3.8: torch has **46 distinct ScalarType
codes, 0..45**, of which triton accepts **19**. `canonicalize_dtype`
(`_utils.py`) is a plain dict index, so everything else -- `chalf` `cfloat`
`cdouble`, the `qint*` family, `bits1x8`..`bits16`, `uint2`..`uint7`,
`int2`..`int7`, `float8_e8m0fnu`, `float4_e2m1fn_x2` -- raises before intj is
involved.

That is the argument for compacting. The raw code sits at 45 of a 64-code
space, and the tail of that list is recent; 18 codes of headroom is a handful of
torch releases. A 5-bit index over the triton-usable set is 19 of 32, and it
grows only when *triton* adds support, which is far slower.

It also collapses the layout. At 6 bits, `64 dtypes x D x S` is exactly 256
codes -- the whole byte, with nowhere for the scalar tags -- which would force
`S` out into a separate bit-packed section. At 5 bits everything fits one byte
with room to spare.

`intj_torch_abi` gains a table beside `itemsize`:

```c
uint8_t dtype_index[INTJ_NDTYPES];   /* torch ScalarType code -> 0..31, 0xFF = unsupported */
```

`torch_abi.py` builds it exactly as `itemsize_table()` builds its neighbour --
from the live torch, walking `torch`'s dtype singletons, assigning indices in
`ScalarType` code order to those present in triton's
`type_canonicalisation_dict`. Building it from the running torch and the
installed triton, rather than baking it in, follows the rule the torch-access
spec set: nothing version-specific in the binary unless it must be.

`0xFF` *is* the bound check. A dtype intj has no index for is refused with the
same `RuntimeError` the SHIM reader already raises for an unknown code. That
closes a gap worth naming, because the new encoding would otherwise make it a
silent wrong launch rather than a wide field: the SHIM reader's existing
`code >= INTJ_NDTYPES` check sits *after* the `numel == 0` early return, so a
zero-element tensor skips it, and the CPYTHON reader has no bound check at all
(`intj_dtype_selfcheck` validates `torch.float32`'s code, not every dtype's).
The lookup goes in `INTJ_DECODE`'s tensor branch, immediately after
`intj_read_tensor` returns, so one guard covers all three access modes and every
path through them.

**Cost, stated plainly.** `INTJ_ACCESS_CXX` touches `abi` not at all today --
deliberately, and `INTJ_ACCESS_CPYTHON` only for interned names. This puts one
table load on both their hot paths, a cache line neither touched before. SHIM
already loads `abi.itemsize[code]`, so there the second table is free.

It also makes the table mode-independent, where `itemsize` is installed only in
the SHIM branch of `set_torch_version` and CPYTHON passes `layout=None`. So
`set_torch_version` takes a third argument, `dtype_index`, always a 64-byte
`bytes`, in every mode; `TensorLayout.as_args` keeps carrying `itemsize` for
SHIM alone.

### Layout

Every field is a whole number of bytes at a render-time offset. Nothing
straddles a word boundary, offsets are plain indices rather than shift
constants, and `_validate_spec_key` reads `keyblob[i]`.

```
bytes 0 .. nparams-1    one byte per declared parameter, in declaration order
byte nparams            device ordinal
pad to a word boundary, zeroed
then                    one 64-bit value word per constexpr parameter
```

```python
header_words = ceil((nparams + 1) / 8)
nwords       = header_words + n_cx
```

Parameter `i` is at byte `i`, so `_validate_spec_key` indexes `keyblob[i]` with
no offset to keep in step with the template.

Sizes, all counting `spec_pointer_range` as irrelevant now that `S` lives in the
byte -- HIP and CUDA are identical:

| kernel | today | packed |
|---|---|---|
| 6 tensors + 3 ints + 3 constexpr | 16 words / 128 B | 5 words / 40 B |
| `axpy` (3 tensors, int, float, bool, None, 1 constexpr) | 10 words | 3 words |
| `add_kernel(x, y, out, n, BLOCK_SIZE)` | 7 words | 2 words |

Slot size follows: 144 bytes to 56 for the first.

### The parameter byte

One byte per parameter. The layout is uniform: `do_not_specialize` and
`do_not_specialize_on_alignment` change which codes a parameter can *produce*,
never where its byte sits or how wide it is.

```
0 .. 127      pointer:   (dtype_index << 2) | (D << 1) | S      /* 19 of 32 indices live */
128, 129      I32,  D = 0, 1
130, 131      I64,  D = 0, 1
132, 133      U64,  D = 0, 1
134           FP32
135           U1     (bool)
136           NONE
137           ONE    (int 1 folded to a constexpr; only emitted when spec)
138           CX_NONE
139           CX_BOOL      (value word carries 0 or 1)
140           CX_INT
141           CX_UINT
142           CX_FLOAT     (value word carries the double's bits)
143 .. 255    free
```

No separate "is pointer" bit -- the range partition carries it. No `S` section.

Constexpr and non-constexpr codes are disjoint even though a byte position is
always known at render time to be one or the other. It costs nothing from the
113 spare codes, and it turns a wrong render-time offset into a nonsense code
rather than a silent alias between a constexpr and a pointer.

### Building the key

Byte stores into a `uint64_t[]` followed by a word-sized load for the hash is a
narrow-store-to-wide-load forward, which stalls. So the header is accumulated in
registers and stored to the stack a word at a time.

The accumulators are `uint32_t`, one per four parameter bytes, which keeps the
decode in the same width the macro already works in -- `intj_read_tensor` yields
an `int32_t` dtype and `INTJ_DECODE` builds `uint32_t` flags -- so no operand is
widened before it is placed:

```c
uint64_t key[INTJ_NWORDS] = {0};
uint32_t _a0 = 0, _a1 = 0;
INTJ_DECODE(st, args[3], _a0, 0,  vals, np, 1, 1, 1, ...);   /* byte 0 */
INTJ_DECODE(st, args[4], _a0, 8,  vals, np, 1, 1, 1, ...);   /* byte 1 */
INTJ_DECODE(st, args[5], _a0, 16, vals, np, 1, 1, 1, ...);   /* byte 2 */
INTJ_DECODE(st, args[6], _a0, 24, vals, np, 1, 1, 1, ...);   /* byte 3 */
INTJ_DECODE(st, args[7], _a1, 0,  vals, np, 1, 1, 1, ...);   /* byte 4 */
...
key[0] = (uint64_t)_a0 | ((uint64_t)_a1 << 32);
```

**Each 64-bit word gets exactly one store.** Two adjacent 32-bit stores would
reintroduce the stall this section exists to avoid: x86 store-to-load forwarding
requires the load to be contained in a single store, so an 8-byte load fed by
two 4-byte stores falls back to the store buffer -- the same dozen-odd cycles as
the byte-store version. The pair is folded in a register first.

The shift is always a multiple of eight, under 32, and known at render time, so
the macro's write is a single `|=` with no straddle case. `INTJ_DECODE` gains
the accumulator and shift in place of the `(word)` lvalue it writes today.

The device byte is placed the same way, at accumulator `nparams / 4` and shift
`(nparams % 4) * 8`.

The zero-init covers the header's padding bytes, which `memcmp` compares.
Constexpr value words are written unconditionally and need none.

`device` is one byte, so `entry` gains an upper bound next to the existing
`device < 0` check. Without it, device 256 aliases device 0 -- a wrong-context
launch.

The key is read back byte-wise on the Python side, so the layout is
little-endian-dependent. `_validate_spec_key` already assumes that
(`struct.unpack("<...Q")`).

### Render-time hash and slot

`INTJ_NWORDS` is already a compile-time constant, so `intj_hash`'s loop unrolls
and `memcmp`'s size is a literal. Packing alone therefore speeds both up with no
new code, and `intj_key_eq` needs no change -- a constant-size `memcmp` already
emits the optimal sequence, and replacing it with a chain of `==` would add
branches. Three cases do want code.

**`nwords == 1`: drop the stored key.** Replace the wyhash mix with a bijective
finalizer:

```c
static inline uint64_t intj_hash1(uint64_t x) {
  x ^= x >> 30; x *= 0xbf58476d1ce4e5b9ull;
  x ^= x >> 27; x *= 0x94d049bb133111ebull;
  x ^= x >> 31;
  return x;
}
```

Xorshift-right and multiplication by an odd constant are both bijections on
`uint64_t`, so `s->hash == h` *is* key equality. The slot becomes
`{ uint64_t hash; intj_kernel *val; }` -- 16 bytes, four per cache line -- and
`intj_map_get` drops its `intj_key_eq` call, `intj_map_insert` its `memcpy`. The
tsl and absl backends key on a bare `uint64_t`.

Reachable for a kernel with no constexpr parameters and at most seven
parameters, which given `BLOCK_SIZE` is the minority. Taken anyway.

**`nwords == 2`: one multiply.** `intj_mix(w[0] ^ s0, w[1] ^ s1)` plus the
finalizer, the way wyhash handles short inputs, instead of two chained mixes.
This is the common small case, not a corner:
`add_kernel(x, y, out, n, BLOCK_SIZE: tl.constexpr)` is one device byte and five
parameter bytes -- one header word -- plus one value word.

**`nwords >= 3`: two lanes.** `h = intj_mix(h ^ s1, w[i] ^ s0)` is serial, and
each mix is a multiply of roughly four cycles' latency. The whole chain sits
between the last argument decode and the first table probe: about twenty cycles
at five words. Two independent accumulators folded at the end cut it to about
twelve, for the same reason wyhash runs three lanes over its 48-byte blocks.

All three are `{% if %}` forks in `intj_runtime.h`, selected by `nwords`.

### Python side

`Param.word` is replaced by:

- `byte` -- the parameter's header byte index, which is just its position.
- `cx_word` -- the value word index, constexpr parameters only.

The accumulator and shift the template emits are derived from `byte`
(`byte // 4`, `(byte % 4) * 8`), not stored.

`_validate_spec_key` reads `keyblob[param.byte]` plus, for a constexpr, the
value word at `cx_word`, and compares that against triton's specialization
exactly as it does today. The invariant is unchanged: the same intj key must
never map to two different triton specializations.

`_render_context`'s `nwords` computation at `launcher.py:165`, the `word`
assignment in `_render_params` at `launcher.py:606`, and `spec_pointer_range`'s
use in the template all move to the formulas above. `spec_pointer_range` itself
stays -- it still gates whether `intj_read_tensor` is asked for the storage size
and whether `S` can be set -- but it no longer affects the layout.

## Testing

`test_spec_key_is_never_coarser_than_triton` is the load-bearing test and needs
no change: it treats the key as an opaque blob and asserts that equal keys imply
equal triton specializations. It covers the new encoding's aliasing risk
directly, across dtypes, alignments, `> 2 GiB` storages and int widths.

New:

- A dtype-index test: a tensor whose dtype maps to `0xFF` must raise, including
  the zero-element case that today skips the SHIM reader's bound check. `chalf`
  is a convenient unsupported dtype that torch can allocate.
- `dtype_index_table()` against the live torch: every triton-usable dtype gets a
  distinct index under 32, everything else `0xFF`, and the table is `NDTYPES`
  long.
- A device-bound test: `device >= 256` must raise rather than alias.
- A layout test on `_render_params`: byte offsets, `cx_word` and `nwords` for a
  parameter list spanning constexpr, `do_not_specialize` and
  `do_not_specialize_on_alignment`.
- `tests/bench_kernel_cache.cpp` is parameterized on `INTJ_NWORDS` and currently
  built at 5 and 11. Add 1 and 2 so the `nwords == 1` and `nwords == 2`
  specializations are measured rather than assumed.

The `Param` fixture in `_render_context` constructs positionally and needs
updating with the field list.

`ModuleKey` carries `entry.c.jinja`'s bytes (`launcher.py:191`) and the runtime
header's digest (`launcher.py:408`), so a `.so` built against the old layout is
not reused after either file changes.

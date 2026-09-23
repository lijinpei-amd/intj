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
the second of which is a real 64-bit value and the first of which is a 4-bit
tag.

On a representative kernel -- six tensors, three int arguments, three
`tl.constexpr` -- that is `1 + 9 + 6 = 16` words, 128 bytes. The cost lands in
three places:

- `intj_hash` runs a serial chain of sixteen `intj_mix` calls, each a
  64x64->128 multiply, between the last argument decode and the first table
  probe.
- `intj_key_eq` memcmps 128 bytes.
- `intj_slot` embeds the key inline, so a slot is `8 + 8 + 128 = 144` bytes and
  a single probe touches three cache lines.

The third is the one that matters most: a probe is a cache miss, and the slot
size decides how many lines that miss costs.

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
- Shrinking `INTJ_NDTYPES` below 64. It would buy a byte, at the cost of intj
  refusing a legitimate kernel the day torch adds its 33rd dtype.

## Design

### Byte-granular layout

Every field is a whole number of bytes, at a render-time offset. Nothing
straddles a word boundary, offsets are plain indices rather than shift
constants, and `_validate_spec_key` reads `keyblob[i]` instead of doing bit
arithmetic.

```
byte 0                  device ordinal
bytes 1 .. nparams      one byte per declared parameter, in declaration order
then                    S bits, one per non-constexpr parameter   (HIP only)
pad to a word boundary, zeroed
then                    one 64-bit value word per constexpr parameter
```

The renderer computes, for a parameter list of `nparams` parameters of which
`n_noncx` are not constexpr and `n_cx` are:

```python
off_sbits    = 1 + nparams
sbytes       = ceil(n_noncx / 8) if spec_pointer_range else 0
header_words = ceil((off_sbits + sbytes) / 8)
nwords       = header_words + n_cx
```

The representative kernel above: `1 + 12 + 2 = 15` header bytes -> 2 words, plus
3 constexpr value words. **5 words, 40 bytes, from 16 words and 128 bytes.** The
slot drops from 144 bytes to 56.

`axpy` in the test suite -- three tensors, an int, a float, a bool, a `None` and
one constexpr -- goes from 10 words to 3.

### The parameter byte

One byte per parameter. The layout is uniform: `do_not_specialize` and
`do_not_specialize_on_alignment` change which codes a parameter can *produce*,
never where its byte sits or how wide it is.

Non-constexpr:

```
0 .. 127      pointer:  (torch dtype code << 1) | D
128, 129      I32,  D = 0, 1
130, 131      I64,  D = 0, 1
132, 133      U64,  D = 0, 1
134           FP32
135           U1    (bool)
136           NONE
137           ONE   (int 1 folded to a constexpr; only emitted when spec)
```

Constexpr:

```
138           CX_NONE
139           CX_BOOL      (value word carries 0 or 1)
140           CX_INT
141           CX_UINT
142           CX_FLOAT     (value word carries the double's bits)
```

The two ranges are disjoint even though a byte position is always known at
render time to be one or the other. It costs nothing, and it turns a wrong
render-time offset into a nonsense code rather than a silent alias between a
constexpr and a pointer.

Why `D` fits in the byte and `S` does not. A dense encoding of
`64 dtypes x D x S` is exactly 256 codes -- the entire space -- and the ten
scalar codes above need somewhere to live. `S` is the one flag that is only ever
set for pointers (`INTJ_DECODE` sets `INTJ_FLAG_S` inside the tensor branch
only), so it is the one that can be lifted out into its own section without
leaving a hole. `D` applies to both pointers and ints, so it stays in the byte.

`S` gets a bit per non-constexpr parameter, not per specialized parameter, for
the same uniformity reason. On CUDA `Backend.pointer_range` is `False`
(`launcher.py:515`) and the section is empty.

### Bounding the dtype code

The encoding requires `dtype < 64`. Today nothing guarantees it on every path:

- The SHIM reader checks `code >= INTJ_NDTYPES` only after the `numel == 0`
  early return, so a zero-element tensor skips the check entirely.
- The CPYTHON reader reads the `ScalarType` byte with no bound check at all;
  `intj_dtype_selfcheck` validates `torch.float32`'s code, not every dtype's.
- The CXX reader is bounded by `isScalarType()`, which is under 64 today but is
  torch's number to change.

Harmless while the dtype occupies 32 bits of a 64-bit word. Under the new
encoding a code of 64 or more aliases into the scalar range -- a wrong kernel
launch, silently.

Fix it once, in `INTJ_DECODE`'s tensor branch, immediately after
`intj_read_tensor` returns, rather than in each of the three readers. One guard
covers all three access modes and every path through them, including the
zero-element one. It raises the same `RuntimeError` the SHIM reader already
raises for an out-of-range code.

### Building the key

Byte stores into a `uint64_t[]` followed by a word-sized load for the hash is a
narrow-store-to-wide-load forward, which stalls. So the header is accumulated in
`uint64_t` locals and stored once per word:

```c
uint64_t key[INTJ_NWORDS] = {0};
uint64_t _h0 = (uint64_t)device;          /* byte 0 */
INTJ_DECODE(st, args[3], _h0, 8,  vals, np, 1, 1, 1, ...);   /* byte 1 */
INTJ_DECODE(st, args[4], _h0, 16, vals, np, 1, 1, 1, ...);   /* byte 2 */
...
key[0] = _h0;
```

The shift is always a multiple of eight and always known at render time, so the
macro's write is a single `|=` with no straddle case. `INTJ_DECODE` gains the
accumulator and shift in place of the `(word)` lvalue it writes today, and gains
a second accumulator/shift pair for the `S` bit; on CUDA the renderer passes a
shift the macro compiles away.

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
branches. Three cases do want code:

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
`intj_map_get` drops its `intj_key_eq` call, `intj_map_insert` its `memcpy`.
The tsl and absl backends key on a bare `uint64_t`.

Reachable only for kernels with no constexpr parameters, which given
`BLOCK_SIZE` is the minority. Taken anyway.

**`nwords == 2`: one multiply.** `intj_mix(w[0] ^ s0, w[1] ^ s1)` plus the
finalizer, the way wyhash handles short inputs, instead of two chained mixes.
This is the common small case, not a corner:
`add_kernel(x, y, out, n, BLOCK_SIZE: tl.constexpr)` is one device byte, four
parameter bytes, one constexpr tag byte and (on HIP) one S byte -- one header
word and one value word.

**`nwords >= 3`: two lanes.** `h = intj_mix(h ^ s1, w[i] ^ s0)` is serial, and
each mix is a multiply of roughly four cycles' latency. The whole chain sits
between the last argument decode and the first table probe: about twenty cycles
at five words. Two independent accumulators folded at the end cut it to about
twelve, for the same reason wyhash runs three lanes over its 48-byte blocks.

All three are `{% if %}` forks in `intj_runtime.h`, selected by `nwords`.

### Python side

`Param.word` is replaced by:

- `byte` -- the parameter's header byte index, for every parameter.
- `shift` / `accumulator` -- derived from `byte`, emitted by the template.
- `sbyte` / `sshift` -- the S bit's position; unused when
  `spec_pointer_range` is 0.
- `cx_word` -- the value word index, constexpr parameters only.

`_validate_spec_key` reads `keyblob[param.byte]` plus, for a constexpr, the
value word at `cx_word`, plus the S bit, and compares that tuple against
triton's specialization exactly as it does today. The invariant it enforces is
unchanged: the same intj key must never map to two different triton
specializations.

`_render_context`'s `nwords` computation at `launcher.py:165` and the `word`
assignment in `_render_params` at `launcher.py:606` both move to the formula
above.

## Testing

`test_spec_key_is_never_coarser_than_triton` is the load-bearing test and needs
no change -- it treats the key as an opaque blob and asserts that equal keys
imply equal triton specializations. It covers the new encoding's aliasing risk
directly, across dtypes, alignments, `> 2 GiB` storages and int widths.

New:

- A dtype-bound test: a tensor whose dtype code is at or above `INTJ_NDTYPES`
  must raise, including the zero-element case that today skips the check.
- A device-bound test: `device >= 256` must raise rather than alias.
- A layout test on `_render_params`: byte offsets, S-bit positions and `nwords`
  for a parameter list spanning constexpr, `do_not_specialize` and
  `do_not_specialize_on_alignment`, on both `spec_pointer_range` values.
- `tests/bench_kernel_cache.cpp` is parameterized on `INTJ_NWORDS` and currently
  built at 5 and 11. Add 1 and 2 so the `nwords == 1` and `nwords == 2`
  specializations are measured rather than assumed.

The `Param` fixture in `_render_context` constructs positionally and needs
updating with the field list.

`ModuleKey` carries `entry.c.jinja`'s bytes (`launcher.py:191`) and the runtime
header's digest (`launcher.py:408`), so a `.so` built against the old layout is
not reused after either file changes.

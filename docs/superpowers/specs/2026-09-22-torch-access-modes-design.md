# Torch access modes

## Problem

intj reads three things off every tensor argument: the data pointer, the dtype,
and (on AMD) the storage size. Today it does that one way only: derive an
`AtenTensorHandle` from the `THPVariable` layout and call the AOTI shims.

That derivation, `(char *)obj + sizeof(PyObject)` in `INTJ_TENSOR_HANDLE`, is
correct only on torch >= 2.10. Before 2.10 `THPVariable::cdata` is a
`c10::MaybeOwned<at::Tensor>`, whose `bool isBorrowed_` pushes the tensor to
`+8`. On torch 2.3 through 2.9 the module loads, `intj_exec` succeeds, and the
first launch passes `&isBorrowed_` to `aoti_torch_get_data_ptr`. Segfault or a
garbage pointer, with no diagnostic.

Worse, the torch version appears nowhere in `ModuleKey`, and `libtorch_path` is
the same string across versions. A `.so` built under 2.14 is reused verbatim
after a downgrade to 2.9.

`create_launcher` should let the caller choose how the generated module reaches
torch, and should pick a safe default when they do not.

## Goals

- One compiled `.so` per kernel, usable across as many torch versions as
  possible, with no rebuild when torch changes.
- Never crash on an unsupported torch. Degrade to a slower path or raise.
- No measurable cost on the fast path, which is the whole point of intj: the
  5-argument benchmark decodes in 0.15 us.
- Keep the version policy in Python, where it is testable.

## Non-goals

- A mode that compiles the extension as C++ against torch's headers. Runtime
  dispatch reaches the same versions without a C++ build, torch includes, or
  link flags. Revisit only if a read appears that the shims cannot serve.
- Tensor subclasses beyond `nn.Parameter`. Orthogonal, and a widened type test
  would accept `FakeTensor` and `DTensor`, which have no dense data pointer.

## Background: what the stable ABI does and does not provide

| What intj needs | Stable ABI | Since |
|---|---|---|
| Is this `PyObject *` a tensor? | nothing | - |
| `PyObject *` -> `AtenTensorHandle` | nothing | - |
| data pointer | `aoti_torch_get_data_ptr` | 2.2 |
| dtype | `aoti_torch_get_dtype` | 2.3 |
| storage size | `aoti_torch_get_storage_size` | 2.2 |

The two gaps are structural. `shim.h` states the contract: "Only pointers
(AtenTensorHandle counts), integers and floats in headers." No amount of
`TORCH_TARGET_VERSION` targeting introduces a `PyObject` parameter. Producing a
handle from a Python object is therefore always an unstable act, which is why
there is no genuinely stable mode to offer.

The shims themselves are safe to bind by name: `shim.h` promises ABI stability
and `_v2` suffixes for breaking changes, and `AOTITorchError` has been `int32_t`
from 2.2 through 2.14. They are exported from `libtorch_cpu.so` only.

## Design

### Runtime dispatch, not compile-time selection

The mode is chosen after the module loads, not when it is rendered. Nothing
about torch enters `RenderContext` or `ModuleKey`, so the `.so` is
torch-version-independent and a torch upgrade neither rebuilds it nor
invalidates it.

Module state gains one field and the three function pointers become
mode-dependent:

```c
typedef struct {
  ...
  size_t tensor_offset;          /* PyObject* -> handle */
  PyObject *dtype_str;           /* interned "dtype", fallback path only */
  intj_get_data_ptr_t get_data_ptr;
  intj_get_storage_size_t get_storage_size;
  intj_get_dtype_t get_dtype;
} intj_state;
```

`INTJ_TENSOR_HANDLE` reads the offset instead of baking it in:

```c
#define INTJ_TENSOR_HANDLE(st, o) \
  ((intj_tensor_handle)((char *)(o) + (st)->tensor_offset))
```

The trick that keeps `INTJ_DECODE` free of mode branches: **`tensor_offset == 0`
makes the handle the `PyObject *` itself**. The fallback readers are intj's own
statics with the same three signatures, taking that pointer. The decode macro
always does `handle = o + st->tensor_offset` then
`st->get_data_ptr(handle, &p)`, exactly as today.

Cost on the fast path: one load from a cache line already hot (the function
pointers live beside it) plus an add. The indirect calls exist today.

### The three modes

`create_launcher(..., torch_access=None)`:

| Value | Behaviour |
|---|---|
| `"shim"` | Handle by offset, all three reads through the AOTI shims. Requires a torch version in the table. |
| `"cpython"` | `tensor_offset = 0`, all three reads through intj's fallbacks. Works on any torch. |
| `None` (default) | Consult the table; `"shim"` if the version is known, `"cpython"` otherwise. |

Explicit `"shim"` on an unknown version raises `UnsupportedKernel`. The default
never raises.

### Version table

Lives in `launcher.py`, keyed on `(major, minor)` parsed from
`torch.__version__`:

| torch | `use_shims` | `extra` |
|---|---|---|
| >= 2.10 | yes | 0 |
| 2.3 - 2.9 | yes | 8 |
| < 2.3, or unrecognised | no | - |

The C side computes the offset, so Python never needs `sizeof(PyObject)` and
free-threaded builds need no special case:

```c
st->tensor_offset = use_shims ? sizeof(PyObject) + extra : 0;
```

The table is closed-ended on the upper end: an unknown future version falls back
rather than guessing a layout. That trades "intj stops being fast on torch 2.15
until someone adds a row" against "intj segfaults on torch 2.15", which is the
right trade for a wrong offset that cannot raise.

torch 2.2 gets no row of its own. Its `THPVariable` offset is the same `+8`, but
`aoti_torch_get_dtype` does not exist until 2.3, and carving out a per-read
exception for one release from January 2024 costs more than it buys. 2.2 falls
back like anything else below the floor.

The `+8` for 2.3 - 2.9 is verified, not assumed: `MaybeOwned` is
`bool isBorrowed_;` then the union at v2.3.0, v2.6.0 and v2.9.0, and
`MaybeOwnedTraits<at::TensorBase>` sets `borrow_type = owned_type =
at::TensorBase`, a single intrusive pointer. The union is 8-aligned, sits at
offset 8, and reinterpreting it as `at::Tensor *` is valid whichever arm is
active, so the flag's value never matters.

### The setter

`_loaded_module` calls `module.set_torch_abi(use_shims, extra)` immediately
after `set_compile_callback`, under the same lock. Both are part of one
post-load configuration step, and neither `entry` nor `spec_key` is reachable
before it completes.

The module refuses to launch if the setter has not run, the way it already
refuses a missing compile callback.

`intj_exec` keeps importing torch (it needs `Tensor` and `Parameter` for the
type test) and keeps dlopening `libtorch_cpu.so`, but resolving the three shim
symbols moves into the setter, so a torch without them is not a load error.

### Dtype without the shim

The fallback reads the `ScalarType` enum straight off the `torch.dtype`
singleton:

```c
/* torch/csrc/Dtype.h:
 *   struct THPDtype { PyObject_HEAD at::ScalarType scalar_type; char name[65]; } */
PyObject *d = PyObject_GetAttr(o, st->dtype_str);   /* interned */
int32_t code = *(const int8_t *)((char *)d + sizeof(PyObject));
```

`at::ScalarType` is `enum class ScalarType : int8_t`. The struct is byte-identical
at v1.13.0, v2.0.0, v2.2.0, v2.5.0, v2.9.0, v2.14.0 and `main` - the only edits in
that range are a `PyObject_HEAD` semicolon and `TORCH_API` ->
`TORCH_PYTHON_API`. Verified against `aoti_torch_get_dtype` across all 57
`torch.dtype` singletons on torch 2.14: zero mismatches.

This is a far weaker bet than the `THPVariable` offset, and unlike that one it
can verify itself. `char name[65]` sits immediately after the enum, so at load
time the module reads the name embedded in `torch.float32` and compares it to
`"float32"`. A dtype object is at least 82 bytes, so the read cannot fault. If
the name does not match, the layout changed and `set_torch_abi` reports it;
Python decides what to do.

Because the dtype is an `int32_t` in every mode, `INTJ_WORD` keeps its 32-bit
dtype field, `nwords` is unchanged, and no `PyObject *` ever enters the spec
key. The dtype's numeric value is explicitly not part of torch's ABI contract,
which is fine: intj uses it as an opaque in-process discriminator and never
persists it.

### The other two fallbacks

- `data_ptr`: `PyObject_CallMethodNoArgs(o, "data_ptr")`, unbox. The one
  unavoidably expensive call, roughly 100-200 ns.
- `storage_size`: `o.untyped_storage().nbytes()`. Only read when the backend
  specializes on pointer range, i.e. AMD.

Both set a Python error and return non-zero on failure, matching the shim
contract, so `INTJ_DECODE`'s error handling is untouched.

## Error handling

- Unknown torch with `torch_access="shim"`: `UnsupportedKernel` at
  `create_launcher` time.
- Missing shim symbol when shims were requested: `RuntimeError` from the setter,
  naming the symbol.
- Dtype layout self-check fails: raise `UnsupportedKernel` naming
  `torch.__version__`. The check only runs on the fallback path, and there is no
  further fallback below it, so intj has no working option and should say so
  rather than guess. Shim mode is unaffected, since it never reads `THPDtype`.
- Everything inside `INTJ_DECODE` keeps its current behaviour: set a Python
  error, `goto error`.

## Testing

Extends `tests/test_launcher.py`, which is already differential against triton.

1. **Both modes agree.** Run the existing
   `test_spec_key_is_never_coarser_than_triton` corpus under `"shim"` and
   `"cpython"` and assert each mode is internally consistent. The two modes
   produce different key bytes (different code paths, same discriminating
   power), so compare each against triton's specialization separately rather
   than against each other.
2. **Dtype agreement.** For every `torch.dtype` singleton, assert the offset
   read equals `aoti_torch_get_dtype`. This is the test that catches a future
   `THPDtype` change.
3. **Version table.** Pure-Python unit test over the `(major, minor)` ->
   `(offset, use_shims)` mapping, including the unknown-version fallback and the
   explicit-`"shim"`-raises case. No GPU needed.
4. **Module key stability.** Assert `ModuleKey.digest()` does not change when
   the torch version changes, which is the regression this design exists to
   prevent.
5. **Fallback correctness.** Launch the existing kernels under
   `torch_access="cpython"` and compare results against triton, covering
   aligned/unaligned pointers and the >2 GiB storage case.

`benchmarks/bench_launch.py` gains a row per mode, so the cost of the fallback
is a number rather than an estimate.

## Migration

`torch_access` defaults to `None`, and on torch >= 2.10 the default resolves to
`"shim"`, which is the current behaviour. No caller changes. The `.so` cache
invalidates once, because the template and runtime header are hashed into
`ModuleKey`.

Bump `SCHEMA_VERSION`.

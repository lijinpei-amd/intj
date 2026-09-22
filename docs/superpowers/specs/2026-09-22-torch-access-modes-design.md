# Torch access modes

## Problem

intj reads three things off every tensor argument: the data pointer, the dtype,
and (on AMD) the storage size. Today it does that one way only: derive an
`AtenTensorHandle` from the `THPVariable` layout and call the AOTI shims in
`libtorch_cpu.so`.

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
torch, and should pick a good default when they do not.

## Goals

- Three explicit strategies, selectable per launcher, plus an automatic default.
- The torch version is a render-time input, so the cache cannot serve a module
  built against a different torch.
- Avoid calling back into the Python interpreter on the launch path wherever the
  version makes that possible.
- No measurable cost on the fast path, which is the whole point of intj: the
  5-argument benchmark decodes in 0.15 us.
- Keep the version policy in Python, where it is testable.

## Non-goals

- Tensor subclasses beyond `nn.Parameter`. Orthogonal, and a widened type test
  would accept `FakeTensor` and `DTensor`, which have no dense data pointer.
- Cross-compiling against a torch that is not installed. `torch_version` selects
  which layout constants are rendered, not which headers exist on disk.

## Design

### Compile-time selection

`RenderContext` gains two fields:

```python
torch_access: str          # "direct" | "cpython" | "cxx"
torch_version: tuple[int, int]
```

Both therefore land in `ModuleKey`, so the `.so` is keyed on
`(mode, torch major.minor)` and the stale-cache bug is fixed by inclusion. Every
layout constant is a render-time literal; the generated C has no mode branches,
no offsets in module state, and no post-load setter.

This costs a rebuild when torch's minor version changes. That is the honest
price of baking layout knowledge into the binary, and it is what makes the
binary correct.

### The modes

`create_launcher(..., torch_access=None, torch_version=None)`:

| Value | How it reads a tensor | Needs |
|---|---|---|
| `"direct"` | Struct offsets, rendered as constants. No dlopen, no shims, no interpreter. | version in the layout table |
| `"cpython"` | `data_ptr()` / `untyped_storage().nbytes()` calls, dtype by `THPDtype` offset. | nothing |
| `"cxx"` | `THPVariable_Unpack`, inlined `at::Tensor` methods. | torch headers, a C++ compiler |
| `None` | `"cxx"` if supported, else `"direct"` if the version is in the table, else `"cpython"`. | - |

`torch_version` defaults to the installed `torch.__version__`. When given, it
selects which layout constants are rendered.

One asymmetry worth stating plainly: in `"cxx"` mode it cannot select, because
the headers come from the installed torch. There it is validated instead, and
`create_launcher` raises if it disagrees with `torch.__version__`.

### `"direct"`: reading the structs

The mode calls nothing. Every read is a load at a rendered constant offset.

```c
/* verified on torch 2.14, x86-64 */
TensorImpl  *ti = *(void **)((char *)obj + {{ thpvariable_cdata }});
StorageImpl *si = *(void **)((char *)ti  + {{ tensorimpl_storage }});
int64_t off     = *(int64_t *)((char *)ti + {{ tensorimpl_storage_offset }});
uint8_t dt      = *(uint8_t *)((char *)ti + {{ tensorimpl_data_type }});
void   *data    = *(void **)((char *)si + {{ storageimpl_data }});
int64_t nbytes  = *(int64_t *)((char *)si + {{ storageimpl_nbytes }});

p = (char *)data + off * itemsize[dt];
```

Measured on torch 2.14: `thpvariable_cdata` 16, `tensorimpl_storage` 16,
`tensorimpl_storage_offset` 144, `tensorimpl_data_type` 160, `storageimpl_data`
16, `storageimpl_nbytes` 48. Checked against a view (`base[3:]`, f16) so the
`storage_offset_` multiply is exercised, not just the contiguous case.

`itemsize[]` is rendered from the live torch rather than hardcoded: iterate the
`torch.dtype` singletons and record `torch.empty(0, dtype=d).element_size()`.
The `data_type_` field is a `caffe2::TypeMeta` index, which coincides with
`ScalarType` because the type registry is seeded in `ScalarType` order. That
coincidence is an assumption this mode depends on and a test must pin.

**Risk, recorded deliberately.** This is the least stable surface in the design:
six offsets across two structs, where the earlier version table needed one.
`tensorimpl_storage` at +16 is safe -- it is the first member of a polymorphic
`intrusive_ptr_target` subclass. `storage_offset_` and `data_type_` are not:
their position depends on `autograd_meta_`, `extra_meta_`, `version_counter_`,
`pyobj_slot_` and `sizes_and_strides_`, and that region has moved before
(`extra_meta_` added in 1.13, `pyobj_slot_` restructured around 2.0). torch
maintains `C10_TensorImpl_Size_Check_Dummy_Class` precisely because the layout
drifts.

Two things contain the risk. The constants are render-time and per-version, so a
new torch needs a new row rather than a code change. And the layout table is
closed-ended: an unrecognised version is simply absent from it, so `"direct"` is
unavailable rather than wrong.

### `"cpython"`: no layout knowledge except dtype

- `data_ptr`: `PyObject_CallMethodNoArgs(o, "data_ptr")`, unbox. Roughly
  100-200 ns, the one unavoidably expensive call.
- `storage_size`: `o.untyped_storage().nbytes()`. Only read when the backend
  specializes on pointer range, i.e. AMD.
- `dtype`: read off the `torch.dtype` singleton.

```c
/* torch/csrc/Dtype.h:
 *   struct THPDtype { PyObject_HEAD at::ScalarType scalar_type; char name[65]; } */
PyObject *d = PyObject_GetAttr(o, st->dtype_str);   /* interned */
int32_t code = *(const int8_t *)((char *)d + sizeof(PyObject));
```

`at::ScalarType` is `enum class ScalarType : int8_t`. The struct is
byte-identical at v1.13.0, v2.0.0, v2.2.0, v2.5.0, v2.9.0, v2.14.0 and `main` --
the only edits in that range are a `PyObject_HEAD` semicolon and `TORCH_API` ->
`TORCH_PYTHON_API`. Verified against `aoti_torch_get_dtype` across all 57
`torch.dtype` singletons on torch 2.14: zero mismatches.

Unlike every other layout bet here, this one verifies itself. `char name[65]`
sits immediately after the enum, so at load `intj_exec` reads the name embedded
in `torch.float32` and compares it to `"float32"`. A dtype object is at least 82
bytes, so the read cannot fault. A mismatch raises rather than guessing, since
there is no further fallback beneath this mode.

### `"cxx"`: ask the compiler

```cpp
const at::Tensor &t = THPVariable_Unpack(obj);
void   *p  = t.data_ptr();
int32_t dt = static_cast<int32_t>(t.scalar_type());
int64_t n  = t.storage().nbytes();
```

A layout change becomes a compile error rather than a segfault, which is what
the other two modes' tables approximate. The reads inline, so like `"direct"`
there is no dlopen, no function pointer and no indirect call -- the two should
benchmark the same, and `"cxx"` gets there without asserting any offset.

That is why `None` prefers it. Costs, all accepted:

- **The build stops being self-contained.** It needs `torch/include`,
  `torch/include/torch/csrc/api/include`, `-L <torch>/lib`, `-ltorch_cpu -lc10
  -ltorch_python`, and `-D_GLIBCXX_USE_CXX11_ABI=` matching how torch was built.
  Read that flag from `torch._C._GLIBCXX_USE_CXX11_ABI` rather than guessing.
- **Build latency.** `python_variable.h` pulls in `ATen/Tensor.h` and thousands
  of headers. First launch per kernel goes from well under a second to several.
- **A C++ compiler must exist.** `_compiler_identity()` covers only
  `_find_compiler("c")` today and must cover `"c++"` for this mode.

triton's builder already supports it: `compile_so_from_src(..., language="c++")`
and `_find_compiler` honours `CXX`.

**Open feasibility risk.** The existing source must survive a C++ compile.
`_Static_assert` in `intj_runtime.h` is C-only and needs a guard; every `void *`
conversion must already be explicit; no `goto error` may cross a
non-trivially-destructible local. Inspection suggests a handful of edits -- the
casts look handled and every local on a `goto` path is a POD -- but that is an
expectation, not a verified fact, and confirming it is the first step of the
implementation plan. If one guarded template will not serve both languages, that
is a scope increase to surface before proceeding.

### What leaves the design

`libtorch_path` leaves `RenderContext`. No mode dlopens `libtorch_cpu.so`, and
no mode resolves an `aoti_torch_*` symbol. `intj_exec` still imports torch, for
the `Tensor` and `Parameter` type objects the argument type test needs, and (in
`"cpython"` mode) to run the `THPDtype` self-check.

The dtype is an `int32_t` in every mode, so `INTJ_WORD` keeps its 32-bit dtype
field, `nwords` is unchanged, and no `PyObject *` enters the spec key. intj uses
the dtype as an opaque in-process discriminator and never persists it, so the
fact that `"direct"` and `"cxx"` yield `ScalarType` while `"cpython"` yields the
same number by another route does not need reconciling.

## Error handling

- `torch_access="direct"` with a version absent from the layout table:
  `UnsupportedKernel` at `create_launcher` time, naming the version.
- `torch_access="cxx"` with `torch_version` disagreeing with the installed
  torch: `UnsupportedKernel`, since the headers cannot be selected.
- `"cxx"` with no C++ compiler, or a failed build: the build error propagates
  unwrapped. A link error naming a torch symbol beats anything intj would
  paraphrase.
- `THPDtype` self-check fails in `"cpython"` mode: `UnsupportedKernel` naming
  `torch.__version__`. No fallback exists beneath this mode.
- Everything inside `INTJ_DECODE` keeps its current behaviour: set a Python
  error, `goto error`.

## Testing

Extends `tests/test_launcher.py`, which is already differential against triton.

1. **Every mode agrees with triton.** Run the existing
   `test_spec_key_is_never_coarser_than_triton` corpus under all three modes,
   parametrized. Each is checked against triton's specialization separately, not
   against the others.
2. **Modes agree with each other on reads.** For a corpus covering contiguous
   tensors, views with a non-zero `storage_offset_`, every dtype, and the >2 GiB
   storage case, assert all three modes return the same data pointer, dtype and
   storage size. This is the test that catches a `TensorImpl` layout drift, and
   it is the reason `"cpython"` is worth keeping even though it is slowest: it
   is the independent oracle.
3. **Dtype numbering.** Assert the `THPDtype` offset read, the `TypeMeta` index
   at `tensorimpl_data_type`, and `torch.Tensor.dtype` agree for all 57
   singletons. Pins the `TypeMeta`/`ScalarType` coincidence `"direct"` relies on.
4. **Layout and mode tables.** Pure-Python unit tests over the version ->
   constants mapping and the `None` -> mode resolution, including the
   unrecognised-version path and every raising case. No GPU needed.
5. **Module key.** Assert `ModuleKey.digest()` changes when `torch_version` or
   `torch_access` changes. This is the regression the design exists to prevent.
6. **Launch correctness per mode.** Launch the existing kernels under each mode
   and compare results against triton.

`benchmarks/bench_launch.py` gains a row per mode. Two claims this design makes
without evidence, for the benchmark to settle: that `"direct"` and `"cxx"` land
in the same place, and what `"cpython"` actually costs per tensor argument.

## Migration

`torch_access` defaults to `None`. On a supported torch that resolves to
`"cxx"`, which is a behaviour change from today's shim path -- same results,
different build requirements and a slower first launch. Callers who want the old
build characteristics pass `torch_access="direct"`.

The `.so` cache invalidates once, because the template and runtime header are
hashed into `ModuleKey`, and thereafter re-invalidates on each torch minor
version.

Bump `SCHEMA_VERSION`.

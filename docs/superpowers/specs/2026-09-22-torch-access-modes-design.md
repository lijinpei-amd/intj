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

- Three explicit strategies, selectable per launcher via an enum, plus `AUTO`.
- Nothing version-specific is baked into a binary unless it must be, so the
  cache cannot serve a module built against a different torch -- and does not
  rebuild when nothing relevant changed.
- Avoid calling back into the Python interpreter on the launch path wherever the
  version makes that possible.
- No measurable cost on the fast path, which is the whole point of intj: the
  5-argument benchmark decodes in 0.15 us.
- Keep the version policy in Python, where it is testable.

## Non-goals

- Tensor subclasses beyond `nn.Parameter`. Orthogonal, and a widened type test
  would accept `FakeTensor` and `DTensor`, which have no dense data pointer.
- Cross-compiling against a torch that is not installed. `torch_version` selects
  which layout constants are installed at load, not which headers exist on disk.

## Design

### What is baked in, and what is set at load

The mode is an enum, not a string, so a typo is an `AttributeError` at the call
site rather than an `UnsupportedKernel` at render time:

```python
class TorchAccess(enum.Enum):
    AUTO = "auto"          # resolve by torch version and toolchain
    CPYTHON = "cpython"    # interpreter calls only
    SHIM = "shim"          # intj's own shim: struct reads, no torch code
    CXX = "cxx"            # compiled against torch's headers
```

`SHIM` is named for what intj ships, not what torch ships: the mode replaces the
AOTI shims with intj's own equivalents rather than calling them. No
`aoti_torch_*` symbol is resolved in any mode.

`AUTO` is resolved first, to one of the three concrete members, and only then
does `RenderContext` get filled in. It gets the mode, and for `CXX` the version:

```python
torch_access: TorchAccess               # never AUTO by this point
torch_version: tuple[int, int] | None   # `CXX` only
```

The struct offsets are deliberately *not* here. They live in module state and
are installed by `set_torch_version`, called once after load:

```c
typedef struct {
  ...
  intj_layout layout;   /* six offsets; `SHIM` mode only */
} intj_state;
```

**What the key must contain is every version-dependent input.** `ModuleKey`
already states this contract: "a missing field means a stale module." Applied
per mode:

| mode | version-dependent input | how it is handled |
|---|---|---|
| `SHIM` | seven struct offsets, itemsize table | probed at load, not built in |
| `CPYTHON` | none | - |
| `CXX` | the headers and libraries it links | `torch_version` + ABI flag in the key |

So only `CXX` keys on the version, because only `CXX` bakes something
version-specific into the binary. The other two produce a `.so` that is valid on
every torch version, forever, and never rebuilds.

That is a stronger result than keying `SHIM` on its layout would have been.
Keying fixes the stale-cache bug by making the digest move; setting the layout
at load removes the possibility, because the offsets come from the live process
rather than from whenever the binary happened to be compiled. There is no
version-skew window at all.

The cost is on the fast path: six immediates become six loads from `st->layout`.
They sit in a cache line the decode already touches, and they are invariant
across the arguments of one `entry` call, so the compiler should hoist them --
but "should" is doing work in that sentence, and the benchmark has to settle it.
If it turns out to cost, the fallback is to key `SHIM` on its layout after
all and go back to immediates.

### The setter

`_loaded_module` calls `module.set_torch_version(version, layout)` immediately
after `set_compile_callback`, under the same lock, so neither `entry` nor
`spec_key` is reachable before it completes. The module refuses to launch if it
has not run, the way it already refuses a missing compile callback.

Python resolves version -> layout and passes the constants; C only stores them.
That keeps the table where a unit test can reach it without a GPU, which was the
reason for choosing a setter over parsing `torch.__version__` in C. `version` is
passed too, for error messages.

In `CPYTHON` and `CXX` modes the call still happens but installs no layout;
it is the point where the `THPDtype` self-check runs.

### The modes

`create_launcher(..., torch_access=TorchAccess.AUTO, torch_version=None)`:

| Value | How it reads a tensor | Needs |
|---|---|---|
| `SHIM` | Struct offsets from module state. No dlopen, no torch code, no interpreter. | version in the layout table |
| `CPYTHON` | `data_ptr()` / `untyped_storage().nbytes()` calls, dtype by `THPDtype` offset. | nothing |
| `CXX` | `THPVariable_Unpack`, inlined `at::Tensor` methods. | torch headers, a C++ compiler |
| `AUTO` | `CXX` if supported, else `SHIM` if the version is in the table, else `CPYTHON`. | - |

The layout is **discovered**, not looked up. `torch_abi.probe_layout()` pins each
offset by intersecting, over a set of probe tensors, the positions whose value
matches what torch's own accessors report (`t._cdata` is the `TensorImpl*`,
`storage._cdata` the `StorageImpl*`). Anything that does not come down to exactly
one candidate yields None and `SHIM` is refused. Measured at **4.6 ms**, once per
process.

That replaces the per-version table this spec originally proposed, and is
strictly better: the offsets are true by construction for whatever torch is
loaded, including one intj has never seen, so there is no closed-ended table to
maintain and no "unrecognised version" cliff.

One asymmetry worth stating plainly: in `CXX` mode it cannot select, because
the headers come from the loaded torch. There it is validated instead, and
`create_launcher` raises if it disagrees with `torch.__version__`.

### `SHIM`: reading the structs

The mode calls nothing -- not torch, not the interpreter. Every read is a load
at an offset held in module state.

```c
const intj_layout *L = &st->layout;
TensorImpl  *ti = *(void **)((char *)obj + L->thpvariable_cdata);
StorageImpl *si = *(void **)((char *)ti  + L->tensorimpl_storage);
int64_t off     = *(int64_t *)((char *)ti + L->tensorimpl_storage_offset);
uint8_t dt      = *(uint8_t *)((char *)ti + L->tensorimpl_data_type);
void   *data    = *(void **)((char *)si + L->storageimpl_data);
int64_t nbytes  = *(int64_t *)((char *)si + L->storageimpl_nbytes);

p = (char *)data + off * L->itemsize[dt];
```

Probed on torch 2.14, x86-64: `cdata` 16, `storage` 16, `storage_offset` 144,
`numel` 152, `data_type` 160, `s_data` 16, `s_nbytes` 48.

`numel` is the seventh offset, and it exists only to canonicalize. torch's
`Tensor::data_ptr()` returns null for **every** zero-element tensor, including an
empty slice at the end of a live buffer whose `storage_offset` is not zero --
verified: `torch.zeros(4096, dtype=f16)[4096:]` has storage at `0x17...` and
`storage_offset` 4096, yet `data_ptr()` is 0. The raw arithmetic cannot see that
and would return a past-the-end pointer, which flips the `INTJ_FLAG_D` alignment
bit and makes `SHIM` key differently from the other two modes for identical
arguments. So `SHIM` reads `numel_` and returns NULL when it is zero.

`itemsize[]` is built from the live torch rather than hardcoded: iterate the
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

Two things contain the risk. The constants are data, resolved per version at
load, so a new torch needs a new row rather than a code change. And the layout table is
closed-ended: an unrecognised version is simply absent from it, so `SHIM` is
unavailable rather than wrong.

### `CPYTHON`: no layout knowledge except dtype

- `data_ptr`: `PyObject_CallMethodNoArgs(o, "data_ptr")`, unbox. Measured
  **71.5 ns**, the one unavoidably expensive call.
- `storage_size`: `o.untyped_storage().nbytes()`, **108 ns**. Only read when the
  backend specializes on pointer range, i.e. AMD.
- `dtype`: read off the `torch.dtype` singleton, **52.6 ns**.

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

### `CXX`: ask the compiler

```cpp
const at::Tensor &t = THPVariable_Unpack(obj);
void   *p  = t.data_ptr();
int32_t dt = static_cast<int32_t>(t.scalar_type());
int64_t n  = t.storage().nbytes();
```

A layout change becomes a compile error rather than a segfault, which is what
the other two modes' tables approximate. The reads inline, so like `SHIM`
there is no dlopen, no function pointer and no indirect call -- the two should
benchmark the same, and `CXX` gets there without asserting any offset.

That is why `AUTO` prefers it. Costs, all accepted:

- **The build stops being self-contained.** Every path and flag is read off the
  loaded `torch` module -- never hardcoded, never derived from
  `os.path.dirname(torch.__file__)` by hand:

  | need | source |
  |---|---|
  | include dirs | `torch.utils.cpp_extension.include_paths()` |
  | library dirs | `torch.utils.cpp_extension.library_paths()` |
  | ABI flag | `torch._C._GLIBCXX_USE_CXX11_ABI` |

  The only library needed is **`-lc10`**, and it *is* needed: `THPVariable_Unpack`
  is header-inline but pulls in six undefined c10 error-path symbols, and torch
  loads libc10 `RTLD_LOCAL`, so an unlinked module links fine and then dies at
  import with `undefined symbol: ...throw_data_ptr_access_error`. `-ltorch_cpu`
  and `-ltorch_python` are not needed. On torch 2.14 the helpers
  return `<torch>/include`, `<torch>/include/torch/csrc/api/include` and
  `<torch>/lib`, which is what the feasibility compile used; taking them from the
  helpers instead means a torch that reorganises its tree, or an out-of-tree
  build, keeps working without a change here. `include_paths()` takes no
  required argument on any version.

  Do not add these paths to `ModuleKey`. Two installs of the same torch version
  in different virtualenvs are ABI-identical, and keying on the path would
  rebuild for no reason. `_GLIBCXX_USE_CXX11_ABI` *is* keyed, because a torch
  rebuilt with the other ABI is a genuine incompatibility that
  `torch_version` alone does not capture.
- **A language standard that depends on the torch version.** intj passes
  `-std=c++20`. On g++ 13.3, C++17 also compiles torch 2.14 but only with a
  `-Wc++20-extensions` warning, so "requires C++20" is really "requires C++20 to
  be warning-clean"; clang may be stricter. triton inserts its own `-std=c++17`
  early and appends `ccflags` last, so intj's flag wins -- do not try to suppress
  triton's.
- **Build latency, measured: 10.9 s** for one translation unit
  (`g++ -std=c++20 -O2`, torch 2.14, warm page cache), against 0.011 s for an
  empty C one. This is per kernel, on first launch, in the caller's foreground.
  It is the strongest argument against `AUTO` resolving here, and it should be
  revisited once it is a benchmark row rather than a single measurement.
- **A C++ compiler must exist.** `_compiler_identity()` covers only
  `_find_compiler("c")` today and must cover `"c++"` for this mode.

triton's builder already supports it: `compile_so_from_src(..., language="c++")`
and `_find_compiler` honours `CXX`.

#### Exceptions are mandatory here

`-fno-exceptions` is not an option: torch's headers do not compile without
exception support. `c10/util/Exception.h` throws from `TORCH_CHECK`, and
`ATen/core/ivalue_inl.h` contains `try`/`catch`, both reached through the inline
functions this mode calls -- `data_ptr()` itself throws on uninitialized
storage. Verified: `-fno-exceptions` fails at `Exception.h:400`.

Nor would disabling them pay. Itanium-ABI exception handling is table-driven, so
the non-throwing path carries no runtime cost; `noexcept` and `-fno-exceptions`
buy code size and a little optimizer freedom, not speed, and for a handful of
inline loads that is not measurable. The C modes have nothing to disable.

What this mode does need, for correctness rather than performance: an exception
escaping `entry` into CPython's C frames is undefined behaviour, so the tensor
reads sit inside a `try`/`catch` that converts `c10::Error` into a Python
exception and takes the existing `goto error` path. Free when nothing throws.

**Open feasibility risk.** Torch's side is now verified: a translation unit
including `python_variable.h` and calling `THPVariable_Unpack`, `data_ptr()`,
`scalar_type()` and `storage().nbytes()` compiles clean under `-std=c++20`. What
is *not* yet verified is intj's own source surviving a C++ compile.
`_Static_assert` in `intj_runtime.h` is C-only and needs a guard; every `void *`
conversion must already be explicit; no `goto error` may cross a
non-trivially-destructible local. Inspection suggests a handful of edits, but
confirming it is the first step of the implementation plan. If one guarded
template will not serve both languages, that is a scope increase to surface
before proceeding.

### What leaves the design

`libtorch_path` leaves `RenderContext`. No mode dlopens `libtorch_cpu.so`, and
no mode resolves an `aoti_torch_*` symbol. `intj_exec` still imports torch, for
the `Tensor` and `Parameter` type objects the argument type test needs, and (in
`CPYTHON` mode) to run the `THPDtype` self-check.

The dtype is an `int32_t` in every mode, so `INTJ_WORD` keeps its 32-bit dtype
field, `nwords` is unchanged, and no `PyObject *` enters the spec key. intj uses
the dtype as an opaque in-process discriminator and never persists it, so the
fact that `SHIM` and `CXX` yield `ScalarType` while `CPYTHON` yields the
same number by another route does not need reconciling.

## Error handling

- `torch_access=TorchAccess.SHIM` with a version absent from the layout table:
  `UnsupportedKernel` at `create_launcher` time, naming the version.
- `torch_access=TorchAccess.CXX` with `torch_version` disagreeing with the loaded
  torch: `UnsupportedKernel`, since the headers cannot be selected.
- `CXX` with no C++ compiler, or a failed build: the build error propagates
  unwrapped. A link error naming a torch symbol beats anything intj would
  paraphrase.
- `THPDtype` self-check fails in `CPYTHON` mode: `UnsupportedKernel` naming
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
   it is the reason `CPYTHON` is worth keeping even though it is slowest: it
   is the independent oracle.
3. **Dtype numbering.** Assert the `THPDtype` offset read, the `TypeMeta` index
   at `tensorimpl_data_type`, and `torch.Tensor.dtype` agree for all 57
   singletons. Pins the `TypeMeta`/`ScalarType` coincidence `SHIM` relies on.
4. **Layout and mode tables.** Pure-Python unit tests over the version ->
   constants mapping and the `AUTO` -> mode resolution, including the
   unrecognised-version path and every raising case. No GPU needed.
5. **Module key granularity.** Assert the digest changes when `torch_access`
   changes, and when `torch_version` or the CXX11 ABI flag changes in `CXX`. Assert it does *not*
   change when the torch version moves in `SHIM` or `CPYTHON` -- those binaries
   are version-independent, and a digest that moved would mean something
   version-specific had leaked into the render.
6. **Launch correctness per mode.** Launch the existing kernels under each mode
   and compare results against triton.

`benchmarks/bench_launch.py` gains a row per mode. Two claims this design makes
without evidence, for the benchmark to settle: that `SHIM` and `CXX` land
in the same place, and what `CPYTHON` actually costs per tensor argument.

## Migration

`torch_access` defaults to `TorchAccess.AUTO`. On a supported torch that resolves to
`CXX`, which is a behaviour change from today's shim path -- same results,
different build requirements and a slower first launch. Callers who want the old
build characteristics pass `torch_access=TorchAccess.SHIM`.

The `.so` cache invalidates once, because the template and runtime header are
hashed into `ModuleKey`. After that, rebuild frequency depends on the mode:
`CXX` on every torch minor version, `SHIM` only when the struct layout
changes, `CPYTHON` never.

Bump `SCHEMA_VERSION`.

## Outcome

Implemented and measured on torch 2.14.0.dev+rocm7.2, gfx942, triton 3.8.0.

| mode | decode + spec key (3 tensor args) | first build | `.so` rebuilt on torch upgrade |
|---|---|---|---|
| `SHIM` | 89 ns | 0.6 s | no |
| `CXX` | 101 ns | 11.5 s | yes |
| `CPYTHON` | ~300 ns | 0.6 s | no |

`SHIM` and `CXX` land in the same place, as predicted -- both inline the reads.
`SHIM` is marginally faster than the AOTI-shim path it replaces (0.11 us against
0.15 us) because the three indirect calls are gone.

All three modes produce byte-identical spec keys over a corpus that includes
unaligned views, every dtype, `nn.Parameter`, and both kinds of empty tensor.
Test suite: 64 passed.

Two things this spec claimed that turned out differently, both recorded above:
the link set is `-lc10` alone rather than three libraries, and the layout is
probed rather than looked up by version.

Two bugs the work turned up, both fixed:

* `SHIM` accepted a storage-less tensor (sparse) and would have handed the kernel
  a null pointer where the other two modes raise.  `!storage` and `numel == 0`
  are different conditions and torch treats them differently; so does intj now.
* triton's compile cache is keyed on the source bytes and the include *directory
  names*, not on the headers' contents.  So an edit to `intj_runtime.h` moved
  intj's own digest, intj re-rendered, and triton then served the object it had
  compiled from the previous header.  Pure-header edits silently did nothing.
  Fixed by feeding the header's digest in as `-DINTJ_RUNTIME_HEADER=...`, which
  puts it in triton's key too.  This predates this work.

Still open, now with numbers behind it: whether `AUTO` should resolve to `CXX`.
It costs 11.5 s on first build of each distinct kernel source for 0.02 us more
decode time than `SHIM`, and it is the only mode that must be rebuilt when torch
changes. The build is cached by triton across processes, so the cost is paid once
per kernel rather than once per run, which is milder than feared -- but `SHIM`
gets the same speed for 0.6 s and never rebuilds.

## Postscript: why `CXX` does not reach `SHIM`

`CXX` was rewritten to use torch's check-free accessors (`unsafe_storage`,
`_mutable_data_ptr_no_checks`, `numel_default`, `storage_offset_default`,
`nbytes`), which is the same set of reads the shim makes. It still lands ~4 ns per
tensor behind. Four things were measured on the way, three of which were wrong
guesses:

1. **The five `TORCH_CHECK`s are not the cost.** Removing them changed nothing:
   they are predicted-not-taken branches on flags already in cache.
2. **The policy dispatch in `numel()`/`storage_offset()` is not the cost either.**
   `*_default()` changed nothing measurable.
3. **Inlining was.** g++ left `intj_read_tensor` out of line -- a real call per
   argument -- because a `try` block inside it inflated the size estimate. The
   exception machinery is free at runtime; its effect on the inliner is not. The
   fix is one `try` around the whole decode in the template, not one per read.
4. **`sym_nbytes()` was.** It returns a `SymInt` by value whose destructor does not
   inline: ten PLT calls per `spec_key`. `nbytes()` is a bitfield test instead.

After those, `CXX` is within 48 instructions of `SHIM` with no out-of-line calls
left on the hot path. The remaining gap is the handful of bitfield tests that are
the entire difference in what the two modes verify, so closing it means deleting
the checks that distinguish them.

`intj_as_i64` carries `__attribute__((always_inline))` for the same reason as (3);
it improved every mode, including the C ones (94 -> 89 ns for `SHIM`).

## Postscript: a limitation both modes share

A tensor whose `TensorImpl` overrides `numel()` rather than storing it reads as
empty. `torch.nested.nested_tensor(...)` stores `numel_ == 0` and reports 8, and it
is *exact-type* `torch.Tensor`, so the type gate does not catch it -- my earlier
claim that the gate filters everything with a custom impl was wrong. intj passes a
null pointer and the kernel faults on the GPU.

`SHIM` cannot see the policy bit at all, so `CXX` reads the raw field too rather
than diverge. Documented in `docs/Usage.md` and pinned by a test, not fixed:
neither a nested nor an mkldnn tensor is a usable triton kernel argument. `mkldnn`
has no storage and *is* refused cleanly.

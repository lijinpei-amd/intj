# INTJ usage

## `make_launcher`

```python
from intj import KernelCache, TorchAccessMode, make_launcher

launcher = make_launcher(
    jit_func,              # a @triton.jit function
    dynamic_grid=False,    # reserved, must be False
    dynamic_options=(),    # compile options / knobs.<group>.<name> passed per call
    extra_annotation=None, # parameter name -> annotation or shorthand
    options=None,          # triton compile options, e.g. {"num_warps": 8}
    torch_access_mode=None, # automatic; pass a TorchAccessMode to select one
    kernel_cache=KernelCache.INTJ,
    no_gpu=False,          # host decode/cache path, without GPU work
    verify_annotation=False,
    bind_device=False,
    grid_arg=None,         # keyword-only: 1, 2, or 3 separate grid dimensions
    grid_cpp=None,         # keyword-only: annotated def compiled into the extension
    grid_py=None,          # keyword-only: Python callback evaluated on each launch
    return_compiled=False, # keyword-only: return the cached Triton CompiledKernel
)
```

`make_launcher` returns a callable launcher when no value or device binding
requires a factory; with bindings, `.bind()`/`.bind_device()` return it. Call it
once, outside any hot loop.

### Lazy build

`make_launcher` returns a launcher at once, without touching the GPU. It
validates arguments, resolves annotations, analyzes an autotune/heuristics
chain, lowers `grid_cpp`, and provisions a non-default `kernel_cache`. The
**first call** builds: it queries the GPU target, renders the C extension,
compiles it with the host compiler (cached on disk under intj's module digest),
loads it, fills the launcher, and swaps its entry. After that, every reference
to the launcher, including ones taken before the build, calls the native entry
directly. If the build raises, that call raises and the next call retries.
Concurrent first calls build once.

Decorating a module-level kernel is therefore safe at import on a machine
without a GPU. A few checks need the target and surface on the first call: an
unknown `options=` name, an invalid compile option in an autotune config, and
the C-side checks on bound values (an out-of-range pointer, a failing
`hipDeviceGet`). Triton's `debug`, `instrumentation_mode` and
`fpsan_homomorphic_casts` knobs are read then too, and fixed for the launcher.
Every launcher is a builtin function whose `__self__` is an
`_intj_lazy.Launcher`, whatever it binds; `intj.launcher.module_of(launcher)`
builds if needed and returns the rendered module, for debugging.

Called without `jit_func` (keyword arguments only), `make_launcher` is a
decorator factory instead, so a kernel -- plain or stacked under
`triton.autotune`/`triton.heuristics` -- can be decorated directly:

```python
@make_launcher(grid_cpp=grid)
@triton.autotune(configs=configs, key=["n"])
@triton.heuristics({"BLOCK": lambda a: 128})
@triton.jit
def k(x, o, n, BLOCK: tl.constexpr):
    ...
```

This is equivalent to calling `make_launcher(k, grid_cpp=grid)` on the already
autotuned/heuristic-wrapped `k`. A plain `@make_launcher` (no parens) keeps
working exactly as before.
`extra_annotation` can fix types or specialization facts, bake values, or mark
values for binding. `verify_annotation=True` checks declared promises before
cache lookup; the default trusts them. `bind_device=True` fixes a device on a
bound handle. `no_gpu=True` renders and times host decoding and cache lookup
without querying a GPU target or driver, compiling a kernel, or launching one.
It accepts only default compile options. The `torch_access_mode` and `kernel_cache`
choices are described below.

`options` must be valid triton compile options (`num_warps`, `num_stages`,
`waves_per_eu`, ...) and are fixed for the lifetime of the launcher. `device`,
`stream`, `device_type` and `warp_size` are rejected, and an unknown option name
is an error (on the first call, where the target is known) rather than a silent
fall back to the default.

They are canonicalized through the compiler backend's `parse_options`, so spellings
that mean the same thing — `{}` and `{"num_warps": 4}` on AMD — share one rendered
module instead of building it twice. The canonical options are passed directly to
`triton.compiler.compile` with an annotated `triton.compiler.ASTSource` (or
`GluonASTSource` for a Gluon JIT function) on a cache miss. An explicit `debug`
option overrides the JIT function's debug default;
Triton's runtime debug flag can still enable it. Instrumentation mode comes from
Triton's compilation knob. These effective options also determine module identity.

Most things intj cannot handle raise `intj.launcher.UnsupportedKernel` here. The
rest — `num_ctas > 1`, a cooperative launch, a kernel needing scratch memory — can
only be seen once triton has compiled, so they raise on the first launch that misses
the cache. An unsupported *argument* raises `TypeError` at the launch that passes it.

## Calling the launcher

Choose at most one grid option in `make_launcher`:

| grid option | launcher call after `device, stream` |
|---|---|
| omitted or `grid_arg=None` | `grid, *kernel_args` — `grid` is an `int` or a 1–3 element tuple/list of ints |
| `grid_arg=1`, `2`, or `3` | `gx[, gy[, gz]], *kernel_args` — one positional int per dimension |
| `grid_cpp=grid_fn` | `*extra_args, *kernel_args` — native code computes the grid |
| `grid_py=grid_fn` | `*kernel_args` — Python computes the grid on each launch |

| argument | type |
|---|---|
| `device` | device index, `int` in `[0, 256)` (e.g. `torch.cuda.current_device()`) |
| `stream` | raw stream handle, `int` (e.g. `torch.cuda.current_stream().cuda_stream`) |
| `kernel_args` | public kernel parameters in declaration order; baked and bound values are omitted |

All launch arguments are positional. With `bind_device=True`, omit `device` from
each call. Grid dimensions must be exact ints in `[0, 2**32)`; omitted dimensions
are 1. By default the launcher returns `None`, and a zero-volume grid skips the
launch. With `return_compiled=True`, it returns the cached Triton
`CompiledKernel`; a zero-volume grid still compiles or finds that kernel, but
does not dispatch it. Repeated calls for the same specialization return the
same object. This mode requires GPU compilation and rejects `no_gpu=True`.

### Per-call compile options and knobs: `dynamic_options`

```python
@intj.make_launcher(grid_cpp=grid, dynamic_options=("num_warps", "knobs.runtime.debug"))
@triton.jit
def k(x, out, n, ACT: tl.constexpr, BLOCK: tl.constexpr): ...

k(device, stream, 8, False, x, out, n, "gelu", 128)   # num_warps=8, debug off
```

Each name becomes a positional argument right after the grid controls, in
the declared order, and its value is part of the key: an `int`, `bool` or
`float` by value and kind, a `str` (e.g. `knobs.compilation.instrumentation_mode`)
through the same id table as object constexprs. A name is either a compile
option (`num_warps`, `num_stages`, `waves_per_eu`, ...) or a Triton knob path
`knobs.<group>.<name>` (`knobs.runtime.debug`, `knobs.amd.use_buffer_ops`, ...).
On a miss, options merge into `options=` and go through `parse_options` per
kernel (an invalid value such as `num_warps=3` raises on that call); knobs are
set to the call's values with Triton's `knobs` scope for the duration of the
compile, then restored. The knobs are process globals, so a compile running on
another thread meanwhile sees the declared values too. With `triton.autotune`,
the values reach every config.

Refused at `make_launcher` with `UnsupportedKernel`: an unknown knob path, a
repeated name, a name also given in `options=` (for knobs, the option the knob
feeds, e.g. `debug`), a kernel parameter name, `device`/`stream`/`device_type`/
`warp_size`, a name an autotune config sets, and `no_gpu=True`. An unknown
compile-option name raises on the first call, where the target is known.

Knobs you do not declare are read once, at the first call, and fixed for that
launcher (`debug`, `instrumentation_mode`, fpsan casts feed its options). Other
cache-invalidating knobs are not keyed; see `TODO.md`.

### Compiled grid: `grid_cpp`

Define an undecorated, source-backed Python `def` whose parameters all have
`int` or `bool` annotations. Python lambda parameters cannot be annotated. Its
positional parameter names reuse the corresponding JIT kernel values;
keyword-only names must be distinct from JIT parameters and become extra
launcher arguments, passed before the public kernel arguments in their declared
order:

```python
import triton

# kernel parameters: (out, n, BLOCK)
def grid(n: int, BLOCK: int, *, cap: int):
    return (min(cap, triton.cdiv(n, BLOCK)),)

launcher = make_launcher(kernel, grid_cpp=grid)
launcher(device, stream, cap, out, n, BLOCK)
```

The function is translated at `make_launcher` time and is never called by the
launcher. It can use integer/bool literals and names, simple local assignments,
unary `+`/`-`, `+`, `-`, `*`, `//`, `triton.cdiv`, two-argument builtin `min`, and
conditional expressions. It must end with a literal tuple/list of 1–3 integer
expressions. Reused parameters may be public scalar values or baked `int`/`bool`
values; bound pointer/tensor parameters cannot be read. Defaults, `*args`,
`**kwargs`, free variables, other globals, and other Python syntax are refused
with `UnsupportedKernel`. Intermediate arithmetic is checked signed 64-bit;
`//` follows Python floor division. Wrong input types, division by zero,
overflow, and out-of-range final dimensions raise before the GPU launch.

### Python grid: `grid_py`

`grid_py` accepts a callable of one `meta` dict argument. The dict is fresh on
every launch and contains every original JIT parameter by name, including baked
and bound values. Return an int or a 1–3 element tuple/list of grid dimensions:

```python
launcher = make_launcher(
    kernel,
    grid_py=lambda meta: (triton.cdiv(meta["n"], meta["BLOCK"]),),
)
launcher(device, stream, out, n, BLOCK)
```

The callback runs on every launch, including kernel-cache hits; its exceptions
propagate. Each launcher handle keeps its own callback, even when handles share
one compiled extension. Grid controls do not enter the kernel specialization key.

`device` must be the device `stream` belongs to, and it must be the current device
the first time a given specialization is launched — triton compiles and loads the
binary on the current device, so intj refuses a mismatch instead of launching a
function in the wrong context. Kernels compiled for different devices live under
different keys, so one launcher serves all of them.

Keyword arguments are not accepted, defaults are not filled in, and the launcher does
not read the current device or stream for you — that is where the launch overhead of
`JITFunction` goes.

## Autotune and heuristics

`make_launcher` accepts a `@triton.jit` function wrapped in any nesting of
`@triton.heuristics` layers and at most one `@triton.autotune` layer:

```python
launch = make_launcher(
    triton.autotune(configs=[...], key=["N"])(
        triton.heuristics({"EVEN": lambda a: a["N"] % a["BLOCK"] == 0})(kernel)
    ),
    grid_cpp=grid,   # may read tuned values such as BLOCK
)
launch(device, stream, x, out, N)   # tuned and heuristic values are not passed
```

On a miss, Triton tunes and launches through intj: pruning, `reset_to_zero`,
`restore_value`, `do_bench` and disk caching all run as usual, and
`best_config`, `bench_time` and `configs_timings` are copied back to your
tuners. On a hit, intj launches without calling Triton or any heuristic
written in Python. `Config.pre_hook` therefore runs on misses only.

intj keys a hit on:

- the exact value of every caller argument an autotune layer keys on
  (`int`, `bool`, or `float` by its fp64 bits; a tensor there raises
  `TypeError` at the call);
- the result of every heuristic that reads a caller argument nothing else
  keys. Those heuristics are compiled to C, which accepts ints and bools, `+ - * // %`,
  comparisons, `and/or/not`, conditional expressions, `min`, `max`,
  `triton.cdiv`, `triton.next_power_of_2`, `x is None`, and
  `.numel() .size(i) .shape[i] .stride(i) .dim() .element_size()
  .is_contiguous() .data_ptr() .dtype` on tensor arguments. Scalar inputs must be `int` or
  `bool` at runtime (`TypeError` otherwise, e.g. for a `float`). Tensor
  attribute reads go through the interpreter, so they cost more than scalar
  reads.

Every other heuristic -- one that reads only autotune keys, tuned values or
baked values -- runs in Python on a miss only, and its result is stored per
key. Such a heuristic may call anything, but its result is frozen per key: do
not read mutable globals. A heuristic whose result depends on a value only
known after a lookup is keyed in a nested map, so one launch can take more
than one lookup. Heuristic lambdas cannot close over locals; use module-level
names or literals.

Grids:

- `grid_cpp` may read tuned values (such as `BLOCK`) but not a heuristic
  result that is compiled to C (refused at `make_launcher`; use `grid_py`,
  which sees every value in `meta`). Its grid is computed after the lookup,
  and a zero-volume result skips the launch.
- With the default grid or `grid_arg`, a zero-volume grid on a miss raises
  `ValueError("intj: cannot tune on a zero-volume grid")`, even with
  `return_compiled=True`.

Refused with `UnsupportedKernel`: `no_gpu=True`, bound values,
`extra_annotation` or `options` naming a tuned value, configs with
`num_ctas != 1`, layers assigning a runtime (non-constexpr) parameter, a
name assigned twice (including two nested `@triton.autotune` layers, which
Triton itself cannot run), a heuristic or autotune key reading a value an
inner layer assigns, heuristics outside the subset above, and a tuned value
read by C that is not an `int` or `bool` (autotune config values at
`make_launcher`, heuristic results on the miss). Each bound launcher (every
`bind()`/`bind_device()`) owns its own tuner caches and tunes again. CUDA is
compile-checked only (untested); free-threaded Python builds are
compile-untested.

## Triton-style migration bridge

`intj.compat.launch(kernel, grid, /, *args, return_compiled=False, **kwargs)` accepts a direct
`@triton.jit` function, positional or named kernel arguments, defaults, and
compile options. Its grid can be an int, a tuple/list, or a callable receiving
the bound kernel values as a `meta` dict:

```python
from intj.compat import launch

launch(kernel, (triton.cdiv(n, BLOCK),), out, n=n, BLOCK=BLOCK,
       num_warps=4)
```

The bridge binds Python arguments and reads the current device and stream on
every call. Construct a `make_launcher` handle once for hot loops. Unsupported
kernels and options still raise instead of falling back to Triton;
`@triton.autotune` and `@triton.heuristics` wrappers are refused here (use
`make_launcher`, see [Autotune and heuristics](#autotune-and-heuristics)).
Pass `return_compiled=True` when the caller needs the cached Triton
`CompiledKernel`. `intj.compat.launch_or_interpret` accepts the same flag and
uses Triton's original launcher in interpreter mode.
When Triton's compile-warmup test mode is active, the bridge uses its indexed
launcher so the test runtime can compile fake-pointer inputs without GPU dispatch.
The bridge also raises `UnsupportedKernel` while Triton launch enter or exit
hooks are registered, including profiler hooks. It checks on every call, even
when reusing a cached launcher.
Direct `make_launcher` handles do not run Triton's global launch hooks; leave
hook-sensitive launches on Triton.

## Argument annotations

Import `Argument`, `Constexpr`, `AUTO`, `NEVER`, `Assume`, `EqualTo`, `Aligned`,
`PointerRange`, `BindValue`, `INT_TYPES`, and `FLOAT_TYPES` from `intj`.
`Annotation`, `Specialization`, and `Fact` are their public base classes.
`Argument(type=..., specialize=..., value=..., bind_value=...)` describes an
ordinary parameter. `Constexpr(type=..., power_of_two_or_zero=False,
value=...)` describes a constexpr parameter. Every field is optional.
`INT_TYPES` is `(tl.int32, tl.int64, tl.uint64)` and `FLOAT_TYPES` is
`(tl.float32,)`; either can be used as a type allowlist. A single dtype or a
nonempty tuple of dtypes is accepted. A missing type or `type=AUTO` retains
Triton's inference. An ordinary type allowlist checks the inferred type;
a typed constexpr selects the smallest fitting type, then the name. Neither
coerces the Python value.

Annotations may appear inline on the JIT function or in `extra_annotation`
under the parameter name. Inline `tl.constexpr` means `Constexpr()`; an inline
Triton dtype or `None` is shorthand for `Argument(type=...)`. The same
shorthands work in `extra_annotation`. The two sources merge field by field:
unspecified fields come from the other source, equal fields agree, and
conflicting explicit fields raise at `make_launcher`. Triton's
`do_not_specialize` and `do_not_specialize_on_alignment` are merged into the
corresponding `NEVER` facts; a conflicting explicit fact raises.

`AUTO` retains the applicable Triton specialization: integer equality to 1,
16-byte alignment for integers and pointers, and AMD's <2 GiB pointer-storage
range. `NEVER` omits those facts. `Assume(EqualTo(1), Aligned(16),
PointerRange(32))` fixes selected facts; use only facts applicable to the
parameter, and only the exact values shown. Omitted facts in `Assume` stay
`AUTO`. The aligned and equal-to-one facts cannot both hold for one integer.
An assumed fact is a caller promise unless `verify_annotation=True`.

Ordinary scalar types include `tl.int1` (canonical key name `u1`), signed and
unsigned 8/16/32/64-bit integers, and `tl.float32`/`tl.float64`; pointers use
`tl.pointer_type(element_dtype)` for supported tensor element dtypes.
Constexpr types include `tl.int1`, signed and unsigned 8/16/32/64-bit integers,
`tl.float64`, and exact `None`. A bare constexpr preserves Triton's original
Python value; a checked typed constexpr verifies that the value fits its chosen type.
`tl.float32` is not a supported constexpr type. With
`power_of_two_or_zero=True`, an integer constexpr key uses one byte: zero is
0, positive `2**k` is `k+1`, and negative `-2**k` is `0x80 | (k+1)`.
Verification rejects values outside that set; the unchecked mode trusts the
promise and may alias distinct invalid inputs to one binary.

`None` is exact and distinct from an omitted field: `Argument(type=None)`
requires a dynamic `None`; `Constexpr(type=None)` has a fixed None type and no
dynamic value key. `Argument(value=None)` and `Constexpr(value=None)` bake
`None`, removing that parameter from the public call. A typed pointer can
receive a null `None` or integer 0; an untyped ordinary `None` follows Triton's
constexpr behavior. Baked scalar `int`, `float`, `bool`, or `None` values must
have one effective type, fit it, and cannot also be bound.

### Bound values and fixed devices

`Argument(bind_value=BindValue.TENSOR)` binds an exact `torch.Tensor` or
`torch.nn.Parameter` (or `None`), keeps the owner alive, and reads its current
pointer and storage on every call. Without an explicit pointer type, the
bound tensor fixes its effective dtype at binding; untyped `None` becomes a
constexpr None. Explicit `type=None` accepts only a bound `None` and also compiles
as constexpr None. `BindValue.POINTER` requires one explicit pointer type and
accepts an integer address, `None`, a tensor, or an object with `data_ptr()`.
Its address is captured once at binding, and object owners are retained.

For tensor binding, `TorchAccessMode.STATIC_COMPILE` owns an `at::Tensor` copy;
`RUNTIME_SHIM` and `INTERPRETER` retain the Python object. Torch can preserve
the Python wrapper while the native copy owns its `TensorImpl`. If that wrapper
refers back to the bound handle, the resulting cycle can retain both until the
tensor's back-reference is cleared (for example, `tensor.bound = None`).

Binding annotations return a factory, not a callable launcher:

```python
factory = make_launcher(kernel, extra_annotation={
    "x": Argument(type=tl.pointer_type(tl.float32), specialize=NEVER,
                  bind_value=BindValue.TENSOR),
})
bound = factory.bind(x=tensor)
bound(device, stream, grid, *remaining_public_args)

fixed = make_launcher(kernel, bind_device=True, extra_annotation={
    "x": Argument(type=tl.pointer_type(tl.float32), specialize=NEVER,
                  bind_value=BindValue.TENSOR),
}).bind_device(device, x=tensor)
fixed(stream, grid, *remaining_public_args)
```

`bind()` and `bind_device()` take bound values by keyword; the latter takes one
positional nonnegative int32 device ordinal. Baked and bound values disappear
from the native callable's signature, whose remaining arguments are positional
only. Both methods return a `METH_FASTCALL` builtin, allowing CPython to specialize
explicit positional calls. `inspect.signature(handle)` reads its generated
`__text_signature__`. Each handle keeps its bound objects and extension alive
through its private state at `handle.__self__` (an `_intj_lazy.Launcher`, as for
every launcher); `module_of(handle)` returns the extension module.
Kernel calls support Unicode parameter names, but CPython 3.12's builtin signature
parser limits `inspect.signature(handle)` to ASCII parameter names.
Every launcher and handle has its own kernel cache. A fixed-device handle with
no dynamic key fields stores one nullable kernel directly: no hash or map
operation occurs.

With `verify_annotation=True`, intj checks declared types, ranges, and assumed
facts before a cold or hot lookup. Bound pointer checks happen once when the
address is captured; bound tensor checks use its current storage on each call.
With verification off, declarations are promises, but call shape, value
categories, null handling, and address width remain checked. A false promise
can select the wrong cached binary. `PointerRange(32)` means the entire
underlying storage is at most 2**31 - 1 bytes, not just a tensor view. A raw pointer
address cannot prove that range: even checked `BindValue.POINTER` warns once
at factory creation and trusts the range promise.

## Supported arguments

| passed value | triton type | key |
|---|---|---|
| `torch.Tensor` / `torch.nn.Parameter` (exact type) | `*<dtype>` | `D` when the data pointer is 16B-aligned, `S` (AMD) when the storage is < 2 GiB |
| `int` | `constexpr` when the value is 1, else `i32`/`i64`/`u64` | `D` when divisible by 16 |
| `float` | `fp32` | — |
| `bool` | `u1` | — |
| `None` | `constexpr` | — |
| `tl.constexpr` parameter | `constexpr` | the value itself (`int`, `float`, `bool`, `None`); a `str`, `tl.dtype` or `@triton.jit` function keys by value (see below) |

An untyped, unbaked `tl.constexpr` also takes a `str`, a `tl.dtype`, or a
`@triton.jit` function, keyed by value, as Triton does: two equal strings share
one kernel, and two JIT functions with the same source share one kernel. Each value gets a
small id from a table owned by the launcher (never persisted, never in the
module digest); each argument slot remembers its last object, so passing the
same object again costs one pointer compare (free-threaded builds also take
the launcher's read lock for it). A JIT function's `cache_key` is
read when the function is first seen, so later edits to a callee are not seen
-- Triton has the same limitation. Baking the value with `extra_annotation`
still works and removes it from the call.

`do_not_specialize` and `do_not_specialize_on_alignment` are honoured. The `S` bit is
AMD-only; there it is always part of the key, even when `knobs.amd.use_buffer_ops` is
off, so that toggling the knob can only cost an extra compile, never launch a
buffer-ops binary on a > 2 GiB tensor.

## Not supported

Everything below is refused at `make_launcher` time, or on the first launch that
hits it:

- Backends with no registered `intj.launcher.Backend`. `HipBackend` and
  `CudaBackend` ship; any triton backend whose driver exposes a
  `cuLaunchKernel`-shaped entry point is supported by subclassing `Backend` (dylib,
  launch symbol, error-string convention, whether it specializes pointers on a 2 GiB
  range) and calling `register()`. **CUDA is compile-checked but runtime-untested** --
  there is no NVIDIA GPU on the development machine.
- `TRITON_INTERPRET=1`.
- `dynamic_grid=True`. Use `grid_cpp` or `grid_py` for a callable grid.
- Unsupported parameter annotations, `*args`/`**kwargs`, keyword-only parameters.
- Tuple, `tl.constexpr` object and `TensorDescriptor` arguments, and `str` /
  `tl.dtype` / JIT-function values for anything but an untyped `tl.constexpr`.
- Tensor subclasses other than `torch.nn.Parameter` — the fast path gates on exact
  type, because a subclass can redefine what `data_ptr()` means.

Not refused, and **not detected**: a tensor whose `TensorImpl` overrides `numel()` or
`storage_offset()` rather than storing them, i.e. one with a custom sizes/strides
policy. `torch.nested.nested_tensor(...)` is the case that exists today, and it is
*exact-type* `torch.Tensor`, so the type gate above does not catch it. Its stored
`numel_` is 0 while `numel()` reports the real count, so intj reads it as empty and
passes a null pointer to the kernel; the launch then faults on the GPU rather than
raising. `.to_mkldnn()` has no storage at all and is refused cleanly.

This is a deliberate limitation of reading the fields rather than calling the
accessors. `RUNTIME_SHIM` and `STATIC_COMPILE` share it because `RUNTIME_SHIM`
cannot see the policy bit. Neither kind of tensor is a usable triton kernel
argument in the first place.
- Kernels that read global variables (triton revalidates those on every launch;
  intj cannot, so it refuses instead of silently launching a stale kernel).
- Kernels with pre-run hooks, `num_ctas > 1`, cooperative launches, or non-zero
  global/profile scratch.

## Reaching torch: `torch_access_mode`

Every launch reads three things off each tensor: the data pointer, the dtype, and (on
AMD) the storage size. `make_launcher(..., torch_access_mode=...)` picks how, with
`intj.TorchAccessMode`:

| | how it reads | decode + key, 3 tensors | first build | rebuilt when torch changes |
|---|---|---|---|---|
| `RUNTIME_SHIM` | `TensorImpl`/`StorageImpl` at recorded offsets selected for the running torch | 89 ns | 0.6 s | no |
| `STATIC_COMPILE` | the same fields, offsets supplied by the compiler | 90 ns | 1.9 s | yes |
| `INTERPRETER` | `data_ptr()` / `untyped_storage().nbytes()` through the interpreter | 300 ns | 0.6 s | no |

With `torch_access_mode=None` (the default), intj selects `STATIC_COMPILE` if a
C++ compiler and torch's headers are present, else `RUNTIME_SHIM` if a verified
layout exists, else `INTERPRETER`. An explicit unavailable mode raises.

No mode dlopens `libtorch_cpu.so` or calls an `aoti_torch_*` shim.

`RUNTIME_SHIM` reads offsets from a table of verified torch versions
(`intj/torch_intf/torch_abi.toml`), installed at load by `set_torch_version`.
The recorded `cdata` starts after `PyObject_HEAD`; the generated module adds
its compiled `sizeof(PyObject)` once and saves the absolute offset before any
launch.
A torch with no entry is **refused, never guessed at** — a wrong offset cannot
raise; it reads whatever lies at that address and hands the kernel a pointer
built from it. On an unverified torch,
`torch_access_mode=TorchAccessMode.RUNTIME_SHIM` raises. Automatic selection uses
`INTERPRETER` if `STATIC_COMPILE` is unavailable.

Each entry records the dtypes the running torch had when the offsets were measured
(`[name, element size]`, indexed by dtype code) alongside the offsets themselves. A
torch whose dtype set has drifted from its entry — one renamed, dropped or inserted — is treated as
unverified too, offsets and all: the same refusal, not a partial trust.

`STATIC_COMPILE` needs no entry: the compiler supplies every offset from torch's
headers. It declares the head of `THPVariable` itself instead of including it
(`intj/runtime/intj_thpvariable.h`: a `MaybeOwned<Tensor>` before torch 2.10, a
`Tensor` since); `cpp_detect` below refuses a torch where that declaration is wrong.
On torch older than 2.10 the `STATIC_COMPILE` mode is compile-checked only.

To add a version, run `python -m intj.torch_intf.abi_detect` on a supported
GIL or free-threaded CPython build with that Torch and paste the entry it
prints. The detector subtracts that interpreter's reported header size from
the measured pointer-slot offset. `python -m intj.torch_intf.cpp_detect`
derives the relative offset independently from Torch's C++ headers.
The test suite compares the table entry with both detectors on the Torch
version it runs against.

`INTERPRETER` obtains the pointer and storage size through CPython calls but
reads the dtype code directly from `THPDtype`, whose layout checks itself at load
against the embedded name. It is the independent oracle the test suite compares
the other two against. CPython scalar readers use `STATIC_COMPILE` in every
Torch mode.

## The kernel cache: `kernel_cache`

Most launches turn the spec key into a compiled kernel through a hash map.
`make_launcher(..., kernel_cache=...)` picks which, with `intj.KernelCache`:

| | what it is | first use costs |
|---|---|---|
| `INTJ` (default) | intj's open-addressed table, 72 lines | nothing |
| `TSL` | `tsl::robin_map`, handed the precomputed hash | ~3 s, a download |
| `ABSL` | `absl::flat_hash_map` | a download, then a build of 90 libraries: 28 s on 224 cores, minutes on a few |

`TSL` and `ABSL` are C++ maps, so either one compiles the whole module as C++
even under `RUNTIME_SHIM` or `INTERPRETER` access.

Neither is vendored, and **nothing installed on the machine is searched for**:
`make_launcher` downloads a pinned version, checks its sha256, and (for
abseil) builds it into `$TRITON_HOME/.triton/intj/deps/`. What a module was
built against is then a property of intj's cache rather than of the host. It
happens once, on the path that was already going to invoke a compiler, and is
announced on stderr — an implicit download is otherwise indistinguishable from
a hang. To get it over with ahead of time, or before going offline:

    python -m intj.kernel_cache all
    python -m intj.kernel_cache --force absl   # discard a half-written tree

One installer at a time, machine-wide, so the ranks of a torchrun job that all
miss together do not build into one tree. If the install fails, the error
carries that command and the reason, and the failure is remembered rather than
re-attempted on every later `make_launcher`.

Measured through `tests/bench_kernel_cache.cpp` (google/benchmark, ns per lookup,
hit), at the two ends of the range a packed key reaches — 8 bytes is a kernel with
no `tl.constexpr` parameter, where the hash is a bijection and the slot carries no
key at all; 40 bytes is twelve parameters of which three are constexpr:

| entries | `INTJ` | `TSL` | `ABSL` | | `INTJ` | `TSL` | `ABSL` |
|---|---|---|---|---|---|---|---|
| | *8-byte key* | | | | *40-byte key* | | |
| 1 | 1.39 | 1.84 | 1.39 | | 2.94 | 3.19 | 2.68 |
| 8 | 1.56 | 1.89 | 4.24 | | 3.16 | 3.55 | 9.47 |
| 512 | 2.05 | 2.62 | 4.94 | | 4.46 | 4.94 | 11.22 |

`ABSL` wins at one entry because its small-object path skips hashing entirely;
past that it re-computes the hash intj already has, and no abseil API takes one.
That is the whole reason the default is the 72 lines.

`pytest tests/test_kernel_cache.py` builds and runs that benchmark for whichever
backends are present, and skips without `$INTJ_BENCHMARK_ROOT`.

## How a launch works

0. The first call builds the module and swaps the launcher's entry; later calls
   start at step 1.
1. Parse `device` and `stream`, compute the selected grid, and return early on a
   zero-volume grid.
2. Decode each public or bound argument, checking annotations if requested, and
   pack the dynamic key fields. How tensor fields are read is `torch_access_mode`'s
   choice; no mode calls an `aoti_torch_*` shim. The direct `AUTO` path accumulates
   descriptor bytes in 32-bit registers.
3. Hash and look up the key in the launcher's own cache. A fixed-device handle
   with no dynamic key fields reads its one nullable kernel directly, without
   hashing or a map.
4. On a miss, call back into Python: build the annotated `CompilerInput`, compare
   it with any input this launcher previously recorded for that key, and, unless
   the module's compile cache already holds that kernel (a sibling launcher
   missed first), pass its
   `triton.compiler.ASTSource` (or `GluonASTSource`) and canonical options to
   `triton.compiler.compile`.
   `_init_handles` loads the GPU module; the entry records the function handle,
   block dim and LDS size.
5. Pack the param array (plus the two mandatory trailing scratch slots) and call
   `hipModuleLaunchKernel` / `cuLaunchKernel` (identical argument lists).

Each cached `CompiledKernel` is kept alive by the module's compile cache (with
`return_compiled=True`, by the launcher records that return it), so its GPU
module stays loaded for as long as that owner lives.

## Correctness

The spec key packs only dynamic fields: type and applicable specialization
descriptors, constexpr payloads, and the device ordinal unless it is fixed.
Fields are laid out by descending alignment (8-, 4-, 2-, then 1-byte widths),
with stable declaration order among equal widths, and padded to a zeroed
64-bit word boundary. Baked values, bound values, and fixed facts contribute
no runtime field. The dtype index for pointers comes from the running torch
and Triton; `tl.int1` canonicalizes to `u1`. A fixed-device handle with no
fields has an empty key and uses its nullable-kernel path.

The invariant intj must not break is:

> if two argument tuples produce the same intj key, their annotated compiler
> inputs have the same signature, constexpr values, and specialization attributes

A coarser key does not crash — it launches, say, a `tt.divisibility = 16` binary on an
unaligned pointer. `tests/test_launcher.py::test_spec_key_is_never_coarser_than_triton`
checks it directly through the module's `spec_key(launcher, device, *args)` debug entry
point, and every
GPU cache miss builds a `CompilerInput` containing the final annotated
`ASTSource` signature, tagged constexpr values, and attributes. The callback
compares it with any input previously recorded for the same key and rejects a
mismatch. Compilation uses that source directly without the JIT function's
warmup, binder, or device cache.

## Launch benchmark

```bash
PYTHONPATH=$PWD /tmp/gb2/bin/python benchmarks/bench_launch.py --no-gpu --iters 20000 --batches 7
PYTHONPATH=$PWD /tmp/gb2/bin/python benchmarks/bench_launch.py --iters 20000 --batches 7
```

The first command uses CPU tensors and intj's existing host compile callback;
it measures decoding, key/cache work, and vectorcall overhead only. It does
not query a GPU target or driver, compile a GPU kernel, or launch one. The
second uses GPU tensors and includes driver launch overhead. Each row warms
its launcher, then reports the median nanoseconds per call across repeated
batches, absolute and percentage deltas against the same kernel's `auto map`
baseline, and separate factory construction and binding times in milliseconds.
The matrix covers reduced keys, checked and unchecked facts, baked values,
bound tensor and pointer handles, fixed-device map, and fixed-device no-map.
These are measurements, not pytest performance limits. CUDA remains
compile-checked but runtime-untested on the development machine.

## Cache

Artifacts land at

    $TRITON_HOME/.triton/intj/<digest>/<module>/<kernel><EXT_SUFFIX>
    $TRITON_HOME/.triton/intj/<digest>/<module>/<kernel>.c or .cpp

next to triton's own caches. The rendered source is kept beside the binary: it is what
you read when a launch misbehaves. The symbol is the kernel's own name
(`PyInit_<kernel>`) — what `perf` and `/proc/<pid>/maps` show. Two builds of one kernel
are two directories and two independent modules.

Modules are loaded by hand, so none of this reaches `sys.modules`: the module's name
is just the kernel's, a label rather than a lookup key — two builds of one kernel share
it and are told apart by `__file__`.

`make_launcher` looks for its module in three places, in order: the process-level
`ModuleKey` dict (same module, so the same compile cache), the `.so` on disk (loaded
as-is, nothing rendered or compiled), and only then renders and builds. A warm
process takes ~0.2 ms, a warm disk ~0.6 ms, against ~250 ms for a build.

The digest covers the kernel source (`cache_key`), the parameter
table (including `do_not_specialize*`, which `cache_key` does not cover), the target,
the canonicalized options, intj's own `runtime/` bytes, the triton build, the compiler,
and `EXT_SUFFIX`. `rm -rf $TRITON_HOME/.triton/intj` clears every intj artifact.

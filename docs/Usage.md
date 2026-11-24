# INTJ usage

## `make_launcher`

```python
from intj import KernelCache, TorchAccessMode, make_launcher

launcher = make_launcher(
    jit_func,              # a @triton.jit function, positional; omit it for a decorator
    # every argument below is keyword-only
    options=None,          # triton compile options, e.g. {"num_warps": 8}
    dynamic_options=(),    # compile options / knobs.<group>.<name> passed per call
    bind_device=None,      # a device ordinal: the launcher takes no device argument
    bind_args=None,        # bound parameter name -> value
    grid_arg_num=None,     # 1, 2, or 3 separate grid dimensions
    grid_cpp=None,         # annotated def compiled into the extension
    grid_py=None,          # Python callback evaluated on each launch
    extra_annotation=None, # parameter name -> annotation or shorthand
    torch_access_mode=None, # automatic; pass a TorchAccessMode to select one
    kernel_cache=KernelCache.INTJ,
    no_gpu=False,          # host decode/cache path, without GPU work
    verify_annotation=False,
    return_compiled=False, # return the cached Triton CompiledKernel
    assume_constant_globals=False, # accept kernels reading globals
    ndebug=True,           # compile the module with NDEBUG (asserts off)
    delayed=None,          # () -> more of these keywords, replacing them, run on first use
)
```

`make_launcher` always returns a launcher. Calling it launches the kernel,
with positional arguments only, in this order: `device`, `stream`, the grid
controls, the `dynamic_options` values, then the kernel's public parameters
(bound ones omitted). The grid option picks the grid controls:

```python
launcher(device, stream, gx[, gy[, gz]], *dynamic_options, *kernel_args)  # grid_arg_num=1, 2 or 3
launcher(device, stream, *grid_cpp_args, *dynamic_options, *kernel_args)  # grid_cpp=grid_fn: grid_fn's keyword-only inputs
launcher(device, stream, *dynamic_options, *kernel_args)                  # grid_py=grid_fn: grid_fn(meta) returns the grid
launcher(stream, grid, *dynamic_options, *kernel_args)                    # bind_device=<ordinal>: no device
```

It returns None, or with [`return_compiled=True`](#return_compiled) the
cached Triton `CompiledKernel`.

The launcher is a C function whose `__self__` is an `intj.launcher.Launcher`
with these members:

```python
k = launcher.__self__
k.options                             # the MakeLauncherOption; once built, the resolved one (delayed=None)
k.check()                             # run every check the first call would; builds nothing
k.build()                             # build now if not yet built; returns the rendered module
k.bind_args(**values)                 # new launcher, values merged into bind_args
k.bind_device_args(ordinal, **values) # new launcher fixed to a device (None unbinds it), values merged
k.unbind_args()                       # new launcher with no bound values
k.unbind_device_args()                # new launcher taking the device per call
k.replace_delayed(fn)                 # new launcher whose delayed is fn (None drops it)
k.chain_delayed(fn)                   # new launcher running fn after delayed; fn's keywords win
k.resolve_delayed()                   # new launcher with delayed called now and applied
k.spec_key(device, *dynamic_options, *kernel_args)  # debug: (key bytes, nparams); builds first
k.key_chain(device, *dynamic_options, *kernel_args) # debug: ((key bytes, ...), found); builds first
```

The methods returning a new launcher return it unbuilt. They raise
`TypeError` only for a `replace_delayed` or `chain_delayed` argument that is
not callable; the new launcher's `check()` or first call validates the rest.
Each has an in-place twin with a leading underscore (`_bind_args()`, ...) that
edits this launcher instead and raises `RuntimeError` once it is built, except
`_resolve_delayed()`, which then does nothing: a built launcher has no
`delayed` left. The bind methods edit the explicit `bind_device` and
`bind_args`: a `delayed` that returns either still replaces the edited value
whole when the launcher is built. Call `make_launcher` once, outside any hot
loop.

### Device and stream

A call starts with `device`, a device index in `[0, 256)`
(`torch.cuda.current_device()`), and `stream`, the raw stream handle
(`torch.cuda.current_stream().cuda_stream`). Each device gets its own keys, so
one launcher serves every device.

`bind_device=<ordinal>`, or `bind_device_args(ordinal)`, fixes the device: the
call drops `device` and starts with `stream`.

### Kernel grid

Choose at most one grid option; the grid controls follow `stream`:

- none: one `grid` argument, an `int` or a 1–3 element tuple/list.
- `grid_arg_num=1`, `2` or `3`: that many separate int arguments.
- `grid_cpp=grid_fn`: an annotated `def` compiled into the extension. Its
  positional parameters read the same-name kernel arguments; its keyword-only
  parameters are extra call arguments.
- `grid_py=grid_fn`: a Python callable given a `meta` dict of the kernel's
  arguments on every launch.

Dimensions are exact ints in `[0, 2**32)`; omitted ones are 1, and a
zero-volume grid skips the launch. The grid is not part of the kernel's
specialization key. See
[Grid functions](#grid-functions-grid_cpp-and-grid_py).

### Options passing

Compile options reach a compile in four ways:

- `options={"num_warps": 8, ...}`: fixed for the launcher and part of its
  module identity.
- `dynamic_options=("num_warps", "knobs.runtime.debug", ...)`: one value per
  name in each call, right after the grid controls, keyed. A name is a compile
  option or a Triton knob path.
- a `tl.constexpr` parameter named like an option (`num_warps: tl.constexpr`):
  its value sets the option too. Pass it as a kernel argument, fix it with
  `options={"num_warps": 8}` (which bakes it), or name it in `dynamic_options`
  (its value then moves from the kernel arguments to the dynamic values). Any
  other `tl.constexpr` parameter in `options=` or `dynamic_options` is refused
  on the first call, as an unknown compile option; a runtime parameter in
  `dynamic_options` is refused at `check()`.
- an autotune config or heuristic that assigns the option.

`options` refuses `device`, `stream`, `device_type` and `warp_size`, and is
canonicalized through the backend's `parse_options`, so spellings that mean the
same (`{}` and `{"num_warps": 4}` on AMD) share one module. An explicit `debug`
option overrides the kernel's debug default; Triton's runtime debug knob can
still enable it.

[`delayed`](#delayed-options-delayed) can return any of these keywords when
they need the GPU to compute. Undeclared knobs that feed options (`debug`, `instrumentation_mode`, fpsan
casts) are read once, at the first call, and fixed for the launcher; other
cache-invalidating knobs are not keyed (see `TODO.md`). See
[Per-call compile options and knobs](#per-call-compile-options-and-knobs-dynamic_options)
and [Autotune and heuristics](#autotune-and-heuristics).

### Argument passing

The kernel's parameters come last in a call, positionally, in declaration
order, constexprs included. Keywords and parameter defaults are not supported:
pass a defaulted parameter too, or bake it. A parameter leaves the call when it
is:

- baked: `extra_annotation={"BLOCK": Constexpr(value=128)}` (or
  `Argument(value=...)`) fixes the value in the module, as does `options=` for
  a `tl.constexpr` parameter named like a compile option.
- a dynamic option: a `tl.constexpr` parameter named like a compile option and
  listed in `dynamic_options` is passed with the dynamic values instead.
- bound: annotated `Argument(specialize=NEVER, bind_value=BindValue.TENSOR)`
  (or `POINTER`, or another `specialize` that leaves no fact `AUTO`) and given
  through `bind_args=`, `bind_args()` or `bind_device_args()`. See
  [Bound values](#bound-values).

Each value is decoded and keyed as Triton would specialize it. See
[Argument annotations](#argument-annotations).

### Delayed options: `delayed`

Some `make_launcher` arguments depend on the GPU (a backend-dependent
`num_warps`, say), but the decorator runs when its module is imported, where
you usually do not want to query the GPU. `delayed` defers evaluating them to
the launcher's first call: it takes a function of no arguments returning those
`make_launcher` keywords.

```python
@make_launcher(delayed=lambda: {"options": {"num_warps": pick_warps()}})
@triton.jit
def k(...): ...
```

It may return any keyword but `jit_func` and `delayed`, `dynamic_options`, the
grid options and `bind_args` included. Each one it returns replaces the
explicit one whole (a returned `options` dict is not merged into the
decorator's, nor a `bind_args` into what `bind_args()` merged), entering
module identity the same way.

Each launcher calls it on its first call (`check()` calls it too, storing
nothing); once built, `launcher.__self__.options` holds the result, with
`delayed=None`. A result that is not a mapping of `make_launcher` keywords
raises `TypeError`. It may call or check another launcher, waiting if another
thread is building that one; calling or checking the launcher being built, or
a wait that would deadlock (two builds each waiting on the other), raises
`RuntimeError`.

`replace_delayed(fn)` swaps it, `chain_delayed(fn)` runs `fn` after it with
`fn`'s keywords winning, and `resolve_delayed()` calls it now; each returns a
new, unbuilt launcher.

### Other keywords

#### `extra_annotation`

Parameter name -> annotation or shorthand: fixes types or specialization
facts, bakes values, or marks values for binding. See
[Argument annotations](#argument-annotations).

#### `torch_access_mode`

How the module reads a tensor's pointer, dtype and storage size, as an
`intj.TorchAccessMode`: `RUNTIME_SHIM` (recorded field offsets for verified
torch versions: any release of a recorded major.minor, but a nightly, source
build or release candidate only where its exact version is recorded),
`STATIC_COMPILE` (offsets from torch's C++ headers; rebuilt when the torch
version changes, pre-release and local parts included) or `INTERPRETER`
(through Python calls; slowest). None
selects `STATIC_COMPILE` if a C++ compiler and torch's headers are present,
else `RUNTIME_SHIM` if the running torch is verified, else `INTERPRETER`.

#### `kernel_cache`

The hash map behind the kernel cache, as an `intj.KernelCache`: `INTJ`
(default, intj's own table), `TSL` (`tsl::robin_map`) or `ABSL`
(`absl::flat_hash_map`). `TSL` and `ABSL` compile the module as C++, and are
downloaded at a pinned version (abseil also built) into
`$TRITON_HOME/.triton/intj/deps/` on first use; `python -m intj.kernel_cache
all` does that ahead of time, and `--force` reinstalls, renaming the old tree
away before deleting it so no launcher sees it half-deleted. Unpacking needs
`tarfile` extraction filters (Python 3.8.17, 3.9.17, 3.10.12, 3.11.4, 3.12 or
later); an older interpreter refuses `TSL` and `ABSL` with `UnsupportedKernel`
(tested by removing the filters, untested on such an interpreter).

#### `no_gpu`

`no_gpu=True` renders and times host decoding and cache lookup without a GPU
target or driver: the call keeps its shape, but it builds for
`intj.backend.HostBackend`, a fake driver whose launch does nothing. It accepts
only default compile options, and refuses `dynamic_options`, autotune and
heuristics chains, and `return_compiled`.

#### `verify_annotation`

With `verify_annotation=True`, intj checks declared types, ranges, and assumed
facts before a cold or hot lookup. Bound pointer checks happen once when the
address is captured; bound tensor checks use its current storage on each call.
With verification off, declarations are promises, but call shape, value
categories, null handling, and address width remain checked. A false promise
can select the wrong cached binary. `PointerRange(32)` means the entire
underlying storage is at most 2**31 - 1 bytes, not just a tensor view. A raw pointer
address cannot prove that range: even checked `BindValue.POINTER` warns, per
launcher, on each `check()` and at the build, and trusts the range promise.

#### `return_compiled`

`return_compiled=True` makes a call return the cached Triton `CompiledKernel`,
the same object for every call of one specialization. A zero-volume grid then
still compiles or finds the kernel but does not dispatch it. It needs GPU mode.

#### `assume_constant_globals`

`assume_constant_globals=True` accepts a kernel that reads global variables
(`X = tl.constexpr(...)`, or a plain value a `constexpr_function` reads),
which is refused by default. It is a caller promise: **those globals never
change after the kernel is first hashed** (at `check()`, the first time
Triton computes its `cache_key`). Their values then become part of the
module's identity, so two values never share a module, and a launch never
reads them again -- no hit-path cost. Changing one afterwards is unsupported
and **not detected**: the launcher keeps launching what it built, and a later
cache miss compiles with the new value under the old identity. Triton raises
instead, on every launch. Values must be `tl.constexpr` wrappers, `int`,
`float`, `bool`, `str`, `None`, `tl.dtype`, enum members (Gluon's
`PropagateNan.ALL`), classes, or tuples of these; anything else raises
`UnsupportedKernel` naming the global and its type.

A class global (a `NamedTuple` the kernel constructs, a Gluon layout class such as
`AMDWMMALayout`) is keyed by `module.qualname`, like a called `@triton.jit`
function; the kernel's source is already in the key. Replacing the class under the
same name is not detected, so each launcher emits an `intj.ClassGlobalWarning`
(a `RuntimeWarning`) on every `check()` and again at its build, naming the classes;
launches never warn.
Silence it with
`warnings.filterwarnings("ignore", category=intj.ClassGlobalWarning)` or
`-W ignore::intj.ClassGlobalWarning`. A plain global is not in Triton's own compile-cache key,
so Triton hands two kernels that differ only in one the same binary; intj
keeps them in separate modules but compiles through that cache like Triton.

#### `ndebug`

`ndebug=True`, the default, defines `NDEBUG` at the top of the rendered source,
so `assert()` in CPython's headers and intj's runtime is compiled out: the
module checks its inputs explicitly. `ndebug=False` undefines it, so
those assertions abort the process if a launch breaks one; it is a debugging
aid, and builds a separate module.

### Build on first call

To avoid unnecessary work at module import time, `make_launcher` records its
keywords, as a `MakeLauncherOption`, and checks nothing about the kernel; it
touches no GPU and imports no torch. Its first call in a process also imports
triton and loads `_intj_launcher_base`, the kernel-free C base every launcher
uses, compiling it with the host C compiler unless cached. The launcher is
only built when first called: it calls `delayed`, validates arguments and
annotations, analyzes an autotune/heuristics chain, lowers `grid_cpp`, renders
the launcher source code from a Jinja template, and builds it. Then the launcher is used to cache and
launch kernels.

To run the same checks the launcher build would, without building, call
`launcher.__self__.check()`.

### Per-call compile options and knobs: `dynamic_options`

To pass `CUDAOptions`/`HIPOptions` fields per call, declare them in
`dynamic_options` and pass their values as launcher arguments. The kernel is
keyed on these values, so each combination of them selects its own compiled
kernel:

```python
@intj.make_launcher(dynamic_options=("num_warps", "num_stages"))
@triton.jit
def k(x, out, n, BLOCK: tl.constexpr): ...

k(device, stream, grid, 4, 2, x, out, n, 128)   # num_warps=4, num_stages=2
k(device, stream, grid, 8, 3, x, out, n, 128)   # another kernel: 8 warps, 3 stages
```

`dynamic_options` also accepts Triton knobs (`knobs.<group>.<name>`). Ideally
only `CUDAOptions`/`HIPOptions` would affect compilation, and knobs would only
change those options' defaults. But Triton reads some knobs directly in its
compilation pipeline, bypassing the options, so a knob can matter on its own.
Knobs are process-wide and intj never changes them: a knob in
`dynamic_options` only changes the kernel key, so keep the knob's value and the
argument you pass consistent.

```python
@intj.make_launcher(dynamic_options=("knobs.runtime.debug",))
@triton.jit
def k(x, out, n, BLOCK: tl.constexpr): ...

triton.knobs.runtime.debug = True                 # set the knob through Triton
k(device, stream, grid, True, x, out, n, 128)     # and pass the same value
```

Refused at `check()`: an unknown knob path, a repeated name, a name also in
`options=` (for a knob, the option it feeds), `device`, `stream`,
`device_type`, `warp_size`, `instrumentation_mode` and
`fpsan_homomorphic_casts` (a knob overwrites each: declare its knob path
instead), a name an autotune config sets, and `no_gpu=True`.
An unknown compile option is refused on the first call. A `tl.constexpr`
option parameter named here is keyed exactly, takes no annotation but
`tl.constexpr`, and cannot also be baked.

### Grid functions: `grid_cpp` and `grid_py`

Triton supports computing the grid with a lambda, which can read the runtime
arguments and `tl.constexpr` values. But calling a Python function on every
launch adds overhead. `grid_cpp` instead compiles the grid function to C, whose
source is copied into the launcher. Only limited syntax is supported; if you
need syntax intj does not support yet, contact the maintainers or vibe-code it
yourself.

```python
# kernel parameters: (out, n, BLOCK)
def grid(n: int, BLOCK: int, *, cap: int):
    return (min(cap, triton.cdiv(n, BLOCK)),)

launcher = make_launcher(kernel, grid_cpp=grid)
launcher(device, stream, cap, out, n, BLOCK)
```

The function is an undecorated `def` (not a lambda) whose parameters are all
annotated `int` or `bool`. Positional parameters read the same-name kernel
values (not bound ones); keyword-only ones are extra call arguments, right
after `stream`. It may use int/bool literals and names, local assignments,
unary `+`/`-`, `+ - * //`, `triton.cdiv`, two-argument `min` and conditional
expressions, and must return a literal tuple/list of 1–3 ints.

As a fallback, `grid_py` takes a Python function, given a `meta` dict of every
kernel parameter by name, but it has that launch-time overhead:

```python
launcher = make_launcher(
    kernel,
    grid_py=lambda meta: (triton.cdiv(meta["n"], meta["BLOCK"]),),
)
launcher(device, stream, out, n, BLOCK)
```

## Autotune and heuristics

`make_launcher` accepts a `@triton.jit` function wrapped in any nesting of
`@triton.heuristics` layers and at most one `@triton.autotune` layer:

```python
@make_launcher(grid_cpp=grid)   # grid may read tuned values such as BLOCK
@triton.autotune(configs=[...], key=["N"])
@triton.heuristics({"EVEN": lambda a: a["N"] % a["BLOCK"] == 0})
@triton.jit
def kernel(x, out, N, BLOCK: tl.constexpr, EVEN: tl.constexpr): ...

kernel(device, stream, x, out, N)   # tuned and heuristic values are not passed
```

On a miss, Triton tunes and launches through intj: pruning, `reset_to_zero`,
`restore_value`, `do_bench` and disk caching all run as usual, and
`best_config`, `bench_time` and `configs_timings` are copied back to your
tuners. On a hit, intj launches without calling Triton or any heuristic
written in Python. Triton runs a `Config.pre_hook` before every launch, so a
config with one is refused at `check()`.

intj keys a hit on:

- the exact value of every caller argument an autotune layer keys on, for
  every layer, outer ones included (`int`, `bool`, or `float` by its fp64
  bits). As in Triton, an autotune layer with a single config keys on
  nothing. A tensor key is refused with `UnsupportedKernel`: at `check()`
  when annotated as a tensor or pointer, else at the call;
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
  result that is compiled to C (refused at `check()`; use `grid_py`,
  which sees every value in `meta`). Its grid is computed after the lookup,
  and a zero-volume result skips the launch.
- With the default grid or `grid_arg_num`, a zero-volume grid skips the
  launch. With `return_compiled=True`, which would still compile, a miss on
  one raises `ValueError("intj: cannot tune on a zero-volume grid")` instead.

Refused with `UnsupportedKernel`: `no_gpu=True`, bound values,
`extra_annotation` or `options` naming a tuned value, configs with
`num_ctas != 1` or a `pre_hook`, layers assigning a runtime (non-constexpr) parameter, a
name assigned twice (including two nested `@triton.autotune` layers, which
Triton itself cannot run), a heuristic or autotune key reading a value an
inner layer assigns, heuristics outside the subset above, and a tuned value
read by C that is not an `int` or `bool` (autotune config values at
`check()`, heuristic results on the miss). Each launcher
`bind_device_args(ordinal)` returns owns its own tuner caches and tunes
again; `bind_args()` binds values, which are refused.
Free-threaded Python builds are compile-untested.

## Argument annotations

### Where they go

Each annotation on a `triton.jit` function's parameters can also be passed
through `extra_annotation`, under the parameter name:

| written | means |
|---|---|
| `tl.constexpr` | `Constexpr()` |
| a Triton dtype, or `None` | `Argument(type=...)` |
| a type Triton recognizes (`int`, `bool`, `float`, `tl.int64`, `"tl.int64"`, ...) | `Argument(type=<Triton's type>)`, converted as Triton converts it (see [Types](#types)) |
| `torch.Tensor` or `tl.tensor` | an exact tensor only |
| `torch.Tensor \| None`, `Optional[...]`, `Union[..., None]` | an exact tensor or `None` |

The two sources merge field by field: an unspecified field comes from the other
source, equal fields agree, and conflicting ones raise at `check()`. Triton's
`do_not_specialize` and `do_not_specialize_on_alignment` merge in as the
matching `NEVER` facts.

### `Argument` and `Constexpr`

`Argument` describes an ordinary parameter, `Constexpr` a `tl.constexpr` one.
Every field is optional:

| field | of | meaning |
|---|---|---|
| `type` | both | one dtype, or a tuple allowlist such as `INT_TYPES` (`tl.int32`, `tl.int64`, `tl.uint64`) or `FLOAT_TYPES` (`tl.float32`). Missing or `AUTO` keeps Triton's inference. `Argument` checks the inferred type; `Constexpr` picks the narrowest type that fits, then the name. Neither converts the value. |
| `specialize` | `Argument` | `AUTO`, `NEVER`, `Assume(...)` or `Exactly(...)`: see [Specialization facts](#specialization-facts) |
| `value` | both | bakes the value into the module and removes the parameter from the call |
| `bind_value` | `Argument` | `BindValue.TENSOR` or `BindValue.POINTER`: the value comes from `bind_args` |
| `ptr_range` | `Argument` | reads a tensor's `ptr_range()` as Triton does: see [Pointer range](#pointer-range-ptr_rangetrue) |
| `power_of_two_or_zero` | `Constexpr` | keys an integer in one byte: 0 is 0, `2**k` is `k+1`, `-2**k` is `0x80 \| (k+1)`. Other values are refused under `verify_annotation`, else may share a binary. |

### Types

`type=` takes one of these dtypes, or a tuple of them as an allowlist:

| dtype | `Argument` | `Constexpr` |
|---|---|---|
| `tl.int1` (keyed as `u1`) | yes | yes |
| signed and unsigned 8 to 64-bit integers | yes | yes |
| `tl.float32` | yes | no |
| `tl.float64` | yes | yes |
| `tl.pointer_type(element_dtype)`, for supported tensor dtypes | yes; `None` or 0 is null | no |
| `None` | a per-call `None` | a fixed `None`, adding no key |

An omitted `type` is not `None`: an untyped `None` is a constexpr, as in
Triton. A bare constexpr keeps Triton's Python value; a typed one checks that
the value fits.

These annotations behave specially:

| written | accepts | behavior |
|---|---|---|
| a type Triton recognizes (`n: int`, `tl.int64`, `"tl.int64"`, ...) | what Triton's launcher converts: `3` into a `float` is `3.0`; `2.5` into an `int` raises `TypeError`; beyond 64 bits raises `OverflowError`; narrower integers truncate | equal-to-1 fact is `NEVER`, as in Triton; `verify_annotation` adds no check |
| `torch.Tensor`, `tl.tensor` | an exact `torch.Tensor` or `torch.nn.Parameter`; anything else raises `TypeError` | specializes like an unannotated tensor; combines with a pointer type, conflicts with a constexpr, a scalar or `None` type, a baked value or a binding |
| `torch.Tensor \| None`, `Optional[...]`, `Union[..., None]` | the same, or `None` | `None` is keyed as Triton keys an unannotated `None` |

### Specialization facts

A **fact** is one property Triton can specialize a parameter on.
`Argument(specialize=...)` takes one of these policies; anything else raises
`TypeError` when the `Argument` is constructed:

| policy | listed facts | other facts |
|---|---|---|
| `AUTO` | | `AUTO` |
| `NEVER` | | `NEVER` |
| `Assume(F, ...)` | assumed | unset |
| `Exactly(F, ...)` | assumed | `NEVER` |

An unset fact, from `Assume` or an omitted `specialize`, takes the other
annotation source's mode or `do_not_specialize`'s, and is `AUTO` if neither
sets it. Each fact then ends up in one mode:

| mode | compiled in | keyed | set by |
|---|---|---|---|
| `AUTO` | when the call's value has it | yes | `AUTO`, or an unset fact nothing else sets |
| assumed | always; a promise unless `verify_annotation=True` | no | listed in `Assume` or `Exactly` |
| `NEVER` | never | no | `NEVER`, `Exactly`'s unlisted facts, `do_not_specialize` |

A set mode conflicts with a different one set elsewhere:

```python
@triton.jit(do_not_specialize_on_alignment=["x"])
def k(x, n): ...

# fine: Assume leaves Aligned(16) unset, so it takes NEVER from the decorator
make_launcher(k, extra_annotation={"x": Argument(specialize=Assume(PointerRange(32)))})

# ValueError at check(): AUTO sets Aligned(16), which the decorator sets to NEVER
make_launcher(k, extra_annotation={"x": Argument(specialize=AUTO)})
```

`Assume` and `Exactly` accept only these facts (another `Fact` type or a
non-int value raises `TypeError`, another value such as `Aligned(8)`
`ValueError`):

| fact | applies to | holds when | compiles as |
|---|---|---|---|
| `EqualTo(1)` | integers | the value is 1 | a constexpr 1 |
| `Aligned(16)` | integers and pointers | the value or data pointer is divisible by 16 | `tt.divisibility = 16` |
| `PointerRange(32)` | pointers (keyed on AMD) | the tensor's storage is under 2 GiB | `tt.pointer_range = 32` |

An assumed fact must apply to the parameter's type, and assuming both
`EqualTo(1)` and `Aligned(16)` on an integer is refused. `Exactly(...)` states
what a hand-written `ASTSource` attrs entry does: on an aligned pointer,
`Assume(PointerRange(32))` still compiles `tt.divisibility = 16` (its
alignment stays `AUTO`), `Exactly(PointerRange(32))` does not.

Each fact merges on its own, in this order:

1. The inline annotation with `extra_annotation`: an unset mode takes the other
   source's, equal modes agree, and different ones raise `ValueError` at
   `check()`. `Exactly(Aligned(16))` agrees with `Assume(Aligned(16))`, but
   conflicts with `AUTO`, `NEVER` and `Assume(PointerRange(32))`.
2. Triton's `do_not_specialize`, as `NEVER` for all three facts, and
   `do_not_specialize_on_alignment`, as `NEVER` for `Aligned(16)`. An `AUTO` or
   assumed mode there conflicts the same way.
3. A fact still unset becomes `AUTO`.
4. A Triton-recognized type annotation (`n: int`) turns an `AUTO` `EqualTo(1)`
   into `NEVER`, as Triton does.

### Pointer range: `ptr_range=True`

Triton decides a tensor's pointer range by `ptr_range() <= 2**31 - 1` when the
tensor has a `ptr_range` attribute, else by its storage size. intj reads the
storage size alone unless the parameter is annotated `Argument(ptr_range=True)`.
Then, on every call, it does what Triton does: `hasattr(x, "ptr_range")`, and if
that holds, a second lookup and `x.ptr_range()`. Only the first lookup's
`AttributeError` is swallowed. That costs one or two attribute lookups per
tensor per call, so it is off by default. A miss compiles the range its key
measured. It applies wherever intj measures the range (an `AUTO` fact on AMD,
or an assumed one under `verify_annotation=True`); on a parameter whose range is
never measured (a non-pointer, `NEVER`, `do_not_specialize`, a baked or bound
value) it raises at `check()`.

### Bound values

A bound parameter leaves the call: its value comes once from `bind_args=`,
`bind_args(**values)` or `bind_device_args(ordinal, **values)`. Annotate it with
`bind_value`:

- `BindValue.TENSOR` takes an exact `torch.Tensor` or `torch.nn.Parameter`, or
  `None`, keeps it alive, and reads its current pointer and storage on every
  call. Without a pointer `type`, the bound tensor fixes the dtype, and a bound
  `None` compiles as a constexpr `None`; `type=None` accepts only `None`.
- `BindValue.POINTER` needs exactly one explicit pointer type
  (`type=tl.pointer_type(tl.float32)`) and takes an int address, `None`, a
  tensor, or an object with `data_ptr()`. The address is read once, at
  binding, and an object is kept alive.

The module is built before the value is known, so no fact that applies may be
`AUTO`: use `specialize=NEVER` or `Exactly(...)`. The default, a fact left
`AUTO`, or `POINTER` without one pointer type raises `ValueError` at
`check()`; a bound value missing from `bind_args` raises `TypeError` there.

```python
@triton.jit
def k(x, out, n, BLOCK: tl.constexpr): ...

launcher = make_launcher(k, extra_annotation={
    "x": Argument(specialize=NEVER, bind_value=BindValue.TENSOR),
})
bound = launcher.__self__.bind_args(x=x)
bound(device, stream, grid, out, n, 128)          # x is not passed

fixed = launcher.__self__.bind_device_args(device, x=x)
fixed(stream, grid, out, n, 128)
```

The other modes hold a bound tensor's Python object; `STATIC_COMPILE` holds an
`at::Tensor` copy, which the garbage collector cannot see through. There, if
the tensor's Python object refers back to the launcher, the cycle keeps both
alive until that reference is cleared (`tensor.bound = None`); untested.

## Triton-style migration bridge: `intj.compat`

`intj.compat.launch(kernel, grid, /, *args, return_compiled=False, **kwargs)`
launches a plain `@triton.jit` function as `kernel[grid](*args, **kwargs)`
would: kernel arguments by position or keyword, parameter defaults applied,
and any other keyword as a compile option. `grid` is an int, a 1–3 element
tuple/list, or a callable given the kernel's values as a `meta` dict:

```python
from intj.compat import launch

launch(kernel, (triton.cdiv(n, BLOCK),), out, n=n, BLOCK=BLOCK, num_warps=4)
```

Each call binds the Python arguments and reads the current device and stream,
then reuses a launcher it cached for that kernel and those options; call
`make_launcher` yourself for hot loops. A `str`, `tl.dtype` or JIT-function
constexpr is baked into the launcher it selects, and a `TensorWrapper`
(`triton.reinterpret`) passes its base tensor typed as the wrapper's dtype. As
in Triton, a keyword naming both a `tl.constexpr` parameter and a compile
option (`num_warps=8` for a `num_warps: tl.constexpr` kernel) sets both.
`return_compiled=True` returns the cached Triton `CompiledKernel`.

Nothing falls back to Triton: it raises `UnsupportedKernel` for what
`make_launcher` refuses, for `@triton.autotune` and `@triton.heuristics`
wrappers (give those to `make_launcher`, see
[Autotune and heuristics](#autotune-and-heuristics)), and while a Triton launch
enter or exit hook is registered, checked on every call. An unpinned CPU tensor
raises `ValueError`. It cannot pass `assume_constant_globals`, so a kernel that
reads globals stays refused. Two modes run Triton itself, by design: Triton's
compile-warmup test mode, and `intj.compat.launch_or_interpret`, which takes
the same arguments and calls `kernel[grid]` while Triton's interpreter is on.

Launchers from `make_launcher` never run Triton's launch hooks; leave
hook-sensitive launches on Triton.

## Not supported

Everything below is refused at `check()` (the first call runs it), or on the
first launch that hits it:

- Backends with no registered `intj.backend.Backend`. `HipBackend` and
  `CudaBackend` ship; any triton backend whose driver exposes a
  `cuLaunchKernel`-shaped entry point is supported by subclassing `Backend` (dylib,
  launch symbol, error-string convention, whether it specializes pointers on a 2 GiB
  range) and calling `register()`. Both shipped backends run the full GPU test
  suite: HIP on gfx942, CUDA on sm_89.
- `TRITON_INTERPRET=1` (a kernel decorated while it is on is an
  `InterpretedFunction`, which intj cannot analyze either). Turning it on after
  the build does not stop a built launcher.
- Unsupported parameter annotations, `*args`/`**kwargs`, keyword-only parameters.
- Tuple, `tl.constexpr` object and `TensorDescriptor` arguments, and `str` /
  `tl.dtype` / JIT-function values for anything but an untyped `tl.constexpr`.
- Tensor subclasses other than `torch.nn.Parameter` — the fast path gates on exact
  type, because a subclass can redefine what `data_ptr()` means.
- Kernels that read global variables (triton revalidates those on every launch;
  intj does not, so it refuses instead of silently launching a stale kernel),
  unless `assume_constant_globals=True` promises they never change.
- Triton's compile hooks `knobs.runtime.add_stages_inspection_hook`,
  `jit_cache_hook` and `jit_post_compile_hook`: intj neither keys its kernels
  on them nor calls them. Refused at `check()` and before every compile
  (`intj.compat.launch`: every call); a hook set later does not stop the
  launches of kernels already compiled.
- Kernels with pre-run hooks, `num_ctas > 1`, cooperative launches, or non-zero
  global/profile scratch.

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

## How a launch works

0. The first call builds the module and swaps the launcher's entry; later calls
   start at step 1.
1. Parse `device` and `stream` -- passed, never looked up: that lookup is
   where `JITFunction`'s launch overhead goes -- compute the selected grid, and
   return early on a zero-volume grid.
2. Decode each public or bound argument, checking annotations if requested, and
   pack the dynamic key fields. How tensor fields are read is `torch_access_mode`'s
   choice; no mode calls an `aoti_torch_*` shim. The direct `AUTO` path accumulates
   descriptor bytes in 32-bit registers.
3. Hash and look up the key in the launcher's own cache. A fixed-device handle
   with no dynamic key fields reads its one nullable kernel directly, without
   hashing or a map.
4. On a miss, call back into Python: build the annotated `CompilerInput`, compare
   it with any input this launcher previously recorded for that key, and, unless
   the launcher's Python compiler cache already holds that kernel, pass its
   `triton.compiler.ASTSource` (or `GluonASTSource`) and canonical options to
   `triton.compiler.compile`.
   `_init_handles` loads the GPU module; the entry records the function handle,
   block dim and LDS size. `device` must be the device `stream` belongs to,
   and the current device the first time a specialization launches: Triton
   compiles and loads on the current device, so intj refuses a mismatch instead
   of launching in the wrong context.
5. Pack the param array (plus the two mandatory trailing scratch slots) and call
   `hipModuleLaunchKernel` / `cuLaunchKernel` (identical argument lists).

Each cached `CompiledKernel` is kept alive by the launcher's compiler cache (with
`return_compiled=True`, by the launcher records that return it), so its GPU
module stays loaded for as long as that owner lives.

## Correctness

The spec key packs only dynamic fields: type and applicable specialization
descriptors, constexpr payloads, and the device ordinal unless it is fixed.
Fields are laid out by descending alignment (8-, 4-, 2-, then 1-byte widths),
with stable declaration order among equal widths, and padded to a zeroed
64-bit word boundary. Baked values, bound values, and fixed facts contribute
no runtime field. The dtype index for pointers comes from the running torch
and Triton; `tl.int1` canonicalizes to `u1`. A fixed-device launcher with no
fields has an empty key (step 3 above).

The invariant intj must not break is:

> if two argument tuples produce the same intj key, their annotated compiler
> inputs have the same signature, constexpr values, and specialization attributes

A coarser key does not crash — it launches, say, a `tt.divisibility = 16` binary on an
unaligned pointer. `tests/test_launcher.py::test_spec_key_is_never_coarser_than_triton`
checks it directly through the launcher's `k.__self__.spec_key(device, *args)` debug
entry point, and every GPU cache miss rejects a key whose `CompilerInput` (the
final annotated signature, tagged constexpr values and attributes) differs from
the one recorded for it (step 4 above). Compilation uses that source directly,
without the JIT function's warmup, binder, or device cache.

That source is the one Triton's own `kernel[grid](...)` builds for the same
arguments, down to its hash: Triton hashes `str(attrs)`, and lists an empty
`(i,): []` entry for every constexpr and for every runtime integer or pointer
it specializes, `do_not_specialize` ones excepted. intj lists the same
entries, so a kernel launched through both compiles once and shares one
Triton cache entry. This holds for annotations whose facts are the ones
Triton derives (`NEVER` where it derives none, `Assume` of what it derives);
other facts are a different binary and hash. One case shares a binary but
not a hash: Triton lists no entry for a Triton-typed integer (`n: int`,
`n: tl.int32`) equal to 1 or a bool, and an empty one for other values not
divisible by 16. These compile alike, so intj keys them together, and the
first to miss decides which of Triton's hashes the binary has.
`tests/test_launcher.py::test_compiled_kernel_hash_matches_triton_launch`
and `test_ast_source_hashes_as_triton_launch` check this.

## Launch benchmark

```bash
PYTHONPATH=$PWD /tmp/gb2/bin/python benchmarks/bench_launch.py --no-gpu --iters 20000 --batches 7
PYTHONPATH=$PWD /tmp/gb2/bin/python benchmarks/bench_launch.py --iters 20000 --batches 7
```

The first command uses CPU tensors and intj's existing host compile state;
it measures decoding, key/cache work, and vectorcall overhead only. It does
not query a GPU target or driver, compile a GPU kernel, or launch one. The
second uses GPU tensors and includes driver launch overhead. Each row warms
its launcher, then reports the median nanoseconds per call across repeated
batches, absolute and percentage deltas against the same kernel's `auto map`
baseline, and separate factory construction and binding times in milliseconds.
The matrix covers reduced keys, checked and unchecked facts, baked values,
bound tensor and pointer handles, fixed-device map, and fixed-device no-map.
These are measurements, not pytest performance limits.

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
`RenderKey` dict (C code is shared; Python compiler caches are per launcher),
the `.so` on disk (loaded as-is, nothing rendered or compiled), and only then
renders and builds. Before compiler caches moved into launchers, a warm process
took ~0.2 ms, a warm disk ~0.6 ms, against ~250 ms for a build. Those timings
are untested with per-launcher compiler construction.

The digest covers the kernel source (`cache_key`), the parameter
table (including `do_not_specialize*`, which `cache_key` does not cover), the target,
the canonicalized options, intj's own Python sources and `runtime/` bytes, the triton
build, the compiler, `EXT_SUFFIX`, and for `STATIC_COMPILE` the full torch version.
A build compiles through triton's own cache, which keys on the source and the flags
as strings; a define hashing the runtime headers, the torch version and the triton
build keeps an in-place upgrade of torch or triton from getting the old object back.
`rm -rf $TRITON_HOME/.triton/intj` clears every intj artifact.

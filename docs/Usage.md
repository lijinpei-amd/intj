# INTJ usage

## `create_launcher`

```python
from intj import create_launcher

launcher = create_launcher(
    jit_func,              # a @triton.jit function
    dynamic_grid=False,    # reserved, must be False
    dynamic_options=(),    # reserved, must be empty
    extra_annotation=None, # reserved, must be None
    options=None,          # triton compile options, e.g. {"num_warps": 8}
)
```

`create_launcher` renders a C python extension for `jit_func`, compiles it (through
triton's build + on-disk cache, so it is free after the first time), loads it, and
returns its `entry` function. It is slow; call it once, outside any hot loop.

`options` are forwarded verbatim to `JITFunction.warmup` on every compile, so they
must be valid triton options (`num_warps`, `num_stages`, `waves_per_eu`, ...) and are
fixed for the lifetime of the launcher. `device`, `stream`, `device_type` and
`warp_size` are rejected, and an unknown option name is an error here rather than a
silent fall back to the default.

They are canonicalized through the compiler backend's `parse_options`, so spellings
that mean the same thing — `{}` and `{"num_warps": 4}` on AMD — share one rendered
module instead of building it twice.

Most things intj cannot handle raise `intj.launcher.UnsupportedKernel` here. The
rest — `num_ctas > 1`, a cooperative launch, a kernel needing scratch memory — can
only be seen once triton has compiled, so they raise on the first launch that misses
the cache. An unsupported *argument* raises `TypeError` at the launch that passes it.

## Calling the launcher

```python
launcher(device, stream, grid, arg0, arg1, ...)
```

| argument | type |
|---|---|
| `device` | device index, `int` (e.g. `torch.cuda.current_device()`) |
| `stream` | raw stream handle, `int` (e.g. `torch.cuda.current_stream().cuda_stream`) |
| `grid` | `int`, or a `tuple`/`list` of 1-3 ints |
| `arg0...` | the kernel parameters, **positionally, all of them, in declaration order** |

Returns `None`. A grid whose volume is zero returns without launching.

`device` must be the device `stream` belongs to, and it must be the current device
the first time a given specialization is launched — triton compiles and loads the
binary on the current device, so intj refuses a mismatch instead of launching a
function in the wrong context. Kernels compiled for different devices live under
different keys, so one launcher serves all of them.

Keyword arguments are not accepted, defaults are not filled in, and the launcher does
not read the current device or stream for you — that is where the launch overhead of
`JITFunction` goes.

## Supported arguments

| passed value | triton type | key |
|---|---|---|
| `torch.Tensor` / `torch.nn.Parameter` (exact type) | `*<dtype>` | `D` when the data pointer is 16B-aligned, `S` (AMD) when the storage is ≤ 2 GiB |
| `int` | `constexpr` when the value is 1, else `i32`/`i64`/`u64` | `D` when divisible by 16 |
| `float` | `fp32` | — |
| `bool` | `u1` | — |
| `None` | `constexpr` | — |
| `tl.constexpr` parameter | `constexpr` | the value itself (`int`, `float`, `bool`, `None`) |

`do_not_specialize` and `do_not_specialize_on_alignment` are honoured. The `S` bit is
AMD-only; there it is always part of the key, even when `knobs.amd.use_buffer_ops` is
off, so that toggling the knob can only cost an extra compile, never launch a
buffer-ops binary on a > 2 GiB tensor.

## Not supported

Everything below is refused at `create_launcher` time, or on the first launch that
hits it:

- Backends with no registered `intj.launcher.Backend`. `HipBackend` and
  `CudaBackend` ship; any triton backend whose driver exposes a
  `cuLaunchKernel`-shaped entry point is supported by subclassing `Backend` (dylib,
  launch symbol, error-string convention, whether it specializes pointers on a 2 GiB
  range) and calling `register()`. **NVIDIA is untested** -- there is no NVIDIA GPU
  on the development machine.
- `@triton.autotune` / `@triton.heuristics` wrappers, and `TRITON_INTERPRET=1`.
- Callable grids (`dynamic_grid`), per-launch options (`dynamic_options`), and
  argument annotations (`extra_annotation`).
- Non-constexpr parameter annotations, `*args`/`**kwargs`, keyword-only parameters.
- Tuple, `tl.constexpr` object, `TensorDescriptor`, JIT-function and string arguments.
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
accessors, and it is shared by the `shim` and `cxx` modes alike — they are kept
consistent on purpose, since the `shim` mode cannot see the policy bit at all. Neither
kind of tensor is a usable triton kernel argument in the first place.
- Kernels that read global variables (triton revalidates those on every launch;
  intj cannot, so it refuses instead of silently launching a stale kernel).
- Kernels with pre-run hooks, `num_ctas > 1`, cooperative launches, or non-zero
  global/profile scratch.

## Reaching torch: `torch_access`

Every launch reads three things off each tensor: the data pointer, the dtype, and (on
AMD) the storage size. `create_launcher(..., torch_access=...)` picks how, with
`intj.TorchAccess`:

| | how it reads | decode + key, 3 tensors | first build | rebuilt when torch changes |
|---|---|---|---|---|
| `SHIM` | `TensorImpl`/`StorageImpl` at offsets probed from the running torch | 89 ns | 0.6 s | no |
| `CXX` | the same fields, offsets supplied by the compiler | 90 ns | 1.9 s | yes |
| `CPYTHON` | `data_ptr()` / `untyped_storage().nbytes()` through the interpreter | 300 ns | 0.6 s | no |
| `AUTO` (default) | `CXX` if a C++ compiler and torch's headers are present, else `SHIM`, else `CPYTHON` | | | |

No mode dlopens `libtorch_cpu.so` or calls an `aoti_torch_*` shim.

`SHIM` reads offsets from a table of torch versions intj has been verified against
(`intj.torch_abi._LAYOUTS`), installed at load by `set_torch_version`. A torch with no
row is **refused, never guessed at** — a wrong offset cannot raise, it reads whatever
lies at that address and hands the kernel a pointer built from it. On an unverified
torch, `torch_access=TorchAccess.SHIM` raises and `AUTO` falls through to `CPYTHON`.

`CXX` needs a row too, but only to check itself: it compares its compiled-in
`offsetof(THPVariable, cdata)` against the table's, and refuses the module if they
disagree.

To add a version, run `python -m intj.torch_abi` on it and paste the line it prints.
That derives each offset by matching field values against what torch's own accessors
report, so a row produced that way is verified rather than reasoned about — and the
test suite re-checks the row against the torch it runs on.

`CPYTHON` assumes nothing about torch's layout except `THPDtype`, which checks itself at
load against the name the struct embeds. It is the independent oracle the test suite
compares the other two against.

## How a launch works

1. Parse `device`, `stream` and `grid`; return early on a zero-volume grid.
2. Decode every argument into one spec-key word and at most one param slot. Tensor
   fields are read through libtorch's `aoti_torch_*` C shims, not `data_ptr()`.
3. Hash the key and look it up in the module's open-addressed kernel cache.
4. On a miss, call back into python: `JITFunction.warmup` compiles, `_init_handles`
   loads the module, and the entry records the function handle, block dim and LDS
   size. The callback also asserts that intj's key agrees with triton's
   specialization for these arguments.
5. Pack the param array (plus the two mandatory trailing scratch slots) and call
   `hipModuleLaunchKernel` / `cuLaunchKernel` (identical argument lists).

Each cached `CompiledKernel` is kept alive by the launcher, so its GPU module stays
loaded for the lifetime of the process.

## Correctness

The invariant intj must not break is:

> if two argument tuples produce the same intj key, triton produces the same
> specialization for them

A coarser key does not crash — it launches, say, a `tt.divisibility = 16` binary on an
unaligned pointer. `tests/test_launcher.py::test_spec_key_is_never_coarser_than_triton`
checks it directly through the module's `spec_key(*args)` debug entry point, and every
cache miss re-checks it against triton's own binder.

## Cache

Artifacts land at

    $TRITON_HOME/.triton/intj/<digest>/<module>/<kernel><EXT_SUFFIX>
    $TRITON_HOME/.triton/intj/<digest>/<module>/<kernel>.c

next to triton's own caches. The rendered source is kept beside the binary: it is what
you read when a launch misbehaves. The symbol is the kernel's own name
(`PyInit_<kernel>`) — what `perf` and `/proc/<pid>/maps` show. Two builds of one kernel
are two directories and two independent modules.

Modules are loaded by hand, so none of this reaches `sys.modules`: the module's name
is just the kernel's, a label rather than a lookup key — two builds of one kernel share
it and are told apart by `__file__`.

`create_launcher` looks for its module in three places, in order: the process-level
`ModuleKey` dict (same module, so the same kernel cache), the `.so` on disk (loaded
as-is, nothing rendered or compiled), and only then renders and builds. A warm
process takes ~0.2 ms, a warm disk ~0.6 ms, against ~250 ms for a build.

The digest covers the kernel source (`cache_key`), the parameter
table (including `do_not_specialize*`, which `cache_key` does not cover), the target,
the canonicalized options, intj's own `runtime/` bytes, the triton build, the compiler
and the flags it is handed,
and `EXT_SUFFIX`. `rm -rf $TRITON_HOME/.triton/intj` clears every intj artifact.

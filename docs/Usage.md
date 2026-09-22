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
- Kernels that read global variables (triton revalidates those on every launch;
  intj cannot, so it refuses instead of silently launching a stale kernel).
- Kernels with pre-run hooks, `num_ctas > 1`, cooperative launches, or non-zero
  global/profile scratch.

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

    $TRITON_CACHE_DIR/intj/loaded_modules/<digest>/<module>/<kernel><EXT_SUFFIX>

so the file says where the kernel came from and which build it is, and the symbol is
the kernel's own name (`PyInit_<kernel>`) — what `perf` and `/proc/<pid>/maps` show.
Two builds of one kernel are two directories and two independent modules.

The digest covers the kernel source (`cache_key`), the parameter
table (including `do_not_specialize*`, which `cache_key` does not cover), the target,
the canonicalized options, intj's own `runtime/` bytes, the triton build, the compiler,
and `EXT_SUFFIX`. `rm -rf $TRITON_CACHE_DIR/intj` clears every intj artifact.

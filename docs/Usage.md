# INTJ usage

## `make_launcher`

```python
from intj import make_launcher

launcher = make_launcher(
    jit_func,              # a @triton.jit function
    dynamic_grid=False,    # reserved, must be False
    dynamic_options=(),    # reserved, must be empty
    extra_annotation=None, # reserved, must be None
    options=None,          # triton compile options, e.g. {"num_warps": 8}
)
```

`make_launcher` renders a C python extension for `jit_func`, compiles it (through
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
| `device` | device index, `int` in `[0, 256)` (e.g. `torch.cuda.current_device()`) |
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

Everything below is refused at `make_launcher` time, or on the first launch that
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
AMD) the storage size. `make_launcher(..., torch_access=...)` picks how, with
`intj.TorchAccess`:

| | how it reads | decode + key, 3 tensors | first build | rebuilt when torch changes |
|---|---|---|---|---|
| `SHIM` | `TensorImpl`/`StorageImpl` at offsets probed from the running torch | 89 ns | 0.6 s | no |
| `CXX` | the same fields, offsets supplied by the compiler | 90 ns | 1.9 s | yes |
| `CPYTHON` | `data_ptr()` / `untyped_storage().nbytes()` through the interpreter | 300 ns | 0.6 s | no |
| `AUTO` (default) | `CXX` if a C++ compiler and torch's headers are present, else `SHIM`, else `CPYTHON` | | | |

No mode dlopens `libtorch_cpu.so` or calls an `aoti_torch_*` shim.

`SHIM` reads offsets from a table of torch versions intj has been verified against
(`intj/torch_intf/torch_abi.toml`), installed at load by `set_torch_version`. A torch with no
entry is **refused, never guessed at** — a wrong offset cannot raise, it reads whatever
lies at that address and hands the kernel a pointer built from it. On an unverified
torch, `torch_access=TorchAccess.SHIM` raises and `AUTO` falls through to `CPYTHON`.

Each entry records the dtypes the running torch had when the offsets were measured
(`[name, element size]`, indexed by dtype code) alongside the offsets themselves. A
torch whose dtype set has drifted from its entry — one renamed, dropped or inserted — is treated as
unverified too, offsets and all: the same refusal, not a partial trust.

`CXX` needs no entry: the compiler supplies every offset from torch's headers. The
one thing it declares itself rather than includes is the head of `THPVariable`
(`intj/runtime/intj_thpvariable.h`: a `MaybeOwned<Tensor>` before torch 2.10, a
`Tensor` since); `cpp_detect` below refuses a torch where that declaration is wrong.
On torch older than 2.10 the `CXX` mode is compile-checked only.

To add a version, run `python -m intj.torch_intf.abi_detect` on it and paste the entry it
prints. That derives each offset by matching field values against what torch's own
accessors report, so an entry produced that way is verified rather than reasoned about.
`python -m intj.torch_intf.cpp_detect` derives the same offsets independently, by
compiling against torch's C++ headers. The test suite checks the entry for the torch
it runs on against both.

`CPYTHON` assumes nothing about torch's layout except `THPDtype`, which checks itself at
load against the name the struct embeds. It is the independent oracle the test suite
compares the other two against.

## The kernel cache: `kernel_cache`

Every launch turns the spec key into a compiled kernel through a hash map.
`make_launcher(..., kernel_cache=...)` picks which, with `intj.KernelCache`:

| | what it is | first use costs |
|---|---|---|
| `INTJ` (default) | intj's open-addressed table, 72 lines | nothing |
| `TSL` | `tsl::robin_map`, handed the precomputed hash | ~3 s, a download |
| `ABSL` | `absl::flat_hash_map` | a download, then a build of 90 libraries: 28 s on 224 cores, minutes on a few |

`TSL` and `ABSL` are C++ maps, so either one compiles the whole module as C++
even under `SHIM` or `CPYTHON` access.

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

1. Parse `device`, `stream` and `grid`; return early on a zero-volume grid.
2. Decode every argument into one spec-key *byte* and at most one param slot,
   accumulating the key's header in 32-bit registers. How the tensor fields are read
   is `torch_access`'s choice; no mode calls an `aoti_torch_*` shim.
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

The spec key is a flat byte array:

```
bytes 0 .. nparams-1    one code byte per declared parameter, in declaration order
byte nparams            the device ordinal, hence its [0, 256) bound
pad to a word boundary, zeroed
then                    one 64-bit value word per tl.constexpr parameter
```

A parameter's byte is `(dtype << 2) | divisibility << 1 | pointer_range` for a
pointer, which leaves `128..142` for the scalar and constexpr tags. `dtype` is a
5-bit index over triton's *element types*, not torch's `ScalarType` codes: `bool`,
`uint1` and `int1` all canonicalize to `u1`, so they share an index, and the key
ends up exactly as fine as triton's specialization. The table is built from the
running torch and triton and installed at load, never compiled in.

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

`make_launcher` looks for its module in three places, in order: the process-level
`ModuleKey` dict (same module, so the same kernel cache), the `.so` on disk (loaded
as-is, nothing rendered or compiled), and only then renders and builds. A warm
process takes ~0.2 ms, a warm disk ~0.6 ms, against ~250 ms for a build.

The digest covers the kernel source (`cache_key`), the parameter
table (including `do_not_specialize*`, which `cache_key` does not cover), the target,
the canonicalized options, intj's own `runtime/` bytes, the triton build, the compiler,
and `EXT_SUFFIX`. `rm -rf $TRITON_HOME/.triton/intj` clears every intj artifact.

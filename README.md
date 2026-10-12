# Incompatible Triton Jit

INTJ (INcompatible Triton Jit) is a host side python launcher for triton kernel, it will do the following:

- Compute triton kernel cache key.
- Lookup compiled kernel using the cache key. If missing, compile it.
- Launch the compiled kernel.

INTJ aims at reducing host launch overhead, from our benchmark, triton `JitFunction` has a launch overhead of ~10us, while INTJ has a launch overhead of ~0.3us (overhead defined as the time excluding cuLaunchKernel/hipLaunchKernel, which itself takes ~3us).

This is an early version with limited argument types and grid expressions. Unsupported cases are refused; see `docs/Usage.md`. AMD is tested on gfx942 and NVIDIA on sm_89 (Ada); both run the full GPU test suite.

The base package supports Python 3.8+; launching kernels requires Triton 3.7+.
Tested combinations:

| Python | Triton | Torch | Coverage |
|---|---|---|---|
| 3.12 | 3.8.0 | 2.14.0+rocm7.2 | Full GPU test suite on gfx942 |
| 3.12 | 3.7.1 | 2.14.0+rocm7.2 | Full GPU test suite on gfx942 |
| 3.12 | 3.8.0 | 2.14.1+cu130 | Full GPU test suite on sm_89 (RTX 4080 SUPER) |
| 3.10 | 3.7.1 | 2.10.0+cpu | Selected host/grid and native argument tests; GPU untested |

## Install

From this checkout, install the launcher on a CPython build with a compatible Triton wheel:

```sh
python -m pip install '.[launcher]'
```

On x86-64 Linux, Triton 3.8 wheels are available for CPython 3.10–3.14 with
the GIL and free-threaded 3.14, but not 3.8, 3.9, or free-threaded 3.13. On
those builds, `python -m pip install .` installs the base package. Torch may
bring in an older Triton as a dependency; that does not satisfy the launcher's
Triton 3.7 requirement.

### Development environment

`uv` builds one in `.venv` (git-ignored). On a CUDA machine the default PyPI
`torch` wheel for x86-64 Linux is already a CUDA build and pins a matching
Triton, so no extra index is needed:

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv torch triton pytest jinja2 numpy pyright

PYTHONPATH=$PWD .venv/bin/python -m pytest tests -q
.venv/bin/pyright --pythonpath .venv/bin/python
```

For a ROCm build, point `uv` at the ROCm index (`--torch-backend=rocm`, or
`--index`). If an HTTP(S) proxy is exported, exclude `pypi.org` and
`files.pythonhosted.org` from it — relaying the multi-GB CUDA wheels through a
proxy is what makes this install slow, not `uv`.

## Usage

```python
import functools

import torch
import triton
import triton.language as tl
from intj import make_launcher

# line #1: records the kernel and the options; does not query the GPU.
@make_launcher(options={"num_warps": 8})
@triton.jit
def my_kernel(x, y, o, n, BLOCK: tl.constexpr):
    ...

# line #2: build on the first call; subsequent calls use the native entry.
my_kernel(
    torch.cuda.current_device(),
    torch.cuda.current_stream().cuda_stream,
    (triton.cdiv(n, 128),),
    x, y, o, n, 128,
)
```

At line `#1`, the decorator only records the kernel and its options — nothing
touches the GPU, so a module-level `@make_launcher` imports on a machine with
no visible device. On the first call at line `#2`, INTJ builds and loads the C
extension and launches the kernel. Later calls use the native entry.

`options` is evaluated at declaration, so an option that itself needs the GPU
(selecting `num_warps` by backend, say) needs a factory that runs on first use:

```python
@functools.cache
def my_kernel():
    backend = triton.runtime.driver.active.get_current_target().backend
    return make_launcher(
        _my_kernel, options={"num_warps": {"hip": 4, "cuda": 8}[backend]}
    )

my_kernel()(device, stream, (triton.cdiv(n, 128),), x, y, o, n, 128)
```

Deferring `make_launcher`'s own arguments to the first call, so the decorator
form works here too, is a planned option (`delayed=`) and is **not implemented**;
see `TODO.md`.

To compute the grid in the extension from launch values, use an annotated
Python `def` with `grid_cpp`:

```python
def grid(n: int, BLOCK: int):
    return (triton.cdiv(n, BLOCK),)

@make_launcher(grid_cpp=grid)
@triton.jit
def my_kernel(x, y, o, n, BLOCK: tl.constexpr):
    ...

my_kernel(
    torch.cuda.current_device(),
    torch.cuda.current_stream().cuda_stream,
    x, y, o, n, 128,
)
```

`grid_arg=1`, `2`, or `3` accepts that many dimensions before the kernel
arguments. `grid_py=lambda meta: ...` calls Python on every launch with a dict
of the kernel's named arguments. Omitting all three options keeps the original
single-grid-object call shown above. See [the usage guide](docs/Usage.md) for
the supported native expressions and exact signatures.

## How does it work?

INTJ is faster than upstream `JitFunction` for two reasons:

- Delegate to C++ ASAP.
- Changing some of the user facing interfacr of `JitFunction`. (Thus the name INcompatible)


### Delegate to C++ ASAP

When studying the host launch overhead of triton `JitFunction`, we found that the source of overhead is long tail, not having some low hanging fruit to optimize. Also,the things `JitFunction` does is quite clear to me:

```
key = hash(source, args, options, knobs)
kernel = cache.get_or_create(key, lambda: compile(source, args, options, knobs)
grid = grid_func(args)
launch(kernel, device, stream, grid, args)
```

So I decide to reimplemen the above using C++.

### Not feature-complete as `JitFunction`

Some of the user facing features of triton `JitFunction` incurs launch overhead, which can be avoid if the user interface is changed. For example:

- AMDGPU 'S' spec-key can depends on per object "ptr_range".
- Arbitrary Python grid functions cannot be compiled; `grid_py` calls them on
  every launch, while `grid_cpp` compiles a restricted annotated `def`.
- Polimophic type.

Perhaps some of the feature may not contribute much to the overhead, but, I don't have time to gauge and give verdict to each one, and 
 the absolute zero overhead case is clear: "don't use them".

## Benchmark results

`taskset -c 0 python benchmarks/bench_launch.py --readme --iters 20000 --batches 7`,
5-argument kernel, MI308X (gfx942), triton 3.8.0, torch 2.14+rocm7.2. The
numbers below are the median of five fresh-cache runs; each run reports the
median of seven 20k-call batches. Each process used a separate empty
`TRITON_HOME` for the cold-build column.

| | triton `JitFunction` | INTJ |
|---|---|---|
| `grid=(1,)`, i.e. including `hipModuleLaunchKernel` (~3.0 us) | 17.42 us | 3.20 us |
| `grid=(0,)`, i.e. no driver call | 13.47 us | 0.15 us |
| argument decoding + spec key only | — | 0.12 us |

The same command on an RTX 4080 SUPER (sm_89), triton 3.8.0, torch 2.14.1+cu130,
CPython 3.12, same methodology:

| | triton `JitFunction` | INTJ |
|---|---|---|
| `grid=(1,)`, i.e. including `cuLaunchKernel` (~1.2 us) | 7.67 us | 1.34 us |
| `grid=(0,)`, i.e. no driver call | 6.13 us | 0.03 us |
| argument decoding + spec key only | — | 0.09 us |

Both backends leave INTJ with ~0.1 us of host overhead; the `grid=(1,)` rows
differ mainly because `cuLaunchKernel` is cheaper here than
`hipModuleLaunchKernel` was on the MI308X, and triton's own host path is
cheaper on a desktop CPU.

Argument decoding depends on how the module reads a tensor (`torch_access_mode`), for a
kernel with three tensor arguments. These access-mode measurements came from a
separate 20k-iteration MI300X run:

| `TorchAccessMode` | decode + spec key | first build |
|---|---|---|
| `RUNTIME_SHIM` — torch's structs, at recorded offsets selected at load | 89 ns | 0.6 s |
| `STATIC_COMPILE` — compiled against torch's headers | 90 ns | 1.9 s |
| `INTERPRETER` — pointer and storage size through the interpreter | 300 ns | 0.6 s |

and on the RTX 4080 SUPER, median of the same five fresh-cache runs:

| `TorchAccessMode` | decode + spec key | first build |
|---|---|---|
| `RUNTIME_SHIM` | 86 ns | 1.1 s |
| `STATIC_COMPILE` | 84 ns | 1.4 s |
| `INTERPRETER` | 321 ns | 0.6 s |

Omitting `torch_access_mode` selects `STATIC_COMPILE` when its toolchain is
available, else `RUNTIME_SHIM` when a verified layout exists, else `INTERPRETER`.

`RUNTIME_SHIM` can reuse its `.so` across supported PyTorch versions, but not
across CPython ABIs: the module is compiled and cached for the exact CPython
version and GIL/free-threaded build.

`RUNTIME_SHIM` and `STATIC_COMPILE` make the same loads and land within ~1 ns of
each other; `STATIC_COMPILE` has the compiler supply the field offsets that
`RUNTIME_SHIM` gets from the verified table. It pays a little build time for
that and is the only mode whose `.so` must be rebuilt when torch is upgraded.

The zero-volume row flatters INTJ a little: it returns before decoding arguments,
which is the third row's 0.12 us. Host overhead is therefore ~0.3 us against
~14 us.

The sm_89 numbers and their raw per-run samples are in the
[2026-10-12 CUDA port journal](benchmarks/journals/2026-10-12_cuda-port_0_9682f5c.md).

Run `python benchmarks/bench_launch.py --sweep --iters 100000 --batches 9` for
host-only timings with 4, 16, and 32 integer or tensor arguments. See the
[2026-09-24 optimization measurements](benchmarks/journals/2026-09-24_launcher-optimization_0_4f0b13e.md) for before/after results.

For TVM FFI call, callback, utility, and HIP launch timings, see the
[2026-09-25 script comparison](benchmarks/journals/2026-09-25_tvm-ffi-scripts_0_90b2959.md).

The [same-HSACO HIP launch and Python utility diagnostics](benchmarks/journals/2026-09-25_hip-utilities_0_660d458.md)
compare TVM FFI, Triton, and INTJ and explain the keyword/dataclass timing gaps.

TODO: sweep constexpr argument counts independently.

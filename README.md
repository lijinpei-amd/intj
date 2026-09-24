# Incompatible Triton Jit

INTJ (INcompatible Triton Jit) is a host side python launcher for triton kernel, it will do the following:

- Compute triton kernel cache key.
- Lookup compiled kernel using the cache key. If missing, compile it.
- Launch the compiled kernel.

INTJ aims at reducing host launch overhead, from our benchmark, triton `JitFunction` has a launch overhead of ~14us, while INTJ has a launch overhead of ~0.3us (overhead defined as the time excluding cuLaunchKernel/hipLaunchKernel).

This is the initial version: torch tensors only, static grid. Anything outside that is refused, see `docs/Usage.md`. AMD is tested on gfx942; the NVIDIA path is implemented (`cuLaunchKernel`) but untested, no NVIDIA GPU here.

## Install

From this checkout, install the launcher on a CPython build with a Triton 3.8 wheel:

```sh
python -m pip install '.[launcher]'
```

On x86-64 Linux, Triton 3.8 wheels are available for CPython 3.10–3.14 with
the GIL and free-threaded 3.14, but not 3.8, 3.9, or free-threaded 3.13. On
those builds, `python -m pip install .` installs the base package. Torch may
bring in an older Triton as a dependency; that does not satisfy the launcher's
Triton 3.8 requirement.

## Usage

```python
import torch, triton, triton.language as tl
from intj import make_launcher

@triton.jit
def my_kernel(x, y, o, n, BLOCK: tl.constexpr):
  ...

# line #1: create jit launcher.
# Speed doesn't matter much here.
launcher = make_launcher(my_kernel, options={"num_warps": 4})

# line #2: call the jit launcher.
# Calling the launcher should be fast.
launcher(torch.cuda.current_device(),
         torch.cuda.current_stream().cuda_stream,
         (triton.cdiv(n, 128),),
         x, y, o, n, 128)
```

For complete reference manual, see `docs/Usage.md`

## How does it work?

INTJ is faster than upstream `JitFunction` for two reasons:

- Delegate to C++ ASAP.
- Changing some of the user facing interfacr of `JitFunction`. (Thus the name INcompatible)


### Delegate to C++ ASAP

When studying the host launch overhead of triton `JitFunction`, we found that the source of overhead is long tail, not having some low hanging fruit to optimize. Also,the things `JitFunction` does is quite clear to me:

```
key = hash(source, args, options, knobs)
kernel = cache.get_or_create(key, lambda: compile(source, args, options, knobs)
launch(kernel, device, stream, args)
```

So I decide to reimplemen the above using C++.

In the usage example above, line `#1` will render a python C extension, and load it, line `#2` will call the C extension to do the 3 steps above.

### Not feature-complete as `JitFunction`

Some of the user facing features of triton `JitFunction` incurs launch overhead, which can be avoid if the user interface is changed. For example:

- AMDGPU 'S' spec-key can depends on per object "ptr_range".
- Grid can be a callable.
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

Argument decoding depends on how the module reads a tensor (`torch_access_mode`), for a
kernel with three tensor arguments. These access-mode measurements came from a
separate 20k-iteration MI300X run:

| `TorchAccessMode` | decode + spec key | first build |
|---|---|---|
| `RUNTIME_SHIM` — torch's structs, at recorded offsets selected at load | 89 ns | 0.6 s |
| `STATIC_COMPILE` — compiled against torch's headers | 90 ns | 1.9 s |
| `INTERPRETER` — pointer and storage size through the interpreter | 300 ns | 0.6 s |

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

Run `python benchmarks/bench_launch.py --sweep --iters 100000 --batches 9` for
host-only timings with 4, 16, and 32 integer or tensor arguments. See the
[2026-09-24 optimization measurements](docs/benchmark-2026-09-24.md) for before/after results.

TODO: sweep constexpr argument counts independently.

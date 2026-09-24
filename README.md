# Incompatible Triton Jit

INTJ (INcompatible Triton Jit) is a host side python launcher for triton kernel, it will do the following:

- Compute triton kernel cache key.
- Lookup compiled kernel using the cache key. If missing, compile it.
- Launch the compiled kernel.

INTJ aims at reducing host launch overhead, from our benchmark, triton `JitFunction` has a launch overhead of ~14us, while INTJ has a launch overhead of ~0.3us (overhead defined as the time excluding cuLaunchKernel/hipLaunchKernel).

This is the initial version: torch tensors only, static grid. Anything outside that is refused, see `docs/Usage.md`. AMD is tested on gfx942; the NVIDIA path is implemented (`cuLaunchKernel`) but untested, no NVIDIA GPU here.

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

Argument decoding depends on how the module reads a tensor (`torch_access`), for a
kernel with three tensor arguments:

| `TorchAccess` | decode + spec key | first build |
|---|---|---|
| `SHIM` — torch's structs, at offsets discovered at load | 121.5 ns | 1.07 s |
| `CXX` — compiled against torch's headers | 118.2 ns | 2.28 s |
| `CPYTHON` — through the interpreter | 915.4 ns | 0.74 s |

`SHIM` and `CXX` make the same loads and land within ~4 ns of each other; `CXX`
has the compiler supply the field offsets that `SHIM` probes for. `CXX` pays a little build time for that, and is the only mode whose `.so` must
be rebuilt when torch is upgraded.

The zero-volume row flatters INTJ a little: it returns before decoding arguments,
which is the third row's 0.12 us. Host overhead is therefore ~0.3 us against
~14 us.

Run `python benchmarks/bench_launch.py --sweep --iters 100000 --batches 9` for
host-only timings with 4, 16, and 32 integer or tensor arguments. See the
[2026-09-24 optimization measurements](benchmarks/journals/2026-09-24_4f0b13e_0.md) for before/after results.

TODO: sweep constexpr argument counts independently.

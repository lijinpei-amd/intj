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
from intj import create_launcher

@triton.jit
def my_kernel(x, y, o, n, BLOCK: tl.constexpr):
  ...

# line #1: create jit launcher.
# Speed doesn't matter much here.
launcher = create_launcher(my_kernel, options={"num_warps": 4})

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

`python benchmarks/bench_launch.py`, 5-argument kernel, MI300X (gfx942), triton 3.8.0,
torch 2.14+rocm7.2, 20k iterations:

| | triton `JitFunction` | INTJ |
|---|---|---|
| `grid=(1,)`, i.e. including `hipModuleLaunchKernel` (~3.4 us) | 18.7 us | 3.8 us |
| `grid=(0,)`, i.e. no driver call | 14.0 us | 0.15 us |
| argument decoding + spec key only | — | 0.11 us |

Argument decoding depends on how the module reads a tensor (`torch_access`), for a
kernel with three tensor arguments:

| `TorchAccess` | decode + spec key | first build |
|---|---|---|
| `SHIM` — torch's structs, at offsets discovered at load | 89 ns | 0.6 s |
| `CXX` — compiled against torch's headers | 90 ns | 11.5 s |
| `CPYTHON` — through the interpreter | 300 ns | 0.6 s |

`SHIM` and `CXX` make the same loads and land within ~1 ns of each other; `CXX`
has the compiler supply the field offsets that `SHIM` probes for. `CXX` pays for
that in build time, and is the only mode whose `.so` must be rebuilt when torch
is upgraded.

The zero-volume row flatters INTJ a little: it returns before decoding arguments,
which is the third row's 0.15 us. Host overhead is therefore ~0.3 us against
~14 us.

TODO: sweep the number of dynamic/constexpr arguments and the number of tensors.

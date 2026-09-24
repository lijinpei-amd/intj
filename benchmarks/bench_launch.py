# pyright: standard
"""Per-launch host overhead: intj vs triton's JITFunction.

Usage: python benchmarks/bench_launch.py [iters]

Reports wall-clock per launch for a trivial kernel:

- `grid=(1,)`: the real path, `hipModuleLaunchKernel` (~3.5 us here) included in
  both numbers.
- `grid=(0,)`: both launchers skip the driver call. intj also returns before
  decoding arguments, so this row is its early-out path, not its full host cost.
- `spec_key`: intj's argument decoding plus key computation, i.e. the work the
  zero-volume row skips.  Reported per `TorchAccessMode`, since the mode only
  changes how a tensor argument is read -- everything else is identical.
"""

from __future__ import annotations

import sys
import time

import torch
import triton
import triton.language as tl

from intj import TorchAccessMode, make_launcher


@triton.jit
def noop(x, y, o, n, a, BLOCK: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = off < n
    tl.store(o + off, tl.load(x + off, mask=mask) + tl.load(y + off, mask=mask) * a, mask=mask)


def bench(fn, iters):
    for _ in range(100):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters * 1e6


def main(iters=20000):
    n = 4096
    x = torch.randn(n, device="cuda")
    y = torch.randn(n, device="cuda")
    o = torch.empty(n, device="cuda")
    args = (x, y, o, n, 1.5, 128)

    launcher = make_launcher(noop)
    device = torch.cuda.current_device()
    stream = torch.cuda.current_stream().cuda_stream

    for label, grid in (("grid=(1,)", (1,)), ("grid=(0,)", (0,))):
        # BLOCK is a tl.constexpr parameter; triton takes the plain int at runtime
        triton_us = bench(lambda: noop[grid](*args), iters)  # pyright: ignore[reportArgumentType]
        intj_us = bench(lambda: launcher(device, stream, grid, *args), iters)
        print(f"{label:>10}: triton {triton_us:6.2f} us | intj {intj_us:6.2f} us | {triton_us / intj_us:5.1f}x")

    print()
    for mode in (TorchAccessMode.RUNTIME_SHIM, TorchAccessMode.STATIC_COMPILE,
                 TorchAccessMode.INTERPRETER):
        built = time.perf_counter()
        module = getattr(make_launcher(noop, torch_access_mode=mode), "__self__")
        built = time.perf_counter() - built
        spec_key = module.spec_key
        decode = bench(lambda: spec_key(*args), iters)
        print(f"{'spec_key':>10} {mode.name.lower():>14}: {decode:6.2f} us "
              f"(decode + key, 3 tensor args) | build {built:5.2f} s")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 20000)

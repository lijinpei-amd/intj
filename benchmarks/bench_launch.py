# pyright: standard
"""Repeated per-launch timings for argument annotations.

Usage: python benchmarks/bench_launch.py [--no-gpu] [--iters N] [--batches N]
"""

import argparse
import statistics
import time

import torch
import triton
import triton.language as tl

from intj import Annotation, Argument, Assume, Aligned, BindValue, Constexpr, NEVER, PointerRange, make_launcher


@triton.jit
def noop(x, y, o, n, a, BLOCK: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = off < n
    tl.store(o + off, tl.load(x + off, mask=mask) + tl.load(y + off, mask=mask) * a, mask=mask)


def bench(fn, iters, batches, sync):
    for _ in range(100):
        fn()
    sync()
    samples = []
    for _ in range(batches):
        start = time.perf_counter_ns()
        for _ in range(iters):
            fn()
        sync()
        samples.append((time.perf_counter_ns() - start) / iters)
    return statistics.median(samples)


def main(iters=20000, batches=7, no_gpu=False):
    n = 4096
    device = "cpu" if no_gpu else "cuda"
    x = torch.randn(n, device=device)
    y = torch.randn(n, device=device)
    o = torch.empty(n, device=device)
    args = (x, y, o, n, 1.5, 128)
    ordinal = 0 if no_gpu else torch.cuda.current_device()
    stream = 0 if no_gpu else torch.cuda.current_stream().cuda_stream
    sync = (lambda: None) if no_gpu else torch.cuda.synchronize
    pointer = tl.pointer_type(tl.float32)
    fixed: dict[str, Annotation] = {
        name: Argument(type=pointer, specialize=Assume(Aligned(16), PointerRange(32)))
        for name in ("x", "y", "o")
    }
    fixed.update(n=Argument(type=tl.int32, specialize=NEVER),
                 a=Argument(type=tl.float32, specialize=NEVER),
                 BLOCK=Constexpr(type=tl.int32))
    reduced: dict[str, Annotation] = {
        name: Argument(type=pointer, specialize=NEVER) for name in ("x", "y", "o")
    }
    reduced.update(n=fixed["n"], a=fixed["a"], BLOCK=fixed["BLOCK"])

    print(f"mode={'host-only' if no_gpu else 'gpu'}; {iters} calls × {batches} batches; median ns/call")
    print(f"{'path':>19} {'ns/call':>10} {'Δ ns':>10} {'Δ %':>9} {'build ms':>10} {'bind ms':>9}")

    def row(label, annotation=None, *, verify=False, binding=None, fixed_device=False,
            baked_n=False, baked_block=False):
        start = time.perf_counter_ns()
        factory = make_launcher(noop, extra_annotation=annotation, no_gpu=no_gpu,
                                verify_annotation=verify, bind_device=fixed_device)
        build_ms = (time.perf_counter_ns() - start) / 1e6
        bind_ms = None
        if binding is not None or fixed_device:
            start = time.perf_counter_ns()
            values = {"x": x if binding is BindValue.TENSOR else x.data_ptr()} if binding else {}
            launcher = (factory.bind_device(ordinal, **values) if fixed_device
                        else factory.bind(**values))
            bind_ms = (time.perf_counter_ns() - start) / 1e6
        else:
            launcher = factory
        call_args = args[1:] if binding else args
        if baked_n:
            call_args = call_args[:2 if binding else 3] + call_args[3 if binding else 4:]
        if baked_block:
            call_args = call_args[:-1]
        controls = (stream, (1,)) if fixed_device else (ordinal, stream, (1,))
        elapsed = bench(lambda: launcher(*controls, *call_args), iters, batches, sync)
        baseline = elapsed if label == "auto map" else auto_ns
        delta = elapsed - baseline
        binding_time = f"{bind_ms:.2f}" if bind_ms is not None else "-"
        print(f"{label:>19} {elapsed:10.1f} {delta:+10.1f} "
              f"{delta / baseline * 100:+9.1f} {build_ms:10.2f} {binding_time:>9}")
        return elapsed

    auto_ns = row("auto map")
    row("reduced key", reduced)
    row("verify off", fixed)
    row("verify on", fixed, verify=True)
    row("baked", {**fixed, "n": Argument(type=tl.int32, specialize=NEVER, value=n),
                  "BLOCK": Constexpr(type=tl.int32, value=128)}, baked_n=True, baked_block=True)
    row("bound tensor", {**fixed, "x": Argument(type=pointer, specialize=NEVER,
                                                bind_value=BindValue.TENSOR)}, binding=BindValue.TENSOR)
    row("bound pointer", {**fixed, "x": Argument(type=pointer, specialize=NEVER,
                                                 bind_value=BindValue.POINTER)}, binding=BindValue.POINTER)
    row("fixed device map", fixed_device=True)
    row("fixed device no-map", {**fixed, "BLOCK": Constexpr(type=tl.int32, value=128)},
        fixed_device=True, baked_block=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-gpu", action="store_true", help="decode and cache on the host only")
    parser.add_argument("--iters", type=int, default=20000)
    parser.add_argument("--batches", type=int, default=7)
    options = parser.parse_args()
    if options.iters < 1 or options.batches < 1:
        parser.error("--iters and --batches must be positive")
    main(options.iters, options.batches, options.no_gpu)

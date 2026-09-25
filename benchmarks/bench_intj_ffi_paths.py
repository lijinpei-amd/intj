# pyright: standard
"""INTJ counterparts for TVM FFI's callback, kwargs, and dataclass scripts.

Run from the INTJ root with PYTHONPATH=. python benchmarks/bench_intj_ffi_paths.py
[--iters N] [--batches N]. Callback results measure *cold cache misses*;
all other rows measure warm host-only calls. No GPU kernel is launched.
"""

import argparse
import dataclasses
import statistics
import time
from collections.abc import Callable

import torch
import triton
import triton.language as tl
from tvm_ffi.utils.kwargs_wrapper import make_kwargs_wrapper
from tvm_ffi.utils.unpack_dataclass import unpack_dataclass_to_tuple

from intj import make_launcher


@triton.jit
def three(x, y, z):
    pass


@triton.jit
def two(x, y):
    pass


@triton.jit
def callback_kernel(x, y, z, variant: tl.constexpr):
    pass


@dataclasses.dataclass
class Config:
    x: int
    y: int
    z: int


@dataclasses.dataclass
class Pair:
    x: int
    y: int


def time_calls(fn: Callable[..., object], iters: int, batches: int, args: tuple = ()) -> list[float]:
    for _ in range(100):
        fn(*args)
    samples = []
    for _ in range(batches):
        start = time.perf_counter_ns()
        for _ in range(iters):
            fn(*args)
        samples.append((time.perf_counter_ns() - start) / iters)
    return samples


def report(name: str, samples: list[float]) -> None:
    print(f"{name:<35} median_ns={statistics.median(samples):.1f} samples_ns={samples}")


def callback(iters: int, batches: int) -> None:
    tensors = tuple(torch.zeros(1, device="cuda") for _ in range(3))
    launch = make_launcher(callback_kernel, bind_device=True, no_gpu=True).bind_device(0)
    module = launch.__self__.__self__
    count = 0

    def compile_callback(_key, nparams, _device, *_args):
        nonlocal count
        count += 1
        return 0, 1, 0, nparams  # fake handle: no_gpu skips the driver launch

    module.set_compile_callback(compile_callback)
    for i in range(100):
        launch(0, 1, *tensors, -(i + 1))
    samples = []
    for batch in range(batches):
        start = time.perf_counter_ns()
        for i in range(batch * iters + 1, (batch + 1) * iters + 1):
            launch(0, 1, *tensors, i)
        samples.append((time.perf_counter_ns() - start) / iters)
    assert count == 100 + iters * batches, (count, iters, batches)
    print(f"callback_count={count} (100 warmup + {iters * batches} unique misses)")
    report("INTJ cold compile callback + cache", samples)
    report("INTJ hot call (no callback)", time_calls(
        launch, iters, batches, (0, 1, *tensors, 1)))
    noop = make_launcher(three, no_gpu=True)
    report("INTJ 3-tensor host-only nop", time_calls(
        noop, iters, batches, (0, 0, 1, *tensors)))


def kwargs(iters: int, batches: int) -> None:
    launch = make_launcher(three, no_gpu=True)
    launch(0, 0, 1, 1, 2, 3)

    def adapter(x: int, y: int = 2, z: int = 3) -> None:
        launch(0, 0, 1, x, y, z)

    ffi_wrapper = make_kwargs_wrapper(
        launch, ["device", "stream", "grid", "x", "y", "z"], arg_defaults=(2, 3))
    for label, fn in (
        ("INTJ direct positional", lambda: launch(0, 0, 1, 1, 2, 3)),
        ("INTJ adapter positional", lambda: adapter(1, 2, 3)),
        ("INTJ adapter kwargs", lambda: adapter(x=1, y=2, z=3)),
        ("INTJ adapter defaults", lambda: adapter(1)),
        ("INTJ FFI wrapper positional", lambda: ffi_wrapper(0, 0, 1, 1, 2, 3)),
        ("INTJ FFI wrapper kwargs", lambda: ffi_wrapper(0, 0, 1, x=1, y=2, z=3)),
    ):
        report(label, time_calls(fn, iters, batches))
    try:
        launch(0, 0, 1, x=1, y=2, z=3)
    except TypeError:
        print("INTJ native kwargs: unsupported (verified TypeError)")
    else:
        raise AssertionError("INTJ native kwargs unexpectedly accepted")


def dataclass(iters: int, batches: int) -> None:
    launch = make_launcher(two, no_gpu=True)
    pair = Pair(1, 2)
    cfg = Config(1, 2, 3)
    three_launch = make_launcher(three, no_gpu=True)
    assert unpack_dataclass_to_tuple(pair) == (1, 2)
    assert unpack_dataclass_to_tuple(cfg) == (1, 2, 3)
    fields = (pair.x, pair.y)
    for label, fn in (
        ("INTJ pair direct", lambda: launch(0, 0, 1, pair.x, pair.y)),
        ("INTJ pair prebuilt *tuple", lambda: launch(0, 0, 1, *fields)),
        ("FFI unpack Pair only", lambda: unpack_dataclass_to_tuple(pair)),
        ("INTJ pair manual unpack", lambda: launch(0, 0, 1, *(pair.x, pair.y))),
        ("INTJ pair FFI unpack", lambda: launch(0, 0, 1, *unpack_dataclass_to_tuple(pair))),
        ("INTJ pair stdlib astuple", lambda: launch(0, 0, 1, *dataclasses.astuple(pair))),
        ("INTJ config direct", lambda: three_launch(0, 0, 1, cfg.x, cfg.y, cfg.z)),
        ("INTJ config FFI unpack", lambda: three_launch(0, 0, 1, *unpack_dataclass_to_tuple(cfg))),
    ):
        report(label, time_calls(fn, iters, batches))
    try:
        three_launch(0, 0, 1, pair, 2, 3)
    except TypeError:
        print("INTJ native dataclass unpack: unsupported (verified TypeError)")
    else:
        raise AssertionError("INTJ dataclass unexpectedly accepted")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--batches", type=int, default=9)
    parser.add_argument("--mode", choices=("callback", "kwargs", "dataclass", "all"), default="all")
    args = parser.parse_args()
    if args.iters < 1 or args.batches < 1:
        parser.error("--iters and --batches must be positive")
    print(f"iters={args.iters} batches={args.batches} unit=ns/call; no_gpu=True")
    for name, fn in (("callback", callback), ("kwargs", kwargs), ("dataclass", dataclass)):
        if args.mode in ("all", name):
            print(f"mode={name}")
            fn(args.iters, args.batches)


if __name__ == "__main__":
    main()

# pyright: standard
"""Repeated per-launch timings for argument annotations and README comparisons.

Usage: python benchmarks/bench_launch.py [--no-gpu | --readme | --sweep | --last-key | --tuned | --dynamic] [--iters N] [--batches N]

The `--readme` rows report host time per launch for a trivial kernel:

- `grid=(1,)` includes the driver launch call (`hipModuleLaunchKernel` on HIP).
- `grid=(0,)` skips the driver call. intj also returns before decoding arguments,
  so this row measures its early-out path, not its full host cost.
- `spec_key` measures intj's argument decoding plus key computation, the work
  the zero-volume row skips. Its tensor access modes only change how tensor
  arguments are read.
"""

import argparse
import importlib.util
import pathlib
import statistics
import tempfile
import time

import torch
import triton
import triton.language as tl

from intj import (
    Annotation,
    Argument,
    Assume,
    Aligned,
    BindValue,
    Constexpr,
    NEVER,
    PointerRange,
    TorchAccessMode,
    make_launcher,
)
from intj.launcher import module_of


@triton.jit
def noop(x, y, o, n, a, BLOCK: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = off < n
    tl.store(
        o + off,
        tl.load(x + off, mask=mask) + tl.load(y + off, mask=mask) * a,
        mask=mask,
    )


@triton.jit
def act_noop(x, o, n, ACT: tl.constexpr):
    off = tl.arange(0, 128)
    v = tl.load(x + off, mask=off < n)
    if ACT == "relu":
        v = tl.maximum(v, 0.0)
    tl.store(o + off, v, mask=off < n)


def bench(fn, args, iters, batches, sync):
    """Time calls using a tuple prepared by the caller, without a wrapper lambda."""
    for _ in range(100):
        fn(*args)
    sync()
    samples = []
    for _ in range(batches):
        start = time.perf_counter_ns()
        for _ in range(iters):
            fn(*args)
        elapsed = time.perf_counter_ns() - start
        sync()
        samples.append(elapsed / iters)
    return statistics.median(samples)


def bench_pair(fn, first, second, iters, batches):
    """Time two direct calls per iteration with prepared argument tuples."""
    for _ in range(100):
        fn(*first)
        fn(*second)
    samples = []
    for _ in range(batches):
        start = time.perf_counter_ns()
        for _ in range(iters):
            fn(*first)
            fn(*second)
        samples.append((time.perf_counter_ns() - start) / (2 * iters))
    return statistics.median(samples)


def bench_readme(iters, batches):
    n = 4096
    x = torch.randn(n, device="cuda")
    y = torch.randn(n, device="cuda")
    o = torch.empty(n, device="cuda")
    args = (x, y, o, n, 1.5, 128)
    sync = torch.cuda.synchronize

    access_rows = []
    for mode in (
        TorchAccessMode.RUNTIME_SHIM,
        TorchAccessMode.STATIC_COMPILE,
        TorchAccessMode.INTERPRETER,
    ):
        start = time.perf_counter_ns()
        keyed = make_launcher(noop, torch_access_mode=mode)
        module = module_of(keyed)  # builds: make_launcher alone no longer does
        build_s = (time.perf_counter_ns() - start) / 1e9
        decode_ns = bench(module.spec_key, (keyed, 0, *args), iters, batches, sync)
        access_rows.append((mode.name.lower(), decode_ns, build_s))

    launcher = make_launcher(noop)
    device = torch.cuda.current_device()
    stream = torch.cuda.current_stream().cuda_stream
    print(f"mode=readme; {iters} calls × {batches} batches; median")
    print(f"{'path':>12} {'triton us':>11} {'intj us':>10} {'speedup':>9}")
    for label, grid in (("grid=(1,)", (1,)), ("grid=(0,)", (0,))):
        triton_ns = bench(noop[grid], args, iters, batches, sync)  # pyright: ignore[reportArgumentType]
        launch_args = (device, stream, grid) + args
        intj_ns = bench(launcher, launch_args, iters, batches, sync)
        print(
            f"{label:>12} {triton_ns / 1000:11.2f} {intj_ns / 1000:10.2f} "
            f"{triton_ns / intj_ns:8.1f}x"
        )

    print()
    print(f"{'torch_access_mode':>17} {'decode ns':>11} {'build s':>10}")
    for mode, decode_ns, build_s in access_rows:
        print(f"{mode:>17} {decode_ns:11.1f} {build_s:10.2f}")


def bench_sweep(iters, batches):
    print(f"mode=sweep; host-only; {iters} calls × {batches} batches; median ns/call")
    print(f"{'count':>5} {'kind':>6} {'ns/call':>10}")
    with tempfile.TemporaryDirectory() as tmp:
        for count in (4, 16, 32):
            path = pathlib.Path(tmp) / f"args_{count}.py"
            names = ", ".join(f"x{i}" for i in range(count))
            path.write_text(
                f"import triton\n\n@triton.jit\ndef noop({names}):\n    pass\n"
            )
            spec = importlib.util.spec_from_file_location(path.stem, path)
            assert spec is not None and spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)  # pyright: ignore[reportAttributeAccessIssue]  # 3.8 stubs lack it
            launcher = make_launcher(module.noop, no_gpu=True)
            for kind, value in (("int", 17), ("tensor", torch.empty(4096))):
                args = (0, 0, (1,)) + (value,) * count
                elapsed = bench(launcher, args, iters, batches, lambda: None)
                print(f"{count:5d} {kind:>6} {elapsed:10.1f}")


def bench_last_key(iters, batches):
    print(
        f"mode=last-key; host-only; {2 * iters} calls × {batches} batches; median ns/call"
    )
    print(f"{'count':>5} {'pattern':>11} {'ns/call':>10}")
    with tempfile.TemporaryDirectory() as tmp:
        for count in (4, 16, 32):
            path = pathlib.Path(tmp) / f"args_{count}.py"
            names = ", ".join(f"x{i}" for i in range(count))
            path.write_text(
                f"import triton\n\n@triton.jit\ndef noop({names}):\n    pass\n"
            )
            spec = importlib.util.spec_from_file_location(path.stem, path)
            assert spec is not None and spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)  # pyright: ignore[reportAttributeAccessIssue]  # 3.8 stubs lack it
            launcher = make_launcher(module.noop, no_gpu=True)
            first = (0, 0, (1,)) + (17,) * count
            second = first[:-1] + (16,)
            for pattern, other in (("repeat", first), ("alternate", second)):
                elapsed = bench_pair(launcher, first, other, iters, batches)
                print(f"{count:5d} {pattern:>11} {elapsed:10.1f}")


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
    fixed.update(
        n=Argument(type=tl.int32, specialize=NEVER),
        a=Argument(type=tl.float32, specialize=NEVER),
        BLOCK=Constexpr(type=tl.int32),
    )
    reduced: dict[str, Annotation] = {
        name: Argument(type=pointer, specialize=NEVER) for name in ("x", "y", "o")
    }
    reduced.update(n=fixed["n"], a=fixed["a"], BLOCK=fixed["BLOCK"])

    print(
        f"mode={'host-only' if no_gpu else 'gpu'}; {iters} calls × {batches} batches; median ns/call"
    )
    print(
        f"{'path':>19} {'ns/call':>10} {'Δ ns':>10} {'Δ %':>9} {'build ms':>10} {'bind ms':>9}"
    )

    def row(
        label,
        annotation=None,
        *,
        verify=False,
        binding=None,
        fixed_device=False,
        baked_n=False,
        baked_block=False,
    ):
        start = time.perf_counter_ns()
        factory = make_launcher(
            noop,
            extra_annotation=annotation,
            no_gpu=no_gpu,
            verify_annotation=verify,
            bind_device=fixed_device,
        )
        build_ms = (time.perf_counter_ns() - start) / 1e6
        bind_ms = None
        if binding is not None or fixed_device:
            start = time.perf_counter_ns()
            values = (
                {"x": x if binding is BindValue.TENSOR else x.data_ptr()}
                if binding
                else {}
            )
            launcher = (
                factory.bind_device(ordinal, **values)
                if fixed_device
                else factory.bind(**values)
            )
            bind_ms = (time.perf_counter_ns() - start) / 1e6
        else:
            launcher = factory
        start = time.perf_counter_ns()
        module_of(launcher)  # the build, which make_launcher leaves to the first call
        build_ms += (time.perf_counter_ns() - start) / 1e6
        call_args = args[1:] if binding else args
        if baked_n:
            call_args = (
                call_args[: 2 if binding else 3] + call_args[3 if binding else 4 :]
            )
        if baked_block:
            call_args = call_args[:-1]
        controls = (stream, (1,)) if fixed_device else (ordinal, stream, (1,))
        launch_args = controls + call_args
        elapsed = bench(launcher, launch_args, iters, batches, sync)
        baseline = elapsed if label == "auto map" else auto_ns
        delta = elapsed - baseline
        binding_time = f"{bind_ms:.2f}" if bind_ms is not None else "-"
        print(
            f"{label:>19} {elapsed:10.1f} {delta:+10.1f} "
            f"{delta / baseline * 100:+9.1f} {build_ms:10.2f} {binding_time:>9}"
        )
        return elapsed

    auto_ns = row("auto map")
    # `torch.Tensor` pointers decode tensor-only; same key as unannotated
    tensors: dict[str, object] = {name: torch.Tensor for name in ("x", "y", "o")}
    row("tensor annotated", tensors)
    typed_n: dict[str, object] = {"n": Argument(type=tl.int32)}  # the generic path
    row("generic", typed_n)
    row("generic tensor ann", {**typed_n, **tensors})
    row("reduced key", reduced)
    row("verify off", fixed)
    row("verify on", fixed, verify=True)
    row(
        "baked",
        {
            **fixed,
            "n": Argument(type=tl.int32, specialize=NEVER, value=n),
            "BLOCK": Constexpr(type=tl.int32, value=128),
        },
        baked_n=True,
        baked_block=True,
    )
    row(
        "bound tensor",
        {
            **fixed,
            "x": Argument(type=pointer, specialize=NEVER, bind_value=BindValue.TENSOR),
        },
        binding=BindValue.TENSOR,
    )
    row(
        "bound pointer",
        {
            **fixed,
            "x": Argument(type=pointer, specialize=NEVER, bind_value=BindValue.POINTER),
        },
        binding=BindValue.POINTER,
    )
    row("fixed device map", fixed_device=True)
    row(
        "fixed device no-map",
        {**fixed, "BLOCK": Constexpr(type=tl.int32, value=128)},
        fixed_device=True,
        baked_block=True,
    )


def bench_tuned(iters, batches):
    """A warmed autotuned launcher (hit path) against the same kernel with BLOCK baked."""
    x = torch.zeros(1024, device="cuda")
    device, stream = (
        torch.cuda.current_device(),
        torch.cuda.current_stream().cuda_stream,
    )

    def grid(n: int, BLOCK: int):
        return (triton.cdiv(n, BLOCK) * 0 + 1,)

    tuned = make_launcher(
        triton.autotune(
            configs=[triton.Config({"BLOCK": 64}), triton.Config({"BLOCK": 128})],
            key=["n"],
            do_bench=lambda call, quantiles: [call() or 1.0] * 3,
        )(noop),
        grid_cpp=grid,
    )
    plain = make_launcher(
        noop,
        extra_annotation={"BLOCK": Constexpr(type=tl.int32, value=64)},
        grid_cpp=grid,
    )
    args = (device, stream, x, x, x, 1024, 1.0)
    sync = torch.cuda.synchronize
    for name, fn in (("tuned hit", tuned), ("plain", plain)):
        print(f"{name:>10}: {bench(fn, args, iters, batches, sync):7.1f} ns/launch")


def bench_dynamic(iters, batches):
    """Warmed lazy launchers: plain, with two dynamic options, with a str
    constexpr (and an int one to compare it with).  `launch` includes the ~3 us
    driver call; `spec_key` is the host decode + key alone, at ns resolution."""
    x = torch.zeros(4096, device="cuda")
    device = torch.cuda.current_device()
    stream = torch.cuda.current_stream().cuda_stream
    args = (x, x, x, 4096, 1.0, 128)
    act = make_launcher(act_noop)
    rows = (
        ("lazy plain", make_launcher(noop), (), args),
        (
            "2 dynamic options",
            make_launcher(noop, dynamic_options=("num_warps", "knobs.runtime.debug")),
            (4, False),
            args,
        ),
        ("int constexpr", act, (), (x, x, 4096, 0)),
        ("str constexpr", act, (), (x, x, 4096, "relu")),
    )
    sync = torch.cuda.synchronize
    print(f"mode=dynamic; {iters} calls × {batches} batches; median ns/call")
    print(f"{'path':>19} {'launch':>10} {'spec_key':>10}")
    for label, fn, dyn, public in rows:
        launch_ns = bench(
            fn, (device, stream, (1,), *dyn, *public), iters, batches, sync
        )
        key_ns = bench(
            module_of(fn).spec_key, (fn, device, *dyn, *public), iters, batches, sync
        )
        print(f"{label:>19} {launch_ns:10.1f} {key_ns:10.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-gpu", action="store_true", help="decode and cache on the host only"
    )
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--readme", action="store_true", help="reproduce the README launcher comparison"
    )
    modes.add_argument(
        "--sweep", action="store_true", help="compare argument counts on the host only"
    )
    modes.add_argument(
        "--last-key",
        action="store_true",
        help="compare repeated and alternating keys on the host",
    )
    modes.add_argument(
        "--tuned",
        action="store_true",
        help="compare a warmed autotuned launcher with the same kernel baked",
    )
    modes.add_argument(
        "--dynamic",
        action="store_true",
        help="warmed lazy launchers: plain, 2 dynamic options, str constexpr",
    )
    parser.add_argument("--iters", type=int, default=20000)
    parser.add_argument("--batches", type=int, default=7)
    options = parser.parse_args()
    if options.iters < 1 or options.batches < 1:
        parser.error("--iters and --batches must be positive")
    if (options.readme or options.tuned or options.dynamic) and options.no_gpu:
        parser.error("--readme, --tuned and --dynamic need the GPU; drop --no-gpu")
    if options.readme:
        bench_readme(options.iters, options.batches)
    elif options.sweep:
        bench_sweep(options.iters, options.batches)
    elif options.last_key:
        bench_last_key(options.iters, options.batches)
    elif options.tuned:
        bench_tuned(options.iters, options.batches)
    elif options.dynamic:
        bench_dynamic(options.iters, options.batches)
    else:
        main(options.iters, options.batches, options.no_gpu)

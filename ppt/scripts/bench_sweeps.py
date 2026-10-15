# pyright: standard
"""Triton JITFunction vs intj launch time over argument, constexpr and cached-kernel counts.

usage: python ppt/scripts/bench_sweeps.py {args,constexpr,cache} [--max-kernels N]
           [--gpu-batch N] [--batches N] [--paths p,...]

Every kernel is an empty `@triton.jit` function written to a temp file. Each
point times four paths, one JSON line each:

- `triton_launch`: `kernel[(1,)](*args)`, GPU tensors, includes the driver call.
- `triton_host`: `kernel[(0,)](*args)`: Triton binds, keys, looks up and packs
  every argument, then its launcher skips the driver call on the empty grid.
- `intj_launch`: `launcher(device, stream, (1,), *args)`, includes the driver call.
- `intj_host`: a `no_gpu=True` launcher on CPU tensors: decode, key and cache
  lookup, no driver call. (intj's GPU launcher returns before decoding on an
  empty grid, so `(0,)` would flatter it.)

Sweeps:

- `args`: N in 0..64 parameters, all `int` (17) or all tensor.
- `constexpr`: 4 tensor parameters plus k `tl.constexpr` int parameters.
- `cache`: 4 tensor parameters plus `C: tl.constexpr`; each call picks one of K
  values of C uniformly at random (a fixed sequence of 4096 calls), so K
  kernels are cached and hit in random order.

A batch is a number of passes over a prebuilt list of argument tuples
(`for a in seq: fn(*a)`), timed with perf_counter_ns; the device is
synchronized after each batch, outside the timer. GPU-launch batches are kept
short (`--gpu-batch` calls) so the GPU queue never fills.
"""

import argparse
import importlib.util
import json
import os
import pathlib
import platform
import random
import statistics
import subprocess
import sys
import tempfile
import time

import torch
import triton

import intj
from intj import make_launcher
from intj.launcher import _resolve_torch_access_mode

ARG_COUNTS = (0, 1, 2, 4, 8, 16, 32, 64)
CONSTEXPR_COUNTS = (0, 1, 2, 4, 8, 16, 32)
KERNEL_COUNTS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)
SEQ_LEN = 4096
HOST_BATCH_NS = 4_000_000  # target length of a host-only batch
PATHS = ("triton_launch", "triton_host", "intj_launch", "intj_host")


def load_kernel(tmp, name, params):
    path = pathlib.Path(tmp) / f"{name}.py"
    path.write_text(
        "import triton\nimport triton.language as tl\n\n\n"
        f"@triton.jit\ndef {name}({', '.join(params)}):\n    pass\n"
    )
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # pyright: ignore[reportAttributeAccessIssue]
    return getattr(module, name)


def run_seq(fn, seq, reps):
    start = time.perf_counter_ns()
    for _ in range(reps):
        for a in seq:
            fn(*a)
    return time.perf_counter_ns() - start


def bench_seq(fn, seq, batches, sync, gpu_batch):
    """Returns per-call ns samples, one per batch, and the calls per batch.

    A GPU batch is the next `gpu_batch` calls of `seq`, cycling; a host batch
    is enough passes over `seq` to last ~HOST_BATCH_NS.
    """
    if gpu_batch:
        chunks = [
            seq[i : i + gpu_batch]
            for i in range(0, len(seq) - gpu_batch + 1, gpu_batch)
        ]
        for chunk in chunks:
            run_seq(fn, chunk, 1)
            sync()
        samples = []
        for b in range(batches):
            chunk = chunks[b % len(chunks)]
            elapsed = run_seq(fn, chunk, 1)
            sync()
            samples.append(elapsed / len(chunk))
        return samples, gpu_batch
    run_seq(fn, seq, 1)
    reps = max(1, round(HOST_BATCH_NS / max(run_seq(fn, seq, 1), 1)))
    samples = []
    for _ in range(batches):
        elapsed = run_seq(fn, seq, reps)
        sync()
        samples.append(elapsed / (reps * len(seq)))
    return samples, reps * len(seq)


def points(sweep, max_kernels):
    """Yields (x, kind, params, value lists per call) for one sweep."""
    gpu = torch.empty(4096, device="cuda")
    cpu = torch.empty(4096)
    if sweep == "args":
        for n in ARG_COUNTS:
            params = [f"x{i}" for i in range(n)]
            yield n, "int", params, ([17] * n, [17] * n)
            yield n, "tensor", params, ([gpu] * n, [cpu] * n)
    elif sweep == "constexpr":
        for k in CONSTEXPR_COUNTS:
            params = [f"t{i}" for i in range(4)] + [
                f"c{i}: tl.constexpr" for i in range(k)
            ]
            consts = [3 + i for i in range(k)]
            yield k, "int", params, ([gpu] * 4 + consts, [cpu] * 4 + consts)
    else:
        params = [f"t{i}" for i in range(4)] + ["C: tl.constexpr"]
        for k in (k for k in KERNEL_COUNTS if k <= max_kernels):
            yield k, "int", params, ([gpu] * 4, [cpu] * 4)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("sweep", choices=("args", "constexpr", "cache"))
    p.add_argument("--max-kernels", type=int, default=max(KERNEL_COUNTS))
    p.add_argument(
        "--gpu-batch", type=int, default=16, help="calls per GPU-launch batch"
    )
    p.add_argument("--gpu-batches", type=int, default=600)
    p.add_argument("--batches", type=int, default=25, help="host-only batches")
    p.add_argument("--paths", default=",".join(PATHS))
    p.add_argument("--only", default="", help="comma-separated x values to run")
    p.add_argument("--tag", default="")
    o = p.parse_args()
    paths = o.paths.split(",")

    device = torch.cuda.current_device()
    stream = torch.cuda.current_stream().cuda_stream
    sync = torch.cuda.synchronize
    commit = (
        subprocess.run(
            [
                "git",
                "-C",
                str(pathlib.Path(intj.__file__).parent),
                "rev-parse",
                "--short",
                "HEAD",
            ],
            capture_output=True,
            text=True,
        ).stdout.strip()
        or (pathlib.Path(intj.__file__).parent.parent / "COMMIT").read_text().strip()
    )
    print(
        json.dumps(
            {
                "header": True,
                "sweep": o.sweep,
                "tag": o.tag,
                "host": platform.node(),
                "cpu_affinity": sorted(os.sched_getaffinity(0)),
                "gpu": torch.cuda.get_device_name(device),
                "arch": getattr(
                    torch.cuda.get_device_properties(device), "gcnArchName", ""
                ),
                "hip_visible_devices": os.environ.get("HIP_VISIBLE_DEVICES"),
                "python": sys.version.split()[0],
                "torch": torch.__version__,
                "triton": triton.__version__,
                "intj": intj.__file__,
                "intj_commit": commit,
                "torch_access_mode": _resolve_torch_access_mode(
                    None, confirmed=True
                ).value,
                "gpu_batch": o.gpu_batch,
                "gpu_batches": o.gpu_batches,
                "host_batches": o.batches,
                "host_batch_ns": HOST_BATCH_NS,
                "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            }
        ),
        flush=True,
    )

    rng = random.Random(0)
    with tempfile.TemporaryDirectory() as tmp:
        for i, (x, kind, params, (gpu_vals, cpu_vals)) in enumerate(
            points(o.sweep, o.max_kernels)
        ):
            if o.only and str(x) not in o.only.split(",") and o.sweep != "cache":
                continue
            # the cache sweep reuses one kernel, so its caches grow across points
            name = f"k_{o.sweep}_{0 if o.sweep == 'cache' else i}"
            if o.sweep != "cache" or i == 0:
                kernel = load_kernel(tmp, name, params)
                launchers = {
                    "intj_launch": make_launcher(kernel),
                    "intj_host": make_launcher(kernel, no_gpu=True),
                }
            if o.sweep == "cache":
                picks = [rng.randrange(x) for _ in range(SEQ_LEN)]
                # every key once first, so the warm-up compiles all x kernels
                picks = list(range(x)) + picks
                gpu_seq = [tuple(gpu_vals) + (c,) for c in picks]
                cpu_seq = [tuple(cpu_vals) + (c,) for c in picks]
            else:
                gpu_seq = [tuple(gpu_vals)] * 256
                cpu_seq = [tuple(cpu_vals)] * 256
            cases = {
                "triton_launch": (kernel[(1,)], gpu_seq, o.gpu_batch),
                "triton_host": (kernel[(0,)], gpu_seq, 0),
                "intj_launch": (
                    launchers["intj_launch"],
                    [(device, stream, (1,)) + a for a in gpu_seq],
                    o.gpu_batch,
                ),
                "intj_host": (
                    launchers["intj_host"],
                    [(0, 0, (1,)) + a for a in cpu_seq],
                    0,
                ),
            }
            for path in paths:
                fn, seq, gpu_batch = cases[path]
                warm = seq[:x] if o.sweep == "cache" else seq[:1]
                for a in warm:
                    fn(*a)  # compiles (cache: every kernel)
                sync()
                timed = seq[x:] if o.sweep == "cache" else seq
                samples, calls = bench_seq(
                    fn,
                    timed,
                    o.gpu_batches if gpu_batch else o.batches,
                    sync,
                    gpu_batch,
                )
                print(
                    json.dumps(
                        {
                            "sweep": o.sweep,
                            "x": x,
                            "kind": kind,
                            "path": path,
                            "median_ns": round(statistics.median(samples), 2),
                            "calls_per_batch": calls,
                            "samples_ns": [round(s, 2) for s in samples],
                        }
                    ),
                    flush=True,
                )


if __name__ == "__main__":
    main()

# pyright: standard
"""Host cost of the Triton conveniences intj drops: `triton.cdiv` grids, `kernel[grid]`, ptr_range.

usage (from the intj root, PYTHONPATH=$PWD): python <this> [--number N] [--repeat N]

Each statement runs inline in a `timeit` loop (`number` calls per sample,
`repeat` samples, garbage collector off); a sample is ns per call minus the
same loop around `pass`. Prints one JSON line per statement with the median
and the sample spread. Kernel: `bench_launch.noop(x, y, o, n, a, BLOCK)`.

- launch syntax: `triton.cdiv` (a `ConstexprFunction`) against the same
  arithmetic in a plain `def`; intj with the grid passed in, against intj with
  `grid_cpp` computing `triton.cdiv` in C (both `no_gpu` launchers on CPU
  tensors: decode, key, lookup, no driver); `kernel[grid]`, and the call it
  returns against calling `run` directly, with `run` stubbed out so only the
  wrapper layer is timed.
- ptr_range (ROCm only, where Triton adds the `S` bit): Triton's per-tensor
  `HIPBackend.is_within_2gb` and `get_tensor_specialization` (which Triton's
  native specializer calls back into for every tensor argument), and their parts.
  intj reads the storage size in C; its whole per-tensor cost is the slope of
  `intj_host` in `data/sweep_args_wide.csv`.
"""

import argparse
import json
import pathlib
import statistics
import sys
import timeit

sys.path.insert(0, str(pathlib.Path.cwd() / "benchmarks"))

import torch  # noqa: E402
import triton  # noqa: E402

from bench_launch import noop  # noqa: E402  # pyright: ignore[reportMissingImports]  # benchmarks/ joins sys.path above
from intj import make_launcher  # noqa: E402


def grid(n: int, BLOCK: int):
    return (triton.cdiv(n, BLOCK),)


def cdiv(x, y):
    return (x + y - 1) // y


def stub_run(*_args, grid, warmup, **_kwargs):  # noqa: ARG001  # the signature run is called with
    return None


def main(number, repeat):
    hip = torch.version.hip is not None
    n, a, block = 4096, 1.5, 128
    cpu = [torch.empty(n) for _ in range(3)]
    gpu = [torch.empty(n, device="cuda") for _ in range(3)]
    dev = torch.cuda.current_device()
    stream = torch.cuda.current_stream().cuda_stream
    g = (triton.cdiv(n, block),)
    l_tuple = make_launcher(noop, no_gpu=True)
    l_cpp = make_launcher(noop, no_gpu=True, grid_cpp=grid)
    k = triton.jit(noop.fn)  # a fresh JITFunction whose run can be stubbed
    k.run = stub_run
    env: dict[str, object] = dict(
        triton=triton,
        cdiv=cdiv,
        n=n,
        a=a,
        block=block,
        dev=dev,
        stream=stream,
        g=g,
        l_tuple=l_tuple,
        l_cpp=l_cpp,
        k=k,
        x=cpu[0],
        y=cpu[1],
        o=cpu[2],
    )
    l_tuple(dev, stream, g, *cpu, n, a, block)
    l_cpp(dev, stream, *cpu, n, a, block)
    cases = [
        ("launch_syntax", "triton.cdiv", "triton.cdiv(n, block)"),
        ("launch_syntax", "plain def cdiv", "cdiv(n, block)"),
        ("launch_syntax", "inline (n + b - 1) // b", "(n + block - 1) // block"),
        (
            "launch_syntax",
            "intj, grid passed in",
            "l_tuple(dev, stream, g, x, y, o, n, a, block)",
        ),
        (
            "launch_syntax",
            "intj, grid_cpp in C",
            "l_cpp(dev, stream, x, y, o, n, a, block)",
        ),
        ("launch_syntax", "kernel[grid]", "k[g]"),
        (
            "launch_syntax",
            "kernel[grid](...), run stubbed",
            "k[g](x, y, o, n, a, BLOCK=block)",
        ),
        (
            "launch_syntax",
            "kernel.run(...) directly, stubbed",
            "k.run(x, y, o, n, a, BLOCK=block, grid=g, warmup=False)",
        ),
    ]
    if hip:
        from triton.backends.amd.compiler import (
            HIPBackend,
        )  # absent from a triton built without AMD

        env.update(HIPBackend=HIPBackend, t=gpu[0], torch=torch)
        cases += [
            (
                "ptr_range",
                "triton is_within_2gb(tensor)",
                "HIPBackend.is_within_2gb(t)",
            ),
            (
                "ptr_range",
                "triton get_tensor_specialization(tensor)",
                "HIPBackend.get_tensor_specialization(t, align=True)",
            ),
            ("ptr_range", "  hasattr(tensor, 'ptr_range')", "hasattr(t, 'ptr_range')"),
            (
                "ptr_range",
                "  tensor.untyped_storage().size()",
                "t.untyped_storage().size()",
            ),
            ("ptr_range", "  import torch", "import torch"),
        ]
    base = timeit.Timer("pass", globals=env)
    print(
        json.dumps(
            {
                "header": True,
                "python": sys.version.split()[0],
                "torch": torch.__version__,
                "triton": triton.__version__,
                "hip": hip,
                "number": number,
                "repeat": repeat,
                "gpu": torch.cuda.get_device_name(dev),
            }
        ),
        flush=True,
    )
    for group, name, stmt in cases:
        timer = timeit.Timer(stmt, globals=env)
        timer.timeit(number // 10 or 1)  # warm-up
        samples = []
        for _ in range(repeat):  # interleave the baseline with the statement
            empty = base.timeit(number) / number * 1e9
            samples.append(timer.timeit(number) / number * 1e9 - empty)
        print(
            json.dumps(
                {
                    "group": group,
                    "case": name,
                    "stmt": stmt,
                    "ns": round(statistics.median(samples), 2),
                    "lo": round(min(samples), 2),
                    "hi": round(max(samples), 2),
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--number", type=int, default=200000)
    p.add_argument("--repeat", type=int, default=9)
    o = p.parse_args()
    main(o.number, o.repeat)

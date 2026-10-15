# pyright: standard
"""`benchmarks/bench_launch.py --readme`, tolerant of an unavailable access mode.

usage (from the intj root): python <this> [--iters N] [--batches N]

Same kernel, timer and output as `bench_launch.bench_readme`, except that a
`TorchAccessMode` the machine cannot use (RUNTIME_SHIM on a torch with no
verified layout) prints `n/a <reason>` instead of aborting the run, and the
mode a launcher without `torch_access_mode` resolves to is printed first.
"""

import argparse
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path.cwd() / "benchmarks"))

import torch  # noqa: E402

from bench_launch import bench, noop  # noqa: E402
from intj import TorchAccessMode, make_launcher  # noqa: E402
from intj.launcher import _resolve_torch_access_mode, module_of  # noqa: E402


def main(iters, batches):
    n = 4096
    x = torch.randn(n, device="cuda")
    y = torch.randn(n, device="cuda")
    o = torch.empty(n, device="cuda")
    args = (x, y, o, n, 1.5, 128)
    sync = torch.cuda.synchronize
    default = _resolve_torch_access_mode(None, confirmed=True)
    print(f"default torch_access_mode={default.name.lower()}")

    access_rows = []
    for mode in (
        TorchAccessMode.RUNTIME_SHIM,
        TorchAccessMode.STATIC_COMPILE,
        TorchAccessMode.INTERPRETER,
    ):
        try:
            start = time.perf_counter_ns()
            keyed = make_launcher(noop, torch_access_mode=mode)
            module_of(keyed)
            build_s = (time.perf_counter_ns() - start) / 1e9
            decode_ns = bench(
                keyed.__self__._C_spec_key, (0, *args), iters, batches, sync
            )
            access_rows.append(
                (mode.name.lower(), f"{decode_ns:11.1f} {build_s:10.2f}")
            )
        except Exception as e:  # noqa: BLE001
            access_rows.append(
                (mode.name.lower(), f"n/a {type(e).__name__}: {e}"[:160])
            )

    launcher = make_launcher(noop)
    device = torch.cuda.current_device()
    stream = torch.cuda.current_stream().cuda_stream
    print(f"mode=readme; {iters} calls × {batches} batches; median")
    print(f"{'path':>12} {'triton us':>11} {'intj us':>10} {'speedup':>9}")
    for label, grid in (("grid=(1,)", (1,)), ("grid=(0,)", (0,))):
        triton_ns = bench(noop[grid], args, iters, batches, sync)
        intj_ns = bench(launcher, (device, stream, grid) + args, iters, batches, sync)
        print(
            f"{label:>12} {triton_ns / 1000:11.2f} {intj_ns / 1000:10.2f} "
            f"{triton_ns / intj_ns:8.1f}x"
        )
    print()
    print(f"{'torch_access_mode':>17} {'decode ns':>11} {'build s':>10}")
    for mode, text in access_rows:
        print(f"{mode:>17} {text}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--iters", type=int, default=20000)
    p.add_argument("--batches", type=int, default=7)
    o = p.parse_args()
    main(o.iters, o.batches)

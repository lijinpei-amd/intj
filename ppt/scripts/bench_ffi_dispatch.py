# pyright: standard
"""TVM FFI's per-argument dispatch-map lookup, light and heavy, through a real FFI call.

usage: python <this> [--number N] [--repeat N] [--types K]

Calls TVM FFI's `testing.nop` (a packed function that ignores its arguments) with
0, 1 and 4 ints and 4 CPU torch tensors; each argument goes through
`TVMFFIPyCallManager::SetArgument`, i.e. one `std::unordered_map<PyTypeObject*,
setter>::find` before its setter. Measured first with the map as a fresh
process leaves it (light), then after calls with K distinct `int` subclasses
have added K entries (heavy). Each statement runs inline in a `timeit` loop
minus the loop around `pass`; one JSON line per statement, median of `repeat`.
"""

import argparse
import contextlib
import io
import json
import statistics
import sys
import timeit

import torch
import tvm_ffi  # pyright: ignore[reportMissingImports]  # optional: apache-tvm-ffi, benchmarks only
import tvm_ffi.core  # pyright: ignore[reportMissingImports]


def map_size():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        tvm_ffi.core._print_debug_info()  # pyright: ignore[reportPrivateUsage]  # the only accessor
    return int(buf.getvalue().split(":")[1])


def main(number, repeat, k):
    nop = tvm_ffi.get_global_func("testing.nop")
    t = torch.empty(4)
    env = {"nop": nop, "i": 1, "t": t}
    cases = [
        ("0 args", "nop()"),
        ("1 int", "nop(i)"),
        ("4 ints", "nop(i, i, i, i)"),
        ("4 CPU tensors", "nop(t, t, t, t)"),
    ]
    base = timeit.Timer("pass", globals=env)
    print(
        json.dumps(
            {
                "header": True,
                "python": sys.version.split()[0],
                "tvm_ffi": tvm_ffi.__version__,
                "torch": torch.__version__,
                "number": number,
                "repeat": repeat,
            }
        ),
        flush=True,
    )
    for _, stmt in cases:  # every type this run uses enters the map before timing
        eval(stmt, env)  # noqa: S307
    for load in ("light", "heavy"):
        if load == "heavy":
            for j in range(k):
                nop(type(f"I{j}", (int,), {})(1))
        size = map_size()
        for name, stmt in cases:
            timer = timeit.Timer(stmt, globals=env)
            timer.timeit(number // 10 or 1)
            samples = []
            for _ in range(repeat):
                empty = base.timeit(number) / number * 1e9
                samples.append(timer.timeit(number) / number * 1e9 - empty)
            print(
                json.dumps(
                    {
                        "load": load,
                        "map_entries": size,
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
    p.add_argument("--types", type=int, default=1024)
    o = p.parse_args()
    main(o.number, o.repeat, o.types)

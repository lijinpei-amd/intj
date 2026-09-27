# pyright: standard
"""Host-only launch time with constexpr versus non-constexpr arguments.

Usage: python benchmarks/bench_mix.py [--quick] [--iters N] [--batches N]

Every row times a warmed `no_gpu=True` launcher, two direct calls per iteration
with prebuilt argument tuples, and reports the median ns/call over batches.

- `count`: kernel parameters (4, 16, 32).
- `share`: percentage of them declared `name: tl.constexpr` (0, 25, 50, 100),
  spread evenly over the parameter list.  Rows with the same count and
  kinds differ only in share, so `share=0` is the non-constexpr baseline.
- `ckind`: constexpr value (`int`, `str`, `dtype`); `-` when share is 0.
- `nkind`: non-constexpr values (`int`, CPU `tensor`, or `mixed`
  tensor/int/float); `-` when share is 100.
- `pattern`: `repeat` calls one key twice; `alternate` switches between two
  keys that differ in the last constexpr value, or, at share 0, in one
  specialization bit (int 17/16, tensor aligned/unaligned).

The script must run unmodified against every intj commit that has `no_gpu`:
it probes the API and prints `n/a <reason>` for rows a commit cannot run.
"""

import argparse
import importlib.util
import inspect
import pathlib
import statistics
import sys
import tempfile
import time

import torch
import triton
import triton.language as tl

COUNTS = (4, 16, 32)
SHARES = (0, 25, 50, 100)
CKINDS = ("int", "str", "dtype")
NKINDS = ("int", "tensor", "mixed")
CVALUES = {"int": (3, 5), "str": ("relu", "gelu"), "dtype": (tl.float16, tl.float32)}


def bench_pair(fn, first, second, iters, batches):
    """Two direct calls per iteration with prepared argument tuples."""
    for _ in range(200):
        fn(*first)
        fn(*second)
    samples = []
    for _ in range(batches):
        start = time.perf_counter_ns()
        for _ in range(iters):
            fn(*first)
            fn(*second)
        samples.append((time.perf_counter_ns() - start) / (2 * iters))
    return statistics.median(samples), samples


def constexpr_mask(count, share):
    k = count * share // 100
    return [(i * k) // count != ((i + 1) * k) // count for i in range(count)]


def load_kernel(tmp, count, share):
    mask = constexpr_mask(count, share)
    name = f"mix_{count}_{share}"
    params = ", ".join(
        f"x{i}: tl.constexpr" if c else f"x{i}" for i, c in enumerate(mask)
    )
    path = pathlib.Path(tmp) / f"{name}.py"
    path.write_text(
        "import triton\nimport triton.language as tl\n\n\n"
        f"@triton.jit\ndef {name}({params}):\n    pass\n"
    )
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # pyright: ignore[reportAttributeAccessIssue]
    return getattr(module, name), mask


def arg_pair(mask, ckind, nkind, tensor):
    """Two call tuples that are equal but for one key-relevant value."""
    first, second = [], []
    flip = None
    last_c = -1
    cycle = (tensor, 17, 1.5)
    n = 0
    for i, is_constexpr in enumerate(mask):
        if is_constexpr:
            first.append(CVALUES[ckind][0])
            second.append(CVALUES[ckind][1])
            last_c = i
            continue
        if nkind == "int":
            value = 17
        elif nkind == "tensor":
            value = tensor
        else:
            value = cycle[n % 3]
        n += 1
        first.append(value)
        second.append(value)
        if not isinstance(value, float):
            flip = i
    if last_c >= 0:
        second = first[:last_c] + [second[last_c]] + first[last_c + 1 :]
    else:
        assert flip is not None
        second = list(first)
        second[flip] = 16 if isinstance(first[flip], int) else tensor[1:]
    controls = (0, 0, (1,))
    return controls + tuple(first), controls + tuple(second)


def rows():
    for count in COUNTS:
        for share in SHARES:
            for ckind in CKINDS if share else ("-",):
                for nkind in NKINDS if share < 100 else ("-",):
                    yield count, share, ckind, nkind


def main(iters, batches):
    import intj

    make_launcher = getattr(intj, "make_launcher", None)
    print(
        f"intj {getattr(intj, '__version__', '?')} from {pathlib.Path(intj.__file__).parent}"
    )
    print(
        f"python {sys.version.split()[0]}; triton {triton.__version__}; torch {torch.__version__}"
    )
    print(f"mode=mix; host-only; {2 * iters} calls × {batches} batches; median ns/call")
    header = f"{'count':>5} {'share':>5} {'ckind':>5} {'nkind':>6} {'pattern':>9} {'ns/call':>9}  samples"
    print(header)
    if make_launcher is None:
        print("n/a all rows: intj has no make_launcher")
        return
    if "no_gpu" not in inspect.signature(make_launcher).parameters:
        print("n/a all rows: make_launcher has no no_gpu parameter")
        return
    tensor = torch.empty(4096)
    launchers = {}
    with tempfile.TemporaryDirectory() as tmp:
        for count, share, ckind, nkind in rows():
            prefix = f"{count:5d} {share:5d} {ckind:>5} {nkind:>6}"
            try:
                if (count, share) not in launchers:
                    jit_func, mask = load_kernel(tmp, count, share)
                    launchers[count, share] = (
                        make_launcher(jit_func, no_gpu=True),
                        mask,
                    )
                launcher, mask = launchers[count, share]
                first, second = arg_pair(mask, ckind, nkind, tensor)
                launcher(*first)
                launcher(*second)
            except Exception as e:  # a commit that cannot run this row
                reason = f"{type(e).__name__}: {e}".splitlines()[0][:100]
                for pattern in ("repeat", "alternate"):
                    print(f"{prefix} {pattern:>9} {'n/a':>9}  {reason}")
                sys.stdout.flush()
                continue
            for pattern, other in (("repeat", first), ("alternate", second)):
                median, samples = bench_pair(launcher, first, other, iters, batches)
                listed = " ".join(f"{s:.1f}" for s in samples)
                print(f"{prefix} {pattern:>9} {median:9.1f}  {listed}")
                sys.stdout.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iters", type=int, default=20000)
    parser.add_argument("--batches", type=int, default=9)
    parser.add_argument(
        "--quick", action="store_true", help="2000 calls × 5 batches: a smoke run"
    )
    options = parser.parse_args()
    if options.quick:
        options.iters, options.batches = 1000, 5
    if options.iters < 1 or options.batches < 1:
        parser.error("--iters and --batches must be positive")
    main(options.iters, options.batches)

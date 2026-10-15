# pyright: standard
"""Aggregates ppt/data/raw/<machine>/ into the CSVs under ppt/data/.

usage: python ppt/scripts/aggregate.py   (from the repository root)

Every number is the median over the process rounds (r0..r2) of each round's
own median; `lo`/`hi` columns are the smallest and largest round median.
"""

import csv
import json
import pathlib
import re
import statistics
from collections import defaultdict

DATA = pathlib.Path(__file__).resolve().parent.parent / "data"
RAW = DATA / "raw"
MACHINES = {
    "mi308x": "MI308X (gfx942)",
    "mi350x": "MI350X (gfx950)",
    "rtx4080": "RTX 4080 SUPER (sm_89)",
}
MEASURED = "measured 2026-10-15, 1000 calls x 9 batches, 3 processes"


def rounds(machine, pattern):
    return sorted((RAW / machine / "intj").glob(pattern))


def spread(values):
    return statistics.median(values), min(values), max(values)


def write(name, header, rows):
    with open(DATA / name, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"{name}: {len(rows)} rows")


def fmt(v, digits=3):
    return round(v, digits) if isinstance(v, float) else v


# ------------------------------------------------------------- intj benches


def headline():
    rows, access = [], []
    for m in MACHINES:
        launch = defaultdict(list)
        decode = defaultdict(list)
        default_mode = set()
        for f in rounds(m, "r*_readme.out"):
            for line in f.read_text().splitlines():
                if line.startswith("default torch_access_mode="):
                    default_mode.add(line.split("=", 1)[1])
                g = re.match(r"\s*(grid=\(\d,\))\s+([\d.]+)\s+([\d.]+)", line)
                if g:
                    launch[g[1], "triton"].append(float(g[2]))
                    launch[g[1], "intj"].append(float(g[3]))
                a = re.match(
                    r"\s*(runtime_shim|static_compile|interpreter)\s+(.*)", line
                )
                if a:
                    parts = a[2].split()
                    if parts[0] != "n/a":
                        decode[a[1]].append(float(parts[0]))
                    else:
                        decode.setdefault(a[1], [])
        for grid in ("grid=(1,)", "grid=(0,)"):
            t, i = launch[grid, "triton"], launch[grid, "intj"]
            if t:
                tm, ti = statistics.median(t), statistics.median(i)
                rows.append(
                    [
                        MACHINES[m],
                        grid,
                        fmt(tm),
                        fmt(ti),
                        fmt(tm / ti, 1),
                        len(t),
                        MEASURED,
                    ]
                )
        for mode, vals in decode.items():
            access.append(
                [
                    MACHINES[m],
                    mode,
                    fmt(statistics.median(vals), 1) if vals else "n/a",
                    mode in default_mode,
                    len(vals),
                    MEASURED,
                ]
            )
    write(
        "headline_launch.csv",
        ["machine", "grid", "triton_us", "intj_us", "speedup", "rounds", "source"],
        rows,
    )
    write(
        "headline_decode_by_access_mode.csv",
        [
            "machine",
            "torch_access_mode",
            "decode_spec_key_ns",
            "is_default",
            "rounds",
            "source",
        ],
        access,
    )


def sweep(kind):
    acc = defaultdict(list)
    for m in MACHINES:
        for f in rounds(m, f"r*_sweep_{kind}.out"):
            for line in f.read_text().splitlines():
                d = json.loads(line)
                if d.get("header"):
                    continue
                acc[m, d["x"], d["kind"], d["path"]].append(d["median_ns"])
    xname = {"args": "n_args", "constexpr": "n_constexpr", "cache": "n_cached_kernels"}[
        kind
    ]
    rows = []
    for (m, x, k, path), vals in sorted(
        acc.items(), key=lambda kv: (kv[0][0], kv[0][2], kv[0][1], kv[0][3])
    ):
        med, lo, hi = spread(vals)
        rows.append(
            [
                MACHINES[m],
                x,
                k,
                path,
                fmt(med / 1000),
                fmt(lo / 1000),
                fmt(hi / 1000),
                len(vals),
            ]
        )
    write(
        f"sweep_{kind}.csv",
        ["machine", xname, "arg_kind", "path", "us_per_call", "lo", "hi", "rounds"],
        rows,
    )
    # the same, one row per x: the four paths side by side
    wide = defaultdict(dict)
    for (m, x, k, path), vals in acc.items():
        wide[m, k, x][path] = statistics.median(vals) / 1000
    rows = []
    for (m, k, x), p in sorted(wide.items()):
        r = [MACHINES[m], x, k] + [
            fmt(p.get(c, ""))
            for c in ("triton_launch", "intj_launch", "triton_host", "intj_host")
        ]
        if "triton_launch" in p and "intj_launch" in p:
            r.append(fmt(p["triton_launch"] / p["intj_launch"], 2))
        else:
            r.append("")
        if "triton_host" in p and "intj_host" in p:
            r.append(fmt(p["triton_host"] / p["intj_host"], 1))
        else:
            r.append("")
        rows.append(r)
    write(
        f"sweep_{kind}_wide.csv",
        [
            "machine",
            xname,
            "arg_kind",
            "triton_launch_us",
            "intj_launch_us",
            "triton_host_us",
            "intj_host_us",
            "launch_speedup",
            "host_speedup",
        ],
        rows,
    )


def ffi():
    pat = re.compile(r"(?:args=\s*(\d+) )?(.+?)\s+median=([\d.]+)")
    for name, glob in (
        ("ffi_compare", "r*_ffi_compare.out"),
        ("ffi_sweep", "r*_ffi_sweep*.out"),
        ("hip_module_same_hsaco", "r*_hip_module.out"),
        ("same_binary_1arg", "r*_same_binary_1arg.out"),
    ):
        acc = defaultdict(list)
        for m in MACHINES:
            for f in rounds(m, glob):
                for line in f.read_text().splitlines():
                    g = pat.match(line)
                    if g and not line.startswith(
                        ("iters=", "counts=", "hsaco_sha256", "binary_sha256")
                    ):
                        acc[m, int(g[1]) if g[1] else "", g[2].strip()].append(
                            float(g[3])
                        )
        rows = []
        for (m, n, case), vals in sorted(
            acc.items(), key=lambda kv: (kv[0][0], str(kv[0][1]).zfill(3), kv[0][2])
        ):
            med, lo, hi = spread(vals)
            rows.append(
                [MACHINES[m]]
                + ([n] if name == "ffi_sweep" else [])
                + [case, fmt(med, 4), fmt(lo, 4), fmt(hi, 4), len(vals)]
            )
        write(
            f"{name}.csv",
            ["machine"]
            + (["n_args"] if name == "ffi_sweep" else [])
            + ["case", "us_per_call", "lo", "hi", "rounds"],
            rows,
        )


# ------------------------------------------------------------- AITER


SHAPE_COLS = {
    "flash_kda": ["B", "T", "H"],
    "kda_decode": ["batch", "H"],
    "fp8_mqa_logits": ["seq_q", "seq_kv", "H"],
}


def aiter():
    base = RAW / "mi350x" / "aiter"
    if not base.exists():
        return
    acc = defaultdict(lambda: defaultdict(list))
    sha = defaultdict(set)
    errors = []
    for f in sorted(base.glob("r*_*.jsonl")):
        route = "triton" if "_triton_" in f.stem else "default"
        for line in f.read_text().splitlines():
            if not line.startswith("{"):
                continue
            d = json.loads(line)
            if d.get("header"):
                continue
            key = (d["op"], route, tuple(d["shape"]), d["side"], d["variant"])
            if "error" in d:
                errors.append((f.name, d["shape"], d["error"]))
                continue
            for metric in ("host_us", "eager_us", "do_bench_us"):
                acc[key][metric].append(d[metric])
            sha[d["op"], route, tuple(d["shape"])].add(d["sha1"])
    kernel = {}
    for f in sorted(base.glob("prof_*.summary.jsonl")):
        route = "triton" if "_triton_" in f.name else "default"
        for line in f.read_text().splitlines():
            d = json.loads(line)
            kernel[d["op"], route, tuple(d["shape"]), d["side"]] = d
    summary = []
    for op, cols in SHAPE_COLS.items():
        rows = []
        by_shape = defaultdict(dict)
        for key in sorted(k for k in acc if k[0] == op):
            _, route, shape, side, variant = key
            if variant == "triton_jit" or (side == "base" and op != "flash_kda"):
                label = "base_triton_jit"
            elif side == "base":
                label = "base_fastlaunch"
            else:
                label = "intj"
            med = {
                m: spread(acc[key][m]) for m in ("host_us", "eager_us", "do_bench_us")
            }
            k = kernel.get((op, route, shape, side)) if variant == "default" else None
            # every launch path runs the same kernels: any side's trace serves
            ref = [
                kernel[op, route, shape, s]["kernel_us"]
                for s in ("base", "intj")
                if (op, route, shape, s) in kernel
            ]
            gpu = statistics.mean(ref) if ref else med["do_bench_us"][0]
            stall = med["eager_us"][0] > 1.25 * max(med["host_us"][0], gpu)
            r = [route, *shape, label]
            for m in ("host_us", "eager_us", "do_bench_us"):
                r += [fmt(v, 2) for v in med[m]]
            r += [k["kernel_us"] if k else "", k["kernels_per_call"] if k else ""]
            r += [
                "driver_stall" if stall else "",
                len(sha[op, route, shape]) == 1,
                len(acc[key]["host_us"]),
            ]
            rows.append(r)
            by_shape[route, shape][label] = (
                med["host_us"][0],
                med["eager_us"][0],
                stall,
                gpu,
            )
        write(
            f"aiter_{op}.csv",
            [
                "route",
                *cols,
                "launch_path",
                "host_us",
                "host_lo",
                "host_hi",
                "eager_us",
                "eager_lo",
                "eager_hi",
                "do_bench_us",
                "do_bench_lo",
                "do_bench_hi",
                "gpu_kernel_us",
                "kernels_per_call",
                "eager_flag",
                "output_bitwise_equal",
                "rounds",
            ],
            rows,
        )
        for (route, shape), p in sorted(by_shape.items()):
            names = ("base_triton_jit", "base_fastlaunch", "intj")
            row = [op, route, "x".join(map(str, shape)), fmt(p["intj"][3], 2)]
            for n in names:
                row += [
                    fmt(p[n][0], 2) if n in p else "",
                    "" if n not in p else "stall" if p[n][2] else fmt(p[n][1], 2),
                ]
            base = "base_fastlaunch" if "base_fastlaunch" in p else "base_triton_jit"
            host_bound = p[base][0] > p["intj"][3]
            row.append("upstream host-bound" if host_bound else "gpu-bound")
            row.append(fmt(p[base][0] / p["intj"][0], 2))
            # gpu-bound: both sides run at the kernel time (gpu_kernel_us), and
            # deep eager queues stall on this MI350's driver; no ratio there
            ok = host_bound and not p[base][2] and not p["intj"][2]
            row.append(fmt(p[base][1] / p["intj"][1], 2) if ok else "")
            summary.append(row)
    write(
        "aiter_summary.csv",
        [
            "op",
            "route",
            "shape",
            "gpu_kernel_us",
            "triton_jit_host_us",
            "triton_jit_eager_us",
            "fastlaunch_host_us",
            "fastlaunch_eager_us",
            "intj_host_us",
            "intj_eager_us",
            "regime",
            "host_speedup_vs_upstream",
            "eager_speedup_vs_upstream",
        ],
        summary,
    )
    if errors:
        write("aiter_errors.csv", ["file", "shape", "error"], errors)


JOURNAL = (
    DATA.parent.parent
    / "benchmarks"
    / "journals"
    / "2026-10-14_full-suite_0_bc8caf7.md"
)


def ffi_dispatch():
    """bench_unordered_map_lookup.cpp (ns per argument) and bench_ffi_dispatch.py (ns per call)."""
    acc = defaultdict(list)
    meta = {}
    order = []
    for m in MACHINES:
        for f in rounds(m, "r*_umap.out"):
            for line in f.read_text().splitlines():
                d = json.loads(line)
                key = (m, "unordered_map", d["case"])
                order.append(key) if key not in acc else None
                acc[key].append(d["ns_per_arg"])
                meta[key] = (d["entries"], "ns_per_arg")
        for f in rounds(m, "r*_ffi_dispatch.out"):
            for line in f.read_text().splitlines():
                d = json.loads(line)
                if "case" not in d:
                    continue
                key = (m, "tvm_ffi_nop", f"{d['load']}: {d['case']}")
                order.append(key) if key not in acc else None
                acc[key].append(d["ns"])
                meta[key] = (d["map_entries"], "ns_per_call")
    rows = []
    for key in order:
        med, lo, hi = spread(acc[key])
        entries, unit = meta[key]
        rows.append(
            [
                MACHINES[key[0]],
                key[1],
                key[2],
                entries,
                unit,
                fmt(med, 2),
                fmt(lo, 2),
                fmt(hi, 2),
                len(acc[key]),
            ]
        )
    write(
        "ffi_dispatch.csv",
        [
            "machine",
            "bench",
            "case",
            "map_entries",
            "unit",
            "value",
            "lo",
            "hi",
            "rounds",
        ],
        rows,
    )


def kernel_cache_map():
    """The map-only benchmark (tests/bench_kernel_cache.cpp) from the 2026-10-14 full-suite journal,
    MI308X host (Xeon 8480C); `new=` is that run, the median of its 3 rounds."""
    pat = re.compile(
        r"^kernel_cache\s+(\d)-word key\s+(\S+)\s+(\S+)\s+baseline=\S+ new=([\d.]+)"
    )
    rows = []
    for line in JOURNAL.read_text().splitlines():
        g = pat.match(line)
        if g:
            case = g[2]
            op, _, entries = case.partition("/")
            rows.append([int(g[1]), op, entries, g[3], float(g[4])])
    rows.sort(key=lambda r: (r[0], r[1], int(r[2]) if r[2] else 0, r[3]))
    write("kernel_cache_map.csv", ["key_words", "op", "entries", "backend", "ns"], rows)


def incompatible():
    """bench_incompatible.py: host ns per call of the Triton conveniences intj drops."""
    acc = defaultdict(list)
    order = []
    for m in MACHINES:
        for f in rounds(m, "r*_incompatible.out"):
            for line in f.read_text().splitlines():
                d = json.loads(line)
                if "case" in d:
                    key = (m, d["group"], d["case"].strip())
                    if key not in acc:
                        order.append(key)
                    acc[key].append(d["ns"])
    rows = []
    for m, group, case in order:
        med, lo, hi = spread(acc[m, group, case])
        rows.append(
            [
                MACHINES[m],
                group,
                case,
                fmt(med, 1),
                fmt(lo, 1),
                fmt(hi, 1),
                len(acc[m, group, case]),
            ]
        )
    write(
        "incompatible_micro.csv",
        ["machine", "group", "case", "ns_per_call", "lo", "hi", "rounds"],
        rows,
    )


def fp8_wrapper():
    """fp8_mqa_logits on the MI350: the intj op vs the leaner-wrapper prototype, and the
    host-time breakdown of the op's wrapper (raw/mi350x/aiter/fp8_prototype, fp8_breakdown)."""
    base = RAW / "mi350x" / "aiter"
    acc = defaultdict(list)
    for f in sorted((base / "fp8_prototype").glob("r*.jsonl")):
        for line in f.read_text().splitlines():
            d = json.loads(line)
            acc[d["shape"][0]].append(d)
    rows = []
    for n, ds in sorted(acc.items()):
        row = [n, n, 64]
        for k in (
            "intj_host_us",
            "intj_eager_us",
            "prototype_host_us",
            "prototype_eager_us",
        ):
            med, lo, hi = spread([d[k] for d in ds])
            row += [fmt(med), fmt(lo), fmt(hi)]
        rows.append(row + [all(d["bitwise_equal"] for d in ds), len(ds)])
    cols = []
    for k in (
        "intj_host_us",
        "intj_eager_us",
        "prototype_host_us",
        "prototype_eager_us",
    ):
        cols += [k, k.replace("_us", "_lo"), k.replace("_us", "_hi")]
    write(
        "aiter_fp8_mqa_logits_prototype.csv",
        ["seq_q", "seq_kv", "H"] + cols + ["output_bitwise_equal", "rounds"],
        rows,
    )

    acc = defaultdict(list)
    for f in sorted((base / "fp8_breakdown").glob("r*_*.jsonl")):
        for line in f.read_text().splitlines():
            d = json.loads(line)
            if "sections_us" not in d:
                continue
            for step, v in d["sections_us"].items():
                acc[d["side"], step, d["shape"][0]].append(v)
            acc[d["side"], "op (uninstrumented)", d["shape"][0]].append(
                d["host_us_real"]
            )
    for f in sorted((base / "fp8_breakdown").glob("extra_*.jsonl")):
        for line in f.read_text().splitlines():
            d = json.loads(line)
            acc[d["side"], "free previous logits", d["shape"][0]].append(
                d["free_logits_us"]
            )
    rows = []
    for (side, step, n), vals in sorted(acc.items()):
        med, lo, hi = spread(vals)
        rows.append([side, step, n, fmt(med), fmt(lo), fmt(hi), len(vals)])
    write(
        "aiter_fp8_mqa_logits_breakdown.csv",
        ["side", "step", "seq_q_eq_kv", "host_us", "lo", "hi", "rounds"],
        rows,
    )


if __name__ == "__main__":
    headline()
    for kind in ("args", "constexpr", "cache"):
        sweep(kind)
    ffi()
    aiter()
    fp8_wrapper()
    incompatible()
    kernel_cache_map()
    ffi_dispatch()

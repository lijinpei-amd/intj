# pyright: standard
"""GPU kernel time per call from a `bench_aiter_ops.py --profile N` run.

usage: python summarize_rocprof.py <kernel_trace.csv> <bench stdout.jsonl>

The trace holds one sentinel, then per shape: a sentinel, N calls, a sentinel.
Prints one JSON line per shape: summed kernel duration per call, kernels per
call, and the per-kernel-name breakdown.
"""

import csv
import json
import sys
from collections import defaultdict


def main(trace, jsonl):
    rows = sorted(csv.DictReader(open(trace)), key=lambda r: int(r["Start_Timestamp"]))
    cuts = [i for i, r in enumerate(rows) if r["Kernel_Name"] == "ppt_sentinel_kernel"]
    recs = [
        json.loads(line)
        for line in open(jsonl)
        if line.startswith("{") and "profile_calls" in line
    ]
    assert len(cuts) == 1 + 2 * len(recs), (len(cuts), len(recs))
    for k, rec in enumerate(recs):
        window = rows[cuts[1 + 2 * k] + 1 : cuts[2 + 2 * k]]
        n = rec["profile_calls"]
        by_name = defaultdict(float)
        for r in window:
            by_name[r["Kernel_Name"]] += (
                int(r["End_Timestamp"]) - int(r["Start_Timestamp"])
            ) / 1000
        span = (
            int(window[-1]["End_Timestamp"]) - int(window[0]["Start_Timestamp"])
        ) / 1000
        print(
            json.dumps(
                {
                    "op": rec["op"],
                    "side": rec["side"],
                    "shape": rec["shape"],
                    "sha1": rec["sha1"],
                    "calls": n,
                    "kernel_us": round(sum(by_name.values()) / n, 3),
                    "kernels_per_call": len(window) / n,
                    "span_us": round(span / n, 3),
                    "by_kernel_us": {
                        k: round(v / n, 3) for k, v in sorted(by_name.items())
                    },
                }
            )
        )


if __name__ == "__main__":
    main(*sys.argv[1:])

# pyright: standard
"""AITER fp8_mqa_logits on the intj tree vs a leaner wrapper prototype, timed as bench_aiter_ops.

usage: SCRIPT=$W/scripts/bench_fp8_mqa_prototype.py bash fp8_breakdown_run.sh intj [--shapes 128,...]

The prototype (`bench_fp8_mqa_breakdown.make_proto`) is the same wrapper with the
gfx950 config cached per shape and the logits slice skipped when seq_len_kv is
256-aligned; it keeps clean_logits=True and its output is checked bitwise equal
to the op's. Both are timed in one process with bench_aiter_ops' host_us
(batches of 8 calls) and eager_us (32 back-to-back calls between GPU events);
the order alternates by shape.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("AITER_TRITON_ONLY", "1")

import torch  # noqa: E402

import bench_aiter_ops as A  # noqa: E402  # pyright: ignore[reportMissingImports]  # this directory, joined above
import bench_fp8_mqa_breakdown as B  # noqa: E402  # pyright: ignore[reportMissingImports]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shapes", default="128,256,512,1024,2048,4096,8192")
    o = p.parse_args()
    import aiter.ops.triton.attention.fp8_mqa_logits as M  # pyright: ignore[reportMissingImports]

    assert hasattr(M, "current_device_stream"), "needs the intj AITER tree"
    proto = B.make_proto(M)
    for i, n in enumerate(int(x) for x in o.shapes.split(",")):
        q, kv, s, w, ks, ke = B.inputs(n, n, 64)

        def real():
            return M.fp8_mqa_logits(q, kv, s, w, ks, ke, True)

        def fast():
            return proto(q, kv, s, w, ks, ke, True)

        r = {"shape": [n, n, 64], "bitwise_equal": bool(torch.equal(real(), fast()))}
        cases = [("intj", real), ("prototype", fast)]
        for name, fn in cases if i % 2 == 0 else reversed(cases):
            r[f"{name}_host_us"] = round(A.host_us(fn, 8, 300)[0], 3)
            r[f"{name}_eager_us"] = round(A.eager_us(fn)[0], 3)
        print(json.dumps(r), flush=True)


if __name__ == "__main__":
    main()

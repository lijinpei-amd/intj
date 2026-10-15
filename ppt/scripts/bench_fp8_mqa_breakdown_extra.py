# Sections with clean_logits=False, and the cost of freeing a returned logits tensor.
import json, statistics, sys, time

sys.path.insert(
    0, __import__("os").path.dirname(__import__("os").path.abspath(__file__))
)
import bench_fp8_mqa_breakdown as B
import torch

pc = time.perf_counter_ns
import aiter.ops.triton.attention.fp8_mqa_logits as M

side = "intj" if hasattr(M, "current_device_stream") else "base"
g = dict(M.__dict__)
g["pc"] = pc
src = B.BODY.replace("#STREAM", B.STREAM_INTJ if side == "intj" else B.STREAM_BASE)
src = src.replace("#LAUNCH", B.LAUNCH_INTJ if side == "intj" else B.LAUNCH_BASE)
exec(compile(src, "inst", "exec"), g)
inst = g["inst"]
for n in (128, 512, 1024):
    q, kv, s, w, ks, ke = B.inputs(n, n, 64)
    r = {"side": side, "shape": [n, n, 64]}
    for clean in (True, False):
        rec = []
        for _ in range(20):
            inst(q, kv, s, w, ks, ke, clean, rec)
        torch.cuda.synchronize()
        rec.clear()
        for _ in range(500):
            for _ in range(8):
                inst(q, kv, s, w, ks, ke, clean, rec)
            torch.cuda.synchronize()
        r[f"sections_clean_{clean}"] = {
            k: round(statistics.median(t[i + 1] - t[i] for t in rec) / 1000, 3)
            for i, k in enumerate(B.SECTIONS)
        }
    # free cost: drop 8 returned outputs (full tensor + its view), timed
    fr = []
    for _ in range(300):
        outs = [M.fp8_mqa_logits(q, kv, s, w, ks, ke, True) for _ in range(8)]
        torch.cuda.synchronize()
        t0 = pc()
        outs.clear()
        t1 = pc()
        fr.append((t1 - t0) / 8000)
    r["free_logits_us"] = round(statistics.median(fr), 3)
    print(json.dumps(r), flush=True)

# pyright: standard
"""Host and GPU time of AITER entry points, upstream launches vs intj launchers.

usage: python bench_aiter_ops.py --op {flash_kda,kda_decode,fp8_mqa_logits}
           --side {base,intj} [--bypass] [--shapes ...]

One process measures one AITER tree (the one on PYTHONPATH). Per shape, after
warm-up (compile, any tuning):

- `host_us`: host time per eager call; batches of `--host-iters` calls, timed
  from first call to last return, device synchronized after each batch outside
  the timer. Median over batches.
- `eager_us`: time per call of 32 back-to-back eager calls, between a GPU
  event recorded before the first call and one after the last: what a stream
  of eager calls costs, about max(host, GPU) per call. Median of 41 loops
  after 200 warm-up calls. GPU events, not a host timer around a synchronize:
  on this MI350 a blocking synchronize can wake on a 4 ms tick (loop totals
  land on 8, 12, 16, ... ms), which penalizes whichever side finishes
  enqueueing first. Deeper loops (64 calls of a 5-kernel op) also fall into a
  slow driver path on both sides, and single loops stall for ~8 ms; the
  median absorbs those.
- `do_bench_us`: `triton.testing.do_bench(fn, warmup=25, rep=100)`, median;
  AITER's own benchmark metric (event-timed per call, L2 flushed between calls).
- `sha1`: hash of the first call's output bytes (same seeded inputs on both sides).

`--profile N` replaces the timings: per shape, a sentinel kernel, N calls, a
sentinel; run under `rocprofv3 --kernel-trace` and split the trace at the
sentinels for GPU kernel time per call (`summarize_rocprof.py`).

`--bypass` (FlashKDA on the base tree only) also times every shape inside
`fast_launch.bypassed()`: AITER's `_FastLaunch` cache off, i.e. stock Triton
`JITFunction` launches.
"""

import argparse
import contextlib
import hashlib
import json
import math
import os
import statistics
import sys
import time

os.environ.setdefault("AITER_TRITON_ONLY", "1")

import torch  # noqa: E402
import triton  # noqa: E402

DEV = "cuda"


def output_hash(out):
    h = hashlib.sha1()
    for t in out if isinstance(out, (tuple, list)) else (out,):
        if isinstance(t, torch.Tensor):
            h.update(t.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return h.hexdigest()


def host_us(fn, iters, batches):
    sync = torch.cuda.synchronize
    for _ in range(20):
        fn()
    sync()
    samples = []
    for _ in range(batches):
        t0 = time.perf_counter_ns()
        for _ in range(iters):
            fn()
        t1 = time.perf_counter_ns()
        sync()
        samples.append((t1 - t0) / iters / 1000)
    return statistics.median(samples), samples


def eager_us(fn, reps=41, n=32):
    for _ in range(200):
        fn()
    samples = []
    for _ in range(reps):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(n):
            fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / n)
    return statistics.median(samples), n, samples


@triton.jit
def ppt_sentinel_kernel(x):
    pass


def sentinel():
    ppt_sentinel_kernel[(1,)](0)


# ---------------------------------------------------------------- ops


def flash_kda_case(shape):
    from aiter.ops.triton._triton_kernels.chunk_delta_attn.flash_kda import (
        flash_kda_fwd,
    )

    B, T, H = shape
    K = V = 128
    torch.manual_seed(42)
    dt = torch.bfloat16
    t = {
        "q": torch.randn(B, T, H, K, device=DEV, dtype=dt),
        "k": torch.randn(B, T, H, K, device=DEV, dtype=dt),
        "v": torch.randn(B, T, H, V, device=DEV, dtype=dt),
        "g": torch.randn(B, T, H, K, device=DEV, dtype=dt) * 0.1,
        "beta": torch.randn(B, T, H, device=DEV, dtype=torch.float32),
        "A_log": torch.randn(H, device=DEV, dtype=torch.float32).abs() * 0.5,
        "dt_bias": torch.randn(H * K, device=DEV, dtype=torch.float32) * 0.1,
        "scale": 1.0 / math.sqrt(K),
    }

    def fn():
        return flash_kda_fwd(**t, lower_bound=-5.0)

    return fn, fn


def kda_decode_case(shape):
    from aiter.ops.triton.gated_delta_net.fused_kda_decode import fused_kda_decode

    batch, H = shape
    D, W = 128, 4
    lp = H * D
    slots = batch + 2
    dt = torch.bfloat16
    torch.manual_seed(0)
    inp = {
        "mixed_qkv": torch.randn(batch, 3 * lp, dtype=dt, device=DEV),
        "conv_weight": torch.randn(3 * lp, W, dtype=dt, device=DEV) * 0.1,
        "conv_state": torch.randn(slots, 3 * lp, W - 1, dtype=dt, device=DEV) * 0.1,
        "gate": torch.randn(1, batch, H, D, dtype=dt, device=DEV) * 0.5,
        "beta": torch.randn(1, batch, H, dtype=dt, device=DEV),
        "out_gate": torch.randn(batch, lp, dtype=dt, device=DEV),
        "A_log": torch.randn(H, dtype=dt, device=DEV) * 0.1,
        "dt_bias": torch.randn(lp, dtype=dt, device=DEV) * 0.1,
        "ssm_state": torch.randn(slots, H, D, D, dtype=torch.float32, device=DEV)
        * 0.01,
        "norm_weight": torch.ones(D, dtype=dt, device=DEV),
        "ssm_state_indices": torch.arange(batch, dtype=torch.int32, device=DEV),
        "cu_seqlens": torch.arange(batch + 1, dtype=torch.int64, device=DEV),
    }

    def call(cs, ss):
        return fused_kda_decode(
            inp["mixed_qkv"],
            cs,
            inp["conv_weight"],
            inp["gate"],
            inp["beta"],
            inp["out_gate"],
            inp["A_log"],
            inp["dt_bias"],
            ss,
            inp["ssm_state_indices"],
            inp["cu_seqlens"],
            inp["norm_weight"],
            1e-6,
            D,
            H,
            -5.0,
        )

    # the timed calls update the states in place, like a decode step
    cs, ss = inp["conv_state"].clone(), inp["ssm_state"].clone()

    return lambda: call(
        inp["conv_state"].clone(), inp["ssm_state"].clone()
    ), lambda: call(cs, ss)


def fp8_mqa_logits_case(shape):
    from aiter.ops.triton.attention.fp8_mqa_logits import fp8_mqa_logits
    from aiter.ops.triton.utils.types import e4m3_dtype

    s_q, s_k, H = shape
    D = 128
    torch.manual_seed(0)
    q = torch.randn(s_q, H, D, device=DEV, dtype=torch.bfloat16)
    kv = torch.randn(s_k, D, device=DEV, dtype=torch.bfloat16)
    # per_custom_dims_cast_to_fp8(kv, (0,), False) from the op's test
    amax = kv.abs().float().amax(dim=1, keepdim=True).clamp(1e-4)
    scales = (amax / torch.finfo(e4m3_dtype).max).squeeze(1)
    kv_fp8 = (kv * (1.0 / scales[:, None])).to(e4m3_dtype)
    weights = torch.randn(s_q, H, device=DEV, dtype=torch.float32)
    ks = torch.zeros(s_q, dtype=torch.int, device=DEV)
    ke = (s_k - s_q) + torch.arange(s_q, dtype=torch.int, device=DEV) + 1
    q_fp8 = q.to(e4m3_dtype)

    def fn():
        return fp8_mqa_logits(q_fp8, kv_fp8, scales, weights, ks, ke, True)

    return fn, fn


OPS = {
    # (B, T, H): Kimi's H=12; T from 2 chunks to 16k tokens
    "flash_kda": (
        flash_kda_case,
        [(1, t, 12) for t in (64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384)],
    ),
    # (batch, local heads): one decode token per sequence
    "kda_decode": (
        kda_decode_case,
        [(b, 12) for b in (1, 4, 8, 16, 32, 64, 128, 256)],
    ),
    # (query tokens, KV tokens, heads): DeepSeek-V3.2 indexer, causal prefill
    # (q = kv) and a short chunk over a long context (q << kv)
    "fp8_mqa_logits": (
        fp8_mqa_logits_case,
        [(1, 8192, 64), (16, 8192, 64), (64, 8192, 64)]
        + [(n, n, 64) for n in (128, 256, 512, 1024, 2048, 4096, 8192)],
    ),
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--op", choices=OPS, required=True)
    p.add_argument("--side", choices=("base", "intj"), required=True)
    p.add_argument("--tag", default="")
    p.add_argument("--bypass", action="store_true")
    p.add_argument("--host-iters", type=int, default=8)
    p.add_argument("--host-batches", type=int, default=300)
    p.add_argument("--shapes", default="", help="indices into the op's shape list")
    p.add_argument(
        "--profile", type=int, default=0, help="calls per shape under rocprofv3"
    )
    o = p.parse_args()
    make, shapes = OPS[o.op]
    if o.shapes:
        shapes = [shapes[int(i)] for i in o.shapes.split(",")]

    intj_file = None
    try:
        import intj

        intj_file = intj.__file__
    except ImportError:
        pass
    import aiter

    torch.cuda.init()
    dev = torch.cuda.current_device()
    print(
        json.dumps(
            {
                "header": True,
                "op": o.op,
                "side": o.side,
                "tag": o.tag,
                "aiter": aiter.__file__,
                "intj": intj_file,
                "gpu": torch.cuda.get_device_name(dev),
                "arch": torch.cuda.get_device_properties(dev).gcnArchName,
                "hip_visible_devices": os.environ.get("HIP_VISIBLE_DEVICES"),
                "cpu_affinity": sorted(os.sched_getaffinity(0)),
                "python": sys.version.split()[0],
                "torch": torch.__version__,
                "triton": triton.__version__,
                "env": {
                    k: v
                    for k, v in os.environ.items()
                    if k.startswith(("AITER_", "TRITON_"))
                },
                "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            }
        ),
        flush=True,
    )

    variants = [("default", contextlib.nullcontext)]
    if o.bypass:
        from aiter.ops.triton._triton_kernels.chunk_delta_attn import fast_launch

        variants.append(("triton_jit", fast_launch.bypassed))

    sentinel()
    torch.cuda.synchronize()
    for shape in shapes:
        first, fn = make(shape)
        sha = output_hash(first())
        torch.cuda.synchronize()
        if o.profile:
            for _ in range(20):
                fn()
            torch.cuda.synchronize()
            sentinel()
            for _ in range(o.profile):
                fn()
            sentinel()
            torch.cuda.synchronize()
            rec = {
                "op": o.op,
                "side": o.side,
                "variant": "default",
                "shape": list(shape),
                "sha1": sha,
                "profile_calls": o.profile,
            }
            print(json.dumps(rec), flush=True)
            continue
        for name, ctx in variants:
            rec = {
                "op": o.op,
                "side": o.side,
                "variant": name,
                "shape": list(shape),
                "sha1": sha,
            }
            try:
                with ctx():
                    rec["host_us"], rec["host_samples"] = host_us(
                        fn, o.host_iters, o.host_batches
                    )
                    rec["eager_us"], rec["eager_calls"], rec["eager_samples"] = (
                        eager_us(fn)
                    )
                    rec["do_bench_us"] = 1000 * triton.testing.do_bench(
                        fn, warmup=25, rep=100, return_mode="median"
                    )
            except Exception as e:  # noqa: BLE001
                rec["error"] = f"{type(e).__name__}: {e}"[:500]
            for k in ("host_samples", "eager_samples"):
                if k in rec:
                    rec[k] = [round(v, 3) for v in rec[k]]
            print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()

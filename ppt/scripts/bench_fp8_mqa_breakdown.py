# Host-time breakdown of aiter fp8_mqa_logits (gfx950 Gluon path), one AITER tree per process.
# usage: bash fp8_breakdown_run.sh base|intj --shapes 128,512,1024 [--calls N]   (fp8_breakdown_drive.sh runs it all)
# make_proto is the leaner-wrapper prototype that bench_fp8_mqa_prototype.py times.
import argparse, json, os, statistics, sys, time

os.environ.setdefault("AITER_TRITON_ONLY", "1")
import torch, triton

DEV = "cuda"
pc = time.perf_counter_ns


def inputs(s_q, s_k, H):  # same as bench_aiter_ops.fp8_mqa_logits_case
    from aiter.ops.triton.utils.types import e4m3_dtype

    D = 128
    torch.manual_seed(0)
    q = torch.randn(s_q, H, D, device=DEV, dtype=torch.bfloat16)
    kv = torch.randn(s_k, D, device=DEV, dtype=torch.bfloat16)
    amax = kv.abs().float().amax(dim=1, keepdim=True).clamp(1e-4)
    scales = (amax / torch.finfo(e4m3_dtype).max).squeeze(1)
    kv_fp8 = (kv * (1.0 / scales[:, None])).to(e4m3_dtype)
    weights = torch.randn(s_q, H, device=DEV, dtype=torch.float32)
    ks = torch.zeros(s_q, dtype=torch.int, device=DEV)
    ke = (s_k - s_q) + torch.arange(s_q, dtype=torch.int, device=DEV) + 1
    q_fp8 = q.to(e4m3_dtype)
    return q_fp8, kv_fp8, scales, weights, ks, ke


# Copy of the wrapper body (gfx950 Gluon path), checkpoints between sections.
BODY = """
def inst(Q, KV, kv_scales, weights, cu_starts, cu_ends, clean_logits, rec):
    t0 = pc()
    seq_len, num_heads, head_size = Q.shape
    seq_len_kv = KV.shape[0]
    assert num_heads & (num_heads - 1) == 0, "num q. heads should be power of 2."
    assert head_size & (head_size - 1) == 0, "head size should be power of 2."
    aligned_size = 256
    seq_len_kv_aligned = (seq_len_kv + aligned_size - 1) // aligned_size * aligned_size
    t1 = pc()
    device = Q.device
    t2 = pc()
    if clean_logits:
        full = torch.full((seq_len, seq_len_kv_aligned), fill_value=-float("inf"), dtype=torch.float32, device=device)
    else:
        full = torch.empty((seq_len, seq_len_kv_aligned), dtype=torch.float32, device=device)
    t3 = pc()
    logits = full[:, :seq_len_kv]
    t4 = pc()
    use_gluon = TRITON_GE_36 and _gluon_fp8_mqa_logits_kernel is not None
    stride_q_s, stride_q_h, stride_q_d = Q.stride()
    stride_kv_s, stride_kv_d = KV.stride()
    stride_w_s, stride_w_h = weights.stride()
    stride_logits_s, stride_logits_k = logits.stride()
    t5 = pc()
    if not use_gluon:
        raise RuntimeError("expected the Gluon path")
    INT32_MAX = 2**31 - 1
    max_kv_offset = (seq_len_kv - 1) * stride_kv_s + (head_size - 1) * stride_kv_d
    max_logits_offset = (seq_len - 1) * stride_logits_s + (seq_len_kv - 1) * stride_logits_k
    use_buffer_load = max_kv_offset <= INT32_MAX
    use_buffer_store = max_logits_offset <= INT32_MAX
    num_buffers = 2
    USE_FOLDED_REDUCTION = FOLDED_REDUCTED_SUPPORT and num_heads > 16
    if arch == "gfx950":
        MIN_BLOCK_M2_WGS = 1024
        use_buffer_load = True
        use_buffer_store = True
        num_buffers = 2
        loop_variant = 0
        waves_per_eu = 2 if TRITON_GE_38 else 3
        num_warps = 2
        block_kv = 64
        num_kv_splits = _gfx950_kv_splits(seq_len, seq_len_kv, 2, num_warps, waves_per_eu)
        if (num_heads <= 32 and seq_len >= 2 and (seq_len > 4096 or triton.cdiv(seq_len, 2) * num_kv_splits >= MIN_BLOCK_M2_WGS)):
            block_m = 2
        else:
            block_m = 1
            num_kv_splits = _gfx950_kv_splits(seq_len, seq_len_kv, block_m, num_warps, waves_per_eu)
        if block_m == 1 and seq_len > 4096:
            num_warps = 1
            block_kv = 32
            num_kv_splits = _gfx950_kv_splits(seq_len, seq_len_kv, block_m, num_warps, waves_per_eu)
        mfma_nonk_dim = 32 if (head_size <= 64 or num_heads >= 32) else 16
        num_chains = (2 if block_m == 2 else 1) if USE_FOLDED_REDUCTION else 0
        if (num_chains >= 1 and num_heads > mfma_nonk_dim and block_m == 1 and mfma_nonk_dim == 32):
            m_chunk = mfma_nonk_dim
        else:
            m_chunk = 0
        relaxed_store = 0 if clean_logits else 1
        t6 = pc()
        other = {
            "USE_PADDED_SHARED_LAYOUT": ASYNC_COPY_SUPPORTS_DISTRIBUTED,
            "BLOCK_M": block_m,
            "MFMA_NONK_DIM": mfma_nonk_dim,
            "M_CHUNK": m_chunk,
            "UNROLL": 2,
            "RELAXED_STORE": relaxed_store,
            "HAS_KV_SPLIT": 1 if num_kv_splits > 1 else 0,
            "num_kv_splits": num_kv_splits,
        }
        grid = ((seq_len + block_m - 1) // block_m, num_kv_splits)
    else:
        raise RuntimeError("expected gfx950")
    t7 = pc()
#STREAM
    t8 = pc()
#LAUNCH
    t9 = pc()
    rec.append((t0, t1, t2, t3, t4, t5, t6, t7, t8, t9))
    return logits
"""

STREAM_INTJ = "    dev, stream = current_device_stream()"
STREAM_BASE = "    pass"
LAUNCH_INTJ = """    _gluon_fp8_mqa_logits_kernel(dev, stream, grid, num_warps, waves_per_eu, Q, KV, kv_scales, weights,
        cu_starts, cu_ends, logits, seq_len, seq_len_kv, other["num_kv_splits"], num_heads, head_size,
        stride_q_s, stride_q_h, stride_q_d, stride_kv_s, stride_kv_d, stride_w_s, stride_w_h,
        stride_logits_s, stride_logits_k, block_kv, num_warps, num_buffers, num_chains,
        use_buffer_load, use_buffer_store, other["USE_PADDED_SHARED_LAYOUT"], other["BLOCK_M"],
        other["MFMA_NONK_DIM"], other["M_CHUNK"], other["UNROLL"], other["RELAXED_STORE"], other["HAS_KV_SPLIT"])"""
LAUNCH_BASE = """    _gluon_fp8_mqa_logits_kernel[grid](Q_ptr=Q, KV_ptr=KV, kv_scales_ptr=kv_scales, weights_ptr=weights,
        cu_start_ptr=cu_starts, cu_end_ptr=cu_ends, logits_ptr=logits, seq_len=seq_len, seq_len_kv=seq_len_kv,
        NUM_HEADS=num_heads, HEAD_SIZE=head_size, stride_q_s=stride_q_s, stride_q_h=stride_q_h,
        stride_q_d=stride_q_d, stride_kv_s=stride_kv_s, stride_kv_d=stride_kv_d, stride_w_s=stride_w_s,
        stride_w_h=stride_w_h, stride_logits_s=stride_logits_s, stride_logits_k=stride_logits_k,
        BLOCK_KV=block_kv, NUM_WARPS=num_warps, NUM_BUFFERS=num_buffers, NUM_CHAINS=num_chains,
        USE_BUFFER_LOAD=use_buffer_load, USE_BUFFER_STORE=use_buffer_store, num_warps=num_warps,
        waves_per_eu=waves_per_eu, **other)"""

SECTIONS = [
    "shape+asserts",
    "Q.device",
    "torch.full/empty",
    "slice [:, :kv]",
    "4x stride()",
    "gfx950 heuristics",
    "other dict+grid",
    "current_device_stream",
    "kernel launch",
]


def make_proto(M):
    """intj wrapper with the gfx950 config cached per shape and the slice skipped when aligned."""
    K = M._gluon_fp8_mqa_logits_kernel
    raw_stream = torch._C._cuda_getCurrentRawStream
    current_device = torch.cuda.current_device
    full, empty, f32, ninf = torch.full, torch.empty, torch.float32, -float("inf")
    cache = {}

    def config(seq_len, seq_len_kv, num_heads, head_size, clean_logits):
        aligned = (seq_len_kv + 255) // 256 * 256
        wpe = 2 if M.TRITON_GE_38 else 3
        nw, bkv = 2, 64
        nk = M._gfx950_kv_splits(seq_len, seq_len_kv, 2, nw, wpe)
        if (
            num_heads <= 32
            and seq_len >= 2
            and (seq_len > 4096 or triton.cdiv(seq_len, 2) * nk >= 1024)
        ):
            bm = 2
        else:
            bm = 1
            nk = M._gfx950_kv_splits(seq_len, seq_len_kv, bm, nw, wpe)
        if bm == 1 and seq_len > 4096:
            nw, bkv = 1, 32
            nk = M._gfx950_kv_splits(seq_len, seq_len_kv, bm, nw, wpe)
        mfma = 32 if (head_size <= 64 or num_heads >= 32) else 16
        fold = M.FOLDED_REDUCTED_SUPPORT and num_heads > 16
        nch = (2 if bm == 2 else 1) if fold else 0
        mch = mfma if (nch >= 1 and num_heads > mfma and bm == 1 and mfma == 32) else 0
        grid = ((seq_len + bm - 1) // bm, nk)
        return (
            aligned,
            grid,
            nw,
            wpe,
            nk,
            bkv,
            nch,
            M.ASYNC_COPY_SUPPORTS_DISTRIBUTED,
            bm,
            mfma,
            mch,
            0 if clean_logits else 1,
            1 if nk > 1 else 0,
        )

    def proto(Q, KV, kv_scales, weights, cu_starts, cu_ends, clean_logits=True):
        seq_len, num_heads, head_size = Q.shape
        seq_len_kv = KV.shape[0]
        key = (seq_len, seq_len_kv, num_heads, head_size, clean_logits)
        c = cache.get(key)
        if c is None:
            c = cache[key] = config(*key)
        (
            aligned,
            grid,
            nw,
            wpe,
            nk,
            bkv,
            nch,
            padded,
            bm,
            mfma,
            mch,
            relaxed,
            has_split,
        ) = c
        if clean_logits:
            logits = full((seq_len, aligned), ninf, dtype=f32, device=Q.device)
        else:
            logits = empty((seq_len, aligned), dtype=f32, device=Q.device)
        if aligned != seq_len_kv:
            logits = logits[:, :seq_len_kv]
        sqs, sqh, sqd = Q.stride()
        skvs, skvd = KV.stride()
        sws, swh = weights.stride()
        sls, slk = logits.stride()
        dev = current_device()
        K(
            dev,
            raw_stream(dev),
            grid,
            nw,
            wpe,
            Q,
            KV,
            kv_scales,
            weights,
            cu_starts,
            cu_ends,
            logits,
            seq_len,
            seq_len_kv,
            nk,
            num_heads,
            head_size,
            sqs,
            sqh,
            sqd,
            skvs,
            skvd,
            sws,
            swh,
            sls,
            slk,
            bkv,
            nw,
            2,
            nch,
            True,
            True,
            padded,
            bm,
            mfma,
            mch,
            2,
            relaxed,
            has_split,
        )
        return logits

    return proto


def batch_host_us(fn, iters=8, batches=300):  # bench_aiter_ops.host_us
    sync = torch.cuda.synchronize
    for _ in range(20):
        fn()
    sync()
    s = []
    for _ in range(batches):
        t0 = pc()
        for _ in range(iters):
            fn()
        t1 = pc()
        sync()
        s.append((t1 - t0) / iters / 1000)
    return round(statistics.median(s), 3)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shapes", default="128,512,1024")
    p.add_argument("--calls", type=int, default=4000)
    o = p.parse_args()
    import aiter.ops.triton.attention.fp8_mqa_logits as M

    side = "intj" if hasattr(M, "current_device_stream") else "base"
    assert (
        M.arch == "gfx950"
        and M._gluon_fp8_mqa_logits_kernel is not None
        and M.TRITON_GE_36
    )
    g = dict(M.__dict__)
    g["pc"] = pc
    src = BODY.replace("#STREAM", STREAM_INTJ if side == "intj" else STREAM_BASE)
    src = src.replace("#LAUNCH", LAUNCH_INTJ if side == "intj" else LAUNCH_BASE)
    exec(compile(src, "inst", "exec"), g)
    inst = g["inst"]
    K = M._gluon_fp8_mqa_logits_kernel
    print(
        json.dumps(
            {
                "side": side,
                "aiter": M.__file__,
                "python": sys.version.split()[0],
                "affinity": sorted(os.sched_getaffinity(0)),
                "pyver": sys.version,
            }
        ),
        flush=True,
    )
    timer_ovh = []
    for _ in range(10000):
        a = pc()
        b = pc()
        timer_ovh.append(b - a)
    print(
        json.dumps({"timer_overhead_ns_median": statistics.median(timer_ovh)}),
        flush=True,
    )

    for n in [int(x) for x in o.shapes.split(",")]:
        q, kv, s, w, ks, ke = inputs(n, n, 64)
        real = lambda: M.fp8_mqa_logits(q, kv, s, w, ks, ke, True)
        ref = real()
        torch.cuda.synchronize()
        rec = []
        out = inst(q, kv, s, w, ks, ke, True, rec)
        torch.cuda.synchronize()
        assert torch.equal(out, ref), "instrumented copy differs"
        r = {"shape": [n, n, 64], "side": side}
        # 1. the real op, timed as bench_aiter_ops does
        r["host_us_real"] = batch_host_us(real)
        r["host_us_real_clean_false"] = batch_host_us(
            lambda: M.fp8_mqa_logits(q, kv, s, w, ks, ke, False)
        )
        # 2. instrumented copy: per-section medians, batches of 8, sync outside
        rec.clear()
        sink = []
        r["host_us_inst"] = batch_host_us(lambda: inst(q, kv, s, w, ks, ke, True, sink))
        rec.clear()
        for _ in range(20):
            inst(q, kv, s, w, ks, ke, True, rec)
        torch.cuda.synchronize()
        rec.clear()
        for _ in range(o.calls // 8):
            for _ in range(8):
                inst(q, kv, s, w, ks, ke, True, rec)
            torch.cuda.synchronize()
        secs = {}
        for i, name in enumerate(SECTIONS):
            secs[name] = round(
                statistics.median(t[i + 1] - t[i] for t in rec) / 1000, 3
            )
        r["sections_us"] = secs
        r["sections_sum_us"] = round(sum(secs.values()), 3)
        r["inst_total_median_us"] = round(
            statistics.median(t[-1] - t[0] for t in rec) / 1000, 3
        )
        # 3. isolated pieces, same batch method
        Ls = n  # seq_len_kv_aligned == n for these shapes
        dev = q.device
        full = torch.full((n, Ls), -float("inf"), dtype=torch.float32, device=dev)
        iso = {}
        iso["torch.full"] = batch_host_us(
            lambda: torch.full(
                (n, Ls), fill_value=-float("inf"), dtype=torch.float32, device=dev
            )
        )
        iso["torch.empty"] = batch_host_us(
            lambda: torch.empty((n, Ls), dtype=torch.float32, device=dev)
        )
        iso["empty+fill_"] = batch_host_us(
            lambda: torch.empty((n, Ls), dtype=torch.float32, device=dev).fill_(
                -float("inf")
            )
        )
        iso["slice"] = batch_host_us(lambda: full[:, :n])
        iso["narrow"] = batch_host_us(lambda: full.narrow(1, 0, n))
        iso["Q.device"] = batch_host_us(lambda: q.device)
        iso["torch.cuda.current_device"] = batch_host_us(torch.cuda.current_device)
        if side == "intj":
            iso["current_device_stream"] = batch_host_us(M.current_device_stream)
        iso["triton.cdiv"] = batch_host_us(lambda: triton.cdiv(n, 1))
        iso["py_cdiv_inline"] = batch_host_us(lambda: (n + 1 - 1) // 1)
        iso["_gfx950_kv_splits"] = batch_host_us(
            lambda: M._gfx950_kv_splits(n, n, 1, 2, 2)
        )
        iso["noop_py_call_7args"] = batch_host_us(
            lambda: (lambda a, b, c, d, e, f, g: None)(q, kv, s, w, ks, ke, True)
        )
        # the launch alone, with the args the op computes for this shape
        lg = full[:, :n]
        d_, st_ = (
            torch.cuda.current_device(),
            torch._C._cuda_getCurrentRawStream(torch.cuda.current_device()),
        )
        nk = M._gfx950_kv_splits(n, n, 1, 2, 2)
        grid = (n, nk)
        sq = q.stride()
        skv = kv.stride()
        sw = w.stride()
        sl = lg.stride()
        if side == "intj":
            args = (
                q,
                kv,
                s,
                w,
                ks,
                ke,
                lg,
                n,
                n,
                nk,
                64,
                128,
                *sq,
                *skv,
                *sw,
                *sl,
                64,
                2,
                2,
                1,
                True,
                True,
                M.ASYNC_COPY_SUPPORTS_DISTRIBUTED,
                1,
                32,
                32,
                2,
                0,
                1 if nk > 1 else 0,
            )
            iso["launch_alone"] = batch_host_us(lambda: K(d_, st_, grid, 2, 2, *args))
            iso["launch_alone_grid0"] = batch_host_us(
                lambda: K(d_, st_, (0, nk), 2, 2, *args)
            )
        else:
            kw = dict(
                Q_ptr=q,
                KV_ptr=kv,
                kv_scales_ptr=s,
                weights_ptr=w,
                cu_start_ptr=ks,
                cu_end_ptr=ke,
                logits_ptr=lg,
                seq_len=n,
                seq_len_kv=n,
                NUM_HEADS=64,
                HEAD_SIZE=128,
                stride_q_s=sq[0],
                stride_q_h=sq[1],
                stride_q_d=sq[2],
                stride_kv_s=skv[0],
                stride_kv_d=skv[1],
                stride_w_s=sw[0],
                stride_w_h=sw[1],
                stride_logits_s=sl[0],
                stride_logits_k=sl[1],
                BLOCK_KV=64,
                NUM_WARPS=2,
                NUM_BUFFERS=2,
                NUM_CHAINS=1,
                USE_BUFFER_LOAD=True,
                USE_BUFFER_STORE=True,
                num_warps=2,
                waves_per_eu=2,
                USE_PADDED_SHARED_LAYOUT=M.ASYNC_COPY_SUPPORTS_DISTRIBUTED,
                BLOCK_M=1,
                MFMA_NONK_DIM=32,
                M_CHUNK=32,
                UNROLL=2,
                RELAXED_STORE=0,
                HAS_KV_SPLIT=1 if nk > 1 else 0,
                num_kv_splits=nk,
            )
            iso["launch_alone"] = batch_host_us(lambda: K[grid](**kw))
            iso["launch_alone_grid0"] = batch_host_us(lambda: K[(0, nk)](**kw))
        if side == "intj":
            proto = make_proto(M)
            assert torch.equal(proto(q, kv, s, w, ks, ke, True), ref)
            iso["proto_cached_cfg"] = batch_host_us(
                lambda: proto(q, kv, s, w, ks, ke, True)
            )
            iso["proto_cached_cfg_clean_false"] = batch_host_us(
                lambda: proto(q, kv, s, w, ks, ke, False)
            )
        r["isolated_us"] = iso
        r["config"] = {
            "grid": list(grid),
            "num_kv_splits": nk,
            "M_CHUNK_expected": 32,
            "USE_PADDED_SHARED_LAYOUT": M.ASYNC_COPY_SUPPORTS_DISTRIBUTED,
            "FOLDED_REDUCTED_SUPPORT": M.FOLDED_REDUCTED_SUPPORT,
        }
        print(json.dumps(r), flush=True)
        del rec


if __name__ == "__main__":
    main()

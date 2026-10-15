# Performance data for the deck

Collected 2026-10-15. Every CSV here is produced by `ppt/scripts/aggregate.py` from `raw/`.
A value is the median over 3 separate processes of each process's own median; `lo`/`hi`
are the smallest and largest process median.

## Files by slide

| Slide | Files |
|---|---|
| Benchmark results (headline) | `headline_launch.csv`, `headline_decode_by_access_mode.csv` |
| Varying argument number | `sweep_args_wide.csv` (long form: `sweep_args.csv`) |
| Varying constexpr-argument number | `sweep_constexpr_wide.csv` (`sweep_constexpr.csv`) |
| Varying cached-kernel number | `sweep_cache_wide.csv` (`sweep_cache.csv`) |
| Comparing with TVM FFI | `hip_module_same_hsaco.csv`, `ffi_compare.csv`, `ffi_sweep.csv` |
| AITER KDA / fp8_mqa_logits | `aiter_summary.csv`; per-op detail `aiter_flash_kda.csv`, `aiter_kda_decode.csv`, `aiter_fp8_mqa_logits.csv` |

## Key numbers

| | MI308X (gfx942) | MI350X (gfx950) | RTX 4080 SUPER (sm_89) |
|---|---|---|---|
| launch incl. driver, Triton → intj | 16.56 → 3.45 us | 10.96 → 3.14 us | 8.10 → 1.31 us |
| host only (`grid=(0,)`), Triton → intj | 12.9 → 0.03 us | 8.17 → 0.02 us | 6.11 → 0.02 us |
| host only, 0 → 64 tensor args, Triton | 4.9 → 129.2 us | 3.5 → 87.5 us | 4.1 → 27.7 us |
| host only, 0 → 64 tensor args, intj | 0.022 → 0.298 us | 0.014 → 0.130 us | 0.018 → 0.123 us |
| host only, 1 → 1024 cached kernels, intj | 0.035 → 0.046 us | 0.022 → 0.030 us | 0.024 → 0.027 us |
| same HSACO: Triton / TVM FFI / intj | 15.69 / 3.34 / 3.12 us | 10.77 / 3.05 / 3.00 us | n/a (ROCm-only bench) |
| empty kernel, 3 tensors: TVM FFI / intj | 3.94 / 3.43 us | 2.89 / 2.98 us | 1.51 / 1.30 us |

AITER on MI350X, upstream → intj, outputs bitwise equal on every shape:

| Op | Host time per call | Eager time per call (upstream host-bound shapes) | GPU kernels |
|---|---|---|---|
| fused KDA decode, batch 1-16 | 31.8 → 7.0 us | 32.6-33.1 → 12.6-14.0 us (2.4-2.6x) | 13-15 us, same |
| FlashKDA, T 64-256 (Gluon) | 96-97 (Triton) / 50-52 (AITER `_FastLaunch`) → 27 us | 96-98 / 50-52 → 27-29 us | 18-29 us, same |
| fp8_mqa_logits, q = kv ≤ 1024 | 35.3-35.5 → 14.8-15.2 us | 35.4-35.7 → 14.9-15.2 us (2.35-2.39x) | 6-17 us, same |

## Machines and software

| | MI308X | MI350X | RTX 4080 SUPER |
|---|---|---|---|
| Host | hjbog-srdc-7 (this checkout) | gbt350 (`ssh -p 30004 jinpli@localhost`) | ljp-arch01, 192.168.1.104 (`ssh -p 30006 lijinpei@localhost`) |
| CPU, pinned core | Xeon Platinum 8480C, core 80 (NUMA 1) | EPYC 9965, core 100 (NUMA 0) | Ryzen 9 7950X3D, core 0 (governor `powersave`) |
| GPU | HIP device 6 (NUMA 1) | HIP device 2 (NUMA 0) | CUDA device 0, driver 615.78.08 |
| Python / torch | 3.12.3 / 2.14.0.dev20260804+rocm7.1 | 3.14.4 / 2.15.0.dev20260816+rocm7.14 | 3.12.15 / 2.14.1+cu130 |
| Triton / apache-tvm-ffi | 3.8.0 / 0.1.14.post1 | 3.8.0 / 0.1.14.post1 | 3.8.0 / 0.1.14.post1 |
| intj default `torch_access_mode` | `STATIC_COMPILE` | `STATIC_COMPILE` (`RUNTIME_SHIM` has no layout for torch 2.15) | `STATIC_COMPILE` |

intj for the microbenchmarks is `05e25a9` (clean tracked tree here; on the other two a
`git archive` of it). The NVIDIA box has no PyPI access: its apache-tvm-ffi is the
cp312 manylinux wheel installed here with `uv pip install --target` and copied over. The AITER comparison is upstream `874d253dd3` (the migration's base,
on ROCm/aiter `main`) against the migration branch `5d710e670`
(`lijinpei-amd/codex/intj-all-launches`) with the intj it requires, `20e578f`. Both
AITER trees were verified file-for-file against those commits.
`raw/<machine>/env.txt` has the full record.

## What each path means

Sweeps (`bench_sweeps.py`; empty `@triton.jit` kernels):

- `triton_launch` / `intj_launch`: `kernel[(1,)](*args)` / `launcher(dev, stream, (1,), *args)`,
  includes the driver launch. Batches of 16 calls, synchronized outside the timer.
- `triton_host`: `kernel[(0,)](*args)`: Triton binds, keys and looks up, then skips the
  driver call on the empty grid.
- `intj_host`: a `no_gpu=True` launcher on CPU tensors: decode, key and cache lookup.
  (intj's GPU launcher returns before decoding on an empty grid, so `(0,)` would flatter it.)
- `args`: 0-64 parameters, all `int` or all tensor. `constexpr`: 4 tensors + 0-32
  `tl.constexpr` ints. `cache`: 4 tensors + one constexpr taking K values in a fixed
  random order, so K kernels are cached and hit at random.

AITER (`bench_aiter_ops.py`), per shape after warm-up:

- `host_us`: host time per eager call, batches of 8 calls.
- `eager_us`: per call over 32 back-to-back calls, timed by GPU events: about
  max(host, GPU). `aiter_summary.csv` gives its speedup only where upstream is host-bound.
- `do_bench_us`: AITER's own metric, `triton.testing.do_bench` (L2 flushed per call; the
  flush mostly hides host time, so it tracks GPU time).
- `gpu_kernel_us`: summed kernel time per call from one `rocprofv3 --kernel-trace` run
  of 50 calls per shape.
- `launch_path`: `intj`; `base_triton_jit`: upstream Triton `JITFunction` launch;
  `base_fastlaunch`: FlashKDA's upstream `_FastLaunch`, AITER's hand-written Python
  launch cache (its `base_triton_jit` rows bypass it). FlashKDA `route=triton` sets
  `AITER_FDA_USE_GLUON=0`.

## Caveats

- **gfx950 launch jitter.** `hipModuleLaunchKernel` on the MI350 flips between ~2.3 and
  ~3.1 us within and between processes, on both sides, so `*_launch` rows there move by
  ±0.7 us (`sweep_cache_wide.csv`, 32-128 kernels). Host-only rows don't have this.
- **MI350 eager stalls.** Long eager queues on GPU-bound shapes intermittently stall in
  the driver (a blocking wait wakes on a 4 ms tick). Those rows carry
  `eager_flag=driver_stall`; their kernel times are identical on both sides. Evidence in
  `raw/mi350x/aiter_eager_stall_notes.txt`. Two earlier AITER passes on the MI350 with
  host-timed loops hit this harder and were discarded; only the event-timed pass is in `raw/`.
- **FFI.** In `ffi_compare.csv` and `ffi_sweep.csv`, FFI and intj launch different
  binaries, and FFI gets preconverted `TensorView`s. `hip_module_same_hsaco.csv` is the
  same-kernel comparison. At 0 arguments FFI's empty kernel takes a different stream
  path (cheaper). On the MI350, `ffi_sweep` ran through `bench_ffi_sweep_tolerant.py`
  (no `RUNTIME_SHIM` column). The MI308X FFI benches were rerun (12:05) because the
  first round 0 overlapped a smoke test of mine on the same host.
- The `headline_decode_by_access_mode.csv` `INTERPRETER` row (~920 ns on MI308X) is
  3× the README's 300 ns, which came from an older MI300X run; the 2026-10-14 journal
  agrees with ~900 ns.
- Not measured: AITER on gfx942 or NVIDIA, CUDA-graph replay.

## Regenerate

```sh
bash ppt/scripts/mi308_all.sh /tmp/out/mi308     # here: readme, sweeps, FFI (3 rounds)
# on the MI350, under ~/development/workspace/intj-ppt (src/ = git archive of intj):
setsid nohup scripts/mi350_all.sh out/mi350 &    # same, plus AITER
scripts/mi350_rerun.sh out/mi350                 # tolerant FFI sweep (+ AITER pass 2)
scripts/mi350_aiter_v3.sh out/mi350              # event-timed AITER pass (the one kept)
# on the NVIDIA box, from ~/intj-ppt/src after `. ../env.sh`:
setsid nohup bash ../scripts/nvidia_all.sh ~/intj-ppt/out/rtx4080 &
python3 ppt/scripts/aggregate.py                 # after copying outputs into raw/<machine>/
```

# 12-case gate: assume_constant_globals + tensor-only annotations (index 0)

Candidate `c7125b1` ("Constrain torch.Tensor / tl.tensor parameters to tensors"), on top of `a04c66a`
("Accept global-reading kernels behind assume_constant_globals=True"). `a04c66a` adds no rendered code (a
`ModuleKey` field and a make_launcher keyword); `c7125b1` adds a tensor-only branch to both decode paths of
`entry.c.jinja`, emitted only for `torch.Tensor` / `tl.tensor` parameters, and splits `intj_decode_argument`'s
tensor half into `intj_decode_tensor_checked` in `intj_runtime.h` (same code, always-inlined). Kernels without
those annotations render the same entry as before. Also adds three `bench_launch.py` rows (`tensor annotated`, `generic`, `generic tensor ann`).

## Environment

- Candidate `c7125b1` (branch `develop`, clean checkout) at
  `/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj`. A/B parent: `9075eaa` (immediate parent of
  `a04c66a`), a detached worktree at `/tmp/globals-tensor-parent`. The task named `13aec07`; `9075eaa` sits
  between it and this work and changed the typed-scalar decode, so it is the parent that isolates these two
  commits.
- Gate baseline: `interp-refusal` (`839a0e0`, journal `2026-09-28_interp-refusal_0_839a0e0.md`).
- CPU: `taskset -c 0`, one process at a time. Shared host: load average 7.62/9.01/10.95 before the gate,
  7.51/9.41/10.57 after; 5.63 -> 5.92 around the rerun; 4.92 -> 4.30 around the A/B.
- GPU: AMD Instinct MI308X (`gfx942:sramecc+:xnack-`), GPU 0 (`HIP_VISIBLE_DEVICES=0`).
- Python 3.12.3 (`/tmp/gb2/bin/python`), Torch `2.14.0+rocm7.2`, Triton `3.8.0`, cc 13.3.0.
- `INTJ_BENCHMARK_ROOT=/tmp/gbench`. CUDA: untested.

## Commands

```bash
# 1. the 12-case gate, 3 rounds, core 0 (run_all.sh: see /tmp/intj-bench/run_all.sh)
HIP_VISIBLE_DEVICES=0 /tmp/intj-bench/run_all.sh globals-tensor
/tmp/gb2/bin/python /tmp/intj-bench/bench_compare.py interp-refusal globals-tensor

# 2. rerun of every case with a flagged row (launch_gpu, ffi_compare, ffi_sweep, hip_module), 3 rounds
/tmp/intj-bench/globals-tensor-rerun.sh     # same commands as run_all.sh, -> globals-tensor-rerun/
#    compared with the 8 unflagged cases taken from the gate (bench_compare needs all 12 files)
/tmp/gb2/bin/python /tmp/intj-bench/bench_compare.py interp-refusal globals-tensor-rerun-merged

# 3. still flagged (ffi_compare): interleaved A/B, 5 processes each, parent 9075eaa vs candidate c7125b1
for r in 0 1 2 3 4; do for rev in parent cand; do
  cd <tree of rev>; HIP_VISIBLE_DEVICES=0 PYTHONPATH=<tree> INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 \
    /tmp/gb2/bin/python benchmarks/bench_ffi_compare.py --iters 1000 --batches 9 > globals-tensor-ab/${rev}_round$r.txt
done; done
```

## Result

**No regression.** Every host (ns) row of `launch_host`, `launch_sweep`, `launch_last_key`, `launch_dynamic`,
`launch_tuned`, `launch_readme` and `kernel_cache` passes the rule (>3 % and >2 ns / 0.1 us) against
`interp-refusal`.

- **Gate flags (all GPU-launch rows, us or driver-inclusive ns):** 7 `launch_gpu` rows (+4 to +21 %),
  `ffi_compare`'s 6 kernel rows, 7 `ffi_sweep` "FFI empty kernel" rows and `hip_module` "TVM FFI HIP HSACO".
  The FFI rows are TVM FFI's own launches, which this change cannot touch; they moved with INTJ's.
- **Rerun (3 rounds):** `launch_gpu`, `ffi_sweep`, `hip_module` all pass (`launch_gpu` -5.6 to +0.8 %).
  `ffi_compare` still flags all six kernel rows, FFI's own included (FFI empty +10.1 %, INTJ empty +18.2 %).
- **Interleaved A/B (`ffi_compare`, 5+5 processes):** parent and candidate agree within ±1 % on every row, and
  both sit at the `interp-refusal` baseline's level (INTJ empty 2.995 vs 2.990 us; baseline 2.972). The gate's
  and rerun's ~3.5 us came from a slow GPU/host window, not from these commits. Candidate round 1 is one slow
  process (every kernel row +0.6 us together); it does not move the medians.

**Tensor-only decode, host-only micro-benchmark (`launch_host`, 3 rounds, 100000 calls x 9 batches):**

| row | rounds (ns/call) | median |
|---|---|---|
| auto map (3 unannotated tensors) | 40.2, 40.0, 40.1 | 40.1 |
| tensor annotated (`torch.Tensor` on x, y, o) | 40.0, 40.1, 40.1 | 40.1 |
| generic (`n: Argument(type=tl.int32)`) | 44.4, 44.7, 44.0 | 44.4 |
| generic tensor ann (same + `torch.Tensor` on x, y, o) | 44.4, 44.5, 44.3 | 44.4 |

No measurable saving on either path: both decoders already test the exact tensor type first, so a tensor
argument never reached the scalar branches the annotation removes. What changes is the non-tensor case (a
`TypeError` instead of a scalar or constexpr `None`), not the tensor hit path's cost.

### ffi_compare A/B summary

| row | parent 9075eaa rounds (us) | median | candidate c7125b1 rounds (us) | median | delta |
|---|---|---|---|---|---|
| FFI packed nop | 0.1451, 0.1442, 0.1445, 0.1454, 0.1449 | 0.1449 | 0.1449, 0.1446, 0.1440, 0.1476, 0.1435 | 0.1446 | -0.2% |
| FFI typed nop | 0.1492, 0.1493, 0.1495, 0.1493, 0.1485 | 0.1493 | 0.1497, 0.1496, 0.1501, 0.1487, 0.1506 | 0.1497 | +0.3% |
| FFI empty kernel | 3.2894, 3.3371, 3.3421, 3.3356, 3.3405 | 3.3371 | 3.3178, 3.7573, 3.3561, 3.3186, 3.3146 | 3.3186 | -0.6% |
| INTJ empty kernel | 2.9137, 2.9684, 2.9949, 3.0102, 3.0037 | 2.9949 | 2.9648, 3.6388, 2.9967, 2.9895, 2.9836 | 2.9895 | -0.2% |
| INTJ fixed-device kernel | 2.8936, 2.8533, 2.8971, 2.8852, 2.8201 | 2.8852 | 2.8252, 3.4969, 2.9092, 2.9025, 2.9102 | 2.9092 | +0.8% |
| FFI packed nop mixed | 0.1735, 0.1734, 0.1748, 0.1744, 0.1734 | 0.1735 | 0.1733, 0.1732, 0.1749, 0.1728, 0.1734 | 0.1733 | -0.1% |
| FFI typed nop mixed | 0.1778, 0.1769, 0.1781, 0.1768, 0.1764 | 0.1769 | 0.1756, 0.1744, 0.1772, 0.1774, 0.1765 | 0.1765 | -0.2% |
| FFI mixed kernel | 3.3143, 3.2678, 3.3417, 3.3394, 3.2892 | 3.3143 | 3.3142, 3.7330, 3.2674, 3.3117, 3.2806 | 3.3117 | -0.1% |
| INTJ mixed kernel | 2.9533, 2.9590, 2.9115, 2.9675, 2.9208 | 2.9533 | 2.9461, 3.5908, 2.8919, 2.9554, 2.9603 | 2.9554 | +0.1% |
| INTJ fixed mixed kernel | 2.8894, 2.9157, 2.8341, 2.8924, 2.9232 | 2.8924 | 2.8965, 3.5909, 2.9208, 2.9387, 2.9107 | 2.9208 | +1.0% |

## Raw output

### Gate comparison (`bench_compare.py interp-refusal globals-tensor`)

```
ffi_compare      median           FFI empty kernel                 value    baseline=3.3367 new=3.6843 delta=0.3476 pct=+10.4% (us) REGRESSION
ffi_compare      median           FFI mixed kernel                 value    baseline=3.3396 new=3.8761 delta=0.5365 pct=+16.1% (us) REGRESSION
ffi_compare      median           FFI packed nop                   value    baseline=0.1446 new=0.1446 delta=0.0000 pct=+0.0% (us)
ffi_compare      median           FFI packed nop mixed             value    baseline=0.1744 new=0.1730 delta=-0.0014 pct=-0.8% (us)
ffi_compare      median           FFI typed nop                    value    baseline=0.1498 new=0.1492 delta=-0.0006 pct=-0.4% (us)
ffi_compare      median           FFI typed nop mixed              value    baseline=0.1772 new=0.1762 delta=-0.0010 pct=-0.6% (us)
ffi_compare      median           INTJ empty kernel                value    baseline=2.9716 new=3.5046 delta=0.5330 pct=+17.9% (us) REGRESSION
ffi_compare      median           INTJ fixed mixed kernel          value    baseline=2.9033 new=3.5187 delta=0.6154 pct=+21.2% (us) REGRESSION
ffi_compare      median           INTJ fixed-device kernel         value    baseline=2.8906 new=3.3623 delta=0.4717 pct=+16.3% (us) REGRESSION
ffi_compare      median           INTJ mixed kernel                value    baseline=2.9779 new=3.5657 delta=0.5878 pct=+19.7% (us) REGRESSION
ffi_paths        median_ns        FFI unpack Pair only             value    baseline=167.9 new=166.1 delta=-1.8000 pct=-1.1% (ns)
ffi_paths        median_ns        INTJ 3-tensor host-only nop      value    baseline=32.3000 new=32.4000 delta=0.1000 pct=+0.3% (ns)
ffi_paths        median_ns        INTJ FFI wrapper kwargs          value    baseline=123.4 new=123.4 delta=0.0000 pct=+0.0% (ns)
ffi_paths        median_ns        INTJ FFI wrapper positional      value    baseline=100.6 new=100.4 delta=-0.2000 pct=-0.2% (ns)
ffi_paths        median_ns        INTJ adapter defaults            value    baseline=86.2000 new=85.8000 delta=-0.4000 pct=-0.5% (ns)
ffi_paths        median_ns        INTJ adapter kwargs              value    baseline=103.4 new=103.8 delta=0.4000 pct=+0.4% (ns)
ffi_paths        median_ns        INTJ adapter positional          value    baseline=87.6000 new=87.0000 delta=-0.6000 pct=-0.7% (ns)
ffi_paths        median_ns        INTJ cold compile callback + cache value    baseline=712.0 new=701.3 delta=-10.7000 pct=-1.5% (ns)
ffi_paths        median_ns        INTJ config FFI unpack           value    baseline=324.4 new=325.3 delta=0.9000 pct=+0.3% (ns)
ffi_paths        median_ns        INTJ config direct               value    baseline=78.8000 new=80.0000 delta=1.2000 pct=+1.5% (ns)
ffi_paths        median_ns        INTJ direct positional           value    baseline=63.2000 new=63.0000 delta=-0.2000 pct=-0.3% (ns)
ffi_paths        median_ns        INTJ hot call (no callback)      value    baseline=35.2000 new=35.5000 delta=0.3000 pct=+0.9% (ns)
ffi_paths        median_ns        INTJ pair FFI unpack             value    baseline=302.5 new=305.0 delta=2.5000 pct=+0.8% (ns)
ffi_paths        median_ns        INTJ pair direct                 value    baseline=65.6000 new=66.3000 delta=0.7000 pct=+1.1% (ns)
ffi_paths        median_ns        INTJ pair manual unpack          value    baseline=165.8 new=164.9 delta=-0.9000 pct=-0.5% (ns)
ffi_paths        median_ns        INTJ pair prebuilt *tuple        value    baseline=139.6 new=138.9 delta=-0.7000 pct=-0.5% (ns)
ffi_paths        median_ns        INTJ pair stdlib astuple         value    baseline=1165.0 new=1176.2 delta=11.2000 pct=+1.0% (ns)
ffi_sweep        median           args=0 FFI empty kernel          value    baseline=1.6925 new=2.2218 delta=0.5293 pct=+31.3% (us) REGRESSION
ffi_sweep        median           args=0 FFI packed nop            value    baseline=0.1168 new=0.1169 delta=0.0001 pct=+0.1% (us)
ffi_sweep        median           args=0 FFI typed nop             value    baseline=0.1195 new=0.1189 delta=-0.0006 pct=-0.5% (us)
ffi_sweep        median           args=16 FFI empty kernel         value    baseline=3.6224 new=4.6850 delta=1.0626 pct=+29.3% (us) REGRESSION
ffi_sweep        median           args=16 FFI packed nop           value    baseline=0.2913 new=0.2919 delta=0.0006 pct=+0.2% (us)
ffi_sweep        median           args=16 FFI typed nop            value    baseline=0.3037 new=0.3035 delta=-0.0002 pct=-0.1% (us)
ffi_sweep        median           args=3 FFI empty kernel          value    baseline=3.2137 new=3.7568 delta=0.5431 pct=+16.9% (us) REGRESSION
ffi_sweep        median           args=3 FFI packed nop            value    baseline=0.1450 new=0.1451 delta=0.0001 pct=+0.1% (us)
ffi_sweep        median           args=3 FFI typed nop             value    baseline=0.1493 new=0.1484 delta=-0.0009 pct=-0.6% (us)
ffi_sweep        median           args=32 FFI empty kernel         value    baseline=4.1099 new=4.8751 delta=0.7652 pct=+18.6% (us) REGRESSION
ffi_sweep        median           args=32 FFI packed nop           value    baseline=0.4806 new=0.4803 delta=-0.0003 pct=-0.1% (us)
ffi_sweep        median           args=32 FFI typed nop            value    baseline=0.4944 new=0.4943 delta=-0.0001 pct=-0.0% (us)
ffi_sweep        median           args=5 FFI empty kernel          value    baseline=3.2477 new=3.8319 delta=0.5842 pct=+18.0% (us) REGRESSION
ffi_sweep        median           args=5 FFI packed nop            value    baseline=0.1740 new=0.1741 delta=0.0001 pct=+0.1% (us)
ffi_sweep        median           args=5 FFI typed nop             value    baseline=0.1779 new=0.1774 delta=-0.0005 pct=-0.3% (us)
ffi_sweep        median           args=64 FFI empty kernel         value    baseline=5.0757 new=5.9270 delta=0.8513 pct=+16.8% (us) REGRESSION
ffi_sweep        median           args=64 FFI packed nop           value    baseline=0.8451 new=0.8383 delta=-0.0068 pct=-0.8% (us)
ffi_sweep        median           args=64 FFI typed nop            value    baseline=0.8642 new=0.8598 delta=-0.0044 pct=-0.5% (us)
ffi_sweep        median           args=8 FFI empty kernel          value    baseline=3.3744 new=3.9806 delta=0.6062 pct=+18.0% (us) REGRESSION
ffi_sweep        median           args=8 FFI packed nop            value    baseline=0.2061 new=0.2064 delta=0.0003 pct=+0.1% (us)
ffi_sweep        median           args=8 FFI typed nop             value    baseline=0.2098 new=0.2106 delta=0.0008 pct=+0.4% (us)
hip_module       median           INTJ same function               value    baseline=2.9253 new=2.9775 delta=0.0522 pct=+1.8% (us)
hip_module       median           TVM FFI HIP HSACO                value    baseline=3.1619 new=3.2768 delta=0.1149 pct=+3.6% (us) REGRESSION
hip_module       median           Triton same HSACO                value    baseline=16.0663 new=15.9969 delta=-0.0694 pct=-0.4% (us)
kernel_cache     1-word key       hash_only                        absl     baseline=1.3600 new=1.3600 delta=0.0000 pct=+0.0% (ns)
kernel_cache     1-word key       hash_only                        intj     baseline=1.3600 new=1.3600 delta=0.0000 pct=+0.0% (ns)
kernel_cache     1-word key       hash_only                        tsl      baseline=1.3600 new=1.3500 delta=-0.0100 pct=-0.7% (ns)
kernel_cache     1-word key       hit/1                            absl     baseline=1.4400 new=1.4400 delta=0.0000 pct=+0.0% (ns)
kernel_cache     1-word key       hit/1                            intj     baseline=1.5600 new=1.5800 delta=0.0200 pct=+1.3% (ns)
kernel_cache     1-word key       hit/1                            tsl      baseline=2.0100 new=2.0200 delta=0.0100 pct=+0.5% (ns)
kernel_cache     1-word key       hit/512                          absl     baseline=5.3100 new=5.3000 delta=-0.0100 pct=-0.2% (ns)
kernel_cache     1-word key       hit/512                          intj     baseline=2.3000 new=2.3000 delta=0.0000 pct=+0.0% (ns)
kernel_cache     1-word key       hit/512                          tsl      baseline=2.9300 new=2.9000 delta=-0.0300 pct=-1.0% (ns)
kernel_cache     1-word key       hit/64                           absl     baseline=4.5400 new=4.5400 delta=0.0000 pct=+0.0% (ns)
kernel_cache     1-word key       hit/64                           intj     baseline=1.6600 new=1.6500 delta=-0.0100 pct=-0.6% (ns)
kernel_cache     1-word key       hit/64                           tsl      baseline=2.1300 new=2.1400 delta=0.0100 pct=+0.5% (ns)
kernel_cache     1-word key       hit/8                            absl     baseline=4.3600 new=4.3700 delta=0.0100 pct=+0.2% (ns)
kernel_cache     1-word key       hit/8                            intj     baseline=1.6500 new=2.3400 delta=0.6900 pct=+41.8% (ns)
kernel_cache     1-word key       hit/8                            tsl      baseline=2.1000 new=2.1000 delta=0.0000 pct=+0.0% (ns)
kernel_cache     1-word key       hit_child                        absl     baseline=3.0100 new=3.0000 delta=-0.0100 pct=-0.3% (ns)
kernel_cache     1-word key       hit_child                        intj     baseline=3.0100 new=3.0000 delta=-0.0100 pct=-0.3% (ns)
kernel_cache     1-word key       hit_child                        tsl      baseline=3.0000 new=3.0000 delta=0.0000 pct=+0.0% (ns)
kernel_cache     1-word key       miss/1                           absl     baseline=1.3900 new=1.3800 delta=-0.0100 pct=-0.7% (ns)
kernel_cache     1-word key       miss/1                           intj     baseline=1.2200 new=1.2200 delta=0.0000 pct=+0.0% (ns)
kernel_cache     1-word key       miss/1                           tsl      baseline=1.5300 new=1.5700 delta=0.0400 pct=+2.6% (ns)
kernel_cache     1-word key       miss/512                         absl     baseline=4.0600 new=4.6500 delta=0.5900 pct=+14.5% (ns)
kernel_cache     1-word key       miss/512                         intj     baseline=2.9400 new=2.9500 delta=0.0100 pct=+0.3% (ns)
kernel_cache     1-word key       miss/512                         tsl      baseline=1.9900 new=2.0000 delta=0.0100 pct=+0.5% (ns)
kernel_cache     1-word key       miss/64                          absl     baseline=3.5300 new=3.5300 delta=0.0000 pct=+0.0% (ns)
kernel_cache     1-word key       miss/64                          intj     baseline=2.7900 new=2.7700 delta=-0.0200 pct=-0.7% (ns)
kernel_cache     1-word key       miss/64                          tsl      baseline=1.7400 new=2.3600 delta=0.6200 pct=+35.6% (ns)
kernel_cache     1-word key       miss/8                           absl     baseline=3.2800 new=3.2800 delta=0.0000 pct=+0.0% (ns)
kernel_cache     1-word key       miss/8                           intj     baseline=2.7300 new=2.6600 delta=-0.0700 pct=-2.6% (ns)
kernel_cache     1-word key       miss/8                           tsl      baseline=1.7600 new=1.7700 delta=0.0100 pct=+0.6% (ns)
kernel_cache     2-word key       hash_only                        absl     baseline=1.4200 new=1.4200 delta=0.0000 pct=+0.0% (ns)
kernel_cache     2-word key       hash_only                        intj     baseline=1.4200 new=1.4200 delta=0.0000 pct=+0.0% (ns)
kernel_cache     2-word key       hash_only                        tsl      baseline=1.4200 new=1.4200 delta=0.0000 pct=+0.0% (ns)
kernel_cache     2-word key       hit/1                            absl     baseline=2.0200 new=2.0000 delta=-0.0200 pct=-1.0% (ns)
kernel_cache     2-word key       hit/1                            intj     baseline=1.8400 new=1.9200 delta=0.0800 pct=+4.3% (ns)
kernel_cache     2-word key       hit/1                            tsl      baseline=2.5700 new=2.5700 delta=0.0000 pct=+0.0% (ns)
kernel_cache     2-word key       hit/512                          absl     baseline=6.3800 new=6.3900 delta=0.0100 pct=+0.2% (ns)
kernel_cache     2-word key       hit/512                          intj     baseline=3.0300 new=3.0200 delta=-0.0100 pct=-0.3% (ns)
kernel_cache     2-word key       hit/512                          tsl      baseline=4.0400 new=4.0400 delta=0.0000 pct=+0.0% (ns)
kernel_cache     2-word key       hit/64                           absl     baseline=5.2300 new=5.2200 delta=-0.0100 pct=-0.2% (ns)
kernel_cache     2-word key       hit/64                           intj     baseline=2.2500 new=2.2100 delta=-0.0400 pct=-1.8% (ns)
kernel_cache     2-word key       hit/64                           tsl      baseline=2.9200 new=2.9200 delta=0.0000 pct=+0.0% (ns)
kernel_cache     2-word key       hit/8                            absl     baseline=5.0400 new=5.0400 delta=0.0000 pct=+0.0% (ns)
kernel_cache     2-word key       hit/8                            intj     baseline=1.8300 new=1.8500 delta=0.0200 pct=+1.1% (ns)
kernel_cache     2-word key       hit/8                            tsl      baseline=2.5600 new=2.5700 delta=0.0100 pct=+0.4% (ns)
kernel_cache     2-word key       hit_child                        absl     baseline=3.0100 new=3.0100 delta=0.0000 pct=+0.0% (ns)
kernel_cache     2-word key       hit_child                        intj     baseline=3.0000 new=3.0000 delta=0.0000 pct=+0.0% (ns)
kernel_cache     2-word key       hit_child                        tsl      baseline=3.0100 new=3.0100 delta=0.0000 pct=+0.0% (ns)
kernel_cache     2-word key       miss/1                           absl     baseline=1.4800 new=1.4900 delta=0.0100 pct=+0.7% (ns)
kernel_cache     2-word key       miss/1                           intj     baseline=1.3800 new=1.3800 delta=0.0000 pct=+0.0% (ns)
kernel_cache     2-word key       miss/1                           tsl      baseline=2.3200 new=2.9800 delta=0.6600 pct=+28.4% (ns)
kernel_cache     2-word key       miss/512                         absl     baseline=3.9900 new=4.0000 delta=0.0100 pct=+0.3% (ns)
kernel_cache     2-word key       miss/512                         intj     baseline=2.8800 new=2.8800 delta=0.0000 pct=+0.0% (ns)
kernel_cache     2-word key       miss/512                         tsl      baseline=2.7200 new=2.7200 delta=0.0000 pct=+0.0% (ns)
kernel_cache     2-word key       miss/64                          absl     baseline=3.8900 new=4.3900 delta=0.5000 pct=+12.9% (ns)
kernel_cache     2-word key       miss/64                          intj     baseline=3.1600 new=3.1300 delta=-0.0300 pct=-0.9% (ns)
kernel_cache     2-word key       miss/64                          tsl      baseline=2.1000 new=2.1100 delta=0.0100 pct=+0.5% (ns)
kernel_cache     2-word key       miss/8                           absl     baseline=3.2000 new=3.2100 delta=0.0100 pct=+0.3% (ns)
kernel_cache     2-word key       miss/8                           intj     baseline=2.3800 new=2.4500 delta=0.0700 pct=+2.9% (ns)
kernel_cache     2-word key       miss/8                           tsl      baseline=1.8600 new=1.8600 delta=0.0000 pct=+0.0% (ns)
kernel_cache     5-word key       hash_only                        absl     baseline=3.5800 new=3.5900 delta=0.0100 pct=+0.3% (ns)
kernel_cache     5-word key       hash_only                        intj     baseline=3.6000 new=3.5800 delta=-0.0200 pct=-0.6% (ns)
kernel_cache     5-word key       hash_only                        tsl      baseline=3.5800 new=3.5900 delta=0.0100 pct=+0.3% (ns)
kernel_cache     5-word key       hit/1                            absl     baseline=2.4800 new=2.4600 delta=-0.0200 pct=-0.8% (ns)
kernel_cache     5-word key       hit/1                            intj     baseline=3.0000 new=3.0000 delta=0.0000 pct=+0.0% (ns)
kernel_cache     5-word key       hit/1                            tsl      baseline=3.0600 new=3.0600 delta=0.0000 pct=+0.0% (ns)
kernel_cache     5-word key       hit/512                          absl     baseline=11.1800 new=11.2200 delta=0.0400 pct=+0.4% (ns)
kernel_cache     5-word key       hit/512                          intj     baseline=4.5300 new=4.5500 delta=0.0200 pct=+0.4% (ns)
kernel_cache     5-word key       hit/512                          tsl      baseline=4.7800 new=4.7700 delta=-0.0100 pct=-0.2% (ns)
kernel_cache     5-word key       hit/64                           absl     baseline=9.7800 new=9.7800 delta=0.0000 pct=+0.0% (ns)
kernel_cache     5-word key       hit/64                           intj     baseline=3.2700 new=3.2800 delta=0.0100 pct=+0.3% (ns)
kernel_cache     5-word key       hit/64                           tsl      baseline=3.5100 new=3.5100 delta=0.0000 pct=+0.0% (ns)
kernel_cache     5-word key       hit/8                            absl     baseline=9.4900 new=9.4800 delta=-0.0100 pct=-0.1% (ns)
kernel_cache     5-word key       hit/8                            intj     baseline=3.2000 new=3.2000 delta=0.0000 pct=+0.0% (ns)
kernel_cache     5-word key       hit/8                            tsl      baseline=3.4100 new=3.4100 delta=0.0000 pct=+0.0% (ns)
kernel_cache     5-word key       hit_child                        absl     baseline=3.0000 new=3.0000 delta=0.0000 pct=+0.0% (ns)
kernel_cache     5-word key       hit_child                        intj     baseline=3.0100 new=3.0000 delta=-0.0100 pct=-0.3% (ns)
kernel_cache     5-word key       hit_child                        tsl      baseline=3.0100 new=3.0100 delta=0.0000 pct=+0.0% (ns)
kernel_cache     5-word key       miss/1                           absl     baseline=2.1100 new=2.2400 delta=0.1300 pct=+6.2% (ns)
kernel_cache     5-word key       miss/1                           intj     baseline=1.3900 new=1.4200 delta=0.0300 pct=+2.2% (ns)
kernel_cache     5-word key       miss/1                           tsl      baseline=2.1000 new=2.1000 delta=0.0000 pct=+0.0% (ns)
kernel_cache     5-word key       miss/512                         absl     baseline=7.6900 new=7.7000 delta=0.0100 pct=+0.1% (ns)
kernel_cache     5-word key       miss/512                         intj     baseline=3.4800 new=3.4900 delta=0.0100 pct=+0.3% (ns)
kernel_cache     5-word key       miss/512                         tsl      baseline=2.5100 new=2.5100 delta=0.0000 pct=+0.0% (ns)
kernel_cache     5-word key       miss/64                          absl     baseline=6.7200 new=6.7400 delta=0.0200 pct=+0.3% (ns)
kernel_cache     5-word key       miss/64                          intj     baseline=2.8700 new=2.8600 delta=-0.0100 pct=-0.3% (ns)
kernel_cache     5-word key       miss/64                          tsl      baseline=1.9700 new=1.9700 delta=0.0000 pct=+0.0% (ns)
kernel_cache     5-word key       miss/8                           absl     baseline=6.9400 new=6.9400 delta=0.0000 pct=+0.0% (ns)
kernel_cache     5-word key       miss/8                           intj     baseline=2.9900 new=3.0100 delta=0.0200 pct=+0.7% (ns)
kernel_cache     5-word key       miss/8                           tsl      baseline=2.2400 new=2.2200 delta=-0.0200 pct=-0.9% (ns)
launch_dynamic   dynamic          2 dynamic options                launch   baseline=3015.4 new=3048.4 delta=33.0000 pct=+1.1% (ns)
launch_dynamic   dynamic          2 dynamic options                spec_key baseline=98.5000 new=99.8000 delta=1.3000 pct=+1.3% (ns)
launch_dynamic   dynamic          int constexpr                    launch   baseline=2966.1 new=2969.2 delta=3.1000 pct=+0.1% (ns)
launch_dynamic   dynamic          int constexpr                    spec_key baseline=88.4000 new=89.9000 delta=1.5000 pct=+1.7% (ns)
launch_dynamic   dynamic          lazy plain                       launch   baseline=3001.0 new=3034.5 delta=33.5000 pct=+1.1% (ns)
launch_dynamic   dynamic          lazy plain                       spec_key baseline=94.9000 new=95.3000 delta=0.4000 pct=+0.4% (ns)
launch_dynamic   dynamic          str constexpr                    launch   baseline=2977.3 new=3013.8 delta=36.5000 pct=+1.2% (ns)
launch_dynamic   dynamic          str constexpr                    spec_key baseline=92.9000 new=92.1000 delta=-0.8000 pct=-0.9% (ns)
launch_gpu       launch           auto map                         ns/call  baseline=3145.6 new=3151.6 delta=6.0000 pct=+0.2% (ns)
launch_gpu       launch           baked                            ns/call  baseline=2939.1 new=3167.5 delta=228.4 pct=+7.8% (ns) REGRESSION
launch_gpu       launch           bound pointer                    ns/call  baseline=2949.1 new=3124.8 delta=175.7 pct=+6.0% (ns) REGRESSION
launch_gpu       launch           bound tensor                     ns/call  baseline=2998.3 new=3161.1 delta=162.8 pct=+5.4% (ns) REGRESSION
launch_gpu       launch           fixed device map                 ns/call  baseline=2895.3 new=3158.1 delta=262.8 pct=+9.1% (ns) REGRESSION
launch_gpu       launch           fixed device no-map              ns/call  baseline=2893.0 new=3510.1 delta=617.1 pct=+21.3% (ns) REGRESSION
launch_gpu/launch/generic/ns/call                            MISSING in baseline
launch_gpu/launch/generic tensor ann/ns/call                 MISSING in baseline
launch_gpu       launch           reduced key                      ns/call  baseline=3148.4 new=3132.9 delta=-15.5000 pct=-0.5% (ns)
launch_gpu/launch/tensor annotated/ns/call                   MISSING in baseline
launch_gpu       launch           verify off                       ns/call  baseline=3020.7 new=3147.3 delta=126.6 pct=+4.2% (ns) REGRESSION
launch_gpu       launch           verify on                        ns/call  baseline=2933.9 new=3144.8 delta=210.9 pct=+7.2% (ns) REGRESSION
launch_host      launch           auto map                         ns/call  baseline=40.0000 new=40.1000 delta=0.1000 pct=+0.3% (ns)
launch_host      launch           baked                            ns/call  baseline=38.4000 new=38.3000 delta=-0.1000 pct=-0.3% (ns)
launch_host      launch           bound pointer                    ns/call  baseline=42.0000 new=41.9000 delta=-0.1000 pct=-0.2% (ns)
launch_host      launch           bound tensor                     ns/call  baseline=44.1000 new=43.7000 delta=-0.4000 pct=-0.9% (ns)
launch_host      launch           fixed device map                 ns/call  baseline=41.3000 new=41.1000 delta=-0.2000 pct=-0.5% (ns)
launch_host      launch           fixed device no-map              ns/call  baseline=42.7000 new=42.8000 delta=0.1000 pct=+0.2% (ns)
launch_host/launch/generic/ns/call                           MISSING in baseline
launch_host/launch/generic tensor ann/ns/call                MISSING in baseline
launch_host      launch           reduced key                      ns/call  baseline=41.2000 new=41.1000 delta=-0.1000 pct=-0.2% (ns)
launch_host/launch/tensor annotated/ns/call                  MISSING in baseline
launch_host      launch           verify off                       ns/call  baseline=41.3000 new=41.6000 delta=0.3000 pct=+0.7% (ns)
launch_host      launch           verify on                        ns/call  baseline=43.1000 new=43.0000 delta=-0.1000 pct=-0.2% (ns)
launch_last_key  sweep            16 alternate                     ns/call  baseline=64.4000 new=64.3000 delta=-0.1000 pct=-0.2% (ns)
launch_last_key  sweep            16 repeat                        ns/call  baseline=61.1000 new=61.1000 delta=0.0000 pct=+0.0% (ns)
launch_last_key  sweep            32 alternate                     ns/call  baseline=102.7 new=102.7 delta=0.0000 pct=+0.0% (ns)
launch_last_key  sweep            32 repeat                        ns/call  baseline=98.2000 new=98.6000 delta=0.4000 pct=+0.4% (ns)
launch_last_key  sweep            4 alternate                      ns/call  baseline=34.3000 new=34.4000 delta=0.1000 pct=+0.3% (ns)
launch_last_key  sweep            4 repeat                         ns/call  baseline=32.4000 new=32.5000 delta=0.1000 pct=+0.3% (ns)
launch_readme    readme_path      grid=(0,) intj                   us       baseline=0.0300 new=0.0300 delta=0.0000 pct=+0.0% (us)
launch_readme    readme_path      grid=(0,) triton                 us       baseline=13.0300 new=13.2300 delta=0.2000 pct=+1.5% (us)
launch_readme    readme_path      grid=(1,) intj                   us       baseline=3.1500 new=3.1100 delta=-0.0400 pct=-1.3% (us)
launch_readme    readme_path      grid=(1,) triton                 us       baseline=16.4200 new=16.5700 delta=0.1500 pct=+0.9% (us)
launch_readme    readme_torch_access interpreter                      decode ns baseline=897.2 new=900.9 delta=3.7000 pct=+0.4% (ns)
launch_readme    readme_torch_access runtime_shim                     decode ns baseline=93.5000 new=93.3000 delta=-0.2000 pct=-0.2% (ns)
launch_readme    readme_torch_access static_compile                   decode ns baseline=96.8000 new=95.8000 delta=-1.0000 pct=-1.0% (ns)
launch_sweep     sweep            16 int                           ns/call  baseline=67.2000 new=67.1000 delta=-0.1000 pct=-0.1% (ns)
launch_sweep     sweep            16 tensor                        ns/call  baseline=62.8000 new=62.8000 delta=0.0000 pct=+0.0% (ns)
launch_sweep     sweep            32 int                           ns/call  baseline=104.9 new=105.0 delta=0.1000 pct=+0.1% (ns)
launch_sweep     sweep            32 tensor                        ns/call  baseline=106.3 new=108.7 delta=2.4000 pct=+2.3% (ns)
launch_sweep     sweep            4 int                            ns/call  baseline=38.7000 new=38.8000 delta=0.1000 pct=+0.3% (ns)
launch_sweep     sweep            4 tensor                         ns/call  baseline=36.9000 new=36.9000 delta=0.0000 pct=+0.0% (ns)
launch_tuned     tuned            plain                            ns/launch baseline=3043.8 new=3099.8 delta=56.0000 pct=+1.8% (ns)
launch_tuned     tuned            tuned hit                        ns/launch baseline=3036.2 new=3060.0 delta=23.8000 pct=+0.8% (ns)
```

### Rerun comparison (`bench_compare.py interp-refusal globals-tensor-rerun-merged`, flagged cases only)

```
ffi_compare      median           FFI empty kernel                 value    baseline=3.3367 new=3.6737 delta=0.3370 pct=+10.1% (us) REGRESSION
ffi_compare      median           FFI mixed kernel                 value    baseline=3.3396 new=3.7700 delta=0.4304 pct=+12.9% (us) REGRESSION
ffi_compare      median           FFI packed nop                   value    baseline=0.1446 new=0.1447 delta=0.0001 pct=+0.1% (us)
ffi_compare      median           FFI packed nop mixed             value    baseline=0.1744 new=0.1740 delta=-0.0004 pct=-0.2% (us)
ffi_compare      median           FFI typed nop                    value    baseline=0.1498 new=0.1502 delta=0.0004 pct=+0.3% (us)
ffi_compare      median           FFI typed nop mixed              value    baseline=0.1772 new=0.1764 delta=-0.0008 pct=-0.5% (us)
ffi_compare      median           INTJ empty kernel                value    baseline=2.9716 new=3.5133 delta=0.5417 pct=+18.2% (us) REGRESSION
ffi_compare      median           INTJ fixed mixed kernel          value    baseline=2.9033 new=3.4724 delta=0.5691 pct=+19.6% (us) REGRESSION
ffi_compare      median           INTJ fixed-device kernel         value    baseline=2.8906 new=3.3629 delta=0.4723 pct=+16.3% (us) REGRESSION
ffi_compare      median           INTJ mixed kernel                value    baseline=2.9779 new=3.4790 delta=0.5011 pct=+16.8% (us) REGRESSION
ffi_sweep        median           args=0 FFI empty kernel          value    baseline=1.6925 new=1.7300 delta=0.0375 pct=+2.2% (us)
ffi_sweep        median           args=0 FFI packed nop            value    baseline=0.1168 new=0.1165 delta=-0.0003 pct=-0.3% (us)
ffi_sweep        median           args=0 FFI typed nop             value    baseline=0.1195 new=0.1185 delta=-0.0010 pct=-0.8% (us)
ffi_sweep        median           args=16 FFI empty kernel         value    baseline=3.6224 new=3.6545 delta=0.0321 pct=+0.9% (us)
ffi_sweep        median           args=16 FFI packed nop           value    baseline=0.2913 new=0.2931 delta=0.0018 pct=+0.6% (us)
ffi_sweep        median           args=16 FFI typed nop            value    baseline=0.3037 new=0.3023 delta=-0.0014 pct=-0.5% (us)
ffi_sweep        median           args=3 FFI empty kernel          value    baseline=3.2137 new=3.2894 delta=0.0757 pct=+2.4% (us)
ffi_sweep        median           args=3 FFI packed nop            value    baseline=0.1450 new=0.1441 delta=-0.0009 pct=-0.6% (us)
ffi_sweep        median           args=3 FFI typed nop             value    baseline=0.1493 new=0.1477 delta=-0.0016 pct=-1.1% (us)
ffi_sweep        median           args=32 FFI empty kernel         value    baseline=4.1099 new=4.1319 delta=0.0220 pct=+0.5% (us)
ffi_sweep        median           args=32 FFI packed nop           value    baseline=0.4806 new=0.4798 delta=-0.0008 pct=-0.2% (us)
ffi_sweep        median           args=32 FFI typed nop            value    baseline=0.4944 new=0.4938 delta=-0.0006 pct=-0.1% (us)
ffi_sweep        median           args=5 FFI empty kernel          value    baseline=3.2477 new=3.3019 delta=0.0542 pct=+1.7% (us)
ffi_sweep        median           args=5 FFI packed nop            value    baseline=0.1740 new=0.1732 delta=-0.0008 pct=-0.5% (us)
ffi_sweep        median           args=5 FFI typed nop             value    baseline=0.1779 new=0.1771 delta=-0.0008 pct=-0.4% (us)
ffi_sweep        median           args=64 FFI empty kernel         value    baseline=5.0757 new=5.0959 delta=0.0202 pct=+0.4% (us)
ffi_sweep        median           args=64 FFI packed nop           value    baseline=0.8451 new=0.8368 delta=-0.0083 pct=-1.0% (us)
ffi_sweep        median           args=64 FFI typed nop            value    baseline=0.8642 new=0.8576 delta=-0.0066 pct=-0.8% (us)
ffi_sweep        median           args=8 FFI empty kernel          value    baseline=3.3744 new=3.3906 delta=0.0162 pct=+0.5% (us)
ffi_sweep        median           args=8 FFI packed nop            value    baseline=0.2061 new=0.2069 delta=0.0008 pct=+0.4% (us)
ffi_sweep        median           args=8 FFI typed nop             value    baseline=0.2098 new=0.2102 delta=0.0004 pct=+0.2% (us)
hip_module       median           INTJ same function               value    baseline=2.9253 new=2.9886 delta=0.0633 pct=+2.2% (us)
hip_module       median           TVM FFI HIP HSACO                value    baseline=3.1619 new=3.1709 delta=0.0090 pct=+0.3% (us)
hip_module       median           Triton same HSACO                value    baseline=16.0663 new=15.8437 delta=-0.2226 pct=-1.4% (us)
launch_gpu       launch           auto map                         ns/call  baseline=3145.6 new=3096.1 delta=-49.5000 pct=-1.6% (ns)
launch_gpu       launch           baked                            ns/call  baseline=2939.1 new=2879.6 delta=-59.5000 pct=-2.0% (ns)
launch_gpu       launch           bound pointer                    ns/call  baseline=2949.1 new=2887.8 delta=-61.3000 pct=-2.1% (ns)
launch_gpu       launch           bound tensor                     ns/call  baseline=2998.3 new=2883.6 delta=-114.7 pct=-3.8% (ns)
launch_gpu       launch           fixed device map                 ns/call  baseline=2895.3 new=2899.9 delta=4.6000 pct=+0.2% (ns)
launch_gpu       launch           fixed device no-map              ns/call  baseline=2893.0 new=2897.2 delta=4.2000 pct=+0.1% (ns)
launch_gpu/launch/generic/ns/call                            MISSING in baseline
launch_gpu/launch/generic tensor ann/ns/call                 MISSING in baseline
launch_gpu       launch           reduced key                      ns/call  baseline=3148.4 new=2971.0 delta=-177.4 pct=-5.6% (ns)
launch_gpu/launch/tensor annotated/ns/call                   MISSING in baseline
launch_gpu       launch           verify off                       ns/call  baseline=3020.7 new=2988.9 delta=-31.8000 pct=-1.1% (ns)
launch_gpu       launch           verify on                        ns/call  baseline=2933.9 new=2956.1 delta=22.2000 pct=+0.8% (ns)
```

### `/tmp/intj-bench/globals-tensor`

```
$ cat /tmp/intj-bench/globals-tensor/round0_ffi_compare.txt
commit=c7125b1dcc97f25b7021ba02183510da8ceecfe7
started=2026-09-28T14:37:29+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_ffi_compare.py --iters 1000 --batches 9
iters=1000 batches=9 unit=us/call (host-only, no timed sync)
FFI packed nop             median=0.1449 samples=[0.15135099999999999, 0.145622, 0.148309, 0.144047, 0.14247300000000002, 0.1449, 0.14411000000000002, 0.144153, 0.14561000000000002]
FFI typed nop              median=0.1488 samples=[0.150127, 0.14843199999999998, 0.14884899999999998, 0.154728, 0.146263, 0.150308, 0.148355, 0.151238, 0.14835800000000002]
FFI empty kernel           median=3.8559 samples=[3.870204, 4.006207, 3.87443, 3.873566, 3.711969, 3.855248, 3.7370140000000003, 3.855928, 3.723621]
INTJ empty kernel          median=3.5046 samples=[3.404418, 3.691619, 3.594553, 3.513632, 3.415592, 3.519125, 3.431166, 3.50463, 3.487969]
INTJ fixed-device kernel   median=3.4399 samples=[3.47182, 3.439879, 3.617697, 3.316835, 3.410585, 3.286118, 3.496622, 3.28975, 3.508553]
FFI packed nop mixed       median=0.1730 samples=[0.171231, 0.171725, 0.172991, 0.173704, 0.176738, 0.17583000000000001, 0.172475, 0.17364500000000002, 0.17297]
FFI typed nop mixed        median=0.1769 samples=[0.17391900000000002, 0.177234, 0.17513900000000002, 0.17799500000000001, 0.176948, 0.17894, 0.174345, 0.17874, 0.176448]
FFI mixed kernel           median=3.8761 samples=[3.793573, 4.111301, 3.912739, 3.947728, 3.678418, 3.898868, 3.746734, 3.876135, 3.764094]
INTJ mixed kernel          median=3.5657 samples=[3.647793, 3.700333, 3.619368, 3.593081, 3.5092339999999997, 3.5068, 3.49134, 3.56566, 3.507951]
INTJ fixed mixed kernel    median=3.5187 samples=[3.689407, 3.78512, 3.550449, 3.573424, 3.472208, 3.501771, 3.4637040000000003, 3.51867, 3.407443]
elapsed_ns=11928069400
exit_status=0
ended=2026-09-28T14:37:41+08:00
```

```
$ cat /tmp/intj-bench/globals-tensor/round0_ffi_paths.txt
commit=c7125b1dcc97f25b7021ba02183510da8ceecfe7
started=2026-09-28T14:38:18+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_intj_ffi_paths.py --mode all --iters 1000 --batches 9
iters=1000 batches=9 unit=ns/call; no_gpu=True
mode=callback
callback_count=9100 (100 warmup + 9000 unique misses)
INTJ cold compile callback + cache  median_ns=707.4 samples_ns=[749.984, 778.941, 688.632, 877.851, 656.553, 681.239, 693.206, 707.389, 1071.737]
INTJ hot call (no callback)         median_ns=35.1 samples_ns=[37.644, 35.873, 34.656, 35.231, 35.064, 35.136, 34.951, 35.054, 34.226]
INTJ 3-tensor host-only nop         median_ns=32.1 samples_ns=[40.331, 32.489, 32.176, 32.14, 32.03, 31.865, 31.877, 32.074, 32.025]
mode=kwargs
INTJ direct positional              median_ns=63.2 samples_ns=[65.626, 63.873, 63.491, 63.228, 63.596, 63.04, 62.954, 63.066, 63.036]
INTJ adapter positional             median_ns=86.0 samples_ns=[86.413, 85.841, 86.245, 86.037, 90.461, 85.577, 85.865, 85.938, 85.969]
INTJ adapter kwargs                 median_ns=102.7 samples_ns=[103.885, 103.07, 102.36, 102.624, 102.738, 106.934, 105.213, 102.4, 100.942]
INTJ adapter defaults               median_ns=85.4 samples_ns=[85.439, 85.293, 84.786, 85.53, 85.36, 85.273, 85.273, 89.85, 85.884]
INTJ FFI wrapper positional         median_ns=100.0 samples_ns=[100.308, 99.708, 99.142, 99.881, 100.003, 101.312, 100.082, 99.82, 103.81]
INTJ FFI wrapper kwargs             median_ns=123.4 samples_ns=[125.921, 123.588, 123.147, 123.61, 123.372, 122.774, 122.451, 125.811, 120.595]
INTJ native kwargs: unsupported (verified TypeError)
mode=dataclass
INTJ pair direct                    median_ns=66.6 samples_ns=[75.556, 67.009, 66.648, 66.617, 66.493, 67.07, 66.67, 66.568, 66.52]
INTJ pair prebuilt *tuple           median_ns=136.5 samples_ns=[135.87, 139.432, 135.219, 136.386, 135.888, 137.576, 136.488, 136.533, 143.067]
FFI unpack Pair only                median_ns=169.0 samples_ns=[168.974, 168.679, 169.024, 167.459, 167.673, 171.893, 169.796, 168.713, 169.172]
INTJ pair manual unpack             median_ns=166.8 samples_ns=[171.629, 171.786, 166.769, 166.195, 158.688, 159.952, 168.843, 160.261, 180.032]
INTJ pair FFI unpack                median_ns=302.6 samples_ns=[303.46, 295.495, 300.188, 299.144, 298.781, 308.246, 308.519, 312.242, 302.57]
INTJ pair stdlib astuple            median_ns=1170.3 samples_ns=[1203.735, 1168.244, 1165.822, 1177.21, 1162.556, 1186.535, 1170.278, 1164.755, 1171.756]
INTJ config direct                  median_ns=78.7 samples_ns=[79.988, 78.972, 78.951, 78.605, 78.47, 85.165, 78.711, 78.729, 78.62]
INTJ config FFI unpack              median_ns=325.3 samples_ns=[320.722, 339.786, 323.928, 327.922, 319.498, 336.022, 322.172, 332.647, 325.311]
INTJ native dataclass unpack: unsupported (verified TypeError)
elapsed_ns=9561439968
exit_status=0
ended=2026-09-28T14:38:27+08:00
```

```
$ cat /tmp/intj-bench/globals-tensor/round0_ffi_sweep.txt
commit=c7125b1dcc97f25b7021ba02183510da8ceecfe7
started=2026-09-28T14:37:41+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_ffi_compare.py --sweep --iters 1000 --batches 9
counts=[0, 3, 5, 8, 16, 32, 64] iters=1000 batches=9 unit=us/call (host-only, no timed sync); INTJ device-bound, no specialization key
args= 0 FFI packed nop           median=0.1159 samples=[0.126707, 0.11593099999999999, 0.115218, 0.11999800000000001, 0.114643, 0.11456999999999999, 0.116904, 0.116694, 0.115584]
args= 0 FFI typed nop            median=0.1187 samples=[0.123219, 0.11870099999999999, 0.11765099999999999, 0.121468, 0.1173, 0.117588, 0.124873, 0.11840300000000001, 0.119289]
args= 0 FFI empty kernel         median=2.1643 samples=[1.830279, 2.954189, 2.164346, 3.031038, 2.121018, 3.05442, 2.077213, 2.921063, 2.0894630000000003]
args= 0 INTJ static_compile kernel median=3.2975 samples=[3.297512, 3.328658, 3.317502, 3.296714, 3.307324, 3.273043, 3.2481619999999998, 3.300748, 3.295877]
args= 0 INTJ runtime_shim kernel median=3.1843 samples=[3.1843220000000003, 3.115416, 3.407482, 3.069891, 3.32868, 3.073185, 3.3058989999999997, 3.058074, 3.317645]
args= 3 FFI packed nop           median=0.1451 samples=[0.144394, 0.145263, 0.14552099999999998, 0.14435900000000002, 0.14339, 0.145414, 0.14514500000000002, 0.145947, 0.14191700000000002]
args= 3 FFI typed nop            median=0.1485 samples=[0.154232, 0.148013, 0.14872, 0.14853899999999998, 0.146978, 0.149194, 0.14752099999999999, 0.15378999999999998, 0.148226]
args= 3 FFI empty kernel         median=3.7568 samples=[3.7568110000000003, 3.8455459999999997, 3.7515169999999998, 3.858752, 3.715681, 3.812496, 3.687134, 3.808726, 3.690975]
args= 3 INTJ static_compile kernel median=3.5444 samples=[3.3075569999999996, 3.570145, 3.5546260000000003, 3.544408, 3.563334, 3.55288, 3.5259989999999997, 3.532784, 3.530425]
args= 3 INTJ runtime_shim kernel median=3.4655 samples=[3.339307, 3.4565520000000003, 3.597374, 3.465535, 3.6082899999999998, 3.443263, 3.57498, 3.41877, 3.559831]
args= 5 FFI packed nop           median=0.1741 samples=[0.172588, 0.171117, 0.17602099999999998, 0.17038599999999998, 0.17408, 0.174314, 0.17438, 0.172633, 0.176718]
args= 5 FFI typed nop            median=0.1765 samples=[0.17646199999999998, 0.18071500000000001, 0.176809, 0.17443199999999998, 0.176547, 0.175496, 0.17604, 0.178499, 0.17563499999999999]
args= 5 FFI empty kernel         median=3.8208 samples=[3.724548, 4.2875879999999995, 3.81138, 4.2711049999999995, 3.8207579999999997, 4.269127999999999, 3.747357, 4.261694, 3.744395]
args= 5 INTJ static_compile kernel median=3.5886 samples=[3.353577, 3.674973, 3.614877, 3.57911, 3.624235, 3.5885540000000002, 3.596901, 3.5570410000000003, 3.584436]
args= 5 INTJ runtime_shim kernel median=3.4813 samples=[3.355747, 3.48129, 4.07568, 3.46633, 3.576289, 3.44552, 3.529187, 3.40997, 4.108826]
args= 8 FFI packed nop           median=0.2080 samples=[0.21055600000000002, 0.20804, 0.21169900000000003, 0.208762, 0.20625, 0.208342, 0.206964, 0.206839, 0.207377]
args= 8 FFI typed nop            median=0.2106 samples=[0.21133000000000002, 0.211769, 0.210191, 0.210125, 0.210574, 0.209861, 0.210813, 0.208861, 0.210823]
args= 8 FFI empty kernel         median=3.9438 samples=[3.943815, 4.47592, 3.845455, 4.396799, 3.849812, 4.405042000000001, 3.8315639999999997, 4.440303, 3.848364]
args= 8 INTJ static_compile kernel median=3.6084 samples=[3.41727, 3.608373, 9.828078, 3.586128, 3.572694, 3.6702179999999998, 3.542022, 3.6613200000000004, 4.762351]
args= 8 INTJ runtime_shim kernel median=3.5211 samples=[3.4444229999999996, 3.5211080000000003, 15.209432000000001, 3.296472, 3.658293, 3.304955, 3.624496, 3.269669, 13.188865]
args=16 FFI packed nop           median=0.2935 samples=[0.293452, 0.295987, 0.298584, 0.29262, 0.294863, 0.291998, 0.291284, 0.293467, 0.6731889999999999]
args=16 FFI typed nop            median=0.3035 samples=[0.30348899999999995, 0.306406, 0.305404, 0.303087, 0.301173, 0.301971, 0.302699, 0.305546, 0.629035]
args=16 FFI empty kernel         median=5.9086 samples=[4.291858, 5.967532, 4.196556999999999, 6.097462999999999, 4.141004, 5.9326289999999995, 4.148140000000001, 5.908577, 10.304495999999999]
args=16 INTJ static_compile kernel median=5.2806 samples=[3.814142, 5.280572, 6.005515999999999, 5.050587, 5.6169530000000005, 5.180687, 5.74268, 5.065268, 5.655328000000001]
args=16 INTJ runtime_shim kernel median=3.8145 samples=[3.8145439999999997, 3.718706, 5.584198000000001, 3.620235, 5.465117, 3.602607, 5.524926000000001, 3.557355, 5.304727]
args=32 FFI packed nop           median=0.4808 samples=[0.480469, 0.480999, 0.487222, 0.493236, 0.47601299999999996, 0.481011, 0.476813, 0.48078899999999997, 0.476182]
args=32 FFI typed nop            median=0.4938 samples=[0.49788600000000005, 0.502011, 0.49393200000000004, 0.514815, 0.489404, 0.490116, 0.493778, 0.487919, 0.490507]
args=32 FFI empty kernel         median=4.8751 samples=[4.875085, 6.270862, 4.73552, 6.099604, 4.683112, 5.9701450000000005, 4.7035540000000005, 6.111975999999999, 4.728578000000001]
args=32 INTJ static_compile kernel median=5.7710 samples=[4.050037, 5.7710230000000005, 5.782147, 5.700119, 5.889407, 5.615779000000001, 5.8888549999999995, 5.577672, 5.853933]
args=32 INTJ runtime_shim kernel median=5.5458 samples=[4.120256, 4.151223, 5.729017, 4.134031, 5.7834080000000005, 11.418878000000001, 5.616357, 4.090916, 5.54578]
args=64 FFI packed nop           median=0.8383 samples=[0.849568, 0.871985, 0.856592, 0.8382820000000001, 0.831203, 2.3050569999999997, 0.837669, 0.83136, 0.823181]
args=64 FFI typed nop            median=0.8598 samples=[0.863576, 0.898448, 0.852302, 0.861707, 0.845996, 1.6051769999999999, 0.848409, 0.8598300000000001, 0.849127]
args=64 FFI empty kernel         median=6.2167 samples=[5.91026, 6.558877000000001, 6.216723, 6.55809, 5.830814, 15.817604, 5.861764, 6.572207000000001, 5.871359]
args=64 INTJ static_compile kernel median=6.1863 samples=[5.469981, 6.169924, 6.24981, 6.358881, 6.1361170000000005, 6.2103019999999995, 6.186264, 6.214005, 6.138326]
args=64 INTJ runtime_shim kernel median=6.1198 samples=[5.413257, 6.142041, 6.119845000000001, 6.352746, 6.092193, 6.196432, 6.035140999999999, 6.233816, 6.073384]
elapsed_ns=36392415530
exit_status=0
ended=2026-09-28T14:38:18+08:00
```

```
$ cat /tmp/intj-bench/globals-tensor/round0_hip_module.txt
commit=c7125b1dcc97f25b7021ba02183510da8ceecfe7
started=2026-09-28T14:38:27+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_hip_module_launch.py --iters 1000 --batches 9
hsaco_sha256=d0bc01bcbace10aa262a50fc2e16c6fd2cd9a2bedb394cbf4fc38c179dee8e46 kernel=empty_kernel threads=256 shared=0 callback_count=1
iters=1000 batches=9 unit=us/call (host enqueue, no timed sync)
TVM FFI HIP HSACO         median=3.8409 samples=[3.686197, 3.833377, 3.885941, 3.838829, 3.926699, 3.840899, 3.836603, 3.867913, 3.857003]
Triton same HSACO         median=16.4061 samples=[16.643978, 16.48714, 16.497087999999998, 16.448505, 16.406107, 16.402666, 16.325161, 16.374864, 16.306863]
INTJ same function        median=3.6730 samples=[3.7520599999999997, 3.730818, 3.672951, 3.74124, 3.702238, 3.598979, 3.662892, 3.586419, 3.6712130000000003]
elapsed_ns=6174000657
exit_status=0
ended=2026-09-28T14:38:33+08:00
```

```
$ cat /tmp/intj-bench/globals-tensor/round0_kernel_cache.txt
commit=c7125b1dcc97f25b7021ba02183510da8ceecfe7
started=2026-09-28T14:38:33+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python -m pytest tests/test_kernel_cache.py -s -q

kernel cache, 1-word key, ns per operation

                      intj       tsl      absl
hash_only             1.36      1.36      1.35
hit/1                 1.58      2.00      1.44
hit/8                 1.60      2.09      4.37
hit/64                1.62      2.13      4.52
hit/512               2.30      2.92      5.32
hit_child             3.00      3.00      3.00
miss/1                1.22      1.45      1.38
miss/8                2.66      1.77      3.28
miss/64               2.76      2.70      3.53
miss/512              2.89      2.00      4.07

kernel cache, 2-word key, ns per operation

                      intj       tsl      absl
hash_only             1.42      1.42      1.42
hit/1                 1.92      2.60      1.98
hit/8                 1.83      2.56      5.03
hit/64                2.18      2.91      5.22
hit/512               3.02      4.04      6.38
hit_child             3.01      3.77      3.00
miss/1                1.38      2.98      1.51
miss/8                2.38      1.86      3.20
miss/64               3.13      2.11      5.17
miss/512              2.88      2.72      4.00

kernel cache, 5-word key, ns per operation

                      intj       tsl      absl
hash_only             3.59      3.59      3.59
hit/1                 6.03      3.06      2.46
hit/8                 3.20      3.41      9.48
hit/64                3.27      3.56      9.78
hit/512               4.55      4.77     11.17
hit_child             5.67      3.02      3.00
miss/1                1.42      2.11      2.20
miss/8                2.98      2.20      6.93
miss/64               2.85      2.83      6.74
miss/512              3.53      2.51      7.70
........
8 passed in 34.38s
elapsed_ns=35334700765
exit_status=0
ended=2026-09-28T14:39:09+08:00
```

```
$ cat /tmp/intj-bench/globals-tensor/round0_launch_dynamic.txt
commit=c7125b1dcc97f25b7021ba02183510da8ceecfe7
started=2026-09-28T14:39:19+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --dynamic --iters 20000 --batches 9
mode=dynamic; 20000 calls × 9 batches; median ns/call
               path     launch   spec_key
         lazy plain     3034.5       95.3
  2 dynamic options     3048.4       99.7
      int constexpr     2969.2       87.7
      str constexpr     3013.8       94.6
elapsed_ns=10674112521
exit_status=0
ended=2026-09-28T14:39:29+08:00
```

```
$ cat /tmp/intj-bench/globals-tensor/round0_launch_gpu.txt
commit=c7125b1dcc97f25b7021ba02183510da8ceecfe7
started=2026-09-28T14:36:09+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --iters 1000 --batches 9
mode=gpu; 1000 calls × 9 batches; median ns/call
               path    ns/call       Δ ns       Δ %   build ms   bind ms
           auto map     3118.9       +0.0      +0.0    2473.54         -
   tensor annotated     3134.6      +15.7      +0.5    2174.87         -
            generic     3146.6      +27.7      +0.9    2367.32         -
 generic tensor ann     3163.4      +44.4      +1.4    2254.08         -
        reduced key     3132.9      +13.9      +0.4    2138.14         -
         verify off     3147.3      +28.4      +0.9    2155.91         -
          verify on     3144.8      +25.9      +0.8    2229.61         -
              baked     3167.5      +48.6      +1.6    2075.32         -
       bound tensor     3161.1      +42.2      +1.4    2123.49      0.15
      bound pointer     3124.8       +5.9      +0.2    2151.37      0.15
   fixed device map     3158.1      +39.1      +1.3    2250.71      0.17
fixed device no-map     3160.0      +41.1      +1.3    2051.98      0.14
elapsed_ns=30571262921
exit_status=0
ended=2026-09-28T14:36:40+08:00
```

```
$ cat /tmp/intj-bench/globals-tensor/round0_launch_host.txt
commit=c7125b1dcc97f25b7021ba02183510da8ceecfe7
started=2026-09-28T14:36:40+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --no-gpu --iters 100000 --batches 9
mode=host-only; 100000 calls × 9 batches; median ns/call
               path    ns/call       Δ ns       Δ %   build ms   bind ms
           auto map       40.2       +0.0      +0.0    2404.80         -
   tensor annotated       40.0       -0.2      -0.6    2156.29         -
            generic       44.4       +4.2     +10.5    2333.59         -
 generic tensor ann       44.4       +4.1     +10.3    2216.80         -
        reduced key       41.1       +0.8      +2.0    2121.86         -
         verify off       42.0       +1.7      +4.3    2131.48         -
          verify on       43.3       +3.0      +7.6    2214.50         -
              baked       38.3       -2.0      -4.9    2034.32         -
       bound tensor       43.7       +3.4      +8.5    2122.35      0.14
      bound pointer       41.8       +1.6      +3.9    2135.99      0.15
   fixed device map       41.5       +1.3      +3.2    2245.47      0.15
fixed device no-map       42.8       +2.6      +6.3    2054.77      0.13
elapsed_ns=29164217890
exit_status=0
ended=2026-09-28T14:37:09+08:00
```

```
$ cat /tmp/intj-bench/globals-tensor/round0_launch_last_key.txt
commit=c7125b1dcc97f25b7021ba02183510da8ceecfe7
started=2026-09-28T14:37:26+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --last-key --iters 100000 --batches 9
mode=last-key; host-only; 200000 calls × 9 batches; median ns/call
count     pattern    ns/call
    4      repeat       32.5
    4   alternate       34.4
   16      repeat       61.1
   16   alternate       64.3
   32      repeat       98.6
   32   alternate      103.6
elapsed_ns=3328637472
exit_status=0
ended=2026-09-28T14:37:29+08:00
```

```
$ cat /tmp/intj-bench/globals-tensor/round0_launch_readme.txt
commit=c7125b1dcc97f25b7021ba02183510da8ceecfe7
started=2026-09-28T14:37:09+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --readme --iters 1000 --batches 9
mode=readme; 1000 calls × 9 batches; median
        path   triton us    intj us   speedup
   grid=(1,)       16.52       3.17      5.2x
   grid=(0,)       12.99       0.03    475.8x

torch_access_mode   decode ns    build s
     runtime_shim        94.7       0.86
   static_compile        95.6       0.13
      interpreter       889.2       0.88
elapsed_ns=5420660168
exit_status=0
ended=2026-09-28T14:37:14+08:00
```

```
$ cat /tmp/intj-bench/globals-tensor/round0_launch_sweep.txt
commit=c7125b1dcc97f25b7021ba02183510da8ceecfe7
started=2026-09-28T14:37:14+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --sweep --iters 100000 --batches 9
mode=sweep; host-only; 100000 calls × 9 batches; median ns/call
count   kind    ns/call
    4    int       38.6
    4 tensor       36.7
   16    int       67.1
   16 tensor       62.8
   32    int      105.0
   32 tensor      107.9
elapsed_ns=11691628372
exit_status=0
ended=2026-09-28T14:37:26+08:00
```

```
$ cat /tmp/intj-bench/globals-tensor/round0_launch_tuned.txt
commit=c7125b1dcc97f25b7021ba02183510da8ceecfe7
started=2026-09-28T14:39:09+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --tuned --iters 20000 --batches 9
 tuned hit:  3773.7 ns/launch
     plain:  3856.6 ns/launch
elapsed_ns=9865808417
exit_status=0
ended=2026-09-28T14:39:19+08:00
```

```
$ cat /tmp/intj-bench/globals-tensor/round1_ffi_compare.txt
commit=c7125b1dcc97f25b7021ba02183510da8ceecfe7
started=2026-09-28T14:39:47+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_ffi_compare.py --iters 1000 --batches 9
iters=1000 batches=9 unit=us/call (host-only, no timed sync)
FFI packed nop             median=0.1442 samples=[0.15166300000000002, 0.143608, 0.143994, 0.145203, 0.14479, 0.144068, 0.144211, 0.142588, 0.147694]
FFI typed nop              median=0.1492 samples=[0.147886, 0.150316, 0.14921600000000002, 0.15026699999999998, 0.148958, 0.15119, 0.148453, 0.149274, 0.148393]
FFI empty kernel           median=3.6843 samples=[3.683091, 4.688101, 3.68428, 4.40529, 3.585124, 4.4302209999999995, 3.606727, 4.496234, 3.591855]
INTJ empty kernel          median=3.8152 samples=[3.261469, 3.815169, 4.194408, 3.560504, 4.087933, 3.701636, 4.11929, 3.7480569999999997, 4.14117]
INTJ fixed-device kernel   median=3.3623 samples=[3.046211, 3.3622959999999997, 3.592943, 3.238909, 3.650798, 3.2612979999999996, 3.439744, 3.2696300000000003, 3.507184]
FFI packed nop mixed       median=0.1731 samples=[0.173328, 0.17248500000000003, 0.172647, 0.172676, 0.176465, 0.175617, 0.173097, 0.274435, 0.17158600000000002]
FFI typed nop mixed        median=0.1762 samples=[0.176232, 0.17816200000000001, 0.17439, 0.17762, 0.17361600000000002, 0.177721, 0.175912, 0.431015, 0.173848]
FFI mixed kernel           median=3.8941 samples=[3.7233539999999996, 4.034826, 3.894071, 3.9174189999999998, 3.868408, 3.936714, 3.749632, 18.530257000000002, 3.781898]
INTJ mixed kernel          median=3.6514 samples=[4.098017, 3.797969, 3.5803949999999998, 3.651354, 3.6699, 3.6469699999999996, 3.491364, 4.60224, 3.542747]
INTJ fixed mixed kernel    median=3.6566 samples=[3.799832, 3.786701, 3.613931, 3.679561, 3.659568, 3.635471, 3.656583, 3.638523, 3.645509]
elapsed_ns=3928559044
exit_status=0
ended=2026-09-28T14:39:51+08:00
```

```
$ cat /tmp/intj-bench/globals-tensor/round1_ffi_paths.txt
commit=c7125b1dcc97f25b7021ba02183510da8ceecfe7
started=2026-09-28T14:39:56+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_intj_ffi_paths.py --mode all --iters 1000 --batches 9
iters=1000 batches=9 unit=ns/call; no_gpu=True
mode=callback
callback_count=9100 (100 warmup + 9000 unique misses)
INTJ cold compile callback + cache  median_ns=701.3 samples_ns=[726.584, 742.458, 694.127, 1174.016, 648.406, 688.843, 701.314, 696.298, 2041.244]
INTJ hot call (no callback)         median_ns=35.5 samples_ns=[40.028, 36.364, 35.782, 36.462, 35.216, 35.538, 34.728, 35.03, 34.171]
INTJ 3-tensor host-only nop         median_ns=32.5 samples_ns=[34.862, 34.11, 32.451, 32.602, 32.4, 32.557, 32.388, 32.54, 32.337]
mode=kwargs
INTJ direct positional              median_ns=62.7 samples_ns=[76.188, 63.213, 62.682, 63.074, 62.775, 62.747, 62.695, 62.671, 62.364]
INTJ adapter positional             median_ns=88.1 samples_ns=[88.025, 88.092, 87.846, 86.771, 91.069, 87.959, 88.254, 88.34, 88.355]
INTJ adapter kwargs                 median_ns=104.6 samples_ns=[106.569, 104.236, 105.021, 104.618, 104.026, 111.489, 104.534, 102.266, 104.85]
INTJ adapter defaults               median_ns=87.8 samples_ns=[87.775, 87.608, 87.894, 87.792, 86.772, 88.044, 87.455, 90.888, 89.399]
INTJ FFI wrapper positional         median_ns=101.0 samples_ns=[101.424, 100.478, 100.914, 101.367, 101.094, 100.49, 100.909, 100.961, 103.767]
INTJ FFI wrapper kwargs             median_ns=123.1 samples_ns=[123.668, 122.959, 123.425, 122.951, 122.21, 123.12, 122.703, 128.017, 124.328]
INTJ native kwargs: unsupported (verified TypeError)
mode=dataclass
INTJ pair direct                    median_ns=66.3 samples_ns=[75.208, 66.258, 66.129, 66.289, 66.467, 66.362, 69.53, 65.863, 65.575]
INTJ pair prebuilt *tuple           median_ns=138.9 samples_ns=[139.165, 139.365, 138.365, 138.933, 137.952, 141.451, 137.2, 137.7, 139.063]
FFI unpack Pair only                median_ns=166.1 samples_ns=[165.322, 166.71, 172.194, 168.724, 166.111, 165.061, 165.781, 165.348, 168.952]
INTJ pair manual unpack             median_ns=162.5 samples_ns=[164.523, 161.67, 161.099, 162.086, 163.167, 163.946, 161.064, 162.519, 162.528]
INTJ pair FFI unpack                median_ns=305.7 samples_ns=[303.217, 310.615, 302.661, 301.67, 310.32, 306.318, 303.922, 305.66, 306.011]
INTJ pair stdlib astuple            median_ns=1176.2 samples_ns=[1239.664, 1208.182, 1182.827, 1176.213, 1171.234, 1168.643, 1177.66, 1172.884, 1174.58]
INTJ config direct                  median_ns=80.0 samples_ns=[84.525, 79.09, 79.403, 80.06, 79.938, 79.826, 79.999, 80.01, 90.405]
INTJ config FFI unpack              median_ns=325.8 samples_ns=[329.056, 323.892, 325.53, 329.43, 325.754, 322.515, 328.249, 327.86, 325.607]
INTJ native dataclass unpack: unsupported (verified TypeError)
elapsed_ns=2969222182
exit_status=0
ended=2026-09-28T14:39:59+08:00
```

```
$ cat /tmp/intj-bench/globals-tensor/round1_ffi_sweep.txt
commit=c7125b1dcc97f25b7021ba02183510da8ceecfe7
started=2026-09-28T14:39:51+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_ffi_compare.py --sweep --iters 1000 --batches 9
counts=[0, 3, 5, 8, 16, 32, 64] iters=1000 batches=9 unit=us/call (host-only, no timed sync); INTJ device-bound, no specialization key
args= 0 FFI packed nop           median=0.1175 samples=[0.122164, 0.118103, 0.117425, 0.118275, 0.11791, 0.117476, 0.114851, 0.11656799999999999, 0.117468]
args= 0 FFI typed nop            median=0.1209 samples=[0.121783, 0.12303, 0.12011100000000001, 0.12092, 0.118853, 0.120629, 0.117922, 0.121339, 0.123304]
args= 0 FFI empty kernel         median=2.2218 samples=[1.73233, 2.925596, 2.221801, 2.867642, 2.061302, 2.966589, 2.062365, 2.927299, 2.052321]
args= 0 INTJ static_compile kernel median=3.3020 samples=[3.215967, 3.372503, 3.374645, 3.302027, 3.303086, 3.305098, 3.2370300000000003, 3.2416419999999997, 3.224795]
args= 0 INTJ runtime_shim kernel median=3.1834 samples=[3.183399, 3.141259, 3.445537, 3.0774529999999998, 3.292873, 3.121705, 3.2858389999999997, 3.0275030000000003, 3.25646]
args= 3 FFI packed nop           median=0.1451 samples=[0.145286, 0.14636500000000002, 0.144529, 0.14523, 0.14703899999999998, 0.145072, 0.14352099999999998, 0.144679, 0.141695]
args= 3 FFI typed nop            median=0.1484 samples=[0.151834, 0.15054900000000002, 0.147228, 0.15246, 0.14841200000000002, 0.150491, 0.14676, 0.148208, 0.14646]
args= 3 FFI empty kernel         median=3.8011 samples=[3.732182, 4.443390999999999, 3.805565, 3.8161359999999998, 3.743435, 3.869427, 3.7092530000000004, 3.80106, 3.6435169999999997]
args= 3 INTJ static_compile kernel median=3.5219 samples=[3.3248249999999997, 3.621064, 3.5219319999999996, 3.5338510000000003, 3.524455, 3.634106, 3.49517, 3.4504810000000004, 3.4666770000000002]
args= 3 INTJ runtime_shim kernel median=3.4959 samples=[3.307462, 3.2733220000000003, 3.567777, 3.4488060000000003, 4.050146, 3.49911, 3.5370079999999997, 3.3918850000000003, 3.495905]
args= 5 FFI packed nop           median=0.1741 samples=[0.173464, 0.17536500000000002, 0.17561000000000002, 0.174529, 0.171648, 0.17141, 0.17485, 0.170298, 0.174144]
args= 5 FFI typed nop            median=0.1782 samples=[0.179098, 0.182863, 0.178548, 0.17663399999999999, 0.175331, 0.176798, 0.177612, 0.17818, 0.17989]
args= 5 FFI empty kernel         median=3.8319 samples=[3.7395419999999997, 4.38965, 3.8318659999999998, 4.323385, 3.7926460000000004, 4.38432, 3.730406, 4.163652, 3.712558]
args= 5 INTJ static_compile kernel median=3.5185 samples=[3.338916, 3.6406419999999997, 3.52271, 3.504645, 3.5184879999999996, 3.5775259999999998, 3.51896, 3.458285, 3.484212]
args= 5 INTJ runtime_shim kernel median=3.4945 samples=[3.370788, 3.5774540000000004, 4.051628, 3.42223, 3.536804, 3.493926, 3.5293330000000003, 3.417491, 3.494509]
args= 8 FFI packed nop           median=0.2051 samples=[0.202682, 0.208447, 0.207422, 0.207057, 0.20447200000000001, 0.205367, 0.20475, 0.2051, 0.203396]
args= 8 FFI typed nop            median=0.2090 samples=[0.21154599999999998, 0.210396, 0.210823, 0.206665, 0.207465, 0.208965, 0.20853, 0.207948, 0.20899]
args= 8 FFI empty kernel         median=3.9806 samples=[3.98055, 4.43492, 3.8154299999999997, 4.367473, 3.820978, 4.381462000000001, 3.774045, 4.33054, 3.824061]
args= 8 INTJ static_compile kernel median=3.5826 samples=[3.409602, 3.652894, 4.193294, 3.5826320000000003, 3.531775, 3.719921, 3.524749, 3.698047, 3.514287]
args= 8 INTJ runtime_shim kernel median=3.4696 samples=[3.469603, 3.4155990000000003, 3.641848, 3.311517, 3.659998, 3.371913, 3.600177, 3.338548, 3.6138380000000003]
args=16 FFI packed nop           median=0.2919 samples=[0.292706, 0.296646, 0.300323, 0.290704, 0.29391300000000004, 0.291879, 0.29181799999999997, 0.289253, 0.291647]
args=16 FFI typed nop            median=0.3028 samples=[0.297414, 0.31011, 0.30269799999999997, 0.309591, 0.302848, 0.31038299999999996, 0.300214, 0.299937, 0.306724]
args=16 FFI empty kernel         median=4.2274 samples=[4.227384, 5.886929, 4.211882999999999, 5.6790020000000005, 4.127676, 5.683667000000001, 4.133433, 5.774229, 4.122255]
args=16 INTJ static_compile kernel median=5.4729 samples=[3.806251, 5.429264, 5.438893, 5.510152, 5.788228, 5.343093, 5.76178, 5.472854999999999, 5.673313]
args=16 INTJ runtime_shim kernel median=5.1134 samples=[3.781786, 3.7821260000000003, 5.113354999999999, 3.667111, 5.621263, 5.824727, 5.494962999999999, 3.689556, 5.619717]
args=32 FFI packed nop           median=0.4790 samples=[0.473189, 0.479325, 0.481293, 0.477188, 0.47712099999999996, 1.270613, 0.479891, 0.47902, 0.47701]
args=32 FFI typed nop            median=0.4970 samples=[0.481599, 0.488768, 0.498335, 0.497209, 0.491283, 1.099881, 0.48573099999999997, 0.496981, 0.498158]
args=32 FFI empty kernel         median=4.8143 samples=[4.7722039999999994, 5.889968, 4.814305999999999, 5.868412, 4.6593469999999995, 18.756647, 4.692925, 6.046341, 4.648916]
args=32 INTJ static_compile kernel median=5.6680 samples=[4.229266, 5.4759459999999995, 5.947426, 5.571496, 5.667973, 7.748381999999999, 5.6804499999999996, 5.583569000000001, 5.776491]
args=32 INTJ runtime_shim kernel median=4.1656 samples=[4.132944999999999, 4.1655619999999995, 5.774896, 4.111962, 5.393471, 4.102442, 5.600341, 4.109442, 4.925599]
args=64 FFI packed nop           median=0.8328 samples=[0.845544, 0.826096, 0.8637050000000001, 0.815652, 0.83247, 0.8328490000000001, 0.862889, 0.8306910000000001, 0.840818]
args=64 FFI typed nop            median=0.8665 samples=[0.888088, 0.869544, 0.84926, 0.865281, 0.86393, 0.877866, 0.900163, 0.86645, 0.86451]
args=64 FFI empty kernel         median=5.9270 samples=[5.848492, 6.37017, 5.926988000000001, 6.416184, 5.857136000000001, 6.365658000000001, 5.897273, 6.460134, 5.8664499999999995]
args=64 INTJ static_compile kernel median=6.0317 samples=[5.427974, 5.9767280000000005, 6.003582000000001, 6.030428, 6.031661, 6.041643, 6.096253, 6.139783, 6.104787]
args=64 INTJ runtime_shim kernel median=6.0706 samples=[5.695351, 6.10479, 6.070915, 6.1665410000000005, 5.9376109999999995, 6.070622999999999, 5.993656, 6.292886, 6.0382560000000005]
elapsed_ns=4770316854
exit_status=0
ended=2026-09-28T14:39:56+08:00
```

```
$ cat /tmp/intj-bench/globals-tensor/round1_hip_module.txt
commit=c7125b1dcc97f25b7021ba02183510da8ceecfe7
started=2026-09-28T14:39:59+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_hip_module_launch.py --iters 1000 --batches 9
hsaco_sha256=d0bc01bcbace10aa262a50fc2e16c6fd2cd9a2bedb394cbf4fc38c179dee8e46 kernel=empty_kernel threads=256 shared=0 callback_count=1
iters=1000 batches=9 unit=us/call (host enqueue, no timed sync)
TVM FFI HIP HSACO         median=3.2768 samples=[3.5898809999999997, 3.276822, 3.289493, 3.238112, 3.265832, 3.256855, 3.261665, 3.511545, 3.967189]
Triton same HSACO         median=15.9969 samples=[16.065545, 16.030472, 16.05495, 15.954013000000002, 15.918327999999999, 15.951924000000002, 15.960804, 15.996889, 46.825942000000005]
INTJ same function        median=2.9775 samples=[3.0277109999999996, 3.006986, 2.974054, 3.016476, 2.977511, 2.970561, 2.942763, 2.945434, 2.9980189999999998]
elapsed_ns=3949390541
exit_status=0
ended=2026-09-28T14:40:02+08:00
```

```
$ cat /tmp/intj-bench/globals-tensor/round1_kernel_cache.txt
commit=c7125b1dcc97f25b7021ba02183510da8ceecfe7
started=2026-09-28T14:40:02+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python -m pytest tests/test_kernel_cache.py -s -q

kernel cache, 1-word key, ns per operation

                      intj       tsl      absl
hash_only             1.35      1.35      1.36
hit/1                 1.56      2.02      1.44
hit/8                 2.35      2.10      4.36
hit/64                1.65      2.14      4.55
hit/512               2.30      2.90      5.30
hit_child             3.00      3.00      3.00
miss/1                1.22      1.98      1.38
miss/8                2.75      1.79      3.29
miss/64               2.79      1.78      3.52
miss/512              2.95      1.99      4.65

kernel cache, 2-word key, ns per operation

                      intj       tsl      absl
hash_only             1.42      1.42      1.42
hit/1                 1.92      2.57      2.00
hit/8                 2.40      3.25      5.04
hit/64                2.30      2.92      5.22
hit/512               3.02      4.03      6.39
hit_child             3.00      3.00      3.01
miss/1                1.38      2.32      1.49
miss/8                2.96      1.87      4.09
miss/64               3.11      2.12      3.88
miss/512              2.87      3.37      3.98

kernel cache, 5-word key, ns per operation

                      intj       tsl      absl
hash_only             3.58      3.59      3.62
hit/1                 3.00      3.06      2.65
hit/8                 3.20      3.41      9.47
hit/64                3.28      3.51     11.99
hit/512               4.53      4.77     11.22
hit_child             3.00      3.01      3.00
miss/1                1.99      2.10      2.26
miss/8                3.02      2.22      6.95
miss/64               2.88      1.95      7.50
miss/512              3.48      2.50      7.60
........
8 passed in 25.81s
elapsed_ns=26764810667
exit_status=0
ended=2026-09-28T14:40:29+08:00
```

```
$ cat /tmp/intj-bench/globals-tensor/round1_launch_dynamic.txt
commit=c7125b1dcc97f25b7021ba02183510da8ceecfe7
started=2026-09-28T14:40:34+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --dynamic --iters 20000 --batches 9
mode=dynamic; 20000 calls × 9 batches; median ns/call
               path     launch   spec_key
         lazy plain     3075.3       94.9
  2 dynamic options     3072.9       99.8
      int constexpr     3008.1       89.9
      str constexpr     3035.4       92.1
elapsed_ns=5926956112
exit_status=0
ended=2026-09-28T14:40:40+08:00
```

```
$ cat /tmp/intj-bench/globals-tensor/round1_launch_gpu.txt
commit=c7125b1dcc97f25b7021ba02183510da8ceecfe7
started=2026-09-28T14:39:29+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --iters 1000 --batches 9
mode=gpu; 1000 calls × 9 batches; median ns/call
               path    ns/call       Δ ns       Δ %   build ms   bind ms
           auto map     3151.6       +0.0      +0.0      80.58         -
   tensor annotated     3157.0       +5.4      +0.2       2.97         -
            generic     3025.3     -126.3      -4.0       2.53         -
 generic tensor ann     2947.2     -204.5      -6.5       2.57         -
        reduced key     2969.1     -182.6      -5.8       2.70         -
         verify off     2943.7     -207.9      -6.6       2.80         -
          verify on     2976.5     -175.2      -5.6       2.73         -
              baked     2892.1     -259.5      -8.2       2.77         -
       bound tensor     2898.0     -253.7      -8.0       2.52      0.13
      bound pointer     2907.3     -244.4      -7.8       2.54      0.12
   fixed device map     2881.1     -270.6      -8.6       2.47      0.13
fixed device no-map     4326.4    +1174.8     +37.3       2.51      0.12
elapsed_ns=3925339992
exit_status=0
ended=2026-09-28T14:39:33+08:00
```

```
$ cat /tmp/intj-bench/globals-tensor/round1_launch_host.txt
commit=c7125b1dcc97f25b7021ba02183510da8ceecfe7
started=2026-09-28T14:39:33+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --no-gpu --iters 100000 --batches 9
mode=host-only; 100000 calls × 9 batches; median ns/call
               path    ns/call       Δ ns       Δ %   build ms   bind ms
           auto map       40.0       +0.0      +0.0      68.52         -
   tensor annotated       40.1       +0.1      +0.3       2.42         -
            generic       44.7       +4.7     +11.8       2.32         -
 generic tensor ann       44.5       +4.5     +11.3       2.48         -
        reduced key       40.8       +0.7      +1.9       2.35         -
         verify off       41.2       +1.2      +3.0       2.27         -
          verify on       43.0       +3.0      +7.4       2.26         -
              baked       38.3       -1.7      -4.2       2.25         -
       bound tensor       43.7       +3.7      +9.2       2.18      0.11
      bound pointer       41.9       +1.9      +4.8       2.17      0.12
   fixed device map       41.1       +1.1      +2.7       2.13      0.12
fixed device no-map       43.2       +3.2      +8.0      17.91      0.73
elapsed_ns=3203193568
exit_status=0
ended=2026-09-28T14:39:36+08:00
```

```
$ cat /tmp/intj-bench/globals-tensor/round1_launch_last_key.txt
commit=c7125b1dcc97f25b7021ba02183510da8ceecfe7
started=2026-09-28T14:39:43+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --last-key --iters 100000 --batches 9
mode=last-key; host-only; 200000 calls × 9 batches; median ns/call
count     pattern    ns/call
    4      repeat       32.4
    4   alternate       34.4
   16      repeat       61.1
   16   alternate       64.4
   32      repeat       98.7
   32   alternate      102.7
elapsed_ns=3485472104
exit_status=0
ended=2026-09-28T14:39:47+08:00
```

```
$ cat /tmp/intj-bench/globals-tensor/round1_launch_readme.txt
commit=c7125b1dcc97f25b7021ba02183510da8ceecfe7
started=2026-09-28T14:39:36+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --readme --iters 1000 --batches 9
mode=readme; 1000 calls × 9 batches; median
        path   triton us    intj us   speedup
   grid=(1,)       16.57       3.06      5.4x
   grid=(0,)       13.23       0.03    490.3x

torch_access_mode   decode ns    build s
     runtime_shim        93.3       0.02
   static_compile        97.1       0.06
      interpreter       906.6       0.00
elapsed_ns=3963324665
exit_status=0
ended=2026-09-28T14:39:40+08:00
```

```
$ cat /tmp/intj-bench/globals-tensor/round1_launch_sweep.txt
commit=c7125b1dcc97f25b7021ba02183510da8ceecfe7
started=2026-09-28T14:39:40+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --sweep --iters 100000 --batches 9
mode=sweep; host-only; 100000 calls × 9 batches; median ns/call
count   kind    ns/call
    4    int       38.8
    4 tensor       37.0
   16    int       67.0
   16 tensor       62.8
   32    int      104.3
   32 tensor      109.3
elapsed_ns=3057541413
exit_status=0
ended=2026-09-28T14:39:43+08:00
```

```
$ cat /tmp/intj-bench/globals-tensor/round1_launch_tuned.txt
commit=c7125b1dcc97f25b7021ba02183510da8ceecfe7
started=2026-09-28T14:40:29+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --tuned --iters 20000 --batches 9
 tuned hit:  3006.4 ns/launch
     plain:  3014.2 ns/launch
elapsed_ns=4700620493
exit_status=0
ended=2026-09-28T14:40:34+08:00
```

```
$ cat /tmp/intj-bench/globals-tensor/round2_ffi_compare.txt
commit=c7125b1dcc97f25b7021ba02183510da8ceecfe7
started=2026-09-28T14:40:58+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_ffi_compare.py --iters 1000 --batches 9
iters=1000 batches=9 unit=us/call (host-only, no timed sync)
FFI packed nop             median=0.1446 samples=[0.152834, 0.144376, 0.14399199999999998, 0.147822, 0.143214, 0.14455400000000002, 0.14463399999999998, 0.143864, 0.144724]
FFI typed nop              median=0.1505 samples=[0.155568, 0.150201, 0.15174700000000002, 0.150511, 0.149407, 0.154115, 0.14937799999999998, 0.15175899999999998, 0.148257]
FFI empty kernel           median=3.3365 samples=[3.5855189999999997, 3.433346, 3.4449229999999997, 3.2429699999999997, 3.278171, 3.3365189999999996, 3.210976, 3.405346, 3.216614]
INTJ empty kernel          median=2.9941 samples=[3.0891930000000003, 3.093726, 3.085869, 2.950456, 2.895741, 3.003666, 2.994118, 2.922914, 2.983474]
INTJ fixed-device kernel   median=2.8851 samples=[2.911983, 3.087828, 3.07624, 2.931269, 2.885135, 2.828253, 2.805071, 2.81759, 2.812255]
FFI packed nop mixed       median=0.1721 samples=[0.17362899999999998, 0.17213900000000001, 0.17515799999999998, 0.173277, 0.170257, 0.169431, 0.174903, 0.171523, 0.17041399999999998]
FFI typed nop mixed        median=0.1749 samples=[0.174838, 0.177927, 0.174873, 0.180107, 0.177833, 0.174194, 0.17369300000000001, 0.177672, 0.174906]
FFI mixed kernel           median=3.3182 samples=[3.361498, 3.431028, 3.3715059999999997, 3.318241, 3.25871, 3.315698, 3.267297, 3.3330349999999997, 3.252273]
INTJ mixed kernel          median=2.9325 samples=[3.149681, 3.119273, 2.93254, 2.9704029999999997, 2.904785, 2.919052, 2.878571, 3.000037, 2.874879]
INTJ fixed mixed kernel    median=2.9461 samples=[3.118996, 3.1189430000000002, 2.9918270000000002, 2.9710300000000003, 2.912898, 2.879744, 2.9068490000000002, 2.9460509999999998, 2.908114]
elapsed_ns=3767796953
exit_status=0
ended=2026-09-28T14:41:01+08:00
```

```
$ cat /tmp/intj-bench/globals-tensor/round2_ffi_paths.txt
commit=c7125b1dcc97f25b7021ba02183510da8ceecfe7
started=2026-09-28T14:41:06+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_intj_ffi_paths.py --mode all --iters 1000 --batches 9
iters=1000 batches=9 unit=ns/call; no_gpu=True
mode=callback
callback_count=9100 (100 warmup + 9000 unique misses)
INTJ cold compile callback + cache  median_ns=695.0 samples_ns=[718.612, 733.88, 671.552, 1141.574, 641.196, 665.678, 680.722, 695.001, 1993.468]
INTJ hot call (no callback)         median_ns=36.0 samples_ns=[39.105, 36.725, 35.855, 36.028, 35.812, 36.144, 35.229, 36.974, 35.366]
INTJ 3-tensor host-only nop         median_ns=32.4 samples_ns=[34.405, 32.704, 32.215, 32.308, 31.967, 36.115, 32.355, 32.43, 32.076]
mode=kwargs
INTJ direct positional              median_ns=63.0 samples_ns=[65.098, 63.216, 63.005, 62.943, 63.077, 62.862, 62.982, 62.84, 62.927]
INTJ adapter positional             median_ns=87.0 samples_ns=[89.657, 93.436, 86.217, 85.985, 86.766, 87.7, 86.958, 86.919, 87.743]
INTJ adapter kwargs                 median_ns=103.8 samples_ns=[104.425, 103.599, 103.733, 107.835, 103.281, 103.833, 103.846, 104.821, 103.741]
INTJ adapter defaults               median_ns=85.8 samples_ns=[88.3, 86.151, 85.543, 85.927, 89.775, 85.37, 85.182, 85.845, 84.699]
INTJ FFI wrapper positional         median_ns=100.4 samples_ns=[101.447, 99.585, 100.495, 99.688, 99.732, 100.137, 105.536, 100.69, 100.411]
INTJ FFI wrapper kwargs             median_ns=124.9 samples_ns=[123.078, 123.438, 123.411, 123.248, 124.954, 142.29, 125.912, 124.908, 125.25]
INTJ native kwargs: unsupported (verified TypeError)
mode=dataclass
INTJ pair direct                    median_ns=65.1 samples_ns=[69.046, 66.311, 69.932, 65.688, 64.388, 64.576, 64.727, 65.076, 64.927]
INTJ pair prebuilt *tuple           median_ns=140.1 samples_ns=[141.709, 140.611, 139.743, 151.498, 140.89, 138.78, 140.056, 137.781, 138.079]
FFI unpack Pair only                median_ns=162.1 samples_ns=[162.539, 165.002, 161.44, 162.066, 161.665, 162.055, 162.094, 172.505, 184.819]
INTJ pair manual unpack             median_ns=164.9 samples_ns=[164.877, 165.75, 165.674, 165.436, 167.408, 162.274, 162.773, 162.682, 162.659]
INTJ pair FFI unpack                median_ns=305.0 samples_ns=[307.185, 305.811, 304.497, 309.034, 304.034, 304.966, 301.773, 308.435, 302.556]
INTJ pair stdlib astuple            median_ns=1179.5 samples_ns=[1236.232, 1212.379, 1172.82, 1186.519, 1170.265, 1171.549, 1179.498, 1181.112, 1175.932]
INTJ config direct                  median_ns=81.3 samples_ns=[82.815, 81.192, 81.077, 81.125, 81.218, 81.493, 81.255, 81.263, 84.902]
INTJ config FFI unpack              median_ns=323.5 samples_ns=[325.351, 323.236, 326.341, 323.284, 320.9, 323.495, 326.852, 323.461, 321.73]
INTJ native dataclass unpack: unsupported (verified TypeError)
elapsed_ns=3003918839
exit_status=0
ended=2026-09-28T14:41:09+08:00
```

```
$ cat /tmp/intj-bench/globals-tensor/round2_ffi_sweep.txt
commit=c7125b1dcc97f25b7021ba02183510da8ceecfe7
started=2026-09-28T14:41:01+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_ffi_compare.py --sweep --iters 1000 --batches 9
counts=[0, 3, 5, 8, 16, 32, 64] iters=1000 batches=9 unit=us/call (host-only, no timed sync); INTJ device-bound, no specialization key
args= 0 FFI packed nop           median=0.1169 samples=[0.120214, 0.11614400000000001, 0.11695399999999999, 0.12143999999999999, 0.116926, 0.11609399999999999, 0.117077, 0.116039, 0.115701]
args= 0 FFI typed nop            median=0.1189 samples=[0.118034, 0.11772400000000001, 0.118884, 0.122977, 0.119577, 0.12300900000000001, 0.117753, 0.118127, 0.122014]
args= 0 FFI empty kernel         median=2.3597 samples=[1.833211, 3.113663, 2.359694, 3.0765569999999998, 2.287246, 3.0684229999999997, 2.139645, 3.094976, 2.251672]
args= 0 INTJ static_compile kernel median=3.2316 samples=[3.231621, 3.415084, 3.345105, 3.290328, 3.304475, 3.2260880000000003, 3.142789, 3.2254720000000003, 3.155516]
args= 0 INTJ runtime_shim kernel median=3.2101 samples=[3.260409, 3.127085, 3.4252800000000003, 3.0746860000000003, 3.318864, 3.067468, 3.21007, 3.0724940000000003, 3.245387]
args= 3 FFI packed nop           median=0.1454 samples=[0.147715, 0.14535599999999999, 0.147584, 0.144834, 0.14158099999999998, 0.145554, 0.144059, 0.14580500000000002, 0.144038]
args= 3 FFI typed nop            median=0.1480 samples=[0.14728899999999998, 0.147957, 0.147012, 0.151625, 0.146667, 0.152113, 0.14797200000000002, 0.1502, 0.14732499999999998]
args= 3 FFI empty kernel         median=3.7358 samples=[3.735801, 3.869907, 3.6833620000000002, 3.846663, 3.605273, 3.8171109999999997, 3.589413, 3.811234, 3.6143110000000003]
args= 3 INTJ static_compile kernel median=3.8775 samples=[3.3440819999999998, 12.172600000000001, 3.927982, 3.474535, 3.927757, 3.351714, 3.877475, 3.497264, 3.888806]
args= 3 INTJ runtime_shim kernel median=3.4415 samples=[3.3348299999999997, 13.207865, 3.571797, 3.441507, 3.416385, 3.3121840000000002, 3.509777, 3.40626, 3.514399]
args= 5 FFI packed nop           median=0.1745 samples=[0.17666800000000002, 0.172273, 0.174536, 0.17336500000000002, 0.174126, 0.17777199999999999, 0.17929699999999998, 0.175982, 0.172392]
args= 5 FFI typed nop            median=0.1774 samples=[0.176386, 0.176066, 0.177955, 0.179215, 0.178672, 0.17535, 0.177241, 0.17946299999999998, 0.177414]
args= 5 FFI empty kernel         median=3.8592 samples=[3.7419830000000003, 4.323147, 3.859187, 4.281115, 3.811511, 4.437131, 3.801357, 4.2013370000000005, 3.79743]
args= 5 INTJ static_compile kernel median=3.5193 samples=[3.322237, 3.563663, 3.531927, 3.462991, 3.488682, 3.519313, 4.420798, 3.458866, 3.530057]
args= 5 INTJ runtime_shim kernel median=3.5027 samples=[3.3994470000000003, 3.502748, 4.09383, 3.5272669999999997, 4.012704, 3.122608, 4.855733, 3.494219, 3.476802]
args= 8 FFI packed nop           median=0.2064 samples=[0.208291, 0.205858, 0.203399, 0.206363, 0.207082, 0.208574, 0.206433, 0.20528200000000002, 0.206376]
args= 8 FFI typed nop            median=0.2107 samples=[0.210675, 0.212404, 0.204429, 0.217479, 0.20813900000000002, 0.213357, 0.210672, 0.21280600000000002, 0.207518]
args= 8 FFI empty kernel         median=4.2632 samples=[4.013937, 4.374617, 3.811596, 4.371723, 3.800091, 4.322775, 4.950501, 4.263234000000001, 3.7788069999999996]
args= 8 INTJ static_compile kernel median=3.6641 samples=[3.418712, 3.569471, 4.112212, 3.664136, 4.065067, 3.587908, 17.494367999999998, 3.633054, 3.9962600000000004]
args= 8 INTJ runtime_shim kernel median=3.4976 samples=[3.497632, 3.364573, 3.551571, 3.330216, 3.5256190000000003, 3.306146, 6.551226, 3.2894609999999997, 3.52775]
args=16 FFI packed nop           median=0.2914 samples=[0.290328, 0.294724, 0.290755, 0.289949, 0.29136900000000004, 0.28871199999999997, 0.295983, 0.298707, 0.291928]
args=16 FFI typed nop            median=0.3050 samples=[0.296555, 0.309362, 0.301051, 0.30869, 0.30681, 0.302282, 0.305033, 0.309742, 0.29988400000000004]
args=16 FFI empty kernel         median=4.6850 samples=[4.285903, 5.775207999999999, 4.261256, 5.658542000000001, 4.684965, 5.974692, 4.683493, 5.915108, 4.507618000000001]
args=16 INTJ static_compile kernel median=5.1878 samples=[3.882615, 5.324111, 5.075144, 5.158857, 5.187755, 5.357303, 5.520028, 5.261824, 4.956633]
args=16 INTJ runtime_shim kernel median=3.8073 samples=[3.8073420000000002, 3.7513319999999997, 4.781277, 3.617982, 4.824491, 3.678107, 5.4690829999999995, 3.582268, 4.801863]
args=32 FFI packed nop           median=0.4803 samples=[0.477301, 0.490954, 0.477144, 0.477645, 0.4803, 0.485514, 0.499149, 0.477695, 0.48255000000000003]
args=32 FFI typed nop            median=0.4943 samples=[0.493241, 0.511999, 0.494313, 0.49537200000000003, 0.48444, 0.510908, 0.5026619999999999, 0.485333, 0.479411]
args=32 FFI empty kernel         median=5.1573 samples=[4.7924489999999995, 5.809488, 4.813886999999999, 5.993417999999999, 5.157301, 6.029592, 4.711259, 5.971825, 5.00846]
args=32 INTJ static_compile kernel median=5.5440 samples=[4.212668, 5.481921000000001, 6.065256000000001, 5.544038, 6.102825, 5.5297089999999995, 5.601521, 5.298176000000001, 6.205674]
args=32 INTJ runtime_shim kernel median=4.1745 samples=[4.123498, 4.174533, 5.7958940000000005, 4.069808, 5.916744, 4.133785, 5.457371, 4.046771000000001, 5.839058]
args=64 FFI packed nop           median=0.8416 samples=[0.8508830000000001, 0.848567, 0.867481, 0.835634, 0.831894, 0.833343, 0.8655, 0.8355549999999999, 0.84158]
args=64 FFI typed nop            median=0.8592 samples=[0.859188, 0.886892, 0.848755, 0.875969, 0.8437279999999999, 0.864633, 0.86251, 0.857993, 0.853213]
args=64 FFI empty kernel         median=5.8870 samples=[5.8314200000000005, 6.391236, 5.886994, 6.455623999999999, 5.8512830000000005, 6.377884, 5.838068, 6.3427929999999995, 5.848694]
args=64 INTJ static_compile kernel median=6.0527 samples=[5.4291149999999995, 6.0891530000000005, 6.052666, 6.103899999999999, 6.033919, 6.104944000000001, 6.039154000000001, 6.061845, 5.982916]
args=64 INTJ runtime_shim kernel median=5.9602 samples=[5.373906, 6.079761, 5.853859000000001, 6.053822, 5.960226, 6.10398, 5.919021, 6.126276, 5.877132]
elapsed_ns=4679037900
exit_status=0
ended=2026-09-28T14:41:06+08:00
```

```
$ cat /tmp/intj-bench/globals-tensor/round2_hip_module.txt
commit=c7125b1dcc97f25b7021ba02183510da8ceecfe7
started=2026-09-28T14:41:09+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_hip_module_launch.py --iters 1000 --batches 9
hsaco_sha256=d0bc01bcbace10aa262a50fc2e16c6fd2cd9a2bedb394cbf4fc38c179dee8e46 kernel=empty_kernel threads=256 shared=0 callback_count=1
iters=1000 batches=9 unit=us/call (host enqueue, no timed sync)
TVM FFI HIP HSACO         median=3.2036 samples=[3.428988, 3.216165, 3.2035839999999998, 3.1892869999999998, 3.227353, 3.1949430000000003, 3.174276, 3.198498, 3.214605]
Triton same HSACO         median=15.8945 samples=[16.091763, 15.938715, 15.862406, 15.935122999999999, 15.89445, 15.873790000000001, 15.824888999999999, 15.8963, 15.8781]
INTJ same function        median=2.9698 samples=[3.001938, 2.985468, 2.963317, 2.9674009999999997, 2.9697959999999997, 2.960659, 2.947096, 3.015987, 2.997721]
elapsed_ns=3897102906
exit_status=0
ended=2026-09-28T14:41:13+08:00
```

```
$ cat /tmp/intj-bench/globals-tensor/round2_kernel_cache.txt
commit=c7125b1dcc97f25b7021ba02183510da8ceecfe7
started=2026-09-28T14:41:13+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python -m pytest tests/test_kernel_cache.py -s -q

kernel cache, 1-word key, ns per operation

                      intj       tsl      absl
hash_only             1.36      1.35      1.36
hit/1                 1.59      2.02      1.44
hit/8                 2.34      2.11      5.21
hit/64                1.65      2.14      4.54
hit/512               2.30      2.89      5.30
hit_child             3.00      3.00      3.01
miss/1                1.24      1.57      1.56
miss/8                2.59      1.75      3.28
miss/64               2.77      2.36      3.55
miss/512              3.89      2.00      5.17

kernel cache, 2-word key, ns per operation

                      intj       tsl      absl
hash_only             1.36      1.42      1.42
hit/1                 1.88      2.57      2.25
hit/8                 1.85      2.57      5.05
hit/64                2.21      2.93      5.24
hit/512               3.53      4.12      7.39
hit_child             3.00      3.01      3.63
miss/1                1.38      3.10      1.47
miss/8                2.45      1.86      3.21
miss/64               3.13      2.10      4.39
miss/512              2.89      2.72      4.00

kernel cache, 5-word key, ns per operation

                      intj       tsl      absl
hash_only             3.58      3.58      3.58
hit/1                 3.00      3.05      2.46
hit/8                 3.20      3.42      9.49
hit/64                3.38      3.50      9.78
hit/512               4.55      4.78     12.57
hit_child             3.00      3.01      3.01
miss/1                1.38      2.10      2.24
miss/8                3.01      2.57      6.94
miss/64               2.86      1.97      6.74
miss/512              3.49      2.51      7.78
........
8 passed in 26.05s
elapsed_ns=27054477605
exit_status=0
ended=2026-09-28T14:41:40+08:00
```

```
$ cat /tmp/intj-bench/globals-tensor/round2_launch_dynamic.txt
commit=c7125b1dcc97f25b7021ba02183510da8ceecfe7
started=2026-09-28T14:41:45+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --dynamic --iters 20000 --batches 9
mode=dynamic; 20000 calls × 9 batches; median ns/call
               path     launch   spec_key
         lazy plain     3002.5      196.5
  2 dynamic options     3002.8      100.2
      int constexpr     2963.8       89.9
      str constexpr     3000.5       92.1
elapsed_ns=5895929800
exit_status=0
ended=2026-09-28T14:41:51+08:00
```

```
$ cat /tmp/intj-bench/globals-tensor/round2_launch_gpu.txt
commit=c7125b1dcc97f25b7021ba02183510da8ceecfe7
started=2026-09-28T14:40:40+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --iters 1000 --batches 9
mode=gpu; 1000 calls × 9 batches; median ns/call
               path    ns/call       Δ ns       Δ %   build ms   bind ms
           auto map     3637.9       +0.0      +0.0      83.67         -
   tensor annotated     3668.4      +30.5      +0.8       3.08         -
            generic     3551.6      -86.3      -2.4       2.71         -
 generic tensor ann     3653.2      +15.3      +0.4       2.67         -
        reduced key     3515.0     -122.9      -3.4       2.83         -
         verify off     3863.8     +226.0      +6.2       3.00         -
          verify on     3874.8     +237.0      +6.5       3.09         -
              baked     3643.3       +5.4      +0.1       2.91         -
       bound tensor     3588.9      -49.0      -1.3       2.59      0.13
      bound pointer     3585.8      -52.1      -1.4       2.78      0.12
   fixed device map     3594.9      -43.0      -1.2       2.58      0.13
fixed device no-map     3510.1     -127.8      -3.5       2.85      0.11
elapsed_ns=4110041464
exit_status=0
ended=2026-09-28T14:40:44+08:00
```

```
$ cat /tmp/intj-bench/globals-tensor/round2_launch_host.txt
commit=c7125b1dcc97f25b7021ba02183510da8ceecfe7
started=2026-09-28T14:40:44+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --no-gpu --iters 100000 --batches 9
mode=host-only; 100000 calls × 9 batches; median ns/call
               path    ns/call       Δ ns       Δ %   build ms   bind ms
           auto map       40.1       +0.0      +0.0      68.43         -
   tensor annotated       40.1       -0.0      -0.0       2.44         -
            generic       44.0       +3.9      +9.6       2.32         -
 generic tensor ann       44.3       +4.2     +10.5       2.39         -
        reduced key       41.2       +1.1      +2.6       2.24         -
         verify off       41.6       +1.5      +3.6       2.26         -
          verify on       42.5       +2.4      +6.0       2.26         -
              baked       38.2       -1.9      -4.8       2.27         -
       bound tensor       43.2       +3.1      +7.6       2.16      0.11
      bound pointer       42.0       +1.9      +4.6       2.20      0.12
   fixed device map       41.0       +0.9      +2.2       2.16      0.12
fixed device no-map       42.3       +2.2      +5.4       2.27      0.11
elapsed_ns=3228756669
exit_status=0
ended=2026-09-28T14:40:47+08:00
```

```
$ cat /tmp/intj-bench/globals-tensor/round2_launch_last_key.txt
commit=c7125b1dcc97f25b7021ba02183510da8ceecfe7
started=2026-09-28T14:40:54+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --last-key --iters 100000 --batches 9
mode=last-key; host-only; 200000 calls × 9 batches; median ns/call
count     pattern    ns/call
    4      repeat       32.5
    4   alternate       34.3
   16      repeat       61.1
   16   alternate       64.3
   32      repeat       98.2
   32   alternate      102.6
elapsed_ns=3449030574
exit_status=0
ended=2026-09-28T14:40:58+08:00
```

```
$ cat /tmp/intj-bench/globals-tensor/round2_launch_readme.txt
commit=c7125b1dcc97f25b7021ba02183510da8ceecfe7
started=2026-09-28T14:40:47+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --readme --iters 1000 --batches 9
mode=readme; 1000 calls × 9 batches; median
        path   triton us    intj us   speedup
   grid=(1,)       17.12       3.11      5.5x
   grid=(0,)       13.61       0.03    507.4x

torch_access_mode   decode ns    build s
     runtime_shim        93.0       0.02
   static_compile        95.8       0.06
      interpreter       900.9       0.00
elapsed_ns=3920866486
exit_status=0
ended=2026-09-28T14:40:51+08:00
```

```
$ cat /tmp/intj-bench/globals-tensor/round2_launch_sweep.txt
commit=c7125b1dcc97f25b7021ba02183510da8ceecfe7
started=2026-09-28T14:40:51+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --sweep --iters 100000 --batches 9
mode=sweep; host-only; 100000 calls × 9 batches; median ns/call
count   kind    ns/call
    4    int       38.8
    4 tensor       36.9
   16    int       67.3
   16 tensor       62.9
   32    int      105.1
   32 tensor      108.7
elapsed_ns=3054576924
exit_status=0
ended=2026-09-28T14:40:54+08:00
```

```
$ cat /tmp/intj-bench/globals-tensor/round2_launch_tuned.txt
commit=c7125b1dcc97f25b7021ba02183510da8ceecfe7
started=2026-09-28T14:41:40+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --tuned --iters 20000 --batches 9
 tuned hit:  3060.0 ns/launch
     plain:  3099.8 ns/launch
elapsed_ns=4798572914
exit_status=0
ended=2026-09-28T14:41:45+08:00
```

### `/tmp/intj-bench/globals-tensor-rerun`

```
$ cat /tmp/intj-bench/globals-tensor-rerun/round0_ffi_compare.txt
commit=c7125b1dcc97f25b7021ba02183510da8ceecfe7
started=2026-09-28T14:45:48+08:00
command=HIP_VISIBLE_DEVICES=0 PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_ffi_compare.py --iters 1000 --batches 9
iters=1000 batches=9 unit=us/call (host-only, no timed sync)
FFI packed nop             median=0.1447 samples=[0.153578, 0.144197, 0.14473, 0.146314, 0.143641, 0.14633000000000002, 0.14466900000000002, 0.143153, 0.143458]
FFI typed nop              median=0.1502 samples=[0.148638, 0.151251, 0.149414, 0.150643, 0.149481, 0.14954599999999998, 0.151608, 0.151134, 0.15017599999999998]
FFI empty kernel           median=3.3237 samples=[3.604548, 3.53355, 3.4035, 3.220075, 3.217884, 3.323687, 3.2085100000000004, 3.342283, 3.2003519999999996]
INTJ empty kernel          median=2.9707 samples=[3.0568519999999997, 3.008516, 2.982901, 2.824047, 2.812658, 2.9524160000000004, 2.982491, 2.9428069999999997, 2.970658]
INTJ fixed-device kernel   median=2.8296 samples=[2.893394, 3.014449, 2.98213, 2.868839, 2.814538, 2.82852, 2.818715, 2.826105, 2.829602]
FFI packed nop mixed       median=0.1740 samples=[0.170624, 0.17316700000000002, 0.174547, 0.17396199999999998, 0.17358400000000002, 0.178451, 0.173957, 0.175588, 0.172361]
FFI typed nop mixed        median=0.1758 samples=[0.17234, 0.183672, 0.17756, 0.178488, 0.17488800000000002, 0.178367, 0.175754, 0.175707, 0.17580400000000002]
FFI mixed kernel           median=3.3444 samples=[3.344439, 3.472791, 3.3465100000000003, 3.296679, 3.231469, 3.3624169999999998, 3.2734360000000002, 3.364149, 3.25615]
INTJ mixed kernel          median=2.9864 samples=[3.188031, 3.09069, 2.957415, 2.9509369999999997, 2.927699, 2.9125039999999998, 2.986431, 3.0665790000000004, 3.055473]
INTJ fixed mixed kernel    median=2.8975 samples=[3.083211, 3.007288, 2.897486, 2.8981559999999997, 2.836816, 2.7975320000000004, 2.89453, 2.902342, 2.8680309999999998]
elapsed_ns=3643887424
exit_status=0
ended=2026-09-28T14:45:51+08:00
```

```
$ cat /tmp/intj-bench/globals-tensor-rerun/round0_ffi_sweep.txt
commit=c7125b1dcc97f25b7021ba02183510da8ceecfe7
started=2026-09-28T14:45:51+08:00
command=HIP_VISIBLE_DEVICES=0 PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_ffi_compare.py --sweep --iters 1000 --batches 9
counts=[0, 3, 5, 8, 16, 32, 64] iters=1000 batches=9 unit=us/call (host-only, no timed sync); INTJ device-bound, no specialization key
args= 0 FFI packed nop           median=0.1156 samples=[0.1205, 0.115759, 0.11523399999999999, 0.11984600000000001, 0.115163, 0.115617, 0.115384, 0.11604, 0.114744]
args= 0 FFI typed nop            median=0.1204 samples=[0.120408, 0.120742, 0.117904, 0.124376, 0.122199, 0.117664, 0.120005, 0.122446, 0.118214]
args= 0 FFI empty kernel         median=2.3214 samples=[1.7042460000000001, 3.148387, 2.321383, 3.187423, 2.272316, 3.133978, 2.26004, 3.173683, 2.284319]
args= 0 INTJ static_compile kernel median=3.2655 samples=[3.229863, 3.254711, 3.295962, 3.246383, 3.286006, 3.212809, 3.2655369999999997, 3.322562, 3.301141]
args= 0 INTJ runtime_shim kernel median=3.1894 samples=[3.1894099999999996, 3.072681, 3.274197, 3.055411, 3.347524, 3.044677, 3.318837, 3.065426, 3.213986]
args= 3 FFI packed nop           median=0.1441 samples=[0.14629599999999998, 0.145037, 0.143512, 0.14409200000000003, 0.14815199999999998, 0.143917, 0.143421, 0.144893, 0.143429]
args= 3 FFI typed nop            median=0.1477 samples=[0.147891, 0.14898599999999998, 0.145597, 0.15259999999999999, 0.144398, 0.147476, 0.148392, 0.14768799999999999, 0.147177]
args= 3 FFI empty kernel         median=3.6917 samples=[3.691747, 3.866889, 3.646004, 3.879963, 3.678461, 3.849193, 3.641582, 3.8465610000000003, 3.6479920000000003]
args= 3 INTJ static_compile kernel median=3.5408 samples=[3.313914, 3.484212, 3.9359140000000004, 3.5408020000000002, 3.8903939999999997, 3.495892, 3.927759, 3.497871, 3.903199]
args= 3 INTJ runtime_shim kernel median=3.4431 samples=[3.299727, 3.402703, 3.529163, 3.443137, 4.072457, 3.43239, 3.532007, 3.3872489999999997, 3.474049]
args= 5 FFI packed nop           median=0.1728 samples=[0.172657, 0.174974, 0.173089, 0.171731, 0.172209, 0.172068, 0.17311400000000002, 0.17284, 0.17408]
args= 5 FFI typed nop            median=0.1766 samples=[0.176633, 0.17935900000000002, 0.176327, 0.17893299999999998, 0.174808, 0.177844, 0.17988900000000002, 0.176428, 0.17535499999999998]
args= 5 FFI empty kernel         median=3.8258 samples=[3.746414, 4.2228639999999995, 3.825754, 4.293924, 3.775061, 4.326480999999999, 3.8083560000000003, 4.252607, 3.772494]
args= 5 INTJ static_compile kernel median=3.4740 samples=[3.3029520000000003, 3.5238009999999997, 3.486792, 3.4740230000000003, 3.46728, 3.4911640000000004, 3.479146, 3.464583, 3.455036]
args= 5 INTJ runtime_shim kernel median=3.4682 samples=[3.340684, 3.4906930000000003, 4.072514, 3.426552, 3.4682049999999998, 3.4191019999999996, 4.035283, 3.423487, 4.048169]
args= 8 FFI packed nop           median=0.2063 samples=[0.205316, 0.206451, 0.207133, 0.206236, 0.20629599999999998, 0.208732, 0.205614, 0.206549, 0.206101]
args= 8 FFI typed nop            median=0.2102 samples=[0.207876, 0.210696, 0.212876, 0.210023, 0.208781, 0.210196, 0.206559, 0.212141, 0.217098]
args= 8 FFI empty kernel         median=3.9505 samples=[3.950467, 4.339060000000001, 3.818551, 4.4429099999999995, 3.829501, 4.370158, 3.842414, 4.343903, 3.771993]
args= 8 INTJ static_compile kernel median=3.7199 samples=[3.434078, 3.5654879999999998, 4.144466, 3.719904, 4.213558, 3.716875, 4.184573, 3.661366, 4.148803]
args= 8 INTJ runtime_shim kernel median=3.4600 samples=[3.4600210000000002, 3.2986, 3.622, 3.282872, 3.605976, 3.272371, 3.626572, 3.285572, 3.6180790000000003]
args=16 FFI packed nop           median=0.2924 samples=[0.29078699999999996, 0.296958, 0.294976, 0.289011, 0.29359199999999996, 0.291735, 0.291279, 0.29321600000000003, 0.292362]
args=16 FFI typed nop            median=0.3019 samples=[0.301255, 0.30237, 0.30193200000000003, 0.301162, 0.30097399999999996, 0.304947, 0.301865, 0.304302, 0.30388299999999996]
args=16 FFI empty kernel         median=4.3127 samples=[4.312667, 5.952342, 4.237094, 5.971914, 4.187978, 5.811263, 4.199832000000001, 5.929265, 4.2265299999999995]
args=16 INTJ static_compile kernel median=5.5600 samples=[3.8052020000000004, 5.529705, 5.687192, 5.392532, 5.631946, 5.296721, 5.560049, 5.662897, 5.575786]
args=16 INTJ runtime_shim kernel median=3.8352 samples=[3.835214, 3.706103, 5.668046, 3.636339, 5.675173, 3.6428290000000003, 5.613406, 3.667477, 5.6814]
args=32 FFI packed nop           median=0.4806 samples=[0.48084899999999997, 0.481007, 0.480132, 0.471394, 0.482277, 0.48259199999999997, 0.473418, 0.468305, 0.48063]
args=32 FFI typed nop            median=0.4938 samples=[0.494406, 0.491405, 0.48306, 0.499791, 0.5124, 0.494741, 0.493534, 0.487839, 0.49380900000000005]
args=32 FFI empty kernel         median=4.7807 samples=[4.780714, 5.873279, 4.6835569999999995, 5.990916, 4.6757420000000005, 5.906928, 4.706623, 6.072680999999999, 4.712506]
args=32 INTJ static_compile kernel median=5.6433 samples=[4.08604, 5.666881, 6.016310000000001, 5.6411679999999995, 5.7643, 5.371279, 5.643314, 5.580329, 5.723817]
args=32 INTJ runtime_shim kernel median=4.0870 samples=[4.071791, 4.080589, 5.695054, 4.075131, 5.608489, 4.08697, 5.547376, 4.077634, 5.6651620000000005]
args=64 FFI packed nop           median=0.8355 samples=[0.848854, 0.881726, 0.82733, 0.823481, 0.835452, 0.8426910000000001, 0.836503, 0.825061, 0.829723]
args=64 FFI typed nop            median=0.8576 samples=[0.853041, 0.9005259999999999, 0.8619249999999999, 0.856245, 0.857313, 0.866468, 0.870324, 0.857632, 0.849746]
args=64 FFI empty kernel         median=5.8567 samples=[5.736612, 6.459969, 5.8534679999999994, 6.535884, 5.856698000000001, 6.511851, 5.828785, 6.417928, 5.836647]
args=64 INTJ static_compile kernel median=6.1728 samples=[5.488304, 6.172829, 6.1512389999999995, 6.2412209999999995, 6.139778000000001, 6.222733, 6.1935270000000004, 6.026446, 6.252612]
args=64 INTJ runtime_shim kernel median=6.0354 samples=[5.2691989999999995, 6.132143, 5.925452, 6.114351, 5.857419, 6.035425, 5.924372999999999, 6.210701, 6.0504620000000005]
elapsed_ns=4463771994
exit_status=0
ended=2026-09-28T14:45:56+08:00
```

```
$ cat /tmp/intj-bench/globals-tensor-rerun/round0_hip_module.txt
commit=c7125b1dcc97f25b7021ba02183510da8ceecfe7
started=2026-09-28T14:45:56+08:00
command=HIP_VISIBLE_DEVICES=0 PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_hip_module_launch.py --iters 1000 --batches 9
hsaco_sha256=d0bc01bcbace10aa262a50fc2e16c6fd2cd9a2bedb394cbf4fc38c179dee8e46 kernel=empty_kernel threads=256 shared=0 callback_count=1
iters=1000 batches=9 unit=us/call (host enqueue, no timed sync)
TVM FFI HIP HSACO         median=3.1709 samples=[3.356482, 3.224665, 3.188948, 3.173826, 3.17092, 3.149489, 3.125271, 3.143926, 3.1258049999999997]
Triton same HSACO         median=15.8437 samples=[15.876903, 15.883333, 15.945818, 15.857138, 15.843699, 15.787498, 15.811914, 15.736036, 15.764873]
INTJ same function        median=2.9886 samples=[3.002282, 3.018031, 3.007585, 2.998235, 2.988645, 2.9639409999999997, 2.971171, 2.972812, 2.952129]
elapsed_ns=3664618215
exit_status=0
ended=2026-09-28T14:45:59+08:00
```

```
$ cat /tmp/intj-bench/globals-tensor-rerun/round0_launch_gpu.txt
commit=c7125b1dcc97f25b7021ba02183510da8ceecfe7
started=2026-09-28T14:45:44+08:00
command=HIP_VISIBLE_DEVICES=0 PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --iters 1000 --batches 9
mode=gpu; 1000 calls × 9 batches; median ns/call
               path    ns/call       Δ ns       Δ %   build ms   bind ms
           auto map     3117.5       +0.0      +0.0      81.47         -
   tensor annotated     3102.1      -15.5      -0.5       2.97         -
            generic     3047.4      -70.1      -2.2       2.56         -
 generic tensor ann     2979.7     -137.9      -4.4       2.49         -
        reduced key     3018.8      -98.7      -3.2       2.74         -
         verify off     2989.7     -127.9      -4.1       2.76         -
          verify on     2913.0     -204.6      -6.6       2.70         -
              baked     2879.6     -237.9      -7.6       2.71         -
       bound tensor     2883.6     -233.9      -7.5       2.53      0.13
      bound pointer     2874.9     -242.6      -7.8       2.49      0.12
   fixed device map     2875.8     -241.8      -7.8       2.51      0.13
fixed device no-map     2897.2     -220.3      -7.1       2.58      0.11
elapsed_ns=3763324410
exit_status=0
ended=2026-09-28T14:45:48+08:00
```

```
$ cat /tmp/intj-bench/globals-tensor-rerun/round1_ffi_compare.txt
commit=c7125b1dcc97f25b7021ba02183510da8ceecfe7
started=2026-09-28T14:46:03+08:00
command=HIP_VISIBLE_DEVICES=0 PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_ffi_compare.py --iters 1000 --batches 9
iters=1000 batches=9 unit=us/call (host-only, no timed sync)
FFI packed nop             median=0.1451 samples=[0.152911, 0.145485, 0.145264, 0.145147, 0.14554, 0.14454499999999998, 0.143209, 0.142813, 0.142226]
FFI typed nop              median=0.1511 samples=[0.14971600000000002, 0.15110900000000002, 0.152415, 0.15319300000000002, 0.14879900000000001, 0.15121700000000002, 0.147549, 0.150617, 0.15231299999999998]
FFI empty kernel           median=3.6737 samples=[3.5721, 4.504148, 3.673723, 3.7180880000000003, 3.617004, 4.44163, 3.546799, 4.310623, 3.5302040000000003]
INTJ empty kernel          median=3.5133 samples=[3.3699850000000002, 3.523192, 4.156353, 3.513314, 3.4889200000000002, 3.479668, 3.9889099999999997, 3.505689, 3.998569]
INTJ fixed-device kernel   median=3.3629 samples=[3.048775, 3.332373, 3.631424, 3.3784870000000002, 3.445392, 3.151552, 3.38923, 3.1401019999999997, 3.362905]
FFI packed nop mixed       median=0.1743 samples=[0.172339, 0.17555600000000002, 0.175259, 0.17247900000000002, 0.175428, 0.17426499999999998, 0.17371799999999998, 0.172602, 0.174457]
FFI typed nop mixed        median=0.1771 samples=[0.176453, 0.17668999999999999, 0.178187, 0.17828899999999998, 0.17580199999999999, 0.177051, 0.174206, 0.17710900000000002, 0.17912999999999998]
FFI mixed kernel           median=3.7700 samples=[3.684363, 4.48926, 3.8322179999999997, 3.816157, 3.639805, 3.770008, 3.642694, 4.413067, 3.628018]
INTJ mixed kernel          median=3.4790 samples=[3.456422, 3.594076, 3.564319, 3.576966, 3.496061, 3.478978, 3.3978960000000002, 3.478853, 3.4205189999999996]
INTJ fixed mixed kernel    median=3.4724 samples=[3.4779769999999997, 3.6015859999999997, 3.5650459999999997, 3.590034, 3.472389, 3.457097, 3.4387779999999997, 3.447091, 3.42661]
elapsed_ns=3723751433
exit_status=0
ended=2026-09-28T14:46:07+08:00
```

```
$ cat /tmp/intj-bench/globals-tensor-rerun/round1_ffi_sweep.txt
commit=c7125b1dcc97f25b7021ba02183510da8ceecfe7
started=2026-09-28T14:46:07+08:00
command=HIP_VISIBLE_DEVICES=0 PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_ffi_compare.py --sweep --iters 1000 --batches 9
counts=[0, 3, 5, 8, 16, 32, 64] iters=1000 batches=9 unit=us/call (host-only, no timed sync); INTJ device-bound, no specialization key
args= 0 FFI packed nop           median=0.1166 samples=[0.12205200000000001, 0.116794, 0.11662900000000001, 0.11756100000000001, 0.11565500000000001, 0.116355, 0.11486299999999999, 0.117699, 0.11662600000000001]
args= 0 FFI typed nop            median=0.1185 samples=[0.12477500000000001, 0.11854600000000001, 0.118881, 0.11852700000000001, 0.117392, 0.118291, 0.11742100000000001, 0.120447, 0.118612]
args= 0 FFI empty kernel         median=1.6981 samples=[1.698059, 1.919293, 1.6841110000000001, 1.869259, 1.5802070000000001, 1.870137, 1.571356, 1.8827129999999999, 1.6463800000000002]
args= 0 INTJ static_compile kernel median=2.7538 samples=[3.194584, 2.768576, 2.7637069999999997, 2.667384, 2.6816619999999998, 2.664349, 2.688981, 2.80946, 2.75375]
args= 0 INTJ runtime_shim kernel median=2.7039 samples=[2.884058, 2.757723, 2.7824340000000003, 2.663981, 2.6946179999999997, 2.668539, 2.703875, 2.683042, 2.767897]
args= 3 FFI packed nop           median=0.1437 samples=[0.143747, 0.14477299999999999, 0.14485900000000002, 0.143416, 0.143733, 0.144886, 0.142496, 0.14255400000000001, 0.142494]
args= 3 FFI typed nop            median=0.1477 samples=[0.145998, 0.148886, 0.14693, 0.148489, 0.14535, 0.149521, 0.145672, 0.148671, 0.147695]
args= 3 FFI empty kernel         median=3.2130 samples=[3.343368, 3.26517, 3.196716, 3.2129540000000003, 3.158711, 3.213906, 3.161074, 3.2413339999999997, 3.1699810000000004]
args= 3 INTJ static_compile kernel median=2.9513 samples=[3.0082220000000004, 2.872085, 2.989592, 2.8297489999999996, 2.965861, 2.829118, 2.951342, 2.881423, 2.985253]
args= 3 INTJ runtime_shim kernel median=2.8395 samples=[3.019877, 2.839453, 2.8458989999999997, 2.828547, 2.772384, 2.835807, 2.8314369999999998, 2.893059, 2.881859]
args= 5 FFI packed nop           median=0.1736 samples=[0.17358099999999999, 0.173177, 0.175216, 0.17337799999999998, 0.17468999999999998, 0.171868, 0.178887, 0.173733, 0.173509]
args= 5 FFI typed nop            median=0.1786 samples=[0.17660599999999999, 0.17857, 0.17581899999999998, 0.18024500000000002, 0.179258, 0.17884899999999998, 0.17586000000000002, 0.17678899999999997, 0.180377]
args= 5 FFI empty kernel         median=3.2595 samples=[3.3856010000000003, 3.3536819999999996, 3.212326, 3.316359, 3.1917020000000003, 3.306879, 3.193558, 3.2595479999999997, 3.2572829999999997]
args= 5 INTJ static_compile kernel median=2.8799 samples=[3.0256480000000003, 2.902279, 2.8859310000000002, 2.823732, 2.864281, 2.820021, 2.876789, 2.8799430000000004, 2.919159]
args= 5 INTJ runtime_shim kernel median=2.8440 samples=[3.034295, 2.8677539999999997, 2.843983, 2.768246, 2.825533, 2.764466, 2.8274920000000003, 2.870689, 2.861605]
args= 8 FFI packed nop           median=0.2071 samples=[0.206961, 0.210249, 0.207591, 0.205195, 0.20713499999999999, 0.204004, 0.20697900000000002, 0.208459, 0.209955]
args= 8 FFI typed nop            median=0.2108 samples=[0.20827, 0.213118, 0.210785, 0.20837, 0.208487, 0.211205, 0.20853899999999997, 0.211811, 0.213584]
args= 8 FFI empty kernel         median=3.3856 samples=[3.508364, 3.438484, 3.280973, 3.3897269999999997, 3.248456, 3.385628, 3.2527489999999997, 3.381754, 3.464746]
args= 8 INTJ static_compile kernel median=3.0523 samples=[3.052273, 2.919751, 3.131652, 2.9817020000000003, 3.074668, 2.923739, 3.073562, 2.9861050000000002, 3.05443]
args= 8 INTJ runtime_shim kernel median=2.9397 samples=[5.3914930000000005, 2.939724, 3.0226819999999996, 2.891973, 2.985484, 2.885241, 2.997607, 2.8884499999999997, 2.898249]
args=16 FFI packed nop           median=0.2931 samples=[0.295279, 0.302201, 0.29306, 0.286495, 0.291931, 0.29074099999999997, 0.29782, 0.293786, 0.29144400000000004]
args=16 FFI typed nop            median=0.3036 samples=[0.304712, 0.309817, 0.30311099999999996, 0.301473, 0.303551, 0.304794, 0.30361, 0.302056, 0.29435]
args=16 FFI empty kernel         median=3.6427 samples=[3.735007, 3.691418, 3.560027, 3.642665, 3.522739, 3.6455189999999997, 3.533831, 3.701773, 3.535278]
args=16 INTJ static_compile kernel median=3.3670 samples=[3.2987979999999997, 3.3670210000000003, 3.575339, 3.3067130000000002, 3.5344029999999997, 3.303509, 3.5040169999999997, 3.354098, 3.473935]
args=16 INTJ runtime_shim kernel median=3.2977 samples=[3.297748, 3.147141, 3.370116, 3.045163, 3.34667, 3.018471, 3.3606599999999998, 3.02514, 3.371676]
args=32 FFI packed nop           median=0.4798 samples=[0.477949, 0.482846, 0.479764, 0.481305, 0.476209, 0.475374, 0.47623000000000004, 0.48389, 0.482435]
args=32 FFI typed nop            median=0.4928 samples=[0.49161, 0.507316, 0.488168, 0.499855, 0.49275599999999997, 0.491289, 0.49435700000000005, 0.497166, 0.479556]
args=32 FFI empty kernel         median=4.0986 samples=[4.173618, 4.145706000000001, 3.9600929999999996, 4.115589, 3.9414160000000003, 4.098618999999999, 3.948477, 4.1055, 3.965121]
args=32 INTJ static_compile kernel median=3.6608 samples=[3.76831, 3.6608470000000004, 3.683552, 3.571431, 3.6725770000000004, 3.580143, 3.6662310000000002, 3.597878, 3.6590700000000003]
args=32 INTJ runtime_shim kernel median=3.6021 samples=[3.602093, 3.518596, 3.653699, 3.467623, 3.6314360000000003, 3.459366, 3.642659, 3.454016, 3.635668]
args=64 FFI packed nop           median=0.8368 samples=[0.835799, 0.840693, 0.849433, 0.836831, 0.868928, 0.83399, 0.838741, 0.808447, 0.818726]
args=64 FFI typed nop            median=0.8573 samples=[0.850974, 0.886828, 0.8724270000000001, 1.007373, 0.855052, 0.883802, 0.857321, 0.844562, 0.84205]
args=64 FFI empty kernel         median=5.0181 samples=[5.178738999999999, 5.141826, 5.035518, 5.014075, 5.009861, 5.017677, 5.018067, 5.01396, 5.018211]
args=64 INTJ static_compile kernel median=4.5648 samples=[4.613779, 4.599155, 4.581493999999999, 4.564819000000001, 4.536613999999999, 4.564767, 4.543311, 4.580161, 4.5393609999999995]
args=64 INTJ runtime_shim kernel median=4.5109 samples=[4.525081, 4.595815, 4.4879560000000005, 4.510896, 4.4860240000000005, 4.566409999999999, 4.4748149999999995, 4.5507539999999995, 4.484001]
elapsed_ns=4281153307
exit_status=0
ended=2026-09-28T14:46:11+08:00
```

```
$ cat /tmp/intj-bench/globals-tensor-rerun/round1_hip_module.txt
commit=c7125b1dcc97f25b7021ba02183510da8ceecfe7
started=2026-09-28T14:46:11+08:00
command=HIP_VISIBLE_DEVICES=0 PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_hip_module_launch.py --iters 1000 --batches 9
hsaco_sha256=d0bc01bcbace10aa262a50fc2e16c6fd2cd9a2bedb394cbf4fc38c179dee8e46 kernel=empty_kernel threads=256 shared=0 callback_count=1
iters=1000 batches=9 unit=us/call (host enqueue, no timed sync)
TVM FFI HIP HSACO         median=3.1294 samples=[3.410924, 3.194051, 3.171125, 3.129356, 3.129854, 3.027783, 3.050723, 3.03727, 3.03375]
Triton same HSACO         median=15.8765 samples=[16.183870000000002, 15.924109, 15.878540000000001, 15.894797, 15.876468000000001, 15.771135000000001, 15.794899, 15.790514, 15.776019]
INTJ same function        median=2.9260 samples=[2.985632, 2.911857, 2.930965, 2.978953, 2.890462, 2.8780479999999997, 2.9260010000000003, 2.9394850000000003, 2.881933]
elapsed_ns=3644146951
exit_status=0
ended=2026-09-28T14:46:15+08:00
```

```
$ cat /tmp/intj-bench/globals-tensor-rerun/round1_launch_gpu.txt
commit=c7125b1dcc97f25b7021ba02183510da8ceecfe7
started=2026-09-28T14:45:59+08:00
command=HIP_VISIBLE_DEVICES=0 PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --iters 1000 --batches 9
mode=gpu; 1000 calls × 9 batches; median ns/call
               path    ns/call       Δ ns       Δ %   build ms   bind ms
           auto map     3089.4       +0.0      +0.0      81.85         -
   tensor annotated     3103.7      +14.2      +0.5       2.98         -
            generic     2999.2      -90.3      -2.9       2.62         -
 generic tensor ann     2996.6      -92.8      -3.0       2.55         -
        reduced key     2971.0     -118.5      -3.8       2.80         -
         verify off     2988.9     -100.5      -3.3       2.82         -
          verify on     2988.0     -101.4      -3.3       2.79         -
              baked     2936.3     -153.2      -5.0       2.77         -
       bound tensor     2964.9     -124.5      -4.0       2.54      0.12
      bound pointer     2930.2     -159.3      -5.2       2.51      0.12
   fixed device map     2941.4     -148.0      -4.8       2.51      0.13
fixed device no-map     2959.4     -130.1      -4.2       2.54      0.11
elapsed_ns=3803908008
exit_status=0
ended=2026-09-28T14:46:03+08:00
```

```
$ cat /tmp/intj-bench/globals-tensor-rerun/round2_ffi_compare.txt
commit=c7125b1dcc97f25b7021ba02183510da8ceecfe7
started=2026-09-28T14:46:19+08:00
command=HIP_VISIBLE_DEVICES=0 PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_ffi_compare.py --iters 1000 --batches 9
iters=1000 batches=9 unit=us/call (host-only, no timed sync)
FFI packed nop             median=0.1439 samples=[0.160502, 0.143828, 0.145061, 0.145705, 0.143246, 0.144429, 0.142399, 0.143934, 0.14387200000000003]
FFI typed nop              median=0.1494 samples=[0.146006, 0.14983600000000002, 0.147324, 0.151758, 0.14693, 0.151238, 0.148033, 0.150011, 0.149436]
FFI empty kernel           median=3.7622 samples=[3.587721, 3.94887, 3.82846, 3.762232, 3.638068, 4.341061, 3.541427, 4.283162, 3.537692]
INTJ empty kernel          median=3.5946 samples=[3.119625, 3.744934, 3.749526, 3.586402, 3.526372, 3.513681, 3.950714, 3.5946100000000003, 3.9248220000000003]
INTJ fixed-device kernel   median=3.3934 samples=[3.016703, 3.5915749999999997, 3.705403, 3.453767, 3.5389250000000003, 3.156853, 3.3915279999999997, 3.178523, 3.393354]
FFI packed nop mixed       median=0.1729 samples=[0.172913, 0.171854, 0.17338499999999998, 0.172377, 0.171304, 0.175745, 0.174481, 0.17407, 0.172491]
FFI typed nop mixed        median=0.1764 samples=[0.178772, 0.17944900000000003, 0.176216, 0.176673, 0.174423, 0.176412, 0.175736, 0.177147, 0.17606200000000002]
FFI mixed kernel           median=3.7925 samples=[3.7085630000000003, 3.991822, 3.792487, 3.8550079999999998, 3.6690970000000003, 3.865748, 3.681002, 4.351548, 3.687502]
INTJ mixed kernel          median=3.6130 samples=[4.087366, 3.761576, 3.628105, 3.615363, 3.593165, 3.57535, 3.408922, 3.613014, 3.44758]
INTJ fixed mixed kernel    median=3.5901 samples=[3.7681199999999997, 3.744174, 3.618286, 3.627977, 3.550928, 3.516995, 3.504444, 3.557045, 3.5900700000000003]
elapsed_ns=3705852627
exit_status=0
ended=2026-09-28T14:46:22+08:00
```

```
$ cat /tmp/intj-bench/globals-tensor-rerun/round2_ffi_sweep.txt
commit=c7125b1dcc97f25b7021ba02183510da8ceecfe7
started=2026-09-28T14:46:22+08:00
command=HIP_VISIBLE_DEVICES=0 PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_ffi_compare.py --sweep --iters 1000 --batches 9
counts=[0, 3, 5, 8, 16, 32, 64] iters=1000 batches=9 unit=us/call (host-only, no timed sync); INTJ device-bound, no specialization key
args= 0 FFI packed nop           median=0.1165 samples=[0.116991, 0.117021, 0.117328, 0.114425, 0.114232, 0.11988299999999999, 0.11653100000000001, 0.116448, 0.115316]
args= 0 FFI typed nop            median=0.1183 samples=[0.122184, 0.120292, 0.11726600000000001, 0.121087, 0.116616, 0.11829300000000001, 0.11587900000000001, 0.122167, 0.116618]
args= 0 FFI empty kernel         median=1.7300 samples=[1.730035, 1.9020380000000001, 1.689531, 1.924396, 1.6144020000000001, 1.9144919999999999, 1.604653, 1.8307149999999999, 1.628438]
args= 0 INTJ static_compile kernel median=2.7369 samples=[3.0909229999999996, 2.8188, 2.80077, 2.694019, 2.694752, 2.695937, 2.704317, 2.755456, 2.736863]
args= 0 INTJ runtime_shim kernel median=2.7161 samples=[2.854578, 2.7979670000000003, 2.84535, 2.666436, 2.716107, 2.6756170000000004, 2.714527, 2.69527, 2.7571309999999998]
args= 3 FFI packed nop           median=0.1446 samples=[0.14984, 0.14547100000000002, 0.144582, 0.143727, 0.144381, 0.14444200000000001, 0.146786, 0.14449199999999998, 0.144869]
args= 3 FFI typed nop            median=0.1491 samples=[0.14606899999999998, 0.148598, 0.147071, 0.15006, 0.149118, 0.149505, 0.149603, 0.15052000000000001, 0.14867599999999997]
args= 3 FFI empty kernel         median=3.2894 samples=[3.341346, 3.30273, 3.269249, 3.291646, 3.17902, 3.289399, 3.1702820000000003, 3.3011060000000003, 3.228539]
args= 3 INTJ static_compile kernel median=2.9063 samples=[3.000383, 2.9062710000000003, 2.887321, 2.9016509999999998, 2.9732350000000003, 2.8596280000000003, 2.95127, 2.947127, 2.883681]
args= 3 INTJ runtime_shim kernel median=2.9022 samples=[2.990267, 2.902214, 2.907113, 2.8238879999999997, 2.907751, 2.805121, 2.9319119999999996, 2.8188449999999996, 2.897838]
args= 5 FFI packed nop           median=0.1732 samples=[0.17527099999999998, 0.17223500000000003, 0.173559, 0.171274, 0.17809899999999998, 0.173119, 0.171951, 0.17322300000000002, 0.178157]
args= 5 FFI typed nop            median=0.1771 samples=[0.17738800000000002, 0.178429, 0.174767, 0.182678, 0.174343, 0.177587, 0.17419900000000002, 0.173148, 0.17708600000000002]
args= 5 FFI empty kernel         median=3.3019 samples=[3.364717, 3.378627, 3.285383, 3.264202, 3.303255, 3.307509, 3.2256460000000002, 3.30188, 3.267688]
args= 5 INTJ static_compile kernel median=2.9187 samples=[3.026509, 2.946654, 2.899123, 2.9187399999999997, 2.929575, 2.9267179999999997, 2.9050599999999998, 2.9128429999999996, 2.9107469999999998]
args= 5 INTJ runtime_shim kernel median=2.8861 samples=[3.0009789999999996, 2.8860970000000004, 2.8541849999999998, 2.916252, 2.875658, 2.92586, 2.851866, 2.902555, 2.870244]
args= 8 FFI packed nop           median=0.2069 samples=[0.206396, 0.20686000000000002, 0.20289500000000002, 0.20990299999999998, 0.211461, 0.208707, 0.206338, 0.202486, 0.20827099999999998]
args= 8 FFI typed nop            median=0.2102 samples=[0.21024, 0.209376, 0.20441700000000002, 0.213876, 0.208976, 0.21294200000000002, 0.20713800000000002, 0.21391, 0.21275899999999998]
args= 8 FFI empty kernel         median=3.3906 samples=[3.4926, 3.4507660000000002, 3.285163, 3.4193960000000003, 3.317161, 3.403651, 3.27567, 3.390635, 3.266929]
args= 8 INTJ static_compile kernel median=3.0432 samples=[3.049233, 2.97654, 3.038506, 3.063477, 3.0432449999999998, 3.034632, 3.075467, 3.019391, 3.067666]
args= 8 INTJ runtime_shim kernel median=2.9084 samples=[3.039085, 2.9519029999999997, 2.894201, 2.961522, 2.880537, 2.918829, 2.906236, 2.908386, 2.875374]
args=16 FFI packed nop           median=0.2933 samples=[0.293308, 0.296204, 0.291483, 0.29810000000000003, 0.292665, 0.290272, 0.293313, 0.290828, 0.294365]
args=16 FFI typed nop            median=0.3023 samples=[0.318233, 0.306147, 0.30080900000000005, 0.301628, 0.301962, 0.303939, 0.30012099999999997, 0.304813, 0.302325]
args=16 FFI empty kernel         median=3.6545 samples=[3.701926, 3.7494699999999996, 3.5567480000000002, 3.727842, 3.554238, 3.691447, 3.539152, 3.654469, 3.560038]
args=16 INTJ static_compile kernel median=3.3277 samples=[3.250549, 3.374956, 3.423592, 3.3924380000000003, 3.2938829999999997, 3.312163, 3.4787779999999997, 3.252294, 3.327651]
args=16 INTJ runtime_shim kernel median=3.2523 samples=[3.252325, 3.1672919999999998, 3.351909, 3.0843949999999998, 3.267734, 3.054024, 3.384463, 3.0810120000000003, 3.323904]
args=32 FFI packed nop           median=0.4789 samples=[0.48344400000000004, 0.47462099999999996, 0.482499, 0.477814, 0.470625, 0.480026, 0.478901, 0.48351900000000003, 0.476656]
args=32 FFI typed nop            median=0.4945 samples=[0.48524700000000004, 0.49845999999999996, 0.488808, 0.49478500000000003, 0.485536, 0.495452, 0.49446300000000004, 0.49386399999999997, 0.495907]
args=32 FFI empty kernel         median=4.1319 samples=[4.19057, 4.13189, 3.982771, 4.20484, 4.029402, 4.156959, 4.00013, 4.365224, 4.04623]
args=32 INTJ static_compile kernel median=3.6354 samples=[3.541577, 3.635377, 3.662944, 3.598509, 3.769024, 3.588013, 3.676492, 3.599501, 3.8029450000000002]
args=32 INTJ runtime_shim kernel median=3.5127 samples=[3.4943560000000002, 3.512692, 3.655372, 3.475357, 3.639905, 3.4582710000000003, 3.6127779999999996, 3.473769, 3.682988]
args=64 FFI packed nop           median=0.8412 samples=[0.840058, 0.836606, 0.841158, 0.831309, 0.8416520000000001, 0.848701, 0.863586, 0.835174, 0.842332]
args=64 FFI typed nop            median=0.8632 samples=[0.877962, 0.861579, 0.861279, 0.864047, 0.863205, 0.865713, 0.8616050000000001, 0.870171, 0.852075]
args=64 FFI empty kernel         median=5.0959 samples=[5.187514, 5.176984999999999, 5.122669, 5.049733, 5.098482, 5.0624080000000005, 5.095889000000001, 5.060242000000001, 5.053656]
args=64 INTJ static_compile kernel median=4.5902 samples=[4.602747, 4.590527, 4.6150150000000005, 4.613549, 4.588175000000001, 4.586639, 4.590222, 4.580063, 4.584007]
args=64 INTJ runtime_shim kernel median=4.5572 samples=[4.638999, 4.566881, 4.555085, 4.5571530000000005, 4.517786, 4.555036, 4.578724, 4.580839, 4.518665]
elapsed_ns=4304000513
exit_status=0
ended=2026-09-28T14:46:27+08:00
```

```
$ cat /tmp/intj-bench/globals-tensor-rerun/round2_hip_module.txt
commit=c7125b1dcc97f25b7021ba02183510da8ceecfe7
started=2026-09-28T14:46:27+08:00
command=HIP_VISIBLE_DEVICES=0 PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_hip_module_launch.py --iters 1000 --batches 9
hsaco_sha256=d0bc01bcbace10aa262a50fc2e16c6fd2cd9a2bedb394cbf4fc38c179dee8e46 kernel=empty_kernel threads=256 shared=0 callback_count=1
iters=1000 batches=9 unit=us/call (host enqueue, no timed sync)
TVM FFI HIP HSACO         median=3.2310 samples=[3.5743400000000003, 3.279505, 3.274181, 3.299915, 3.218769, 3.2281869999999997, 3.230992, 3.222433, 3.222175]
Triton same HSACO         median=15.7260 samples=[15.949459000000001, 15.729885, 15.787135000000001, 15.73233, 15.726036, 15.649072, 15.667834999999998, 15.699326999999998, 15.690974]
INTJ same function        median=2.9893 samples=[2.9841509999999998, 2.980313, 2.989331, 2.9824740000000003, 2.9893330000000002, 3.00488, 2.968701, 3.0500920000000002, 2.990681]
elapsed_ns=3739006165
exit_status=0
ended=2026-09-28T14:46:30+08:00
```

```
$ cat /tmp/intj-bench/globals-tensor-rerun/round2_launch_gpu.txt
commit=c7125b1dcc97f25b7021ba02183510da8ceecfe7
started=2026-09-28T14:46:15+08:00
command=HIP_VISIBLE_DEVICES=0 PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --iters 1000 --batches 9
mode=gpu; 1000 calls × 9 batches; median ns/call
               path    ns/call       Δ ns       Δ %   build ms   bind ms
           auto map     3096.1       +0.0      +0.0      80.86         -
   tensor annotated     3097.6       +1.5      +0.0       2.95         -
            generic     2982.0     -114.1      -3.7       2.53         -
 generic tensor ann     2969.0     -127.1      -4.1       2.52         -
        reduced key     2878.3     -217.8      -7.0       2.67         -
         verify off     2944.2     -151.9      -4.9       2.70         -
          verify on     2956.1     -140.0      -4.5       2.75         -
              baked     2868.4     -227.7      -7.4       2.74         -
       bound tensor     2882.7     -213.4      -6.9       2.56      0.12
      bound pointer     2887.8     -208.3      -6.7       2.52      0.12
   fixed device map     2899.9     -196.3      -6.3       2.46      0.14
fixed device no-map     2844.7     -251.5      -8.1       2.50      0.12
elapsed_ns=3788233708
exit_status=0
ended=2026-09-28T14:46:19+08:00
```

### `/tmp/intj-bench/globals-tensor-ab`

```
$ cat /tmp/intj-bench/globals-tensor-ab/cand_round0.txt
commit=c7125b1dcc97f25b7021ba02183510da8ceecfe7
started=2026-09-28T14:47:09+08:00
command=cd /mnt/nvme2/jinpli/workspace/home/jinpli/development/intj; HIP_VISIBLE_DEVICES=0 PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_ffi_compare.py --iters 1000 --batches 9
iters=1000 batches=9 unit=us/call (host-only, no timed sync)
FFI packed nop             median=0.1449 samples=[0.150734, 0.14516800000000002, 0.1449, 0.144455, 0.145348, 0.145, 0.14368, 0.143928, 0.143487]
FFI typed nop              median=0.1497 samples=[0.14749, 0.149726, 0.149594, 0.15043700000000002, 0.151875, 0.15174, 0.14921700000000002, 0.150251, 0.14669]
FFI empty kernel           median=3.3178 samples=[3.567113, 3.4660100000000003, 3.343584, 3.189565, 3.13525, 3.317805, 3.1832629999999997, 3.318378, 3.1766170000000002]
INTJ empty kernel          median=2.9648 samples=[3.061149, 3.026623, 3.190517, 2.900938, 2.930067, 2.94892, 3.218713, 2.963342, 2.964775]
INTJ fixed-device kernel   median=2.8252 samples=[2.9363560000000004, 3.002034, 3.090169, 2.9519450000000003, 2.773088, 2.809434, 2.803317, 2.825179, 2.81719]
FFI packed nop mixed       median=0.1733 samples=[0.183872, 0.17198500000000003, 0.172387, 0.17873599999999998, 0.173341, 0.174042, 0.17201, 0.17387899999999998, 0.172547]
FFI typed nop mixed        median=0.1756 samples=[0.175447, 0.178001, 0.17481, 0.179261, 0.17555, 0.17726, 0.174407, 0.17517500000000003, 0.18109]
FFI mixed kernel           median=3.3142 samples=[3.421635, 3.520889, 3.261415, 3.314243, 3.190183, 3.3752600000000004, 3.2620549999999997, 3.375956, 3.262076]
INTJ mixed kernel          median=2.9461 samples=[3.163168, 3.053512, 2.994903, 2.946121, 2.792043, 2.83678, 2.851032, 2.970709, 2.8510790000000004]
INTJ fixed mixed kernel    median=2.8965 samples=[3.0520300000000002, 3.041632, 2.981095, 2.96941, 2.828228, 2.815883, 2.867068, 2.8965349999999996, 2.837325]
exit_status=0
ended=2026-09-28T14:47:13+08:00
```

```
$ cat /tmp/intj-bench/globals-tensor-ab/cand_round1.txt
commit=c7125b1dcc97f25b7021ba02183510da8ceecfe7
started=2026-09-28T14:47:16+08:00
command=cd /mnt/nvme2/jinpli/workspace/home/jinpli/development/intj; HIP_VISIBLE_DEVICES=0 PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_ffi_compare.py --iters 1000 --batches 9
iters=1000 batches=9 unit=us/call (host-only, no timed sync)
FFI packed nop             median=0.1446 samples=[0.151939, 0.14479, 0.144184, 0.143931, 0.144988, 0.143853, 0.14477, 0.144566, 0.143876]
FFI typed nop              median=0.1496 samples=[0.148532, 0.15109899999999998, 0.147422, 0.149625, 0.149598, 0.152055, 0.14819900000000003, 0.151945, 0.146583]
FFI empty kernel           median=3.7573 samples=[3.618737, 3.958127, 3.840717, 3.7572959999999997, 3.669729, 4.2293829999999994, 3.554417, 4.315711, 3.5449050000000004]
INTJ empty kernel          median=3.6388 samples=[3.1304450000000004, 3.7698840000000002, 3.7613359999999996, 3.55949, 3.522824, 3.5984540000000003, 4.015625, 3.638816, 3.983526]
INTJ fixed-device kernel   median=3.4969 samples=[3.02076, 3.716088, 3.792112, 3.496904, 3.596527, 3.172725, 3.400912, 3.207339, 3.594271]
FFI packed nop mixed       median=0.1732 samples=[0.173249, 0.174766, 0.170594, 0.173679, 0.170285, 0.17236500000000002, 0.17074899999999998, 0.17447, 0.174227]
FFI typed nop mixed        median=0.1744 samples=[0.174261, 0.174846, 0.173797, 0.17746199999999998, 0.172571, 0.17702299999999999, 0.172793, 0.174416, 0.17572300000000002]
FFI mixed kernel           median=3.7330 samples=[3.7074830000000003, 3.963839, 3.733041, 3.838758, 3.6808490000000003, 3.868684, 3.656809, 4.358796999999999, 3.682766]
INTJ mixed kernel          median=3.5908 samples=[4.111891, 3.766417, 3.679456, 3.59526, 3.5806649999999998, 3.5907579999999997, 3.446287, 3.574346, 3.5528299999999997]
INTJ fixed mixed kernel    median=3.5909 samples=[3.790226, 3.786464, 3.641006, 3.6685239999999997, 3.549875, 3.590835, 3.528615, 3.5579769999999997, 3.590868]
exit_status=0
ended=2026-09-28T14:47:20+08:00
```

```
$ cat /tmp/intj-bench/globals-tensor-ab/cand_round2.txt
commit=c7125b1dcc97f25b7021ba02183510da8ceecfe7
started=2026-09-28T14:47:23+08:00
command=cd /mnt/nvme2/jinpli/workspace/home/jinpli/development/intj; HIP_VISIBLE_DEVICES=0 PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_ffi_compare.py --iters 1000 --batches 9
iters=1000 batches=9 unit=us/call (host-only, no timed sync)
FFI packed nop             median=0.1440 samples=[0.152302, 0.14576499999999998, 0.143735, 0.143739, 0.14493799999999998, 0.143327, 0.144424, 0.143038, 0.143953]
FFI typed nop              median=0.1501 samples=[0.147012, 0.145848, 0.14740199999999998, 0.150111, 0.154733, 0.15238300000000002, 0.150308, 0.15013200000000002, 0.151218]
FFI empty kernel           median=3.3561 samples=[3.5775970000000004, 3.572383, 3.447652, 3.350061, 3.150951, 3.359371, 3.197529, 3.356086, 3.195806]
INTJ empty kernel          median=2.9967 samples=[3.059941, 3.067761, 2.998194, 2.880381, 2.986225, 2.955239, 3.015208, 2.964661, 2.996731]
INTJ fixed-device kernel   median=2.9092 samples=[2.90919, 2.993308, 3.0260569999999998, 2.831603, 2.880274, 2.807759, 2.9257, 2.820687, 2.918679]
FFI packed nop mixed       median=0.1749 samples=[0.172607, 0.175155, 0.173907, 0.178904, 0.172883, 0.17521, 0.174921, 0.17657, 0.17161400000000002]
FFI typed nop mixed        median=0.1772 samples=[0.177892, 0.17571, 0.17716900000000002, 0.179114, 0.178894, 0.17628899999999997, 0.177145, 0.17765899999999998, 0.172922]
FFI mixed kernel           median=3.2674 samples=[3.368253, 3.432515, 3.248855, 3.37608, 3.220853, 3.26743, 3.261948, 3.388785, 3.245853]
INTJ mixed kernel          median=2.8919 samples=[3.187303, 3.114491, 2.876035, 2.910101, 2.891474, 2.891921, 2.863963, 3.010893, 2.857396]
INTJ fixed mixed kernel    median=2.9208 samples=[3.123475, 3.1082379999999996, 2.936159, 2.932522, 2.890707, 2.865313, 2.9107779999999996, 2.920823, 2.893335]
exit_status=0
ended=2026-09-28T14:47:27+08:00
```

```
$ cat /tmp/intj-bench/globals-tensor-ab/cand_round3.txt
commit=c7125b1dcc97f25b7021ba02183510da8ceecfe7
started=2026-09-28T14:47:31+08:00
command=cd /mnt/nvme2/jinpli/workspace/home/jinpli/development/intj; HIP_VISIBLE_DEVICES=0 PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_ffi_compare.py --iters 1000 --batches 9
iters=1000 batches=9 unit=us/call (host-only, no timed sync)
FFI packed nop             median=0.1476 samples=[0.15162299999999998, 0.148562, 0.153201, 0.14703200000000002, 0.147572, 0.144161, 0.14697300000000002, 0.142558, 0.14765]
FFI typed nop              median=0.1487 samples=[0.15072300000000002, 0.148485, 0.146054, 0.15243, 0.147315, 0.150627, 0.14676599999999998, 0.149595, 0.14867599999999997]
FFI empty kernel           median=3.3186 samples=[3.5559499999999997, 3.4863899999999997, 3.447, 3.243199, 3.254286, 3.318576, 3.221762, 3.322348, 3.202819]
INTJ empty kernel          median=2.9895 samples=[3.0604270000000002, 3.066215, 3.07646, 2.889062, 2.8529340000000003, 2.986795, 3.001871, 2.989493, 2.965639]
INTJ fixed-device kernel   median=2.9025 samples=[2.9043319999999997, 3.0207539999999997, 3.048442, 2.902451, 2.8461619999999996, 2.8312869999999997, 2.806211, 2.818976, 2.925042]
FFI packed nop mixed       median=0.1728 samples=[0.17389, 0.170972, 0.17233199999999999, 0.171567, 0.17283500000000002, 0.17246, 0.173161, 0.17339, 0.173382]
FFI typed nop mixed        median=0.1774 samples=[0.17736600000000002, 0.17754499999999998, 0.176201, 0.181811, 0.172516, 0.181344, 0.176041, 0.176785, 0.180368]
FFI mixed kernel           median=3.3117 samples=[3.405475, 3.5014969999999996, 3.31169, 3.2920439999999997, 3.216894, 3.3171239999999997, 3.290347, 3.3500210000000004, 3.270933]
INTJ mixed kernel          median=2.9554 samples=[3.163233, 3.104734, 2.939977, 2.931172, 2.95536, 2.8887359999999997, 2.868406, 3.021138, 2.975531]
INTJ fixed mixed kernel    median=2.9387 samples=[3.165049, 3.038024, 2.9391260000000003, 2.938709, 2.878168, 2.858812, 2.9106509999999997, 2.944864, 2.8988090000000004]
exit_status=0
ended=2026-09-28T14:47:34+08:00
```

```
$ cat /tmp/intj-bench/globals-tensor-ab/cand_round4.txt
commit=c7125b1dcc97f25b7021ba02183510da8ceecfe7
started=2026-09-28T14:47:38+08:00
command=cd /mnt/nvme2/jinpli/workspace/home/jinpli/development/intj; HIP_VISIBLE_DEVICES=0 PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_ffi_compare.py --iters 1000 --batches 9
iters=1000 batches=9 unit=us/call (host-only, no timed sync)
FFI packed nop             median=0.1435 samples=[0.150625, 0.14212899999999998, 0.142057, 0.142739, 0.144248, 0.14551, 0.142941, 0.143731, 0.143455]
FFI typed nop              median=0.1506 samples=[0.156508, 0.15299600000000002, 0.147425, 0.150597, 0.147673, 0.154122, 0.14933600000000002, 0.153118, 0.14727]
FFI empty kernel           median=3.3146 samples=[3.539345, 3.398634, 3.395381, 3.338362, 3.29208, 3.314622, 3.192666, 3.310143, 3.18514]
INTJ empty kernel          median=2.9836 samples=[3.089179, 3.1196729999999997, 3.098522, 2.882022, 2.83858, 2.947925, 3.011828, 2.89914, 2.983565]
INTJ fixed-device kernel   median=2.9102 samples=[2.910214, 3.094309, 3.080154, 2.803576, 2.847268, 2.829252, 2.928813, 2.8429740000000003, 2.912916]
FFI packed nop mixed       median=0.1734 samples=[0.175843, 0.172504, 0.174549, 0.17342, 0.174171, 0.172442, 0.172894, 0.172476, 0.174527]
FFI typed nop mixed        median=0.1765 samples=[0.175338, 0.17873599999999998, 0.176511, 0.179335, 0.174784, 0.178414, 0.173671, 0.177762, 0.173701]
FFI mixed kernel           median=3.2806 samples=[3.382096, 3.4524630000000003, 3.2684960000000003, 3.2805790000000004, 3.2333600000000002, 3.303856, 3.257786, 3.3304099999999996, 3.238631]
INTJ mixed kernel          median=2.9603 samples=[3.1603809999999997, 3.157673, 2.960298, 2.9750189999999996, 2.861446, 2.873484, 2.85876, 3.000095, 2.8631320000000002]
INTJ fixed mixed kernel    median=2.9107 samples=[3.1367689999999997, 3.141129, 2.983567, 2.962177, 2.8553870000000003, 2.8375030000000003, 2.875816, 2.9107060000000002, 2.844348]
exit_status=0
ended=2026-09-28T14:47:42+08:00
```

```
$ cat /tmp/intj-bench/globals-tensor-ab/parent_round0.txt
commit=9075eaa9bf48e91d94ebb0cfd8a54584fdb071f8
started=2026-09-28T14:46:57+08:00
command=cd /tmp/globals-tensor-parent; HIP_VISIBLE_DEVICES=0 PYTHONPATH=/tmp/globals-tensor-parent INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_ffi_compare.py --iters 1000 --batches 9
iters=1000 batches=9 unit=us/call (host-only, no timed sync)
FFI packed nop             median=0.1451 samples=[0.150901, 0.14485, 0.143157, 0.145656, 0.143644, 0.14427, 0.146804, 0.14513399999999999, 0.145627]
FFI typed nop              median=0.1492 samples=[0.149552, 0.153634, 0.14741100000000001, 0.147123, 0.14850899999999997, 0.150346, 0.147687, 0.152816, 0.14918700000000001]
FFI empty kernel           median=3.2894 samples=[3.728836, 3.4520329999999997, 3.443426, 3.298724, 3.278458, 3.286982, 3.2894259999999997, 3.27569, 3.2889459999999997]
INTJ empty kernel          median=2.9137 samples=[3.2096970000000002, 3.05763, 3.067589, 2.9060520000000003, 2.866703, 2.913662, 2.913873, 2.9112579999999997, 2.891844]
INTJ fixed-device kernel   median=2.8936 samples=[3.2811109999999997, 3.0771990000000002, 3.0776790000000003, 2.910987, 2.866378, 2.8671219999999997, 2.886971, 2.8721819999999996, 2.893602]
FFI packed nop mixed       median=0.1735 samples=[0.173917, 0.17280600000000002, 0.17345500000000003, 0.177081, 0.17343799999999998, 0.173507, 0.17247800000000002, 0.174057, 0.175049]
FFI typed nop mixed        median=0.1778 samples=[0.180364, 0.17768199999999998, 0.180087, 0.17777500000000002, 0.175064, 0.181979, 0.17760599999999999, 0.178705, 0.174761]
FFI mixed kernel           median=3.3143 samples=[3.592249, 3.476465, 3.470743, 3.311052, 3.263472, 3.314322, 3.312851, 3.3028380000000004, 3.343112]
INTJ mixed kernel          median=2.9533 samples=[3.1143539999999996, 3.12336, 3.0840639999999997, 2.948095, 2.892326, 2.893444, 2.939652, 2.9729270000000003, 2.9532849999999997]
INTJ fixed mixed kernel    median=2.8894 samples=[3.17175, 3.114219, 2.940969, 2.949601, 2.8893969999999998, 2.860898, 2.857664, 2.882695, 2.8572469999999996]
exit_status=0
ended=2026-09-28T14:47:09+08:00
```

```
$ cat /tmp/intj-bench/globals-tensor-ab/parent_round1.txt
commit=9075eaa9bf48e91d94ebb0cfd8a54584fdb071f8
started=2026-09-28T14:47:13+08:00
command=cd /tmp/globals-tensor-parent; HIP_VISIBLE_DEVICES=0 PYTHONPATH=/tmp/globals-tensor-parent INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_ffi_compare.py --iters 1000 --batches 9
iters=1000 batches=9 unit=us/call (host-only, no timed sync)
FFI packed nop             median=0.1442 samples=[0.151475, 0.143503, 0.143708, 0.143398, 0.14416800000000002, 0.1436, 0.144703, 0.148665, 0.144233]
FFI typed nop              median=0.1493 samples=[0.150075, 0.14927600000000002, 0.14502500000000002, 0.148567, 0.14808600000000002, 0.15514, 0.151713, 0.15192699999999998, 0.149263]
FFI empty kernel           median=3.3371 samples=[3.559046, 3.404097, 3.393742, 3.191005, 3.1977399999999996, 3.3414650000000004, 3.191645, 3.337146, 3.182703]
INTJ empty kernel          median=2.9684 samples=[3.058084, 3.1007089999999997, 3.102915, 2.9084090000000002, 2.871242, 2.967549, 2.9684160000000004, 2.91602, 2.974401]
INTJ fixed-device kernel   median=2.8533 samples=[2.899629, 3.0900390000000004, 3.0745189999999996, 2.910393, 2.8533090000000003, 2.790641, 2.798609, 2.793383, 2.783524]
FFI packed nop mixed       median=0.1734 samples=[0.17343199999999998, 0.17391900000000002, 0.177833, 0.173289, 0.173461, 0.174775, 0.172437, 0.171518, 0.17343199999999998]
FFI typed nop mixed        median=0.1769 samples=[0.17556899999999998, 0.18186000000000002, 0.174361, 0.177308, 0.176393, 0.17862899999999998, 0.176869, 0.17981999999999998, 0.174215]
FFI mixed kernel           median=3.2678 samples=[3.397486, 3.4381239999999997, 3.266978, 3.275688, 3.2069259999999997, 3.267798, 3.249451, 3.376697, 3.2597530000000003]
INTJ mixed kernel          median=2.9590 samples=[3.169001, 3.126908, 2.958989, 2.9661999999999997, 2.8882269999999997, 2.885401, 2.831064, 2.998337, 2.835109]
INTJ fixed mixed kernel    median=2.9157 samples=[3.121433, 3.1277579999999996, 2.9809259999999997, 2.954614, 2.8732330000000004, 2.8653739999999996, 2.902222, 2.915703, 2.889806]
exit_status=0
ended=2026-09-28T14:47:16+08:00
```

```
$ cat /tmp/intj-bench/globals-tensor-ab/parent_round2.txt
commit=9075eaa9bf48e91d94ebb0cfd8a54584fdb071f8
started=2026-09-28T14:47:20+08:00
command=cd /tmp/globals-tensor-parent; HIP_VISIBLE_DEVICES=0 PYTHONPATH=/tmp/globals-tensor-parent INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_ffi_compare.py --iters 1000 --batches 9
iters=1000 batches=9 unit=us/call (host-only, no timed sync)
FFI packed nop             median=0.1445 samples=[0.15181899999999998, 0.144322, 0.14587, 0.144403, 0.144361, 0.144457, 0.142211, 0.14532, 0.144894]
FFI typed nop              median=0.1495 samples=[0.148416, 0.149503, 0.154007, 0.150427, 0.148202, 0.152339, 0.143706, 0.150771, 0.147835]
FFI empty kernel           median=3.3421 samples=[3.597924, 3.506606, 3.463156, 3.327634, 3.122666, 3.342105, 3.167714, 3.543891, 3.175907]
INTJ empty kernel          median=2.9949 samples=[3.067981, 3.063311, 3.0013850000000004, 2.870077, 2.9264810000000003, 2.88579, 2.994876, 3.1403339999999997, 2.965233]
INTJ fixed-device kernel   median=2.8971 samples=[2.916864, 3.004278, 3.053391, 2.831826, 2.768878, 2.797755, 2.897053, 2.908525, 2.804379]
FFI packed nop mixed       median=0.1748 samples=[0.17503200000000002, 0.173215, 0.17541900000000002, 0.172902, 0.174177, 0.17628899999999997, 0.176845, 0.174833, 0.174734]
FFI typed nop mixed        median=0.1781 samples=[0.176664, 0.178064, 0.176181, 0.17866200000000002, 0.181763, 0.18721700000000002, 0.177238, 0.18193399999999998, 0.17521199999999998]
FFI mixed kernel           median=3.3417 samples=[3.3873539999999998, 3.422709, 3.23743, 3.361603, 3.173949, 3.5971260000000003, 3.256098, 3.341669, 3.239351]
INTJ mixed kernel          median=2.9115 samples=[3.1756770000000003, 3.118509, 2.848865, 2.929658, 2.795105, 2.823228, 2.9445970000000004, 2.911541, 2.846888]
INTJ fixed mixed kernel    median=2.8341 samples=[3.1193310000000003, 3.110078, 2.949897, 2.93794, 2.826062, 2.809002, 2.826039, 2.834082, 2.8297109999999996]
exit_status=0
ended=2026-09-28T14:47:23+08:00
```

```
$ cat /tmp/intj-bench/globals-tensor-ab/parent_round3.txt
commit=9075eaa9bf48e91d94ebb0cfd8a54584fdb071f8
started=2026-09-28T14:47:27+08:00
command=cd /tmp/globals-tensor-parent; HIP_VISIBLE_DEVICES=0 PYTHONPATH=/tmp/globals-tensor-parent INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_ffi_compare.py --iters 1000 --batches 9
iters=1000 batches=9 unit=us/call (host-only, no timed sync)
FFI packed nop             median=0.1454 samples=[0.15076499999999998, 0.14415, 0.145397, 0.145571, 0.146127, 0.145198, 0.144617, 0.147589, 0.14368799999999998]
FFI typed nop              median=0.1493 samples=[0.147371, 0.150985, 0.149794, 0.14855500000000002, 0.148816, 0.155451, 0.147155, 0.15078, 0.149268]
FFI empty kernel           median=3.3356 samples=[3.564331, 3.431536, 3.37568, 3.188671, 3.197238, 3.335576, 3.174241, 3.3455500000000002, 3.16588]
INTJ empty kernel          median=3.0102 samples=[3.075688, 3.079801, 3.080493, 2.904781, 2.862139, 2.878826, 3.010216, 2.938016, 3.01115]
INTJ fixed-device kernel   median=2.8852 samples=[2.919465, 3.071866, 3.059153, 2.916703, 2.863879, 2.799986, 2.8851590000000003, 2.799432, 2.8797669999999997]
FFI packed nop mixed       median=0.1744 samples=[0.17294800000000002, 0.17186, 0.178161, 0.17497300000000002, 0.173152, 0.169654, 0.174607, 0.174653, 0.174428]
FFI typed nop mixed        median=0.1768 samples=[0.17532499999999998, 0.178121, 0.176767, 0.17747, 0.176589, 0.17897, 0.17320500000000003, 0.177328, 0.176485]
FFI mixed kernel           median=3.3394 samples=[3.602204, 3.43081, 3.3393710000000003, 3.2785219999999997, 3.208935, 3.3707, 3.241619, 3.362337, 3.238448]
INTJ mixed kernel          median=2.9675 samples=[3.101526, 3.1552249999999997, 2.9674899999999997, 2.978382, 2.7904720000000003, 2.8215250000000003, 2.8336770000000002, 2.9769340000000004, 2.830625]
INTJ fixed mixed kernel    median=2.8924 samples=[3.120805, 3.105192, 2.966666, 2.954957, 2.8305599999999997, 2.79391, 2.872163, 2.8924369999999997, 2.881666]
exit_status=0
ended=2026-09-28T14:47:31+08:00
```

```
$ cat /tmp/intj-bench/globals-tensor-ab/parent_round4.txt
commit=9075eaa9bf48e91d94ebb0cfd8a54584fdb071f8
started=2026-09-28T14:47:34+08:00
command=cd /tmp/globals-tensor-parent; HIP_VISIBLE_DEVICES=0 PYTHONPATH=/tmp/globals-tensor-parent INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_ffi_compare.py --iters 1000 --batches 9
iters=1000 batches=9 unit=us/call (host-only, no timed sync)
FFI packed nop             median=0.1449 samples=[0.15146500000000002, 0.145012, 0.14562799999999998, 0.145375, 0.14495, 0.144779, 0.143427, 0.144202, 0.144373]
FFI typed nop              median=0.1485 samples=[0.148514, 0.14927600000000002, 0.149567, 0.153545, 0.14800899999999997, 0.14944300000000002, 0.14707900000000002, 0.14821, 0.147552]
FFI empty kernel           median=3.3405 samples=[3.608609, 3.3870270000000002, 3.3752, 3.313181, 3.204473, 3.340463, 3.180586, 3.361263, 3.200995]
INTJ empty kernel          median=3.0037 samples=[3.071781, 3.0680650000000003, 3.066358, 2.830602, 2.744227, 2.9291129999999996, 3.003664, 2.9659250000000004, 3.027473]
INTJ fixed-device kernel   median=2.8201 samples=[2.916208, 3.070784, 3.063038, 2.815401, 2.80579, 2.782787, 2.879225, 2.8090279999999996, 2.820093]
FFI packed nop mixed       median=0.1734 samples=[0.174011, 0.174507, 0.172759, 0.170519, 0.17188900000000001, 0.173039, 0.17510900000000001, 0.173427, 0.173609]
FFI typed nop mixed        median=0.1764 samples=[0.175883, 0.179477, 0.17976, 0.176382, 0.176435, 0.175392, 0.177175, 0.17588900000000002, 0.176519]
FFI mixed kernel           median=3.2892 samples=[3.3569609999999996, 3.4192519999999997, 3.289223, 3.265679, 3.165466, 3.379585, 3.2465450000000002, 3.3672649999999997, 3.272913]
INTJ mixed kernel          median=2.9208 samples=[3.15402, 3.128569, 2.9231480000000003, 2.9394430000000003, 2.781109, 2.824325, 2.830553, 2.920826, 2.880777]
INTJ fixed mixed kernel    median=2.9232 samples=[3.145839, 3.1292199999999997, 2.9789540000000003, 2.9746889999999997, 2.840364, 2.837442, 2.8966689999999997, 2.923203, 2.908668]
exit_status=0
ended=2026-09-28T14:47:38+08:00
```


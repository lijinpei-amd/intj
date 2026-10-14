# Full benchmark suite at `develop` HEAD (index 0)

Candidate `bc8caf7` ("TODO: pin down which pythons triton leaves untested"), 51 commits after the last full
suite (`76bef91`, `2026-09-29_full-suite_0_76bef91.md`). Ran the same 12-case gate plus `bench_mix.py`, 3 rounds,
in the project's uv venv instead of `/tmp/gb2`. The GPU rows read slower than on 2026-09-29; the same-day A/B in
[`2026-10-14_gpu-drift-ab_0_bc8caf7.md`](2026-10-14_gpu-drift-ab_0_bc8caf7.md) shows `76bef91` slows by the same
amount, so the shift is the machine, not these commits.

## Environment

- Source: `bc8caf7` (branch `develop`), clean checkout at `/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj`.
- CPU: Intel Xeon Platinum 8480C, `taskset -c 0`, one process at a time. Load average 0.84, 0.99, 0.78 before.
- GPU: AMD Instinct MI308X (`gfx942`), GPU 0 (`HIP_VISIBLE_DEVICES=0`), all 8 GPUs 0 % busy before (`rocm-smi`).
- Python 3.12.3 (`workspace/intj/venv`), Torch `2.14.0.dev20260804+rocm7.1`, Triton `3.8.0`, apache-tvm-ffi
  `0.1.14.post1` (PyPI), cc 13.3.0. Not the `/tmp/gb2` environment of earlier records (Torch `2.14.0+rocm7.2`,
  tvm-ffi dev build).
- `INTJ_BENCHMARK_ROOT=/mnt/nvme2/jinpli/workspace/home/jinpli/development/benchmark`: google/benchmark `073a74f`,
  `cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -DBENCHMARK_ENABLE_TESTING=OFF -DBENCHMARK_ENABLE_GTEST_TESTS=OFF`.
- Default `TRITON_HOME` (warm module cache; builds are outside the timers).
- 2026-10-14 16:26:48 to 16:34:42 +08:00. CUDA: untested.

## Commands

`/tmp/intj-bench/run_all_venv.sh rerun-venv-bc8caf7`: `run_all.sh` from the 2026-09-29 record with the venv's
python, the `INTJ_BENCHMARK_ROOT` above, and one extra case, `mix|benchmarks/bench_mix.py` (default 20000 iters x
9 batches). Every case: `PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=... taskset -c 0 <venv>/bin/python <args>`, 3 rounds.

```
launch_gpu      benchmarks/bench_launch.py --iters 1000 --batches 9
launch_host     benchmarks/bench_launch.py --no-gpu --iters 100000 --batches 9
launch_readme   benchmarks/bench_launch.py --readme --iters 1000 --batches 9
launch_sweep    benchmarks/bench_launch.py --sweep --iters 100000 --batches 9
launch_last_key benchmarks/bench_launch.py --last-key --iters 100000 --batches 9
ffi_compare     benchmarks/bench_ffi_compare.py --iters 1000 --batches 9
ffi_sweep       benchmarks/bench_ffi_compare.py --sweep --iters 1000 --batches 9
ffi_paths       benchmarks/bench_intj_ffi_paths.py --mode all --iters 1000 --batches 9
hip_module      benchmarks/bench_hip_module_launch.py --iters 1000 --batches 9
kernel_cache    -m pytest tests/test_kernel_cache.py -s -q
launch_tuned    benchmarks/bench_launch.py --tuned --iters 20000 --batches 9
launch_dynamic  benchmarks/bench_launch.py --dynamic --iters 20000 --batches 9
mix             benchmarks/bench_mix.py
```

```bash
python /tmp/intj-bench/bench_compare.py full-suite rerun-venv-bc8caf7    # vs 76bef91 (2026-09-29)
python /tmp/intj-bench/mix_compare.py rerun-venv-bc8caf7                 # bench_mix vs 76bef91's 3 rounds
```

All 39 files have `exit_status=0`.

## Result

**No regression attributable to intj.** Rule: slower by >3 % and by >2 ns (ns rows) or >0.1 us (us rows).
Medians of the 3 process medians against `76bef91`:

| case | rows | Δ % range | flagged |
|---|---:|---|---:|
| `launch_host` | 12 | -3.7 .. -0.2 | 0 |
| `launch_sweep` | 6 | -1.6 .. +0.4 | 0 |
| `launch_last_key` | 6 | -1.5 .. +0.0 | 0 |
| `launch_tuned` | 2 | +1.2 .. +1.6 | 0 |
| `kernel_cache` | 90 | -27.9 .. +3.1 | 0 |
| `hip_module` | 3 | -1.1 .. +3.0 | 0 |
| `ffi_compare` | 10 | -1.5 .. +2.6 | 0 |
| `launch_readme` | 7 | -3.5 .. +5.8 | 1 |
| `ffi_paths` | 17 | -3.8 .. +5.0 | 5 |
| `ffi_sweep` | 21 | -3.7 .. +6.3 | 4 |
| `launch_gpu` | 12 | +3.6 .. +8.5 | 12 |
| `launch_dynamic` | 8 | -7.7 .. +24.7 | 4 |
| `bench_mix.py` | 144 | -5.9 .. +4.4 (median -0.3) | 4 |

- **`launch_gpu` +4-9 %, `launch_readme` intj +5.8 %:** GPU enqueue rows only; the host-only cases are flat or
  faster. TVM FFI's own empty-kernel rows in `ffi_sweep` (not intj code) rose +3-6 % too. The A/B record measures
  `76bef91` at the same speed as `bc8caf7` today: a machine shift since 2026-09-29.
- **`launch_dynamic` launch +24 %:** bimodal across processes, round 0 at 3.10 us and rounds 1-2 at 3.80 us; the
  host-only `spec_key` column is unchanged (~92 ns). It read 3.0-3.1 us in all 6 A/B processes. Noise.
- **`ffi_paths` +3.5-5 % (adapter and FFI wrapper rows, +3.5-5.6 ns):** these are Python-side wrappers around the
  INTJ call. An unjournaled `/tmp/gb2` run at `bc8caf7` the same morning (`/tmp/intj-bench/full-suite-bc8caf7`)
  shows the same rows within -1.4 .. +1.7 % of this run, and itself flags the same 4 rows against `76bef91`.
- **`bench_mix.py`:** 4 flagged rows (+2.9-4.5 ns, 16-32 args) against 2 rows that moved -4.6/-4.8 ns at the same
  sizes. Single-row jitter.

## Raw output

### Gate comparison (`bench_compare.py full-suite rerun-venv-bc8caf7`)

```
ffi_compare      median           FFI empty kernel                 value    baseline=3.3657 new=3.3165 delta=-0.0492 pct=-1.5% (us)
ffi_compare      median           FFI mixed kernel                 value    baseline=3.3903 new=3.3570 delta=-0.0333 pct=-1.0% (us)
ffi_compare      median           FFI packed nop                   value    baseline=0.1459 new=0.1443 delta=-0.0016 pct=-1.1% (us)
ffi_compare      median           FFI packed nop mixed             value    baseline=0.1742 new=0.1717 delta=-0.0025 pct=-1.4% (us)
ffi_compare      median           FFI typed nop                    value    baseline=0.1496 new=0.1500 delta=0.0004 pct=+0.3% (us)
ffi_compare      median           FFI typed nop mixed              value    baseline=0.1779 new=0.1762 delta=-0.0017 pct=-1.0% (us)
ffi_compare      median           INTJ empty kernel                value    baseline=2.9834 new=2.9602 delta=-0.0232 pct=-0.8% (us)
ffi_compare      median           INTJ fixed mixed kernel          value    baseline=2.9159 new=2.9929 delta=0.0770 pct=+2.6% (us)
ffi_compare      median           INTJ fixed-device kernel         value    baseline=2.9304 new=2.9497 delta=0.0193 pct=+0.7% (us)
ffi_compare      median           INTJ mixed kernel                value    baseline=2.9907 new=2.9532 delta=-0.0375 pct=-1.3% (us)
ffi_paths        median_ns        FFI unpack Pair only             value    baseline=163.5 new=164.4 delta=0.9000 pct=+0.6% (ns)
ffi_paths        median_ns        INTJ 3-tensor host-only nop      value    baseline=32.4000 new=32.1000 delta=-0.3000 pct=-0.9% (ns)
ffi_paths        median_ns        INTJ FFI wrapper kwargs          value    baseline=123.3 new=128.9 delta=5.6000 pct=+4.5% (ns) REGRESSION
ffi_paths        median_ns        INTJ FFI wrapper positional      value    baseline=100.8 new=104.3 delta=3.5000 pct=+3.5% (ns) REGRESSION
ffi_paths        median_ns        INTJ adapter defaults            value    baseline=85.8000 new=89.7000 delta=3.9000 pct=+4.5% (ns) REGRESSION
ffi_paths        median_ns        INTJ adapter kwargs              value    baseline=103.4 new=108.6 delta=5.2000 pct=+5.0% (ns) REGRESSION
ffi_paths        median_ns        INTJ adapter positional          value    baseline=86.8000 new=90.6000 delta=3.8000 pct=+4.4% (ns) REGRESSION
ffi_paths        median_ns        INTJ cold compile callback + cache value    baseline=709.6 new=682.3 delta=-27.3000 pct=-3.8% (ns)
ffi_paths        median_ns        INTJ config FFI unpack           value    baseline=323.0 new=322.5 delta=-0.5000 pct=-0.2% (ns)
ffi_paths        median_ns        INTJ config direct               value    baseline=77.7000 new=77.4000 delta=-0.3000 pct=-0.4% (ns)
ffi_paths        median_ns        INTJ direct positional           value    baseline=63.0000 new=64.0000 delta=1.0000 pct=+1.6% (ns)
ffi_paths        median_ns        INTJ hot call (no callback)      value    baseline=35.3000 new=34.6000 delta=-0.7000 pct=-2.0% (ns)
ffi_paths        median_ns        INTJ pair FFI unpack             value    baseline=306.0 new=310.1 delta=4.1000 pct=+1.3% (ns)
ffi_paths        median_ns        INTJ pair direct                 value    baseline=65.4000 new=64.0000 delta=-1.4000 pct=-2.1% (ns)
ffi_paths        median_ns        INTJ pair manual unpack          value    baseline=161.8 new=160.8 delta=-1.0000 pct=-0.6% (ns)
ffi_paths        median_ns        INTJ pair prebuilt *tuple        value    baseline=137.0 new=134.6 delta=-2.4000 pct=-1.8% (ns)
ffi_paths        median_ns        INTJ pair stdlib astuple         value    baseline=1184.4 new=1139.5 delta=-44.9000 pct=-3.8% (ns)
ffi_sweep        median           args=0 FFI empty kernel          value    baseline=1.6957 new=1.6332 delta=-0.0625 pct=-3.7% (us)
ffi_sweep        median           args=0 FFI packed nop            value    baseline=0.1162 new=0.1151 delta=-0.0011 pct=-0.9% (us)
ffi_sweep        median           args=0 FFI typed nop             value    baseline=0.1200 new=0.1199 delta=-0.0001 pct=-0.1% (us)
ffi_sweep        median           args=16 FFI empty kernel         value    baseline=3.6448 new=3.7452 delta=0.1004 pct=+2.8% (us)
ffi_sweep        median           args=16 FFI packed nop           value    baseline=0.2980 new=0.3052 delta=0.0072 pct=+2.4% (us)
ffi_sweep        median           args=16 FFI typed nop            value    baseline=0.3055 new=0.3148 delta=0.0093 pct=+3.0% (us)
ffi_sweep        median           args=3 FFI empty kernel          value    baseline=3.2193 new=3.3309 delta=0.1116 pct=+3.5% (us) REGRESSION
ffi_sweep        median           args=3 FFI packed nop            value    baseline=0.1436 new=0.1444 delta=0.0008 pct=+0.6% (us)
ffi_sweep        median           args=3 FFI typed nop             value    baseline=0.1486 new=0.1507 delta=0.0021 pct=+1.4% (us)
ffi_sweep        median           args=32 FFI empty kernel         value    baseline=4.1137 new=4.2462 delta=0.1325 pct=+3.2% (us) REGRESSION
ffi_sweep        median           args=32 FFI packed nop           value    baseline=0.4811 new=0.4830 delta=0.0019 pct=+0.4% (us)
ffi_sweep        median           args=32 FFI typed nop            value    baseline=0.4990 new=0.4981 delta=-0.0009 pct=-0.2% (us)
ffi_sweep        median           args=5 FFI empty kernel          value    baseline=3.2626 new=3.3809 delta=0.1183 pct=+3.6% (us) REGRESSION
ffi_sweep        median           args=5 FFI packed nop            value    baseline=0.1735 new=0.1722 delta=-0.0013 pct=-0.7% (us)
ffi_sweep        median           args=5 FFI typed nop             value    baseline=0.1774 new=0.1784 delta=0.0010 pct=+0.6% (us)
ffi_sweep        median           args=64 FFI empty kernel         value    baseline=5.0646 new=5.3837 delta=0.3191 pct=+6.3% (us) REGRESSION
ffi_sweep        median           args=64 FFI packed nop           value    baseline=0.8615 new=0.8459 delta=-0.0156 pct=-1.8% (us)
ffi_sweep        median           args=64 FFI typed nop            value    baseline=0.8843 new=0.8780 delta=-0.0063 pct=-0.7% (us)
ffi_sweep        median           args=8 FFI empty kernel          value    baseline=3.4002 new=3.4467 delta=0.0465 pct=+1.4% (us)
ffi_sweep        median           args=8 FFI packed nop            value    baseline=0.2074 new=0.2052 delta=-0.0022 pct=-1.1% (us)
ffi_sweep        median           args=8 FFI typed nop             value    baseline=0.2100 new=0.2119 delta=0.0019 pct=+0.9% (us)
hip_module       median           INTJ same function               value    baseline=2.9236 new=3.0117 delta=0.0881 pct=+3.0% (us)
hip_module       median           TVM FFI HIP HSACO                value    baseline=3.1852 new=3.2197 delta=0.0345 pct=+1.1% (us)
hip_module       median           Triton same HSACO                value    baseline=15.7752 new=15.6022 delta=-0.1730 pct=-1.1% (us)
kernel_cache     1-word key       hash_only                        absl     baseline=1.3600 new=1.3600 delta=0.0000 pct=+0.0% (ns)
kernel_cache     1-word key       hash_only                        intj     baseline=1.3500 new=1.3600 delta=0.0100 pct=+0.7% (ns)
kernel_cache     1-word key       hash_only                        tsl      baseline=1.3600 new=1.3500 delta=-0.0100 pct=-0.7% (ns)
kernel_cache     1-word key       hit/1                            absl     baseline=1.4500 new=1.4600 delta=0.0100 pct=+0.7% (ns)
kernel_cache     1-word key       hit/1                            intj     baseline=1.9600 new=1.5700 delta=-0.3900 pct=-19.9% (ns)
kernel_cache     1-word key       hit/1                            tsl      baseline=2.0200 new=2.0300 delta=0.0100 pct=+0.5% (ns)
kernel_cache     1-word key       hit/512                          absl     baseline=5.3100 new=5.3100 delta=0.0000 pct=+0.0% (ns)
kernel_cache     1-word key       hit/512                          intj     baseline=2.3000 new=2.2900 delta=-0.0100 pct=-0.4% (ns)
kernel_cache     1-word key       hit/512                          tsl      baseline=2.9700 new=2.9100 delta=-0.0600 pct=-2.0% (ns)
kernel_cache     1-word key       hit/64                           absl     baseline=4.5400 new=4.5400 delta=0.0000 pct=+0.0% (ns)
kernel_cache     1-word key       hit/64                           intj     baseline=1.6300 new=1.6800 delta=0.0500 pct=+3.1% (ns)
kernel_cache     1-word key       hit/64                           tsl      baseline=2.1300 new=2.1400 delta=0.0100 pct=+0.5% (ns)
kernel_cache     1-word key       hit/8                            absl     baseline=4.3700 new=4.3600 delta=-0.0100 pct=-0.2% (ns)
kernel_cache     1-word key       hit/8                            intj     baseline=1.6900 new=1.5900 delta=-0.1000 pct=-5.9% (ns)
kernel_cache     1-word key       hit/8                            tsl      baseline=2.1000 new=2.1000 delta=0.0000 pct=+0.0% (ns)
kernel_cache     1-word key       hit_child                        absl     baseline=3.0000 new=3.0100 delta=0.0100 pct=+0.3% (ns)
kernel_cache     1-word key       hit_child                        intj     baseline=3.0000 new=3.0000 delta=0.0000 pct=+0.0% (ns)
kernel_cache     1-word key       hit_child                        tsl      baseline=3.0000 new=3.0000 delta=0.0000 pct=+0.0% (ns)
kernel_cache     1-word key       miss/1                           absl     baseline=1.3800 new=1.3800 delta=0.0000 pct=+0.0% (ns)
kernel_cache     1-word key       miss/1                           intj     baseline=1.2400 new=1.2700 delta=0.0300 pct=+2.4% (ns)
kernel_cache     1-word key       miss/1                           tsl      baseline=1.9100 new=1.8100 delta=-0.1000 pct=-5.2% (ns)
kernel_cache     1-word key       miss/512                         absl     baseline=4.0700 new=4.0700 delta=0.0000 pct=+0.0% (ns)
kernel_cache     1-word key       miss/512                         intj     baseline=2.9400 new=2.9200 delta=-0.0200 pct=-0.7% (ns)
kernel_cache     1-word key       miss/512                         tsl      baseline=2.0000 new=2.0000 delta=0.0000 pct=+0.0% (ns)
kernel_cache     1-word key       miss/64                          absl     baseline=3.5300 new=3.5200 delta=-0.0100 pct=-0.3% (ns)
kernel_cache     1-word key       miss/64                          intj     baseline=2.8000 new=2.7600 delta=-0.0400 pct=-1.4% (ns)
kernel_cache     1-word key       miss/64                          tsl      baseline=1.7400 new=1.7500 delta=0.0100 pct=+0.6% (ns)
kernel_cache     1-word key       miss/8                           absl     baseline=3.8700 new=3.2900 delta=-0.5800 pct=-15.0% (ns)
kernel_cache     1-word key       miss/8                           intj     baseline=2.7100 new=2.6800 delta=-0.0300 pct=-1.1% (ns)
kernel_cache     1-word key       miss/8                           tsl      baseline=1.7600 new=1.7500 delta=-0.0100 pct=-0.6% (ns)
kernel_cache     2-word key       hash_only                        absl     baseline=1.7100 new=1.4200 delta=-0.2900 pct=-17.0% (ns)
kernel_cache     2-word key       hash_only                        intj     baseline=1.4200 new=1.4200 delta=0.0000 pct=+0.0% (ns)
kernel_cache     2-word key       hash_only                        tsl      baseline=1.9700 new=1.4200 delta=-0.5500 pct=-27.9% (ns)
kernel_cache     2-word key       hit/1                            absl     baseline=2.0400 new=2.0000 delta=-0.0400 pct=-2.0% (ns)
kernel_cache     2-word key       hit/1                            intj     baseline=1.8400 new=1.8200 delta=-0.0200 pct=-1.1% (ns)
kernel_cache     2-word key       hit/1                            tsl      baseline=2.5700 new=2.5600 delta=-0.0100 pct=-0.4% (ns)
kernel_cache     2-word key       hit/512                          absl     baseline=6.3900 new=6.3800 delta=-0.0100 pct=-0.2% (ns)
kernel_cache     2-word key       hit/512                          intj     baseline=3.0200 new=3.0900 delta=0.0700 pct=+2.3% (ns)
kernel_cache     2-word key       hit/512                          tsl      baseline=4.0300 new=4.0300 delta=0.0000 pct=+0.0% (ns)
kernel_cache     2-word key       hit/64                           absl     baseline=5.2300 new=5.2200 delta=-0.0100 pct=-0.2% (ns)
kernel_cache     2-word key       hit/64                           intj     baseline=2.2900 new=2.2000 delta=-0.0900 pct=-3.9% (ns)
kernel_cache     2-word key       hit/64                           tsl      baseline=2.9200 new=2.9200 delta=0.0000 pct=+0.0% (ns)
kernel_cache     2-word key       hit/8                            absl     baseline=5.5200 new=5.0400 delta=-0.4800 pct=-8.7% (ns)
kernel_cache     2-word key       hit/8                            intj     baseline=1.9200 new=1.7900 delta=-0.1300 pct=-6.8% (ns)
kernel_cache     2-word key       hit/8                            tsl      baseline=2.5700 new=2.5600 delta=-0.0100 pct=-0.4% (ns)
kernel_cache     2-word key       hit_child                        absl     baseline=3.0100 new=3.0100 delta=0.0000 pct=+0.0% (ns)
kernel_cache     2-word key       hit_child                        intj     baseline=3.0000 new=3.0000 delta=0.0000 pct=+0.0% (ns)
kernel_cache     2-word key       hit_child                        tsl      baseline=3.0000 new=3.0000 delta=0.0000 pct=+0.0% (ns)
kernel_cache     2-word key       miss/1                           absl     baseline=1.4800 new=1.4600 delta=-0.0200 pct=-1.4% (ns)
kernel_cache     2-word key       miss/1                           intj     baseline=1.3800 new=1.3800 delta=0.0000 pct=+0.0% (ns)
kernel_cache     2-word key       miss/1                           tsl      baseline=2.3200 new=2.3200 delta=0.0000 pct=+0.0% (ns)
kernel_cache     2-word key       miss/512                         absl     baseline=4.0100 new=3.9800 delta=-0.0300 pct=-0.7% (ns)
kernel_cache     2-word key       miss/512                         intj     baseline=2.8800 new=2.9000 delta=0.0200 pct=+0.7% (ns)
kernel_cache     2-word key       miss/512                         tsl      baseline=2.7200 new=2.7400 delta=0.0200 pct=+0.7% (ns)
kernel_cache     2-word key       miss/64                          absl     baseline=3.9300 new=3.8900 delta=-0.0400 pct=-1.0% (ns)
kernel_cache     2-word key       miss/64                          intj     baseline=3.1200 new=3.1400 delta=0.0200 pct=+0.6% (ns)
kernel_cache     2-word key       miss/64                          tsl      baseline=2.1000 new=2.1000 delta=0.0000 pct=+0.0% (ns)
kernel_cache     2-word key       miss/8                           absl     baseline=3.2000 new=3.2000 delta=0.0000 pct=+0.0% (ns)
kernel_cache     2-word key       miss/8                           intj     baseline=2.4200 new=2.4100 delta=-0.0100 pct=-0.4% (ns)
kernel_cache     2-word key       miss/8                           tsl      baseline=1.8600 new=1.8600 delta=0.0000 pct=+0.0% (ns)
kernel_cache     5-word key       hash_only                        absl     baseline=3.5800 new=3.5800 delta=0.0000 pct=+0.0% (ns)
kernel_cache     5-word key       hash_only                        intj     baseline=3.5900 new=3.5800 delta=-0.0100 pct=-0.3% (ns)
kernel_cache     5-word key       hash_only                        tsl      baseline=3.5800 new=3.5800 delta=0.0000 pct=+0.0% (ns)
kernel_cache     5-word key       hit/1                            absl     baseline=2.4600 new=2.5000 delta=0.0400 pct=+1.6% (ns)
kernel_cache     5-word key       hit/1                            intj     baseline=3.0000 new=3.0000 delta=0.0000 pct=+0.0% (ns)
kernel_cache     5-word key       hit/1                            tsl      baseline=3.0600 new=3.0600 delta=0.0000 pct=+0.0% (ns)
kernel_cache     5-word key       hit/512                          absl     baseline=11.1600 new=11.2000 delta=0.0400 pct=+0.4% (ns)
kernel_cache     5-word key       hit/512                          intj     baseline=4.5400 new=4.5200 delta=-0.0200 pct=-0.4% (ns)
kernel_cache     5-word key       hit/512                          tsl      baseline=4.8000 new=4.7800 delta=-0.0200 pct=-0.4% (ns)
kernel_cache     5-word key       hit/64                           absl     baseline=9.7800 new=9.7700 delta=-0.0100 pct=-0.1% (ns)
kernel_cache     5-word key       hit/64                           intj     baseline=3.9700 new=3.2600 delta=-0.7100 pct=-17.9% (ns)
kernel_cache     5-word key       hit/64                           tsl      baseline=3.5100 new=3.5000 delta=-0.0100 pct=-0.3% (ns)
kernel_cache     5-word key       hit/8                            absl     baseline=9.4900 new=9.4800 delta=-0.0100 pct=-0.1% (ns)
kernel_cache     5-word key       hit/8                            intj     baseline=3.2100 new=3.2000 delta=-0.0100 pct=-0.3% (ns)
kernel_cache     5-word key       hit/8                            tsl      baseline=3.4100 new=3.4100 delta=0.0000 pct=+0.0% (ns)
kernel_cache     5-word key       hit_child                        absl     baseline=3.3800 new=3.0000 delta=-0.3800 pct=-11.2% (ns)
kernel_cache     5-word key       hit_child                        intj     baseline=3.0100 new=3.0100 delta=0.0000 pct=+0.0% (ns)
kernel_cache     5-word key       hit_child                        tsl      baseline=3.0100 new=3.0100 delta=0.0000 pct=+0.0% (ns)
kernel_cache     5-word key       miss/1                           absl     baseline=2.1600 new=1.9900 delta=-0.1700 pct=-7.9% (ns)
kernel_cache     5-word key       miss/1                           intj     baseline=1.4600 new=1.4000 delta=-0.0600 pct=-4.1% (ns)
kernel_cache     5-word key       miss/1                           tsl      baseline=2.1200 new=2.1000 delta=-0.0200 pct=-0.9% (ns)
kernel_cache     5-word key       miss/512                         absl     baseline=7.7300 new=7.7100 delta=-0.0200 pct=-0.3% (ns)
kernel_cache     5-word key       miss/512                         intj     baseline=3.5200 new=3.4600 delta=-0.0600 pct=-1.7% (ns)
kernel_cache     5-word key       miss/512                         tsl      baseline=2.5200 new=2.5000 delta=-0.0200 pct=-0.8% (ns)
kernel_cache     5-word key       miss/64                          absl     baseline=6.8300 new=6.7200 delta=-0.1100 pct=-1.6% (ns)
kernel_cache     5-word key       miss/64                          intj     baseline=2.8600 new=2.8400 delta=-0.0200 pct=-0.7% (ns)
kernel_cache     5-word key       miss/64                          tsl      baseline=1.9600 new=1.9700 delta=0.0100 pct=+0.5% (ns)
kernel_cache     5-word key       miss/8                           absl     baseline=6.9400 new=6.9300 delta=-0.0100 pct=-0.1% (ns)
kernel_cache     5-word key       miss/8                           intj     baseline=2.9900 new=2.9600 delta=-0.0300 pct=-1.0% (ns)
kernel_cache     5-word key       miss/8                           tsl      baseline=2.2200 new=2.2500 delta=0.0300 pct=+1.4% (ns)
launch_dynamic   dynamic          2 dynamic options                launch   baseline=3080.3 new=3803.2 delta=722.9 pct=+23.5% (ns) REGRESSION
launch_dynamic   dynamic          2 dynamic options                spec_key baseline=99.7000 new=92.0000 delta=-7.7000 pct=-7.7% (ns)
launch_dynamic   dynamic          int constexpr                    launch   baseline=3015.2 new=3733.1 delta=717.9 pct=+23.8% (ns) REGRESSION
launch_dynamic   dynamic          int constexpr                    spec_key baseline=90.8000 new=88.2000 delta=-2.6000 pct=-2.9% (ns)
launch_dynamic   dynamic          lazy plain                       launch   baseline=3060.6 new=3801.8 delta=741.2 pct=+24.2% (ns) REGRESSION
launch_dynamic   dynamic          lazy plain                       spec_key baseline=95.4000 new=93.0000 delta=-2.4000 pct=-2.5% (ns)
launch_dynamic   dynamic          str constexpr                    launch   baseline=3006.9 new=3750.3 delta=743.4 pct=+24.7% (ns) REGRESSION
launch_dynamic   dynamic          str constexpr                    spec_key baseline=89.3000 new=88.2000 delta=-1.1000 pct=-1.2% (ns)
launch_gpu       launch           auto map                         ns/call  baseline=3154.9 new=3268.4 delta=113.5 pct=+3.6% (ns) REGRESSION
launch_gpu       launch           baked                            ns/call  baseline=2882.2 new=3126.1 delta=243.9 pct=+8.5% (ns) REGRESSION
launch_gpu       launch           bound pointer                    ns/call  baseline=2887.0 new=3114.4 delta=227.4 pct=+7.9% (ns) REGRESSION
launch_gpu       launch           bound tensor                     ns/call  baseline=2978.5 new=3133.7 delta=155.2 pct=+5.2% (ns) REGRESSION
launch_gpu       launch           fixed device map                 ns/call  baseline=2902.8 new=3128.6 delta=225.8 pct=+7.8% (ns) REGRESSION
launch_gpu       launch           fixed device no-map              ns/call  baseline=2909.6 new=3129.6 delta=220.0 pct=+7.6% (ns) REGRESSION
launch_gpu       launch           generic                          ns/call  baseline=3012.5 new=3219.6 delta=207.1 pct=+6.9% (ns) REGRESSION
launch_gpu       launch           generic tensor ann               ns/call  baseline=2963.8 new=3178.1 delta=214.3 pct=+7.2% (ns) REGRESSION
launch_gpu       launch           reduced key                      ns/call  baseline=2972.5 new=3126.5 delta=154.0 pct=+5.2% (ns) REGRESSION
launch_gpu       launch           tensor annotated                 ns/call  baseline=3168.7 new=3329.0 delta=160.3 pct=+5.1% (ns) REGRESSION
launch_gpu       launch           verify off                       ns/call  baseline=2960.1 new=3136.4 delta=176.3 pct=+6.0% (ns) REGRESSION
launch_gpu       launch           verify on                        ns/call  baseline=2971.5 new=3122.2 delta=150.7 pct=+5.1% (ns) REGRESSION
launch_host      launch           auto map                         ns/call  baseline=41.0000 new=39.8000 delta=-1.2000 pct=-2.9% (ns)
launch_host      launch           baked                            ns/call  baseline=38.4000 new=37.5000 delta=-0.9000 pct=-2.3% (ns)
launch_host      launch           bound pointer                    ns/call  baseline=42.0000 new=41.2000 delta=-0.8000 pct=-1.9% (ns)
launch_host      launch           bound tensor                     ns/call  baseline=43.7000 new=42.9000 delta=-0.8000 pct=-1.8% (ns)
launch_host      launch           fixed device map                 ns/call  baseline=41.0000 new=40.9000 delta=-0.1000 pct=-0.2% (ns)
launch_host      launch           fixed device no-map              ns/call  baseline=42.7000 new=41.1000 delta=-1.6000 pct=-3.7% (ns)
launch_host      launch           generic                          ns/call  baseline=44.4000 new=43.7000 delta=-0.7000 pct=-1.6% (ns)
launch_host      launch           generic tensor ann               ns/call  baseline=44.5000 new=43.8000 delta=-0.7000 pct=-1.6% (ns)
launch_host      launch           reduced key                      ns/call  baseline=41.2000 new=40.7000 delta=-0.5000 pct=-1.2% (ns)
launch_host      launch           tensor annotated                 ns/call  baseline=39.6000 new=39.5000 delta=-0.1000 pct=-0.3% (ns)
launch_host      launch           verify off                       ns/call  baseline=41.5000 new=41.2000 delta=-0.3000 pct=-0.7% (ns)
launch_host      launch           verify on                        ns/call  baseline=42.6000 new=41.7000 delta=-0.9000 pct=-2.1% (ns)
launch_last_key  sweep            16 alternate                     ns/call  baseline=64.4000 new=63.9000 delta=-0.5000 pct=-0.8% (ns)
launch_last_key  sweep            16 repeat                        ns/call  baseline=61.1000 new=61.1000 delta=0.0000 pct=+0.0% (ns)
launch_last_key  sweep            32 alternate                     ns/call  baseline=103.3 new=102.7 delta=-0.6000 pct=-0.6% (ns)
launch_last_key  sweep            32 repeat                        ns/call  baseline=98.6000 new=98.1000 delta=-0.5000 pct=-0.5% (ns)
launch_last_key  sweep            4 alternate                      ns/call  baseline=34.5000 new=34.1000 delta=-0.4000 pct=-1.2% (ns)
launch_last_key  sweep            4 repeat                         ns/call  baseline=32.4000 new=31.9000 delta=-0.5000 pct=-1.5% (ns)
launch_readme    readme_path      grid=(0,) intj                   us       baseline=0.0300 new=0.0300 delta=0.0000 pct=+0.0% (us)
launch_readme    readme_path      grid=(0,) triton                 us       baseline=13.1000 new=12.6400 delta=-0.4600 pct=-3.5% (us)
launch_readme    readme_path      grid=(1,) intj                   us       baseline=3.1300 new=3.3100 delta=0.1800 pct=+5.8% (us) REGRESSION
launch_readme    readme_path      grid=(1,) triton                 us       baseline=16.6100 new=16.5100 delta=-0.1000 pct=-0.6% (us)
launch_readme    readme_torch_access interpreter                      decode ns baseline=894.1 new=892.5 delta=-1.6000 pct=-0.2% (ns)
launch_readme    readme_torch_access runtime_shim                     decode ns baseline=92.2000 new=89.3000 delta=-2.9000 pct=-3.1% (ns)
launch_readme    readme_torch_access static_compile                   decode ns baseline=95.0000 new=93.2000 delta=-1.8000 pct=-1.9% (ns)
launch_sweep     sweep            16 int                           ns/call  baseline=67.1000 new=66.7000 delta=-0.4000 pct=-0.6% (ns)
launch_sweep     sweep            16 tensor                        ns/call  baseline=63.0000 new=62.8000 delta=-0.2000 pct=-0.3% (ns)
launch_sweep     sweep            32 int                           ns/call  baseline=105.3 new=105.7 delta=0.4000 pct=+0.4% (ns)
launch_sweep     sweep            32 tensor                        ns/call  baseline=106.7 new=105.7 delta=-1.0000 pct=-0.9% (ns)
launch_sweep     sweep            4 int                            ns/call  baseline=38.7000 new=38.1000 delta=-0.6000 pct=-1.6% (ns)
launch_sweep     sweep            4 tensor                         ns/call  baseline=37.1000 new=36.6000 delta=-0.5000 pct=-1.3% (ns)
launch_tuned     tuned            plain                            ns/launch baseline=3044.1 new=3091.9 delta=47.8000 pct=+1.6% (ns)
launch_tuned     tuned            tuned hit                        ns/launch baseline=3104.0 new=3140.7 delta=36.7000 pct=+1.2% (ns)
```

### `bench_mix.py` comparison (`mix_compare.py`, rows beyond ±5 % or flagged)

```
16 100 str - alternate 75.8 -> 78.7  +2.9 ns +3.8% REGRESSION
32 25 dtype tensor repeat 100.6 -> 105.0  +4.4 ns +4.4% REGRESSION
32 50 int int repeat 87.7 -> 90.6  +2.9 ns +3.3% REGRESSION
32 50 int int alternate 111.2 -> 115.7  +4.5 ns +4.0% REGRESSION
32 100 str - repeat 81.0 -> 76.4  -4.6 ns -5.7% 
32 100 dtype - repeat 81.3 -> 76.5  -4.8 ns -5.9% 
rows=144 flagged=4 pct range -5.9..+4.4 median -0.3
```

### `/tmp/intj-bench/rerun-venv-bc8caf7`

#### `round0_ffi_compare.txt`

```
commit=bc8caf71f9323fb05dc2458f1e12ed8dd9ede5f8
started=2026-10-14T16:27:37+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/mnt/nvme2/jinpli/workspace/home/jinpli/development/benchmark taskset -c 0 /mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/venv/bin/python benchmarks/bench_ffi_compare.py --iters 1000 --batches 9
iters=1000 batches=9 unit=us/call (host-only, no timed sync)
FFI packed nop             median=0.1443 samples=[0.149011, 0.145007, 0.143262, 0.144332, 0.14284200000000002, 0.146604, 0.143241, 0.145895, 0.144194]
FFI typed nop              median=0.1488 samples=[0.14843299999999998, 0.15196, 0.146563, 0.150924, 0.14877500000000002, 0.150167, 0.14635499999999999, 0.14884899999999998, 0.15121600000000002]
FFI empty kernel           median=3.3165 samples=[3.810563, 3.5253609999999997, 3.55835, 3.315991, 3.344228, 3.289297, 3.31323, 3.285677, 3.316514]
INTJ empty kernel          median=2.9602 samples=[3.116085, 3.154604, 3.173993, 2.9601729999999997, 2.9595569999999998, 2.9156210000000002, 2.9372559999999996, 2.9064609999999997, 2.966169]
INTJ fixed-device kernel   median=2.9392 samples=[3.021525, 3.130989, 3.130467, 2.978958, 2.927736, 2.939179, 2.92451, 2.9205900000000002, 2.9237800000000003]
FFI packed nop mixed       median=0.1709 samples=[0.172944, 0.172236, 0.172697, 0.170893, 0.170422, 0.169971, 0.17002099999999998, 0.17155099999999998, 0.170712]
FFI typed nop mixed        median=0.1762 samples=[0.17575200000000002, 0.17622300000000002, 0.175514, 0.177083, 0.173779, 0.180318, 0.17463, 0.17846199999999998, 0.177643]
FFI mixed kernel           median=3.3570 samples=[3.49594, 3.527851, 3.4036939999999998, 3.399756, 3.34347, 3.330041, 3.35697, 3.312164, 3.3370659999999996]
INTJ mixed kernel          median=2.9431 samples=[3.1394810000000004, 3.114532, 3.008496, 2.934947, 2.9431410000000002, 2.85903, 2.937172, 2.867147, 2.951765]
INTJ fixed mixed kernel    median=2.9929 samples=[3.138192, 3.1675590000000002, 3.056482, 3.016168, 2.985786, 2.921773, 2.9757890000000002, 2.92123, 2.992947]
elapsed_ns=10161568385
exit_status=0
ended=2026-10-14T16:27:47+08:00
```

#### `round0_ffi_paths.txt`

```
commit=bc8caf71f9323fb05dc2458f1e12ed8dd9ede5f8
started=2026-10-14T16:27:58+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/mnt/nvme2/jinpli/workspace/home/jinpli/development/benchmark taskset -c 0 /mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/venv/bin/python benchmarks/bench_intj_ffi_paths.py --mode all --iters 1000 --batches 9
iters=1000 batches=9 unit=ns/call; no_gpu=True
mode=callback
callback_count=9100 (100 warmup + 9000 unique misses)
INTJ cold compile callback + cache  median_ns=686.2 samples_ns=[713.855, 712.631, 657.365, 770.385, 650.514, 656.167, 663.56, 686.212, 1031.243]
INTJ hot call (no callback)         median_ns=36.4 samples_ns=[37.688, 36.437, 36.197, 36.78, 36.257, 35.784, 36.581, 36.509, 35.637]
INTJ 3-tensor host-only nop         median_ns=32.1 samples_ns=[34.555, 32.265, 32.056, 32.002, 32.052, 32.017, 32.091, 32.055, 32.055]
mode=kwargs
INTJ direct positional              median_ns=64.9 samples_ns=[64.894, 65.038, 68.776, 63.143, 63.884, 67.237, 65.434, 63.333, 63.558]
INTJ adapter positional             median_ns=90.6 samples_ns=[89.19, 90.802, 89.53, 90.564, 89.588, 89.706, 101.678, 92.07, 91.942]
INTJ adapter kwargs                 median_ns=108.7 samples_ns=[109.447, 108.314, 108.93, 108.732, 109.857, 108.409, 113.445, 106.213, 108.451]
INTJ adapter defaults               median_ns=89.4 samples_ns=[89.41, 87.415, 89.703, 89.0, 91.172, 88.201, 89.846, 87.348, 92.184]
INTJ FFI wrapper positional         median_ns=104.0 samples_ns=[104.224, 103.093, 104.018, 103.645, 104.619, 103.905, 103.882, 104.471, 106.297]
INTJ FFI wrapper kwargs             median_ns=128.4 samples_ns=[130.217, 128.767, 128.505, 129.307, 128.434, 126.654, 127.905, 127.6, 125.125]
INTJ native kwargs: unsupported (verified TypeError)
mode=dataclass
INTJ pair direct                    median_ns=64.0 samples_ns=[66.738, 63.875, 64.12, 63.971, 64.087, 63.946, 63.987, 64.017, 67.326]
INTJ pair prebuilt *tuple           median_ns=134.6 samples_ns=[135.552, 134.587, 136.135, 137.105, 132.944, 134.362, 139.275, 133.616, 133.911]
FFI unpack Pair only                median_ns=165.3 samples_ns=[166.179, 165.076, 165.426, 164.189, 166.02, 165.291, 165.408, 163.941, 163.915]
INTJ pair manual unpack             median_ns=161.6 samples_ns=[161.393, 164.889, 161.121, 161.736, 161.063, 161.945, 161.564, 168.853, 158.824]
INTJ pair FFI unpack                median_ns=312.1 samples_ns=[312.951, 313.184, 320.819, 310.365, 312.122, 314.834, 311.913, 310.858, 311.712]
INTJ pair stdlib astuple            median_ns=1134.6 samples_ns=[1227.001, 1159.214, 1125.996, 1128.173, 1140.005, 1134.61, 1136.111, 1126.178, 1127.696]
INTJ config direct                  median_ns=77.3 samples_ns=[78.076, 77.335, 77.111, 76.871, 77.29, 77.485, 80.435, 76.385, 76.112]
INTJ config FFI unpack              median_ns=325.6 samples_ns=[323.895, 326.082, 329.612, 325.559, 324.877, 327.483, 323.944, 323.286, 328.998]
INTJ native dataclass unpack: unsupported (verified TypeError)
elapsed_ns=9361819823
exit_status=0
ended=2026-10-14T16:28:07+08:00
```

#### `round0_ffi_sweep.txt`

```
commit=bc8caf71f9323fb05dc2458f1e12ed8dd9ede5f8
started=2026-10-14T16:27:47+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/mnt/nvme2/jinpli/workspace/home/jinpli/development/benchmark taskset -c 0 /mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/venv/bin/python benchmarks/bench_ffi_compare.py --sweep --iters 1000 --batches 9
counts=[0, 3, 5, 8, 16, 32, 64] iters=1000 batches=9 unit=us/call (host-only, no timed sync); INTJ device-bound, no specialization key
args= 0 FFI packed nop           median=0.1163 samples=[0.117592, 0.11615, 0.116354, 0.116747, 0.114941, 0.115098, 0.117164, 0.116288, 0.111775]
args= 0 FFI typed nop            median=0.1215 samples=[0.12621, 0.122512, 0.122507, 0.121516, 0.121747, 0.121324, 0.11498900000000001, 0.12026099999999999, 0.116495]
args= 0 FFI empty kernel         median=1.5759 samples=[1.505604, 1.876043, 1.565803, 1.797094, 1.5696569999999999, 1.798884, 1.572762, 1.7946980000000001, 1.57593]
args= 0 INTJ static_compile kernel median=2.8035 samples=[3.1636640000000003, 2.906438, 2.785549, 2.766536, 2.803455, 2.774067, 2.815444, 2.781277, 2.8105149999999997]
args= 0 INTJ runtime_shim kernel median=2.7187 samples=[2.906597, 2.7185900000000003, 2.718668, 2.699387, 2.747133, 2.697565, 2.756363, 2.694735, 2.7511419999999998]
args= 3 FFI packed nop           median=0.1444 samples=[0.144386, 0.143263, 0.148499, 0.144719, 0.146206, 0.142764, 0.145478, 0.143817, 0.14433600000000002]
args= 3 FFI typed nop            median=0.1500 samples=[0.150001, 0.15514599999999998, 0.146892, 0.151097, 0.149255, 0.150781, 0.14918, 0.152384, 0.147767]
args= 3 FFI empty kernel         median=3.3172 samples=[3.425432, 3.413387, 3.214928, 3.318483, 3.2663800000000003, 3.490043, 3.260905, 3.3171619999999997, 3.267096]
args= 3 INTJ static_compile kernel median=2.9786 samples=[3.050746, 2.968719, 3.0663899999999997, 2.941545, 2.98165, 2.946407, 2.999589, 2.946701, 2.978619]
args= 3 INTJ runtime_shim kernel median=2.9316 samples=[3.058115, 2.94082, 2.938042, 2.93155, 2.927993, 2.930234, 2.916093, 2.919242, 2.932087]
args= 5 FFI packed nop           median=0.1722 samples=[0.17185599999999998, 0.17077099999999998, 0.174868, 0.170088, 0.173073, 0.172181, 0.169962, 0.174588, 0.17274199999999998]
args= 5 FFI typed nop            median=0.1787 samples=[0.176048, 0.179953, 0.176292, 0.179101, 0.176302, 0.181506, 0.17637899999999998, 0.179655, 0.178711]
args= 5 FFI empty kernel         median=3.3645 samples=[3.4558310000000003, 3.382037, 3.582573, 3.366269, 3.321189, 3.358652, 3.334036, 3.3644819999999998, 3.331097]
args= 5 INTJ static_compile kernel median=3.0125 samples=[3.066103, 3.013376, 3.028114, 2.973784, 3.010255, 2.973949, 3.012526, 2.9718359999999997, 3.013529]
args= 5 INTJ runtime_shim kernel median=2.9878 samples=[3.045752, 2.943644, 3.009382, 2.920209, 2.987759, 2.8841170000000003, 2.992684, 2.905805, 2.9894789999999998]
args= 8 FFI packed nop           median=0.2035 samples=[0.204231, 0.20869900000000002, 0.204706, 0.203505, 0.208781, 0.202132, 0.20314, 0.203068, 0.202906]
args= 8 FFI typed nop            median=0.2119 samples=[0.213028, 0.211877, 0.208854, 0.213477, 0.20697, 0.213053, 0.208313, 0.212118, 0.209696]
args= 8 FFI empty kernel         median=3.4331 samples=[3.5488020000000002, 3.434289, 3.445207, 3.400676, 3.434691, 3.4043989999999997, 3.433105, 3.394029, 3.4290979999999998]
args= 8 INTJ static_compile kernel median=3.0414 samples=[3.130591, 3.008826, 3.109542, 3.052988, 2.9862040000000003, 3.041417, 2.9820680000000004, 3.0495360000000002, 3.001536]
args= 8 INTJ runtime_shim kernel median=2.9930 samples=[3.14021, 2.978853, 3.013966, 2.952208, 2.99299, 2.935644, 3.019841, 2.940254, 3.0028319999999997]
args=16 FFI packed nop           median=0.3069 samples=[0.30489499999999997, 0.301548, 0.31366000000000005, 0.309953, 0.30685, 0.308964, 0.301263, 0.307315, 0.30431400000000003]
args=16 FFI typed nop            median=0.3143 samples=[0.31679199999999996, 0.31213799999999997, 0.309818, 0.317699, 0.308884, 0.316591, 0.314286, 0.310796, 0.315152]
args=16 FFI empty kernel         median=3.7425 samples=[3.801024, 3.781416, 3.624071, 3.77971, 3.626448, 3.775613, 3.6463560000000004, 3.742462, 3.634029]
args=16 INTJ static_compile kernel median=3.4414 samples=[3.3870479999999996, 3.441357, 3.638072, 3.42779, 3.606513, 3.373047, 3.666188, 3.397818, 3.650676]
args=16 INTJ runtime_shim kernel median=3.3805 samples=[3.3804600000000002, 3.203071, 3.490958, 3.118231, 3.484172, 3.12247, 3.500998, 3.10876, 3.474602]
args=32 FFI packed nop           median=0.4859 samples=[0.482119, 0.48585300000000003, 0.485075, 0.489008, 0.482432, 0.487679, 0.490825, 0.476117, 0.488343]
args=32 FFI typed nop            median=0.4988 samples=[0.493407, 0.496525, 0.49503800000000003, 0.499698, 0.498893, 0.49143000000000003, 0.498939, 0.499045, 0.49879399999999996]
args=32 FFI empty kernel         median=4.2266 samples=[4.334785, 4.304746, 4.163704, 4.268413, 4.152849, 4.272299, 4.174758, 4.226647, 4.167734]
args=32 INTJ static_compile kernel median=3.8220 samples=[3.6145259999999997, 3.8671480000000003, 3.896038, 3.773304, 3.84535, 3.785919, 3.8220300000000003, 3.762538, 3.861493]
args=32 INTJ runtime_shim kernel median=3.6685 samples=[3.6423449999999997, 3.668483, 3.7845549999999997, 3.5677339999999997, 3.7609090000000003, 3.545508, 3.7471419999999998, 3.563896, 3.784955]
args=64 FFI packed nop           median=0.8493 samples=[0.856075, 0.84469, 0.841536, 0.8493099999999999, 0.832091, 0.8510599999999999, 0.854943, 0.8469679999999999, 0.8510890000000001]
args=64 FFI typed nop            median=0.8769 samples=[0.861094, 0.8768859999999999, 0.880445, 0.8805510000000001, 0.872246, 0.874414, 0.866574, 0.878326, 0.880837]
args=64 FFI empty kernel         median=5.3841 samples=[5.43221, 5.509789, 5.292948, 5.384148, 5.298865, 5.391818, 5.274198, 5.400881999999999, 5.323923]
args=64 INTJ static_compile kernel median=4.9376 samples=[4.955527, 4.967943, 4.944416, 4.856309, 4.9375789999999995, 4.9069970000000005, 4.922322, 4.914402, 4.941876000000001]
args=64 INTJ runtime_shim kernel median=4.8300 samples=[4.834692, 4.89647, 4.821416, 4.732553, 4.8202110000000005, 4.835448, 4.831486, 4.830043, 4.829472]
elapsed_ns=10788152408
exit_status=0
ended=2026-10-14T16:27:58+08:00
```

#### `round0_hip_module.txt`

```
commit=bc8caf71f9323fb05dc2458f1e12ed8dd9ede5f8
started=2026-10-14T16:28:07+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/mnt/nvme2/jinpli/workspace/home/jinpli/development/benchmark taskset -c 0 /mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/venv/bin/python benchmarks/bench_hip_module_launch.py --iters 1000 --batches 9
hsaco_sha256=03409dde23b75ed425f9f7cc5341892b5f85c3c81874562ccaec070852969125 kernel=empty_kernel threads=256 shared=0 callback_count=1
iters=1000 batches=9 unit=us/call (host enqueue, no timed sync)
TVM FFI HIP HSACO         median=3.2322 samples=[3.40205, 3.233158, 3.2321709999999997, 3.300695, 3.272373, 3.225242, 3.2144009999999996, 3.2166770000000002, 3.2218910000000003]
Triton same HSACO         median=15.6022 samples=[15.744254000000002, 15.708191000000001, 15.633815, 15.712168, 15.505443, 15.538359, 15.507658, 15.602182, 15.601475]
INTJ same function        median=3.0391 samples=[3.050541, 3.039072, 3.117432, 3.121405, 2.992077, 2.999447, 2.994577, 3.222973, 2.9915410000000002]
elapsed_ns=10144594744
exit_status=0
ended=2026-10-14T16:28:17+08:00
```

#### `round0_kernel_cache.txt`

```
commit=bc8caf71f9323fb05dc2458f1e12ed8dd9ede5f8
started=2026-10-14T16:28:17+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/mnt/nvme2/jinpli/workspace/home/jinpli/development/benchmark taskset -c 0 /mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/venv/bin/python -m pytest tests/test_kernel_cache.py -s -q

kernel cache, 1-word key, ns per operation

                      intj       tsl      absl
hash_only             1.36      1.36      1.36
hit/1                 1.56      2.02      1.44
hit/8                 1.59      2.10      4.36
hit/64                1.68      2.14      4.64
hit/512               2.29      2.91      5.31
hit_child             3.01      3.00      3.01
miss/1                1.26      1.60      1.39
miss/8                2.68      1.75      3.29
miss/64               2.75      1.77      3.52
miss/512              2.91      2.00      4.06

kernel cache, 2-word key, ns per operation

                      intj       tsl      absl
hash_only             1.42      1.42      1.42
hit/1                 1.82      2.56      1.99
hit/8                 1.79      2.56      5.04
hit/64                2.19      2.93      5.22
hit/512               3.10      4.04      6.38
hit_child             3.00      3.00      3.01
miss/1                1.38      2.33      1.47
miss/8                2.41      1.86      3.20
miss/64               3.14      2.11      3.89
miss/512              2.92      2.73      3.98

kernel cache, 5-word key, ns per operation

                      intj       tsl      absl
hash_only             3.58      3.59      3.58
hit/1                 3.01      3.05      2.46
hit/8                 3.20      3.41      9.48
hit/64                3.27      3.51      9.77
hit/512               4.53      4.78     11.20
hit_child             3.00      3.01      3.00
miss/1                1.40      2.11      2.01
miss/8                2.95      2.25      6.93
miss/64               2.83      1.96      6.93
miss/512              3.46      2.50      7.71
........
8 passed in 31.56s
elapsed_ns=32372155786
exit_status=0
ended=2026-10-14T16:28:50+08:00
```

#### `round0_launch_dynamic.txt`

```
commit=bc8caf71f9323fb05dc2458f1e12ed8dd9ede5f8
started=2026-10-14T16:29:01+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/mnt/nvme2/jinpli/workspace/home/jinpli/development/benchmark taskset -c 0 /mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/venv/bin/python benchmarks/bench_launch.py --dynamic --iters 20000 --batches 9
mode=dynamic; 20000 calls × 9 batches; median ns/call
               path     launch   spec_key
         lazy plain     3102.7       94.1
  2 dynamic options     3074.3       90.0
      int constexpr     3014.9       85.5
      str constexpr     3019.2       86.9
elapsed_ns=12178154763
exit_status=0
ended=2026-10-14T16:29:13+08:00
```

#### `round0_launch_gpu.txt`

```
commit=bc8caf71f9323fb05dc2458f1e12ed8dd9ede5f8
started=2026-10-14T16:26:48+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/mnt/nvme2/jinpli/workspace/home/jinpli/development/benchmark taskset -c 0 /mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/venv/bin/python benchmarks/bench_launch.py --iters 1000 --batches 9
mode=gpu; 1000 calls × 9 batches; median ns/call
               path    ns/call       Δ ns       Δ %   build ms   bind ms
           auto map     3259.5       +0.0      +0.0      18.21         -
   tensor annotated     3329.0      +69.5      +2.1       3.68         -
            generic     3183.4      -76.1      -2.3       2.90         -
 generic tensor ann     3146.9     -112.6      -3.5       2.86         -
        reduced key     3123.7     -135.8      -4.2       3.01         -
         verify off     3102.4     -157.1      -4.8       2.88         -
          verify on     3119.5     -140.0      -4.3       2.86         -
              baked     3131.6     -127.9      -3.9       2.91         -
       bound tensor     3133.7     -125.8      -3.9       2.81      0.03
      bound pointer     3114.4     -145.1      -4.5       2.80      0.02
   fixed device map     3119.0     -140.5      -4.3       2.72      0.02
fixed device no-map     3117.6     -141.9      -4.4       2.76      0.02
elapsed_ns=10282023112
exit_status=0
ended=2026-10-14T16:26:58+08:00
```

#### `round0_launch_host.txt`

```
commit=bc8caf71f9323fb05dc2458f1e12ed8dd9ede5f8
started=2026-10-14T16:26:58+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/mnt/nvme2/jinpli/workspace/home/jinpli/development/benchmark taskset -c 0 /mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/venv/bin/python benchmarks/bench_launch.py --no-gpu --iters 100000 --batches 9
mode=host-only; 100000 calls × 9 batches; median ns/call
               path    ns/call       Δ ns       Δ %   build ms   bind ms
           auto map       39.8       +0.0      +0.0       6.81         -
   tensor annotated       39.4       -0.4      -1.0       2.26         -
            generic       43.7       +3.9      +9.8       2.17         -
 generic tensor ann       43.8       +4.0     +10.1       2.16         -
        reduced key       40.7       +0.9      +2.1       2.18         -
         verify off       41.1       +1.3      +3.3       2.19         -
          verify on       41.7       +1.9      +4.7       2.24         -
              baked       37.7       -2.1      -5.2       2.17         -
       bound tensor       42.8       +3.0      +7.5       2.24      0.02
      bound pointer       41.2       +1.4      +3.4       2.16      0.02
   fixed device map       40.9       +1.0      +2.6       2.12      0.02
fixed device no-map       41.0       +1.1      +2.8       2.19      0.02
elapsed_ns=9472474743
exit_status=0
ended=2026-10-14T16:27:08+08:00
```

#### `round0_launch_last_key.txt`

```
commit=bc8caf71f9323fb05dc2458f1e12ed8dd9ede5f8
started=2026-10-14T16:27:27+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/mnt/nvme2/jinpli/workspace/home/jinpli/development/benchmark taskset -c 0 /mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/venv/bin/python benchmarks/bench_launch.py --last-key --iters 100000 --batches 9
mode=last-key; host-only; 200000 calls × 9 batches; median ns/call
count     pattern    ns/call
    4      repeat       31.9
    4   alternate       34.2
   16      repeat       61.1
   16   alternate       64.0
   32      repeat       97.8
   32   alternate      103.2
elapsed_ns=9706355890
exit_status=0
ended=2026-10-14T16:27:37+08:00
```

#### `round0_launch_readme.txt`

```
commit=bc8caf71f9323fb05dc2458f1e12ed8dd9ede5f8
started=2026-10-14T16:27:08+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/mnt/nvme2/jinpli/workspace/home/jinpli/development/benchmark taskset -c 0 /mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/venv/bin/python benchmarks/bench_launch.py --readme --iters 1000 --batches 9
mode=readme; 1000 calls × 9 batches; median
        path   triton us    intj us   speedup
   grid=(1,)       16.56       3.31      5.0x
   grid=(0,)       12.89       0.03    474.6x

torch_access_mode   decode ns    build s
     runtime_shim        89.3       0.02
   static_compile        92.5       0.00
      interpreter       892.5       0.00
elapsed_ns=10206149596
exit_status=0
ended=2026-10-14T16:27:18+08:00
```

#### `round0_launch_sweep.txt`

```
commit=bc8caf71f9323fb05dc2458f1e12ed8dd9ede5f8
started=2026-10-14T16:27:18+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/mnt/nvme2/jinpli/workspace/home/jinpli/development/benchmark taskset -c 0 /mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/venv/bin/python benchmarks/bench_launch.py --sweep --iters 100000 --batches 9
mode=sweep; host-only; 100000 calls × 9 batches; median ns/call
count   kind    ns/call
    4    int       38.1
    4 tensor       36.6
   16    int       66.8
   16 tensor       63.2
   32    int      104.3
   32 tensor      105.7
elapsed_ns=9364910109
exit_status=0
ended=2026-10-14T16:27:27+08:00
```

#### `round0_launch_tuned.txt`

```
commit=bc8caf71f9323fb05dc2458f1e12ed8dd9ede5f8
started=2026-10-14T16:28:50+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/mnt/nvme2/jinpli/workspace/home/jinpli/development/benchmark taskset -c 0 /mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/venv/bin/python benchmarks/bench_launch.py --tuned --iters 20000 --batches 9
 tuned hit:  3141.6 ns/launch
     plain:  3091.5 ns/launch
elapsed_ns=11030464559
exit_status=0
ended=2026-10-14T16:29:01+08:00
```

#### `round0_mix.txt`

```
commit=bc8caf71f9323fb05dc2458f1e12ed8dd9ede5f8
started=2026-10-14T16:29:13+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/mnt/nvme2/jinpli/workspace/home/jinpli/development/benchmark taskset -c 0 /mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/venv/bin/python benchmarks/bench_mix.py
intj 0.1.0 from /mnt/nvme2/jinpli/workspace/home/jinpli/development/intj/intj
python 3.12.3; triton 3.8.0; torch 2.14.0.dev20260804+rocm7.1
mode=mix; host-only; 40000 calls × 9 batches; median ns/call
count share ckind  nkind   pattern   ns/call  samples
    4     0     -    int    repeat      31.9  31.9 32.0 31.9 31.8 31.9 31.8 31.9 31.9 32.3
    4     0     -    int alternate      33.9  33.8 34.0 34.3 33.9 34.0 33.8 33.8 34.0 33.9
    4     0     - tensor    repeat      30.6  30.4 30.6 30.6 30.6 30.4 30.5 30.8 30.7 30.5
    4     0     - tensor alternate      33.0  32.9 33.2 33.1 32.8 33.0 32.8 32.9 33.0 33.0
    4     0     -  mixed    repeat      30.4  30.4 30.3 30.3 30.4 30.4 30.4 30.3 31.6 31.3
    4     0     -  mixed alternate      32.4  32.5 32.7 32.5 32.5 32.4 32.3 32.4 32.4 32.4
    4    25   int    int    repeat      31.0  31.0 30.9 31.1 31.0 30.9 30.9 31.0 31.0 30.9
    4    25   int    int alternate      33.3  33.5 33.4 33.3 33.5 33.1 33.5 33.2 33.2 33.2
    4    25   int tensor    repeat      30.1  30.2 30.0 30.0 30.2 30.1 30.1 30.0 30.1 30.2
    4    25   int tensor alternate      32.7  32.6 32.7 32.7 32.7 33.1 32.7 32.7 32.7 32.8
    4    25   int  mixed    repeat      29.5  29.5 29.5 29.5 29.5 29.4 29.7 29.7 29.5 29.6
    4    25   int  mixed alternate      31.9  31.8 31.8 31.7 31.9 32.0 31.9 31.9 31.7 31.9
    4    25   str    int    repeat      30.7  30.8 30.8 30.7 30.7 30.7 30.8 30.7 30.6 30.7
    4    25   str    int alternate      38.9  38.8 38.9 38.9 38.9 39.0 38.9 38.9 38.8 38.8
    4    25   str tensor    repeat      30.1  30.1 30.0 30.0 30.1 30.0 30.3 30.1 30.1 30.1
    4    25   str tensor alternate      37.8  37.8 37.9 37.8 37.9 37.8 37.8 37.7 38.1 38.0
    4    25   str  mixed    repeat      29.5  29.9 29.5 29.3 29.5 29.5 29.5 29.7 29.5 29.6
    4    25   str  mixed alternate      38.8  38.8 39.0 41.1 38.8 38.9 38.7 39.3 38.7 38.8
    4    25 dtype    int    repeat      30.8  30.8 30.8 30.8 30.7 30.7 30.7 31.0 30.8 30.9
    4    25 dtype    int alternate      38.2  38.3 38.1 38.2 38.2 38.2 38.2 38.2 38.0 38.2
    4    25 dtype tensor    repeat      30.0  30.0 30.1 30.0 30.0 30.0 30.0 30.0 30.0 30.0
    4    25 dtype tensor alternate      36.1  36.0 36.1 36.5 36.2 36.1 36.1 36.3 36.1 36.2
    4    25 dtype  mixed    repeat      29.5  29.5 29.5 29.5 29.5 29.6 29.6 29.5 29.5 29.5
    4    25 dtype  mixed alternate      36.0  36.0 35.9 36.0 35.9 36.0 36.0 35.8 36.3 35.8
    4    50   int    int    repeat      30.4  30.5 30.7 30.4 30.4 30.4 30.5 30.3 30.2 30.3
    4    50   int    int alternate      33.6  33.6 33.5 33.6 33.5 33.9 33.6 33.5 33.4 33.7
    4    50   int tensor    repeat      29.8  29.7 29.8 29.8 29.7 29.7 30.0 35.0 30.0 29.9
    4    50   int tensor alternate      33.8  33.6 33.8 33.5 33.6 33.9 34.3 33.9 34.2 33.5
    4    50   int  mixed    repeat      30.0  30.1 30.0 30.0 30.1 30.0 30.0 29.9 29.9 30.1
    4    50   int  mixed alternate      33.9  34.3 33.8 33.9 34.1 34.3 33.9 34.2 33.8 33.9
    4    50   str    int    repeat      30.2  30.2 30.1 30.2 30.1 30.2 30.4 30.6 30.0 30.3
    4    50   str    int alternate      39.8  39.7 39.9 39.8 39.8 39.8 39.9 39.8 39.9 39.7
    4    50   str tensor    repeat      29.9  29.9 29.9 30.0 29.8 30.0 29.9 29.8 29.8 29.9
    4    50   str tensor alternate      38.9  38.7 38.8 38.7 38.9 38.9 39.4 39.0 38.9 38.9
    4    50   str  mixed    repeat      30.1  30.2 30.2 30.0 30.1 30.0 30.0 30.2 30.1 30.0
    4    50   str  mixed alternate      39.1  39.1 39.3 38.9 39.0 38.8 39.1 38.9 39.1 39.1
    4    50 dtype    int    repeat      30.2  30.0 30.1 30.2 30.2 30.4 30.3 30.2 30.2 30.2
    4    50 dtype    int alternate      38.1  38.2 38.1 38.1 38.1 38.1 38.0 38.0 38.0 37.8
    4    50 dtype tensor    repeat      30.0  29.9 30.0 30.0 29.9 30.1 30.0 30.0 29.9 30.0
    4    50 dtype tensor alternate      37.0  36.9 37.0 37.0 36.9 37.0 37.1 36.9 37.0 36.9
    4    50 dtype  mixed    repeat      30.1  30.3 30.1 30.1 30.1 30.1 30.1 29.9 30.2 30.1
    4    50 dtype  mixed alternate      37.7  37.8 37.4 37.5 37.6 37.8 37.5 37.7 38.0 37.7
    4   100   int      -    repeat      28.6  28.6 28.6 28.7 28.6 28.5 28.5 28.4 28.6 28.8
    4   100   int      - alternate      33.8  33.6 33.8 33.6 33.8 33.8 33.8 34.0 33.8 33.6
    4   100   str      -    repeat      29.5  29.5 29.4 29.5 29.7 29.4 29.5 29.5 29.6 29.4
    4   100   str      - alternate      39.8  39.8 39.7 39.8 39.7 39.8 39.7 39.8 39.7 39.8
    4   100 dtype      -    repeat      29.4  29.7 29.4 29.7 29.4 29.3 29.3 29.4 29.4 29.4
    4   100 dtype      - alternate      38.5  38.3 38.5 38.4 38.4 38.3 38.6 38.5 38.5 38.5
   16     0     -    int    repeat      61.1  61.1 61.2 61.2 61.4 61.0 61.1 61.0 61.5 61.0
   16     0     -    int alternate      63.9  63.9 64.1 64.1 63.8 63.9 63.9 64.1 64.1 63.8
   16     0     - tensor    repeat      57.3  57.3 57.3 57.3 57.4 57.3 57.2 57.4 57.3 57.3
   16     0     - tensor alternate      60.8  60.8 61.0 60.8 60.9 60.7 60.7 61.1 60.7 61.1
   16     0     -  mixed    repeat      55.4  55.5 55.4 55.5 55.4 55.7 55.2 55.3 55.4 55.2
   16     0     -  mixed alternate      59.2  59.2 59.2 59.3 59.4 59.1 59.2 59.1 59.2 59.1
   16    25   int    int    repeat      56.4  56.2 56.2 56.2 57.1 56.5 56.6 56.4 56.6 56.2
   16    25   int    int alternate      62.4  62.5 62.4 62.4 62.3 62.5 62.5 62.5 62.2 62.3
   16    25   int tensor    repeat      55.0  55.0 55.2 55.0 55.0 55.0 55.3 55.1 55.1 54.9
   16    25   int tensor alternate      62.0  61.8 62.1 62.0 61.9 61.9 62.1 62.0 61.8 62.0
   16    25   int  mixed    repeat      51.9  52.0 51.9 51.9 51.9 51.9 51.9 52.1 52.0 52.2
   16    25   int  mixed alternate      58.2  58.5 58.2 58.4 58.2 58.2 58.1 58.3 58.1 58.1
   16    25   str    int    repeat      57.1  57.0 57.3 57.1 57.2 57.3 57.1 57.3 57.1 57.1
   16    25   str    int alternate      68.6  68.8 68.3 68.6 68.6 68.5 69.0 68.7 68.9 68.5
   16    25   str tensor    repeat      55.4  55.3 55.4 55.4 55.3 55.4 55.4 55.7 55.4 55.8
   16    25   str tensor alternate      67.0  66.9 66.9 67.2 67.0 66.9 67.5 67.1 66.9 67.0
   16    25   str  mixed    repeat      53.1  53.1 53.0 53.3 53.2 53.1 53.1 53.1 52.7 52.9
   16    25   str  mixed alternate      63.9  63.7 63.9 63.9 63.9 63.9 64.0 63.8 63.6 64.0
   16    25 dtype    int    repeat      57.3  57.0 57.3 57.6 57.3 57.4 57.3 57.2 57.5 57.3
   16    25 dtype    int alternate      66.8  66.8 66.6 66.8 67.0 66.7 67.0 66.9 66.6 67.0
   16    25 dtype tensor    repeat      55.6  55.2 55.5 55.7 55.9 55.6 55.6 55.7 55.5 55.6
   16    25 dtype tensor alternate      66.4  66.5 66.5 66.3 66.6 66.3 66.3 66.3 66.4 66.4
   16    25 dtype  mixed    repeat      52.9  53.1 53.0 52.9 52.9 52.8 53.0 52.8 52.9 53.2
   16    25 dtype  mixed alternate      64.2  64.6 64.5 64.2 64.2 64.5 64.4 64.0 64.0 64.2
   16    50   int    int    repeat      56.4  56.5 56.2 56.4 56.0 56.6 55.9 56.8 56.1 56.7
   16    50   int    int alternate      67.7  68.2 67.6 67.9 67.5 67.8 67.5 67.7 67.4 67.8
   16    50   int tensor    repeat      55.3  54.9 55.3 55.1 55.3 56.8 55.5 54.8 55.3 54.8
   16    50   int tensor alternate      66.4  66.0 66.4 66.0 66.4 66.5 66.5 66.1 66.5 65.9
   16    50   int  mixed    repeat      54.1  54.4 54.0 54.4 54.0 54.2 53.6 54.1 53.9 54.4
   16    50   int  mixed alternate      65.2  65.2 64.7 65.5 65.0 65.4 65.0 65.2 65.5 65.2
   16    50   str    int    repeat      56.2  56.1 56.3 56.1 56.7 56.1 56.4 56.2 56.5 56.1
   16    50   str    int alternate      73.8  73.7 74.1 73.5 74.2 73.5 74.5 73.8 74.4 73.3
   16    50   str tensor    repeat      55.4  55.4 55.0 55.5 55.2 55.6 55.0 55.5 54.9 55.9
   16    50   str tensor alternate      71.4  71.4 71.0 75.5 71.0 71.6 72.5 71.3 71.4 71.5
   16    50   str  mixed    repeat      54.3  54.3 54.4 54.1 54.6 54.0 54.4 54.0 54.4 53.9
   16    50   str  mixed alternate      71.0  70.8 71.4 70.3 71.6 70.5 71.2 70.7 71.3 71.0
   16    50 dtype    int    repeat      56.3  56.4 56.0 56.5 56.1 56.4 56.0 56.4 56.2 56.3
   16    50 dtype    int alternate      70.6  70.9 70.5 70.7 70.4 70.8 70.4 70.6 70.2 71.0
   16    50 dtype tensor    repeat      55.3  55.4 55.4 54.8 55.2 54.7 55.7 54.9 55.3 55.3
   16    50 dtype tensor alternate      70.2  70.4 70.2 70.3 69.9 70.2 70.1 70.0 70.1 70.4
   16    50 dtype  mixed    repeat      54.3  54.5 53.7 54.4 53.7 54.4 53.8 54.3 54.0 54.4
   16    50 dtype  mixed alternate      68.3  68.4 68.0 68.6 68.2 68.3 68.2 68.6 68.7 68.3
   16   100   int      -    repeat      49.2  49.2 49.3 49.2 49.0 49.1 49.2 49.1 49.1 49.4
   16   100   int      - alternate      71.1  71.4 71.1 71.3 71.1 71.3 71.1 71.1 71.0 71.0
   16   100   str      -    repeat      49.6  49.6 49.6 49.6 49.6 49.6 49.7 49.6 49.5 49.6
   16   100   str      - alternate      78.7  78.2 78.8 78.3 78.7 78.1 78.8 79.1 78.7 78.4
   16   100 dtype      -    repeat      49.7  49.9 49.5 49.7 49.9 49.7 49.6 49.6 49.8 50.0
   16   100 dtype      - alternate      73.6  73.5 73.6 73.6 73.8 73.5 73.8 73.6 73.6 73.6
   32     0     -    int    repeat      97.7  97.7 97.5 97.4 97.3 97.7 97.0 100.9 98.7 100.0
   32     0     -    int alternate     104.2  102.7 103.5 106.4 104.2 106.2 104.7 101.5 102.9 104.6
   32     0     - tensor    repeat      98.1  106.2 103.8 103.5 99.2 98.1 96.1 95.9 97.9 96.2
   32     0     - tensor alternate     102.8  101.9 101.8 102.2 105.0 105.0 102.8 102.7 104.0 102.8
   32     0     -  mixed    repeat      88.2  87.8 88.4 88.1 88.0 87.9 88.2 88.3 88.4 88.4
   32     0     -  mixed alternate      93.5  93.5 93.6 94.1 93.9 93.1 93.5 93.4 93.7 93.4
   32    25   int    int    repeat      88.8  89.0 90.4 88.5 90.6 89.1 88.8 88.5 88.6 88.3
   32    25   int    int alternate     101.4  101.7 101.2 101.2 101.3 103.0 104.1 103.8 101.3 101.4
   32    25   int tensor    repeat      90.2  101.1 92.3 90.2 90.0 90.8 90.2 89.4 90.3 89.5
   32    25   int tensor alternate     102.5  104.0 103.2 102.1 102.7 101.8 102.6 101.6 102.5 101.4
   32    25   int  mixed    repeat      82.1  82.1 82.1 81.6 82.2 82.2 82.1 82.3 81.9 82.1
   32    25   int  mixed alternate      94.1  94.2 94.1 94.1 94.2 94.2 94.0 94.0 93.9 93.9
   32    25   str    int    repeat      89.7  90.0 89.6 89.7 89.7 89.7 89.6 89.8 89.9 89.7
   32    25   str    int alternate     107.6  107.9 107.3 107.5 107.5 108.0 107.6 107.6 107.4 107.8
   32    25   str tensor    repeat     102.9  105.9 103.0 102.9 103.6 101.9 103.8 102.8 102.6 98.7
   32    25   str tensor alternate     117.4  122.7 127.7 118.5 117.4 118.4 116.6 115.5 116.7 116.1
   32    25   str  mixed    repeat      83.7  83.7 83.4 83.9 83.4 84.0 84.0 84.0 83.5 83.2
   32    25   str  mixed alternate     103.0  102.8 102.9 103.4 102.6 103.0 103.0 103.4 103.4 103.2
   32    25 dtype    int    repeat      90.8  90.8 89.4 90.8 89.3 90.7 93.4 91.1 89.6 92.2
   32    25 dtype    int alternate     106.1  106.3 106.3 106.2 106.2 106.0 106.1 106.0 105.9 105.8
   32    25 dtype tensor    repeat     105.3  109.3 106.7 105.8 104.7 105.5 105.3 105.2 101.3 98.6
   32    25 dtype tensor alternate     111.7  114.4 111.3 111.6 111.3 111.9 113.2 111.7 111.7 111.8
   32    25 dtype  mixed    repeat      83.8  84.2 83.6 83.8 84.5 83.8 83.5 84.4 83.3 83.7
   32    25 dtype  mixed alternate      99.0  99.0 99.1 99.1 99.5 98.9 98.7 99.0 98.8 98.8
   32    50   int    int    repeat      90.6  90.3 90.1 90.6 91.5 91.5 93.0 90.6 90.9 89.2
   32    50   int    int alternate     114.3  115.0 115.5 115.0 114.2 114.3 115.8 111.3 112.9 114.1
   32    50   int tensor    repeat      86.1  86.0 86.0 86.6 86.1 85.9 86.2 86.1 86.1 85.9
   32    50   int tensor alternate     108.6  108.7 108.6 108.8 108.6 108.7 108.5 108.6 108.6 108.7
   32    50   int  mixed    repeat      82.7  82.6 83.3 82.6 82.6 82.7 82.7 82.7 82.7 83.3
   32    50   int  mixed alternate     106.4  106.4 106.3 106.4 106.3 106.5 106.4 106.5 106.3 106.3
   32    50   str    int    repeat      87.9  87.8 88.0 87.9 87.9 88.0 87.7 87.7 87.7 87.9
   32    50   str    int alternate     118.2  118.0 118.1 117.7 118.2 118.4 118.3 118.2 118.5 117.8
   32    50   str tensor    repeat      93.9  94.8 93.8 93.9 94.6 93.5 93.9 93.8 94.1 94.0
   32    50   str tensor alternate     118.7  124.4 122.5 121.2 118.7 118.2 120.7 118.4 118.6 118.2
   32    50   str  mixed    repeat      84.9  85.0 84.5 85.7 85.0 84.6 84.7 84.9 84.7 84.9
   32    50   str  mixed alternate     111.5  111.3 111.0 110.7 111.6 111.5 111.4 112.0 111.8 111.9
   32    50 dtype    int    repeat      87.8  87.7 87.8 87.8 87.5 87.7 87.6 88.0 87.8 87.8
   32    50 dtype    int alternate     116.7  116.6 116.8 116.7 116.7 116.8 116.6 116.8 116.8 116.7
   32    50 dtype tensor    repeat      94.2  96.2 94.8 94.2 93.6 94.2 94.1 95.3 95.0 93.7
   32    50 dtype tensor alternate     115.4  115.7 115.3 115.6 114.5 115.5 114.7 115.4 115.4 115.7
   32    50 dtype  mixed    repeat      84.8  84.7 84.5 84.8 85.6 84.9 84.6 84.7 84.8 85.4
   32    50 dtype  mixed alternate     110.1  110.7 110.3 110.1 109.9 110.4 109.7 110.1 109.8 109.8
   32   100   int      -    repeat      70.3  70.9 70.4 70.3 70.3 70.2 70.2 70.5 70.3 70.2
   32   100   int      - alternate     113.9  113.8 113.9 113.7 115.7 114.0 113.8 113.8 114.1 113.9
   32   100   str      -    repeat      75.6  80.8 81.4 76.1 75.6 76.7 75.6 75.6 75.5 75.6
   32   100   str      - alternate     123.4  125.7 123.9 123.4 122.7 123.0 124.1 122.7 123.9 123.1
   32   100 dtype      -    repeat      76.5  74.2 83.6 75.2 74.4 82.6 83.8 80.1 76.5 74.9
   32   100 dtype      - alternate     120.3  120.3 120.2 120.3 120.4 120.6 120.2 120.4 120.2 121.0
elapsed_ns=12369125473
exit_status=0
ended=2026-10-14T16:29:25+08:00
```

#### `round1_ffi_compare.txt`

```
commit=bc8caf71f9323fb05dc2458f1e12ed8dd9ede5f8
started=2026-10-14T16:30:14+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/mnt/nvme2/jinpli/workspace/home/jinpli/development/benchmark taskset -c 0 /mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/venv/bin/python benchmarks/bench_ffi_compare.py --iters 1000 --batches 9
iters=1000 batches=9 unit=us/call (host-only, no timed sync)
FFI packed nop             median=0.1436 samples=[0.16023300000000001, 0.146206, 0.14411000000000002, 0.141945, 0.143749, 0.143376, 0.14257599999999998, 0.14335900000000001, 0.143569]
FFI typed nop              median=0.1500 samples=[0.150972, 0.150006, 0.151899, 0.15025899999999998, 0.145755, 0.149481, 0.148675, 0.151834, 0.147166]
FFI empty kernel           median=3.3158 samples=[3.8112199999999996, 3.5287040000000003, 3.567704, 3.315772, 3.34442, 3.286409, 3.306095, 3.285411, 3.298822]
INTJ empty kernel          median=2.9569 samples=[3.1082959999999997, 3.169887, 3.185474, 2.9805569999999997, 2.9503139999999997, 2.9260129999999998, 2.956885, 2.9060639999999998, 2.949097]
INTJ fixed-device kernel   median=2.9497 samples=[3.033692, 3.142179, 3.151633, 2.979682, 2.949657, 2.93524, 2.926543, 2.9301350000000004, 2.935882]
FFI packed nop mixed       median=0.1718 samples=[0.175707, 0.173277, 0.169942, 0.172074, 0.170906, 0.17178, 0.171519, 0.17287, 0.169734]
FFI typed nop mixed        median=0.1759 samples=[0.175897, 0.176538, 0.17543999999999998, 0.176636, 0.175725, 0.178066, 0.175509, 0.17857800000000001, 0.172654]
FFI mixed kernel           median=3.3357 samples=[3.451335, 3.4990140000000003, 3.416832, 3.3845859999999997, 3.335699, 3.31863, 3.317268, 3.3114, 3.32063]
INTJ mixed kernel          median=2.9532 samples=[3.2240219999999997, 3.1189850000000003, 3.005987, 2.941305, 2.945794, 2.867133, 2.960067, 2.8632359999999997, 2.953223]
INTJ fixed mixed kernel    median=2.9795 samples=[3.158052, 3.1365250000000002, 3.0622179999999997, 2.9893009999999998, 2.979534, 2.917155, 2.9750010000000002, 2.9067950000000002, 2.9686179999999998]
elapsed_ns=10155976704
exit_status=0
ended=2026-10-14T16:30:25+08:00
```

#### `round1_ffi_paths.txt`

```
commit=bc8caf71f9323fb05dc2458f1e12ed8dd9ede5f8
started=2026-10-14T16:30:35+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/mnt/nvme2/jinpli/workspace/home/jinpli/development/benchmark taskset -c 0 /mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/venv/bin/python benchmarks/bench_intj_ffi_paths.py --mode all --iters 1000 --batches 9
iters=1000 batches=9 unit=ns/call; no_gpu=True
mode=callback
callback_count=9100 (100 warmup + 9000 unique misses)
INTJ cold compile callback + cache  median_ns=682.3 samples_ns=[719.541, 724.851, 646.795, 772.253, 652.159, 659.478, 664.177, 682.341, 1042.647]
INTJ hot call (no callback)         median_ns=34.4 samples_ns=[37.396, 35.154, 34.457, 34.383, 34.359, 34.384, 34.334, 34.311, 34.509]
INTJ 3-tensor host-only nop         median_ns=31.9 samples_ns=[34.493, 35.932, 31.9, 31.835, 31.864, 31.863, 31.857, 31.933, 31.824]
mode=kwargs
INTJ direct positional              median_ns=64.0 samples_ns=[65.63, 64.831, 64.787, 63.896, 63.981, 64.143, 63.412, 63.657, 63.878]
INTJ adapter positional             median_ns=91.0 samples_ns=[95.992, 90.507, 89.85, 90.436, 89.545, 91.115, 90.992, 91.186, 91.91]
INTJ adapter kwargs                 median_ns=108.6 samples_ns=[109.839, 111.942, 107.706, 107.149, 108.576, 107.262, 107.411, 110.224, 109.938]
INTJ adapter defaults               median_ns=89.7 samples_ns=[89.668, 90.165, 90.298, 88.989, 88.373, 89.306, 88.712, 90.816, 92.218]
INTJ FFI wrapper positional         median_ns=107.9 samples_ns=[105.282, 106.072, 104.763, 118.272, 109.342, 107.693, 108.674, 107.871, 107.854]
INTJ FFI wrapper kwargs             median_ns=129.2 samples_ns=[129.162, 127.949, 130.881, 129.204, 128.058, 127.439, 129.634, 132.35, 128.547]
INTJ native kwargs: unsupported (verified TypeError)
mode=dataclass
INTJ pair direct                    median_ns=63.7 samples_ns=[66.554, 63.734, 63.655, 63.529, 63.687, 64.507, 64.19, 64.467, 63.552]
INTJ pair prebuilt *tuple           median_ns=133.9 samples_ns=[136.582, 134.429, 136.297, 133.947, 133.582, 133.793, 133.413, 133.243, 135.247]
FFI unpack Pair only                median_ns=164.4 samples_ns=[166.678, 164.59, 164.869, 164.094, 164.401, 162.833, 165.845, 164.13, 163.664]
INTJ pair manual unpack             median_ns=160.8 samples_ns=[160.79, 159.249, 161.877, 162.979, 160.356, 160.235, 160.417, 162.531, 163.26]
INTJ pair FFI unpack                median_ns=310.1 samples_ns=[314.048, 306.723, 307.895, 317.004, 308.415, 309.389, 318.888, 310.072, 310.549]
INTJ pair stdlib astuple            median_ns=1144.0 samples_ns=[1238.011, 1174.801, 1139.7, 1143.963, 1144.471, 1157.773, 1136.986, 1142.773, 1143.227]
INTJ config direct                  median_ns=78.9 samples_ns=[79.951, 79.034, 78.839, 79.378, 80.119, 78.816, 78.774, 78.494, 78.871]
INTJ config FFI unpack              median_ns=322.5 samples_ns=[322.462, 318.617, 320.654, 323.17, 318.309, 318.029, 324.669, 323.911, 323.371]
INTJ native dataclass unpack: unsupported (verified TypeError)
elapsed_ns=9366795178
exit_status=0
ended=2026-10-14T16:30:45+08:00
```

#### `round1_ffi_sweep.txt`

```
commit=bc8caf71f9323fb05dc2458f1e12ed8dd9ede5f8
started=2026-10-14T16:30:25+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/mnt/nvme2/jinpli/workspace/home/jinpli/development/benchmark taskset -c 0 /mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/venv/bin/python benchmarks/bench_ffi_compare.py --sweep --iters 1000 --batches 9
counts=[0, 3, 5, 8, 16, 32, 64] iters=1000 batches=9 unit=us/call (host-only, no timed sync); INTJ device-bound, no specialization key
args= 0 FFI packed nop           median=0.1149 samples=[0.117142, 0.114967, 0.112678, 0.114173, 0.11337900000000001, 0.11517100000000001, 0.11757, 0.11368600000000001, 0.11488200000000001]
args= 0 FFI typed nop            median=0.1199 samples=[0.12130500000000001, 0.125166, 0.118215, 0.119167, 0.121284, 0.119312, 0.118645, 0.119887, 0.123587]
args= 0 FFI empty kernel         median=1.6417 samples=[1.584453, 1.981002, 1.6416659999999998, 1.815416, 1.6040139999999998, 1.802346, 1.605741, 1.789837, 1.591316]
args= 0 INTJ static_compile kernel median=2.7993 samples=[3.1696419999999996, 2.799279, 2.843015, 2.7790880000000002, 2.8156619999999997, 2.769628, 2.799248, 2.771549, 2.8191219999999997]
args= 0 INTJ runtime_shim kernel median=2.7516 samples=[2.91973, 2.724606, 2.802031, 2.713204, 2.764184, 2.702516, 2.7717310000000004, 2.698747, 2.751614]
args= 3 FFI packed nop           median=0.1442 samples=[0.145768, 0.143004, 0.14393999999999998, 0.144064, 0.145415, 0.14559, 0.146048, 0.144197, 0.142963]
args= 3 FFI typed nop            median=0.1507 samples=[0.151521, 0.150778, 0.14815799999999998, 0.153274, 0.14967, 0.150661, 0.15181299999999998, 0.15037799999999998, 0.148938]
args= 3 FFI empty kernel         median=3.3318 samples=[3.449281, 3.336387, 3.3175920000000003, 3.3572550000000003, 3.294596, 3.331835, 3.273964, 3.346408, 3.286344]
args= 3 INTJ static_compile kernel median=2.9819 samples=[3.059672, 2.964439, 3.0065340000000003, 2.963849, 2.989739, 2.953376, 3.014591, 2.9601460000000004, 2.981876]
args= 3 INTJ runtime_shim kernel median=2.9441 samples=[3.087965, 2.944057, 2.935194, 2.936741, 2.948502, 2.932553, 2.947295, 2.929339, 2.945433]
args= 5 FFI packed nop           median=0.1727 samples=[0.173328, 0.172981, 0.170844, 0.172686, 0.17299899999999999, 0.16969800000000002, 0.171072, 0.171791, 0.174564]
args= 5 FFI typed nop            median=0.1781 samples=[0.17977600000000002, 0.17899199999999998, 0.176509, 0.179167, 0.17719100000000002, 0.178857, 0.17807800000000001, 0.17703899999999997, 0.176255]
args= 5 FFI empty kernel         median=3.3809 samples=[3.468892, 3.390142, 3.350875, 3.4010949999999998, 3.3341, 3.397691, 3.335587, 3.3808719999999997, 3.3319490000000003]
args= 5 INTJ static_compile kernel median=2.9947 samples=[3.062728, 2.99268, 3.017549, 2.984067, 2.9997409999999998, 2.994719, 2.999727, 2.968578, 2.989022]
args= 5 INTJ runtime_shim kernel median=2.9881 samples=[3.04948, 2.9258960000000003, 3.007771, 2.9132260000000003, 2.995232, 2.904095, 2.988066, 2.894302, 3.000252]
args= 8 FFI packed nop           median=0.2052 samples=[0.20581, 0.202071, 0.203543, 0.20633500000000002, 0.206511, 0.204206, 0.205192, 0.205959, 0.204567]
args= 8 FFI typed nop            median=0.2110 samples=[0.212503, 0.210999, 0.21463, 0.21067599999999997, 0.20999, 0.208151, 0.211876, 0.214173, 0.210064]
args= 8 FFI empty kernel         median=3.4554 samples=[3.561935, 3.4307339999999997, 3.4685129999999997, 3.442184, 3.462132, 3.423235, 3.459817, 3.401102, 3.455419]
args= 8 INTJ static_compile kernel median=2.9953 samples=[3.147362, 3.0038020000000003, 2.991142, 2.999555, 2.985073, 2.998999, 2.9953339999999997, 2.972167, 2.981747]
args= 8 INTJ runtime_shim kernel median=3.0244 samples=[3.143568, 2.983708, 3.024386, 2.957094, 3.033394, 2.9537199999999997, 3.027703, 2.940645, 3.066201]
args=16 FFI packed nop           median=0.3043 samples=[0.31038600000000005, 0.304404, 0.304298, 0.30341199999999996, 0.30406900000000003, 0.305426, 0.305604, 0.301589, 0.303291]
args=16 FFI typed nop            median=0.3148 samples=[0.313747, 0.308636, 0.30938099999999996, 0.31593400000000005, 0.317142, 0.31649099999999997, 0.314816, 0.317288, 0.314698]
args=16 FFI empty kernel         median=3.7560 samples=[3.923259, 3.833281, 3.656169, 3.756801, 3.646501, 3.756034, 3.6477869999999997, 3.744924, 3.87864]
args=16 INTJ static_compile kernel median=3.4004 samples=[3.365664, 3.400447, 3.61302, 3.3702010000000002, 3.5751, 3.354768, 3.595375, 3.3695399999999998, 3.58199]
args=16 INTJ runtime_shim kernel median=3.3466 samples=[3.3465700000000003, 3.197837, 3.433845, 3.101673, 3.402661, 3.1141170000000002, 3.429059, 3.10182, 3.415607]
args=32 FFI packed nop           median=0.4830 samples=[0.487123, 0.48871499999999996, 0.477802, 0.482597, 0.482997, 0.48503500000000005, 0.489185, 0.48192, 0.482122]
args=32 FFI typed nop            median=0.4936 samples=[0.503568, 0.494878, 0.48989699999999997, 0.500825, 0.49271, 0.496495, 0.491969, 0.491047, 0.49356700000000003]
args=32 FFI empty kernel         median=4.2462 samples=[4.35196, 4.291459, 4.181756, 4.246173, 4.172317, 4.24987, 4.1726220000000005, 4.255433, 4.1874780000000005]
args=32 INTJ static_compile kernel median=3.8004 samples=[3.582895, 3.800423, 3.844912, 3.774756, 3.83894, 3.7906619999999998, 3.805958, 3.7708069999999996, 3.830149]
args=32 INTJ runtime_shim kernel median=3.6115 samples=[3.6068629999999997, 3.611487, 3.764219, 3.575222, 3.762641, 3.5600880000000004, 3.6490859999999996, 3.573522, 3.749087]
args=64 FFI packed nop           median=0.8407 samples=[0.8409160000000001, 0.817144, 0.853974, 0.849385, 0.837172, 0.840727, 0.853169, 0.815758, 0.8256789999999999]
args=64 FFI typed nop            median=0.8780 samples=[0.859213, 0.8809450000000001, 0.8779969999999999, 0.880667, 0.880283, 0.869352, 0.8522580000000001, 0.890648, 0.874282]
args=64 FFI empty kernel         median=5.3831 samples=[5.425694999999999, 5.467013, 5.302961, 5.394384, 5.287916, 5.383061, 5.28247, 5.397108, 5.2889219999999995]
args=64 INTJ static_compile kernel median=4.9205 samples=[4.952236, 4.943252, 4.909628, 4.942748, 4.918446, 4.930554, 4.920544, 4.898816, 4.920286]
args=64 INTJ runtime_shim kernel median=4.8488 samples=[4.831029, 4.873937, 4.8366869999999995, 4.88411, 4.842154000000001, 4.880515999999999, 4.861416, 4.848814, 4.815785]
elapsed_ns=10784035390
exit_status=0
ended=2026-10-14T16:30:35+08:00
```

#### `round1_hip_module.txt`

```
commit=bc8caf71f9323fb05dc2458f1e12ed8dd9ede5f8
started=2026-10-14T16:30:45+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/mnt/nvme2/jinpli/workspace/home/jinpli/development/benchmark taskset -c 0 /mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/venv/bin/python benchmarks/bench_hip_module_launch.py --iters 1000 --batches 9
hsaco_sha256=03409dde23b75ed425f9f7cc5341892b5f85c3c81874562ccaec070852969125 kernel=empty_kernel threads=256 shared=0 callback_count=1
iters=1000 batches=9 unit=us/call (host enqueue, no timed sync)
TVM FFI HIP HSACO         median=3.2197 samples=[3.3318090000000002, 3.2036759999999997, 3.224426, 3.315131, 3.28507, 3.197574, 3.192806, 3.2197109999999998, 3.2132069999999997]
Triton same HSACO         median=15.5273 samples=[15.757679, 15.541383, 15.537082, 15.720672, 15.527282999999999, 15.489199000000001, 15.449138000000001, 15.479453, 15.5101]
INTJ same function        median=3.0117 samples=[3.0431239999999997, 3.057651, 3.120117, 3.118924, 2.9907779999999997, 2.966549, 2.991505, 2.985992, 3.011717]
elapsed_ns=10144921628
exit_status=0
ended=2026-10-14T16:30:55+08:00
```

#### `round1_kernel_cache.txt`

```
commit=bc8caf71f9323fb05dc2458f1e12ed8dd9ede5f8
started=2026-10-14T16:30:55+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/mnt/nvme2/jinpli/workspace/home/jinpli/development/benchmark taskset -c 0 /mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/venv/bin/python -m pytest tests/test_kernel_cache.py -s -q

kernel cache, 1-word key, ns per operation

                      intj       tsl      absl
hash_only             1.36      1.35      1.36
hit/1                 1.57      2.03      1.49
hit/8                 1.59      2.10      4.36
hit/64                1.77      2.14      4.53
hit/512               2.29      2.90      5.30
hit_child             3.00      3.00      3.01
miss/1                1.27      1.81      1.38
miss/8                2.60      1.74      3.27
miss/64               2.76      1.74      3.54
miss/512              2.92      2.00      4.08

kernel cache, 2-word key, ns per operation

                      intj       tsl      absl
hash_only             1.42      1.42      1.42
hit/1                 1.86      2.56      2.00
hit/8                 1.79      2.56      5.04
hit/64                2.21      2.92      5.24
hit/512               3.08      4.03      6.56
hit_child             3.00      3.00      3.01
miss/1                1.38      2.32      1.46
miss/8                2.42      1.87      3.20
miss/64               3.13      2.10      3.89
miss/512              2.90      2.74      3.98

kernel cache, 5-word key, ns per operation

                      intj       tsl      absl
hash_only             3.58      3.58      3.58
hit/1                 3.00      3.06      2.52
hit/8                 3.20      3.41      9.47
hit/64                3.26      3.50      9.76
hit/512               4.52      4.79     11.17
hit_child             3.01      3.00      3.00
miss/1                1.40      2.10      1.99
miss/8                2.96      2.25      6.92
miss/64               2.84      1.97      6.72
miss/512              3.46      2.50      7.69
........
8 passed in 31.53s
elapsed_ns=32335938900
exit_status=0
ended=2026-10-14T16:31:27+08:00
```

#### `round1_launch_dynamic.txt`

```
commit=bc8caf71f9323fb05dc2458f1e12ed8dd9ede5f8
started=2026-10-14T16:31:38+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/mnt/nvme2/jinpli/workspace/home/jinpli/development/benchmark taskset -c 0 /mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/venv/bin/python benchmarks/bench_launch.py --dynamic --iters 20000 --batches 9
mode=dynamic; 20000 calls × 9 batches; median ns/call
               path     launch   spec_key
         lazy plain     3801.8       91.6
  2 dynamic options     3937.7       92.0
      int constexpr     3733.1       88.3
      str constexpr     3750.3       88.2
elapsed_ns=12721161119
exit_status=0
ended=2026-10-14T16:31:51+08:00
```

#### `round1_launch_gpu.txt`

```
commit=bc8caf71f9323fb05dc2458f1e12ed8dd9ede5f8
started=2026-10-14T16:29:25+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/mnt/nvme2/jinpli/workspace/home/jinpli/development/benchmark taskset -c 0 /mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/venv/bin/python benchmarks/bench_launch.py --iters 1000 --batches 9
mode=gpu; 1000 calls × 9 batches; median ns/call
               path    ns/call       Δ ns       Δ %   build ms   bind ms
           auto map     3268.4       +0.0      +0.0      18.25         -
   tensor annotated     3340.8      +72.4      +2.2       3.68         -
            generic     3261.2       -7.2      -0.2       2.91         -
 generic tensor ann     3206.2      -62.2      -1.9       2.87         -
        reduced key     3129.1     -139.4      -4.3       3.01         -
         verify off     3168.4     -100.1      -3.1       2.90         -
          verify on     3127.1     -141.4      -4.3       2.85         -
              baked     3126.1     -142.3      -4.4       2.98         -
       bound tensor     3092.8     -175.7      -5.4       2.82      0.03
      bound pointer     3043.1     -225.3      -6.9       2.79      0.03
   fixed device map     3142.1     -126.3      -3.9       2.76      0.02
fixed device no-map     3142.7     -125.7      -3.8       2.78      0.02
elapsed_ns=10280508285
exit_status=0
ended=2026-10-14T16:29:36+08:00
```

#### `round1_launch_host.txt`

```
commit=bc8caf71f9323fb05dc2458f1e12ed8dd9ede5f8
started=2026-10-14T16:29:36+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/mnt/nvme2/jinpli/workspace/home/jinpli/development/benchmark taskset -c 0 /mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/venv/bin/python benchmarks/bench_launch.py --no-gpu --iters 100000 --batches 9
mode=host-only; 100000 calls × 9 batches; median ns/call
               path    ns/call       Δ ns       Δ %   build ms   bind ms
           auto map       39.9       +0.0      +0.0       6.83         -
   tensor annotated       39.6       -0.3      -0.8       2.28         -
            generic       43.8       +3.9      +9.7       2.20         -
 generic tensor ann       43.8       +3.8      +9.6       2.14         -
        reduced key       40.7       +0.8      +2.0       2.12         -
         verify off       41.2       +1.3      +3.3       2.15         -
          verify on       43.1       +3.2      +8.1       2.25         -
              baked       37.5       -2.4      -6.0       2.17         -
       bound tensor       42.9       +3.0      +7.5       2.15      0.02
      bound pointer       41.1       +1.2      +3.0       2.11      0.02
   fixed device map       41.0       +1.1      +2.8       2.13      0.02
fixed device no-map       41.4       +1.5      +3.6       2.13      0.02
elapsed_ns=9471319881
exit_status=0
ended=2026-10-14T16:29:45+08:00
```

#### `round1_launch_last_key.txt`

```
commit=bc8caf71f9323fb05dc2458f1e12ed8dd9ede5f8
started=2026-10-14T16:30:05+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/mnt/nvme2/jinpli/workspace/home/jinpli/development/benchmark taskset -c 0 /mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/venv/bin/python benchmarks/bench_launch.py --last-key --iters 100000 --batches 9
mode=last-key; host-only; 200000 calls × 9 batches; median ns/call
count     pattern    ns/call
    4      repeat       32.0
    4   alternate       34.1
   16      repeat       61.1
   16   alternate       63.9
   32      repeat       98.4
   32   alternate      102.7
elapsed_ns=9692428919
exit_status=0
ended=2026-10-14T16:30:14+08:00
```

#### `round1_launch_readme.txt`

```
commit=bc8caf71f9323fb05dc2458f1e12ed8dd9ede5f8
started=2026-10-14T16:29:45+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/mnt/nvme2/jinpli/workspace/home/jinpli/development/benchmark taskset -c 0 /mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/venv/bin/python benchmarks/bench_launch.py --readme --iters 1000 --batches 9
mode=readme; 1000 calls × 9 batches; median
        path   triton us    intj us   speedup
   grid=(1,)       16.45       3.33      4.9x
   grid=(0,)       12.55       0.03    464.9x

torch_access_mode   decode ns    build s
     runtime_shim        89.0       0.02
   static_compile        95.2       0.00
      interpreter       921.0       0.00
elapsed_ns=10250813546
exit_status=0
ended=2026-10-14T16:29:55+08:00
```

#### `round1_launch_sweep.txt`

```
commit=bc8caf71f9323fb05dc2458f1e12ed8dd9ede5f8
started=2026-10-14T16:29:55+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/mnt/nvme2/jinpli/workspace/home/jinpli/development/benchmark taskset -c 0 /mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/venv/bin/python benchmarks/bench_launch.py --sweep --iters 100000 --batches 9
mode=sweep; host-only; 100000 calls × 9 batches; median ns/call
count   kind    ns/call
    4    int       38.1
    4 tensor       36.6
   16    int       66.5
   16 tensor       62.6
   32    int      105.7
   32 tensor      102.4
elapsed_ns=9372474666
exit_status=0
ended=2026-10-14T16:30:05+08:00
```

#### `round1_launch_tuned.txt`

```
commit=bc8caf71f9323fb05dc2458f1e12ed8dd9ede5f8
started=2026-10-14T16:31:27+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/mnt/nvme2/jinpli/workspace/home/jinpli/development/benchmark taskset -c 0 /mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/venv/bin/python benchmarks/bench_launch.py --tuned --iters 20000 --batches 9
 tuned hit:  3140.7 ns/launch
     plain:  3091.9 ns/launch
elapsed_ns=11032118948
exit_status=0
ended=2026-10-14T16:31:38+08:00
```

#### `round1_mix.txt`

```
commit=bc8caf71f9323fb05dc2458f1e12ed8dd9ede5f8
started=2026-10-14T16:31:51+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/mnt/nvme2/jinpli/workspace/home/jinpli/development/benchmark taskset -c 0 /mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/venv/bin/python benchmarks/bench_mix.py
intj 0.1.0 from /mnt/nvme2/jinpli/workspace/home/jinpli/development/intj/intj
python 3.12.3; triton 3.8.0; torch 2.14.0.dev20260804+rocm7.1
mode=mix; host-only; 40000 calls × 9 batches; median ns/call
count share ckind  nkind   pattern   ns/call  samples
    4     0     -    int    repeat      31.9  32.0 31.9 32.0 31.8 32.2 31.8 31.8 32.2 31.9
    4     0     -    int alternate      34.0  33.8 34.1 33.9 34.0 33.9 34.0 34.0 34.1 34.0
    4     0     - tensor    repeat      30.5  30.6 30.6 30.5 30.5 30.5 30.4 30.6 30.6 30.5
    4     0     - tensor alternate      33.1  33.1 32.9 33.1 33.1 33.0 33.1 33.9 32.7 32.8
    4     0     -  mixed    repeat      30.3  30.2 30.3 30.5 30.3 30.3 30.3 30.4 30.3 30.4
    4     0     -  mixed alternate      32.4  32.4 32.4 32.5 32.4 32.5 32.4 32.5 32.5 32.3
    4    25   int    int    repeat      31.1  31.0 31.1 31.2 30.8 31.1 31.1 31.3 31.2 31.0
    4    25   int    int alternate      33.2  33.4 33.2 33.2 33.1 33.4 33.4 33.1 33.3 33.2
    4    25   int tensor    repeat      30.1  30.0 30.1 30.1 30.0 30.0 30.1 30.1 30.2 30.2
    4    25   int tensor alternate      32.7  32.6 32.6 32.7 32.6 32.6 33.0 32.7 32.7 33.0
    4    25   int  mixed    repeat      29.5  29.6 29.5 29.5 29.4 29.6 29.5 29.5 29.5 29.4
    4    25   int  mixed alternate      31.8  31.8 31.9 31.7 31.8 32.0 31.6 31.7 32.0 32.0
    4    25   str    int    repeat      30.7  30.7 30.6 30.6 30.7 30.9 30.7 30.7 30.8 30.7
    4    25   str    int alternate      39.0  39.1 39.0 38.9 38.9 39.0 39.0 39.0 39.0 39.0
    4    25   str tensor    repeat      30.1  30.1 30.1 30.2 30.0 30.1 30.5 30.3 30.3 30.1
    4    25   str tensor alternate      37.8  37.7 37.9 37.8 37.8 37.8 40.4 40.1 37.6 37.8
    4    25   str  mixed    repeat      29.6  29.5 29.5 29.5 29.6 29.6 29.6 29.6 31.7 29.6
    4    25   str  mixed alternate      38.8  39.2 38.9 38.6 38.7 38.8 38.8 38.8 38.8 38.9
    4    25 dtype    int    repeat      30.8  30.9 30.9 30.7 30.7 30.8 31.1 30.8 30.8 30.7
    4    25 dtype    int alternate      38.0  38.0 38.1 37.9 38.1 37.9 37.9 38.0 38.1 37.9
    4    25 dtype tensor    repeat      30.0  30.0 30.1 30.1 30.1 30.0 30.0 30.2 30.0 29.9
    4    25 dtype tensor alternate      36.0  36.0 35.7 35.8 36.0 35.9 36.1 35.8 36.2 36.0
    4    25 dtype  mixed    repeat      29.6  29.5 29.6 29.6 29.6 29.6 29.6 29.8 29.8 29.7
    4    25 dtype  mixed alternate      35.8  35.7 36.2 35.8 35.7 35.6 35.8 35.9 35.7 36.1
    4    50   int    int    repeat      30.4  30.5 35.3 30.5 30.3 30.4 30.5 30.3 30.3 30.3
    4    50   int    int alternate      33.6  33.6 34.5 33.6 33.5 33.9 33.9 33.6 33.3 33.6
    4    50   int tensor    repeat      29.9  30.1 29.9 29.8 29.8 30.0 29.8 29.8 29.9 30.2
    4    50   int tensor alternate      33.5  33.5 33.6 33.6 33.4 33.5 33.4 33.3 33.5 33.6
    4    50   int  mixed    repeat      30.1  30.1 30.2 30.1 30.1 30.2 30.3 30.2 30.1 30.1
    4    50   int  mixed alternate      34.1  34.1 34.0 33.9 34.1 34.1 34.2 34.3 34.0 34.1
    4    50   str    int    repeat      30.1  30.1 30.2 30.2 30.2 30.0 30.0 30.0 30.1 30.1
    4    50   str    int alternate      39.7  39.7 39.9 40.7 39.9 39.8 39.7 39.6 39.6 39.6
    4    50   str tensor    repeat      29.8  29.9 29.8 30.0 29.7 29.7 29.8 30.4 30.1 29.8
    4    50   str tensor alternate      38.9  38.8 38.9 38.8 38.9 38.9 38.9 38.9 38.9 39.0
    4    50   str  mixed    repeat      30.1  30.1 30.1 30.0 30.3 30.3 29.9 30.1 29.9 30.3
    4    50   str  mixed alternate      39.0  38.9 39.0 39.2 39.1 38.9 38.8 39.1 39.1 39.0
    4    50 dtype    int    repeat      30.0  30.4 30.2 30.0 30.1 29.9 30.3 30.0 30.0 29.9
    4    50 dtype    int alternate      37.8  37.8 37.9 37.7 37.9 37.7 38.2 37.7 37.8 37.7
    4    50 dtype tensor    repeat      29.8  29.8 29.9 30.0 29.8 29.8 29.9 30.0 29.8 29.8
    4    50 dtype tensor alternate      36.6  36.6 36.6 36.9 36.8 36.7 36.6 36.5 36.5 36.8
    4    50 dtype  mixed    repeat      30.2  30.0 30.5 30.5 30.6 30.1 30.3 30.2 30.0 30.0
    4    50 dtype  mixed alternate      37.4  37.4 37.4 37.5 37.4 37.6 37.3 37.6 37.4 37.4
    4   100   int      -    repeat      28.6  28.6 28.4 28.5 28.6 28.6 28.5 28.6 28.6 28.4
    4   100   int      - alternate      33.7  33.7 33.6 33.6 33.7 34.0 34.2 33.5 33.9 33.8
    4   100   str      -    repeat      29.4  29.4 29.4 29.4 29.6 29.5 29.5 29.4 29.4 29.6
    4   100   str      - alternate      39.7  39.9 39.7 39.9 39.7 39.7 39.7 39.7 39.8 40.0
    4   100 dtype      -    repeat      29.5  29.7 29.5 29.4 29.7 29.3 29.4 29.4 29.5 29.5
    4   100 dtype      - alternate      38.2  38.0 38.3 38.1 38.1 38.1 38.2 38.5 38.2 38.2
   16     0     -    int    repeat      61.1  61.4 61.3 61.1 61.0 61.1 61.0 61.0 61.2 61.0
   16     0     -    int alternate      63.9  63.8 64.0 63.8 63.9 63.8 63.8 63.9 63.9 64.0
   16     0     - tensor    repeat      57.4  57.7 57.3 57.2 57.6 57.6 57.9 57.4 57.4 57.2
   16     0     - tensor alternate      60.7  60.9 60.7 61.0 60.7 60.8 60.7 60.7 60.9 60.7
   16     0     -  mixed    repeat      55.4  55.4 55.3 55.3 55.4 56.1 55.5 55.2 55.4 55.3
   16     0     -  mixed alternate      59.2  59.2 59.0 59.0 59.0 59.2 59.2 59.2 59.2 59.5
   16    25   int    int    repeat      56.3  56.5 56.1 56.0 56.2 56.5 56.8 56.3 56.5 56.3
   16    25   int    int alternate      62.4  62.2 62.3 62.2 62.4 62.4 62.4 62.3 62.5 62.5
   16    25   int tensor    repeat      55.0  54.9 55.0 54.9 54.9 55.0 55.0 55.6 55.1 55.1
   16    25   int tensor alternate      61.9  61.8 62.3 62.4 61.9 61.9 61.9 61.7 61.7 61.7
   16    25   int  mixed    repeat      52.1  52.1 52.1 52.0 52.1 52.2 52.7 52.0 51.9 51.9
   16    25   int  mixed alternate      58.3  58.0 58.2 58.3 58.6 58.3 58.3 58.3 58.5 58.2
   16    25   str    int    repeat      57.1  57.0 57.5 57.2 57.3 56.8 57.0 57.0 57.1 57.5
   16    25   str    int alternate      68.5  69.1 68.5 68.3 68.3 68.2 68.4 69.1 68.5 68.8
   16    25   str tensor    repeat      55.4  55.6 55.3 55.4 55.6 55.4 55.3 55.1 55.2 55.4
   16    25   str tensor alternate      66.8  66.7 66.7 66.7 66.9 67.4 66.7 67.4 66.8 66.8
   16    25   str  mixed    repeat      53.0  52.7 52.8 53.1 53.0 53.0 53.0 53.0 53.1 53.6
   16    25   str  mixed alternate      63.9  64.0 63.8 63.7 63.9 64.0 64.4 63.7 63.8 64.0
   16    25 dtype    int    repeat      57.2  57.3 57.2 57.1 56.4 56.6 57.3 57.6 57.2 58.0
   16    25 dtype    int alternate      66.4  66.6 66.3 66.6 66.4 66.4 66.3 66.5 66.4 66.6
   16    25 dtype tensor    repeat      55.1  55.7 55.1 55.0 55.0 55.0 55.1 55.2 55.1 55.1
   16    25 dtype tensor alternate      65.9  66.0 65.7 65.6 65.9 66.0 66.1 66.2 65.5 65.8
   16    25 dtype  mixed    repeat      53.0  52.8 53.0 53.0 53.1 53.2 53.0 53.0 53.1 53.1
   16    25 dtype  mixed alternate      63.8  64.0 63.8 63.8 63.8 64.0 63.9 63.8 63.7 64.3
   16    50   int    int    repeat      56.1  56.3 55.9 56.2 56.0 56.1 55.9 56.5 56.3 56.1
   16    50   int    int alternate      66.9  66.9 66.8 66.9 66.9 66.8 67.0 66.9 66.8 67.1
   16    50   int tensor    repeat      55.3  55.0 55.0 55.3 55.1 56.6 55.4 55.3 55.3 55.1
   16    50   int tensor alternate      65.9  66.4 65.8 65.9 66.1 65.7 66.1 65.8 65.8 66.1
   16    50   int  mixed    repeat      54.0  54.1 54.0 54.0 54.0 54.1 54.0 53.9 54.1 54.2
   16    50   int  mixed alternate      64.5  64.6 64.5 64.5 64.6 64.5 64.5 64.6 64.5 64.5
   16    50   str    int    repeat      56.0  56.1 56.1 56.3 56.0 56.0 55.8 56.1 56.0 55.9
   16    50   str    int alternate      73.6  73.6 73.6 73.6 73.6 73.6 73.8 74.1 73.7 73.8
   16    50   str tensor    repeat      55.2  55.2 55.0 59.9 55.4 55.2 55.0 55.0 57.0 55.0
   16    50   str tensor alternate      71.6  71.6 71.6 71.6 71.2 71.9 71.3 71.6 71.5 71.4
   16    50   str  mixed    repeat      54.2  54.2 53.9 54.4 54.2 54.3 54.1 54.2 53.9 54.1
   16    50   str  mixed alternate      70.4  70.2 70.1 70.6 70.5 70.2 70.4 70.7 70.6 70.3
   16    50 dtype    int    repeat      56.2  56.3 56.1 56.5 56.3 56.2 56.3 56.1 56.2 56.2
   16    50 dtype    int alternate      70.8  71.0 70.9 70.8 70.9 70.8 70.8 70.8 70.8 71.0
   16    50 dtype tensor    repeat      54.9  54.9 54.6 55.6 55.2 55.1 54.8 54.9 54.7 54.9
   16    50 dtype tensor alternate      70.0  69.8 69.9 70.0 69.8 70.0 70.0 69.8 70.0 70.5
   16    50 dtype  mixed    repeat      54.2  54.2 54.1 54.5 54.2 54.1 54.1 54.0 54.3 54.3
   16    50 dtype  mixed alternate      68.2  68.2 68.3 68.3 68.2 68.2 68.4 68.5 68.0 68.0
   16   100   int      -    repeat      48.9  49.4 48.7 48.8 48.8 48.5 48.9 48.9 49.3 49.0
   16   100   int      - alternate      71.0  71.5 70.9 70.9 71.3 71.0 71.3 71.3 71.0 70.9
   16   100   str      -    repeat      50.1  49.8 51.5 49.7 50.7 50.0 50.1 50.0 52.6 50.3
   16   100   str      - alternate      78.7  79.2 79.1 78.4 79.0 78.4 78.7 78.7 79.0 78.5
   16   100 dtype      -    repeat      49.8  50.0 49.6 49.6 49.8 49.6 49.8 49.9 49.8 50.0
   16   100 dtype      - alternate      73.5  73.6 73.4 73.3 73.5 73.4 73.7 73.7 73.4 73.5
   32     0     -    int    repeat      97.6  100.0 97.5 99.0 97.1 97.5 98.9 97.6 101.8 97.1
   32     0     -    int alternate     101.6  103.2 102.1 101.8 101.6 101.6 101.6 101.6 101.6 101.6
   32     0     - tensor    repeat     100.8  109.8 104.6 102.3 100.7 103.0 100.8 99.3 99.7 97.2
   32     0     - tensor alternate     105.3  113.1 105.3 110.4 113.2 108.2 104.1 102.9 104.8 103.3
   32     0     -  mixed    repeat      88.2  88.2 88.7 88.1 88.2 88.1 88.3 88.2 88.2 88.2
   32     0     -  mixed alternate      93.4  93.2 93.2 93.5 93.6 93.0 93.8 93.7 93.1 93.4
   32    25   int    int    repeat      88.3  88.5 88.2 90.4 88.0 88.2 89.4 88.2 88.3 89.4
   32    25   int    int alternate     101.6  101.7 101.8 101.8 102.0 101.4 101.5 101.5 101.6 101.5
   32    25   int tensor    repeat      91.1  91.7 89.8 90.7 91.5 91.1 91.0 89.8 91.6 91.7
   32    25   int tensor alternate     103.0  106.3 104.1 105.3 103.0 103.1 102.4 102.1 102.1 101.9
   32    25   int  mixed    repeat      81.7  81.7 81.8 81.7 81.5 81.8 81.8 81.5 81.6 81.9
   32    25   int  mixed alternate      94.1  94.2 94.1 94.1 94.2 94.4 94.1 94.0 93.9 94.0
   32    25   str    int    repeat      89.3  89.1 89.2 89.8 89.0 89.3 89.3 89.3 89.3 89.2
   32    25   str    int alternate     107.7  107.6 107.7 107.6 107.7 107.4 109.6 108.6 108.4 107.7
   32    25   str tensor    repeat      98.1  96.6 98.3 99.3 97.0 96.2 98.1 98.0 101.8 99.3
   32    25   str tensor alternate     117.6  133.1 125.3 123.2 118.5 117.6 115.4 116.0 116.6 114.8
   32    25   str  mixed    repeat      83.0  83.2 83.6 83.0 82.4 82.9 83.3 82.7 83.4 83.0
   32    25   str  mixed alternate     102.1  102.0 101.8 102.6 102.3 102.7 101.9 101.9 103.0 102.1
   32    25 dtype    int    repeat      89.2  93.6 89.2 89.1 90.1 89.1 89.3 89.4 89.2 89.2
   32    25 dtype    int alternate     105.7  105.6 105.7 105.8 105.9 105.9 105.6 105.4 105.7 105.9
   32    25 dtype tensor    repeat      98.6  97.9 98.3 98.9 98.8 98.6 99.0 98.0 97.2 98.8
   32    25 dtype tensor alternate     110.1  116.5 111.3 109.6 109.7 110.1 110.5 110.0 109.7 110.4
   32    25 dtype  mixed    repeat      84.8  86.4 85.5 85.0 84.5 85.5 84.8 84.6 83.7 83.9
   32    25 dtype  mixed alternate      98.8  98.9 98.8 98.9 98.7 98.8 98.7 98.7 99.1 98.9
   32    50   int    int    repeat      90.9  92.1 91.6 89.8 91.5 90.9 90.8 93.1 90.2 90.2
   32    50   int    int alternate     115.7  116.2 116.5 115.0 115.7 116.2 116.4 115.0 112.6 112.2
   32    50   int tensor    repeat      86.0  86.0 86.0 86.0 86.0 86.0 85.9 86.2 86.3 86.1
   32    50   int tensor alternate     107.3  107.5 107.3 107.3 107.3 107.3 107.6 107.4 107.3 107.3
   32    50   int  mixed    repeat      83.7  83.8 83.5 83.9 83.7 83.6 83.9 83.7 83.9 83.5
   32    50   int  mixed alternate     104.9  105.0 104.8 104.9 104.9 105.2 104.8 104.9 104.8 105.1
   32    50   str    int    repeat      88.2  88.2 88.2 88.1 88.5 88.2 88.2 88.4 88.0 88.1
   32    50   str    int alternate     116.4  116.7 116.4 116.1 116.2 116.8 116.2 116.9 116.4 116.3
   32    50   str tensor    repeat      94.5  93.9 95.5 95.6 94.2 94.0 99.9 96.4 94.5 94.0
   32    50   str tensor alternate     117.8  119.1 119.0 118.6 117.6 117.3 117.8 118.2 117.2 117.6
   32    50   str  mixed    repeat      85.1  85.6 85.3 85.1 85.2 84.9 85.1 85.3 85.0 84.8
   32    50   str  mixed alternate     111.7  112.0 111.8 111.5 111.6 111.7 111.7 111.8 111.7 111.7
   32    50 dtype    int    repeat      88.0  88.0 88.0 88.1 88.1 88.3 87.9 87.8 87.8 88.5
   32    50 dtype    int alternate     114.9  115.1 114.7 114.9 115.4 114.6 114.6 115.2 114.9 114.7
   32    50 dtype tensor    repeat      94.1  95.1 94.4 97.9 93.9 94.0 93.5 94.0 94.1 94.5
   32    50 dtype tensor alternate     117.5  118.4 117.3 117.5 118.2 118.1 117.9 117.2 117.0 116.9
   32    50 dtype  mixed    repeat      85.3  85.1 85.3 85.0 85.5 85.2 85.2 85.6 85.4 85.6
   32    50 dtype  mixed alternate     110.7  110.7 110.8 110.7 110.8 112.1 110.4 111.2 110.6 110.4
   32   100   int      -    repeat      71.0  70.7 72.2 70.9 73.6 71.0 71.3 70.9 70.7 71.0
   32   100   int      - alternate     113.5  113.5 113.3 113.6 113.4 113.2 113.1 113.5 113.5 113.5
   32   100   str      -    repeat      76.4  76.9 76.4 76.4 76.4 76.3 76.5 76.4 76.8 76.4
   32   100   str      - alternate     124.6  125.2 124.6 123.9 124.4 123.2 124.6 128.8 123.3 127.8
   32   100 dtype      -    repeat      76.7  76.7 76.7 76.6 77.0 76.8 76.8 76.7 76.4 76.9
   32   100 dtype      - alternate     120.3  120.4 120.1 120.1 120.3 120.3 120.2 120.8 120.3 120.5
elapsed_ns=12360543660
exit_status=0
ended=2026-10-14T16:32:03+08:00
```

#### `round2_ffi_compare.txt`

```
commit=bc8caf71f9323fb05dc2458f1e12ed8dd9ede5f8
started=2026-10-14T16:32:52+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/mnt/nvme2/jinpli/workspace/home/jinpli/development/benchmark taskset -c 0 /mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/venv/bin/python benchmarks/bench_ffi_compare.py --iters 1000 --batches 9
iters=1000 batches=9 unit=us/call (host-only, no timed sync)
FFI packed nop             median=0.1448 samples=[0.151863, 0.146457, 0.144313, 0.145793, 0.144779, 0.144986, 0.144802, 0.143823, 0.143683]
FFI typed nop              median=0.1512 samples=[0.151155, 0.151364, 0.149861, 0.154784, 0.14890299999999998, 0.15523699999999999, 0.147277, 0.152225, 0.150275]
FFI empty kernel           median=3.3663 samples=[3.829049, 3.554051, 3.6456660000000003, 3.3663380000000003, 3.4105459999999996, 3.309605, 3.3330300000000004, 3.312649, 3.3268519999999997]
INTJ empty kernel          median=2.9944 samples=[3.121437, 3.437063, 3.1982440000000003, 3.016991, 2.994449, 2.93016, 2.98066, 2.924436, 2.956403]
INTJ fixed-device kernel   median=2.9704 samples=[3.026464, 3.162105, 3.169512, 3.024692, 2.9664859999999997, 2.942843, 2.970433, 2.9476430000000002, 2.954797]
FFI packed nop mixed       median=0.1717 samples=[0.17045, 0.17119399999999999, 0.171929, 0.174214, 0.172599, 0.171703, 0.171475, 0.17213399999999998, 0.171096]
FFI typed nop mixed        median=0.1764 samples=[0.17269800000000002, 0.179886, 0.175953, 0.177504, 0.175277, 0.17713800000000002, 0.17643999999999999, 0.177881, 0.174183]
FFI mixed kernel           median=3.3748 samples=[3.4836810000000002, 3.52948, 3.446139, 3.424742, 3.3748139999999998, 3.3378870000000003, 3.357814, 3.345214, 3.3545920000000002]
INTJ mixed kernel          median=2.9779 samples=[3.146031, 3.132706, 3.04617, 2.977864, 2.977603, 2.877576, 2.963382, 2.896545, 2.982716]
INTJ fixed mixed kernel    median=2.9993 samples=[3.154966, 3.146134, 3.089299, 3.027359, 2.9993339999999997, 2.923221, 2.994532, 2.919205, 2.994897]
elapsed_ns=10157127160
exit_status=0
ended=2026-10-14T16:33:02+08:00
```

#### `round2_ffi_paths.txt`

```
commit=bc8caf71f9323fb05dc2458f1e12ed8dd9ede5f8
started=2026-10-14T16:33:13+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/mnt/nvme2/jinpli/workspace/home/jinpli/development/benchmark taskset -c 0 /mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/venv/bin/python benchmarks/bench_intj_ffi_paths.py --mode all --iters 1000 --batches 9
iters=1000 batches=9 unit=ns/call; no_gpu=True
mode=callback
callback_count=9100 (100 warmup + 9000 unique misses)
INTJ cold compile callback + cache  median_ns=675.1 samples_ns=[720.86, 709.25, 656.52, 763.14, 647.266, 656.335, 665.505, 675.11, 1050.927]
INTJ hot call (no callback)         median_ns=34.6 samples_ns=[36.666, 35.266, 34.625, 34.892, 34.552, 34.492, 34.528, 34.579, 34.519]
INTJ 3-tensor host-only nop         median_ns=32.2 samples_ns=[34.653, 32.436, 32.222, 32.114, 32.175, 32.143, 32.083, 32.301, 32.157]
mode=kwargs
INTJ direct positional              median_ns=63.6 samples_ns=[65.898, 63.597, 63.545, 63.865, 63.287, 63.973, 63.935, 63.283, 63.629]
INTJ adapter positional             median_ns=90.4 samples_ns=[90.403, 89.523, 90.686, 88.954, 91.595, 90.181, 90.406, 91.781, 91.865]
INTJ adapter kwargs                 median_ns=108.2 samples_ns=[107.693, 107.56, 110.909, 108.15, 110.775, 107.231, 109.923, 111.698, 107.017]
INTJ adapter defaults               median_ns=90.3 samples_ns=[90.262, 89.191, 89.656, 89.297, 90.326, 92.509, 89.721, 92.072, 91.637]
INTJ FFI wrapper positional         median_ns=104.3 samples_ns=[104.343, 103.419, 104.175, 104.024, 105.121, 103.828, 120.941, 105.718, 106.434]
INTJ FFI wrapper kwargs             median_ns=128.9 samples_ns=[128.913, 128.86, 128.879, 129.635, 129.335, 143.57, 130.028, 128.617, 127.883]
INTJ native kwargs: unsupported (verified TypeError)
mode=dataclass
INTJ pair direct                    median_ns=68.5 samples_ns=[71.349, 69.102, 68.496, 68.63, 68.558, 67.605, 63.321, 63.306, 63.351]
INTJ pair prebuilt *tuple           median_ns=136.1 samples_ns=[137.074, 136.061, 134.067, 135.975, 134.981, 143.701, 136.454, 135.138, 137.403]
FFI unpack Pair only                median_ns=163.9 samples_ns=[165.415, 163.917, 168.187, 164.853, 162.936, 162.241, 163.032, 163.563, 166.68]
INTJ pair manual unpack             median_ns=160.6 samples_ns=[161.445, 160.288, 158.753, 160.608, 159.524, 158.043, 172.243, 164.57, 165.093]
INTJ pair FFI unpack                median_ns=309.4 samples_ns=[311.756, 313.477, 307.7, 309.393, 311.152, 307.788, 309.123, 312.753, 309.398]
INTJ pair stdlib astuple            median_ns=1139.5 samples_ns=[1224.544, 1184.92, 1141.127, 1139.5, 1134.778, 1130.052, 1131.814, 1138.895, 1152.768]
INTJ config direct                  median_ns=77.4 samples_ns=[78.768, 78.389, 77.717, 81.211, 77.428, 77.253, 76.865, 76.807, 77.391]
INTJ config FFI unpack              median_ns=322.2 samples_ns=[321.034, 323.931, 320.393, 319.944, 322.641, 322.24, 320.538, 325.857, 326.842]
INTJ native dataclass unpack: unsupported (verified TypeError)
elapsed_ns=9363567248
exit_status=0
ended=2026-10-14T16:33:23+08:00
```

#### `round2_ffi_sweep.txt`

```
commit=bc8caf71f9323fb05dc2458f1e12ed8dd9ede5f8
started=2026-10-14T16:33:02+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/mnt/nvme2/jinpli/workspace/home/jinpli/development/benchmark taskset -c 0 /mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/venv/bin/python benchmarks/bench_ffi_compare.py --sweep --iters 1000 --batches 9
counts=[0, 3, 5, 8, 16, 32, 64] iters=1000 batches=9 unit=us/call (host-only, no timed sync); INTJ device-bound, no specialization key
args= 0 FFI packed nop           median=0.1151 samples=[0.118616, 0.115606, 0.114229, 0.117785, 0.11698099999999999, 0.114593, 0.115113, 0.112997, 0.11452]
args= 0 FFI typed nop            median=0.1192 samples=[0.120448, 0.119789, 0.117148, 0.121693, 0.118563, 0.12042900000000001, 0.11923399999999999, 0.119211, 0.11599]
args= 0 FFI empty kernel         median=1.6332 samples=[1.536359, 1.89533, 1.6332339999999999, 1.7723959999999999, 1.587955, 1.890603, 1.597553, 1.892854, 1.612855]
args= 0 INTJ static_compile kernel median=2.7949 samples=[3.1970970000000003, 2.7208930000000002, 2.816202, 2.780907, 2.803201, 2.710978, 2.7949479999999998, 2.873886, 2.7863580000000003]
args= 0 INTJ runtime_shim kernel median=2.7155 samples=[2.922855, 2.714334, 2.763799, 2.6877310000000003, 2.7558789999999997, 2.696837, 2.715486, 2.690698, 2.727217]
args= 3 FFI packed nop           median=0.1451 samples=[0.146929, 0.145427, 0.146148, 0.14240799999999998, 0.145077, 0.144793, 0.14771299999999998, 0.14460699999999999, 0.14488800000000002]
args= 3 FFI typed nop            median=0.1515 samples=[0.151416, 0.15332099999999999, 0.15043700000000002, 0.150917, 0.14732699999999999, 0.151718, 0.151801, 0.153518, 0.151485]
args= 3 FFI empty kernel         median=3.3309 samples=[3.444496, 3.3475349999999997, 3.267728, 3.3871689999999997, 3.312738, 3.3308910000000003, 3.2369470000000002, 3.331869, 3.245436]
args= 3 INTJ static_compile kernel median=3.0364 samples=[3.062355, 2.96431, 3.055987, 2.984376, 2.965843, 3.1884639999999997, 3.0364299999999997, 2.9524470000000003, 3.0516970000000003]
args= 3 INTJ runtime_shim kernel median=2.9312 samples=[3.0633939999999997, 2.9589969999999997, 2.9731959999999997, 2.838422, 2.915142, 2.942902, 2.9311570000000002, 2.929078, 2.9299]
args= 5 FFI packed nop           median=0.1712 samples=[0.173124, 0.170571, 0.17322200000000001, 0.17315, 0.171234, 0.170982, 0.170578, 0.172817, 0.17047700000000002]
args= 5 FFI typed nop            median=0.1784 samples=[0.17769900000000002, 0.179228, 0.17793799999999999, 0.17835900000000002, 0.175485, 0.18009, 0.17909999999999998, 0.181092, 0.178273]
args= 5 FFI empty kernel         median=3.3944 samples=[3.488785, 3.40198, 3.3943429999999997, 3.405312, 3.348883, 3.3923110000000003, 3.606688, 3.394407, 3.340038]
args= 5 INTJ static_compile kernel median=2.9898 samples=[3.063287, 2.986521, 3.023422, 2.957828, 2.973279, 3.0212269999999997, 2.989848, 2.958817, 2.99778]
args= 5 INTJ runtime_shim kernel median=2.9876 samples=[3.055341, 2.9727020000000004, 3.224373, 2.920195, 2.982488, 3.206402, 3.0046720000000002, 2.919163, 2.9875749999999996]
args= 8 FFI packed nop           median=0.2061 samples=[0.207034, 0.203902, 0.20715899999999998, 0.204763, 0.20497200000000002, 0.20605099999999998, 0.207227, 0.20441700000000002, 0.20615199999999997]
args= 8 FFI typed nop            median=0.2131 samples=[0.21381299999999998, 0.211978, 0.209219, 0.214048, 0.21013300000000001, 0.211894, 0.213492, 0.213061, 0.213161]
args= 8 FFI empty kernel         median=3.4467 samples=[3.5728519999999997, 3.454808, 3.467603, 3.440448, 3.4263820000000003, 3.414316, 3.4541619999999997, 3.420596, 3.446676]
args= 8 INTJ static_compile kernel median=3.0952 samples=[3.127427, 2.989806, 3.1197779999999997, 3.0553879999999998, 3.1302849999999998, 3.0503449999999996, 3.095163, 3.037793, 3.09922]
args= 8 INTJ runtime_shim kernel median=3.0096 samples=[3.1311799999999996, 2.988255, 3.028188, 2.951467, 3.080154, 2.932114, 3.047942, 2.9291329999999998, 3.0096100000000003]
args=16 FFI packed nop           median=0.3052 samples=[0.30524799999999996, 0.30430900000000005, 0.31049400000000005, 0.304911, 0.303659, 0.30600299999999997, 0.310444, 0.305053, 0.30744]
args=16 FFI typed nop            median=0.3149 samples=[0.316791, 0.316793, 0.314306, 0.314861, 0.314791, 0.316204, 0.316464, 0.313261, 0.309191]
args=16 FFI empty kernel         median=3.7452 samples=[3.8332319999999998, 3.8084520000000004, 3.696243, 3.780576, 3.679751, 3.8359479999999997, 3.649384, 3.745174, 3.639631]
args=16 INTJ static_compile kernel median=3.3693 samples=[3.348717, 3.3984490000000003, 3.598828, 3.363711, 3.3693139999999997, 3.357094, 3.583002, 3.345158, 3.592392]
args=16 INTJ runtime_shim kernel median=3.3710 samples=[3.3710169999999997, 3.184518, 3.4588850000000004, 3.128353, 3.540758, 3.127736, 3.415445, 3.11048, 3.42217]
args=32 FFI packed nop           median=0.4810 samples=[0.486668, 0.486945, 0.48098, 0.479763, 0.480551, 0.48479099999999997, 0.479721, 0.4785, 0.486604]
args=32 FFI typed nop            median=0.4981 samples=[0.500622, 0.501754, 0.486947, 0.510645, 0.487027, 0.497712, 0.494553, 0.503219, 0.498127]
args=32 FFI empty kernel         median=4.2628 samples=[4.350055, 4.3263549999999995, 4.206047, 4.262829, 4.183388, 4.262073999999999, 4.312564999999999, 4.268167999999999, 4.191349]
args=32 INTJ static_compile kernel median=3.7965 samples=[3.633875, 3.79652, 3.8370670000000002, 3.748933, 3.803058, 3.821888, 3.807254, 3.740313, 3.7697979999999998]
args=32 INTJ runtime_shim kernel median=3.6259 samples=[3.618975, 3.625922, 3.784612, 3.566387, 3.758642, 3.5684989999999996, 3.810768, 3.586557, 3.765512]
args=64 FFI packed nop           median=0.8459 samples=[0.855893, 0.845822, 0.8710789999999999, 0.842653, 0.842712, 0.8546180000000001, 0.847088, 0.841341, 0.845923]
args=64 FFI typed nop            median=0.8807 samples=[0.886242, 0.883049, 0.870291, 0.88597, 0.868722, 0.8806660000000001, 0.8682000000000001, 0.8706280000000001, 0.88871]
args=64 FFI empty kernel         median=5.3837 samples=[5.472384, 5.507162999999999, 5.3382879999999995, 5.406766999999999, 5.3214570000000005, 5.394167, 5.303046, 5.383724, 5.331729999999999]
args=64 INTJ static_compile kernel median=4.9000 samples=[4.943037, 4.95874, 4.891889, 4.886686, 4.929867, 4.899965, 4.909272, 4.896082000000001, 4.883402]
args=64 INTJ runtime_shim kernel median=4.8522 samples=[4.814198, 4.90804, 4.869205, 4.873396, 4.846587, 4.8522430000000005, 4.83276, 4.871658, 4.814962]
elapsed_ns=10773519663
exit_status=0
ended=2026-10-14T16:33:13+08:00
```

#### `round2_hip_module.txt`

```
commit=bc8caf71f9323fb05dc2458f1e12ed8dd9ede5f8
started=2026-10-14T16:33:23+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/mnt/nvme2/jinpli/workspace/home/jinpli/development/benchmark taskset -c 0 /mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/venv/bin/python benchmarks/bench_hip_module_launch.py --iters 1000 --batches 9
hsaco_sha256=03409dde23b75ed425f9f7cc5341892b5f85c3c81874562ccaec070852969125 kernel=empty_kernel threads=256 shared=0 callback_count=1
iters=1000 batches=9 unit=us/call (host enqueue, no timed sync)
TVM FFI HIP HSACO         median=3.2040 samples=[3.285011, 3.192022, 3.204029, 3.2826869999999997, 3.239671, 3.20928, 3.174102, 3.1774050000000003, 3.1845079999999997]
Triton same HSACO         median=15.6830 samples=[15.731713, 15.73791, 15.682962999999999, 15.850156, 15.655918, 15.729681000000001, 15.678737, 15.628245000000001, 15.601764]
INTJ same function        median=2.9703 samples=[3.010047, 3.0200859999999996, 3.1037109999999997, 3.100588, 2.9702640000000002, 2.925377, 2.959598, 2.954796, 2.9570100000000004]
elapsed_ns=10139474786
exit_status=0
ended=2026-10-14T16:33:33+08:00
```

#### `round2_kernel_cache.txt`

```
commit=bc8caf71f9323fb05dc2458f1e12ed8dd9ede5f8
started=2026-10-14T16:33:33+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/mnt/nvme2/jinpli/workspace/home/jinpli/development/benchmark taskset -c 0 /mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/venv/bin/python -m pytest tests/test_kernel_cache.py -s -q

kernel cache, 1-word key, ns per operation

                      intj       tsl      absl
hash_only             1.36      1.35      1.35
hit/1                 1.62      2.04      1.46
hit/8                 1.67      2.12      4.38
hit/64                1.61      2.13      4.54
hit/512               2.30      2.91      5.31
hit_child             3.00      3.00      3.00
miss/1                1.28      1.86      1.38
miss/8                2.72      1.75      3.30
miss/64               2.77      1.75      3.51
miss/512              2.94      2.00      4.07

kernel cache, 2-word key, ns per operation

                      intj       tsl      absl
hash_only             1.42      1.42      1.41
hit/1                 1.82      2.57      2.02
hit/8                 1.82      2.57      5.05
hit/64                2.20      2.92      5.22
hit/512               3.09      4.03      6.37
hit_child             3.00      3.00      3.01
miss/1                1.38      2.31      1.45
miss/8                2.41      1.86      3.21
miss/64               3.15      2.10      3.88
miss/512              2.90      2.74      4.01

kernel cache, 5-word key, ns per operation

                      intj       tsl      absl
hash_only             3.59      3.58      3.59
hit/1                 3.00      3.06      2.50
hit/8                 3.20      3.42      9.49
hit/64                3.26      3.50      9.78
hit/512               4.52      4.77     13.23
hit_child             3.01      3.01      3.00
miss/1                1.38      2.10      1.95
miss/8                2.97      2.24      6.93
miss/64               2.86      1.98      6.72
miss/512              3.52      2.50      9.59
........
8 passed in 32.36s
elapsed_ns=33205921041
exit_status=0
ended=2026-10-14T16:34:06+08:00
```

#### `round2_launch_dynamic.txt`

```
commit=bc8caf71f9323fb05dc2458f1e12ed8dd9ede5f8
started=2026-10-14T16:34:17+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/mnt/nvme2/jinpli/workspace/home/jinpli/development/benchmark taskset -c 0 /mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/venv/bin/python benchmarks/bench_launch.py --dynamic --iters 20000 --batches 9
mode=dynamic; 20000 calls × 9 batches; median ns/call
               path     launch   spec_key
         lazy plain     3834.6       93.0
  2 dynamic options     3803.2       93.2
      int constexpr     3766.6       88.2
      str constexpr     3783.1       88.6
elapsed_ns=12804313477
exit_status=0
ended=2026-10-14T16:34:30+08:00
```

#### `round2_launch_gpu.txt`

```
commit=bc8caf71f9323fb05dc2458f1e12ed8dd9ede5f8
started=2026-10-14T16:32:03+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/mnt/nvme2/jinpli/workspace/home/jinpli/development/benchmark taskset -c 0 /mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/venv/bin/python benchmarks/bench_launch.py --iters 1000 --batches 9
mode=gpu; 1000 calls × 9 batches; median ns/call
               path    ns/call       Δ ns       Δ %   build ms   bind ms
           auto map     3289.7       +0.0      +0.0      18.34         -
   tensor annotated     3320.3      +30.6      +0.9       3.66         -
            generic     3219.6      -70.1      -2.1       2.93         -
 generic tensor ann     3178.1     -111.7      -3.4       2.88         -
        reduced key     3126.5     -163.3      -5.0       2.98         -
         verify off     3136.4     -153.3      -4.7       2.88         -
          verify on     3122.2     -167.6      -5.1       2.90         -
              baked     3107.9     -181.8      -5.5       2.90         -
       bound tensor     3144.6     -145.2      -4.4       2.81      0.03
      bound pointer     3130.5     -159.3      -4.8       2.80      0.03
   fixed device map     3128.6     -161.1      -4.9       2.73      0.02
fixed device no-map     3129.6     -160.2      -4.9       2.81      0.02
elapsed_ns=10272492956
exit_status=0
ended=2026-10-14T16:32:14+08:00
```

#### `round2_launch_host.txt`

```
commit=bc8caf71f9323fb05dc2458f1e12ed8dd9ede5f8
started=2026-10-14T16:32:14+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/mnt/nvme2/jinpli/workspace/home/jinpli/development/benchmark taskset -c 0 /mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/venv/bin/python benchmarks/bench_launch.py --no-gpu --iters 100000 --batches 9
mode=host-only; 100000 calls × 9 batches; median ns/call
               path    ns/call       Δ ns       Δ %   build ms   bind ms
           auto map       39.8       +0.0      +0.0       6.79         -
   tensor annotated       39.5       -0.4      -0.9       2.27         -
            generic       43.7       +3.9      +9.8       2.15         -
 generic tensor ann       43.7       +3.9      +9.8       2.16         -
        reduced key       40.8       +0.9      +2.4       2.20         -
         verify off       41.2       +1.4      +3.5       2.19         -
          verify on       41.7       +1.9      +4.8       2.26         -
              baked       37.5       -2.4      -5.9       2.20         -
       bound tensor       42.9       +3.0      +7.6       2.18      0.02
      bound pointer       41.4       +1.6      +4.0       2.16      0.02
   fixed device map       40.7       +0.8      +2.1       2.11      0.02
fixed device no-map       41.1       +1.2      +3.0       2.23      0.02
elapsed_ns=9446180106
exit_status=0
ended=2026-10-14T16:32:23+08:00
```

#### `round2_launch_last_key.txt`

```
commit=bc8caf71f9323fb05dc2458f1e12ed8dd9ede5f8
started=2026-10-14T16:32:43+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/mnt/nvme2/jinpli/workspace/home/jinpli/development/benchmark taskset -c 0 /mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/venv/bin/python benchmarks/bench_launch.py --last-key --iters 100000 --batches 9
mode=last-key; host-only; 200000 calls × 9 batches; median ns/call
count     pattern    ns/call
    4      repeat       31.9
    4   alternate       34.0
   16      repeat       61.0
   16   alternate       63.9
   32      repeat       98.1
   32   alternate      102.7
elapsed_ns=9677875610
exit_status=0
ended=2026-10-14T16:32:52+08:00
```

#### `round2_launch_readme.txt`

```
commit=bc8caf71f9323fb05dc2458f1e12ed8dd9ede5f8
started=2026-10-14T16:32:23+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/mnt/nvme2/jinpli/workspace/home/jinpli/development/benchmark taskset -c 0 /mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/venv/bin/python benchmarks/bench_launch.py --readme --iters 1000 --batches 9
mode=readme; 1000 calls × 9 batches; median
        path   triton us    intj us   speedup
   grid=(1,)       16.51       3.31      5.0x
   grid=(0,)       12.64       0.03    466.6x

torch_access_mode   decode ns    build s
     runtime_shim        89.5       0.02
   static_compile        93.2       0.00
      interpreter       891.4       0.00
elapsed_ns=10188181944
exit_status=0
ended=2026-10-14T16:32:33+08:00
```

#### `round2_launch_sweep.txt`

```
commit=bc8caf71f9323fb05dc2458f1e12ed8dd9ede5f8
started=2026-10-14T16:32:33+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/mnt/nvme2/jinpli/workspace/home/jinpli/development/benchmark taskset -c 0 /mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/venv/bin/python benchmarks/bench_launch.py --sweep --iters 100000 --batches 9
mode=sweep; host-only; 100000 calls × 9 batches; median ns/call
count   kind    ns/call
    4    int       38.1
    4 tensor       36.6
   16    int       66.7
   16 tensor       62.8
   32    int      105.7
   32 tensor      105.9
elapsed_ns=9359699303
exit_status=0
ended=2026-10-14T16:32:43+08:00
```

#### `round2_launch_tuned.txt`

```
commit=bc8caf71f9323fb05dc2458f1e12ed8dd9ede5f8
started=2026-10-14T16:34:06+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/mnt/nvme2/jinpli/workspace/home/jinpli/development/benchmark taskset -c 0 /mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/venv/bin/python benchmarks/bench_launch.py --tuned --iters 20000 --batches 9
 tuned hit:  3129.4 ns/launch
     plain:  3101.0 ns/launch
elapsed_ns=11356906211
exit_status=0
ended=2026-10-14T16:34:17+08:00
```

#### `round2_mix.txt`

```
commit=bc8caf71f9323fb05dc2458f1e12ed8dd9ede5f8
started=2026-10-14T16:34:30+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/mnt/nvme2/jinpli/workspace/home/jinpli/development/benchmark taskset -c 0 /mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/venv/bin/python benchmarks/bench_mix.py
intj 0.1.0 from /mnt/nvme2/jinpli/workspace/home/jinpli/development/intj/intj
python 3.12.3; triton 3.8.0; torch 2.14.0.dev20260804+rocm7.1
mode=mix; host-only; 40000 calls × 9 batches; median ns/call
count share ckind  nkind   pattern   ns/call  samples
    4     0     -    int    repeat      31.9  31.9 31.9 31.9 31.9 32.5 32.1 31.9 32.0 34.8
    4     0     -    int alternate      34.0  34.2 34.1 34.0 34.0 34.1 33.9 33.9 34.1 33.8
    4     0     - tensor    repeat      30.6  30.8 30.7 30.6 30.6 30.6 30.8 30.6 30.6 30.6
    4     0     - tensor alternate      33.1  33.0 33.1 33.0 33.1 33.1 33.0 33.1 33.2 33.2
    4     0     -  mixed    repeat      30.3  30.5 30.4 30.3 30.3 30.3 30.2 30.4 30.5 30.3
    4     0     -  mixed alternate      32.4  32.5 32.4 32.4 32.4 32.4 32.6 32.3 32.4 32.5
    4    25   int    int    repeat      31.1  31.1 31.1 31.3 31.1 31.1 31.0 31.0 30.9 31.0
    4    25   int    int alternate      33.2  33.0 33.5 33.4 33.2 33.4 33.3 33.2 33.2 33.2
    4    25   int tensor    repeat      30.1  30.1 30.0 30.2 30.2 30.1 30.3 30.0 30.1 30.0
    4    25   int tensor alternate      32.6  32.5 32.6 32.6 32.6 32.7 32.6 32.6 32.7 32.7
    4    25   int  mixed    repeat      29.6  29.5 29.6 29.5 29.5 35.0 29.5 29.6 29.6 29.8
    4    25   int  mixed alternate      31.9  31.9 31.8 31.9 32.8 31.8 31.9 31.9 32.1 31.8
    4    25   str    int    repeat      30.7  30.7 30.7 30.7 30.7 30.8 30.6 30.7 30.8 30.7
    4    25   str    int alternate      39.0  38.9 38.8 39.1 39.0 39.1 39.2 39.0 38.9 39.0
    4    25   str tensor    repeat      30.0  30.0 30.1 30.0 30.0 30.1 30.2 30.0 29.9 30.0
    4    25   str tensor alternate      38.1  37.8 38.1 37.8 37.7 37.7 38.3 38.3 38.2 38.3
    4    25   str  mixed    repeat      29.6  29.5 29.6 29.5 29.7 29.5 29.5 29.6 29.6 31.4
    4    25   str  mixed alternate      38.8  38.7 38.7 38.6 38.9 39.0 38.7 38.8 38.9 38.8
    4    25 dtype    int    repeat      30.7  30.6 30.7 30.7 30.8 31.1 31.4 30.9 30.7 30.7
    4    25 dtype    int alternate      38.2  38.1 38.5 38.2 38.3 38.4 38.2 38.2 38.4 38.1
    4    25 dtype tensor    repeat      30.1  30.0 30.0 30.3 30.0 30.2 30.1 30.7 30.2 30.0
    4    25 dtype tensor alternate      36.1  36.2 36.1 36.1 36.0 36.0 36.1 36.0 36.2 36.0
    4    25 dtype  mixed    repeat      29.6  29.7 29.6 29.7 29.5 29.8 29.5 29.6 29.6 29.7
    4    25 dtype  mixed alternate      35.8  35.9 36.5 36.1 35.9 35.7 35.7 35.7 35.8 35.8
    4    50   int    int    repeat      30.5  30.5 30.5 30.5 30.5 30.3 30.4 30.5 31.1 31.3
    4    50   int    int alternate      33.8  33.7 33.7 33.8 33.8 34.1 33.7 33.8 33.6 33.8
    4    50   int tensor    repeat      29.7  29.9 29.7 30.1 29.7 29.7 29.8 29.7 29.6 29.8
    4    50   int tensor alternate      33.4  33.4 33.4 33.4 34.1 34.6 33.4 33.4 33.6 33.4
    4    50   int  mixed    repeat      30.2  30.2 30.1 30.2 30.0 30.0 30.0 30.3 30.4 30.2
    4    50   int  mixed alternate      34.0  34.1 33.9 34.2 34.0 33.9 33.8 34.0 34.0 34.1
    4    50   str    int    repeat      30.1  29.9 30.0 30.1 30.1 30.1 30.1 30.2 30.3 30.2
    4    50   str    int alternate      39.8  39.6 39.9 39.8 39.9 39.8 39.8 39.9 39.9 39.7
    4    50   str tensor    repeat      29.9  31.2 29.8 29.7 29.9 29.9 29.9 29.8 29.9 29.8
    4    50   str tensor alternate      38.9  39.0 38.8 39.0 39.0 38.9 38.9 39.0 38.9 38.8
    4    50   str  mixed    repeat      30.1  30.1 30.3 30.1 30.3 30.1 30.1 30.1 30.0 29.9
    4    50   str  mixed alternate      38.9  38.9 38.9 39.0 38.7 38.9 38.8 38.9 38.9 39.1
    4    50 dtype    int    repeat      30.1  30.1 30.1 30.1 30.2 30.1 30.1 30.0 30.1 30.3
    4    50 dtype    int alternate      38.0  38.0 38.4 37.9 38.0 38.0 38.1 38.2 38.0 38.0
    4    50 dtype tensor    repeat      29.9  29.9 29.9 30.0 29.9 29.9 29.8 30.0 29.9 29.6
    4    50 dtype tensor alternate      36.8  36.8 36.7 36.9 36.7 36.8 36.7 37.0 36.9 36.8
    4    50 dtype  mixed    repeat      30.1  30.5 30.0 30.0 30.0 30.3 30.2 30.1 30.0 30.1
    4    50 dtype  mixed alternate      37.5  37.5 37.5 37.5 37.6 37.6 37.6 37.5 37.5 37.7
    4   100   int      -    repeat      28.5  28.7 28.4 28.4 28.6 28.5 28.6 28.7 28.4 28.5
    4   100   int      - alternate      33.7  33.7 33.8 33.7 33.7 33.6 33.8 34.0 33.6 33.6
    4   100   str      -    repeat      29.4  29.4 29.4 29.4 29.4 29.4 29.3 29.6 29.4 29.5
    4   100   str      - alternate      39.9  40.0 39.9 39.7 39.7 39.9 39.6 39.9 40.0 39.7
    4   100 dtype      -    repeat      29.4  29.5 29.4 29.4 29.4 29.5 29.3 29.4 29.4 29.5
    4   100 dtype      - alternate      38.4  38.4 38.5 38.5 38.4 38.4 38.3 38.5 38.3 38.4
   16     0     -    int    repeat      61.1  61.2 61.1 61.4 60.9 61.3 61.1 60.9 61.3 61.0
   16     0     -    int alternate      63.9  63.9 64.2 63.9 64.0 63.9 64.5 63.7 63.9 63.8
   16     0     - tensor    repeat      58.0  59.0 58.1 58.0 58.1 57.9 57.9 58.1 57.9 57.9
   16     0     - tensor alternate      61.5  61.5 61.5 61.6 61.3 61.8 61.4 61.6 61.4 61.4
   16     0     -  mixed    repeat      55.5  55.3 55.5 55.4 55.4 55.5 55.7 55.7 55.4 55.6
   16     0     -  mixed alternate      59.3  59.1 59.2 59.2 59.3 59.3 59.5 59.6 59.4 59.4
   16    25   int    int    repeat      56.4  56.6 56.4 56.3 56.1 56.5 57.0 56.2 56.1 56.4
   16    25   int    int alternate      62.5  62.3 63.1 62.5 62.6 62.5 62.7 62.4 62.4 62.3
   16    25   int tensor    repeat      55.0  55.3 54.9 55.1 55.0 55.0 54.9 54.9 55.0 54.9
   16    25   int tensor alternate      61.9  61.9 61.8 61.9 62.0 62.3 61.8 62.0 61.9 61.9
   16    25   int  mixed    repeat      52.1  52.0 52.2 52.1 51.9 52.2 52.2 52.0 52.0 52.5
   16    25   int  mixed alternate      58.1  58.1 58.3 58.1 58.0 58.6 58.2 58.3 58.1 57.9
   16    25   str    int    repeat      57.0  57.1 57.3 56.9 57.0 57.1 57.3 57.0 57.0 57.0
   16    25   str    int alternate      68.4  68.6 68.5 68.6 68.3 68.4 68.3 69.0 68.4 68.4
   16    25   str tensor    repeat      55.1  55.2 55.0 55.1 55.3 55.0 54.9 55.0 55.1 55.1
   16    25   str tensor alternate      66.9  66.7 66.7 67.1 66.7 67.3 66.7 67.0 67.3 66.9
   16    25   str  mixed    repeat      53.0  53.0 52.7 53.0 53.1 53.1 53.1 52.9 53.5 53.0
   16    25   str  mixed alternate      64.0  64.0 64.6 64.0 64.2 64.0 63.9 63.8 64.1 64.1
   16    25 dtype    int    repeat      57.6  57.3 57.0 57.2 71.7 91.4 95.1 79.7 57.5 57.6
   16    25 dtype    int alternate      66.8  66.8 66.8 66.8 66.6 67.2 66.7 66.7 67.0 67.1
   16    25 dtype tensor    repeat      55.3  55.2 55.2 55.3 57.4 55.3 55.2 55.2 55.8 55.3
   16    25 dtype tensor alternate      66.1  66.3 66.1 66.1 66.2 66.2 66.1 66.0 67.0 66.1
   16    25 dtype  mixed    repeat      53.0  52.9 53.3 53.0 53.0 52.8 53.4 53.1 52.9 52.8
   16    25 dtype  mixed alternate      64.2  64.2 64.0 64.1 64.2 65.0 64.3 64.6 64.2 64.3
   16    50   int    int    repeat      56.1  56.1 56.0 56.2 56.1 60.4 56.0 56.1 56.2 56.1
   16    50   int    int alternate      67.0  69.3 67.0 66.9 66.9 67.2 67.0 67.0 67.1 67.3
   16    50   int tensor    repeat      54.9  55.2 55.1 55.4 55.2 54.9 54.9 54.9 54.9 54.8
   16    50   int tensor alternate      66.0  66.0 65.8 66.2 66.0 65.8 65.8 65.8 66.0 66.0
   16    50   int  mixed    repeat      54.0  53.9 53.8 54.3 54.1 54.0 53.9 53.8 54.0 54.0
   16    50   int  mixed alternate      64.8  64.7 64.8 64.7 64.7 64.8 64.9 64.8 64.9 64.6
   16    50   str    int    repeat      56.2  56.1 56.3 56.2 56.6 56.2 56.1 56.3 56.1 56.3
   16    50   str    int alternate      74.0  73.5 73.6 74.2 74.1 74.1 73.9 74.2 73.8 74.0
   16    50   str tensor    repeat      54.7  54.7 54.9 54.7 54.9 54.7 55.1 54.9 54.6 54.7
   16    50   str tensor alternate      71.0  71.3 71.0 71.6 70.9 71.1 70.9 70.9 70.7 71.2
   16    50   str  mixed    repeat      53.9  54.3 53.9 53.8 53.9 54.0 54.1 53.8 53.7 54.1
   16    50   str  mixed alternate      70.6  70.6 70.4 70.3 70.9 70.7 70.7 70.7 70.4 70.6
   16    50 dtype    int    repeat      56.1  56.0 56.1 56.1 56.3 56.0 56.1 56.1 56.1 56.1
   16    50 dtype    int alternate      71.2  71.1 71.6 71.2 71.4 71.2 71.5 71.4 71.2 71.1
   16    50 dtype tensor    repeat      54.8  54.9 54.6 54.8 54.9 54.8 54.8 54.7 54.8 54.9
   16    50 dtype tensor alternate      69.9  69.8 69.8 70.0 69.8 70.2 70.0 69.8 69.9 69.9
   16    50 dtype  mixed    repeat      53.9  53.7 53.9 54.0 54.0 54.2 54.0 53.8 53.8 53.7
   16    50 dtype  mixed alternate      68.4  68.5 68.2 68.7 68.1 68.8 68.4 68.4 68.6 68.2
   16   100   int      -    repeat      48.6  48.8 48.6 48.8 48.6 48.5 48.4 48.5 49.0 48.8
   16   100   int      - alternate      71.1  71.8 71.0 71.1 71.0 71.2 70.9 71.1 71.3 71.3
   16   100   str      -    repeat      49.8  50.2 49.6 49.5 49.8 49.8 50.0 49.7 49.9 49.8
   16   100   str      - alternate      78.5  79.4 78.7 78.5 78.3 78.0 78.0 78.7 78.0 78.6
   16   100 dtype      -    repeat      49.6  49.5 49.6 49.5 49.7 49.6 49.6 49.4 49.6 49.6
   16   100 dtype      - alternate      74.1  74.1 74.3 74.1 73.9 73.9 74.1 74.0 74.1 74.3
   32     0     -    int    repeat      97.2  97.8 97.2 97.1 97.3 97.8 97.2 97.2 97.4 97.2
   32     0     -    int alternate     101.6  101.6 101.4 101.4 101.6 101.6 103.8 105.2 102.5 101.9
   32     0     - tensor    repeat      97.4  101.1 96.4 107.8 104.2 98.7 97.4 97.1 97.4 96.5
   32     0     - tensor alternate     104.5  110.7 108.4 105.5 104.5 102.6 104.4 104.1 103.1 105.7
   32     0     -  mixed    repeat      88.5  88.4 88.3 88.4 88.5 88.5 88.5 88.3 88.5 88.6
   32     0     -  mixed alternate      93.4  93.3 93.9 93.3 93.4 94.6 93.6 93.4 93.9 93.4
   32    25   int    int    repeat      88.9  88.6 88.8 89.0 89.1 88.9 88.7 89.0 89.2 88.6
   32    25   int    int alternate     101.8  101.7 101.7 101.8 101.8 101.8 101.9 108.0 106.4 105.3
   32    25   int tensor    repeat      91.4  99.7 96.6 94.2 90.3 90.7 90.6 91.4 91.3 92.4
   32    25   int tensor alternate     104.3  104.6 103.9 103.5 106.6 105.4 105.6 104.3 103.7 102.9
   32    25   int  mixed    repeat      82.1  82.1 82.1 81.9 81.8 82.0 82.1 82.1 82.1 81.9
   32    25   int  mixed alternate      94.2  94.3 94.1 94.2 94.0 94.2 94.4 94.1 94.1 94.2
   32    25   str    int    repeat      89.6  89.6 89.5 89.6 89.7 89.9 89.7 89.5 89.6 89.6
   32    25   str    int alternate     107.9  107.6 108.1 111.7 107.5 107.6 108.7 107.8 107.9 107.9
   32    25   str tensor    repeat     100.9  105.6 101.9 100.0 100.6 100.9 102.1 100.1 100.9 101.1
   32    25   str tensor alternate     119.9  127.9 125.6 118.1 119.1 123.6 121.6 119.9 118.6 118.5
   32    25   str  mixed    repeat      83.9  83.9 84.4 83.9 83.9 84.8 83.9 84.9 83.9 84.3
   32    25   str  mixed alternate     102.7  102.8 102.5 102.2 102.6 102.7 102.7 102.9 102.5 103.0
   32    25 dtype    int    repeat      89.7  89.9 90.1 89.6 89.7 90.1 89.7 89.8 89.5 89.7
   32    25 dtype    int alternate     106.2  106.2 106.1 106.2 106.1 106.1 106.2 106.5 106.2 106.0
   32    25 dtype tensor    repeat     105.0  106.4 104.9 105.6 102.9 103.7 105.0 104.3 105.2 105.6
   32    25 dtype tensor alternate     114.3  114.0 114.3 112.2 113.1 112.6 116.6 116.3 115.4 116.8
   32    25 dtype  mixed    repeat      84.9  84.6 84.4 84.9 84.4 85.3 84.9 84.5 85.0 85.9
   32    25 dtype  mixed alternate      99.5  99.2 99.5 99.3 99.4 99.1 99.5 99.5 99.5 99.5
   32    50   int    int    repeat      90.0  89.0 88.0 91.5 88.8 90.7 89.7 90.8 91.2 90.0
   32    50   int    int alternate     115.8  115.4 116.1 115.5 116.6 116.6 115.8 116.3 115.8 115.6
   32    50   int tensor    repeat      85.2  85.2 85.1 85.2 84.8 84.9 85.2 85.4 85.3 85.1
   32    50   int tensor alternate     107.5  107.4 107.4 107.5 107.6 107.6 107.4 107.5 107.5 107.5
   32    50   int  mixed    repeat      82.6  82.5 82.5 82.6 82.8 82.6 82.6 82.7 82.8 82.9
   32    50   int  mixed alternate     104.9  104.8 104.9 104.8 104.7 105.0 105.0 105.1 104.9 104.8
   32    50   str    int    repeat      88.2  88.1 88.4 88.1 88.2 88.0 88.5 88.2 88.4 88.1
   32    50   str    int alternate     116.7  116.7 117.3 116.6 116.4 117.1 116.7 116.7 117.0 116.8
   32    50   str tensor    repeat      95.6  94.9 95.6 95.6 95.5 95.6 95.1 99.6 104.2 101.0
   32    50   str tensor alternate     119.7  126.1 119.8 119.7 120.1 119.7 118.8 119.3 118.7 118.8
   32    50   str  mixed    repeat      85.7  85.8 85.8 85.1 85.2 85.5 85.7 85.6 85.7 86.1
   32    50   str  mixed alternate     112.4  112.6 112.5 112.2 112.4 112.3 113.0 112.3 112.4 112.1
   32    50 dtype    int    repeat      88.0  88.0 87.8 88.2 88.0 87.8 88.0 88.0 87.6 88.1
   32    50 dtype    int alternate     115.6  115.6 115.5 115.6 115.5 115.8 115.4 117.3 115.7 115.9
   32    50 dtype tensor    repeat      95.2  95.6 94.6 95.3 95.1 95.1 95.3 95.2 95.2 94.9
   32    50 dtype tensor alternate     117.5  118.6 117.6 117.4 117.1 117.8 117.5 117.2 116.2 117.5
   32    50 dtype  mixed    repeat      85.4  85.4 85.2 85.3 85.2 86.0 89.2 85.6 85.7 85.4
   32    50 dtype  mixed alternate     111.0  111.6 111.1 110.4 111.0 111.3 111.3 111.0 110.9 110.8
   32   100   int      -    repeat      70.2  70.4 70.7 70.4 70.1 70.0 70.3 70.2 70.1 70.2
   32   100   int      - alternate     113.8  113.8 113.7 113.7 113.9 113.5 114.8 113.7 114.0 114.3
   32   100   str      -    repeat      81.2  81.1 83.6 81.6 76.8 81.2 84.6 81.6 78.1 81.1
   32   100   str      - alternate     127.3  127.7 126.8 127.5 127.3 127.6 127.6 127.3 126.9 127.1
   32   100 dtype      -    repeat      74.3  75.2 76.1 74.7 74.0 73.9 74.4 74.3 74.3 74.2
   32   100 dtype      - alternate     124.5  125.0 124.6 124.3 124.7 124.5 124.4 124.4 124.3 125.4
elapsed_ns=12368749167
exit_status=0
ended=2026-10-14T16:34:42+08:00
```

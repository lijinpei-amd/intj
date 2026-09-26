# Autotune/heuristics Task 3 benchmark gate

Task 3 of the autotune/heuristics plan ("Single-level tuned launchers end to end"),
measured against `2026-09-26_autotune-baseline_0_87c814b.md`.

Measured at commit `97ef0ae` (Launch autotuned kernels natively after Triton tunes
them), clean checkout (this journal is committed after it). Only untuned launchers are
in the suite; the tuned path is new and has no baseline.

Same suite, driver and order as the baseline: `/tmp/intj-bench/run_all.sh task3`, then,
because rows were flagged, `/tmp/intj-bench/run_all.sh task3-rerun` once. Ten cases x
3 rounds = 30 runs each; all 60 report `exit_status=0` and `commit=97ef0aee...`.
Comparison: `/tmp/gb2/bin/python .superpowers/sdd/2026-09-26-autotune-heuristics/bench_compare.py baseline <label>`.

## Environment

- CPU: x86-64 (224 logical CPUs), `taskset -c 0`, one benchmark process at a time.
  Shared host: load average 3-10 from other users during the runs.
- GPU: AMD Instinct MI308X, `gfx942`.
- Python 3.12.3 at `/tmp/gb2/bin/python`; Torch `2.14.0+rocm7.2`; Triton `3.8.0`;
  `apache-tvm-ffi` `0.1.14.post2.dev1+g424558557.d20260924`.
- `INTJ_BENCHMARK_ROOT=/tmp/gbench` (google/benchmark, kernel-cache case only).
- Rendered modules are built by triton at `-O3`; `kernel_cache` builds with `g++ -O3`.

## Commands

For each round 0-2, each line its own process, prefixed by
`PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python`:

```sh
benchmarks/bench_launch.py --iters 1000 --batches 9                        # launch_gpu
benchmarks/bench_launch.py --no-gpu --iters 100000 --batches 9             # launch_host
benchmarks/bench_launch.py --readme --iters 1000 --batches 9               # launch_readme
benchmarks/bench_launch.py --sweep --iters 100000 --batches 9              # launch_sweep
benchmarks/bench_launch.py --last-key --iters 100000 --batches 9           # launch_last_key
benchmarks/bench_ffi_compare.py --iters 1000 --batches 9                   # ffi_compare
benchmarks/bench_ffi_compare.py --sweep --iters 1000 --batches 9           # ffi_sweep
benchmarks/bench_intj_ffi_paths.py --mode all --iters 1000 --batches 9     # ffi_paths
benchmarks/bench_hip_module_launch.py --iters 1000 --batches 9             # hip_module
-m pytest tests/test_kernel_cache.py -s -q                                 # kernel_cache
```

Timing caveats are the baseline's: GPU rows time host submission, `ffi_paths` cold
compile row uses a unique key per call, kernel-cache rows are map-only C++ costs.

## Verdict

No real regression. The first run flagged 14 rows, all GPU-submission rows plus
`ffi_paths INTJ pair direct`; the rerun flagged 4: `ffi_paths INTJ pair direct`
(+3.0 ns), `hip_module INTJ same function` (+0.47 us), `hip_module TVM FFI HIP HSACO`
(+0.44 us) and `launch_gpu verify on` (+0.11 us). All four are noise:

- **The untuned C is unchanged.** Rendering seven untuned launchers (default grid,
  `grid_cpp`, `grid_arg`+`return_compiled`, INTERPRETER, a baked constexpr, `grid_py`,
  `bind_device`) from `6af5613` and from `97ef0ae` gives sources that differ only in
  blank lines, the rename of the launch tail's local `result` to `out`, one `(void)f;`
  in `intj_final_release`, and the `entry()` refusal message for `grid_py`. The Python
  changes run in `make_launcher` only.
- **Rows that run no changed code moved as much.** `hip_module TVM FFI HIP HSACO`
  (TVM FFI, no intj) moved +14% alongside `INTJ same function` +16%.
- **Interleaved A/B** (below) against a worktree of the pre-task commit `6af5613`: the
  hip_module rows come out 1-2% *faster* in task3, and `pair direct` +1.0 ns (under the
  threshold; Task 1 accepted the same row as noise). `launch_gpu` is bimodal per
  process (about 2.9 or 3.5 us) and both trees land in both modes: base round 2 at
  3470 ns, task3 round 0 at 2904 ns. Host-only launch rows (`launch_host`,
  `launch_sweep`, `launch_last_key`) are within 3% or 2 ns in both runs.

## Baseline vs task3 medians

Median of 3 rounds per label; Δ is against the baseline median. `hit_child` rows are
Task 1 additions with no baseline entry (omitted).

| case | row | unit | baseline | task3 | Δ | Δ% | flag |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| ffi_compare | FFI empty kernel | us | 3.3617 | 3.3280 | -0.0337 | -1.0% |  |
| ffi_compare | FFI mixed kernel | us | 3.3407 | 3.3587 | +0.018 | +0.5% |  |
| ffi_compare | FFI packed nop | us | 0.1446 | 0.1446 | +0 | +0.0% |  |
| ffi_compare | FFI packed nop mixed | us | 0.1745 | 0.1748 | +0.0003 | +0.2% |  |
| ffi_compare | FFI typed nop | us | 0.1491 | 0.1501 | +0.001 | +0.7% |  |
| ffi_compare | FFI typed nop mixed | us | 0.1772 | 0.1769 | -0.0003 | -0.2% |  |
| ffi_compare | INTJ empty kernel | us | 2.9849 | 2.9626 | -0.0223 | -0.7% |  |
| ffi_compare | INTJ fixed mixed kernel | us | 2.9387 | 2.9312 | -0.0075 | -0.3% |  |
| ffi_compare | INTJ fixed-device kernel | us | 2.9186 | 2.8684 | -0.0502 | -1.7% |  |
| ffi_compare | INTJ mixed kernel | us | 2.9280 | 2.9508 | +0.0228 | +0.8% |  |
| ffi_paths | FFI unpack Pair only | ns | 162.7 | 167.3 | +4.6 | +2.8% |  |
| ffi_paths | INTJ 3-tensor host-only nop | ns | 33.8000 | 34.0000 | +0.2 | +0.6% |  |
| ffi_paths | INTJ FFI wrapper kwargs | ns | 126.7 | 126.5 | -0.2 | -0.2% |  |
| ffi_paths | INTJ FFI wrapper positional | ns | 103.4 | 103.3 | -0.1 | -0.1% |  |
| ffi_paths | INTJ adapter defaults | ns | 89.2000 | 90.1000 | +0.9 | +1.0% |  |
| ffi_paths | INTJ adapter kwargs | ns | 108.2 | 107.8 | -0.4 | -0.4% |  |
| ffi_paths | INTJ adapter positional | ns | 90.6000 | 90.5000 | -0.1 | -0.1% |  |
| ffi_paths | INTJ config FFI unpack | ns | 321.1 | 316.8 | -4.3 | -1.3% |  |
| ffi_paths | INTJ config direct | ns | 79.1000 | 80.1000 | +1 | +1.3% |  |
| ffi_paths | INTJ direct positional | ns | 64.7000 | 65.5000 | +0.8 | +1.2% |  |
| ffi_paths | INTJ hot call (no callback) | ns | 37.2000 | 38.7000 | +1.5 | +4.0% |  |
| ffi_paths | INTJ pair FFI unpack | ns | 309.1 | 317.3 | +8.2 | +2.7% |  |
| ffi_paths | INTJ pair direct | ns | 65.6000 | 68.0000 | +2.4 | +3.7% | flag |
| ffi_paths | INTJ pair manual unpack | ns | 171.6 | 173.7 | +2.1 | +1.2% |  |
| ffi_paths | INTJ pair prebuilt *tuple | ns | 137.4 | 139.4 | +2 | +1.5% |  |
| ffi_paths | INTJ pair stdlib astuple | ns | 1168.1 | 1159.5 | -8.6 | -0.7% |  |
| ffi_sweep | args=0 FFI empty kernel | us | 1.6784 | 1.6445 | -0.0339 | -2.0% |  |
| ffi_sweep | args=0 FFI packed nop | us | 0.1175 | 0.1167 | -0.0008 | -0.7% |  |
| ffi_sweep | args=0 FFI typed nop | us | 0.1210 | 0.1195 | -0.0015 | -1.2% |  |
| ffi_sweep | args=16 FFI empty kernel | us | 3.6540 | 3.7082 | +0.0542 | +1.5% |  |
| ffi_sweep | args=16 FFI packed nop | us | 0.2923 | 0.2955 | +0.0032 | +1.1% |  |
| ffi_sweep | args=16 FFI typed nop | us | 0.3021 | 0.3062 | +0.0041 | +1.4% |  |
| ffi_sweep | args=3 FFI empty kernel | us | 3.2503 | 3.3712 | +0.1209 | +3.7% | flag |
| ffi_sweep | args=3 FFI packed nop | us | 0.1451 | 0.1452 | +0.0001 | +0.1% |  |
| ffi_sweep | args=3 FFI typed nop | us | 0.1492 | 0.1488 | -0.0004 | -0.3% |  |
| ffi_sweep | args=32 FFI empty kernel | us | 4.2249 | 4.2654 | +0.0405 | +1.0% |  |
| ffi_sweep | args=32 FFI packed nop | us | 0.4805 | 0.4915 | +0.011 | +2.3% |  |
| ffi_sweep | args=32 FFI typed nop | us | 0.4946 | 0.5022 | +0.0076 | +1.5% |  |
| ffi_sweep | args=5 FFI empty kernel | us | 3.2901 | 3.3491 | +0.059 | +1.8% |  |
| ffi_sweep | args=5 FFI packed nop | us | 0.1740 | 0.1737 | -0.0003 | -0.2% |  |
| ffi_sweep | args=5 FFI typed nop | us | 0.1783 | 0.1780 | -0.0003 | -0.2% |  |
| ffi_sweep | args=64 FFI empty kernel | us | 5.0544 | 5.0619 | +0.0075 | +0.1% |  |
| ffi_sweep | args=64 FFI packed nop | us | 0.8387 | 0.8390 | +0.0003 | +0.0% |  |
| ffi_sweep | args=64 FFI typed nop | us | 0.8587 | 0.8783 | +0.0196 | +2.3% |  |
| ffi_sweep | args=8 FFI empty kernel | us | 3.3924 | 3.4819 | +0.0895 | +2.6% |  |
| ffi_sweep | args=8 FFI packed nop | us | 0.2060 | 0.2066 | +0.0006 | +0.3% |  |
| ffi_sweep | args=8 FFI typed nop | us | 0.2099 | 0.2086 | -0.0013 | -0.6% |  |
| hip_module | INTJ same function | us | 2.9336 | 3.4477 | +0.5141 | +17.5% | flag |
| hip_module | TVM FFI HIP HSACO | us | 3.1473 | 3.6552 | +0.5079 | +16.1% | flag |
| hip_module | Triton same HSACO | us | 15.9539 | 16.4724 | +0.5185 | +3.2% | flag |
| kernel_cache | key       hash_only | ns | 1.3600 | 1.3700 | +0.01 | +0.7% |  |
| kernel_cache | key       hash_only | ns | 1.3600 | 1.4700 | +0.11 | +8.1% |  |
| kernel_cache | key       hash_only | ns | 1.3600 | 1.3600 | +0 | +0.0% |  |
| kernel_cache | key       hit/1 | ns | 1.3500 | 1.5400 | +0.19 | +14.1% |  |
| kernel_cache | key       hit/1 | ns | 1.2000 | 1.7600 | +0.56 | +46.7% |  |
| kernel_cache | key       hit/1 | ns | 1.8500 | 2.0400 | +0.19 | +10.3% |  |
| kernel_cache | key       hit/512 | ns | 4.9400 | 5.8000 | +0.86 | +17.4% |  |
| kernel_cache | key       hit/512 | ns | 1.4900 | 2.5400 | +1.05 | +70.5% |  |
| kernel_cache | key       hit/512 | ns | 2.6200 | 3.0300 | +0.41 | +15.6% |  |
| kernel_cache | key       hit/64 | ns | 4.3500 | 4.9700 | +0.62 | +14.3% |  |
| kernel_cache | key       hit/64 | ns | 1.3100 | 1.8700 | +0.56 | +42.7% |  |
| kernel_cache | key       hit/64 | ns | 1.9100 | 2.1700 | +0.26 | +13.6% |  |
| kernel_cache | key       hit/8 | ns | 4.2300 | 4.7600 | +0.53 | +12.5% |  |
| kernel_cache | key       hit/8 | ns | 1.2600 | 1.8500 | +0.59 | +46.8% |  |
| kernel_cache | key       hit/8 | ns | 1.8800 | 2.1400 | +0.26 | +13.8% |  |
| kernel_cache | key       miss/1 | ns | 1.3800 | 1.3800 | +0 | +0.0% |  |
| kernel_cache | key       miss/1 | ns | 1.0300 | 1.3900 | +0.36 | +35.0% |  |
| kernel_cache | key       miss/1 | ns | 1.4600 | 1.4600 | +0 | +0.0% |  |
| kernel_cache | key       miss/512 | ns | 3.9400 | 4.4000 | +0.46 | +11.7% |  |
| kernel_cache | key       miss/512 | ns | 2.2300 | 2.8200 | +0.59 | +26.5% |  |
| kernel_cache | key       miss/512 | ns | 1.9700 | 2.4100 | +0.44 | +22.3% |  |
| kernel_cache | key       miss/64 | ns | 3.8100 | 3.9100 | +0.1 | +2.6% |  |
| kernel_cache | key       miss/64 | ns | 2.1500 | 2.7300 | +0.58 | +27.0% |  |
| kernel_cache | key       miss/64 | ns | 1.7300 | 2.1100 | +0.38 | +22.0% |  |
| kernel_cache | key       miss/8 | ns | 3.2500 | 3.6500 | +0.4 | +12.3% |  |
| kernel_cache | key       miss/8 | ns | 2.2200 | 2.4200 | +0.2 | +9.0% |  |
| kernel_cache | key       miss/8 | ns | 1.7500 | 2.1200 | +0.37 | +21.1% |  |
| kernel_cache | key       hash_only | ns | 1.4200 | 1.4200 | +0 | +0.0% |  |
| kernel_cache | key       hash_only | ns | 1.4200 | 1.4200 | +0 | +0.0% |  |
| kernel_cache | key       hash_only | ns | 1.4200 | 1.4200 | +0 | +0.0% |  |
| kernel_cache | key       hit/1 | ns | 1.7000 | 1.9900 | +0.29 | +17.1% |  |
| kernel_cache | key       hit/1 | ns | 1.8900 | 2.1800 | +0.29 | +15.3% |  |
| kernel_cache | key       hit/1 | ns | 2.2600 | 2.5700 | +0.31 | +13.7% |  |
| kernel_cache | key       hit/512 | ns | 5.8300 | 6.8000 | +0.97 | +16.6% |  |
| kernel_cache | key       hit/512 | ns | 2.7300 | 3.3300 | +0.6 | +22.0% |  |
| kernel_cache | key       hit/512 | ns | 3.4600 | 4.0400 | +0.58 | +16.8% |  |
| kernel_cache | key       hit/64 | ns | 5.2500 | 5.6000 | +0.35 | +6.7% |  |
| kernel_cache | key       hit/64 | ns | 2.1100 | 2.5000 | +0.39 | +18.5% |  |
| kernel_cache | key       hit/64 | ns | 2.5800 | 2.9200 | +0.34 | +13.2% |  |
| kernel_cache | key       hit/8 | ns | 4.7600 | 5.4000 | +0.64 | +13.4% |  |
| kernel_cache | key       hit/8 | ns | 1.8900 | 2.5500 | +0.66 | +34.9% |  |
| kernel_cache | key       hit/8 | ns | 2.2700 | 2.5700 | +0.3 | +13.2% |  |
| kernel_cache | key       miss/1 | ns | 1.5400 | 1.7100 | +0.17 | +11.0% |  |
| kernel_cache | key       miss/1 | ns | 1.3200 | 1.4100 | +0.09 | +6.8% |  |
| kernel_cache | key       miss/1 | ns | 2.1600 | 2.3200 | +0.16 | +7.4% |  |
| kernel_cache | key       miss/512 | ns | 3.9000 | 4.4200 | +0.52 | +13.3% |  |
| kernel_cache | key       miss/512 | ns | 2.9500 | 3.2100 | +0.26 | +8.8% |  |
| kernel_cache | key       miss/512 | ns | 2.3000 | 2.7300 | +0.43 | +18.7% |  |
| kernel_cache | key       miss/64 | ns | 3.3600 | 4.1700 | +0.81 | +24.1% |  |
| kernel_cache | key       miss/64 | ns | 3.0800 | 3.3600 | +0.28 | +9.1% |  |
| kernel_cache | key       miss/64 | ns | 2.0100 | 2.3400 | +0.33 | +16.4% |  |
| kernel_cache | key       miss/8 | ns | 3.2700 | 4.2200 | +0.95 | +29.1% |  |
| kernel_cache | key       miss/8 | ns | 2.7100 | 2.8700 | +0.16 | +5.9% |  |
| kernel_cache | key       miss/8 | ns | 1.7500 | 1.8000 | +0.05 | +2.9% |  |
| kernel_cache | key       hash_only | ns | 3.5700 | 3.5800 | +0.01 | +0.3% |  |
| kernel_cache | key       hash_only | ns | 3.5800 | 4.5300 | +0.95 | +26.5% |  |
| kernel_cache | key       hash_only | ns | 3.5800 | 3.5800 | +0 | +0.0% |  |
| kernel_cache | key       hit/1 | ns | 2.6800 | 2.8400 | +0.16 | +6.0% |  |
| kernel_cache | key       hit/1 | ns | 2.9400 | 3.1500 | +0.21 | +7.1% |  |
| kernel_cache | key       hit/1 | ns | 3.1900 | 3.1800 | -0.01 | -0.3% |  |
| kernel_cache | key       hit/512 | ns | 11.1800 | 11.6800 | +0.5 | +4.5% |  |
| kernel_cache | key       hit/512 | ns | 4.4700 | 4.7300 | +0.26 | +5.8% |  |
| kernel_cache | key       hit/512 | ns | 4.9400 | 4.9200 | -0.02 | -0.4% |  |
| kernel_cache | key       hit/64 | ns | 9.7500 | 10.3200 | +0.57 | +5.8% |  |
| kernel_cache | key       hit/64 | ns | 3.2500 | 3.4800 | +0.23 | +7.1% |  |
| kernel_cache | key       hit/64 | ns | 3.6400 | 3.6200 | -0.02 | -0.5% |  |
| kernel_cache | key       hit/8 | ns | 9.4700 | 10.0100 | +0.54 | +5.7% |  |
| kernel_cache | key       hit/8 | ns | 3.1700 | 3.3700 | +0.2 | +6.3% |  |
| kernel_cache | key       hit/8 | ns | 3.5400 | 3.5300 | -0.01 | -0.3% |  |
| kernel_cache | key       miss/1 | ns | 1.6600 | 1.7500 | +0.09 | +5.4% |  |
| kernel_cache | key       miss/1 | ns | 1.8500 | 2.0300 | +0.18 | +9.7% |  |
| kernel_cache | key       miss/1 | ns | 2.2700 | 2.1200 | -0.15 | -6.6% |  |
| kernel_cache | key       miss/512 | ns | 7.6600 | 7.9400 | +0.28 | +3.7% |  |
| kernel_cache | key       miss/512 | ns | 4.1000 | 4.1500 | +0.05 | +1.2% |  |
| kernel_cache | key       miss/512 | ns | 2.7600 | 2.5600 | -0.2 | -7.2% |  |
| kernel_cache | key       miss/64 | ns | 6.7100 | 6.9200 | +0.21 | +3.1% |  |
| kernel_cache | key       miss/64 | ns | 3.3800 | 3.7100 | +0.33 | +9.8% |  |
| kernel_cache | key       miss/64 | ns | 2.5500 | 2.4500 | -0.1 | -3.9% |  |
| kernel_cache | key       miss/8 | ns | 6.9300 | 7.1500 | +0.22 | +3.2% |  |
| kernel_cache | key       miss/8 | ns | 3.5700 | 3.8200 | +0.25 | +7.0% |  |
| kernel_cache | key       miss/8 | ns | 3.0900 | 2.1600 | -0.93 | -30.1% |  |
| launch_gpu | auto map | ns | 3137.8 | 3247.4 | +109.6 | +3.5% | flag |
| launch_gpu | baked | ns | 3054.5 | 3215.7 | +161.2 | +5.3% | flag |
| launch_gpu | bound pointer | ns | 2935.5 | 3205.8 | +270.3 | +9.2% | flag |
| launch_gpu | bound tensor | ns | 3086.1 | 3222.3 | +136.2 | +4.4% | flag |
| launch_gpu | fixed device map | ns | 2932.4 | 3250.9 | +318.5 | +10.9% | flag |
| launch_gpu | fixed device no-map | ns | 3010.8 | 3226.9 | +216.1 | +7.2% | flag |
| launch_gpu | reduced key | ns | 3123.5 | 3192.9 | +69.4 | +2.2% |  |
| launch_gpu | verify off | ns | 2994.9 | 3317.7 | +322.8 | +10.8% | flag |
| launch_gpu | verify on | ns | 2913.9 | 3233.6 | +319.7 | +11.0% | flag |
| launch_host | auto map | ns | 41.6000 | 41.8000 | +0.2 | +0.5% |  |
| launch_host | baked | ns | 39.2000 | 40.2000 | +1 | +2.6% |  |
| launch_host | bound pointer | ns | 44.8000 | 45.0000 | +0.2 | +0.4% |  |
| launch_host | bound tensor | ns | 45.6000 | 46.0000 | +0.4 | +0.9% |  |
| launch_host | fixed device map | ns | 42.1000 | 43.1000 | +1 | +2.4% |  |
| launch_host | fixed device no-map | ns | 44.0000 | 44.5000 | +0.5 | +1.1% |  |
| launch_host | reduced key | ns | 41.9000 | 42.0000 | +0.1 | +0.2% |  |
| launch_host | verify off | ns | 42.5000 | 42.2000 | -0.3 | -0.7% |  |
| launch_host | verify on | ns | 44.2000 | 44.1000 | -0.1 | -0.2% |  |
| launch_last_key | 16 alternate | ns | 65.4000 | 65.2000 | -0.2 | -0.3% |  |
| launch_last_key | 16 repeat | ns | 61.9000 | 62.5000 | +0.6 | +1.0% |  |
| launch_last_key | 32 alternate | ns | 109.1 | 104.2 | -4.9 | -4.5% |  |
| launch_last_key | 32 repeat | ns | 104.8 | 99.8000 | -5 | -4.8% |  |
| launch_last_key | 4 alternate | ns | 35.9000 | 36.2000 | +0.3 | +0.8% |  |
| launch_last_key | 4 repeat | ns | 33.9000 | 33.8000 | -0.1 | -0.3% |  |
| launch_readme | grid=(0,) intj | us | 0.0300 | 0.0300 | +0 | +0.0% |  |
| launch_readme | grid=(0,) triton | us | 12.9200 | 13.3500 | +0.43 | +3.3% | flag |
| launch_readme | grid=(1,) intj | us | 3.1200 | 3.1300 | +0.01 | +0.3% |  |
| launch_readme | grid=(1,) triton | us | 16.4500 | 16.9100 | +0.46 | +2.8% |  |
| launch_sweep | 16 int | ns | 67.9000 | 68.1000 | +0.2 | +0.3% |  |
| launch_sweep | 16 tensor | ns | 65.2000 | 64.4000 | -0.8 | -1.2% |  |
| launch_sweep | 32 int | ns | 112.8 | 106.0 | -6.8 | -6.0% |  |
| launch_sweep | 32 tensor | ns | 114.0 | 110.8 | -3.2 | -2.8% |  |
| launch_sweep | 4 int | ns | 40.5000 | 40.7000 | +0.2 | +0.5% |  |
| launch_sweep | 4 tensor | ns | 39.2000 | 39.4000 | +0.2 | +0.5% |  |

## Baseline vs task3-rerun medians

| case | row | unit | baseline | task3 | Δ | Δ% | flag |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| ffi_compare | FFI empty kernel | us | 3.3617 | 3.3395 | -0.0222 | -0.7% |  |
| ffi_compare | FFI mixed kernel | us | 3.3407 | 3.3704 | +0.0297 | +0.9% |  |
| ffi_compare | FFI packed nop | us | 0.1446 | 0.1457 | +0.0011 | +0.8% |  |
| ffi_compare | FFI packed nop mixed | us | 0.1745 | 0.1749 | +0.0004 | +0.2% |  |
| ffi_compare | FFI typed nop | us | 0.1491 | 0.1497 | +0.0006 | +0.4% |  |
| ffi_compare | FFI typed nop mixed | us | 0.1772 | 0.1770 | -0.0002 | -0.1% |  |
| ffi_compare | INTJ empty kernel | us | 2.9849 | 2.9852 | +0.0003 | +0.0% |  |
| ffi_compare | INTJ fixed mixed kernel | us | 2.9387 | 2.9173 | -0.0214 | -0.7% |  |
| ffi_compare | INTJ fixed-device kernel | us | 2.9186 | 2.9311 | +0.0125 | +0.4% |  |
| ffi_compare | INTJ mixed kernel | us | 2.9280 | 2.8913 | -0.0367 | -1.3% |  |
| ffi_paths | FFI unpack Pair only | ns | 162.7 | 164.1 | +1.4 | +0.9% |  |
| ffi_paths | INTJ 3-tensor host-only nop | ns | 33.8000 | 34.1000 | +0.3 | +0.9% |  |
| ffi_paths | INTJ FFI wrapper kwargs | ns | 126.7 | 126.6 | -0.1 | -0.1% |  |
| ffi_paths | INTJ FFI wrapper positional | ns | 103.4 | 103.3 | -0.1 | -0.1% |  |
| ffi_paths | INTJ adapter defaults | ns | 89.2000 | 89.1000 | -0.1 | -0.1% |  |
| ffi_paths | INTJ adapter kwargs | ns | 108.2 | 108.0 | -0.2 | -0.2% |  |
| ffi_paths | INTJ adapter positional | ns | 90.6000 | 90.7000 | +0.1 | +0.1% |  |
| ffi_paths | INTJ config FFI unpack | ns | 321.1 | 318.0 | -3.1 | -1.0% |  |
| ffi_paths | INTJ config direct | ns | 79.1000 | 80.4000 | +1.3 | +1.6% |  |
| ffi_paths | INTJ direct positional | ns | 64.7000 | 64.9000 | +0.2 | +0.3% |  |
| ffi_paths | INTJ hot call (no callback) | ns | 37.2000 | 38.4000 | +1.2 | +3.2% |  |
| ffi_paths | INTJ pair FFI unpack | ns | 309.1 | 316.7 | +7.6 | +2.5% |  |
| ffi_paths | INTJ pair direct | ns | 65.6000 | 68.6000 | +3 | +4.6% | flag |
| ffi_paths | INTJ pair manual unpack | ns | 171.6 | 173.5 | +1.9 | +1.1% |  |
| ffi_paths | INTJ pair prebuilt *tuple | ns | 137.4 | 138.2 | +0.8 | +0.6% |  |
| ffi_paths | INTJ pair stdlib astuple | ns | 1168.1 | 1167.9 | -0.2 | -0.0% |  |
| ffi_sweep | args=0 FFI empty kernel | us | 1.6784 | 1.6808 | +0.0024 | +0.1% |  |
| ffi_sweep | args=0 FFI packed nop | us | 0.1175 | 0.1168 | -0.0007 | -0.6% |  |
| ffi_sweep | args=0 FFI typed nop | us | 0.1210 | 0.1208 | -0.0002 | -0.2% |  |
| ffi_sweep | args=16 FFI empty kernel | us | 3.6540 | 3.6524 | -0.0016 | -0.0% |  |
| ffi_sweep | args=16 FFI packed nop | us | 0.2923 | 0.2928 | +0.0005 | +0.2% |  |
| ffi_sweep | args=16 FFI typed nop | us | 0.3021 | 0.3053 | +0.0032 | +1.1% |  |
| ffi_sweep | args=3 FFI empty kernel | us | 3.2503 | 3.2860 | +0.0357 | +1.1% |  |
| ffi_sweep | args=3 FFI packed nop | us | 0.1451 | 0.1448 | -0.0003 | -0.2% |  |
| ffi_sweep | args=3 FFI typed nop | us | 0.1492 | 0.1497 | +0.0005 | +0.3% |  |
| ffi_sweep | args=32 FFI empty kernel | us | 4.2249 | 4.2269 | +0.002 | +0.0% |  |
| ffi_sweep | args=32 FFI packed nop | us | 0.4805 | 0.4785 | -0.002 | -0.4% |  |
| ffi_sweep | args=32 FFI typed nop | us | 0.4946 | 0.4946 | +0 | +0.0% |  |
| ffi_sweep | args=5 FFI empty kernel | us | 3.2901 | 3.2910 | +0.0009 | +0.0% |  |
| ffi_sweep | args=5 FFI packed nop | us | 0.1740 | 0.1742 | +0.0002 | +0.1% |  |
| ffi_sweep | args=5 FFI typed nop | us | 0.1783 | 0.1774 | -0.0009 | -0.5% |  |
| ffi_sweep | args=64 FFI empty kernel | us | 5.0544 | 5.0860 | +0.0316 | +0.6% |  |
| ffi_sweep | args=64 FFI packed nop | us | 0.8387 | 0.8475 | +0.0088 | +1.0% |  |
| ffi_sweep | args=64 FFI typed nop | us | 0.8587 | 0.8675 | +0.0088 | +1.0% |  |
| ffi_sweep | args=8 FFI empty kernel | us | 3.3924 | 3.3956 | +0.0032 | +0.1% |  |
| ffi_sweep | args=8 FFI packed nop | us | 0.2060 | 0.2074 | +0.0014 | +0.7% |  |
| ffi_sweep | args=8 FFI typed nop | us | 0.2099 | 0.2096 | -0.0003 | -0.1% |  |
| hip_module | INTJ same function | us | 2.9336 | 3.4002 | +0.4666 | +15.9% | flag |
| hip_module | TVM FFI HIP HSACO | us | 3.1473 | 3.5912 | +0.4439 | +14.1% | flag |
| hip_module | Triton same HSACO | us | 15.9539 | 16.1804 | +0.2265 | +1.4% |  |
| kernel_cache | key       hash_only | ns | 1.3600 | 1.3600 | +0 | +0.0% |  |
| kernel_cache | key       hash_only | ns | 1.3600 | 1.3700 | +0.01 | +0.7% |  |
| kernel_cache | key       hash_only | ns | 1.3600 | 1.3600 | +0 | +0.0% |  |
| kernel_cache | key       hit/1 | ns | 1.3500 | 1.5100 | +0.16 | +11.9% |  |
| kernel_cache | key       hit/1 | ns | 1.2000 | 1.7600 | +0.56 | +46.7% |  |
| kernel_cache | key       hit/1 | ns | 1.8500 | 2.0400 | +0.19 | +10.3% |  |
| kernel_cache | key       hit/512 | ns | 4.9400 | 5.7900 | +0.85 | +17.2% |  |
| kernel_cache | key       hit/512 | ns | 1.4900 | 2.5500 | +1.06 | +71.1% |  |
| kernel_cache | key       hit/512 | ns | 2.6200 | 3.0400 | +0.42 | +16.0% |  |
| kernel_cache | key       hit/64 | ns | 4.3500 | 4.9500 | +0.6 | +13.8% |  |
| kernel_cache | key       hit/64 | ns | 1.3100 | 1.8800 | +0.57 | +43.5% |  |
| kernel_cache | key       hit/64 | ns | 1.9100 | 2.1700 | +0.26 | +13.6% |  |
| kernel_cache | key       hit/8 | ns | 4.2300 | 4.7700 | +0.54 | +12.8% |  |
| kernel_cache | key       hit/8 | ns | 1.2600 | 1.8500 | +0.59 | +46.8% |  |
| kernel_cache | key       hit/8 | ns | 1.8800 | 2.1400 | +0.26 | +13.8% |  |
| kernel_cache | key       miss/1 | ns | 1.3800 | 1.8000 | +0.42 | +30.4% |  |
| kernel_cache | key       miss/1 | ns | 1.0300 | 1.3900 | +0.36 | +35.0% |  |
| kernel_cache | key       miss/1 | ns | 1.4600 | 1.5300 | +0.07 | +4.8% |  |
| kernel_cache | key       miss/512 | ns | 3.9400 | 4.4200 | +0.48 | +12.2% |  |
| kernel_cache | key       miss/512 | ns | 2.2300 | 2.8000 | +0.57 | +25.6% |  |
| kernel_cache | key       miss/512 | ns | 1.9700 | 2.4400 | +0.47 | +23.9% |  |
| kernel_cache | key       miss/64 | ns | 3.8100 | 3.9200 | +0.11 | +2.9% |  |
| kernel_cache | key       miss/64 | ns | 2.1500 | 2.7900 | +0.64 | +29.8% |  |
| kernel_cache | key       miss/64 | ns | 1.7300 | 2.5700 | +0.84 | +48.6% |  |
| kernel_cache | key       miss/8 | ns | 3.2500 | 3.6500 | +0.4 | +12.3% |  |
| kernel_cache | key       miss/8 | ns | 2.2200 | 2.4000 | +0.18 | +8.1% |  |
| kernel_cache | key       miss/8 | ns | 1.7500 | 2.1200 | +0.37 | +21.1% |  |
| kernel_cache | key       hash_only | ns | 1.4200 | 1.4200 | +0 | +0.0% |  |
| kernel_cache | key       hash_only | ns | 1.4200 | 1.4200 | +0 | +0.0% |  |
| kernel_cache | key       hash_only | ns | 1.4200 | 1.4300 | +0.01 | +0.7% |  |
| kernel_cache | key       hit/1 | ns | 1.7000 | 1.9800 | +0.28 | +16.5% |  |
| kernel_cache | key       hit/1 | ns | 1.8900 | 2.1800 | +0.29 | +15.3% |  |
| kernel_cache | key       hit/1 | ns | 2.2600 | 2.5700 | +0.31 | +13.7% |  |
| kernel_cache | key       hit/512 | ns | 5.8300 | 6.8300 | +1 | +17.2% |  |
| kernel_cache | key       hit/512 | ns | 2.7300 | 3.3600 | +0.63 | +23.1% |  |
| kernel_cache | key       hit/512 | ns | 3.4600 | 4.0400 | +0.58 | +16.8% |  |
| kernel_cache | key       hit/64 | ns | 5.2500 | 5.6000 | +0.35 | +6.7% |  |
| kernel_cache | key       hit/64 | ns | 2.1100 | 2.5100 | +0.4 | +19.0% |  |
| kernel_cache | key       hit/64 | ns | 2.5800 | 2.9200 | +0.34 | +13.2% |  |
| kernel_cache | key       hit/8 | ns | 4.7600 | 5.3900 | +0.63 | +13.2% |  |
| kernel_cache | key       hit/8 | ns | 1.8900 | 2.1800 | +0.29 | +15.3% |  |
| kernel_cache | key       hit/8 | ns | 2.2700 | 2.5600 | +0.29 | +12.8% |  |
| kernel_cache | key       miss/1 | ns | 1.5400 | 1.7300 | +0.19 | +12.3% |  |
| kernel_cache | key       miss/1 | ns | 1.3200 | 1.4300 | +0.11 | +8.3% |  |
| kernel_cache | key       miss/1 | ns | 2.1600 | 2.3200 | +0.16 | +7.4% |  |
| kernel_cache | key       miss/512 | ns | 3.9000 | 4.4200 | +0.52 | +13.3% |  |
| kernel_cache | key       miss/512 | ns | 2.9500 | 3.2800 | +0.33 | +11.2% |  |
| kernel_cache | key       miss/512 | ns | 2.3000 | 2.7200 | +0.42 | +18.3% |  |
| kernel_cache | key       miss/64 | ns | 3.3600 | 4.1700 | +0.81 | +24.1% |  |
| kernel_cache | key       miss/64 | ns | 3.0800 | 3.3500 | +0.27 | +8.8% |  |
| kernel_cache | key       miss/64 | ns | 2.0100 | 2.3500 | +0.34 | +16.9% |  |
| kernel_cache | key       miss/8 | ns | 3.2700 | 3.6100 | +0.34 | +10.4% |  |
| kernel_cache | key       miss/8 | ns | 2.7100 | 2.7600 | +0.05 | +1.8% |  |
| kernel_cache | key       miss/8 | ns | 1.7500 | 1.8100 | +0.06 | +3.4% |  |
| kernel_cache | key       hash_only | ns | 3.5700 | 3.5800 | +0.01 | +0.3% |  |
| kernel_cache | key       hash_only | ns | 3.5800 | 3.5700 | -0.01 | -0.3% |  |
| kernel_cache | key       hash_only | ns | 3.5800 | 3.5600 | -0.02 | -0.6% |  |
| kernel_cache | key       hit/1 | ns | 2.6800 | 2.8300 | +0.15 | +5.6% |  |
| kernel_cache | key       hit/1 | ns | 2.9400 | 3.1300 | +0.19 | +6.5% |  |
| kernel_cache | key       hit/1 | ns | 3.1900 | 3.1800 | -0.01 | -0.3% |  |
| kernel_cache | key       hit/512 | ns | 11.1800 | 11.6600 | +0.48 | +4.3% |  |
| kernel_cache | key       hit/512 | ns | 4.4700 | 4.7300 | +0.26 | +5.8% |  |
| kernel_cache | key       hit/512 | ns | 4.9400 | 4.9300 | -0.01 | -0.2% |  |
| kernel_cache | key       hit/64 | ns | 9.7500 | 10.3100 | +0.56 | +5.7% |  |
| kernel_cache | key       hit/64 | ns | 3.2500 | 3.4500 | +0.2 | +6.2% |  |
| kernel_cache | key       hit/64 | ns | 3.6400 | 3.6200 | -0.02 | -0.5% |  |
| kernel_cache | key       hit/8 | ns | 9.4700 | 10.0200 | +0.55 | +5.8% |  |
| kernel_cache | key       hit/8 | ns | 3.1700 | 3.5200 | +0.35 | +11.0% |  |
| kernel_cache | key       hit/8 | ns | 3.5400 | 3.5300 | -0.01 | -0.3% |  |
| kernel_cache | key       miss/1 | ns | 1.6600 | 1.7200 | +0.06 | +3.6% |  |
| kernel_cache | key       miss/1 | ns | 1.8500 | 2.0000 | +0.15 | +8.1% |  |
| kernel_cache | key       miss/1 | ns | 2.2700 | 2.1200 | -0.15 | -6.6% |  |
| kernel_cache | key       miss/512 | ns | 7.6600 | 7.9200 | +0.26 | +3.4% |  |
| kernel_cache | key       miss/512 | ns | 4.1000 | 4.1200 | +0.02 | +0.5% |  |
| kernel_cache | key       miss/512 | ns | 2.7600 | 2.5600 | -0.2 | -7.2% |  |
| kernel_cache | key       miss/64 | ns | 6.7100 | 6.9100 | +0.2 | +3.0% |  |
| kernel_cache | key       miss/64 | ns | 3.3800 | 3.7000 | +0.32 | +9.5% |  |
| kernel_cache | key       miss/64 | ns | 2.5500 | 1.7700 | -0.78 | -30.6% |  |
| kernel_cache | key       miss/8 | ns | 6.9300 | 7.1200 | +0.19 | +2.7% |  |
| kernel_cache | key       miss/8 | ns | 3.5700 | 3.8000 | +0.23 | +6.4% |  |
| kernel_cache | key       miss/8 | ns | 3.0900 | 2.1600 | -0.93 | -30.1% |  |
| launch_gpu | auto map | ns | 3137.8 | 3122.6 | -15.2 | -0.5% |  |
| launch_gpu | baked | ns | 3054.5 | 2973.0 | -81.5 | -2.7% |  |
| launch_gpu | bound pointer | ns | 2935.5 | 2998.9 | +63.4 | +2.2% |  |
| launch_gpu | bound tensor | ns | 3086.1 | 2996.1 | -90 | -2.9% |  |
| launch_gpu | fixed device map | ns | 2932.4 | 2891.4 | -41 | -1.4% |  |
| launch_gpu | fixed device no-map | ns | 3010.8 | 2916.4 | -94.4 | -3.1% |  |
| launch_gpu | reduced key | ns | 3123.5 | 3091.2 | -32.3 | -1.0% |  |
| launch_gpu | verify off | ns | 2994.9 | 3042.6 | +47.7 | +1.6% |  |
| launch_gpu | verify on | ns | 2913.9 | 3023.9 | +110 | +3.8% | flag |
| launch_host | auto map | ns | 41.6000 | 42.1000 | +0.5 | +1.2% |  |
| launch_host | baked | ns | 39.2000 | 39.5000 | +0.3 | +0.8% |  |
| launch_host | bound pointer | ns | 44.8000 | 44.4000 | -0.4 | -0.9% |  |
| launch_host | bound tensor | ns | 45.6000 | 46.8000 | +1.2 | +2.6% |  |
| launch_host | fixed device map | ns | 42.1000 | 42.7000 | +0.6 | +1.4% |  |
| launch_host | fixed device no-map | ns | 44.0000 | 43.9000 | -0.1 | -0.2% |  |
| launch_host | reduced key | ns | 41.9000 | 41.8000 | -0.1 | -0.2% |  |
| launch_host | verify off | ns | 42.5000 | 42.1000 | -0.4 | -0.9% |  |
| launch_host | verify on | ns | 44.2000 | 43.6000 | -0.6 | -1.4% |  |
| launch_last_key | 16 alternate | ns | 65.4000 | 65.3000 | -0.1 | -0.2% |  |
| launch_last_key | 16 repeat | ns | 61.9000 | 62.5000 | +0.6 | +1.0% |  |
| launch_last_key | 32 alternate | ns | 109.1 | 104.4 | -4.7 | -4.3% |  |
| launch_last_key | 32 repeat | ns | 104.8 | 99.8000 | -5 | -4.8% |  |
| launch_last_key | 4 alternate | ns | 35.9000 | 36.2000 | +0.3 | +0.8% |  |
| launch_last_key | 4 repeat | ns | 33.9000 | 33.6000 | -0.3 | -0.9% |  |
| launch_readme | grid=(0,) intj | us | 0.0300 | 0.0300 | +0 | +0.0% |  |
| launch_readme | grid=(0,) triton | us | 12.9200 | 12.9300 | +0.01 | +0.1% |  |
| launch_readme | grid=(1,) intj | us | 3.1200 | 3.0900 | -0.03 | -1.0% |  |
| launch_readme | grid=(1,) triton | us | 16.4500 | 16.4600 | +0.01 | +0.1% |  |
| launch_sweep | 16 int | ns | 67.9000 | 68.2000 | +0.3 | +0.4% |  |
| launch_sweep | 16 tensor | ns | 65.2000 | 64.4000 | -0.8 | -1.2% |  |
| launch_sweep | 32 int | ns | 112.8 | 106.4 | -6.4 | -5.7% |  |
| launch_sweep | 32 tensor | ns | 114.0 | 111.1 | -2.9 | -2.5% |  |
| launch_sweep | 4 int | ns | 40.5000 | 40.7000 | +0.2 | +0.5% |  |
| launch_sweep | 4 tensor | ns | 39.2000 | 39.3000 | +0.1 | +0.3% |  |

## Diagnostic: interleaved 6af5613 vs task3 (flagged cases)

`/tmp/intj-bench/task3-ab.sh`: a git worktree of `6af5613` and the task3 checkout at
`97ef0ae`, alternated 5 times per case, each run a separate process, `PYTHONPATH=<tree>
INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python <case command>`. Load
average 2.1-3.6. Medians over the 5 pairs; the last column lists task3 minus base, pair by pair.

| case | row | base median | task3 median | delta | per-pair deltas |
|---|---|---:|---:|---:|---|
| hip_module | INTJ same function | 2.947 | 2.899 | -0.0488 (-1.7%) | -0.0275, -0.05, -0.0413, -0.516, +0.0014 |
| hip_module | TVM FFI HIP HSACO | 3.16 | 3.086 | -0.0742 (-2.3%) | -0.0084, -0.0162, -0.105, -0.574, -0.0069 |
| hip_module | Triton same HSACO | 16.04 | 15.89 | -0.1564 (-1.0%) | -0.0419, -0.116, -0.18, -0.251, -0.0792 |
| launch_gpu | verify on | 2932 | 3432 | +500.7 (+17.1%) | +16, +628, -497, +622, +501 |
| launch_gpu | verify off | 2966 | 3018 | +52.3 (+1.8%) | +42.8, +610, -494, +643, +65.1 |
| launch_gpu | fixed device map | 2888 | 2947 | +59.1 (+2.0%) | +65.2, +574, -586, +678, -7.6 |
| launch_gpu | auto map | 3130 | 3118 | -11.5 (-0.4%) | +39.5, +570, -562, +510, -62.3 |
| ffi_paths | INTJ pair direct | 66.4 | 67.4 | +1 (+1.5%) | +0.2, +8.7, +1.3, +0.2, +3.5 |

`launch_gpu` per-process medians (ns), `verify on` / `auto map`:

```text
r0 base 2887.8 / 3078.6   task3 2903.8 / 3118.1
r1 base 2933.7 / 3129.6   task3 3562.2 / 3699.5
r2 base 3470.4 / 3662.4   task3 2973.3 / 3100.9
r3 base 2874.5 / 3119.9   task3 3496.4 / 3629.5
r4 base 2931.8 / 3155.8   task3 3432.5 / 3093.5
```


## Raw outputs

### task3

#### round0_ffi_compare
```
commit=97ef0aee3d02754ad0a48250b9c32eaf37b6df6f
started=2026-09-26T23:54:31+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_ffi_compare.py --iters 1000 --batches 9
iters=1000 batches=9 unit=us/call (host-only, no timed sync)
FFI packed nop             median=0.1446 samples=[0.150271, 0.14340799999999998, 0.14460900000000002, 0.14602099999999998, 0.14697200000000002, 0.14387200000000003, 0.145598, 0.142626, 0.141371]
FFI typed nop              median=0.1501 samples=[0.147414, 0.154494, 0.1477, 0.150147, 0.15301599999999999, 0.152302, 0.148469, 0.151161, 0.145327]
FFI empty kernel           median=3.2498 samples=[3.727275, 3.413358, 3.4265410000000003, 3.19556, 3.4454450000000003, 3.2349740000000002, 3.2498090000000004, 3.2338560000000003, 3.237908]
INTJ empty kernel          median=2.8619 samples=[3.079065, 3.026311, 3.032927, 2.846733, 2.804153, 2.865456, 2.846205, 2.861918, 2.852346]
INTJ fixed-device kernel   median=2.8453 samples=[3.0114549999999998, 3.0430059999999997, 3.012559, 2.853043, 2.818294, 2.8453150000000003, 2.844958, 2.8374029999999997, 2.841131]
FFI packed nop mixed       median=0.1748 samples=[0.17479599999999998, 0.173126, 0.175886, 0.17566800000000002, 0.17294800000000002, 0.17639, 0.17338599999999998, 0.169678, 0.17645]
FFI typed nop mixed        median=0.1757 samples=[0.17526599999999998, 0.176678, 0.175976, 0.175707, 0.175679, 0.17899700000000002, 0.17610900000000002, 0.175655, 0.174944]
FFI mixed kernel           median=3.2818 samples=[3.435833, 3.459127, 3.281819, 3.29979, 3.25105, 3.3281199999999997, 3.2817, 3.278255, 3.26971]
INTJ mixed kernel          median=2.8750 samples=[3.041477, 3.045674, 2.881471, 2.8535999999999997, 2.808944, 2.967244, 2.8594899999999996, 2.874976, 2.849026]
INTJ fixed mixed kernel    median=2.8747 samples=[3.103589, 3.098312, 2.939015, 2.942287, 2.8627689999999997, 2.8451779999999998, 2.830406, 2.874672, 2.833694]
elapsed_ns=12396811990
exit_status=0
ended=2026-09-26T23:54:44+08:00
```

#### round0_ffi_paths
```
commit=97ef0aee3d02754ad0a48250b9c32eaf37b6df6f
started=2026-09-26T23:55:21+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_intj_ffi_paths.py --mode all --iters 1000 --batches 9
iters=1000 batches=9 unit=ns/call; no_gpu=True
mode=callback
callback_count=9100 (100 warmup + 9000 unique misses)
INTJ cold compile callback + cache  median_ns=409.2 samples_ns=[484.687, 498.945, 384.596, 638.894, 369.639, 393.148, 399.13, 409.169, 936.498]
INTJ hot call (no callback)         median_ns=38.7 samples_ns=[46.765, 39.427, 38.006, 38.237, 38.739, 38.573, 38.836, 38.661, 38.047]
INTJ 3-tensor host-only nop         median_ns=33.8 samples_ns=[41.827, 34.079, 33.841, 33.72, 33.725, 34.488, 33.782, 33.722, 33.806]
mode=kwargs
INTJ direct positional              median_ns=64.7 samples_ns=[67.236, 64.693, 64.62, 64.533, 64.642, 64.507, 74.055, 64.83, 64.865]
INTJ adapter positional             median_ns=88.8 samples_ns=[88.811, 88.127, 88.136, 87.416, 88.081, 88.909, 89.072, 89.649, 89.398]
INTJ adapter kwargs                 median_ns=107.8 samples_ns=[109.812, 108.32, 109.205, 106.704, 107.303, 107.811, 109.637, 106.948, 107.363]
INTJ adapter defaults               median_ns=86.7 samples_ns=[91.996, 86.848, 86.47, 86.638, 86.74, 86.755, 86.621, 87.739, 86.44]
INTJ FFI wrapper positional         median_ns=101.9 samples_ns=[106.001, 105.263, 101.387, 102.196, 100.413, 102.223, 100.701, 101.903, 101.814]
INTJ FFI wrapper kwargs             median_ns=124.1 samples_ns=[126.536, 124.601, 127.729, 123.922, 124.056, 124.24, 123.394, 123.769, 123.595]
INTJ native kwargs: unsupported (verified TypeError)
mode=dataclass
INTJ pair direct                    median_ns=66.0 samples_ns=[74.349, 66.295, 65.959, 66.779, 65.815, 65.756, 66.011, 65.75, 66.021]
INTJ pair prebuilt *tuple           median_ns=139.1 samples_ns=[138.563, 139.395, 144.732, 138.02, 140.195, 138.936, 139.128, 140.188, 137.463]
FFI unpack Pair only                median_ns=164.8 samples_ns=[169.338, 164.76, 163.964, 163.728, 164.852, 164.77, 167.078, 164.421, 163.278]
INTJ pair manual unpack             median_ns=172.5 samples_ns=[172.487, 172.504, 182.218, 185.812, 172.826, 169.849, 170.323, 169.984, 175.187]
INTJ pair FFI unpack                median_ns=311.8 samples_ns=[312.361, 312.204, 316.947, 311.449, 311.37, 310.566, 317.981, 310.925, 311.835]
INTJ pair stdlib astuple            median_ns=1153.0 samples_ns=[1198.086, 1152.879, 1155.412, 1158.919, 1166.921, 1152.307, 1145.052, 1150.581, 1153.015]
INTJ config direct                  median_ns=80.8 samples_ns=[79.857, 79.847, 79.839, 80.947, 80.8, 79.281, 84.922, 80.928, 81.161]
INTJ config FFI unpack              median_ns=316.8 samples_ns=[318.246, 316.949, 318.783, 310.609, 312.507, 316.769, 313.496, 310.364, 317.907]
INTJ native dataclass unpack: unsupported (verified TypeError)
elapsed_ns=9051991751
exit_status=0
ended=2026-09-26T23:55:30+08:00
```

#### round0_ffi_sweep
```
commit=97ef0aee3d02754ad0a48250b9c32eaf37b6df6f
started=2026-09-26T23:54:44+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_ffi_compare.py --sweep --iters 1000 --batches 9
counts=[0, 3, 5, 8, 16, 32, 64] iters=1000 batches=9 unit=us/call (host-only, no timed sync); INTJ device-bound, no specialization key
args= 0 FFI packed nop           median=0.1167 samples=[0.121143, 0.11670900000000001, 0.11573900000000001, 0.116758, 0.11651, 0.117725, 0.113467, 0.116816, 0.11628100000000001]
args= 0 FFI typed nop            median=0.1195 samples=[0.119506, 0.123266, 0.118427, 0.1207, 0.11810999999999999, 0.123095, 0.118245, 0.11941800000000001, 0.121462]
args= 0 FFI empty kernel         median=2.0952 samples=[1.7639369999999999, 2.506148, 2.092777, 2.881799, 2.086363, 2.735709, 2.09516, 2.8722890000000003, 2.082131]
args= 0 INTJ static_compile kernel median=3.3310 samples=[3.291505, 3.342228, 3.321543, 3.350017, 3.31194, 3.3363270000000003, 3.3309699999999998, 3.329587, 3.333862]
args= 0 INTJ runtime_shim kernel median=3.1875 samples=[3.18747, 3.142578, 3.402096, 3.137147, 3.4097779999999998, 3.176391, 3.453035, 3.14867, 3.38959]
args= 3 FFI packed nop           median=0.1441 samples=[0.14355600000000002, 0.14407599999999998, 0.144137, 0.144964, 0.14552500000000002, 0.145807, 0.14404, 0.143535, 0.141571]
args= 3 FFI typed nop            median=0.1488 samples=[0.14783600000000002, 0.146544, 0.148989, 0.15126699999999998, 0.148758, 0.156183, 0.147166, 0.152642, 0.148703]
args= 3 FFI empty kernel         median=3.7461 samples=[3.74609, 4.251068, 3.734687, 4.193566, 3.7296750000000003, 4.286265, 3.715239, 3.8194470000000003, 3.669744]
args= 3 INTJ static_compile kernel median=3.4893 samples=[3.317822, 3.5081320000000003, 3.4502330000000003, 3.51765, 3.385808, 3.517489, 3.4358449999999996, 3.4893170000000002, 3.508426]
args= 3 INTJ runtime_shim kernel median=3.3836 samples=[3.303878, 3.139834, 3.466408, 3.1537689999999996, 3.420534, 3.164667, 3.466439, 3.383648, 3.48941]
args= 5 FFI packed nop           median=0.1735 samples=[0.18054900000000002, 0.173546, 0.175809, 0.173221, 0.173985, 0.173951, 0.172185, 0.17237200000000003, 0.17133199999999998]
args= 5 FFI typed nop            median=0.1771 samples=[0.179584, 0.17710499999999998, 0.175358, 0.176954, 0.173724, 0.18127500000000002, 0.181052, 0.17710499999999998, 0.175397]
args= 5 FFI empty kernel         median=3.8732 samples=[3.750652, 4.401692, 3.8091939999999997, 4.348684, 3.873212, 4.318126, 3.8263510000000003, 4.329129, 3.819773]
args= 5 INTJ static_compile kernel median=3.5074 samples=[3.332786, 3.553604, 3.4837979999999997, 3.584008, 3.432043, 3.5074029999999996, 3.491092, 3.6096179999999998, 3.526993]
args= 5 INTJ runtime_shim kernel median=3.5657 samples=[3.3829119999999997, 3.515857, 3.747231, 3.579638, 3.552679, 3.503588, 3.5657170000000002, 3.605617, 3.8649180000000003]
args= 8 FFI packed nop           median=0.2066 samples=[0.208227, 0.204871, 0.203623, 0.206566, 0.211675, 0.205574, 0.20491900000000002, 0.206824, 0.206955]
args= 8 FFI typed nop            median=0.2083 samples=[0.210768, 0.21057599999999999, 0.207372, 0.213866, 0.20659, 0.21293600000000001, 0.20595, 0.20763, 0.208338]
args= 8 FFI empty kernel         median=3.9325 samples=[3.932496, 4.303592, 3.7347770000000002, 4.344499, 3.7439340000000003, 4.4123540000000006, 3.725644, 4.422736, 3.802767]
args= 8 INTJ static_compile kernel median=3.6466 samples=[3.423227, 3.563121, 4.128078, 3.61617, 4.261058, 3.653899, 4.191539, 3.6465639999999997, 3.5101660000000003]
args= 8 INTJ runtime_shim kernel median=3.4076 samples=[3.407638, 3.252388, 3.592564, 3.253148, 7.866359999999999, 3.265031, 3.614654, 3.259785, 3.638439]
args=16 FFI packed nop           median=0.3018 samples=[0.293025, 0.305831, 0.313779, 0.293819, 0.660808, 0.301752, 0.30263999999999996, 0.287841, 0.30180799999999997]
args=16 FFI typed nop            median=0.3117 samples=[0.32290199999999997, 0.298657, 0.32282299999999997, 0.30706599999999995, 0.549809, 0.311732, 0.311698, 0.299339, 0.30588299999999996]
args=16 FFI empty kernel         median=5.6114 samples=[4.2920110000000005, 5.70535, 4.19978, 5.631175, 19.902703000000002, 5.6113599999999995, 4.187822, 5.777572, 4.157001]
args=16 INTJ static_compile kernel median=5.4101 samples=[3.890396, 4.746678, 5.687749, 5.27973, 5.809208, 5.251218, 5.705463, 5.410116, 5.862964]
args=16 INTJ runtime_shim kernel median=3.8487 samples=[3.848675, 3.735481, 5.4038900000000005, 3.6612739999999997, 5.349029000000001, 3.69782, 5.431337, 3.673819, 5.524636999999999]
args=32 FFI packed nop           median=0.4915 samples=[0.470437, 0.49737400000000004, 0.476728, 0.488982, 0.509344, 0.48977800000000005, 0.514317, 0.496344, 0.491496]
args=32 FFI typed nop            median=0.5086 samples=[0.511332, 0.5134719999999999, 0.508569, 0.499308, 0.504354, 0.495649, 0.521563, 0.513573, 0.49555200000000005]
args=32 FFI empty kernel         median=4.9972 samples=[4.997248, 5.812456, 4.7707120000000005, 5.88565, 4.810881999999999, 5.914298, 4.7999160000000005, 5.827379, 4.78584]
args=32 INTJ static_compile kernel median=5.4972 samples=[4.074409, 5.463685000000001, 5.800135, 5.400988999999999, 5.674999, 5.497191, 5.555034, 5.404326, 5.5140649999999996]
args=32 INTJ runtime_shim kernel median=4.0754 samples=[3.97378, 4.0693090000000005, 5.462021, 4.075414, 5.542639, 4.05712, 5.511253, 4.068744000000001, 5.701357]
args=64 FFI packed nop           median=0.8390 samples=[0.830434, 0.8469589999999999, 0.839019, 0.878497, 0.842495, 0.836585, 0.876621, 0.830127, 0.836492]
args=64 FFI typed nop            median=0.8783 samples=[0.887158, 0.878265, 0.845502, 0.885066, 0.9035599999999999, 0.8650829999999999, 0.89509, 0.856478, 0.856337]
args=64 FFI empty kernel         median=5.9253 samples=[5.760965000000001, 6.3925730000000005, 5.869968, 6.408315, 5.921145999999999, 6.4437359999999995, 5.925319, 6.396282, 5.890878]
args=64 INTJ static_compile kernel median=6.0296 samples=[5.25283, 6.077673, 5.912775, 6.0296199999999995, 6.080587, 6.10365, 6.023413, 6.085541, 5.9764859999999995]
args=64 INTJ runtime_shim kernel median=5.9380 samples=[4.9816769999999995, 5.93796, 5.848474, 6.0331790000000005, 5.974899000000001, 6.104043, 5.855281, 6.127933, 5.899842]
elapsed_ns=36786171025
exit_status=0
ended=2026-09-26T23:55:21+08:00
```

#### round0_hip_module
```
commit=97ef0aee3d02754ad0a48250b9c32eaf37b6df6f
started=2026-09-26T23:55:30+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_hip_module_launch.py --iters 1000 --batches 9
hsaco_sha256=df3e529e178196ad62f7c07ab8593ede5b8596c76e7c6fbd12a5d947a696e65c kernel=empty_kernel threads=256 shared=0 callback_count=1
iters=1000 batches=9 unit=us/call (host enqueue, no timed sync)
TVM FFI HIP HSACO         median=3.8133 samples=[3.7313400000000003, 3.818256, 3.892788, 3.897109, 3.923128, 3.781424, 3.813316, 3.764839, 3.7762040000000003]
Triton same HSACO         median=16.9436 samples=[17.35321, 16.993399, 16.933593000000002, 16.943627, 17.052575, 16.781841, 17.000816, 16.925258000000003, 16.829902999999998]
INTJ same function        median=3.8571 samples=[3.857101, 3.904006, 3.8936840000000004, 3.920705, 3.863553, 3.799217, 3.80921, 3.802349, 3.748224]
elapsed_ns=5960033015
exit_status=0
ended=2026-09-26T23:55:36+08:00
```

#### round0_kernel_cache
```
commit=97ef0aee3d02754ad0a48250b9c32eaf37b6df6f
started=2026-09-26T23:55:36+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python -m pytest tests/test_kernel_cache.py -s -q

kernel cache, 1-word key, ns per operation

                      intj       tsl      absl
hash_only             1.77      1.35      1.36
hit/1                 1.76      2.12      1.86
hit/8                 2.57      2.14      4.76
hit/64                1.87      2.17      4.97
hit/512               2.54      3.03      5.81
hit_child             2.46      2.45      2.46
miss/1                1.39      1.50      1.76
miss/8                2.40      2.13      3.65
miss/64               2.73      2.16      3.91
miss/512              3.23      2.40      5.17

kernel cache, 2-word key, ns per operation

                      intj       tsl      absl
hash_only             1.42      1.42      1.57
hit/1                 2.18      2.59      1.96
hit/8                 2.55      2.56      5.41
hit/64                2.50      2.92      5.90
hit/512               3.33      4.05      6.79
hit_child             2.46      2.45      2.46
miss/1                1.40      2.32      1.76
miss/8                2.74      1.80      4.22
miss/64               4.29      2.34      4.18
miss/512              3.21      2.73      4.42

kernel cache, 5-word key, ns per operation

                      intj       tsl      absl
hash_only             4.53      3.58      3.58
hit/1                 3.14      3.18      3.42
hit/8                 3.37      3.53     10.02
hit/64                3.48      3.62     10.32
hit/512               6.12      4.92     11.68
hit_child             2.46      2.46      2.45
miss/1                2.03      2.12      1.72
miss/8                3.82      2.16      9.53
miss/64               3.71      1.77      6.92
miss/512              3.93      2.57      7.95
....
4 passed in 29.49s
elapsed_ns=30445498021
exit_status=0
ended=2026-09-26T23:56:06+08:00
```

#### round0_launch_gpu
```
commit=97ef0aee3d02754ad0a48250b9c32eaf37b6df6f
started=2026-09-26T23:53:25+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --iters 1000 --batches 9
mode=gpu; 1000 calls × 9 batches; median ns/call
               path    ns/call       Δ ns       Δ %   build ms   bind ms
           auto map     3247.4       +0.0      +0.0    2396.77         -
        reduced key     3192.9      -54.4      -1.7    2196.76         -
         verify off     3317.7      +70.3      +2.2    2107.78         -
          verify on     3233.6      -13.7      -0.4    2285.69         -
              baked     3215.7      -31.7      -1.0    2085.29         -
       bound tensor     3222.3      -25.1      -0.8       1.25   2016.04
      bound pointer     3205.8      -41.6      -1.3       0.44   2136.19
   fixed device map     3250.9       +3.5      +0.1       0.42   2190.45
fixed device no-map     3226.9      -20.4      -0.6       0.46   2091.72
elapsed_ns=23138944650
exit_status=0
ended=2026-09-26T23:53:49+08:00
```

#### round0_launch_host
```
commit=97ef0aee3d02754ad0a48250b9c32eaf37b6df6f
started=2026-09-26T23:53:49+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --no-gpu --iters 100000 --batches 9
mode=host-only; 100000 calls × 9 batches; median ns/call
               path    ns/call       Δ ns       Δ %   build ms   bind ms
           auto map       43.5       +0.0      +0.0    2300.77         -
        reduced key       42.0       -1.5      -3.4    2137.73         -
         verify off       42.2       -1.3      -2.9    2099.38         -
          verify on       44.2       +0.7      +1.6    2252.54         -
              baked       40.2       -3.3      -7.5    2096.16         -
       bound tensor       46.0       +2.5      +5.7       0.24   1975.87
      bound pointer       44.1       +0.6      +1.4       0.23   2103.62
   fixed device map       43.7       +0.1      +0.3       0.19   2185.49
fixed device no-map       44.7       +1.2      +2.8       0.24   2025.26
elapsed_ns=22090153256
exit_status=0
ended=2026-09-26T23:54:11+08:00
```

#### round0_launch_last_key
```
commit=97ef0aee3d02754ad0a48250b9c32eaf37b6df6f
started=2026-09-26T23:54:28+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --last-key --iters 100000 --batches 9
mode=last-key; host-only; 200000 calls × 9 batches; median ns/call
count     pattern    ns/call
    4      repeat       33.8
    4   alternate       36.0
   16      repeat       62.6
   16   alternate       65.2
   32      repeat       99.8
   32   alternate      104.0
elapsed_ns=3275360458
exit_status=0
ended=2026-09-26T23:54:31+08:00
```

#### round0_launch_readme
```
commit=97ef0aee3d02754ad0a48250b9c32eaf37b6df6f
started=2026-09-26T23:54:11+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --readme --iters 1000 --batches 9
mode=readme; 1000 calls × 9 batches; median
        path   triton us    intj us   speedup
   grid=(1,)       17.14       3.18      5.4x
   grid=(0,)       12.92       0.03    460.5x

torch_access_mode   decode ns    build s
     runtime_shim        94.3       0.75
   static_compile        91.3       0.15
      interpreter       869.7       0.78
elapsed_ns=5461969924
exit_status=0
ended=2026-09-26T23:54:16+08:00
```

#### round0_launch_sweep
```
commit=97ef0aee3d02754ad0a48250b9c32eaf37b6df6f
started=2026-09-26T23:54:16+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --sweep --iters 100000 --batches 9
mode=sweep; host-only; 100000 calls × 9 batches; median ns/call
count   kind    ns/call
    4    int       40.1
    4 tensor       38.8
   16    int       68.1
   16 tensor       64.2
   32    int      106.1
   32 tensor      112.0
elapsed_ns=11990653882
exit_status=0
ended=2026-09-26T23:54:28+08:00
```

#### round1_ffi_compare
```
commit=97ef0aee3d02754ad0a48250b9c32eaf37b6df6f
started=2026-09-26T23:56:23+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_ffi_compare.py --iters 1000 --batches 9
iters=1000 batches=9 unit=us/call (host-only, no timed sync)
FFI packed nop             median=0.1444 samples=[0.14833600000000002, 0.144331, 0.143612, 0.144695, 0.14456200000000002, 0.144421, 0.144157, 0.14407499999999998, 0.14435800000000001]
FFI typed nop              median=0.1493 samples=[0.15390299999999998, 0.151004, 0.14780000000000001, 0.148036, 0.14934899999999998, 0.14909999999999998, 0.14797, 0.149827, 0.149371]
FFI empty kernel           median=3.3673 samples=[3.501779, 3.385618, 3.367306, 3.3751149999999996, 3.261113, 3.341372, 3.195109, 3.3679479999999997, 3.211583]
INTJ empty kernel          median=3.0150 samples=[3.106167, 3.0594409999999996, 3.06246, 2.883977, 2.838269, 2.9563319999999997, 3.01496, 2.968203, 3.026033]
INTJ fixed-device kernel   median=2.8684 samples=[3.003595, 3.0775770000000002, 3.07392, 2.843163, 2.868429, 2.841279, 8.702299, 2.850548, 2.829406]
FFI packed nop mixed       median=0.1742 samples=[0.174914, 0.173579, 0.17446799999999998, 0.173195, 0.175125, 0.17332, 0.301332, 0.173249, 0.174167]
FFI typed nop mixed        median=0.1779 samples=[0.177232, 0.182532, 0.179143, 0.177292, 0.177955, 0.177917, 0.384976, 0.17743899999999999, 0.17638399999999999]
FFI mixed kernel           median=3.4105 samples=[3.3625610000000004, 3.416979, 3.4105390000000004, 3.2648029999999997, 3.2479609999999997, 3.3224769999999997, 11.782311, 3.425922, 3.431556]
INTJ mixed kernel          median=2.9508 samples=[3.340086, 3.081527, 2.9508330000000003, 2.921596, 2.8658, 2.864613, 3.0379430000000003, 2.975898, 2.865841]
INTJ fixed mixed kernel    median=2.9312 samples=[3.11362, 3.107117, 2.9794899999999997, 2.968605, 2.8688670000000003, 2.843658, 2.825485, 2.931179, 2.906535]
elapsed_ns=3826383677
exit_status=0
ended=2026-09-26T23:56:27+08:00
```

#### round1_ffi_paths
```
commit=97ef0aee3d02754ad0a48250b9c32eaf37b6df6f
started=2026-09-26T23:56:31+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_intj_ffi_paths.py --mode all --iters 1000 --batches 9
iters=1000 batches=9 unit=ns/call; no_gpu=True
mode=callback
callback_count=9100 (100 warmup + 9000 unique misses)
INTJ cold compile callback + cache  median_ns=418.2 samples_ns=[434.632, 457.063, 385.059, 3733.505, 384.824, 393.974, 406.08, 418.173, 10094.037]
INTJ hot call (no callback)         median_ns=88.2 samples_ns=[109.854, 84.539, 83.844, 115.573, 88.22, 116.455, 92.848, 87.294, 73.436]
INTJ 3-tensor host-only nop         median_ns=89.1 samples_ns=[122.468, 74.126, 92.115, 104.31, 66.21, 97.606, 78.62, 89.1, 76.562]
mode=kwargs
INTJ direct positional              median_ns=129.2 samples_ns=[243.0, 159.387, 192.531, 129.228, 89.99, 137.726, 84.417, 84.584, 85.739]
INTJ adapter positional             median_ns=127.0 samples_ns=[127.012, 129.242, 126.0, 144.147, 186.002, 145.904, 94.82, 90.125, 90.113]
INTJ adapter kwargs                 median_ns=106.0 samples_ns=[106.857, 113.259, 105.184, 105.385, 104.079, 112.057, 105.896, 106.886, 106.049]
INTJ adapter defaults               median_ns=93.8 samples_ns=[92.858, 92.511, 101.042, 93.788, 93.783, 94.219, 93.832, 93.978, 103.41]
INTJ FFI wrapper positional         median_ns=103.3 samples_ns=[109.226, 111.664, 103.928, 102.309, 103.097, 103.299, 102.548, 103.037, 108.191]
INTJ FFI wrapper kwargs             median_ns=126.5 samples_ns=[126.736, 125.653, 126.681, 126.991, 126.449, 125.886, 125.414, 131.36, 126.496]
INTJ native kwargs: unsupported (verified TypeError)
mode=dataclass
INTJ pair direct                    median_ns=68.0 samples_ns=[71.368, 69.192, 72.474, 68.044, 67.043, 67.464, 67.812, 67.467, 68.428]
INTJ pair prebuilt *tuple           median_ns=139.4 samples_ns=[138.653, 141.352, 139.365, 145.126, 136.695, 139.802, 137.318, 139.984, 138.431]
FFI unpack Pair only                median_ns=170.6 samples_ns=[169.583, 177.01, 169.14, 171.157, 169.676, 170.625, 169.559, 176.015, 173.086]
INTJ pair manual unpack             median_ns=175.0 samples_ns=[175.218, 175.705, 172.09, 175.733, 176.517, 173.937, 172.952, 175.023, 172.916]
INTJ pair FFI unpack                median_ns=317.9 samples_ns=[319.068, 321.004, 320.344, 324.966, 315.872, 317.862, 317.437, 316.492, 311.568]
INTJ pair stdlib astuple            median_ns=1159.5 samples_ns=[1219.724, 1194.852, 1151.256, 1156.529, 1142.394, 1159.526, 1153.219, 1162.727, 1165.851]
INTJ config direct                  median_ns=79.6 samples_ns=[79.968, 81.701, 79.634, 80.624, 79.649, 86.476, 78.135, 79.383, 77.904]
INTJ config FFI unpack              median_ns=318.1 samples_ns=[318.057, 319.251, 321.507, 316.308, 315.152, 322.542, 314.261, 315.884, 320.056]
INTJ native dataclass unpack: unsupported (verified TypeError)
elapsed_ns=2934732183
exit_status=0
ended=2026-09-26T23:56:34+08:00
```

#### round1_ffi_sweep
```
commit=97ef0aee3d02754ad0a48250b9c32eaf37b6df6f
started=2026-09-26T23:56:27+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_ffi_compare.py --sweep --iters 1000 --batches 9
counts=[0, 3, 5, 8, 16, 32, 64] iters=1000 batches=9 unit=us/call (host-only, no timed sync); INTJ device-bound, no specialization key
args= 0 FFI packed nop           median=0.1173 samples=[0.12026600000000001, 0.11624599999999999, 0.121583, 0.121435, 0.116971, 0.119115, 0.117259, 0.116723, 0.117233]
args= 0 FFI typed nop            median=0.1192 samples=[0.122561, 0.121196, 0.118087, 0.12132899999999999, 0.118387, 0.123242, 0.115593, 0.11913, 0.119217]
args= 0 FFI empty kernel         median=1.6151 samples=[1.615079, 1.910622, 1.579562, 1.948666, 1.580248, 1.93519, 1.5756430000000001, 1.983314, 1.576396]
args= 0 INTJ static_compile kernel median=2.6994 samples=[3.145935, 2.700865, 2.701736, 2.695797, 2.692792, 2.698609, 2.69937, 2.6974229999999997, 2.7022]
args= 0 INTJ runtime_shim kernel median=2.6934 samples=[2.912831, 2.678937, 2.709056, 2.656353, 2.7122710000000003, 2.6645149999999997, 2.7022150000000003, 2.657311, 2.693419]
args= 3 FFI packed nop           median=0.1452 samples=[0.14576, 0.145203, 0.1459, 0.143582, 0.143749, 0.146673, 0.144582, 0.14396799999999998, 0.148574]
args= 3 FFI typed nop            median=0.1482 samples=[0.14779, 0.153151, 0.146833, 0.151881, 0.14775200000000002, 0.15292599999999998, 0.146808, 0.149113, 0.14821700000000002]
args= 3 FFI empty kernel         median=3.3712 samples=[3.405672, 3.388725, 3.203333, 3.371223, 3.217632, 3.3864479999999997, 3.201178, 3.388365, 3.207662]
args= 3 INTJ static_compile kernel median=2.9514 samples=[3.0218890000000003, 2.801695, 2.9935479999999997, 2.828064, 2.9877399999999996, 2.8331239999999998, 2.951404, 2.835141, 2.9761480000000002]
args= 3 INTJ runtime_shim kernel median=2.8792 samples=[3.021589, 2.794452, 5.053292000000001, 2.782689, 2.879218, 2.787081, 2.888489, 2.7869639999999998, 2.8916779999999997]
args= 5 FFI packed nop           median=0.1737 samples=[0.17538, 0.17318799999999998, 0.35067200000000004, 0.174405, 0.17422900000000002, 0.173153, 0.173655, 0.173233, 0.17325100000000002]
args= 5 FFI typed nop            median=0.1780 samples=[0.178028, 0.176415, 0.359625, 0.177034, 0.174512, 0.17871700000000001, 0.177217, 0.177975, 0.17999700000000002]
args= 5 FFI empty kernel         median=3.3491 samples=[3.4580569999999997, 3.377417, 14.299289, 3.3491039999999996, 3.2942080000000002, 3.342454, 3.310695, 3.378509, 3.303152]
args= 5 INTJ static_compile kernel median=2.8898 samples=[3.041348, 2.941705, 15.218024999999999, 2.9206350000000003, 2.878535, 2.8741849999999998, 2.88784, 2.859863, 2.889791]
args= 5 INTJ runtime_shim kernel median=2.8778 samples=[3.057986, 3.140318, 3.586501, 2.956512, 2.8571660000000003, 2.8738, 2.875065, 2.877822, 2.863758]
args= 8 FFI packed nop           median=0.2064 samples=[0.202945, 0.20546, 0.208234, 0.208548, 0.206371, 0.20684, 0.207666, 0.204828, 0.205644]
args= 8 FFI typed nop            median=0.2086 samples=[0.207531, 0.214528, 0.205426, 0.209505, 0.208676, 0.208084, 0.208588, 0.20950899999999997, 0.20834899999999998]
args= 8 FFI empty kernel         median=3.4819 samples=[3.5907289999999996, 3.529292, 3.442605, 3.4815970000000003, 3.347693, 3.4819, 3.568505, 3.481611, 3.570712]
args= 8 INTJ static_compile kernel median=3.0642 samples=[3.1026700000000003, 2.9519740000000003, 3.169181, 2.999142, 3.0713209999999997, 3.016217, 3.0642069999999997, 2.924975, 3.065486]
args= 8 INTJ runtime_shim kernel median=2.9753 samples=[3.153864, 2.9425850000000002, 3.095786, 2.9063600000000003, 2.982602, 2.896674, 2.981913, 2.894844, 2.975301]
args=16 FFI packed nop           median=0.2916 samples=[0.293851, 0.293817, 0.291568, 0.29117200000000004, 0.286726, 0.291662, 0.285754, 0.294068, 0.290611]
args=16 FFI typed nop            median=0.3026 samples=[0.302732, 0.30567500000000003, 0.30132400000000004, 0.303173, 0.297702, 0.302615, 0.29695299999999997, 0.30390100000000003, 0.301376]
args=16 FFI empty kernel         median=3.7082 samples=[3.848011, 3.754372, 3.719815, 3.7287489999999996, 3.637321, 3.708155, 3.646401, 3.696821, 3.639148]
args=16 INTJ static_compile kernel median=3.3814 samples=[3.2876350000000003, 3.38142, 3.612896, 3.3769400000000003, 3.551925, 3.3335100000000004, 3.598388, 3.291401, 3.5356520000000002]
args=16 INTJ runtime_shim kernel median=3.2968 samples=[3.296785, 3.1640189999999997, 3.446947, 3.0401599999999998, 3.3757040000000003, 3.031315, 3.333535, 3.04307, 3.35657]
args=32 FFI packed nop           median=0.4809 samples=[0.474609, 0.482858, 0.477279, 0.48144400000000004, 0.478165, 0.480927, 0.48330900000000004, 0.478448, 0.48150099999999996]
args=32 FFI typed nop            median=0.4926 samples=[0.48380399999999996, 0.49611, 0.48121499999999995, 0.49264800000000003, 0.49215800000000004, 0.496067, 0.495484, 0.499231, 0.487075]
args=32 FFI empty kernel         median=4.2654 samples=[4.341756, 4.265696, 4.215823, 4.283136000000001, 4.125214, 4.270486, 4.130243, 4.265419, 4.137611]
args=32 INTJ static_compile kernel median=3.6502 samples=[3.602067, 3.650214, 3.7116599999999997, 3.5792040000000003, 3.671815, 3.561671, 3.651694, 3.5384409999999997, 3.679169]
args=32 INTJ runtime_shim kernel median=3.6115 samples=[3.571335, 3.7541309999999997, 3.722312, 3.427075, 3.618215, 3.429117, 3.662312, 3.431165, 3.6114960000000003]
args=64 FFI packed nop           median=0.8388 samples=[0.825781, 0.865842, 0.8527039999999999, 0.835896, 0.849893, 0.830677, 0.838842, 0.834985, 0.845894]
args=64 FFI typed nop            median=0.8642 samples=[0.864185, 0.901702, 0.869487, 0.8662089999999999, 0.852755, 0.858323, 0.862187, 0.8581219999999999, 0.869439]
args=64 FFI empty kernel         median=5.0619 samples=[5.258157, 5.16822, 5.168102, 5.045457000000001, 5.0461, 5.042138, 5.028613999999999, 5.070098, 5.061867]
args=64 INTJ static_compile kernel median=4.5840 samples=[4.705546999999999, 4.633685000000001, 4.674393, 4.573547, 4.550091, 4.56693, 4.578836, 4.5839859999999994, 4.5921710000000004]
args=64 INTJ runtime_shim kernel median=4.5733 samples=[4.640232, 4.691917, 4.573265, 4.612189, 4.52693, 4.514318, 4.558839, 4.578016, 4.543334]
elapsed_ns=4450255968
exit_status=0
ended=2026-09-26T23:56:31+08:00
```

#### round1_hip_module
```
commit=97ef0aee3d02754ad0a48250b9c32eaf37b6df6f
started=2026-09-26T23:56:34+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_hip_module_launch.py --iters 1000 --batches 9
hsaco_sha256=df3e529e178196ad62f7c07ab8593ede5b8596c76e7c6fbd12a5d947a696e65c kernel=empty_kernel threads=256 shared=0 callback_count=1
iters=1000 batches=9 unit=us/call (host enqueue, no timed sync)
TVM FFI HIP HSACO         median=3.1751 samples=[3.4387510000000003, 3.185703, 3.175129, 3.218855, 3.204796, 3.161931, 3.163962, 3.130605, 3.158719]
Triton same HSACO         median=16.4724 samples=[16.704468000000002, 16.587058000000003, 16.57953, 16.579992999999998, 16.472437000000003, 16.469737000000002, 16.445347, 16.46639, 16.386542000000002]
INTJ same function        median=2.9573 samples=[2.9921729999999997, 3.010446, 2.98116, 3.007633, 2.9573139999999998, 2.9221190000000004, 2.946468, 2.9271570000000002, 2.8671260000000003]
elapsed_ns=3769064044
exit_status=0
ended=2026-09-26T23:56:38+08:00
```

#### round1_kernel_cache
```
commit=97ef0aee3d02754ad0a48250b9c32eaf37b6df6f
started=2026-09-26T23:56:38+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python -m pytest tests/test_kernel_cache.py -s -q

kernel cache, 1-word key, ns per operation

                      intj       tsl      absl
hash_only             1.47      1.87      1.64
hit/1                 1.83      2.04      1.54
hit/8                 1.84      2.14      5.78
hit/64                1.86      2.17      4.95
hit/512               2.54      3.03      5.79
hit_child             2.46      2.46      2.45
miss/1                1.40      1.46      1.38
miss/8                2.42      2.12      3.65
miss/64               2.71      2.11      3.93
miss/512              2.82      2.41      4.40

kernel cache, 2-word key, ns per operation

                      intj       tsl      absl
hash_only             1.42      1.42      1.42
hit/1                 2.18      2.57      2.44
hit/8                 2.18      2.57      5.40
hit/64                2.50      2.92      5.60
hit/512               3.32      4.03      6.80
hit_child             2.46      2.46      3.07
miss/1                1.41      2.30      1.70
miss/8                2.87      1.80      4.68
miss/64               3.34      2.33      4.17
miss/512              4.36      2.99      4.42

kernel cache, 5-word key, ns per operation

                      intj       tsl      absl
hash_only             3.57      3.57      3.57
hit/1                 3.15      3.18      2.83
hit/8                 3.42      3.53     10.00
hit/64                3.45      3.63     10.32
hit/512               4.73      4.92     11.68
hit_child             2.46      2.46      2.45
miss/1                2.01      2.12      1.75
miss/8                6.33      2.15      7.12
miss/64               3.69      2.49      6.91
miss/512              4.22      2.56      7.94
....
4 passed in 25.45s
elapsed_ns=26346146445
exit_status=0
ended=2026-09-26T23:57:04+08:00
```

#### round1_launch_gpu
```
commit=97ef0aee3d02754ad0a48250b9c32eaf37b6df6f
started=2026-09-26T23:56:06+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --iters 1000 --batches 9
mode=gpu; 1000 calls × 9 batches; median ns/call
               path    ns/call       Δ ns       Δ %   build ms   bind ms
           auto map     3589.6       +0.0      +0.0      75.32         -
        reduced key     3736.0     +146.4      +4.1      94.99         -
         verify off     3590.9       +1.3      +0.0       2.43         -
          verify on     3581.6       -7.9      -0.2       2.40         -
              baked     3737.9     +148.4      +4.1       2.43         -
       bound tensor     3706.6     +117.0      +3.3       0.41      2.16
      bound pointer     3719.5     +129.9      +3.6       0.41      2.17
   fixed device map     3453.3     -136.2      -3.8       0.34      2.29
fixed device no-map     3533.6      -56.0      -1.6       0.41      2.12
elapsed_ns=3823571050
exit_status=0
ended=2026-09-26T23:56:10+08:00
```

#### round1_launch_host
```
commit=97ef0aee3d02754ad0a48250b9c32eaf37b6df6f
started=2026-09-26T23:56:10+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --no-gpu --iters 100000 --batches 9
mode=host-only; 100000 calls × 9 batches; median ns/call
               path    ns/call       Δ ns       Δ %   build ms   bind ms
           auto map       41.8       +0.0      +0.0      60.83         -
        reduced key       42.1       +0.3      +0.8       1.92         -
         verify off       42.0       +0.2      +0.6       1.79         -
          verify on       44.1       +2.3      +5.6       1.90         -
              baked       39.2       -2.6      -6.2       1.81         -
       bound tensor       45.9       +4.1      +9.9       0.15      1.70
      bound pointer       45.0       +3.3      +7.8       0.16      1.69
   fixed device map       43.1       +1.4      +3.2       0.99     11.48
fixed device no-map       43.9       +2.1      +5.0       0.16      1.72
elapsed_ns=3071761646
exit_status=0
ended=2026-09-26T23:56:13+08:00
```

#### round1_launch_last_key
```
commit=97ef0aee3d02754ad0a48250b9c32eaf37b6df6f
started=2026-09-26T23:56:20+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --last-key --iters 100000 --batches 9
mode=last-key; host-only; 200000 calls × 9 batches; median ns/call
count     pattern    ns/call
    4      repeat       33.8
    4   alternate       36.2
   16      repeat       62.5
   16   alternate       65.1
   32      repeat      100.0
   32   alternate      104.2
elapsed_ns=3251745241
exit_status=0
ended=2026-09-26T23:56:23+08:00
```

#### round1_launch_readme
```
commit=97ef0aee3d02754ad0a48250b9c32eaf37b6df6f
started=2026-09-26T23:56:13+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --readme --iters 1000 --batches 9
mode=readme; 1000 calls × 9 batches; median
        path   triton us    intj us   speedup
   grid=(1,)       16.91       3.13      5.4x
   grid=(0,)       13.35       0.03    447.9x

torch_access_mode   decode ns    build s
     runtime_shim        91.7       0.02
   static_compile        92.0       0.06
      interpreter       879.6       0.00
elapsed_ns=3844859049
exit_status=0
ended=2026-09-26T23:56:17+08:00
```

#### round1_launch_sweep
```
commit=97ef0aee3d02754ad0a48250b9c32eaf37b6df6f
started=2026-09-26T23:56:17+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --sweep --iters 100000 --batches 9
mode=sweep; host-only; 100000 calls × 9 batches; median ns/call
count   kind    ns/call
    4    int       40.7
    4 tensor       39.4
   16    int       68.1
   16 tensor       64.8
   32    int      106.0
   32 tensor      110.8
elapsed_ns=3065034314
exit_status=0
ended=2026-09-26T23:56:20+08:00
```

#### round2_ffi_compare
```
commit=97ef0aee3d02754ad0a48250b9c32eaf37b6df6f
started=2026-09-26T23:57:22+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_ffi_compare.py --iters 1000 --batches 9
iters=1000 batches=9 unit=us/call (host-only, no timed sync)
FFI packed nop             median=0.1457 samples=[0.151679, 0.14382499999999998, 0.14391800000000002, 0.14441800000000002, 0.145984, 0.145727, 0.149331, 0.146927, 0.14371899999999999]
FFI typed nop              median=0.1501 samples=[0.149249, 0.15165700000000001, 0.147448, 0.151221, 0.151953, 0.15232400000000001, 0.148181, 0.150149, 0.14932900000000002]
FFI empty kernel           median=3.3280 samples=[3.408974, 3.489827, 3.482092, 3.3014699999999997, 3.155013, 3.34368, 3.226527, 3.328029, 3.213283]
INTJ empty kernel          median=2.9626 samples=[3.053916, 3.134461, 3.105347, 2.884804, 2.934866, 3.199216, 2.954469, 2.962581, 2.9552910000000003]
INTJ fixed-device kernel   median=2.9391 samples=[3.054011, 3.1285610000000004, 3.049663, 2.857199, 2.8765419999999997, 2.837122, 2.9896309999999997, 2.84441, 2.939129]
FFI packed nop mixed       median=0.1748 samples=[0.175644, 0.173012, 0.173617, 0.17721199999999998, 0.174557, 0.173673, 0.17478200000000002, 0.17552500000000001, 0.177936]
FFI typed nop mixed        median=0.1769 samples=[0.177397, 0.176347, 0.179593, 0.17652099999999998, 0.175886, 0.176901, 0.17954599999999998, 0.179977, 0.176895]
FFI mixed kernel           median=3.3587 samples=[3.463342, 3.554185, 3.3747190000000002, 3.3837170000000003, 3.180514, 3.357516, 3.3586869999999998, 3.331697, 3.223647]
INTJ mixed kernel          median=2.9770 samples=[3.146826, 3.139851, 2.969845, 2.982364, 2.896477, 2.869873, 2.979469, 2.951247, 2.977011]
INTJ fixed mixed kernel    median=2.9436 samples=[3.1672629999999997, 3.1602770000000002, 3.019103, 3.011484, 2.875896, 2.865361, 2.8874229999999996, 2.806444, 2.943605]
elapsed_ns=3642025733
exit_status=0
ended=2026-09-26T23:57:25+08:00
```

#### round2_ffi_paths
```
commit=97ef0aee3d02754ad0a48250b9c32eaf37b6df6f
started=2026-09-26T23:57:30+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_intj_ffi_paths.py --mode all --iters 1000 --batches 9
iters=1000 batches=9 unit=ns/call; no_gpu=True
mode=callback
callback_count=9100 (100 warmup + 9000 unique misses)
INTJ cold compile callback + cache  median_ns=411.6 samples_ns=[431.45, 460.728, 381.469, 1411.914, 374.954, 390.792, 409.556, 411.617, 2353.739]
INTJ hot call (no callback)         median_ns=38.7 samples_ns=[39.666, 39.831, 37.246, 38.414, 45.806, 38.696, 38.268, 38.848, 37.834]
INTJ 3-tensor host-only nop         median_ns=34.0 samples_ns=[36.215, 34.424, 33.7, 34.085, 33.663, 34.1, 33.508, 34.044, 33.562]
mode=kwargs
INTJ direct positional              median_ns=65.5 samples_ns=[67.673, 65.808, 65.366, 65.519, 65.5, 65.379, 65.501, 65.422, 65.533]
INTJ adapter positional             median_ns=90.5 samples_ns=[93.211, 95.569, 91.071, 90.452, 90.232, 91.114, 89.89, 90.415, 90.433]
INTJ adapter kwargs                 median_ns=110.3 samples_ns=[109.335, 113.222, 107.874, 113.177, 111.766, 110.299, 109.387, 111.162, 109.893]
INTJ adapter defaults               median_ns=90.1 samples_ns=[89.901, 91.206, 90.925, 98.575, 90.502, 90.106, 89.051, 87.992, 88.336]
INTJ FFI wrapper positional         median_ns=106.5 samples_ns=[105.048, 103.001, 103.123, 103.533, 131.17, 106.635, 106.705, 106.548, 106.796]
INTJ FFI wrapper kwargs             median_ns=129.3 samples_ns=[130.277, 129.258, 128.563, 132.564, 127.399, 130.768, 131.475, 128.133, 127.143]
INTJ native kwargs: unsupported (verified TypeError)
mode=dataclass
INTJ pair direct                    median_ns=68.3 samples_ns=[71.043, 68.51, 68.151, 68.302, 67.756, 68.383, 67.924, 70.01, 67.786]
INTJ pair prebuilt *tuple           median_ns=141.2 samples_ns=[139.522, 145.149, 137.999, 142.346, 139.461, 142.328, 139.386, 142.34, 141.231]
FFI unpack Pair only                median_ns=167.3 samples_ns=[167.347, 168.198, 166.802, 168.101, 167.862, 168.294, 161.434, 162.396, 161.611]
INTJ pair manual unpack             median_ns=173.7 samples_ns=[170.844, 174.256, 178.825, 173.666, 172.833, 175.511, 171.497, 179.114, 173.3]
INTJ pair FFI unpack                median_ns=317.3 samples_ns=[317.987, 319.212, 318.044, 316.189, 316.55, 319.064, 312.154, 314.95, 317.308]
INTJ pair stdlib astuple            median_ns=1167.0 samples_ns=[1210.649, 1211.445, 1144.58, 1176.507, 1155.462, 1166.995, 1156.532, 1171.394, 1151.834]
INTJ config direct                  median_ns=80.1 samples_ns=[80.268, 85.001, 78.979, 80.109, 80.309, 80.35, 79.039, 79.158, 79.644]
INTJ config FFI unpack              median_ns=316.8 samples_ns=[316.289, 326.4, 317.86, 318.264, 316.848, 315.227, 313.538, 320.163, 314.924]
INTJ native dataclass unpack: unsupported (verified TypeError)
elapsed_ns=2769473023
exit_status=0
ended=2026-09-26T23:57:32+08:00
```

#### round2_ffi_sweep
```
commit=97ef0aee3d02754ad0a48250b9c32eaf37b6df6f
started=2026-09-26T23:57:25+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_ffi_compare.py --sweep --iters 1000 --batches 9
counts=[0, 3, 5, 8, 16, 32, 64] iters=1000 batches=9 unit=us/call (host-only, no timed sync); INTJ device-bound, no specialization key
args= 0 FFI packed nop           median=0.1149 samples=[0.12161100000000001, 0.11651, 0.117895, 0.11486299999999999, 0.11461400000000001, 0.114667, 0.11380599999999999, 0.114486, 0.11563899999999999]
args= 0 FFI typed nop            median=0.1214 samples=[0.12795600000000001, 0.12185599999999999, 0.121366, 0.122059, 0.11766299999999999, 0.120727, 0.116325, 0.12163, 0.119404]
args= 0 FFI empty kernel         median=1.6445 samples=[1.600592, 1.7863959999999999, 1.4878030000000002, 1.8762260000000002, 1.635856, 1.898701, 1.63984, 1.893848, 1.644472]
args= 0 INTJ static_compile kernel median=2.7005 samples=[3.340093, 2.727052, 2.77, 2.716721, 2.696368, 2.693054, 2.700241, 2.679134, 2.7005149999999998]
args= 0 INTJ runtime_shim kernel median=2.6754 samples=[2.8059250000000002, 2.665395, 2.675402, 2.66791, 2.725128, 2.6714290000000003, 2.709663, 2.6648870000000002, 2.6963649999999997]
args= 3 FFI packed nop           median=0.1457 samples=[0.14698599999999998, 0.146427, 0.14460800000000001, 0.145982, 0.149105, 0.144097, 0.145166, 0.14574, 0.144464]
args= 3 FFI typed nop            median=0.1501 samples=[0.151477, 0.15244300000000002, 0.147953, 0.149998, 0.14824, 0.150136, 0.147455, 0.151254, 0.150701]
args= 3 FFI empty kernel         median=3.1840 samples=[3.238053, 3.183977, 3.1419119999999996, 3.215408, 3.154499, 3.217534, 3.1346640000000003, 3.195625, 3.143543]
args= 3 INTJ static_compile kernel median=2.9022 samples=[2.9021559999999997, 2.833093, 2.95341, 2.868833, 2.9641460000000004, 2.822624, 2.94237, 2.823813, 2.9639949999999997]
args= 3 INTJ runtime_shim kernel median=2.8410 samples=[2.907937, 2.8409920000000004, 2.890833, 2.847516, 2.784433, 2.852424, 2.778378, 2.8244890000000002, 2.7733980000000003]
args= 5 FFI packed nop           median=0.1742 samples=[0.174313, 0.172819, 0.172642, 0.174165, 0.17499299999999998, 0.174323, 0.172619, 0.17693, 0.17329]
args= 5 FFI typed nop            median=0.1784 samples=[0.176591, 0.178748, 0.175231, 0.179268, 0.179855, 0.179228, 0.174483, 0.178391, 0.17749299999999998]
args= 5 FFI empty kernel         median=3.2783 samples=[3.296034, 3.268898, 3.278261, 3.3849940000000003, 3.2387710000000003, 3.344178, 3.210561, 3.363188, 3.238108]
args= 5 INTJ static_compile kernel median=2.8848 samples=[2.89871, 2.875976, 2.884814, 3.1749929999999997, 2.9032199999999997, 2.87714, 2.844496, 2.799904, 2.912761]
args= 5 INTJ runtime_shim kernel median=2.8309 samples=[2.915565, 2.891983, 2.868477, 2.816734, 2.835544, 2.7642469999999997, 2.829114, 2.7767579999999996, 2.830879]
args= 8 FFI packed nop           median=0.2082 samples=[0.20825, 0.207292, 0.214537, 0.209012, 0.206216, 0.20721299999999998, 0.209704, 0.203763, 0.2112]
args= 8 FFI typed nop            median=0.2100 samples=[0.207679, 0.210008, 0.209791, 0.210881, 0.20630400000000002, 0.213134, 0.210393, 0.211446, 0.209571]
args= 8 FFI empty kernel         median=3.4031 samples=[3.475132, 3.40313, 3.3184340000000003, 3.455279, 3.271166, 3.420928, 3.274399, 3.409581, 3.2783189999999998]
args= 8 INTJ static_compile kernel median=3.0076 samples=[3.001741, 2.892414, 3.078824, 3.0076199999999997, 3.0779099999999997, 2.928321, 3.098145, 2.982332, 3.081082]
args= 8 INTJ runtime_shim kernel median=2.9456 samples=[3.058423, 2.858327, 2.9577869999999997, 2.9160909999999998, 2.963947, 2.878955, 2.945596, 2.877973, 2.957625]
args=16 FFI packed nop           median=0.2955 samples=[0.29817899999999997, 0.295529, 0.29368700000000003, 0.29982600000000004, 0.29567200000000005, 0.293285, 0.29987, 0.29544400000000004, 0.29508100000000004]
args=16 FFI typed nop            median=0.3062 samples=[0.304312, 0.309682, 0.306165, 0.305908, 0.309957, 0.312113, 0.306077, 0.312728, 0.304385]
args=16 FFI empty kernel         median=3.6314 samples=[3.694538, 3.656033, 3.516279, 3.631405, 3.528304, 3.632832, 3.509262, 3.640556, 3.5163539999999998]
args=16 INTJ static_compile kernel median=3.3239 samples=[3.226014, 3.31156, 3.4825619999999997, 3.3239360000000002, 3.4698420000000003, 3.282592, 3.407993, 3.282676, 3.4508229999999998]
args=16 INTJ runtime_shim kernel median=3.2357 samples=[3.235719, 3.072715, 3.3485479999999996, 3.056925, 3.363766, 3.027872, 3.3286930000000003, 3.030193, 3.332932]
args=32 FFI packed nop           median=0.4917 samples=[0.472314, 0.48148, 0.491656, 0.49011200000000005, 0.488223, 0.492382, 0.494844, 0.494437, 0.494351]
args=32 FFI typed nop            median=0.5022 samples=[0.496585, 0.504055, 0.504247, 0.508321, 0.494195, 0.502234, 0.493453, 0.501278, 0.502585]
args=32 FFI empty kernel         median=4.1896 samples=[4.2735330000000005, 4.189622999999999, 4.092003, 4.255739999999999, 4.125239, 4.251652, 4.101827, 4.225252, 4.100386]
args=32 INTJ static_compile kernel median=3.6001 samples=[3.4629899999999996, 3.600128, 3.6414299999999997, 3.568696, 3.661082, 3.558917, 3.651487, 3.5257840000000003, 3.664704]
args=32 INTJ runtime_shim kernel median=3.4381 samples=[3.432879, 3.412743, 3.6029899999999997, 3.4380520000000003, 3.626806, 3.430872, 3.615109, 3.413514, 3.6089659999999997]
args=64 FFI packed nop           median=0.8606 samples=[0.8191459999999999, 0.839335, 0.8606, 0.879239, 0.8745599999999999, 0.857293, 0.8666849999999999, 0.851706, 0.861911]
args=64 FFI typed nop            median=0.8883 samples=[0.8661319999999999, 0.90039, 0.867248, 0.905218, 0.8925080000000001, 0.896144, 0.862981, 0.88324, 0.888255]
args=64 FFI empty kernel         median=5.0523 samples=[5.057414, 5.0012479999999995, 5.069921, 5.044642, 5.088588, 5.044466000000001, 5.0455429999999994, 5.052327, 5.053956]
args=64 INTJ static_compile kernel median=4.5969 samples=[4.574773, 4.565026, 4.629166000000001, 4.648967, 4.620064, 4.596941, 4.591305, 4.56435, 4.604078]
args=64 INTJ runtime_shim kernel median=4.5141 samples=[4.398421, 4.4733540000000005, 4.511963, 4.521326, 4.533619, 4.563133, 4.4747330000000005, 4.514079, 4.538289]
elapsed_ns=4197412561
exit_status=0
ended=2026-09-26T23:57:30+08:00
```

#### round2_hip_module
```
commit=97ef0aee3d02754ad0a48250b9c32eaf37b6df6f
started=2026-09-26T23:57:32+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_hip_module_launch.py --iters 1000 --batches 9
hsaco_sha256=df3e529e178196ad62f7c07ab8593ede5b8596c76e7c6fbd12a5d947a696e65c kernel=empty_kernel threads=256 shared=0 callback_count=1
iters=1000 batches=9 unit=us/call (host enqueue, no timed sync)
TVM FFI HIP HSACO         median=3.6552 samples=[3.381834, 3.734629, 3.7647310000000003, 3.692292, 3.6979810000000004, 3.5530839999999997, 3.574111, 3.655184, 3.6220030000000003]
Triton same HSACO         median=16.2511 samples=[16.508920999999997, 16.526366999999997, 16.251124, 16.327728, 16.212996999999998, 16.215091, 16.170931, 16.318006999999998, 16.221479]
INTJ same function        median=3.4477 samples=[3.447731, 3.511474, 3.5188989999999998, 3.49942, 3.472455, 3.381063, 3.375722, 3.408713, 3.347779]
elapsed_ns=3626251110
exit_status=0
ended=2026-09-26T23:57:36+08:00
```

#### round2_kernel_cache
```
commit=97ef0aee3d02754ad0a48250b9c32eaf37b6df6f
started=2026-09-26T23:57:36+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python -m pytest tests/test_kernel_cache.py -s -q

kernel cache, 1-word key, ns per operation

                      intj       tsl      absl
hash_only             1.36      1.36      1.37
hit/1                 1.76      2.04      1.51
hit/8                 1.85      2.14      4.76
hit/64                1.87      2.17      4.98
hit/512               2.55      3.05      5.80
hit_child             2.46      2.46      2.65
miss/1                1.39      1.45      1.38
miss/8                2.44      2.12      3.65
miss/64               2.78      2.11      3.91
miss/512              2.80      2.41      4.40

kernel cache, 2-word key, ns per operation

                      intj       tsl      absl
hash_only             1.42      1.42      1.42
hit/1                 2.18      2.57      1.99
hit/8                 2.76      2.57      5.40
hit/64                2.51      2.92      5.57
hit/512               3.33      4.04      6.81
hit_child             2.79      2.46      2.45
miss/1                1.41      2.32      1.71
miss/8                3.17      2.25      3.63
miss/64               3.36      2.34      4.17
miss/512              3.12      2.72      4.42

kernel cache, 5-word key, ns per operation

                      intj       tsl      absl
hash_only             4.60      3.64      3.58
hit/1                 3.15      4.28      2.84
hit/8                 3.35      3.53     10.01
hit/64                3.60      3.62     10.32
hit/512               4.73      4.91     11.72
hit_child             2.46      2.45      2.46
miss/1                2.32      2.12      1.77
miss/8                3.78      2.17      7.15
miss/64               3.72      2.45      8.99
miss/512              4.15      2.56      7.91
....
4 passed in 25.34s
elapsed_ns=26269249551
exit_status=0
ended=2026-09-26T23:58:02+08:00
```

#### round2_launch_gpu
```
commit=97ef0aee3d02754ad0a48250b9c32eaf37b6df6f
started=2026-09-26T23:57:04+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --iters 1000 --batches 9
mode=gpu; 1000 calls × 9 batches; median ns/call
               path    ns/call       Δ ns       Δ %   build ms   bind ms
           auto map     3083.7       +0.0      +0.0      75.16         -
        reduced key     3127.8      +44.2      +1.4      74.31         -
         verify off     3079.1       -4.6      -0.1       2.20         -
          verify on     2980.1     -103.6      -3.4       2.09         -
              baked     3000.3      -83.4      -2.7       2.04         -
       bound tensor     3023.1      -60.6      -2.0       0.36      2.02
      bound pointer     3030.7      -53.0      -1.7       0.35      2.03
   fixed device map     2906.2     -177.5      -5.8       0.33      1.98
fixed device no-map     2887.2     -196.5      -6.4       0.28      1.98
elapsed_ns=3840072143
exit_status=0
ended=2026-09-26T23:57:08+08:00
```

#### round2_launch_host
```
commit=97ef0aee3d02754ad0a48250b9c32eaf37b6df6f
started=2026-09-26T23:57:08+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --no-gpu --iters 100000 --batches 9
mode=host-only; 100000 calls × 9 batches; median ns/call
               path    ns/call       Δ ns       Δ %   build ms   bind ms
           auto map       41.8       +0.0      +0.0      62.48         -
        reduced key       41.7       -0.1      -0.3       1.96         -
         verify off       42.4       +0.7      +1.6       1.84         -
          verify on       43.9       +2.1      +5.0       1.94         -
              baked       40.6       -1.2      -2.9       1.84         -
       bound tensor       46.4       +4.6     +11.0       0.14      1.71
      bound pointer       68.3      +26.5     +63.4       0.15      1.68
   fixed device map       42.7       +1.0      +2.3       0.11      1.69
fixed device no-map       44.5       +2.7      +6.5       0.25      4.20
elapsed_ns=3068249773
exit_status=0
ended=2026-09-26T23:57:11+08:00
```

#### round2_launch_last_key
```
commit=97ef0aee3d02754ad0a48250b9c32eaf37b6df6f
started=2026-09-26T23:57:18+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --last-key --iters 100000 --batches 9
mode=last-key; host-only; 200000 calls × 9 batches; median ns/call
count     pattern    ns/call
    4      repeat       33.6
    4   alternate       36.2
   16      repeat       62.5
   16   alternate       65.3
   32      repeat       99.8
   32   alternate      104.3
elapsed_ns=3359678011
exit_status=0
ended=2026-09-26T23:57:22+08:00
```

#### round2_launch_readme
```
commit=97ef0aee3d02754ad0a48250b9c32eaf37b6df6f
started=2026-09-26T23:57:11+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --readme --iters 1000 --batches 9
mode=readme; 1000 calls × 9 batches; median
        path   triton us    intj us   speedup
   grid=(1,)       16.80       3.12      5.4x
   grid=(0,)       13.44       0.03    474.5x

torch_access_mode   decode ns    build s
     runtime_shim        93.3       0.02
   static_compile        90.7       0.06
      interpreter       874.8       0.00
elapsed_ns=3804410283
exit_status=0
ended=2026-09-26T23:57:15+08:00
```

#### round2_launch_sweep
```
commit=97ef0aee3d02754ad0a48250b9c32eaf37b6df6f
started=2026-09-26T23:57:15+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --sweep --iters 100000 --batches 9
mode=sweep; host-only; 100000 calls × 9 batches; median ns/call
count   kind    ns/call
    4    int       64.9
    4 tensor       39.4
   16    int       68.1
   16 tensor       64.4
   32    int      106.0
   32 tensor      109.1
elapsed_ns=3069660993
exit_status=0
ended=2026-09-26T23:57:18+08:00
```

### task3-rerun

#### round0_ffi_compare
```
commit=97ef0aee3d02754ad0a48250b9c32eaf37b6df6f
started=2026-09-27T00:00:08+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_ffi_compare.py --iters 1000 --batches 9
iters=1000 batches=9 unit=us/call (host-only, no timed sync)
FFI packed nop             median=0.1437 samples=[0.14859899999999998, 0.14301, 0.143316, 0.14287200000000003, 0.142725, 0.143727, 0.14560599999999999, 0.144043, 0.144899]
FFI typed nop              median=0.1494 samples=[0.148607, 0.151199, 0.148355, 0.15196700000000002, 0.150518, 0.149427, 0.14616800000000002, 0.15259999999999999, 0.147459]
FFI empty kernel           median=3.3087 samples=[3.5316140000000003, 3.4852469999999998, 3.42179, 3.1852139999999998, 3.2207719999999997, 3.339937, 3.190093, 3.308674, 3.205265]
INTJ empty kernel          median=2.9782 samples=[3.198532, 3.0193890000000003, 3.020375, 2.83733, 2.8040540000000003, 2.957086, 2.978169, 2.947791, 2.989081]
INTJ fixed-device kernel   median=2.8345 samples=[3.048481, 3.028129, 3.023617, 2.843863, 2.834469, 2.8228180000000003, 2.814457, 2.824907, 2.809505]
FFI packed nop mixed       median=0.1749 samples=[0.171546, 0.177778, 0.174755, 0.17294800000000002, 0.174864, 0.17302199999999998, 0.175171, 0.175281, 0.17611000000000002]
FFI typed nop mixed        median=0.1768 samples=[0.176838, 0.1768, 0.178406, 0.181828, 0.173953, 0.179111, 0.17278200000000002, 0.178856, 0.172233]
FFI mixed kernel           median=3.3236 samples=[3.437581, 3.5614310000000002, 3.323634, 3.280668, 3.220388, 3.363018, 3.2711390000000002, 3.357241, 3.254135]
INTJ mixed kernel          median=2.8913 samples=[3.1113519999999997, 3.0881399999999997, 2.8907190000000003, 2.891337, 2.7986350000000004, 2.829851, 2.872089, 2.980337, 2.9288600000000002]
INTJ fixed mixed kernel    median=2.9173 samples=[3.058303, 3.1127849999999997, 2.920925, 2.905786, 2.864743, 2.833595, 2.90596, 2.9235770000000003, 2.91735]
elapsed_ns=3704380039
exit_status=0
ended=2026-09-27T00:00:12+08:00
```

#### round0_ffi_paths
```
commit=97ef0aee3d02754ad0a48250b9c32eaf37b6df6f
started=2026-09-27T00:00:16+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_intj_ffi_paths.py --mode all --iters 1000 --batches 9
iters=1000 batches=9 unit=ns/call; no_gpu=True
mode=callback
callback_count=9100 (100 warmup + 9000 unique misses)
INTJ cold compile callback + cache  median_ns=411.3 samples_ns=[447.664, 456.805, 385.534, 1261.317, 374.63, 393.746, 400.198, 411.313, 2141.464]
INTJ hot call (no callback)         median_ns=38.8 samples_ns=[40.089, 38.753, 36.854, 39.113, 37.842, 39.022, 37.918, 38.973, 37.418]
INTJ 3-tensor host-only nop         median_ns=34.0 samples_ns=[36.076, 34.427, 33.908, 34.46, 33.736, 34.118, 33.808, 34.042, 33.678]
mode=kwargs
INTJ direct positional              median_ns=65.0 samples_ns=[67.393, 65.133, 64.995, 64.827, 64.849, 64.996, 64.813, 64.917, 74.514]
INTJ adapter positional             median_ns=89.4 samples_ns=[93.938, 89.592, 90.442, 89.639, 88.951, 88.984, 89.397, 88.795, 89.299]
INTJ adapter kwargs                 median_ns=108.0 samples_ns=[111.76, 108.367, 107.755, 108.626, 109.129, 107.964, 108.026, 106.221, 106.98]
INTJ adapter defaults               median_ns=89.1 samples_ns=[90.578, 96.748, 88.391, 88.794, 89.081, 88.519, 89.33, 89.542, 88.911]
INTJ FFI wrapper positional         median_ns=103.3 samples_ns=[105.961, 103.731, 113.313, 103.289, 102.712, 102.845, 103.049, 103.206, 103.784]
INTJ FFI wrapper kwargs             median_ns=126.5 samples_ns=[127.034, 127.181, 131.026, 126.977, 126.221, 126.502, 124.667, 124.162, 125.175]
INTJ native kwargs: unsupported (verified TypeError)
mode=dataclass
INTJ pair direct                    median_ns=68.6 samples_ns=[72.898, 69.877, 68.68, 66.525, 66.01, 66.917, 68.536, 69.163, 68.631]
INTJ pair prebuilt *tuple           median_ns=140.5 samples_ns=[142.336, 141.83, 140.478, 140.488, 139.297, 140.826, 138.677, 143.583, 138.062]
FFI unpack Pair only                median_ns=165.2 samples_ns=[164.28, 164.81, 162.464, 164.034, 171.816, 170.688, 168.36, 166.84, 165.197]
INTJ pair manual unpack             median_ns=172.4 samples_ns=[172.445, 177.745, 171.739, 172.834, 171.841, 172.526, 171.152, 181.718, 171.247]
INTJ pair FFI unpack                median_ns=316.7 samples_ns=[312.265, 316.73, 316.916, 314.57, 312.115, 320.651, 313.363, 317.348, 317.799]
INTJ pair stdlib astuple            median_ns=1183.7 samples_ns=[1219.848, 1219.578, 1189.185, 1200.962, 1183.739, 1177.038, 1163.313, 1168.651, 1155.381]
INTJ config direct                  median_ns=80.4 samples_ns=[80.365, 80.803, 80.382, 80.605, 80.267, 80.547, 79.73, 80.343, 79.934]
INTJ config FFI unpack              median_ns=318.0 samples_ns=[323.407, 316.624, 314.217, 328.535, 312.964, 319.054, 317.966, 319.607, 317.262]
INTJ native dataclass unpack: unsupported (verified TypeError)
elapsed_ns=2982568122
exit_status=0
ended=2026-09-27T00:00:19+08:00
```

#### round0_ffi_sweep
```
commit=97ef0aee3d02754ad0a48250b9c32eaf37b6df6f
started=2026-09-27T00:00:12+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_ffi_compare.py --sweep --iters 1000 --batches 9
counts=[0, 3, 5, 8, 16, 32, 64] iters=1000 batches=9 unit=us/call (host-only, no timed sync); INTJ device-bound, no specialization key
args= 0 FFI packed nop           median=0.1174 samples=[0.121804, 0.118847, 0.117395, 0.11614400000000001, 0.115818, 0.116471, 0.117779, 0.121395, 0.11609900000000001]
args= 0 FFI typed nop            median=0.1208 samples=[0.120782, 0.12305500000000001, 0.119085, 0.124352, 0.11937, 0.124313, 0.11679, 0.121855, 0.118752]
args= 0 FFI empty kernel         median=1.6808 samples=[1.680815, 1.903926, 1.6222249999999998, 1.820732, 1.571637, 1.848344, 1.591769, 1.890152, 1.6088]
args= 0 INTJ static_compile kernel median=2.7236 samples=[3.119282, 2.712933, 2.7410010000000002, 2.7371410000000003, 2.734563, 2.671729, 2.721128, 2.6736350000000004, 2.723645]
args= 0 INTJ runtime_shim kernel median=2.7188 samples=[2.851331, 2.720274, 2.72486, 2.673777, 2.7210680000000003, 2.678477, 2.7181529999999996, 2.6744630000000003, 2.71879]
args= 3 FFI packed nop           median=0.1451 samples=[0.145958, 0.145052, 0.14657599999999998, 0.145536, 0.14321799999999998, 0.146383, 0.143566, 0.14353200000000002, 0.14453200000000002]
args= 3 FFI typed nop            median=0.1492 samples=[0.150756, 0.14793299999999998, 0.149197, 0.150884, 0.15019300000000002, 0.15093399999999998, 0.147721, 0.146916, 0.148417]
args= 3 FFI empty kernel         median=3.2860 samples=[3.337852, 3.317129, 3.156678, 3.285961, 3.206006, 3.28826, 3.151056, 3.2999769999999997, 3.165982]
args= 3 INTJ static_compile kernel median=2.8531 samples=[2.97429, 2.847353, 2.9823939999999998, 2.832074, 2.842793, 2.809349, 2.959016, 2.853072, 3.001645]
args= 3 INTJ runtime_shim kernel median=2.8471 samples=[2.9467060000000003, 2.794746, 2.88902, 2.788261, 2.847124, 2.79063, 2.88578, 2.8027130000000002, 2.916937]
args= 5 FFI packed nop           median=0.1742 samples=[0.170603, 0.177047, 0.177407, 0.176277, 0.173749, 0.174207, 0.175363, 0.173788, 0.172357]
args= 5 FFI typed nop            median=0.1774 samples=[0.17887899999999998, 0.179197, 0.17537, 0.17741300000000002, 0.175234, 0.18173699999999998, 0.176364, 0.17799199999999998, 0.171799]
args= 5 FFI empty kernel         median=3.2700 samples=[3.3160250000000002, 3.274154, 3.294356, 3.252709, 3.24233, 3.270025, 3.255114, 3.271678, 3.26901]
args= 5 INTJ static_compile kernel median=2.8885 samples=[2.958047, 2.907464, 2.911844, 2.8637979999999996, 2.838593, 2.862272, 2.888474, 2.878681, 2.9115770000000003]
args= 5 INTJ runtime_shim kernel median=2.8761 samples=[2.961464, 2.9221950000000003, 2.863718, 2.8789119999999997, 2.824231, 2.870138, 2.861075, 2.8859090000000003, 2.876057]
args= 8 FFI packed nop           median=0.2060 samples=[0.207038, 0.20839, 0.205965, 0.20949, 0.211251, 0.204956, 0.204793, 0.204738, 0.205604]
args= 8 FFI typed nop            median=0.2097 samples=[0.209749, 0.209831, 0.210167, 0.20924199999999998, 0.208309, 0.210151, 0.209998, 0.20862899999999998, 0.20666]
args= 8 FFI empty kernel         median=3.3956 samples=[3.4716129999999996, 3.3955770000000003, 3.5045659999999996, 3.390371, 3.264113, 3.394601, 3.552504, 3.420126, 3.323306]
args= 8 INTJ static_compile kernel median=3.0154 samples=[3.0366210000000002, 2.9225790000000003, 3.0508960000000003, 2.911133, 3.089096, 2.940269, 3.015421, 2.979984, 3.047227]
args= 8 INTJ runtime_shim kernel median=2.9025 samples=[3.084283, 2.937371, 2.8737179999999998, 2.902496, 2.968292, 2.891942, 2.861004, 2.971985, 2.8885639999999997]
args=16 FFI packed nop           median=0.2928 samples=[0.290862, 0.29580900000000004, 0.293165, 0.291538, 0.28696499999999997, 0.29385, 0.293928, 0.292246, 0.292762]
args=16 FFI typed nop            median=0.3029 samples=[0.299814, 0.306382, 0.302947, 0.301205, 0.300721, 0.304342, 0.30115499999999995, 0.305958, 0.30643]
args=16 FFI empty kernel         median=3.6524 samples=[3.738128, 18.546886999999998, 3.586524, 3.652371, 3.553275, 3.6630599999999998, 3.5683719999999997, 3.7218359999999997, 3.5925819999999997]
args=16 INTJ static_compile kernel median=3.4201 samples=[3.277187, 9.207934999999999, 3.42009, 3.292725, 3.492522, 3.332515, 3.426882, 3.364007, 3.482008]
args=16 INTJ runtime_shim kernel median=3.2570 samples=[3.256974, 3.0918229999999998, 3.308031, 3.031834, 3.26945, 3.034334, 3.332973, 3.098745, 3.366419]
args=32 FFI packed nop           median=0.4785 samples=[0.47985, 0.47731999999999997, 0.481686, 0.476777, 0.477622, 0.478498, 0.478291, 0.480514, 0.479008]
args=32 FFI typed nop            median=0.4898 samples=[0.49079, 0.49435399999999996, 0.48665600000000003, 0.492933, 0.480332, 0.489199, 0.48595299999999997, 0.48980599999999996, 0.492993]
args=32 FFI empty kernel         median=4.2269 samples=[4.306614, 4.226894000000001, 4.119890000000001, 4.233233, 4.127114, 4.279719, 4.138043, 4.342823999999999, 4.120794]
args=32 INTJ static_compile kernel median=3.5944 samples=[3.43898, 3.594368, 3.624557, 3.5374, 3.623306, 3.568381, 6.642415, 3.594248, 3.646902]
args=32 INTJ runtime_shim kernel median=3.4960 samples=[3.4187849999999997, 3.4148620000000003, 3.592618, 3.426368, 3.60902, 3.4260520000000003, 18.359752, 3.495962, 3.61637]
args=64 FFI packed nop           median=0.8418 samples=[0.841754, 0.850093, 0.8236180000000001, 0.809476, 0.838047, 0.826093, 1.9721, 0.8453039999999999, 0.853903]
args=64 FFI typed nop            median=0.8633 samples=[0.834927, 0.887644, 0.8609349999999999, 0.863274, 0.84911, 0.86688, 1.607588, 0.8631040000000001, 0.866463]
args=64 FFI empty kernel         median=5.0494 samples=[5.063669, 5.011532, 5.024368000000001, 5.026817, 5.05316, 5.049448, 5.189148, 5.089024, 5.040592999999999]
args=64 INTJ static_compile kernel median=4.5548 samples=[4.520033000000001, 4.526041, 4.545807, 4.555715, 4.563681, 4.550568, 4.6097600000000005, 4.595125, 4.5548090000000006]
args=64 INTJ runtime_shim kernel median=4.5273 samples=[4.463258, 4.527324999999999, 4.481509, 4.558598, 4.520561, 4.543367, 4.575552, 4.591412999999999, 4.52107]
elapsed_ns=4461671984
exit_status=0
ended=2026-09-27T00:00:16+08:00
```

#### round0_hip_module
```
commit=97ef0aee3d02754ad0a48250b9c32eaf37b6df6f
started=2026-09-27T00:00:19+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_hip_module_launch.py --iters 1000 --batches 9
hsaco_sha256=df3e529e178196ad62f7c07ab8593ede5b8596c76e7c6fbd12a5d947a696e65c kernel=empty_kernel threads=256 shared=0 callback_count=1
iters=1000 batches=9 unit=us/call (host enqueue, no timed sync)
TVM FFI HIP HSACO         median=3.2065 samples=[3.4660990000000003, 3.2338400000000003, 3.220555, 3.213098, 3.206493, 3.171692, 3.1730430000000003, 3.181956, 3.1854]
Triton same HSACO         median=16.0391 samples=[16.190177, 16.039092, 16.163176, 16.097494, 16.006997, 16.050997, 16.004283, 15.984347, 15.938336999999999]
INTJ same function        median=3.0039 samples=[3.0039070000000003, 3.016878, 3.033306, 3.001176, 3.0084389999999996, 3.059755, 2.973169, 2.9578629999999997, 2.945463]
elapsed_ns=3934752753
exit_status=0
ended=2026-09-27T00:00:23+08:00
```

#### round0_kernel_cache
```
commit=97ef0aee3d02754ad0a48250b9c32eaf37b6df6f
started=2026-09-27T00:00:23+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python -m pytest tests/test_kernel_cache.py -s -q

kernel cache, 1-word key, ns per operation

                      intj       tsl      absl
hash_only             1.37      1.36      1.37
hit/1                 1.76      2.05      1.54
hit/8                 1.85      2.14      4.77
hit/64                1.88      2.20      4.95
hit/512               2.54      3.50      5.79
hit_child             2.46      2.46      2.46
miss/1                1.39      1.52      1.80
miss/8                2.46      2.12      3.65
miss/64               2.77      2.57      3.92
miss/512              2.80      2.44      4.43

kernel cache, 2-word key, ns per operation

                      intj       tsl      absl
hash_only             1.42      1.74      1.42
hit/1                 2.18      2.57      1.98
hit/8                 2.18      2.56      5.37
hit/64                2.51      2.92      5.59
hit/512               4.20      4.04      6.80
hit_child             2.46      2.45      2.46
miss/1                1.42      2.31      1.73
miss/8                2.85      2.13      4.80
miss/64               3.35      2.35      4.17
miss/512              3.34      2.72      4.45

kernel cache, 5-word key, ns per operation

                      intj       tsl      absl
hash_only             4.57      3.56      3.58
hit/1                 3.15      3.18      2.83
hit/8                 5.50      3.53     10.02
hit/64                3.45      3.62     10.33
hit/512               4.73      4.93     11.66
hit_child             2.46      2.46      2.77
miss/1                1.98      2.12      2.13
miss/8                3.83      2.16      7.15
miss/64               3.70      1.78      6.91
miss/512              4.05      2.57      7.92
....
4 passed in 25.66s
elapsed_ns=26631294446
exit_status=0
ended=2026-09-27T00:00:50+08:00
```

#### round0_launch_gpu
```
commit=97ef0aee3d02754ad0a48250b9c32eaf37b6df6f
started=2026-09-26T23:59:51+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --iters 1000 --batches 9
mode=gpu; 1000 calls × 9 batches; median ns/call
               path    ns/call       Δ ns       Δ %   build ms   bind ms
           auto map     3101.6       +0.0      +0.0      75.74         -
        reduced key     3107.6       +6.1      +0.2      72.96         -
         verify off     2968.9     -132.7      -4.3       2.10         -
          verify on     2954.0     -147.5      -4.8       2.04         -
              baked     2918.3     -183.2      -5.9       2.13         -
       bound tensor     2961.1     -140.4      -4.5       0.36      2.03
      bound pointer     2998.9     -102.7      -3.3       0.35      2.00
   fixed device map     2874.4     -227.2      -7.3       0.32      1.96
fixed device no-map     2916.4     -185.2      -6.0       0.28      1.95
elapsed_ns=3792761158
exit_status=0
ended=2026-09-26T23:59:55+08:00
```

#### round0_launch_host
```
commit=97ef0aee3d02754ad0a48250b9c32eaf37b6df6f
started=2026-09-26T23:59:55+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --no-gpu --iters 100000 --batches 9
mode=host-only; 100000 calls × 9 batches; median ns/call
               path    ns/call       Δ ns       Δ %   build ms   bind ms
           auto map       42.1       +0.0      +0.0      62.31         -
        reduced key       41.8       -0.3      -0.7       1.95         -
         verify off       42.4       +0.3      +0.7       1.81         -
          verify on       43.6       +1.5      +3.6       1.91         -
              baked       39.6       -2.5      -6.0       1.81         -
       bound tensor       47.2       +5.1     +12.2       0.15      1.72
      bound pointer       44.6       +2.5      +5.9       0.15      1.70
   fixed device map       42.7       +0.6      +1.5       0.12      1.67
fixed device no-map       44.6       +2.5      +5.9       0.15      1.65
elapsed_ns=2966052174
exit_status=0
ended=2026-09-26T23:59:58+08:00
```

#### round0_launch_last_key
```
commit=97ef0aee3d02754ad0a48250b9c32eaf37b6df6f
started=2026-09-27T00:00:05+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --last-key --iters 100000 --batches 9
mode=last-key; host-only; 200000 calls × 9 batches; median ns/call
count     pattern    ns/call
    4      repeat       33.6
    4   alternate       36.2
   16      repeat       62.5
   16   alternate       65.1
   32      repeat      100.1
   32   alternate      104.6
elapsed_ns=3417378371
exit_status=0
ended=2026-09-27T00:00:08+08:00
```

#### round0_launch_readme
```
commit=97ef0aee3d02754ad0a48250b9c32eaf37b6df6f
started=2026-09-26T23:59:58+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --readme --iters 1000 --batches 9
mode=readme; 1000 calls × 9 batches; median
        path   triton us    intj us   speedup
   grid=(1,)       16.46       3.06      5.4x
   grid=(0,)       12.93       0.03    459.1x

torch_access_mode   decode ns    build s
     runtime_shim        91.0       0.02
   static_compile        91.2       0.08
      interpreter       937.6       0.00
elapsed_ns=3824571494
exit_status=0
ended=2026-09-27T00:00:01+08:00
```

#### round0_launch_sweep
```
commit=97ef0aee3d02754ad0a48250b9c32eaf37b6df6f
started=2026-09-27T00:00:01+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --sweep --iters 100000 --batches 9
mode=sweep; host-only; 100000 calls × 9 batches; median ns/call
count   kind    ns/call
    4    int       40.8
    4 tensor       39.3
   16    int       68.1
   16 tensor       64.2
   32    int      107.4
   32 tensor      111.1
elapsed_ns=3075253350
exit_status=0
ended=2026-09-27T00:00:05+08:00
```

#### round1_ffi_compare
```
commit=97ef0aee3d02754ad0a48250b9c32eaf37b6df6f
started=2026-09-27T00:01:07+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_ffi_compare.py --iters 1000 --batches 9
iters=1000 batches=9 unit=us/call (host-only, no timed sync)
FFI packed nop             median=0.1457 samples=[0.149469, 0.145078, 0.14787899999999998, 0.145738, 0.144761, 0.14628899999999997, 0.145244, 0.14551, 0.14658500000000002]
FFI typed nop              median=0.1500 samples=[0.147772, 0.14999600000000002, 0.149227, 0.151471, 0.15160300000000002, 0.154887, 0.14962899999999998, 0.1514, 0.147841]
FFI empty kernel           median=3.3395 samples=[3.528632, 3.3959740000000003, 3.379648, 3.309825, 3.134475, 3.339509, 3.1651350000000003, 3.3712600000000004, 3.210355]
INTJ empty kernel          median=2.9852 samples=[3.090409, 3.069473, 3.064142, 2.848704, 2.925093, 2.927999, 2.985245, 2.965363, 3.019122]
INTJ fixed-device kernel   median=2.9311 samples=[3.077594, 3.133888, 3.124342, 2.888254, 2.8460520000000002, 2.875203, 2.8600529999999997, 2.931075, 2.9698789999999997]
FFI packed nop mixed       median=0.1769 samples=[0.176928, 0.176071, 0.174013, 0.177224, 0.17757599999999998, 0.176708, 0.18004900000000001, 0.261925, 0.174697]
FFI typed nop mixed        median=0.1794 samples=[0.17896299999999998, 0.179428, 0.17580600000000002, 0.181939, 0.17836600000000002, 0.17999500000000002, 0.179553, 0.235907, 0.178181]
FFI mixed kernel           median=3.3704 samples=[3.3770949999999997, 3.469904, 3.3689720000000003, 3.3709949999999997, 3.185934, 3.370407, 3.262166, 17.558167, 3.2519899999999997]
INTJ mixed kernel          median=2.8819 samples=[3.142313, 3.0968139999999997, 2.827596, 2.88191, 2.762462, 2.792759, 2.82598, 5.159448, 2.9873060000000002]
INTJ fixed mixed kernel    median=2.8705 samples=[3.1072710000000003, 3.252409, 2.9031819999999997, 2.8958180000000002, 2.824549, 2.790315, 2.8109360000000003, 2.870542, 2.7863870000000004]
elapsed_ns=3768808258
exit_status=0
ended=2026-09-27T00:01:11+08:00
```

#### round1_ffi_paths
```
commit=97ef0aee3d02754ad0a48250b9c32eaf37b6df6f
started=2026-09-27T00:01:16+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_intj_ffi_paths.py --mode all --iters 1000 --batches 9
iters=1000 batches=9 unit=ns/call; no_gpu=True
mode=callback
callback_count=9100 (100 warmup + 9000 unique misses)
INTJ cold compile callback + cache  median_ns=414.9 samples_ns=[429.572, 460.2, 378.981, 1243.767, 374.796, 387.056, 399.741, 414.93, 2080.516]
INTJ hot call (no callback)         median_ns=38.4 samples_ns=[39.927, 39.241, 37.457, 38.416, 37.379, 38.449, 37.782, 38.766, 37.477]
INTJ 3-tensor host-only nop         median_ns=34.1 samples_ns=[36.596, 34.794, 34.13, 34.662, 33.858, 34.147, 33.811, 34.223, 33.836]
mode=kwargs
INTJ direct positional              median_ns=64.9 samples_ns=[67.31, 65.815, 71.251, 64.909, 64.907, 64.807, 64.894, 65.312, 64.814]
INTJ adapter positional             median_ns=90.7 samples_ns=[90.535, 90.347, 90.229, 90.366, 90.681, 91.029, 94.798, 90.666, 90.948]
INTJ adapter kwargs                 median_ns=107.0 samples_ns=[109.789, 107.046, 108.037, 105.322, 106.937, 106.692, 111.934, 107.096, 106.198]
INTJ adapter defaults               median_ns=88.7 samples_ns=[88.94, 88.177, 89.005, 88.848, 88.611, 88.514, 88.63, 88.704, 96.649]
INTJ FFI wrapper positional         median_ns=103.3 samples_ns=[103.265, 106.601, 105.709, 104.193, 103.008, 102.662, 102.439, 103.029, 105.862]
INTJ FFI wrapper kwargs             median_ns=126.9 samples_ns=[127.582, 126.357, 126.943, 127.047, 127.211, 125.829, 126.319, 128.647, 126.301]
INTJ native kwargs: unsupported (verified TypeError)
mode=dataclass
INTJ pair direct                    median_ns=69.0 samples_ns=[71.296, 69.806, 69.029, 69.274, 68.622, 73.457, 65.828, 66.726, 65.802]
INTJ pair prebuilt *tuple           median_ns=138.2 samples_ns=[138.183, 141.934, 138.127, 141.389, 138.178, 150.533, 136.134, 139.46, 136.307]
FFI unpack Pair only                median_ns=162.9 samples_ns=[164.552, 165.072, 165.545, 162.91, 161.419, 161.982, 161.778, 162.322, 166.022]
INTJ pair manual unpack             median_ns=173.6 samples_ns=[173.337, 173.793, 172.789, 174.631, 173.138, 178.183, 172.474, 176.319, 173.628]
INTJ pair FFI unpack                median_ns=315.9 samples_ns=[310.348, 319.415, 312.912, 316.062, 316.799, 315.854, 315.718, 319.827, 314.723]
INTJ pair stdlib astuple            median_ns=1162.3 samples_ns=[1200.112, 1200.411, 1139.75, 1179.113, 1148.954, 1166.879, 1152.229, 1162.295, 1135.76]
INTJ config direct                  median_ns=80.4 samples_ns=[79.047, 80.341, 80.385, 80.894, 79.015, 81.929, 79.561, 80.573, 80.384]
INTJ config FFI unpack              median_ns=320.5 samples_ns=[313.179, 320.843, 314.629, 323.623, 316.093, 320.517, 322.218, 322.632, 318.613]
INTJ native dataclass unpack: unsupported (verified TypeError)
elapsed_ns=2849482490
exit_status=0
ended=2026-09-27T00:01:18+08:00
```

#### round1_ffi_sweep
```
commit=97ef0aee3d02754ad0a48250b9c32eaf37b6df6f
started=2026-09-27T00:01:11+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_ffi_compare.py --sweep --iters 1000 --batches 9
counts=[0, 3, 5, 8, 16, 32, 64] iters=1000 batches=9 unit=us/call (host-only, no timed sync); INTJ device-bound, no specialization key
args= 0 FFI packed nop           median=0.1164 samples=[0.120944, 0.118848, 0.12045099999999999, 0.116373, 0.116211, 0.114162, 0.11511199999999999, 0.116121, 0.118242]
args= 0 FFI typed nop            median=0.1200 samples=[0.122377, 0.11974299999999999, 0.119003, 0.119975, 0.119589, 0.120533, 0.116863, 0.12129300000000001, 0.120448]
args= 0 FFI empty kernel         median=2.1217 samples=[1.694509, 2.465539, 2.015504, 2.854624, 2.1216880000000002, 2.900141, 2.068873, 2.923812, 2.0479670000000003]
args= 0 INTJ static_compile kernel median=3.2393 samples=[3.2250680000000003, 3.207838, 3.2190149999999997, 11.661997999999999, 3.2688249999999996, 3.900166, 3.256092, 3.2392890000000003, 3.218869]
args= 0 INTJ runtime_shim kernel median=3.1833 samples=[3.166651, 2.996388, 3.183342, 14.825616, 3.314548, 3.013131, 3.268036, 3.010632, 3.22765]
args= 3 FFI packed nop           median=0.1448 samples=[0.14509, 0.14433600000000002, 0.14481, 0.246506, 0.14438900000000002, 0.14797200000000002, 0.142934, 0.142816, 0.145785]
args= 3 FFI typed nop            median=0.1510 samples=[0.150975, 0.15189599999999998, 0.14637, 0.249139, 0.14544900000000002, 0.153651, 0.146478, 0.15121700000000002, 0.14981]
args= 3 FFI empty kernel         median=3.7228 samples=[3.6830439999999998, 4.220057, 3.656004, 14.423376000000001, 3.722827, 3.838117, 3.7033229999999997, 3.8172930000000003, 3.6774229999999997]
args= 3 INTJ static_compile kernel median=3.4189 samples=[3.2409229999999996, 3.320138, 3.306408, 3.3979160000000004, 3.418934, 3.474518, 3.439421, 3.459133, 3.4368760000000003]
args= 3 INTJ runtime_shim kernel median=3.3999 samples=[3.268943, 3.1158919999999997, 3.3998690000000003, 3.149922, 3.4307939999999997, 3.4007240000000003, 3.434831, 3.3366860000000003, 3.4414659999999997]
args= 5 FFI packed nop           median=0.1748 samples=[0.17371, 0.173354, 0.176475, 0.174239, 0.17741900000000002, 0.175876, 0.174764, 0.17485599999999998, 0.17441900000000002]
args= 5 FFI typed nop            median=0.1792 samples=[0.177666, 0.180068, 0.179003, 0.182094, 0.172769, 0.182702, 0.18065, 0.179156, 0.176778]
args= 5 FFI empty kernel         median=3.7898 samples=[3.7219290000000003, 4.193631, 3.730433, 4.1881189999999995, 3.789755, 4.3074129999999995, 3.7487440000000003, 4.185823999999999, 3.731059]
args= 5 INTJ static_compile kernel median=3.4159 samples=[3.261792, 3.440023, 3.321739, 3.342732, 3.411882, 3.470113, 3.458583, 3.4159050000000004, 3.454681]
args= 5 INTJ runtime_shim kernel median=3.3854 samples=[3.374661, 3.385224, 3.385448, 3.353434, 4.069566, 3.416426, 3.488802, 3.372329, 3.4921170000000004]
args= 8 FFI packed nop           median=0.2074 samples=[0.208798, 0.207414, 0.208944, 0.209298, 0.205068, 0.205981, 0.205403, 0.206197, 0.20866200000000001]
args= 8 FFI typed nop            median=0.2089 samples=[0.21184999999999998, 0.21446700000000002, 0.20777400000000001, 0.21116, 0.20784, 0.208625, 0.21093299999999998, 0.208877, 0.208403]
args= 8 FFI empty kernel         median=3.9589 samples=[3.958873, 4.251575, 3.675122, 4.217866, 3.774513, 4.446045, 3.798178, 4.297362, 3.771639]
args= 8 INTJ static_compile kernel median=3.5964 samples=[3.3656930000000003, 3.506542, 4.081364, 3.621641, 4.188007, 3.6589490000000002, 3.528616, 3.59637, 3.472773]
args= 8 INTJ runtime_shim kernel median=3.4078 samples=[3.407827, 3.248692, 3.478411, 3.250496, 3.524707, 3.244499, 3.582591, 3.2494769999999997, 3.558941]
args=16 FFI packed nop           median=0.2928 samples=[0.296423, 0.29936, 0.296846, 0.290551, 0.289691, 0.292499, 0.29283800000000004, 0.288126, 0.29789299999999996]
args=16 FFI typed nop            median=0.3053 samples=[0.311541, 0.309488, 0.306842, 0.30189499999999997, 0.301465, 0.30565499999999995, 0.305299, 0.30232600000000004, 0.303515]
args=16 FFI empty kernel         median=4.6879 samples=[4.2278590000000005, 5.871724, 4.687884, 5.89575, 4.648837, 5.8183810000000005, 4.165178, 5.7357380000000004, 4.108661]
args=16 INTJ static_compile kernel median=5.3075 samples=[3.8033, 5.307468, 5.487196, 4.964249, 5.131600000000001, 5.334274000000001, 5.634238, 5.043918, 5.77961]
args=16 INTJ runtime_shim kernel median=3.7831 samples=[3.78309, 3.5587220000000004, 5.374388, 3.578998, 4.870468, 3.551243, 5.551906, 3.586973, 5.551829]
args=32 FFI packed nop           median=0.4768 samples=[0.468546, 0.489212, 0.477651, 0.47457499999999997, 0.473985, 0.476841, 0.473319, 0.478875, 0.48004199999999997]
args=32 FFI typed nop            median=0.4956 samples=[0.49721499999999996, 0.524679, 0.494515, 0.496133, 0.48131799999999997, 0.495591, 0.486442, 0.490539, 0.497836]
args=32 FFI empty kernel         median=5.1059 samples=[4.777009, 5.73828, 4.644297, 5.960085, 5.105865, 5.893495, 4.643438, 5.785296000000001, 4.648445]
args=32 INTJ static_compile kernel median=5.5969 samples=[4.1173969999999995, 5.521917, 5.596894, 5.291658999999999, 6.043091, 5.701423, 5.7321040000000005, 5.3873109999999995, 5.642805]
args=32 INTJ runtime_shim kernel median=4.0860 samples=[4.086049, 4.067985999999999, 5.555981, 4.012274, 5.922413, 4.017874, 5.538494, 3.9965479999999998, 5.525062]
args=64 FFI packed nop           median=0.8475 samples=[0.826766, 0.870103, 0.858961, 0.874946, 0.855915, 0.838806, 0.847514, 0.817351, 0.838499]
args=64 FFI typed nop            median=0.8675 samples=[0.8688830000000001, 0.909285, 0.872726, 0.89522, 0.841785, 0.867456, 0.855058, 0.850032, 0.856441]
args=64 FFI empty kernel         median=5.8253 samples=[5.7945020000000005, 6.4238800000000005, 5.825272, 6.413798, 5.8046750000000005, 6.412594, 5.809687, 6.455977, 5.802631]
args=64 INTJ static_compile kernel median=6.0535 samples=[5.316112, 6.150863999999999, 6.0874250000000005, 6.05867, 5.988474, 6.033562, 6.052995, 6.053525, 6.087693]
args=64 INTJ runtime_shim kernel median=5.9786 samples=[5.212273, 6.1005069999999995, 5.854664, 6.043356, 5.978628, 6.133055000000001, 5.9561530000000005, 6.071096, 5.9093]
elapsed_ns=4664029718
exit_status=0
ended=2026-09-27T00:01:16+08:00
```

#### round1_hip_module
```
commit=97ef0aee3d02754ad0a48250b9c32eaf37b6df6f
started=2026-09-27T00:01:18+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_hip_module_launch.py --iters 1000 --batches 9
hsaco_sha256=df3e529e178196ad62f7c07ab8593ede5b8596c76e7c6fbd12a5d947a696e65c kernel=empty_kernel threads=256 shared=0 callback_count=1
iters=1000 batches=9 unit=us/call (host enqueue, no timed sync)
TVM FFI HIP HSACO         median=3.5912 samples=[3.5911709999999997, 3.6111039999999996, 3.71786, 3.680318, 3.650835, 3.477629, 3.534239, 3.494774, 3.5398009999999998]
Triton same HSACO         median=16.1804 samples=[16.269275, 16.328615, 16.244547, 16.180413, 16.036741, 16.150861, 16.201826, 16.038812, 15.962102999999999]
INTJ same function        median=3.4002 samples=[3.3797829999999998, 3.484441, 3.4965889999999997, 3.527154, 3.4027179999999997, 3.400166, 3.3280250000000002, 3.357658, 3.336157]
elapsed_ns=3831277146
exit_status=0
ended=2026-09-27T00:01:22+08:00
```

#### round1_kernel_cache
```
commit=97ef0aee3d02754ad0a48250b9c32eaf37b6df6f
started=2026-09-27T00:01:22+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python -m pytest tests/test_kernel_cache.py -s -q

kernel cache, 1-word key, ns per operation

                      intj       tsl      absl
hash_only             1.36      1.36      1.36
hit/1                 1.76      2.02      1.49
hit/8                 2.29      2.13      4.77
hit/64                1.88      2.17      4.94
hit/512               2.55      3.04      5.79
hit_child             2.46      2.46      2.45
miss/1                1.39      1.56      2.16
miss/8                2.40      2.12      3.65
miss/64               2.80      2.81      3.92
miss/512              3.61      2.40      4.40

kernel cache, 2-word key, ns per operation

                      intj       tsl      absl
hash_only             1.42      1.42      1.42
hit/1                 2.18      2.57      1.98
hit/8                 2.18      2.58      5.39
hit/64                2.51      2.92      6.66
hit/512               3.33      4.04      6.83
hit_child             2.46      2.45      2.46
miss/1                1.45      2.32      1.77
miss/8                2.76      1.79      3.61
miss/64               3.37      2.34      5.45
miss/512              3.22      2.72      4.37

kernel cache, 5-word key, ns per operation

                      intj       tsl      absl
hash_only             3.56      3.57      3.58
hit/1                 3.13      3.18      2.85
hit/8                 3.52      3.52     10.02
hit/64                4.42      5.79     10.31
hit/512               4.73      4.94     11.67
hit_child             2.46      2.46      2.46
miss/1                2.02      2.12      1.72
miss/8                3.80      2.16      7.12
miss/64               3.71      1.77      9.68
miss/512              4.44      2.56      7.92
....
4 passed in 25.21s
elapsed_ns=26125322967
exit_status=0
ended=2026-09-27T00:01:48+08:00
```

#### round1_launch_gpu
```
commit=97ef0aee3d02754ad0a48250b9c32eaf37b6df6f
started=2026-09-27T00:00:50+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --iters 1000 --batches 9
mode=gpu; 1000 calls × 9 batches; median ns/call
               path    ns/call       Δ ns       Δ %   build ms   bind ms
           auto map     3122.6       +0.0      +0.0      77.54         -
        reduced key     3073.8      -48.8      -1.6      84.17         -
         verify off     3091.3      -31.3      -1.0       2.34         -
          verify on     3047.7      -74.9      -2.4       2.19         -
              baked     3006.0     -116.6      -3.7       2.23         -
       bound tensor     3076.8      -45.8      -1.5       0.30      2.15
      bound pointer     3069.7      -52.9      -1.7       0.37      2.04
   fixed device map     3006.2     -116.4      -3.7       0.33      1.99
fixed device no-map     3018.8     -103.8      -3.3       0.29      1.99
elapsed_ns=3898859950
exit_status=0
ended=2026-09-27T00:00:54+08:00
```

#### round1_launch_host
```
commit=97ef0aee3d02754ad0a48250b9c32eaf37b6df6f
started=2026-09-27T00:00:54+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --no-gpu --iters 100000 --batches 9
mode=host-only; 100000 calls × 9 batches; median ns/call
               path    ns/call       Δ ns       Δ %   build ms   bind ms
           auto map       41.8       +0.0      +0.0      63.35         -
        reduced key       42.1       +0.2      +0.6       2.01         -
         verify off       42.1       +0.3      +0.7       1.98         -
          verify on       43.8       +2.0      +4.8       1.99         -
              baked       39.4       -2.5      -5.9       1.85         -
       bound tensor       46.8       +4.9     +11.8       0.16      1.71
      bound pointer       44.3       +2.5      +5.9       0.15      1.74
   fixed device map       43.8       +1.9      +4.6       0.11      1.71
fixed device no-map       43.9       +2.1      +5.0       0.17      1.70
elapsed_ns=3169427447
exit_status=0
ended=2026-09-27T00:00:57+08:00
```

#### round1_launch_last_key
```
commit=97ef0aee3d02754ad0a48250b9c32eaf37b6df6f
started=2026-09-27T00:01:04+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --last-key --iters 100000 --batches 9
mode=last-key; host-only; 200000 calls × 9 batches; median ns/call
count     pattern    ns/call
    4      repeat       33.6
    4   alternate       36.1
   16      repeat       62.5
   16   alternate       65.3
   32      repeat       99.5
   32   alternate      103.6
elapsed_ns=3425250507
exit_status=0
ended=2026-09-27T00:01:07+08:00
```

#### round1_launch_readme
```
commit=97ef0aee3d02754ad0a48250b9c32eaf37b6df6f
started=2026-09-27T00:00:57+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --readme --iters 1000 --batches 9
mode=readme; 1000 calls × 9 batches; median
        path   triton us    intj us   speedup
   grid=(1,)       16.80       3.09      5.4x
   grid=(0,)       13.02       0.03    461.2x

torch_access_mode   decode ns    build s
     runtime_shim        92.9       0.02
   static_compile        92.9       0.06
      interpreter       875.0       0.00
elapsed_ns=3869465939
exit_status=0
ended=2026-09-27T00:01:01+08:00
```

#### round1_launch_sweep
```
commit=97ef0aee3d02754ad0a48250b9c32eaf37b6df6f
started=2026-09-27T00:01:01+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --sweep --iters 100000 --batches 9
mode=sweep; host-only; 100000 calls × 9 batches; median ns/call
count   kind    ns/call
    4    int       40.7
    4 tensor       39.2
   16    int       68.5
   16 tensor       64.7
   32    int      106.4
   32 tensor      114.7
elapsed_ns=3084759977
exit_status=0
ended=2026-09-27T00:01:04+08:00
```

#### round2_ffi_compare
```
commit=97ef0aee3d02754ad0a48250b9c32eaf37b6df6f
started=2026-09-27T00:02:05+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_ffi_compare.py --iters 1000 --batches 9
iters=1000 batches=9 unit=us/call (host-only, no timed sync)
FFI packed nop             median=0.1460 samples=[0.148906, 0.148451, 0.147894, 0.14724500000000001, 0.14599, 0.14365199999999997, 0.143433, 0.14494900000000002, 0.144206]
FFI typed nop              median=0.1497 samples=[0.14761600000000002, 0.154589, 0.150278, 0.1515, 0.14840899999999999, 0.154715, 0.147506, 0.149689, 0.147777]
FFI empty kernel           median=3.6760 samples=[3.6217669999999997, 4.579721999999999, 3.675986, 4.334597, 3.531702, 4.282045, 3.5402620000000002, 4.3433329999999994, 3.545413]
INTJ empty kernel          median=3.7334 samples=[3.634949, 3.7333600000000002, 4.181675, 3.525826, 3.9594630000000004, 3.5868789999999997, 3.891546, 3.606464, 3.9377370000000003]
INTJ fixed-device kernel   median=3.3520 samples=[3.385844, 3.3860300000000003, 3.540459, 3.248978, 3.5306379999999997, 3.159432, 3.3519989999999997, 3.164232, 3.3519699999999997]
FFI packed nop mixed       median=0.1735 samples=[0.17433600000000002, 0.17326, 0.16961199999999999, 0.172767, 0.173453, 0.176219, 0.177619, 0.17341399999999998, 0.173454]
FFI typed nop mixed        median=0.1770 samples=[0.17597100000000002, 0.17891200000000002, 0.176329, 0.177951, 0.17601, 0.180564, 0.177043, 0.18045599999999998, 0.174774]
FFI mixed kernel           median=3.8623 samples=[3.913329, 4.239885, 3.738705, 4.417828999999999, 3.717058, 3.8623220000000003, 3.682114, 4.355675, 3.638546]
INTJ mixed kernel          median=3.5611 samples=[3.846074, 3.860845, 3.401806, 3.605571, 3.5095970000000003, 3.561106, 3.386997, 3.592158, 3.367918]
INTJ fixed mixed kernel    median=3.5584 samples=[3.835424, 3.907016, 3.558415, 3.638164, 3.5267890000000004, 3.5219270000000003, 3.501719, 3.570163, 3.52753]
elapsed_ns=3867868963
exit_status=0
ended=2026-09-27T00:02:09+08:00
```

#### round2_ffi_paths
```
commit=97ef0aee3d02754ad0a48250b9c32eaf37b6df6f
started=2026-09-27T00:02:14+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_intj_ffi_paths.py --mode all --iters 1000 --batches 9
iters=1000 batches=9 unit=ns/call; no_gpu=True
mode=callback
callback_count=9100 (100 warmup + 9000 unique misses)
INTJ cold compile callback + cache  median_ns=413.5 samples_ns=[426.676, 542.237, 381.238, 1270.356, 376.428, 386.716, 402.5, 413.514, 2124.407]
INTJ hot call (no callback)         median_ns=38.4 samples_ns=[40.21, 39.076, 37.73, 39.522, 37.771, 38.68, 37.687, 38.362, 37.633]
INTJ 3-tensor host-only nop         median_ns=34.3 samples_ns=[36.361, 34.547, 34.771, 34.304, 33.822, 46.404, 33.793, 34.166, 33.789]
mode=kwargs
INTJ direct positional              median_ns=64.7 samples_ns=[85.892, 65.259, 64.823, 64.622, 64.587, 64.638, 64.646, 64.655, 64.957]
INTJ adapter positional             median_ns=93.4 samples_ns=[93.297, 93.79, 92.214, 92.084, 99.188, 94.802, 94.868, 93.411, 90.712]
INTJ adapter kwargs                 median_ns=109.8 samples_ns=[110.207, 109.768, 108.601, 109.306, 116.783, 109.37, 109.615, 110.861, 114.671]
INTJ adapter defaults               median_ns=89.1 samples_ns=[90.141, 88.587, 88.386, 88.15, 88.136, 96.779, 90.959, 89.229, 89.145]
INTJ FFI wrapper positional         median_ns=102.4 samples_ns=[104.403, 104.253, 102.447, 103.214, 102.443, 102.427, 107.253, 102.448, 101.005]
INTJ FFI wrapper kwargs             median_ns=126.6 samples_ns=[126.501, 126.393, 125.339, 126.771, 126.81, 129.338, 125.195, 126.803, 126.629]
INTJ native kwargs: unsupported (verified TypeError)
mode=dataclass
INTJ pair direct                    median_ns=67.7 samples_ns=[69.703, 74.933, 66.554, 67.913, 66.59, 67.732, 66.704, 68.223, 66.782]
INTJ pair prebuilt *tuple           median_ns=137.6 samples_ns=[137.573, 140.77, 136.381, 143.839, 137.167, 139.417, 135.979, 140.534, 136.988]
FFI unpack Pair only                median_ns=164.1 samples_ns=[163.115, 168.305, 162.91, 164.09, 164.319, 168.491, 163.876, 168.541, 163.698]
INTJ pair manual unpack             median_ns=173.5 samples_ns=[172.878, 175.009, 172.322, 174.679, 176.463, 173.98, 170.746, 173.496, 172.705]
INTJ pair FFI unpack                median_ns=319.5 samples_ns=[320.235, 319.653, 317.512, 322.822, 315.765, 317.044, 322.931, 319.478, 315.888]
INTJ pair stdlib astuple            median_ns=1167.9 samples_ns=[1210.884, 1206.69, 1142.823, 1174.299, 1147.103, 1205.287, 1146.239, 1167.872, 1142.5]
INTJ config direct                  median_ns=79.8 samples_ns=[78.877, 80.225, 78.737, 79.792, 81.587, 80.223, 79.049, 80.111, 78.939]
INTJ config FFI unpack              median_ns=317.5 samples_ns=[317.524, 321.503, 317.037, 316.396, 314.503, 322.375, 317.203, 318.767, 319.086]
INTJ native dataclass unpack: unsupported (verified TypeError)
elapsed_ns=2845674037
exit_status=0
ended=2026-09-27T00:02:16+08:00
```

#### round2_ffi_sweep
```
commit=97ef0aee3d02754ad0a48250b9c32eaf37b6df6f
started=2026-09-27T00:02:09+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_ffi_compare.py --sweep --iters 1000 --batches 9
counts=[0, 3, 5, 8, 16, 32, 64] iters=1000 batches=9 unit=us/call (host-only, no timed sync); INTJ device-bound, no specialization key
args= 0 FFI packed nop           median=0.1168 samples=[0.119353, 0.115048, 0.11915300000000001, 0.120323, 0.11770399999999999, 0.116727, 0.11539, 0.11676099999999999, 0.115621]
args= 0 FFI typed nop            median=0.1210 samples=[0.119382, 0.12163, 0.11760599999999999, 0.12165300000000001, 0.12100799999999999, 0.123709, 0.118642, 0.12456, 0.11766700000000001]
args= 0 FFI empty kernel         median=1.6116 samples=[1.611626, 1.870963, 1.5858889999999999, 1.756789, 1.5707529999999998, 1.798958, 1.5760070000000002, 1.7932460000000001, 1.588917]
args= 0 INTJ static_compile kernel median=2.7364 samples=[3.2768960000000003, 2.738449, 2.70881, 2.7363560000000002, 2.729661, 2.731663, 2.730318, 2.74377, 2.754623]
args= 0 INTJ runtime_shim kernel median=2.7056 samples=[2.8792150000000003, 2.688593, 2.7055830000000003, 2.6842620000000004, 2.741977, 2.6942269999999997, 2.7309699999999997, 2.683949, 2.7470909999999997]
args= 3 FFI packed nop           median=0.1441 samples=[0.146643, 0.144126, 0.143309, 0.146802, 0.141822, 0.143435, 0.144548, 0.145343, 0.14338499999999998]
args= 3 FFI typed nop            median=0.1497 samples=[0.150817, 0.14843, 0.14893199999999998, 0.149719, 0.146118, 0.14978, 0.150983, 0.15041, 0.14664]
args= 3 FFI empty kernel         median=3.2814 samples=[3.34231, 3.28419, 3.1915, 3.304303, 3.281362, 3.313318, 3.2497399999999996, 3.2711930000000002, 3.278674]
args= 3 INTJ static_compile kernel median=2.9299 samples=[2.977991, 2.9012890000000002, 2.9495259999999996, 2.9325129999999997, 2.878657, 2.929933, 2.8685419999999997, 2.935368, 2.878024]
args= 3 INTJ runtime_shim kernel median=2.8736 samples=[3.086903, 2.832043, 2.8865990000000004, 2.8427179999999996, 2.8716280000000003, 2.859201, 2.873577, 2.878569, 2.8762779999999997]
args= 5 FFI packed nop           median=0.1733 samples=[0.16945400000000002, 0.173996, 0.173344, 0.17327199999999998, 0.176757, 0.172003, 0.17256, 0.173661, 0.173225]
args= 5 FFI typed nop            median=0.1774 samples=[0.177811, 0.178812, 0.177422, 0.17831899999999998, 0.17724199999999998, 0.17630500000000002, 0.17390899999999998, 0.177673, 0.175845]
args= 5 FFI empty kernel         median=3.2910 samples=[3.370088, 3.291342, 3.3272199999999996, 3.286271, 3.290983, 3.2603910000000003, 3.2844520000000004, 3.262917, 3.2943919999999998]
args= 5 INTJ static_compile kernel median=2.8930 samples=[3.00452, 2.923168, 2.8930219999999998, 2.89352, 2.8784479999999997, 2.89127, 2.8976129999999998, 2.8823440000000002, 2.883817]
args= 5 INTJ runtime_shim kernel median=2.8922 samples=[2.996383, 2.921881, 2.8478719999999997, 2.897316, 2.8492260000000003, 2.8959029999999997, 2.842122, 2.892165, 2.862293]
args= 8 FFI packed nop           median=0.2083 samples=[0.20829, 0.204484, 0.20513800000000001, 0.21016300000000002, 0.211215, 0.209954, 0.207167, 0.21090799999999998, 0.20674]
args= 8 FFI typed nop            median=0.2096 samples=[0.210228, 0.210287, 0.20829, 0.213094, 0.209351, 0.212052, 0.207928, 0.209619, 0.207552]
args= 8 FFI empty kernel         median=3.3684 samples=[3.535542, 3.365736, 3.523181, 3.380743, 3.2734140000000003, 3.368423, 3.2706399999999998, 3.3603, 3.464402]
args= 8 INTJ static_compile kernel median=3.0100 samples=[3.06891, 2.935991, 3.0194479999999997, 3.010026, 3.0214920000000003, 2.959491, 3.008823, 3.017219, 2.9984960000000003]
args= 8 INTJ runtime_shim kernel median=2.8737 samples=[3.077496, 2.873737, 2.863681, 2.8901280000000003, 2.8511640000000003, 2.8715770000000003, 2.863727, 2.8742710000000002, 2.883983]
args=16 FFI packed nop           median=0.2954 samples=[0.2932, 0.29456, 0.295365, 0.296487, 0.297139, 0.29847199999999996, 0.296276, 0.293113, 0.29259199999999996]
args=16 FFI typed nop            median=0.3061 samples=[0.29733, 0.310095, 0.307093, 0.309995, 0.303952, 0.306012, 0.30742200000000003, 0.304296, 0.306075]
args=16 FFI empty kernel         median=3.6077 samples=[3.7303859999999998, 3.652999, 3.661674, 3.607747, 3.5172800000000004, 3.623223, 3.5075030000000003, 3.597549, 3.507653]
args=16 INTJ static_compile kernel median=3.2941 samples=[3.2717899999999998, 3.3099279999999998, 3.317851, 3.286053, 3.461906, 3.2715549999999998, 3.294142, 3.2705770000000003, 3.344931]
args=16 INTJ runtime_shim kernel median=3.2452 samples=[3.260595, 3.097971, 3.281688, 3.034828, 3.337442, 3.0125300000000004, 3.245174, 3.017888, 3.267903]
args=32 FFI packed nop           median=0.4880 samples=[0.48319999999999996, 0.488134, 0.486053, 0.484144, 0.48808300000000004, 0.492276, 0.488014, 0.479222, 0.488327]
args=32 FFI typed nop            median=0.4946 samples=[0.491483, 0.508402, 0.496752, 0.504516, 0.49367700000000003, 0.505424, 0.494572, 0.491019, 0.488907]
args=32 FFI empty kernel         median=4.0788 samples=[4.1858, 4.078843, 3.999469, 4.0959389999999996, 3.968792, 4.107102, 3.988077, 4.095741, 3.985011]
args=32 INTJ static_compile kernel median=3.5769 samples=[3.463915, 3.576904, 3.699953, 3.532923, 3.628831, 3.521527, 3.714192, 3.5124470000000003, 3.7403969999999997]
args=32 INTJ runtime_shim kernel median=3.4466 samples=[3.426621, 3.446609, 3.5858760000000003, 3.4162660000000002, 3.6013699999999997, 3.4176759999999997, 3.5715619999999997, 3.409119, 3.575377]
args=64 FFI packed nop           median=0.8483 samples=[0.823539, 0.8680370000000001, 0.848294, 0.859309, 0.8537509999999999, 0.846328, 0.853596, 0.827921, 0.834621]
args=64 FFI typed nop            median=0.8792 samples=[0.879164, 0.886131, 0.8701530000000001, 0.907046, 0.891152, 0.885565, 0.86025, 0.8532150000000001, 0.859309]
args=64 FFI empty kernel         median=5.0860 samples=[5.096302, 5.0860069999999995, 5.097645000000001, 5.081535, 5.107905, 5.061919, 5.086003, 5.064459, 5.079511]
args=64 INTJ static_compile kernel median=4.5412 samples=[4.553895000000001, 4.467445, 4.541184, 4.544925, 4.547586, 4.512484000000001, 4.530899000000001, 4.570995, 4.533407]
args=64 INTJ runtime_shim kernel median=4.4939 samples=[4.46253, 4.494703, 4.4765500000000005, 4.5472209999999995, 4.486502000000001, 4.560772, 4.493904000000001, 4.50612, 4.461193000000001]
elapsed_ns=4427835265
exit_status=0
ended=2026-09-27T00:02:14+08:00
```

#### round2_hip_module
```
commit=97ef0aee3d02754ad0a48250b9c32eaf37b6df6f
started=2026-09-27T00:02:16+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_hip_module_launch.py --iters 1000 --batches 9
hsaco_sha256=df3e529e178196ad62f7c07ab8593ede5b8596c76e7c6fbd12a5d947a696e65c kernel=empty_kernel threads=256 shared=0 callback_count=1
iters=1000 batches=9 unit=us/call (host enqueue, no timed sync)
TVM FFI HIP HSACO         median=3.6301 samples=[3.553821, 3.663635, 3.755043, 3.701634, 3.652975, 3.573804, 3.5565770000000003, 3.615732, 3.630134]
Triton same HSACO         median=16.2995 samples=[16.513589, 16.40093, 16.299515, 16.350585, 16.210603, 16.376994, 16.156007, 16.296992, 16.252673]
INTJ same function        median=3.4764 samples=[3.486193, 3.573042, 3.512376, 3.5433589999999997, 3.476362, 3.38069, 3.474046, 3.466784, 3.4008760000000002]
elapsed_ns=3813010392
exit_status=0
ended=2026-09-27T00:02:20+08:00
```

#### round2_kernel_cache
```
commit=97ef0aee3d02754ad0a48250b9c32eaf37b6df6f
started=2026-09-27T00:02:20+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python -m pytest tests/test_kernel_cache.py -s -q

kernel cache, 1-word key, ns per operation

                      intj       tsl      absl
hash_only             1.55      1.35      1.36
hit/1                 1.76      2.04      1.51
hit/8                 1.84      2.14      5.75
hit/64                1.88      2.17      4.95
hit/512               3.27      3.03      5.81
hit_child             2.46      2.46      2.46
miss/1                1.39      1.53      1.38
miss/8                2.40      2.37      3.66
miss/64               2.79      2.10      4.99
miss/512              2.80      2.68      4.42

kernel cache, 2-word key, ns per operation

                      intj       tsl      absl
hash_only             1.42      1.43      1.42
hit/1                 2.18      2.59      1.95
hit/8                 2.18      2.56      5.39
hit/64                2.50      3.17      5.60
hit/512               3.36      4.04      6.83
hit_child             2.46      2.46      2.46
miss/1                1.43      2.32      1.71
miss/8                2.74      1.81      3.61
miss/64               3.32      2.45      4.15
miss/512              3.28      2.72      4.42

kernel cache, 5-word key, ns per operation

                      intj       tsl      absl
hash_only             3.57      3.56      3.58
hit/1                 3.13      3.18      2.83
hit/8                 3.37      3.53     10.06
hit/64                3.44      3.62     10.31
hit/512               4.73      4.93     11.66
hit_child             2.46      2.46      2.45
miss/1                2.00      2.12      1.71
miss/8                3.76      2.15      7.12
miss/64               3.70      1.76      6.91
miss/512              4.12      2.56      7.92
....
4 passed in 25.03s
elapsed_ns=25888037993
exit_status=0
ended=2026-09-27T00:02:46+08:00
```

#### round2_launch_gpu
```
commit=97ef0aee3d02754ad0a48250b9c32eaf37b6df6f
started=2026-09-27T00:01:48+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --iters 1000 --batches 9
mode=gpu; 1000 calls × 9 batches; median ns/call
               path    ns/call       Δ ns       Δ %   build ms   bind ms
           auto map     3126.4       +0.0      +0.0      74.23         -
        reduced key     3091.2      -35.3      -1.1      82.83         -
         verify off     3042.6      -83.8      -2.7       2.14         -
          verify on     3023.9     -102.5      -3.3       2.04         -
              baked     2973.0     -153.4      -4.9       2.15         -
       bound tensor     2996.1     -130.3      -4.2       0.36      2.02
      bound pointer     2944.6     -181.9      -5.8       1.81      9.51
   fixed device map     2891.4     -235.1      -7.5       0.33      1.99
fixed device no-map     2871.3     -255.1      -8.2       0.28      1.98
elapsed_ns=3676674637
exit_status=0
ended=2026-09-27T00:01:52+08:00
```

#### round2_launch_host
```
commit=97ef0aee3d02754ad0a48250b9c32eaf37b6df6f
started=2026-09-27T00:01:52+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --no-gpu --iters 100000 --batches 9
mode=host-only; 100000 calls × 9 batches; median ns/call
               path    ns/call       Δ ns       Δ %   build ms   bind ms
           auto map       42.4       +0.0      +0.0      62.57         -
        reduced key       41.8       -0.7      -1.6       4.64         -
         verify off       41.9       -0.5      -1.1       1.86         -
          verify on       43.5       +1.1      +2.6       1.96         -
              baked       39.5       -2.9      -6.9       1.83         -
       bound tensor       46.7       +4.3     +10.1       0.16      1.75
      bound pointer       44.4       +2.0      +4.7       0.14      1.68
   fixed device map       42.6       +0.2      +0.5       0.12      1.68
fixed device no-map       43.5       +1.1      +2.6       0.15      1.65
elapsed_ns=3027226110
exit_status=0
ended=2026-09-27T00:01:55+08:00
```

#### round2_launch_last_key
```
commit=97ef0aee3d02754ad0a48250b9c32eaf37b6df6f
started=2026-09-27T00:02:02+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --last-key --iters 100000 --batches 9
mode=last-key; host-only; 200000 calls × 9 batches; median ns/call
count     pattern    ns/call
    4      repeat       33.6
    4   alternate       36.3
   16      repeat       62.5
   16   alternate       65.3
   32      repeat       99.8
   32   alternate      104.4
elapsed_ns=3377712540
exit_status=0
ended=2026-09-27T00:02:05+08:00
```

#### round2_launch_readme
```
commit=97ef0aee3d02754ad0a48250b9c32eaf37b6df6f
started=2026-09-27T00:01:55+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --readme --iters 1000 --batches 9
mode=readme; 1000 calls × 9 batches; median
        path   triton us    intj us   speedup
   grid=(1,)       16.23       3.12      5.2x
   grid=(0,)       12.87       0.03    455.5x

torch_access_mode   decode ns    build s
     runtime_shim        92.4       0.02
   static_compile        89.5       0.06
      interpreter       869.0       0.00
elapsed_ns=3852381765
exit_status=0
ended=2026-09-27T00:01:59+08:00
```

#### round2_launch_sweep
```
commit=97ef0aee3d02754ad0a48250b9c32eaf37b6df6f
started=2026-09-27T00:01:59+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --sweep --iters 100000 --batches 9
mode=sweep; host-only; 100000 calls × 9 batches; median ns/call
count   kind    ns/call
    4    int       40.6
    4 tensor       39.3
   16    int       68.2
   16 tensor       64.4
   32    int      106.0
   32 tensor      109.8
elapsed_ns=2959509778
exit_status=0
ended=2026-09-27T00:02:02+08:00
```

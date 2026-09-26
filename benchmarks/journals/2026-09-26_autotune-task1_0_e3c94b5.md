# Autotune/heuristics Task 1 benchmark gate

Task 1 of the autotune/heuristics plan ("By-value cache records behind a read/write
lock"), measured against `2026-09-26_autotune-baseline_0_87c814b.md`.

Measured at commit `e3c94b5` (Store cache records by value behind a read/write lock),
September 26, 2026, clean checkout (this journal is committed after it). Baseline
revision: `1f5e5fa` (code-identical to `cab358b`, the parent of `e3c94b5`).

Same suite, driver and order as the baseline: `/tmp/intj-bench/run_all.sh task1`, then,
because rows were flagged, `/tmp/intj-bench/run_all.sh task1-rerun` once. Ten cases x
3 rounds = 30 runs each; all 60 report `exit_status=0` and `commit=e3c94b54...`.
Comparison: `/tmp/gb2/bin/python .superpowers/sdd/2026-09-26-autotune-heuristics/bench_compare.py baseline <label>`.

## Environment

- CPU: x86-64 (224 logical CPUs), `taskset -c 0`, one benchmark process at a time.
  The host is shared: load average was 6-13 from other users during the runs, so
  process-to-process variance was larger than in the baseline (FFI rows, which run no
  intj code, moved by up to 10%).
- GPU: AMD Instinct MI308X, `gfx942`.
- Python 3.12.3 at `/tmp/gb2/bin/python`; Torch `2.14.0+rocm7.2`; Triton `3.8.0`;
  `apache-tvm-ffi` `0.1.14.post2.dev1+g424558557.d20260924`.
- `INTJ_BENCHMARK_ROOT=/tmp/gbench` (google/benchmark, kernel-cache case only).
- Rendered modules are built by triton at `-O3`; `kernel_cache` builds with `g++ -O3`.

## Commands

As in the baseline journal: for each round 0-2, each line its own process, prefixed by
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

No real regression on the launch path. Flagged rows (**R** in the table):

- `task1` flagged 9 rows; `task1-rerun` flagged one, `ffi_paths INTJ pair direct`
  (+2.4 ns, +3.7%, just over both thresholds). Every other flag cleared on rerun,
  and several were rows that run no intj code (`ffi_sweep ... FFI empty kernel`,
  `hip_module TVM FFI HIP HSACO`), i.e. host noise.
- `pair direct` was then measured interleaved against the baseline checkout
  (diagnostic below): median 66.9 ns baseline vs 67.2 ns task1 over 5 alternating
  process pairs, +0.3 ns. The gate's baseline for this row (65.3-65.8 ns) was taken
  on a quieter host; it read 66.2-67.2 ns in the interleaved run.
- Accepted cost: within ~1 ns on one-word-key host rows (`hot call`, `3-tensor
  host-only nop`, `direct positional`, `launch_host`), as the plan allows.
- New row `hit_child` (a 1-word child map with two entries, the shape Task 4's
  computed-bool level will have) has no baseline.

## Kernel-cache slot layout: why the key is stored at one word

The plan's slot dropped the key at one word and compared the memo by hash. Measured
first (`task1-keyless`, not committed), that cost 1.3-2 ns on one-word launch rows,
because a memo hit now hashed (interleaved diagnostic: `3-tensor host-only nop`
33.6 -> 35.0 ns, `pair direct` 66.4 -> 68.3 ns). Storing both key and hash
(`task1-keyed`, not committed) fixed the launch path but grew the one-word slot to
48 bytes and flagged `kernel_cache 1-word miss/8..512 intj` at +2.3-2.5 ns. The
committed layout stores the key and, above one word, the hash; at one word the hash
is a bijection of the key and is recomputed on rehash. Slot size is back to 40 bytes
and the memo compares without hashing.

## Baseline vs task1 medians

Median of 3 rounds per label; Δ is against the baseline median.

| case | row | unit | baseline | task1 | Δ | Δ% | rerun | Δ | Δ% |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ffi_compare | median FFI empty kernel value | us | 3.362 | 3.394 | +0.032 | +1.0% | 3.394 | +0.0324 | +1.0% |
| ffi_compare | median FFI mixed kernel value | us | 3.341 | 3.331 | -0.0093 | -0.3% | 3.409 | +0.0687 | +2.1% |
| ffi_compare | median FFI packed nop value | us | 0.1446 | 0.1448 | +0.0002 | +0.1% | 0.1438 | -0.0008 | -0.6% |
| ffi_compare | median FFI packed nop mixed value | us | 0.1745 | 0.1736 | -0.0009 | -0.5% | 0.1746 | +0.0001 | +0.1% |
| ffi_compare | median FFI typed nop value | us | 0.1491 | 0.1483 | -0.0008 | -0.5% | 0.1489 | -0.0002 | -0.1% |
| ffi_compare | median FFI typed nop mixed value | us | 0.1772 | 0.1755 | -0.0017 | -1.0% | 0.1775 | +0.0003 | +0.2% |
| ffi_compare | median INTJ empty kernel value | us | 2.985 | 3.014 | +0.029 | +1.0% | 3.005 | +0.0197 | +0.7% |
| ffi_compare | median INTJ fixed mixed kernel value | us | 2.939 | 2.911 | -0.028 | -1.0% | 3.029 | +0.0906 | +3.1% |
| ffi_compare | median INTJ fixed-device kernel value | us | 2.919 | 2.841 | -0.0778 | -2.7% | 2.928 | +0.0097 | +0.3% |
| ffi_compare | median INTJ mixed kernel value | us | 2.928 | 2.921 | -0.0071 | -0.2% | 2.969 | +0.0405 | +1.4% |
| ffi_paths | median_ns FFI unpack Pair only value | ns | 162.7 | 167.5 | +4.8 | +3.0% | 162 | -0.7 | -0.4% |
| ffi_paths | median_ns INTJ 3-tensor host-only nop value | ns | 33.8 | 34.3 | +0.5 | +1.5% | 34.2 | +0.4 | +1.2% |
| ffi_paths | median_ns INTJ FFI wrapper kwargs value | ns | 126.7 | 127.1 | +0.4 | +0.3% | 126.8 | +0.1 | +0.1% |
| ffi_paths | median_ns INTJ FFI wrapper positional value | ns | 103.4 | 103.8 | +0.4 | +0.4% | 104.9 | +1.5 | +1.5% |
| ffi_paths | median_ns INTJ adapter defaults value | ns | 89.2 | 89.7 | +0.5 | +0.6% | 89.4 | +0.2 | +0.2% |
| ffi_paths | median_ns INTJ adapter kwargs value | ns | 108.2 | 110.4 | +2.2 | +2.0% | 108.7 | +0.5 | +0.5% |
| ffi_paths | median_ns INTJ adapter positional value | ns | 90.6 | 92.4 | +1.8 | +2.0% | 91.8 | +1.2 | +1.3% |
| ffi_paths | median_ns INTJ cold compile callback + cache value | ns | 429.9 | 414.5 | -15.4 | -3.6% | 414 | -15.9 | -3.7% |
| ffi_paths | median_ns INTJ config FFI unpack value | ns | 321.1 | 321 | -0.1 | -0.0% | 318 | -3.1 | -1.0% |
| ffi_paths | median_ns INTJ config direct value | ns | 79.1 | 83.5 | +4.4 **R** | +5.6% | 79.4 | +0.3 | +0.4% |
| ffi_paths | median_ns INTJ direct positional value | ns | 64.7 | 65.4 | +0.7 | +1.1% | 65.4 | +0.7 | +1.1% |
| ffi_paths | median_ns INTJ hot call (no callback) value | ns | 37.2 | 39.1 | +1.9 | +5.1% | 38.7 | +1.5 | +4.0% |
| ffi_paths | median_ns INTJ pair FFI unpack value | ns | 309.1 | 318.3 | +9.2 | +3.0% | 315.2 | +6.1 | +2.0% |
| ffi_paths | median_ns INTJ pair direct value | ns | 65.6 | 69.5 | +3.9 **R** | +5.9% | 68 | +2.4 **R** | +3.7% |
| ffi_paths | median_ns INTJ pair manual unpack value | ns | 171.6 | 173.7 | +2.1 | +1.2% | 174.5 | +2.9 | +1.7% |
| ffi_paths | median_ns INTJ pair prebuilt *tuple value | ns | 137.4 | 139.9 | +2.5 | +1.8% | 138.7 | +1.3 | +0.9% |
| ffi_paths | median_ns INTJ pair stdlib astuple value | ns | 1168 | 1160 | -7.6 | -0.7% | 1162 | -6.5 | -0.6% |
| ffi_sweep | median args=0 FFI empty kernel value | us | 1.678 | 1.854 | +0.1756 **R** | +10.5% | 1.636 | -0.0426 | -2.5% |
| ffi_sweep | median args=0 FFI packed nop value | us | 0.1175 | 0.1168 | -0.0007 | -0.6% | 0.1165 | -0.001 | -0.9% |
| ffi_sweep | median args=0 FFI typed nop value | us | 0.121 | 0.122 | +0.001 | +0.8% | 0.1202 | -0.0008 | -0.7% |
| ffi_sweep | median args=16 FFI empty kernel value | us | 3.654 | 3.777 | +0.1231 **R** | +3.4% | 3.636 | -0.0177 | -0.5% |
| ffi_sweep | median args=16 FFI packed nop value | us | 0.2923 | 0.2931 | +0.0008 | +0.3% | 0.2956 | +0.0033 | +1.1% |
| ffi_sweep | median args=16 FFI typed nop value | us | 0.3021 | 0.3044 | +0.0023 | +0.8% | 0.3046 | +0.0025 | +0.8% |
| ffi_sweep | median args=3 FFI empty kernel value | us | 3.25 | 3.312 | +0.0622 | +1.9% | 3.193 | -0.0571 | -1.8% |
| ffi_sweep | median args=3 FFI packed nop value | us | 0.1451 | 0.1445 | -0.0006 | -0.4% | 0.1447 | -0.0004 | -0.3% |
| ffi_sweep | median args=3 FFI typed nop value | us | 0.1492 | 0.1487 | -0.0005 | -0.3% | 0.1485 | -0.0007 | -0.5% |
| ffi_sweep | median args=32 FFI empty kernel value | us | 4.225 | 4.295 | +0.0701 | +1.7% | 4.138 | -0.0873 | -2.1% |
| ffi_sweep | median args=32 FFI packed nop value | us | 0.4805 | 0.4809 | +0.0004 | +0.1% | 0.4808 | +0.0003 | +0.1% |
| ffi_sweep | median args=32 FFI typed nop value | us | 0.4946 | 0.4924 | -0.0022 | -0.4% | 0.4953 | +0.0007 | +0.1% |
| ffi_sweep | median args=5 FFI empty kernel value | us | 3.29 | 3.288 | -0.0019 | -0.1% | 3.255 | -0.0353 | -1.1% |
| ffi_sweep | median args=5 FFI packed nop value | us | 0.174 | 0.1734 | -0.0006 | -0.3% | 0.1755 | +0.0015 | +0.9% |
| ffi_sweep | median args=5 FFI typed nop value | us | 0.1783 | 0.1772 | -0.0011 | -0.6% | 0.1789 | +0.0006 | +0.3% |
| ffi_sweep | median args=64 FFI empty kernel value | us | 5.054 | 5.076 | +0.0214 | +0.4% | 5.051 | -0.0032 | -0.1% |
| ffi_sweep | median args=64 FFI packed nop value | us | 0.8387 | 0.8415 | +0.0028 | +0.3% | 0.8483 | +0.0096 | +1.1% |
| ffi_sweep | median args=64 FFI typed nop value | us | 0.8587 | 0.8624 | +0.0037 | +0.4% | 0.8811 | +0.0224 | +2.6% |
| ffi_sweep | median args=8 FFI empty kernel value | us | 3.392 | 3.531 | +0.1384 **R** | +4.1% | 3.387 | -0.0054 | -0.2% |
| ffi_sweep | median args=8 FFI packed nop value | us | 0.206 | 0.2074 | +0.0014 | +0.7% | 0.2067 | +0.0007 | +0.3% |
| ffi_sweep | median args=8 FFI typed nop value | us | 0.2099 | 0.2099 | +0 | +0.0% | 0.2095 | -0.0004 | -0.2% |
| hip_module | median INTJ same function value | us | 2.934 | 3.02 | +0.0864 | +2.9% | 2.939 | +0.0057 | +0.2% |
| hip_module | median TVM FFI HIP HSACO value | us | 3.147 | 3.309 | +0.1621 **R** | +5.2% | 3.145 | -0.0019 | -0.1% |
| hip_module | median Triton same HSACO value | us | 15.95 | 16.1 | +0.148 | +0.9% | 15.92 | -0.0386 | -0.2% |
| kernel_cache | 1-word key hash_only absl | ns | 1.36 | 1.36 | +0 | +0.0% | 1.36 | +0 | +0.0% |
| kernel_cache | 1-word key hash_only intj | ns | 1.36 | 1.36 | +0 | +0.0% | 1.37 | +0.01 | +0.7% |
| kernel_cache | 1-word key hash_only tsl | ns | 1.36 | 1.35 | -0.01 | -0.7% | 1.35 | -0.01 | -0.7% |
| kernel_cache | 1-word key hit/1 absl | ns | 1.35 | 1.52 | +0.17 | +12.6% | 1.54 | +0.19 | +14.1% |
| kernel_cache | 1-word key hit/1 intj | ns | 1.2 | 1.76 | +0.56 | +46.7% | 1.76 | +0.56 | +46.7% |
| kernel_cache | 1-word key hit/1 tsl | ns | 1.85 | 2.04 | +0.19 | +10.3% | 2.03 | +0.18 | +9.7% |
| kernel_cache | 1-word key hit/512 absl | ns | 4.94 | 5.79 | +0.85 | +17.2% | 5.79 | +0.85 | +17.2% |
| kernel_cache | 1-word key hit/512 intj | ns | 1.49 | 2.56 | +1.07 | +71.8% | 2.54 | +1.05 | +70.5% |
| kernel_cache | 1-word key hit/512 tsl | ns | 2.62 | 3.03 | +0.41 | +15.6% | 3.04 | +0.42 | +16.0% |
| kernel_cache | 1-word key hit/64 absl | ns | 4.35 | 4.95 | +0.6 | +13.8% | 4.95 | +0.6 | +13.8% |
| kernel_cache | 1-word key hit/64 intj | ns | 1.31 | 1.87 | +0.56 | +42.7% | 1.88 | +0.57 | +43.5% |
| kernel_cache | 1-word key hit/64 tsl | ns | 1.91 | 2.17 | +0.26 | +13.6% | 2.17 | +0.26 | +13.6% |
| kernel_cache | 1-word key hit/8 absl | ns | 4.23 | 6.26 | +2.03 **R** | +48.0% | 4.77 | +0.54 | +12.8% |
| kernel_cache | 1-word key hit/8 intj | ns | 1.26 | 1.84 | +0.58 | +46.0% | 1.83 | +0.57 | +45.2% |
| kernel_cache | 1-word key hit/8 tsl | ns | 1.88 | 2.13 | +0.25 | +13.3% | 2.13 | +0.25 | +13.3% |
| kernel_cache | 1-word key miss/1 absl | ns | 1.38 | 1.38 | +0 | +0.0% | 1.38 | +0 | +0.0% |
| kernel_cache | 1-word key miss/1 intj | ns | 1.03 | 1.39 | +0.36 | +35.0% | 1.39 | +0.36 | +35.0% |
| kernel_cache | 1-word key miss/1 tsl | ns | 1.46 | 1.61 | +0.15 | +10.3% | 1.58 | +0.12 | +8.2% |
| kernel_cache | 1-word key miss/512 absl | ns | 3.94 | 6.12 | +2.18 **R** | +55.3% | 4.42 | +0.48 | +12.2% |
| kernel_cache | 1-word key miss/512 intj | ns | 2.23 | 2.81 | +0.58 | +26.0% | 2.79 | +0.56 | +25.1% |
| kernel_cache | 1-word key miss/512 tsl | ns | 1.97 | 2.41 | +0.44 | +22.3% | 2.42 | +0.45 | +22.8% |
| kernel_cache | 1-word key miss/64 absl | ns | 3.81 | 3.94 | +0.13 | +3.4% | 3.91 | +0.1 | +2.6% |
| kernel_cache | 1-word key miss/64 intj | ns | 2.15 | 2.74 | +0.59 | +27.4% | 2.84 | +0.69 | +32.1% |
| kernel_cache | 1-word key miss/64 tsl | ns | 1.73 | 2.1 | +0.37 | +21.4% | 2.12 | +0.39 | +22.5% |
| kernel_cache | 1-word key miss/8 absl | ns | 3.25 | 3.65 | +0.4 | +12.3% | 3.66 | +0.41 | +12.6% |
| kernel_cache | 1-word key miss/8 intj | ns | 2.22 | 2.43 | +0.21 | +9.5% | 2.4 | +0.18 | +8.1% |
| kernel_cache | 1-word key miss/8 tsl | ns | 1.75 | 2.12 | +0.37 | +21.1% | 2.12 | +0.37 | +21.1% |
| kernel_cache | 2-word key hash_only absl | ns | 1.42 | 1.42 | +0 | +0.0% | 1.42 | +0 | +0.0% |
| kernel_cache | 2-word key hash_only intj | ns | 1.42 | 1.44 | +0.02 | +1.4% | 1.42 | +0 | +0.0% |
| kernel_cache | 2-word key hash_only tsl | ns | 1.42 | 1.43 | +0.01 | +0.7% | 1.42 | +0 | +0.0% |
| kernel_cache | 2-word key hit/1 absl | ns | 1.7 | 1.96 | +0.26 | +15.3% | 1.98 | +0.28 | +16.5% |
| kernel_cache | 2-word key hit/1 intj | ns | 1.89 | 2.18 | +0.29 | +15.3% | 2.18 | +0.29 | +15.3% |
| kernel_cache | 2-word key hit/1 tsl | ns | 2.26 | 2.57 | +0.31 | +13.7% | 2.57 | +0.31 | +13.7% |
| kernel_cache | 2-word key hit/512 absl | ns | 5.83 | 6.82 | +0.99 | +17.0% | 6.83 | +1 | +17.2% |
| kernel_cache | 2-word key hit/512 intj | ns | 2.73 | 3.33 | +0.6 | +22.0% | 3.33 | +0.6 | +22.0% |
| kernel_cache | 2-word key hit/512 tsl | ns | 3.46 | 4.32 | +0.86 | +24.9% | 4.04 | +0.58 | +16.8% |
| kernel_cache | 2-word key hit/64 absl | ns | 5.25 | 5.6 | +0.35 | +6.7% | 5.58 | +0.33 | +6.3% |
| kernel_cache | 2-word key hit/64 intj | ns | 2.11 | 2.52 | +0.41 | +19.4% | 2.5 | +0.39 | +18.5% |
| kernel_cache | 2-word key hit/64 tsl | ns | 2.58 | 2.92 | +0.34 | +13.2% | 2.91 | +0.33 | +12.8% |
| kernel_cache | 2-word key hit/8 absl | ns | 4.76 | 5.4 | +0.64 | +13.4% | 5.39 | +0.63 | +13.2% |
| kernel_cache | 2-word key hit/8 intj | ns | 1.89 | 2.18 | +0.29 | +15.3% | 2.18 | +0.29 | +15.3% |
| kernel_cache | 2-word key hit/8 tsl | ns | 2.27 | 2.57 | +0.3 | +13.2% | 2.57 | +0.3 | +13.2% |
| kernel_cache | 2-word key miss/1 absl | ns | 1.54 | 1.71 | +0.17 | +11.0% | 1.72 | +0.18 | +11.7% |
| kernel_cache | 2-word key miss/1 intj | ns | 1.32 | 1.42 | +0.1 | +7.6% | 1.42 | +0.1 | +7.6% |
| kernel_cache | 2-word key miss/1 tsl | ns | 2.16 | 2.32 | +0.16 | +7.4% | 2.32 | +0.16 | +7.4% |
| kernel_cache | 2-word key miss/512 absl | ns | 3.9 | 4.44 | +0.54 | +13.8% | 4.43 | +0.53 | +13.6% |
| kernel_cache | 2-word key miss/512 intj | ns | 2.95 | 3.33 | +0.38 | +12.9% | 3.19 | +0.24 | +8.1% |
| kernel_cache | 2-word key miss/512 tsl | ns | 2.3 | 2.71 | +0.41 | +17.8% | 2.72 | +0.42 | +18.3% |
| kernel_cache | 2-word key miss/64 absl | ns | 3.36 | 4.89 | +1.53 | +45.5% | 4.16 | +0.8 | +23.8% |
| kernel_cache | 2-word key miss/64 intj | ns | 3.08 | 3.36 | +0.28 | +9.1% | 3.42 | +0.34 | +11.0% |
| kernel_cache | 2-word key miss/64 tsl | ns | 2.01 | 2.34 | +0.33 | +16.4% | 2.34 | +0.33 | +16.4% |
| kernel_cache | 2-word key miss/8 absl | ns | 3.27 | 3.61 | +0.34 | +10.4% | 3.61 | +0.34 | +10.4% |
| kernel_cache | 2-word key miss/8 intj | ns | 2.71 | 2.81 | +0.1 | +3.7% | 2.77 | +0.06 | +2.2% |
| kernel_cache | 2-word key miss/8 tsl | ns | 1.75 | 1.79 | +0.04 | +2.3% | 1.8 | +0.05 | +2.9% |
| kernel_cache | 5-word key hash_only absl | ns | 3.57 | 3.58 | +0.01 | +0.3% | 3.58 | +0.01 | +0.3% |
| kernel_cache | 5-word key hash_only intj | ns | 3.58 | 3.57 | -0.01 | -0.3% | 3.58 | +0 | +0.0% |
| kernel_cache | 5-word key hash_only tsl | ns | 3.58 | 3.6 | +0.02 | +0.6% | 3.58 | +0 | +0.0% |
| kernel_cache | 5-word key hit/1 absl | ns | 2.68 | 2.83 | +0.15 | +5.6% | 2.83 | +0.15 | +5.6% |
| kernel_cache | 5-word key hit/1 intj | ns | 2.94 | 3.14 | +0.2 | +6.8% | 3.14 | +0.2 | +6.8% |
| kernel_cache | 5-word key hit/1 tsl | ns | 3.19 | 3.18 | -0.01 | -0.3% | 3.18 | -0.01 | -0.3% |
| kernel_cache | 5-word key hit/512 absl | ns | 11.18 | 11.65 | +0.47 | +4.2% | 11.66 | +0.48 | +4.3% |
| kernel_cache | 5-word key hit/512 intj | ns | 4.47 | 4.76 | +0.29 | +6.5% | 4.73 | +0.26 | +5.8% |
| kernel_cache | 5-word key hit/512 tsl | ns | 4.94 | 4.92 | -0.02 | -0.4% | 4.92 | -0.02 | -0.4% |
| kernel_cache | 5-word key hit/64 absl | ns | 9.75 | 10.31 | +0.56 | +5.7% | 10.3 | +0.55 | +5.6% |
| kernel_cache | 5-word key hit/64 intj | ns | 3.25 | 3.5 | +0.25 | +7.7% | 3.52 | +0.27 | +8.3% |
| kernel_cache | 5-word key hit/64 tsl | ns | 3.64 | 3.62 | -0.02 | -0.5% | 3.62 | -0.02 | -0.5% |
| kernel_cache | 5-word key hit/8 absl | ns | 9.47 | 10.01 | +0.54 | +5.7% | 9.99 | +0.52 | +5.5% |
| kernel_cache | 5-word key hit/8 intj | ns | 3.17 | 3.36 | +0.19 | +6.0% | 3.39 | +0.22 | +6.9% |
| kernel_cache | 5-word key hit/8 tsl | ns | 3.54 | 3.53 | -0.01 | -0.3% | 3.53 | -0.01 | -0.3% |
| kernel_cache | 5-word key miss/1 absl | ns | 1.66 | 1.73 | +0.07 | +4.2% | 1.72 | +0.06 | +3.6% |
| kernel_cache | 5-word key miss/1 intj | ns | 1.85 | 2.01 | +0.16 | +8.6% | 2.05 | +0.2 | +10.8% |
| kernel_cache | 5-word key miss/1 tsl | ns | 2.27 | 2.12 | -0.15 | -6.6% | 2.12 | -0.15 | -6.6% |
| kernel_cache | 5-word key miss/512 absl | ns | 7.66 | 7.92 | +0.26 | +3.4% | 7.92 | +0.26 | +3.4% |
| kernel_cache | 5-word key miss/512 intj | ns | 4.1 | 4.15 | +0.05 | +1.2% | 4.09 | -0.01 | -0.2% |
| kernel_cache | 5-word key miss/512 tsl | ns | 2.76 | 2.57 | -0.19 | -6.9% | 2.57 | -0.19 | -6.9% |
| kernel_cache | 5-word key miss/64 absl | ns | 6.71 | 6.91 | +0.2 | +3.0% | 6.91 | +0.2 | +3.0% |
| kernel_cache | 5-word key miss/64 intj | ns | 3.38 | 4.6 | +1.22 | +36.1% | 3.71 | +0.33 | +9.8% |
| kernel_cache | 5-word key miss/64 tsl | ns | 2.55 | 1.77 | -0.78 | -30.6% | 1.77 | -0.78 | -30.6% |
| kernel_cache | 5-word key miss/8 absl | ns | 6.93 | 7.13 | +0.2 | +2.9% | 7.14 | +0.21 | +3.0% |
| kernel_cache | 5-word key miss/8 intj | ns | 3.57 | 3.85 | +0.28 | +7.8% | 3.84 | +0.27 | +7.6% |
| kernel_cache | 5-word key miss/8 tsl | ns | 3.09 | 2.16 | -0.93 | -30.1% | 2.15 | -0.94 | -30.4% |
| launch_gpu | launch auto map ns/call | ns | 3138 | 3134 | -3.3 | -0.1% | 3089 | -49.1 | -1.6% |
| launch_gpu | launch baked ns/call | ns | 3054 | 2986 | -68.3 | -2.2% | 3000 | -54.7 | -1.8% |
| launch_gpu | launch bound pointer ns/call | ns | 2936 | 3012 | +76.8 | +2.6% | 3012 | +76.7 | +2.6% |
| launch_gpu | launch bound tensor ns/call | ns | 3086 | 3104 | +17.8 | +0.6% | 3022 | -63.6 | -2.1% |
| launch_gpu | launch fixed device map ns/call | ns | 2932 | 2910 | -22 | -0.8% | 2911 | -21.1 | -0.7% |
| launch_gpu | launch fixed device no-map ns/call | ns | 3011 | 2936 | -74.9 | -2.5% | 2900 | -111.2 | -3.7% |
| launch_gpu | launch reduced key ns/call | ns | 3124 | 3112 | -11.6 | -0.4% | 3103 | -20.1 | -0.6% |
| launch_gpu | launch verify off ns/call | ns | 2995 | 2971 | -24.2 | -0.8% | 2976 | -19.1 | -0.6% |
| launch_gpu | launch verify on ns/call | ns | 2914 | 2940 | +26 | +0.9% | 2906 | -8.1 | -0.3% |
| launch_host | launch auto map ns/call | ns | 41.6 | 42.3 | +0.7 | +1.7% | 41.8 | +0.2 | +0.5% |
| launch_host | launch baked ns/call | ns | 39.2 | 39.6 | +0.4 | +1.0% | 40.4 | +1.2 | +3.1% |
| launch_host | launch bound pointer ns/call | ns | 44.8 | 45.3 | +0.5 | +1.1% | 45 | +0.2 | +0.4% |
| launch_host | launch bound tensor ns/call | ns | 45.6 | 46.2 | +0.6 | +1.3% | 46.1 | +0.5 | +1.1% |
| launch_host | launch fixed device map ns/call | ns | 42.1 | 42.8 | +0.7 | +1.7% | 42.8 | +0.7 | +1.7% |
| launch_host | launch fixed device no-map ns/call | ns | 44 | 44.1 | +0.1 | +0.2% | 44.3 | +0.3 | +0.7% |
| launch_host | launch reduced key ns/call | ns | 41.9 | 42 | +0.1 | +0.2% | 41.8 | -0.1 | -0.2% |
| launch_host | launch verify off ns/call | ns | 42.5 | 42.4 | -0.1 | -0.2% | 42.4 | -0.1 | -0.2% |
| launch_host | launch verify on ns/call | ns | 44.2 | 43.4 | -0.8 | -1.8% | 43.7 | -0.5 | -1.1% |
| launch_last_key | sweep 16 alternate ns/call | ns | 65.4 | 65.3 | -0.1 | -0.2% | 65.4 | +0 | +0.0% |
| launch_last_key | sweep 16 repeat ns/call | ns | 61.9 | 62.7 | +0.8 | +1.3% | 62.7 | +0.8 | +1.3% |
| launch_last_key | sweep 32 alternate ns/call | ns | 109.1 | 104.3 | -4.8 | -4.4% | 104 | -5.1 | -4.7% |
| launch_last_key | sweep 32 repeat ns/call | ns | 104.8 | 99.8 | -5 | -4.8% | 100 | -4.8 | -4.6% |
| launch_last_key | sweep 4 alternate ns/call | ns | 35.9 | 36.1 | +0.2 | +0.6% | 36.2 | +0.3 | +0.8% |
| launch_last_key | sweep 4 repeat ns/call | ns | 33.9 | 33.9 | +0 | +0.0% | 34 | +0.1 | +0.3% |
| launch_readme | readme_path grid=(0,) intj us | us | 0.03 | 0.03 | +0 | +0.0% | 0.03 | +0 | +0.0% |
| launch_readme | readme_path grid=(0,) triton us | us | 12.92 | 13.11 | +0.19 | +1.5% | 12.95 | +0.03 | +0.2% |
| launch_readme | readme_path grid=(1,) intj us | us | 3.12 | 3.24 | +0.12 **R** | +3.8% | 3.13 | +0.01 | +0.3% |
| launch_readme | readme_path grid=(1,) triton us | us | 16.45 | 16.9 | +0.45 | +2.7% | 16.44 | -0.01 | -0.1% |
| launch_readme | readme_torch_access interpreter decode ns | ns | 872.4 | 874.8 | +2.4 | +0.3% | 874.1 | +1.7 | +0.2% |
| launch_readme | readme_torch_access runtime_shim decode ns | ns | 92 | 93.7 | +1.7 | +1.8% | 92.7 | +0.7 | +0.8% |
| launch_readme | readme_torch_access static_compile decode ns | ns | 91.4 | 93.1 | +1.7 | +1.9% | 92 | +0.6 | +0.7% |
| launch_sweep | sweep 16 int ns/call | ns | 67.9 | 68.6 | +0.7 | +1.0% | 68.4 | +0.5 | +0.7% |
| launch_sweep | sweep 16 tensor ns/call | ns | 65.2 | 64.8 | -0.4 | -0.6% | 64.8 | -0.4 | -0.6% |
| launch_sweep | sweep 32 int ns/call | ns | 112.8 | 106.5 | -6.3 | -5.6% | 106.2 | -6.6 | -5.9% |
| launch_sweep | sweep 32 tensor ns/call | ns | 114 | 111.5 | -2.5 | -2.2% | 110.8 | -3.2 | -2.8% |
| launch_sweep | sweep 4 int ns/call | ns | 40.5 | 40.7 | +0.2 | +0.5% | 40.8 | +0.3 | +0.7% |
| launch_sweep | sweep 4 tensor ns/call | ns | 39.2 | 39.5 | +0.3 | +0.8% | 39.4 | +0.2 | +0.5% |

Rows new in task1 (no baseline): kernel_cache/1-word, kernel_cache/1-word, kernel_cache/1-word, kernel_cache/2-word, kernel_cache/2-word, kernel_cache/2-word, kernel_cache/5-word, kernel_cache/5-word, kernel_cache/5-word.

## Diagnostic: interleaved baseline vs task1 (`ffi_paths`, host-only rows)

Baseline checkout at `cab358b` in a git worktree, task1 working tree at `e3c94b5`
(uncommitted at the time, byte-identical to it), alternated 5 times, each a separate
process, `PYTHONPATH=<tree> taskset -c 0 /tmp/gb2/bin/python
benchmarks/bench_intj_ffi_paths.py --mode all --iters 1000 --batches 9`. Medians (ns)
of `hot call`, `3-tensor host-only nop`, `direct positional`, `pair direct`,
`config direct`:

```text
base 1: 37.5 33.5 64.7 66.6 80.8    task1 1: 37.7 33.7 65.1 68.7 78.6
base 2: 37.3 33.6 64.6 66.2 79.4    task1 2: 39.5 34.1 66.5 67.2 79.7
base 3: 37.3 33.5 65.1 66.9 80.0    task1 3: 38.5 34.2 65.8 66.3 82.2
base 4: 38.2 33.4 65.2 67.2 80.4    task1 4: 38.6 35.4 65.5 67.2 79.7
base 5: 38.1 33.3 64.6 66.9 79.1    task1 5: 39.3 34.2 65.4 67.2 78.5
median: 37.5 33.5 64.7 66.9 80.0    median:  38.6 34.2 65.5 67.2 79.7
```

## Raw outputs

### task1

#### launch_gpu

round 0:
```text
commit=e3c94b54ffe8159ea62fb7f94a22a7fc6b331c1f
started=2026-09-26T22:41:14+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --iters 1000 --batches 9
mode=gpu; 1000 calls × 9 batches; median ns/call
               path    ns/call       Δ ns       Δ %   build ms   bind ms
           auto map     4069.2       +0.0      +0.0    2426.83         -
        reduced key     3813.3     -255.8      -6.3    2208.17         -
         verify off     3873.1     -196.0      -4.8    2228.27         -
          verify on     3837.6     -231.6      -5.7    2261.04         -
              baked     3803.1     -266.1      -6.5    2142.74         -
       bound tensor     3826.8     -242.4      -6.0       0.50   2188.55
      bound pointer     3899.0     -170.2      -4.2       0.50   2100.94
   fixed device map     3877.6     -191.6      -4.7       0.46   2246.18
fixed device no-map     3875.6     -193.6      -4.8       0.52   2090.78
elapsed_ns=23782335888
exit_status=0
ended=2026-09-26T22:41:38+08:00
```

round 1:
```text
commit=e3c94b54ffe8159ea62fb7f94a22a7fc6b331c1f
started=2026-09-26T22:43:28+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --iters 1000 --batches 9
mode=gpu; 1000 calls × 9 batches; median ns/call
               path    ns/call       Δ ns       Δ %   build ms   bind ms
           auto map     3134.5       +0.0      +0.0      74.36         -
        reduced key     3111.9      -22.6      -0.7       3.73         -
         verify off     2969.2     -165.3      -5.3       2.02         -
          verify on     2939.9     -194.6      -6.2       1.98         -
              baked     2986.2     -148.2      -4.7      72.89         -
       bound tensor     3103.9      -30.6      -1.0       0.28      2.00
      bound pointer     3012.3     -122.2      -3.9       0.35      1.91
   fixed device map     2905.9     -228.5      -7.3       0.32      1.89
fixed device no-map     2935.9     -198.6      -6.3       0.28      1.87
elapsed_ns=3870026356
exit_status=0
ended=2026-09-26T22:43:32+08:00
```

round 2:
```text
commit=e3c94b54ffe8159ea62fb7f94a22a7fc6b331c1f
started=2026-09-26T22:44:27+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --iters 1000 --batches 9
mode=gpu; 1000 calls × 9 batches; median ns/call
               path    ns/call       Δ ns       Δ %   build ms   bind ms
           auto map     3112.2       +0.0      +0.0      73.63         -
        reduced key     3051.8      -60.4      -1.9       3.66         -
         verify off     2970.7     -141.5      -4.5       1.98         -
          verify on     2926.6     -185.5      -6.0       1.95         -
              baked     2976.3     -135.9      -4.4      88.21         -
       bound tensor     2975.2     -137.0      -4.4       0.38      1.88
      bound pointer     2940.4     -171.8      -5.5       0.36      1.91
   fixed device map     2910.4     -201.7      -6.5       0.31      1.90
fixed device no-map     2862.2     -250.0      -8.0       0.28      1.85
elapsed_ns=3769094664
exit_status=0
ended=2026-09-26T22:44:30+08:00
```

#### launch_host

round 0:
```text
commit=e3c94b54ffe8159ea62fb7f94a22a7fc6b331c1f
started=2026-09-26T22:41:38+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --no-gpu --iters 100000 --batches 9
mode=host-only; 100000 calls × 9 batches; median ns/call
               path    ns/call       Δ ns       Δ %   build ms   bind ms
           auto map       42.1       +0.0      +0.0      62.07         -
        reduced key       42.0       -0.1      -0.2       1.85         -
         verify off       42.4       +0.3      +0.6       1.78         -
          verify on       43.6       +1.5      +3.5       1.70         -
              baked       39.9       -2.2      -5.3       3.09         -
       bound tensor       46.2       +4.1      +9.8       0.15      1.63
      bound pointer       45.3       +3.2      +7.5       0.13      1.60
   fixed device map       42.4       +0.3      +0.8       0.11      1.59
fixed device no-map       44.2       +2.1      +4.9       0.14      1.55
elapsed_ns=3031842737
exit_status=0
ended=2026-09-26T22:41:41+08:00
```

round 1:
```text
commit=e3c94b54ffe8159ea62fb7f94a22a7fc6b331c1f
started=2026-09-26T22:43:32+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --no-gpu --iters 100000 --batches 9
mode=host-only; 100000 calls × 9 batches; median ns/call
               path    ns/call       Δ ns       Δ %   build ms   bind ms
           auto map       42.4       +0.0      +0.0      60.93         -
        reduced key       41.9       -0.5      -1.2       1.85         -
         verify off       42.5       +0.1      +0.2       1.73         -
          verify on       43.4       +1.1      +2.5       1.73         -
              baked       39.6       -2.8      -6.5       2.81         -
       bound tensor       45.8       +3.4      +8.1       0.15      1.60
      bound pointer       45.2       +2.9      +6.8       0.14      1.58
   fixed device map       43.0       +0.6      +1.5       0.11      1.59
fixed device no-map       43.4       +1.0      +2.4       0.15      1.54
elapsed_ns=2982579457
exit_status=0
ended=2026-09-26T22:43:35+08:00
```

round 2:
```text
commit=e3c94b54ffe8159ea62fb7f94a22a7fc6b331c1f
started=2026-09-26T22:44:30+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --no-gpu --iters 100000 --batches 9
mode=host-only; 100000 calls × 9 batches; median ns/call
               path    ns/call       Δ ns       Δ %   build ms   bind ms
           auto map       42.3       +0.0      +0.0      61.11         -
        reduced key       42.0       -0.3      -0.7       9.00         -
         verify off       42.0       -0.3      -0.7       1.76         -
          verify on       43.2       +0.9      +2.0       1.66         -
              baked       39.3       -3.0      -7.2       2.92         -
       bound tensor       46.2       +3.9      +9.2       0.15      1.57
      bound pointer       45.3       +3.0      +7.1       0.13      1.60
   fixed device map       42.8       +0.5      +1.2       0.11      1.55
fixed device no-map       44.1       +1.7      +4.1       0.14      1.55
elapsed_ns=3025857095
exit_status=0
ended=2026-09-26T22:44:33+08:00
```

#### launch_readme

round 0:
```text
commit=e3c94b54ffe8159ea62fb7f94a22a7fc6b331c1f
started=2026-09-26T22:41:41+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --readme --iters 1000 --batches 9
mode=readme; 1000 calls × 9 batches; median
        path   triton us    intj us   speedup
   grid=(1,)       16.90       3.24      5.2x
   grid=(0,)       13.07       0.07    195.5x

torch_access_mode   decode ns    build s
     runtime_shim        96.1       0.76
   static_compile        93.1       0.13
      interpreter       883.4       0.80
elapsed_ns=5483246268
exit_status=0
ended=2026-09-26T22:41:47+08:00
```

round 1:
```text
commit=e3c94b54ffe8159ea62fb7f94a22a7fc6b331c1f
started=2026-09-26T22:43:35+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --readme --iters 1000 --batches 9
mode=readme; 1000 calls × 9 batches; median
        path   triton us    intj us   speedup
   grid=(1,)       17.01       3.77      4.5x
   grid=(0,)       13.11       0.03    466.6x

torch_access_mode   decode ns    build s
     runtime_shim        93.7       0.02
   static_compile        94.0       0.06
      interpreter       867.9       0.00
elapsed_ns=3751826767
exit_status=0
ended=2026-09-26T22:43:39+08:00
```

round 2:
```text
commit=e3c94b54ffe8159ea62fb7f94a22a7fc6b331c1f
started=2026-09-26T22:44:33+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --readme --iters 1000 --batches 9
mode=readme; 1000 calls × 9 batches; median
        path   triton us    intj us   speedup
   grid=(1,)       16.42       3.16      5.2x
   grid=(0,)       13.13       0.03    463.5x

torch_access_mode   decode ns    build s
     runtime_shim        90.8       0.02
   static_compile        89.5       0.06
      interpreter       874.8       0.00
elapsed_ns=3807525847
exit_status=0
ended=2026-09-26T22:44:37+08:00
```

#### launch_sweep

round 0:
```text
commit=e3c94b54ffe8159ea62fb7f94a22a7fc6b331c1f
started=2026-09-26T22:41:47+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --sweep --iters 100000 --batches 9
mode=sweep; host-only; 100000 calls × 9 batches; median ns/call
count   kind    ns/call
    4    int       40.1
    4 tensor       86.4
   16    int       68.2
   16 tensor       64.8
   32    int      107.1
   32 tensor      109.9
elapsed_ns=12337123503
exit_status=0
ended=2026-09-26T22:41:59+08:00
```

round 1:
```text
commit=e3c94b54ffe8159ea62fb7f94a22a7fc6b331c1f
started=2026-09-26T22:43:39+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --sweep --iters 100000 --batches 9
mode=sweep; host-only; 100000 calls × 9 batches; median ns/call
count   kind    ns/call
    4    int       40.8
    4 tensor       39.3
   16    int       68.6
   16 tensor       65.0
   32    int      106.0
   32 tensor      112.4
elapsed_ns=3037938515
exit_status=0
ended=2026-09-26T22:43:42+08:00
```

round 2:
```text
commit=e3c94b54ffe8159ea62fb7f94a22a7fc6b331c1f
started=2026-09-26T22:44:37+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --sweep --iters 100000 --batches 9
mode=sweep; host-only; 100000 calls × 9 batches; median ns/call
count   kind    ns/call
    4    int       40.7
    4 tensor       39.5
   16    int       68.8
   16 tensor       64.8
   32    int      106.5
   32 tensor      111.5
elapsed_ns=3106760924
exit_status=0
ended=2026-09-26T22:44:40+08:00
```

#### launch_last_key

round 0:
```text
commit=e3c94b54ffe8159ea62fb7f94a22a7fc6b331c1f
started=2026-09-26T22:41:59+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --last-key --iters 100000 --batches 9
mode=last-key; host-only; 200000 calls × 9 batches; median ns/call
count     pattern    ns/call
    4      repeat       33.8
    4   alternate       36.1
   16      repeat       62.7
   16   alternate       65.3
   32      repeat       99.5
   32   alternate      104.1
elapsed_ns=3548989678
exit_status=0
ended=2026-09-26T22:42:02+08:00
```

round 1:
```text
commit=e3c94b54ffe8159ea62fb7f94a22a7fc6b331c1f
started=2026-09-26T22:43:42+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --last-key --iters 100000 --batches 9
mode=last-key; host-only; 200000 calls × 9 batches; median ns/call
count     pattern    ns/call
    4      repeat       33.9
    4   alternate       36.1
   16      repeat       62.7
   16   alternate       65.2
   32      repeat       99.8
   32   alternate      104.3
elapsed_ns=3391542693
exit_status=0
ended=2026-09-26T22:43:45+08:00
```

round 2:
```text
commit=e3c94b54ffe8159ea62fb7f94a22a7fc6b331c1f
started=2026-09-26T22:44:40+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --last-key --iters 100000 --batches 9
mode=last-key; host-only; 200000 calls × 9 batches; median ns/call
count     pattern    ns/call
    4      repeat       34.0
    4   alternate       36.4
   16      repeat       62.6
   16   alternate       65.5
   32      repeat       99.8
   32   alternate      104.3
elapsed_ns=3346731281
exit_status=0
ended=2026-09-26T22:44:44+08:00
```

#### ffi_compare

round 0:
```text
commit=e3c94b54ffe8159ea62fb7f94a22a7fc6b331c1f
started=2026-09-26T22:42:02+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_ffi_compare.py --iters 1000 --batches 9
iters=1000 batches=9 unit=us/call (host-only, no timed sync)
FFI packed nop             median=0.1456 samples=[0.293531, 0.146761, 0.14397100000000002, 0.14590899999999998, 0.14547200000000002, 0.1472, 0.14284, 0.14474199999999998, 0.145588]
FFI typed nop              median=0.1483 samples=[0.250896, 0.149885, 0.145434, 0.148342, 0.15367699999999998, 0.14740299999999998, 0.14652, 0.150469, 0.147595]
FFI empty kernel           median=3.8724 samples=[15.57066, 3.979712, 3.872417, 3.8287739999999997, 3.695574, 3.882972, 3.724836, 3.89335, 3.7349580000000002]
INTJ empty kernel          median=3.4988 samples=[3.156485, 3.666043, 3.531633, 3.4988249999999996, 3.40777, 3.537426, 3.404156, 3.520822, 3.468849]
INTJ fixed-device kernel   median=3.5007 samples=[3.3075300000000003, 3.505216, 3.61863, 3.3234470000000003, 3.5006779999999997, 3.301732, 3.503629, 3.347875, 3.5463310000000003]
FFI packed nop mixed       median=0.1736 samples=[0.179312, 0.169911, 0.172567, 0.17474199999999998, 0.174319, 0.173456, 0.17091800000000001, 0.173641, 0.17363399999999998]
FFI typed nop mixed        median=0.1755 samples=[0.175533, 0.17404499999999998, 0.173238, 0.178844, 0.175758, 0.178537, 0.175403, 0.175474, 0.17297200000000001]
FFI mixed kernel           median=3.7780 samples=[3.7763739999999997, 4.10861, 3.661098, 3.869461, 3.6945859999999997, 3.926754, 3.777996, 3.922858, 3.7684029999999997]
INTJ mixed kernel          median=3.5217 samples=[3.563724, 3.68448, 3.4136379999999997, 3.540607, 3.356376, 3.52277, 3.448308, 3.521715, 3.498261]
INTJ fixed mixed kernel    median=3.5211 samples=[3.6757199999999997, 3.722851, 3.5342510000000003, 3.521126, 3.27488, 3.4983299999999997, 3.4721379999999997, 3.535148, 3.48495]
elapsed_ns=12508836057
exit_status=0
ended=2026-09-26T22:42:15+08:00
```

round 1:
```text
commit=e3c94b54ffe8159ea62fb7f94a22a7fc6b331c1f
started=2026-09-26T22:43:45+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_ffi_compare.py --iters 1000 --batches 9
iters=1000 batches=9 unit=us/call (host-only, no timed sync)
FFI packed nop             median=0.1448 samples=[0.149281, 0.146133, 0.145102, 0.14475100000000002, 0.14433, 0.14315199999999997, 0.143769, 0.145394, 0.14463900000000002]
FFI typed nop              median=0.1495 samples=[0.149821, 0.15201800000000001, 0.14947200000000002, 0.148947, 0.149371, 0.151831, 0.147142, 0.15152000000000002, 0.14766900000000002]
FFI empty kernel           median=3.3563 samples=[3.555201, 3.413861, 3.387921, 3.342398, 3.273292, 3.356431, 3.192703, 3.3563449999999997, 3.187922]
INTJ empty kernel          median=2.9952 samples=[3.058427, 3.0549609999999996, 3.058477, 2.868254, 2.8305100000000003, 2.901134, 2.995175, 2.890358, 3.0135430000000003]
INTJ fixed-device kernel   median=2.8408 samples=[3.114072, 3.074446, 3.0465109999999997, 2.806398, 2.840801, 2.815496, 2.798684, 2.8180970000000003, 2.9069789999999998]
FFI packed nop mixed       median=0.1742 samples=[0.174724, 0.17313900000000002, 0.17509899999999998, 0.174406, 0.17410499999999998, 0.17341, 0.174277, 0.174194, 0.173368]
FFI typed nop mixed        median=0.1769 samples=[0.175977, 0.18127600000000002, 0.177574, 0.177728, 0.176393, 0.176886, 0.175339, 0.178318, 0.17635499999999998]
FFI mixed kernel           median=3.2782 samples=[3.384214, 3.467695, 3.270233, 3.269395, 3.27815, 3.353027, 3.269669, 3.389078, 3.2727600000000003]
INTJ mixed kernel          median=2.9044 samples=[3.195481, 3.094784, 2.9274270000000002, 2.911927, 2.841665, 2.874519, 2.831897, 2.904443, 2.834644]
INTJ fixed mixed kernel    median=2.9079 samples=[3.141985, 3.126161, 2.973745, 2.973241, 2.907931, 2.867916, 2.8589879999999996, 2.8923319999999997, 2.866194]
elapsed_ns=3816417378
exit_status=0
ended=2026-09-26T22:43:49+08:00
```

round 2:
```text
commit=e3c94b54ffe8159ea62fb7f94a22a7fc6b331c1f
started=2026-09-26T22:44:44+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_ffi_compare.py --iters 1000 --batches 9
iters=1000 batches=9 unit=us/call (host-only, no timed sync)
FFI packed nop             median=0.1438 samples=[0.150238, 0.142837, 0.14286600000000002, 0.143813, 0.14619900000000002, 0.145623, 0.148028, 0.141058, 0.143108]
FFI typed nop              median=0.1477 samples=[0.147728, 0.15024600000000002, 0.146342, 0.152627, 0.148817, 0.147709, 0.14743799999999999, 0.15065299999999998, 0.143673]
FFI empty kernel           median=3.3937 samples=[3.629623, 3.446952, 3.4321080000000004, 3.374165, 3.28137, 3.4085520000000002, 3.227846, 3.3936509999999998, 3.241361]
INTJ empty kernel          median=3.0139 samples=[3.065613, 3.0912159999999997, 3.072789, 2.860995, 2.835982, 2.986109, 3.0138890000000003, 2.916258, 3.024151]
INTJ fixed-device kernel   median=2.8330 samples=[3.035557, 3.130099, 3.085386, 2.809469, 2.8774409999999997, 2.833036, 2.791984, 2.818128, 2.799171]
FFI packed nop mixed       median=0.1729 samples=[0.174627, 0.171104, 0.17135, 0.169187, 0.171578, 0.173195, 0.172906, 0.17494200000000001, 0.173173]
FFI typed nop mixed        median=0.1755 samples=[0.17547300000000002, 0.17783600000000002, 0.174403, 0.180844, 0.17324, 0.176876, 0.174841, 0.178181, 0.172261]
FFI mixed kernel           median=3.3314 samples=[3.415778, 3.500213, 3.3314340000000002, 3.299979, 3.278175, 3.333332, 3.3030340000000002, 3.399762, 3.2916469999999998]
INTJ mixed kernel          median=2.9209 samples=[3.1648330000000002, 3.1206810000000003, 2.920886, 2.935864, 2.835757, 2.858745, 2.832306, 2.9730990000000004, 2.828991]
INTJ fixed mixed kernel    median=2.9107 samples=[3.142798, 3.138072, 2.958647, 2.987127, 2.863288, 2.8599720000000004, 2.902692, 2.910686, 2.874266]
elapsed_ns=3714037656
exit_status=0
ended=2026-09-26T22:44:47+08:00
```

#### ffi_sweep

round 0:
```text
commit=e3c94b54ffe8159ea62fb7f94a22a7fc6b331c1f
started=2026-09-26T22:42:15+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_ffi_compare.py --sweep --iters 1000 --batches 9
counts=[0, 3, 5, 8, 16, 32, 64] iters=1000 batches=9 unit=us/call (host-only, no timed sync); INTJ device-bound, no specialization key
args= 0 FFI packed nop           median=0.1171 samples=[0.126777, 0.181371, 0.182597, 0.11664400000000001, 0.116914, 0.116559, 0.117753, 0.11706100000000001, 0.116742]
args= 0 FFI typed nop            median=0.1220 samples=[0.120017, 0.192579, 0.19755799999999998, 0.12177299999999999, 0.11835, 0.124044, 0.120637, 0.124188, 0.122021]
args= 0 FFI empty kernel         median=1.8540 samples=[1.704803, 2.65396, 2.342733, 1.8912639999999998, 1.609934, 1.8891980000000002, 1.600809, 1.853966, 1.585899]
args= 0 INTJ static_compile kernel median=2.7450 samples=[3.164015, 3.416395, 3.2916860000000003, 2.754369, 2.739638, 2.744972, 2.7296799999999997, 2.728709, 2.7185390000000003]
args= 0 INTJ runtime_shim kernel median=2.7388 samples=[2.91496, 3.3456550000000003, 3.305131, 2.6822510000000004, 2.74533, 2.686859, 2.7387829999999997, 2.67409, 2.714962]
args= 3 FFI packed nop           median=0.1445 samples=[0.145367, 0.227632, 0.233127, 0.143935, 0.144337, 0.144399, 0.145911, 0.144506, 0.142337]
args= 3 FFI typed nop            median=0.1494 samples=[0.14940299999999998, 0.254152, 0.24831399999999998, 0.147173, 0.147333, 0.149815, 0.149342, 0.14959299999999998, 0.14849600000000002]
args= 3 FFI empty kernel         median=3.3125 samples=[3.39202, 4.183565, 4.161393, 3.312473, 3.231411, 3.322046, 3.2051410000000002, 3.310305, 3.205578]
args= 3 INTJ static_compile kernel median=2.8604 samples=[3.1189679999999997, 3.518303, 3.565728, 2.912715, 2.860441, 2.840068, 2.849287, 2.8468519999999997, 2.8440100000000004]
args= 3 INTJ runtime_shim kernel median=2.8925 samples=[3.0645819999999997, 3.609234, 3.595379, 3.247135, 2.892523, 2.792725, 2.866387, 2.7961810000000002, 2.8625149999999997]
args= 5 FFI packed nop           median=0.1750 samples=[0.17497100000000002, 0.267798, 0.27851, 0.277995, 0.173322, 0.17318, 0.173944, 0.180637, 0.17274899999999999]
args= 5 FFI typed nop            median=0.1788 samples=[0.17588399999999998, 0.28708, 0.28924500000000003, 0.29638400000000004, 0.178037, 0.178799, 0.176851, 0.18965700000000002, 0.17447200000000002]
args= 5 FFI empty kernel         median=3.2882 samples=[3.402783, 4.252353, 4.257915, 4.213429, 3.260487, 3.2847359999999997, 3.229897, 3.288248, 3.232641]
args= 5 INTJ static_compile kernel median=2.8833 samples=[3.038595, 3.632236, 3.592316, 3.5364319999999996, 2.883308, 2.8741849999999998, 2.869287, 2.872541, 2.8565549999999997]
args= 5 INTJ runtime_shim kernel median=3.0669 samples=[3.066913, 3.663301, 3.548375, 3.58589, 3.685917, 2.894925, 2.840563, 2.894748, 2.836987]
args= 8 FFI packed nop           median=0.2107 samples=[0.203911, 0.326102, 0.340229, 0.319151, 0.207111, 0.203369, 0.20591, 0.211647, 0.210703]
args= 8 FFI typed nop            median=0.2116 samples=[0.20843299999999998, 0.343884, 0.335203, 0.344591, 0.208031, 0.212989, 0.21022300000000002, 0.21161000000000002, 0.206893]
args= 8 FFI empty kernel         median=3.5308 samples=[3.530844, 4.4584470000000005, 4.40751, 4.397593, 5.273926, 3.519702, 3.295882, 3.4797510000000003, 3.279534]
args= 8 INTJ static_compile kernel median=3.1272 samples=[3.1272249999999997, 3.6773200000000004, 3.740835, 3.657616, 7.961064, 3.025846, 3.081016, 3.006045, 3.108393]
args= 8 INTJ runtime_shim kernel median=2.9693 samples=[3.092101, 3.75385, 3.7719989999999997, 3.792564, 2.9693470000000004, 2.89977, 2.871015, 2.903512, 2.883918]
args=16 FFI packed nop           median=0.2935 samples=[0.29016000000000003, 0.437127, 0.445091, 0.446748, 0.294085, 0.29066000000000003, 0.291443, 0.292737, 0.293531]
args=16 FFI typed nop            median=0.3075 samples=[0.310114, 0.48148399999999997, 0.463589, 0.46917899999999996, 0.29802300000000004, 0.302562, 0.300666, 0.307497, 0.299413]
args=16 FFI empty kernel         median=3.7771 samples=[3.821931, 4.946445, 4.863925, 4.853167999999999, 3.777134, 3.732638, 3.578152, 3.731176, 3.591642]
args=16 INTJ static_compile kernel median=3.4522 samples=[3.2982750000000003, 4.2113119999999995, 4.192009, 4.091610999999999, 3.6089160000000002, 3.3625770000000004, 3.4521770000000003, 3.3667800000000003, 3.4020859999999997]
args=16 INTJ runtime_shim kernel median=3.3217 samples=[3.294998, 4.143036, 4.083558, 4.014669, 3.392688, 3.065086, 3.3217109999999996, 3.0616309999999998, 3.3151170000000003]
args=32 FFI packed nop           median=0.4822 samples=[0.482228, 0.641708, 0.703311, 0.690539, 0.472143, 0.481739, 0.476266, 0.47029000000000004, 0.487889]
args=32 FFI typed nop            median=0.4900 samples=[0.48553199999999996, 0.795997, 0.7429690000000001, 0.727447, 0.485548, 0.48996100000000004, 0.488404, 0.49508800000000003, 0.48608100000000004]
args=32 FFI empty kernel         median=4.2950 samples=[4.295012, 20.331774000000003, 5.603758, 5.7116940000000005, 4.1535150000000005, 4.323867, 4.15955, 4.267539, 4.147714]
args=32 INTJ static_compile kernel median=3.7995 samples=[3.4966039999999996, 3.6517530000000002, 4.692422, 4.666604, 3.6668290000000003, 3.581368, 3.7994899999999996, 3.8312199999999996, 3.809278]
args=32 INTJ runtime_shim kernel median=3.6734 samples=[3.4673499999999997, 3.480967, 4.658596999999999, 4.687395, 3.673416, 3.449835, 3.697403, 3.43278, 3.709209]
args=64 FFI packed nop           median=0.8452 samples=[0.832234, 0.845186, 1.168618, 1.174603, 0.852587, 0.838795, 0.8375159999999999, 0.8405900000000001, 0.8491230000000001]
args=64 FFI typed nop            median=0.8692 samples=[0.869375, 0.8679020000000001, 1.2544110000000002, 1.2464549999999999, 0.869242, 0.86829, 0.8523890000000001, 0.882086, 0.8613569999999999]
args=64 FFI empty kernel         median=5.0758 samples=[5.142015000000001, 5.1212089999999995, 7.025075, 6.951137, 5.060729, 5.075817, 5.041924, 5.05565, 5.035887]
args=64 INTJ static_compile kernel median=4.5968 samples=[4.609966999999999, 4.579318, 6.046869, 5.959487, 4.614299, 4.5967780000000005, 4.583059, 4.590958, 4.576275]
args=64 INTJ runtime_shim kernel median=4.5860 samples=[4.619785, 4.541544, 6.073626, 5.999852, 4.552563, 4.619252, 4.555586, 4.585966, 4.565364]
elapsed_ns=37843226245
exit_status=0
ended=2026-09-26T22:42:53+08:00
```

round 1:
```text
commit=e3c94b54ffe8159ea62fb7f94a22a7fc6b331c1f
started=2026-09-26T22:43:49+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_ffi_compare.py --sweep --iters 1000 --batches 9
counts=[0, 3, 5, 8, 16, 32, 64] iters=1000 batches=9 unit=us/call (host-only, no timed sync); INTJ device-bound, no specialization key
args= 0 FFI packed nop           median=0.1166 samples=[0.11965, 0.113838, 0.116877, 0.11354, 0.117644, 0.116803, 0.116023, 0.116554, 0.116178]
args= 0 FFI typed nop            median=0.1188 samples=[0.11906399999999999, 0.117775, 0.117662, 0.11953, 0.1183, 0.120049, 0.11879300000000001, 0.121051, 0.118684]
args= 0 FFI empty kernel         median=1.6728 samples=[1.633216, 1.907251, 1.672798, 1.869555, 1.635653, 1.882404, 1.6324159999999999, 1.831407, 1.571405]
args= 0 INTJ static_compile kernel median=2.7201 samples=[3.096038, 2.710175, 2.780516, 2.690924, 2.762779, 2.684192, 2.7628739999999996, 2.72012, 2.709925]
args= 0 INTJ runtime_shim kernel median=2.7105 samples=[2.8574119999999996, 2.691871, 2.813471, 2.681888, 2.772862, 2.676318, 2.777072, 2.6729439999999998, 2.7105349999999997]
args= 3 FFI packed nop           median=0.1445 samples=[0.143716, 0.14681, 0.143244, 0.14693, 0.14447300000000002, 0.14418999999999998, 0.144986, 0.14805000000000001, 0.143731]
args= 3 FFI typed nop            median=0.1481 samples=[0.14579499999999998, 0.151525, 0.144676, 0.15165299999999998, 0.148096, 0.149842, 0.147346, 0.153077, 0.147834]
args= 3 FFI empty kernel         median=3.1871 samples=[3.2919929999999997, 3.20125, 3.164337, 3.187083, 3.154898, 3.19192, 3.152431, 3.2101379999999997, 3.179349]
args= 3 INTJ static_compile kernel median=2.8746 samples=[2.9703530000000002, 2.81941, 2.94926, 2.8745909999999997, 2.945798, 2.819517, 2.941338, 2.844225, 2.829402]
args= 3 INTJ runtime_shim kernel median=2.8208 samples=[2.992661, 2.8207739999999997, 2.7841729999999996, 2.8207109999999997, 2.922634, 2.815125, 2.785424, 2.836376, 2.8287869999999997]
args= 5 FFI packed nop           median=0.1718 samples=[0.17380500000000002, 0.16923, 0.17518, 0.17281100000000002, 0.171821, 0.169525, 0.175476, 0.171845, 0.16899]
args= 5 FFI typed nop            median=0.1772 samples=[0.178097, 0.177184, 0.180727, 0.194595, 0.175533, 0.17485, 0.175846, 0.17815899999999998, 0.17544300000000002]
args= 5 FFI empty kernel         median=3.2484 samples=[3.3439989999999997, 3.24843, 3.187268, 3.349169, 3.189189, 3.286489, 3.180475, 3.317296, 3.233451]
args= 5 INTJ static_compile kernel median=2.8514 samples=[2.981233, 2.84512, 2.851356, 2.865438, 2.8515230000000003, 2.8050010000000003, 2.862944, 2.8282800000000003, 2.846591]
args= 5 INTJ runtime_shim kernel median=2.8199 samples=[2.9994009999999998, 2.899571, 2.822487, 2.792634, 2.835341, 2.773606, 2.819907, 2.791672, 2.817762]
args= 8 FFI packed nop           median=0.2060 samples=[0.208094, 0.205988, 0.206428, 0.206653, 0.205843, 0.207412, 0.203564, 0.20588900000000002, 0.202783]
args= 8 FFI typed nop            median=0.2092 samples=[0.20799299999999998, 0.21126699999999998, 0.20866300000000002, 0.211023, 0.210703, 0.209195, 0.206735, 0.210125, 0.205233]
args= 8 FFI empty kernel         median=3.3910 samples=[3.521077, 3.460299, 3.269104, 3.402339, 3.258103, 3.4024870000000003, 3.2731500000000002, 3.390991, 3.243262]
args= 8 INTJ static_compile kernel median=3.0506 samples=[3.055683, 3.0455520000000003, 3.0505709999999997, 3.053769, 3.047447, 2.9733229999999997, 3.0562240000000003, 2.966055, 3.059393]
args= 8 INTJ runtime_shim kernel median=2.9647 samples=[3.049839, 2.981627, 2.97171, 2.908164, 2.982239, 2.88937, 2.9646779999999997, 2.902667, 2.921955]
args=16 FFI packed nop           median=0.2925 samples=[0.294781, 0.294277, 0.292643, 0.29252999999999996, 0.293906, 0.292055, 0.292145, 0.28776999999999997, 0.289705]
args=16 FFI typed nop            median=0.3037 samples=[0.304226, 0.306967, 0.30540100000000003, 0.30667700000000003, 0.30115499999999995, 0.303685, 0.30362, 0.30238, 0.302017]
args=16 FFI empty kernel         median=3.6725 samples=[3.737894, 3.760295, 3.5368690000000003, 3.685331, 3.544256, 3.672467, 3.5355839999999996, 3.675393, 3.5392550000000003]
args=16 INTJ static_compile kernel median=3.4450 samples=[3.2672939999999997, 3.4450410000000002, 3.5012800000000004, 3.3488789999999997, 3.482593, 3.3388, 3.512207, 3.3332689999999996, 3.47519]
args=16 INTJ runtime_shim kernel median=3.2793 samples=[3.279308, 3.220779, 3.3251720000000002, 3.0551120000000003, 3.376913, 3.039115, 3.344276, 3.047122, 3.3700259999999997]
args=32 FFI packed nop           median=0.4788 samples=[0.471308, 0.47702300000000003, 0.47946, 0.47985700000000003, 0.498056, 0.471528, 0.47981599999999996, 0.478796, 0.477103]
args=32 FFI typed nop            median=0.4924 samples=[0.497064, 0.496589, 0.49238099999999996, 0.492336, 0.485073, 0.491255, 0.48473, 0.497527, 0.493911]
args=32 FFI empty kernel         median=4.2559 samples=[4.303316, 4.318925, 4.111542, 4.266889, 4.1025540000000005, 4.255938, 4.113262, 4.2599920000000004, 4.086174]
args=32 INTJ static_compile kernel median=3.6379 samples=[3.4473730000000002, 3.676231, 3.644242, 3.588992, 3.65981, 3.581782, 3.671342, 3.540233, 3.637862]
args=32 INTJ runtime_shim kernel median=3.5810 samples=[3.52195, 3.5810340000000003, 3.676158, 3.5131509999999997, 3.696167, 3.490585, 3.6994569999999998, 3.499002, 3.669002]
args=64 FFI packed nop           median=0.8409 samples=[0.853851, 0.851889, 0.8389650000000001, 0.834995, 0.8428260000000001, 0.847404, 0.8409139999999999, 0.835671, 0.8356399999999999]
args=64 FFI typed nop            median=0.8608 samples=[0.8480979999999999, 0.8627870000000001, 0.8573379999999999, 0.864783, 0.853911, 0.8713110000000001, 0.877082, 0.857932, 0.8608450000000001]
args=64 FFI empty kernel         median=5.0569 samples=[5.0875259999999995, 5.139543, 5.051137, 5.046022, 5.056946, 5.0646130000000005, 5.0498199999999995, 5.034517, 5.091729999999999]
args=64 INTJ static_compile kernel median=4.5956 samples=[4.895236, 4.624535, 4.597553, 4.595596, 4.558871, 4.587731, 4.588749, 4.60163, 4.593229999999999]
args=64 INTJ runtime_shim kernel median=4.5826 samples=[4.890524, 4.5831610000000005, 4.5470299999999995, 4.614669, 4.518708999999999, 4.582618999999999, 4.545673, 4.583557, 4.490164]
elapsed_ns=4318075996
exit_status=0
ended=2026-09-26T22:43:53+08:00
```

round 2:
```text
commit=e3c94b54ffe8159ea62fb7f94a22a7fc6b331c1f
started=2026-09-26T22:44:47+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_ffi_compare.py --sweep --iters 1000 --batches 9
counts=[0, 3, 5, 8, 16, 32, 64] iters=1000 batches=9 unit=us/call (host-only, no timed sync); INTJ device-bound, no specialization key
args= 0 FFI packed nop           median=0.1168 samples=[0.122559, 0.11651399999999999, 0.118415, 0.117023, 0.114943, 0.116297, 0.12182599999999999, 0.116786, 0.116369]
args= 0 FFI typed nop            median=0.1221 samples=[0.122586, 0.122989, 0.118448, 0.122061, 0.120367, 0.122558, 0.118397, 0.122921, 0.116877]
args= 0 FFI empty kernel         median=2.2685 samples=[1.745678, 3.058576, 2.119948, 3.055103, 2.2260720000000003, 3.006513, 2.268542, 3.022114, 2.230816]
args= 0 INTJ static_compile kernel median=3.2270 samples=[3.332045, 3.203815, 3.172626, 3.226967, 3.285895, 3.2191590000000003, 3.156172, 3.311973, 3.28016]
args= 0 INTJ runtime_shim kernel median=3.1615 samples=[3.1614780000000002, 2.999976, 3.2515039999999997, 3.0387440000000003, 3.297737, 3.062935, 3.233761, 3.0725100000000003, 3.2932609999999998]
args= 3 FFI packed nop           median=0.1450 samples=[0.148348, 0.145694, 0.145259, 0.14518999999999999, 0.14497100000000002, 0.14255199999999998, 0.14415, 0.14388900000000002, 0.145031]
args= 3 FFI typed nop            median=0.1487 samples=[0.147405, 0.148703, 0.14821199999999998, 0.150171, 0.14804499999999998, 0.150803, 0.15155000000000002, 0.150008, 0.148012]
args= 3 FFI empty kernel         median=3.6928 samples=[3.692793, 3.778937, 3.573589, 3.825732, 3.563253, 3.820052, 3.583116, 3.867758, 3.5534250000000003]
args= 3 INTJ static_compile kernel median=3.5686 samples=[3.311706, 3.429155, 3.867056, 3.545795, 3.887003, 3.568614, 3.8767289999999996, 3.520708, 3.9380949999999997]
args= 3 INTJ runtime_shim kernel median=3.4187 samples=[3.309931, 3.23532, 4.018774, 3.40369, 3.418728, 3.402965, 3.476147, 3.433987, 4.110639]
args= 5 FFI packed nop           median=0.1734 samples=[0.175318, 0.17342, 0.174263, 0.17536500000000002, 0.17363, 0.173, 0.172721, 0.172777, 0.171637]
args= 5 FFI typed nop            median=0.1765 samples=[0.176451, 0.177083, 0.177376, 0.19281399999999999, 0.17487200000000003, 0.175933, 0.175303, 0.178237, 0.173428]
args= 5 FFI empty kernel         median=3.7752 samples=[3.728078, 4.349438, 3.773978, 4.304179, 3.756614, 4.2559629999999995, 3.775224, 4.285226, 3.7135320000000003]
args= 5 INTJ static_compile kernel median=3.5020 samples=[3.31323, 3.509718, 3.501847, 3.59693, 3.452719, 3.519476, 3.5020320000000003, 3.625206, 3.4809639999999997]
args= 5 INTJ runtime_shim kernel median=3.4670 samples=[3.362803, 3.143458, 3.502796, 3.5411729999999997, 3.431569, 3.407467, 3.5099009999999997, 3.549226, 3.4669659999999998]
args= 8 FFI packed nop           median=0.2074 samples=[0.206625, 0.20739, 0.207724, 0.20966200000000002, 0.206089, 0.210518, 0.20485499999999998, 0.205708, 0.20749700000000001]
args= 8 FFI typed nop            median=0.2099 samples=[0.210228, 0.21009899999999998, 0.21245, 0.20950200000000002, 0.209533, 0.208361, 0.20860900000000002, 0.212566, 0.209874]
args= 8 FFI empty kernel         median=4.0253 samples=[4.025315, 4.31722, 3.831145, 4.255171000000001, 3.742073, 4.276272, 3.818594, 4.281234, 3.761355]
args= 8 INTJ static_compile kernel median=3.6557 samples=[3.407278, 3.548237, 4.031331, 3.6471489999999998, 4.034246, 3.6140770000000004, 4.036219, 3.655734, 3.998119]
args= 8 INTJ runtime_shim kernel median=3.4105 samples=[3.4105090000000002, 3.272903, 3.51788, 3.2849009999999996, 3.5117800000000003, 3.246427, 3.5462629999999997, 3.277846, 3.4935039999999997]
args=16 FFI packed nop           median=0.2931 samples=[0.296916, 0.299132, 0.296742, 0.29139499999999996, 0.293918, 0.29188, 0.29171199999999997, 0.29282400000000003, 0.29307799999999995]
args=16 FFI typed nop            median=0.3044 samples=[0.300406, 0.30620400000000003, 0.3049, 0.305387, 0.304399, 0.308419, 0.301084, 0.302812, 0.29651299999999997]
args=16 FFI empty kernel         median=4.5028 samples=[4.219658000000001, 5.853928, 4.109304, 5.751982, 4.161613, 5.823232, 4.139278, 5.602004, 4.502782]
args=16 INTJ static_compile kernel median=5.1589 samples=[3.778684, 5.158912999999999, 5.425649999999999, 5.063818, 5.253666, 5.113747, 4.859979, 14.740731, 5.189443]
args=16 INTJ runtime_shim kernel median=4.4727 samples=[3.7759699999999996, 3.6784499999999998, 4.8802389999999995, 3.618165, 5.33324, 3.600734, 4.4727250000000005, 12.690062, 4.771201]
args=32 FFI packed nop           median=0.4809 samples=[0.48057900000000003, 0.484271, 0.482772, 0.472134, 0.482559, 0.475265, 0.471158, 0.480859, 0.481748]
args=32 FFI typed nop            median=0.4951 samples=[0.492222, 0.5065419999999999, 0.7993920000000001, 0.503341, 0.493906, 0.495513, 0.48911099999999996, 0.49509699999999995, 0.49354899999999996]
args=32 FFI empty kernel         median=5.8722 samples=[4.879246, 5.87217, 20.107962, 5.953824, 4.801806, 5.990608, 4.8590860000000005, 5.961755, 5.092849]
args=32 INTJ static_compile kernel median=5.4980 samples=[3.9811080000000003, 5.430534, 11.236671, 5.432322, 5.542768, 5.497997000000001, 6.29326, 5.345362, 6.268993]
args=32 INTJ runtime_shim kernel median=4.2149 samples=[4.001449, 4.214898, 5.5803519999999995, 4.049398, 5.4877579999999995, 4.052433, 5.807949, 4.058941, 5.783416]
args=64 FFI packed nop           median=0.8415 samples=[0.835398, 0.858289, 0.867197, 0.865206, 0.860592, 0.833503, 0.834005, 0.839678, 0.841456]
args=64 FFI typed nop            median=0.8624 samples=[0.881638, 0.8864460000000001, 0.894283, 0.896779, 0.860041, 0.861943, 0.852854, 0.862404, 0.858656]
args=64 FFI empty kernel         median=6.0472 samples=[5.714928, 6.3929030000000004, 6.0471710000000005, 6.498583, 5.784047999999999, 6.416393, 5.9072309999999995, 6.377143, 5.815269]
args=64 INTJ static_compile kernel median=6.0878 samples=[5.283791, 6.044627, 6.14627, 6.0989700000000004, 6.045625, 6.051037999999999, 6.109515999999999, 6.090651, 6.087791]
args=64 INTJ runtime_shim kernel median=5.9560 samples=[5.099134, 6.152953, 5.918675, 6.125757999999999, 5.956034, 6.0677129999999995, 5.939426, 6.053497, 5.856754]
elapsed_ns=4655626165
exit_status=0
ended=2026-09-26T22:44:52+08:00
```

#### ffi_paths

round 0:
```text
commit=e3c94b54ffe8159ea62fb7f94a22a7fc6b331c1f
started=2026-09-26T22:42:53+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_intj_ffi_paths.py --mode all --iters 1000 --batches 9
iters=1000 batches=9 unit=ns/call; no_gpu=True
mode=callback
callback_count=9100 (100 warmup + 9000 unique misses)
INTJ cold compile callback + cache  median_ns=406.9 samples_ns=[431.943, 468.503, 379.942, 1242.057, 374.237, 387.376, 402.684, 406.924, 2097.77]
INTJ hot call (no callback)         median_ns=39.1 samples_ns=[40.19, 39.66, 37.882, 39.663, 41.677, 38.856, 37.761, 39.083, 38.463]
INTJ 3-tensor host-only nop         median_ns=33.9 samples_ns=[37.243, 38.05, 33.859, 34.759, 33.674, 33.94, 33.538, 34.159, 33.814]
mode=kwargs
INTJ direct positional              median_ns=65.4 samples_ns=[67.567, 65.49, 65.308, 65.389, 65.49, 64.971, 66.1, 64.889, 65.276]
INTJ adapter positional             median_ns=92.4 samples_ns=[91.825, 94.304, 95.082, 95.734, 92.446, 92.727, 92.416, 91.725, 91.926]
INTJ adapter kwargs                 median_ns=110.1 samples_ns=[111.984, 111.167, 111.082, 110.061, 110.503, 108.24, 108.059, 109.143, 108.447]
INTJ adapter defaults               median_ns=88.5 samples_ns=[88.063, 88.221, 88.537, 88.756, 94.025, 89.887, 88.772, 88.362, 88.347]
INTJ FFI wrapper positional         median_ns=103.8 samples_ns=[104.032, 102.503, 102.521, 103.089, 103.348, 103.829, 129.279, 106.417, 106.677]
INTJ FFI wrapper kwargs             median_ns=127.2 samples_ns=[127.157, 126.721, 126.387, 126.081, 125.931, 133.999, 127.816, 128.63, 127.383]
INTJ native kwargs: unsupported (verified TypeError)
mode=dataclass
INTJ pair direct                    median_ns=66.4 samples_ns=[68.424, 67.248, 69.745, 66.661, 65.897, 66.417, 66.35, 66.147, 65.968]
INTJ pair prebuilt *tuple           median_ns=139.4 samples_ns=[138.81, 141.886, 137.696, 143.769, 140.224, 141.122, 138.913, 139.401, 137.242]
FFI unpack Pair only                median_ns=165.9 samples_ns=[166.584, 172.106, 165.083, 166.567, 164.793, 165.918, 163.91, 169.58, 164.169]
INTJ pair manual unpack             median_ns=175.5 samples_ns=[175.463, 177.816, 176.67, 180.673, 175.45, 175.629, 173.532, 175.548, 172.38]
INTJ pair FFI unpack                median_ns=319.3 samples_ns=[327.34, 324.595, 320.717, 324.817, 319.274, 317.174, 319.293, 318.095, 313.243]
INTJ pair stdlib astuple            median_ns=1182.7 samples_ns=[1337.443, 1229.448, 1185.026, 1176.955, 1157.349, 1194.658, 1155.914, 1182.722, 1157.146]
INTJ config direct                  median_ns=80.5 samples_ns=[80.425, 81.363, 80.688, 83.945, 80.465, 81.836, 79.607, 80.255, 79.504]
INTJ config FFI unpack              median_ns=322.1 samples_ns=[320.036, 325.771, 319.888, 322.102, 323.953, 322.32, 315.985, 324.935, 317.421]
INTJ native dataclass unpack: unsupported (verified TypeError)
elapsed_ns=2953403253
exit_status=0
ended=2026-09-26T22:42:56+08:00
```

round 1:
```text
commit=e3c94b54ffe8159ea62fb7f94a22a7fc6b331c1f
started=2026-09-26T22:43:53+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_intj_ffi_paths.py --mode all --iters 1000 --batches 9
iters=1000 batches=9 unit=ns/call; no_gpu=True
mode=callback
callback_count=9100 (100 warmup + 9000 unique misses)
INTJ cold compile callback + cache  median_ns=414.6 samples_ns=[428.419, 465.791, 381.083, 1208.622, 380.108, 390.194, 412.525, 414.632, 2108.216]
INTJ hot call (no callback)         median_ns=40.2 samples_ns=[41.841, 41.239, 38.803, 40.448, 38.947, 40.183, 38.926, 43.91, 39.717]
INTJ 3-tensor host-only nop         median_ns=34.4 samples_ns=[36.083, 34.726, 33.986, 34.457, 37.535, 34.442, 33.923, 34.264, 33.842]
mode=kwargs
INTJ direct positional              median_ns=66.8 samples_ns=[82.937, 67.432, 66.485, 67.019, 66.677, 66.814, 66.943, 66.435, 66.047]
INTJ adapter positional             median_ns=94.3 samples_ns=[91.029, 91.095, 92.276, 92.373, 96.147, 94.256, 94.635, 94.851, 94.392]
INTJ adapter kwargs                 median_ns=110.4 samples_ns=[111.109, 110.084, 110.092, 110.386, 123.007, 111.404, 110.628, 109.721, 109.642]
INTJ adapter defaults               median_ns=90.5 samples_ns=[89.896, 90.88, 91.247, 90.554, 90.496, 98.769, 88.55, 88.106, 88.133]
INTJ FFI wrapper positional         median_ns=105.8 samples_ns=[105.069, 105.821, 105.767, 104.185, 105.789, 104.743, 112.843, 107.411, 105.426]
INTJ FFI wrapper kwargs             median_ns=127.0 samples_ns=[124.645, 126.54, 126.968, 127.336, 127.337, 130.839, 127.584, 125.669, 126.64]
INTJ native kwargs: unsupported (verified TypeError)
mode=dataclass
INTJ pair direct                    median_ns=69.5 samples_ns=[70.587, 68.838, 72.195, 69.838, 69.169, 69.878, 69.208, 69.494, 69.396]
INTJ pair prebuilt *tuple           median_ns=139.9 samples_ns=[139.872, 140.617, 138.013, 144.724, 138.747, 142.547, 137.433, 141.753, 139.708]
FFI unpack Pair only                median_ns=168.0 samples_ns=[168.198, 171.818, 167.515, 167.781, 163.798, 169.628, 168.034, 170.138, 162.243]
INTJ pair manual unpack             median_ns=173.7 samples_ns=[174.063, 176.886, 173.983, 182.438, 170.911, 173.749, 171.443, 173.387, 172.181]
INTJ pair FFI unpack                median_ns=316.8 samples_ns=[320.573, 320.193, 317.356, 321.804, 312.418, 315.132, 316.145, 316.844, 313.465]
INTJ pair stdlib astuple            median_ns=1153.4 samples_ns=[1339.579, 1202.037, 1145.693, 1160.264, 1143.008, 1153.371, 1140.355, 1158.736, 1149.37]
INTJ config direct                  median_ns=83.5 samples_ns=[82.537, 83.478, 82.633, 83.411, 83.507, 86.487, 84.709, 85.274, 84.72]
INTJ config FFI unpack              median_ns=321.0 samples_ns=[321.756, 322.767, 324.263, 321.72, 317.351, 320.989, 316.721, 315.718, 317.456]
INTJ native dataclass unpack: unsupported (verified TypeError)
elapsed_ns=2955250384
exit_status=0
ended=2026-09-26T22:43:56+08:00
```

round 2:
```text
commit=e3c94b54ffe8159ea62fb7f94a22a7fc6b331c1f
started=2026-09-26T22:44:52+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_intj_ffi_paths.py --mode all --iters 1000 --batches 9
iters=1000 batches=9 unit=ns/call; no_gpu=True
mode=callback
callback_count=9100 (100 warmup + 9000 unique misses)
INTJ cold compile callback + cache  median_ns=414.5 samples_ns=[429.477, 461.073, 377.504, 1233.022, 373.821, 389.111, 405.747, 414.464, 2103.435]
INTJ hot call (no callback)         median_ns=38.1 samples_ns=[39.868, 40.834, 37.327, 38.064, 38.117, 41.427, 37.637, 39.123, 37.026]
INTJ 3-tensor host-only nop         median_ns=34.3 samples_ns=[36.519, 36.552, 37.654, 35.421, 33.827, 34.118, 33.912, 34.26, 33.664]
mode=kwargs
INTJ direct positional              median_ns=65.4 samples_ns=[66.808, 65.432, 65.22, 65.538, 65.4, 65.578, 65.55, 65.306, 65.331]
INTJ adapter positional             median_ns=91.6 samples_ns=[92.556, 90.877, 91.103, 94.659, 91.85, 95.07, 91.569, 89.873, 89.01]
INTJ adapter kwargs                 median_ns=112.3 samples_ns=[110.217, 108.843, 109.879, 109.847, 119.624, 113.156, 113.215, 112.679, 112.331]
INTJ adapter defaults               median_ns=89.7 samples_ns=[89.724, 89.974, 90.566, 88.769, 89.175, 103.947, 89.732, 88.268, 88.191]
INTJ FFI wrapper positional         median_ns=103.8 samples_ns=[104.049, 103.863, 103.803, 101.433, 101.346, 102.421, 106.346, 103.532, 103.919]
INTJ FFI wrapper kwargs             median_ns=127.1 samples_ns=[127.399, 130.113, 125.887, 126.236, 125.865, 130.849, 127.407, 126.785, 127.051]
INTJ native kwargs: unsupported (verified TypeError)
mode=dataclass
INTJ pair direct                    median_ns=70.3 samples_ns=[72.627, 71.828, 78.268, 70.271, 69.728, 70.216, 70.061, 70.651, 69.942]
INTJ pair prebuilt *tuple           median_ns=141.1 samples_ns=[139.761, 141.525, 139.584, 144.237, 141.551, 143.481, 137.28, 141.051, 137.401]
FFI unpack Pair only                median_ns=167.5 samples_ns=[163.869, 171.428, 167.548, 168.44, 165.719, 166.458, 169.607, 168.339, 163.048]
INTJ pair manual unpack             median_ns=173.1 samples_ns=[171.781, 174.405, 172.68, 177.683, 172.986, 173.783, 173.072, 174.356, 171.796]
INTJ pair FFI unpack                median_ns=318.3 samples_ns=[318.318, 318.969, 315.087, 320.422, 316.282, 321.458, 322.386, 317.908, 317.004]
INTJ pair stdlib astuple            median_ns=1160.5 samples_ns=[1340.428, 1202.96, 1154.168, 1174.283, 1146.308, 1156.521, 1147.644, 1180.889, 1160.547]
INTJ config direct                  median_ns=84.9 samples_ns=[85.169, 86.006, 84.948, 85.605, 88.364, 84.59, 83.789, 84.438, 83.858]
INTJ config FFI unpack              median_ns=319.1 samples_ns=[316.383, 323.139, 317.473, 318.611, 320.164, 320.534, 319.131, 321.687, 318.49]
INTJ native dataclass unpack: unsupported (verified TypeError)
elapsed_ns=2966526478
exit_status=0
ended=2026-09-26T22:44:55+08:00
```

#### hip_module

round 0:
```text
commit=e3c94b54ffe8159ea62fb7f94a22a7fc6b331c1f
started=2026-09-26T22:42:56+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_hip_module_launch.py --iters 1000 --batches 9
hsaco_sha256=df3e529e178196ad62f7c07ab8593ede5b8596c76e7c6fbd12a5d947a696e65c kernel=empty_kernel threads=256 shared=0 callback_count=1
iters=1000 batches=9 unit=us/call (host enqueue, no timed sync)
TVM FFI HIP HSACO         median=3.3094 samples=[3.606742, 3.3080749999999997, 3.3046550000000003, 3.313701, 3.309418, 4.0582530000000006, 15.222759, 3.201873, 3.1948380000000003]
Triton same HSACO         median=16.1019 samples=[16.322516, 16.199688000000002, 16.117751000000002, 16.10191, 15.974283, 15.855665, 21.727328, 15.873085, 15.851656]
INTJ same function        median=3.0200 samples=[3.1078330000000003, 3.097131, 3.04194, 3.0380770000000004, 3.019965, 2.8778159999999997, 2.95464, 2.943572, 2.9145839999999996]
elapsed_ns=5970943497
exit_status=0
ended=2026-09-26T22:43:02+08:00
```

round 1:
```text
commit=e3c94b54ffe8159ea62fb7f94a22a7fc6b331c1f
started=2026-09-26T22:43:56+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_hip_module_launch.py --iters 1000 --batches 9
hsaco_sha256=df3e529e178196ad62f7c07ab8593ede5b8596c76e7c6fbd12a5d947a696e65c kernel=empty_kernel threads=256 shared=0 callback_count=1
iters=1000 batches=9 unit=us/call (host enqueue, no timed sync)
TVM FFI HIP HSACO         median=3.6630 samples=[3.478799, 3.582357, 3.758834, 3.6629699999999996, 3.701197, 3.634885, 3.693928, 3.6069180000000003, 3.6702440000000003]
Triton same HSACO         median=16.4123 samples=[16.490830000000003, 16.420232, 16.408109, 16.47543, 16.360279000000002, 38.002775, 16.324718999999998, 16.349169, 16.41231]
INTJ same function        median=3.4933 samples=[3.38616, 3.4847089999999996, 3.562013, 3.5223760000000004, 3.521118, 3.597945, 3.4666460000000003, 3.469413, 3.4932559999999997]
elapsed_ns=3843101856
exit_status=0
ended=2026-09-26T22:44:00+08:00
```

round 2:
```text
commit=e3c94b54ffe8159ea62fb7f94a22a7fc6b331c1f
started=2026-09-26T22:44:55+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_hip_module_launch.py --iters 1000 --batches 9
hsaco_sha256=df3e529e178196ad62f7c07ab8593ede5b8596c76e7c6fbd12a5d947a696e65c kernel=empty_kernel threads=256 shared=0 callback_count=1
iters=1000 batches=9 unit=us/call (host enqueue, no timed sync)
TVM FFI HIP HSACO         median=3.1320 samples=[3.329356, 3.182958, 3.139502, 3.133899, 3.132047, 3.072425, 3.063241, 3.034018, 3.042889]
Triton same HSACO         median=16.0146 samples=[16.121107, 15.974943, 16.038216000000002, 16.03738, 15.907077, 15.874167, 16.107229, 16.014618, 15.830781]
INTJ same function        median=2.9188 samples=[2.976526, 2.918784, 2.930979, 2.930724, 2.975622, 2.9062379999999997, 2.823992, 2.9129140000000002, 2.871773]
elapsed_ns=3763186801
exit_status=0
ended=2026-09-26T22:44:59+08:00
```

#### kernel_cache

round 0:
```text
commit=e3c94b54ffe8159ea62fb7f94a22a7fc6b331c1f
started=2026-09-26T22:43:02+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python -m pytest tests/test_kernel_cache.py -s -q

kernel cache, 1-word key, ns per operation

                      intj       tsl      absl
hash_only             1.36      1.35      1.36
hit/1                 1.76      2.02      1.52
hit/8                 1.83      2.13      6.26
hit/64                1.87      2.17      4.95
hit/512               2.54      3.03      5.82
hit_child             2.47      3.53      2.45
miss/1                1.39      1.61      1.38
miss/8                2.43      2.12      3.66
miss/64               2.74      2.10      3.96
miss/512              5.12      2.41      4.40

kernel cache, 2-word key, ns per operation

                      intj       tsl      absl
hash_only             1.53      1.44      1.42
hit/1                 2.18      3.27      2.36
hit/8                 2.18      2.57      5.39
hit/64                2.51      2.92      5.60
hit/512               3.32      5.00      6.83
hit_child             2.46      2.45      2.45
miss/1                1.42      2.32      1.70
miss/8                2.81      1.78      3.61
miss/64               3.36      2.31      5.05
miss/512              3.18      3.62      4.44

kernel cache, 5-word key, ns per operation

                      intj       tsl      absl
hash_only             3.57      3.60      3.58
hit/1                 3.13      3.18      2.83
hit/8                 3.36      3.53     10.01
hit/64                3.50      5.46     10.33
hit/512               4.75      4.92     11.65
hit_child             2.46      2.46      2.45
miss/1                2.06      2.11      1.73
miss/8                3.85      2.15      7.13
miss/64               4.60      1.77      6.91
miss/512              4.05      2.56      7.96
....
4 passed in 25.61s
elapsed_ns=26541035246
exit_status=0
ended=2026-09-26T22:43:28+08:00
```

round 1:
```text
commit=e3c94b54ffe8159ea62fb7f94a22a7fc6b331c1f
started=2026-09-26T22:44:00+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python -m pytest tests/test_kernel_cache.py -s -q

kernel cache, 1-word key, ns per operation

                      intj       tsl      absl
hash_only             1.36      1.80      1.36
hit/1                 1.77      2.43      1.49
hit/8                 1.85      2.14      6.39
hit/64                1.87      2.17      4.95
hit/512               2.56      3.03      5.79
hit_child             2.46      2.47      2.46
miss/1                1.39      1.77      1.38
miss/8                2.43      2.12      3.65
miss/64               2.74      2.11      3.93
miss/512              2.78      2.42      6.42

kernel cache, 2-word key, ns per operation

                      intj       tsl      absl
hash_only             1.42      1.43      1.42
hit/1                 2.19      2.57      1.95
hit/8                 2.18      2.82      6.07
hit/64                3.50      2.94      5.59
hit/512               3.33      4.32      6.82
hit_child             2.46      2.92      2.46
miss/1                1.42      2.32      1.71
miss/8                2.79      1.79      3.61
miss/64               3.34      2.75      4.89
miss/512              3.33      2.71      4.45

kernel cache, 5-word key, ns per operation

                      intj       tsl      absl
hash_only             3.57      3.93      3.58
hit/1                 3.14      3.18      2.83
hit/8                 3.35      3.53     10.01
hit/64                3.45      3.62     10.31
hit/512               6.80      4.95     11.66
hit_child             2.46      2.46      2.46
miss/1                2.01      2.26      1.71
miss/8                3.76      2.16      7.12
miss/64               3.73      1.96      6.82
miss/512              4.21      2.57      7.84
....
4 passed in 25.42s
elapsed_ns=26294698129
exit_status=0
ended=2026-09-26T22:44:27+08:00
```

round 2:
```text
commit=e3c94b54ffe8159ea62fb7f94a22a7fc6b331c1f
started=2026-09-26T22:44:59+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python -m pytest tests/test_kernel_cache.py -s -q

kernel cache, 1-word key, ns per operation

                      intj       tsl      absl
hash_only             1.37      1.35      1.36
hit/1                 1.76      2.04      1.54
hit/8                 1.84      2.13      4.76
hit/64                1.87      2.17      6.65
hit/512               3.93      3.03      5.79
hit_child             2.46      2.74      2.45
miss/1                1.39      1.45      1.38
miss/8                2.47      2.12      3.65
miss/64               2.77      2.10      3.94
miss/512              2.81      2.41      6.12

kernel cache, 2-word key, ns per operation

                      intj       tsl      absl
hash_only             1.44      1.42      1.42
hit/1                 2.18      2.57      1.96
hit/8                 2.18      2.56      5.40
hit/64                2.52      2.92      8.67
hit/512               3.34      4.04      6.80
hit_child             2.49      2.45      2.46
miss/1                1.39      2.35      1.72
miss/8                2.98      1.79      3.61
miss/64               3.47      2.34      4.16
miss/512              3.80      2.70      4.44

kernel cache, 5-word key, ns per operation

                      intj       tsl      absl
hash_only             3.57      3.58      3.57
hit/1                 3.15      3.18      2.83
hit/8                 3.36      3.53     10.01
hit/64                3.53      3.62     10.31
hit/512               4.76      4.92     11.65
hit_child             2.46      2.46      2.46
miss/1                2.00      2.12      2.06
miss/8                3.87      2.16      7.16
miss/64               5.20      1.77      6.93
miss/512              4.15      2.57      7.92
....
4 passed in 25.40s
elapsed_ns=26302358232
exit_status=0
ended=2026-09-26T22:45:25+08:00
```

### task1-rerun

#### launch_gpu

round 0:
```text
commit=e3c94b54ffe8159ea62fb7f94a22a7fc6b331c1f
started=2026-09-26T22:45:32+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --iters 1000 --batches 9
mode=gpu; 1000 calls × 9 batches; median ns/call
               path    ns/call       Δ ns       Δ %   build ms   bind ms
           auto map     3128.9       +0.0      +0.0      74.26         -
        reduced key     3111.5      -17.4      -0.6       3.81         -
         verify off     2942.1     -186.8      -6.0       2.04         -
          verify on     2905.8     -223.1      -7.1       1.94         -
              baked     2999.8     -129.0      -4.1      72.37         -
       bound tensor     2957.9     -171.0      -5.5       0.37      1.89
      bound pointer     3011.3     -117.6      -3.8       0.35      1.91
   fixed device map     2891.6     -237.3      -7.6       0.33      1.90
fixed device no-map     2911.5     -217.4      -6.9       0.28      1.86
elapsed_ns=3900506944
exit_status=0
ended=2026-09-26T22:45:36+08:00
```

round 1:
```text
commit=e3c94b54ffe8159ea62fb7f94a22a7fc6b331c1f
started=2026-09-26T22:46:29+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --iters 1000 --batches 9
mode=gpu; 1000 calls × 9 batches; median ns/call
               path    ns/call       Δ ns       Δ %   build ms   bind ms
           auto map     3069.3       +0.0      +0.0      74.68         -
        reduced key     3088.9      +19.6      +0.6       3.55         -
         verify off     2975.8      -93.5      -3.0       1.98         -
          verify on     2886.8     -182.5      -5.9       1.93         -
              baked     3005.0      -64.3      -2.1      72.05         -
       bound tensor     3022.5      -46.8      -1.5       0.35      1.75
      bound pointer     3012.2      -57.1      -1.9       0.34      1.86
   fixed device map     2911.3     -158.0      -5.1       0.30      1.83
fixed device no-map     2899.6     -169.7      -5.5       0.36      1.85
elapsed_ns=3624197584
exit_status=0
ended=2026-09-26T22:46:33+08:00
```

round 2:
```text
commit=e3c94b54ffe8159ea62fb7f94a22a7fc6b331c1f
started=2026-09-26T22:47:25+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --iters 1000 --batches 9
mode=gpu; 1000 calls × 9 batches; median ns/call
               path    ns/call       Δ ns       Δ %   build ms   bind ms
           auto map     3088.7       +0.0      +0.0      74.44         -
        reduced key     3103.4      +14.7      +0.5       3.70         -
         verify off     2977.5     -111.2      -3.6       2.01         -
          verify on     2931.4     -157.4      -5.1       1.95         -
              baked     2943.8     -144.9      -4.7      72.31         -
       bound tensor     3023.0      -65.7      -2.1       0.36      1.77
      bound pointer     3014.8      -73.9      -2.4       0.36      1.86
   fixed device map     2949.7     -139.0      -4.5       0.31      1.83
fixed device no-map     2893.0     -195.7      -6.3       0.37      1.82
elapsed_ns=3615905265
exit_status=0
ended=2026-09-26T22:47:29+08:00
```

#### launch_host

round 0:
```text
commit=e3c94b54ffe8159ea62fb7f94a22a7fc6b331c1f
started=2026-09-26T22:45:36+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --no-gpu --iters 100000 --batches 9
mode=host-only; 100000 calls × 9 batches; median ns/call
               path    ns/call       Δ ns       Δ %   build ms   bind ms
           auto map       42.7       +0.0      +0.0      61.53         -
        reduced key       42.0       -0.7      -1.6       1.87         -
         verify off       42.5       -0.2      -0.4       1.74         -
          verify on       43.4       +0.7      +1.7       1.66         -
              baked       39.2       -3.6      -8.3       2.96         -
       bound tensor       45.5       +2.8      +6.6       0.15      1.58
      bound pointer       45.0       +2.3      +5.4       0.13      1.59
   fixed device map       43.6       +0.8      +2.0       0.11      1.54
fixed device no-map       44.3       +1.6      +3.7       0.14      1.54
elapsed_ns=3060554553
exit_status=0
ended=2026-09-26T22:45:39+08:00
```

round 1:
```text
commit=e3c94b54ffe8159ea62fb7f94a22a7fc6b331c1f
started=2026-09-26T22:46:33+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --no-gpu --iters 100000 --batches 9
mode=host-only; 100000 calls × 9 batches; median ns/call
               path    ns/call       Δ ns       Δ %   build ms   bind ms
           auto map       41.8       +0.0      +0.0      61.33         -
        reduced key       41.7       -0.2      -0.4       1.86         -
         verify off       42.3       +0.5      +1.2       1.69         -
          verify on       43.8       +2.0      +4.7       1.69         -
              baked       40.4       -1.5      -3.6       2.83         -
       bound tensor       46.1       +4.3     +10.3       0.15      1.60
      bound pointer       44.7       +2.9      +6.9       0.14      1.59
   fixed device map       42.4       +0.6      +1.4       0.11      1.55
fixed device no-map       43.7       +1.8      +4.4       0.15      1.55
elapsed_ns=2869662947
exit_status=0
ended=2026-09-26T22:46:36+08:00
```

round 2:
```text
commit=e3c94b54ffe8159ea62fb7f94a22a7fc6b331c1f
started=2026-09-26T22:47:29+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --no-gpu --iters 100000 --batches 9
mode=host-only; 100000 calls × 9 batches; median ns/call
               path    ns/call       Δ ns       Δ %   build ms   bind ms
           auto map       41.7       +0.0      +0.0      60.70         -
        reduced key       41.8       +0.0      +0.0       1.85         -
         verify off       42.4       +0.6      +1.5       1.72         -
          verify on       43.7       +1.9      +4.6       1.66         -
              baked       41.8       +0.0      +0.0       2.89         -
       bound tensor       46.2       +4.4     +10.6       0.15      1.59
      bound pointer       61.7      +19.9     +47.7       0.14      1.58
   fixed device map       42.8       +1.1      +2.6       0.11      1.54
fixed device no-map       44.5       +2.8      +6.7       0.15      1.55
elapsed_ns=2877279085
exit_status=0
ended=2026-09-26T22:47:32+08:00
```

#### launch_readme

round 0:
```text
commit=e3c94b54ffe8159ea62fb7f94a22a7fc6b331c1f
started=2026-09-26T22:45:39+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --readme --iters 1000 --batches 9
mode=readme; 1000 calls × 9 batches; median
        path   triton us    intj us   speedup
   grid=(1,)       16.44       3.10      5.3x
   grid=(0,)       12.95       0.03    456.3x

torch_access_mode   decode ns    build s
     runtime_shim        92.7       0.02
   static_compile        92.9       0.10
      interpreter       887.5       0.00
elapsed_ns=3756469732
exit_status=0
ended=2026-09-26T22:45:42+08:00
```

round 1:
```text
commit=e3c94b54ffe8159ea62fb7f94a22a7fc6b331c1f
started=2026-09-26T22:46:36+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --readme --iters 1000 --batches 9
mode=readme; 1000 calls × 9 batches; median
        path   triton us    intj us   speedup
   grid=(1,)       16.25       3.14      5.2x
   grid=(0,)       12.89       0.03    456.7x

torch_access_mode   decode ns    build s
     runtime_shim        91.9       0.02
   static_compile        90.7       0.06
      interpreter       869.0       0.00
elapsed_ns=3645919302
exit_status=0
ended=2026-09-26T22:46:39+08:00
```

round 2:
```text
commit=e3c94b54ffe8159ea62fb7f94a22a7fc6b331c1f
started=2026-09-26T22:47:32+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --readme --iters 1000 --batches 9
mode=readme; 1000 calls × 9 batches; median
        path   triton us    intj us   speedup
   grid=(1,)       16.53       3.13      5.3x
   grid=(0,)       13.14       0.03    463.1x

torch_access_mode   decode ns    build s
     runtime_shim        93.5       0.02
   static_compile        92.0       0.06
      interpreter       874.1       0.00
elapsed_ns=3693906902
exit_status=0
ended=2026-09-26T22:47:35+08:00
```

#### launch_sweep

round 0:
```text
commit=e3c94b54ffe8159ea62fb7f94a22a7fc6b331c1f
started=2026-09-26T22:45:42+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --sweep --iters 100000 --batches 9
mode=sweep; host-only; 100000 calls × 9 batches; median ns/call
count   kind    ns/call
    4    int       40.8
    4 tensor       39.5
   16    int       68.5
   16 tensor       65.6
   32    int      107.6
   32 tensor      110.8
elapsed_ns=3157987401
exit_status=0
ended=2026-09-26T22:45:46+08:00
```

round 1:
```text
commit=e3c94b54ffe8159ea62fb7f94a22a7fc6b331c1f
started=2026-09-26T22:46:39+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --sweep --iters 100000 --batches 9
mode=sweep; host-only; 100000 calls × 9 batches; median ns/call
count   kind    ns/call
    4    int       40.7
    4 tensor       39.4
   16    int       68.4
   16 tensor       64.8
   32    int      106.2
   32 tensor      113.3
elapsed_ns=2904933209
exit_status=0
ended=2026-09-26T22:46:42+08:00
```

round 2:
```text
commit=e3c94b54ffe8159ea62fb7f94a22a7fc6b331c1f
started=2026-09-26T22:47:35+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --sweep --iters 100000 --batches 9
mode=sweep; host-only; 100000 calls × 9 batches; median ns/call
count   kind    ns/call
    4    int       40.9
    4 tensor       39.4
   16    int       68.3
   16 tensor       64.7
   32    int      106.2
   32 tensor      110.1
elapsed_ns=2887524128
exit_status=0
ended=2026-09-26T22:47:38+08:00
```

#### launch_last_key

round 0:
```text
commit=e3c94b54ffe8159ea62fb7f94a22a7fc6b331c1f
started=2026-09-26T22:45:46+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --last-key --iters 100000 --batches 9
mode=last-key; host-only; 200000 calls × 9 batches; median ns/call
count     pattern    ns/call
    4      repeat       34.1
    4   alternate       36.4
   16      repeat       62.6
   16   alternate       65.4
   32      repeat      100.2
   32   alternate      104.0
elapsed_ns=3392677042
exit_status=0
ended=2026-09-26T22:45:49+08:00
```

round 1:
```text
commit=e3c94b54ffe8159ea62fb7f94a22a7fc6b331c1f
started=2026-09-26T22:46:42+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --last-key --iters 100000 --batches 9
mode=last-key; host-only; 200000 calls × 9 batches; median ns/call
count     pattern    ns/call
    4      repeat       34.0
    4   alternate       36.2
   16      repeat       62.7
   16   alternate       65.5
   32      repeat      100.0
   32   alternate      104.0
elapsed_ns=3242131880
exit_status=0
ended=2026-09-26T22:46:45+08:00
```

round 2:
```text
commit=e3c94b54ffe8159ea62fb7f94a22a7fc6b331c1f
started=2026-09-26T22:47:38+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --last-key --iters 100000 --batches 9
mode=last-key; host-only; 200000 calls × 9 batches; median ns/call
count     pattern    ns/call
    4      repeat       34.0
    4   alternate       36.2
   16      repeat       62.7
   16   alternate       65.3
   32      repeat       99.6
   32   alternate      104.2
elapsed_ns=3236442290
exit_status=0
ended=2026-09-26T22:47:41+08:00
```

#### ffi_compare

round 0:
```text
commit=e3c94b54ffe8159ea62fb7f94a22a7fc6b331c1f
started=2026-09-26T22:45:49+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_ffi_compare.py --iters 1000 --batches 9
iters=1000 batches=9 unit=us/call (host-only, no timed sync)
FFI packed nop             median=0.1601 samples=[0.167101, 0.160081, 0.160092, 0.15940000000000001, 0.15971000000000002, 0.15985, 0.16246100000000002, 0.159857, 0.166618]
FFI typed nop              median=0.1633 samples=[0.171785, 0.16327, 0.165797, 0.162697, 0.159458, 0.165056, 0.159581, 0.164531, 0.160135]
FFI empty kernel           median=3.3941 samples=[3.5985039999999997, 3.441591, 3.4145320000000003, 3.244304, 3.237007, 3.432833, 3.253061, 3.3941399999999997, 3.263121]
INTJ empty kernel          median=3.0046 samples=[3.073692, 3.120605, 3.1071280000000003, 2.9354989999999996, 2.938731, 3.02168, 3.004618, 3.0021039999999997, 2.9792829999999997]
INTJ fixed-device kernel   median=2.9283 samples=[3.040884, 3.121105, 3.106376, 2.953123, 2.8087530000000003, 2.844823, 2.827951, 2.844055, 2.928276]
FFI packed nop mixed       median=0.1922 samples=[0.193207, 0.191309, 0.189728, 0.18951300000000001, 0.19218000000000002, 0.196289, 0.192459, 0.190265, 0.193578]
FFI typed nop mixed        median=0.1952 samples=[0.194712, 0.197829, 0.190489, 0.19563, 0.194495, 0.19528800000000002, 0.197276, 0.195198, 0.194943]
FFI mixed kernel           median=3.3366 samples=[3.409148, 3.470302, 3.336643, 3.328873, 3.2499409999999997, 3.408991, 3.318521, 3.413119, 3.292154]
INTJ mixed kernel          median=2.9685 samples=[3.130772, 3.138123, 2.968494, 2.979118, 2.812451, 2.903203, 2.8693969999999998, 3.0183690000000003, 2.9380390000000003]
INTJ fixed mixed kernel    median=3.0293 samples=[3.193464, 3.1873829999999996, 3.05625, 3.0604299999999998, 2.925409, 2.915543, 2.9785169999999996, 3.029276, 3.005292]
elapsed_ns=3812884452
exit_status=0
ended=2026-09-26T22:45:53+08:00
```

round 1:
```text
commit=e3c94b54ffe8159ea62fb7f94a22a7fc6b331c1f
started=2026-09-26T22:46:45+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_ffi_compare.py --iters 1000 --batches 9
iters=1000 batches=9 unit=us/call (host-only, no timed sync)
FFI packed nop             median=0.1438 samples=[0.148993, 0.140256, 0.142574, 0.147865, 0.142482, 0.14607699999999998, 0.145458, 0.14384, 0.14361000000000002]
FFI typed nop              median=0.1489 samples=[0.15742599999999998, 0.151869, 0.145581, 0.148301, 0.148521, 0.15135099999999999, 0.147787, 0.149262, 0.14885400000000001]
FFI empty kernel           median=3.6450 samples=[3.579968, 4.440327, 3.645046, 3.745774, 3.5598769999999997, 4.3901639999999995, 3.52292, 4.322502999999999, 3.5368600000000003]
INTJ empty kernel          median=3.5365 samples=[3.170739, 3.511629, 4.082654, 3.5365010000000003, 3.828701, 3.514565, 3.915779, 3.496547, 3.8771419999999996]
INTJ fixed-device kernel   median=3.3318 samples=[3.370497, 3.289462, 3.6788960000000004, 3.433232, 3.32638, 3.13409, 3.334482, 3.13971, 3.331815]
FFI packed nop mixed       median=0.1742 samples=[0.173966, 0.174216, 0.17375200000000002, 0.17380299999999999, 0.17724700000000002, 0.174177, 0.17425200000000002, 0.17465199999999997, 0.173451]
FFI typed nop mixed        median=0.1775 samples=[0.17534200000000003, 0.17813800000000002, 0.175147, 0.183243, 0.175081, 0.181536, 0.177414, 0.17851, 0.177508]
FFI mixed kernel           median=3.8905 samples=[3.686429, 3.922904, 3.891196, 3.890498, 3.620308, 4.357403000000001, 3.658801, 4.361256999999999, 3.671401]
INTJ mixed kernel          median=3.5516 samples=[4.220738, 3.60479, 3.554757, 3.5585859999999996, 3.34279, 3.436648, 3.379482, 3.551576, 3.331719]
INTJ fixed mixed kernel    median=3.5191 samples=[3.6592510000000003, 3.612603, 3.662857, 3.618343, 3.472452, 3.4855430000000003, 3.484802, 3.510135, 3.5191149999999998]
elapsed_ns=3684926663
exit_status=0
ended=2026-09-26T22:46:49+08:00
```

round 2:
```text
commit=e3c94b54ffe8159ea62fb7f94a22a7fc6b331c1f
started=2026-09-26T22:47:41+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_ffi_compare.py --iters 1000 --batches 9
iters=1000 batches=9 unit=us/call (host-only, no timed sync)
FFI packed nop             median=0.1431 samples=[0.149439, 0.140755, 0.142434, 0.142741, 0.147177, 0.144398, 0.14222200000000002, 0.14306, 0.147787]
FFI typed nop              median=0.1484 samples=[0.148884, 0.14573599999999998, 0.14719900000000002, 0.147769, 0.147584, 0.148436, 0.15083600000000003, 0.149664, 0.148656]
FFI empty kernel           median=3.3601 samples=[3.426935, 3.516269, 3.54265, 3.3287809999999998, 3.171446, 3.391904, 3.2173760000000002, 3.360127, 3.224592]
INTJ empty kernel          median=2.9692 samples=[3.07967, 3.130726, 3.119084, 2.897798, 2.9230189999999996, 2.997815, 2.950645, 2.96698, 2.969154]
INTJ fixed-device kernel   median=2.8419 samples=[2.959112, 3.1284720000000004, 3.107072, 2.823698, 2.778828, 2.8439740000000002, 2.806784, 2.841917, 2.8077310000000004]
FFI packed nop mixed       median=0.1746 samples=[0.17168, 0.173059, 0.17722900000000003, 0.17571199999999998, 0.173269, 0.17450100000000002, 0.174627, 0.17499299999999998, 0.176009]
FFI typed nop mixed        median=0.1752 samples=[0.17324799999999999, 0.17466399999999999, 0.17430099999999998, 0.17677400000000001, 0.17361500000000002, 0.175445, 0.176965, 0.177209, 0.175213]
FFI mixed kernel           median=3.4094 samples=[3.475364, 3.581775, 3.4318310000000003, 3.358791, 3.2197150000000003, 3.409389, 3.298517, 3.415731, 3.2780970000000003]
INTJ mixed kernel          median=2.8795 samples=[3.14975, 3.143982, 2.856297, 3.041578, 2.776537, 2.8795129999999998, 2.8498989999999997, 2.941486, 2.837495]
INTJ fixed mixed kernel    median=2.9517 samples=[3.180542, 3.16975, 2.994654, 2.972765, 2.883395, 2.862772, 2.873471, 2.951662, 2.915676]
elapsed_ns=3657871553
exit_status=0
ended=2026-09-26T22:47:45+08:00
```

#### ffi_sweep

round 0:
```text
commit=e3c94b54ffe8159ea62fb7f94a22a7fc6b331c1f
started=2026-09-26T22:45:53+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_ffi_compare.py --sweep --iters 1000 --batches 9
counts=[0, 3, 5, 8, 16, 32, 64] iters=1000 batches=9 unit=us/call (host-only, no timed sync); INTJ device-bound, no specialization key
args= 0 FFI packed nop           median=0.1165 samples=[0.119057, 0.118965, 0.118073, 0.11551399999999999, 0.116547, 0.115324, 0.11612900000000001, 0.116074, 0.119007]
args= 0 FFI typed nop            median=0.1199 samples=[0.11991, 0.12066299999999999, 0.120992, 0.125047, 0.11903799999999999, 0.122, 0.117271, 0.117846, 0.118005]
args= 0 FFI empty kernel         median=1.6530 samples=[1.65299, 1.866676, 1.624723, 1.823717, 1.589494, 1.866082, 1.58225, 1.876217, 1.595258]
args= 0 INTJ static_compile kernel median=2.7049 samples=[3.13522, 2.670258, 2.732559, 2.694109, 2.7112979999999998, 2.680939, 2.713655, 2.6761060000000003, 2.7048769999999998]
args= 0 INTJ runtime_shim kernel median=2.7008 samples=[2.859001, 2.6800059999999997, 2.7223490000000004, 2.716036, 2.70077, 2.6572869999999997, 2.704165, 2.666369, 2.6917739999999997]
args= 3 FFI packed nop           median=0.1447 samples=[0.14579499999999998, 0.14386600000000002, 0.143901, 0.14527600000000002, 0.14465899999999998, 0.145583, 0.140596, 0.145048, 0.143427]
args= 3 FFI typed nop            median=0.1483 samples=[0.14953999999999998, 0.149363, 0.14679599999999998, 0.14759999999999998, 0.148335, 0.149271, 0.145328, 0.15168399999999999, 0.145665]
args= 3 FFI empty kernel         median=3.1931 samples=[3.324111, 3.193101, 3.1662649999999997, 3.2306559999999998, 3.135118, 3.2085700000000004, 3.130291, 3.207832, 3.127337]
args= 3 INTJ static_compile kernel median=2.9431 samples=[2.9754490000000002, 2.8583670000000003, 2.946662, 2.877278, 2.9430680000000002, 2.86216, 2.9576770000000003, 2.8815459999999997, 2.9451300000000002]
args= 3 INTJ runtime_shim kernel median=2.8932 samples=[2.9913119999999997, 2.870519, 2.89076, 3.130214, 3.326361, 2.902022, 2.873112, 2.8932260000000003, 2.883019]
args= 5 FFI packed nop           median=0.1755 samples=[0.176815, 0.1737, 0.175995, 0.175549, 0.174704, 0.178095, 0.17576499999999998, 0.173916, 0.17161500000000002]
args= 5 FFI typed nop            median=0.1789 samples=[0.174544, 0.17831899999999998, 0.178353, 0.19512100000000002, 0.17784899999999998, 0.180346, 0.178939, 0.179529, 0.179984]
args= 5 FFI empty kernel         median=3.2507 samples=[3.3409299999999997, 3.2357460000000002, 3.263172, 3.250675, 4.089952, 3.250481, 3.246728, 3.2268980000000003, 3.252221]
args= 5 INTJ static_compile kernel median=2.8759 samples=[2.9837350000000002, 2.8825540000000003, 2.8762600000000003, 2.875646, 10.607644, 2.860614, 2.875944, 2.8558339999999998, 2.872522]
args= 5 INTJ runtime_shim kernel median=2.8827 samples=[2.9975859999999996, 2.94327, 2.8619250000000003, 2.881406, 11.376601, 2.882695, 2.85828, 2.887989, 2.861557]
args= 8 FFI packed nop           median=0.2067 samples=[0.210748, 0.20484200000000002, 0.20536500000000002, 0.202663, 0.320167, 0.20718899999999998, 0.20672200000000002, 0.206311, 0.209531]
args= 8 FFI typed nop            median=0.2088 samples=[0.21588, 0.210494, 0.208777, 0.20671299999999998, 0.414772, 0.215243, 0.208677, 0.208703, 0.20791300000000001]
args= 8 FFI empty kernel         median=3.4199 samples=[3.509004, 3.413597, 3.342256, 3.419899, 3.734261, 3.40473, 3.511518, 3.3898029999999997, 3.5107939999999997]
args= 8 INTJ static_compile kernel median=3.0262 samples=[3.066129, 2.95715, 3.0261880000000003, 3.010399, 3.100787, 3.013115, 3.044978, 3.012887, 3.0283789999999997]
args= 8 INTJ runtime_shim kernel median=2.8828 samples=[3.0662469999999997, 2.926888, 2.864149, 2.874237, 2.956446, 2.884405, 2.8541849999999998, 2.882812, 2.877609]
args=16 FFI packed nop           median=0.2926 samples=[0.29023000000000004, 0.293427, 0.28833600000000004, 0.291131, 0.292619, 0.293597, 0.297502, 0.292125, 0.293291]
args=16 FFI typed nop            median=0.3046 samples=[0.304649, 0.30580799999999997, 0.300308, 0.30801999999999996, 0.31159699999999996, 0.305998, 0.301979, 0.30421800000000004, 0.29985500000000004]
args=16 FFI empty kernel         median=3.6267 samples=[3.726209, 3.703411, 3.5219009999999997, 3.626733, 3.5819929999999998, 3.662584, 3.525522, 3.671649, 3.512129]
args=16 INTJ static_compile kernel median=3.3651 samples=[3.303364, 3.409613, 3.4608820000000002, 3.301284, 3.369752, 3.341631, 3.460673, 3.312581, 3.36513]
args=16 INTJ runtime_shim kernel median=3.2643 samples=[3.2642710000000004, 3.120873, 3.28973, 3.03537, 3.435121, 3.029023, 3.303174, 3.028219, 3.370371]
args=32 FFI packed nop           median=0.4765 samples=[0.49328500000000003, 0.48563799999999996, 0.476526, 0.476428, 0.476213, 0.475317, 0.481529, 0.473158, 0.48087799999999997]
args=32 FFI typed nop            median=0.4951 samples=[0.48891500000000004, 0.504333, 0.485154, 0.495113, 0.49507100000000004, 0.48615600000000003, 0.496064, 0.49607999999999997, 0.489188]
args=32 FFI empty kernel         median=4.0946 samples=[4.1751130000000005, 4.1371649999999995, 3.954535, 4.0946370000000005, 4.013964, 4.1435, 4.0169369999999995, 4.108558, 3.966519]
args=32 INTJ static_compile kernel median=3.5934 samples=[3.432782, 3.59342, 3.618874, 3.5101419999999997, 3.653934, 3.5572939999999997, 3.622229, 3.522237, 3.5996010000000003]
args=32 INTJ runtime_shim kernel median=3.4579 samples=[3.4246019999999997, 3.445207, 3.5648429999999998, 3.4080340000000002, 3.651549, 3.4578960000000003, 3.579694, 3.407607, 3.553489]
args=64 FFI packed nop           median=0.8443 samples=[0.840491, 0.8603930000000001, 0.8550979999999999, 0.819778, 0.844271, 0.836569, 0.856245, 0.83161, 0.8451069999999999]
args=64 FFI typed nop            median=0.8629 samples=[0.869525, 0.88402, 0.874381, 0.847228, 0.843456, 0.8646900000000001, 0.8616699999999999, 0.862947, 0.845048]
args=64 FFI empty kernel         median=5.0512 samples=[5.042172, 5.051344, 5.104328000000001, 5.0512250000000005, 5.172761, 5.082593, 5.050864, 5.030391, 5.026296]
args=64 INTJ static_compile kernel median=4.5469 samples=[4.61174, 4.521269, 4.554398, 4.537725999999999, 4.615601, 4.58859, 4.546929, 4.538982, 4.525135000000001]
args=64 INTJ runtime_shim kernel median=4.5236 samples=[4.556721, 4.585374, 4.499876, 4.571642, 4.523553, 4.5424560000000005, 4.499626, 4.523586, 4.469259]
elapsed_ns=4467147716
exit_status=0
ended=2026-09-26T22:45:57+08:00
```

round 1:
```text
commit=e3c94b54ffe8159ea62fb7f94a22a7fc6b331c1f
started=2026-09-26T22:46:49+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_ffi_compare.py --sweep --iters 1000 --batches 9
counts=[0, 3, 5, 8, 16, 32, 64] iters=1000 batches=9 unit=us/call (host-only, no timed sync); INTJ device-bound, no specialization key
args= 0 FFI packed nop           median=0.1175 samples=[0.119078, 0.11804200000000001, 0.115628, 0.117506, 0.11659399999999999, 0.116335, 0.117851, 0.11809900000000001, 0.114577]
args= 0 FFI typed nop            median=0.1207 samples=[0.129312, 0.122771, 0.119229, 0.12126300000000001, 0.12066800000000001, 0.11934499999999999, 0.117132, 0.12276999999999999, 0.117625]
args= 0 FFI empty kernel         median=1.6289 samples=[1.582195, 1.966906, 1.62886, 1.824883, 1.5836649999999999, 1.880865, 1.56919, 1.898339, 1.593216]
args= 0 INTJ static_compile kernel median=2.7350 samples=[3.163845, 2.763556, 2.7480729999999998, 2.743681, 2.7349789999999996, 2.712199, 2.698953, 2.685954, 2.715496]
args= 0 INTJ runtime_shim kernel median=2.7001 samples=[2.908406, 2.7101010000000003, 2.765024, 2.679148, 2.734133, 2.688421, 2.6941930000000003, 2.695373, 2.700139]
args= 3 FFI packed nop           median=0.1439 samples=[0.14601, 0.143344, 0.144936, 0.143944, 0.143664, 0.143701, 0.141493, 0.14475100000000002, 0.144451]
args= 3 FFI typed nop            median=0.1485 samples=[0.149272, 0.147273, 0.148138, 0.149317, 0.148604, 0.148861, 0.14846, 0.14827, 0.147137]
args= 3 FFI empty kernel         median=3.2148 samples=[3.337198, 3.241385, 3.179191, 3.2195120000000004, 3.207262, 3.214827, 3.149861, 3.220451, 3.1677]
args= 3 INTJ static_compile kernel median=2.9358 samples=[2.990784, 2.935796, 2.976546, 2.901995, 2.910024, 2.861482, 2.967375, 2.8619760000000003, 2.9384409999999996]
args= 3 INTJ runtime_shim kernel median=2.8577 samples=[2.990957, 2.805873, 2.793528, 2.882616, 2.880814, 2.847705, 2.790301, 2.857704, 2.880569]
args= 5 FFI packed nop           median=0.1738 samples=[0.189607, 0.173845, 0.176679, 0.17478, 0.17362200000000003, 0.174108, 0.172547, 0.173407, 0.173607]
args= 5 FFI typed nop            median=0.1776 samples=[0.17573, 0.18055000000000002, 0.17852500000000002, 0.195304, 0.179644, 0.175451, 0.17317, 0.177558, 0.175286]
args= 5 FFI empty kernel         median=3.2666 samples=[3.3789450000000003, 3.339097, 3.204199, 3.2666370000000002, 3.254515, 3.300322, 3.1987229999999998, 3.308439, 3.254462]
args= 5 INTJ static_compile kernel median=2.8963 samples=[3.017786, 2.833974, 2.882012, 2.866943, 2.8963490000000003, 2.8980259999999998, 2.877059, 2.903393, 2.897049]
args= 5 INTJ runtime_shim kernel median=2.8625 samples=[3.0459899999999998, 2.900224, 2.882183, 2.8846529999999997, 2.862452, 2.7726669999999998, 2.839746, 2.7788209999999998, 2.853902]
args= 8 FFI packed nop           median=0.2064 samples=[0.207816, 0.209596, 0.211257, 0.20636600000000002, 0.20716800000000002, 0.20500100000000002, 0.20588800000000002, 0.204882, 0.204686]
args= 8 FFI typed nop            median=0.2095 samples=[0.208982, 0.21278899999999998, 0.20851599999999998, 0.21619, 0.20693, 0.210936, 0.211391, 0.209519, 0.207194]
args= 8 FFI empty kernel         median=3.3870 samples=[3.536397, 3.469995, 3.303969, 3.380988, 3.301866, 3.4000019999999997, 3.275532, 3.387015, 3.537776]
args= 8 INTJ static_compile kernel median=3.0142 samples=[3.077495, 3.021743, 2.976563, 3.010719, 2.917316, 3.010702, 3.046399, 3.014189, 3.0592759999999997]
args= 8 INTJ runtime_shim kernel median=2.9494 samples=[3.0863560000000003, 2.987301, 2.964989, 2.894892, 2.949449, 2.886446, 2.965111, 2.8910790000000004, 2.8863980000000002]
args=16 FFI packed nop           median=0.2956 samples=[0.29562900000000003, 0.297427, 0.295999, 0.295904, 0.295989, 0.288056, 0.291736, 0.289031, 0.28639600000000004]
args=16 FFI typed nop            median=0.3034 samples=[0.30323700000000003, 0.308179, 0.303435, 0.306712, 0.304944, 0.305159, 0.302801, 0.29809199999999997, 0.301106]
args=16 FFI empty kernel         median=3.6613 samples=[3.764181, 3.749395, 3.548263, 3.661288, 3.5583449999999996, 3.6825390000000002, 3.542257, 3.680256, 3.546315]
args=16 INTJ static_compile kernel median=3.3615 samples=[3.2796089999999998, 3.430454, 3.538754, 3.342264, 3.535324, 3.361469, 3.485569, 3.339179, 3.353098]
args=16 INTJ runtime_shim kernel median=3.2433 samples=[3.2827640000000002, 3.204874, 3.374732, 3.041886, 3.3576289999999998, 3.03035, 3.3102910000000003, 3.0296030000000003, 3.243252]
args=32 FFI packed nop           median=0.4808 samples=[0.478076, 0.477255, 0.488554, 0.485608, 0.485343, 0.478531, 0.481255, 0.47224400000000005, 0.480777]
args=32 FFI typed nop            median=0.4953 samples=[0.493199, 0.500902, 0.497396, 0.49893, 0.491469, 0.494515, 0.498058, 0.495256, 0.493022]
args=32 FFI empty kernel         median=4.2427 samples=[4.289033000000001, 4.3193850000000005, 4.1222520000000005, 4.257046, 4.125665, 4.261845999999999, 4.1075230000000005, 4.2426710000000005, 4.148671]
args=32 INTJ static_compile kernel median=3.6464 samples=[3.4475, 3.682959, 3.648107, 3.573057, 3.64642, 3.549414, 3.666191, 3.601403, 3.757886]
args=32 INTJ runtime_shim kernel median=3.5351 samples=[3.446963, 3.53507, 3.641587, 3.438973, 3.6292489999999997, 3.423525, 3.627182, 3.437078, 3.658185]
args=64 FFI packed nop           median=0.8483 samples=[0.8582949999999999, 0.8743650000000001, 0.884574, 0.878738, 0.835375, 0.826589, 0.827795, 0.832375, 0.848264]
args=64 FFI typed nop            median=0.8811 samples=[0.881116, 0.8989539999999999, 0.89543, 0.903304, 0.866894, 0.8534299999999999, 0.841168, 0.856563, 0.882947]
args=64 FFI empty kernel         median=5.0489 samples=[5.098639, 5.160888, 5.077992, 5.076149999999999, 5.048879, 5.048247999999999, 5.030935, 5.036334, 5.034039]
args=64 INTJ static_compile kernel median=4.5888 samples=[4.794599, 4.624383, 4.58146, 4.593689, 4.564987, 4.571343, 4.586531, 4.592518, 4.588761000000001]
args=64 INTJ runtime_shim kernel median=4.5573 samples=[4.703622, 4.611502000000001, 4.4850330000000005, 4.565525, 4.540337999999999, 4.557344, 4.512263, 4.569451, 4.531482]
elapsed_ns=4223161403
exit_status=0
ended=2026-09-26T22:46:53+08:00
```

round 2:
```text
commit=e3c94b54ffe8159ea62fb7f94a22a7fc6b331c1f
started=2026-09-26T22:47:45+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_ffi_compare.py --sweep --iters 1000 --batches 9
counts=[0, 3, 5, 8, 16, 32, 64] iters=1000 batches=9 unit=us/call (host-only, no timed sync); INTJ device-bound, no specialization key
args= 0 FFI packed nop           median=0.1165 samples=[0.118718, 0.11919, 0.116483, 0.116343, 0.117498, 0.115451, 0.11349, 0.115753, 0.117395]
args= 0 FFI typed nop            median=0.1202 samples=[0.12095, 0.12273999999999999, 0.120109, 0.121509, 0.120869, 0.120225, 0.116245, 0.11706100000000001, 0.11832599999999999]
args= 0 FFI empty kernel         median=1.6358 samples=[1.6358320000000002, 1.781455, 1.578864, 1.782042, 1.5835409999999999, 1.756145, 1.564085, 1.800318, 1.5642719999999999]
args= 0 INTJ static_compile kernel median=2.7168 samples=[3.051678, 2.716759, 2.71923, 2.7287060000000003, 2.721005, 2.7057379999999998, 2.666166, 2.706789, 2.7001709999999997]
args= 0 INTJ runtime_shim kernel median=2.7089 samples=[2.779032, 2.668114, 2.737406, 2.6667739999999998, 2.715689, 2.645366, 2.742538, 2.651114, 2.708902]
args= 3 FFI packed nop           median=0.1448 samples=[0.14641300000000002, 0.147644, 0.143757, 0.146918, 0.14446799999999999, 0.14367500000000002, 0.144841, 0.148093, 0.143314]
args= 3 FFI typed nop            median=0.1491 samples=[0.146778, 0.151323, 0.146345, 0.150618, 0.149119, 0.151073, 0.146873, 0.151947, 0.147043]
args= 3 FFI empty kernel         median=3.1932 samples=[3.271205, 3.1834520000000004, 3.193247, 3.238546, 3.205078, 3.190587, 3.123754, 3.193465, 3.191176]
args= 3 INTJ static_compile kernel median=2.8658 samples=[2.9196999999999997, 2.82961, 2.865848, 2.87316, 2.878, 2.8456170000000003, 2.943705, 2.853053, 2.854924]
args= 3 INTJ runtime_shim kernel median=2.8777 samples=[2.942468, 2.860051, 2.877112, 2.903128, 2.8777060000000003, 2.8854290000000002, 2.886091, 2.877316, 2.866715]
args= 5 FFI packed nop           median=0.1760 samples=[0.176006, 0.175517, 0.180417, 0.17594100000000001, 0.17763900000000002, 0.178832, 0.17612799999999998, 0.17566800000000002, 0.175886]
args= 5 FFI typed nop            median=0.1816 samples=[0.182743, 0.181592, 0.17977500000000002, 0.19739199999999998, 0.179785, 0.182881, 0.179364, 0.181624, 0.178361]
args= 5 FFI empty kernel         median=3.2548 samples=[3.283754, 3.2452379999999996, 3.257315, 3.2662, 3.2548209999999997, 3.2206390000000003, 3.2548649999999997, 3.2400949999999997, 3.233647]
args= 5 INTJ static_compile kernel median=2.8925 samples=[2.914182, 2.8795450000000002, 3.1299639999999997, 2.877362, 2.892528, 2.851474, 2.896078, 2.862998, 2.903719]
args= 5 INTJ runtime_shim kernel median=2.8479 samples=[2.930031, 2.903137, 2.833848, 2.88211, 2.8418609999999997, 2.8502310000000004, 2.8428739999999997, 2.8479029999999996, 2.8330230000000003]
args= 8 FFI packed nop           median=0.2105 samples=[0.210368, 0.21052099999999999, 0.21253999999999998, 0.20990299999999998, 0.212535, 0.210111, 0.21127500000000002, 0.20847300000000002, 0.21057800000000002]
args= 8 FFI typed nop            median=0.2139 samples=[0.213833, 0.21506299999999998, 0.21368, 0.215998, 0.217939, 0.214337, 0.2129, 0.213866, 0.213438]
args= 8 FFI empty kernel         median=3.3733 samples=[3.395703, 3.416718, 3.266403, 3.3939589999999997, 3.273653, 3.379841, 3.2640700000000002, 3.3733090000000003, 3.2558000000000002]
args= 8 INTJ static_compile kernel median=2.9732 samples=[3.0008429999999997, 2.883096, 2.9362310000000003, 2.982561, 2.893503, 2.98002, 3.0226480000000002, 2.9731680000000003, 2.88731]
args= 8 INTJ runtime_shim kernel median=2.9105 samples=[2.9994479999999997, 2.87789, 2.856292, 2.91053, 2.914974, 2.858457, 2.944579, 2.8613069999999996, 2.916473]
args=16 FFI packed nop           median=0.3031 samples=[0.295887, 0.305212, 0.313079, 0.30282299999999995, 0.30831200000000003, 0.303102, 0.302274, 0.30156900000000003, 0.30430399999999996]
args=16 FFI typed nop            median=0.3140 samples=[0.308327, 0.315605, 0.313709, 0.317542, 0.320407, 0.315047, 0.313921, 0.314035, 0.312526]
args=16 FFI empty kernel         median=3.6363 samples=[3.648092, 3.688635, 3.516251, 3.663141, 3.5666979999999997, 3.63632, 3.5330529999999998, 3.676208, 3.505612]
args=16 INTJ static_compile kernel median=3.3221 samples=[3.156641, 3.3599, 3.233117, 3.3221350000000003, 3.5234940000000003, 3.2958960000000004, 3.4469279999999998, 3.315389, 3.535939]
args=16 INTJ runtime_shim kernel median=3.2025 samples=[3.202481, 3.089418, 3.349279, 3.05085, 3.3743890000000003, 3.0119659999999997, 3.3308649999999997, 3.016601, 3.365624]
args=32 FFI packed nop           median=0.5011 samples=[0.49285500000000004, 0.502362, 0.505964, 0.5011410000000001, 0.503858, 0.49283, 0.501346, 0.48833800000000005, 0.494869]
args=32 FFI typed nop            median=0.5115 samples=[0.50104, 0.518774, 0.510354, 0.518404, 0.511506, 0.519342, 0.537333, 0.5110629999999999, 0.499907]
args=32 FFI empty kernel         median=4.1376 samples=[4.165568, 4.137568, 4.013896, 4.179996, 4.018293, 4.162029, 3.981217, 4.142196, 3.9610030000000003]
args=32 INTJ static_compile kernel median=3.6104 samples=[3.4134740000000003, 3.6104209999999997, 3.6696500000000003, 3.568578, 3.643911, 3.5679819999999998, 3.6329569999999998, 3.526349, 3.614057]
args=32 INTJ runtime_shim kernel median=3.4343 samples=[3.4015500000000003, 3.396368, 3.6382660000000002, 3.434253, 3.630404, 3.429729, 3.6199920000000003, 3.40949, 3.6015070000000002]
args=64 FFI packed nop           median=0.8693 samples=[0.8502970000000001, 0.869331, 0.871477, 0.865965, 0.88281, 0.867581, 0.895759, 0.8669669999999999, 0.8859239999999999]
args=64 FFI typed nop            median=0.9028 samples=[0.884888, 0.893266, 0.909006, 0.915393, 0.902755, 0.92443, 0.910836, 0.898343, 0.875958]
args=64 FFI empty kernel         median=5.0582 samples=[5.030501, 5.080628, 5.089102, 5.058224, 5.082696, 5.070892, 5.032735, 5.035659, 5.041957]
args=64 INTJ static_compile kernel median=4.5647 samples=[4.535138, 4.538589, 4.5743860000000005, 4.5862110000000005, 4.568566, 4.581393, 4.55734, 4.549079, 4.564696]
args=64 INTJ runtime_shim kernel median=4.5466 samples=[4.547810999999999, 4.554765000000001, 4.534421999999999, 4.5566130000000005, 4.512623, 4.613193, 4.506223, 4.5466109999999995, 4.491703]
elapsed_ns=4229487172
exit_status=0
ended=2026-09-26T22:47:49+08:00
```

#### ffi_paths

round 0:
```text
commit=e3c94b54ffe8159ea62fb7f94a22a7fc6b331c1f
started=2026-09-26T22:45:57+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_intj_ffi_paths.py --mode all --iters 1000 --batches 9
iters=1000 batches=9 unit=ns/call; no_gpu=True
mode=callback
callback_count=9100 (100 warmup + 9000 unique misses)
INTJ cold compile callback + cache  median_ns=414.0 samples_ns=[420.76, 453.369, 372.838, 1376.779, 368.868, 395.61, 398.809, 414.006, 2431.914]
INTJ hot call (no callback)         median_ns=38.6 samples_ns=[40.471, 39.27, 37.553, 39.136, 38.0, 38.584, 37.78, 38.952, 37.809]
INTJ 3-tensor host-only nop         median_ns=34.3 samples_ns=[36.987, 34.641, 34.851, 34.505, 34.332, 34.312, 34.095, 34.28, 34.313]
mode=kwargs
INTJ direct positional              median_ns=66.1 samples_ns=[67.494, 65.699, 65.474, 65.258, 65.618, 69.679, 66.098, 66.555, 66.415]
INTJ adapter positional             median_ns=90.1 samples_ns=[90.534, 90.215, 90.037, 89.421, 90.293, 89.771, 88.891, 90.08, 93.178]
INTJ adapter kwargs                 median_ns=108.7 samples_ns=[106.135, 110.069, 108.497, 109.431, 108.258, 110.605, 108.522, 108.687, 114.981]
INTJ adapter defaults               median_ns=89.7 samples_ns=[89.864, 89.407, 89.656, 88.898, 91.181, 89.7, 89.823, 89.803, 89.12]
INTJ FFI wrapper positional         median_ns=107.2 samples_ns=[137.0, 107.62, 105.532, 107.129, 105.212, 107.591, 107.42, 107.23, 106.733]
INTJ FFI wrapper kwargs             median_ns=126.9 samples_ns=[133.255, 126.935, 128.105, 128.296, 125.599, 125.225, 126.414, 126.348, 131.574]
INTJ native kwargs: unsupported (verified TypeError)
mode=dataclass
INTJ pair direct                    median_ns=66.8 samples_ns=[69.702, 67.549, 65.843, 67.291, 65.502, 66.843, 65.47, 66.766, 70.418]
INTJ pair prebuilt *tuple           median_ns=140.5 samples_ns=[139.152, 141.456, 138.256, 141.247, 138.316, 140.937, 142.601, 140.522, 138.498]
FFI unpack Pair only                median_ns=164.2 samples_ns=[164.24, 165.691, 165.404, 168.352, 162.951, 166.388, 163.458, 163.055, 161.037]
INTJ pair manual unpack             median_ns=174.9 samples_ns=[176.573, 175.57, 173.759, 174.545, 171.779, 175.559, 175.96, 174.943, 171.517]
INTJ pair FFI unpack                median_ns=317.6 samples_ns=[317.627, 325.418, 313.615, 317.581, 323.197, 318.408, 313.975, 321.852, 312.77]
INTJ pair stdlib astuple            median_ns=1161.6 samples_ns=[1334.437, 1205.853, 1155.717, 1165.185, 1151.067, 1166.703, 1151.634, 1161.578, 1150.825]
INTJ config direct                  median_ns=79.0 samples_ns=[78.83, 79.641, 78.805, 79.637, 78.98, 79.784, 79.05, 80.274, 78.976]
INTJ config FFI unpack              median_ns=319.2 samples_ns=[320.198, 319.355, 319.184, 322.621, 316.337, 316.097, 319.26, 316.962, 313.723]
INTJ native dataclass unpack: unsupported (verified TypeError)
elapsed_ns=2884978110
exit_status=0
ended=2026-09-26T22:46:00+08:00
```

round 1:
```text
commit=e3c94b54ffe8159ea62fb7f94a22a7fc6b331c1f
started=2026-09-26T22:46:53+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_intj_ffi_paths.py --mode all --iters 1000 --batches 9
iters=1000 batches=9 unit=ns/call; no_gpu=True
mode=callback
callback_count=9100 (100 warmup + 9000 unique misses)
INTJ cold compile callback + cache  median_ns=414.6 samples_ns=[434.306, 552.485, 382.785, 1197.758, 380.167, 393.522, 405.162, 414.581, 2024.163]
INTJ hot call (no callback)         median_ns=39.3 samples_ns=[41.062, 40.579, 39.341, 44.805, 38.259, 38.946, 38.818, 39.613, 38.978]
INTJ 3-tensor host-only nop         median_ns=34.1 samples_ns=[42.023, 34.531, 34.135, 34.22, 34.055, 34.137, 33.897, 34.218, 33.792]
mode=kwargs
INTJ direct positional              median_ns=65.3 samples_ns=[66.472, 65.859, 65.094, 65.291, 64.998, 65.487, 64.959, 65.287, 65.697]
INTJ adapter positional             median_ns=93.4 samples_ns=[91.447, 90.959, 95.357, 93.762, 92.575, 93.371, 93.131, 94.06, 93.899]
INTJ adapter kwargs                 median_ns=109.9 samples_ns=[109.766, 109.136, 107.336, 117.182, 110.332, 109.709, 110.302, 109.929, 109.931]
INTJ adapter defaults               median_ns=89.4 samples_ns=[88.974, 88.359, 89.438, 88.867, 98.163, 90.916, 89.765, 89.45, 88.893]
INTJ FFI wrapper positional         median_ns=103.3 samples_ns=[103.251, 102.504, 101.631, 103.319, 101.335, 115.671, 105.165, 102.542, 103.314]
INTJ FFI wrapper kwargs             median_ns=126.2 samples_ns=[128.466, 126.443, 125.935, 127.339, 131.021, 126.205, 125.345, 124.754, 124.126]
INTJ native kwargs: unsupported (verified TypeError)
mode=dataclass
INTJ pair direct                    median_ns=68.3 samples_ns=[70.005, 75.073, 67.825, 68.251, 67.419, 68.196, 67.499, 68.305, 68.692]
INTJ pair prebuilt *tuple           median_ns=138.7 samples_ns=[139.73, 140.646, 138.187, 143.012, 136.162, 138.676, 135.031, 139.577, 136.666]
FFI unpack Pair only                median_ns=161.7 samples_ns=[164.675, 165.552, 160.948, 162.368, 159.504, 161.725, 160.695, 164.76, 161.432]
INTJ pair manual unpack             median_ns=172.5 samples_ns=[172.118, 174.423, 172.465, 178.412, 172.513, 174.33, 171.207, 173.036, 170.266]
INTJ pair FFI unpack                median_ns=315.2 samples_ns=[316.062, 313.494, 310.432, 316.014, 312.817, 316.529, 317.291, 315.198, 312.864]
INTJ pair stdlib astuple            median_ns=1164.5 samples_ns=[1366.005, 1204.932, 1153.044, 1166.085, 1152.592, 1166.464, 1164.512, 1155.146, 1147.274]
INTJ config direct                  median_ns=81.1 samples_ns=[81.343, 81.161, 80.591, 81.377, 80.519, 83.869, 80.168, 81.084, 79.245]
INTJ config FFI unpack              median_ns=318.0 samples_ns=[319.396, 321.065, 313.678, 310.528, 309.906, 320.984, 314.303, 317.967, 318.018]
INTJ native dataclass unpack: unsupported (verified TypeError)
elapsed_ns=2778892305
exit_status=0
ended=2026-09-26T22:46:56+08:00
```

round 2:
```text
commit=e3c94b54ffe8159ea62fb7f94a22a7fc6b331c1f
started=2026-09-26T22:47:49+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_intj_ffi_paths.py --mode all --iters 1000 --batches 9
iters=1000 batches=9 unit=ns/call; no_gpu=True
mode=callback
callback_count=9100 (100 warmup + 9000 unique misses)
INTJ cold compile callback + cache  median_ns=406.1 samples_ns=[428.097, 481.82, 378.66, 1225.123, 382.068, 387.238, 398.989, 406.15, 2095.294]
INTJ hot call (no callback)         median_ns=38.7 samples_ns=[40.32, 39.826, 37.655, 38.567, 37.434, 39.07, 41.685, 38.652, 36.915]
INTJ 3-tensor host-only nop         median_ns=34.2 samples_ns=[35.978, 34.493, 34.202, 38.762, 33.999, 34.251, 33.782, 34.174, 33.777]
mode=kwargs
INTJ direct positional              median_ns=65.4 samples_ns=[66.968, 65.858, 65.451, 65.421, 65.433, 65.373, 65.438, 65.509, 65.352]
INTJ adapter positional             median_ns=91.8 samples_ns=[90.09, 90.807, 90.453, 91.821, 92.666, 92.638, 92.846, 92.036, 91.323]
INTJ adapter kwargs                 median_ns=107.8 samples_ns=[107.154, 107.674, 106.857, 107.615, 111.107, 108.273, 109.03, 107.85, 109.9]
INTJ adapter defaults               median_ns=88.6 samples_ns=[89.805, 89.048, 88.61, 88.492, 88.493, 92.155, 87.402, 88.609, 88.142]
INTJ FFI wrapper positional         median_ns=104.9 samples_ns=[105.275, 104.97, 104.81, 103.503, 102.692, 105.998, 110.582, 104.89, 103.493]
INTJ FFI wrapper kwargs             median_ns=126.8 samples_ns=[126.412, 127.131, 126.876, 126.844, 127.074, 135.397, 124.96, 125.972, 125.66]
INTJ native kwargs: unsupported (verified TypeError)
mode=dataclass
INTJ pair direct                    median_ns=68.0 samples_ns=[75.362, 71.161, 68.815, 99.005, 66.106, 68.033, 65.763, 66.854, 65.762]
INTJ pair prebuilt *tuple           median_ns=137.9 samples_ns=[139.235, 139.875, 136.596, 139.237, 138.721, 137.892, 134.448, 137.298, 134.976]
FFI unpack Pair only                median_ns=162.0 samples_ns=[168.204, 170.622, 161.76, 162.134, 161.127, 161.847, 160.416, 164.844, 162.029]
INTJ pair manual unpack             median_ns=174.5 samples_ns=[173.888, 176.729, 174.511, 176.302, 176.523, 177.14, 173.237, 174.273, 172.199]
INTJ pair FFI unpack                median_ns=315.2 samples_ns=[315.398, 315.163, 313.876, 318.51, 312.206, 315.82, 316.4, 314.082, 312.252]
INTJ pair stdlib astuple            median_ns=1161.4 samples_ns=[1326.622, 1209.768, 1147.475, 1161.557, 1138.199, 1162.735, 1149.822, 1161.361, 1143.973]
INTJ config direct                  median_ns=79.4 samples_ns=[78.426, 79.381, 78.295, 80.789, 78.618, 80.553, 78.342, 82.433, 79.667]
INTJ config FFI unpack              median_ns=313.7 samples_ns=[310.505, 313.693, 318.482, 318.818, 315.19, 323.182, 312.094, 313.654, 309.609]
INTJ native dataclass unpack: unsupported (verified TypeError)
elapsed_ns=2782987560
exit_status=0
ended=2026-09-26T22:47:52+08:00
```

#### hip_module

round 0:
```text
commit=e3c94b54ffe8159ea62fb7f94a22a7fc6b331c1f
started=2026-09-26T22:46:00+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_hip_module_launch.py --iters 1000 --batches 9
hsaco_sha256=df3e529e178196ad62f7c07ab8593ede5b8596c76e7c6fbd12a5d947a696e65c kernel=empty_kernel threads=256 shared=0 callback_count=1
iters=1000 batches=9 unit=us/call (host enqueue, no timed sync)
TVM FFI HIP HSACO         median=3.1454 samples=[3.4779299999999997, 3.237199, 3.211875, 3.1454050000000002, 3.127691, 3.0887689999999997, 3.09611, 3.0901509999999996, 3.155091]
Triton same HSACO         median=16.0250 samples=[16.205758, 16.142579, 16.024987, 16.110571, 16.127837, 15.888041, 15.923058000000001, 15.89009, 15.901243000000001]
INTJ same function        median=3.0007 samples=[3.066063, 3.0430729999999997, 3.0007159999999997, 3.035206, 3.131002, 2.9126, 2.9612779999999996, 2.97065, 2.861322]
elapsed_ns=3655967340
exit_status=0
ended=2026-09-26T22:46:04+08:00
```

round 1:
```text
commit=e3c94b54ffe8159ea62fb7f94a22a7fc6b331c1f
started=2026-09-26T22:46:56+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_hip_module_launch.py --iters 1000 --batches 9
hsaco_sha256=df3e529e178196ad62f7c07ab8593ede5b8596c76e7c6fbd12a5d947a696e65c kernel=empty_kernel threads=256 shared=0 callback_count=1
iters=1000 batches=9 unit=us/call (host enqueue, no timed sync)
TVM FFI HIP HSACO         median=3.1387 samples=[3.431529, 3.1632689999999997, 3.163585, 3.141074, 3.138721, 3.047444, 3.110993, 3.0420770000000004, 3.0320590000000003]
Triton same HSACO         median=15.9153 samples=[16.05526, 16.036979, 16.057421, 16.033709, 15.901596999999999, 15.84294, 15.915344, 15.895615, 15.756749]
INTJ same function        median=2.8988 samples=[2.982784, 2.959586, 2.9541999999999997, 2.95214, 2.89879, 2.7842040000000003, 2.8069029999999997, 2.7983000000000002, 2.8084729999999998]
elapsed_ns=3607503897
exit_status=0
ended=2026-09-26T22:47:00+08:00
```

round 2:
```text
commit=e3c94b54ffe8159ea62fb7f94a22a7fc6b331c1f
started=2026-09-26T22:47:52+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_hip_module_launch.py --iters 1000 --batches 9
hsaco_sha256=df3e529e178196ad62f7c07ab8593ede5b8596c76e7c6fbd12a5d947a696e65c kernel=empty_kernel threads=256 shared=0 callback_count=1
iters=1000 batches=9 unit=us/call (host enqueue, no timed sync)
TVM FFI HIP HSACO         median=3.1611 samples=[3.3957020000000004, 3.193886, 3.202997, 3.159756, 3.161128, 3.126215, 3.200312, 3.154351, 3.135173]
Triton same HSACO         median=15.7641 samples=[15.964552, 15.846757, 15.82834, 15.80172, 15.726754999999999, 15.742152, 15.7545, 15.750357, 15.76408]
INTJ same function        median=2.9393 samples=[2.958991, 2.939305, 2.94179, 2.921518, 2.966528, 2.910308, 2.8799609999999998, 2.9739009999999997, 2.9009270000000003]
elapsed_ns=3626222361
exit_status=0
ended=2026-09-26T22:47:56+08:00
```

#### kernel_cache

round 0:
```text
commit=e3c94b54ffe8159ea62fb7f94a22a7fc6b331c1f
started=2026-09-26T22:46:04+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python -m pytest tests/test_kernel_cache.py -s -q

kernel cache, 1-word key, ns per operation

                      intj       tsl      absl
hash_only             1.37      1.35      1.36
hit/1                 1.76      2.04      1.57
hit/8                 1.84      2.13      4.76
hit/64                1.88      2.17      4.96
hit/512               2.54      3.04      5.79
hit_child             2.45      2.45      2.46
miss/1                1.39      1.61      1.39
miss/8                2.41      2.13      3.65
miss/64               2.84      2.11      3.91
miss/512              2.79      2.44      4.46

kernel cache, 2-word key, ns per operation

                      intj       tsl      absl
hash_only             1.41      1.42      1.42
hit/1                 2.18      2.57      2.00
hit/8                 2.18      2.57      5.37
hit/64                2.50      2.91      5.58
hit/512               3.35      4.03      6.83
hit_child             2.46      2.45      2.46
miss/1                1.42      2.32      1.75
miss/8                2.90      1.79      3.61
miss/64               5.17      2.32      4.16
miss/512              3.13      2.72      4.42

kernel cache, 5-word key, ns per operation

                      intj       tsl      absl
hash_only             3.58      3.60      3.58
hit/1                 3.16      3.18      2.83
hit/8                 3.38      3.52     10.01
hit/64                3.44      3.61     10.31
hit/512               4.73      4.91     11.71
hit_child             2.46      2.46      2.46
miss/1                2.05      2.12      1.72
miss/8                3.79      2.15      7.15
miss/64               3.70      1.78      6.91
miss/512              4.01      3.17      7.92
....
4 passed in 24.40s
elapsed_ns=25250707254
exit_status=0
ended=2026-09-26T22:46:29+08:00
```

round 1:
```text
commit=e3c94b54ffe8159ea62fb7f94a22a7fc6b331c1f
started=2026-09-26T22:47:00+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python -m pytest tests/test_kernel_cache.py -s -q

kernel cache, 1-word key, ns per operation

                      intj       tsl      absl
hash_only             1.37      1.36      1.36
hit/1                 1.76      2.03      1.51
hit/8                 1.83      2.13      4.77
hit/64                1.87      2.17      4.95
hit/512               2.54      3.04      5.79
hit_child             2.46      2.49      2.45
miss/1                1.39      1.54      1.38
miss/8                2.40      2.12      3.67
miss/64               2.80      2.12      3.92
miss/512              2.80      2.41      4.42

kernel cache, 2-word key, ns per operation

                      intj       tsl      absl
hash_only             1.42      1.42      1.42
hit/1                 2.18      2.57      1.98
hit/8                 2.19      2.58      5.39
hit/64                2.50      2.92      5.58
hit/512               3.33      4.04      6.83
hit_child             2.46      2.45      2.45
miss/1                1.44      2.38      1.72
miss/8                2.77      1.87      3.61
miss/64               3.42      2.34      4.16
miss/512              3.27      2.72      4.44

kernel cache, 5-word key, ns per operation

                      intj       tsl      absl
hash_only             3.58      3.58      3.58
hit/1                 3.13      3.18      2.83
hit/8                 3.53      3.53      9.98
hit/64                3.67      3.62     10.30
hit/512               4.73      4.92     11.66
hit_child             2.45      2.46      2.45
miss/1                2.05      2.12      1.73
miss/8                3.84      2.18      7.12
miss/64               3.71      1.77      6.91
miss/512              4.09      2.57      7.92
....
4 passed in 24.49s
elapsed_ns=25353051013
exit_status=0
ended=2026-09-26T22:47:25+08:00
```

round 2:
```text
commit=e3c94b54ffe8159ea62fb7f94a22a7fc6b331c1f
started=2026-09-26T22:47:56+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python -m pytest tests/test_kernel_cache.py -s -q

kernel cache, 1-word key, ns per operation

                      intj       tsl      absl
hash_only             1.37      1.35      1.36
hit/1                 1.76      2.03      1.54
hit/8                 1.83      2.14      4.77
hit/64                1.88      2.17      4.94
hit/512               2.54      3.03      5.77
hit_child             2.45      2.46      2.45
miss/1                1.39      1.58      1.38
miss/8                2.38      2.12      3.66
miss/64               2.87      2.12      3.91
miss/512              2.79      2.42      4.42

kernel cache, 2-word key, ns per operation

                      intj       tsl      absl
hash_only             1.42      1.43      1.42
hit/1                 2.18      2.57      1.97
hit/8                 2.18      2.57      5.41
hit/64                2.51      2.91      5.60
hit/512               3.32      4.04      6.82
hit_child             2.46      2.44      2.45
miss/1                1.40      2.31      1.69
miss/8                2.75      1.80      3.61
miss/64               3.27      2.35      4.18
miss/512              3.19      2.72      4.43

kernel cache, 5-word key, ns per operation

                      intj       tsl      absl
hash_only             3.57      3.56      3.58
hit/1                 3.14      3.18      2.83
hit/8                 3.39      3.53      9.99
hit/64                3.52      3.62     10.30
hit/512               4.73      4.96     11.66
hit_child             2.47      2.46      2.46
miss/1                2.01      2.12      1.72
miss/8                3.85      2.15      7.14
miss/64               3.71      1.76      6.91
miss/512              4.15      2.56      7.93
....
4 passed in 24.40s
elapsed_ns=25264744868
exit_status=0
ended=2026-09-26T22:48:21+08:00
```

# Autotune/heuristics benchmark baseline

Task 0 of the autotune/heuristics plan (`docs/superpowers/plans/2026-09-26-autotune-heuristics.md`,
"## Benchmark Gate"). Establishes the baseline every later task's benchmark run is
compared against. This journal records medians only; raw per-round output is included
below in full.

Measured at commit `1f5e5fa` (Design autotune and heuristics support in make_launcher),
September 26, 2026. HEAD at write time is `87c814b` (Plan autotune and heuristics
support in make_launcher), which only adds
`docs/superpowers/plans/2026-09-26-autotune-heuristics.md` on top of `1f5e5fa`
(`git diff 1f5e5fa 87c814b --stat` shows a single added doc file) — no code changed,
so the measurement at `1f5e5fa` is identical to what HEAD would produce.

All four Python benchmark scripts in `benchmarks/`, in every mode (nine cases), plus
the C++ kernel-cache benchmark (`tests/test_kernel_cache.py` / `tests/bench_kernel_cache.cpp`),
run three rounds, one case per process, pinned to core 0, in that order per round
(ten cases x 3 rounds = 30 runs). Driver: `/tmp/intj-bench/run_all.sh baseline`
(copy kept at `.superpowers/sdd/2026-09-26-autotune-heuristics/run_all.sh`). All 30
raw files report `exit_status=0`; `/tmp/intj-bench/baseline/DONE` exists.

## Environment

- CPU: x86-64, `taskset -c 0`, one benchmark process at a time, nothing else running.
- GPU: AMD Instinct MI308X, `gfx942:sramecc+:xnack-` (rocm-smi/rocminfo).
- Python 3.12.3 at `/tmp/gb2/bin/python`; Torch `2.14.0+rocm7.2`; Triton `3.8.0`;
  `apache-tvm-ffi` `0.1.14.post2.dev1+g424558557.d20260924`.
- `INTJ_BENCHMARK_ROOT=/tmp/gbench` (google/benchmark, for the kernel-cache case only).
- Tests/pyright use the uv venv per `AGENTS.md`; benchmarks use `/tmp/gb2/bin/python`
  (no tvm-ffi in the uv venv), matching every earlier journal.

## Commands

Run from `/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj`, for each of
rounds 0, 1, 2, each as its own process, `PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench
taskset -c 0 /tmp/gb2/bin/python` before each line (`kernel_cache` runs pytest instead):

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

## Timing caveats

Same caveats as `benchmarks/AGENTS.md`: GPU rows time host submission/enqueue, not
GPU completion. FFI no-ops launch no kernel and use preconverted TensorViews, not
Torch tensors, so `ffi_compare`/`ffi_sweep` FFI and INTJ rows are different binaries;
`hip_module` shares one HSACO across FFI/Triton/INTJ instead. The `ffi_paths` cold
compile-callback row has a per-call unique cache key and is not comparable to hot
call rows. `launch_sweep`/`launch_last_key`/`launch_host` are host-only (no GPU
submission). CUDA was not exercised on this ROCm host. Kernel-cache numbers are
map-only C++ operation costs (google/benchmark), not the launcher path.

## Per-case medians (3 rounds + median)

Produced with the reusable comparison helper at
`.superpowers/sdd/2026-09-26-autotune-heuristics/bench_compare.py`
(`python bench_compare.py baseline`). It parses every numeric row of every case
(fixed-width `bench_launch.py` tables, `median=`/`median_ns=` lines, and the
kernel-cache intj/tsl/absl tables) and reports the median of the 3 round values.
Later tasks reuse it with `python bench_compare.py baseline <task-label>` to get
per-row deltas and the plan's regression flag.

### GPU launch matrix (`launch_gpu`)

| Row | Round 0 | Round 1 | Round 2 | Median |
| --- | ---: | ---: | ---: | ---: |
| auto map ns/call (ns) | 3137.8 | 3124.0 | 3168.9 | 3137.8 |
| baked ns/call (ns) | 3174.5 | 3054.5 | 3016.9 | 3054.5 |
| bound pointer ns/call (ns) | 3205.1 | 2935.5 | 2918.3 | 2935.5 |
| bound tensor ns/call (ns) | 3198.8 | 3064.7 | 3086.1 | 3086.1 |
| fixed device map ns/call (ns) | 3202.0 | 2932.4 | 2885.3 | 2932.4 |
| fixed device no-map ns/call (ns) | 3218.6 | 3010.8 | 2892.7 | 3010.8 |
| reduced key ns/call (ns) | 3185.2 | 3123.5 | 3098.7 | 3123.5 |
| verify off ns/call (ns) | 3164.9 | 2994.9 | 2976.2 | 2994.9 |
| verify on ns/call (ns) | 3184.6 | 2913.9 | 2882.1 | 2913.9 |

### Host launch matrix (`launch_host`)

| Row | Round 0 | Round 1 | Round 2 | Median |
| --- | ---: | ---: | ---: | ---: |
| auto map ns/call (ns) | 41.5000 | 41.7000 | 41.6000 | 41.6000 |
| baked ns/call (ns) | 38.9000 | 39.2000 | 39.2000 | 39.2000 |
| bound pointer ns/call (ns) | 44.8000 | 44.8000 | 45.5000 | 44.8000 |
| bound tensor ns/call (ns) | 45.6000 | 44.5000 | 69.1000 | 45.6000 |
| fixed device map ns/call (ns) | 42.3000 | 42.0000 | 42.1000 | 42.1000 |
| fixed device no-map ns/call (ns) | 43.8000 | 44.0000 | 44.8000 | 44.0000 |
| reduced key ns/call (ns) | 41.6000 | 41.9000 | 41.9000 | 41.9000 |
| verify off ns/call (ns) | 42.2000 | 42.5000 | 42.6000 | 42.5000 |
| verify on ns/call (ns) | 44.0000 | 44.2000 | 44.2000 | 44.2000 |

### README triton-vs-intj / torch_access (`launch_readme`)

| Row | Round 0 | Round 1 | Round 2 | Median |
| --- | ---: | ---: | ---: | ---: |
| grid=(0,) intj us (us) | 0.0300 | 0.0300 | 0.0300 | 0.0300 |
| grid=(0,) triton us (us) | 12.9600 | 12.9200 | 12.8500 | 12.9200 |
| grid=(1,) intj us (us) | 3.1200 | 3.1100 | 3.1300 | 3.1200 |
| grid=(1,) triton us (us) | 16.4000 | 16.6100 | 16.4500 | 16.4500 |
| interpreter decode ns (ns) | 871.5 | 872.4 | 895.7 | 872.4 |
| runtime_shim decode ns (ns) | 120.8 | 92.0000 | 91.5000 | 92.0000 |
| static_compile decode ns (ns) | 91.4000 | 91.4000 | 91.3000 | 91.4000 |

### Host sweep (4/16/32 args) (`launch_sweep`)

| Row | Round 0 | Round 1 | Round 2 | Median |
| --- | ---: | ---: | ---: | ---: |
| 16 int ns/call (ns) | 67.7000 | 68.0000 | 67.9000 | 67.9000 |
| 16 tensor ns/call (ns) | 65.0000 | 65.2000 | 65.2000 | 65.2000 |
| 32 int ns/call (ns) | 112.1 | 113.4 | 112.8 | 112.8 |
| 32 tensor ns/call (ns) | 116.0 | 113.2 | 114.0 | 114.0 |
| 4 int ns/call (ns) | 40.5000 | 40.5000 | 40.6000 | 40.5000 |
| 4 tensor ns/call (ns) | 39.2000 | 39.1000 | 39.2000 | 39.2000 |

### Last-key sweep (repeat/alternate) (`launch_last_key`)

| Row | Round 0 | Round 1 | Round 2 | Median |
| --- | ---: | ---: | ---: | ---: |
| 16 alternate ns/call (ns) | 65.1000 | 65.4000 | 65.5000 | 65.4000 |
| 16 repeat ns/call (ns) | 62.0000 | 61.8000 | 61.9000 | 61.9000 |
| 32 alternate ns/call (ns) | 109.1 | 109.7 | 109.1 | 109.1 |
| 32 repeat ns/call (ns) | 104.7 | 104.9 | 104.8 | 104.8 |
| 4 alternate ns/call (ns) | 36.0000 | 35.8000 | 35.9000 | 35.9000 |
| 4 repeat ns/call (ns) | 33.9000 | 33.8000 | 33.9000 | 33.9000 |

### FFI vs INTJ compare (`ffi_compare`)

| Row | Round 0 | Round 1 | Round 2 | Median |
| --- | ---: | ---: | ---: | ---: |
| FFI empty kernel (us) | 3.3190 | 3.4138 | 3.3617 | 3.3617 |
| FFI mixed kernel (us) | 3.3407 | 3.3969 | 3.2684 | 3.3407 |
| FFI packed nop (us) | 0.1446 | 0.1444 | 0.1446 | 0.1446 |
| FFI packed nop mixed (us) | 0.1763 | 0.1727 | 0.1745 | 0.1745 |
| FFI typed nop (us) | 0.1499 | 0.1491 | 0.1484 | 0.1491 |
| FFI typed nop mixed (us) | 0.1785 | 0.1769 | 0.1772 | 0.1772 |
| INTJ empty kernel (us) | 2.9064 | 3.0803 | 2.9849 | 2.9849 |
| INTJ fixed mixed kernel (us) | 2.8866 | 3.1054 | 2.9387 | 2.9387 |
| INTJ fixed-device kernel (us) | 2.8998 | 3.0849 | 2.9186 | 2.9186 |
| INTJ mixed kernel (us) | 2.9090 | 3.0217 | 2.9280 | 2.9280 |

### FFI vs INTJ sweep (0..64 args) (`ffi_sweep`)

| Row | Round 0 | Round 1 | Round 2 | Median |
| --- | ---: | ---: | ---: | ---: |
| args=0 FFI empty kernel (us) | 1.6784 | 1.6972 | 1.6464 | 1.6784 |
| args=0 FFI packed nop (us) | 0.1175 | 0.1186 | 0.1172 | 0.1175 |
| args=0 FFI typed nop (us) | 0.1210 | 0.1214 | 0.1204 | 0.1210 |
| args=16 FFI empty kernel (us) | 3.6758 | 3.6540 | 3.6440 | 3.6540 |
| args=16 FFI packed nop (us) | 0.2915 | 0.2932 | 0.2923 | 0.2923 |
| args=16 FFI typed nop (us) | 0.3009 | 0.3046 | 0.3021 | 0.3021 |
| args=3 FFI empty kernel (us) | 3.2295 | 3.2503 | 3.2608 | 3.2503 |
| args=3 FFI packed nop (us) | 0.1451 | 0.1450 | 0.1452 | 0.1451 |
| args=3 FFI typed nop (us) | 0.1501 | 0.1492 | 0.1491 | 0.1492 |
| args=32 FFI empty kernel (us) | 4.2478 | 4.2249 | 4.2123 | 4.2249 |
| args=32 FFI packed nop (us) | 0.4794 | 0.4805 | 0.4816 | 0.4805 |
| args=32 FFI typed nop (us) | 0.4935 | 0.4979 | 0.4946 | 0.4946 |
| args=5 FFI empty kernel (us) | 3.2901 | 3.2918 | 3.2774 | 3.2901 |
| args=5 FFI packed nop (us) | 0.1740 | 0.1733 | 0.1740 | 0.1740 |
| args=5 FFI typed nop (us) | 0.1783 | 0.1769 | 0.1787 | 0.1783 |
| args=64 FFI empty kernel (us) | 5.0502 | 5.0544 | 5.0785 | 5.0544 |
| args=64 FFI packed nop (us) | 0.8379 | 0.8387 | 0.8389 | 0.8387 |
| args=64 FFI typed nop (us) | 0.8587 | 0.8722 | 0.8512 | 0.8587 |
| args=8 FFI empty kernel (us) | 3.3924 | 3.4282 | 3.3830 | 3.3924 |
| args=8 FFI packed nop (us) | 0.2064 | 0.2060 | 0.2060 | 0.2060 |
| args=8 FFI typed nop (us) | 0.2099 | 0.2086 | 0.2101 | 0.2099 |

### INTJ/FFI paths (`ffi_paths`)

| Row | Round 0 | Round 1 | Round 2 | Median |
| --- | ---: | ---: | ---: | ---: |
|  INTJ cold compile callback + cache value (ns) | 466.5 | 428.5 | 429.9 | 429.9 |
| FFI unpack Pair only (ns) | 161.9 | 163.3 | 162.7 | 162.7 |
| INTJ 3-tensor host-only nop (ns) | 33.4000 | 33.8000 | 33.8000 | 33.8000 |
| INTJ FFI wrapper kwargs (ns) | 125.2 | 126.7 | 127.8 | 126.7 |
| INTJ FFI wrapper positional (ns) | 102.3 | 104.6 | 103.4 | 103.4 |
| INTJ adapter defaults (ns) | 88.8000 | 89.8000 | 89.2000 | 89.2000 |
| INTJ adapter kwargs (ns) | 106.1 | 108.2 | 110.3 | 108.2 |
| INTJ adapter positional (ns) | 90.4000 | 90.6000 | 93.8000 | 90.6000 |
| INTJ config FFI unpack (ns) | 318.8 | 321.1 | 322.9 | 321.1 |
| INTJ config direct (ns) | 79.5000 | 79.1000 | 78.6000 | 79.1000 |
| INTJ direct positional (ns) | 64.7000 | 65.2000 | 64.6000 | 64.7000 |
| INTJ hot call (no callback) (ns) | 38.1000 | 37.2000 | 37.2000 | 37.2000 |
| INTJ pair FFI unpack (ns) | 307.6 | 311.3 | 309.1 | 309.1 |
| INTJ pair direct (ns) | 65.3000 | 65.6000 | 65.8000 | 65.6000 |
| INTJ pair manual unpack (ns) | 167.2 | 171.6 | 173.4 | 171.6 |
| INTJ pair prebuilt *tuple (ns) | 136.5 | 138.3 | 137.4 | 137.4 |
| INTJ pair stdlib astuple (ns) | 1189.5 | 1158.1 | 1168.1 | 1168.1 |

### Same-HSACO HIP module launch (`hip_module`)

| Row | Round 0 | Round 1 | Round 2 | Median |
| --- | ---: | ---: | ---: | ---: |
| INTJ same function (us) | 3.0240 | 2.9165 | 2.9336 | 2.9336 |
| TVM FFI HIP HSACO (us) | 3.2595 | 3.1267 | 3.1473 | 3.1473 |
| Triton same HSACO (us) | 15.9169 | 15.9539 | 16.3194 | 15.9539 |

### Kernel cache (C++) (`kernel_cache`)

| Row | Round 0 | Round 1 | Round 2 | Median |
| --- | ---: | ---: | ---: | ---: |
| 1-word: key hash_only absl (ns) | 1.3600 | 1.3600 | 1.3600 | 1.3600 |
| 1-word: key hash_only intj (ns) | 1.3800 | 1.3600 | 1.3600 | 1.3600 |
| 1-word: key hash_only tsl (ns) | 1.3600 | 1.3600 | 1.3600 | 1.3600 |
| 1-word: key hit/1 absl (ns) | 1.3500 | 1.3300 | 1.3600 | 1.3500 |
| 1-word: key hit/1 intj (ns) | 1.2000 | 1.2000 | 1.2100 | 1.2000 |
| 1-word: key hit/1 tsl (ns) | 1.8500 | 1.8500 | 1.8400 | 1.8500 |
| 1-word: key hit/512 absl (ns) | 4.9400 | 4.9300 | 4.9500 | 4.9400 |
| 1-word: key hit/512 intj (ns) | 1.4900 | 1.4800 | 1.5000 | 1.4900 |
| 1-word: key hit/512 tsl (ns) | 2.6200 | 2.6000 | 2.6500 | 2.6200 |
| 1-word: key hit/64 absl (ns) | 4.3600 | 4.3400 | 4.3500 | 4.3500 |
| 1-word: key hit/64 intj (ns) | 1.3000 | 1.3100 | 1.3100 | 1.3100 |
| 1-word: key hit/64 tsl (ns) | 1.9100 | 1.9100 | 1.9100 | 1.9100 |
| 1-word: key hit/8 absl (ns) | 4.2300 | 4.2300 | 4.2400 | 4.2300 |
| 1-word: key hit/8 intj (ns) | 1.2500 | 1.2700 | 1.2600 | 1.2600 |
| 1-word: key hit/8 tsl (ns) | 1.8800 | 1.8800 | 1.8800 | 1.8800 |
| 1-word: key miss/1 absl (ns) | 1.3800 | 1.3800 | 1.3900 | 1.3800 |
| 1-word: key miss/1 intj (ns) | 1.0100 | 1.0300 | 1.0300 | 1.0300 |
| 1-word: key miss/1 tsl (ns) | 1.5600 | 1.4600 | 1.4300 | 1.4600 |
| 1-word: key miss/512 absl (ns) | 3.9400 | 3.9400 | 3.9400 | 3.9400 |
| 1-word: key miss/512 intj (ns) | 2.2300 | 2.2200 | 2.2300 | 2.2300 |
| 1-word: key miss/512 tsl (ns) | 2.0000 | 1.9700 | 1.9700 | 1.9700 |
| 1-word: key miss/64 absl (ns) | 3.8000 | 3.8100 | 3.8200 | 3.8100 |
| 1-word: key miss/64 intj (ns) | 2.1200 | 2.1500 | 2.1700 | 2.1500 |
| 1-word: key miss/64 tsl (ns) | 1.7600 | 1.7300 | 1.7300 | 1.7300 |
| 1-word: key miss/8 absl (ns) | 3.2300 | 3.2500 | 3.2600 | 3.2500 |
| 1-word: key miss/8 intj (ns) | 2.2200 | 2.1900 | 2.2200 | 2.2200 |
| 1-word: key miss/8 tsl (ns) | 1.7400 | 1.7600 | 1.7500 | 1.7500 |
| 2-word: key hash_only absl (ns) | 1.4100 | 1.4200 | 1.4200 | 1.4200 |
| 2-word: key hash_only intj (ns) | 1.4300 | 1.4200 | 1.4200 | 1.4200 |
| 2-word: key hash_only tsl (ns) | 1.4200 | 1.4200 | 1.4200 | 1.4200 |
| 2-word: key hit/1 absl (ns) | 1.7000 | 1.7100 | 1.7000 | 1.7000 |
| 2-word: key hit/1 intj (ns) | 1.8900 | 1.9000 | 1.8900 | 1.8900 |
| 2-word: key hit/1 tsl (ns) | 2.2800 | 2.2600 | 2.2600 | 2.2600 |
| 2-word: key hit/512 absl (ns) | 5.8300 | 5.7900 | 5.8500 | 5.8300 |
| 2-word: key hit/512 intj (ns) | 2.7300 | 2.7300 | 2.7300 | 2.7300 |
| 2-word: key hit/512 tsl (ns) | 3.4600 | 3.4600 | 3.4700 | 3.4600 |
| 2-word: key hit/64 absl (ns) | 5.2500 | 5.2400 | 5.2700 | 5.2500 |
| 2-word: key hit/64 intj (ns) | 2.1300 | 2.1000 | 2.1100 | 2.1100 |
| 2-word: key hit/64 tsl (ns) | 2.5700 | 2.5800 | 2.5900 | 2.5800 |
| 2-word: key hit/8 absl (ns) | 4.7800 | 4.7600 | 4.7600 | 4.7600 |
| 2-word: key hit/8 intj (ns) | 1.8900 | 1.9000 | 1.8900 | 1.8900 |
| 2-word: key hit/8 tsl (ns) | 2.2600 | 2.2800 | 2.2700 | 2.2700 |
| 2-word: key miss/1 absl (ns) | 1.6100 | 1.4700 | 1.5400 | 1.5400 |
| 2-word: key miss/1 intj (ns) | 1.3200 | 1.3000 | 1.3200 | 1.3200 |
| 2-word: key miss/1 tsl (ns) | 2.1600 | 2.1600 | 2.1700 | 2.1600 |
| 2-word: key miss/512 absl (ns) | 3.9500 | 3.9000 | 3.8700 | 3.9000 |
| 2-word: key miss/512 intj (ns) | 2.9600 | 2.9400 | 2.9500 | 2.9500 |
| 2-word: key miss/512 tsl (ns) | 2.3000 | 2.3000 | 2.3000 | 2.3000 |
| 2-word: key miss/64 absl (ns) | 3.3600 | 3.3600 | 3.3600 | 3.3600 |
| 2-word: key miss/64 intj (ns) | 3.0800 | 3.1300 | 3.0700 | 3.0800 |
| 2-word: key miss/64 tsl (ns) | 2.0100 | 2.0200 | 2.0000 | 2.0100 |
| 2-word: key miss/8 absl (ns) | 3.2700 | 3.2700 | 3.2600 | 3.2700 |
| 2-word: key miss/8 intj (ns) | 2.7600 | 2.7100 | 2.4500 | 2.7100 |
| 2-word: key miss/8 tsl (ns) | 1.7000 | 1.7500 | 1.7700 | 1.7500 |
| 5-word: key hash_only absl (ns) | 3.5600 | 3.5700 | 3.5800 | 3.5700 |
| 5-word: key hash_only intj (ns) | 3.5700 | 3.5900 | 3.5800 | 3.5800 |
| 5-word: key hash_only tsl (ns) | 3.5700 | 3.5800 | 3.5900 | 3.5800 |
| 5-word: key hit/1 absl (ns) | 2.6800 | 2.6800 | 2.6800 | 2.6800 |
| 5-word: key hit/1 intj (ns) | 2.9400 | 2.9400 | 2.9400 | 2.9400 |
| 5-word: key hit/1 tsl (ns) | 3.1900 | 3.1900 | 3.1900 | 3.1900 |
| 5-word: key hit/512 absl (ns) | 11.1700 | 11.2100 | 11.1800 | 11.1800 |
| 5-word: key hit/512 intj (ns) | 4.4600 | 4.4700 | 4.4700 | 4.4700 |
| 5-word: key hit/512 tsl (ns) | 4.9400 | 4.9400 | 4.9400 | 4.9400 |
| 5-word: key hit/64 absl (ns) | 9.7300 | 9.7500 | 9.7600 | 9.7500 |
| 5-word: key hit/64 intj (ns) | 3.2500 | 3.2500 | 3.2500 | 3.2500 |
| 5-word: key hit/64 tsl (ns) | 3.6500 | 3.6400 | 3.6400 | 3.6400 |
| 5-word: key hit/8 absl (ns) | 9.5000 | 9.4600 | 9.4700 | 9.4700 |
| 5-word: key hit/8 intj (ns) | 3.1700 | 3.1700 | 3.1700 | 3.1700 |
| 5-word: key hit/8 tsl (ns) | 3.5400 | 3.5400 | 3.5400 | 3.5400 |
| 5-word: key miss/1 absl (ns) | 1.6400 | 1.6600 | 1.6700 | 1.6600 |
| 5-word: key miss/1 intj (ns) | 1.8700 | 1.8300 | 1.8500 | 1.8500 |
| 5-word: key miss/1 tsl (ns) | 2.2700 | 2.2900 | 2.2700 | 2.2700 |
| 5-word: key miss/512 absl (ns) | 7.6600 | 7.6600 | 7.6500 | 7.6600 |
| 5-word: key miss/512 intj (ns) | 4.1100 | 4.0800 | 4.1000 | 4.1000 |
| 5-word: key miss/512 tsl (ns) | 2.7800 | 2.7600 | 2.7000 | 2.7600 |
| 5-word: key miss/64 absl (ns) | 6.7200 | 6.7100 | 6.7000 | 6.7100 |
| 5-word: key miss/64 intj (ns) | 3.4200 | 3.3700 | 3.3800 | 3.3800 |
| 5-word: key miss/64 tsl (ns) | 2.5400 | 2.5800 | 2.5500 | 2.5500 |
| 5-word: key miss/8 absl (ns) | 6.9300 | 6.9300 | 6.9400 | 6.9300 |
| 5-word: key miss/8 intj (ns) | 3.6200 | 3.5700 | 3.5500 | 3.5700 |
| 5-word: key miss/8 tsl (ns) | 3.0300 | 3.2500 | 3.0900 | 3.0900 |

## Raw outputs

### launch_gpu

round 0:
```text
commit=1f5e5fa559c2564249026cd7f765420b8aa90eca
started=2026-09-26T21:18:12+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --iters 1000 --batches 9
mode=gpu; 1000 calls × 9 batches; median ns/call
               path    ns/call       Δ ns       Δ %   build ms   bind ms
           auto map     3137.8       +0.0      +0.0      73.97         -
        reduced key     3185.2      +47.4      +1.5    2113.07         -
         verify off     3164.9      +27.1      +0.9    2028.03         -
          verify on     3184.6      +46.8      +1.5    2117.13         -
              baked     3174.5      +36.6      +1.2    1949.71         -
       bound tensor     3198.8      +60.9      +1.9       0.45   1922.62
      bound pointer     3205.1      +67.2      +2.1       0.44   1955.08
   fixed device map     3202.0      +64.2      +2.0       0.40   2031.35
fixed device no-map     3218.6      +80.7      +2.6       0.45   1946.93
elapsed_ns=19678315438
exit_status=0
ended=2026-09-26T21:18:32+08:00
```

round 1:
```text
commit=1f5e5fa559c2564249026cd7f765420b8aa90eca
started=2026-09-26T21:20:07+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --iters 1000 --batches 9
mode=gpu; 1000 calls × 9 batches; median ns/call
               path    ns/call       Δ ns       Δ %   build ms   bind ms
           auto map     3124.0       +0.0      +0.0      73.70         -
        reduced key     3123.5       -0.5      -0.0       3.63         -
         verify off     2994.9     -129.0      -4.1       1.92         -
          verify on     2913.9     -210.1      -6.7       1.84         -
              baked     3054.5      -69.4      -2.2      72.82         -
       bound tensor     3064.7      -59.3      -1.9       0.37      1.69
      bound pointer     2935.5     -188.4      -6.0       0.38      1.76
   fixed device map     2932.4     -191.5      -6.1       0.32      1.76
fixed device no-map     3010.8     -113.2      -3.6       0.28      1.73
elapsed_ns=3655655255
exit_status=0
ended=2026-09-26T21:20:10+08:00
```

round 2:
```text
commit=1f5e5fa559c2564249026cd7f765420b8aa90eca
started=2026-09-26T21:21:00+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --iters 1000 --batches 9
mode=gpu; 1000 calls × 9 batches; median ns/call
               path    ns/call       Δ ns       Δ %   build ms   bind ms
           auto map     3168.9       +0.0      +0.0      73.73         -
        reduced key     3098.7      -70.2      -2.2       3.51         -
         verify off     2976.2     -192.8      -6.1       1.90         -
          verify on     2882.1     -286.9      -9.1       1.86         -
              baked     3016.9     -152.0      -4.8      72.79         -
       bound tensor     3086.1      -82.9      -2.6       0.36      1.71
      bound pointer     2918.3     -250.6      -7.9       0.35      1.79
   fixed device map     2885.3     -283.7      -9.0       0.32      1.79
fixed device no-map     2892.7     -276.2      -8.7       0.29      1.73
elapsed_ns=3625862668
exit_status=0
ended=2026-09-26T21:21:03+08:00
```

### launch_host

round 0:
```text
commit=1f5e5fa559c2564249026cd7f765420b8aa90eca
started=2026-09-26T21:18:32+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --no-gpu --iters 100000 --batches 9
mode=host-only; 100000 calls × 9 batches; median ns/call
               path    ns/call       Δ ns       Δ %   build ms   bind ms
           auto map       41.5       +0.0      +0.0      61.03         -
        reduced key       41.6       +0.1      +0.4       1.77         -
         verify off       42.2       +0.7      +1.7       1.62         -
          verify on       44.0       +2.6      +6.2       1.58         -
              baked       38.9       -2.6      -6.2       2.75         -
       bound tensor       45.6       +4.1      +9.9       0.15      1.51
      bound pointer       44.8       +3.4      +8.1       0.13      1.49
   fixed device map       42.3       +0.9      +2.1       0.11      1.46
fixed device no-map       43.8       +2.4      +5.8       0.15      1.47
elapsed_ns=2857240434
exit_status=0
ended=2026-09-26T21:18:35+08:00
```

round 1:
```text
commit=1f5e5fa559c2564249026cd7f765420b8aa90eca
started=2026-09-26T21:20:10+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --no-gpu --iters 100000 --batches 9
mode=host-only; 100000 calls × 9 batches; median ns/call
               path    ns/call       Δ ns       Δ %   build ms   bind ms
           auto map       41.7       +0.0      +0.0      61.12         -
        reduced key       41.9       +0.2      +0.6       1.77         -
         verify off       42.5       +0.8      +1.9       1.62         -
          verify on       44.2       +2.5      +6.0       1.57         -
              baked       39.2       -2.5      -6.0       2.86         -
       bound tensor       44.5       +2.8      +6.7       0.15      1.51
      bound pointer       44.8       +3.2      +7.6       0.13      1.47
   fixed device map       42.0       +0.4      +0.9       0.11      1.48
fixed device no-map       44.0       +2.4      +5.7       0.15      1.47
elapsed_ns=2867298887
exit_status=0
ended=2026-09-26T21:20:13+08:00
```

round 2:
```text
commit=1f5e5fa559c2564249026cd7f765420b8aa90eca
started=2026-09-26T21:21:03+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --no-gpu --iters 100000 --batches 9
mode=host-only; 100000 calls × 9 batches; median ns/call
               path    ns/call       Δ ns       Δ %   build ms   bind ms
           auto map       41.6       +0.0      +0.0      60.86         -
        reduced key       41.9       +0.3      +0.8       1.77         -
         verify off       42.6       +1.0      +2.3       1.62         -
          verify on       44.2       +2.6      +6.3       1.55         -
              baked       39.2       -2.4      -5.8       2.74         -
       bound tensor       69.1      +27.5     +66.1       0.15      1.50
      bound pointer       45.5       +3.9      +9.4       0.13      1.47
   fixed device map       42.1       +0.5      +1.3       0.11      1.44
fixed device no-map       44.8       +3.2      +7.6       0.14      1.47
elapsed_ns=2887382333
exit_status=0
ended=2026-09-26T21:21:06+08:00
```

### launch_readme

round 0:
```text
commit=1f5e5fa559c2564249026cd7f765420b8aa90eca
started=2026-09-26T21:18:35+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --readme --iters 1000 --batches 9
mode=readme; 1000 calls × 9 batches; median
        path   triton us    intj us   speedup
   grid=(1,)       16.40       3.12      5.3x
   grid=(0,)       12.96       0.03    450.7x

torch_access_mode   decode ns    build s
     runtime_shim       120.8       0.02
   static_compile        91.4       0.06
      interpreter       871.5       0.00
elapsed_ns=3648139142
exit_status=0
ended=2026-09-26T21:18:38+08:00
```

round 1:
```text
commit=1f5e5fa559c2564249026cd7f765420b8aa90eca
started=2026-09-26T21:20:13+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --readme --iters 1000 --batches 9
mode=readme; 1000 calls × 9 batches; median
        path   triton us    intj us   speedup
   grid=(1,)       16.61       3.11      5.3x
   grid=(0,)       12.92       0.03    459.0x

torch_access_mode   decode ns    build s
     runtime_shim        92.0       0.02
   static_compile        91.4       0.06
      interpreter       872.4       0.00
elapsed_ns=3653901056
exit_status=0
ended=2026-09-26T21:20:17+08:00
```

round 2:
```text
commit=1f5e5fa559c2564249026cd7f765420b8aa90eca
started=2026-09-26T21:21:06+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --readme --iters 1000 --batches 9
mode=readme; 1000 calls × 9 batches; median
        path   triton us    intj us   speedup
   grid=(1,)       16.45       3.13      5.3x
   grid=(0,)       12.85       0.03    456.0x

torch_access_mode   decode ns    build s
     runtime_shim        91.5       0.02
   static_compile        91.3       0.06
      interpreter       895.7       0.00
elapsed_ns=3657835381
exit_status=0
ended=2026-09-26T21:21:10+08:00
```

### launch_sweep

round 0:
```text
commit=1f5e5fa559c2564249026cd7f765420b8aa90eca
started=2026-09-26T21:18:38+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --sweep --iters 100000 --batches 9
mode=sweep; host-only; 100000 calls × 9 batches; median ns/call
count   kind    ns/call
    4    int       40.5
    4 tensor       39.2
   16    int       67.7
   16 tensor       65.0
   32    int      112.1
   32 tensor      116.0
elapsed_ns=2904235810
exit_status=0
ended=2026-09-26T21:18:41+08:00
```

round 1:
```text
commit=1f5e5fa559c2564249026cd7f765420b8aa90eca
started=2026-09-26T21:20:17+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --sweep --iters 100000 --batches 9
mode=sweep; host-only; 100000 calls × 9 batches; median ns/call
count   kind    ns/call
    4    int       40.5
    4 tensor       39.1
   16    int       68.0
   16 tensor       65.2
   32    int      113.4
   32 tensor      113.2
elapsed_ns=2925436351
exit_status=0
ended=2026-09-26T21:20:20+08:00
```

round 2:
```text
commit=1f5e5fa559c2564249026cd7f765420b8aa90eca
started=2026-09-26T21:21:10+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --sweep --iters 100000 --batches 9
mode=sweep; host-only; 100000 calls × 9 batches; median ns/call
count   kind    ns/call
    4    int       40.6
    4 tensor       39.2
   16    int       67.9
   16 tensor       65.2
   32    int      112.8
   32 tensor      114.0
elapsed_ns=2907327511
exit_status=0
ended=2026-09-26T21:21:13+08:00
```

### launch_last_key

round 0:
```text
commit=1f5e5fa559c2564249026cd7f765420b8aa90eca
started=2026-09-26T21:18:41+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --last-key --iters 100000 --batches 9
mode=last-key; host-only; 200000 calls × 9 batches; median ns/call
count     pattern    ns/call
    4      repeat       33.9
    4   alternate       36.0
   16      repeat       62.0
   16   alternate       65.1
   32      repeat      104.7
   32   alternate      109.1
elapsed_ns=3248135892
exit_status=0
ended=2026-09-26T21:18:44+08:00
```

round 1:
```text
commit=1f5e5fa559c2564249026cd7f765420b8aa90eca
started=2026-09-26T21:20:20+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --last-key --iters 100000 --batches 9
mode=last-key; host-only; 200000 calls × 9 batches; median ns/call
count     pattern    ns/call
    4      repeat       33.8
    4   alternate       35.8
   16      repeat       61.8
   16   alternate       65.4
   32      repeat      104.9
   32   alternate      109.7
elapsed_ns=3241534704
exit_status=0
ended=2026-09-26T21:20:23+08:00
```

round 2:
```text
commit=1f5e5fa559c2564249026cd7f765420b8aa90eca
started=2026-09-26T21:21:13+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --last-key --iters 100000 --batches 9
mode=last-key; host-only; 200000 calls × 9 batches; median ns/call
count     pattern    ns/call
    4      repeat       33.9
    4   alternate       35.9
   16      repeat       61.9
   16   alternate       65.5
   32      repeat      104.8
   32   alternate      109.1
elapsed_ns=3250181098
exit_status=0
ended=2026-09-26T21:21:16+08:00
```

### ffi_compare

round 0:
```text
commit=1f5e5fa559c2564249026cd7f765420b8aa90eca
started=2026-09-26T21:18:44+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_ffi_compare.py --iters 1000 --batches 9
iters=1000 batches=9 unit=us/call (host-only, no timed sync)
FFI packed nop             median=0.1446 samples=[0.151205, 0.14487899999999998, 0.1441, 0.145135, 0.14315899999999998, 0.144601, 0.143257, 0.14461600000000002, 0.14421799999999999]
FFI typed nop              median=0.1499 samples=[0.151047, 0.151124, 0.15020599999999998, 0.149893, 0.15111000000000002, 0.149688, 0.14829699999999998, 0.149117, 0.146787]
FFI empty kernel           median=3.3190 samples=[3.744017, 3.461862, 3.466701, 3.259777, 3.246889, 3.3022069999999997, 3.29615, 3.5314479999999997, 3.319026]
INTJ empty kernel          median=2.9064 samples=[3.060978, 3.07847, 3.0791350000000004, 2.870116, 2.8449850000000003, 2.908699, 2.898745, 2.9063600000000003, 2.899934]
INTJ fixed-device kernel   median=2.8998 samples=[3.0302, 3.09763, 3.087062, 2.8863290000000004, 2.865793, 2.880253, 2.8847840000000002, 2.899759, 2.961018]
FFI packed nop mixed       median=0.1763 samples=[0.172023, 0.176338, 0.17702099999999998, 0.177727, 0.17550200000000002, 0.176073, 0.176057, 0.177695, 0.17987999999999998]
FFI typed nop mixed        median=0.1785 samples=[0.175753, 0.179096, 0.179074, 0.177716, 0.176277, 0.178534, 0.179948, 0.179187, 0.177429]
FFI mixed kernel           median=3.3407 samples=[3.4457750000000003, 3.5110129999999997, 3.349659, 3.329838, 3.273504, 3.325796, 3.340652, 3.326032, 3.3408420000000003]
INTJ mixed kernel          median=2.9090 samples=[3.094007, 3.088265, 2.916103, 2.908962, 2.832556, 2.8549670000000003, 2.906643, 2.934317, 2.906511]
INTJ fixed mixed kernel    median=2.8866 samples=[3.122111, 3.1068409999999997, 2.9543719999999998, 2.9378249999999997, 2.8865659999999997, 2.833569, 2.861747, 2.865367, 2.861658]
elapsed_ns=11428382617
exit_status=0
ended=2026-09-26T21:18:56+08:00
```

round 1:
```text
commit=1f5e5fa559c2564249026cd7f765420b8aa90eca
started=2026-09-26T21:20:23+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_ffi_compare.py --iters 1000 --batches 9
iters=1000 batches=9 unit=us/call (host-only, no timed sync)
FFI packed nop             median=0.1444 samples=[0.147112, 0.144687, 0.147548, 0.144361, 0.144331, 0.14439, 0.143633, 0.144244, 0.144191]
FFI typed nop              median=0.1491 samples=[0.14970599999999998, 0.14918199999999998, 0.14765799999999998, 0.14994, 0.147401, 0.14981999999999998, 0.148147, 0.149052, 0.146448]
FFI empty kernel           median=3.4138 samples=[3.560962, 3.504929, 3.481663, 3.287274, 3.3434229999999996, 3.413844, 3.2106999999999997, 3.541918, 3.2269200000000002]
INTJ empty kernel          median=3.0803 samples=[3.0661970000000003, 3.178196, 3.156655, 2.983837, 2.951256, 3.080289, 3.081484, 3.087547, 3.06279]
INTJ fixed-device kernel   median=3.0849 samples=[3.105285, 3.1829769999999997, 3.1588249999999998, 3.229825, 2.9682150000000003, 2.8270929999999996, 2.817447, 3.08488, 2.8515680000000003]
FFI packed nop mixed       median=0.1727 samples=[0.169886, 0.169912, 0.174519, 0.171952, 0.17385, 0.172565, 0.173433, 0.172681, 0.17429]
FFI typed nop mixed        median=0.1769 samples=[0.177177, 0.17665199999999998, 0.175043, 0.178364, 0.174132, 0.176931, 0.175869, 0.17788900000000002, 0.179611]
FFI mixed kernel           median=3.3969 samples=[3.396859, 3.556362, 3.372341, 3.3734070000000003, 3.316751, 3.379648, 3.411916, 3.507542, 3.418653]
INTJ mixed kernel          median=3.0217 samples=[3.225639, 3.212662, 3.021732, 3.025005, 2.964498, 2.9963960000000003, 2.865296, 3.1276669999999998, 2.848436]
INTJ fixed mixed kernel    median=3.1054 samples=[3.29301, 3.272184, 3.130625, 3.218259, 3.0287379999999997, 3.0175419999999997, 3.087016, 3.09847, 3.1054009999999996]
elapsed_ns=3702949421
exit_status=0
ended=2026-09-26T21:20:27+08:00
```

round 2:
```text
commit=1f5e5fa559c2564249026cd7f765420b8aa90eca
started=2026-09-26T21:21:16+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_ffi_compare.py --iters 1000 --batches 9
iters=1000 batches=9 unit=us/call (host-only, no timed sync)
FFI packed nop             median=0.1446 samples=[0.149189, 0.142333, 0.143101, 0.142174, 0.145921, 0.144625, 0.147451, 0.143167, 0.145465]
FFI typed nop              median=0.1484 samples=[0.148391, 0.14735499999999999, 0.149914, 0.149, 0.148504, 0.148375, 0.14486500000000002, 0.147197, 0.146482]
FFI empty kernel           median=3.3617 samples=[3.591088, 3.388694, 3.37519, 3.19583, 3.195667, 3.361742, 3.1888449999999997, 3.373125, 3.197521]
INTJ empty kernel          median=2.9849 samples=[3.123665, 3.10928, 3.099573, 2.926053, 2.890087, 2.998108, 2.98493, 2.982828, 2.982429]
INTJ fixed-device kernel   median=2.9186 samples=[3.0189850000000003, 3.124624, 3.1330050000000003, 2.941149, 2.90719, 2.8192939999999997, 2.813864, 2.842235, 2.918589]
FFI packed nop mixed       median=0.1745 samples=[0.17530199999999999, 0.174534, 0.175147, 0.172359, 0.17393, 0.177053, 0.173102, 0.175535, 0.173547]
FFI typed nop mixed        median=0.1772 samples=[0.17441399999999999, 0.177725, 0.174008, 0.179083, 0.17991, 0.180091, 0.176917, 0.17718899999999999, 0.175966]
FFI mixed kernel           median=3.2684 samples=[3.388817, 3.444338, 3.368755, 3.264773, 3.195246, 3.268409, 3.262284, 3.345288, 3.258909]
INTJ mixed kernel          median=2.9280 samples=[3.150524, 3.108348, 2.9475230000000003, 2.9279699999999997, 2.868913, 2.8725259999999997, 2.827877, 2.977885, 2.81702]
INTJ fixed mixed kernel    median=2.9387 samples=[3.144986, 3.129715, 2.98868, 3.002627, 2.8917770000000003, 2.860831, 2.913542, 2.925549, 2.9386970000000003]
elapsed_ns=3626076659
exit_status=0
ended=2026-09-26T21:21:20+08:00
```

### ffi_sweep

round 0:
```text
commit=1f5e5fa559c2564249026cd7f765420b8aa90eca
started=2026-09-26T21:18:56+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_ffi_compare.py --sweep --iters 1000 --batches 9
counts=[0, 3, 5, 8, 16, 32, 64] iters=1000 batches=9 unit=us/call (host-only, no timed sync); INTJ device-bound, no specialization key
args= 0 FFI packed nop           median=0.1175 samples=[0.129673, 0.119565, 0.11800400000000001, 0.118351, 0.117491, 0.11647199999999999, 0.11584900000000001, 0.11681699999999999, 0.11363200000000001]
args= 0 FFI typed nop            median=0.1210 samples=[0.124109, 0.122881, 0.121009, 0.122703, 0.12037099999999999, 0.123987, 0.119965, 0.118127, 0.11567400000000001]
args= 0 FFI empty kernel         median=1.6784 samples=[1.6783569999999999, 1.778185, 1.629625, 1.82856, 1.593698, 1.820249, 1.601942, 1.787036, 1.6386189999999998]
args= 0 INTJ static_compile kernel median=2.7547 samples=[3.243148, 2.754669, 2.765601, 2.7411640000000004, 2.73123, 2.7201709999999997, 2.716312, 2.766735, 2.757841]
args= 0 INTJ runtime_shim kernel median=2.7243 samples=[2.887703, 2.719578, 2.769657, 2.6779319999999998, 2.746205, 2.673788, 2.7243359999999996, 2.70208, 2.7501860000000002]
args= 3 FFI packed nop           median=0.1451 samples=[0.14461600000000002, 0.14455199999999999, 0.146694, 0.145625, 0.14765999999999999, 0.14505500000000002, 0.14432, 0.146914, 0.14510499999999998]
args= 3 FFI typed nop            median=0.1501 samples=[0.148842, 0.15007900000000002, 0.14698, 0.151202, 0.14793199999999998, 0.150564, 0.15009999999999998, 0.150561, 0.147863]
args= 3 FFI empty kernel         median=3.2295 samples=[3.387801, 3.217219, 3.2423270000000004, 3.240582, 3.2189810000000003, 3.229464, 3.196792, 3.2656300000000003, 3.2257710000000004]
args= 3 INTJ static_compile kernel median=2.9102 samples=[3.015379, 2.878519, 2.917771, 2.8835610000000003, 2.9101619999999997, 2.883759, 2.8941950000000003, 2.93137, 2.915427]
args= 3 INTJ runtime_shim kernel median=2.9017 samples=[3.022569, 2.881525, 2.915349, 2.894446, 2.9017060000000003, 2.884363, 2.8838820000000003, 2.918797, 2.9019749999999997]
args= 5 FFI packed nop           median=0.1740 samples=[0.17497, 0.17399799999999999, 0.17424299999999998, 0.17313399999999998, 0.172807, 0.172761, 0.175107, 0.170526, 0.175725]
args= 5 FFI typed nop            median=0.1783 samples=[0.175127, 0.178161, 0.178677, 0.178804, 0.178263, 0.178937, 0.176293, 0.177563, 0.17890799999999998]
args= 5 FFI empty kernel         median=3.2901 samples=[3.384047, 3.290085, 3.290361, 3.289917, 3.296697, 3.275263, 3.2490949999999996, 3.349029, 3.279124]
args= 5 INTJ static_compile kernel median=2.9015 samples=[3.0120639999999996, 2.9014699999999998, 2.916192, 2.871201, 2.900319, 2.855334, 2.8849169999999997, 2.921191, 2.910468]
args= 5 INTJ runtime_shim kernel median=2.8943 samples=[3.040465, 2.91094, 2.8742289999999997, 2.890857, 2.9023209999999997, 2.86464, 2.8465700000000003, 2.9286779999999997, 2.894278]
args= 8 FFI packed nop           median=0.2064 samples=[0.206064, 0.204994, 0.207148, 0.20736500000000002, 0.207411, 0.21061600000000003, 0.206429, 0.20538399999999998, 0.20569300000000001]
args= 8 FFI typed nop            median=0.2099 samples=[0.209063, 0.21178899999999998, 0.209487, 0.209941, 0.208091, 0.21124500000000002, 0.206259, 0.213277, 0.209867]
args= 8 FFI empty kernel         median=3.3924 samples=[3.507317, 3.392382, 3.337297, 3.41236, 3.3182240000000003, 3.424486, 3.308281, 3.462864, 3.323555]
args= 8 INTJ static_compile kernel median=2.9344 samples=[3.08085, 2.897786, 2.9446779999999997, 3.006504, 2.9210439999999998, 2.9344319999999997, 2.914661, 3.005734, 2.931415]
args= 8 INTJ runtime_shim kernel median=2.9301 samples=[3.078877, 2.9204380000000003, 2.9680050000000002, 2.883916, 2.953975, 2.889146, 2.92531, 2.9301239999999997, 2.9662469999999996]
args=16 FFI packed nop           median=0.2915 samples=[0.29509399999999997, 0.287525, 0.291618, 0.290359, 0.292548, 0.293414, 0.291477, 0.287452, 0.28846]
args=16 FFI typed nop            median=0.3009 samples=[0.303517, 0.303713, 0.30092399999999997, 0.299623, 0.30091500000000004, 0.304031, 0.304073, 0.300772, 0.30087]
args=16 FFI empty kernel         median=3.6758 samples=[3.761476, 3.686201, 3.572696, 3.675771, 3.5496849999999998, 3.7168, 3.539273, 3.695685, 3.56581]
args=16 INTJ static_compile kernel median=3.4701 samples=[3.279059, 3.470134, 3.582009, 3.3324499999999997, 3.5922069999999997, 3.366212, 3.561814, 3.377153, 3.53452]
args=16 INTJ runtime_shim kernel median=3.2314 samples=[3.231449, 3.1138760000000003, 3.4256390000000003, 3.052708, 3.409902, 3.040371, 3.422903, 3.0837939999999997, 3.402008]
args=32 FFI packed nop           median=0.4794 samples=[0.476852, 0.48393200000000003, 0.48207799999999995, 0.47936, 0.478901, 0.478691, 0.48200299999999996, 0.47102999999999995, 0.480607]
args=32 FFI typed nop            median=0.4935 samples=[0.493541, 0.493363, 0.49618599999999996, 0.494103, 0.49328, 0.493836, 0.497151, 0.487719, 0.487656]
args=32 FFI empty kernel         median=4.2478 samples=[4.300192, 4.258527, 4.099817, 4.247822, 4.0929459999999995, 4.278723, 4.0815079999999995, 4.283599000000001, 4.105083]
args=32 INTJ static_compile kernel median=3.6394 samples=[3.4706129999999997, 3.639421, 3.7064969999999997, 3.5995700000000004, 3.693474, 3.6204029999999996, 3.697483, 3.63626, 3.68795]
args=32 INTJ runtime_shim kernel median=3.4640 samples=[3.429372, 3.451428, 3.651861, 3.4341109999999997, 3.655984, 3.431324, 4.087748, 3.464001, 3.629171]
args=64 FFI packed nop           median=0.8379 samples=[0.833171, 0.847947, 0.841302, 0.817451, 0.846699, 0.835415, 0.8378680000000001, 0.844076, 0.835814]
args=64 FFI typed nop            median=0.8587 samples=[0.842128, 0.866491, 0.8654919999999999, 0.855234, 0.853171, 0.854968, 0.865043, 0.873922, 0.858711]
args=64 FFI empty kernel         median=5.0502 samples=[5.069063, 5.020689999999999, 5.051361, 5.064387, 5.039569, 5.029472, 5.035072, 5.052972, 5.050224]
args=64 INTJ static_compile kernel median=4.6714 samples=[4.671488999999999, 4.672147, 4.6345860000000005, 4.696838, 4.648658999999999, 4.653488, 4.671403000000001, 4.665877999999999, 4.699464]
args=64 INTJ runtime_shim kernel median=4.6030 samples=[4.548666, 4.680350000000001, 4.641189000000001, 4.623066, 4.545599, 4.602969, 4.594758, 4.632957, 4.571744]
elapsed_ns=34621291303
exit_status=0
ended=2026-09-26T21:19:30+08:00
```

round 1:
```text
commit=1f5e5fa559c2564249026cd7f765420b8aa90eca
started=2026-09-26T21:20:27+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_ffi_compare.py --sweep --iters 1000 --batches 9
counts=[0, 3, 5, 8, 16, 32, 64] iters=1000 batches=9 unit=us/call (host-only, no timed sync); INTJ device-bound, no specialization key
args= 0 FFI packed nop           median=0.1186 samples=[0.123625, 0.120269, 0.118493, 0.11642, 0.119044, 0.119134, 0.117404, 0.118368, 0.11858]
args= 0 FFI typed nop            median=0.1214 samples=[0.121141, 0.12579, 0.118973, 0.121426, 0.118117, 0.121812, 0.118831, 0.122921, 0.121352]
args= 0 FFI empty kernel         median=1.6972 samples=[1.57464, 1.9266800000000002, 1.6971829999999999, 1.82623, 1.578828, 1.940027, 1.645699, 1.912224, 1.65196]
args= 0 INTJ static_compile kernel median=2.7684 samples=[3.0481030000000002, 2.697519, 2.8080749999999997, 2.743849, 2.728764, 2.64043, 2.768437, 2.7809909999999998, 2.773269]
args= 0 INTJ runtime_shim kernel median=2.7305 samples=[2.777901, 2.7305279999999996, 2.8191979999999996, 2.677156, 2.733204, 2.6803310000000002, 2.704677, 2.672536, 2.767501]
args= 3 FFI packed nop           median=0.1450 samples=[0.145262, 0.14502500000000002, 0.144907, 0.146916, 0.14468899999999998, 0.146373, 0.14355500000000002, 0.14392, 0.148386]
args= 3 FFI typed nop            median=0.1492 samples=[0.148587, 0.151643, 0.15114599999999997, 0.153797, 0.147351, 0.148668, 0.14778899999999998, 0.14942599999999998, 0.14917599999999998]
args= 3 FFI empty kernel         median=3.2503 samples=[3.342698, 3.250263, 3.222025, 3.257061, 3.249374, 3.297087, 3.186929, 3.2681970000000002, 3.183472]
args= 3 INTJ static_compile kernel median=2.9143 samples=[2.931916, 2.904221, 2.992369, 2.892067, 2.9142829999999997, 2.9020349999999997, 2.975362, 2.887659, 2.9621489999999997]
args= 3 INTJ runtime_shim kernel median=2.8957 samples=[3.001094, 2.841421, 2.844654, 2.8957420000000003, 2.8879119999999996, 2.912043, 2.89575, 2.8964969999999997, 2.881725]
args= 5 FFI packed nop           median=0.1733 samples=[0.173349, 0.173058, 0.173309, 0.17627600000000002, 0.176725, 0.17108500000000001, 0.174124, 0.173041, 0.172054]
args= 5 FFI typed nop            median=0.1769 samples=[0.17462899999999998, 0.176904, 0.178647, 0.199042, 0.17677299999999999, 0.180031, 0.175644, 0.177611, 0.176066]
args= 5 FFI empty kernel         median=3.2918 samples=[3.337548, 3.385084, 3.23259, 3.283705, 3.319764, 3.28958, 3.291833, 3.280894, 3.30435]
args= 5 INTJ static_compile kernel median=2.9326 samples=[2.9624, 2.855235, 2.9427109999999996, 2.936346, 3.139893, 2.879677, 2.890313, 2.932579, 2.8914389999999996]
args= 5 INTJ runtime_shim kernel median=2.9048 samples=[3.013799, 2.9099850000000003, 2.904809, 2.9397469999999997, 2.8708739999999997, 2.8877770000000003, 2.861664, 2.9349209999999997, 2.860201]
args= 8 FFI packed nop           median=0.2060 samples=[0.203475, 0.206566, 0.20796, 0.20402099999999998, 0.20549299999999998, 0.206029, 0.206008, 0.206137, 0.207013]
args= 8 FFI typed nop            median=0.2086 samples=[0.20513800000000001, 0.211081, 0.207839, 0.20924700000000002, 0.208597, 0.20865199999999998, 0.206748, 0.210809, 0.208485]
args= 8 FFI empty kernel         median=3.4282 samples=[3.476531, 3.525205, 3.400665, 3.412885, 3.55198, 3.428209, 3.54182, 3.409284, 3.351467]
args= 8 INTJ static_compile kernel median=3.0098 samples=[3.0098409999999998, 3.015623, 3.0880199999999998, 3.000036, 2.952496, 3.006041, 3.059995, 2.979731, 3.05964]
args= 8 INTJ runtime_shim kernel median=2.8841 samples=[3.048497, 3.009215, 2.980693, 2.8839989999999998, 2.9453449999999997, 2.877241, 2.884147, 2.877734, 2.87488]
args=16 FFI packed nop           median=0.2932 samples=[0.293131, 0.294499, 0.294662, 0.292151, 0.289852, 0.290274, 0.295442, 0.293205, 0.29591300000000004]
args=16 FFI typed nop            median=0.3046 samples=[0.303439, 0.307986, 0.304617, 0.30678300000000003, 0.30370800000000003, 0.30168599999999995, 0.30174900000000004, 0.308376, 0.310336]
args=16 FFI empty kernel         median=3.6540 samples=[3.660726, 3.783232, 3.542529, 3.680886, 3.519184, 3.668788, 3.521638, 3.654049, 3.51505]
args=16 INTJ static_compile kernel median=3.4768 samples=[3.258904, 3.4768339999999998, 3.485208, 3.3251720000000002, 3.545858, 3.340392, 3.493094, 3.29025, 3.513404]
args=16 INTJ runtime_shim kernel median=3.2659 samples=[3.2658620000000003, 3.2019729999999997, 3.36654, 3.0470230000000003, 3.371904, 3.031538, 3.311951, 3.038924, 3.34312]
args=32 FFI packed nop           median=0.4805 samples=[0.478143, 0.485072, 0.48946, 0.464647, 0.476547, 0.474323, 0.480483, 0.48923, 0.48928699999999997]
args=32 FFI typed nop            median=0.4979 samples=[0.497462, 0.499461, 0.501559, 0.494954, 0.47901, 0.49099, 0.497935, 0.501503, 0.499459]
args=32 FFI empty kernel         median=4.2249 samples=[4.265627, 4.295078999999999, 4.106806, 4.269787999999999, 4.101234000000001, 4.256729999999999, 4.088748, 4.2248850000000004, 4.11191]
args=32 INTJ static_compile kernel median=3.6522 samples=[3.482633, 3.695851, 3.693244, 3.600144, 3.6643090000000003, 3.6007179999999996, 3.65795, 3.544989, 3.652242]
args=32 INTJ runtime_shim kernel median=3.5304 samples=[3.425656, 3.5303850000000003, 3.624292, 3.433179, 3.6033809999999997, 3.4340889999999997, 3.5919920000000003, 3.4249169999999998, 3.597632]
args=64 FFI packed nop           median=0.8387 samples=[0.830249, 0.838679, 0.8727010000000001, 0.8179460000000001, 0.853562, 0.829711, 0.823994, 0.842471, 0.875305]
args=64 FFI typed nop            median=0.8722 samples=[0.874049, 0.872244, 0.88906, 0.8567440000000001, 0.863456, 0.862264, 0.845214, 0.885758, 0.8890009999999999]
args=64 FFI empty kernel         median=5.0544 samples=[5.0637550000000005, 5.242778, 5.051821, 5.044968, 5.058773, 5.054406, 5.035006999999999, 5.057511000000001, 5.035033]
args=64 INTJ static_compile kernel median=4.5815 samples=[4.878171999999999, 4.687741, 4.592372, 4.576681, 4.58147, 4.579413, 4.55566, 4.538683, 4.581669]
args=64 INTJ runtime_shim kernel median=4.5747 samples=[4.7781, 4.679436, 4.564843, 4.574726, 4.506795, 4.602428, 4.501214, 4.585377, 4.562858]
elapsed_ns=4277157161
exit_status=0
ended=2026-09-26T21:20:31+08:00
```

round 2:
```text
commit=1f5e5fa559c2564249026cd7f765420b8aa90eca
started=2026-09-26T21:21:20+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_ffi_compare.py --sweep --iters 1000 --batches 9
counts=[0, 3, 5, 8, 16, 32, 64] iters=1000 batches=9 unit=us/call (host-only, no timed sync); INTJ device-bound, no specialization key
args= 0 FFI packed nop           median=0.1172 samples=[0.122376, 0.117423, 0.11886100000000001, 0.116312, 0.117191, 0.116193, 0.11812399999999999, 0.11615900000000001, 0.116423]
args= 0 FFI typed nop            median=0.1204 samples=[0.119613, 0.122021, 0.118536, 0.12176000000000001, 0.12042499999999999, 0.122254, 0.11971, 0.12197799999999999, 0.119358]
args= 0 FFI empty kernel         median=1.6464 samples=[1.62714, 1.8485740000000002, 1.616072, 1.982741, 1.646383, 1.937306, 1.601512, 1.788682, 1.606997]
args= 0 INTJ static_compile kernel median=2.7531 samples=[3.059628, 2.9127530000000004, 2.7530590000000004, 2.732373, 2.7946880000000003, 2.718314, 2.708775, 2.7432559999999997, 2.75359]
args= 0 INTJ runtime_shim kernel median=2.7062 samples=[2.862984, 2.706188, 2.760455, 2.650855, 2.759556, 2.679919, 2.704739, 2.69321, 2.759649]
args= 3 FFI packed nop           median=0.1452 samples=[0.152255, 0.14525, 0.145178, 0.143581, 0.14486500000000002, 0.14418799999999998, 0.143957, 0.147481, 0.14604599999999998]
args= 3 FFI typed nop            median=0.1491 samples=[0.147917, 0.149119, 0.149493, 0.148368, 0.148374, 0.152511, 0.15014, 0.149234, 0.148564]
args= 3 FFI empty kernel         median=3.2608 samples=[3.296734, 3.2867330000000003, 3.260843, 3.315743, 3.167535, 3.328147, 3.161987, 3.24885, 3.214216]
args= 3 INTJ static_compile kernel median=2.9324 samples=[2.966098, 2.931399, 2.887406, 2.888798, 3.0093769999999997, 2.980632, 3.056564, 2.92923, 2.9323560000000004]
args= 3 INTJ runtime_shim kernel median=2.9158 samples=[2.9627939999999997, 2.761264, 2.901683, 2.7854050000000004, 2.9596729999999996, 2.818723, 2.9258319999999998, 2.933135, 2.915763]
args= 5 FFI packed nop           median=0.1740 samples=[0.173894, 0.1754, 0.174653, 0.174111, 0.175989, 0.172141, 0.173474, 0.169034, 0.174]
args= 5 FFI typed nop            median=0.1787 samples=[0.17867, 0.178929, 0.179948, 0.19326400000000002, 0.177195, 0.17578200000000002, 0.175013, 0.179824, 0.17715]
args= 5 FFI empty kernel         median=3.2774 samples=[3.333805, 3.2526509999999997, 3.268982, 3.277444, 3.1993110000000002, 3.3176819999999996, 3.313934, 3.2688040000000003, 3.284627]
args= 5 INTJ static_compile kernel median=2.9033 samples=[2.9685639999999998, 2.893069, 2.89631, 2.90328, 2.894415, 2.9220740000000003, 2.933646, 2.894109, 2.934641]
args= 5 INTJ runtime_shim kernel median=2.9159 samples=[3.0096350000000003, 2.927075, 2.862176, 2.914027, 2.925805, 2.931271, 2.8838220000000003, 2.915855, 2.884]
args= 8 FFI packed nop           median=0.2060 samples=[0.207401, 0.206009, 0.20744900000000002, 0.205459, 0.20587200000000003, 0.20458600000000002, 0.20643899999999998, 0.20338499999999998, 0.2089]
args= 8 FFI typed nop            median=0.2101 samples=[0.20847, 0.21119, 0.211342, 0.211945, 0.207749, 0.209195, 0.21008600000000002, 0.214285, 0.208563]
args= 8 FFI empty kernel         median=3.3830 samples=[3.498628, 3.383022, 3.290245, 3.4049270000000003, 3.3553870000000003, 3.417394, 3.308012, 3.4058159999999997, 3.333031]
args= 8 INTJ static_compile kernel median=3.0520 samples=[3.0221430000000002, 2.943151, 3.068702, 2.964188, 3.096353, 3.0520859999999996, 3.086019, 3.052047, 2.9462689999999996]
args= 8 INTJ runtime_shim kernel median=2.9949 samples=[3.0324340000000003, 2.8817220000000003, 2.997578, 2.914268, 2.9948609999999998, 2.918683, 3.0107220000000003, 2.9193029999999998, 3.119332]
args=16 FFI packed nop           median=0.2923 samples=[0.291774, 0.298842, 0.296401, 0.28646, 0.291959, 0.289041, 0.295614, 0.293606, 0.292341]
args=16 FFI typed nop            median=0.3021 samples=[0.30471499999999996, 0.309479, 0.301754, 0.29696300000000003, 0.29897399999999996, 0.297196, 0.305234, 0.304114, 0.302137]
args=16 FFI empty kernel         median=3.6440 samples=[3.66408, 3.7185859999999997, 3.537104, 3.643982, 3.540693, 4.708009, 3.569351, 3.6759079999999997, 3.5649499999999996]
args=16 INTJ static_compile kernel median=3.3919 samples=[3.236978, 3.391856, 3.593415, 3.288191, 3.495323, 3.3082800000000003, 3.548593, 3.353438, 3.5582800000000003]
args=16 INTJ runtime_shim kernel median=3.2754 samples=[3.2754090000000002, 3.0732269999999997, 3.372865, 3.053163, 3.335134, 3.043884, 3.391218, 3.062207, 3.3813470000000003]
args=32 FFI packed nop           median=0.4816 samples=[0.48233800000000004, 0.47363, 0.484157, 0.478362, 0.476505, 0.481576, 0.48301, 0.483209, 0.472912]
args=32 FFI typed nop            median=0.4946 samples=[0.497848, 0.498691, 0.489428, 0.49917700000000004, 0.495114, 0.49013, 0.485192, 0.494614, 0.490438]
args=32 FFI empty kernel         median=4.2123 samples=[4.276539, 4.212281, 4.129722, 4.293411, 4.139639, 4.292355, 4.167218, 4.307448, 4.144767]
args=32 INTJ static_compile kernel median=3.6084 samples=[3.448887, 3.6084050000000003, 3.662445, 3.597076, 3.6320680000000003, 3.564137, 3.6867379999999996, 3.606403, 3.685993]
args=32 INTJ runtime_shim kernel median=3.4612 samples=[3.4080839999999997, 3.407655, 3.6285, 3.4449720000000004, 3.6010549999999997, 3.434779, 3.65256, 3.461181, 3.630087]
args=64 FFI packed nop           median=0.8389 samples=[0.842917, 0.8477210000000001, 0.838097, 0.8284539999999999, 0.843446, 0.830032, 0.8389030000000001, 0.843889, 0.819169]
args=64 FFI typed nop            median=0.8512 samples=[0.848565, 0.8804690000000001, 0.860349, 0.85122, 0.848187, 0.853302, 0.849414, 0.863814, 0.835938]
args=64 FFI empty kernel         median=5.0785 samples=[5.071902, 5.040299, 5.090549, 5.078519, 5.082175, 5.060535, 5.084505, 5.089529000000001, 5.063427]
args=64 INTJ static_compile kernel median=4.5890 samples=[4.580527, 4.553661, 4.609329, 4.578263, 4.585979999999999, 4.589019, 4.620599, 4.637627999999999, 4.600871]
args=64 INTJ runtime_shim kernel median=4.5474 samples=[4.562137, 4.533513, 4.561745, 4.553241, 4.545395999999999, 4.547445, 4.527075, 4.599528, 4.5255600000000005]
elapsed_ns=4210363269
exit_status=0
ended=2026-09-26T21:21:24+08:00
```

### ffi_paths

round 0:
```text
commit=1f5e5fa559c2564249026cd7f765420b8aa90eca
started=2026-09-26T21:19:30+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_intj_ffi_paths.py --mode all --iters 1000 --batches 9
iters=1000 batches=9 unit=ns/call; no_gpu=True
mode=callback
callback_count=9100 (100 warmup + 9000 unique misses)
INTJ cold compile callback + cache  median_ns=466.5 samples_ns=[551.354, 555.243, 424.908, 604.479, 421.171, 410.697, 448.7, 466.505, 774.897]
INTJ hot call (no callback)         median_ns=38.1 samples_ns=[40.006, 38.663, 37.72, 40.932, 38.502, 38.143, 37.837, 37.918, 37.473]
INTJ 3-tensor host-only nop         median_ns=33.4 samples_ns=[41.324, 33.588, 33.371, 33.455, 33.384, 33.478, 33.416, 33.415, 33.416]
mode=kwargs
INTJ direct positional              median_ns=64.7 samples_ns=[67.159, 64.881, 72.345, 64.644, 64.69, 64.673, 64.613, 66.701, 64.695]
INTJ adapter positional             median_ns=90.4 samples_ns=[88.627, 90.484, 88.793, 88.37, 93.311, 95.632, 90.431, 90.694, 90.432]
INTJ adapter kwargs                 median_ns=106.1 samples_ns=[107.865, 106.191, 106.026, 105.734, 105.366, 105.415, 110.111, 107.151, 106.126]
INTJ adapter defaults               median_ns=88.8 samples_ns=[88.43, 89.258, 89.429, 91.212, 88.745, 87.09, 88.771, 91.203, 86.995]
INTJ FFI wrapper positional         median_ns=102.3 samples_ns=[102.969, 101.806, 101.646, 102.28, 103.691, 103.136, 101.972, 101.718, 105.949]
INTJ FFI wrapper kwargs             median_ns=125.2 samples_ns=[125.209, 125.49, 125.637, 124.602, 124.032, 124.094, 125.795, 128.89, 125.207]
INTJ native kwargs: unsupported (verified TypeError)
mode=dataclass
INTJ pair direct                    median_ns=65.3 samples_ns=[74.375, 65.67, 65.273, 65.293, 65.256, 65.27, 72.242, 65.19, 65.258]
INTJ pair prebuilt *tuple           median_ns=136.5 samples_ns=[136.765, 136.908, 135.164, 136.507, 136.563, 137.957, 135.494, 134.024, 134.096]
FFI unpack Pair only                median_ns=161.9 samples_ns=[166.063, 165.293, 164.488, 164.539, 161.926, 160.698, 160.775, 161.083, 160.553]
INTJ pair manual unpack             median_ns=167.2 samples_ns=[181.267, 172.262, 166.404, 165.764, 165.938, 166.244, 170.353, 167.183, 167.391]
INTJ pair FFI unpack                median_ns=307.6 samples_ns=[305.25, 307.625, 305.418, 304.092, 312.49, 308.383, 306.592, 308.834, 315.933]
INTJ pair stdlib astuple            median_ns=1189.5 samples_ns=[1345.983, 1190.415, 1189.506, 1188.145, 1183.261, 1188.132, 1189.65, 1199.696, 1175.005]
INTJ config direct                  median_ns=79.5 samples_ns=[79.535, 80.127, 80.66, 78.924, 80.292, 79.276, 79.024, 79.066, 81.912]
INTJ config FFI unpack              median_ns=318.8 samples_ns=[320.881, 319.433, 324.103, 318.178, 318.124, 318.763, 321.164, 316.018, 315.461]
INTJ native dataclass unpack: unsupported (verified TypeError)
elapsed_ns=8654864335
exit_status=0
ended=2026-09-26T21:19:39+08:00
```

round 1:
```text
commit=1f5e5fa559c2564249026cd7f765420b8aa90eca
started=2026-09-26T21:20:31+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_intj_ffi_paths.py --mode all --iters 1000 --batches 9
iters=1000 batches=9 unit=ns/call; no_gpu=True
mode=callback
callback_count=9100 (100 warmup + 9000 unique misses)
INTJ cold compile callback + cache  median_ns=428.5 samples_ns=[465.676, 454.084, 400.141, 563.641, 386.851, 413.021, 414.647, 428.505, 1748.138]
INTJ hot call (no callback)         median_ns=37.2 samples_ns=[42.39, 39.203, 36.535, 37.349, 36.721, 37.393, 36.424, 37.192, 36.645]
INTJ 3-tensor host-only nop         median_ns=33.8 samples_ns=[36.189, 34.253, 33.631, 33.753, 33.437, 33.909, 33.497, 35.204, 33.674]
mode=kwargs
INTJ direct positional              median_ns=65.2 samples_ns=[67.047, 64.771, 69.585, 65.231, 65.288, 65.135, 65.241, 65.213, 65.229]
INTJ adapter positional             median_ns=90.6 samples_ns=[91.101, 91.138, 89.098, 89.5, 90.633, 90.41, 93.61, 90.389, 90.858]
INTJ adapter kwargs                 median_ns=108.2 samples_ns=[108.774, 108.425, 108.485, 106.476, 106.986, 107.299, 109.298, 108.172, 107.758]
INTJ adapter defaults               median_ns=89.8 samples_ns=[89.646, 90.32, 89.996, 89.812, 89.181, 89.756, 89.981, 89.226, 96.296]
INTJ FFI wrapper positional         median_ns=104.6 samples_ns=[106.576, 104.775, 104.06, 105.179, 104.567, 102.811, 104.591, 102.424, 107.174]
INTJ FFI wrapper kwargs             median_ns=126.7 samples_ns=[127.872, 126.692, 127.291, 125.254, 124.49, 125.535, 124.997, 136.65, 130.0]
INTJ native kwargs: unsupported (verified TypeError)
mode=dataclass
INTJ pair direct                    median_ns=65.6 samples_ns=[67.88, 66.289, 65.099, 65.848, 64.961, 65.601, 64.854, 69.599, 65.499]
INTJ pair prebuilt *tuple           median_ns=138.3 samples_ns=[139.47, 137.582, 134.422, 139.162, 134.929, 138.347, 146.62, 139.128, 135.823]
FFI unpack Pair only                median_ns=163.3 samples_ns=[165.31, 165.837, 163.665, 168.304, 163.295, 162.989, 163.08, 163.115, 162.882]
INTJ pair manual unpack             median_ns=171.6 samples_ns=[172.252, 172.089, 169.812, 172.324, 168.897, 171.628, 172.118, 169.804, 167.262]
INTJ pair FFI unpack                median_ns=311.3 samples_ns=[306.595, 323.794, 307.147, 311.566, 315.019, 312.518, 307.591, 311.328, 311.172]
INTJ pair stdlib astuple            median_ns=1158.1 samples_ns=[1324.032, 1207.093, 1165.036, 1167.642, 1157.762, 1156.297, 1139.601, 1158.064, 1146.974]
INTJ config direct                  median_ns=79.1 samples_ns=[79.076, 79.779, 79.028, 79.292, 78.925, 79.364, 78.922, 79.313, 79.061]
INTJ config FFI unpack              median_ns=321.1 samples_ns=[320.587, 321.793, 321.054, 326.964, 320.338, 323.38, 324.961, 320.925, 318.247]
INTJ native dataclass unpack: unsupported (verified TypeError)
elapsed_ns=2788054392
exit_status=0
ended=2026-09-26T21:20:34+08:00
```

round 2:
```text
commit=1f5e5fa559c2564249026cd7f765420b8aa90eca
started=2026-09-26T21:21:24+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_intj_ffi_paths.py --mode all --iters 1000 --batches 9
iters=1000 batches=9 unit=ns/call; no_gpu=True
mode=callback
callback_count=9100 (100 warmup + 9000 unique misses)
INTJ cold compile callback + cache  median_ns=429.9 samples_ns=[458.041, 458.825, 401.565, 562.581, 385.779, 413.122, 411.845, 429.872, 1444.268]
INTJ hot call (no callback)         median_ns=37.2 samples_ns=[39.755, 39.152, 36.688, 37.737, 36.433, 37.598, 36.81, 37.164, 36.124]
INTJ 3-tensor host-only nop         median_ns=33.8 samples_ns=[35.832, 34.231, 33.784, 33.897, 33.596, 33.908, 33.571, 33.846, 33.596]
mode=kwargs
INTJ direct positional              median_ns=64.6 samples_ns=[67.213, 64.883, 64.57, 64.443, 64.414, 64.499, 64.573, 65.026, 64.508]
INTJ adapter positional             median_ns=93.8 samples_ns=[90.845, 113.951, 95.08, 93.841, 93.846, 94.13, 93.858, 92.499, 93.563]
INTJ adapter kwargs                 median_ns=110.3 samples_ns=[110.308, 111.383, 123.838, 111.973, 111.088, 108.556, 107.711, 107.446, 108.375]
INTJ adapter defaults               median_ns=89.2 samples_ns=[90.054, 89.154, 91.883, 89.902, 89.172, 89.911, 89.001, 87.676, 88.03]
INTJ FFI wrapper positional         median_ns=103.4 samples_ns=[104.76, 102.472, 102.469, 106.74, 103.759, 102.629, 108.938, 103.417, 101.974]
INTJ FFI wrapper kwargs             median_ns=127.8 samples_ns=[127.512, 125.481, 126.928, 134.102, 131.545, 131.148, 128.638, 127.51, 127.76]
INTJ native kwargs: unsupported (verified TypeError)
mode=dataclass
INTJ pair direct                    median_ns=65.8 samples_ns=[73.077, 67.091, 65.154, 66.733, 64.951, 66.499, 64.802, 65.773, 64.766]
INTJ pair prebuilt *tuple           median_ns=137.4 samples_ns=[136.01, 137.831, 146.36, 139.062, 135.962, 137.374, 135.428, 137.49, 134.028]
FFI unpack Pair only                median_ns=162.7 samples_ns=[165.681, 163.516, 161.339, 164.035, 161.516, 162.726, 164.535, 161.665, 162.362]
INTJ pair manual unpack             median_ns=173.4 samples_ns=[173.402, 175.663, 171.438, 177.901, 170.929, 176.633, 172.174, 175.905, 171.093]
INTJ pair FFI unpack                median_ns=309.1 samples_ns=[316.277, 307.497, 304.323, 311.376, 308.871, 310.778, 309.655, 309.114, 305.35]
INTJ pair stdlib astuple            median_ns=1168.1 samples_ns=[1329.332, 1222.352, 1156.58, 1168.079, 1165.556, 1172.594, 1162.004, 1178.338, 1163.186]
INTJ config direct                  median_ns=78.6 samples_ns=[78.504, 78.629, 77.95, 78.883, 80.765, 79.011, 78.545, 78.767, 78.275]
INTJ config FFI unpack              median_ns=322.9 samples_ns=[319.555, 323.73, 321.67, 322.862, 321.216, 328.537, 319.756, 323.26, 323.561]
INTJ native dataclass unpack: unsupported (verified TypeError)
elapsed_ns=2784017257
exit_status=0
ended=2026-09-26T21:21:27+08:00
```

### hip_module

round 0:
```text
commit=1f5e5fa559c2564249026cd7f765420b8aa90eca
started=2026-09-26T21:19:39+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_hip_module_launch.py --iters 1000 --batches 9
hsaco_sha256=df3e529e178196ad62f7c07ab8593ede5b8596c76e7c6fbd12a5d947a696e65c kernel=empty_kernel threads=256 shared=0 callback_count=1
iters=1000 batches=9 unit=us/call (host enqueue, no timed sync)
TVM FFI HIP HSACO         median=3.2595 samples=[3.672933, 3.3429279999999997, 3.285406, 3.25948, 3.259665, 3.1165920000000003, 3.1046129999999996, 3.0789609999999996, 3.057799]
Triton same HSACO         median=15.9169 samples=[16.129359, 16.064092000000002, 15.983781, 16.021138, 15.916856, 15.801974, 15.816312, 15.754024, 15.754707]
INTJ same function        median=3.0240 samples=[3.117192, 3.124596, 3.066898, 3.068672, 3.023956, 2.9025090000000002, 2.901581, 2.897901, 2.8580929999999998]
elapsed_ns=5666331362
exit_status=0
ended=2026-09-26T21:19:45+08:00
```

round 1:
```text
commit=1f5e5fa559c2564249026cd7f765420b8aa90eca
started=2026-09-26T21:20:34+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_hip_module_launch.py --iters 1000 --batches 9
hsaco_sha256=df3e529e178196ad62f7c07ab8593ede5b8596c76e7c6fbd12a5d947a696e65c kernel=empty_kernel threads=256 shared=0 callback_count=1
iters=1000 batches=9 unit=us/call (host enqueue, no timed sync)
TVM FFI HIP HSACO         median=3.1267 samples=[3.487782, 3.210068, 3.185433, 3.165504, 3.126727, 3.0221750000000003, 3.090585, 3.0213400000000004, 3.0305619999999998]
Triton same HSACO         median=15.9539 samples=[16.180252, 16.02875, 16.031581, 16.028190000000002, 15.91741, 15.804421, 15.851712, 15.766566000000001, 15.953895000000001]
INTJ same function        median=2.9165 samples=[3.0220949999999998, 3.02054, 3.011491, 3.000301, 2.916533, 2.8272060000000003, 2.843187, 2.838487, 2.811601]
elapsed_ns=3609364260
exit_status=0
ended=2026-09-26T21:20:38+08:00
```

round 2:
```text
commit=1f5e5fa559c2564249026cd7f765420b8aa90eca
started=2026-09-26T21:21:27+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_hip_module_launch.py --iters 1000 --batches 9
hsaco_sha256=df3e529e178196ad62f7c07ab8593ede5b8596c76e7c6fbd12a5d947a696e65c kernel=empty_kernel threads=256 shared=0 callback_count=1
iters=1000 batches=9 unit=us/call (host enqueue, no timed sync)
TVM FFI HIP HSACO         median=3.1473 samples=[3.321222, 3.1939499999999996, 3.153295, 3.1473180000000003, 3.1531640000000003, 3.1213420000000003, 3.1182779999999997, 3.028421, 3.048339]
Triton same HSACO         median=16.3194 samples=[16.4375, 16.385271, 16.346363, 16.37906, 16.319391, 16.092290000000002, 16.151743, 16.147212, 16.102171000000002]
INTJ same function        median=2.9336 samples=[2.983436, 2.9899769999999997, 2.9959540000000002, 2.984427, 2.922925, 2.924625, 2.933575, 2.927137, 2.846583]
elapsed_ns=3634815082
exit_status=0
ended=2026-09-26T21:21:30+08:00
```

### kernel_cache

round 0:
```text
commit=1f5e5fa559c2564249026cd7f765420b8aa90eca
started=2026-09-26T21:19:45+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python -m pytest tests/test_kernel_cache.py -s -q

kernel cache, 1-word key, ns per operation

                      intj       tsl      absl
hash_only             1.38      1.36      1.36
hit/1                 1.20      1.85      1.35
hit/8                 1.25      1.88      4.23
hit/64                1.30      1.91      4.36
hit/512               1.49      2.62      4.94
miss/1                1.01      1.56      1.38
miss/8                2.22      1.74      3.23
miss/64               2.12      1.76      3.80
miss/512              2.23      2.00      3.94

kernel cache, 2-word key, ns per operation

                      intj       tsl      absl
hash_only             1.43      1.42      1.41
hit/1                 1.89      2.28      1.70
hit/8                 1.89      2.26      4.78
hit/64                2.13      2.57      5.25
hit/512               2.73      3.46      5.83
miss/1                1.32      2.16      1.61
miss/8                2.76      1.70      3.27
miss/64               3.08      2.01      3.36
miss/512              2.96      2.30      3.95

kernel cache, 5-word key, ns per operation

                      intj       tsl      absl
hash_only             3.57      3.57      3.56
hit/1                 2.94      3.19      2.68
hit/8                 3.17      3.54      9.50
hit/64                3.25      3.65      9.73
hit/512               4.46      4.94     11.17
miss/1                1.87      2.27      1.64
miss/8                3.62      3.03      6.93
miss/64               3.42      2.54      6.72
miss/512              4.11      2.78      7.66
.
1 passed in 21.79s
elapsed_ns=22029595928
exit_status=0
ended=2026-09-26T21:20:07+08:00
```

round 1:
```text
commit=1f5e5fa559c2564249026cd7f765420b8aa90eca
started=2026-09-26T21:20:38+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python -m pytest tests/test_kernel_cache.py -s -q

kernel cache, 1-word key, ns per operation

                      intj       tsl      absl
hash_only             1.36      1.36      1.36
hit/1                 1.20      1.85      1.33
hit/8                 1.27      1.88      4.23
hit/64                1.31      1.91      4.34
hit/512               1.48      2.60      4.93
miss/1                1.03      1.46      1.38
miss/8                2.19      1.76      3.25
miss/64               2.15      1.73      3.81
miss/512              2.22      1.97      3.94

kernel cache, 2-word key, ns per operation

                      intj       tsl      absl
hash_only             1.42      1.42      1.42
hit/1                 1.90      2.26      1.71
hit/8                 1.90      2.28      4.76
hit/64                2.10      2.58      5.24
hit/512               2.73      3.46      5.79
miss/1                1.30      2.16      1.47
miss/8                2.71      1.75      3.27
miss/64               3.13      2.02      3.36
miss/512              2.94      2.30      3.90

kernel cache, 5-word key, ns per operation

                      intj       tsl      absl
hash_only             3.59      3.58      3.57
hit/1                 2.94      3.19      2.68
hit/8                 3.17      3.54      9.46
hit/64                3.25      3.64      9.75
hit/512               4.47      4.94     11.21
miss/1                1.83      2.29      1.66
miss/8                3.57      3.25      6.93
miss/64               3.37      2.58      6.71
miss/512              4.08      2.76      7.66
.
1 passed in 21.76s
elapsed_ns=21997304763
exit_status=0
ended=2026-09-26T21:21:00+08:00
```

round 2:
```text
commit=1f5e5fa559c2564249026cd7f765420b8aa90eca
started=2026-09-26T21:21:30+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python -m pytest tests/test_kernel_cache.py -s -q

kernel cache, 1-word key, ns per operation

                      intj       tsl      absl
hash_only             1.36      1.36      1.36
hit/1                 1.21      1.84      1.36
hit/8                 1.26      1.88      4.24
hit/64                1.31      1.91      4.35
hit/512               1.50      2.65      4.95
miss/1                1.03      1.43      1.39
miss/8                2.22      1.75      3.26
miss/64               2.17      1.73      3.82
miss/512              2.23      1.97      3.94

kernel cache, 2-word key, ns per operation

                      intj       tsl      absl
hash_only             1.42      1.42      1.42
hit/1                 1.89      2.26      1.70
hit/8                 1.89      2.27      4.76
hit/64                2.11      2.59      5.27
hit/512               2.73      3.47      5.85
miss/1                1.32      2.17      1.54
miss/8                2.45      1.77      3.26
miss/64               3.07      2.00      3.36
miss/512              2.95      2.30      3.87

kernel cache, 5-word key, ns per operation

                      intj       tsl      absl
hash_only             3.58      3.59      3.58
hit/1                 2.94      3.19      2.68
hit/8                 3.17      3.54      9.47
hit/64                3.25      3.64      9.76
hit/512               4.47      4.94     11.18
miss/1                1.85      2.27      1.67
miss/8                3.55      3.09      6.94
miss/64               3.38      2.55      6.70
miss/512              4.10      2.70      7.65
.
1 passed in 21.76s
elapsed_ns=21992549467
exit_status=0
ended=2026-09-26T21:21:52+08:00
```

# Launcher benchmark baseline before lazy launchers (Task 0)

Baseline for the lazy, per-call-variant `make_launcher` plan
(`docs/superpowers/plans/2026-09-27-lazy-dynamic-launchers.md`), taken at the plan's
starting commit, clean checkout, no uncommitted changes.

## Environment

- Commit: `f507cfcc57a41d73f66152aff7e1b3b0562c8324` (branch `develop`), clean
  (`git status --short` empty).
- CPU: `taskset -c 0`, one benchmark process at a time. Shared host; 1-min load
  averages 3.06 (before the run) to 20.19 (during, other users) -- logged per
  process in the raw files.
- GPU: AMD Instinct MI308X (`gfx942`), GPU 0, idle otherwise.
- Python 3.12.3 (`/tmp/gb2/bin/python`), Torch `2.14.0+rocm7.2`, Triton `3.8.0`,
  g++ 13.3.0.
- `INTJ_BENCHMARK_ROOT=/tmp/gbench`.
- CUDA: untested (no NVIDIA GPU on this machine).

## Command

```bash
/tmp/intj-bench/run_all.sh lazy-baseline
```

All 10 cases x 3 rounds (30 files) under `/tmp/intj-bench/lazy-baseline/`, every
file `exit_status=0`. `/tmp/intj-bench/lazy-baseline/DONE` and `COMMIT` written.

```bash
/tmp/gb2/bin/python /tmp/intj-bench/bench_compare.py lazy-baseline
```

## Medians (baseline)

### `launch_gpu` (ns/call)

| variant | median ns |
|---|---:|
| auto map | 3143.5 |
| baked | 3052.8 |
| bound pointer | 2985.7 |
| bound tensor | 3067.2 |
| fixed device map | 2931.6 |
| fixed device no-map | 2928.6 |
| reduced key | 3162.2 |
| verify off | 3102.8 |
| verify on | 2970.6 |

### `launch_host` (`--no-gpu`, ns/call)

| variant | median ns |
|---|---:|
| auto map | 42.5 |
| baked | 38.7 |
| bound pointer | 44.6 |
| bound tensor | 46.3 |
| fixed device map | 42.6 |
| fixed device no-map | 43.7 |
| reduced key | 41.8 |
| verify off | 42.3 |
| verify on | 43.3 |

### `launch_readme`

| row | metric | intj | triton |
|---|---|---:|---:|
| grid=(0,) | us | 0.0300 | 13.0400 |
| grid=(1,) | us | 3.1300 | 16.5700 |

| readme_torch_access | decode ns |
|---|---:|
| interpreter | 875.1 |
| runtime_shim | 91.1 |
| static_compile | 89.7 |

### `launch_sweep` (`--sweep --no-gpu` implied by driver command, ns/call)

| args | int | tensor |
|---|---:|---:|
| 4 | 39.7 | 38.6 |
| 16 | 67.6 | 63.8 |
| 32 | 105.9 | 110.0 |

### `launch_last_key` (ns/call)

| args | alternate | repeat |
|---|---:|---:|
| 4 | 35.7 | 33.4 |
| 16 | 64.8 | 61.8 |
| 32 | 103.3 | 99.0 |

### `ffi_compare` (us)

| row | median |
|---|---:|
| FFI empty kernel | 3.6831 |
| FFI mixed kernel | 3.7881 |
| FFI packed nop | 0.1456 |
| FFI packed nop mixed | 0.1771 |
| FFI typed nop | 0.1496 |
| FFI typed nop mixed | 0.1794 |
| INTJ empty kernel | 3.3618 |
| INTJ fixed mixed kernel | 3.4130 |
| INTJ fixed-device kernel | 3.3525 |
| INTJ mixed kernel | 3.3529 |

### `ffi_sweep` (us, INTJ not included -- `bench_ffi_compare.py --sweep` covers FFI paths only)

| args | FFI empty kernel | FFI packed nop | FFI typed nop |
|---|---:|---:|---:|
| 0 | 1.7993 | 0.1172 | 0.1198 |
| 3 | 3.2500 | 0.1451 | 0.1500 |
| 5 | 3.2977 | 0.1736 | 0.1766 |
| 8 | 3.4058 | 0.2064 | 0.2088 |
| 16 | 3.6718 | 0.2918 | 0.3041 |
| 32 | 4.2685 | 0.4783 | 0.4944 |
| 64 | 5.0938 | 0.8441 | 0.8671 |

### `ffi_paths` (ns unless noted)

| row | median |
|---|---:|
| FFI unpack Pair only | 162.4 |
| INTJ 3-tensor host-only nop | 33.8 |
| INTJ FFI wrapper kwargs | 125.8 |
| INTJ FFI wrapper positional | 102.6 |
| INTJ adapter defaults | 89.4 |
| INTJ adapter kwargs | 105.8 |
| INTJ adapter positional | 91.4 |
| INTJ cold compile callback + cache | 404.6 |
| INTJ config FFI unpack | 318.0 |
| INTJ config direct | 80.1 |
| INTJ direct positional | 65.0 |
| INTJ hot call (no callback) | 38.1 |
| INTJ pair FFI unpack | 314.5 |
| INTJ pair direct | 67.7 |
| INTJ pair manual unpack | 171.8 |
| INTJ pair prebuilt *tuple | 136.9 |
| INTJ pair stdlib astuple | 1153.3 |

### `hip_module` (us)

| row | median |
|---|---:|
| INTJ same function | 3.4372 |
| TVM FFI HIP HSACO | 3.5929 |
| Triton same HSACO | 16.3112 |

### `kernel_cache` (ns, google/benchmark via `/tmp/gbench`)

| key size | op | absl | intj | tsl |
|---|---|---:|---:|---:|
| 1-word | hash_only | 1.36 | 1.36 | 1.36 |
| 1-word | hit/1 | 1.44 | 1.58 | 2.02 |
| 1-word | hit/8 | 4.36 | 1.66 | 2.10 |
| 1-word | hit/64 | 4.54 | 1.71 | 2.13 |
| 1-word | hit/512 | 5.32 | 2.30 | 2.90 |
| 1-word | hit_child | 3.00 | 3.01 | 3.00 |
| 1-word | miss/1 | 1.38 | 1.21 | 1.63 |
| 1-word | miss/8 | 3.28 | 2.68 | 1.76 |
| 1-word | miss/64 | 3.54 | 2.78 | 1.75 |
| 1-word | miss/512 | 4.06 | 2.94 | 2.00 |
| 2-word | hash_only | 1.42 | 1.65 | 1.42 |
| 2-word | hit/1 | 2.02 | 1.84 | 2.59 |
| 2-word | hit/8 | 5.04 | 1.83 | 2.57 |
| 2-word | hit/64 | 5.24 | 2.20 | 2.92 |
| 2-word | hit/512 | 6.39 | 3.02 | 4.04 |
| 2-word | hit_child | 3.36 | 3.00 | 3.00 |
| 2-word | miss/1 | 1.48 | 1.38 | 2.33 |
| 2-word | miss/8 | 3.20 | 2.44 | 1.86 |
| 2-word | miss/64 | 3.99 | 3.18 | 2.10 |
| 2-word | miss/512 | 3.99 | 2.87 | 2.71 |
| 5-word | hash_only | 4.25 | 3.59 | 3.59 |
| 5-word | hit/1 | 2.46 | 2.99 | 3.06 |
| 5-word | hit/8 | 9.49 | 3.20 | 3.41 |
| 5-word | hit/64 | 9.77 | 3.26 | 3.51 |
| 5-word | hit/512 | 11.16 | 4.54 | 4.77 |
| 5-word | hit_child | 3.01 | 3.01 | 3.01 |
| 5-word | miss/1 | 2.14 | 1.41 | 2.10 |
| 5-word | miss/8 | 6.93 | 3.00 | 2.24 |
| 5-word | miss/64 | 6.73 | 3.36 | 1.97 |
| 5-word | miss/512 | 7.78 | 3.50 | 2.52 |

## Timing caveats

Same caveats as `benchmarks/AGENTS.md`: host call/enqueue time only (no GPU sync
timing), FFI no-ops and INTJ launches use different kernel binaries in
`ffi_compare`/`ffi_sweep`, `readme_torch_access` interpreter row includes the
Python interpreter path, and CUDA is untested. No new caveats found for this
baseline run.

This is index 0 (baseline); later task journals compare against it via
`/tmp/gb2/bin/python /tmp/intj-bench/bench_compare.py lazy-baseline lazy-task<N>`.

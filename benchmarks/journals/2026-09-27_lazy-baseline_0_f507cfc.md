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
## Raw outputs

Local artifact path: `/tmp/intj-bench/lazy-baseline/round<0-2>_<case>.txt` (30 files), plus `DONE` and `COMMIT` in the same directory.

### launch_gpu

round 0:
```text
commit=f507cfcc57a41d73f66152aff7e1b3b0562c8324
started=2026-09-27T21:04:23+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --iters 1000 --batches 9
mode=gpu; 1000 calls × 9 batches; median ns/call
               path    ns/call       Δ ns       Δ %   build ms   bind ms
           auto map     3850.9       +0.0      +0.0     299.49         -
        reduced key     3982.8     +131.9      +3.4     145.03         -
         verify off     3709.4     -141.5      -3.7     135.24         -
          verify on     3852.9       +2.0      +0.1     131.47         -
              baked     3647.5     -203.5      -5.3     131.07         -
       bound tensor     3696.2     -154.8      -4.0       0.41    146.75
      bound pointer     3874.3      +23.4      +0.6       0.42    145.90
   fixed device map     3670.5     -180.4      -4.7       0.38    135.25
fixed device no-map     3884.5      +33.6      +0.9       0.44    130.45
elapsed_ns=5068917456
exit_status=0
ended=2026-09-27T21:04:28+08:00
```

round 1:
```text
commit=f507cfcc57a41d73f66152aff7e1b3b0562c8324
started=2026-09-27T21:05:29+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --iters 1000 --batches 9
mode=gpu; 1000 calls × 9 batches; median ns/call
               path    ns/call       Δ ns       Δ %   build ms   bind ms
           auto map     3126.7       +0.0      +0.0      74.33         -
        reduced key     3162.2      +35.5      +1.1      73.76         -
         verify off     3047.5      -79.2      -2.5       2.24         -
          verify on     2959.6     -167.2      -5.3       2.17         -
              baked     2958.9     -167.9      -5.4       2.41         -
       bound tensor     3047.4      -79.3      -2.5       0.37      2.17
      bound pointer     2970.9     -155.9      -5.0       0.36      2.13
   fixed device map     2872.8     -253.9      -8.1       0.32      2.10
fixed device no-map     2928.6     -198.2      -6.3       0.28      2.08
elapsed_ns=3802287683
exit_status=0
ended=2026-09-27T21:05:33+08:00
```

round 2:
```text
commit=f507cfcc57a41d73f66152aff7e1b3b0562c8324
started=2026-09-27T21:06:28+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --iters 1000 --batches 9
mode=gpu; 1000 calls × 9 batches; median ns/call
               path    ns/call       Δ ns       Δ %   build ms   bind ms
           auto map     3143.5       +0.0      +0.0      75.44         -
        reduced key     3128.5      -15.0      -0.5      73.85         -
         verify off     3102.8      -40.6      -1.3       2.24         -
          verify on     2970.6     -172.9      -5.5       2.21         -
              baked     3052.8      -90.7      -2.9       2.35         -
       bound tensor     3067.2      -76.3      -2.4       0.37      2.17
      bound pointer     2985.7     -157.8      -5.0       0.36      2.13
   fixed device map     2931.6     -211.8      -6.7       0.34      2.12
fixed device no-map     2922.3     -221.1      -7.0       0.28      2.09
elapsed_ns=3767936145
exit_status=0
ended=2026-09-27T21:06:32+08:00
```

### launch_host

round 0:
```text
commit=f507cfcc57a41d73f66152aff7e1b3b0562c8324
started=2026-09-27T21:04:28+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --no-gpu --iters 100000 --batches 9
mode=host-only; 100000 calls × 9 batches; median ns/call
               path    ns/call       Δ ns       Δ %   build ms   bind ms
           auto map       42.5       +0.0      +0.0     326.28         -
        reduced key       42.0       -0.5      -1.2     134.87         -
         verify off       42.3       -0.2      -0.5     156.80         -
          verify on       43.3       +0.8      +1.8     131.14         -
              baked       38.7       -3.8      -8.9     130.58         -
       bound tensor       46.3       +3.8      +9.0       0.17    156.73
      bound pointer       44.9       +2.4      +5.7       0.17    130.50
   fixed device map       43.4       +0.9      +2.0       0.14    130.38
fixed device no-map       43.8       +1.3      +2.9       0.17    129.23
elapsed_ns=4383046994
exit_status=0
ended=2026-09-27T21:04:32+08:00
```

round 1:
```text
commit=f507cfcc57a41d73f66152aff7e1b3b0562c8324
started=2026-09-27T21:05:33+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --no-gpu --iters 100000 --batches 9
mode=host-only; 100000 calls × 9 batches; median ns/call
               path    ns/call       Δ ns       Δ %   build ms   bind ms
           auto map       42.5       +0.0      +0.0      62.25         -
        reduced key       41.5       -1.0      -2.3       2.11         -
         verify off       42.9       +0.4      +1.0       1.97         -
          verify on       43.5       +1.0      +2.4       2.10         -
              baked       38.6       -3.9      -9.2       1.98         -
       bound tensor       46.5       +4.0      +9.4       0.14      1.84
      bound pointer       44.6       +2.1      +4.9       0.15      1.84
   fixed device map       42.1       -0.4      -0.9       0.11      1.81
fixed device no-map       43.2       +0.7      +1.6       0.15      1.82
elapsed_ns=3006387093
exit_status=0
ended=2026-09-27T21:05:36+08:00
```

round 2:
```text
commit=f507cfcc57a41d73f66152aff7e1b3b0562c8324
started=2026-09-27T21:06:32+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --no-gpu --iters 100000 --batches 9
mode=host-only; 100000 calls × 9 batches; median ns/call
               path    ns/call       Δ ns       Δ %   build ms   bind ms
           auto map       42.6       +0.0      +0.0      62.11         -
        reduced key       41.8       -0.8      -1.8      16.85         -
         verify off       42.2       -0.4      -1.0       2.03         -
          verify on       43.2       +0.6      +1.5       2.11         -
              baked       38.8       -3.8      -8.9       1.97         -
       bound tensor       45.3       +2.7      +6.4       0.15      1.83
      bound pointer       44.2       +1.6      +3.9       0.15      1.84
   fixed device map       42.6       +0.0      +0.1       0.11      1.77
fixed device no-map       43.7       +1.1      +2.7       0.16      1.81
elapsed_ns=3006648744
exit_status=0
ended=2026-09-27T21:06:35+08:00
```

### launch_readme

round 0:
```text
commit=f507cfcc57a41d73f66152aff7e1b3b0562c8324
started=2026-09-27T21:04:32+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --readme --iters 1000 --batches 9
mode=readme; 1000 calls × 9 batches; median
        path   triton us    intj us   speedup
   grid=(1,)       17.08       3.86      4.4x
   grid=(0,)       12.93       0.03    443.8x

torch_access_mode   decode ns    build s
     runtime_shim        91.1       0.19
   static_compile        89.7       0.14
      interpreter       870.5       0.14
elapsed_ns=4200148179
exit_status=0
ended=2026-09-27T21:04:36+08:00
```

round 1:
```text
commit=f507cfcc57a41d73f66152aff7e1b3b0562c8324
started=2026-09-27T21:05:36+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --readme --iters 1000 --batches 9
mode=readme; 1000 calls × 9 batches; median
        path   triton us    intj us   speedup
   grid=(1,)       16.57       3.13      5.3x
   grid=(0,)       13.07       0.03    464.1x

torch_access_mode   decode ns    build s
     runtime_shim        91.7       0.02
   static_compile        89.7       0.06
      interpreter       876.2       0.00
elapsed_ns=3747670057
exit_status=0
ended=2026-09-27T21:05:40+08:00
```

round 2:
```text
commit=f507cfcc57a41d73f66152aff7e1b3b0562c8324
started=2026-09-27T21:06:35+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --readme --iters 1000 --batches 9
mode=readme; 1000 calls × 9 batches; median
        path   triton us    intj us   speedup
   grid=(1,)       16.34       3.12      5.2x
   grid=(0,)       13.04       0.03    460.9x

torch_access_mode   decode ns    build s
     runtime_shim        89.7       0.02
   static_compile        90.9       0.06
      interpreter       875.1       0.00
elapsed_ns=3791050896
exit_status=0
ended=2026-09-27T21:06:39+08:00
```

### launch_sweep

round 0:
```text
commit=f507cfcc57a41d73f66152aff7e1b3b0562c8324
started=2026-09-27T21:04:36+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --sweep --iters 100000 --batches 9
mode=sweep; host-only; 100000 calls × 9 batches; median ns/call
count   kind    ns/call
    4    int       40.2
    4 tensor       38.6
   16    int       67.8
   16 tensor       63.8
   32    int      105.9
   32 tensor      110.0
elapsed_ns=3571004639
exit_status=0
ended=2026-09-27T21:04:40+08:00
```

round 1:
```text
commit=f507cfcc57a41d73f66152aff7e1b3b0562c8324
started=2026-09-27T21:05:40+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --sweep --iters 100000 --batches 9
mode=sweep; host-only; 100000 calls × 9 batches; median ns/call
count   kind    ns/call
    4    int       39.7
    4 tensor       38.5
   16    int       67.6
   16 tensor       63.8
   32    int      104.8
   32 tensor      112.5
elapsed_ns=3068588911
exit_status=0
ended=2026-09-27T21:05:43+08:00
```

round 2:
```text
commit=f507cfcc57a41d73f66152aff7e1b3b0562c8324
started=2026-09-27T21:06:39+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --sweep --iters 100000 --batches 9
mode=sweep; host-only; 100000 calls × 9 batches; median ns/call
count   kind    ns/call
    4    int       39.6
    4 tensor       38.6
   16    int       67.6
   16 tensor       63.8
   32    int      106.2
   32 tensor      107.1
elapsed_ns=3075824714
exit_status=0
ended=2026-09-27T21:06:42+08:00
```

### launch_last_key

round 0:
```text
commit=f507cfcc57a41d73f66152aff7e1b3b0562c8324
started=2026-09-27T21:04:40+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --last-key --iters 100000 --batches 9
mode=last-key; host-only; 200000 calls × 9 batches; median ns/call
count     pattern    ns/call
    4      repeat       33.6
    4   alternate       35.7
   16      repeat       61.8
   16   alternate       65.5
   32      repeat       99.1
   32   alternate      103.3
elapsed_ns=3378221217
exit_status=0
ended=2026-09-27T21:04:43+08:00
```

round 1:
```text
commit=f507cfcc57a41d73f66152aff7e1b3b0562c8324
started=2026-09-27T21:05:43+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --last-key --iters 100000 --batches 9
mode=last-key; host-only; 200000 calls × 9 batches; median ns/call
count     pattern    ns/call
    4      repeat       33.4
    4   alternate       35.7
   16      repeat       61.8
   16   alternate       64.6
   32      repeat       99.0
   32   alternate      103.2
elapsed_ns=3389877970
exit_status=0
ended=2026-09-27T21:05:46+08:00
```

round 2:
```text
commit=f507cfcc57a41d73f66152aff7e1b3b0562c8324
started=2026-09-27T21:06:42+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --last-key --iters 100000 --batches 9
mode=last-key; host-only; 200000 calls × 9 batches; median ns/call
count     pattern    ns/call
    4      repeat       33.4
    4   alternate       35.6
   16      repeat       61.8
   16   alternate       64.8
   32      repeat       98.5
   32   alternate      103.4
elapsed_ns=3459669909
exit_status=0
ended=2026-09-27T21:06:45+08:00
```

### ffi_compare

round 0:
```text
commit=f507cfcc57a41d73f66152aff7e1b3b0562c8324
started=2026-09-27T21:04:43+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_ffi_compare.py --iters 1000 --batches 9
iters=1000 batches=9 unit=us/call (host-only, no timed sync)
FFI packed nop             median=0.1456 samples=[0.153378, 0.144033, 0.145261, 0.14662899999999998, 0.143321, 0.145991, 0.14564, 0.14660499999999999, 0.144733]
FFI typed nop              median=0.1503 samples=[0.15021600000000002, 0.15373599999999998, 0.148626, 0.150372, 0.150309, 0.15212299999999998, 0.147185, 0.151345, 0.148357]
FFI empty kernel           median=3.8066 samples=[3.92009, 3.9006089999999998, 3.782711, 3.806603, 3.5956289999999997, 3.829679, 3.621229, 3.811933, 3.6646439999999996]
INTJ empty kernel          median=3.3618 samples=[3.189364, 3.520251, 3.427654, 3.361753, 3.2689079999999997, 3.405282, 3.2799490000000002, 3.3916570000000004, 3.309168]
INTJ fixed-device kernel   median=3.3657 samples=[3.256008, 3.370167, 3.5371819999999996, 3.360447, 3.3657, 3.138803, 3.3799740000000003, 3.19881, 3.3795970000000004]
FFI packed nop mixed       median=0.1771 samples=[0.18073699999999998, 0.17690899999999998, 0.178019, 0.175944, 0.176874, 0.178487, 0.183749, 0.174498, 0.17711600000000002]
FFI typed nop mixed        median=0.1794 samples=[0.178593, 0.18129599999999998, 0.1794, 0.178896, 0.179538, 0.180063, 0.18194300000000002, 0.179221, 0.179368]
FFI mixed kernel           median=3.7881 samples=[3.788147, 4.040861, 3.715781, 3.883998, 3.622404, 3.863774, 3.698059, 3.878024, 3.6991840000000002]
INTJ mixed kernel          median=3.3529 samples=[3.489275, 3.512491, 3.342457, 3.412442, 3.285483, 3.352112, 3.352938, 3.460622, 3.330082]
INTJ fixed mixed kernel    median=3.4130 samples=[3.530809, 3.616873, 3.4394549999999997, 3.455985, 3.330993, 3.335841, 3.3313960000000002, 3.412997, 3.3541529999999997]
elapsed_ns=4437660656
exit_status=0
ended=2026-09-27T21:04:48+08:00
```

round 1:
```text
commit=f507cfcc57a41d73f66152aff7e1b3b0562c8324
started=2026-09-27T21:05:46+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_ffi_compare.py --iters 1000 --batches 9
iters=1000 batches=9 unit=us/call (host-only, no timed sync)
FFI packed nop             median=0.1443 samples=[0.153134, 0.14301499999999998, 0.142882, 0.146194, 0.144263, 0.141289, 0.141308, 0.145559, 0.145422]
FFI typed nop              median=0.1496 samples=[0.157445, 0.146261, 0.146591, 0.14961000000000002, 0.149584, 0.15335, 0.148081, 0.15193199999999998, 0.14907800000000002]
FFI empty kernel           median=3.6831 samples=[3.587373, 4.52569, 3.683096, 4.2270389999999995, 3.4973270000000003, 4.377035, 3.5415360000000002, 4.419442, 3.532073]
INTJ empty kernel          median=3.6582 samples=[3.1577689999999996, 3.658241, 4.165671000000001, 3.52946, 3.9656640000000003, 3.6284029999999996, 4.003353, 3.5098719999999997, 3.953667]
INTJ fixed-device kernel   median=3.3525 samples=[3.443392, 3.3525120000000004, 3.509005, 3.1524929999999998, 3.325625, 3.174666, 3.376651, 3.153056, 3.361905]
FFI packed nop mixed       median=0.1734 samples=[0.175373, 0.173359, 0.173321, 0.172881, 0.173829, 0.172044, 0.174637, 0.173644, 0.17093]
FFI typed nop mixed        median=0.1764 samples=[0.17925, 0.177739, 0.176057, 0.179875, 0.17621, 0.177613, 0.175959, 0.17638800000000002, 0.17619100000000001]
FFI mixed kernel           median=3.8189 samples=[3.818872, 4.607868000000001, 3.63137, 4.858345, 3.571825, 4.374211, 3.618674, 4.444259, 3.578434]
INTJ mixed kernel          median=3.6612 samples=[3.661168, 3.67867, 3.384837, 4.032029, 3.366724, 4.006183, 3.3693519999999997, 3.533562, 3.9668850000000004]
INTJ fixed mixed kernel    median=3.4850 samples=[3.605221, 3.6906779999999997, 3.503433, 3.591859, 3.425221, 3.484962, 3.4294000000000002, 3.4567620000000003, 3.416417]
elapsed_ns=3814828976
exit_status=0
ended=2026-09-27T21:05:50+08:00
```

round 2:
```text
commit=f507cfcc57a41d73f66152aff7e1b3b0562c8324
started=2026-09-27T21:06:45+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_ffi_compare.py --iters 1000 --batches 9
iters=1000 batches=9 unit=us/call (host-only, no timed sync)
FFI packed nop             median=0.1490 samples=[0.15292, 0.147512, 0.14903, 0.147871, 0.151921, 0.152161, 0.147837, 0.151414, 0.147201]
FFI typed nop              median=0.1486 samples=[0.147851, 0.149752, 0.149337, 0.148122, 0.148643, 0.15151599999999998, 0.14821600000000001, 0.15015799999999999, 0.147457]
FFI empty kernel           median=3.3391 samples=[3.582237, 3.392324, 3.400254, 3.203363, 3.2097890000000002, 3.3732159999999998, 3.191029, 3.339118, 3.2025419999999998]
INTJ empty kernel          median=2.9892 samples=[3.076636, 3.095724, 3.09123, 2.924254, 2.896763, 2.98024, 2.989223, 2.917081, 3.000405]
INTJ fixed-device kernel   median=2.9028 samples=[3.104955, 3.1087800000000003, 3.0857620000000003, 2.934911, 2.90279, 2.80789, 2.80883, 2.819098, 2.819734]
FFI packed nop mixed       median=0.1800 samples=[0.18171199999999998, 0.179952, 0.180533, 0.188474, 0.179551, 0.179489, 0.17807499999999998, 0.17915899999999998, 0.18126599999999998]
FFI typed nop mixed        median=0.1802 samples=[0.180219, 0.181125, 0.178439, 0.179573, 0.179761, 0.180592, 0.178347, 0.182096, 0.181874]
FFI mixed kernel           median=3.3149 samples=[3.3770149999999997, 3.448276, 3.314892, 3.273447, 3.2172199999999997, 3.339269, 3.260539, 3.350863, 3.257281]
INTJ mixed kernel          median=2.9520 samples=[3.155629, 3.1220250000000003, 2.9520399999999998, 2.928382, 2.865583, 2.906213, 2.8455079999999997, 2.972826, 3.012295]
INTJ fixed mixed kernel    median=2.9180 samples=[3.137564, 3.120024, 2.963787, 2.9891900000000002, 2.884394, 2.8606219999999998, 2.8934879999999996, 2.918027, 2.89172]
elapsed_ns=3747386244
exit_status=0
ended=2026-09-27T21:06:49+08:00
```

### ffi_sweep

round 0:
```text
commit=f507cfcc57a41d73f66152aff7e1b3b0562c8324
started=2026-09-27T21:04:48+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_ffi_compare.py --sweep --iters 1000 --batches 9
counts=[0, 3, 5, 8, 16, 32, 64] iters=1000 batches=9 unit=us/call (host-only, no timed sync); INTJ device-bound, no specialization key
args= 0 FFI packed nop           median=0.1162 samples=[0.11955299999999999, 0.115595, 0.114958, 0.11621899999999999, 0.116571, 0.118566, 0.115812, 0.116242, 0.114753]
args= 0 FFI typed nop            median=0.1192 samples=[0.13033, 0.119217, 0.117527, 0.122375, 0.118857, 0.125606, 0.11781, 0.122632, 0.118039]
args= 0 FFI empty kernel         median=1.7993 samples=[1.572735, 1.80819, 9.720558, 1.813728, 1.5874380000000001, 1.799318, 1.607753, 1.8357059999999998, 1.576512]
args= 0 INTJ static_compile kernel median=2.7440 samples=[3.160998, 2.73829, 11.910868, 2.745243, 2.732941, 2.986543, 2.736328, 2.742542, 2.744033]
args= 0 INTJ runtime_shim kernel median=2.7191 samples=[2.892883, 2.6778910000000002, 12.042909, 2.661794, 2.723719, 2.680504, 2.738505, 2.6798200000000003, 2.719067]
args= 3 FFI packed nop           median=0.1451 samples=[0.14714, 0.145101, 0.146208, 0.144103, 0.147519, 0.147126, 0.143632, 0.144375, 0.144255]
args= 3 FFI typed nop            median=0.1501 samples=[0.148203, 0.152286, 0.14921299999999998, 0.15009, 0.146594, 0.15259299999999998, 0.15858699999999998, 0.15248599999999998, 0.14946600000000002]
args= 3 FFI empty kernel         median=3.2451 samples=[3.3764589999999997, 3.264442, 3.225392, 3.296542, 3.22826, 3.245091, 3.217079, 3.285987, 3.212603]
args= 3 INTJ static_compile kernel median=2.8923 samples=[3.024207, 2.857897, 2.85396, 2.8943290000000004, 2.854991, 3.1115, 2.892275, 2.906168, 2.838984]
args= 3 INTJ runtime_shim kernel median=2.8622 samples=[3.024391, 2.7477080000000003, 2.872928, 2.794634, 2.862214, 2.9177939999999998, 2.906435, 2.788611, 2.8430079999999998]
args= 5 FFI packed nop           median=0.1736 samples=[0.17388599999999999, 0.173563, 0.170519, 0.177741, 0.17798599999999998, 0.17243, 0.172413, 0.171961, 0.17358]
args= 5 FFI typed nop            median=0.1766 samples=[0.174524, 0.17823599999999998, 0.173954, 0.178656, 0.176778, 0.180707, 0.17622300000000002, 0.17662799999999998, 0.176434]
args= 5 FFI empty kernel         median=3.2722 samples=[3.3891880000000003, 3.212848, 3.272172, 3.2795880000000004, 3.237196, 3.27215, 3.272536, 3.2771329999999996, 3.256525]
args= 5 INTJ static_compile kernel median=2.8771 samples=[3.030048, 2.875146, 2.872533, 2.8808350000000003, 2.848537, 2.888299, 2.90314, 2.877102, 2.870287]
args= 5 INTJ runtime_shim kernel median=2.9130 samples=[3.041219, 2.955859, 2.84863, 2.929127, 2.851603, 2.916503, 2.897603, 2.913007, 2.875833]
args= 8 FFI packed nop           median=0.2067 samples=[0.206687, 0.205976, 0.205333, 0.20714500000000002, 0.210624, 0.205173, 0.206809, 0.208548, 0.206291]
args= 8 FFI typed nop            median=0.2088 samples=[0.208797, 0.210263, 0.206996, 0.21238300000000002, 0.209563, 0.213091, 0.206957, 0.208253, 0.208339]
args= 8 FFI empty kernel         median=3.3974 samples=[3.531815, 3.397406, 3.299414, 3.4200630000000003, 3.284746, 3.422522, 3.321648, 3.401346, 3.280917]
args= 8 INTJ static_compile kernel median=3.0786 samples=[3.078589, 2.919165, 3.100181, 2.941396, 3.091597, 3.179448, 2.925773, 2.946565, 3.086525]
args= 8 INTJ runtime_shim kernel median=2.8921 samples=[3.081947, 2.900688, 2.979933, 2.8902229999999998, 2.888916, 2.878564, 2.9320030000000004, 2.8825439999999998, 2.892067]
args=16 FFI packed nop           median=0.2917 samples=[0.291143, 0.29147500000000004, 0.300038, 0.30170600000000003, 0.301228, 0.294887, 0.29135300000000003, 0.291651, 0.291676]
args=16 FFI typed nop            median=0.3049 samples=[0.30419999999999997, 0.30680599999999997, 0.312359, 0.30999400000000005, 0.313073, 0.304911, 0.297643, 0.298346, 0.300126]
args=16 FFI empty kernel         median=3.6688 samples=[3.798607, 3.650098, 3.574303, 3.668825, 3.693987, 3.697294, 3.540982, 3.645037, 3.6835500000000003]
args=16 INTJ static_compile kernel median=3.3831 samples=[3.3525650000000002, 3.383118, 3.5601860000000003, 3.450771, 3.382483, 3.394456, 3.651101, 3.322992, 3.323142]
args=16 INTJ runtime_shim kernel median=3.2543 samples=[3.305231, 3.0983020000000003, 3.3357669999999997, 3.044729, 3.274603, 3.031337, 3.304518, 3.034519, 3.254343]
args=32 FFI packed nop           median=0.4783 samples=[0.476757, 0.47608300000000003, 0.508007, 0.485477, 0.48876, 0.476793, 0.473457, 0.479419, 0.478284]
args=32 FFI typed nop            median=0.4974 samples=[0.489625, 0.510662, 0.506721, 0.497045, 0.507312, 0.49742899999999995, 0.485345, 0.499113, 0.48605000000000004]
args=32 FFI empty kernel         median=4.2685 samples=[4.347243000000001, 4.229029000000001, 4.147955, 4.374769000000001, 4.316555999999999, 4.268534, 4.097386, 4.224810000000001, 4.300612]
args=32 INTJ static_compile kernel median=3.6214 samples=[3.5257840000000003, 3.585473, 3.6213960000000003, 3.811877, 3.776453, 3.593004, 3.625953, 3.555583, 3.737734]
args=32 INTJ runtime_shim kernel median=3.5961 samples=[3.526397, 3.434492, 3.5961030000000003, 3.609549, 3.6287420000000004, 3.43901, 3.705294, 3.422449, 3.606837]
args=64 FFI packed nop           median=0.8441 samples=[0.8440850000000001, 0.847236, 0.861968, 0.842063, 0.8563379999999999, 0.833434, 0.8529869999999999, 0.822108, 0.833241]
args=64 FFI typed nop            median=0.8671 samples=[0.869231, 0.861722, 0.875217, 0.85926, 0.889242, 0.877809, 0.8554539999999999, 0.86712, 0.8608680000000001]
args=64 FFI empty kernel         median=5.0938 samples=[5.2692380000000005, 5.095720999999999, 5.08174, 5.052191, 5.100902, 5.093831, 5.098639, 5.05582, 5.045966999999999]
args=64 INTJ static_compile kernel median=4.5541 samples=[4.7137910000000005, 4.557581, 4.53943, 4.502012, 4.554063, 4.591844, 4.542313, 4.559243, 4.548039]
args=64 INTJ runtime_shim kernel median=4.5748 samples=[4.622162, 4.595881, 4.507351, 4.518506, 4.5364070000000005, 4.584926, 4.498729, 4.574763, 5.327326]
elapsed_ns=6398170730
exit_status=0
ended=2026-09-27T21:04:54+08:00
```

round 1:
```text
commit=f507cfcc57a41d73f66152aff7e1b3b0562c8324
started=2026-09-27T21:05:50+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_ffi_compare.py --sweep --iters 1000 --batches 9
counts=[0, 3, 5, 8, 16, 32, 64] iters=1000 batches=9 unit=us/call (host-only, no timed sync); INTJ device-bound, no specialization key
args= 0 FFI packed nop           median=0.1173 samples=[0.120851, 0.11667, 0.119104, 0.11488899999999999, 0.115488, 0.11802800000000001, 0.11626099999999999, 0.12332299999999999, 0.117262]
args= 0 FFI typed nop            median=0.1198 samples=[0.127998, 0.118544, 0.119417, 0.122695, 0.117681, 0.120295, 0.11809099999999999, 0.124144, 0.119825]
args= 0 FFI empty kernel         median=1.6552 samples=[1.62807, 1.86479, 1.6071900000000001, 1.873056, 1.596839, 1.9159220000000001, 1.655199, 1.967944, 1.60223]
args= 0 INTJ static_compile kernel median=2.7118 samples=[3.153105, 2.67306, 2.733261, 2.77442, 2.710064, 2.691606, 2.764357, 2.711755, 2.69941]
args= 0 INTJ runtime_shim kernel median=2.7243 samples=[2.9086480000000003, 2.740499, 2.774217, 2.7015680000000004, 2.7228000000000003, 2.668222, 2.7778449999999997, 2.669106, 2.724272]
args= 3 FFI packed nop           median=0.1450 samples=[0.14931899999999998, 0.145148, 0.146565, 0.144807, 0.143476, 0.144785, 0.142548, 0.14725899999999997, 0.144975]
args= 3 FFI typed nop            median=0.1500 samples=[0.147185, 0.151505, 0.148651, 0.150825, 0.149991, 0.15180600000000002, 0.14852, 0.153306, 0.145864]
args= 3 FFI empty kernel         median=3.2500 samples=[3.367406, 3.316675, 3.1884789999999996, 3.2500259999999996, 3.179246, 3.325872, 3.169569, 3.316525, 3.1726509999999997]
args= 3 INTJ static_compile kernel median=2.9845 samples=[3.028243, 3.145384, 2.992201, 2.877263, 3.000089, 2.887315, 2.9844589999999998, 2.879487, 2.96536]
args= 3 INTJ runtime_shim kernel median=2.9075 samples=[3.013575, 2.7954470000000002, 2.9259340000000003, 2.9172689999999997, 2.804134, 2.80059, 2.9075059999999997, 2.8025729999999998, 2.907594]
args= 5 FFI packed nop           median=0.1734 samples=[0.175695, 0.173236, 0.174264, 0.17438, 0.172338, 0.173429, 0.17859, 0.17263399999999998, 0.17191800000000002]
args= 5 FFI typed nop            median=0.1764 samples=[0.178537, 0.179211, 0.17562799999999998, 0.177005, 0.175411, 0.179904, 0.175153, 0.176398, 0.173416]
args= 5 FFI empty kernel         median=3.2977 samples=[3.40463, 3.280726, 3.314878, 3.334911, 3.239441, 3.289975, 3.297656, 3.292805, 3.309863]
args= 5 INTJ static_compile kernel median=2.9685 samples=[3.057622, 2.9967020000000004, 2.978206, 2.991421, 2.8337269999999997, 2.9492249999999998, 2.968479, 2.9463719999999998, 2.96817]
args= 5 INTJ runtime_shim kernel median=2.9288 samples=[3.072629, 2.992647, 2.924941, 3.055003, 2.8836370000000002, 2.9302089999999996, 2.914934, 2.9288049999999997, 2.92522]
args= 8 FFI packed nop           median=0.2064 samples=[0.210695, 0.206427, 0.20702199999999998, 0.206412, 0.206313, 0.204315, 0.20944900000000002, 0.205088, 0.20597]
args= 8 FFI typed nop            median=0.2084 samples=[0.20630600000000002, 0.209122, 0.20522100000000001, 0.207964, 0.20825200000000002, 0.211784, 0.20884, 0.21003899999999998, 0.208405]
args= 8 FFI empty kernel         median=3.4058 samples=[3.541047, 3.3917800000000002, 3.492398, 3.426076, 3.2904560000000003, 3.405815, 3.312903, 3.397384, 3.5663679999999998]
args= 8 INTJ static_compile kernel median=3.0360 samples=[3.0787370000000003, 2.914626, 3.073119, 3.035973, 3.114181, 3.0000340000000003, 3.034746, 2.997601, 3.056783]
args= 8 INTJ runtime_shim kernel median=2.9072 samples=[3.105061, 2.907199, 2.889886, 2.925993, 2.9855869999999998, 2.894871, 10.024004000000001, 2.898455, 2.883279]
args=16 FFI packed nop           median=0.2940 samples=[0.291555, 0.295151, 0.29201699999999997, 0.29122699999999996, 0.293951, 0.294837, 0.470514, 0.289903, 0.295017]
args=16 FFI typed nop            median=0.3041 samples=[0.29575999999999997, 0.307195, 0.302479, 0.30568, 0.30410899999999996, 0.310118, 0.655018, 0.30173300000000003, 0.303985]
args=16 FFI empty kernel         median=3.6718 samples=[3.737657, 3.7073470000000004, 3.572917, 3.6593139999999997, 3.5697910000000004, 3.695201, 6.324171000000001, 3.671809, 3.547982]
args=16 INTJ static_compile kernel median=3.3596 samples=[3.285993, 3.3595740000000003, 3.49468, 3.312708, 3.447715, 3.311611, 3.516393, 3.317399, 3.480197]
args=16 INTJ runtime_shim kernel median=3.2922 samples=[3.292207, 3.101967, 3.39067, 3.072789, 3.4019679999999997, 3.0377240000000003, 3.383789, 3.0449319999999997, 3.311808]
args=32 FFI packed nop           median=0.4780 samples=[0.478599, 0.471314, 0.483817, 0.4757, 0.477363, 0.48105000000000003, 0.481573, 0.47802, 0.47316800000000003]
args=32 FFI typed nop            median=0.4944 samples=[0.49997699999999995, 0.494765, 0.489879, 0.494417, 0.49491199999999996, 0.491634, 0.496392, 0.491365, 0.478005]
args=32 FFI empty kernel         median=4.2310 samples=[4.2984279999999995, 4.230989999999999, 4.140584, 4.24897, 4.133718, 4.2655330000000005, 4.137006, 4.243617, 4.110855]
args=32 INTJ static_compile kernel median=3.6191 samples=[3.503643, 3.6191210000000003, 3.696858, 3.571583, 3.660529, 3.559025, 3.695233, 3.586095, 3.648359]
args=32 INTJ runtime_shim kernel median=3.4787 samples=[3.4786840000000003, 3.466186, 3.6483600000000003, 3.461878, 3.596044, 3.445023, 3.6651190000000002, 3.444637, 3.587426]
args=64 FFI packed nop           median=0.8393 samples=[0.824324, 0.846259, 0.839341, 0.825761, 0.844543, 0.83639, 0.8389650000000001, 0.8403229999999999, 0.84945]
args=64 FFI typed nop            median=0.8616 samples=[0.849626, 0.870452, 0.863404, 0.856834, 0.847086, 0.861581, 0.863773, 0.865516, 0.860337]
args=64 FFI empty kernel         median=5.0675 samples=[5.103331, 5.051884, 5.105572, 5.065981, 5.0423919999999995, 5.067470999999999, 5.071747, 5.073061, 5.046104000000001]
args=64 INTJ static_compile kernel median=4.5821 samples=[4.569979, 4.570189999999999, 4.646074, 4.5953479999999995, 4.556916, 4.582141, 4.582758, 4.593216, 4.5686040000000006]
args=64 INTJ runtime_shim kernel median=4.5761 samples=[4.502216000000001, 4.577546, 4.554855, 4.612401, 4.577073, 4.5761400000000005, 4.5320339999999995, 4.616356, 4.568963]
elapsed_ns=4317674718
exit_status=0
ended=2026-09-27T21:05:55+08:00
```

round 2:
```text
commit=f507cfcc57a41d73f66152aff7e1b3b0562c8324
started=2026-09-27T21:06:49+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_ffi_compare.py --sweep --iters 1000 --batches 9
counts=[0, 3, 5, 8, 16, 32, 64] iters=1000 batches=9 unit=us/call (host-only, no timed sync); INTJ device-bound, no specialization key
args= 0 FFI packed nop           median=0.1172 samples=[0.120063, 0.114971, 0.11612900000000001, 0.116471, 0.120249, 0.115119, 0.118806, 0.11723, 0.118339]
args= 0 FFI typed nop            median=0.1206 samples=[0.119586, 0.125994, 0.11852599999999999, 0.123265, 0.118873, 0.123506, 0.119049, 0.120619, 0.12100799999999999]
args= 0 FFI empty kernel         median=2.2353 samples=[1.658247, 2.9479740000000003, 2.235306, 2.811269, 2.0192159999999997, 2.804815, 2.04798, 2.6252530000000003, 2.0166020000000002]
args= 0 INTJ static_compile kernel median=3.2109 samples=[3.365232, 3.291705, 3.286634, 3.210859, 3.098471, 3.233452, 3.205125, 3.1974679999999998, 3.1790100000000003]
args= 0 INTJ runtime_shim kernel median=3.1805 samples=[3.185572, 3.005991, 3.3266280000000004, 3.0136399999999997, 3.1804720000000004, 3.020375, 3.249304, 2.970382, 3.1851909999999997]
args= 3 FFI packed nop           median=0.1456 samples=[0.14632900000000001, 0.14489500000000002, 0.143684, 0.146352, 0.145648, 0.148435, 0.144539, 0.144625, 0.147563]
args= 3 FFI typed nop            median=0.1482 samples=[0.148074, 0.147691, 0.14557599999999998, 0.15027000000000001, 0.145828, 0.151605, 0.14821700000000002, 0.152478, 0.14936000000000002]
args= 3 FFI empty kernel         median=3.7378 samples=[3.7378359999999997, 4.254805999999999, 3.553886, 4.191702, 3.672375, 4.374037, 3.721984, 4.201642, 3.6729789999999998]
args= 3 INTJ static_compile kernel median=3.4209 samples=[3.293131, 3.423562, 3.834932, 3.478747, 3.4014119999999997, 3.510877, 3.375108, 3.420919, 3.3772379999999997]
args= 3 INTJ runtime_shim kernel median=3.3384 samples=[3.2940050000000003, 3.3430549999999997, 3.338359, 3.131691, 3.423777, 3.138613, 3.416076, 3.320076, 3.415753]
args= 5 FFI packed nop           median=0.1743 samples=[0.17639, 0.174287, 0.170875, 0.174013, 0.170565, 0.178455, 0.17250800000000002, 0.17726499999999998, 0.174401]
args= 5 FFI typed nop            median=0.1782 samples=[0.179577, 0.17753, 0.17366499999999999, 0.181124, 0.178204, 0.179298, 0.177519, 0.177854, 0.178365]
args= 5 FFI empty kernel         median=3.7828 samples=[3.772641, 4.305390999999999, 3.613584, 4.2755, 3.7827919999999997, 4.295313, 3.7739059999999998, 4.288685, 3.7711230000000002]
args= 5 INTJ static_compile kernel median=3.4155 samples=[3.2982289999999996, 3.510012, 3.449777, 3.434078, 3.3653380000000004, 3.546349, 3.370849, 3.415451, 3.381815]
args= 5 INTJ runtime_shim kernel median=3.4092 samples=[3.3021190000000002, 3.427481, 3.45293, 3.319645, 3.409151, 3.4507820000000002, 3.4258919999999997, 3.330627, 3.409]
args= 8 FFI packed nop           median=0.2058 samples=[0.206421, 0.206014, 0.205191, 0.20677, 0.207012, 0.205516, 0.202283, 0.205778, 0.205643]
args= 8 FFI typed nop            median=0.2102 samples=[0.21301499999999998, 0.21520699999999998, 0.20666800000000002, 0.210226, 0.210778, 0.20738, 0.20562200000000003, 0.21114, 0.21013900000000002]
args= 8 FFI empty kernel         median=4.0297 samples=[4.0296590000000005, 4.284563, 3.882576, 4.266234000000001, 3.737768, 4.2613509999999994, 3.758014, 4.271337000000001, 3.7244800000000002]
args= 8 INTJ static_compile kernel median=3.6492 samples=[3.396676, 3.566922, 4.054453, 3.576142, 4.002689, 3.649226, 4.095521, 3.6404029999999996, 4.059897]
args= 8 INTJ runtime_shim kernel median=3.4575 samples=[3.563907, 3.257777, 3.457501, 3.262555, 3.498663, 3.286128, 3.493248, 3.2521970000000002, 3.675193]
args=16 FFI packed nop           median=0.2918 samples=[0.291842, 0.290861, 0.291709, 0.291383, 0.29284, 0.288677, 0.29475799999999996, 0.296148, 0.29866899999999996]
args=16 FFI typed nop            median=0.3024 samples=[0.302691, 0.30021499999999995, 0.302603, 0.301144, 0.302254, 0.30238299999999996, 0.29893200000000003, 0.311584, 0.306543]
args=16 FFI empty kernel         median=4.3356 samples=[4.335559, 5.9982169999999995, 4.102345000000001, 5.925397, 4.145089, 5.979641, 4.133457999999999, 5.744600999999999, 4.157127999999999]
args=16 INTJ static_compile kernel median=5.0993 samples=[3.8269810000000004, 5.2271, 5.421835, 5.091374, 5.09933, 5.1496189999999995, 5.168964, 4.8800349999999995, 5.06987]
args=16 INTJ runtime_shim kernel median=3.8321 samples=[3.8321039999999997, 3.643772, 5.44853, 3.548498, 5.242388, 3.578881, 4.8238639999999995, 3.5589630000000003, 4.982895]
args=32 FFI packed nop           median=0.4798 samples=[0.482606, 0.476819, 0.489579, 0.47180900000000003, 0.473353, 0.479843, 0.465581, 0.480526, 0.49048]
args=32 FFI typed nop            median=0.4934 samples=[0.489652, 0.493385, 0.501689, 0.501101, 0.491203, 0.493395, 0.486479, 0.50717, 0.505092]
args=32 FFI empty kernel         median=4.9591 samples=[4.9590559999999995, 5.970539, 4.927422, 5.9847969999999995, 4.757274000000001, 6.072743, 4.813419, 5.881575, 4.808372]
args=32 INTJ static_compile kernel median=5.5285 samples=[4.252689, 5.472542000000001, 5.784665, 5.4710410000000005, 13.797975000000001, 5.528457, 6.114456000000001, 5.454821, 6.309302]
args=32 INTJ runtime_shim kernel median=4.1650 samples=[4.165037, 4.15638, 5.599939, 3.963181, 22.237976, 4.039124, 5.895005, 4.047919, 5.793812]
args=64 FFI packed nop           median=0.8473 samples=[0.8360299999999999, 0.845414, 0.825477, 0.854407, 1.922422, 0.8507089999999999, 0.845434, 0.847279, 0.85576]
args=64 FFI typed nop            median=0.8721 samples=[0.8720589999999999, 0.901134, 0.857452, 0.883317, 2.3485709999999997, 0.864221, 0.859768, 0.878739, 0.8701409999999999]
args=64 FFI empty kernel         median=6.3986 samples=[5.866695, 6.460811, 5.735093, 6.407112, 7.579930999999999, 6.4212110000000004, 5.794146, 6.3985590000000006, 5.781549]
args=64 INTJ static_compile kernel median=6.0666 samples=[5.387949, 5.999433, 6.272108, 6.0665640000000005, 6.058587, 6.075633, 6.15184, 6.080937, 6.0505320000000005]
args=64 INTJ runtime_shim kernel median=5.9950 samples=[5.2519610000000005, 6.066390999999999, 5.994951, 6.108493999999999, 5.958867000000001, 6.065761, 5.9778459999999995, 6.127052, 5.959452000000001]
elapsed_ns=4674359922
exit_status=0
ended=2026-09-27T21:06:54+08:00
```

### ffi_paths

round 0:
```text
commit=f507cfcc57a41d73f66152aff7e1b3b0562c8324
started=2026-09-27T21:04:54+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_intj_ffi_paths.py --mode all --iters 1000 --batches 9
iters=1000 batches=9 unit=ns/call; no_gpu=True
mode=callback
callback_count=9100 (100 warmup + 9000 unique misses)
INTJ cold compile callback + cache  median_ns=404.6 samples_ns=[479.648, 488.388, 383.016, 586.854, 365.719, 374.085, 402.956, 404.594, 753.873]
INTJ hot call (no callback)         median_ns=37.2 samples_ns=[39.647, 37.146, 36.283, 37.161, 36.842, 36.58, 37.185, 37.27, 37.224]
INTJ 3-tensor host-only nop         median_ns=33.8 samples_ns=[44.577, 33.934, 33.725, 33.766, 33.77, 33.71, 33.754, 33.744, 33.773]
mode=kwargs
INTJ direct positional              median_ns=63.8 samples_ns=[65.772, 63.737, 63.826, 63.688, 63.702, 63.884, 63.767, 63.713, 63.807]
INTJ adapter positional             median_ns=88.3 samples_ns=[89.156, 94.16, 89.256, 87.521, 86.793, 88.72, 87.415, 86.805, 88.32]
INTJ adapter kwargs                 median_ns=105.2 samples_ns=[104.83, 105.915, 104.731, 110.915, 106.209, 105.181, 104.719, 105.268, 104.538]
INTJ adapter defaults               median_ns=89.6 samples_ns=[89.843, 89.629, 90.291, 93.418, 89.594, 88.573, 88.927, 88.379, 88.445]
INTJ FFI wrapper positional         median_ns=101.7 samples_ns=[102.293, 102.017, 100.247, 100.273, 100.808, 104.733, 101.721, 100.965, 101.812]
INTJ FFI wrapper kwargs             median_ns=125.8 samples_ns=[126.534, 126.187, 124.056, 125.593, 128.577, 126.477, 125.484, 125.788, 125.526]
INTJ native kwargs: unsupported (verified TypeError)
mode=dataclass
INTJ pair direct                    median_ns=65.9 samples_ns=[68.51, 65.88, 72.01, 65.731, 65.907, 65.796, 65.733, 65.745, 65.938]
INTJ pair prebuilt *tuple           median_ns=136.5 samples_ns=[136.349, 137.329, 135.352, 136.536, 141.586, 135.923, 135.708, 137.863, 138.598]
FFI unpack Pair only                median_ns=162.4 samples_ns=[159.988, 164.79, 163.546, 162.808, 162.324, 162.37, 162.004, 162.312, 165.008]
INTJ pair manual unpack             median_ns=171.0 samples_ns=[170.956, 171.589, 171.851, 171.651, 173.664, 168.693, 169.808, 168.337, 168.752]
INTJ pair FFI unpack                median_ns=312.5 samples_ns=[313.738, 309.833, 311.142, 313.824, 313.84, 311.519, 312.521, 314.74, 308.813]
INTJ pair stdlib astuple            median_ns=1153.3 samples_ns=[1164.795, 1161.061, 1152.767, 1144.121, 1151.02, 1154.438, 1149.214, 1157.916, 1153.296]
INTJ config direct                  median_ns=80.1 samples_ns=[80.249, 80.108, 80.411, 81.004, 81.103, 79.666, 79.884, 79.805, 79.625]
INTJ config FFI unpack              median_ns=312.8 samples_ns=[330.968, 311.986, 311.276, 320.686, 311.202, 311.187, 314.458, 312.801, 315.327]
INTJ native dataclass unpack: unsupported (verified TypeError)
elapsed_ns=3428446528
exit_status=0
ended=2026-09-27T21:04:58+08:00
```

round 1:
```text
commit=f507cfcc57a41d73f66152aff7e1b3b0562c8324
started=2026-09-27T21:05:55+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_intj_ffi_paths.py --mode all --iters 1000 --batches 9
iters=1000 batches=9 unit=ns/call; no_gpu=True
mode=callback
callback_count=9100 (100 warmup + 9000 unique misses)
INTJ cold compile callback + cache  median_ns=391.7 samples_ns=[450.796, 454.392, 376.939, 1042.313, 360.311, 371.012, 382.463, 391.72, 2087.295]
INTJ hot call (no callback)         median_ns=38.2 samples_ns=[39.647, 38.301, 36.874, 38.465, 36.571, 42.228, 36.804, 38.241, 37.426]
INTJ 3-tensor host-only nop         median_ns=33.8 samples_ns=[35.951, 34.193, 33.598, 33.81, 33.483, 33.832, 33.436, 33.855, 33.503]
mode=kwargs
INTJ direct positional              median_ns=65.0 samples_ns=[66.979, 65.276, 64.827, 64.963, 64.866, 64.95, 64.866, 64.857, 65.013]
INTJ adapter positional             median_ns=91.4 samples_ns=[99.321, 89.938, 90.653, 91.613, 90.291, 91.641, 91.353, 91.995, 91.291]
INTJ adapter kwargs                 median_ns=105.8 samples_ns=[106.477, 111.566, 105.689, 106.069, 105.85, 106.163, 104.969, 105.47, 105.277]
INTJ adapter defaults               median_ns=89.3 samples_ns=[91.238, 89.998, 95.311, 89.091, 88.965, 88.975, 88.864, 89.343, 89.374]
INTJ FFI wrapper positional         median_ns=102.6 samples_ns=[102.112, 102.149, 102.261, 111.329, 104.391, 103.461, 103.841, 102.572, 101.509]
INTJ FFI wrapper kwargs             median_ns=128.3 samples_ns=[131.419, 130.23, 123.24, 129.332, 128.328, 128.448, 127.662, 127.195, 125.744]
INTJ native kwargs: unsupported (verified TypeError)
mode=dataclass
INTJ pair direct                    median_ns=67.7 samples_ns=[69.575, 68.247, 67.147, 66.573, 65.792, 67.861, 67.264, 68.445, 67.703]
INTJ pair prebuilt *tuple           median_ns=136.9 samples_ns=[136.863, 140.534, 135.29, 138.71, 136.185, 137.734, 136.433, 142.358, 135.502]
FFI unpack Pair only                median_ns=166.9 samples_ns=[169.287, 170.868, 170.481, 170.197, 166.866, 163.741, 161.874, 164.078, 162.428]
INTJ pair manual unpack             median_ns=171.8 samples_ns=[171.818, 177.134, 170.495, 172.163, 169.392, 173.03, 174.678, 171.557, 168.945]
INTJ pair FFI unpack                median_ns=314.5 samples_ns=[313.765, 318.256, 312.542, 316.756, 314.367, 319.656, 314.375, 314.471, 317.883]
INTJ pair stdlib astuple            median_ns=1150.3 samples_ns=[1206.884, 1200.455, 1149.07, 1152.338, 1145.823, 1157.82, 1143.283, 1150.266, 1148.034]
INTJ config direct                  median_ns=78.9 samples_ns=[82.612, 80.08, 78.615, 79.526, 78.268, 79.537, 78.269, 78.907, 77.985]
INTJ config FFI unpack              median_ns=318.8 samples_ns=[326.544, 321.465, 318.784, 323.41, 317.948, 317.774, 315.716, 321.199, 314.645]
INTJ native dataclass unpack: unsupported (verified TypeError)
elapsed_ns=3000662018
exit_status=0
ended=2026-09-27T21:05:58+08:00
```

round 2:
```text
commit=f507cfcc57a41d73f66152aff7e1b3b0562c8324
started=2026-09-27T21:06:54+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_intj_ffi_paths.py --mode all --iters 1000 --batches 9
iters=1000 batches=9 unit=ns/call; no_gpu=True
mode=callback
callback_count=9100 (100 warmup + 9000 unique misses)
INTJ cold compile callback + cache  median_ns=404.8 samples_ns=[456.45, 453.118, 377.562, 938.375, 358.666, 380.231, 398.058, 404.827, 1836.077]
INTJ hot call (no callback)         median_ns=38.1 samples_ns=[40.243, 38.636, 38.097, 38.083, 37.717, 38.624, 38.071, 38.03, 37.465]
INTJ 3-tensor host-only nop         median_ns=34.2 samples_ns=[35.938, 34.232, 33.693, 33.948, 33.509, 33.685, 37.029, 35.573, 34.872]
mode=kwargs
INTJ direct positional              median_ns=66.8 samples_ns=[72.224, 65.85, 65.62, 65.482, 66.863, 66.997, 66.938, 66.811, 66.827]
INTJ adapter positional             median_ns=91.8 samples_ns=[93.147, 93.01, 91.616, 93.183, 93.814, 89.575, 89.698, 91.797, 89.867]
INTJ adapter kwargs                 median_ns=106.8 samples_ns=[105.998, 105.983, 106.78, 107.078, 107.216, 110.332, 106.979, 106.677, 106.797]
INTJ adapter defaults               median_ns=89.4 samples_ns=[93.825, 90.01, 89.278, 89.205, 88.982, 89.374, 93.225, 89.745, 88.372]
INTJ FFI wrapper positional         median_ns=102.8 samples_ns=[104.329, 103.508, 102.132, 102.463, 102.529, 102.443, 102.763, 107.978, 104.942]
INTJ FFI wrapper kwargs             median_ns=124.7 samples_ns=[124.968, 124.668, 124.569, 123.604, 124.441, 124.576, 129.947, 126.871, 126.387]
INTJ native kwargs: unsupported (verified TypeError)
mode=dataclass
INTJ pair direct                    median_ns=67.9 samples_ns=[75.397, 68.372, 67.705, 67.885, 67.271, 67.94, 66.548, 68.736, 66.819]
INTJ pair prebuilt *tuple           median_ns=138.6 samples_ns=[138.654, 139.864, 139.619, 138.623, 135.236, 139.232, 134.592, 137.806, 134.09]
FFI unpack Pair only                median_ns=162.1 samples_ns=[165.002, 162.666, 160.1, 162.052, 161.063, 161.521, 168.614, 162.658, 161.172]
INTJ pair manual unpack             median_ns=172.5 samples_ns=[171.052, 173.569, 171.042, 188.139, 171.347, 174.325, 172.025, 174.404, 172.549]
INTJ pair FFI unpack                median_ns=315.5 samples_ns=[319.978, 317.024, 315.465, 317.837, 313.287, 314.258, 320.948, 315.225, 310.316]
INTJ pair stdlib astuple            median_ns=1174.4 samples_ns=[1207.576, 1192.434, 1143.988, 1178.929, 1150.743, 1166.864, 1174.403, 1178.466, 1156.673]
INTJ config direct                  median_ns=81.3 samples_ns=[81.307, 81.989, 81.735, 82.267, 83.18, 80.016, 79.544, 80.256, 79.634]
INTJ config FFI unpack              median_ns=318.0 samples_ns=[315.402, 322.888, 317.218, 317.983, 321.001, 317.697, 315.405, 318.832, 318.715]
INTJ native dataclass unpack: unsupported (verified TypeError)
elapsed_ns=2871427770
exit_status=0
ended=2026-09-27T21:06:57+08:00
```

### hip_module

round 0:
```text
commit=f507cfcc57a41d73f66152aff7e1b3b0562c8324
started=2026-09-27T21:04:58+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_hip_module_launch.py --iters 1000 --batches 9
hsaco_sha256=df3e529e178196ad62f7c07ab8593ede5b8596c76e7c6fbd12a5d947a696e65c kernel=empty_kernel threads=256 shared=0 callback_count=1
iters=1000 batches=9 unit=us/call (host enqueue, no timed sync)
TVM FFI HIP HSACO         median=3.2051 samples=[3.425755, 3.327305, 3.32925, 3.233905, 3.2050520000000002, 3.138277, 3.125228, 3.081057, 3.118097]
Triton same HSACO         median=16.0743 samples=[16.219226, 16.223054, 16.135161, 16.156154, 16.074285, 16.047247, 16.059807, 16.018881999999998, 15.987043]
INTJ same function        median=2.9864 samples=[3.104619, 3.103054, 3.050232, 3.042859, 2.986447, 2.91486, 2.889551, 2.9036210000000002, 2.880756]
elapsed_ns=3998286692
exit_status=0
ended=2026-09-27T21:05:02+08:00
```

round 1:
```text
commit=f507cfcc57a41d73f66152aff7e1b3b0562c8324
started=2026-09-27T21:05:58+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_hip_module_launch.py --iters 1000 --batches 9
hsaco_sha256=df3e529e178196ad62f7c07ab8593ede5b8596c76e7c6fbd12a5d947a696e65c kernel=empty_kernel threads=256 shared=0 callback_count=1
iters=1000 batches=9 unit=us/call (host enqueue, no timed sync)
TVM FFI HIP HSACO         median=3.6393 samples=[3.467518, 3.697328, 3.773351, 3.690042, 3.6393429999999998, 3.649001, 3.6323499999999997, 3.569739, 3.570893]
Triton same HSACO         median=16.3112 samples=[16.436698, 16.343093, 16.311199000000002, 16.324277, 16.317819, 16.126992, 16.140719, 16.135738, 16.051894]
INTJ same function        median=3.4372 samples=[3.437185, 3.5276300000000003, 3.523562, 3.52088, 3.515254, 3.429802, 3.4181340000000002, 3.433046, 3.393666]
elapsed_ns=3896736584
exit_status=0
ended=2026-09-27T21:06:01+08:00
```

round 2:
```text
commit=f507cfcc57a41d73f66152aff7e1b3b0562c8324
started=2026-09-27T21:06:57+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_hip_module_launch.py --iters 1000 --batches 9
hsaco_sha256=df3e529e178196ad62f7c07ab8593ede5b8596c76e7c6fbd12a5d947a696e65c kernel=empty_kernel threads=256 shared=0 callback_count=1
iters=1000 batches=9 unit=us/call (host enqueue, no timed sync)
TVM FFI HIP HSACO         median=3.5929 samples=[3.592872, 3.667311, 3.7903789999999997, 3.696812, 3.64106, 3.586216, 3.571002, 3.520183, 3.537442]
Triton same HSACO         median=16.3513 samples=[16.630982, 16.392937999999997, 16.396815, 16.404863000000002, 16.351287, 16.237341, 16.182528, 16.234486, 16.198578]
INTJ same function        median=3.4534 samples=[3.43342, 3.492228, 3.5896120000000002, 3.613778, 3.4829499999999998, 3.453439, 3.388893, 3.389557, 3.3360459999999996]
elapsed_ns=3861357380
exit_status=0
ended=2026-09-27T21:07:00+08:00
```

### kernel_cache

round 0:
```text
commit=f507cfcc57a41d73f66152aff7e1b3b0562c8324
started=2026-09-27T21:05:02+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python -m pytest tests/test_kernel_cache.py -s -q

kernel cache, 1-word key, ns per operation

                      intj       tsl      absl
hash_only             1.35      1.75      1.36
hit/1                 1.76      2.02      1.44
hit/8                 1.60      2.10      4.36
hit/64                1.95      2.13      4.54
hit/512               2.30      3.74      5.30
hit_child             3.01      3.00      3.00
miss/1                1.21      2.06      1.38
miss/8                2.68      1.76      3.28
miss/64               2.72      1.76      3.53
miss/512              3.40      2.00      4.06

kernel cache, 2-word key, ns per operation

                      intj       tsl      absl
hash_only             1.65      1.42      1.42
hit/1                 1.82      2.71      2.03
hit/8                 1.83      2.57      5.05
hit/64                2.57      2.92      5.24
hit/512               3.03      4.03      6.37
hit_child             3.01      3.00      3.36
miss/1                1.38      2.33      1.48
miss/8                2.89      2.57      3.22
miss/64               3.12      2.09      3.99
miss/512              2.87      2.71      4.00

kernel cache, 5-word key, ns per operation

                      intj       tsl      absl
hash_only             3.59      3.59      4.25
hit/1                 2.99      3.06      2.46
hit/8                 3.20      4.71      9.51
hit/64                3.27      3.50      9.30
hit/512               4.55      4.77     12.28
hit_child             3.00      3.01      3.01
miss/1                1.73      2.11      2.14
miss/8                3.00      2.24      6.93
miss/64               4.39      2.49      6.92
miss/512              4.78      2.52      7.74
........
8 passed in 26.71s
elapsed_ns=27640191674
exit_status=0
ended=2026-09-27T21:05:29+08:00
```

round 1:
```text
commit=f507cfcc57a41d73f66152aff7e1b3b0562c8324
started=2026-09-27T21:06:01+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python -m pytest tests/test_kernel_cache.py -s -q

kernel cache, 1-word key, ns per operation

                      intj       tsl      absl
hash_only             1.42      1.36      1.36
hit/1                 1.58      2.03      1.62
hit/8                 1.66      2.10      4.35
hit/64                1.71      2.13      4.54
hit/512               2.30      2.90      5.32
hit_child             3.66      3.00      3.00
miss/1                1.22      1.63      1.67
miss/8                3.68      1.75      3.28
miss/64               2.78      1.75      3.54
miss/512              2.91      2.00      4.27

kernel cache, 2-word key, ns per operation

                      intj       tsl      absl
hash_only             1.42      1.42      1.42
hit/1                 1.84      2.59      2.00
hit/8                 1.82      2.56      5.04
hit/64                2.20      2.93      5.24
hit/512               3.02      4.05     10.65
hit_child             3.00      3.00      3.01
miss/1                1.38      2.32      1.45
miss/8                2.43      1.86      3.19
miss/64               3.18      2.27      3.89
miss/512              2.90      2.70      3.99

kernel cache, 5-word key, ns per operation

                      intj       tsl      absl
hash_only             3.58      3.59      3.58
hit/1                 2.99      3.06      2.50
hit/8                 3.20      3.41      9.48
hit/64                3.26      3.51      9.77
hit/512               4.54      4.79     11.16
hit_child             3.01      3.01      3.00
miss/1                1.41      2.10      2.16
miss/8                3.03      2.24      9.37
miss/64               3.36      1.93      6.72
miss/512              3.49      2.88      7.81
........
8 passed in 25.71s
elapsed_ns=26629147369
exit_status=0
ended=2026-09-27T21:06:28+08:00
```

round 2:
```text
commit=f507cfcc57a41d73f66152aff7e1b3b0562c8324
started=2026-09-27T21:07:00+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python -m pytest tests/test_kernel_cache.py -s -q

kernel cache, 1-word key, ns per operation

                      intj       tsl      absl
hash_only             1.36      1.36      1.37
hit/1                 1.58      2.01      1.44
hit/8                 1.66      2.10      4.36
hit/64                1.61      2.58      5.41
hit/512               2.30      2.90      5.33
hit_child             3.00      3.00      3.00
miss/1                1.21      1.58      1.38
miss/8                2.54      2.21      3.28
miss/64               2.82      1.72      3.54
miss/512              2.94      1.99      4.05

kernel cache, 2-word key, ns per operation

                      intj       tsl      absl
hash_only             2.19      1.42      1.42
hit/1                 2.93      2.57      2.02
hit/8                 1.91      2.57      5.03
hit/64                2.18      2.92      6.08
hit/512               3.02      4.04      6.39
hit_child             3.00      3.00      3.38
miss/1                1.39      3.19      1.49
miss/8                2.44      1.85      3.20
miss/64               3.21      2.10      4.47
miss/512              2.87      2.72      3.99

kernel cache, 5-word key, ns per operation

                      intj       tsl      absl
hash_only             3.59      3.60      5.14
hit/1                 2.99      3.05      2.46
hit/8                 3.86      3.41      9.49
hit/64                3.26      3.52      9.77
hit/512               4.53      4.77     11.16
hit_child             3.02      3.01      3.01
miss/1                1.41      2.10      2.05
miss/8                2.98      3.26      6.93
miss/64               2.84      1.97      6.73
miss/512              3.50      2.51      7.78
........
8 passed in 25.67s
elapsed_ns=26596494280
exit_status=0
ended=2026-09-27T21:07:27+08:00
```

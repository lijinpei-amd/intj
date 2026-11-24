# Review fixes: `c47f011` vs the uncommitted review-fix tree `b9da963` (index 0)

The code review of 2026-11-24 fixed its findings in five parallel slices (wave 1) and one rendered-C
refactor (wave 2), all uncommitted on `develop` over `c47f011`. This A/B checks the launch rows, and the
bisect below explains the one flagged row.

## Environment

- Baseline `c47f011` (`git archive` export, `/tmp/intj-bench/c47f011`). Candidate: the uncommitted working
  tree, exported from `git stash create` object `b9da9634f58b868a896758a779fc0e6564d99596` (not on any
  branch; source tree hash `b02f727b0f6f9acd6e5c4faaa066cf2c149cc979`) to `/tmp/intj-bench/fixes`. It is not
  a clean checkout of any commit.
- Xeon Platinum 8480C on `taskset -c 0`, MI308X (`gfx942`) GPU 1 for the A/B, picked and held by the
  `gpu-lock.sh` protocol (`HIP_VISIBLE_DEVICES=1`). uv venv, Python 3.12.3, Torch
  `2.14.0.dev20260804+rocm7.1`, Triton `3.8.0`. A fresh `TRITON_HOME` per side
  (`/tmp/intj-bench/home-<side>`, intj deps symlinked), so neither side reads the other's modules.
- 2026-11-24, A/B 19:53 to 20:03 +08:00; bisect from 20:03 (tree bisect) to 21:21 +08:00. Load average 1 to 4 from other users.
  CUDA: untested.

## Command

`/tmp/intj-bench/fixes_ab.sh` (the driver of
[`2026-10-29_compile-options-ab_0_87e8a54.md`](2026-10-29_compile-options-ab_0_87e8a54.md) with the sides
replaced and a private `TRITON_HOME` per side), run as `gpu-lock.sh /tmp/intj-bench/fixes_ab.sh`: 3 rounds;
each round runs every case on `c47f011`, then on the candidate. Every process:
`cd <tree>; TRITON_HOME=/tmp/intj-bench/home-<side> PYTHONPATH=$PWD taskset -c 0 <venv>/bin/python <args>`.

```
launch_gpu      benchmarks/bench_launch.py --iters 1000 --batches 9
launch_host     benchmarks/bench_launch.py --no-gpu --iters 100000 --batches 9
launch_readme   benchmarks/bench_launch.py --readme --iters 1000 --batches 9
launch_sweep    benchmarks/bench_launch.py --sweep --iters 100000 --batches 9
launch_last_key benchmarks/bench_launch.py --last-key --iters 100000 --batches 9
launch_tuned    benchmarks/bench_launch.py --tuned --iters 20000 --batches 9
launch_dynamic  benchmarks/bench_launch.py --dynamic --iters 20000 --batches 9
```

Compared with `/tmp/intj-bench/fixes_ab_summary.py` (`bench_compare.parse_file` and its regression rule:
slower by more than 3 % and by more than 2 ns, or 0.1 us), median of the 3 process medians. All 42 files
have `exit_status=0`. `bench_launch.py` prints each process's median of its 9 batches, not per-batch samples.

## Result

**1 of 53 rows flagged: `--last-key` 32 int arguments, alternating keys, 117.9 -> 128.8 ns (+9.2 %).**
Every other row is within -5.3 .. +2.9 %:

| case | rows | delta % range |
|---|---:|---|
| `launch_gpu` | 12 | -1.5 .. +2.9 |
| `launch_host` | 12 | -1.5 .. +1.6 |
| `launch_readme` | 7 | -2.7 .. +0.3 |
| `launch_sweep` | 6 | -2.7 .. +1.0 |
| `launch_last_key` | 6 | -0.8 .. +9.2 |
| `launch_tuned` | 2 | -1.1 .. +1.7 |
| `launch_dynamic` | 8 | -5.3 .. +2.8 |

### Bisect of the flagged row: code layout, no single change

Host-only `--last-key`, 32-argument rows only, ns/call (one value per process), each table one
`gpu-lock.sh` hold with sides interleaved, core 0. `wave1` is `c47f011` plus the five wave-1 slices;
`fixes` adds wave 2. Variants below build only the 32-argument module from an edited copy of the
rendered `noop.cpp` and runtime headers, inside the `fixes` tree (`/tmp/intj-bench/bis/variant_bench.py`,
which substitutes the source at `_compile_so_bytes` with the build's own flags).

| run | variant | what it is | 32 alternate | 32 repeat |
|---|---|---|---|---|
| tree bisect, GPU 3 | `c47f011` | baseline tree | 116.7 117.8 117.2 117.7 | 114.7 112.7 114.6 114.9 |
| tree bisect, GPU 3 | `wave1` | wave-1 tree | 117.3 117.8 135.0 117.7 | 114.0 112.9 131.3 113.1 |
| tree bisect, GPU 3 | `fixes` | wave 1 + wave 2 tree | 129.3 131.3 130.5 132.1 | 112.6 111.9 113.9 114.3 |
| r1, GPU 3 | `w1tree` | wave-1 tree | 117.9 117.3 118.3 | 114.3 113.9 114.5 |
| r1, GPU 3 | `fxtree` | fixes tree | 130.9 130.0 131.0 | 113.2 113.0 112.8 |
| r1, GPU 3 | `fxown` | fixes C rebuilt by the harness | 131.0 128.1 131.7 | 114.2 113.4 112.0 |
| r1, GPU 3 | `fxw1src` | wave-1 C (source + headers) in the fixes tree | 117.7 117.2 117.8 | 114.1 113.1 114.7 |
| r2, GPU 6 | `w1src` | wave-1 C | 118.2 117.3 117.2 | 114.9 113.5 113.1 |
| r2, GPU 6 | `fxown` | fixes C | 125.2 128.5 131.4 | 113.2 113.7 113.2 |
| r2, GPU 6 | `RA` | fixes, per-argument `Py_TYPE` local instead of `intj_is_tensor` | 123.6 122.2 122.7 | 115.5 116.2 116.3 |
| r2, GPU 6 | `RB` | fixes, old `intj_grid_dim` body | 131.5 129.4 130.7 | 115.1 114.2 114.6 |
| r2, GPU 6 | `RC` | fixes, put without `intj_refs_reserve`/`intj_final_list` | 131.6 124.9 131.0 | 113.2 113.7 113.7 |
| r2, GPU 6 | `RD` | fixes, old `intj_launch_error` | 128.0 132.8 131.6 | 113.0 113.1 113.2 |
| r2, GPU 6 | `ALL` | fixes, RA+RB+RC+RD | 126.0 124.0 124.7 | 117.4 117.9 117.0 |
| r3, GPU 5 | `w1src` | wave-1 C | 117.7 117.2 118.5 | 114.0 113.8 112.9 |
| r3, GPU 5 | `fxown` | fixes C | 131.9 128.7 131.4 | 113.0 112.8 113.5 |
| r3, GPU 5 | `NP` | fixes, `intj_note_param` `noinline, cold` | 123.7 122.6 123.0 | 119.1 117.9 119.5 |
| r3, GPU 5 | `W1NP` | wave-1 C, `intj_note_param` `noinline, cold` | 129.1 128.9 128.8 | 123.8 123.9 124.4 |
| r3, GPU 5 | `W1DH` | wave-1 C without `device_handle` | 118.1 117.7 116.7 | 114.1 113.8 113.7 |
| r3, GPU 5 | `W1REFS` | wave-1 C with a 32-byte pad where `refs` sits | 117.8 117.1 118.0 | 113.8 113.6 114.7 |
| r3, GPU 5 | `FXDH` | fixes C with `device_handle` back | 129.1 129.9 131.1 | 113.3 112.4 113.3 |
| r4, GPU 7 | `w1src` | wave-1 C | 117.0 117.7 117.8 | 113.8 113.3 113.7 |
| r4, GPU 7 | `fxown` | fixes C | 130.6 130.6 129.6 | 113.9 113.7 113.6 |
| r4, GPU 7 | `LO` | fixes, hash + probe in a `noinline` function | 131.7 130.8 131.7 | 125.1 124.6 124.9 |
| r4, GPU 7 | `W1LO` | wave-1 C, hash + probe in a `noinline` function | 122.0 121.4 122.0 | 116.3 117.1 116.3 |

Code alignment (`/tmp/intj-bench/align_ab.sh`, GPU 0, flags appended to every rendered module), 32 alternate / repeat:

| flags | wave1 | fixes |
|---|---|---|
| none | 117.9 118.2 118.1 / 113.3 115.1 114.2 | 132.3 129.2 132.2 / 113.5 113.2 113.1 |
| `-falign-functions=64 -falign-jumps=64 -falign-loops=64` | 119.2 149.9 119.1 / 115.2 146.2 114.9 | 122.3 123.2 122.2 / 119.8 119.2 117.5 |
| `-falign-functions=32 -falign-loops=32` | 118.5 118.4 119.0 / 113.9 113.4 114.1 | 136.9 136.6 136.0 / 129.9 127.4 127.6 |

Reading:

- The regression enters with wave 2 (tree bisect), and lives entirely in the 32-argument module's C:
  wave-1 C built inside the fixes tree runs at wave-1 speed (`r1 fxw1src`).
- No wave-2 change is the trigger. Reverting each source change (`RA`..`RD`), all of them (`ALL`), or
  the launcher-state layout (`FXDH`; `W1DH`, `W1REFS` from the other side) leaves 122 to 131 ns.
- Behaviour-free perturbations swing either tree as far as wave 2 did, in both directions: one
  `noinline, cold` attribute moves wave 1 to 129 / 124 ns (`W1NP`); an out-of-line probe moves fixes
  to 131 / 125 ns (`LO`); `-falign-*` moves fixes between 122 and 136 ns while wave 1 stays at 118.
- In the fixes layout the memo-miss path itself is slow: alternate minus repeat is ~17 ns, against
  ~4 to 7 ns in every other variant. The rendered hit and miss path is the same C in both trees
  (decode, `intj_walk`, `intj_cache_lookup`); `intj_call` is ~32 KB (32 inlined argument decodes), and
  where gcc places it decides the cost. Without hardware counters (`perf` and `valgrind` are
  unavailable here), front-end or branch-predictor aliasing is the likely mechanism, not measured.
- No source change was applied: any variant that lands on a faster layout for this one row is the same
  kind of accident as the regression, and none was faster on both trees.

## Raw output

### Comparison (`c47f011` = baseline, `fixes` = candidate)

| case | table | row | col | unit | c47f011 | fixes | delta | pct | flag |
|---|---|---|---|---|---:|---:|---:|---:|---|
| launch_dynamic | dynamic | 2 dynamic options | launch | ns | 3118.10 | 3109.40 | -8.70 | -0.3% |  |
| launch_dynamic | dynamic | 2 dynamic options | spec_key | ns | 99.20 | 93.90 | -5.30 | -5.3% |  |
| launch_dynamic | dynamic | int constexpr | launch | ns | 3059.30 | 3098.60 | +39.30 | +1.3% |  |
| launch_dynamic | dynamic | int constexpr | spec_key | ns | 86.70 | 89.10 | +2.40 | +2.8% |  |
| launch_dynamic | dynamic | plain | launch | ns | 3092.20 | 3100.20 | +8.00 | +0.3% |  |
| launch_dynamic | dynamic | plain | spec_key | ns | 93.00 | 93.40 | +0.40 | +0.4% |  |
| launch_dynamic | dynamic | str constexpr | launch | ns | 3088.10 | 3046.70 | -41.40 | -1.3% |  |
| launch_dynamic | dynamic | str constexpr | spec_key | ns | 86.60 | 88.10 | +1.50 | +1.7% |  |
| launch_gpu | launch | auto map | ns/call | ns | 3288.70 | 3277.00 | -11.70 | -0.4% |  |
| launch_gpu | launch | baked | ns/call | ns | 3097.80 | 3148.60 | +50.80 | +1.6% |  |
| launch_gpu | launch | bound pointer | ns/call | ns | 3127.90 | 3115.40 | -12.50 | -0.4% |  |
| launch_gpu | launch | bound tensor | ns/call | ns | 3100.40 | 3086.10 | -14.30 | -0.5% |  |
| launch_gpu | launch | fixed device map | ns/call | ns | 3168.70 | 3160.50 | -8.20 | -0.3% |  |
| launch_gpu | launch | fixed device no-map | ns/call | ns | 3107.00 | 3091.00 | -16.00 | -0.5% |  |
| launch_gpu | launch | generic | ns/call | ns | 3200.70 | 3196.00 | -4.70 | -0.1% |  |
| launch_gpu | launch | generic tensor ann | ns/call | ns | 3189.00 | 3141.80 | -47.20 | -1.5% |  |
| launch_gpu | launch | reduced key | ns/call | ns | 3152.30 | 3116.00 | -36.30 | -1.2% |  |
| launch_gpu | launch | tensor annotated | ns/call | ns | 3278.60 | 3287.20 | +8.60 | +0.3% |  |
| launch_gpu | launch | verify off | ns/call | ns | 3132.30 | 3122.60 | -9.70 | -0.3% |  |
| launch_gpu | launch | verify on | ns/call | ns | 3111.70 | 3201.20 | +89.50 | +2.9% |  |
| launch_host | launch | auto map | ns/call | ns | 44.10 | 44.10 | +0.00 | +0.0% |  |
| launch_host | launch | baked | ns/call | ns | 40.90 | 40.90 | +0.00 | +0.0% |  |
| launch_host | launch | bound pointer | ns/call | ns | 45.50 | 44.80 | -0.70 | -1.5% |  |
| launch_host | launch | bound tensor | ns/call | ns | 47.40 | 46.90 | -0.50 | -1.1% |  |
| launch_host | launch | fixed device map | ns/call | ns | 45.50 | 45.50 | +0.00 | +0.0% |  |
| launch_host | launch | fixed device no-map | ns/call | ns | 44.70 | 45.40 | +0.70 | +1.6% |  |
| launch_host | launch | generic | ns/call | ns | 47.30 | 47.00 | -0.30 | -0.6% |  |
| launch_host | launch | generic tensor ann | ns/call | ns | 47.30 | 47.40 | +0.10 | +0.2% |  |
| launch_host | launch | reduced key | ns/call | ns | 44.10 | 44.10 | +0.00 | +0.0% |  |
| launch_host | launch | tensor annotated | ns/call | ns | 43.70 | 44.10 | +0.40 | +0.9% |  |
| launch_host | launch | verify off | ns/call | ns | 44.60 | 44.90 | +0.30 | +0.7% |  |
| launch_host | launch | verify on | ns/call | ns | 45.60 | 45.70 | +0.10 | +0.2% |  |
| launch_last_key | sweep | 16 alternate | ns/call | ns | 69.70 | 69.40 | -0.30 | -0.4% |  |
| launch_last_key | sweep | 16 repeat | ns/call | ns | 66.50 | 66.00 | -0.50 | -0.8% |  |
| launch_last_key | sweep | 32 alternate | ns/call | ns | 117.90 | 128.80 | +10.90 | +9.2% | REGRESSION |
| launch_last_key | sweep | 32 repeat | ns/call | ns | 113.40 | 112.70 | -0.70 | -0.6% |  |
| launch_last_key | sweep | 4 alternate | ns/call | ns | 37.80 | 37.60 | -0.20 | -0.5% |  |
| launch_last_key | sweep | 4 repeat | ns/call | ns | 35.30 | 35.40 | +0.10 | +0.3% |  |
| launch_readme | readme_path | grid=(0,) intj | us | us | 0.03 | 0.03 | +0.00 | +0.0% |  |
| launch_readme | readme_path | grid=(0,) triton | us | us | 12.60 | 12.64 | +0.04 | +0.3% |  |
| launch_readme | readme_path | grid=(1,) intj | us | us | 3.32 | 3.23 | -0.09 | -2.7% |  |
| launch_readme | readme_path | grid=(1,) triton | us | us | 16.38 | 16.30 | -0.08 | -0.5% |  |
| launch_readme | readme_torch_access | interpreter | decode ns | ns | 902.80 | 891.20 | -11.60 | -1.3% |  |
| launch_readme | readme_torch_access | runtime_shim | decode ns | ns | 92.20 | 92.10 | -0.10 | -0.1% |  |
| launch_readme | readme_torch_access | static_compile | decode ns | ns | 94.90 | 93.80 | -1.10 | -1.2% |  |
| launch_sweep | sweep | 16 int | ns/call | ns | 72.60 | 72.30 | -0.30 | -0.4% |  |
| launch_sweep | sweep | 16 tensor | ns/call | ns | 68.00 | 68.40 | +0.40 | +0.6% |  |
| launch_sweep | sweep | 32 int | ns/call | ns | 122.00 | 121.30 | -0.70 | -0.6% |  |
| launch_sweep | sweep | 32 tensor | ns/call | ns | 126.20 | 122.80 | -3.40 | -2.7% |  |
| launch_sweep | sweep | 4 int | ns/call | ns | 41.70 | 42.10 | +0.40 | +1.0% |  |
| launch_sweep | sweep | 4 tensor | ns/call | ns | 40.60 | 40.60 | +0.00 | +0.0% |  |
| launch_tuned | tuned | plain | ns/launch | ns | 3122.70 | 3176.70 | +54.00 | +1.7% |  |
| launch_tuned | tuned | tuned hit | ns/launch | ns | 3157.50 | 3123.00 | -34.50 | -1.1% |  |
rows=53 flagged=1

### Per-process output (A/B)

#### `c47f011_r0_launch_dynamic.txt`

```
commit=c47f011 (git archive export)
gpu=HIP_VISIBLE_DEVICES=1
started=2026-11-24T19:55:24+08:00 load=1.94 2.30 4.80
command=TRITON_HOME=/tmp/intj-bench/home-c47f011 PYTHONPATH=$PWD taskset -c 0 /mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/venv/bin/python benchmarks/bench_launch.py --dynamic --iters 20000 --batches 9
mode=dynamic; 20000 calls × 9 batches; median ns/call
               path     launch   spec_key
              plain     3083.2       93.0
  2 dynamic options     3106.8       99.2
      int constexpr     3068.3       86.1
      str constexpr     3088.1       87.0
elapsed_ns=17117007701
exit_status=0
```

#### `c47f011_r0_launch_gpu.txt`

```
commit=c47f011 (git archive export)
gpu=HIP_VISIBLE_DEVICES=1
started=2026-11-24T19:53:11+08:00 load=1.16 2.44 5.24
command=TRITON_HOME=/tmp/intj-bench/home-c47f011 PYTHONPATH=$PWD taskset -c 0 /mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/venv/bin/python benchmarks/bench_launch.py --iters 1000 --batches 9
mode=gpu; 1000 calls × 9 batches; median ns/call
               path    ns/call       Δ ns       Δ %   build ms   bind ms
           auto map     3976.4       +0.0      +0.0    2992.31         -
   tensor annotated     4111.8     +135.4      +3.4    2181.45         -
            generic     4091.4     +115.0      +2.9    2459.65         -
 generic tensor ann     4106.9     +130.5      +3.3    2312.12         -
        reduced key     4003.1      +26.7      +0.7    2272.88         -
         verify off     3947.8      -28.6      -0.7    2199.79         -
          verify on     4092.9     +116.5      +2.9    2317.85         -
              baked     4016.4      +40.0      +1.0    2087.06         -
       bound tensor     4027.4      +51.0      +1.3    2193.51      0.04
      bound pointer     4060.1      +83.7      +2.1    2205.58      0.04
   fixed device map     4070.0      +93.6      +2.4    2310.07      0.04
fixed device no-map     4048.7      +72.3      +1.8    2107.18      0.04
elapsed_ns=38257672543
exit_status=0
```

#### `c47f011_r0_launch_host.txt`

```
commit=c47f011 (git archive export)
gpu=HIP_VISIBLE_DEVICES=1
started=2026-11-24T19:53:49+08:00 load=1.87 2.46 5.12
command=TRITON_HOME=/tmp/intj-bench/home-c47f011 PYTHONPATH=$PWD taskset -c 0 /mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/venv/bin/python benchmarks/bench_launch.py --no-gpu --iters 100000 --batches 9
mode=host-only; 100000 calls × 9 batches; median ns/call
               path    ns/call       Δ ns       Δ %   build ms   bind ms
           auto map       45.0       +0.0      +0.0    2443.84         -
   tensor annotated       44.3       -0.7      -1.6    2196.21         -
            generic       48.0       +2.9      +6.5    2433.25         -
 generic tensor ann       47.7       +2.7      +6.0    2277.86         -
        reduced key       44.7       -0.3      -0.7    2183.67         -
         verify off       44.9       -0.2      -0.4    2226.08         -
          verify on       45.9       +0.9      +2.0    2328.36         -
              baked       41.5       -3.5      -7.9    2088.46         -
       bound tensor       50.4       +5.4     +12.0    2186.46      0.04
      bound pointer       46.5       +1.5      +3.3    2195.09      0.05
   fixed device map       46.1       +1.1      +2.5    2316.79      0.04
fixed device no-map       46.1       +1.1      +2.4    2105.37      0.04
elapsed_ns=36551823037
exit_status=0
```

#### `c47f011_r0_launch_last_key.txt`

```
commit=c47f011 (git archive export)
gpu=HIP_VISIBLE_DEVICES=1
started=2026-11-24T19:54:58+08:00 load=1.76 2.31 4.87
command=TRITON_HOME=/tmp/intj-bench/home-c47f011 PYTHONPATH=$PWD taskset -c 0 /mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/venv/bin/python benchmarks/bench_launch.py --last-key --iters 100000 --batches 9
mode=last-key; host-only; 200000 calls × 9 batches; median ns/call
count     pattern    ns/call
    4      repeat       35.3
    4   alternate       37.8
   16      repeat       66.4
   16   alternate       69.7
   32      repeat      113.3
   32   alternate      117.4
elapsed_ns=9791425645
exit_status=0
```

#### `c47f011_r0_launch_readme.txt`

```
commit=c47f011 (git archive export)
gpu=HIP_VISIBLE_DEVICES=1
started=2026-11-24T19:54:26+08:00 load=1.80 2.38 5.00
command=TRITON_HOME=/tmp/intj-bench/home-c47f011 PYTHONPATH=$PWD taskset -c 0 /mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/venv/bin/python benchmarks/bench_launch.py --readme --iters 1000 --batches 9
mode=readme; 1000 calls × 9 batches; median
        path   triton us    intj us   speedup
   grid=(1,)       16.64       3.32      5.0x
   grid=(0,)       12.60       0.03    463.6x

torch_access_mode   decode ns    build s
     runtime_shim        92.2       0.97
   static_compile        94.8       0.00
      interpreter       894.2       1.02
elapsed_ns=12279427721
exit_status=0
```

#### `c47f011_r0_launch_sweep.txt`

```
commit=c47f011 (git archive export)
gpu=HIP_VISIBLE_DEVICES=1
started=2026-11-24T19:54:38+08:00 load=1.77 2.34 4.94
command=TRITON_HOME=/tmp/intj-bench/home-c47f011 PYTHONPATH=$PWD taskset -c 0 /mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/venv/bin/python benchmarks/bench_launch.py --sweep --iters 100000 --batches 9
mode=sweep; host-only; 100000 calls × 9 batches; median ns/call
count   kind    ns/call
    4    int       42.3
    4 tensor       41.1
   16    int       72.6
   16 tensor       67.8
   32    int      121.8
   32 tensor      126.2
elapsed_ns=19908428359
exit_status=0
```

#### `c47f011_r0_launch_tuned.txt`

```
commit=c47f011 (git archive export)
gpu=HIP_VISIBLE_DEVICES=1
started=2026-11-24T19:55:08+08:00 load=1.72 2.28 4.84
command=TRITON_HOME=/tmp/intj-bench/home-c47f011 PYTHONPATH=$PWD taskset -c 0 /mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/venv/bin/python benchmarks/bench_launch.py --tuned --iters 20000 --batches 9
 tuned hit:  3160.3 ns/launch
     plain:  3201.1 ns/launch
elapsed_ns=16026444240
exit_status=0
```

#### `fixes_r0_launch_dynamic.txt`

```
commit=fixes (git archive export)
gpu=HIP_VISIBLE_DEVICES=1
started=2026-11-24T19:57:54+08:00 load=2.10 2.21 4.39
command=TRITON_HOME=/tmp/intj-bench/home-fixes PYTHONPATH=$PWD taskset -c 0 /mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/venv/bin/python benchmarks/bench_launch.py --dynamic --iters 20000 --batches 9
mode=dynamic; 20000 calls × 9 batches; median ns/call
               path     launch   spec_key
              plain     3062.8       93.0
  2 dynamic options     3109.4       96.7
      int constexpr     3068.7       88.7
      str constexpr     3072.3       86.1
elapsed_ns=17186984476
exit_status=0
```

#### `fixes_r0_launch_gpu.txt`

```
commit=fixes (git archive export)
gpu=HIP_VISIBLE_DEVICES=1
started=2026-11-24T19:55:41+08:00 load=1.89 2.27 4.75
command=TRITON_HOME=/tmp/intj-bench/home-fixes PYTHONPATH=$PWD taskset -c 0 /mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/venv/bin/python benchmarks/bench_launch.py --iters 1000 --batches 9
mode=gpu; 1000 calls × 9 batches; median ns/call
               path    ns/call       Δ ns       Δ %   build ms   bind ms
           auto map     3299.7       +0.0      +0.0    3008.98         -
   tensor annotated     3347.0      +47.3      +1.4    2232.26         -
            generic     3379.1      +79.4      +2.4    2475.35         -
 generic tensor ann     3378.3      +78.6      +2.4    2324.55         -
        reduced key     3345.7      +46.0      +1.4    2228.91         -
         verify off     3347.3      +47.6      +1.4    2233.48         -
          verify on     3365.0      +65.3      +2.0    2367.38         -
              baked     3350.4      +50.8      +1.5    2120.13         -
       bound tensor     3357.8      +58.1      +1.8    2237.61      0.04
      bound pointer     3361.1      +61.4      +1.9    2244.42      0.05
   fixed device map     3378.5      +78.8      +2.4    2332.92      0.04
fixed device no-map     3358.2      +58.5      +1.8    2139.13      0.04
elapsed_ns=38485531283
exit_status=0
```

#### `fixes_r0_launch_host.txt`

```
commit=fixes (git archive export)
gpu=HIP_VISIBLE_DEVICES=1
started=2026-11-24T19:56:19+08:00 load=2.08 2.26 4.64
command=TRITON_HOME=/tmp/intj-bench/home-fixes PYTHONPATH=$PWD taskset -c 0 /mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/venv/bin/python benchmarks/bench_launch.py --no-gpu --iters 100000 --batches 9
mode=host-only; 100000 calls × 9 batches; median ns/call
               path    ns/call       Δ ns       Δ %   build ms   bind ms
           auto map       44.2       +0.0      +0.0    2461.82         -
   tensor annotated       45.2       +1.0      +2.3    2211.49         -
            generic       47.6       +3.4      +7.7    2450.12         -
 generic tensor ann       48.0       +3.8      +8.6    2299.69         -
        reduced key       44.6       +0.4      +0.8    2216.03         -
         verify off       45.6       +1.4      +3.3    2242.70         -
          verify on       46.3       +2.1      +4.8    2366.46         -
              baked       41.5       -2.7      -6.2    2121.71         -
       bound tensor       47.5       +3.3      +7.5    2209.51      0.03
      bound pointer       45.4       +1.2      +2.7    2244.89      0.05
   fixed device map       46.0       +1.8      +4.0    2328.30      0.04
fixed device no-map       46.6       +2.4      +5.5    2144.12      0.04
elapsed_ns=36866156076
exit_status=0
```

#### `fixes_r0_launch_last_key.txt`

```
commit=fixes (git archive export)
gpu=HIP_VISIBLE_DEVICES=1
started=2026-11-24T19:57:28+08:00 load=2.13 2.22 4.45
command=TRITON_HOME=/tmp/intj-bench/home-fixes PYTHONPATH=$PWD taskset -c 0 /mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/venv/bin/python benchmarks/bench_launch.py --last-key --iters 100000 --batches 9
mode=last-key; host-only; 200000 calls × 9 batches; median ns/call
count     pattern    ns/call
    4      repeat       35.4
    4   alternate       37.6
   16      repeat       66.0
   16   alternate       69.3
   32      repeat      111.4
   32   alternate      128.8
elapsed_ns=9839511060
exit_status=0
```

#### `fixes_r0_launch_readme.txt`

```
commit=fixes (git archive export)
gpu=HIP_VISIBLE_DEVICES=1
started=2026-11-24T19:56:56+08:00 load=1.93 2.20 4.53
command=TRITON_HOME=/tmp/intj-bench/home-fixes PYTHONPATH=$PWD taskset -c 0 /mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/venv/bin/python benchmarks/bench_launch.py --readme --iters 1000 --batches 9
mode=readme; 1000 calls × 9 batches; median
        path   triton us    intj us   speedup
   grid=(1,)       16.28       3.35      4.9x
   grid=(0,)       12.64       0.03    459.5x

torch_access_mode   decode ns    build s
     runtime_shim        92.1       0.99
   static_compile        94.3       0.00
      interpreter       881.1       1.03
elapsed_ns=12360135933
exit_status=0
```

#### `fixes_r0_launch_sweep.txt`

```
commit=fixes (git archive export)
gpu=HIP_VISIBLE_DEVICES=1
started=2026-11-24T19:57:08+08:00 load=2.17 2.24 4.51
command=TRITON_HOME=/tmp/intj-bench/home-fixes PYTHONPATH=$PWD taskset -c 0 /mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/venv/bin/python benchmarks/bench_launch.py --sweep --iters 100000 --batches 9
mode=sweep; host-only; 100000 calls × 9 batches; median ns/call
count   kind    ns/call
    4    int       42.3
    4 tensor       41.1
   16    int       72.3
   16 tensor       68.4
   32    int      120.8
   32 tensor      122.8
elapsed_ns=19893432305
exit_status=0
```

#### `fixes_r0_launch_tuned.txt`

```
commit=fixes (git archive export)
gpu=HIP_VISIBLE_DEVICES=1
started=2026-11-24T19:57:38+08:00 load=2.03 2.20 4.42
command=TRITON_HOME=/tmp/intj-bench/home-fixes PYTHONPATH=$PWD taskset -c 0 /mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/venv/bin/python benchmarks/bench_launch.py --tuned --iters 20000 --batches 9
 tuned hit:  3123.0 ns/launch
     plain:  3105.2 ns/launch
elapsed_ns=16041505909
exit_status=0
```

#### `c47f011_r1_launch_dynamic.txt`

```
commit=c47f011 (git archive export)
gpu=HIP_VISIBLE_DEVICES=1
started=2026-11-24T19:59:15+08:00 load=2.67 2.32 4.24
command=TRITON_HOME=/tmp/intj-bench/home-c47f011 PYTHONPATH=$PWD taskset -c 0 /mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/venv/bin/python benchmarks/bench_launch.py --dynamic --iters 20000 --batches 9
mode=dynamic; 20000 calls × 9 batches; median ns/call
               path     launch   spec_key
              plain     3092.2       93.0
  2 dynamic options     3118.1      177.4
      int constexpr     3059.3       88.3
      str constexpr     3093.3       85.0
elapsed_ns=13077023268
exit_status=0
```

#### `c47f011_r1_launch_gpu.txt`

```
commit=c47f011 (git archive export)
gpu=HIP_VISIBLE_DEVICES=1
started=2026-11-24T19:58:11+08:00 load=2.16 2.21 4.35
command=TRITON_HOME=/tmp/intj-bench/home-c47f011 PYTHONPATH=$PWD taskset -c 0 /mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/venv/bin/python benchmarks/bench_launch.py --iters 1000 --batches 9
mode=gpu; 1000 calls × 9 batches; median ns/call
               path    ns/call       Δ ns       Δ %   build ms   bind ms
           auto map     3244.2       +0.0      +0.0      19.59         -
   tensor annotated     3225.7      -18.4      -0.6       3.33         -
            generic     3132.6     -111.6      -3.4       2.90         -
 generic tensor ann     3147.5      -96.7      -3.0       2.85         -
        reduced key     3099.9     -144.3      -4.4       2.82         -
         verify off     3087.7     -156.5      -4.8       3.00         -
          verify on     3111.7     -132.5      -4.1       2.87         -
              baked     3095.9     -148.3      -4.6       2.89         -
       bound tensor     3083.0     -161.1      -5.0       2.85      0.03
      bound pointer     3084.2     -159.9      -4.9       2.84      0.03
   fixed device map     3168.7      -75.5      -2.3       2.76      0.03
fixed device no-map     3086.0     -158.2      -4.9       3.27      0.02
elapsed_ns=10317954435
exit_status=0
```

#### `c47f011_r1_launch_host.txt`

```
commit=c47f011 (git archive export)
gpu=HIP_VISIBLE_DEVICES=1
started=2026-11-24T19:58:22+08:00 load=2.05 2.19 4.32
command=TRITON_HOME=/tmp/intj-bench/home-c47f011 PYTHONPATH=$PWD taskset -c 0 /mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/venv/bin/python benchmarks/bench_launch.py --no-gpu --iters 100000 --batches 9
mode=host-only; 100000 calls × 9 batches; median ns/call
               path    ns/call       Δ ns       Δ %   build ms   bind ms
           auto map       44.1       +0.0      +0.0       6.80         -
   tensor annotated       43.7       -0.4      -1.0       2.31         -
            generic       47.0       +2.8      +6.5       2.31         -
 generic tensor ann       47.3       +3.2      +7.3       2.17         -
        reduced key       44.1       -0.0      -0.1       2.20         -
         verify off       44.3       +0.2      +0.3       2.20         -
          verify on       45.5       +1.4      +3.1       2.18         -
              baked       40.8       -3.3      -7.4       2.20         -
       bound tensor       46.5       +2.4      +5.5       2.20      0.02
      bound pointer       44.8       +0.7      +1.5       2.18      0.03
   fixed device map       45.5       +1.4      +3.2       2.19      0.02
fixed device no-map       44.2       +0.1      +0.2       2.15      0.02
elapsed_ns=9525795596
exit_status=0
```

#### `c47f011_r1_launch_last_key.txt`

```
commit=c47f011 (git archive export)
gpu=HIP_VISIBLE_DEVICES=1
started=2026-11-24T19:58:51+08:00 load=2.17 2.20 4.26
command=TRITON_HOME=/tmp/intj-bench/home-c47f011 PYTHONPATH=$PWD taskset -c 0 /mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/venv/bin/python benchmarks/bench_launch.py --last-key --iters 100000 --batches 9
mode=last-key; host-only; 200000 calls × 9 batches; median ns/call
count     pattern    ns/call
    4      repeat       35.4
    4   alternate       37.9
   16      repeat       66.5
   16   alternate       69.8
   32      repeat      113.4
   32   alternate      118.7
elapsed_ns=10786792226
exit_status=0
```

#### `c47f011_r1_launch_readme.txt`

```
commit=c47f011 (git archive export)
gpu=HIP_VISIBLE_DEVICES=1
started=2026-11-24T19:58:31+08:00 load=2.04 2.18 4.30
command=TRITON_HOME=/tmp/intj-bench/home-c47f011 PYTHONPATH=$PWD taskset -c 0 /mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/venv/bin/python benchmarks/bench_launch.py --readme --iters 1000 --batches 9
mode=readme; 1000 calls × 9 batches; median
        path   triton us    intj us   speedup
   grid=(1,)       16.24       3.34      4.9x
   grid=(0,)       12.54       0.03    461.2x

torch_access_mode   decode ns    build s
     runtime_shim        92.2       0.02
   static_compile        94.9       0.00
      interpreter       902.8       0.00
elapsed_ns=10252969414
exit_status=0
```

#### `c47f011_r1_launch_sweep.txt`

```
commit=c47f011 (git archive export)
gpu=HIP_VISIBLE_DEVICES=1
started=2026-11-24T19:58:42+08:00 load=2.03 2.18 4.27
command=TRITON_HOME=/tmp/intj-bench/home-c47f011 PYTHONPATH=$PWD taskset -c 0 /mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/venv/bin/python benchmarks/bench_launch.py --sweep --iters 100000 --batches 9
mode=sweep; host-only; 100000 calls × 9 batches; median ns/call
count   kind    ns/call
    4    int       41.7
    4 tensor       40.6
   16    int       72.7
   16 tensor       68.2
   32    int      122.0
   32 tensor      127.4
elapsed_ns=9742617193
exit_status=0
```

#### `c47f011_r1_launch_tuned.txt`

```
commit=c47f011 (git archive export)
gpu=HIP_VISIBLE_DEVICES=1
started=2026-11-24T19:59:02+08:00 load=2.51 2.28 4.25
command=TRITON_HOME=/tmp/intj-bench/home-c47f011 PYTHONPATH=$PWD taskset -c 0 /mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/venv/bin/python benchmarks/bench_launch.py --tuned --iters 20000 --batches 9
 tuned hit:  3157.5 ns/launch
     plain:  3122.7 ns/launch
elapsed_ns=12462301378
exit_status=0
```

#### `fixes_r1_launch_dynamic.txt`

```
commit=fixes (git archive export)
gpu=HIP_VISIBLE_DEVICES=1
started=2026-11-24T20:00:29+08:00 load=6.77 4.60 4.95
command=TRITON_HOME=/tmp/intj-bench/home-fixes PYTHONPATH=$PWD taskset -c 0 /mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/venv/bin/python benchmarks/bench_launch.py --dynamic --iters 20000 --batches 9
mode=dynamic; 20000 calls × 9 batches; median ns/call
               path     launch   spec_key
              plain     3100.2       93.4
  2 dynamic options     3115.1       93.9
      int constexpr     3124.5       89.1
      str constexpr     3046.7       89.2
elapsed_ns=12277136650
exit_status=0
```

#### `fixes_r1_launch_gpu.txt`

```
commit=fixes (git archive export)
gpu=HIP_VISIBLE_DEVICES=1
started=2026-11-24T19:59:28+08:00 load=13.89 4.99 5.09
command=TRITON_HOME=/tmp/intj-bench/home-fixes PYTHONPATH=$PWD taskset -c 0 /mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/venv/bin/python benchmarks/bench_launch.py --iters 1000 --batches 9
mode=gpu; 1000 calls × 9 batches; median ns/call
               path    ns/call       Δ ns       Δ %   build ms   bind ms
           auto map     3211.1       +0.0      +0.0      20.20         -
   tensor annotated     3271.7      +60.6      +1.9       3.39         -
            generic     3132.8      -78.3      -2.4       2.94         -
 generic tensor ann     3136.2      -74.8      -2.3       2.97         -
        reduced key     3065.0     -146.1      -4.5       2.94         -
         verify off     3122.6      -88.5      -2.8       3.07         -
          verify on     3201.2       -9.8      -0.3      16.53         -
              baked     3148.6      -62.4      -1.9       2.97         -
       bound tensor     3059.8     -151.3      -4.7       2.92      0.04
      bound pointer     3114.2      -96.9      -3.0       2.95      0.03
   fixed device map     3134.2      -76.9      -2.4       2.90      0.03
fixed device no-map     3089.4     -121.7      -3.8       2.90      0.03
elapsed_ns=10789482241
exit_status=0
```

#### `fixes_r1_launch_host.txt`

```
commit=fixes (git archive export)
gpu=HIP_VISIBLE_DEVICES=1
started=2026-11-24T19:59:38+08:00 load=12.68 5.02 5.10
command=TRITON_HOME=/tmp/intj-bench/home-fixes PYTHONPATH=$PWD taskset -c 0 /mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/venv/bin/python benchmarks/bench_launch.py --no-gpu --iters 100000 --batches 9
mode=host-only; 100000 calls × 9 batches; median ns/call
               path    ns/call       Δ ns       Δ %   build ms   bind ms
           auto map       44.1       +0.0      +0.0       7.87         -
   tensor annotated       44.1       +0.0      +0.0       2.38         -
            generic       46.9       +2.7      +6.2       2.22         -
 generic tensor ann       47.4       +3.3      +7.5       2.18         -
        reduced key       44.1       -0.0      -0.0       2.22         -
         verify off       44.9       +0.7      +1.7       2.27         -
          verify on       45.7       +1.6      +3.6       2.26         -
              baked       40.8       -3.3      -7.5       2.26         -
       bound tensor       46.9       +2.8      +6.4       2.18      0.03
      bound pointer       44.7       +0.6      +1.3       2.25      0.03
   fixed device map       45.4       +1.3      +3.0       2.19      0.02
fixed device no-map       45.4       +1.2      +2.8       2.23      0.02
elapsed_ns=9805470816
exit_status=0
```

#### `fixes_r1_launch_last_key.txt`

```
commit=fixes (git archive export)
gpu=HIP_VISIBLE_DEVICES=1
started=2026-11-24T20:00:08+08:00 load=8.49 4.74 5.01
command=TRITON_HOME=/tmp/intj-bench/home-fixes PYTHONPATH=$PWD taskset -c 0 /mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/venv/bin/python benchmarks/bench_launch.py --last-key --iters 100000 --batches 9
mode=last-key; host-only; 200000 calls × 9 batches; median ns/call
count     pattern    ns/call
    4      repeat       35.7
    4   alternate       38.2
   16      repeat       69.3
   16   alternate       72.7
   32      repeat      115.0
   32   alternate      126.9
elapsed_ns=9875107675
exit_status=0
```

#### `fixes_r1_launch_readme.txt`

```
commit=fixes (git archive export)
gpu=HIP_VISIBLE_DEVICES=1
started=2026-11-24T19:59:48+08:00 load=11.18 4.95 5.08
command=TRITON_HOME=/tmp/intj-bench/home-fixes PYTHONPATH=$PWD taskset -c 0 /mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/venv/bin/python benchmarks/bench_launch.py --readme --iters 1000 --batches 9
mode=readme; 1000 calls × 9 batches; median
        path   triton us    intj us   speedup
   grid=(1,)       16.30       3.23      5.0x
   grid=(0,)       12.57       0.03    458.6x

torch_access_mode   decode ns    build s
     runtime_shim        91.3       0.02
   static_compile        93.4       0.00
      interpreter       893.4       0.00
elapsed_ns=10626627134
exit_status=0
```

#### `fixes_r1_launch_sweep.txt`

```
commit=fixes (git archive export)
gpu=HIP_VISIBLE_DEVICES=1
started=2026-11-24T19:59:59+08:00 load=9.76 4.85 5.05
command=TRITON_HOME=/tmp/intj-bench/home-fixes PYTHONPATH=$PWD taskset -c 0 /mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/venv/bin/python benchmarks/bench_launch.py --sweep --iters 100000 --batches 9
mode=sweep; host-only; 100000 calls × 9 batches; median ns/call
count   kind    ns/call
    4    int       41.6
    4 tensor       40.6
   16    int       72.3
   16 tensor       68.3
   32    int      121.3
   32 tensor      123.6
elapsed_ns=9467529129
exit_status=0
```

#### `fixes_r1_launch_tuned.txt`

```
commit=fixes (git archive export)
gpu=HIP_VISIBLE_DEVICES=1
started=2026-11-24T20:00:18+08:00 load=7.64 4.68 4.98
command=TRITON_HOME=/tmp/intj-bench/home-fixes PYTHONPATH=$PWD taskset -c 0 /mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/venv/bin/python benchmarks/bench_launch.py --tuned --iters 20000 --batches 9
 tuned hit:  3158.8 ns/launch
     plain:  3176.7 ns/launch
elapsed_ns=11096295742
exit_status=0
```

#### `c47f011_r2_launch_dynamic.txt`

```
commit=c47f011 (git archive export)
gpu=HIP_VISIBLE_DEVICES=1
started=2026-11-24T20:01:42+08:00 load=3.95 4.30 4.83
command=TRITON_HOME=/tmp/intj-bench/home-c47f011 PYTHONPATH=$PWD taskset -c 0 /mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/venv/bin/python benchmarks/bench_launch.py --dynamic --iters 20000 --batches 9
mode=dynamic; 20000 calls × 9 batches; median ns/call
               path     launch   spec_key
              plain     3152.0       95.0
  2 dynamic options     3140.3       95.3
      int constexpr     3040.7       86.7
      str constexpr     3053.2       86.6
elapsed_ns=12214010966
exit_status=0
```

#### `c47f011_r2_launch_gpu.txt`

```
commit=c47f011 (git archive export)
gpu=HIP_VISIBLE_DEVICES=1
started=2026-11-24T20:00:42+08:00 load=5.96 4.49 4.91
command=TRITON_HOME=/tmp/intj-bench/home-c47f011 PYTHONPATH=$PWD taskset -c 0 /mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/venv/bin/python benchmarks/bench_launch.py --iters 1000 --batches 9
mode=gpu; 1000 calls × 9 batches; median ns/call
               path    ns/call       Δ ns       Δ %   build ms   bind ms
           auto map     3288.7       +0.0      +0.0      19.41         -
   tensor annotated     3278.6      -10.1      -0.3       3.44         -
            generic     3200.7      -88.0      -2.7       2.89         -
 generic tensor ann     3189.0      -99.7      -3.0       2.86         -
        reduced key     3152.3     -136.4      -4.1       2.90         -
         verify off     3132.3     -156.4      -4.8       2.98         -
          verify on     3109.5     -179.2      -5.4       2.84         -
              baked     3097.8     -190.9      -5.8       2.82         -
       bound tensor     3100.4     -188.3      -5.7       2.82      0.03
      bound pointer     3127.9     -160.8      -4.9       2.82      0.03
   fixed device map     3121.9     -166.8      -5.1       2.74      0.03
fixed device no-map     3107.0     -181.7      -5.5       2.80      0.03
elapsed_ns=10294614333
exit_status=0
```

#### `c47f011_r2_launch_host.txt`

```
commit=c47f011 (git archive export)
gpu=HIP_VISIBLE_DEVICES=1
started=2026-11-24T20:00:52+08:00 load=6.63 4.74 4.99
command=TRITON_HOME=/tmp/intj-bench/home-c47f011 PYTHONPATH=$PWD taskset -c 0 /mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/venv/bin/python benchmarks/bench_launch.py --no-gpu --iters 100000 --batches 9
mode=host-only; 100000 calls × 9 batches; median ns/call
               path    ns/call       Δ ns       Δ %   build ms   bind ms
           auto map       44.1       +0.0      +0.0       6.70         -
   tensor annotated       43.7       -0.4      -0.8       2.27         -
            generic       47.3       +3.2      +7.2       2.31         -
 generic tensor ann       47.1       +3.0      +6.8       2.12         -
        reduced key       44.1       +0.0      +0.1       2.14         -
         verify off       44.6       +0.5      +1.1       2.19         -
          verify on       45.6       +1.5      +3.4       2.17         -
              baked       40.9       -3.2      -7.3       2.15         -
       bound tensor       47.4       +3.3      +7.4       2.16      0.03
      bound pointer       45.5       +1.4      +3.1       2.19      0.03
   fixed device map       45.2       +1.1      +2.6       2.18      0.02
fixed device no-map       44.7       +0.6      +1.4       2.17      0.02
elapsed_ns=9509926347
exit_status=0
```

#### `c47f011_r2_launch_last_key.txt`

```
commit=c47f011 (git archive export)
gpu=HIP_VISIBLE_DEVICES=1
started=2026-11-24T20:01:21+08:00 load=5.05 4.52 4.91
command=TRITON_HOME=/tmp/intj-bench/home-c47f011 PYTHONPATH=$PWD taskset -c 0 /mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/venv/bin/python benchmarks/bench_launch.py --last-key --iters 100000 --batches 9
mode=last-key; host-only; 200000 calls × 9 batches; median ns/call
count     pattern    ns/call
    4      repeat       35.3
    4   alternate       37.8
   16      repeat       66.5
   16   alternate       69.7
   32      repeat      113.4
   32   alternate      117.9
elapsed_ns=9831002781
exit_status=0
```

#### `c47f011_r2_launch_readme.txt`

```
commit=c47f011 (git archive export)
gpu=HIP_VISIBLE_DEVICES=1
started=2026-11-24T20:01:01+08:00 load=6.18 4.68 4.97
command=TRITON_HOME=/tmp/intj-bench/home-c47f011 PYTHONPATH=$PWD taskset -c 0 /mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/venv/bin/python benchmarks/bench_launch.py --readme --iters 1000 --batches 9
mode=readme; 1000 calls × 9 batches; median
        path   triton us    intj us   speedup
   grid=(1,)       16.38       3.31      5.0x
   grid=(0,)       12.70       0.03    465.4x

torch_access_mode   decode ns    build s
     runtime_shim        91.6       0.02
   static_compile        95.2       0.00
      interpreter       926.5       0.00
elapsed_ns=10237481131
exit_status=0
```

#### `c47f011_r2_launch_sweep.txt`

```
commit=c47f011 (git archive export)
gpu=HIP_VISIBLE_DEVICES=1
started=2026-11-24T20:01:12+08:00 load=5.70 4.62 4.95
command=TRITON_HOME=/tmp/intj-bench/home-c47f011 PYTHONPATH=$PWD taskset -c 0 /mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/venv/bin/python benchmarks/bench_launch.py --sweep --iters 100000 --batches 9
mode=sweep; host-only; 100000 calls × 9 batches; median ns/call
count   kind    ns/call
    4    int       41.6
    4 tensor       40.6
   16    int       72.6
   16 tensor       68.0
   32    int      122.7
   32 tensor      126.1
elapsed_ns=9455185765
exit_status=0
```

#### `c47f011_r2_launch_tuned.txt`

```
commit=c47f011 (git archive export)
gpu=HIP_VISIBLE_DEVICES=1
started=2026-11-24T20:01:31+08:00 load=4.50 4.42 4.87
command=TRITON_HOME=/tmp/intj-bench/home-c47f011 PYTHONPATH=$PWD taskset -c 0 /mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/venv/bin/python benchmarks/bench_launch.py --tuned --iters 20000 --batches 9
 tuned hit:  3105.3 ns/launch
     plain:  3086.8 ns/launch
elapsed_ns=11082502692
exit_status=0
```

#### `fixes_r2_launch_dynamic.txt`

```
commit=fixes (git archive export)
gpu=HIP_VISIBLE_DEVICES=1
started=2026-11-24T20:02:55+08:00 load=2.50 3.80 4.61
command=TRITON_HOME=/tmp/intj-bench/home-fixes PYTHONPATH=$PWD taskset -c 0 /mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/venv/bin/python benchmarks/bench_launch.py --dynamic --iters 20000 --batches 9
mode=dynamic; 20000 calls × 9 batches; median ns/call
               path     launch   spec_key
              plain     3116.2       94.3
  2 dynamic options     3081.9       93.5
      int constexpr     3098.6       89.4
      str constexpr     3028.0       88.1
elapsed_ns=12254829009
exit_status=0
```

#### `fixes_r2_launch_gpu.txt`

```
commit=fixes (git archive export)
gpu=HIP_VISIBLE_DEVICES=1
started=2026-11-24T20:01:54+08:00 load=3.74 4.24 4.80
command=TRITON_HOME=/tmp/intj-bench/home-fixes PYTHONPATH=$PWD taskset -c 0 /mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/venv/bin/python benchmarks/bench_launch.py --iters 1000 --batches 9
mode=gpu; 1000 calls × 9 batches; median ns/call
               path    ns/call       Δ ns       Δ %   build ms   bind ms
           auto map     3277.0       +0.0      +0.0      19.88         -
   tensor annotated     3287.2      +10.1      +0.3       3.31         -
            generic     3196.0      -81.0      -2.5       2.95         -
 generic tensor ann     3141.8     -135.2      -4.1       2.85         -
        reduced key     3116.0     -161.0      -4.9       2.93         -
         verify off     3103.0     -174.0      -5.3       3.04         -
          verify on     3115.7     -161.3      -4.9       2.88         -
              baked     3118.9     -158.1      -4.8       2.88         -
       bound tensor     3086.1     -190.9      -5.8       2.86      0.03
      bound pointer     3115.4     -161.6      -4.9       2.85      0.03
   fixed device map     3160.5     -116.5      -3.6       2.81      0.03
fixed device no-map     3091.0     -186.0      -5.7       2.88      0.02
elapsed_ns=10323462060
exit_status=0
```

#### `fixes_r2_launch_host.txt`

```
commit=fixes (git archive export)
gpu=HIP_VISIBLE_DEVICES=1
started=2026-11-24T20:02:05+08:00 load=3.40 4.15 4.76
command=TRITON_HOME=/tmp/intj-bench/home-fixes PYTHONPATH=$PWD taskset -c 0 /mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/venv/bin/python benchmarks/bench_launch.py --no-gpu --iters 100000 --batches 9
mode=host-only; 100000 calls × 9 batches; median ns/call
               path    ns/call       Δ ns       Δ %   build ms   bind ms
           auto map       44.0       +0.0      +0.0       7.68         -
   tensor annotated       44.1       +0.1      +0.2       2.34         -
            generic       47.0       +2.9      +6.6       2.24         -
 generic tensor ann       47.4       +3.4      +7.6       2.19         -
        reduced key       44.0       -0.0      -0.1       2.21         -
         verify off       44.8       +0.8      +1.8       2.25         -
          verify on       45.6       +1.6      +3.7       2.16         -
              baked       40.9       -3.1      -7.1       2.28         -
       bound tensor       46.9       +2.8      +6.4       2.22      0.02
      bound pointer       44.8       +0.8      +1.8       2.25      0.03
   fixed device map       45.5       +1.5      +3.3       2.15      0.02
fixed device no-map       44.4       +0.4      +0.9       2.20      0.02
elapsed_ns=9545196672
exit_status=0
```

#### `fixes_r2_launch_last_key.txt`

```
commit=fixes (git archive export)
gpu=HIP_VISIBLE_DEVICES=1
started=2026-11-24T20:02:34+08:00 load=2.88 3.96 4.68
command=TRITON_HOME=/tmp/intj-bench/home-fixes PYTHONPATH=$PWD taskset -c 0 /mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/venv/bin/python benchmarks/bench_launch.py --last-key --iters 100000 --batches 9
mode=last-key; host-only; 200000 calls × 9 batches; median ns/call
count     pattern    ns/call
    4      repeat       35.4
    4   alternate       37.6
   16      repeat       65.9
   16   alternate       69.4
   32      repeat      112.7
   32   alternate      133.0
elapsed_ns=9837375781
exit_status=0
```

#### `fixes_r2_launch_readme.txt`

```
commit=fixes (git archive export)
gpu=HIP_VISIBLE_DEVICES=1
started=2026-11-24T20:02:14+08:00 load=3.33 4.11 4.74
command=TRITON_HOME=/tmp/intj-bench/home-fixes PYTHONPATH=$PWD taskset -c 0 /mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/venv/bin/python benchmarks/bench_launch.py --readme --iters 1000 --batches 9
mode=readme; 1000 calls × 9 batches; median
        path   triton us    intj us   speedup
   grid=(1,)       16.31       3.23      5.1x
   grid=(0,)       12.95       0.03    470.9x

torch_access_mode   decode ns    build s
     runtime_shim        92.8       0.02
   static_compile        93.8       0.00
      interpreter       891.2       0.00
elapsed_ns=10224856467
exit_status=0
```

#### `fixes_r2_launch_sweep.txt`

```
commit=fixes (git archive export)
gpu=HIP_VISIBLE_DEVICES=1
started=2026-11-24T20:02:24+08:00 load=3.13 4.04 4.71
command=TRITON_HOME=/tmp/intj-bench/home-fixes PYTHONPATH=$PWD taskset -c 0 /mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/venv/bin/python benchmarks/bench_launch.py --sweep --iters 100000 --batches 9
mode=sweep; host-only; 100000 calls × 9 batches; median ns/call
count   kind    ns/call
    4    int       42.1
    4 tensor       40.6
   16    int       72.6
   16 tensor       68.4
   32    int      121.7
   32 tensor      120.5
elapsed_ns=9434527996
exit_status=0
```

#### `fixes_r2_launch_tuned.txt`

```
commit=fixes (git archive export)
gpu=HIP_VISIBLE_DEVICES=1
started=2026-11-24T20:02:44+08:00 load=2.67 3.88 4.65
command=TRITON_HOME=/tmp/intj-bench/home-fixes PYTHONPATH=$PWD taskset -c 0 /mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/venv/bin/python benchmarks/bench_launch.py --tuned --iters 20000 --batches 9
 tuned hit:  3115.7 ns/launch
     plain:  3186.6 ns/launch
elapsed_ns=11082970842
exit_status=0
```

### Per-process output (bisect)

#### `lastkey-bisect/c47f011_r0.txt`

```
side=c47f011 gpu=3 started=2026-11-24T20:03:55+08:00 load=1.46 3.28 4.38
mode=last-key; host-only; 200000 calls × 9 batches; median ns/call
count     pattern    ns/call
    4      repeat       35.3
    4   alternate       37.8
   16      repeat       66.4
   16   alternate       69.7
   32      repeat      114.7
   32   alternate      116.7
exit_status=0
```

#### `lastkey-bisect/c47f011_r1.txt`

```
side=c47f011 gpu=3 started=2026-11-24T20:04:35+08:00 load=1.57 3.07 4.26
mode=last-key; host-only; 200000 calls × 9 batches; median ns/call
count     pattern    ns/call
    4      repeat       35.3
    4   alternate       37.7
   16      repeat       66.4
   16   alternate       69.8
   32      repeat      112.7
   32   alternate      117.8
exit_status=0
```

#### `lastkey-bisect/c47f011_r2.txt`

```
side=c47f011 gpu=3 started=2026-11-24T20:05:05+08:00 load=1.61 2.94 4.18
mode=last-key; host-only; 200000 calls × 9 batches; median ns/call
count     pattern    ns/call
    4      repeat       35.4
    4   alternate       37.8
   16      repeat       66.4
   16   alternate       69.7
   32      repeat      114.6
   32   alternate      117.2
exit_status=0
```

#### `lastkey-bisect/c47f011_r3.txt`

```
side=c47f011 gpu=3 started=2026-11-24T20:05:35+08:00 load=1.75 2.85 4.11
mode=last-key; host-only; 200000 calls × 9 batches; median ns/call
count     pattern    ns/call
    4      repeat       35.3
    4   alternate       37.8
   16      repeat       66.4
   16   alternate       69.7
   32      repeat      114.9
   32   alternate      117.7
exit_status=0
```

#### `lastkey-bisect/fixes_r0.txt`

```
side=fixes gpu=3 started=2026-11-24T20:04:26+08:00 load=1.48 3.11 4.29
mode=last-key; host-only; 200000 calls × 9 batches; median ns/call
count     pattern    ns/call
    4      repeat       35.4
    4   alternate       37.6
   16      repeat       66.0
   16   alternate       69.4
   32      repeat      112.6
   32   alternate      129.3
exit_status=0
```

#### `lastkey-bisect/fixes_r1.txt`

```
side=fixes gpu=3 started=2026-11-24T20:04:55+08:00 load=1.63 2.99 4.21
mode=last-key; host-only; 200000 calls × 9 batches; median ns/call
count     pattern    ns/call
    4      repeat       37.0
    4   alternate       39.6
   16      repeat       66.1
   16   alternate       69.3
   32      repeat      111.9
   32   alternate      131.3
exit_status=0
```

#### `lastkey-bisect/fixes_r2.txt`

```
side=fixes gpu=3 started=2026-11-24T20:05:25+08:00 load=1.80 2.89 4.14
mode=last-key; host-only; 200000 calls × 9 batches; median ns/call
count     pattern    ns/call
    4      repeat       35.4
    4   alternate       37.6
   16      repeat       66.0
   16   alternate       69.6
   32      repeat      113.9
   32   alternate      130.5
exit_status=0
```

#### `lastkey-bisect/fixes_r3.txt`

```
side=fixes gpu=3 started=2026-11-24T20:05:54+08:00 load=1.69 2.76 4.05
mode=last-key; host-only; 200000 calls × 9 batches; median ns/call
count     pattern    ns/call
    4      repeat       35.4
    4   alternate       37.7
   16      repeat       66.0
   16   alternate       69.6
   32      repeat      114.3
   32   alternate      132.1
exit_status=0
```

#### `lastkey-bisect/wave1_r0.txt`

```
side=wave1 gpu=3 started=2026-11-24T20:04:05+08:00 load=1.47 3.22 4.35
mode=last-key; host-only; 200000 calls × 9 batches; median ns/call
count     pattern    ns/call
    4      repeat       35.6
    4   alternate       38.1
   16      repeat       66.5
   16   alternate       69.8
   32      repeat      114.0
   32   alternate      117.3
exit_status=0
```

#### `lastkey-bisect/wave1_r1.txt`

```
side=wave1 gpu=3 started=2026-11-24T20:04:45+08:00 load=1.55 3.02 4.23
mode=last-key; host-only; 200000 calls × 9 batches; median ns/call
count     pattern    ns/call
    4      repeat       35.4
    4   alternate       37.9
   16      repeat       66.4
   16   alternate       69.7
   32      repeat      112.9
   32   alternate      117.8
exit_status=0
```

#### `lastkey-bisect/wave1_r2.txt`

```
side=wave1 gpu=3 started=2026-11-24T20:05:15+08:00 load=1.60 2.89 4.15
mode=last-key; host-only; 200000 calls × 9 batches; median ns/call
count     pattern    ns/call
    4      repeat       35.3
    4   alternate       37.8
   16      repeat       66.5
   16   alternate       69.8
   32      repeat      131.3
   32   alternate      135.0
exit_status=0
```

#### `lastkey-bisect/wave1_r3.txt`

```
side=wave1 gpu=3 started=2026-11-24T20:05:44+08:00 load=1.72 2.80 4.08
mode=last-key; host-only; 200000 calls × 9 batches; median ns/call
count     pattern    ns/call
    4      repeat       35.4
    4   alternate       37.8
   16      repeat       66.5
   16   alternate       69.7
   32      repeat      113.1
   32   alternate      117.7
exit_status=0
```

#### `r1/fxown_r0.txt`

```
label=fxown tree=/tmp/intj-bench/fixes variant=/tmp/intj-bench/bis/v-fixes gpu=3 load=0.76 1.01 0.91
mode=last-key; host-only; 200000 calls × 9 batches; median ns/call
count     pattern    ns/call
    4      repeat       35.7
    4   alternate       37.9
   16      repeat       66.0
   16   alternate       69.5
   32      repeat      114.2
   32   alternate      131.0
exit_status=0
```

#### `r1/fxown_r1.txt`

```
label=fxown tree=/tmp/intj-bench/fixes variant=/tmp/intj-bench/bis/v-fixes gpu=3 load=1.03 1.04 0.92
mode=last-key; host-only; 200000 calls × 9 batches; median ns/call
count     pattern    ns/call
    4      repeat       35.4
    4   alternate       37.6
   16      repeat       66.0
   16   alternate       69.5
   32      repeat      113.4
   32   alternate      128.1
exit_status=0
```

#### `r1/fxown_r2.txt`

```
label=fxown tree=/tmp/intj-bench/fixes variant=/tmp/intj-bench/bis/v-fixes gpu=3 load=1.06 1.05 0.93
mode=last-key; host-only; 200000 calls × 9 batches; median ns/call
count     pattern    ns/call
    4      repeat       35.4
    4   alternate       37.6
   16      repeat       66.0
   16   alternate       69.4
   32      repeat      112.0
   32   alternate      131.7
exit_status=0
```

#### `r1/fxtree_r0.txt`

```
label=fxtree tree=/tmp/intj-bench/fixes variant= gpu=3 load=0.44 0.97 0.89
mode=last-key; host-only; 200000 calls × 9 batches; median ns/call
count     pattern    ns/call
    4      repeat       35.8
    4   alternate       38.2
   16      repeat       66.0
   16   alternate       69.5
   32      repeat      113.2
   32   alternate      130.9
exit_status=0
```

#### `r1/fxtree_r1.txt`

```
label=fxtree tree=/tmp/intj-bench/fixes variant= gpu=3 load=1.04 1.04 0.92
mode=last-key; host-only; 200000 calls × 9 batches; median ns/call
count     pattern    ns/call
    4      repeat       35.4
    4   alternate       37.6
   16      repeat       66.1
   16   alternate       69.3
   32      repeat      113.0
   32   alternate      130.0
exit_status=0
```

#### `r1/fxtree_r2.txt`

```
label=fxtree tree=/tmp/intj-bench/fixes variant= gpu=3 load=1.08 1.05 0.93
mode=last-key; host-only; 200000 calls × 9 batches; median ns/call
count     pattern    ns/call
    4      repeat       35.4
    4   alternate       37.5
   16      repeat       66.7
   16   alternate       70.1
   32      repeat      112.8
   32   alternate      131.0
exit_status=0
```

#### `r1/fxw1src_r0.txt`

```
label=fxw1src tree=/tmp/intj-bench/fixes variant=/tmp/intj-bench/bis/v-wave1 gpu=3 load=0.92 1.02 0.91
mode=last-key; host-only; 200000 calls × 9 batches; median ns/call
count     pattern    ns/call
    4      repeat       35.8
    4   alternate       38.0
   16      repeat       65.9
   16   alternate       69.4
   32      repeat      114.1
   32   alternate      117.7
exit_status=0
```

#### `r1/fxw1src_r1.txt`

```
label=fxw1src tree=/tmp/intj-bench/fixes variant=/tmp/intj-bench/bis/v-wave1 gpu=3 load=1.10 1.05 0.93
mode=last-key; host-only; 200000 calls × 9 batches; median ns/call
count     pattern    ns/call
    4      repeat       35.4
    4   alternate       37.5
   16      repeat       66.0
   16   alternate       69.4
   32      repeat      113.1
   32   alternate      117.2
exit_status=0
```

#### `r1/fxw1src_r2.txt`

```
label=fxw1src tree=/tmp/intj-bench/fixes variant=/tmp/intj-bench/bis/v-wave1 gpu=3 load=1.05 1.04 0.93
mode=last-key; host-only; 200000 calls × 9 batches; median ns/call
count     pattern    ns/call
    4      repeat       35.4
    4   alternate       37.5
   16      repeat       66.2
   16   alternate       69.7
   32      repeat      114.7
   32   alternate      117.8
exit_status=0
```

#### `r1/w1tree_r0.txt`

```
label=w1tree tree=/tmp/intj-bench/wave1 variant= gpu=3 load=0.22 0.97 0.89
mode=last-key; host-only; 200000 calls × 9 batches; median ns/call
count     pattern    ns/call
    4      repeat       35.6
    4   alternate       38.1
   16      repeat       66.6
   16   alternate       69.8
   32      repeat      114.3
   32   alternate      117.9
exit_status=0
```

#### `r1/w1tree_r1.txt`

```
label=w1tree tree=/tmp/intj-bench/wave1 variant= gpu=3 load=0.95 1.02 0.92
mode=last-key; host-only; 200000 calls × 9 batches; median ns/call
count     pattern    ns/call
    4      repeat       35.4
    4   alternate       37.8
   16      repeat       66.5
   16   alternate       69.8
   32      repeat      113.9
   32   alternate      117.3
exit_status=0
```

#### `r1/w1tree_r2.txt`

```
label=w1tree tree=/tmp/intj-bench/wave1 variant= gpu=3 load=1.09 1.05 0.93
mode=last-key; host-only; 200000 calls × 9 batches; median ns/call
count     pattern    ns/call
    4      repeat       35.3
    4   alternate       37.8
   16      repeat       66.5
   16   alternate       69.8
   32      repeat      114.5
   32   alternate      118.3
exit_status=0
```

#### `r2/ALL_r0.txt`

```
label=ALL tree=/tmp/intj-bench/fixes variant=/tmp/intj-bench/bis/v-ALL gpu=6 load=1.86 1.22 1.01
mode=last-key; host-only; 200000 calls × 9 batches; median ns/call
count     pattern    ns/call
    4      repeat       35.7
    4   alternate       37.9
   16      repeat       66.1
   16   alternate       69.4
   32      repeat      117.4
   32   alternate      126.0
exit_status=0
```

#### `r2/ALL_r1.txt`

```
label=ALL tree=/tmp/intj-bench/fixes variant=/tmp/intj-bench/bis/v-ALL gpu=6 load=1.64 1.29 1.05
mode=last-key; host-only; 200000 calls × 9 batches; median ns/call
count     pattern    ns/call
    4      repeat       35.5
    4   alternate       37.6
   16      repeat       66.0
   16   alternate       69.5
   32      repeat      117.9
   32   alternate      124.0
exit_status=0
```

#### `r2/ALL_r2.txt`

```
label=ALL tree=/tmp/intj-bench/fixes variant=/tmp/intj-bench/bis/v-ALL gpu=6 load=1.60 1.34 1.08
mode=last-key; host-only; 200000 calls × 9 batches; median ns/call
count     pattern    ns/call
    4      repeat       35.4
    4   alternate       37.7
   16      repeat       66.0
   16   alternate       69.5
   32      repeat      117.0
   32   alternate      124.7
exit_status=0
```

#### `r2/RA_r0.txt`

```
label=RA tree=/tmp/intj-bench/fixes variant=/tmp/intj-bench/bis/v-RA gpu=6 load=0.72 0.92 0.90
mode=last-key; host-only; 200000 calls × 9 batches; median ns/call
count     pattern    ns/call
    4      repeat       35.8
    4   alternate       38.0
   16      repeat       66.0
   16   alternate       69.5
   32      repeat      115.5
   32   alternate      123.6
exit_status=0
```

#### `r2/RA_r1.txt`

```
label=RA tree=/tmp/intj-bench/fixes variant=/tmp/intj-bench/bis/v-RA gpu=6 load=1.52 1.21 1.01
mode=last-key; host-only; 200000 calls × 9 batches; median ns/call
count     pattern    ns/call
    4      repeat       35.5
    4   alternate       37.5
   16      repeat       65.9
   16   alternate       69.5
   32      repeat      116.2
   32   alternate      122.2
exit_status=0
```

#### `r2/RA_r2.txt`

```
label=RA tree=/tmp/intj-bench/fixes variant=/tmp/intj-bench/bis/v-RA gpu=6 load=1.62 1.31 1.06
mode=last-key; host-only; 200000 calls × 9 batches; median ns/call
count     pattern    ns/call
    4      repeat       35.4
    4   alternate       37.5
   16      repeat       66.0
   16   alternate       69.4
   32      repeat      116.3
   32   alternate      122.7
exit_status=0
```

#### `r2/RB_r0.txt`

```
label=RB tree=/tmp/intj-bench/fixes variant=/tmp/intj-bench/bis/v-RB gpu=6 load=1.03 0.97 0.92
mode=last-key; host-only; 200000 calls × 9 batches; median ns/call
count     pattern    ns/call
    4      repeat       35.7
    4   alternate       38.0
   16      repeat       66.1
   16   alternate       69.5
   32      repeat      115.1
   32   alternate      131.5
exit_status=0
```

#### `r2/RB_r1.txt`

```
label=RB tree=/tmp/intj-bench/fixes variant=/tmp/intj-bench/bis/v-RB gpu=6 load=1.66 1.25 1.03
mode=last-key; host-only; 200000 calls × 9 batches; median ns/call
count     pattern    ns/call
    4      repeat       35.4
    4   alternate       37.6
   16      repeat       66.0
   16   alternate       69.5
   32      repeat      114.2
   32   alternate      129.4
exit_status=0
```

#### `r2/RB_r2.txt`

```
label=RB tree=/tmp/intj-bench/fixes variant=/tmp/intj-bench/bis/v-RB gpu=6 load=1.67 1.33 1.07
mode=last-key; host-only; 200000 calls × 9 batches; median ns/call
count     pattern    ns/call
    4      repeat       35.4
    4   alternate       37.6
   16      repeat       66.7
   16   alternate       70.2
   32      repeat      114.6
   32   alternate      130.7
exit_status=0
```

#### `r2/RC_r0.txt`

```
label=RC tree=/tmp/intj-bench/fixes variant=/tmp/intj-bench/bis/v-RC gpu=6 load=1.36 1.06 0.95
mode=last-key; host-only; 200000 calls × 9 batches; median ns/call
count     pattern    ns/call
    4      repeat       35.8
    4   alternate       38.1
   16      repeat       66.7
   16   alternate       70.1
   32      repeat      113.2
   32   alternate      131.6
exit_status=0
```

#### `r2/RC_r1.txt`

```
label=RC tree=/tmp/intj-bench/fixes variant=/tmp/intj-bench/bis/v-RC gpu=6 load=1.79 1.29 1.05
mode=last-key; host-only; 200000 calls × 9 batches; median ns/call
count     pattern    ns/call
    4      repeat       35.4
    4   alternate       37.6
   16      repeat       66.7
   16   alternate       70.1
   32      repeat      113.7
   32   alternate      124.9
exit_status=0
```

#### `r2/RC_r2.txt`

```
label=RC tree=/tmp/intj-bench/fixes variant=/tmp/intj-bench/bis/v-RC gpu=6 load=1.62 1.33 1.07
mode=last-key; host-only; 200000 calls × 9 batches; median ns/call
count     pattern    ns/call
    4      repeat       35.4
    4   alternate       37.6
   16      repeat       66.0
   16   alternate       69.4
   32      repeat      113.7
   32   alternate      131.0
exit_status=0
```

#### `r2/RD_r0.txt`

```
label=RD tree=/tmp/intj-bench/fixes variant=/tmp/intj-bench/bis/v-RD gpu=6 load=1.70 1.15 0.98
mode=last-key; host-only; 200000 calls × 9 batches; median ns/call
count     pattern    ns/call
    4      repeat       35.8
    4   alternate       38.1
   16      repeat       65.9
   16   alternate       69.4
   32      repeat      113.0
   32   alternate      128.0
exit_status=0
```

#### `r2/RD_r1.txt`

```
label=RD tree=/tmp/intj-bench/fixes variant=/tmp/intj-bench/bis/v-RD gpu=6 load=1.67 1.28 1.04
mode=last-key; host-only; 200000 calls × 9 batches; median ns/call
count     pattern    ns/call
    4      repeat       35.5
    4   alternate       37.6
   16      repeat       66.2
   16   alternate       69.3
   32      repeat      113.1
   32   alternate      132.8
exit_status=0
```

#### `r2/RD_r2.txt`

```
label=RD tree=/tmp/intj-bench/fixes variant=/tmp/intj-bench/bis/v-RD gpu=6 load=1.52 1.32 1.07
mode=last-key; host-only; 200000 calls × 9 batches; median ns/call
count     pattern    ns/call
    4      repeat       35.4
    4   alternate       37.7
   16      repeat       66.1
   16   alternate       69.6
   32      repeat      113.2
   32   alternate      131.6
exit_status=0
```

#### `r2/fxown_r0.txt`

```
label=fxown tree=/tmp/intj-bench/fixes variant=/tmp/intj-bench/bis/v-fixes gpu=6 load=0.67 0.91 0.90
mode=last-key; host-only; 200000 calls × 9 batches; median ns/call
count     pattern    ns/call
    4      repeat       35.4
    4   alternate       57.2
   16      repeat      112.5
   16   alternate       69.4
   32      repeat      113.2
   32   alternate      125.2
exit_status=0
```

#### `r2/fxown_r1.txt`

```
label=fxown tree=/tmp/intj-bench/fixes variant=/tmp/intj-bench/bis/v-fixes gpu=6 load=1.52 1.20 1.01
mode=last-key; host-only; 200000 calls × 9 batches; median ns/call
count     pattern    ns/call
    4      repeat       35.5
    4   alternate       37.5
   16      repeat       66.0
   16   alternate       69.3
   32      repeat      113.7
   32   alternate      128.5
exit_status=0
```

#### `r2/fxown_r2.txt`

```
label=fxown tree=/tmp/intj-bench/fixes variant=/tmp/intj-bench/bis/v-fixes gpu=6 load=1.46 1.27 1.05
mode=last-key; host-only; 200000 calls × 9 batches; median ns/call
count     pattern    ns/call
    4      repeat       35.4
    4   alternate       37.5
   16      repeat       66.1
   16   alternate       69.3
   32      repeat      113.2
   32   alternate      131.4
exit_status=0
```

#### `r2/w1src_r0.txt`

```
label=w1src tree=/tmp/intj-bench/fixes variant=/tmp/intj-bench/bis/v-wave1 gpu=6 load=0.54 0.91 0.89
mode=last-key; host-only; 200000 calls × 9 batches; median ns/call
count     pattern    ns/call
    4      repeat       35.7
    4   alternate       38.0
   16      repeat       66.0
   16   alternate       69.4
   32      repeat      114.9
   32   alternate      118.2
exit_status=0
```

#### `r2/w1src_r1.txt`

```
label=w1src tree=/tmp/intj-bench/fixes variant=/tmp/intj-bench/bis/v-wave1 gpu=6 load=1.62 1.21 1.01
mode=last-key; host-only; 200000 calls × 9 batches; median ns/call
count     pattern    ns/call
    4      repeat       35.8
    4   alternate       37.7
   16      repeat       66.0
   16   alternate       69.4
   32      repeat      113.5
   32   alternate      117.3
exit_status=0
```

#### `r2/w1src_r2.txt`

```
label=w1src tree=/tmp/intj-bench/fixes variant=/tmp/intj-bench/bis/v-wave1 gpu=6 load=1.54 1.28 1.05
mode=last-key; host-only; 200000 calls × 9 batches; median ns/call
count     pattern    ns/call
    4      repeat       35.4
    4   alternate       37.6
   16      repeat       66.1
   16   alternate       69.4
   32      repeat      113.1
   32   alternate      117.2
exit_status=0
```

#### `r3/FXDH_r0.txt`

```
label=FXDH tree=/tmp/intj-bench/fixes variant=/tmp/intj-bench/bis/v-FXDH gpu=5 load=1.18 1.18 1.06
mode=last-key; host-only; 200000 calls × 9 batches; median ns/call
count     pattern    ns/call
    4      repeat       35.8
    4   alternate       38.2
   16      repeat       66.1
   16   alternate       69.5
   32      repeat      113.3
   32   alternate      129.1
exit_status=0
```

#### `r3/FXDH_r1.txt`

```
label=FXDH tree=/tmp/intj-bench/fixes variant=/tmp/intj-bench/bis/v-FXDH gpu=5 load=1.08 1.15 1.06
mode=last-key; host-only; 200000 calls × 9 batches; median ns/call
count     pattern    ns/call
    4      repeat       35.4
    4   alternate       37.6
   16      repeat       66.1
   16   alternate       69.3
   32      repeat      112.4
   32   alternate      129.9
exit_status=0
```

#### `r3/FXDH_r2.txt`

```
label=FXDH tree=/tmp/intj-bench/fixes variant=/tmp/intj-bench/bis/v-FXDH gpu=5 load=8.95 3.39 1.83
mode=last-key; host-only; 200000 calls × 9 batches; median ns/call
count     pattern    ns/call
    4      repeat       35.5
    4   alternate       37.6
   16      repeat       66.0
   16   alternate       69.5
   32      repeat      113.3
   32   alternate      131.1
exit_status=0
```

#### `r3/NP_r0.txt`

```
label=NP tree=/tmp/intj-bench/fixes variant=/tmp/intj-bench/bis/v-NP gpu=5 load=0.83 1.14 1.04
mode=last-key; host-only; 200000 calls × 9 batches; median ns/call
count     pattern    ns/call
    4      repeat       35.8
    4   alternate       38.0
   16      repeat       66.0
   16   alternate       69.4
   32      repeat      119.1
   32   alternate      123.7
exit_status=0
```

#### `r3/NP_r1.txt`

```
label=NP tree=/tmp/intj-bench/fixes variant=/tmp/intj-bench/bis/v-NP gpu=5 load=1.16 1.17 1.06
mode=last-key; host-only; 200000 calls × 9 batches; median ns/call
count     pattern    ns/call
    4      repeat       35.4
    4   alternate       37.6
   16      repeat       66.0
   16   alternate       69.4
   32      repeat      117.9
   32   alternate      122.6
exit_status=0
```

#### `r3/NP_r2.txt`

```
label=NP tree=/tmp/intj-bench/fixes variant=/tmp/intj-bench/bis/v-NP gpu=5 load=1.19 1.17 1.06
mode=last-key; host-only; 200000 calls × 9 batches; median ns/call
count     pattern    ns/call
    4      repeat       35.4
    4   alternate       37.6
   16      repeat       66.0
   16   alternate       69.4
   32      repeat      119.5
   32   alternate      123.0
exit_status=0
```

#### `r3/W1DH_r0.txt`

```
label=W1DH tree=/tmp/intj-bench/fixes variant=/tmp/intj-bench/bis/v-W1DH gpu=5 load=1.02 1.15 1.04
mode=last-key; host-only; 200000 calls × 9 batches; median ns/call
count     pattern    ns/call
    4      repeat       35.7
    4   alternate       38.1
   16      repeat       65.9
   16   alternate       69.4
   32      repeat      114.1
   32   alternate      118.1
exit_status=0
```

#### `r3/W1DH_r1.txt`

```
label=W1DH tree=/tmp/intj-bench/fixes variant=/tmp/intj-bench/bis/v-W1DH gpu=5 load=1.11 1.16 1.06
mode=last-key; host-only; 200000 calls × 9 batches; median ns/call
count     pattern    ns/call
    4      repeat       35.4
    4   alternate       37.7
   16      repeat       66.1
   16   alternate       69.5
   32      repeat      113.8
   32   alternate      117.7
exit_status=0
```

#### `r3/W1DH_r2.txt`

```
label=W1DH tree=/tmp/intj-bench/fixes variant=/tmp/intj-bench/bis/v-W1DH gpu=5 load=11.59 3.47 1.82
mode=last-key; host-only; 200000 calls × 9 batches; median ns/call
count     pattern    ns/call
    4      repeat       35.4
    4   alternate       37.6
   16      repeat       66.0
   16   alternate       69.5
   32      repeat      113.7
   32   alternate      116.7
exit_status=0
```

#### `r3/W1NP_r0.txt`

```
label=W1NP tree=/tmp/intj-bench/fixes variant=/tmp/intj-bench/bis/v-W1NP gpu=5 load=1.03 1.17 1.05
mode=last-key; host-only; 200000 calls × 9 batches; median ns/call
count     pattern    ns/call
    4      repeat       35.8
    4   alternate       38.0
   16      repeat       65.9
   16   alternate       69.4
   32      repeat      123.8
   32   alternate      129.1
exit_status=0
```

#### `r3/W1NP_r1.txt`

```
label=W1NP tree=/tmp/intj-bench/fixes variant=/tmp/intj-bench/bis/v-W1NP gpu=5 load=1.13 1.17 1.06
mode=last-key; host-only; 200000 calls × 9 batches; median ns/call
count     pattern    ns/call
    4      repeat       35.4
    4   alternate       37.5
   16      repeat       66.1
   16   alternate       69.5
   32      repeat      123.9
   32   alternate      128.9
exit_status=0
```

#### `r3/W1NP_r2.txt`

```
label=W1NP tree=/tmp/intj-bench/fixes variant=/tmp/intj-bench/bis/v-W1NP gpu=5 load=1.32 1.20 1.07
mode=last-key; host-only; 200000 calls × 9 batches; median ns/call
count     pattern    ns/call
    4      repeat       35.4
    4   alternate       37.6
   16      repeat       66.1
   16   alternate       69.5
   32      repeat      124.4
   32   alternate      128.8
exit_status=0
```

#### `r3/W1REFS_r0.txt`

```
label=W1REFS tree=/tmp/intj-bench/fixes variant=/tmp/intj-bench/bis/v-W1REFS gpu=5 load=1.08 1.16 1.05
mode=last-key; host-only; 200000 calls × 9 batches; median ns/call
count     pattern    ns/call
    4      repeat       35.8
    4   alternate       38.1
   16      repeat       66.0
   16   alternate       69.4
   32      repeat      113.8
   32   alternate      117.8
exit_status=0
```

#### `r3/W1REFS_r1.txt`

```
label=W1REFS tree=/tmp/intj-bench/fixes variant=/tmp/intj-bench/bis/v-W1REFS gpu=5 load=1.09 1.16 1.06
mode=last-key; host-only; 200000 calls × 9 batches; median ns/call
count     pattern    ns/call
    4      repeat       35.4
    4   alternate       37.5
   16      repeat       66.0
   16   alternate       69.5
   32      repeat      113.6
   32   alternate      117.1
exit_status=0
```

#### `r3/W1REFS_r2.txt`

```
label=W1REFS tree=/tmp/intj-bench/fixes variant=/tmp/intj-bench/bis/v-W1REFS gpu=5 load=10.12 3.43 1.82
mode=last-key; host-only; 200000 calls × 9 batches; median ns/call
count     pattern    ns/call
    4      repeat       35.5
    4   alternate       37.6
   16      repeat       66.0
   16   alternate       69.5
   32      repeat      114.7
   32   alternate      118.0
exit_status=0
```

#### `r3/fxown_r0.txt`

```
label=fxown tree=/tmp/intj-bench/fixes variant=/tmp/intj-bench/bis/v-fixes gpu=5 load=0.79 1.15 1.04
mode=last-key; host-only; 200000 calls × 9 batches; median ns/call
count     pattern    ns/call
    4      repeat       35.4
    4   alternate       37.6
   16      repeat       66.0
   16   alternate       69.4
   32      repeat      113.0
   32   alternate      131.9
exit_status=0
```

#### `r3/fxown_r1.txt`

```
label=fxown tree=/tmp/intj-bench/fixes variant=/tmp/intj-bench/bis/v-fixes gpu=5 load=1.18 1.18 1.06
mode=last-key; host-only; 200000 calls × 9 batches; median ns/call
count     pattern    ns/call
    4      repeat       35.4
    4   alternate       37.6
   16      repeat       66.0
   16   alternate       69.4
   32      repeat      112.8
   32   alternate      128.7
exit_status=0
```

#### `r3/fxown_r2.txt`

```
label=fxown tree=/tmp/intj-bench/fixes variant=/tmp/intj-bench/bis/v-fixes gpu=5 load=1.13 1.16 1.06
mode=last-key; host-only; 200000 calls × 9 batches; median ns/call
count     pattern    ns/call
    4      repeat       35.4
    4   alternate       37.6
   16      repeat       66.1
   16   alternate       69.5
   32      repeat      113.5
   32   alternate      131.4
exit_status=0
```

#### `r3/w1src_r0.txt`

```
label=w1src tree=/tmp/intj-bench/fixes variant=/tmp/intj-bench/bis/v-wave1 gpu=5 load=0.75 1.15 1.04
mode=last-key; host-only; 200000 calls × 9 batches; median ns/call
count     pattern    ns/call
    4      repeat       35.4
    4   alternate       37.5
   16      repeat       66.0
   16   alternate       69.4
   32      repeat      114.0
   32   alternate      117.7
exit_status=0
```

#### `r3/w1src_r1.txt`

```
label=w1src tree=/tmp/intj-bench/fixes variant=/tmp/intj-bench/bis/v-wave1 gpu=5 load=1.13 1.17 1.06
mode=last-key; host-only; 200000 calls × 9 batches; median ns/call
count     pattern    ns/call
    4      repeat       35.4
    4   alternate       37.5
   16      repeat       66.0
   16   alternate       69.3
   32      repeat      113.8
   32   alternate      117.2
exit_status=0
```

#### `r3/w1src_r2.txt`

```
label=w1src tree=/tmp/intj-bench/fixes variant=/tmp/intj-bench/bis/v-wave1 gpu=5 load=1.07 1.14 1.05
mode=last-key; host-only; 200000 calls × 9 batches; median ns/call
count     pattern    ns/call
    4      repeat       35.4
    4   alternate       37.6
   16      repeat       66.0
   16   alternate       69.4
   32      repeat      112.9
   32   alternate      118.5
exit_status=0
```

#### `r4/LO_r0.txt`

```
label=LO tree=/tmp/intj-bench/fixes variant=/tmp/intj-bench/bis/v-LO gpu=7 load=5.28 3.15 1.81
mode=last-key; host-only; 200000 calls × 9 batches; median ns/call
count     pattern    ns/call
    4      repeat       35.8
    4   alternate       37.9
   16      repeat       66.1
   16   alternate       69.4
   32      repeat      125.1
   32   alternate      131.7
exit_status=0
```

#### `r4/LO_r1.txt`

```
label=LO tree=/tmp/intj-bench/fixes variant=/tmp/intj-bench/bis/v-LO gpu=7 load=2.65 2.79 1.77
mode=last-key; host-only; 200000 calls × 9 batches; median ns/call
count     pattern    ns/call
    4      repeat       35.4
    4   alternate       37.7
   16      repeat       66.0
   16   alternate       69.4
   32      repeat      124.6
   32   alternate      130.8
exit_status=0
```

#### `r4/LO_r2.txt`

```
label=LO tree=/tmp/intj-bench/fixes variant=/tmp/intj-bench/bis/v-LO gpu=7 load=1.96 2.59 1.75
mode=last-key; host-only; 200000 calls × 9 batches; median ns/call
count     pattern    ns/call
    4      repeat       35.4
    4   alternate       37.6
   16      repeat       66.0
   16   alternate       69.4
   32      repeat      124.9
   32   alternate      131.7
exit_status=0
```

#### `r4/W1LO_r0.txt`

```
label=W1LO tree=/tmp/intj-bench/fixes variant=/tmp/intj-bench/bis/v-W1LO gpu=7 load=4.21 3.05 1.81
mode=last-key; host-only; 200000 calls × 9 batches; median ns/call
count     pattern    ns/call
    4      repeat       35.7
    4   alternate       38.0
   16      repeat       66.1
   16   alternate       69.3
   32      repeat      116.3
   32   alternate      122.0
exit_status=0
```

#### `r4/W1LO_r1.txt`

```
label=W1LO tree=/tmp/intj-bench/fixes variant=/tmp/intj-bench/bis/v-W1LO gpu=7 load=2.39 2.73 1.76
mode=last-key; host-only; 200000 calls × 9 batches; median ns/call
count     pattern    ns/call
    4      repeat       35.4
    4   alternate       37.6
   16      repeat       66.7
   16   alternate       70.1
   32      repeat      117.1
   32   alternate      121.4
exit_status=0
```

#### `r4/W1LO_r2.txt`

```
label=W1LO tree=/tmp/intj-bench/fixes variant=/tmp/intj-bench/bis/v-W1LO gpu=7 load=1.81 2.54 1.74
mode=last-key; host-only; 200000 calls × 9 batches; median ns/call
count     pattern    ns/call
    4      repeat       35.5
    4   alternate       37.6
   16      repeat       66.1
   16   alternate       69.4
   32      repeat      116.3
   32   alternate      122.0
exit_status=0
```

#### `r4/fxown_r0.txt`

```
label=fxown tree=/tmp/intj-bench/fixes variant=/tmp/intj-bench/bis/v-fixes gpu=7 load=6.06 3.23 1.82
mode=last-key; host-only; 200000 calls × 9 batches; median ns/call
count     pattern    ns/call
    4      repeat       35.4
    4   alternate       37.5
   16      repeat       66.0
   16   alternate       69.4
   32      repeat      113.9
   32   alternate      130.6
exit_status=0
```

#### `r4/fxown_r1.txt`

```
label=fxown tree=/tmp/intj-bench/fixes variant=/tmp/intj-bench/bis/v-fixes gpu=7 load=2.94 2.85 1.78
mode=last-key; host-only; 200000 calls × 9 batches; median ns/call
count     pattern    ns/call
    4      repeat       35.4
    4   alternate       37.6
   16      repeat       65.9
   16   alternate       69.5
   32      repeat      113.7
   32   alternate      130.6
exit_status=0
```

#### `r4/fxown_r2.txt`

```
label=fxown tree=/tmp/intj-bench/fixes variant=/tmp/intj-bench/bis/v-fixes gpu=7 load=2.14 2.65 1.76
mode=last-key; host-only; 200000 calls × 9 batches; median ns/call
count     pattern    ns/call
    4      repeat       35.3
    4   alternate       37.6
   16      repeat       65.9
   16   alternate       69.4
   32      repeat      113.6
   32   alternate      129.6
exit_status=0
```

#### `r4/w1src_r0.txt`

```
label=w1src tree=/tmp/intj-bench/fixes variant=/tmp/intj-bench/bis/v-wave1 gpu=7 load=6.89 3.29 1.83
mode=last-key; host-only; 200000 calls × 9 batches; median ns/call
count     pattern    ns/call
    4      repeat       35.4
    4   alternate       37.6
   16      repeat       66.0
   16   alternate       69.4
   32      repeat      113.8
   32   alternate      117.0
exit_status=0
```

#### `r4/w1src_r1.txt`

```
label=w1src tree=/tmp/intj-bench/fixes variant=/tmp/intj-bench/bis/v-wave1 gpu=7 load=3.30 2.91 1.79
mode=last-key; host-only; 200000 calls × 9 batches; median ns/call
count     pattern    ns/call
    4      repeat       35.4
    4   alternate       37.6
   16      repeat       66.0
   16   alternate       69.4
   32      repeat      113.3
   32   alternate      117.7
exit_status=0
```

#### `r4/w1src_r2.txt`

```
label=w1src tree=/tmp/intj-bench/fixes variant=/tmp/intj-bench/bis/v-wave1 gpu=7 load=2.26 2.69 1.76
mode=last-key; host-only; 200000 calls × 9 batches; median ns/call
count     pattern    ns/call
    4      repeat       35.4
    4   alternate       37.7
   16      repeat       66.0
   16   alternate       69.5
   32      repeat      113.7
   32   alternate      117.8
exit_status=0
```

#### `align-ab/fixes_align32_r0.txt`

```
side=fixes variant=align32 flags=-falign-functions=32 -falign-loops=32 gpu=0 load=1.30 1.92 3.47
mode=last-key; host-only; 200000 calls × 9 batches; median ns/call
count     pattern    ns/call
    4      repeat       35.7
    4   alternate       38.0
   16      repeat       66.6
   16   alternate       70.1
   32      repeat      129.9
   32   alternate      136.9
exit_status=0
```

#### `align-ab/fixes_align32_r1.txt`

```
side=fixes variant=align32 flags=-falign-functions=32 -falign-loops=32 gpu=0 load=1.40 1.83 3.33
mode=last-key; host-only; 200000 calls × 9 batches; median ns/call
count     pattern    ns/call
    4      repeat       35.4
    4   alternate       37.7
   16      repeat       66.5
   16   alternate       70.1
   32      repeat      127.4
   32   alternate      136.6
exit_status=0
```

#### `align-ab/fixes_align32_r2.txt`

```
side=fixes variant=align32 flags=-falign-functions=32 -falign-loops=32 gpu=0 load=1.37 1.72 3.20
mode=last-key; host-only; 200000 calls × 9 batches; median ns/call
count     pattern    ns/call
    4      repeat       35.4
    4   alternate       37.6
   16      repeat       66.6
   16   alternate       70.2
   32      repeat      127.6
   32   alternate      136.0
exit_status=0
```

#### `align-ab/fixes_align64_r0.txt`

```
side=fixes variant=align64 flags=-falign-functions=64 -falign-jumps=64 -falign-loops=64 gpu=0 load=0.95 1.96 3.56
mode=last-key; host-only; 200000 calls × 9 batches; median ns/call
count     pattern    ns/call
    4      repeat       35.9
    4   alternate       38.1
   16      repeat       67.4
   16   alternate       70.9
   32      repeat      119.8
   32   alternate      122.3
exit_status=0
```

#### `align-ab/fixes_align64_r1.txt`

```
side=fixes variant=align64 flags=-falign-functions=64 -falign-jumps=64 -falign-loops=64 gpu=0 load=1.42 1.85 3.36
mode=last-key; host-only; 200000 calls × 9 batches; median ns/call
count     pattern    ns/call
    4      repeat       35.7
    4   alternate       37.7
   16      repeat       67.8
   16   alternate       71.1
   32      repeat      119.2
   32   alternate      123.2
exit_status=0
```

#### `align-ab/fixes_align64_r2.txt`

```
side=fixes variant=align64 flags=-falign-functions=64 -falign-jumps=64 -falign-loops=64 gpu=0 load=1.20 1.72 3.23
mode=last-key; host-only; 200000 calls × 9 batches; median ns/call
count     pattern    ns/call
    4      repeat       35.7
    4   alternate       37.8
   16      repeat       67.6
   16   alternate       71.0
   32      repeat      117.5
   32   alternate      122.2
exit_status=0
```

#### `align-ab/fixes_plain_r0.txt`

```
side=fixes variant=plain flags= gpu=0 load=0.66 2.06 3.66
mode=last-key; host-only; 200000 calls × 9 batches; median ns/call
count     pattern    ns/call
    4      repeat       35.8
    4   alternate       38.1
   16      repeat       66.7
   16   alternate       70.2
   32      repeat      113.5
   32   alternate      132.3
exit_status=0
```

#### `align-ab/fixes_plain_r1.txt`

```
side=fixes variant=plain flags= gpu=0 load=1.32 1.86 3.40
mode=last-key; host-only; 200000 calls × 9 batches; median ns/call
count     pattern    ns/call
    4      repeat       35.4
    4   alternate       37.6
   16      repeat       66.1
   16   alternate       69.4
   32      repeat      113.2
   32   alternate      129.2
exit_status=0
```

#### `align-ab/fixes_plain_r2.txt`

```
side=fixes variant=plain flags= gpu=0 load=1.28 1.77 3.28
mode=last-key; host-only; 200000 calls × 9 batches; median ns/call
count     pattern    ns/call
    4      repeat       35.5
    4   alternate       37.6
   16      repeat       66.8
   16   alternate       70.3
   32      repeat      113.1
   32   alternate      132.2
exit_status=0
```

#### `align-ab/wave1_align32_r0.txt`

```
side=wave1 variant=align32 flags=-falign-functions=32 -falign-loops=32 gpu=0 load=1.23 1.96 3.52
mode=last-key; host-only; 200000 calls × 9 batches; median ns/call
count     pattern    ns/call
    4      repeat       35.6
    4   alternate       38.2
   16      repeat       67.0
   16   alternate       70.0
   32      repeat      113.9
   32   alternate      118.5
exit_status=0
```

#### `align-ab/wave1_align32_r1.txt`

```
side=wave1 variant=align32 flags=-falign-functions=32 -falign-loops=32 gpu=0 load=1.47 1.86 3.35
mode=last-key; host-only; 200000 calls × 9 batches; median ns/call
count     pattern    ns/call
    4      repeat       35.4
    4   alternate       38.0
   16      repeat       66.9
   16   alternate       70.1
   32      repeat      113.4
   32   alternate      118.4
exit_status=0
```

#### `align-ab/wave1_align32_r2.txt`

```
side=wave1 variant=align32 flags=-falign-functions=32 -falign-loops=32 gpu=0 load=1.17 1.70 3.20
mode=last-key; host-only; 200000 calls × 9 batches; median ns/call
count     pattern    ns/call
    4      repeat       35.3
    4   alternate       37.9
   16      repeat       66.9
   16   alternate       70.0
   32      repeat      114.1
   32   alternate      119.0
exit_status=0
```

#### `align-ab/wave1_align64_r0.txt`

```
side=wave1 variant=align64 flags=-falign-functions=64 -falign-jumps=64 -falign-loops=64 gpu=0 load=0.83 2.01 3.61
mode=last-key; host-only; 200000 calls × 9 batches; median ns/call
count     pattern    ns/call
    4      repeat       35.9
    4   alternate       38.2
   16      repeat       66.3
   16   alternate       69.4
   32      repeat      115.2
   32   alternate      119.2
exit_status=0
```

#### `align-ab/wave1_align64_r1.txt`

```
side=wave1 variant=align64 flags=-falign-functions=64 -falign-jumps=64 -falign-loops=64 gpu=0 load=1.50 1.88 3.39
mode=last-key; host-only; 200000 calls × 9 batches; median ns/call
count     pattern    ns/call
    4      repeat       35.5
    4   alternate       37.8
   16      repeat       66.4
   16   alternate       69.5
   32      repeat      146.2
   32   alternate      149.9
exit_status=0
```

#### `align-ab/wave1_align64_r2.txt`

```
side=wave1 variant=align64 flags=-falign-functions=64 -falign-jumps=64 -falign-loops=64 gpu=0 load=1.24 1.75 3.25
mode=last-key; host-only; 200000 calls × 9 batches; median ns/call
count     pattern    ns/call
    4      repeat       35.6
    4   alternate       37.8
   16      repeat       66.4
   16   alternate       69.5
   32      repeat      114.9
   32   alternate      119.1
exit_status=0
```

#### `align-ab/wave1_plain_r0.txt`

```
side=wave1 variant=plain flags= gpu=0 load=0.53 2.14 3.72
mode=last-key; host-only; 200000 calls × 9 batches; median ns/call
count     pattern    ns/call
    4      repeat       35.7
    4   alternate       38.3
   16      repeat       66.6
   16   alternate       70.1
   32      repeat      113.3
   32   alternate      117.9
exit_status=0
```

#### `align-ab/wave1_plain_r1.txt`

```
side=wave1 variant=plain flags= gpu=0 load=1.37 1.89 3.42
mode=last-key; host-only; 200000 calls × 9 batches; median ns/call
count     pattern    ns/call
    4      repeat       35.4
    4   alternate       37.8
   16      repeat       66.4
   16   alternate       69.8
   32      repeat      115.1
   32   alternate      118.2
exit_status=0
```

#### `align-ab/wave1_plain_r2.txt`

```
side=wave1 variant=plain flags= gpu=0 load=1.34 1.80 3.30
mode=last-key; host-only; 200000 calls × 9 batches; median ns/call
count     pattern    ns/call
    4      repeat       35.3
    4   alternate       37.8
   16      repeat       66.5
   16   alternate       69.8
   32      repeat      114.2
   32   alternate      118.1
exit_status=0
```


# Compile-options refactor: `acbd9ad` vs `87e8a54` (index 0)

`87e8a54` moved compile-option merging and the knob table into `intj/compile_options.py`. It runs only at
build and cache-miss time and leaves the rendered C unchanged (every `RenderKey` a test run builds keeps its
digest), so this A/B checks that the launch rows did not move.

## Environment

- Baseline `acbd9ad` and candidate `87e8a54` as `git archive` exports in `/tmp/intj-bench/<commit>` (no local
  changes). Every raw file records its commit and `/proc/loadavg`.
- Xeon Platinum 8480C on `taskset -c 0` (core 0 idle before), MI308X (`gfx942`) GPU 1 via
  `HIP_VISIBLE_DEVICES=1` (GPU 0 was 100 % busy with another user's job; GPU 1 was 0 %). uv venv with Python
  3.12.3, Torch `2.14.0.dev20260804+rocm7.1`, Triton `3.8.0`, default `TRITON_HOME` (warm cache). Machine load
  average 2.9 at the start and end, from other users on its 224 cores.
- 2026-10-29, 16:51 to 16:59 +08:00. CUDA: untested.

## Command

`/tmp/intj-bench/copts_ab.sh`, the driver of
[`2026-10-29_arch-refactor-ab_0_20773a3.md`](2026-10-29_arch-refactor-ab_0_20773a3.md) with the commits
replaced: 3 rounds; each round runs every case on `acbd9ad`, then every case on `87e8a54`. Every process:
`cd <tree>; HIP_VISIBLE_DEVICES=1 PYTHONPATH=$PWD taskset -c 0 <venv>/bin/python <args>`.

```
launch_gpu      benchmarks/bench_launch.py --iters 1000 --batches 9
launch_host     benchmarks/bench_launch.py --no-gpu --iters 100000 --batches 9
launch_readme   benchmarks/bench_launch.py --readme --iters 1000 --batches 9
launch_sweep    benchmarks/bench_launch.py --sweep --iters 100000 --batches 9
launch_last_key benchmarks/bench_launch.py --last-key --iters 100000 --batches 9
launch_tuned    benchmarks/bench_launch.py --tuned --iters 20000 --batches 9
launch_dynamic  benchmarks/bench_launch.py --dynamic --iters 20000 --batches 9
```

Compared with `/tmp/intj-bench/copts_ab_summary.py` (`bench_compare.parse_file` and its regression rule:
slower by more than 3 % and by more than 2 ns, or 0.1 us), median of the 3 process medians. All 42 files have
`exit_status=0`. `bench_launch.py` prints each process's median of its 9 batches, not per-batch samples.

## Result

**No regression; 0 of 53 rows flagged.** `87e8a54` against `acbd9ad`; a positive change is slower:

| case | rows | Δ % range |
|---|---:|---|
| `launch_host` | 12 | -0.4 .. +0.9 |
| `launch_sweep` | 6 | +0.0 .. +1.1 |
| `launch_last_key` | 6 | -0.1 .. +0.8 |
| `launch_readme` | 7 | -0.6 .. +1.5 |
| `launch_dynamic` | 8 | -1.5 .. +0.8 |
| `launch_tuned` | 2 | -1.2 .. +0.1 |
| `launch_gpu` | 12 | -0.8 .. +0.4 |

Host-only rows moved by at most 0.4 ns (`launch_host` bound pointer) and 1.4 ns (`launch_sweep` 32 tensor),
within run-to-run spread; the GPU enqueue rows (about 3.1 us) by at most 47 ns, in both directions.

## Raw output

### Comparison (`acbd9ad` = baseline, `87e8a54` = new)

| case | table | row | col | unit | acbd9ad | 87e8a54 | delta | pct | flag |
|---|---|---|---|---|---:|---:|---:|---:|---|
| launch_dynamic | dynamic | 2 dynamic options | launch | ns | 3154.80 | 3108.30 | -46.50 | -1.5% |  |
| launch_dynamic | dynamic | 2 dynamic options | spec_key | ns | 99.10 | 98.70 | -0.40 | -0.4% |  |
| launch_dynamic | dynamic | int constexpr | launch | ns | 3058.10 | 3057.30 | -0.80 | -0.0% |  |
| launch_dynamic | dynamic | int constexpr | spec_key | ns | 86.70 | 86.70 | +0.00 | +0.0% |  |
| launch_dynamic | dynamic | plain | launch | ns | 3095.60 | 3111.80 | +16.20 | +0.5% |  |
| launch_dynamic | dynamic | plain | spec_key | ns | 93.90 | 94.00 | +0.10 | +0.1% |  |
| launch_dynamic | dynamic | str constexpr | launch | ns | 3067.90 | 3067.70 | -0.20 | -0.0% |  |
| launch_dynamic | dynamic | str constexpr | spec_key | ns | 86.50 | 87.20 | +0.70 | +0.8% |  |
| launch_gpu | launch | auto map | ns/call | ns | 3292.60 | 3286.10 | -6.50 | -0.2% |  |
| launch_gpu | launch | baked | ns/call | ns | 3140.60 | 3129.80 | -10.80 | -0.3% |  |
| launch_gpu | launch | bound pointer | ns/call | ns | 3144.50 | 3120.50 | -24.00 | -0.8% |  |
| launch_gpu | launch | bound tensor | ns/call | ns | 3127.80 | 3137.40 | +9.60 | +0.3% |  |
| launch_gpu | launch | fixed device map | ns/call | ns | 3129.00 | 3132.80 | +3.80 | +0.1% |  |
| launch_gpu | launch | fixed device no-map | ns/call | ns | 3123.90 | 3117.10 | -6.80 | -0.2% |  |
| launch_gpu | launch | generic | ns/call | ns | 3242.70 | 3238.20 | -4.50 | -0.1% |  |
| launch_gpu | launch | generic tensor ann | ns/call | ns | 3193.50 | 3206.20 | +12.70 | +0.4% |  |
| launch_gpu | launch | reduced key | ns/call | ns | 3135.00 | 3141.60 | +6.60 | +0.2% |  |
| launch_gpu | launch | tensor annotated | ns/call | ns | 3363.40 | 3376.90 | +13.50 | +0.4% |  |
| launch_gpu | launch | verify off | ns/call | ns | 3157.80 | 3144.60 | -13.20 | -0.4% |  |
| launch_gpu | launch | verify on | ns/call | ns | 3175.10 | 3158.00 | -17.10 | -0.5% |  |
| launch_host | launch | auto map | ns/call | ns | 44.00 | 44.10 | +0.10 | +0.2% |  |
| launch_host | launch | baked | ns/call | ns | 40.80 | 40.70 | -0.10 | -0.2% |  |
| launch_host | launch | bound pointer | ns/call | ns | 44.90 | 45.30 | +0.40 | +0.9% |  |
| launch_host | launch | bound tensor | ns/call | ns | 46.50 | 46.60 | +0.10 | +0.2% |  |
| launch_host | launch | fixed device map | ns/call | ns | 45.40 | 45.20 | -0.20 | -0.4% |  |
| launch_host | launch | fixed device no-map | ns/call | ns | 44.60 | 44.70 | +0.10 | +0.2% |  |
| launch_host | launch | generic | ns/call | ns | 46.90 | 47.00 | +0.10 | +0.2% |  |
| launch_host | launch | generic tensor ann | ns/call | ns | 47.10 | 47.20 | +0.10 | +0.2% |  |
| launch_host | launch | reduced key | ns/call | ns | 44.10 | 44.10 | +0.00 | +0.0% |  |
| launch_host | launch | tensor annotated | ns/call | ns | 43.60 | 43.60 | +0.00 | +0.0% |  |
| launch_host | launch | verify off | ns/call | ns | 44.20 | 44.40 | +0.20 | +0.5% |  |
| launch_host | launch | verify on | ns/call | ns | 45.40 | 45.30 | -0.10 | -0.2% |  |
| launch_last_key | sweep | 16 alternate | ns/call | ns | 69.80 | 69.70 | -0.10 | -0.1% |  |
| launch_last_key | sweep | 16 repeat | ns/call | ns | 66.40 | 66.40 | +0.00 | +0.0% |  |
| launch_last_key | sweep | 32 alternate | ns/call | ns | 116.70 | 116.90 | +0.20 | +0.2% |  |
| launch_last_key | sweep | 32 repeat | ns/call | ns | 112.30 | 113.20 | +0.90 | +0.8% |  |
| launch_last_key | sweep | 4 alternate | ns/call | ns | 37.80 | 37.80 | +0.00 | +0.0% |  |
| launch_last_key | sweep | 4 repeat | ns/call | ns | 35.40 | 35.40 | +0.00 | +0.0% |  |
| launch_readme | readme_path | grid=(0,) intj | us | us | 0.03 | 0.03 | +0.00 | +0.0% |  |
| launch_readme | readme_path | grid=(0,) triton | us | us | 12.52 | 12.68 | +0.16 | +1.3% |  |
| launch_readme | readme_path | grid=(1,) intj | us | us | 3.36 | 3.34 | -0.02 | -0.6% |  |
| launch_readme | readme_path | grid=(1,) triton | us | us | 16.26 | 16.45 | +0.19 | +1.2% |  |
| launch_readme | readme_torch_access | interpreter | decode ns | ns | 897.30 | 910.40 | +13.10 | +1.5% |  |
| launch_readme | readme_torch_access | runtime_shim | decode ns | ns | 90.90 | 90.80 | -0.10 | -0.1% |  |
| launch_readme | readme_torch_access | static_compile | decode ns | ns | 94.70 | 94.10 | -0.60 | -0.6% |  |
| launch_sweep | sweep | 16 int | ns/call | ns | 72.70 | 72.70 | +0.00 | +0.0% |  |
| launch_sweep | sweep | 16 tensor | ns/call | ns | 68.00 | 68.00 | +0.00 | +0.0% |  |
| launch_sweep | sweep | 32 int | ns/call | ns | 119.60 | 120.00 | +0.40 | +0.3% |  |
| launch_sweep | sweep | 32 tensor | ns/call | ns | 125.50 | 126.90 | +1.40 | +1.1% |  |
| launch_sweep | sweep | 4 int | ns/call | ns | 41.70 | 41.70 | +0.00 | +0.0% |  |
| launch_sweep | sweep | 4 tensor | ns/call | ns | 40.50 | 40.50 | +0.00 | +0.0% |  |
| launch_tuned | tuned | plain | ns/launch | ns | 3130.70 | 3093.40 | -37.30 | -1.2% |  |
| launch_tuned | tuned | tuned hit | ns/launch | ns | 3121.20 | 3125.20 | +4.00 | +0.1% |  |

rows=53 flagged=0

### Per-process output

#### `acbd9ad_r0_launch_gpu.txt`

```
commit=acbd9ad (git archive export)
started=2026-10-29T16:51:23+08:00 load=2.93 6.77 8.14
command=HIP_VISIBLE_DEVICES=1 PYTHONPATH=$PWD taskset -c 0 /mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/venv/bin/python benchmarks/bench_launch.py --iters 1000 --batches 9
mode=gpu; 1000 calls × 9 batches; median ns/call
               path    ns/call       Δ ns       Δ %   build ms   bind ms
           auto map     3327.1       +0.0      +0.0      18.09         -
   tensor annotated     3378.8      +51.7      +1.6       3.57         -
            generic     3266.3      -60.9      -1.8       2.92         -
 generic tensor ann     3266.3      -60.9      -1.8       2.86         -
        reduced key     3184.6     -142.5      -4.3       2.81         -
         verify off     3158.7     -168.5      -5.1       2.96         -
          verify on     3175.1     -152.1      -4.6       2.79         -
              baked     3157.0     -170.1      -5.1       2.77         -
       bound tensor     3139.2     -187.9      -5.6       2.80      0.03
      bound pointer     3144.5     -182.6      -5.5       2.76      0.03
   fixed device map     3141.7     -185.4      -5.6       2.68      0.03
fixed device no-map     3123.9     -203.3      -6.1       2.75      0.03
elapsed_ns=10667937883
exit_status=0
```

#### `acbd9ad_r0_launch_host.txt`

```
commit=acbd9ad (git archive export)
started=2026-10-29T16:51:34+08:00 load=2.78 6.61 8.07
command=HIP_VISIBLE_DEVICES=1 PYTHONPATH=$PWD taskset -c 0 /mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/venv/bin/python benchmarks/bench_launch.py --no-gpu --iters 100000 --batches 9
mode=host-only; 100000 calls × 9 batches; median ns/call
               path    ns/call       Δ ns       Δ %   build ms   bind ms
           auto map       44.0       +0.0      +0.0       7.93         -
   tensor annotated       43.7       -0.4      -0.8       2.27         -
            generic       46.9       +2.9      +6.5       2.13         -
 generic tensor ann       47.1       +3.1      +7.0       2.12         -
        reduced key       44.1       +0.0      +0.1       2.13         -
         verify off       44.2       +0.2      +0.5       2.14         -
          verify on       45.4       +1.3      +3.0       2.22         -
              baked       40.8       -3.3      -7.4       2.14         -
       bound tensor       81.8      +37.8     +85.8       2.15      0.03
      bound pointer       45.3       +1.3      +2.9       2.16      0.03
   fixed device map       45.4       +1.4      +3.1       2.08      0.02
fixed device no-map       45.3       +1.2      +2.8       2.16      0.02
elapsed_ns=9553575584
exit_status=0
```

#### `acbd9ad_r0_launch_readme.txt`

```
commit=acbd9ad (git archive export)
started=2026-10-29T16:51:43+08:00 load=2.74 6.48 8.01
command=HIP_VISIBLE_DEVICES=1 PYTHONPATH=$PWD taskset -c 0 /mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/venv/bin/python benchmarks/bench_launch.py --readme --iters 1000 --batches 9
mode=readme; 1000 calls × 9 batches; median
        path   triton us    intj us   speedup
   grid=(1,)       16.16       3.35      4.8x
   grid=(0,)       12.52       0.03    457.8x

torch_access_mode   decode ns    build s
     runtime_shim        90.0       0.02
   static_compile        92.6       0.00
      interpreter       891.6       0.00
elapsed_ns=10230565959
exit_status=0
```

#### `acbd9ad_r0_launch_sweep.txt`

```
commit=acbd9ad (git archive export)
started=2026-10-29T16:51:54+08:00 load=2.86 6.38 7.97
command=HIP_VISIBLE_DEVICES=1 PYTHONPATH=$PWD taskset -c 0 /mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/venv/bin/python benchmarks/bench_launch.py --sweep --iters 100000 --batches 9
mode=sweep; host-only; 100000 calls × 9 batches; median ns/call
count   kind    ns/call
    4    int       41.7
    4 tensor       40.5
   16    int       73.0
   16 tensor       67.9
   32    int      119.5
   32 tensor      126.8
elapsed_ns=9444499956
exit_status=0
```

#### `acbd9ad_r0_launch_last_key.txt`

```
commit=acbd9ad (git archive export)
started=2026-10-29T16:52:03+08:00 load=2.73 6.23 7.90
command=HIP_VISIBLE_DEVICES=1 PYTHONPATH=$PWD taskset -c 0 /mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/venv/bin/python benchmarks/bench_launch.py --last-key --iters 100000 --batches 9
mode=last-key; host-only; 200000 calls × 9 batches; median ns/call
count     pattern    ns/call
    4      repeat       35.3
    4   alternate       37.7
   16      repeat       66.4
   16   alternate       69.9
   32      repeat      112.3
   32   alternate      116.7
elapsed_ns=9785434768
exit_status=0
```

#### `acbd9ad_r0_launch_tuned.txt`

```
commit=acbd9ad (git archive export)
started=2026-10-29T16:52:13+08:00 load=2.62 6.09 7.84
command=HIP_VISIBLE_DEVICES=1 PYTHONPATH=$PWD taskset -c 0 /mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/venv/bin/python benchmarks/bench_launch.py --tuned --iters 20000 --batches 9
 tuned hit:  3121.2 ns/launch
     plain:  3085.3 ns/launch
elapsed_ns=11060865582
exit_status=0
```

#### `acbd9ad_r0_launch_dynamic.txt`

```
commit=acbd9ad (git archive export)
started=2026-10-29T16:52:24+08:00 load=2.52 5.96 7.77
command=HIP_VISIBLE_DEVICES=1 PYTHONPATH=$PWD taskset -c 0 /mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/venv/bin/python benchmarks/bench_launch.py --dynamic --iters 20000 --batches 9
mode=dynamic; 20000 calls × 9 batches; median ns/call
               path     launch   spec_key
              plain     3095.6       90.0
  2 dynamic options     3154.8       98.0
      int constexpr     3095.8       87.7
      str constexpr     3056.8       86.5
elapsed_ns=12229331468
exit_status=0
```

#### `acbd9ad_r1_launch_gpu.txt`

```
commit=acbd9ad (git archive export)
started=2026-10-29T16:53:49+08:00 load=2.41 5.08 7.30
command=HIP_VISIBLE_DEVICES=1 PYTHONPATH=$PWD taskset -c 0 /mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/venv/bin/python benchmarks/bench_launch.py --iters 1000 --batches 9
mode=gpu; 1000 calls × 9 batches; median ns/call
               path    ns/call       Δ ns       Δ %   build ms   bind ms
           auto map     3292.6       +0.0      +0.0      18.37         -
   tensor annotated     3363.4      +70.8      +2.2       3.56         -
            generic     3242.7      -49.8      -1.5       2.92         -
 generic tensor ann     3193.5      -99.1      -3.0       2.82         -
        reduced key     3135.0     -157.5      -4.8       2.85         -
         verify off     3157.8     -134.8      -4.1       2.98         -
          verify on     3203.9      -88.7      -2.7       2.87         -
              baked     3140.6     -152.0      -4.6       2.82         -
       bound tensor     3127.8     -164.8      -5.0       2.79      0.03
      bound pointer     3192.2     -100.4      -3.0       2.76      0.03
   fixed device map     3129.0     -163.6      -5.0       2.70      0.03
fixed device no-map     3200.6      -92.0      -2.8       2.75      0.03
elapsed_ns=10304448427
exit_status=0
```

#### `acbd9ad_r1_launch_host.txt`

```
commit=acbd9ad (git archive export)
started=2026-10-29T16:53:59+08:00 load=2.35 4.97 7.24
command=HIP_VISIBLE_DEVICES=1 PYTHONPATH=$PWD taskset -c 0 /mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/venv/bin/python benchmarks/bench_launch.py --no-gpu --iters 100000 --batches 9
mode=host-only; 100000 calls × 9 batches; median ns/call
               path    ns/call       Δ ns       Δ %   build ms   bind ms
           auto map       44.0       +0.0      +0.0       6.89         -
   tensor annotated       43.6       -0.4      -0.8       2.21         -
            generic       47.0       +3.0      +6.8       2.11         -
 generic tensor ann       47.1       +3.1      +7.0       2.10         -
        reduced key       44.1       +0.1      +0.3       2.12         -
         verify off       44.2       +0.2      +0.3       2.10         -
          verify on       45.4       +1.4      +3.1       2.19         -
              baked       40.9       -3.1      -7.1       2.14         -
       bound tensor       46.3       +2.3      +5.3       2.10      0.02
      bound pointer       44.7       +0.6      +1.5       2.11      0.03
   fixed device map       45.1       +1.1      +2.6       2.08      0.02
fixed device no-map       44.5       +0.5      +1.2       2.14      0.02
elapsed_ns=9505597208
exit_status=0
```

#### `acbd9ad_r1_launch_readme.txt`

```
commit=acbd9ad (git archive export)
started=2026-10-29T16:54:09+08:00 load=2.29 4.88 7.19
command=HIP_VISIBLE_DEVICES=1 PYTHONPATH=$PWD taskset -c 0 /mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/venv/bin/python benchmarks/bench_launch.py --readme --iters 1000 --batches 9
mode=readme; 1000 calls × 9 batches; median
        path   triton us    intj us   speedup
   grid=(1,)       16.26       3.36      4.8x
   grid=(0,)       12.34       0.03    451.1x

torch_access_mode   decode ns    build s
     runtime_shim        90.9       0.02
   static_compile        95.4       0.00
      interpreter       897.3       0.00
elapsed_ns=10212856429
exit_status=0
```

#### `acbd9ad_r1_launch_sweep.txt`

```
commit=acbd9ad (git archive export)
started=2026-10-29T16:54:19+08:00 load=2.25 4.78 7.13
command=HIP_VISIBLE_DEVICES=1 PYTHONPATH=$PWD taskset -c 0 /mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/venv/bin/python benchmarks/bench_launch.py --sweep --iters 100000 --batches 9
mode=sweep; host-only; 100000 calls × 9 batches; median ns/call
count   kind    ns/call
    4    int       41.6
    4 tensor       40.6
   16    int       72.7
   16 tensor       68.0
   32    int      120.3
   32 tensor      125.2
elapsed_ns=9419349258
exit_status=0
```

#### `acbd9ad_r1_launch_last_key.txt`

```
commit=acbd9ad (git archive export)
started=2026-10-29T16:54:28+08:00 load=2.21 4.69 7.08
command=HIP_VISIBLE_DEVICES=1 PYTHONPATH=$PWD taskset -c 0 /mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/venv/bin/python benchmarks/bench_launch.py --last-key --iters 100000 --batches 9
mode=last-key; host-only; 200000 calls × 9 batches; median ns/call
count     pattern    ns/call
    4      repeat       35.4
    4   alternate       37.8
   16      repeat       66.4
   16   alternate       69.7
   32      repeat      112.1
   32   alternate      116.8
elapsed_ns=9769551654
exit_status=0
```

#### `acbd9ad_r1_launch_tuned.txt`

```
commit=acbd9ad (git archive export)
started=2026-10-29T16:54:38+08:00 load=2.25 4.62 7.03
command=HIP_VISIBLE_DEVICES=1 PYTHONPATH=$PWD taskset -c 0 /mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/venv/bin/python benchmarks/bench_launch.py --tuned --iters 20000 --batches 9
 tuned hit:  3104.4 ns/launch
     plain:  3130.7 ns/launch
elapsed_ns=11065744808
exit_status=0
```

#### `acbd9ad_r1_launch_dynamic.txt`

```
commit=acbd9ad (git archive export)
started=2026-10-29T16:54:49+08:00 load=2.21 4.53 6.97
command=HIP_VISIBLE_DEVICES=1 PYTHONPATH=$PWD taskset -c 0 /mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/venv/bin/python benchmarks/bench_launch.py --dynamic --iters 20000 --batches 9
mode=dynamic; 20000 calls × 9 batches; median ns/call
               path     launch   spec_key
              plain     3111.5       94.6
  2 dynamic options     3087.8       99.1
      int constexpr     3036.2       86.2
      str constexpr     3067.9       90.5
elapsed_ns=12201387613
exit_status=0
```

#### `acbd9ad_r2_launch_gpu.txt`

```
commit=acbd9ad (git archive export)
started=2026-10-29T16:56:14+08:00 load=2.56 4.05 6.59
command=HIP_VISIBLE_DEVICES=1 PYTHONPATH=$PWD taskset -c 0 /mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/venv/bin/python benchmarks/bench_launch.py --iters 1000 --batches 9
mode=gpu; 1000 calls × 9 batches; median ns/call
               path    ns/call       Δ ns       Δ %   build ms   bind ms
           auto map     3252.5       +0.0      +0.0      18.17         -
   tensor annotated     3307.9      +55.3      +1.7       3.57         -
            generic     3206.1      -46.4      -1.4       2.92         -
 generic tensor ann     3169.8      -82.7      -2.5       2.87         -
        reduced key     3133.7     -118.8      -3.7       2.86         -
         verify off     3108.8     -143.7      -4.4       2.98         -
          verify on     3103.9     -148.6      -4.6       2.82         -
              baked     3111.7     -140.9      -4.3       2.83         -
       bound tensor     3111.6     -140.9      -4.3       2.79      0.03
      bound pointer     3110.9     -141.6      -4.4       2.76      0.03
   fixed device map     3113.9     -138.7      -4.3       2.71      0.03
fixed device no-map     3105.9     -146.6      -4.5       2.75      0.03
elapsed_ns=10278590121
exit_status=0
```

#### `acbd9ad_r2_launch_host.txt`

```
commit=acbd9ad (git archive export)
started=2026-10-29T16:56:24+08:00 load=2.55 4.00 6.54
command=HIP_VISIBLE_DEVICES=1 PYTHONPATH=$PWD taskset -c 0 /mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/venv/bin/python benchmarks/bench_launch.py --no-gpu --iters 100000 --batches 9
mode=host-only; 100000 calls × 9 batches; median ns/call
               path    ns/call       Δ ns       Δ %   build ms   bind ms
           auto map       44.1       +0.0      +0.0       6.89         -
   tensor annotated       43.6       -0.4      -1.0       2.26         -
            generic       46.9       +2.8      +6.4       2.14         -
 generic tensor ann       47.1       +3.0      +6.9       2.13         -
        reduced key       44.1       -0.0      -0.0       2.11         -
         verify off       44.2       +0.1      +0.3       2.13         -
          verify on       45.6       +1.5      +3.3       2.82         -
              baked       40.7       -3.3      -7.6       2.18         -
       bound tensor       46.5       +2.4      +5.4       2.13      0.02
      bound pointer       44.9       +0.9      +1.9       2.14      0.03
   fixed device map       45.5       +1.4      +3.2       2.07      0.02
fixed device no-map       44.6       +0.5      +1.1       2.15      0.02
elapsed_ns=9492384146
exit_status=0
```

#### `acbd9ad_r2_launch_readme.txt`

```
commit=acbd9ad (git archive export)
started=2026-10-29T16:56:34+08:00 load=2.55 3.95 6.50
command=HIP_VISIBLE_DEVICES=1 PYTHONPATH=$PWD taskset -c 0 /mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/venv/bin/python benchmarks/bench_launch.py --readme --iters 1000 --batches 9
mode=readme; 1000 calls × 9 batches; median
        path   triton us    intj us   speedup
   grid=(1,)       16.34       3.37      4.8x
   grid=(0,)       12.68       0.03    461.1x

torch_access_mode   decode ns    build s
     runtime_shim        93.4       0.02
   static_compile        94.7       0.00
      interpreter       897.9       0.00
elapsed_ns=10208800613
exit_status=0
```

#### `acbd9ad_r2_launch_sweep.txt`

```
commit=acbd9ad (git archive export)
started=2026-10-29T16:56:44+08:00 load=2.54 3.90 6.45
command=HIP_VISIBLE_DEVICES=1 PYTHONPATH=$PWD taskset -c 0 /mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/venv/bin/python benchmarks/bench_launch.py --sweep --iters 100000 --batches 9
mode=sweep; host-only; 100000 calls × 9 batches; median ns/call
count   kind    ns/call
    4    int       41.7
    4 tensor       40.5
   16    int       72.7
   16 tensor       68.0
   32    int      119.6
   32 tensor      125.5
elapsed_ns=9418433750
exit_status=0
```

#### `acbd9ad_r2_launch_last_key.txt`

```
commit=acbd9ad (git archive export)
started=2026-10-29T16:56:53+08:00 load=2.45 3.84 6.41
command=HIP_VISIBLE_DEVICES=1 PYTHONPATH=$PWD taskset -c 0 /mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/venv/bin/python benchmarks/bench_launch.py --last-key --iters 100000 --batches 9
mode=last-key; host-only; 200000 calls × 9 batches; median ns/call
count     pattern    ns/call
    4      repeat       35.4
    4   alternate       37.8
   16      repeat       66.4
   16   alternate       69.8
   32      repeat      112.8
   32   alternate      116.6
elapsed_ns=9780014207
exit_status=0
```

#### `acbd9ad_r2_launch_tuned.txt`

```
commit=acbd9ad (git archive export)
started=2026-10-29T16:57:03+08:00 load=2.38 3.77 6.36
command=HIP_VISIBLE_DEVICES=1 PYTHONPATH=$PWD taskset -c 0 /mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/venv/bin/python benchmarks/bench_launch.py --tuned --iters 20000 --batches 9
 tuned hit:  3819.7 ns/launch
     plain:  3996.7 ns/launch
elapsed_ns=11350831093
exit_status=0
```

#### `acbd9ad_r2_launch_dynamic.txt`

```
commit=acbd9ad (git archive export)
started=2026-10-29T16:57:14+08:00 load=2.32 3.72 6.31
command=HIP_VISIBLE_DEVICES=1 PYTHONPATH=$PWD taskset -c 0 /mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/venv/bin/python benchmarks/bench_launch.py --dynamic --iters 20000 --batches 9
mode=dynamic; 20000 calls × 9 batches; median ns/call
               path     launch   spec_key
              plain     3081.4       93.9
  2 dynamic options     3157.3       99.5
      int constexpr     3058.1       86.7
      str constexpr     3154.9       86.2
elapsed_ns=12217390382
exit_status=0
```

#### `87e8a54_r0_launch_gpu.txt`

```
commit=87e8a54 (git archive export)
started=2026-10-29T16:52:36+08:00 load=2.63 5.81 7.70
command=HIP_VISIBLE_DEVICES=1 PYTHONPATH=$PWD taskset -c 0 /mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/venv/bin/python benchmarks/bench_launch.py --iters 1000 --batches 9
mode=gpu; 1000 calls × 9 batches; median ns/call
               path    ns/call       Δ ns       Δ %   build ms   bind ms
           auto map     3286.1       +0.0      +0.0      18.29         -
   tensor annotated     3376.9      +90.9      +2.8       3.57         -
            generic     3238.2      -47.8      -1.5       2.94         -
 generic tensor ann     3206.2      -79.9      -2.4       2.82         -
        reduced key     3141.6     -144.5      -4.4       2.82         -
         verify off     3144.6     -141.5      -4.3       3.01         -
          verify on     3158.0     -128.1      -3.9       2.86         -
              baked     3129.8     -156.3      -4.8       2.81         -
       bound tensor     3050.0     -236.1      -7.2       2.81      0.03
      bound pointer     3120.5     -165.6      -5.0       2.78      0.03
   fixed device map     3032.5     -253.6      -7.7       2.70      0.03
fixed device no-map     3106.1     -180.0      -5.5       2.79      0.03
elapsed_ns=10324487462
exit_status=0
```

#### `87e8a54_r0_launch_host.txt`

```
commit=87e8a54 (git archive export)
started=2026-10-29T16:52:47+08:00 load=2.61 5.70 7.64
command=HIP_VISIBLE_DEVICES=1 PYTHONPATH=$PWD taskset -c 0 /mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/venv/bin/python benchmarks/bench_launch.py --no-gpu --iters 100000 --batches 9
mode=host-only; 100000 calls × 9 batches; median ns/call
               path    ns/call       Δ ns       Δ %   build ms   bind ms
           auto map       44.1       +0.0      +0.0       6.84         -
   tensor annotated       43.6       -0.5      -1.2       2.27         -
            generic       47.1       +3.0      +6.7       2.14         -
 generic tensor ann       47.1       +3.0      +6.7       2.16         -
        reduced key       44.0       -0.1      -0.2       2.15         -
         verify off       44.9       +0.7      +1.6       2.24         -
          verify on       45.2       +1.1      +2.4       2.14         -
              baked       40.7       -3.4      -7.7       2.13         -
       bound tensor       46.6       +2.5      +5.6       2.15      0.03
      bound pointer       45.0       +0.8      +1.8       2.13      0.03
   fixed device map       45.2       +1.0      +2.3       2.11      0.02
fixed device no-map       44.5       +0.3      +0.7       2.12      0.02
elapsed_ns=9506179117
exit_status=0
```

#### `87e8a54_r0_launch_readme.txt`

```
commit=87e8a54 (git archive export)
started=2026-10-29T16:52:56+08:00 load=2.52 5.58 7.58
command=HIP_VISIBLE_DEVICES=1 PYTHONPATH=$PWD taskset -c 0 /mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/venv/bin/python benchmarks/bench_launch.py --readme --iters 1000 --batches 9
mode=readme; 1000 calls × 9 batches; median
        path   triton us    intj us   speedup
   grid=(1,)       16.45       3.33      4.9x
   grid=(0,)       12.68       0.03    465.6x

torch_access_mode   decode ns    build s
     runtime_shim        90.8       0.02
   static_compile        94.1       0.00
      interpreter      1020.3       0.00
elapsed_ns=10232646679
exit_status=0
```

#### `87e8a54_r0_launch_sweep.txt`

```
commit=87e8a54 (git archive export)
started=2026-10-29T16:53:06+08:00 load=2.52 5.48 7.53
command=HIP_VISIBLE_DEVICES=1 PYTHONPATH=$PWD taskset -c 0 /mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/venv/bin/python benchmarks/bench_launch.py --sweep --iters 100000 --batches 9
mode=sweep; host-only; 100000 calls × 9 batches; median ns/call
count   kind    ns/call
    4    int       41.6
    4 tensor       40.5
   16    int       72.6
   16 tensor       67.9
   32    int      118.5
   32 tensor      126.3
elapsed_ns=9436746567
exit_status=0
```

#### `87e8a54_r0_launch_last_key.txt`

```
commit=87e8a54 (git archive export)
started=2026-10-29T16:53:16+08:00 load=2.44 5.37 7.47
command=HIP_VISIBLE_DEVICES=1 PYTHONPATH=$PWD taskset -c 0 /mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/venv/bin/python benchmarks/bench_launch.py --last-key --iters 100000 --batches 9
mode=last-key; host-only; 200000 calls × 9 batches; median ns/call
count     pattern    ns/call
    4      repeat       35.3
    4   alternate       37.8
   16      repeat       66.4
   16   alternate       69.7
   32      repeat      112.2
   32   alternate      116.7
elapsed_ns=9793484339
exit_status=0
```

#### `87e8a54_r0_launch_tuned.txt`

```
commit=87e8a54 (git archive export)
started=2026-10-29T16:53:25+08:00 load=2.37 5.25 7.41
command=HIP_VISIBLE_DEVICES=1 PYTHONPATH=$PWD taskset -c 0 /mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/venv/bin/python benchmarks/bench_launch.py --tuned --iters 20000 --batches 9
 tuned hit:  3125.2 ns/launch
     plain:  3082.6 ns/launch
elapsed_ns=11083216677
exit_status=0
```

#### `87e8a54_r0_launch_dynamic.txt`

```
commit=87e8a54 (git archive export)
started=2026-10-29T16:53:37+08:00 load=2.39 5.16 7.35
command=HIP_VISIBLE_DEVICES=1 PYTHONPATH=$PWD taskset -c 0 /mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/venv/bin/python benchmarks/bench_launch.py --dynamic --iters 20000 --batches 9
mode=dynamic; 20000 calls × 9 batches; median ns/call
               path     launch   spec_key
              plain     3111.8       94.0
  2 dynamic options     3095.4       98.0
      int constexpr     3101.6       86.7
      str constexpr     3022.9       88.3
elapsed_ns=12224835089
exit_status=0
```

#### `87e8a54_r1_launch_gpu.txt`

```
commit=87e8a54 (git archive export)
started=2026-10-29T16:55:01+08:00 load=2.38 4.45 6.91
command=HIP_VISIBLE_DEVICES=1 PYTHONPATH=$PWD taskset -c 0 /mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/venv/bin/python benchmarks/bench_launch.py --iters 1000 --batches 9
mode=gpu; 1000 calls × 9 batches; median ns/call
               path    ns/call       Δ ns       Δ %   build ms   bind ms
           auto map     3889.5       +0.0      +0.0      18.17         -
   tensor annotated     3871.6      -17.9      -0.5       3.51         -
            generic     3840.7      -48.8      -1.3       3.02         -
 generic tensor ann     3776.8     -112.7      -2.9       3.12         -
        reduced key     3750.4     -139.1      -3.6       2.99         -
         verify off     3784.8     -104.6      -2.7       3.01         -
          verify on     3840.1      -49.3      -1.3       2.98         -
              baked     3856.3      -33.2      -0.9       2.98         -
       bound tensor     3691.6     -197.9      -5.1       2.93      0.04
      bound pointer     3842.9      -46.6      -1.2       2.95      0.03
   fixed device map     3735.4     -154.1      -4.0       2.85      0.03
fixed device no-map     3764.1     -125.4      -3.2       2.90      0.03
elapsed_ns=10364089003
exit_status=0
```

#### `87e8a54_r1_launch_host.txt`

```
commit=87e8a54 (git archive export)
started=2026-10-29T16:55:12+08:00 load=2.40 4.39 6.86
command=HIP_VISIBLE_DEVICES=1 PYTHONPATH=$PWD taskset -c 0 /mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/venv/bin/python benchmarks/bench_launch.py --no-gpu --iters 100000 --batches 9
mode=host-only; 100000 calls × 9 batches; median ns/call
               path    ns/call       Δ ns       Δ %   build ms   bind ms
           auto map       44.0       +0.0      +0.0       6.83         -
   tensor annotated       43.6       -0.4      -0.9       2.25         -
            generic       46.9       +2.9      +6.6       2.10         -
 generic tensor ann       47.2       +3.2      +7.2       2.11         -
        reduced key       44.1       +0.0      +0.1       2.11         -
         verify off       44.2       +0.2      +0.4       2.24         -
          verify on       45.5       +1.5      +3.3       2.16         -
              baked       40.7       -3.3      -7.5       2.15         -
       bound tensor       46.6       +2.5      +5.8       2.11      0.03
      bound pointer       45.5       +1.4      +3.3       2.15      0.03
   fixed device map       45.2       +1.2      +2.7       2.10      0.02
fixed device no-map       44.9       +0.8      +1.9       2.14      0.02
elapsed_ns=9502509402
exit_status=0
```

#### `87e8a54_r1_launch_readme.txt`

```
commit=87e8a54 (git archive export)
started=2026-10-29T16:55:21+08:00 load=2.34 4.31 6.81
command=HIP_VISIBLE_DEVICES=1 PYTHONPATH=$PWD taskset -c 0 /mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/venv/bin/python benchmarks/bench_launch.py --readme --iters 1000 --batches 9
mode=readme; 1000 calls × 9 batches; median
        path   triton us    intj us   speedup
   grid=(1,)       16.54       3.34      5.0x
   grid=(0,)       12.81       0.03    468.9x

torch_access_mode   decode ns    build s
     runtime_shim        89.6       0.02
   static_compile        94.1       0.00
      interpreter       890.2       0.00
elapsed_ns=10202616245
exit_status=0
```

#### `87e8a54_r1_launch_sweep.txt`

```
commit=87e8a54 (git archive export)
started=2026-10-29T16:55:31+08:00 load=2.37 4.25 6.76
command=HIP_VISIBLE_DEVICES=1 PYTHONPATH=$PWD taskset -c 0 /mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/venv/bin/python benchmarks/bench_launch.py --sweep --iters 100000 --batches 9
mode=sweep; host-only; 100000 calls × 9 batches; median ns/call
count   kind    ns/call
    4    int       41.7
    4 tensor       40.5
   16    int       72.7
   16 tensor       68.0
   32    int      121.1
   32 tensor      126.9
elapsed_ns=9423460295
exit_status=0
```

#### `87e8a54_r1_launch_last_key.txt`

```
commit=87e8a54 (git archive export)
started=2026-10-29T16:55:41+08:00 load=2.31 4.17 6.71
command=HIP_VISIBLE_DEVICES=1 PYTHONPATH=$PWD taskset -c 0 /mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/venv/bin/python benchmarks/bench_launch.py --last-key --iters 100000 --batches 9
mode=last-key; host-only; 200000 calls × 9 batches; median ns/call
count     pattern    ns/call
    4      repeat       35.5
    4   alternate       37.7
   16      repeat       66.4
   16   alternate       69.8
   32      repeat      113.4
   32   alternate      116.9
elapsed_ns=9785610122
exit_status=0
```

#### `87e8a54_r1_launch_tuned.txt`

```
commit=87e8a54 (git archive export)
started=2026-10-29T16:55:51+08:00 load=2.26 4.10 6.66
command=HIP_VISIBLE_DEVICES=1 PYTHONPATH=$PWD taskset -c 0 /mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/venv/bin/python benchmarks/bench_launch.py --tuned --iters 20000 --batches 9
 tuned hit:  3116.2 ns/launch
     plain:  3093.4 ns/launch
elapsed_ns=11047195565
exit_status=0
```

#### `87e8a54_r1_launch_dynamic.txt`

```
commit=87e8a54 (git archive export)
started=2026-10-29T16:56:02+08:00 load=2.38 4.07 6.62
command=HIP_VISIBLE_DEVICES=1 PYTHONPATH=$PWD taskset -c 0 /mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/venv/bin/python benchmarks/bench_launch.py --dynamic --iters 20000 --batches 9
mode=dynamic; 20000 calls × 9 batches; median ns/call
               path     launch   spec_key
              plain     3130.7       96.2
  2 dynamic options     3153.4       98.7
      int constexpr     3044.0       88.3
      str constexpr     3067.7       86.9
elapsed_ns=12224328905
exit_status=0
```

#### `87e8a54_r2_launch_gpu.txt`

```
commit=87e8a54 (git archive export)
started=2026-10-29T16:57:27+08:00 load=2.33 3.65 6.25
command=HIP_VISIBLE_DEVICES=1 PYTHONPATH=$PWD taskset -c 0 /mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/venv/bin/python benchmarks/bench_launch.py --iters 1000 --batches 9
mode=gpu; 1000 calls × 9 batches; median ns/call
               path    ns/call       Δ ns       Δ %   build ms   bind ms
           auto map     3248.0       +0.0      +0.0      18.59         -
   tensor annotated     3327.3      +79.3      +2.4       3.39         -
            generic     3203.0      -44.9      -1.4       2.94         -
 generic tensor ann     3158.7      -89.3      -2.7       2.98         -
        reduced key     3135.4     -112.6      -3.5       2.86         -
         verify off     3121.4     -126.6      -3.9       2.85         -
          verify on     3105.1     -142.9      -4.4       2.86         -
              baked     3091.1     -156.9      -4.8       2.82         -
       bound tensor     3137.4     -110.5      -3.4       2.78      0.03
      bound pointer     3112.6     -135.3      -4.2       2.77      0.03
   fixed device map     3132.8     -115.2      -3.5       2.74      0.03
fixed device no-map     3117.1     -130.9      -4.0       2.83      0.03
elapsed_ns=10301682313
exit_status=0
```

#### `87e8a54_r2_launch_host.txt`

```
commit=87e8a54 (git archive export)
started=2026-10-29T16:57:37+08:00 load=2.52 3.64 6.22
command=HIP_VISIBLE_DEVICES=1 PYTHONPATH=$PWD taskset -c 0 /mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/venv/bin/python benchmarks/bench_launch.py --no-gpu --iters 100000 --batches 9
mode=host-only; 100000 calls × 9 batches; median ns/call
               path    ns/call       Δ ns       Δ %   build ms   bind ms
           auto map       44.1       +0.0      +0.0       7.08         -
   tensor annotated       43.7       -0.5      -1.1       2.23         -
            generic       47.0       +2.8      +6.4       2.17         -
 generic tensor ann       47.2       +3.0      +6.8       2.11         -
        reduced key       44.1       -0.0      -0.1       2.16         -
         verify off       44.4       +0.2      +0.6       2.23         -
          verify on       45.3       +1.1      +2.6       2.18         -
              baked       40.8       -3.3      -7.5       2.14         -
       bound tensor       46.7       +2.6      +5.9       2.20      0.02
      bound pointer       45.3       +1.2      +2.6       2.15      0.03
   fixed device map       45.3       +1.2      +2.6       2.10      0.02
fixed device no-map       44.7       +0.6      +1.4       2.21      0.02
elapsed_ns=9877870521
exit_status=0
```

#### `87e8a54_r2_launch_readme.txt`

```
commit=87e8a54 (git archive export)
started=2026-10-29T16:57:47+08:00 load=2.52 3.60 6.18
command=HIP_VISIBLE_DEVICES=1 PYTHONPATH=$PWD taskset -c 0 /mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/venv/bin/python benchmarks/bench_launch.py --readme --iters 1000 --batches 9
mode=readme; 1000 calls × 9 batches; median
        path   triton us    intj us   speedup
   grid=(1,)       16.30       3.34      4.9x
   grid=(0,)       12.45       0.03    456.3x

torch_access_mode   decode ns    build s
     runtime_shim        93.2       0.02
   static_compile        95.9       0.00
      interpreter       910.4       0.00
elapsed_ns=10689779319
exit_status=0
```

#### `87e8a54_r2_launch_sweep.txt`

```
commit=87e8a54 (git archive export)
started=2026-10-29T16:57:58+08:00 load=2.68 3.60 6.15
command=HIP_VISIBLE_DEVICES=1 PYTHONPATH=$PWD taskset -c 0 /mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/venv/bin/python benchmarks/bench_launch.py --sweep --iters 100000 --batches 9
mode=sweep; host-only; 100000 calls × 9 batches; median ns/call
count   kind    ns/call
    4    int       41.8
    4 tensor       40.7
   16    int       72.8
   16 tensor       68.1
   32    int      120.0
   32 tensor      127.5
elapsed_ns=9949082287
exit_status=0
```

#### `87e8a54_r2_launch_last_key.txt`

```
commit=87e8a54 (git archive export)
started=2026-10-29T16:58:07+08:00 load=2.73 3.58 6.11
command=HIP_VISIBLE_DEVICES=1 PYTHONPATH=$PWD taskset -c 0 /mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/venv/bin/python benchmarks/bench_launch.py --last-key --iters 100000 --batches 9
mode=last-key; host-only; 200000 calls × 9 batches; median ns/call
count     pattern    ns/call
    4      repeat       35.4
    4   alternate       37.8
   16      repeat       66.4
   16   alternate       69.7
   32      repeat      113.2
   32   alternate      117.7
elapsed_ns=10250461030
exit_status=0
```

#### `87e8a54_r2_launch_tuned.txt`

```
commit=87e8a54 (git archive export)
started=2026-10-29T16:58:18+08:00 load=2.85 3.58 6.08
command=HIP_VISIBLE_DEVICES=1 PYTHONPATH=$PWD taskset -c 0 /mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/venv/bin/python benchmarks/bench_launch.py --tuned --iters 20000 --batches 9
 tuned hit:  3953.3 ns/launch
     plain:  3825.2 ns/launch
elapsed_ns=11881964804
exit_status=0
```

#### `87e8a54_r2_launch_dynamic.txt`

```
commit=87e8a54 (git archive export)
started=2026-10-29T16:58:30+08:00 load=2.87 3.56 6.05
command=HIP_VISIBLE_DEVICES=1 PYTHONPATH=$PWD taskset -c 0 /mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/venv/bin/python benchmarks/bench_launch.py --dynamic --iters 20000 --batches 9
mode=dynamic; 20000 calls × 9 batches; median ns/call
               path     launch   spec_key
              plain     3088.2       93.6
  2 dynamic options     3108.3       99.7
      int constexpr     3057.3       85.0
      str constexpr     3105.8       87.2
elapsed_ns=12875457965
exit_status=0
```

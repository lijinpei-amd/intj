# Latest develop remerge: paired launcher and FFI measurements

Measured 2026-09-25, local baseline `develop` at `ef698d2` and initially merged `torch_abi` at `2a9f891` (parents `9243a11`, `ef698d2`). All measured source checkouts were clean while timed. Three serial processes per case, nine warmed batches each; the large GPU sweep was repeated as three interleaved baseline/candidate pairs with 100 calls per batch.

## Median of three process medians

| Case | develop | merged | change |
| --- | ---: | ---: | ---: |
| host auto map (ns) | 44.200 | 43.900 | -0.7% |
| sweep 32 tensors (ns) | 119.500 | 115.200 | -3.6% |
| grid=(1,) INTJ (us) | 3.120 | 3.130 | +0.3% |
| same HSACO INTJ (us) | 2.956 | 2.990 | +1.2% |
| FFI paths host hot (ns) | 38.900 | 40.000 | +2.8% |
| FFI comparison INTJ empty (us) | 3.029 | 2.958 | -2.3% |
| long sweep 64 INTJ static (us) | 4.604 | 5.913 | +28.5% |
| long sweep 64 FFI GPU (us) | 5.051 | 5.824 | +15.3% |

The long 1,000-call FFI GPU sweep shows a shared queue-dependent increase in two merged processes: FFI GPU launch rows increased together with INTJ, while FFI host-only no-ops remained stable. One merged process had baseline-speed GPU launches. The repeated 100-call sweep below checks this effect using alternating checkout order and the same compiled launch implementations. With shorter batches, INTJ 64-argument static and runtime-shim rows differ by less than 1% between revision medians. These are host enqueue times, excluding GPU synchronization; long batches can saturate the queue. The `bench_ffi_compare.py` GPU binaries are different between FFI and INTJ; only compare the same row across revisions. `bench_hip_module_launch.py` uses the same HSACO/compiled function for its three paths. Keyword/dataclass adapter and cold callback rows in `bench_intj_ffi_paths.py` have distinct work boundaries.

## Final branch code: post-type-check repeat

After the initial measurements, `03a8f52` added postponed type annotations to the two new FFI benchmark scripts and a targeted type-check annotation. The executable launch paths are unchanged, but the affected scripts were measured again on this final source commit. Baseline `develop` remains `ef698d2`. Medians below are from three separate processes, with nine batches each and unchanged commands/settings.

| Affected case | develop | final merged | change |
| --- | ---: | ---: | ---: |
| FFI paths hot call (ns) | 38.900 | 39.800 | +2.3% |
| FFI paths direct positional (ns) | 66.200 | 66.500 | +0.5% |
| FFI comparison INTJ empty (µs) | 3.029 | 2.986 | -1.4% |
| long sweep 64 INTJ static (µs) | 4.604 | 4.608 | +0.1% |
| long sweep 64 FFI GPU (µs) | 5.051 | 5.069 | +0.4% |

The large apparent long-batch slowdown in intermediate `2a9f891` was absent in all three final-code repeats. Original and final raw outputs are preserved below; shorter interleaved sweeps remain a separate diagnostic.

## Environment and commands

- CPU: x86-64, pinned to core 0 with `taskset -c 0`; one process and one benchmark at a time on an otherwise idle GPU.
- GPU: AMD Instinct MI308X, `gfx942:sramecc+:xnack-`. Python 3.12.3; Torch 2.14.0+rocm7.2; Triton 3.8.0; TVM FFI 0.1.14.post2.dev1+g424558557.d20260924; Torch C++11 ABI enabled.
- Interpreter: `/tmp/gb2/bin/python`; `TRITON_HOME`, `CXX`, `CC`, and `INTJ_BENCHMARK_ROOT` unset, using project defaults and normal caches. Compilation, cache preparation, and argument construction are outside timers. GPU output checks are in the benchmark scripts.
- For `develop`, run from `/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj`; for merged, run from `/mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/torch_abi`.
- Prefix each command with `PYTHONPATH=$PWD taskset -c 0 /tmp/gb2/bin/python`. Run each command in a separate process, in the listed order, for rounds 0, 1, and 2.

```sh
benchmarks/bench_launch.py --no-gpu --iters 100000 --batches 9
benchmarks/bench_launch.py --sweep --no-gpu --iters 100000 --batches 9
benchmarks/bench_launch.py --readme --iters 1000 --batches 9
benchmarks/bench_hip_module_launch.py --iters 1000 --batches 9
benchmarks/bench_intj_ffi_paths.py --iters 1000 --batches 9
benchmarks/bench_ffi_compare.py --iters 1000 --batches 9
benchmarks/bench_ffi_compare.py --sweep --iters 1000 --batches 9
```

Controlled interleaved diagnostic: same prefix, run `benchmarks/bench_ffi_compare.py --sweep --iters 100 --batches 9` in order `develop, merged` for each of three rounds. This is a different call count and must not be compared directly to the 1,000-call sweep.

## Controlled 100-call sweep: selected GPU rows (µs/call)

| Row | develop process medians | merged process medians | median change |
| --- | --- | --- | ---: |
| 0 args, FFI GPU | 1.766, 2.719, 1.873 | 1.835, 1.846, 1.851 | -1.5% |
| 0 args, INTJ static | 2.859, 3.589, 2.916 | 2.996, 2.925, 2.962 | +1.6% |
| 0 args, INTJ runtime shim | 2.901, 3.592, 2.962 | 2.999, 2.962, 2.995 | +1.1% |
| 3 args, FFI GPU | 3.365, 4.152, 3.538 | 3.498, 3.415, 3.463 | -2.1% |
| 3 args, INTJ static | 3.113, 3.775, 3.075 | 3.085, 3.047, 3.159 | -0.9% |
| 3 args, INTJ runtime shim | 3.065, 3.870, 3.091 | 3.139, 3.158, 3.208 | +2.2% |
| 16 args, FFI GPU | 4.182, 6.158, 4.108 | 3.984, 4.091, 4.267 | -2.2% |
| 16 args, INTJ static | 3.536, 5.849, 3.495 | 3.486, 3.602, 3.607 | +1.9% |
| 16 args, INTJ runtime shim | 3.548, 5.642, 3.542 | 3.602, 3.572, 3.660 | +1.5% |
| 32 args, FFI GPU | 4.427, 6.282, 4.431 | 4.371, 4.417, 4.553 | -0.3% |
| 32 args, INTJ static | 3.813, 6.114, 3.810 | 3.755, 3.895, 3.861 | +1.3% |
| 32 args, INTJ runtime shim | 3.740, 5.846, 3.770 | 3.825, 3.928, 3.960 | +4.2% |
| 64 args, FFI GPU | 5.232, 6.750, 5.301 | 5.360, 5.354, 5.363 | +1.1% |
| 64 args, INTJ static | 4.764, 6.451, 4.839 | 4.860, 4.865, 5.043 | +0.5% |
| 64 args, INTJ runtime shim | 4.699, 6.394, 4.741 | 4.762, 4.755, 4.947 | +0.4% |

## Raw outputs (including batch samples and run metadata)

### host: develop

```text
commit=ef698d24dae8790fffdf84fe766a7bf06db428f6
started=2026-09-25T16:23:37+08:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --no-gpu --iters 100000 --batches 9
mode=host-only; 100000 calls × 9 batches; median ns/call
               path    ns/call       Δ ns       Δ %   build ms   bind ms
           auto map       44.2       +0.0      +0.0      59.12         -
        reduced key       43.5       -0.8      -1.7       1.31         -
         verify off       43.4       -0.8      -1.7       1.20         -
          verify on       44.1       -0.1      -0.3       1.15         -
              baked       39.5       -4.7     -10.6       1.30         -
       bound tensor      131.3      +87.0    +196.8       0.13      1.07
      bound pointer       44.8       +0.6      +1.3       0.11      1.04
   fixed device map       44.7       +0.5      +1.0       1.77      4.56
fixed device no-map       45.8       +1.6      +3.5       0.13      1.09
elapsed_ns=3227686019
exit_status=0
ended=2026-09-25T16:23:40+08:00
```

### host: merged

```text
commit=2a9f891443cf6763d26483c7b1f3d0f451d90a3c
started=2026-09-25T16:25:35+08:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/torch_abi
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/torch_abi taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --no-gpu --iters 100000 --batches 9
mode=host-only; 100000 calls × 9 batches; median ns/call
               path    ns/call       Δ ns       Δ %   build ms   bind ms
           auto map       44.0       +0.0      +0.0      60.15         -
        reduced key       43.5       -0.6      -1.3       1.53         -
         verify off       43.6       -0.5      -1.1       1.42         -
          verify on       44.4       +0.4      +0.8       1.35         -
              baked       39.9       -4.2      -9.5       2.57         -
       bound tensor       70.3      +26.2     +59.5       0.16      1.31
      bound pointer       45.1       +1.1      +2.5       0.16      1.37
   fixed device map       44.4       +0.3      +0.8       0.10      1.25
fixed device no-map       43.9       -0.2      -0.4       0.14      1.24
elapsed_ns=3008315357
exit_status=0
ended=2026-09-25T16:25:38+08:00
```

### sweep: develop

```text
commit=ef698d24dae8790fffdf84fe766a7bf06db428f6
started=2026-09-25T16:23:40+08:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --sweep --no-gpu --iters 100000 --batches 9
mode=sweep; host-only; 100000 calls × 9 batches; median ns/call
count   kind    ns/call
    4    int       41.6
    4 tensor       40.5
   16    int       74.2
   16 tensor       67.7
   32    int      118.4
   32 tensor      118.7
elapsed_ns=3005333222
exit_status=0
ended=2026-09-25T16:23:43+08:00
```

### sweep: merged

```text
commit=2a9f891443cf6763d26483c7b1f3d0f451d90a3c
started=2026-09-25T16:25:38+08:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/torch_abi
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/torch_abi taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --sweep --no-gpu --iters 100000 --batches 9
mode=sweep; host-only; 100000 calls × 9 batches; median ns/call
count   kind    ns/call
    4    int       42.5
    4 tensor       41.5
   16    int       74.0
   16 tensor       68.4
   32    int      120.3
   32 tensor      115.6
elapsed_ns=3011743732
exit_status=0
ended=2026-09-25T16:25:41+08:00
```

### readme: develop

```text
commit=ef698d24dae8790fffdf84fe766a7bf06db428f6
started=2026-09-25T16:23:43+08:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --readme --iters 1000 --batches 9
mode=readme; 1000 calls × 9 batches; median
        path   triton us    intj us   speedup
   grid=(1,)       16.62       3.12      5.3x
   grid=(0,)       13.19       0.03    468.5x

torch_access   decode ns    build s
        shim       199.7       0.03
         cxx        91.5       0.08
     cpython       871.2       0.00
elapsed_ns=3906653496
exit_status=0
ended=2026-09-25T16:23:47+08:00
```

### readme: merged

```text
commit=2a9f891443cf6763d26483c7b1f3d0f451d90a3c
started=2026-09-25T16:25:41+08:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/torch_abi
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/torch_abi taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --readme --iters 1000 --batches 9
mode=readme; 1000 calls × 9 batches; median
        path   triton us    intj us   speedup
   grid=(1,)       16.55       3.11      5.3x
   grid=(0,)       13.03       0.03    457.9x

torch_access_mode   decode ns    build s
     runtime_shim        92.2       0.02
   static_compile        91.2       0.06
      interpreter       872.9       0.00
elapsed_ns=3639515031
exit_status=0
ended=2026-09-25T16:25:45+08:00
```

### hip: develop

```text
commit=ef698d24dae8790fffdf84fe766a7bf06db428f6
started=2026-09-25T16:23:47+08:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_hip_module_launch.py --iters 1000 --batches 9
hsaco_sha256=df3e529e178196ad62f7c07ab8593ede5b8596c76e7c6fbd12a5d947a696e65c kernel=empty_kernel threads=256 shared=0 callback_count=1
iters=1000 batches=9 unit=us/call (host enqueue, no timed sync)
TVM FFI HIP HSACO         median=3.1964 samples=[3.586817, 3.196402, 3.199109, 3.199101, 3.3880790000000003, 3.149156, 3.1537930000000003, 3.097593, 3.0956390000000003]
Triton same HSACO         median=15.7614 samples=[15.849793, 15.809132, 15.764721999999999, 15.769162, 15.754864, 15.739984, 15.761377, 15.711687, 15.604766999999999]
INTJ same function        median=2.9686 samples=[2.971412, 2.9820659999999997, 2.990396, 2.990168, 2.9648290000000004, 2.9176759999999997, 2.920319, 2.968609, 2.855473]
elapsed_ns=3785408854
exit_status=0
ended=2026-09-25T16:23:51+08:00
```

### hip: merged

```text
commit=2a9f891443cf6763d26483c7b1f3d0f451d90a3c
started=2026-09-25T16:25:45+08:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/torch_abi
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/torch_abi taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_hip_module_launch.py --iters 1000 --batches 9
hsaco_sha256=df3e529e178196ad62f7c07ab8593ede5b8596c76e7c6fbd12a5d947a696e65c kernel=empty_kernel threads=256 shared=0 callback_count=1
iters=1000 batches=9 unit=us/call (host enqueue, no timed sync)
TVM FFI HIP HSACO         median=3.2957 samples=[3.568092, 3.366324, 3.303465, 3.295718, 3.344101, 3.254853, 3.251299, 3.243468, 3.232071]
Triton same HSACO         median=16.0228 samples=[16.263575, 16.125370999999998, 16.036188, 16.014615, 16.022819, 15.933234, 15.997734000000001, 16.025968000000002, 15.974864]
INTJ same function        median=3.0743 samples=[3.14397, 3.1539360000000003, 3.094375, 3.104724, 3.074271, 2.970649, 3.030532, 3.025797, 3.0236199999999998]
elapsed_ns=5688200784
exit_status=0
ended=2026-09-25T16:25:51+08:00
```

### ffi_paths: develop

```text
commit=ef698d24dae8790fffdf84fe766a7bf06db428f6
started=2026-09-25T16:23:51+08:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_intj_ffi_paths.py --iters 1000 --batches 9
iters=1000 batches=9 unit=ns/call; no_gpu=True
mode=callback
callback_count=9100 (100 warmup + 9000 unique misses)
INTJ cold compile callback + cache  median_ns=469.3 samples_ns=[557.677, 554.861, 426.671, 568.718, 409.814, 441.167, 469.276, 467.944, 3631.986]
INTJ hot call (no callback)         median_ns=91.4 samples_ns=[111.252, 103.213, 107.998, 81.904, 97.453, 73.858, 91.439, 85.785, 89.553]
INTJ 3-tensor host-only nop         median_ns=81.4 samples_ns=[137.582, 73.938, 84.506, 71.399, 64.503, 58.941, 86.556, 338.983, 81.409]
mode=kwargs
INTJ direct positional              median_ns=65.9 samples_ns=[67.948, 66.112, 65.923, 65.758, 65.636, 65.767, 65.688, 74.514, 65.853]
INTJ adapter positional             median_ns=88.7 samples_ns=[89.325, 88.654, 88.434, 88.532, 88.707, 88.667, 88.731, 88.785, 88.204]
INTJ adapter kwargs                 median_ns=105.5 samples_ns=[112.219, 105.521, 107.111, 104.944, 105.716, 105.335, 105.531, 104.711, 104.919]
INTJ adapter defaults               median_ns=88.9 samples_ns=[88.273, 92.29, 89.702, 88.77, 88.936, 87.782, 89.319, 90.688, 88.435]
INTJ FFI wrapper positional         median_ns=104.7 samples_ns=[103.911, 103.821, 118.003, 105.133, 105.496, 104.221, 104.725, 105.29, 104.283]
INTJ FFI wrapper kwargs             median_ns=125.0 samples_ns=[128.505, 127.402, 130.196, 125.028, 124.961, 124.82, 125.014, 123.783, 123.906]
INTJ native kwargs: unsupported (verified TypeError)
mode=dataclass
INTJ pair direct                    median_ns=67.0 samples_ns=[69.359, 66.99, 66.807, 67.462, 68.323, 67.035, 66.508, 70.603, 66.965]
INTJ pair prebuilt *tuple           median_ns=140.9 samples_ns=[140.775, 140.913, 142.791, 138.44, 143.21, 140.807, 146.18, 140.614, 142.242]
FFI unpack Pair only                median_ns=162.8 samples_ns=[162.901, 164.642, 164.03, 166.177, 161.409, 162.837, 161.299, 160.841, 161.223]
INTJ pair manual unpack             median_ns=174.0 samples_ns=[177.166, 172.358, 173.582, 174.229, 174.045, 172.565, 175.883, 174.418, 173.598]
INTJ pair FFI unpack                median_ns=317.2 samples_ns=[313.227, 318.051, 318.595, 317.86, 318.488, 314.877, 315.535, 315.7, 317.238]
INTJ pair stdlib astuple            median_ns=1172.6 samples_ns=[1388.167, 1236.305, 1186.797, 1180.088, 1162.914, 1166.183, 1172.565, 1168.688, 1170.446]
INTJ config direct                  median_ns=81.8 samples_ns=[83.331, 81.733, 81.745, 81.757, 81.9, 82.993, 81.819, 85.537, 81.443]
INTJ config FFI unpack              median_ns=332.2 samples_ns=[332.186, 332.42, 334.341, 329.165, 329.582, 332.56, 324.628, 328.372, 332.282]
INTJ native dataclass unpack: unsupported (verified TypeError)
elapsed_ns=5049387705
exit_status=0
ended=2026-09-25T16:23:56+08:00
```

### ffi_paths: merged

```text
commit=2a9f891443cf6763d26483c7b1f3d0f451d90a3c
started=2026-09-25T16:25:51+08:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/torch_abi
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/torch_abi taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_intj_ffi_paths.py --iters 1000 --batches 9
iters=1000 batches=9 unit=ns/call; no_gpu=True
mode=callback
callback_count=9100 (100 warmup + 9000 unique misses)
INTJ cold compile callback + cache  median_ns=466.4 samples_ns=[559.276, 555.257, 451.684, 594.662, 418.734, 437.287, 446.204, 466.4, 795.044]
INTJ hot call (no callback)         median_ns=38.9 samples_ns=[41.447, 40.534, 38.73, 38.834, 38.755, 38.85, 39.018, 42.85, 38.773]
INTJ 3-tensor host-only nop         median_ns=34.8 samples_ns=[42.982, 34.919, 34.771, 34.718, 34.796, 34.837, 34.708, 34.676, 34.764]
mode=kwargs
INTJ direct positional              median_ns=66.0 samples_ns=[68.58, 66.02, 66.028, 66.147, 65.964, 66.031, 72.944, 65.903, 65.878]
INTJ adapter positional             median_ns=89.1 samples_ns=[89.489, 89.195, 89.029, 89.121, 90.103, 88.693, 88.918, 88.98, 95.811]
INTJ adapter kwargs                 median_ns=107.4 samples_ns=[106.158, 106.91, 107.564, 107.372, 107.57, 107.534, 106.085, 106.464, 112.738]
INTJ adapter defaults               median_ns=87.4 samples_ns=[87.512, 87.422, 87.204, 87.34, 87.361, 87.36, 87.294, 87.371, 86.769]
INTJ FFI wrapper positional         median_ns=101.9 samples_ns=[101.816, 105.858, 102.205, 102.044, 101.899, 103.181, 101.374, 101.306, 101.371]
INTJ FFI wrapper kwargs             median_ns=121.7 samples_ns=[123.461, 126.367, 121.992, 121.708, 121.646, 121.282, 120.933, 122.292, 121.344]
INTJ native kwargs: unsupported (verified TypeError)
mode=dataclass
INTJ pair direct                    median_ns=67.8 samples_ns=[77.308, 68.145, 67.815, 68.318, 73.402, 66.119, 65.926, 65.752, 65.932]
INTJ pair prebuilt *tuple           median_ns=138.9 samples_ns=[140.861, 137.244, 138.905, 138.666, 144.17, 139.001, 139.692, 137.755, 138.059]
FFI unpack Pair only                median_ns=161.5 samples_ns=[161.483, 161.355, 166.585, 161.165, 160.535, 160.553, 162.725, 161.626, 168.947]
INTJ pair manual unpack             median_ns=171.3 samples_ns=[172.577, 170.322, 170.514, 170.537, 170.362, 174.858, 172.118, 171.823, 171.276]
INTJ pair FFI unpack                median_ns=311.8 samples_ns=[316.057, 310.247, 310.466, 311.17, 315.484, 310.688, 311.83, 314.745, 314.526]
INTJ pair stdlib astuple            median_ns=1171.3 samples_ns=[1336.456, 1178.598, 1156.815, 1150.64, 1153.891, 1187.077, 1184.991, 1171.254, 1159.046]
INTJ config direct                  median_ns=82.3 samples_ns=[82.323, 82.73, 82.661, 81.982, 81.749, 82.813, 82.105, 88.05, 79.983]
INTJ config FFI unpack              median_ns=323.7 samples_ns=[321.079, 322.832, 325.371, 324.016, 325.008, 328.609, 321.057, 321.403, 323.657]
INTJ native dataclass unpack: unsupported (verified TypeError)
elapsed_ns=8484698963
exit_status=0
ended=2026-09-25T16:25:59+08:00
```

### ffi_default: develop

```text
commit=ef698d24dae8790fffdf84fe766a7bf06db428f6
started=2026-09-25T16:23:56+08:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_ffi_compare.py --iters 1000 --batches 9
iters=1000 batches=9 unit=us/call (host-only, no timed sync)
FFI packed nop             median=0.1446 samples=[0.147202, 0.144741, 0.143976, 0.144617, 0.1446, 0.14191, 0.14318, 0.14820599999999998, 0.142363]
FFI typed nop              median=0.1491 samples=[0.14743199999999998, 0.15190700000000001, 0.149076, 0.150695, 0.148928, 0.150985, 0.14633600000000002, 0.155454, 0.14604]
FFI empty kernel           median=3.3640 samples=[3.415111, 3.413382, 3.393852, 3.1965239999999997, 3.2052199999999997, 3.418975, 3.217442, 3.364006, 3.211162]
INTJ empty kernel          median=3.0293 samples=[3.100905, 3.103736, 3.174744, 2.9044160000000003, 2.866462, 3.136667, 3.007939, 2.965141, 3.029293]
INTJ fixed-device kernel   median=2.9269 samples=[3.237513, 3.1252600000000004, 3.0860790000000002, 2.918766, 2.894354, 2.828962, 2.932815, 2.842261, 2.926904]
FFI packed nop mixed       median=0.1723 samples=[0.172267, 0.172579, 0.17036400000000002, 0.171912, 0.17450100000000002, 0.172263, 0.169328, 0.17158, 0.17541800000000002]
FFI typed nop mixed        median=0.1778 samples=[0.178278, 0.177744, 0.179648, 0.179097, 0.177758, 0.177578, 0.173103, 0.18066200000000002, 0.175375]
FFI mixed kernel           median=3.3284 samples=[3.393439, 3.454886, 3.328386, 3.27015, 3.232433, 3.391664, 3.280124, 3.3879989999999998, 3.252978]
INTJ mixed kernel          median=2.9383 samples=[3.195076, 3.119413, 2.951376, 2.938268, 2.803264, 2.841796, 2.8509450000000003, 2.9253679999999997, 2.953516]
INTJ fixed mixed kernel    median=2.9123 samples=[3.163492, 3.146765, 2.9848939999999997, 2.9679949999999997, 2.8900259999999998, 2.838344, 2.85659, 2.8757170000000003, 2.912288]
elapsed_ns=3786287037
exit_status=0
ended=2026-09-25T16:24:00+08:00
```

### ffi_default: merged

```text
commit=2a9f891443cf6763d26483c7b1f3d0f451d90a3c
started=2026-09-25T16:25:59+08:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/torch_abi
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/torch_abi taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_ffi_compare.py --iters 1000 --batches 9
iters=1000 batches=9 unit=us/call (host-only, no timed sync)
FFI packed nop             median=0.1461 samples=[0.151477, 0.147566, 0.14511000000000002, 0.143374, 0.14369800000000002, 0.146254, 0.146587, 0.14605600000000002, 0.14607900000000001]
FFI typed nop              median=0.1496 samples=[0.14629599999999998, 0.152483, 0.150757, 0.151228, 0.14861600000000003, 0.152231, 0.14716100000000001, 0.149648, 0.14874199999999999]
FFI empty kernel           median=3.2827 samples=[3.658274, 3.4026419999999997, 3.4147730000000003, 3.212482, 3.188977, 3.2587930000000003, 3.285647, 3.2826950000000004, 3.2814340000000004]
INTJ empty kernel          median=2.9214 samples=[2.961792, 3.057331, 3.050967, 2.8998850000000003, 2.8475349999999997, 2.921415, 2.917611, 2.977769, 2.90995]
INTJ fixed-device kernel   median=2.8789 samples=[2.992004, 3.040438, 3.0311529999999998, 2.87892, 2.853703, 2.869459, 2.868446, 2.8734960000000003, 2.9040500000000002]
FFI packed nop mixed       median=0.1794 samples=[0.179428, 0.179931, 0.178886, 0.177811, 0.181867, 0.176423, 0.18291200000000002, 0.17715199999999998, 0.18051499999999998]
FFI typed nop mixed        median=0.1821 samples=[0.18129900000000002, 0.18579400000000001, 0.18044300000000002, 0.182572, 0.182416, 0.182122, 0.18169, 0.182541, 0.179685]
FFI mixed kernel           median=3.3036 samples=[3.406672, 3.457685, 3.2756849999999997, 3.285898, 3.209524, 3.300932, 3.3228470000000003, 3.3036329999999996, 3.30565]
INTJ mixed kernel          median=2.9069 samples=[3.0529830000000002, 3.0723000000000003, 2.8947510000000003, 2.906902, 2.824358, 2.83819, 2.888391, 2.9181, 2.924286]
INTJ fixed mixed kernel    median=2.9124 samples=[3.125487, 3.132061, 2.949473, 2.9621950000000004, 2.8905390000000004, 2.863907, 2.867752, 2.9124250000000003, 2.89135]
elapsed_ns=11397618456
exit_status=0
ended=2026-09-25T16:26:11+08:00
```

### ffi_sweep: develop

```text
commit=ef698d24dae8790fffdf84fe766a7bf06db428f6
started=2026-09-25T16:24:00+08:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_ffi_compare.py --sweep --iters 1000 --batches 9
counts=[0, 3, 5, 8, 16, 32, 64] iters=1000 batches=9 unit=us/call (host-only, no timed sync); INTJ device-bound, no specialization key
args= 0 FFI packed nop           median=0.1169 samples=[0.118581, 0.11998099999999999, 0.11584, 0.119024, 0.117387, 0.11598900000000001, 0.115334, 0.114568, 0.116923]
args= 0 FFI typed nop            median=0.1208 samples=[0.122363, 0.12304999999999999, 0.116836, 0.122893, 0.118994, 0.12080800000000001, 0.117565, 0.123692, 0.117845]
args= 0 FFI empty kernel         median=1.6006 samples=[1.600363, 1.88134, 1.60059, 1.868308, 1.6000450000000002, 1.899515, 1.6003610000000001, 1.8824400000000001, 1.5964770000000001]
args= 0 INTJ cxx kernel          median=2.7101 samples=[3.1167860000000003, 2.657558, 2.720463, 2.650522, 2.6993690000000004, 2.787143, 2.7101260000000003, 2.674732, 2.7124580000000003]
args= 0 INTJ shim kernel         median=2.6982 samples=[2.887553, 2.687777, 2.71981, 2.685457, 2.719069, 2.673139, 2.698241, 2.6823159999999997, 2.7043649999999997]
args= 3 FFI packed nop           median=0.1441 samples=[0.150852, 0.14350100000000002, 0.143111, 0.145498, 0.143853, 0.146741, 0.14407599999999998, 0.14574600000000001, 0.143536]
args= 3 FFI typed nop            median=0.1503 samples=[0.149792, 0.151348, 0.14643899999999999, 0.150268, 0.14674600000000002, 0.151028, 0.145549, 0.150363, 0.15470699999999998]
args= 3 FFI empty kernel         median=3.1839 samples=[3.326648, 3.176379, 3.1839229999999996, 3.224469, 3.158309, 3.348709, 3.174831, 3.2278939999999996, 3.151147]
args= 3 INTJ cxx kernel          median=2.9534 samples=[2.978896, 2.8504899999999997, 2.9756729999999996, 2.861585, 2.992415, 2.855794, 2.953424, 2.8824549999999998, 2.956473]
args= 3 INTJ shim kernel         median=2.8407 samples=[3.003867, 2.789096, 2.9278299999999997, 2.840705, 2.8213760000000003, 2.859143, 2.7879099999999997, 2.893354, 2.79034]
args= 5 FFI packed nop           median=0.1772 samples=[0.177436, 0.17316800000000002, 0.177268, 0.176263, 0.178148, 0.17636500000000002, 0.17715799999999998, 0.17430600000000002, 0.17757699999999998]
args= 5 FFI typed nop            median=0.1806 samples=[0.182174, 0.179435, 0.18004599999999998, 0.182614, 0.180638, 0.182499, 0.180469, 0.180867, 0.179554]
args= 5 FFI empty kernel         median=3.2991 samples=[3.408611, 3.390963, 3.252853, 3.366379, 3.2279769999999997, 3.350955, 3.217626, 3.299056, 3.223712]
args= 5 INTJ cxx kernel          median=2.8775 samples=[3.008674, 2.877321, 2.877539, 2.904451, 2.9384430000000004, 2.905917, 2.855721, 2.862512, 2.875967]
args= 5 INTJ shim kernel         median=2.8424 samples=[3.0145399999999998, 2.801163, 2.842354, 2.783408, 2.901048, 2.777703, 2.847178, 2.884676, 2.83237]
args= 8 FFI packed nop           median=0.2124 samples=[0.213335, 0.21392, 0.21057800000000002, 0.21473599999999998, 0.21395699999999998, 0.21079900000000001, 0.21207800000000002, 0.210538, 0.212381]
args= 8 FFI typed nop            median=0.2144 samples=[0.213499, 0.21591, 0.213791, 0.216868, 0.21221199999999998, 0.214752, 0.21344, 0.214865, 0.214418]
args= 8 FFI empty kernel         median=3.4018 samples=[3.537485, 3.401797, 3.314934, 3.4329650000000003, 3.3786840000000002, 3.434858, 3.301236, 3.434205, 3.311248]
args= 8 INTJ cxx kernel          median=3.0424 samples=[3.074904, 2.901656, 3.052219, 2.9516199999999997, 3.042421, 2.929274, 3.103169, 2.958189, 3.0860630000000002]
args= 8 INTJ shim kernel         median=2.9702 samples=[3.088851, 2.881158, 2.9770689999999997, 2.9224859999999997, 2.9756460000000002, 2.898841, 2.970158, 2.907735, 2.973029]
args=16 FFI packed nop           median=0.3037 samples=[0.300014, 0.30563999999999997, 0.30430399999999996, 0.30296, 0.303699, 0.30347, 0.30592899999999995, 0.302282, 0.304584]
args=16 FFI typed nop            median=0.3113 samples=[0.31042200000000003, 0.311332, 0.315682, 0.30902999999999997, 0.312866, 0.312007, 0.309257, 0.31365499999999996, 0.310906]
args=16 FFI empty kernel         median=3.7332 samples=[3.793429, 3.752945, 3.615252, 3.733197, 3.594746, 3.740734, 3.604216, 13.118469, 3.62241]
args=16 INTJ cxx kernel          median=3.4592 samples=[3.26832, 3.360916, 3.470054, 3.316652, 3.4768220000000003, 3.3085430000000002, 3.459199, 12.814300999999999, 3.49543]
args=16 INTJ shim kernel         median=3.0789 samples=[3.263637, 3.078869, 3.3501860000000003, 3.062916, 3.074536, 3.043612, 3.379141, 3.0475659999999998, 3.3881460000000003]
args=32 FFI packed nop           median=0.4948 samples=[0.495229, 0.49501799999999996, 0.48629, 0.49516899999999997, 0.490219, 0.497844, 0.494194, 0.485539, 0.49482]
args=32 FFI typed nop            median=0.5096 samples=[0.500125, 0.509603, 0.500098, 0.511545, 0.50202, 0.516224, 0.508161, 0.511425, 0.5146069999999999]
args=32 FFI empty kernel         median=4.1298 samples=[4.211962000000001, 4.129757, 4.018978, 4.178496, 4.009808, 4.177213, 4.009779, 4.204055, 4.010167]
args=32 INTJ cxx kernel          median=3.6183 samples=[3.4708930000000002, 3.614308, 4.0655920000000005, 3.6182950000000003, 3.700836, 3.596657, 3.6682289999999997, 3.599306, 3.6826149999999997]
args=32 INTJ shim kernel         median=3.4561 samples=[3.419404, 3.411377, 7.063662999999999, 3.456065, 3.634214, 3.426494, 3.6106550000000004, 3.409326, 3.608712]
args=64 FFI packed nop           median=0.8684 samples=[0.882375, 0.868356, 0.853012, 0.8489070000000001, 0.8870359999999999, 0.866927, 0.866183, 0.8734310000000001, 0.8825]
args=64 FFI typed nop            median=0.8949 samples=[0.914104, 0.9017329999999999, 0.858591, 0.891331, 0.9087000000000001, 0.8941849999999999, 0.896766, 0.890044, 0.894899]
args=64 FFI empty kernel         median=5.0580 samples=[5.082084, 5.0221800000000005, 10.884767, 5.047979000000001, 5.058046, 5.059674, 5.037647, 5.051342, 5.060471]
args=64 INTJ cxx kernel          median=4.6035 samples=[4.576595, 4.592799, 5.624746, 4.637832, 4.614069000000001, 4.603466, 4.595233, 4.608867, 4.6002790000000005]
args=64 INTJ shim kernel         median=4.5679 samples=[4.538791, 4.560047, 4.600324, 4.5967780000000005, 4.567933, 4.581345000000001, 4.549964, 4.594498, 4.5674660000000005]
elapsed_ns=4362514485
exit_status=0
ended=2026-09-25T16:24:04+08:00
```

### ffi_sweep: merged

```text
commit=2a9f891443cf6763d26483c7b1f3d0f451d90a3c
started=2026-09-25T16:26:11+08:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/torch_abi
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/torch_abi taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_ffi_compare.py --sweep --iters 1000 --batches 9
counts=[0, 3, 5, 8, 16, 32, 64] iters=1000 batches=9 unit=us/call (host-only, no timed sync); INTJ device-bound, no specialization key
args= 0 FFI packed nop           median=0.1166 samples=[0.13070099999999998, 0.116562, 0.11774, 0.115021, 0.115503, 0.115524, 0.114314, 0.11659399999999999, 0.116738]
args= 0 FFI typed nop            median=0.1203 samples=[0.120681, 0.12288400000000001, 0.11691700000000001, 0.121392, 0.117218, 0.120291, 0.116397, 0.124487, 0.117687]
args= 0 FFI empty kernel         median=2.1361 samples=[1.76889, 2.819119, 2.136147, 2.713615, 2.061441, 2.831654, 2.089906, 2.831154, 2.063055]
args= 0 INTJ static_compile kernel median=3.2813 samples=[3.407261, 3.292681, 3.304341, 3.2450419999999998, 3.2298090000000004, 3.281327, 3.270181, 3.295275, 3.247577]
args= 0 INTJ runtime_shim kernel median=3.1594 samples=[3.159364, 3.0518110000000003, 3.358589, 3.0311239999999997, 3.243188, 3.039511, 3.2725940000000002, 3.033324, 3.255348]
args= 3 FFI packed nop           median=0.1459 samples=[0.14658500000000002, 0.144881, 0.146946, 0.14590199999999998, 0.148146, 0.145967, 0.14446799999999999, 0.143994, 0.143937]
args= 3 FFI typed nop            median=0.1482 samples=[0.14687899999999998, 0.15071199999999998, 0.146976, 0.15100999999999998, 0.14452, 0.149406, 0.147048, 0.151617, 0.148195]
args= 3 FFI empty kernel         median=3.7545 samples=[3.754476, 3.795487, 3.7359250000000004, 3.784835, 3.6656269999999997, 3.793838, 3.6746999999999996, 3.804637, 3.6567600000000002]
args= 3 INTJ static_compile kernel median=3.5430 samples=[3.3031010000000003, 3.5450079999999997, 3.574666, 3.543535, 3.5231999999999997, 3.565523, 3.4945, 3.543016, 3.531454]
args= 3 INTJ runtime_shim kernel median=3.4622 samples=[3.292132, 3.412441, 3.569837, 3.431166, 3.5087330000000003, 3.45398, 4.016941, 3.4622379999999997, 3.4974209999999997]
args= 5 FFI packed nop           median=0.1748 samples=[0.17909899999999998, 0.17479599999999998, 0.17395500000000003, 0.177014, 0.176206, 0.173, 0.174006, 0.173981, 0.17555500000000002]
args= 5 FFI typed nop            median=0.1793 samples=[0.180415, 0.180358, 0.179263, 0.178761, 0.177375, 0.181513, 0.173961, 0.18026, 0.174527]
args= 5 FFI empty kernel         median=3.7884 samples=[3.74756, 4.202832, 3.788384, 4.244623, 3.752967, 4.2728969999999995, 3.749206, 4.25229, 3.7305949999999997]
args= 5 INTJ static_compile kernel median=3.5102 samples=[3.317783, 3.551831, 3.53733, 3.4975859999999996, 3.5299050000000003, 3.521458, 3.490009, 3.504437, 3.510249]
args= 5 INTJ runtime_shim kernel median=3.4338 samples=[3.330496, 3.436095, 3.494918, 3.421449, 4.034514, 3.4011810000000002, 3.4337910000000003, 3.408363, 3.4834229999999997]
args= 8 FFI packed nop           median=0.2080 samples=[0.205118, 0.207123, 0.20916900000000002, 0.20382499999999998, 0.207985, 0.204912, 0.210734, 0.209369, 0.208732]
args= 8 FFI typed nop            median=0.2106 samples=[0.210634, 0.21569300000000002, 0.211636, 0.207727, 0.209437, 0.214262, 0.208965, 0.210842, 0.21024500000000002]
args= 8 FFI empty kernel         median=3.9457 samples=[3.945656, 4.236993, 3.920844, 4.350205, 3.8585770000000004, 4.294627, 3.880743, 4.363217, 3.8715230000000003]
args= 8 INTJ static_compile kernel median=3.5643 samples=[3.3918090000000003, 3.5571460000000004, 3.56434, 3.5817620000000003, 3.522384, 3.622726, 3.567665, 3.646013, 3.507887]
args= 8 INTJ runtime_shim kernel median=3.4028 samples=[3.4027730000000003, 3.2279619999999998, 3.6241860000000004, 3.2468380000000003, 3.549753, 3.243049, 3.576543, 3.24703, 3.5780250000000002]
args=16 FFI packed nop           median=0.2948 samples=[0.291154, 0.29915800000000004, 0.298637, 0.295354, 0.297622, 0.294801, 0.29365199999999997, 0.291296, 0.29309199999999996]
args=16 FFI typed nop            median=0.3073 samples=[0.299519, 0.309651, 0.307968, 0.308334, 0.304573, 0.305656, 0.308572, 0.307274, 0.300207]
args=16 FFI empty kernel         median=4.2072 samples=[4.207243999999999, 5.582219, 4.129957999999999, 5.492541, 4.091812, 5.603007, 4.05866, 5.413164, 4.136721]
args=16 INTJ static_compile kernel median=5.1582 samples=[3.690533, 5.103206, 5.605341999999999, 5.066654000000001, 5.4123280000000005, 5.158220999999999, 5.554787999999999, 5.0658, 5.594215]
args=16 INTJ runtime_shim kernel median=3.7275 samples=[3.7275259999999997, 3.6342849999999998, 5.420812, 3.589166, 5.336443, 3.579186, 5.2761629999999995, 3.581159, 5.391063]
args=32 FFI packed nop           median=0.4823 samples=[0.483088, 0.48233800000000004, 0.49225599999999997, 0.48495499999999997, 0.475704, 0.48339499999999996, 0.477889, 0.47668299999999997, 0.476785]
args=32 FFI typed nop            median=0.4951 samples=[0.49731000000000003, 0.493226, 0.48246100000000003, 0.503554, 0.49147, 0.49512, 0.492068, 0.496541, 0.49601799999999996]
args=32 FFI empty kernel         median=4.9634 samples=[4.963432, 5.718915, 4.720983, 5.749487, 4.69505, 5.70311, 4.714897, 5.732412, 4.739581]
args=32 INTJ static_compile kernel median=5.3430 samples=[4.070471, 5.319318, 5.483389, 5.257205, 5.420203000000001, 5.3430159999999995, 5.396988, 5.115421, 5.436494]
args=32 INTJ runtime_shim kernel median=4.0604 samples=[4.008493, 4.019731, 5.39156, 4.060413, 5.582781, 4.038978999999999, 5.322648, 4.014788, 5.336047]
args=64 FFI packed nop           median=0.8382 samples=[0.826307, 0.8433390000000001, 0.8382269999999999, 0.838207, 0.8198970000000001, 0.823028, 0.842277, 0.820256, 0.868851]
args=64 FFI typed nop            median=0.8594 samples=[0.877669, 0.877154, 0.855198, 0.8879049999999999, 0.845121, 0.842623, 0.859429, 0.855223, 0.8737140000000001]
args=64 FFI empty kernel         median=5.8240 samples=[5.819554, 6.2596300000000005, 5.801629, 6.332741, 5.807621, 5.919932, 5.823967, 6.335069000000001, 5.816789]
args=64 INTJ static_compile kernel median=5.9132 samples=[5.428324, 5.906861, 5.886742, 5.965864, 5.91321, 5.993443, 5.9145330000000005, 5.957806, 5.8624350000000005]
args=64 INTJ runtime_shim kernel median=5.8527 samples=[5.194481, 5.937229, 5.732033, 6.020035, 5.852659, 6.1699139999999995, 5.748359000000001, 5.999896, 5.765817]
elapsed_ns=34503012422
exit_status=0
ended=2026-09-25T16:26:45+08:00
```

### Controlled 100-call sweep: develop

```text
commit=ef698d24dae8790fffdf84fe766a7bf06db428f6
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_ffi_compare.py --sweep --iters 100 --batches 9
counts=[0, 3, 5, 8, 16, 32, 64] iters=100 batches=9 unit=us/call (host-only, no timed sync); INTJ device-bound, no specialization key
args= 0 FFI packed nop           median=0.1156 samples=[0.12076, 0.11562, 0.11553000000000001, 0.11408, 0.1159, 0.11469, 0.11645, 0.11685, 0.11541]
args= 0 FFI typed nop            median=0.1189 samples=[0.12269, 0.12125, 0.11796, 0.12157, 0.11894, 0.11804, 0.11842, 0.11976, 0.11845]
args= 0 FFI empty kernel         median=1.7655 samples=[1.76508, 2.0872100000000002, 1.91694, 1.8326600000000002, 1.76021, 1.76555, 1.73658, 1.76397, 1.76924]
args= 0 INTJ cxx kernel          median=2.8592 samples=[3.1764699999999997, 2.95856, 2.97321, 2.8592199999999997, 2.79042, 2.8044000000000002, 2.81753, 2.97035, 2.82967]
args= 0 INTJ shim kernel         median=2.9006 samples=[3.12904, 2.95623, 2.96198, 3.03774, 2.90062, 2.76312, 2.84646, 2.84394, 2.81733]
args= 3 FFI packed nop           median=0.1417 samples=[0.14558000000000001, 0.14025, 0.1426, 0.14173, 0.14117, 0.14321, 0.14067, 0.14202, 0.1391]
args= 3 FFI typed nop            median=0.1456 samples=[0.14793, 0.1466, 0.14667, 0.14858000000000002, 0.145, 0.14479, 0.143, 0.14558000000000001, 0.1433]
args= 3 FFI empty kernel         median=3.3654 samples=[3.79412, 3.45486, 3.47257, 3.5215500000000004, 3.3654499999999996, 3.35087, 3.32321, 3.34275, 3.35364]
args= 3 INTJ cxx kernel          median=3.1131 samples=[3.18343, 3.11309, 3.13658, 3.17893, 3.19622, 2.97116, 2.89901, 2.9959499999999997, 3.06921]
args= 3 INTJ shim kernel         median=3.0646 samples=[4.32643, 3.06464, 3.17255, 3.16587, 3.0008600000000003, 3.0252800000000004, 2.9758, 3.1207800000000003, 3.00916]
args= 5 FFI packed nop           median=0.1705 samples=[0.17018, 0.16837, 0.17286, 0.16946, 0.19946, 0.16965, 0.17368, 0.17047, 0.17064]
args= 5 FFI typed nop            median=0.1756 samples=[0.17690999999999998, 0.17274, 0.17292, 0.17786000000000002, 0.17333, 0.17555, 0.17272, 0.17562, 0.17556]
args= 5 FFI empty kernel         median=3.4516 samples=[4.33912, 3.46975, 3.60096, 3.52138, 3.33554, 3.3449400000000002, 3.33571, 3.45163, 3.32479]
args= 5 INTJ cxx kernel          median=3.0451 samples=[3.09233, 3.13772, 3.19078, 3.12076, 3.00252, 2.95255, 2.9573400000000003, 3.0450999999999997, 3.02205]
args= 5 INTJ shim kernel         median=3.1288 samples=[3.12881, 3.13435, 3.13589, 3.21077, 3.03624, 2.96701, 3.6708499999999997, 3.0565, 2.96635]
args= 8 FFI packed nop           median=0.2039 samples=[0.20384, 0.20391, 0.20239, 0.20518, 0.20489, 0.20284, 0.32612, 0.2048, 0.20333]
args= 8 FFI typed nop            median=0.2073 samples=[0.2727, 0.20671, 0.20603, 0.20729, 0.22405, 0.20726, 0.33957, 0.20933000000000002, 0.20538]
args= 8 FFI empty kernel         median=3.4755 samples=[3.46544, 3.53612, 3.6311, 3.6205100000000003, 3.4412399999999996, 3.4755100000000003, 4.21978, 3.43723, 3.47426]
args= 8 INTJ cxx kernel          median=3.1151 samples=[3.11505, 3.23376, 3.1719899999999996, 3.25331, 3.10263, 3.03702, 3.4881599999999997, 3.1013800000000002, 3.0257899999999998]
args= 8 INTJ shim kernel         median=3.1129 samples=[3.16086, 3.15842, 3.11292, 3.1659, 3.0151399999999997, 3.0368000000000004, 3.3136900000000002, 3.04093, 3.0362600000000004]
args=16 FFI packed nop           median=0.2922 samples=[0.28908999999999996, 0.29128, 0.29236, 0.29489, 0.29312, 0.29218, 0.28835, 0.29189, 0.29578]
args=16 FFI typed nop            median=0.3031 samples=[0.30315, 0.30175, 0.30267, 0.30525, 0.30341, 0.32997000000000004, 0.29913, 0.30007, 0.46519]
args=16 FFI empty kernel         median=4.1818 samples=[3.93647, 3.90892, 4.39116, 3.96377, 4.34621, 3.80152, 4.2403900000000005, 4.62695, 4.18178]
args=16 INTJ cxx kernel          median=3.5364 samples=[3.66566, 3.5126399999999998, 3.6443000000000003, 3.53639, 3.69835, 3.5081599999999997, 3.50062, 4.01244, 3.43593]
args=16 INTJ shim kernel         median=3.5480 samples=[3.77434, 3.65517, 3.65844, 3.548, 3.41566, 3.52278, 3.33864, 4.17475, 3.44586]
args=32 FFI packed nop           median=0.4858 samples=[0.48575, 0.53949, 0.47591, 0.48938, 0.48523, 0.70738, 0.48792, 0.48242, 0.47948]
args=32 FFI typed nop            median=0.4949 samples=[0.49489999999999995, 0.49145999999999995, 0.49401, 0.5013, 0.5004700000000001, 0.49349, 0.49936, 0.5026, 0.48986]
args=32 FFI empty kernel         median=4.4273 samples=[4.671810000000001, 4.43488, 4.73772, 4.53811, 4.3413699999999995, 4.42732, 4.378, 4.41132, 4.42281]
args=32 INTJ cxx kernel          median=3.8131 samples=[3.91137, 3.7256199999999997, 3.9381399999999998, 3.92527, 3.77106, 3.71802, 3.81308, 3.7141100000000002, 3.8239699999999996]
args=32 INTJ shim kernel         median=3.7402 samples=[3.76325, 3.86597, 3.79183, 3.9245799999999997, 3.71949, 3.68698, 3.73115, 3.73871, 3.74017]
args=64 FFI packed nop           median=0.8555 samples=[0.8142999999999999, 0.85563, 0.82505, 0.85781, 0.84681, 0.85554, 0.85646, 0.86715, 0.85457]
args=64 FFI typed nop            median=0.8781 samples=[0.86037, 0.85522, 0.86721, 0.89637, 0.87806, 0.88095, 0.87254, 0.89706, 0.8821]
args=64 FFI empty kernel         median=5.2323 samples=[5.4215, 5.23229, 5.4189099999999994, 5.498819999999999, 5.20815, 5.24085, 5.17171, 5.213760000000001, 5.21597]
args=64 INTJ cxx kernel          median=4.7643 samples=[4.90202, 4.84416, 4.92601, 4.8101, 4.764270000000001, 4.64102, 4.70225, 4.61397, 4.665430000000001]
args=64 INTJ shim kernel         median=4.6995 samples=[4.80945, 4.69946, 4.76415, 4.85468, 4.61141, 4.66379, 4.65697, 4.73578, 4.66981]
```

### Controlled 100-call sweep: merged

```text
commit=2a9f891443cf6763d26483c7b1f3d0f451d90a3c
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/torch_abi taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_ffi_compare.py --sweep --iters 100 --batches 9
counts=[0, 3, 5, 8, 16, 32, 64] iters=100 batches=9 unit=us/call (host-only, no timed sync); INTJ device-bound, no specialization key
args= 0 FFI packed nop           median=0.1167 samples=[0.12311, 0.11702, 0.11672, 0.11745, 0.11779, 0.11359, 0.11506999999999999, 0.11554, 0.11425]
args= 0 FFI typed nop            median=0.1187 samples=[0.12192, 0.12068999999999999, 0.11823, 0.11923, 0.11957999999999999, 0.11722, 0.11498, 0.11875, 0.11597]
args= 0 FFI empty kernel         median=1.8349 samples=[1.82774, 1.96072, 1.94845, 1.8848399999999998, 2.01792, 1.83297, 1.83491, 1.83029, 1.77325]
args= 0 INTJ static_compile kernel median=2.9964 samples=[3.20891, 3.08582, 3.0744499999999997, 3.08727, 2.92958, 2.93444, 2.9245900000000002, 2.9964299999999997, 2.8911700000000002]
args= 0 INTJ runtime_shim kernel median=2.9989 samples=[3.1481, 2.99894, 3.07371, 3.01985, 3.02893, 2.8200700000000003, 2.85683, 2.92633, 2.84744]
args= 3 FFI packed nop           median=0.1419 samples=[0.14595, 0.14368, 0.14149, 0.1418, 0.1411, 0.17022, 0.14191, 0.14134, 0.14267]
args= 3 FFI typed nop            median=0.1448 samples=[0.14476, 0.14648, 0.14178, 0.1433, 0.145, 0.15009999999999998, 0.14474, 0.14477, 0.14606]
args= 3 FFI empty kernel         median=3.4979 samples=[3.7921799999999997, 3.53814, 3.6029899999999997, 3.53798, 3.4978800000000003, 3.34911, 3.3178400000000003, 3.37704, 3.38675]
args= 3 INTJ static_compile kernel median=3.0852 samples=[3.08521, 3.20763, 3.20138, 3.28795, 3.05961, 3.0723000000000003, 2.95325, 3.07255, 3.0949299999999997]
args= 3 INTJ runtime_shim kernel median=3.1394 samples=[3.1929600000000002, 3.29067, 3.18357, 3.25791, 3.1393899999999997, 3.07009, 3.03394, 3.1000199999999998, 3.08482]
args= 5 FFI packed nop           median=0.1692 samples=[0.1722, 0.1686, 0.17073, 0.16874, 0.17026, 0.16837, 0.16921, 0.17089, 0.16835]
args= 5 FFI typed nop            median=0.1736 samples=[0.17493999999999998, 0.17233, 0.17363, 0.1736, 0.17425, 0.17740999999999998, 0.17403, 0.17318, 0.17217]
args= 5 FFI empty kernel         median=3.4444 samples=[3.44442, 3.76642, 3.59727, 3.58508, 3.4074, 3.3474, 3.35858, 3.46548, 3.31235]
args= 5 INTJ static_compile kernel median=3.0921 samples=[3.0751500000000003, 3.25646, 3.22464, 3.22303, 3.25069, 3.0680500000000004, 3.06046, 3.09212, 3.0749299999999997]
args= 5 INTJ runtime_shim kernel median=3.1464 samples=[3.18275, 3.2360100000000003, 3.17868, 3.28202, 3.1326300000000002, 3.06493, 3.00488, 3.1464499999999997, 3.06285]
args= 8 FFI packed nop           median=0.2022 samples=[0.20402, 0.2036, 0.20199, 0.2031, 0.20222, 0.20367, 0.20144, 0.20183, 0.202]
args= 8 FFI typed nop            median=0.2067 samples=[0.20632, 0.20959999999999998, 0.2068, 0.21028, 0.20673, 0.20772, 0.20568999999999998, 0.20555, 0.2055]
args= 8 FFI empty kernel         median=3.5509 samples=[3.47636, 3.6437600000000003, 3.71311, 3.6145300000000002, 3.55437, 3.5508699999999997, 3.46484, 3.44713, 3.54183]
args= 8 INTJ static_compile kernel median=3.3384 samples=[3.36557, 3.34131, 3.33842, 3.3438499999999998, 3.39302, 3.20492, 3.08169, 3.17221, 3.11549]
args= 8 INTJ runtime_shim kernel median=3.1891 samples=[3.21679, 3.2262399999999998, 3.25398, 3.22752, 3.1890500000000004, 3.15903, 3.03884, 3.0868, 3.0721999999999996]
args=16 FFI packed nop           median=0.2920 samples=[0.29137, 0.29554, 0.29258999999999996, 0.29073000000000004, 0.28837, 0.29136, 0.2934, 0.292, 0.29405000000000003]
args=16 FFI typed nop            median=0.3018 samples=[0.29852999999999996, 0.30698000000000003, 0.30125, 0.30548000000000003, 0.30121, 0.30355, 0.30177, 0.30251999999999996, 0.30161]
args=16 FFI empty kernel         median=3.9842 samples=[3.93846, 3.9442, 4.355180000000001, 3.9841599999999997, 4.364850000000001, 3.79605, 4.13501, 3.84183, 4.20582]
args=16 INTJ static_compile kernel median=3.4865 samples=[3.5837399999999997, 3.4505700000000004, 3.62735, 3.61772, 3.48651, 3.4802600000000004, 3.48614, 3.33615, 3.48759]
args=16 INTJ runtime_shim kernel median=3.6017 samples=[3.5964899999999997, 3.70215, 3.76603, 3.6496399999999998, 3.57139, 3.60166, 3.4045300000000003, 3.65967, 3.4783000000000004]
args=32 FFI packed nop           median=0.4863 samples=[0.47657, 0.48052, 0.48625, 0.49185, 0.48634, 0.47689, 0.51467, 0.5112, 0.47701]
args=32 FFI typed nop            median=0.4947 samples=[0.49415, 0.49473, 0.50094, 0.50188, 0.50175, 0.49275, 0.50036, 0.48707, 0.48691]
args=32 FFI empty kernel         median=4.3714 samples=[4.5457600000000005, 4.37144, 4.48151, 4.54056, 4.367319999999999, 4.5993900000000005, 4.35875, 4.368720000000001, 4.35316]
args=32 INTJ static_compile kernel median=3.7552 samples=[3.8792, 3.75521, 3.8875300000000004, 3.8275799999999998, 3.74781, 3.64259, 3.72599, 3.6659200000000003, 3.78602]
args=32 INTJ runtime_shim kernel median=3.8246 samples=[3.90606, 3.88842, 3.83523, 3.96956, 3.8182, 3.81129, 3.66223, 3.82463, 3.73522]
args=64 FFI packed nop           median=0.8440 samples=[0.8413099999999999, 0.92007, 0.8443200000000001, 0.83282, 0.84402, 0.86218, 0.86211, 0.8423999999999999, 0.84012]
args=64 FFI typed nop            median=0.8717 samples=[0.87171, 0.8805599999999999, 0.87566, 0.88144, 0.86914, 0.88161, 0.8655700000000001, 0.85512, 0.86038]
args=64 FFI empty kernel         median=5.3602 samples=[5.39095, 5.36023, 5.38695, 5.45471, 5.306430000000001, 5.3989899999999995, 5.24294, 5.213850000000001, 5.13857]
args=64 INTJ static_compile kernel median=4.8602 samples=[4.96038, 5.51304, 4.901260000000001, 4.93088, 4.86016, 4.83284, 4.6164499999999995, 4.70486, 4.6812]
args=64 INTJ runtime_shim kernel median=4.7616 samples=[5.845, 5.735189999999999, 4.81915, 4.73681, 4.76156, 4.62356, 4.63881, 4.661899999999999, 4.76725]
```


## Final source `03a8f52`: affected benchmark repeats

### ffi_paths: final merged, round 0

```text
commit=03a8f52651248cb333ed50347fbabac6bd15103a
started=2026-09-25T16:35:15+08:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/torch_abi
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/torch_abi taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_intj_ffi_paths.py --iters 1000 --batches 9
iters=1000 batches=9 unit=ns/call; no_gpu=True
mode=callback
callback_count=9100 (100 warmup + 9000 unique misses)
INTJ cold compile callback + cache  median_ns=438.6 samples_ns=[486.117, 499.575, 414.869, 547.756, 409.153, 438.565, 434.398, 436.13, 619.186]
INTJ hot call (no callback)         median_ns=39.8 samples_ns=[41.838, 40.611, 39.877, 39.779, 39.732, 43.893, 37.989, 38.419, 37.88]
INTJ 3-tensor host-only nop         median_ns=35.7 samples_ns=[38.31, 35.917, 35.67, 35.755, 35.723, 41.534, 34.864, 34.79, 34.805]
mode=kwargs
INTJ direct positional              median_ns=65.0 samples_ns=[67.671, 64.904, 64.893, 64.859, 65.035, 65.071, 65.068, 65.063, 65.018]
INTJ adapter positional             median_ns=88.8 samples_ns=[94.574, 89.239, 88.637, 88.754, 88.613, 88.769, 88.716, 88.651, 88.784]
INTJ adapter kwargs                 median_ns=106.0 samples_ns=[107.713, 110.773, 106.258, 105.711, 105.482, 106.472, 105.538, 106.024, 105.935]
INTJ adapter defaults               median_ns=89.7 samples_ns=[88.731, 88.765, 92.925, 89.193, 89.491, 89.912, 89.804, 89.852, 89.683]
INTJ FFI wrapper positional         median_ns=104.2 samples_ns=[101.423, 101.127, 100.332, 105.137, 103.924, 104.253, 104.324, 104.432, 104.168]
INTJ FFI wrapper kwargs             median_ns=123.3 samples_ns=[124.075, 124.002, 123.109, 131.238, 123.349, 122.554, 122.565, 123.515, 122.465]
INTJ native kwargs: unsupported (verified TypeError)
mode=dataclass
INTJ pair direct                    median_ns=67.8 samples_ns=[70.977, 67.916, 67.778, 67.581, 67.834, 67.802, 67.952, 67.809, 67.696]
INTJ pair prebuilt *tuple           median_ns=139.9 samples_ns=[145.476, 140.749, 137.793, 138.417, 138.02, 139.212, 139.883, 142.574, 140.628]
FFI unpack Pair only                median_ns=161.4 samples_ns=[162.12, 161.985, 160.579, 161.298, 165.417, 160.926, 160.968, 161.456, 161.382]
INTJ pair manual unpack             median_ns=171.4 samples_ns=[171.453, 176.316, 170.39, 169.71, 171.444, 171.47, 169.871, 179.328, 170.539]
INTJ pair FFI unpack                median_ns=312.6 samples_ns=[312.127, 312.356, 327.195, 312.602, 313.026, 316.74, 312.248, 310.978, 317.404]
INTJ pair stdlib astuple            median_ns=1169.1 samples_ns=[1228.131, 1166.659, 1199.105, 1165.544, 1169.112, 1173.254, 1185.354, 1166.677, 1163.902]
INTJ config direct                  median_ns=82.1 samples_ns=[82.82, 86.476, 81.863, 83.038, 81.968, 82.054, 81.846, 81.533, 82.917]
INTJ config FFI unpack              median_ns=322.3 samples_ns=[325.137, 326.279, 322.261, 321.672, 328.807, 321.722, 320.696, 324.805, 320.359]
INTJ native dataclass unpack: unsupported (verified TypeError)
elapsed_ns=3156507517
exit_status=0
ended=2026-09-25T16:35:18+08:00
```

### ffi_default: final merged, round 0

```text
commit=03a8f52651248cb333ed50347fbabac6bd15103a
started=2026-09-25T16:35:18+08:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/torch_abi
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/torch_abi taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_ffi_compare.py --iters 1000 --batches 9
iters=1000 batches=9 unit=us/call (host-only, no timed sync)
FFI packed nop             median=0.1437 samples=[0.144786, 0.143241, 0.143436, 0.142495, 0.144041, 0.143453, 0.144423, 0.143661, 0.144952]
FFI typed nop              median=0.1492 samples=[0.149219, 0.15039, 0.14719200000000002, 0.150492, 0.148239, 0.151742, 0.14813900000000002, 0.150257, 0.149166]
FFI empty kernel           median=3.2901 samples=[3.5989250000000004, 3.4622930000000003, 3.4522220000000003, 3.252208, 3.240283, 3.2883229999999997, 3.290056, 3.280245, 3.299412]
INTJ empty kernel          median=2.9156 samples=[2.9225790000000003, 3.07425, 3.086105, 2.8931999999999998, 2.861972, 2.920275, 2.899002, 2.9156050000000002, 2.9023290000000004]
INTJ fixed-device kernel   median=2.8860 samples=[2.990303, 3.067581, 3.0442519999999997, 2.8959989999999998, 2.852337, 2.877809, 2.8768249999999997, 2.878016, 2.886004]
FFI packed nop mixed       median=0.1740 samples=[0.17439, 0.170483, 0.177469, 0.17244800000000002, 0.173961, 0.172118, 0.173382, 0.174144, 0.17416800000000002]
FFI typed nop mixed        median=0.1768 samples=[0.175394, 0.18230000000000002, 0.176646, 0.179219, 0.175958, 0.177384, 0.17678899999999997, 0.178892, 0.175428]
FFI mixed kernel           median=3.3059 samples=[3.435029, 3.491946, 3.300509, 3.305893, 3.254458, 3.316563, 3.322756, 3.296808, 3.3026269999999998]
INTJ mixed kernel          median=2.9074 samples=[3.085277, 3.086927, 2.890526, 2.90876, 2.8420949999999996, 2.855229, 2.907381, 2.917424, 2.896486]
INTJ fixed mixed kernel    median=2.8989 samples=[3.1265549999999998, 3.110259, 2.966683, 2.957229, 2.868764, 2.860059, 2.858838, 2.8988780000000003, 2.856938]
elapsed_ns=4032373401
exit_status=0
ended=2026-09-25T16:35:22+08:00
```

### ffi_sweep: final merged, round 0

```text
commit=03a8f52651248cb333ed50347fbabac6bd15103a
started=2026-09-25T16:35:22+08:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/torch_abi
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/torch_abi taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_ffi_compare.py --sweep --iters 1000 --batches 9
counts=[0, 3, 5, 8, 16, 32, 64] iters=1000 batches=9 unit=us/call (host-only, no timed sync); INTJ device-bound, no specialization key
args= 0 FFI packed nop           median=0.1180 samples=[0.119307, 0.11731699999999999, 0.11837099999999999, 0.115825, 0.117479, 0.118072, 0.117631, 0.11804300000000001, 0.118964]
args= 0 FFI typed nop            median=0.1196 samples=[0.120717, 0.123606, 0.118725, 0.12428499999999999, 0.119617, 0.123666, 0.11893899999999999, 0.11962, 0.11876600000000001]
args= 0 FFI empty kernel         median=1.6633 samples=[1.663334, 1.9079860000000002, 1.615663, 1.9294310000000001, 1.6469369999999999, 1.9501220000000001, 1.647423, 1.8147529999999998, 1.589304]
args= 0 INTJ static_compile kernel median=2.7192 samples=[3.049362, 2.717306, 2.719156, 2.7606550000000003, 2.773183, 2.669844, 2.7799050000000003, 2.668622, 2.69003]
args= 0 INTJ runtime_shim kernel median=2.6907 samples=[2.8566599999999998, 2.66768, 2.709985, 2.662874, 2.727255, 2.6640680000000003, 2.7364050000000004, 2.679245, 2.690682]
args= 3 FFI packed nop           median=0.1457 samples=[0.146519, 0.144464, 0.14751499999999998, 0.145715, 0.15218500000000001, 0.143247, 0.147927, 0.142348, 0.145297]
args= 3 FFI typed nop            median=0.1492 samples=[0.150024, 0.14942, 0.147453, 0.149554, 0.14921, 0.146342, 0.149151, 0.148202, 0.146619]
args= 3 FFI empty kernel         median=3.2301 samples=[3.292537, 3.243902, 3.1653200000000004, 3.294257, 3.144693, 3.2792510000000004, 3.1408899999999997, 3.230096, 3.143565]
args= 3 INTJ static_compile kernel median=2.9224 samples=[2.9253229999999997, 2.836312, 3.045233, 2.859935, 2.940529, 2.842841, 3.028163, 2.860432, 2.922409]
args= 3 INTJ runtime_shim kernel median=2.8740 samples=[2.944131, 2.759699, 2.948996, 2.814163, 2.874015, 2.8425439999999997, 2.894918, 2.851725, 2.882088]
args= 5 FFI packed nop           median=0.1737 samples=[0.172046, 0.178093, 0.173012, 0.173694, 0.173363, 0.173974, 0.175828, 0.175785, 0.171391]
args= 5 FFI typed nop            median=0.1765 samples=[0.17393199999999998, 0.17791900000000002, 0.17708600000000002, 0.195473, 0.176508, 0.174823, 0.177111, 0.175357, 0.17605500000000002]
args= 5 FFI empty kernel         median=3.2588 samples=[3.348159, 3.2274499999999997, 3.212893, 3.242896, 3.262676, 3.258782, 3.1837750000000002, 3.27967, 3.273181]
args= 5 INTJ static_compile kernel median=2.8784 samples=[2.965304, 2.882435, 2.901295, 2.872443, 2.878426, 2.8712910000000003, 2.816125, 2.8341190000000003, 2.90039]
args= 5 INTJ runtime_shim kernel median=2.8770 samples=[2.94591, 2.924795, 2.9079200000000003, 2.889303, 2.8570610000000003, 2.875631, 2.8769780000000003, 2.778328, 2.846716]
args= 8 FFI packed nop           median=0.2054 samples=[0.206595, 0.205036, 0.203754, 0.20887899999999998, 0.20513499999999998, 0.202434, 0.20738900000000002, 0.205536, 0.205413]
args= 8 FFI typed nop            median=0.2096 samples=[0.20986000000000002, 0.21076699999999998, 0.207014, 0.20959899999999998, 0.20657599999999998, 0.208862, 0.209697, 0.212227, 0.20627099999999998]
args= 8 FFI empty kernel         median=3.3726 samples=[3.487013, 3.373513, 3.3726410000000002, 3.4093139999999997, 3.3045120000000003, 3.366144, 3.3323180000000003, 3.358407, 3.484166]
args= 8 INTJ static_compile kernel median=3.0459 samples=[3.057237, 2.915677, 3.0844940000000003, 3.053937, 3.043869, 3.045881, 3.0621579999999997, 3.00558, 3.022348]
args= 8 INTJ runtime_shim kernel median=2.9160 samples=[3.036306, 2.873898, 2.9787779999999997, 2.9160100000000004, 2.899982, 2.8776680000000003, 3.019506, 2.889776, 2.955886]
args=16 FFI packed nop           median=0.2942 samples=[0.295151, 0.289291, 0.295706, 0.289539, 0.294217, 0.29454199999999997, 0.292576, 0.294741, 0.293829]
args=16 FFI typed nop            median=0.3035 samples=[0.302771, 0.30355, 0.299623, 0.305712, 0.30260899999999996, 0.303615, 0.305984, 0.30774900000000005, 0.300454]
args=16 FFI empty kernel         median=3.6510 samples=[3.6868290000000004, 3.669632, 3.586195, 3.660632, 3.5351559999999997, 3.658321, 3.539444, 3.651048, 3.515746]
args=16 INTJ static_compile kernel median=3.3270 samples=[3.209154, 3.327043, 3.4700219999999997, 3.323673, 3.44067, 3.3230459999999997, 3.445016, 3.258585, 3.48018]
args=16 INTJ runtime_shim kernel median=3.3339 samples=[3.3987570000000003, 3.077818, 3.3463049999999996, 3.067244, 3.333875, 3.023377, 3.3349290000000003, 3.0295859999999997, 3.346399]
args=32 FFI packed nop           median=0.4760 samples=[0.48433699999999996, 0.478845, 0.47591100000000003, 0.47597500000000004, 0.478518, 0.479082, 0.475356, 0.469663, 0.47358300000000003]
args=32 FFI typed nop            median=0.4934 samples=[0.499496, 0.494568, 0.480979, 0.492679, 0.49340300000000004, 0.493756, 0.494347, 0.487148, 0.479762]
args=32 FFI empty kernel         median=4.2023 samples=[4.303015, 4.202271, 4.118225000000001, 4.2805219999999995, 4.096161, 4.2397920000000004, 4.090728, 4.228857, 4.07192]
args=32 INTJ static_compile kernel median=3.5888 samples=[3.40542, 3.588811, 3.671399, 3.5460529999999997, 3.634983, 3.575038, 3.6477, 3.56338, 3.673209]
args=32 INTJ runtime_shim kernel median=3.4644 samples=[3.431092, 3.417596, 3.638465, 3.4644310000000003, 3.636706, 3.43779, 3.6048620000000002, 3.436543, 3.6433240000000002]
args=64 FFI packed nop           median=0.8486 samples=[0.844005, 0.84864, 0.830727, 0.819399, 0.854932, 0.823473, 0.8532029999999999, 0.858635, 0.8531409999999999]
args=64 FFI typed nop            median=0.8598 samples=[0.858194, 0.864101, 0.857085, 0.859538, 0.857293, 0.859771, 0.877621, 0.8828210000000001, 0.867833]
args=64 FFI empty kernel         median=5.0514 samples=[5.051404000000001, 4.999446, 5.0673900000000005, 5.070849, 5.069553, 5.0464210000000005, 5.043342, 5.076906, 5.050326]
args=64 INTJ static_compile kernel median=4.6077 samples=[4.600020000000001, 4.569253, 4.672140000000001, 4.637999, 4.603806, 4.637202, 4.603595, 4.607691, 4.6180140000000005]
args=64 INTJ runtime_shim kernel median=4.5916 samples=[4.609987, 4.539432, 4.578602, 4.6083050000000005, 4.549448, 4.605479, 4.591563, 4.598329, 4.5634250000000005]
elapsed_ns=4190790679
exit_status=0
ended=2026-09-25T16:35:27+08:00
```

### controlled: final merged, round 0

```text
commit=03a8f52651248cb333ed50347fbabac6bd15103a
started=2026-09-25T16:35:27+08:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/torch_abi
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/torch_abi taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_ffi_compare.py --sweep --iters 100 --batches 9
counts=[0, 3, 5, 8, 16, 32, 64] iters=100 batches=9 unit=us/call (host-only, no timed sync); INTJ device-bound, no specialization key
args= 0 FFI packed nop           median=0.1163 samples=[0.12398, 0.11626, 0.11564, 0.11629, 0.11714, 0.11615, 0.11559, 0.11742, 0.1164]
args= 0 FFI typed nop            median=0.1186 samples=[0.11979000000000001, 0.11816, 0.11856, 0.12096, 0.11944, 0.11795, 0.11605, 0.12072, 0.11762]
args= 0 FFI empty kernel         median=2.5258 samples=[2.52109, 2.7561799999999996, 2.69579, 2.52582, 2.51258, 2.4835, 2.47277, 2.5493, 2.57044]
args= 0 INTJ static_compile kernel median=3.6072 samples=[3.83923, 3.65989, 3.69399, 3.4801100000000003, 3.50448, 3.93127, 3.4218200000000003, 3.60718, 3.4623000000000004]
args= 0 INTJ runtime_shim kernel median=3.6003 samples=[3.79385, 3.7052199999999997, 3.6887600000000003, 3.7134, 3.60032, 3.37335, 3.43617, 3.4763, 3.48771]
args= 3 FFI packed nop           median=0.1422 samples=[0.14586000000000002, 0.14335, 0.14166, 0.14255, 0.14515, 0.14221999999999999, 0.14156, 0.14211000000000001, 0.14187]
args= 3 FFI typed nop            median=0.1462 samples=[0.14194, 0.14629, 0.14503, 0.14876, 0.14619, 0.1467, 0.14342, 0.147, 0.14622]
args= 3 FFI empty kernel         median=4.0860 samples=[4.45756, 4.25234, 4.21405, 4.1887799999999995, 4.086, 3.97618, 3.8730599999999997, 3.9806, 3.98106]
args= 3 INTJ static_compile kernel median=3.7903 samples=[3.8019000000000003, 3.8698, 3.87685, 3.8226, 3.7902600000000004, 3.71971, 3.51203, 3.67175, 3.6564099999999997]
args= 3 INTJ runtime_shim kernel median=3.7641 samples=[3.7256, 3.94154, 3.98873, 3.90285, 3.9385700000000003, 3.6692600000000004, 3.65314, 3.76415, 3.76236]
args= 5 FFI packed nop           median=0.1703 samples=[0.1714, 0.17022, 0.16968, 0.16795, 0.17435, 0.1711, 0.17029, 0.17173, 0.17034]
args= 5 FFI typed nop            median=0.1756 samples=[0.17505, 0.17679, 0.17286, 0.17414, 0.17429, 0.17629, 0.17595, 0.17809999999999998, 0.17563]
args= 5 FFI empty kernel         median=4.2190 samples=[4.22872, 4.2695799999999995, 4.43409, 4.219, 4.1061000000000005, 4.58254, 4.02794, 4.16497, 4.06468]
args= 5 INTJ static_compile kernel median=3.7959 samples=[3.6629, 3.96925, 3.94109, 3.8250100000000002, 3.9761100000000003, 3.6176, 3.7866500000000003, 3.75262, 3.7958600000000002]
args= 5 INTJ runtime_shim kernel median=3.6978 samples=[3.6165100000000003, 3.9566500000000002, 3.69055, 3.85546, 3.7432800000000004, 3.64281, 3.55896, 3.74999, 3.69781]
args= 8 FFI packed nop           median=0.2080 samples=[0.20605, 0.20743, 0.20531, 0.20804, 0.20902, 0.20875, 0.23965, 0.20673, 0.20905]
args= 8 FFI typed nop            median=0.2111 samples=[0.21054, 0.21112, 0.21108000000000002, 0.21169, 0.21092, 0.21231999999999998, 0.2099, 0.21147, 0.20852]
args= 8 FFI empty kernel         median=4.2162 samples=[4.22529, 4.32047, 4.459239999999999, 4.44683, 4.12953, 4.21624, 4.05995, 4.11834, 4.18007]
args= 8 INTJ static_compile kernel median=3.8231 samples=[3.8417600000000003, 3.8546199999999997, 4.00092, 3.8644000000000003, 3.77839, 3.6456500000000003, 3.65529, 3.69737, 3.82309]
args= 8 INTJ runtime_shim kernel median=3.8479 samples=[3.89455, 3.8479, 3.91017, 3.91189, 3.6247399999999996, 4.126060000000001, 3.47308, 3.62924, 3.72945]
args=16 FFI packed nop           median=0.2944 samples=[0.29086, 0.29439, 0.29251, 0.29685, 0.29296, 0.29449000000000003, 0.29363, 0.29514999999999997, 0.295]
args=16 FFI typed nop            median=0.3041 samples=[0.30071, 0.30363, 0.30010000000000003, 0.30423, 0.30407, 0.30521, 0.30344, 0.30669, 0.30471]
args=16 FFI empty kernel         median=6.1327 samples=[6.0047299999999995, 5.843430000000001, 6.23053, 5.931109999999999, 6.540310000000001, 5.8573699999999995, 6.19269, 6.19597, 6.132680000000001]
args=16 INTJ static_compile kernel median=5.6760 samples=[5.821680000000001, 5.44576, 5.71634, 5.89052, 5.630529999999999, 5.49417, 5.39983, 5.67597, 5.70996]
args=16 INTJ runtime_shim kernel median=5.4672 samples=[5.82387, 5.99834, 5.34413, 6.067609999999999, 5.11133, 6.203720000000001, 5.00133, 5.46723, 4.85954]
args=32 FFI packed nop           median=0.4845 samples=[0.47494, 0.47128, 0.48743000000000003, 0.4701, 0.47942, 0.50285, 0.48881, 0.48749000000000003, 0.48451]
args=32 FFI typed nop            median=0.4939 samples=[0.49783999999999995, 0.49299, 0.49279, 0.49081, 0.49392, 0.49177, 0.53431, 0.50112, 0.54367]
args=32 FFI empty kernel         median=6.3301 samples=[6.35969, 6.24842, 6.249770000000001, 6.3300600000000005, 6.45335, 6.34438, 6.290760000000001, 6.2348, 6.38389]
args=32 INTJ static_compile kernel median=5.9724 samples=[6.40292, 5.7603, 6.22158, 6.165109999999999, 5.966600000000001, 6.01361, 5.87868, 5.796930000000001, 5.9723999999999995]
args=32 INTJ runtime_shim kernel median=5.8726 samples=[5.6931899999999995, 5.8726, 5.2684, 6.27086, 5.582050000000001, 6.20126, 5.9930200000000005, 6.14616, 5.71454]
args=64 FFI packed nop           median=0.8637 samples=[0.88577, 0.82443, 0.90247, 0.85624, 0.86561, 0.8351799999999999, 0.86372, 0.85323, 0.89463]
args=64 FFI typed nop            median=0.8921 samples=[0.87913, 0.86958, 0.8978200000000001, 0.88532, 0.9043, 0.86313, 0.8920800000000001, 0.90725, 0.89774]
args=64 FFI empty kernel         median=6.8209 samples=[6.87883, 6.76496, 6.906479999999999, 6.58964, 6.82092, 6.882, 6.76405, 6.649760000000001, 6.91572]
args=64 INTJ static_compile kernel median=6.5411 samples=[6.70303, 6.37949, 6.83228, 6.276770000000001, 6.66376, 6.1140799999999995, 6.54114, 6.14121, 6.667140000000001]
args=64 INTJ runtime_shim kernel median=6.3078 samples=[6.54593, 6.20928, 6.19532, 6.2412, 6.33203, 5.98795, 6.307840000000001, 6.318359999999999, 6.32973]
elapsed_ns=3594475842
exit_status=0
ended=2026-09-25T16:35:30+08:00
```

# Complete `develop` Python benchmark run

Measured September 25, 2026, on clean local `develop` at `ef698d2`. All four Python benchmark scripts in `benchmarks/` were run in every distinct mode (eight cases), serially in three separate processes per case. Values below are the median of three process medians; each process includes nine timed batches.

| Case | Round 0 | Round 1 | Round 2 | Median |
| --- | ---: | ---: | ---: | ---: |
| GPU launcher, auto map (ns) | 3115.400 | 3729.200 | 3590.200 | 3590.200 |
| Host launcher, auto map (ns) | 43.400 | 43.500 | 44.400 | 43.500 |
| README grid 1, Triton (µs) | 16.480 | 16.800 | 16.350 | 16.480 |
| README grid 1, INTJ (µs) | 3.140 | 3.130 | 3.110 | 3.130 |
| Host sweep, 32 tensors (ns) | 126.000 | 118.300 | 120.200 | 120.200 |
| FFI comparison, INTJ empty (µs) | 2.977 | 3.023 | 2.988 | 2.988 |
| FFI comparison, FFI empty (µs) | 3.327 | 3.353 | 3.357 | 3.353 |
| FFI sweep, 64 INTJ CXX (µs) | 4.594 | 4.599 | 4.659 | 4.599 |
| FFI sweep, 64 INTJ SHIM (µs) | 4.563 | 4.596 | 4.622 | 4.596 |
| INTJ/FFI paths, hot call (ns) | 39.300 | 38.100 | 38.200 | 38.200 |
| INTJ/FFI paths, adapter kwargs (ns) | 107.500 | 106.900 | 109.600 | 107.500 |
| Same HSACO, INTJ (µs) | 2.944 | 2.904 | 2.992 | 2.944 |
| Same HSACO, TVM FFI (µs) | 3.171 | 3.127 | 3.202 | 3.171 |

The GPU rows time host submission and stop before synchronization. Their medians do not measure GPU completion. FFI packed no-ops do not launch a kernel; INTJ and FFI GPU kernels in `bench_ffi_compare.py` are different binaries, whereas `bench_hip_module_launch.py` uses one HSACO. The cold callback row has unique cache misses and is not comparable to hot launch rows. The no-GPU sweep is host-only. CUDA runtime was not tested on this ROCm host.

## Setup and commands

- CPU: x86-64; `taskset -c 0`; one benchmark process at a time. GPU: AMD Instinct MI308X `gfx942:sramecc+:xnack-`. No other benchmark process was running before the first round.
- Python 3.12.3 (`/tmp/gb2/bin/python`); Torch 2.14.0+rocm7.2; Triton 3.8.0; TVM FFI 0.1.14.post2.dev1+g424558557.d20260924.
- `TRITON_HOME`, `CXX`, `CC`, and `INTJ_BENCHMARK_ROOT` unset. Default compiler and module caches; compilation, setup, and argument construction are outside call timers. Each benchmark warms its cases; the scripts that write data check results before timing.
- Run from `/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj`, with `PYTHONPATH=$PWD taskset -c 0 /tmp/gb2/bin/python` before each line below, in order for each of rounds 0, 1, 2:

```sh
benchmarks/bench_launch.py --iters 1000 --batches 9
benchmarks/bench_launch.py --no-gpu --iters 100000 --batches 9
benchmarks/bench_launch.py --readme --iters 1000 --batches 9
benchmarks/bench_launch.py --sweep --iters 100000 --batches 9
benchmarks/bench_ffi_compare.py --iters 1000 --batches 9
benchmarks/bench_ffi_compare.py --sweep --iters 1000 --batches 9
benchmarks/bench_intj_ffi_paths.py --mode all --iters 1000 --batches 9
benchmarks/bench_hip_module_launch.py --iters 1000 --batches 9
```

The C++ kernel cache benchmark lives under `tests/`, outside `benchmarks/`, and is not part of this request. It requires Google Benchmark.

## Raw outputs

### launch_gpu

```text
commit=ef698d24dae8790fffdf84fe766a7bf06db428f6
started=2026-09-25T21:00:19+08:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --iters 1000 --batches 9
mode=gpu; 1000 calls × 9 batches; median ns/call
               path    ns/call       Δ ns       Δ %   build ms   bind ms
           auto map     3115.4       +0.0      +0.0      71.76         -
        reduced key     3235.2     +119.8      +3.8    2122.22         -
         verify off     3212.1      +96.7      +3.1    2099.33         -
          verify on     3196.6      +81.2      +2.6    2210.85         -
              baked     3247.0     +131.6      +4.2    2012.55         -
       bound tensor     3212.7      +97.3      +3.1       0.45   1971.14
      bound pointer     3208.6      +93.2      +3.0       0.43   2005.96
   fixed device map     3191.1      +75.7      +2.4       0.40   1979.94
fixed device no-map     3191.5      +76.1      +2.4       0.43   2015.45
elapsed_ns=20282342347
exit_status=0
ended=2026-09-25T21:00:40+08:00
```

### launch_host

```text
commit=ef698d24dae8790fffdf84fe766a7bf06db428f6
started=2026-09-25T21:00:40+08:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --no-gpu --iters 100000 --batches 9
mode=host-only; 100000 calls × 9 batches; median ns/call
               path    ns/call       Δ ns       Δ %   build ms   bind ms
           auto map       43.4       +0.0      +0.0      59.84         -
        reduced key       43.0       -0.5      -1.1       1.37         -
         verify off       43.8       +0.4      +0.8       1.20         -
          verify on       44.3       +0.8      +1.9       1.21         -
              baked       39.5       -3.9      -9.0       1.37         -
       bound tensor       45.0       +1.5      +3.5       0.13      1.11
      bound pointer       44.6       +1.2      +2.7       0.12      1.40
   fixed device map       44.0       +0.6      +1.3       0.11      1.16
fixed device no-map       44.6       +1.2      +2.7       0.12      1.06
elapsed_ns=3019282684
exit_status=0
ended=2026-09-25T21:00:43+08:00
```

### launch_readme

```text
commit=ef698d24dae8790fffdf84fe766a7bf06db428f6
started=2026-09-25T21:00:43+08:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --readme --iters 1000 --batches 9
mode=readme; 1000 calls × 9 batches; median
        path   triton us    intj us   speedup
   grid=(1,)       16.48       3.14      5.3x
   grid=(0,)       13.16       0.03    466.6x

torch_access   decode ns    build s
        shim        93.3       0.02
         cxx        92.5       0.06
     cpython       876.4       0.00
elapsed_ns=3708777243
exit_status=0
ended=2026-09-25T21:00:46+08:00
```

### launch_sweep

```text
commit=ef698d24dae8790fffdf84fe766a7bf06db428f6
started=2026-09-25T21:00:46+08:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --sweep --iters 100000 --batches 9
mode=sweep; host-only; 100000 calls × 9 batches; median ns/call
count   kind    ns/call
    4    int       41.5
    4 tensor       40.7
   16    int       74.4
   16 tensor       67.9
   32    int      117.6
   32 tensor      126.0
elapsed_ns=3110800258
exit_status=0
ended=2026-09-25T21:00:50+08:00
```

### ffi_default

```text
commit=ef698d24dae8790fffdf84fe766a7bf06db428f6
started=2026-09-25T21:00:50+08:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_ffi_compare.py --iters 1000 --batches 9
iters=1000 batches=9 unit=us/call (host-only, no timed sync)
FFI packed nop             median=0.1440 samples=[0.14971399999999999, 0.143959, 0.142261, 0.143681, 0.14384, 0.145078, 0.144579, 0.14387200000000003, 0.14416900000000002]
FFI typed nop              median=0.1501 samples=[0.15944999999999998, 0.150075, 0.14629, 0.149092, 0.14795, 0.150842, 0.148172, 0.151441, 0.150564]
FFI empty kernel           median=3.3271 samples=[3.517533, 3.690336, 3.334412, 3.230711, 3.2386709999999996, 3.3275949999999996, 3.184067, 3.3271390000000003, 3.176933]
INTJ empty kernel          median=2.9769 samples=[3.077518, 3.020757, 3.123021, 2.858581, 2.80933, 2.9241170000000003, 2.995463, 2.9392139999999998, 2.97688]
INTJ fixed-device kernel   median=2.8441 samples=[2.9399430000000004, 3.006009, 2.9863429999999997, 2.869094, 2.844147, 2.814007, 2.827009, 2.795866, 2.816227]
FFI packed nop mixed       median=0.1749 samples=[0.17358099999999999, 0.17515799999999998, 0.175065, 0.177845, 0.17443199999999998, 0.172548, 0.175897, 0.174897, 0.172232]
FFI typed nop mixed        median=0.1774 samples=[0.176422, 0.17875200000000002, 0.173936, 0.178687, 0.176755, 0.17742, 0.180431, 0.17811600000000002, 0.17459200000000002]
FFI mixed kernel           median=3.2951 samples=[3.374031, 3.553988, 3.2155430000000003, 3.358347, 3.245254, 3.295133, 3.254924, 3.35725, 3.2459290000000003]
INTJ mixed kernel          median=2.8363 samples=[2.997061, 3.014056, 2.818835, 2.873757, 2.83633, 2.82874, 2.81737, 2.943348, 2.826099]
INTJ fixed mixed kernel    median=2.8890 samples=[3.025038, 3.035428, 2.927679, 2.9134189999999998, 2.831136, 2.820746, 2.873179, 2.8890100000000003, 2.8513699999999997]
elapsed_ns=3812847690
exit_status=0
ended=2026-09-25T21:00:53+08:00
```

### ffi_sweep

```text
commit=ef698d24dae8790fffdf84fe766a7bf06db428f6
started=2026-09-25T21:00:53+08:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_ffi_compare.py --sweep --iters 1000 --batches 9
counts=[0, 3, 5, 8, 16, 32, 64] iters=1000 batches=9 unit=us/call (host-only, no timed sync); INTJ device-bound, no specialization key
args= 0 FFI packed nop           median=0.1183 samples=[0.120172, 0.28662, 0.251493, 0.115312, 0.11546899999999999, 0.115225, 0.11716, 0.11829300000000001, 0.119687]
args= 0 FFI typed nop            median=0.1205 samples=[0.120658, 0.244702, 0.312969, 0.119202, 0.11630700000000001, 0.117316, 0.117701, 0.12053900000000001, 0.121798]
args= 0 FFI empty kernel         median=1.8630 samples=[1.619657, 13.888691999999999, 3.011791, 1.940258, 1.576025, 1.863032, 1.6451500000000001, 1.980393, 1.564092]
args= 0 INTJ cxx kernel          median=2.7273 samples=[3.058794, 3.832462, 2.727348, 2.6931019999999997, 2.694813, 2.667493, 2.761421, 2.676357, 2.7787379999999997]
args= 0 INTJ shim kernel         median=2.6992 samples=[2.828447, 2.6861599999999997, 2.726958, 2.669305, 2.6991590000000003, 2.6825590000000004, 2.773318, 2.661765, 2.731324]
args= 3 FFI packed nop           median=0.1443 samples=[0.14443999999999999, 0.143538, 0.14313, 0.14506899999999998, 0.144267, 0.1422, 0.144394, 0.144632, 0.143916]
args= 3 FFI typed nop            median=0.1477 samples=[0.151149, 0.147679, 0.14631899999999998, 0.14906, 0.14613800000000002, 0.14851499999999998, 0.146254, 0.148446, 0.147726]
args= 3 FFI empty kernel         median=3.2776 samples=[3.277573, 3.2851559999999997, 3.156371, 3.517982, 3.1371379999999998, 3.282525, 3.14906, 3.28954, 3.150033]
args= 3 INTJ cxx kernel          median=2.9308 samples=[2.924165, 2.8859209999999997, 2.967232, 2.864328, 3.0812109999999997, 2.9308159999999996, 2.9450439999999998, 2.861642, 3.00379]
args= 3 INTJ shim kernel         median=2.8146 samples=[2.951683, 2.772738, 2.800014, 2.806366, 2.866873, 2.8145830000000003, 2.8761750000000004, 2.804568, 2.854184]
args= 5 FFI packed nop           median=0.1743 samples=[0.173875, 0.17802500000000002, 0.174254, 0.1724, 0.17940299999999998, 0.174393, 0.178072, 0.174186, 0.173749]
args= 5 FFI typed nop            median=0.1788 samples=[0.17852099999999999, 0.180914, 0.17879699999999998, 0.179232, 0.178089, 0.180899, 0.177732, 0.182948, 0.176137]
args= 5 FFI empty kernel         median=3.2922 samples=[3.292177, 3.303335, 3.193044, 3.295807, 3.189824, 3.299616, 3.2675680000000003, 3.3086979999999997, 3.1912409999999998]
args= 5 INTJ cxx kernel          median=2.8858 samples=[2.995189, 2.904549, 2.846374, 2.837472, 2.847538, 2.913945, 2.885835, 2.839025, 2.9833670000000003]
args= 5 INTJ shim kernel         median=2.7923 samples=[2.9777020000000003, 2.7922919999999998, 2.8271729999999997, 2.781182, 2.774037, 2.780742, 2.861464, 2.776472, 2.844883]
args= 8 FFI packed nop           median=0.2105 samples=[0.210508, 0.210778, 0.209399, 0.21048599999999998, 0.21048599999999998, 0.21059, 0.21051599999999998, 0.207809, 0.211485]
args= 8 FFI typed nop            median=0.2129 samples=[0.212282, 0.21553899999999998, 0.211868, 0.21684399999999998, 0.212041, 0.21548599999999998, 0.21802000000000002, 0.21158000000000002, 0.21288200000000002]
args= 8 FFI empty kernel         median=3.3914 samples=[3.496953, 3.379542, 3.274765, 3.400892, 3.26423, 3.3976439999999997, 3.5178510000000003, 3.391428, 3.282151]
args= 8 INTJ cxx kernel          median=3.0240 samples=[3.065863, 2.862783, 3.097133, 2.994287, 3.0970500000000003, 3.0022699999999998, 3.024024, 2.976534, 3.0858220000000003]
args= 8 INTJ shim kernel         median=2.8987 samples=[3.0427779999999998, 2.854639, 2.955047, 2.898674, 2.876068, 2.891622, 2.8655999999999997, 2.938399, 2.949932]
args=16 FFI packed nop           median=0.3043 samples=[0.304298, 0.30638099999999996, 0.304296, 0.30399, 0.30577499999999996, 0.302146, 0.305586, 0.301115, 0.30377499999999996]
args=16 FFI typed nop            median=0.3117 samples=[0.30649, 0.31422500000000003, 0.310635, 0.31440300000000004, 0.315743, 0.309533, 0.310996, 0.311668, 0.311745]
args=16 FFI empty kernel         median=3.6360 samples=[3.6823609999999998, 3.657062, 3.518358, 3.671384, 3.540424, 3.65388, 3.6334899999999997, 3.636031, 3.5185929999999996]
args=16 INTJ cxx kernel          median=3.3753 samples=[3.263722, 3.375318, 3.4542759999999997, 3.308652, 3.495835, 3.3667109999999996, 3.413212, 3.242929, 3.409308]
args=16 INTJ shim kernel         median=3.2289 samples=[3.2289090000000003, 3.087814, 3.3388359999999997, 3.04379, 3.31821, 3.044926, 3.284686, 3.032623, 3.355918]
args=32 FFI packed nop           median=0.4952 samples=[0.48578899999999997, 0.503176, 0.491954, 0.49997800000000003, 0.49518, 0.48439699999999997, 0.524015, 0.485429, 0.501753]
args=32 FFI typed nop            median=0.5041 samples=[0.509242, 0.520752, 0.500024, 0.499685, 0.510749, 0.5035729999999999, 0.500111, 0.506158, 0.504074]
args=32 FFI empty kernel         median=4.2456 samples=[4.257643, 4.210315, 4.094558, 4.250893, 4.094662, 5.0069, 4.112121, 4.258966999999999, 4.245640000000001]
args=32 INTJ cxx kernel          median=3.6394 samples=[3.4652730000000003, 3.58557, 3.682739, 3.580806, 3.6526080000000003, 19.109057, 3.639396, 3.546663, 3.6825680000000003]
args=32 INTJ shim kernel         median=3.4714 samples=[3.4713600000000002, 3.411505, 3.637812, 3.4683479999999998, 3.5754430000000004, 3.4508539999999996, 3.616851, 3.457011, 3.648524]
args=64 FFI packed nop           median=0.8728 samples=[0.865155, 0.889953, 0.8700589999999999, 0.886602, 0.911119, 0.8727569999999999, 0.858673, 0.8701639999999999, 0.875574]
args=64 FFI typed nop            median=0.9015 samples=[0.865811, 0.905649, 0.888091, 0.920088, 0.927212, 0.901479, 0.892153, 0.887349, 0.923475]
args=64 FFI empty kernel         median=5.0869 samples=[5.087155, 5.053326, 5.094354, 5.090778, 5.077311, 5.088951, 5.049105, 5.056782, 5.086895999999999]
args=64 INTJ cxx kernel          median=4.5941 samples=[4.554572, 4.530509, 4.608099, 4.611852, 4.594086, 4.601877, 4.579648, 4.598801000000001, 4.575273]
args=64 INTJ shim kernel         median=4.5630 samples=[4.508327, 4.563029, 4.5777969999999994, 4.619817, 4.557134, 4.614101, 4.538946, 4.614581, 4.539968]
elapsed_ns=4286669614
exit_status=0
ended=2026-09-25T21:00:58+08:00
```

### ffi_paths

```text
commit=ef698d24dae8790fffdf84fe766a7bf06db428f6
started=2026-09-25T21:00:58+08:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_intj_ffi_paths.py --mode all --iters 1000 --batches 9
iters=1000 batches=9 unit=ns/call; no_gpu=True
mode=callback
callback_count=9100 (100 warmup + 9000 unique misses)
INTJ cold compile callback + cache  median_ns=423.6 samples_ns=[460.86, 451.602, 400.22, 541.752, 378.732, 403.342, 409.166, 423.605, 1533.224]
INTJ hot call (no callback)         median_ns=39.3 samples_ns=[42.608, 40.555, 38.538, 39.577, 38.52, 39.168, 38.386, 39.332, 42.859]
INTJ 3-tensor host-only nop         median_ns=35.6 samples_ns=[36.524, 35.868, 35.28, 35.65, 34.896, 35.623, 35.067, 35.597, 35.117]
mode=kwargs
INTJ direct positional              median_ns=66.7 samples_ns=[68.159, 67.126, 71.01, 66.681, 66.705, 66.684, 66.396, 66.489, 66.504]
INTJ adapter positional             median_ns=90.9 samples_ns=[92.941, 91.16, 89.325, 89.928, 90.136, 106.617, 92.576, 90.705, 90.9]
INTJ adapter kwargs                 median_ns=107.5 samples_ns=[109.059, 107.488, 106.897, 107.024, 106.292, 107.053, 122.358, 109.277, 107.565]
INTJ adapter defaults               median_ns=89.5 samples_ns=[91.991, 89.432, 89.65, 88.246, 89.216, 89.581, 89.457, 92.892, 88.377]
INTJ FFI wrapper positional         median_ns=103.6 samples_ns=[104.881, 102.531, 103.29, 103.303, 107.225, 103.761, 103.533, 103.591, 113.281]
INTJ FFI wrapper kwargs             median_ns=128.9 samples_ns=[129.396, 128.144, 128.856, 127.846, 127.446, 126.568, 133.822, 129.917, 129.722]
INTJ native kwargs: unsupported (verified TypeError)
mode=dataclass
INTJ pair direct                    median_ns=68.4 samples_ns=[68.765, 67.956, 71.206, 68.374, 67.89, 68.518, 67.758, 68.604, 67.562]
INTJ pair prebuilt *tuple           median_ns=142.8 samples_ns=[142.021, 145.93, 142.842, 147.25, 142.57, 144.281, 140.786, 145.572, 140.511]
FFI unpack Pair only                median_ns=162.8 samples_ns=[165.579, 176.559, 161.968, 162.776, 161.709, 164.909, 161.693, 168.381, 162.024]
INTJ pair manual unpack             median_ns=173.5 samples_ns=[173.009, 175.045, 171.969, 175.81, 174.155, 173.527, 171.712, 173.795, 171.259]
INTJ pair FFI unpack                median_ns=315.2 samples_ns=[318.327, 314.937, 313.839, 318.983, 312.262, 315.012, 315.168, 319.008, 315.608]
INTJ pair stdlib astuple            median_ns=1161.8 samples_ns=[1322.415, 1209.697, 1150.917, 1161.75, 1152.645, 1161.19, 1167.471, 1164.787, 1154.0]
INTJ config direct                  median_ns=81.7 samples_ns=[81.725, 81.995, 81.28, 81.992, 81.474, 85.283, 81.37, 82.034, 81.104]
INTJ config FFI unpack              median_ns=326.6 samples_ns=[322.415, 326.996, 326.585, 326.942, 322.516, 331.007, 323.713, 327.823, 325.623]
INTJ native dataclass unpack: unsupported (verified TypeError)
elapsed_ns=3010289595
exit_status=0
ended=2026-09-25T21:01:01+08:00
```

### hip_same_hsaco

```text
commit=ef698d24dae8790fffdf84fe766a7bf06db428f6
started=2026-09-25T21:01:01+08:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_hip_module_launch.py --iters 1000 --batches 9
hsaco_sha256=df3e529e178196ad62f7c07ab8593ede5b8596c76e7c6fbd12a5d947a696e65c kernel=empty_kernel threads=256 shared=0 callback_count=1
iters=1000 batches=9 unit=us/call (host enqueue, no timed sync)
TVM FFI HIP HSACO         median=3.1706 samples=[3.363039, 3.1931309999999997, 3.213932, 3.169824, 3.16136, 3.0752979999999996, 3.1705569999999996, 3.09519, 3.182275]
Triton same HSACO         median=15.8787 samples=[16.032558, 15.956642, 21.490818, 15.909618, 15.816668, 15.782759, 15.819787, 15.807082000000001, 15.878736]
INTJ same function        median=2.9440 samples=[2.9970700000000003, 2.961141, 2.941017, 2.943989, 2.9050949999999998, 2.916617, 2.934144, 2.9599729999999997, 2.953717]
elapsed_ns=3743106589
exit_status=0
ended=2026-09-25T21:01:04+08:00
```

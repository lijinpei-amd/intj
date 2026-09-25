# Exact merged `develop` benchmark: `38cdabc`

Measured September 25, 2026 (Asia/Shanghai) on clean local `develop` at `38cdabc604b6c9a1a3ea57667e1f942fc5bc7960`. All eight modes of the four Python scripts in `benchmarks/` ran serially for three separate processes each. The earlier [pre-merge develop run](2026-09-25_develop-full_0_ef698d2.md) measured `ef698d2` with the same flags; the old numbers below are historical and were not interleaved with this run.

## Median of three process medians

| Case | pre-merge develop | exact merge | change | exact merge rounds 0, 1, 2 |
| --- | ---: | ---: | ---: | --- |
| GPU auto-map launcher (ns) | 3590.200 | 3085.600 | -14.1% | 3085.600, 3085.000, 3117.500 |
| GPU bound-pointer control (ns) | 3641.700 | 2994.900 | -17.8% | 3144.400, 2991.800, 2994.900 |
| GPU fixed-device no-map control (ns) | 3540.000 | 2908.200 | -17.8% | 2892.300, 3014.600, 2908.200 |
| Host auto-map launcher (ns) | 43.500 | 43.700 | +0.5% | 43.700, 43.600, 43.700 |
| README grid=(1,) INTJ (µs) | 3.130 | 3.120 | -0.3% | 3.120, 3.110, 3.140 |
| Host sweep 32 tensors (ns) | 120.200 | 119.600 | -0.5% | 118.100, 167.000, 119.600 |
| FFI comparison INTJ empty (µs) | 2.988 | 2.994 | +0.2% | 4.055, 2.974, 2.994 |
| FFI sweep 64 INTJ static/CXX (µs) | 4.599 | 4.622 | +0.5% | 4.622, 4.627, 4.606 |
| FFI sweep 64 FFI GPU (µs) | 5.087 | 5.065 | -0.4% | 5.071, 5.065, 5.057 |
| FFI utilities INTJ hot call (ns) | 38.200 | 39.200 | +2.6% | 40.800, 39.200, 39.200 |
| Same-HSACO INTJ (µs) | 2.944 | 3.044 | +3.4% | 3.065, 3.044, 2.917 |
| Same-HSACO FFI (µs) | 3.171 | 3.198 | +0.9% | 3.288, 3.198, 3.152 |

The -14% GPU auto-map change is not evidence of a corresponding software speedup: GPU bound-pointer and no-map controls moved by similar or larger amounts, and the old processes were run earlier rather than interleaved. CPU host auto-map, README grid=(1,), FFI sweep and same-HSACO comparisons are much closer. The FFI default INTJ empty row has one slow merged process; retain all samples instead of discarding it. These timings measure host call/enqueue time; the host timer stops before GPU synchronization. Long batches can fill the GPU queue. FFI packed no-ops do not launch kernels; preconverted FFI arguments differ from INTJ Torch tensors, and `bench_ffi_compare.py` uses different GPU binaries. Only `bench_hip_module_launch.py` uses the same HSACO across its FFI, Triton and INTJ paths. Cold callback, Python keyword adapters, and dataclass helpers in `bench_intj_ffi_paths.py` have separate boundaries. CUDA runtime was not tested on this ROCm host.

## Environment and commands

- Host x86-64, CPU pinned to core 0 with `taskset -c 0`. AMD Instinct MI308X GPU, architecture `gfx942:sramecc+:xnack-`; ROCm tooling showed GPU use 0% and no KFD process at the start and after the run. One benchmark ran at a time; every script warmed its cases and verified output where it wrote data.
- Python 3.12.3 (`/tmp/gb2/bin/python`), Torch 2.14.0+rocm7.2 (C++11 ABI enabled), Triton 3.8.0, TVM FFI 0.1.14.post2.dev1+g424558557.d20260924; extension suffix `.cpython-312-x86_64-linux-gnu.so`.
- `TRITON_HOME`, `CXX`, `CC`, `INTJ_BENCHMARK_ROOT` unset. Default caches; default `c++` and `cc`: Ubuntu GCC 13.3.0. Compilation and argument construction occurred outside timed loops.
- From `/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj`, prefix each line below with `PYTHONPATH=$PWD taskset -c 0 /tmp/gb2/bin/python`. Run all eight lines in order for each round 0, 1, 2. The README mode uses 1,000 calls and 9 batches to match the earlier recorded revision and avoid the documented long-batch GPU queue effect.

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

The separate `tests/test_kernel_cache.py` benchmark entry was invoked once using `PYTHONPATH=$PWD taskset -c 0 /tmp/gb2/bin/python -m pytest tests/test_kernel_cache.py -s -q`. It skipped because Google Benchmark was unavailable, so there is no kernel-cache timing. Full raw output is included in round 0. `bench_launch.py` itself reports per-case medians. A separate recorder below captures its individual batch samples after each timed loop without editing the measured source; the FFI/HSACO scripts already report individual samples.

## Raw outputs

### launch_gpu

```text
commit=38cdabc604b6c9a1a3ea57667e1f942fc5bc7960
started=2026-09-25T21:31:16+08:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --iters 1000 --batches 9
mode=gpu; 1000 calls × 9 batches; median ns/call
               path    ns/call       Δ ns       Δ %   build ms   bind ms
           auto map     3085.6       +0.0      +0.0      72.26         -
        reduced key     3075.7       -9.9      -0.3       3.45         -
         verify off     2974.5     -111.1      -3.6       1.73         -
          verify on     2919.5     -166.1      -5.4       1.64         -
              baked     2951.6     -134.0      -4.3      73.14         -
       bound tensor     3616.6     +531.0     +17.2       0.58      2.84
      bound pointer     3144.4      +58.8      +1.9       0.38      1.51
   fixed device map     2919.1     -166.5      -5.4       0.34      1.51
fixed device no-map     2892.3     -193.3      -6.3       0.27      1.40
elapsed_ns=4006990676
exit_status=0
ended=2026-09-25T21:31:20+08:00
```

### launch_host

```text
commit=38cdabc604b6c9a1a3ea57667e1f942fc5bc7960
started=2026-09-25T21:31:20+08:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --no-gpu --iters 100000 --batches 9
mode=host-only; 100000 calls × 9 batches; median ns/call
               path    ns/call       Δ ns       Δ %   build ms   bind ms
           auto map       43.7       +0.0      +0.0      60.09         -
        reduced key       43.7       +0.0      +0.0       1.51         -
         verify off       43.7       +0.0      +0.0       1.40         -
          verify on       44.6       +0.9      +2.1       1.37         -
              baked       39.9       -3.8      -8.6       2.76         -
       bound tensor       45.9       +2.2      +5.1       0.16      1.32
      bound pointer       65.2      +21.5     +49.2       0.63      4.83
   fixed device map       45.1       +1.4      +3.1       0.13      1.35
fixed device no-map       43.2       -0.5      -1.2       0.15      1.22
elapsed_ns=3066632739
exit_status=0
ended=2026-09-25T21:31:23+08:00
```

### launch_readme

```text
commit=38cdabc604b6c9a1a3ea57667e1f942fc5bc7960
started=2026-09-25T21:31:23+08:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --readme --iters 1000 --batches 9
mode=readme; 1000 calls × 9 batches; median
        path   triton us    intj us   speedup
   grid=(1,)       16.57       3.12      5.3x
   grid=(0,)       13.20       0.03    461.2x

torch_access_mode   decode ns    build s
     runtime_shim        90.4       0.02
   static_compile        90.5       0.07
      interpreter       875.1       0.00
elapsed_ns=3948200006
exit_status=0
ended=2026-09-25T21:31:27+08:00
```

### launch_sweep

```text
commit=38cdabc604b6c9a1a3ea57667e1f942fc5bc7960
started=2026-09-25T21:31:27+08:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --sweep --iters 100000 --batches 9
mode=sweep; host-only; 100000 calls × 9 batches; median ns/call
count   kind    ns/call
    4    int       42.5
    4 tensor       41.5
   16    int       73.9
   16 tensor       68.9
   32    int      120.2
   32 tensor      118.1
elapsed_ns=3266965062
exit_status=0
ended=2026-09-25T21:31:30+08:00
```

### ffi_default

```text
commit=38cdabc604b6c9a1a3ea57667e1f942fc5bc7960
started=2026-09-25T21:31:30+08:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_ffi_compare.py --iters 1000 --batches 9
iters=1000 batches=9 unit=us/call (host-only, no timed sync)
FFI packed nop             median=0.2333 samples=[0.241123, 0.241189, 0.23405, 0.237947, 0.23333600000000002, 0.207004, 0.206285, 0.142784, 0.145144]
FFI typed nop              median=0.2420 samples=[0.257238, 0.241978, 0.253044, 0.26239, 0.25389, 0.215379, 0.215022, 0.15010300000000001, 0.143248]
FFI empty kernel           median=4.5408 samples=[4.540763, 4.643999, 4.624121, 4.516971, 4.510671, 7.0841009999999995, 6.1289750000000005, 4.325773, 3.575952]
INTJ empty kernel          median=4.0549 samples=[3.970635, 4.078259, 4.254072, 3.9956869999999998, 4.054931, 4.017322, 4.917001, 3.748965, 4.058893]
INTJ fixed-device kernel   median=4.0681 samples=[4.27159, 4.127575, 4.193218, 3.992922, 4.0681460000000005, 4.0498330000000005, 4.244643, 3.2524580000000003, 3.405669]
FFI packed nop mixed       median=0.2643 samples=[0.26429899999999995, 0.28388, 0.290831, 0.289658, 0.278509, 0.247806, 0.229677, 0.207131, 0.17354499999999998]
FFI typed nop mixed        median=0.2756 samples=[0.275584, 0.293605, 0.295, 0.30266000000000004, 0.30776400000000004, 0.254595, 0.236559, 0.252656, 0.177143]
FFI mixed kernel           median=4.7025 samples=[4.826613, 4.7488280000000005, 4.690421000000001, 4.619345999999999, 4.639934, 8.070259, 4.7024740000000005, 5.466363, 3.751004]
INTJ mixed kernel          median=4.0718 samples=[4.986841, 4.172736, 4.032960999999999, 3.977458, 4.155014, 3.799571, 4.26552, 4.071825, 3.40959]
INTJ fixed mixed kernel    median=3.9952 samples=[4.135197, 4.128632, 4.047924, 4.0445720000000005, 3.952384, 3.852788, 3.993844, 3.995154, 3.4519830000000002]
elapsed_ns=4148224538
exit_status=0
ended=2026-09-25T21:31:34+08:00
```

### ffi_sweep

```text
commit=38cdabc604b6c9a1a3ea57667e1f942fc5bc7960
started=2026-09-25T21:31:34+08:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_ffi_compare.py --sweep --iters 1000 --batches 9
counts=[0, 3, 5, 8, 16, 32, 64] iters=1000 batches=9 unit=us/call (host-only, no timed sync); INTJ device-bound, no specialization key
args= 0 FFI packed nop           median=0.1159 samples=[0.12188299999999999, 0.11629099999999999, 0.117883, 0.114782, 0.115083, 0.11519199999999999, 0.120515, 0.115943, 0.114939]
args= 0 FFI typed nop            median=0.1202 samples=[0.12214, 0.123692, 0.117593, 0.120467, 0.119438, 0.120233, 0.116757, 0.120984, 0.11722]
args= 0 FFI empty kernel         median=1.6652 samples=[1.6377650000000001, 1.845728, 1.615195, 1.925251, 1.665202, 1.833957, 1.576673, 1.873562, 1.588178]
args= 0 INTJ static_compile kernel median=2.7123 samples=[3.145178, 2.6497379999999997, 2.714452, 2.770723, 2.7629859999999997, 2.685416, 2.702295, 2.674703, 2.71233]
args= 0 INTJ runtime_shim kernel median=2.7011 samples=[2.8804830000000003, 2.7007689999999998, 2.732132, 2.77903, 2.713356, 2.680798, 2.701076, 2.686412, 2.699871]
args= 3 FFI packed nop           median=0.1453 samples=[0.145323, 0.14677, 0.14246899999999998, 0.144264, 0.144413, 0.149685, 0.146342, 0.147417, 0.143761]
args= 3 FFI typed nop            median=0.1498 samples=[0.147221, 0.150156, 0.15074500000000002, 0.147468, 0.149827, 0.152477, 0.14790899999999998, 0.151234, 0.14894900000000003]
args= 3 FFI empty kernel         median=3.2067 samples=[3.356516, 3.206667, 3.1561280000000003, 3.329151, 3.144139, 3.235369, 3.131732, 3.2309699999999997, 3.144655]
args= 3 INTJ static_compile kernel median=2.9347 samples=[3.0195, 2.881383, 2.920579, 2.982003, 2.942142, 2.900688, 2.9386370000000004, 2.9056599999999997, 2.934674]
args= 3 INTJ runtime_shim kernel median=2.8847 samples=[3.0117190000000003, 2.877692, 2.875016, 2.989582, 2.884685, 2.903751, 2.884705, 2.908931, 2.88354]
args= 5 FFI packed nop           median=0.1735 samples=[0.169798, 0.171986, 0.174213, 0.170779, 0.17522, 0.173323, 0.173518, 0.174741, 0.17480600000000002]
args= 5 FFI typed nop            median=0.1784 samples=[0.17402, 0.178374, 0.17668199999999998, 0.19124000000000002, 0.17690799999999998, 0.178904, 0.1784, 0.178767, 0.17652600000000002]
args= 5 FFI empty kernel         median=3.2494 samples=[3.38115, 3.242889, 3.241193, 3.324387, 3.2567429999999997, 3.243306, 3.24939, 3.269505, 3.249445]
args= 5 INTJ static_compile kernel median=2.8994 samples=[3.006192, 2.914721, 2.86756, 3.01723, 2.88008, 2.8800149999999998, 2.899436, 2.9297, 2.88627]
args= 5 INTJ runtime_shim kernel median=2.9037 samples=[3.028022, 3.0103429999999998, 2.8816840000000004, 3.062589, 2.882905, 2.903665, 2.894758, 2.954922, 2.885098]
args= 8 FFI packed nop           median=0.2085 samples=[0.204685, 0.2085, 0.210189, 0.208998, 0.206595, 0.208872, 0.210172, 0.207681, 0.20403200000000002]
args= 8 FFI typed nop            median=0.2091 samples=[0.209101, 0.212057, 0.208998, 0.211179, 0.21068199999999998, 0.209042, 0.20908500000000002, 0.209549, 0.20475]
args= 8 FFI empty kernel         median=3.4646 samples=[3.553983, 3.3992020000000003, 3.4831790000000002, 3.4645520000000003, 3.301024, 3.3771489999999997, 3.526468, 3.359984, 3.509467]
args= 8 INTJ static_compile kernel median=3.0331 samples=[3.0757060000000003, 2.9817489999999998, 3.025446, 3.8089079999999997, 3.052259, 3.033143, 3.0206239999999998, 3.0144830000000002, 3.041524]
args= 8 INTJ runtime_shim kernel median=2.8850 samples=[3.103499, 2.9425320000000004, 2.885038, 13.074605, 2.873201, 2.872481, 2.9029119999999997, 2.881501, 2.8759259999999998]
args=16 FFI packed nop           median=0.2963 samples=[0.289695, 0.296313, 0.297028, 0.521711, 0.29838099999999995, 0.29152300000000003, 0.29750099999999996, 0.294736, 0.294005]
args=16 FFI typed nop            median=0.3066 samples=[0.298553, 0.30761099999999997, 0.309851, 0.455034, 0.304699, 0.30462, 0.30655099999999996, 0.303762, 0.306627]
args=16 FFI empty kernel         median=3.6607 samples=[3.8061529999999997, 3.734667, 3.5917, 20.47816, 3.5709940000000002, 3.6606840000000003, 3.585274, 3.703202, 3.557451]
args=16 INTJ static_compile kernel median=3.3256 samples=[3.270846, 3.373492, 3.396616, 12.483450999999999, 3.420283, 3.291007, 3.307654, 3.325634, 3.29517]
args=16 INTJ runtime_shim kernel median=3.2524 samples=[3.2620189999999996, 3.138055, 3.246056, 3.584142, 3.304891, 3.190382, 3.398558, 3.037778, 3.2524189999999997]
args=32 FFI packed nop           median=0.4802 samples=[0.476948, 0.486341, 0.480245, 0.48433, 0.49026400000000003, 0.475065, 0.48637400000000003, 0.479934, 0.478868]
args=32 FFI typed nop            median=0.4998 samples=[0.490894, 0.503858, 0.48704000000000003, 0.499824, 0.499667, 0.5149900000000001, 0.500263, 0.495319, 0.515736]
args=32 FFI empty kernel         median=4.2227 samples=[4.290268, 4.263876, 4.101795, 4.246169, 4.093289, 4.224475999999999, 4.071751, 4.222732, 4.1256509999999995]
args=32 INTJ static_compile kernel median=3.6072 samples=[3.4528600000000003, 3.6273690000000003, 3.607211, 3.574265, 3.625764, 3.533599, 3.6481280000000003, 3.557882, 3.757516]
args=32 INTJ runtime_shim kernel median=3.5067 samples=[3.459342, 3.506709, 3.5863359999999997, 3.458531, 3.606181, 3.434536, 3.605966, 3.4441829999999998, 3.626462]
args=64 FFI packed nop           median=0.8550 samples=[0.8485069999999999, 0.877283, 0.848635, 0.858567, 0.8682770000000001, 0.8595929999999999, 0.841481, 0.8550359999999999, 0.8388410000000001]
args=64 FFI typed nop            median=0.8808 samples=[0.866256, 0.906201, 0.8690370000000001, 0.891301, 0.884225, 0.894511, 0.837711, 0.880768, 0.862873]
args=64 FFI empty kernel         median=5.0708 samples=[5.121238, 5.15069, 5.087648, 5.080783, 5.0707510000000005, 5.040612, 5.040362, 5.058566, 5.018933]
args=64 INTJ static_compile kernel median=4.6219 samples=[4.707496, 4.689228, 4.63618, 4.588112, 4.621901, 4.620773, 4.632599, 4.596163, 4.619253]
args=64 INTJ runtime_shim kernel median=4.5624 samples=[4.56225, 4.623858, 4.597409, 4.562407, 4.545783999999999, 4.609814, 4.533721, 4.640653, 4.539238999999999]
elapsed_ns=4619341299
exit_status=0
ended=2026-09-25T21:31:39+08:00
```

### ffi_paths

```text
commit=38cdabc604b6c9a1a3ea57667e1f942fc5bc7960
started=2026-09-25T21:31:39+08:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_intj_ffi_paths.py --mode all --iters 1000 --batches 9
iters=1000 batches=9 unit=ns/call; no_gpu=True
mode=callback
callback_count=9100 (100 warmup + 9000 unique misses)
INTJ cold compile callback + cache  median_ns=423.9 samples_ns=[456.43, 455.389, 410.304, 497.089, 385.725, 401.821, 413.865, 423.907, 1063.207]
INTJ hot call (no callback)         median_ns=40.8 samples_ns=[41.952, 40.954, 41.518, 40.811, 39.946, 40.71, 39.682, 40.918, 39.539]
INTJ 3-tensor host-only nop         median_ns=35.4 samples_ns=[37.854, 36.044, 35.182, 35.533, 35.065, 35.448, 35.184, 35.507, 35.156]
mode=kwargs
INTJ direct positional              median_ns=66.7 samples_ns=[68.978, 66.897, 66.676, 66.658, 66.554, 66.665, 66.603, 66.57, 66.679]
INTJ adapter positional             median_ns=91.6 samples_ns=[90.624, 89.792, 90.941, 94.941, 92.826, 92.569, 92.168, 91.477, 91.565]
INTJ adapter kwargs                 median_ns=109.6 samples_ns=[106.632, 105.662, 104.34, 104.788, 118.699, 110.011, 110.679, 109.574, 111.267]
INTJ adapter defaults               median_ns=90.9 samples_ns=[90.71, 90.148, 90.248, 91.819, 90.215, 93.976, 91.201, 91.067, 90.911]
INTJ FFI wrapper positional         median_ns=104.3 samples_ns=[105.379, 105.093, 104.306, 104.305, 103.982, 103.986, 107.714, 103.396, 104.046]
INTJ FFI wrapper kwargs             median_ns=124.5 samples_ns=[128.209, 124.526, 124.391, 124.031, 125.528, 132.624, 127.713, 123.285, 123.265]
INTJ native kwargs: unsupported (verified TypeError)
mode=dataclass
INTJ pair direct                    median_ns=68.2 samples_ns=[70.17, 68.775, 67.908, 68.386, 67.653, 68.261, 68.193, 68.104, 67.308]
INTJ pair prebuilt *tuple           median_ns=142.5 samples_ns=[146.161, 143.249, 142.222, 144.785, 140.61, 142.534, 139.761, 144.683, 137.279]
FFI unpack Pair only                median_ns=163.0 samples_ns=[163.806, 164.224, 163.039, 163.202, 166.155, 162.821, 159.891, 161.455, 159.963]
INTJ pair manual unpack             median_ns=172.1 samples_ns=[170.905, 183.375, 170.606, 172.784, 172.125, 173.972, 171.985, 176.311, 169.973]
INTJ pair FFI unpack                median_ns=315.2 samples_ns=[312.773, 315.209, 316.053, 313.995, 313.793, 319.122, 313.599, 315.776, 318.956]
INTJ pair stdlib astuple            median_ns=1148.7 samples_ns=[1313.292, 1191.349, 1137.305, 1151.736, 1133.276, 1146.375, 1139.44, 1154.419, 1148.669]
INTJ config direct                  median_ns=83.7 samples_ns=[83.715, 84.072, 93.163, 84.381, 82.21, 84.184, 82.465, 82.918, 82.649]
INTJ config FFI unpack              median_ns=321.8 samples_ns=[319.804, 327.395, 318.253, 321.48, 323.307, 321.8, 328.072, 330.687, 320.024]
INTJ native dataclass unpack: unsupported (verified TypeError)
elapsed_ns=2999834699
exit_status=0
ended=2026-09-25T21:31:42+08:00
```

### hip_same_hsaco

```text
commit=38cdabc604b6c9a1a3ea57667e1f942fc5bc7960
started=2026-09-25T21:31:42+08:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_hip_module_launch.py --iters 1000 --batches 9
hsaco_sha256=df3e529e178196ad62f7c07ab8593ede5b8596c76e7c6fbd12a5d947a696e65c kernel=empty_kernel threads=256 shared=0 callback_count=1
iters=1000 batches=9 unit=us/call (host enqueue, no timed sync)
TVM FFI HIP HSACO         median=3.2885 samples=[3.493711, 3.341549, 3.34377, 3.2884510000000002, 3.308147, 3.196088, 3.217543, 3.197721, 3.194226]
Triton same HSACO         median=16.3319 samples=[16.545724, 16.432863, 16.399398, 16.39278, 16.320138999999998, 16.331945, 16.317781999999998, 16.275262, 16.302404]
INTJ same function        median=3.0647 samples=[3.197782, 3.174813, 3.117381, 3.119707, 3.06471, 2.985337, 3.003009, 3.0033119999999998, 2.942605]
elapsed_ns=3970024572
exit_status=0
ended=2026-09-25T21:31:46+08:00
```

### kernel_cache (skipped)

```text
commit=38cdabc604b6c9a1a3ea57667e1f942fc5bc7960
started=2026-09-25T21:33:15+08:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj taskset -c 0 /tmp/gb2/bin/python -m pytest tests/test_kernel_cache.py -s -q
s
1 skipped in 0.04s
elapsed_ns=240265865
exit_status=0
ended=2026-09-25T21:33:16+08:00
```

## Launcher per-batch diagnostics (supplemental)

The direct-run medians above remain the primary comparison. `bench_launch.py` prints only medians, so the following second pass logs each list of nine batch samples by intercepting `statistics.median` after its timed loop. This changes output between benchmark cases and could affect later-case timings; do not substitute these medians for the direct-run aggregate. The measured `bench_launch.py` is still the clean `38cdabc` source. Three separate processes per mode, pinned CPU 0; each output block contains its exact command. Driver source:

```python
"""Print each bench_launch.py batch list after statistics.median, outside its timer."""
import os
import runpy
import statistics
import sys

original_median = statistics.median


def record_median(samples):
    result = original_median(samples)
    if isinstance(samples, list) and len(samples) == 9 and all(isinstance(x, float) for x in samples):
        print('recorded_batch_samples_ns=' + repr(samples), flush=True)
    return result


statistics.median = record_median
script = sys.argv[1]
sys.argv = [script, *sys.argv[2:]]
sys.path.insert(0, os.getcwd())
runpy.run_path(script, run_name='__main__')
```

### launch_gpu per-batch, round 0

```text
commit=38cdabc604b6c9a1a3ea57667e1f942fc5bc7960
started=2026-09-25T21:35:09+08:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj taskset -c 0 /tmp/gb2/bin/python /tmp/intj-launch-batch-recorder-38cdabc.py benchmarks/bench_launch.py --iters 1000 --batches 9
mode=gpu; 1000 calls × 9 batches; median ns/call
               path    ns/call       Δ ns       Δ %   build ms   bind ms
recorded_batch_samples_ns=[3153.148, 3485.033, 3109.722, 3122.172, 3091.794, 3120.597, 3112.601, 3125.3, 3104.3]
           auto map     3120.6       +0.0      +0.0      93.84         -
recorded_batch_samples_ns=[2980.372, 3271.534, 3109.958, 3110.443, 3123.623, 3115.873, 3125.549, 3017.875, 2976.968]
        reduced key     3110.4      -10.2      -0.3      73.37         -
recorded_batch_samples_ns=[2832.159, 2953.645, 3023.047, 2990.314, 3013.812, 3012.24, 3024.719, 2974.258, 2962.066]
         verify off     2990.3     -130.3      -4.2       1.71         -
recorded_batch_samples_ns=[2741.634, 2950.084, 2966.559, 2967.864, 2960.751, 2944.102, 2974.036, 2888.116, 2902.398]
          verify on     2950.1     -170.5      -5.5       1.64         -
recorded_batch_samples_ns=[2661.18, 2949.883, 2938.278, 2944.535, 2977.416, 2936.163, 2939.037, 2945.668, 2932.843]
              baked     2939.0     -181.6      -5.8       1.72         -
recorded_batch_samples_ns=[2736.281, 2961.275, 2980.353, 2990.069, 2994.413, 2966.097, 2973.591, 2969.616, 2974.171]
       bound tensor     2973.6     -147.0      -4.7       0.34      1.44
recorded_batch_samples_ns=[2744.677, 2928.24, 2960.553, 2946.923, 2969.454, 2958.175, 2946.937, 2945.674, 2965.474]
      bound pointer     2946.9     -173.7      -5.6       0.34      1.40
recorded_batch_samples_ns=[2777.122, 2890.913, 2947.068, 2926.433, 2917.275, 2915.781, 2940.172, 2915.182, 2919.875]
   fixed device map     2917.3     -203.3      -6.5       0.53      1.48
recorded_batch_samples_ns=[2750.835, 2937.991, 2936.411, 2914.815, 2924.243, 2922.531, 2925.433, 2935.382, 2935.609]
fixed device no-map     2925.4     -195.2      -6.3       0.29      1.46
elapsed_ns=3641811801
exit_status=0
ended=2026-09-25T21:35:13+08:00
```

### launch_host per-batch, round 0

```text
commit=38cdabc604b6c9a1a3ea57667e1f942fc5bc7960
started=2026-09-25T21:35:13+08:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj taskset -c 0 /tmp/gb2/bin/python /tmp/intj-launch-batch-recorder-38cdabc.py benchmarks/bench_launch.py --no-gpu --iters 100000 --batches 9
mode=host-only; 100000 calls × 9 batches; median ns/call
               path    ns/call       Δ ns       Δ %   build ms   bind ms
recorded_batch_samples_ns=[44.11447, 44.22994, 43.47985, 43.99199, 43.87788, 44.02345, 43.39862, 44.03421, 43.7562]
           auto map       44.0       +0.0      +0.0      68.62         -
recorded_batch_samples_ns=[43.1409, 46.02417, 43.1646, 43.62532, 43.50896, 43.5508, 43.10374, 43.6186, 43.33666]
        reduced key       43.5       -0.5      -1.1       1.53         -
recorded_batch_samples_ns=[43.16831, 43.72877, 43.9831, 43.85953, 43.12538, 43.85156, 43.25219, 43.74392, 43.21517]
         verify off       43.7       -0.3      -0.6       1.44         -
recorded_batch_samples_ns=[44.13769, 45.86893, 44.23601, 44.75805, 44.41309, 44.63993, 44.12694, 44.76227, 44.21957]
          verify on       44.4       +0.4      +1.0       1.37         -
recorded_batch_samples_ns=[39.86525, 40.35145, 39.88897, 40.41995, 39.89324, 40.2878, 39.74679, 40.45522, 39.81212]
              baked       39.9       -4.1      -9.3       1.97         -
recorded_batch_samples_ns=[45.42174, 45.98349, 45.42325, 47.51971, 46.34527, 46.34797, 45.85483, 45.96295, 45.46159]
       bound tensor       46.0       +2.0      +4.5       0.15      1.29
recorded_batch_samples_ns=[44.72433, 44.06387, 43.87921, 44.18541, 44.20894, 44.04396, 43.8757, 44.04664, 44.08402]
      bound pointer       44.1       +0.1      +0.2       0.13      1.24
recorded_batch_samples_ns=[43.82688, 44.53425, 43.7413, 48.59174, 43.77355, 44.66065, 43.9043, 44.41678, 43.88101]
   fixed device map       43.9       -0.1      -0.2       0.10      1.25
recorded_batch_samples_ns=[43.4221, 44.80861, 42.81585, 44.61026, 43.19342, 45.62121, 42.73667, 44.61506, 43.21126]
fixed device no-map       43.4       -0.6      -1.3       0.14      1.24
elapsed_ns=2895907560
exit_status=0
ended=2026-09-25T21:35:16+08:00
```

### launch_readme per-batch, round 0

```text
commit=38cdabc604b6c9a1a3ea57667e1f942fc5bc7960
started=2026-09-25T21:35:16+08:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj taskset -c 0 /tmp/gb2/bin/python /tmp/intj-launch-batch-recorder-38cdabc.py benchmarks/bench_launch.py --readme --iters 1000 --batches 9
recorded_batch_samples_ns=[92.169, 90.034, 90.325, 89.736, 90.438, 89.214, 88.783, 107.266, 89.005]
recorded_batch_samples_ns=[92.594, 95.567, 92.677, 94.837, 106.205, 91.38, 90.15, 89.34, 91.697]
recorded_batch_samples_ns=[874.625, 879.669, 868.023, 867.549, 861.707, 881.799, 864.234, 888.367, 839.391]
mode=readme; 1000 calls × 9 batches; median
        path   triton us    intj us   speedup
recorded_batch_samples_ns=[16840.738, 16793.821, 16549.115, 16613.835, 16565.987, 16660.817, 16498.769, 16607.338, 16482.146]
recorded_batch_samples_ns=[3020.539, 3244.731, 3081.148, 3084.058, 3079.543, 3067.887, 3086.008, 2956.602, 2937.237]
   grid=(1,)       16.61       3.08      5.4x
recorded_batch_samples_ns=[12888.95, 12827.902, 12869.173, 12869.239, 12896.234, 12885.781, 12820.795, 12840.133, 12901.548]
recorded_batch_samples_ns=[28.951, 28.965, 28.666, 28.765, 28.965, 28.844, 28.878, 28.815, 28.775]
   grid=(0,)       12.87       0.03    446.2x

torch_access_mode   decode ns    build s
     runtime_shim        90.0       0.02
   static_compile        92.6       0.08
      interpreter       868.0       0.00
elapsed_ns=3740707440
exit_status=0
ended=2026-09-25T21:35:19+08:00
```

### launch_sweep per-batch, round 0

```text
commit=38cdabc604b6c9a1a3ea57667e1f942fc5bc7960
started=2026-09-25T21:35:19+08:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj taskset -c 0 /tmp/gb2/bin/python /tmp/intj-launch-batch-recorder-38cdabc.py benchmarks/bench_launch.py --sweep --iters 100000 --batches 9
mode=sweep; host-only; 100000 calls × 9 batches; median ns/call
count   kind    ns/call
recorded_batch_samples_ns=[41.33599, 42.15085, 41.45321, 41.72574, 41.40418, 41.71073, 41.34473, 41.94041, 41.41477]
    4    int       41.5
recorded_batch_samples_ns=[40.64797, 40.85638, 40.50429, 40.85908, 40.42604, 41.07341, 40.49674, 41.59162, 40.51637]
    4 tensor       40.6
recorded_batch_samples_ns=[73.73279, 74.50122, 74.12031, 74.25302, 73.7055, 74.07582, 73.8071, 74.39848, 73.7109]
   16    int       74.1
recorded_batch_samples_ns=[70.6805, 69.12274, 67.82456, 70.06467, 68.52021, 68.80081, 68.28341, 68.56147, 67.82232]
   16 tensor       68.6
recorded_batch_samples_ns=[120.56434, 121.86892, 121.97865, 123.97842, 120.10073, 122.67637, 123.42984, 117.51344, 118.67124]
   32    int      121.9
recorded_batch_samples_ns=[123.55613, 119.35833, 117.02568, 116.29687, 117.32075, 115.97771, 132.65006, 138.31609, 127.61405]
   32 tensor      119.4
elapsed_ns=2971761817
exit_status=0
ended=2026-09-25T21:35:22+08:00
```

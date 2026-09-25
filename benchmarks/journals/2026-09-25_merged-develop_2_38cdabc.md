# Exact merged `develop` benchmark: round 2

Source `38cdabc604b6c9a1a3ea57667e1f942fc5bc7960`. Environment, commands, median-of-process medians and caveats: [round 0](2026-09-25_merged-develop_0_38cdabc.md). Raw outputs below retain every value printed by each benchmark.

## Raw outputs

### launch_gpu

```text
commit=38cdabc604b6c9a1a3ea57667e1f942fc5bc7960
started=2026-09-25T21:32:16+08:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --iters 1000 --batches 9
mode=gpu; 1000 calls × 9 batches; median ns/call
               path    ns/call       Δ ns       Δ %   build ms   bind ms
           auto map     3117.5       +0.0      +0.0      71.86         -
        reduced key     3109.1       -8.4      -0.3       3.39         -
         verify off     2997.8     -119.7      -3.8       1.72         -
          verify on     3039.7      -77.8      -2.5       1.65         -
              baked     2976.0     -141.5      -4.5      72.37         -
       bound tensor     3141.5      +24.0      +0.8       1.78     11.99
      bound pointer     2994.9     -122.6      -3.9       0.35      1.50
   fixed device map     2907.4     -210.1      -6.7       0.31      1.52
fixed device no-map     2908.2     -209.3      -6.7       0.29      1.50
elapsed_ns=4019874983
exit_status=0
ended=2026-09-25T21:32:20+08:00
```

### launch_host

```text
commit=38cdabc604b6c9a1a3ea57667e1f942fc5bc7960
started=2026-09-25T21:32:20+08:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --no-gpu --iters 100000 --batches 9
mode=host-only; 100000 calls × 9 batches; median ns/call
               path    ns/call       Δ ns       Δ %   build ms   bind ms
           auto map       43.7       +0.0      +0.0      59.73         -
        reduced key       43.3       -0.4      -0.9       1.55         -
         verify off       43.8       +0.2      +0.4       1.40         -
          verify on       44.7       +1.0      +2.3       1.36         -
              baked       39.8       -3.8      -8.8       2.75         -
       bound tensor       45.9       +2.3      +5.2       0.15      1.32
      bound pointer       45.3       +1.6      +3.7       0.14      1.24
   fixed device map       57.8      +14.2     +32.5       0.10      1.23
fixed device no-map       44.1       +0.4      +1.0       0.17      1.32
elapsed_ns=3063279403
exit_status=0
ended=2026-09-25T21:32:23+08:00
```

### launch_readme

```text
commit=38cdabc604b6c9a1a3ea57667e1f942fc5bc7960
started=2026-09-25T21:32:23+08:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --readme --iters 1000 --batches 9
mode=readme; 1000 calls × 9 batches; median
        path   triton us    intj us   speedup
   grid=(1,)       16.50       3.14      5.2x
   grid=(0,)       13.07       0.03    462.0x

torch_access_mode   decode ns    build s
     runtime_shim        89.9       0.02
   static_compile        93.8       0.06
      interpreter       939.6       0.00
elapsed_ns=3973447686
exit_status=0
ended=2026-09-25T21:32:27+08:00
```

### launch_sweep

```text
commit=38cdabc604b6c9a1a3ea57667e1f942fc5bc7960
started=2026-09-25T21:32:27+08:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --sweep --iters 100000 --batches 9
mode=sweep; host-only; 100000 calls × 9 batches; median ns/call
count   kind    ns/call
    4    int       42.4
    4 tensor       41.6
   16    int       74.2
   16 tensor       68.4
   32    int      121.1
   32 tensor      119.6
elapsed_ns=3097553897
exit_status=0
ended=2026-09-25T21:32:30+08:00
```

### ffi_default

```text
commit=38cdabc604b6c9a1a3ea57667e1f942fc5bc7960
started=2026-09-25T21:32:30+08:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_ffi_compare.py --iters 1000 --batches 9
iters=1000 batches=9 unit=us/call (host-only, no timed sync)
FFI packed nop             median=0.1448 samples=[0.155606, 0.14369800000000002, 0.148062, 0.1472, 0.144464, 0.144185, 0.144781, 0.145208, 0.14363399999999998]
FFI typed nop              median=0.1503 samples=[0.15247, 0.14774299999999999, 0.146734, 0.151537, 0.14849299999999999, 0.150568, 0.15026599999999998, 0.15201800000000001, 0.147617]
FFI empty kernel           median=3.3221 samples=[3.5221489999999998, 3.493965, 3.341274, 3.211042, 3.1636819999999997, 3.333446, 3.180728, 3.32206, 3.187373]
INTJ empty kernel          median=2.9935 samples=[3.088475, 3.0341709999999997, 3.175381, 2.9133739999999997, 2.9551529999999997, 2.949936, 2.9939400000000003, 2.94246, 2.993525]
INTJ fixed-device kernel   median=2.8794 samples=[3.060629, 2.987362, 3.072815, 2.9190880000000003, 2.879428, 2.8035279999999996, 2.804047, 2.812853, 2.8134319999999997]
FFI packed nop mixed       median=0.1733 samples=[0.171892, 0.172344, 0.172569, 0.173397, 0.176788, 0.176142, 0.173257, 0.172399, 0.17433500000000002]
FFI typed nop mixed        median=0.1779 samples=[0.175596, 0.17801599999999998, 0.17473, 0.18319, 0.181279, 0.178466, 0.176846, 0.177893, 0.17449299999999998]
FFI mixed kernel           median=3.2637 samples=[3.364054, 3.4238980000000003, 3.3488189999999998, 3.262168, 3.2333279999999998, 3.2637449999999997, 3.2387040000000002, 3.3424169999999997, 3.243343]
INTJ mixed kernel          median=2.9382 samples=[2.993033, 2.989522, 2.948395, 2.9382159999999997, 2.855331, 2.854745, 2.853323, 2.981549, 2.820808]
INTJ fixed mixed kernel    median=2.9254 samples=[3.017605, 3.019806, 2.970081, 3.18452, 2.8708020000000003, 2.856699, 2.884951, 2.925375, 2.875643]
elapsed_ns=3904862707
exit_status=0
ended=2026-09-25T21:32:34+08:00
```

### ffi_sweep

```text
commit=38cdabc604b6c9a1a3ea57667e1f942fc5bc7960
started=2026-09-25T21:32:34+08:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_ffi_compare.py --sweep --iters 1000 --batches 9
counts=[0, 3, 5, 8, 16, 32, 64] iters=1000 batches=9 unit=us/call (host-only, no timed sync); INTJ device-bound, no specialization key
args= 0 FFI packed nop           median=0.1172 samples=[0.12243000000000001, 0.11719, 0.116625, 0.118181, 0.116824, 0.11701600000000001, 0.116627, 0.11864799999999999, 0.11740099999999999]
args= 0 FFI typed nop            median=0.1212 samples=[0.121152, 0.124273, 0.118985, 0.12460299999999999, 0.11889, 0.12245199999999999, 0.119033, 0.121854, 0.11946999999999999]
args= 0 FFI empty kernel         median=1.6660 samples=[1.519523, 1.8500969999999999, 1.609406, 1.8458109999999999, 1.666049, 1.850653, 1.5800340000000002, 1.8555540000000001, 1.583484]
args= 0 INTJ static_compile kernel median=2.7098 samples=[3.1547739999999997, 2.6723290000000004, 2.718815, 2.7098079999999998, 2.764497, 2.675949, 2.6936329999999997, 2.777208, 2.7049659999999998]
args= 0 INTJ runtime_shim kernel median=2.7003 samples=[2.911187, 2.700344, 2.71848, 2.694942, 2.755947, 2.677719, 2.6986329999999996, 2.686068, 2.704692]
args= 3 FFI packed nop           median=0.1456 samples=[0.147814, 0.144183, 0.144226, 0.145601, 0.145595, 0.14578200000000002, 0.144725, 0.14699500000000001, 0.14586500000000002]
args= 3 FFI typed nop            median=0.1500 samples=[0.148568, 0.15065299999999998, 0.14743799999999999, 0.150044, 0.15262799999999999, 0.151733, 0.148437, 0.15096299999999999, 0.14821199999999998]
args= 3 FFI empty kernel         median=3.1864 samples=[3.346717, 3.202028, 3.163938, 3.201605, 3.1610650000000002, 3.186439, 3.147144, 3.210142, 3.158583]
args= 3 INTJ static_compile kernel median=2.9349 samples=[3.022907, 2.862938, 2.9349000000000003, 2.910963, 2.938956, 2.881728, 2.9370439999999998, 2.876674, 2.94382]
args= 3 INTJ runtime_shim kernel median=2.8391 samples=[3.001948, 2.850243, 2.786917, 2.839118, 2.7770010000000003, 2.827002, 2.779478, 2.88925, 2.875383]
args= 5 FFI packed nop           median=0.1762 samples=[0.17500100000000002, 0.178269, 0.175125, 0.17765899999999998, 0.177066, 0.17624299999999998, 0.17555099999999998, 0.176397, 0.17609200000000003]
args= 5 FFI typed nop            median=0.1824 samples=[0.182415, 0.179155, 0.175764, 0.197589, 0.179186, 0.182627, 0.182508, 0.18091, 0.183505]
args= 5 FFI empty kernel         median=3.2502 samples=[3.4058919999999997, 3.248869, 3.199207, 3.318255, 3.192764, 3.3174580000000002, 3.182673, 3.2614430000000003, 3.25019]
args= 5 INTJ static_compile kernel median=2.8517 samples=[3.015548, 2.8368, 2.8516500000000002, 2.782703, 2.845835, 2.788337, 2.8667710000000004, 2.856401, 2.8834180000000003]
args= 5 INTJ runtime_shim kernel median=2.8489 samples=[3.038772, 2.8778200000000003, 2.8385010000000004, 2.7889090000000003, 2.832305, 2.7739070000000003, 2.865669, 2.92865, 2.84886]
args= 8 FFI packed nop           median=0.2087 samples=[0.213485, 0.207784, 0.212094, 0.208654, 0.212708, 0.208267, 0.208407, 0.207196, 0.20880600000000002]
args= 8 FFI typed nop            median=0.2128 samples=[0.212262, 0.217321, 0.210452, 0.217113, 0.213358, 0.212817, 0.21259, 0.216231, 0.210447]
args= 8 FFI empty kernel         median=3.3683 samples=[3.5540079999999996, 3.437478, 3.271431, 3.3863879999999997, 3.2708760000000003, 3.379239, 3.2911149999999996, 3.3683229999999997, 3.301462]
args= 8 INTJ static_compile kernel median=3.0205 samples=[3.08852, 2.964429, 3.0517339999999997, 2.964382, 3.041953, 3.020516, 3.030433, 3.006482, 3.0142260000000003]
args= 8 INTJ runtime_shim kernel median=2.9198 samples=[3.0949340000000003, 2.919813, 2.9485230000000002, 2.903421, 2.943766, 2.8821529999999997, 2.95086, 2.890151, 2.874227]
args=16 FFI packed nop           median=0.2941 samples=[0.294149, 0.299405, 0.296511, 0.297388, 0.299107, 0.290902, 0.294012, 0.29291500000000004, 0.29292399999999996]
args=16 FFI typed nop            median=0.3063 samples=[0.302846, 0.310026, 0.30818799999999996, 0.307043, 0.306316, 0.30518, 0.30642, 0.30062900000000004, 0.304406]
args=16 FFI empty kernel         median=3.6316 samples=[3.7426280000000003, 3.719967, 3.5314029999999996, 3.631551, 3.5081239999999996, 3.654368, 3.520927, 3.6680590000000004, 3.541814]
args=16 INTJ static_compile kernel median=3.3491 samples=[3.288296, 3.349108, 3.4392240000000003, 3.2712350000000003, 3.4340189999999997, 3.299826, 3.495491, 3.269639, 3.428718]
args=16 INTJ runtime_shim kernel median=3.2741 samples=[3.286331, 3.200767, 3.349805, 3.053253, 3.313752, 3.031694, 3.327162, 3.0322489999999998, 3.2741149999999997]
args=32 FFI packed nop           median=0.4817 samples=[0.481708, 0.487272, 0.49394, 0.48335700000000004, 0.486846, 0.476766, 0.476464, 0.480062, 0.478474]
args=32 FFI typed nop            median=0.4991 samples=[0.494159, 0.50894, 0.500229, 0.503577, 0.49913799999999997, 0.498497, 0.499815, 0.491651, 0.498995]
args=32 FFI empty kernel         median=4.2668 samples=[4.356826, 4.266841, 4.151552, 4.317005, 4.163, 4.2751790000000005, 4.159988, 4.300466, 4.188278]
args=32 INTJ static_compile kernel median=3.6235 samples=[3.44833, 3.696065, 3.631867, 3.530014, 3.645908, 3.523467, 3.626036, 3.540137, 3.6234789999999997]
args=32 INTJ runtime_shim kernel median=3.4908 samples=[3.473913, 3.49079, 3.635566, 3.446501, 3.636876, 3.4338699999999998, 3.612374, 3.435803, 3.585873]
args=64 FFI packed nop           median=0.8414 samples=[0.82478, 0.83712, 0.840588, 0.862987, 0.84523, 0.8213239999999999, 0.865622, 0.856538, 0.8413619999999999]
args=64 FFI typed nop            median=0.8561 samples=[0.844429, 0.889178, 0.867688, 0.895086, 0.887149, 0.854005, 0.845587, 0.856087, 0.854494]
args=64 FFI empty kernel         median=5.0568 samples=[5.0966130000000005, 5.1091750000000005, 5.037649999999999, 5.058778, 5.065143, 5.056722, 5.036694000000001, 5.056761000000001, 5.040316]
args=64 INTJ static_compile kernel median=4.6059 samples=[4.697011000000001, 4.647066, 4.605932, 4.61031, 4.592364, 4.597727, 4.6010990000000005, 4.60037, 4.619871]
args=64 INTJ runtime_shim kernel median=4.5984 samples=[4.59838, 4.626397, 4.605837, 4.608876, 4.543871, 4.5905510000000005, 4.563394000000001, 4.615231, 4.583642]
elapsed_ns=4416805059
exit_status=0
ended=2026-09-25T21:32:38+08:00
```

### ffi_paths

```text
commit=38cdabc604b6c9a1a3ea57667e1f942fc5bc7960
started=2026-09-25T21:32:38+08:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_intj_ffi_paths.py --mode all --iters 1000 --batches 9
iters=1000 batches=9 unit=ns/call; no_gpu=True
mode=callback
callback_count=9100 (100 warmup + 9000 unique misses)
INTJ cold compile callback + cache  median_ns=427.7 samples_ns=[457.32, 447.235, 411.738, 497.992, 375.58, 407.189, 409.834, 427.684, 1157.14]
INTJ hot call (no callback)         median_ns=39.2 samples_ns=[40.981, 41.291, 39.224, 40.126, 38.264, 39.633, 37.818, 39.022, 37.88]
INTJ 3-tensor host-only nop         median_ns=35.0 samples_ns=[36.834, 35.602, 34.789, 35.034, 34.821, 35.023, 34.611, 35.022, 34.623]
mode=kwargs
INTJ direct positional              median_ns=67.5 samples_ns=[68.403, 66.053, 71.735, 65.877, 65.83, 65.808, 67.541, 67.494, 67.481]
INTJ adapter positional             median_ns=91.8 samples_ns=[92.315, 91.87, 91.184, 90.953, 92.159, 95.148, 90.918, 91.157, 91.78]
INTJ adapter kwargs                 median_ns=106.1 samples_ns=[108.293, 106.118, 105.531, 105.122, 104.59, 105.8, 123.683, 111.917, 111.382]
INTJ adapter defaults               median_ns=89.3 samples_ns=[90.225, 89.123, 90.939, 89.182, 89.164, 89.348, 89.271, 93.955, 89.766]
INTJ FFI wrapper positional         median_ns=104.0 samples_ns=[105.283, 104.652, 103.005, 106.341, 102.896, 103.886, 103.255, 112.053, 104.024]
INTJ FFI wrapper kwargs             median_ns=126.0 samples_ns=[125.504, 125.759, 125.979, 125.724, 125.597, 126.3, 133.978, 130.67, 130.075]
INTJ native kwargs: unsupported (verified TypeError)
mode=dataclass
INTJ pair direct                    median_ns=68.1 samples_ns=[69.233, 68.388, 67.034, 68.148, 66.759, 68.95, 66.813, 68.943, 67.807]
INTJ pair prebuilt *tuple           median_ns=140.4 samples_ns=[141.172, 147.333, 140.443, 140.137, 137.937, 139.87, 137.534, 143.379, 147.164]
FFI unpack Pair only                median_ns=163.4 samples_ns=[163.435, 163.672, 161.834, 164.248, 164.313, 169.101, 162.303, 163.434, 162.442]
INTJ pair manual unpack             median_ns=172.6 samples_ns=[170.077, 171.943, 173.093, 172.796, 171.864, 172.589, 169.709, 174.831, 174.014]
INTJ pair FFI unpack                median_ns=316.4 samples_ns=[314.596, 317.353, 320.255, 316.44, 315.176, 320.491, 312.748, 315.655, 318.19]
INTJ pair stdlib astuple            median_ns=1154.6 samples_ns=[1334.868, 1192.988, 1155.653, 1154.939, 1139.226, 1154.63, 1144.55, 1149.265, 1144.077]
INTJ config direct                  median_ns=82.3 samples_ns=[83.656, 82.961, 85.915, 82.906, 81.093, 82.261, 81.112, 81.809, 82.219]
INTJ config FFI unpack              median_ns=320.7 samples_ns=[319.1, 324.214, 319.123, 321.542, 329.628, 320.655, 320.552, 331.206, 319.205]
INTJ native dataclass unpack: unsupported (verified TypeError)
elapsed_ns=3001243734
exit_status=0
ended=2026-09-25T21:32:41+08:00
```

### hip_same_hsaco

```text
commit=38cdabc604b6c9a1a3ea57667e1f942fc5bc7960
started=2026-09-25T21:32:41+08:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_hip_module_launch.py --iters 1000 --batches 9
hsaco_sha256=df3e529e178196ad62f7c07ab8593ede5b8596c76e7c6fbd12a5d947a696e65c kernel=empty_kernel threads=256 shared=0 callback_count=1
iters=1000 batches=9 unit=us/call (host enqueue, no timed sync)
TVM FFI HIP HSACO         median=3.1523 samples=[3.4559729999999997, 3.223229, 3.2062, 3.181382, 3.152302, 3.062386, 3.0664000000000002, 3.0407100000000002, 3.0380920000000002]
Triton same HSACO         median=15.7530 samples=[15.919745, 15.940702, 15.868113, 15.885860000000001, 15.753012, 15.739259, 15.723892, 15.745918, 15.722949]
INTJ same function        median=2.9165 samples=[3.0693710000000003, 3.069339, 3.006074, 3.009303, 2.916452, 2.852334, 2.8571950000000004, 2.870384, 2.824574]
elapsed_ns=4023351051
exit_status=0
ended=2026-09-25T21:32:45+08:00
```

## Launcher per-batch diagnostics (supplemental)

The external driver and timing caveat are in [round 0](2026-09-25_merged-develop_0_38cdabc.md).

### launch_gpu per-batch, round 2

```text
commit=38cdabc604b6c9a1a3ea57667e1f942fc5bc7960
started=2026-09-25T21:35:36+08:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj taskset -c 0 /tmp/gb2/bin/python /tmp/intj-launch-batch-recorder-38cdabc.py benchmarks/bench_launch.py --iters 1000 --batches 9
mode=gpu; 1000 calls × 9 batches; median ns/call
               path    ns/call       Δ ns       Δ %   build ms   bind ms
recorded_batch_samples_ns=[3141.291, 3455.304, 3141.684, 3131.453, 3143.788, 3131.393, 3111.549, 3114.44, 3100.763]
           auto map     3131.5       +0.0      +0.0      93.28         -
recorded_batch_samples_ns=[2928.498, 3329.141, 3114.883, 3142.025, 3134.035, 3120.134, 3107.146, 3018.836, 2992.724]
        reduced key     3114.9      -16.6      -0.5      71.84         -
recorded_batch_samples_ns=[2794.761, 3027.314, 2980.249, 3008.459, 3020.807, 3004.667, 3018.924, 2976.334, 2950.778]
         verify off     3004.7     -126.8      -4.0       1.67         -
recorded_batch_samples_ns=[2779.211, 2980.459, 2979.685, 2944.744, 2966.222, 2933.07, 2935.412, 2913.097, 2919.387]
          verify on     2935.4     -196.0      -6.3       1.63         -
recorded_batch_samples_ns=[2724.813, 2964.621, 2954.484, 2971.547, 2983.298, 2930.323, 2929.226, 2917.369, 2921.447]
              baked     2930.3     -201.1      -6.4       1.73         -
recorded_batch_samples_ns=[2760.481, 3006.035, 3001.318, 2991.749, 3021.31, 2980.075, 2988.56, 2984.972, 2985.364]
       bound tensor     2988.6     -142.9      -4.6       0.28      1.46
recorded_batch_samples_ns=[2755.879, 2955.863, 2961.04, 2959.085, 2962.13, 2942.714, 2942.501, 2948.264, 2961.524]
      bound pointer     2955.9     -175.6      -5.6       0.36      1.49
recorded_batch_samples_ns=[2809.05, 2901.944, 2900.535, 2918.112, 2906.487, 2914.11, 2881.176, 2903.898, 2921.858]
   fixed device map     2903.9     -227.6      -7.3       0.41      1.54
recorded_batch_samples_ns=[2816.53, 2873.452, 2918.594, 2907.803, 2894.141, 2901.361, 2899.578, 2903.364, 2894.29]
fixed device no-map     2899.6     -231.9      -7.4       0.27      1.52
elapsed_ns=3661088683
exit_status=0
ended=2026-09-25T21:35:39+08:00
```

### launch_host per-batch, round 2

```text
commit=38cdabc604b6c9a1a3ea57667e1f942fc5bc7960
started=2026-09-25T21:35:39+08:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj taskset -c 0 /tmp/gb2/bin/python /tmp/intj-launch-batch-recorder-38cdabc.py benchmarks/bench_launch.py --no-gpu --iters 100000 --batches 9
mode=host-only; 100000 calls × 9 batches; median ns/call
               path    ns/call       Δ ns       Δ %   build ms   bind ms
recorded_batch_samples_ns=[44.00438, 44.12638, 43.53083, 44.01048, 43.70633, 43.9392, 43.44448, 43.98184, 43.71496]
           auto map       43.9       +0.0      +0.0      67.42         -
recorded_batch_samples_ns=[43.19408, 43.56034, 43.52801, 43.60595, 43.06221, 43.55931, 43.08528, 43.54486, 43.07768]
        reduced key       43.5       -0.4      -0.9       1.52         -
recorded_batch_samples_ns=[43.33597, 43.88932, 43.45135, 44.06986, 44.04492, 43.98033, 43.78378, 44.03647, 43.56116]
         verify off       43.9       -0.0      -0.1       1.41         -
recorded_batch_samples_ns=[44.12996, 44.98569, 44.12738, 44.96593, 44.18261, 44.53652, 44.41766, 44.89729, 44.72782]
          verify on       44.5       +0.6      +1.4       1.41         -
recorded_batch_samples_ns=[39.7965, 40.21558, 39.79087, 40.35278, 39.73888, 40.4967, 39.91788, 40.3539, 39.96241]
              baked       40.0       -4.0      -9.1       1.93         -
recorded_batch_samples_ns=[46.11688, 47.8499, 45.60887, 46.36277, 45.66821, 45.96699, 45.40968, 45.82713, 45.97635]
       bound tensor       46.0       +2.0      +4.6       0.15      1.29
recorded_batch_samples_ns=[44.50828, 44.15414, 44.7736, 44.11225, 44.056, 43.92029, 44.07534, 44.02254, 44.2473]
      bound pointer       44.1       +0.2      +0.4       0.13      1.27
recorded_batch_samples_ns=[43.86744, 44.37022, 44.03082, 44.6393, 44.06308, 44.44305, 43.93795, 44.41722, 43.85054]
   fixed device map       44.1       +0.1      +0.3       0.10      1.24
recorded_batch_samples_ns=[43.69339, 45.27124, 42.85766, 45.07268, 43.07651, 44.72568, 42.74312, 44.29901, 42.67849]
fixed device no-map       43.7       -0.2      -0.6       0.14      1.25
elapsed_ns=2876553320
exit_status=0
ended=2026-09-25T21:35:42+08:00
```

### launch_readme per-batch, round 2

```text
commit=38cdabc604b6c9a1a3ea57667e1f942fc5bc7960
started=2026-09-25T21:35:42+08:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj taskset -c 0 /tmp/gb2/bin/python /tmp/intj-launch-batch-recorder-38cdabc.py benchmarks/bench_launch.py --readme --iters 1000 --batches 9
recorded_batch_samples_ns=[97.74, 91.981, 93.56, 91.082, 89.664, 87.157, 88.377, 88.584, 89.911]
recorded_batch_samples_ns=[93.016, 94.626, 97.594, 90.922, 90.485, 93.165, 91.883, 92.008, 90.26]
recorded_batch_samples_ns=[938.98, 939.594, 937.212, 940.488, 938.16, 948.071, 933.194, 957.843, 936.283]
mode=readme; 1000 calls × 9 batches; median
        path   triton us    intj us   speedup
recorded_batch_samples_ns=[16908.533, 16758.816, 16611.447, 16660.602, 16624.085, 16833.577, 16605.742, 16691.247, 16586.937]
recorded_batch_samples_ns=[3034.915, 3256.402, 3111.799, 3092.438, 3100.12, 3064.912, 3098.348, 2976.338, 2959.968]
   grid=(1,)       16.66       3.09      5.4x
recorded_batch_samples_ns=[13089.425, 13051.508, 13075.876, 13074.258, 13098.948, 13078.24, 13080.045, 13053.092, 13072.131]
recorded_batch_samples_ns=[28.98, 29.105, 28.991, 28.438, 28.529, 29.221, 28.572, 28.982, 29.622]
   grid=(0,)       13.08       0.03    451.2x

torch_access_mode   decode ns    build s
     runtime_shim        89.9       0.02
   static_compile        92.0       0.08
      interpreter       939.0       0.00
elapsed_ns=3714221267
exit_status=0
ended=2026-09-25T21:35:46+08:00
```

### launch_sweep per-batch, round 2

```text
commit=38cdabc604b6c9a1a3ea57667e1f942fc5bc7960
started=2026-09-25T21:35:46+08:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj taskset -c 0 /tmp/gb2/bin/python /tmp/intj-launch-batch-recorder-38cdabc.py benchmarks/bench_launch.py --sweep --iters 100000 --batches 9
mode=sweep; host-only; 100000 calls × 9 batches; median ns/call
count   kind    ns/call
recorded_batch_samples_ns=[41.21473, 42.25793, 42.06956, 42.0119, 41.5829, 41.9769, 41.38089, 41.82688, 41.44899]
    4    int       41.8
recorded_batch_samples_ns=[40.30078, 41.04406, 40.51895, 41.19324, 40.32805, 40.92013, 40.33306, 40.87009, 40.44566]
    4 tensor       40.5
recorded_batch_samples_ns=[73.85501, 74.04292, 73.72176, 74.07746, 73.91506, 74.27192, 73.68827, 74.44238, 73.85115]
   16    int       73.9
recorded_batch_samples_ns=[68.41152, 68.71149, 67.79919, 68.42094, 67.89512, 69.32991, 67.89465, 73.42338, 67.72278]
   16 tensor       68.4
recorded_batch_samples_ns=[121.04803, 123.08145, 120.47586, 116.523, 117.96718, 118.1815, 119.45823, 117.91343, 117.1752]
   32    int      118.2
recorded_batch_samples_ns=[121.68189, 116.1253, 113.51729, 113.73472, 115.82293, 115.14435, 114.95372, 114.83475, 114.46374]
   32 tensor      115.0
elapsed_ns=2940337705
exit_status=0
ended=2026-09-25T21:35:49+08:00
```

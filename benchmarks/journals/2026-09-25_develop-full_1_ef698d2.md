# Complete `develop` Python benchmark run: round 1

Source `ef698d2`; setup, commands, and aggregate results are in [round 0](2026-09-25_develop-full_0_ef698d2.md). Each block includes commit, exact command, all reported per-batch samples, duration, and exit status.

## Raw outputs

### launch_gpu

```text
commit=ef698d24dae8790fffdf84fe766a7bf06db428f6
started=2026-09-25T21:01:04+08:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --iters 1000 --batches 9
mode=gpu; 1000 calls × 9 batches; median ns/call
               path    ns/call       Δ ns       Δ %   build ms   bind ms
           auto map     3729.2       +0.0      +0.0      88.43         -
        reduced key     3699.6      -29.6      -0.8       2.13         -
         verify off     3594.3     -134.9      -3.6       1.85         -
          verify on     3573.7     -155.4      -4.2       1.75         -
              baked     3527.3     -201.9      -5.4       1.81         -
       bound tensor     3798.0      +68.8      +1.8       0.42      1.49
      bound pointer     3641.7      -87.5      -2.3       0.38      1.53
   fixed device map     3571.0     -158.2      -4.2       0.35      1.63
fixed device no-map     3719.9       -9.3      -0.2       0.41      1.45
elapsed_ns=3892142829
exit_status=0
ended=2026-09-25T21:01:08+08:00
```

### launch_host

```text
commit=ef698d24dae8790fffdf84fe766a7bf06db428f6
started=2026-09-25T21:01:08+08:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --no-gpu --iters 100000 --batches 9
mode=host-only; 100000 calls × 9 batches; median ns/call
               path    ns/call       Δ ns       Δ %   build ms   bind ms
           auto map       43.5       +0.0      +0.0      59.80         -
        reduced key       43.1       -0.4      -0.9       1.32         -
         verify off       43.5       -0.0      -0.0       1.23         -
          verify on       44.5       +1.0      +2.3       1.15         -
              baked       39.9       -3.6      -8.3       1.31         -
       bound tensor       45.4       +1.9      +4.4       0.13      1.09
      bound pointer       44.5       +1.0      +2.3       0.12      1.10
   fixed device map       44.1       +0.6      +1.5       0.09      1.07
fixed device no-map       44.8       +1.3      +3.1       0.12      1.08
elapsed_ns=2999438258
exit_status=0
ended=2026-09-25T21:01:11+08:00
```

### launch_readme

```text
commit=ef698d24dae8790fffdf84fe766a7bf06db428f6
started=2026-09-25T21:01:11+08:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --readme --iters 1000 --batches 9
mode=readme; 1000 calls × 9 batches; median
        path   triton us    intj us   speedup
   grid=(1,)       16.80       3.13      5.4x
   grid=(0,)       13.17       0.03    468.4x

torch_access   decode ns    build s
        shim        93.0       0.02
         cxx        91.7       0.06
     cpython       879.6       0.00
elapsed_ns=3769357298
exit_status=0
ended=2026-09-25T21:01:15+08:00
```

### launch_sweep

```text
commit=ef698d24dae8790fffdf84fe766a7bf06db428f6
started=2026-09-25T21:01:15+08:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --sweep --iters 100000 --batches 9
mode=sweep; host-only; 100000 calls × 9 batches; median ns/call
count   kind    ns/call
    4    int       41.5
    4 tensor       40.6
   16    int       74.1
   16 tensor       68.1
   32    int      118.1
   32 tensor      118.3
elapsed_ns=3072637673
exit_status=0
ended=2026-09-25T21:01:18+08:00
```

### ffi_default

```text
commit=ef698d24dae8790fffdf84fe766a7bf06db428f6
started=2026-09-25T21:01:18+08:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_ffi_compare.py --iters 1000 --batches 9
iters=1000 batches=9 unit=us/call (host-only, no timed sync)
FFI packed nop             median=0.1444 samples=[0.152415, 0.14582599999999998, 0.144421, 0.14352600000000001, 0.14502600000000002, 0.143781, 0.144161, 0.143396, 0.146603]
FFI typed nop              median=0.1511 samples=[0.15684, 0.151071, 0.147357, 0.14998, 0.15198699999999998, 0.151812, 0.14843299999999998, 0.15112299999999998, 0.149992]
FFI empty kernel           median=3.3529 samples=[3.5193119999999998, 3.418467, 3.390422, 3.339268, 3.1655450000000003, 3.358172, 3.232987, 3.3529400000000003, 3.188311]
INTJ empty kernel          median=3.0233 samples=[3.242921, 3.091707, 3.1089960000000003, 2.90894, 2.987395, 2.980856, 3.023269, 2.906603, 3.031507]
INTJ fixed-device kernel   median=2.9078 samples=[2.990872, 3.101334, 3.100545, 2.8458989999999997, 2.7921869999999998, 2.8316260000000004, 2.923085, 2.821256, 2.907802]
FFI packed nop mixed       median=0.1758 samples=[0.17763900000000002, 0.180201, 0.173213, 0.17419, 0.175358, 0.172619, 0.17580500000000002, 0.177924, 0.176785]
FFI typed nop mixed        median=0.1795 samples=[0.17952500000000002, 0.179082, 0.179781, 0.18123, 0.175636, 0.176491, 0.179478, 0.180835, 0.179052]
FFI mixed kernel           median=3.3888 samples=[3.409829, 3.461917, 3.3983659999999998, 3.314197, 3.235293, 3.439762, 3.288965, 3.388754, 3.2962469999999997]
INTJ mixed kernel          median=2.9404 samples=[3.1478159999999997, 3.299239, 2.9427109999999996, 2.9403960000000002, 2.784679, 2.833237, 2.829627, 2.9587339999999998, 2.832172]
INTJ fixed mixed kernel    median=2.9023 samples=[3.1439920000000003, 3.13451, 2.989224, 3.0063969999999998, 2.857738, 2.8265819999999997, 2.84261, 2.902251, 2.900687]
elapsed_ns=3787288133
exit_status=0
ended=2026-09-25T21:01:22+08:00
```

### ffi_sweep

```text
commit=ef698d24dae8790fffdf84fe766a7bf06db428f6
started=2026-09-25T21:01:22+08:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_ffi_compare.py --sweep --iters 1000 --batches 9
counts=[0, 3, 5, 8, 16, 32, 64] iters=1000 batches=9 unit=us/call (host-only, no timed sync); INTJ device-bound, no specialization key
args= 0 FFI packed nop           median=0.1164 samples=[0.119796, 0.116352, 0.11543099999999999, 0.116497, 0.116241, 0.122247, 0.114565, 0.11539, 0.116971]
args= 0 FFI typed nop            median=0.1194 samples=[0.120816, 0.124264, 0.11745900000000001, 0.122817, 0.11679600000000001, 0.12082899999999999, 0.117362, 0.119422, 0.11804]
args= 0 FFI empty kernel         median=1.6957 samples=[1.563171, 1.919873, 1.6956600000000002, 1.854722, 1.61608, 1.8961949999999999, 1.59741, 1.88447, 1.664825]
args= 0 INTJ cxx kernel          median=2.7144 samples=[3.2582199999999997, 2.697228, 2.77582, 2.714692, 2.710343, 2.648159, 2.714379, 2.6706469999999998, 2.767444]
args= 0 INTJ shim kernel         median=2.7018 samples=[2.9076329999999997, 2.6949490000000003, 2.796504, 2.686159, 2.707119, 2.683266, 2.701825, 2.675243, 2.760375]
args= 3 FFI packed nop           median=0.1455 samples=[0.14502500000000002, 0.14405199999999999, 0.14682699999999999, 0.145017, 0.148405, 0.145531, 0.145681, 0.145308, 0.147278]
args= 3 FFI typed nop            median=0.1498 samples=[0.149642, 0.150708, 0.149804, 0.150308, 0.148887, 0.153862, 0.148124, 0.151149, 0.147695]
args= 3 FFI empty kernel         median=3.2334 samples=[3.451041, 3.246494, 3.155227, 3.239784, 3.158827, 3.2333629999999998, 3.132032, 3.284427, 3.151466]
args= 3 INTJ cxx kernel          median=2.9644 samples=[3.016451, 2.896948, 2.965717, 2.875959, 2.964395, 2.872495, 2.9658870000000004, 2.844768, 2.9777460000000002]
args= 3 INTJ shim kernel         median=2.9156 samples=[3.039872, 2.9270859999999996, 2.925724, 2.8959789999999996, 2.918698, 2.897606, 2.91556, 2.8102240000000003, 2.907655]
args= 5 FFI packed nop           median=0.1741 samples=[0.181952, 0.1739, 0.17489, 0.174064, 0.175786, 0.174817, 0.173319, 0.172641, 0.173672]
args= 5 FFI typed nop            median=0.1784 samples=[0.173738, 0.18853999999999999, 0.175183, 0.175009, 0.180655, 0.178429, 0.175726, 0.18344300000000002, 0.17915199999999998]
args= 5 FFI empty kernel         median=3.2728 samples=[3.391255, 3.2551, 3.27461, 3.2991729999999997, 3.272828, 3.330707, 3.253934, 3.2483760000000004, 3.257927]
args= 5 INTJ cxx kernel          median=2.8823 samples=[3.011328, 2.936051, 2.899415, 2.8508090000000004, 2.880806, 2.8494319999999997, 2.892058, 2.8808290000000003, 2.882275]
args= 5 INTJ shim kernel         median=2.8748 samples=[3.0189090000000003, 2.9393569999999998, 2.87482, 2.776325, 2.878136, 2.772053, 2.872789, 2.900756, 2.87043]
args= 8 FFI packed nop           median=0.2066 samples=[0.208791, 0.205174, 0.206875, 0.207061, 0.206568, 0.20569800000000002, 0.207096, 0.206104, 0.206311]
args= 8 FFI typed nop            median=0.2093 samples=[0.21001499999999998, 0.208784, 0.209572, 0.212815, 0.20797, 0.209278, 0.209081, 0.213196, 0.208398]
args= 8 FFI empty kernel         median=3.4319 samples=[3.535183, 3.45034, 3.575518, 3.431877, 3.332052, 3.4141239999999997, 3.567707, 3.403383, 3.324912]
args= 8 INTJ cxx kernel          median=3.0446 samples=[3.066683, 2.9796709999999997, 3.0687260000000003, 3.0213200000000002, 3.046463, 2.961557, 3.044593, 3.0089319999999997, 3.079975]
args= 8 INTJ shim kernel         median=2.8896 samples=[3.068546, 2.941036, 2.864531, 2.9048760000000002, 2.867645, 2.8927310000000004, 2.8590549999999997, 2.889597, 2.863962]
args=16 FFI packed nop           median=0.2936 samples=[0.295757, 0.290263, 0.296136, 0.296919, 0.294356, 0.288793, 0.291574, 0.289831, 0.29362299999999997]
args=16 FFI typed nop            median=0.3053 samples=[0.306993, 0.30453600000000003, 0.303073, 0.303678, 0.305307, 0.30209600000000003, 0.31061500000000003, 0.31118, 0.311089]
args=16 FFI empty kernel         median=3.6276 samples=[3.7261309999999996, 3.731357, 3.681487, 3.624034, 3.52502, 3.6275720000000002, 3.51886, 3.629007, 3.521482]
args=16 INTJ cxx kernel          median=3.3029 samples=[3.28009, 3.372131, 3.333215, 3.2661100000000003, 3.428056, 3.2926889999999998, 3.317501, 3.3028899999999997, 3.300745]
args=16 INTJ shim kernel         median=3.2544 samples=[3.2687399999999998, 3.173495, 3.284408, 3.046611, 3.3024769999999997, 3.027564, 3.254355, 3.0478400000000003, 3.280354]
args=32 FFI packed nop           median=0.4761 samples=[0.484089, 0.474171, 0.476118, 0.474622, 0.480802, 0.470208, 0.479481, 0.472233, 0.479097]
args=32 FFI typed nop            median=0.4943 samples=[0.48832, 0.492513, 0.49189299999999997, 0.496888, 0.49874599999999997, 0.500885, 0.49197, 0.496739, 0.494336]
args=32 FFI empty kernel         median=4.2177 samples=[4.302446, 4.265501, 4.290321, 4.229256, 4.097218, 4.215108, 4.138832000000001, 4.217656, 4.138332]
args=32 INTJ cxx kernel          median=3.6358 samples=[3.4483099999999998, 3.635783, 3.8150120000000003, 3.573362, 3.665855, 3.543113, 3.779517, 3.574509, 3.774942]
args=32 INTJ shim kernel         median=3.5675 samples=[9.890722, 3.469161, 3.6204549999999998, 3.430589, 3.5674960000000002, 3.410662, 3.627342, 3.426291, 3.6037179999999998]
args=64 FFI packed nop           median=0.8392 samples=[1.741037, 0.875834, 0.837288, 0.839197, 0.822998, 0.8320660000000001, 0.847658, 0.837077, 0.841035]
args=64 FFI typed nop            median=0.8657 samples=[1.463994, 0.901044, 0.855808, 0.861837, 0.8560030000000001, 0.86038, 0.873383, 0.866385, 0.865722]
args=64 FFI empty kernel         median=5.0447 samples=[16.989496, 5.084555, 5.219402, 5.044734, 5.050109, 5.040832999999999, 5.031575, 5.027715, 5.032983]
args=64 INTJ cxx kernel          median=4.5995 samples=[4.631203999999999, 4.6239740000000005, 4.645238999999999, 4.585793, 4.574616, 4.59952, 4.592639, 4.5776069999999995, 14.510799]
args=64 INTJ shim kernel         median=4.5964 samples=[4.614959, 4.6358760000000006, 4.556541, 4.563683, 4.495398, 4.5813999999999995, 4.596408, 4.616866, 19.045153]
elapsed_ns=4330966495
exit_status=0
ended=2026-09-25T21:01:26+08:00
```

### ffi_paths

```text
commit=ef698d24dae8790fffdf84fe766a7bf06db428f6
started=2026-09-25T21:01:26+08:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_intj_ffi_paths.py --mode all --iters 1000 --batches 9
iters=1000 batches=9 unit=ns/call; no_gpu=True
mode=callback
callback_count=9100 (100 warmup + 9000 unique misses)
INTJ cold compile callback + cache  median_ns=419.7 samples_ns=[458.348, 472.014, 405.626, 569.908, 379.431, 399.314, 413.891, 419.683, 1473.455]
INTJ hot call (no callback)         median_ns=38.1 samples_ns=[42.069, 39.81, 38.088, 38.457, 37.85, 38.091, 37.719, 38.418, 37.564]
INTJ 3-tensor host-only nop         median_ns=35.1 samples_ns=[36.339, 35.237, 34.519, 35.037, 34.456, 35.059, 39.612, 35.632, 34.741]
mode=kwargs
INTJ direct positional              median_ns=65.3 samples_ns=[67.966, 65.589, 65.341, 65.201, 65.466, 65.24, 65.278, 65.342, 65.36]
INTJ adapter positional             median_ns=90.8 samples_ns=[95.826, 90.075, 89.597, 89.113, 92.348, 90.653, 90.784, 91.081, 90.864]
INTJ adapter kwargs                 median_ns=106.9 samples_ns=[108.938, 109.588, 110.521, 106.828, 106.991, 106.01, 105.713, 106.9, 106.93]
INTJ adapter defaults               median_ns=92.0 samples_ns=[89.463, 89.091, 108.501, 91.412, 91.871, 91.984, 92.107, 92.321, 92.452]
INTJ FFI wrapper positional         median_ns=105.5 samples_ns=[104.773, 103.768, 103.809, 125.732, 106.082, 105.665, 105.606, 105.536, 104.787]
INTJ FFI wrapper kwargs             median_ns=125.8 samples_ns=[128.819, 127.487, 128.072, 131.42, 124.93, 125.421, 124.342, 125.735, 125.839]
INTJ native kwargs: unsupported (verified TypeError)
mode=dataclass
INTJ pair direct                    median_ns=70.3 samples_ns=[69.967, 72.042, 67.714, 71.749, 67.325, 71.813, 67.406, 71.719, 70.257]
INTJ pair prebuilt *tuple           median_ns=145.3 samples_ns=[145.971, 145.999, 142.487, 145.531, 142.477, 145.272, 141.706, 146.308, 141.096]
FFI unpack Pair only                median_ns=163.5 samples_ns=[164.945, 165.631, 163.529, 164.679, 169.622, 163.348, 162.972, 162.505, 163.056]
INTJ pair manual unpack             median_ns=174.6 samples_ns=[174.208, 183.264, 176.989, 173.671, 172.095, 174.942, 181.676, 174.568, 170.85]
INTJ pair FFI unpack                median_ns=316.6 samples_ns=[315.852, 319.674, 313.805, 314.187, 313.19, 326.806, 316.575, 319.632, 320.742]
INTJ pair stdlib astuple            median_ns=1153.6 samples_ns=[1306.835, 1202.553, 1145.68, 1170.41, 1137.436, 1153.635, 1151.88, 1164.28, 1148.394]
INTJ config direct                  median_ns=84.4 samples_ns=[86.536, 84.722, 82.641, 85.031, 82.433, 84.65, 82.303, 84.411, 82.262]
INTJ config FFI unpack              median_ns=327.2 samples_ns=[332.244, 327.186, 323.291, 335.926, 325.264, 328.006, 329.197, 326.958, 322.429]
INTJ native dataclass unpack: unsupported (verified TypeError)
elapsed_ns=2891959613
exit_status=0
ended=2026-09-25T21:01:29+08:00
```

### hip_same_hsaco

```text
commit=ef698d24dae8790fffdf84fe766a7bf06db428f6
started=2026-09-25T21:01:29+08:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_hip_module_launch.py --iters 1000 --batches 9
hsaco_sha256=df3e529e178196ad62f7c07ab8593ede5b8596c76e7c6fbd12a5d947a696e65c kernel=empty_kernel threads=256 shared=0 callback_count=1
iters=1000 batches=9 unit=us/call (host enqueue, no timed sync)
TVM FFI HIP HSACO         median=3.1270 samples=[3.58404, 3.202194, 3.224415, 3.190984, 3.126975, 3.07188, 3.0786239999999996, 3.035165, 3.1166039999999997]
Triton same HSACO         median=16.2116 samples=[16.364209000000002, 16.211561, 16.164904, 16.148348, 16.099615, 16.434668000000002, 16.437402, 16.400269, 16.11881]
INTJ same function        median=2.9036 samples=[3.019295, 2.988168, 2.984973, 3.0072170000000003, 2.903644, 2.871262, 2.858858, 2.8639050000000004, 2.808074]
elapsed_ns=3794746385
exit_status=0
ended=2026-09-25T21:01:33+08:00
```

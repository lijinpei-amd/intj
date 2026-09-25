# Latest develop remerge: measurement round 1

Baseline `ef698d2`; intermediate merged `2a9f891`, final merged `03a8f52`. Same environment, commands, and caveats as [round 0](2026-09-25_develop-remerge_0_03a8f52.md).

## Raw outputs (including batch samples and run metadata)

### host: develop

```text
commit=ef698d24dae8790fffdf84fe766a7bf06db428f6
started=2026-09-25T16:24:04+08:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --no-gpu --iters 100000 --batches 9
mode=host-only; 100000 calls × 9 batches; median ns/call
               path    ns/call       Δ ns       Δ %   build ms   bind ms
           auto map       44.4       +0.0      +0.0      59.34         -
        reduced key       43.4       -1.0      -2.3       1.33         -
         verify off       43.5       -0.9      -2.1       1.19         -
          verify on       44.8       +0.3      +0.7       1.13         -
              baked       39.9       -4.5     -10.1       1.32         -
       bound tensor       45.4       +1.0      +2.2       0.13      1.09
      bound pointer       44.2       -0.2      -0.4       0.12      1.04
   fixed device map       45.7       +1.2      +2.8       0.10      1.01
fixed device no-map       44.5       +0.1      +0.2       0.12      0.99
elapsed_ns=3034521349
exit_status=0
ended=2026-09-25T16:24:07+08:00
```

### host: merged

```text
commit=2a9f891443cf6763d26483c7b1f3d0f451d90a3c
started=2026-09-25T16:26:45+08:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/torch_abi
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/torch_abi taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --no-gpu --iters 100000 --batches 9
mode=host-only; 100000 calls × 9 batches; median ns/call
               path    ns/call       Δ ns       Δ %   build ms   bind ms
           auto map       43.8       +0.0      +0.0      59.45         -
        reduced key       43.2       -0.6      -1.3       1.53         -
         verify off       43.3       -0.5      -1.1       1.39         -
          verify on       44.8       +1.0      +2.2       1.35         -
              baked       40.0       -3.8      -8.6       2.56         -
       bound tensor       45.4       +1.6      +3.7       0.16      1.31
      bound pointer       45.2       +1.4      +3.2       0.13      1.25
   fixed device map       44.3       +0.5      +1.2       0.10      1.23
fixed device no-map       42.9       -0.9      -2.2       0.13      1.23
elapsed_ns=2858475618
exit_status=0
ended=2026-09-25T16:26:48+08:00
```

### sweep: develop

```text
commit=ef698d24dae8790fffdf84fe766a7bf06db428f6
started=2026-09-25T16:24:07+08:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --sweep --no-gpu --iters 100000 --batches 9
mode=sweep; host-only; 100000 calls × 9 batches; median ns/call
count   kind    ns/call
    4    int       41.5
    4 tensor       40.7
   16    int       74.0
   16 tensor       68.6
   32    int      116.8
   32 tensor      120.3
elapsed_ns=3010514222
exit_status=0
ended=2026-09-25T16:24:10+08:00
```

### sweep: merged

```text
commit=2a9f891443cf6763d26483c7b1f3d0f451d90a3c
started=2026-09-25T16:26:48+08:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/torch_abi
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/torch_abi taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --sweep --no-gpu --iters 100000 --batches 9
mode=sweep; host-only; 100000 calls × 9 batches; median ns/call
count   kind    ns/call
    4    int       42.5
    4 tensor       41.7
   16    int       74.2
   16 tensor       68.6
   32    int      121.3
   32 tensor      115.1
elapsed_ns=2922906538
exit_status=0
ended=2026-09-25T16:26:51+08:00
```

### readme: develop

```text
commit=ef698d24dae8790fffdf84fe766a7bf06db428f6
started=2026-09-25T16:24:10+08:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --readme --iters 1000 --batches 9
mode=readme; 1000 calls × 9 batches; median
        path   triton us    intj us   speedup
   grid=(1,)       16.71       3.08      5.4x
   grid=(0,)       13.30       0.03    472.8x

torch_access   decode ns    build s
        shim        92.8       0.02
         cxx       174.3       0.06
     cpython       884.3       0.01
elapsed_ns=3798321293
exit_status=0
ended=2026-09-25T16:24:14+08:00
```

### readme: merged

```text
commit=2a9f891443cf6763d26483c7b1f3d0f451d90a3c
started=2026-09-25T16:26:51+08:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/torch_abi
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/torch_abi taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --readme --iters 1000 --batches 9
mode=readme; 1000 calls × 9 batches; median
        path   triton us    intj us   speedup
   grid=(1,)       16.93       3.68      4.6x
   grid=(0,)       13.10       0.03    436.7x

torch_access_mode   decode ns    build s
     runtime_shim        92.2       0.02
   static_compile        91.7       0.06
      interpreter       875.6       0.00
elapsed_ns=3657776445
exit_status=0
ended=2026-09-25T16:26:54+08:00
```

### hip: develop

```text
commit=ef698d24dae8790fffdf84fe766a7bf06db428f6
started=2026-09-25T16:24:14+08:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_hip_module_launch.py --iters 1000 --batches 9
hsaco_sha256=df3e529e178196ad62f7c07ab8593ede5b8596c76e7c6fbd12a5d947a696e65c kernel=empty_kernel threads=256 shared=0 callback_count=1
iters=1000 batches=9 unit=us/call (host enqueue, no timed sync)
TVM FFI HIP HSACO         median=3.1928 samples=[3.5381080000000003, 3.1927719999999997, 3.2000010000000003, 3.187216, 3.1679, 3.0590300000000004, 3.065448, 13.438449, 5.7156009999999995]
Triton same HSACO         median=15.9598 samples=[16.119324, 16.027525999999998, 16.047228999999998, 15.95977, 15.839692999999999, 15.808174000000001, 16.003415, 15.767453999999999, 15.657005999999999]
INTJ same function        median=2.9398 samples=[2.962548, 2.97735, 2.982076, 3.044069, 2.939846, 2.8773589999999998, 2.901814, 2.93572, 2.852431]
elapsed_ns=3784072183
exit_status=0
ended=2026-09-25T16:24:18+08:00
```

### hip: merged

```text
commit=2a9f891443cf6763d26483c7b1f3d0f451d90a3c
started=2026-09-25T16:26:54+08:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/torch_abi
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/torch_abi taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_hip_module_launch.py --iters 1000 --batches 9
hsaco_sha256=df3e529e178196ad62f7c07ab8593ede5b8596c76e7c6fbd12a5d947a696e65c kernel=empty_kernel threads=256 shared=0 callback_count=1
iters=1000 batches=9 unit=us/call (host enqueue, no timed sync)
TVM FFI HIP HSACO         median=3.0803 samples=[3.342287, 3.13517, 3.1458980000000003, 3.080282, 3.093908, 2.9819560000000003, 3.061076, 2.968804, 3.051986]
Triton same HSACO         median=15.8705 samples=[16.182424, 15.934781000000001, 15.936739, 15.841906000000002, 15.870504, 15.754451, 15.76059, 15.743438, 15.881219]
INTJ same function        median=2.8681 samples=[2.9488760000000003, 2.974676, 2.92977, 2.908071, 2.8681419999999997, 2.854816, 2.8060419999999997, 2.854252, 2.754509]
elapsed_ns=3606627916
exit_status=0
ended=2026-09-25T16:26:58+08:00
```

### ffi_paths: develop

```text
commit=ef698d24dae8790fffdf84fe766a7bf06db428f6
started=2026-09-25T16:24:18+08:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_intj_ffi_paths.py --iters 1000 --batches 9
iters=1000 batches=9 unit=ns/call; no_gpu=True
mode=callback
callback_count=9100 (100 warmup + 9000 unique misses)
INTJ cold compile callback + cache  median_ns=418.2 samples_ns=[459.308, 461.356, 398.169, 554.977, 378.628, 397.59, 411.027, 418.237, 1577.745]
INTJ hot call (no callback)         median_ns=38.9 samples_ns=[40.302, 38.93, 38.182, 39.989, 37.9, 38.886, 37.846, 39.119, 37.797]
INTJ 3-tensor host-only nop         median_ns=35.3 samples_ns=[36.907, 40.971, 35.19, 35.43, 34.577, 35.296, 34.871, 35.333, 34.947]
mode=kwargs
INTJ direct positional              median_ns=66.8 samples_ns=[68.549, 66.945, 66.753, 66.788, 66.751, 66.681, 67.32, 155.462, 66.336]
INTJ adapter positional             median_ns=92.8 samples_ns=[93.764, 92.944, 92.658, 92.124, 92.804, 93.423, 92.419, 91.91, 95.381]
INTJ adapter kwargs                 median_ns=108.7 samples_ns=[112.494, 109.825, 108.212, 109.17, 108.693, 108.337, 108.432, 107.707, 111.879]
INTJ adapter defaults               median_ns=89.7 samples_ns=[90.983, 89.8, 89.649, 89.607, 89.306, 89.745, 89.727, 90.323, 90.615]
INTJ FFI wrapper positional         median_ns=104.6 samples_ns=[112.374, 105.721, 105.406, 103.942, 104.224, 104.026, 104.637, 104.821, 103.844]
INTJ FFI wrapper kwargs             median_ns=128.3 samples_ns=[135.489, 128.571, 129.018, 128.348, 127.353, 127.077, 127.512, 127.388, 143.537]
INTJ native kwargs: unsupported (verified TypeError)
mode=dataclass
INTJ pair direct                    median_ns=70.0 samples_ns=[72.091, 70.528, 69.702, 70.025, 69.507, 77.357, 69.628, 70.162, 69.458]
INTJ pair prebuilt *tuple           median_ns=142.0 samples_ns=[143.05, 144.195, 141.355, 143.867, 140.944, 148.627, 140.215, 141.994, 138.89]
FFI unpack Pair only                median_ns=169.9 samples_ns=[162.801, 163.304, 169.496, 170.38, 171.862, 171.445, 168.597, 169.91, 171.818]
INTJ pair manual unpack             median_ns=174.9 samples_ns=[174.91, 176.132, 173.36, 175.472, 172.591, 184.293, 172.504, 174.875, 173.662]
INTJ pair FFI unpack                median_ns=327.5 samples_ns=[325.775, 327.304, 325.622, 330.17, 327.692, 329.783, 330.056, 327.492, 324.108]
INTJ pair stdlib astuple            median_ns=1171.0 samples_ns=[1339.122, 1209.925, 1161.806, 1175.732, 1159.32, 1170.975, 1158.943, 1178.707, 1168.204]
INTJ config direct                  median_ns=81.1 samples_ns=[80.876, 81.061, 80.235, 80.627, 79.811, 85.175, 83.358, 84.323, 83.143]
INTJ config FFI unpack              median_ns=326.0 samples_ns=[325.932, 328.104, 326.933, 326.023, 321.695, 332.895, 325.37, 327.49, 325.664]
INTJ native dataclass unpack: unsupported (verified TypeError)
elapsed_ns=2987447665
exit_status=0
ended=2026-09-25T16:24:21+08:00
```

### ffi_paths: merged

```text
commit=2a9f891443cf6763d26483c7b1f3d0f451d90a3c
started=2026-09-25T16:26:58+08:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/torch_abi
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/torch_abi taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_intj_ffi_paths.py --iters 1000 --batches 9
iters=1000 batches=9 unit=ns/call; no_gpu=True
mode=callback
callback_count=9100 (100 warmup + 9000 unique misses)
INTJ cold compile callback + cache  median_ns=421.4 samples_ns=[463.584, 455.966, 402.497, 701.303, 390.485, 402.097, 404.238, 421.407, 1466.982]
INTJ hot call (no callback)         median_ns=40.0 samples_ns=[42.511, 41.313, 39.544, 40.034, 38.666, 44.607, 39.351, 40.865, 39.536]
INTJ 3-tensor host-only nop         median_ns=35.6 samples_ns=[37.657, 36.009, 35.451, 36.467, 35.153, 35.632, 35.115, 35.67, 35.12]
mode=kwargs
INTJ direct positional              median_ns=67.1 samples_ns=[69.591, 67.408, 67.227, 67.079, 67.147, 67.082, 67.048, 67.092, 67.231]
INTJ adapter positional             median_ns=90.9 samples_ns=[90.924, 95.871, 90.609, 90.647, 90.964, 91.524, 90.483, 90.066, 90.983]
INTJ adapter kwargs                 median_ns=109.0 samples_ns=[108.143, 107.676, 122.379, 111.683, 111.407, 112.806, 108.699, 108.824, 108.959]
INTJ adapter defaults               median_ns=89.6 samples_ns=[90.851, 93.474, 93.351, 89.623, 88.481, 90.01, 88.518, 88.68, 88.8]
INTJ FFI wrapper positional         median_ns=104.2 samples_ns=[103.163, 103.092, 103.317, 103.252, 111.291, 104.618, 105.17, 104.502, 104.172]
INTJ FFI wrapper kwargs             median_ns=127.8 samples_ns=[126.596, 123.717, 124.722, 133.924, 127.148, 127.815, 128.601, 128.838, 128.132]
INTJ native kwargs: unsupported (verified TypeError)
mode=dataclass
INTJ pair direct                    median_ns=69.0 samples_ns=[71.641, 69.149, 67.953, 69.161, 67.453, 68.953, 67.684, 72.415, 67.057]
INTJ pair prebuilt *tuple           median_ns=142.2 samples_ns=[141.83, 143.52, 140.287, 142.246, 138.147, 146.624, 141.364, 145.264, 142.335]
FFI unpack Pair only                median_ns=162.0 samples_ns=[164.778, 165.889, 163.455, 166.329, 160.073, 161.67, 160.333, 161.993, 161.304]
INTJ pair manual unpack             median_ns=173.6 samples_ns=[173.414, 173.616, 170.458, 173.392, 170.745, 174.938, 177.36, 176.031, 174.382]
INTJ pair FFI unpack                median_ns=316.6 samples_ns=[315.038, 320.005, 313.289, 316.57, 318.097, 316.626, 315.264, 320.72, 317.666]
INTJ pair stdlib astuple            median_ns=1165.2 samples_ns=[1339.492, 1224.202, 1154.808, 1169.488, 1157.359, 1165.248, 1150.693, 1174.004, 1157.641]
INTJ config direct                  median_ns=82.2 samples_ns=[81.941, 83.408, 81.596, 82.855, 81.853, 82.93, 81.832, 83.307, 82.172]
INTJ config FFI unpack              median_ns=323.4 samples_ns=[323.357, 323.533, 325.337, 323.557, 320.293, 323.446, 323.898, 321.664, 316.735]
INTJ native dataclass unpack: unsupported (verified TypeError)
elapsed_ns=2790755959
exit_status=0
ended=2026-09-25T16:27:01+08:00
```

### ffi_default: develop

```text
commit=ef698d24dae8790fffdf84fe766a7bf06db428f6
started=2026-09-25T16:24:21+08:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_ffi_compare.py --iters 1000 --batches 9
iters=1000 batches=9 unit=us/call (host-only, no timed sync)
FFI packed nop             median=0.1439 samples=[0.148961, 0.143672, 0.143867, 0.14515999999999998, 0.14491900000000002, 0.14241800000000002, 0.142594, 0.14438900000000002, 0.14377099999999998]
FFI typed nop              median=0.1493 samples=[0.149533, 0.149393, 0.147597, 0.14977000000000001, 0.149252, 0.149142, 0.146687, 0.155114, 0.148029]
FFI empty kernel           median=3.2927 samples=[3.524629, 3.453683, 3.46835, 3.228406, 3.1501799999999998, 3.300852, 3.19117, 3.292706, 3.1881709999999996]
INTJ empty kernel          median=2.9622 samples=[3.100708, 3.0820410000000003, 3.084065, 2.844688, 2.926745, 2.920518, 2.993846, 2.938551, 2.9622089999999996]
INTJ fixed-device kernel   median=2.8203 samples=[3.048299, 3.096694, 3.074975, 2.8538040000000002, 2.78263, 2.813653, 2.813064, 2.820344, 2.8068589999999998]
FFI packed nop mixed       median=0.1732 samples=[0.17102699999999998, 0.175171, 0.17804599999999998, 0.17680099999999999, 0.173157, 0.172988, 0.170708, 0.16998500000000002, 0.17529599999999998]
FFI typed nop mixed        median=0.1756 samples=[0.17282499999999998, 0.176917, 0.17558600000000002, 0.177876, 0.1748, 0.181855, 0.171794, 0.178042, 0.17227199999999998]
FFI mixed kernel           median=3.3368 samples=[3.412646, 3.5276129999999997, 3.385805, 3.375223, 3.167904, 3.3368339999999996, 3.248255, 3.285393, 3.2357739999999997]
INTJ mixed kernel          median=2.8742 samples=[3.102368, 3.1003130000000003, 2.919701, 2.943619, 2.77791, 2.7875799999999997, 2.821657, 2.874188, 2.836254]
INTJ fixed mixed kernel    median=2.8649 samples=[3.13078, 3.118808, 2.990324, 2.973935, 2.828335, 2.8029070000000003, 2.778697, 2.86079, 2.864944]
elapsed_ns=3799147147
exit_status=0
ended=2026-09-25T16:24:24+08:00
```

### ffi_default: merged

```text
commit=2a9f891443cf6763d26483c7b1f3d0f451d90a3c
started=2026-09-25T16:27:01+08:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/torch_abi
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/torch_abi taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_ffi_compare.py --iters 1000 --batches 9
iters=1000 batches=9 unit=us/call (host-only, no timed sync)
FFI packed nop             median=0.1450 samples=[0.151503, 0.14405400000000002, 0.143997, 0.144996, 0.14497200000000002, 0.145965, 0.148414, 0.14363800000000002, 0.14488]
FFI typed nop              median=0.1492 samples=[0.148308, 0.150302, 0.14847, 0.154046, 0.148361, 0.152029, 0.14922200000000002, 0.14952500000000002, 0.14871600000000001]
FFI empty kernel           median=3.3102 samples=[3.525951, 3.526326, 3.436703, 3.230661, 3.22106, 3.310225, 3.186538, 3.3134650000000003, 3.1942]
INTJ empty kernel          median=2.9904 samples=[3.0789340000000003, 3.039931, 3.044563, 2.86385, 2.819442, 2.89762, 3.00808, 2.890385, 2.9903760000000004]
INTJ fixed-device kernel   median=2.9129 samples=[3.025749, 2.9925590000000004, 3.023507, 2.860006, 2.837826, 2.818331, 2.913274, 2.826276, 2.91295]
FFI packed nop mixed       median=0.1729 samples=[0.17255199999999998, 0.17349799999999999, 0.17514500000000002, 0.17285, 0.171563, 0.172313, 0.17413499999999998, 0.17587, 0.171993]
FFI typed nop mixed        median=0.1784 samples=[0.18424100000000002, 0.179333, 0.179581, 0.178615, 0.172238, 0.178034, 0.178357, 0.178177, 0.17488900000000002]
FFI mixed kernel           median=3.3069 samples=[3.361321, 3.528908, 3.306884, 3.305355, 3.236136, 3.333957, 3.251689, 3.310653, 3.254421]
INTJ mixed kernel          median=2.9078 samples=[2.982939, 3.019137, 2.889884, 2.913091, 2.907804, 2.838717, 2.844038, 2.9674560000000003, 2.8362249999999998]
INTJ fixed mixed kernel    median=2.9000 samples=[3.014918, 3.041575, 2.962969, 2.925085, 2.8734499999999996, 2.858462, 2.8804499999999997, 2.899955, 2.848763]
elapsed_ns=3625792457
exit_status=0
ended=2026-09-25T16:27:05+08:00
```

### ffi_sweep: develop

```text
commit=ef698d24dae8790fffdf84fe766a7bf06db428f6
started=2026-09-25T16:24:25+08:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_ffi_compare.py --sweep --iters 1000 --batches 9
counts=[0, 3, 5, 8, 16, 32, 64] iters=1000 batches=9 unit=us/call (host-only, no timed sync); INTJ device-bound, no specialization key
args= 0 FFI packed nop           median=0.1176 samples=[0.11927299999999999, 0.118812, 0.116272, 0.117638, 0.11687399999999999, 0.11751900000000001, 0.117802, 0.118633, 0.116498]
args= 0 FFI typed nop            median=0.1203 samples=[0.120642, 0.12166400000000001, 0.116779, 0.120587, 0.117554, 0.120281, 0.11928799999999999, 0.121195, 0.11943300000000001]
args= 0 FFI empty kernel         median=1.6589 samples=[1.5994970000000002, 1.932836, 1.6588910000000001, 1.8406310000000001, 1.5605909999999998, 1.836724, 1.5746310000000001, 1.835859, 1.583489]
args= 0 INTJ cxx kernel          median=2.7132 samples=[3.0370369999999998, 2.75575, 2.788616, 2.713188, 2.696407, 2.6994209999999996, 2.704114, 2.724873, 2.7088919999999996]
args= 0 INTJ shim kernel         median=2.7004 samples=[2.822485, 2.664065, 2.7740880000000003, 2.658629, 2.708501, 2.653306, 2.700356, 2.6619200000000003, 2.7263040000000003]
args= 3 FFI packed nop           median=0.1439 samples=[0.14574, 0.147044, 0.143539, 0.14383500000000002, 0.144392, 0.14596, 0.143748, 0.142839, 0.143944]
args= 3 FFI typed nop            median=0.1492 samples=[0.14974500000000002, 0.147724, 0.147786, 0.153418, 0.146881, 0.151098, 0.14733000000000002, 0.14918199999999998, 0.149228]
args= 3 FFI empty kernel         median=3.2645 samples=[3.265176, 3.264492, 3.1830160000000003, 3.273558, 3.189475, 3.269563, 3.203843, 3.285927, 3.217275]
args= 3 INTJ cxx kernel          median=2.8574 samples=[2.907989, 2.797337, 2.9695810000000002, 2.892149, 2.8710839999999997, 2.845466, 2.8414639999999998, 2.847946, 2.857367]
args= 3 INTJ shim kernel         median=2.8761 samples=[2.942032, 2.794879, 2.9366950000000003, 2.804808, 2.9649940000000004, 2.81349, 2.876136, 2.814692, 2.890089]
args= 5 FFI packed nop           median=0.1729 samples=[0.177578, 0.172923, 0.173599, 0.172745, 0.17155600000000001, 0.177164, 0.172657, 0.171923, 0.17316399999999998]
args= 5 FFI typed nop            median=0.1790 samples=[0.179027, 0.179239, 0.174578, 0.178695, 0.174079, 0.18285200000000001, 0.177208, 0.180689, 0.181249]
args= 5 FFI empty kernel         median=3.2932 samples=[3.3808939999999996, 3.300211, 3.312935, 3.278174, 10.044923, 3.2931559999999998, 3.279794, 3.285085, 3.286025]
args= 5 INTJ cxx kernel          median=2.9019 samples=[2.98019, 2.88857, 2.9018629999999996, 2.8571869999999997, 8.442681, 2.879623, 3.104752, 14.954993, 2.880267]
args= 5 INTJ shim kernel         median=2.8863 samples=[2.975431, 2.889835, 2.869036, 2.886326, 2.830805, 2.892394, 2.826812, 6.271026, 2.830129]
args= 8 FFI packed nop           median=0.2049 samples=[0.206422, 0.206082, 0.20493199999999998, 0.203406, 0.20316900000000002, 0.204399, 0.203753, 0.20522100000000001, 0.207854]
args= 8 FFI typed nop            median=0.2091 samples=[0.209341, 0.20958000000000002, 0.208094, 0.210081, 0.208014, 0.213494, 0.20559200000000002, 0.209054, 0.207412]
args= 8 FFI empty kernel         median=3.3842 samples=[3.485639, 3.362264, 3.50932, 3.416142, 3.312728, 3.4286689999999997, 3.281896, 3.3842489999999996, 3.318286]
args= 8 INTJ cxx kernel          median=3.0213 samples=[3.021271, 2.8878519999999996, 3.027812, 2.997577, 3.092404, 2.9951239999999997, 3.050833, 3.003962, 3.059909]
args= 8 INTJ shim kernel         median=2.9145 samples=[3.023008, 2.881322, 2.9749830000000004, 2.8934, 2.968204, 2.914514, 2.9673939999999996, 2.882232, 2.8864720000000004]
args=16 FFI packed nop           median=0.2918 samples=[0.29092599999999996, 0.289834, 0.291782, 0.29028, 0.296235, 0.295584, 0.291707, 0.291949, 0.29352]
args=16 FFI typed nop            median=0.3049 samples=[0.30346100000000004, 0.304873, 0.305502, 0.30324599999999996, 0.29780700000000004, 0.305225, 0.302449, 0.30568599999999996, 0.307576]
args=16 FFI empty kernel         median=3.6511 samples=[3.7014299999999998, 3.707058, 3.559292, 3.651103, 3.555114, 3.6931619999999996, 3.556962, 3.6626909999999997, 3.56866]
args=16 INTJ cxx kernel          median=3.3546 samples=[3.2465439999999997, 3.34064, 3.499946, 3.327703, 3.534977, 3.354574, 3.459414, 3.32958, 3.487597]
args=16 INTJ shim kernel         median=3.2281 samples=[3.2280819999999997, 3.081293, 3.397207, 3.040922, 3.371082, 3.064644, 3.365714, 3.047287, 3.341752]
args=32 FFI packed nop           median=0.4765 samples=[0.474922, 0.47652999999999995, 0.478122, 0.475834, 0.469198, 0.480482, 0.477669, 0.475599, 0.491979]
args=32 FFI typed nop            median=0.4943 samples=[0.49324, 0.494338, 0.495356, 0.494272, 0.48379500000000003, 0.491092, 0.492953, 0.5122859999999999, 0.497634]
args=32 FFI empty kernel         median=4.1359 samples=[4.211378, 4.135872, 4.048922999999999, 4.1829719999999995, 4.046763, 4.160997999999999, 4.022943000000001, 4.1620550000000005, 4.03444]
args=32 INTJ cxx kernel          median=3.6492 samples=[3.504453, 7.928286, 3.681612, 3.614364, 3.699043, 3.622118, 3.649168, 3.588895, 3.663454]
args=32 INTJ shim kernel         median=3.5971 samples=[3.430582, 4.096036, 3.6040970000000003, 3.4300189999999997, 3.642405, 3.425983, 3.61847, 3.4143939999999997, 3.5971379999999997]
args=64 FFI packed nop           median=0.8415 samples=[0.842135, 0.806509, 0.8461040000000001, 0.852908, 0.8266140000000001, 0.8395990000000001, 0.841474, 0.8241240000000001, 0.858032]
args=64 FFI typed nop            median=0.8620 samples=[0.861607, 0.9035449999999999, 0.8716309999999999, 0.857951, 0.861982, 0.865845, 0.8582989999999999, 0.8493160000000001, 0.885532]
args=64 FFI empty kernel         median=5.0474 samples=[5.045299, 5.04321, 5.062977, 5.047129, 5.088924, 5.057783000000001, 5.0676689999999995, 5.047435999999999, 5.026537]
args=64 INTJ cxx kernel          median=4.6889 samples=[4.646624, 4.655798, 4.685899, 4.688877000000001, 4.720245, 4.710831, 4.689143, 4.694545, 4.645936]
args=64 INTJ shim kernel         median=4.6781 samples=[4.672470000000001, 4.678061, 4.6460230000000005, 4.70162, 4.673310000000001, 4.707648, 4.6487680000000005, 4.6949939999999994, 4.683528]
elapsed_ns=4283732438
exit_status=0
ended=2026-09-25T16:24:29+08:00
```

### ffi_sweep: merged

```text
commit=2a9f891443cf6763d26483c7b1f3d0f451d90a3c
started=2026-09-25T16:27:05+08:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/torch_abi
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/torch_abi taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_ffi_compare.py --sweep --iters 1000 --batches 9
counts=[0, 3, 5, 8, 16, 32, 64] iters=1000 batches=9 unit=us/call (host-only, no timed sync); INTJ device-bound, no specialization key
args= 0 FFI packed nop           median=0.1171 samples=[0.121797, 0.1173, 0.117149, 0.11556699999999999, 0.117616, 0.113827, 0.11381699999999999, 0.118521, 0.116573]
args= 0 FFI typed nop            median=0.1186 samples=[0.12214900000000001, 0.118583, 0.117395, 0.120796, 0.117644, 0.118601, 0.117358, 0.1215, 0.121286]
args= 0 FFI empty kernel         median=1.6606 samples=[1.625412, 1.90354, 1.660626, 1.871981, 1.6549390000000002, 1.888142, 1.6476359999999999, 1.796326, 1.5879770000000002]
args= 0 INTJ static_compile kernel median=2.7424 samples=[3.058122, 2.697173, 2.755241, 2.650388, 2.742442, 2.76071, 2.754201, 2.721519, 2.722605]
args= 0 INTJ runtime_shim kernel median=2.7245 samples=[2.820944, 2.65071, 2.761445, 2.662127, 2.762967, 2.659465, 2.7537979999999997, 2.674092, 2.7244699999999997]
args= 3 FFI packed nop           median=0.1444 samples=[0.144674, 0.142901, 0.145631, 0.143527, 0.144357, 0.141982, 0.14392, 0.14459, 0.14455]
args= 3 FFI typed nop            median=0.1495 samples=[0.150275, 0.153898, 0.14704599999999998, 0.149618, 0.15308000000000002, 0.14947, 0.14801599999999998, 0.14703899999999998, 0.149195]
args= 3 FFI empty kernel         median=3.1941 samples=[3.2495410000000002, 3.162299, 3.161404, 3.1995839999999998, 3.1391199999999997, 3.194107, 3.152535, 3.237327, 3.208274]
args= 3 INTJ static_compile kernel median=2.8756 samples=[2.909798, 2.775676, 2.962944, 2.816022, 2.9387339999999997, 2.813879, 2.91674, 2.867487, 2.875581]
args= 3 INTJ runtime_shim kernel median=2.8186 samples=[2.9186959999999997, 2.782736, 2.781703, 2.818623, 2.800966, 2.8205839999999998, 2.773726, 2.8770100000000003, 2.865739]
args= 5 FFI packed nop           median=0.1727 samples=[0.172344, 0.172262, 0.172685, 0.173757, 0.173324, 0.17124199999999998, 0.17601499999999998, 0.172745, 0.17380299999999999]
args= 5 FFI typed nop            median=0.1781 samples=[0.179356, 0.179922, 0.176014, 0.19346000000000002, 0.175315, 0.175755, 0.181993, 0.178136, 0.17184]
args= 5 FFI empty kernel         median=3.2718 samples=[3.275735, 3.308524, 3.1775100000000003, 3.284746, 3.180669, 3.325523, 3.156345, 3.239727, 3.271806]
args= 5 INTJ static_compile kernel median=2.8701 samples=[2.953228, 2.853325, 2.8700579999999998, 2.801968, 2.829634, 2.8000010000000004, 2.966017, 2.887355, 2.916257]
args= 5 INTJ runtime_shim kernel median=2.8460 samples=[2.963348, 2.7897469999999998, 2.8525500000000004, 2.758773, 2.79798, 2.762156, 2.8459540000000003, 2.8472890000000004, 2.8518890000000003]
args= 8 FFI packed nop           median=0.2059 samples=[0.20510599999999998, 0.20613399999999998, 0.209987, 0.204178, 0.20686600000000002, 0.205933, 0.208399, 0.204917, 0.204089]
args= 8 FFI typed nop            median=0.2081 samples=[0.206954, 0.210068, 0.20647900000000002, 0.20793, 0.206707, 0.213828, 0.208081, 0.21021700000000001, 0.21096]
args= 8 FFI empty kernel         median=3.3956 samples=[3.463727, 3.395644, 3.3244290000000003, 3.4053, 3.2733049999999997, 3.401126, 3.300719, 3.398993, 3.338352]
args= 8 INTJ static_compile kernel median=2.9989 samples=[2.998884, 2.879298, 3.059185, 2.976509, 3.043373, 2.974469, 3.029849, 3.1712260000000003, 2.927955]
args= 8 INTJ runtime_shim kernel median=2.9204 samples=[2.994103, 2.854133, 2.920372, 2.881206, 2.9213470000000004, 2.878779, 2.942554, 2.8730659999999997, 2.9427440000000002]
args=16 FFI packed nop           median=0.2915 samples=[0.298546, 0.29131799999999997, 0.29152100000000003, 0.28928899999999996, 0.295119, 0.288217, 0.294356, 0.293222, 0.29017899999999996]
args=16 FFI typed nop            median=0.3036 samples=[0.306232, 0.303635, 0.301139, 0.301647, 0.304476, 0.304924, 0.297053, 0.30697800000000003, 0.300573]
args=16 FFI empty kernel         median=3.6228 samples=[3.64455, 3.6419099999999998, 3.5129029999999997, 3.638627, 3.502746, 3.622827, 3.514208, 3.651399, 3.536714]
args=16 INTJ static_compile kernel median=3.3219 samples=[3.203462, 3.295527, 3.444059, 3.3219160000000003, 3.459158, 3.285693, 3.4532640000000003, 3.280967, 3.585098]
args=16 INTJ runtime_shim kernel median=3.1952 samples=[3.1951959999999997, 3.074332, 3.352224, 3.036067, 3.312852, 3.020664, 3.3766260000000003, 3.034287, 3.3941660000000002]
args=32 FFI packed nop           median=0.4760 samples=[0.47145299999999996, 0.475482, 0.468906, 0.471481, 0.477678, 0.479904, 0.47793599999999997, 0.476037, 0.47759500000000005]
args=32 FFI typed nop            median=0.4908 samples=[0.491779, 0.49743200000000004, 0.47651699999999997, 0.49732499999999996, 0.482022, 0.491404, 0.490591, 0.49080900000000005, 0.489958]
args=32 FFI empty kernel         median=4.1836 samples=[4.294916, 4.183612, 4.141555, 4.270912999999999, 4.124104, 4.2692619999999994, 4.127800000000001, 4.222934, 4.142446]
args=32 INTJ static_compile kernel median=3.5682 samples=[3.4452510000000003, 3.567804, 3.626883, 3.5682289999999997, 3.6467330000000002, 3.5466439999999997, 3.651927, 3.557725, 3.675405]
args=32 INTJ runtime_shim kernel median=3.4284 samples=[3.428351, 3.420229, 3.620225, 3.427794, 3.630833, 3.419183, 3.639442, 3.425487, 3.679965]
args=64 FFI packed nop           median=0.8318 samples=[0.846012, 0.86037, 0.831809, 0.8315499999999999, 0.83362, 0.835935, 0.831705, 0.8182269999999999, 0.818798]
args=64 FFI typed nop            median=0.8607 samples=[0.872367, 0.879355, 0.864539, 0.860735, 0.852485, 0.864811, 0.8516520000000001, 0.846877, 0.837703]
args=64 FFI empty kernel         median=5.0279 samples=[5.049803, 5.009195, 5.027937, 5.015235, 5.017076, 5.0374799999999995, 5.0219309999999995, 5.038966, 5.034122999999999]
args=64 INTJ static_compile kernel median=4.5899 samples=[4.598361, 4.523964, 4.587548, 4.588520000000001, 4.57111, 4.589889, 4.598584, 4.632785, 4.613356]
args=64 INTJ runtime_shim kernel median=4.5430 samples=[4.4866019999999995, 4.574444000000001, 4.540466, 4.554689000000001, 4.526002, 4.542957, 4.576314999999999, 4.575771, 4.5292889999999995]
elapsed_ns=4169489848
exit_status=0
ended=2026-09-25T16:27:09+08:00
```

### Controlled 100-call sweep: develop

```text
commit=ef698d24dae8790fffdf84fe766a7bf06db428f6
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_ffi_compare.py --sweep --iters 100 --batches 9
counts=[0, 3, 5, 8, 16, 32, 64] iters=100 batches=9 unit=us/call (host-only, no timed sync); INTJ device-bound, no specialization key
args= 0 FFI packed nop           median=0.1175 samples=[0.12012, 0.11952, 0.11866, 0.11616, 0.11548, 0.11635, 0.11631999999999999, 0.11768, 0.11754]
args= 0 FFI typed nop            median=0.1198 samples=[0.12243000000000001, 0.123, 0.11842, 0.12042, 0.11846999999999999, 0.11984, 0.11729, 0.12088, 0.11963]
args= 0 FFI empty kernel         median=2.7191 samples=[2.88222, 2.78146, 2.70489, 2.71908, 2.70088, 2.53042, 2.39379, 2.82415, 2.90111]
args= 0 INTJ cxx kernel          median=3.5892 samples=[3.8241, 3.59037, 3.8715300000000004, 3.5625, 3.57502, 3.54502, 3.5585999999999998, 3.73471, 3.58923]
args= 0 INTJ shim kernel         median=3.5924 samples=[3.70008, 3.49521, 3.7177800000000003, 3.7177, 3.61731, 3.46051, 3.39195, 3.57943, 3.59243]
args= 3 FFI packed nop           median=0.1423 samples=[0.15649000000000002, 0.14226, 0.14189, 0.14132, 0.1404, 0.14405, 0.14327, 0.14353, 0.142]
args= 3 FFI typed nop            median=0.1452 samples=[0.14393999999999998, 0.14704, 0.14523, 0.14606, 0.14475, 0.14837, 0.1422, 0.14893, 0.1426]
args= 3 FFI empty kernel         median=4.1523 samples=[5.54629, 4.26613, 4.27844, 4.21816, 4.1523, 4.04539, 4.09134, 4.08775, 4.04185]
args= 3 INTJ cxx kernel          median=3.7753 samples=[3.8994, 4.15796, 3.9869, 3.9201599999999996, 3.7465900000000003, 3.65821, 3.65834, 3.7092199999999997, 3.77527]
args= 3 INTJ shim kernel         median=3.8698 samples=[3.8698, 4.03408, 4.036239999999999, 4.01075, 3.84085, 3.87767, 3.7338, 3.79544, 3.7784]
args= 5 FFI packed nop           median=0.1714 samples=[0.17054, 0.17073, 0.172, 0.16877, 0.17274, 0.17168, 0.17163, 0.17012, 0.17145]
args= 5 FFI typed nop            median=0.1740 samples=[0.17392, 0.17717, 0.17643999999999999, 0.17306, 0.174, 0.17342, 0.17425, 0.1752, 0.1713]
args= 5 FFI empty kernel         median=4.2446 samples=[4.18252, 4.2445699999999995, 4.40811, 4.26876, 4.07113, 4.10546, 4.2698599999999995, 4.29665, 4.0069]
args= 5 INTJ cxx kernel          median=3.8456 samples=[3.8336900000000003, 3.9019, 3.96745, 3.97019, 3.8713, 3.7792600000000003, 3.66821, 3.83877, 3.84562]
args= 5 INTJ shim kernel         median=3.8662 samples=[3.9865399999999998, 3.9578699999999998, 3.96867, 3.9490700000000003, 3.86619, 3.73944, 3.677, 3.7789699999999997, 3.78113]
args= 8 FFI packed nop           median=0.2044 samples=[0.2063, 0.20274, 0.20321, 0.20441, 0.20451, 0.20441, 0.20333, 0.20095, 0.20653]
args= 8 FFI typed nop            median=0.2066 samples=[0.20664, 0.2061, 0.20635, 0.20661000000000002, 0.20615, 0.20692, 0.20733000000000001, 0.20743, 0.20563]
args= 8 FFI empty kernel         median=4.1737 samples=[4.49482, 4.3715, 4.43289, 4.42411, 4.07427, 4.1545, 4.02738, 4.06346, 4.17373]
args= 8 INTJ cxx kernel          median=3.9470 samples=[3.96802, 3.94697, 4.00051, 4.071219999999999, 3.82961, 3.89413, 3.8623499999999997, 8.47965, 3.82245]
args= 8 INTJ shim kernel         median=4.0149 samples=[4.11993, 4.03955, 4.03242, 3.8466799999999997, 4.01492, 3.71304, 3.6684200000000002, 12.46686, 3.72064]
args=16 FFI packed nop           median=0.2916 samples=[0.29628, 0.29152999999999996, 0.29108999999999996, 0.29275, 0.29029000000000005, 0.28964, 0.29158, 0.6929099999999999, 0.29334]
args=16 FFI typed nop            median=0.3016 samples=[0.30051, 0.3049, 0.30185, 0.30165, 0.29999000000000003, 0.30242, 0.30143000000000003, 0.76532, 0.29813]
args=16 FFI empty kernel         median=6.1578 samples=[5.8938999999999995, 6.157760000000001, 6.691470000000001, 6.028359999999999, 6.33908, 5.77145, 6.389729999999999, 23.90564, 5.78427]
args=16 INTJ cxx kernel          median=5.8486 samples=[5.54418, 6.08284, 5.8486, 5.72564, 5.70108, 5.487010000000001, 6.0165500000000005, 19.865389999999998, 6.00084]
args=16 INTJ shim kernel         median=5.6416 samples=[5.246239999999999, 5.9291, 5.2, 6.22001, 5.70529, 5.641649999999999, 5.37997, 18.85575, 5.54525]
args=32 FFI packed nop           median=0.4788 samples=[0.47132, 0.47714, 0.51275, 0.47877, 0.5125, 0.47444, 0.51225, 0.99301, 0.47335000000000005]
args=32 FFI typed nop            median=0.4950 samples=[0.48355000000000004, 0.49027, 0.48981, 0.49513, 0.49597, 0.4972, 0.49498000000000003, 1.02894, 0.49243000000000003]
args=32 FFI empty kernel         median=6.2823 samples=[6.28228, 5.99423, 6.36439, 6.36172, 6.11725, 6.0538, 6.269640000000001, 14.755870000000002, 6.34437]
args=32 INTJ cxx kernel          median=6.1145 samples=[5.8848, 6.14747, 6.12024, 5.584569999999999, 5.9277, 5.82601, 6.11451, 6.3346, 6.15323]
args=32 INTJ shim kernel         median=5.8465 samples=[5.9346499999999995, 6.19708, 5.84648, 6.04683, 5.2681000000000004, 5.76736, 5.7500100000000005, 7.01054, 5.35612]
args=64 FFI packed nop           median=0.8427 samples=[0.84269, 0.87597, 0.8370299999999999, 0.84975, 0.85399, 0.83182, 0.8375199999999999, 0.83674, 0.87181]
args=64 FFI typed nop            median=0.8614 samples=[0.85862, 0.8826799999999999, 0.85078, 0.8606, 0.91038, 0.88787, 0.89615, 0.86066, 0.86139]
args=64 FFI empty kernel         median=6.7498 samples=[6.76753, 6.74983, 6.97184, 6.7076400000000005, 6.89225, 6.64625, 6.75296, 6.55752, 6.74226]
args=64 INTJ cxx kernel          median=6.4509 samples=[6.77266, 6.45095, 6.701779999999999, 6.417859999999999, 6.6513100000000005, 6.183979999999999, 6.52968, 6.2251, 6.376270000000001]
args=64 INTJ shim kernel         median=6.3938 samples=[6.77023, 6.65398, 6.488689999999999, 6.0711, 6.24758, 6.19504, 6.393800000000001, 6.2678, 6.51679]
```

### Controlled 100-call sweep: merged

```text
commit=2a9f891443cf6763d26483c7b1f3d0f451d90a3c
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/torch_abi taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_ffi_compare.py --sweep --iters 100 --batches 9
counts=[0, 3, 5, 8, 16, 32, 64] iters=100 batches=9 unit=us/call (host-only, no timed sync); INTJ device-bound, no specialization key
args= 0 FFI packed nop           median=0.1167 samples=[0.12403, 0.1173, 0.11679, 0.11677, 0.11669, 0.11578000000000001, 0.11443, 0.11525, 0.11368]
args= 0 FFI typed nop            median=0.1180 samples=[0.12315000000000001, 0.12117, 0.11798, 0.11777, 0.11955, 0.11906, 0.11766, 0.11669, 0.11729]
args= 0 FFI empty kernel         median=1.8461 samples=[1.77289, 2.0283599999999997, 2.0517800000000004, 1.85473, 1.8600999999999999, 1.84609, 1.77926, 1.80323, 1.8175599999999998]
args= 0 INTJ static_compile kernel median=2.9254 samples=[3.29259, 3.09175, 3.15108, 2.92536, 2.89982, 2.87469, 2.90366, 2.99517, 2.87326]
args= 0 INTJ runtime_shim kernel median=2.9625 samples=[3.23427, 3.0303299999999997, 3.04562, 3.05793, 2.96254, 2.88596, 2.8268, 2.78784, 2.84371]
args= 3 FFI packed nop           median=0.1428 samples=[0.14581, 0.14442, 0.14113, 0.14193, 0.14282, 0.14315, 0.14153, 0.14327, 0.14145]
args= 3 FFI typed nop            median=0.1466 samples=[0.1468, 0.14733000000000002, 0.14368, 0.14704, 0.14209, 0.14658000000000002, 0.14215, 0.1477, 0.14288]
args= 3 FFI empty kernel         median=3.4148 samples=[4.794029999999999, 3.63676, 3.56692, 3.59552, 3.36255, 3.337, 3.30117, 3.36525, 3.41481]
args= 3 INTJ static_compile kernel median=3.0470 samples=[3.19451, 3.24617, 3.22864, 3.2328, 3.04704, 2.91225, 2.94963, 3.03065, 3.0107399999999997]
args= 3 INTJ runtime_shim kernel median=3.1581 samples=[3.30329, 3.32912, 3.2731500000000002, 3.26996, 3.1581, 3.1184499999999997, 3.0880900000000002, 3.07126, 3.01272]
args= 5 FFI packed nop           median=0.1704 samples=[0.17105, 0.17182, 0.16852, 0.1704, 0.17035, 0.16908, 0.16905, 0.20095, 0.16921]
args= 5 FFI typed nop            median=0.1737 samples=[0.1739, 0.1752, 0.17242, 0.17335, 0.17292, 0.17404, 0.17458, 0.17374, 0.17139]
args= 5 FFI empty kernel         median=3.4359 samples=[3.57559, 3.60252, 3.6273400000000002, 3.6307300000000002, 3.37702, 3.33689, 3.3296, 3.43592, 3.33477]
args= 5 INTJ static_compile kernel median=3.1220 samples=[3.2361, 3.29155, 3.22106, 3.20321, 3.122, 2.96063, 3.00457, 3.0751500000000003, 3.02529]
args= 5 INTJ runtime_shim kernel median=3.1168 samples=[3.32266, 3.26988, 3.22165, 3.24579, 3.11681, 3.10017, 3.00914, 3.05902, 3.00456]
args= 8 FFI packed nop           median=0.2028 samples=[0.20521, 0.20285, 0.20254, 0.20448, 0.20181, 0.20258, 0.20217, 0.20408, 0.20279]
args= 8 FFI typed nop            median=0.2062 samples=[0.20857, 0.20676, 0.20957, 0.20446999999999999, 0.20624, 0.20646, 0.20535, 0.20622, 0.20523]
args= 8 FFI empty kernel         median=3.5497 samples=[3.74363, 3.6292600000000004, 3.74488, 3.6632399999999996, 3.4785399999999997, 3.50041, 3.4571799999999997, 3.43711, 3.54971]
args= 8 INTJ static_compile kernel median=3.1800 samples=[3.2632, 3.3224, 3.29788, 3.34231, 3.1256399999999998, 3.11648, 3.07003, 3.17998, 3.03763]
args= 8 INTJ runtime_shim kernel median=3.0890 samples=[3.1925, 3.28042, 3.2807, 3.24069, 3.06615, 3.0659099999999997, 3.0219899999999997, 3.0889699999999998, 2.99477]
args=16 FFI packed nop           median=0.2921 samples=[0.29516000000000003, 0.28702, 0.29175, 0.29327, 0.29214999999999997, 0.29295, 0.29014999999999996, 0.29406, 0.29123000000000004]
args=16 FFI typed nop            median=0.3021 samples=[0.30210000000000004, 0.30057999999999996, 0.30096, 0.30557999999999996, 0.29543, 0.30589, 0.30033, 0.30234, 0.30260000000000004]
args=16 FFI empty kernel         median=4.0913 samples=[4.0652, 3.99351, 4.44175, 4.09131, 4.409800000000001, 3.99771, 4.295100000000001, 3.8806, 4.19404]
args=16 INTJ static_compile kernel median=3.6024 samples=[3.7988899999999997, 3.60236, 3.77731, 3.65768, 3.5532600000000003, 3.63389, 3.50898, 3.42246, 3.5841999999999996]
args=16 INTJ runtime_shim kernel median=3.5723 samples=[3.73646, 3.83981, 3.74507, 3.7785, 3.57233, 3.5545500000000003, 3.38319, 3.55302, 3.38543]
args=32 FFI packed nop           median=0.4756 samples=[0.51859, 0.5173300000000001, 0.4775, 0.47369, 0.47562, 0.47764999999999996, 0.47252999999999995, 0.47454, 0.47413]
args=32 FFI typed nop            median=0.4901 samples=[0.48363, 0.49387000000000003, 0.48611, 0.48502999999999996, 0.49086, 0.49134, 0.49011, 0.49429, 0.48857]
args=32 FFI empty kernel         median=4.4168 samples=[4.69473, 4.52322, 4.47797, 4.50754, 4.41676, 4.35963, 4.35855, 4.35914, 4.36784]
args=32 INTJ static_compile kernel median=3.8952 samples=[4.064760000000001, 3.9067800000000004, 4.015280000000001, 3.97094, 3.74152, 3.80415, 3.85801, 3.78393, 3.89521]
args=32 INTJ runtime_shim kernel median=3.9276 samples=[4.13604, 4.084, 3.92173, 4.04664, 3.92764, 3.82609, 3.66326, 3.9285300000000003, 3.7448200000000003]
args=64 FFI packed nop           median=0.8327 samples=[0.8422999999999999, 0.82485, 0.8327, 0.81186, 0.8269099999999999, 0.8409, 0.83417, 0.82801, 0.8342999999999999]
args=64 FFI typed nop            median=0.8646 samples=[0.8533999999999999, 0.90652, 0.90106, 0.8576, 0.86466, 0.85634, 0.86463, 0.85715, 0.8976900000000001]
args=64 FFI empty kernel         median=5.3544 samples=[5.50563, 5.40492, 5.48055, 5.51247, 5.2716400000000005, 5.354430000000001, 5.23775, 5.20495, 5.283189999999999]
args=64 INTJ static_compile kernel median=4.8653 samples=[5.01791, 4.92415, 5.01121, 5.01705, 4.865270000000001, 4.69542, 4.71087, 4.70051, 4.78628]
args=64 INTJ runtime_shim kernel median=4.7550 samples=[5.06102, 4.98362, 4.91776, 5.01382, 4.7237, 4.73655, 4.75497, 4.71096, 4.72223]
```


## Final source `03a8f52`: affected benchmark repeats

### ffi_paths: final merged, round 1

```text
commit=03a8f52651248cb333ed50347fbabac6bd15103a
started=2026-09-25T16:35:30+08:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/torch_abi
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/torch_abi taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_intj_ffi_paths.py --iters 1000 --batches 9
iters=1000 batches=9 unit=ns/call; no_gpu=True
mode=callback
callback_count=9100 (100 warmup + 9000 unique misses)
INTJ cold compile callback + cache  median_ns=419.2 samples_ns=[462.706, 453.658, 406.216, 678.483, 394.004, 402.142, 415.222, 419.159, 1515.858]
INTJ hot call (no callback)         median_ns=39.8 samples_ns=[41.517, 41.322, 38.586, 39.819, 38.821, 40.042, 38.742, 40.147, 38.717]
INTJ 3-tensor host-only nop         median_ns=35.1 samples_ns=[37.169, 39.524, 34.878, 35.361, 34.788, 35.163, 34.75, 35.139, 34.73]
mode=kwargs
INTJ direct positional              median_ns=66.5 samples_ns=[67.966, 65.795, 66.02, 65.741, 66.41, 69.77, 66.514, 66.708, 67.246]
INTJ adapter positional             median_ns=89.2 samples_ns=[90.326, 90.486, 89.006, 89.047, 89.074, 89.067, 89.541, 89.157, 103.689]
INTJ adapter kwargs                 median_ns=107.7 samples_ns=[107.579, 107.746, 107.516, 107.957, 110.567, 107.251, 107.857, 107.441, 112.224]
INTJ adapter defaults               median_ns=89.5 samples_ns=[90.301, 89.382, 89.322, 89.467, 89.734, 89.343, 89.612, 89.909, 89.478]
INTJ FFI wrapper positional         median_ns=104.9 samples_ns=[119.519, 105.479, 104.929, 104.177, 105.375, 104.749, 105.187, 103.875, 104.849]
INTJ FFI wrapper kwargs             median_ns=126.2 samples_ns=[138.301, 126.633, 125.476, 126.234, 125.417, 125.299, 126.198, 126.155, 133.391]
INTJ native kwargs: unsupported (verified TypeError)
mode=dataclass
INTJ pair direct                    median_ns=67.5 samples_ns=[69.633, 72.593, 68.221, 67.548, 66.895, 73.607, 67.064, 67.413, 66.882]
INTJ pair prebuilt *tuple           median_ns=140.9 samples_ns=[142.063, 143.979, 140.6, 144.618, 137.831, 142.153, 139.172, 140.933, 138.132]
FFI unpack Pair only                median_ns=162.5 samples_ns=[171.886, 165.452, 161.435, 163.127, 161.641, 162.494, 164.924, 162.395, 160.796]
INTJ pair manual unpack             median_ns=172.5 samples_ns=[170.174, 172.75, 171.015, 174.847, 172.301, 173.645, 173.554, 172.468, 171.188]
INTJ pair FFI unpack                median_ns=317.4 samples_ns=[318.385, 318.443, 314.814, 323.951, 313.477, 316.355, 317.399, 317.962, 315.126]
INTJ pair stdlib astuple            median_ns=1160.9 samples_ns=[1347.094, 1228.997, 1146.923, 1185.185, 1160.89, 1157.485, 1159.574, 1187.227, 1150.254]
INTJ config direct                  median_ns=81.8 samples_ns=[80.473, 80.758, 79.955, 83.762, 81.583, 81.951, 81.831, 82.584, 81.766]
INTJ config FFI unpack              median_ns=320.5 samples_ns=[320.329, 327.624, 318.663, 319.635, 320.766, 321.061, 320.484, 324.339, 319.553]
INTJ native dataclass unpack: unsupported (verified TypeError)
elapsed_ns=2772593925
exit_status=0
ended=2026-09-25T16:35:33+08:00
```

### ffi_default: final merged, round 1

```text
commit=03a8f52651248cb333ed50347fbabac6bd15103a
started=2026-09-25T16:35:33+08:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/torch_abi
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/torch_abi taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_ffi_compare.py --iters 1000 --batches 9
iters=1000 batches=9 unit=us/call (host-only, no timed sync)
FFI packed nop             median=0.1453 samples=[0.152439, 0.144921, 0.14622300000000002, 0.145151, 0.143213, 0.145285, 0.145731, 0.144631, 0.146646]
FFI typed nop              median=0.1503 samples=[0.15031899999999998, 0.151888, 0.14772200000000002, 0.152639, 0.147074, 0.15104599999999999, 0.147522, 0.151895, 0.148053]
FFI empty kernel           median=3.3038 samples=[3.545978, 3.4613359999999997, 3.303757, 3.2019140000000004, 3.202789, 3.365606, 3.2146280000000003, 3.366981, 3.199154]
INTJ empty kernel          median=3.0181 samples=[3.0538220000000003, 3.031419, 3.1249499999999997, 2.902918, 2.870639, 2.986993, 3.018071, 3.189223, 3.0145030000000004]
INTJ fixed-device kernel   median=2.9048 samples=[2.979897, 2.998917, 3.029217, 2.9048130000000003, 2.865266, 2.823884, 2.843975, 2.834028, 2.938743]
FFI packed nop mixed       median=0.1737 samples=[0.172123, 0.17228100000000002, 0.17366900000000002, 0.174517, 0.174487, 0.174231, 0.173733, 0.173382, 0.17393]
FFI typed nop mixed        median=0.1783 samples=[0.17412, 0.178287, 0.178649, 0.178283, 0.177172, 0.179323, 0.179437, 0.179172, 0.177339]
FFI mixed kernel           median=3.2736 samples=[3.3707979999999997, 3.449509, 3.2453629999999998, 3.262647, 3.19111, 3.262652, 3.287703, 3.350644, 3.273623]
INTJ mixed kernel          median=2.9170 samples=[3.029411, 3.032156, 2.917025, 2.924086, 2.871322, 2.874345, 2.8616129999999997, 2.976806, 2.867489]
INTJ fixed mixed kernel    median=2.9515 samples=[3.08004, 3.067869, 2.974286, 2.970969, 2.926369, 2.8705439999999998, 2.910214, 2.928164, 2.951535]
elapsed_ns=3620066460
exit_status=0
ended=2026-09-25T16:35:37+08:00
```

### ffi_sweep: final merged, round 1

```text
commit=03a8f52651248cb333ed50347fbabac6bd15103a
started=2026-09-25T16:35:37+08:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/torch_abi
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/torch_abi taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_ffi_compare.py --sweep --iters 1000 --batches 9
counts=[0, 3, 5, 8, 16, 32, 64] iters=1000 batches=9 unit=us/call (host-only, no timed sync); INTJ device-bound, no specialization key
args= 0 FFI packed nop           median=0.1178 samples=[0.122671, 0.117753, 0.116163, 0.12061, 0.114343, 0.12025100000000001, 0.115694, 0.118019, 0.114648]
args= 0 FFI typed nop            median=0.1203 samples=[0.119461, 0.122984, 0.12028900000000001, 0.122117, 0.117143, 0.122083, 0.11679, 0.12209300000000001, 0.11780800000000001]
args= 0 FFI empty kernel         median=1.6704 samples=[1.670431, 1.854571, 1.6376279999999999, 1.877031, 1.5977860000000002, 1.88395, 1.6059860000000001, 1.858815, 1.610996]
args= 0 INTJ static_compile kernel median=2.7480 samples=[3.0739989999999997, 2.7799699999999996, 2.761415, 2.7480300000000004, 2.749409, 2.707914, 2.7126750000000004, 2.6640279999999996, 2.704402]
args= 0 INTJ runtime_shim kernel median=2.7056 samples=[2.831417, 2.7281329999999997, 2.784456, 2.690585, 2.74868, 2.692692, 2.705581, 2.694285, 2.700212]
args= 3 FFI packed nop           median=0.1436 samples=[0.146477, 0.143377, 0.143589, 0.142741, 0.142898, 0.14283, 0.145548, 0.14514, 0.143594]
args= 3 FFI typed nop            median=0.1485 samples=[0.149851, 0.148126, 0.146494, 0.150804, 0.147317, 0.153958, 0.14841100000000002, 0.15255000000000002, 0.148476]
args= 3 FFI empty kernel         median=3.2293 samples=[3.3021909999999997, 3.229323, 3.231922, 3.314701, 3.22119, 3.2369899999999996, 3.163114, 3.225364, 3.164552]
args= 3 INTJ static_compile kernel median=2.9280 samples=[2.944077, 2.902533, 2.928047, 2.937522, 2.8839789999999996, 2.890569, 2.951171, 2.8944319999999997, 2.932142]
args= 3 INTJ runtime_shim kernel median=2.8900 samples=[2.914676, 2.899335, 2.943012, 2.804056, 2.8899529999999998, 2.876346, 2.797821, 2.92258, 2.7870369999999998]
args= 5 FFI packed nop           median=0.1735 samples=[0.175074, 0.17505400000000002, 0.174256, 0.173536, 0.172013, 0.17086199999999999, 0.174167, 0.16987100000000002, 0.171975]
args= 5 FFI typed nop            median=0.1769 samples=[0.176217, 0.178876, 0.176882, 0.19405799999999998, 0.172868, 0.17704499999999998, 0.175035, 0.18791300000000002, 0.17463]
args= 5 FFI empty kernel         median=3.2794 samples=[3.30562, 3.29835, 3.271747, 3.286975, 3.277025, 3.32218, 3.185241, 3.279391, 3.1956170000000004]
args= 5 INTJ static_compile kernel median=2.9001 samples=[2.8887680000000002, 2.9535489999999998, 2.935079, 2.90011, 2.895317, 2.92737, 2.878053, 2.893018, 2.9286179999999997]
args= 5 INTJ runtime_shim kernel median=2.8892 samples=[2.998924, 3.031946, 2.889154, 3.03501, 2.867206, 2.77665, 2.839901, 2.883731, 2.8937530000000002]
args= 8 FFI packed nop           median=0.2079 samples=[0.20844100000000002, 0.208318, 0.20466800000000002, 0.205225, 0.205764, 0.207945, 0.210096, 0.203155, 0.208094]
args= 8 FFI typed nop            median=0.2086 samples=[0.205824, 0.208581, 0.20601, 0.21087999999999998, 0.208224, 0.211254, 0.209643, 0.208999, 0.208063]
args= 8 FFI empty kernel         median=3.3799 samples=[3.485763, 3.495498, 3.324326, 3.405681, 3.296384, 3.394227, 3.287947, 3.379917, 3.338032]
args= 8 INTJ static_compile kernel median=3.0411 samples=[3.04454, 3.041072, 3.135004, 3.0423470000000004, 3.075781, 3.038752, 3.032273, 3.0221109999999998, 3.0355090000000002]
args= 8 INTJ runtime_shim kernel median=2.9501 samples=[3.045171, 3.0202370000000003, 2.979607, 2.9269499999999997, 2.880544, 2.89336, 2.95006, 2.884369, 2.954731]
args=16 FFI packed nop           median=0.2942 samples=[0.289635, 0.292097, 0.311736, 0.290106, 0.294234, 0.296311, 0.298519, 0.292558, 0.294499]
args=16 FFI typed nop            median=0.3041 samples=[0.30350499999999997, 0.307004, 0.29836599999999996, 0.30405099999999996, 0.3038, 0.305421, 0.304084, 0.30549400000000004, 0.302595]
args=16 FFI empty kernel         median=3.6845 samples=[3.6937640000000003, 3.762165, 3.595409, 3.7113449999999997, 3.557709, 3.691798, 3.558476, 3.6845320000000004, 3.530426]
args=16 INTJ static_compile kernel median=3.3543 samples=[3.221295, 3.402474, 3.573811, 3.33211, 3.354251, 3.338937, 3.4910210000000004, 3.314538, 3.47544]
args=16 INTJ runtime_shim kernel median=3.2253 samples=[3.2253000000000003, 3.181922, 3.370947, 3.101676, 3.312794, 3.048433, 3.388147, 3.056268, 3.345364]
args=32 FFI packed nop           median=0.4799 samples=[0.46912400000000004, 0.47775599999999996, 0.48976400000000003, 0.479914, 0.480546, 0.483192, 0.480952, 0.476629, 0.479825]
args=32 FFI typed nop            median=0.4968 samples=[0.491873, 0.498641, 0.5025729999999999, 0.496822, 0.494369, 0.5060899999999999, 0.490585, 0.501528, 0.491767]
args=32 FFI empty kernel         median=4.3081 samples=[4.308082000000001, 4.338556, 4.230322, 4.324598, 4.195060000000001, 4.316217, 4.202640000000001, 4.312104000000001, 4.124249]
args=32 INTJ static_compile kernel median=3.6560 samples=[3.443024, 3.688729, 4.771941, 3.604555, 3.816118, 3.582603, 3.666642, 3.552128, 3.656024]
args=32 INTJ runtime_shim kernel median=3.5755 samples=[3.441094, 3.519485, 3.653016, 3.4906289999999998, 3.741426, 3.450127, 3.6295520000000003, 3.662164, 3.575516]
args=64 FFI packed nop           median=0.8348 samples=[0.8210810000000001, 0.823318, 0.814028, 0.834808, 0.86699, 0.8502240000000001, 0.817548, 0.859423, 0.8419059999999999]
args=64 FFI typed nop            median=0.8667 samples=[0.872614, 0.884066, 0.850289, 0.859373, 0.8667050000000001, 0.8617279999999999, 0.845502, 0.8905339999999999, 0.8738790000000001]
args=64 FFI empty kernel         median=5.0860 samples=[5.063799, 5.129376000000001, 5.12497, 5.094758, 5.063180999999999, 5.066297, 5.092256, 5.071621, 5.085989]
args=64 INTJ static_compile kernel median=4.6260 samples=[4.85461, 4.6775839999999995, 4.674652, 4.633045, 4.623429, 4.625975, 4.600558, 4.611429, 4.591495]
args=64 INTJ runtime_shim kernel median=4.6262 samples=[4.701984, 4.726311, 4.626162999999999, 4.6810730000000005, 4.566447, 4.61385, 4.621911, 4.643382, 4.550915]
elapsed_ns=4204404500
exit_status=0
ended=2026-09-25T16:35:41+08:00
```

### controlled: final merged, round 1

```text
commit=03a8f52651248cb333ed50347fbabac6bd15103a
started=2026-09-25T16:35:41+08:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/torch_abi
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/torch_abi taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_ffi_compare.py --sweep --iters 100 --batches 9
counts=[0, 3, 5, 8, 16, 32, 64] iters=100 batches=9 unit=us/call (host-only, no timed sync); INTJ device-bound, no specialization key
args= 0 FFI packed nop           median=0.1166 samples=[0.12585, 0.11702, 0.11529, 0.1166, 0.11562, 0.11684, 0.11668, 0.11452, 0.11522]
args= 0 FFI typed nop            median=0.1184 samples=[0.12323, 0.11929000000000001, 0.11706, 0.1184, 0.11777, 0.12068999999999999, 0.11809, 0.12148, 0.11709]
args= 0 FFI empty kernel         median=1.7819 samples=[1.7818900000000002, 1.97175, 1.98066, 1.8933, 1.85405, 1.7621600000000002, 1.76887, 1.77772, 1.7508]
args= 0 INTJ static_compile kernel median=2.9153 samples=[3.01757, 3.0071399999999997, 3.21889, 2.8105599999999997, 2.94252, 2.9153000000000002, 2.73048, 2.8555, 2.88345]
args= 0 INTJ runtime_shim kernel median=3.0193 samples=[3.18223, 3.05207, 3.0193499999999998, 3.08221, 3.0371300000000003, 2.82937, 2.86328, 2.9171799999999997, 2.82224]
args= 3 FFI packed nop           median=0.1412 samples=[0.14381, 0.1404, 0.14085, 0.14098, 0.13904, 0.14123, 0.14127, 0.14358, 0.144]
args= 3 FFI typed nop            median=0.1460 samples=[0.14598, 0.1494, 0.14307, 0.15092, 0.14603, 0.14325, 0.14408, 0.14971, 0.14577]
args= 3 FFI empty kernel         median=3.5025 samples=[3.7461100000000003, 3.5323, 3.5246, 3.50251, 3.37921, 3.32708, 3.3295100000000004, 3.5852199999999996, 3.4591999999999996]
args= 3 INTJ static_compile kernel median=3.0975 samples=[3.20919, 3.3024, 3.21288, 3.32598, 3.0858499999999998, 3.03679, 3.03392, 3.09747, 3.04482]
args= 3 INTJ runtime_shim kernel median=3.1306 samples=[3.26322, 3.33428, 3.34742, 3.2329899999999996, 3.13062, 3.0337899999999998, 3.05794, 3.10508, 3.05919]
args= 5 FFI packed nop           median=0.1698 samples=[0.17223, 0.16976, 0.1704, 0.17123, 0.17052, 0.16956, 0.16974, 0.16665, 0.16948]
args= 5 FFI typed nop            median=0.1740 samples=[0.174, 0.17708000000000002, 0.17165, 0.17708000000000002, 0.17475, 0.17456, 0.17331, 0.17385, 0.1725]
args= 5 FFI empty kernel         median=3.4287 samples=[3.5215300000000003, 3.51127, 3.63362, 3.60589, 3.42865, 3.40977, 3.3417, 3.35333, 3.3188299999999997]
args= 5 INTJ static_compile kernel median=3.0900 samples=[3.09004, 3.18813, 3.22661, 3.2046799999999998, 3.0742, 2.9739400000000002, 3.02481, 3.15814, 3.04004]
args= 5 INTJ runtime_shim kernel median=3.1282 samples=[3.15898, 3.12824, 3.2535, 3.21659, 3.12821, 2.99891, 2.96643, 2.95107, 3.0683800000000003]
args= 8 FFI packed nop           median=0.2029 samples=[0.20292, 0.20195, 0.20221, 0.20203, 0.20171, 0.20457, 0.20305, 0.20343, 0.20444]
args= 8 FFI typed nop            median=0.2064 samples=[0.20548, 0.20975, 0.208, 0.2047, 0.20235, 0.20690999999999998, 0.2064, 0.20662, 0.20575]
args= 8 FFI empty kernel         median=3.5655 samples=[3.66056, 3.6956599999999997, 3.71773, 3.67109, 3.48159, 3.38683, 3.5163699999999998, 3.56548, 3.42136]
args= 8 INTJ static_compile kernel median=3.1475 samples=[3.15423, 3.3229, 3.26358, 3.26408, 3.14749, 3.11989, 2.96944, 3.11286, 3.0635100000000004]
args= 8 INTJ runtime_shim kernel median=3.1107 samples=[3.1478200000000003, 3.2236599999999997, 3.23936, 3.23009, 3.09086, 2.99231, 3.06013, 3.11074, 3.02136]
args=16 FFI packed nop           median=0.2956 samples=[0.2932, 0.29464999999999997, 0.29481, 0.29556, 0.2933, 0.31444, 0.29666000000000003, 0.29597, 0.29868]
args=16 FFI typed nop            median=0.3030 samples=[0.29869999999999997, 0.30488, 0.29958999999999997, 0.31283, 0.30034, 0.30302999999999997, 0.30263999999999996, 0.30473, 0.30372000000000005]
args=16 FFI empty kernel         median=3.9437 samples=[3.8838600000000003, 3.8667800000000003, 4.35704, 3.9436999999999998, 4.224069999999999, 3.8406100000000003, 4.0739600000000005, 3.88312, 4.15654]
args=16 INTJ static_compile kernel median=3.5267 samples=[3.76176, 3.48935, 3.71969, 3.62982, 3.6798, 3.3415100000000004, 3.5267399999999998, 3.32895, 3.50776]
args=16 INTJ runtime_shim kernel median=3.5233 samples=[3.5567399999999996, 3.64327, 3.61655, 3.4987600000000003, 3.40159, 3.52329, 3.33656, 3.54508, 3.3977]
args=32 FFI packed nop           median=0.4881 samples=[0.48089, 0.51374, 0.48904000000000003, 0.4899, 0.48813, 0.48929, 0.48172000000000004, 0.48797, 0.47883]
args=32 FFI typed nop            median=0.5006 samples=[0.49795999999999996, 0.5310199999999999, 0.5002300000000001, 0.50488, 0.50278, 0.50426, 0.48926, 0.50059, 0.49497]
args=32 FFI empty kernel         median=4.2846 samples=[4.52973, 4.36533, 4.40632, 4.5361400000000005, 4.28455, 4.23157, 4.20076, 4.27632, 4.17519]
args=32 INTJ static_compile kernel median=3.7846 samples=[4.08239, 3.7845500000000003, 3.9660100000000003, 3.95961, 3.7850900000000003, 3.62557, 3.78112, 3.60732, 3.74892]
args=32 INTJ runtime_shim kernel median=3.7288 samples=[3.7087, 3.8756, 3.8483400000000003, 4.00208, 3.72877, 3.71474, 3.67596, 3.7734699999999997, 3.7137800000000003]
args=64 FFI packed nop           median=0.8594 samples=[0.8100499999999999, 0.8249, 0.8734500000000001, 0.86268, 0.85937, 0.8683500000000001, 0.8547, 0.8658899999999999, 0.8370599999999999]
args=64 FFI typed nop            median=0.8835 samples=[0.90164, 0.8682799999999999, 0.85634, 0.88654, 0.8819600000000001, 0.88355, 0.88871, 0.88428, 0.87954]
args=64 FFI empty kernel         median=5.2885 samples=[5.354430000000001, 5.29516, 5.3842, 5.42838, 5.246779999999999, 5.2884899999999995, 5.20287, 5.22634, 5.23978]
args=64 INTJ static_compile kernel median=4.7236 samples=[4.934279999999999, 4.86823, 4.91595, 4.961399999999999, 4.72361, 4.66832, 4.67189, 4.68182, 4.665760000000001]
args=64 INTJ runtime_shim kernel median=4.6870 samples=[4.97448, 4.94509, 4.89795, 4.88925, 4.687, 4.65838, 4.6517100000000005, 4.65365, 4.62509]
elapsed_ns=3555004974
exit_status=0
ended=2026-09-25T16:35:44+08:00
```

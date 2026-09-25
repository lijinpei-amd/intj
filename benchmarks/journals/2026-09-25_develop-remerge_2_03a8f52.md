# Latest develop remerge: measurement round 2

Baseline `ef698d2`; intermediate merged `2a9f891`, final merged `03a8f52`. Same environment, commands, and caveats as [round 0](2026-09-25_develop-remerge_0_03a8f52.md).

## Raw outputs (including batch samples and run metadata)

### host: develop

```text
commit=ef698d24dae8790fffdf84fe766a7bf06db428f6
started=2026-09-25T16:24:29+08:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --no-gpu --iters 100000 --batches 9
mode=host-only; 100000 calls × 9 batches; median ns/call
               path    ns/call       Δ ns       Δ %   build ms   bind ms
           auto map       43.6       +0.0      +0.0      59.25         -
        reduced key       43.1       -0.5      -1.1       1.34         -
         verify off       43.2       -0.4      -0.9       1.20         -
          verify on       44.4       +0.8      +1.8       1.15         -
              baked       40.0       -3.6      -8.2       8.49         -
       bound tensor       45.3       +1.8      +4.0       0.14      1.18
      bound pointer       47.0       +3.4      +7.8       0.11      1.09
   fixed device map       43.8       +0.2      +0.4       0.09      1.10
fixed device no-map       45.4       +1.8      +4.2       0.12      1.05
elapsed_ns=3019516709
exit_status=0
ended=2026-09-25T16:24:32+08:00
```

### host: merged

```text
commit=2a9f891443cf6763d26483c7b1f3d0f451d90a3c
started=2026-09-25T16:27:09+08:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/torch_abi
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/torch_abi taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --no-gpu --iters 100000 --batches 9
mode=host-only; 100000 calls × 9 batches; median ns/call
               path    ns/call       Δ ns       Δ %   build ms   bind ms
           auto map       43.9       +0.0      +0.0      59.30         -
        reduced key       43.4       -0.4      -1.0       1.54         -
         verify off       43.5       -0.4      -0.8       1.36         -
          verify on       44.4       +0.5      +1.2       1.36         -
              baked       40.0       -3.9      -8.9       2.57         -
       bound tensor       46.5       +2.6      +6.0       0.15      1.29
      bound pointer       45.1       +1.3      +2.9       0.13      1.25
   fixed device map       44.4       +0.5      +1.2       0.10      1.22
fixed device no-map       43.5       -0.4      -0.8       0.14      1.26
elapsed_ns=2856442218
exit_status=0
ended=2026-09-25T16:27:12+08:00
```

### sweep: develop

```text
commit=ef698d24dae8790fffdf84fe766a7bf06db428f6
started=2026-09-25T16:24:32+08:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --sweep --no-gpu --iters 100000 --batches 9
mode=sweep; host-only; 100000 calls × 9 batches; median ns/call
count   kind    ns/call
    4    int       41.6
    4 tensor       40.5
   16    int       74.6
   16 tensor       67.9
   32    int      117.9
   32 tensor      119.5
elapsed_ns=3061773674
exit_status=0
ended=2026-09-25T16:24:35+08:00
```

### sweep: merged

```text
commit=2a9f891443cf6763d26483c7b1f3d0f451d90a3c
started=2026-09-25T16:27:12+08:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/torch_abi
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/torch_abi taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --sweep --no-gpu --iters 100000 --batches 9
mode=sweep; host-only; 100000 calls × 9 batches; median ns/call
count   kind    ns/call
    4    int       42.5
    4 tensor       41.7
   16    int       73.8
   16 tensor       68.6
   32    int      120.7
   32 tensor      115.2
elapsed_ns=2910905616
exit_status=0
ended=2026-09-25T16:27:14+08:00
```

### readme: develop

```text
commit=ef698d24dae8790fffdf84fe766a7bf06db428f6
started=2026-09-25T16:24:35+08:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --readme --iters 1000 --batches 9
mode=readme; 1000 calls × 9 batches; median
        path   triton us    intj us   speedup
   grid=(1,)       16.42       3.12      5.3x
   grid=(0,)       12.99       0.04    294.2x

torch_access   decode ns    build s
        shim        90.9       0.02
         cxx       114.1       0.06
     cpython       872.8       0.00
elapsed_ns=3837108117
exit_status=0
ended=2026-09-25T16:24:39+08:00
```

### readme: merged

```text
commit=2a9f891443cf6763d26483c7b1f3d0f451d90a3c
started=2026-09-25T16:27:14+08:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/torch_abi
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/torch_abi taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --readme --iters 1000 --batches 9
mode=readme; 1000 calls × 9 batches; median
        path   triton us    intj us   speedup
   grid=(1,)       16.55       3.13      5.3x
   grid=(0,)       13.22       0.03    463.4x

torch_access_mode   decode ns    build s
     runtime_shim        92.5       0.02
   static_compile        93.9       0.06
      interpreter       872.5       0.00
elapsed_ns=3646115717
exit_status=0
ended=2026-09-25T16:27:18+08:00
```

### hip: develop

```text
commit=ef698d24dae8790fffdf84fe766a7bf06db428f6
started=2026-09-25T16:24:39+08:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_hip_module_launch.py --iters 1000 --batches 9
hsaco_sha256=df3e529e178196ad62f7c07ab8593ede5b8596c76e7c6fbd12a5d947a696e65c kernel=empty_kernel threads=256 shared=0 callback_count=1
iters=1000 batches=9 unit=us/call (host enqueue, no timed sync)
TVM FFI HIP HSACO         median=3.1663 samples=[3.540209, 3.1797220000000004, 3.180219, 3.141524, 3.1662820000000003, 3.12446, 3.186689, 3.072584, 3.06493]
Triton same HSACO         median=15.5878 samples=[15.758025, 15.722541, 15.610083000000001, 15.597206, 15.587831, 15.569389, 15.572231, 15.532442999999999, 15.496656999999999]
INTJ same function        median=2.9558 samples=[2.992884, 2.964839, 2.974913, 2.967415, 2.9558449999999996, 2.882621, 2.887252, 2.894729, 2.831578]
elapsed_ns=3674189150
exit_status=0
ended=2026-09-25T16:24:42+08:00
```

### hip: merged

```text
commit=2a9f891443cf6763d26483c7b1f3d0f451d90a3c
started=2026-09-25T16:27:18+08:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/torch_abi
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/torch_abi taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_hip_module_launch.py --iters 1000 --batches 9
hsaco_sha256=df3e529e178196ad62f7c07ab8593ede5b8596c76e7c6fbd12a5d947a696e65c kernel=empty_kernel threads=256 shared=0 callback_count=1
iters=1000 batches=9 unit=us/call (host enqueue, no timed sync)
TVM FFI HIP HSACO         median=3.1991 samples=[3.338937, 3.2560279999999997, 3.252087, 3.1994140000000004, 3.1990700000000003, 3.125018, 3.1359760000000003, 3.095753, 3.104342]
Triton same HSACO         median=15.7782 samples=[15.971542, 16.072406, 16.174296000000002, 15.905576, 15.74262, 15.778213, 15.713954, 15.711468, 15.706562]
INTJ same function        median=2.9899 samples=[3.131841, 3.1075459999999997, 3.056192, 3.039391, 2.989859, 2.909398, 2.941212, 2.9384650000000003, 2.8853649999999997]
elapsed_ns=3602097095
exit_status=0
ended=2026-09-25T16:27:22+08:00
```

### ffi_paths: develop

```text
commit=ef698d24dae8790fffdf84fe766a7bf06db428f6
started=2026-09-25T16:24:42+08:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_intj_ffi_paths.py --iters 1000 --batches 9
iters=1000 batches=9 unit=ns/call; no_gpu=True
mode=callback
callback_count=9100 (100 warmup + 9000 unique misses)
INTJ cold compile callback + cache  median_ns=418.2 samples_ns=[457.254, 451.564, 400.79, 548.541, 386.385, 395.395, 410.54, 418.229, 1431.478]
INTJ hot call (no callback)         median_ns=38.1 samples_ns=[42.182, 44.15, 38.051, 38.918, 37.522, 38.301, 37.792, 37.869, 37.224]
INTJ 3-tensor host-only nop         median_ns=35.1 samples_ns=[36.574, 35.371, 34.977, 35.24, 34.475, 35.18, 34.791, 35.091, 34.456]
mode=kwargs
INTJ direct positional              median_ns=66.2 samples_ns=[68.725, 66.481, 66.279, 66.31, 66.238, 66.229, 66.19, 66.179, 66.178]
INTJ adapter positional             median_ns=89.7 samples_ns=[90.956, 91.235, 91.513, 93.392, 89.677, 89.539, 89.455, 89.28, 89.446]
INTJ adapter kwargs                 median_ns=107.1 samples_ns=[107.862, 107.958, 106.666, 106.21, 109.501, 106.446, 107.702, 107.103, 106.66]
INTJ adapter defaults               median_ns=90.3 samples_ns=[89.657, 90.291, 89.508, 88.205, 88.151, 102.945, 91.919, 92.631, 91.917]
INTJ FFI wrapper positional         median_ns=107.2 samples_ns=[107.214, 108.517, 104.954, 106.17, 106.331, 109.315, 107.225, 105.726, 108.435]
INTJ FFI wrapper kwargs             median_ns=129.4 samples_ns=[129.44, 128.398, 127.77, 127.816, 127.411, 145.037, 131.148, 129.549, 130.712]
INTJ native kwargs: unsupported (verified TypeError)
mode=dataclass
INTJ pair direct                    median_ns=67.9 samples_ns=[69.249, 68.356, 67.422, 73.329, 67.345, 67.885, 67.592, 67.886, 67.379]
INTJ pair prebuilt *tuple           median_ns=141.3 samples_ns=[139.931, 142.126, 145.403, 141.34, 139.614, 142.484, 139.046, 143.057, 141.143]
FFI unpack Pair only                median_ns=163.0 samples_ns=[161.595, 163.108, 161.196, 162.228, 162.312, 163.739, 169.407, 164.449, 162.955]
INTJ pair manual unpack             median_ns=173.9 samples_ns=[172.023, 173.931, 174.709, 174.843, 172.34, 174.608, 171.872, 172.995, 179.956]
INTJ pair FFI unpack                median_ns=317.5 samples_ns=[313.541, 316.882, 317.464, 316.932, 315.39, 323.056, 321.413, 320.877, 317.604]
INTJ pair stdlib astuple            median_ns=1158.2 samples_ns=[1315.659, 1215.72, 1146.926, 1172.368, 1155.25, 1158.249, 1156.277, 1161.6, 1151.781]
INTJ config direct                  median_ns=83.0 samples_ns=[82.091, 82.839, 85.817, 84.077, 82.939, 83.863, 82.989, 83.611, 82.867]
INTJ config FFI unpack              median_ns=327.3 samples_ns=[325.252, 332.804, 324.697, 327.782, 332.725, 327.163, 324.957, 341.553, 327.324]
INTJ native dataclass unpack: unsupported (verified TypeError)
elapsed_ns=2921113595
exit_status=0
ended=2026-09-25T16:24:45+08:00
```

### ffi_paths: merged

```text
commit=2a9f891443cf6763d26483c7b1f3d0f451d90a3c
started=2026-09-25T16:27:22+08:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/torch_abi
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/torch_abi taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_intj_ffi_paths.py --iters 1000 --batches 9
iters=1000 batches=9 unit=ns/call; no_gpu=True
mode=callback
callback_count=9100 (100 warmup + 9000 unique misses)
INTJ cold compile callback + cache  median_ns=420.6 samples_ns=[460.852, 452.981, 398.459, 572.107, 378.775, 405.391, 407.017, 420.611, 1483.08]
INTJ hot call (no callback)         median_ns=40.9 samples_ns=[41.806, 41.318, 39.538, 40.715, 39.661, 40.876, 39.583, 42.562, 41.112]
INTJ 3-tensor host-only nop         median_ns=35.2 samples_ns=[36.992, 35.63, 34.965, 36.239, 34.866, 35.209, 34.748, 35.206, 34.656]
mode=kwargs
INTJ direct positional              median_ns=66.9 samples_ns=[68.843, 67.426, 66.913, 66.833, 66.934, 66.856, 66.877, 66.898, 66.922]
INTJ adapter positional             median_ns=89.8 samples_ns=[90.753, 90.456, 102.663, 90.258, 89.76, 89.084, 89.156, 89.453, 89.376]
INTJ adapter kwargs                 median_ns=105.9 samples_ns=[110.086, 105.159, 105.308, 109.373, 106.151, 105.916, 106.193, 105.845, 105.847]
INTJ adapter defaults               median_ns=89.4 samples_ns=[88.269, 90.109, 89.889, 88.032, 91.628, 89.37, 89.54, 89.189, 89.442]
INTJ FFI wrapper positional         median_ns=104.6 samples_ns=[104.579, 102.994, 102.665, 102.148, 102.788, 115.421, 108.233, 108.265, 107.85]
INTJ FFI wrapper kwargs             median_ns=127.3 samples_ns=[128.292, 128.86, 128.852, 126.365, 133.598, 127.296, 126.543, 125.755, 124.326]
INTJ native kwargs: unsupported (verified TypeError)
mode=dataclass
INTJ pair direct                    median_ns=71.5 samples_ns=[72.743, 72.11, 70.084, 71.952, 71.522, 70.828, 69.817, 70.613, 73.077]
INTJ pair prebuilt *tuple           median_ns=142.2 samples_ns=[140.055, 142.484, 139.487, 142.588, 140.815, 143.702, 142.642, 140.466, 142.186]
FFI unpack Pair only                median_ns=163.4 samples_ns=[164.999, 165.37, 162.114, 164.955, 168.363, 163.443, 160.736, 162.328, 161.708]
INTJ pair manual unpack             median_ns=172.6 samples_ns=[173.327, 173.145, 169.979, 174.012, 171.581, 172.636, 172.145, 173.016, 170.312]
INTJ pair FFI unpack                median_ns=316.8 samples_ns=[309.26, 316.125, 311.72, 316.893, 312.669, 321.628, 316.823, 317.677, 318.053]
INTJ pair stdlib astuple            median_ns=1174.1 samples_ns=[1342.836, 1208.594, 1162.274, 1191.916, 1164.738, 1174.129, 1177.652, 1170.982, 1158.884]
INTJ config direct                  median_ns=84.0 samples_ns=[84.096, 83.714, 85.222, 85.301, 84.545, 84.011, 83.043, 84.023, 83.164]
INTJ config FFI unpack              median_ns=322.1 samples_ns=[321.169, 320.649, 318.422, 324.805, 320.644, 322.115, 325.742, 325.214, 322.611]
INTJ native dataclass unpack: unsupported (verified TypeError)
elapsed_ns=2782446339
exit_status=0
ended=2026-09-25T16:27:25+08:00
```

### ffi_default: develop

```text
commit=ef698d24dae8790fffdf84fe766a7bf06db428f6
started=2026-09-25T16:24:45+08:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_ffi_compare.py --iters 1000 --batches 9
iters=1000 batches=9 unit=us/call (host-only, no timed sync)
FFI packed nop             median=0.1446 samples=[0.14912299999999998, 0.14630500000000002, 0.145777, 0.143434, 0.14488800000000002, 0.14342500000000002, 0.144369, 0.144583, 0.14243199999999998]
FFI typed nop              median=0.1472 samples=[0.14871199999999998, 0.146975, 0.146929, 0.149954, 0.146218, 0.150272, 0.146837, 0.150362, 0.147171]
FFI empty kernel           median=3.3283 samples=[3.590012, 3.472242, 3.4725949999999997, 3.269483, 3.16821, 3.342899, 3.201179, 3.328274, 3.201451]
INTJ empty kernel          median=3.0491 samples=[3.065171, 3.148287, 3.153, 2.974638, 3.005208, 3.0334659999999998, 3.0875019999999997, 3.023017, 3.049107]
INTJ fixed-device kernel   median=2.8582 samples=[2.9916729999999996, 3.107218, 3.072024, 2.858226, 2.790763, 2.819465, 2.819091, 2.8180549999999998, 2.924623]
FFI packed nop mixed       median=0.1736 samples=[0.17236400000000002, 0.169827, 0.17262799999999998, 0.174715, 0.174854, 0.173613, 0.177611, 0.17307499999999998, 0.177653]
FFI typed nop mixed        median=0.1756 samples=[0.17410499999999998, 0.17555199999999999, 0.17322800000000002, 0.178232, 0.17458500000000002, 0.176573, 0.174287, 0.176643, 0.177218]
FFI mixed kernel           median=3.3496 samples=[3.4471060000000002, 3.5096950000000002, 3.373076, 3.349628, 3.190636, 3.318536, 3.254524, 3.369996, 3.26371]
INTJ mixed kernel          median=2.9134 samples=[3.105534, 3.116589, 2.9323710000000003, 2.9475, 2.904093, 2.849115, 2.8512779999999998, 2.913373, 2.8459090000000002]
INTJ fixed mixed kernel    median=2.9122 samples=[3.124559, 3.1239679999999996, 2.9875700000000003, 2.993638, 2.8498580000000002, 2.819031, 2.90093, 2.912248, 2.872738]
elapsed_ns=3861460269
exit_status=0
ended=2026-09-25T16:24:49+08:00
```

### ffi_default: merged

```text
commit=2a9f891443cf6763d26483c7b1f3d0f451d90a3c
started=2026-09-25T16:27:25+08:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/torch_abi
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/torch_abi taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_ffi_compare.py --iters 1000 --batches 9
iters=1000 batches=9 unit=us/call (host-only, no timed sync)
FFI packed nop             median=0.1448 samples=[0.152558, 0.144478, 0.144476, 0.14518799999999998, 0.144758, 0.14500200000000002, 0.145454, 0.14425100000000002, 0.144625]
FFI typed nop              median=0.1498 samples=[0.149818, 0.15105600000000002, 0.151921, 0.149076, 0.14803899999999998, 0.151482, 0.148315, 0.152699, 0.149534]
FFI empty kernel           median=3.3252 samples=[3.592556, 3.373769, 3.381475, 3.315267, 3.259891, 3.3251709999999997, 3.186366, 3.341334, 3.193452]
INTJ empty kernel          median=2.9582 samples=[3.078117, 3.074938, 3.077861, 2.926615, 2.864637, 2.964003, 2.958243, 2.9198310000000003, 2.9573989999999997]
INTJ fixed-device kernel   median=2.9044 samples=[2.9654879999999997, 3.067957, 3.066402, 2.817747, 2.845404, 2.825484, 2.904367, 2.821764, 2.924484]
FFI packed nop mixed       median=0.1734 samples=[0.173369, 0.173181, 0.174191, 0.172055, 0.17429499999999998, 0.173557, 0.174476, 0.17176, 0.172704]
FFI typed nop mixed        median=0.1763 samples=[0.178868, 0.17653899999999997, 0.176096, 0.17677400000000001, 0.176178, 0.176287, 0.17419800000000002, 0.17675200000000002, 0.175331]
FFI mixed kernel           median=3.2747 samples=[3.379527, 3.427889, 3.27467, 3.257012, 3.257321, 3.3189520000000003, 3.261241, 3.348651, 3.245863]
INTJ mixed kernel          median=2.8981 samples=[3.117563, 3.08221, 2.898092, 2.909489, 2.844678, 2.86522, 2.851187, 2.9945549999999996, 2.8333000000000004]
INTJ fixed mixed kernel    median=2.9223 samples=[3.1264659999999997, 3.096542, 2.952247, 2.937989, 2.8715680000000003, 2.8414989999999998, 2.911217, 2.922289, 2.9152199999999997]
elapsed_ns=3613779933
exit_status=0
ended=2026-09-25T16:27:28+08:00
```

### ffi_sweep: develop

```text
commit=ef698d24dae8790fffdf84fe766a7bf06db428f6
started=2026-09-25T16:24:49+08:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_ffi_compare.py --sweep --iters 1000 --batches 9
counts=[0, 3, 5, 8, 16, 32, 64] iters=1000 batches=9 unit=us/call (host-only, no timed sync); INTJ device-bound, no specialization key
args= 0 FFI packed nop           median=0.1153 samples=[0.11945099999999999, 0.116616, 0.116893, 0.117273, 0.114425, 0.115285, 0.114643, 0.114709, 0.11469]
args= 0 FFI typed nop            median=0.1186 samples=[0.13815, 0.11948099999999999, 0.118605, 0.121101, 0.117787, 0.11863, 0.11620699999999999, 0.120899, 0.11665]
args= 0 FFI empty kernel         median=1.6907 samples=[1.69069, 1.860732, 1.607747, 1.918843, 1.5899539999999999, 1.864338, 1.602152, 1.8763489999999998, 1.595043]
args= 0 INTJ cxx kernel          median=2.7004 samples=[3.05281, 2.745249, 2.7250569999999996, 2.7004479999999997, 2.6929659999999997, 2.6480509999999997, 2.6981460000000004, 2.704047, 2.69471]
args= 0 INTJ shim kernel         median=2.7038 samples=[2.847921, 2.676004, 2.727506, 2.6543609999999997, 2.703831, 2.664685, 2.704775, 2.6755169999999997, 2.703799]
args= 3 FFI packed nop           median=0.1447 samples=[0.144672, 0.14632599999999998, 0.14484200000000003, 0.144671, 0.144386, 0.14369200000000001, 0.143131, 0.14519300000000002, 0.146954]
args= 3 FFI typed nop            median=0.1488 samples=[0.148831, 0.14877, 0.14847200000000002, 0.152046, 0.144617, 0.14808000000000002, 0.146147, 0.15029599999999999, 0.14899199999999999]
args= 3 FFI empty kernel         median=3.2281 samples=[3.285175, 3.214685, 3.228067, 3.2323899999999997, 3.155357, 3.2851999999999997, 3.158461, 3.23915, 3.1514119999999997]
args= 3 INTJ cxx kernel          median=2.8337 samples=[2.968235, 2.810199, 2.833652, 2.8245, 2.9789630000000002, 2.832081, 2.9723, 2.832975, 2.979774]
args= 3 INTJ shim kernel         median=2.8480 samples=[2.942224, 2.822047, 2.8505030000000002, 2.817012, 2.776408, 2.8433040000000003, 2.8906199999999997, 2.8480309999999998, 2.88687]
args= 5 FFI packed nop           median=0.1766 samples=[0.175756, 0.176558, 0.17338599999999998, 0.174683, 0.178373, 0.17693899999999999, 0.17788900000000002, 0.17711500000000002, 0.173724]
args= 5 FFI typed nop            median=0.1785 samples=[0.17773, 0.183298, 0.183046, 0.17852600000000002, 0.17475100000000002, 0.18087, 0.176753, 0.178757, 0.17760800000000002]
args= 5 FFI empty kernel         median=3.3337 samples=[3.333712, 3.352462, 3.270691, 3.356469, 3.193299, 3.350661, 3.2953560000000004, 3.349263, 3.264221]
args= 5 INTJ cxx kernel          median=2.8749 samples=[3.025585, 2.899442, 3.126765, 2.8038380000000003, 2.832967, 2.8362089999999998, 2.874881, 2.85364, 2.892185]
args= 5 INTJ shim kernel         median=2.8488 samples=[3.0291439999999996, 2.9100569999999997, 2.8487910000000003, 2.822829, 2.823377, 2.8085, 2.880479, 2.8170509999999997, 2.860447]
args= 8 FFI packed nop           median=0.2058 samples=[0.205527, 0.207073, 0.20582499999999998, 0.213837, 0.209824, 0.20283099999999998, 0.206132, 0.20241800000000001, 0.20344900000000002]
args= 8 FFI typed nop            median=0.2093 samples=[0.207853, 0.211598, 0.213788, 0.214449, 0.20929499999999998, 0.209123, 0.206903, 0.212351, 0.20572900000000002]
args= 8 FFI empty kernel         median=3.3984 samples=[3.4577240000000002, 3.438557, 3.259854, 3.4099589999999997, 3.2570949999999996, 3.411827, 3.2890770000000003, 3.398351, 3.304357]
args= 8 INTJ cxx kernel          median=3.0484 samples=[3.0686799999999996, 2.958614, 3.10327, 3.177, 3.077083, 2.921847, 3.048388, 2.907683, 3.043049]
args= 8 INTJ shim kernel         median=2.9491 samples=[3.062838, 2.939678, 2.9691959999999997, 2.881002, 2.949117, 2.899283, 2.9598850000000003, 2.8811869999999997, 2.96234]
args=16 FFI packed nop           median=0.2962 samples=[0.29685700000000004, 0.29615499999999995, 0.300776, 0.294998, 0.297929, 0.289983, 0.298495, 0.29206099999999996, 0.290555]
args=16 FFI typed nop            median=0.3061 samples=[0.303377, 0.308967, 0.306063, 0.30426400000000003, 0.305178, 0.306594, 0.306744, 0.306307, 0.302968]
args=16 FFI empty kernel         median=3.6992 samples=[3.6992109999999996, 3.803385, 3.567551, 3.701321, 3.524593, 3.7343409999999997, 3.5209360000000003, 3.724179, 3.5299169999999997]
args=16 INTJ cxx kernel          median=3.4772 samples=[3.256315, 3.477248, 3.608225, 3.3896819999999996, 3.553625, 3.4158969999999997, 3.57769, 3.358129, 3.5794189999999997]
args=16 INTJ shim kernel         median=3.2376 samples=[3.237555, 3.167666, 3.464027, 3.0579340000000004, 3.4223090000000003, 3.041188, 3.430454, 3.04593, 3.376832]
args=32 FFI packed nop           median=0.4892 samples=[0.478948, 0.49094600000000005, 0.48667099999999996, 0.493953, 0.48924900000000004, 0.492834, 0.494369, 0.479591, 0.477472]
args=32 FFI typed nop            median=0.4906 samples=[0.48470100000000005, 0.5075149999999999, 0.48767099999999997, 0.503935, 0.489834, 0.502985, 0.49059800000000003, 0.494021, 0.49001100000000003]
args=32 FFI empty kernel         median=4.2587 samples=[4.290304, 4.3170910000000005, 4.119225, 4.258744, 4.109198, 4.288551999999999, 4.12333, 4.293784, 4.101542]
args=32 INTJ cxx kernel          median=3.6806 samples=[3.47146, 3.750522, 3.739128, 3.623967, 3.700897, 3.627141, 3.7047559999999997, 3.62275, 3.680648]
args=32 INTJ shim kernel         median=3.5336 samples=[3.417085, 3.5335520000000002, 3.673724, 3.4410160000000003, 3.647598, 3.437411, 3.643315, 3.438081, 3.619146]
args=64 FFI packed nop           median=0.8492 samples=[0.849164, 0.8770180000000001, 0.8672759999999999, 0.847146, 0.85022, 0.8656820000000001, 0.833214, 0.83599, 0.833132]
args=64 FFI typed nop            median=0.8767 samples=[0.876687, 0.8998769999999999, 0.92162, 0.879984, 0.8651760000000001, 0.894907, 0.868081, 0.8637100000000001, 0.856035]
args=64 FFI empty kernel         median=5.0514 samples=[5.036992000000001, 5.131379, 5.076435, 5.086615, 5.051395, 5.075332, 5.026183, 5.043063, 5.035137]
args=64 INTJ cxx kernel          median=4.6020 samples=[11.170966, 4.639829, 4.601615, 4.584328, 4.602043, 4.612477, 4.595438, 4.612099, 4.598906]
args=64 INTJ shim kernel         median=4.5844 samples=[10.24501, 4.685695, 4.574792, 4.584384, 4.549615, 4.625948999999999, 4.5762160000000005, 4.603702, 4.534246]
elapsed_ns=4364876621
exit_status=0
ended=2026-09-25T16:24:54+08:00
```

### ffi_sweep: merged

```text
commit=2a9f891443cf6763d26483c7b1f3d0f451d90a3c
started=2026-09-25T16:27:28+08:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/torch_abi
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/torch_abi taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_ffi_compare.py --sweep --iters 1000 --batches 9
counts=[0, 3, 5, 8, 16, 32, 64] iters=1000 batches=9 unit=us/call (host-only, no timed sync); INTJ device-bound, no specialization key
args= 0 FFI packed nop           median=0.1167 samples=[0.12172799999999999, 0.117698, 0.11631399999999999, 0.117013, 0.117335, 0.11655700000000001, 0.11622, 0.11489100000000001, 0.116727]
args= 0 FFI typed nop            median=0.1211 samples=[0.12298999999999999, 0.121099, 0.120442, 0.121575, 0.12012600000000001, 0.126848, 0.119561, 0.12275, 0.11782]
args= 0 FFI empty kernel         median=2.1272 samples=[1.692671, 3.101, 2.125989, 3.037842, 2.1192860000000002, 3.0269899999999996, 2.1272159999999998, 3.055589, 2.1121529999999997]
args= 0 INTJ static_compile kernel median=3.2940 samples=[3.32655, 3.302553, 3.3028299999999997, 3.331478, 3.276395, 3.2939789999999998, 3.2893429999999997, 3.274844, 3.277038]
args= 0 INTJ runtime_shim kernel median=3.1142 samples=[3.114179, 3.03351, 3.3074830000000004, 3.0347869999999997, 3.2962890000000002, 3.0489479999999998, 3.30153, 3.0456790000000002, 3.315373]
args= 3 FFI packed nop           median=0.1460 samples=[0.14616200000000001, 0.146263, 0.143565, 0.145968, 0.143787, 0.142557, 0.140541, 0.147751, 0.14794900000000002]
args= 3 FFI typed nop            median=0.1497 samples=[0.150698, 0.149729, 0.147093, 0.152155, 0.15340600000000001, 0.151018, 0.146612, 0.14963900000000002, 0.146572]
args= 3 FFI empty kernel         median=3.7102 samples=[3.6740169999999996, 3.815298, 3.705129, 3.8397669999999997, 3.710168, 4.217366, 3.7039, 3.806789, 3.677397]
args= 3 INTJ static_compile kernel median=3.4886 samples=[3.275422, 3.4885949999999997, 3.450781, 3.570537, 3.541721, 3.477898, 3.4791930000000004, 3.5408380000000004, 3.52867]
args= 3 INTJ runtime_shim kernel median=3.4863 samples=[3.281446, 3.293851, 3.492217, 3.4245140000000003, 3.5621129999999996, 3.165487, 3.486274, 3.8804819999999998, 3.515748]
args= 5 FFI packed nop           median=0.1735 samples=[0.174098, 0.17312799999999998, 0.175053, 0.177559, 0.172946, 0.17354, 0.17353, 0.17260599999999998, 0.17245500000000002]
args= 5 FFI typed nop            median=0.1765 samples=[0.182645, 0.17894, 0.17417, 0.195637, 0.1759, 0.176714, 0.175663, 0.17653, 0.17557599999999998]
args= 5 FFI empty kernel         median=3.7510 samples=[3.700935, 4.262703999999999, 3.741111, 4.2834650000000005, 3.751014, 4.275214, 3.7504340000000003, 4.324022, 3.735165]
args= 5 INTJ static_compile kernel median=3.5353 samples=[3.254625, 3.4930920000000003, 3.503924, 3.539529, 3.544784, 3.560993, 3.485405, 3.535295, 3.5479969999999996]
args= 5 INTJ runtime_shim kernel median=3.4320 samples=[3.2740929999999997, 3.211818, 3.497017, 3.419013, 3.555109, 3.432043, 3.501911, 3.386662, 3.5306640000000002]
args= 8 FFI packed nop           median=0.2065 samples=[0.206484, 0.206007, 0.209885, 0.20605400000000001, 0.206632, 0.207151, 0.205614, 0.204663, 0.207753]
args= 8 FFI typed nop            median=0.2099 samples=[0.20941900000000002, 0.209875, 0.206924, 0.209098, 0.21287799999999998, 0.21273599999999998, 0.207678, 0.21304599999999999, 0.214082]
args= 8 FFI empty kernel         median=3.8413 samples=[3.827672, 4.431214, 3.773846, 4.380018, 3.8412710000000003, 4.3628670000000005, 3.762942, 4.362157, 3.816633]
args= 8 INTJ static_compile kernel median=3.6764 samples=[3.393596, 3.6132649999999997, 4.094204, 3.676447, 3.5748789999999997, 3.691291, 4.248600000000001, 3.709051, 3.573362]
args= 8 INTJ runtime_shim kernel median=3.3642 samples=[3.3641680000000003, 3.2847370000000002, 3.51258, 3.273383, 3.614114, 3.2663870000000004, 3.539109, 3.282273, 3.617165]
args=16 FFI packed nop           median=0.2950 samples=[0.29959399999999997, 0.29457799999999995, 0.291045, 0.294959, 0.294639, 0.290733, 0.29639, 0.301197, 0.297769]
args=16 FFI typed nop            median=0.3033 samples=[0.298269, 0.310711, 0.302959, 0.303279, 0.299484, 0.303213, 0.30485399999999996, 0.306705, 0.306152]
args=16 FFI empty kernel         median=4.2108 samples=[4.210786, 6.42932, 4.126087999999999, 6.377881, 4.169835, 6.274951, 4.193664, 6.228073, 4.160105]
args=16 INTJ static_compile kernel median=5.1524 samples=[3.795971, 5.135711000000001, 5.129524999999999, 5.209594, 5.724565999999999, 5.184724, 5.131238000000001, 5.152405, 5.716957000000001]
args=16 INTJ runtime_shim kernel median=3.7972 samples=[3.7972159999999997, 3.673719, 4.915152, 3.6127399999999996, 5.740009, 3.5740790000000002, 4.663109, 3.612208, 5.693293]
args=32 FFI packed nop           median=0.4804 samples=[0.5073840000000001, 0.490286, 0.46967899999999996, 0.480401, 0.479237, 0.48018, 0.47654199999999997, 0.49315699999999996, 0.49038400000000004]
args=32 FFI typed nop            median=0.4913 samples=[0.493827, 0.504145, 0.476516, 0.481217, 0.48788600000000004, 0.491274, 0.489285, 0.500023, 0.497389]
args=32 FFI empty kernel         median=4.9169 samples=[4.916872000000001, 6.275874, 4.893365, 6.224836, 4.805136999999999, 6.324991, 4.830478, 6.220773, 4.819927]
args=32 INTJ static_compile kernel median=5.8668 samples=[4.049838, 5.86676, 6.248968, 5.638834, 6.124754, 5.649921, 6.360227, 5.614206, 5.954035]
args=32 INTJ runtime_shim kernel median=4.1269 samples=[4.055442, 3.9608470000000002, 6.334689999999999, 4.076434, 5.88549, 4.1269219999999995, 6.0962250000000004, 4.096546999999999, 5.7207550000000005]
args=64 FFI packed nop           median=0.8289 samples=[0.82596, 0.861706, 0.8499410000000001, 0.824981, 0.827296, 0.828879, 0.822293, 0.83354, 0.8643970000000001]
args=64 FFI typed nop            median=0.8706 samples=[0.845553, 0.888668, 0.880623, 0.848964, 0.847565, 0.8706050000000001, 0.8505320000000001, 0.882317, 0.890813]
args=64 FFI empty kernel         median=5.9137 samples=[5.7541329999999995, 6.6179380000000005, 5.854301, 6.64626, 5.884018, 6.651398, 5.870323, 6.577921, 5.9136940000000005]
args=64 INTJ static_compile kernel median=6.3422 samples=[5.339843, 6.36682, 6.2009799999999995, 6.435131999999999, 6.3422160000000005, 6.354016, 6.258234000000001, 6.4224939999999995, 6.221963]
args=64 INTJ runtime_shim kernel median=6.2563 samples=[5.546155, 6.370075, 6.257844, 6.256341, 6.1840150000000005, 6.369684, 6.042288999999999, 6.343056, 6.162314]
elapsed_ns=4412485395
exit_status=0
ended=2026-09-25T16:27:33+08:00
```

### Controlled 100-call sweep: develop

```text
commit=ef698d24dae8790fffdf84fe766a7bf06db428f6
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_ffi_compare.py --sweep --iters 100 --batches 9
counts=[0, 3, 5, 8, 16, 32, 64] iters=100 batches=9 unit=us/call (host-only, no timed sync); INTJ device-bound, no specialization key
args= 0 FFI packed nop           median=0.1168 samples=[0.12267, 0.11783, 0.11679, 0.116, 0.11652, 0.11677, 0.14476, 0.11431999999999999, 0.11484]
args= 0 FFI typed nop            median=0.1192 samples=[0.12247, 0.12012, 0.11832, 0.11921999999999999, 0.11876, 0.12212999999999999, 0.11827, 0.12053, 0.11703]
args= 0 FFI empty kernel         median=1.8733 samples=[1.87325, 2.02274, 1.93717, 1.87968, 1.88094, 1.76786, 1.76939, 1.80829, 1.80398]
args= 0 INTJ cxx kernel          median=2.9158 samples=[3.24096, 3.14648, 3.00575, 2.9158000000000004, 2.91315, 2.86044, 2.9068, 3.04155, 2.86728]
args= 0 INTJ shim kernel         median=2.9615 samples=[3.1155, 3.23381, 3.06584, 3.04523, 2.96147, 2.80145, 2.8456300000000003, 2.88197, 2.8840100000000004]
args= 3 FFI packed nop           median=0.1414 samples=[0.14611000000000002, 0.14143, 0.13861, 0.14377, 0.14028, 0.14098, 0.14187, 0.14385, 0.14111]
args= 3 FFI typed nop            median=0.1461 samples=[0.14695, 0.1458, 0.14555, 0.15046, 0.14679, 0.15031999999999998, 0.14411000000000002, 0.1461, 0.14232]
args= 3 FFI empty kernel         median=3.5377 samples=[3.84879, 3.59259, 3.55335, 3.53767, 3.32382, 3.54113, 3.30762, 3.38931, 3.4011]
args= 3 INTJ cxx kernel          median=3.0752 samples=[3.06839, 3.2321999999999997, 3.2120900000000003, 3.2166300000000003, 3.07521, 2.9676, 2.9611199999999998, 3.0794, 3.06737]
args= 3 INTJ shim kernel         median=3.0912 samples=[3.2212199999999998, 3.36621, 3.4431599999999998, 3.27882, 3.0911500000000003, 3.0573200000000003, 3.03733, 3.07448, 3.03558]
args= 5 FFI packed nop           median=0.1710 samples=[0.17315, 0.17047, 0.17025, 0.17157, 0.16928, 0.17348, 0.17104, 0.17106, 0.17019]
args= 5 FFI typed nop            median=0.1737 samples=[0.1737, 0.17564, 0.17171, 0.17466, 0.17206, 0.17481, 0.17136, 0.173, 0.17389]
args= 5 FFI empty kernel         median=3.5208 samples=[3.4431700000000003, 3.5634099999999997, 3.61725, 3.65892, 3.3826300000000002, 3.55624, 3.35831, 3.52084, 3.3612800000000003]
args= 5 INTJ cxx kernel          median=3.1760 samples=[3.17596, 3.29123, 3.29162, 3.25139, 3.21466, 3.0619099999999997, 3.12599, 3.17194, 3.11331]
args= 5 INTJ shim kernel         median=3.1230 samples=[3.42206, 3.43454, 3.22709, 3.2336300000000002, 3.0839000000000003, 3.01386, 3.01119, 3.1230300000000004, 3.00625]
args= 8 FFI packed nop           median=0.2047 samples=[0.20545, 0.2034, 0.203, 0.22353, 0.20266, 0.20317, 0.20568, 0.20468, 0.20471999999999999]
args= 8 FFI typed nop            median=0.2068 samples=[0.20757, 0.20624, 0.20531, 0.2068, 0.20546999999999999, 0.20788, 0.20684, 0.20819, 0.20657]
args= 8 FFI empty kernel         median=3.6636 samples=[3.66525, 3.6636100000000003, 3.80939, 3.66933, 3.48387, 3.69489, 3.43921, 3.49996, 3.56617]
args= 8 INTJ cxx kernel          median=3.1451 samples=[3.30049, 3.32362, 3.29503, 3.4032600000000004, 3.11652, 3.13281, 3.10324, 3.1450500000000003, 3.0837399999999997]
args= 8 INTJ shim kernel         median=3.1294 samples=[3.2127399999999997, 3.1966300000000003, 3.25744, 3.25882, 3.12941, 3.08775, 2.9857199999999997, 3.09285, 3.08318]
args=16 FFI packed nop           median=0.2940 samples=[0.29258999999999996, 0.29534, 0.29560000000000003, 0.29335, 0.29404, 0.29312, 0.29369999999999996, 0.29522000000000004, 0.29511000000000004]
args=16 FFI typed nop            median=0.3044 samples=[0.30252999999999997, 0.30457999999999996, 0.30316000000000004, 0.30484, 0.30274, 0.30644, 0.30317, 0.30693, 0.30438]
args=16 FFI empty kernel         median=4.1078 samples=[4.1078, 3.97737, 4.31754, 4.06621, 4.26376, 3.93452, 4.17475, 3.9027800000000004, 4.13039]
args=16 INTJ cxx kernel          median=3.4950 samples=[3.61734, 3.5190799999999998, 3.65755, 3.49497, 3.55146, 3.4503600000000003, 3.47591, 3.37358, 3.42213]
args=16 INTJ shim kernel         median=3.5424 samples=[3.62187, 3.71484, 3.73095, 3.73662, 3.4994099999999997, 3.54241, 3.40442, 3.48142, 3.32698]
args=32 FFI packed nop           median=0.4870 samples=[0.47064, 0.50828, 0.48682, 0.47858, 0.48984, 0.47449, 0.48698, 0.48723, 0.49086]
args=32 FFI typed nop            median=0.4965 samples=[0.49187000000000003, 0.49309, 0.5303, 0.49645, 0.5012300000000001, 0.49476, 0.49293, 0.50032, 0.5036]
args=32 FFI empty kernel         median=4.4314 samples=[4.5005, 4.43139, 4.456600000000001, 4.5666899999999995, 4.38233, 4.46986, 4.39784, 4.39592, 4.282520000000001]
args=32 INTJ cxx kernel          median=3.8102 samples=[4.03381, 3.7545, 4.00223, 3.88754, 3.85454, 3.6930500000000004, 3.80095, 3.69002, 3.8102]
args=32 INTJ shim kernel         median=3.7696 samples=[3.90259, 3.94402, 3.71333, 4.01808, 3.77651, 3.7696, 3.69809, 3.7415599999999998, 3.52473]
args=64 FFI packed nop           median=0.8621 samples=[0.8343700000000001, 0.86253, 0.86523, 0.8616900000000001, 0.8604700000000001, 0.86578, 0.89159, 0.86215, 0.86191]
args=64 FFI typed nop            median=0.8900 samples=[0.879, 0.88938, 0.89064, 0.8900399999999999, 0.89358, 0.8884500000000001, 0.89249, 0.89195, 0.87491]
args=64 FFI empty kernel         median=5.3006 samples=[5.3194300000000005, 5.4040799999999996, 5.35531, 5.472659999999999, 5.186640000000001, 5.30058, 5.2133400000000005, 5.22896, 5.15574]
args=64 INTJ cxx kernel          median=4.8394 samples=[5.12247, 4.84796, 4.95336, 4.89983, 4.80882, 4.65529, 4.7569799999999995, 4.839390000000001, 4.635350000000001]
args=64 INTJ shim kernel         median=4.7413 samples=[4.98902, 4.80855, 4.8818, 4.866350000000001, 4.67654, 4.62411, 4.73567, 4.74132, 4.5821499999999995]
```

### Controlled 100-call sweep: merged

```text
commit=2a9f891443cf6763d26483c7b1f3d0f451d90a3c
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/torch_abi taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_ffi_compare.py --sweep --iters 100 --batches 9
counts=[0, 3, 5, 8, 16, 32, 64] iters=100 batches=9 unit=us/call (host-only, no timed sync); INTJ device-bound, no specialization key
args= 0 FFI packed nop           median=0.1165 samples=[0.12262999999999999, 0.11723, 0.11729, 0.11699, 0.11652, 0.11411, 0.1144, 0.11437, 0.11593]
args= 0 FFI typed nop            median=0.1192 samples=[0.11919, 0.12125, 0.12122, 0.11923, 0.11831, 0.11782, 0.1163, 0.11916, 0.11857]
args= 0 FFI empty kernel         median=1.8515 samples=[1.85147, 2.02818, 2.06706, 1.9422000000000001, 1.90255, 1.79608, 1.8218699999999999, 1.77247, 1.79301]
args= 0 INTJ static_compile kernel median=2.9618 samples=[3.32202, 3.0702, 3.05991, 3.02052, 2.9618200000000003, 2.93248, 2.91347, 2.9599, 2.8681799999999997]
args= 0 INTJ runtime_shim kernel median=2.9949 samples=[3.26229, 3.05334, 3.1145300000000002, 3.0778499999999998, 2.9949299999999996, 2.8181100000000003, 2.85917, 2.87453, 2.86073]
args= 3 FFI packed nop           median=0.1446 samples=[0.14461000000000002, 0.14294, 0.1407, 0.14453, 0.17711000000000002, 0.14472, 0.14464, 0.14143, 0.173]
args= 3 FFI typed nop            median=0.1463 samples=[0.14594, 0.14626, 0.14439, 0.14792, 0.14664, 0.15005000000000002, 0.14322, 0.1492, 0.14587]
args= 3 FFI empty kernel         median=3.4631 samples=[3.87859, 3.63239, 3.6485100000000004, 3.54928, 3.4630900000000002, 3.35733, 3.3165, 3.37811, 3.3649899999999997]
args= 3 INTJ static_compile kernel median=3.1592 samples=[3.15921, 3.21523, 3.3633200000000003, 3.24225, 3.16244, 3.01233, 2.93404, 3.03769, 3.0446999999999997]
args= 3 INTJ runtime_shim kernel median=3.2075 samples=[3.253, 3.2882399999999996, 3.4512199999999997, 3.24982, 3.20751, 3.04461, 3.09615, 3.10977, 3.08956]
args= 5 FFI packed nop           median=0.1715 samples=[0.17467, 0.17072, 0.17318, 0.17159, 0.17202, 0.17114, 0.17147, 0.17128, 0.17038]
args= 5 FFI typed nop            median=0.1762 samples=[0.17365, 0.17702, 0.17458, 0.17823, 0.17625, 0.17794, 0.17461000000000002, 0.1777, 0.17608000000000001]
args= 5 FFI empty kernel         median=3.4469 samples=[3.61313, 3.5807800000000003, 3.5684, 3.65014, 3.3979899999999996, 3.31721, 3.35646, 3.44693, 3.29521]
args= 5 INTJ static_compile kernel median=3.1757 samples=[3.3, 3.26987, 3.2946999999999997, 3.2200900000000003, 3.1756599999999997, 3.1686199999999998, 3.01331, 3.0761399999999997, 3.05673]
args= 5 INTJ runtime_shim kernel median=3.1614 samples=[3.3008699999999997, 3.30096, 3.25331, 3.2412199999999998, 3.1614, 3.0332, 3.0499099999999997, 3.11838, 3.073]
args= 8 FFI packed nop           median=0.2033 samples=[0.20387, 0.20285, 0.20468, 0.20486000000000001, 0.205, 0.20329, 0.20177, 0.20277, 0.20293]
args= 8 FFI typed nop            median=0.2068 samples=[0.20347, 0.20911000000000002, 0.20603, 0.211, 0.20684, 0.20657, 0.20749, 0.20721, 0.2037]
args= 8 FFI empty kernel         median=3.5175 samples=[3.8196, 3.65086, 3.75108, 3.6634, 3.51748, 3.46256, 3.49802, 3.4485799999999998, 3.50656]
args= 8 INTJ static_compile kernel median=3.1859 samples=[3.27684, 3.3465599999999998, 3.2869200000000003, 3.35834, 3.18587, 3.13021, 3.0716799999999997, 3.1333, 3.05155]
args= 8 INTJ runtime_shim kernel median=3.1359 samples=[3.20172, 3.19546, 3.2950500000000003, 3.22727, 3.1359299999999997, 3.1176500000000003, 2.97865, 3.08539, 3.0484899999999997]
args=16 FFI packed nop           median=0.2982 samples=[0.29788, 0.29664999999999997, 0.30036, 0.29887, 0.29975, 0.29818, 0.29351, 0.29877, 0.29697]
args=16 FFI typed nop            median=0.3054 samples=[0.30601999999999996, 0.305, 0.30542, 0.30859, 0.30199000000000004, 0.31118, 0.30107999999999996, 0.30882, 0.30228]
args=16 FFI empty kernel         median=4.2671 samples=[4.03772, 4.5183, 4.46148, 3.98767, 4.487520000000001, 3.7325100000000004, 4.26712, 3.7839, 4.27624]
args=16 INTJ static_compile kernel median=3.6074 samples=[3.68102, 3.6239899999999996, 3.70552, 3.60737, 3.7285100000000004, 3.3703600000000002, 3.42419, 3.3852800000000003, 3.43241]
args=16 INTJ runtime_shim kernel median=3.6598 samples=[3.7070700000000003, 3.7834899999999996, 3.66424, 3.82676, 3.6598, 3.52139, 3.35989, 3.55908, 3.39013]
args=32 FFI packed nop           median=0.4876 samples=[0.47857, 0.49149, 0.49029, 0.48671, 0.48413, 0.48756, 0.48834, 0.5165, 0.47905000000000003]
args=32 FFI typed nop            median=0.5009 samples=[0.49285, 0.5066700000000001, 0.5009, 0.50449, 0.49507999999999996, 0.50107, 0.50544, 0.49576, 0.49301]
args=32 FFI empty kernel         median=4.5534 samples=[4.61128, 4.55337, 4.6291, 4.61202, 4.435029999999999, 4.48161, 4.33114, 4.62858, 4.3858999999999995]
args=32 INTJ static_compile kernel median=3.8610 samples=[4.13389, 3.9354899999999997, 4.00676, 4.03803, 3.86098, 3.7531, 3.77896, 3.72155, 3.7485399999999998]
args=32 INTJ runtime_shim kernel median=3.9598 samples=[4.00786, 4.04129, 3.9598, 4.11548, 3.84138, 3.96705, 3.7694099999999997, 3.83999, 3.81593]
args=64 FFI packed nop           median=0.8557 samples=[0.82521, 0.8557100000000001, 0.86753, 0.84834, 0.84277, 0.84214, 0.8630599999999999, 0.86597, 0.86156]
args=64 FFI typed nop            median=0.8847 samples=[0.8590800000000001, 0.89352, 0.87864, 0.89213, 0.8846900000000001, 0.88358, 0.8924500000000001, 0.87495, 0.89001]
args=64 FFI empty kernel         median=5.3630 samples=[5.42024, 5.43929, 5.45706, 5.64364, 5.36301, 5.34574, 5.27844, 5.23988, 5.17362]
args=64 INTJ static_compile kernel median=5.0432 samples=[5.2729, 5.043229999999999, 5.052689999999999, 5.08292, 5.0539, 4.9276, 4.8550699999999996, 4.82258, 4.84035]
args=64 INTJ runtime_shim kernel median=4.9467 samples=[5.1124600000000004, 5.135770000000001, 5.0555200000000005, 5.06564, 4.94672, 4.8499799999999995, 4.84701, 4.7909, 4.8395600000000005]
```


## Final source `03a8f52`: affected benchmark repeats

### ffi_paths: final merged, round 2

```text
commit=03a8f52651248cb333ed50347fbabac6bd15103a
started=2026-09-25T16:35:44+08:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/torch_abi
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/torch_abi taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_intj_ffi_paths.py --iters 1000 --batches 9
iters=1000 batches=9 unit=ns/call; no_gpu=True
mode=callback
callback_count=9100 (100 warmup + 9000 unique misses)
INTJ cold compile callback + cache  median_ns=416.7 samples_ns=[463.597, 461.43, 404.184, 576.741, 391.092, 403.778, 409.255, 416.726, 1727.161]
INTJ hot call (no callback)         median_ns=40.1 samples_ns=[40.451, 41.764, 39.69, 41.071, 38.958, 40.592, 39.546, 40.068, 38.292]
INTJ 3-tensor host-only nop         median_ns=35.7 samples_ns=[37.482, 36.051, 35.359, 35.671, 35.255, 35.688, 35.211, 35.683, 35.126]
mode=kwargs
INTJ direct positional              median_ns=66.6 samples_ns=[67.967, 72.106, 66.546, 66.55, 66.392, 66.598, 66.5, 66.759, 66.906]
INTJ adapter positional             median_ns=89.7 samples_ns=[89.721, 89.36, 89.22, 89.447, 89.309, 103.353, 92.415, 92.01, 91.801]
INTJ adapter kwargs                 median_ns=109.6 samples_ns=[110.262, 110.175, 109.629, 108.913, 107.25, 112.66, 111.123, 108.01, 106.955]
INTJ adapter defaults               median_ns=88.4 samples_ns=[88.711, 87.895, 88.529, 88.394, 88.361, 88.204, 88.326, 92.771, 90.279]
INTJ FFI wrapper positional         median_ns=104.5 samples_ns=[104.717, 104.7, 105.084, 104.483, 104.457, 103.792, 104.303, 109.922, 103.146]
INTJ FFI wrapper kwargs             median_ns=125.5 samples_ns=[126.112, 125.142, 125.468, 124.905, 124.611, 124.114, 132.328, 126.895, 126.627]
INTJ native kwargs: unsupported (verified TypeError)
mode=dataclass
INTJ pair direct                    median_ns=68.8 samples_ns=[71.463, 70.505, 67.768, 68.758, 67.43, 68.954, 67.505, 69.269, 67.924]
INTJ pair prebuilt *tuple           median_ns=141.2 samples_ns=[141.227, 147.142, 139.78, 142.591, 140.144, 141.199, 140.233, 142.647, 141.665]
FFI unpack Pair only                median_ns=162.5 samples_ns=[162.461, 163.555, 161.346, 163.186, 161.807, 165.405, 160.56, 162.477, 160.242]
INTJ pair manual unpack             median_ns=172.5 samples_ns=[170.413, 172.454, 174.944, 175.558, 170.323, 172.109, 170.582, 172.623, 174.132]
INTJ pair FFI unpack                median_ns=316.0 samples_ns=[312.899, 315.822, 316.014, 317.002, 313.524, 316.614, 313.586, 317.853, 319.597]
INTJ pair stdlib astuple            median_ns=1172.6 samples_ns=[1326.153, 1208.422, 1175.684, 1169.508, 1154.266, 1164.307, 1172.582, 1183.061, 1151.176]
INTJ config direct                  median_ns=83.9 samples_ns=[84.341, 88.477, 83.037, 83.858, 83.363, 83.951, 83.236, 84.149, 83.133]
INTJ config FFI unpack              median_ns=318.8 samples_ns=[321.601, 327.701, 320.629, 318.761, 318.807, 318.55, 315.624, 324.275, 316.718]
INTJ native dataclass unpack: unsupported (verified TypeError)
elapsed_ns=2776119240
exit_status=0
ended=2026-09-25T16:35:47+08:00
```

### ffi_default: final merged, round 2

```text
commit=03a8f52651248cb333ed50347fbabac6bd15103a
started=2026-09-25T16:35:47+08:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/torch_abi
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/torch_abi taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_ffi_compare.py --iters 1000 --batches 9
iters=1000 batches=9 unit=us/call (host-only, no timed sync)
FFI packed nop             median=0.1438 samples=[0.14855000000000002, 0.14266399999999999, 0.144588, 0.14546, 0.14372200000000002, 0.14188900000000002, 0.14341900000000002, 0.143818, 0.147498]
FFI typed nop              median=0.1505 samples=[0.15047, 0.149924, 0.147225, 0.151381, 0.15062299999999998, 0.153272, 0.14640899999999998, 0.15174, 0.146227]
FFI empty kernel           median=3.2719 samples=[3.400847, 3.4661239999999998, 3.468272, 3.271933, 3.304515, 3.263864, 3.193928, 3.2408330000000003, 3.189036]
INTJ empty kernel          median=2.9859 samples=[3.227229, 3.092428, 3.090569, 2.928795, 2.873263, 2.9024029999999996, 2.985888, 2.856432, 2.9864360000000003]
INTJ fixed-device kernel   median=2.8782 samples=[3.050303, 3.07906, 3.070492, 2.969821, 2.8782069999999997, 2.826855, 2.825641, 2.793308, 2.803366]
FFI packed nop mixed       median=0.1740 samples=[0.17387200000000003, 0.174016, 0.17455099999999998, 0.175376, 0.174318, 0.173724, 0.172944, 0.173732, 0.174392]
FFI typed nop mixed        median=0.1776 samples=[0.177341, 0.17808500000000002, 0.177627, 0.179771, 0.176864, 0.177694, 0.175031, 0.181345, 0.175957]
FFI mixed kernel           median=3.3478 samples=[3.427727, 3.50325, 3.479213, 3.375248, 3.303183, 3.3478209999999997, 3.23003, 3.2558200000000004, 3.2273539999999996]
INTJ mixed kernel          median=2.8682 samples=[3.075578, 3.088414, 2.980439, 2.928352, 2.868224, 2.8522559999999997, 2.821351, 2.838961, 2.8097510000000003]
INTJ fixed mixed kernel    median=2.8571 samples=[3.105541, 3.088844, 2.9932869999999996, 3.218883, 2.8571210000000002, 2.848315, 2.787828, 2.798592, 2.778534]
elapsed_ns=3627118589
exit_status=0
ended=2026-09-25T16:35:51+08:00
```

### ffi_sweep: final merged, round 2

```text
commit=03a8f52651248cb333ed50347fbabac6bd15103a
started=2026-09-25T16:35:51+08:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/torch_abi
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/torch_abi taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_ffi_compare.py --sweep --iters 1000 --batches 9
counts=[0, 3, 5, 8, 16, 32, 64] iters=1000 batches=9 unit=us/call (host-only, no timed sync); INTJ device-bound, no specialization key
args= 0 FFI packed nop           median=0.1176 samples=[0.122397, 0.117581, 0.120395, 0.118613, 0.117586, 0.117539, 0.11730299999999999, 0.117522, 0.117213]
args= 0 FFI typed nop            median=0.1201 samples=[0.119982, 0.120129, 0.11905, 0.121532, 0.119059, 0.12157, 0.119315, 0.121267, 0.121476]
args= 0 FFI empty kernel         median=1.6506 samples=[1.613616, 1.867925, 1.650637, 1.849005, 1.598402, 1.894701, 1.575076, 1.883394, 1.574092]
args= 0 INTJ static_compile kernel median=2.6969 samples=[3.0941080000000003, 2.786757, 2.698073, 2.688257, 2.696898, 2.6509989999999997, 2.684597, 2.784525, 2.690532]
args= 0 INTJ runtime_shim kernel median=2.6913 samples=[2.847661, 2.701873, 2.701105, 2.680688, 2.7072950000000002, 2.6816060000000004, 2.688904, 2.6743580000000002, 2.691272]
args= 3 FFI packed nop           median=0.1467 samples=[0.14610900000000002, 0.146655, 0.146001, 0.14674500000000001, 0.14466900000000002, 0.145309, 0.14841300000000002, 0.14697200000000002, 0.15031299999999997]
args= 3 FFI typed nop            median=0.1496 samples=[0.15169, 0.151808, 0.149561, 0.15116300000000002, 0.148135, 0.149345, 0.150974, 0.148681, 0.14624299999999998]
args= 3 FFI empty kernel         median=3.2564 samples=[3.318991, 3.262443, 3.146701, 3.272569, 3.156655, 3.256442, 3.1440659999999996, 3.283142, 3.1466860000000003]
args= 3 INTJ static_compile kernel median=2.9374 samples=[2.9757190000000002, 2.849716, 2.93742, 2.8636939999999997, 2.9471640000000003, 2.863654, 2.943061, 2.861459, 2.942192]
args= 3 INTJ runtime_shim kernel median=2.8589 samples=[2.971243, 2.860513, 2.797062, 2.868059, 2.763441, 2.852424, 2.865569, 2.858851, 2.760147]
args= 5 FFI packed nop           median=0.1803 samples=[0.180339, 0.179784, 0.180448, 0.17999199999999999, 0.18106, 0.179754, 0.182067, 0.17809, 0.181805]
args= 5 FFI typed nop            median=0.1864 samples=[0.186405, 0.18712, 0.181985, 0.20157499999999998, 0.18212, 0.186764, 0.184751, 0.18654400000000002, 0.182733]
args= 5 FFI empty kernel         median=3.2913 samples=[3.362289, 3.2984769999999997, 3.232205, 3.2913110000000003, 3.227147, 3.3254520000000003, 3.2766529999999996, 3.306632, 3.20675]
args= 5 INTJ static_compile kernel median=2.9235 samples=[2.9638299999999997, 2.980576, 2.9427559999999997, 2.923458, 2.928612, 2.849926, 2.885221, 2.904072, 2.9058539999999997]
args= 5 INTJ runtime_shim kernel median=2.8365 samples=[2.998129, 2.8364540000000003, 2.845451, 2.7674209999999997, 2.9104050000000004, 2.758293, 2.833819, 2.755198, 2.882301]
args= 8 FFI packed nop           median=0.2137 samples=[0.211971, 0.21199500000000002, 0.216733, 0.209643, 0.214249, 0.212671, 0.21598699999999998, 0.213665, 0.21565600000000001]
args= 8 FFI typed nop            median=0.2179 samples=[0.218523, 0.218023, 0.21593600000000002, 0.214469, 0.215923, 0.217934, 0.22134399999999999, 0.21800299999999997, 0.216262]
args= 8 FFI empty kernel         median=3.3842 samples=[3.5371840000000003, 3.4264699999999997, 3.32715, 3.397154, 3.384194, 3.3807869999999998, 3.3078980000000002, 3.390046, 3.369421]
args= 8 INTJ static_compile kernel median=3.0303 samples=[3.046529, 2.9591930000000004, 3.052543, 3.0303020000000003, 3.0363119999999997, 3.008374, 3.030348, 3.004912, 3.052185]
args= 8 INTJ runtime_shim kernel median=2.9070 samples=[3.060123, 2.906999, 2.968141, 2.873389, 2.963512, 2.8641889999999997, 2.843895, 2.865652, 2.945878]
args=16 FFI packed nop           median=0.3052 samples=[0.312899, 0.305183, 0.30524799999999996, 0.303581, 0.299568, 0.304807, 0.30956900000000004, 0.306778, 0.30441]
args=16 FFI typed nop            median=0.3163 samples=[0.31942000000000004, 0.317851, 0.31459, 0.31620299999999996, 0.318384, 0.316346, 0.313347, 0.316736, 0.310716]
args=16 FFI empty kernel         median=3.6096 samples=[3.711103, 3.698007, 3.527557, 3.632611, 3.530574, 3.662047, 3.5312959999999998, 3.6095639999999998, 3.528209]
args=16 INTJ static_compile kernel median=3.3685 samples=[3.29582, 3.368733, 3.4691590000000003, 3.273688, 3.4477249999999997, 3.330285, 3.3685039999999997, 3.270526, 3.470791]
args=16 INTJ runtime_shim kernel median=3.2506 samples=[3.25063, 3.130722, 3.302263, 3.0179, 3.3152440000000003, 3.008753, 3.2664, 3.010341, 3.298681]
args=32 FFI packed nop           median=0.4929 samples=[0.505726, 0.5175700000000001, 0.527076, 0.491447, 0.498104, 0.492832, 0.492867, 0.48956099999999997, 0.489528]
args=32 FFI typed nop            median=0.5178 samples=[0.519713, 0.536625, 0.507558, 0.5373239999999999, 0.517774, 0.510337, 0.507993, 0.518138, 0.505966]
args=32 FFI empty kernel         median=4.2338 samples=[4.317266, 4.27875, 4.13246, 4.238873, 4.099346, 4.23382, 4.168284, 4.281896, 4.1634340000000005]
args=32 INTJ static_compile kernel median=3.5705 samples=[3.4412220000000002, 3.570475, 3.5967800000000003, 3.539029, 3.602029, 3.5023400000000002, 3.7101610000000003, 3.484993, 3.638109]
args=32 INTJ runtime_shim kernel median=3.5851 samples=[3.436219, 3.5070140000000003, 3.608226, 3.428187, 3.599189, 3.6632260000000003, 3.603604, 3.4246529999999997, 3.585107]
args=64 FFI packed nop           median=0.8698 samples=[0.874726, 0.890816, 0.874096, 0.855773, 0.8648980000000001, 0.8661960000000001, 0.8697509999999999, 0.866745, 0.870287]
args=64 FFI typed nop            median=0.8889 samples=[0.911729, 0.909985, 0.88667, 0.888948, 0.874122, 0.893875, 0.8876649999999999, 0.890237, 0.880032]
args=64 FFI empty kernel         median=5.0694 samples=[5.125857, 5.13985, 5.078884, 5.038574, 5.071626, 5.062636, 5.060269, 5.069447, 5.062903]
args=64 INTJ static_compile kernel median=4.5910 samples=[4.712549, 4.652714, 4.602685, 4.575398, 4.571564, 4.56383, 4.600398, 4.587466999999999, 4.590981]
args=64 INTJ runtime_shim kernel median=4.5767 samples=[4.632853, 4.576690999999999, 4.527892, 4.582088, 4.533105, 4.592807, 4.496469, 4.584382, 4.508641]
elapsed_ns=4180193235
exit_status=0
ended=2026-09-25T16:35:55+08:00
```

### controlled: final merged, round 2

```text
commit=03a8f52651248cb333ed50347fbabac6bd15103a
started=2026-09-25T16:35:55+08:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/torch_abi
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/torch_abi taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_ffi_compare.py --sweep --iters 100 --batches 9
counts=[0, 3, 5, 8, 16, 32, 64] iters=100 batches=9 unit=us/call (host-only, no timed sync); INTJ device-bound, no specialization key
args= 0 FFI packed nop           median=0.1156 samples=[0.12146, 0.11393, 0.11533, 0.11551, 0.11564, 0.11616, 0.11591, 0.11591, 0.11543]
args= 0 FFI typed nop            median=0.1181 samples=[0.12434999999999999, 0.12043999999999999, 0.11814, 0.12053, 0.11754, 0.11804, 0.11678, 0.11906, 0.11781]
args= 0 FFI empty kernel         median=1.8287 samples=[1.84274, 2.1207, 1.92173, 1.92643, 1.82867, 1.81229, 1.75791, 1.76993, 1.75188]
args= 0 INTJ static_compile kernel median=2.8570 samples=[3.20565, 2.91125, 2.90414, 2.8570300000000004, 2.82048, 2.81627, 2.84033, 2.95465, 2.85575]
args= 0 INTJ runtime_shim kernel median=2.9129 samples=[3.04183, 3.01766, 3.01858, 3.04656, 2.9128600000000002, 2.76256, 2.7727, 2.8235300000000003, 2.8391100000000002]
args= 3 FFI packed nop           median=0.1427 samples=[0.14385, 0.1433, 0.13956, 0.14422, 0.14151, 0.14266, 0.13968, 0.14493999999999999, 0.14008]
args= 3 FFI typed nop            median=0.1459 samples=[0.14556, 0.1472, 0.14361000000000002, 0.14604, 0.14589, 0.14961000000000002, 0.14379, 0.14984999999999998, 0.14218]
args= 3 FFI empty kernel         median=3.3989 samples=[3.80201, 3.5309299999999997, 3.80393, 3.43582, 3.34552, 3.2862, 3.39894, 3.3370900000000003, 3.31209]
args= 3 INTJ static_compile kernel median=3.0220 samples=[3.05619, 3.15943, 3.1459, 3.14389, 3.02203, 2.93279, 2.88756, 3.01279, 2.99333]
args= 3 INTJ runtime_shim kernel median=3.0737 samples=[3.1552800000000003, 3.24961, 3.22473, 3.17429, 3.02684, 3.03912, 2.97542, 3.07365, 3.0139299999999998]
args= 5 FFI packed nop           median=0.1723 samples=[0.17248, 0.17259, 0.1736, 0.16973, 0.17232, 0.17236, 0.17211, 0.17202, 0.17118]
args= 5 FFI typed nop            median=0.1767 samples=[0.17464, 0.18159999999999998, 0.17671, 0.17563, 0.17344, 0.17896, 0.17707, 0.17907, 0.17531]
args= 5 FFI empty kernel         median=3.4793 samples=[3.55754, 3.51771, 3.56568, 3.53591, 3.36161, 3.29667, 3.31758, 3.4793499999999997, 3.3095100000000004]
args= 5 INTJ static_compile kernel median=3.0557 samples=[3.16584, 3.13815, 3.16458, 3.14236, 3.0556799999999997, 2.9832199999999998, 2.98277, 3.02448, 3.00408]
args= 5 INTJ runtime_shim kernel median=3.0538 samples=[3.1854899999999997, 3.21061, 3.09416, 3.1151500000000003, 3.0499, 2.99011, 2.97008, 3.0538499999999997, 3.0306599999999997]
args= 8 FFI packed nop           median=0.2044 samples=[0.20276, 0.2037, 0.20471, 0.20588, 0.20542, 0.20443, 0.20527, 0.20407, 0.2027]
args= 8 FFI typed nop            median=0.2072 samples=[0.2062, 0.20883000000000002, 0.20818, 0.21043, 0.20715, 0.20722, 0.2071, 0.20584, 0.20548]
args= 8 FFI empty kernel         median=3.5563 samples=[3.69081, 3.6980999999999997, 3.6961399999999998, 3.64858, 3.4544699999999997, 3.52484, 3.45443, 3.46256, 3.55628]
args= 8 INTJ static_compile kernel median=3.2107 samples=[3.23462, 3.2106500000000002, 3.37176, 3.25415, 3.21075, 3.0606999999999998, 3.20959, 3.12823, 3.10899]
args= 8 INTJ runtime_shim kernel median=3.0898 samples=[3.1300700000000004, 3.16365, 3.0955700000000004, 3.16174, 3.06071, 3.05267, 2.93059, 3.08983, 2.99702]
args=16 FFI packed nop           median=0.2961 samples=[0.29145, 0.29842, 0.29605000000000004, 0.29328, 0.29917, 0.29597, 0.29846, 0.29467000000000004, 0.29629]
args=16 FFI typed nop            median=0.3049 samples=[0.30052999999999996, 0.30488, 0.30338, 0.33529000000000003, 0.30710000000000004, 0.3338, 0.30523, 0.30426, 0.30473]
args=16 FFI empty kernel         median=4.0489 samples=[3.73889, 3.95418, 4.133319999999999, 4.05111, 4.21516, 3.7603400000000002, 4.05702, 3.89375, 4.04894]
args=16 INTJ static_compile kernel median=3.5623 samples=[3.64028, 3.47126, 3.5673000000000004, 3.56485, 3.56885, 3.5623, 3.4948699999999997, 3.44105, 3.47485]
args=16 INTJ runtime_shim kernel median=3.5293 samples=[3.6270700000000002, 3.64513, 3.61346, 3.6499, 3.44962, 3.52904, 3.31969, 3.52927, 3.36956]
args=32 FFI packed nop           median=0.4895 samples=[0.48062, 0.48016000000000003, 0.49219999999999997, 0.49024, 0.49260000000000004, 0.48951, 0.48806, 0.48923, 0.49095]
args=32 FFI typed nop            median=0.5011 samples=[0.49589, 0.496, 0.49733, 0.50794, 0.50455, 0.50393, 0.49471, 0.50138, 0.50106]
args=32 FFI empty kernel         median=4.3011 samples=[4.43634, 4.36101, 4.38624, 4.36095, 4.30114, 4.19878, 4.26887, 4.2744, 4.18116]
args=32 INTJ static_compile kernel median=3.7682 samples=[4.096430000000001, 3.7678499999999997, 3.87033, 3.84672, 3.757, 3.64396, 3.76817, 3.6801399999999997, 3.85704]
args=32 INTJ runtime_shim kernel median=3.7914 samples=[3.90186, 3.92142, 3.73527, 3.9249099999999997, 3.83144, 3.7914299999999996, 3.74808, 3.78682, 3.7292199999999998]
args=64 FFI packed nop           median=0.8580 samples=[0.82914, 0.8580399999999999, 0.84693, 0.86365, 0.87063, 0.8547899999999999, 0.85391, 0.8579600000000001, 0.86854]
args=64 FFI typed nop            median=0.8883 samples=[0.8503999999999999, 0.89277, 0.8874099999999999, 0.8826, 0.88497, 0.89179, 0.90063, 0.88976, 0.8882899999999999]
args=64 FFI empty kernel         median=5.2633 samples=[5.3703, 5.3109399999999996, 5.371689999999999, 5.4019200000000005, 5.26334, 5.20925, 5.19242, 5.206300000000001, 5.2172]
args=64 INTJ static_compile kernel median=4.7908 samples=[4.975899999999999, 4.87506, 4.89861, 4.9327, 4.779, 4.64344, 4.79079, 4.70778, 4.67034]
args=64 INTJ runtime_shim kernel median=4.7640 samples=[4.9021099999999995, 4.95575, 4.817489999999999, 4.836189999999999, 4.6965200000000005, 4.646319999999999, 4.68534, 4.68467, 4.76403]
elapsed_ns=3559941602
exit_status=0
ended=2026-09-25T16:35:58+08:00
```

# Complete `develop` Python benchmark run: round 2

Source `ef698d2`; setup, commands, and aggregate results are in [round 0](2026-09-25_develop-full_0_ef698d2.md). Each block includes commit, exact command, all reported per-batch samples, duration, and exit status.

## Raw outputs

### launch_gpu

```text
commit=ef698d24dae8790fffdf84fe766a7bf06db428f6
started=2026-09-25T21:01:33+08:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --iters 1000 --batches 9
mode=gpu; 1000 calls × 9 batches; median ns/call
               path    ns/call       Δ ns       Δ %   build ms   bind ms
           auto map     3590.2       +0.0      +0.0      72.28         -
        reduced key     3699.8     +109.6      +3.1       1.96         -
         verify off     3503.1      -87.0      -2.4       1.77         -
          verify on     3564.9      -25.2      -0.7       1.78         -
              baked     3690.5     +100.3      +2.8       1.74         -
       bound tensor     3564.3      -25.9      -0.7       0.39      1.45
      bound pointer     3689.9      +99.7      +2.8       0.43      1.43
   fixed device map     3519.6      -70.5      -2.0       0.35      1.59
fixed device no-map     3540.0      -50.2      -1.4       0.38      1.50
elapsed_ns=3847168518
exit_status=0
ended=2026-09-25T21:01:37+08:00
```

### launch_host

```text
commit=ef698d24dae8790fffdf84fe766a7bf06db428f6
started=2026-09-25T21:01:37+08:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --no-gpu --iters 100000 --batches 9
mode=host-only; 100000 calls × 9 batches; median ns/call
               path    ns/call       Δ ns       Δ %   build ms   bind ms
           auto map       44.4       +0.0      +0.0     103.94         -
        reduced key       43.5       -0.9      -2.0       1.38         -
         verify off       43.5       -0.9      -2.0       1.22         -
          verify on       44.6       +0.1      +0.3       1.16         -
              baked       39.5       -4.9     -11.0       1.44         -
       bound tensor       45.5       +1.1      +2.5       0.12      1.12
      bound pointer       44.4       +0.0      +0.1       0.11      1.07
   fixed device map       45.2       +0.8      +1.8       0.09      1.08
fixed device no-map       45.4       +1.0      +2.3       0.12      1.05
elapsed_ns=3045927682
exit_status=0
ended=2026-09-25T21:01:40+08:00
```

### launch_readme

```text
commit=ef698d24dae8790fffdf84fe766a7bf06db428f6
started=2026-09-25T21:01:40+08:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --readme --iters 1000 --batches 9
mode=readme; 1000 calls × 9 batches; median
        path   triton us    intj us   speedup
   grid=(1,)       16.35       3.11      5.3x
   grid=(0,)       12.83       0.03    447.9x

torch_access   decode ns    build s
        shim        93.9       0.02
         cxx        91.9       0.06
     cpython       870.3       0.00
elapsed_ns=3697768574
exit_status=0
ended=2026-09-25T21:01:44+08:00
```

### launch_sweep

```text
commit=ef698d24dae8790fffdf84fe766a7bf06db428f6
started=2026-09-25T21:01:44+08:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --sweep --iters 100000 --batches 9
mode=sweep; host-only; 100000 calls × 9 batches; median ns/call
count   kind    ns/call
    4    int       41.7
    4 tensor       40.5
   16    int       74.4
   16 tensor       72.9
   32    int      116.5
   32 tensor      120.2
elapsed_ns=2970039380
exit_status=0
ended=2026-09-25T21:01:47+08:00
```

### ffi_default

```text
commit=ef698d24dae8790fffdf84fe766a7bf06db428f6
started=2026-09-25T21:01:47+08:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_ffi_compare.py --iters 1000 --batches 9
iters=1000 batches=9 unit=us/call (host-only, no timed sync)
FFI packed nop             median=0.1437 samples=[0.148861, 0.143737, 0.14351, 0.144651, 0.143702, 0.14237, 0.142916, 0.14597900000000003, 0.145548]
FFI typed nop              median=0.1484 samples=[0.147133, 0.14844900000000003, 0.146787, 0.149498, 0.148563, 0.154248, 0.14740999999999999, 0.149012, 0.147006]
FFI empty kernel           median=3.3565 samples=[3.3977269999999997, 3.505824, 3.4371210000000003, 3.2430529999999997, 3.241244, 3.356535, 3.207287, 3.366302, 3.2088159999999997]
INTJ empty kernel          median=2.9880 samples=[3.08134, 3.019842, 3.024371, 2.840951, 2.8320790000000002, 2.93338, 2.9879879999999996, 2.938969, 3.009114]
INTJ fixed-device kernel   median=2.8811 samples=[3.3059160000000003, 3.0052269999999996, 3.056244, 2.881085, 2.861406, 2.838386, 2.812023, 2.8340189999999996, 2.915841]
FFI packed nop mixed       median=0.1740 samples=[0.171674, 0.17438, 0.17399799999999999, 0.171662, 0.173238, 0.17360599999999998, 0.17666300000000001, 0.17568199999999998, 0.17624199999999998]
FFI typed nop mixed        median=0.1774 samples=[0.175474, 0.179645, 0.177488, 0.17743199999999998, 0.17594200000000002, 0.17759899999999998, 0.174319, 0.178817, 0.175425]
FFI mixed kernel           median=3.3127 samples=[8.776451, 3.4458409999999997, 3.437539, 3.312672, 3.2382020000000002, 3.30939, 3.261467, 3.373382, 3.26048]
INTJ mixed kernel          median=2.9134 samples=[16.081593, 3.0983159999999996, 3.003204, 2.913389, 2.831785, 2.8433409999999997, 2.852042, 2.969326, 2.8299899999999996]
INTJ fixed mixed kernel    median=2.9249 samples=[6.229121, 3.12404, 3.030091, 2.990582, 2.858058, 2.8132689999999996, 2.913606, 2.924945, 2.9155819999999997]
elapsed_ns=3858806464
exit_status=0
ended=2026-09-25T21:01:50+08:00
```

### ffi_sweep

```text
commit=ef698d24dae8790fffdf84fe766a7bf06db428f6
started=2026-09-25T21:01:50+08:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_ffi_compare.py --sweep --iters 1000 --batches 9
counts=[0, 3, 5, 8, 16, 32, 64] iters=1000 batches=9 unit=us/call (host-only, no timed sync); INTJ device-bound, no specialization key
args= 0 FFI packed nop           median=0.1179 samples=[0.122824, 0.117867, 0.118699, 0.117517, 0.117181, 0.11668200000000001, 0.11633, 0.12259099999999999, 0.119075]
args= 0 FFI typed nop            median=0.1213 samples=[0.122459, 0.121714, 0.118737, 0.121347, 0.11939100000000001, 0.120277, 0.11791299999999999, 0.125642, 0.124433]
args= 0 FFI empty kernel         median=1.7864 samples=[1.617147, 2.8365859999999996, 2.076965, 1.808948, 1.637315, 1.786427, 1.5668630000000001, 1.809221, 1.5746179999999999]
args= 0 INTJ cxx kernel          median=2.7867 samples=[3.2736509999999996, 3.2874760000000003, 3.287688, 2.806011, 2.786711, 2.7351889999999996, 2.725859, 2.73841, 2.748315]
args= 0 INTJ shim kernel         median=2.7824 samples=[3.184396, 3.068112, 3.383293, 2.782044, 3.066904, 2.6957289999999996, 2.760402, 2.697441, 2.782444]
args= 3 FFI packed nop           median=0.1454 samples=[0.14747200000000002, 0.144954, 0.144186, 0.145399, 0.147057, 0.145122, 0.14767599999999997, 0.149787, 0.143069]
args= 3 FFI typed nop            median=0.1490 samples=[0.153092, 0.14902500000000002, 0.147118, 0.149872, 0.14793199999999998, 0.151386, 0.146907, 0.153142, 0.147702]
args= 3 FFI empty kernel         median=3.3208 samples=[3.671325, 3.732553, 3.7158130000000003, 3.3984409999999996, 3.258352, 3.3207739999999997, 3.211524, 3.3087199999999997, 3.221971]
args= 3 INTJ cxx kernel          median=2.8933 samples=[3.3224340000000003, 3.4079699999999997, 3.476984, 3.10915, 2.8932919999999998, 2.840705, 2.835288, 2.830779, 2.825821]
args= 3 INTJ shim kernel         median=2.8856 samples=[3.3553479999999998, 3.335022, 3.49364, 16.066083, 2.8856149999999996, 2.794745, 2.8570160000000002, 2.7830340000000002, 2.852351]
args= 5 FFI packed nop           median=0.1756 samples=[0.17560800000000001, 0.179303, 0.173647, 0.293739, 0.173307, 0.179685, 0.174216, 0.17369300000000001, 0.178048]
args= 5 FFI typed nop            median=0.1780 samples=[0.17530600000000002, 0.178671, 0.177974, 0.456442, 0.177881, 0.184891, 0.17695, 0.182923, 0.172193]
args= 5 FFI empty kernel         median=3.3090 samples=[3.7546, 4.205137, 3.740076, 14.953087, 3.309043, 3.253534, 3.229117, 3.253987, 3.2438789999999997]
args= 5 INTJ cxx kernel          median=2.9002 samples=[3.329382, 3.457546, 3.464074, 6.030363, 2.900202, 2.849174, 2.8239340000000004, 2.843022, 2.839985]
args= 5 INTJ shim kernel         median=2.8787 samples=[3.344764, 3.414663, 3.47381, 3.374932, 2.854777, 2.8720529999999997, 2.8238670000000003, 2.878749, 2.841592]
args= 8 FFI packed nop           median=0.2058 samples=[0.204375, 0.205499, 0.20849199999999998, 0.204313, 0.21074700000000002, 0.20622300000000002, 0.20991200000000002, 0.20441700000000002, 0.205816]
args= 8 FFI typed nop            median=0.2095 samples=[0.208321, 0.209738, 0.209798, 0.211127, 0.209043, 0.214775, 0.209466, 0.20908000000000002, 0.20845]
args= 8 FFI empty kernel         median=3.4117 samples=[3.931354, 4.29068, 3.771141, 4.2659769999999995, 3.275071, 3.411651, 3.257219, 3.400576, 3.263375]
args= 8 INTJ cxx kernel          median=3.1749 samples=[3.399133, 3.477768, 3.47299, 3.564984, 3.174933, 3.002169, 3.060549, 2.9659319999999996, 3.105487]
args= 8 INTJ shim kernel         median=2.9518 samples=[3.394272, 3.260861, 3.561456, 3.2772710000000003, 2.8769839999999998, 2.877788, 2.861425, 2.8911550000000004, 2.9517640000000003]
args=16 FFI packed nop           median=0.2934 samples=[0.297471, 0.289323, 0.295225, 0.293397, 0.29242, 0.2916, 0.294271, 0.288723, 0.297648]
args=16 FFI typed nop            median=0.3046 samples=[0.302, 0.304586, 0.304639, 0.304643, 0.305423, 0.304388, 0.301846, 0.306724, 0.302209]
args=16 FFI empty kernel         median=3.6866 samples=[4.2783999999999995, 5.770314999999999, 4.138838, 5.839335, 3.533375, 3.686643, 3.54877, 3.647965, 3.546714]
args=16 INTJ cxx kernel          median=3.5770 samples=[3.818375, 5.458819999999999, 5.707221, 5.205353, 3.382316, 3.329736, 3.3423670000000003, 3.344974, 3.577018]
args=16 INTJ shim kernel         median=3.3773 samples=[3.819842, 3.643554, 5.45092, 3.60238, 3.295372, 3.032486, 3.293016, 3.051853, 3.3772539999999998]
args=32 FFI packed nop           median=0.4816 samples=[0.478543, 0.479633, 0.48299200000000003, 0.48157, 0.48379700000000003, 0.480522, 0.484132, 0.47751299999999997, 0.481568]
args=32 FFI typed nop            median=0.4946 samples=[0.496596, 0.49918, 0.496891, 0.49250499999999997, 0.487807, 0.494592, 0.498475, 0.490274, 0.490008]
args=32 FFI empty kernel         median=4.1403 samples=[4.755485999999999, 5.998094, 4.664104999999999, 6.005437, 4.066764, 4.1402849999999995, 4.011158, 4.119624, 3.9781950000000004]
args=32 INTJ cxx kernel          median=3.8010 samples=[4.103535, 5.585116, 5.796709, 5.587180999999999, 3.80098, 3.575292, 3.793021, 3.596597, 3.679042]
args=32 INTJ shim kernel         median=3.6379 samples=[4.0403139999999995, 4.056291, 5.527178, 4.047245, 3.637875, 3.412786, 3.6064879999999997, 3.604717, 3.5809830000000002]
args=64 FFI packed nop           median=0.8412 samples=[0.826447, 0.849602, 0.867045, 0.824226, 0.8412050000000001, 0.822331, 0.848248, 0.8412229999999999, 0.845547]
args=64 FFI typed nop            median=0.8621 samples=[0.855476, 0.888096, 0.886173, 0.858796, 0.8714550000000001, 0.854519, 0.8583310000000001, 0.862776, 0.862094]
args=64 FFI empty kernel         median=5.1168 samples=[5.749102, 6.413041, 5.83083, 6.449577000000001, 5.116844, 5.065361, 5.040636999999999, 5.050211, 5.069338]
args=64 INTJ cxx kernel          median=4.6590 samples=[5.355948000000001, 6.138407, 6.086691, 6.13366, 4.638457000000001, 4.629255, 4.6227979999999995, 4.658968, 4.622553]
args=64 INTJ shim kernel         median=4.6220 samples=[5.154424, 6.058576, 6.048777, 6.224220999999999, 4.581605, 4.603282, 4.586518, 4.622042, 4.523671]
elapsed_ns=4470166720
exit_status=0
ended=2026-09-25T21:01:55+08:00
```

### ffi_paths

```text
commit=ef698d24dae8790fffdf84fe766a7bf06db428f6
started=2026-09-25T21:01:55+08:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_intj_ffi_paths.py --mode all --iters 1000 --batches 9
iters=1000 batches=9 unit=ns/call; no_gpu=True
mode=callback
callback_count=9100 (100 warmup + 9000 unique misses)
INTJ cold compile callback + cache  median_ns=413.9 samples_ns=[452.903, 446.445, 399.896, 542.823, 382.276, 393.912, 410.336, 413.943, 1905.408]
INTJ hot call (no callback)         median_ns=38.2 samples_ns=[40.574, 39.0, 38.115, 38.228, 38.052, 38.262, 38.041, 38.937, 37.864]
INTJ 3-tensor host-only nop         median_ns=35.4 samples_ns=[37.413, 35.869, 35.274, 35.684, 35.348, 35.508, 35.241, 35.366, 34.581]
mode=kwargs
INTJ direct positional              median_ns=66.8 samples_ns=[69.755, 67.071, 66.811, 66.751, 66.692, 72.561, 65.74, 65.802, 65.722]
INTJ adapter positional             median_ns=91.6 samples_ns=[92.482, 92.664, 92.761, 91.568, 91.589, 91.321, 91.338, 90.864, 95.639]
INTJ adapter kwargs                 median_ns=109.6 samples_ns=[109.635, 108.321, 109.705, 110.04, 109.188, 108.744, 109.452, 109.867, 126.622]
INTJ adapter defaults               median_ns=89.1 samples_ns=[89.149, 89.816, 89.086, 88.859, 88.964, 89.417, 89.804, 89.225, 88.772]
INTJ FFI wrapper positional         median_ns=104.2 samples_ns=[108.174, 104.126, 104.846, 104.652, 104.234, 104.129, 104.332, 103.729, 104.245]
INTJ FFI wrapper kwargs             median_ns=128.9 samples_ns=[137.315, 129.405, 129.233, 128.12, 128.805, 128.928, 128.896, 127.439, 137.856]
INTJ native kwargs: unsupported (verified TypeError)
mode=dataclass
INTJ pair direct                    median_ns=67.8 samples_ns=[70.099, 73.328, 67.132, 67.996, 67.246, 72.673, 66.89, 67.841, 67.219]
INTJ pair prebuilt *tuple           median_ns=142.1 samples_ns=[140.842, 144.654, 141.902, 145.314, 142.143, 146.65, 141.182, 143.44, 139.87]
FFI unpack Pair only                median_ns=161.7 samples_ns=[166.879, 163.952, 176.914, 161.418, 160.96, 161.485, 160.215, 161.656, 167.29]
INTJ pair manual unpack             median_ns=171.5 samples_ns=[171.023, 174.065, 170.754, 177.299, 171.099, 177.755, 169.425, 174.256, 171.544]
INTJ pair FFI unpack                median_ns=315.0 samples_ns=[310.109, 318.9, 314.225, 314.969, 318.256, 313.878, 312.737, 320.645, 315.5]
INTJ pair stdlib astuple            median_ns=1155.3 samples_ns=[1326.625, 1220.155, 1146.59, 1165.398, 1147.538, 1154.715, 1155.26, 1171.025, 1145.968]
INTJ config direct                  median_ns=82.2 samples_ns=[83.273, 82.473, 82.671, 82.227, 81.307, 81.907, 81.36, 81.808, 90.828]
INTJ config FFI unpack              median_ns=327.8 samples_ns=[325.423, 328.485, 327.752, 326.763, 325.399, 333.403, 327.443, 330.821, 331.195]
INTJ native dataclass unpack: unsupported (verified TypeError)
elapsed_ns=2871096393
exit_status=0
ended=2026-09-25T21:01:58+08:00
```

### hip_same_hsaco

```text
commit=ef698d24dae8790fffdf84fe766a7bf06db428f6
started=2026-09-25T21:01:58+08:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_hip_module_launch.py --iters 1000 --batches 9
hsaco_sha256=df3e529e178196ad62f7c07ab8593ede5b8596c76e7c6fbd12a5d947a696e65c kernel=empty_kernel threads=256 shared=0 callback_count=1
iters=1000 batches=9 unit=us/call (host enqueue, no timed sync)
TVM FFI HIP HSACO         median=3.2022 samples=[3.557114, 3.2163530000000002, 3.211843, 3.187393, 3.3864699999999996, 3.20221, 3.135685, 3.113512, 3.119669]
Triton same HSACO         median=15.7194 samples=[15.807388999999999, 15.75741, 15.71936, 15.725791, 15.744757, 15.621401, 15.643593000000001, 15.665066000000001, 15.671877]
INTJ same function        median=2.9917 samples=[3.011788, 2.996381, 2.991698, 3.052354, 3.0002370000000003, 2.981944, 2.9198380000000004, 2.926824, 2.887383]
elapsed_ns=3749638031
exit_status=0
ended=2026-09-25T21:02:02+08:00
```

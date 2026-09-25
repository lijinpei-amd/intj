# Exact merged `develop` benchmark: round 1

Source `38cdabc604b6c9a1a3ea57667e1f942fc5bc7960`. Environment, commands, median-of-process medians and caveats: [round 0](2026-09-25_merged-develop_0_38cdabc.md). Raw outputs below retain every value printed by each benchmark.

## Raw outputs

### launch_gpu

```text
commit=38cdabc604b6c9a1a3ea57667e1f942fc5bc7960
started=2026-09-25T21:31:46+08:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --iters 1000 --batches 9
mode=gpu; 1000 calls × 9 batches; median ns/call
               path    ns/call       Δ ns       Δ %   build ms   bind ms
           auto map     3085.0       +0.0      +0.0     107.95         -
        reduced key     3094.7       +9.7      +0.3       3.44         -
         verify off     2946.9     -138.1      -4.5       1.69         -
          verify on     2986.5      -98.5      -3.2       1.65         -
              baked     3072.0      -13.0      -0.4      73.03         -
       bound tensor     3057.1      -27.9      -0.9       0.36      1.46
      bound pointer     2991.8      -93.1      -3.0       0.35      1.44
   fixed device map     2898.7     -186.3      -6.0       0.31      1.49
fixed device no-map     3014.6      -70.4      -2.3       0.27      1.43
elapsed_ns=3949738600
exit_status=0
ended=2026-09-25T21:31:50+08:00
```

### launch_host

```text
commit=38cdabc604b6c9a1a3ea57667e1f942fc5bc7960
started=2026-09-25T21:31:50+08:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --no-gpu --iters 100000 --batches 9
mode=host-only; 100000 calls × 9 batches; median ns/call
               path    ns/call       Δ ns       Δ %   build ms   bind ms
           auto map       43.6       +0.0      +0.0      60.30         -
        reduced key       43.3       -0.3      -0.7       1.54         -
         verify off       43.6       -0.0      -0.0       1.39         -
          verify on       44.5       +0.8      +1.9       1.37         -
              baked       39.8       -3.8      -8.7       2.55         -
       bound tensor       45.6       +2.0      +4.5       0.15      1.31
      bound pointer       45.4       +1.8      +4.1       0.13      1.25
   fixed device map       44.5       +0.9      +2.0       0.10      1.24
fixed device no-map       43.1       -0.5      -1.1       0.14      1.23
elapsed_ns=3086433038
exit_status=0
ended=2026-09-25T21:31:53+08:00
```

### launch_readme

```text
commit=38cdabc604b6c9a1a3ea57667e1f942fc5bc7960
started=2026-09-25T21:31:53+08:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --readme --iters 1000 --batches 9
mode=readme; 1000 calls × 9 batches; median
        path   triton us    intj us   speedup
   grid=(1,)       16.32       3.11      5.3x
   grid=(0,)       12.96       0.03    428.2x

torch_access_mode   decode ns    build s
     runtime_shim        90.4       0.02
   static_compile        89.9       0.06
      interpreter       874.1       0.00
elapsed_ns=3867434652
exit_status=0
ended=2026-09-25T21:31:57+08:00
```

### launch_sweep

```text
commit=38cdabc604b6c9a1a3ea57667e1f942fc5bc7960
started=2026-09-25T21:31:57+08:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --sweep --iters 100000 --batches 9
mode=sweep; host-only; 100000 calls × 9 batches; median ns/call
count   kind    ns/call
    4    int       42.7
    4 tensor       41.4
   16    int       74.2
   16 tensor       68.9
   32    int      120.8
   32 tensor      167.0
elapsed_ns=3176036594
exit_status=0
ended=2026-09-25T21:32:00+08:00
```

### ffi_default

```text
commit=38cdabc604b6c9a1a3ea57667e1f942fc5bc7960
started=2026-09-25T21:32:00+08:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_ffi_compare.py --iters 1000 --batches 9
iters=1000 batches=9 unit=us/call (host-only, no timed sync)
FFI packed nop             median=0.1454 samples=[0.149839, 0.143832, 0.145393, 0.145164, 0.14540999999999998, 0.14568199999999998, 0.145897, 0.144656, 0.14529599999999998]
FFI typed nop              median=0.1502 samples=[0.158249, 0.149801, 0.150825, 0.15091300000000002, 0.15024500000000002, 0.150234, 0.148817, 0.14886600000000003, 0.14829]
FFI empty kernel           median=3.3328 samples=[3.405865, 3.494797, 3.332783, 3.1992420000000004, 3.197332, 3.354286, 3.188824, 3.335302, 3.1996529999999996]
INTJ empty kernel          median=2.9743 samples=[3.100886, 3.049487, 3.138562, 2.910969, 2.865245, 2.974277, 2.959895, 2.928203, 3.007865]
INTJ fixed-device kernel   median=2.9072 samples=[3.03647, 3.005967, 3.088404, 2.907226, 2.8602220000000003, 2.789603, 2.787692, 2.965267, 2.89535]
FFI packed nop mixed       median=0.1743 samples=[0.177284, 0.174254, 0.174928, 0.17793899999999999, 0.17331, 0.171524, 0.17227199999999998, 0.177126, 0.17275800000000002]
FFI typed nop mixed        median=0.1773 samples=[0.179884, 0.18010400000000001, 0.177255, 0.175484, 0.178677, 0.176667, 0.17341499999999999, 0.181674, 0.17381200000000002]
FFI mixed kernel           median=3.2645 samples=[3.3623719999999997, 3.5097869999999998, 3.390444, 3.251533, 3.1878409999999997, 3.264489, 3.255427, 3.353783, 3.254631]
INTJ mixed kernel          median=2.9255 samples=[3.141663, 3.033986, 2.927228, 2.925523, 2.8516120000000003, 2.860989, 2.820243, 2.934995, 2.8251239999999997]
INTJ fixed mixed kernel    median=2.9198 samples=[3.160105, 3.0568, 3.002196, 2.991104, 2.902662, 2.899548, 2.913183, 2.918729, 2.919784]
elapsed_ns=3999689431
exit_status=0
ended=2026-09-25T21:32:04+08:00
```

### ffi_sweep

```text
commit=38cdabc604b6c9a1a3ea57667e1f942fc5bc7960
started=2026-09-25T21:32:04+08:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_ffi_compare.py --sweep --iters 1000 --batches 9
counts=[0, 3, 5, 8, 16, 32, 64] iters=1000 batches=9 unit=us/call (host-only, no timed sync); INTJ device-bound, no specialization key
args= 0 FFI packed nop           median=0.1163 samples=[0.121237, 0.116742, 0.11966500000000001, 0.12296699999999999, 0.11422, 0.11630700000000001, 0.11604099999999999, 0.11431999999999999, 0.11455]
args= 0 FFI typed nop            median=0.1183 samples=[0.120244, 0.12190000000000001, 0.116235, 0.11946, 0.117248, 0.11827700000000001, 0.117451, 0.121794, 0.116887]
args= 0 FFI empty kernel         median=1.7103 samples=[1.5627039999999999, 2.178467, 1.710285, 2.032257, 1.569389, 1.810067, 1.577202, 1.868297, 1.645598]
args= 0 INTJ static_compile kernel median=2.7787 samples=[3.124791, 6.519334, 3.032879, 2.734223, 2.722385, 2.885736, 2.7106239999999997, 2.778656, 2.7672440000000003]
args= 0 INTJ runtime_shim kernel median=2.7524 samples=[2.894832, 19.022217, 3.001095, 2.658128, 2.752386, 2.6870610000000004, 2.706674, 2.70633, 2.756089]
args= 3 FFI packed nop           median=0.1457 samples=[0.14753, 0.300585, 0.145984, 0.144984, 0.14249, 0.145653, 0.150051, 0.145749, 0.144264]
args= 3 FFI typed nop            median=0.1500 samples=[0.148041, 0.274278, 0.14505600000000002, 0.15142, 0.14580500000000002, 0.149787, 0.15073, 0.15027600000000002, 0.15001]
args= 3 FFI empty kernel         median=3.2330 samples=[3.343921, 16.268558000000002, 3.286731, 3.3094859999999997, 3.134935, 3.22819, 3.15628, 3.2330039999999998, 3.154785]
args= 3 INTJ static_compile kernel median=3.0128 samples=[3.025404, 6.917812, 3.113553, 2.949328, 3.022635, 2.92938, 3.0128359999999996, 2.936083, 2.990029]
args= 3 INTJ runtime_shim kernel median=2.9408 samples=[3.011377, 3.8724659999999997, 3.1003429999999996, 2.805761, 2.961915, 2.940832, 2.935011, 2.9383690000000002, 2.926362]
args= 5 FFI packed nop           median=0.1735 samples=[0.171819, 0.174596, 0.173487, 0.171655, 0.174534, 0.17242500000000002, 0.17458500000000002, 0.173028, 0.173944]
args= 5 FFI typed nop            median=0.1781 samples=[0.174283, 0.183425, 0.17324, 0.193549, 0.175034, 0.17871, 0.175254, 0.178595, 0.178072]
args= 5 FFI empty kernel         median=3.2965 samples=[3.385305, 4.1912139999999996, 3.354234, 3.315236, 3.209369, 3.2965, 3.275098, 3.275422, 3.275214]
args= 5 INTJ static_compile kernel median=2.9010 samples=[2.988781, 2.920172, 3.0256979999999998, 2.9178490000000004, 2.89066, 2.8852379999999997, 2.9009650000000002, 2.879696, 2.899743]
args= 5 INTJ runtime_shim kernel median=2.8941 samples=[3.0035410000000002, 2.977725, 3.016655, 2.863295, 3.139762, 2.888449, 2.877897, 2.8941060000000003, 2.88151]
args= 8 FFI packed nop           median=0.2059 samples=[0.205162, 0.205848, 0.205864, 0.20616, 0.20655500000000002, 0.204794, 0.210697, 0.205945, 0.20612200000000003]
args= 8 FFI typed nop            median=0.2091 samples=[0.206683, 0.212909, 0.20495500000000003, 0.20937799999999998, 0.209101, 0.209549, 0.208304, 0.210613, 0.206679]
args= 8 FFI empty kernel         median=3.4089 samples=[3.517217, 3.450859, 3.410088, 3.408885, 3.3212550000000003, 3.4226579999999998, 3.3261689999999997, 3.400043, 3.349685]
args= 8 INTJ static_compile kernel median=3.0477 samples=[3.0719600000000002, 2.962781, 3.151836, 3.030088, 3.047727, 2.9684009999999996, 3.051748, 2.9509000000000003, 3.1932519999999998]
args= 8 INTJ runtime_shim kernel median=2.8862 samples=[3.084011, 2.945105, 3.093104, 2.8757170000000003, 2.883203, 2.879449, 2.868051, 2.896816, 2.886173]
args=16 FFI packed nop           median=0.2923 samples=[0.294005, 0.29907999999999996, 0.295862, 0.28958, 0.29229, 0.29795, 0.290626, 0.290715, 0.291219]
args=16 FFI typed nop            median=0.3031 samples=[0.305117, 0.30689999999999995, 0.298064, 0.305515, 0.303114, 0.30229, 0.301921, 0.30086399999999996, 0.303254]
args=16 FFI empty kernel         median=3.6557 samples=[3.7259729999999998, 3.7137249999999997, 3.5602069999999997, 3.9148989999999997, 3.536362, 3.6640659999999996, 3.5549299999999997, 3.6557049999999998, 3.536078]
args=16 INTJ static_compile kernel median=3.3226 samples=[3.282944, 3.364515, 3.786689, 3.492247, 3.463944, 3.309109, 3.3226419999999997, 3.29034, 3.311988]
args=16 INTJ runtime_shim kernel median=3.2261 samples=[3.301349, 3.161684, 3.569869, 3.0235149999999997, 3.318032, 3.028121, 3.296746, 3.047536, 3.226124]
args=32 FFI packed nop           median=0.4787 samples=[0.478985, 0.477798, 0.4781, 0.49249400000000004, 0.475674, 0.47886700000000004, 0.47868299999999997, 0.47141000000000005, 0.505765]
args=32 FFI typed nop            median=0.4949 samples=[0.488311, 0.49492200000000003, 0.494591, 0.496019, 0.49861500000000003, 0.495641, 0.49077, 0.49117500000000003, 0.5109239999999999]
args=32 FFI empty kernel         median=4.1369 samples=[4.194113000000001, 4.138494, 3.908751, 4.332694, 3.97287, 4.149112, 4.017832, 4.136906, 4.0321750000000005]
args=32 INTJ static_compile kernel median=3.6449 samples=[3.4648499999999998, 3.6329160000000003, 3.8577339999999998, 3.762654, 3.644917, 3.560969, 3.7515300000000003, 3.5781039999999997, 3.766264]
args=32 INTJ runtime_shim kernel median=3.5102 samples=[3.463281, 3.5102260000000003, 3.8674, 3.3966410000000002, 3.610592, 3.4447170000000003, 3.621655, 3.4466170000000003, 3.642523]
args=64 FFI packed nop           median=0.8356 samples=[0.835586, 0.849854, 0.845283, 0.834812, 0.835579, 0.831232, 0.808728, 0.833309, 0.8471559999999999]
args=64 FFI typed nop            median=0.8629 samples=[0.873472, 0.892615, 0.868654, 0.865078, 0.857873, 0.862115, 0.853144, 0.860904, 0.8629]
args=64 FFI empty kernel         median=5.0653 samples=[5.065259, 5.124147, 4.927709, 5.138824, 5.04717, 5.050768, 5.071024, 5.05561, 5.066828999999999]
args=64 INTJ static_compile kernel median=4.6265 samples=[4.707484, 4.668137, 4.683582, 4.741427, 4.6254930000000005, 4.6156049999999995, 4.626533, 4.606078999999999, 4.622657]
args=64 INTJ runtime_shim kernel median=4.6035 samples=[4.7141779999999995, 4.714093, 4.673005, 4.699618, 4.517259, 4.603524, 4.515397, 4.590415, 4.548499]
elapsed_ns=4620112236
exit_status=0
ended=2026-09-25T21:32:09+08:00
```

### ffi_paths

```text
commit=38cdabc604b6c9a1a3ea57667e1f942fc5bc7960
started=2026-09-25T21:32:09+08:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_intj_ffi_paths.py --mode all --iters 1000 --batches 9
iters=1000 batches=9 unit=ns/call; no_gpu=True
mode=callback
callback_count=9100 (100 warmup + 9000 unique misses)
INTJ cold compile callback + cache  median_ns=421.9 samples_ns=[472.337, 454.011, 402.658, 674.69, 386.708, 411.724, 405.083, 421.868, 1553.645]
INTJ hot call (no callback)         median_ns=39.2 samples_ns=[41.064, 40.916, 38.192, 38.665, 40.659, 45.146, 38.756, 39.191, 38.024]
INTJ 3-tensor host-only nop         median_ns=35.3 samples_ns=[36.677, 35.807, 35.066, 35.415, 34.922, 35.324, 34.918, 35.328, 35.203]
mode=kwargs
INTJ direct positional              median_ns=66.2 samples_ns=[68.349, 66.361, 66.163, 65.951, 65.961, 66.059, 66.264, 66.272, 66.156]
INTJ adapter positional             median_ns=93.5 samples_ns=[90.351, 90.062, 89.985, 107.755, 94.777, 93.948, 93.701, 93.47, 93.297]
INTJ adapter kwargs                 median_ns=109.4 samples_ns=[109.094, 107.542, 107.125, 112.08, 109.397, 110.159, 110.027, 109.441, 110.249]
INTJ adapter defaults               median_ns=90.3 samples_ns=[91.004, 91.325, 89.981, 91.21, 94.994, 90.262, 90.227, 89.372, 89.46]
INTJ FFI wrapper positional         median_ns=106.7 samples_ns=[107.503, 105.253, 104.969, 105.084, 104.654, 118.065, 107.176, 106.741, 106.756]
INTJ FFI wrapper kwargs             median_ns=127.3 samples_ns=[126.467, 125.212, 127.716, 125.486, 134.521, 127.307, 127.051, 127.368, 127.737]
INTJ native kwargs: unsupported (verified TypeError)
mode=dataclass
INTJ pair direct                    median_ns=68.6 samples_ns=[71.484, 75.145, 67.942, 68.653, 67.538, 68.334, 67.884, 68.563, 71.788]
INTJ pair prebuilt *tuple           median_ns=141.4 samples_ns=[141.433, 141.911, 138.243, 141.057, 137.618, 141.415, 143.376, 141.444, 138.921]
FFI unpack Pair only                median_ns=164.7 samples_ns=[169.314, 169.873, 169.125, 171.806, 164.102, 164.519, 164.337, 164.734, 164.114]
INTJ pair manual unpack             median_ns=171.1 samples_ns=[183.295, 180.255, 169.092, 170.89, 169.472, 171.952, 181.419, 171.124, 169.863]
INTJ pair FFI unpack                median_ns=313.8 samples_ns=[312.949, 317.783, 310.054, 314.272, 313.756, 312.544, 310.165, 318.169, 313.975]
INTJ pair stdlib astuple            median_ns=1161.9 samples_ns=[1310.598, 1201.061, 1151.417, 1174.66, 1161.898, 1152.171, 1149.074, 1168.898, 1153.88]
INTJ config direct                  median_ns=82.5 samples_ns=[81.894, 84.426, 81.912, 84.024, 82.915, 82.247, 82.465, 82.459, 84.065]
INTJ config FFI unpack              median_ns=322.4 samples_ns=[324.093, 323.157, 322.236, 326.619, 322.35, 325.034, 321.371, 320.801, 318.168]
INTJ native dataclass unpack: unsupported (verified TypeError)
elapsed_ns=2960494290
exit_status=0
ended=2026-09-25T21:32:12+08:00
```

### hip_same_hsaco

```text
commit=38cdabc604b6c9a1a3ea57667e1f942fc5bc7960
started=2026-09-25T21:32:12+08:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_hip_module_launch.py --iters 1000 --batches 9
hsaco_sha256=df3e529e178196ad62f7c07ab8593ede5b8596c76e7c6fbd12a5d947a696e65c kernel=empty_kernel threads=256 shared=0 callback_count=1
iters=1000 batches=9 unit=us/call (host enqueue, no timed sync)
TVM FFI HIP HSACO         median=3.1980 samples=[3.449532, 3.2528989999999998, 3.467308, 3.197984, 3.1988380000000003, 3.131901, 3.1407249999999998, 3.0901840000000003, 3.0915079999999997]
Triton same HSACO         median=15.7019 samples=[15.846495, 15.8094, 15.858641, 15.795172, 15.701943, 15.661112999999999, 15.683388, 15.644292, 15.656754999999999]
INTJ same function        median=3.0440 samples=[3.113065, 3.122585, 3.044035, 3.058515, 3.046346, 2.916469, 2.982781, 2.940682, 2.893738]
elapsed_ns=3812826660
exit_status=0
ended=2026-09-25T21:32:16+08:00
```

## Launcher per-batch diagnostics (supplemental)

The external driver and timing caveat are in [round 0](2026-09-25_merged-develop_0_38cdabc.md).

### launch_gpu per-batch, round 1

```text
commit=38cdabc604b6c9a1a3ea57667e1f942fc5bc7960
started=2026-09-25T21:35:22+08:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj taskset -c 0 /tmp/gb2/bin/python /tmp/intj-launch-batch-recorder-38cdabc.py benchmarks/bench_launch.py --iters 1000 --batches 9
mode=gpu; 1000 calls × 9 batches; median ns/call
               path    ns/call       Δ ns       Δ %   build ms   bind ms
recorded_batch_samples_ns=[3154.656, 3450.722, 3136.724, 3140.575, 3116.798, 3107.279, 3141.928, 3140.383, 3114.99]
           auto map     3140.4       +0.0      +0.0      93.72         -
recorded_batch_samples_ns=[2955.316, 3242.717, 3133.946, 3114.708, 3143.105, 3178.862, 3088.64, 3026.672, 3046.424]
        reduced key     3114.7      -25.7      -0.8      73.94         -
recorded_batch_samples_ns=[2834.984, 3109.049, 3004.883, 3001.129, 3006.411, 3002.409, 3011.679, 2955.374, 2933.532]
         verify off     3002.4     -138.0      -4.4       1.73         -
recorded_batch_samples_ns=[2765.347, 2925.999, 2930.442, 2923.458, 2927.741, 2915.987, 2911.685, 2873.551, 2868.14]
          verify on     2916.0     -224.4      -7.1       1.65         -
recorded_batch_samples_ns=[2687.064, 2952.861, 2941.568, 2949.079, 2950.679, 2951.411, 2949.207, 2917.628, 2924.095]
              baked     2949.1     -191.3      -6.1       1.73         -
recorded_batch_samples_ns=[2704.193, 2930.61, 2940.531, 2934.263, 2951.638, 2933.115, 2936.802, 2931.578, 2945.743]
       bound tensor     2934.3     -206.1      -6.6       0.35      1.46
recorded_batch_samples_ns=[2736.478, 2938.246, 2955.306, 2951.568, 2954.571, 2937.705, 2938.987, 2944.49, 2937.184]
      bound pointer     2939.0     -201.4      -6.4       0.34      1.43
recorded_batch_samples_ns=[2763.054, 2886.618, 2886.849, 2882.225, 2892.371, 2894.494, 2893.425, 2880.258, 2885.325]
   fixed device map     2886.6     -253.8      -8.1       0.41      1.48
recorded_batch_samples_ns=[2749.14, 2901.054, 2898.816, 2878.844, 2892.94, 2878.514, 2885.222, 2870.978, 2888.919]
fixed device no-map     2885.2     -255.2      -8.1       0.27      1.45
elapsed_ns=3637246961
exit_status=0
ended=2026-09-25T21:35:26+08:00
```

### launch_host per-batch, round 1

```text
commit=38cdabc604b6c9a1a3ea57667e1f942fc5bc7960
started=2026-09-25T21:35:26+08:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj taskset -c 0 /tmp/gb2/bin/python /tmp/intj-launch-batch-recorder-38cdabc.py benchmarks/bench_launch.py --no-gpu --iters 100000 --batches 9
mode=host-only; 100000 calls × 9 batches; median ns/call
               path    ns/call       Δ ns       Δ %   build ms   bind ms
recorded_batch_samples_ns=[43.52526, 44.29205, 43.67708, 44.05287, 43.29372, 44.05452, 43.93581, 44.55845, 43.66294]
           auto map       43.9       +0.0      +0.0      67.90         -
recorded_batch_samples_ns=[43.05496, 43.62371, 43.14672, 43.66138, 43.15124, 44.07439, 43.01726, 43.62833, 43.05808]
        reduced key       43.2       -0.8      -1.8       1.52         -
recorded_batch_samples_ns=[43.25759, 45.43289, 43.25766, 43.79226, 43.27824, 43.74157, 43.22275, 43.79031, 43.19718]
         verify off       43.3       -0.7      -1.5       1.42         -
recorded_batch_samples_ns=[44.19443, 44.78587, 44.2275, 45.3097, 44.46914, 44.96131, 44.49836, 44.67718, 44.09247]
          verify on       44.5       +0.6      +1.3       1.39         -
recorded_batch_samples_ns=[39.81436, 40.31257, 39.76884, 41.2284, 39.72212, 40.31381, 39.67023, 40.32146, 40.04834]
              baked       40.0       -3.9      -8.8       1.95         -
recorded_batch_samples_ns=[45.74184, 46.06534, 45.44536, 46.19144, 45.44534, 46.42705, 45.49225, 47.50658, 45.49496]
       bound tensor       45.7       +1.8      +4.1       0.14      1.29
recorded_batch_samples_ns=[45.29743, 44.33563, 43.97249, 44.13697, 44.95064, 44.26757, 44.28769, 44.03776, 43.99487]
      bound pointer       44.3       +0.3      +0.8       0.13      1.27
recorded_batch_samples_ns=[43.96691, 44.26833, 43.79851, 44.33659, 43.83668, 44.73397, 43.85529, 44.71617, 44.32865]
   fixed device map       44.3       +0.3      +0.8       0.11      1.23
recorded_batch_samples_ns=[42.95659, 44.33731, 42.84902, 44.67801, 43.26614, 44.63933, 42.65651, 44.26549, 42.60258]
fixed device no-map       43.3       -0.7      -1.5       0.14      1.22
elapsed_ns=2888276746
exit_status=0
ended=2026-09-25T21:35:29+08:00
```

### launch_readme per-batch, round 1

```text
commit=38cdabc604b6c9a1a3ea57667e1f942fc5bc7960
started=2026-09-25T21:35:29+08:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj taskset -c 0 /tmp/gb2/bin/python /tmp/intj-launch-batch-recorder-38cdabc.py benchmarks/bench_launch.py --readme --iters 1000 --batches 9
recorded_batch_samples_ns=[89.55, 88.45, 90.71, 89.134, 93.487, 88.435, 87.996, 88.981, 87.35]
recorded_batch_samples_ns=[100.67, 96.263, 92.575, 92.639, 92.292, 90.238, 89.131, 91.189, 92.181]
recorded_batch_samples_ns=[880.491, 875.917, 868.764, 872.28, 858.696, 886.463, 870.896, 910.33, 886.668]
mode=readme; 1000 calls × 9 batches; median
        path   triton us    intj us   speedup
recorded_batch_samples_ns=[16762.067, 16974.837, 16795.729, 17113.518, 16801.468, 16907.048, 16853.573, 16913.179, 16856.823]
recorded_batch_samples_ns=[3323.51, 3798.708, 3636.077, 3618.678, 3613.02, 3629.604, 3610.714, 3475.593, 3471.4]
   grid=(1,)       16.86       3.61      4.7x
recorded_batch_samples_ns=[12935.805, 12963.776, 12904.415, 12948.698, 12932.677, 12952.185, 12930.203, 12907.897, 12913.95]
recorded_batch_samples_ns=[29.168, 28.583, 28.486, 28.547, 28.327, 28.502, 28.511, 28.301, 28.754]
   grid=(0,)       12.93       0.03    453.6x

torch_access_mode   decode ns    build s
     runtime_shim        89.0       0.02
   static_compile        92.3       0.08
      interpreter       875.9       0.00
elapsed_ns=3713823938
exit_status=0
ended=2026-09-25T21:35:33+08:00
```

### launch_sweep per-batch, round 1

```text
commit=38cdabc604b6c9a1a3ea57667e1f942fc5bc7960
started=2026-09-25T21:35:33+08:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj taskset -c 0 /tmp/gb2/bin/python /tmp/intj-launch-batch-recorder-38cdabc.py benchmarks/bench_launch.py --sweep --iters 100000 --batches 9
mode=sweep; host-only; 100000 calls × 9 batches; median ns/call
count   kind    ns/call
recorded_batch_samples_ns=[41.69538, 42.62068, 41.28403, 41.91503, 41.33828, 41.86346, 41.33519, 41.73641, 41.40345]
    4    int       41.7
recorded_batch_samples_ns=[40.35977, 40.88049, 40.39782, 40.95263, 40.64287, 41.00846, 40.60764, 40.92398, 40.42098]
    4 tensor       40.6
recorded_batch_samples_ns=[73.70054, 74.17188, 73.74277, 74.93455, 73.72661, 74.08858, 73.64125, 74.64298, 73.4865]
   16    int       73.7
recorded_batch_samples_ns=[68.34104, 72.98593, 68.42854, 69.52813, 69.56634, 70.1147, 67.9831, 68.92956, 68.01422]
   16 tensor       68.9
recorded_batch_samples_ns=[121.0547, 121.79736, 122.95232, 120.61275, 121.08306, 115.54817, 119.33058, 122.66841, 121.20737]
   32    int      121.1
recorded_batch_samples_ns=[118.40354, 113.78991, 115.28035, 115.49749, 115.52544, 125.48207, 117.69911, 115.66622, 115.20165]
   32 tensor      115.5
elapsed_ns=2958086847
exit_status=0
ended=2026-09-25T21:35:36+08:00
```

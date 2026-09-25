# Callable-grid targeted ABBA controls — 2026-09-26, repeat 3 of 4

[Aggregate, environment, commands, and caveats](2026-09-26_callable-grid-targeted_0_c5295b1.md).
These are the six process outputs for targeted repeat 3, in actual
C3 → D3 order within
each mode. They include nine-batch samples, absolute commands, UTC times,
commits, and exit status. The local date is 2026-09-26 (Asia/Shanghai).

## Raw outputs — repeat 3

### ffi_sweep: candidate r3

```text
commit=c5295b18f29a15fc1184eb7e93dbc099e6b75bf2
started=2026-09-25T18:39:09.538421+00:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/callable_grid
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/callable_grid taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_ffi_compare.py --sweep --iters 1000 --batches 9
counts=[0, 3, 5, 8, 16, 32, 64] iters=1000 batches=9 unit=us/call (host-only, no timed sync); INTJ device-bound, no specialization key
args= 0 FFI packed nop           median=0.1171 samples=[0.119217, 0.117132, 0.11751099999999999, 0.11609699999999999, 0.11519599999999999, 0.11626600000000001, 0.117555, 0.11617400000000001, 0.11870900000000001]
args= 0 FFI typed nop            median=0.1201 samples=[0.120616, 0.126821, 0.11744, 0.122757, 0.11879, 0.12157599999999999, 0.11630800000000001, 0.120147, 0.118379]
args= 0 FFI empty kernel         median=1.6411 samples=[1.604861, 1.881225, 1.6158620000000001, 1.8755519999999999, 1.579404, 1.825764, 1.5789190000000002, 1.859427, 1.6410740000000001]
args= 0 INTJ static_compile kernel median=2.7212 samples=[3.029871, 2.693711, 2.721158, 2.730625, 2.683748, 2.6527190000000003, 2.704119, 9.950361000000001, 2.7654929999999998]
args= 0 INTJ runtime_shim kernel median=2.6957 samples=[5.673405, 2.6878629999999997, 2.726534, 2.6505569999999996, 2.6906, 2.668017, 2.69573, 4.996433, 2.750553]
args= 3 FFI packed nop           median=0.1448 samples=[0.45238799999999996, 0.146281, 0.144561, 0.14457599999999998, 0.14574199999999998, 0.143565, 0.142685, 0.145581, 0.144817]
args= 3 FFI typed nop            median=0.1502 samples=[0.361182, 0.151117, 0.15096, 0.149334, 0.14732, 0.152559, 0.147518, 0.150233, 0.147543]
args= 3 FFI empty kernel         median=3.1885 samples=[4.365501, 3.207671, 3.157103, 3.199886, 3.134929, 3.197328, 3.130965, 3.188526, 3.143544]
args= 3 INTJ static_compile kernel median=2.9714 samples=[2.978031, 2.865081, 3.011603, 2.877911, 2.994596, 2.881779, 2.971381, 2.867743, 3.008759]
args= 3 INTJ runtime_shim kernel median=2.8528 samples=[2.9990390000000002, 2.852763, 2.814681, 2.8690680000000004, 2.93771, 2.8523519999999998, 2.808716, 2.842749, 2.901003]
args= 5 FFI packed nop           median=0.1740 samples=[0.17551499999999998, 0.173647, 0.17399, 0.17715199999999998, 0.173249, 0.174013, 0.174832, 0.17302099999999998, 0.1761]
args= 5 FFI typed nop            median=0.1771 samples=[0.178237, 0.178675, 0.175536, 0.178535, 0.177006, 0.17563399999999998, 0.178339, 0.17711000000000002, 0.17644300000000002]
args= 5 FFI empty kernel         median=3.2697 samples=[3.3315520000000003, 3.480833, 3.171168, 3.3149699999999998, 3.167729, 3.26969, 3.174756, 3.2860189999999996, 3.2433609999999997]
args= 5 INTJ static_compile kernel median=2.8845 samples=[2.964368, 2.912188, 2.829826, 3.14818, 2.818377, 2.8683400000000003, 2.9041840000000003, 2.840372, 2.884502]
args= 5 INTJ runtime_shim kernel median=2.8089 samples=[2.988697, 2.858144, 2.808853, 2.7714589999999997, 2.804818, 2.761958, 2.870105, 2.766556, 2.852205]
args= 8 FFI packed nop           median=0.2064 samples=[0.20459200000000002, 0.20103800000000002, 0.20923599999999998, 0.205489, 0.20641, 0.207058, 0.20750200000000002, 0.20565, 0.20754599999999998]
args= 8 FFI typed nop            median=0.2096 samples=[0.208154, 0.207176, 0.20957900000000002, 0.212554, 0.208518, 0.209574, 0.213695, 0.20974500000000001, 0.208635]
args= 8 FFI empty kernel         median=3.3643 samples=[3.465465, 3.43211, 3.227004, 3.399609, 3.235768, 3.366749, 3.3084949999999997, 3.364306, 3.2978]
args= 8 INTJ static_compile kernel median=3.0321 samples=[3.03208, 2.9293609999999997, 3.076276, 2.909607, 3.0668189999999997, 2.9148470000000004, 3.066848, 2.981924, 3.035688]
args= 8 INTJ runtime_shim kernel median=2.9015 samples=[3.036968, 2.901513, 2.913156, 2.895808, 2.9290770000000004, 2.8667629999999997, 3.011498, 2.8792310000000003, 2.8759050000000004]
args=16 FFI packed nop           median=0.2923 samples=[0.29752300000000004, 0.289842, 0.293648, 0.29244600000000004, 0.284417, 0.294751, 0.291404, 0.29006, 0.292313]
args=16 FFI typed nop            median=0.3018 samples=[0.301834, 0.30691399999999996, 0.30346100000000004, 0.30317, 0.298284, 0.301358, 0.30051, 0.301995, 0.301351]
args=16 FFI empty kernel         median=3.6309 samples=[3.666848, 3.6819789999999997, 3.516292, 3.63827, 3.505479, 3.6693119999999997, 3.4966619999999997, 3.6308890000000003, 3.516518]
args=16 INTJ static_compile kernel median=3.4167 samples=[3.271225, 3.3349569999999997, 3.467045, 3.353048, 3.433692, 11.865331, 3.416725, 3.327698, 3.423812]
args=16 INTJ runtime_shim kernel median=3.2459 samples=[3.2458739999999997, 3.1149989999999996, 3.369179, 3.068777, 3.301175, 3.025886, 3.279216, 3.041474, 3.2607399999999997]
args=32 FFI packed nop           median=0.4788 samples=[0.47702100000000003, 0.46010700000000004, 0.483137, 0.47967000000000004, 0.479768, 0.483526, 0.478837, 0.476625, 0.475018]
args=32 FFI typed nop            median=0.4938 samples=[0.497627, 0.499695, 0.493779, 0.494679, 0.48786900000000005, 0.49224, 0.492495, 0.49885, 0.491257]
args=32 FFI empty kernel         median=4.2078 samples=[4.289324, 4.23477, 4.114783, 4.254093, 4.079618, 4.209499, 4.0707249999999995, 4.207848, 4.0924000000000005]
args=32 INTJ static_compile kernel median=3.6096 samples=[3.454898, 3.672748, 3.6362330000000003, 3.594229, 3.641576, 3.5413200000000002, 3.629828, 3.577368, 3.609587]
args=32 INTJ runtime_shim kernel median=3.4900 samples=[3.410902, 3.4900349999999998, 3.610898, 3.433567, 3.584821, 3.398393, 3.596927, 3.4087310000000004, 3.557402]
args=64 FFI packed nop           median=0.8367 samples=[0.8393390000000001, 0.8461369999999999, 0.8315220000000001, 0.825328, 0.822157, 0.811754, 0.844341, 0.83665, 0.837615]
args=64 FFI typed nop            median=0.8633 samples=[0.8659070000000001, 0.865082, 0.86398, 0.86326, 0.8477509999999999, 0.846628, 0.855368, 0.8715259999999999, 0.859553]
args=64 FFI empty kernel         median=5.1194 samples=[5.146482, 5.138848, 5.136247, 5.127718, 5.1194489999999995, 5.111641, 5.108087, 5.106938, 5.0823]
args=64 INTJ static_compile kernel median=4.5626 samples=[4.684068, 4.576579, 4.555359999999999, 4.58233, 4.562585, 4.570575, 4.542324, 4.543551000000001, 4.534485]
args=64 INTJ runtime_shim kernel median=4.5432 samples=[4.616616, 4.565085, 4.543157, 12.792959000000002, 4.447209, 4.552326, 4.522643, 4.5372520000000005, 4.462382]
exit_status=0
ended=2026-09-25T18:39:13.946199+00:00
```

### ffi_sweep: develop r3

```text
commit=1bf689b37137e373b6baecf241a8d8a7da23630b
started=2026-09-25T18:39:13.948702+00:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_ffi_compare.py --sweep --iters 1000 --batches 9
counts=[0, 3, 5, 8, 16, 32, 64] iters=1000 batches=9 unit=us/call (host-only, no timed sync); INTJ device-bound, no specialization key
args= 0 FFI packed nop           median=0.1192 samples=[0.118862, 0.11592, 0.11987, 0.120589, 0.117258, 0.11989799999999999, 0.115108, 0.119214, 0.120642]
args= 0 FFI typed nop            median=0.1208 samples=[0.120245, 0.122721, 0.11941, 0.121022, 0.117775, 0.12371, 0.11918000000000001, 0.122751, 0.120833]
args= 0 FFI empty kernel         median=1.6842 samples=[1.6562750000000002, 1.8787, 1.684243, 1.853453, 1.6075139999999999, 1.8638160000000001, 1.59283, 1.880674, 1.589776]
args= 0 INTJ static_compile kernel median=2.7017 samples=[3.1200810000000003, 2.753908, 2.781949, 2.669617, 2.701727, 2.677832, 2.700743, 2.7940270000000003, 2.692056]
args= 0 INTJ runtime_shim kernel median=2.6954 samples=[2.880638, 2.688506, 2.77717, 2.6802330000000003, 2.707744, 2.69538, 2.70464, 2.666802, 2.69301]
args= 3 FFI packed nop           median=0.1454 samples=[0.145362, 0.148979, 0.14673599999999998, 0.145228, 0.14272900000000002, 0.145498, 0.144387, 0.146012, 0.143875]
args= 3 FFI typed nop            median=0.1497 samples=[0.147813, 0.152098, 0.150475, 0.153448, 0.14829499999999998, 0.149659, 0.14851499999999998, 0.151142, 0.146017]
args= 3 FFI empty kernel         median=3.2168 samples=[3.366784, 3.216754, 3.18211, 3.251876, 3.181746, 3.2486439999999996, 3.175034, 3.223441, 3.159658]
args= 3 INTJ static_compile kernel median=2.9264 samples=[2.991523, 2.809437, 2.942985, 2.830276, 2.9625500000000002, 2.851027, 2.9363110000000003, 2.826763, 2.9264200000000002]
args= 3 INTJ runtime_shim kernel median=2.8279 samples=[3.0088820000000003, 2.806353, 2.793768, 2.827921, 2.9367910000000004, 2.844471, 2.802636, 2.839886, 2.793029]
args= 5 FFI packed nop           median=0.1744 samples=[0.17422100000000001, 0.174398, 0.17446899999999999, 0.175731, 0.17387899999999998, 0.173239, 0.174933, 0.173367, 0.17504]
args= 5 FFI typed nop            median=0.1790 samples=[0.17760599999999999, 0.179915, 0.179034, 0.194905, 0.173708, 0.17868299999999998, 0.180401, 0.17991200000000002, 0.175577]
args= 5 FFI empty kernel         median=3.2877 samples=[3.3579470000000002, 3.287668, 3.202587, 3.293509, 3.2167440000000003, 3.2893380000000003, 3.194471, 3.299713, 3.208783]
args= 5 INTJ static_compile kernel median=2.8466 samples=[2.979717, 2.833846, 2.846635, 2.804261, 2.903602, 2.844974, 2.9209899999999998, 2.8149349999999997, 2.935997]
args= 5 INTJ runtime_shim kernel median=2.8364 samples=[3.014134, 2.8363690000000004, 2.808962, 2.7734870000000003, 2.8765, 2.7736680000000002, 2.873641, 2.765664, 2.889134]
args= 8 FFI packed nop           median=0.2073 samples=[0.205564, 0.207611, 0.207891, 0.207331, 0.21060800000000002, 0.206428, 0.20513900000000002, 0.20934999999999998, 0.207077]
args= 8 FFI typed nop            median=0.2109 samples=[0.207412, 0.210935, 0.21222300000000002, 0.214083, 0.209318, 0.210369, 0.207565, 0.21415199999999998, 0.211946]
args= 8 FFI empty kernel         median=3.3658 samples=[3.483774, 3.397041, 3.2601750000000003, 3.38387, 3.339925, 3.390374, 3.339131, 3.365791, 3.35833]
args= 8 INTJ static_compile kernel median=3.0349 samples=[3.053093, 2.889815, 3.2516239999999996, 2.979331, 3.0557060000000003, 2.9997399999999996, 3.043825, 2.989457, 3.034893]
args= 8 INTJ runtime_shim kernel median=2.9540 samples=[3.0619050000000003, 2.881466, 2.9540189999999997, 2.898502, 11.011467, 2.890897, 2.96414, 2.886624, 2.978327]
args=16 FFI packed nop           median=0.2939 samples=[0.289968, 0.297055, 0.29821800000000004, 0.293933, 0.5464779999999999, 0.29095, 0.294711, 0.29221600000000003, 0.290714]
args=16 FFI typed nop            median=0.3030 samples=[0.301642, 0.304899, 0.307199, 0.307022, 0.588108, 0.300632, 0.30077699999999996, 0.30220199999999997, 0.303031]
args=16 FFI empty kernel         median=3.6720 samples=[3.718803, 3.690705, 3.523873, 3.672047, 17.627339, 3.674612, 3.530425, 3.625971, 3.5245450000000003]
args=16 INTJ static_compile kernel median=3.3905 samples=[3.253188, 3.390514, 3.691564, 3.306492, 6.7854589999999995, 3.28107, 3.436324, 3.32126, 3.4231030000000002]
args=16 INTJ runtime_shim kernel median=3.2603 samples=[3.260321, 3.0831370000000002, 3.291778, 3.0336849999999997, 3.362003, 3.040547, 3.298626, 3.0428290000000002, 3.326196]
args=32 FFI packed nop           median=0.4780 samples=[0.478271, 0.477974, 0.488139, 0.48101900000000003, 0.491106, 0.475626, 0.472108, 0.47095299999999995, 0.473271]
args=32 FFI typed nop            median=0.4948 samples=[0.494193, 0.510029, 0.495712, 0.494816, 0.510336, 0.495143, 0.490999, 0.493502, 0.490641]
args=32 FFI empty kernel         median=4.2342 samples=[4.2988360000000005, 4.243937, 4.131488999999999, 4.234169, 4.187316, 4.269894, 4.103839, 4.243182, 4.102131999999999]
args=32 INTJ static_compile kernel median=3.6206 samples=[3.43709, 3.608692, 3.659906, 3.5518539999999996, 3.710795, 3.6206419999999997, 3.665133, 3.5635079999999997, 3.64683]
args=32 INTJ runtime_shim kernel median=3.5143 samples=[3.4773739999999997, 3.507254, 3.6134749999999998, 3.4626900000000003, 3.687962, 3.5142710000000004, 3.636502, 3.4607979999999996, 3.618884]
args=64 FFI packed nop           median=0.8402 samples=[0.8446119999999999, 0.83657, 0.840212, 0.871191, 0.864792, 0.834917, 0.843178, 0.83102, 0.838053]
args=64 FFI typed nop            median=0.8719 samples=[0.884079, 0.8719349999999999, 0.884127, 0.8945919999999999, 0.887865, 0.8634080000000001, 0.868598, 0.861177, 0.856481]
args=64 FFI empty kernel         median=5.0838 samples=[5.1087560000000005, 5.055899999999999, 5.083777, 5.08583, 5.153944999999999, 5.109596, 5.049627999999999, 5.076038, 5.0488040000000005]
args=64 INTJ static_compile kernel median=4.6292 samples=[4.638457000000001, 4.5620519999999996, 4.60508, 4.61317, 4.642755, 4.629218, 4.648026, 4.659422999999999, 4.604541]
args=64 INTJ runtime_shim kernel median=4.5778 samples=[4.552534, 4.596106, 4.577754, 4.630767, 4.546271, 4.609229999999999, 4.573016, 4.628543, 4.567121]
exit_status=0
ended=2026-09-25T18:39:18.409461+00:00
```

### ffi_kwargs: candidate r3

```text
commit=c5295b18f29a15fc1184eb7e93dbc099e6b75bf2
started=2026-09-25T18:39:35.760149+00:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/callable_grid
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/callable_grid taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_intj_ffi_paths.py --mode=kwargs --iters 10000 --batches 9
iters=10000 batches=9 unit=ns/call; no_gpu=True
mode=kwargs
INTJ direct positional              median_ns=67.8 samples_ns=[67.3934, 68.539, 67.4292, 67.7911, 67.9799, 68.4646, 67.2712, 67.7807, 67.1693]
INTJ adapter positional             median_ns=93.9 samples_ns=[91.3379, 93.9321, 89.7814, 94.0834, 93.8888, 95.4446, 92.9257, 94.1409, 91.7889]
INTJ adapter kwargs                 median_ns=106.7 samples_ns=[106.6758, 106.4203, 105.4148, 109.2527, 107.251, 108.0678, 104.26, 108.1986, 106.2014]
INTJ adapter defaults               median_ns=90.9 samples_ns=[92.3822, 90.849, 89.4113, 91.4242, 90.1718, 90.8632, 89.6852, 91.6864, 94.116]
INTJ FFI wrapper positional         median_ns=104.6 samples_ns=[103.858, 104.8181, 103.5084, 104.6387, 103.5306, 105.5096, 104.0757, 111.35, 108.1993]
INTJ FFI wrapper kwargs             median_ns=125.6 samples_ns=[125.1568, 126.5954, 124.9154, 127.9294, 125.599, 127.2195, 124.5762, 126.432, 124.4674]
INTJ native kwargs: unsupported (verified TypeError)
exit_status=0
ended=2026-09-25T18:39:38.656956+00:00
```

### ffi_kwargs: develop r3

```text
commit=1bf689b37137e373b6baecf241a8d8a7da23630b
started=2026-09-25T18:39:38.659527+00:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_intj_ffi_paths.py --mode=kwargs --iters 10000 --batches 9
iters=10000 batches=9 unit=ns/call; no_gpu=True
mode=kwargs
INTJ direct positional              median_ns=67.7 samples_ns=[67.4235, 68.3618, 68.5414, 67.4404, 67.7079, 68.8041, 67.7422, 69.0346, 67.6012]
INTJ adapter positional             median_ns=91.1 samples_ns=[91.9637, 92.0161, 89.9947, 90.861, 90.9508, 92.0694, 91.0866, 91.9347, 89.6671]
INTJ adapter kwargs                 median_ns=107.2 samples_ns=[106.8477, 107.1764, 109.5655, 112.9237, 108.7565, 107.4459, 106.9866, 106.9796, 106.0464]
INTJ adapter defaults               median_ns=90.5 samples_ns=[89.0251, 91.1712, 89.6736, 90.4968, 90.2628, 91.1888, 93.1334, 93.3649, 90.0649]
INTJ FFI wrapper positional         median_ns=104.1 samples_ns=[106.7616, 105.6571, 103.6031, 104.3309, 103.5754, 104.1295, 103.102, 107.8577, 103.2763]
INTJ FFI wrapper kwargs             median_ns=123.7 samples_ns=[124.6517, 123.3945, 121.8622, 125.4949, 123.7095, 125.3134, 123.3856, 128.1074, 122.9105]
INTJ native kwargs: unsupported (verified TypeError)
exit_status=0
ended=2026-09-25T18:39:41.531363+00:00
```

### ffi_dataclass: candidate r3

```text
commit=c5295b18f29a15fc1184eb7e93dbc099e6b75bf2
started=2026-09-25T18:39:59.948126+00:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/callable_grid
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/callable_grid taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_intj_ffi_paths.py --mode=dataclass --iters 10000 --batches 9
iters=10000 batches=9 unit=ns/call; no_gpu=True
mode=dataclass
INTJ pair direct                    median_ns=70.7 samples_ns=[69.9638, 70.6552, 72.0767, 77.1963, 69.3324, 72.3235, 69.6485, 71.1064, 69.3469]
INTJ pair prebuilt *tuple           median_ns=142.6 samples_ns=[145.0232, 145.9655, 142.1376, 145.9164, 142.4838, 143.1189, 140.6052, 142.5566, 139.9789]
FFI unpack Pair only                median_ns=164.7 samples_ns=[164.6842, 165.7657, 163.1042, 164.6972, 164.0911, 166.1855, 188.1681, 164.1081, 164.0876]
INTJ pair manual unpack             median_ns=177.4 samples_ns=[175.8477, 178.8398, 176.7231, 178.0744, 183.9036, 177.6254, 175.8135, 177.3862, 174.9377]
INTJ pair FFI unpack                median_ns=317.1 samples_ns=[316.711, 322.9592, 317.1052, 321.5135, 315.4668, 320.3393, 315.4964, 321.4291, 315.9237]
INTJ pair stdlib astuple            median_ns=1169.2 samples_ns=[1178.3837, 1175.1429, 1160.5167, 1170.223, 1151.2429, 1171.2414, 1157.1224, 1169.1621, 1155.1309]
INTJ config direct                  median_ns=84.9 samples_ns=[81.9248, 84.1711, 90.8055, 85.374, 83.2362, 85.3527, 84.8501, 86.0049, 83.6695]
INTJ config FFI unpack              median_ns=328.2 samples_ns=[329.1466, 329.1714, 327.0664, 328.7526, 327.2027, 329.6934, 327.3089, 328.1961, 327.1352]
INTJ native dataclass unpack: unsupported (verified TypeError)
exit_status=0
ended=2026-09-25T18:40:03.020969+00:00
```

### ffi_dataclass: develop r3

```text
commit=1bf689b37137e373b6baecf241a8d8a7da23630b
started=2026-09-25T18:40:03.023497+00:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_intj_ffi_paths.py --mode=dataclass --iters 10000 --batches 9
iters=10000 batches=9 unit=ns/call; no_gpu=True
mode=dataclass
INTJ pair direct                    median_ns=69.3 samples_ns=[70.2049, 69.4805, 68.8105, 69.4449, 69.2724, 68.8655, 68.2349, 69.2995, 67.839]
INTJ pair prebuilt *tuple           median_ns=144.1 samples_ns=[143.7457, 146.0401, 144.8413, 145.7115, 144.082, 142.8417, 141.8859, 144.5217, 141.9161]
FFI unpack Pair only                median_ns=166.6 samples_ns=[164.9161, 166.6681, 165.505, 171.3017, 170.1709, 170.0336, 166.0735, 166.6413, 165.6174]
INTJ pair manual unpack             median_ns=180.8 samples_ns=[187.9605, 178.9693, 176.4462, 178.5922, 177.0595, 181.8617, 189.633, 180.8245, 191.1258]
INTJ pair FFI unpack                median_ns=319.3 samples_ns=[320.2666, 320.2434, 317.7772, 320.6328, 318.3566, 319.4869, 318.6192, 319.3066, 318.7356]
INTJ pair stdlib astuple            median_ns=1165.5 samples_ns=[1191.4277, 1163.8591, 1151.2493, 1165.4526, 1154.4491, 1168.3669, 1165.6434, 1166.1325, 1164.6043]
INTJ config direct                  median_ns=84.6 samples_ns=[83.2078, 91.5909, 85.1039, 85.7521, 83.4605, 84.596, 85.1211, 82.0304, 82.8934]
INTJ config FFI unpack              median_ns=326.4 samples_ns=[333.8568, 326.3743, 321.5763, 326.414, 322.8835, 324.6023, 332.6346, 323.7273, 334.47]
INTJ native dataclass unpack: unsupported (verified TypeError)
exit_status=0
ended=2026-09-25T18:40:06.054429+00:00
```

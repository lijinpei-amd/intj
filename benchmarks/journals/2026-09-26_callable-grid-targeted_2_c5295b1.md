# Callable-grid targeted ABBA controls — 2026-09-26, repeat 2 of 4

[Aggregate, environment, commands, and caveats](2026-09-26_callable-grid-targeted_0_c5295b1.md).
These are the six process outputs for targeted repeat 2, in actual
D2 → C2 order within
each mode. They include nine-batch samples, absolute commands, UTC times,
commits, and exit status. The local date is 2026-09-26 (Asia/Shanghai).

## Raw outputs — repeat 2

### ffi_sweep: develop r2

```text
commit=1bf689b37137e373b6baecf241a8d8a7da23630b
started=2026-09-25T18:39:00.533475+00:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_ffi_compare.py --sweep --iters 1000 --batches 9
counts=[0, 3, 5, 8, 16, 32, 64] iters=1000 batches=9 unit=us/call (host-only, no timed sync); INTJ device-bound, no specialization key
args= 0 FFI packed nop           median=0.1170 samples=[0.121217, 0.11653400000000001, 0.120667, 0.11680800000000001, 0.116235, 0.114912, 0.117018, 0.11768200000000001, 0.117348]
args= 0 FFI typed nop            median=0.1205 samples=[0.120839, 0.120921, 0.119495, 0.122422, 0.117931, 0.12138, 0.118749, 0.12047799999999999, 0.119859]
args= 0 FFI empty kernel         median=1.6213 samples=[1.6180139999999998, 1.904645, 1.6213499999999998, 1.9081590000000002, 1.5900139999999998, 1.9045519999999998, 1.5937029999999999, 1.9238989999999998, 1.582785]
args= 0 INTJ static_compile kernel median=2.7608 samples=[3.097817, 2.779852, 2.723815, 2.7607869999999997, 2.709992, 2.767688, 2.703143, 2.769134, 2.7000439999999997]
args= 0 INTJ runtime_shim kernel median=2.7021 samples=[2.856577, 2.670238, 2.715045, 2.662884, 2.7047399999999997, 2.664089, 2.71478, 2.649872, 2.702111]
args= 3 FFI packed nop           median=0.1444 samples=[0.14994200000000002, 0.143476, 0.144254, 0.142911, 0.144371, 0.145403, 0.147543, 0.145987, 0.144091]
args= 3 FFI typed nop            median=0.1487 samples=[0.149309, 0.14683500000000002, 0.147084, 0.151023, 0.14922, 0.152284, 0.14702199999999999, 0.148739, 0.146012]
args= 3 FFI empty kernel         median=3.2694 samples=[3.302803, 3.2824940000000002, 3.1673649999999998, 3.269393, 3.162341, 3.313284, 3.147799, 3.2933820000000003, 3.1588789999999998]
args= 3 INTJ static_compile kernel median=2.9427 samples=[2.950859, 2.820402, 2.9427119999999998, 2.881875, 3.0430770000000003, 2.8784490000000003, 3.049011, 2.871554, 3.063564]
args= 3 INTJ runtime_shim kernel median=2.8104 samples=[2.990219, 2.746893, 2.783242, 2.784261, 2.889519, 2.810438, 2.870212, 2.777999, 2.864325]
args= 5 FFI packed nop           median=0.1733 samples=[0.175757, 0.17251499999999997, 0.172987, 0.173344, 0.174157, 0.17296899999999998, 0.174822, 0.173009, 0.17602099999999998]
args= 5 FFI typed nop            median=0.1783 samples=[0.178369, 0.176706, 0.177136, 0.19492400000000001, 0.173625, 0.17834, 0.18276499999999998, 0.180542, 0.17522200000000002]
args= 5 FFI empty kernel         median=3.2407 samples=[3.3346199999999997, 3.260759, 3.189072, 3.240678, 3.2054549999999997, 3.232592, 3.2013510000000003, 3.242706, 3.340558]
args= 5 INTJ static_compile kernel median=2.8590 samples=[2.969341, 2.889581, 2.8634459999999997, 2.852826, 2.845283, 2.8432049999999998, 2.8590340000000003, 2.85475, 2.908888]
args= 5 INTJ runtime_shim kernel median=2.8621 samples=[2.9849479999999997, 2.952442, 2.827665, 2.869179, 2.836926, 2.8574699999999997, 2.817852, 2.862085, 2.883807]
args= 8 FFI packed nop           median=0.2065 samples=[0.205207, 0.205906, 0.20688499999999999, 0.206458, 0.207112, 0.20625200000000002, 0.20647, 0.21388, 0.207861]
args= 8 FFI typed nop            median=0.2128 samples=[0.205095, 0.21021700000000001, 0.213429, 0.212804, 0.209036, 0.215377, 0.20915799999999998, 0.215834, 0.213407]
args= 8 FFI empty kernel         median=3.3856 samples=[3.469088, 3.444066, 3.263487, 3.401714, 3.2850520000000003, 3.385624, 3.490478, 3.365163, 3.3364659999999997]
args= 8 INTJ static_compile kernel median=3.0622 samples=[3.062206, 2.969859, 3.149143, 2.972627, 3.169458, 2.92758, 3.14176, 3.036761, 3.161549]
args= 8 INTJ runtime_shim kernel median=2.8838 samples=[3.0323409999999997, 2.955702, 2.859174, 2.8838429999999997, 3.000915, 2.889712, 2.877208, 2.8829740000000004, 2.869835]
args=16 FFI packed nop           median=0.2941 samples=[0.294116, 0.297399, 0.29485700000000004, 0.293302, 0.290266, 0.293668, 0.298168, 0.293534, 0.30011200000000005]
args=16 FFI typed nop            median=0.3056 samples=[0.30235700000000004, 0.310247, 0.30205200000000004, 0.30677499999999996, 0.315073, 0.302331, 0.302888, 0.309639, 0.30555200000000005]
args=16 FFI empty kernel         median=3.6590 samples=[3.738272, 3.755599, 3.5653249999999996, 3.645533, 3.581445, 3.659865, 3.659038, 3.869633, 3.539526]
args=16 INTJ static_compile kernel median=3.3886 samples=[3.238998, 3.3885880000000004, 3.456261, 3.2586709999999997, 3.444414, 3.278947, 3.430281, 3.276043, 3.441506]
args=16 INTJ runtime_shim kernel median=3.2420 samples=[3.24197, 3.1866120000000002, 3.3402399999999997, 3.023594, 3.33896, 3.0375639999999997, 3.330655, 3.030237, 3.33359]
args=32 FFI packed nop           median=0.4824 samples=[0.477142, 0.49492899999999995, 0.474889, 0.472154, 0.482362, 0.472685, 0.49122899999999997, 0.491825, 0.48913799999999996]
args=32 FFI typed nop            median=0.4941 samples=[0.494136, 0.50527, 0.48757999999999996, 0.493276, 0.496527, 0.490329, 0.48556099999999996, 0.502316, 0.496719]
args=32 FFI empty kernel         median=4.0938 samples=[4.187165, 4.142258, 3.943616, 4.093844, 3.982342, 4.137694000000001, 3.9637029999999998, 4.120027, 3.956613]
args=32 INTJ static_compile kernel median=3.6644 samples=[3.526795, 3.786705, 3.6644229999999998, 3.59655, 3.692412, 3.601646, 3.6904079999999997, 3.618697, 3.7193989999999997]
args=32 INTJ runtime_shim kernel median=3.5854 samples=[3.4767330000000003, 3.585374, 3.624641, 3.457697, 3.677513, 3.466473, 3.65762, 3.442176, 3.644803]
args=64 FFI packed nop           median=0.8317 samples=[0.820288, 0.831697, 0.838869, 0.835071, 0.837683, 0.826265, 0.825303, 0.8471609999999999, 0.821922]
args=64 FFI typed nop            median=0.8522 samples=[0.844979, 0.874599, 0.848682, 0.87926, 0.852229, 0.8525159999999999, 0.845963, 0.8545349999999999, 0.8472540000000001]
args=64 FFI empty kernel         median=5.0489 samples=[5.073167000000001, 5.096316, 5.021586, 5.064081, 5.058993999999999, 5.048931, 5.036842, 5.022179, 4.994443]
args=64 INTJ static_compile kernel median=4.6046 samples=[4.744260000000001, 4.6148299999999995, 4.581843, 4.594532, 4.603172, 4.604555, 4.598596, 4.628103, 4.620972]
args=64 INTJ runtime_shim kernel median=4.5966 samples=[4.689411, 4.596604, 4.546533999999999, 4.613142, 4.545567, 4.612654, 4.5669949999999995, 4.599053, 4.562676000000001]
exit_status=0
ended=2026-09-25T18:39:04.909906+00:00
```

### ffi_sweep: candidate r2

```text
commit=c5295b18f29a15fc1184eb7e93dbc099e6b75bf2
started=2026-09-25T18:39:04.912462+00:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/callable_grid
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/callable_grid taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_ffi_compare.py --sweep --iters 1000 --batches 9
counts=[0, 3, 5, 8, 16, 32, 64] iters=1000 batches=9 unit=us/call (host-only, no timed sync); INTJ device-bound, no specialization key
args= 0 FFI packed nop           median=0.1178 samples=[0.121003, 0.11729600000000001, 0.171429, 0.118282, 0.118259, 0.114949, 0.11597299999999999, 0.11783, 0.11748900000000001]
args= 0 FFI typed nop            median=0.1197 samples=[0.117045, 0.117357, 0.159972, 0.123693, 0.11866800000000001, 0.11873, 0.11996599999999999, 0.12142499999999999, 0.11968899999999999]
args= 0 FFI empty kernel         median=2.7484 samples=[1.688251, 6.255565, 5.959643, 2.9184479999999997, 2.05952, 2.748393, 2.095053, 2.805437, 2.0738879999999997]
args= 0 INTJ static_compile kernel median=3.2569 samples=[3.34127, 4.880534, 5.366569999999999, 3.261478, 3.25688, 3.230786, 3.225759, 3.200848, 3.222651]
args= 0 INTJ runtime_shim kernel median=3.2608 samples=[3.1758409999999997, 5.93627, 3.308034, 3.04407, 3.260846, 3.025439, 3.333548, 3.016142, 3.284144]
args= 3 FFI packed nop           median=0.1431 samples=[0.142923, 0.142346, 0.145845, 0.142579, 0.142298, 0.146061, 0.143715, 0.144368, 0.143107]
args= 3 FFI typed nop            median=0.1488 samples=[0.146353, 0.14712799999999998, 0.146178, 0.14946, 0.14527400000000001, 0.149763, 0.148772, 0.151262, 0.151779]
args= 3 FFI empty kernel         median=3.7985 samples=[3.720917, 4.321772, 3.605069, 3.8210830000000002, 3.7000279999999997, 3.803781, 4.813103, 3.798536, 3.6852600000000004]
args= 3 INTJ static_compile kernel median=3.4861 samples=[3.283065, 3.48739, 4.013047, 3.5077, 3.4861109999999997, 3.4708870000000003, 5.5232399999999995, 3.4551439999999998, 3.484276]
args= 3 INTJ runtime_shim kernel median=3.3648 samples=[3.279367, 3.100664, 3.497207, 3.3647739999999997, 3.4665529999999998, 3.364389, 14.224207, 3.338684, 4.025893]
args= 5 FFI packed nop           median=0.1755 samples=[0.17840899999999998, 0.17547900000000002, 0.174882, 0.175185, 0.175707, 0.17416800000000002, 0.383185, 0.17546799999999999, 0.17633000000000001]
args= 5 FFI typed nop            median=0.1793 samples=[0.175935, 0.179735, 0.17952500000000002, 0.177201, 0.193292, 0.178936, 0.408282, 0.179301, 0.174445]
args= 5 FFI empty kernel         median=4.2214 samples=[3.7153359999999997, 4.2617449999999995, 3.806542, 4.282875, 3.723022, 4.325444999999999, 7.893681, 4.2214409999999996, 3.718792]
args= 5 INTJ static_compile kernel median=3.4507 samples=[3.3022, 3.4730079999999997, 3.507624, 3.4430810000000003, 3.39859, 3.450726, 3.4746799999999998, 3.413245, 3.47896]
args= 5 INTJ runtime_shim kernel median=3.3894 samples=[3.3473800000000002, 3.463944, 4.123823000000001, 3.389405, 3.505541, 3.375711, 3.3742460000000003, 3.360335, 3.49971]
args= 8 FFI packed nop           median=0.2056 samples=[0.20563499999999998, 0.205044, 0.20396799999999998, 0.202797, 0.207984, 0.20392, 0.20809, 0.20629, 0.20774600000000001]
args= 8 FFI typed nop            median=0.2097 samples=[0.212088, 0.20869300000000002, 0.20582499999999998, 0.209347, 0.21154699999999999, 0.209718, 0.210907, 0.209606, 0.212827]
args= 8 FFI empty kernel         median=3.9343 samples=[3.934292, 4.3530169999999995, 3.834075, 4.311279000000001, 3.7992199999999996, 4.339488, 3.8191610000000003, 4.291192, 3.8092550000000003]
args= 8 INTJ static_compile kernel median=3.5559 samples=[3.385882, 3.555866, 4.10008, 3.638253, 3.555295, 3.635772, 3.5409119999999996, 3.6421889999999997, 3.537666]
args= 8 INTJ runtime_shim kernel median=3.4091 samples=[3.409076, 3.267551, 3.615698, 3.274829, 3.560104, 3.25595, 3.595185, 3.259404, 3.596104]
args=16 FFI packed nop           median=0.2943 samples=[0.294321, 0.294965, 0.290674, 0.293273, 0.292319, 0.294196, 0.299263, 0.29512299999999997, 0.298506]
args=16 FFI typed nop            median=0.3041 samples=[0.300677, 0.304572, 0.299848, 0.299203, 0.305497, 0.30406700000000003, 0.30239699999999997, 0.312529, 0.304439]
args=16 FFI empty kernel         median=4.2323 samples=[4.232279, 5.722265, 4.1593599999999995, 5.587549, 4.066166, 5.714197, 4.150577, 5.588446, 4.108792]
args=16 INTJ static_compile kernel median=5.3525 samples=[3.770541, 5.352539, 5.5875, 5.350275, 5.687467, 5.087695, 5.833747000000001, 5.280774, 5.931523]
args=16 INTJ runtime_shim kernel median=3.7891 samples=[3.7891280000000003, 3.628078, 4.935955, 3.597579, 5.471973, 3.580374, 5.501065, 3.573465, 5.5307640000000005]
args=32 FFI packed nop           median=0.4819 samples=[0.500424, 0.473129, 0.48189499999999996, 0.477634, 0.469206, 0.483854, 0.47824500000000003, 0.494014, 0.49047500000000005]
args=32 FFI typed nop            median=0.4957 samples=[0.495288, 0.494239, 0.485243, 0.49573500000000004, 0.48786799999999997, 0.496859, 0.51313, 0.50574, 0.501288]
args=32 FFI empty kernel         median=4.8937 samples=[4.893738999999999, 5.819401, 4.823015000000001, 5.802933, 4.796305, 5.9087950000000005, 4.797248, 5.831926999999999, 4.796165]
args=32 INTJ static_compile kernel median=5.4418 samples=[4.052239, 5.4073459999999995, 5.567727, 5.411187, 5.684279999999999, 5.441839, 5.660529, 5.352019, 5.8120829999999994]
args=32 INTJ runtime_shim kernel median=4.0685 samples=[3.944297, 4.056514, 5.436083, 4.032134, 5.395328999999999, 4.068475, 5.5920749999999995, 4.044225, 5.439741]
args=64 FFI packed nop           median=0.8506 samples=[0.823905, 0.855673, 0.855307, 0.818297, 0.8593310000000001, 0.825043, 0.8505689999999999, 0.833927, 0.8563200000000001]
args=64 FFI typed nop            median=0.8629 samples=[0.846336, 0.8740739999999999, 0.8543010000000001, 0.8751169999999999, 0.846802, 0.8548049999999999, 0.8629199999999999, 0.879793, 0.876681]
args=64 FFI empty kernel         median=5.8508 samples=[5.718534, 6.357381999999999, 5.836792, 6.405389, 5.850805, 6.334161, 5.815639, 6.357400999999999, 5.8314070000000005]
args=64 INTJ static_compile kernel median=5.9864 samples=[5.265843, 5.933589, 5.867979999999999, 6.032525, 5.950793, 6.049681, 5.9863800000000005, 6.0086189999999995, 5.998520999999999]
args=64 INTJ runtime_shim kernel median=5.9277 samples=[5.331746, 5.927685, 5.913043, 5.950279999999999, 5.850607, 6.019247999999999, 5.8974470000000005, 5.934683, 5.939376]
exit_status=0
ended=2026-09-25T18:39:09.535870+00:00
```

### ffi_kwargs: develop r2

```text
commit=1bf689b37137e373b6baecf241a8d8a7da23630b
started=2026-09-25T18:39:29.974483+00:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_intj_ffi_paths.py --mode=kwargs --iters 10000 --batches 9
iters=10000 batches=9 unit=ns/call; no_gpu=True
mode=kwargs
INTJ direct positional              median_ns=67.8 samples_ns=[68.0049, 68.2099, 67.35, 67.7082, 67.4226, 68.1671, 67.835, 68.2603, 67.096]
INTJ adapter positional             median_ns=91.7 samples_ns=[91.6716, 95.7294, 92.7756, 93.9692, 90.5322, 94.3668, 90.6285, 91.6249, 90.2146]
INTJ adapter kwargs                 median_ns=106.7 samples_ns=[107.2083, 106.4039, 107.5374, 106.7052, 106.403, 106.5624, 107.3569, 106.9198, 105.3394]
INTJ adapter defaults               median_ns=90.4 samples_ns=[89.7878, 91.8242, 89.5188, 91.2208, 90.3253, 91.1299, 90.4148, 91.2357, 89.9234]
INTJ FFI wrapper positional         median_ns=104.6 samples_ns=[103.9057, 106.1688, 105.485, 105.7809, 103.4209, 104.6599, 102.4365, 104.595, 103.6345]
INTJ FFI wrapper kwargs             median_ns=124.4 samples_ns=[124.541, 125.7546, 123.1657, 126.22, 122.8624, 124.7464, 122.5587, 124.4337, 121.0821]
INTJ native kwargs: unsupported (verified TypeError)
exit_status=0
ended=2026-09-25T18:39:32.929138+00:00
```

### ffi_kwargs: candidate r2

```text
commit=c5295b18f29a15fc1184eb7e93dbc099e6b75bf2
started=2026-09-25T18:39:32.931674+00:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/callable_grid
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/callable_grid taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_intj_ffi_paths.py --mode=kwargs --iters 10000 --batches 9
iters=10000 batches=9 unit=ns/call; no_gpu=True
mode=kwargs
INTJ direct positional              median_ns=103.2 samples_ns=[67.068, 68.1356, 67.2271, 67.4697, 103.1923, 129.115, 120.812, 145.1682, 208.9785]
INTJ adapter positional             median_ns=91.1 samples_ns=[213.7678, 161.3803, 90.5153, 91.0405, 90.3709, 91.0964, 90.2441, 91.404, 91.478]
INTJ adapter kwargs                 median_ns=107.0 samples_ns=[106.9796, 107.0457, 106.8961, 108.4117, 105.5092, 108.771, 105.3183, 107.0895, 111.5446]
INTJ adapter defaults               median_ns=90.6 samples_ns=[90.6588, 90.592, 90.9961, 91.1768, 89.8997, 89.9509, 89.1582, 91.1162, 89.4712]
INTJ FFI wrapper positional         median_ns=104.3 samples_ns=[105.2645, 105.865, 103.8975, 106.2651, 104.3238, 105.3987, 103.7029, 103.9433, 103.0126]
INTJ FFI wrapper kwargs             median_ns=124.7 samples_ns=[123.5718, 125.6056, 124.3898, 126.3233, 124.1422, 129.6173, 124.651, 126.1839, 124.5889]
INTJ native kwargs: unsupported (verified TypeError)
exit_status=0
ended=2026-09-25T18:39:35.757603+00:00
```

### ffi_dataclass: develop r2

```text
commit=1bf689b37137e373b6baecf241a8d8a7da23630b
started=2026-09-25T18:39:53.789701+00:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_intj_ffi_paths.py --mode=dataclass --iters 10000 --batches 9
iters=10000 batches=9 unit=ns/call; no_gpu=True
mode=dataclass
INTJ pair direct                    median_ns=69.5 samples_ns=[69.7816, 70.5889, 69.1396, 71.7513, 68.3323, 69.5118, 69.2565, 70.2097, 68.8474]
INTJ pair prebuilt *tuple           median_ns=144.9 samples_ns=[144.916, 147.4341, 146.7975, 147.3377, 142.9364, 146.1885, 141.9849, 144.6756, 142.3877]
FFI unpack Pair only                median_ns=166.7 samples_ns=[164.0743, 171.4392, 164.6446, 169.585, 165.4775, 167.2521, 164.7565, 168.2939, 166.671]
INTJ pair manual unpack             median_ns=181.4 samples_ns=[189.4137, 181.4445, 178.3045, 183.8874, 178.5532, 180.5385, 189.7131, 178.4607, 190.3059]
INTJ pair FFI unpack                median_ns=319.1 samples_ns=[319.3, 319.7722, 316.8344, 320.6538, 318.6273, 321.1653, 317.6345, 319.1085, 318.0894]
INTJ pair stdlib astuple            median_ns=1179.3 samples_ns=[1204.3945, 2305.0047, 1173.6368, 1188.9611, 1170.8956, 1174.3012, 1172.4935, 1183.8526, 1179.3008]
INTJ config direct                  median_ns=83.8 samples_ns=[84.3675, 85.0046, 83.7509, 85.0478, 83.5283, 84.5499, 82.5022, 82.7976, 82.1751]
INTJ config FFI unpack              median_ns=327.4 samples_ns=[335.0689, 325.2099, 324.101, 325.7437, 327.4006, 327.68, 335.7554, 325.2198, 333.7434]
INTJ native dataclass unpack: unsupported (verified TypeError)
exit_status=0
ended=2026-09-25T18:39:56.818121+00:00
```

### ffi_dataclass: candidate r2

```text
commit=c5295b18f29a15fc1184eb7e93dbc099e6b75bf2
started=2026-09-25T18:39:56.820671+00:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/callable_grid
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/callable_grid taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_intj_ffi_paths.py --mode=dataclass --iters 10000 --batches 9
iters=10000 batches=9 unit=ns/call; no_gpu=True
mode=dataclass
INTJ pair direct                    median_ns=69.5 samples_ns=[68.5305, 70.4782, 69.4652, 68.8857, 70.1125, 70.2367, 69.5461, 73.6321, 69.4298]
INTJ pair prebuilt *tuple           median_ns=140.2 samples_ns=[139.3599, 146.0396, 139.3007, 143.1182, 139.1001, 142.7402, 138.8351, 141.899, 140.1801]
FFI unpack Pair only                median_ns=166.6 samples_ns=[166.2424, 167.5567, 167.2595, 167.7108, 165.2864, 166.5628, 165.2267, 166.8476, 166.0182]
INTJ pair manual unpack             median_ns=177.0 samples_ns=[176.2062, 178.5766, 175.3361, 179.023, 176.9956, 177.379, 174.0754, 178.0288, 171.2396]
INTJ pair FFI unpack                median_ns=316.9 samples_ns=[315.4574, 317.8545, 315.7286, 318.4368, 316.9483, 318.5029, 314.4308, 318.7798, 312.9001]
INTJ pair stdlib astuple            median_ns=1175.4 samples_ns=[1175.3809, 1189.5079, 1154.7943, 1184.9017, 1153.1391, 1166.4159, 1292.2962, 1177.6647, 1157.847]
INTJ config direct                  median_ns=84.1 samples_ns=[83.905, 91.3652, 83.5636, 84.5182, 84.142, 83.3965, 82.2817, 84.559, 84.5734]
INTJ config FFI unpack              median_ns=327.2 samples_ns=[326.6428, 330.9785, 327.186, 329.8418, 326.2658, 329.4341, 325.7563, 329.7808, 325.8189]
INTJ native dataclass unpack: unsupported (verified TypeError)
exit_status=0
ended=2026-09-25T18:39:59.945470+00:00
```

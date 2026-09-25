# Callable-grid targeted ABBA controls — 2026-09-26, repeat 1 of 4

[Aggregate, environment, commands, and caveats](2026-09-26_callable-grid-targeted_0_c5295b1.md).
These are the six process outputs for targeted repeat 1, in actual
C1 → D1 order within
each mode. They include nine-batch samples, absolute commands, UTC times,
commits, and exit status. The local date is 2026-09-26 (Asia/Shanghai).

## Raw outputs — repeat 1

### ffi_sweep: candidate r1

```text
commit=c5295b18f29a15fc1184eb7e93dbc099e6b75bf2
started=2026-09-25T18:38:51.657015+00:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/callable_grid
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/callable_grid taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_ffi_compare.py --sweep --iters 1000 --batches 9
counts=[0, 3, 5, 8, 16, 32, 64] iters=1000 batches=9 unit=us/call (host-only, no timed sync); INTJ device-bound, no specialization key
args= 0 FFI packed nop           median=0.1165 samples=[0.12035599999999999, 0.116517, 0.115524, 0.115453, 0.117309, 0.11739400000000001, 0.11848600000000001, 0.11392000000000001, 0.115517]
args= 0 FFI typed nop            median=0.1193 samples=[0.122524, 0.120128, 0.118769, 0.12083100000000001, 0.119787, 0.1193, 0.11893300000000001, 0.118271, 0.117283]
args= 0 FFI empty kernel         median=1.6461 samples=[1.639387, 1.862424, 1.595065, 1.874836, 1.573482, 1.868037, 1.6460599999999999, 1.890771, 1.593854]
args= 0 INTJ static_compile kernel median=2.7125 samples=[3.090657, 2.6724050000000004, 2.7236, 2.8113539999999997, 2.701321, 2.641924, 2.763506, 2.712485, 2.704401]
args= 0 INTJ runtime_shim kernel median=2.7492 samples=[6.605486, 2.731959, 2.749176, 2.692109, 2.733023, 2.750314, 2.8199929999999997, 2.876606, 2.7240230000000003]
args= 3 FFI packed nop           median=0.1448 samples=[0.239224, 0.144274, 0.144899, 0.144832, 0.144651, 0.144583, 0.14452600000000002, 0.145382, 0.145534]
args= 3 FFI typed nop            median=0.1481 samples=[0.261032, 0.14804499999999998, 0.14746, 0.147181, 0.14958000000000002, 0.15120699999999998, 0.14664, 0.148698, 0.148126]
args= 3 FFI empty kernel         median=3.2167 samples=[18.783322000000002, 3.21667, 3.1678, 3.226467, 3.144122, 3.236079, 3.155067, 3.234866, 3.163366]
args= 3 INTJ static_compile kernel median=2.9269 samples=[2.926917, 2.8701280000000002, 2.96841, 2.8563, 2.968377, 2.8687910000000003, 2.953291, 2.900274, 2.978566]
args= 3 INTJ runtime_shim kernel median=2.8878 samples=[2.9634940000000003, 2.8787469999999997, 2.887837, 2.864887, 2.898181, 3.083192, 2.786758, 2.891675, 2.868722]
args= 5 FFI packed nop           median=0.1751 samples=[0.175588, 0.172809, 0.175619, 0.17389500000000002, 0.173174, 0.17612, 0.17326499999999997, 0.175059, 0.177604]
args= 5 FFI typed nop            median=0.1786 samples=[0.184256, 0.180322, 0.175031, 0.177426, 0.18171700000000002, 0.178594, 0.17663900000000002, 0.17946, 0.177088]
args= 5 FFI empty kernel         median=3.2802 samples=[3.496125, 3.270462, 3.2802420000000003, 3.2991449999999998, 3.19355, 3.3239720000000004, 3.207465, 3.4409430000000003, 3.2676399999999997]
args= 5 INTJ static_compile kernel median=2.8875 samples=[2.994767, 2.903192, 2.8874920000000004, 2.890877, 2.8577489999999997, 2.84456, 2.8776230000000003, 2.926113, 2.882946]
args= 5 INTJ runtime_shim kernel median=2.8595 samples=[3.010351, 2.975877, 2.865492, 2.7634119999999998, 2.825955, 2.7700489999999998, 2.8595010000000003, 2.947781, 2.856948]
args= 8 FFI packed nop           median=0.2065 samples=[0.205684, 0.206003, 0.209642, 0.206446, 0.206778, 0.208044, 0.208969, 0.20486500000000002, 0.20647100000000002]
args= 8 FFI typed nop            median=0.2111 samples=[0.211521, 0.215681, 0.209964, 0.21343, 0.20954599999999998, 0.21106, 0.209702, 0.210292, 0.211741]
args= 8 FFI empty kernel         median=3.3939 samples=[3.527453, 3.433874, 3.309886, 3.3939369999999998, 3.2962689999999997, 3.421877, 3.299074, 3.38571, 3.495327]
args= 8 INTJ static_compile kernel median=3.0550 samples=[3.0549749999999998, 2.9727669999999997, 3.038085, 3.011259, 3.058991, 3.06189, 3.0782979999999998, 3.196096, 3.024574]
args= 8 INTJ runtime_shim kernel median=2.9045 samples=[3.045292, 2.911103, 2.842982, 2.8774699999999998, 2.978561, 2.902698, 2.9836210000000003, 2.9045430000000003, 2.856122]
args=16 FFI packed nop           median=0.2971 samples=[0.290527, 0.30039299999999997, 0.294117, 0.29707799999999995, 0.297448, 0.295981, 0.297618, 0.29368099999999997, 0.303664]
args=16 FFI typed nop            median=0.3094 samples=[0.302752, 0.309632, 0.303187, 0.312216, 0.30941399999999997, 0.310293, 0.310311, 0.307631, 0.301074]
args=16 FFI empty kernel         median=3.6045 samples=[3.72567, 3.661889, 3.52364, 3.6045059999999998, 3.528968, 3.669913, 3.558891, 3.7035549999999997, 3.5156370000000003]
args=16 INTJ static_compile kernel median=3.3220 samples=[3.28143, 3.361081, 3.410659, 3.2731350000000003, 3.4621779999999998, 3.321963, 3.4849319999999997, 3.279833, 3.306451]
args=16 INTJ runtime_shim kernel median=3.2447 samples=[3.253625, 3.1493960000000003, 3.331223, 3.056133, 3.353157, 3.077422, 3.354045, 3.0805990000000003, 3.2447130000000004]
args=32 FFI packed nop           median=0.4890 samples=[0.47158999999999995, 0.482067, 0.49073700000000003, 0.495755, 0.485815, 0.490545, 0.489026, 0.481996, 0.490019]
args=32 FFI typed nop            median=0.5043 samples=[0.490216, 0.5058279999999999, 0.50355, 0.504543, 0.50426, 0.507472, 0.504433, 0.495624, 0.500381]
args=32 FFI empty kernel         median=4.1026 samples=[4.168499, 4.102582, 3.9498580000000003, 4.1095619999999995, 3.959962, 4.152887, 3.974257, 4.114216000000001, 3.998098]
args=32 INTJ static_compile kernel median=3.6575 samples=[3.466244, 3.628797, 3.657516, 3.5582190000000002, 3.682248, 3.594368, 3.778797, 3.6730590000000003, 3.749559]
args=32 INTJ runtime_shim kernel median=3.4570 samples=[3.433598, 3.457011, 3.591922, 3.42679, 3.6989430000000003, 3.453081, 3.62369, 3.4311689999999997, 3.638337]
args=64 FFI packed nop           median=0.8520 samples=[0.835683, 0.8826280000000001, 0.859073, 0.852019, 0.933793, 0.845146, 0.8663719999999999, 0.836854, 0.843446]
args=64 FFI typed nop            median=0.8893 samples=[0.864403, 0.902914, 0.898593, 0.889291, 1.473955, 0.880949, 0.893809, 0.872706, 0.863325]
args=64 FFI empty kernel         median=5.0898 samples=[5.072266, 5.077856, 5.091442, 5.06477, 24.929809000000002, 5.101692, 5.089805, 5.121822, 5.0442610000000005]
args=64 INTJ static_compile kernel median=4.6141 samples=[4.699251, 4.566345999999999, 4.616689999999999, 4.60458, 5.9345810000000006, 4.589902, 4.645023, 4.614063, 4.6087169999999995]
args=64 INTJ runtime_shim kernel median=4.5279 samples=[4.6404179999999995, 4.529754, 4.51292, 4.527913, 4.50221, 4.572475000000001, 4.519801, 4.608480999999999, 4.524489]
exit_status=0
ended=2026-09-25T18:38:56.103168+00:00
```

### ffi_sweep: develop r1

```text
commit=1bf689b37137e373b6baecf241a8d8a7da23630b
started=2026-09-25T18:38:56.105321+00:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_ffi_compare.py --sweep --iters 1000 --batches 9
counts=[0, 3, 5, 8, 16, 32, 64] iters=1000 batches=9 unit=us/call (host-only, no timed sync); INTJ device-bound, no specialization key
args= 0 FFI packed nop           median=0.1171 samples=[0.118764, 0.119505, 0.116425, 0.117316, 0.117069, 0.116115, 0.113448, 0.11793600000000001, 0.116789]
args= 0 FFI typed nop            median=0.1183 samples=[0.118019, 0.119093, 0.117002, 0.118862, 0.11796, 0.11617100000000001, 0.11934399999999999, 0.120452, 0.118269]
args= 0 FFI empty kernel         median=1.7390 samples=[1.589866, 1.951793, 1.739039, 1.9041210000000002, 1.615228, 1.897153, 1.6131099999999998, 1.891207, 1.677163]
args= 0 INTJ static_compile kernel median=2.6979 samples=[3.1276819999999996, 2.696834, 2.796824, 2.813145, 2.7021460000000004, 2.669631, 2.688138, 2.6604140000000003, 2.69787]
args= 0 INTJ runtime_shim kernel median=2.7029 samples=[2.8799189999999997, 2.725844, 2.803787, 2.70292, 2.7029270000000003, 2.687624, 2.701989, 2.68295, 2.6916100000000003]
args= 3 FFI packed nop           median=0.1443 samples=[0.144898, 0.143482, 0.14431899999999998, 0.143014, 0.146799, 0.145429, 0.14296799999999998, 0.144672, 0.14283600000000002]
args= 3 FFI typed nop            median=0.1491 samples=[0.151257, 0.152708, 0.14803899999999998, 0.150291, 0.145351, 0.14912299999999998, 0.144485, 0.150577, 0.14743]
args= 3 FFI empty kernel         median=3.2444 samples=[3.343799, 3.294504, 3.205181, 3.270845, 3.1795479999999996, 3.39359, 3.1651170000000004, 3.244377, 3.181901]
args= 3 INTJ static_compile kernel median=2.9405 samples=[2.9944140000000004, 2.8687579999999997, 2.962933, 2.916684, 2.981717, 2.84875, 2.94047, 2.860789, 2.959545]
args= 3 INTJ runtime_shim kernel median=2.8604 samples=[3.08507, 2.864667, 2.788404, 2.919049, 2.786548, 2.858486, 2.786672, 2.860418, 2.873029]
args= 5 FFI packed nop           median=0.1761 samples=[0.176315, 0.176072, 0.17373, 0.176363, 0.17421899999999998, 0.174442, 0.178701, 0.175788, 0.176511]
args= 5 FFI typed nop            median=0.1799 samples=[0.179274, 0.17985900000000002, 0.18115299999999998, 0.196823, 0.179232, 0.190965, 0.177893, 0.181979, 0.179095]
args= 5 FFI empty kernel         median=3.3229 samples=[3.405939, 3.357034, 3.217837, 3.3249920000000004, 3.222502, 3.322854, 3.221731, 3.3394310000000003, 3.2792440000000003]
args= 5 INTJ static_compile kernel median=2.9303 samples=[3.012723, 2.963144, 2.936692, 2.901574, 2.854356, 2.9002869999999996, 2.9453270000000003, 2.930273, 2.899682]
args= 5 INTJ runtime_shim kernel median=2.8788 samples=[3.010387, 2.878754, 2.91539, 2.897432, 2.842314, 2.767457, 2.901806, 2.773634, 2.8568890000000002]
args= 8 FFI packed nop           median=0.2119 samples=[0.211961, 0.20876, 0.211864, 0.21159899999999998, 0.21322300000000002, 0.212864, 0.21072300000000002, 0.213339, 0.20950200000000002]
args= 8 FFI typed nop            median=0.2152 samples=[0.219078, 0.211665, 0.211499, 0.215231, 0.215719, 0.21561000000000002, 0.218125, 0.21417, 0.21321]
args= 8 FFI empty kernel         median=3.4192 samples=[3.545185, 3.460386, 3.373249, 3.431437, 3.3078499999999997, 3.434483, 3.389817, 3.4191640000000003, 3.333203]
args= 8 INTJ static_compile kernel median=3.0418 samples=[3.067552, 2.9756709999999997, 3.114988, 2.972553, 3.087676, 2.9533739999999997, 3.0775680000000003, 3.019698, 3.041836]
args= 8 INTJ runtime_shim kernel median=2.9470 samples=[3.070499, 2.9469589999999997, 2.962671, 2.919664, 2.962167, 2.891256, 2.971934, 2.891536, 2.873576]
args=16 FFI packed nop           median=0.3062 samples=[0.30333, 0.301613, 0.308026, 0.303477, 0.306238, 0.30604899999999996, 0.307862, 0.31013799999999997, 0.30798000000000003]
args=16 FFI typed nop            median=0.3150 samples=[0.314587, 0.314965, 0.31456, 0.31685700000000006, 0.315731, 0.316654, 0.31219, 0.317884, 0.310899]
args=16 FFI empty kernel         median=3.6705 samples=[3.7607660000000003, 3.7043589999999997, 3.580483, 3.670506, 3.565312, 3.6765369999999997, 3.567179, 3.6651550000000004, 3.723283]
args=16 INTJ static_compile kernel median=3.3023 samples=[3.266502, 3.18753, 3.465654, 3.271493, 3.4656190000000002, 3.299471, 3.4514389999999997, 3.302316, 3.471961]
args=16 INTJ runtime_shim kernel median=3.2816 samples=[3.2815770000000004, 3.1823319999999997, 3.330259, 3.0614310000000002, 3.338409, 3.037787, 9.086694999999999, 3.050788, 3.337686]
args=32 FFI packed nop           median=0.4979 samples=[0.493938, 0.499335, 0.491162, 0.496453, 0.499964, 0.49785399999999996, 1.164194, 0.498724, 0.49741599999999997]
args=32 FFI typed nop            median=0.5162 samples=[0.510365, 0.5128010000000001, 0.499678, 0.537376, 0.5171760000000001, 0.516201, 1.262899, 0.518126, 0.516009]
args=32 FFI empty kernel         median=4.1687 samples=[4.232715, 4.207964, 4.003906, 4.178086, 4.009263, 4.168742, 11.30467, 4.160092, 4.002658]
args=32 INTJ static_compile kernel median=3.6508 samples=[3.4412040000000004, 3.655611, 3.6508339999999997, 9.901109, 3.6602020000000004, 3.54739, 3.633896, 3.5422860000000003, 3.656527]
args=32 INTJ runtime_shim kernel median=3.6204 samples=[10.719357, 3.586808, 3.5967089999999997, 10.958564, 3.630589, 3.441607, 3.623631, 3.4496909999999996, 3.620355]
args=64 FFI packed nop           median=0.8932 samples=[2.1527789999999998, 0.8809389999999999, 0.857718, 0.866754, 0.8991399999999999, 0.8965890000000001, 0.8807569999999999, 0.893159, 0.900471]
args=64 FFI typed nop            median=0.9108 samples=[2.365207, 0.910791, 0.88437, 0.8953289999999999, 0.9037430000000001, 0.920048, 0.898813, 0.92449, 0.926648]
args=64 FFI empty kernel         median=5.1050 samples=[7.418848, 5.139164, 5.083973, 5.097835, 5.10501, 5.107661, 5.087225, 5.131774, 5.08366]
args=64 INTJ static_compile kernel median=4.6174 samples=[4.770296999999999, 4.652428, 4.610354, 4.650970999999999, 4.5958000000000006, 4.643359, 4.616560000000001, 4.61742, 4.615372000000001]
args=64 INTJ runtime_shim kernel median=4.5999 samples=[4.6435, 4.625872, 4.504252, 4.609748, 4.579821, 4.612738, 4.599905, 4.574012, 4.585436]
exit_status=0
ended=2026-09-25T18:39:00.530816+00:00
```

### ffi_kwargs: candidate r1

```text
commit=c5295b18f29a15fc1184eb7e93dbc099e6b75bf2
started=2026-09-25T18:39:24.205658+00:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/callable_grid
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/callable_grid taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_intj_ffi_paths.py --mode=kwargs --iters 10000 --batches 9
iters=10000 batches=9 unit=ns/call; no_gpu=True
mode=kwargs
INTJ direct positional              median_ns=67.8 samples_ns=[68.4367, 68.5345, 67.1825, 67.6403, 67.5347, 67.9059, 67.201, 68.5035, 67.8154]
INTJ adapter positional             median_ns=92.3 samples_ns=[91.5198, 91.9897, 92.6033, 92.3247, 92.5269, 91.5605, 89.8938, 92.8811, 92.6133]
INTJ adapter kwargs                 median_ns=106.9 samples_ns=[106.9007, 106.6459, 108.5933, 106.8922, 105.889, 107.5324, 106.9313, 107.5847, 104.0511]
INTJ adapter defaults               median_ns=91.3 samples_ns=[92.6069, 97.0964, 96.584, 95.3457, 89.5099, 90.4505, 88.9151, 91.2542, 89.9387]
INTJ FFI wrapper positional         median_ns=103.9 samples_ns=[102.7411, 105.3736, 105.3668, 106.4257, 103.6859, 106.0522, 103.6476, 103.8833, 102.9599]
INTJ FFI wrapper kwargs             median_ns=125.3 samples_ns=[124.6004, 126.4443, 125.5721, 126.258, 124.593, 124.8577, 125.2992, 125.5642, 124.05]
INTJ native kwargs: unsupported (verified TypeError)
exit_status=0
ended=2026-09-25T18:39:27.059968+00:00
```

### ffi_kwargs: develop r1

```text
commit=1bf689b37137e373b6baecf241a8d8a7da23630b
started=2026-09-25T18:39:27.062450+00:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_intj_ffi_paths.py --mode=kwargs --iters 10000 --batches 9
iters=10000 batches=9 unit=ns/call; no_gpu=True
mode=kwargs
INTJ direct positional              median_ns=68.0 samples_ns=[68.1271, 68.5733, 67.1033, 68.2669, 67.2659, 67.7135, 68.0261, 68.4763, 66.9778]
INTJ adapter positional             median_ns=93.1 samples_ns=[93.1389, 93.1696, 90.8931, 94.3743, 91.082, 93.5592, 90.9252, 93.5535, 90.6554]
INTJ adapter kwargs                 median_ns=107.0 samples_ns=[107.0792, 107.8936, 105.2403, 106.9662, 106.3366, 110.0463, 106.8556, 109.5096, 105.8224]
INTJ adapter defaults               median_ns=90.8 samples_ns=[89.3701, 90.5099, 90.0911, 91.7123, 90.77, 93.4564, 94.5318, 91.8053, 89.4667]
INTJ FFI wrapper positional         median_ns=105.5 samples_ns=[104.3579, 106.8648, 105.4681, 107.8149, 104.9118, 105.6657, 104.1833, 106.0821, 103.901]
INTJ FFI wrapper kwargs             median_ns=124.2 samples_ns=[124.2463, 124.3932, 124.037, 125.83, 123.4587, 126.1729, 123.5884, 124.7386, 122.265]
INTJ native kwargs: unsupported (verified TypeError)
exit_status=0
ended=2026-09-25T18:39:29.972033+00:00
```

### ffi_dataclass: candidate r1

```text
commit=c5295b18f29a15fc1184eb7e93dbc099e6b75bf2
started=2026-09-25T18:39:47.683468+00:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/callable_grid
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/callable_grid taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_intj_ffi_paths.py --mode=dataclass --iters 10000 --batches 9
iters=10000 batches=9 unit=ns/call; no_gpu=True
mode=dataclass
INTJ pair direct                    median_ns=70.0 samples_ns=[70.0099, 69.766, 74.8123, 70.1725, 69.2787, 70.8299, 68.1952, 73.9713, 68.7288]
INTJ pair prebuilt *tuple           median_ns=140.6 samples_ns=[140.6454, 145.112, 136.7816, 141.9218, 139.2977, 141.897, 138.1593, 141.4261, 137.9149]
FFI unpack Pair only                median_ns=167.4 samples_ns=[166.5991, 166.884, 165.64, 167.8821, 168.4623, 176.1052, 165.4388, 168.9983, 167.3569]
INTJ pair manual unpack             median_ns=175.6 samples_ns=[174.79, 179.3383, 175.3307, 178.8157, 175.63, 176.5147, 173.7758, 176.7769, 172.7195]
INTJ pair FFI unpack                median_ns=319.2 samples_ns=[319.156, 320.6109, 318.2383, 323.7894, 319.5246, 318.6062, 315.2044, 319.9965, 317.8668]
INTJ pair stdlib astuple            median_ns=1169.3 samples_ns=[1183.9401, 1172.4713, 1156.3865, 1174.125, 1155.8901, 1169.341, 1156.9523, 1304.7437, 1154.8211]
INTJ config direct                  median_ns=84.1 samples_ns=[86.7274, 85.6025, 85.0215, 91.278, 82.806, 84.1276, 82.9747, 82.3109, 81.9413]
INTJ config FFI unpack              median_ns=329.2 samples_ns=[328.657, 332.957, 329.2069, 332.1275, 329.212, 331.9784, 328.9123, 332.8018, 328.7765]
INTJ native dataclass unpack: unsupported (verified TypeError)
exit_status=0
ended=2026-09-25T18:39:50.780461+00:00
```

### ffi_dataclass: develop r1

```text
commit=1bf689b37137e373b6baecf241a8d8a7da23630b
started=2026-09-25T18:39:50.782996+00:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_intj_ffi_paths.py --mode=dataclass --iters 10000 --batches 9
iters=10000 batches=9 unit=ns/call; no_gpu=True
mode=dataclass
INTJ pair direct                    median_ns=69.6 samples_ns=[69.8323, 74.3957, 69.5625, 69.4801, 69.3003, 69.347, 68.232, 71.6525, 72.0529]
INTJ pair prebuilt *tuple           median_ns=142.8 samples_ns=[142.7115, 146.3913, 142.8972, 146.5605, 142.5228, 142.5618, 142.7813, 143.7628, 141.0461]
FFI unpack Pair only                median_ns=165.0 samples_ns=[163.9732, 166.4435, 164.6466, 166.2068, 165.0454, 164.8016, 163.5724, 166.2635, 165.9907]
INTJ pair manual unpack             median_ns=178.9 samples_ns=[188.4216, 178.3553, 175.4316, 179.4135, 178.1165, 178.8668, 187.3535, 177.6306, 186.0656]
INTJ pair FFI unpack                median_ns=319.5 samples_ns=[318.3249, 320.0003, 319.371, 320.8962, 317.7349, 321.6391, 319.5143, 322.092, 317.9135]
INTJ pair stdlib astuple            median_ns=1164.0 samples_ns=[1191.338, 1163.9687, 1147.4688, 1162.5774, 1151.711, 1162.6656, 1166.4464, 1167.563, 1166.8721]
INTJ config direct                  median_ns=85.2 samples_ns=[82.8517, 83.558, 86.6973, 87.4088, 83.0015, 85.1709, 85.3503, 85.6143, 84.6095]
INTJ config FFI unpack              median_ns=327.8 samples_ns=[340.5393, 327.7715, 325.6052, 331.4576, 327.1073, 326.6938, 337.8832, 327.6302, 358.4749]
INTJ native dataclass unpack: unsupported (verified TypeError)
exit_status=0
ended=2026-09-25T18:39:53.787149+00:00
```

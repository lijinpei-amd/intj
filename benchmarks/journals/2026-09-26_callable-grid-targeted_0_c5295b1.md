# Callable-grid targeted ABBA controls — 2026-09-26, repeat 0 of 4

Baseline `develop` is `1bf689b37137e373b6baecf241a8d8a7da23630b`;
candidate `callable_grid` is `c5295b18f29a15fc1184eb7e93dbc099e6b75bf2`.
The local measurement date is 2026-09-26 (Asia/Shanghai), although logged
starts are 2026-09-25 UTC. The three modes run serially, each in D0 → C0 → C1
→ D1 → D2 → C2 → C3 → D3 order. All 24 processes exited 0. This file has
round 0's six raw outputs; [repeat 1](2026-09-26_callable-grid-targeted_1_c5295b1.md),
[repeat 2](2026-09-26_callable-grid-targeted_2_c5295b1.md), and [repeat 3](2026-09-26_callable-grid-targeted_3_c5295b1.md) retain the rest.
These controls follow the [full comparison](2026-09-26_callable-grid-full_0_c5295b1.md).

## Aggregate across four targeted rounds

Each input is a process median of nine batches. The table takes the median
of the four process medians for each revision; change is candidate minus
baseline. GPU rows are host enqueue µs/call; keyword/dataclass rows are
host-only ns/call.

| Row | Develop | Candidate | Change |
| --- | ---: | ---: | ---: |
| FFI sweep, 16 args: FFI empty kernel (µs) | 3.66475 | 3.61790 | −0.04685 |
| FFI sweep, 16 args: INTJ static compile (µs) | 3.38955 | 3.36935 | −0.02020 |
| FFI sweep, 16 args: INTJ runtime shim (µs) | 3.26675 | 3.25770 | −0.00905 |
| Keyword adapter, kwargs (ns) | 106.85 | 106.95 | +0.10 |
| Dataclass, FFI unpack Pair only (ns) | 166.65 | 165.65 | −1.00 |
| Dataclass, INTJ pair FFI unpack (ns) | 319.40 | 317.00 | −2.40 |

Candidate C2 has a shared GPU spike: the 16-argument FFI empty-kernel control
rises from D2 3.6590 to 4.2323 µs, INTJ static compile from 3.3886 to 5.3525
µs, and INTJ runtime shim from 3.2420 to 3.7891 µs. For the other three
pairs, 16-argument INTJ static differences are −0.0857, +0.0197, +0.0262 µs;
runtime-shim differences are −0.0037, −0.0369, −0.0144 µs. The higher-call
adapter-kwargs and dataclass rows do not reproduce the contaminated full-suite
D1 slowdown. No repeatable material slowdown was observed; a small effect is
still possible.

## Environment, commands, and caveats

Intel Xeon Platinum 8480C on CPU 0 (`taskset -c 0`); AMD Instinct MI308X GPU 0,
`gfx942:sramecc+:xnack-`. Python 3.12.3 (`/tmp/gb2/bin/python`), Torch
`2.14.0+rocm7.2` (`torch.version.hip=7.2.53211`), Triton `3.8.0`, TVM FFI
`0.1.14.post2.dev1+g424558557.d20260924`. The baseline environment snapshot
had `TRITON_HOME`, `TVM_FFI_CACHE_DIR`, `CC`, `CXX`, and GPU visibility masks
unset. The driver `/tmp/intj_targeted_abba_2026-09-26.py` inherited its
process environment and set only `PYTHONPATH` for each checkout. It recorded
each child's exact command in the raw block, but not whole-tree status or a
source hash. It did not clear caches; the logs do not establish a cold cache.

- `benchmarks/bench_ffi_compare.py --sweep --iters 1000 --batches 9`:
  1,000 calls per batch, host enqueue timing for kernel rows. FFI packed and
  typed no-ops do not launch kernels; FFI and INTJ GPU kernels differ.
- `benchmarks/bench_intj_ffi_paths.py --mode=kwargs --iters 10000 --batches 9`:
  10,000 host-only calls per batch. INTJ's Python adapters add keyword support.
- `benchmarks/bench_intj_ffi_paths.py --mode=dataclass --iters 10000 --batches 9`:
  10,000 host-only calls per batch. Dataclass conversion is outside native INTJ.

Each case warms before timing and constructs arguments outside the loop. GPU
cases synchronize after the host timer, so queue pressure can change enqueue
times. CUDA runtime was not tested. The logs do not establish an idle GPU.

## Raw outputs — repeat 0

### ffi_sweep: develop r0

```text
commit=1bf689b37137e373b6baecf241a8d8a7da23630b
started=2026-09-25T18:38:42.694202+00:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_ffi_compare.py --sweep --iters 1000 --batches 9
counts=[0, 3, 5, 8, 16, 32, 64] iters=1000 batches=9 unit=us/call (host-only, no timed sync); INTJ device-bound, no specialization key
args= 0 FFI packed nop           median=0.1168 samples=[0.12323999999999999, 0.116375, 0.117145, 0.115836, 0.116805, 0.118395, 0.116383, 0.11659699999999999, 0.117055]
args= 0 FFI typed nop            median=0.1180 samples=[0.127018, 0.120213, 0.11579300000000001, 0.117706, 0.11768, 0.121209, 0.11722199999999999, 0.12229999999999999, 0.118012]
args= 0 FFI empty kernel         median=1.6600 samples=[1.571292, 1.875345, 1.626184, 1.886073, 1.611102, 1.859121, 1.6545969999999999, 1.8769829999999998, 1.659973]
args= 0 INTJ static_compile kernel median=2.7206 samples=[3.152416, 2.7285999999999997, 2.720641, 2.6775160000000002, 2.709739, 2.673165, 2.766542, 2.670312, 2.772867]
args= 0 INTJ runtime_shim kernel median=2.7089 samples=[2.873619, 2.703444, 2.735873, 2.695405, 2.70889, 2.6836309999999997, 2.7687370000000002, 2.678202, 2.765729]
args= 3 FFI packed nop           median=0.1446 samples=[0.146028, 0.14552299999999999, 0.143337, 0.146761, 0.1446, 0.142316, 0.14325200000000002, 0.1457, 0.14393899999999998]
args= 3 FFI typed nop            median=0.1492 samples=[0.14841200000000002, 0.150846, 0.15139, 0.151397, 0.146702, 0.149761, 0.148332, 0.149229, 0.14552]
args= 3 FFI empty kernel         median=3.2144 samples=[3.3923870000000003, 3.216345, 3.1596550000000003, 3.228935, 3.1560659999999996, 3.214404, 3.161143, 3.2185569999999997, 3.162408]
args= 3 INTJ static_compile kernel median=2.9545 samples=[2.996205, 2.844608, 2.9545250000000003, 2.8865079999999996, 2.954903, 2.868958, 2.95747, 2.882382, 2.963572]
args= 3 INTJ runtime_shim kernel median=2.8549 samples=[2.9860010000000003, 2.854863, 2.816224, 2.895089, 2.7785320000000002, 2.855901, 2.7854740000000002, 2.896343, 2.7752890000000003]
args= 5 FFI packed nop           median=0.1757 samples=[0.177529, 0.175655, 0.177097, 0.175969, 0.176816, 0.175068, 0.17248, 0.17430400000000001, 0.17241399999999998]
args= 5 FFI typed nop            median=0.1813 samples=[0.17869, 0.181278, 0.178113, 0.19523300000000002, 0.179498, 0.18156, 0.17777099999999998, 0.18549600000000002, 0.182067]
args= 5 FFI empty kernel         median=3.3107 samples=[4.616968, 3.320868, 3.189276, 3.357186, 3.194208, 3.310692, 3.219329, 3.31321, 3.193076]
args= 5 INTJ static_compile kernel median=2.9052 samples=[10.004033, 2.958205, 2.850008, 2.881838, 2.905222, 2.883016, 2.9215050000000002, 2.896964, 2.9202489999999997]
args= 5 INTJ runtime_shim kernel median=2.8808 samples=[3.0173769999999998, 2.9074250000000004, 2.814763, 2.824743, 2.8807910000000003, 2.775621, 2.894197, 2.802465, 2.902121]
args= 8 FFI packed nop           median=0.2108 samples=[0.2099, 0.210782, 0.212838, 0.213244, 0.213001, 0.20995, 0.211071, 0.209274, 0.206644]
args= 8 FFI typed nop            median=0.2134 samples=[0.218422, 0.21472300000000002, 0.213165, 0.214863, 0.212855, 0.213445, 0.213076, 0.21489599999999998, 0.209739]
args= 8 FFI empty kernel         median=3.3945 samples=[3.497261, 3.4460520000000003, 3.27267, 3.4343589999999997, 3.332288, 3.3945369999999997, 3.334454, 3.397384, 3.332845]
args= 8 INTJ static_compile kernel median=3.0679 samples=[3.0709720000000003, 2.9922109999999997, 3.018228, 3.067879, 3.083063, 3.261082, 3.0619549999999998, 3.024114, 3.0735740000000003]
args= 8 INTJ runtime_shim kernel median=2.9425 samples=[3.079624, 2.932682, 2.942265, 2.945375, 2.9424870000000003, 2.881924, 2.957381, 2.91749, 2.966378]
args=16 FFI packed nop           median=0.3023 samples=[0.303908, 0.300557, 0.305824, 0.306453, 0.307469, 0.298079, 0.300379, 0.300765, 0.302332]
args=16 FFI typed nop            median=0.3109 samples=[0.30994900000000003, 0.308694, 0.311268, 0.315822, 0.31104899999999996, 0.310444, 0.309679, 0.31431200000000004, 0.310871]
args=16 FFI empty kernel         median=3.6545 samples=[3.721811, 3.785863, 3.537935, 3.7326129999999997, 3.524357, 3.654516, 3.5546320000000002, 3.69217, 3.547038]
args=16 INTJ static_compile kernel median=3.3998 samples=[3.247043, 3.436708, 3.408381, 3.34091, 3.4515059999999997, 3.2809540000000004, 3.3997930000000003, 3.309159, 3.410981]
args=16 INTJ runtime_shim kernel median=3.2732 samples=[3.273205, 3.152953, 3.2803910000000003, 3.1257420000000002, 3.306299, 3.028977, 3.290288, 3.063497, 3.295194]
args=32 FFI packed nop           median=0.4940 samples=[0.483586, 0.49180599999999997, 0.478562, 0.505586, 0.507062, 0.492894, 0.495545, 0.500419, 0.494039]
args=32 FFI typed nop            median=0.5078 samples=[0.505157, 0.5078469999999999, 0.499936, 0.518344, 0.516454, 0.514959, 0.507292, 0.51181, 0.501385]
args=32 FFI empty kernel         median=4.1395 samples=[4.171543, 4.139976, 3.971203, 4.217269, 3.977252, 4.139539, 3.980482, 4.154061, 3.980445]
args=32 INTJ static_compile kernel median=3.6237 samples=[3.4572950000000002, 3.623735, 3.651498, 3.616939, 3.641471, 3.5414540000000003, 3.638313, 3.582395, 3.642418]
args=32 INTJ runtime_shim kernel median=3.5468 samples=[3.460979, 3.54681, 3.630352, 3.4907399999999997, 3.607703, 3.45771, 3.617348, 3.488994, 3.612971]
args=64 FFI packed nop           median=0.8705 samples=[0.873676, 0.8822709999999999, 0.869297, 0.884524, 0.869052, 0.850191, 0.870469, 0.8510460000000001, 0.8960549999999999]
args=64 FFI typed nop            median=0.8872 samples=[0.887225, 0.92137, 0.883396, 0.93304, 0.8926029999999999, 0.881707, 0.878587, 0.8808680000000001, 0.905154]
args=64 FFI empty kernel         median=5.0935 samples=[5.118965, 5.137118999999999, 5.090064999999999, 5.163064, 5.081863, 5.070889, 5.075449, 16.265572, 5.093546999999999]
args=64 INTJ static_compile kernel median=4.6417 samples=[4.755648, 4.64166, 4.625811, 4.641675, 4.6226329999999995, 4.659683, 4.610791, 12.096432, 4.638586]
args=64 INTJ runtime_shim kernel median=4.5937 samples=[4.740600000000001, 4.604908, 9.347014, 17.764549, 4.534586, 4.566021, 4.525171, 4.593735, 4.560881999999999]
exit_status=0
ended=2026-09-25T18:38:47.147338+00:00
```

### ffi_sweep: candidate r0

```text
commit=c5295b18f29a15fc1184eb7e93dbc099e6b75bf2
started=2026-09-25T18:38:47.149644+00:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/callable_grid
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/callable_grid taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_ffi_compare.py --sweep --iters 1000 --batches 9
counts=[0, 3, 5, 8, 16, 32, 64] iters=1000 batches=9 unit=us/call (host-only, no timed sync); INTJ device-bound, no specialization key
args= 0 FFI packed nop           median=0.1179 samples=[0.121207, 0.117577, 0.118037, 0.11743899999999999, 0.115696, 0.117875, 0.118544, 0.117854, 0.11368600000000001]
args= 0 FFI typed nop            median=0.1210 samples=[0.123404, 0.124098, 0.118965, 0.12064799999999999, 0.11688, 0.121012, 0.122832, 0.122708, 0.11745]
args= 0 FFI empty kernel         median=1.7126 samples=[1.6408779999999998, 1.881671, 1.712567, 1.929006, 1.5770609999999998, 1.8316620000000001, 1.6324960000000002, 1.884536, 1.644708]
args= 0 INTJ static_compile kernel median=2.7475 samples=[3.149739, 2.7511, 2.818858, 2.751547, 2.682883, 2.670355, 2.747484, 2.742795, 2.736969]
args= 0 INTJ runtime_shim kernel median=2.7324 samples=[2.898973, 2.7789789999999996, 2.846203, 2.643074, 2.6886439999999996, 2.6682689999999996, 2.746042, 2.6367220000000002, 2.732389]
args= 3 FFI packed nop           median=0.1447 samples=[0.143873, 0.144905, 0.145505, 0.143986, 0.144651, 0.147334, 0.14338399999999998, 0.14681, 0.143191]
args= 3 FFI typed nop            median=0.1506 samples=[0.148211, 0.154242, 0.15087899999999999, 0.148861, 0.150601, 0.151658, 0.147785, 0.15191200000000002, 0.149738]
args= 3 FFI empty kernel         median=3.2530 samples=[3.3563180000000004, 3.318988, 3.170847, 3.2770770000000002, 3.1254389999999996, 3.2656680000000002, 3.122831, 3.253031, 3.116538]
args= 3 INTJ static_compile kernel median=2.9556 samples=[2.9845680000000003, 2.9818919999999998, 2.979104, 2.932721, 2.971413, 2.8895549999999997, 2.955638, 2.891707, 2.933539]
args= 3 INTJ runtime_shim kernel median=2.8685 samples=[2.998489, 2.871256, 2.925071, 2.799026, 2.868005, 2.768912, 2.868477, 2.778802, 2.874866]
args= 5 FFI packed nop           median=0.1740 samples=[0.17128800000000002, 0.173696, 0.173976, 0.174882, 0.174528, 0.17488, 0.174257, 0.170003, 0.17363]
args= 5 FFI typed nop            median=0.1763 samples=[0.176332, 0.177617, 0.176373, 0.17633500000000002, 0.17619200000000002, 0.178581, 0.17599199999999998, 0.180505, 0.174817]
args= 5 FFI empty kernel         median=3.2498 samples=[3.35882, 3.328443, 3.275549, 3.2252979999999996, 3.2498, 3.20883, 3.217485, 3.228431, 3.26317]
args= 5 INTJ static_compile kernel median=2.8863 samples=[3.023117, 2.990628, 2.900482, 2.877785, 2.864276, 2.9207240000000003, 2.879931, 2.850927, 2.886287]
args= 5 INTJ runtime_shim kernel median=2.8639 samples=[3.00797, 3.063589, 2.8639189999999997, 2.8947570000000002, 2.845862, 2.908337, 2.842003, 2.8519099999999997, 2.8534200000000003]
args= 8 FFI packed nop           median=0.2053 samples=[0.204716, 0.205307, 0.207121, 0.20979699999999998, 0.20662, 0.205172, 0.20578200000000002, 0.204345, 0.203706]
args= 8 FFI typed nop            median=0.2090 samples=[0.20671199999999998, 0.21538300000000002, 0.209818, 0.209038, 0.207581, 0.20904, 0.20715799999999998, 0.210779, 0.205611]
args= 8 FFI empty kernel         median=3.3741 samples=[3.515843, 3.5410399999999997, 3.547707, 3.396617, 3.2981599999999998, 3.374099, 3.288321, 3.367735, 3.299148]
args= 8 INTJ static_compile kernel median=3.0430 samples=[3.068387, 3.093118, 3.05057, 3.030124, 3.0498600000000002, 2.993741, 3.0429749999999998, 2.9919520000000004, 3.027521]
args= 8 INTJ runtime_shim kernel median=2.8788 samples=[3.080446, 3.032166, 2.893497, 2.9040239999999997, 2.86965, 2.863603, 2.868604, 2.878799, 2.878666]
args=16 FFI packed nop           median=0.2938 samples=[0.298892, 0.29812, 0.296605, 0.297986, 0.292899, 0.293499, 0.289529, 0.288584, 0.29379700000000003]
args=16 FFI typed nop            median=0.3038 samples=[0.301942, 0.31043099999999996, 0.303093, 0.305649, 0.30380799999999997, 0.302536, 0.304273, 0.304162, 0.303253]
args=16 FFI empty kernel         median=3.6049 samples=[3.690481, 3.78502, 3.536414, 3.6115340000000002, 3.505929, 3.604888, 3.497045, 3.6642609999999998, 3.489598]
args=16 INTJ static_compile kernel median=3.3141 samples=[3.2611399999999997, 3.4234229999999997, 3.3141190000000003, 3.2812710000000003, 3.399352, 3.27818, 3.4393890000000003, 3.306239, 3.396874]
args=16 INTJ runtime_shim kernel median=3.2695 samples=[3.3183339999999997, 3.269516, 3.4461709999999997, 3.055709, 3.266332, 3.012207, 3.326905, 3.019968, 3.30395]
args=32 FFI packed nop           median=0.4794 samples=[0.479397, 0.489575, 0.49069, 0.476781, 0.482096, 0.47613900000000003, 0.50381, 0.473211, 0.47720100000000004]
args=32 FFI typed nop            median=0.4938 samples=[0.493793, 0.502772, 0.502163, 0.492428, 0.492014, 0.49175599999999997, 0.514342, 0.49614400000000003, 0.481854]
args=32 FFI empty kernel         median=4.1792 samples=[4.3104949999999995, 4.34612, 4.083079000000001, 4.211061, 4.074618, 4.1801, 4.035571, 4.179206, 4.0697849999999995]
args=32 INTJ static_compile kernel median=3.6046 samples=[3.564797, 3.7275039999999997, 3.6419319999999997, 3.5657330000000003, 3.604639, 3.535231, 3.6251770000000003, 3.568154, 3.657435]
args=32 INTJ runtime_shim kernel median=3.5485 samples=[3.519315, 3.576961, 3.5653319999999997, 3.4399070000000003, 3.562395, 3.401814, 3.5642620000000003, 3.4035509999999998, 3.548498]
args=64 FFI packed nop           median=0.8414 samples=[0.821369, 0.8637229999999999, 0.8814740000000001, 0.836222, 0.841367, 0.822804, 0.8623730000000001, 0.830654, 0.844971]
args=64 FFI typed nop            median=0.8604 samples=[0.859375, 0.88627, 0.880148, 0.866911, 0.853169, 0.855005, 0.8363200000000001, 0.860362, 0.861636]
args=64 FFI empty kernel         median=5.0500 samples=[5.175623, 5.199336000000001, 5.098409999999999, 5.033626, 5.0499719999999995, 5.144845999999999, 5.022009, 4.994834, 4.986281]
args=64 INTJ static_compile kernel median=4.5521 samples=[5.031462, 5.408457, 4.599784, 4.5501570000000005, 4.556959, 4.527298, 4.540961, 4.5520760000000005, 4.529064999999999]
args=64 INTJ runtime_shim kernel median=4.4964 samples=[4.846644, 4.622974, 4.450316, 4.528006, 4.478295, 4.496448999999999, 4.469220999999999, 4.537243999999999, 4.474428]
exit_status=0
ended=2026-09-25T18:38:51.654740+00:00
```

### ffi_kwargs: develop r0

```text
commit=1bf689b37137e373b6baecf241a8d8a7da23630b
started=2026-09-25T18:39:18.412029+00:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_intj_ffi_paths.py --mode=kwargs --iters 10000 --batches 9
iters=10000 batches=9 unit=ns/call; no_gpu=True
mode=kwargs
INTJ direct positional              median_ns=67.9 samples_ns=[68.7917, 68.5295, 67.4855, 67.5237, 67.0387, 68.2391, 68.2358, 67.8665, 67.0142]
INTJ adapter positional             median_ns=91.4 samples_ns=[90.7226, 91.5331, 91.0483, 93.5009, 92.0719, 92.6727, 90.1219, 91.3719, 91.0706]
INTJ adapter kwargs                 median_ns=106.6 samples_ns=[106.8815, 106.5705, 105.4069, 106.6015, 104.9851, 105.9384, 107.6871, 107.5133, 105.4994]
INTJ adapter defaults               median_ns=90.3 samples_ns=[90.0161, 90.7616, 90.8283, 91.3286, 89.7838, 89.9055, 89.4528, 92.3469, 90.3013]
INTJ FFI wrapper positional         median_ns=105.0 samples_ns=[104.4188, 105.6921, 103.5898, 106.2175, 103.1612, 105.8302, 104.9805, 105.3934, 104.1568]
INTJ FFI wrapper kwargs             median_ns=125.0 samples_ns=[124.7407, 126.0065, 124.4508, 125.0329, 124.9665, 126.6913, 123.5842, 126.7052, 123.1387]
INTJ native kwargs: unsupported (verified TypeError)
exit_status=0
ended=2026-09-25T18:39:21.294297+00:00
```

### ffi_kwargs: candidate r0

```text
commit=c5295b18f29a15fc1184eb7e93dbc099e6b75bf2
started=2026-09-25T18:39:21.296465+00:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/callable_grid
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/callable_grid taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_intj_ffi_paths.py --mode=kwargs --iters 10000 --batches 9
iters=10000 batches=9 unit=ns/call; no_gpu=True
mode=kwargs
INTJ direct positional              median_ns=67.7 samples_ns=[67.113, 68.0022, 67.9952, 68.5451, 67.6593, 67.6515, 67.6678, 67.8914, 66.98]
INTJ adapter positional             median_ns=91.7 samples_ns=[90.6911, 92.3581, 91.2962, 92.9981, 92.496, 94.3359, 91.7202, 91.1816, 90.5681]
INTJ adapter kwargs                 median_ns=108.1 samples_ns=[108.1288, 109.4614, 106.7215, 105.9643, 106.7619, 109.685, 108.4471, 109.5731, 106.7317]
INTJ adapter defaults               median_ns=90.5 samples_ns=[90.3991, 92.579, 89.5535, 90.1955, 89.8729, 95.4011, 90.4709, 91.258, 91.8142]
INTJ FFI wrapper positional         median_ns=104.8 samples_ns=[104.8105, 104.8634, 103.6178, 105.3763, 104.1636, 106.4696, 103.1359, 105.5801, 104.3089]
INTJ FFI wrapper kwargs             median_ns=124.8 samples_ns=[123.8709, 124.8427, 124.6501, 128.0078, 125.8366, 127.9584, 123.8382, 124.8283, 123.0661]
INTJ native kwargs: unsupported (verified TypeError)
exit_status=0
ended=2026-09-25T18:39:24.203188+00:00
```

### ffi_dataclass: develop r0

```text
commit=1bf689b37137e373b6baecf241a8d8a7da23630b
started=2026-09-25T18:39:41.533925+00:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_intj_ffi_paths.py --mode=dataclass --iters 10000 --batches 9
iters=10000 batches=9 unit=ns/call; no_gpu=True
mode=dataclass
INTJ pair direct                    median_ns=69.7 samples_ns=[68.87, 69.3034, 68.3106, 70.1783, 69.6749, 69.8275, 71.1878, 70.3653, 68.4924]
INTJ pair prebuilt *tuple           median_ns=282.9 samples_ns=[143.9272, 157.6973, 313.5163, 255.6458, 244.4653, 282.9394, 308.0542, 338.1061, 353.5646]
FFI unpack Pair only                median_ns=170.2 samples_ns=[403.3805, 258.5194, 166.6393, 167.9977, 166.2811, 170.2018, 168.3576, 172.0705, 171.1771]
INTJ pair manual unpack             median_ns=179.1 samples_ns=[187.7719, 177.4739, 175.641, 177.8334, 176.0317, 179.1468, 189.2607, 179.3437, 187.6667]
INTJ pair FFI unpack                median_ns=323.7 samples_ns=[322.5835, 325.6669, 323.6828, 324.7078, 322.978, 325.8388, 322.4916, 324.6087, 322.4039]
INTJ pair stdlib astuple            median_ns=1152.9 samples_ns=[1180.8802, 1153.9063, 1142.5455, 1152.9428, 1144.1116, 1153.5133, 1151.0313, 1153.9821, 1148.3254]
INTJ config direct                  median_ns=82.9 samples_ns=[81.8454, 82.2876, 84.4832, 82.245, 82.5061, 84.4453, 83.209, 82.9366, 82.8721]
INTJ config FFI unpack              median_ns=332.7 samples_ns=[341.5704, 330.3158, 330.4094, 333.7838, 329.9716, 332.73, 340.5618, 331.4879, 339.8146]
INTJ native dataclass unpack: unsupported (verified TypeError)
exit_status=0
ended=2026-09-25T18:39:44.595485+00:00
```

### ffi_dataclass: candidate r0

```text
commit=c5295b18f29a15fc1184eb7e93dbc099e6b75bf2
started=2026-09-25T18:39:44.598152+00:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/callable_grid
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/callable_grid taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_intj_ffi_paths.py --mode=dataclass --iters 10000 --batches 9
iters=10000 batches=9 unit=ns/call; no_gpu=True
mode=dataclass
INTJ pair direct                    median_ns=69.4 samples_ns=[67.9496, 69.3554, 69.932, 69.6142, 68.8587, 70.9843, 69.3496, 69.2346, 69.5728]
INTJ pair prebuilt *tuple           median_ns=140.0 samples_ns=[138.2128, 144.3278, 138.5491, 142.9389, 138.7497, 142.934, 138.6908, 142.163, 140.0035]
FFI unpack Pair only                median_ns=164.7 samples_ns=[164.7223, 167.0396, 164.9535, 163.4824, 162.2412, 165.2583, 164.1086, 165.598, 164.3494]
INTJ pair manual unpack             median_ns=176.4 samples_ns=[177.571, 180.5318, 174.6948, 177.472, 175.0129, 177.1313, 174.767, 176.4173, 173.607]
INTJ pair FFI unpack                median_ns=316.1 samples_ns=[313.6306, 318.415, 316.1429, 320.351, 317.744, 318.0823, 311.0449, 315.9173, 312.2245]
INTJ pair stdlib astuple            median_ns=1181.4 samples_ns=[1184.1031, 1189.416, 1156.368, 1183.9112, 1158.4837, 1184.0288, 1159.3671, 1181.3605, 1160.1529]
INTJ config direct                  median_ns=83.0 samples_ns=[83.0133, 83.7605, 84.4111, 84.9933, 82.3663, 83.4621, 81.1058, 82.4269, 82.5616]
INTJ config FFI unpack              median_ns=329.0 samples_ns=[326.3797, 330.7535, 326.3721, 329.085, 326.0644, 330.7498, 328.4066, 330.6383, 328.9637]
INTJ native dataclass unpack: unsupported (verified TypeError)
exit_status=0
ended=2026-09-25T18:39:47.680786+00:00
```

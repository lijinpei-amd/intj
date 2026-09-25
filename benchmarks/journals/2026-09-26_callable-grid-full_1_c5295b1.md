# Callable-grid full ABBA comparison — 2026-09-26, repeat 1 of 2

[Aggregate, environment, commands, caveats, and host-parser history](2026-09-26_callable-grid-full_0_c5295b1.md).
These are the remaining 16 full-suite process outputs, in actual per-mode
C1 → D1 order. Each raw block includes its command, commit, source-template
hash, UTC time, nine-batch samples, and exit status. The local date is
2026-09-26 (Asia/Shanghai).

## Raw outputs — repeat 1

### launch_host: candidate r1

```text
commit=c5295b18f29a15fc1184eb7e93dbc099e6b75bf2
source_sha256=50f04d4c6bc58bd50a029525208892353ac1ba7184540ca7088d3ebaa9c2cb3f
started=2026-09-25T18:33:56.686712+00:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/callable_grid
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/callable_grid taskset -c 0 /tmp/gb2/bin/python /tmp/intj_develop_baseline_2026-09-26_1bf689b/record_launch_batches.py benchmarks/bench_launch.py --no-gpu --iters 100000 --batches 9
mode=host-only; 100000 calls × 9 batches; median ns/call
               path    ns/call       Δ ns       Δ %   build ms   bind ms
recorded_batch_samples_ns=[43.0699, 44.85393, 43.37834, 44.73951, 43.33783, 44.73007, 43.12288, 44.69873, 43.32773]
           auto map       43.4       +0.0      +0.0      61.66         -
recorded_batch_samples_ns=[43.01739, 43.38493, 42.70892, 43.14863, 42.78052, 43.82993, 42.80046, 43.28753, 42.8859]
        reduced key       43.0       -0.4      -0.8       1.74         -
recorded_batch_samples_ns=[43.87385, 45.26373, 44.05661, 44.69374, 43.98459, 44.80515, 43.98828, 44.35369, 43.74208]
         verify off       44.1       +0.7      +1.6       1.60         -
recorded_batch_samples_ns=[44.09832, 44.51939, 43.98045, 44.36236, 44.14816, 44.54013, 44.53592, 44.59756, 44.20245]
          verify on       44.4       +1.0      +2.3       1.60         -
recorded_batch_samples_ns=[39.5573, 40.66939, 39.65154, 40.31481, 41.18207, 41.65178, 39.82603, 40.10116, 39.73885]
              baked       40.1       -3.3      -7.6       1.63         -
recorded_batch_samples_ns=[45.12665, 73.7858, 102.78072, 113.79177, 56.29979, 46.19701, 45.14962, 45.59907, 45.12668]
       bound tensor       46.2       +2.8      +6.5       0.15      1.49
recorded_batch_samples_ns=[43.9902, 43.79598, 43.85281, 43.71729, 44.11773, 44.1113, 44.21584, 44.33668, 43.82182]
      bound pointer       44.0       +0.6      +1.4       0.15      1.55
recorded_batch_samples_ns=[44.01567, 44.38334, 44.5454, 44.55021, 43.77151, 44.3022, 43.78616, 44.49611, 43.92249]
   fixed device map       44.3       +0.9      +2.1       0.11      1.50
recorded_batch_samples_ns=[44.65295, 43.97906, 43.49107, 45.01152, 42.54672, 44.93608, 42.67233, 43.06496, 42.85478]
fixed device no-map       43.5       +0.1      +0.3       0.15      1.42
exit_status=0
ended=2026-09-25T18:33:59.726740+00:00
```

### launch_host: develop r1

```text
commit=1bf689b37137e373b6baecf241a8d8a7da23630b
source_sha256=7faebcc700fe0b5bd1363acfc638b0621eebf7af754ab3dcf915ebd0c4e41806
started=2026-09-25T18:33:59.729202+00:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj taskset -c 0 /tmp/gb2/bin/python /tmp/intj_develop_baseline_2026-09-26_1bf689b/record_launch_batches.py benchmarks/bench_launch.py --no-gpu --iters 100000 --batches 9
mode=host-only; 100000 calls × 9 batches; median ns/call
               path    ns/call       Δ ns       Δ %   build ms   bind ms
recorded_batch_samples_ns=[43.87364, 44.00718, 44.01182, 44.05355, 43.44527, 43.95311, 43.53228, 44.24097, 43.68243]
           auto map       44.0       +0.0      +0.0      91.51         -
recorded_batch_samples_ns=[43.51642, 43.57227, 43.04174, 43.62106, 42.98864, 43.58396, 43.08948, 43.59567, 43.38313]
        reduced key       43.5       -0.4      -1.0       1.56         -
recorded_batch_samples_ns=[43.40668, 43.75816, 43.18268, 43.63681, 43.25698, 43.68483, 43.15756, 43.73416, 43.29579]
         verify off       43.4       -0.5      -1.2       1.41         -
recorded_batch_samples_ns=[44.04813, 45.00147, 44.34502, 44.79393, 44.0338, 44.55446, 44.03814, 45.8484, 44.07915]
          verify on       44.3       +0.4      +0.9       1.40         -
recorded_batch_samples_ns=[39.80594, 40.5932, 39.71789, 40.39971, 39.7859, 40.50154, 39.82981, 40.91133, 39.85057]
              baked       39.9       -4.1      -9.3       1.51         -
recorded_batch_samples_ns=[45.89737, 46.0659, 45.34167, 45.90027, 45.50642, 46.29573, 45.39272, 45.78768, 45.43663]
       bound tensor       45.8       +1.8      +4.2       0.14      1.28
recorded_batch_samples_ns=[44.62721, 44.01821, 44.79783, 43.93859, 44.10034, 44.02776, 43.9789, 43.79308, 43.92633]
      bound pointer       44.0       +0.1      +0.1       0.13      1.23
recorded_batch_samples_ns=[43.8839, 44.3436, 43.78984, 44.34928, 43.98314, 44.40848, 43.76378, 44.4018, 43.84723]
   fixed device map       44.0       +0.0      +0.1       0.11      1.22
recorded_batch_samples_ns=[43.1419, 44.53762, 42.73211, 44.23354, 43.1709, 44.37383, 42.64629, 44.38204, 42.92593]
fixed device no-map       43.2       -0.8      -1.8       0.14      1.20
exit_status=0
ended=2026-09-25T18:34:02.783230+00:00
```

### launch_sweep: candidate r1

```text
commit=c5295b18f29a15fc1184eb7e93dbc099e6b75bf2
source_sha256=50f04d4c6bc58bd50a029525208892353ac1ba7184540ca7088d3ebaa9c2cb3f
started=2026-09-25T18:34:08.874250+00:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/callable_grid
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/callable_grid taskset -c 0 /tmp/gb2/bin/python /tmp/intj_develop_baseline_2026-09-26_1bf689b/record_launch_batches.py benchmarks/bench_launch.py --sweep --iters 100000 --batches 9
mode=sweep; host-only; 100000 calls × 9 batches; median ns/call
count   kind    ns/call
recorded_batch_samples_ns=[41.38848, 41.67928, 41.44179, 41.62027, 41.25469, 41.43214, 41.63864, 41.66646, 41.21495]
    4    int       41.4
recorded_batch_samples_ns=[40.32174, 40.66116, 40.11507, 40.60208, 40.22753, 40.84798, 40.26548, 40.83177, 40.20082]
    4 tensor       40.3
recorded_batch_samples_ns=[70.9076, 70.22654, 70.45989, 70.01169, 70.30121, 71.39331, 71.32353, 70.93267, 75.21892]
   16    int       70.9
recorded_batch_samples_ns=[68.20793, 142.48494, 106.48777, 67.41845, 67.49492, 69.2405, 69.03095, 68.09914, 68.01159]
   16 tensor       68.2
recorded_batch_samples_ns=[110.68397, 113.45457, 108.30309, 113.52626, 109.35756, 110.49073, 111.85023, 109.68267, 107.88745]
   32    int      110.5
recorded_batch_samples_ns=[126.94503, 120.16773, 116.37937, 111.3201, 113.90107, 112.43182, 110.69362, 112.81443, 114.3267]
   32 tensor      113.9
exit_status=0
ended=2026-09-25T18:34:11.957009+00:00
```

### launch_sweep: develop r1

```text
commit=1bf689b37137e373b6baecf241a8d8a7da23630b
source_sha256=7faebcc700fe0b5bd1363acfc638b0621eebf7af754ab3dcf915ebd0c4e41806
started=2026-09-25T18:34:11.959447+00:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj taskset -c 0 /tmp/gb2/bin/python /tmp/intj_develop_baseline_2026-09-26_1bf689b/record_launch_batches.py benchmarks/bench_launch.py --sweep --iters 100000 --batches 9
mode=sweep; host-only; 100000 calls × 9 batches; median ns/call
count   kind    ns/call
recorded_batch_samples_ns=[41.51643, 41.80404, 41.60482, 42.02892, 41.44804, 41.87315, 41.45662, 41.85208, 41.37792]
    4    int       41.6
recorded_batch_samples_ns=[40.33367, 40.98522, 40.45892, 40.89036, 40.40033, 40.90065, 40.36906, 40.74398, 40.64652]
    4 tensor       40.6
recorded_batch_samples_ns=[74.4041, 74.18129, 74.47415, 74.29513, 74.20036, 74.22921, 74.60843, 74.62833, 74.28666]
   16    int       74.3
recorded_batch_samples_ns=[71.73012, 71.56863, 71.43904, 74.19688, 69.68086, 68.57995, 68.33168, 69.24569, 68.35201]
   16 tensor       69.7
recorded_batch_samples_ns=[121.81991, 121.64538, 123.51432, 121.78381, 125.16206, 122.25003, 116.55263, 120.00553, 123.5075]
   32    int      121.8
recorded_batch_samples_ns=[118.28494, 246.24405, 160.25592, 115.85054, 114.99276, 119.38206, 116.72856, 115.27554, 115.38975]
   32 tensor      116.7
exit_status=0
ended=2026-09-25T18:34:15.057876+00:00
```

### ffi_paths: candidate r1

```text
commit=c5295b18f29a15fc1184eb7e93dbc099e6b75bf2
source_sha256=50f04d4c6bc58bd50a029525208892353ac1ba7184540ca7088d3ebaa9c2cb3f
started=2026-09-25T18:34:27.000867+00:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/callable_grid
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/callable_grid taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_intj_ffi_paths.py --mode=all --iters 1000 --batches 9
iters=1000 batches=9 unit=ns/call; no_gpu=True
mode=callback
callback_count=9100 (100 warmup + 9000 unique misses)
INTJ cold compile callback + cache  median_ns=417.5 samples_ns=[462.669, 450.141, 402.223, 572.325, 385.563, 406.673, 411.319, 417.517, 1517.626]
INTJ hot call (no callback)         median_ns=39.3 samples_ns=[47.135, 40.294, 38.923, 39.563, 38.574, 39.292, 38.597, 40.216, 38.401]
INTJ 3-tensor host-only nop         median_ns=35.5 samples_ns=[37.549, 36.02, 35.51, 35.868, 35.272, 35.343, 35.317, 35.875, 35.267]
mode=kwargs
INTJ direct positional              median_ns=66.7 samples_ns=[69.1, 67.125, 66.767, 66.674, 73.198, 66.512, 66.04, 66.225, 66.663]
INTJ adapter positional             median_ns=90.0 samples_ns=[91.535, 89.864, 89.95, 89.614, 89.817, 89.587, 90.156, 105.449, 93.923]
INTJ adapter kwargs                 median_ns=107.4 samples_ns=[107.804, 106.948, 108.57, 106.933, 106.085, 107.384, 107.198, 110.531, 108.43]
INTJ adapter defaults               median_ns=89.9 samples_ns=[91.068, 90.071, 89.945, 89.224, 89.723, 90.499, 89.882, 89.896, 89.486]
INTJ FFI wrapper positional         median_ns=103.1 samples_ns=[114.401, 103.841, 102.028, 103.206, 101.913, 103.061, 103.129, 102.773, 102.309]
INTJ FFI wrapper kwargs             median_ns=126.7 samples_ns=[134.41, 125.276, 126.42, 127.01, 126.184, 126.18, 126.744, 127.099, 130.037]
INTJ native kwargs: unsupported (verified TypeError)
mode=dataclass
INTJ pair direct                    median_ns=68.0 samples_ns=[70.525, 68.344, 67.503, 68.099, 67.274, 68.005, 67.47, 68.188, 67.256]
INTJ pair prebuilt *tuple           median_ns=139.6 samples_ns=[141.822, 140.149, 136.191, 139.645, 136.157, 140.104, 135.636, 143.616, 136.036]
FFI unpack Pair only                median_ns=161.9 samples_ns=[161.787, 161.851, 161.294, 163.883, 171.232, 162.558, 161.636, 162.087, 160.798]
INTJ pair manual unpack             median_ns=169.0 samples_ns=[167.878, 173.412, 169.024, 171.839, 168.737, 169.891, 167.946, 175.672, 168.205]
INTJ pair FFI unpack                median_ns=311.4 samples_ns=[311.373, 313.937, 311.73, 311.554, 307.222, 315.396, 310.526, 309.436, 310.481]
INTJ pair stdlib astuple            median_ns=1156.9 samples_ns=[1339.499, 1204.608, 1139.325, 1158.088, 1156.923, 1163.58, 1152.756, 1156.526, 1135.189]
INTJ config direct                  median_ns=81.7 samples_ns=[81.806, 89.245, 81.4, 81.723, 81.495, 81.662, 81.187, 81.732, 81.096]
INTJ config FFI unpack              median_ns=326.2 samples_ns=[321.822, 332.073, 318.911, 321.769, 327.137, 328.176, 326.24, 328.024, 321.347]
INTJ native dataclass unpack: unsupported (verified TypeError)
exit_status=0
ended=2026-09-25T18:34:29.991582+00:00
```

### ffi_paths: develop r1

```text
commit=1bf689b37137e373b6baecf241a8d8a7da23630b
source_sha256=7faebcc700fe0b5bd1363acfc638b0621eebf7af754ab3dcf915ebd0c4e41806
started=2026-09-25T18:34:29.993967+00:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_intj_ffi_paths.py --mode=all --iters 1000 --batches 9
iters=1000 batches=9 unit=ns/call; no_gpu=True
mode=callback
callback_count=9100 (100 warmup + 9000 unique misses)
INTJ cold compile callback + cache  median_ns=804.6 samples_ns=[843.315, 847.67, 776.628, 987.884, 756.77, 777.263, 792.205, 804.616, 2224.169]
INTJ hot call (no callback)         median_ns=66.0 samples_ns=[68.01, 67.361, 64.588, 66.036, 64.949, 73.279, 65.702, 66.034, 65.581]
INTJ 3-tensor host-only nop         median_ns=61.8 samples_ns=[63.823, 62.471, 61.192, 62.901, 60.811, 61.582, 61.829, 61.901, 61.382]
mode=kwargs
INTJ direct positional              median_ns=114.7 samples_ns=[118.874, 120.564, 114.742, 115.217, 114.698, 114.09, 114.888, 114.392, 114.307]
INTJ adapter positional             median_ns=169.6 samples_ns=[173.299, 169.573, 168.107, 166.567, 170.963, 170.757, 173.591, 167.923, 169.408]
INTJ adapter kwargs                 median_ns=213.2 samples_ns=[205.072, 205.235, 220.542, 213.813, 208.691, 213.381, 205.557, 213.247, 215.864]
INTJ adapter defaults               median_ns=171.1 samples_ns=[171.066, 171.571, 173.731, 175.544, 169.861, 170.997, 172.608, 170.308, 170.009]
INTJ FFI wrapper positional         median_ns=198.9 samples_ns=[212.538, 203.58, 198.422, 197.457, 195.888, 204.074, 195.378, 205.588, 198.919]
INTJ FFI wrapper kwargs             median_ns=258.4 samples_ns=[261.804, 261.602, 258.361, 256.031, 257.103, 260.186, 260.578, 248.076, 255.934]
INTJ native kwargs: unsupported (verified TypeError)
mode=dataclass
INTJ pair direct                    median_ns=131.6 samples_ns=[143.513, 130.7, 129.618, 133.502, 124.913, 131.594, 131.915, 131.372, 133.096]
INTJ pair prebuilt *tuple           median_ns=299.8 samples_ns=[297.673, 300.6, 300.944, 293.665, 297.539, 304.738, 299.828, 301.043, 293.585]
FFI unpack Pair only                median_ns=344.3 samples_ns=[352.256, 351.622, 344.284, 346.567, 340.023, 343.125, 341.264, 351.745, 337.633]
INTJ pair manual unpack             median_ns=351.6 samples_ns=[350.594, 347.67, 351.996, 347.281, 352.012, 353.357, 351.599, 348.06, 356.596]
INTJ pair FFI unpack                median_ns=626.9 samples_ns=[616.77, 638.703, 621.616, 627.06, 626.915, 629.229, 620.15, 636.489, 612.373]
INTJ pair stdlib astuple            median_ns=2236.3 samples_ns=[2410.819, 2274.58, 2245.873, 2236.283, 2195.55, 2195.047, 2193.865, 2246.749, 2195.311]
INTJ config direct                  median_ns=139.5 samples_ns=[139.91, 140.904, 139.52, 140.256, 142.788, 138.342, 138.498, 138.832, 138.476]
INTJ config FFI unpack              median_ns=626.1 samples_ns=[621.609, 626.954, 626.08, 628.522, 618.974, 630.733, 624.757, 624.607, 627.574]
INTJ native dataclass unpack: unsupported (verified TypeError)
exit_status=0
ended=2026-09-25T18:34:32.951723+00:00
```

### launch_gpu: candidate r1

```text
commit=c5295b18f29a15fc1184eb7e93dbc099e6b75bf2
source_sha256=50f04d4c6bc58bd50a029525208892353ac1ba7184540ca7088d3ebaa9c2cb3f
started=2026-09-25T18:34:57.619256+00:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/callable_grid
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/callable_grid taskset -c 0 /tmp/gb2/bin/python /tmp/intj_develop_baseline_2026-09-26_1bf689b/record_launch_batches.py benchmarks/bench_launch.py --iters 1000 --batches 9
mode=gpu; 1000 calls × 9 batches; median ns/call
               path    ns/call       Δ ns       Δ %   build ms   bind ms
recorded_batch_samples_ns=[3159.035, 3526.405, 3088.842, 3122.08, 3079.35, 3097.063, 3101.806, 3106.164, 3104.232]
           auto map     3104.2       +0.0      +0.0      74.10         -
recorded_batch_samples_ns=[3023.62, 3232.669, 3081.652, 3099.048, 3098.707, 3168.446, 3115.565, 3160.436, 3099.81]
        reduced key     3099.8       -4.4      -0.1      74.95         -
recorded_batch_samples_ns=[2970.691, 3069.026, 3114.92, 3110.259, 3358.624, 3154.626, 3106.81, 3026.321, 3031.408]
         verify off     3106.8       +2.6      +0.1       2.03         -
recorded_batch_samples_ns=[2897.891, 3044.514, 3014.61, 3014.055, 3013.756, 3007.444, 2985.598, 2961.633, 2945.892]
          verify on     3007.4      -96.8      -3.1       1.91         -
recorded_batch_samples_ns=[2832.234, 3029.436, 3019.725, 3005.688, 3014.132, 3000.313, 3010.213, 2957.099, 2955.669]
              baked     3005.7      -98.5      -3.2       1.99         -
recorded_batch_samples_ns=[2835.431, 2983.501, 2995.49, 2981.568, 3006.479, 3009.962, 2993.945, 2979.626, 2989.82]
       bound tensor     2989.8     -114.4      -3.7       0.36      1.79
recorded_batch_samples_ns=[2854.731, 3025.258, 3002.536, 2996.812, 3012.252, 2976.356, 2975.575, 2974.95, 2978.937]
      bound pointer     2978.9     -125.3      -4.0       0.36      1.81
recorded_batch_samples_ns=[2824.809, 2932.706, 2930.317, 3033.553, 3022.152, 3004.881, 3008.003, 2999.519, 2988.909]
   fixed device map     2999.5     -104.7      -3.4       0.31      1.78
recorded_batch_samples_ns=[2797.156, 2879.942, 2911.812, 2884.954, 2898.716, 2888.394, 2881.951, 2874.697, 2868.477]
fixed device no-map     2882.0     -222.3      -7.2       0.36      1.75
exit_status=0
ended=2026-09-25T18:35:01.434302+00:00
```

### launch_gpu: develop r1

```text
commit=1bf689b37137e373b6baecf241a8d8a7da23630b
source_sha256=7faebcc700fe0b5bd1363acfc638b0621eebf7af754ab3dcf915ebd0c4e41806
started=2026-09-25T18:35:01.436688+00:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj taskset -c 0 /tmp/gb2/bin/python /tmp/intj_develop_baseline_2026-09-26_1bf689b/record_launch_batches.py benchmarks/bench_launch.py --iters 1000 --batches 9
mode=gpu; 1000 calls × 9 batches; median ns/call
               path    ns/call       Δ ns       Δ %   build ms   bind ms
recorded_batch_samples_ns=[3159.981, 3481.429, 3221.719, 3037.373, 3122.418, 3098.018, 3109.01, 3088.229, 3087.127]
           auto map     3109.0       +0.0      +0.0      73.14         -
recorded_batch_samples_ns=[2941.973, 3161.705, 3094.276, 3109.974, 3107.653, 3102.981, 3105.636, 3009.987, 2966.172]
        reduced key     3103.0       -6.0      -0.2      74.29         -
recorded_batch_samples_ns=[2807.68, 2963.728, 2956.161, 2971.872, 2951.629, 2986.878, 5568.177, 4185.812, 4601.154]
         verify off     2971.9     -137.1      -4.4       1.70         -
recorded_batch_samples_ns=[2936.798, 2968.797, 2975.844, 3001.026, 2978.394, 2952.416, 2975.884, 2924.122, 2923.737]
          verify on     2968.8     -140.2      -4.5      12.39         -
recorded_batch_samples_ns=[2720.458, 2988.018, 2998.073, 3006.678, 3043.592, 3023.478, 3037.26, 2972.692, 2971.286]
              baked     2998.1     -110.9      -3.6       1.76         -
recorded_batch_samples_ns=[2748.1, 3002.601, 3039.888, 3018.979, 3032.971, 3015.065, 3004.984, 2993.059, 2998.32]
       bound tensor     3005.0     -104.0      -3.3       0.36      1.43
recorded_batch_samples_ns=[2748.685, 2920.03, 2927.33, 2909.494, 2923.98, 2907.063, 2916.323, 2905.657, 2915.46]
      bound pointer     2915.5     -193.6      -6.2       0.34      1.49
recorded_batch_samples_ns=[2796.052, 2842.828, 2895.629, 2874.519, 2872.846, 2887.713, 2885.078, 2876.01, 2888.352]
   fixed device map     2876.0     -233.0      -7.5       0.40      1.54
recorded_batch_samples_ns=[2768.399, 2884.518, 2927.906, 2905.788, 2901.051, 2917.504, 2928.363, 2926.884, 3020.143]
fixed device no-map     2917.5     -191.5      -6.2       0.27      1.52
exit_status=0
ended=2026-09-25T18:35:05.276833+00:00
```

### launch_readme: candidate r1

```text
commit=c5295b18f29a15fc1184eb7e93dbc099e6b75bf2
source_sha256=50f04d4c6bc58bd50a029525208892353ac1ba7184540ca7088d3ebaa9c2cb3f
started=2026-09-25T18:35:13.071231+00:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/callable_grid
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/callable_grid taskset -c 0 /tmp/gb2/bin/python /tmp/intj_develop_baseline_2026-09-26_1bf689b/record_launch_batches.py benchmarks/bench_launch.py --readme --iters 1000 --batches 9
recorded_batch_samples_ns=[96.356, 92.4, 90.871, 91.913, 90.858, 91.599, 89.17, 88.977, 89.896]
recorded_batch_samples_ns=[93.713, 96.365, 91.901, 94.318, 91.364, 90.047, 89.797, 91.577, 92.789]
recorded_batch_samples_ns=[874.199, 886.674, 880.18, 880.188, 871.193, 874.155, 871.723, 893.087, 877.856]
mode=readme; 1000 calls × 9 batches; median
        path   triton us    intj us   speedup
recorded_batch_samples_ns=[16623.553, 16556.378, 16421.236, 16491.753, 16335.866, 16490.523, 16342.15, 16510.626, 16336.168]
recorded_batch_samples_ns=[3049.322, 3255.51, 3100.036, 3094.547, 3114.608, 3091.56, 3110.326, 2953.111, 2944.996]
   grid=(1,)       16.49       3.09      5.3x
recorded_batch_samples_ns=[13031.299, 13219.856, 13042.704, 12971.684, 12980.241, 12948.249, 12961.84, 12958.732, 12984.639]
recorded_batch_samples_ns=[28.184, 27.902, 27.788, 27.916, 27.954, 27.796, 27.811, 27.794, 27.785]
   grid=(0,)       12.98       0.03    466.7x

torch_access_mode   decode ns    build s
     runtime_shim        90.9       0.02
   static_compile        91.9       0.06
      interpreter       877.9       0.00
exit_status=0
ended=2026-09-25T18:35:16.944748+00:00
```

### launch_readme: develop r1

```text
commit=1bf689b37137e373b6baecf241a8d8a7da23630b
source_sha256=7faebcc700fe0b5bd1363acfc638b0621eebf7af754ab3dcf915ebd0c4e41806
started=2026-09-25T18:35:16.947144+00:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj taskset -c 0 /tmp/gb2/bin/python /tmp/intj_develop_baseline_2026-09-26_1bf689b/record_launch_batches.py benchmarks/bench_launch.py --readme --iters 1000 --batches 9
recorded_batch_samples_ns=[100.184, 92.628, 92.966, 92.335, 90.814, 88.969, 87.093, 87.965, 88.755]
recorded_batch_samples_ns=[91.192, 92.032, 90.243, 98.262, 88.801, 90.038, 87.642, 89.044, 88.828]
recorded_batch_samples_ns=[884.631, 877.666, 1063.733, 896.92, 859.56, 887.758, 881.172, 969.239, 873.671]
mode=readme; 1000 calls × 9 batches; median
        path   triton us    intj us   speedup
recorded_batch_samples_ns=[16364.024, 16394.722, 16189.858, 16185.353, 16079.584, 16157.056, 16049.688, 16163.057, 16069.241]
recorded_batch_samples_ns=[3033.211, 3254.141, 3102.687, 3103.315, 3086.359, 3169.922, 3095.645, 3012.407, 3004.655]
   grid=(1,)       16.16       3.10      5.2x
recorded_batch_samples_ns=[62250.044, 13510.594, 12976.722, 12910.919, 12944.633, 12907.459, 12897.546, 12885.456, 12902.304]
recorded_batch_samples_ns=[28.875, 31.925, 28.471, 28.348, 28.364, 28.528, 28.345, 28.528, 28.126]
   grid=(0,)       12.91       0.03    453.5x

torch_access_mode   decode ns    build s
     runtime_shim        90.8       0.02
   static_compile        90.0       0.06
      interpreter       884.6       0.00
exit_status=0
ended=2026-09-25T18:35:20.800883+00:00
```

### ffi_default: candidate r1

```text
commit=c5295b18f29a15fc1184eb7e93dbc099e6b75bf2
source_sha256=50f04d4c6bc58bd50a029525208892353ac1ba7184540ca7088d3ebaa9c2cb3f
started=2026-09-25T18:35:36.517261+00:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/callable_grid
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/callable_grid taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_ffi_compare.py --iters 1000 --batches 9
iters=1000 batches=9 unit=us/call (host-only, no timed sync)
FFI packed nop             median=0.1456 samples=[0.152555, 0.144352, 0.144719, 0.145745, 0.14631, 0.147499, 0.143234, 0.144394, 0.145646]
FFI typed nop              median=0.1500 samples=[0.149697, 0.15349000000000002, 0.150778, 0.152528, 0.14907900000000002, 0.149384, 0.147667, 0.15, 0.151453]
FFI empty kernel           median=3.3150 samples=[3.57497, 3.492924, 3.341148, 3.203777, 3.209993, 3.332291, 3.175966, 3.315047, 3.1844609999999998]
INTJ empty kernel          median=2.9976 samples=[3.198134, 3.088149, 3.136808, 2.921814, 2.907928, 2.947452, 2.997909, 2.9428319999999997, 2.99759]
INTJ fixed-device kernel   median=2.8746 samples=[2.971219, 3.0001480000000003, 3.099604, 2.925973, 2.874572, 2.794816, 2.793145, 2.816271, 2.797373]
FFI packed nop mixed       median=0.1743 samples=[0.175348, 0.17437899999999998, 0.17432499999999998, 0.173267, 0.175371, 0.17191800000000002, 0.17359200000000002, 0.17490199999999997, 0.174041]
FFI typed nop mixed        median=0.1780 samples=[0.17781899999999998, 0.178927, 0.178809, 0.178186, 0.17660900000000002, 0.177957, 0.179156, 0.176707, 0.177066]
FFI mixed kernel           median=3.2571 samples=[3.4106840000000003, 3.5225619999999997, 3.3920500000000002, 3.257067, 3.2042710000000003, 3.246558, 3.2384020000000002, 3.338117, 3.240197]
INTJ mixed kernel          median=2.9380 samples=[3.0656529999999997, 3.047024, 3.1082199999999998, 2.937962, 2.859274, 2.855234, 2.814849, 2.9507719999999997, 2.804798]
INTJ fixed mixed kernel    median=2.8850 samples=[3.064487, 3.061261, 2.9663530000000002, 2.970704, 2.868563, 2.851142, 2.879975, 2.879494, 2.884964]
exit_status=0
ended=2026-09-25T18:35:40.349712+00:00
```

### ffi_default: develop r1

```text
commit=1bf689b37137e373b6baecf241a8d8a7da23630b
source_sha256=7faebcc700fe0b5bd1363acfc638b0621eebf7af754ab3dcf915ebd0c4e41806
started=2026-09-25T18:35:40.351987+00:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_ffi_compare.py --iters 1000 --batches 9
iters=1000 batches=9 unit=us/call (host-only, no timed sync)
FFI packed nop             median=0.1446 samples=[0.243375, 0.147415, 0.144614, 0.144755, 0.14596, 0.14336500000000002, 0.142236, 0.142663, 0.142911]
FFI typed nop              median=0.1494 samples=[0.24719300000000002, 0.149435, 0.148115, 0.152494, 0.151381, 0.150112, 0.146666, 0.14759999999999998, 0.149449]
FFI empty kernel           median=3.3676 samples=[4.484795, 3.5242910000000003, 3.3676120000000003, 3.230764, 3.25675, 3.3731210000000003, 3.201035, 3.5052060000000003, 3.20872]
INTJ empty kernel          median=2.9848 samples=[3.088578, 3.066876, 3.131599, 2.922009, 2.898029, 2.989989, 2.9848090000000003, 2.963667, 2.981962]
INTJ fixed-device kernel   median=2.8994 samples=[2.9948609999999998, 3.005283, 3.122062, 2.933937, 2.899443, 2.805939, 2.7925050000000002, 2.7934989999999997, 2.80883]
FFI packed nop mixed       median=0.1758 samples=[0.174241, 0.176903, 0.176369, 0.177132, 0.177477, 0.172354, 0.17317, 0.17583600000000002, 0.173875]
FFI typed nop mixed        median=0.1791 samples=[0.17955600000000002, 0.18096, 0.178234, 0.17924199999999998, 0.17781200000000003, 0.17912299999999998, 0.179081, 0.17907599999999999, 0.17561500000000002]
FFI mixed kernel           median=3.3125 samples=[3.427657, 3.561975, 3.399641, 3.312481, 3.2376329999999998, 3.304446, 3.275661, 3.401425, 3.282702]
INTJ mixed kernel          median=2.9623 samples=[3.1039760000000003, 3.111143, 2.962311, 2.9769859999999997, 2.9010819999999997, 2.906621, 2.83416, 2.979596, 2.8170889999999997]
INTJ fixed mixed kernel    median=2.9232 samples=[3.0523029999999998, 3.1240419999999998, 2.98871, 3.003143, 2.892128, 2.876036, 2.899056, 2.9232139999999998, 2.911024]
exit_status=0
ended=2026-09-25T18:35:44.309411+00:00
```

### ffi_sweep: candidate r1

```text
commit=c5295b18f29a15fc1184eb7e93dbc099e6b75bf2
source_sha256=50f04d4c6bc58bd50a029525208892353ac1ba7184540ca7088d3ebaa9c2cb3f
started=2026-09-25T18:36:25.347165+00:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/callable_grid
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/callable_grid taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_ffi_compare.py --sweep --iters 1000 --batches 9
counts=[0, 3, 5, 8, 16, 32, 64] iters=1000 batches=9 unit=us/call (host-only, no timed sync); INTJ device-bound, no specialization key
args= 0 FFI packed nop           median=0.1179 samples=[0.126427, 0.117895, 0.119243, 0.118172, 0.117404, 0.117157, 0.1185, 0.117272, 0.117453]
args= 0 FFI typed nop            median=0.1195 samples=[0.12074599999999999, 0.119485, 0.118146, 0.124274, 0.117103, 0.121113, 0.118435, 0.12260299999999999, 0.11771899999999999]
args= 0 FFI empty kernel         median=1.6502 samples=[1.602029, 1.874164, 1.6502439999999998, 1.861304, 1.643488, 1.8520029999999998, 1.641902, 1.8022799999999999, 1.5753320000000002]
args= 0 INTJ static_compile kernel median=2.7631 samples=[3.1406039999999997, 2.674973, 2.7630630000000003, 2.6657640000000002, 2.765138, 2.78071, 2.7713200000000002, 2.726694, 2.7150079999999996]
args= 0 INTJ runtime_shim kernel median=2.7349 samples=[2.914502, 2.665482, 2.766403, 2.663493, 2.7532330000000003, 2.734933, 2.790511, 2.676663, 2.707725]
args= 3 FFI packed nop           median=0.1457 samples=[0.147921, 0.14674199999999998, 0.158525, 0.14643799999999998, 0.145012, 0.145718, 0.144172, 0.144291, 0.145687]
args= 3 FFI typed nop            median=0.1486 samples=[0.14701599999999998, 0.148189, 0.147657, 0.150815, 0.150159, 0.15071600000000002, 0.14488399999999999, 0.151083, 0.148557]
args= 3 FFI empty kernel         median=3.2038 samples=[3.359554, 3.179945, 3.139511, 3.203833, 3.144622, 3.277182, 3.261204, 3.219658, 3.181746]
args= 3 INTJ static_compile kernel median=2.8698 samples=[3.002051, 2.831688, 2.963759, 2.860721, 2.9574059999999998, 2.900931, 2.8698420000000002, 2.8286930000000003, 2.836613]
args= 3 INTJ runtime_shim kernel median=2.8458 samples=[3.047067, 2.826398, 2.7889209999999998, 2.876193, 2.8767069999999997, 2.877585, 2.838493, 2.823365, 2.8458229999999998]
args= 5 FFI packed nop           median=0.1747 samples=[0.17424199999999998, 0.174778, 0.174524, 0.174954, 0.177008, 0.173262, 0.17569300000000002, 0.174713, 0.17407499999999998]
args= 5 FFI typed nop            median=0.1776 samples=[0.17707699999999998, 0.179024, 0.176541, 0.177931, 0.175676, 0.180233, 0.177561, 0.17619100000000001, 0.181177]
args= 5 FFI empty kernel         median=3.2652 samples=[3.4013679999999997, 3.2488490000000003, 3.201944, 3.265182, 3.260389, 3.372828, 3.258368, 3.2917289999999997, 3.267967]
args= 5 INTJ static_compile kernel median=2.8611 samples=[3.014306, 2.882796, 2.826105, 2.8395659999999996, 2.864583, 2.8871089999999997, 2.855764, 2.824366, 2.861067]
args= 5 INTJ runtime_shim kernel median=2.8412 samples=[3.0399209999999997, 2.948351, 2.823218, 2.856883, 2.858624, 2.820852, 2.8411779999999998, 2.790791, 2.8254859999999997]
args= 8 FFI packed nop           median=0.2057 samples=[0.20780500000000002, 0.205582, 0.20425100000000002, 0.206214, 0.20583400000000002, 0.20291599999999999, 0.205667, 0.203715, 0.205758]
args= 8 FFI typed nop            median=0.2099 samples=[0.209695, 0.211406, 0.207617, 0.209645, 0.212027, 0.21351699999999998, 0.209905, 0.210807, 0.208991]
args= 8 FFI empty kernel         median=3.4060 samples=[3.526904, 3.4803, 3.264851, 3.4072959999999997, 3.3187330000000004, 16.41244, 3.268868, 3.405986, 3.275591]
args= 8 INTJ static_compile kernel median=3.0666 samples=[3.076546, 2.972014, 3.095191, 3.000471, 3.045565, 12.669300999999999, 3.0791720000000002, 2.935395, 3.06658]
args= 8 INTJ runtime_shim kernel median=2.8955 samples=[3.0728139999999997, 2.914956, 2.956275, 2.88281, 2.865086, 8.879234, 2.8618560000000004, 2.895529, 2.87392]
args=16 FFI packed nop           median=0.2925 samples=[0.290128, 0.295868, 0.295647, 0.2917, 0.297556, 0.29247300000000004, 0.295979, 0.28744099999999995, 0.290571]
args=16 FFI typed nop            median=0.3038 samples=[0.302012, 0.307087, 0.303882, 0.303524, 0.30419799999999997, 0.30780399999999997, 0.299553, 0.303844, 0.303812]
args=16 FFI empty kernel         median=3.6709 samples=[3.713795, 3.7238800000000003, 3.514591, 3.6709479999999997, 3.5231149999999998, 3.687651, 3.528349, 3.6830369999999997, 3.5335]
args=16 INTJ static_compile kernel median=3.3861 samples=[3.279673, 3.386404, 3.494107, 3.150717, 3.483438, 3.3312600000000003, 3.4831619999999996, 3.386092, 3.330578]
args=16 INTJ runtime_shim kernel median=3.2999 samples=[3.299922, 3.158599, 3.340259, 3.053357, 3.341217, 3.034088, 3.351632, 3.048766, 3.447714]
args=32 FFI packed nop           median=0.4781 samples=[0.467257, 0.478117, 0.49077800000000005, 0.476664, 0.479162, 0.473993, 0.48272000000000004, 0.472625, 0.479506]
args=32 FFI typed nop            median=0.4922 samples=[0.490862, 0.494525, 0.504187, 0.49214800000000003, 0.493156, 0.488468, 0.49067200000000005, 0.494542, 0.492241]
args=32 FFI empty kernel         median=4.2415 samples=[4.347406, 4.265827, 4.112389, 4.2531289999999995, 4.085224, 4.241466999999999, 4.088779, 4.280365, 4.067449]
args=32 INTJ static_compile kernel median=3.6195 samples=[3.504818, 3.6584119999999998, 3.6195, 3.5694529999999998, 3.674553, 3.606847, 3.6672469999999997, 3.616007, 3.6644270000000003]
args=32 INTJ runtime_shim kernel median=3.5098 samples=[3.4313789999999997, 3.509832, 3.588156, 3.421453, 3.598386, 3.420066, 3.634453, 3.4410659999999997, 3.6151060000000004]
args=64 FFI packed nop           median=0.8462 samples=[0.8407899999999999, 0.840342, 0.8535900000000001, 0.838563, 0.846523, 0.817399, 0.846228, 0.852029, 0.869262]
args=64 FFI typed nop            median=0.8710 samples=[0.871014, 0.875803, 0.854623, 0.878198, 0.853916, 0.857369, 0.861906, 0.892197, 0.8844500000000001]
args=64 FFI empty kernel         median=5.0304 samples=[5.077751, 5.108847, 5.014411, 5.030257, 5.022649, 5.030397, 5.058035, 5.049272, 5.023442]
args=64 INTJ static_compile kernel median=4.5656 samples=[4.712141, 4.565618000000001, 4.545503, 4.572665, 4.557558, 4.559665, 4.579592, 4.592102, 4.560747]
args=64 INTJ runtime_shim kernel median=4.5453 samples=[4.5882510000000005, 4.600744, 4.472139, 4.535046, 4.545274, 4.557282, 4.519528, 4.552898, 4.520961000000001]
exit_status=0
ended=2026-09-25T18:36:29.770998+00:00
```

### ffi_sweep: develop r1

```text
commit=1bf689b37137e373b6baecf241a8d8a7da23630b
source_sha256=7faebcc700fe0b5bd1363acfc638b0621eebf7af754ab3dcf915ebd0c4e41806
started=2026-09-25T18:36:29.773502+00:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_ffi_compare.py --sweep --iters 1000 --batches 9
counts=[0, 3, 5, 8, 16, 32, 64] iters=1000 batches=9 unit=us/call (host-only, no timed sync); INTJ device-bound, no specialization key
args= 0 FFI packed nop           median=0.1158 samples=[0.119497, 0.117089, 0.115086, 0.11507899999999999, 0.117805, 0.115162, 0.115788, 0.11756699999999999, 0.113702]
args= 0 FFI typed nop            median=0.1190 samples=[0.12090999999999999, 0.11793600000000001, 0.118958, 0.122075, 0.118601, 0.12334999999999999, 0.117285, 0.11973099999999999, 0.11658]
args= 0 FFI empty kernel         median=1.6398 samples=[1.594746, 1.930862, 1.624962, 1.907888, 1.6398119999999998, 1.901562, 1.5982539999999998, 1.8886479999999999, 1.6085820000000002]
args= 0 INTJ static_compile kernel median=2.7132 samples=[3.0763000000000003, 2.714213, 2.713214, 2.675024, 2.768686, 2.652655, 2.7095819999999997, 2.7045030000000003, 2.725015]
args= 0 INTJ runtime_shim kernel median=2.7055 samples=[2.828967, 2.691767, 2.731944, 2.681511, 2.764525, 2.6915500000000003, 2.7055230000000003, 2.6999389999999996, 2.719689]
args= 3 FFI packed nop           median=0.1480 samples=[0.145363, 0.147395, 0.148003, 0.148114, 0.149315, 0.146307, 0.153838, 0.146227, 0.148448]
args= 3 FFI typed nop            median=0.1512 samples=[0.148864, 0.152327, 0.150602, 0.154552, 0.15617799999999998, 0.151177, 0.15081999999999998, 0.155366, 0.14932800000000002]
args= 3 FFI empty kernel         median=3.2227 samples=[3.3443609999999997, 3.2227159999999997, 3.1875430000000002, 3.2258470000000004, 3.17467, 3.302525, 3.165022, 3.246156, 3.189269]
args= 3 INTJ static_compile kernel median=2.9592 samples=[2.9592289999999997, 2.8423580000000004, 2.974851, 2.860332, 2.991255, 2.913742, 2.973705, 2.879098, 3.006942]
args= 3 INTJ runtime_shim kernel median=2.8889 samples=[2.961767, 2.8377339999999998, 2.911616, 2.850882, 2.890755, 2.9104870000000003, 2.8867510000000003, 2.883575, 2.8888789999999998]
args= 5 FFI packed nop           median=0.1760 samples=[0.173657, 0.17758000000000002, 0.175961, 0.175181, 0.17755500000000002, 0.176595, 0.17722, 0.173877, 0.172494]
args= 5 FFI typed nop            median=0.1797 samples=[0.179707, 0.180876, 0.17762899999999998, 0.19558099999999998, 0.18106, 0.17781, 0.182791, 0.17956, 0.17666300000000001]
args= 5 FFI empty kernel         median=3.3284 samples=[3.3593319999999998, 3.336147, 3.2321489999999997, 3.3186869999999997, 3.30125, 3.369078, 3.3284119999999997, 3.355521, 3.297004]
args= 5 INTJ static_compile kernel median=2.8942 samples=[2.982842, 2.936418, 2.889998, 2.894243, 2.890058, 2.9048629999999998, 2.893624, 2.9100770000000002, 2.884778]
args= 5 INTJ runtime_shim kernel median=2.8592 samples=[2.971205, 2.859226, 2.8434690000000002, 2.764897, 2.864681, 2.772723, 2.859229, 2.793609, 2.862038]
args= 8 FFI packed nop           median=0.2086 samples=[0.208082, 0.208647, 0.21121299999999998, 0.208643, 0.208372, 0.209081, 0.210946, 0.208328, 0.210274]
args= 8 FFI typed nop            median=0.2140 samples=[0.20852500000000002, 0.21569999999999998, 0.210832, 0.217398, 0.211707, 0.21771600000000002, 0.211279, 0.213952, 0.215974]
args= 8 FFI empty kernel         median=3.4056 samples=[3.459078, 3.451828, 3.2999699999999996, 3.405605, 3.316388, 3.636325, 3.307946, 3.429862, 3.303852]
args= 8 INTJ static_compile kernel median=3.0859 samples=[3.0858659999999998, 2.9828609999999998, 3.131162, 3.0383090000000004, 3.101554, 3.071281, 3.101016, 3.050199, 3.119912]
args= 8 INTJ runtime_shim kernel median=2.9043 samples=[3.04084, 2.935588, 2.9695929999999997, 2.893551, 3.0151529999999998, 2.903861, 2.882688, 2.904303, 2.89609]
args=16 FFI packed nop           median=0.3024 samples=[0.30428, 0.303084, 0.29844, 0.304073, 0.301113, 0.299402, 0.30311, 0.299647, 0.302356]
args=16 FFI typed nop            median=0.3112 samples=[0.312025, 0.309311, 0.30804000000000004, 0.31313, 0.311998, 0.30860000000000004, 0.31048899999999996, 0.311819, 0.31121499999999996]
args=16 FFI empty kernel         median=3.6573 samples=[3.7270390000000004, 3.71029, 3.567361, 3.6144760000000002, 3.56337, 3.6572910000000003, 3.710483, 3.7026190000000003, 3.576487]
args=16 INTJ static_compile kernel median=3.3243 samples=[3.256547, 3.390265, 3.4566660000000002, 3.24711, 3.367832, 3.259645, 3.324321, 3.3146970000000002, 3.501866]
args=16 INTJ runtime_shim kernel median=3.2357 samples=[3.235701, 3.156471, 3.335322, 3.033, 3.468112, 3.046283, 3.280545, 3.055485, 3.334901]
args=32 FFI packed nop           median=0.4937 samples=[0.48846300000000004, 0.48683, 0.49435399999999996, 0.496839, 0.493702, 0.494255, 0.48956, 0.485643, 0.494718]
args=32 FFI typed nop            median=0.5093 samples=[0.516854, 0.513258, 0.504974, 0.507452, 0.50933, 0.5130030000000001, 0.5017389999999999, 0.517476, 0.502258]
args=32 FFI empty kernel         median=4.2508 samples=[4.472614999999999, 4.250827, 4.1087489999999995, 4.246878, 4.117247, 4.252507, 4.349633, 4.252275, 4.117583]
args=32 INTJ static_compile kernel median=3.6146 samples=[3.465721, 3.614558, 3.636342, 3.5349250000000003, 3.6362379999999996, 3.558822, 3.804453, 3.5500010000000004, 3.978475]
args=32 INTJ runtime_shim kernel median=3.4918 samples=[3.448774, 3.4917860000000003, 3.671504, 3.45199, 13.579246, 3.4528600000000003, 3.6326359999999998, 3.4595160000000003, 18.855482]
args=64 FFI packed nop           median=0.8695 samples=[0.838208, 0.8647469999999999, 0.8855019999999999, 0.869506, 1.996279, 0.828266, 0.8721319999999999, 0.859741, 1.97344]
args=64 FFI typed nop            median=0.8869 samples=[0.870297, 0.886853, 0.880115, 0.9044160000000001, 1.5508499999999998, 0.863536, 0.8719349999999999, 0.8925879999999999, 1.928337]
args=64 FFI empty kernel         median=5.0971 samples=[5.097899999999999, 5.106123999999999, 5.097134, 5.05746, 6.780551999999999, 5.087463, 5.084821, 5.091551000000001, 7.371341999999999]
args=64 INTJ static_compile kernel median=4.6286 samples=[21.886212, 4.677603, 4.599439, 4.628614, 6.117222, 4.606952000000001, 4.6139719999999995, 4.621643, 4.656688]
args=64 INTJ runtime_shim kernel median=4.5949 samples=[4.613415, 4.644517, 4.508152, 4.569893, 5.815854, 4.594887, 4.550437, 4.620992, 4.539549]
exit_status=0
ended=2026-09-25T18:36:34.142904+00:00
```

### hip_same_hsaco: candidate r1

```text
commit=c5295b18f29a15fc1184eb7e93dbc099e6b75bf2
source_sha256=50f04d4c6bc58bd50a029525208892353ac1ba7184540ca7088d3ebaa9c2cb3f
started=2026-09-25T18:36:43.984413+00:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/callable_grid
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/callable_grid taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_hip_module_launch.py --iters 1000 --batches 9
hsaco_sha256=df3e529e178196ad62f7c07ab8593ede5b8596c76e7c6fbd12a5d947a696e65c kernel=empty_kernel threads=256 shared=0 callback_count=1
iters=1000 batches=9 unit=us/call (host enqueue, no timed sync)
TVM FFI HIP HSACO         median=3.1085 samples=[3.415038, 3.169197, 3.161247, 3.114359, 3.073561, 3.027145, 3.108538, 3.057068, 3.042881]
Triton same HSACO         median=16.0776 samples=[16.203562, 16.085926999999998, 16.077597, 16.105788, 15.990442, 15.968326, 15.946886000000001, 24.698691999999998, 15.907939]
INTJ same function        median=2.9549 samples=[2.9581039999999996, 2.956302, 2.954927, 2.9673119999999997, 2.96764, 2.902002, 2.863596, 2.898971, 2.887533]
exit_status=0
ended=2026-09-25T18:36:47.751419+00:00
```

### hip_same_hsaco: develop r1

```text
commit=1bf689b37137e373b6baecf241a8d8a7da23630b
source_sha256=7faebcc700fe0b5bd1363acfc638b0621eebf7af754ab3dcf915ebd0c4e41806
started=2026-09-25T18:36:47.753878+00:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_hip_module_launch.py --iters 1000 --batches 9
hsaco_sha256=df3e529e178196ad62f7c07ab8593ede5b8596c76e7c6fbd12a5d947a696e65c kernel=empty_kernel threads=256 shared=0 callback_count=1
iters=1000 batches=9 unit=us/call (host enqueue, no timed sync)
TVM FFI HIP HSACO         median=3.1844 samples=[3.4603249999999997, 3.229905, 3.226318, 3.192831, 3.184408, 3.14308, 3.1477310000000003, 3.1613890000000002, 3.156781]
Triton same HSACO         median=16.0653 samples=[16.065253000000002, 16.194388, 16.243769, 16.165229, 15.945958000000001, 15.961113, 16.028585, 48.944837, 16.033542]
INTJ same function        median=2.9634 samples=[3.066462, 3.0802869999999998, 3.0344, 3.038063, 2.96342, 2.927707, 2.915292, 2.9270549999999997, 2.91123]
exit_status=0
ended=2026-09-25T18:36:51.599844+00:00
```

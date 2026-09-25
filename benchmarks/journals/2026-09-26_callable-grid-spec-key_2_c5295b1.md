# Callable-grid `spec_key` ABBA probe — 2026-09-26, repeat 2 of 4

[Aggregate, environment, script source, and timing caveats](2026-09-26_callable-grid-spec-key_0_c5295b1.md).
These two process outputs ran in D2 →
C2 order. They include nine-batch samples, exact
commands, UTC times, commits, and exit status. The local date is 2026-09-26
(Asia/Shanghai).

## Raw outputs — repeat 2

### develop r2

```text
commit=1bf689b37137e373b6baecf241a8d8a7da23630b
started=2026-09-25T18:43:23.310015+00:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj taskset -c 0 /tmp/gb2/bin/python /tmp/intj_spec_key_bench_2026-09-26.py --iters 100000 --batches 9
runtime_shim median_ns=120.1140 samples_ns=[118.81561, 120.48507, 118.35306, 119.34295, 121.80516, 120.58046, 120.11395, 120.1682, 117.89369]
static_compile median_ns=114.9326 samples_ns=[114.78849, 114.2496, 116.59294, 115.42201, 116.43573, 115.05383, 113.66958, 114.92815, 114.93261]
interpreter median_ns=909.6708 samples_ns=[910.38312, 1100.22044, 906.32705, 906.07278, 911.79441, 909.67079, 909.21188, 909.21374, 1062.92855]
exit_status=0
ended=2026-09-25T18:43:27.290931+00:00
```

### candidate r2

```text
commit=c5295b18f29a15fc1184eb7e93dbc099e6b75bf2
started=2026-09-25T18:43:27.293528+00:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/callable_grid
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/callable_grid taskset -c 0 /tmp/gb2/bin/python /tmp/intj_spec_key_bench_2026-09-26.py --iters 100000 --batches 9
runtime_shim median_ns=119.5039 samples_ns=[119.50387, 120.96912, 118.72087, 121.44551, 120.3734, 118.26374, 115.02537, 119.88823, 119.39146]
static_compile median_ns=116.6709 samples_ns=[119.3522, 115.63397, 115.2583, 114.95354, 115.35456, 116.82776, 118.07492, 116.67093, 116.92535]
interpreter median_ns=912.1350 samples_ns=[912.13504, 914.17138, 1080.92644, 915.36231, 916.21329, 909.771, 907.97545, 909.86161, 911.04027]
exit_status=0
ended=2026-09-25T18:43:31.244620+00:00
```

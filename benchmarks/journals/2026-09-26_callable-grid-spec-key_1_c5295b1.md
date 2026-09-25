# Callable-grid `spec_key` ABBA probe — 2026-09-26, repeat 1 of 4

[Aggregate, environment, script source, and timing caveats](2026-09-26_callable-grid-spec-key_0_c5295b1.md).
These two process outputs ran in C1 →
D1 order. They include nine-batch samples, exact
commands, UTC times, commits, and exit status. The local date is 2026-09-26
(Asia/Shanghai).

## Raw outputs — repeat 1

### candidate r1

```text
commit=c5295b18f29a15fc1184eb7e93dbc099e6b75bf2
started=2026-09-25T18:43:15.401977+00:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/callable_grid
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/callable_grid taskset -c 0 /tmp/gb2/bin/python /tmp/intj_spec_key_bench_2026-09-26.py --iters 100000 --batches 9
runtime_shim median_ns=119.3693 samples_ns=[120.55764, 119.82824, 119.74528, 119.86228, 119.36933, 117.57556, 118.07062, 116.82027, 114.92951]
static_compile median_ns=118.2014 samples_ns=[118.20634, 117.26594, 115.70977, 116.03922, 115.15582, 119.16319, 119.78404, 118.20138, 121.44966]
interpreter median_ns=904.9665 samples_ns=[904.63656, 930.1318, 1134.41909, 913.45027, 905.85534, 900.19437, 901.6071, 904.96652, 902.77834]
exit_status=0
ended=2026-09-25T18:43:19.326695+00:00
```

### develop r1

```text
commit=1bf689b37137e373b6baecf241a8d8a7da23630b
started=2026-09-25T18:43:19.329127+00:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj taskset -c 0 /tmp/gb2/bin/python /tmp/intj_spec_key_bench_2026-09-26.py --iters 100000 --batches 9
runtime_shim median_ns=116.4521 samples_ns=[118.66988, 118.70423, 117.45364, 117.93546, 116.45214, 114.97796, 113.66923, 116.20501, 115.33147]
static_compile median_ns=114.0653 samples_ns=[115.08856, 114.26734, 115.70746, 115.48741, 113.78244, 113.475, 113.07116, 113.83879, 114.06534]
interpreter median_ns=980.0412 samples_ns=[983.12011, 990.09607, 973.74498, 973.50825, 1107.54006, 978.01349, 980.04116, 978.7197, 1098.33111]
exit_status=0
ended=2026-09-25T18:43:23.307475+00:00
```

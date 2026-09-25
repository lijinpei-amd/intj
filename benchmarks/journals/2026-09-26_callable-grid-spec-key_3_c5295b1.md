# Callable-grid `spec_key` ABBA probe — 2026-09-26, repeat 3 of 4

[Aggregate, environment, script source, and timing caveats](2026-09-26_callable-grid-spec-key_0_c5295b1.md).
These two process outputs ran in C3 →
D3 order. They include nine-batch samples, exact
commands, UTC times, commits, and exit status. The local date is 2026-09-26
(Asia/Shanghai).

## Raw outputs — repeat 3

### candidate r3

```text
commit=c5295b18f29a15fc1184eb7e93dbc099e6b75bf2
started=2026-09-25T18:43:31.247176+00:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/callable_grid
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/callable_grid taskset -c 0 /tmp/gb2/bin/python /tmp/intj_spec_key_bench_2026-09-26.py --iters 100000 --batches 9
runtime_shim median_ns=121.2035 samples_ns=[123.08843, 123.66295, 123.13446, 123.04289, 121.20349, 121.09185, 118.31426, 121.15392, 119.27125]
static_compile median_ns=121.4238 samples_ns=[121.42379, 118.70429, 120.28921, 118.93771, 118.27308, 121.86663, 123.94053, 122.59561, 122.72832]
interpreter median_ns=909.8821 samples_ns=[907.20834, 911.05201, 911.98137, 909.88205, 913.0774, 910.91264, 907.9342, 905.36986, 903.67303]
exit_status=0
ended=2026-09-25T18:43:35.083280+00:00
```

### develop r3

```text
commit=1bf689b37137e373b6baecf241a8d8a7da23630b
started=2026-09-25T18:43:35.085668+00:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj taskset -c 0 /tmp/gb2/bin/python /tmp/intj_spec_key_bench_2026-09-26.py --iters 100000 --batches 9
runtime_shim median_ns=116.3984 samples_ns=[117.14299, 118.72109, 116.3984, 118.0342, 116.16176, 114.71707, 112.86113, 116.49868, 114.71028]
static_compile median_ns=114.0310 samples_ns=[113.41481, 114.46798, 114.03096, 114.56263, 114.17599, 113.77258, 113.08463, 113.19899, 115.62307]
interpreter median_ns=902.7371 samples_ns=[905.11541, 899.52515, 900.57066, 1005.20054, 901.05467, 902.73712, 906.71471, 900.60202, 1076.92324]
exit_status=0
ended=2026-09-25T18:43:39.088197+00:00
```

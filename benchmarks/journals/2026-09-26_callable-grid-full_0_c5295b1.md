# Callable-grid full ABBA comparison — 2026-09-26, repeat 0 of 2

The local measurement date is 2026-09-26 (Asia/Shanghai); log timestamps are
2026-09-25 UTC. Baseline `develop` is `1bf689b37137e373b6baecf241a8d8a7da23630b`;
candidate `callable_grid` is `c5295b18f29a15fc1184eb7e93dbc099e6b75bf2`.
Eight benchmark modes ran one process at a time in D0 → C0 → C1 → D1 order
*within each mode*. All 32 processes exited 0; each revision-round pass has
the same labeled measurements across the eight modes. This record has
round 0's 16 raw outputs; [repeat 1](2026-09-26_callable-grid-full_1_c5295b1.md) has the other 16.
The [targeted controls](2026-09-26_callable-grid-targeted_0_c5295b1.md) and
[longer spec-key probe](2026-09-26_callable-grid-spec-key_0_c5295b1.md) resolve noisy rows.

## Aggregate across the two full rounds

Each cell is one process median of nine timed batches. D = baseline, C =
candidate. The host rows are ns/call; the GPU rows are host enqueue µs/call.
No process was dropped from the raw record.

| Host row (ns/call) | D0 | C0 | C1 | D1 |
| --- | ---: | ---: | ---: | ---: |
| Bound tensor | 45.5 | 45.4 | 46.2 | 45.8 |
| Bound pointer | 44.0 | 43.9 | 44.0 | 44.0 |
| 16 integer arguments | 74.2 | 71.0 | 70.9 | 74.3 |
| 32 integer arguments | 120.9 | 111.0 | 110.5 | 121.8 |
| Static-compile `spec_key` | 90.3 | 91.4 | 91.9 | 90.0 |

The bound-tensor host path is near the baseline in both pairs; 32 integer
arguments improve by 9.9 and 11.3 ns. The README static-compile key row is
1.1 and 1.9 ns higher; the longer direct-key probe finds a small gap of similar
size in its runtime-shim control too.

| GPU row (µs/call) | D0 | C0 | C1 | D1 |
| --- | ---: | ---: | ---: | ---: |
| Launcher bound tensor | 3.5032 | 3.2436 | 2.9898 | 3.0050 |
| FFI comparison: INTJ empty kernel | 3.0012 | 2.9035 | 2.9976 | 2.9848 |
| FFI sweep, 16 args: INTJ static compile | 3.3331 | 5.4852 | 3.3861 | 3.3243 |
| FFI sweep, 16 args: INTJ runtime shim | 3.2681 | 4.9484 | 3.2999 | 3.2357 |
| FFI sweep, 16 args: FFI empty-kernel control | 3.5406 | 4.2932 | 3.6709 | 3.6573 |
| Same HSACO: INTJ | 2.9622 | 3.0566 | 2.9549 | 2.9634 |

The C0 FFI-sweep spike also raises FFI's independent kernel control, and C1
returns close to D1; [targeted sweep repeats](2026-09-26_callable-grid-targeted_0_c5295b1.md) include another
shared queue spike and three cleaner comparisons. D0 GPU launcher rows are
slower than D1 on several paths, so the apparent C0 gain is not a reliable
candidate speedup. The full `ffi_paths` D1 process is contaminated: INTJ
keyword/dataclass rows and the FFI-only `unpack Pair` control are roughly twice
as slow. Its raw output remains below, while [10,000-call utility controls](2026-09-26_callable-grid-targeted_0_c5295b1.md)
carry the comparison. After those controls, no repeatable material slowdown was
observed; this does not rule out smaller effects.

## Environment, command order, and timing boundary

- Intel Xeon Platinum 8480C, pinned to CPU 0 with `taskset -c 0`; AMD Instinct
  MI308X GPU 0, `gfx942:sramecc+:xnack-` (eight GPUs visible). Processes ran
  serially. The logs do not prove that other GPU users were absent.
- Python 3.12.3 at `/tmp/gb2/bin/python`, extension suffix
  `.cpython-312-x86_64-linux-gnu.so`; Torch `2.14.0+rocm7.2`
  (`torch.version.hip=7.2.53211`, C++11 ABI enabled), Triton `3.8.0`, TVM FFI
  `0.1.14.post2.dev1+g424558557.d20260924`.
- The baseline environment snapshot in
  `/tmp/intj_develop_baseline_2026-09-26_1bf689b/environment.txt` recorded
  `TRITON_HOME`, `TVM_FFI_CACHE_DIR`, `CC`, `CXX`, and all GPU visibility masks
  unset. The driver inherited its process environment and set only each
  checkout's `PYTHONPATH`. It did not clear caches; the logs do not establish
  a cold cache.
- The full driver `/tmp/intj_full_bench_abba_2026-09-26.py` required
  `INTJ_BENCH_OUT=/tmp/intj_full_bench_abba_2026-09-26_c5295b1`. Each raw block
  records its exact child command, checkout, UTC start/end, commit, and exit
  status. The driver recorded SHA-256 of `intj/runtime/entry.c.jinja`:
  baseline `7faebcc700fe0b5bd1363acfc638b0621eebf7af754ab3dcf915ebd0c4e41806`,
  candidate `50f04d4c6bc58bd50a029525208892353ac1ba7184540ca7088d3ebaa9c2cb3f`.
  Both match their committed Git blobs; whole-tree status was not logged.

The modes and exact call counts are: `launch_host` and `launch_sweep`: 100,000
calls × nine batches; `ffi_paths`, `launch_gpu`, `launch_readme`, `ffi_default`,
`ffi_sweep`, and `hip_same_hsaco`: 1,000 calls × nine batches. The `bench_launch.py`
modes run through the following recorder; its print occurs after the batch
timer. The raw `command=` lines give each mode flag and absolute checkout path.

```python
"""Print each bench_launch.py batch list after statistics.median, outside its timer."""
import os
import runpy
import statistics
import sys

original_median = statistics.median


def record_median(samples):
    result = original_median(samples)
    if isinstance(samples, list) and len(samples) == 9 and all(isinstance(x, float) for x in samples):
        print('recorded_batch_samples_ns=' + repr(samples), flush=True)
    return result


statistics.median = record_median
script = sys.argv[1]
sys.argv = [script, *sys.argv[2:]]
sys.path.insert(0, os.getcwd())
runpy.run_path(script, run_name='__main__')
```

The scripts warm each case; argument construction and initial compilation are
outside the timed loops except for the intentional cold compile-callback row
in `ffi_paths`. GPU cases time host calls/enqueues and synchronize afterward,
so queue backpressure can move the result without a kernel change. FFI no-ops
do not launch a GPU kernel; FFI and INTJ kernels in `bench_ffi_compare.py` use
different binaries. `bench_hip_module_launch.py` reuses one HSACO in FFI,
Triton, and INTJ: SHA-256
`df3e529e178196ad62f7c07ab8593ede5b8596c76e7c6fbd12a5d947a696e65c`
for all four runs, with one compile callback before timing. The cold callback
row in `ffi_paths` uses unique cache misses and has a different timing boundary
from a hot call or FFI's C++ callback. CUDA runtime was not tested on this ROCm
machine.

## Host-parser history

The earlier four-pair, host-only ABBA at candidate HEAD `34cbba2` showed a
small repeated bound-tensor increase. Two later four-pair probes of the raw
pointer parser removed that pattern. Each process warmed the case and timed
100,000 calls in each of nine batches on CPU 0; this table recomputes each
process median from its nine samples, then subtracts the paired develop median.
All 24 diagnostic processes exited 0. Their command is recorded in each source
log; it invokes `record_launch_batches.py benchmarks/bench_launch.py --no-gpu
--iters 100000 --batches 9` with the checkout's `PYTHONPATH` and `taskset -c 0`.
Within each set the order is D0, C0, C1, D1, D2, C2, C3, D3.

The baseline is `1bf689b` with committed `entry.c.jinja` SHA-256
`7faebcc700fe0b5bd1363acfc638b0621eebf7af754ab3dcf915ebd0c4e41806`.
The pre-fix logs record candidate HEAD `34cbba2` but no runtime source hash or
working-tree status. The **committed Git blob** at `34cbba2` hashes to
`45ce34f97efa5def1986a0d013b840d957de156bcd128d313832eabc42175aa7`;
that value is a reference, not a hash recorded by that run. In both final
probes, HEAD still reads `34cbba2`, but the checkout had uncommitted source.
The logs record candidate template SHA-256
`50f04d4c6bc58bd50a029525208892353ac1ba7184540ca7088d3ebaa9c2cb3f`,
which matches the later committed `c5295b1` template. A matching template hash
does not establish that every file in the uncommitted probe matched `c5295b1`.

| Study | Pair | Develop median (ns) | Candidate/probe median (ns) | Change (ns) |
| --- | ---: | ---: | ---: | ---: |
| Pre-fix host ABBA | 0 | 45.62017 | 46.40658 | +0.78641 |
| Pre-fix host ABBA | 1 | 45.85935 | 48.04529 | +2.18594 |
| Pre-fix host ABBA | 2 | 45.64747 | 47.97725 | +2.32978 |
| Pre-fix host ABBA | 3 | 45.47390 | 46.63286 | +1.15896 |
| Final raw-pointer probe 1 | 0 | 45.83377 | 46.05008 | +0.21631 |
| Final raw-pointer probe 1 | 1 | 46.40685 | 45.35396 | -1.05289 |
| Final raw-pointer probe 1 | 2 | 45.84986 | 45.86115 | +0.01129 |
| Final raw-pointer probe 1 | 3 | 45.88958 | 45.61065 | -0.27893 |
| Final raw-pointer probe 2 | 0 | 45.80667 | 45.52596 | -0.28071 |
| Final raw-pointer probe 2 | 1 | 45.86126 | 45.46926 | -0.39200 |
| Final raw-pointer probe 2 | 2 | 45.83269 | 45.44751 | -0.38518 |
| Final raw-pointer probe 2 | 3 | 45.78203 | 45.58590 | -0.19613 |

The following lines are copied from each source log, including every bound-tensor batch sample and the benchmark's rounded median.

### Pre-fix host ABBA

```text
[develop_r0.log]
recorded_batch_samples_ns=[45.62017, 45.84801, 45.43664, 45.85546, 45.33098, 46.52351, 45.34998, 46.1444, 45.36012]
       bound tensor       45.6       +2.0      +4.6       0.14      1.31
[candidate_r0.log]
recorded_batch_samples_ns=[46.11166, 48.43023, 46.2377, 47.32658, 46.40658, 48.78875, 45.57924, 46.66432, 46.18727]
       bound tensor       46.4       +2.2      +4.9       0.14      1.51
[candidate_r1.log]
recorded_batch_samples_ns=[48.04529, 48.4506, 46.32843, 48.40613, 46.74264, 48.49403, 46.20538, 48.69633, 46.90628]
       bound tensor       48.0       +4.0      +9.1       0.14      1.48
[develop_r1.log]
recorded_batch_samples_ns=[45.94141, 46.90017, 45.49488, 46.72418, 45.43039, 45.85935, 45.45768, 45.87073, 45.46587]
       bound tensor       45.9       +1.9      +4.2       0.15      1.32
[develop_r2.log]
recorded_batch_samples_ns=[45.48199, 46.16353, 45.41425, 46.5592, 45.64747, 46.4091, 45.47077, 46.02861, 45.33071]
       bound tensor       45.6       +1.7      +4.0       0.14      1.30
[candidate_r2.log]
recorded_batch_samples_ns=[47.97725, 50.44943, 47.73638, 50.1448, 47.91261, 49.04745, 46.55094, 48.43737, 46.72793]
       bound tensor       48.0       +4.0      +9.1       0.14      1.54
[candidate_r3.log]
recorded_batch_samples_ns=[46.54364, 47.31618, 46.41192, 47.15118, 46.63286, 47.32934, 46.30158, 47.78761, 46.52147]
       bound tensor       46.6       +1.7      +3.9       0.14      1.48
[develop_r3.log]
recorded_batch_samples_ns=[45.4739, 46.68779, 45.3989, 45.87023, 45.44639, 45.86618, 45.45281, 46.11315, 45.40947]
       bound tensor       45.5       +1.5      +3.5       0.14      1.29
```

### Final raw-pointer probe 1

```text
[develop_r0.log]
recorded_batch_samples_ns=[46.37694, 45.83377, 45.37102, 46.31455, 45.94698, 45.82416, 45.69516, 45.90832, 45.36962]
       bound tensor       45.8       +1.6      +3.6       0.14      1.30
[probe_r0.log]
recorded_batch_samples_ns=[46.24798, 46.82143, 45.78821, 46.98131, 46.77834, 46.05008, 45.0504, 45.7247, 45.18135]
       bound tensor       46.1       +2.5      +5.8       0.23   2032.51
[probe_r1.log]
recorded_batch_samples_ns=[45.35396, 46.37747, 45.15161, 46.475, 45.28944, 45.60567, 45.00552, 45.45907, 45.01304]
       bound tensor       45.4       +1.1      +2.6       0.15      1.50
[develop_r1.log]
recorded_batch_samples_ns=[46.02972, 49.02579, 45.46277, 46.40685, 45.88599, 47.04183, 45.55184, 47.11065, 47.13944]
       bound tensor       46.4       +2.6      +5.8       0.13      1.29
[develop_r2.log]
recorded_batch_samples_ns=[45.50237, 46.42179, 45.63932, 46.79506, 45.84986, 46.23122, 45.79978, 46.89885, 45.63508]
       bound tensor       45.8       +2.2      +5.0       0.13      1.27
[probe_r2.log]
recorded_batch_samples_ns=[46.00785, 45.86115, 44.99411, 46.03214, 45.30079, 46.06623, 45.09968, 45.9738, 45.11762]
       bound tensor       45.9       +2.2      +5.1       0.15      1.51
[probe_r3.log]
recorded_batch_samples_ns=[45.61065, 45.91631, 45.15666, 46.19721, 45.03164, 45.88854, 45.00501, 46.27021, 44.9414]
       bound tensor       45.6       +2.2      +5.2       0.14      1.48
[develop_r3.log]
recorded_batch_samples_ns=[45.41446, 46.09543, 45.39548, 45.88958, 45.44109, 46.1288, 46.2163, 46.32725, 45.53759]
       bound tensor       45.9       +2.0      +4.6       0.13      1.29
```

### Final raw-pointer probe 2

```text
[develop_r0.log]
recorded_batch_samples_ns=[46.23657, 45.86793, 45.38127, 46.01196, 45.39327, 45.80667, 45.32929, 45.84336, 45.32168]
       bound tensor       45.8       +1.9      +4.4       0.13      1.29
[probe_r0.log]
recorded_batch_samples_ns=[45.52596, 47.99474, 45.09016, 46.07645, 45.31558, 47.58915, 45.19991, 46.60835, 44.97394]
       bound tensor       45.5       +1.4      +3.1       0.14      1.48
[probe_r1.log]
recorded_batch_samples_ns=[45.1334, 45.98105, 45.46926, 46.20385, 45.11962, 45.48176, 45.11534, 46.4902, 45.11496]
       bound tensor       45.5       +1.9      +4.4       0.14      1.47
[develop_r1.log]
recorded_batch_samples_ns=[45.86126, 45.86615, 45.47649, 45.92198, 45.4569, 46.21763, 45.58276, 46.41262, 45.43134]
       bound tensor       45.9       +2.0      +4.7       0.13      1.30
[develop_r2.log]
recorded_batch_samples_ns=[45.46424, 45.82648, 45.41026, 45.83269, 45.49238, 45.93857, 47.28449, 46.21681, 45.9577]
       bound tensor       45.8       +2.1      +4.9       0.14      1.27
[probe_r2.log]
recorded_batch_samples_ns=[45.2984, 46.2545, 45.10345, 45.92866, 45.44751, 46.05133, 45.06923, 45.58183, 45.14085]
       bound tensor       45.4       +2.1      +4.7       0.15      1.49
[probe_r3.log]
recorded_batch_samples_ns=[45.59634, 45.5859, 45.10215, 46.25055, 45.0148, 45.89488, 45.11806, 45.90826, 45.14449]
       bound tensor       45.6       +2.0      +4.7       0.15      1.49
[develop_r3.log]
recorded_batch_samples_ns=[45.40484, 45.95943, 45.42283, 46.45281, 45.78203, 47.21588, 45.51405, 47.07861, 45.42096]
       bound tensor       45.8       +1.9      +4.3       0.15      1.33
```

## Raw outputs — repeat 0

### launch_host: develop r0

```text
commit=1bf689b37137e373b6baecf241a8d8a7da23630b
source_sha256=7faebcc700fe0b5bd1363acfc638b0621eebf7af754ab3dcf915ebd0c4e41806
started=2026-09-25T18:33:50.783852+00:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj taskset -c 0 /tmp/gb2/bin/python /tmp/intj_develop_baseline_2026-09-26_1bf689b/record_launch_batches.py benchmarks/bench_launch.py --no-gpu --iters 100000 --batches 9
mode=host-only; 100000 calls × 9 batches; median ns/call
               path    ns/call       Δ ns       Δ %   build ms   bind ms
recorded_batch_samples_ns=[43.65776, 44.05857, 43.59203, 43.98829, 43.15937, 43.92845, 43.53957, 43.89405, 43.32533]
           auto map       43.7       +0.0      +0.0      60.45         -
recorded_batch_samples_ns=[43.16618, 43.60921, 43.09566, 43.58073, 42.98323, 43.5873, 43.15805, 44.02627, 43.25333]
        reduced key       43.3       -0.4      -0.9       1.54         -
recorded_batch_samples_ns=[43.8506, 44.01132, 43.17646, 43.68983, 43.41157, 44.5786, 43.37139, 43.85665, 43.38093]
         verify off       43.7       +0.0      +0.1       1.41         -
recorded_batch_samples_ns=[44.24257, 44.84225, 44.03784, 44.96309, 43.94049, 44.96176, 43.9459, 45.13883, 43.95261]
          verify on       44.2       +0.6      +1.3       1.35         -
recorded_batch_samples_ns=[39.89221, 40.22286, 39.79018, 40.45543, 39.70592, 40.70383, 39.77451, 40.3589, 39.74123]
              baked       39.9       -3.8      -8.6       1.53         -
recorded_batch_samples_ns=[45.41561, 46.23678, 45.51564, 45.92324, 45.40041, 46.47991, 45.53832, 45.7826, 45.34305]
       bound tensor       45.5       +1.9      +4.3       0.13      1.28
recorded_batch_samples_ns=[44.21799, 44.03181, 44.6149, 43.88953, 44.22958, 44.04661, 43.92937, 43.85109, 44.71072]
      bound pointer       44.0       +0.4      +0.9       0.14      1.26
recorded_batch_samples_ns=[43.75151, 44.69384, 44.16743, 45.53636, 44.06186, 44.66228, 43.99476, 44.70419, 44.02622]
   fixed device map       44.2       +0.5      +1.2       0.10      1.23
recorded_batch_samples_ns=[44.37961, 45.35104, 43.04365, 45.56409, 43.49873, 44.97144, 42.61144, 44.7793, 88.25371]
fixed device no-map       44.8       +1.1      +2.6       0.13      1.24
exit_status=0
ended=2026-09-25T18:33:53.786650+00:00
```

### launch_host: candidate r0

```text
commit=c5295b18f29a15fc1184eb7e93dbc099e6b75bf2
source_sha256=50f04d4c6bc58bd50a029525208892353ac1ba7184540ca7088d3ebaa9c2cb3f
started=2026-09-25T18:33:53.789196+00:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/callable_grid
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/callable_grid taskset -c 0 /tmp/gb2/bin/python /tmp/intj_develop_baseline_2026-09-26_1bf689b/record_launch_batches.py benchmarks/bench_launch.py --no-gpu --iters 100000 --batches 9
mode=host-only; 100000 calls × 9 batches; median ns/call
               path    ns/call       Δ ns       Δ %   build ms   bind ms
recorded_batch_samples_ns=[43.18909, 44.13167, 43.30865, 43.5706, 43.04454, 43.58579, 43.31981, 43.53374, 43.26698]
           auto map       43.3       +0.0      +0.0      61.35         -
recorded_batch_samples_ns=[42.36543, 43.05849, 42.47594, 43.16794, 42.61904, 42.92796, 42.38295, 43.02203, 42.28391]
        reduced key       42.6       -0.7      -1.6       1.74         -
recorded_batch_samples_ns=[42.8154, 43.19235, 42.68084, 43.14618, 42.92812, 43.18504, 42.64222, 43.27386, 42.72067]
         verify off       42.9       -0.4      -0.9       1.57         -
recorded_batch_samples_ns=[44.27385, 44.59116, 44.12732, 44.6338, 43.98942, 44.45542, 44.07372, 44.46119, 43.95659]
          verify on       44.3       +1.0      +2.2       1.53         -
recorded_batch_samples_ns=[39.68504, 40.39292, 39.80188, 40.09692, 39.79065, 41.60217, 39.53025, 39.97393, 39.54728]
              baked       39.8       -3.5      -8.1       1.61         -
recorded_batch_samples_ns=[45.01016, 79.48044, 57.74067, 45.57604, 44.94413, 45.68059, 44.90634, 45.40707, 45.08346]
       bound tensor       45.4       +2.1      +4.8       0.14      1.49
recorded_batch_samples_ns=[43.91342, 43.61431, 44.04647, 43.88784, 44.39888, 43.73031, 43.77963, 44.29253, 43.84469]
      bound pointer       43.9       +0.6      +1.3       0.14      1.51
recorded_batch_samples_ns=[44.10205, 44.32313, 43.83031, 44.37766, 44.20035, 44.30323, 43.80952, 44.39963, 43.83377]
   fixed device map       44.2       +0.9      +2.0       0.11      1.42
recorded_batch_samples_ns=[43.69032, 45.41386, 42.49969, 44.14974, 42.69634, 43.89575, 42.63507, 47.82687, 43.11342]
fixed device no-map       43.7       +0.4      +0.9       0.15      1.39
exit_status=0
ended=2026-09-25T18:33:56.684280+00:00
```

### launch_sweep: develop r0

```text
commit=1bf689b37137e373b6baecf241a8d8a7da23630b
source_sha256=7faebcc700fe0b5bd1363acfc638b0621eebf7af754ab3dcf915ebd0c4e41806
started=2026-09-25T18:34:02.785696+00:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj taskset -c 0 /tmp/gb2/bin/python /tmp/intj_develop_baseline_2026-09-26_1bf689b/record_launch_batches.py benchmarks/bench_launch.py --sweep --iters 100000 --batches 9
mode=sweep; host-only; 100000 calls × 9 batches; median ns/call
count   kind    ns/call
recorded_batch_samples_ns=[41.75119, 42.07535, 41.4931, 41.70278, 41.20147, 41.79097, 41.22436, 41.85713, 41.32846]
    4    int       41.7
recorded_batch_samples_ns=[40.63646, 40.94811, 40.51909, 41.05715, 40.46455, 40.96304, 40.78226, 40.81931, 40.39656]
    4 tensor       40.8
recorded_batch_samples_ns=[74.27973, 74.19487, 74.19727, 74.06704, 74.19014, 74.26021, 74.16743, 74.24257, 74.22131]
   16    int       74.2
recorded_batch_samples_ns=[74.12902, 68.84279, 68.97301, 68.44122, 71.85139, 68.404, 68.28841, 69.07437, 111.43996]
   16 tensor       69.0
recorded_batch_samples_ns=[265.05271, 218.88316, 118.06543, 120.87332, 121.0503, 119.21957, 117.69177, 118.93645, 122.4576]
   32    int      120.9
recorded_batch_samples_ns=[127.41626, 127.5239, 124.24612, 117.88722, 120.68191, 118.93273, 117.91425, 117.95585, 117.92506]
   32 tensor      118.9
exit_status=0
ended=2026-09-25T18:34:05.834855+00:00
```

### launch_sweep: candidate r0

```text
commit=c5295b18f29a15fc1184eb7e93dbc099e6b75bf2
source_sha256=50f04d4c6bc58bd50a029525208892353ac1ba7184540ca7088d3ebaa9c2cb3f
started=2026-09-25T18:34:05.837404+00:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/callable_grid
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/callable_grid taskset -c 0 /tmp/gb2/bin/python /tmp/intj_develop_baseline_2026-09-26_1bf689b/record_launch_batches.py benchmarks/bench_launch.py --sweep --iters 100000 --batches 9
mode=sweep; host-only; 100000 calls × 9 batches; median ns/call
count   kind    ns/call
recorded_batch_samples_ns=[41.34635, 41.74462, 41.50421, 41.57747, 41.13823, 41.63521, 41.37472, 41.87176, 41.20291]
    4    int       41.5
recorded_batch_samples_ns=[40.19258, 40.68499, 40.2903, 40.69913, 40.31393, 41.18053, 40.18914, 40.88696, 40.35256]
    4 tensor       40.4
recorded_batch_samples_ns=[71.30029, 70.43107, 70.43068, 70.0784, 70.21507, 72.69659, 71.09717, 71.01131, 71.98217]
   16    int       71.0
recorded_batch_samples_ns=[68.26952, 67.4781, 67.20561, 67.25914, 67.47805, 68.02064, 67.91238, 68.28211, 71.06239]
   16 tensor       67.9
recorded_batch_samples_ns=[111.04734, 111.54199, 111.48454, 111.13558, 110.13756, 109.57436, 109.55511, 112.59061, 110.17157]
   32    int      111.0
recorded_batch_samples_ns=[123.58767, 119.55852, 116.59123, 115.63814, 112.26228, 112.98228, 113.92696, 115.32033, 121.54513]
   32 tensor      115.6
exit_status=0
ended=2026-09-25T18:34:08.871708+00:00
```

### ffi_paths: develop r0

```text
commit=1bf689b37137e373b6baecf241a8d8a7da23630b
source_sha256=7faebcc700fe0b5bd1363acfc638b0621eebf7af754ab3dcf915ebd0c4e41806
started=2026-09-25T18:34:15.060436+00:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_intj_ffi_paths.py --mode=all --iters 1000 --batches 9
iters=1000 batches=9 unit=ns/call; no_gpu=True
mode=callback
callback_count=9100 (100 warmup + 9000 unique misses)
INTJ cold compile callback + cache  median_ns=420.3 samples_ns=[455.28, 456.351, 397.43, 577.227, 385.927, 405.821, 412.636, 420.325, 1492.971]
INTJ hot call (no callback)         median_ns=41.3 samples_ns=[42.341, 42.803, 40.302, 41.438, 40.273, 41.555, 40.431, 41.299, 40.109]
INTJ 3-tensor host-only nop         median_ns=35.3 samples_ns=[37.419, 35.761, 35.139, 35.591, 35.025, 35.589, 34.927, 35.326, 35.086]
mode=kwargs
INTJ direct positional              median_ns=66.7 samples_ns=[69.121, 67.047, 66.625, 66.721, 66.691, 73.086, 66.07, 66.142, 66.187]
INTJ adapter positional             median_ns=90.8 samples_ns=[91.551, 91.89, 90.438, 90.268, 90.455, 90.842, 90.648, 110.759, 93.648]
INTJ adapter kwargs                 median_ns=108.7 samples_ns=[108.131, 109.019, 108.653, 108.585, 109.002, 108.978, 107.136, 115.527, 107.158]
INTJ adapter defaults               median_ns=89.1 samples_ns=[89.958, 89.498, 88.949, 89.021, 88.868, 89.123, 89.157, 88.877, 90.168]
INTJ FFI wrapper positional         median_ns=103.5 samples_ns=[107.704, 103.61, 103.422, 103.788, 103.318, 103.633, 102.463, 102.592, 103.505]
INTJ FFI wrapper kwargs             median_ns=126.2 samples_ns=[134.051, 128.114, 125.165, 125.023, 126.161, 126.017, 126.517, 125.48, 131.489]
INTJ native kwargs: unsupported (verified TypeError)
mode=dataclass
INTJ pair direct                    median_ns=67.6 samples_ns=[74.78, 68.97, 66.814, 67.64, 66.301, 68.721, 66.461, 68.369, 67.074]
INTJ pair prebuilt *tuple           median_ns=141.9 samples_ns=[140.436, 143.108, 145.867, 145.25, 141.073, 141.916, 140.203, 142.086, 139.569]
FFI unpack Pair only                median_ns=165.3 samples_ns=[169.289, 166.294, 164.528, 165.658, 165.051, 164.509, 169.426, 165.341, 164.351]
INTJ pair manual unpack             median_ns=173.6 samples_ns=[173.6, 174.43, 172.37, 178.247, 173.356, 173.982, 171.451, 173.571, 172.314]
INTJ pair FFI unpack                median_ns=316.6 samples_ns=[314.463, 317.132, 314.444, 327.124, 316.675, 315.304, 316.636, 317.904, 313.659]
INTJ pair stdlib astuple            median_ns=1157.6 samples_ns=[1322.36, 1207.657, 1147.726, 1156.838, 1170.424, 1157.63, 1158.173, 1153.185, 1146.109]
INTJ config direct                  median_ns=82.9 samples_ns=[84.076, 83.939, 83.118, 84.192, 82.924, 81.93, 81.231, 81.656, 79.872]
INTJ config FFI unpack              median_ns=321.1 samples_ns=[321.085, 327.231, 317.854, 318.92, 321.585, 321.865, 318.575, 321.068, 319.826]
INTJ native dataclass unpack: unsupported (verified TypeError)
exit_status=0
ended=2026-09-25T18:34:17.946878+00:00
```

### ffi_paths: candidate r0

```text
commit=c5295b18f29a15fc1184eb7e93dbc099e6b75bf2
source_sha256=50f04d4c6bc58bd50a029525208892353ac1ba7184540ca7088d3ebaa9c2cb3f
started=2026-09-25T18:34:17.949372+00:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/callable_grid
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/callable_grid taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_intj_ffi_paths.py --mode=all --iters 1000 --batches 9
iters=1000 batches=9 unit=ns/call; no_gpu=True
mode=callback
callback_count=9100 (100 warmup + 9000 unique misses)
INTJ cold compile callback + cache  median_ns=463.9 samples_ns=[562.807, 552.293, 452.97, 568.549, 424.149, 430.605, 445.475, 463.869, 804.875]
INTJ hot call (no callback)         median_ns=38.7 samples_ns=[41.453, 39.094, 38.115, 38.026, 38.185, 38.569, 38.881, 38.711, 39.208]
INTJ 3-tensor host-only nop         median_ns=35.4 samples_ns=[43.76, 35.658, 35.434, 35.379, 35.437, 35.271, 35.102, 42.622, 34.909]
mode=kwargs
INTJ direct positional              median_ns=65.2 samples_ns=[67.906, 72.044, 65.134, 65.152, 65.146, 65.365, 65.163, 65.265, 65.229]
INTJ adapter positional             median_ns=89.9 samples_ns=[90.361, 88.498, 90.191, 88.867, 93.575, 94.258, 89.91, 88.552, 88.656]
INTJ adapter kwargs                 median_ns=106.2 samples_ns=[106.184, 106.3, 106.207, 104.867, 105.865, 106.053, 110.865, 109.516, 109.609]
INTJ adapter defaults               median_ns=87.8 samples_ns=[87.483, 91.464, 87.733, 87.704, 87.748, 87.833, 87.871, 95.291, 88.137]
INTJ FFI wrapper positional         median_ns=103.0 samples_ns=[102.971, 103.621, 103.069, 102.634, 102.63, 102.5, 102.695, 103.17, 107.455]
INTJ FFI wrapper kwargs             median_ns=125.0 samples_ns=[126.166, 126.025, 125.626, 125.02, 124.61, 124.928, 124.953, 128.424, 123.576]
INTJ native kwargs: unsupported (verified TypeError)
mode=dataclass
INTJ pair direct                    median_ns=67.9 samples_ns=[76.318, 68.22, 67.909, 73.11, 67.752, 68.005, 67.375, 67.557, 67.625]
INTJ pair prebuilt *tuple           median_ns=136.5 samples_ns=[136.32, 137.625, 137.242, 136.485, 140.522, 136.199, 136.7, 134.527, 135.454]
FFI unpack Pair only                median_ns=162.7 samples_ns=[162.744, 180.543, 161.896, 161.032, 162.546, 162.414, 163.252, 166.273, 163.551]
INTJ pair manual unpack             median_ns=168.6 samples_ns=[168.148, 167.816, 166.946, 167.503, 170.638, 168.595, 169.843, 170.983, 170.095]
INTJ pair FFI unpack                median_ns=312.3 samples_ns=[317.409, 312.293, 314.415, 317.189, 310.678, 309.347, 309.535, 312.85, 309.527]
INTJ pair stdlib astuple            median_ns=1176.4 samples_ns=[1350.864, 1173.268, 1163.427, 1169.704, 1182.415, 1182.219, 1176.415, 1175.147, 1202.075]
INTJ config direct                  median_ns=80.8 samples_ns=[81.163, 80.83, 81.022, 80.667, 82.117, 115.284, 79.945, 80.058, 80.225]
INTJ config FFI unpack              median_ns=323.0 samples_ns=[322.468, 324.247, 325.608, 321.986, 321.313, 326.52, 322.616, 322.955, 326.753]
INTJ native dataclass unpack: unsupported (verified TypeError)
exit_status=0
ended=2026-09-25T18:34:26.998335+00:00
```

### launch_gpu: develop r0

```text
commit=1bf689b37137e373b6baecf241a8d8a7da23630b
source_sha256=7faebcc700fe0b5bd1363acfc638b0621eebf7af754ab3dcf915ebd0c4e41806
started=2026-09-25T18:34:32.954149+00:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj taskset -c 0 /tmp/gb2/bin/python /tmp/intj_develop_baseline_2026-09-26_1bf689b/record_launch_batches.py benchmarks/bench_launch.py --iters 1000 --batches 9
mode=gpu; 1000 calls × 9 batches; median ns/call
               path    ns/call       Δ ns       Δ %   build ms   bind ms
recorded_batch_samples_ns=[3229.904, 4009.071, 3509.038, 3657.859, 3673.838, 3754.564, 3623.824, 3719.226, 3684.889]
           auto map     3673.8       +0.0      +0.0     109.54         -
recorded_batch_samples_ns=[3140.222, 4036.792, 3684.174, 3642.601, 3538.328, 3626.076, 3659.06, 3558.155, 3553.469]
        reduced key     3626.1      -47.8      -1.3      73.46         -
recorded_batch_samples_ns=[3033.944, 3700.152, 3609.001, 3633.379, 3625.184, 3468.034, 3576.552, 3535.702, 3562.761]
         verify off     3576.6      -97.3      -2.6       1.97         -
recorded_batch_samples_ns=[2972.663, 3581.329, 3612.637, 3606.586, 3620.937, 3584.374, 3585.874, 3463.31, 3582.296]
          verify on     3584.4      -89.5      -2.4       1.93         -
recorded_batch_samples_ns=[2871.626, 3506.896, 3527.569, 3408.444, 3525.821, 3509.828, 3522.95, 3396.996, 3399.881]
              baked     3506.9     -166.9      -4.5       2.01         -
recorded_batch_samples_ns=[2920.826, 3488.833, 3544.178, 3503.235, 3438.406, 3435.045, 3509.839, 3514.421, 3508.706]
       bound tensor     3503.2     -170.6      -4.6       0.42      1.60
recorded_batch_samples_ns=[2932.378, 3503.487, 3521.566, 3515.033, 3520.379, 3487.07, 3418.624, 3484.244, 3513.97]
      bound pointer     3503.5     -170.4      -4.6       0.40      1.62
recorded_batch_samples_ns=[10802.784, 17323.401, 7948.044, 3552.535, 3492.791, 3495.336, 3465.488, 3535.2, 3507.252]
   fixed device map     3535.2     -138.6      -3.8       0.50      1.63
recorded_batch_samples_ns=[3034.31, 3818.93, 3795.521, 3794.404, 3781.125, 3820.117, 3793.897, 3825.548, 3677.602]
fixed device no-map     3794.4     +120.6      +3.3       0.42      1.59
exit_status=0
ended=2026-09-25T18:34:36.899043+00:00
```

### launch_gpu: candidate r0

```text
commit=c5295b18f29a15fc1184eb7e93dbc099e6b75bf2
source_sha256=50f04d4c6bc58bd50a029525208892353ac1ba7184540ca7088d3ebaa9c2cb3f
started=2026-09-25T18:34:36.901540+00:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/callable_grid
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/callable_grid taskset -c 0 /tmp/gb2/bin/python /tmp/intj_develop_baseline_2026-09-26_1bf689b/record_launch_batches.py benchmarks/bench_launch.py --iters 1000 --batches 9
mode=gpu; 1000 calls × 9 batches; median ns/call
               path    ns/call       Δ ns       Δ %   build ms   bind ms
recorded_batch_samples_ns=[3159.714, 3363.081, 3113.846, 3121.617, 3114.272, 3153.635, 3170.735, 3150.169, 3125.23]
           auto map     3150.2       +0.0      +0.0      72.88         -
recorded_batch_samples_ns=[3328.704, 3421.173, 3205.23, 3193.541, 3200.121, 3250.808, 3191.47, 3190.968, 3186.836]
        reduced key     3200.1      +50.0      +1.6    2265.94         -
recorded_batch_samples_ns=[3309.778, 3427.627, 3163.505, 3184.874, 3220.838, 3173.631, 3182.547, 3410.685, 3171.212]
         verify off     3184.9      +34.7      +1.1    2171.30         -
recorded_batch_samples_ns=[3350.296, 3473.523, 3207.297, 3216.062, 3208.058, 3209.076, 3222.962, 3188.651, 3186.526]
          verify on     3209.1      +58.9      +1.9    2188.95         -
recorded_batch_samples_ns=[3411.921, 3293.827, 3154.579, 3178.084, 3176.136, 3167.599, 3162.953, 3153.87, 3157.167]
              baked     3167.6      +17.4      +0.6    2072.93         -
recorded_batch_samples_ns=[3310.251, 3483.16, 3246.842, 3241.382, 3252.854, 3243.556, 3219.853, 3213.358, 3219.374]
       bound tensor     3243.6      +93.4      +3.0       0.45   2054.93
recorded_batch_samples_ns=[3456.093, 3334.786, 3233.43, 3253.782, 3254.706, 3228.518, 3234.501, 3229.507, 3239.162]
      bound pointer     3239.2      +89.0      +2.8       0.44   2055.19
recorded_batch_samples_ns=[3320.639, 3466.284, 3236.015, 3267.066, 3226.399, 3216.425, 3215.05, 3218.75, 3212.133]
   fixed device map     3226.4      +76.2      +2.4       0.42   2087.65
recorded_batch_samples_ns=[3316.952, 3461.842, 3203.341, 3213.997, 3243.254, 3232.658, 3209.558, 3197.919, 3199.94]
fixed device no-map     3214.0      +63.8      +2.0       0.44   2099.14
exit_status=0
ended=2026-09-25T18:34:57.616712+00:00
```

### launch_readme: develop r0

```text
commit=1bf689b37137e373b6baecf241a8d8a7da23630b
source_sha256=7faebcc700fe0b5bd1363acfc638b0621eebf7af754ab3dcf915ebd0c4e41806
started=2026-09-25T18:35:05.279282+00:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj taskset -c 0 /tmp/gb2/bin/python /tmp/intj_develop_baseline_2026-09-26_1bf689b/record_launch_batches.py benchmarks/bench_launch.py --readme --iters 1000 --batches 9
recorded_batch_samples_ns=[91.24, 93.228, 91.072, 91.907, 91.635, 88.607, 89.791, 89.128, 88.613]
recorded_batch_samples_ns=[90.281, 95.773, 90.545, 90.726, 89.686, 89.713, 88.09, 96.17, 89.387]
recorded_batch_samples_ns=[910.637, 876.039, 868.835, 873.888, 862.016, 879.595, 875.587, 873.359, 859.281]
mode=readme; 1000 calls × 9 batches; median
        path   triton us    intj us   speedup
recorded_batch_samples_ns=[16590.778, 16969.229, 16950.38, 16777.303, 16680.515, 16821.807, 16715.95, 16741.542, 16678.902]
recorded_batch_samples_ns=[3352.92, 8115.324, 4942.043, 3723.685, 3605.463, 3602.951, 3704.463, 3546.446, 3557.697]
   grid=(1,)       16.74       3.61      4.6x
recorded_batch_samples_ns=[12958.34, 12901.866, 12851.615, 12841.213, 12861.568, 12973.311, 12920.889, 12863.463, 12839.065]
recorded_batch_samples_ns=[29.158, 28.57, 28.389, 28.622, 28.522, 28.48, 28.782, 28.523, 28.366]
   grid=(0,)       12.86       0.03    451.0x

torch_access_mode   decode ns    build s
     runtime_shim        91.1       0.02
   static_compile        90.3       0.06
      interpreter       873.9       0.00
exit_status=0
ended=2026-09-25T18:35:09.112683+00:00
```

### launch_readme: candidate r0

```text
commit=c5295b18f29a15fc1184eb7e93dbc099e6b75bf2
source_sha256=50f04d4c6bc58bd50a029525208892353ac1ba7184540ca7088d3ebaa9c2cb3f
started=2026-09-25T18:35:09.115239+00:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/callable_grid
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/callable_grid taskset -c 0 /tmp/gb2/bin/python /tmp/intj_develop_baseline_2026-09-26_1bf689b/record_launch_batches.py benchmarks/bench_launch.py --readme --iters 1000 --batches 9
recorded_batch_samples_ns=[96.231, 94.514, 98.493, 92.226, 91.665, 89.499, 89.721, 89.019, 89.42]
recorded_batch_samples_ns=[88.96, 90.161, 92.292, 92.045, 99.763, 91.531, 90.245, 91.361, 90.421]
recorded_batch_samples_ns=[885.671, 895.09, 861.167, 881.204, 861.013, 873.286, 863.508, 869.433, 880.027]
mode=readme; 1000 calls × 9 batches; median
        path   triton us    intj us   speedup
recorded_batch_samples_ns=[16562.782, 16609.709, 16306.007, 16510.187, 16348.591, 16494.831, 16501.691, 16529.192, 16312.95]
recorded_batch_samples_ns=[3024.537, 3241.362, 3120.402, 3082.093, 3131.345, 3148.389, 3088.948, 2985.035, 2939.353]
   grid=(1,)       16.50       3.09      5.3x
recorded_batch_samples_ns=[13012.892, 13036.941, 13018.577, 13015.68, 13000.862, 12970.085, 12956.715, 12984.685, 13016.885]
recorded_batch_samples_ns=[28.28, 27.932, 27.895, 27.95, 27.829, 27.856, 27.886, 27.852, 27.87]
   grid=(0,)       13.01       0.03    466.6x

torch_access_mode   decode ns    build s
     runtime_shim        91.7       0.02
   static_compile        91.4       0.06
      interpreter       873.3       0.00
exit_status=0
ended=2026-09-25T18:35:13.068649+00:00
```

### ffi_default: develop r0

```text
commit=1bf689b37137e373b6baecf241a8d8a7da23630b
source_sha256=7faebcc700fe0b5bd1363acfc638b0621eebf7af754ab3dcf915ebd0c4e41806
started=2026-09-25T18:35:20.803298+00:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_ffi_compare.py --iters 1000 --batches 9
iters=1000 batches=9 unit=us/call (host-only, no timed sync)
FFI packed nop             median=0.1463 samples=[0.150119, 0.143649, 0.14890299999999998, 0.14627, 0.145781, 0.148461, 0.148934, 0.14477199999999998, 0.143714]
FFI typed nop              median=0.1494 samples=[0.149351, 0.15226800000000001, 0.150126, 0.150054, 0.149324, 0.150944, 0.14651, 0.149321, 0.14760800000000002]
FFI empty kernel           median=3.3470 samples=[3.4150120000000004, 3.458069, 3.490607, 3.346977, 3.367494, 3.201415, 3.203562, 3.2165340000000002, 3.206119]
INTJ empty kernel          median=3.0012 samples=[3.062379, 3.103176, 4.360092, 2.967728, 2.965591, 2.879435, 3.019007, 2.861219, 3.001186]
INTJ fixed-device kernel   median=2.9685 samples=[3.124175, 3.0911880000000003, 12.313540000000001, 2.968516, 2.984915, 2.8410680000000004, 2.801895, 2.8097109999999996, 2.931429]
FFI packed nop mixed       median=0.1749 samples=[0.171612, 0.174411, 0.24150899999999997, 0.173794, 0.176896, 0.17247800000000002, 0.177219, 0.178974, 0.174914]
FFI typed nop mixed        median=0.1797 samples=[0.175244, 0.180228, 0.238183, 0.18071, 0.177093, 0.17973, 0.180656, 0.17865999999999999, 0.177155]
FFI mixed kernel           median=3.3974 samples=[3.415404, 3.539605, 8.227506, 3.40731, 3.3973850000000003, 3.354948, 3.236258, 3.245366, 3.231283]
INTJ mixed kernel          median=2.9467 samples=[3.127026, 3.10593, 2.946723, 2.9759499999999997, 2.987454, 2.8804499999999997, 2.834485, 2.86245, 2.819381]
INTJ fixed mixed kernel    median=2.9928 samples=[3.123004, 3.125171, 2.993576, 3.001188, 2.898339, 2.873536, 2.821445, 2.8261350000000003, 2.992801]
exit_status=0
ended=2026-09-25T18:35:24.623903+00:00
```

### ffi_default: candidate r0

```text
commit=c5295b18f29a15fc1184eb7e93dbc099e6b75bf2
source_sha256=50f04d4c6bc58bd50a029525208892353ac1ba7184540ca7088d3ebaa9c2cb3f
started=2026-09-25T18:35:24.626420+00:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/callable_grid
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/callable_grid taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_ffi_compare.py --iters 1000 --batches 9
iters=1000 batches=9 unit=us/call (host-only, no timed sync)
FFI packed nop             median=0.1446 samples=[0.148465, 0.144647, 0.14545, 0.146017, 0.143643, 0.146863, 0.144374, 0.14282, 0.14443199999999998]
FFI typed nop              median=0.1493 samples=[0.147749, 0.151066, 0.150125, 0.157328, 0.14926, 0.148899, 0.14771, 0.154408, 0.148894]
FFI empty kernel           median=3.3012 samples=[3.7625279999999997, 3.449835, 3.46583, 3.248005, 3.236029, 3.2943890000000002, 3.303025, 3.297253, 3.301211]
INTJ empty kernel          median=2.9035 samples=[3.057998, 3.068393, 3.062036, 2.863765, 2.83648, 2.9034720000000003, 2.894931, 2.908781, 2.898227]
INTJ fixed-device kernel   median=2.8980 samples=[3.016314, 3.074021, 3.053736, 2.904306, 2.8451, 2.880136, 2.877516, 2.886965, 2.897964]
FFI packed nop mixed       median=0.1741 samples=[0.174131, 0.17849299999999999, 0.172794, 0.173429, 0.17735, 0.17269900000000002, 0.178386, 0.172402, 0.174761]
FFI typed nop mixed        median=0.1762 samples=[0.17528200000000002, 0.17782, 0.176216, 0.178614, 0.175028, 0.177157, 0.17599199999999998, 0.178356, 0.17530099999999998]
FFI mixed kernel           median=3.3306 samples=[3.434542, 3.488027, 3.3305599999999997, 3.337697, 3.246614, 3.316746, 3.327131, 3.317484, 3.330789]
INTJ mixed kernel          median=2.8918 samples=[3.0764549999999997, 3.074962, 2.9009650000000002, 2.8851590000000003, 2.8619920000000003, 2.839887, 2.891797, 2.914704, 2.889498]
INTJ fixed mixed kernel    median=2.8744 samples=[3.111862, 3.093797, 2.937258, 2.938539, 2.852904, 2.842495, 2.845322, 2.87443, 2.830977]
exit_status=0
ended=2026-09-25T18:35:36.514767+00:00
```

### ffi_sweep: develop r0

```text
commit=1bf689b37137e373b6baecf241a8d8a7da23630b
source_sha256=7faebcc700fe0b5bd1363acfc638b0621eebf7af754ab3dcf915ebd0c4e41806
started=2026-09-25T18:35:44.311806+00:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_ffi_compare.py --sweep --iters 1000 --batches 9
counts=[0, 3, 5, 8, 16, 32, 64] iters=1000 batches=9 unit=us/call (host-only, no timed sync); INTJ device-bound, no specialization key
args= 0 FFI packed nop           median=0.1179 samples=[0.120536, 0.11734, 0.117271, 0.11794400000000001, 0.11581699999999999, 0.11878, 0.117587, 0.118008, 0.118779]
args= 0 FFI typed nop            median=0.1199 samples=[0.120892, 0.121876, 0.118818, 0.118643, 0.12199, 0.118946, 0.119505, 0.12407800000000001, 0.11991500000000001]
args= 0 FFI empty kernel         median=1.6166 samples=[1.608454, 1.84462, 1.603195, 1.93544, 1.616633, 1.8328030000000002, 1.6019459999999999, 1.828509, 1.607863]
args= 0 INTJ static_compile kernel median=2.7331 samples=[3.1797150000000003, 2.646743, 2.703798, 2.709711, 2.701585, 2.741355, 2.741948, 2.73468, 2.733076]
args= 0 INTJ runtime_shim kernel median=2.7108 samples=[2.858712, 2.676649, 2.71081, 2.657125, 2.709753, 2.729586, 2.743672, 2.67542, 2.733325]
args= 3 FFI packed nop           median=0.1474 samples=[0.147376, 0.145925, 0.147394, 0.14921199999999998, 0.144154, 0.14574, 0.148201, 0.145219, 0.165663]
args= 3 FFI typed nop            median=0.1506 samples=[0.150452, 0.15059899999999998, 0.14846, 0.14864, 0.151285, 0.15059899999999998, 0.147089, 0.15058000000000002, 0.216421]
args= 3 FFI empty kernel         median=3.2498 samples=[3.302925, 3.302187, 3.157752, 3.2921060000000004, 3.160274, 3.2497890000000003, 3.2040349999999997, 3.292687, 3.214866]
args= 3 INTJ static_compile kernel median=2.8909 samples=[2.972133, 2.8659149999999998, 2.9688220000000003, 2.8447220000000004, 2.986826, 2.9079029999999997, 2.8909070000000003, 2.8505390000000004, 2.8667420000000003]
args= 3 INTJ runtime_shim kernel median=2.8972 samples=[2.997712, 2.8034, 2.905679, 2.82776, 2.897183, 2.940165, 2.9074989999999996, 2.837834, 2.878775]
args= 5 FFI packed nop           median=0.1739 samples=[0.174557, 0.176093, 0.174726, 0.175788, 0.1739, 0.17325800000000002, 0.173591, 0.171179, 0.173571]
args= 5 FFI typed nop            median=0.1776 samples=[0.17440799999999998, 0.17761600000000002, 0.175679, 0.193743, 0.174671, 0.17862899999999998, 0.17880000000000001, 0.179655, 0.17647900000000002]
args= 5 FFI empty kernel         median=3.2643 samples=[3.35668, 3.252922, 3.244659, 3.274645, 3.271737, 3.28229, 3.2386329999999997, 3.256767, 3.264321]
args= 5 INTJ static_compile kernel median=2.8817 samples=[2.992699, 2.8765500000000004, 4.038975, 2.858431, 2.8778490000000003, 2.881737, 2.8968049999999996, 2.8575100000000004, 3.089022]
args= 5 INTJ runtime_shim kernel median=2.8796 samples=[3.013957, 2.9199859999999997, 3.0770790000000003, 2.8693, 2.846551, 2.882992, 2.8624140000000002, 2.879648, 2.834985]
args= 8 FFI packed nop           median=0.2064 samples=[0.203607, 0.206361, 0.205126, 0.20488499999999998, 0.209623, 0.206348, 0.20911600000000002, 0.208091, 0.208405]
args= 8 FFI typed nop            median=0.2092 samples=[0.210339, 0.211121, 0.209216, 0.207944, 0.205786, 0.21058600000000002, 0.208895, 0.206405, 0.212345]
args= 8 FFI empty kernel         median=3.4034 samples=[3.5141660000000003, 3.40532, 3.306336, 3.42828, 3.5351790000000003, 3.403433, 3.312551, 3.393295, 3.299763]
args= 8 INTJ static_compile kernel median=2.9515 samples=[3.052963, 2.910317, 3.057339, 2.927727, 3.041131, 2.951513, 2.915241, 2.9490410000000002, 3.0747600000000004]
args= 8 INTJ runtime_shim kernel median=2.9352 samples=[3.081183, 2.888156, 2.965462, 2.8944769999999997, 2.9459560000000002, 2.890294, 2.935172, 2.890496, 2.981935]
args=16 FFI packed nop           median=0.2940 samples=[0.293952, 0.297579, 0.298414, 0.29302, 0.292107, 0.294262, 0.290111, 0.295204, 0.29242700000000005]
args=16 FFI typed nop            median=0.3055 samples=[0.30224900000000005, 0.31228500000000003, 0.30600900000000003, 0.30851799999999996, 0.303062, 0.305496, 0.303821, 0.305525, 0.31035700000000005]
args=16 FFI empty kernel         median=3.5406 samples=[3.721596, 3.6677310000000003, 3.5257959999999997, 3.478007, 3.522666, 3.662238, 3.540578, 3.641137, 3.530625]
args=16 INTJ static_compile kernel median=3.3331 samples=[3.287846, 3.333139, 3.483356, 3.268036, 3.473487, 3.293777, 3.5190639999999997, 3.310655, 3.457244]
args=16 INTJ runtime_shim kernel median=3.2681 samples=[3.2681210000000003, 3.109255, 3.316426, 3.047387, 3.345011, 3.039185, 3.363152, 3.042501, 3.375143]
args=32 FFI packed nop           median=0.4762 samples=[0.476097, 0.480993, 0.48155200000000004, 0.47631799999999996, 0.48281, 0.476021, 0.474414, 0.471677, 0.476246]
args=32 FFI typed nop            median=0.4904 samples=[0.509843, 0.497301, 0.48709800000000003, 0.491813, 0.490416, 0.492201, 0.489085, 0.49006299999999997, 0.489313]
args=32 FFI empty kernel         median=4.2198 samples=[4.296633, 4.227079, 4.122508, 4.257857, 4.095269, 4.256557, 4.100487, 4.219813, 4.120013]
args=32 INTJ static_compile kernel median=3.5850 samples=[3.416537, 3.585032, 3.6413800000000003, 3.535714, 3.637642, 3.553154, 3.638847, 3.5731819999999996, 3.686665]
args=32 INTJ runtime_shim kernel median=3.4464 samples=[3.437506, 3.417568, 3.623312, 3.437907, 3.641147, 3.446379, 3.595683, 3.442476, 3.643944]
args=64 FFI packed nop           median=0.8455 samples=[0.851444, 0.889554, 0.8454740000000001, 0.813498, 0.8417279999999999, 0.8414579999999999, 0.836139, 0.856874, 0.8463930000000001]
args=64 FFI typed nop            median=0.8602 samples=[0.865885, 0.8860629999999999, 0.85873, 0.859649, 0.860245, 0.8590270000000001, 0.856731, 0.862013, 0.872374]
args=64 FFI empty kernel         median=5.0583 samples=[5.080418, 5.0886819999999995, 5.058345999999999, 5.048377, 5.073587000000001, 5.039313, 5.028029, 5.045412000000001, 5.067319]
args=64 INTJ static_compile kernel median=4.6133 samples=[4.6133370000000005, 4.630154, 4.607178, 4.646211, 4.611655, 4.635288999999999, 4.610162, 4.602262, 4.656035]
args=64 INTJ runtime_shim kernel median=4.5778 samples=[4.495622999999999, 4.550434, 4.57782, 4.60928, 4.488538999999999, 4.631734000000001, 4.514549, 4.614138, 4.588139]
exit_status=0
ended=2026-09-25T18:35:48.715656+00:00
```

### ffi_sweep: candidate r0

```text
commit=c5295b18f29a15fc1184eb7e93dbc099e6b75bf2
source_sha256=50f04d4c6bc58bd50a029525208892353ac1ba7184540ca7088d3ebaa9c2cb3f
started=2026-09-25T18:35:48.718121+00:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/callable_grid
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/callable_grid taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_ffi_compare.py --sweep --iters 1000 --batches 9
counts=[0, 3, 5, 8, 16, 32, 64] iters=1000 batches=9 unit=us/call (host-only, no timed sync); INTJ device-bound, no specialization key
args= 0 FFI packed nop           median=0.1164 samples=[0.125661, 0.114603, 0.11507899999999999, 0.11653, 0.11511400000000001, 0.117444, 0.11576399999999999, 0.116542, 0.116424]
args= 0 FFI typed nop            median=0.1197 samples=[0.119698, 0.123943, 0.116741, 0.122574, 0.117878, 0.119691, 0.116577, 0.122991, 0.118228]
args= 0 FFI empty kernel         median=2.3025 samples=[1.9118330000000001, 2.9793429999999996, 2.153437, 2.8605189999999996, 2.1449059999999998, 3.051703, 2.3024989999999996, 2.977514, 2.121279]
args= 0 INTJ static_compile kernel median=3.3397 samples=[3.399037, 3.3497269999999997, 3.361692, 3.356757, 3.339712, 3.217031, 3.298705, 3.241498, 3.170102]
args= 0 INTJ runtime_shim kernel median=3.1745 samples=[3.174507, 3.1059650000000003, 3.400169, 3.0968899999999997, 3.36169, 3.071429, 3.220784, 3.092221, 3.2450520000000003]
args= 3 FFI packed nop           median=0.1451 samples=[0.14509200000000003, 0.146937, 0.14513800000000002, 0.14869900000000003, 0.143119, 0.145744, 0.14769200000000002, 0.14505500000000002, 0.1444]
args= 3 FFI typed nop            median=0.1502 samples=[0.151701, 0.148929, 0.15021299999999999, 0.155878, 0.14539, 0.151279, 0.15181899999999998, 0.147557, 0.147504]
args= 3 FFI empty kernel         median=3.7876 samples=[3.77264, 3.842817, 3.787632, 3.858806, 3.732656, 3.834638, 3.579131, 3.874781, 3.5995839999999997]
args= 3 INTJ static_compile kernel median=3.5793 samples=[3.321975, 3.526361, 3.5268629999999996, 3.5956, 3.5672020000000004, 3.579329, 3.8041590000000003, 3.592572, 3.8591979999999997]
args= 3 INTJ runtime_shim kernel median=3.4862 samples=[3.348089, 3.382661, 3.581516, 3.471559, 4.036624, 3.486237, 3.487065, 3.4648719999999997, 4.116597]
args= 5 FFI packed nop           median=0.1745 samples=[0.175399, 0.174543, 0.174561, 0.176297, 0.173279, 0.175291, 0.17252699999999999, 0.171627, 0.174513]
args= 5 FFI typed nop            median=0.1786 samples=[0.182264, 0.182565, 0.178673, 0.179233, 0.17688800000000002, 0.17859, 0.175142, 0.177875, 0.17794100000000002]
args= 5 FFI empty kernel         median=3.8134 samples=[3.722522, 4.3578090000000005, 3.81338, 4.391487, 3.8119520000000002, 4.321225999999999, 3.788395, 4.407437, 3.7676350000000003]
args= 5 INTJ static_compile kernel median=3.5586 samples=[3.323091, 3.6375, 3.5586100000000003, 3.573007, 3.568063, 3.558774, 3.463894, 3.557909, 3.4964589999999998]
args= 5 INTJ runtime_shim kernel median=3.4701 samples=[3.358888, 3.15476, 3.575046, 3.46044, 3.564562, 3.448985, 4.079131, 3.477135, 3.4700949999999997]
args= 8 FFI packed nop           median=0.2058 samples=[0.20521799999999998, 0.205213, 0.20532599999999998, 0.205787, 0.20807900000000001, 0.204594, 0.20582, 0.207584, 0.206983]
args= 8 FFI typed nop            median=0.2103 samples=[0.207547, 0.21249600000000002, 0.210288, 0.213784, 0.20847, 0.213451, 0.209685, 0.209427, 0.210637]
args= 8 FFI empty kernel         median=4.2800 samples=[3.973665, 4.34239, 3.854049, 4.337537, 6.126769, 4.279980999999999, 3.825227, 4.3248999999999995, 3.805843]
args= 8 INTJ static_compile kernel median=3.7124 samples=[3.418029, 3.600444, 4.1563490000000005, 3.7066, 16.008997, 3.622103, 4.071486, 3.712383, 4.04825]
args= 8 INTJ runtime_shim kernel median=3.4300 samples=[3.430041, 3.2976680000000003, 3.535904, 3.31093, 3.710382, 3.273902, 3.590943, 3.312712, 3.608517]
args=16 FFI packed nop           median=0.2936 samples=[0.294649, 0.290024, 0.29625799999999997, 0.29541500000000004, 0.295598, 0.28537799999999997, 0.293626, 0.292306, 0.291887]
args=16 FFI typed nop            median=0.3063 samples=[0.299856, 0.299119, 0.306938, 0.306301, 0.303682, 0.301813, 0.30927699999999997, 0.30891399999999997, 0.307504]
args=16 FFI empty kernel         median=4.2932 samples=[4.293187, 7.5438860000000005, 4.224761, 5.873771, 4.238375, 5.833961, 4.222281, 5.650011, 4.173476]
args=16 INTJ static_compile kernel median=5.4852 samples=[3.779245, 16.313901, 5.485227, 5.257566, 6.142243000000001, 5.293464, 5.699247000000001, 5.2293, 5.85407]
args=16 INTJ runtime_shim kernel median=4.9484 samples=[3.766444, 3.6716379999999997, 4.948394, 3.615663, 5.626668, 3.5974609999999996, 5.557135000000001, 6.712858, 5.551151]
args=32 FFI packed nop           median=0.4829 samples=[0.486909, 0.488331, 0.48868900000000004, 0.48291199999999995, 0.474526, 0.463922, 0.473762, 1.025723, 0.478328]
args=32 FFI typed nop            median=0.4962 samples=[0.48876, 0.506252, 0.50552, 0.493623, 0.499004, 0.49200099999999997, 0.495953, 0.986916, 0.496238]
args=32 FFI empty kernel         median=4.8719 samples=[4.871902, 5.924048, 4.777522, 6.017227, 4.775258, 5.83892, 4.692118000000001, 17.943194, 4.701625]
args=32 INTJ static_compile kernel median=5.5287 samples=[4.3730709999999995, 5.640456, 6.212445, 5.483809, 5.651911, 5.3473370000000005, 5.717344, 5.507565, 5.528685]
args=32 INTJ runtime_shim kernel median=4.1063 samples=[4.086668, 4.106296, 5.785273, 4.09009, 5.515441, 4.0953919999999995, 5.668781, 4.075114, 5.466388]
args=64 FFI packed nop           median=0.8437 samples=[0.8241660000000001, 0.8475539999999999, 0.837765, 0.866481, 0.8570030000000001, 0.839696, 0.843677, 0.818649, 0.8636820000000001]
args=64 FFI typed nop            median=0.8640 samples=[0.861924, 0.895548, 0.8532519999999999, 0.897223, 0.8585320000000001, 0.871933, 0.8562960000000001, 0.8640209999999999, 0.866458]
args=64 FFI empty kernel         median=6.0881 samples=[5.966279999999999, 6.388654000000001, 5.912256, 6.438229000000001, 5.949967, 6.488278, 5.885212999999999, 6.471569000000001, 6.088056]
args=64 INTJ static_compile kernel median=6.0193 samples=[5.442543, 6.129414, 6.005437, 6.019336, 5.562644000000001, 6.14945, 5.997801000000001, 6.118271, 6.141741]
args=64 INTJ runtime_shim kernel median=5.9211 samples=[5.444249, 6.142475, 5.7785519999999995, 6.070011999999999, 5.883627000000001, 6.105846, 5.867140999999999, 6.147758, 5.921098]
exit_status=0
ended=2026-09-25T18:36:25.344663+00:00
```

### hip_same_hsaco: develop r0

```text
commit=1bf689b37137e373b6baecf241a8d8a7da23630b
source_sha256=7faebcc700fe0b5bd1363acfc638b0621eebf7af754ab3dcf915ebd0c4e41806
started=2026-09-25T18:36:34.145319+00:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_hip_module_launch.py --iters 1000 --batches 9
hsaco_sha256=df3e529e178196ad62f7c07ab8593ede5b8596c76e7c6fbd12a5d947a696e65c kernel=empty_kernel threads=256 shared=0 callback_count=1
iters=1000 batches=9 unit=us/call (host enqueue, no timed sync)
TVM FFI HIP HSACO         median=3.1790 samples=[3.328884, 3.179012, 3.239406, 3.1860169999999997, 3.192708, 3.096924, 3.125895, 3.070293, 3.0634699999999997]
Triton same HSACO         median=15.8911 samples=[15.938455, 15.948215, 15.94567, 15.91481, 15.891119, 15.858967, 15.829647000000001, 15.837026, 15.760804]
INTJ same function        median=2.9622 samples=[2.962168, 2.9656170000000004, 2.9831239999999997, 3.0284389999999997, 2.987072, 2.87981, 2.884563, 2.880397, 2.8454450000000002]
exit_status=0
ended=2026-09-25T18:36:38.006002+00:00
```

### hip_same_hsaco: candidate r0

```text
commit=c5295b18f29a15fc1184eb7e93dbc099e6b75bf2
source_sha256=50f04d4c6bc58bd50a029525208892353ac1ba7184540ca7088d3ebaa9c2cb3f
started=2026-09-25T18:36:38.008503+00:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/callable_grid
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/callable_grid taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_hip_module_launch.py --iters 1000 --batches 9
hsaco_sha256=df3e529e178196ad62f7c07ab8593ede5b8596c76e7c6fbd12a5d947a696e65c kernel=empty_kernel threads=256 shared=0 callback_count=1
iters=1000 batches=9 unit=us/call (host enqueue, no timed sync)
TVM FFI HIP HSACO         median=3.2591 samples=[3.638842, 3.2760749999999996, 3.2591010000000002, 3.260738, 3.26527, 3.1301889999999997, 3.152819, 3.121647, 3.0975949999999997]
Triton same HSACO         median=16.3616 samples=[16.56707, 16.478497, 16.379906000000002, 16.361604, 16.584092000000002, 16.248557, 16.207252, 16.181985, 16.336723]
INTJ same function        median=3.0566 samples=[3.130402, 3.1166750000000003, 3.056629, 3.0625459999999998, 3.016285, 2.899294, 2.918091, 2.9112579999999997, 14.509969]
exit_status=0
ended=2026-09-25T18:36:43.981934+00:00
```

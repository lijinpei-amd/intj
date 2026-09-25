# Callable-grid `spec_key` ABBA probe — 2026-09-26, repeat 0 of 4

Baseline `develop` is `1bf689b37137e373b6baecf241a8d8a7da23630b`;
candidate `callable_grid` is `c5295b18f29a15fc1184eb7e93dbc099e6b75bf2`.
The local date is 2026-09-26 (Asia/Shanghai); log timestamps are
2026-09-25 UTC. The eight processes ran D0 → C0 → C1 → D1 → D2 → C2 → C3 → D3,
all with exit status 0. This file holds round 0's two raw logs;
[repeat 1](2026-09-26_callable-grid-spec-key_1_c5295b1.md), [repeat 2](2026-09-26_callable-grid-spec-key_2_c5295b1.md), and
[repeat 3](2026-09-26_callable-grid-spec-key_3_c5295b1.md) hold the others. This isolates the README decode
row from the [full comparison](2026-09-26_callable-grid-full_0_c5295b1.md).

## Aggregate across four decode-only rounds

Each cell below is a process median in ns/call from nine batches of 100,000
calls. Paired changes are candidate minus develop for matching round indices;
the last column is their median. This direct `spec_key` call includes Python
loop/call overhead and does not time a kernel launch.

| Mode | Develop r0–r3 | Candidate r0–r3 | Paired changes | Median change |
| --- | --- | --- | --- | ---: |
| Runtime shim | 121.7335, 116.4521, 120.1140, 116.3984 | 125.3748, 119.3693, 119.5039, 121.2035 | +3.6413, +2.9172, −0.6101, +4.8051 | +3.2793 |
| Static compile | 119.8449, 114.0653, 114.9326, 114.0310 | 121.3914, 118.2014, 116.6709, 121.4238 | +1.5465, +4.1361, +1.7383, +7.3928 | +2.9372 |
| Interpreter | 907.5351, 980.0412, 909.6708, 902.7371 | 914.5784, 904.9665, 912.1350, 909.8821 | +7.0433, −75.0747, +2.4642, +7.1450 | +4.7538 |

Static compile has a small repeated positive difference, and runtime shim
moves by a similar amount. A [same-process diagnostic](2026-09-26_callable-grid-spec-key-same-process_0_c5295b1.md)
finds roughly 1–2 ns/call higher candidate cost in its cleaner pairs, but
host state and interrupted rounds prevent a precise estimate. The generated
`spec_key` code is identical across revisions; its linked address changes.
These timings do not isolate a decoding-logic regression, and the full
launcher does not show a material repeatable slowdown. The interpreter r1
baseline is noisy.

## Environment, exact command, and timing boundary

Intel Xeon Platinum 8480C pinned to CPU 0; AMD Instinct MI308X GPU 0,
`gfx942:sramecc+:xnack-`. Python 3.12.3 at `/tmp/gb2/bin/python`, Torch
`2.14.0+rocm7.2` (`torch.version.hip=7.2.53211`), Triton `3.8.0`, TVM FFI
`0.1.14.post2.dev1+g424558557.d20260924`. `TRITON_HOME`,
`TVM_FFI_CACHE_DIR`, `CC`, `CXX`, and GPU visibility masks were unset in the
baseline environment snapshot; the driver inherited its environment and set
only checkout-specific `PYTHONPATH`. It did not clear caches; the logs do not
establish a cold cache. The driver logged Git HEAD but not a source hash or
whole-tree status.

Each raw block contains its exact child command:
`PYTHONPATH=<checkout> taskset -c 0 /tmp/gb2/bin/python
/tmp/intj_spec_key_bench_2026-09-26.py --iters 100000 --batches 9`.
The script creates CUDA-device tensors and the launcher before timing, warms
100 calls per mode, synchronizes, then times direct calls to the generated
`spec_key` method in nine 100,000-call batches. It synchronizes after each
batch; no GPU kernel is launched in the timed loop. Torch's `cuda` device is
backed by ROCm here; CUDA runtime was not tested. Its complete source is
preserved here because the command uses a temporary `/tmp` script:

```python
"""Time the README spec_key rows with longer batches and no GPU launches."""

import argparse
import statistics
import time

import torch

from benchmarks.bench_launch import noop
from intj import TorchAccessMode, make_launcher


parser = argparse.ArgumentParser()
parser.add_argument("--iters", type=int, default=100000)
parser.add_argument("--batches", type=int, default=9)
args = parser.parse_args()

n = 4096
inputs = (
    torch.randn(n, device="cuda"),
    torch.randn(n, device="cuda"),
    torch.empty(n, device="cuda"),
    n,
    1.5,
    128,
)
for mode in (
    TorchAccessMode.RUNTIME_SHIM,
    TorchAccessMode.STATIC_COMPILE,
    TorchAccessMode.INTERPRETER,
):
    module = make_launcher(noop, torch_access_mode=mode).__self__
    fn = module.spec_key
    for _ in range(100):
        fn(*inputs)
    torch.cuda.synchronize()
    samples = []
    for _ in range(args.batches):
        start = time.perf_counter_ns()
        for _ in range(args.iters):
            fn(*inputs)
        samples.append((time.perf_counter_ns() - start) / args.iters)
        torch.cuda.synchronize()
    print(f"{mode.name.lower()} median_ns={statistics.median(samples):.4f} samples_ns={samples}")
```

## Raw outputs — repeat 0

### develop r0

```text
commit=1bf689b37137e373b6baecf241a8d8a7da23630b
started=2026-09-25T18:43:00.210761+00:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj taskset -c 0 /tmp/gb2/bin/python /tmp/intj_spec_key_bench_2026-09-26.py --iters 100000 --batches 9
runtime_shim median_ns=121.7335 samples_ns=[121.18548, 122.19539, 120.25976, 121.81907, 120.7605, 121.41396, 122.97855, 121.7536, 121.73348]
static_compile median_ns=119.8449 samples_ns=[121.53171, 124.9072, 119.9919, 120.28256, 118.2183, 119.62344, 118.73519, 119.57446, 119.84492]
interpreter median_ns=907.5351 samples_ns=[917.02185, 906.16983, 904.84584, 906.06832, 906.31645, 907.53514, 908.46506, 907.58652, 908.3114]
exit_status=0
ended=2026-09-25T18:43:07.720164+00:00
```

### candidate r0

```text
commit=c5295b18f29a15fc1184eb7e93dbc099e6b75bf2
started=2026-09-25T18:43:07.722730+00:00
cwd=/mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/callable_grid
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/callable_grid taskset -c 0 /tmp/gb2/bin/python /tmp/intj_spec_key_bench_2026-09-26.py --iters 100000 --batches 9
runtime_shim median_ns=125.3748 samples_ns=[123.87455, 127.52583, 124.66123, 125.3748, 126.35749, 123.37598, 123.7461, 235.21222, 129.94962]
static_compile median_ns=121.3914 samples_ns=[123.53943, 121.84353, 121.52133, 121.35164, 121.23498, 120.22105, 121.39139, 120.77541, 226.29799]
interpreter median_ns=914.5784 samples_ns=[1106.54739, 913.13341, 913.6583, 906.57568, 908.09835, 915.07149, 914.57841, 916.46999, 977.62125]
exit_status=0
ended=2026-09-25T18:43:15.399438+00:00
```

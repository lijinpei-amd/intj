# Callable-grid same-process `spec_key` diagnostic — 2026-09-26, repeat 0 of 1

The baseline `develop` checkout was clean at
`1bf689b37137e373b6baecf241a8d8a7da23630b`. The candidate
`callable_grid` code was at `c5295b18f29a15fc1184eb7e93dbc099e6b75bf2`;
only benchmark journals were untracked there. The local measurement date was
2026-09-26 (Asia/Shanghai), and all three commands exited 0. This diagnostic
follows the [separate-process `spec_key` probe](2026-09-26_callable-grid-spec-key_0_c5295b1.md)
and the [full launcher comparison](2026-09-26_callable-grid-full_0_c5295b1.md).

| Run | Median candidate minus develop | Positive pairs | Main observation |
| --- | ---: | ---: | --- |
| File run 1 | +7.23835 ns/call | 12/12 | Both modules cost about twice as much as in later runs; host state changed. |
| Stdin run | +2.28936 ns/call | 12/12 | One pair has a +32.42398 ns interruption. |
| File run 2 | +1.04380 ns/call | 9/12 | Two pairs have large interruptions of opposite sign. |

These are direct `module.spec_key(*args)` calls, not kernel launches. The paired
medians include every round without filtering. The varying absolute cost and
interrupted pairs prevent a precise regression estimate; cleaner pairs
repeatedly put the candidate roughly 1–2 ns/call higher. This small key-only
difference remains unresolved. The generated `intj_pack` and `spec_key` source
slices are byte-identical across the two modules, so no change to the decode
logic explains it. The `spec_key` symbols have the same 0x1484-byte size but
move from offset 0x30e0 to 0x3130, consistent with binary-placement or
host-state effects. The full launcher comparison did not show a repeatable
material slowdown.

## Environment and timing boundary

Intel Xeon Platinum 8480C, CPU 0 (`taskset -c 0`), AMD Instinct MI308X
(`gfx942`). Python 3.12.3, Torch `2.14.0+rocm7.2` (HIP `7.2.53211`), and
Triton `3.8.0`. The process inherited the shell environment; this diagnostic
did not snapshot it or clear caches. Both extension binaries were built before
measurement. Their SHA-256 values are in every raw output below. Baseline and
candidate loaded into one Python process; it allocated ROCm tensors before
timing, warmed each method with 10,000 calls, synchronized once, and disabled
GC. Each run measured 12 alternating DCCD/CDDC rounds. Each of four
observations per round timed 250,000 calls; D and C are arithmetic means of
their two observations. No GPU kernel launch or synchronization is in the timed
loop. The host changed state between processes: the first run's roughly
180 ns/call absolute cost fell to roughly 90 ns/call in the subsequent runs.
This diagnostic did not capture the cause.

The generated C++ byte slices from `static INTJ_ALWAYS_INLINE int intj_pack(`
to `/* Parse one grid dimension.` have matching SHA-256
`1f28c981329fb9528613d9d361416f8a43475134197e77a0ceda7ccfdf617556`.
The slices from `static PyObject *spec_key(` to
`static PyObject *set_compile_callback(` have matching SHA-256
`be8e8050f168b704627e389061f7b91fd0535f93108cc4b73b897b8e5e8f24df`.
`nm -a -S` on the two `.so` files gives offsets and sizes `0x30e0/0x1484` and
`0x3130/0x1484`, respectively.

## Exact script and commands

The script was `/tmp/intj_spec_key_same_process_2026-09-26.py`:

```python
"""Compare the same two compiled spec_key methods in one pinned process."""

import gc
import hashlib
import importlib.util
import statistics
import sys
import time

import torch
import triton

from benchmarks.bench_launch import noop
from intj import TorchAccessMode, make_launcher
from intj.torch_intf.torch_abi import dtype_index_table, layout_for, torch_version

candidate_path = "/mnt/nvme2/jinpli/workspace/home/jinpli/.triton/intj/f15128c1c3ee8923ee6b0abb4a3bed9f49c1152c78418531c35f420dbff64e2f/benchmarks.bench_launch/noop.cpython-312-x86_64-linux-gnu.so"
baseline = make_launcher(noop, torch_access_mode=TorchAccessMode.STATIC_COMPILE).__self__
spec = importlib.util.spec_from_file_location("noop", candidate_path)
assert spec is not None and spec.loader is not None
candidate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(candidate)
candidate.set_torch_version(torch_version(), layout_for().as_args(), dtype_index_table())
args = (
    torch.randn(4096, device="cuda"),
    torch.randn(4096, device="cuda"),
    torch.empty(4096, device="cuda"),
    4096,
    1.5,
    128,
)
fn = {"D": baseline.spec_key, "C": candidate.spec_key}
assert fn["D"](*args) == fn["C"](*args)
for method in fn.values():
    for _ in range(10000):
        method(*args)
torch.cuda.synchronize()
print(f"python={sys.version.split()[0]} torch={torch.__version__} hip={torch.version.hip} triton={triton.__version__}")
print(f"gpu={torch.cuda.get_device_name(0)}")
for tag, module in (("D", baseline), ("C", candidate)):
    with open(module.__file__, "rb") as binary:
        digest = hashlib.file_digest(binary, "sha256").hexdigest()
    print(f"{tag}_so={module.__file__} sha256={digest}")

def measure(method, n=250000):
    start = time.perf_counter_ns()
    for _ in range(n):
        method(*args)
    return (time.perf_counter_ns() - start) / n

gc.disable()
deltas = []
for round_index in range(12):
    order = "CDDC" if round_index % 2 else "DCCD"
    raw = [(tag, measure(fn[tag])) for tag in order]
    develop = statistics.mean(value for tag, value in raw if tag == "D")
    candidate_value = statistics.mean(value for tag, value in raw if tag == "C")
    delta = candidate_value - develop
    deltas.append(delta)
    print(f"round={round_index} order={order} raw_ns={[round(value, 5) for _, value in raw]} D={develop:.5f} C={candidate_value:.5f} delta={delta:+.5f} ns/call", flush=True)
print(f"paired_median_delta={statistics.median(deltas):+.5f} ns/call; positive={sum(value > 0 for value in deltas)}/{len(deltas)}; range=[{min(deltas):+.5f},{max(deltas):+.5f}]")
```

### file run 1

```sh
PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj taskset -c 0 /tmp/gb2/bin/python /tmp/intj_spec_key_same_process_2026-09-26.py > /tmp/intj_spec_key_same_process_2026-09-26.log
```

```text
python=3.12.3 torch=2.14.0+rocm7.2 hip=7.2.53211 triton=3.8.0
gpu=AMD Instinct MI308X
D_so=/mnt/nvme2/jinpli/workspace/home/jinpli/.triton/intj/2f26203d7bfbb2f0812cbc522a29f8e2a88746e5f606d3f7a5082fe412b13a99/benchmarks.bench_launch/noop.cpython-312-x86_64-linux-gnu.so sha256=835ad469afbf348032ec089ef03ebed92072bf53b8d2511e43ad6850fbcab878
C_so=/mnt/nvme2/jinpli/workspace/home/jinpli/.triton/intj/f15128c1c3ee8923ee6b0abb4a3bed9f49c1152c78418531c35f420dbff64e2f/benchmarks.bench_launch/noop.cpython-312-x86_64-linux-gnu.so sha256=90c4ba785cda0ba050f7c5ded78fb35bcb36a661202840e94f0904a447bf90e1
round=0 order=DCCD raw_ns=[192.59231, 185.48168, 185.00945, 176.41595] D=184.50413 C=185.24556 delta=+0.74143 ns/call
round=1 order=CDDC raw_ns=[190.44653, 183.19037, 183.41105, 190.7079] D=183.30071 C=190.57721 delta=+7.27650 ns/call
round=2 order=DCCD raw_ns=[183.45453, 190.7851, 190.98806, 183.22062] D=183.33758 C=190.88658 delta=+7.54900 ns/call
round=3 order=CDDC raw_ns=[190.26767, 183.28642, 183.4574, 190.24263] D=183.37191 C=190.25515 delta=+6.88324 ns/call
round=4 order=DCCD raw_ns=[183.40193, 189.80171, 190.3973, 184.92354] D=184.16273 C=190.09950 delta=+5.93677 ns/call
round=5 order=CDDC raw_ns=[190.34567, 183.44176, 183.66522, 190.50457] D=183.55349 C=190.42512 delta=+6.87163 ns/call
round=6 order=DCCD raw_ns=[183.3122, 190.51262, 190.19497, 182.96646] D=183.13933 C=190.35380 delta=+7.21447 ns/call
round=7 order=CDDC raw_ns=[191.26865, 183.37485, 183.36302, 190.55017] D=183.36894 C=190.90941 delta=+7.54047 ns/call
round=8 order=DCCD raw_ns=[183.567, 190.55835, 190.67948, 183.14638] D=183.35669 C=190.61892 delta=+7.26223 ns/call
round=9 order=CDDC raw_ns=[191.07136, 183.2337, 183.33086, 190.44127] D=183.28228 C=190.75632 delta=+7.47404 ns/call
round=10 order=DCCD raw_ns=[183.97248, 190.77471, 191.0589, 183.23338] D=183.60293 C=190.91681 delta=+7.31388 ns/call
round=11 order=CDDC raw_ns=[190.74816, 183.46062, 183.50687, 190.4145] D=183.48375 C=190.58133 delta=+7.09758 ns/call
paired_median_delta=+7.23835 ns/call; positive=12/12; range=[+0.74143,+7.54900]
```

### stdin run

```sh
PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj taskset -c 0 /tmp/gb2/bin/python - < /tmp/intj_spec_key_same_process_2026-09-26.py > /tmp/intj_spec_key_same_process_stdin_2026-09-26.log
```

```text
python=3.12.3 torch=2.14.0+rocm7.2 hip=7.2.53211 triton=3.8.0
gpu=AMD Instinct MI308X
D_so=/mnt/nvme2/jinpli/workspace/home/jinpli/.triton/intj/2f26203d7bfbb2f0812cbc522a29f8e2a88746e5f606d3f7a5082fe412b13a99/benchmarks.bench_launch/noop.cpython-312-x86_64-linux-gnu.so sha256=835ad469afbf348032ec089ef03ebed92072bf53b8d2511e43ad6850fbcab878
C_so=/mnt/nvme2/jinpli/workspace/home/jinpli/.triton/intj/f15128c1c3ee8923ee6b0abb4a3bed9f49c1152c78418531c35f420dbff64e2f/benchmarks.bench_launch/noop.cpython-312-x86_64-linux-gnu.so sha256=90c4ba785cda0ba050f7c5ded78fb35bcb36a661202840e94f0904a447bf90e1
round=0 order=DCCD raw_ns=[92.69195, 94.7799, 93.9038, 91.99813] D=92.34504 C=94.34185 delta=+1.99681 ns/call
round=1 order=CDDC raw_ns=[93.49548, 91.28031, 91.1099, 93.42598] D=91.19511 C=93.46073 delta=+2.26562 ns/call
round=2 order=DCCD raw_ns=[91.85826, 93.11082, 93.32418, 91.2789] D=91.56858 C=93.21750 delta=+1.64892 ns/call
round=3 order=CDDC raw_ns=[94.87044, 91.46047, 91.0586, 94.38937] D=91.25954 C=94.62991 delta=+3.37037 ns/call
round=4 order=DCCD raw_ns=[92.55526, 93.09, 92.85774, 91.14944] D=91.85235 C=92.97387 delta=+1.12152 ns/call
round=5 order=CDDC raw_ns=[91.24522, 89.04846, 121.49522, 184.14642] D=105.27184 C=137.69582 delta=+32.42398 ns/call
round=6 order=DCCD raw_ns=[89.8402, 90.08513, 90.13454, 89.08256] D=89.46138 C=90.10984 delta=+0.64846 ns/call
round=7 order=CDDC raw_ns=[92.15715, 89.72941, 88.50294, 90.95454] D=89.11617 C=91.55584 delta=+2.43967 ns/call
round=8 order=DCCD raw_ns=[90.15879, 90.60645, 93.01464, 88.8361] D=89.49744 C=91.81054 delta=+2.31310 ns/call
round=9 order=CDDC raw_ns=[94.20089, 91.45365, 90.92892, 93.4784] D=91.19128 C=93.83964 delta=+2.64836 ns/call
round=10 order=DCCD raw_ns=[90.91669, 93.25538, 93.61253, 91.08104] D=90.99887 C=93.43395 delta=+2.43509 ns/call
round=11 order=CDDC raw_ns=[93.66556, 91.64359, 90.60474, 92.44348] D=91.12416 C=93.05452 delta=+1.93036 ns/call
paired_median_delta=+2.28936 ns/call; positive=12/12; range=[+0.64846,+32.42398]
```

### file run 2

```sh
PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj taskset -c 0 /tmp/gb2/bin/python /tmp/intj_spec_key_same_process_2026-09-26.py > /tmp/intj_spec_key_same_process_file_repeat_2026-09-26.log
```

```text
python=3.12.3 torch=2.14.0+rocm7.2 hip=7.2.53211 triton=3.8.0
gpu=AMD Instinct MI308X
D_so=/mnt/nvme2/jinpli/workspace/home/jinpli/.triton/intj/2f26203d7bfbb2f0812cbc522a29f8e2a88746e5f606d3f7a5082fe412b13a99/benchmarks.bench_launch/noop.cpython-312-x86_64-linux-gnu.so sha256=835ad469afbf348032ec089ef03ebed92072bf53b8d2511e43ad6850fbcab878
C_so=/mnt/nvme2/jinpli/workspace/home/jinpli/.triton/intj/f15128c1c3ee8923ee6b0abb4a3bed9f49c1152c78418531c35f420dbff64e2f/benchmarks.bench_launch/noop.cpython-312-x86_64-linux-gnu.so sha256=90c4ba785cda0ba050f7c5ded78fb35bcb36a661202840e94f0904a447bf90e1
round=0 order=DCCD raw_ns=[96.42826, 91.83278, 92.94547, 88.89986] D=92.66406 C=92.38912 delta=-0.27494 ns/call
round=1 order=CDDC raw_ns=[94.86644, 91.98999, 93.01601, 93.05008] D=92.50300 C=93.95826 delta=+1.45526 ns/call
round=2 order=DCCD raw_ns=[92.19171, 93.02112, 94.54392, 198.26255] D=145.22713 C=93.78252 delta=-51.44461 ns/call
round=3 order=CDDC raw_ns=[103.67893, 91.45574, 91.07154, 93.55691] D=91.26364 C=98.61792 delta=+7.35428 ns/call
round=4 order=DCCD raw_ns=[92.3566, 93.01694, 93.8783, 91.7252] D=92.04090 C=93.44762 delta=+1.40672 ns/call
round=5 order=CDDC raw_ns=[91.38156, 89.50837, 89.1173, 90.54706] D=89.31284 C=90.96431 delta=+1.65147 ns/call
round=6 order=DCCD raw_ns=[90.02048, 91.24078, 91.02454, 89.2658] D=89.64314 C=91.13266 delta=+1.48952 ns/call
round=7 order=CDDC raw_ns=[91.48008, 88.00233, 91.93532, 89.70544] D=89.96882 C=90.59276 delta=+0.62394 ns/call
round=8 order=DCCD raw_ns=[90.03822, 90.82254, 121.41947, 136.44767] D=113.24295 C=106.12100 delta=-7.12194 ns/call
round=9 order=CDDC raw_ns=[94.77466, 92.67852, 92.06786, 92.71043] D=92.37319 C=93.74255 delta=+1.36936 ns/call
round=10 order=DCCD raw_ns=[92.49366, 92.37711, 93.35119, 91.978] D=92.23583 C=92.86415 delta=+0.62832 ns/call
round=11 order=CDDC raw_ns=[92.6094, 92.4654, 91.43206, 92.72454] D=91.94873 C=92.66697 delta=+0.71824 ns/call
paired_median_delta=+1.04380 ns/call; positive=9/12; range=[-51.44461,+7.35428]
```

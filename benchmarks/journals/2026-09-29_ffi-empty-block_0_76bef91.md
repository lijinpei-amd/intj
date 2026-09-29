# `ffi_compare` with matched empty-kernel block size (index 0)

`bench_ffi_compare.py`'s FFI `launch_empty` launched `EmptyKernel<<<1, 1>>>` (1 thread), while INTJ's Triton
`empty_kernel` launches 1 block of 4 warps x 64 = 256 threads. Changed it to `<<<1, kMixedThreads>>>` (256 on ROCm,
128 on CUDA), the block every other FFI kernel in the script already uses, and reran `ffi_compare`.
Before: `2026-09-29_full-suite_0_76bef91.md` (same day, same commit, `ffi_compare` case).

## Environment

- Source: `76bef91` (branch `develop`) plus the **uncommitted** one-line fix in `benchmarks/bench_ffi_compare.py`
  (sha256 `ef710426ebb9bfa15c58954187c996e842c8b105d17f792070b87592b7e89f3f`); no `intj/` change.
- CPU: Intel Xeon Platinum 8480C, `taskset -c 0`, one process at a time; `uptime` load 5.03, 5.22, 7.84 before,
  5.85, 5.39, 7.85 after (per-process `/proc/loadavg` in the raw output).
- GPU: AMD Instinct MI308X (`gfx942`), GPU 0 (`HIP_VISIBLE_DEVICES=0`).
- Python 3.12.3 (`/tmp/gb2/bin/python`), Torch `2.14.0+rocm7.2`, Triton `3.8.0`, apache-tvm-ffi
  `0.1.14.post2.dev1+g424558557.d20260924`, cc 13.3.0. CUDA: untested.

## Command

```bash
for r in 0 1 2; do PYTHONPATH=$PWD HIP_VISIBLE_DEVICES=0 taskset -c 0 /tmp/gb2/bin/python \
  benchmarks/bench_ffi_compare.py --iters 1000 --batches 9 > /tmp/intj-bench/empty-block/round$r.txt; done
```

1000 calls x 9 batches per process, 3 processes; host call/enqueue time (timer stops before the sync).

## Result

Median of the 3 process medians (us/call):

| row | before (FFI empty `<<<1, 1>>>`) | after (`<<<1, 256>>>`) | after rounds |
|---|---:|---:|---|
| FFI packed nop | 0.1459 | 0.1453 | 0.1453, 0.1448, 0.1456 |
| FFI typed nop | 0.1496 | 0.1500 | 0.1506, 0.1500, 0.1492 |
| FFI empty kernel | 3.3657 | 3.3519 | 3.3519, 3.3029, 3.6764 |
| INTJ empty kernel | 2.9834 | 3.0692 | 3.0692, 3.0044, 3.6439 |
| INTJ fixed-device kernel | 2.9304 | 2.9339 | 2.9339, 2.8482, 3.3358 |
| FFI packed nop mixed | 0.1742 | 0.1738 | 0.1739, 0.1737, 0.1738 |
| FFI typed nop mixed | 0.1779 | 0.1779 | 0.1765, 0.1779, 0.1779 |
| FFI mixed kernel | 3.3903 | 3.3725 | 3.3725, 3.3045, 3.8562 |
| INTJ mixed kernel | 2.9907 | 2.9901 | 2.9901, 2.9306, 3.5703 |
| INTJ fixed mixed kernel | 2.9159 | 2.9748 | 2.9748, 2.9418, 3.6071 |

- The block size does not measurably change the FFI empty-kernel enqueue: 3.37 -> 3.35 us. INTJ stays ahead on
  the matched pair (3.07 vs 3.35 us), as it did before the fix.
- Round 2 is a slow process: every kernel row, FFI and INTJ alike, is +0.3-0.6 us; no-op rows are unchanged. This
  is the GPU/host window effect seen in earlier FFI journals; the median drops it.
- Kernels are still different binaries (HIP C++ vs Triton), and FFI receives preconverted TensorViews while INTJ
  receives Torch tensors. Only `bench_hip_module_launch.py` compares one HSACO.

## Raw output

### `round0.txt`

```
commit=23b44640d0b97d73e80d343a9aec79695f40219f (+uncommitted bench_ffi_compare.py fix)
loadavg=5.03 5.22 7.84 4/15497 1411229
iters=1000 batches=9 unit=us/call (host-only, no timed sync)
FFI packed nop             median=0.1453 samples=[0.152666, 0.145327, 0.145677, 0.146237, 0.144716, 0.143783, 0.14716300000000002, 0.144028, 0.14402]
FFI typed nop              median=0.1506 samples=[0.147648, 0.15118, 0.149884, 0.151372, 0.153272, 0.15063300000000002, 0.148154, 0.151445, 0.14778899999999998]
FFI empty kernel           median=3.3519 samples=[3.555688, 3.398497, 3.351899, 3.18384, 3.219589, 3.4771729999999996, 3.307954, 3.39617, 3.2265680000000003]
INTJ empty kernel          median=3.0692 samples=[3.0691840000000004, 3.082872, 3.1688359999999998, 2.917498, 2.853333, 3.09879, 3.157127, 3.000938, 3.0135650000000003]
INTJ fixed-device kernel   median=2.9339 samples=[2.921275, 3.083168, 3.065338, 2.889818, 2.852744, 7.817279999999999, 2.9338640000000002, 2.889966, 2.93838]
FFI packed nop mixed       median=0.1739 samples=[0.173923, 0.170308, 0.177715, 0.17421899999999998, 0.172672, 0.621347, 0.17392, 0.17316499999999999, 0.172676]
FFI typed nop mixed        median=0.1765 samples=[0.174882, 0.17569300000000002, 0.172956, 0.179696, 0.177985, 0.397296, 0.176542, 0.18048599999999998, 0.17350100000000002]
FFI mixed kernel           median=3.3725 samples=[3.377211, 3.441095, 3.305995, 3.241622, 3.2229270000000003, 18.413297, 3.3725479999999997, 3.436267, 3.271174]
INTJ mixed kernel          median=2.9901 samples=[3.19923, 3.13957, 2.9460279999999996, 2.9359830000000002, 2.889888, 13.904736000000002, 2.990147, 3.095501, 2.92482]
INTJ fixed mixed kernel    median=2.9748 samples=[3.122554, 3.1113850000000003, 2.955648, 2.974812, 2.86889, 9.885196, 2.960773, 3.020745, 2.898536]
exit_status=0
```

### `round1.txt`

```
commit=23b44640d0b97d73e80d343a9aec79695f40219f (+uncommitted bench_ffi_compare.py fix)
loadavg=5.40 5.29 7.83 6/15504 1411263
iters=1000 batches=9 unit=us/call (host-only, no timed sync)
FFI packed nop             median=0.1448 samples=[0.148898, 0.144373, 0.14537899999999998, 0.144788, 0.146047, 0.14469900000000002, 0.144244, 0.143767, 0.145352]
FFI typed nop              median=0.1500 samples=[0.15552000000000002, 0.14985400000000001, 0.151492, 0.149985, 0.148563, 0.152125, 0.147991, 0.15318500000000002, 0.14861600000000003]
FFI empty kernel           median=3.3029 samples=[3.553969, 3.393221, 3.4035949999999997, 3.203038, 3.157022, 3.307484, 3.206074, 3.302889, 3.205075]
INTJ empty kernel          median=3.0044 samples=[3.049951, 3.079056, 3.087951, 2.9287240000000003, 2.934578, 3.0043130000000002, 3.0119000000000002, 2.995826, 3.0044020000000002]
INTJ fixed-device kernel   median=2.8482 samples=[2.92793, 3.0969450000000003, 3.0897229999999998, 2.9223049999999997, 2.789641, 2.8399870000000003, 2.816554, 2.848217, 2.839998]
FFI packed nop mixed       median=0.1737 samples=[0.17216900000000002, 0.172138, 0.171486, 0.173744, 0.174185, 0.174277, 0.175228, 0.173321, 0.176009]
FFI typed nop mixed        median=0.1779 samples=[0.17786600000000002, 0.180044, 0.173518, 0.178457, 0.177839, 0.17871199999999998, 0.17538800000000002, 0.178587, 0.175184]
FFI mixed kernel           median=3.3045 samples=[3.398619, 3.4458029999999997, 3.304536, 3.2744560000000003, 3.202645, 3.324331, 3.265243, 3.331239, 3.2888800000000002]
INTJ mixed kernel          median=2.9306 samples=[3.101336, 3.094011, 2.947908, 2.930595, 2.820095, 2.8803389999999998, 2.8653560000000002, 2.962189, 2.887542]
INTJ fixed mixed kernel    median=2.9418 samples=[3.0884050000000003, 3.102148, 2.947556, 2.941823, 2.872112, 2.864194, 2.926363, 2.935596, 2.942681]
exit_status=0
```

### `round2.txt`

```
commit=23b44640d0b97d73e80d343a9aec79695f40219f (+uncommitted bench_ffi_compare.py fix)
loadavg=5.85 5.39 7.85 5/15512 1411275
iters=1000 batches=9 unit=us/call (host-only, no timed sync)
FFI packed nop             median=0.1456 samples=[0.150416, 0.144644, 0.14438900000000002, 0.146147, 0.143512, 0.149874, 0.145589, 0.146339, 0.145238]
FFI typed nop              median=0.1492 samples=[0.148781, 0.15021500000000002, 0.147054, 0.149232, 0.15303, 0.15001499999999998, 0.147198, 0.15162299999999998, 0.149117]
FFI empty kernel           median=3.6764 samples=[3.635919, 4.487163, 3.676364, 4.414169, 3.511415, 4.448353, 3.558921, 4.412675, 3.5726430000000002]
INTJ empty kernel          median=3.6439 samples=[3.153397, 3.590149, 4.1150839999999995, 3.543796, 3.957294, 3.6438539999999997, 3.92729, 3.60983, 3.913603]
INTJ fixed-device kernel   median=3.3358 samples=[3.0036840000000002, 3.335823, 3.5565700000000002, 3.190099, 3.508068, 3.1855300000000004, 12.455465, 3.2211399999999997, 3.431757]
FFI packed nop mixed       median=0.1738 samples=[0.174103, 0.173208, 0.178422, 0.173797, 0.173426, 0.173759, 0.409479, 0.173018, 0.17408500000000002]
FFI typed nop mixed        median=0.1779 samples=[0.173261, 0.17777, 0.17669300000000002, 0.17951, 0.17678, 0.17787799999999998, 0.377777, 0.182885, 0.180417]
FFI mixed kernel           median=3.8562 samples=[3.713692, 4.629614999999999, 3.782788, 4.526407, 3.728283, 3.856162, 4.892585, 4.524116, 3.725541]
INTJ mixed kernel          median=3.5703 samples=[4.25063, 3.737555, 3.549004, 3.585503, 3.512458, 3.567138, 3.570325, 3.6440360000000003, 3.53706]
INTJ fixed mixed kernel    median=3.6071 samples=[3.743456, 3.659025, 3.626953, 3.6448690000000004, 3.499976, 3.550034, 3.491308, 3.607096, 3.510406]
exit_status=0
```

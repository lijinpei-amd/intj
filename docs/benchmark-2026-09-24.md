# Launcher optimization measurements — 2026-09-24

The repeatable improvement is for large scalar argument lists: 32 integer
arguments fell from **370.9 ns to 325.6 ns (12.2%)**. Across the five processes,
these measurements ranged from 369.3–373.5 ns before and 325.2–326.2 ns after.
The standard six-argument host path and GPU launch show no meaningful gain.
The 16-tensor case became 1.7% slower; the full results below retain regressions.

## Changes

1. Disable redundant CPython inline assertions while including `Python.h`.
   Restore the caller's `NDEBUG` setting before Torch headers, preserving Torch's
   checks. Integer arguments already pass explicit exact-type checks before
   `PyUnstable_Long_IsCompact` / `PyUnstable_Long_CompactValue`. Keeping this in
   the template makes the module digest invalidate existing binaries.
2. Check each grid dimension for zero directly. This removes two multiplies and
   fixes nonzero grids such as `(1 << 22, 1 << 21, 1 << 21)` being skipped when
   their product overflows a 64-bit integer. Tests reproduce the old failure.
3. Stop the host timer before GPU synchronization and add `--sweep` to reproduce
   the argument-count measurements using the existing timing helper.

## Method

- Baseline: `4a3df7f359093f717f181fa1e7d684e53baa706f`, archived before editing.
- Candidate: working-tree runtime template, SHA-256 `38b8a7ae495d22eeb3ecebfad40d2d85dba1f6142ac38d6321097a467e63481c`.
- Intel Xeon Platinum 8480C, CPU 0 affinity; AMD Instinct MI308X, gfx942, GPU 0.
- CPython 3.12.3, Torch 2.14.0+rocm7.2, Triton 3.8.0, GCC 13.3.0;
  default tensor access resolves to CXX.
- Five separate processes per revision and mode, alternating before/after order.
  Each value is the median of the five process medians. Host modes use nine
  batches of 100,000 calls; GPU/README modes use seven batches of 20,000 calls.

Both revisions use the updated benchmark harness, including the corrected timing
boundary. Separate initially empty `TRITON_HOME` directories force each revision
to build its own extensions; subsequent processes reuse those artifacts.
Every process records the imported `intj` source path. No runs were discarded.
GPU comparisons ran after tests and other benchmarks completed.

The sweep uses an empty kernel in `no_gpu=True` mode, with all arguments set to
integer 17 or the same CPU tensor. It measures dispatch, argument decoding, key
construction, and cache lookup with hot tensor metadata. It has no constexpr
arguments. The standard matrix uses three tensors, an integer, a float, and a
constexpr. Its Python calling wrapper differs from the sweep, so compare each
row against the same row in the other revision.

## Argument-count sweep

Negative time changes are faster.

| Case | Before (ns) | After (ns) | Time change |
|---|---:|---:|---:|
| 4 int | 144.7 | 144.4 | -0.2% |
| 4 tensor | 148.6 | 147.9 | -0.5% |
| 16 int | 224.9 | 222.4 | -1.1% |
| 16 tensor | 257.9 | 262.4 | +1.7% |
| 32 int | 370.9 | 325.6 | -12.2% |
| 32 tensor | 496.4 | 488.0 | -1.7% |

## Standard host matrix

| Case | Before (ns) | After (ns) | Time change |
|---|---:|---:|---:|
| auto map | 166.4 | 167.8 | +0.8% |
| reduced key | 162.2 | 160.4 | -1.1% |
| verify off | 164.4 | 167.7 | +2.0% |
| verify on | 163.4 | 163.2 | -0.1% |
| baked | 150.6 | 146.1 | -3.0% |
| bound tensor | 162.6 | 163.8 | +0.7% |
| bound pointer | 160.7 | 157.4 | -2.1% |
| fixed device map | 155.1 | 154.0 | -0.7% |
| fixed device no-map | 152.9 | 153.5 | +0.4% |

## Standard GPU matrix

These values include the driver launch call. The changes here are small compared
with variation between runs; there is no demonstrated GPU launch speedup.

| Case | Before (us) | After (us) | Time change |
|---|---:|---:|---:|
| auto map | 3.188 | 3.196 | +0.3% |
| reduced key | 3.184 | 3.203 | +0.6% |
| verify off | 3.212 | 3.186 | -0.8% |
| verify on | 3.210 | 3.218 | +0.3% |
| baked | 3.179 | 3.184 | +0.2% |
| bound tensor | 3.204 | 3.242 | +1.2% |
| bound pointer | 3.199 | 3.239 | +1.2% |
| fixed device map | 3.215 | 3.245 | +0.9% |
| fixed device no-map | 3.225 | 3.247 | +0.7% |

## README comparison

All values below are nanoseconds. Triton is unchanged and serves as a timing
reference; its changes between columns reflect run variation. The README mode
prints launch results rounded to 0.01 microseconds.

| Case | Before (ns) | After (ns) | Time change |
|---|---:|---:|---:|
| grid=(1,) triton | 16950.0 | 16740.0 | -1.2% |
| grid=(1,) intj | 3180.0 | 3220.0 | +1.3% |
| grid=(0,) triton | 13270.0 | 13170.0 | -0.8% |
| grid=(0,) intj | 150.0 | 150.0 | +0.0% |
| shim decode | 121.3 | 119.4 | -1.6% |
| cxx decode | 116.2 | 115.7 | -0.4% |
| cpython decode | 904.4 | 895.5 | -1.0% |

## Reproduce

```sh
PYTHONPATH=$PWD taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --sweep --iters 100000 --batches 9
PYTHONPATH=$PWD taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --no-gpu --iters 100000 --batches 9
PYTHONPATH=$PWD taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --iters 20000 --batches 7
PYTHONPATH=$PWD taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --readme --iters 20000 --batches 7
```

For a baseline comparison, use this benchmark script with `PYTHONPATH` pointing
to an archive of the baseline revision, and a separate `TRITON_HOME` for each
revision. Repeat five times, reversing the revision order on alternate rounds.

The local run artifacts are in `/tmp/intj-optimize-7g_as4_o/`: all 40 raw result
files (`{mode}-{before,after}-{0..4}.txt`), `results.json`, the source archive,
`run_comparison.py`, and verification logs. Temporary artifacts are local to this
machine; the tables above retain the results in the repository.

## Validation

`PYTHONPATH=$PWD /tmp/gb2/bin/python -m pytest tests -q -rs`: **525 passed,
1 skipped**. The skipped cache microbenchmark requires Google Benchmark, which
is unavailable. Existing sparse/nested Torch warnings remain.

`pyright --pythonpath /tmp/gb2/bin/python`: **0 errors**. The new sweep smoke test
and grid overflow regressions were each checked failing before their fixes.
CUDA is compile-checked by the suite; NVIDIA GPU runtime behavior remains untested.

## TVM FFI comparison — 2026-09-25

Measured at `00087e3` on an Intel Xeon Platinum 8480C (CPU 0 pinned) and
AMD Instinct MI308X (gfx942). Python 3.12.3, Torch 2.14.0+rocm7.2, Triton
3.8.0, and apache-tvm-ffi 0.1.14.post2.dev1+g424558557.d20260924.
Each number is the median of three separate process medians, in **µs per host
call**; each process used nine batches of 1,000 calls. The benchmark warms each
case, prepares its argument tuple, compiles the launchers, and converts FFI
tensors before timing. The timer stops before GPU synchronization, so kernel
rows include host enqueue and possible queue backpressure, not GPU completion.
The FFI no-op rows make no GPU launch.

The FFI kernels are compiled HIP C++ kernels; the INTJ kernels are Triton
kernels. FFI receives preconverted TensorViews while INTJ receives Torch
tensors, so these are the actual call paths, not identical GPU binaries. In
the zero-argument sweep, FFI selects the stream using a constant device type
and ID; other FFI kernel rows read it from a tensor.

### Default: three tensor arguments

| Case | µs/call |
|---|---:|
| FFI packed nop | 0.1451 |
| FFI typed nop | 0.1490 |
| FFI empty kernel | 3.3306 |
| INTJ empty kernel | 2.9967 |
| INTJ fixed-device kernel | 2.8683 |

### Mixed: three tensors, integer, float

| Case | µs/call |
|---|---:|
| FFI packed nop mixed | 0.1740 |
| FFI typed nop mixed | 0.1778 |
| FFI mixed kernel | 3.2923 |
| INTJ mixed kernel | 2.8943 |
| INTJ fixed mixed kernel | 2.8973 |

### Argument-count sweep

For each count, the first three arguments are tensors; later arguments repeat
integer, float, tensor. INTJ uses device-bound launchers with CXX or SHIM
tensor access and one value-independent specialization key per count.

| Arguments | FFI packed nop | FFI typed nop | FFI empty kernel | INTJ CXX kernel | INTJ SHIM kernel |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.1174 | 0.1204 | 1.6480 | 2.7306 | 2.7129 |
| 3 | 0.1436 | 0.1491 | 3.2107 | 2.9134 | 2.8924 |
| 5 | 0.1737 | 0.1772 | 3.2947 | 2.8931 | 2.8570 |
| 8 | 0.2061 | 0.2096 | 3.3981 | 3.0216 | 2.8900 |
| 16 | 0.2931 | 0.3039 | 3.6678 | 3.3526 | 3.2411 |
| 32 | 0.4778 | 0.4941 | 4.2349 | 3.6359 | 3.4357 |
| 64 | 0.8366 | 0.8686 | 5.0639 | 4.6199 | 4.5667 |

A preliminary five-batch run at 10,000 calls per batch gave wide launch
variation within a single case (for example, 32-argument FFI kernel:
4.28–15.01 µs/call). These tables use 1,000 calls per batch and should not be
compared directly to the 20,000-call historical launcher measurements above.
The zero-argument FFI kernel also varied across processes: 1.5929–1.8904
µs/call (process medians).

Repeat each command three times from the repository root:

```sh
PYTHONPATH=$PWD taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_ffi_compare.py --iters 1000 --batches 9
PYTHONPATH=$PWD taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_ffi_compare.py --sweep --iters 1000 --batches 9
```

Raw process outputs are local at `/tmp/intj-tvm-ffi-20260925.ryY7sD/`:
`{default,sweep}-{2,3,4}.log` are the reported runs; `*-1.log` are the
10,000-call diagnostics.

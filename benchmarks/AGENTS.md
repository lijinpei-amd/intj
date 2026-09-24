# Benchmarking

Run benchmarks from the repository root, using an environment with the project's
Torch and Triton dependencies. Pin one available CPU (`taskset -c 0` on this
machine), run one benchmark at a time on an idle GPU, warm the workload,
keep compilation and argument construction outside the timer, and repeat in
separate processes. Use the same timing boundary, call
count, hardware, and software when comparing revisions. Validate kernel output
before timing when the benchmark writes data.

## Available benchmarks

- `bench_launch.py`: INTJ launcher matrix (`--no-gpu` for host-only decoding and
  cache work; default for GPU launches), `--readme` for Triton versus INTJ, and
  `--sweep` for 4, 16, and 32 integer or tensor arguments.
- `bench_ffi_compare.py`: cached INTJ versus preconverted TVM FFI no-ops and GPU
  kernel launches; `--sweep` measures 0, 3, 5, 8, 16, 32, and 64 arguments.
  Requires `apache-tvm-ffi` and a ROCm or CUDA GPU.
- `../tests/bench_kernel_cache.cpp`, run by `../tests/test_kernel_cache.py`:
  direct cache hits, misses, and hashes for 1-, 2-, and 5-word keys and 1, 8,
  64, or 512 entries. Requires Google Benchmark (`INTJ_BENCHMARK_ROOT` if it is
  not installed system-wide); pytest skips this benchmark when unavailable.

For example, with the project interpreter active:

```sh
PYTHONPATH=$PWD taskset -c 0 python benchmarks/bench_launch.py --no-gpu --iters 100000 --batches 9
PYTHONPATH=$PWD taskset -c 0 python benchmarks/bench_launch.py --readme --iters 20000 --batches 7
PYTHONPATH=$PWD taskset -c 0 python benchmarks/bench_ffi_compare.py --iters 1000 --batches 9
PYTHONPATH=$PWD taskset -c 0 python benchmarks/bench_ffi_compare.py --sweep --iters 1000 --batches 9
PYTHONPATH=$PWD python -m pytest tests/test_kernel_cache.py -s
```

## Timing caveats

- The Python benchmarks warm each case, call `fn(*prebuilt_args)` in the timed
  loop, stop the host timer, then synchronize the GPU when relevant. Report
  host call/enqueue time, not GPU completion. A long batch can fill a GPU queue and change the
  measured enqueue time; the 2026-09-25 FFI run documents this effect.
- FFI no-ops do not launch a GPU kernel. FFI receives preconverted TensorViews,
  while INTJ receives Torch tensors; their GPU kernels are different binaries.
  In the zero-argument FFI sweep, stream selection also differs from rows with
  tensor arguments. Label these paths instead of claiming equivalent kernels.
- The 2026-09-24 optimization used an older runner that timed a Python wrapper.
  Commit `30803c4` changed it to direct calls with prebuilt tuples. Its old
  numbers and newer launcher numbers have different timing boundaries.
- CUDA runtime cannot be tested on this development machine; mark it untested
  when reporting GPU results. A skipped cache benchmark supplies no timing data.

## Result journals

`journals/` contains benchmark result history. Name each Markdown record
`YYYY-MM-DD_<topic>_<index>_<commit>.md`, using the local measurement date, a
short kebab-case topic, a zero-based repeat-round index for that topic, and
the measured source commit's short Git SHA.
Record the exact command (or the saved driver and its flags), raw output
(including per-batch samples), CPU/core affinity, GPU/architecture, Python and
dependency versions, relevant
compiler/cache settings, number of calls and batches, and timing caveats.
A round may contain several serial commands (modes or revisions); record each.
For a two-revision comparison, name the file after the candidate commit and
identify both revisions inside. If source was uncommitted while measured,
record that fact and a source hash; never imply a clean checkout from a filename.
Keep aggregate results in the first reported repeat record and link the other
repeats. Diagnostics with different parameters get their own indexed record.

The 2026-09-24 optimization history starts at
`journals/2026-09-24_launcher-optimization_0_4f0b13e.md`; the 2026-09-25 TVM FFI comparison starts
at `journals/2026-09-25_tvm-ffi_1_00087e3.md` (index 0 is the long-batch diagnostic).

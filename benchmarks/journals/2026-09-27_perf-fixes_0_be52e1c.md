# Perf fixes from the 2026-09-27 regression report

Three commits on top of `b4e9f1f`, each measured against its parent with interleaved
A/B (one process per revision per round, alternating), then the whole suite at the
last one:

| commit | change |
|---|---|
| `84fd60a` | plain memo load/store on GIL builds; `Py_RETURN_NONE` after the unlock |
| `9dacc19` | kernel-cache benchmark built with `-falign-loops=64`; production flags unchanged |
| `be52e1c` | slot shrink: no `full` word, no stored hash at 1-2 words (32/40/72 B slots) |

Every tree was a clean detached worktree of its commit, except `pf1a` (below), which is
`84fd60a` plus one uncommitted line (`"-falign-loops=64"` appended to `_build_flags`'s
`ccflags`). The suite ran on the clean `be52e1c` checkout.

## Environment

- CPU: Intel Xeon Platinum 8480C (224 logical CPUs), `taskset -c 0`, one benchmark
  process at a time. Shared host: 1-min load 2-23 during these runs (other users),
  logged per process in the raw files and by `uptime` around each driver.
- GPU: AMD Instinct MI308X (`gfx942`), `HIP_VISIBLE_DEVICES=0`; GPU 0 idle otherwise.
- Python 3.12.3 (`/tmp/gb2/bin/python`), Torch `2.14.0+rocm7.2`, Triton `3.8.0`,
  g++ 13.3.0. Rendered modules built by triton at `-O3`; `kernel_cache` binaries by
  `tests/test_kernel_cache.py::_build` (`g++ -O3 -falign-loops=64` from `9dacc19`).
- `INTJ_BENCHMARK_ROOT=/tmp/gbench`. Each A/B tree has its own `TRITON_HOME`.
- CUDA: untested (no NVIDIA GPU).

## Drivers

- Host counters: `/tmp/intj-regress-tools/ab.sh <group> <rounds> <out> <treeA> <treeB>`
  runs `hc.py` (the benchmark rows' kernels and arguments, user-space cycles/instructions
  via `perf_event_open`) per tree; groups `hot` and `ffi` at 1000 x 9, `host` at 100000 x 9.
  Values are medians over processes of each process's median.
- `bench_launch.py --no-gpu --iters 100000 --batches 9` and
  `bench_intj_ffi_paths.py --mode all --iters 1000 --batches 9`, interleaved, 4 rounds.
- Kernel cache: the test's `_build` via `/tmp/intj-pf/kc/build.py` (header directory
  prepended to the include path to pick the revision), run with
  `--benchmark_format=json --benchmark_min_time=0.05s` on core 0, interleaved, 5-9 rounds.
- Suite: `HIP_VISIBLE_DEVICES=0 /tmp/intj-bench/run_all.sh perf` (10 cases x 3 rounds,
  all `exit_status=0`), compared with `/tmp/gb2/bin/python /tmp/intj-bench/bench_compare.py
  {baseline,task5} perf`.

## 1. Memo without atomics, `Py_RETURN_NONE` (`b4e9f1f` -> `84fd60a`)

| row | b4e9f1f ns / cyc / ins | 84fd60a ns / cyc / ins |
|---|---|---|
| ffi hot call (10 rounds) | 40.3 / 123.9 / 671.3 | 39.5 / 121.7 / 667.3 |
| ffi hot call (6 rounds) | 40.8 / 125.6 / 671.3 | 39.4 / 122.0 / 667.4 |
| ffi 3-tensor nop | 35.2 / 108.5 / 636.5 | 34.7 / 107.4 / 633.5 |
| ffi direct positional | 66.2 / 198.6 / 1188.5 | 65.7 / 197.7 / 1185.5 |
| ffi pair direct | 67.7 / 203.0 / 1224.4 | 66.8 / 200.4 / 1220.4 |
| host auto map | 42.4 / 122.7 / 787.3 | 42.2 / 122.2 / 784.3 |
| host bound tensor | 47.0 / 135.9 / 832.8 | 45.5 / 131.8 / 828.8 |

-3 to -4 instructions on every row. The hot call is back to the pre-regression
`cab358b` numbers in the report (121.7 cycles, 666.7 instructions).

## 2. `-falign-loops=64`

Benchmark (miss loops are identical source in the report's C and D binaries, so C-D
should be ~0; 5 rounds, ns):

| nw | row | default C / D | loops C / D | functions+loops C / D |
|---|---|---|---|---|
| 1 | miss/8 | 3.24 / 2.33 | 2.51 / 2.50 | 2.52 / 2.52 |
| 1 | miss/512 | 3.84 / 2.75 | 3.25 / 3.18 | 3.31 / 3.29 |
| 5 | miss/1 | 1.42 / 2.03 | 1.69 / 1.67 | 1.68 / 1.70 |
| 5 | miss/8 | 2.79 / 3.86 | 3.07 / 3.20 | 3.11 / 3.17 |
| 5 | miss/512 | 3.23 / 4.09 | 3.38 / 3.41 | 3.39 / 3.41 |

`-falign-loops=64` alone removes the up-to-1.1 ns spread, as well as adding
`-falign-functions=64` does, so the test uses only the loop flag.

Production (`84fd60a` vs `pf1a` = same + flag in `_build_flags`):

| row | 84fd60a ns / cyc / ins | + -falign-loops=64 |
|---|---|---|
| ffi hot call | 39.7 / 122.5 / 667.3 | 38.3 / 118.9 / 667.3 |
| ffi 3-tensor nop | 34.7 / 107.2 / 633.5 | 34.2 / 106.3 / 633.5 |
| ffi direct positional | 65.6 / 197.1 / 1185.5 | 66.0 / 198.9 / 1185.5 |
| host auto map | 42.3 / 122.2 / 784.3 | 42.1 / 121.8 / 784.3 |
| host verify on | 43.0 / 124.4 / 828.8 | 43.5 / 125.9 / 828.8 |
| host bound pointer | 43.7 / 126.6 / 800.8 | 44.4 / 128.6 / 800.8 |
| host fixed device no-map | 43.7 / 126.5 / 780.8 | 44.4 / 128.3 / 780.8 |
| bench_launch --no-gpu auto map | 42.3 | 43.0 |
| bench_launch --no-gpu fixed device no-map | 43.2 | 44.1 |

Instruction counts are identical on every row: the hit path runs no loop, so the flag
only moves code. Cycles move -3.6 to +2 either way and the per-process spread is not
tighter. `.so` text grows 160-430 bytes (~1%). gcc 13.3, Ubuntu clang 22.1 and ROCm
clang 22.0 all accept the flag, and it would reach both `ModuleKey.build_flags` and
triton's `_compile_so` cache key through `ccflags`. **Not adopted** for production
(comment in `_build_flags`).

## 3. Slot shrink (`9dacc19` -> `be52e1c`)

Slot sizes, `intj_kernel` value (24 B): 1 word 40 -> 32, 2 words 56 -> 40, 5 words
80 -> 72; 2-word child map with a record pointer 40 -> 24. Kernel cache, 9 interleaved
rounds, both built with `-falign-loops=64`, ns:

| nw | row | 9dacc19 | be52e1c |
|---|---|---:|---:|
| 1 | hit/1 | 1.59 | 1.56 |
| 1 | hit/64 | 1.72 | 1.67 |
| 1 | hit/512 | 2.35 | 2.30 |
| 1 | miss/512 | 3.18 | 2.93 |
| 2 | hit/1 | 2.06 | 1.86 |
| 2 | hit/8 | 2.06 | 1.83 |
| 2 | hit/512 | 3.12 | 3.02 |
| 2 | miss/64 | 4.41 | 3.14 |
| 5 | hit/512 | 4.67 | 4.54 |
| all | hit_child (2-word lookup) | 3.46 | 3.01 |

Variants tried and rejected (`slot.summ`, `slot2.summ`): keeping the hash at 2 words
(hits +0.2 ns); dropping it at 5 words (misses -0.8 ns, hit/512 +0.15 ns); 16-bit
`block_dim`/`nparams` (1-word slot 24 B, 2-word 32 B): hit/512 -0.1 ns but 1-64-key
hits +0.05 to +0.18 ns. The 1-word hit/512 gain is 0.05 ns, not the report's
estimated 0.4 ns.

Launch path (memo hits; `84fd60a` vs `be52e1c`; `9dacc19` has the same `intj/`
render):

| row | 84fd60a ns / cyc / ins | be52e1c |
|---|---|---|
| ffi hot call | 39.7 / 123.5 / 667.3 | 39.3 / 121.2 / 667.3 |
| ffi direct positional | 66.0 / 198.8 / 1185.5 | 65.4 / 196.4 / 1184.5 |
| host verify off | 42.1 / 122.0 / 792.8 | 43.2 / 125.1 / 791.8 |
| host bound pointer | 44.3 / 128.2 / 800.8 | 45.6 / 132.1 / 799.8 |
| host fixed device map | 43.4 / 125.4 / 778.8 | 42.3 / 122.2 / 776.8 |
| bench_launch --no-gpu verify off | 42.2 | 43.0 |
| bench_launch --no-gpu bound pointer | 43.8 | 44.5 |
| ffi_paths hot call | 38.1 | 37.4 |

0 to -2 instructions; cycles -3 to +4 by row. `verify off` and `bound pointer` are
slower in both drivers, with fewer instructions and a code path the change does not
touch on a memo hit. The same size of move appears with the production flag change in
section 2, where no instruction changed, so these rows are most likely code layout.

## Suite at `be52e1c`

`bench_compare.py baseline perf` and `task5 perf` each flag the same 6 rows: all
`ffi_compare` GPU kernel rows, TVM FFI (+10-11%) and INTJ (+14-22%). Rounds 0-1 were
in the slow per-process GPU-submission mode, round 2 was not (INTJ empty kernel
3.49 / 3.65 / 2.99 us). Interleaved A/B (5 processes each, `fc.txt`): INTJ empty kernel
`b4e9f1f` 3.035 us, `be52e1c` 2.965; FFI empty 3.342 / 3.315. `b4e9f1f` had 2 slow
processes and `be52e1c` none. This is the noise the report's section 5 describes.

vs task5: `ffi_paths` hot call 39.2 -> 38.0 ns, adapter kwargs 110.0 -> 107.0,
`launch_host` rows -0.7 to +0.9 ns. `kernel_cache` rows are not comparable to task5:
the build flags changed. `hit_child` 2.46 -> 3.00 comes from that; under equal flags
it went 3.46 -> 3.01.

## Raw output

### hot1.txt (summary, then raw)
```
row                                                          b4e9f1f                               pf1
ffi hot call                         ns=  40.3 cyc= 123.9 ins= 671.3   ns=  39.5 cyc= 121.7 ins= 667.3
ffi hot call x10                     ns=  41.2 cyc= 118.6 ins= 684.9   ns=  39.5 cyc= 113.8 ins= 680.9
n per cell: {'b4e9f1f': 10, 'pf1': 10}
## round=1 commit=b4e9f1f load=7.64 9.28 10.76 t=12:10:26 gpu_use=0,0,0,0,0,0,0,0,
ROW ffi hot call                 ns=   40.34 cyc=   124.1 ins=   672.7 bmiss= 0.044
ROW ffi hot call x10             ns=   41.15 cyc=   118.9 ins=   684.9 bmiss= 0.004
## round=1 commit=pf1 load=7.64 9.28 10.76 t=12:10:30 gpu_use=0,0,0,0,0,0,0,0,
ROW ffi hot call                 ns=   38.77 cyc=   119.7 ins=   665.6 bmiss= 0.043
ROW ffi hot call x10             ns=   39.22 cyc=   114.0 ins=   680.9 bmiss= 0.003
## round=2 commit=b4e9f1f load=7.43 9.21 10.73 t=12:10:35 gpu_use=0,0,0,0,0,0,0,0,
ROW ffi hot call                 ns=   40.08 cyc=   123.4 ins=   671.3 bmiss= 0.046
ROW ffi hot call x10             ns=   40.57 cyc=   117.5 ins=   684.9 bmiss= 0.003
## round=2 commit=pf1 load=24.37 12.70 11.85 t=12:10:38 gpu_use=0,0,0,0,0,0,0,0,
ROW ffi hot call                 ns=   39.27 cyc=   120.9 ins=   667.3 bmiss= 0.048
ROW ffi hot call x10             ns=   39.16 cyc=   113.6 ins=   680.9 bmiss= 0.003
## round=3 commit=b4e9f1f load=30.67 14.19 12.34 t=12:10:42 gpu_use=0,0,0,0,0,0,0,0,
ROW ffi hot call                 ns=   41.16 cyc=   126.7 ins=   671.3 bmiss= 0.042
ROW ffi hot call x10             ns=   41.41 cyc=   120.8 ins=   684.9 bmiss= 0.003
## round=3 commit=pf1 load=30.67 14.19 12.34 t=12:10:45 gpu_use=0,0,0,0,0,0,0,0,
ROW ffi hot call                 ns=   39.15 cyc=   120.4 ins=   667.3 bmiss= 0.042
ROW ffi hot call x10             ns=   39.38 cyc=   114.9 ins=   680.9 bmiss= 0.004
## round=4 commit=b4e9f1f load=28.69 14.06 12.30 t=12:10:48 gpu_use=0,0,0,0,0,0,0,0,
ROW ffi hot call                 ns=   40.25 cyc=   123.7 ins=   672.7 bmiss= 0.040
ROW ffi hot call x10             ns=   41.23 cyc=   117.8 ins=   685.0 bmiss= 0.004
## round=4 commit=pf1 load=26.95 13.94 12.27 t=12:10:51 gpu_use=0,0,0,0,0,0,0,0,
ROW ffi hot call                 ns=   39.67 cyc=   122.1 ins=   668.7 bmiss= 0.043
ROW ffi hot call x10             ns=   39.29 cyc=   112.6 ins=   680.9 bmiss= 0.003
## round=5 commit=b4e9f1f load=26.95 13.94 12.27 t=12:10:55 gpu_use=0,0,0,0,0,0,0,0,
ROW ffi hot call                 ns=   40.33 cyc=   124.6 ins=   671.3 bmiss= 0.048
ROW ffi hot call x10             ns=   41.21 cyc=   118.3 ins=   684.9 bmiss= 0.003
## round=5 commit=pf1 load=25.12 13.78 12.23 t=12:10:58 gpu_use=0,0,0,0,0,0,0,0,
ROW ffi hot call                 ns=   39.72 cyc=   122.3 ins=   667.4 bmiss= 0.043
ROW ffi hot call x10             ns=   41.24 cyc=   120.3 ins=   681.0 bmiss= 0.003
## round=6 commit=b4e9f1f load=23.74 13.68 12.20 t=12:11:01 gpu_use=0,0,0,0,0,0,0,0,
ROW ffi hot call                 ns=   39.93 cyc=   122.7 ins=   671.5 bmiss= 0.049
ROW ffi hot call x10             ns=   41.26 cyc=   120.2 ins=   685.0 bmiss= 0.003
## round=6 commit=pf1 load=23.74 13.68 12.20 t=12:11:05 gpu_use=0,0,0,0,0,0,0,0,
ROW ffi hot call                 ns=   39.64 cyc=   122.3 ins=   667.3 bmiss= 0.045
ROW ffi hot call x10             ns=   39.90 cyc=   113.7 ins=   680.9 bmiss= 0.004
## round=7 commit=b4e9f1f load=22.16 13.52 12.16 t=12:11:08 gpu_use=0,0,0,0,0,0,0,0,
ROW ffi hot call                 ns=   41.46 cyc=   127.1 ins=   672.6 bmiss= 0.043
ROW ffi hot call x10             ns=   41.84 cyc=   120.7 ins=   684.9 bmiss= 0.004
## round=7 commit=pf1 load=35.76 16.48 13.13 t=12:11:12 gpu_use=0,0,0,0,0,0,0,0,
ROW ffi hot call                 ns=   39.35 cyc=   121.0 ins=   667.3 bmiss= 0.045
ROW ffi hot call x10             ns=   39.02 cyc=   113.5 ins=   680.9 bmiss= 0.003
## round=8 commit=b4e9f1f load=35.76 16.48 13.13 t=12:11:15 gpu_use=0,0,0,0,0,0,0,0,
ROW ffi hot call                 ns=   40.32 cyc=   123.8 ins=   671.3 bmiss= 0.044
ROW ffi hot call x10             ns=   40.55 cyc=   118.0 ins=   684.9 bmiss= 0.003
## round=8 commit=pf1 load=51.24 20.01 14.29 t=12:11:18 gpu_use=0,0,0,0,0,0,0,0,
ROW ffi hot call                 ns=   39.96 cyc=   123.7 ins=   667.4 bmiss= 0.051
ROW ffi hot call x10             ns=   39.56 cyc=   115.5 ins=   681.0 bmiss= 0.004
## round=9 commit=b4e9f1f load=47.78 19.81 14.25 t=12:11:21 gpu_use=0,0,0,0,0,0,0,0,
ROW ffi hot call                 ns=   40.35 cyc=   124.0 ins=   671.3 bmiss= 0.046
ROW ffi hot call x10             ns=   41.57 cyc=   119.8 ins=   684.9 bmiss= 0.003
## round=9 commit=pf1 load=47.78 19.81 14.25 t=12:11:25 gpu_use=0,0,0,0,0,0,0,0,
ROW ffi hot call                 ns=   39.42 cyc=   121.2 ins=   667.3 bmiss= 0.049
ROW ffi hot call x10             ns=   40.56 cyc=   118.4 ins=   680.9 bmiss= 0.003
## round=10 commit=b4e9f1f load=44.27 19.55 14.20 t=12:11:28 gpu_use=0,0,0,0,0,0,0,0,
ROW ffi hot call                 ns=   40.00 cyc=   123.3 ins=   671.3 bmiss= 0.051
ROW ffi hot call x10             ns=   40.21 cyc=   117.1 ins=   684.9 bmiss= 0.004
## round=10 commit=pf1 load=58.42 22.89 15.31 t=12:11:32 gpu_use=0,0,0,0,0,0,0,0,
ROW ffi hot call                 ns=   40.27 cyc=   124.3 ins=   668.7 bmiss= 0.040
ROW ffi hot call x10             ns=   39.92 cyc=   113.3 ins=   680.9 bmiss= 0.004
```

### ffi1.txt (summary, then raw)
```
row                                                          b4e9f1f                               pf1
ffi hot call                         ns=  40.8 cyc= 125.6 ins= 671.3   ns=  39.4 cyc= 122.0 ins= 667.4
ffi 3-tensor nop                     ns=  35.2 cyc= 108.5 ins= 636.5   ns=  34.7 cyc= 107.4 ins= 633.5
ffi direct positional                ns=  66.2 cyc= 198.6 ins=1188.5   ns=  65.7 cyc= 197.7 ins=1185.5
ffi direct positional (no lambda)    ns=  34.1 cyc= 106.0 ins= 640.3   ns=  34.0 cyc= 105.7 ins= 637.3
ffi pair direct                      ns=  67.7 cyc= 203.0 ins=1224.4   ns=  66.8 cyc= 200.4 ins=1220.4
ffi pair prebuilt *tuple             ns= 141.6 cyc= 416.6 ins=2187.2   ns= 139.5 cyc= 411.4 ins=2181.2
ffi pair (no lambda)                 ns=  31.9 cyc=  99.6 ins= 600.3   ns=  31.5 cyc=  98.5 ins= 596.3
n per cell: {'b4e9f1f': 6, 'pf1': 6}
## round=1 commit=b4e9f1f load=58.42 22.89 15.31 t=12:11:35 gpu_use=3,3,3,3,3,3,3,3,
ROW ffi hot call                 ns=   41.25 cyc=   127.0 ins=   671.3 bmiss= 0.044
ROW ffi 3-tensor nop             ns=   35.09 cyc=   107.6 ins=   636.5 bmiss= 0.040
ROW ffi direct positional        ns=   66.88 cyc=   200.8 ins=  1188.5 bmiss= 0.036
ROW ffi direct positional (no lambda) ns=   34.30 cyc=   106.4 ins=   640.3 bmiss= 0.030
ROW ffi pair direct              ns=   67.46 cyc=   202.5 ins=  1224.4 bmiss= 0.031
ROW ffi pair prebuilt *tuple     ns=  140.82 cyc=   415.0 ins=  2189.1 bmiss= 0.041
ROW ffi pair (no lambda)         ns=   31.92 cyc=    99.6 ins=   600.3 bmiss= 0.033
## round=1 commit=pf1 load=55.50 22.88 15.34 t=12:11:38 gpu_use=0,0,0,0,0,0,0,0,
ROW ffi hot call                 ns=   39.42 cyc=   122.7 ins=   667.3 bmiss= 0.043
ROW ffi 3-tensor nop             ns=   34.99 cyc=   108.5 ins=   633.5 bmiss= 0.032
ROW ffi direct positional        ns=   66.79 cyc=   200.7 ins=  1185.5 bmiss= 0.036
ROW ffi direct positional (no lambda) ns=   33.99 cyc=   105.6 ins=   637.3 bmiss= 0.031
ROW ffi pair direct              ns=   68.33 cyc=   205.1 ins=  1220.4 bmiss= 0.034
ROW ffi pair prebuilt *tuple     ns=  139.17 cyc=   410.5 ins=  2185.1 bmiss= 0.033
ROW ffi pair (no lambda)         ns=   31.59 cyc=    98.7 ins=   596.3 bmiss= 0.035
## round=2 commit=b4e9f1f load=52.60 23.33 15.57 t=12:11:46 gpu_use=1,0,0,0,1,1,3,1,
ROW ffi hot call                 ns=   40.86 cyc=   125.9 ins=   671.4 bmiss= 0.053
ROW ffi 3-tensor nop             ns=   34.82 cyc=   108.0 ins=   636.5 bmiss= 0.035
ROW ffi direct positional        ns=   66.31 cyc=   199.6 ins=  1188.5 bmiss= 0.035
ROW ffi direct positional (no lambda) ns=   34.12 cyc=   106.0 ins=   640.3 bmiss= 0.031
ROW ffi pair direct              ns=   68.38 cyc=   204.0 ins=  1224.4 bmiss= 0.037
ROW ffi pair prebuilt *tuple     ns=  143.41 cyc=   421.2 ins=  2185.2 bmiss= 0.036
ROW ffi pair (no lambda)         ns=   31.96 cyc=    99.8 ins=   600.3 bmiss= 0.033
## round=2 commit=pf1 load=52.60 23.33 15.57 t=12:11:50 gpu_use=2,2,6,2,2,2,1,1,
ROW ffi hot call                 ns=   39.16 cyc=   122.2 ins=   667.4 bmiss= 0.051
ROW ffi 3-tensor nop             ns=   34.55 cyc=   107.2 ins=   633.5 bmiss= 0.034
ROW ffi direct positional        ns=   65.72 cyc=   198.3 ins=  1185.6 bmiss= 0.035
ROW ffi direct positional (no lambda) ns=   34.01 cyc=   106.0 ins=   637.4 bmiss= 0.034
ROW ffi pair direct              ns=   67.04 cyc=   201.2 ins=  1220.5 bmiss= 0.031
ROW ffi pair prebuilt *tuple     ns=  141.82 cyc=   418.1 ins=  2181.2 bmiss= 0.040
ROW ffi pair (no lambda)         ns=   31.53 cyc=    98.6 ins=   596.4 bmiss= 0.032
## round=3 commit=b4e9f1f load=49.59 23.19 15.57 t=12:11:53 gpu_use=1,1,1,52,1,1,3,1,
ROW ffi hot call                 ns=   40.83 cyc=   125.3 ins=   671.3 bmiss= 0.047
ROW ffi 3-tensor nop             ns=   35.22 cyc=   109.0 ins=   636.5 bmiss= 0.034
ROW ffi direct positional        ns=   65.65 cyc=   197.8 ins=  1188.5 bmiss= 0.039
ROW ffi direct positional (no lambda) ns=   34.21 cyc=   106.4 ins=   640.3 bmiss= 0.032
ROW ffi pair direct              ns=   67.12 cyc=   201.6 ins=  1224.4 bmiss= 0.034
ROW ffi pair prebuilt *tuple     ns=  141.60 cyc=   416.0 ins=  2189.1 bmiss= 0.033
ROW ffi pair (no lambda)         ns=   31.87 cyc=    99.6 ins=   600.3 bmiss= 0.034
## round=3 commit=pf1 load=48.18 23.33 15.66 t=12:11:57 gpu_use=0,0,0,0,0,98,0,0,
ROW ffi hot call                 ns=   39.30 cyc=   121.3 ins=   667.4 bmiss= 0.044
ROW ffi 3-tensor nop             ns=   34.21 cyc=   106.1 ins=   633.5 bmiss= 0.031
ROW ffi direct positional        ns=   65.65 cyc=   197.1 ins=  1185.6 bmiss= 0.036
ROW ffi direct positional (no lambda) ns=   33.90 cyc=   105.4 ins=   637.4 bmiss= 0.031
ROW ffi pair direct              ns=   66.47 cyc=   199.5 ins=  1220.4 bmiss= 0.030
ROW ffi pair prebuilt *tuple     ns=  141.05 cyc=   414.3 ins=  2181.2 bmiss= 0.039
ROW ffi pair (no lambda)         ns=   31.43 cyc=    98.3 ins=   596.4 bmiss= 0.033
## round=4 commit=b4e9f1f load=48.18 23.33 15.66 t=12:12:00 gpu_use=0,0,0,0,0,81,0,0,
ROW ffi hot call                 ns=   39.12 cyc=   120.3 ins=   672.7 bmiss= 0.037
ROW ffi 3-tensor nop             ns=   35.78 cyc=   110.9 ins=   636.5 bmiss= 0.032
ROW ffi direct positional        ns=   66.19 cyc=   198.6 ins=  1188.5 bmiss= 0.034
ROW ffi direct positional (no lambda) ns=   34.14 cyc=   106.1 ins=   640.3 bmiss= 0.031
ROW ffi pair direct              ns=   67.88 cyc=   203.5 ins=  1224.4 bmiss= 0.031
ROW ffi pair prebuilt *tuple     ns=  140.33 cyc=   410.7 ins=  2172.3 bmiss= 0.030
ROW ffi pair (no lambda)         ns=   31.94 cyc=    99.6 ins=   600.3 bmiss= 0.032
## round=4 commit=pf1 load=45.44 23.18 15.65 t=12:12:03 gpu_use=0,1,0,1,9,0,1,0,
ROW ffi hot call                 ns=   39.26 cyc=   121.8 ins=   668.7 bmiss= 0.041
ROW ffi 3-tensor nop             ns=   36.00 cyc=   108.6 ins=   633.5 bmiss= 0.034
ROW ffi direct positional        ns=   65.52 cyc=   196.8 ins=  1185.5 bmiss= 0.031
ROW ffi direct positional (no lambda) ns=   33.94 cyc=   106.1 ins=   637.3 bmiss= 0.031
ROW ffi pair direct              ns=   66.55 cyc=   199.9 ins=  1220.4 bmiss= 0.028
ROW ffi pair prebuilt *tuple     ns=  139.91 cyc=   412.4 ins=  2185.3 bmiss= 0.036
ROW ffi pair (no lambda)         ns=   31.63 cyc=    98.8 ins=   596.3 bmiss= 0.033
## round=5 commit=b4e9f1f load=43.32 23.11 15.67 t=12:12:07 gpu_use=1,6,1,1,7,2,1,1,
ROW ffi hot call                 ns=   41.06 cyc=   126.1 ins=   672.8 bmiss= 0.039
ROW ffi 3-tensor nop             ns=   34.75 cyc=   107.5 ins=   636.5 bmiss= 0.034
ROW ffi direct positional        ns=   66.18 cyc=   198.7 ins=  1188.6 bmiss= 0.037
ROW ffi direct positional (no lambda) ns=   34.07 cyc=   105.9 ins=   640.4 bmiss= 0.032
ROW ffi pair direct              ns=   67.63 cyc=   202.9 ins=  1224.4 bmiss= 0.030
ROW ffi pair prebuilt *tuple     ns=  141.59 cyc=   417.3 ins=  2185.4 bmiss= 0.038
ROW ffi pair (no lambda)         ns=   31.95 cyc=    99.7 ins=   600.4 bmiss= 0.034
## round=5 commit=pf1 load=43.32 23.11 15.67 t=12:12:10 gpu_use=0,0,0,0,0,1,0,0,
ROW ffi hot call                 ns=   39.53 cyc=   121.8 ins=   667.3 bmiss= 0.040
ROW ffi 3-tensor nop             ns=   34.58 cyc=   107.3 ins=   633.5 bmiss= 0.031
ROW ffi direct positional        ns=   66.17 cyc=   200.1 ins=  1185.5 bmiss= 0.036
ROW ffi direct positional (no lambda) ns=   33.91 cyc=   105.6 ins=   637.3 bmiss= 0.030
ROW ffi pair direct              ns=   66.80 cyc=   200.5 ins=  1220.4 bmiss= 0.037
ROW ffi pair prebuilt *tuple     ns=  137.58 cyc=   406.0 ins=  2168.3 bmiss= 0.034
ROW ffi pair (no lambda)         ns=   31.53 cyc=    98.5 ins=   596.3 bmiss= 0.033
## round=6 commit=b4e9f1f load=41.62 23.09 15.70 t=12:12:14 gpu_use=2,17,2,1,2,2,1,2,
ROW ffi hot call                 ns=   39.88 cyc=   123.8 ins=   671.3 bmiss= 0.045
ROW ffi 3-tensor nop             ns=   35.27 cyc=   109.5 ins=   636.5 bmiss= 0.034
ROW ffi direct positional        ns=   66.15 cyc=   198.5 ins=  1188.5 bmiss= 0.034
ROW ffi direct positional (no lambda) ns=   34.11 cyc=   105.9 ins=   640.3 bmiss= 0.031
ROW ffi pair direct              ns=   67.67 cyc=   203.1 ins=  1224.4 bmiss= 0.030
ROW ffi pair prebuilt *tuple     ns=  142.87 cyc=   419.4 ins=  2189.2 bmiss= 0.037
ROW ffi pair (no lambda)         ns=   31.94 cyc=    99.5 ins=   600.3 bmiss= 0.033
## round=6 commit=pf1 load=57.50 26.69 16.90 t=12:12:17 gpu_use=1,0,0,1,2,1,0,0,
ROW ffi hot call                 ns=   40.07 cyc=   123.2 ins=   667.4 bmiss= 0.046
ROW ffi 3-tensor nop             ns=   34.80 cyc=   107.5 ins=   633.6 bmiss= 0.035
ROW ffi direct positional        ns=   65.40 cyc=   196.8 ins=  1185.6 bmiss= 0.034
ROW ffi direct positional (no lambda) ns=   34.05 cyc=   105.7 ins=   637.4 bmiss= 0.033
ROW ffi pair direct              ns=   66.76 cyc=   200.4 ins=  1220.4 bmiss= 0.028
ROW ffi pair prebuilt *tuple     ns=  138.26 cyc=   407.8 ins=  2168.4 bmiss= 0.031
ROW ffi pair (no lambda)         ns=   31.51 cyc=    98.4 ins=   596.4 bmiss= 0.031
```

### host1.txt (summary, then raw)
```
row                                                          b4e9f1f                               pf1
host auto map                        ns=  42.4 cyc= 122.7 ins= 787.3   ns=  42.2 cyc= 122.2 ins= 784.3
host reduced key                     ns=  41.8 cyc= 120.9 ins= 789.8   ns=  41.7 cyc= 120.7 ins= 788.8
host verify off                      ns=  42.1 cyc= 121.8 ins= 794.8   ns=  41.9 cyc= 121.6 ins= 792.8
host verify on                       ns=  44.0 cyc= 127.5 ins= 831.8   ns=  43.1 cyc= 124.8 ins= 828.8
host baked                           ns=  39.6 cyc= 114.7 ins= 732.8   ns=  39.4 cyc= 114.0 ins= 728.8
host bound tensor                    ns=  47.0 cyc= 135.9 ins= 832.8   ns=  45.5 cyc= 131.8 ins= 828.8
host bound pointer                   ns=  44.1 cyc= 127.7 ins= 804.8   ns=  44.3 cyc= 128.3 ins= 800.8
host fixed device map                ns=  43.0 cyc= 124.7 ins= 779.8   ns=  43.6 cyc= 126.1 ins= 778.8
host fixed device no-map             ns=  44.4 cyc= 128.4 ins= 783.8   ns=  43.7 cyc= 126.6 ins= 780.8
n per cell: {'b4e9f1f': 5, 'pf1': 5}
## round=1 commit=b4e9f1f load=38.84 28.53 18.53 t=12:14:00 gpu_use=0,0,0,0,0,0,0,0,
ROW host auto map                ns=   43.08 cyc=   124.6 ins=   787.3 bmiss= 0.001
ROW host reduced key             ns=   42.35 cyc=   122.4 ins=   789.8 bmiss= 0.000
ROW host verify off              ns=   42.05 cyc=   121.6 ins=   794.8 bmiss= 0.000
ROW host verify on               ns=   43.58 cyc=   126.2 ins=   831.8 bmiss= 0.000
ROW host baked                   ns=   39.85 cyc=   115.3 ins=   732.8 bmiss= 0.000
ROW host bound tensor            ns=   47.43 cyc=   137.1 ins=   832.8 bmiss= 0.000
ROW host bound pointer           ns=   44.38 cyc=   128.2 ins=   804.8 bmiss= 0.000
ROW host fixed device map        ns=   43.41 cyc=   125.6 ins=   779.8 bmiss= 0.000
ROW host fixed device no-map     ns=   44.48 cyc=   128.5 ins=   783.8 bmiss= 0.000
## round=1 commit=pf1 load=36.21 28.15 18.47 t=12:14:04 gpu_use=0,0,0,0,0,0,93,0,
ROW host auto map                ns=   41.65 cyc=   120.2 ins=   775.8 bmiss= 0.000
ROW host reduced key             ns=   41.43 cyc=   119.7 ins=   788.8 bmiss= 0.000
ROW host verify off              ns=   41.77 cyc=   120.9 ins=   792.8 bmiss= 0.000
ROW host verify on               ns=   42.89 cyc=   124.0 ins=   828.8 bmiss= 0.000
ROW host baked                   ns=   39.39 cyc=   114.0 ins=   728.8 bmiss= 0.000
ROW host bound tensor            ns=   46.01 cyc=   133.2 ins=   828.8 bmiss= 0.000
ROW host bound pointer           ns=   45.39 cyc=   131.1 ins=   800.8 bmiss= 0.000
ROW host fixed device map        ns=   43.80 cyc=   126.9 ins=   778.8 bmiss= 0.001
ROW host fixed device no-map     ns=   43.16 cyc=   125.0 ins=   780.8 bmiss= 0.000
## round=2 commit=b4e9f1f load=26.06 26.41 18.14 t=12:14:27 gpu_use=0,0,0,0,0,0,0,0,
ROW host auto map                ns=   42.66 cyc=   123.5 ins=   787.3 bmiss= 0.001
ROW host reduced key             ns=   41.80 cyc=   120.9 ins=   789.8 bmiss= 0.000
ROW host verify off              ns=   41.92 cyc=   121.4 ins=   794.8 bmiss= 0.000
ROW host verify on               ns=   44.20 cyc=   127.9 ins=   831.8 bmiss= 0.001
ROW host baked                   ns=   39.58 cyc=   114.7 ins=   732.8 bmiss= 0.000
ROW host bound tensor            ns=   45.87 cyc=   132.9 ins=   832.8 bmiss= 0.000
ROW host bound pointer           ns=   44.06 cyc=   127.6 ins=   804.8 bmiss= 0.000
ROW host fixed device map        ns=   42.62 cyc=   123.5 ins=   779.8 bmiss= 0.000
ROW host fixed device no-map     ns=   44.29 cyc=   128.2 ins=   783.8 bmiss= 0.001
## round=2 commit=pf1 load=26.06 26.41 18.14 t=12:14:30 gpu_use=0,0,0,0,0,0,0,0,
ROW host auto map                ns=   43.27 cyc=   125.1 ins=   784.3 bmiss= 0.001
ROW host reduced key             ns=   41.94 cyc=   121.3 ins=   788.8 bmiss= 0.001
ROW host verify off              ns=   41.71 cyc=   120.8 ins=   792.8 bmiss= 0.000
ROW host verify on               ns=   43.91 cyc=   125.9 ins=   828.8 bmiss= 0.000
ROW host baked                   ns=   40.23 cyc=   116.5 ins=   728.8 bmiss= 0.000
ROW host bound tensor            ns=   45.70 cyc=   132.5 ins=   828.8 bmiss= 0.001
ROW host bound pointer           ns=   44.34 cyc=   128.3 ins=   800.8 bmiss= 0.000
ROW host fixed device map        ns=   43.17 cyc=   125.1 ins=   778.8 bmiss= 0.000
ROW host fixed device no-map     ns=   43.71 cyc=   126.6 ins=   780.8 bmiss= 0.000
## round=3 commit=b4e9f1f load=24.69 26.12 18.09 t=12:14:34 gpu_use=0,0,0,0,0,0,0,0,
ROW host auto map                ns=   42.15 cyc=   121.9 ins=   787.3 bmiss= 0.001
ROW host reduced key             ns=   42.04 cyc=   121.7 ins=   789.8 bmiss= 0.000
ROW host verify off              ns=   42.51 cyc=   122.9 ins=   794.8 bmiss= 0.000
ROW host verify on               ns=   43.27 cyc=   125.3 ins=   831.8 bmiss= 0.000
ROW host baked                   ns=   39.98 cyc=   115.7 ins=   732.8 bmiss= 0.000
ROW host bound tensor            ns=   46.97 cyc=   135.3 ins=   832.8 bmiss= 0.001
ROW host bound pointer           ns=   43.94 cyc=   127.2 ins=   804.8 bmiss= 0.000
ROW host fixed device map        ns=   44.14 cyc=   127.8 ins=   779.8 bmiss= 0.000
ROW host fixed device no-map     ns=   44.20 cyc=   127.8 ins=   783.8 bmiss= 0.000
## round=3 commit=pf1 load=23.35 25.82 18.04 t=12:14:37 gpu_use=0,0,0,0,0,0,0,0,
ROW host auto map                ns=   43.48 cyc=   125.8 ins=   784.3 bmiss= 0.001
ROW host reduced key             ns=   41.74 cyc=   121.0 ins=   788.8 bmiss= 0.000
ROW host verify off              ns=   41.93 cyc=   121.6 ins=   792.8 bmiss= 0.000
ROW host verify on               ns=   43.35 cyc=   125.5 ins=   828.8 bmiss= 0.000
ROW host baked                   ns=   39.13 cyc=   113.3 ins=   728.8 bmiss= 0.000
ROW host bound tensor            ns=   44.86 cyc=   129.8 ins=   828.8 bmiss= 0.000
ROW host bound pointer           ns=   44.06 cyc=   127.5 ins=   800.8 bmiss= 0.000
ROW host fixed device map        ns=   43.58 cyc=   126.1 ins=   778.8 bmiss= 0.000
ROW host fixed device no-map     ns=   44.16 cyc=   127.9 ins=   780.8 bmiss= 0.000
## round=4 commit=b4e9f1f load=23.35 25.82 18.04 t=12:14:40 gpu_use=0,0,0,0,0,0,0,0,
ROW host auto map                ns=   42.41 cyc=   122.7 ins=   787.3 bmiss= 0.001
ROW host reduced key             ns=   41.67 cyc=   120.5 ins=   789.8 bmiss= 0.000
ROW host verify off              ns=   42.43 cyc=   122.9 ins=   794.8 bmiss= 0.000
ROW host verify on               ns=   44.01 cyc=   127.5 ins=   831.8 bmiss= 0.000
ROW host baked                   ns=   39.06 cyc=   113.0 ins=   732.8 bmiss= 0.000
ROW host bound tensor            ns=   49.23 cyc=   140.5 ins=   832.8 bmiss= 0.001
ROW host bound pointer           ns=   44.41 cyc=   128.7 ins=   804.8 bmiss= 0.000
ROW host fixed device map        ns=   43.02 cyc=   124.7 ins=   779.8 bmiss= 0.000
ROW host fixed device no-map     ns=   44.98 cyc=   130.3 ins=   783.8 bmiss= 0.000
## round=4 commit=pf1 load=22.12 25.52 17.99 t=12:14:44 gpu_use=0,0,0,0,0,0,0,0,
ROW host auto map                ns=   42.14 cyc=   121.8 ins=   784.3 bmiss= 0.001
ROW host reduced key             ns=   41.70 cyc=   120.7 ins=   788.8 bmiss= 0.000
ROW host verify off              ns=   70.92 cyc=   205.0 ins=   792.8 bmiss= 0.001
ROW host verify on               ns=   43.11 cyc=   124.8 ins=   828.8 bmiss= 0.000
ROW host baked                   ns=   39.27 cyc=   113.8 ins=   728.8 bmiss= 0.000
ROW host bound tensor            ns=   44.91 cyc=   130.0 ins=   828.8 bmiss= 0.000
ROW host bound pointer           ns=   44.03 cyc=   127.6 ins=   800.8 bmiss= 0.000
ROW host fixed device map        ns=   42.98 cyc=   124.5 ins=   778.8 bmiss= 0.000
ROW host fixed device no-map     ns=   43.92 cyc=   127.3 ins=   780.8 bmiss= 0.001
## round=5 commit=b4e9f1f load=20.91 25.21 17.93 t=12:14:47 gpu_use=0,0,0,0,0,0,0,0,
ROW host auto map                ns=   42.28 cyc=   122.2 ins=   787.3 bmiss= 0.001
ROW host reduced key             ns=   41.82 cyc=   120.6 ins=   789.8 bmiss= 0.000
ROW host verify off              ns=   42.07 cyc=   121.8 ins=   794.8 bmiss= 0.000
ROW host verify on               ns=   44.62 cyc=   129.2 ins=   831.8 bmiss= 0.000
ROW host baked                   ns=   39.62 cyc=   114.7 ins=   732.8 bmiss= 0.000
ROW host bound tensor            ns=   46.98 cyc=   135.9 ins=   832.8 bmiss= 0.000
ROW host bound pointer           ns=   44.08 cyc=   127.7 ins=   804.8 bmiss= 0.000
ROW host fixed device map        ns=   42.76 cyc=   123.7 ins=   779.8 bmiss= 0.000
ROW host fixed device no-map     ns=   44.36 cyc=   128.4 ins=   783.8 bmiss= 0.000
## round=5 commit=pf1 load=20.91 25.21 17.93 t=12:14:51 gpu_use=0,0,0,0,0,0,0,0,
ROW host auto map                ns=   42.22 cyc=   122.2 ins=   784.3 bmiss= 0.001
ROW host reduced key             ns=   41.32 cyc=   119.7 ins=   788.8 bmiss= 0.000
ROW host verify off              ns=   42.31 cyc=   121.9 ins=   792.8 bmiss= 0.001
ROW host verify on               ns=   42.79 cyc=   124.0 ins=   828.8 bmiss= 0.000
ROW host baked                   ns=   39.57 cyc=   114.6 ins=   728.8 bmiss= 0.000
ROW host bound tensor            ns=   45.51 cyc=   131.8 ins=   828.8 bmiss= 0.000
ROW host bound pointer           ns=   44.38 cyc=   128.5 ins=   800.8 bmiss= 0.000
ROW host fixed device map        ns=   44.02 cyc=   127.5 ins=   778.8 bmiss= 0.000
ROW host fixed device no-map     ns=   43.55 cyc=   126.1 ins=   780.8 bmiss= 0.001
```

### hot2.txt (summary, then raw)
```
row                                                              pf1                              pf1a
ffi hot call                         ns=  39.7 cyc= 122.5 ins= 667.3   ns=  38.3 cyc= 118.9 ins= 667.3
ffi hot call x10                     ns=  39.9 cyc= 114.6 ins= 680.9   ns=  38.5 cyc= 111.8 ins= 680.9
n per cell: {'pf1': 10, 'pf1a': 10}
## round=1 commit=pf1 load=22.65 16.55 15.64 t=12:22:08 gpu_use=0,0,0,0,0,0,0,3,
ROW ffi hot call                 ns=   39.15 cyc=   122.5 ins=   667.3 bmiss= 0.042
ROW ffi hot call x10             ns=   40.44 cyc=   116.5 ins=   680.9 bmiss= 0.003
## round=1 commit=pf1a load=21.39 16.39 15.59 t=12:22:11 gpu_use=0,0,0,0,0,0,0,1,
ROW ffi hot call                 ns=   38.21 cyc=   118.8 ins=   665.4 bmiss= 0.035
ROW ffi hot call x10             ns=   37.51 cyc=   109.5 ins=   680.9 bmiss= 0.003
## round=2 commit=pf1 load=20.40 16.27 15.56 t=12:22:17 gpu_use=0,0,0,0,0,0,0,6,
ROW ffi hot call                 ns=   39.40 cyc=   121.0 ins=   667.3 bmiss= 0.044
ROW ffi hot call x10             ns=   39.28 cyc=   113.0 ins=   680.9 bmiss= 0.003
## round=2 commit=pf1a load=20.40 16.27 15.56 t=12:22:20 gpu_use=0,0,0,0,0,0,0,1,
ROW ffi hot call                 ns=   40.99 cyc=   126.1 ins=   667.4 bmiss= 0.050
ROW ffi hot call x10             ns=   41.72 cyc=   121.3 ins=   681.0 bmiss= 0.004
## round=3 commit=pf1 load=19.57 16.17 15.53 t=12:22:23 gpu_use=0,0,0,0,0,0,0,1,
ROW ffi hot call                 ns=   39.24 cyc=   121.8 ins=   667.3 bmiss= 0.048
ROW ffi hot call x10             ns=   39.90 cyc=   112.7 ins=   680.9 bmiss= 0.003
## round=3 commit=pf1a load=18.88 16.08 15.50 t=12:22:27 gpu_use=0,0,0,0,0,0,0,0,
ROW ffi hot call                 ns=   38.03 cyc=   117.9 ins=   667.6 bmiss= 0.044
ROW ffi hot call x10             ns=   38.39 cyc=   110.6 ins=   681.0 bmiss= 0.003
## round=4 commit=pf1 load=18.88 16.08 15.50 t=12:22:30 gpu_use=0,0,0,0,0,0,0,0,
ROW ffi hot call                 ns=   40.22 cyc=   123.7 ins=   667.3 bmiss= 0.042
ROW ffi hot call x10             ns=   39.90 cyc=   114.4 ins=   680.9 bmiss= 0.003
## round=4 commit=pf1a load=30.98 18.64 16.33 t=12:22:33 gpu_use=0,0,0,0,0,0,0,0,
ROW ffi hot call                 ns=   38.65 cyc=   119.0 ins=   667.3 bmiss= 0.052
ROW ffi hot call x10             ns=   38.21 cyc=   111.2 ins=   680.9 bmiss= 0.004
## round=5 commit=pf1 load=28.98 18.43 16.28 t=12:22:37 gpu_use=0,0,0,0,0,0,0,0,
ROW ffi hot call                 ns=   39.72 cyc=   122.4 ins=   667.4 bmiss= 0.052
ROW ffi hot call x10             ns=   39.69 cyc=   114.8 ins=   681.0 bmiss= 0.003
## round=5 commit=pf1a load=28.98 18.43 16.28 t=12:22:40 gpu_use=0,0,0,0,0,0,0,0,
ROW ffi hot call                 ns=   37.48 cyc=   115.6 ins=   667.5 bmiss= 0.045
ROW ffi hot call x10             ns=   38.77 cyc=   112.5 ins=   681.0 bmiss= 0.003
## round=6 commit=pf1 load=27.06 18.20 16.21 t=12:22:43 gpu_use=0,0,0,0,0,0,0,0,
ROW ffi hot call                 ns=   39.15 cyc=   120.7 ins=   667.6 bmiss= 0.048
ROW ffi hot call x10             ns=   39.88 cyc=   114.4 ins=   681.0 bmiss= 0.003
## round=6 commit=pf1a load=25.29 17.98 16.15 t=12:22:46 gpu_use=0,0,0,0,0,0,0,0,
ROW ffi hot call                 ns=   38.45 cyc=   119.9 ins=   667.3 bmiss= 0.047
ROW ffi hot call x10             ns=   38.54 cyc=   112.4 ins=   680.9 bmiss= 0.003
## round=7 commit=pf1 load=25.29 17.98 16.15 t=12:22:49 gpu_use=0,0,0,0,0,0,0,0,
ROW ffi hot call                 ns=   40.13 cyc=   124.4 ins=   667.3 bmiss= 0.040
ROW ffi hot call x10             ns=   39.24 cyc=   113.5 ins=   680.9 bmiss= 0.003
## round=7 commit=pf1a load=23.67 17.77 16.09 t=12:22:53 gpu_use=0,0,0,0,0,0,0,0,
ROW ffi hot call                 ns=   37.79 cyc=   116.3 ins=   667.3 bmiss= 0.044
ROW ffi hot call x10             ns=   38.32 cyc=   109.7 ins=   680.9 bmiss= 0.003
## round=8 commit=pf1 load=23.67 17.77 16.09 t=12:22:56 gpu_use=0,0,0,0,0,0,0,0,
ROW ffi hot call                 ns=   39.95 cyc=   122.9 ins=   668.7 bmiss= 0.040
ROW ffi hot call x10             ns=   39.91 cyc=   116.5 ins=   680.9 bmiss= 0.004
## round=8 commit=pf1a load=22.17 17.56 16.03 t=12:22:59 gpu_use=0,0,0,0,0,0,0,0,
ROW ffi hot call                 ns=   37.54 cyc=   116.3 ins=   667.6 bmiss= 0.050
ROW ffi hot call x10             ns=   38.10 cyc=   110.6 ins=   681.0 bmiss= 0.004
## round=9 commit=pf1 load=20.96 17.38 15.98 t=12:23:02 gpu_use=0,0,0,0,0,0,0,0,
ROW ffi hot call                 ns=   40.18 cyc=   126.2 ins=   667.3 bmiss= 0.045
ROW ffi hot call x10             ns=   39.88 cyc=   116.4 ins=   680.9 bmiss= 0.003
## round=9 commit=pf1a load=20.96 17.38 15.98 t=12:23:06 gpu_use=0,0,0,0,0,0,0,0,
ROW ffi hot call                 ns=   38.99 cyc=   120.1 ins=   667.3 bmiss= 0.045
ROW ffi hot call x10             ns=   38.54 cyc=   112.4 ins=   680.9 bmiss= 0.003
## round=10 commit=pf1 load=19.68 17.17 15.93 t=12:23:09 gpu_use=0,0,0,0,0,0,0,0,
ROW ffi hot call                 ns=   39.66 cyc=   122.0 ins=   667.3 bmiss= 0.041
ROW ffi hot call x10             ns=   39.87 cyc=   115.4 ins=   680.9 bmiss= 0.003
## round=10 commit=pf1a load=18.58 16.99 15.87 t=12:23:12 gpu_use=0,0,0,0,0,0,0,0,
ROW ffi hot call                 ns=   38.88 cyc=   120.1 ins=   667.3 bmiss= 0.051
ROW ffi hot call x10             ns=   39.77 cyc=   115.5 ins=   680.9 bmiss= 0.003
```

### ffi2.txt (summary, then raw)
```
row                                                              pf1                              pf1a
ffi hot call                         ns=  40.0 cyc= 122.9 ins= 667.4   ns=  38.3 cyc= 118.6 ins= 667.3
ffi 3-tensor nop                     ns=  34.7 cyc= 107.2 ins= 633.5   ns=  34.2 cyc= 106.3 ins= 633.5
ffi direct positional                ns=  65.6 cyc= 197.1 ins=1185.5   ns=  66.0 cyc= 198.9 ins=1185.5
ffi direct positional (no lambda)    ns=  34.0 cyc= 105.9 ins= 637.3   ns=  34.1 cyc= 106.0 ins= 637.3
ffi pair direct                      ns=  67.0 cyc= 201.1 ins=1220.4   ns=  67.1 cyc= 201.4 ins=1220.4
ffi pair prebuilt *tuple             ns= 139.8 cyc= 411.1 ins=2181.1   ns= 147.4 cyc= 433.2 ins=2168.4
ffi pair (no lambda)                 ns=  31.5 cyc=  98.4 ins= 596.3   ns=  31.6 cyc=  98.8 ins= 596.3
n per cell: {'pf1': 6, 'pf1a': 6}
## round=1 commit=pf1 load=18.58 16.99 15.87 t=12:23:15 gpu_use=0,0,0,0,0,0,0,0,
ROW ffi hot call                 ns=   40.52 cyc=   124.8 ins=   667.3 bmiss= 0.046
ROW ffi 3-tensor nop             ns=   36.10 cyc=   109.9 ins=   633.5 bmiss= 0.037
ROW ffi direct positional        ns=   65.68 cyc=   197.2 ins=  1185.5 bmiss= 0.036
ROW ffi direct positional (no lambda) ns=   34.00 cyc=   105.9 ins=   637.3 bmiss= 0.031
ROW ffi pair direct              ns=   67.10 cyc=   201.4 ins=  1220.4 bmiss= 0.029
ROW ffi pair prebuilt *tuple     ns=  139.80 cyc=   412.0 ins=  2181.1 bmiss= 0.038
ROW ffi pair (no lambda)         ns=   31.50 cyc=    98.3 ins=   596.3 bmiss= 0.032
## round=1 commit=pf1a load=17.50 16.79 15.81 t=12:23:19 gpu_use=0,0,0,0,0,0,0,0,
ROW ffi hot call                 ns=   38.44 cyc=   118.4 ins=   667.3 bmiss= 0.049
ROW ffi 3-tensor nop             ns=   34.25 cyc=   106.4 ins=   633.4 bmiss= 0.032
ROW ffi direct positional        ns=   66.13 cyc=   200.4 ins=  1185.5 bmiss= 0.033
ROW ffi direct positional (no lambda) ns=   34.18 cyc=   106.0 ins=   637.3 bmiss= 0.033
ROW ffi pair direct              ns=   66.78 cyc=   200.4 ins=  1220.4 bmiss= 0.036
ROW ffi pair prebuilt *tuple     ns=  147.57 cyc=   435.1 ins=  2185.1 bmiss= 0.037
ROW ffi pair (no lambda)         ns=   31.62 cyc=    98.7 ins=   596.3 bmiss= 0.031
## round=2 commit=pf1 load=16.58 16.61 15.76 t=12:23:26 gpu_use=0,0,0,0,0,0,0,0,
ROW ffi hot call                 ns=   39.34 cyc=   121.5 ins=   668.7 bmiss= 0.044
ROW ffi 3-tensor nop             ns=   34.69 cyc=   107.8 ins=   633.5 bmiss= 0.036
ROW ffi direct positional        ns=   65.54 cyc=   197.0 ins=  1185.6 bmiss= 0.035
ROW ffi direct positional (no lambda) ns=   34.15 cyc=   105.9 ins=   637.4 bmiss= 0.032
ROW ffi pair direct              ns=   66.85 cyc=   200.7 ins=  1220.4 bmiss= 0.030
ROW ffi pair prebuilt *tuple     ns=  140.88 cyc=   408.1 ins=  2168.4 bmiss= 0.034
ROW ffi pair (no lambda)         ns=   31.57 cyc=    98.7 ins=   596.4 bmiss= 0.031
## round=2 commit=pf1a load=15.89 16.47 15.72 t=12:23:29 gpu_use=0,0,0,0,0,0,0,0,
ROW ffi hot call                 ns=   38.25 cyc=   118.8 ins=   667.5 bmiss= 0.053
ROW ffi 3-tensor nop             ns=   34.18 cyc=   106.3 ins=   633.6 bmiss= 0.039
ROW ffi direct positional        ns=   65.74 cyc=   197.4 ins=  1185.6 bmiss= 0.032
ROW ffi direct positional (no lambda) ns=   34.20 cyc=   106.1 ins=   637.5 bmiss= 0.031
ROW ffi pair direct              ns=   67.59 cyc=   201.7 ins=  1220.5 bmiss= 0.035
ROW ffi pair prebuilt *tuple     ns=  147.03 cyc=   433.3 ins=  2168.4 bmiss= 0.028
ROW ffi pair (no lambda)         ns=   31.71 cyc=    98.9 ins=   596.4 bmiss= 0.036
## round=3 commit=pf1 load=15.02 16.28 15.66 t=12:23:33 gpu_use=0,0,0,0,0,0,0,0,
ROW ffi hot call                 ns=   40.04 cyc=   123.1 ins=   667.5 bmiss= 0.045
ROW ffi 3-tensor nop             ns=   35.35 cyc=   109.5 ins=   633.6 bmiss= 0.040
ROW ffi direct positional        ns=   65.63 cyc=   197.3 ins=  1185.6 bmiss= 0.033
ROW ffi direct positional (no lambda) ns=   34.00 cyc=   105.6 ins=   637.5 bmiss= 0.035
ROW ffi pair direct              ns=   66.97 cyc=   201.1 ins=  1220.5 bmiss= 0.029
ROW ffi pair prebuilt *tuple     ns=  139.07 cyc=   410.2 ins=  2185.3 bmiss= 0.036
ROW ffi pair (no lambda)         ns=   31.39 cyc=    98.1 ins=   596.5 bmiss= 0.031
## round=3 commit=pf1a load=15.02 16.28 15.66 t=12:23:36 gpu_use=0,0,0,0,0,0,0,0,
ROW ffi hot call                 ns=   38.12 cyc=   117.8 ins=   667.3 bmiss= 0.049
ROW ffi 3-tensor nop             ns=   34.43 cyc=   106.7 ins=   633.5 bmiss= 0.038
ROW ffi direct positional        ns=   66.23 cyc=   198.8 ins=  1185.5 bmiss= 0.035
ROW ffi direct positional (no lambda) ns=   34.14 cyc=   106.1 ins=   637.3 bmiss= 0.030
ROW ffi pair direct              ns=   66.89 cyc=   200.7 ins=  1220.4 bmiss= 0.036
ROW ffi pair prebuilt *tuple     ns=  147.85 cyc=   433.2 ins=  2168.3 bmiss= 0.030
ROW ffi pair (no lambda)         ns=   31.62 cyc=    98.9 ins=   596.3 bmiss= 0.033
## round=4 commit=pf1 load=14.21 16.09 15.60 t=12:23:39 gpu_use=0,0,0,0,0,0,10,0,
ROW ffi hot call                 ns=   40.31 cyc=   124.3 ins=   668.7 bmiss= 0.043
ROW ffi 3-tensor nop             ns=   34.18 cyc=   106.0 ins=   633.6 bmiss= 0.035
ROW ffi direct positional        ns=   65.39 cyc=   197.0 ins=  1185.6 bmiss= 0.032
ROW ffi direct positional (no lambda) ns=   33.96 cyc=   105.9 ins=   637.4 bmiss= 0.031
ROW ffi pair direct              ns=   66.58 cyc=   199.9 ins=  1220.4 bmiss= 0.036
ROW ffi pair prebuilt *tuple     ns=  139.78 cyc=   412.3 ins=  2181.4 bmiss= 0.034
ROW ffi pair (no lambda)         ns=   31.57 cyc=    98.5 ins=   596.4 bmiss= 0.032
## round=4 commit=pf1a load=13.56 15.92 15.55 t=12:23:42 gpu_use=0,0,0,0,0,0,0,0,
ROW ffi hot call                 ns=   38.61 cyc=   119.0 ins=   667.3 bmiss= 0.045
ROW ffi 3-tensor nop             ns=   34.19 cyc=   106.1 ins=   633.5 bmiss= 0.031
ROW ffi direct positional        ns=   66.23 cyc=   199.4 ins=  1185.5 bmiss= 0.035
ROW ffi direct positional (no lambda) ns=   34.08 cyc=   105.8 ins=   637.3 bmiss= 0.032
ROW ffi pair direct              ns=   67.31 cyc=   201.9 ins=  1220.4 bmiss= 0.038
ROW ffi pair prebuilt *tuple     ns=  147.18 cyc=   431.1 ins=  2168.3 bmiss= 0.031
ROW ffi pair (no lambda)         ns=   31.69 cyc=    98.7 ins=   596.3 bmiss= 0.035
## round=5 commit=pf1 load=13.56 15.92 15.55 t=12:23:46 gpu_use=0,0,0,0,0,0,0,0,
ROW ffi hot call                 ns=   39.64 cyc=   122.5 ins=   667.3 bmiss= 0.045
ROW ffi 3-tensor nop             ns=   34.30 cyc=   106.4 ins=   633.5 bmiss= 0.033
ROW ffi direct positional        ns=   65.93 cyc=   198.3 ins=  1185.5 bmiss= 0.032
ROW ffi direct positional (no lambda) ns=   33.92 cyc=   105.4 ins=   637.3 bmiss= 0.033
ROW ffi pair direct              ns=   67.03 cyc=   201.3 ins=  1220.4 bmiss= 0.030
ROW ffi pair prebuilt *tuple     ns=  140.95 cyc=   415.6 ins=  2181.2 bmiss= 0.037
ROW ffi pair (no lambda)         ns=   31.54 cyc=    98.4 ins=   596.3 bmiss= 0.034
## round=5 commit=pf1a load=12.87 15.74 15.49 t=12:23:49 gpu_use=0,0,0,0,0,0,0,0,
ROW ffi hot call                 ns=   38.67 cyc=   119.6 ins=   667.3 bmiss= 0.046
ROW ffi 3-tensor nop             ns=   34.36 cyc=   106.6 ins=   633.5 bmiss= 0.035
ROW ffi direct positional        ns=   65.45 cyc=   197.1 ins=  1185.5 bmiss= 0.032
ROW ffi direct positional (no lambda) ns=   34.01 cyc=   106.0 ins=   637.3 bmiss= 0.031
ROW ffi pair direct              ns=   66.97 cyc=   201.1 ins=  1220.4 bmiss= 0.035
ROW ffi pair prebuilt *tuple     ns=  145.20 cyc=   427.9 ins=  2168.3 bmiss= 0.029
ROW ffi pair (no lambda)         ns=   31.62 cyc=    99.0 ins=   596.3 bmiss= 0.035
## round=6 commit=pf1 load=12.40 15.59 15.45 t=12:23:52 gpu_use=0,0,0,0,0,0,0,0,
ROW ffi hot call                 ns=   39.89 cyc=   122.7 ins=   667.3 bmiss= 0.045
ROW ffi 3-tensor nop             ns=   34.79 cyc=   106.5 ins=   633.5 bmiss= 0.033
ROW ffi direct positional        ns=   65.48 cyc=   197.0 ins=  1185.5 bmiss= 0.035
ROW ffi direct positional (no lambda) ns=   34.15 cyc=   106.0 ins=   637.3 bmiss= 0.032
ROW ffi pair direct              ns=   67.01 cyc=   201.0 ins=  1220.4 bmiss= 0.031
ROW ffi pair prebuilt *tuple     ns=  138.06 cyc=   407.3 ins=  2168.3 bmiss= 0.030
ROW ffi pair (no lambda)         ns=   31.50 cyc=    98.4 ins=   596.3 bmiss= 0.033
## round=6 commit=pf1a load=12.40 15.59 15.45 t=12:23:55 gpu_use=0,0,0,0,0,0,0,0,
ROW ffi hot call                 ns=   38.08 cyc=   117.5 ins=   667.3 bmiss= 0.049
ROW ffi 3-tensor nop             ns=   34.19 cyc=   106.1 ins=   633.5 bmiss= 0.033
ROW ffi direct positional        ns=   65.91 cyc=   199.0 ins=  1185.5 bmiss= 0.040
ROW ffi direct positional (no lambda) ns=   33.95 cyc=   105.7 ins=   637.3 bmiss= 0.033
ROW ffi pair direct              ns=   82.63 cyc=   245.6 ins=  1220.4 bmiss= 0.037
ROW ffi pair prebuilt *tuple     ns=  149.18 cyc=   439.3 ins=  2181.1 bmiss= 0.038
ROW ffi pair (no lambda)         ns=   31.64 cyc=    98.8 ins=   596.3 bmiss= 0.036
```

### host2.txt (summary, then raw)
```
row                                                              pf1                              pf1a
host auto map                        ns=  42.3 cyc= 122.2 ins= 784.3   ns=  42.1 cyc= 121.8 ins= 784.3
host reduced key                     ns=  41.7 cyc= 120.7 ins= 788.8   ns=  41.6 cyc= 120.3 ins= 788.8
host verify off                      ns=  42.2 cyc= 122.1 ins= 792.8   ns=  42.3 cyc= 122.5 ins= 792.8
host verify on                       ns=  43.0 cyc= 124.4 ins= 828.8   ns=  43.5 cyc= 125.9 ins= 828.8
host baked                           ns=  39.5 cyc= 114.1 ins= 728.8   ns=  39.3 cyc= 113.4 ins= 728.8
host bound tensor                    ns=  45.5 cyc= 131.5 ins= 828.8   ns=  44.8 cyc= 129.6 ins= 828.8
host bound pointer                   ns=  43.7 cyc= 126.6 ins= 800.8   ns=  44.4 cyc= 128.6 ins= 800.8
host fixed device map                ns=  43.0 cyc= 124.5 ins= 778.8   ns=  43.4 cyc= 125.7 ins= 778.8
host fixed device no-map             ns=  43.7 cyc= 126.5 ins= 780.8   ns=  44.4 cyc= 128.3 ins= 780.8
n per cell: {'pf1': 6, 'pf1a': 6}
## round=1 commit=pf1 load=11.89 15.43 15.40 t=12:23:58 gpu_use=0,0,0,0,0,0,0,0,
ROW host auto map                ns=   42.20 cyc=   122.0 ins=   784.3 bmiss= 0.001
ROW host reduced key             ns=   41.98 cyc=   121.1 ins=   788.8 bmiss= 0.000
ROW host verify off              ns=   62.18 cyc=   179.9 ins=   792.8 bmiss= 0.000
ROW host verify on               ns=   42.94 cyc=   124.3 ins=   828.8 bmiss= 0.000
ROW host baked                   ns=   39.68 cyc=   114.3 ins=   728.8 bmiss= 0.000
ROW host bound tensor            ns=   45.53 cyc=   131.8 ins=   828.8 bmiss= 0.000
ROW host bound pointer           ns=   44.03 cyc=   127.4 ins=   800.8 bmiss= 0.000
ROW host fixed device map        ns=   42.91 cyc=   124.1 ins=   778.8 bmiss= 0.000
ROW host fixed device no-map     ns=   43.65 cyc=   126.2 ins=   780.8 bmiss= 0.000
## round=1 commit=pf1a load=11.34 15.26 15.34 t=12:24:02 gpu_use=0,0,0,0,0,0,0,0,
ROW host auto map                ns=   42.10 cyc=   121.5 ins=   775.8 bmiss= 0.000
ROW host reduced key             ns=   41.73 cyc=   120.6 ins=   788.8 bmiss= 0.000
ROW host verify off              ns=   42.20 cyc=   121.9 ins=   792.8 bmiss= 0.000
ROW host verify on               ns=   43.25 cyc=   124.9 ins=   828.8 bmiss= 0.000
ROW host baked                   ns=   39.36 cyc=   113.8 ins=   728.8 bmiss= 0.000
ROW host bound tensor            ns=   44.61 cyc=   129.2 ins=   828.8 bmiss= 0.000
ROW host bound pointer           ns=   44.92 cyc=   129.2 ins=   800.8 bmiss= 0.000
ROW host fixed device map        ns=   43.17 cyc=   124.4 ins=   778.8 bmiss= 0.000
ROW host fixed device no-map     ns=   44.15 cyc=   127.8 ins=   780.8 bmiss= 0.000
## round=2 commit=pf1 load=9.93 14.69 15.15 t=12:24:24 gpu_use=0,0,0,0,0,0,6,0,
ROW host auto map                ns=   42.35 cyc=   122.6 ins=   784.3 bmiss= 0.001
ROW host reduced key             ns=   41.76 cyc=   120.9 ins=   788.8 bmiss= 0.000
ROW host verify off              ns=   42.18 cyc=   122.0 ins=   792.8 bmiss= 0.000
ROW host verify on               ns=   43.47 cyc=   125.8 ins=   828.8 bmiss= 0.000
ROW host baked                   ns=   40.34 cyc=   114.1 ins=   728.8 bmiss= 0.000
ROW host bound tensor            ns=   45.69 cyc=   132.1 ins=   828.8 bmiss= 0.001
ROW host bound pointer           ns=   44.00 cyc=   127.3 ins=   800.8 bmiss= 0.000
ROW host fixed device map        ns=   43.79 cyc=   126.7 ins=   778.8 bmiss= 0.000
ROW host fixed device no-map     ns=   44.17 cyc=   127.7 ins=   780.8 bmiss= 0.000
## round=2 commit=pf1a load=9.45 14.51 15.09 t=12:24:27 gpu_use=0,0,0,0,0,0,0,0,
ROW host auto map                ns=   42.04 cyc=   121.7 ins=   784.3 bmiss= 0.001
ROW host reduced key             ns=   41.42 cyc=   120.0 ins=   788.8 bmiss= 0.000
ROW host verify off              ns=   42.90 cyc=   124.1 ins=   792.8 bmiss= 0.001
ROW host verify on               ns=   43.85 cyc=   126.9 ins=   828.8 bmiss= 0.001
ROW host baked                   ns=   39.32 cyc=   113.0 ins=   728.8 bmiss= 0.001
ROW host bound tensor            ns=   44.72 cyc=   129.4 ins=   828.8 bmiss= 0.000
ROW host bound pointer           ns=   44.71 cyc=   129.3 ins=   800.8 bmiss= 0.000
ROW host fixed device map        ns=   43.43 cyc=   125.6 ins=   778.8 bmiss= 0.000
ROW host fixed device no-map     ns=   44.62 cyc=   129.1 ins=   780.8 bmiss= 0.000
## round=3 commit=pf1 load=9.45 14.51 15.09 t=12:24:30 gpu_use=0,0,0,0,0,0,10,0,
ROW host auto map                ns=   42.03 cyc=   121.7 ins=   784.3 bmiss= 0.001
ROW host reduced key             ns=   41.82 cyc=   121.1 ins=   788.8 bmiss= 0.000
ROW host verify off              ns=   42.20 cyc=   122.2 ins=   792.8 bmiss= 0.000
ROW host verify on               ns=   42.91 cyc=   124.3 ins=   828.8 bmiss= 0.000
ROW host baked                   ns=   39.40 cyc=   114.0 ins=   728.8 bmiss= 0.000
ROW host bound tensor            ns=   44.92 cyc=   130.0 ins=   828.8 bmiss= 0.001
ROW host bound pointer           ns=   43.30 cyc=   125.4 ins=   800.8 bmiss= 0.000
ROW host fixed device map        ns=   43.00 cyc=   124.3 ins=   778.8 bmiss= 0.000
ROW host fixed device no-map     ns=   42.89 cyc=   124.2 ins=   780.8 bmiss= 0.000
## round=3 commit=pf1a load=8.93 14.32 15.03 t=12:24:33 gpu_use=0,0,0,0,0,0,0,0,
ROW host auto map                ns=   42.19 cyc=   121.9 ins=   784.3 bmiss= 0.001
ROW host reduced key             ns=   41.54 cyc=   120.3 ins=   788.8 bmiss= 0.000
ROW host verify off              ns=   42.39 cyc=   122.7 ins=   792.8 bmiss= 0.000
ROW host verify on               ns=   43.13 cyc=   125.0 ins=   828.8 bmiss= 0.001
ROW host baked                   ns=   39.06 cyc=   113.1 ins=   728.8 bmiss= 0.000
ROW host bound tensor            ns=   44.34 cyc=   128.5 ins=   828.8 bmiss= 0.000
ROW host bound pointer           ns=   44.34 cyc=   128.4 ins=   800.8 bmiss= 0.000
ROW host fixed device map        ns=   43.41 cyc=   125.7 ins=   778.8 bmiss= 0.001
ROW host fixed device no-map     ns=   44.38 cyc=   128.5 ins=   780.8 bmiss= 0.000
## round=4 commit=pf1 load=8.54 14.15 14.97 t=12:24:37 gpu_use=0,0,0,0,0,0,1,0,
ROW host auto map                ns=   42.84 cyc=   123.9 ins=   784.3 bmiss= 0.001
ROW host reduced key             ns=   41.55 cyc=   120.2 ins=   788.8 bmiss= 0.000
ROW host verify off              ns=   42.04 cyc=   121.7 ins=   792.8 bmiss= 0.000
ROW host verify on               ns=   43.29 cyc=   125.3 ins=   828.8 bmiss= 0.001
ROW host baked                   ns=   39.61 cyc=   114.7 ins=   728.8 bmiss= 0.000
ROW host bound tensor            ns=   45.78 cyc=   132.5 ins=   828.8 bmiss= 0.001
ROW host bound pointer           ns=   43.45 cyc=   125.8 ins=   800.8 bmiss= 0.000
ROW host fixed device map        ns=   42.96 cyc=   124.3 ins=   778.8 bmiss= 0.001
ROW host fixed device no-map     ns=   44.21 cyc=   127.9 ins=   780.8 bmiss= 0.000
## round=4 commit=pf1a load=8.54 14.15 14.97 t=12:24:40 gpu_use=0,0,0,0,0,0,0,0,
ROW host auto map                ns=   45.19 cyc=   130.7 ins=   784.3 bmiss= 0.001
ROW host reduced key             ns=   41.57 cyc=   120.3 ins=   788.8 bmiss= 0.000
ROW host verify off              ns=   42.19 cyc=   122.2 ins=   792.8 bmiss= 0.001
ROW host verify on               ns=   43.94 cyc=   126.9 ins=   828.8 bmiss= 0.000
ROW host baked                   ns=   39.29 cyc=   113.8 ins=   728.8 bmiss= 0.000
ROW host bound tensor            ns=   45.65 cyc=   132.2 ins=   828.8 bmiss= 0.000
ROW host bound pointer           ns=   44.42 cyc=   128.6 ins=   800.8 bmiss= 0.000
ROW host fixed device map        ns=   43.87 cyc=   126.9 ins=   778.8 bmiss= 0.000
ROW host fixed device no-map     ns=   44.33 cyc=   128.2 ins=   780.8 bmiss= 0.000
## round=5 commit=pf1 load=8.10 13.96 14.90 t=12:24:43 gpu_use=2,2,2,2,2,2,2,2,
ROW host auto map                ns=   42.37 cyc=   122.3 ins=   784.3 bmiss= 0.001
ROW host reduced key             ns=   41.59 cyc=   120.4 ins=   788.8 bmiss= 0.000
ROW host verify off              ns=   41.79 cyc=   120.8 ins=   792.8 bmiss= 0.001
ROW host verify on               ns=   42.99 cyc=   124.5 ins=   828.8 bmiss= 0.000
ROW host baked                   ns=   39.43 cyc=   114.1 ins=   728.8 bmiss= 0.000
ROW host bound tensor            ns=   45.40 cyc=   131.2 ins=   828.8 bmiss= 0.001
ROW host bound pointer           ns=   44.04 cyc=   127.4 ins=   800.8 bmiss= 0.000
ROW host fixed device map        ns=   43.08 cyc=   124.8 ins=   778.8 bmiss= 0.000
ROW host fixed device no-map     ns=   43.75 cyc=   126.7 ins=   780.8 bmiss= 0.000
## round=5 commit=pf1a load=8.10 13.96 14.90 t=12:24:46 gpu_use=0,0,0,0,0,0,4,0,
ROW host auto map                ns=   41.98 cyc=   121.5 ins=   784.3 bmiss= 0.001
ROW host reduced key             ns=   41.59 cyc=   120.3 ins=   788.8 bmiss= 0.000
ROW host verify off              ns=   43.26 cyc=   124.7 ins=   792.8 bmiss= 0.000
ROW host verify on               ns=   43.67 cyc=   126.5 ins=   828.8 bmiss= 0.000
ROW host baked                   ns=   39.11 cyc=   113.3 ins=   728.8 bmiss= 0.000
ROW host bound tensor            ns=   45.18 cyc=   130.4 ins=   828.8 bmiss= 0.001
ROW host bound pointer           ns=   44.44 cyc=   128.7 ins=   800.8 bmiss= 0.000
ROW host fixed device map        ns=   43.55 cyc=   126.2 ins=   778.8 bmiss= 0.001
ROW host fixed device no-map     ns=   44.80 cyc=   129.6 ins=   780.8 bmiss= 0.000
## round=6 commit=pf1 load=7.85 13.82 14.85 t=12:24:49 gpu_use=0,0,0,0,0,0,0,0,
ROW host auto map                ns=   42.23 cyc=   121.9 ins=   784.3 bmiss= 0.001
ROW host reduced key             ns=   41.64 cyc=   120.1 ins=   788.8 bmiss= 0.000
ROW host verify off              ns=   42.24 cyc=   122.3 ins=   792.8 bmiss= 0.000
ROW host verify on               ns=   42.95 cyc=   124.3 ins=   828.8 bmiss= 0.001
ROW host baked                   ns=   39.38 cyc=   114.0 ins=   728.8 bmiss= 0.000
ROW host bound tensor            ns=   45.08 cyc=   130.5 ins=   828.8 bmiss= 0.000
ROW host bound pointer           ns=   43.45 cyc=   125.9 ins=   800.8 bmiss= 0.000
ROW host fixed device map        ns=   43.36 cyc=   125.5 ins=   778.8 bmiss= 0.001
ROW host fixed device no-map     ns=   43.16 cyc=   125.0 ins=   780.8 bmiss= 0.001
## round=6 commit=pf1a load=7.70 13.69 14.80 t=12:24:53 gpu_use=0,0,0,0,0,0,0,0,
ROW host auto map                ns=   42.21 cyc=   122.1 ins=   784.3 bmiss= 0.001
ROW host reduced key             ns=   41.54 cyc=   120.0 ins=   788.8 bmiss= 0.000
ROW host verify off              ns=   42.09 cyc=   121.8 ins=   792.8 bmiss= 0.000
ROW host verify on               ns=   43.27 cyc=   125.3 ins=   828.8 bmiss= 0.001
ROW host baked                   ns=   39.22 cyc=   113.5 ins=   728.8 bmiss= 0.000
ROW host bound tensor            ns=   44.82 cyc=   129.7 ins=   828.8 bmiss= 0.001
ROW host bound pointer           ns=   44.31 cyc=   128.3 ins=   800.8 bmiss= 0.000
ROW host fixed device map        ns=   43.06 cyc=   124.6 ins=   778.8 bmiss= 0.000
ROW host fixed device no-map     ns=   44.06 cyc=   127.5 ins=   780.8 bmiss= 0.001
```

### hot3.txt (summary, then raw)
```
row                                                              pf1                               pf3
ffi hot call                         ns=  39.7 cyc= 123.5 ins= 667.3   ns=  39.3 cyc= 121.2 ins= 667.3
ffi hot call x10                     ns=  39.9 cyc= 115.2 ins= 680.9   ns=  39.2 cyc= 114.0 ins= 680.9
n per cell: {'pf1': 8, 'pf3': 8}
## round=1 commit=pf1 load=6.14 14.88 14.90 t=12:47:02 gpu_use=0,0,0,0,0,0,0,0,
ROW ffi hot call                 ns=   63.25 cyc=   195.5 ins=   667.3 bmiss= 0.048
ROW ffi hot call x10             ns=   66.28 cyc=   192.6 ins=   680.9 bmiss= 0.004
## round=1 commit=pf3 load=6.14 14.88 14.90 t=12:47:05 gpu_use=0,0,0,0,0,0,0,0,
ROW ffi hot call                 ns=   38.23 cyc=   118.0 ins=   665.4 bmiss= 0.041
ROW ffi hot call x10             ns=   38.54 cyc=   112.3 ins=   680.9 bmiss= 0.003
## round=2 commit=pf1 load=5.81 14.67 14.83 t=12:47:11 gpu_use=0,0,0,0,0,0,0,0,
ROW ffi hot call                 ns=   39.34 cyc=   121.5 ins=   668.7 bmiss= 0.041
ROW ffi hot call x10             ns=   39.33 cyc=   114.8 ins=   680.9 bmiss= 0.003
## round=2 commit=pf3 load=5.66 14.49 14.77 t=12:47:14 gpu_use=0,0,0,0,0,0,0,0,
ROW ffi hot call                 ns=   39.47 cyc=   122.1 ins=   667.3 bmiss= 0.042
ROW ffi hot call x10             ns=   39.26 cyc=   113.8 ins=   680.9 bmiss= 0.003
## round=3 commit=pf1 load=5.61 14.33 14.72 t=12:47:17 gpu_use=0,0,0,0,0,0,0,0,
ROW ffi hot call                 ns=   40.41 cyc=   125.8 ins=   667.3 bmiss= 0.045
ROW ffi hot call x10             ns=   39.90 cyc=   116.4 ins=   680.9 bmiss= 0.003
## round=3 commit=pf3 load=5.61 14.33 14.72 t=12:47:20 gpu_use=0,0,0,0,0,0,0,0,
ROW ffi hot call                 ns=   39.12 cyc=   121.3 ins=   667.3 bmiss= 0.047
ROW ffi hot call x10             ns=   39.64 cyc=   114.5 ins=   680.9 bmiss= 0.004
## round=4 commit=pf1 load=5.32 14.13 14.65 t=12:47:23 gpu_use=0,0,0,0,0,0,0,0,
ROW ffi hot call                 ns=   40.43 cyc=   124.7 ins=   667.3 bmiss= 0.040
ROW ffi hot call x10             ns=   40.08 cyc=   116.7 ins=   680.9 bmiss= 0.004
## round=4 commit=pf3 load=5.32 14.13 14.65 t=12:47:26 gpu_use=0,0,0,0,0,0,0,0,
ROW ffi hot call                 ns=   39.85 cyc=   122.5 ins=   667.3 bmiss= 0.049
ROW ffi hot call x10             ns=   38.47 cyc=   112.1 ins=   680.9 bmiss= 0.004
## round=5 commit=pf1 load=5.06 13.93 14.58 t=12:47:29 gpu_use=0,0,0,0,0,0,0,0,
ROW ffi hot call                 ns=   39.45 cyc=   124.2 ins=   667.4 bmiss= 0.048
ROW ffi hot call x10             ns=   39.15 cyc=   113.2 ins=   681.0 bmiss= 0.004
## round=5 commit=pf3 load=4.97 13.76 14.52 t=12:47:32 gpu_use=0,0,0,0,0,0,0,0,
ROW ffi hot call                 ns=   38.93 cyc=   119.9 ins=   667.6 bmiss= 0.051
ROW ffi hot call x10             ns=   39.24 cyc=   114.5 ins=   681.0 bmiss= 0.003
## round=6 commit=pf1 load=4.97 13.76 14.52 t=12:47:35 gpu_use=0,0,0,0,0,0,0,0,
ROW ffi hot call                 ns=   39.34 cyc=   121.1 ins=   667.3 bmiss= 0.045
ROW ffi hot call x10             ns=   39.89 cyc=   115.5 ins=   680.9 bmiss= 0.004
## round=6 commit=pf3 load=4.73 13.57 14.46 t=12:47:38 gpu_use=0,0,0,0,0,0,0,0,
ROW ffi hot call                 ns=   39.90 cyc=   123.0 ins=   667.3 bmiss= 0.045
ROW ffi hot call x10             ns=   39.51 cyc=   114.6 ins=   680.9 bmiss= 0.003
## round=7 commit=pf1 load=4.73 13.57 14.46 t=12:47:41 gpu_use=0,0,0,0,0,0,0,0,
ROW ffi hot call                 ns=   39.74 cyc=   122.7 ins=   667.4 bmiss= 0.043
ROW ffi hot call x10             ns=   39.88 cyc=   114.9 ins=   681.0 bmiss= 0.003
## round=7 commit=pf3 load=4.67 13.41 14.40 t=12:47:44 gpu_use=0,0,0,0,0,0,0,0,
ROW ffi hot call                 ns=   39.40 cyc=   121.2 ins=   667.4 bmiss= 0.049
ROW ffi hot call x10             ns=   39.44 cyc=   114.2 ins=   681.0 bmiss= 0.003
## round=8 commit=pf1 load=4.94 13.32 14.37 t=12:47:47 gpu_use=0,0,0,0,0,0,0,0,
ROW ffi hot call                 ns=   39.56 cyc=   122.2 ins=   668.1 bmiss= 0.046
ROW ffi hot call x10             ns=   39.02 cyc=   111.9 ins=   681.0 bmiss= 0.003
## round=8 commit=pf3 load=4.94 13.32 14.37 t=12:47:50 gpu_use=0,0,0,0,0,0,0,0,
ROW ffi hot call                 ns=   38.95 cyc=   119.9 ins=   665.9 bmiss= 0.049
ROW ffi hot call x10             ns=   39.14 cyc=   113.4 ins=   677.1 bmiss= 0.004
```

### ffi3.txt (summary, then raw)
```
row                                                              pf1                               pf3
ffi hot call                         ns=  40.1 cyc= 124.6 ins= 667.5   ns=  39.3 cyc= 121.2 ins= 667.4
ffi 3-tensor nop                     ns=  34.9 cyc= 108.1 ins= 633.5   ns=  34.6 cyc= 107.5 ins= 632.5
ffi direct positional                ns=  66.0 cyc= 198.8 ins=1185.5   ns=  65.4 cyc= 196.4 ins=1184.5
ffi direct positional (no lambda)    ns=  34.0 cyc= 105.8 ins= 637.3   ns=  33.9 cyc= 105.5 ins= 636.4
ffi pair direct                      ns=  67.2 cyc= 201.9 ins=1220.4   ns=  67.3 cyc= 202.1 ins=1220.4
ffi pair prebuilt *tuple             ns= 139.0 cyc= 410.0 ins=2168.4   ns= 141.3 cyc= 414.6 ins=2174.8
ffi pair (no lambda)                 ns=  31.6 cyc=  98.8 ins= 596.3   ns=  31.6 cyc=  98.8 ins= 596.4
n per cell: {'pf1': 6, 'pf3': 6}
## round=1 commit=pf1 load=5.18 13.23 14.33 t=12:47:53 gpu_use=0,0,0,0,0,0,0,0,
ROW ffi hot call                 ns=   40.59 cyc=   125.7 ins=   667.5 bmiss= 0.041
ROW ffi 3-tensor nop             ns=   34.88 cyc=   108.3 ins=   633.6 bmiss= 0.035
ROW ffi direct positional        ns=   65.83 cyc=   197.8 ins=  1185.6 bmiss= 0.037
ROW ffi direct positional (no lambda) ns=   34.00 cyc=   105.8 ins=   637.5 bmiss= 0.029
ROW ffi pair direct              ns=   67.27 cyc=   202.2 ins=  1220.5 bmiss= 0.030
ROW ffi pair prebuilt *tuple     ns=  137.69 cyc=   406.1 ins=  2168.4 bmiss= 0.030
ROW ffi pair (no lambda)         ns=   31.61 cyc=    99.0 ins=   596.4 bmiss= 0.031
## round=1 commit=pf3 load=5.18 13.23 14.33 t=12:47:56 gpu_use=0,0,0,0,0,0,0,0,
ROW ffi hot call                 ns=   39.74 cyc=   122.2 ins=   667.4 bmiss= 0.047
ROW ffi 3-tensor nop             ns=   35.76 cyc=   108.4 ins=   632.5 bmiss= 0.040
ROW ffi direct positional        ns=   65.42 cyc=   196.4 ins=  1184.5 bmiss= 0.035
ROW ffi direct positional (no lambda) ns=   33.86 cyc=   105.2 ins=   636.4 bmiss= 0.033
ROW ffi pair direct              ns=   67.57 cyc=   203.3 ins=  1220.4 bmiss= 0.034
ROW ffi pair prebuilt *tuple     ns=  141.61 cyc=   417.8 ins=  2181.2 bmiss= 0.034
ROW ffi pair (no lambda)         ns=   31.50 cyc=    99.0 ins=   596.4 bmiss= 0.033
## round=2 commit=pf1 load=4.85 12.89 14.21 t=12:48:03 gpu_use=0,0,0,0,0,0,0,0,
ROW ffi hot call                 ns=   40.40 cyc=   126.2 ins=   668.7 bmiss= 0.042
ROW ffi 3-tensor nop             ns=   35.12 cyc=   108.7 ins=   633.5 bmiss= 0.037
ROW ffi direct positional        ns=   65.79 cyc=   197.5 ins=  1185.5 bmiss= 0.034
ROW ffi direct positional (no lambda) ns=   34.12 cyc=   105.9 ins=   637.3 bmiss= 0.032
ROW ffi pair direct              ns=   67.13 cyc=   201.4 ins=  1220.4 bmiss= 0.032
ROW ffi pair prebuilt *tuple     ns=  138.93 cyc=   409.8 ins=  2181.3 bmiss= 0.036
ROW ffi pair (no lambda)         ns=   31.68 cyc=    99.0 ins=   596.3 bmiss= 0.032
## round=2 commit=pf3 load=4.85 12.89 14.21 t=12:48:06 gpu_use=0,0,0,0,0,0,0,0,
ROW ffi hot call                 ns=   38.97 cyc=   120.3 ins=   667.3 bmiss= 0.049
ROW ffi 3-tensor nop             ns=   34.65 cyc=   107.5 ins=   632.5 bmiss= 0.037
ROW ffi direct positional        ns=   65.46 cyc=   197.8 ins=  1184.5 bmiss= 0.038
ROW ffi direct positional (no lambda) ns=   33.98 cyc=   105.7 ins=   636.3 bmiss= 0.032
ROW ffi pair direct              ns=   66.18 cyc=   198.8 ins=  1220.4 bmiss= 0.032
ROW ffi pair prebuilt *tuple     ns=  166.23 cyc=   482.4 ins=  2168.3 bmiss= 1.014
ROW ffi pair (no lambda)         ns=   57.50 cyc=   173.9 ins=   596.3 bmiss= 1.035
## round=3 commit=pf1 load=4.62 12.71 14.14 t=12:48:09 gpu_use=0,0,0,0,0,0,0,0,
ROW ffi hot call                 ns=   40.47 cyc=   124.4 ins=   667.3 bmiss= 0.047
ROW ffi 3-tensor nop             ns=   35.30 cyc=   109.4 ins=   633.5 bmiss= 0.035
ROW ffi direct positional        ns=   67.06 cyc=   201.2 ins=  1185.5 bmiss= 0.034
ROW ffi direct positional (no lambda) ns=   33.94 cyc=   105.5 ins=   637.3 bmiss= 0.032
ROW ffi pair direct              ns=   66.68 cyc=   200.4 ins=  1220.4 bmiss= 0.042
ROW ffi pair prebuilt *tuple     ns=  139.06 cyc=   410.2 ins=  2168.3 bmiss= 0.034
ROW ffi pair (no lambda)         ns=   31.61 cyc=    98.7 ins=   596.3 bmiss= 0.030
## round=3 commit=pf3 load=4.49 12.55 14.08 t=12:48:12 gpu_use=0,0,0,0,0,0,0,0,
ROW ffi hot call                 ns=   39.34 cyc=   121.2 ins=   667.3 bmiss= 0.052
ROW ffi 3-tensor nop             ns=   34.79 cyc=   108.0 ins=   632.5 bmiss= 0.034
ROW ffi direct positional        ns=   64.75 cyc=   195.5 ins=  1184.5 bmiss= 0.033
ROW ffi direct positional (no lambda) ns=   33.82 cyc=   104.9 ins=   636.3 bmiss= 0.031
ROW ffi pair direct              ns=   67.37 cyc=   202.0 ins=  1220.4 bmiss= 0.030
ROW ffi pair prebuilt *tuple     ns=  140.50 cyc=   414.3 ins=  2181.1 bmiss= 0.038
ROW ffi pair (no lambda)         ns=   31.49 cyc=    98.6 ins=   596.3 bmiss= 0.034
## round=4 commit=pf1 load=4.49 12.55 14.08 t=12:48:15 gpu_use=0,0,0,0,0,0,0,0,
ROW ffi hot call                 ns=   39.64 cyc=   122.0 ins=   667.3 bmiss= 0.044
ROW ffi 3-tensor nop             ns=   34.30 cyc=   106.8 ins=   633.5 bmiss= 0.036
ROW ffi direct positional        ns=   66.01 cyc=   199.5 ins=  1185.5 bmiss= 0.034
ROW ffi direct positional (no lambda) ns=   34.04 cyc=   105.8 ins=   637.3 bmiss= 0.031
ROW ffi pair direct              ns=   67.43 cyc=   202.3 ins=  1220.4 bmiss= 0.037
ROW ffi pair prebuilt *tuple     ns=  137.34 cyc=   405.1 ins=  2168.3 bmiss= 0.029
ROW ffi pair (no lambda)         ns=   31.61 cyc=    98.7 ins=   596.3 bmiss= 0.031
## round=4 commit=pf3 load=4.37 12.39 14.02 t=12:48:18 gpu_use=0,0,0,0,0,0,0,0,
ROW ffi hot call                 ns=   39.04 cyc=   120.8 ins=   668.7 bmiss= 0.043
ROW ffi 3-tensor nop             ns=   34.53 cyc=   107.2 ins=   632.6 bmiss= 0.034
ROW ffi direct positional        ns=   65.22 cyc=   196.2 ins=  1184.6 bmiss= 0.035
ROW ffi direct positional (no lambda) ns=   33.85 cyc=   105.4 ins=   636.4 bmiss= 0.033
ROW ffi pair direct              ns=   66.74 cyc=   200.6 ins=  1220.4 bmiss= 0.029
ROW ffi pair prebuilt *tuple     ns=  140.70 cyc=   414.9 ins=  2185.4 bmiss= 0.035
ROW ffi pair (no lambda)         ns=   31.61 cyc=    98.7 ins=   596.4 bmiss= 0.033
## round=5 commit=pf1 load=4.37 12.39 14.02 t=12:48:21 gpu_use=0,0,0,0,0,0,0,0,
ROW ffi hot call                 ns=   39.86 cyc=   122.9 ins=   667.6 bmiss= 0.054
ROW ffi 3-tensor nop             ns=   34.83 cyc=   107.9 ins=   633.7 bmiss= 0.034
ROW ffi direct positional        ns=   66.07 cyc=   198.3 ins=  1185.7 bmiss= 0.037
ROW ffi direct positional (no lambda) ns=   34.05 cyc=   105.9 ins=   637.5 bmiss= 0.034
ROW ffi pair direct              ns=   67.19 cyc=   201.6 ins=  1220.6 bmiss= 0.033
ROW ffi pair prebuilt *tuple     ns=  140.97 cyc=   415.6 ins=  2168.5 bmiss= 0.035
ROW ffi pair (no lambda)         ns=   31.69 cyc=    98.8 ins=   596.5 bmiss= 0.033
## round=5 commit=pf3 load=4.18 12.22 13.96 t=12:48:24 gpu_use=0,0,0,0,0,0,0,0,
ROW ffi hot call                 ns=   39.27 cyc=   121.2 ins=   667.4 bmiss= 0.044
ROW ffi 3-tensor nop             ns=   34.49 cyc=   107.4 ins=   632.6 bmiss= 0.035
ROW ffi direct positional        ns=   65.31 cyc=   196.5 ins=  1184.5 bmiss= 0.033
ROW ffi direct positional (no lambda) ns=   34.08 cyc=   105.6 ins=   636.4 bmiss= 0.035
ROW ffi pair direct              ns=   68.70 cyc=   206.1 ins=  1220.4 bmiss= 0.031
ROW ffi pair prebuilt *tuple     ns=  140.96 cyc=   406.4 ins=  2168.4 bmiss= 0.032
ROW ffi pair (no lambda)         ns=   31.53 cyc=    98.7 ins=   596.4 bmiss= 0.034
## round=6 commit=pf1 load=4.25 12.10 13.91 t=12:48:27 gpu_use=0,0,0,0,0,0,0,0,
ROW ffi hot call                 ns=   39.79 cyc=   124.7 ins=   667.4 bmiss= 0.047
ROW ffi 3-tensor nop             ns=   34.45 cyc=   107.3 ins=   633.5 bmiss= 0.035
ROW ffi direct positional        ns=   66.31 cyc=   199.2 ins=  1185.5 bmiss= 0.035
ROW ffi direct positional (no lambda) ns=   34.04 cyc=   105.6 ins=   637.4 bmiss= 0.033
ROW ffi pair direct              ns=   67.82 cyc=   203.1 ins=  1220.4 bmiss= 0.031
ROW ffi pair prebuilt *tuple     ns=  141.85 cyc=   417.6 ins=  2181.2 bmiss= 0.037
ROW ffi pair (no lambda)         ns=   31.48 cyc=    98.5 ins=   596.4 bmiss= 0.034
## round=6 commit=pf3 load=4.25 12.10 13.91 t=12:48:30 gpu_use=0,0,0,0,0,0,0,0,
ROW ffi hot call                 ns=   39.78 cyc=   122.5 ins=   667.4 bmiss= 0.041
ROW ffi 3-tensor nop             ns=   34.37 cyc=   106.8 ins=   632.6 bmiss= 0.035
ROW ffi direct positional        ns=   65.68 cyc=   197.2 ins=  1184.6 bmiss= 0.037
ROW ffi direct positional (no lambda) ns=   33.91 cyc=   105.5 ins=   636.4 bmiss= 0.034
ROW ffi pair direct              ns=   67.32 cyc=   202.1 ins=  1220.5 bmiss= 0.028
ROW ffi pair prebuilt *tuple     ns=  142.11 cyc=   413.1 ins=  2168.4 bmiss= 0.030
ROW ffi pair (no lambda)         ns=   31.59 cyc=    98.8 ins=   596.4 bmiss= 0.034
```

### host3.txt (summary, then raw)
```
row                                                              pf1                               pf3
host auto map                        ns=  42.2 cyc= 122.0 ins= 784.3   ns=  42.2 cyc= 122.0 ins= 784.3
host reduced key                     ns=  41.4 cyc= 119.7 ins= 788.8   ns=  41.8 cyc= 120.8 ins= 787.8
host verify off                      ns=  42.1 cyc= 122.0 ins= 792.8   ns=  43.2 cyc= 125.1 ins= 791.8
host verify on                       ns=  43.3 cyc= 124.9 ins= 828.8   ns=  43.1 cyc= 124.5 ins= 827.8
host baked                           ns=  39.9 cyc= 115.4 ins= 728.8   ns=  39.3 cyc= 113.8 ins= 728.8
host bound tensor                    ns=  45.7 cyc= 132.2 ins= 828.8   ns=  46.3 cyc= 133.9 ins= 827.8
host bound pointer                   ns=  44.3 cyc= 128.2 ins= 800.8   ns=  45.6 cyc= 132.1 ins= 799.8
host fixed device map                ns=  43.4 cyc= 125.4 ins= 778.8   ns=  42.3 cyc= 122.2 ins= 776.8
host fixed device no-map             ns=  43.7 cyc= 126.5 ins= 780.8   ns=  44.3 cyc= 128.3 ins= 780.8
n per cell: {'pf1': 6, 'pf3': 6}
## round=1 commit=pf1 load=4.15 11.95 13.85 t=12:48:33 gpu_use=0,0,0,0,0,0,0,0,
ROW host auto map                ns=   42.23 cyc=   122.1 ins=   784.3 bmiss= 0.001
ROW host reduced key             ns=   41.52 cyc=   120.0 ins=   788.8 bmiss= 0.000
ROW host verify off              ns=   41.81 cyc=   121.0 ins=   792.8 bmiss= 0.000
ROW host verify on               ns=   43.10 cyc=   124.7 ins=   828.8 bmiss= 0.000
ROW host baked                   ns=   39.54 cyc=   114.5 ins=   728.8 bmiss= 0.000
ROW host bound tensor            ns=   45.21 cyc=   130.8 ins=   828.8 bmiss= 0.000
ROW host bound pointer           ns=   44.23 cyc=   128.0 ins=   800.8 bmiss= 0.000
ROW host fixed device map        ns=   43.58 cyc=   126.2 ins=   778.8 bmiss= 0.000
ROW host fixed device no-map     ns=   44.23 cyc=   128.1 ins=   780.8 bmiss= 0.000
## round=1 commit=pf3 load=4.05 11.80 13.79 t=12:48:37 gpu_use=0,0,0,0,0,0,0,0,
ROW host auto map                ns=   41.71 cyc=   120.4 ins=   775.8 bmiss= 0.000
ROW host reduced key             ns=   41.56 cyc=   120.1 ins=   787.8 bmiss= 0.000
ROW host verify off              ns=   43.40 cyc=   125.6 ins=   791.8 bmiss= 0.000
ROW host verify on               ns=   42.90 cyc=   123.9 ins=   827.8 bmiss= 0.000
ROW host baked                   ns=   39.23 cyc=   113.2 ins=   728.8 bmiss= 0.000
ROW host bound tensor            ns=   46.02 cyc=   133.1 ins=   827.8 bmiss= 0.000
ROW host bound pointer           ns=   45.97 cyc=   133.1 ins=   799.8 bmiss= 0.000
ROW host fixed device map        ns=   42.19 cyc=   122.1 ins=   776.8 bmiss= 0.000
ROW host fixed device no-map     ns=   44.25 cyc=   128.0 ins=   780.8 bmiss= 0.000
## round=2 commit=pf1 load=3.47 11.16 13.54 t=12:48:58 gpu_use=0,0,0,0,0,0,0,0,
ROW host auto map                ns=   42.13 cyc=   121.9 ins=   784.3 bmiss= 0.001
ROW host reduced key             ns=   41.82 cyc=   121.1 ins=   788.8 bmiss= 0.000
ROW host verify off              ns=   41.79 cyc=   121.1 ins=   792.8 bmiss= 0.000
ROW host verify on               ns=   43.21 cyc=   125.1 ins=   828.8 bmiss= 0.000
ROW host baked                   ns=   40.36 cyc=   116.8 ins=   728.8 bmiss= 0.000
ROW host bound tensor            ns=   45.83 cyc=   132.6 ins=   828.8 bmiss= 0.001
ROW host bound pointer           ns=   43.78 cyc=   126.7 ins=   800.8 bmiss= 0.000
ROW host fixed device map        ns=   43.24 cyc=   125.1 ins=   778.8 bmiss= 0.000
ROW host fixed device no-map     ns=   43.35 cyc=   125.4 ins=   780.8 bmiss= 0.000
## round=2 commit=pf3 load=3.47 11.16 13.54 t=12:49:01 gpu_use=0,0,0,0,0,0,0,0,
ROW host auto map                ns=   43.32 cyc=   125.3 ins=   784.3 bmiss= 0.001
ROW host reduced key             ns=   41.82 cyc=   121.0 ins=   787.8 bmiss= 0.000
ROW host verify off              ns=   43.06 cyc=   124.7 ins=   791.8 bmiss= 0.000
ROW host verify on               ns=   43.07 cyc=   124.5 ins=   827.8 bmiss= 0.000
ROW host baked                   ns=   38.82 cyc=   112.5 ins=   728.8 bmiss= 0.000
ROW host bound tensor            ns=   46.42 cyc=   134.3 ins=   827.8 bmiss= 0.000
ROW host bound pointer           ns=   45.51 cyc=   131.7 ins=   799.8 bmiss= 0.000
ROW host fixed device map        ns=   42.18 cyc=   122.1 ins=   776.8 bmiss= 0.000
ROW host fixed device no-map     ns=   44.73 cyc=   129.0 ins=   780.8 bmiss= 0.000
## round=3 commit=pf1 load=3.51 11.04 13.49 t=12:49:05 gpu_use=0,0,0,0,0,0,0,0,
ROW host auto map                ns=   42.15 cyc=   122.0 ins=   784.3 bmiss= 0.000
ROW host reduced key             ns=   41.46 cyc=   119.7 ins=   788.8 bmiss= 0.000
ROW host verify off              ns=   42.53 cyc=   122.9 ins=   792.8 bmiss= 0.000
ROW host verify on               ns=   43.11 cyc=   124.6 ins=   828.8 bmiss= 0.000
ROW host baked                   ns=   39.93 cyc=   115.6 ins=   728.8 bmiss= 0.000
ROW host bound tensor            ns=   44.92 cyc=   130.1 ins=   828.8 bmiss= 0.000
ROW host bound pointer           ns=   44.19 cyc=   128.0 ins=   800.8 bmiss= 0.000
ROW host fixed device map        ns=   43.57 cyc=   125.5 ins=   778.8 bmiss= 0.000
ROW host fixed device no-map     ns=   43.17 cyc=   124.9 ins=   780.8 bmiss= 0.000
## round=3 commit=pf3 load=3.55 10.92 13.44 t=12:49:08 gpu_use=0,0,0,0,0,0,0,0,
ROW host auto map                ns=   42.27 cyc=   122.4 ins=   784.3 bmiss= 0.001
ROW host reduced key             ns=   41.87 cyc=   121.1 ins=   787.8 bmiss= 0.000
ROW host verify off              ns=   42.64 cyc=   123.5 ins=   791.8 bmiss= 0.000
ROW host verify on               ns=   43.81 cyc=   124.6 ins=   827.8 bmiss= 0.000
ROW host baked                   ns=   39.37 cyc=   114.0 ins=   728.8 bmiss= 0.000
ROW host bound tensor            ns=   46.21 cyc=   133.8 ins=   827.8 bmiss= 0.000
ROW host bound pointer           ns=   45.57 cyc=   132.0 ins=   799.8 bmiss= 0.000
ROW host fixed device map        ns=   42.66 cyc=   123.5 ins=   776.8 bmiss= 0.000
ROW host fixed device no-map     ns=   44.13 cyc=   127.8 ins=   780.8 bmiss= 0.000
## round=4 commit=pf1 load=3.55 10.92 13.44 t=12:49:11 gpu_use=0,0,0,0,0,0,0,0,
ROW host auto map                ns=   42.19 cyc=   122.1 ins=   784.3 bmiss= 0.001
ROW host reduced key             ns=   41.39 cyc=   119.6 ins=   788.8 bmiss= 0.000
ROW host verify off              ns=   42.05 cyc=   121.7 ins=   792.8 bmiss= 0.000
ROW host verify on               ns=   43.90 cyc=   126.8 ins=   828.8 bmiss= 0.000
ROW host baked                   ns=   40.55 cyc=   117.3 ins=   728.8 bmiss= 0.000
ROW host bound tensor            ns=   46.04 cyc=   133.2 ins=   828.8 bmiss= 0.000
ROW host bound pointer           ns=   44.49 cyc=   128.7 ins=   800.8 bmiss= 0.000
ROW host fixed device map        ns=   42.80 cyc=   123.9 ins=   778.8 bmiss= 0.000
ROW host fixed device no-map     ns=   43.58 cyc=   126.1 ins=   780.8 bmiss= 0.000
## round=4 commit=pf3 load=3.59 10.81 13.39 t=12:49:14 gpu_use=0,0,0,0,0,0,0,0,
ROW host auto map                ns=   42.22 cyc=   122.0 ins=   784.3 bmiss= 0.001
ROW host reduced key             ns=   41.59 cyc=   120.3 ins=   787.8 bmiss= 0.000
ROW host verify off              ns=   43.99 cyc=   127.2 ins=   791.8 bmiss= 0.000
ROW host verify on               ns=   43.69 cyc=   126.4 ins=   827.8 bmiss= 0.000
ROW host baked                   ns=   39.60 cyc=   114.6 ins=   728.8 bmiss= 0.000
ROW host bound tensor            ns=   46.33 cyc=   134.0 ins=   827.8 bmiss= 0.000
ROW host bound pointer           ns=   45.73 cyc=   132.2 ins=   799.8 bmiss= 0.000
ROW host fixed device map        ns=   43.22 cyc=   125.0 ins=   776.8 bmiss= 0.000
ROW host fixed device no-map     ns=   44.42 cyc=   128.6 ins=   780.8 bmiss= 0.000
## round=5 commit=pf1 load=3.70 10.71 13.34 t=12:49:17 gpu_use=0,0,0,0,0,0,0,0,
ROW host auto map                ns=   42.10 cyc=   121.8 ins=   784.3 bmiss= 0.001
ROW host reduced key             ns=   41.34 cyc=   119.6 ins=   788.8 bmiss= 0.000
ROW host verify off              ns=   42.19 cyc=   122.2 ins=   792.8 bmiss= 0.000
ROW host verify on               ns=   43.71 cyc=   123.9 ins=   828.8 bmiss= 0.000
ROW host baked                   ns=   39.83 cyc=   115.3 ins=   728.8 bmiss= 0.000
ROW host bound tensor            ns=   45.55 cyc=   131.9 ins=   828.8 bmiss= 0.001
ROW host bound pointer           ns=   44.63 cyc=   129.2 ins=   800.8 bmiss= 0.000
ROW host fixed device map        ns=   43.55 cyc=   126.0 ins=   778.8 bmiss= 0.000
ROW host fixed device no-map     ns=   43.96 cyc=   127.2 ins=   780.8 bmiss= 0.000
## round=5 commit=pf3 load=3.70 10.71 13.34 t=12:49:20 gpu_use=0,0,0,0,0,0,0,0,
ROW host auto map                ns=   42.25 cyc=   122.1 ins=   784.3 bmiss= 0.001
ROW host reduced key             ns=   42.01 cyc=   121.4 ins=   787.8 bmiss= 0.000
ROW host verify off              ns=   43.34 cyc=   125.4 ins=   791.8 bmiss= 0.000
ROW host verify on               ns=   43.00 cyc=   124.4 ins=   827.8 bmiss= 0.000
ROW host baked                   ns=   39.66 cyc=   114.2 ins=   728.8 bmiss= 0.000
ROW host bound tensor            ns=   46.70 cyc=   135.1 ins=   827.8 bmiss= 0.000
ROW host bound pointer           ns=   46.43 cyc=   134.3 ins=   799.8 bmiss= 0.000
ROW host fixed device map        ns=   42.33 cyc=   122.4 ins=   776.8 bmiss= 0.000
ROW host fixed device no-map     ns=   44.43 cyc=   128.6 ins=   780.8 bmiss= 0.000
## round=6 commit=pf1 load=3.81 10.62 13.30 t=12:49:24 gpu_use=0,0,0,0,0,0,0,0,
ROW host auto map                ns=   42.27 cyc=   122.4 ins=   784.3 bmiss= 0.001
ROW host reduced key             ns=   41.30 cyc=   119.6 ins=   788.8 bmiss= 0.000
ROW host verify off              ns=   42.32 cyc=   122.5 ins=   792.8 bmiss= 0.000
ROW host verify on               ns=   43.35 cyc=   125.4 ins=   828.8 bmiss= 0.000
ROW host baked                   ns=   39.30 cyc=   113.8 ins=   728.8 bmiss= 0.000
ROW host bound tensor            ns=   46.39 cyc=   133.7 ins=   828.8 bmiss= 0.000
ROW host bound pointer           ns=   44.38 cyc=   128.4 ins=   800.8 bmiss= 0.000
ROW host fixed device map        ns=   43.26 cyc=   125.3 ins=   778.8 bmiss= 0.000
ROW host fixed device no-map     ns=   43.84 cyc=   127.0 ins=   780.8 bmiss= 0.001
## round=6 commit=pf3 load=3.82 10.51 13.25 t=12:49:27 gpu_use=0,0,0,0,0,0,0,0,
ROW host auto map                ns=   42.14 cyc=   121.9 ins=   784.3 bmiss= 0.001
ROW host reduced key             ns=   41.70 cyc=   120.6 ins=   787.8 bmiss= 0.000
ROW host verify off              ns=   42.98 cyc=   124.5 ins=   791.8 bmiss= 0.000
ROW host verify on               ns=   43.10 cyc=   124.8 ins=   827.8 bmiss= 0.000
ROW host baked                   ns=   39.26 cyc=   113.7 ins=   728.8 bmiss= 0.000
ROW host bound tensor            ns=   46.02 cyc=   133.3 ins=   827.8 bmiss= 0.000
ROW host bound pointer           ns=   45.30 cyc=   131.2 ins=   799.8 bmiss= 0.000
ROW host fixed device map        ns=   42.08 cyc=   121.8 ins=   776.8 bmiss= 0.000
ROW host fixed device no-map     ns=   44.17 cyc=   127.8 ins=   780.8 bmiss= 0.000
```

### bl2.txt
```
## round=1 commit=pf1 load=7.70 13.69 14.80
mode=host-only; 100000 calls × 9 batches; median ns/call
               path    ns/call       Δ ns       Δ %   build ms   bind ms
           auto map       42.1       +0.0      +0.0     302.31         -
        reduced key       41.4       -0.8      -1.8     137.17         -
         verify off       42.0       -0.1      -0.2     134.51         -
          verify on       43.5       +1.4      +3.3     135.15         -
              baked       39.4       -2.7      -6.4     132.77         -
       bound tensor       44.7       +2.6      +6.2       0.20    130.74
      bound pointer       43.4       +1.3      +3.1       0.18    129.82
   fixed device map       43.0       +0.9      +2.1       0.14    129.86
fixed device no-map       43.2       +1.1      +2.6       0.18    129.30
## round=1 commit=pf1a load=7.88 13.62 14.78
mode=host-only; 100000 calls × 9 batches; median ns/call
               path    ns/call       Δ ns       Δ %   build ms   bind ms
           auto map       41.4       +0.0      +0.0     305.64         -
        reduced key       41.8       +0.4      +0.9     136.89         -
         verify off       43.2       +1.7      +4.1     134.06         -
          verify on       43.0       +1.6      +3.9     131.91         -
              baked       38.8       -2.6      -6.3     130.42         -
       bound tensor       44.9       +3.5      +8.4       0.20    131.77
      bound pointer       44.2       +2.8      +6.6       0.20    133.58
   fixed device map       43.0       +1.5      +3.7       0.18    133.66
fixed device no-map       44.2       +2.8      +6.8       0.22    132.33
## round=2 commit=pf1 load=7.73 13.50 14.73
mode=host-only; 100000 calls × 9 batches; median ns/call
               path    ns/call       Δ ns       Δ %   build ms   bind ms
           auto map       42.4       +0.0      +0.0      63.67         -
        reduced key       42.5       +0.1      +0.2       2.15         -
         verify off       42.1       -0.3      -0.8       2.00         -
          verify on       43.1       +0.7      +1.7       2.11         -
              baked       39.5       -2.9      -6.7       1.99         -
       bound tensor       45.1       +2.7      +6.3       0.17      1.90
      bound pointer       43.0       +0.6      +1.3       0.16      1.85
   fixed device map       42.5       +0.1      +0.2       0.12      1.82
fixed device no-map       43.1       +0.7      +1.6       0.16      1.86
## round=2 commit=pf1a load=7.59 13.37 14.68
mode=host-only; 100000 calls × 9 batches; median ns/call
               path    ns/call       Δ ns       Δ %   build ms   bind ms
           auto map       43.0       +0.0      +0.0      63.40         -
        reduced key       42.6       -0.4      -0.8       2.17         -
         verify off       42.5       -0.4      -1.0       2.00         -
          verify on       42.9       -0.1      -0.2       2.13         -
              baked       39.1       -3.9      -9.1       1.96         -
       bound tensor       45.0       +2.0      +4.6       0.16      1.85
      bound pointer       44.1       +1.1      +2.6       0.15      1.87
   fixed device map       42.5       -0.5      -1.2       0.12      1.81
fixed device no-map       44.3       +1.3      +3.0       0.16      1.85
## round=3 commit=pf1 load=7.59 13.37 14.68
mode=host-only; 100000 calls × 9 batches; median ns/call
               path    ns/call       Δ ns       Δ %   build ms   bind ms
           auto map       42.3       +0.0      +0.0      62.86         -
        reduced key       42.7       +0.4      +1.0       2.10         -
         verify off       42.0       -0.4      -0.8       1.99         -
          verify on       43.3       +1.0      +2.4       2.12         -
              baked       39.2       -3.1      -7.4       1.95         -
       bound tensor       45.1       +2.8      +6.5       0.15      1.80
      bound pointer       43.4       +1.1      +2.6       0.14      1.81
   fixed device map       43.8       +1.4      +3.4       0.11      1.77
fixed device no-map       42.6       +0.3      +0.8       0.16      1.83
## round=3 commit=pf1a load=7.47 13.25 14.63
mode=host-only; 100000 calls × 9 batches; median ns/call
               path    ns/call       Δ ns       Δ %   build ms   bind ms
           auto map       43.0       +0.0      +0.0      62.98         -
        reduced key       42.6       -0.5      -1.1       2.14         -
         verify off       42.1       -0.9      -2.1       1.98         -
          verify on       43.3       +0.2      +0.6       2.15         -
              baked       39.1       -3.9      -9.1       1.99         -
       bound tensor       44.3       +1.3      +3.0       0.15      1.84
      bound pointer       43.8       +0.8      +1.8       0.15      1.84
   fixed device map       43.0       -0.1      -0.1       0.12      1.80
fixed device no-map       44.0       +1.0      +2.3       0.16      1.83
## round=4 commit=pf1 load=7.47 13.25 14.63
mode=host-only; 100000 calls × 9 batches; median ns/call
               path    ns/call       Δ ns       Δ %   build ms   bind ms
           auto map       42.4       +0.0      +0.0      63.39         -
        reduced key       42.4       +0.0      +0.0       2.13         -
         verify off       42.3       -0.1      -0.2       1.95         -
          verify on       42.8       +0.3      +0.8       2.09         -
              baked       39.0       -3.5      -8.2       1.93         -
       bound tensor       44.8       +2.4      +5.7       0.15      1.79
      bound pointer       43.6       +1.2      +2.7       0.15      1.81
   fixed device map       42.5       +0.1      +0.1       0.11      1.73
fixed device no-map       43.3       +0.9      +2.2       0.14      1.82
## round=4 commit=pf1a load=7.19 13.10 14.58
mode=host-only; 100000 calls × 9 batches; median ns/call
               path    ns/call       Δ ns       Δ %   build ms   bind ms
           auto map       43.0       +0.0      +0.0      62.87         -
        reduced key       42.5       -0.5      -1.1       2.10         -
         verify off       42.3       -0.7      -1.7       1.97         -
          verify on       43.9       +0.9      +2.1       2.08         -
              baked       38.8       -4.2      -9.8       1.92         -
       bound tensor       45.3       +2.3      +5.4       0.15      1.79
      bound pointer       44.1       +1.1      +2.6       0.14      1.78
   fixed device map       43.0       -0.0      -0.0       0.12      1.78
fixed device no-map       44.0       +1.0      +2.2       0.15      1.77
```

### bl3.txt
```
## round=1 commit=pf1 load=3.82 10.51 13.25
mode=host-only; 100000 calls × 9 batches; median ns/call
               path    ns/call       Δ ns       Δ %   build ms   bind ms
           auto map       42.3       +0.0      +0.0      62.80         -
        reduced key       42.4       +0.1      +0.3       2.12         -
         verify off       42.2       -0.0      -0.1       1.97         -
          verify on       43.7       +1.4      +3.4       2.06         -
              baked       39.0       -3.2      -7.7       1.95         -
       bound tensor       45.7       +3.4      +8.1       0.15      1.82
      bound pointer       43.1       +0.9      +2.1       0.15      1.83
   fixed device map       42.5       +0.3      +0.6       0.11      1.78
fixed device no-map       43.8       +1.6      +3.7       0.14      1.81
## round=1 commit=pf3 load=3.68 10.37 13.19
mode=host-only; 100000 calls × 9 batches; median ns/call
               path    ns/call       Δ ns       Δ %   build ms   bind ms
           auto map       42.0       +0.0      +0.0     276.94         -
        reduced key       41.5       -0.5      -1.2     132.85         -
         verify off       43.0       +1.0      +2.5     131.17         -
          verify on       43.0       +1.1      +2.5     131.43         -
              baked       39.1       -2.8      -6.7     129.86         -
       bound tensor       45.3       +3.3      +7.9       0.17    130.45
      bound pointer       44.9       +2.9      +7.0       0.17    131.38
   fixed device map       42.0       +0.1      +0.1       0.14    130.10
fixed device no-map       43.5       +1.5      +3.7       0.17    135.97
## round=2 commit=pf1 load=3.56 10.12 13.08
mode=host-only; 100000 calls × 9 batches; median ns/call
               path    ns/call       Δ ns       Δ %   build ms   bind ms
           auto map       42.3       +0.0      +0.0      62.53         -
        reduced key       42.4       +0.1      +0.3       2.11         -
         verify off       42.4       +0.1      +0.2       1.96         -
          verify on       42.9       +0.6      +1.4       2.06         -
              baked       39.2       -3.1      -7.2       1.95         -
       bound tensor       45.3       +3.0      +7.1       0.15      1.80
      bound pointer       43.9       +1.6      +3.9       0.15      1.84
   fixed device map       42.9       +0.6      +1.4       0.12      1.82
fixed device no-map       71.9      +29.6     +69.9       0.15      1.76
## round=2 commit=pf3 load=3.44 9.99 13.02
mode=host-only; 100000 calls × 9 batches; median ns/call
               path    ns/call       Δ ns       Δ %   build ms   bind ms
           auto map       42.3       +0.0      +0.0      62.70         -
        reduced key       42.5       +0.2      +0.5       2.10         -
         verify off       43.0       +0.7      +1.8       1.98         -
          verify on       43.1       +0.8      +2.0       2.13         -
              baked       38.8       -3.5      -8.3       1.97         -
       bound tensor       45.2       +2.9      +7.0       0.14      1.83
      bound pointer       44.5       +2.2      +5.2       0.15      1.82
   fixed device map       43.0       +0.7      +1.7       0.11      1.80
fixed device no-map       43.9       +1.6      +3.9       0.16      1.82
## round=3 commit=pf1 load=3.32 9.85 12.96
mode=host-only; 100000 calls × 9 batches; median ns/call
               path    ns/call       Δ ns       Δ %   build ms   bind ms
           auto map       42.2       +0.0      +0.0      62.51         -
        reduced key       42.5       +0.3      +0.7       2.10         -
         verify off       42.1       -0.1      -0.2       1.98         -
          verify on       42.9       +0.7      +1.5       2.05         -
              baked       39.4       -2.8      -6.5       1.93         -
       bound tensor       44.9       +2.7      +6.4       0.15      1.84
      bound pointer       43.8       +1.5      +3.7       0.14      1.79
   fixed device map       42.7       +0.5      +1.3       0.12      1.77
fixed device no-map       43.3       +1.1      +2.7       0.15      1.78
## round=3 commit=pf3 load=3.22 9.72 12.90
mode=host-only; 100000 calls × 9 batches; median ns/call
               path    ns/call       Δ ns       Δ %   build ms   bind ms
           auto map       43.3       +0.0      +0.0      62.80         -
        reduced key       42.8       -0.4      -1.0       2.11         -
         verify off       44.1       +0.8      +1.8       1.97         -
          verify on       43.2       -0.1      -0.2       2.07         -
              baked       38.8       -4.5     -10.3       1.91         -
       bound tensor       45.9       +2.7      +6.2       0.15      1.84
      bound pointer       44.5       +1.2      +2.8       0.14      1.83
   fixed device map       43.5       +0.3      +0.7       0.11      1.79
fixed device no-map       43.7       +0.4      +1.0       0.15      1.85
## round=4 commit=pf1 load=3.20 9.61 12.84
mode=host-only; 100000 calls × 9 batches; median ns/call
               path    ns/call       Δ ns       Δ %   build ms   bind ms
           auto map       42.0       +0.0      +0.0      62.27         -
        reduced key       42.5       +0.5      +1.1       2.09         -
         verify off       41.9       -0.1      -0.3       1.94         -
          verify on       43.1       +1.1      +2.6       2.09         -
              baked       39.3       -2.7      -6.5       1.95         -
       bound tensor       45.6       +3.6      +8.5       0.16      1.81
      bound pointer       44.0       +1.9      +4.6       0.15      1.77
   fixed device map       42.4       +0.3      +0.8       0.11      1.80
fixed device no-map       43.9       +1.9      +4.5       0.15      1.82
## round=4 commit=pf3 load=3.17 9.39 12.74
mode=host-only; 100000 calls × 9 batches; median ns/call
               path    ns/call       Δ ns       Δ %   build ms   bind ms
           auto map       42.2       +0.0      +0.0      62.32         -
        reduced key       42.4       +0.2      +0.4       2.10         -
         verify off       42.5       +0.3      +0.7       1.97         -
          verify on       43.0       +0.8      +1.9       2.05         -
              baked       38.9       -3.3      -7.8       1.97         -
       bound tensor       45.8       +3.6      +8.5       0.15      1.79
      bound pointer       44.0       +1.8      +4.3       0.14      1.83
   fixed device map       42.1       -0.1      -0.3       0.11      1.77
fixed device no-map       44.2       +2.0      +4.8       0.15      1.83
```

### fp3.txt
```
## round=1 commit=pf1 load=3.68 10.37 13.19
iters=1000 batches=9 unit=ns/call; no_gpu=True
mode=callback
callback_count=9100 (100 warmup + 9000 unique misses)
INTJ cold compile callback + cache  median_ns=401.3 samples_ns=[485.735, 492.824, 382.298, 617.4, 366.287, 379.752, 388.817, 401.332, 869.921]
INTJ hot call (no callback)         median_ns=37.9 samples_ns=[40.045, 39.009, 37.576, 38.03, 38.594, 37.524, 37.665, 37.673, 37.938]
INTJ 3-tensor host-only nop         median_ns=33.8 samples_ns=[36.073, 34.04, 33.753, 33.824, 33.776, 33.64, 33.713, 33.721, 33.669]
mode=kwargs
INTJ direct positional              median_ns=64.3 samples_ns=[65.866, 64.328, 63.985, 63.949, 63.96, 71.941, 64.648, 64.372, 64.305]
INTJ adapter positional             median_ns=89.1 samples_ns=[90.399, 89.777, 89.07, 88.822, 89.369, 88.921, 88.112, 87.996, 91.046]
INTJ adapter kwargs                 median_ns=105.2 samples_ns=[105.977, 107.077, 105.154, 104.705, 104.63, 104.916, 104.557, 106.23, 105.848]
INTJ adapter defaults               median_ns=87.9 samples_ns=[88.198, 88.101, 87.932, 87.249, 88.201, 87.579, 88.252, 87.744, 87.426]
INTJ FFI wrapper positional         median_ns=100.9 samples_ns=[102.68, 104.516, 101.722, 99.068, 100.913, 105.103, 100.716, 99.472, 100.067]
INTJ FFI wrapper kwargs             median_ns=125.5 samples_ns=[122.888, 126.471, 124.589, 125.507, 125.292, 124.672, 126.152, 125.907, 126.66]
INTJ native kwargs: unsupported (verified TypeError)
mode=dataclass
INTJ pair direct                    median_ns=67.1 samples_ns=[69.683, 67.145, 67.475, 67.186, 66.784, 66.89, 66.758, 66.252, 72.038]
INTJ pair prebuilt *tuple           median_ns=138.0 samples_ns=[139.666, 140.473, 138.249, 137.519, 137.993, 135.168, 139.953, 135.853, 135.478]
FFI unpack Pair only                median_ns=163.1 samples_ns=[161.531, 162.04, 162.213, 163.323, 166.8, 163.758, 162.578, 165.038, 163.054]
INTJ pair manual unpack             median_ns=171.2 samples_ns=[173.229, 175.659, 171.17, 172.005, 171.231, 169.748, 171.46, 167.626, 167.138]
INTJ pair FFI unpack                median_ns=313.8 samples_ns=[309.659, 315.947, 312.618, 313.753, 313.775, 316.44, 311.385, 314.324, 314.837]
INTJ pair stdlib astuple            median_ns=1140.5 samples_ns=[1131.973, 1141.227, 1140.453, 1136.178, 1142.597, 1135.338, 1134.995, 1151.177, 1145.174]
INTJ config direct                  median_ns=78.5 samples_ns=[78.188, 78.932, 78.795, 79.778, 93.592, 78.467, 78.259, 78.255, 78.417]
INTJ config FFI unpack              median_ns=317.4 samples_ns=[315.874, 323.61, 317.694, 317.424, 315.463, 317.468, 313.584, 316.136, 318.327]
INTJ native dataclass unpack: unsupported (verified TypeError)
## round=1 commit=pf3 load=3.70 10.26 13.14
iters=1000 batches=9 unit=ns/call; no_gpu=True
mode=callback
callback_count=9100 (100 warmup + 9000 unique misses)
INTJ cold compile callback + cache  median_ns=396.6 samples_ns=[458.841, 488.503, 376.758, 572.233, 358.7, 376.868, 381.652, 396.589, 763.46]
INTJ hot call (no callback)         median_ns=37.3 samples_ns=[39.998, 37.318, 37.411, 36.654, 36.919, 37.348, 37.272, 36.711, 36.174]
INTJ 3-tensor host-only nop         median_ns=33.6 samples_ns=[35.922, 33.783, 33.661, 33.624, 33.425, 33.39, 33.385, 41.757, 33.551]
mode=kwargs
INTJ direct positional              median_ns=64.2 samples_ns=[75.339, 64.006, 64.218, 64.265, 64.122, 64.233, 63.948, 64.358, 64.333]
INTJ adapter positional             median_ns=91.6 samples_ns=[94.465, 91.565, 91.772, 91.819, 94.656, 91.492, 90.271, 89.659, 90.535]
INTJ adapter kwargs                 median_ns=106.8 samples_ns=[108.832, 108.619, 106.17, 105.858, 105.594, 109.123, 106.821, 107.991, 106.701]
INTJ adapter defaults               median_ns=87.9 samples_ns=[89.684, 88.96, 88.267, 87.753, 87.815, 87.91, 90.76, 87.218, 87.125]
INTJ FFI wrapper positional         median_ns=104.8 samples_ns=[105.758, 104.864, 105.238, 103.488, 104.805, 104.222, 103.189, 110.28, 104.571]
INTJ FFI wrapper kwargs             median_ns=127.4 samples_ns=[126.723, 127.37, 128.645, 128.181, 127.097, 127.391, 133.034, 127.25, 126.453]
INTJ native kwargs: unsupported (verified TypeError)
mode=dataclass
INTJ pair direct                    median_ns=66.5 samples_ns=[69.051, 66.475, 66.173, 72.336, 66.438, 66.516, 66.493, 66.415, 66.362]
INTJ pair prebuilt *tuple           median_ns=138.0 samples_ns=[138.868, 137.997, 139.638, 140.9, 141.986, 135.259, 135.114, 136.498, 137.271]
FFI unpack Pair only                median_ns=162.0 samples_ns=[163.911, 166.809, 161.018, 161.928, 160.624, 161.981, 161.768, 165.847, 181.903]
INTJ pair manual unpack             median_ns=172.5 samples_ns=[172.509, 172.463, 172.449, 171.741, 174.995, 178.759, 169.881, 178.072, 169.641]
INTJ pair FFI unpack                median_ns=318.8 samples_ns=[330.746, 323.416, 317.154, 337.755, 311.103, 308.001, 321.83, 310.727, 318.819]
INTJ pair stdlib astuple            median_ns=1166.2 samples_ns=[1166.183, 1169.291, 1158.342, 1162.074, 1162.907, 1171.479, 1158.634, 1176.456, 1170.837]
INTJ config direct                  median_ns=79.8 samples_ns=[79.832, 79.755, 80.868, 82.202, 79.514, 79.578, 80.269, 79.627, 83.167]
INTJ config FFI unpack              median_ns=313.1 samples_ns=[312.518, 313.136, 319.324, 314.998, 312.238, 327.99, 310.731, 318.289, 308.204]
INTJ native dataclass unpack: unsupported (verified TypeError)
## round=2 commit=pf1 load=3.56 10.12 13.08
iters=1000 batches=9 unit=ns/call; no_gpu=True
mode=callback
callback_count=9100 (100 warmup + 9000 unique misses)
INTJ cold compile callback + cache  median_ns=395.6 samples_ns=[445.441, 494.245, 383.651, 1567.792, 375.256, 379.62, 392.974, 395.588, 2674.527]
INTJ hot call (no callback)         median_ns=38.1 samples_ns=[41.144, 38.47, 37.873, 37.844, 38.061, 38.163, 37.994, 37.564, 38.296]
INTJ 3-tensor host-only nop         median_ns=33.8 samples_ns=[36.163, 34.065, 33.875, 33.907, 33.844, 33.775, 33.785, 33.81, 33.725]
mode=kwargs
INTJ direct positional              median_ns=64.3 samples_ns=[65.303, 64.869, 63.821, 64.149, 69.951, 65.206, 63.7, 64.321, 63.468]
INTJ adapter positional             median_ns=90.7 samples_ns=[89.995, 92.148, 90.721, 91.783, 89.628, 90.675, 89.19, 100.202, 90.79]
INTJ adapter kwargs                 median_ns=108.8 samples_ns=[106.275, 108.847, 108.774, 106.798, 105.199, 109.595, 104.568, 116.766, 108.857]
INTJ adapter defaults               median_ns=88.2 samples_ns=[88.427, 89.028, 87.663, 88.527, 87.377, 88.98, 87.697, 88.187, 87.243]
INTJ FFI wrapper positional         median_ns=102.4 samples_ns=[107.834, 102.484, 100.512, 102.409, 101.289, 104.664, 101.304, 102.44, 102.438]
INTJ FFI wrapper kwargs             median_ns=126.2 samples_ns=[129.292, 126.237, 125.15, 126.698, 124.791, 128.472, 124.398, 125.377, 129.075]
INTJ native kwargs: unsupported (verified TypeError)
mode=dataclass
INTJ pair direct                    median_ns=66.8 samples_ns=[68.674, 66.85, 66.766, 66.769, 72.339, 66.129, 65.226, 65.931, 65.199]
INTJ pair prebuilt *tuple           median_ns=138.0 samples_ns=[137.903, 140.202, 137.114, 139.886, 142.25, 139.899, 134.974, 138.028, 136.439]
FFI unpack Pair only                median_ns=162.1 samples_ns=[162.648, 164.659, 165.532, 162.085, 161.524, 162.116, 160.071, 161.849, 168.371]
INTJ pair manual unpack             median_ns=172.3 samples_ns=[172.595, 172.35, 169.399, 172.143, 170.557, 176.54, 169.527, 172.968, 173.601]
INTJ pair FFI unpack                median_ns=317.0 samples_ns=[322.786, 316.849, 313.848, 318.087, 320.217, 316.98, 315.316, 322.79, 313.633]
INTJ pair stdlib astuple            median_ns=1151.1 samples_ns=[1193.042, 1199.759, 1139.809, 1151.104, 1136.859, 1155.356, 1139.219, 1157.171, 1140.366]
INTJ config direct                  median_ns=78.8 samples_ns=[78.876, 79.504, 79.527, 78.779, 77.871, 78.535, 77.741, 79.293, 77.74]
INTJ config FFI unpack              median_ns=318.3 samples_ns=[314.166, 317.809, 318.298, 322.214, 313.775, 320.343, 319.189, 319.609, 314.807]
INTJ native dataclass unpack: unsupported (verified TypeError)
## round=2 commit=pf3 load=3.32 9.85 12.96
iters=1000 batches=9 unit=ns/call; no_gpu=True
mode=callback
callback_count=9100 (100 warmup + 9000 unique misses)
INTJ cold compile callback + cache  median_ns=401.9 samples_ns=[442.467, 462.828, 380.96, 948.172, 362.348, 380.885, 384.424, 401.911, 1824.613]
INTJ hot call (no callback)         median_ns=36.7 samples_ns=[38.897, 37.114, 36.652, 36.755, 36.661, 35.856, 36.646, 36.655, 37.131]
INTJ 3-tensor host-only nop         median_ns=33.5 samples_ns=[35.911, 33.687, 33.466, 33.563, 33.45, 33.443, 33.414, 33.381, 33.439]
mode=kwargs
INTJ direct positional              median_ns=65.3 samples_ns=[66.547, 65.854, 64.563, 65.267, 64.626, 65.311, 64.421, 83.399, 64.611]
INTJ adapter positional             median_ns=90.0 samples_ns=[90.986, 92.132, 91.314, 90.079, 88.507, 89.893, 88.842, 90.023, 88.99]
INTJ adapter kwargs                 median_ns=107.0 samples_ns=[122.96, 108.021, 107.508, 107.948, 106.222, 106.998, 105.956, 106.909, 104.946]
INTJ adapter defaults               median_ns=86.8 samples_ns=[90.944, 92.645, 86.448, 86.93, 86.449, 86.799, 86.774, 87.451, 86.728]
INTJ FFI wrapper positional         median_ns=101.8 samples_ns=[103.63, 114.482, 101.786, 102.334, 100.605, 102.552, 99.92, 101.766, 100.742]
INTJ FFI wrapper kwargs             median_ns=126.1 samples_ns=[126.148, 127.3, 127.581, 126.501, 125.698, 126.588, 122.798, 124.784, 122.977]
INTJ native kwargs: unsupported (verified TypeError)
mode=dataclass
INTJ pair direct                    median_ns=66.5 samples_ns=[69.819, 67.599, 65.74, 67.204, 65.382, 66.463, 66.326, 79.187, 65.753]
INTJ pair prebuilt *tuple           median_ns=138.4 samples_ns=[135.711, 138.559, 137.133, 141.234, 137.135, 138.671, 138.383, 138.842, 136.25]
FFI unpack Pair only                median_ns=164.8 samples_ns=[165.752, 166.165, 164.639, 170.825, 164.763, 165.092, 163.005, 163.678, 162.427]
INTJ pair manual unpack             median_ns=173.4 samples_ns=[177.805, 173.417, 171.414, 174.178, 170.916, 173.362, 180.508, 174.372, 170.131]
INTJ pair FFI unpack                median_ns=315.2 samples_ns=[313.703, 317.829, 315.156, 314.536, 317.889, 316.78, 312.672, 318.37, 313.605]
INTJ pair stdlib astuple            median_ns=1145.7 samples_ns=[1193.778, 1198.542, 1136.736, 1154.982, 1134.922, 1156.323, 1139.642, 1145.721, 1142.301]
INTJ config direct                  median_ns=79.3 samples_ns=[78.583, 80.159, 79.413, 79.481, 79.255, 79.562, 79.257, 78.828, 79.253]
INTJ config FFI unpack              median_ns=317.2 samples_ns=[321.408, 321.642, 315.51, 322.455, 313.428, 317.238, 323.09, 316.862, 315.105]
INTJ native dataclass unpack: unsupported (verified TypeError)
## round=3 commit=pf1 load=3.22 9.72 12.90
iters=1000 batches=9 unit=ns/call; no_gpu=True
mode=callback
callback_count=9100 (100 warmup + 9000 unique misses)
INTJ cold compile callback + cache  median_ns=397.0 samples_ns=[453.198, 501.476, 381.77, 1185.458, 367.28, 375.852, 390.183, 397.025, 1991.691]
INTJ hot call (no callback)         median_ns=38.2 samples_ns=[40.743, 38.874, 37.986, 37.518, 38.113, 37.869, 38.219, 38.594, 44.741]
INTJ 3-tensor host-only nop         median_ns=33.6 samples_ns=[36.037, 33.76, 33.57, 33.462, 33.437, 33.509, 33.599, 33.544, 33.638]
mode=kwargs
INTJ direct positional              median_ns=65.2 samples_ns=[66.219, 65.418, 64.539, 65.374, 64.876, 65.225, 64.43, 65.199, 64.462]
INTJ adapter positional             median_ns=91.1 samples_ns=[91.137, 90.934, 94.929, 92.098, 90.665, 91.599, 89.029, 91.397, 88.276]
INTJ adapter kwargs                 median_ns=105.6 samples_ns=[106.714, 109.248, 105.03, 113.657, 102.867, 105.619, 104.748, 106.765, 105.19]
INTJ adapter defaults               median_ns=89.6 samples_ns=[88.183, 89.116, 88.793, 89.493, 117.112, 92.139, 90.99, 92.104, 89.593]
INTJ FFI wrapper positional         median_ns=103.0 samples_ns=[102.474, 107.733, 102.362, 103.435, 101.573, 107.949, 102.146, 103.123, 103.034]
INTJ FFI wrapper kwargs             median_ns=128.1 samples_ns=[128.089, 128.989, 127.491, 131.963, 130.066, 125.901, 123.758, 125.989, 128.711]
INTJ native kwargs: unsupported (verified TypeError)
mode=dataclass
INTJ pair direct                    median_ns=65.7 samples_ns=[68.886, 65.831, 65.397, 65.696, 65.272, 65.793, 65.607, 65.727, 65.376]
INTJ pair prebuilt *tuple           median_ns=139.1 samples_ns=[137.379, 143.013, 135.882, 139.65, 134.808, 139.09, 136.555, 140.452, 140.167]
FFI unpack Pair only                median_ns=162.7 samples_ns=[161.465, 165.221, 161.711, 165.552, 162.564, 170.247, 163.788, 162.737, 161.921]
INTJ pair manual unpack             median_ns=171.8 samples_ns=[168.54, 171.831, 173.23, 172.047, 168.122, 171.301, 169.579, 172.273, 174.303]
INTJ pair FFI unpack                median_ns=316.5 samples_ns=[314.773, 317.158, 327.502, 316.469, 307.569, 320.008, 312.092, 314.693, 319.498]
INTJ pair stdlib astuple            median_ns=1158.1 samples_ns=[1204.506, 1198.401, 1170.684, 1167.932, 1155.317, 1158.093, 1142.246, 1157.408, 1148.419]
INTJ config direct                  median_ns=80.9 samples_ns=[78.896, 81.292, 89.237, 81.366, 78.739, 80.961, 78.465, 80.857, 78.401]
INTJ config FFI unpack              median_ns=319.5 samples_ns=[314.605, 319.644, 313.238, 319.48, 321.475, 328.184, 318.47, 331.597, 313.896]
INTJ native dataclass unpack: unsupported (verified TypeError)
## round=3 commit=pf3 load=3.20 9.61 12.84
iters=1000 batches=9 unit=ns/call; no_gpu=True
mode=callback
callback_count=9100 (100 warmup + 9000 unique misses)
INTJ cold compile callback + cache  median_ns=396.7 samples_ns=[437.479, 456.427, 381.147, 974.385, 360.619, 382.055, 383.71, 396.68, 1905.135]
INTJ hot call (no callback)         median_ns=38.5 samples_ns=[40.247, 38.353, 37.924, 38.456, 38.507, 45.826, 38.003, 38.45, 37.819]
INTJ 3-tensor host-only nop         median_ns=33.5 samples_ns=[35.89, 33.573, 33.476, 33.445, 33.569, 33.538, 33.468, 33.402, 33.368]
mode=kwargs
INTJ direct positional              median_ns=64.8 samples_ns=[65.899, 65.273, 64.186, 64.908, 64.268, 64.946, 64.294, 64.843, 63.814]
INTJ adapter positional             median_ns=90.9 samples_ns=[97.021, 92.72, 89.844, 91.787, 90.98, 90.473, 90.278, 90.919, 90.566]
INTJ adapter kwargs                 median_ns=108.1 samples_ns=[108.079, 111.716, 109.086, 110.122, 107.905, 109.474, 106.891, 108.067, 106.703]
INTJ adapter defaults               median_ns=89.5 samples_ns=[89.49, 90.735, 96.149, 91.047, 87.661, 89.194, 88.961, 89.865, 87.836]
INTJ FFI wrapper positional         median_ns=101.7 samples_ns=[102.777, 103.515, 100.723, 106.982, 101.424, 102.543, 100.877, 101.705, 101.181]
INTJ FFI wrapper kwargs             median_ns=127.0 samples_ns=[126.153, 125.624, 125.806, 153.274, 124.363, 127.418, 126.97, 128.097, 127.636]
INTJ native kwargs: unsupported (verified TypeError)
mode=dataclass
INTJ pair direct                    median_ns=65.6 samples_ns=[68.147, 66.25, 65.239, 65.698, 65.152, 65.583, 65.124, 65.82, 65.087]
INTJ pair prebuilt *tuple           median_ns=139.3 samples_ns=[142.553, 139.872, 135.954, 139.345, 138.439, 141.029, 136.726, 142.492, 135.362]
FFI unpack Pair only                median_ns=168.2 samples_ns=[166.5, 167.138, 165.942, 166.186, 170.045, 168.885, 169.603, 170.346, 168.175]
INTJ pair manual unpack             median_ns=170.8 samples_ns=[170.392, 178.343, 169.024, 171.781, 170.33, 173.525, 170.806, 178.757, 168.943]
INTJ pair FFI unpack                median_ns=316.1 samples_ns=[312.843, 316.408, 319.291, 316.082, 315.644, 320.361, 313.479, 315.441, 318.49]
INTJ pair stdlib astuple            median_ns=1152.7 samples_ns=[1197.768, 1207.102, 1149.004, 1154.677, 1149.543, 1160.658, 1150.083, 1152.675, 1143.975]
INTJ config direct                  median_ns=80.2 samples_ns=[83.076, 80.404, 80.248, 80.217, 79.662, 79.83, 79.846, 80.434, 79.327]
INTJ config FFI unpack              median_ns=320.1 samples_ns=[320.737, 320.05, 318.184, 323.096, 315.749, 321.424, 316.718, 327.91, 320.071]
INTJ native dataclass unpack: unsupported (verified TypeError)
## round=4 commit=pf1 load=3.18 9.50 12.79
iters=1000 batches=9 unit=ns/call; no_gpu=True
mode=callback
callback_count=9100 (100 warmup + 9000 unique misses)
INTJ cold compile callback + cache  median_ns=411.0 samples_ns=[448.815, 523.38, 380.017, 1304.893, 382.868, 396.247, 405.092, 411.035, 2227.865]
INTJ hot call (no callback)         median_ns=38.1 samples_ns=[39.808, 38.674, 37.62, 38.261, 38.127, 37.835, 37.893, 41.461, 37.679]
INTJ 3-tensor host-only nop         median_ns=34.4 samples_ns=[37.293, 34.793, 34.154, 34.529, 34.092, 34.062, 34.468, 34.051, 34.449]
mode=kwargs
INTJ direct positional              median_ns=65.2 samples_ns=[66.928, 65.716, 65.169, 65.391, 64.933, 65.186, 64.792, 65.137, 64.876]
INTJ adapter positional             median_ns=90.9 samples_ns=[89.198, 94.228, 90.119, 91.752, 90.466, 91.248, 90.701, 92.036, 90.912]
INTJ adapter kwargs                 median_ns=110.3 samples_ns=[110.572, 110.502, 115.719, 112.518, 109.575, 110.212, 108.875, 110.259, 107.376]
INTJ adapter defaults               median_ns=88.3 samples_ns=[88.289, 89.614, 97.434, 89.821, 87.491, 89.088, 87.549, 87.812, 87.508]
INTJ FFI wrapper positional         median_ns=103.2 samples_ns=[103.474, 105.68, 102.421, 103.177, 105.259, 102.467, 102.219, 104.514, 102.056]
INTJ FFI wrapper kwargs             median_ns=124.1 samples_ns=[123.644, 124.601, 124.095, 128.663, 123.471, 124.929, 124.982, 123.833, 121.343]
INTJ native kwargs: unsupported (verified TypeError)
mode=dataclass
INTJ pair direct                    median_ns=70.6 samples_ns=[72.864, 70.85, 69.973, 70.558, 70.387, 71.241, 70.422, 70.731, 70.269]
INTJ pair prebuilt *tuple           median_ns=139.8 samples_ns=[152.039, 151.091, 136.111, 140.22, 134.079, 139.836, 135.053, 143.054, 136.757]
FFI unpack Pair only                median_ns=166.6 samples_ns=[164.604, 166.912, 162.934, 166.579, 167.586, 167.964, 164.589, 167.446, 164.182]
INTJ pair manual unpack             median_ns=170.9 samples_ns=[170.906, 180.692, 169.487, 175.033, 168.76, 176.212, 168.518, 180.051, 169.046]
INTJ pair FFI unpack                median_ns=315.4 samples_ns=[311.766, 317.18, 323.314, 313.728, 311.639, 320.893, 312.449, 316.016, 315.365]
INTJ pair stdlib astuple            median_ns=1157.6 samples_ns=[1186.204, 1194.16, 1142.578, 1172.958, 1141.045, 1157.599, 1137.46, 1167.045, 1150.891]
INTJ config direct                  median_ns=81.2 samples_ns=[84.65, 86.627, 82.271, 81.287, 80.684, 81.242, 80.647, 81.115, 80.694]
INTJ config FFI unpack              median_ns=316.8 samples_ns=[314.214, 322.702, 312.722, 319.916, 316.803, 317.631, 311.959, 328.184, 315.372]
INTJ native dataclass unpack: unsupported (verified TypeError)
## round=4 commit=pf3 load=3.17 9.39 12.74
iters=1000 batches=9 unit=ns/call; no_gpu=True
mode=callback
callback_count=9100 (100 warmup + 9000 unique misses)
INTJ cold compile callback + cache  median_ns=401.0 samples_ns=[440.612, 458.89, 373.093, 922.989, 358.288, 378.178, 381.86, 400.962, 1809.936]
INTJ hot call (no callback)         median_ns=37.5 samples_ns=[39.056, 39.829, 40.101, 38.898, 37.485, 37.361, 36.832, 37.229, 37.225]
INTJ 3-tensor host-only nop         median_ns=33.8 samples_ns=[36.402, 33.908, 33.865, 33.845, 33.765, 33.673, 33.652, 33.654, 33.655]
mode=kwargs
INTJ direct positional              median_ns=65.1 samples_ns=[65.87, 65.933, 63.792, 65.086, 63.712, 65.29, 64.13, 64.879, 80.982]
INTJ adapter positional             median_ns=90.4 samples_ns=[90.197, 90.979, 89.287, 90.391, 90.385, 91.988, 90.44, 91.903, 90.676]
INTJ adapter kwargs                 median_ns=111.5 samples_ns=[108.544, 128.645, 111.537, 112.614, 108.148, 112.729, 108.776, 111.71, 108.51]
INTJ adapter defaults               median_ns=87.7 samples_ns=[88.198, 101.089, 93.149, 89.753, 87.031, 87.615, 87.034, 87.72, 87.177]
INTJ FFI wrapper positional         median_ns=104.0 samples_ns=[101.463, 107.967, 104.007, 103.265, 102.59, 104.23, 103.826, 105.41, 104.694]
INTJ FFI wrapper kwargs             median_ns=126.4 samples_ns=[124.581, 125.228, 128.985, 128.51, 125.658, 126.833, 125.374, 126.662, 126.427]
INTJ native kwargs: unsupported (verified TypeError)
mode=dataclass
INTJ pair direct                    median_ns=66.3 samples_ns=[68.412, 66.575, 65.704, 66.343, 65.448, 66.463, 65.417, 66.164, 76.327]
INTJ pair prebuilt *tuple           median_ns=137.9 samples_ns=[137.892, 140.434, 136.759, 139.041, 135.528, 139.367, 139.927, 137.718, 136.838]
FFI unpack Pair only                median_ns=162.4 samples_ns=[161.052, 164.315, 164.814, 164.01, 164.995, 161.717, 161.325, 162.36, 160.712]
INTJ pair manual unpack             median_ns=173.1 samples_ns=[173.703, 172.95, 171.405, 173.055, 171.464, 174.123, 181.261, 177.129, 172.898]
INTJ pair FFI unpack                median_ns=312.0 samples_ns=[309.244, 317.297, 311.086, 312.022, 309.089, 315.034, 311.859, 313.544, 315.962]
INTJ pair stdlib astuple            median_ns=1161.5 samples_ns=[1193.751, 1203.998, 1173.903, 1150.754, 1143.082, 1152.508, 1146.442, 1172.25, 1161.535]
INTJ config direct                  median_ns=79.8 samples_ns=[79.002, 80.58, 79.62, 80.696, 79.767, 80.185, 79.606, 80.14, 79.459]
INTJ config FFI unpack              median_ns=316.1 samples_ns=[317.911, 316.314, 311.813, 317.101, 314.969, 315.638, 316.531, 316.137, 313.575]
INTJ native dataclass unpack: unsupported (verified TypeError)
```

### fc.txt
```
## round=1 commit=b4e9f1f load=2.17 3.36 7.36 gpu=0,0,0,0,0,0,0,0,
iters=1000 batches=9 unit=us/call (host-only, no timed sync)
FFI packed nop             median=0.1453 samples=[0.149243, 0.143985, 0.144132, 0.145622, 0.145294, 0.145988, 0.14720599999999998, 0.142619, 0.14440799999999998]
FFI typed nop              median=0.1489 samples=[0.148856, 0.15201699999999999, 0.147101, 0.151221, 0.148802, 0.15267599999999998, 0.148505, 0.15104499999999998, 0.14687]
FFI empty kernel           median=3.8540 samples=[3.663087, 4.005917, 3.9195569999999997, 3.85405, 3.750596, 3.915864, 3.785597, 3.904298, 3.778347]
INTJ empty kernel          median=3.5465 samples=[2.992064, 3.691566, 3.621947, 3.546523, 3.435035, 3.5709470000000003, 3.4660520000000004, 3.5605949999999997, 3.459794]
INTJ fixed-device kernel   median=3.5204 samples=[3.282658, 3.532756, 3.665167, 3.340359, 3.520415, 3.340898, 3.532842, 3.334774, 3.544412]
FFI packed nop mixed       median=0.1731 samples=[0.173952, 0.171715, 0.17302099999999998, 0.17297800000000002, 0.174098, 0.176263, 0.17380400000000001, 0.17203, 0.173144]
FFI typed nop mixed        median=0.1765 samples=[0.17914, 0.176538, 0.179777, 0.177971, 0.17519200000000001, 0.177985, 0.176292, 0.17635499999999998, 0.175163]
FFI mixed kernel           median=3.8513 samples=[3.8513270000000004, 4.07934, 3.79608, 3.937503, 3.748612, 3.9749250000000003, 3.8038890000000003, 3.940921, 3.8322109999999996]
INTJ mixed kernel          median=3.5237 samples=[3.648198, 3.695238, 3.5099549999999997, 3.53531, 3.422248, 3.500147, 3.523695, 3.5717269999999997, 3.4933229999999997]
INTJ fixed mixed kernel    median=3.5239 samples=[3.711833, 3.768872, 3.562408, 3.58453, 3.485544, 3.476592, 3.4846570000000003, 3.5238560000000003, 3.494837]
## round=1 commit=pf3 load=2.44 3.38 7.32 gpu=0,0,0,0,0,0,0,0,
iters=1000 batches=9 unit=us/call (host-only, no timed sync)
FFI packed nop             median=0.1442 samples=[0.14896, 0.144184, 0.143353, 0.14560800000000002, 0.14473, 0.144074, 0.14471799999999999, 0.142666, 0.14296]
FFI typed nop              median=0.1482 samples=[0.14853899999999998, 0.148842, 0.148077, 0.151256, 0.148231, 0.147835, 0.146117, 0.152468, 0.14821299999999998]
FFI empty kernel           median=3.2920 samples=[3.589673, 3.456446, 3.456261, 3.2615030000000003, 3.239861, 3.289946, 3.288153, 3.291978, 3.3058780000000003]
INTJ empty kernel          median=2.9036 samples=[2.9336640000000003, 3.065789, 3.0847629999999997, 2.88594, 2.8604520000000004, 2.910521, 2.9017049999999998, 2.902134, 2.903579]
INTJ fixed-device kernel   median=2.8988 samples=[3.006116, 3.0955749999999997, 3.067905, 2.907871, 2.87125, 2.887113, 2.890067, 2.898071, 2.898762]
FFI packed nop mixed       median=0.1729 samples=[0.17067, 0.17515199999999997, 0.177862, 0.172854, 0.174761, 0.171961, 0.174329, 0.168339, 0.172509]
FFI typed nop mixed        median=0.1764 samples=[0.17610800000000001, 0.176572, 0.174104, 0.178013, 0.176338, 0.177445, 0.176433, 0.181387, 0.173201]
FFI mixed kernel           median=3.3271 samples=[3.429649, 3.50963, 3.314087, 3.334743, 3.251922, 3.317364, 3.330656, 3.32714, 3.3165839999999998]
INTJ mixed kernel          median=2.9305 samples=[3.097025, 3.103801, 2.914968, 2.931849, 2.8610990000000003, 2.859583, 2.930505, 2.933652, 2.906506]
INTJ fixed mixed kernel    median=2.8792 samples=[3.1062109999999996, 3.104812, 2.958836, 2.959319, 2.868195, 2.852028, 2.85459, 2.879217, 2.8618609999999998]
## round=2 commit=b4e9f1f load=2.58 3.36 7.25 gpu=0,0,0,0,0,0,0,0,
iters=1000 batches=9 unit=us/call (host-only, no timed sync)
FFI packed nop             median=0.1457 samples=[0.14996199999999998, 0.143868, 0.14355099999999998, 0.150068, 0.145739, 0.145744, 0.14496799999999999, 0.14413399999999998, 0.148434]
FFI typed nop              median=0.1500 samples=[0.149655, 0.14949199999999999, 0.150034, 0.151047, 0.14704699999999998, 0.151796, 0.151367, 0.150218, 0.147999]
FFI empty kernel           median=3.7825 samples=[3.615597, 3.953111, 3.7824560000000003, 4.432206, 3.509988, 4.338122, 3.56508, 4.373011, 3.5814969999999997]
INTJ empty kernel          median=3.5900 samples=[3.137156, 3.590036, 3.709184, 3.423011, 3.873647, 3.516223, 3.9011080000000002, 3.489353, 3.913647]
INTJ fixed-device kernel   median=3.3498 samples=[3.437922, 3.5732269999999997, 3.651135, 3.159081, 3.5337240000000003, 3.168, 3.349843, 3.183165, 3.340023]
FFI packed nop mixed       median=0.1735 samples=[0.172937, 0.171117, 0.173213, 0.169497, 0.173961, 0.173641, 0.173454, 0.173643, 0.17648599999999998]
FFI typed nop mixed        median=0.1755 samples=[0.174708, 0.174018, 0.176861, 0.17518999999999998, 0.17554599999999998, 0.178568, 0.176955, 0.175927, 0.17402600000000001]
FFI mixed kernel           median=3.7365 samples=[3.736459, 3.985272, 3.7164050000000004, 3.884574, 3.667188, 3.8581280000000002, 3.6731550000000004, 4.484656, 3.6704749999999997]
INTJ mixed kernel          median=3.5000 samples=[4.109478, 3.654537, 3.542281, 3.500038, 3.4930630000000003, 3.4751819999999998, 3.357736, 3.54825, 3.393119]
INTJ fixed mixed kernel    median=3.4998 samples=[3.622732, 3.652791, 3.5109749999999997, 3.500245, 3.4337530000000003, 3.421857, 3.49985, 3.4928939999999997, 3.464272]
## round=2 commit=pf3 load=2.58 3.36 7.25 gpu=0,0,0,0,0,0,0,0,
iters=1000 batches=9 unit=us/call (host-only, no timed sync)
FFI packed nop             median=0.1442 samples=[0.150597, 0.14349, 0.141851, 0.144161, 0.14450200000000002, 0.144204, 0.143798, 0.146332, 0.14390799999999998]
FFI typed nop              median=0.1488 samples=[0.14849700000000002, 0.154149, 0.14524299999999998, 0.153899, 0.148835, 0.153368, 0.14940799999999999, 0.148832, 0.146136]
FFI empty kernel           median=3.3152 samples=[3.564955, 3.409539, 3.404404, 3.2258, 3.209309, 3.339313, 3.176406, 3.3152109999999997, 3.171417]
INTJ empty kernel          median=2.9701 samples=[3.049196, 3.104279, 3.1170549999999997, 2.910993, 2.888334, 2.978729, 2.954256, 2.9700889999999998, 2.967331]
INTJ fixed-device kernel   median=2.9026 samples=[3.029962, 3.1225810000000003, 3.09484, 2.9356210000000003, 2.902551, 2.802218, 2.8075210000000004, 2.8054270000000003, 2.891916]
FFI packed nop mixed       median=0.1735 samples=[0.17349199999999998, 0.171961, 0.175268, 0.172514, 0.17397900000000002, 0.17091399999999998, 0.17529599999999998, 0.171222, 0.173783]
FFI typed nop mixed        median=0.1764 samples=[0.175746, 0.17638900000000002, 0.175506, 0.17784899999999998, 0.17484899999999998, 0.178893, 0.17372100000000001, 0.177596, 0.179127]
FFI mixed kernel           median=3.2936 samples=[3.4110189999999996, 3.474602, 3.290273, 3.2829789999999996, 3.244051, 3.293636, 3.315776, 3.3861280000000002, 3.288916]
INTJ mixed kernel          median=2.9451 samples=[3.3755949999999997, 3.173414, 2.945118, 2.959483, 2.882104, 2.890105, 2.8271610000000003, 2.974026, 2.811279]
INTJ fixed mixed kernel    median=2.9230 samples=[3.1454, 3.1316990000000002, 3.004389, 3.0036840000000002, 2.893306, 2.872119, 2.906015, 2.922982, 2.897571]
## round=3 commit=b4e9f1f load=2.69 3.37 7.24 gpu=0,0,0,0,0,0,0,0,
iters=1000 batches=9 unit=us/call (host-only, no timed sync)
FFI packed nop             median=0.1439 samples=[0.149561, 0.143655, 0.14484899999999998, 0.142475, 0.14259899999999998, 0.143338, 0.146984, 0.14386000000000002, 0.144547]
FFI typed nop              median=0.1478 samples=[0.148784, 0.15162799999999999, 0.147375, 0.148171, 0.147847, 0.147149, 0.145238, 0.147817, 0.145414]
FFI empty kernel           median=3.3066 samples=[3.554528, 3.375854, 3.364241, 3.3065900000000004, 3.129735, 3.226616, 3.1883139999999996, 3.324857, 3.198672]
INTJ empty kernel          median=2.9397 samples=[3.052928, 3.0560549999999997, 3.046941, 2.8116640000000004, 2.920709, 2.933203, 2.9149380000000003, 2.939674, 2.969679]
INTJ fixed-device kernel   median=2.8167 samples=[3.003781, 3.08189, 3.041128, 2.790431, 2.841521, 2.805752, 2.794113, 2.816654, 2.796401]
FFI packed nop mixed       median=0.1741 samples=[0.17382, 0.17443799999999998, 0.174684, 0.172162, 0.17915199999999998, 0.171451, 0.174929, 0.174125, 0.174121]
FFI typed nop mixed        median=0.1756 samples=[0.17648599999999998, 0.17743299999999998, 0.175646, 0.176637, 0.175009, 0.175998, 0.17405400000000001, 0.174843, 0.174985]
FFI mixed kernel           median=3.3262 samples=[3.3801930000000002, 3.407686, 3.216348, 3.326214, 3.15222, 3.379845, 3.23824, 3.3341570000000003, 3.236289]
INTJ mixed kernel          median=2.8384 samples=[3.126358, 3.083501, 2.825903, 2.8730569999999997, 2.772392, 2.813615, 2.838364, 2.9442, 2.828181]
INTJ fixed mixed kernel    median=2.8946 samples=[3.10024, 3.098496, 2.908962, 2.896474, 2.847648, 2.818286, 2.870595, 2.894627, 2.893762]
## round=3 commit=pf3 load=2.64 3.35 7.21 gpu=0,0,0,0,0,0,0,0,
iters=1000 batches=9 unit=us/call (host-only, no timed sync)
FFI packed nop             median=0.1449 samples=[0.148574, 0.144088, 0.144881, 0.145905, 0.145621, 0.149858, 0.144524, 0.143847, 0.144131]
FFI typed nop              median=0.1493 samples=[0.15826400000000002, 0.150198, 0.148905, 0.150596, 0.14926, 0.151083, 0.145867, 0.14921600000000002, 0.148064]
FFI empty kernel           median=3.3037 samples=[3.5176309999999997, 3.43982, 3.4610819999999998, 3.229166, 3.130701, 3.303658, 3.17471, 3.313661, 3.1737330000000004]
INTJ empty kernel          median=2.9507 samples=[3.065355, 3.054355, 3.065429, 2.836818, 2.9011060000000004, 2.933011, 2.953294, 2.867873, 2.950659]
INTJ fixed-device kernel   median=2.8034 samples=[2.9719949999999997, 3.077419, 3.061031, 2.8073110000000003, 2.764708, 2.8034, 2.79391, 2.802923, 2.79815]
FFI packed nop mixed       median=0.1749 samples=[0.174396, 0.174843, 0.177048, 0.17453, 0.174876, 0.172486, 0.175534, 0.176922, 0.175498]
FFI typed nop mixed        median=0.1786 samples=[0.177793, 0.18340399999999998, 0.173991, 0.18036000000000002, 0.17859999999999998, 0.179388, 0.180668, 0.178133, 0.177603]
FFI mixed kernel           median=3.3250 samples=[3.396922, 3.518935, 3.317599, 3.3250070000000003, 3.1394360000000003, 3.330619, 3.242311, 3.3341, 3.239305]
INTJ mixed kernel          median=2.8415 samples=[3.1033589999999998, 3.0749400000000002, 2.841478, 2.849154, 2.7522710000000004, 2.789315, 2.818894, 2.866315, 2.822362]
INTJ fixed mixed kernel    median=2.8395 samples=[3.106179, 3.096694, 2.885008, 2.86373, 2.8227849999999997, 2.786008, 2.8015369999999997, 2.839547, 2.817445]
## round=4 commit=b4e9f1f load=3.23 3.46 7.22 gpu=0,0,0,0,0,0,0,0,
iters=1000 batches=9 unit=us/call (host-only, no timed sync)
FFI packed nop             median=0.1456 samples=[0.152363, 0.14660800000000002, 0.148063, 0.141734, 0.144828, 0.14557900000000001, 0.142535, 0.14801499999999998, 0.143563]
FFI typed nop              median=0.1493 samples=[0.150278, 0.150755, 0.149999, 0.149273, 0.147234, 0.14828899999999998, 0.144876, 0.149901, 0.147386]
FFI empty kernel           median=3.3358 samples=[3.50494, 3.3868560000000003, 3.377, 3.184375, 3.163256, 3.338294, 3.178208, 3.335808, 3.176131]
INTJ empty kernel          median=3.0351 samples=[3.058118, 3.077754, 3.073522, 2.8762440000000002, 2.851384, 2.95798, 3.035125, 2.945087, 3.041484]
INTJ fixed-device kernel   median=2.8930 samples=[3.038916, 3.089878, 3.0639969999999996, 2.887915, 2.866107, 2.796845, 2.8930160000000003, 2.814896, 2.9001819999999996]
FFI packed nop mixed       median=0.1769 samples=[0.17597900000000002, 0.177132, 0.176851, 0.17336500000000002, 0.177458, 0.176864, 0.18025, 0.174811, 0.18]
FFI typed nop mixed        median=0.1777 samples=[0.175048, 0.179948, 0.176323, 0.175132, 0.17906, 0.178134, 0.177709, 0.178487, 0.176013]
FFI mixed kernel           median=3.2558 samples=[3.3775839999999997, 3.430151, 3.2299189999999998, 3.255795, 3.19219, 3.261078, 3.254833, 3.3394310000000003, 3.242951]
INTJ mixed kernel          median=2.8995 samples=[3.1469720000000003, 3.098119, 2.909349, 2.899512, 2.84829, 2.84988, 2.827976, 2.909514, 2.81982]
INTJ fixed mixed kernel    median=2.8641 samples=[3.1244560000000003, 3.111539, 2.931003, 2.934659, 2.864102, 2.836792, 2.854196, 2.854278, 2.832971]
## round=4 commit=pf3 load=3.45 3.50 7.22 gpu=0,0,0,0,0,0,0,0,
iters=1000 batches=9 unit=us/call (host-only, no timed sync)
FFI packed nop             median=0.1451 samples=[0.15203, 0.14803, 0.142088, 0.142348, 0.145146, 0.148726, 0.14435900000000002, 0.148988, 0.14462]
FFI typed nop              median=0.1510 samples=[0.155999, 0.150968, 0.14844200000000002, 0.149827, 0.15201499999999998, 0.15196600000000002, 0.147698, 0.15156899999999998, 0.147614]
FFI empty kernel           median=3.3221 samples=[3.567025, 3.3817060000000003, 3.389761, 3.166846, 3.1710010000000004, 3.3221410000000002, 3.228813, 3.328229, 3.226907]
INTJ empty kernel          median=2.9645 samples=[3.061478, 2.97967, 2.9755920000000002, 2.7864780000000002, 2.75406, 2.935121, 2.968663, 2.888183, 2.9645300000000003]
INTJ fixed-device kernel   median=2.8773 samples=[3.009507, 2.9975500000000004, 2.967835, 2.80227, 2.774046, 2.896032, 2.858627, 2.8773, 2.857246]
FFI packed nop mixed       median=0.1738 samples=[0.169858, 0.174238, 0.174315, 0.172274, 0.174637, 0.173517, 0.173764, 0.17216700000000001, 0.175579]
FFI typed nop mixed        median=0.1784 samples=[0.17260599999999998, 0.178363, 0.176315, 0.180165, 0.176768, 0.18123, 0.17680500000000002, 0.182405, 0.179318]
FFI mixed kernel           median=3.3324 samples=[3.433366, 3.445776, 3.24555, 3.261629, 3.209664, 3.360366, 3.332432, 3.349597, 3.316549]
INTJ mixed kernel          median=2.8451 samples=[3.001529, 2.993322, 2.814578, 2.819736, 2.781392, 2.800388, 2.845544, 2.881895, 2.845102]
INTJ fixed mixed kernel    median=2.8296 samples=[3.009411, 3.0029760000000003, 2.8293150000000002, 2.840938, 2.826884, 2.791111, 2.814694, 2.829623, 2.830731]
## round=5 commit=b4e9f1f load=3.45 3.50 7.22 gpu=0,0,0,0,0,0,0,0,
iters=1000 batches=9 unit=us/call (host-only, no timed sync)
FFI packed nop             median=0.1449 samples=[0.151118, 0.14723, 0.140904, 0.143711, 0.144863, 0.147007, 0.14510599999999998, 0.141587, 0.143927]
FFI typed nop              median=0.1497 samples=[0.147155, 0.15068399999999998, 0.14586500000000002, 0.14968299999999998, 0.153112, 0.151227, 0.148482, 0.15138, 0.146853]
FFI empty kernel           median=3.3417 samples=[3.567159, 3.404585, 3.391963, 3.190418, 3.192551, 3.349051, 3.189799, 3.3417269999999997, 3.192597]
INTJ empty kernel          median=2.9758 samples=[3.077458, 3.074293, 3.054598, 2.8614140000000003, 2.842875, 2.944747, 2.975833, 2.942353, 2.986547]
INTJ fixed-device kernel   median=2.8529 samples=[3.0099549999999997, 3.079084, 3.0574850000000002, 2.901735, 2.852874, 2.805144, 2.7923240000000003, 2.8199340000000004, 2.8230999999999997]
FFI packed nop mixed       median=0.1739 samples=[0.17583600000000002, 0.171263, 0.174049, 0.17530400000000002, 0.17394, 0.173617, 0.17726, 0.17330600000000002, 0.173724]
FFI typed nop mixed        median=0.1776 samples=[0.176907, 0.17901499999999998, 0.176351, 0.178812, 0.175845, 0.178895, 0.1787, 0.17688800000000002, 0.177554]
FFI mixed kernel           median=3.2627 samples=[3.394672, 3.442665, 3.2578110000000002, 3.262659, 3.217216, 3.291958, 3.261693, 3.3841170000000003, 3.244194]
INTJ mixed kernel          median=2.8898 samples=[3.1568449999999997, 3.0862489999999996, 2.895261, 2.889768, 2.85063, 2.841814, 2.835466, 2.9537139999999997, 2.824463]
INTJ fixed mixed kernel    median=2.9078 samples=[3.126484, 3.110531, 2.943794, 2.9473200000000004, 2.853696, 2.852192, 2.88822, 2.907787, 2.903659]
## round=5 commit=pf3 load=3.41 3.49 7.19 gpu=0,0,0,0,0,0,0,0,
iters=1000 batches=9 unit=us/call (host-only, no timed sync)
FFI packed nop             median=0.1444 samples=[0.150414, 0.14509200000000003, 0.140428, 0.14438, 0.143687, 0.141916, 0.148308, 0.144433, 0.143701]
FFI typed nop              median=0.1490 samples=[0.15424100000000002, 0.150451, 0.144875, 0.148423, 0.149048, 0.150441, 0.146348, 0.150377, 0.147234]
FFI empty kernel           median=3.3481 samples=[3.5867310000000003, 3.433861, 3.404627, 3.2241370000000003, 3.228034, 3.3941179999999997, 3.203801, 3.348083, 3.19262]
INTJ empty kernel          median=2.9768 samples=[3.0532660000000003, 3.120525, 3.114719, 2.942761, 2.903201, 3.00522, 2.957177, 2.976758, 2.97295]
INTJ fixed-device kernel   median=2.9190 samples=[3.2318200000000004, 3.127759, 3.111093, 2.936504, 2.9156999999999997, 2.844397, 2.80239, 2.820448, 2.918975]
FFI packed nop mixed       median=0.1738 samples=[0.173392, 0.173034, 0.17377, 0.170251, 0.174453, 0.173965, 0.174589, 0.17590199999999998, 0.17365]
FFI typed nop mixed        median=0.1764 samples=[0.178027, 0.177583, 0.178767, 0.174093, 0.175369, 0.17704499999999998, 0.176372, 0.176234, 0.175697]
FFI mixed kernel           median=3.2819 samples=[3.400423, 3.481821, 3.281935, 3.280597, 3.2283600000000003, 3.361557, 3.281286, 3.3696930000000003, 3.274917]
INTJ mixed kernel          median=2.9449 samples=[3.167714, 3.130622, 2.945205, 2.944885, 2.774361, 2.845168, 2.835741, 2.9920210000000003, 2.8340189999999996]
INTJ fixed mixed kernel    median=2.9498 samples=[3.151618, 3.148846, 2.994654, 2.995774, 2.9294830000000003, 2.858762, 2.9275189999999998, 2.94983, 2.922364]
```

### kernel cache align (median summary; raw jsonl at /tmp/intj-pf/kc/align.jsonl)
```
nw row             Cdef    Ddef   Cloop   Dloop   Cboth   Dboth
 1 hash_only       1.36    1.36    1.36    1.35    1.36    1.36
 1 hit/1           1.75    1.55    1.76    1.70    1.76    1.62
 1 hit/8           1.82    1.63    1.83    1.80    1.82    1.73
 1 hit/64          1.87    1.68    1.87    1.88    2.10    1.84
 1 hit/512         2.40    2.45    2.54    2.52    2.54    2.46
 1 hit_child       3.47    3.47    3.47    3.46    3.46    3.46
 1 miss/1          1.39    1.39    1.41    1.39    1.40    1.56
 1 miss/8          3.24    2.33    2.51    2.50    2.52    2.52
 1 miss/64         3.49    2.72    3.15    3.08    3.49    3.14
 1 miss/512        3.84    2.75    3.25    3.18    3.31    3.29
 2 hash_only       1.42    1.42    1.42    1.42    1.42    1.42
 2 hit/1           2.18    1.95    2.18    2.01    2.19    2.01
 2 hit/8           2.18    1.95    2.32    2.01    2.18    2.01
 2 hit/64          2.48    2.23    2.49    2.30    2.49    2.31
 2 hit/512         3.34    3.09    3.34    3.14    3.35    3.14
 2 hit_child       3.46    3.46    3.46    3.47    4.07    3.46
 2 miss/1          1.41    1.42    1.39    1.41    1.39    1.39
 2 miss/8          3.12    2.86    2.91    2.93    3.11    3.17
 2 miss/64         3.66    3.41    3.68    3.71    3.77    3.82
 2 miss/512        3.31    3.26    3.29    3.59    3.34    3.35
 5 hash_only       3.56    3.58    3.58    3.59    3.58    3.58
 5 hit/1           3.16    2.94    3.16    2.95    3.13    2.93
 5 hit/8           3.35    3.16    3.44    3.18    3.37    3.20
 5 hit/64          3.46    3.23    3.46    3.32    3.45    3.30
 5 hit/512         4.74    4.56    4.74    4.56    4.73    4.60
 5 hit_child       3.46    3.46    3.46    3.46    3.46    3.46
 5 miss/1          1.42    2.03    1.69    1.67    1.68    1.70
 5 miss/8          2.79    3.86    3.07    3.20    3.11    3.17
 5 miss/64         2.82    3.71    3.13    3.06    3.06    3.08
 5 miss/512        3.23    4.09    3.38    3.41    3.39    3.41
```

### kernel cache slot (median summary; raw jsonl at /tmp/intj-pf/kc/slot.jsonl)
```
nw row             base      w1      w2      w5      n1      n2      n5
 1 hash_only       1.35    1.35    1.35    1.35    1.36    1.35    1.36
 1 hit/1           1.59    1.57    1.59    1.57    1.60    1.60    1.62
 1 hit/8           1.67    1.63    1.64    1.65    1.66    1.67    1.67
 1 hit/64          1.72    1.67    1.69    1.68    1.75    1.74    1.72
 1 hit/512         2.35    2.30    2.30    2.30    2.18    2.17    2.18
 1 hit_child       3.46    3.24    3.00    3.00    3.24    3.03    3.04
 1 miss/1          1.28    1.30    1.30    1.31    1.28    1.27    1.27
 1 miss/8          2.71    2.60    2.54    2.62    2.78    2.76    2.84
 1 miss/64         2.87    2.78    2.76    2.76    2.90    2.90    2.92
 1 miss/512        3.07    2.89    2.89    2.89    3.15    3.12    3.18
 2 hash_only       1.42    1.42    1.42    1.42    1.42    1.42    1.42
 2 hit/1           2.06    2.06    1.85    1.86    2.00    2.01    2.02
 2 hit/8           2.06    2.06    1.85    1.87    1.99    2.02    2.04
 2 hit/64          2.34    2.35    2.27    2.24    2.30    2.44    2.44
 2 hit/512         3.12    3.07    3.02    3.02    2.93    2.92    2.92
 2 hit_child       3.46    3.24    3.00    3.00    3.24    3.04    3.04
 2 miss/1          1.51    1.53    1.38    1.38    1.40    1.23    1.21
 2 miss/8          3.37    3.42    2.41    2.42    3.08    2.19    2.18
 2 miss/64         4.35    3.92    3.18    3.13    3.72    2.77    2.77
 2 miss/512        3.94    3.57    2.90    2.90    3.56    2.61    2.61
 5 hash_only       3.58    3.57    3.58    3.58    3.58    3.59    3.58
 5 hit/1           3.05    2.99    2.99    2.77    3.00    3.01    2.95
 5 hit/8           3.28    3.20    3.20    3.13    3.18    3.18    3.32
 5 hit/64          3.35    3.27    3.27    3.23    3.25    3.25    3.43
 5 hit/512         4.67    4.53    4.53    4.68    4.58    4.58    4.65
 5 hit_child       3.46    3.24    3.01    3.00    3.22    3.04    3.04
 5 miss/1          1.41    1.39    1.43    1.25    1.38    1.40    1.45
 5 miss/8          3.03    2.98    2.98    2.22    2.92    2.92    2.89
 5 miss/64         2.87    2.86    2.86    2.08    2.72    2.73    2.79
 5 miss/512        3.47    3.46    3.46    3.11    3.16    3.15    3.36
```

### kernel cache slot2 (median summary; raw jsonl at /tmp/intj-pf/kc/slot2.jsonl)
```
nw row             base      w2      n2
 1 hash_only       1.35    1.35    1.36
 1 hit/1           1.59    1.58    1.60
 1 hit/8           1.67    1.63    1.67
 1 hit/64          1.72    1.67    1.72
 1 hit/512         2.35    2.30    2.19
 1 hit_child       3.46    3.00    3.04
 1 miss/1          1.29    1.30    1.27
 1 miss/8          2.74    2.62    2.77
 1 miss/64         2.87    2.77    2.92
 1 miss/512        3.06    2.88    3.14
 2 hash_only       1.42    1.42    1.42
 2 hit/1           2.06    1.87    2.00
 2 hit/8           2.06    1.87    2.05
 2 hit/64          2.34    2.23    2.39
 2 hit/512         3.12    3.02    2.92
 2 hit_child       3.46    3.00    3.04
 2 miss/1          1.56    1.38    1.22
 2 miss/8          3.34    2.41    2.18
 2 miss/64         4.38    3.16    2.79
 2 miss/512        3.94    2.90    2.61
 5 hash_only       3.58    3.58    3.59
 5 hit/1           3.05    2.99    3.00
 5 hit/8           3.28    3.20    3.18
 5 hit/64          3.35    3.26    3.25
 5 hit/512         4.67    4.53    4.59
 5 hit_child       3.46    3.01    3.04
 5 miss/1          1.40    1.39    1.39
 5 miss/8          3.04    2.95    2.89
 5 miss/64         2.85    2.85    2.73
 5 miss/512        3.47    3.46    3.15
```

### kernel cache final3 (median summary; raw jsonl at /tmp/intj-pf/kc/final3.jsonl)
```
nw row               c2      c3
 1 hash_only       1.36    1.35
 1 hit/1           1.59    1.56
 1 hit/8           1.67    1.62
 1 hit/64          1.72    1.67
 1 hit/512         2.35    2.30
 1 hit_child       3.47    3.00
 1 miss/1          1.27    1.23
 1 miss/8          2.89    2.64
 1 miss/64         2.87    2.78
 1 miss/512        3.18    2.93
 2 hash_only       1.42    1.42
 2 hit/1           2.06    1.86
 2 hit/8           2.06    1.83
 2 hit/64          2.34    2.21
 2 hit/512         3.12    3.02
 2 hit_child       3.46    3.01
 2 miss/1          1.40    1.38
 2 miss/8          3.51    2.43
 2 miss/64         4.41    3.14
 2 miss/512        3.95    2.88
 5 hash_only       3.58    3.58
 5 hit/1           3.04    2.99
 5 hit/8           3.27    3.20
 5 hit/64          3.35    3.27
 5 hit/512         4.67    4.54
 5 hit_child       3.46    3.01
 5 miss/1          1.39    1.39
 5 miss/8          2.97    3.01
 5 miss/64         2.85    2.86
 5 miss/512        3.46    3.48
```

### uptime
```
 12:22:08 up 57 days, 21:06,  0 user,  load average: 22.65, 16.55, 15.64
 12:25:22 up 57 days, 21:10,  0 user,  load average: 7.01, 12.96, 14.53
 12:47:02 up 57 days, 21:31,  0 user,  load average: 6.14, 14.88, 14.90
 12:50:17 up 57 days, 21:35,  0 user,  load average: 3.15, 9.29, 12.69
 12:54:13 up 57 days, 21:39,  0 user,  load average: 7.20, 7.56, 11.13
 12:58:40 up 57 days, 21:43,  0 user,  load average: 4.29, 5.53, 9.39
```

### bench_compare baseline perf
```
ffi_compare      median           FFI empty kernel                 value    baseline=3.3617 new=3.7018 delta=0.3401 pct=+10.1% (us) REGRESSION
ffi_compare      median           FFI mixed kernel                 value    baseline=3.3407 new=3.7047 delta=0.3640 pct=+10.9% (us) REGRESSION
ffi_compare      median           FFI packed nop                   value    baseline=0.1446 new=0.1449 delta=0.0003 pct=+0.2% (us)
ffi_compare      median           FFI packed nop mixed             value    baseline=0.1745 new=0.1731 delta=-0.0014 pct=-0.8% (us)
ffi_compare      median           FFI typed nop                    value    baseline=0.1491 new=0.1490 delta=-0.0001 pct=-0.1% (us)
ffi_compare      median           FFI typed nop mixed              value    baseline=0.1772 new=0.1774 delta=0.0002 pct=+0.1% (us)
ffi_compare      median           INTJ empty kernel                value    baseline=2.9849 new=3.4937 delta=0.5088 pct=+17.0% (us) REGRESSION
ffi_compare      median           INTJ fixed mixed kernel          value    baseline=2.9387 new=3.5480 delta=0.6093 pct=+20.7% (us) REGRESSION
ffi_compare      median           INTJ fixed-device kernel         value    baseline=2.9186 new=3.3145 delta=0.3959 pct=+13.6% (us) REGRESSION
ffi_compare      median           INTJ mixed kernel                value    baseline=2.9280 new=3.4576 delta=0.5296 pct=+18.1% (us) REGRESSION
ffi_paths        median_ns        FFI unpack Pair only             value    baseline=162.7 new=164.4 delta=1.7000 pct=+1.0% (ns)
ffi_paths        median_ns        INTJ 3-tensor host-only nop      value    baseline=33.8000 new=34.1000 delta=0.3000 pct=+0.9% (ns)
ffi_paths        median_ns        INTJ FFI wrapper kwargs          value    baseline=126.7 new=126.6 delta=-0.1000 pct=-0.1% (ns)
ffi_paths        median_ns        INTJ FFI wrapper positional      value    baseline=103.4 new=102.7 delta=-0.7000 pct=-0.7% (ns)
ffi_paths        median_ns        INTJ adapter defaults            value    baseline=89.2000 new=88.7000 delta=-0.5000 pct=-0.6% (ns)
ffi_paths        median_ns        INTJ adapter kwargs              value    baseline=108.2 new=107.0 delta=-1.2000 pct=-1.1% (ns)
ffi_paths        median_ns        INTJ adapter positional          value    baseline=90.6000 new=89.5000 delta=-1.1000 pct=-1.2% (ns)
ffi_paths        median_ns        INTJ cold compile callback + cache value    baseline=429.9 new=404.7 delta=-25.2000 pct=-5.9% (ns)
ffi_paths        median_ns        INTJ config FFI unpack           value    baseline=321.1 new=316.3 delta=-4.8000 pct=-1.5% (ns)
ffi_paths        median_ns        INTJ config direct               value    baseline=79.1000 new=80.2000 delta=1.1000 pct=+1.4% (ns)
ffi_paths        median_ns        INTJ direct positional           value    baseline=64.7000 new=64.9000 delta=0.2000 pct=+0.3% (ns)
ffi_paths        median_ns        INTJ hot call (no callback)      value    baseline=37.2000 new=38.0000 delta=0.8000 pct=+2.2% (ns)
ffi_paths        median_ns        INTJ pair FFI unpack             value    baseline=309.1 new=312.8 delta=3.7000 pct=+1.2% (ns)
ffi_paths        median_ns        INTJ pair direct                 value    baseline=65.6000 new=66.5000 delta=0.9000 pct=+1.4% (ns)
ffi_paths        median_ns        INTJ pair manual unpack          value    baseline=171.6 new=171.2 delta=-0.4000 pct=-0.2% (ns)
ffi_paths        median_ns        INTJ pair prebuilt *tuple        value    baseline=137.4 new=137.5 delta=0.1000 pct=+0.1% (ns)
ffi_paths        median_ns        INTJ pair stdlib astuple         value    baseline=1168.1 new=1145.4 delta=-22.7000 pct=-1.9% (ns)
ffi_sweep        median           args=0 FFI empty kernel          value    baseline=1.6784 new=1.6607 delta=-0.0177 pct=-1.1% (us)
ffi_sweep        median           args=0 FFI packed nop            value    baseline=0.1175 new=0.1171 delta=-0.0004 pct=-0.3% (us)
ffi_sweep        median           args=0 FFI typed nop             value    baseline=0.1210 new=0.1204 delta=-0.0006 pct=-0.5% (us)
ffi_sweep        median           args=16 FFI empty kernel         value    baseline=3.6540 new=3.6521 delta=-0.0019 pct=-0.1% (us)
ffi_sweep        median           args=16 FFI packed nop           value    baseline=0.2923 new=0.2944 delta=0.0021 pct=+0.7% (us)
ffi_sweep        median           args=16 FFI typed nop            value    baseline=0.3021 new=0.3070 delta=0.0049 pct=+1.6% (us)
ffi_sweep        median           args=3 FFI empty kernel          value    baseline=3.2503 new=3.2107 delta=-0.0396 pct=-1.2% (us)
ffi_sweep        median           args=3 FFI packed nop            value    baseline=0.1451 new=0.1443 delta=-0.0008 pct=-0.6% (us)
ffi_sweep        median           args=3 FFI typed nop             value    baseline=0.1492 new=0.1485 delta=-0.0007 pct=-0.5% (us)
ffi_sweep        median           args=32 FFI empty kernel         value    baseline=4.2249 new=4.1329 delta=-0.0920 pct=-2.2% (us)
ffi_sweep        median           args=32 FFI packed nop           value    baseline=0.4805 new=0.4801 delta=-0.0004 pct=-0.1% (us)
ffi_sweep        median           args=32 FFI typed nop            value    baseline=0.4946 new=0.4964 delta=0.0018 pct=+0.4% (us)
ffi_sweep        median           args=5 FFI empty kernel          value    baseline=3.2901 new=3.2770 delta=-0.0131 pct=-0.4% (us)
ffi_sweep        median           args=5 FFI packed nop            value    baseline=0.1740 new=0.1750 delta=0.0010 pct=+0.6% (us)
ffi_sweep        median           args=5 FFI typed nop             value    baseline=0.1783 new=0.1790 delta=0.0007 pct=+0.4% (us)
ffi_sweep        median           args=64 FFI empty kernel         value    baseline=5.0544 new=5.0690 delta=0.0146 pct=+0.3% (us)
ffi_sweep        median           args=64 FFI packed nop           value    baseline=0.8387 new=0.8462 delta=0.0075 pct=+0.9% (us)
ffi_sweep        median           args=64 FFI typed nop            value    baseline=0.8587 new=0.8670 delta=0.0083 pct=+1.0% (us)
ffi_sweep        median           args=8 FFI empty kernel          value    baseline=3.3924 new=3.3639 delta=-0.0285 pct=-0.8% (us)
ffi_sweep        median           args=8 FFI packed nop            value    baseline=0.2060 new=0.2065 delta=0.0005 pct=+0.2% (us)
ffi_sweep        median           args=8 FFI typed nop             value    baseline=0.2099 new=0.2102 delta=0.0003 pct=+0.1% (us)
hip_module       median           INTJ same function               value    baseline=2.9336 new=3.0092 delta=0.0756 pct=+2.6% (us)
hip_module       median           TVM FFI HIP HSACO                value    baseline=3.1473 new=3.1187 delta=-0.0286 pct=-0.9% (us)
hip_module       median           Triton same HSACO                value    baseline=15.9539 new=15.9201 delta=-0.0338 pct=-0.2% (us)
kernel_cache     1-word key       hash_only                        absl     baseline=1.3600 new=1.3500 delta=-0.0100 pct=-0.7% (ns)
kernel_cache     1-word key       hash_only                        intj     baseline=1.3600 new=1.3600 delta=0.0000 pct=+0.0% (ns)
kernel_cache     1-word key       hash_only                        tsl      baseline=1.3600 new=1.3600 delta=0.0000 pct=+0.0% (ns)
kernel_cache     1-word key       hit/1                            absl     baseline=1.3500 new=1.4400 delta=0.0900 pct=+6.7% (ns)
kernel_cache     1-word key       hit/1                            intj     baseline=1.2000 new=1.5700 delta=0.3700 pct=+30.8% (ns)
kernel_cache     1-word key       hit/1                            tsl      baseline=1.8500 new=2.0100 delta=0.1600 pct=+8.6% (ns)
kernel_cache     1-word key       hit/512                          absl     baseline=4.9400 new=5.3100 delta=0.3700 pct=+7.5% (ns)
kernel_cache     1-word key       hit/512                          intj     baseline=1.4900 new=2.3000 delta=0.8100 pct=+54.4% (ns)
kernel_cache     1-word key       hit/512                          tsl      baseline=2.6200 new=2.9200 delta=0.3000 pct=+11.5% (ns)
kernel_cache     1-word key       hit/64                           absl     baseline=4.3500 new=4.5400 delta=0.1900 pct=+4.4% (ns)
kernel_cache     1-word key       hit/64                           intj     baseline=1.3100 new=1.7300 delta=0.4200 pct=+32.1% (ns)
kernel_cache     1-word key       hit/64                           tsl      baseline=1.9100 new=2.1200 delta=0.2100 pct=+11.0% (ns)
kernel_cache     1-word key       hit/8                            absl     baseline=4.2300 new=4.3600 delta=0.1300 pct=+3.1% (ns)
kernel_cache     1-word key       hit/8                            intj     baseline=1.2600 new=1.7500 delta=0.4900 pct=+38.9% (ns)
kernel_cache     1-word key       hit/8                            tsl      baseline=1.8800 new=2.1000 delta=0.2200 pct=+11.7% (ns)
kernel_cache/1-word key/hit_child/absl                       MISSING in baseline
kernel_cache/1-word key/hit_child/intj                       MISSING in baseline
kernel_cache/1-word key/hit_child/tsl                        MISSING in baseline
kernel_cache     1-word key       miss/1                           absl     baseline=1.3800 new=1.3800 delta=0.0000 pct=+0.0% (ns)
kernel_cache     1-word key       miss/1                           intj     baseline=1.0300 new=1.2200 delta=0.1900 pct=+18.4% (ns)
kernel_cache     1-word key       miss/1                           tsl      baseline=1.4600 new=1.5700 delta=0.1100 pct=+7.5% (ns)
kernel_cache     1-word key       miss/512                         absl     baseline=3.9400 new=4.0600 delta=0.1200 pct=+3.0% (ns)
kernel_cache     1-word key       miss/512                         intj     baseline=2.2300 new=2.9500 delta=0.7200 pct=+32.3% (ns)
kernel_cache     1-word key       miss/512                         tsl      baseline=1.9700 new=1.9900 delta=0.0200 pct=+1.0% (ns)
kernel_cache     1-word key       miss/64                          absl     baseline=3.8100 new=3.5300 delta=-0.2800 pct=-7.3% (ns)
kernel_cache     1-word key       miss/64                          intj     baseline=2.1500 new=2.7500 delta=0.6000 pct=+27.9% (ns)
kernel_cache     1-word key       miss/64                          tsl      baseline=1.7300 new=1.7600 delta=0.0300 pct=+1.7% (ns)
kernel_cache     1-word key       miss/8                           absl     baseline=3.2500 new=3.2700 delta=0.0200 pct=+0.6% (ns)
kernel_cache     1-word key       miss/8                           intj     baseline=2.2200 new=2.7400 delta=0.5200 pct=+23.4% (ns)
kernel_cache     1-word key       miss/8                           tsl      baseline=1.7500 new=1.7600 delta=0.0100 pct=+0.6% (ns)
kernel_cache     2-word key       hash_only                        absl     baseline=1.4200 new=1.4200 delta=0.0000 pct=+0.0% (ns)
kernel_cache     2-word key       hash_only                        intj     baseline=1.4200 new=1.4200 delta=0.0000 pct=+0.0% (ns)
kernel_cache     2-word key       hash_only                        tsl      baseline=1.4200 new=1.4200 delta=0.0000 pct=+0.0% (ns)
kernel_cache     2-word key       hit/1                            absl     baseline=1.7000 new=2.0200 delta=0.3200 pct=+18.8% (ns)
kernel_cache     2-word key       hit/1                            intj     baseline=1.8900 new=1.8600 delta=-0.0300 pct=-1.6% (ns)
kernel_cache     2-word key       hit/1                            tsl      baseline=2.2600 new=2.5700 delta=0.3100 pct=+13.7% (ns)
kernel_cache     2-word key       hit/512                          absl     baseline=5.8300 new=6.3900 delta=0.5600 pct=+9.6% (ns)
kernel_cache     2-word key       hit/512                          intj     baseline=2.7300 new=3.0200 delta=0.2900 pct=+10.6% (ns)
kernel_cache     2-word key       hit/512                          tsl      baseline=3.4600 new=4.0400 delta=0.5800 pct=+16.8% (ns)
kernel_cache     2-word key       hit/64                           absl     baseline=5.2500 new=5.2400 delta=-0.0100 pct=-0.2% (ns)
kernel_cache     2-word key       hit/64                           intj     baseline=2.1100 new=2.2300 delta=0.1200 pct=+5.7% (ns)
kernel_cache     2-word key       hit/64                           tsl      baseline=2.5800 new=2.9200 delta=0.3400 pct=+13.2% (ns)
kernel_cache     2-word key       hit/8                            absl     baseline=4.7600 new=5.0400 delta=0.2800 pct=+5.9% (ns)
kernel_cache     2-word key       hit/8                            intj     baseline=1.8900 new=1.8300 delta=-0.0600 pct=-3.2% (ns)
kernel_cache     2-word key       hit/8                            tsl      baseline=2.2700 new=2.5700 delta=0.3000 pct=+13.2% (ns)
kernel_cache/2-word key/hit_child/absl                       MISSING in baseline
kernel_cache/2-word key/hit_child/intj                       MISSING in baseline
kernel_cache/2-word key/hit_child/tsl                        MISSING in baseline
kernel_cache     2-word key       miss/1                           absl     baseline=1.5400 new=1.5000 delta=-0.0400 pct=-2.6% (ns)
kernel_cache     2-word key       miss/1                           intj     baseline=1.3200 new=1.3800 delta=0.0600 pct=+4.5% (ns)
kernel_cache     2-word key       miss/1                           tsl      baseline=2.1600 new=2.3200 delta=0.1600 pct=+7.4% (ns)
kernel_cache     2-word key       miss/512                         absl     baseline=3.9000 new=3.9900 delta=0.0900 pct=+2.3% (ns)
kernel_cache     2-word key       miss/512                         intj     baseline=2.9500 new=2.8800 delta=-0.0700 pct=-2.4% (ns)
kernel_cache     2-word key       miss/512                         tsl      baseline=2.3000 new=2.7300 delta=0.4300 pct=+18.7% (ns)
kernel_cache     2-word key       miss/64                          absl     baseline=3.3600 new=3.8900 delta=0.5300 pct=+15.8% (ns)
kernel_cache     2-word key       miss/64                          intj     baseline=3.0800 new=3.1600 delta=0.0800 pct=+2.6% (ns)
kernel_cache     2-word key       miss/64                          tsl      baseline=2.0100 new=2.1000 delta=0.0900 pct=+4.5% (ns)
kernel_cache     2-word key       miss/8                           absl     baseline=3.2700 new=3.2100 delta=-0.0600 pct=-1.8% (ns)
kernel_cache     2-word key       miss/8                           intj     baseline=2.7100 new=2.4200 delta=-0.2900 pct=-10.7% (ns)
kernel_cache     2-word key       miss/8                           tsl      baseline=1.7500 new=1.8700 delta=0.1200 pct=+6.9% (ns)
kernel_cache     5-word key       hash_only                        absl     baseline=3.5700 new=3.5900 delta=0.0200 pct=+0.6% (ns)
kernel_cache     5-word key       hash_only                        intj     baseline=3.5800 new=3.5800 delta=0.0000 pct=+0.0% (ns)
kernel_cache     5-word key       hash_only                        tsl      baseline=3.5800 new=3.5900 delta=0.0100 pct=+0.3% (ns)
kernel_cache     5-word key       hit/1                            absl     baseline=2.6800 new=2.4700 delta=-0.2100 pct=-7.8% (ns)
kernel_cache     5-word key       hit/1                            intj     baseline=2.9400 new=3.0000 delta=0.0600 pct=+2.0% (ns)
kernel_cache     5-word key       hit/1                            tsl      baseline=3.1900 new=3.0600 delta=-0.1300 pct=-4.1% (ns)
kernel_cache     5-word key       hit/512                          absl     baseline=11.1800 new=11.1800 delta=0.0000 pct=+0.0% (ns)
kernel_cache     5-word key       hit/512                          intj     baseline=4.4700 new=4.5300 delta=0.0600 pct=+1.3% (ns)
kernel_cache     5-word key       hit/512                          tsl      baseline=4.9400 new=4.7800 delta=-0.1600 pct=-3.2% (ns)
kernel_cache     5-word key       hit/64                           absl     baseline=9.7500 new=9.7800 delta=0.0300 pct=+0.3% (ns)
kernel_cache     5-word key       hit/64                           intj     baseline=3.2500 new=3.2700 delta=0.0200 pct=+0.6% (ns)
kernel_cache     5-word key       hit/64                           tsl      baseline=3.6400 new=3.5000 delta=-0.1400 pct=-3.8% (ns)
kernel_cache     5-word key       hit/8                            absl     baseline=9.4700 new=9.4900 delta=0.0200 pct=+0.2% (ns)
kernel_cache     5-word key       hit/8                            intj     baseline=3.1700 new=3.2000 delta=0.0300 pct=+0.9% (ns)
kernel_cache     5-word key       hit/8                            tsl      baseline=3.5400 new=3.4100 delta=-0.1300 pct=-3.7% (ns)
kernel_cache/5-word key/hit_child/absl                       MISSING in baseline
kernel_cache/5-word key/hit_child/intj                       MISSING in baseline
kernel_cache/5-word key/hit_child/tsl                        MISSING in baseline
kernel_cache     5-word key       miss/1                           absl     baseline=1.6600 new=2.1400 delta=0.4800 pct=+28.9% (ns)
kernel_cache     5-word key       miss/1                           intj     baseline=1.8500 new=1.3900 delta=-0.4600 pct=-24.9% (ns)
kernel_cache     5-word key       miss/1                           tsl      baseline=2.2700 new=2.1000 delta=-0.1700 pct=-7.5% (ns)
kernel_cache     5-word key       miss/512                         absl     baseline=7.6600 new=7.7100 delta=0.0500 pct=+0.7% (ns)
kernel_cache     5-word key       miss/512                         intj     baseline=4.1000 new=3.5200 delta=-0.5800 pct=-14.1% (ns)
kernel_cache     5-word key       miss/512                         tsl      baseline=2.7600 new=2.5100 delta=-0.2500 pct=-9.1% (ns)
kernel_cache     5-word key       miss/64                          absl     baseline=6.7100 new=6.7200 delta=0.0100 pct=+0.1% (ns)
kernel_cache     5-word key       miss/64                          intj     baseline=3.3800 new=2.8600 delta=-0.5200 pct=-15.4% (ns)
kernel_cache     5-word key       miss/64                          tsl      baseline=2.5500 new=1.9600 delta=-0.5900 pct=-23.1% (ns)
kernel_cache     5-word key       miss/8                           absl     baseline=6.9300 new=6.9400 delta=0.0100 pct=+0.1% (ns)
kernel_cache     5-word key       miss/8                           intj     baseline=3.5700 new=2.9800 delta=-0.5900 pct=-16.5% (ns)
kernel_cache     5-word key       miss/8                           tsl      baseline=3.0900 new=2.2700 delta=-0.8200 pct=-26.5% (ns)
launch_gpu       launch           auto map                         ns/call  baseline=3137.8 new=3197.5 delta=59.7000 pct=+1.9% (ns)
launch_gpu       launch           baked                            ns/call  baseline=3054.5 new=2938.1 delta=-116.4 pct=-3.8% (ns)
launch_gpu       launch           bound pointer                    ns/call  baseline=2935.5 new=2966.8 delta=31.3000 pct=+1.1% (ns)
launch_gpu       launch           bound tensor                     ns/call  baseline=3086.1 new=3055.0 delta=-31.1000 pct=-1.0% (ns)
launch_gpu       launch           fixed device map                 ns/call  baseline=2932.4 new=2908.3 delta=-24.1000 pct=-0.8% (ns)
launch_gpu       launch           fixed device no-map              ns/call  baseline=3010.8 new=2910.2 delta=-100.6 pct=-3.3% (ns)
launch_gpu       launch           reduced key                      ns/call  baseline=3123.5 new=3131.0 delta=7.5000 pct=+0.2% (ns)
launch_gpu       launch           verify off                       ns/call  baseline=2994.9 new=2986.3 delta=-8.6000 pct=-0.3% (ns)
launch_gpu       launch           verify on                        ns/call  baseline=2913.9 new=2967.7 delta=53.8000 pct=+1.8% (ns)
launch_host      launch           auto map                         ns/call  baseline=41.6000 new=42.2000 delta=0.6000 pct=+1.4% (ns)
launch_host      launch           baked                            ns/call  baseline=39.2000 new=38.7000 delta=-0.5000 pct=-1.3% (ns)
launch_host      launch           bound pointer                    ns/call  baseline=44.8000 new=44.5000 delta=-0.3000 pct=-0.7% (ns)
launch_host      launch           bound tensor                     ns/call  baseline=45.6000 new=45.7000 delta=0.1000 pct=+0.2% (ns)
launch_host      launch           fixed device map                 ns/call  baseline=42.1000 new=42.1000 delta=0.0000 pct=+0.0% (ns)
launch_host      launch           fixed device no-map              ns/call  baseline=44.0000 new=44.1000 delta=0.1000 pct=+0.2% (ns)
launch_host      launch           reduced key                      ns/call  baseline=41.9000 new=42.4000 delta=0.5000 pct=+1.2% (ns)
launch_host      launch           verify off                       ns/call  baseline=42.5000 new=42.4000 delta=-0.1000 pct=-0.2% (ns)
launch_host      launch           verify on                        ns/call  baseline=44.2000 new=43.1000 delta=-1.1000 pct=-2.5% (ns)
launch_last_key  sweep            16 alternate                     ns/call  baseline=65.4000 new=64.7000 delta=-0.7000 pct=-1.1% (ns)
launch_last_key  sweep            16 repeat                        ns/call  baseline=61.9000 new=61.8000 delta=-0.1000 pct=-0.2% (ns)
launch_last_key  sweep            32 alternate                     ns/call  baseline=109.1 new=103.3 delta=-5.8000 pct=-5.3% (ns)
launch_last_key  sweep            32 repeat                        ns/call  baseline=104.8 new=98.9000 delta=-5.9000 pct=-5.6% (ns)
launch_last_key  sweep            4 alternate                      ns/call  baseline=35.9000 new=35.5000 delta=-0.4000 pct=-1.1% (ns)
launch_last_key  sweep            4 repeat                         ns/call  baseline=33.9000 new=33.5000 delta=-0.4000 pct=-1.2% (ns)
launch_readme    readme_path      grid=(0,) intj                   us       baseline=0.0300 new=0.0300 delta=0.0000 pct=+0.0% (us)
launch_readme    readme_path      grid=(0,) triton                 us       baseline=12.9200 new=12.9700 delta=0.0500 pct=+0.4% (us)
launch_readme    readme_path      grid=(1,) intj                   us       baseline=3.1200 new=3.1500 delta=0.0300 pct=+1.0% (us)
launch_readme    readme_path      grid=(1,) triton                 us       baseline=16.4500 new=16.6700 delta=0.2200 pct=+1.3% (us)
launch_readme    readme_torch_access interpreter                      decode ns baseline=872.4 new=875.8 delta=3.4000 pct=+0.4% (ns)
launch_readme    readme_torch_access runtime_shim                     decode ns baseline=92.0000 new=90.8000 delta=-1.2000 pct=-1.3% (ns)
launch_readme    readme_torch_access static_compile                   decode ns baseline=91.4000 new=91.1000 delta=-0.3000 pct=-0.3% (ns)
launch_sweep     sweep            16 int                           ns/call  baseline=67.9000 new=67.6000 delta=-0.3000 pct=-0.4% (ns)
launch_sweep     sweep            16 tensor                        ns/call  baseline=65.2000 new=63.9000 delta=-1.3000 pct=-2.0% (ns)
launch_sweep     sweep            32 int                           ns/call  baseline=112.8 new=104.8 delta=-8.0000 pct=-7.1% (ns)
launch_sweep     sweep            32 tensor                        ns/call  baseline=114.0 new=107.6 delta=-6.4000 pct=-5.6% (ns)
launch_sweep     sweep            4 int                            ns/call  baseline=40.5000 new=39.9000 delta=-0.6000 pct=-1.5% (ns)
launch_sweep     sweep            4 tensor                         ns/call  baseline=39.2000 new=38.4000 delta=-0.8000 pct=-2.0% (ns)
```

### bench_compare task5 perf
```
ffi_compare      median           FFI empty kernel                 value    baseline=3.3363 new=3.7018 delta=0.3655 pct=+11.0% (us) REGRESSION
ffi_compare      median           FFI mixed kernel                 value    baseline=3.3244 new=3.7047 delta=0.3803 pct=+11.4% (us) REGRESSION
ffi_compare      median           FFI packed nop                   value    baseline=0.1451 new=0.1449 delta=-0.0002 pct=-0.1% (us)
ffi_compare      median           FFI packed nop mixed             value    baseline=0.1741 new=0.1731 delta=-0.0010 pct=-0.6% (us)
ffi_compare      median           FFI typed nop                    value    baseline=0.1509 new=0.1490 delta=-0.0019 pct=-1.3% (us)
ffi_compare      median           FFI typed nop mixed              value    baseline=0.1770 new=0.1774 delta=0.0004 pct=+0.2% (us)
ffi_compare      median           INTJ empty kernel                value    baseline=2.9801 new=3.4937 delta=0.5136 pct=+17.2% (us) REGRESSION
ffi_compare      median           INTJ fixed mixed kernel          value    baseline=2.9001 new=3.5480 delta=0.6479 pct=+22.3% (us) REGRESSION
ffi_compare      median           INTJ fixed-device kernel         value    baseline=2.8987 new=3.3145 delta=0.4158 pct=+14.3% (us) REGRESSION
ffi_compare      median           INTJ mixed kernel                value    baseline=2.9519 new=3.4576 delta=0.5057 pct=+17.1% (us) REGRESSION
ffi_paths        median_ns        FFI unpack Pair only             value    baseline=166.8 new=164.4 delta=-2.4000 pct=-1.4% (ns)
ffi_paths        median_ns        INTJ 3-tensor host-only nop      value    baseline=34.2000 new=34.1000 delta=-0.1000 pct=-0.3% (ns)
ffi_paths        median_ns        INTJ FFI wrapper kwargs          value    baseline=126.6 new=126.6 delta=0.0000 pct=+0.0% (ns)
ffi_paths        median_ns        INTJ FFI wrapper positional      value    baseline=103.5 new=102.7 delta=-0.8000 pct=-0.8% (ns)
ffi_paths        median_ns        INTJ adapter defaults            value    baseline=89.4000 new=88.7000 delta=-0.7000 pct=-0.8% (ns)
ffi_paths        median_ns        INTJ adapter kwargs              value    baseline=110.0 new=107.0 delta=-3.0000 pct=-2.7% (ns)
ffi_paths        median_ns        INTJ adapter positional          value    baseline=90.7000 new=89.5000 delta=-1.2000 pct=-1.3% (ns)
ffi_paths        median_ns        INTJ cold compile callback + cache value    baseline=404.3 new=404.7 delta=0.4000 pct=+0.1% (ns)
ffi_paths        median_ns        INTJ config FFI unpack           value    baseline=318.3 new=316.3 delta=-2.0000 pct=-0.6% (ns)
ffi_paths        median_ns        INTJ config direct               value    baseline=80.1000 new=80.2000 delta=0.1000 pct=+0.1% (ns)
ffi_paths        median_ns        INTJ direct positional           value    baseline=65.5000 new=64.9000 delta=-0.6000 pct=-0.9% (ns)
ffi_paths        median_ns        INTJ hot call (no callback)      value    baseline=39.2000 new=38.0000 delta=-1.2000 pct=-3.1% (ns)
ffi_paths        median_ns        INTJ pair FFI unpack             value    baseline=316.5 new=312.8 delta=-3.7000 pct=-1.2% (ns)
ffi_paths        median_ns        INTJ pair direct                 value    baseline=67.3000 new=66.5000 delta=-0.8000 pct=-1.2% (ns)
ffi_paths        median_ns        INTJ pair manual unpack          value    baseline=174.3 new=171.2 delta=-3.1000 pct=-1.8% (ns)
ffi_paths        median_ns        INTJ pair prebuilt *tuple        value    baseline=140.2 new=137.5 delta=-2.7000 pct=-1.9% (ns)
ffi_paths        median_ns        INTJ pair stdlib astuple         value    baseline=1160.6 new=1145.4 delta=-15.2000 pct=-1.3% (ns)
ffi_sweep        median           args=0 FFI empty kernel          value    baseline=2.1848 new=1.6607 delta=-0.5241 pct=-24.0% (us)
ffi_sweep        median           args=0 FFI packed nop            value    baseline=0.1177 new=0.1171 delta=-0.0006 pct=-0.5% (us)
ffi_sweep        median           args=0 FFI typed nop             value    baseline=0.1201 new=0.1204 delta=0.0003 pct=+0.2% (us)
ffi_sweep        median           args=16 FFI empty kernel         value    baseline=4.2666 new=3.6521 delta=-0.6145 pct=-14.4% (us)
ffi_sweep        median           args=16 FFI packed nop           value    baseline=0.2942 new=0.2944 delta=0.0002 pct=+0.1% (us)
ffi_sweep        median           args=16 FFI typed nop            value    baseline=0.3021 new=0.3070 delta=0.0049 pct=+1.6% (us)
ffi_sweep        median           args=3 FFI empty kernel          value    baseline=3.6901 new=3.2107 delta=-0.4794 pct=-13.0% (us)
ffi_sweep        median           args=3 FFI packed nop            value    baseline=0.1447 new=0.1443 delta=-0.0004 pct=-0.3% (us)
ffi_sweep        median           args=3 FFI typed nop             value    baseline=0.1488 new=0.1485 delta=-0.0003 pct=-0.2% (us)
ffi_sweep        median           args=32 FFI empty kernel         value    baseline=4.9045 new=4.1329 delta=-0.7716 pct=-15.7% (us)
ffi_sweep        median           args=32 FFI packed nop           value    baseline=0.4766 new=0.4801 delta=0.0035 pct=+0.7% (us)
ffi_sweep        median           args=32 FFI typed nop            value    baseline=0.4937 new=0.4964 delta=0.0027 pct=+0.5% (us)
ffi_sweep        median           args=5 FFI empty kernel          value    baseline=3.7842 new=3.2770 delta=-0.5072 pct=-13.4% (us)
ffi_sweep        median           args=5 FFI packed nop            value    baseline=0.1745 new=0.1750 delta=0.0005 pct=+0.3% (us)
ffi_sweep        median           args=5 FFI typed nop             value    baseline=0.1781 new=0.1790 delta=0.0009 pct=+0.5% (us)
ffi_sweep        median           args=64 FFI empty kernel         value    baseline=5.9485 new=5.0690 delta=-0.8795 pct=-14.8% (us)
ffi_sweep        median           args=64 FFI packed nop           value    baseline=0.8336 new=0.8462 delta=0.0126 pct=+1.5% (us)
ffi_sweep        median           args=64 FFI typed nop            value    baseline=0.8850 new=0.8670 delta=-0.0180 pct=-2.0% (us)
ffi_sweep        median           args=8 FFI empty kernel          value    baseline=3.9624 new=3.3639 delta=-0.5985 pct=-15.1% (us)
ffi_sweep        median           args=8 FFI packed nop            value    baseline=0.2075 new=0.2065 delta=-0.0010 pct=-0.5% (us)
ffi_sweep        median           args=8 FFI typed nop             value    baseline=0.2123 new=0.2102 delta=-0.0021 pct=-1.0% (us)
hip_module       median           INTJ same function               value    baseline=2.9662 new=3.0092 delta=0.0430 pct=+1.4% (us)
hip_module       median           TVM FFI HIP HSACO                value    baseline=3.1726 new=3.1187 delta=-0.0539 pct=-1.7% (us)
hip_module       median           Triton same HSACO                value    baseline=16.3694 new=15.9201 delta=-0.4493 pct=-2.7% (us)
kernel_cache     1-word key       hash_only                        absl     baseline=1.3600 new=1.3500 delta=-0.0100 pct=-0.7% (ns)
kernel_cache     1-word key       hash_only                        intj     baseline=1.3700 new=1.3600 delta=-0.0100 pct=-0.7% (ns)
kernel_cache     1-word key       hash_only                        tsl      baseline=1.3500 new=1.3600 delta=0.0100 pct=+0.7% (ns)
kernel_cache     1-word key       hit/1                            absl     baseline=1.5000 new=1.4400 delta=-0.0600 pct=-4.0% (ns)
kernel_cache     1-word key       hit/1                            intj     baseline=1.7600 new=1.5700 delta=-0.1900 pct=-10.8% (ns)
kernel_cache     1-word key       hit/1                            tsl      baseline=2.0400 new=2.0100 delta=-0.0300 pct=-1.5% (ns)
kernel_cache     1-word key       hit/512                          absl     baseline=5.7800 new=5.3100 delta=-0.4700 pct=-8.1% (ns)
kernel_cache     1-word key       hit/512                          intj     baseline=2.5500 new=2.3000 delta=-0.2500 pct=-9.8% (ns)
kernel_cache     1-word key       hit/512                          tsl      baseline=3.0300 new=2.9200 delta=-0.1100 pct=-3.6% (ns)
kernel_cache     1-word key       hit/64                           absl     baseline=4.9500 new=4.5400 delta=-0.4100 pct=-8.3% (ns)
kernel_cache     1-word key       hit/64                           intj     baseline=1.8800 new=1.7300 delta=-0.1500 pct=-8.0% (ns)
kernel_cache     1-word key       hit/64                           tsl      baseline=2.1700 new=2.1200 delta=-0.0500 pct=-2.3% (ns)
kernel_cache     1-word key       hit/8                            absl     baseline=4.7600 new=4.3600 delta=-0.4000 pct=-8.4% (ns)
kernel_cache     1-word key       hit/8                            intj     baseline=1.9200 new=1.7500 delta=-0.1700 pct=-8.9% (ns)
kernel_cache     1-word key       hit/8                            tsl      baseline=2.1300 new=2.1000 delta=-0.0300 pct=-1.4% (ns)
kernel_cache     1-word key       hit_child                        absl     baseline=2.4600 new=3.0000 delta=0.5400 pct=+22.0% (ns)
kernel_cache     1-word key       hit_child                        intj     baseline=2.4600 new=3.0000 delta=0.5400 pct=+22.0% (ns)
kernel_cache     1-word key       hit_child                        tsl      baseline=2.4600 new=3.0000 delta=0.5400 pct=+22.0% (ns)
kernel_cache     1-word key       miss/1                           absl     baseline=1.3800 new=1.3800 delta=0.0000 pct=+0.0% (ns)
kernel_cache     1-word key       miss/1                           intj     baseline=1.3900 new=1.2200 delta=-0.1700 pct=-12.2% (ns)
kernel_cache     1-word key       miss/1                           tsl      baseline=1.5300 new=1.5700 delta=0.0400 pct=+2.6% (ns)
kernel_cache     1-word key       miss/512                         absl     baseline=4.4000 new=4.0600 delta=-0.3400 pct=-7.7% (ns)
kernel_cache     1-word key       miss/512                         intj     baseline=2.8000 new=2.9500 delta=0.1500 pct=+5.4% (ns)
kernel_cache     1-word key       miss/512                         tsl      baseline=2.4100 new=1.9900 delta=-0.4200 pct=-17.4% (ns)
kernel_cache     1-word key       miss/64                          absl     baseline=3.9100 new=3.5300 delta=-0.3800 pct=-9.7% (ns)
kernel_cache     1-word key       miss/64                          intj     baseline=2.7800 new=2.7500 delta=-0.0300 pct=-1.1% (ns)
kernel_cache     1-word key       miss/64                          tsl      baseline=2.1100 new=1.7600 delta=-0.3500 pct=-16.6% (ns)
kernel_cache     1-word key       miss/8                           absl     baseline=3.6800 new=3.2700 delta=-0.4100 pct=-11.1% (ns)
kernel_cache     1-word key       miss/8                           intj     baseline=2.4300 new=2.7400 delta=0.3100 pct=+12.8% (ns)
kernel_cache     1-word key       miss/8                           tsl      baseline=2.1200 new=1.7600 delta=-0.3600 pct=-17.0% (ns)
kernel_cache     2-word key       hash_only                        absl     baseline=1.4200 new=1.4200 delta=0.0000 pct=+0.0% (ns)
kernel_cache     2-word key       hash_only                        intj     baseline=1.4200 new=1.4200 delta=0.0000 pct=+0.0% (ns)
kernel_cache     2-word key       hash_only                        tsl      baseline=1.4300 new=1.4200 delta=-0.0100 pct=-0.7% (ns)
kernel_cache     2-word key       hit/1                            absl     baseline=1.9500 new=2.0200 delta=0.0700 pct=+3.6% (ns)
kernel_cache     2-word key       hit/1                            intj     baseline=2.1800 new=1.8600 delta=-0.3200 pct=-14.7% (ns)
kernel_cache     2-word key       hit/1                            tsl      baseline=2.5800 new=2.5700 delta=-0.0100 pct=-0.4% (ns)
kernel_cache     2-word key       hit/512                          absl     baseline=6.8200 new=6.3900 delta=-0.4300 pct=-6.3% (ns)
kernel_cache     2-word key       hit/512                          intj     baseline=3.3300 new=3.0200 delta=-0.3100 pct=-9.3% (ns)
kernel_cache     2-word key       hit/512                          tsl      baseline=4.0400 new=4.0400 delta=0.0000 pct=+0.0% (ns)
kernel_cache     2-word key       hit/64                           absl     baseline=5.6000 new=5.2400 delta=-0.3600 pct=-6.4% (ns)
kernel_cache     2-word key       hit/64                           intj     baseline=2.5100 new=2.2300 delta=-0.2800 pct=-11.2% (ns)
kernel_cache     2-word key       hit/64                           tsl      baseline=2.9300 new=2.9200 delta=-0.0100 pct=-0.3% (ns)
kernel_cache     2-word key       hit/8                            absl     baseline=5.3900 new=5.0400 delta=-0.3500 pct=-6.5% (ns)
kernel_cache     2-word key       hit/8                            intj     baseline=2.1800 new=1.8300 delta=-0.3500 pct=-16.1% (ns)
kernel_cache     2-word key       hit/8                            tsl      baseline=2.5700 new=2.5700 delta=0.0000 pct=+0.0% (ns)
kernel_cache     2-word key       hit_child                        absl     baseline=2.4600 new=3.0100 delta=0.5500 pct=+22.4% (ns)
kernel_cache     2-word key       hit_child                        intj     baseline=2.4600 new=3.0000 delta=0.5400 pct=+22.0% (ns)
kernel_cache     2-word key       hit_child                        tsl      baseline=2.4500 new=3.0000 delta=0.5500 pct=+22.4% (ns)
kernel_cache     2-word key       miss/1                           absl     baseline=1.8100 new=1.5000 delta=-0.3100 pct=-17.1% (ns)
kernel_cache     2-word key       miss/1                           intj     baseline=1.4200 new=1.3800 delta=-0.0400 pct=-2.8% (ns)
kernel_cache     2-word key       miss/1                           tsl      baseline=2.3200 new=2.3200 delta=0.0000 pct=+0.0% (ns)
kernel_cache     2-word key       miss/512                         absl     baseline=4.4300 new=3.9900 delta=-0.4400 pct=-9.9% (ns)
kernel_cache     2-word key       miss/512                         intj     baseline=3.6100 new=2.8800 delta=-0.7300 pct=-20.2% (ns)
kernel_cache     2-word key       miss/512                         tsl      baseline=2.7200 new=2.7300 delta=0.0100 pct=+0.4% (ns)
kernel_cache     2-word key       miss/64                          absl     baseline=4.1900 new=3.8900 delta=-0.3000 pct=-7.2% (ns)
kernel_cache     2-word key       miss/64                          intj     baseline=3.4000 new=3.1600 delta=-0.2400 pct=-7.1% (ns)
kernel_cache     2-word key       miss/64                          tsl      baseline=2.7600 new=2.1000 delta=-0.6600 pct=-23.9% (ns)
kernel_cache     2-word key       miss/8                           absl     baseline=3.6100 new=3.2100 delta=-0.4000 pct=-11.1% (ns)
kernel_cache     2-word key       miss/8                           intj     baseline=2.8400 new=2.4200 delta=-0.4200 pct=-14.8% (ns)
kernel_cache     2-word key       miss/8                           tsl      baseline=1.8000 new=1.8700 delta=0.0700 pct=+3.9% (ns)
kernel_cache     5-word key       hash_only                        absl     baseline=3.5800 new=3.5900 delta=0.0100 pct=+0.3% (ns)
kernel_cache     5-word key       hash_only                        intj     baseline=3.5700 new=3.5800 delta=0.0100 pct=+0.3% (ns)
kernel_cache     5-word key       hash_only                        tsl      baseline=3.5800 new=3.5900 delta=0.0100 pct=+0.3% (ns)
kernel_cache     5-word key       hit/1                            absl     baseline=2.8400 new=2.4700 delta=-0.3700 pct=-13.0% (ns)
kernel_cache     5-word key       hit/1                            intj     baseline=3.1300 new=3.0000 delta=-0.1300 pct=-4.2% (ns)
kernel_cache     5-word key       hit/1                            tsl      baseline=3.1800 new=3.0600 delta=-0.1200 pct=-3.8% (ns)
kernel_cache     5-word key       hit/512                          absl     baseline=11.7100 new=11.1800 delta=-0.5300 pct=-4.5% (ns)
kernel_cache     5-word key       hit/512                          intj     baseline=4.7300 new=4.5300 delta=-0.2000 pct=-4.2% (ns)
kernel_cache     5-word key       hit/512                          tsl      baseline=4.9300 new=4.7800 delta=-0.1500 pct=-3.0% (ns)
kernel_cache     5-word key       hit/64                           absl     baseline=10.3200 new=9.7800 delta=-0.5400 pct=-5.2% (ns)
kernel_cache     5-word key       hit/64                           intj     baseline=3.4500 new=3.2700 delta=-0.1800 pct=-5.2% (ns)
kernel_cache     5-word key       hit/64                           tsl      baseline=3.6200 new=3.5000 delta=-0.1200 pct=-3.3% (ns)
kernel_cache     5-word key       hit/8                            absl     baseline=10.0500 new=9.4900 delta=-0.5600 pct=-5.6% (ns)
kernel_cache     5-word key       hit/8                            intj     baseline=3.3800 new=3.2000 delta=-0.1800 pct=-5.3% (ns)
kernel_cache     5-word key       hit/8                            tsl      baseline=3.5300 new=3.4100 delta=-0.1200 pct=-3.4% (ns)
kernel_cache     5-word key       hit_child                        absl     baseline=2.4500 new=3.0100 delta=0.5600 pct=+22.9% (ns)
kernel_cache     5-word key       hit_child                        intj     baseline=2.4600 new=3.0100 delta=0.5500 pct=+22.4% (ns)
kernel_cache     5-word key       hit_child                        tsl      baseline=2.4600 new=3.0100 delta=0.5500 pct=+22.4% (ns)
kernel_cache     5-word key       miss/1                           absl     baseline=1.7200 new=2.1400 delta=0.4200 pct=+24.4% (ns)
kernel_cache     5-word key       miss/1                           intj     baseline=2.2800 new=1.3900 delta=-0.8900 pct=-39.0% (ns)
kernel_cache     5-word key       miss/1                           tsl      baseline=2.1200 new=2.1000 delta=-0.0200 pct=-0.9% (ns)
kernel_cache     5-word key       miss/512                         absl     baseline=7.9400 new=7.7100 delta=-0.2300 pct=-2.9% (ns)
kernel_cache     5-word key       miss/512                         intj     baseline=3.9500 new=3.5200 delta=-0.4300 pct=-10.9% (ns)
kernel_cache     5-word key       miss/512                         tsl      baseline=2.5600 new=2.5100 delta=-0.0500 pct=-2.0% (ns)
kernel_cache     5-word key       miss/64                          absl     baseline=6.9100 new=6.7200 delta=-0.1900 pct=-2.7% (ns)
kernel_cache     5-word key       miss/64                          intj     baseline=3.7000 new=2.8600 delta=-0.8400 pct=-22.7% (ns)
kernel_cache     5-word key       miss/64                          tsl      baseline=1.7600 new=1.9600 delta=0.2000 pct=+11.4% (ns)
kernel_cache     5-word key       miss/8                           absl     baseline=7.1200 new=6.9400 delta=-0.1800 pct=-2.5% (ns)
kernel_cache     5-word key       miss/8                           intj     baseline=3.8400 new=2.9800 delta=-0.8600 pct=-22.4% (ns)
kernel_cache     5-word key       miss/8                           tsl      baseline=2.1500 new=2.2700 delta=0.1200 pct=+5.6% (ns)
launch_gpu       launch           auto map                         ns/call  baseline=3124.4 new=3197.5 delta=73.1000 pct=+2.3% (ns)
launch_gpu       launch           baked                            ns/call  baseline=3007.0 new=2938.1 delta=-68.9000 pct=-2.3% (ns)
launch_gpu       launch           bound pointer                    ns/call  baseline=3040.3 new=2966.8 delta=-73.5000 pct=-2.4% (ns)
launch_gpu       launch           bound tensor                     ns/call  baseline=3009.5 new=3055.0 delta=45.5000 pct=+1.5% (ns)
launch_gpu       launch           fixed device map                 ns/call  baseline=2943.2 new=2908.3 delta=-34.9000 pct=-1.2% (ns)
launch_gpu       launch           fixed device no-map              ns/call  baseline=2922.8 new=2910.2 delta=-12.6000 pct=-0.4% (ns)
launch_gpu       launch           reduced key                      ns/call  baseline=3113.5 new=3131.0 delta=17.5000 pct=+0.6% (ns)
launch_gpu       launch           verify off                       ns/call  baseline=3054.2 new=2986.3 delta=-67.9000 pct=-2.2% (ns)
launch_gpu       launch           verify on                        ns/call  baseline=3005.5 new=2967.7 delta=-37.8000 pct=-1.3% (ns)
launch_host      launch           auto map                         ns/call  baseline=42.8000 new=42.2000 delta=-0.6000 pct=-1.4% (ns)
launch_host      launch           baked                            ns/call  baseline=39.4000 new=38.7000 delta=-0.7000 pct=-1.8% (ns)
launch_host      launch           bound pointer                    ns/call  baseline=43.6000 new=44.5000 delta=0.9000 pct=+2.1% (ns)
launch_host      launch           bound tensor                     ns/call  baseline=46.3000 new=45.7000 delta=-0.6000 pct=-1.3% (ns)
launch_host      launch           fixed device map                 ns/call  baseline=42.7000 new=42.1000 delta=-0.6000 pct=-1.4% (ns)
launch_host      launch           fixed device no-map              ns/call  baseline=43.9000 new=44.1000 delta=0.2000 pct=+0.5% (ns)
launch_host      launch           reduced key                      ns/call  baseline=42.6000 new=42.4000 delta=-0.2000 pct=-0.5% (ns)
launch_host      launch           verify off                       ns/call  baseline=42.2000 new=42.4000 delta=0.2000 pct=+0.5% (ns)
launch_host      launch           verify on                        ns/call  baseline=43.5000 new=43.1000 delta=-0.4000 pct=-0.9% (ns)
launch_last_key  sweep            16 alternate                     ns/call  baseline=65.3000 new=64.7000 delta=-0.6000 pct=-0.9% (ns)
launch_last_key  sweep            16 repeat                        ns/call  baseline=62.6000 new=61.8000 delta=-0.8000 pct=-1.3% (ns)
launch_last_key  sweep            32 alternate                     ns/call  baseline=104.2 new=103.3 delta=-0.9000 pct=-0.9% (ns)
launch_last_key  sweep            32 repeat                        ns/call  baseline=99.9000 new=98.9000 delta=-1.0000 pct=-1.0% (ns)
launch_last_key  sweep            4 alternate                      ns/call  baseline=35.7000 new=35.5000 delta=-0.2000 pct=-0.6% (ns)
launch_last_key  sweep            4 repeat                         ns/call  baseline=33.6000 new=33.5000 delta=-0.1000 pct=-0.3% (ns)
launch_readme    readme_path      grid=(0,) intj                   us       baseline=0.0300 new=0.0300 delta=0.0000 pct=+0.0% (us)
launch_readme    readme_path      grid=(0,) triton                 us       baseline=13.0500 new=12.9700 delta=-0.0800 pct=-0.6% (us)
launch_readme    readme_path      grid=(1,) intj                   us       baseline=3.1400 new=3.1500 delta=0.0100 pct=+0.3% (us)
launch_readme    readme_path      grid=(1,) triton                 us       baseline=16.5200 new=16.6700 delta=0.1500 pct=+0.9% (us)
launch_readme    readme_torch_access interpreter                      decode ns baseline=873.0 new=875.8 delta=2.8000 pct=+0.3% (ns)
launch_readme    readme_torch_access runtime_shim                     decode ns baseline=91.5000 new=90.8000 delta=-0.7000 pct=-0.8% (ns)
launch_readme    readme_torch_access static_compile                   decode ns baseline=90.9000 new=91.1000 delta=0.2000 pct=+0.2% (ns)
launch_sweep     sweep            16 int                           ns/call  baseline=68.2000 new=67.6000 delta=-0.6000 pct=-0.9% (ns)
launch_sweep     sweep            16 tensor                        ns/call  baseline=64.3000 new=63.9000 delta=-0.4000 pct=-0.6% (ns)
launch_sweep     sweep            32 int                           ns/call  baseline=107.1 new=104.8 delta=-2.3000 pct=-2.1% (ns)
launch_sweep     sweep            32 tensor                        ns/call  baseline=110.1 new=107.6 delta=-2.5000 pct=-2.3% (ns)
launch_sweep     sweep            4 int                            ns/call  baseline=40.0000 new=39.9000 delta=-0.1000 pct=-0.3% (ns)
launch_sweep     sweep            4 tensor                         ns/call  baseline=38.7000 new=38.4000 delta=-0.3000 pct=-0.8% (ns)
```

### perf/round0_ffi_compare.txt
```
commit=be52e1c24c29ac6a277e81b5d65771e233586bbd
started=2026-09-27T12:55:19+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_ffi_compare.py --iters 1000 --batches 9
iters=1000 batches=9 unit=us/call (host-only, no timed sync)
FFI packed nop             median=0.1452 samples=[0.147936, 0.14431899999999998, 0.143397, 0.14715999999999999, 0.145117, 0.144477, 0.145161, 0.147635, 0.14575200000000002]
FFI typed nop              median=0.1490 samples=[0.150989, 0.148175, 0.151098, 0.1494, 0.14697300000000002, 0.15156899999999998, 0.147177, 0.14782800000000001, 0.148953]
FFI empty kernel           median=3.8876 samples=[3.833329, 4.016721, 3.9401219999999997, 3.8876109999999997, 3.775921, 3.918582, 3.7606100000000002, 3.916087, 3.7667770000000003]
INTJ empty kernel          median=3.4937 samples=[3.1507289999999997, 3.6835549999999997, 3.4937199999999997, 3.540348, 3.4111700000000003, 3.4947880000000002, 3.436114, 3.508983, 3.4784450000000002]
INTJ fixed-device kernel   median=3.5274 samples=[3.302211, 6.174181, 3.6380019999999997, 3.8501, 3.51597, 3.309558, 3.531424, 3.3376129999999997, 3.52742]
FFI packed nop mixed       median=0.1737 samples=[0.173566, 0.175679, 0.17405500000000002, 0.173656, 0.17830500000000002, 0.173152, 0.171976, 0.172804, 0.177461]
FFI typed nop mixed        median=0.1768 samples=[0.17990199999999998, 0.177968, 0.175683, 0.176831, 0.174851, 0.178768, 0.175215, 0.180444, 0.17546899999999999]
FFI mixed kernel           median=3.8282 samples=[3.820389, 4.1333329999999995, 3.8281579999999997, 3.953926, 3.743975, 3.9416170000000004, 3.821147, 3.9363249999999996, 3.794909]
INTJ mixed kernel          median=3.4972 samples=[3.567003, 3.648496, 3.4832899999999998, 3.581174, 3.434152, 3.497203, 3.455804, 3.547098, 3.463509]
INTJ fixed mixed kernel    median=3.5640 samples=[3.614987, 3.755243, 3.595538, 3.6383069999999997, 3.5416350000000003, 3.5640479999999997, 3.472759, 3.5503460000000002, 3.4641460000000004]
elapsed_ns=12229994208
exit_status=0
ended=2026-09-27T12:55:32+08:00
```

### perf/round0_ffi_paths.txt
```
commit=be52e1c24c29ac6a277e81b5d65771e233586bbd
started=2026-09-27T12:56:08+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_intj_ffi_paths.py --mode all --iters 1000 --batches 9
iters=1000 batches=9 unit=ns/call; no_gpu=True
mode=callback
callback_count=9100 (100 warmup + 9000 unique misses)
INTJ cold compile callback + cache  median_ns=399.3 samples_ns=[492.434, 507.552, 376.924, 637.425, 362.434, 383.016, 388.16, 399.28, 950.76]
INTJ hot call (no callback)         median_ns=37.0 samples_ns=[40.348, 38.783, 45.367, 37.153, 36.511, 36.959, 36.557, 36.758, 36.634]
INTJ 3-tensor host-only nop         median_ns=33.7 samples_ns=[41.789, 34.028, 33.674, 33.639, 33.684, 33.686, 33.737, 33.791, 33.537]
mode=kwargs
INTJ direct positional              median_ns=64.6 samples_ns=[66.748, 64.558, 64.358, 64.305, 71.452, 64.679, 64.634, 64.64, 64.484]
INTJ adapter positional             median_ns=89.5 samples_ns=[89.531, 91.023, 89.717, 88.919, 89.628, 88.895, 89.212, 95.237, 88.968]
INTJ adapter kwargs                 median_ns=105.1 samples_ns=[105.918, 105.817, 103.688, 103.641, 103.364, 105.066, 104.541, 113.598, 105.591]
INTJ adapter defaults               median_ns=88.2 samples_ns=[88.554, 88.362, 88.183, 88.397, 88.185, 88.01, 88.293, 88.13, 88.218]
INTJ FFI wrapper positional         median_ns=101.7 samples_ns=[107.261, 101.777, 101.433, 101.611, 101.487, 102.102, 101.737, 101.788, 101.636]
INTJ FFI wrapper kwargs             median_ns=124.3 samples_ns=[132.488, 125.516, 124.296, 123.338, 124.201, 124.192, 124.189, 125.464, 129.846]
INTJ native kwargs: unsupported (verified TypeError)
mode=dataclass
INTJ pair direct                    median_ns=70.1 samples_ns=[78.155, 69.45, 70.763, 69.646, 70.107, 70.328, 69.443, 69.708, 72.516]
INTJ pair prebuilt *tuple           median_ns=136.5 samples_ns=[137.066, 136.062, 135.802, 136.541, 136.15, 135.402, 140.154, 138.253, 137.51]
FFI unpack Pair only                median_ns=163.8 samples_ns=[163.684, 162.992, 162.1, 162.181, 204.121, 164.252, 164.083, 163.922, 163.847]
INTJ pair manual unpack             median_ns=170.3 samples_ns=[184.564, 170.577, 170.129, 168.8, 170.297, 170.821, 173.936, 168.995, 169.606]
INTJ pair FFI unpack                median_ns=314.6 samples_ns=[311.717, 315.299, 312.594, 314.617, 317.247, 314.636, 312.197, 311.435, 316.825]
INTJ pair stdlib astuple            median_ns=1145.1 samples_ns=[1163.601, 1145.146, 1164.347, 1144.595, 1145.064, 1131.228, 1139.103, 1152.137, 1171.692]
INTJ config direct                  median_ns=80.5 samples_ns=[79.547, 79.545, 84.602, 81.267, 80.498, 80.529, 80.979, 80.569, 79.292]
INTJ config FFI unpack              median_ns=316.3 samples_ns=[318.737, 318.622, 313.262, 311.855, 317.524, 316.312, 314.851, 317.032, 313.898]
INTJ native dataclass unpack: unsupported (verified TypeError)
elapsed_ns=8718651864
exit_status=0
ended=2026-09-27T12:56:16+08:00
```

### perf/round0_ffi_sweep.txt
```
commit=be52e1c24c29ac6a277e81b5d65771e233586bbd
started=2026-09-27T12:55:32+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_ffi_compare.py --sweep --iters 1000 --batches 9
counts=[0, 3, 5, 8, 16, 32, 64] iters=1000 batches=9 unit=us/call (host-only, no timed sync); INTJ device-bound, no specialization key
args= 0 FFI packed nop           median=0.1173 samples=[0.126352, 0.117074, 0.11834, 0.118081, 0.11731699999999999, 0.11668300000000001, 0.11540600000000001, 0.117828, 0.115245]
args= 0 FFI typed nop            median=0.1205 samples=[0.122072, 0.122178, 0.11984099999999999, 0.120457, 0.11951200000000001, 0.12193000000000001, 0.11896599999999999, 0.122457, 0.117899]
args= 0 FFI empty kernel         median=1.7914 samples=[1.791442, 1.8518320000000001, 1.626117, 1.8259269999999999, 1.600258, 1.876131, 1.6613499999999999, 1.845366, 1.5841500000000002]
args= 0 INTJ static_compile kernel median=2.7298 samples=[3.1509560000000003, 2.7531309999999998, 2.742343, 2.722703, 2.72985, 2.683484, 2.7076379999999998, 2.735728, 2.719869]
args= 0 INTJ runtime_shim kernel median=2.7153 samples=[2.903781, 2.694233, 2.755397, 2.6747229999999997, 2.734331, 2.6684229999999998, 2.715266, 2.684681, 2.7238249999999997]
args= 3 FFI packed nop           median=0.1443 samples=[0.145044, 0.144483, 0.142152, 0.144258, 0.144197, 0.142363, 0.146219, 0.14676, 0.14221999999999999]
args= 3 FFI typed nop            median=0.1485 samples=[0.14851599999999998, 0.147011, 0.147952, 0.152417, 0.149361, 0.147288, 0.148022, 0.14992599999999998, 0.151407]
args= 3 FFI empty kernel         median=3.2107 samples=[3.3808200000000004, 3.2226190000000003, 3.2230149999999997, 3.220141, 3.207884, 3.2107379999999996, 3.16147, 3.210029, 3.195763]
args= 3 INTJ static_compile kernel median=2.8646 samples=[3.010889, 2.834678, 2.865163, 2.851901, 2.864569, 2.880801, 2.935029, 2.849529, 2.86276]
args= 3 INTJ runtime_shim kernel median=2.8851 samples=[3.029028, 2.834797, 2.8885709999999998, 2.872699, 2.885088, 2.9115320000000002, 2.8972629999999997, 2.857757, 2.875939]
args= 5 FFI packed nop           median=0.1750 samples=[0.17161400000000002, 0.175261, 0.175627, 0.174998, 0.17386500000000002, 0.17492500000000002, 0.175507, 0.173941, 0.17693299999999998]
args= 5 FFI typed nop            median=0.1790 samples=[0.173317, 0.185115, 0.176823, 0.179927, 0.179292, 0.179287, 0.177583, 0.178279, 0.178999]
args= 5 FFI empty kernel         median=3.2661 samples=[3.397932, 3.2914160000000003, 3.2789859999999997, 3.289728, 3.262864, 3.250347, 3.263127, 3.2661350000000002, 3.264836]
args= 5 INTJ static_compile kernel median=2.8689 samples=[3.020464, 2.89309, 2.86885, 2.864617, 2.880073, 2.86016, 2.887348, 2.86885, 2.845603]
args= 5 INTJ runtime_shim kernel median=2.8594 samples=[3.0401309999999997, 2.803114, 2.8593629999999997, 2.763903, 2.876298, 2.877912, 2.869881, 2.7706, 2.8297820000000002]
args= 8 FFI packed nop           median=0.2061 samples=[0.203604, 0.204069, 0.207977, 0.205717, 0.20797300000000002, 0.206739, 0.206088, 0.204609, 0.211895]
args= 8 FFI typed nop            median=0.2102 samples=[0.206074, 0.20528, 0.210187, 0.21037299999999998, 0.20563700000000001, 0.213304, 0.208572, 0.214678, 0.210704]
args= 8 FFI empty kernel         median=3.3639 samples=[3.54174, 3.3730279999999997, 3.28625, 3.439948, 3.293127, 3.363897, 3.315007, 3.389002, 3.271806]
args= 8 INTJ static_compile kernel median=3.0337 samples=[3.107355, 2.902969, 3.079145, 3.0194490000000003, 3.048962, 3.022195, 3.033711, 3.003654, 3.0542040000000004]
args= 8 INTJ runtime_shim kernel median=2.9644 samples=[3.125707, 2.896734, 2.976807, 2.9071190000000002, 2.966747, 2.9059369999999998, 2.9708710000000003, 2.895323, 2.964384]
args=16 FFI packed nop           median=0.2944 samples=[0.292757, 0.294377, 0.302242, 0.29788600000000004, 0.296512, 0.292796, 0.294374, 0.29351299999999997, 0.29162400000000005]
args=16 FFI typed nop            median=0.3070 samples=[0.298504, 0.30963, 0.30702999999999997, 0.30792, 0.309342, 0.306072, 0.29877699999999996, 0.31135, 0.304547]
args=16 FFI empty kernel         median=3.6518 samples=[3.8020050000000003, 3.670694, 3.5306610000000003, 3.8704769999999997, 3.538062, 3.667661, 3.526107, 3.651835, 3.526464]
args=16 INTJ static_compile kernel median=3.3152 samples=[3.276154, 3.315241, 3.443956, 3.264869, 3.472111, 3.282542, 3.477789, 3.3045340000000003, 3.467251]
args=16 INTJ runtime_shim kernel median=3.2779 samples=[3.277919, 3.1230149999999997, 3.309949, 3.023386, 3.317745, 3.035791, 3.3163519999999997, 3.0235540000000003, 3.28549]
args=32 FFI packed nop           median=0.4801 samples=[0.476351, 0.488184, 0.498639, 0.490906, 0.478029, 0.46981799999999996, 0.481697, 0.48008100000000004, 0.476024]
args=32 FFI typed nop            median=0.4943 samples=[0.493724, 0.50227, 0.516986, 0.511937, 0.494253, 0.491714, 0.494327, 0.495261, 0.48404]
args=32 FFI empty kernel         median=4.1444 samples=[4.266095999999999, 4.157533, 4.031622, 4.168152, 3.988934, 4.144407, 4.05549, 4.152226, 4.00349]
args=32 INTJ static_compile kernel median=3.5817 samples=[3.5245900000000003, 3.581742, 3.641457, 3.543026, 3.63526, 3.542438, 3.620444, 3.539919, 3.757289]
args=32 INTJ runtime_shim kernel median=3.4800 samples=[3.479978, 3.43053, 3.581542, 3.418693, 3.584783, 3.425125, 3.559638, 3.4054499999999996, 3.603005]
args=64 FFI packed nop           median=0.8462 samples=[0.830799, 0.857615, 0.8655700000000001, 0.850348, 0.839897, 0.8462430000000001, 0.831254, 0.833265, 0.8464299999999999]
args=64 FFI typed nop            median=0.8670 samples=[0.88261, 0.8894160000000001, 0.902889, 0.885823, 0.85862, 0.866996, 0.857394, 0.866221, 0.858623]
args=64 FFI empty kernel         median=5.0480 samples=[5.236868, 5.047982, 5.10095, 5.043897, 5.07733, 5.036786, 5.045809, 5.0757449999999995, 5.039334]
args=64 INTJ static_compile kernel median=4.5643 samples=[4.74434, 4.536045, 4.583951, 4.564322, 4.571664, 4.53128, 4.5512120000000005, 4.531214, 4.570928]
args=64 INTJ runtime_shim kernel median=4.5396 samples=[4.7158299999999995, 4.55614, 4.533551999999999, 4.619707, 4.531411, 4.557225000000001, 4.539571, 4.359965, 4.4939480000000005]
elapsed_ns=36192670125
exit_status=0
ended=2026-09-27T12:56:08+08:00
```

### perf/round0_hip_module.txt
```
commit=be52e1c24c29ac6a277e81b5d65771e233586bbd
started=2026-09-27T12:56:16+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_hip_module_launch.py --iters 1000 --batches 9
hsaco_sha256=df3e529e178196ad62f7c07ab8593ede5b8596c76e7c6fbd12a5d947a696e65c kernel=empty_kernel threads=256 shared=0 callback_count=1
iters=1000 batches=9 unit=us/call (host enqueue, no timed sync)
TVM FFI HIP HSACO         median=3.3108 samples=[3.6853339999999997, 3.327722, 3.3107759999999997, 3.3145830000000003, 3.3122469999999997, 3.159086, 3.176996, 3.1226260000000003, 3.1334899999999997]
Triton same HSACO         median=16.2445 samples=[16.530223000000003, 16.414291000000002, 16.314437, 16.331035, 16.244544, 16.122085, 16.078747, 16.101503, 16.179182]
INTJ same function        median=3.0897 samples=[3.212661, 3.213444, 3.1218939999999997, 3.1276990000000002, 3.08973, 2.931447, 2.94613, 2.939301, 2.892351]
elapsed_ns=5738062608
exit_status=0
ended=2026-09-27T12:56:22+08:00
```

### perf/round0_kernel_cache.txt
```
commit=be52e1c24c29ac6a277e81b5d65771e233586bbd
started=2026-09-27T12:56:22+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python -m pytest tests/test_kernel_cache.py -s -q

kernel cache, 1-word key, ns per operation

                      intj       tsl      absl
hash_only             1.36      1.36      1.35
hit/1                 1.58      2.01      1.45
hit/8                 1.66      2.10      4.36
hit/64                1.73      2.12      4.54
hit/512               2.30      2.92      5.30
hit_child             3.00      3.00      3.00
miss/1                1.22      1.57      1.38
miss/8                2.68      1.76      3.26
miss/64               2.75      1.76      3.53
miss/512              2.96      2.01      4.08

kernel cache, 2-word key, ns per operation

                      intj       tsl      absl
hash_only             1.42      1.42      1.42
hit/1                 1.81      2.57      2.01
hit/8                 1.83      2.59      5.04
hit/64                2.22      2.93      5.26
hit/512               3.02      4.02      6.39
hit_child             3.00      3.00      3.01
miss/1                1.38      2.31      1.52
miss/8                2.41      1.87      3.20
miss/64               3.16      2.10      3.91
miss/512              2.87      2.73      3.99

kernel cache, 5-word key, ns per operation

                      intj       tsl      absl
hash_only             3.58      3.59      3.59
hit/1                 3.00      3.06      2.47
hit/8                 3.20      3.41      9.48
hit/64                3.27      3.50      9.78
hit/512               4.53      4.77     11.18
hit_child             3.01      3.01      3.01
miss/1                1.39      2.10      2.14
miss/8                2.98      2.28      6.94
miss/64               2.86      1.95      6.72
miss/512              3.46      2.51      7.72
........
8 passed in 25.28s
elapsed_ns=26155980199
exit_status=0
ended=2026-09-27T12:56:48+08:00
```

### perf/round0_launch_gpu.txt
```
commit=be52e1c24c29ac6a277e81b5d65771e233586bbd
started=2026-09-27T12:54:13+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --iters 1000 --batches 9
mode=gpu; 1000 calls × 9 batches; median ns/call
               path    ns/call       Δ ns       Δ %   build ms   bind ms
           auto map     3232.0       +0.0      +0.0    2300.25         -
        reduced key     3179.9      -52.1      -1.6    2186.71         -
         verify off     3180.8      -51.2      -1.6    2218.14         -
          verify on     3245.0      +13.0      +0.4    2215.97         -
              baked     3164.2      -67.8      -2.1    2114.65         -
       bound tensor     3174.1      -57.9      -1.8       0.45   2117.83
      bound pointer     3219.6      -12.5      -0.4       0.44   2118.39
   fixed device map     3219.5      -12.5      -0.4       0.42   2158.06
fixed device no-map     3185.3      -46.7      -1.4       0.46   2046.05
elapsed_ns=23179719750
exit_status=0
ended=2026-09-27T12:54:36+08:00
```

### perf/round0_launch_host.txt
```
commit=be52e1c24c29ac6a277e81b5d65771e233586bbd
started=2026-09-27T12:54:36+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --no-gpu --iters 100000 --batches 9
mode=host-only; 100000 calls × 9 batches; median ns/call
               path    ns/call       Δ ns       Δ %   build ms   bind ms
           auto map       41.9       +0.0      +0.0    2358.38         -
        reduced key       41.5       -0.5      -1.2    2165.44         -
         verify off       42.2       +0.2      +0.6    2173.90         -
          verify on       43.0       +1.0      +2.5    2271.16         -
              baked       38.7       -3.2      -7.6    2051.96         -
       bound tensor       45.7       +3.8      +9.0       0.23   1988.00
      bound pointer       45.1       +3.1      +7.4       0.23   2111.31
   fixed device map       42.1       +0.1      +0.2       0.19   2207.96
fixed device no-map       44.1       +2.1      +5.0       0.25   2110.97
elapsed_ns=22403106548
exit_status=0
ended=2026-09-27T12:54:58+08:00
```

### perf/round0_launch_last_key.txt
```
commit=be52e1c24c29ac6a277e81b5d65771e233586bbd
started=2026-09-27T12:55:16+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --last-key --iters 100000 --batches 9
mode=last-key; host-only; 200000 calls × 9 batches; median ns/call
count     pattern    ns/call
    4      repeat       33.5
    4   alternate       35.7
   16      repeat       61.8
   16   alternate       64.7
   32      repeat       98.9
   32   alternate      103.4
elapsed_ns=3270048605
exit_status=0
ended=2026-09-27T12:55:19+08:00
```

### perf/round0_launch_readme.txt
```
commit=be52e1c24c29ac6a277e81b5d65771e233586bbd
started=2026-09-27T12:54:58+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --readme --iters 1000 --batches 9
mode=readme; 1000 calls × 9 batches; median
        path   triton us    intj us   speedup
   grid=(1,)       16.67       3.15      5.3x
   grid=(0,)       12.97       0.03    457.6x

torch_access_mode   decode ns    build s
     runtime_shim        90.8       0.79
   static_compile        91.8       0.13
      interpreter       930.6       0.84
elapsed_ns=5447548079
exit_status=0
ended=2026-09-27T12:55:04+08:00
```

### perf/round0_launch_sweep.txt
```
commit=be52e1c24c29ac6a277e81b5d65771e233586bbd
started=2026-09-27T12:55:04+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --sweep --iters 100000 --batches 9
mode=sweep; host-only; 100000 calls × 9 batches; median ns/call
count   kind    ns/call
    4    int       65.9
    4 tensor       67.3
   16    int       67.6
   16 tensor       63.9
   32    int      104.8
   32 tensor      107.6
elapsed_ns=12182787747
exit_status=0
ended=2026-09-27T12:55:16+08:00
```

### perf/round1_ffi_compare.txt
```
commit=be52e1c24c29ac6a277e81b5d65771e233586bbd
started=2026-09-27T12:57:05+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_ffi_compare.py --iters 1000 --batches 9
iters=1000 batches=9 unit=us/call (host-only, no timed sync)
FFI packed nop             median=0.1449 samples=[0.151225, 0.146558, 0.14465999999999998, 0.14481200000000002, 0.144352, 0.144945, 0.145506, 0.144907, 0.143397]
FFI typed nop              median=0.1486 samples=[0.148602, 0.15124100000000001, 0.146707, 0.147863, 0.151357, 0.150617, 0.15016300000000002, 0.14757800000000001, 0.14668299999999998]
FFI empty kernel           median=3.7018 samples=[3.645266, 3.91555, 3.701791, 4.218573, 3.4431979999999998, 4.3974269999999995, 3.5011889999999997, 4.393327, 3.55443]
INTJ empty kernel          median=3.6542 samples=[3.1680059999999997, 3.6542269999999997, 4.128284, 3.353509, 3.95322, 3.4773389999999997, 3.962837, 3.45115, 3.9289609999999997]
INTJ fixed-device kernel   median=3.3145 samples=[3.236324, 3.549751, 3.5918400000000004, 3.2143490000000003, 3.314508, 3.13958, 3.392324, 3.172826, 3.358631]
FFI packed nop mixed       median=0.1729 samples=[0.179372, 0.173243, 0.172441, 0.171375, 0.173373, 0.172892, 0.172041, 0.1767, 0.172477]
FFI typed nop mixed        median=0.1774 samples=[0.1733, 0.179335, 0.176737, 0.177448, 0.176364, 0.177791, 0.178248, 0.17826599999999998, 0.175748]
FFI mixed kernel           median=3.7047 samples=[3.704703, 3.933461, 3.633383, 3.761114, 3.562876, 4.419225, 3.6266819999999997, 4.416828000000001, 3.6413510000000002]
INTJ mixed kernel          median=3.4576 samples=[4.135823, 3.687553, 3.457575, 3.513839, 3.349957, 3.416747, 3.376073, 3.506024, 3.404744]
INTJ fixed mixed kernel    median=3.5480 samples=[3.628314, 3.654892, 3.459787, 3.548025, 3.873361, 3.946029, 3.441855, 3.501854, 3.51759]
elapsed_ns=3647518944
exit_status=0
ended=2026-09-27T12:57:08+08:00
```

### perf/round1_ffi_paths.txt
```
commit=be52e1c24c29ac6a277e81b5d65771e233586bbd
started=2026-09-27T12:57:13+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_intj_ffi_paths.py --mode all --iters 1000 --batches 9
iters=1000 batches=9 unit=ns/call; no_gpu=True
mode=callback
callback_count=9100 (100 warmup + 9000 unique misses)
INTJ cold compile callback + cache  median_ns=404.7 samples_ns=[446.852, 461.426, 374.251, 953.275, 364.023, 377.004, 388.607, 404.684, 1914.537]
INTJ hot call (no callback)         median_ns=38.4 samples_ns=[39.765, 39.809, 36.993, 38.376, 36.876, 39.565, 37.562, 38.766, 37.44]
INTJ 3-tensor host-only nop         median_ns=34.1 samples_ns=[36.35, 34.396, 34.08, 34.095, 33.893, 34.152, 33.876, 34.216, 33.958]
mode=kwargs
INTJ direct positional              median_ns=64.9 samples_ns=[68.127, 65.477, 64.865, 64.949, 69.239, 64.771, 65.207, 64.491, 64.834]
INTJ adapter positional             median_ns=94.2 samples_ns=[94.992, 94.016, 94.338, 94.371, 94.249, 92.626, 93.465, 95.663, 90.921]
INTJ adapter kwargs                 median_ns=107.0 samples_ns=[108.786, 107.011, 106.199, 107.081, 106.945, 107.288, 106.146, 109.697, 105.825]
INTJ adapter defaults               median_ns=88.7 samples_ns=[90.688, 89.323, 89.62, 88.495, 89.425, 88.25, 88.705, 88.258, 87.993]
INTJ FFI wrapper positional         median_ns=102.7 samples_ns=[122.125, 103.605, 102.9, 103.192, 102.347, 102.099, 102.444, 102.685, 102.141]
INTJ FFI wrapper kwargs             median_ns=129.0 samples_ns=[140.297, 129.587, 129.334, 128.865, 128.919, 128.304, 129.011, 127.747, 130.738]
INTJ native kwargs: unsupported (verified TypeError)
mode=dataclass
INTJ pair direct                    median_ns=65.8 samples_ns=[68.265, 65.987, 65.214, 65.767, 70.122, 65.829, 65.654, 65.955, 65.809]
INTJ pair prebuilt *tuple           median_ns=140.3 samples_ns=[137.58, 140.535, 136.536, 140.298, 140.835, 141.967, 138.417, 142.821, 137.687]
FFI unpack Pair only                median_ns=166.1 samples_ns=[163.264, 162.643, 176.317, 167.437, 163.363, 166.386, 165.761, 166.104, 171.624]
INTJ pair manual unpack             median_ns=172.3 samples_ns=[169.741, 172.29, 170.193, 172.967, 174.932, 173.486, 169.765, 173.931, 170.486]
INTJ pair FFI unpack                median_ns=312.8 samples_ns=[312.783, 312.341, 309.696, 312.585, 317.126, 314.953, 313.102, 317.324, 311.998]
INTJ pair stdlib astuple            median_ns=1153.5 samples_ns=[1201.122, 1189.408, 1143.601, 1153.461, 1134.354, 1147.783, 1398.515, 1158.051, 1150.934]
INTJ config direct                  median_ns=80.2 samples_ns=[80.545, 81.769, 80.195, 81.221, 79.476, 87.834, 77.597, 78.317, 77.551]
INTJ config FFI unpack              median_ns=316.2 samples_ns=[312.619, 318.054, 316.226, 312.983, 310.905, 318.762, 315.696, 318.233, 317.799]
INTJ native dataclass unpack: unsupported (verified TypeError)
elapsed_ns=2774446735
exit_status=0
ended=2026-09-27T12:57:15+08:00
```

### perf/round1_ffi_sweep.txt
```
commit=be52e1c24c29ac6a277e81b5d65771e233586bbd
started=2026-09-27T12:57:08+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_ffi_compare.py --sweep --iters 1000 --batches 9
counts=[0, 3, 5, 8, 16, 32, 64] iters=1000 batches=9 unit=us/call (host-only, no timed sync); INTJ device-bound, no specialization key
args= 0 FFI packed nop           median=0.1161 samples=[0.118975, 0.113834, 0.114402, 0.121757, 0.115966, 0.118279, 0.12103900000000001, 0.11605, 0.115912]
args= 0 FFI typed nop            median=0.1192 samples=[0.12121599999999999, 0.120411, 0.11748099999999999, 0.12212, 0.11810599999999999, 0.121988, 0.11762900000000001, 0.119208, 0.117738]
args= 0 FFI empty kernel         median=1.6238 samples=[1.581465, 1.861652, 1.5697349999999999, 1.882787, 1.568815, 1.859716, 1.623789, 1.853261, 1.572665]
args= 0 INTJ static_compile kernel median=2.7462 samples=[3.13843, 2.7941149999999997, 2.6869229999999997, 2.795748, 2.694778, 2.671927, 2.755187, 2.740003, 2.7461770000000003]
args= 0 INTJ runtime_shim kernel median=2.6840 samples=[2.874841, 2.667094, 2.700797, 2.680506, 2.682705, 2.683978, 2.773123, 2.68002, 2.7277739999999997]
args= 3 FFI packed nop           median=0.1443 samples=[0.14849, 0.142344, 0.14334, 0.148156, 0.143894, 0.148732, 0.14432599999999998, 0.145383, 0.14343]
args= 3 FFI typed nop            median=0.1485 samples=[0.150112, 0.14427299999999998, 0.148023, 0.150881, 0.148506, 0.150335, 0.148291, 0.149787, 0.143886]
args= 3 FFI empty kernel         median=3.2083 samples=[3.324344, 3.189158, 3.13644, 3.218563, 3.145377, 3.2254110000000003, 3.1251219999999997, 3.217398, 3.208265]
args= 3 INTJ static_compile kernel median=2.8680 samples=[2.998888, 2.80748, 2.972543, 2.8612480000000002, 2.9780390000000003, 2.853719, 2.952315, 2.8642350000000003, 2.868036]
args= 3 INTJ runtime_shim kernel median=2.8812 samples=[3.0351179999999998, 2.82329, 2.836707, 2.880284, 2.923073, 2.881158, 2.891982, 2.8777779999999997, 2.882355]
args= 5 FFI packed nop           median=0.1726 samples=[0.17105199999999998, 0.169327, 0.170641, 0.169917, 0.17347100000000001, 0.17255199999999998, 0.172601, 0.173933, 0.173339]
args= 5 FFI typed nop            median=0.1773 samples=[0.17826, 0.17846700000000001, 0.17568199999999998, 0.17544800000000002, 0.17413800000000001, 0.177348, 0.174203, 0.178583, 0.178091]
args= 5 FFI empty kernel         median=3.2770 samples=[3.372696, 3.274042, 3.181819, 3.302099, 3.185588, 3.2770349999999997, 3.450459, 3.308244, 3.254697]
args= 5 INTJ static_compile kernel median=2.8546 samples=[3.011977, 2.814622, 2.857981, 2.838948, 2.854617, 2.823814, 2.871205, 2.8278980000000002, 2.8758670000000004]
args= 5 INTJ runtime_shim kernel median=2.8340 samples=[3.027214, 2.789552, 2.834023, 2.770316, 2.847637, 2.764243, 2.845206, 2.781917, 2.847116]
args= 8 FFI packed nop           median=0.2065 samples=[0.20682599999999998, 0.20439500000000002, 0.207137, 0.204743, 0.20672200000000002, 0.20652299999999998, 0.20835499999999998, 0.20438800000000001, 0.20455099999999998]
args= 8 FFI typed nop            median=0.2084 samples=[0.20930500000000002, 0.208704, 0.209004, 0.20835, 0.204377, 0.207687, 0.20663399999999998, 0.209561, 0.207641]
args= 8 FFI empty kernel         median=3.3534 samples=[3.507784, 3.346752, 3.2513009999999998, 3.353429, 3.262168, 3.377339, 3.5352379999999997, 3.3889549999999997, 3.272877]
args= 8 INTJ static_compile kernel median=3.0073 samples=[3.058279, 2.893763, 3.0544059999999997, 2.9992330000000003, 3.0681030000000002, 2.989326, 3.007259, 2.9253899999999997, 3.040725]
args= 8 INTJ runtime_shim kernel median=2.8820 samples=[3.080902, 2.870263, 2.953898, 2.8820140000000003, 2.9738939999999996, 2.8778989999999998, 2.8661280000000002, 2.896407, 2.8687460000000002]
args=16 FFI packed nop           median=0.2913 samples=[0.291253, 0.298905, 0.290981, 0.288356, 0.29138200000000003, 0.296626, 0.292989, 0.289874, 0.291306]
args=16 FFI typed nop            median=0.3010 samples=[0.29855000000000004, 0.306151, 0.300343, 0.299473, 0.29889, 0.301341, 0.300976, 0.303685, 0.30169]
args=16 FFI empty kernel         median=3.6521 samples=[3.713127, 3.6807109999999996, 3.502665, 3.652055, 3.507181, 3.637517, 3.500995, 3.699602, 3.6639920000000004]
args=16 INTJ static_compile kernel median=3.3365 samples=[3.277358, 3.4013020000000003, 3.4862550000000003, 3.33648, 3.474599, 3.304903, 3.301438, 3.367732, 3.334101]
args=16 INTJ runtime_shim kernel median=3.2744 samples=[3.2768, 3.08652, 3.342116, 3.0408150000000003, 3.338301, 3.028091, 3.274426, 3.057538, 3.275285]
args=32 FFI packed nop           median=0.4791 samples=[0.47938099999999995, 0.49291199999999996, 0.490639, 0.472683, 0.481162, 0.476819, 0.47912299999999997, 0.47225900000000004, 0.47813799999999995]
args=32 FFI typed nop            median=0.4964 samples=[0.489759, 0.506717, 0.496394, 0.49761500000000003, 0.489008, 0.497121, 0.489716, 0.491608, 0.49724599999999997]
args=32 FFI empty kernel         median=4.1066 samples=[4.170852, 4.081049, 3.965681, 4.106618999999999, 3.9531460000000003, 4.112617, 3.974308, 4.128976, 4.124706]
args=32 INTJ static_compile kernel median=3.6179 samples=[3.4679, 3.626121, 3.625608, 3.566971, 3.6179029999999996, 3.529666, 3.722083, 3.5943359999999998, 3.774082]
args=32 INTJ runtime_shim kernel median=3.5359 samples=[3.469563, 3.535931, 3.6180120000000002, 3.44184, 3.602906, 3.427268, 3.636109, 3.454141, 3.647535]
args=64 FFI packed nop           median=0.8367 samples=[0.817988, 0.873614, 0.888073, 0.8578250000000001, 0.8324640000000001, 0.836672, 0.83304, 0.823318, 0.851133]
args=64 FFI typed nop            median=0.8662 samples=[0.882211, 0.877471, 0.913196, 0.8948630000000001, 0.8487619999999999, 0.865639, 0.856101, 0.854847, 0.866189]
args=64 FFI empty kernel         median=5.0690 samples=[5.146305999999999, 5.160616, 5.0824679999999995, 5.071077, 5.068957, 5.033097, 5.063402, 5.060685, 5.054201]
args=64 INTJ static_compile kernel median=4.5262 samples=[4.515051000000001, 4.511684, 4.526162, 4.557265999999999, 4.524812, 4.4946779999999995, 4.543968, 4.551234, 4.548881000000001]
args=64 INTJ runtime_shim kernel median=4.5256 samples=[4.48761, 4.502776, 4.522682, 4.579693, 4.503895000000001, 4.552595, 4.525603, 4.560611, 4.53632]
elapsed_ns=4191315610
exit_status=0
ended=2026-09-27T12:57:13+08:00
```

### perf/round1_hip_module.txt
```
commit=be52e1c24c29ac6a277e81b5d65771e233586bbd
started=2026-09-27T12:57:15+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_hip_module_launch.py --iters 1000 --batches 9
hsaco_sha256=df3e529e178196ad62f7c07ab8593ede5b8596c76e7c6fbd12a5d947a696e65c kernel=empty_kernel threads=256 shared=0 callback_count=1
iters=1000 batches=9 unit=us/call (host enqueue, no timed sync)
TVM FFI HIP HSACO         median=3.1187 samples=[3.415256, 3.187761, 3.201625, 3.1361950000000003, 3.1186640000000003, 3.065988, 3.0684839999999998, 3.039447, 3.079482]
Triton same HSACO         median=15.8475 samples=[16.016216, 15.948780000000001, 15.910977999999998, 15.915916, 15.735773, 15.753098, 15.800636, 15.84751, 15.683672]
INTJ same function        median=2.9066 samples=[3.016857, 2.9751260000000004, 2.96544, 2.9836660000000004, 2.900031, 2.832745, 2.836537, 2.906596, 2.812665]
elapsed_ns=3603166026
exit_status=0
ended=2026-09-27T12:57:19+08:00
```

### perf/round1_kernel_cache.txt
```
commit=be52e1c24c29ac6a277e81b5d65771e233586bbd
started=2026-09-27T12:57:19+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python -m pytest tests/test_kernel_cache.py -s -q

kernel cache, 1-word key, ns per operation

                      intj       tsl      absl
hash_only             1.36      1.36      1.36
hit/1                 1.57      2.01      1.44
hit/8                 1.75      2.09      4.36
hit/64                1.68      2.13      4.54
hit/512               2.31      2.91      5.33
hit_child             3.00      3.01      3.02
miss/1                1.26      1.57      1.38
miss/8                2.74      1.75      3.29
miss/64               2.74      1.75      3.52
miss/512              2.92      1.99      4.05

kernel cache, 2-word key, ns per operation

                      intj       tsl      absl
hash_only             1.42      1.42      1.42
hit/1                 1.86      2.57      2.02
hit/8                 1.82      2.57      5.04
hit/64                2.23      2.92      5.24
hit/512               3.03      4.04      6.39
hit_child             3.01      3.00      3.01
miss/1                1.38      2.33      1.50
miss/8                2.43      1.87      3.21
miss/64               3.16      2.10      3.88
miss/512              2.88      2.74      3.99

kernel cache, 5-word key, ns per operation

                      intj       tsl      absl
hash_only             3.58      3.59      3.59
hit/1                 3.00      3.05      2.46
hit/8                 3.20      3.41      9.49
hit/64                3.31      3.50      9.77
hit/512               4.53      4.78     11.17
hit_child             3.01      3.01      3.00
miss/1                1.41      2.10      2.20
miss/8                2.97      2.22      6.94
miss/64               2.86      1.96      6.72
miss/512              3.52      2.51      7.69
........
8 passed in 24.50s
elapsed_ns=25351793285
exit_status=0
ended=2026-09-27T12:57:44+08:00
```

### perf/round1_launch_gpu.txt
```
commit=be52e1c24c29ac6a277e81b5d65771e233586bbd
started=2026-09-27T12:56:48+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --iters 1000 --batches 9
mode=gpu; 1000 calls × 9 batches; median ns/call
               path    ns/call       Δ ns       Δ %   build ms   bind ms
           auto map     3197.5       +0.0      +0.0      74.63         -
        reduced key     3124.9      -72.6      -2.3      73.31         -
         verify off     2986.3     -211.2      -6.6       2.22         -
          verify on     2945.1     -252.4      -7.9       2.17         -
              baked     2938.1     -259.4      -8.1       2.27         -
       bound tensor     3055.0     -142.5      -4.5       0.36      2.16
      bound pointer     2966.8     -230.6      -7.2       0.35      2.11
   fixed device map     2908.3     -289.2      -9.0       0.31      2.12
fixed device no-map     2910.2     -287.3      -9.0       0.28      2.11
elapsed_ns=3614341106
exit_status=0
ended=2026-09-27T12:56:52+08:00
```

### perf/round1_launch_host.txt
```
commit=be52e1c24c29ac6a277e81b5d65771e233586bbd
started=2026-09-27T12:56:52+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --no-gpu --iters 100000 --batches 9
mode=host-only; 100000 calls × 9 batches; median ns/call
               path    ns/call       Δ ns       Δ %   build ms   bind ms
           auto map       42.2       +0.0      +0.0      62.16         -
        reduced key       42.4       +0.2      +0.4       2.07         -
         verify off       42.5       +0.3      +0.7       1.98         -
          verify on       43.1       +0.9      +2.1       2.08         -
              baked       38.7       -3.5      -8.4       2.00         -
       bound tensor       45.2       +3.0      +7.0       0.15      1.88
      bound pointer       44.3       +2.0      +4.8       0.15      1.79
   fixed device map       42.0       -0.3      -0.7       0.12      1.88
fixed device no-map       43.3       +1.1      +2.5       0.15      1.82
elapsed_ns=2855518573
exit_status=0
ended=2026-09-27T12:56:55+08:00
```

### perf/round1_launch_last_key.txt
```
commit=be52e1c24c29ac6a277e81b5d65771e233586bbd
started=2026-09-27T12:57:01+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --last-key --iters 100000 --batches 9
mode=last-key; host-only; 200000 calls × 9 batches; median ns/call
count     pattern    ns/call
    4      repeat       33.4
    4   alternate       35.5
   16      repeat       61.9
   16   alternate       64.9
   32      repeat       98.9
   32   alternate      103.3
elapsed_ns=3213363617
exit_status=0
ended=2026-09-27T12:57:05+08:00
```

### perf/round1_launch_readme.txt
```
commit=be52e1c24c29ac6a277e81b5d65771e233586bbd
started=2026-09-27T12:56:55+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --readme --iters 1000 --batches 9
mode=readme; 1000 calls × 9 batches; median
        path   triton us    intj us   speedup
   grid=(1,)       16.95       3.56      4.8x
   grid=(0,)       13.16       0.03    461.8x

torch_access_mode   decode ns    build s
     runtime_shim        89.7       0.02
   static_compile        91.1       0.06
      interpreter       873.2       0.00
elapsed_ns=3657548269
exit_status=0
ended=2026-09-27T12:56:59+08:00
```

### perf/round1_launch_sweep.txt
```
commit=be52e1c24c29ac6a277e81b5d65771e233586bbd
started=2026-09-27T12:56:59+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --sweep --iters 100000 --batches 9
mode=sweep; host-only; 100000 calls × 9 batches; median ns/call
count   kind    ns/call
    4    int       39.6
    4 tensor       38.2
   16    int       67.6
   16 tensor       63.9
   32    int      104.8
   32 tensor      107.6
elapsed_ns=2898319288
exit_status=0
ended=2026-09-27T12:57:01+08:00
```

### perf/round2_ffi_compare.txt
```
commit=be52e1c24c29ac6a277e81b5d65771e233586bbd
started=2026-09-27T12:58:01+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_ffi_compare.py --iters 1000 --batches 9
iters=1000 batches=9 unit=us/call (host-only, no timed sync)
FFI packed nop             median=0.1446 samples=[0.151797, 0.14457599999999998, 0.14414, 0.145534, 0.14418999999999998, 0.142854, 0.147027, 0.144504, 0.14575200000000002]
FFI typed nop              median=0.1496 samples=[0.15834, 0.14826, 0.147143, 0.149519, 0.14987899999999998, 0.149264, 0.14961000000000002, 0.15172999999999998, 0.15156299999999998]
FFI empty kernel           median=3.3477 samples=[3.554988, 3.393581, 3.36081, 3.1728180000000004, 3.180659, 3.347687, 3.191614, 3.350978, 3.18674]
INTJ empty kernel          median=2.9882 samples=[3.08227, 3.065448, 3.079377, 2.877119, 2.901134, 2.874098, 2.992464, 2.8672869999999997, 2.98817]
INTJ fixed-device kernel   median=2.8470 samples=[3.055184, 3.077575, 3.0542689999999997, 2.878228, 2.847026, 2.81758, 2.816134, 2.814487, 2.803741]
FFI packed nop mixed       median=0.1731 samples=[0.173112, 0.171739, 0.17258500000000002, 0.17191, 0.174151, 0.173093, 0.172743, 0.17966300000000002, 0.174293]
FFI typed nop mixed        median=0.1779 samples=[0.174179, 0.180013, 0.178543, 0.177886, 0.178234, 0.17982800000000002, 0.175014, 0.17610599999999998, 0.176154]
FFI mixed kernel           median=3.2599 samples=[3.381372, 3.421719, 3.259855, 3.242508, 3.189801, 3.317923, 3.242118, 3.391118, 3.250083]
INTJ mixed kernel          median=2.9145 samples=[3.150098, 3.098721, 2.914504, 2.920153, 2.8613739999999996, 2.914564, 2.833865, 2.90299, 2.8491350000000004]
INTJ fixed mixed kernel    median=2.8798 samples=[3.118582, 3.114134, 2.96046, 2.9501939999999998, 2.860434, 2.8445970000000003, 2.83127, 2.836224, 2.879802]
elapsed_ns=3607149187
exit_status=0
ended=2026-09-27T12:58:04+08:00
```

### perf/round2_ffi_paths.txt
```
commit=be52e1c24c29ac6a277e81b5d65771e233586bbd
started=2026-09-27T12:58:08+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_intj_ffi_paths.py --mode all --iters 1000 --batches 9
iters=1000 batches=9 unit=ns/call; no_gpu=True
mode=callback
callback_count=9100 (100 warmup + 9000 unique misses)
INTJ cold compile callback + cache  median_ns=406.3 samples_ns=[447.219, 455.226, 372.837, 930.287, 358.329, 382.943, 388.986, 406.292, 1852.987]
INTJ hot call (no callback)         median_ns=38.0 samples_ns=[38.655, 38.943, 39.854, 37.992, 37.208, 37.386, 37.353, 38.906, 37.267]
INTJ 3-tensor host-only nop         median_ns=34.1 samples_ns=[36.325, 34.438, 33.808, 34.09, 33.551, 33.992, 33.61, 34.096, 35.032]
mode=kwargs
INTJ direct positional              median_ns=65.4 samples_ns=[67.641, 65.41, 65.107, 65.069, 65.169, 65.121, 81.128, 65.491, 65.797]
INTJ adapter positional             median_ns=89.3 samples_ns=[89.941, 89.319, 92.696, 88.819, 88.752, 88.637, 89.143, 90.362, 93.893]
INTJ adapter kwargs                 median_ns=107.3 samples_ns=[107.656, 107.73, 107.897, 107.255, 106.863, 105.963, 107.01, 107.114, 111.407]
INTJ adapter defaults               median_ns=88.8 samples_ns=[90.949, 89.686, 88.326, 88.874, 88.788, 88.097, 87.454, 89.673, 88.771]
INTJ FFI wrapper positional         median_ns=104.4 samples_ns=[106.903, 112.54, 105.333, 104.266, 104.126, 104.254, 104.231, 104.711, 104.362]
INTJ FFI wrapper kwargs             median_ns=126.6 samples_ns=[126.356, 146.639, 126.984, 126.421, 125.669, 126.83, 124.652, 126.61, 126.789]
INTJ native kwargs: unsupported (verified TypeError)
mode=dataclass
INTJ pair direct                    median_ns=66.5 samples_ns=[68.322, 66.781, 65.677, 66.454, 65.657, 70.913, 65.996, 66.583, 66.045]
INTJ pair prebuilt *tuple           median_ns=137.5 samples_ns=[137.486, 139.383, 135.366, 139.374, 134.352, 149.925, 135.723, 138.663, 136.49]
FFI unpack Pair only                median_ns=164.4 samples_ns=[162.971, 164.605, 166.433, 163.396, 160.593, 164.366, 163.947, 165.015, 169.387]
INTJ pair manual unpack             median_ns=171.2 samples_ns=[171.245, 175.133, 171.04, 173.421, 170.392, 176.126, 167.99, 172.682, 169.9]
INTJ pair FFI unpack                median_ns=312.7 samples_ns=[311.478, 317.378, 310.309, 313.26, 323.709, 312.651, 310.68, 322.962, 307.81]
INTJ pair stdlib astuple            median_ns=1145.4 samples_ns=[1202.69, 1186.721, 1136.028, 1146.285, 1133.153, 1145.483, 1131.025, 1145.384, 1141.089]
INTJ config direct                  median_ns=78.8 samples_ns=[78.771, 78.945, 78.101, 78.953, 78.119, 78.802, 77.754, 79.403, 77.798]
INTJ config FFI unpack              median_ns=316.4 samples_ns=[317.954, 316.155, 313.768, 320.199, 314.946, 318.471, 318.115, 316.427, 313.799]
INTJ native dataclass unpack: unsupported (verified TypeError)
elapsed_ns=2775087014
exit_status=0
ended=2026-09-27T12:58:11+08:00
```

### perf/round2_ffi_sweep.txt
```
commit=be52e1c24c29ac6a277e81b5d65771e233586bbd
started=2026-09-27T12:58:04+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_ffi_compare.py --sweep --iters 1000 --batches 9
counts=[0, 3, 5, 8, 16, 32, 64] iters=1000 batches=9 unit=us/call (host-only, no timed sync); INTJ device-bound, no specialization key
args= 0 FFI packed nop           median=0.1171 samples=[0.13600399999999999, 0.117071, 0.11629600000000001, 0.117146, 0.115687, 0.11787399999999999, 0.118379, 0.11707899999999999, 0.118051]
args= 0 FFI typed nop            median=0.1204 samples=[0.11739400000000001, 0.12175799999999999, 0.11650400000000001, 0.12166800000000001, 0.119514, 0.123022, 0.12041299999999999, 0.120365, 0.11872100000000001]
args= 0 FFI empty kernel         median=1.6607 samples=[1.61979, 1.953443, 1.6606500000000002, 1.925669, 1.625219, 1.900733, 1.64478, 1.87661, 1.629648]
args= 0 INTJ static_compile kernel median=2.6974 samples=[3.140479, 2.697407, 2.7036770000000003, 2.702129, 2.686673, 2.690053, 2.684852, 2.65889, 2.7085630000000003]
args= 0 INTJ runtime_shim kernel median=2.6892 samples=[2.8941559999999997, 2.659476, 2.711297, 2.650881, 2.689188, 2.655623, 2.696505, 2.867023, 2.685757]
args= 3 FFI packed nop           median=0.1441 samples=[0.14291800000000002, 0.145131, 0.143685, 0.144141, 0.146661, 0.142494, 0.143903, 0.14474199999999998, 0.149124]
args= 3 FFI typed nop            median=0.1483 samples=[0.150293, 0.154637, 0.147749, 0.150209, 0.14832800000000002, 0.147867, 0.14753899999999998, 0.149852, 0.14818]
args= 3 FFI empty kernel         median=3.2695 samples=[3.389246, 3.2793319999999997, 3.179552, 3.3004119999999997, 3.171275, 3.269538, 3.16225, 3.321955, 3.179208]
args= 3 INTJ static_compile kernel median=2.9594 samples=[3.017376, 2.788557, 2.982961, 2.832335, 2.981463, 2.9072489999999998, 2.95939, 2.85438, 2.97911]
args= 3 INTJ runtime_shim kernel median=2.8782 samples=[3.013854, 2.756228, 2.8917919999999997, 2.7947330000000004, 2.878175, 2.795303, 2.889394, 2.783765, 2.882576]
args= 5 FFI packed nop           median=0.1754 samples=[0.176255, 0.175383, 0.174822, 0.174916, 0.176074, 0.175967, 0.17650200000000002, 0.17330500000000001, 0.173514]
args= 5 FFI typed nop            median=0.1810 samples=[0.177918, 0.182189, 0.17969, 0.181777, 0.18291200000000002, 0.181675, 0.18009999999999998, 0.18095599999999998, 0.181043]
args= 5 FFI empty kernel         median=3.2971 samples=[3.382547, 3.331124, 3.284346, 3.316338, 3.280227, 3.30026, 3.289152, 3.297091, 3.2945100000000003]
args= 5 INTJ static_compile kernel median=2.8785 samples=[3.0134630000000002, 2.834741, 2.886741, 2.8567359999999997, 2.879883, 2.848815, 2.8785100000000003, 2.8533380000000004, 2.882358]
args= 5 INTJ runtime_shim kernel median=2.8686 samples=[3.087055, 2.805932, 2.8623670000000003, 2.870459, 2.867414, 2.868575, 2.869206, 2.877979, 2.864973]
args= 8 FFI packed nop           median=0.2099 samples=[0.21384999999999998, 0.21335400000000002, 0.209356, 0.209918, 0.211018, 0.20977, 0.20916300000000002, 0.21376499999999998, 0.209786]
args= 8 FFI typed nop            median=0.2131 samples=[0.213076, 0.21475899999999998, 0.212808, 0.21843, 0.20952600000000002, 0.21586000000000002, 0.211702, 0.21598699999999998, 0.21287999999999999]
args= 8 FFI empty kernel         median=3.3817 samples=[3.516625, 3.3781619999999997, 3.475562, 3.4056550000000003, 3.288619, 3.39443, 3.290787, 3.381729, 3.293528]
args= 8 INTJ static_compile kernel median=3.0217 samples=[3.078613, 2.898479, 3.058619, 2.971488, 3.031498, 2.972827, 3.0216950000000002, 2.9709830000000004, 3.043173]
args= 8 INTJ runtime_shim kernel median=2.8856 samples=[3.105949, 2.880268, 3.006652, 2.885552, 2.87353, 2.886337, 2.852259, 2.888011, 2.874943]
args=16 FFI packed nop           median=0.3028 samples=[0.306563, 0.302769, 0.300996, 0.300722, 0.30276400000000003, 0.300084, 0.302329, 0.306791, 0.305829]
args=16 FFI typed nop            median=0.3138 samples=[0.30981000000000003, 0.309067, 0.314587, 0.318211, 0.313236, 0.316976, 0.313839, 0.313512, 0.313935]
args=16 FFI empty kernel         median=3.6617 samples=[3.7488919999999997, 3.7146350000000004, 3.684855, 3.640396, 3.548971, 3.661736, 3.518203, 3.66924, 3.4949079999999997]
args=16 INTJ static_compile kernel median=3.3178 samples=[3.301125, 3.238439, 3.317847, 3.318721, 3.470609, 3.294244, 3.373069, 3.276437, 3.359627]
args=16 INTJ runtime_shim kernel median=3.2868 samples=[3.286811, 3.118282, 3.4093299999999997, 3.0264789999999997, 3.309473, 3.020913, 3.305273, 3.046953, 3.2993609999999998]
args=32 FFI packed nop           median=0.5119 samples=[0.508875, 0.510992, 0.502731, 0.5119130000000001, 0.509869, 0.514745, 0.5145299999999999, 0.523408, 0.5214840000000001]
args=32 FFI typed nop            median=0.5303 samples=[0.5338740000000001, 0.530316, 0.5201549999999999, 0.504309, 0.52211, 0.526908, 0.537259, 0.5359299999999999, 0.5387609999999999]
args=32 FFI empty kernel         median=4.1329 samples=[4.206252, 4.143696, 4.006234, 4.152336, 3.993765, 4.141513, 4.003365, 4.132912999999999, 3.999843]
args=32 INTJ static_compile kernel median=3.6221 samples=[3.4736860000000003, 3.681794, 3.627844, 3.543906, 3.622066, 3.568299, 3.639446, 3.560778, 3.6467080000000003]
args=32 INTJ runtime_shim kernel median=3.5116 samples=[3.4670880000000004, 3.511557, 3.5766999999999998, 3.420919, 3.57837, 3.421198, 3.574237, 3.423922, 3.589862]
args=64 FFI packed nop           median=0.9098 samples=[0.910832, 0.906088, 0.909833, 0.899726, 0.91025, 0.906375, 0.932024, 0.8871169999999999, 0.941686]
args=64 FFI typed nop            median=0.9288 samples=[0.9288390000000001, 0.9346760000000001, 0.923652, 0.932619, 0.926558, 0.927706, 0.948589, 0.928181, 0.952016]
args=64 FFI empty kernel         median=5.0788 samples=[5.165005, 5.07876, 5.075569000000001, 5.07352, 5.069537, 5.071305000000001, 5.079225999999999, 5.091772, 5.084649000000001]
args=64 INTJ static_compile kernel median=4.5389 samples=[4.569853, 4.512159, 4.535632, 4.554152, 4.559951, 4.535688, 4.538934, 4.571768, 4.522978]
args=64 INTJ runtime_shim kernel median=4.5251 samples=[4.525128, 4.505675, 4.503753, 4.5398000000000005, 4.526086, 4.5508299999999995, 4.510586, 4.597233, 4.522608]
elapsed_ns=4182170041
exit_status=0
ended=2026-09-27T12:58:08+08:00
```

### perf/round2_hip_module.txt
```
commit=be52e1c24c29ac6a277e81b5d65771e233586bbd
started=2026-09-27T12:58:11+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_hip_module_launch.py --iters 1000 --batches 9
hsaco_sha256=df3e529e178196ad62f7c07ab8593ede5b8596c76e7c6fbd12a5d947a696e65c kernel=empty_kernel threads=256 shared=0 callback_count=1
iters=1000 batches=9 unit=us/call (host enqueue, no timed sync)
TVM FFI HIP HSACO         median=3.0952 samples=[3.327883, 3.210627, 3.203138, 3.145882, 3.0952330000000003, 3.061152, 3.089593, 3.043113, 3.041617]
Triton same HSACO         median=15.9201 samples=[15.995739, 15.954702, 15.923356, 15.92206, 15.829015, 15.762267, 15.920093999999999, 15.750641, 15.768186]
INTJ same function        median=3.0092 samples=[3.0837269999999997, 3.106258, 3.117288, 3.0468200000000003, 2.918717, 2.889732, 2.921322, 3.009174, 2.951425]
elapsed_ns=3598741195
exit_status=0
ended=2026-09-27T12:58:15+08:00
```

### perf/round2_kernel_cache.txt
```
commit=be52e1c24c29ac6a277e81b5d65771e233586bbd
started=2026-09-27T12:58:15+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python -m pytest tests/test_kernel_cache.py -s -q

kernel cache, 1-word key, ns per operation

                      intj       tsl      absl
hash_only             1.36      1.36      1.35
hit/1                 1.56      2.01      1.44
hit/8                 1.75      2.10      4.36
hit/64                1.81      2.12      4.54
hit/512               2.30      2.93      5.31
hit_child             3.01      3.00      3.00
miss/1                1.21      1.79      1.38
miss/8                2.78      1.76      3.27
miss/64               2.76      1.76      3.53
miss/512              2.95      1.99      4.06

kernel cache, 2-word key, ns per operation

                      intj       tsl      absl
hash_only             1.42      1.42      1.42
hit/1                 1.94      2.57      2.04
hit/8                 2.40      2.57      5.04
hit/64                2.25      2.92      5.24
hit/512               3.02      4.04      6.38
hit_child             3.00      3.00      3.01
miss/1                1.54      2.32      1.45
miss/8                2.42      1.85      3.21
miss/64               3.13      2.11      3.89
miss/512              2.89      2.72      4.00

kernel cache, 5-word key, ns per operation

                      intj       tsl      absl
hash_only             3.58      3.59      3.58
hit/1                 3.00      3.07      2.48
hit/8                 3.20      3.41      9.49
hit/64                3.27      3.50      9.78
hit/512               4.55      4.78     11.21
hit_child             3.02      3.01      3.01
miss/1                1.38      2.11      2.07
miss/8                2.99      2.27      6.93
miss/64               2.86      1.97      6.71
miss/512              3.52      2.50      7.71
........
8 passed in 24.53s
elapsed_ns=25393630482
exit_status=0
ended=2026-09-27T12:58:40+08:00
```

### perf/round2_launch_gpu.txt
```
commit=be52e1c24c29ac6a277e81b5d65771e233586bbd
started=2026-09-27T12:57:44+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --iters 1000 --batches 9
mode=gpu; 1000 calls × 9 batches; median ns/call
               path    ns/call       Δ ns       Δ %   build ms   bind ms
           auto map     3137.9       +0.0      +0.0      74.49         -
        reduced key     3131.0       -6.8      -0.2      73.44         -
         verify off     2985.3     -152.6      -4.9       2.20         -
          verify on     2967.7     -170.2      -5.4       2.15         -
              baked     2906.6     -231.3      -7.4       2.25         -
       bound tensor     2951.3     -186.5      -5.9       0.36      2.16
      bound pointer     2937.8     -200.1      -6.4       0.37      2.13
   fixed device map     2885.9     -251.9      -8.0       0.32      2.12
fixed device no-map     2894.6     -243.2      -7.8       0.28      2.12
elapsed_ns=3609316576
exit_status=0
ended=2026-09-27T12:57:48+08:00
```

### perf/round2_launch_host.txt
```
commit=be52e1c24c29ac6a277e81b5d65771e233586bbd
started=2026-09-27T12:57:48+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --no-gpu --iters 100000 --batches 9
mode=host-only; 100000 calls × 9 batches; median ns/call
               path    ns/call       Δ ns       Δ %   build ms   bind ms
           auto map       42.4       +0.0      +0.0      62.58         -
        reduced key       42.4       -0.0      -0.1       2.09         -
         verify off       42.4       -0.1      -0.1       1.96         -
          verify on       43.1       +0.6      +1.5       2.06         -
              baked       39.2       -3.2      -7.6       1.90         -
       bound tensor      110.9      +68.4    +161.4       0.15      1.75
      bound pointer       44.5       +2.1      +4.9       0.15      1.78
   fixed device map       42.2       -0.2      -0.4       0.12      1.72
fixed device no-map       44.1       +1.7      +3.9       0.15      1.73
elapsed_ns=2919541841
exit_status=0
ended=2026-09-27T12:57:51+08:00
```

### perf/round2_launch_last_key.txt
```
commit=be52e1c24c29ac6a277e81b5d65771e233586bbd
started=2026-09-27T12:57:57+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --last-key --iters 100000 --batches 9
mode=last-key; host-only; 200000 calls × 9 batches; median ns/call
count     pattern    ns/call
    4      repeat       33.6
    4   alternate       35.5
   16      repeat       61.7
   16   alternate       64.7
   32      repeat       98.7
   32   alternate      103.3
elapsed_ns=3232013855
exit_status=0
ended=2026-09-27T12:58:01+08:00
```

### perf/round2_launch_readme.txt
```
commit=be52e1c24c29ac6a277e81b5d65771e233586bbd
started=2026-09-27T12:57:51+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --readme --iters 1000 --batches 9
mode=readme; 1000 calls × 9 batches; median
        path   triton us    intj us   speedup
   grid=(1,)       16.51       3.13      5.3x
   grid=(0,)       12.95       0.03    460.0x

torch_access_mode   decode ns    build s
     runtime_shim        91.2       0.02
   static_compile        90.6       0.06
      interpreter       875.8       0.00
elapsed_ns=3645031779
exit_status=0
ended=2026-09-27T12:57:54+08:00
```

### perf/round2_launch_sweep.txt
```
commit=be52e1c24c29ac6a277e81b5d65771e233586bbd
started=2026-09-27T12:57:54+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --sweep --iters 100000 --batches 9
mode=sweep; host-only; 100000 calls × 9 batches; median ns/call
count   kind    ns/call
    4    int       39.9
    4 tensor       38.4
   16    int       67.5
   16 tensor       64.0
   32    int      105.7
   32 tensor      106.0
elapsed_ns=2881950166
exit_status=0
ended=2026-09-27T12:57:57+08:00
```

### kernel cache final3.jsonl (raw, 9dacc19=c2 vs be52e1c=c3)
```
{"bin": "c2_1", "round": 1, "name": "hit/1", "ns": 1.5904903201564986, "err": null}
{"bin": "c2_1", "round": 1, "name": "hit/8", "ns": 1.660966263243655, "err": null}
{"bin": "c2_1", "round": 1, "name": "hit/64", "ns": 1.9140468652426708, "err": null}
{"bin": "c2_1", "round": 1, "name": "hit/512", "ns": 2.348330246456126, "err": null}
{"bin": "c2_1", "round": 1, "name": "miss/1", "ns": 1.2706727546796281, "err": null}
{"bin": "c2_1", "round": 1, "name": "miss/8", "ns": 2.794536380458945, "err": null}
{"bin": "c2_1", "round": 1, "name": "miss/64", "ns": 2.8690222208456704, "err": null}
{"bin": "c2_1", "round": 1, "name": "miss/512", "ns": 3.149309594071891, "err": null}
{"bin": "c2_1", "round": 1, "name": "hash_only", "ns": 1.8956624060060894, "err": null}
{"bin": "c2_1", "round": 1, "name": "hit_child", "ns": 3.4706320164992857, "err": null}
{"bin": "c3_1", "round": 1, "name": "hit/1", "ns": 1.5583200831897281, "err": null}
{"bin": "c3_1", "round": 1, "name": "hit/8", "ns": 1.6001524605129076, "err": null}
{"bin": "c3_1", "round": 1, "name": "hit/64", "ns": 1.6740017881495934, "err": null}
{"bin": "c3_1", "round": 1, "name": "hit/512", "ns": 2.2927927832814134, "err": null}
{"bin": "c3_1", "round": 1, "name": "miss/1", "ns": 1.2288799122793888, "err": null}
{"bin": "c3_1", "round": 1, "name": "miss/8", "ns": 2.80400387597767, "err": null}
{"bin": "c3_1", "round": 1, "name": "miss/64", "ns": 3.1781462629727253, "err": null}
{"bin": "c3_1", "round": 1, "name": "miss/512", "ns": 2.8881061823249077, "err": null}
{"bin": "c3_1", "round": 1, "name": "hash_only", "ns": 1.3534662684287335, "err": null}
{"bin": "c3_1", "round": 1, "name": "hit_child", "ns": 3.0081668303561844, "err": null}
{"bin": "c2_2", "round": 1, "name": "hit/1", "ns": 2.0643424571900475, "err": null}
{"bin": "c2_2", "round": 1, "name": "hit/8", "ns": 2.0611748674828867, "err": null}
{"bin": "c2_2", "round": 1, "name": "hit/64", "ns": 2.3326769697359526, "err": null}
{"bin": "c2_2", "round": 1, "name": "hit/512", "ns": 3.118548305958772, "err": null}
{"bin": "c2_2", "round": 1, "name": "miss/1", "ns": 1.3979263732574836, "err": null}
{"bin": "c2_2", "round": 1, "name": "miss/8", "ns": 3.5053277001387584, "err": null}
{"bin": "c2_2", "round": 1, "name": "miss/64", "ns": 5.661592539399862, "err": null}
{"bin": "c2_2", "round": 1, "name": "miss/512", "ns": 3.951546114829384, "err": null}
{"bin": "c2_2", "round": 1, "name": "hash_only", "ns": 1.420067556714244, "err": null}
{"bin": "c2_2", "round": 1, "name": "hit_child", "ns": 3.4613321616351787, "err": null}
{"bin": "c3_2", "round": 1, "name": "hit/1", "ns": 1.8169155246526452, "err": null}
{"bin": "c3_2", "round": 1, "name": "hit/8", "ns": 1.9467247083278008, "err": null}
{"bin": "c3_2", "round": 1, "name": "hit/64", "ns": 2.1855803328353307, "err": null}
{"bin": "c3_2", "round": 1, "name": "hit/512", "ns": 3.0167098452682803, "err": null}
{"bin": "c3_2", "round": 1, "name": "miss/1", "ns": 1.3840895615395028, "err": null}
{"bin": "c3_2", "round": 1, "name": "miss/8", "ns": 2.4145042077155225, "err": null}
{"bin": "c3_2", "round": 1, "name": "miss/64", "ns": 3.109538178930591, "err": null}
{"bin": "c3_2", "round": 1, "name": "miss/512", "ns": 4.353441769387146, "err": null}
{"bin": "c3_2", "round": 1, "name": "hash_only", "ns": 1.4210907273166518, "err": null}
{"bin": "c3_2", "round": 1, "name": "hit_child", "ns": 3.011018934786606, "err": null}
{"bin": "c2_5", "round": 1, "name": "hit/1", "ns": 3.0391638940498322, "err": null}
{"bin": "c2_5", "round": 1, "name": "hit/8", "ns": 3.2726056932995804, "err": null}
{"bin": "c2_5", "round": 1, "name": "hit/64", "ns": 3.3471481678336272, "err": null}
{"bin": "c2_5", "round": 1, "name": "hit/512", "ns": 4.671741297989267, "err": null}
{"bin": "c2_5", "round": 1, "name": "miss/1", "ns": 1.3943786343717233, "err": null}
{"bin": "c2_5", "round": 1, "name": "miss/8", "ns": 2.9742931125424263, "err": null}
{"bin": "c2_5", "round": 1, "name": "miss/64", "ns": 2.852865779491841, "err": null}
{"bin": "c2_5", "round": 1, "name": "miss/512", "ns": 3.4864312494979632, "err": null}
{"bin": "c2_5", "round": 1, "name": "hash_only", "ns": 3.5922502804065024, "err": null}
{"bin": "c2_5", "round": 1, "name": "hit_child", "ns": 5.237327785914383, "err": null}
{"bin": "c3_5", "round": 1, "name": "hit/1", "ns": 2.9893154553170906, "err": null}
{"bin": "c3_5", "round": 1, "name": "hit/8", "ns": 3.197781713704961, "err": null}
{"bin": "c3_5", "round": 1, "name": "hit/64", "ns": 3.2664269973733617, "err": null}
{"bin": "c3_5", "round": 1, "name": "hit/512", "ns": 4.5428193840302376, "err": null}
{"bin": "c3_5", "round": 1, "name": "miss/1", "ns": 1.3831438799521478, "err": null}
{"bin": "c3_5", "round": 1, "name": "miss/8", "ns": 2.98594761758335, "err": null}
{"bin": "c3_5", "round": 1, "name": "miss/64", "ns": 2.8532354185388025, "err": null}
{"bin": "c3_5", "round": 1, "name": "miss/512", "ns": 3.4756860219612356, "err": null}
{"bin": "c3_5", "round": 1, "name": "hash_only", "ns": 3.5749106065377165, "err": null}
{"bin": "c3_5", "round": 1, "name": "hit_child", "ns": 3.0036753176731184, "err": null}
{"bin": "c2_1", "round": 2, "name": "hit/1", "ns": 1.6059210057316076, "err": null}
{"bin": "c2_1", "round": 2, "name": "hit/8", "ns": 1.6639206788144663, "err": null}
{"bin": "c2_1", "round": 2, "name": "hit/64", "ns": 1.7201348591185937, "err": null}
{"bin": "c2_1", "round": 2, "name": "hit/512", "ns": 2.3516993338746914, "err": null}
{"bin": "c2_1", "round": 2, "name": "miss/1", "ns": 1.2708246669215897, "err": null}
{"bin": "c2_1", "round": 2, "name": "miss/8", "ns": 2.776199828093004, "err": null}
{"bin": "c2_1", "round": 2, "name": "miss/64", "ns": 2.8574027898541012, "err": null}
{"bin": "c2_1", "round": 2, "name": "miss/512", "ns": 3.176290860006637, "err": null}
{"bin": "c2_1", "round": 2, "name": "hash_only", "ns": 1.3553845150700061, "err": null}
{"bin": "c2_1", "round": 2, "name": "hit_child", "ns": 3.4668084296895585, "err": null}
{"bin": "c3_1", "round": 2, "name": "hit/1", "ns": 1.5600993295563534, "err": null}
{"bin": "c3_1", "round": 2, "name": "hit/8", "ns": 1.694386779484423, "err": null}
{"bin": "c3_1", "round": 2, "name": "hit/64", "ns": 1.6298938568696673, "err": null}
{"bin": "c3_1", "round": 2, "name": "hit/512", "ns": 2.291829941135401, "err": null}
{"bin": "c3_1", "round": 2, "name": "miss/1", "ns": 1.2215954962523736, "err": null}
{"bin": "c3_1", "round": 2, "name": "miss/8", "ns": 2.606883938565207, "err": null}
{"bin": "c3_1", "round": 2, "name": "miss/64", "ns": 2.7332444359860006, "err": null}
{"bin": "c3_1", "round": 2, "name": "miss/512", "ns": 2.9165783545268598, "err": null}
{"bin": "c3_1", "round": 2, "name": "hash_only", "ns": 1.4488074038003542, "err": null}
{"bin": "c3_1", "round": 2, "name": "hit_child", "ns": 2.9987102520275113, "err": null}
{"bin": "c2_2", "round": 2, "name": "hit/1", "ns": 2.206136046734492, "err": null}
{"bin": "c2_2", "round": 2, "name": "hit/8", "ns": 2.0632840913818815, "err": null}
{"bin": "c2_2", "round": 2, "name": "hit/64", "ns": 2.591716803137183, "err": null}
{"bin": "c2_2", "round": 2, "name": "hit/512", "ns": 3.1194407106809683, "err": null}
{"bin": "c2_2", "round": 2, "name": "miss/1", "ns": 1.5851056834388226, "err": null}
{"bin": "c2_2", "round": 2, "name": "miss/8", "ns": 3.454298990920696, "err": null}
{"bin": "c2_2", "round": 2, "name": "miss/64", "ns": 5.020030029118061, "err": null}
{"bin": "c2_2", "round": 2, "name": "miss/512", "ns": 3.954770801197634, "err": null}
{"bin": "c2_2", "round": 2, "name": "hash_only", "ns": 1.4949760054142918, "err": null}
{"bin": "c2_2", "round": 2, "name": "hit_child", "ns": 3.4550117735500954, "err": null}
{"bin": "c3_2", "round": 2, "name": "hit/1", "ns": 1.900089627216651, "err": null}
{"bin": "c3_2", "round": 2, "name": "hit/8", "ns": 1.8238863375201804, "err": null}
{"bin": "c3_2", "round": 2, "name": "hit/64", "ns": 2.1952341493609993, "err": null}
{"bin": "c3_2", "round": 2, "name": "hit/512", "ns": 3.245900492336794, "err": null}
{"bin": "c3_2", "round": 2, "name": "miss/1", "ns": 1.3831918319808414, "err": null}
{"bin": "c3_2", "round": 2, "name": "miss/8", "ns": 2.3860049783906416, "err": null}
{"bin": "c3_2", "round": 2, "name": "miss/64", "ns": 3.8026307152093897, "err": null}
{"bin": "c3_2", "round": 2, "name": "miss/512", "ns": 2.880567278518815, "err": null}
{"bin": "c3_2", "round": 2, "name": "hash_only", "ns": 1.4203606450029658, "err": null}
{"bin": "c3_2", "round": 2, "name": "hit_child", "ns": 3.498640105455696, "err": null}
{"bin": "c2_5", "round": 2, "name": "hit/1", "ns": 3.052330824738096, "err": null}
{"bin": "c2_5", "round": 2, "name": "hit/8", "ns": 3.273178446102286, "err": null}
{"bin": "c2_5", "round": 2, "name": "hit/64", "ns": 4.186116041947442, "err": null}
{"bin": "c2_5", "round": 2, "name": "hit/512", "ns": 4.671310209934094, "err": null}
{"bin": "c2_5", "round": 2, "name": "miss/1", "ns": 1.3882189208651938, "err": null}
{"bin": "c2_5", "round": 2, "name": "miss/8", "ns": 3.014198216930566, "err": null}
{"bin": "c2_5", "round": 2, "name": "miss/64", "ns": 3.531065540463549, "err": null}
{"bin": "c2_5", "round": 2, "name": "miss/512", "ns": 3.4564699601097444, "err": null}
{"bin": "c2_5", "round": 2, "name": "hash_only", "ns": 3.5811390863760653, "err": null}
{"bin": "c2_5", "round": 2, "name": "hit_child", "ns": 3.4557537351480074, "err": null}
{"bin": "c3_5", "round": 2, "name": "hit/1", "ns": 3.9635981573131787, "err": null}
{"bin": "c3_5", "round": 2, "name": "hit/8", "ns": 3.1967707541677757, "err": null}
{"bin": "c3_5", "round": 2, "name": "hit/64", "ns": 3.2667529390716665, "err": null}
{"bin": "c3_5", "round": 2, "name": "hit/512", "ns": 4.525033824063071, "err": null}
{"bin": "c3_5", "round": 2, "name": "miss/1", "ns": 1.8152495577101662, "err": null}
{"bin": "c3_5", "round": 2, "name": "miss/8", "ns": 3.008763125101201, "err": null}
{"bin": "c3_5", "round": 2, "name": "miss/64", "ns": 2.8676089745915148, "err": null}
{"bin": "c3_5", "round": 2, "name": "miss/512", "ns": 3.459563642508098, "err": null}
{"bin": "c3_5", "round": 2, "name": "hash_only", "ns": 3.576027963795132, "err": null}
{"bin": "c3_5", "round": 2, "name": "hit_child", "ns": 3.6979064312294234, "err": null}
{"bin": "c2_1", "round": 3, "name": "hit/1", "ns": 1.5930437594095788, "err": null}
{"bin": "c2_1", "round": 3, "name": "hit/8", "ns": 1.6667115006229933, "err": null}
{"bin": "c2_1", "round": 3, "name": "hit/64", "ns": 1.7558570578378105, "err": null}
{"bin": "c2_1", "round": 3, "name": "hit/512", "ns": 2.3475906258196884, "err": null}
{"bin": "c2_1", "round": 3, "name": "miss/1", "ns": 1.704905579365287, "err": null}
{"bin": "c2_1", "round": 3, "name": "miss/8", "ns": 2.739392314964548, "err": null}
{"bin": "c2_1", "round": 3, "name": "miss/64", "ns": 2.8246852391347432, "err": null}
{"bin": "c2_1", "round": 3, "name": "miss/512", "ns": 3.1811457556353813, "err": null}
{"bin": "c2_1", "round": 3, "name": "hash_only", "ns": 1.3533903251858634, "err": null}
{"bin": "c2_1", "round": 3, "name": "hit_child", "ns": 3.461414174809488, "err": null}
{"bin": "c3_1", "round": 3, "name": "hit/1", "ns": 1.8979272862205243, "err": null}
{"bin": "c3_1", "round": 3, "name": "hit/8", "ns": 1.737037208146204, "err": null}
{"bin": "c3_1", "round": 3, "name": "hit/64", "ns": 1.6932358116711121, "err": null}
{"bin": "c3_1", "round": 3, "name": "hit/512", "ns": 2.299886727947389, "err": null}
{"bin": "c3_1", "round": 3, "name": "miss/1", "ns": 1.2310381318620418, "err": null}
{"bin": "c3_1", "round": 3, "name": "miss/8", "ns": 2.756061570405708, "err": null}
{"bin": "c3_1", "round": 3, "name": "miss/64", "ns": 2.7675864386642757, "err": null}
{"bin": "c3_1", "round": 3, "name": "miss/512", "ns": 2.970177953725066, "err": null}
{"bin": "c3_1", "round": 3, "name": "hash_only", "ns": 1.6603123629822594, "err": null}
{"bin": "c3_1", "round": 3, "name": "hit_child", "ns": 2.998128502538445, "err": null}
{"bin": "c2_2", "round": 3, "name": "hit/1", "ns": 2.0632917408439497, "err": null}
{"bin": "c2_2", "round": 3, "name": "hit/8", "ns": 2.0657355477868053, "err": null}
{"bin": "c2_2", "round": 3, "name": "hit/64", "ns": 2.3337338600375337, "err": null}
{"bin": "c2_2", "round": 3, "name": "hit/512", "ns": 3.1188181224338054, "err": null}
{"bin": "c2_2", "round": 3, "name": "miss/1", "ns": 1.4567108093507828, "err": null}
{"bin": "c2_2", "round": 3, "name": "miss/8", "ns": 5.411058198660612, "err": null}
{"bin": "c2_2", "round": 3, "name": "miss/64", "ns": 4.426948810715423, "err": null}
{"bin": "c2_2", "round": 3, "name": "miss/512", "ns": 3.9532914319170724, "err": null}
{"bin": "c2_2", "round": 3, "name": "hash_only", "ns": 1.420566266918168, "err": null}
{"bin": "c2_2", "round": 3, "name": "hit_child", "ns": 3.4500926471697504, "err": null}
{"bin": "c3_2", "round": 3, "name": "hit/1", "ns": 1.8336767504986056, "err": null}
{"bin": "c3_2", "round": 3, "name": "hit/8", "ns": 1.8263969961272322, "err": null}
{"bin": "c3_2", "round": 3, "name": "hit/64", "ns": 2.2072926596157325, "err": null}
{"bin": "c3_2", "round": 3, "name": "hit/512", "ns": 5.3715054877102375, "err": null}
{"bin": "c3_2", "round": 3, "name": "miss/1", "ns": 1.3834917403319362, "err": null}
{"bin": "c3_2", "round": 3, "name": "miss/8", "ns": 2.45773225218536, "err": null}
{"bin": "c3_2", "round": 3, "name": "miss/64", "ns": 3.14223288025895, "err": null}
{"bin": "c3_2", "round": 3, "name": "miss/512", "ns": 2.867756374493893, "err": null}
{"bin": "c3_2", "round": 3, "name": "hash_only", "ns": 1.420309209472993, "err": null}
{"bin": "c3_2", "round": 3, "name": "hit_child", "ns": 3.0011527213206746, "err": null}
{"bin": "c2_5", "round": 3, "name": "hit/1", "ns": 3.0434231415404973, "err": null}
{"bin": "c2_5", "round": 3, "name": "hit/8", "ns": 3.272017120396516, "err": null}
{"bin": "c2_5", "round": 3, "name": "hit/64", "ns": 3.3582221701114543, "err": null}
{"bin": "c2_5", "round": 3, "name": "hit/512", "ns": 6.747611984610558, "err": null}
{"bin": "c2_5", "round": 3, "name": "miss/1", "ns": 1.3911026238466917, "err": null}
{"bin": "c2_5", "round": 3, "name": "miss/8", "ns": 2.959915152192247, "err": null}
{"bin": "c2_5", "round": 3, "name": "miss/64", "ns": 2.844287531750342, "err": null}
{"bin": "c2_5", "round": 3, "name": "miss/512", "ns": 3.4562414006579987, "err": null}
{"bin": "c2_5", "round": 3, "name": "hash_only", "ns": 3.581020399035767, "err": null}
{"bin": "c2_5", "round": 3, "name": "hit_child", "ns": 3.453467767793385, "err": null}
{"bin": "c3_5", "round": 3, "name": "hit/1", "ns": 2.990336864208685, "err": null}
{"bin": "c3_5", "round": 3, "name": "hit/8", "ns": 3.2004117775641068, "err": null}
{"bin": "c3_5", "round": 3, "name": "hit/64", "ns": 3.2661219501539787, "err": null}
{"bin": "c3_5", "round": 3, "name": "hit/512", "ns": 7.6053365133702755, "err": null}
{"bin": "c3_5", "round": 3, "name": "miss/1", "ns": 1.407902492691099, "err": null}
{"bin": "c3_5", "round": 3, "name": "miss/8", "ns": 3.0060219678333353, "err": null}
{"bin": "c3_5", "round": 3, "name": "miss/64", "ns": 2.8530892049032945, "err": null}
{"bin": "c3_5", "round": 3, "name": "miss/512", "ns": 3.4922282037682946, "err": null}
{"bin": "c3_5", "round": 3, "name": "hash_only", "ns": 3.579275276607032, "err": null}
{"bin": "c3_5", "round": 3, "name": "hit_child", "ns": 3.0061925283787376, "err": null}
{"bin": "c2_1", "round": 4, "name": "hit/1", "ns": 1.5901471218192693, "err": null}
{"bin": "c2_1", "round": 4, "name": "hit/8", "ns": 1.6931430520481285, "err": null}
{"bin": "c2_1", "round": 4, "name": "hit/64", "ns": 1.726847878331381, "err": null}
{"bin": "c2_1", "round": 4, "name": "hit/512", "ns": 2.3506654845309862, "err": null}
{"bin": "c2_1", "round": 4, "name": "miss/1", "ns": 1.2706928063356069, "err": null}
{"bin": "c2_1", "round": 4, "name": "miss/8", "ns": 3.660102303436586, "err": null}
{"bin": "c2_1", "round": 4, "name": "miss/64", "ns": 2.8978599751804994, "err": null}
{"bin": "c2_1", "round": 4, "name": "miss/512", "ns": 3.1701871466592704, "err": null}
{"bin": "c2_1", "round": 4, "name": "hash_only", "ns": 1.3550466609637861, "err": null}
{"bin": "c2_1", "round": 4, "name": "hit_child", "ns": 3.465253064954863, "err": null}
{"bin": "c3_1", "round": 4, "name": "hit/1", "ns": 1.5593080178671177, "err": null}
{"bin": "c3_1", "round": 4, "name": "hit/8", "ns": 1.6029276734902556, "err": null}
{"bin": "c3_1", "round": 4, "name": "hit/64", "ns": 1.678790971885177, "err": null}
{"bin": "c3_1", "round": 4, "name": "hit/512", "ns": 2.287257970355884, "err": null}
{"bin": "c3_1", "round": 4, "name": "miss/1", "ns": 1.2114237211670698, "err": null}
{"bin": "c3_1", "round": 4, "name": "miss/8", "ns": 2.674209061449526, "err": null}
{"bin": "c3_1", "round": 4, "name": "miss/64", "ns": 2.7810438013938996, "err": null}
{"bin": "c3_1", "round": 4, "name": "miss/512", "ns": 2.9247299827524817, "err": null}
{"bin": "c3_1", "round": 4, "name": "hash_only", "ns": 1.3548525936478668, "err": null}
{"bin": "c3_1", "round": 4, "name": "hit_child", "ns": 2.9978895864930744, "err": null}
{"bin": "c2_2", "round": 4, "name": "hit/1", "ns": 2.062188917197753, "err": null}
{"bin": "c2_2", "round": 4, "name": "hit/8", "ns": 2.063090796085695, "err": null}
{"bin": "c2_2", "round": 4, "name": "hit/64", "ns": 2.3417858411438, "err": null}
{"bin": "c2_2", "round": 4, "name": "hit/512", "ns": 3.1156637090353487, "err": null}
{"bin": "c2_2", "round": 4, "name": "miss/1", "ns": 1.457186635278614, "err": null}
{"bin": "c2_2", "round": 4, "name": "miss/8", "ns": 3.8771428874837612, "err": null}
{"bin": "c2_2", "round": 4, "name": "miss/64", "ns": 4.3937196221784705, "err": null}
{"bin": "c2_2", "round": 4, "name": "miss/512", "ns": 3.951683562835928, "err": null}
{"bin": "c2_2", "round": 4, "name": "hash_only", "ns": 1.557030823486658, "err": null}
{"bin": "c2_2", "round": 4, "name": "hit_child", "ns": 3.4557542848413596, "err": null}
{"bin": "c3_2", "round": 4, "name": "hit/1", "ns": 1.8867500423048011, "err": null}
{"bin": "c3_2", "round": 4, "name": "hit/8", "ns": 1.8266008777382352, "err": null}
{"bin": "c3_2", "round": 4, "name": "hit/64", "ns": 2.222143634506694, "err": null}
{"bin": "c3_2", "round": 4, "name": "hit/512", "ns": 3.2721731584530658, "err": null}
{"bin": "c3_2", "round": 4, "name": "miss/1", "ns": 1.3831406512189544, "err": null}
{"bin": "c3_2", "round": 4, "name": "miss/8", "ns": 2.4126117690740454, "err": null}
{"bin": "c3_2", "round": 4, "name": "miss/64", "ns": 3.711250246867271, "err": null}
{"bin": "c3_2", "round": 4, "name": "miss/512", "ns": 2.8542909211135923, "err": null}
{"bin": "c3_2", "round": 4, "name": "hash_only", "ns": 1.4200781680316927, "err": null}
{"bin": "c3_2", "round": 4, "name": "hit_child", "ns": 3.766457673225577, "err": null}
{"bin": "c2_5", "round": 4, "name": "hit/1", "ns": 3.042903387720114, "err": null}
{"bin": "c2_5", "round": 4, "name": "hit/8", "ns": 3.278792388814295, "err": null}
{"bin": "c2_5", "round": 4, "name": "hit/64", "ns": 4.318304886112182, "err": null}
{"bin": "c2_5", "round": 4, "name": "hit/512", "ns": 4.670164333485712, "err": null}
{"bin": "c2_5", "round": 4, "name": "miss/1", "ns": 1.3867434299949206, "err": null}
{"bin": "c2_5", "round": 4, "name": "miss/8", "ns": 3.0072459233485143, "err": null}
{"bin": "c2_5", "round": 4, "name": "miss/64", "ns": 3.2853567799092374, "err": null}
{"bin": "c2_5", "round": 4, "name": "miss/512", "ns": 3.4546891763061724, "err": null}
{"bin": "c2_5", "round": 4, "name": "hash_only", "ns": 3.581255258145181, "err": null}
{"bin": "c2_5", "round": 4, "name": "hit_child", "ns": 3.453924337141284, "err": null}
{"bin": "c3_5", "round": 4, "name": "hit/1", "ns": 3.868748659338532, "err": null}
{"bin": "c3_5", "round": 4, "name": "hit/8", "ns": 3.1989315581109716, "err": null}
{"bin": "c3_5", "round": 4, "name": "hit/64", "ns": 3.263023739802045, "err": null}
{"bin": "c3_5", "round": 4, "name": "hit/512", "ns": 4.537966388364244, "err": null}
{"bin": "c3_5", "round": 4, "name": "miss/1", "ns": 1.3832918082230483, "err": null}
{"bin": "c3_5", "round": 4, "name": "miss/8", "ns": 3.4342336667064033, "err": null}
{"bin": "c3_5", "round": 4, "name": "miss/64", "ns": 2.844477083924654, "err": null}
{"bin": "c3_5", "round": 4, "name": "miss/512", "ns": 3.482010859133493, "err": null}
{"bin": "c3_5", "round": 4, "name": "hash_only", "ns": 3.582517656905323, "err": null}
{"bin": "c3_5", "round": 4, "name": "hit_child", "ns": 3.0181866664340147, "err": null}
{"bin": "c2_1", "round": 5, "name": "hit/1", "ns": 1.7940503025059307, "err": null}
{"bin": "c2_1", "round": 5, "name": "hit/8", "ns": 1.817055448768939, "err": null}
{"bin": "c2_1", "round": 5, "name": "hit/64", "ns": 1.7089770991055384, "err": null}
{"bin": "c2_1", "round": 5, "name": "hit/512", "ns": 2.3573855173769322, "err": null}
{"bin": "c2_1", "round": 5, "name": "miss/1", "ns": 1.2704774959432368, "err": null}
{"bin": "c2_1", "round": 5, "name": "miss/8", "ns": 2.9216468839392693, "err": null}
{"bin": "c2_1", "round": 5, "name": "miss/64", "ns": 2.8163216492191765, "err": null}
{"bin": "c2_1", "round": 5, "name": "miss/512", "ns": 3.7485831873378768, "err": null}
{"bin": "c2_1", "round": 5, "name": "hash_only", "ns": 1.3522345830726268, "err": null}
{"bin": "c2_1", "round": 5, "name": "hit_child", "ns": 3.4638980656270153, "err": null}
{"bin": "c3_1", "round": 5, "name": "hit/1", "ns": 1.5962538210026322, "err": null}
{"bin": "c3_1", "round": 5, "name": "hit/8", "ns": 1.6246548321112775, "err": null}
{"bin": "c3_1", "round": 5, "name": "hit/64", "ns": 1.6308024552179878, "err": null}
{"bin": "c3_1", "round": 5, "name": "hit/512", "ns": 2.3009591403573917, "err": null}
{"bin": "c3_1", "round": 5, "name": "miss/1", "ns": 1.6748127296302755, "err": null}
{"bin": "c3_1", "round": 5, "name": "miss/8", "ns": 2.571716961571806, "err": null}
{"bin": "c3_1", "round": 5, "name": "miss/64", "ns": 2.7881312052437663, "err": null}
{"bin": "c3_1", "round": 5, "name": "miss/512", "ns": 2.908041559501823, "err": null}
{"bin": "c3_1", "round": 5, "name": "hash_only", "ns": 1.354447173592208, "err": null}
{"bin": "c3_1", "round": 5, "name": "hit_child", "ns": 3.0173244209281704, "err": null}
{"bin": "c2_2", "round": 5, "name": "hit/1", "ns": 2.0672924300184627, "err": null}
{"bin": "c2_2", "round": 5, "name": "hit/8", "ns": 2.0605232199154115, "err": null}
{"bin": "c2_2", "round": 5, "name": "hit/64", "ns": 3.313229649438707, "err": null}
{"bin": "c2_2", "round": 5, "name": "hit/512", "ns": 3.1521620341180876, "err": null}
{"bin": "c2_2", "round": 5, "name": "miss/1", "ns": 1.391290956717219, "err": null}
{"bin": "c2_2", "round": 5, "name": "miss/8", "ns": 3.4818069718350837, "err": null}
{"bin": "c2_2", "round": 5, "name": "miss/64", "ns": 4.415757295830984, "err": null}
{"bin": "c2_2", "round": 5, "name": "miss/512", "ns": 3.952490066985489, "err": null}
{"bin": "c2_2", "round": 5, "name": "hash_only", "ns": 1.4192387430213405, "err": null}
{"bin": "c2_2", "round": 5, "name": "hit_child", "ns": 3.4550590553976823, "err": null}
{"bin": "c3_2", "round": 5, "name": "hit/1", "ns": 1.862967192826454, "err": null}
{"bin": "c3_2", "round": 5, "name": "hit/8", "ns": 2.713931258933823, "err": null}
{"bin": "c3_2", "round": 5, "name": "hit/64", "ns": 2.209994395923465, "err": null}
{"bin": "c3_2", "round": 5, "name": "hit/512", "ns": 3.0234697450121613, "err": null}
{"bin": "c3_2", "round": 5, "name": "miss/1", "ns": 1.383442613675656, "err": null}
{"bin": "c3_2", "round": 5, "name": "miss/8", "ns": 2.4401513672211497, "err": null}
{"bin": "c3_2", "round": 5, "name": "miss/64", "ns": 3.074139882464743, "err": null}
{"bin": "c3_2", "round": 5, "name": "miss/512", "ns": 2.8682198249170825, "err": null}
{"bin": "c3_2", "round": 5, "name": "hash_only", "ns": 1.4204772375572225, "err": null}
{"bin": "c3_2", "round": 5, "name": "hit_child", "ns": 2.99858817390058, "err": null}
{"bin": "c2_5", "round": 5, "name": "hit/1", "ns": 3.0438663479919126, "err": null}
{"bin": "c2_5", "round": 5, "name": "hit/8", "ns": 5.700509343296289, "err": null}
{"bin": "c2_5", "round": 5, "name": "hit/64", "ns": 3.3492132796228167, "err": null}
{"bin": "c2_5", "round": 5, "name": "hit/512", "ns": 4.681340979722983, "err": null}
{"bin": "c2_5", "round": 5, "name": "miss/1", "ns": 1.3868585658443815, "err": null}
{"bin": "c2_5", "round": 5, "name": "miss/8", "ns": 2.9361591544977803, "err": null}
{"bin": "c2_5", "round": 5, "name": "miss/64", "ns": 2.8407778056627753, "err": null}
{"bin": "c2_5", "round": 5, "name": "miss/512", "ns": 3.4583920599953846, "err": null}
{"bin": "c2_5", "round": 5, "name": "hash_only", "ns": 3.5892303427687575, "err": null}
{"bin": "c2_5", "round": 5, "name": "hit_child", "ns": 3.457740134485457, "err": null}
{"bin": "c3_5", "round": 5, "name": "hit/1", "ns": 2.9941370041857405, "err": null}
{"bin": "c3_5", "round": 5, "name": "hit/8", "ns": 3.201148522967728, "err": null}
{"bin": "c3_5", "round": 5, "name": "hit/64", "ns": 3.264814609289808, "err": null}
{"bin": "c3_5", "round": 5, "name": "hit/512", "ns": 4.552555581783596, "err": null}
{"bin": "c3_5", "round": 5, "name": "miss/1", "ns": 1.3839990827405626, "err": null}
{"bin": "c3_5", "round": 5, "name": "miss/8", "ns": 2.9747689007033316, "err": null}
{"bin": "c3_5", "round": 5, "name": "miss/64", "ns": 2.869969651255126, "err": null}
{"bin": "c3_5", "round": 5, "name": "miss/512", "ns": 3.476679642330773, "err": null}
{"bin": "c3_5", "round": 5, "name": "hash_only", "ns": 3.5730596837266773, "err": null}
{"bin": "c3_5", "round": 5, "name": "hit_child", "ns": 3.0061813563325313, "err": null}
{"bin": "c2_1", "round": 6, "name": "hit/1", "ns": 1.591235983210801, "err": null}
{"bin": "c2_1", "round": 6, "name": "hit/8", "ns": 1.664834494210765, "err": null}
{"bin": "c2_1", "round": 6, "name": "hit/64", "ns": 1.704486098774764, "err": null}
{"bin": "c2_1", "round": 6, "name": "hit/512", "ns": 2.3478173375883573, "err": null}
{"bin": "c2_1", "round": 6, "name": "miss/1", "ns": 1.2718348626132205, "err": null}
{"bin": "c2_1", "round": 6, "name": "miss/8", "ns": 2.843029730072776, "err": null}
{"bin": "c2_1", "round": 6, "name": "miss/64", "ns": 4.870699564870742, "err": null}
{"bin": "c2_1", "round": 6, "name": "miss/512", "ns": 3.172902861263156, "err": null}
{"bin": "c2_1", "round": 6, "name": "hash_only", "ns": 1.3573734651386216, "err": null}
{"bin": "c2_1", "round": 6, "name": "hit_child", "ns": 3.4627512803753024, "err": null}
{"bin": "c3_1", "round": 6, "name": "hit/1", "ns": 1.5631372478043366, "err": null}
{"bin": "c3_1", "round": 6, "name": "hit/8", "ns": 1.6704230193543212, "err": null}
{"bin": "c3_1", "round": 6, "name": "hit/64", "ns": 1.671814071512139, "err": null}
{"bin": "c3_1", "round": 6, "name": "hit/512", "ns": 2.2981480916072377, "err": null}
{"bin": "c3_1", "round": 6, "name": "miss/1", "ns": 1.2609313614257092, "err": null}
{"bin": "c3_1", "round": 6, "name": "miss/8", "ns": 2.672096155324238, "err": null}
{"bin": "c3_1", "round": 6, "name": "miss/64", "ns": 2.7446835120614455, "err": null}
{"bin": "c3_1", "round": 6, "name": "miss/512", "ns": 2.9491899777435515, "err": null}
{"bin": "c3_1", "round": 6, "name": "hash_only", "ns": 1.3554492343396318, "err": null}
{"bin": "c3_1", "round": 6, "name": "hit_child", "ns": 2.9973881675302727, "err": null}
{"bin": "c2_2", "round": 6, "name": "hit/1", "ns": 2.0617064086168435, "err": null}
{"bin": "c2_2", "round": 6, "name": "hit/8", "ns": 2.060720747783674, "err": null}
{"bin": "c2_2", "round": 6, "name": "hit/64", "ns": 2.3350521531262762, "err": null}
{"bin": "c2_2", "round": 6, "name": "hit/512", "ns": 3.1148507099844442, "err": null}
{"bin": "c2_2", "round": 6, "name": "miss/1", "ns": 1.3952460477300805, "err": null}
{"bin": "c2_2", "round": 6, "name": "miss/8", "ns": 3.519104422997938, "err": null}
{"bin": "c2_2", "round": 6, "name": "miss/64", "ns": 4.311741290114045, "err": null}
{"bin": "c2_2", "round": 6, "name": "miss/512", "ns": 3.9546180963057336, "err": null}
{"bin": "c2_2", "round": 6, "name": "hash_only", "ns": 1.4206307196871335, "err": null}
{"bin": "c2_2", "round": 6, "name": "hit_child", "ns": 3.4558023174258965, "err": null}
{"bin": "c3_2", "round": 6, "name": "hit/1", "ns": 1.938783725758087, "err": null}
{"bin": "c3_2", "round": 6, "name": "hit/8", "ns": 1.8237762736781604, "err": null}
{"bin": "c3_2", "round": 6, "name": "hit/64", "ns": 2.4842591599153074, "err": null}
{"bin": "c3_2", "round": 6, "name": "hit/512", "ns": 3.014011048980081, "err": null}
{"bin": "c3_2", "round": 6, "name": "miss/1", "ns": 1.383452166071512, "err": null}
{"bin": "c3_2", "round": 6, "name": "miss/8", "ns": 2.4268680455232174, "err": null}
{"bin": "c3_2", "round": 6, "name": "miss/64", "ns": 3.1186175146023443, "err": null}
{"bin": "c3_2", "round": 6, "name": "miss/512", "ns": 3.2791529545263964, "err": null}
{"bin": "c3_2", "round": 6, "name": "hash_only", "ns": 1.4227780066721458, "err": null}
{"bin": "c3_2", "round": 6, "name": "hit_child", "ns": 3.2022012132198427, "err": null}
{"bin": "c2_5", "round": 6, "name": "hit/1", "ns": 3.045054588377309, "err": null}
{"bin": "c2_5", "round": 6, "name": "hit/8", "ns": 3.2707611357519157, "err": null}
{"bin": "c2_5", "round": 6, "name": "hit/64", "ns": 3.347342907458426, "err": null}
{"bin": "c2_5", "round": 6, "name": "hit/512", "ns": 5.61192306817949, "err": null}
{"bin": "c2_5", "round": 6, "name": "miss/1", "ns": 1.4078096861797746, "err": null}
{"bin": "c2_5", "round": 6, "name": "miss/8", "ns": 2.9508539179831073, "err": null}
{"bin": "c2_5", "round": 6, "name": "miss/64", "ns": 3.4643666466734255, "err": null}
{"bin": "c2_5", "round": 6, "name": "miss/512", "ns": 3.458048979238097, "err": null}
{"bin": "c2_5", "round": 6, "name": "hash_only", "ns": 3.5836992824480176, "err": null}
{"bin": "c2_5", "round": 6, "name": "hit_child", "ns": 3.4557286124051565, "err": null}
{"bin": "c3_5", "round": 6, "name": "hit/1", "ns": 2.9943652077405347, "err": null}
{"bin": "c3_5", "round": 6, "name": "hit/8", "ns": 3.9829818297146358, "err": null}
{"bin": "c3_5", "round": 6, "name": "hit/64", "ns": 3.2658149151067555, "err": null}
{"bin": "c3_5", "round": 6, "name": "hit/512", "ns": 4.529338896623358, "err": null}
{"bin": "c3_5", "round": 6, "name": "miss/1", "ns": 1.487357483109937, "err": null}
{"bin": "c3_5", "round": 6, "name": "miss/8", "ns": 3.0358693775919714, "err": null}
{"bin": "c3_5", "round": 6, "name": "miss/64", "ns": 2.928832040218758, "err": null}
{"bin": "c3_5", "round": 6, "name": "miss/512", "ns": 3.495202442395386, "err": null}
{"bin": "c3_5", "round": 6, "name": "hash_only", "ns": 3.580714745309353, "err": null}
{"bin": "c3_5", "round": 6, "name": "hit_child", "ns": 3.009184984992815, "err": null}
{"bin": "c2_1", "round": 7, "name": "hit/1", "ns": 1.589022638822814, "err": null}
{"bin": "c2_1", "round": 7, "name": "hit/8", "ns": 1.6656694530623275, "err": null}
{"bin": "c2_1", "round": 7, "name": "hit/64", "ns": 1.7101100902391677, "err": null}
{"bin": "c2_1", "round": 7, "name": "hit/512", "ns": 2.523680494094423, "err": null}
{"bin": "c2_1", "round": 7, "name": "miss/1", "ns": 1.2710163894824162, "err": null}
{"bin": "c2_1", "round": 7, "name": "miss/8", "ns": 2.8998269538012362, "err": null}
{"bin": "c2_1", "round": 7, "name": "miss/64", "ns": 2.812144643416261, "err": null}
{"bin": "c2_1", "round": 7, "name": "miss/512", "ns": 3.3528056462240303, "err": null}
{"bin": "c2_1", "round": 7, "name": "hash_only", "ns": 1.351893516340979, "err": null}
{"bin": "c2_1", "round": 7, "name": "hit_child", "ns": 3.7823921964898464, "err": null}
{"bin": "c3_1", "round": 7, "name": "hit/1", "ns": 1.591951951937462, "err": null}
{"bin": "c3_1", "round": 7, "name": "hit/8", "ns": 1.6582959837357096, "err": null}
{"bin": "c3_1", "round": 7, "name": "hit/64", "ns": 1.702326551826921, "err": null}
{"bin": "c3_1", "round": 7, "name": "hit/512", "ns": 2.5571479517500606, "err": null}
{"bin": "c3_1", "round": 7, "name": "miss/1", "ns": 1.227495821789147, "err": null}
{"bin": "c3_1", "round": 7, "name": "miss/8", "ns": 2.639181302337188, "err": null}
{"bin": "c3_1", "round": 7, "name": "miss/64", "ns": 2.8079557781467277, "err": null}
{"bin": "c3_1", "round": 7, "name": "miss/512", "ns": 2.925604740978331, "err": null}
{"bin": "c3_1", "round": 7, "name": "hash_only", "ns": 1.5015438106663657, "err": null}
{"bin": "c3_1", "round": 7, "name": "hit_child", "ns": 3.000233288203264, "err": null}
{"bin": "c2_2", "round": 7, "name": "hit/1", "ns": 2.0645143468322904, "err": null}
{"bin": "c2_2", "round": 7, "name": "hit/8", "ns": 2.4545151884688443, "err": null}
{"bin": "c2_2", "round": 7, "name": "hit/64", "ns": 2.3341374865797717, "err": null}
{"bin": "c2_2", "round": 7, "name": "hit/512", "ns": 3.118232544898558, "err": null}
{"bin": "c2_2", "round": 7, "name": "miss/1", "ns": 1.3917416084578162, "err": null}
{"bin": "c2_2", "round": 7, "name": "miss/8", "ns": 3.4714502473083124, "err": null}
{"bin": "c2_2", "round": 7, "name": "miss/64", "ns": 4.408421511422425, "err": null}
{"bin": "c2_2", "round": 7, "name": "miss/512", "ns": 4.24152824619334, "err": null}
{"bin": "c2_2", "round": 7, "name": "hash_only", "ns": 1.4193072227769177, "err": null}
{"bin": "c2_2", "round": 7, "name": "hit_child", "ns": 3.462502255069121, "err": null}
{"bin": "c3_2", "round": 7, "name": "hit/1", "ns": 1.8586058708719368, "err": null}
{"bin": "c3_2", "round": 7, "name": "hit/8", "ns": 1.826543595906648, "err": null}
{"bin": "c3_2", "round": 7, "name": "hit/64", "ns": 2.244448152798731, "err": null}
{"bin": "c3_2", "round": 7, "name": "hit/512", "ns": 3.0191791838260955, "err": null}
{"bin": "c3_2", "round": 7, "name": "miss/1", "ns": 1.3831926946076127, "err": null}
{"bin": "c3_2", "round": 7, "name": "miss/8", "ns": 2.4346529804761583, "err": null}
{"bin": "c3_2", "round": 7, "name": "miss/64", "ns": 4.042513096823092, "err": null}
{"bin": "c3_2", "round": 7, "name": "miss/512", "ns": 2.875788416299464, "err": null}
{"bin": "c3_2", "round": 7, "name": "hash_only", "ns": 1.415391006249996, "err": null}
{"bin": "c3_2", "round": 7, "name": "hit_child", "ns": 3.006333157496081, "err": null}
{"bin": "c2_5", "round": 7, "name": "hit/1", "ns": 3.748446095933221, "err": null}
{"bin": "c2_5", "round": 7, "name": "hit/8", "ns": 3.286772272076121, "err": null}
{"bin": "c2_5", "round": 7, "name": "hit/64", "ns": 3.34905775776529, "err": null}
{"bin": "c2_5", "round": 7, "name": "hit/512", "ns": 4.670335723145523, "err": null}
{"bin": "c2_5", "round": 7, "name": "miss/1", "ns": 1.3840434482057062, "err": null}
{"bin": "c2_5", "round": 7, "name": "miss/8", "ns": 3.798444754290585, "err": null}
{"bin": "c2_5", "round": 7, "name": "miss/64", "ns": 2.8677633151800417, "err": null}
{"bin": "c2_5", "round": 7, "name": "miss/512", "ns": 3.453975915105842, "err": null}
{"bin": "c2_5", "round": 7, "name": "hash_only", "ns": 3.5826251833560296, "err": null}
{"bin": "c2_5", "round": 7, "name": "hit_child", "ns": 3.4548867060784842, "err": null}
{"bin": "c3_5", "round": 7, "name": "hit/1", "ns": 2.991962752410841, "err": null}
{"bin": "c3_5", "round": 7, "name": "hit/8", "ns": 3.198741470842766, "err": null}
{"bin": "c3_5", "round": 7, "name": "hit/64", "ns": 3.2650950096785833, "err": null}
{"bin": "c3_5", "round": 7, "name": "hit/512", "ns": 4.538213170354857, "err": null}
{"bin": "c3_5", "round": 7, "name": "miss/1", "ns": 1.3864240011030367, "err": null}
{"bin": "c3_5", "round": 7, "name": "miss/8", "ns": 2.9697042673602643, "err": null}
{"bin": "c3_5", "round": 7, "name": "miss/64", "ns": 2.8555695809637363, "err": null}
{"bin": "c3_5", "round": 7, "name": "miss/512", "ns": 3.4780463685599607, "err": null}
{"bin": "c3_5", "round": 7, "name": "hash_only", "ns": 3.5800954894426025, "err": null}
{"bin": "c3_5", "round": 7, "name": "hit_child", "ns": 3.004570427359209, "err": null}
{"bin": "c2_1", "round": 8, "name": "hit/1", "ns": 1.5964126272683268, "err": null}
{"bin": "c2_1", "round": 8, "name": "hit/8", "ns": 1.935453004437238, "err": null}
{"bin": "c2_1", "round": 8, "name": "hit/64", "ns": 1.7115748595885676, "err": null}
{"bin": "c2_1", "round": 8, "name": "hit/512", "ns": 2.6472630635505827, "err": null}
{"bin": "c2_1", "round": 8, "name": "miss/1", "ns": 1.2711874251768776, "err": null}
{"bin": "c2_1", "round": 8, "name": "miss/8", "ns": 2.8938343089177336, "err": null}
{"bin": "c2_1", "round": 8, "name": "miss/64", "ns": 3.2594585892518344, "err": null}
{"bin": "c2_1", "round": 8, "name": "miss/512", "ns": 3.1975295103714862, "err": null}
{"bin": "c2_1", "round": 8, "name": "hash_only", "ns": 1.3589857974062733, "err": null}
{"bin": "c2_1", "round": 8, "name": "hit_child", "ns": 4.034040996704345, "err": null}
{"bin": "c3_1", "round": 8, "name": "hit/1", "ns": 1.5582864396323879, "err": null}
{"bin": "c3_1", "round": 8, "name": "hit/8", "ns": 1.6214724090683186, "err": null}
{"bin": "c3_1", "round": 8, "name": "hit/64", "ns": 1.9091889909362347, "err": null}
{"bin": "c3_1", "round": 8, "name": "hit/512", "ns": 2.3005310932122396, "err": null}
{"bin": "c3_1", "round": 8, "name": "miss/1", "ns": 1.2696231751683364, "err": null}
{"bin": "c3_1", "round": 8, "name": "miss/8", "ns": 2.6379530059069336, "err": null}
{"bin": "c3_1", "round": 8, "name": "miss/64", "ns": 3.2841083280812593, "err": null}
{"bin": "c3_1", "round": 8, "name": "miss/512", "ns": 2.9806735663327317, "err": null}
{"bin": "c3_1", "round": 8, "name": "hash_only", "ns": 1.3546063742496917, "err": null}
{"bin": "c3_1", "round": 8, "name": "hit_child", "ns": 3.003697162019037, "err": null}
{"bin": "c2_2", "round": 8, "name": "hit/1", "ns": 2.438214778220968, "err": null}
{"bin": "c2_2", "round": 8, "name": "hit/8", "ns": 2.061830254524152, "err": null}
{"bin": "c2_2", "round": 8, "name": "hit/64", "ns": 2.334900576556301, "err": null}
{"bin": "c2_2", "round": 8, "name": "hit/512", "ns": 3.1313309379304304, "err": null}
{"bin": "c2_2", "round": 8, "name": "miss/1", "ns": 1.7551441317671674, "err": null}
{"bin": "c2_2", "round": 8, "name": "miss/8", "ns": 3.531444652949315, "err": null}
{"bin": "c2_2", "round": 8, "name": "miss/64", "ns": 4.40640002432048, "err": null}
{"bin": "c2_2", "round": 8, "name": "miss/512", "ns": 3.957746244565398, "err": null}
{"bin": "c2_2", "round": 8, "name": "hash_only", "ns": 1.3962146138853753, "err": null}
{"bin": "c2_2", "round": 8, "name": "hit_child", "ns": 5.058104917407036, "err": null}
{"bin": "c3_2", "round": 8, "name": "hit/1", "ns": 1.8392139621904915, "err": null}
{"bin": "c3_2", "round": 8, "name": "hit/8", "ns": 1.9357231468990785, "err": null}
{"bin": "c3_2", "round": 8, "name": "hit/64", "ns": 2.1926165147883796, "err": null}
{"bin": "c3_2", "round": 8, "name": "hit/512", "ns": 3.021722371624252, "err": null}
{"bin": "c3_2", "round": 8, "name": "miss/1", "ns": 1.383215072976791, "err": null}
{"bin": "c3_2", "round": 8, "name": "miss/8", "ns": 3.0915445528664924, "err": null}
{"bin": "c3_2", "round": 8, "name": "miss/64", "ns": 3.083183313354736, "err": null}
{"bin": "c3_2", "round": 8, "name": "miss/512", "ns": 2.8581752399307248, "err": null}
{"bin": "c3_2", "round": 8, "name": "hash_only", "ns": 1.4189413408767966, "err": null}
{"bin": "c3_2", "round": 8, "name": "hit_child", "ns": 3.000959130930688, "err": null}
{"bin": "c2_5", "round": 8, "name": "hit/1", "ns": 3.04155389841702, "err": null}
{"bin": "c2_5", "round": 8, "name": "hit/8", "ns": 3.329048306335202, "err": null}
{"bin": "c2_5", "round": 8, "name": "hit/64", "ns": 3.3506061467141444, "err": null}
{"bin": "c2_5", "round": 8, "name": "hit/512", "ns": 4.684922788811085, "err": null}
{"bin": "c2_5", "round": 8, "name": "miss/1", "ns": 1.3844894563147725, "err": null}
{"bin": "c2_5", "round": 8, "name": "miss/8", "ns": 3.0103762649821846, "err": null}
{"bin": "c2_5", "round": 8, "name": "miss/64", "ns": 2.8447183830952687, "err": null}
{"bin": "c2_5", "round": 8, "name": "miss/512", "ns": 4.277295918551779, "err": null}
{"bin": "c2_5", "round": 8, "name": "hash_only", "ns": 3.581086558524817, "err": null}
{"bin": "c2_5", "round": 8, "name": "hit_child", "ns": 3.455779421582688, "err": null}
{"bin": "c3_5", "round": 8, "name": "hit/1", "ns": 2.9945147254258138, "err": null}
{"bin": "c3_5", "round": 8, "name": "hit/8", "ns": 3.1995841100857536, "err": null}
{"bin": "c3_5", "round": 8, "name": "hit/64", "ns": 3.2739312785342936, "err": null}
{"bin": "c3_5", "round": 8, "name": "hit/512", "ns": 4.530613352940369, "err": null}
{"bin": "c3_5", "round": 8, "name": "miss/1", "ns": 1.3835244777444373, "err": null}
{"bin": "c3_5", "round": 8, "name": "miss/8", "ns": 4.1081619373339295, "err": null}
{"bin": "c3_5", "round": 8, "name": "miss/64", "ns": 2.8725759230061603, "err": null}
{"bin": "c3_5", "round": 8, "name": "miss/512", "ns": 3.511018853456718, "err": null}
{"bin": "c3_5", "round": 8, "name": "hash_only", "ns": 3.5803181115864344, "err": null}
{"bin": "c3_5", "round": 8, "name": "hit_child", "ns": 3.014151514659691, "err": null}
{"bin": "c2_1", "round": 9, "name": "hit/1", "ns": 1.5987566913893392, "err": null}
{"bin": "c2_1", "round": 9, "name": "hit/8", "ns": 1.66568663638415, "err": null}
{"bin": "c2_1", "round": 9, "name": "hit/64", "ns": 1.7237523698384654, "err": null}
{"bin": "c2_1", "round": 9, "name": "hit/512", "ns": 2.3504272136607702, "err": null}
{"bin": "c2_1", "round": 9, "name": "miss/1", "ns": 1.2749705287114654, "err": null}
{"bin": "c2_1", "round": 9, "name": "miss/8", "ns": 3.2919002425198114, "err": null}
{"bin": "c2_1", "round": 9, "name": "miss/64", "ns": 2.95107929620577, "err": null}
{"bin": "c2_1", "round": 9, "name": "miss/512", "ns": 3.160228700355625, "err": null}
{"bin": "c2_1", "round": 9, "name": "hash_only", "ns": 1.3548775467616758, "err": null}
{"bin": "c2_1", "round": 9, "name": "hit_child", "ns": 3.4618894131680484, "err": null}
{"bin": "c3_1", "round": 9, "name": "hit/1", "ns": 1.5581486298633946, "err": null}
{"bin": "c3_1", "round": 9, "name": "hit/8", "ns": 1.6204837190140458, "err": null}
{"bin": "c3_1", "round": 9, "name": "hit/64", "ns": 1.6179192329560634, "err": null}
{"bin": "c3_1", "round": 9, "name": "hit/512", "ns": 2.2982239871296803, "err": null}
{"bin": "c3_1", "round": 9, "name": "miss/1", "ns": 1.2270927541909578, "err": null}
{"bin": "c3_1", "round": 9, "name": "miss/8", "ns": 2.620407718915597, "err": null}
{"bin": "c3_1", "round": 9, "name": "miss/64", "ns": 2.759032403766848, "err": null}
{"bin": "c3_1", "round": 9, "name": "miss/512", "ns": 2.92892431857746, "err": null}
{"bin": "c3_1", "round": 9, "name": "hash_only", "ns": 1.3517356739628368, "err": null}
{"bin": "c3_1", "round": 9, "name": "hit_child", "ns": 2.999801422800451, "err": null}
{"bin": "c2_2", "round": 9, "name": "hit/1", "ns": 2.062853966706634, "err": null}
{"bin": "c2_2", "round": 9, "name": "hit/8", "ns": 2.058920603979075, "err": null}
{"bin": "c2_2", "round": 9, "name": "hit/64", "ns": 2.3402681302972295, "err": null}
{"bin": "c2_2", "round": 9, "name": "hit/512", "ns": 3.1180533922274907, "err": null}
{"bin": "c2_2", "round": 9, "name": "miss/1", "ns": 1.4034709881979788, "err": null}
{"bin": "c2_2", "round": 9, "name": "miss/8", "ns": 3.426897432294472, "err": null}
{"bin": "c2_2", "round": 9, "name": "miss/64", "ns": 4.393921710128016, "err": null}
{"bin": "c2_2", "round": 9, "name": "miss/512", "ns": 3.9551046285416667, "err": null}
{"bin": "c2_2", "round": 9, "name": "hash_only", "ns": 1.3950718423668274, "err": null}
{"bin": "c2_2", "round": 9, "name": "hit_child", "ns": 3.4562275122816453, "err": null}
{"bin": "c3_2", "round": 9, "name": "hit/1", "ns": 1.8204644451588157, "err": null}
{"bin": "c3_2", "round": 9, "name": "hit/8", "ns": 1.8294021537994565, "err": null}
{"bin": "c3_2", "round": 9, "name": "hit/64", "ns": 2.2052874995664795, "err": null}
{"bin": "c3_2", "round": 9, "name": "hit/512", "ns": 3.016185350269384, "err": null}
{"bin": "c3_2", "round": 9, "name": "miss/1", "ns": 1.383133419828146, "err": null}
{"bin": "c3_2", "round": 9, "name": "miss/8", "ns": 2.38248397571951, "err": null}
{"bin": "c3_2", "round": 9, "name": "miss/64", "ns": 3.137140494117192, "err": null}
{"bin": "c3_2", "round": 9, "name": "miss/512", "ns": 2.882086225419529, "err": null}
{"bin": "c3_2", "round": 9, "name": "hash_only", "ns": 1.4167284769853397, "err": null}
{"bin": "c3_2", "round": 9, "name": "hit_child", "ns": 3.000241858990129, "err": null}
{"bin": "c2_5", "round": 9, "name": "hit/1", "ns": 3.0414557240455724, "err": null}
{"bin": "c2_5", "round": 9, "name": "hit/8", "ns": 3.2742238867353195, "err": null}
{"bin": "c2_5", "round": 9, "name": "hit/64", "ns": 3.358384915527401, "err": null}
{"bin": "c2_5", "round": 9, "name": "hit/512", "ns": 4.6691316519061274, "err": null}
{"bin": "c2_5", "round": 9, "name": "miss/1", "ns": 1.389355238823699, "err": null}
{"bin": "c2_5", "round": 9, "name": "miss/8", "ns": 2.9643826123017853, "err": null}
{"bin": "c2_5", "round": 9, "name": "miss/64", "ns": 2.853852376645218, "err": null}
{"bin": "c2_5", "round": 9, "name": "miss/512", "ns": 3.4545496663474777, "err": null}
{"bin": "c2_5", "round": 9, "name": "hash_only", "ns": 3.5790425262165897, "err": null}
{"bin": "c2_5", "round": 9, "name": "hit_child", "ns": 3.4531604651832883, "err": null}
{"bin": "c3_5", "round": 9, "name": "hit/1", "ns": 3.0064381941674894, "err": null}
{"bin": "c3_5", "round": 9, "name": "hit/8", "ns": 3.204817757562906, "err": null}
{"bin": "c3_5", "round": 9, "name": "hit/64", "ns": 3.264362488052166, "err": null}
{"bin": "c3_5", "round": 9, "name": "hit/512", "ns": 4.530298574184214, "err": null}
{"bin": "c3_5", "round": 9, "name": "miss/1", "ns": 1.3872894171607224, "err": null}
{"bin": "c3_5", "round": 9, "name": "miss/8", "ns": 3.005721230347676, "err": null}
{"bin": "c3_5", "round": 9, "name": "miss/64", "ns": 2.8326841164610155, "err": null}
{"bin": "c3_5", "round": 9, "name": "miss/512", "ns": 3.461072825333907, "err": null}
{"bin": "c3_5", "round": 9, "name": "hash_only", "ns": 3.579082509738044, "err": null}
{"bin": "c3_5", "round": 9, "name": "hit_child", "ns": 3.0060484366600053, "err": null}
```

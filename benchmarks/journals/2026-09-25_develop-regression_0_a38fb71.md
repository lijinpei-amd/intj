# Paired regression diagnostic: `develop` versus `torch_abi`

Measured September 25, 2026. Baseline: clean local `develop` at `7ed7882` (only benchmark journal changes since executable source `ef698d2`). Candidate: clean `torch_abi` at `a38fb71` (executable source `03a8f52`). Both used the same Python 3.12.3, Torch 2.14.0+rocm7.2, Triton 3.8.0, MI308X `gfx942:sramecc+:xnack-`, and CPU 0. One process at a time. Five separate process pairs per diagnostic, alternating order baseline/candidate then candidate/baseline. GPU workloads were not run concurrently.

| Case | develop process medians | torch_abi process medians | median change |
| --- | --- | --- | ---: |
| GPU auto map (ns) | 3104.30, 3140.30, 3133.00, 3127.80, 3119.90 | 3185.30, 3551.40, 3812.90, 3126.40, 3157.30 | +1.8% |
| GPU bound pointer control (ns) | 2966.10, 3579.30, 3000.20, 2990.00, 2896.80 | 3191.00, 2985.90, 3707.00, 2970.50, 2948.80 | -0.1% |
| GPU fixed device no-map control (ns) | 2932.80, 2974.50, 2906.60, 2944.30, 3026.10 | 3179.80, 2950.90, 3782.60, 2881.00, 2890.50 | +0.2% |
| Host auto map (ns) | 43.60, 43.50, 43.50, 43.40, 45.20 | 44.00, 44.10, 44.00, 45.20, 45.30 | +1.4% |
| Host fixed device no-map (ns) | 45.10, 44.70, 44.70, 46.40, 46.50 | 43.50, 43.40, 43.30, 43.40, 43.80 | -3.8% |
| Static tensor spec key (ns) | 117.00, 114.81, 114.85, 117.11, 116.99 | 116.75, 116.79, 119.52, 120.18, 115.98 | -0.2% |
| Shim tensor spec key (ns) | 119.21, 117.45, 115.48, 116.78, 117.28 | 116.63, 116.39, 120.70, 120.94, 117.71 | +0.4% |
| Zero-grid integer (ns) | 56.78, 53.78, 53.59, 51.85, 59.82 | 52.26, 51.94, 53.98, 53.36, 55.80 | -0.8% |
| Zero-grid tuple (ns) | 57.44, 55.26, 54.74, 52.95, 61.29 | 53.22, 53.37, 55.16, 54.61, 57.16 | -1.2% |
| Host entry with launch grid (ns) | 73.01, 69.79, 71.82, 67.58, 79.31 | 68.05, 70.86, 69.74, 68.99, 72.42 | -2.9% |

## Interpretation

The default GPU launcher has a +1.8% process-median difference, but the candidate also has anomalously slow runs across unrelated GPU rows (including bound-pointer and no-map controls). In the later pairs, default GPU values are 3,126/3,157 ns versus 3,128/3,120 ns for develop. The prior 1,000-call FFI sweep had a much larger apparent slowdown shared by FFI GPU launches; the [remerge comparison](2026-09-25_develop-remerge_0_03a8f52.md) records its stable final-code and shorter-batch repeats. In this run, static-mode tensor key generation is 116.99 versus 116.79 ns; the GIL-build cache lock macros expand to no-ops, and the no-GPU zero-grid entry path is slightly faster. The host auto-map case is 0.6 ns slower, while the host no-map control is 1.7 ns faster. These measurements do not identify a reproducible tensor decoding or call-entry regression; the remaining 1-2% variation is within the changing controls and code generation/timing noise. No code change was made based on it.

## Setup and exact commands

- CPU 0 pinned via `taskset -c 0`; env `TRITON_HOME`, `CXX`, `CC`, `INTJ_BENCHMARK_ROOT` unset. Torch, Triton, and TVM FFI versions and full MI308X hardware details are in the full develop benchmark journal (`2026-09-25_develop-full_0_ef698d2.md`); both worktrees used `/tmp/gb2/bin/python`. The earlier full-run record lives in the local develop worktree; raw logs below identify both roots.
- In the `develop` and `torch_abi` roots respectively, run `PYTHONPATH=$PWD taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --iters 1000 --batches 9` for GPU and the same with `--no-gpu --iters 100000 --batches 9` for host. Each process warmed cases before timing; host timer stops before synchronization.
- For tensor-key and entry diagnostics, run `PYTHONPATH=$PWD:$PWD/benchmarks taskset -c 0 /tmp/gb2/bin/python /tmp/intj-regression-spec-key-20260925.py` and `/tmp/intj-regression-entry-20260925.py`. Self-contained source of both drivers is below.

### Tensor-key driver

```python
import statistics
import torch
import triton
import os,sys
sys.path.insert(0, os.path.join(os.environ['PYTHONPATH'].split(':')[0], 'benchmarks'))
import bench_launch
import intj

n = 4096
args = (torch.randn(n, device='cuda'), torch.randn(n, device='cuda'), torch.empty(n, device='cuda'), n, 1.5, 128)
if hasattr(intj, 'TorchAccessMode'):
    modes = [(name, getattr(intj.TorchAccessMode, member), 'torch_access_mode') for name, member in [('static', 'STATIC_COMPILE'), ('shim', 'RUNTIME_SHIM'), ('interpreter', 'INTERPRETER')]]
else:
    from intj.torch_abi import TorchAccess
    modes = [(name, getattr(TorchAccess, member), 'torch_access') for name, member in [('static', 'CXX'), ('shim', 'SHIM'), ('interpreter', 'CPYTHON')]]
for name, mode, keyword in modes:
    factory = intj.make_launcher(bench_launch.noop, **{keyword: mode})
    fn = factory.__self__.spec_key
    for _ in range(100): fn(*args)
    results = []
    for i in range(9):
        start = __import__('time').perf_counter_ns()
        for _ in range(100000): fn(*args)
        results.append((__import__('time').perf_counter_ns()-start)/100000)
    print(name, round(statistics.median(results), 2), [round(x, 2) for x in results], flush=True)
```

### Entry driver

```python
import os,sys,statistics,time
sys.path.insert(0,os.path.join(os.environ['PYTHONPATH'].split(':')[0],'benchmarks'))
import torch,bench_launch,intj
n=4096
x=torch.randn(n); y=torch.randn(n); o=torch.empty(n)
fn=intj.make_launcher(bench_launch.noop,no_gpu=True)
for label,grid in [('zero-int',0),('zero-tuple',(0,)),('one-tuple',(1,))]:
 args=(0,0,grid,x,y,o,n,1.5,128)
 for _ in range(100): fn(*args)
 results=[]
 for _ in range(9):
  start=time.perf_counter_ns()
  for _ in range(100000): fn(*args)
  results.append((time.perf_counter_ns()-start)/100000)
 print(label,round(statistics.median(results),2),[round(x,2) for x in results],flush=True)
```

## Raw outputs

### gpu

#### Pair 0, baseline

```text
sha=7ed7882c64b334d8839605d7d542b40acbfbe066
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --iters 1000 --batches 9
mode=gpu; 1000 calls × 9 batches; median ns/call
               path    ns/call       Δ ns       Δ %   build ms   bind ms
           auto map     3104.3       +0.0      +0.0      71.47         -
        reduced key     3111.0       +6.6      +0.2       1.75         -
         verify off     2949.9     -154.5      -5.0       1.49         -
          verify on     2901.3     -203.1      -6.5       1.52         -
              baked     2951.3     -153.0      -4.9       7.83         -
       bound tensor     2973.3     -131.0      -4.2       0.34      1.28
      bound pointer     2966.1     -138.2      -4.5       0.32      1.24
   fixed device map     2909.1     -195.2      -6.3       0.29      1.34
fixed device no-map     2932.8     -171.6      -5.5       0.28      1.27
```

#### Pair 0, candidate

```text
sha=a38fb71c95e4064f0d451601f1c4925f1c7444cb
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/torch_abi taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --iters 1000 --batches 9
mode=gpu; 1000 calls × 9 batches; median ns/call
               path    ns/call       Δ ns       Δ %   build ms   bind ms
           auto map     3185.3       +0.0      +0.0      71.31         -
        reduced key     3225.7      +40.4      +1.3    2113.17         -
         verify off     3204.5      +19.2      +0.6    2128.86         -
          verify on     3213.7      +28.4      +0.9    2243.52         -
              baked     3205.7      +20.4      +0.6    2000.10         -
       bound tensor     3238.2      +52.9      +1.7       0.45   1997.39
      bound pointer     3191.0       +5.7      +0.2       0.43   2080.17
   fixed device map     3191.7       +6.4      +0.2       0.41   2110.66
fixed device no-map     3179.8       -5.5      -0.2       0.45   2009.24
```

#### Pair 1, candidate

```text
sha=a38fb71c95e4064f0d451601f1c4925f1c7444cb
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/torch_abi taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --iters 1000 --batches 9
mode=gpu; 1000 calls × 9 batches; median ns/call
               path    ns/call       Δ ns       Δ %   build ms   bind ms
           auto map     3551.4       +0.0      +0.0      71.99         -
        reduced key     3118.2     -433.2     -12.2       3.45         -
         verify off     2933.4     -618.0     -17.4       1.70         -
          verify on     2914.7     -636.7     -17.9       1.65         -
              baked     3013.6     -537.8     -15.1      72.61         -
       bound tensor     2982.5     -568.9     -16.0       0.40      1.46
      bound pointer     2985.9     -565.5     -15.9       0.35      1.46
   fixed device map     2985.1     -566.3     -15.9       0.33      1.53
fixed device no-map     2950.9     -600.5     -16.9       0.27      1.47
```

#### Pair 1, baseline

```text
sha=7ed7882c64b334d8839605d7d542b40acbfbe066
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --iters 1000 --batches 9
mode=gpu; 1000 calls × 9 batches; median ns/call
               path    ns/call       Δ ns       Δ %   build ms   bind ms
           auto map     3140.3       +0.0      +0.0      70.62         -
        reduced key     3835.0     +694.8     +22.1       1.80         -
         verify off     3753.3     +613.0     +19.5       2.08         -
          verify on     3616.7     +476.4     +15.2       2.02         -
              baked     3586.4     +446.1     +14.2       2.13         -
       bound tensor     3669.3     +529.1     +16.8       0.46      1.79
      bound pointer     3579.3     +439.1     +14.0       0.45      1.78
   fixed device map     2988.3     -151.9      -4.8       0.42      1.91
fixed device no-map     2974.5     -165.7      -5.3       0.38      1.26
```

#### Pair 2, baseline

```text
sha=7ed7882c64b334d8839605d7d542b40acbfbe066
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --iters 1000 --batches 9
mode=gpu; 1000 calls × 9 batches; median ns/call
               path    ns/call       Δ ns       Δ %   build ms   bind ms
           auto map     3133.0       +0.0      +0.0     101.73         -
        reduced key     3094.9      -38.1      -1.2       1.72         -
         verify off     2981.4     -151.6      -4.8       1.50         -
          verify on     2993.7     -139.3      -4.4       1.58         -
              baked     2984.2     -148.7      -4.7       1.47         -
       bound tensor     2944.5     -188.5      -6.0       0.33      1.28
      bound pointer     3000.2     -132.8      -4.2       0.33      1.26
   fixed device map     2932.6     -200.4      -6.4       0.30      1.33
fixed device no-map     2906.6     -226.4      -7.2       0.28      1.27
```

#### Pair 2, candidate

```text
sha=a38fb71c95e4064f0d451601f1c4925f1c7444cb
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/torch_abi taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --iters 1000 --batches 9
mode=gpu; 1000 calls × 9 batches; median ns/call
               path    ns/call       Δ ns       Δ %   build ms   bind ms
           auto map     3812.9       +0.0      +0.0      71.46         -
        reduced key     3585.2     -227.7      -6.0       3.57         -
         verify off     3604.2     -208.7      -5.5       1.98         -
          verify on     3745.5      -67.4      -1.8       1.90         -
              baked     3587.2     -225.7      -5.9      71.73         -
       bound tensor     3619.4     -193.4      -5.1       0.42      1.62
      bound pointer     3707.0     -105.9      -2.8       0.42      1.61
   fixed device map     3485.3     -327.6      -8.6       0.36      1.83
fixed device no-map     3782.6      -30.3      -0.8       0.39      1.58
```

#### Pair 3, candidate

```text
sha=a38fb71c95e4064f0d451601f1c4925f1c7444cb
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/torch_abi taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --iters 1000 --batches 9
mode=gpu; 1000 calls × 9 batches; median ns/call
               path    ns/call       Δ ns       Δ %   build ms   bind ms
           auto map     3126.4       +0.0      +0.0      99.82         -
        reduced key     3117.8       -8.6      -0.3       3.30         -
         verify off     2980.1     -146.3      -4.7       1.69         -
          verify on     2931.2     -195.3      -6.2       1.64         -
              baked     2957.3     -169.2      -5.4      72.22         -
       bound tensor     2977.8     -148.6      -4.8       0.27      1.46
      bound pointer     2970.5     -156.0      -5.0       0.34      1.50
   fixed device map     2943.1     -183.3      -5.9       0.32      1.56
fixed device no-map     2881.0     -245.5      -7.9       0.36      1.52
```

#### Pair 3, baseline

```text
sha=7ed7882c64b334d8839605d7d542b40acbfbe066
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --iters 1000 --batches 9
mode=gpu; 1000 calls × 9 batches; median ns/call
               path    ns/call       Δ ns       Δ %   build ms   bind ms
           auto map     3127.8       +0.0      +0.0      70.96         -
        reduced key     3122.1       -5.7      -0.2       1.84         -
         verify off     2986.7     -141.1      -4.5       1.52         -
          verify on     2979.9     -147.9      -4.7       1.54         -
              baked     2951.7     -176.2      -5.6       1.55         -
       bound tensor     2920.2     -207.7      -6.6       0.35      1.29
      bound pointer     2990.0     -137.8      -4.4       0.34      1.27
   fixed device map     2947.2     -180.6      -5.8       0.32      1.35
fixed device no-map     2944.3     -183.6      -5.9       0.27      1.29
```

#### Pair 4, baseline

```text
sha=7ed7882c64b334d8839605d7d542b40acbfbe066
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --iters 1000 --batches 9
mode=gpu; 1000 calls × 9 batches; median ns/call
               path    ns/call       Δ ns       Δ %   build ms   bind ms
           auto map     3119.9       +0.0      +0.0      71.37         -
        reduced key     3130.2      +10.3      +0.3       1.80         -
         verify off     2973.0     -146.9      -4.7       1.53         -
          verify on     2912.9     -207.0      -6.6       1.45         -
              baked     2892.9     -227.0      -7.3       1.47         -
       bound tensor     2995.5     -124.3      -4.0       0.34      1.28
      bound pointer     2896.8     -223.0      -7.1       0.34      1.28
   fixed device map     2907.6     -212.3      -6.8       0.29      1.34
fixed device no-map     3026.1      -93.7      -3.0       0.27      1.27
```

#### Pair 4, candidate

```text
sha=a38fb71c95e4064f0d451601f1c4925f1c7444cb
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/torch_abi taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --iters 1000 --batches 9
mode=gpu; 1000 calls × 9 batches; median ns/call
               path    ns/call       Δ ns       Δ %   build ms   bind ms
           auto map     3157.3       +0.0      +0.0      82.75         -
        reduced key     3124.3      -33.0      -1.0       3.31         -
         verify off     3023.5     -133.8      -4.2       1.70         -
          verify on     2965.9     -191.4      -6.1       1.64         -
              baked     2951.7     -205.6      -6.5      88.07         -
       bound tensor     3011.4     -145.9      -4.6       0.35      1.46
      bound pointer     2948.8     -208.5      -6.6       0.34      1.49
   fixed device map     2928.1     -229.2      -7.3       0.31      1.53
fixed device no-map     2890.5     -266.8      -8.4       0.27      1.48
```

### host

#### Pair 0, baseline

```text
sha=7ed7882c64b334d8839605d7d542b40acbfbe066
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --no-gpu --iters 100000 --batches 9
mode=host-only; 100000 calls × 9 batches; median ns/call
               path    ns/call       Δ ns       Δ %   build ms   bind ms
           auto map       43.6       +0.0      +0.0      87.83         -
        reduced key       43.1       -0.5      -1.2       1.34         -
         verify off       43.6       -0.0      -0.1       1.20         -
          verify on       44.4       +0.8      +1.7       1.16         -
              baked       39.6       -4.0      -9.2       1.34         -
       bound tensor       45.4       +1.7      +3.9       0.13      1.10
      bound pointer       45.0       +1.4      +3.2       0.11      1.07
   fixed device map       43.8       +0.2      +0.4       0.09      1.08
fixed device no-map       45.1       +1.4      +3.3       0.12      1.03
```

#### Pair 0, candidate

```text
sha=a38fb71c95e4064f0d451601f1c4925f1c7444cb
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/torch_abi taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --no-gpu --iters 100000 --batches 9
mode=host-only; 100000 calls × 9 batches; median ns/call
               path    ns/call       Δ ns       Δ %   build ms   bind ms
           auto map       44.0       +0.0      +0.0      59.60         -
        reduced key       43.5       -0.5      -1.1       1.52         -
         verify off       43.3       -0.7      -1.6       5.44         -
          verify on       44.2       +0.2      +0.5       1.37         -
              baked       40.2       -3.8      -8.7       2.54         -
       bound tensor       45.9       +1.9      +4.4       0.15      1.30
      bound pointer       45.1       +1.1      +2.5       0.13      1.24
   fixed device map       44.5       +0.5      +1.1       0.33      4.91
fixed device no-map       43.5       -0.5      -1.1       0.15      1.22
```

#### Pair 1, candidate

```text
sha=a38fb71c95e4064f0d451601f1c4925f1c7444cb
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/torch_abi taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --no-gpu --iters 100000 --batches 9
mode=host-only; 100000 calls × 9 batches; median ns/call
               path    ns/call       Δ ns       Δ %   build ms   bind ms
           auto map       44.1       +0.0      +0.0      59.63         -
        reduced key       43.3       -0.8      -1.8       1.57         -
         verify off       43.3       -0.7      -1.7       1.39         -
          verify on       44.6       +0.5      +1.1       1.35         -
              baked       40.3       -3.8      -8.6       2.58         -
       bound tensor       45.8       +1.8      +4.0       0.15      1.31
      bound pointer       45.1       +1.1      +2.4       0.15      1.29
   fixed device map       44.4       +0.3      +0.8       0.11      1.23
fixed device no-map       43.4       -0.7      -1.5       0.13      1.22
```

#### Pair 1, baseline

```text
sha=7ed7882c64b334d8839605d7d542b40acbfbe066
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --no-gpu --iters 100000 --batches 9
mode=host-only; 100000 calls × 9 batches; median ns/call
               path    ns/call       Δ ns       Δ %   build ms   bind ms
           auto map       43.5       +0.0      +0.0      58.91         -
        reduced key       42.9       -0.6      -1.3       1.33         -
         verify off       43.8       +0.3      +0.7       1.20         -
          verify on       44.4       +0.9      +2.2       1.14         -
              baked       39.6       -3.9      -8.9       1.31         -
       bound tensor       45.5       +2.0      +4.7       0.13      1.11
      bound pointer       44.3       +0.8      +1.9       0.11      1.08
   fixed device map       43.8       +0.4      +0.9       0.09      1.06
fixed device no-map       44.7       +1.3      +2.9       0.12      1.04
```

#### Pair 2, baseline

```text
sha=7ed7882c64b334d8839605d7d542b40acbfbe066
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --no-gpu --iters 100000 --batches 9
mode=host-only; 100000 calls × 9 batches; median ns/call
               path    ns/call       Δ ns       Δ %   build ms   bind ms
           auto map       43.5       +0.0      +0.0      60.22         -
        reduced key       43.2       -0.3      -0.7       1.34         -
         verify off       43.5       -0.0      -0.1       1.21         -
          verify on       44.2       +0.6      +1.5       1.15         -
              baked       39.8       -3.7      -8.5       5.89         -
       bound tensor       45.4       +1.9      +4.3       0.13      1.16
      bound pointer       44.3       +0.8      +1.9       0.13      1.14
   fixed device map       44.7       +1.2      +2.8       0.09      1.08
fixed device no-map       44.7       +1.2      +2.7       0.12      1.05
```

#### Pair 2, candidate

```text
sha=a38fb71c95e4064f0d451601f1c4925f1c7444cb
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/torch_abi taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --no-gpu --iters 100000 --batches 9
mode=host-only; 100000 calls × 9 batches; median ns/call
               path    ns/call       Δ ns       Δ %   build ms   bind ms
           auto map       44.0       +0.0      +0.0      59.49         -
        reduced key       43.6       -0.5      -1.1      13.22         -
         verify off       43.8       -0.3      -0.6       1.46         -
          verify on       44.3       +0.3      +0.6       1.36         -
              baked       39.8       -4.2      -9.6       2.64         -
       bound tensor       45.5       +1.5      +3.3       0.15      1.29
      bound pointer       45.2       +1.2      +2.7       0.13      1.27
   fixed device map       46.3       +2.2      +5.0       0.11      1.22
fixed device no-map       43.3       -0.7      -1.6       0.13      1.24
```

#### Pair 3, candidate

```text
sha=a38fb71c95e4064f0d451601f1c4925f1c7444cb
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/torch_abi taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --no-gpu --iters 100000 --batches 9
mode=host-only; 100000 calls × 9 batches; median ns/call
               path    ns/call       Δ ns       Δ %   build ms   bind ms
           auto map       45.2       +0.0      +0.0      58.81         -
        reduced key       43.9       -1.3      -2.9      12.08         -
         verify off       43.5       -1.6      -3.7       1.44         -
          verify on       44.6       -0.6      -1.3       1.34         -
              baked       40.0       -5.2     -11.4       2.37         -
       bound tensor       46.0       +0.8      +1.8       0.15      1.28
      bound pointer       45.1       -0.0      -0.1       0.13      1.23
   fixed device map       44.5       -0.6      -1.4       0.10      1.19
fixed device no-map       43.4       -1.7      -3.8       0.14      1.19
```

#### Pair 3, baseline

```text
sha=7ed7882c64b334d8839605d7d542b40acbfbe066
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --no-gpu --iters 100000 --batches 9
mode=host-only; 100000 calls × 9 batches; median ns/call
               path    ns/call       Δ ns       Δ %   build ms   bind ms
           auto map       43.4       +0.0      +0.0      59.01         -
        reduced key       43.0       -0.4      -0.9       1.36         -
         verify off       43.7       +0.2      +0.5       1.17         -
          verify on       44.0       +0.6      +1.3       1.27         -
              baked       39.7       -3.8      -8.7       1.36         -
       bound tensor       45.5       +2.1      +4.8       0.13      1.11
      bound pointer       44.2       +0.8      +1.7       0.12      1.08
   fixed device map       46.4       +3.0      +6.9       0.09      1.09
fixed device no-map       46.4       +3.0      +6.8       0.13      1.06
```

#### Pair 4, baseline

```text
sha=7ed7882c64b334d8839605d7d542b40acbfbe066
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --no-gpu --iters 100000 --batches 9
mode=host-only; 100000 calls × 9 batches; median ns/call
               path    ns/call       Δ ns       Δ %   build ms   bind ms
           auto map       45.2       +0.0      +0.0      59.72         -
        reduced key       43.4       -1.8      -4.0       1.35         -
         verify off       43.9       -1.3      -2.8       1.21         -
          verify on       44.2       -1.0      -2.2       1.18         -
              baked       39.6       -5.6     -12.4       1.34         -
       bound tensor       45.2       -0.0      -0.0       0.12      1.10
      bound pointer       44.3       -0.9      -1.9       0.12      1.07
   fixed device map       43.7       -1.5      -3.4       0.10      1.05
fixed device no-map       46.5       +1.4      +3.0       0.12      1.01
```

#### Pair 4, candidate

```text
sha=a38fb71c95e4064f0d451601f1c4925f1c7444cb
command=PYTHONPATH=/mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/torch_abi taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --no-gpu --iters 100000 --batches 9
mode=host-only; 100000 calls × 9 batches; median ns/call
               path    ns/call       Δ ns       Δ %   build ms   bind ms
           auto map       45.3       +0.0      +0.0      59.38         -
        reduced key       43.6       -1.7      -3.8       1.54         -
         verify off       43.5       -1.8      -3.9       1.40         -
          verify on       44.5       -0.8      -1.7       1.33         -
              baked       40.0       -5.3     -11.7       2.51         -
       bound tensor       45.8       +0.5      +1.2       0.16      1.30
      bound pointer       45.2       -0.1      -0.2       0.13      1.24
   fixed device map       45.0       -0.3      -0.7       0.10      1.22
fixed device no-map       43.8       -1.5      -3.3       0.14      1.25
```

### tensor-key

#### Pair 0, baseline

```text
sha=7ed7882c64b334d8839605d7d542b40acbfbe066
static 117.0 [116.58, 118.39, 116.75, 116.71, 117.92, 117.61, 116.98, 117.0, 201.33]
shim 119.21 [118.05, 119.21, 120.03, 119.62, 118.07, 121.19, 119.99, 119.11, 118.25]
interpreter 911.99 [910.53, 910.55, 1053.51, 913.11, 905.94, 911.99, 1097.84, 964.84, 908.62]
```

#### Pair 0, candidate

```text
sha=a38fb71c95e4064f0d451601f1c4925f1c7444cb
static 116.75 [116.4, 117.27, 116.75, 119.26, 115.91, 116.57, 119.49, 116.36, 118.06]
shim 116.63 [116.67, 117.75, 116.81, 116.53, 116.05, 116.63, 115.74, 117.03, 116.44]
interpreter 906.13 [909.07, 906.94, 1173.51, 906.13, 900.22, 905.13, 905.94, 1081.82, 901.13]
```

#### Pair 1, candidate

```text
sha=a38fb71c95e4064f0d451601f1c4925f1c7444cb
static 116.79 [113.44, 115.14, 115.67, 116.79, 114.84, 117.92, 118.37, 140.27, 264.18]
shim 116.39 [135.38, 115.86, 115.22, 118.24, 116.0, 116.39, 116.59, 117.16, 114.66]
interpreter 909.7 [906.43, 904.82, 1118.81, 902.8, 909.7, 904.62, 1045.5, 1355.48, 1342.77]
```

#### Pair 1, baseline

```text
sha=7ed7882c64b334d8839605d7d542b40acbfbe066
static 114.81 [112.74, 115.77, 115.92, 115.28, 114.81, 113.85, 115.25, 113.89, 113.46]
shim 117.45 [118.56, 118.96, 118.75, 117.44, 118.06, 117.15, 117.45, 116.68, 117.23]
interpreter 967.73 [957.36, 965.88, 967.65, 968.97, 968.72, 970.11, 967.73, 965.04, 968.18]
```

#### Pair 2, baseline

```text
sha=7ed7882c64b334d8839605d7d542b40acbfbe066
static 114.85 [113.21, 116.01, 115.07, 116.14, 115.12, 113.81, 114.85, 114.41, 113.58]
shim 115.48 [114.24, 116.5, 114.7, 114.96, 115.48, 117.87, 114.55, 116.18, 115.75]
interpreter 1144.05 [1152.0, 1505.88, 1518.92, 1229.93, 900.8, 902.22, 901.34, 902.81, 1144.05]
```

#### Pair 2, candidate

```text
sha=a38fb71c95e4064f0d451601f1c4925f1c7444cb
static 119.52 [120.58, 121.09, 123.92, 120.41, 119.52, 116.18, 116.22, 115.66, 117.06]
shim 120.7 [119.83, 122.19, 119.04, 122.16, 120.7, 121.56, 120.58, 121.29, 118.9]
interpreter 911.07 [911.07, 906.85, 906.85, 912.17, 906.01, 906.59, 975.54, 957.58, 992.86]
```

#### Pair 3, candidate

```text
sha=a38fb71c95e4064f0d451601f1c4925f1c7444cb
static 120.18 [120.11, 120.79, 122.02, 120.73, 120.18, 117.82, 119.26, 117.52, 183.39]
shim 120.94 [120.94, 121.09, 119.58, 122.47, 119.24, 121.99, 118.58, 121.93, 119.92]
interpreter 904.22 [905.83, 904.22, 901.9, 1143.96, 903.76, 910.41, 902.68, 902.21, 1100.93]
```

#### Pair 3, baseline

```text
sha=7ed7882c64b334d8839605d7d542b40acbfbe066
static 117.11 [117.47, 119.74, 116.91, 118.09, 117.11, 115.27, 122.31, 115.03, 115.05]
shim 116.78 [115.26, 115.89, 116.78, 116.37, 117.5, 117.06, 119.14, 116.92, 116.59]
interpreter 905.44 [897.12, 1095.48, 902.88, 902.34, 908.77, 904.9, 1253.88, 906.49, 905.44]
```

#### Pair 4, baseline

```text
sha=7ed7882c64b334d8839605d7d542b40acbfbe066
static 116.99 [114.76, 118.48, 116.99, 115.55, 116.51, 116.8, 117.28, 117.8, 118.08]
shim 117.28 [116.23, 117.28, 117.81, 118.03, 118.84, 116.11, 116.5, 119.67, 116.35]
interpreter 906.24 [1045.47, 907.09, 905.74, 1122.65, 905.83, 904.01, 906.24, 1072.23, 906.09]
```

#### Pair 4, candidate

```text
sha=a38fb71c95e4064f0d451601f1c4925f1c7444cb
static 115.98 [115.04, 117.17, 115.33, 115.63, 115.24, 117.61, 116.17, 116.21, 115.98]
shim 117.71 [117.71, 120.27, 116.51, 118.71, 116.24, 120.74, 116.54, 118.63, 116.93]
interpreter 919.26 [996.6, 919.26, 908.0, 1039.73, 945.8, 905.83, 910.24, 1048.3, 907.13]
```

### entry

#### Pair 0, baseline

```text
sha=7ed7882c64b334d8839605d7d542b40acbfbe066
zero-int 56.78 [56.86, 57.11, 57.41, 58.5, 55.39, 56.78, 55.02, 56.37, 55.62]
zero-tuple 57.44 [57.07, 56.84, 57.15, 57.96, 59.22, 58.16, 57.26, 58.2, 57.44]
one-tuple 73.01 [72.91, 73.18, 73.38, 72.22, 71.89, 73.01, 74.76, 74.08, 72.81]
```

#### Pair 0, candidate

```text
sha=a38fb71c95e4064f0d451601f1c4925f1c7444cb
zero-int 52.26 [51.7, 52.56, 51.51, 52.69, 52.61, 52.31, 51.64, 52.26, 51.95]
zero-tuple 53.22 [53.22, 53.4, 55.44, 53.22, 54.25, 53.19, 53.17, 53.31, 53.19]
one-tuple 68.05 [68.39, 68.99, 68.05, 67.27, 68.08, 67.43, 69.83, 68.0, 67.92]
```

#### Pair 1, candidate

```text
sha=a38fb71c95e4064f0d451601f1c4925f1c7444cb
zero-int 51.94 [51.94, 53.59, 51.53, 52.19, 51.32, 53.35, 51.66, 53.36, 51.43]
zero-tuple 53.37 [53.62, 52.67, 55.17, 52.69, 54.14, 53.06, 53.37, 52.61, 54.24]
one-tuple 70.86 [69.53, 73.6, 68.86, 71.41, 69.02, 70.86, 71.94, 71.24, 69.71]
```

#### Pair 1, baseline

```text
sha=7ed7882c64b334d8839605d7d542b40acbfbe066
zero-int 53.78 [53.94, 53.95, 53.33, 53.78, 53.33, 54.02, 53.11, 54.49, 53.55]
zero-tuple 55.26 [55.44, 55.26, 55.71, 55.43, 54.68, 54.38, 55.09, 55.2, 55.36]
one-tuple 69.79 [69.56, 69.53, 69.84, 71.1, 69.79, 69.78, 68.99, 70.12, 69.83]
```

#### Pair 2, baseline

```text
sha=7ed7882c64b334d8839605d7d542b40acbfbe066
zero-int 53.59 [53.59, 54.78, 53.53, 53.82, 53.2, 54.18, 53.17, 54.07, 53.24]
zero-tuple 54.74 [56.82, 54.43, 55.57, 54.58, 54.88, 54.71, 55.06, 54.04, 54.74]
one-tuple 71.82 [70.92, 72.54, 71.49, 72.31, 71.49, 71.96, 72.14, 71.82, 71.1]
```

#### Pair 2, candidate

```text
sha=a38fb71c95e4064f0d451601f1c4925f1c7444cb
zero-int 53.98 [53.73, 54.44, 53.37, 54.75, 53.98, 54.29, 53.67, 54.14, 53.27]
zero-tuple 55.16 [55.16, 54.28, 55.14, 55.02, 56.99, 56.1, 57.97, 54.66, 56.1]
one-tuple 69.74 [68.91, 71.33, 72.08, 70.91, 68.97, 69.38, 69.74, 70.41, 69.46]
```

#### Pair 3, candidate

```text
sha=a38fb71c95e4064f0d451601f1c4925f1c7444cb
zero-int 53.36 [54.2, 53.48, 52.6, 53.54, 52.9, 53.34, 53.8, 53.36, 53.01]
zero-tuple 54.61 [54.22, 53.48, 54.64, 53.42, 54.61, 55.41, 55.66, 54.58, 55.43]
one-tuple 68.99 [68.9, 69.68, 68.79, 69.24, 68.07, 71.76, 68.99, 69.39, 68.28]
```

#### Pair 3, baseline

```text
sha=7ed7882c64b334d8839605d7d542b40acbfbe066
zero-int 51.85 [51.85, 52.14, 51.47, 51.92, 51.41, 51.97, 51.49, 52.1, 51.34]
zero-tuple 52.95 [53.29, 52.51, 53.05, 52.72, 53.64, 52.55, 52.95, 52.78, 54.3]
one-tuple 67.58 [67.62, 67.96, 67.07, 67.61, 67.28, 67.58, 67.34, 67.67, 67.41]
```

#### Pair 4, baseline

```text
sha=7ed7882c64b334d8839605d7d542b40acbfbe066
zero-int 59.82 [59.05, 60.29, 59.57, 59.82, 60.2, 60.43, 59.07, 60.42, 59.21]
zero-tuple 61.29 [60.59, 61.5, 62.1, 60.96, 61.62, 61.22, 61.29, 60.51, 62.35]
one-tuple 79.31 [79.87, 80.13, 79.18, 79.86, 80.23, 79.31, 78.52, 79.18, 78.75]
```

#### Pair 4, candidate

```text
sha=a38fb71c95e4064f0d451601f1c4925f1c7444cb
zero-int 55.8 [55.09, 56.22, 55.63, 56.25, 55.51, 55.93, 55.64, 56.03, 55.8]
zero-tuple 57.16 [57.76, 56.79, 57.16, 56.35, 57.46, 56.28, 57.39, 56.32, 58.66]
one-tuple 72.42 [73.54, 73.48, 72.42, 72.36, 71.75, 73.58, 72.26, 73.17, 71.62]
```

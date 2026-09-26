# One-entry last-key cache — 2026-09-25

The keyed launcher now remembers the most recent spec key and its compiled
kernel in the owning cache. It still packs and validates arguments on every
nonzero launch. An identical key skips the hash and map probe; a changed key
uses the existing map. The no-key bound launcher still uses `fixed_kernel`.

## Host benchmark

Baseline: `3adb2de90b69c02c7a1966f33a73581cfe8d6192`. Candidate: this
working tree. The new `--last-key` mode calls each launcher twice per loop with
prebuilt tuples. `repeat` calls the same key twice; `alternate` changes only
the final integer argument from 17 to 16. Both calls are inside the same timed
loop, so the Python loop boundary is identical.

Five fresh processes per revision ran in alternating baseline/candidate order,
each with 9 batches of 100,000 pairs on CPU 0. Each number is the median of
the five process medians, in ns per call. The baseline was a `git archive` of
the commit above; each revision had a separate `TRITON_HOME`.

| Arguments | Pattern | Baseline | Candidate | Change |
|---:|---|---:|---:|---:|
| 4 | repeat | 35.4 | 33.6 | -5.1% |
| 4 | alternate | 35.5 | 35.9 | +1.1% |
| 16 | repeat | 68.6 | 66.3 | -3.4% |
| 16 | alternate | 68.5 | 69.9 | +2.0% |
| 32 | repeat | 110.5 | 108.0 | -2.3% |
| 32 | alternate | 110.9 | 118.1 | +6.5% |

The 32-argument alternating case pays for a full-key comparison before its
usual hash and map probe. A word-first guard did not reduce that cost in five
repeat runs, so it was removed.

The existing `--no-gpu` matrix, run with the same five-process method and
100,000 calls per batch, also shows the keyed path improving. The no-map row is
a control because it never uses this cache.

| Host launch path | Baseline (ns) | Candidate (ns) | Change |
|---|---:|---:|---:|
| auto map | 43.5 | 41.6 | -4.4% |
| fixed device map | 44.2 | 42.5 | -3.8% |
| fixed device no-map | 45.2 | 45.2 | 0.0% |

Reproduce one process:

```sh
PYTHONPATH=$PWD taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --last-key --iters 100000 --batches 9
PYTHONPATH=$PWD taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --no-gpu --iters 100000 --batches 9
```

## GPU check and validation

Three paired runs of the standard GPU matrix used 20,000 calls and 7 batches
per row. Even `fixed device no-map`, which never uses this cache, ranged from
3.02 to 3.75 us in the baseline and 3.09 to 3.92 us in the candidate. Those
runs cannot isolate a GPU launch effect from machine variation.

`PYTHONPATH=$PWD /tmp/gb2/bin/python -m pytest tests -q`: 530 passed,
1 skipped (Google Benchmark unavailable). `pyright --pythonpath
/tmp/gb2/bin/python`: 0 errors. CUDA was compile-checked; NVIDIA runtime was
not tested.

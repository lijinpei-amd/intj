# 12-case gate: TRITON_INTERPRET=1 refusal deferred to first call (index 0)

Candidate `839a0e0` ("Defer the TRITON_INTERPRET=1 refusal to a launcher's first call"). Per its own commit
message the diff touches `LauncherFactory._build`'s lazy-build path and `make_launcher`'s decoration-time
dispatch only (`git show --stat 839a0e0`: `intj/launcher.py`, `tests/test_launcher.py`, `docs/Usage.md`), not
any rendered entry or per-call path, so no launch/decode/hit-path change was expected. Ran the full 12-case
gate to confirm.

## Environment

- Candidate `839a0e0` (branch `develop`, clean checkout) at
  `/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj`. Immediate parent `429d8e1` as a detached worktree
  at `/tmp/interp-refusal-parent` for the A/B round.
- CPU: `taskset -c 0`, one process at a time. Shared host: `uptime` load average 17.33/16.18/16.24 before the
  gate started, 3.81/9.41/13.52 right after it finished (another agent's aiter tests pinned to GPUs 4-7 during
  this run); 17.25/11.21/13.30 after the rerun + A/B rounds.
- GPU: AMD Instinct MI308X (`gfx942`), GPU 0 (`HIP_VISIBLE_DEVICES=0`).
- Python 3.12.3 (`/tmp/gb2/bin/python`), Torch `2.14.0+rocm7.2`, Triton `3.8.0`, cc 13.3.0.
- `INTJ_BENCHMARK_ROOT=/tmp/gbench`. CUDA: untested.

## Commands

```bash
# 1. the 12-case gate, 3 rounds, core 0
HIP_VISIBLE_DEVICES=0 /tmp/intj-bench/run_all.sh interp-refusal
/tmp/gb2/bin/python /tmp/intj-bench/bench_compare.py obj-lookup interp-refusal        # primary: same hit path as 2a876ff
/tmp/gb2/bin/python /tmp/intj-bench/bench_compare.py lazy-baseline interp-refusal     # secondary, older reference

# 2. one rerun of the sole primary-comparison flag, launch_dynamic only, 3 rounds
PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python \
  benchmarks/bench_launch.py --dynamic --iters 20000 --batches 9        # x3, -> interp-refusal-rerun/round{0,1,2}

# 3. still flagged: interleaved A/B, 5 processes each, parent (429d8e1, /tmp/interp-refusal-parent) vs
#    candidate (839a0e0, this tree), alternating
for r in 0 1 2 3 4; do for rev in parent cand; do
  cd <tree of rev>; PYTHONPATH=<tree> INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python \
    benchmarks/bench_launch.py --dynamic --iters 20000 --batches 9 > interp-refusal-ab/${rev}_round$r.txt
done; done
```

## Result

**No regression.** All 12 cases pass against `obj-lookup` (`2a876ff`, same hit path per its own commit
message -- `429d8e1` between them didn't touch it either) except one `launch_dynamic` row; that row also
flagged the same commit's own first rerun, but an interleaved parent/candidate A/B put both revisions at the
same value, so it is host placement noise, not a cost from this commit.

**Primary gate (vs `obj-lookup`, 3 rounds): one flag.** `launch_dynamic` `str constexpr` `spec_key`:
89.1 -> 92.9 ns (+3.8, +4.3 %). Every other host (ns) row across `launch_gpu`, `launch_host`, `launch_last_key`,
`launch_readme` (`readme_path`, `readme_torch_access`), `launch_sweep`, `launch_dynamic`'s other three rows,
`launch_tuned`, `kernel_cache` (all three key sizes, all three implementations), `ffi_paths`' INTJ rows, and
`ffi_sweep` was within the rule (largest unflagged deltas: `launch_sweep` 32 int -3.8 %, `hip_module` INTJ same
function -15.9 % *faster*, GPU (us) rows on `ffi_compare`/`hip_module` broadly 8-17 % faster including TVM FFI's
own rows -- moved with the GPU/host, not this commit, consistent with the obj-lookup journal's own finding on
the same rows). `ffi_paths` "FFI unpack Pair only" (+2.2%) and "INTJ cold compile callback + cache" (+0.9%,
baseline 705.8) were within the rule against `obj-lookup`.

**Secondary gate (vs `lazy-baseline`, older reference, informational only):** three additional flags that do
not reproduce against `obj-lookup`: `ffi_paths` "FFI unpack Pair only" (162.4 -> 167.9, +3.4%, the FFI library's
own row, not INTJ's, and not flagged against the closer `obj-lookup` baseline of 164.3), `ffi_paths` "INTJ cold
compile callback + cache" (404.6 -> 712.0, +76%, but `obj-lookup`'s own baseline for this row was already 705.8
-- the jump happened before this commit, between `lazy-baseline` and `obj-lookup`, not here), and
`launch_readme` `readme_torch_access` `static_compile` decode ns (89.7 -> 96.8, +7.9%, `obj-lookup`'s baseline
was already 95.7, same story). `launch_dynamic`/`launch_tuned` are `MISSING in baseline` there (added after
`lazy-baseline` was taken). None of these three needed a rerun: the `obj-lookup` comparison is the one this
task's diff is judged against, and none reproduce there.

**Rerun of the one primary flag (`launch_dynamic`, 3 processes, `interp-refusal-rerun`):** `str constexpr`
`spec_key` medians 92.9, 92.6, 92.4 ns (round-of-3 median 92.6) against the `obj-lookup` baseline's 89.1, 89.5,
87.7 ns (median 89.1) -- still flagged (+3.5 ns, +3.9 %).

**Interleaved A/B (5 + 5 processes, `429d8e1` vs `839a0e0`, `interp-refusal-ab`):** `str constexpr` `spec_key`,
parent rounds 92.4, 92.3, 92.8, 90.8, 93.3 ns (median 92.4); candidate rounds 92.4, 92.2, 92.6, 179.5\*, 92.0 ns
(median 92.3, or 92.4 with the outlier's rank included). Parent and candidate land on the same number when run
back-to-back on this host; the ~89 ns seen in `obj-lookup` and its own baseline came from a quieter host window,
not this commit. \*Round 3's 4850.6 ns `launch` / 179.5 ns `spec_key` is a single stalled process (the whole
row inflated together, not a per-call cost) and does not change the median.

**Verdict: not a regression.** `839a0e0`'s diff is confirmed Python-build-path only by both the commit message
and this gate: the one row that ever flagged moves identically on the unchanged parent commit when measured in
the same interleaved run, so the difference is host placement, not `839a0e0`. First-call (build) latency was
not measured here; only warmed launches per the twelve cases above.

### Gate raw output, round 0 of 3 (`interp-refusal`, commit `839a0e0`)

```
$ cat /tmp/intj-bench/interp-refusal/round0_launch_dynamic.txt
commit=839a0e0c4be8b0d050d6b02971fa1c21e44abd03
started=2026-09-28T12:16:07+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --dynamic --iters 20000 --batches 9
mode=dynamic; 20000 calls x 9 batches; median ns/call
               path     launch   spec_key
         lazy plain     3001.0       94.9
  2 dynamic options     3015.4       98.5
      int constexpr     2966.1       88.4
      str constexpr     2977.3       92.9
elapsed_ns=5568963671
exit_status=0
```

```
$ cat /tmp/intj-bench/interp-refusal/round0_kernel_cache.txt | head -12
commit=839a0e0c4be8b0d050d6b02971fa1c21e44abd03
started=2026-09-28T12:16:23+08:00
command=PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python -m pytest tests/test_kernel_cache.py -s -q

kernel cache, 1-word key, ns per operation
```

Full round0-2 raw files for all 12 cases are under `/tmp/intj-bench/interp-refusal/round{0,1,2}_<case>.txt`
(not reproduced in full here; `bench_compare.py`'s per-row table above is the complete diff against both
baselines). Rerun raw files: `/tmp/intj-bench/interp-refusal-rerun/round{0,1,2}_launch_dynamic.txt`. A/B raw
files: `/tmp/intj-bench/interp-refusal-ab/{parent,cand}_round{0,1,2,3,4}.txt`.

### `launch_dynamic` `str constexpr` `spec_key`, all measurements (ns)

```
obj-lookup (baseline, 3 rounds):        89.1   89.5   87.7   (median 89.1)
interp-refusal (candidate, 3 rounds):   92.9   ...    ...    (median 92.9, see full gate table above)
interp-refusal-rerun (candidate, 3):    92.9   92.6   92.4   (median 92.6)
interp-refusal-ab parent  429e8d1 (5):  92.4   92.3   92.8   90.8   93.3   (median 92.4)
interp-refusal-ab cand    839a0e0 (5):  92.4   92.2   92.6  179.5*  92.0   (median 92.3)
```

# 2026-09-28 mix-history round 0: constexpr vs non-constexpr launch time across intj history

`benchmarks/bench_mix.py`, host-only (`no_gpu=True`), run against every commit on
`develop` that touches `intj/` (`git rev-list --reverse develop -- intj/`, 126
commits, `67f2c06`..`6e0d4bc`). 50 commits measured; 76 cannot run `no_gpu`.

## Setup

- Host: Intel Xeon Platinum 8480C, 224 CPUs; every process pinned with
  `taskset -c 8`. GPU unused (`no_gpu=True`, CPU `torch.empty(4096)` tensors).
- Python 3.12.3, triton 3.8.0, torch 2.14.0+rocm7.2, gcc 13.3.0 (`/tmp/gb2`).
- One reused worktree `/tmp/intj-mix-wt` (`git checkout --detach -f <sha>` +
  `git clean -fdx`), `PYTHONPATH=/tmp/intj-mix-wt`, and a fresh
  `TRITON_HOME=/tmp/intj-mix/home/<sha>` per commit, so no commit loads a
  module another commit built (old `ModuleKey`s do not hash every header).
- Script: a copy of `bench_mix.py` at `/tmp/intj-mix/bench_mix.py` (sha256
  `eb914c99...aa906b4`). The committed file differs only by a pyright fix
  (`first_c` renamed `last_c` and initialised to `-1`; same tuples) and ruff formatting.
- Timing: `bench_pair` style: 200 warm-up pairs, then 9 batches of 20000
  iterations, two direct calls per iteration with prebuilt tuples
  (40000 calls per batch); a row is the median of batch means. Controls are
  `(0, 0, (1,))`. Build and the first two calls sit outside the timer.
- Sweep: 3 separate processes per commit, value = median of the 3 medians.
  Load average (1 min) before each process is recorded; it was 2-12 for most
  commits and peaked at 55 around `51f6cc5`/`f4454e9` (other agents' jobs).
- Confirmation: every transition flagged by the sweep was rerun interleaved
  (base, candidate, base, ... 5 processes each), same core.
- Flag rule: slower than the nearest measured first-parent ancestor ("base")
  by more than 3 % and more than 2 ns. History is non-linear (side branches
  merge into `develop`), so the base is the ancestor, not the previous
  commit in `rev-list` order.
- Placement check: `objdump -d` of the rendered `.so`, per function with
  addresses stripped (`/tmp/intj-mix/insn_diff.py <base> <cand> mix_<n>_<share>`).

Commands:

```sh
/tmp/gb2/bin/python /tmp/intj-mix/sweep.py 3     # per-commit sweep, raw -> /tmp/intj-mix/raw/<idx>_<sha>.txt
/tmp/gb2/bin/python /tmp/intj-mix/analyze.py     # medians, bases, flags -> summary.json, flag_pairs.json
/tmp/gb2/bin/python /tmp/intj-mix/confirm.py     # interleaved 5+5 reruns -> /tmp/intj-mix/confirm/<cand>_vs_<base>.txt
/tmp/gb2/bin/python /tmp/intj-mix/confirm_analyze.py
# each process:
TRITON_HOME=/tmp/intj-mix/home/<sha> PYTHONPATH=/tmp/intj-mix-wt taskset -c 8 \
  /tmp/gb2/bin/python /tmp/intj-mix/bench_mix.py            # --iters 20000 --batches 9
```

Grid: 3 counts (4, 16, 32) x constexpr share (0, 25, 50, 100 %) x constexpr
kind (`int` 3/5, `str` "relu"/"gelu", `dtype` float16/float32) x non-constexpr
kind (`int` 17, CPU `tensor`, `mixed` tensor/17/1.5) x pattern (`repeat`;
`alternate` flips the last constexpr, or at 0 % an int 17/16 or an aligned
/unaligned tensor). 144 rows; about 16 s per warm process at HEAD.

## Skipped commits

| idx | commits | reason |
|---|---|---|
| 0-44 | `67f2c06`..`122e2e7` | `intj` has no `make_launcher` (it was `create_launcher`) |
| 45-66, 71, 73, 81-84, 87-89 | `9297e34`..`2fb0950`, and branch commits `15f646a`, `afd7075`, `d2e433f`..`0fa5f10`, `d054d23`..`ac05dea` | `make_launcher` has no `no_gpu` (added in `f87f970`; the torch_abi branch forked before it) |

`str` and `dtype` constexpr rows are `n/a` before `a6292eb` (`TypeError: intj:
unsupported constexpr argument ... pass an int, float, bool or None`).

## Per-commit summary (32 args, int constexpr, int non-constexpr; ns/call)

`gap50`/`gap100` = share 50 %/100 % minus share 0 % (negative: constexprs are
cheaper). `load` = highest 1-min load average before the 3 processes.

| idx | commit | base | load | 0% rep | 50% rep | 100% rep | gap50 rep | gap100 rep | 0% alt | 50% alt | gap50 alt | str50 rep | str50 alt | dtype50 alt |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 67 | f87f970 | - | 3 | 118.2 | 123.9 | 124.8 | +5.7 | +6.6 | 118.7 | 123.7 | +5.0 | n/a | n/a | n/a |
| 68 | 7cda2f1 | f87f970 | 3 | 192.4 | 147.0 | 126.4 | -45.4 | -66.0 | 196.7 | 144.4 | -52.3 | n/a | n/a | n/a |
| 69 | 915a65f | 7cda2f1 | 3 | 215.1 | 161.5 | 130.6 | -53.6 | -84.5 | 208.7 | 163.5 | -45.2 | n/a | n/a | n/a |
| 70 | de5813d | 915a65f | 12 | 213.5 | 161.2 | 130.7 | -52.3 | -82.8 | 207.7 | 163.1 | -44.6 | n/a | n/a | n/a |
| 72 | 01eb03b | de5813d | 11 | 214.0 | 161.5 | 130.9 | -52.5 | -83.1 | 208.6 | 161.7 | -46.9 | n/a | n/a | n/a |
| 74 | 848dd22 | 01eb03b | 13 | 188.2 | 155.2 | 132.8 | -33.0 | -55.4 | 182.9 | 156.6 | -26.3 | n/a | n/a | n/a |
| 75 | 82ef454 | 848dd22 | 9 | 188.7 | 155.2 | 133.5 | -33.5 | -55.2 | 188.5 | 156.3 | -32.2 | n/a | n/a | n/a |
| 76 | f6079e5 | 82ef454 | 18 | 184.4 | 159.1 | 132.2 | -25.3 | -52.2 | 183.4 | 158.3 | -25.1 | n/a | n/a | n/a |
| 77 | be45d99 | f6079e5 | 30 | 183.2 | 155.9 | 132.2 | -27.3 | -51.0 | 181.0 | 163.3 | -17.7 | n/a | n/a | n/a |
| 78 | c7c1be5 | be45d99 | 16 | 117.7 | 107.5 | 112.7 | -10.2 | -5.0 | 109.3 | 108.2 | -1.1 | n/a | n/a | n/a |
| 79 | 4f0b13e | c7c1be5 | 21 | 111.7 | 111.5 | 107.4 | -0.2 | -4.3 | 109.0 | 109.5 | +0.5 | n/a | n/a | n/a |
| 80 | 69c24ba | 4f0b13e | 18 | 110.4 | 110.1 | 107.1 | -0.3 | -3.3 | 110.2 | 109.3 | -0.9 | n/a | n/a | n/a |
| 85 | 39d66a4 | c7c1be5 | 15 | 114.0 | 118.3 | 109.6 | +4.3 | -4.4 | 112.7 | 117.3 | +4.6 | n/a | n/a | n/a |
| 86 | 68ba1eb | 39d66a4 | 9 | 112.9 | 116.2 | 109.8 | +3.3 | -3.1 | 112.8 | 118.1 | +5.3 | n/a | n/a | n/a |
| 90 | 9243a11 | 69c24ba | 6 | 114.1 | 110.2 | 107.9 | -3.9 | -6.2 | 112.8 | 110.3 | -2.5 | n/a | n/a | n/a |
| 91 | 8619f45 | 68ba1eb | 5 | 117.8 | 113.4 | 111.7 | -4.4 | -6.1 | 117.8 | 115.3 | -2.5 | n/a | n/a | n/a |
| 92 | 1bf689b | 69c24ba | 4 | 116.3 | 110.1 | 108.7 | -6.2 | -7.6 | 113.1 | 112.2 | -0.9 | n/a | n/a | n/a |
| 93 | 34cbba2 | 8619f45 | 4 | 104.4 | 103.7 | 106.1 | -0.7 | +1.7 | 105.9 | 102.8 | -3.1 | n/a | n/a | n/a |
| 94 | c5295b1 | 34cbba2 | 4 | 103.8 | 100.3 | 106.1 | -3.5 | +2.3 | 102.6 | 100.7 | -1.9 | n/a | n/a | n/a |
| 95 | 7ad507d | 69c24ba | 3 | 108.5 | 89.4 | 69.1 | -19.1 | -39.4 | 113.5 | 114.2 | +0.7 | n/a | n/a | n/a |
| 96 | 9b92589 | c5295b1 | 13 | 116.8 | 81.9 | 68.8 | -34.9 | -48.0 | 109.4 | 106.2 | -3.2 | n/a | n/a | n/a |
| 97 | a46ea78 | 9b92589 | 12 | 104.6 | 81.6 | 68.9 | -23.0 | -35.7 | 109.5 | 106.0 | -3.5 | n/a | n/a | n/a |
| 98 | 1d19c0f | a46ea78 | 8 | 104.8 | 81.8 | 68.8 | -23.0 | -36.0 | 110.4 | 106.5 | -3.9 | n/a | n/a | n/a |
| 99 | e3c94b5 | 1d19c0f | 6 | 100.1 | 83.3 | 69.0 | -16.8 | -31.1 | 103.7 | 106.7 | +3.0 | n/a | n/a | n/a |
| 100 | 0601f8f | e3c94b5 | 25 | 99.5 | 83.3 | 69.5 | -16.2 | -30.0 | 104.8 | 105.6 | +0.8 | n/a | n/a | n/a |
| 101 | 97ef0ae | 0601f8f | 28 | 100.0 | 83.1 | 68.9 | -16.9 | -31.1 | 103.5 | 105.7 | +2.2 | n/a | n/a | n/a |
| 102 | 3a96100 | 97ef0ae | 25 | 99.5 | 83.1 | 69.1 | -16.4 | -30.4 | 103.5 | 107.0 | +3.5 | n/a | n/a | n/a |
| 103 | 13b4810 | 3a96100 | 15 | 99.6 | 83.5 | 68.8 | -16.1 | -30.8 | 103.7 | 106.2 | +2.5 | n/a | n/a | n/a |
| 104 | 26fc98e | 13b4810 | 8 | 99.4 | 83.3 | 69.5 | -16.1 | -29.9 | 103.8 | 105.8 | +2.0 | n/a | n/a | n/a |
| 105 | d38d6f8 | 26fc98e | 6 | 99.2 | 83.0 | 69.1 | -16.2 | -30.1 | 103.4 | 106.5 | +3.1 | n/a | n/a | n/a |
| 106 | 7c68c32 | d38d6f8 | 5 | 99.5 | 83.3 | 69.3 | -16.2 | -30.2 | 103.8 | 107.1 | +3.3 | n/a | n/a | n/a |
| 107 | b4e9f1f | 7c68c32 | 4 | 99.6 | 83.0 | 69.1 | -16.6 | -30.5 | 103.8 | 106.6 | +2.8 | n/a | n/a | n/a |
| 108 | 84fd60a | b4e9f1f | 4 | 105.5 | 82.8 | 69.2 | -22.7 | -36.3 | 110.5 | 106.7 | -3.8 | n/a | n/a | n/a |
| 109 | 9dacc19 | 84fd60a | 4 | 106.5 | 82.8 | 69.7 | -23.7 | -36.8 | 109.2 | 106.8 | -2.4 | n/a | n/a | n/a |
| 110 | be52e1c | 9dacc19 | 4 | 99.5 | 82.9 | 69.4 | -16.6 | -30.1 | 103.6 | 106.9 | +3.3 | n/a | n/a | n/a |
| 111 | bb25722 | be52e1c | 5 | 98.3 | 83.1 | 69.5 | -15.2 | -28.8 | 103.4 | 105.5 | +2.1 | n/a | n/a | n/a |
| 112 | b6ade5d | bb25722 | 5 | 98.4 | 83.2 | 69.4 | -15.2 | -29.0 | 102.8 | 106.8 | +4.0 | n/a | n/a | n/a |
| 113 | ad8c832 | b6ade5d | 5 | 98.6 | 82.9 | 69.8 | -15.7 | -28.8 | 103.1 | 105.0 | +1.9 | n/a | n/a | n/a |
| 114 | 39f2547 | ad8c832 | 5 | 98.2 | 82.8 | 69.5 | -15.4 | -28.7 | 103.3 | 106.2 | +2.9 | n/a | n/a | n/a |
| 115 | 5ad90c0 | 39f2547 | 5 | 105.3 | 84.5 | 69.6 | -20.8 | -35.7 | 108.8 | 107.1 | -1.7 | n/a | n/a | n/a |
| 116 | 53feab3 | 5ad90c0 | 5 | 97.2 | 82.9 | 69.3 | -14.3 | -27.9 | 103.0 | 106.6 | +3.6 | n/a | n/a | n/a |
| 117 | 8d634b6 | 53feab3 | 5 | 97.8 | 82.8 | 69.6 | -15.0 | -28.2 | 102.3 | 106.6 | +4.3 | n/a | n/a | n/a |
| 118 | a6292eb | 8d634b6 | 19 | 106.3 | 82.0 | 68.6 | -24.3 | -37.7 | 112.9 | 104.8 | -8.1 | 84.8 | 810.8 | 770.4 |
| 119 | 7223147 | a6292eb | 16 | 108.1 | 82.0 | 68.8 | -26.1 | -39.3 | 112.8 | 104.8 | -8.0 | 84.6 | 823.1 | 773.4 |
| 120 | 51f6cc5 | 7223147 | 54 | 108.3 | 82.1 | 68.8 | -26.2 | -39.5 | 112.3 | 105.1 | -7.2 | 84.9 | 811.9 | 770.8 |
| 121 | f4454e9 | 51f6cc5 | 55 | 107.4 | 82.1 | 68.8 | -25.3 | -38.6 | 111.2 | 104.9 | -6.3 | 84.5 | 816.8 | 776.4 |
| 122 | da89d89 | f4454e9 | 23 | 106.1 | 82.0 | 69.0 | -24.1 | -37.1 | 113.2 | 104.9 | -8.3 | 84.8 | 824.4 | 783.4 |
| 123 | 8d95e66 | da89d89 | 12 | 105.8 | 82.2 | 68.5 | -23.6 | -37.3 | 113.0 | 104.8 | -8.2 | 84.5 | 811.0 | 778.5 |
| 124 | 9868517 | 8d95e66 | 7 | 106.2 | 82.1 | 68.7 | -24.1 | -37.5 | 113.6 | 105.2 | -8.4 | 84.4 | 842.2 | 799.7 |
| 125 | 6e0d4bc | 9868517 | 3 | 107.6 | 82.3 | 68.5 | -25.3 | -39.1 | 112.6 | 105.0 | -7.6 | 84.6 | 818.6 | 783.1 |

4 and 16 args (share 0 / 50 / 100 %, repeat): `f87f970` 38.5 / 38.7 / 38.7 and
75.2 / 79.5 / 79.7; `c7c1be5` 36.1 / 34.7 / 34.8 and 65.0 / 65.4 / 66.0;
`6e0d4bc` 32.2 / 31.1 / 29.0 and 60.9 / 54.2 / 48.5.

### Constexpr versus non-constexpr gap

- `f87f970` (first measurable): constexprs cost more. 32 args, 50 % int is
  +5.7 ns over 0 %, 100 % is +6.6 ns.
- `7cda2f1`..`be45d99`: the gap reads -45 ns, but only because the
  non-constexpr path regressed (+74 ns, see below).
- `c7c1be5` (key decoding optimized): 50 % is -10 ns, 100 % -5 ns.
- `7ad507d`/`9b92589` (last-key slot): repeated keys with constexprs get much
  cheaper, 100 % drops from 106 to 69 ns; alternating keys lose 5-12 ns.
- `6e0d4bc` (now): repeat 107.6 / 82.3 / 68.5 ns, gap50 -25.3, gap100
  -39.1 ns; alternate gap50 -7.6, gap100 -0.7 ns.
- `str`/`dtype` constexprs (from `a6292eb`): repeat costs the same as `int`
  (32 args 50 %: 84.6 vs 82.3 ns), but alternating two values costs about
  800 ns/call (str 818.6, dtype 783.1 vs int 105.0). Each object slot memoizes
  one object; the other value falls to the Python intern table on every call.

## Flagged transitions

The sweep flagged 189 row transitions in 31 commit pairs. The interleaved
rerun confirmed 14 pairs; the rest are noise.

| base -> commit | confirmed / flagged rows | largest confirmed rows (rerun medians, ns) | hot-path code | verdict |
|---|---|---|---|---|
| `f87f970` -> `7cda2f1` Render ordinary argument annotations | 38/39 | 32 0 % tensor alt 107.5 -> 185.5 (+73 %); 32 0 % int rep 118.2 -> 189.0 (+60 %); 16 0 % tensor +26 %; 4 0 % +2 ns | `entry` 6059 -> 8116 insns | real; undone by `c7c1be5` |
| `7cda2f1` -> `915a65f` Pack typed constexpr annotations | 20/25 | 32 50 % int rep 140.7 -> 158.5 (+13 %); 32 0 % int rep +18 ns (+10 %) | `entry` changed | real; undone by `c7c1be5` |
| `01eb03b` -> `848dd22` Bind arguments into vectorcall launchers | 3/3 | 16 25 % int/tensor alt 74.9 -> 84.8 (+13 %); 32 25 % int rep +6 ns | `entry` changed | real; undone by `c7c1be5` |
| `c7c1be5` -> `4f0b13e` Optimize launcher checks | 5/7 | 16 0 % int rep 64.9 -> 68.2 (+5 %); 32 0 % tensor alt +4 ns | `entry`, grid check changed | real, small |
| `c7c1be5` -> `39d66a4` callable grid (branch) | 27/35 | 16 0 % int +5 ns (+8 %); 32 50 % int/tensor +5.5 ns; 4 0 % int +2.7 ns | `entry`, `spec_key`, `intj_call` changed | real |
| `69c24ba` -> `9243a11`, `69c24ba` -> `1bf689b` (torch_abi merge) | 2/7, 2/5 | 32 25 % int +4.5 ns (+4 %) | `entry`, `intj_bound_entry` changed | real, small |
| `68ba1eb` -> `8619f45` Return cached CompiledKernel | 2/3 | 32 0 % int rep 112.3 -> 117.1 (+4 %) | `intj_call` +14 insns | real, small |
| `69c24ba` -> `7ad507d`, `c5295b1` -> `9b92589` last-key slot | 10/11, 12/16 | alternate only: 32 100 % alt 106.2 -> 118.8 (+12 %); 16 100 % alt +6 ns | `intj_call` changed | real, by design (repeat rows gain up to -38 ns) |
| `b4e9f1f` -> `84fd60a` no atomics on the hot path | 2/3 | 32 0 % int rep 99.3 -> 105.3 (+6 %) | `intj_call` changed | real; undone by `be52e1c` |
| `39f2547` -> `5ad90c0` one fixed bound header | 3/4 | 32 0 % tensor alt 97.7 -> 108.7 (+11 %); 32 0 % int rep +5 ns | `intj_bound_entry` changed | real; 32 0 % int rep back to 97 at `53feab3` |
| `5ad90c0` -> `53feab3` lazy launchers | 1/2 | 32 25 % int alt 103.5 -> 109.5 (+6 %) | launcher path rewritten | real, small |
| `8d634b6` -> `a6292eb` value-keyed str/dtype constexprs | 2/2 | 32 0 % int rep 97.6 -> 107.8 (+10 %); alt +8.7 ns | `mix_32_0`: `intj_call`, `intj_bound_entry`, `spec_key` identical, moved +0x90 | placement; persists to `6e0d4bc` |
| `da89d89` -> `8d95e66` Pin the dynamic-value invariants | 6/6 | 32 100 % str rep 74.0 -> 82.6 (+12 %), dtype same; 32 25 % str/dtype tensor rep +8-10 ns | source change: a NULL check dropped from the miss-only `intj_object_id_slow`; in `intj_call` the per-slot memo compares are unchanged, the cold code between them shrinks and padding moves | placement; persists to `6e0d4bc` |

Noise (0 rows confirmed): `915a65f`->`de5813d`, `de5813d`->`01eb03b`,
`848dd22`->`82ef454`, `82ef454`->`f6079e5`, `f6079e5`->`be45d99`,
`8619f45`->`34cbba2`, `e3c94b5`->`0601f8f`, `0601f8f`->`97ef0ae`,
`be52e1c`->`bb25722`, `bb25722`->`b6ade5d` (16 25 % int +43 % in the sweep),
`a6292eb`->`7223147`, `7223147`->`51f6cc5`, `51f6cc5`->`f4454e9`
(16 25 % int/tensor +54 % in the sweep, load 55), `f4454e9`->`da89d89`,
`8d95e66`->`9868517`, `9868517`->`6e0d4bc`. Single-process outliers of
+25-60 ns on one row appear a few times per sweep; median of 3 absorbs most,
not all.

"real" means hot-path instructions changed; the size of each change was not
attributed further. All GPU paths are untested here (host-only).

## Raw output

Full raw files (every process, every row, per-batch samples, load average)
live outside the repository:

- `/tmp/intj-mix/raw/<idx>_<sha>.txt`: sweep, 3 rounds per measured commit.
- `/tmp/intj-mix/confirm/<cand>_vs_<base>.txt`: interleaved reruns.
- `/tmp/intj-mix/confirm_summary.txt`: every flagged row, sweep vs rerun.
- `/tmp/intj-mix/tables.md`: per-count tables (4/16/32 args x share x
  pattern, and 32-arg tensor/mixed) for every measured commit.
- `/tmp/intj-mix/insn_diff_summary.txt`: per-function instruction diffs for
  the confirmed pairs.

Representative blocks, one per range (round 0; 32-arg rows for older ranges,
all rows at HEAD). Columns: median, then the 9 batch means.

```
--- 068_7cda2f1 round 0 (32-arg rows)
=== round 0 exit 0 wall 34.0s load_before 3.10 5.83 7.51 2/15118 186206 load_after 2.73 5.44 7.32 2/15132 186580
count share ckind  nkind   pattern   ns/call  samples
   32     0     -    int    repeat     192.4  204.0 201.2 200.4 192.4 187.1 190.3 194.2 191.0 189.6
   32     0     -    int alternate     190.9  201.5 192.7 190.9 187.7 190.2 191.3 189.3 190.8 191.0
   32     0     - tensor    repeat     186.8  201.3 192.7 183.5 186.8 186.2 187.9 186.9 180.8 180.6
   32     0     - tensor alternate     189.9  200.0 191.0 191.7 192.4 188.5 189.4 189.9 186.5 186.1
   32     0     -  mixed    repeat     163.8  167.1 165.8 165.4 163.8 163.6 163.7 163.6 163.7 164.0
   32     0     -  mixed alternate     156.5  173.4 161.0 158.0 156.5 157.4 155.9 155.9 156.2 156.2
   32    25   int    int    repeat     182.5  181.6 186.5 187.5 182.5 184.5 181.0 214.0 175.3 174.9
   32    25   int    int alternate     167.6  168.6 167.5 167.3 167.7 168.5 167.6 166.4 166.3 168.2
   32    25   int tensor    repeat     184.5  189.8 185.9 184.5 184.9 185.0 184.2 181.9 182.7 182.2
   32    25   int tensor alternate     188.3  192.6 194.8 191.3 188.6 188.3 186.9 185.5 181.8 179.5
   32    25   int  mixed    repeat     156.9  157.2 157.0 156.8 156.3 157.1 157.0 156.9 156.0 155.6
   32    25   int  mixed alternate     157.2  160.1 157.6 156.8 156.8 157.2 158.4 157.9 157.2 156.1
   32    50   int    int    repeat     148.4  150.8 151.9 148.6 152.0 146.2 146.2 144.3 148.4 145.5
   32    50   int    int alternate     145.6  152.1 145.9 147.5 145.6 146.2 144.0 143.9 145.6 145.2
   32    50   int tensor    repeat     152.5  154.1 153.4 152.5 153.1 152.8 152.1 151.9 151.9 152.3
   32    50   int tensor alternate     153.7  155.8 153.8 153.7 153.7 154.8 150.8 155.8 152.9 153.5
   32    50   int  mixed    repeat     139.3  140.4 141.2 139.8 140.1 139.0 138.1 139.3 138.2 138.9
   32    50   int  mixed alternate     137.9  137.9 138.5 138.6 138.7 138.1 137.4 136.9 136.7 137.0
   32   100   int      -    repeat     126.8  127.5 126.3 126.7 126.9 127.4 126.9 126.6 126.7 126.8
   32   100   int      - alternate     126.8  127.0 126.1 126.8 126.1 126.6 127.1 126.8 127.1 126.7
--- 080_69c24ba round 0 (32-arg rows)
=== round 0 exit 0 wall 36.2s load_before 18.43 13.73 10.30 7/15227 195237 load_after 18.40 14.23 10.59 7/15228 195435
count share ckind  nkind   pattern   ns/call  samples
   32     0     -    int    repeat     110.2  111.8 110.2 111.3 107.7 109.9 110.2 112.2 107.9 108.5
   32     0     -    int alternate     110.2  114.9 110.2 110.0 110.2 111.3 111.6 115.1 109.8 108.3
   32     0     - tensor    repeat     112.3  119.5 118.0 114.7 112.5 109.4 110.6 112.3 111.5 110.8
   32     0     - tensor alternate     112.5  122.7 117.4 112.6 112.5 112.3 112.6 111.1 110.9 110.3
   32     0     -  mixed    repeat      98.1  98.1 98.2 98.1 98.6 98.2 97.9 98.2 98.1 98.1
   32     0     -  mixed alternate      97.4  96.9 97.5 97.4 97.4 97.1 97.4 97.5 97.2 97.1
   32    25   int    int    repeat     108.0  109.6 111.4 107.4 108.0 111.9 107.3 109.0 107.5 107.9
   32    25   int    int alternate     107.2  111.1 106.0 107.5 107.2 107.0 107.3 107.2 107.2 105.8
   32    25   int tensor    repeat     103.2  102.4 110.3 101.8 109.1 103.2 110.2 101.5 110.2 102.0
   32    25   int tensor alternate     104.0  101.9 110.0 104.0 110.7 102.3 109.3 101.6 108.6 102.0
   32    25   int  mixed    repeat      95.5  95.5 95.6 95.3 95.7 95.5 96.0 95.3 95.7 95.4
   32    25   int  mixed alternate      95.8  95.5 96.0 96.0 95.9 95.6 95.8 95.6 96.5 95.8
   32    50   int    int    repeat     110.1  112.5 108.0 110.1 110.7 111.4 106.4 109.5 110.2 106.8
   32    50   int    int alternate     110.6  112.9 109.0 111.2 106.6 110.6 112.7 112.8 109.0 107.8
   32    50   int tensor    repeat     103.6  103.9 103.6 103.6 103.6 103.3 103.5 103.5 103.9 103.5
   32    50   int tensor alternate     103.1  103.1 103.2 102.8 103.7 103.1 103.3 103.0 103.2 102.9
   32    50   int  mixed    repeat      98.4  98.6 98.7 98.2 98.4 98.3 98.6 98.1 98.4 98.6
   32    50   int  mixed alternate      99.3  99.5 99.4 99.2 99.8 99.4 99.3 99.2 99.2 99.3
   32   100   int      -    repeat     107.1  107.0 107.0 108.0 107.3 107.9 108.0 107.0 107.1 107.1
   32   100   int      - alternate     107.0  107.5 107.0 106.8 107.0 107.3 107.8 107.4 106.8 107.0
--- 114_39f2547 round 0 (32-arg rows)
=== round 0 exit 0 wall 34.6s load_before 4.75 5.90 7.63 6/15176 207454 load_after 4.69 5.77 7.52 4/15169 207829
count share ckind  nkind   pattern   ns/call  samples
   32     0     -    int    repeat      99.8  99.0 98.5 100.4 101.4 100.1 99.8 100.0 98.7 99.4
   32     0     -    int alternate     103.5  103.5 102.8 104.9 105.8 103.5 103.2 106.7 103.9 103.3
   32     0     - tensor    repeat     102.3  109.2 109.8 102.3 108.0 97.0 104.5 95.4 102.2 99.3
   32     0     - tensor alternate     105.9  108.6 110.0 103.9 105.9 99.8 107.1 99.3 106.0 97.4
   32     0     -  mixed    repeat      88.9  89.2 88.9 88.6 89.3 88.8 89.2 88.9 89.3 88.6
   32     0     -  mixed alternate      93.7  93.7 94.0 93.7 93.6 93.3 94.4 93.1 93.7 93.7
   32    25   int    int    repeat      90.6  90.7 90.6 90.8 90.4 91.3 90.6 90.2 96.3 90.6
   32    25   int    int alternate     103.4  103.4 107.1 104.7 103.2 102.9 103.3 103.1 104.0 108.9
   32    25   int tensor    repeat      93.8  93.8 98.4 91.3 98.7 91.0 98.5 91.5 98.9 91.9
   32    25   int tensor alternate     103.9  103.9 108.0 103.5 107.8 103.5 108.4 103.4 109.0 103.4
   32    25   int  mixed    repeat      83.4  83.2 83.4 83.2 84.2 83.2 83.4 83.2 83.4 83.4
   32    25   int  mixed alternate      95.7  95.6 95.7 95.5 96.0 95.8 96.1 95.4 95.8 95.7
   32    50   int    int    repeat      82.8  82.8 83.1 82.9 82.5 82.3 84.0 82.7 83.0 82.8
   32    50   int    int alternate     106.9  108.4 106.9 107.6 106.7 106.5 107.1 106.7 108.1 106.5
   32    50   int tensor    repeat      83.3  83.5 83.3 82.7 83.4 83.3 83.2 83.0 83.3 82.5
   32    50   int tensor alternate     106.2  106.1 106.3 106.3 106.1 105.9 106.2 106.0 106.2 106.2
   32    50   int  mixed    repeat      79.2  79.1 79.2 79.1 79.6 79.9 79.4 78.9 79.5 79.2
   32    50   int  mixed alternate     102.0  101.7 102.2 101.8 102.2 101.9 102.0 101.9 102.1 102.0
   32   100   int      -    repeat      69.5  69.2 69.7 69.1 70.1 69.3 69.5 69.6 70.7 69.1
   32   100   int      - alternate     112.6  112.6 112.7 112.6 112.7 112.6 113.1 112.4 112.7 112.5
--- 125_6e0d4bc round 0 (all rows)
=== round 0 exit 0 wall 48.0s load_before 2.22 8.59 10.54 2/15118 215438 load_after 2.92 7.79 10.16 8/15118 215830
intj 0.1.0 from /tmp/intj-mix-wt/intj
python 3.12.3; triton 3.8.0; torch 2.14.0+rocm7.2
mode=mix; host-only; 40000 calls × 9 batches; median ns/call
count share ckind  nkind   pattern   ns/call  samples
    4     0     -    int    repeat      32.2  32.4 32.2 32.2 32.2 32.1 32.2 32.5 32.2 31.8
    4     0     -    int alternate      34.0  33.7 33.9 33.9 33.8 34.1 34.0 34.0 34.6 34.2
    4     0     - tensor    repeat      30.5  30.7 30.7 30.5 30.5 30.5 30.7 30.4 30.5 30.6
    4     0     - tensor alternate      32.7  32.8 32.5 32.6 32.8 32.9 32.7 32.7 32.7 32.6
    4     0     -  mixed    repeat      30.2  30.5 30.4 30.2 30.1 30.1 30.2 30.2 30.3 30.4
    4     0     -  mixed alternate      32.2  32.3 32.3 32.0 32.1 32.2 32.2 32.2 32.1 32.3
    4    25   int    int    repeat      31.5  31.5 31.6 31.5 32.2 31.4 31.8 31.5 31.5 32.0
    4    25   int    int alternate      33.3  33.9 33.2 33.1 33.3 33.4 33.6 33.2 33.2 33.3
    4    25   int tensor    repeat      30.2  30.2 30.1 30.1 30.0 30.9 30.3 30.2 30.0 30.3
    4    25   int tensor alternate      32.6  32.8 32.6 32.6 32.6 32.5 32.7 32.7 32.5 32.4
    4    25   int  mixed    repeat      29.7  29.8 29.7 29.8 29.8 29.7 29.7 29.9 29.7 29.7
    4    25   int  mixed alternate      31.8  31.8 31.8 31.9 31.8 31.7 31.8 31.9 31.7 31.9
    4    25   str    int    repeat      31.0  31.9 31.0 31.1 31.0 31.0 30.9 31.0 31.2 31.0
    4    25   str    int alternate     670.3  669.2 672.9 671.2 671.9 669.5 673.5 670.3 668.9 668.9
    4    25   str tensor    repeat      30.1  30.2 30.1 30.4 30.1 30.2 30.2 30.1 30.1 30.1
    4    25   str tensor alternate     669.8  669.2 667.8 672.3 671.6 673.3 667.8 669.8 669.1 672.7
    4    25   str  mixed    repeat      29.8  29.8 29.8 29.9 29.7 29.8 29.8 29.7 29.8 29.8
    4    25   str  mixed alternate     675.4  675.4 673.9 679.4 676.1 678.5 674.9 670.5 675.5 673.3
    4    25 dtype    int    repeat      31.1  30.9 31.1 31.1 31.1 31.4 31.1 31.0 31.0 31.1
    4    25 dtype    int alternate     631.1  630.0 633.7 630.1 634.7 632.4 630.1 632.0 631.1 628.6
    4    25 dtype tensor    repeat      30.2  30.1 30.3 30.2 30.1 30.2 30.2 30.1 30.2 30.1
    4    25 dtype tensor alternate     629.8  629.8 629.3 633.1 629.3 633.5 629.1 629.0 632.2 633.0
    4    25 dtype  mixed    repeat      29.8  29.9 29.8 29.9 29.8 29.9 30.0 29.8 29.7 29.8
    4    25 dtype  mixed alternate     627.4  624.2 628.8 626.6 626.2 629.5 627.4 626.8 628.7 629.0
    4    50   int    int    repeat      31.0  31.0 30.8 31.3 30.8 30.9 32.2 31.1 31.1 31.0
    4    50   int    int alternate      33.6  33.6 33.7 33.6 33.5 33.6 33.8 33.7 33.6 34.2
    4    50   int tensor    repeat      30.1  30.0 30.0 30.2 30.1 30.0 30.1 30.3 30.0 30.1
    4    50   int tensor alternate      33.4  33.6 33.4 33.4 33.4 33.5 33.3 33.2 33.5 33.3
    4    50   int  mixed    repeat      30.3  30.3 30.4 30.3 30.3 30.3 30.3 30.3 30.5 30.2
    4    50   int  mixed alternate      33.9  33.9 34.0 33.9 33.9 34.1 34.0 33.8 34.0 33.9
    4    50   str    int    repeat      31.1  31.2 31.0 30.9 31.2 31.1 31.2 31.4 31.0 30.9
    4    50   str    int alternate     675.3  674.4 672.7 676.0 677.5 675.3 676.1 674.1 675.5 673.7
    4    50   str tensor    repeat      30.3  30.5 30.3 30.3 30.4 30.2 30.4 30.2 30.3 30.5
    4    50   str tensor alternate     675.7  679.8 674.9 679.7 675.7 675.7 679.0 672.8 671.1 677.8
    4    50   str  mixed    repeat      30.6  30.6 30.7 30.6 30.6 30.5 30.6 30.6 30.5 30.4
    4    50   str  mixed alternate     668.4  670.2 668.4 670.4 668.1 668.0 670.4 671.9 667.8 665.4
    4    50 dtype    int    repeat      31.0  32.0 31.0 31.0 31.6 30.9 31.0 30.9 31.0 30.9
    4    50 dtype    int alternate     628.8  627.4 631.0 626.5 629.4 628.8 628.6 630.6 632.3 626.8
    4    50 dtype tensor    repeat      30.3  30.3 30.3 30.3 30.2 30.2 30.2 30.5 30.3 30.3
    4    50 dtype tensor alternate     619.9  620.0 619.2 620.1 616.6 620.6 617.5 619.9 616.2 622.0
    4    50 dtype  mixed    repeat      30.6  30.6 30.6 30.6 30.4 30.7 30.6 30.5 30.5 30.5
    4    50 dtype  mixed alternate     627.3  627.3 627.2 628.2 632.2 627.9 628.5 624.1 626.6 625.3
    4   100   int      -    repeat      29.0  29.1 29.1 29.0 29.1 28.8 28.9 29.0 28.9 29.2
    4   100   int      - alternate      34.0  34.2 33.9 34.0 34.1 33.9 34.2 34.0 35.7 33.8
    4   100   str      -    repeat      29.4  29.4 29.3 29.3 29.3 29.4 29.4 29.4 29.4 29.3
    4   100   str      - alternate     682.9  684.3 682.0 683.8 682.9 682.3 682.5 681.7 683.3 685.9
    4   100 dtype      -    repeat      29.3  29.4 29.4 29.3 29.3 29.3 29.3 29.2 29.3 29.2
    4   100 dtype      - alternate     640.0  640.0 640.0 637.6 639.4 638.6 639.8 644.4 641.8 643.0
   16     0     -    int    repeat      60.9  61.0 60.8 60.8 61.1 60.7 60.8 60.9 61.0 60.9
   16     0     -    int alternate      64.1  64.2 64.2 64.0 64.8 64.1 64.1 64.1 64.3 64.0
   16     0     - tensor    repeat      57.0  57.1 56.9 56.8 57.4 57.1 57.1 56.7 57.0 56.9
   16     0     - tensor alternate      61.0  61.3 61.1 60.9 60.9 61.2 61.0 60.8 61.1 60.9
   16     0     -  mixed    repeat      55.4  55.3 55.7 55.6 56.9 55.4 55.4 55.2 55.3 55.2
   16     0     -  mixed alternate      59.6  59.7 59.7 59.5 59.6 59.7 59.5 59.8 59.5 59.4
   16    25   int    int    repeat      55.7  55.7 55.7 55.7 55.4 55.9 55.4 55.4 55.5 56.7
   16    25   int    int alternate      62.2  68.1 62.3 62.2 62.2 62.1 62.4 62.1 62.4 62.1
   16    25   int tensor    repeat      54.6  54.6 54.6 54.5 54.5 54.7 54.5 54.5 54.6 54.7
   16    25   int tensor alternate      61.9  62.3 61.9 61.9 61.9 61.9 62.0 61.9 62.1 62.3
   16    25   int  mixed    repeat      51.8  51.8 51.9 51.7 51.8 52.1 51.6 51.8 51.7 51.7
   16    25   int  mixed alternate      58.1  58.1 58.4 58.2 58.0 58.2 58.0 57.9 58.3 58.1
   16    25   str    int    repeat      56.3  56.2 56.1 56.3 56.4 56.2 56.3 56.3 56.0 56.3
   16    25   str    int alternate     736.4  736.4 740.9 735.3 732.8 736.4 739.6 735.4 735.5 737.1
   16    25   str tensor    repeat      54.8  55.0 54.8 54.8 54.7 54.8 54.9 54.8 54.8 54.7
   16    25   str tensor alternate     736.9  733.4 737.6 737.9 736.2 736.9 735.6 739.4 738.5 734.7
   16    25   str  mixed    repeat      52.5  52.5 52.5 52.5 52.5 52.5 52.4 52.4 52.4 52.5
   16    25   str  mixed alternate     728.0  728.0 726.3 729.6 727.2 729.2 726.2 726.6 730.5 728.5
   16    25 dtype    int    repeat      56.2  56.1 56.2 56.6 56.2 56.2 56.1 56.2 56.2 56.2
   16    25 dtype    int alternate     688.7  686.1 690.2 687.1 690.1 688.5 687.4 689.5 688.7 688.8
   16    25 dtype tensor    repeat      55.0  54.9 55.0 54.9 54.9 55.7 55.0 54.9 55.3 55.4
   16    25 dtype tensor alternate     698.5  699.2 698.5 697.2 700.7 697.7 697.0 699.6 699.0 696.6
   16    25 dtype  mixed    repeat      52.4  52.4 52.4 52.4 52.4 52.4 52.5 52.3 52.4 52.5
   16    25 dtype  mixed alternate     696.0  692.9 697.7 694.3 701.1 696.9 693.5 696.3 695.5 696.0
   16    50   int    int    repeat      54.2  54.2 54.4 54.0 54.0 54.7 54.2 54.3 54.5 54.2
   16    50   int    int alternate      65.4  65.3 65.5 65.5 65.4 65.5 67.0 65.4 65.2 65.3
   16    50   int tensor    repeat      54.8  54.8 54.8 54.7 54.9 54.8 54.8 54.9 54.7 54.8
   16    50   int tensor alternate      65.8  65.9 65.8 65.7 65.8 65.8 65.7 65.7 65.7 65.8
   16    50   int  mixed    repeat      52.8  52.7 52.8 52.9 53.1 52.8 52.8 52.8 52.8 52.9
   16    50   int  mixed alternate      63.6  63.6 63.6 63.6 63.5 63.6 63.5 63.7 63.6 63.7
   16    50   str    int    repeat      54.5  54.5 54.6 54.5 54.5 54.4 54.3 54.4 54.5 54.7
   16    50   str    int alternate     730.8  732.0 731.3 730.8 729.5 732.2 730.3 731.7 726.4 725.7
   16    50   str tensor    repeat      55.4  55.4 55.4 55.5 55.5 55.6 55.4 55.3 55.4 55.4
   16    50   str tensor alternate     726.1  725.6 722.2 726.5 725.7 727.4 725.0 726.1 726.6 730.1
   16    50   str  mixed    repeat      53.4  53.4 53.4 53.4 53.3 53.5 53.4 53.3 53.5 53.4
   16    50   str  mixed alternate     723.7  723.7 722.4 722.6 722.0 724.8 727.4 724.7 725.7 722.2
   16    50 dtype    int    repeat      54.4  54.4 54.5 54.6 54.4 54.4 54.6 54.3 54.4 54.4
   16    50 dtype    int alternate     680.9  677.7 679.1 686.4 680.9 680.2 680.6 681.0 683.4 685.7
   16    50 dtype tensor    repeat      55.5  55.4 55.7 55.4 55.4 55.5 55.4 55.5 55.6 55.5
   16    50 dtype tensor alternate     686.7  686.7 692.4 684.0 687.7 684.3 689.9 683.2 688.2 684.9
   16    50 dtype  mixed    repeat      53.4  53.8 53.3 53.5 53.3 53.6 53.4 53.4 53.3 53.5
   16    50 dtype  mixed alternate     681.2  680.3 680.3 682.3 681.2 680.8 681.2 678.3 683.0 681.6
   16   100   int      -    repeat      48.3  48.3 48.3 48.0 48.2 48.8 48.1 48.4 48.4 48.1
   16   100   int      - alternate      70.9  70.8 71.1 71.0 70.9 70.9 71.0 70.9 70.9 71.0
   16   100   str      -    repeat      49.3  49.8 49.1 49.3 49.0 49.1 49.0 50.0 49.5 49.3
   16   100   str      - alternate     739.9  740.5 740.0 738.3 738.7 742.6 741.6 739.1 738.1 739.9
   16   100 dtype      -    repeat      49.7  50.2 48.8 50.1 49.2 49.7 49.6 50.7 48.7 50.0
   16   100 dtype      - alternate     694.3  694.4 697.9 695.1 694.1 692.7 692.8 694.3 693.2 694.6
   32     0     -    int    repeat     109.2  109.8 110.8 106.6 109.4 108.2 104.1 108.0 109.2 109.3
   32     0     -    int alternate     112.6  113.9 115.0 113.2 111.3 112.6 112.9 107.8 109.5 107.5
   32     0     - tensor    repeat     107.5  115.4 109.3 110.2 107.5 107.5 106.3 107.0 107.9 106.9
   32     0     - tensor alternate     111.1  119.1 115.2 114.1 113.2 111.1 110.6 110.9 110.4 109.4
   32     0     -  mixed    repeat      88.1  88.2 88.0 88.0 88.5 88.1 88.1 88.0 88.2 88.2
   32     0     -  mixed alternate      93.6  93.6 93.7 93.8 93.8 93.4 93.6 93.5 93.6 93.6
   32    25   int    int    repeat      88.6  88.6 88.2 90.0 88.6 89.4 89.0 88.2 88.5 88.5
   32    25   int    int alternate     101.8  101.9 101.8 101.6 103.3 103.8 101.6 101.5 101.8 103.4
   32    25   int tensor    repeat      90.2  89.5 99.4 89.3 96.2 90.2 96.2 89.4 95.9 89.4
   32    25   int tensor alternate     102.4  102.2 106.8 102.3 106.4 102.4 106.7 102.0 106.1 101.8
   32    25   int  mixed    repeat      82.2  82.3 82.2 81.8 82.2 82.0 82.0 82.0 82.3 82.9
   32    25   int  mixed alternate      95.4  95.2 95.8 95.1 95.8 95.8 95.3 95.0 97.0 95.4
   32    25   str    int    repeat      89.6  89.6 89.4 89.5 90.3 91.4 90.1 90.2 89.4 89.6
   32    25   str    int alternate     817.6  816.2 823.7 817.6 825.6 813.1 824.8 814.6 825.1 815.3
   32    25   str tensor    repeat     104.9  104.9 107.3 104.0 108.0 104.2 109.9 102.1 109.3 102.8
   32    25   str tensor alternate     849.3  849.3 856.3 846.0 857.7 844.8 854.9 846.6 860.1 844.7
   32    25   str  mixed    repeat      84.1  83.5 90.4 84.1 90.7 83.0 90.2 83.8 90.4 83.1
   32    25   str  mixed alternate     819.8  819.8 821.7 818.8 823.6 819.5 821.1 818.2 824.8 817.4
   32    25 dtype    int    repeat      90.2  90.2 91.4 89.8 90.5 90.4 90.1 89.3 89.7 90.6
   32    25 dtype    int alternate     774.9  774.9 783.7 774.2 782.7 774.4 778.1 770.0 781.9 770.7
   32    25 dtype tensor    repeat     106.0  106.0 108.7 103.7 109.0 101.7 108.5 104.9 110.2 105.8
   32    25 dtype tensor alternate     802.4  800.1 812.4 801.3 813.6 800.5 811.0 798.7 811.9 802.4
   32    25 dtype  mixed    repeat      84.2  83.5 90.3 84.0 90.4 84.2 90.4 83.3 90.6 83.4
   32    25 dtype  mixed alternate     775.7  775.7 777.0 773.7 780.7 772.1 781.7 774.3 779.7 773.3
   32    50   int    int    repeat      82.3  82.3 82.4 82.3 82.2 82.2 83.9 82.2 82.5 82.0
   32    50   int    int alternate     105.9  105.8 106.0 105.9 105.9 107.9 105.9 105.9 106.0 105.8
   32    50   int tensor    repeat      83.1  82.9 83.3 82.8 83.1 83.1 83.2 82.8 83.3 83.1
   32    50   int tensor alternate     106.8  107.0 107.0 106.7 107.0 106.7 107.1 106.6 106.8 106.5
   32    50   int  mixed    repeat      79.2  78.8 79.2 78.8 79.3 79.4 79.5 79.0 79.3 79.1
   32    50   int  mixed alternate     101.6  101.7 101.9 101.5 101.6 101.5 102.1 101.7 101.5 101.6
   32    50   str    int    repeat      84.4  84.3 84.7 84.3 84.4 84.2 84.7 84.4 84.6 84.1
   32    50   str    int alternate     837.5  837.5 847.7 833.3 843.0 834.5 844.7 836.8 849.0 834.4
   32    50   str tensor    repeat      96.2  96.2 101.7 91.2 100.9 94.0 98.9 91.5 99.7 91.4
   32    50   str tensor alternate     855.1  853.0 861.4 855.1 860.6 854.9 859.4 851.9 865.6 853.6
   32    50   str  mixed    repeat      81.1  80.8 87.6 80.9 87.2 81.0 87.4 81.1 87.6 80.9
   32    50   str  mixed alternate     837.3  832.6 840.4 833.6 843.2 831.7 839.8 837.3 840.9 832.3
   32    50 dtype    int    repeat      84.4  84.3 84.6 84.3 84.4 84.5 84.4 84.2 84.5 84.5
   32    50 dtype    int alternate     789.8  786.5 797.0 785.3 796.3 786.2 796.7 783.8 792.8 789.8
   32    50 dtype tensor    repeat      94.2  94.2 99.5 91.1 100.0 91.9 100.0 91.4 99.3 91.2
   32    50 dtype tensor alternate     805.3  802.3 814.0 805.3 812.2 804.1 814.1 803.6 814.1 805.1
   32    50 dtype  mixed    repeat      81.2  81.2 87.6 81.0 87.3 80.9 87.6 81.1 87.4 80.9
   32    50 dtype  mixed alternate     785.5  784.6 793.5 782.9 792.7 785.5 792.5 784.4 789.0 781.8
   32   100   int      -    repeat      68.5  68.5 68.3 68.6 68.6 68.3 68.5 69.1 68.9 68.2
   32   100   int      - alternate     112.1  111.9 112.0 112.2 112.0 112.1 112.4 112.1 112.2 111.9
   32   100   str      -    repeat      82.7  87.8 83.2 82.5 83.0 82.9 82.4 82.5 82.3 82.7
   32   100   str      - alternate     811.7  812.5 810.1 813.5 808.9 811.7 807.8 811.5 814.7 812.9
   32   100 dtype      -    repeat      81.6  83.4 82.2 81.6 81.7 81.5 81.5 81.4 81.5 81.7
   32   100 dtype      - alternate     770.7  772.8 774.2 769.0 767.6 771.4 767.9 770.7 772.9 770.4
```

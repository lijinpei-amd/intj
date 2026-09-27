# Lazy, Per-Call-Variant `make_launcher` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `make_launcher` returns a launcher without touching the GPU, builds it on the first call, and keys compile options, Triton knobs, and `str` / `tl.dtype` / JIT-function constexprs per call, so one decorated handle serves every variant of a launch site.

**Architecture:**
- Every launcher becomes a real `PyCFunction` whose `self` is a fixed-layout bound object allocated by a small generic C extension, `_intj_lazy`. Its `PyMethodDef` lives inside that object and starts on a build shim.
- The first call renders, compiles and loads the module as today, fills the bound object in place through the module's `init_bound`, and swaps `ml_meth` to the rendered entry. CPython reads `m_ml->ml_meth` on every call, so references taken before the build reach the entry too.
- Dynamic values sit in the level-0 key: scalars as a kind byte plus 8 value bytes, and objects as an id from the launcher's own intern table, memoized per argument slot by pointer.
- All mutable state lives in the launcher: its cache, memos, intern table, rwlock, key self-check and miss callback. Module state is read-only after load; the module keeps only a Python compile function with a compile cache, so sibling launchers compile a key once.

**Tech Stack:** CPython 3.8–3.14 (plus 3.13t/3.14t) C API, Jinja2-rendered C/C++ extension, Triton ≥3.7 (`triton.knobs`, `parse_options`, `Autotuner`), pthread rwlock on free-threaded builds.

**Spec:** `docs/superpowers/specs/2026-09-27-lazy-dynamic-launchers-design.md`

## Global Constraints

- Every refusal raises `intj.launcher.UnsupportedKernel`. Raise it at `make_launcher` whenever the check needs no GPU. Never fall back to `JITFunction` silently.
- Invariant: same intj key ⟹ same Triton specialization, same compile options, same knob values. `test_spec_key_is_never_coarser_than_triton` and the tuned invariant test must keep passing. Each new keyed value gets a mutation check that makes its test fail.
- `RenderContext`, `ModuleKey`, `TuningRender` and the new `DynamicSlot` are frozen value types with tuple fields only. They must hash and JSON-serialize (`test_value_types_compare_hash_and_serialize`).
- Rendered modules **and `_intj_lazy`** are loaded by hand (`spec_from_file_location` + `exec_module`) and never land in `sys.modules`. Both are multi-phase (`PyModuleDef_Init`).
- Backends are data: the template branches on `Backend` fields, never on a backend name.
- Anything that reaches the network (`_provision`) stays at `make_launcher`, never on a call. It is GPU-free, so laziness does not move it.
- `intj/` is pyright-strict. `tests/` and `benchmarks/` start with `# pyright: standard`. Private triton names get `# pyright: ignore[reportPrivateUsage]` plus a reason.
- Python ≥3.8 syntax in `intj/`: no `match`, no runtime `X | Y` types. `intj/lazy.py` and `intj/runtime/*` must import and build on 3.8 without triton, because `tests/test_runtime.py` runs them on the whole matrix.
- Every CPython internal read goes in `intj/python_intf/cpython_abi.h`. The swap writes only our own `PyMethodDef` (a public struct). The CPython *behaviour* it relies on is recorded in `intj/python_intf/AGENTS.md` and pinned by `tests/test_runtime.py`.
- CUDA is compile-checked only. Say "untested" in docs.
- Every commit message ends with `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`.
- Commands (venv via uv; `/tmp/gb2/bin/python` only for benchmarks):

```
V=/mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/venv
PYTHONPATH=$PWD VIRTUAL_ENV=$V uv run --active --no-sync python -m pytest tests -q
PYTHONPATH=$PWD VIRTUAL_ENV=$V uv run --active --no-sync pyright            # 0 errors
bash tests/run_python_matrix.sh              # test_runtime.py on 3.8 … 3.14t
bash tests/run_python_matrix.sh 3.13t 3.14t  # the free-threaded pair
```

### CPython facts, checked 2026-09-27

A throwaway probe built a `METH_FASTCALL` `PyCFunction` over a `PyMethodDef` embedded in its `self` object. It swapped `ml_meth` after 100 calls. It ran on 3.8.20, 3.9.25, 3.10.20, 3.11.15, 3.12.3, 3.13.13, 3.13.13t, 3.14.4 and 3.14.4t:

- **Swap takes effect for old references: true everywhere.** Call 101 through a reference taken at creation ran the new function, both before and after the call site specialized. `PyCFunction_GET_FUNCTION` is `func->m_ml->ml_meth` in every header from 3.8 to 3.14t. `cfunction_vectorcall_FASTCALL` and the specialized opcodes read it at execution time.
- **Embedded/heap `PyMethodDef` lifetime: safe.** `meth_dealloc` never reads `m_ml` after dropping `m_self` (3.9+ even documents that `m_ml` may be kept alive by `m_self`). The probe scribbled `0xAB` over the def in `self`'s dealloc across 1000 create/call/`__doc__`/`repr`/free cycles and stayed clean.
- **"`CALL_BUILTIN_FAST` specialization still applies": false on 3.13t and 3.8–3.10.** 3.13t runs no specializing interpreter at all, so no builtin call is specialized, with or without intj. 3.8–3.10 have no specializing interpreter. The opcode is also named per version: `PRECALL_NO_KW_BUILTIN_FAST` (3.11), `CALL_NO_KW_BUILTIN_FAST` (3.12), `CALL_BUILTIN_FAST` (3.13, 3.14, 3.14t). *Fallback:* none is needed, because the vectorcall path reads `ml_meth` too. The specialization test accepts any of the three names and skips 3.8–3.10 and 3.13t.
- **Free-threaded publication.** CPython reads `ml_meth` with a plain load. The shim stores it with release order after `init_bound` publishes `state` last. That is enough on x86-64 (TSO), the only host intj supports today (`TODO.md`: non-x86-64 hosts). ARM64 free-threaded builds are an open risk, noted in `TODO.md` by Task 5.

### Deliberate deviations from the spec (each one is a correctness or cost fix)

1. **All mutable state lives in the launcher; module state is read-only after load** (spec amended, "Uniform shape").
   - **Each launcher (bound object) owns:** its level-0 cache and memo, its intern table and slot memos, its own `intj_rwlock` (cold part of the header), its `seen` key self-check map, and its miss callback (`compile_cb`, which is the tuned callback for tuned launchers).
   - **The module holds, read-only after load:** torch types, the ABI layout and dtype table, and the driver symbols. It also holds the Python compile function, as the module object's `_intj_compile` attribute. That function keeps a per-module compile cache behind its own `threading.Lock`, keyed on `(CompilerInput, canonical options hash, declared knob values, device)`, so a sibling launcher's first miss builds only its C record.
   - **Removed:** `module.entry`, `st->cache`, `st->compile_cb`, `set_compile_callback`, and the module rwlock. No module lock remains. `set_torch_version` runs once, under `_LOAD_LOCK`, before the module is reachable. The callback is passed per launcher to `init_bound`. Tests replace the compile function with `intj.launcher.override_compile(module, fn)`, which each launcher's callback looks up at miss time.
   - **`module.spec_key(launcher, device, *args)`** decodes with the launcher's intern table. It is the level-0 half of `key_chain`, so object values work.
   - **The "one key maps to two compiler inputs" check** runs per launcher again, on every call, object values included.
   - **Memory:** each launcher carries a 16-slot map (about 0.5–0.8 KB), a small dict, and a lock (56 bytes on free-threaded builds).
2. **Free-threaded memo reads take the read side of the launcher's own rwlock**, the lock that also guards its cache and table. A `(object, id)` pair cannot be read atomically without a lock: a writer between the two loads pairs an old object with a new id. GIL builds compile the lock out, so the spec's "one pointer compare, no lock" holds there. `_intj_lazy` initializes the lock at allocation and destroys it in dealloc, never in clear. A failed first call re-runs `init_bound`, so the lock must outlive it; the spec's "init in init_bound" would re-initialize a live lock.
3. **The hot header carries `state` (the module's `intj_state *`).** The warmed entry reads it on every call. The `module` object pointer stays cold, as the spec lists.
4. **C-side bind checks move to the first call.** Pointer overflow, `data_ptr()` failures, STATIC_COMPILE tensor capture and `hipDeviceGet` all need the built module. Python-side bind checks stay at `.bind()`. The device-ordinal range check moves into Python, so it stays at `.bind_device()`.
5. **Unknown `dynamic_options` compile-option names raise on the first call.** The option set belongs to the target's backend. Knob paths, overlaps, and tuned-name clashes are refused at `make_launcher`.
6. **Recursion guard unchanged per shape.** The entry calls `Py_EnterRecursiveCall` only where today's bound entry did (fixed device, bound values, `grid_py`, tuned). Plain launchers keep today's guard-free path. That is the "within 1 ns" acceptance case.

## Benchmark Gate

Every task ends by rerunning the launcher benchmarks against a baseline taken at this plan's starting commit (Task 0). A task does not commit with an unexplained regression.

**Suite.** All four scripts in `benchmarks/`, in every mode, plus the C++ kernel-cache benchmark that `benchmarks/AGENTS.md` lists. That is 12 cases, 3 rounds each (`launch_tuned` and `launch_dynamic` were added in Task 5; `lazy-baseline` predates them, so `bench_compare.py` reports them as MISSING against it). Every case runs in its own process, one at a time, pinned to core 0. The driver is `/tmp/intj-bench/run_all.sh <label>`. It writes `/tmp/intj-bench/<label>/round<r>_<case>.txt` with the commit, command, raw output, elapsed time and exit status:

| case | command (after `PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 /tmp/gb2/bin/python`) |
|---|---|
| launch_gpu | `benchmarks/bench_launch.py --iters 1000 --batches 9` |
| launch_host | `benchmarks/bench_launch.py --no-gpu --iters 100000 --batches 9` |
| launch_readme | `benchmarks/bench_launch.py --readme --iters 1000 --batches 9` |
| launch_sweep | `benchmarks/bench_launch.py --sweep --iters 100000 --batches 9` |
| launch_last_key | `benchmarks/bench_launch.py --last-key --iters 100000 --batches 9` |
| ffi_compare | `benchmarks/bench_ffi_compare.py --iters 1000 --batches 9` |
| ffi_sweep | `benchmarks/bench_ffi_compare.py --sweep --iters 1000 --batches 9` |
| ffi_paths | `benchmarks/bench_intj_ffi_paths.py --mode all --iters 1000 --batches 9` |
| hip_module | `benchmarks/bench_hip_module_launch.py --iters 1000 --batches 9` |
| kernel_cache | `-m pytest tests/test_kernel_cache.py -s -q` (google/benchmark in `/tmp/gbench`) |
| launch_tuned | `benchmarks/bench_launch.py --tuned --iters 20000 --batches 9` |
| launch_dynamic | `benchmarks/bench_launch.py --dynamic --iters 20000 --batches 9` |

If `/tmp` was cleaned, recreate the driver from this copy (unchanged from the autotune plan):

```bash
#!/bin/bash
# usage: run_all.sh <label>  -- every benchmark, 3 rounds, separate processes, core 0
set -u
label=$1; out=/tmp/intj-bench/$label; mkdir -p $out
cd /mnt/nvme2/jinpli/workspace/home/jinpli/development/intj
PY=/tmp/gb2/bin/python
cases=(
  "launch_gpu|benchmarks/bench_launch.py --iters 1000 --batches 9"
  "launch_host|benchmarks/bench_launch.py --no-gpu --iters 100000 --batches 9"
  "launch_readme|benchmarks/bench_launch.py --readme --iters 1000 --batches 9"
  "launch_sweep|benchmarks/bench_launch.py --sweep --iters 100000 --batches 9"
  "launch_last_key|benchmarks/bench_launch.py --last-key --iters 100000 --batches 9"
  "ffi_compare|benchmarks/bench_ffi_compare.py --iters 1000 --batches 9"
  "ffi_sweep|benchmarks/bench_ffi_compare.py --sweep --iters 1000 --batches 9"
  "ffi_paths|benchmarks/bench_intj_ffi_paths.py --mode all --iters 1000 --batches 9"
  "hip_module|benchmarks/bench_hip_module_launch.py --iters 1000 --batches 9"
  "kernel_cache|-m pytest tests/test_kernel_cache.py -s -q"
  "launch_tuned|benchmarks/bench_launch.py --tuned --iters 20000 --batches 9"
  "launch_dynamic|benchmarks/bench_launch.py --dynamic --iters 20000 --batches 9"
)
for r in 0 1 2; do
  for c in "${cases[@]}"; do
    name=${c%%|*}; args=${c#*|}
    f=$out/round${r}_$name.txt
    {
      echo "commit=$(git rev-parse HEAD)"
      echo "started=$(date -Iseconds)"
      echo "command=PYTHONPATH=\$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 $PY $args"
      s=$(date +%s%N)
      PYTHONPATH=$PWD INTJ_BENCHMARK_ROOT=/tmp/gbench taskset -c 0 $PY $args 2>&1
      st=$?
      echo "elapsed_ns=$(( $(date +%s%N) - s ))"
      echo "exit_status=$st"
      echo "ended=$(date -Iseconds)"
    } > $f
  done
done
echo done > $out/DONE
```

The benchmarks use `/tmp/gb2/bin/python` (torch 2.14.0+rocm7.2, triton 3.8.0, apache-tvm-ffi): the uv venv has no tvm-ffi. Tests and pyright use the venv.

**Compare.** `/tmp/gb2/bin/python /tmp/intj-bench/bench_compare.py lazy-baseline lazy-task<N>` prints the baseline median, the new median, and the deltas, and flags `REGRESSION`:
- A row **regresses** if it is slower by more than 3% *and* by more than 2 ns (host rows) or 0.1 µs (GPU/README rows).
- On a regression, rerun `/tmp/intj-bench/run_all.sh lazy-task<N>-rerun` once. The regression is real only if the rerun also exceeds the threshold (GPU submission rows vary about 20% between processes).
- A real regression gets fixed in that task. If it is the known, accepted cost of the task, say so in the journal and in the commit message.

**Interleaved A/B (acceptance: an untuned launcher without dynamic values is within 1 ns of today on `--no-gpu`).** Tasks 2 and 5 run 5 interleaved rounds of `launch_host`: the Task 0 commit in a detached worktree against `HEAD`. Recreate both scripts if missing:

```bash
#!/bin/bash
# /tmp/intj-bench/lazy_ab.sh <label> -- 5 interleaved rounds: baseline worktree vs HEAD, host matrix
set -u
label=$1; out=/tmp/intj-bench/$label-ab; mkdir -p $out
repo=/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj
base=/tmp/intj-lazy-base
[ -d $base ] || git -C $repo worktree add --detach $base "$(cat /tmp/intj-bench/lazy-baseline/COMMIT)"
for r in 0 1 2 3 4; do
  for tree in base:$base $label:$repo; do
    name=${tree%%:*}; dir=${tree#*:}
    (cd $dir; echo "commit=$(git rev-parse HEAD)"
     PYTHONPATH=$dir taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --no-gpu --iters 100000 --batches 9 2>&1
    ) > $out/${name}_r${r}_launch_host.txt
  done
done
echo done > $out/DONE
```

```python
# /tmp/intj-bench/lazy_ab_summary.py <dir> <label> -- median per row across the 5 rounds
import statistics, sys
from pathlib import Path
sys.path.insert(0, "/tmp/intj-bench")
import bench_compare as bc
d, new = Path(sys.argv[1]), sys.argv[2]
per = {}
for side in ("base", new):
    for f in sorted(d.glob(f"{side}_r*_launch_host.txt")):
        m = {}
        bc.parse_file(f, "launch_host", m)
        for (_, _, row, _), metric in m.items():
            per.setdefault((row, side), []).append(metric.values[0])
print(f"| row | base ns | {new} ns | delta |\n|---|---:|---:|---:|")
for row in sorted({r for r, _ in per}):
    b, n = statistics.median(per[(row, "base")]), statistics.median(per[(row, new)])
    print(f"| {row} | {b:.1f} | {n:.1f} | {n - b:+.1f} |")
```

**Per-launcher state.** A second launcher of one `ModuleKey` pays one miss on its first call of each key. That miss builds its C record from the module's compile cache and runs no Triton compile (`test_sibling_launchers_compile_once`). Warmed rows read the launcher's own cache and lock. With the GIL the lock is a no-op, so they stay comparable to the baseline. One accepted change: `--readme`'s `spec_key` decode row now passes `(launcher, device)` ahead of the arguments, about 1 ns. Say so in the Task 2 journal.

**Record.** Write `benchmarks/journals/$(date +%F)_lazy-task<N>_0_<sha>.md` (baseline: `..._lazy-baseline_0_<sha>.md`) in the format of `benchmarks/AGENTS.md` "Result journals". Include the commands, raw outputs, a baseline-vs-task table of medians with the delta per row, the A/B table where the task runs one, and the environment. Commit it with the task.

---

## Review Focus

1. **The first call fails after `init_bound` has filled part of the header** (a bound pointer of `2**64`, a failing `hipDeviceGet`). Expect the error to propagate, no leaked owner references, and a clean retry. Pinned in Task 2 (`test_bind_native_failure_releases_partial_owners`, rewritten).
2. **A launcher is never called, and its builder closure sits in a reference cycle** (a module-level handle that is dropped, or an object holding its own launcher). Expect GC to collect it before and after the build. Pinned in Task 2 (`test_unbuilt_and_built_launchers_are_collected`).
3. **An object constexpr of an unsupported kind** (a `list`, a `str` subclass, a tensor). Expect a `TypeError` that names the parameter, with the launcher still usable afterwards. Pinned in Task 3 (`test_unsupported_object_constexpr_names_the_parameter`).
4. **A declared knob's passed value differs from the live knob on a miss.** Expect `ValueError`, the knob untouched (intj never sets knobs), no record, and a later call with the live value that compiles; an old value still hits its record after the knob changes. Pinned in Task 4 (`test_declared_knob_only_keys_and_must_match_the_live_value`, `test_tuned_dynamic_knob_mismatch_raises_and_records_nothing`).
5. **Two launchers of one module intern different strings first**, so the same id means different values in each. Expect every value to launch its own binary through either handle, and each variant to compile once across both, through the module's compile cache. Pinned in Task 3 (`test_sibling_launchers_intern_independently`).

## File Map

| file | responsibility | tasks |
|---|---|---|
| `intj/runtime/intj_lazy.h` (new) | `intj_bound_header`: the fixed header (hot fields first, `offsetof` asserts; the launcher's own cache storage, intern table, rwlock, miss callback); `intj_rwlock` moves here | 1 |
| `intj/python_intf/cpython_abi.h`, `intj/python_intf/AGENTS.md` | the rwlock block moves out to `intj_lazy.h` (it is pthread, not a CPython internal) | 1 |
| `intj/runtime/intj_lazy.c` (new) | `_intj_lazy`: launcher type, build shim, `new_launcher`, GC | 2 |
| `intj/lazy.py` (new) | compile, cache and load `_intj_lazy` without triton | 2 |
| `intj/runtime/intj_runtime.h` | header typedef, trailing arrays (`intj_bound_tail`), object ids and memos | 1, 3 |
| `intj/runtime/entry.c.jinja` | tail accessors; `init_bound` replaces `make_bound`; every launcher on its own cache, lock and callback; module `entry`/`st->cache`/`st->lock`/`set_compile_callback` removed; `spec_key(launcher, …)`; hooks; object and dynamic decode; dynamic argument layout | 1–4 |
| `intj/launcher.py` | lazy `LauncherFactory`, `module_of`, knob options at the first call, the module compile function and cache, `_launcher_compile`/`override_compile`, per-launcher `_Interner`, `DynamicSlot`, `dynamic_options` | 2–4 |
| `intj/tuning.py` | `knob_values`/`dynamic` in the tuned miss path | 2, 4 |
| `intj/grid.py` | `compile_grid(..., offset=)` for dynamic arguments | 4 |
| `tests/test_runtime.py` | stub, swap, retry, concurrency, GC and memo tests on the Python matrix | 2, 3 |
| `tests/test_launcher.py`, `test_grid.py`, `test_return_compiled.py`, `test_kernel_cache.py`, `test_tuned.py` | `module_of` migration, lazy/dynamic/object tests, invariants | 1–5 |
| `benchmarks/bench_launch.py`, `bench_ffi_compare.py`, `bench_intj_ffi_paths.py` | `module_of` migration; `--dynamic` rows | 2, 5 |
| `docs/Usage.md`, `AGENTS.md`, `intj/python_intf/AGENTS.md`, `benchmarks/AGENTS.md`, `TODO.md` | docs | 2–5 |

**Out of scope (follow-up plans after this lands):** the spec's "Migration" section. That covers moving the Triton tree's 24 per-call-variant sites, 39 fpsan sites, handles and `partial` stacks to `@make_launcher(dynamic_options=...)`, and the aiter work: its 228 `intj_handle`, 47 `launch_tuned` and 134 `compat.launch` sites, plus deleting `intj_tuned.py`/`intj_handle.py`. `intj.compat` is unchanged here.

---

### Task 0: Benchmark baseline

About 30 minutes, mostly waiting on runs.

- [ ] **Step 1: Run the suite on the unchanged tree**

```bash
git status --short   # clean apart from this plan
/tmp/intj-bench/run_all.sh lazy-baseline
git rev-parse HEAD > /tmp/intj-bench/lazy-baseline/COMMIT
```

Expected: `/tmp/intj-bench/lazy-baseline/DONE` exists, and all 30 files report `exit_status=0`.

- [ ] **Step 2: Record and commit the baseline journal**

Write `benchmarks/journals/$(date +%F)_lazy-baseline_0_$(git rev-parse --short HEAD).md` per the Benchmark Gate's "Record" rule. It holds the baseline medians only (`/tmp/gb2/bin/python /tmp/intj-bench/bench_compare.py lazy-baseline`).

```bash
git add benchmarks/journals/*_lazy-baseline_0_*.md
git commit -m "Record launcher benchmark baseline before lazy launchers

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 1: Uniform bound-launcher layout

About 3 hours. There is no behaviour change. Every bound object gets one fixed header, with the hot fields first, followed by trailing arrays. The rendered module still allocates it from its own type. Task 2 moves the allocation to `_intj_lazy`.

**Files:**
- Create: `intj/runtime/intj_lazy.h`
- Modify: `intj/runtime/intj_runtime.h:694-775` (bound struct, `intj_bind_tensor`, `intj_decode_bound_tensor`)
- Modify: `intj/runtime/entry.c.jinja:68-79` (struct-selecting defines), `:371`, `:1234`, `:1244`, `:1308-1341` (`intj_bound_entry`), `:1361-1432` (traverse/clear), `:1451-1461` (spec), `:1497-1581` (`make_bound` body), `:1654` (`key_chain`)
- Test: `tests/test_launcher.py`

**Interfaces:**
- Produces, C, `intj_lazy.h`:
  - `typedef struct intj_bound_header {...} intj_bound_header;` with fields `state, device_ordinal, device_handle, cache_ready, cache[3], fixed_kernel` (hot, in that order), then `def, doc, builder, module, intern, compile_cb, lock, grid_py, grid_hidden, build_lock, build_thread, traverse, clear` (cold)
  - `intj_rwlock` and `INTJ_RWLOCK_INIT/DESTROY`, `INTJ_RDLOCK`, `INTJ_WRLOCK`, `INTJ_RWUNLOCK`, moved here from `cpython_abi.h`
  - `typedef PyObject *(*intj_fast_fn)(PyObject *, PyObject *const *, Py_ssize_t);`
- Produces, C, `intj_runtime.h`:
  - `typedef intj_bound_header intj_bound_launcher;`
  - `intj_memo {PyObject *obj; uint64_t id;}`, `intj_bound_tail {memo[], owners[], pointer_bits[], tensors[]}`
  - `INTJ_TAIL(b)`, `INTJ_BOUND_CACHE_OF(b)`
  - `INTJ_NMEMO` (defaults to 0), `INTJ_MEMO_SLOTS`, `INTJ_BOUND_SLOTS`
- Invariant for Task 2: `sizeof(intj_bound_tail) == 16 * max(NMEMO,1) + 24 * max(NBOUND,1)`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_launcher.py`:

```python
def test_bound_handles_share_one_header_layout(bound_kernel, device_kernel):
    """Every bound object is one fixed header plus trailing arrays sized per module."""
    pointer = Argument(
        type=tl.pointer_type(tl.float32), specialize=NEVER, bind_value=BindValue.POINTER
    )
    two = make_launcher(
        bound_kernel,
        extra_annotation={"x": pointer, "p": pointer},
        no_gpu=True,
        torch_access_mode=TorchAccessMode.INTERPRETER,
    ).bind(x=0, p=0)
    none = make_launcher(
        device_kernel,
        bind_device=True,
        no_gpu=True,
        torch_access_mode=TorchAccessMode.INTERPRETER,
    ).bind_device(0)
    assert two(0, 0, 1, 7) is None and none(0, 1, 7) is None
    t2, t0 = type(two.__self__), type(none.__self__)
    assert t2.__basicsize__ == t0.__basicsize__
    assert t2.__itemsize__ == t0.__itemsize__ == 1
    # tail: 16 bytes per memo slot and 24 per bound slot, each at least one
    assert sys.getsizeof(two.__self__) - sys.getsizeof(none.__self__) == 24
```

- [ ] **Step 2: Run it to verify it fails**

Run: `PYTHONPATH=$PWD VIRTUAL_ENV=$V uv run --active --no-sync python -m pytest tests/test_launcher.py -q -k one_header_layout`
Expected: FAIL. `__basicsize__` differs between the two modules, and `__itemsize__` is 0.

- [ ] **Step 3: Create `intj/runtime/intj_lazy.h`**

```c
/* The bound launcher's fixed header.  `_intj_lazy` (intj_lazy.c) allocates
 * it at its final size; a rendered module's init_bound fills it in place.
 * Both compile this one definition, so the static asserts below pin the same
 * layout on both sides.  Trailing arrays follow the header: their layout is
 * the rendered module's (`intj_bound_tail`, intj_runtime.h), their byte
 * count is ob_size. */
#pragma once

#include <Python.h>
#include <assert.h>
#include <stddef.h>
#include <stdint.h>

typedef PyObject *(*intj_fast_fn)(PyObject *, PyObject *const *, Py_ssize_t);

/* A launcher's lock, for what the GIL guards on a default build: its kernel
 * cache, memos and intern writes.  Launches take it shared across their
 * lookups and the launch, because records live in the map and a put can move
 * them; a put takes it exclusive.  Compiled out with the GIL, so a default
 * build pays nothing.  CPython's own rwlock is private, hence pthread.
 * (Moved from python_intf/cpython_abi.h: it is not a CPython internal.) */
#ifdef Py_GIL_DISABLED
#include <pthread.h>
typedef pthread_rwlock_t intj_rwlock;
#define INTJ_RWLOCK_INIT(l) pthread_rwlock_init((l), NULL)
#define INTJ_RWLOCK_DESTROY(l) pthread_rwlock_destroy(l)
#define INTJ_RDLOCK(l) pthread_rwlock_rdlock(l)
#define INTJ_WRLOCK(l) pthread_rwlock_wrlock(l)
#define INTJ_RWUNLOCK(l) pthread_rwlock_unlock(l)
#else
typedef char intj_rwlock;
#define INTJ_RWLOCK_INIT(l) ((void)(l))
#define INTJ_RWLOCK_DESTROY(l) ((void)(l))
#define INTJ_RDLOCK(l) ((void)(l))
#define INTJ_WRLOCK(l) ((void)(l))
#define INTJ_RWUNLOCK(l) ((void)(l))
#endif

typedef struct intj_bound_header {
  PyObject_VAR_HEAD /* ob_size: bytes of trailing arrays */
  /* Hot: a warmed launch reads these and nothing else in the header. */
  void *state;             /* the module's intj_state; NULL until built, and after clear */
  int64_t device_ordinal;  /* bind_device only */
  int32_t device_handle;   /* bind_device only */
  int32_t cache_ready;
  uint64_t cache[3];       /* opaque intj_cache storage, only ever accessed as one */
  void *fixed_kernel;      /* intj_final *: a keyless fixed-device launcher's kernel */
  /* Cold. */
  PyMethodDef def;         /* the launcher PyCFunction's; ml_meth is swapped once */
  PyObject *doc;           /* bytes behind def.ml_doc */
  PyObject *builder;       /* until built: callable(header) -> entry address */
  PyObject *module;
  PyObject *intern;        /* callable(value) -> id or None: this launcher's intern table */
  PyObject *compile_cb;    /* this launcher's miss callback (the tuned callback if tuned) */
  intj_rwlock lock;        /* guards this launcher's cache, memos and intern writes */
  PyObject *grid_py;
  PyObject *grid_hidden;
  PyThread_type_lock build_lock;
  unsigned long build_thread; /* thread running the builder, else 0 */
  int (*traverse)(PyObject *self, visitproc visit, void *arg); /* the module's, once built */
  int (*clear)(PyObject *self);
} intj_bound_header;

static_assert(offsetof(intj_bound_header, state) == sizeof(PyVarObject),
              "intj: state leads the hot fields");
static_assert(offsetof(intj_bound_header, fixed_kernel) + sizeof(void *) <=
                  sizeof(PyVarObject) + 64,
              "intj: hot fields fit the 64 bytes after the object header");
static_assert(offsetof(intj_bound_header, def) >=
                  offsetof(intj_bound_header, fixed_kernel) + sizeof(void *),
              "intj: cold fields follow the hot ones");
static_assert(sizeof(intj_bound_header) % 8 == 0,
              "intj: trailing arrays start 8-byte aligned");
```

- [ ] **Step 4: Replace the bound struct in `intj_runtime.h`**

Delete the rwlock block from `intj/python_intf/cpython_abi.h` (lines 127–147), because `intj_lazy.h` now defines it. Add `#include "intj_lazy.h"` right after the `#include INTJ_CPYTHON_STATIC_COMPILE_HEADER` line (27) of `intj_runtime.h`, so everything below can use the lock. In `intj/python_intf/AGENTS.md` "Free-threaded modules", change "`intj_rwlock`, owned by the module," to "`intj_rwlock` (in `runtime/intj_lazy.h`), owned by each launcher,".

Replace lines 694–726 (from `#ifndef INTJ_NBOUND` through `} intj_bound_launcher;`) with:

```c

#ifndef INTJ_NBOUND
#define INTJ_NBOUND 0
#endif
#ifndef INTJ_NMEMO
#define INTJ_NMEMO 0
#endif
#define INTJ_BOUND_SLOTS (INTJ_NBOUND ? INTJ_NBOUND : 1)
#define INTJ_MEMO_SLOTS (INTJ_NMEMO ? INTJ_NMEMO : 1)

typedef intj_bound_header intj_bound_launcher;

/* An object-valued slot's last object and its interned id. */
typedef struct {
  PyObject *obj;
  uint64_t id;
} intj_memo;

/* The trailing arrays: memos first, then one entry per bound parameter. */
typedef struct {
  intj_memo memo[INTJ_MEMO_SLOTS];
  PyObject *owners[INTJ_BOUND_SLOTS];
  uint64_t pointer_bits[INTJ_BOUND_SLOTS];
  void *tensors[INTJ_BOUND_SLOTS]; /* at::Tensor * in STATIC_COMPILE */
} intj_bound_tail;
/* launcher._tail_bytes computes the same size for _intj_lazy's allocation. */
static_assert(sizeof(intj_bound_tail) == 16 * INTJ_MEMO_SLOTS + 24 * INTJ_BOUND_SLOTS,
              "intj: bound tail layout");
#define INTJ_TAIL(b) ((intj_bound_tail *)((char *)(b) + sizeof(intj_bound_header)))

/* The header's cache storage, typed.  Never accessed any other way. */
#define INTJ_BOUND_CACHE_OF(b) ((intj_cache *)(void *)(b)->cache)
static_assert(sizeof(intj_cache) <= sizeof(((intj_bound_header *)0)->cache),
              "intj: intj_cache fits the header's storage");
```

In `intj_bind_tensor` (old lines 738–748), replace the body under `#if defined(INTJ_TORCH_ACCESS_STATIC_COMPILE)` with:

```c
#if defined(INTJ_TORCH_ACCESS_STATIC_COMPILE)
  if (value != Py_None) {
    at::Tensor *tensor = new (std::nothrow) at::Tensor(intj_cdata(value));
    if (!tensor) {
      PyErr_NoMemory();
      return -1;
    }
    INTJ_TAIL(bound)->tensors[index] = tensor;
  }
#else
  INTJ_TAIL(bound)->owners[index] = Py_NewRef(value);
#endif
```

In `intj_decode_bound_tensor`, replace `bound->tensors[index]` (both uses) with a local `at::Tensor *tensor = (at::Tensor *)INTJ_TAIL(bound)->tensors[index];`. Use `!tensor` and `*tensor`. Replace `bound->owners[index]` with `INTJ_TAIL(bound)->owners[index]`.

- [ ] **Step 5: Update `entry.c.jinja`**

1. Delete the five defines that only selected struct fields (lines 68–79): `INTJ_GRID_PY`, `INTJ_FIXED_DEVICE`, `INTJ_TUNED`, `INTJ_BOUND_CACHE`, `INTJ_BOUND_FIXED_KERNEL`. Keep `INTJ_RETURN_COMPILED`, because `intj_map.h` reads it.
2. Line 371: `bound->pointer_bits[` → `INTJ_TAIL(bound)->pointer_bits[`.
3. Lines 1234 and 1244: `bound->fixed_kernel` → `(intj_final *)bound->fixed_kernel`.
4. `intj_bound_entry` (1310–1315): replace the `module` check and the `PyModule_GetState` call with:

```c
  intj_bound_launcher *bound = (intj_bound_launcher *)self;
  intj_state *st = (intj_state *)bound->state;
  if (INTJ_UNLIKELY(!st)) {
    PyErr_SetString(PyExc_RuntimeError, "intj: bound launcher has been cleared");
    return NULL;
  }
```

   and on line 1327 change `'&bound->cache'` to `'INTJ_BOUND_CACHE_OF(bound)'`.
5. Replace `intj_bound_traverse` and `intj_bound_clear` (1361–1424) with:

```c
static int intj_bound_traverse(PyObject *self, visitproc visit, void *arg) {
  intj_bound_launcher *bound = (intj_bound_launcher *)self;
  intj_bound_tail *tail = INTJ_TAIL(bound);
  Py_VISIT(Py_TYPE(self));
  Py_VISIT(bound->module);
  Py_VISIT(bound->grid_py);
  Py_VISIT(bound->grid_hidden);
  Py_VISIT(bound->compile_cb);
  for (int i = 0; i < INTJ_NBOUND; i++)
    Py_VISIT(tail->owners[i]);
{% if (return_compiled or (tuned and grid_py_mode)) and ((fixed_device and nwords) or tuned) %}
  if (bound->cache_ready && bound->state)
    return intj_cache_traverse((intj_state *)bound->state, INTJ_BOUND_CACHE_OF(bound), visit, arg);
{% elif return_compiled and fixed_device %}
  if (bound->state) {
    intj_state *st = (intj_state *)bound->state;
    intj_snapshot s = {NULL, 0, 0};
    INTJ_RDLOCK(&st->lock);
    int failed = bound->fixed_kernel ? intj_final_collect((intj_final *)bound->fixed_kernel, &s) : 0;
    INTJ_RWUNLOCK(&st->lock);
    return intj_snapshot_visit(&s, failed, visit, arg);
  }
{% endif %}
  return 0;
}

static int intj_bound_clear(PyObject *self) {
  intj_bound_launcher *bound = (intj_bound_launcher *)self;
  intj_bound_tail *tail = INTJ_TAIL(bound);
  /* Mark closed before dropping owners whose finalizers can reenter Python. */
  bound->state = NULL;
  Py_CLEAR(bound->module);
  Py_CLEAR(bound->grid_py);
  Py_CLEAR(bound->grid_hidden);
  Py_CLEAR(bound->compile_cb);
  for (int i = 0; i < INTJ_NBOUND; i++) {
    Py_CLEAR(tail->owners[i]);
#if defined(INTJ_TORCH_ACCESS_STATIC_COMPILE)
    delete (at::Tensor *)tail->tensors[i];
    tail->tensors[i] = NULL;
#endif
  }
{% if (fixed_device and nwords) or tuned %}
  if (bound->cache_ready) {
    bound->cache_ready = 0;
    intj_cache_free(INTJ_BOUND_CACHE_OF(bound), INTJ_V0_RELEASE);
  }
{% elif fixed_device %}
  intj_final *kernel = (intj_final *)bound->fixed_kernel;
  bound->fixed_kernel = NULL;
  if (kernel) {
    intj_final_release(kernel);
    PyMem_RawFree(kernel);
  }
{% endif %}
  return 0;
}
```

6. `intj_bound_spec` (1451–1452): `sizeof(intj_bound_launcher), 0,` → `sizeof(intj_bound_launcher), 1,`.
7. `make_bound`:
   - line 1501: allocate with `st->bound_type->tp_alloc(st->bound_type, (Py_ssize_t)sizeof(intj_bound_tail))`;
   - line 1534: `intj_cache_init(INTJ_BOUND_CACHE_OF(bound))`;
   - lines 1574–1576: `INTJ_TAIL(bound)->pointer_bits[...]` and `INTJ_TAIL(bound)->owners[...]`;
   - just before `PyObject *result = PyCFunction_New(...)`, add `bound->state = st;`.
8. `key_chain` line 1654: `&bound->cache` → `INTJ_BOUND_CACHE_OF(bound)`.
9. Rename the header field in its three uses: `bound->tuned_cb` → `bound->compile_cb` (`intj_tuned_fill`, `make_bound`, and the traverse/clear written above). The module keeps using `st->lock` until Task 2; `bound->lock` stays unused here.

- [ ] **Step 6: Run the test and the full suite**

```
PYTHONPATH=$PWD VIRTUAL_ENV=$V uv run --active --no-sync python -m pytest tests -q
bash tests/run_python_matrix.sh 3.8 3.12 3.14t
```

Expected: everything passes, including `test_bound_handles_share_one_header_layout`, in all three access modes, in C++ builds (tsl/absl, STATIC_COMPILE), and on 3.8/3.14t.

- [ ] **Step 7: Check that the layout asserts can fail**

Temporarily move `void *fixed_kernel;` below `PyMethodDef def;` in `intj_lazy.h`, then run `... pytest tests/test_launcher.py -q -k one_header_layout`.
Expected: the build fails with `intj: hot fields fit the 64 bytes after the object header` (or `cold fields follow the hot ones`). Revert the move.

- [ ] **Step G: Benchmark gate**

```bash
/tmp/intj-bench/run_all.sh lazy-task1
/tmp/gb2/bin/python /tmp/intj-bench/bench_compare.py lazy-baseline lazy-task1
```

Write `benchmarks/journals/$(date +%F)_lazy-task1_0_<sha>.md`. Expected: no row regresses. Bound rows now index the tail at a constant offset, the same instructions as before.

- [ ] **Step 8: Commit**

```bash
git add intj/runtime/intj_lazy.h intj/runtime/intj_runtime.h intj/runtime/entry.c.jinja \
  intj/python_intf/cpython_abi.h intj/python_intf/AGENTS.md \
  tests/test_launcher.py benchmarks/journals/*_lazy-task1_0_*.md
git commit -m "Give every bound launcher one fixed header plus trailing arrays

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: `_intj_lazy` and lazy launchers

About 1.5 days. This is the largest task: the stub, the swap, `init_bound`, the lazy `LauncherFactory`, knob reads moving to the first call, and the migration of tests off `launch.__self__`-is-the-module.

**Files:**
- Create: `intj/runtime/intj_lazy.c`, `intj/lazy.py`
- Modify: `intj/runtime/entry.c.jinja`: `intj_state` 153–169; `intj_bound_entry` 1308–1341; delete 1343–1359 (control names and `intj_bound_method`); 1426–1461 (dealloc/getset/slots/spec); 1463–1596 (`make_bound` → `init_bound`); `key_chain` 1638–1644; `intj_exec` 1921–1927; module traverse/clear 1945/1961; methods 1981–1991
- Modify: `intj/launcher.py`: imports; `LauncherFactory` 216–350; `make_launcher` 585–640; `_materialize_module` 643–758; `_loaded_module` 875–919; `_compile_so_bytes` 991–997; `_plan_tuning` 1268–1279; `_make_compile_callback` 1637–1652
- Modify: `intj/tuning.py:286-326` (`make_tuned_callback`)
- Test: `tests/test_runtime.py`, `tests/test_launcher.py`, `tests/test_tuned.py`, `tests/test_return_compiled.py`; migrate `tests/test_grid.py`, `tests/test_return_compiled.py`, `tests/test_kernel_cache.py`, `tests/test_compat.py`, `benchmarks/bench_launch.py`, `benchmarks/bench_ffi_compare.py`, `benchmarks/bench_intj_ffi_paths.py`, `benchmarks/bench_hip_module_launch.py`
- Docs: `docs/Usage.md:26-53`, `AGENTS.md` ("Modules stay out of the import system"), `intj/python_intf/AGENTS.md` ("Compatibility helpers")

**Interfaces:**
- Consumes: `intj_bound_header`, `intj_bound_tail`, `INTJ_TAIL`, `INTJ_BOUND_CACHE_OF` (Task 1)
- Produces, `_intj_lazy` module: `new_launcher(tail_bytes: int, doc: bytes, builder: Callable[[object], int]) -> builtin_function_or_method`; `header_size() -> int`; the launcher's `__self__` has `.build() -> ModuleType` and `.__self__` (the module, or `None` before the build)
- Produces, rendered module: `init_bound(header, callback, [grid_py, hidden], [device], *bound_values) -> int` (the entry's address); `header_size() -> int`; `spec_key(launcher, device, *public) -> (key, nparams)`; `key_chain` (tuned) and `set_torch_version` unchanged. `make_bound`, `entry`, `set_compile_callback`, `st->cache`, `st->compile_cb` and `st->lock` are removed.
- Produces, `intj/lazy.py`: `load_stub(root: Path, cc: str) -> ModuleType`, `python_include() -> str`
- Produces, `intj/launcher.py`:
  - `module_of(launcher) -> ModuleType`, `_tail_bytes(nmemo: int, nbound: int) -> int`, `_stub() -> ModuleType`
  - `_launch_doc(params, fixed_device, grid_arg, grid_cpp, grid_py, dynamic=()) -> bytes`
  - `_KNOB_OPTIONS`, `_live_knobs() -> dict[str, object]`, `_knob_options(jit_func, options, knob_values) -> dict[str, Any]`
  - `_check_tuned_configs(plan, resolved, options, knob_values) -> None`
  - `LauncherFactory._build(header, resolved, args, hidden) -> int`
  - `_materialize_module(..., knob_values: Mapping[str, object] | None = None)`
  - `_loaded_module(..., baked_values, knob_values=None)`
  - `_make_compile_callback(..., *, return_compiled=False, knob_values: Mapping[str, object])`, which returns the module compile function `(seen, keyblob, nparams, device, *args)`
  - `_launcher_compile(module) -> Callable` (one launcher's callback, with its own `seen`), `override_compile(module, fn) -> previous`
- Produces, `intj/tuning.py`: `make_tuned_callback(plan, resolved, options, grid, render, return_compiled, knob_values)`
- Produces, ownership: every launcher with a key (`nwords > 0`) uses `INTJ_BOUND_CACHE_OF(bound)`, and a keyless fixed-device launcher uses `fixed_kernel`. Everything is guarded by `bound->lock`. The module's C state is written only at load.

- [ ] **Step 1: Write the failing runtime tests (triton-free, run on the whole matrix)**

In `tests/test_runtime.py`, add `import dis`, `import gc`, `import time` and `import weakref` to the imports, plus `from intj import lazy`. In the `built` fixture, replace `module.set_compile_callback(compile_cb)` with `launcher.override_compile(module, compile_cb)`. Then append:

```python
@pytest.fixture(scope="module")
def stub_module(tmp_path_factory):
    return lazy.load_stub(tmp_path_factory.mktemp("lazy"), _cc("c"))


_CALL = (0, 0, 1)  # device, stream, grid
_TAIL = launcher._tail_bytes(0, 0)


@pytest.fixture(scope="module")
def launch(built, stub_module):
    """A built launcher over the fixture module: what `module.entry` used to be."""
    module, _, _ = built
    fn = stub_module.new_launcher(
        _TAIL, b"", lambda header: module.init_bound(header, launcher._launcher_compile(module))
    )
    fn(*_args())
    return fn


def _args():
    return (*_CALL, torch.zeros(4), 5, -7, 1.5, True, 64)


def test_stub_stays_out_of_sys_modules_and_matches_the_header(built, stub_module):
    module, _, _ = built
    assert "_intj_lazy" not in sys.modules
    assert stub_module.header_size() == module.header_size()


def test_first_call_builds_and_swaps_for_old_references(built, stub_module):
    module, _, _ = built
    builds = []

    def build(header):
        builds.append(1)
        return module.init_bound(header, launcher._launcher_compile(module))

    launch = stub_module.new_launcher(_TAIL, b"launch(a, /)\n--\n\n", build)
    before = launch  # a reference taken before the build
    function = ctypes.pythonapi.PyCFunction_GetFunction
    function.argtypes, function.restype = [ctypes.py_object], ctypes.c_void_p
    shim = function(launch)
    assert launch.__self__.__self__ is None and builds == []
    args = _args()
    assert launch(*args) is None
    assert builds == [1] and function(before) != shim
    assert launch.__self__.__self__ is module and launch.__self__.build() is module

    def call():
        return before(*args)

    for _ in range(100):
        assert call() is None
    assert builds == [1]
    free_threaded = bool(sysconfig.get_config_var("Py_GIL_DISABLED"))
    if sys.version_info >= (3, 11) and not (free_threaded and sys.version_info[:2] == (3, 13)):
        names = {i.opname for i in dis.get_instructions(call, adaptive=True)}  # pyright: ignore[reportCallIssue]
        assert names & {"PRECALL_NO_KW_BUILTIN_FAST", "CALL_NO_KW_BUILTIN_FAST", "CALL_BUILTIN_FAST"}


def test_failed_build_raises_and_the_next_call_retries(built, stub_module):
    module, _, _ = built
    attempts = []

    def build(header):
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("first build fails")
        return module.init_bound(header, launcher._launcher_compile(module))

    launch = stub_module.new_launcher(_TAIL, b"", build)
    with pytest.raises(RuntimeError, match="first build fails"):
        launch(*_args())
    assert launch(*_args()) is None and len(attempts) == 2


def test_concurrent_first_calls_build_once(built, stub_module):
    module, _, _ = built
    builds = []

    def build(header):
        builds.append(1)
        time.sleep(0.05)  # hold the build lock while the others arrive
        return module.init_bound(header, launcher._launcher_compile(module))

    launch = stub_module.new_launcher(_TAIL, b"", build)
    barrier, results, errors = threading.Barrier(8), [], []

    def worker():
        barrier.wait()
        try:
            results.append(launch(*_args()))
        except BaseException as error:  # noqa: BLE001  report, do not hang
            errors.append(error)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == [] and results == [None] * 8 and builds == [1]


def test_call_during_build_is_refused(built, stub_module):
    module, _, _ = built
    seen = []

    def build(header):
        try:
            launch(*_args())
        except RuntimeError as error:
            seen.append(str(error))
        return module.init_bound(header, launcher._launcher_compile(module))

    launch = stub_module.new_launcher(_TAIL, b"", build)
    assert launch(*_args()) is None
    assert seen and "while it was being built" in seen[0]


def test_unbuilt_and_built_launchers_are_collected(built, stub_module):
    module, _, _ = built

    class Owner:
        pass

    for call_first in (False, True):
        owner = Owner()

        def build(header, owner=owner):  # the builder holds its owner: a cycle
            return module.init_bound(header, launcher._launcher_compile(module))

        owner.launch = stub_module.new_launcher(_TAIL, b"", build)
        if call_first:
            owner.launch(*_args())
        ref = weakref.ref(owner)
        del owner, build
        gc.collect()
        assert ref() is None, f"leaked (called first: {call_first})"
```

In the `built` fixture, `module.header_size()` and `module.init_bound` do not exist yet.

- [ ] **Step 2: Run them to verify they fail**

Run: `PYTHONPATH=$PWD VIRTUAL_ENV=$V uv run --active --no-sync python -m pytest tests/test_runtime.py -q -k "stub or swap or retries or build or collected"`
Expected: FAIL at import with `ImportError: cannot import name 'lazy' from 'intj'`.

- [ ] **Step 3: Write `intj/runtime/intj_lazy.c`**

```c
/* `_intj_lazy`: the type every launcher is allocated from, and the shim its
 * PyCFunction starts on.  Generic -- no kernel code, no GPU -- so a launcher
 * exists before its module is rendered.  The first call builds: the builder
 * renders and loads the module, whose init_bound fills this header in place
 * and returns its entry; the shim swaps def.ml_meth to that entry.  CPython
 * reads m_ml->ml_meth on every call (tests/test_runtime.py pins it, 3.8 to
 * 3.14t), so references taken before the build reach the entry as well. */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <string.h>

#include "intj_lazy.h"

typedef struct {
  PyTypeObject *type;
} intj_lazy_state;

static PyObject *intj_lazy_call(PyObject *self, PyObject *const *args, Py_ssize_t nargs);
#define INTJ_SHIM ((PyCFunction)(void (*)(void))intj_lazy_call)

/* Run the builder once, under the header's lock: 0, or -1 with an error set.
 * A failure leaves the shim in place, so the next call retries. */
static int intj_lazy_build(intj_bound_header *h) {
  unsigned long me = PyThread_get_thread_ident();
  if (__atomic_load_n(&h->build_thread, __ATOMIC_RELAXED) == me) {
    PyErr_SetString(PyExc_RuntimeError,
                    "intj: a launcher was called while it was being built");
    return -1;
  }
  if (!PyThread_acquire_lock(h->build_lock, NOWAIT_LOCK)) {
    /* detached while waiting: the builder runs Python, and a free-threaded
     * stop-the-world must not wait on us */
    Py_BEGIN_ALLOW_THREADS
    PyThread_acquire_lock(h->build_lock, WAIT_LOCK);
    Py_END_ALLOW_THREADS
  }
  int rc = 0;
  PyObject *done = NULL;
  if (__atomic_load_n(&h->def.ml_meth, __ATOMIC_ACQUIRE) == INTJ_SHIM) {
    if (!h->builder) {
      PyErr_SetString(PyExc_RuntimeError, "intj: launcher has been cleared");
      rc = -1;
    } else {
      __atomic_store_n(&h->build_thread, me, __ATOMIC_RELAXED);
      PyObject *entry = PyObject_CallFunctionObjArgs(h->builder, (PyObject *)h, NULL);
      __atomic_store_n(&h->build_thread, 0UL, __ATOMIC_RELAXED);
      void *address = entry ? PyLong_AsVoidPtr(entry) : NULL;
      Py_XDECREF(entry);
      if (!address) {
        if (!PyErr_Occurred())
          PyErr_SetString(PyExc_RuntimeError, "intj: launcher builder returned no entry");
        rc = -1;
      } else {
        PyCFunction fn;
        memcpy(&fn, &address, sizeof(fn));
        __atomic_store_n(&h->def.ml_meth, fn, __ATOMIC_RELEASE);
        done = h->builder;
        h->builder = NULL;
      }
    }
  }
  PyThread_release_lock(h->build_lock);
  Py_XDECREF(done); /* outside the lock: the closure's finalizers run Python */
  return rc;
}

static PyObject *intj_lazy_call(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
  intj_bound_header *h = (intj_bound_header *)self;
  if (intj_lazy_build(h) != 0)
    return NULL;
  PyCFunction entry = __atomic_load_n(&h->def.ml_meth, __ATOMIC_ACQUIRE);
  return ((intj_fast_fn)(void (*)(void))entry)(self, args, nargs);
}

static PyObject *intj_lazy_build_method(PyObject *self, PyObject *unused) {
  (void)unused;
  intj_bound_header *h = (intj_bound_header *)self;
  if (intj_lazy_build(h) != 0)
    return NULL;
  if (!h->module) {
    PyErr_SetString(PyExc_RuntimeError, "intj: launcher has been cleared");
    return NULL;
  }
  Py_INCREF(h->module);
  return h->module;
}

static PyObject *intj_lazy_module(PyObject *self, void *closure) {
  (void)closure;
  PyObject *module = ((intj_bound_header *)self)->module;
  module = module ? module : Py_None;
  Py_INCREF(module);
  return module;
}

static int intj_lazy_traverse(PyObject *self, visitproc visit, void *arg) {
  intj_bound_header *h = (intj_bound_header *)self;
  Py_VISIT(Py_TYPE(self));
  Py_VISIT(h->builder);
  Py_VISIT(h->module);
  /* the module's hook visits what init_bound stored; never these two again */
  return h->traverse ? h->traverse(self, visit, arg) : 0;
}

static int intj_lazy_clear(PyObject *self) {
  intj_bound_header *h = (intj_bound_header *)self;
  if (h->clear)
    h->clear(self); /* marks the launcher closed first, then drops its state */
  Py_CLEAR(h->builder);
  Py_CLEAR(h->module);
  return 0;
}

static void intj_lazy_dealloc(PyObject *self) {
  intj_bound_header *h = (intj_bound_header *)self;
  PyTypeObject *type = Py_TYPE(self);
  PyObject_GC_UnTrack(self);
  intj_lazy_clear(self);
  Py_CLEAR(h->doc); /* only here: a live PyCFunction may still read ml_doc */
  if (h->build_lock)
    PyThread_free_lock(h->build_lock);
  INTJ_RWLOCK_DESTROY(&h->lock);
  type->tp_free(self);
  Py_DECREF(type);
}

static PyGetSetDef intj_lazy_getset[] = {
    {(char *)"__self__", intj_lazy_module, NULL, NULL, NULL},
    {NULL, NULL, NULL, NULL, NULL}};

static PyMethodDef intj_lazy_type_methods[] = {
    {"build", intj_lazy_build_method, METH_NOARGS,
     "Build now if the first call has not, and return the rendered module."},
    {NULL, NULL, 0, NULL}};

static PyType_Slot intj_lazy_slots[] = {
    {Py_tp_traverse, (void *)intj_lazy_traverse},
    {Py_tp_clear, (void *)intj_lazy_clear},
    {Py_tp_dealloc, (void *)intj_lazy_dealloc},
    {Py_tp_getset, (void *)intj_lazy_getset},
    {Py_tp_methods, (void *)intj_lazy_type_methods},
    {0, NULL}};

static PyType_Spec intj_lazy_spec = {
    "_intj_lazy.Launcher", sizeof(intj_bound_header), 1,
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC
#ifdef Py_TPFLAGS_IMMUTABLETYPE
        | Py_TPFLAGS_IMMUTABLETYPE
#endif
#ifdef Py_TPFLAGS_DISALLOW_INSTANTIATION
        | Py_TPFLAGS_DISALLOW_INSTANTIATION
#endif
    ,
    intj_lazy_slots};

/* new_launcher(tail_bytes, doc, builder) -> an unbuilt launcher */
static PyObject *new_launcher(PyObject *m, PyObject *const *args, Py_ssize_t nargs) {
  intj_lazy_state *st = (intj_lazy_state *)PyModule_GetState(m);
  Py_ssize_t tail = -1;
  if (nargs == 3 && PyLong_CheckExact(args[0]))
    tail = PyLong_AsSsize_t(args[0]);
  if (tail < 0 || !PyBytes_CheckExact(args[1]) || !PyCallable_Check(args[2])) {
    if (!PyErr_Occurred())
      PyErr_SetString(PyExc_TypeError, "intj: new_launcher(tail_bytes, doc, builder)");
    return NULL;
  }
  intj_bound_header *h = (intj_bound_header *)st->type->tp_alloc(st->type, tail);
  if (!h)
    return NULL;
  INTJ_RWLOCK_INIT(&h->lock); /* lives as long as the header; see dealloc */
  h->build_lock = PyThread_allocate_lock();
  if (!h->build_lock) {
    Py_DECREF(h);
    return PyErr_NoMemory();
  }
  Py_INCREF(args[1]);
  h->doc = args[1];
  Py_INCREF(args[2]);
  h->builder = args[2];
  h->def.ml_name = "launch";
  h->def.ml_meth = INTJ_SHIM;
  h->def.ml_flags = METH_FASTCALL;
  h->def.ml_doc = PyBytes_AS_STRING(h->doc);
  PyObject *launcher = PyCFunction_New(&h->def, (PyObject *)h);
  Py_DECREF(h);
  return launcher;
}

static PyObject *header_size(PyObject *m, PyObject *unused) {
  (void)m;
  (void)unused;
  return PyLong_FromSize_t(sizeof(intj_bound_header));
}

static int intj_lazy_exec(PyObject *m) {
  intj_lazy_state *st = (intj_lazy_state *)PyModule_GetState(m);
#if PY_VERSION_HEX >= 0x03090000
  st->type = (PyTypeObject *)PyType_FromModuleAndSpec(m, &intj_lazy_spec, NULL);
#else
  st->type = (PyTypeObject *)PyType_FromSpec(&intj_lazy_spec);
#endif
  return st->type ? 0 : -1;
}

static int intj_lazy_module_traverse(PyObject *m, visitproc visit, void *arg) {
  Py_VISIT(((intj_lazy_state *)PyModule_GetState(m))->type);
  return 0;
}

static int intj_lazy_module_clear(PyObject *m) {
  Py_CLEAR(((intj_lazy_state *)PyModule_GetState(m))->type);
  return 0;
}

static PyMethodDef intj_lazy_methods[] = {
    {"new_launcher", (PyCFunction)(void (*)(void))new_launcher, METH_FASTCALL,
     "new_launcher(tail_bytes, doc, builder) -> launcher"},
    {"header_size", header_size, METH_NOARGS, NULL},
    {NULL, NULL, 0, NULL}};

static PyModuleDef_Slot intj_lazy_module_slots[] = {
    {Py_mod_exec, (void *)intj_lazy_exec},
#ifdef Py_mod_multiple_interpreters
    {Py_mod_multiple_interpreters, Py_MOD_MULTIPLE_INTERPRETERS_NOT_SUPPORTED},
#endif
#ifdef Py_mod_gil
    {Py_mod_gil, Py_MOD_GIL_NOT_USED},
#endif
    {0, NULL}};

static struct PyModuleDef intj_lazy_module_def = {
    PyModuleDef_HEAD_INIT, "_intj_lazy", NULL, sizeof(intj_lazy_state),
    intj_lazy_methods, intj_lazy_module_slots, intj_lazy_module_traverse,
    intj_lazy_module_clear, NULL};

PyMODINIT_FUNC PyInit__intj_lazy(void) { return PyModuleDef_Init(&intj_lazy_module_def); }
```

- [ ] **Step 4: Write `intj/lazy.py`**

```python
"""Build and load `_intj_lazy`, the generic stub every launcher starts as.

Kernel-free, so it needs only the host C compiler and no GPU: `make_launcher`
allocates launchers from it at decoration time (see runtime/intj_lazy.c).
Cached on disk per interpreter and intj version; loaded by hand, never
imported, like every module intj builds.  No triton here: the Python-version
matrix (tests/run_python_matrix.sh) loads it on interpreters triton skips.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import subprocess
import sys
import sysconfig
import tempfile
import types
from pathlib import Path

from ._version import __version__

_RUNTIME = Path(__file__).parent / "runtime"
_SOURCE = _RUNTIME / "intj_lazy.c"
_HEADER = _RUNTIME / "intj_lazy.h"


def python_include() -> str:
    """This interpreter's `Python.h` directory; Debian's posix_local scheme included."""
    get_scheme = getattr(sysconfig, "get_default_scheme", None)  # 3.10+
    scheme = get_scheme() if get_scheme is not None else None
    if scheme == "posix_local":
        scheme = "posix_prefix"
    paths = sysconfig.get_paths(scheme=scheme) if scheme else sysconfig.get_paths()
    return paths["include"]


def _digest(cc: str) -> str:
    h = hashlib.sha256(_SOURCE.read_bytes() + _HEADER.read_bytes())
    h.update(repr((__version__, sys.version, sysconfig.get_config_var("EXT_SUFFIX"), cc)).encode())
    return h.hexdigest()[:32]


def load_stub(root: Path, cc: str) -> types.ModuleType:
    """`_intj_lazy`, compiled into `root/<digest>/` on first use."""
    suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
    so = root / _digest(cc) / f"_intj_lazy{suffix}"
    if not so.exists():
        so.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=so.parent) as staging:
            built = Path(staging) / so.name
            subprocess.check_call(
                [cc, "-O2", "-shared", "-fPIC", f"-I{_RUNTIME}", f"-I{python_include()}",
                 str(_SOURCE), "-o", str(built)],
                stdout=subprocess.DEVNULL,
            )
            os.replace(built, so)  # atomic: a racing loader sees all of it or none
    spec = importlib.util.spec_from_file_location("_intj_lazy", so)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"intj: cannot load {so}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # pyright: ignore[reportAttributeAccessIssue]  # 3.8 stubs lack it
    return module
```

- [ ] **Step 5: Turn `make_bound` into `init_bound` in `entry.c.jinja`**

1. `intj_state` (153–169): delete `PyTypeObject *bound_type;`.
2. `intj_bound_entry` (1308–1341): wrap the recursion guard, so plain launchers keep today's guard-free path. Put `{% set guarded = fixed_device or bound_params or grid_py_mode or tuned %}` above the function, and wrap both `Py_EnterRecursiveCall` (with its `return NULL`) and `Py_LeaveRecursiveCall()` in `{% if guarded %} … {% endif %}`.
3. Delete lines 1343–1359: the control-name block and `static PyMethodDef intj_bound_method`. Python renders the signature now (Step 7).
4. In `intj_bound_traverse` (from Task 1), delete `Py_VISIT(Py_TYPE(self));` and `Py_VISIT(bound->module);`. `_intj_lazy` visits both, and a second visit would corrupt GC's reference accounting.
5. Delete `intj_bound_dealloc`, `intj_bound_module`, `intj_bound_getset`, `intj_bound_slots` and `intj_bound_spec` (1426–1461).
6. Replace `make_bound`'s head, through its `abi.ready` check, with the head below (lines 1463–1473). Then:
   - delete lines 1497–1504 (the `bound` declaration, the `tp_alloc` call and `bound->module = Py_NewRef(self);`);
   - add `bound->module = Py_NewRef(self);` and `bound->compile_cb = Py_NewRef(callback);` as the first statements inside `try {`;
   - drop the `{% if tuned %}` callable check on `args[0]` and the `bound->tuned_cb = …` line: `callback` replaces them for every launcher;
   - in every remaining `args[{{ … }}]` index, drop the `tuned_offset` term. The head consumed the callback slot, so tuned and untuned launchers now index alike.
   - Replace the tail from `PyObject *result = PyCFunction_New(...)` through the end of the function with the tail below.

```c
/* init_bound(header, callback, [grid_py, fixed values], [device], *bound)
 * -> the entry's address.  Fills an unbuilt `_intj_lazy` header in place.  On
 * failure the header is left unbuilt, so the launcher's next call retries.
 * `callback` is this launcher's miss callback: the tuned callback, or a
 * per-launcher wrapper of the module's compile function. */
static PyObject *init_bound(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
  intj_state *st = (intj_state *)PyModule_GetState(self);
  if (nargs != 2 + {{ bound_arg_offset + grid_py_offset }} + INTJ_NBOUND) {
    PyErr_Format(PyExc_TypeError, "intj: init_bound takes %d arguments, got %zd",
                 2 + {{ bound_arg_offset + grid_py_offset }} + INTJ_NBOUND, nargs);
    return NULL;
  }
  PyTypeObject *type = Py_TYPE(args[0]);
  intj_bound_launcher *bound = (intj_bound_launcher *)args[0];
  if (type->tp_basicsize != (Py_ssize_t)sizeof(intj_bound_header) || type->tp_itemsize != 1 ||
      Py_SIZE(bound) != (Py_ssize_t)sizeof(intj_bound_tail) || bound->state || bound->module) {
    PyErr_SetString(PyExc_TypeError,
                    "intj: init_bound needs an unbuilt launcher sized for this module");
    return NULL;
  }
  PyObject *callback = args[1];
  if (!PyCallable_Check(callback)) {
    PyErr_SetString(PyExc_TypeError, "intj: init_bound needs the launcher's miss callback");
    return NULL;
  }
  args += 2;
  if (!st->abi.ready) {
    PyErr_SetString(PyExc_RuntimeError, "intj: set_torch_version has not been called");
    return NULL;
  }
```

```c
    PyObject *entry = PyLong_FromVoidPtr((void *)intj_bound_entry);
    if (!entry)
      goto fail;
    bound->traverse = intj_bound_traverse;
    bound->clear = intj_bound_clear;
    bound->state = st; /* last: the entry reads it first */
    return entry;
#ifdef __cplusplus
  } catch (const std::bad_alloc &) {
    PyErr_NoMemory();
  } catch (const std::exception &error) {
    PyErr_SetString(PyExc_RuntimeError, error.what());
  } catch (...) {
    PyErr_SetString(PyExc_RuntimeError, "intj: C++ exception while binding arguments");
  }
#endif
fail:
  intj_bound_clear((PyObject *)bound); /* drops whatever was filled; the header stays unbuilt */
  return NULL;
}

static PyObject *header_size(PyObject *self, PyObject *unused) {
  (void)self;
  (void)unused;
  return PyLong_FromSize_t(sizeof(intj_bound_header));
}
```

7. `key_chain` (1638–1644): replace `Py_TYPE(owner) != st->bound_type` with `Py_TYPE(owner)->tp_basicsize != (Py_ssize_t)sizeof(intj_bound_header)`.
8. `intj_exec`: delete the `bound_type` creation (1921–1927). `intj_traverse`: delete `Py_VISIT(st->bound_type);`. `intj_clear`: delete `Py_CLEAR(st->bound_type);`.
9. `intj_methods`:
   - replace the `make_bound` row with `{"init_bound", (PyCFunction)(void (*)(void))init_bound, METH_FASTCALL, NULL},`;
   - add `{"header_size", header_size, METH_NOARGS, NULL},`;
   - delete the `entry` and `set_compile_callback` rows.
10. **Module state becomes read-only after load.**
    - Delete the `entry` function (1291–1306) and `set_compile_callback` (1680–1689).
    - From `intj_state`, delete `compile_cb`, `cache` and `lock`. In `intj_exec`, `intj_traverse`, `intj_clear` and `intj_free`, delete everything touching them: `INTJ_RWLOCK_INIT/DESTROY(&st->lock)`, `intj_cache_init/free(&st->cache, …)`, `intj_cache_traverse(st, &st->cache, …)`, and `Py_VISIT/Py_CLEAR(st->compile_cb)`.
    - `set_torch_version` still writes `st->abi` once. It runs under `_LOAD_LOCK`, before any launcher can reach the module, so it needs no lock.
11. **Every lock is the launcher's.** Replace every `&st->lock` with `&bound->lock`. The functions that have no `bound` take it in place of their cache pointer, and derive the cache as `INTJ_BOUND_CACHE_OF(bound)`:
    - `intj_cache_traverse(intj_bound_launcher *bound, visitproc, void *)`;
    - `intj_walk(st, bound, kargs, key, comp, dep, &missed)`;
    - `intj_call(st, device, stream, args, nargs, bound)`, where the `cache` parameter goes;
    - `intj_fill(st, bound, key, np, device_ordinal, call_args)` and `intj_tuned_fill(st, bound, …)`.
    `intj_bound_entry` calls `intj_call(st, device, stream, …, bound)`.
12. **The miss callback is the launcher's.** In `intj_fill`, replace the `st->compile_cb` read and its lock with `PyObject *cb = Py_NewRef(bound->compile_cb);`. `init_bound` set it before publishing, and only `tp_clear` of an unreachable launcher clears it. In `intj_tuned_fill`, `bound->compile_cb` already is the tuned callback (Task 1 renamed it).
13. **`spec_key(launcher, device, *public) -> (key, nparams)`**, the level-0 half of `key_chain`. Replace its head with the launcher check below, then call `intj_pack(st, args + 2, key, vals, &np, {{ '(uint32_t)bound->device_ordinal' if fixed_device else '(uint32_t)device' }}, bound{{ ', comp' if tuned }})`:

```c
  intj_state *st = (intj_state *)PyModule_GetState(self);
  int64_t device = 0;
  if (!st->abi.ready) {
    PyErr_SetString(PyExc_RuntimeError, "intj: set_torch_version has not been called");
    return NULL;
  }
  if (nargs != 2 + INTJ_NPARAMS || !PyCFunction_Check(args[0]) ||
      !PyLong_CheckExact(args[1]) || intj_as_i64(args[1], &device) != 0 ||
      device < 0 || device > 255) {
    PyErr_Format(PyExc_TypeError, "intj: spec_key(launcher, device, *%d public args)", INTJ_NPARAMS);
    return NULL;
  }
  PyObject *owner = PyCFunction_GET_SELF(args[0]);
  if (!owner || Py_TYPE(owner)->tp_basicsize != (Py_ssize_t)sizeof(intj_bound_header) ||
      ((intj_bound_launcher *)owner)->module != self) {
    PyErr_SetString(PyExc_TypeError, "intj: spec_key needs a built launcher of this module");
    return NULL;
  }
  intj_bound_launcher *bound = (intj_bound_launcher *)owner;
```

- [ ] **Step 5b: Every launcher owns its cache and callback; the module keeps a compile function**

In `entry.c.jinja`, the bound cache was selected by `(fixed_device and nwords) or tuned`. Select it by `nwords` alone. `nwords` is 0 only for a keyless fixed-device launcher, which keeps `fixed_kernel`.
- `intj_call` derives its cache: `{{ 'INTJ_BOUND_CACHE_OF(bound)' if nwords else 'NULL' }}`.
- `init_bound`: `{% if (fixed_device and nwords) or tuned %}` around `intj_cache_init` → `{% if nwords %}`.
- `intj_bound_clear`: `{% if nwords %}` (free the cache) / `{% elif fixed_device %}` (free `fixed_kernel`).
- `intj_bound_traverse`: `{% if (return_compiled or (tuned and grid_py_mode)) and nwords %}`.

`intj_lazy.c` (Step 3) already initializes `h->lock` in `new_launcher` and destroys it in dealloc. The lock lives exactly as long as the header, so a retried `init_bound` never re-initializes it.

In `launcher.py`, add `import weakref`, and `MutableMapping` to the `collections.abc` import. `_make_compile_callback` returns the **module's** compile function. Replace its body from `canonical_options = …` to the end with:

```python
    canonical_options = _canonical_options(target, _knob_options(jit_func, options, knob_values))
    # The module's compile cache: a sibling launcher's miss reuses the kernel
    # and only builds its own C record.  With return_compiled the records own
    # their CompiledKernel, so the cache must not keep it alive past them.
    compiled: MutableMapping[tuple[object, ...], CompiledKernel] = (
        weakref.WeakValueDictionary() if return_compiled else {}
    )
    lock = threading.Lock()  # one compile per input across the module's launchers

    def compile_callback(
        seen: dict[bytes, object], keyblob: bytes, nparams: int, device: int, *args: Any
    ) -> tuple[int, int, int, int] | tuple[int, int, int, int, CompiledKernel]:
        """`seen` is the calling launcher's own key -> input map."""
        current = _current_device()
        if device != current:
            raise UnsupportedKernel(
                f"intj: launching on device {device} while device {current} is current; "
                "make the target device current before the first launch"
            )
        compiler_input = _compiler_input(jit_func, params, args, backend, baked_values=baked_values)
        # b"" is a keyless launcher's one key; seen is per launcher, so that is exact
        if seen.setdefault(keyblob, compiler_input) != compiler_input:
            raise RuntimeError(
                "intj: one spec key maps to two annotated ASTSource inputs; this is an intj bug"
            )
        cache_key = (compiler_input, canonical_options.hash(), current)
        with lock:
            kernel = compiled.get(cache_key)
            if kernel is None:
                kernel = compiled[cache_key] = _checked_compile(
                    jit_func, compiler_input, target, canonical_options
                )
        md = kernel.metadata
        expected = sum(1 for ty in kernel.src.signature.values() if ty != "constexpr")
        if expected != nparams:
            raise RuntimeError(
                f"intj: packed {nparams} kernel arguments but triton compiled {expected}; "
                "this is an intj bug"
            )
        result = (kernel.function, md.warp_size * md.num_warps, md.shared, nparams)
        return (*result, kernel) if return_compiled else result

    return compile_callback
```

`_make_host_compile_callback` takes the same leading `seen` argument, and `_refuse_module_compile` already takes `*args`. Add after it:

```python
def _launcher_compile(module: types.ModuleType) -> Callable[..., Any]:
    """One launcher's miss callback: its own `seen` map, then the module's
    compile function, looked up per miss so `override_compile` reaches built
    launchers too."""
    seen: dict[bytes, object] = {}

    def callback(*call: Any) -> Any:
        return getattr(module, "_intj_compile")(seen, *call)

    return callback


def override_compile(module: types.ModuleType, fn: Callable[..., Any]) -> Any:
    """Tests and benchmarks: compile misses of `module`'s launchers with
    `fn(keyblob, nparams, device, *args)`.  Returns the previous function;
    restore it with `setattr(module, "_intj_compile", previous)`."""
    previous = getattr(module, "_intj_compile", None)
    setattr(module, "_intj_compile", lambda seen, *call: fn(*call))
    return previous
```

In `_loaded_module`, replace `module.set_compile_callback(<expr>)` with `setattr(module, "_intj_compile", <expr>)`. This is the module object's dict, written once under `_LOAD_LOCK`, not C state. In `LauncherFactory._build`, the callback is `prefix[0]` for tuned launchers. Otherwise build it and pass it first:

```python
        callback = prefix[0] if self.tuned is not None else _launcher_compile(module)
        rest = prefix[1:] if self.tuned is not None else prefix
        return module.init_bound(header, callback, *rest, *args)
```

- [ ] **Step 6: Run the runtime tests**

Run: `PYTHONPATH=$PWD VIRTUAL_ENV=$V uv run --active --no-sync python -m pytest tests/test_runtime.py -q`
Expected: the stub tests still fail on `launcher._tail_bytes` (Step 7 adds it). Everything else in the file passes.

- [ ] **Step 7: Make `launcher.py` lazy**

Imports: add `import contextlib` (used in Task 4), `from collections.abc import Iterator` (Task 4), `from . import lazy`, and `from .annotation import ...` stays.

Add the helpers after `_provision`:

```python
@functools.lru_cache(maxsize=1)
def _stub() -> types.ModuleType:
    """The process's one `_intj_lazy`, beside the modules intj renders."""
    from triton import knobs

    return lazy.load_stub(
        Path(knobs.cache.get_triton_dir("intj")) / "lazy", _compiler_path("c")
    )


def _tail_bytes(nmemo: int, nbound: int) -> int:
    """sizeof(intj_bound_tail) in intj_runtime.h: 16 per memo and 24 per bound
    slot, each at least one.  The rendered module static_asserts the same."""
    return 16 * max(nmemo, 1) + 24 * max(nbound, 1)


def module_of(launcher: Callable[..., Any]) -> types.ModuleType:
    """The rendered module behind `launcher`, building it now if its first
    call has not.  For tests and debugging: `module_of(k).spec_key(...)`."""
    return getattr(launcher, "__self__").build()


def _launch_doc(
    params: Sequence[Param],
    fixed_device: bool,
    grid_arg: int | None,
    grid_cpp: GridCode | None,
    grid_py: bool,
    dynamic: Sequence[str] = (),
) -> bytes:
    """The launcher's `__text_signature__`: controls renamed away from kernel names."""
    public = [p.name for p in params if p.call_index is not None]
    extras = list(grid_cpp.extras) if grid_cpp else []
    values = [name.replace(".", "_") for name in dynamic]
    grid = (
        []
        if grid_cpp or grid_py
        else ["grid"] if grid_arg is None else ["grid_x", "grid_y", "grid_z"][:grid_arg]
    )
    occupied = set(public) | set(extras) | set(values)
    controls: list[str] = []
    for name in (["stream"] if fixed_device else ["device", "stream"]) + grid:
        while name in occupied or name in controls:
            name += "_"
        controls.append(name)
    return f"launch({', '.join(controls + extras + values + public)}, /)\n--\n\n".encode()


#: knob path -> the compile option triton derives from it
_KNOB_OPTIONS: tuple[tuple[str, str], ...] = (
    ("knobs.runtime.debug", "debug"),
    ("knobs.compilation.instrumentation_mode", "instrumentation_mode"),
    ("knobs.compilation.fpsan_homomorphic_casts", "fpsan_homomorphic_casts"),
)


def _live_knobs() -> dict[str, object]:
    """The knobs triton turns into compile options, as they are now.  Read at
    a launcher's first call and fixed for it."""
    from triton import knobs

    values: dict[str, object] = {}
    for path, _ in _KNOB_OPTIONS:
        _, group, name = path.split(".")
        values[path] = getattr(getattr(knobs, group), name, None)
    return values


def _knob_options(
    jit_func: JitFunction, options: Mapping[str, Any], knob_values: Mapping[str, object]
) -> dict[str, Any]:
    """`options` plus what triton derives from knobs: an explicit `debug`
    overrides the kernel's default, and the runtime knob can still enable it."""
    merged = dict(options)
    merged["debug"] = (
        options.get("debug", jit_func.debug) or knob_values["knobs.runtime.debug"]
    )
    merged["instrumentation_mode"] = knob_values["knobs.compilation.instrumentation_mode"]
    casts = knob_values["knobs.compilation.fpsan_homomorphic_casts"]
    if casts is not None:
        merged["fpsan_homomorphic_casts"] = casts
    return merged
```

Replace `LauncherFactory.bind_device` and `_bind` (238–350), and add `_binding` and `_build`:

```python
    def bind_device(self, /, *args: object, **values: object) -> Callable[..., Any]:
        if not self.bind_device_requested:
            raise TypeError("intj: bind_device was not requested")
        if len(args) != 1:
            raise TypeError(
                "intj: bind_device requires exactly one positional device ordinal"
            )
        ordinal = args[0]
        # here, not on the first call, so a bad ordinal fails where it is written
        if type(ordinal) is not int or not 0 <= ordinal <= 2**31 - 1:
            raise TypeError("intj: device ordinal must be an int in [0, 2**31)")
        return self._bind(ordinal, values)

    def _binding(self) -> DeviceBinding:
        return DeviceBinding.FIXED if self.bind_device_requested else DeviceBinding.NOT_FIXED

    def _bind(self, device: object, values: Mapping[str, object]) -> Callable[..., Any]:
        """A lazy launcher.  Everything here is GPU-free; `_build` runs on its first call."""
        # <lines 248-287 unchanged: name checks, torch/type imports, `resolved`>
        state = tuple(resolved)
        bound_values = tuple(values[p.name] for p in state if p.name in names)
        hidden = tuple(
            values[p.name] if p.name in names else p.baked
            for p in state
            if p.annotation.bind_value is not None or p.annotation.baked_value
        )
        args = (*((device,) if self.bind_device_requested else ()), *bound_values)
        params, _, _ = _render_params(state, self._binding())
        doc = _launch_doc(
            params, self.bind_device_requested, self.grid_arg, self.grid_cpp,
            self.grid_py is not None,
        )

        def build(header: object) -> int:
            return self._build(header, state, args, hidden)

        return _stub().new_launcher(_tail_bytes(0, len(bound_values)), doc, build)

    def _build(
        self,
        header: object,
        resolved: tuple[ResolvedParam, ...],
        args: tuple[object, ...],
        hidden: tuple[object, ...],
    ) -> int:
        """The first call: query the target, render, compile, load, fill `header`.
        Returns the rendered entry's address for `_intj_lazy` to swap in."""
        knob_values = None if self.no_gpu else _live_knobs()
        if self.tuned is not None:
            assert knob_values is not None, "_plan_tuning refuses no_gpu"
            _check_tuned_configs(self.tuned.plan, resolved, dict(self.options), knob_values)
        module = _materialize_module(
            self.jit_func,
            resolved,
            dict(self.options),
            self.torch_access_mode,
            self.kernel_cache,
            self.no_gpu,
            self.verify_annotation,
            self._binding(),
            grid_arg=self.grid_arg,
            grid_py_mode=self.grid_py is not None,
            grid_cpp=self.grid_cpp,
            return_compiled=self.return_compiled,
            tuning=self.tuned.render if self.tuned else None,
            knob_values=knob_values,
        )
        prefix: tuple[object, ...] = ()
        if self.tuned is not None:
            from .tuning import TunedGrid, make_tuned_callback

            assert self.tuned.render is not None and knob_values is not None
            if self.grid_py is not None:
                grid = TunedGrid("py", self.grid_py)
            elif self.grid_cpp is not None:
                code = getattr(self.grid_fn, "__code__")
                grid = TunedGrid(
                    "cpp",
                    self.grid_fn,
                    tuple(code.co_varnames[: code.co_argcount]),
                    self.grid_cpp.extras,
                )
            else:
                grid = TunedGrid("dims")
            prefix = (
                make_tuned_callback(
                    self.tuned.plan,
                    resolved,
                    dict(self.options),
                    grid,
                    self.tuned.render,
                    self.return_compiled,
                    knob_values,
                ),
            )
        if self.grid_py is not None:
            prefix += (self.grid_py, hidden)
        return module.init_bound(header, *prefix, *args)
```

In `make_launcher`:
- Delete lines 585–592 (the knob block).
- Replace lines 593–640 with the code below. `_provision` runs here: it is GPU-free, and anything that reaches the network must not run on a call.

```python
    access = _resolve_torch_access_mode(torch_access_mode)
    needs_binding = bind_device or any(
        p.annotation.bind_value is not None for p in resolved
    )
    if needs_binding and verify_annotation:
        for p in resolved:
            if (
                p.annotation.bind_value == "pointer"
                and p.annotation.pointer_range_32 == "assume"
            ):
                warnings.warn(
                    f"intj: pointer-range assumption for bound pointer {p.name!r} "
                    "cannot be verified from an address and will be trusted",
                    RuntimeWarning,
                    stacklevel=2,
                )
    # GPU-free but may reach the network: here, never on a call
    _provision(kernel_cache)
    factory = LauncherFactory(
        kernel,
        resolved,
        tuple(options.items()),
        access,
        kernel_cache,
        bool(verify_annotation),
        no_gpu,
        bool(bind_device),
        grid_arg,
        grid_py,
        compiled_grid,
        bool(return_compiled),
        tuned,
        grid_cpp,
    )
    return factory if needs_binding else factory.bind()
```

Update the docstring paragraph that says "everything that needs the kernel -- annotation resolution, compilation, the GPU -- waits for the inner call...". The new text: "make_launcher does no GPU work: the returned launcher builds its module on its first call (see docs/Usage.md, "Lazy build")."

`_plan_tuning` (1268–1279): keep the `num_ctas` loop, because it is GPU-free. Move the `_canonical_options` call out into:

```python
def _check_tuned_configs(
    plan: Any,
    resolved: tuple[ResolvedParam, ...],
    options: Mapping[str, Any],
    knob_values: Mapping[str, object],
) -> None:
    """Every config's options through the target's `parse_options`: on the
    first call, because the target is a GPU query."""
    from triton.runtime.autotuner import Autotuner

    target = _current_target()
    names = {p.name for p in resolved}
    for layer in plan.layers:
        if type(layer) is Autotuner:
            for config in layer.configs:
                config_options: dict[str, Any] = {
                    str(k): v for k, v in config.all_kwargs().items() if k not in names
                }
                _canonical_options(
                    target, _knob_options(plan.jit_func, {**options, **config_options}, knob_values)
                )
```

In `_plan_tuning`, delete `target = _current_target()` and the `_canonical_options(...)` line inside the loop.

`_materialize_module`:
- Add the parameter `knob_values: Mapping[str, object] | None = None`.
- In the GPU branch, replace `canonical_options = _canonical_options(target, options)` with:

```python
        assert knob_values is not None, "GPU builds read knobs at the first call"
        canonical_options = _canonical_options(
            target, _knob_options(jit_func, options, knob_values)
        )
```

- End with `return _loaded_module(key, jit_func, context, params, options, layout, baked_values, knob_values)`.

`_loaded_module`:
- Add the parameter `knob_values: Mapping[str, object] | None = None`.
- Pass `knob_values=knob_values or {}` to `_make_compile_callback`. Only the GPU branch reads it, and that branch always has knob values.
- Right after `module = _load(key, jit_func, context)`, add:

```python
            if module.header_size() != _stub().header_size():
                raise ImportError(
                    f"intj: {module.__file__} disagrees with _intj_lazy on the launcher "
                    "header; it was built by another intj -- delete it to rebuild"
                )
```

`_make_compile_callback`:
- Signature: `(jit_func, params, options, baked_values=None, *, return_compiled=False, knob_values: Mapping[str, object])`.
- Its canonical options become `_canonical_options(target, _knob_options(jit_func, options, knob_values))`.

`_compile_so_bytes` (991–997): replace the `scheme` lines and `sysconfig.get_paths(scheme=scheme)["include"]` with `lazy.python_include()`.

`intj/tuning.py` `make_tuned_callback`:
- Add the parameter `knob_values: Mapping[str, object]` after `return_compiled`.
- Import `_knob_options` with the other launcher internals.
- In `compile_kernel`: `canonical = _canonical_options(target, _knob_options(jit_func, {**options, **config_options}, knob_values))`.

- [ ] **Step 8: Run the runtime tests on the matrix**

```
PYTHONPATH=$PWD VIRTUAL_ENV=$V uv run --active --no-sync python -m pytest tests/test_runtime.py -q
bash tests/run_python_matrix.sh
```

Expected: pass on all nine builds. On 3.13t and 3.14t this covers the concurrency test with the GIL off.

- [ ] **Step 9: Write the launcher-level lazy tests**

Append to `tests/test_launcher.py` (`from intj.launcher import module_of` at the top):

```python
def test_sibling_launchers_compile_once(monkeypatch):
    """Each launcher owns its cache; the module's compile cache shares kernels."""
    import intj.launcher as launcher

    monkeypatch.setattr(launcher, "_LOADED", {})  # a fresh module: an empty compile cache
    real, compiles = launcher._checked_compile, []

    def counted(*args, **kwargs):
        compiles.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(launcher, "_checked_compile", counted)
    first, second = make_launcher(scale), make_launcher(scale)
    assert first.__self__ is not second.__self__
    x = torch.ones(64, device="cuda")
    o = torch.zeros_like(x)
    device, stream = torch.cuda.current_device(), torch.cuda.current_stream().cuda_stream
    first(device, stream, 1, x, o, 64, 2.0, 64)
    second(device, stream, 1, x, o, 64, 3.0, 64)  # its own miss: a record, no compile
    torch.cuda.synchronize()
    assert o[0].item() == 3.0
    assert module_of(first) is module_of(second) and len(compiles) == 1


def test_decoration_needs_no_gpu(monkeypatch):
    """make_launcher, the decorator form and bind_device never query a target."""
    import intj.launcher as launcher

    def forbidden(*args, **kwargs):
        raise AssertionError("decoration touched the GPU")

    with monkeypatch.context() as patch:
        patch.setattr(launcher, "_current_target", forbidden)
        patch.setattr(launcher, "_current_device", forbidden)
        plain = make_launcher(grid_arg=1)(scale)
        fixed = make_launcher(scale, bind_device=True).bind_device(torch.cuda.current_device())
    x = torch.ones(64, device="cuda")
    o = torch.zeros_like(x)
    device, stream = torch.cuda.current_device(), torch.cuda.current_stream().cuda_stream
    plain(device, stream, 1, x, o, 64, 2.0, 64)
    fixed(stream, (1,), x, o, 64, 3.0, 64)
    torch.cuda.synchronize()
    assert o[0].item() == 3.0


def test_failed_first_call_leaves_the_launcher_unbuilt(scalar_kernel, monkeypatch):
    import intj.launcher as launcher

    real, calls = launcher._materialize_module, []

    def flaky(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise UnsupportedKernel("intj: transient")
        return real(*args, **kwargs)

    monkeypatch.setattr(launcher, "_materialize_module", flaky)
    launch = make_launcher(
        scalar_kernel, no_gpu=True, torch_access_mode=TorchAccessMode.INTERPRETER
    )
    with pytest.raises(UnsupportedKernel, match="transient"):
        launch(0, 0, 1, 7)
    assert launch(0, 0, 1, 7) is None
    assert len(calls) == 2
```

Append to `tests/test_return_compiled.py` (its `store_value(x, value)` kernel is already there):

```python
def test_undeclared_knobs_are_read_at_the_first_call(monkeypatch):
    from triton import knobs

    launch = make_launcher(store_value, return_compiled=True)
    monkeypatch.setattr(knobs.runtime, "debug", True)  # after make_launcher
    x = torch.zeros(1, device="cuda", dtype=torch.int32)
    kernel = launch(torch.cuda.current_device(), torch.cuda.current_stream().cuda_stream, 1, x, 5)
    assert kernel.metadata.debug is True
```

In `tests/test_tuned.py`, append:

```python
def test_tuned_decoration_needs_no_gpu(monkeypatch):
    from intj import launcher as launcher_mod

    def forbidden(*args, **kwargs):
        raise AssertionError("decoration touched the GPU")

    monkeypatch.setattr(launcher_mod, "_current_target", forbidden)
    launch = make_launcher(
        triton.autotune(
            configs=configs(), key=["n"], do_bench=lambda call, quantiles: [call() or 1.0] * 3
        )(tagged),
        grid_cpp=grid,
    )
    monkeypatch.undo()
    x = torch.zeros(256, device="cuda", dtype=torch.int32)
    out = torch.zeros_like(x)
    launch(*controls(), x, out, 256)
    torch.cuda.synchronize()
    assert int(out[0].item()) in (1, 2)
```

- [ ] **Step 10: Migrate `__self__`-means-module uses**

Nothing but the stub owns `__self__` now: `launch.__self__` is the `_intj_lazy.Launcher`, and its `__self__` is the module, or `None` before the build. Apply these rules in `tests/*.py` and `benchmarks/*.py`:

1. `getattr(X, "__self__")`, including the multi-line `getattr(\n make_launcher(...),\n "__self__",\n)` form → `module_of(X)`.
2. `X.__self__.__self__` → `module_of(X)`.
3. A bare `X.__self__` that stands for the module → `module_of(X)`. That means `.spec_key`, `.set_compile_callback`, `.__file__`, `.key_chain`, or `module = X.__self__`. Sites: test_kernel_cache 164/184/201/215; test_return_compiled 70–71/174/332/366; test_grid 123/125/341/372; test_launcher 398/1827/1935/1954/2045/2054/2169/2237/2263/2333; bench_launch.py:98.
4. Keep `X.__self__` where the bound object itself is meant: test_launcher 851 (attribute set), 925 (`first.__self__ is not second.__self__`), 1498 (`gc.get_referents(bound.__self__)`), 1671; test_grid 143; test_return_compiled 325.
5. `_launcher_module` (test_launcher 182–183) becomes `return module_of(launcher)`.
6. In bench_launch.py `row()`, add `start = time.perf_counter_ns(); module_of(launcher); build_ms += (time.perf_counter_ns() - start) / 1e6` after the launcher exists, so `build ms` still means something.

Then fix the tests whose premise moved:
- `test_bind_device_validates_exact_nonnegative_int32`: delete the `make_bound(value)` assertion. The Python check keeps `factory.bind_device(value)` raising.
- `test_bind_native_failure_releases_partial_owners`: replace its body after the factory with:

```python
    tensor = torch.empty(4)
    bound = factory.bind(x=tensor, p=2**64)  # p fails in C, on the first call
    references, native = sys.getrefcount(tensor), getattr(tensor, "_use_count")()
    for _ in range(3):
        with pytest.raises(OverflowError, match="p"):
            bound(0, 0, 1, 4)
    assert sys.getrefcount(tensor) == references
    assert getattr(tensor, "_use_count")() == native
    module = module_of(factory.bind(x=None, p=0))
    with pytest.raises(TypeError, match="arguments"):
        module.init_bound(tensor)
```

- Source-string test (~1336): rename `"static PyObject *make_bound"` to `"static PyObject *init_bound"`.
- Tests that stub `_loaded_module` or `_materialize_module` to capture a build: the knob-options test (~2830), `test_newer_triton_fpsan_knob_is_baked_into_compile_options` (~2878), and the kernel-cache install test (~3977). Make each stub record what it checks and then `raise Captured` (a local `class Captured(Exception)`). Trigger the build with `with pytest.raises(Captured): module_of(make_launcher(...))`. The captured `options` are now the user's. Also:
  - `_loaded_module` stubs take a trailing `knob_values` argument, and `_make_compile_callback` needs `knob_values=knob_values`;
  - the fpsan test asserts on `launcher._knob_options(_kernel, options, kwargs["knob_values"])["fpsan_homomorphic_casts"] is True`.
- `test_modules_stay_out_of_the_import_system`: switch to `module_of(...)` and add `assert "_intj_lazy" not in sys.modules`.

- Tests that install a counting compile callback and launch through two sibling handles (test_launcher ~340–380 `compile_once_per_handle` / `compile_each_constant`, test_grid ~151 and ~341) now see one callback call per launcher per key, because each launcher owns its cache. Update the expected counts. Do not reintroduce sharing.
- **Module API removed or changed** (counted with `grep -c` on today's tree):
  - `module.entry(...)`: 19 calls. 16 are in `test_runtime.py`: use the new module-scoped `launch` fixture and call `launch(...)` with the same arguments. 3 are in `test_launcher.py`: ~231 and ~1728 only asserted that `entry` refuses bound modules, so delete those assertions; ~2129 calls the launcher itself.
  - `module.spec_key(*args)` without a launcher: about 66 sites (test_launcher 58, test_runtime 2, test_return_compiled 1, bench_ffi_compare 3, bench_launch 2). Change each to `module.spec_key(launch, 0, *args)`, passing the launcher the module came from, with device 0 as before. Delete the ~1730 assertion that `spec_key` refuses bound modules. In `--readme`, time `module.spec_key` over `(launcher, device, *args)`.
  - `set_compile_callback(fn)`: 27 sites (tests plus `bench_intj_ffi_paths.py` and `bench_hip_module_launch.py`). Use `override_compile(module, fn)`. Where a test restores a previous callback (test_return_compiled ~374), keep the value `override_compile` returned and `setattr(module, "_intj_compile", previous)`.

Check the migration: `grep -n "__self__" tests/*.py benchmarks/*.py` lists only the rule-4 sites and `module_of` internals.

- [ ] **Step 11: Run the full suite and pyright**

```
PYTHONPATH=$PWD VIRTUAL_ENV=$V uv run --active --no-sync python -m pytest tests -q
PYTHONPATH=$PWD VIRTUAL_ENV=$V uv run --active --no-sync pyright
```

Expected: all tests pass, and pyright reports 0 errors. Any other failure is a test that assumed an eager build. Trigger its build with `module_of(...)` or a call; do not make the launcher eager.

- [ ] **Step 12: Docs**

In `docs/Usage.md`:
- Replace the paragraph at 26–30 ("`make_launcher` renders a C python extension…") with the section below.
- Delete the sentence at 46–48 ("The kernel still needs building at decoration time … a GPU must be available then").
- Add to "How a launch works" a step 0: "The first call builds the module and swaps the launcher's entry; later calls start at step 1."

```markdown
### Lazy build

`make_launcher` returns a launcher at once, without touching the GPU. It
validates arguments, resolves annotations, analyzes an autotune/heuristics
chain, lowers `grid_cpp`, and provisions a non-default `kernel_cache`. The
**first call** builds: it queries the GPU target, renders the C extension,
compiles it with the host compiler (cached on disk under intj's module digest),
loads it, fills the launcher, and swaps its entry. After that, every reference
to the launcher, including ones taken before the build, calls the native entry
directly. If the build raises, that call raises and the next call retries.
Concurrent first calls build once.

Decorating a module-level kernel is therefore safe at import on a machine
without a GPU. A few checks need the target and surface on the first call: an
invalid compile option in an autotune config, and the C-side checks on bound
values (an out-of-range pointer, a failing `hipDeviceGet`). Every launcher is a
builtin function whose `__self__` is an `_intj_lazy.Launcher`, whatever it
binds; `intj.launcher.module_of(launcher)` builds if needed and returns the
rendered module, for debugging.
```

In `AGENTS.md` "Modules stay out of the import system", add after the first paragraph:

"`_intj_lazy` (`intj/lazy.py`, `runtime/intj_lazy.c`), the stub every launcher starts as, follows the same rule. Every launcher is a `PyCFunction` over a `PyMethodDef` embedded in its `_intj_lazy.Launcher`. The first call swaps its `ml_meth` from the build shim to the rendered entry, and GC reaches the module's state through the `traverse`/`clear` hooks that `init_bound` installs."

In `intj/python_intf/AGENTS.md` "Compatibility helpers", add: "Lazy launchers rely on CPython reading `PyCFunctionObject.m_ml->ml_meth` at every call, including from specialized call sites (3.11 `PRECALL_NO_KW_BUILTIN_FAST`, 3.12 `CALL_NO_KW_BUILTIN_FAST`, 3.13+ `CALL_BUILTIN_FAST`; 3.13t does not specialize). They also rely on `meth_dealloc` not reading `m_ml` after dropping `m_self`. `test_first_call_builds_and_swaps_for_old_references` pins both on the matrix. Recheck it first when adding a Python version."

- [ ] **Step G: Benchmark gate and interleaved A/B**

```bash
/tmp/intj-bench/run_all.sh lazy-task2
/tmp/gb2/bin/python /tmp/intj-bench/bench_compare.py lazy-baseline lazy-task2
bash /tmp/intj-bench/lazy_ab.sh lazy-task2
/tmp/gb2/bin/python /tmp/intj-bench/lazy_ab_summary.py /tmp/intj-bench/lazy-task2-ab lazy-task2
```

Write the journal with both tables. Expected:
- no row regresses;
- in the A/B, `auto map`, `reduced key`, `verify off`, `verify on` and `baked` are within 1 ns of `base`;
- bound and fixed-device rows are within the gate.

If a plain row is over 1 ns, compare `perf annotate` of `intj_bound_entry` against the old `entry`. The expected difference is one load (`bound->state` vs `PyModule_GetState`).

- [ ] **Step 13: Commit**

```bash
git add intj/runtime/intj_lazy.c intj/lazy.py intj/runtime/entry.c.jinja intj/launcher.py \
  intj/tuning.py tests benchmarks/bench_launch.py benchmarks/bench_ffi_compare.py \
  benchmarks/bench_intj_ffi_paths.py benchmarks/bench_hip_module_launch.py docs/Usage.md AGENTS.md intj/python_intf/AGENTS.md \
  benchmarks/journals/*_lazy-task2_0_*.md
git commit -m "Build launchers on their first call through a swapped PyCFunction

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Object constexprs keyed by value

About 4 hours. Untyped, unbaked `tl.constexpr` parameters accept `str`, `tl.dtype` and JIT functions. The key carries a per-launcher id, memoized per slot.

**Files:**
- Modify: `intj/runtime/intj_runtime.h`: codes 596–602, `intj_value_kind` 610–617, new `intj_object_id`/`intj_decode_value` after `intj_decode_constexpr` (655), `intj_infer_type`/`intj_constexpr_type` 814–849
- Modify: `intj/runtime/entry.c.jinja`: defines ~63, `intj_state`, `decode_auto_constexpr` 309–316, the generic constexpr decode 378–380, bound hooks, module traverse/clear, methods
- Modify: `intj/launcher.py`: `Param` 83–90, `_render_key` 1363–1411, `_bind` `_tail_bytes` call, `_loaded_module`, new `_object_capable` and `_Interner`
- Test: `tests/test_runtime.py` (fixture interner and memo tests), `tests/test_launcher.py`
- Docs: `docs/Usage.md` "Supported arguments" and "Not supported"

**Interfaces:**
- Consumes: `intj_memo`, `INTJ_TAIL`, `INTJ_NMEMO` (Task 1); `_tail_bytes`, `module_of` (Task 2)
- Produces, C: `INTJ_VALUE_OBJECT`, `INTJ_B_CX_OBJECT` (154)
  - `int intj_object_id(PyObject *intern, intj_rwlock *lock, intj_memo *memo, PyObject *o, const char *pname, uint64_t *id)`
  - `int intj_decode_value(PyObject *o, const char *pname, PyObject *intern, intj_rwlock *lock, intj_memo *memo, intj_decoded *out)`
- Produces, rendered module: `init_bound(header, callback, interner, [grid_py, hidden], [device], *bound)`; the header's `intern`. `spec_key(launcher, …)` (Task 2) decodes objects with that launcher's table.
- Produces, Python: `Param.memo: int | None`; `_object_capable(annotation) -> bool`; `_Interner.__call__(value) -> int | None`, one per launcher

- [ ] **Step 1: Write the failing tests**

In `tests/test_runtime.py`:
- add a per-launcher test interner, and add `_interner()` as the third argument of every `module.init_bound(header, launcher._launcher_compile(module))` in this file (the `launch` fixture included):

```python
def _interner():
    ids = {}
    return lambda v: ids.setdefault(v, len(ids)) if type(v) is str else None
```

- change `_TAIL` to `launcher._tail_bytes(sum(p.memo is not None for p in _PARAMS), 0)`;
- append:

```python
def test_str_constexpr_keys_by_value(built, stub_module):
    module, _, _ = built
    launch = stub_module.new_launcher(
        _TAIL, b"", lambda h: module.init_bound(h, launcher._launcher_compile(module), _interner())
    )
    launch(*_args())
    x = torch.zeros(4)

    def key(value):
        return module.spec_key(launch, 0, x, 5, -7, 1.5, True, value)[0]

    assert key("act") == key("".join(["a", "c", "t"])) != key("other")
    assert key(0) != key("act")  # an int never shares an object's code


def test_memo_holds_its_object_until_replaced(built, stub_module):
    module, _, _ = built
    launch = stub_module.new_launcher(
        _TAIL, b"", lambda h: module.init_bound(h, launcher._launcher_compile(module), _interner())
    )
    first, second = "".join(["o", "n", "e"]), "".join(["t", "w", "o"])
    launch(*_CALL, torch.zeros(4), 5, -7, 1.5, True, first)
    held = sys.getrefcount(first)
    launch(*_CALL, torch.zeros(4), 5, -7, 1.5, True, second)
    assert sys.getrefcount(first) == held - 1  # the memo moved and let go
    launch(*_CALL, torch.zeros(4), 5, -7, 1.5, True, first)
    assert sys.getrefcount(first) == held
    with pytest.raises(TypeError, match="unsupported argument 'BLOCK' of type list"):
        launch(*_CALL, torch.zeros(4), 5, -7, 1.5, True, [1])
```

In `tests/test_launcher.py`:

```python
@triton.jit
def act_store(o, x, ACT: tl.constexpr):
    if ACT == "neg":
        tl.store(o, -x)
    else:
        tl.store(o, x)


@triton.jit
def cast_store(o, x, DT: tl.constexpr):
    tl.store(o, x.to(DT).to(tl.float32))


@triton.jit
def _twice(v):
    return v * 2


@triton.jit
def _thrice(v):
    return v * 3


@triton.jit
def apply_store(o, x, FN: tl.constexpr):
    tl.store(o, FN(x))


def _gpu_controls():
    return torch.cuda.current_device(), torch.cuda.current_stream().cuda_stream, 1


def test_str_constexpr_keys_by_value():
    launch = make_launcher(act_store, return_compiled=True)
    o = torch.zeros(1, device="cuda")
    neg = launch(*_gpu_controls(), o, 2.0, "neg")
    torch.cuda.synchronize()
    assert o.item() == -2.0
    other = "".join(["n", "e", "g"])  # equal value, another object
    assert launch(*_gpu_controls(), o, 2.0, other) is neg
    pos = launch(*_gpu_controls(), o, 2.0, "pos")
    torch.cuda.synchronize()
    assert o.item() == 2.0 and pos is not neg


def test_dtype_and_jit_constexprs():
    o = torch.zeros(1, device="cuda")
    cast = make_launcher(cast_store, return_compiled=True)
    as_int = cast(*_gpu_controls(), o, 2.7, tl.int32)
    torch.cuda.synchronize()
    assert o.item() == 2.0
    assert cast(*_gpu_controls(), o, 2.7, tl.float32) is not as_int
    torch.cuda.synchronize()
    assert abs(o.item() - 2.7) < 1e-6
    apply = make_launcher(apply_store, return_compiled=True)
    twice = apply(*_gpu_controls(), o, 3.0, _twice)
    torch.cuda.synchronize()
    assert o.item() == 6.0
    again = triton.jit(_twice.fn)  # another JITFunction, same cache_key
    assert again is not _twice and apply(*_gpu_controls(), o, 3.0, again) is twice
    apply(*_gpu_controls(), o, 3.0, _thrice)
    torch.cuda.synchronize()
    assert o.item() == 9.0


def test_sibling_launchers_intern_independently(monkeypatch):
    """Id 0 is "neg" in one launcher and "pos" in the other; each launches right,
    and the module's compile cache compiles each variant once."""
    import intj.launcher as launcher

    monkeypatch.setattr(launcher, "_LOADED", {})
    real, compiles = launcher._checked_compile, []

    def counted(*args, **kwargs):
        compiles.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(launcher, "_checked_compile", counted)
    first, second = make_launcher(act_store), make_launcher(act_store)
    o = torch.zeros(1, device="cuda")
    second(*_gpu_controls(), o, 2.0, "neg")  # id 0 = "neg" in second
    first(*_gpu_controls(), o, 2.0, "pos")  # id 0 = "pos" in first
    for launch, act, want in (
        (first, "neg", -2.0), (second, "pos", 2.0), (first, "pos", 2.0), (second, "neg", -2.0)
    ):
        launch(*_gpu_controls(), o, 2.0, act)
        torch.cuda.synchronize()
        assert o.item() == want, (act, want)
    assert module_of(first) is module_of(second) and len(compiles) == 2  # one per variant


def test_unsupported_object_constexpr_names_the_parameter():
    launch = make_launcher(act_store)
    o = torch.zeros(1, device="cuda")

    class Name(str):
        pass

    with pytest.raises(TypeError, match="unsupported argument 'ACT' of type list"):
        launch(*_gpu_controls(), o, 2.0, ["neg"])
    with pytest.raises(TypeError, match="'ACT' of type Name"):
        launch(*_gpu_controls(), o, 2.0, Name("neg"))
    launch(*_gpu_controls(), o, 2.0, "neg")  # still usable
    torch.cuda.synchronize()
    assert o.item() == -2.0
```

- [ ] **Step 2: Run them to verify they fail**

Run: `... pytest tests/test_runtime.py tests/test_launcher.py -q -k "constexpr or memo or intern"`
Expected: FAIL. `init_bound` takes no interner, `Param` has no `memo`, and a `str` constexpr raises `TypeError: … pass a scalar int, float, bool or None`.

- [ ] **Step 3: Runtime helpers in `intj_runtime.h`**

- After `#define INTJ_B_U32 152u`, add `#define INTJ_B_CX_OBJECT 154u /* id of a str / tl.dtype / JIT function */` and change the comment to `155..255 unused`.
- Add `INTJ_VALUE_OBJECT` as the last `intj_value_kind`.
- In `intj_infer_type`, add `case INTJ_VALUE_OBJECT:` to the `INTJ_VALUE_NONE` case (arguments never decode objects; `-Wswitch` needs the case).
- In `intj_constexpr_type`, add `case INTJ_VALUE_OBJECT: return INTJ_B_CX_OBJECT;` before `default`.
- After `intj_decode_constexpr`, add:

```c
/* An object value's id: one pointer compare when this slot saw the same object
 * last, else the module's interner (Python, cold).  The memo holds a strong
 * reference, so a memoized object's address cannot come back as another value.
 * Free-threaded, the (obj, id) pair is read under the module lock's read side:
 * a writer replaces both, and two plain loads could pair them wrongly.  With
 * the GIL the lock compiles away. */
static inline int intj_object_id_slow(PyObject *intern, intj_rwlock *lock, intj_memo *memo,
                               PyObject *o, const char *pname, uint64_t *id) {
  if (!intern) {
    PyErr_Format(PyExc_RuntimeError, "intj: no intern table for '%s'; this is an intj bug", pname);
    return -1;
  }
  PyObject *r = PyObject_CallFunctionObjArgs(intern, o, NULL);
  if (!r)
    return -1;
  if (r == Py_None) {
    Py_DECREF(r);
    PyErr_Format(PyExc_TypeError,
                 "intj: unsupported argument '%s' of type %s; pass a scalar int, float, "
                 "bool or None, a str, a tl.dtype or a @triton.jit function",
                 pname, Py_TYPE(o)->tp_name);
    return -1;
  }
  uint64_t value = PyLong_AsUnsignedLongLong(r);
  Py_DECREF(r);
  if (value == (uint64_t)-1 && PyErr_Occurred())
    return -1;
  PyObject *old = NULL;
  if (memo) {
    Py_INCREF(o);
    INTJ_WRLOCK(lock);
    old = memo->obj;
    memo->obj = o;
    memo->id = value;
    INTJ_RWUNLOCK(lock);
  }
  Py_XDECREF(old); /* outside the lock: a finalizer can run Python */
  *id = value;
  return 0;
}

static INTJ_ALWAYS_INLINE int intj_object_id(PyObject *intern, intj_rwlock *lock,
                                             intj_memo *memo, PyObject *o,
                                             const char *pname, uint64_t *id) {
  if (memo) {
    INTJ_RDLOCK(lock);
    int hit = memo->obj == o;
    uint64_t value = memo->id;
    INTJ_RWUNLOCK(lock);
    if (INTJ_LIKELY(hit)) {
      *id = value;
      return 0;
    }
  }
  return intj_object_id_slow(intern, lock, memo, o, pname, id);
}

/* A constexpr or dynamic value: scalars as intj_decode_constexpr, anything
 * else as an object id.  `memo` is NULL without a launcher (spec_key). */
static INTJ_ALWAYS_INLINE int intj_decode_value(PyObject *o, const char *pname,
                                                PyObject *intern, intj_rwlock *lock,
                                                intj_memo *memo, intj_decoded *out) {
  PyTypeObject *type = Py_TYPE(o);
  if (type == &PyLong_Type || type == &PyFloat_Type || type == &PyBool_Type || o == Py_None)
    return intj_decode_constexpr(o, pname, out);
  memset(out, 0, sizeof(*out));
  out->kind = INTJ_VALUE_OBJECT;
  return intj_object_id(intern, lock, memo, o, pname, &out->bits);
}
```

- [ ] **Step 4: Template**

1. After `#define INTJ_NBOUND …`, add `#define INTJ_NMEMO {{ params | rejectattr('memo', 'none') | list | length }}`.
2. Every object decode reads the launcher's table, `bound ? bound->intern : NULL`, never module state.
3. In `decode_auto_constexpr`, replace the final `else { PyErr_Format(...); return -1; }` with the code below. In the auto path every constexpr is untyped and public, so it always has a memo.

```jinja
    } else {
      code = INTJ_B_CX_OBJECT;
      if (intj_object_id(bound ? bound->intern : NULL, &bound->lock,
                         bound ? &INTJ_TAIL(bound)->memo[{{ p.memo }}] : NULL,
                         o, intj_param_names[{{ p.index }}], &bits) != 0)
        return -1;
    }
```

4. In the generic path, replace the `{% elif a.kind == 'constexpr' %}` decode (378–380) with:

```jinja
{% elif a.kind == 'constexpr' and p.memo is not none %}
  if (intj_decode_value(args[{{ p.call_index }}], intj_param_names[{{ i }}],
                        bound ? bound->intern : NULL, &bound->lock,
                        bound ? &INTJ_TAIL(bound)->memo[{{ p.memo }}] : NULL, &{{ d }}) != 0)
    return -1;
{% elif a.kind == 'constexpr' %}
  if (intj_decode_constexpr(args[{{ p.call_index }}], intj_param_names[{{ i }}], &{{ d }}) != 0)
    return -1;
```

5. In `intj_bound_traverse`, visit the memos: `for (int k = 0; k < INTJ_NMEMO; k++) Py_VISIT(tail->memo[k].obj);`. In `intj_bound_clear`, clear them: `for (int k = 0; k < INTJ_NMEMO; k++) Py_CLEAR(tail->memo[k].obj);`.
5b. Visit and clear the table in the hooks as well: `Py_VISIT(bound->intern);` and `Py_CLEAR(bound->intern);`.
6. `init_bound` takes the launcher's interner as its third argument, after the callback:
   - make the count check `3 + {{ bound_arg_offset + grid_py_offset }} + INTJ_NBOUND` in both places;
   - after the callback check, add `PyObject *interner = args[2]; if (!PyCallable_Check(interner)) { PyErr_SetString(PyExc_TypeError, "intj: init_bound needs the launcher's interner"); return NULL; }`;
   - change `args += 2;` to `args += 3;`;
   - inside `try {`, after `bound->compile_cb = …`, add `bound->intern = Py_NewRef(interner);`.
   The `fail:` path's `intj_bound_clear` drops it again.
7. `spec_key` needs nothing more: it already takes the launcher (Task 2), so an object value decodes with that launcher's table and memos.

- [ ] **Step 5: Python**

`Param`: add `memo: int | None = None  # this slot's intj_memo, if it takes objects`.

Add after `_render_params`:

```python
def _object_capable(annotation: CanonicalAnnotation) -> bool:
    """A public constexpr keyed by descriptor + 8-byte payload takes objects:
    str, tl.dtype and JIT functions key as INTJ_B_CX_OBJECT plus an id."""
    return (
        annotation.kind == "constexpr"
        and annotation.types is None
        and not annotation.power_of_two_or_zero
    )


class _Interner:
    """One launcher's object values -> small ids, by canonical value.

    Per launcher, like its kernel cache: an id means one value within one
    launcher and nothing outside it.  Ids live for the process and never enter
    a `ModuleKey`.  A JIT function's `cache_key` is read once, when its object
    is first seen, as triton does."""

    def __init__(self) -> None:
        self._ids: dict[tuple[object, ...], int] = {}
        self._lock = threading.Lock()

    def __call__(self, value: object) -> int | None:
        try:
            canonical = _canonical_value(value)
        except ValueError:
            return None  # C raises the TypeError, naming the parameter
        found = self._ids.get(canonical)
        if found is None:
            with self._lock:  # two new values must not both take len(_ids)
                found = self._ids.setdefault(canonical, len(self._ids))
        return found
```

In `_render_key`'s params loop, assign memo slots in call order:

```python
    params: list[Param] = []
    call_index = 0
    memo = 0
    for p in resolved:
        annotation = ...  # unchanged
        public = ...      # unchanged
        slot = None
        if public and _object_capable(annotation):
            slot, memo = memo, memo + 1
        params.append(
            Param(p.name, p.index, call_index if public else None, annotation, slot)
        )
        if public:
            call_index += 1
```

- In `LauncherFactory._bind`, pass `_tail_bytes(sum(p.memo is not None for p in params), len(bound_values))`.
- In `LauncherFactory._build`, pass a fresh table after the callback: `return module.init_bound(header, callback, _Interner(), *rest, *args)`.
- `_make_compile_callback` is unchanged. Its `seen` check is per launcher (Task 2), so it covers object values as well.

- [ ] **Step 6: Run the tests**

```
PYTHONPATH=$PWD VIRTUAL_ENV=$V uv run --active --no-sync python -m pytest tests -q
bash tests/run_python_matrix.sh 3.8 3.13t 3.14t
PYTHONPATH=$PWD VIRTUAL_ENV=$V uv run --active --no-sync pyright
```

Expected: all tests pass, including the Step 1 tests and the memo refcount test on the free-threaded builds, and pyright reports 0 errors.

- [ ] **Step 7: Docs**

In `docs/Usage.md` "Supported arguments", change the `tl.constexpr` row to: "the value itself (`int`, `float`, `bool`, `None`); a `str`, `tl.dtype` or `@triton.jit` function keys by value (see below)". Then add:

```markdown
An untyped, unbaked `tl.constexpr` also takes a `str`, a `tl.dtype`, or a
`@triton.jit` function, keyed by value, as Triton does: two equal strings share
one kernel, and two JIT functions with the same source share one kernel. Each value gets a
small id from a table owned by the loaded module (never persisted, never in the
module digest); each argument slot remembers its last object, so passing the
same object again costs one pointer compare. A JIT function's `cache_key` is
read when the function is first seen, so later edits to a callee are not seen
-- Triton has the same limitation. Baking the value with `extra_annotation`
still works and removes it from the call.
```

In "Not supported", change the bullet "Tuple, `tl.constexpr` object, `TensorDescriptor`, JIT-function and string arguments." to "Tuple and `TensorDescriptor` arguments, and `str` / `tl.dtype` / JIT-function values for anything but an untyped `tl.constexpr`."

- [ ] **Step G: Benchmark gate**

```bash
/tmp/intj-bench/run_all.sh lazy-task3
/tmp/gb2/bin/python /tmp/intj-bench/bench_compare.py lazy-baseline lazy-task3
```

Expected: no row regresses. The auto-decode path gained only a cold branch after the `None` test.

- [ ] **Step 8: Commit**

```bash
git add intj/runtime/intj_runtime.h intj/runtime/entry.c.jinja intj/launcher.py \
  tests/test_runtime.py tests/test_launcher.py docs/Usage.md benchmarks/journals/*_lazy-task3_0_*.md
git commit -m "Key str, tl.dtype and JIT-function constexprs by value

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: `dynamic_options` for compile options and knobs

About 1 day.

> **Amended after implementation (Task 4b).** Declared knobs only key: intj never sets a
> knob, so `_knob_scope` below no longer exists. On a miss (untuned and tuned) each
> declared knob's passed value must equal the live knob, else `ValueError` and no record;
> the compile runs under the live knobs. The knob values stay in both compile-cache keys.
> The knob tests below were replaced accordingly (see the Task 4 report).

**Files:**
- Modify: `intj/launcher.py`:
  - new `DynamicSlot` after `TuningRender`;
  - `RenderContext` (135–184) gets `dynamic`;
  - `LauncherFactory` gets the `dynamic` field;
  - `_validate_grid_kwargs` 377–380;
  - `make_launcher` (docstring, `_check_dynamic`, `compile_grid(..., offset=)`);
  - `_render_key`, `_materialize_module`, `_loaded_module`, `_make_compile_callback`;
  - new `_dynamic_key_fields`, `_check_dynamic`, `_split_dynamic`, `_knob_scope`
- Modify: `intj/grid.py:34-44,160-166` (`offset`)
- Modify: `intj/tuning.py` (`make_tuned_callback`: `dynamic`, knob scope)
- Modify: `intj/runtime/entry.c.jinja`: `intj_pack` signature and dynamic decode; `intj_parse_call` nargs; `intj_call` argument pointers; `intj_fill`; `intj_tuned_fill`; untuned `grid_py` meta; `spec_key`; `key_chain`; the `entry` call
- Test: `tests/test_launcher.py`, `tests/test_tuned.py`
- Docs: `docs/Usage.md`

**Interfaces:**
- Consumes: `intj_decode_value`, `init_bound(header, callback, interner, ...)`, memos (Task 3); `spec_key(launcher, …)`, the module compile function and `_launcher_compile` (Task 2); `_knob_options`, `_live_knobs`, `_KNOB_OPTIONS` (Task 2)
- Produces, Python:
  - `DynamicSlot(name: str, value_offset: int | None, kind_offset: int | None, memo: int)`
  - `RenderContext.dynamic: tuple[DynamicSlot, ...] = ()`, `LauncherFactory.dynamic: tuple[str, ...] = ()`
  - `_dynamic_key_fields(name: str) -> tuple[KeyField, ...]`
  - `_split_dynamic(names, values) -> tuple[dict[str, object], dict[str, object]]`
  - `_knob_scope(values: Mapping[str, object]) -> ContextManager[None]`
  - `_render_key(...) -> (params, device_offset, nwords, computed_fields, dynamic_slots)`
  - `compile_grid(fn, params, baked, deps=None, offset=0)`
- Produces, call layout: `launch(device, stream, *grid controls, *dynamic values in declared order, *public args)`. The compile and tuned callbacks receive `(..., *dynamic values, *public args)`. `spec_key(launcher, device, *dynamic, *public)` and `key_chain(launcher, device, *dynamic, *public)`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_launcher.py`:

```python
def _hip():
    from triton.runtime.driver import driver

    return driver.active.get_current_target().backend == "hip"


def test_dynamic_num_warps_keys_separate_records():
    launch = make_launcher(act_store, dynamic_options=("num_warps",), return_compiled=True)
    o = torch.zeros(1, device="cuda")
    two = launch(*_gpu_controls(), 2, o, 1.0, "pos")
    four = launch(*_gpu_controls(), 4, o, 1.0, "pos")
    assert two.metadata.num_warps == 2 and four.metadata.num_warps == 4
    assert launch(*_gpu_controls(), 2, o, 1.0, "pos") is two
    assert tuple(inspect.signature(launch).parameters) == (
        "device", "stream", "grid", "num_warps", "o", "x", "ACT",
    )
    with pytest.raises(AssertionError):  # parse_options: not a power of two, raised on this miss
        launch(*_gpu_controls(), 3, o, 1.0, "pos")


def test_declared_knob_is_set_for_the_compile_and_restored(monkeypatch):
    import intj.launcher as launcher
    from triton import knobs

    before = knobs.runtime.debug
    seen, real = [], launcher._checked_compile

    def spy(*args, **kwargs):
        seen.append(knobs.runtime.debug)
        if len(seen) == 1:
            raise RuntimeError("compile failed")
        return real(*args, **kwargs)

    monkeypatch.setattr(launcher, "_checked_compile", spy)
    launch = make_launcher(
        act_store, dynamic_options=("knobs.runtime.debug",), return_compiled=True
    )
    o = torch.zeros(1, device="cuda")
    with pytest.raises(RuntimeError, match="compile failed"):
        launch(*_gpu_controls(), not before, o, 1.0, "pos")
    assert knobs.runtime.debug == before
    kernel = launch(*_gpu_controls(), not before, o, 1.0, "pos")
    assert seen == [not before, not before] and knobs.runtime.debug == before
    assert kernel.metadata.debug == (act_store.debug or not before)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"dynamic_options": ("knobs.nope.x",)}, "unknown knob"),
        ({"dynamic_options": ("knobs.runtime.nope",)}, "unknown knob"),
        ({"dynamic_options": ("num_warps",), "options": {"num_warps": 4}}, "both"),
        ({"dynamic_options": ("knobs.runtime.debug",), "options": {"debug": True}}, "both"),
        ({"dynamic_options": ("num_warps", "num_warps")}, "repeats"),
        ({"dynamic_options": ("x",)}, "kernel parameter"),
        ({"dynamic_options": ("stream",)}, "not allowed"),
        ({"dynamic_options": ("num_warps",), "no_gpu": True}, "GPU"),
    ],
)
def test_dynamic_option_refusals(kwargs, match):
    with pytest.raises(UnsupportedKernel, match=match):
        make_launcher(act_store, **kwargs)


def test_unknown_dynamic_option_is_refused_on_the_first_call():
    launch = make_launcher(act_store, dynamic_options=("num_warpz",))
    with pytest.raises(UnsupportedKernel, match="unknown compile option"):
        launch(*_gpu_controls(), 4, torch.zeros(1, device="cuda"), 1.0, "pos")
```

Append to `tests/test_tuned.py`:

```python
def test_dynamic_option_clashing_with_a_config_is_refused():
    kernel = triton.autotune(configs=configs(), key=["n"])(tagged)
    with pytest.raises(UnsupportedKernel, match="set by the autotune configs"):
        make_launcher(kernel, grid_cpp=grid, dynamic_options=("num_warps",))


_needs_hip = pytest.mark.skipif(
    triton.runtime.driver.active.get_current_target().backend != "hip",
    reason="waves_per_eu is a HIP option",
)


@_needs_hip
def test_tuned_dynamic_option_reaches_every_config():
    kernel = triton.autotune(
        configs=configs(), key=["n"], do_bench=lambda call, quantiles: [call() or 1.0] * 3
    )(tagged)
    launch = make_launcher(kernel, grid_cpp=grid, dynamic_options=("waves_per_eu",), return_compiled=True)
    x = torch.zeros(256, device="cuda", dtype=torch.int32)
    out = torch.zeros_like(x)
    one = launch(*controls(), 1, x, out, 256)
    two = launch(*controls(), 2, x, out, 256)
    assert one.metadata.waves_per_eu == 1 and two.metadata.waves_per_eu == 2
    assert launch(*controls(), 1, x, out, 256) is one
```

- [ ] **Step 2: Run them to verify they fail**

Run: `... pytest tests/test_launcher.py tests/test_tuned.py -q -k "dynamic or declared_knob"`
Expected: FAIL with `UnsupportedKernel: intj: dynamic_options is not implemented`.

- [ ] **Step 3: Python — declaration, refusals, key layout**

```python
@dataclasses.dataclass(frozen=True)
class DynamicSlot:
    """One `dynamic_options` entry as `entry.c.jinja` renders it."""

    name: str  # compile option or `knobs.<group>.<name>`
    value_offset: int | None  # None only under an invariant test's mutation
    kind_offset: int | None
    memo: int  # its intj_memo in the launcher's trailing arrays
```

- `RenderContext`: add `dynamic: tuple[DynamicSlot, ...] = ()`.
- `LauncherFactory`: add the trailing field `dynamic: tuple[str, ...] = ()`.

`_validate_grid_kwargs`: replace the `if dynamic_options:` refusal with:

```python
    if isinstance(dynamic_options, str):
        raise TypeError("intj: dynamic_options is a sequence of names, not one string")
    if dynamic_options and no_gpu:
        raise UnsupportedKernel("intj: dynamic_options needs GPU mode; no_gpu=True compiles nothing")
```

Add:

```python
def _dynamic_key_fields(name: str) -> tuple[KeyField, ...]:
    """One dynamic value's key: a kind byte and 8 value bytes, like an exact
    key.  `name` picks nothing here; the invariant tests' mutation checks drop
    one value's fields by name."""
    del name
    return (KeyField("exact_kind", 1), KeyField("exact", 8))


def _check_dynamic(
    dynamic: tuple[str, ...],
    options: Mapping[str, Any],
    kernel: JitFunction,
    tuned: _Tuned | None,
) -> None:
    """GPU-free refusals for `dynamic_options`.  Unknown compile-option names
    wait for the target, in `_materialize_module`."""
    from triton import knobs

    if len(set(dynamic)) != len(dynamic):
        raise UnsupportedKernel(f"intj: dynamic_options repeats a name: {list(dynamic)}")
    for name in dynamic:
        if type(name) is not str:
            raise TypeError(f"intj: dynamic_options names must be str, got {name!r}")
        if name.startswith("knobs."):
            parts = name.split(".")
            group = getattr(knobs, parts[1], None) if len(parts) == 3 else None
            attr = (
                type(group).__dict__.get(parts[2])
                if isinstance(group, knobs.base_knobs)
                else None
            )
            if not (isinstance(attr, knobs.env_base) or type(attr) in (bool, int, float, str)):
                raise UnsupportedKernel(
                    f"intj: unknown knob {name!r}; expected knobs.<group>.<name>, "
                    "e.g. knobs.runtime.debug"
                )
        elif name in ("device", "stream", "device_type", "warp_size"):
            raise UnsupportedKernel(f"intj: option {name!r} is not allowed")
        elif name in kernel.arg_names:
            raise UnsupportedKernel(f"intj: dynamic option {name!r} is a kernel parameter")
    derived = dict(_KNOB_OPTIONS)
    both = sorted(n for n in dynamic if n in options or derived.get(n) in options)
    if both:
        raise UnsupportedKernel(
            f"intj: {both} given both in options= and dynamic_options; "
            "a value is fixed or per call, not both"
        )
    owned = sorted(set(dynamic) & tuned.plan.tuned) if tuned is not None else []
    if owned:
        raise UnsupportedKernel(f"intj: dynamic_options {owned} are set by the autotune configs")


def _split_dynamic(
    names: Sequence[str], values: Sequence[object]
) -> tuple[dict[str, object], dict[str, object]]:
    """(compile options, knob paths) of one call's dynamic values."""
    pairs = list(zip(names, values))
    return (
        {n: v for n, v in pairs if not n.startswith("knobs.")},
        {n: v for n, v in pairs if n.startswith("knobs.")},
    )


@contextlib.contextmanager
def _knob_scope(values: Mapping[str, object]) -> Iterator[None]:
    """Set declared knobs to one call's values for one compile, then restore
    them, through triton's own per-group `scope()`."""
    from triton import knobs

    with contextlib.ExitStack() as stack:
        for group in sorted({path.split(".")[1] for path in values}):
            stack.enter_context(getattr(knobs, group).scope())
        for path, value in values.items():
            _, group, name = path.split(".")
            setattr(getattr(knobs, group), name, value)
        yield
```

In `make_launcher`:
- after `_plan_tuning`/`resolved` and before `compile_grid`, add `dynamic = tuple(dynamic_options)` and `_check_dynamic(dynamic, options, kernel, tuned)`;
- pass `offset=len(dynamic)` to `compile_grid`;
- pass `dynamic=dynamic` to `LauncherFactory(...)` as a keyword.
- Rewrite the docstring's "`dynamic_grid` and `dynamic_options` are reserved and unsupported." as: "`dynamic_options` names compile options and `knobs.<group>.<name>` paths passed per call, right after the grid controls, and keyed; `dynamic_grid` is reserved."

`_render_key(resolved, device_binding, computed0=0, dynamic=())`:
- after the computed fields, add the block below;
- after `computed_fields`, build the slots as shown;
- return `(tuple(params), layout.device_offset, layout.nwords, computed_fields, slots)`.
- `_render_params` takes the first three.

```python
    fields += tuple(
        (-1 - computed0 - j, field)
        for j, name in enumerate(dynamic)
        for field in _dynamic_key_fields(name)
    )
```

```python
    slots = tuple(
        DynamicSlot(
            name,
            placed.get((-1 - computed0 - j, "exact")),
            placed.get((-1 - computed0 - j, "exact_kind")),
            memo + j,
        )
        for j, name in enumerate(dynamic)
    )
```

`_materialize_module(..., knob_values=None, dynamic: tuple[str, ...] = ())`:
- unpack five values from `_render_key(resolved, device_binding, computed0, dynamic)`;
- set `dynamic=slots` on the `RenderContext`;
- in the GPU branch, after `canonical_options`, add the check below;
- `_loaded_module(key, jit_func, context, params, options, layout, baked_values, knob_values=None, dynamic: tuple[str, ...] = ())` takes it and forwards it as `_make_compile_callback(..., dynamic=dynamic)`.

```python
        fields = {f.name for f in dataclasses.fields(canonical_options)}
        unknown = sorted(n for n in dynamic if not n.startswith("knobs.") and n not in fields)
        if unknown:
            raise UnsupportedKernel(f"intj: unknown compile option(s) {unknown}")
```

`LauncherFactory`:
- `_bind` passes `dynamic=self.dynamic` to `_launch_doc`, and `_tail_bytes(sum(p.memo is not None for p in params) + len(self.dynamic), len(bound_values))`;
- `_build` passes `dynamic=self.dynamic` to `_materialize_module` and as the last argument of `make_tuned_callback`.

- [ ] **Step 4: Python — the miss paths**

`_make_compile_callback(..., knob_values, dynamic: Sequence[str] = ())`. In the Task 2 body, replace everything from `canonical_options = …` through the `with lock:` block with the code below. Options are now canonicalized per record.

```python
    compiled: MutableMapping[tuple[object, ...], CompiledKernel] = (
        weakref.WeakValueDictionary() if return_compiled else {}
    )
    lock = threading.Lock()

    def compile_callback(
        seen: dict[bytes, object], keyblob: bytes, nparams: int, device: int, *args: Any
    ) -> tuple[int, int, int, int] | tuple[int, int, int, int, CompiledKernel]:
        current = _current_device()
        if device != current:
            raise UnsupportedKernel(...)  # unchanged message
        values, args = args[: len(dynamic)], args[len(dynamic) :]
        dyn_options, dyn_knobs = _split_dynamic(dynamic, values)
        canonical = _canonical_options(
            target,
            _knob_options(jit_func, {**options, **dyn_options}, {**knob_values, **dyn_knobs}),
        )
        compiler_input = _compiler_input(jit_func, params, args, backend, baked_values=baked_values)
        identity = (compiler_input, canonical.hash(), tuple(sorted(dyn_knobs.items())))
        if seen.setdefault(keyblob, identity) != identity:  # per launcher, every call
            raise RuntimeError(
                "intj: one spec key maps to two annotated ASTSource inputs; this is an intj bug"
            )
        cache_key = (*identity, current)
        with lock:
            kernel = compiled.get(cache_key)
            if kernel is None:
                with _knob_scope(dyn_knobs):
                    kernel = compiled[cache_key] = _checked_compile(
                        jit_func, compiler_input, target, canonical
                    )
        # <from `md = kernel.metadata` on: unchanged>
```

`intj/grid.py` `compile_grid(fn, params, baked, deps=None, offset=0)`: document `offset` as "dynamic values between the grid extras and the public arguments", and change line 161 to `index = len(extras) + offset + param.call_index`.

`intj/tuning.py` `make_tuned_callback(..., knob_values, dynamic: Sequence[str] = ())`:
- import `_knob_scope` and `_split_dynamic`;
- add a closure `call_knobs: list[dict[str, object]] = [{}]` (misses are serialized by `lock`);
- in `compile_kernel`: `_knob_options(jit_func, {**options, **config_options}, {**knob_values, **call_knobs[0]})`;
- in `tune`, split first:

```python
        dyn_values, args = args[: len(dynamic)], args[len(dynamic) :]
        dyn_options, dyn_knobs = _split_dynamic(dynamic, dyn_values)
        call_knobs[0] = dyn_knobs
```

and run the chain as

```python
            with torch.cuda.stream(torch.cuda.ExternalStream(stream, device=device)), _knob_scope(dyn_knobs):
                private[0].run(*prefix, grid=triton_grid, warmup=False, **named, **dyn_options)
```

Autotune passes `dyn_options` through to the shim as extra kwargs. There they merge into `config_options` for every config: this is "the shim merges option values with each config's options", and `_check_dynamic` already refused overlap. Reset `call_knobs[0] = {}` in the existing `finally`.

- [ ] **Step 5: Template — argument layout and keying**

1. Add `{% set ndyn = dynamic | length %}` beside `ncontrols`, and `#define INTJ_NDYN {{ ndyn }}` beside `INTJ_NPARAMS`. Change `INTJ_NMEMO` to `{{ (params | rejectattr('memo', 'none') | list | length) + ndyn }}`.
2. Signature: `intj_pack(intj_state *st, PyObject *const *dyn, PyObject *const *args, uint64_t *key, …)`. Before `{% if device_offset is not none %}`, add:

```jinja
{% for slot in dynamic %}
  {
    intj_decoded dv;
    if (intj_decode_value(dyn[{{ loop.index0 }}], "{{ slot.name }}", bound ? bound->intern : NULL, &bound->lock,
                          bound ? &INTJ_TAIL(bound)->memo[{{ slot.memo }}] : NULL, &dv) != 0)
      return -1;
{% if slot.value_offset is not none %}
{{ key_accumulate({'offset': slot.kind_offset, 'width': 1}, 'dv.kind') }}
{{ key_accumulate({'offset': slot.value_offset, 'width': 8}, 'dv.bits') }}
{% else %}
    (void)dv;
{% endif %}
  }
{% endfor %}
{% if not ndyn %}  (void)dyn;
{% endif %}
```

3. `intj_parse_call`:
   - the count becomes `{{ ncontrols }} + INTJ_NDYN + INTJ_NPARAMS` in the check and in the message;
   - after `"%d kernel arguments"`, add ` and %d dynamic values`, passing `INTJ_NDYN`.
4. `intj_call`: after `(void)nargs;`, add the two pointers below. Then:
   - `intj_pack(st, dyn, kargs, key, vals, &np, …)`;
   - `intj_walk(st, bound, kargs, …)`;
   - `intj_fill(st, bound, key, np, device_ordinal, dyn)`;
   - the untuned `grid_py` meta reads `kargs[{{ p.call_index }}]`;
   - `intj_tuned_grid_py_launch(st, bound, &copy, stream, kargs, vals, np)`.

```c
  /* args: grid controls, then dynamic values, then the public kernel arguments */
  PyObject *const *dyn = args + {{ grid_count }};
  PyObject *const *kargs = dyn + INTJ_NDYN;
```

5. `intj_fill`: rename `kernel_args` to `call_args` (dynamic then public). Size `cb_args[3 + INTJ_NDYN + INTJ_NPARAMS]`, loop `i < INTJ_NDYN + INTJ_NPARAMS`, and vectorcall with that count.
6. `intj_tuned_fill`:
   - size `cb_args[5 + INTJ_NDYN + INTJ_NPARAMS]`, fill `cb_args[5 + i] = args[{{ grid_count }} + i]` for `i < INTJ_NDYN + INTJ_NPARAMS`, and use that count in the vectorcall;
   - the level functions read `args + {{ grid_count }} + INTJ_NDYN`.
7. `spec_key`: `nargs != 2 + INTJ_NDYN + INTJ_NPARAMS` (in the message too) and `intj_pack(st, args + 2, args + 2 + INTJ_NDYN, …)`.
8. `key_chain`:
   - `nargs != 2 + INTJ_NDYN + INTJ_NPARAMS`;
   - `intj_pack(st, args + 2, args + 2 + INTJ_NDYN, …)`;
   - `intj_walk(st, bound, args + 2 + INTJ_NDYN, …)`.
9. There is no module `entry` any more (Task 2), so nothing else calls `intj_call`.

- [ ] **Step 6: Run the tests and pyright**

```
PYTHONPATH=$PWD VIRTUAL_ENV=$V uv run --active --no-sync python -m pytest tests -q
PYTHONPATH=$PWD VIRTUAL_ENV=$V uv run --active --no-sync pyright
bash tests/run_python_matrix.sh 3.8 3.14t
```

Expected: all tests pass, and pyright reports 0 errors. Without dynamic values the rendered source differs only by the `(void)dyn;` lines and the `INTJ_NDYN` term.

- [ ] **Step 7: Docs**

In `docs/Usage.md`:
- change the signature block line 11 to `dynamic_options=(),    # compile options / knobs.<group>.<name> passed per call`;
- in "Not supported", reduce the `dynamic_grid` bullet to "`dynamic_grid=True`. Use `grid_cpp` or `grid_py` for a callable grid.";
- add after "Calling the launcher":

````markdown
### Per-call compile options and knobs: `dynamic_options`

```python
@intj.make_launcher(grid_cpp=grid, dynamic_options=("num_warps", "knobs.runtime.debug"))
@triton.jit
def k(x, out, n, ACT: tl.constexpr, BLOCK: tl.constexpr): ...

k(device, stream, 8, False, x, out, n, "gelu", 128)   # num_warps=8, debug off
```

Each name becomes a positional argument right after the grid controls, in
the declared order, and its value is part of the key: an `int`, `bool` or
`float` by value and kind, a `str` (e.g. `knobs.compilation.instrumentation_mode`)
through the same id table as object constexprs. A name is either a compile
option (`num_warps`, `num_stages`, `waves_per_eu`, ...) or a Triton knob path
`knobs.<group>.<name>` (`knobs.runtime.debug`, `knobs.amd.use_buffer_ops`, ...).
On a miss, options merge into `options=` and go through `parse_options` per
kernel (an invalid value such as `num_warps=3` raises on that call); knobs are
set to the call's values with Triton's `knobs` scope for the duration of the
compile, then restored. With `triton.autotune`, the values reach every config.

Refused at `make_launcher` with `UnsupportedKernel`: an unknown knob path, a
repeated name, a name also given in `options=` (for knobs, the option the knob
feeds, e.g. `debug`), a kernel parameter name, `device`/`stream`/`device_type`/
`warp_size`, a name an autotune config sets, and `no_gpu=True`. An unknown
compile-option name raises on the first call, where the target is known.

Knobs you do not declare are read once, at the first call, and fixed for that
launcher (`debug`, `instrumentation_mode`, fpsan casts feed its options). Other
cache-invalidating knobs are not keyed; see `TODO.md`.
````

- [ ] **Step G: Benchmark gate**

```bash
/tmp/intj-bench/run_all.sh lazy-task4
/tmp/gb2/bin/python /tmp/intj-bench/bench_compare.py lazy-baseline lazy-task4
```

Expected: no row regresses. No benchmarked launcher declares dynamic values.

- [ ] **Step 8: Commit**

```bash
git add intj/launcher.py intj/grid.py intj/tuning.py intj/runtime/entry.c.jinja \
  tests/test_launcher.py tests/test_tuned.py docs/Usage.md benchmarks/journals/*_lazy-task4_0_*.md
git commit -m "Key compile options and Triton knobs per call with dynamic_options

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Invariant tests, bench rows, TODO

About half a day.

**Files:**
- Modify: `tests/test_launcher.py`, `tests/test_tuned.py` (invariants and mutation checks)
- Modify: `benchmarks/bench_launch.py` (`--dynamic`), `benchmarks/AGENTS.md`, `TODO.md`

**Interfaces:**
- Consumes: `spec_key(launcher, device, *dynamic, *public)`, `key_chain(launcher, device, *dynamic, *public)` (tuned), `_dynamic_key_fields`, `_Interner`, `_knob_options`, `_live_knobs`, `_split_dynamic`, `module_of`

- [ ] **Step 1: Untuned invariant with mutation checks**

Append to `tests/test_launcher.py`:

```python
@triton.jit
def dyn_axpy(o, x, n, ACT: tl.constexpr, DT: tl.constexpr, FN: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    v = FN(tl.load(x + offs, mask=offs < n)).to(DT)
    if ACT == "neg":
        v = -v
    tl.store(o + offs, v, mask=offs < n)


_DYNAMIC = ("num_warps", "knobs.runtime.debug")


def check_dynamic_invariant():
    """Same key => same annotated ASTSource, same canonical options, same knobs."""
    from triton.compiler import make_backend

    from intj.annotation import DeviceBinding, _resolve_annotations
    from intj.launcher import (
        _canonical_options, _compiler_input, _current_target, _knob_options, _live_knobs,
        _render_params,
    )

    launch = make_launcher(dyn_axpy, dynamic_options=_DYNAMIC)
    module, device = module_of(launch), torch.cuda.current_device()
    params, _, _ = _render_params(_resolve_annotations(dyn_axpy, None), DeviceBinding.NOT_FIXED)
    target = _current_target()
    backend, knobs_now = make_backend(target), _live_knobs()
    x = torch.randn(64, device="cuda")
    o = torch.empty_like(x)
    seen = {}
    for warps in (2, 4):
        for debug in (False, True):
            for act in ("neg", "".join(["n", "e", "g"]), "pos"):
                for dt in (tl.float32, tl.float16):
                    for fn in (_twice, _thrice, triton.jit(_twice.fn)):
                        public = (o, x, 64, act, dt, fn, 64)
                        key, _ = module.spec_key(launch, device, warps, debug, *public)
                        options = _knob_options(
                            dyn_axpy, {"num_warps": warps}, {**knobs_now, "knobs.runtime.debug": debug}
                        )
                        truth = (
                            _compiler_input(dyn_axpy, params, public, backend),
                            _canonical_options(target, options).hash(),
                        )
                        assert seen.setdefault(key, truth) == truth, (
                            "intj key collides for dynamic values"
                        )


def test_dynamic_values_are_never_coarser_than_triton():
    check_dynamic_invariant()


@pytest.mark.parametrize("dropped", _DYNAMIC)
def test_dynamic_invariant_catches_a_dropped_value(monkeypatch, dropped):
    original = launcher._dynamic_key_fields
    monkeypatch.setattr(
        launcher, "_dynamic_key_fields", lambda name: () if name == dropped else original(name)
    )
    with pytest.raises(AssertionError, match="collides"):
        check_dynamic_invariant()


def test_dynamic_invariant_catches_a_dropped_object_id(monkeypatch):
    monkeypatch.setattr(launcher._Interner, "__call__", lambda self, value: 0)
    with pytest.raises(AssertionError, match="collides"):
        check_dynamic_invariant()
```

What each mutation breaks: dropping `num_warps` collides the 2/4 pairs, whose options hashes differ. Dropping the debug knob collides debug on and off. A constant interner makes `"neg"`/`"pos"`, the two dtypes and the two JIT callees collide.

- [ ] **Step 2: Tuned invariant with dynamic values and an object constexpr**

In `tests/test_tuned.py`, generalize `_reference` and `check_tuned_invariant`. The old test keeps calling them with the defaults.

```python
def _reference(kernel, args_by_name, dynamic=None):
    """What Triton decides for these arguments and per-call options/knobs."""
    import copy

    from triton.runtime.autotuner import Autotuner

    from intj.launcher import _split_dynamic

    dynamic = dynamic or {}
    run_options, run_knobs = _split_dynamic(tuple(dynamic), tuple(dynamic.values()))
    # <layers/Record setup unchanged>
    # the caller set the live knobs to `run_knobs`: declared knobs only key
    layers[0].run(grid=(1,), warmup=False, **args_by_name, **run_options)
    # <tuning_keys, params, options, heuristics unchanged>
    return (
        repr(triton_specialization(inner, params, options)),
        tuple(tuning_keys),
        heuristics,
        tuple(sorted(run_knobs.items())),
    )


def check_tuned_invariant(make_kernel, cases, dynamic=(), dyn_values=((),)):
    from triton import knobs

    from intj.launcher import _split_dynamic

    launch = make_launcher(make_kernel(), dynamic_options=dynamic)
    device, stream = controls()
    seen, before = {}, {}
    try:
        for dyn in dyn_values:
            # a declared knob only keys: set the live value the call passes
            for path, value in _split_dynamic(dynamic, dyn)[1].items():
                _, group, name = path.split(".")
                before.setdefault((group, name), getattr(getattr(knobs, group), name))
                setattr(getattr(knobs, group), name, value)
            for case in cases:
                launch(device, stream, 1, *dyn, *case.values())
                chain, found = module_of(launch).key_chain(launch, device, *dyn, *case.values())
                assert found
                truth = _reference(make_kernel(), case, dict(zip(dynamic, dyn)))
                assert seen.setdefault(chain, truth) == truth, (
                    "intj key chain collides across Triton decisions"
                )
    finally:
        for (group, name), value in before.items():
            setattr(getattr(knobs, group), name, value)


@triton.jit
def aligned_act(x, out, N, stride, ACT: tl.constexpr, BLOCK: tl.constexpr, ALIGNED: tl.constexpr):
    v = tl.load(x) + BLOCK + ALIGNED
    if ACT == "neg":
        v = -v
    tl.store(out, v)


_TUNED_DYNAMIC = ("waves_per_eu", "knobs.runtime.debug")
_TUNED_DYN_VALUES = [(w, d) for w in (1, 2) for d in (False, True)]


def _dynamic_invariant_kernel():
    def bench(call, quantiles):
        return [1.0, 1.0, 1.0]

    return triton.autotune(
        configs=[triton.Config({"BLOCK": 32}), triton.Config({"BLOCK": 64})], key=["N"], do_bench=bench
    )(triton.heuristics({"ALIGNED": lambda a: a["stride"] % a["BLOCK"] == 0})(aligned_act))


def _dynamic_invariant_cases():
    x = torch.zeros(1, device="cuda", dtype=torch.int32)
    out = torch.zeros_like(x)
    return [
        {"x": x, "out": out, "N": n, "stride": 32, "ACT": act}
        for n in (16, 17)
        for act in ("neg", "".join(["n", "e", "g"]), "pos")
    ]


@_needs_hip  # defined in Task 4
def test_tuned_dynamic_values_are_never_coarser_than_triton():
    check_tuned_invariant(
        _dynamic_invariant_kernel, _dynamic_invariant_cases(), _TUNED_DYNAMIC, _TUNED_DYN_VALUES
    )


@_needs_hip
@pytest.mark.parametrize("dropped", _TUNED_DYNAMIC)
def test_tuned_invariant_catches_a_dropped_dynamic_value(monkeypatch, dropped):
    from intj import launcher as launcher_mod

    original = launcher_mod._dynamic_key_fields
    monkeypatch.setattr(
        launcher_mod, "_dynamic_key_fields", lambda name: () if name == dropped else original(name)
    )
    with pytest.raises(AssertionError, match="collides"):
        check_tuned_invariant(
            _dynamic_invariant_kernel, _dynamic_invariant_cases(), _TUNED_DYNAMIC, _TUNED_DYN_VALUES
        )


@_needs_hip
def test_tuned_invariant_catches_a_dropped_object_id(monkeypatch):
    from intj import launcher as launcher_mod

    monkeypatch.setattr(launcher_mod._Interner, "__call__", lambda self, value: 0)
    with pytest.raises(AssertionError, match="collides"):
        check_tuned_invariant(
            _dynamic_invariant_kernel, _dynamic_invariant_cases(), _TUNED_DYNAMIC, _TUNED_DYN_VALUES
        )
```

Import `module_of` from `intj.launcher` at the top of `test_tuned.py`.

- [ ] **Step 3: Run the invariants**

```
PYTHONPATH=$PWD VIRTUAL_ENV=$V uv run --active --no-sync python -m pytest tests/test_launcher.py tests/test_tuned.py -q -k "invariant or coarser"
```

Expected:
- the three "never coarser" tests pass;
- each mutation test passes, because it raises `AssertionError: … collides`;
- the existing `test_spec_key_is_never_coarser_than_triton` and `test_tuned_invariant_catches_a_dropped_exact_key` still pass.

If a mutation test fails with "DID NOT RAISE", the invariant test cannot fail for that value. Fix the cases, not the mutation.

- [ ] **Step 4: Bench rows**

In `benchmarks/bench_launch.py`, add the kernel and function below. Register `--dynamic` in the `modes` group (`help="warmed lazy launchers: plain, 2 dynamic options, str constexpr"`), add `options.dynamic` to the `--no-gpu` guard, and dispatch it:

```python
@triton.jit
def act_noop(x, o, n, ACT: tl.constexpr):
    off = tl.arange(0, 128)
    v = tl.load(x + off, mask=off < n)
    if ACT == "relu":
        v = tl.maximum(v, 0.0)
    tl.store(o + off, v, mask=off < n)


def bench_dynamic(iters, batches):
    """Warmed lazy launchers: plain, with two dynamic options, with a str constexpr."""
    x = torch.zeros(4096, device="cuda")
    device, stream = torch.cuda.current_device(), torch.cuda.current_stream().cuda_stream
    args = (x, x, x, 4096, 1.0, 128)
    rows = (
        ("lazy plain", make_launcher(noop), (device, stream, (1,), *args)),
        (
            "2 dynamic options",
            make_launcher(noop, dynamic_options=("num_warps", "knobs.runtime.debug")),
            (device, stream, (1,), 4, False, *args),
        ),
        ("str constexpr", make_launcher(act_noop), (device, stream, (1,), x, x, 4096, "relu")),
    )
    print(f"mode=dynamic; {iters} calls × {batches} batches; median ns/call")
    print(f"{'path':>19} {'ns/call':>10}")
    for label, fn, call in rows:
        print(f"{label:>19} {bench(fn, call, iters, batches, torch.cuda.synchronize):10.1f}")
```

In the module docstring's usage line, add `--dynamic`. In `benchmarks/AGENTS.md`'s `bench_launch.py` bullet, add: "`--dynamic` times warmed lazy launchers: plain, with two `dynamic_options`, and with a `str` constexpr."

```
PYTHONPATH=$PWD taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --dynamic --iters 100000 --batches 9
```

Expected: `2 dynamic options` costs about 2–4 ns over `lazy plain` (two scalar decodes and 18 key bytes). `str constexpr` costs about 1–3 ns over a scalar constexpr (one pointer compare). Record the three rows. They have no baseline, so they are reported but not gated.

- [ ] **Step 5: TODO**

In `TODO.md`:
- delete the "**Per-call variants.**" bullet;
- in the `knobs.runtime.debug` bullet, replace "flipping one after `make_launcher` keeps launching the old binary" with "they are read at a launcher's first call; flipping one later keeps launching the old binary unless it is declared in `dynamic_options`";
- add under Correctness: "**Free-threaded lazy-launcher publication on non-TSO hosts.** CPython reads `ml_meth` with a plain load; the build publishes `state` before a release store of `ml_meth`, which x86-64 orders. Recheck (or have the entry fall back to an acquire reload of `state`) when ARM64 is supported."

- [ ] **Step G: Benchmark gate and final A/B**

```bash
/tmp/intj-bench/run_all.sh lazy-task5
/tmp/gb2/bin/python /tmp/intj-bench/bench_compare.py lazy-baseline lazy-task5
bash /tmp/intj-bench/lazy_ab.sh lazy-task5
/tmp/gb2/bin/python /tmp/intj-bench/lazy_ab_summary.py /tmp/intj-bench/lazy-task5-ab lazy-task5
for r in 0 1 2; do PYTHONPATH=$PWD taskset -c 0 /tmp/gb2/bin/python benchmarks/bench_launch.py --dynamic --iters 100000 --batches 9 > /tmp/intj-bench/lazy-task5/extra_round${r}_launch_dynamic.txt 2>&1; done
```

Write the journal with the gate table, the A/B table and the `--dynamic` medians. Expected:
- no row regresses;
- the plain host rows are within 1 ns of `base` in the A/B, which is the spec's acceptance criterion.

- [ ] **Step 6: Full suite, matrix, pyright, commit**

```
PYTHONPATH=$PWD VIRTUAL_ENV=$V uv run --active --no-sync python -m pytest tests -q
bash tests/run_python_matrix.sh
PYTHONPATH=$PWD VIRTUAL_ENV=$V uv run --active --no-sync pyright
git add tests/test_launcher.py tests/test_tuned.py benchmarks/bench_launch.py benchmarks/AGENTS.md \
  TODO.md benchmarks/journals/*_lazy-task5_0_*.md
git commit -m "Pin the dynamic-value invariants and bench lazy launchers

<paste the three --dynamic rows here>

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

Expected: all tests pass on the venv and on all nine matrix builds, and pyright reports 0 errors.

---

## Deferred (marked `ponytail:` in code where they land)

- The memo is one entry per slot. Alternating two string objects in one slot costs an interner call per launch. Add a two-entry memo if `--dynamic` shows that pattern in real sites.
- Knobs that feed no compile option (e.g. `knobs.amd.use_buffer_ops`) are keyed only when declared. Keying the rest stays on `TODO.md`.

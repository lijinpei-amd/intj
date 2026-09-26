# Autotune and Heuristics in `make_launcher` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `make_launcher` accepts `@triton.jit` wrapped in any nesting of `@triton.autotune` / `@triton.heuristics`. On a miss, Triton tunes and launches through intj, and the result is recorded. On a hit, intj walks an unrolled chain of nested maps and launches natively.

**Architecture:**
- `make_launcher` visits the layers outer to inner and gives every name a role: an exact key, a dependent value, or a computed key.
- Dependent values live in cache records. Computed keys are heuristics lowered to C and keyed at the level where their inputs become known.
- On a miss, a private copy of the wrapper chain runs with a shim at the bottom. The shim compiles through intj's own compile path and launches, and its final call fills every missing level.

**Tech Stack:** Python ≥3.8, Triton 3.7/3.8 (`triton.runtime.autotuner`), Jinja2-rendered C/C++ CPython extension, intj's own map plus tsl/absl, pthread rwlock on free-threaded builds.

**Spec:** `docs/superpowers/specs/2026-09-26-autotune-heuristics-design.md`

## Global Constraints

- Every refusal raises `intj.launcher.UnsupportedKernel`, at `make_launcher` whenever it can be seen there. Never fall back to `JITFunction` silently.
- Invariant: same intj key chain ⟹ same Triton specialization, same tuning key, same heuristic outputs. `test_spec_key_is_never_coarser_than_triton` must keep passing, and the new tuned invariant test must be able to fail.
- `RenderContext`, `ModuleKey` and the new `TuningRender` are frozen value types with tuple fields only. They must hash and JSON-serialize.
- Rendered modules stay out of `sys.modules`, and the template stays multi-phase.
- `intj/` is pyright-strict. `tests/` and `benchmarks/` start with `# pyright: standard`. Private triton names get `# pyright: ignore[reportPrivateUsage]` plus a reason.
- Python ≥3.8 syntax in `intj/`: no `match`, no runtime `X | Y` types, and `ast.Index` still exists on 3.8.
- CUDA is compile-checked only. Say "untested" in docs.
- The backend-specific template branches on `Backend` fields, never on a backend name.
- Every commit message ends with `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`.
- Run everything through the venv:

```
V=/mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/venv
PYTHONPATH=$PWD VIRTUAL_ENV=$V uv run --active --no-sync python -m pytest tests -q
PYTHONPATH=$PWD VIRTUAL_ENV=$V uv run --active --no-sync pyright
```

## Benchmark Gate

Every task ends by rerunning the existing launcher benchmarks against a baseline taken before Task 1. A task does not commit with an unexplained regression.

**Suite.** All four scripts in `benchmarks/`, in every mode, plus the C++ kernel-cache benchmark that `benchmarks/AGENTS.md` lists. That is 10 cases, run for 3 rounds, each case in its own process, one at a time, pinned to core 0. The driver is `/tmp/intj-bench/run_all.sh <label>`; the same script produced the baseline. It writes `/tmp/intj-bench/<label>/round<r>_<case>.txt` with commit, command, raw output, elapsed time and exit status:

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

If `/tmp` was cleaned, recreate the driver from this copy:

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

The benchmarks use `/tmp/gb2/bin/python` (torch 2.14.0+rocm7.2, triton 3.8.0, apache-tvm-ffi 0.1.14): the uv venv has no tvm-ffi. This matches every earlier journal. Tests and pyright still use the venv. The baseline is at `/tmp/intj-bench/baseline/` and recorded in `benchmarks/journals/2026-09-26_autotune-baseline_0_<sha>.md`.

**Compare.** For every printed row, take the median of the 3 round values and compare it with the baseline median.
- A row **regresses** if it is slower by more than 3% *and* by more than 2 ns (host rows) or 0.1 µs (GPU/README rows).
- On a regression, rerun `/tmp/intj-bench/run_all.sh <label>-rerun` once. It is real only if the rerun also exceeds the threshold, since GPU submission rows vary about 20% between processes.
- A real regression gets fixed in that task. If it is the known, accepted cost of the task, say so in the journal and the commit message.

**Record.** Write `benchmarks/journals/2026-09-26_autotune-task<N>_0_<sha>.md` (baseline: `..._autotune-baseline_0_<sha>.md`) in the format of `benchmarks/AGENTS.md` "Result journals". Include the commands, raw outputs, a baseline-vs-task table of medians with the delta per row, and the environment. Commit it with the task.

---

## Review Focus

1. **A tuned constexpr sits before public parameters** in the JIT signature (`def k(x, BLOCK: tl.constexpr, out, n)`). The miss path must still bind every argument to the right name. Pinned in Task 3 (`test_tuned_param_before_public_params`).
2. **Two threads miss the same key at once.** Expect exactly one tuning run and one record, and both calls return. Pinned in Task 3 (`test_concurrent_misses_tune_once`).
3. **Configs that differ only in `num_warps` / `num_stages`.** Each record must launch with its own block size. Pinned in Task 3 (`test_config_compile_options_are_per_record`).
4. **Tuning raises** (a config fails to compile, or a prune function throws). Expect no record, the error reaches the caller, and the next call retries. Pinned in Task 3 (`test_failed_tuning_leaves_no_record`).
5. **A short-circuiting heuristic guards a `None` tensor** (`a["b"] is not None and a["b"].stride(0) == 1`, with `b=None`). This must not touch the tensor. Pinned in Task 4 (`test_computed_key_short_circuits_before_tensor_reads`).

## File Map

| file | responsibility | tasks |
|---|---|---|
| `intj/runtime/intj_map.h` (new) | hash, `INTJ_DEFINE_MAP`, `INTJ_DEFINE_CACHE`, `intj_kernel` | 1 |
| `intj/runtime/intj_runtime.h` | drops the cache code, adds snapshot helpers, the dep/heuristic helpers, and `tuned_cb` in the bound struct | 1, 3, 4 |
| `intj/python_intf/cpython_abi.h` | `intj_rwlock` replaces `intj_mutex` | 1 |
| `intj/runtime/entry.c.jinja` | by-value records, rwlock, tuned hit/miss, levels, `key_chain` | 1, 3, 4 |
| `intj/annotation.py` | `tuned` / `exact_key` flags, exact key fields | 3 |
| `intj/heuristic.py` (new) | parse heuristic functions, lower computed ones to C | 2, 4 |
| `intj/tuning.py` (new) | chain analysis, private chain, shim, miss callback | 2, 3, 4 |
| `intj/grid.py` | `grid_cpp` reads dependent values | 3 |
| `intj/launcher.py` | `TuningRender`, `make_launcher` integration, `_checked_compile` | 3, 4 |
| `tests/bench_kernel_cache.cpp`, `tests/test_kernel_cache.py` | by-value maps, child-map row | 1 |
| `tests/test_tuning.py` (new) | analysis and lowering, no launch | 2, 4 |
| `tests/test_tuned.py` (new) | end-to-end tuned launchers on the GPU | 3, 4, 5 |
| `docs/Usage.md`, `AGENTS.md`, `intj/python_intf/AGENTS.md`, `TODO.md`, `benchmarks/bench_launch.py` | docs and bench | 1, 5 |

---

### Task 0: Benchmark baseline

About 30 minutes, mostly waiting on runs.

- [ ] **Step 1: Run the suite on the unchanged tree**

```bash
git status --short   # must be clean apart from this plan
/tmp/intj-bench/run_all.sh baseline
```

Expected: `/tmp/intj-bench/baseline/DONE` exists, and all 30 files report `exit_status=0`.

- [ ] **Step 2: Record and commit the baseline journal**

Write `benchmarks/journals/2026-09-26_autotune-baseline_0_$(git rev-parse --short HEAD).md` per the Benchmark Gate's "Record" rule, with the baseline medians only.

```bash
git add benchmarks/journals/2026-09-26_autotune-baseline_0_*.md
git commit -m "Record launcher benchmark baseline before autotune support

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 1: By-value cache records behind a read/write lock

About 1 day. This is a pure refactor for every existing launcher. It is the base that Tasks 3–4 build nested levels on.

**Files:**
- Create: `intj/runtime/intj_map.h`
- Modify: `intj/runtime/intj_runtime.h` (remove lines 136–531: hash, `intj_kernel`, `intj_slot`/`intj_map`, the cache section, `intj_cache_remember`/`lookup`; bound struct at 967–991)
- Modify: `intj/python_intf/cpython_abi.h:126-141`
- Modify: `intj/runtime/entry.c.jinja` (state struct 82–98; `intj_call` 596–711; bound traverse/clear 822–881; `set_compile_callback` 1071–1080; exec/traverse/clear/free 1305–1368)
- Modify: `tests/bench_kernel_cache.cpp`, `tests/test_launcher.py:414-417` (source-string assertion)
- Modify: `AGENTS.md` (kernel-cache section), `intj/python_intf/AGENTS.md:112`
- Test: `tests/test_kernel_cache.py`, `tests/test_return_compiled.py`

**Interfaces:**
- Produces, C, in `intj_map.h`:
  - `uint64_t intj_hash_n(const uint64_t *w, int nw)`
  - `uint64_t intj_hash(const uint64_t *w)` when `INTJ_NWORDS` is defined
  - `typedef struct {void *function; uint32_t block_dim, shared, nparams; [PyObject *compiled]} intj_kernel;`
  - `INTJ_DEFINE_MAP(P, NW, V, CAP)` defines `P`, `P##_slot`, `P##_init(P*)`, `P##_get(const P*, const uint64_t*, uint64_t h) -> V*`, `P##_lookup(P*, const uint64_t*) -> V*`, `P##_put(P*, const uint64_t*, const V*) -> V*` (NULL on OOM), `P##_each(const P*, int (*)(V*, void*), void*)`, `P##_free(P*, void (*)(V*))`
  - `INTJ_DEFINE_CACHE(V)` defines `intj_cache` with the same six operations (`init` takes no capacity) for the selected backend
- Produces, C, in `cpython_abi.h`: `intj_rwlock`, `INTJ_RWLOCK_INIT/DESTROY`, `INTJ_RDLOCK`, `INTJ_WRLOCK`, `INTJ_RWUNLOCK`
- Produces, C, in `intj_runtime.h`: `intj_snapshot`, `intj_snapshot_add(intj_snapshot*, PyObject*)`, `intj_snapshot_visit(intj_snapshot*, int failed, visitproc, void*)`
- Produces, template:
  - `typedef struct {intj_kernel k;} intj_final;`, `intj_final_release(intj_final*)`, `intj_final_collect(intj_final*, void*)`
  - `INTJ_V0_RELEASE`, `INTJ_V0_COLLECT`
  - `intj_fill(...)`, `intj_launch(...)`
  - state field `intj_rwlock lock` (replaces `mutex`)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_kernel_cache.py`:

```python
import torch
import triton
import triton.language as tl

from intj import Constexpr, TorchAccessMode, make_launcher


@triton.jit
def keyed_store(x, K: tl.constexpr):
    tl.store(x, K)


@pytest.mark.parametrize("cache", list(KernelCache))
def test_records_survive_rehash(cache):
    """Records live in the slots now, so every growth moves them."""
    launch = make_launcher(
        keyed_store,
        no_gpu=True,
        kernel_cache=cache,
        torch_access_mode=TorchAccessMode.INTERPRETER,
    )
    compiled = []

    def compile_key(key, nparams, device, x, k):
        compiled.append(k)
        return 0, 1, 0, nparams

    launch.__self__.set_compile_callback(compile_key)
    for _ in range(2):
        for k in range(100):
            launch(0, 0, 1, 0, k)
    assert compiled == list(range(100))
```

Append to `tests/test_return_compiled.py`:

```python
@triton.jit
def keyed_value(x, K: tl.constexpr):
    tl.store(x, K)


def test_compiled_objects_survive_rehash():
    launch = make_launcher(keyed_value, return_compiled=True)
    x = torch.zeros(1, device="cuda", dtype=torch.int32)
    device = torch.cuda.current_device()
    stream = torch.cuda.current_stream().cuda_stream
    first = [launch(device, stream, 1, x, k) for k in range(40)]
    again = [launch(device, stream, 1, x, k) for k in range(40)]
    assert all(a is b for a, b in zip(first, again))
    torch.cuda.synchronize()
    assert int(x.item()) == 39
```

- [ ] **Step 2: Run them and confirm they pass on the old code**

```
PYTHONPATH=$PWD VIRTUAL_ENV=$V uv run --active --no-sync python -m pytest tests/test_kernel_cache.py::test_records_survive_rehash tests/test_return_compiled.py::test_compiled_objects_survive_rehash -q
```

Expected: PASS. These are regression guards for the move, not new behaviour. They have to stay green after Step 7.

- [ ] **Step 3: Create `intj/runtime/intj_map.h`**

```c
/* Kernel-cache maps: the key hash, intj's open-addressed map instantiated per
 * key width and value type, and the swappable level-0 cache.
 *
 * Values live in the map.  A rehash moves them, so a `V *` from a lookup or put
 * is valid until the next put into that map.  A launch holds the module's read
 * lock across its lookups and the launch; a put holds the write lock.  A value
 * that must not move -- a record owning a child map -- is stored by pointer.
 */
#pragma once

#include <Python.h>
#include <stdint.h>
#include <string.h>

#if defined(__GNUC__) || defined(__clang__)
#define INTJ_ALWAYS_INLINE __attribute__((always_inline)) inline
/* Only for branches whose outcome is a property of the design, not a guess
 * about the caller: a bad argument, a launch failure, a cold cache.  The
 * argument *types* are deliberately not marked -- the render cannot know
 * whether a parameter is usually a tensor or an int. */
#define INTJ_LIKELY(x) __builtin_expect(!!(x), 1)
#define INTJ_UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
#define INTJ_ALWAYS_INLINE inline
#define INTJ_LIKELY(x) (x)
#define INTJ_UNLIKELY(x) (x)
#endif

static inline uint64_t intj_mix(uint64_t a, uint64_t b) {
  __uint128_t r = (__uint128_t)a * b;
  return (uint64_t)(r >> 64) ^ (uint64_t)r;
}

/* `nw` is a compile-time constant at every call site, so each one folds to the
 * shape its width needs.  At one word the mix is a bijection -- xorshift-right
 * and an odd multiply both are -- which makes hash equality key equality and
 * lets the slot drop the key.  At two, one multiply suffices, the way wyhash
 * handles a short input.  Above that the chain is split across two lanes: a mix
 * is a ~4-cycle multiply sitting between the last decode and the first probe,
 * so five words is ~20 cycles serially and ~12 in pairs. */
static INTJ_ALWAYS_INLINE uint64_t intj_hash_n(const uint64_t *w, int nw) {
  if (nw == 1) {
    uint64_t x = w[0];
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ull;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebull;
    x ^= x >> 31;
    return x;
  }
  const uint64_t s0 = 0xa0761d6478bd642full, s1 = 0xe7037ed1a0b428dbull;
  if (nw == 2)
    return intj_mix(intj_mix(w[0] ^ s0, w[1] ^ s1), 2 * 8 + s1);
  uint64_t h0 = s0, h1 = s1;
  int i = 0;
  for (; i + 1 < nw; i += 2) {
    h0 = intj_mix(h0 ^ s1, w[i] ^ s0);
    h1 = intj_mix(h1 ^ s0, w[i + 1] ^ s1);
  }
  if (i < nw)
    h0 = intj_mix(h0 ^ s1, w[i] ^ s0);
  return intj_mix(h0 ^ h1, (uint64_t)nw * 8 + s1);
}

#ifdef INTJ_NWORDS
static inline uint64_t intj_hash(const uint64_t *w) {
  return intj_hash_n(w, INTJ_NWORDS);
}
#endif

typedef struct {
  void *function;     /* hipFunction_t / CUfunction */
  uint32_t block_dim; /* warp_size * num_warps */
  uint32_t shared;    /* dynamic LDS bytes */
  uint32_t nparams;   /* kernel params, scratch slots excluded */
#ifdef INTJ_RETURN_COMPILED
  PyObject *compiled; /* owns the CompiledKernel behind function */
#endif
} intj_kernel;

/* At NW == 1 the hash is the key, so the slot stores none: a zero-length array,
 * which gcc and clang accept in C and C++.  `full` marks occupancy because no
 * value field is free to act as a sentinel.
 *
 * `last` is the one-entry memo: the slot of the last hit.  Readers under the
 * read lock store it with relaxed atomics; a slot cannot move while any reader
 * holds the lock, and a put -- under the write lock -- clears it.
 *
 * gcc 13.3.0 miscompiled the old one-word lookup at `-O1 -fsanitize=undefined`
 * (a present key read as absent, with no UBSan diagnostic); -O2/-O3 and clang
 * were clean.  Nothing intj builds uses those flags.  Recheck here first if a
 * sanitizer build of `tests/bench_kernel_cache.cpp` reports a wrong kernel. */
#define INTJ_KEY_WORDS(NW) ((NW) > 1 ? (NW) : 0)

#define INTJ_DEFINE_MAP(P, NW, V, CAP)                                         \
  typedef struct {                                                             \
    uint64_t hash;                                                             \
    uint64_t full;                                                             \
    uint64_t key[INTJ_KEY_WORDS(NW)];                                          \
    V val;                                                                     \
  } P##_slot;                                                                  \
  typedef struct {                                                             \
    P##_slot *slots;                                                           \
    uint32_t mask;                                                             \
    uint32_t used;                                                             \
    P##_slot *last;                                                            \
  } P;                                                                         \
  static inline int P##_init(P *m) {                                           \
    m->slots = (P##_slot *)PyMem_RawCalloc((CAP), sizeof(P##_slot));           \
    m->mask = (CAP) - 1;                                                       \
    m->used = 0;                                                               \
    m->last = NULL;                                                            \
    return m->slots ? 0 : -1;                                                  \
  }                                                                            \
  static INTJ_ALWAYS_INLINE P##_slot *P##_find(const P *m, const uint64_t *k,  \
                                               uint64_t h) {                   \
    uint32_t i = (uint32_t)h & m->mask;                                        \
    for (;;) {                                                                 \
      P##_slot *s = &m->slots[i];                                              \
      if (INTJ_UNLIKELY(!s->full))                                             \
        return NULL;                                                           \
      if (INTJ_LIKELY(s->hash == h &&                                          \
                      memcmp(s->key, k, sizeof(s->key)) == 0))                 \
        return s;                                                              \
      i = (i + 1) & m->mask;                                                   \
    }                                                                          \
  }                                                                            \
  static INTJ_ALWAYS_INLINE V *P##_get(const P *m, const uint64_t *k,          \
                                       uint64_t h) {                           \
    P##_slot *s = P##_find(m, k, h);                                           \
    return s ? &s->val : NULL;                                                 \
  }                                                                            \
  static INTJ_ALWAYS_INLINE V *P##_lookup(P *m, const uint64_t *k) {           \
    P##_slot *s = __atomic_load_n(&m->last, __ATOMIC_RELAXED);                 \
    if ((NW) > 1 && s && memcmp(s->key, k, sizeof(s->key)) == 0)               \
      return &s->val;                                                          \
    uint64_t h = intj_hash_n(k, (NW));                                         \
    if ((NW) == 1 && s && s->hash == h)                                        \
      return &s->val;                                                          \
    s = P##_find(m, k, h);                                                     \
    if (!s)                                                                    \
      return NULL;                                                             \
    __atomic_store_n(&m->last, s, __ATOMIC_RELAXED);                           \
    return &s->val;                                                            \
  }                                                                            \
  static inline V *P##_put(P *m, const uint64_t *k, const V *val) {            \
    if ((m->used + 1) * 2 > m->mask + 1) {                                     \
      uint32_t cap = (m->mask + 1) * 2;                                        \
      P##_slot *grown = (P##_slot *)PyMem_RawCalloc(cap, sizeof(P##_slot));    \
      if (!grown)                                                              \
        return NULL;                                                           \
      for (uint32_t i = 0; i <= m->mask; i++) {                                \
        if (!m->slots[i].full)                                                 \
          continue;                                                            \
        uint32_t j = (uint32_t)m->slots[i].hash & (cap - 1);                   \
        while (grown[j].full)                                                  \
          j = (j + 1) & (cap - 1);                                             \
        grown[j] = m->slots[i];                                                \
      }                                                                        \
      PyMem_RawFree(m->slots);                                                 \
      m->slots = grown;                                                        \
      m->mask = cap - 1;                                                       \
    }                                                                          \
    uint64_t h = intj_hash_n(k, (NW));                                         \
    uint32_t i = (uint32_t)h & m->mask;                                        \
    while (m->slots[i].full)                                                   \
      i = (i + 1) & m->mask;                                                   \
    P##_slot *s = &m->slots[i];                                                \
    s->hash = h;                                                               \
    s->full = 1;                                                               \
    memcpy(s->key, k, sizeof(s->key));                                         \
    s->val = *val;                                                             \
    m->used++;                                                                 \
    __atomic_store_n(&m->last, (P##_slot *)NULL, __ATOMIC_RELAXED);            \
    return &s->val;                                                            \
  }                                                                            \
  static inline int P##_each(const P *m, int (*fn)(V *, void *), void *ctx) {  \
    if (!m->slots)                                                             \
      return 0;                                                                \
    for (uint32_t i = 0; i <= m->mask; i++)                                    \
      if (m->slots[i].full) {                                                  \
        int r = fn(&m->slots[i].val, ctx);                                     \
        if (r)                                                                 \
          return r;                                                            \
      }                                                                        \
    return 0;                                                                  \
  }                                                                            \
  /* Detach before releasing: a CompiledKernel finalizer may reenter GC. */   \
  static inline void P##_free(P *m, void (*release)(V *)) {                    \
    P##_slot *slots = m->slots;                                                \
    uint32_t mask = m->mask;                                                   \
    m->slots = NULL;                                                           \
    m->used = 0;                                                               \
    m->last = NULL;                                                            \
    if (!slots)                                                                \
      return;                                                                  \
    for (uint32_t i = 0; i <= mask; i++)                                       \
      if (slots[i].full)                                                       \
        release(&slots[i].val);                                                \
    PyMem_RawFree(slots);                                                      \
  }

/* The level-0 cache behind one interface, so the entry template never names an
 * implementation.  INTJ_CACHE_{INTJ,TSL,ABSL} selects it; the last two are C++
 * and force the module to be compiled as C++.  `intj_cache` is POD in every
 * mode -- it lives in zeroed module state -- so the C++ maps are held by
 * pointer.  Only tsl takes the precomputed hash; abseil re-computes it, which
 * is the measured cost of that option. */
#if defined(INTJ_CACHE_TSL) || defined(INTJ_CACHE_ABSL)

#include <new>
#if defined(INTJ_CACHE_TSL)
#include <tsl/robin_map.h>
#define INTJ_CACHE_MAP(V) tsl::robin_map<intj_key, V, intj_key_hash>
#define INTJ_CACHE_FIND(map, key, h) (map)->find((key), (size_t)(h))
#else
#include <absl/container/flat_hash_map.h>
#define INTJ_CACHE_MAP(V) absl::flat_hash_map<intj_key, V, intj_key_hash>
#define INTJ_CACHE_FIND(map, key, h) ((void)(h), (map)->find(key))
#endif

struct intj_key {
  uint64_t w[INTJ_NWORDS];
  bool operator==(const intj_key &o) const {
    return memcmp(w, o.w, sizeof(w)) == 0;
  }
};

struct intj_key_hash {
  using is_avalanching = void; /* wyhash output: no further mixing wanted */
  size_t operator()(const intj_key &k) const { return (size_t)intj_hash(k.w); }
};

/* Entries are const_cast back to V: tsl hands out const pairs so the key
 * cannot change, but the stored object itself is not const. */
#define INTJ_DEFINE_CACHE(V)                                                   \
  typedef INTJ_CACHE_MAP(V) intj_cache_map;                                    \
  typedef intj_cache_map::value_type intj_cache_entry;                         \
  typedef struct {                                                             \
    intj_cache_map *map;                                                       \
    const intj_cache_entry *last;                                              \
  } intj_cache;                                                                \
  static inline int intj_cache_init(intj_cache *c) {                           \
    c->map = new (std::nothrow) intj_cache_map();                              \
    c->last = NULL;                                                            \
    return c->map ? 0 : -1;                                                    \
  }                                                                            \
  static INTJ_ALWAYS_INLINE V *intj_cache_get(const intj_cache *c,             \
                                              const uint64_t *k, uint64_t h) { \
    auto it = INTJ_CACHE_FIND(c->map, *(const intj_key *)k, h);                \
    return it == c->map->end() ? NULL : const_cast<V *>(&it->second);          \
  }                                                                            \
  static INTJ_ALWAYS_INLINE V *intj_cache_lookup(intj_cache *c,                \
                                                 const uint64_t *k) {          \
    const intj_cache_entry *e = __atomic_load_n(&c->last, __ATOMIC_RELAXED);   \
    if (e && memcmp(e->first.w, k, sizeof(e->first.w)) == 0)                   \
      return const_cast<V *>(&e->second);                                     \
    auto it = INTJ_CACHE_FIND(c->map, *(const intj_key *)k, intj_hash(k));     \
    if (it == c->map->end())                                                   \
      return NULL;                                                             \
    e = &*it;                                                                  \
    __atomic_store_n(&c->last, e, __ATOMIC_RELAXED);                           \
    return const_cast<V *>(&e->second);                                        \
  }                                                                            \
  /* The maps throw where intj returns, and an exception reaching CPython's   \
   * C frames is std::terminate, so the throw stops here. */                  \
  static inline V *intj_cache_put(intj_cache *c, const uint64_t *k,            \
                                  const V *val) {                              \
    try {                                                                      \
      auto r = c->map->insert(intj_cache_entry(*(const intj_key *)k, *val));   \
      __atomic_store_n(&c->last, (const intj_cache_entry *)NULL,               \
                       __ATOMIC_RELAXED);                                      \
      return const_cast<V *>(&r.first->second);                                \
    } catch (...) {                                                            \
      return NULL;                                                             \
    }                                                                          \
  }                                                                            \
  static inline int intj_cache_each(const intj_cache *c,                       \
                                    int (*fn)(V *, void *), void *ctx) {       \
    if (!c->map)                                                               \
      return 0;                                                                \
    for (const auto &e : *c->map) {                                            \
      int r = fn(const_cast<V *>(&e.second), ctx);                             \
      if (r)                                                                   \
        return r;                                                              \
    }                                                                          \
    return 0;                                                                  \
  }                                                                            \
  static inline void intj_cache_free(intj_cache *c, void (*release)(V *)) {    \
    intj_cache_map *map = c->map;                                              \
    c->map = NULL;                                                             \
    c->last = NULL;                                                            \
    if (!map)                                                                  \
      return;                                                                  \
    for (auto &e : *map)                                                       \
      release(const_cast<V *>(&e.second));                                     \
    delete map;                                                                \
  }

#else /* INTJ_CACHE_INTJ */

#define INTJ_DEFINE_CACHE(V) INTJ_DEFINE_MAP(intj_cache, INTJ_NWORDS, V, 16)

#endif
```

- [ ] **Step 4: Trim `intj_runtime.h` and add the snapshot helpers**

In `intj/runtime/intj_runtime.h`:

- Replace the `INTJ_ALWAYS_INLINE`/`INTJ_LIKELY` block (lines 20–32) with `#include "intj_map.h"`.
- Delete lines 136–531: `intj_mix` through `intj_cache_lookup`, which covers the hash, `intj_kernel`, `intj_last_key`, the slot/map code, `intj_visit_snapshot`, the whole cache section and `intj_cache_remember`/`intj_cache_lookup`.
- Insert this where they were:

```c
/* Objects a GC traverse visits, collected under the read lock.  A visitor can
 * reenter Python and grow the cache, so the visits happen after the lock is
 * released, with every object kept alive through the last visit. */
typedef struct {
  PyObject **items;
  size_t n, cap;
} intj_snapshot;

static inline int intj_snapshot_add(intj_snapshot *s, PyObject *o) {
  if (!o)
    return 0;
  if (s->n == s->cap) {
    size_t cap = s->cap ? 2 * s->cap : 16;
    PyObject **items =
        (PyObject **)PyMem_RawRealloc(s->items, cap * sizeof(*items));
    if (!items)
      return -1;
    s->items = items;
    s->cap = cap;
  }
  s->items[s->n++] = Py_NewRef(o);
  return 0;
}

static inline int intj_snapshot_visit(intj_snapshot *s, int failed,
                                      visitproc visit, void *arg) {
  int result = 0;
  if (failed) {
    PyErr_NoMemory();
    result = -1;
  }
  for (size_t i = 0; !result && i < s->n; i++)
    result = visit(s->items[i], arg);
  for (size_t i = 0; i < s->n; i++)
    Py_DECREF(s->items[i]);
  PyMem_RawFree(s->items);
  return result;
}
```

In the `intj_bound_launcher` struct, change `intj_kernel *fixed_kernel;` to `intj_final *fixed_kernel;`. The template now defines `intj_final` and `intj_cache` before including this header.

Update the header comment at the top:

```c
/* INTJ launcher runtime: helpers shared by every rendered entry module.
 *
 * The rendered module defines INTJ_NWORDS, its record types and its cache
 * (`INTJ_DEFINE_CACHE`, from intj_map.h) before including this header.
 */
```

- [ ] **Step 5: Replace the mutex with a read/write lock in `cpython_abi.h`**

Replace lines 126–141 with:

```c
/* The module's one lock, for what the GIL guards on a default build: the kernel
 * caches and the compile callback.  Launches take it shared across their
 * lookups and the launch, because records live in the map and a put can move
 * them; a put takes it exclusive.  Compiled out with the GIL, so a default
 * build pays nothing.  CPython's own rwlock is private, hence pthread. */
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
```

Update `intj/python_intf/AGENTS.md:112` to say: `intj_rwlock` is a pthread rwlock on free-threaded builds, a no-op otherwise, taken shared by launches and exclusive by puts.

- [ ] **Step 6: Rewrite the template's cache use**

In `intj/runtime/entry.c.jinja`:

(a) Replace `#include "intj_runtime.h"` (line 73) with the record block. The `INTJ_*` defines above it stay:

```jinja
#include "intj_map.h"

/* The final-level record, stored by value in whichever map holds it. */
typedef struct {
  intj_kernel k;
} intj_final;

/* Outside any lock: dropping a CompiledKernel may run a finalizer. */
static void intj_final_release(intj_final *f) {
{% if return_compiled %}
  Py_XDECREF(f->k.compiled);
{% else %}
  (void)f;
{% endif %}
}

INTJ_DEFINE_CACHE(intj_final)
#define INTJ_V0_RELEASE intj_final_release

#include "intj_runtime.h"
```

(b) In `intj_state`, replace `intj_mutex mutex; ...` with `intj_rwlock lock; /* every cache and compile_cb, on a free-threaded build */`.

(c) After `intj_grid_object` (before `{% if grid_cpp_source %}`), add the collector and the launch helper:

```jinja
static int intj_final_collect(intj_final *f, void *ctx) {
  intj_snapshot *s = (intj_snapshot *)ctx;
  (void)s;
{% if return_compiled %}
  if (intj_snapshot_add(s, f->k.compiled) != 0)
    return -1;
{% endif %}
  return 0;
}
#define INTJ_V0_COLLECT intj_final_collect

static int intj_cache_traverse(intj_state *st, const intj_cache *c,
                               visitproc visit, void *arg) {
  intj_snapshot s = {NULL, 0, 0};
  INTJ_RDLOCK(&st->lock);
  int failed = intj_cache_each(c, INTJ_V0_COLLECT, &s);
  INTJ_RWUNLOCK(&st->lock);
  return intj_snapshot_visit(&s, failed, visit, arg);
}

{% if not no_gpu %}
/* Two trailing scratch slots are mandatory on AMD; intj refuses kernels that
 * need non-empty scratch, so they are always null. */
static INTJ_ALWAYS_INLINE int32_t intj_launch(intj_state *st, const intj_kernel *k,
                                              uint32_t gx, uint32_t gy, uint32_t gz,
                                              uint64_t stream, uint64_t *vals, int np) {
  void *params[INTJ_NSLOTS + 2];
  for (int i = 0; i < np; i++)
    params[i] = &vals[i];
  uint64_t scratch[2] = {0, 0};
  params[np] = &scratch[0];
  params[np + 1] = &scratch[1];
  return st->launch(k->function, gx, gy, gz, k->block_dim, 1, 1, k->shared,
                    (void *)(uintptr_t)stream, params, NULL);
}

static PyObject *intj_launch_error(intj_state *st, int32_t status) {
  const char *message = NULL;
{% if error_style == 'return' %}
  if (st->error_string)
    message = st->error_string(status);
{% else %}
  if (st->error_string)
    st->error_string(status, &message);
{% endif %}
  PyErr_Format(PyExc_RuntimeError, "intj: {{ launch_symbol }} failed: %s",
               message ? message : "unknown error");
  return NULL;
}
{% endif %}

/* Cold path: Python assembles the final canonical ASTSource from the public
 * arguments and fixed annotations.  No lock is held while Python runs; another
 * thread may fill this key meanwhile, and the put rechecks. */
static int intj_fill(intj_state *st, intj_cache *cache, intj_bound_launcher *bound,
                     const uint64_t *key, int np, int64_t device_ordinal,
                     PyObject *const *kernel_args) {
  (void)cache;
  (void)bound;
  INTJ_RDLOCK(&st->lock);
  PyObject *cb = st->compile_cb;
  Py_XINCREF(cb);
  INTJ_RWUNLOCK(&st->lock);
  if (!cb) {
    PyErr_SetString(PyExc_RuntimeError, "intj: compile callback is not set");
    return -1;
  }
  PyObject *blob = PyBytes_FromStringAndSize((const char *)key, INTJ_KEY_BYTES);
  PyObject *nparams_obj = blob ? PyLong_FromLong(np) : NULL;
  PyObject *device_obj = nparams_obj ? PyLong_FromLongLong(device_ordinal) : NULL;
  if (!device_obj) {
    Py_XDECREF(blob);
    Py_XDECREF(nparams_obj);
    Py_DECREF(cb);
    return -1;
  }
  PyObject *cb_args[3 + INTJ_NPARAMS];
  cb_args[0] = blob;
  cb_args[1] = nparams_obj;
  cb_args[2] = device_obj;
  for (int i = 0; i < INTJ_NPARAMS; i++)
    cb_args[3 + i] = kernel_args[i];
  PyObject *res = PyObject_Vectorcall(cb, cb_args, 3 + INTJ_NPARAMS, NULL);
  Py_DECREF(cb);
  Py_DECREF(blob);
  Py_DECREF(nparams_obj);
  Py_DECREF(device_obj);
  if (!res)
    return -1;
  unsigned long long function;
  unsigned int block_dim, shared, nparams;
{% if return_compiled %}
  PyObject *compiled;
  if (!PyArg_ParseTuple(res, "KIIIO", &function, &block_dim, &shared, &nparams,
                        &compiled)) {
{% else %}
  if (!PyArg_ParseTuple(res, "KIII", &function, &block_dim, &shared, &nparams)) {
{% endif %}
    Py_DECREF(res);
    return -1;
  }
  intj_final fresh;
  memset(&fresh, 0, sizeof(fresh));
  fresh.k.function = (void *)(uintptr_t)function;
  fresh.k.block_dim = block_dim;
  fresh.k.shared = shared;
  fresh.k.nparams = nparams;
{% if return_compiled %}
  fresh.k.compiled = Py_NewRef(compiled);
{% endif %}
  Py_DECREF(res);
  int taken = 0, failed = 0;
  INTJ_WRLOCK(&st->lock);
{% if nwords %}
  if (!intj_cache_lookup(cache, key)) {
    if (intj_cache_put(cache, key, &fresh))
      taken = 1;
    else
      failed = 1;
  }
{% else %}
  if (!bound->fixed_kernel) {
    intj_final *heap = (intj_final *)PyMem_RawMalloc(sizeof(intj_final));
    if (heap) {
      *heap = fresh;
      bound->fixed_kernel = heap;
      taken = 1;
    } else {
      failed = 1;
    }
  }
{% endif %}
  INTJ_RWUNLOCK(&st->lock);
  if (!taken)
    intj_final_release(&fresh);
  if (failed) {
    PyErr_NoMemory();
    return -1;
  }
  return 0;
}
```

(d) In `intj_call`, replace everything from `{% if nwords %}\n  uint64_t hash;` (line 603) to the end of the function (line 750) with:

```jinja
  INTJ_RDLOCK(&st->lock);
{% if nwords %}
  intj_final *kernel = intj_cache_lookup(cache, key);
{% else %}
  intj_final *kernel = bound->fixed_kernel;
{% endif %}
  if (INTJ_UNLIKELY(!kernel)) {
    INTJ_RWUNLOCK(&st->lock);
    if (intj_fill(st, cache, bound, key, np, device_ordinal, args + {{ grid_count }}) != 0)
      return NULL;
    INTJ_RDLOCK(&st->lock);
{% if nwords %}
    kernel = intj_cache_lookup(cache, key);
{% else %}
    kernel = bound->fixed_kernel;
{% endif %}
  }

  /* Held through the launch: `kernel` lives in the map.  Errors are raised
   * after the unlock, because raising allocates and can run Python. */
  PyObject *result = NULL;
  uint32_t recorded_np = kernel->k.nparams;
  int32_t status = 0;
  if (INTJ_LIKELY((uint32_t)np == recorded_np)) {
{% if not no_gpu %}
{% if return_compiled %}
    if (gx && gy && gz)
{% endif %}
      status = intj_launch(st, &kernel->k, gx, gy, gz, stream, vals, np);
{% endif %}
{% if return_compiled %}
    if (!status)
      result = Py_NewRef(kernel->k.compiled);
{% else %}
    if (!status)
      result = Py_NewRef(Py_None);
{% endif %}
  }
  INTJ_RWUNLOCK(&st->lock);
  if (INTJ_UNLIKELY((uint32_t)np != recorded_np)) {
    PyErr_Format(PyExc_RuntimeError,
                 "intj: kernel argument count changed for an identical spec "
                 "key (%d vs %u); this is an intj bug",
                 np, recorded_np);
    return NULL;
  }
{% if not no_gpu %}
  if (INTJ_UNLIKELY(status != 0))
    return intj_launch_error(st, status);
{% endif %}
  return result;
}
```

Also delete the old `/* Launch. Two trailing scratch slots ... */` block, which now lives in `intj_launch`.

(e) Bound traverse (lines 832–850): replace both branches with:

```jinja
{% if return_compiled and fixed_device and nwords %}
  if (bound->cache_ready && bound->module) {
    intj_state *st = (intj_state *)PyModule_GetState(bound->module);
    return intj_cache_traverse(st, &bound->cache, visit, arg);
  }
{% elif return_compiled and fixed_device %}
  if (bound->module) {
    intj_state *st = (intj_state *)PyModule_GetState(bound->module);
    intj_snapshot s = {NULL, 0, 0};
    INTJ_RDLOCK(&st->lock);
    int failed = bound->fixed_kernel ? intj_final_collect(bound->fixed_kernel, &s) : 0;
    INTJ_RWUNLOCK(&st->lock);
    return intj_snapshot_visit(&s, failed, visit, arg);
  }
{% endif %}
```

(f) Bound clear (lines 870–879):

```jinja
{% if fixed_device and nwords %}
  if (bound->cache_ready) {
    bound->cache_ready = 0;
    intj_cache_free(&bound->cache, INTJ_V0_RELEASE);
  }
{% elif fixed_device %}
  intj_final *kernel = bound->fixed_kernel;
  bound->fixed_kernel = NULL;
  if (kernel) {
    intj_final_release(kernel);
    PyMem_RawFree(kernel);
  }
{% endif %}
```

(g) `set_compile_callback`: `INTJ_LOCK(&st->mutex)` becomes `INTJ_WRLOCK(&st->lock)`, and `INTJ_UNLOCK` becomes `INTJ_RWUNLOCK(&st->lock)`.

(h) `intj_exec`: add `INTJ_RWLOCK_INIT(&st->lock);` as its first statement after `st` is read.

(i) `intj_traverse`: read `compile_cb` under `INTJ_RDLOCK`/`INTJ_RWUNLOCK`, and change the tail to `return intj_cache_traverse(st, &st->cache, visit, arg);`.

(j) `intj_clear` and `intj_free`: `intj_cache_free(&st->cache)` becomes `intj_cache_free(&st->cache, INTJ_V0_RELEASE)`. `intj_free` also ends with `INTJ_RWLOCK_DESTROY(&st->lock);`, outside the `{% if not fixed_device %}`.

(k) `grep -n "mutex\|INTJ_LOCK\|INTJ_UNLOCK\|intj_kernel_free\|intj_cache_remember" intj/runtime/entry.c.jinja` must print nothing.

- [ ] **Step 7: Update the benchmark and source assertions**

In `tests/bench_kernel_cache.cpp`:
- Replace `#include "intj_runtime.h"` and the `#define INTJ_TORCH_ACCESS_RUNTIME_SHIM` line with the block below.
- In `hit`/`miss`, drop the `kernels` heap vector and `PyMem_RawCalloc`. Put `intj_kernel{(void *)(uintptr_t)(i + 1), 1, 0, 0}` by value and compare `found->function`.
- Replace `intj_cache_free(&cache)` with `intj_cache_free(&cache, release)`.
- Add a `hit_child` benchmark on a 1-word `INTJ_DEFINE_MAP` with 2 entries, the shape of a computed-bool level:

```cpp
#include "intj_map.h"

static void release(intj_kernel *) {}
INTJ_DEFINE_CACHE(intj_kernel)
INTJ_DEFINE_MAP(bench_child, 1, intj_kernel, 4)

/* A child level keyed by one computed bool: what a heuristic like
 * `N % BLOCK == 0` adds to a launch after its level-0 lookup. */
void hit_child(benchmark::State &state) {
  bench_child map;
  if (bench_child_init(&map) != 0) {
    state.SkipWithError("init failed");
    return;
  }
  uint64_t keys[2][1] = {{0}, {1}};
  for (int i = 0; i < 2; i++) {
    intj_kernel k = {(void *)(uintptr_t)(i + 1), 1, 0, 0};
    bench_child_put(&map, keys[i], &k);
  }
  int i = 0;
  for (auto _ : state) {
    intj_kernel *found = bench_child_lookup(&map, keys[i++ & 1]);
    benchmark::DoNotOptimize(found);
  }
  bench_child_free(&map, release);
  state.SetLabel(INTJ_CACHE_NAME);
}
BENCHMARK(hit_child);
```

In `tests/test_launcher.py:416`, change `"intj_cache_lookup(cache, key, &hash)"` to `"intj_cache_lookup(cache, key)"`.

In `AGENTS.md`, replace the kernel-cache paragraph's first two sentences with:

> `intj_cache_{init,lookup,get,put,each,free}`, instantiated by `INTJ_DEFINE_CACHE(V)` in `runtime/intj_map.h`, is the whole interface; the entry template never names an implementation. Every backend stores records by value, so a record pointer is valid only while the module's read lock is held; `lookup` checks a one-entry memo before hashing or probing, and a put clears it.

- [ ] **Step 8: Run the full suite and pyright**

```
PYTHONPATH=$PWD VIRTUAL_ENV=$V uv run --active --no-sync python -m pytest tests -q
PYTHONPATH=$PWD VIRTUAL_ENV=$V uv run --active --no-sync pyright
```

Expected:
- All tests pass, including both Step 1 tests. `test_kernel_cache.py` benchmarks run or skip depending on google/benchmark.
- pyright reports 0 errors.
- If a C++ build fails on a zero-length array warning, add `-Wno-zero-length-array` only for clang in `_build_flags`. gcc accepts it silently.

- [ ] **Step 9: Check the hit path did not regress**

```
PYTHONPATH=$PWD taskset -c 0 $V/bin/python benchmarks/bench_launch.py --last-key --iters 20000 --batches 7
```

Expected: within run-to-run noise of `docs/benchmark-2026-09-25-last-key.md`. A one-word key now pays the bijection on a memo hit, about 1 ns. Anything above 3 ns is a regression to fix before committing.

- [ ] **Step G: Benchmark gate**

```bash
/tmp/intj-bench/run_all.sh task1
```

Compare with the baseline and write `benchmarks/journals/2026-09-26_autotune-task1_0_<sha>.md` per the Benchmark Gate. Expected: no row regresses. Task 1 accepts at most about 1 ns on one-word keys from the memo's bijection (see Step 9); anything more is fixed here. Add the journal to this task's `git add`.

- [ ] **Step 10: Commit**

```bash
git add benchmarks/journals/2026-09-26_autotune-task1_0_*.md \
  intj/runtime/intj_map.h intj/runtime/intj_runtime.h intj/python_intf/cpython_abi.h \
  intj/python_intf/AGENTS.md intj/runtime/entry.c.jinja tests/bench_kernel_cache.cpp \
  tests/test_kernel_cache.py tests/test_return_compiled.py tests/test_launcher.py AGENTS.md
git commit -m "Store cache records by value behind a read/write lock

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Analyze the wrapper chain

About 2 hours. Pure Python, and nothing launches.

**Files:**
- Create: `intj/heuristic.py` (parsing only; Task 4 adds lowering)
- Create: `intj/tuning.py` (analysis only; Task 3 adds the miss path)
- Test: `tests/test_tuning.py`

**Interfaces:**
- Produces:
  - `heuristic.HeuristicError(ValueError)`
  - `heuristic.Heuristic(name: str, inputs: tuple[str, ...], arg: str, expr: ast.expr, fn: Any, where: str)`
  - `heuristic.parse_heuristic(name: str, fn: object) -> Heuristic`
  - `tuning.Dependent(name: str, level: int)`
  - `tuning.Computed(name: str, level: int, heuristic: Heuristic)`
  - `tuning.TuningPlan(jit_func, layers, exact_keys: tuple[str, ...], dependent: tuple[Dependent, ...], computed: tuple[Computed, ...], levels: int)` with property `tuned: frozenset[str]`
  - `tuning.analyze(kernel: Any, fixed: Iterable[str] = ()) -> TuningPlan`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_tuning.py`:

```python
# pyright: standard
"""Role assignment for autotune and heuristics layers; nothing here launches."""

import pytest
import triton
import triton.language as tl

from intj.heuristic import HeuristicError, parse_heuristic
from intj.launcher import UnsupportedKernel
from intj.tuning import Computed, Dependent, analyze


@triton.jit
def mm(a, b, c, M, N, K, BLOCK_K: tl.constexpr, SPLIT_K: tl.constexpr, EVEN_K: tl.constexpr):
    pass


@triton.jit
def strided(x, N, stride, BLOCK: tl.constexpr, ALIGNED: tl.constexpr):
    pass


@triton.jit
def evens(x, N, EVEN_N: tl.constexpr):
    pass


CONFIGS_K = [
    triton.Config({"BLOCK_K": 32, "SPLIT_K": 1}),
    triton.Config({"BLOCK_K": 64, "SPLIT_K": 2}, num_warps=8),
]


def test_matmul_needs_one_lookup():
    kernel = triton.autotune(configs=CONFIGS_K, key=["M", "N", "K"])(
        triton.heuristics({"EVEN_K": lambda a: a["K"] % (a["BLOCK_K"] * a["SPLIT_K"]) == 0})(mm)
    )
    plan = analyze(kernel)
    assert plan.exact_keys == ("M", "N", "K")
    assert plan.levels == 1
    assert plan.computed == ()
    assert set(plan.dependent) == {
        Dependent(name, 0)
        for name in ("BLOCK_K", "SPLIT_K", "num_warps", "num_ctas", "num_stages", "EVEN_K")
    }
    assert plan.tuned >= {"BLOCK_K", "SPLIT_K", "EVEN_K"}


def test_heuristic_over_unkeyed_var_and_tuned_value_is_a_level_one_key():
    kernel = triton.autotune(configs=[triton.Config({"BLOCK": 32}), triton.Config({"BLOCK": 64})], key=["N"])(
        triton.heuristics({"ALIGNED": lambda a: a["stride"] % a["BLOCK"] == 0})(strided)
    )
    plan = analyze(kernel)
    assert plan.levels == 2
    assert [(c.name, c.level) for c in plan.computed] == [("ALIGNED", 1)]
    assert Dependent("BLOCK", 0) in plan.dependent


def test_heuristic_over_caller_var_is_a_level_zero_key():
    plan = analyze(triton.heuristics({"EVEN_N": lambda a: a["N"] % 2 == 0})(evens))
    assert plan.levels == 1
    assert plan.exact_keys == ()
    assert [(c.name, c.level) for c in plan.computed] == [("EVEN_N", 0)]


def test_heuristics_outside_autotune_feed_its_key():
    kernel = triton.heuristics({"EVEN_N": lambda a: a["N"] % 2 == 0})(
        triton.autotune(configs=[triton.Config({"BLOCK": 32})], key=["EVEN_N"])(strided_even)
    )
    plan = analyze(kernel)
    assert [(c.name, c.level) for c in plan.computed] == [("EVEN_N", 0)]
    assert Dependent("BLOCK", 0) in plan.dependent


@triton.jit
def strided_even(x, N, BLOCK: tl.constexpr, EVEN_N: tl.constexpr):
    pass


def test_baked_var_counts_as_exact():
    plan = analyze(
        triton.heuristics({"EVEN_N": lambda a: a["N"] % 2 == 0})(evens), fixed={"N"}
    )
    assert plan.computed == ()
    assert plan.dependent == (Dependent("EVEN_N", 0),)


@pytest.mark.parametrize(
    "build,match",
    [
        (lambda: triton.autotune(configs=[triton.Config({"N": 1})], key=[])(evens), "runtime parameter 'N'"),
        (lambda: triton.heuristics({"EVEN_N": lambda a: a["EVEN_N"]})(evens), "its own result"),
        (lambda: triton.heuristics({"EVEN_N": lambda a: a["nope"]})(evens), "'nope'"),
        (
            lambda: triton.heuristics({"EVEN_N": lambda a: 1})(triton.heuristics({"EVEN_N": lambda a: 0})(evens)),
            "two layers",
        ),
        (lambda: triton.heuristics({"EVEN_N": lambda a: a.get("N")})(evens), "string literal"),
    ],
)
def test_refusals(build, match):
    with pytest.raises(UnsupportedKernel, match=match):
        analyze(build())


def test_parse_finds_the_right_lambda_on_a_shared_line():
    first, second = (lambda a: a["N"] + 1), (lambda a: a["M"] * 2)  # noqa: E731
    assert parse_heuristic("X", first).inputs == ("N",)
    assert parse_heuristic("Y", second).inputs == ("M",)


def test_parse_accepts_a_single_return_def():
    def pick(args):
        """Doc."""
        return args["N"] > 4

    assert parse_heuristic("X", pick).inputs == ("N",)


def test_parse_refuses_free_variables():
    limit = 4
    with pytest.raises(HeuristicError, match="free variables"):
        parse_heuristic("X", lambda a: a["N"] > limit)
```

- [ ] **Step 2: Run them and confirm they fail**

```
PYTHONPATH=$PWD VIRTUAL_ENV=$V uv run --active --no-sync python -m pytest tests/test_tuning.py -q
```

Expected: collection error, `No module named 'intj.heuristic'`.

- [ ] **Step 3: Write `intj/heuristic.py` (parsing)**

```python
"""Read `@triton.heuristics` functions: which names each one reads.

Task-4 lowering to C lives here too; this part only parses.
"""

from __future__ import annotations

import ast
import dataclasses
import inspect
import linecache
import types
from typing import Any, cast

# Python 3.8 wraps a subscript's index in ast.Index; later versions do not.
_INDEX: Any = getattr(ast, "Index", ())


class HeuristicError(ValueError):
    """A heuristic function is outside the subset intj reads."""


@dataclasses.dataclass(frozen=True)
class Heuristic:
    name: str  # the value it assigns
    inputs: tuple[str, ...]  # names it reads, first-use order
    arg: str  # its one parameter
    expr: ast.expr = dataclasses.field(compare=False, hash=False, repr=False)
    fn: Any = dataclasses.field(compare=False, hash=False, repr=False)
    where: str = ""  # file:line, for messages


def parse_heuristic(name: str, fn: object) -> Heuristic:
    """Locate `fn`'s source without calling it, and list the names it reads."""
    if type(fn) is not types.FunctionType:
        raise HeuristicError(f"heuristic {name!r} must be a lambda or def")
    code = fn.__code__
    if code.co_freevars:
        raise HeuristicError(
            f"heuristic {name!r} cannot reference free variables {code.co_freevars}"
        )
    varargs = code.co_flags & (inspect.CO_VARARGS | inspect.CO_VARKEYWORDS)
    if (
        code.co_argcount != 1
        or code.co_posonlyargcount
        or code.co_kwonlyargcount
        or varargs
        or fn.__defaults__
    ):
        raise HeuristicError(f"heuristic {name!r} must take exactly one argument")
    node = _locate(fn)
    where = f"{code.co_filename}:{node.lineno}"
    if isinstance(node, ast.Lambda):
        expr = node.body
        arg = node.args.args[0].arg
    else:
        body = node.body
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and type(body[0].value.value) is str
        ):
            body = body[1:]
        if len(body) != 1 or not isinstance(body[0], ast.Return) or body[0].value is None:
            raise HeuristicError(f"{where}: heuristic {name!r} must be a single return")
        expr = body[0].value
        arg = (node.args.posonlyargs + node.args.args)[0].arg
    inputs: dict[str, None] = {}
    subscripts = 0
    for sub in ast.walk(expr):
        key = _subscript_key(sub, arg, where)
        if key is not None:
            inputs[key] = None
            subscripts += 1
    uses = sum(isinstance(sub, ast.Name) and sub.id == arg for sub in ast.walk(expr))
    if uses != subscripts:
        raise HeuristicError(
            f"{where}: heuristic {name!r} may use {arg!r} only as {arg}[\"name\"] with a string literal"
        )
    return Heuristic(name, tuple(inputs), arg, expr, fn, where)


def _subscript_key(node: ast.AST, arg: str, where: str) -> str | None:
    if not (
        isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Name)
        and node.value.id == arg
    ):
        return None
    index: Any = node.slice
    if isinstance(index, _INDEX):
        index = index.value
    if isinstance(index, ast.Constant) and type(index.value) is str:
        return cast(str, index.value)
    raise HeuristicError(f"{where}: index {arg!r} with a string literal")


def _locate(fn: types.FunctionType) -> ast.Lambda | ast.FunctionDef:
    """The one lambda or def in `fn`'s file that compiles to `fn`'s code.

    Several lambdas can share a line, and inspect.getsource returns lines, not
    nodes; compiling each candidate and comparing bytecode picks the right one.
    """
    code = fn.__code__
    lines = linecache.getlines(code.co_filename, fn.__globals__)
    if not lines:
        raise HeuristicError(f"heuristic {fn.__name__!r} needs available Python source")
    try:
        tree = ast.parse("".join(lines))
    except SyntaxError as error:
        raise HeuristicError(f"cannot parse {code.co_filename}") from error
    matches: list[ast.Lambda | ast.FunctionDef] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Lambda):
            first = node.lineno
        elif isinstance(node, ast.FunctionDef) and node.name == fn.__name__:
            first = min([d.lineno for d in node.decorator_list] + [node.lineno])
        else:
            continue
        if first == code.co_firstlineno and _same_code(node, code):
            matches.append(node)
    if len(matches) != 1:
        raise HeuristicError(
            f"cannot locate the source of heuristic {fn.__name__!r} at "
            f"{code.co_filename}:{code.co_firstlineno}"
        )
    return matches[0]


def _same_code(node: ast.Lambda | ast.FunctionDef, code: types.CodeType) -> bool:
    if isinstance(node, ast.Lambda):
        compiled = compile(ast.Expression(body=node), code.co_filename, "eval")
    else:
        compiled = compile(ast.Module(body=[node], type_ignores=[]), code.co_filename, "exec")
    return any(
        isinstance(const, types.CodeType)
        and const.co_code == code.co_code
        and const.co_names == code.co_names
        and const.co_varnames == code.co_varnames
        and const.co_consts == code.co_consts
        for const in compiled.co_consts
    )
```

- [ ] **Step 4: Write `intj/tuning.py` (analysis)**

```python
"""Autotune and heuristics layers between `make_launcher` and a `@triton.jit`.

`analyze` gives every name a role, visiting layers from the outermost inward:

- an *exact key*: a caller var an autotune layer keys on, keyed by its value;
- a *dependent* value: fixed by keys at or below its level, stored in that
  level's cache record;
- a *computed* key: a heuristic reading a caller var nothing keys, lowered to C
  and keyed at the level where its inputs become known.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable
from typing import Any

from .heuristic import Heuristic, HeuristicError, parse_heuristic
from .launcher import UnsupportedKernel


@dataclasses.dataclass(frozen=True)
class Dependent:
    name: str
    level: int  # stored in this level's record


@dataclasses.dataclass(frozen=True)
class Computed:
    name: str
    level: int  # keyed by this level's lookup
    heuristic: Heuristic


@dataclasses.dataclass(frozen=True)
class TuningPlan:
    jit_func: Any = dataclasses.field(compare=False)
    layers: tuple[Any, ...] = dataclasses.field(compare=False)  # outermost first
    exact_keys: tuple[str, ...] = ()
    dependent: tuple[Dependent, ...] = ()
    computed: tuple[Computed, ...] = ()
    levels: int = 1

    @property
    def tuned(self) -> frozenset[str]:
        """Every name a layer assigns: removed from the call, never passed by it."""
        return frozenset(
            [d.name for d in self.dependent] + [c.name for c in self.computed]
        )


def analyze(kernel: Any, fixed: Iterable[str] = ()) -> TuningPlan:
    """Assign roles for `kernel`, an Autotuner/Heuristics chain over a JITFunction.

    `fixed` names baked parameters: their values are in the module, so exact.
    """
    from triton.runtime.autotuner import Autotuner, Heuristics
    from triton.runtime.jit import JITFunction

    layers: list[Any] = []
    inner = kernel
    while type(inner) in (Autotuner, Heuristics):
        layers.append(inner)
        inner = inner.fn
    if not layers:
        raise UnsupportedKernel("intj: expected an autotune or heuristics wrapper")
    if not isinstance(inner, JITFunction):
        raise UnsupportedKernel(
            f"intj: the innermost layer must be a @triton.jit function, got {type(inner).__name__}"
        )
    params = {p.name: p for p in inner.params}
    frozen = frozenset(fixed)
    # name -> (exact, level, known): `level` is the key level of an exact value
    # (or its record level, if dependent); `known` is the lookup count after
    # which C can read it.
    info: dict[str, tuple[bool, int, int]] = {
        name: (bool(p.is_constexpr) or name in frozen, 0, 0) for name, p in params.items()
    }
    assigned: set[str] = set()
    exact_keys: list[str] = []
    dependent: list[Dependent] = []
    computed: list[Computed] = []

    def assign(name: str, what: str) -> None:
        if name in assigned:
            raise UnsupportedKernel(f"intj: {name!r} is assigned by two layers")
        if name in params and not params[name].is_constexpr:
            raise UnsupportedKernel(
                f"intj: {what} assigns runtime parameter {name!r}; only tl.constexpr "
                "parameters and compile options can be tuned"
            )
        assigned.add(name)

    for layer in layers:
        if type(layer) is Autotuner:
            level = 0
            for key in layer.keys:
                if key not in params:
                    continue  # Triton filters keys to parameter names too
                exact, _, known = info[key]
                if not exact:
                    info[key] = (True, known, known)
                    exact_keys.append(key)
                level = max(level, info[key][1])
            names: dict[str, None] = {}
            for config in layer.configs:
                for name in config.all_kwargs():
                    names[name] = None
            for name in names:
                assign(name, "autotune")
                info[name] = (True, level, level + 1)
                dependent.append(Dependent(name, level))
            continue
        for name, fn in layer.values.items():
            try:
                heuristic = parse_heuristic(name, fn)
            except HeuristicError as error:
                raise UnsupportedKernel(f"intj: {error}") from error
            for read in heuristic.inputs:
                if read not in info:
                    raise UnsupportedKernel(
                        f"intj: heuristic {name!r} reads {read!r}, which is neither a "
                        "kernel parameter nor assigned by an outer layer"
                    )
            if name in heuristic.inputs:
                raise UnsupportedKernel(f"intj: heuristic {name!r} reads its own result")
            assign(name, "heuristic")
            if all(info[read][0] for read in heuristic.inputs):
                level = max((info[read][1] for read in heuristic.inputs), default=0)
                info[name] = (True, level, level + 1)
                dependent.append(Dependent(name, level))
            else:
                level = max((info[read][2] for read in heuristic.inputs), default=0)
                info[name] = (True, level, level)
                computed.append(Computed(name, level, heuristic))
    levels = 1 + max((c.level for c in computed), default=0)
    return TuningPlan(
        inner, tuple(layers), tuple(exact_keys), tuple(dependent), tuple(computed), levels
    )
```

- [ ] **Step 5: Run the tests and pyright**

```
PYTHONPATH=$PWD VIRTUAL_ENV=$V uv run --active --no-sync python -m pytest tests/test_tuning.py -q
PYTHONPATH=$PWD VIRTUAL_ENV=$V uv run --active --no-sync pyright
```

Expected: all tests in `test_tuning.py` pass, and pyright reports 0 errors.

- [ ] **Step G: Benchmark gate**

```bash
/tmp/intj-bench/run_all.sh task2
```

Compare with the baseline and write `benchmarks/journals/2026-09-26_autotune-task2_0_<sha>.md` per the Benchmark Gate. Expected: no row regresses. Add the journal to this task's `git add`.

- [ ] **Step 6: Commit**

```bash
git add benchmarks/journals/2026-09-26_autotune-task2_0_*.md \
  intj/heuristic.py intj/tuning.py tests/test_tuning.py
git commit -m "Assign tuning roles to autotune and heuristics layers

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Single-level tuned launchers end to end

About 1.5 days. This covers autotune, plus heuristics whose inputs are all exact: one lookup, with dependent values in the record. Computed keys are refused here, and Task 4 lifts that refusal.

**Files:**
- Modify: `intj/annotation.py` (`CanonicalAnnotation`, `_key_fields`)
- Modify: `intj/launcher.py`: `TuningRender`, `RenderContext.tuning`, `LauncherFactory.tuned`, `_render_params`, `make_launcher`, `_materialize_module`, `_loaded_module`, `_check_kernel`, `_checked_compile`
- Modify: `intj/grid.py` (`compile_grid(..., deps=...)`)
- Modify: `intj/tuning.py` (private chain, shim, callback)
- Modify: `intj/runtime/intj_runtime.h` (bound `tuned_cb`, `intj_grid_input_dep`)
- Modify: `intj/runtime/entry.c.jinja`
- Test: `tests/test_tuned.py`

**Interfaces:**
- Consumes: `analyze`, `TuningPlan` (Task 2); `INTJ_DEFINE_CACHE`, `intj_final`, `intj_launch`, `INTJ_V0_*` (Task 1)
- Produces:
  - `CanonicalAnnotation.tuned: bool = False`, `CanonicalAnnotation.exact_key: bool = False`
  - `KeyField` kinds `"exact_kind"` (width 1) and `"exact"` (width 8)
  - `launcher.TuningRender(levels, computed, computed_fields, computed_names, deps, dep_names, meta_names, source)`
  - `launcher._checked_compile(jit_func, compiler_input, target, canonical_options) -> CompiledKernel`
  - `tuning.make_tuned_callback(plan, resolved: tuple[ResolvedParam, ...], options: Mapping[str, Any], grid: TunedGrid, render: TuningRender, return_compiled: bool) -> Callable`
  - `tuning.TunedGrid(mode: str, fn: Any = None, positional: tuple[str, ...] = (), extras: tuple[str, ...] = ())`, where mode is one of `"dims"`, `"cpp"`, `"py"`
- Produces, the C callback protocol:
  - C calls `cb(keyblob, nparams, device, stream, controls, *public_args)`.
  - `controls` is `(gx, gy, gz)` for the default grid and `grid_arg`, the tuple of grid extras for `grid_cpp`, and `None` for `grid_py`.
  - The callback returns `(function, block_dim, shared, nparams, compiled, deps, computed, meta)`:
    - `deps` holds one int or bool per `render.dep_names`
    - `computed` holds one value per `render.computed_names`
    - `meta` is a tuple per `render.meta_names`, or `None`

- [ ] **Step 1: Write the failing end-to-end tests**

Create `tests/test_tuned.py`:

```python
# pyright: standard
"""Autotuned and heuristic kernels through make_launcher, on the GPU."""

import threading

import pytest
import torch
import triton
import triton.language as tl

from intj import Argument, BindValue, Constexpr, make_launcher
from intj.launcher import UnsupportedKernel


@triton.jit
def tagged(x, out, n, TAG: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(out + offs, tl.load(x + offs, mask=mask) + TAG, mask=mask)


def configs():
    return [
        triton.Config({"TAG": 1, "BLOCK": 32}),
        triton.Config({"TAG": 2, "BLOCK": 64}),
    ]


class Bench:
    """Deterministic do_bench: reads the TAG the config wrote, scores by table."""

    def __init__(self, out, table):
        self.out, self.table, self.calls, self.key = out, table, [], None

    def __call__(self, kernel_call, quantiles):
        self.out.zero_()
        kernel_call()
        tag = int(self.out[0].item())
        self.calls.append(tag)
        score = self.table[self.key][tag]
        return [score, score, score]


def grid(n: int, BLOCK: int):
    return (triton.cdiv(n, BLOCK),)


def controls():
    return torch.cuda.current_device(), torch.cuda.current_stream().cuda_stream


def tuned_launcher(bench, **kwargs):
    kernel = triton.autotune(configs=configs(), key=["n"], do_bench=bench)(tagged)
    return kernel, make_launcher(kernel, grid_cpp=grid, **kwargs)


def test_tunes_on_miss_and_launches_natively_on_hit():
    x = torch.zeros(256, device="cuda", dtype=torch.int32)
    out = torch.zeros_like(x)
    bench = Bench(out, {16: {1: 1.0, 2: 2.0}, 64: {1: 2.0, 2: 1.0}})
    kernel, launch = tuned_launcher(bench)
    device, stream = controls()
    for n, tag in ((16, 1), (64, 2)):
        bench.key = n
        out.zero_()
        launch(device, stream, x, out, n)
        torch.cuda.synchronize()
        assert int(out[0].item()) == tag
        assert kernel.best_config.kwargs["TAG"] == tag
    assert bench.calls == [1, 2, 1, 2]
    for n, tag in ((16, 1), (64, 2), (16, 1)):
        out.zero_()
        launch(device, stream, x, out, n)
        torch.cuda.synchronize()
        assert int(out[n - 1].item()) == tag
    assert bench.calls == [1, 2, 1, 2], "a hit must not reach Triton"


def test_float_autotune_key_is_exact():
    @triton.jit
    def scaled(x, out, s, TAG: tl.constexpr):
        tl.store(out, TAG + (s * 0).to(tl.int32))

    x = torch.zeros(1, device="cuda", dtype=torch.int32)
    out = torch.zeros_like(x)
    bench = Bench(out, {None: {1: 1.0, 2: 2.0}})
    kernel = triton.autotune(
        configs=[triton.Config({"TAG": 1}), triton.Config({"TAG": 2})], key=["s"], do_bench=bench
    )(scaled)
    launch = make_launcher(kernel)
    device, stream = controls()
    for s in (1.0, 1.5, 1.0):
        launch(device, stream, 1, x, out, s)
    assert bench.calls == [1, 2, 1, 2]


def test_heuristic_over_tuned_value_is_stored_not_recomputed():
    @triton.jit
    def half(out, TAG: tl.constexpr, HALF: tl.constexpr):
        tl.store(out, TAG * 10 + HALF)

    out = torch.zeros(1, device="cuda", dtype=torch.int32)
    bench = Bench(out, {None: {10: 2.0, 21: 1.0}})  # TAG * 10 + HALF
    seen = []

    def pick_half(a):
        seen.append(a["TAG"])
        return a["TAG"] // 2

    kernel = triton.autotune(
        configs=[triton.Config({"TAG": 1}), triton.Config({"TAG": 2})], key=[], do_bench=bench
    )(triton.heuristics({"HALF": pick_half})(half))
    launch = make_launcher(kernel)
    device, stream = controls()
    launch(device, stream, 1, out)
    calls = len(seen)
    launch(device, stream, 1, out)
    torch.cuda.synchronize()
    assert int(out.item()) == 21
    assert len(seen) == calls, "a dependent heuristic runs on misses only"


@pytest.mark.parametrize("mode", ["grid_arg", "default", "grid_py"])
def test_grid_modes(mode):
    x = torch.zeros(256, device="cuda", dtype=torch.int32)
    out = torch.zeros_like(x)
    bench = Bench(out, {None: {1: 2.0, 2: 1.0}})
    kernel = triton.autotune(configs=configs(), key=[], do_bench=bench)(tagged)
    device, stream = controls()
    metas = []
    if mode == "grid_arg":
        launch = make_launcher(kernel, grid_arg=1)
        call = lambda: launch(device, stream, 4, x, out, 256)  # noqa: E731
    elif mode == "default":
        launch = make_launcher(kernel)
        call = lambda: launch(device, stream, (4,), x, out, 256)  # noqa: E731
    else:

        def grid_py(meta):
            metas.append(meta["BLOCK"])
            return (triton.cdiv(meta["n"], meta["BLOCK"]),)

        launch = make_launcher(kernel, grid_py=grid_py)
        call = lambda: launch(device, stream, x, out, 256)  # noqa: E731
    call()
    out.zero_()
    call()
    torch.cuda.synchronize()
    assert int(out[255].item()) == 2
    if mode == "grid_py":
        assert metas[-1] == 64


def test_return_compiled_is_the_selected_kernel():
    x = torch.zeros(64, device="cuda", dtype=torch.int32)
    out = torch.zeros_like(x)
    bench = Bench(out, {None: {1: 2.0, 2: 1.0}})
    kernel = triton.autotune(configs=configs(), key=[], do_bench=bench)(tagged)
    launch = make_launcher(kernel, grid_cpp=grid, return_compiled=True)
    device, stream = controls()
    first = launch(device, stream, x, out, 64)
    assert launch(device, stream, x, out, 64) is first
    assert first.src.constants[(3,)] == 2


def test_tuned_param_before_public_params():
    @triton.jit
    def early(out, TAG: tl.constexpr, n):
        tl.store(out + n - n, TAG)

    out = torch.zeros(1, device="cuda", dtype=torch.int32)
    bench = Bench(out, {None: {1: 2.0, 2: 1.0}})
    kernel = triton.autotune(
        configs=[triton.Config({"TAG": 1}), triton.Config({"TAG": 2})], key=["n"], do_bench=bench
    )(early)
    launch = make_launcher(kernel)
    device, stream = controls()
    launch(device, stream, 1, out, 7)
    torch.cuda.synchronize()
    assert int(out.item()) == 2


def test_concurrent_misses_tune_once():
    x = torch.zeros(256, device="cuda", dtype=torch.int32)
    out = torch.zeros_like(x)
    bench = Bench(out, {16: {1: 1.0, 2: 2.0}})
    bench.key = 16
    _, launch = tuned_launcher(bench)
    device, _ = controls()
    errors = []

    def worker():
        try:
            with torch.cuda.stream(torch.cuda.Stream()):
                launch(device, torch.cuda.current_stream().cuda_stream, x, out, 16)
        except Exception as error:  # pragma: no cover - reported below
            errors.append(error)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert bench.calls == [1, 2]


def test_config_compile_options_are_per_record():
    @triton.jit
    def fill(out, n, TAG: tl.constexpr, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        tl.store(out + offs, TAG, mask=offs < n)

    out = torch.zeros(1024, device="cuda", dtype=torch.int32)
    bench = Bench(out, {128: {1: 1.0, 2: 2.0}, 1024: {1: 2.0, 2: 1.0}})
    kernel = triton.autotune(
        configs=[
            triton.Config({"TAG": 1, "BLOCK": 256}, num_warps=1),
            triton.Config({"TAG": 2, "BLOCK": 256}, num_warps=8),
        ],
        key=["n"],
        do_bench=bench,
    )(fill)
    launch = make_launcher(kernel, grid_cpp=lambda_grid, return_compiled=True)
    device, stream = controls()
    for n, warps in ((128, 1), (1024, 8)):
        bench.key = n
        out.zero_()
        compiled = launch(device, stream, out, n)
        torch.cuda.synchronize()
        assert compiled.metadata.num_warps == warps
        assert int(out[:n].min().item()) == int(out[:n].max().item()) != 0


def lambda_grid(n: int, BLOCK: int):
    return (triton.cdiv(n, BLOCK),)


def test_failed_tuning_leaves_no_record():
    x = torch.zeros(64, device="cuda", dtype=torch.int32)
    out = torch.zeros_like(x)
    bench = Bench(out, {None: {1: 2.0, 2: 1.0}})
    failures = [RuntimeError("prune failed")]

    def prune(configs, named_args, **kwargs):
        if failures:
            raise failures.pop()
        return configs

    kernel = triton.autotune(
        configs=configs(), key=[], do_bench=bench, prune_configs_by={"early_config_prune": prune}
    )(tagged)
    launch = make_launcher(kernel, grid_cpp=grid)
    device, stream = controls()
    with pytest.raises(RuntimeError, match="prune failed"):
        launch(device, stream, x, out, 64)
    launch(device, stream, x, out, 64)
    torch.cuda.synchronize()
    assert int(out[0].item()) == 2


def test_autotune_key_refuses_a_tensor_at_call():
    x = torch.zeros(64, device="cuda", dtype=torch.int32)
    kernel = triton.autotune(configs=configs(), key=["n"])(tagged)
    launch = make_launcher(kernel, grid_arg=1)
    device, stream = controls()
    with pytest.raises(TypeError, match="autotune key 'n'"):
        launch(device, stream, 1, x, x, x)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"no_gpu": True}, "no_gpu"),
        ({"options": {"num_warps": 8}}, "num_warps"),
        ({"extra_annotation": {"TAG": Constexpr(value=1)}}, "TAG"),
        ({"extra_annotation": {"x": Argument(bind_value=BindValue.TENSOR)}}, "bound"),
    ],
)
def test_refusals(kwargs, match):
    kernel = triton.autotune(configs=configs(), key=["n"])(tagged)
    with pytest.raises(UnsupportedKernel, match=match):
        make_launcher(kernel, **kwargs)


def test_computed_keys_are_refused_until_task_4():
    @triton.jit
    def evens(out, n, EVEN: tl.constexpr):
        tl.store(out, EVEN)

    with pytest.raises(UnsupportedKernel, match="computed"):
        make_launcher(triton.heuristics({"EVEN": lambda a: a["n"] % 2 == 0})(evens))
```

`test_computed_keys_are_refused_until_task_4` is deleted in Task 4. It exists so this task's refusal is pinned.

- [ ] **Step 2: Run them and confirm they fail**

```
PYTHONPATH=$PWD VIRTUAL_ENV=$V uv run --active --no-sync python -m pytest tests/test_tuned.py -q
```

Expected: every test fails with `UnsupportedKernel: intj: expected a @triton.jit function, got Autotuner`, or with `Heuristics` in place of `Autotuner`.

- [ ] **Step 3: Annotations: tuned and exact-key flags**

In `intj/annotation.py`, add two fields to `CanonicalAnnotation` after `key_fields`:

```python
    #: assigned by an autotune/heuristics layer: never passed, never keyed
    tuned: bool = False
    #: a caller var an autotune layer keys on: keyed by its exact value
    exact_key: bool = False
```

In `_key_fields`, make the first lines:

```python
    if annotation.tuned or annotation.baked_value or annotation.bind_value is not None:
        return ()
```

Wrap its final `return` so exact keys add their fields. The kind byte tells `1` from `True` from `1.0`, and the 8 bytes hold int64/uint64 bits or fp64 bits:

```python
    fields = (KeyField("descriptor", 1),) if descriptor else ()
    if annotation.exact_key:
        fields += (KeyField("exact_kind", 1), KeyField("exact", 8))
    return fields
```

- [ ] **Step 4: Grid reads dependent values**

In `intj/grid.py`, change the signature to

```python
def compile_grid(
    fn: object,
    params: Sequence[Param],
    baked: Mapping[int, object],
    deps: Mapping[str, int] | None = None,
) -> GridCode:
```

Update the docstring to say: `deps` maps tuned parameter names to their slot in the record's dependent values. When it is given, the evaluator takes them as `const int64_t *dep`.

Replace the function header line with:

```python
    lines = [
        "static int intj_eval_grid(PyObject *const *args, "
        + ("const int64_t *dep, " if deps is not None else "")
        + "uint32_t dims[3]) {"
    ]
    if deps is not None:
        lines.append("  (void)dep;")
```

In the `else:` branch for non-extra names, check tuned names first:

```python
        else:
            param = by_name[name]
            if deps is not None and name in deps:
                slot = deps[name]
                lines.append(f"  int64_t {var};")
                lines.append(
                    f"  if (intj_grid_input_dep(dep[{2 * slot}], dep[{2 * slot + 1}], "
                    f"{1 if kind == 'bool' else 0}, {json.dumps(name)}, &{var}) != 0) return -1;"
                )
            elif param.call_index is not None:
```

The existing `elif param.call_index is not None:` becomes that `elif`, and the rest is unchanged. Add to `intj_runtime.h`, next to `intj_grid_input_bool`:

```c
/* A dependent value from a tuned record: (value, 1 if it was a bool). */
static INTJ_ALWAYS_INLINE int intj_grid_input_dep(int64_t value, int64_t is_bool,
                                                  int want_bool, const char *name,
                                                  int64_t *out) {
  if (INTJ_UNLIKELY(is_bool != want_bool)) {
    PyErr_Format(PyExc_TypeError, "intj: grid argument '%s' must be %s", name,
                 want_bool ? "a bool" : "an int");
    return -1;
  }
  *out = value;
  return 0;
}
```

- [ ] **Step 5: `TuningRender`, `_checked_compile`, `make_launcher` integration**

In `intj/launcher.py`:

(a) Add after `Param`:

```python
@dataclasses.dataclass(frozen=True)
class TuningRender:
    """What `entry.c.jinja` renders for an autotune/heuristics launcher."""

    levels: int  # lookups per launch
    computed: tuple[int, ...]  # computed keys keyed at each level
    #: level-0 computed keys: (value, kind) byte offsets in the level-0 key
    computed_fields: tuple[tuple[int, int], ...]
    computed_names: tuple[str, ...]  # level-major; the callback's `computed`
    deps: tuple[int, ...]  # dependent values C reads, stored per level
    dep_names: tuple[str, ...]  # level-major; the callback's `deps`
    meta_names: tuple[str, ...]  # grid_py: tuned parameters, declaration order
    source: str  # generated `intj_tuned_level_<n>` functions
```

and to `RenderContext` (last field): `tuning: TuningRender | None = None`.

(b) Factor compilation out of `_make_compile_callback`:

```python
def _checked_compile(
    jit_func: JitFunction, compiler_input: CompilerInput, target: Any, canonical_options: Any
) -> CompiledKernel:
    """Compile, load on the current device, and refuse what intj cannot launch."""
    from triton.compiler import compile as triton_compile

    kernel = triton_compile(
        compiler_input.ast_source(jit_func), target=target, options=canonical_options.__dict__
    )
    kernel._init_handles()
    md = kernel.metadata
    if md.num_ctas != 1:
        raise UnsupportedKernel("intj: num_ctas > 1 is not supported")
    if md.launch_cooperative_grid:
        raise UnsupportedKernel("intj: launch_cooperative_grid is not supported")
    if getattr(md, "launch_pdl", False):  # nvidia only
        raise UnsupportedKernel("intj: launch_pdl is not supported")
    if getattr(md, "global_scratch_size", 0) or md.profile_scratch_size:
        raise UnsupportedKernel("intj: kernels requiring scratch memory are not supported")
    return kernel
```

`compile_callback` then calls `kernel = _checked_compile(jit_func, compiler_input, target, canonical_options)` in place of its compile-and-check block.

(c) `_render_params`: `public = not annotation.baked_value and annotation.bind_value is None and not annotation.tuned`.

(d) `_check_kernel`: change the non-JIT message to `"intj: expected a @triton.jit function, optionally wrapped in @triton.autotune / @triton.heuristics, got {type}"`.

(e) In `make_launcher`, replace `jit_func = _check_kernel(jit_func, options)` / `resolved = _resolve_annotations(...)` with:

```python
    from triton.runtime.autotuner import Autotuner, Heuristics

    chain = jit_func
    while type(jit_func) in (Autotuner, Heuristics):
        jit_func = jit_func.fn
    jit_func = _check_kernel(jit_func, options)
    resolved = _resolve_annotations(jit_func, extra_annotation)
    tuned: _Tuned | None = None
    if chain is not jit_func:
        tuned = _plan_tuning(chain, resolved, extra_annotation, options, no_gpu)
        resolved = tuple(
            dataclasses.replace(
                p,
                annotation=dataclasses.replace(
                    p.annotation,
                    tuned=p.name in tuned.plan.tuned,
                    exact_key=p.name in tuned.plan.exact_keys,
                ),
            )
            for p in resolved
        )
```

(f) Add the planner. `_Tuned` is internal state, so it is not in `ModuleKey`:

```python
@dataclasses.dataclass(frozen=True)
class _Tuned:
    plan: Any  # tuning.TuningPlan
    render: TuningRender | None = None  # filled once the grid is known


def _plan_tuning(
    chain: Any,
    resolved: tuple[ResolvedParam, ...],
    extra_annotation: Mapping[str, object] | None,
    options: Mapping[str, Any],
    no_gpu: bool,
) -> _Tuned:
    from triton.runtime.autotuner import Autotuner

    from .tuning import analyze

    if no_gpu:
        raise UnsupportedKernel("intj: autotune/heuristics launchers need a GPU; no_gpu=True is not supported")
    plan = analyze(chain, fixed=[p.name for p in resolved if p.annotation.baked_value])
    if plan.computed:
        raise UnsupportedKernel(
            f"intj: computed heuristic keys {[c.name for c in plan.computed]} are not supported yet"
        )
    for p in resolved:
        if p.annotation.bind_value is not None:
            raise UnsupportedKernel(f"intj: bound parameter {p.name!r} cannot be tuned over")
    clash = sorted(set(extra_annotation or {}) & plan.tuned)
    if clash:
        raise UnsupportedKernel(f"intj: extra_annotation names tuned value(s) {clash}")
    owned = sorted(set(options) & plan.tuned)
    if owned:
        raise UnsupportedKernel(f"intj: options {owned} are set by the autotune configs")
    target = _current_target()
    names = {p.name for p in resolved}
    for layer in plan.layers:
        if type(layer) is Autotuner:
            for config in layer.configs:
                config_options = {k: v for k, v in config.all_kwargs().items() if k not in names}
                if config_options.get("num_ctas", 1) != 1:
                    raise UnsupportedKernel("intj: num_ctas > 1 is not supported")
                _canonical_options(target, {**options, **config_options})
    return _Tuned(plan)
```

(g) Grid: when `tuned` is set and `grid_cpp` is given, compile it with dep slots for the tuned names it reads. Then build the render:

```python
    if grid_cpp is not None:
        from .grid import GridError, compile_grid

        grid_params, _, _ = _render_params(resolved, DeviceBinding.NOT_FIXED)
        grid_code = getattr(grid_cpp, "__code__", None)
        positional = grid_code.co_varnames[: grid_code.co_argcount] if grid_code else ()
        dep_names = tuple(n for n in positional if tuned is not None and n in tuned.plan.tuned)
        try:
            compiled_grid = compile_grid(
                grid_cpp,
                grid_params,
                {p.index: p.baked for p in resolved if p.annotation.baked_value},
                {n: i for i, n in enumerate(dep_names)} if tuned is not None else None,
            )
        except GridError as error:
            raise UnsupportedKernel(f"intj: {error}") from error
    else:
        dep_names = ()
    if tuned is not None:
        tuned = dataclasses.replace(
            tuned,
            render=TuningRender(
                levels=1,
                computed=(0,),
                computed_fields=(),
                computed_names=(),
                deps=(len(dep_names),),
                dep_names=dep_names,
                meta_names=tuple(p.name for p in resolved if p.annotation.tuned)
                if grid_py is not None
                else (),
                source="",
            ),
        )
```

(h) Tuned launchers always bind, so each bound handle owns its cache, private tuners and callback. Make the factory condition `if needs_binding or grid_py is not None or tuned is not None:`, pass `tuned=tuned, grid_fn=grid_cpp` to `LauncherFactory` (new fields `tuned: _Tuned | None = None`, `grid_fn: object | None = None`), and keep `return factory if needs_binding else factory.bind()`.

(i) `LauncherFactory._bind`: pass `tuning=self.tuned.render if self.tuned else None` to `_materialize_module`. Before `module.make_bound`, build the callback and prepend it:

```python
        prefix: tuple[object, ...] = ()
        if self.tuned is not None:
            from .tuning import TunedGrid, make_tuned_callback

            assert self.tuned.render is not None
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
                    tuple(resolved),
                    dict(self.options),
                    grid,
                    self.tuned.render,
                    self.return_compiled,
                ),
            )
```

Every `module.make_bound(` call gets `*prefix` as its first arguments.

(j) `_materialize_module(..., tuning: TuningRender | None = None)`: pass it into `RenderContext(tuning=tuning)` and set `nwords = max(nwords, 1) if tuning is not None else nwords`, since tuned levels always use a map. In `_loaded_module`, when `context.tuning is not None`, install a module callback that raises `RuntimeError("intj: tuned modules compile through their bound launcher")`. Tuned launches never read it.

- [ ] **Step 6: The miss path in `intj/tuning.py`**

Append:

```python
import copy
import inspect
import threading
from collections.abc import Callable, Mapping


@dataclasses.dataclass(frozen=True)
class TunedGrid:
    """How the miss path hands Triton a grid, per launcher grid mode."""

    mode: str  # "dims" | "cpp" | "py"
    fn: Any = None
    positional: tuple[str, ...] = ()
    extras: tuple[str, ...] = ()


class _Shim:
    """The innermost layer of a private chain: compiles and launches through intj.

    Triton's layers call `run` for every benchmark candidate and last for the
    final launch, so after the outermost `run` returns, `final` is that call.
    """

    def __init__(self, jit_func: Any, compile_kernel: Callable[[dict[str, Any], dict[str, Any]], Any]):
        self.fn = jit_func  # Triton unwraps `.fn` to reach the JITFunction
        self._signature = inspect.signature(jit_func.fn)
        self._compile = compile_kernel
        self.final: tuple[dict[str, Any], dict[str, Any], Any] | None = None

    def run(self, *args: Any, grid: Any, warmup: bool, **kwargs: Any) -> Any:
        params = {k: v for k, v in kwargs.items() if k in self._signature.parameters}
        options = {k: v for k, v in kwargs.items() if k not in params}
        bound = self._signature.bind(*args, **params)
        bound.apply_defaults()
        values = dict(bound.arguments)
        kernel = self._compile(values, options)
        if not warmup:
            dims = tuple(grid(values) if callable(grid) else grid)
            dims = dims + (1,) * (3 - len(dims))
            if all(dims):
                kernel[dims](*values.values())
        self.final = (values, options, kernel)
        return kernel


def _private_chain(layers: tuple[Any, ...], shim: _Shim) -> list[Any]:
    """Shallow copies with their own tuner caches, bottoming out in `shim`."""
    from triton.runtime.autotuner import Autotuner

    private = [copy.copy(layer) for layer in layers]
    for outer, inner in zip(private, [*private[1:], shim]):
        outer.fn = inner
    for layer in private:
        if type(layer) is Autotuner:
            layer.cache = {}
            for name in ("best_config", "bench_time", "configs_timings", "nargs"):
                layer.__dict__.pop(name, None)
    return private


def _copy_back(originals: tuple[Any, ...], private: list[Any]) -> None:
    for original, mine in zip(originals, private):
        for name in ("best_config", "bench_time", "configs_timings"):
            if name in mine.__dict__:
                setattr(original, name, mine.__dict__[name])


def _c_scalar(name: str, value: object) -> object:
    if type(value) is bool or type(value) is int and -(1 << 63) <= value < (1 << 63):
        return value
    raise UnsupportedKernel(
        f"intj: {name!r} is read by a heuristic or grid_cpp but was tuned to {value!r}; "
        "only int and bool are supported"
    )


def make_tuned_callback(
    plan: TuningPlan,
    resolved: tuple[Any, ...],
    options: Mapping[str, Any],
    grid: TunedGrid,
    render: Any,
    return_compiled: bool,
) -> Callable[..., tuple[Any, ...]]:
    """The C miss callback for one bound launcher, which owns its private tuners."""
    from triton.compiler import make_backend

    from .launcher import (
        Param,
        _canonical_options,  # pyright: ignore[reportPrivateUsage]  # launcher internals
        _checked_compile,  # pyright: ignore[reportPrivateUsage]  # launcher internals
        _compiler_input,  # pyright: ignore[reportPrivateUsage]  # launcher internals
        _current_device,  # pyright: ignore[reportPrivateUsage]  # launcher internals
        _current_target,  # pyright: ignore[reportPrivateUsage]  # launcher internals
    )

    del return_compiled  # C decides whether the record keeps the object
    jit_func = plan.jit_func
    target = _current_target()
    backend = make_backend(target)
    # every parameter public: the shim sees the full call Triton makes
    params = tuple(Param(p.name, p.index, p.index, p.annotation) for p in resolved)
    compiled: dict[tuple[Any, str], Any] = {}  # keeps every CompiledKernel alive

    def compile_kernel(values: dict[str, Any], config_options: dict[str, Any]) -> Any:
        canonical = _canonical_options(target, {**options, **config_options})
        compiler_input = _compiler_input(
            jit_func, params, [values[p.name] for p in params], backend
        )
        key = (compiler_input, canonical.hash())
        kernel = compiled.get(key)
        if kernel is None:
            kernel = compiled[key] = _checked_compile(jit_func, compiler_input, target, canonical)
        return kernel

    shim = _Shim(jit_func, compile_kernel)
    private = _private_chain(plan.layers, shim)
    public = [p.name for p in resolved if not p.annotation.baked_value and not p.annotation.tuned]
    baked = {p.name: p.baked for p in resolved if p.annotation.baked_value}
    lock = threading.RLock()
    running = [False]

    def callback(
        keyblob: bytes, nparams: int, device: int, stream: int, controls: Any, *args: Any
    ) -> tuple[Any, ...]:
        del keyblob
        with lock:
            if running[0]:
                raise UnsupportedKernel("intj: a tuned launcher was re-entered while tuning")
            running[0] = True
            try:
                return tune(nparams, device, stream, controls, args)
            finally:
                running[0] = False

    def tune(nparams: int, device: int, stream: int, controls: Any, args: tuple[Any, ...]) -> tuple[Any, ...]:
        import torch

        current = _current_device()
        if device != current:
            raise UnsupportedKernel(
                f"intj: launching on device {device} while device {current} is current; "
                "make the target device current before the first launch"
            )
        named = {**dict(zip(public, args)), **baked}
        # Positional up to the first tuned parameter, as a Triton caller would
        # write it: prune functions read `named_args`, which is positional only.
        prefix: list[Any] = []
        for name in jit_func.arg_names:
            if name in plan.tuned:
                break
            prefix.append(named.pop(name))
        if grid.mode == "dims":
            if not all(controls):
                raise ValueError("intj: cannot tune on a zero-volume grid")
            triton_grid: Any = controls
        elif grid.mode == "cpp":
            extras = dict(zip(grid.extras, controls))
            triton_grid = lambda meta: grid.fn(  # noqa: E731
                *[meta[n] for n in grid.positional], **extras
            )
        else:
            triton_grid = grid.fn
        shim.final = None
        with torch.cuda.stream(torch.cuda.ExternalStream(stream, device=device)):
            private[0].run(*prefix, grid=triton_grid, warmup=False, **named)
        _copy_back(plan.layers, private)
        assert shim.final is not None, "Triton finished without a final launch"
        values, config_options, kernel = shim.final
        tuned_values = {**values, **config_options}
        md = kernel.metadata
        expected = sum(1 for ty in kernel.src.signature.values() if ty != "constexpr")
        if expected != nparams:
            raise RuntimeError(
                f"intj: packed {nparams} kernel arguments but triton compiled {expected}; "
                "this is an intj bug"
            )
        return (
            kernel.function,
            md.warp_size * md.num_warps,
            md.shared,
            nparams,
            kernel,
            tuple(_c_scalar(n, tuned_values[n]) for n in render.dep_names),
            tuple(tuned_values[n] for n in render.computed_names),
            tuple(values[n] for n in render.meta_names) if grid.mode == "py" else None,
        )

    return callback
```

Put the new imports at the top of the module with the existing ones.

- [ ] **Step 7: Template: tuned single level**

In `intj/runtime/entry.c.jinja`:

(a) At the top: `{% set tuned = tuning is not none %}` and `{% set ndeps = tuning.deps | sum if tuned else 0 %}`. Disable auto decode: after the `auto_decode` loop add `{% if tuned %}{% set auto_decode.enabled = false %}{% endif %}` with the comment `{# ponytail: exact keys need intj_decoded's kind/bits; add them to the auto macros if a tuned launch shows up in profiles #}`.

(b) Defines: `{% if tuned %}#define INTJ_TUNED 1\n{% endif %}`. Change the bound-cache define to `{% if (fixed_device and nwords) or tuned %}#define INTJ_BOUND_CACHE 1`. Exclude tuned params from the grid_py hidden list: `{% set hidden_params = params | selectattr('call_index', 'none') | rejectattr('annotation.tuned') | list %}`.

(c) `intj_final` gains fields, and its release/collect handle `meta`:

```jinja
typedef struct {
  intj_kernel k;
{% if tuned %}
  int64_t dep[{{ 2 * ndeps }}]; /* (value, is_bool) per dependent value C reads */
{% endif %}
{% if tuned and grid_py_mode %}
  PyObject *meta; /* tuned parameter values, for grid_py */
{% endif %}
} intj_final;
```

`intj_final_release` adds `Py_XDECREF(f->meta);` under `{% if tuned and grid_py_mode %}`. `intj_final_collect` adds `if (intj_snapshot_add(s, f->meta) != 0) return -1;` under the same condition. The module/bound traverse and clear conditions that read `return_compiled` become `(return_compiled or (tuned and grid_py_mode))`.

(d) In `intj_runtime.h`'s `intj_bound_launcher`, add `#ifdef INTJ_TUNED\n  PyObject *tuned_cb;\n#endif` after `grid_hidden`.

(e) Exact keys in the generic decode (inside the per-param loop, after the descriptor `key_accumulate`, before `{% endif %}` of the non-baked branch):

```jinja
{% set exact = a.key_fields | selectattr('kind', 'equalto', 'exact') | list %}
{% if exact %}
{% set exact_kind = a.key_fields | selectattr('kind', 'equalto', 'exact_kind') | list %}
  if (INTJ_UNLIKELY({{ d }}.kind == INTJ_VALUE_TENSOR)) {
    PyErr_Format(PyExc_TypeError, "intj: autotune key '%s' must be a scalar, not a tensor",
                 intj_param_names[{{ i }}]);
    return -1;
  }
{{ key_accumulate(exact_kind[0], d ~ '.kind') }}
{{ key_accumulate(exact[0], d ~ '.bits') }}
{% endif %}
```

(f) Tuned grid: `grid_cpp` and `grid_py` may read tuned values, so in tuned mode they are evaluated after lookup. The pre-decode grid block in `intj_call` becomes:

```jinja
{% if grid_arg is not none %}
  ... unchanged ...
{% elif grid_cpp_source and not tuned %}
  ... unchanged intj_eval_grid(args, dims) ...
{% elif grid_py_mode and not tuned %}
  ... unchanged grid_py call ...
{% elif grid_cpp_source or grid_py_mode %}
  /* tuned: the grid may read tuned values, so it is evaluated after lookup */
{% else %}
  if (intj_grid_object(args[0], &gx, &gy, &gz) != 0)
    return NULL;
{% endif %}
```

(g) In `intj_call`, before the non-tuned `INTJ_RDLOCK` block from Task 1, add the tuned hit path and route the rest of the function behind `{% if not tuned %}`:

```jinja
{% if tuned %}
  INTJ_RDLOCK(&st->lock);
  intj_final *kernel = intj_cache_lookup(cache, key);
  if (INTJ_UNLIKELY(!kernel)) {
    INTJ_RWUNLOCK(&st->lock);
    return intj_tuned_fill(st, cache, bound, key, np, device_ordinal, stream,
                           gx, gy, gz, args);
  }
  const int64_t *dep = kernel->dep;
  (void)dep;
{% if grid_py_mode %}
  /* grid_py runs Python: leave the lock with a copy of the record. */
  intj_final copy = *kernel;
  Py_XINCREF(copy.meta);
{% if return_compiled %}
  Py_XINCREF(copy.k.compiled);
{% endif %}
  INTJ_RWUNLOCK(&st->lock);
  PyObject *result = intj_tuned_grid_py_launch(st, bound, &copy, stream, args, vals, np);
  Py_XDECREF(copy.meta);
{% if return_compiled %}
  Py_XDECREF(copy.k.compiled);
{% endif %}
  return result;
{% else %}
{% if grid_cpp_source %}
  /* ponytail: evaluated under the read lock, so a grid error allocates while
   * holding it; a free-threaded finalizer that re-enters this launcher on a
   * miss would deadlock.  Evaluate from a copied dep array if that is seen. */
  uint32_t dims[3] = {1, 1, 1};
  if (intj_eval_grid(args, dep, dims) != 0) {
    INTJ_RWUNLOCK(&st->lock);
    return NULL;
  }
  gx = dims[0]; gy = dims[1]; gz = dims[2];
{% endif %}
  /* falls through to the shared launch tail below */
{% endif %}
{% else %}
  ... Task 1's lookup/fill block, unchanged ...
{% endif %}
```

The shared launch tail from Task 1 (`PyObject *result = NULL; uint32_t recorded_np ...`) is reused unchanged by the tuned non-grid_py path. Rename its local `result` to `out` to avoid clashing with the grid_py branch's `result`. The grid_py branch returns before the tail.

(h) Add the two tuned helpers after `intj_fill`, all under `{% if tuned %}`:

```jinja
{% if tuned %}
/* A dependent value from the callback: an exact int or a bool. */
static int intj_dep_value(PyObject *o, int64_t *value, int64_t *is_bool) {
  if (o == Py_True || o == Py_False) {
    *value = o == Py_True;
    *is_bool = 1;
    return 0;
  }
  *is_bool = 0;
  if (PyLong_CheckExact(o) && intj_as_i64(o, value) == 0)
    return 0;
  PyErr_SetString(PyExc_TypeError, "intj: tuned value read by C must be an int64 or bool");
  return -1;
}

/* Miss: Python runs the private chain, which tunes and launches.  The record
 * is only filled here, never launched a second time. */
static PyObject *intj_tuned_fill(intj_state *st, intj_cache *cache,
                                 intj_bound_launcher *bound, const uint64_t *key,
                                 int np, int64_t device_ordinal, uint64_t stream,
                                 uint32_t gx, uint32_t gy, uint32_t gz,
                                 PyObject *const *args) {
{% if grid_cpp_source %}
  PyObject *controls = PyTuple_New({{ grid_count }});
  if (!controls)
    return NULL;
  for (int i = 0; i < {{ grid_count }}; i++)
    PyTuple_SET_ITEM(controls, i, Py_NewRef(args[i]));
{% elif grid_py_mode %}
  PyObject *controls = Py_NewRef(Py_None);
{% else %}
  PyObject *controls = Py_BuildValue("(III)", gx, gy, gz);
  if (!controls)
    return NULL;
{% endif %}
  (void)gx; (void)gy; (void)gz;
  PyObject *blob = PyBytes_FromStringAndSize((const char *)key, INTJ_KEY_BYTES);
  PyObject *nparams_obj = blob ? PyLong_FromLong(np) : NULL;
  PyObject *device_obj = nparams_obj ? PyLong_FromLongLong(device_ordinal) : NULL;
  PyObject *stream_obj = device_obj ? PyLong_FromUnsignedLongLong(stream) : NULL;
  if (!stream_obj) {
    Py_XDECREF(blob); Py_XDECREF(nparams_obj); Py_XDECREF(device_obj);
    Py_DECREF(controls);
    return NULL;
  }
  PyObject *cb_args[5 + INTJ_NPARAMS];
  cb_args[0] = blob; cb_args[1] = nparams_obj; cb_args[2] = device_obj;
  cb_args[3] = stream_obj; cb_args[4] = controls;
  for (int i = 0; i < INTJ_NPARAMS; i++)
    cb_args[5 + i] = args[{{ grid_count }} + i];
  PyObject *cb = Py_NewRef(bound->tuned_cb);
  PyObject *res = PyObject_Vectorcall(cb, cb_args, 5 + INTJ_NPARAMS, NULL);
  Py_DECREF(cb); Py_DECREF(blob); Py_DECREF(nparams_obj);
  Py_DECREF(device_obj); Py_DECREF(stream_obj); Py_DECREF(controls);
  if (!res)
    return NULL;
  unsigned long long function;
  unsigned int block_dim, shared, nparams;
  PyObject *compiled, *deps, *computed, *meta;
  if (!PyArg_ParseTuple(res, "KIIIOOOO", &function, &block_dim, &shared, &nparams,
                        &compiled, &deps, &computed, &meta)) {
    Py_DECREF(res);
    return NULL;
  }
  intj_final fresh;
  memset(&fresh, 0, sizeof(fresh));
  if (!PyTuple_CheckExact(deps) || PyTuple_GET_SIZE(deps) != {{ ndeps }}) {
    Py_DECREF(res);
    PyErr_SetString(PyExc_RuntimeError, "intj: tuned callback returned the wrong deps; this is an intj bug");
    return NULL;
  }
  for (int i = 0; i < {{ ndeps }}; i++)
    if (intj_dep_value(PyTuple_GET_ITEM(deps, i), &fresh.dep[2 * i], &fresh.dep[2 * i + 1]) != 0) {
      Py_DECREF(res);
      return NULL;
    }
  fresh.k.function = (void *)(uintptr_t)function;
  fresh.k.block_dim = block_dim;
  fresh.k.shared = shared;
  fresh.k.nparams = nparams;
{% if return_compiled %}
  fresh.k.compiled = Py_NewRef(compiled);
{% endif %}
{% if grid_py_mode %}
  fresh.meta = Py_NewRef(meta);
{% endif %}
  (void)computed;
  Py_DECREF(res);
  int taken = 0, failed = 0, mismatch = 0;
  PyObject *out = NULL;
  INTJ_WRLOCK(&st->lock);
  intj_final *stored = intj_cache_lookup(cache, key);
  if (stored) {
    mismatch = memcmp(stored->dep, fresh.dep, sizeof(fresh.dep)) != 0;
  } else {
    stored = intj_cache_put(cache, key, &fresh);
    taken = stored != NULL;
    failed = stored == NULL;
  }
{% if return_compiled %}
  if (stored && !mismatch)
    out = Py_NewRef(stored->k.compiled);
{% endif %}
  INTJ_RWUNLOCK(&st->lock);
  if (!taken)
    intj_final_release(&fresh);
  if (failed)
    return PyErr_NoMemory();
  if (mismatch) {
    PyErr_SetString(PyExc_RuntimeError,
                    "intj: two tuning runs recorded different values for one key; this is an intj bug");
    return NULL;
  }
{% if return_compiled %}
  return out;
{% else %}
  (void)out;
  Py_RETURN_NONE;
{% endif %}
}

{% if grid_py_mode %}
static PyObject *intj_tuned_grid_py_launch(intj_state *st, intj_bound_launcher *bound,
                                           const intj_final *copy, uint64_t stream,
                                           PyObject *const *args, uint64_t *vals, int np) {
  PyObject *meta = PyDict_New();
  if (!meta)
    return NULL;
{% for p in params %}
{% if p.call_index is not none %}
  if (PyDict_SetItemString(meta, intj_param_names[{{ p.index }}], args[{{ p.call_index }}]) != 0) goto fail;
{% elif p.annotation.tuned %}
  if (PyDict_SetItemString(meta, intj_param_names[{{ p.index }}],
                           PyTuple_GET_ITEM(copy->meta, {{ tuning.meta_names.index(p.name) }})) != 0) goto fail;
{% else %}
  if (PyDict_SetItemString(meta, intj_param_names[{{ p.index }}],
                           PyTuple_GET_ITEM(bound->grid_hidden, {{ hidden_params.index(p) }})) != 0) goto fail;
{% endif %}
{% endfor %}
  {
    PyObject *grid_args[1] = {meta};
    PyObject *g = PyObject_Vectorcall(bound->grid_py, grid_args, 1, NULL);
    Py_DECREF(meta);
    meta = NULL;
    if (!g)
      return NULL;
    uint32_t gx = 1, gy = 1, gz = 1;
    int status = intj_grid_object(g, &gx, &gy, &gz);
    Py_DECREF(g);
    if (status != 0)
      return NULL;
    if (INTJ_UNLIKELY((uint32_t)np != copy->k.nparams)) {
      PyErr_SetString(PyExc_RuntimeError, "intj: kernel argument count changed; this is an intj bug");
      return NULL;
    }
    if (gx && gy && gz) {
      int32_t rc = intj_launch(st, &copy->k, gx, gy, gz, stream, vals, np);
      if (rc)
        return intj_launch_error(st, rc);
    }
  }
{% if return_compiled %}
  return Py_NewRef(copy->k.compiled);
{% else %}
  Py_RETURN_NONE;
{% endif %}
fail:
  Py_XDECREF(meta);
  return NULL;
}
{% endif %}
{% endif %}
```

`args` inside `intj_tuned_grid_py_launch` is the public-argument pointer: the call site passes `args + {{ grid_count }}`, and `grid_count` is 0 for grid_py. Two more places need the tuned case:
- `intj_grid_object` must be emitted when `grid_py_mode` is set or when neither grid_cpp nor grid_arg is. Change its guard to `{% if (grid_arg is none and not grid_cpp_source) or grid_py_mode %}`.
- The tuned fill call site uses the `args` pointer at the grid controls, as `intj_call` receives it.

(i) `make_bound`: with `tuned`, the first argument is the callback. Add `{% set tuned_offset = 1 if tuned else 0 %}` and add `tuned_offset` to every offset expression that uses `grid_py_offset`: the nargs check, `args[0]`/`args[1]` for grid_py, the device ordinal index, and `arg_index`. Store it:

```jinja
{% if tuned %}
  if (!PyCallable_Check(args[0])) {
    PyErr_SetString(PyExc_TypeError, "intj: make_bound needs the tuned callback first");
    return NULL;
  }
{% endif %}
  ...
{% if tuned %}
    bound->tuned_cb = Py_NewRef(args[0]);
{% endif %}
```

The cache init block that runs for `fixed_device and nwords` must also run for `tuned`: change `{% if nwords %}` inside the `{% if fixed_device %}` block to run for `tuned` too, by hoisting it out as `{% if (fixed_device and nwords) or tuned %}`.

(j) `intj_call`'s cache argument at the bound entry becomes `{{ '&bound->cache' if (fixed_device and nwords) or tuned else 'NULL' if fixed_device else '&st->cache' }}`. `entry()` refuses the tuned case like grid_py: `{% elif grid_py_mode or tuned %}` → `"intj: this launcher requires its bound handle"`. Bound traverse/clear: `Py_VISIT(bound->tuned_cb)` / `Py_CLEAR(bound->tuned_cb)` under `{% if tuned %}`, and their cache traverse/free branches use `(fixed_device and nwords) or tuned`.

- [ ] **Step 8: Run the new tests**

```
PYTHONPATH=$PWD VIRTUAL_ENV=$V uv run --active --no-sync python -m pytest tests/test_tuned.py -q
```

Expected: all pass. If `test_grid_modes[grid_py]` fails on the meta dict, check that `hidden_params` excludes tuned params in both the template and `LauncherFactory._bind`'s `hidden` tuple.

- [ ] **Step 9: Full suite and pyright**

```
PYTHONPATH=$PWD VIRTUAL_ENV=$V uv run --active --no-sync python -m pytest tests -q
PYTHONPATH=$PWD VIRTUAL_ENV=$V uv run --active --no-sync pyright
```

Expected: all pass, and pyright reports 0 errors.

- [ ] **Step G: Benchmark gate**

```bash
/tmp/intj-bench/run_all.sh task3
```

Compare with the baseline and write `benchmarks/journals/2026-09-26_autotune-task3_0_<sha>.md` per the Benchmark Gate. Expected: no row regresses. New tuned code paths are not in the existing suite; only untuned launchers are compared. Add the journal to this task's `git add`.

- [ ] **Step 10: Commit**

```bash
git add benchmarks/journals/2026-09-26_autotune-task3_0_*.md \
  intj/annotation.py intj/launcher.py intj/grid.py intj/tuning.py \
  intj/runtime/intj_runtime.h intj/runtime/entry.c.jinja tests/test_tuned.py
git commit -m "Launch autotuned kernels natively after Triton tunes them

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Computed keys and nested levels

About 1.5 days. Heuristics that read an unkeyed var are lowered to C and keyed at their level, and levels ≥1 are child maps inside heap records.

**Files:**
- Modify: `intj/heuristic.py` (add lowering)
- Modify: `intj/runtime/intj_runtime.h` (heuristic helpers)
- Modify: `intj/launcher.py` (`_render_key`, level-0 computed fields, render with levels; drop the Task 3 refusal)
- Modify: `intj/tuning.py` (dep slots for lowered reads)
- Modify: `intj/runtime/entry.c.jinja` (records, child maps, walk, fill across levels, `key_chain`)
- Test: `tests/test_tuning.py`, `tests/test_tuned.py`

**Interfaces:**
- Consumes: `TuningPlan.computed`, `TuningRender`, `intj_tuned_fill` (Task 3); `INTJ_DEFINE_MAP` (Task 1)
- Produces:
  - `heuristic.Source(kind: str, index: int = 0, value: object = None)`, where kind is `"arg"`, `"fixed"`, `"dep"` or `"comp"`
  - `heuristic.lower(computed: Sequence[tuple[Heuristic, int]], sources: Mapping[str, Source], levels: int) -> str`, which emits `static int intj_tuned_level_<n>(intj_state *st, PyObject *const *args, const int64_t *dep, int64_t *comp)` for every level
  - `launcher._render_key(resolved, binding, computed0: int) -> (params, device_offset, nwords, computed_fields)`
  - Module debug function `key_chain(launcher, device, *public_args) -> (tuple[bytes, ...], bool)`: the per-level keys walked, and whether the final record exists. `device` is ignored by fixed-device handles.

- [ ] **Step 1: Write the failing tests**

In `tests/test_tuned.py`, delete `test_computed_keys_are_refused_until_task_4` and append:

```python
@triton.jit
def aligned_tag(x, out, N, stride, BLOCK: tl.constexpr, ALIGNED: tl.constexpr):
    tl.store(out, BLOCK * 10 + ALIGNED + (N + stride) * 0)


def test_two_level_chain():
    out = torch.zeros(1, device="cuda", dtype=torch.int32)
    x = torch.zeros(1, device="cuda", dtype=torch.int32)

    class ByBlock(Bench):
        def __call__(self, kernel_call, quantiles):
            self.out.zero_()
            kernel_call()
            block = int(self.out.item()) // 10
            self.calls.append(block)
            return [1.0 if block == 64 else 2.0] * 3

    bench = ByBlock(out, None)
    kernel = triton.autotune(
        configs=[triton.Config({"BLOCK": 32}), triton.Config({"BLOCK": 64})], key=["N"], do_bench=bench
    )(triton.heuristics({"ALIGNED": lambda a: a["stride"] % a["BLOCK"] == 0})(aligned_tag))
    launch = make_launcher(kernel)
    device, stream = controls()
    for stride, aligned in ((128, 1), (130, 0), (192, 1)):
        launch(device, stream, 1, x, out, 8, stride)
        torch.cuda.synchronize()
        assert int(out.item()) == 640 + aligned
    assert bench.calls == [32, 64], "one tuning run for N=8; strides only add level-1 keys"
    module = launch.__self__.__self__
    keys, found = module.key_chain(launch, device, x, out, 8, 256)
    assert found and len(keys) == 2


@triton.jit
def even_store(out, n, EVEN: tl.constexpr):
    tl.store(out, EVEN + n * 0)


def test_heuristics_only_computed_key_shares_records():
    kernel = triton.heuristics({"EVEN": lambda a: a["n"] % 2 == 0})(even_store)
    launch = make_launcher(kernel, return_compiled=True)
    out = torch.zeros(1, device="cuda", dtype=torch.int32)
    device, stream = controls()
    a = launch(device, stream, 1, out, 2)
    b = launch(device, stream, 1, out, 4)
    c = launch(device, stream, 1, out, 3)
    assert a is b and a is not c
    torch.cuda.synchronize()
    assert int(out.item()) == 0


@triton.jit
def maybe_strided(out, b, B_UNIT: tl.constexpr):
    tl.store(out, B_UNIT)


def test_computed_key_short_circuits_before_tensor_reads():
    kernel = triton.heuristics(
        {"B_UNIT": lambda a: a["b"] is not None and a["b"].stride(0) == 1}
    )(maybe_strided)
    launch = make_launcher(kernel)
    out = torch.zeros(1, device="cuda", dtype=torch.int32)
    device, stream = controls()
    launch(device, stream, 1, out, None)
    torch.cuda.synchronize()
    assert int(out.item()) == 0
    launch(device, stream, 1, out, torch.zeros(4, device="cuda"))
    torch.cuda.synchronize()
    assert int(out.item()) == 1
    launch(device, stream, 1, out, torch.zeros(4, 2, device="cuda").t()[0])
    torch.cuda.synchronize()
    assert int(out.item()) == 0


@triton.jit
def kinds(out, n, K: tl.constexpr):
    tl.store(out, K + n * 0)


def test_computed_key_keeps_bool_and_int_apart():
    # n = 3 and n = 7 share a spec key; K is 1 for one and True for the other.
    # A key holding only the value would launch the int-specialized binary for
    # the bool (Triton compiles them apart).
    kernel = triton.heuristics({"K": lambda a: a["n"] > 0 if a["n"] > 5 else 1})(kinds)
    launch = make_launcher(kernel, return_compiled=True)
    out = torch.zeros(1, device="cuda", dtype=torch.int32)
    device, stream = controls()
    as_int = launch(device, stream, 1, out, 3)
    as_bool = launch(device, stream, 1, out, 7)
    assert as_int is not as_bool
```

In `tests/test_tuning.py`, append the lowering refusals:

```python
from intj.heuristic import Source, lower


@pytest.mark.parametrize(
    "fn,match",
    [
        (lambda a: a["N"] / 2, "operator"),
        (lambda a: a["N"] < a["M"] < 4, "chained"),
        (lambda a: len(a["x"]), "call"),
        (lambda a: a["x"].stride(a["N"]), "literal"),
        (lambda a: a["N"] ** 2, "operator"),
    ],
)
def test_lowering_refusals(fn, match):
    h = parse_heuristic("K", fn)
    sources = {name: Source("arg", i) for i, name in enumerate(h.inputs)}
    with pytest.raises(HeuristicError, match=match):
        lower([(h, 0)], sources, 1)
```

- [ ] **Step 2: Run them and confirm they fail**

```
PYTHONPATH=$PWD VIRTUAL_ENV=$V uv run --active --no-sync python -m pytest tests/test_tuned.py tests/test_tuning.py -q
```

Expected: the new `test_tuned.py` tests fail with `UnsupportedKernel: ... computed heuristic keys`, and `test_lowering_refusals` fails with `ImportError: cannot import name 'Source'`.

- [ ] **Step 3: C helpers in `intj_runtime.h`**

Next to the grid helpers:

```c
/* Heuristic inputs are Python values of either kind; C carries (value, is_bool)
 * so that a result keys True apart from 1, as Triton's specialization does. */
static INTJ_ALWAYS_INLINE int intj_heur_arg(PyObject *o, const char *name,
                                            int64_t *value, int64_t *is_bool) {
  if (o == Py_True || o == Py_False) {
    *value = o == Py_True;
    *is_bool = 1;
    return 0;
  }
  *is_bool = 0;
  if (INTJ_LIKELY(PyLong_CheckExact(o))) {
    if (INTJ_UNLIKELY(intj_as_i64(o, value) != 0))
      return intj_grid_overflow();
    return 0;
  }
  PyErr_Format(PyExc_TypeError, "intj: heuristic input '%s' must be an int or bool, got %s",
               name, Py_TYPE(o)->tp_name);
  return -1;
}

static INTJ_ALWAYS_INLINE int intj_grid_mod(int64_t a, int64_t b, int64_t *out) {
  int64_t q;
  if (intj_grid_floor(a, b, &q) != 0)
    return -1;
  /* a - q*b cannot overflow once the floor quotient fits. */
  *out = (int64_t)((__int128)a - (__int128)q * b);
  return 0;
}

/* triton.next_power_of_2 on Python ints, bit for bit, within int64. */
static INTJ_ALWAYS_INLINE int intj_grid_next_pow2(int64_t n, int64_t *out) {
  int64_t x = n - 1; /* n == INT64_MIN wraps in Python too only past int64 */
  if (INTJ_UNLIKELY(n == INT64_MIN))
    return intj_grid_overflow();
  x |= x >> 1; x |= x >> 2; x |= x >> 4; x |= x >> 8; x |= x >> 16; x |= x >> 32;
  return intj_grid_add(x, 1, out);
}

/* Tensor attributes a heuristic may read, through the interpreter: rare, and
 * off the hit path's shape readers on purpose (ponytail: numel/size/stride
 * come from TensorImpl directly if a tuned launch profiles hot here). */
static int intj_heur_tensor(PyObject *o, PyTypeObject *tensor, PyTypeObject *param,
                            const char *name, const char *method, int has_index,
                            long index, int64_t *out) {
  if (Py_TYPE(o) != tensor && Py_TYPE(o) != param) {
    PyErr_Format(PyExc_TypeError, "intj: heuristic reads '%s.%s' but '%s' is a %s",
                 name, method, name, Py_TYPE(o)->tp_name);
    return -1;
  }
  PyObject *r = has_index ? PyObject_CallMethod(o, method, "l", index)
                          : PyObject_CallMethod(o, method, NULL);
  if (!r)
    return -1;
  int rc = 0;
  if (r == Py_True || r == Py_False)
    *out = r == Py_True;
  else if (!PyLong_CheckExact(r) || intj_as_i64(r, out) != 0)
    rc = intj_grid_overflow();
  Py_DECREF(r);
  return rc;
}

/* dtype objects are torch singletons, so their addresses compare as identity. */
static int intj_heur_dtype(PyObject *o, PyTypeObject *tensor, PyTypeObject *param,
                           const char *name, int64_t *out) {
  if (Py_TYPE(o) != tensor && Py_TYPE(o) != param) {
    PyErr_Format(PyExc_TypeError, "intj: heuristic reads '%s.dtype' but '%s' is a %s",
                 name, name, Py_TYPE(o)->tp_name);
    return -1;
  }
  PyObject *d = PyObject_GetAttrString(o, "dtype");
  if (!d)
    return -1;
  *out = (int64_t)(uintptr_t)d;
  Py_DECREF(d);
  return 0;
}
```

- [ ] **Step 4: Lowering in `intj/heuristic.py`**

Move the new imports (`builtins`, `json`, `Mapping`/`Sequence`, `triton`) to the top of the module, then append:

```python
import builtins
import json
from collections.abc import Mapping, Sequence

import triton

_BUILTIN = {"min": builtins.min, "max": builtins.max}
_TRITON = {"cdiv": triton.cdiv, "next_power_of_2": triton.next_power_of_2}
#: tensor method -> takes one literal index
_TENSOR_METHODS = {
    "numel": False,
    "size": True,
    "stride": True,
    "dim": False,
    "element_size": False,
    "is_contiguous": False,
}
_ARITH = {ast.Add: "add", ast.Sub: "sub", ast.Mult: "mul", ast.FloorDiv: "floor", ast.Mod: "mod"}
_COMPARE = {ast.Eq: "==", ast.NotEq: "!=", ast.Lt: "<", ast.LtE: "<=", ast.Gt: ">", ast.GtE: ">="}


@dataclasses.dataclass(frozen=True)
class Source:
    """Where a heuristic input comes from in C."""

    kind: str  # "arg" | "fixed" | "dep" | "comp"
    index: int = 0  # args index, dep slot or comp slot
    value: object = None  # fixed: the baked value


def lower(computed: Sequence[tuple[Heuristic, int]], sources: Mapping[str, Source], levels: int) -> str:
    """One `intj_tuned_level_<n>` per level, writing (value, is_bool) pairs to `comp`.

    `computed` is level-major; its position is the comp slot.  Every read of a
    tensor attribute sits behind the short-circuit it has in Python.
    """
    out: list[str] = []
    for level in range(levels):
        lines = [
            f"static int intj_tuned_level_{level}(intj_state *st, PyObject *const *args, "
            "const int64_t *dep, int64_t *comp) {",
            "  (void)st; (void)args; (void)dep; (void)comp;",
        ]
        serial = [0]  # temp names: deterministic, since the source is in the module digest
        for slot, (h, at) in enumerate(computed):
            if at != level:
                continue
            emitter = _Emitter(h, sources, lines, serial)
            value, is_bool = emitter.emit(h.expr, "  ")
            lines.append(f"  comp[{2 * slot}] = {value}; comp[{2 * slot + 1}] = {is_bool};")
        lines += ["  return 0;", "}"]
        out.append("\n".join(lines))
    return "\n\n".join(out)


class _Emitter:
    def __init__(
        self, h: Heuristic, sources: Mapping[str, Source], lines: list[str], serial: list[int]
    ):
        self.h, self.sources, self.lines, self.serial = h, sources, lines, serial
        namespace = h.fn.__globals__
        self.builtins_ok = {
            name: namespace.get(name, getattr(builtins, name)) is fn for name, fn in _BUILTIN.items()
        }
        self.triton_ok = namespace.get("triton") is triton

    def fail(self, node: ast.AST, message: str) -> HeuristicError:
        return HeuristicError(f"{self.h.where}: heuristic {self.h.name!r}: {message}")

    def temp(self) -> str:
        self.serial[0] += 1
        return f"hv_{self.serial[0]}"

    def pair(self, indent: str) -> tuple[str, str]:
        v, b = self.temp(), self.temp()
        self.lines.append(f"{indent}int64_t {v} = 0, {b} = 0;")
        return v, b

    def input(self, node: ast.expr) -> tuple[str, Source] | None:
        key = _subscript_key(node, self.h.arg, self.h.where)
        return None if key is None else (key, self.sources[key])

    def tensor_arg(self, node: ast.expr, what: str) -> tuple[str, int]:
        found = self.input(node)
        if found is None or found[1].kind != "arg":
            raise self.fail(node, f"{what} needs a caller tensor argument")
        return found[0], found[1].index

    def emit(self, node: ast.expr, indent: str) -> tuple[str, str]:
        found = self.input(node)
        if found is not None:
            name, source = found
            if source.kind == "arg":
                v, b = self.pair(indent)
                self.lines.append(
                    f"{indent}if (intj_heur_arg(args[{source.index}], {json.dumps(name)}, &{v}, &{b}) != 0) return -1;"
                )
                return v, b
            if source.kind == "fixed":
                if type(source.value) not in (int, bool):
                    raise self.fail(node, f"baked {name!r} must be an int or bool")
                return str(int(cast(int, source.value))), "1" if type(source.value) is bool else "0"
            array = "dep" if source.kind == "dep" else "comp"
            return f"{array}[{2 * source.index}]", f"{array}[{2 * source.index + 1}]"
        if isinstance(node, ast.Constant) and type(node.value) in (int, bool):
            value = int(cast(int, node.value))
            if not -(1 << 63) < value < (1 << 63):
                raise self.fail(node, "integer literal is outside int64")
            return str(value), "1" if type(node.value) is bool else "0"
        if isinstance(node, ast.BinOp):
            op = _ARITH.get(type(node.op))
            if op is None:
                raise self.fail(node, "unsupported operator")
            left, _ = self.emit(node.left, indent)
            right, _ = self.emit(node.right, indent)
            v, b = self.pair(indent)
            self.lines.append(f"{indent}if (intj_grid_{op}({left}, {right}, &{v}) != 0) return -1;")
            return v, b
        if isinstance(node, ast.UnaryOp):
            operand, operand_bool = self.emit(node.operand, indent)
            v, b = self.pair(indent)
            if isinstance(node.op, ast.Not):
                self.lines.append(f"{indent}{v} = !{operand}; {b} = 1;")
            elif isinstance(node.op, ast.USub):
                self.lines.append(f"{indent}if (intj_grid_sub(0, {operand}, &{v}) != 0) return -1;")
            elif isinstance(node.op, ast.UAdd):
                self.lines.append(f"{indent}{v} = {operand};")
            else:
                raise self.fail(node, "unsupported operator")
            del operand_bool
            return v, b
        if isinstance(node, ast.Compare):
            if len(node.ops) != 1:
                raise self.fail(node, "chained comparisons are not supported")
            op, right_node = node.ops[0], node.comparators[0]
            v, b = self.pair(indent)
            if isinstance(op, (ast.Is, ast.IsNot)):
                found = self.input(node.left)
                if found is None or not (isinstance(right_node, ast.Constant) and right_node.value is None):
                    raise self.fail(node, "`is` only compares an input with None")
                name, source = found
                negate = "!" if isinstance(op, ast.IsNot) else ""
                test = (
                    f"args[{source.index}] == Py_None"
                    if source.kind == "arg"
                    else ("1" if source.kind == "fixed" and source.value is None else "0")
                )
                del name
                self.lines.append(f"{indent}{v} = {negate}({test}); {b} = 1;")
                return v, b
            symbol = _COMPARE.get(type(op))
            if symbol is None:
                raise self.fail(node, "unsupported comparison")
            if _is_dtype(node.left) or _is_dtype(right_node):
                if symbol not in ("==", "!=") or not (_is_dtype(node.left) and _is_dtype(right_node)):
                    raise self.fail(node, ".dtype only compares with another .dtype by == or !=")
                left, right = self.dtype(node.left, indent), self.dtype(right_node, indent)
            else:
                left, _ = self.emit(node.left, indent)
                right, _ = self.emit(right_node, indent)
            self.lines.append(f"{indent}{v} = {left} {symbol} {right}; {b} = 1;")
            return v, b
        if isinstance(node, ast.BoolOp):
            v, b = self.pair(indent)
            first, first_bool = self.emit(node.values[0], indent)
            self.lines.append(f"{indent}{v} = {first}; {b} = {first_bool};")
            test = f"{v}" if isinstance(node.op, ast.And) else f"!{v}"
            depth = indent
            for value in node.values[1:]:
                self.lines.append(f"{depth}if ({test}) {{")
                inner = depth + "  "
                nv, nb = self.emit(value, inner)
                self.lines.append(f"{inner}{v} = {nv}; {b} = {nb};")
                depth = inner
            for _ in node.values[1:]:
                depth = depth[:-2]
                self.lines.append(f"{depth}}}")
            return v, b
        if isinstance(node, ast.IfExp):
            v, b = self.pair(indent)
            test, _ = self.emit(node.test, indent)
            self.lines.append(f"{indent}if ({test}) {{")
            yes, yes_bool = self.emit(node.body, indent + "  ")
            self.lines.append(f"{indent}  {v} = {yes}; {b} = {yes_bool};")
            self.lines.append(f"{indent}}} else {{")
            no, no_bool = self.emit(node.orelse, indent + "  ")
            self.lines.append(f"{indent}  {v} = {no}; {b} = {no_bool};")
            self.lines.append(f"{indent}}}")
            return v, b
        if isinstance(node, ast.Subscript) and _is_attr(node.value, "shape"):
            index = node.slice
            if isinstance(index, _INDEX):
                index = index.value  # pyright: ignore[reportAttributeAccessIssue]  # 3.8 ast.Index
            return self.tensor_call(cast(ast.Attribute, node.value).value, "size", [index], node, indent)
        if isinstance(node, ast.Call) and not node.keywords:
            func = node.func
            if isinstance(func, ast.Name) and func.id in _BUILTIN and len(node.args) == 2:
                if not self.builtins_ok[func.id]:
                    raise self.fail(node, f"{func.id} must resolve to the builtin")
                a, a_bool = self.emit(node.args[0], indent)
                c, c_bool = self.emit(node.args[1], indent)
                v, b = self.pair(indent)
                op = "<" if func.id == "min" else ">"
                # Python returns the first argument unless the second beats it.
                self.lines.append(
                    f"{indent}if ({c} {op} {a}) {{ {v} = {c}; {b} = {c_bool}; }} else {{ {v} = {a}; {b} = {a_bool}; }}"
                )
                return v, b
            if (
                isinstance(func, ast.Attribute)
                and isinstance(func.value, ast.Name)
                and func.value.id == "triton"
                and func.attr in _TRITON
            ):
                if not self.triton_ok:
                    raise self.fail(node, "triton must resolve to the triton module")
                arity = 2 if func.attr == "cdiv" else 1
                if len(node.args) != arity:
                    raise self.fail(node, f"triton.{func.attr} takes {arity} argument(s)")
                operands = [self.emit(arg, indent)[0] for arg in node.args]
                v, b = self.pair(indent)
                helper = "intj_grid_cdiv" if func.attr == "cdiv" else "intj_grid_next_pow2"
                self.lines.append(f"{indent}if ({helper}({', '.join(operands)}, &{v}) != 0) return -1;")
                return v, b
            if isinstance(func, ast.Attribute) and func.attr in _TENSOR_METHODS:
                return self.tensor_call(func.value, func.attr, node.args, node, indent)
        raise self.fail(node, f"unsupported {type(node).__name__} call" if isinstance(node, ast.Call) else f"unsupported {type(node).__name__}")

    def tensor_call(
        self, target: ast.expr, method: str, args: Sequence[ast.expr], node: ast.AST, indent: str
    ) -> tuple[str, str]:
        name, index = self.tensor_arg(target, f".{method}")
        takes_index = _TENSOR_METHODS[method]
        if len(args) != int(takes_index):
            raise self.fail(node, f".{method} takes {int(takes_index)} argument(s)")
        literal = 0
        if takes_index:
            arg = args[0]
            if not (isinstance(arg, ast.Constant) and type(arg.value) is int):
                raise self.fail(node, f".{method} needs a literal int index")
            literal = int(arg.value)
        v, b = self.pair(indent)
        self.lines.append(
            f"{indent}if (intj_heur_tensor(args[{index}], st->tensor_type, st->param_type, "
            f"{json.dumps(name)}, {json.dumps(method)}, {int(takes_index)}, {literal}L, &{v}) != 0) return -1;"
        )
        if method == "is_contiguous":
            self.lines.append(f"{indent}{b} = 1;")
        return v, b

    def dtype(self, node: ast.expr, indent: str) -> str:
        name, index = self.tensor_arg(cast(ast.Attribute, node).value, ".dtype")
        v = self.temp()
        self.lines.append(f"{indent}int64_t {v};")
        self.lines.append(
            f"{indent}if (intj_heur_dtype(args[{index}], st->tensor_type, st->param_type, {json.dumps(name)}, &{v}) != 0) return -1;"
        )
        return v


def _is_attr(node: ast.expr, attr: str) -> bool:
    return isinstance(node, ast.Attribute) and node.attr == attr


def _is_dtype(node: ast.expr) -> bool:
    return _is_attr(node, "dtype")
```

The lowering refusal tests expect these messages:
- `"unsupported operator"` for `/` and `**`
- `"chained comparisons"`
- `"unsupported Call call"` for `len(...)`: `match="call"` matches it
- `"literal int index"`

- [ ] **Step 5: Render levels in `launcher.py`**

(a) `_render_key`, with `_render_params` as its wrapper:

```python
def _render_key(
    resolved: tuple[ResolvedParam, ...], device_binding: DeviceBinding, computed0: int = 0
) -> tuple[tuple[Param, ...], int | None, int, tuple[tuple[int, int], ...]]:
    """Place key fields, level-0 computed keys included, and number the call."""
    fields = tuple((p.index, field) for p in resolved for field in _key_fields(p.annotation))
    fields += tuple(
        (-1 - j, KeyField(kind, width))
        for j in range(computed0)
        for kind, width in (("payload", 8), ("descriptor", 1))
    )
    layout = _layout_fields(fields, device_binding)
    params: list[Param] = []  # the body of today's _render_params, unchanged:
    call_index = 0
    for p in resolved:
        annotation = dataclasses.replace(
            p.annotation,
            key_fields=tuple(
                sorted(
                    (field for index, field in layout.fields if index == p.index),
                    key=lambda field: field.offset,
                )
            ),
        )
        public = (
            not annotation.baked_value
            and annotation.bind_value is None
            and not annotation.tuned
        )
        params.append(Param(p.name, p.index, call_index if public else None, annotation))
        if public:
            call_index += 1
    placed = dict(((index, field.kind), field.offset) for index, field in layout.fields if index < 0)
    computed_fields = tuple(
        (placed[(-1 - j, "payload")], placed[(-1 - j, "descriptor")]) for j in range(computed0)
    )
    return tuple(params), layout.device_offset, layout.nwords, computed_fields


def _render_params(
    resolved: tuple[ResolvedParam, ...], device_binding: DeviceBinding
) -> tuple[tuple[Param, ...], int | None, int]:
    params, device_offset, nwords, _ = _render_key(resolved, device_binding)
    return params, device_offset, nwords
```

Import `KeyField` from `.annotation`. `_materialize_module` calls `_render_key(resolved, device_binding, tuning.computed[0] if tuning else 0)`. When `tuning` is set, it replaces `tuning.computed_fields` with the placed offsets: `tuning = dataclasses.replace(tuning, computed_fields=computed_fields)`. The render is only final once placement is known, which is why placement happens here.

(b) In `_plan_tuning`, delete the `if plan.computed:` refusal.

(c) In `make_launcher`, when `tuned` is set, build the full render after grid compilation. This replaces Task 3's `TuningRender(levels=1, ...)`:

```python
        from .heuristic import Source, lower

        plan = tuned.plan
        # call indices of the public parameters, and the grid's tuned reads
        by_name = {p.name: p for p in _render_params(resolved, DeviceBinding.NOT_FIXED)[0]}
        grid_code = getattr(grid_cpp, "__code__", None)
        grid_dep_names = [
            n for n in (grid_code.co_varnames[: grid_code.co_argcount] if grid_code else ())
            if n in plan.tuned
        ]
        levels = plan.levels
        dependent_level = {d.name: d.level for d in plan.dependent}
        lowered_reads = [
            name for c in plan.computed for name in c.heuristic.inputs if name in dependent_level
        ]
        wanted = list(dict.fromkeys([*lowered_reads, *grid_dep_names]))
        dep_names = tuple(sorted(wanted, key=lambda n: dependent_level[n]))  # stable: level-major
        deps = tuple(sum(dependent_level[n] == level for n in dep_names) for level in range(levels))
        computed = tuple(sorted(plan.computed, key=lambda c: c.level))
        sources: dict[str, Source] = {}
        for p in resolved:
            if p.annotation.baked_value:
                sources[p.name] = Source("fixed", value=p.baked)
            elif not p.annotation.tuned:
                sources[p.name] = Source("arg", by_name[p.name].call_index or 0)
        for slot, name in enumerate(dep_names):
            sources[name] = Source("dep", slot)
        for slot, c in enumerate(computed):
            sources[c.name] = Source("comp", slot)
        try:
            source = lower([(c.heuristic, c.level) for c in computed], sources, levels)
        except HeuristicError as error:
            raise UnsupportedKernel(f"intj: {error}") from error
        render = TuningRender(
            levels=levels,
            computed=tuple(sum(c.level == level for c in computed) for level in range(levels)),
            computed_fields=(),  # placed by _materialize_module
            computed_names=tuple(c.name for c in computed),
            deps=deps,
            dep_names=dep_names,
            meta_names=tuple(p.name for p in resolved if p.annotation.tuned) if grid_py is not None else (),
            source=source,
        )
```

This block replaces Task 3's `dep_names`/`TuningRender` code and must run *before* `compile_grid`. Then call `compile_grid(..., {n: i for i, n in enumerate(dep_names)})`, so the grid and the lowered heuristics agree on dep slots.

A dependent value read by a lowered heuristic must be readable when that level's key is computed. The walk stores level-n deps in record n, and computed keys at level L read deps of record level < L by construction (Task 2). Assert it: `assert all(dependent_level[n] < c.level for c in computed for n in c.heuristic.inputs if n in dependent_level)`.

(d) `tuning.make_tuned_callback` already returns `computed` by `render.computed_names` and `deps` by `render.dep_names`, so it needs no change.

- [ ] **Step 6: Template: records, child maps, walk, multi-level fill, `key_chain`**

(a) Replace Task 1/3's record block (from `typedef struct { intj_kernel k; ...} intj_final;` through `#define INTJ_V0_RELEASE ...`) with the level-aware version. Set `{% set levels = tuning.levels if tuned else 1 %}`, `{% set dep_offset = [] %}` and `{% set comp_offset = [] %}`, filled with running sums in a small loop:

```jinja
{% set ns = namespace(d=0, c=0) %}
{% for n in range(levels) %}
{% set _ = dep_offset.append(ns.d) %}{% set _ = comp_offset.append(ns.c) %}
{% set ns.d = ns.d + (tuning.deps[n] if tuned else 0) %}{% set ns.c = ns.c + (tuning.computed[n] if tuned else 0) %}
{% endfor %}
{% set ncomp = ns.c %}
typedef struct {
  intj_kernel k;
{% if tuned %}
  int64_t dep[{{ 2 * tuning.deps[levels - 1] }}];
{% endif %}
{% if tuned and grid_py_mode %}
  PyObject *meta;
{% endif %}
} intj_final;

... intj_final_release / intj_final_collect as in Task 3 ...

{% for n in range(levels - 1, 0, -1) %}
{% set vn = 'intj_final' if n == levels - 1 else 'intj_rec_' ~ n ~ ' *' %}
INTJ_DEFINE_MAP(intj_map_{{ n }}, {{ 2 * tuning.computed[n] }}, {{ vn }}, 4)
/* Level {{ n - 1 }}: never moves once published, so a walk reads it unlocked. */
typedef struct {
  int64_t dep[{{ 2 * tuning.deps[n - 1] }}];
  intj_map_{{ n }} child;
} intj_rec_{{ n - 1 }};
static void intj_rec_{{ n - 1 }}_release(intj_rec_{{ n - 1 }} **p) {
  intj_rec_{{ n - 1 }} *r = *p;
  intj_map_{{ n }}_free(&r->child, {{ 'intj_final_release' if n == levels - 1 else 'intj_rec_' ~ n ~ '_release' }});
  PyMem_RawFree(r);
}
static int intj_rec_{{ n - 1 }}_collect(intj_rec_{{ n - 1 }} **p, void *ctx) {
  return intj_map_{{ n }}_each(&(*p)->child, {{ 'intj_final_collect' if n == levels - 1 else 'intj_rec_' ~ n ~ '_collect' }}, ctx);
}
{% endfor %}
{% set v0 = 'intj_final' if levels == 1 else 'intj_rec_0 *' %}
INTJ_DEFINE_CACHE({{ v0 }})
#define INTJ_V0_RELEASE {{ 'intj_final_release' if levels == 1 else 'intj_rec_0_release' }}
#define INTJ_V0_COLLECT {{ 'intj_final_collect' if levels == 1 else 'intj_rec_0_collect' }}
```

The collect functions must follow `intj_final_collect`, so move `intj_final_collect` up next to `intj_final_release`, and delete the later `#define INTJ_V0_COLLECT intj_final_collect` from Task 1.

(b) Emit the lowered source after `intj_state` is defined (next to `{{ grid_cpp_source }}`): `{% if tuned %}{{ tuning.source }}{% endif %}`.

(c) `intj_pack` gains `int64_t *comp` in tuned mode. Its signature becomes `(..., intj_bound_launcher *bound{{ ', int64_t *comp' if tuned }})`. Before the key-word stores, level-0 computed keys are evaluated and placed:

```jinja
{% if tuned and tuning.computed[0] %}
  if (intj_tuned_level_0(st, args, NULL, comp) != 0)
    return -1;
{% for value_offset, kind_offset in tuning.computed_fields %}
  key_word_{{ value_offset // 8 }} |= (uint64_t)comp[{{ 2 * loop.index0 }}];
  key_word_{{ kind_offset // 8 }} |= (uint64_t)(uint8_t)comp[{{ 2 * loop.index0 + 1 }}]{% if kind_offset % 8 %} << {{ (kind_offset % 8) * 8 }}{% endif %};
{% endfor %}
{% endif %}
```

Callers: `intj_call` declares `int64_t comp[{{ [2 * ncomp, 1] | max }}];` and passes it. `spec_key` does the same.

(d) The walk, as a function after the lowered source:

```jinja
{% if tuned %}
/* The final record with the read lock held, or NULL with it released:
 * *missed is then the first level without a record, or -1 if a heuristic
 * raised.  Non-final records never move, so each is read after its lookup's
 * unlock, and every heuristic runs unlocked. */
static INTJ_ALWAYS_INLINE intj_final *intj_walk(intj_state *st, intj_cache *cache,
                                                PyObject *const *kargs, const uint64_t *key,
                                                int64_t *comp, int64_t *dep, int *missed) {
  (void)kargs; (void)comp; (void)dep;
{% for n in range(levels) %}
{% if n == 0 %}
  INTJ_RDLOCK(&st->lock);
  {{ v0 }} *s0 = intj_cache_lookup(cache, key);
{% else %}
  if (intj_tuned_level_{{ n }}(st, kargs, dep, comp) != 0) {
    *missed = -1;
    return NULL;
  }
  INTJ_RDLOCK(&st->lock);
  {{ 'intj_final' if n == levels - 1 else 'intj_rec_' ~ n ~ ' *' }} *s{{ n }} =
      intj_map_{{ n }}_lookup(&rec_{{ n - 1 }}->child, (const uint64_t *)(comp + {{ 2 * comp_offset[n] }}));
{% endif %}
  if (INTJ_UNLIKELY(!s{{ n }})) {
    INTJ_RWUNLOCK(&st->lock);
    *missed = {{ n }};
    return NULL;
  }
{% if n < levels - 1 %}
  intj_rec_{{ n }} *rec_{{ n }} = *s{{ n }};
  INTJ_RWUNLOCK(&st->lock);
  memcpy(dep + {{ 2 * dep_offset[n] }}, rec_{{ n }}->dep, sizeof(rec_{{ n }}->dep));
{% else %}
  memcpy(dep + {{ 2 * dep_offset[n] }}, s{{ n }}->dep, sizeof(s{{ n }}->dep));
  return s{{ n }};
{% endif %}
{% endfor %}
}
{% endif %}
```

A level-n child key is `comp[2*comp_offset[n] ...]` as `uint64_t` words: the (value, is_bool) pairs, reinterpreted. `int64_t` and `uint64_t` have the same size and alignment.

(e) Task 3's tuned hit path changes to use it:

```jinja
{% if tuned %}
  int64_t dep[{{ [2 * ndeps, 1] | max }}];
  int missed = 0;
  intj_final *kernel = intj_walk(st, cache, args + {{ grid_count }}, key, comp, dep, &missed);
  if (INTJ_UNLIKELY(!kernel)) {
    if (missed < 0)
      return NULL;
    return intj_tuned_fill(st, cache, bound, key, np, device_ordinal, stream,
                           gx, gy, gz, args, comp, dep);
  }
  (void)dep;
```

The rest is unchanged, but the grid code must read the local `dep` array (all levels), not `kernel->dep` (final level only). Delete Task 3's `const int64_t *dep = kernel->dep;` line.

(f) `intj_tuned_fill` gains `int64_t *comp, int64_t *dep` and handles every level:
1. After parsing `deps`, write all of them into `dep` (the global slots), instead of into `fresh.dep`.
2. Compute every level ≥1 key, `for n in 1..levels-1: intj_tuned_level_{{ n }}(st, args + {{ grid_count }}, dep, comp)`, returning NULL on error. Level-0 comps are already in `comp` from `intj_pack`.
3. Verify `computed`. It must be a tuple of length `ncomp`. For item `i`: a Python bool must match `comp[2i] == (item == Py_True) && comp[2i+1] == 1`, and a Python int must match `comp[2i+1] == 0 && intj_as_i64 == comp[2i]`. A mismatch raises `RuntimeError("intj: heuristic '%s' lowered to %lld but Triton computed %R; this is an intj bug")`, using a `static const char *const intj_computed_names[]` array rendered from `tuning.computed_names`.
4. Fill `fresh.dep` from `dep + 2*dep_offset[levels-1]`.
5. Insert under `INTJ_WRLOCK`, level by level, with the code below. A new non-final record is `PyMem_RawCalloc`'d with its dep slice copied and its child map initialized, then put. A present record's dep slice is compared, and a difference sets `mismatch`:

```jinja
  INTJ_WRLOCK(&st->lock);
{% if levels == 1 %}
  intj_final *stored = intj_cache_lookup(cache, key);
  ... Task 3's single-level body, reading and writing `stored`/`fresh` ...
{% else %}
  {{ v0 }} *s0 = intj_cache_lookup(cache, key);
  if (!s0) {
    intj_rec_0 *r = (intj_rec_0 *)PyMem_RawCalloc(1, sizeof(intj_rec_0));
    if (r && intj_map_1_init(&r->child) == 0) {
      memcpy(r->dep, dep + {{ 2 * dep_offset[0] }}, sizeof(r->dep));
      s0 = intj_cache_put(cache, key, &r);
      if (!s0) { intj_map_1_free(&r->child, {{ 'intj_final_release' if levels == 2 else 'intj_rec_1_release' }}); PyMem_RawFree(r); }
    } else {
      PyMem_RawFree(r);
    }
    failed = !s0;
  } else {
    mismatch |= memcmp((*s0)->dep, dep + {{ 2 * dep_offset[0] }}, sizeof((*s0)->dep)) != 0;
  }
{% for n in range(1, levels) %}
  {{ 'intj_final' if n == levels - 1 else 'intj_rec_' ~ n ~ ' *' }} *s{{ n }} = NULL;
  if (!failed) {
    const uint64_t *k{{ n }} = (const uint64_t *)(comp + {{ 2 * comp_offset[n] }});
    s{{ n }} = intj_map_{{ n }}_lookup(&(*s{{ n - 1 }})->child, k{{ n }});
{% if n == levels - 1 %}
    if (!s{{ n }}) {
      s{{ n }} = intj_map_{{ n }}_put(&(*s{{ n - 1 }})->child, k{{ n }}, &fresh);
      taken = s{{ n }} != NULL;
      failed = !s{{ n }};
    } else {
      mismatch |= memcmp(s{{ n }}->dep, fresh.dep, sizeof(fresh.dep)) != 0;
    }
{% else %}
    if (!s{{ n }}) {
      intj_rec_{{ n }} *r = (intj_rec_{{ n }} *)PyMem_RawCalloc(1, sizeof(intj_rec_{{ n }}));
      if (r && intj_map_{{ n + 1 }}_init(&r->child) == 0) {
        memcpy(r->dep, dep + {{ 2 * dep_offset[n] }}, sizeof(r->dep));
        s{{ n }} = intj_map_{{ n }}_put(&(*s{{ n - 1 }})->child, k{{ n }}, &r);
        if (!s{{ n }}) { intj_map_{{ n + 1 }}_free(&r->child, {{ 'intj_final_release' if n + 1 == levels - 1 else 'intj_rec_' ~ (n + 1) ~ '_release' }}); PyMem_RawFree(r); }
      } else {
        PyMem_RawFree(r);
      }
      failed = !s{{ n }};
    } else {
      mismatch |= memcmp((*s{{ n }})->dep, dep + {{ 2 * dep_offset[n] }}, sizeof((*s{{ n }})->dep)) != 0;
    }
{% endif %}
  }
{% endfor %}
  intj_final *stored = failed ? NULL : s{{ levels - 1 }};
{% if return_compiled %}
  if (stored && !mismatch)
    out = Py_NewRef(stored->k.compiled);
{% endif %}
{% endif %}
  INTJ_RWUNLOCK(&st->lock);
```

The tail (release the untaken `fresh`, raise on `failed`/`mismatch`, return) is Task 3's.

(g) `key_chain` debug entry, for the invariant test:

```jinja
{% if tuned %}
/* Debug: the key of every level this call would walk, and whether its final
 * record exists.  key_chain(launcher, device, *public_args) -> ((bytes, ...), found) */
static PyObject *key_chain(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
  intj_state *st = (intj_state *)PyModule_GetState(self);
  int64_t device = 0;
  if (nargs != 2 + INTJ_NPARAMS || !PyCFunction_Check(args[0]) ||
      !PyLong_CheckExact(args[1]) || intj_as_i64(args[1], &device) != 0 ||
      device < 0 || device > 255) {
    PyErr_SetString(PyExc_TypeError, "intj: key_chain(launcher, device, *public_args)");
    return NULL;
  }
  intj_bound_launcher *bound = (intj_bound_launcher *)PyCFunction_GET_SELF(args[0]);
  uint64_t key[INTJ_NWORDS];
  uint64_t vals[INTJ_NSLOTS];
  int64_t comp[{{ [2 * ncomp, 1] | max }}];
  int64_t dep[{{ [2 * ndeps, 1] | max }}];
  int np = 0;
  if (intj_pack(st, args + 2, key, vals, &np, {{ '(uint32_t)bound->device_ordinal' if fixed_device else '(uint32_t)device' }}, bound, comp) != 0)
    return NULL;
  int missed = 0;
  intj_final *kernel = intj_walk(st, &bound->cache, args + 2, key, comp, dep, &missed);
  if (kernel)
    INTJ_RWUNLOCK(&st->lock);
  else if (missed < 0)
    return NULL;
  int walked = kernel ? {{ levels }} : missed + 1;
  PyObject *keys = PyTuple_New(walked);
  if (!keys)
    return NULL;
  for (int n = 0; n < walked; n++) {
    static const int offsets[] = { {% for n in range(levels) %}{{ comp_offset[n] }}, {% endfor %}0 };
    static const int counts[] = { {% for n in range(levels) %}{{ tuning.computed[n] }}, {% endfor %}0 };
    PyObject *item = n == 0 ? PyBytes_FromStringAndSize((const char *)key, INTJ_KEY_BYTES)
                            : PyBytes_FromStringAndSize((const char *)(comp + 2 * offsets[n]),
                                                        16 * counts[n]);
    if (!item) {
      Py_DECREF(keys);
      return NULL;
    }
    PyTuple_SET_ITEM(keys, n, item);
  }
  return Py_BuildValue("(NO)", keys, kernel ? Py_True : Py_False);
}
{% endif %}
```

Register `{"key_chain", (PyCFunction)(void (*)(void))key_chain, METH_FASTCALL, NULL}` in `intj_methods` under `{% if tuned %}`. The walk evaluates level-n heuristics only when level n-1 was found, so when `missed == m`, the keys of levels `0..m` are all known.

- [ ] **Step 7: Run the new tests, then the full suite and pyright**

```
PYTHONPATH=$PWD VIRTUAL_ENV=$V uv run --active --no-sync python -m pytest tests/test_tuned.py tests/test_tuning.py -q
PYTHONPATH=$PWD VIRTUAL_ENV=$V uv run --active --no-sync python -m pytest tests -q
PYTHONPATH=$PWD VIRTUAL_ENV=$V uv run --active --no-sync pyright
```

Expected: all pass, and pyright reports 0 errors. If `test_computed_key_keeps_bool_and_int_apart` fails, the kind byte is missing from a key: check `computed_fields` placement at level 0, and the `2 *` words at level ≥1.

- [ ] **Step G: Benchmark gate**

```bash
/tmp/intj-bench/run_all.sh task4
```

Compare with the baseline and write `benchmarks/journals/2026-09-26_autotune-task4_0_<sha>.md` per the Benchmark Gate. Expected: no row regresses. New tuned code paths are not in the existing suite; only untuned launchers are compared. Add the journal to this task's `git add`.

- [ ] **Step 8: Commit**

```bash
git add benchmarks/journals/2026-09-26_autotune-task4_0_*.md \
  intj/heuristic.py intj/launcher.py intj/runtime/intj_runtime.h \
  intj/runtime/entry.c.jinja tests/test_tuned.py tests/test_tuning.py
git commit -m "Key heuristics over unkeyed values at their level, in nested maps

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Invariant test, docs, bench row

About half a day.

**Files:**
- Modify: `tests/test_tuned.py`
- Modify: `docs/Usage.md` (new section; drop the refusal lines 159 and 307), `TODO.md`, `benchmarks/bench_launch.py`, `benchmarks/AGENTS.md`

**Interfaces:**
- Consumes: `key_chain` (Task 4), `launcher.triton_specialization`

- [ ] **Step 1: Write the invariant test and its mutation check**

Append to `tests/test_tuned.py`:

```python
from intj.launcher import triton_specialization


def _reference(kernel, args_by_name):
    """What Triton itself decides for these arguments: its tuning keys, the
    heuristic outputs and the final call, from a private chain whose innermost
    layer records instead of launching."""
    import copy

    from triton.runtime.autotuner import Autotuner

    layers, inner = [], kernel
    while not hasattr(inner, "params"):
        layers.append(copy.copy(inner))
        inner = inner.fn
    final = {}

    class Record:
        fn = inner

        def run(self, *args, grid, warmup, **kwargs):
            final.update(zip(inner.arg_names, args))
            final.update(kwargs)

    for outer, nxt in zip(layers, [*layers[1:], Record()]):
        outer.fn = nxt
    tuning_keys = []
    for layer in layers:
        if type(layer) is Autotuner:
            layer.cache = {}
    layers[0].run(grid=(1,), warmup=False, **args_by_name)
    for layer in layers:
        if type(layer) is Autotuner:
            tuning_keys.append(tuple(layer.cache))
    params = [final[n] for n in inner.arg_names]
    options = {k: v for k, v in final.items() if k not in inner.arg_names}
    # only what the layers decided; raw arguments like `stride` are not keyed
    heuristics = tuple(sorted((k, repr(v)) for k, v in final.items() if k not in args_by_name))
    return repr(triton_specialization(inner, params, options)), tuple(tuning_keys), heuristics


def check_tuned_invariant(make_kernel, cases):
    kernel = make_kernel()
    launch = make_launcher(kernel)
    module = launch.__self__.__self__
    device, stream = controls()
    seen = {}
    for case in cases:
        launch(device, stream, 1, *case.values())
        chain, found = module.key_chain(launch, device, *case.values())
        assert found
        truth = _reference(make_kernel(), case)
        assert seen.setdefault(chain, truth) == truth, "intj key chain collides across Triton decisions"


def _invariant_kernel():
    bench = lambda call, quantiles: [1.0, 1.0, 1.0]  # noqa: E731  first config wins, deterministically
    return triton.autotune(
        configs=[triton.Config({"BLOCK": 32}), triton.Config({"BLOCK": 64})], key=["N"], do_bench=bench
    )(triton.heuristics({"ALIGNED": lambda a: a["stride"] % a["BLOCK"] == 0})(aligned_tag))


def _invariant_cases():
    x = torch.zeros(1, device="cuda", dtype=torch.int32)
    out = torch.zeros_like(x)
    half = torch.zeros(1, device="cuda", dtype=torch.float16)
    return [
        {"x": t, "out": out, "N": n, "stride": s}
        for t in (x, half)
        for n in (1, 8, 16, 17, 2**31)
        for s in (0, 32, 33, 64)
    ]


def test_tuned_key_chain_is_never_coarser_than_triton():
    check_tuned_invariant(_invariant_kernel, _invariant_cases())


def test_tuned_invariant_catches_a_dropped_exact_key(monkeypatch):
    from intj import annotation

    original = annotation._key_fields

    def without_exact(a):
        return tuple(f for f in original(a) if f.kind not in ("exact", "exact_kind"))

    monkeypatch.setattr(annotation, "_key_fields", without_exact)
    monkeypatch.setattr("intj.launcher._key_fields", without_exact)
    with pytest.raises(AssertionError, match="collides"):
        check_tuned_invariant(_invariant_kernel, _invariant_cases())
```

The mutation keeps the invariant test able to fail: `N=16` and `N=17` share every other key field, so dropping the exact word makes their chains collide while Triton's tuning keys differ.

- [ ] **Step 2: Run them**

```
PYTHONPATH=$PWD VIRTUAL_ENV=$V uv run --active --no-sync python -m pytest tests/test_tuned.py -q -k invariant
```

Expected: both pass. The first holds the invariant, and the second shows the test catches the mutation.

- [ ] **Step 3: Document it**

In `docs/Usage.md`:
- Delete the refusal sentence at line 159 and the `@triton.autotune` bullet at line 307.
- Add a `## Autotune and heuristics` section after "Calling the launcher":

````markdown
## Autotune and heuristics

`make_launcher` accepts a `@triton.jit` function wrapped in any nesting of
`@triton.autotune` and `@triton.heuristics`:

```python
launch = make_launcher(
    triton.autotune(configs=[...], key=["N"])(
        triton.heuristics({"EVEN": lambda a: a["N"] % a["BLOCK"] == 0})(kernel)
    ),
    grid_cpp=grid,   # may read tuned values such as BLOCK
)
launch(device, stream, x, out, N)   # tuned and heuristic values are not passed
```

On a miss, Triton tunes and launches through intj: pruning, `reset_to_zero`,
`restore_value`, `do_bench` and disk caching all run as usual, and
`best_config`, `bench_time` and `configs_timings` are copied back to your
tuners. On a hit, intj launches without calling Triton or any heuristic
written in Python. `Config.pre_hook` therefore runs on misses only.

intj keys a hit on:

- the exact value of every caller argument an autotune layer keys on
  (`int`, `bool`, or `float` by its fp64 bits);
- the result of every heuristic that reads a caller argument nothing else
  keys. Those heuristics are compiled to C, which accepts ints and bools, `+ - * // %`,
  comparisons, `and/or/not`, conditional expressions, `min`, `max`,
  `triton.cdiv`, `triton.next_power_of_2`, `x is None`, and
  `.numel() .size(i) .shape[i] .stride(i) .dim() .element_size()
  .is_contiguous() .dtype` on tensor arguments.

Every other heuristic runs once per key and is stored. A heuristic whose
result depends on a value only known after a lookup is keyed in a nested map,
so one launch can take more than one lookup.

Refused with `UnsupportedKernel`: `no_gpu=True`, bound values,
`extra_annotation` or `options` naming a tuned value, configs with
`num_ctas != 1`, layers assigning a runtime (non-constexpr) parameter, a
name assigned twice, and heuristics outside the subset above. Each
`make_launcher` call owns its own tuner caches. CUDA is compile-checked
only (untested).
````

Remove the `autotune` line from `TODO.md` if present, or add a one-line entry: "Tuned launches decode arguments on the generic path; the auto-decode macros do not handle exact keys yet".

- [ ] **Step 4: Bench row**

In `benchmarks/bench_launch.py`, add a `--tuned` mode. It times a warmed tuned launcher (autotune over two `BLOCK` configs on `noop`, key `["n"]`, `grid_cpp`) against the same kernel's plain launcher with `BLOCK` baked, both `grid=(1,)`, and prints one row each. It reuses `bench()`:

```python
def bench_tuned(iters, batches):
    x = torch.zeros(1024, device="cuda")
    device, stream = torch.cuda.current_device(), torch.cuda.current_stream().cuda_stream

    def grid(n: int, BLOCK: int):
        return (triton.cdiv(n, BLOCK) * 0 + 1,)

    tuned = make_launcher(
        triton.autotune(
            configs=[triton.Config({"BLOCK": 64}), triton.Config({"BLOCK": 128})],
            key=["n"],
            do_bench=lambda call, quantiles: [call() or 1.0] * 3,
        )(noop),
        grid_cpp=grid,
    )
    plain = make_launcher(noop, extra_annotation={"BLOCK": 64}, grid_cpp=grid)
    args = (device, stream, x, x, x, 1024, 1.0)
    sync = torch.cuda.synchronize
    for name, fn in (("tuned hit", tuned), ("plain", plain)):
        print(f"{name:>10}: {bench(fn, args, iters, batches, sync):7.1f} ns/launch")
```

Wire it into the mutually exclusive `modes` group as `--tuned`. Add one line to `benchmarks/AGENTS.md`'s list: "`--tuned` compares a warmed autotuned launcher with the same kernel baked".

- [ ] **Step 5: Run the bench and record the number**

```
PYTHONPATH=$PWD taskset -c 0 $V/bin/python benchmarks/bench_launch.py --tuned --iters 20000 --batches 7
```

Expected: `tuned hit` within about 15 ns of `plain`. The exact-key word and the generic decode path are the difference. Paste the two lines into the commit message body.

- [ ] **Step G: Benchmark gate**

```bash
/tmp/intj-bench/run_all.sh task5
```

Compare with the baseline and write `benchmarks/journals/2026-09-26_autotune-task5_0_<sha>.md` per the Benchmark Gate. Expected: no row regresses. `--tuned` is new and has no baseline, so it is reported only, not gated. Add the journal to this task's `git add`.

- [ ] **Step 6: Full suite, pyright, commit**

```
PYTHONPATH=$PWD VIRTUAL_ENV=$V uv run --active --no-sync python -m pytest tests -q
PYTHONPATH=$PWD VIRTUAL_ENV=$V uv run --active --no-sync pyright
git add benchmarks/journals/2026-09-26_autotune-task5_0_*.md \
  tests/test_tuned.py docs/Usage.md TODO.md benchmarks/bench_launch.py benchmarks/AGENTS.md
git commit -m "Pin the tuned key-chain invariant and document autotune support

<paste the two --tuned bench lines here>

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

Expected: all tests pass, and pyright reports 0 errors.

---

## Deferred (marked `ponytail:` in code)

- Exact keys and computed keys disable the auto-decode fast macros (Task 3 7a). Add them to `decode_auto_*` if `--tuned` shows the gap matters.
- Tensor attributes in heuristics go through the interpreter (Task 4 Step 3).
- A `grid_cpp` error in tuned mode is raised under the read lock (Task 3 7g).

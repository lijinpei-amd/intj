# Return CompiledKernel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Return the cached Triton CompiledKernel from an opt-in intj launcher.

**Architecture:** The generated native cache record owns one Python reference
to the kernel. The existing cache-hit launch uses that record and hands the
caller a new reference. The return mode has its own rendered module identity.

**Tech Stack:** CPython 3.10+, Triton 3.7+, Jinja2 C/C++ extension, ROCm GPU,
pytest, pyright.

**Spec:** [2026-09-25-return-compiled-design.md](../specs/2026-09-25-return-compiled-design.md)

## Global Constraints

- Default return_compiled=False code and zero-grid behavior remain unchanged.
- No GPU compilation or launch occurs for no_gpu=True.
- All three native cache backends and bound no-map ownership are covered.
- The Triton migration excludes annotation parser and unrelated runtime work.

---

### Task 1: Observable return contract

**Files:** Create tests/test_return_compiled.py; modify intj/launcher.py and docs/Usage.md.

**Interfaces:** make_launcher(..., return_compiled: bool = False); true-mode
compile callback returns (function, block_dim, shared, nparams, kernel).

- [x] Write a GPU test using a supported scalar parameter that
  checks type(h), h.asm["ttir"], cache-hit identity, and zero-grid return.
  Add a no_gpu=True refusal and default-None assertion.
- [x] Run PYTHONPATH=$PWD /tmp/gb2/bin/python -m pytest
  tests/test_return_compiled.py -q and observe the expected missing-argument
  failure before implementation.
- [x] Add the keyword-only flag, early no_gpu refusal, immutable
  RenderContext field, LauncherFactory propagation, and conditional fifth
  compile-callback value. Keep the default four-tuple callback.
- [x] Rerun the focused test after the Python and native result paths are in place.

### Task 2: Native result and ownership

**Files:** Modify intj/runtime/intj_runtime.h and intj/runtime/entry.c.jinja;
extend tests/test_return_compiled.py.

**Interfaces:** In return-mode modules, intj_kernel.compiled owns one
PyObject* reference. All cache free/GC paths release or visit that reference.

- [x] Add bound no-map GPU and callback-reentry/ownership tests that
  exercise the cache record lifetime.
- [x] Parse the callback fifth item while its tuple is alive. After the
  existing recheck, set kernel->compiled = Py_NewRef(compiled) only for a new
  record. On failed insertion, decref it. On a cache hit, use the winning
  record's compiled pointer.
- [x] Move only the true-mode zero-grid return behind cache selection; skip
  GPU dispatch for zero volume. Return Py_NewRef(kernel->compiled) after
  successful nonzero dispatch.
- [x] Add record-free and cache-traverse helpers for INTJ, TSL, and ABSL.
  Visit records from module and bound traverse, release them from clear,
  and make module free idempotent. Clear the ready flag before release so
  finalizer reentry cannot use a half-cleared cache.
- [x] Run the focused tests and the full intj suite using
  PYTHONPATH=$PWD /tmp/gb2/bin/python -m pytest tests -q.

### Task 3: Validation and review

**Files:** intj/launcher.py, intj/runtime/intj_runtime.h,
intj/runtime/entry.c.jinja, tests/test_return_compiled.py, docs/Usage.md.

- [x] Run pyright, the repository GPU benchmark, and existing CUDA template
  compile checks. Inspect generated true/false C source for the expected
  conditional hot paths.
- [x] Run git diff --check and review the exact file diff. Request an
  independent code review covering ownership, GC, reentry, module identity,
  zero-grid behavior, and no_gpu refusal.
- [x] Commit only the intended intj files after all checks pass.

The Triton worktree migration is tracked by its exact exception manifest and
audit. Compile-warmup mode uses Triton's interception path; CUDA TMA kernels
that allocate scratch remain documented exceptions.

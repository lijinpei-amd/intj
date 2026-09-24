# Callable Grid Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add explicit grid dimensions, compiled annotated grid functions, and Python grid callbacks to `make_launcher`.

**Architecture:** Reuse the rendered native launcher for all three modes. Put source validation and AST lowering in `intj/grid.py`; put per-mode argument routing and callback ownership in `entry.c.jinja`. Preserve the old one-object grid call when no option is selected.

**Tech Stack:** CPython 3.10+, Python stdlib `ast`, existing Jinja2/C/C++ extension build, Triton 3.7+, pytest and pyright.

**Spec:** [2026-09-24-callable-grid-design.md](../specs/2026-09-24-callable-grid-design.md)

## Global Constraints

- Every unsupported grid form raises `UnsupportedKernel` before provisioning or on its first unavailable value.
- `RenderContext` and `ModuleKey` remain frozen, hashable JSON value types.
- No copied Python/C constants: grid mode is a render-time choice, not a runtime shared enum.
- Grid controls do not enter the GPU specialization key.
- CUDA is compile checked; GPU behavior is tested on gfx942.

---

### Task 1: Explicit grid dimensions

**Files:** `intj/launcher.py`, `intj/runtime/entry.c.jinja`, `tests/test_grid.py`, `docs/Usage.md`

**Interfaces:** `make_launcher(..., grid_arg: int | None = None)`; explicit `grid_arg=N` consumes N scalar dimension arguments after stream.

- [ ] Add a GPU test that creates a kernel writing its 3D program IDs and launches with `grid_arg=1`, `2`, and `3`; check that exactly the requested coordinates were written. Include host-only invalid-count and zero-volume checks.
- [ ] Run `/tmp/gb2/bin/python -m pytest tests/test_grid.py -q` and confirm failure because `grid_arg` is not accepted.
- [ ] Add immutable grid mode/count fields to `RenderContext`, route `ncontrols` and public-argument offsets through the template, parse each dimension with existing `intj_grid_dim`, and keep the omitted-option legacy branch.
- [ ] Run the focused test, then the existing grid spelling, zero-volume and bound-signature tests.

### Task 2: Python callback grid

**Files:** `intj/launcher.py`, `intj/runtime/entry.c.jinja`, `intj/runtime/intj_runtime.h`, `tests/test_grid.py`, `docs/Usage.md`

**Interfaces:** `make_launcher(..., grid_py: Callable[[dict[str, object]], object] | None = None)` returns a native vectorcall handle; it owns the callback and fixed metadata values.

- [ ] Test that two launchers using different `grid_py` functions can share a module without exchanging callbacks; exercise a cache hit, current scalar arguments, baked values, a bound value and a callback exception.
- [ ] Run the focused test and confirm failure because `grid_py` is not accepted.
- [ ] Add callback/fixed-values references to the bound launcher with GC traverse/clear. Construct a fresh formal-name dict from public and hidden values inside C, call `grid_py` and parse its return through the existing grid validation. Create an otherwise unbound handle for this mode.
- [ ] Run the focused test and bound-owner tests; verify the callback receives current values and no stale tensor is retained beyond its handle.

### Task 3: Annotated compiled grid

**Files:** `intj/grid.py`, `intj/launcher.py`, `intj/runtime/entry.c.jinja`, `intj/runtime/intj_runtime.h`, `tests/test_grid.py`, `docs/Usage.md`

**Interfaces:** `make_launcher(..., grid_cpp=grid_def)` accepts an annotated `def`. Positional names map to original kernel parameters; keyword-only names become extra launch arguments. `intj/grid.py` returns immutable validated C source and extra-name metadata.

- [ ] Test a grid with reused `n` and `BLOCK`, an extra `cap`, a local assignment, `min`/`triton.cdiv`, and a conditional; verify GPU output for two cap values without calling the Python function.
- [ ] Run the focused test and confirm failure because `grid_cpp` is not accepted.
- [ ] Parse `inspect.getsource` with `ast`, validate parameter annotations/names and no closure/global loads beyond recognized intrinsics, then emit left-to-right checked signed-64-bit operations into a generated native function. Validate scalar inputs and final dimensions before launch.
- [ ] Test missing annotations, unknown names, closures, unsupported AST, zero division, signed floor division, integer overflow and final dimension overflow; run the focused test to green.

### Task 4: API validation and integration

**Files:** `intj/launcher.py`, `tests/test_grid.py`, `docs/Usage.md`, `README.md`

**Interfaces:** The three explicit mode options are mutually exclusive. Omitted options retain the old grid-object call.

- [ ] Test conflicting/invalid mode options, bound and fixed-device signatures, and grid values staying outside the GPU specialization key.
- [ ] Run focused tests and confirm any missing behavior before implementation.
- [ ] Document the three call signatures, supported `grid_cpp` syntax and failure behavior. Remove the old blanket “callable grids unsupported” statement.
- [ ] Run `PYTHONPATH=$PWD /tmp/gb2/bin/python -m pytest tests -q`, `pyright`, CUDA template compile checks and `git diff --check`; review the full diff for unrelated changes.

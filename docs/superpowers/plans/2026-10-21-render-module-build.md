# Render Module Build Implementation Plan

> **For agentic workers:** Execute the checkboxes in order in this checkout, preserving the uncommitted `_build_module` rename. Review the final diff before reporting completion.

**Goal:** Move `_build_module` and its exclusive helpers from `launcher.py` into `intj/render/` without changing generated modules or launch behavior.

**Architecture:** `render/module.py` owns assembling a module's `RenderKey` and selecting its Torch access configuration. `launcher.py` keeps the shared option checks, parameter layout, per-process module cache, and compile callbacks. Import launcher dependencies locally when building so importing intj still imports neither Torch nor Triton.

**Tech Stack:** Python, Jinja2, CPython extensions, Torch, Triton, pytest, Pyright.

**Spec:** The user's move request and `AGENTS.md` rendering, import, module ownership, and specialization contracts.

## Global Constraints

- Preserve the pending rename and every unrelated edit.
- Keep `_LOADED`, `_LOAD_LOCK`, `_loaded_module`, and compile callbacks in `launcher.py`.
- Move `_build_module`, `_for_access`, `_cache_version`, `_cxx_abi`, `_pointer_types`, and `_triton_identity` together.
- Keep `_spec_key_layout` and access-mode checks shared with checking, tuning, and compile paths.
- Import Torch and Triton only through `_imports` on first use.
- Keep RenderKey values and rendered C unchanged.
- Leave changes uncommitted; this request authorizes the refactor.

### Task 1: Capture a baseline

**Files:** Temporary capture script `/tmp/intj-render-module-capture.py`; no repository test additions.

- [x] Wrap `intj.render.build.build`, collect its RenderKeys, and pickle each key with `intj.render.context.render(key)`.
- [x] Run `PYTHONPATH=$PWD INTJ_RENDER_SNAPSHOT=/tmp/intj-before-render-module.pkl /tmp/gb2/bin/python /tmp/intj-render-module-capture.py` against the full suite.

### Task 2: Move module construction

**Files:** Create `intj/render/module.py`; modify `intj/launcher.py`, `intj/render/__init__.py`, `intj/tuning.py`, `tests/test_launcher.py`, and `tests/test_runtime.py`.

**Interfaces:** `_build_module(b: _BuildOptions) -> types.ModuleType` retains its existing arguments and behavior. Launcher calls `render_module._build_module(b)`. Shared launcher helpers are imported inside that function, after module initialization.

- [x] Transfer the six functions verbatim, adjusting only relative imports and shared helper access.
- [x] Remove imports from `launcher.py` that only those functions used.
- [x] Change existing monkeypatches to `intj.render.module._build_module`; change ABI/version helper tests to their new owner.
- [x] Update module responsibility docstrings and qualify references to `render.module._build_module`.
- [x] Run formatting on the changed Python files and `PYTHON=/tmp/gb2/bin/python ninja -C ci typecheck`.

### Task 3: Verify the final tree

**Files:** Existing launcher, runtime, and tuning tests; temporary baseline pickle.

- [x] Render every captured key again and compare C with blank lines removed.
- [x] Run the full suite again with the same capture wrapper and compare RenderKey values for builds seen in both runs.
- [x] Confirm fresh import isolation, cached-build reuse, access-mode fallback, module ownership, and ordinary/tuned specialization invariants through the existing suite.
- [x] Review `git diff --check` and the final diff for accidental changes, duplicate helpers, and import cycles.
- [x] Report changed ownership, test counts, rendering equivalence, and any actual limitations.

## Validation Results

- Full suite before and after: 1,108 passed, 15 skipped, two existing Torch warnings.
- Pyright: zero errors. Changed-file pre-commit hooks and `git diff --check`: passed.
- All six moved function bodies are AST-equivalent; the build function only adds local imports.
- All 11 baseline keys render identical C. Nine captured keys match across runs; the fallback tests use temporary paths, and the final run also builds the corrected Torch-version test's module.
- Independent code review: no remaining findings after correcting the Torch-version and absent-ABI-table patches.

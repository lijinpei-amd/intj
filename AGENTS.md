# Conventions

## Shared C/C++ and python definitions

Anything both sides need (enums, key layout, structs) is defined **once in C++**:

- `intj/runtime/<name>.h` — the C++ definition.
- `intj/runtime/<name>_python.h` — the nanobind wrapper exposing it to python.

Python imports it from there. Never re-declare a constant in `launcher.py` "kept in
sync" with a header; that copy goes stale silently. Nothing crosses the boundary yet,
so there is no `_python.h` and no nanobind dependency in the tree — add both with the
first thing that actually needs sharing.

`intj/runtime/` also holds the launcher runtime itself: `intj_runtime.h` (helpers
included by every rendered module) and `entry.c.jinja` (the rendered entry point).

## Backends are data, not branches

Everything backend-specific lives in the `BACKENDS` dict in `launcher.py`, and the
template branches on its fields (`error_style`), never on a backend name. Supporting
another triton backend should be one dict entry.

## Refusals are loud and early

Anything intj cannot do raises `UnsupportedKernel`, preferably in `create_launcher`,
otherwise on the first cache miss. Never fall back to `JITFunction` silently — a
silent fallback turns a 0.3 us call into a 14 us one with no signal.

## The invariant

> same intj spec key ⟹ same triton specialization

A key finer than triton's costs a redundant compile. A key coarser than triton's
launches the wrong binary and does not crash. Every change to argument decoding or
key layout must keep `test_spec_key_is_never_coarser_than_triton` passing, and that
test must be able to fail: mutate the change away and check the test catches it.

## Testing

No NVIDIA GPU on the development machine, so the CUDA path is compile-checked only.
Mark anything untested as untested in the docs.

```
PYTHONPATH=$PWD /tmp/gb2/bin/python -m pytest tests -q     # triton 3.8.0, torch 2.14+rocm7.2, gfx942
PYTHONPATH=$PWD /tmp/gb2/bin/python benchmarks/bench_launch.py
```

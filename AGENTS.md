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

## RenderContext and ModuleKey are value types

Frozen, every field immutable (tuples, never lists), so both compare by value, hash,
and serialize to JSON (what `ModuleKey.digest()` hashes). A list field anywhere in either compares fine and
blows up on `hash()`; `test_value_types_compare_hash_and_serialize` guards that.

## Backends are data, not branches

Everything backend-specific lives in a `Backend` subclass in `launcher.py`
(`HipBackend`, `CudaBackend`), and the template branches on its fields
(`error_style`), never on a backend name. Supporting another triton backend should be
a subclass plus `register()`.

## Modules stay out of the import system

Rendered extensions are loaded by hand (`spec_from_file_location` +
`module_from_spec` + `exec_module`), never through `import_module`, so nothing lands
in `sys.modules` and intj's `_LOADED` dict is the only owner. The template must stay
multi-phase (`PyModuleDef_Init`): a single-phase `m_size = -1` module gets cached by
the interpreter per (name, path), and two launchers would then share one module's
state. `test_modules_stay_out_of_the_import_system` guards both halves.

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

## Types

`intj/` is pyright-strict (settings in `pyproject.toml`); `tests/` and `benchmarks/`
carry a `# pyright: standard` header. The `reportUnknown*` and `reportMissingTypeStubs`
rules are off project-wide because triton, torch and jinja2 ship no type information —
they say nothing about intj's own code. Values coming out of triton are annotated
`Any` (`JitFunction`, `CompiledKernel`); reaching into a private triton name is fine
where triton has no public equivalent, with a targeted
`# pyright: ignore[reportPrivateUsage]` and a reason.

    pyright   # 0 errors, keep it that way

## Testing

No NVIDIA GPU on the development machine, so the CUDA path is compile-checked only.
Mark anything untested as untested in the docs.

```
PYTHONPATH=$PWD /tmp/gb2/bin/python -m pytest tests -q     # triton 3.8.0, torch 2.14+rocm7.2, gfx942
PYTHONPATH=$PWD /tmp/gb2/bin/python benchmarks/bench_launch.py
```

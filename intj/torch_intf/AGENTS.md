# torch_intf

Read `README.md` first for what this directory is. The rules below are what keep it
correct.

## Never guess an offset

A wrong offset cannot raise: it reads whatever sits at that address and hands the
kernel a pointer built from it. So:

- Never hand-edit a number in the ABI table. Regenerate the entry with a detector
  and paste its output verbatim.
- An entry's layout and dtypes are verified together or not at all. If the running
  torch's dtypes differ from the recorded ones, there is no layout.
- No verified layout found is an error: raise, naming the torch version and how to
  add an entry. Never fall back silently, and never fill a gap with a nearby
  version's entry.
- A release reads its major.minor's table; any other build (a nightly, a source
  build, a release candidate) reads only a table recorded for its exact
  `torch.__version__` (`abi_table_name`). Its dtypes matching the release's
  proves nothing about its offsets.

## The generator checks what STATIC_COMPILE assumes

STATIC_COMPILE mode needs no table, but it declares `intj_THPVariable` itself
(`intj/runtime/intj_thpvariable.h`) instead of including `python_variable.h`, which
drags in pybind11, and does not check that at load.
[tools/cpp_detect.py](../tools/cpp_detect.py) compiles it beside torch's own and
refuses a torch where they differ. Any new layout assumption in C++ code gets
the same treatment: checked in `intj.tools.cpp_detect`, not at load.

## The ABI version files are a trust boundary

Each `torch_abi/<major>.<minor>.toml` contains tables of that major.minor
only: at most one for its releases, named `"major.minor"`, and one per
pre-release build, named by its exact version, each with its measured offsets
and dtypes together. `abi_detect` prints the name the running torch is looked up
by. The parser rejects malformed input with a `ValueError` naming the file,
never skips it. CI pins releases only (`ci/image.py` reads release tables), so a
pre-release table is checked only on a machine running that build.

## Import torch and triton on demand

Through `intj/_imports.py`, inside the function that needs them, never at module
top (see the root `AGENTS.md`): importing `intj` must not import torch, and probing
must not initialize the GPU.

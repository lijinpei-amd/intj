# python_intf

Read `README.md` first for what this directory is. The rules below are what keep it
correct.

## Every CPython internal lives here

Any read past the public API -- a struct field, an `_Py` macro, a `PyUnstable_`
call -- goes in a `cpython_*.h` header (C) or `cpython_abi.py` (python), never inline
in `intj_runtime.h`, the entry template, or `torch_intf`. One place to audit when a
new CPython changes a layout.

## Add a version by checking it, never by assuming

A version enters `cpython_abi._HEADERS` only after `python -m intj.python_intf.check
<header>` passes on it, on both the default and the free-threaded build
(`uv run --no-project --python cpython-3.Nt ...`). No entry is an error that names
the version; never fall back to a neighbour's header. When a check fails, write a new
`cpython_3NN.h` for the new layout rather than `#if`-ing the old one: it defines
`intj_long_compact` and `intj_long_digits` and includes `cpython_common.h`, which keeps
the decoders shared. Public calls a version predates get a shim in that version's
header, guarded by the version that added the call.

## No stable ABI

Never define `Py_LIMITED_API`. intj ships no prebuilt extension, and every module
it builds is compiled for, and loaded into, one interpreter only, so abi3 would buy
nothing. It would cost the launch path its inline reads: a function call per int
and float argument and per refcount, roughly 10-45% of a SHIM launch.

## Compiled against, never assumed

C reads through CPython's own struct definitions and macros, so the compiler checks
where a field is; the digit width is a `static_assert`. What a field *holds* (the
sign encoding) only `check.py` catches, so any new internal read gets a case there.
Python reads layout facts off the interpreter (`object.__basicsize__`), never
computes them from `ctypes` pointer sizes: the free-threaded build's header is not
two pointers.

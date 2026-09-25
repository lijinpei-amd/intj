# python_intf

Read [README.md](README.md) first for the binary layouts and access modes. The
rules and implementation notes below keep this interface correct.

## Scope and files

`python_intf` isolates the CPython details that intj needs to read Python values
on the launch path. It provides integer and float readers, the Python object
header size, and compatibility helpers for generated extension modules. All
tensor access modes use this layer, including `TorchAccessMode.INTERPRETER`.
For the launcher API, see [Usage](../../docs/Usage.md). Tensor layouts and dtype
codes belong to [torch_intf](../torch_intf/README.md).

Intj decodes arguments on every launch. Reading an integer's digits or a
float's stored value directly keeps those operations inline, including when
the launcher is compiled as C++.

| File | Responsibility |
| --- | --- |
| [cpython_abi.py](cpython_abi.py) | Records supported Python versions, selects the header, and reads the interpreter's object header size. |
| [cpython_abi.h](cpython_abi.h) | Implements scalar readers, compatibility shims, and the module mutex. |
| [check.py](check.py) | Compiles the header and compares its results with the running interpreter. |

## Every CPython internal lives here

Any read past the public API -- a struct field, an `_Py` macro, a `PyUnstable_`
call -- goes in `cpython_abi.h` (C) or `cpython_abi.py` (Python), never inline
in `intj_runtime.h`, the entry template, or `torch_intf`. One place to audit
when a new CPython changes a layout.

Each generated extension is compiled against the running interpreter's
`Python.h`. CPython's headers supply the struct definitions and field offsets;
intj maintains no separate CPython offset table. The compiler checks where a
field is. The checker verifies what it contains: a valid field access can
still decode an integer's sign incorrectly after a CPython change. The shared
integer decoder requires 30-bit digits, enforced by a compile-time assertion.
Python reads layout facts from the interpreter (`object.__basicsize__`), never
from `ctypes` pointer sizes; the free-threaded header is not two pointers.
The C side uses `sizeof(PyObject)` to locate `THPDtype::scalar_type` after
`PyObject_HEAD`. `pyobject_size()` supplies the detecting interpreter's header
size and lets `layout_for()` refuse an unrepresentable offset early. For
`RUNTIME_SHIM`, the generated C setter adds its compiled `sizeof(PyObject)` to
the table's post-header `cdata` offset and saves the absolute result before
any launch.

## Scalar-reader contract

Callers must check the object's type and keep it alive while reading it. The
runtime passes exact built-in `int` and `float` objects and handles `bool`
separately. These readers do not call `__int__`, `__index__`, or `__float__`,
take ownership of references, or set Python exceptions on overflow.

| Reader | Result |
| --- | --- |
| `intj_as_i64(o, out)` | Writes a signed 64-bit value and returns `0` for `-2**63 <= value < 2**63`; returns `-1` otherwise. |
| `intj_as_int(o, out)` | Writes the value's bits to a `uint64_t`. Returns `INTJ_INT_I64` for the signed range above, `INTJ_INT_U64` for `2**63 <= value < 2**64`, or `INTJ_INT_TOO_BIG` otherwise. |
| `INTJ_FLOAT_VALUE(o)` | Reads the stored C `double` from `PyFloatObject::ob_fval`. |

For a negative integer, the unsigned output of `intj_as_int` is the value
modulo `2**64`. Callers use either integer reader's output only on success and
turn failure into an appropriate Python exception. Triton specialization
policy stays in [intj_runtime.h](../runtime/intj_runtime.h): it chooses `i32`,
`i64`, or `u64`, handles specialization, and converts ordinary float arguments
to `fp32`. The CPython layer only extracts the value.

## No stable ABI

Never define `Py_LIMITED_API`. intj uses the full CPython C API, including
unstable APIs and internal fields, and ships no prebuilt extension. Every
module it builds is compiled for, and loaded into, one interpreter only, so
`abi3` would buy nothing. It would cost the launch path its inline reads: a
function call per int and float argument and per refcount, roughly 10-45% of
a `RUNTIME_SHIM` launch.

## Python compatibility and module identity

The recorded supported builds are CPython 3.8–3.14 with the GIL, plus
free-threaded CPython 3.13t and 3.14t. This is a CPython-specific interface.

1. `header_for()` looks up the interpreter's major/minor version in `_HEADERS`.
   Every supported entry currently selects `cpython_abi.h`. An unlisted
   version returns `None`; `make_launcher` raises `UnsupportedKernel` before
   building a module.
2. `RenderContext` records the full `(major, minor, micro)` version, selected
   header, and `free_threaded` build flag. The flag comes from
   `sysconfig.get_config_var("Py_GIL_DISABLED")`, independently of whether the
   GIL is currently enabled.
3. `ModuleKey` includes that context, the header contents, and the full
   `EXT_SUFFIX`. A patch release, a regular/free-threaded build change, or a
   header edit therefore changes the module digest. The suffix also preserves
   platform and other ABI tags, such as the debug-build tag.
4. The [entry template](../runtime/entry.c.jinja) rejects compilation if the
   included headers have a different major/minor/micro version or GIL build
   from the render context.

The allowlist selects which versions may build; the module key controls reuse
of a cached extension for a particular version and build ABI. Subinterpreters
are unsupported; the generated module declares this on CPython 3.12+.

## Compatibility helpers

The C header supplies older interpreters with calls the runtime uses:
vectorcall and `PyObject_CallMethodNoArgs` before 3.9, `Py_NewRef` before 3.10,
and the raised-exception get/set functions before 3.12. This keeps version
branches out of callers. Guard each shim by the version that added its call.

## Free-threaded modules

Generated modules declare `Py_MOD_GIL_NOT_USED` on Python 3.13+, so loading them
does not require CPython to enable the GIL. On a free-threaded build,
`intj_mutex` is a `PyMutex` owned by the module. `INTJ_LOCK` and `INTJ_UNLOCK`
protect kernel-cache lookup/insertion and access to the compile-callback
reference. The lock is released before calling Python or the GPU driver. A
cache miss checks again under the lock before inserting its result because
another thread may have filled the same key during compilation. On a regular
build, the lock operations compile away.

For Torch's `RUNTIME_SHIM` mode, `pyobject_size()` supplies the detecting
interpreter's header size and lets `layout_for()` refuse an unrepresentable
offset early. The generated C setter adds its compiled `sizeof(PyObject)` to
the table's post-header `cdata` offset and saves the absolute result before
any launch, so free-threaded builds use a verified Torch layout without a
per-launch CPython size query.

## Verification

From the repository root, check the active interpreter:

```sh
python -m intj.python_intf.check
```

The command needs a C compiler (`CC`, defaulting to `cc`) and the
interpreter's headers. It needs no Torch, Triton, or GPU. On Python 3.8–3.10,
importing intj also requires `tomli`.

The checker builds a temporary shared library and tests integer digit/width
boundaries with both signs, float reads including signed zero and infinity,
`sizeof(PyObject)`, and an exception round trip. It prints `ok` on success and
exits nonzero on a mismatch or build failure. It does not check the module's
threading behavior or launch a kernel.

For the rendered module, run the Python version matrix:

```sh
bash tests/run_python_matrix.sh             # all nine supported builds
bash tests/run_python_matrix.sh 3.13 3.13t  # a selected pair
```

The script uses `uv` and may download interpreters and dependencies. It runs
[test_runtime.py](../../tests/test_runtime.py), which includes the ABI checker
and exercises argument decoding, error propagation, and concurrent launches
against a stub driver. Each environment uses the newest CPU Torch it can
install. The free-threaded runs also check that the GIL stays disabled.

This matrix covers the `INTERPRETER` tensor reader and `RUNTIME_SHIM` where a
verified Torch layout exists, including free-threaded builds. `STATIC_COMPILE`
is covered separately in [test_launcher.py](../../tests/test_launcher.py).
The matrix does not exercise Triton compilation or GPU execution; the full
launcher is not tested below Python 3.10 because `triton>=3.8` has no wheel for
those interpreters.

## Adding a Python version

1. Run `python -m intj.python_intf.check cpython_abi.h` on the candidate
   interpreter. Naming the header allows testing before it is in `_HEADERS`.
2. Check both regular and free-threaded builds when the version provides both
   (for example, `uv run --no-project --python cpython-3.Nt ...`). Inspect
   CPython's definitions; add a `PY_VERSION_HEX` branch or compatibility shim
   only where needed, keeping the shared decoders shared.
3. Add a checker case for every new internal read or changed interpretation.
   Confirm the checker passes before adding the version to `_HEADERS`.
4. Add the supported builds to `tests/run_python_matrix.sh` and run their
   runtime tests. Record the coverage and any remaining limits here.

No entry is an error that names the version; never fall back to a neighbor's
layout.

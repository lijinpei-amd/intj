# python_intf

INTJ's interface to CPython internals. The launch path reads a few CPython objects at binary level instead of going through the API:

- `PyLongObject` digits (`lv_tag`, `ob_digit`, 30-bit digits), for ints too large for the public compact reader
- `PyFloatObject::ob_fval`, so the float read stays inline in the C++ mode
- `sizeof(PyObject)`, the offset of torch's `THPDtype::scalar_type`

Unlike torch's, these need no offset table: every module is compiled against the running interpreter's `Python.h`. What varies by version is the *implementation*, so there is one header per CPython layout, selected by interpreter version:

- `cpython_abi.py`: the version -> header table (`header_for`), plus the python-side reads.
- `cpython_38.h`: the 3.8 to 3.11 int layout (`ob_size`), plus the few public calls the runtime uses that those versions predate.
- `cpython_312.h`: the 3.12 int layout (`lv_tag`). Serves 3.12 through 3.14.
- `cpython_common.h`: the int and float decoders both share, over each layout's two primitives.
- `check.py`: compiles a header against the running interpreter and compares every read with python's own answer. Needs only a C compiler.

The launcher renders the selected header into the module as `INTJ_PYTHON_ABI`, and puts the full python version in `RenderContext`, so a module is never reused across interpreters.

Verified: 3.8 through 3.14, plus the free-threaded 3.13t and 3.14t. `tests/run_python_matrix.sh` runs `check.py` and `tests/test_runtime.py` -- the rendered module against a stub driver, no triton or GPU -- on each, with the newest CPU torch it installs.

Free-threaded builds: the module declares `Py_MOD_GIL_NOT_USED` and guards its kernel cache and compile callback with a `PyMutex` (compiled out on GIL builds). The SHIM access mode is refused there, since every `torch_abi.toml` entry was measured under a 16-byte `PyObject` header.

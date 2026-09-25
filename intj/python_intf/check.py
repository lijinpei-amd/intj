"""Check a CPython layer header against the interpreter that is running.

The compiler checks where a field is, not what it holds.  This compiles the
header `cpython_abi` selects (or one named on the command line) against this
interpreter's `Python.h`, and compares every read with python's own answer.
Needs a C compiler and nothing else: no torch, no triton.

    python -m intj.python_intf.check [header]   # exits non-zero on a mismatch
"""

from __future__ import annotations

import ctypes
import os
import pathlib
import subprocess
import sys
import sysconfig
import tempfile

from .cpython_abi import header_for, pyobject_size, python_version

_HERE = pathlib.Path(__file__).parent

_SOURCE = """
#include <Python.h>
#define INTJ_ALWAYS_INLINE inline
#define INTJ_LIKELY(x) (x)
#define INTJ_UNLIKELY(x) (x)
#include INTJ_CPYTHON_STATIC_COMPILE_HEADER

int check_int(PyObject *o, uint64_t *out) { return intj_as_int(o, out); }
int check_i64(PyObject *o, int64_t *out) { return intj_as_i64(o, out); }
double check_float(PyObject *o) { return INTJ_FLOAT_VALUE(o); }
size_t check_pyobject_size(void) { return sizeof(PyObject); }
/* the error round trip intj_note_param makes, through the shims where there are any */
PyObject *check_reraise(void) {
  PyErr_SetString(PyExc_ValueError, "inner");
  PyObject *exc = PyErr_GetRaisedException();
  PyErr_SetRaisedException(exc);
  return NULL;
}
"""

#: Every digit-count and width boundary the decoders branch on, both signs.
_INTS = sorted({
    s * v
    for b in (0, 1, 29, 30, 31, 32, 60, 62, 63, 64, 89, 90, 100)
    for v in (2**b - 1, 2**b, 2**b + 1)
    for s in (1, -1)
})
_FLOATS = (0.0, -0.0, 1.5, -2.25, 1e300, float("inf"))


def _build(header: str, out: pathlib.Path) -> ctypes.PyDLL:
    src = out.with_suffix(".c")
    src.write_text(_SOURCE)
    subprocess.run(
        [
            os.environ.get("CC", "cc"), "-shared", "-fPIC", "-O2",
            f"-DINTJ_CPYTHON_STATIC_COMPILE_HEADER=\"{header}\"",
            f"-I{sysconfig.get_paths()['include']}", f"-I{_HERE}",
            str(src), "-o", str(out),
        ],
        check=True,
    )
    # PyDLL: the functions touch python objects, so the GIL stays held
    lib = ctypes.PyDLL(str(out))
    lib.check_int.argtypes = [ctypes.py_object, ctypes.POINTER(ctypes.c_uint64)]
    lib.check_i64.argtypes = [ctypes.py_object, ctypes.POINTER(ctypes.c_int64)]
    lib.check_float.argtypes = [ctypes.py_object]
    lib.check_float.restype = ctypes.c_double
    lib.check_pyobject_size.restype = ctypes.c_size_t
    lib.check_reraise.restype = ctypes.py_object
    return lib


def check(header: str | None = None) -> list[str]:
    """Every disagreement between `header` and this interpreter; empty when it holds."""
    header = header or header_for()
    if header is None:
        return [f"no header is recorded for python {sys.version.split()[0]}; name one"]
    errors: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        lib = _build(header, pathlib.Path(tmp, "check.so"))
        if lib.check_pyobject_size() != pyobject_size():
            errors.append(f"sizeof(PyObject) {lib.check_pyobject_size()} != {pyobject_size()}")
        for v in _INTS:
            u, i = ctypes.c_uint64(), ctypes.c_int64()
            kind = lib.check_int(v, ctypes.byref(u))
            want = 1 if -(2**63) <= v < 2**63 else 2 if 2**63 <= v < 2**64 else 0
            if kind != want or (want and u.value != v % 2**64):
                errors.append(f"intj_as_int({v}) = ({kind}, {u.value}), want ({want}, {v % 2**64})")
            rc = lib.check_i64(v, ctypes.byref(i))
            fits = -(2**63) <= v < 2**63
            if rc != (0 if fits else -1) or (fits and i.value != v):
                errors.append(f"intj_as_i64({v}) = ({rc}, {i.value})")
        try:
            lib.check_reraise()
            errors.append("PyErr_SetRaisedException raised nothing")
        except ValueError as e:
            if e.args != ("inner",):
                errors.append(f"PyErr_GetRaisedException round trip gave {e!r}")
        for f in _FLOATS:
            if lib.check_float(f).hex() != f.hex():
                errors.append(f"INTJ_FLOAT_VALUE({f!r}) = {lib.check_float(f)!r}")
    return errors


if __name__ == "__main__":
    header = sys.argv[1] if len(sys.argv) > 1 else None
    errors = check(header)
    build = "free-threaded" if sysconfig.get_config_var("Py_GIL_DISABLED") else "default"
    name = "%d.%d.%d" % python_version()
    for e in errors:
        print(e, file=sys.stderr)
    print(f"python {name} ({build} build): {'FAIL' if errors else 'ok'}")
    sys.exit(1 if errors else 0)

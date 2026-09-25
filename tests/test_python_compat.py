# pyright: standard
"""Compile the launcher's integer decoder against the running CPython headers."""

import importlib.util
import pathlib
import shutil
import subprocess
import sysconfig

import pytest

from intj.python_intf import cpython_abi


def test_integer_decoder_matches_python_on_this_interpreter(tmp_path):
    compiler = shutil.which("cc")
    if compiler is None:
        pytest.skip("a C compiler is needed to test the runtime header")

    source = tmp_path / "_intj_int_probe.c"
    source.write_text(r"""
#define INTJ_NWORDS 1
#define INTJ_TORCH_ACCESS_INTERPRETER 1
#include "intj_runtime.h"

static PyObject *decode(PyObject *self, PyObject *value) {
  (void)self;
  uint64_t bits = 0;
  int64_t signed_value = 0;
  int kind = intj_as_int(value, &bits);
  PyObject *bits_obj = kind == INTJ_INT_TOO_BIG ? Py_NewRef(Py_None)
      : PyLong_FromUnsignedLongLong(bits);
  PyObject *signed_obj = intj_as_i64(value, &signed_value) == 0
      ? PyLong_FromLongLong(signed_value) : Py_NewRef(Py_None);
  if (!bits_obj || !signed_obj) {
    Py_XDECREF(bits_obj);
    Py_XDECREF(signed_obj);
    return NULL;
  }
  return Py_BuildValue("iNN", kind, bits_obj, signed_obj);
}

static PyMethodDef methods[] = {
  {"decode", decode, METH_O, NULL}, {NULL, NULL, 0, NULL}
};
static struct PyModuleDef definition = {
  PyModuleDef_HEAD_INIT, "_intj_int_probe", NULL, -1, methods
};
PyMODINIT_FUNC PyInit__intj_int_probe(void) {
  return PyModule_Create(&definition);
}
""")
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    assert isinstance(suffix, str)
    built = tmp_path / f"_intj_int_probe{suffix}"
    runtime = pathlib.Path(__file__).parents[1] / "intj" / "runtime"
    python_intf = runtime.parent / "python_intf"
    header = cpython_abi.header_for()
    assert header is not None
    result = subprocess.run(
        [
            compiler,
            "-O2",
            "-shared",
            "-fPIC",
            f"-I{sysconfig.get_paths()['include']}",
            f"-I{runtime}",
            f"-I{python_intf}",
            f'-DINTJ_CPYTHON_STATIC_COMPILE_HEADER="{header}"',
            str(source),
            "-o",
            str(built),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    spec = importlib.util.spec_from_file_location("_intj_int_probe", built)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # pyright: ignore[reportAttributeAccessIssue]  # Loader stubs lack exec_module
    for value, expected in (
        (-(2**100), (0, None, None)),
        (-(2**63) - 1, (0, None, None)),
        (-(2**63), (1, 2**63, -(2**63))),
        (-(2**30), (1, 2**64 - 2**30, -(2**30))),
        (-1, (1, 2**64 - 1, -1)),
        (0, (1, 0, 0)),
        (1, (1, 1, 1)),
        (2**30 - 1, (1, 2**30 - 1, 2**30 - 1)),
        (2**30, (1, 2**30, 2**30)),
        (2**63 - 1, (1, 2**63 - 1, 2**63 - 1)),
        (2**63, (2, 2**63, None)),
        (2**64 - 1, (2, 2**64 - 1, None)),
        (2**64, (0, None, None)),
        (2**100, (0, None, None)),
    ):
        assert module.decode(value) == expected


def test_host_launcher_builds_with_installed_triton(tmp_path):
    import torch

    from intj import TorchAccessMode, make_launcher

    source = tmp_path / "compat_kernel.py"
    source.write_text("import triton\n\n@triton.jit\ndef kernel(x):\n    pass\n")
    spec = importlib.util.spec_from_file_location("compat_kernel", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # pyright: ignore[reportAttributeAccessIssue]  # Loader stubs lack exec_module

    launch = make_launcher(
        module.kernel, no_gpu=True, torch_access_mode=TorchAccessMode.INTERPRETER
    )
    assert launch(0, 0, 1, 7) is None
    assert launch(0, 0, 1, torch.arange(4)) is None
    with pytest.raises(RuntimeError, match="cannot read tensor argument 'x'") as error:
        launch(0, 0, 1, torch.zeros(4).to_sparse())
    assert error.value.__cause__ is not None

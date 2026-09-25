/* The head of torch's THPVariable, for the STATIC_COMPILE access mode.
 *
 * Deliberately NOT torch/csrc/autograd/python_variable.h.  That header declares
 * the four-line struct below and nothing else intj needs, but it also pulls in
 * pybind11 -- 79% of the resulting .text, none of it reachable, and 9.5 s of
 * the 11 s build.
 *
 * The module checks the cdata offset at load when a verified runtime layout is
 * available. intj/torch_intf/cpp_detect.py also compiles this declaration beside
 * torch's own for every torch in the ABI table.
 */
#pragma once

#include <Python.h>
#include <ATen/core/Tensor.h>
#include <c10/util/MaybeOwned.h>
#include <torch/version.h>

/* torch 2.10 made `cdata` a plain Tensor; before, a MaybeOwned<Tensor>. */
#define INTJ_CDATA_IS_MAYBE_OWNED                                              \
  (TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR < 10)

struct intj_THPVariable {
  PyObject_HEAD
#if INTJ_CDATA_IS_MAYBE_OWNED
  c10::MaybeOwned<at::Tensor> cdata;
#else
  at::Tensor cdata;
#endif
};

static inline const at::Tensor &intj_cdata(PyObject *o) {
#if INTJ_CDATA_IS_MAYBE_OWNED
  return *((intj_THPVariable *)o)->cdata;
#else
  return ((intj_THPVariable *)o)->cdata;
#endif
}

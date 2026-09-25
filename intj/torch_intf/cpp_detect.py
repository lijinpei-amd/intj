"""Detect the torch ABI by compiling against torch's own headers.

The C++ half of the check `abi_detect` makes at runtime: here the compiler, not
a search over field values, places each field.  The two share no code, so a
table entry both agree on has been checked twice, independently.

Needs a C++ compiler (`$CXX`, else `c++`) and torch's headers, but not triton:
it runs in an environment with nothing but the torch whose entry it checks.

    python -m intj.torch_intf.cpp_detect
"""

from __future__ import annotations

import ctypes
import os
import pathlib
import subprocess
import sysconfig
import tempfile

_RUNTIME = pathlib.Path(__file__).parent.parent / "runtime"

#: Member pointers to private fields, taken the one legal way: access checking
#: does not apply to the arguments of an explicit instantiation.  Not
#: `#define private public`, which compiles classes other than the ones torch
#: was built from.
_SOURCE = r"""
#include <Python.h>
#include <cstddef>
#include <cstdio>
#include <string>
#include <type_traits>
#include <c10/core/ScalarType.h>
#include <c10/core/StorageImpl.h>
#include <c10/core/TensorImpl.h>
#include <c10/util/typeid.h>
#include <torch/csrc/autograd/python_variable.h>

#include "intj_thpvariable.h"

template <class Tag> struct Member { static inline typename Tag::type ptr; };
template <class Tag, typename Tag::type M> struct Steal {
  static inline const int _ = (Member<Tag>::ptr = M, 0);
};
#define STEAL(name, Class, Type, field)        \
  struct name { using type = Type Class::*; }; \
  template struct Steal<name, &Class::field>;

using TensorPtr = c10::intrusive_ptr<c10::TensorImpl, c10::UndefinedTensorImpl>;
using StoragePtr = c10::intrusive_ptr<c10::StorageImpl>;
STEAL(MO_own, c10::MaybeOwned<at::Tensor>, at::Tensor, own_)
STEAL(TB_impl, at::TensorBase, TensorPtr, impl_)
STEAL(TP_target, TensorPtr, c10::TensorImpl *, target_)
STEAL(TI_storage, c10::TensorImpl, c10::Storage, storage_)
STEAL(TI_storage_offset, c10::TensorImpl, int64_t, storage_offset_)
STEAL(TI_numel, c10::TensorImpl, int64_t, numel_)
STEAL(TI_data_type, c10::TensorImpl, caffe2::TypeMeta, data_type_)
STEAL(TM_index, caffe2::TypeMeta, uint16_t, index_)
STEAL(S_impl, c10::Storage, StoragePtr, storage_impl_)
STEAL(SP_target, StoragePtr, c10::StorageImpl *, target_)
STEAL(SI_data_ptr, c10::StorageImpl, c10::DataPtr, data_ptr_)
STEAL(DP_ptr, c10::DataPtr, c10::detail::UniqueVoidPtr, ptr_)
STEAL(UVP_data, c10::detail::UniqueVoidPtr, void *, data_)
STEAL(SI_size_bytes, c10::StorageImpl, c10::SymInt, size_bytes_)
STEAL(SYM_data, c10::SymInt, int64_t, data_)

// offsetof through a member pointer: address arithmetic on aligned storage,
// no object constructed.
template <class C, class T> size_t off(T C::*m) {
  alignas(C) static char buf[sizeof(C)];
  return (size_t)((char *)&(reinterpret_cast<C *>(buf)->*m) - buf);
}
#define OFF(tag) off(Member<tag>::ptr)

// torch < 2.10 holds THPVariable::cdata as a MaybeOwned<Tensor>, whose borrowed
// and owned Tensors share one union after an `isBorrowed_` flag.
using Cdata = decltype(THPVariable::cdata);
constexpr bool cdata_is_tensor = std::is_same_v<Cdata, at::Tensor>;
size_t tensor_in_cdata() {
  if constexpr (cdata_is_tensor)
    return 0;
  else
    return OFF(MO_own);
}

// A shared object rather than a program: pybind11, which python_variable.h
// pulls in, leaves references to libpython that only a live interpreter fills.
extern "C" const char *intj_detect() {
  static std::string out;
  char buf[512];
  auto emit = [&](auto... a) { std::snprintf(buf, sizeof buf, a...); out += buf; };
  static_assert(sizeof(size_t) == 8, "the table is for 64-bit targets");
  // Every intrusive_ptr, Storage, DataPtr and SymInt is followed to the word
  // the reader actually loads; `data_type` is TypeMeta's low byte, which is
  // the ScalarType on a little-endian machine.
  size_t thp_cdata = offsetof(THPVariable, cdata);
  // The STATIC_COMPILE mode's own declaration, which it trusts without a
  // load-time check.
  bool intj_ok = std::is_same_v<Cdata, decltype(intj_THPVariable::cdata)> &&
                 offsetof(intj_THPVariable, cdata) == thp_cdata;
  (void)&intj_cdata;  // and its accessor still compiles against this torch
  emit("pyobject=%zu thpvariable_cdata=%zu cdata_is_tensor=%d intj_thpvariable=%d "
       "tensorimpl=%zu storageimpl=%zu\n",
       sizeof(PyObject), thp_cdata, (int)cdata_is_tensor, (int)intj_ok,
       sizeof(c10::TensorImpl), sizeof(c10::StorageImpl));
  emit("cdata=%zu storage=%zu storage_offset=%zu numel=%zu data_type=%zu "
       "s_data=%zu s_nbytes=%zu\n",
       thp_cdata + tensor_in_cdata() + OFF(TB_impl) + OFF(TP_target),
       OFF(TI_storage) + OFF(S_impl) + OFF(SP_target),
       OFF(TI_storage_offset), OFF(TI_numel),
       OFF(TI_data_type) + OFF(TM_index),
       OFF(SI_data_ptr) + OFF(DP_ptr) + OFF(UVP_data),
       OFF(SI_size_bytes) + OFF(SYM_data));
  for (int t = 0; t < (int)c10::ScalarType::NumOptions; t++)
    emit("%s%d=%zu", t ? " " : "", t,
         caffe2::TypeMeta::fromScalarType((c10::ScalarType)t).itemsize());
  return out.c_str();
}
"""


def measure() -> tuple[dict[str, int], dict[str, int], dict[int, int]]:
    """(facts, offsets, dtype code -> element size), unchecked.

    `facts` are what the layout rests on rather than the layout itself:
    `sizeof(PyObject)`, where and what `THPVariable::cdata` is, whether the
    STATIC_COMPILE mode's `intj_THPVariable` matches it, and the sizes of
    TensorImpl and StorageImpl.
    """
    import torch
    from torch.utils import cpp_extension

    includes, libs = cpp_extension.include_paths(), cpp_extension.library_paths()
    cxx11_abi = int(torch._C._GLIBCXX_USE_CXX11_ABI)  # pyright: ignore[reportPrivateUsage]
    with tempfile.TemporaryDirectory() as tmp:
        src, lib = pathlib.Path(tmp, "detect.cpp"), pathlib.Path(tmp, "detect.so")
        src.write_text(_SOURCE)
        subprocess.run(
            [
                os.environ.get("CXX", "c++"), "-std=c++20", "-shared", "-fPIC", "-w",
                f"-D_GLIBCXX_USE_CXX11_ABI={cxx11_abi}",
                *(f"-I{d}" for d in [*includes, sysconfig.get_paths()["include"], _RUNTIME]),
                str(src), "-o", str(lib),
                *(f"-L{d}" for d in libs), *(f"-Wl,-rpath,{d}" for d in libs),
                "-lc10", "-ltorch_cpu", "-ltorch_python",
            ],
            check=True,
        )
        # `import torch` above loaded the symbols the .so leaves undefined
        detect_fn = ctypes.CDLL(str(lib)).intj_detect
        detect_fn.restype = ctypes.c_char_p
        out = detect_fn().decode()
    head, layout, dtypes = (dict(p.split("=") for p in line.split()) for line in out.splitlines())
    return (
        {k: int(v) for k, v in head.items()},
        {k: int(v) for k, v in layout.items()},
        {int(k): int(v) for k, v in dtypes.items()},
    )


def detect() -> tuple[dict[str, int], dict[int, int]]:
    """(offsets, dtype code -> element size), with cdata relative to PyObject_HEAD.

    `measure()` keeps the raw absolute offsets declared by torch's headers.

    Raises if the toolchain is missing, the program does not build, or the
    STATIC_COMPILE mode's `intj_THPVariable` (runtime/intj_thpvariable.h) no
    longer matches torch's `THPVariable` -- the one layout that mode declares rather than
    includes.
    """
    head, offsets, sizes = measure()
    if not head["intj_thpvariable"]:
        raise RuntimeError(
            "intj: runtime/intj_thpvariable.h does not match this torch's THPVariable "
            f"(cdata at {head['thpvariable_cdata']}, "
            f"{'a Tensor' if head['cdata_is_tensor'] else 'a MaybeOwned<Tensor>'})"
        )
    if offsets["cdata"] < head["pyobject"]:
        raise RuntimeError("intj: torch's cdata pointer slot precedes PyObject_HEAD")
    offsets["cdata"] -= head["pyobject"]
    return offsets, sizes


if __name__ == "__main__":
    offsets, sizes = detect()
    print(" ".join(f"{k}={v}" for k, v in offsets.items()))
    print(" ".join(f"{k}={v}" for k, v in sizes.items()))

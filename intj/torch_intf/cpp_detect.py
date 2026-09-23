"""Detect the torch ABI by compiling against torch's own headers.

The C++ half of the check `abi_detect` makes at runtime: here the compiler, not
a search over field values, places each field.  The two share no code, so a
table entry both agree on has been checked twice, independently.

Needs a C++ compiler and torch's headers -- the toolchain the `CXX` mode uses.

    python -m intj.torch_intf.cpp_detect
"""

import ctypes
import pathlib
import subprocess
import sysconfig
import tempfile

#: Member pointers to private fields, taken the one legal way: access checking
#: does not apply to the arguments of an explicit instantiation.  Not
#: `#define private public`, which compiles classes other than the ones torch
#: was built from.
_SOURCE = r"""
#include <Python.h>
#include <cstddef>
#include <cstdio>
#include <string>
#include <c10/core/ScalarType.h>
#include <c10/core/StorageImpl.h>
#include <c10/core/TensorImpl.h>
#include <c10/util/typeid.h>
#include <torch/csrc/autograd/python_variable.h>

template <class Tag> struct Member { static inline typename Tag::type ptr; };
template <class Tag, typename Tag::type M> struct Steal {
  static inline const int _ = (Member<Tag>::ptr = M, 0);
};
#define STEAL(name, Class, Type, field)        \
  struct name { using type = Type Class::*; }; \
  template struct Steal<name, &Class::field>;

using TensorPtr = c10::intrusive_ptr<c10::TensorImpl, c10::UndefinedTensorImpl>;
using StoragePtr = c10::intrusive_ptr<c10::StorageImpl>;
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
  emit("pyobject=%zu thpvariable_cdata=%zu\n", sizeof(PyObject), thp_cdata);
  emit("cdata=%zu storage=%zu storage_offset=%zu numel=%zu data_type=%zu "
       "s_data=%zu s_nbytes=%zu\n",
       thp_cdata + OFF(TB_impl) + OFF(TP_target),
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


def detect() -> tuple[dict[str, int], dict[int, int]]:
    """(offsets, dtype code -> element size), as torch's headers declare them.

    Raises if the toolchain is missing, the program does not build, or torch's
    `THPVariable` no longer starts with the tensor -- the one layout the `CXX`
    mode assumes rather than includes.
    """
    from ..launcher import _cxx_abi, _cxx_toolchain  # pyright: ignore[reportPrivateUsage]

    toolchain = _cxx_toolchain()
    if toolchain is None:
        raise RuntimeError("intj: cpp_detect needs a C++ compiler and torch's headers")
    includes, libs = toolchain
    with tempfile.TemporaryDirectory() as tmp:
        src, lib = pathlib.Path(tmp, "detect.cpp"), pathlib.Path(tmp, "detect.so")
        src.write_text(_SOURCE)
        subprocess.run(
            [
                "c++", "-std=c++20", "-shared", "-fPIC", "-w", f"-D_GLIBCXX_USE_CXX11_ABI={_cxx_abi()}",
                *(f"-I{d}" for d in [*includes, sysconfig.get_paths()["include"]]),
                str(src), "-o", str(lib),
                *(f"-L{d}" for d in libs), *(f"-Wl,-rpath,{d}" for d in libs),
                "-lc10", "-ltorch_cpu", "-ltorch_python",
            ],
            check=True,
        )
        import torch  # pyright: ignore[reportUnusedImport] -- symbols the .so leaves undefined

        detect_fn = ctypes.CDLL(str(lib)).intj_detect
        detect_fn.restype = ctypes.c_char_p
        out = detect_fn().decode()
    head, layout, dtypes = (dict(p.split("=") for p in line.split()) for line in out.splitlines())
    if head["thpvariable_cdata"] != head["pyobject"]:
        raise RuntimeError(
            f"intj: THPVariable::cdata is at {head['thpvariable_cdata']}, not right after "
            f"PyObject_HEAD ({head['pyobject']}); intj_THPVariable is stale"
        )
    return (
        {k: int(v) for k, v in layout.items()},
        {int(k): int(v) for k, v in dtypes.items()},
    )


if __name__ == "__main__":
    offsets, sizes = detect()
    print(" ".join(f"{k}={v}" for k, v in offsets.items()))
    print(" ".join(f"{k}={v}" for k, v in sizes.items()))

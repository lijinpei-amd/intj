# Access modes

These modes describe how generated C/C++ code obtains values from the Python
interpreter and PyTorch. `CPYTHON_ACCESS_MODE` and `TORCH_ACCESS_MODE` name
separate interfaces. Currently only the Torch mode is selectable; CPython
scalar access uses `STATIC_COMPILE` in every generated module.

| Mode | Definition |
| --- | --- |
| `INTERPRETER` | Call the CPython interpreter to obtain the needed value, equivalent to executing the corresponding Python operations. |
| `RUNTIME_SHIM` | Use the same shim implementation across supported CPython or PyTorch versions. Pass parameters for the running version at runtime and read binary fields using previously verified layouts. |
| `STATIC_COMPILE` | Compile a version-specific shim. CPython or PyTorch version macros and headers select the code and layout at compile time. |

`TORCH_ACCESS_MODE` can be `INTERPRETER`, `RUNTIME_SHIM`, or `STATIC_COMPILE`.
`CPYTHON_ACCESS_MODE` can be `INTERPRETER` or `STATIC_COMPILE`.

`intj.TorchAccessMode` exposes the three Torch modes. By default,
`make_launcher(..., torch_access_mode=None)` selects `STATIC_COMPILE` when the
Torch C++ toolchain is available, else `RUNTIME_SHIM` when a verified layout
exists, else `INTERPRETER`. An explicitly selected unavailable mode raises.

The `INTERPRETER` Torch reader obtains the tensor pointer and storage size
through CPython calls, but reads the checked `THPDtype` code directly. The
CPython scalar readers use `STATIC_COMPILE` through `PY_VERSION_HEX` branches
in `cpython_abi.h`, independently of the Torch mode. One runtime shim
implementation can serve multiple Torch versions, while each compiled
extension still targets a compatible CPython ABI.

For `RUNTIME_SHIM`, the recorded tensor `cdata` offset includes a 16-byte
`PyObject` header. `layout_for()` replaces that portion with CPython's reported
`object.__basicsize__`, and `set_torch_version()` saves the adjusted offset in
module state before the first launch.

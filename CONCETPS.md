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

For `RUNTIME_SHIM`, each version file in `torch_abi/` stores `cdata` relative to the end of
`PyObject_HEAD`. `layout_for()` selects the verified Torch row, and the
generated module's `_set_torch_abi()` adds its compiled `sizeof(PyObject)`
and saves the absolute offset before the first launch.

# Specialization

`intj/specialization.py` holds both sides of the invariant
"same intj spec key ⟹ same triton specialization".

| Term | Definition |
| --- | --- |
| spec key | The bytes a launch looks its compiled kernel up by: the key fields of the parameters, level-0 computed keys and dynamic values, then the device byte unless the launcher is bound to a device. |
| key field | One piece of a parameter's spec key: a `descriptor` (type and facts), a `payload` (a constexpr's value), or `exact_kind`/`exact` (an exact key). |
| call layout | Where each parameter sits in the public call (`call_index`, None when baked, bound or tuned) and which take objects (`memo`). Places no key field. |
| key layout | Where every key field sits in the spec key, widest first, and how many 64-bit words the key takes. |
| compiler input | What Triton is given for one call: signature, constants and attrs. The key must never be coarser than it. |

# Compiling a miss

| Term | Definition |
| --- | --- |
| record | What a miss returns for C to cache and launch: `(function, block_dim, shared, nparams[, compiled[, deps, computed[, meta]]])`. |
| kernel compiler | `intj/compiler.py`'s `KernelCompiler`: turns a miss into a record, compiling each identity once per device through `triton_load`. |
| compile override | A launcher's replacement for its kernel compiler's misses (`override_compile`), for tests and benchmarks. |

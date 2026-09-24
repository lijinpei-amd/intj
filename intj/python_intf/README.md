# Python_intf

INTJ depends on these CPython details at the binary level:

- The integer `PyLongObject` layout.
- The float `PyFloatObject` layout.
- The `PyObject` header size, needed to locate PyTorch's tensor payload
  (`THPVariable::cdata`) from a `PyObject *`.

These details can vary with the CPython version, free-threading configuration,
and CPU architecture; INTJ currently supports x86-64. In every Torch access
mode, the extension is compiled against the running interpreter's `Python.h`.
`PY_VERSION_HEX` and `Py_GIL_DISABLED` select CPython-specific code at compile
time. The module key includes the exact CPython version, build configuration,
and extension suffix, so neither `RUNTIME_SHIM` nor `STATIC_COMPILE` reuses a
binary across CPython ABIs.

## Integer PyObject Layout

- **CPython 3.8–3.11:** the signed `ob_size` (`Py_SIZE`) gives the sign and the
  number of base-2³⁰ digits in `ob_digit[]`.
- **CPython 3.12–3.14:** `long_value.lv_tag` encodes the sign and digit count.
  A compact integer has at most one base-2³⁰ digit, including zero
  (`abs(value) < 2**30`). The inline `PyUnstable_Long_IsCompact` tests
  `lv_tag < (2 << _PyLong_NON_SIZE_BITS)`; `PyUnstable_Long_CompactValue`
  returns its signed value. Larger integers use `long_value.ob_digit[]` for
  their digits.

Free-threaded 3.13t and 3.14t use their version's integer scheme. The decoder
requires 30-bit digits and checks that assumption at compile time.

## Float PyObject Layout

All supported CPython versions, 3.8–3.14 including 3.13t and 3.14t, use the
same scheme: `PyFloatObject.ob_fval` stores a C `double` after the object header.
INTJ reads that field directly; `Python.h` supplies its offset for the build.

## PyObject Header Size

`Python.h` defines `PyObject_HEAD`, so the compiler knows its size as
`sizeof(PyObject)`. In `STATIC_COMPILE`, that header is part of the C++
`THPVariable` declaration and the compiler locates `cdata`. In
`RUNTIME_SHIM`, the recorded `cdata` offset starts after `PyObject_HEAD`.
`cpython_abi.pyobject_size()` (`object.__basicsize__`) supplies the detecting
interpreter's header size. At module load, the compiled setter adds its own
`sizeof(PyObject)` and saves the absolute offset before the first launch.

## Free-threaded Builds

On a free-threaded build, `Py_GIL_DISABLED` selects a `PyMutex` to protect
kernel-cache reads and updates and the compile-callback reference. The lock
operations compile away in a GIL build. On supported x86-64 builds, the
compile-time `sizeof(PyObject)` is 32 bytes with free threading and 16 bytes
with the GIL, shifting the `cdata` field accordingly.

See [AGENTS.md](AGENTS.md) for implementation contracts and verification.

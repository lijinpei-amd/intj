# TODO

- **Record `cdata` relative to `PyObject_HEAD`.** `abi_detect` currently
  records an absolute offset and generates table entries only when
  `sizeof(PyObject) == 16`. Pass the detecting CPython's header size from
  `python_intf`, record `absolute_cdata - header_size`, and reconstruct the
  runtime offset using the target module's compiled `sizeof(PyObject)`. Migrate
  the existing `torch_abi.toml` rows, update the independent `cpp_detect`
  check, remove the 16-byte generation guard, and verify GIL and free-threaded
  builds.

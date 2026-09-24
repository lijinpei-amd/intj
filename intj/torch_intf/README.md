# torch_intf

INTJ's interface to pytorch internals. We need certain internal details of pytorch C++ at binary level:

- field layout of at::Tensor
- Scalar DType enum

And these details depend on pytorch version, and different pytorch interface modes get this info differently:

- RUNTIME_SHIM, same python extension is shared by different pytorch versions, parameterized by an abi table selected from pytorch version.
- STATIC_COMPILE, different python extension is built for each different pytorch version, getting this info from the pytorch C++ headers.
- INTERPRETER calls Torch for the pointer and size, with one checked direct read of the THPDtype code.

For the abi table used by RUNTIME_SHIM, a toml file mapping torch version to abi details is included. To rebuild or verify the table, there are two ways:

Its recorded `cdata` offset includes a 16-byte `PyObject` header. At module load,
intj replaces that portion with the size reported by `python_intf` and saves the
resulting offset in the module's ABI state.

- Through runtime detection on a default-GIL CPython build, see abi_detect.py. This method doesn't need to compile and run a C++ extension.
- Through compile and run detection, see cpp_detect.py. This method compiles some C++ files using pytorch C++ headers, then runs the resulting binary and outputs the needed info.

Our provided table is checked using both method in our CI for supported pytorch version.

Supported: pytorch >= 2.2; for python, see `../python_intf`.

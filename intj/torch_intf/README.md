# torch_intf

INTJ's interface to pytorch internals. We need certain internal details of pytorch C++ at binary level:

- field layout of at::Tensor
- Scalar DType enum

And these details depend on pytorch version, and different pytorch interface modes get this info differently:

- Shim, same python extension is shared by different pytorch versions, parameterized by an abi table selected from pytorch version.
- CXX, different python extension is built for each different pytorch version, getting this info from the pytorch C++ headers.
- CPython: don't need these details.

For the abi table used by shim, a toml file mapping torch version to abi details is included. To rebuild or verify the table, there are two ways:

- Through runtime detection, see abi_detect.py. This method doesn't need to compile and run some C++ extension.
- Through compile and run detection, see cpp_detect.py. This method compiles some C++ files using pytorch C++ headers, then runs the resulting binary and outputs the needed info.

Our provided table is checked using both method in our CI for supported pytorch version.

Supported: python >= 3.12, pytorch >= 2.2.
"""How generated modules read torch tensors.

Modules:
    torch_abi: the torch access modes and the verified `RUNTIME_SHIM` layouts.

Generate layouts with `python -m intj.tools.abi_detect` and check them against
Torch's C++ headers with `python -m intj.tools.cpp_detect`.
"""

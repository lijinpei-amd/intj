"""How generated modules read torch tensors.

Modules:
    torch_abi: the torch access modes and the verified `RUNTIME_SHIM` layouts.
    abi_detect: derives a layout from the running torch
        (`python -m intj.torch_intf.abi_detect`).
    cpp_detect: the independent, compiler-based check of those layouts.
"""

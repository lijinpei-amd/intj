"""Checks and generates ABI information for INTJ's interface layers.

Modules:
    check: compares the CPython header's readers with the running interpreter.
    abi_detect: generates Torch layouts from runtime tensor probes.
    cpp_detect: independently checks Torch layouts against its C++ headers.

Run each tool with `python -m intj.tools.<module>`.
"""

"""How generated modules read CPython objects.

Modules:
    cpython_abi: selects the verified CPython internals header per version.
    check: checks that header against the running interpreter
        (`python -m intj.python_intf.check [header]`).
"""

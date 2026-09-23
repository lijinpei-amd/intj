"""Which CPython internals a module is built with, and the ones python reads.

The C side is one header per CPython int layout, selected by interpreter version
(`header_for`).  A version maps to a header only once `python -m
intj.python_intf.check` has passed on it, default and free-threaded build alike.
"""

from __future__ import annotations

import sys

#: (major, minor) -> the header in this directory implementing that CPython.
#: Only verified versions: an unlisted one gets no guess from its neighbours.
_HEADERS = {
    (3, 8): "cpython_38.h",
    (3, 9): "cpython_38.h",
    (3, 10): "cpython_38.h",
    (3, 11): "cpython_38.h",
    (3, 12): "cpython_312.h",
    (3, 13): "cpython_312.h",
    (3, 14): "cpython_312.h",
}


def python_version() -> tuple[int, int, int]:
    return (sys.version_info.major, sys.version_info.minor, sys.version_info.micro)


def supported_versions() -> list[tuple[int, int]]:
    return sorted(_HEADERS)


def header_for(version: tuple[int, ...] | None = None) -> str | None:
    """The implementation header for `version`, or None if intj has not verified one."""
    major, minor = (version or python_version())[:2]
    return _HEADERS.get((major, minor))


def pyobject_size() -> int:
    """`sizeof(PyObject)`, the offset of the first field past `PyObject_HEAD`.

    `object.__basicsize__` is exactly that, on every build -- including the
    free-threaded one, whose header is not two pointers.
    """
    return object.__basicsize__

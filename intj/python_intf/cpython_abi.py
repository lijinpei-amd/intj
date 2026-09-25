"""Which CPython internals a module is built with, and the ones python reads.

The C header selects its int layout at compile time.  A version maps to it only
once `python -m intj.python_intf.check` has passed on both the default and
free-threaded builds.
"""

from __future__ import annotations

import sys

#: (major, minor) -> the verified C header implementing that CPython.
#: Only verified versions: an unlisted one gets no guess from its neighbours.
_HEADERS = {
    (3, 8): "cpython_abi.h",
    (3, 9): "cpython_abi.h",
    (3, 10): "cpython_abi.h",
    (3, 11): "cpython_abi.h",
    (3, 12): "cpython_abi.h",
    (3, 13): "cpython_abi.h",
    (3, 14): "cpython_abi.h",
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

"""Selects the CPython internals header a module is built with.

Also reports the values python itself reads: its version and `sizeof(PyObject)`.

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
    """Returns the running interpreter's (major, minor, micro)."""
    return (sys.version_info.major, sys.version_info.minor, sys.version_info.micro)


def supported_versions() -> list[tuple[int, int]]:
    """Returns every verified (major, minor), sorted."""
    return sorted(_HEADERS)


def header_for(version: tuple[int, ...] | None = None) -> str | None:
    """Returns the header for `version` (default: running), or None if unverified."""
    major, minor = (version or python_version())[:2]
    return _HEADERS.get((major, minor))


def pyobject_size() -> int:
    """Returns `sizeof(PyObject)`, the offset of the first field past `PyObject_HEAD`.

    `object.__basicsize__` is exactly that, on every build -- including the
    free-threaded one, whose header is not two pointers.
    """
    return object.__basicsize__

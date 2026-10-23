"""Launches `@triton.jit` kernels through generated C entry points.

Exports:
    make_launcher: builds a launcher for a kernel.
    MakeLauncherOption: every make_launcher keyword, as one value.
    KernelCache: the kernel cache backend selection.
    TorchAccessMode: how generated code reads torch tensors.
    ClassGlobalWarning: the warning for class-valued globals.
    Annotation, Argument, Constexpr, SpecializationPolicy, AUTO, NEVER, Assume,
        Exactly, Fact, EqualTo, Aligned, PointerRange, BindValue: per-argument
        annotations.
    INT_TYPES, FLOAT_TYPES: scalar type names, loaded lazily from `annotation`.
    __version__: the package version.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ._version import __version__
from .annotation import (
    AUTO,
    NEVER,
    Aligned,
    Annotation,
    Argument,
    Assume,
    BindValue,
    Constexpr,
    EqualTo,
    Exactly,
    Fact,
    PointerRange,
    SpecializationPolicy,
)
from .kernel_cache import KernelCache
from .launcher import ClassGlobalWarning, make_launcher
from .options import MakeLauncherOption
from .torch_intf.torch_abi import TorchAccessMode

if TYPE_CHECKING:
    from .annotation import FLOAT_TYPES, INT_TYPES


def __getattr__(name: str) -> Any:
    """Returns `INT_TYPES` or `FLOAT_TYPES`, imported from `annotation` on first use.

    Raises:
        AttributeError: `name` is neither.
    """
    if name in ("INT_TYPES", "FLOAT_TYPES"):
        from . import annotation

        return getattr(annotation, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "KernelCache",
    "TorchAccessMode",
    "make_launcher",
    "MakeLauncherOption",
    "ClassGlobalWarning",
    "Annotation",
    "Argument",
    "Constexpr",
    "SpecializationPolicy",
    "AUTO",
    "NEVER",
    "Assume",
    "Exactly",
    "Fact",
    "EqualTo",
    "Aligned",
    "PointerRange",
    "BindValue",
    "INT_TYPES",
    "FLOAT_TYPES",
    "__version__",
]

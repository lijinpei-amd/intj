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
    Fact,
    PointerRange,
    Specialization,
)
from .kernel_cache import KernelCache
from .launcher import make_launcher
from .torch_intf.torch_abi import TorchAccessMode

if TYPE_CHECKING:
    from .annotation import FLOAT_TYPES, INT_TYPES


def __getattr__(name: str) -> Any:
    if name in ("INT_TYPES", "FLOAT_TYPES"):
        from . import annotation

        return getattr(annotation, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "KernelCache",
    "TorchAccessMode",
    "make_launcher",
    "Annotation",
    "Argument",
    "Constexpr",
    "Specialization",
    "AUTO",
    "NEVER",
    "Assume",
    "Fact",
    "EqualTo",
    "Aligned",
    "PointerRange",
    "BindValue",
    "INT_TYPES",
    "FLOAT_TYPES",
    "__version__",
]

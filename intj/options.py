"""`MakeLauncherOption`: every `make_launcher` keyword but `delayed`, as one value."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping, Sequence
from typing import Any, TypedDict

from .kernel_cache import KernelCache
from .torch_intf.torch_abi import TorchAccessMode


@dataclasses.dataclass(frozen=True)
class MakeLauncherOption:
    """Every `make_launcher` keyword except `delayed`, stored as given.

    Nothing is validated here: `k.__self__.check()`, at the latest the first
    call, checks the values with the errors `make_launcher` used to raise.
    Mappings and the `dynamic_options` sequence are copied, so a caller
    changing its dict or list afterwards does not change the launcher.
    `k.__self__.options` is a launcher's; `dataclasses.replace` makes variants.
    """

    jit_func: Any
    dynamic_options: Sequence[str] = ()
    extra_annotation: Mapping[str, object] | None = None
    options: Mapping[str, Any] | None = None
    torch_access_mode: TorchAccessMode | None = None
    kernel_cache: KernelCache = KernelCache.INTJ
    no_gpu: bool = False
    verify_annotation: bool = False
    bind_device: int | None = None
    bind_values: Mapping[str, object] | None = None
    grid_arg: int | None = None
    grid_cpp: object | None = None
    grid_py: Callable[[dict[str, object]], object] | None = None
    return_compiled: bool = False
    assume_constant_globals: bool = False

    def __post_init__(self) -> None:
        # a snapshot: check() reads these later; anything else stays as given
        # for check() to report
        for name in ("extra_annotation", "options", "bind_values"):
            value: object = getattr(self, name)
            if isinstance(value, Mapping):
                object.__setattr__(self, name, dict(value))
        if not isinstance(self.dynamic_options, str):
            try:
                object.__setattr__(self, "dynamic_options", tuple(self.dynamic_options))
            except TypeError:
                pass  # not iterable


class MakeLauncherKwargs(TypedDict, total=False):
    """`make_launcher`'s keywords: `MakeLauncherOption`'s fields but `jit_func`."""

    dynamic_options: Sequence[str]
    extra_annotation: Mapping[str, object] | None
    options: Mapping[str, Any] | None
    torch_access_mode: TorchAccessMode | None
    kernel_cache: KernelCache
    no_gpu: bool
    verify_annotation: bool
    bind_device: int | None
    bind_values: Mapping[str, object] | None
    grid_arg: int | None
    grid_cpp: object | None
    grid_py: Callable[[dict[str, object]], object] | None
    return_compiled: bool
    assume_constant_globals: bool


#: the keywords `make_launcher` (and `delayed`) may set
OPTION_FIELDS: frozenset[str] = frozenset(
    f.name for f in dataclasses.fields(MakeLauncherOption)
) - {"jit_func"}

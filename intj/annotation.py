"""Argument annotations understood by intj launchers."""

import dataclasses
import enum
from collections.abc import Sequence
from typing import Any, cast

import triton.language as tl


@dataclasses.dataclass(frozen=True)
class Annotation:
    pass


@dataclasses.dataclass(frozen=True)
class Specialization:
    pass


@dataclasses.dataclass(frozen=True)
class _Auto(Specialization):
    pass


@dataclasses.dataclass(frozen=True)
class _Never(Specialization):
    pass


AUTO = _Auto()
NEVER = _Never()


@dataclasses.dataclass(frozen=True)
class Fact:
    pass


@dataclasses.dataclass(frozen=True)
class EqualTo(Fact):
    value: int


@dataclasses.dataclass(frozen=True)
class Aligned(Fact):
    value: int


@dataclasses.dataclass(frozen=True)
class PointerRange(Fact):
    value: int


@dataclasses.dataclass(frozen=True, init=False)
class Assume(Specialization):
    facts: tuple[Fact, ...]

    def __init__(self, *facts: Fact) -> None:
        allowed: dict[type[Fact], int] = {
            EqualTo: 1,
            Aligned: 16,
            PointerRange: 32,
        }
        for fact in facts:
            expected = allowed.get(type(fact))
            if expected is None:
                raise TypeError("intj: Assume accepts exact built-in Fact instances")
            if getattr(fact, "value") != expected:
                raise ValueError(
                    f"intj: only {type(fact).__name__}({expected}) is supported"
                )
        object.__setattr__(self, "facts", tuple(facts))


@dataclasses.dataclass(frozen=True)
class _Unset:
    pass


UNSET = _Unset()


class BindValue(enum.Enum):
    TENSOR = "tensor"
    POINTER = "pointer"


def _normalize_type_field(value: object) -> object:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        value = tuple(value)
        if not value:
            raise ValueError("intj: type sequence must not be empty")
    return value


@dataclasses.dataclass(frozen=True)
class Argument(Annotation):
    type: Any = UNSET
    specialize: Specialization | _Unset = UNSET
    value: object = UNSET
    bind_value: BindValue | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "type", _normalize_type_field(self.type))
        if not isinstance(cast(object, self.specialize), (Specialization, _Unset)):
            raise TypeError("intj: specialize must be AUTO, NEVER, or Assume(...)")
        if self.bind_value is not None and not isinstance(
            cast(object, self.bind_value), BindValue
        ):
            raise TypeError("intj: bind_value must be a BindValue or None")
        if self.value is not UNSET and self.bind_value is not None:
            raise ValueError("intj: value and bind_value are mutually exclusive")


@dataclasses.dataclass(frozen=True)
class Constexpr(Annotation):
    type: Any = UNSET
    power_of_two_or_zero: bool = False
    value: object = UNSET

    def __post_init__(self) -> None:
        object.__setattr__(self, "type", _normalize_type_field(self.type))
        if not isinstance(cast(object, self.power_of_two_or_zero), bool):
            raise TypeError("intj: power_of_two_or_zero must be bool")


INT_TYPES = (tl.int32, tl.int64, tl.uint64)
FLOAT_TYPES = (tl.float32,)

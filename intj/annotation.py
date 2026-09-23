"""Argument annotations understood by intj launchers."""

import dataclasses
import enum
import inspect
import struct
from collections.abc import Mapping, Sequence
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
            value = getattr(fact, "value")
            if type(value) is not int:
                raise TypeError("intj: Fact value must be an int")
            if value != expected:
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
        if self.value is not UNSET:
            hash(self.value)


@dataclasses.dataclass(frozen=True)
class Constexpr(Annotation):
    type: Any = UNSET
    power_of_two_or_zero: bool = False
    value: object = UNSET

    def __post_init__(self) -> None:
        object.__setattr__(self, "type", _normalize_type_field(self.type))
        if not isinstance(cast(object, self.power_of_two_or_zero), bool):
            raise TypeError("intj: power_of_two_or_zero must be bool")
        if self.value is not UNSET:
            hash(self.value)


INT_TYPES = (tl.int32, tl.int64, tl.uint64)
FLOAT_TYPES = (tl.float32,)


@dataclasses.dataclass(frozen=True)
class KeyField:
    kind: str
    width: int
    offset: int = -1


@dataclasses.dataclass(frozen=True)
class CanonicalAnnotation:
    kind: str
    types: tuple[str | None, ...] | None
    equal_to_one: str
    aligned_16: str
    pointer_range_32: str
    facts: tuple[tuple[str, int], ...]
    power_of_two_or_zero: bool
    bind_value: str | None
    baked_value: tuple[object, ...]
    key_fields: tuple[KeyField, ...] = ()


@dataclasses.dataclass(frozen=True)
class ResolvedParam:
    name: str
    index: int
    annotation: CanonicalAnnotation
    baked: object = dataclasses.field(default=UNSET, compare=False, hash=False)


def _canonical_value(value: object) -> tuple[object, ...]:
    if value is None:
        return ("none",)
    if type(value) is bool:
        return ("bool", value)
    if type(value) is int:
        return ("int", str(value))
    if type(value) is float:
        return ("float64", struct.pack(">d", value).hex())
    raise ValueError("intj: baked values must be scalar int, float, bool, or None")


def _annotation_source(value: object, name: str) -> Argument | Constexpr | None:
    if value is inspect.Parameter.empty:
        return None
    if type(value) is Argument or type(value) is Constexpr:
        return value
    if value is tl.constexpr:
        return Constexpr()
    if value is None or isinstance(value, tl.dtype):
        return Argument(type=value)
    raise ValueError(f"intj: parameter {name!r} has unsupported annotation {value!r}")


def _merge_field(
    name: str, field: str, left: object, right: object, *, kind: str = "argument"
) -> object:
    if left is UNSET:
        return right
    if right is UNSET:
        return left
    if field == "type":
        equal = _canonical_types(left, kind) == _canonical_types(right, kind)
    elif field == "value" and left is not UNSET and right is not UNSET:
        equal = type(left) is type(right) and (
            struct.pack(">d", left) == struct.pack(">d", right)
            if type(left) is float else left == right
        )
    else:
        equal = left == right
    if not equal:
        raise ValueError(f"intj: parameter {name!r} has conflicting {field}")
    return left


_FACT_NAMES = ("equal_to_one", "aligned_16", "pointer_range_32")
_FACT_CLASSES: dict[type[Fact], tuple[str, int]] = {
    EqualTo: ("equal_to_one", 1),
    Aligned: ("aligned_16", 16),
    PointerRange: ("pointer_range_32", 32),
}


def _specialization_modes(value: object) -> tuple[object, object, object]:
    if value is UNSET:
        return (UNSET, UNSET, UNSET)
    if type(value) is _Auto:
        return ("auto", "auto", "auto")
    if type(value) is _Never:
        return ("never", "never", "never")
    if type(value) is not Assume:
        raise ValueError("intj: unsupported specialization")
    modes: list[object] = [UNSET, UNSET, UNSET]
    for fact in value.facts:
        detail = _FACT_CLASSES.get(type(fact))
        if detail is None:
            raise TypeError("intj: Assume accepts exact built-in Fact instances")
        field, expected = detail
        actual = getattr(fact, "value")
        if type(actual) is not int or actual != expected:
            raise ValueError(f"intj: unsupported {field} fact")
        modes[_FACT_NAMES.index(field)] = "assume"
    return tuple(modes)  # type: ignore[return-value]


def _type_name(value: object, kind: str, *, element: bool = False) -> str | None:
    if value is None:
        return None
    if type(value) is tl.pointer_type and kind == "argument":
        if value.address_space != 1 or value.const:
            raise ValueError("intj: unsupported pointer type")
        item = _type_name(value.element_ty, kind, element=True)
        if item is None:
            raise ValueError("intj: unsupported pointer element type")
        return "*" + item
    if type(value) is not tl.dtype:
        raise ValueError(f"intj: unsupported {kind} type {value!r}")
    name = cast(str, value.name)
    if name == "int1":
        return "u1"
    if name.startswith("int") and name[3:] in ("8", "16", "32", "64"):
        return "i" + name[3:]
    if name.startswith("uint") and name[4:] in ("8", "16", "32", "64"):
        return "u" + name[4:]
    if name in (("fp32", "fp64") if kind == "argument" else ("fp64",)) or (
        element and name in tl.dtype.FP_TYPES
    ):
        return name
    raise ValueError(f"intj: unsupported {kind} type {name}")


def _canonical_types(value: object, kind: str) -> tuple[str | None, ...] | None:
    if value is UNSET or value is AUTO:
        return None
    choices = value if isinstance(value, tuple) else (value,)
    if any(choice is AUTO for choice in choices):
        raise ValueError("intj: AUTO is not allowed inside a type sequence")
    names = {_type_name(choice, kind) for choice in choices}
    return tuple(sorted(names, key=lambda name: (name is not None, name or "")))


def _inferred_scalar_type(value: object) -> str | None:
    if value is None:
        return None
    if type(value) is bool:
        return "u1"
    if type(value) is float:
        return "fp32"
    if type(value) is int:
        if -(1 << 31) <= value < (1 << 31):
            return "i32"
        if -(1 << 63) <= value < (1 << 63):
            return "i64"
        if 0 <= value < (1 << 64):
            return "u64"
    raise ValueError("intj: baked integer is outside Triton's scalar range")


def _fits_scalar(value: object, name: str | None, kind: str) -> bool:
    if name is None:
        return value is None
    if name.startswith("*"):
        return False
    if name == "u1":
        return type(value) is bool
    if name[0] in ("i", "u") and type(value) is int:
        bits = int(name[1:])
        low = -(1 << (bits - 1)) if name[0] == "i" else 0
        high = (1 << (bits - 1)) if name[0] == "i" else (1 << bits)
        return low <= value < high
    if name in ("fp32", "fp64") and type(value) is float:
        if name == "fp64":
            return True
        try:
            rounded = struct.unpack(">f", struct.pack(">f", value))[0]
        except OverflowError:
            return False
        return kind == "argument" or rounded == value
    return False


def _applicable(name: str | None, field: str) -> bool:
    if name is None:
        return False
    if field == "pointer_range_32":
        return name.startswith("*")
    if field == "equal_to_one":
        return name.startswith(("i", "u")) and name != "u1"
    return name.startswith("*") or (name.startswith(("i", "u")) and name != "u1")


def _resolve_annotations(  # pyright: ignore[reportUnusedFunction]  # consumed by launcher
    jit_func: Any, extra_annotation: Mapping[str, object] | None
) -> tuple[ResolvedParam, ...]:
    extra = extra_annotation or {}
    unknown = set(extra) - {param.name for param in jit_func.params}
    if unknown:
        raise ValueError(f"intj: unknown extra_annotation parameter(s) {sorted(unknown)}")
    resolved: list[ResolvedParam] = []
    for index, param in enumerate(jit_func.params):
        name = param.name
        inline = _annotation_source(param._param.annotation, name)
        other = _annotation_source(extra[name], name) if name in extra else None
        if inline is not None and other is not None and type(inline) is not type(other):
            raise ValueError(f"intj: parameter {name!r} has conflicting kind")
        kind = "constexpr" if type(inline or other) is Constexpr else "argument"
        if kind == "constexpr":
            left = inline if inline is not None else Constexpr()
            right = other if other is not None else Constexpr()
            assert isinstance(left, Constexpr) and isinstance(right, Constexpr)
            raw_type = _merge_field(name, "type", left.type, right.type, kind=kind)
            baked = _merge_field(name, "value", left.value, right.value)
            power = left.power_of_two_or_zero or right.power_of_two_or_zero
            modes = ("auto", "auto", "auto")
            bind_value = None
        else:
            left = inline if inline is not None else Argument()
            right = other if other is not None else Argument()
            assert isinstance(left, Argument) and isinstance(right, Argument)
            raw_type = _merge_field(name, "type", left.type, right.type, kind=kind)
            baked = _merge_field(name, "value", left.value, right.value)
            bind_value = _merge_field(name, "bind_value",
                                      left.bind_value if left.bind_value is not None else UNSET,
                                      right.bind_value if right.bind_value is not None else UNSET)
            if bind_value is UNSET:
                bind_value = None
            if baked is not UNSET and bind_value is not None:
                raise ValueError(f"intj: parameter {name!r} has both value and bind_value")
            left_modes = _specialization_modes(left.specialize)
            right_modes = _specialization_modes(right.specialize)
            modes = tuple(_merge_field(name, field, a, b)
                          for field, a, b in zip(_FACT_NAMES, left_modes, right_modes))
            if param.do_not_specialize:
                modes = tuple(_merge_field(name, field, mode, "never") if i < 2 else mode
                              for i, (field, mode) in enumerate(zip(_FACT_NAMES, modes)))
            if param.do_not_specialize_on_alignment:
                modes = (modes[0], _merge_field(name, "aligned_16", modes[1], "never"), modes[2])
            modes = tuple("auto" if mode is UNSET else mode for mode in modes)
            power = False
        types = _canonical_types(raw_type, kind)
        if power and types is not None and any(
            ty is None or not ty.startswith(("i", "u")) or ty == "u1" for ty in types
        ):
            raise ValueError(f"intj: parameter {name!r} power_of_two_or_zero needs an integer type")
        tag = () if baked is UNSET else _canonical_value(baked)
        if baked is not UNSET:
            if kind == "argument":
                inferred = _inferred_scalar_type(baked)
                if types is None:
                    types = (inferred,)
                elif len(types) != 1:
                    if inferred not in types:
                        raise ValueError(f"intj: parameter {name!r} baked value has no allowed type")
                    types = (inferred,)
                if types[0] is not None and types[0].startswith("*"):
                    raise ValueError(f"intj: parameter {name!r} cannot bake a pointer value")
                if not _fits_scalar(baked, types[0], kind):
                    raise ValueError(f"intj: parameter {name!r} baked value does not fit type")
            elif types is not None:
                matching = [ty for ty in types if _fits_scalar(baked, ty, kind)]
                if not matching:
                    raise ValueError(f"intj: parameter {name!r} baked value does not fit type")
                types = (min(matching, key=lambda ty: (
                    0 if ty is None else 1 if ty == "u1" else
                    64 if ty == "fp64" else int(ty[1:]), ty or ""
                )),)
        if power:
            if baked is not UNSET and (type(baked) is not int or
                                       not (baked == 0 or abs(baked).bit_count() == 1)):
                raise ValueError(f"intj: parameter {name!r} baked value is not a power of two or zero")
        assumed = tuple((field, expected) for field, expected in _FACT_CLASSES.values()
                        if modes[_FACT_NAMES.index(field)] == "assume")
        if kind == "argument":
            branches = types if types is not None else ("i32", "*fp32")
            for field, _ in assumed:
                if not any(_applicable(branch, field) for branch in branches):
                    raise ValueError(f"intj: parameter {name!r} has dead {field} fact")
            for branch in branches:
                if branch is not None and _applicable(branch, "equal_to_one") and \
                   _applicable(branch, "aligned_16") and modes[0] == modes[1] == "assume":
                    raise ValueError(f"intj: parameter {name!r} has impossible {branch} branch")
            if baked is not UNSET:
                assert types is not None
                for field, _ in assumed:
                    if _applicable(types[0], field) and (
                        field == "equal_to_one" and baked != 1 or
                        field == "aligned_16" and cast(int, baked) % 16 != 0
                    ):
                        raise ValueError(f"intj: parameter {name!r} baked value violates {field}")
            if bind_value is BindValue.POINTER and (
                raw_type is UNSET or raw_type is AUTO or isinstance(raw_type, tuple) or
                types is None or len(types) != 1 or types[0] is None or
                not types[0].startswith("*")
            ):
                raise ValueError(f"intj: parameter {name!r} BindValue.POINTER needs exactly one explicit pointer type")
            if bind_value is BindValue.TENSOR and types is not None and (
                len(types) != 1 or types[0] is None or not types[0].startswith("*")
            ):
                raise ValueError(f"intj: parameter {name!r} BindValue.TENSOR needs a pointer type")
            if baked is not UNSET or bind_value is not None:
                if types is not None and len(types) != 1:
                    raise ValueError(f"intj: parameter {name!r} needs one effective type")
                if types is None and bind_value is not BindValue.TENSOR:
                    raise ValueError(f"intj: parameter {name!r} needs one effective type")
                applicable = types if types is not None else ("i32", "*fp32")
                if any(mode == "auto" and any(_applicable(ty, field) for ty in applicable)
                       for field, mode in zip(_FACT_NAMES, modes)):
                    raise ValueError(f"intj: parameter {name!r} has AUTO specialization with a fixed value")
        annotation = CanonicalAnnotation(
            kind, types, cast(str, modes[0]), cast(str, modes[1]), cast(str, modes[2]),
            assumed, power, bind_value.value if isinstance(bind_value, BindValue) else None, tag,
        )
        resolved.append(ResolvedParam(name, index, annotation, baked))
    return tuple(resolved)

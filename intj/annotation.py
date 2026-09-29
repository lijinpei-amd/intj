"""Argument annotations understood by intj launchers."""

from __future__ import annotations

import dataclasses
import enum
import inspect
import struct
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    import triton.language as tl

    INT_TYPES = (tl.int32, tl.int64, tl.uint64)
    FLOAT_TYPES = (tl.float32,)


@dataclasses.dataclass(frozen=True)
class Annotation:
    """A per-parameter annotation: the public base of `Argument` and `Constexpr`."""


@dataclasses.dataclass(frozen=True)
class Specialization:
    """A specialization policy for an `Argument`: `AUTO`, `NEVER` or `Assume`."""


@dataclasses.dataclass(frozen=True)
class _Auto(Specialization):
    """The type of `AUTO`: retain the applicable Triton specialization."""


@dataclasses.dataclass(frozen=True)
class _Never(Specialization):
    """The type of `NEVER`: omit every specialization fact."""


AUTO = _Auto()
NEVER = _Never()


@dataclasses.dataclass(frozen=True)
class Fact:
    """A specialization fact a caller promises in `Assume`."""


@dataclasses.dataclass(frozen=True)
class EqualTo(Fact):
    """The fact that an integer argument equals `value` (only 1)."""

    value: int


@dataclasses.dataclass(frozen=True)
class Aligned(Fact):
    """The fact that an argument is divisible by `value` (only 16)."""

    value: int


@dataclasses.dataclass(frozen=True)
class PointerRange(Fact):
    """The fact that a pointer's storage is under 2 GiB (`value` must be 32).

    `PointerRange(32)` means the whole underlying storage is at most
    2**31 - 1 bytes, not just a tensor view.
    """

    value: int


@dataclasses.dataclass(frozen=True, init=False)
class Assume(Specialization):
    """A specialization that fixes selected facts as caller promises.

    Facts omitted from `Assume` stay `AUTO`.

    Attributes:
        facts: The assumed facts, in the order given.
    """

    facts: tuple[Fact, ...]

    def __init__(self, *facts: Fact) -> None:
        """Builds the assumption from `facts`.

        Raises:
            TypeError: A fact is not an exact `EqualTo`, `Aligned` or
                `PointerRange`, or its value is not an int.
            ValueError: A fact value is not the one supported (`EqualTo(1)`,
                `Aligned(16)`, `PointerRange(32)`).
        """
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
    """The type of `UNSET`: a field left unspecified, distinct from `None`."""


UNSET = _Unset()


class BindValue(enum.Enum):
    """How `Argument(bind_value=...)` binds a parameter's value.

    `TENSOR` binds a tensor (or `None`) and reads its current pointer and
    storage on every call. `POINTER` captures an address once at binding and
    needs one explicit pointer type.
    """

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
    """An annotation of an ordinary (non-constexpr) kernel parameter.

    Every field is optional; `UNSET` fields come from the parameter's other
    annotation source or keep Triton's behavior.

    Attributes:
        type: A dtype, pointer type, `None`, or a nonempty sequence of them as an
            allowlist; `AUTO` or `UNSET` retains Triton's inference.
        specialize: `AUTO`, `NEVER` or `Assume(...)`.
        value: A value baked into the launcher, removing the parameter from
            the public call.
        bind_value: Binds the value later through `bind()`; exclusive with
            `value`.

    Raises:
        TypeError: `specialize` or `bind_value` has the wrong type, or `value`
            is unhashable.
        ValueError: `type` is an empty sequence, or both `value` and
            `bind_value` are given.
    """

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
    """An annotation of a constexpr kernel parameter.

    Attributes:
        type: A dtype, `None`, or a nonempty sequence of them; a typed constexpr
            selects the smallest fitting type.
        power_of_two_or_zero: Promises an integer that is zero or +-2**k, keyed
            in one byte.
        value: A value baked into the launcher, removing the parameter from
            the public call.

    Raises:
        TypeError: `power_of_two_or_zero` is not a bool, or `value` is
            unhashable.
        ValueError: `type` is an empty sequence.
    """

    type: Any = UNSET
    power_of_two_or_zero: bool = False
    value: object = UNSET

    def __post_init__(self) -> None:
        object.__setattr__(self, "type", _normalize_type_field(self.type))
        if not isinstance(cast(object, self.power_of_two_or_zero), bool):
            raise TypeError("intj: power_of_two_or_zero must be bool")
        if self.value is not UNSET:
            hash(self.value)


def __getattr__(name: str) -> Any:
    """Returns `INT_TYPES` or `FLOAT_TYPES`, importing triton on first access."""
    if name not in ("INT_TYPES", "FLOAT_TYPES"):
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import triton.language as tl

    value = (tl.int32, tl.int64, tl.uint64) if name == "INT_TYPES" else (tl.float32,)
    globals()[name] = value
    return value


@dataclasses.dataclass(frozen=True)
class KeyField:
    """One field of a parameter's spec key.

    Attributes:
        kind: What the field holds (`"descriptor"`, `"payload"`, `"exact_kind"`,
            `"exact"`, ...).
        width: The field size in bytes.
        offset: The byte offset in the key, or -1 until the layout places it.
    """

    kind: str
    width: int
    offset: int = -1


class DeviceBinding(str, enum.Enum):
    """Whether a launcher is fixed to one device or keys on the device."""

    FIXED = "fixed"
    NOT_FIXED = "not_fixed"


@dataclasses.dataclass(frozen=True)
class KeyLayout:
    """The placement of every key field in a launcher's spec key.

    Attributes:
        fields: `(parameter index, placed field)` pairs, in the input order.
        device_offset: The byte offset of the device slot, or None when the
            device is fixed.
        nwords: The spec-key length, in uint64 words.
    """

    fields: tuple[tuple[int, KeyField], ...]
    device_offset: int | None
    nwords: int


def _layout_fields(  # pyright: ignore[reportUnusedFunction]  # consumed by launcher
    fields: tuple[tuple[int, KeyField], ...], device_binding: DeviceBinding
) -> KeyLayout:
    """Places `fields` in the key, widest first, then the device slot if keyed."""
    ordered = sorted(enumerate(fields), key=lambda item: -item[1][1].width)
    offsets: dict[int, int] = {}
    offset = 0
    for source_index, (_, field) in ordered:
        offsets[source_index] = offset
        offset += field.width
    device_offset = None
    if device_binding is DeviceBinding.NOT_FIXED:
        device_offset = offset
        offset += 1
    placed = tuple(
        (param_index, dataclasses.replace(field, offset=offsets[source_index]))
        for source_index, (param_index, field) in enumerate(fields)
    )
    return KeyLayout(placed, device_offset, (offset + 7) // 8)


@dataclasses.dataclass(frozen=True)
class CanonicalAnnotation:
    """The merged, validated annotation of one parameter, as keyed and rendered.

    `equal_to_one`, `aligned_16` and `pointer_range_32` each hold `"auto"`,
    `"never"` or `"assume"`.
    """

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
    #: assigned by an autotune/heuristics layer: never passed, never keyed
    tuned: bool = False
    #: a caller var an autotune layer keys on: keyed by its exact value
    exact_key: bool = False
    #: the type comes from a Triton annotation (`x: float`): values convert
    #: the way Triton's launcher extracts them
    triton_typed: bool = False
    #: annotated `torch.Tensor` / `tl.tensor`: only a tensor is accepted, keyed
    #: as an unannotated tensor (Triton ignores these annotations)
    tensor_only: bool = False
    #: annotated `torch.Tensor | None` (or `tl.tensor`): a tensor or None, each
    #: keyed as unannotated; implies `tensor_only`
    none_ok: bool = False


@dataclasses.dataclass(frozen=True)
class ResolvedParam:
    """A kernel parameter with its canonical annotation.

    Attributes:
        name: The parameter name.
        index: The parameter position in the kernel signature.
        annotation: The canonical annotation.
        baked: The baked value, or `UNSET`; excluded from comparison.
    """

    name: str
    index: int
    annotation: CanonicalAnnotation
    baked: object = dataclasses.field(default=UNSET, compare=False, hash=False)


def _canonical_value(value: object) -> tuple[object, ...]:
    """Returns a hashable, JSON-serializable tag for a baked value.

    Raises:
        ValueError: The value is not a dtype, str, `JITFunction` or scalar.
    """
    import triton.language as tl
    from triton.runtime.jit import JITFunction

    if isinstance(value, tl.dtype):
        return ("dtype", value.name)
    if type(value) is str:
        return ("str", value)
    if isinstance(value, JITFunction):
        return ("jit", value.module, value.fn.__qualname__, value.cache_key)
    if value is None:
        return ("none",)
    if type(value) is bool:
        return ("bool", value)
    if type(value) is int:
        return ("int", str(value))
    if type(value) is float:
        return ("float64", struct.pack(">d", value).hex())
    raise ValueError("intj: baked values must be scalar int, float, bool, or None")


_TENSOR_NAMES = ("torch.Tensor", "tl.tensor")
_OPTIONAL_TENSOR_NAMES = frozenset(
    form.format(t)
    for t in _TENSOR_NAMES
    for form in (
        "Optional[{}]",
        "Union[{},None]",
        "Union[None,{}]",
        "{}|None",
        "None|{}",
    )
)


def _tensor_annotation(value: object) -> str | None:
    """Classifies a tensor annotation, given as an object or a postponed string.

    Returns "tensor" for `torch.Tensor` / `tl.tensor`, "optional" for either
    `| None` (`Optional[...]`, `Union[..., None]`), and None otherwise.
    """
    import sys
    import types
    import typing

    import triton.language as tl

    if type(value) is str:
        text = "".join(value.split()).replace("typing.", "")
        if text in _TENSOR_NAMES:
            return "tensor"
        return "optional" if text in _OPTIONAL_TENSOR_NAMES else None
    torch = sys.modules.get("torch")
    classes = (tl.tensor,) if torch is None else (tl.tensor, torch.Tensor)
    if any(value is cls for cls in classes):
        return "tensor"
    # `X | None` is a types.UnionType from Python 3.10 on
    unions = (typing.Union, getattr(types, "UnionType", typing.Union))
    if typing.get_origin(value) in unions:
        args = typing.get_args(value)
        if len(args) == 2 and type(None) in args:
            other = args[0] if args[1] is type(None) else args[1]
            if any(other is cls for cls in classes):
                return "optional"
    return None


def _annotation_source(value: object, name: str) -> Argument | Constexpr | None:
    """Returns the `Argument` or `Constexpr` a raw annotation means, or None.

    Raises:
        ValueError: The annotation is not one intj understands.
    """
    import triton.language as tl

    if value is inspect.Parameter.empty:
        return None
    if type(value) is Argument or type(value) is Constexpr:
        return value
    if value is tl.constexpr:
        return Constexpr()
    if value is int or (type(value) is str and value == "int"):
        return Argument(type=tl.int32)
    if value is None or isinstance(value, tl.dtype):
        return Argument(type=value)
    raise ValueError(f"intj: parameter {name!r} has unsupported annotation {value!r}")


def _merge_field(
    name: str, field: str, left: object, right: object, *, kind: str = "argument"
) -> object:
    """Merges one field from two annotation sources; `UNSET` defers to the other.

    Raises:
        ValueError: Both sources set the field to different values.
    """
    if left is UNSET:
        return right
    if right is UNSET:
        return left
    if field == "type":
        equal = _canonical_types(left, kind) == _canonical_types(right, kind)
    elif field == "value" and left is not UNSET and right is not UNSET:
        equal = type(left) is type(right) and (
            struct.pack(">d", left) == struct.pack(">d", right)
            if type(left) is float
            else left == right
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
    """Returns the per-fact modes of a `specialize` field, in `_FACT_NAMES` order.

    Each mode is `"auto"`, `"never"`, `"assume"` or `UNSET`.
    """
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
    """Returns the canonical key name of a type (`"i32"`, `"*fp16"`, ...).

    Raises:
        ValueError: The type is not supported for `kind`.
    """
    import triton.language as tl

    if value is None:
        return None
    if type(value) is tl.pointer_type and kind == "argument":
        if value.address_space not in (1, "global") or value.const:
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
    """Returns the type Triton infers for a baked scalar.

    Raises:
        ValueError: An integer is outside Triton's scalar range.
    """
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
    """Returns whether `value` is representable as the canonical type `name`."""
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


def _key_fields(  # pyright: ignore[reportUnusedFunction]  # consumed by launcher
    annotation: CanonicalAnnotation,
) -> tuple[KeyField, ...]:
    """Returns the unplaced key fields `annotation` contributes to the spec key."""
    if annotation.tuned or annotation.baked_value or annotation.bind_value is not None:
        return ()
    types = annotation.types
    if annotation.kind == "constexpr":
        if annotation.power_of_two_or_zero:
            return (KeyField("payload", 1),)
        if types == (None,):
            return ()
        descriptor = types is None or len(types) > 1
        width = (
            8
            if types is None
            else max(
                0
                if ty is None
                else 8
                if ty == "fp64"
                else 1
                if ty == "u1"
                else int(ty[1:]) // 8
                for ty in types
            )
        )
        return ((KeyField("descriptor", 1),) if descriptor else ()) + (
            (KeyField("payload", width),) if width else ()
        )
    descriptor = (
        types is None
        or len(types) > 1
        or any(
            mode == "auto" and any(_applicable(ty, field) for ty in types)
            for field, mode in zip(
                _FACT_NAMES,
                (
                    annotation.equal_to_one,
                    annotation.aligned_16,
                    annotation.pointer_range_32,
                ),
            )
        )
    )
    fields = (KeyField("descriptor", 1),) if descriptor else ()
    if annotation.exact_key:
        fields += (KeyField("exact_kind", 1), KeyField("exact", 8))
    return fields


def _resolve_annotations(  # pyright: ignore[reportUnusedFunction]  # consumed by launcher
    jit_func: Any, extra_annotation: Mapping[str, object] | None
) -> tuple[ResolvedParam, ...]:
    """Merges each parameter's inline and extra annotations and validates them.

    Args:
        jit_func: The Triton `JITFunction`.
        extra_annotation: Annotations by parameter name, merged field by field
            with the inline ones.

    Returns:
        One `ResolvedParam` per kernel parameter, in signature order.

    Raises:
        ValueError: An unknown parameter name, an unsupported annotation, or a
            conflicting or impossible combination of fields.
    """
    import triton.language as tl
    from triton.runtime.jit import JITFunction

    extra = extra_annotation or {}
    unknown = set(extra) - {param.name for param in jit_func.params}
    if unknown:
        raise ValueError(
            f"intj: unknown extra_annotation parameter(s) {sorted(unknown)}"
        )
    resolved: list[ResolvedParam] = []
    for index, param in enumerate(jit_func.params):
        name = param.name
        raw = param._param.annotation
        # The type Triton's binder stamps on the argument: `int`, `bool`, `float`,
        # `tl.int64`, `"tl.int64"`, pointer types, ... -- whatever it recognizes.
        triton_type = cast(str, param.annotation_type)
        tensor_kinds: set[str] = set()
        if triton_type:
            inline = Argument(type=tl.str_to_ty(triton_type, None))
        elif (tensor_kind := _tensor_annotation(raw)) is not None:
            inline = None
            tensor_kinds.add(tensor_kind)
        else:
            # Preserve Triton's legacy constexpr strings, including postponed annotations.
            inline = _annotation_source(
                tl.constexpr if isinstance(raw, str) and param.is_constexpr else raw,
                name,
            )
        other = None
        if name in extra and (tensor_kind := _tensor_annotation(extra[name])):
            tensor_kinds.add(tensor_kind)
        elif name in extra:
            other = _annotation_source(extra[name], name)
        if len(tensor_kinds) > 1:
            raise ValueError(
                f"intj: parameter {name!r} has conflicting tensor annotations"
            )
        tensor_only = bool(tensor_kinds)
        none_ok = tensor_kinds == {"optional"}
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
            bind_value = _merge_field(
                name,
                "bind_value",
                left.bind_value if left.bind_value is not None else UNSET,
                right.bind_value if right.bind_value is not None else UNSET,
            )
            if bind_value is UNSET:
                bind_value = None
            if baked is not UNSET and bind_value is not None:
                raise ValueError(
                    f"intj: parameter {name!r} has both value and bind_value"
                )
            left_modes = _specialization_modes(left.specialize)
            right_modes = _specialization_modes(right.specialize)
            modes = tuple(
                _merge_field(name, field, a, b)
                for field, a, b in zip(_FACT_NAMES, left_modes, right_modes)
            )
            if param.do_not_specialize:
                # Triton's specialize=False suppresses the pointer-range attribute too.
                modes = tuple(
                    _merge_field(name, field, mode, "never")
                    for field, mode in zip(_FACT_NAMES, modes)
                )
            if param.do_not_specialize_on_alignment:
                modes = (
                    modes[0],
                    _merge_field(name, "aligned_16", modes[1], "never"),
                    modes[2],
                )
            modes = tuple("auto" if mode is UNSET else mode for mode in modes)
            if triton_type and modes[0] == "auto":
                # Triton's binder replaces the specialized type of a type-annotated
                # parameter with the annotation: `("i32",) + specialize_impl(...)[1:]`
                # keeps divisibility but never folds 1 into a constexpr.
                modes = ("never",) + modes[1:]
            power = False
        types = _canonical_types(raw_type, kind)
        if tensor_only and (
            kind != "argument"
            or baked is not UNSET
            or bind_value is not None
            or types is not None
            and any(ty is None or not ty.startswith("*") for ty in types)
        ):
            raise ValueError(
                f"intj: parameter {name!r} is annotated as a tensor; it cannot also "
                "be a constexpr, a non-pointer type, baked or bound"
            )
        if (
            power
            and types is not None
            and any(
                ty is None or not ty.startswith(("i", "u")) or ty == "u1"
                for ty in types
            )
        ):
            raise ValueError(
                f"intj: parameter {name!r} power_of_two_or_zero needs an integer type"
            )
        tag = () if baked is UNSET else _canonical_value(baked)
        if baked is not UNSET:
            if (
                kind == "constexpr"
                and types is None
                and not isinstance(baked, (tl.dtype, JITFunction))
                and type(baked) is not str
            ):
                _inferred_scalar_type(baked)  # Enforce Triton's scalar integer domain.
            if kind == "argument":
                inferred = _inferred_scalar_type(baked)
                if types is None:
                    types = (inferred,)
                elif len(types) != 1:
                    if inferred not in types:
                        raise ValueError(
                            f"intj: parameter {name!r} baked value has no allowed type"
                        )
                    types = (inferred,)
                if types[0] is not None and types[0].startswith("*"):
                    raise ValueError(
                        f"intj: parameter {name!r} cannot bake a pointer value"
                    )
                if not _fits_scalar(baked, types[0], kind):
                    raise ValueError(
                        f"intj: parameter {name!r} baked value does not fit type"
                    )
            elif types is not None:
                matching = [ty for ty in types if _fits_scalar(baked, ty, kind)]
                if not matching:
                    raise ValueError(
                        f"intj: parameter {name!r} baked value does not fit type"
                    )
                types = (
                    min(
                        matching,
                        key=lambda ty: (
                            0
                            if ty is None
                            else 1
                            if ty == "u1"
                            else 64
                            if ty == "fp64"
                            else int(ty[1:]),
                            ty or "",
                        ),
                    ),
                )
        if power:
            if baked is not UNSET and (
                type(baked) is not int
                or not (baked == 0 or abs(baked).bit_count() == 1)
            ):
                raise ValueError(
                    f"intj: parameter {name!r} baked value is not a power of two or zero"
                )
        assumed = tuple(
            (field, expected)
            for field, expected in _FACT_CLASSES.values()
            if modes[_FACT_NAMES.index(field)] == "assume"
        )
        if kind == "argument":
            branches = types if types is not None else ("i32", "*fp32")
            for field, _ in assumed:
                if not any(_applicable(branch, field) for branch in branches):
                    raise ValueError(f"intj: parameter {name!r} has dead {field} fact")
            for branch in branches:
                if (
                    branch is not None
                    and _applicable(branch, "equal_to_one")
                    and _applicable(branch, "aligned_16")
                    and modes[0] == modes[1] == "assume"
                ):
                    raise ValueError(
                        f"intj: parameter {name!r} has impossible {branch} branch"
                    )
            if baked is not UNSET:
                assert types is not None
                for field, _ in assumed:
                    if _applicable(types[0], field) and (
                        field == "equal_to_one"
                        and baked != 1
                        or field == "aligned_16"
                        and cast(int, baked) % 16 != 0
                    ):
                        raise ValueError(
                            f"intj: parameter {name!r} baked value violates {field}"
                        )
            if bind_value is BindValue.POINTER and (
                raw_type is UNSET
                or raw_type is AUTO
                or isinstance(raw_type, tuple)
                or types is None
                or len(types) != 1
                or types[0] is None
                or not types[0].startswith("*")
            ):
                raise ValueError(
                    f"intj: parameter {name!r} BindValue.POINTER needs exactly one explicit pointer type"
                )
            if (
                bind_value is BindValue.TENSOR
                and types is not None
                and (
                    len(types) != 1
                    or types[0] is not None
                    and not types[0].startswith("*")
                )
            ):
                raise ValueError(
                    f"intj: parameter {name!r} BindValue.TENSOR needs a pointer type or None"
                )
            if baked is not UNSET or bind_value is not None:
                if types is not None and len(types) != 1:
                    raise ValueError(
                        f"intj: parameter {name!r} needs one effective type"
                    )
                if types is None and bind_value is not BindValue.TENSOR:
                    raise ValueError(
                        f"intj: parameter {name!r} needs one effective type"
                    )
                applicable = types if types is not None else ("i32", "*fp32")
                if any(
                    mode == "auto" and any(_applicable(ty, field) for ty in applicable)
                    for field, mode in zip(_FACT_NAMES, modes)
                ):
                    raise ValueError(
                        f"intj: parameter {name!r} has AUTO specialization with a fixed value"
                    )
        annotation = CanonicalAnnotation(
            kind,
            types,
            cast(str, modes[0]),
            cast(str, modes[1]),
            cast(str, modes[2]),
            assumed,
            power,
            bind_value.value if isinstance(bind_value, BindValue) else None,
            tag,
            triton_typed=bool(triton_type) and kind == "argument",
            tensor_only=tensor_only,
            none_ok=none_ok,
        )
        resolved.append(ResolvedParam(name, index, annotation, baked))
    return tuple(resolved)

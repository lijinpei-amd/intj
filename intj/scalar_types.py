"""Triton's scalar type names, as intj keys, checks and decodes them.

Every question intj asks of a canonical name (`"u1"`, `"i32"`, `"fp64"`) -- its
kind, its width, whether a value fits it -- is answered here, from one table.
A pointer name is `"*"` plus its element's name; `None` is the type of None.
"""

from __future__ import annotations

import dataclasses
import struct

from . import _imports


@dataclasses.dataclass(frozen=True)
class Scalar:
    """One scalar type a launcher argument or constexpr can have.

    Attributes:
        name: Triton's canonical name.
        kind: `"bool"`, `"signed"`, `"unsigned"` or `"float"`.
        bits: Its width.
    """

    name: str
    kind: str
    bits: int


#: Every scalar name an annotation can resolve to, by name.
SCALARS = {
    s.name: s
    for s in (
        Scalar("u1", "bool", 1),
        *(Scalar(f"i{bits}", "signed", bits) for bits in (8, 16, 32, 64)),
        *(Scalar(f"u{bits}", "unsigned", bits) for bits in (8, 16, 32, 64)),
        Scalar("fp32", "float", 32),
        Scalar("fp64", "float", 64),
    )
}


def is_integer(name: str | None) -> bool:
    """Returns whether `name` is a signed or unsigned integer; `u1` is not."""
    scalar = SCALARS.get(name or "")
    return scalar is not None and scalar.kind in ("signed", "unsigned")


def is_float(name: str | None) -> bool:
    scalar = SCALARS.get(name or "")
    return scalar is not None and scalar.kind == "float"


def key_bytes(name: str | None) -> int:
    """Returns the bytes a constexpr of type `name` keys its value in; 0 for None."""
    if name is None:
        return 0
    return max(1, SCALARS[name].bits // 8)


def narrowness(name: str | None) -> tuple[int, str]:
    """Narrowest first, so the first type a value fits is the one it keys as.

    Only `u1`, the integers and `fp64` have a width here: `fp32`, pointers and
    None sort as width 0, by name. Rendered modules test types in this order,
    and a module's digest does not hash the order itself, so changing it would
    let a cached module disagree with a new render.
    """
    if name is None:
        return (0, "")
    scalar = SCALARS.get(name)
    width = (
        scalar.bits
        if scalar is not None and (scalar.kind != "float" or scalar.bits == 64)
        else 0
    )
    return (width, name)


def fits(value: object, name: str | None, kind: str) -> bool:
    """Returns whether `value` is representable as the canonical type `name`.

    `kind` is the parameter's: an argument's fp32 takes any float, a
    constexpr's only one fp32 represents exactly.
    """
    if name is None:
        return value is None
    scalar = SCALARS.get(name)
    if scalar is None:
        return False
    if scalar.kind == "bool":
        return type(value) is bool
    if is_integer(name) and type(value) is int:
        bits = scalar.bits
        low = -(1 << (bits - 1)) if scalar.kind == "signed" else 0
        high = (1 << (bits - 1)) if scalar.kind == "signed" else (1 << bits)
        return low <= value < high
    if scalar.kind == "float" and type(value) is float:
        if scalar.bits == 64:
            return True
        try:
            rounded = struct.unpack(">f", struct.pack(">f", value))[0]
        except OverflowError:
            return False
        return kind == "argument" or rounded == value
    return False


def triton_name(dtype: object) -> str | None:
    """Returns Triton's name for a torch dtype (`torch.float16` -> `"fp16"`), or None."""
    _imports.ensure_triton_imported()
    # `str(dtype)` is what triton's own canonicalize_dtype splits; the attribute
    # name is not
    return _imports.type_canonicalisation_dict.get(str(dtype).split(".")[-1])

"""Triton's scalar type names, as intj keys, checks and decodes them.

Every question intj asks of a canonical name (`"u1"`, `"i32"`, `"fp64"`,
`"*fp16"`) -- its kind, its width, whether a value fits it, which
specialization facts it can hold -- is answered here, from one table. A
pointer name is `"*"` plus its element's name; `None` is the type of None.
"""

from __future__ import annotations

import dataclasses
import math

from . import _imports

_FLT_MAX = float.fromhex("0x1.fffffep+127")


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


def is_pointer(name: str | None) -> bool:
    return name is not None and name.startswith("*")


def fact_applies(name: str | None, fact: str) -> bool:
    """Returns whether a parameter of type `name` can hold the fact `fact`.

    `equal_to_one` applies to integers, `aligned_16` to integers and pointers,
    `pointer_range_32` to pointers; none to `u1`, a float or None.
    """
    if fact == "pointer_range_32":
        return is_pointer(name)
    if fact == "equal_to_one":
        return is_integer(name)
    return is_pointer(name) or is_integer(name)


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


def fits(value: object, name: str | None) -> bool:
    """Returns whether `value` is in the range of the canonical type `name`.

    An fp32 (only an argument has one) takes any float but a finite one
    beyond fp32's largest, as `verify_annotation`'s range check does; the
    rest round.
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
        return scalar.bits == 64 or not math.isfinite(value) or abs(value) <= _FLT_MAX
    return False


def inferred_type(value: object) -> str | None:
    """Returns the type Triton infers for a runtime scalar.

    None, `u1`, `fp32` (any float, as Triton's launcher converts it), or the
    first of `i32`, `i64` and `u64` an int fits.

    Raises:
        ValueError: `value` is not a scalar, or an int beyond 64 bits.
    """
    if value is None:
        return None
    if type(value) is bool:
        return "u1"
    if type(value) is float:
        return "fp32"
    for name in ("i32", "i64", "u64"):
        if fits(value, name):
            return name
    raise ValueError("intj: baked integer is outside Triton's scalar range")


def triton_name(dtype: object) -> str | None:
    """Returns Triton's name for a torch dtype (`torch.float16` -> `"fp16"`), or None."""
    _imports.ensure_triton_imported()
    # `str(dtype)` is what triton's own canonicalize_dtype splits; the attribute
    # name is not
    return _imports.type_canonicalisation_dict.get(str(dtype).split(".")[-1])

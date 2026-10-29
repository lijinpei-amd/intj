"""The spec key, and the compiler input it must never be coarser than.

Both sides of the invariant "same intj spec key => same triton
specialization" live here: which key fields a parameter contributes and where
the key places them (`key_layout`), how the public call numbers the
parameters (`call_layout`), and what Triton is given for one call
(`compiler_input`).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from typing import Any

from . import _imports
from .annotation import (
    _FACT_NAMES,  # pyright: ignore[reportPrivateUsage]  # canonical specialization facts
    CacheKeyField,
    CanonicalAnnotation,
    DeviceBinding,
    ResolvedParam,
    _applicable,  # pyright: ignore[reportPrivateUsage]  # canonical specialization facts
    _canonical_value,  # pyright: ignore[reportPrivateUsage]  # tagged compiler constants
)
from .render.key import DynamicSlot, Param

# triton ships no type information
JitFunction = Any


def _cache_key_fields(
    annotation: CanonicalAnnotation,
) -> tuple[CacheKeyField, ...]:
    """Returns the unplaced key fields `annotation` contributes to the spec key."""
    if annotation.tuned or annotation.baked_value or annotation.bind_value is not None:
        return ()
    types = annotation.types
    if annotation.kind == "constexpr":
        if annotation.power_of_two_or_zero:
            return (CacheKeyField("payload", 1),)
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
        return ((CacheKeyField("descriptor", 1),) if descriptor else ()) + (
            (CacheKeyField("payload", width),) if width else ()
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
    fields = (CacheKeyField("descriptor", 1),) if descriptor else ()
    if annotation.exact_key:
        fields += (CacheKeyField("exact_kind", 1), CacheKeyField("exact", 8))
    return fields


def _dynamic_key_fields(name: str) -> tuple[CacheKeyField, ...]:
    """Returns one dynamic value's key fields: a kind byte and 8 value bytes.

    Laid out like an exact key.  `name` picks nothing here; the invariant
    tests' mutation checks drop one value's fields by name.
    """
    del name
    return (CacheKeyField("exact_kind", 1), CacheKeyField("exact", 8))


def _object_capable(annotation: CanonicalAnnotation) -> bool:
    """Returns whether a public parameter with `annotation` takes objects.

    A constexpr keyed by descriptor + 8-byte payload does: str, tl.dtype and
    JIT functions key as INTJ_B_CX_OBJECT plus an id.
    """
    return (
        annotation.kind == "constexpr"
        and annotation.types is None
        and not annotation.power_of_two_or_zero
    )


@dataclasses.dataclass(frozen=True)
class _Placement:
    """Where `_layout_fields` put each key field.

    Attributes:
        fields: `(parameter index, placed field)` pairs, in the input order.
        device_offset: The byte offset of the device slot, or None when the
            device is fixed.
        nwords: The spec-key length, in uint64 words.
    """

    fields: tuple[tuple[int, CacheKeyField], ...]
    device_offset: int | None
    nwords: int


def _layout_fields(
    fields: tuple[tuple[int, CacheKeyField], ...], device_binding: DeviceBinding
) -> _Placement:
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
    return _Placement(placed, device_offset, (offset + 7) // 8)


def call_layout(resolved: Sequence[ResolvedParam]) -> tuple[Param, ...]:
    """Numbers the parameters in the public call and gives each object one a memo.

    Places no key field: `key_layout` does, once the whole key is known.
    """
    params: list[Param] = []
    call_index = 0
    memo = 0
    for p in resolved:
        annotation = p.annotation
        public = (
            not annotation.baked_value
            and annotation.bind_value is None
            and not annotation.tuned
        )
        slot = None
        if public and _object_capable(annotation):
            slot, memo = memo, memo + 1
        params.append(
            Param(p.name, p.index, call_index if public else None, annotation, slot)
        )
        if public:
            call_index += 1
    return tuple(params)


@dataclasses.dataclass(frozen=True)
class KeyLayout:
    """The placement of every field in a launcher's spec key.

    Attributes:
        params: The call layout's parameters, their key fields placed.
        device_offset: The byte offset of the device slot, or None when the
            device is fixed.
        nwords: The spec-key length, in uint64 words.
        computed_fields: Each level-0 computed key's (value, kind) byte offsets.
        dynamic: The dynamic values' slots, in call order.
    """

    params: tuple[Param, ...]
    device_offset: int | None
    nwords: int
    computed_fields: tuple[tuple[int, int], ...]
    dynamic: tuple[DynamicSlot, ...]


def key_layout(
    call: Sequence[Param],
    device_binding: DeviceBinding,
    computed0: int = 0,
    dynamic: Sequence[str] = (),
) -> KeyLayout:
    """Places the key fields of `call`, its level-0 computed keys and dynamic values.

    A computed key is a (value, kind) pair: payload and descriptor, the way a
    constexpr is keyed, so True and 1 key apart.  A dynamic value is keyed
    like an exact key, and its memo follows the parameters'.
    """
    fields = tuple(
        (p.index, field) for p in call for field in _cache_key_fields(p.annotation)
    )
    fields += tuple(
        (-1 - j, CacheKeyField(kind, width))
        for j in range(computed0)
        for kind, width in (("payload", 8), ("descriptor", 1))
    )
    fields += tuple(
        (-1 - computed0 - j, field)
        for j, name in enumerate(dynamic)
        for field in _dynamic_key_fields(name)
    )
    placement = _layout_fields(fields, device_binding)
    params = tuple(
        dataclasses.replace(
            p,
            annotation=dataclasses.replace(
                p.annotation,
                key_fields=tuple(
                    sorted(
                        (f for index, f in placement.fields if index == p.index),
                        key=lambda f: f.offset,
                    )
                ),
            ),
        )
        for p in call
    )
    placed = {
        (index, field.kind): field.offset
        for index, field in placement.fields
        if index < 0
    }
    computed_fields = tuple(
        (placed[(-1 - j, "payload")], placed[(-1 - j, "descriptor")])
        for j in range(computed0)
    )
    memo = sum(p.memo is not None for p in call)
    slots = tuple(
        DynamicSlot(
            name,
            placed.get((-1 - computed0 - j, "exact")),
            placed.get((-1 - computed0 - j, "exact_kind")),
            memo + j,
        )
        for j, name in enumerate(dynamic)
    )
    return KeyLayout(
        params, placement.device_offset, placement.nwords, computed_fields, slots
    )


@dataclasses.dataclass(frozen=True)
class CompilerInput:
    """The final annotated ASTSource identity, with its original Python constants.

    `attrs` holds each parameter's facts; `bare` the parameters Triton's launch
    lists in its attrs with no facts (`(i,): []`). Both enter
    `ASTSource.hash()`, so the source hashes as Triton's own does, but an empty
    entry never changes the binary: only `attrs` is compared. A Triton-typed
    integer is listed for 3 and not for 1, under one intj key.
    """

    signature: tuple[tuple[str, str], ...]
    constants: tuple[tuple[tuple[int, ...], tuple[object, ...]], ...]
    attrs: tuple[tuple[tuple[int, ...], tuple[tuple[str, int], ...]], ...]
    values: tuple[tuple[tuple[int, ...], object], ...] = dataclasses.field(
        compare=False, hash=False, repr=False
    )
    bare: tuple[tuple[int, ...], ...] = dataclasses.field(
        default=(), compare=False, hash=False
    )

    def ast_source(self, jit_func: JitFunction) -> Any:
        """Returns the triton `ASTSource` (`GluonASTSource` for Gluon) to compile.

        Its attrs list parameters in Triton's order, empty entries included.
        """
        _imports.ensure_triton_imported()
        source_type = _imports.ASTSource
        if jit_func.is_gluon():
            # not cached with the rest: importing gluon costs ~35 ms
            from triton.experimental.gluon._runtime import GluonASTSource

            source_type = GluonASTSource
        entries = [
            *((path, [list(attr) for attr in attrs]) for path, attrs in self.attrs),
            *((path, []) for path in self.bare),
        ]
        return source_type(
            jit_func,
            dict(self.signature),
            dict(self.values),
            dict(sorted(entries, key=lambda entry: entry[0])),
        )


def _triton_specialize(
    value: object, *, backend: Any, is_const: bool, specialize: bool, align: bool
) -> tuple[str, Any]:
    _imports.ensure_triton_imported()
    return _imports.native_specialize_impl(backend, value, is_const, specialize, align)


def compiler_input(
    jit_func: JitFunction,
    params: Sequence[Param],
    public_args: Sequence[object],
    backend: Any,
    *,
    baked_values: Mapping[int, object] | None = None,
) -> CompilerInput:
    """Returns the annotated compiler input triton would build for one call.

    `public_args` are the call's public arguments; bound parameters count as
    None and fixed ones come from `baked_values`. Annotated facts apply as
    declared; `auto` ones are specialized the way triton does.
    """
    _imports.ensure_triton_imported()
    signature: list[tuple[str, str]] = []
    constants: list[tuple[tuple[int, ...], tuple[object, ...]]] = []
    attrs: list[tuple[tuple[int, ...], tuple[tuple[str, int], ...]]] = []
    values: list[tuple[tuple[int, ...], object]] = []
    bare: list[tuple[int, ...]] = []
    for param in params:
        annotation = param.annotation
        if annotation.bind_value is not None:
            # Bound type/facts are fixed in this module. Only inferred None
            # becomes a compiler constant; no callback owns a bound value.
            value = None
        elif param.call_index is None:
            assert baked_values is not None, (
                "fixed values must accompany their canonical annotations"
            )
            value = baked_values[param.index]
        else:
            value = public_args[param.call_index]
        ty = "constexpr"
        desc = ""
        if annotation.kind == "argument":
            fixed_type = annotation.types is not None and len(annotation.types) == 1
            effective = annotation.types[0] if fixed_type and annotation.types else None
            modes = (
                ("equal_to_one", annotation.equal_to_one),
                ("aligned_16", annotation.aligned_16),
                ("pointer_range_32", annotation.pointer_range_32),
            )
            assumed_one = annotation.equal_to_one == "assume" and _applicable(
                effective, "equal_to_one"
            )
            inferred_desc = None
            if not assumed_one and (
                not fixed_type
                or any(
                    mode == "auto" and _applicable(effective, field)
                    for field, mode in modes
                )
            ):
                inference_value = value
                if (
                    effective is not None
                    and effective.startswith("*")
                    and (value is None or type(value) is int and value == 0)
                ):
                    # Both public null spellings have pointer 0 and storage range 0.
                    inference_value = _imports.MockTensor(effective[1:])
                inferred, inferred_desc = _triton_specialize(
                    inference_value,
                    backend=backend,
                    is_const=jit_func.params[param.index].is_const,
                    specialize=any(mode == "auto" for _, mode in modes),
                    align=annotation.aligned_16 == "auto",
                )
                if not fixed_type:
                    # The native primitive folds only None and integer 1.
                    effective = (
                        (None if value is None else "i32")
                        if inferred == "constexpr"
                        else inferred
                    )
            ty = effective or "constexpr"
            if _applicable(effective, "equal_to_one") and (
                annotation.equal_to_one == "assume"
                or annotation.equal_to_one == "auto"
                and value == 1
            ):
                ty = "constexpr"
                if annotation.equal_to_one == "assume":
                    value = 1
            if ty != "constexpr":
                desc = "".join(
                    char
                    for field, mode, char in (
                        ("aligned_16", annotation.aligned_16, "D"),
                        ("pointer_range_32", annotation.pointer_range_32, "S"),
                    )
                    if _applicable(effective, field)
                    and (
                        mode == "assume"
                        or mode == "auto"
                        and isinstance(inferred_desc, str)
                        and char in inferred_desc
                    )
                )
        path = (param.index,)
        signature.append((param.name, ty))
        parsed: tuple[tuple[str, int], ...] = ()
        if ty == "constexpr":
            constants.append((path, _canonical_value(value)))
            values.append((path, value))
        elif desc:
            parsed = tuple((name, amount) for name, amount in backend.parse_attr(desc))
        if parsed:
            attrs.append((path, parsed))
        elif ty == "constexpr" or _lists_attrs(
            jit_func.params[param.index], annotation, ty, value
        ):
            bare.append(path)
    return CompilerInput(
        tuple(signature), tuple(constants), tuple(attrs), tuple(values), tuple(bare)
    )


def _lists_attrs(
    jit_param: Any, annotation: CanonicalAnnotation, ty: str, value: object
) -> bool:
    # Triton lists a runtime parameter whose specialize_impl key is a str: not
    # under do_not_specialize, nor a bool or a float. A Triton-typed one keeps
    # the key of its value, which is the int 1 for 1 and None for a bool or None.
    if jit_param.do_not_specialize or not _applicable(ty, "aligned_16"):
        return False
    if annotation.triton_typed and annotation.bind_value is None:
        return not (
            value is None or type(value) is bool or type(value) is int and value == 1
        )
    return True

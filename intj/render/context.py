"""`RenderContext`: what `entry.c.jinja` renders from, derived from a `RenderKey`.

The split is fixed: `RenderContext` computes every value the template prints
or branches on (counts, offsets, slots, flags, which decoder and which checks a
parameter gets), and contains no C.  The template writes all the C, branching
on those values and interpolating them, and computes nothing.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from pathlib import Path

from ..annotation import CanonicalAnnotation, DeviceBinding, KeyField
from .key import DynamicSlot, Param, RenderKey, TuningRender

ENTRY_TEMPLATE = Path(__file__).parent / "entry.c.jinja"


def render(key: RenderKey) -> str:
    """Returns the C source of the module for `key`."""
    import jinja2

    context = RenderContext.from_key(key)
    template = jinja2.Template(
        ENTRY_TEMPLATE.read_text(), undefined=jinja2.StrictUndefined
    )
    # shallow, so the views keep their attributes
    return template.render(
        {f.name: getattr(context, f.name) for f in dataclasses.fields(context)}
    )


@dataclasses.dataclass(frozen=True)
class KeyWrite:
    """Where one key field is or-ed in: a word of the key, and a shift into it."""

    word: int  # key_word_<word>; key_part_<word> on the auto path
    bits: int  # the field's width
    shift: int  # its bit position in that word

    @classmethod
    def into_word(cls, offset: int, width: int) -> KeyWrite:
        return cls(offset // 8, width * 8, offset % 8 * 8)

    @classmethod
    def into_part(cls, offset: int) -> KeyWrite:
        """A one-byte field on the auto path, which packs the key in 32-bit parts."""
        return cls(offset // 4, 8, offset % 4 * 8)


@dataclasses.dataclass(frozen=True)
class AutoPart:
    """A 32-bit key part on the auto path, or-ed into its word at the end."""

    part: int
    word: int
    shift: int


@dataclasses.dataclass(frozen=True)
class TypeCode:
    """One Triton type name, classified for the C that tests and encodes it."""

    #: "none", "pointer", "bool", "float" (fp32 and fp64 only), "signed", or
    #: "unsigned" -- which is also what any other float name falls under
    kind: str
    name: str  # upper case: the INTJ_B_<name> constant
    width: str  # the name's digits: INT<width>_MIN
    #: a pointer's live torch ScalarType code; None if no live dtype has it
    dtype: int | None = None


@dataclasses.dataclass(frozen=True)
class Check:
    """One `verify_annotation` check on a decoded value."""

    test: str  # which C test; `check_expr` in entry.c.jinja spells each
    message: str  # what a failure says the argument violates
    error: str  # the Python exception it raises
    type: TypeCode | None = None  # the type a type test tests
    allowed: tuple[TypeCode, ...] = ()  # "allowlist": the types it may be


@dataclasses.dataclass(frozen=True)
class ParamView:
    """One parameter, with every decision its decode, checks and key need."""

    name: str
    index: int
    call_index: int | None
    memo: int | None
    annotation: CanonicalAnnotation
    constexpr: bool
    #: `intj_pack` decodes it: tuned values and baked constexprs are never seen
    decoded: bool
    #: which reader decodes it on the generic path: "baked", "bound_pointer",
    #: "bound_tensor", "constexpr_memo", "constexpr", "tensor", "typed_float",
    #: "typed_int" or "argument"
    decoder: str
    #: a scalar Triton annotation: the value converts the way Triton's
    #: launcher extracts it
    typed: str | None
    typed_unsigned: bool
    #: baked arguments: the value's INTJ_VALUE_<kind>, and its bits as written
    baked_tag: str | None
    baked_kind: str
    baked_bits: str
    baked_negative: bool
    range_check: bool  # the decoder reads the storage size for the range bit
    bound_range_check: bool
    #: where its type code comes from: "constexpr" (the value's own),
    #: "choices", "fixed" or "infer"
    type_from: str
    fixed_type: TypeCode | None
    type_choices: tuple[TypeCode, ...]  # narrowest first: the first that fits wins
    checks: tuple[Check, ...]
    bound_checks: tuple[Check, ...]  # a bound pointer's, run at bind time
    one_code: bool  # an integer can key as INTJ_B_ONE
    one_tests_value: bool  # ... and only does if it is 1
    align_code: bool  # keys its 16-byte divisibility
    range_code: bool  # keys a tensor's 2 GiB range
    descriptor: KeyWrite | None
    descriptor_part: KeyWrite | None  # the descriptor on the auto path
    payload: KeyWrite | None
    exact: KeyWrite | None
    exact_kind: KeyWrite | None
    bound_slot: int | None  # its index among the bound parameters
    init_arg: int | None  # its argument of _init
    hidden_slot: int | None  # grid_py: its index in the fixed-value tuple
    meta_slot: int | None  # tuned grid_py: its index in a record's meta


@dataclasses.dataclass(frozen=True)
class DynamicView:
    """One `dynamic_options` value, and where its kind and bits are keyed."""

    name: str
    memo: int
    kind: KeyWrite | None
    value: KeyWrite | None


@dataclasses.dataclass(frozen=True)
class ComputedWrite:
    """A level-0 computed key: (value, kind) from `comp`, into the key."""

    value_word: int
    value_slot: int
    kind: KeyWrite
    kind_slot: int


@dataclasses.dataclass(frozen=True)
class Level:
    """One lookup of a launch: level 0 is the spec key, each later one a computed key."""

    n: int
    final: bool  # its records are `intj_final`, not a child map
    deps: int  # dependent values stored in its records
    computed: int  # computed keys its lookup keys on
    dep_at: int  # its first slot in `dep`, two per value
    comp_offset: int  # its first computed key
    comp_at: int  # its first slot in `comp`, two per key
    dep_len: int  # its records' `dep` array, never empty
    key_len: int  # its map's key, in int64 slots
    parent: int
    child: Level | None


@dataclasses.dataclass(frozen=True)
class RenderContext:
    """Every value `entry.c.jinja` reads.

    Fixed fields on purpose: a typo fails here instead of silently rendering an
    empty value into C (the template also runs with `StrictUndefined`).
    """

    # From the key as they are.
    module_name: str
    kernel_repr: str
    nwords: int  # spec-key length, in uint64 words
    spec_pointer_range: int
    driver_path: str
    launch_symbol: str
    device_symbol: str
    error_symbol: str
    error_style: str
    torch_access_mode: str
    python_version: tuple[int, int, int]
    free_threaded: bool
    cpython_static_compile_header: str
    grid_arg: int | None
    grid_py_mode: bool
    grid_cpp_source: str
    return_compiled: bool
    ndebug: bool
    tuning: TuningRender | None

    # Derived.
    params: tuple[ParamView, ...]
    bound_params: tuple[ParamView, ...]
    dynamic: tuple[DynamicView, ...]
    fixed_device: bool
    tuned: bool
    tuned_meta: bool  # records hold the tuned values, for grid_py
    tuned_level0: bool  # level 0 keys computed values too
    #: "dims" (grid_arg), "cpp", "py" or "object" (one grid argument)
    grid_mode: str
    #: how a call gets its grid: as `grid_mode`, or "deferred" -- after a tuned
    #: lookup, since the grid may read tuned values
    grid_eval: str
    grid_object: bool  # needs `intj_grid_object`
    grid_count: int  # grid arguments of a call
    lead_args: int  # device and stream
    stream_arg: int
    ncontrols: int
    grid_py_offset: int
    init_extra: int  # _init's arguments before the bound values
    launch_signature: str
    controls_doc: str
    torch_access_name: str
    guarded: bool  # can recurse into Python, so takes the recursion guard
    launch_guard: bool  # a zero-volume grid reaches the launch
    traverse_cache: bool  # GC visits the kernel cache
    traverse_fixed: bool  # GC visits the fixed kernel
    auto_decode: bool  # the inlined, unannotated decode
    auto_parts: tuple[AutoPart, ...]
    device_write: KeyWrite | None
    computed_writes: tuple[ComputedWrite, ...]
    nbound: int
    nhidden: int
    nmemo: int
    nparams: int
    ndyn: int
    nslots: int
    key_words: int  # the key array, never empty
    key_bytes: int
    param_names_len: int
    levels: tuple[Level, ...]
    nlevels: int
    inner_levels: tuple[Level, ...]  # all but the final, last first
    later_levels: tuple[Level, ...]  # all but level 0
    ndeps: int
    ncomp: int
    dep_len: int
    comp_len: int
    computed_names_len: int
    computed_names: tuple[str, ...]  # every level's computed keys, for errors
    nmeta: int

    @classmethod
    def from_key(cls, key: RenderKey) -> RenderContext:
        """Derives every value the template reads from `key`."""
        tuning = key.tuning
        tuned = tuning is not None
        fixed_device = key.device_binding == DeviceBinding.FIXED
        grid_mode = (
            "cpp"
            if key.grid_cpp_source
            else "py"
            if key.grid_py_mode
            else "object"
            if key.grid_arg is None
            else "dims"
        )
        grid_count = {
            "cpp": len(key.grid_extra),
            "py": 0,
            "object": 1,
            "dims": key.grid_arg or 0,
        }[grid_mode]
        lead_args = 1 if fixed_device else 2
        grid_py_offset = 2 if key.grid_py_mode else 0
        init_extra = (1 if fixed_device else 0) + grid_py_offset
        # ponytail: exact keys need intj_decoded's kind/bits; add them to the auto
        # macros if a tuned launch shows up in profiles
        auto_decode = tuning is None and not any(
            _needs_generic_decode(p.annotation) for p in key.params
        )
        codes: dict[str, int] = {}
        for name, code in key.pointer_types:
            codes.setdefault(name, code)
        bound = [p for p in key.params if p.annotation.bind_value is not None]
        hidden = [
            p for p in key.params if p.call_index is None and not p.annotation.tuned
        ]
        meta_names = tuning.meta_names if tuning is not None else ()
        params = tuple(
            _param_view(
                p,
                verify=key.verify_annotation,
                spec_pointer_range=bool(key.spec_pointer_range),
                codes=codes,
                bound_slot=bound.index(p) if p in bound else None,
                init_extra=init_extra,
                hidden_slot=hidden.index(p) if p in hidden else None,
                meta_slot=meta_names.index(p.name) if p.name in meta_names else None,
            )
            for p in key.params
        )
        levels = _levels(tuning)
        ndeps = sum(level.deps for level in levels)
        ncomp = sum(level.computed for level in levels)
        tuned_meta = tuned and key.grid_py_mode
        return cls(
            module_name=key.module_name,
            kernel_repr=key.kernel_repr,
            nwords=key.nwords,
            spec_pointer_range=key.spec_pointer_range,
            driver_path=key.driver_path,
            launch_symbol=key.launch_symbol,
            device_symbol=key.device_symbol,
            error_symbol=key.error_symbol,
            error_style=key.error_style,
            torch_access_mode=key.torch_access_mode,
            python_version=key.python_version,
            free_threaded=key.free_threaded,
            cpython_static_compile_header=key.cpython_static_compile_header,
            grid_arg=key.grid_arg,
            grid_py_mode=key.grid_py_mode,
            grid_cpp_source=key.grid_cpp_source,
            return_compiled=key.return_compiled,
            ndebug=key.ndebug,
            tuning=tuning,
            params=params,
            bound_params=tuple(v for v in params if v.bound_slot is not None),
            dynamic=tuple(_dynamic_view(slot) for slot in key.dynamic),
            fixed_device=fixed_device,
            tuned=tuned,
            tuned_meta=tuned_meta,
            tuned_level0=tuned and levels[0].computed > 0,
            grid_mode=grid_mode,
            grid_eval="deferred" if tuned and grid_mode in ("cpp", "py") else grid_mode,
            grid_object=grid_mode in ("object", "py"),
            grid_count=grid_count,
            lead_args=lead_args,
            stream_arg=lead_args - 1,
            ncontrols=lead_args + grid_count,
            grid_py_offset=grid_py_offset,
            init_extra=init_extra,
            launch_signature=_launch_signature(key, fixed_device, grid_mode),
            controls_doc=_controls_doc(fixed_device, grid_mode),
            torch_access_name=key.torch_access_mode.upper(),
            guarded=fixed_device or bool(bound) or key.grid_py_mode or tuned,
            launch_guard=key.return_compiled or (grid_mode == "cpp" and tuned),
            traverse_cache=(key.return_compiled or tuned_meta) and key.nwords > 0,
            traverse_fixed=key.return_compiled and fixed_device,
            auto_decode=auto_decode,
            auto_parts=_auto_parts(key) if auto_decode else (),
            device_write=None
            if key.device_offset is None
            else KeyWrite.into_part(key.device_offset)
            if auto_decode
            else KeyWrite.into_word(key.device_offset, 1),
            computed_writes=tuple(
                ComputedWrite(
                    value // 8,
                    2 * j,
                    KeyWrite.into_word(kind, 1),
                    2 * j + 1,
                )
                for j, (value, kind) in enumerate(
                    tuning.computed_fields if tuning is not None else ()
                )
            ),
            nbound=len(bound),
            nhidden=len(hidden),
            nmemo=sum(p.memo is not None for p in key.params) + len(key.dynamic),
            nparams=sum(p.call_index is not None for p in key.params),
            ndyn=len(key.dynamic),
            nslots=max(key.max_slots, 1),
            key_words=max(key.nwords, 1),
            key_bytes=key.nwords * 8,
            param_names_len=len(key.params) + 1,
            levels=levels,
            nlevels=len(levels),
            inner_levels=tuple(level for level in reversed(levels) if not level.final),
            later_levels=levels[1:],
            ndeps=ndeps,
            ncomp=ncomp,
            dep_len=max(2 * ndeps, 1),
            comp_len=max(2 * ncomp, 1),
            computed_names_len=ncomp + 1,
            computed_names=tuning.computed_names if tuning is not None else (),
            nmeta=len(meta_names),
        )


def _needs_generic_decode(a: CanonicalAnnotation) -> bool:
    """Whether a parameter is beyond the auto path's unannotated decode."""
    return bool(
        a.types is not None
        or a.baked_value
        or a.bind_value is not None
        or a.power_of_two_or_zero
        or (
            a.kind == "argument"
            and (
                a.equal_to_one != "auto"
                or a.aligned_16 != "auto"
                or a.pointer_range_32 != "auto"
            )
        )
    )


def _auto_parts(key: RenderKey) -> tuple[AutoPart, ...]:
    """The 32-bit parts the auto path's one-byte fields, the device's too, land in."""
    parts = {
        field.offset // 4
        for p in key.params
        for field in p.annotation.key_fields
        if field.width == 1
    }
    if key.device_offset is not None:
        parts.add(key.device_offset // 4)
    return tuple(AutoPart(part, part // 2, part % 2 * 32) for part in sorted(parts))


def _launch_signature(key: RenderKey, fixed_device: bool, grid_mode: str) -> str:
    grid = {
        "cpp": list(key.grid_extra),
        "py": [],
        "object": ["grid"],
        "dims": [f"grid_{'xyz'[i]}" for i in range(key.grid_arg or 0)],
    }[grid_mode]
    return ", ".join(
        [
            *([] if fixed_device else ["device"]),
            "stream",
            *grid,
            *(slot.name for slot in key.dynamic),
            *(p.name for p in key.params if p.call_index is not None),
        ]
    )


def _controls_doc(fixed_device: bool, grid_mode: str) -> str:
    grid = {
        "cpp": ["grid extras"],
        "py": [],
        "object": ["grid"],
        "dims": ["grid dimensions"],
    }[grid_mode]
    return ", ".join([*([] if fixed_device else ["device"]), "stream", *grid])


def _dynamic_view(slot: DynamicSlot) -> DynamicView:
    if slot.value_offset is None:
        return DynamicView(slot.name, slot.memo, None, None)
    assert slot.kind_offset is not None, "placed with its value"
    return DynamicView(
        slot.name,
        slot.memo,
        KeyWrite.into_word(slot.kind_offset, 1),
        KeyWrite.into_word(slot.value_offset, 8),
    )


def _levels(tuning: TuningRender | None) -> tuple[Level, ...]:
    count = tuning.levels if tuning is not None else 1
    deps = tuning.deps if tuning is not None else (0,) * count
    computed = tuning.computed if tuning is not None else (0,) * count
    levels: list[Level] = []
    child = None
    for n in reversed(range(count)):
        child = Level(
            n=n,
            final=n == count - 1,
            deps=deps[n],
            computed=computed[n],
            dep_at=2 * sum(deps[:n]),
            comp_offset=sum(computed[:n]),
            comp_at=2 * sum(computed[:n]),
            dep_len=max(2 * deps[n], 1),
            key_len=2 * computed[n],
            parent=n - 1,
            child=child,
        )
        levels.insert(0, child)
    return tuple(levels)


def _type_code(ty: str | None, codes: Mapping[str, int]) -> TypeCode:
    if ty is None:
        return TypeCode("none", "NONE", "")
    if ty.startswith("*"):
        return TypeCode("pointer", "", "", codes.get(ty))
    kind = (
        "bool"
        if ty == "u1"
        else "float"
        if ty in ("fp32", "fp64")
        else "signed"
        if ty.startswith("i")
        else "unsigned"
    )
    return TypeCode(kind, ty.upper(), ty[1:])


def _choice_order(ty: str | None) -> tuple[int, str]:
    """Narrowest first, so the first type a value fits is the one it keys as.

    Only integer widths are read: floats other than fp64 sort as width 0, by
    name, which still puts fp16 ahead of fp32.
    """
    if ty is None:
        return (0, "")
    width = (
        1
        if ty == "u1"
        else 64
        if ty == "fp64"
        else int(ty[1:])
        if ty[1:].isdigit()
        else 0
    )
    return (width, ty)


def _param_view(
    p: Param,
    *,
    verify: bool,
    spec_pointer_range: bool,
    codes: Mapping[str, int],
    bound_slot: int | None,
    init_extra: int,
    hidden_slot: int | None,
    meta_slot: int | None,
) -> ParamView:
    a = p.annotation
    fields: dict[str, KeyField] = {}
    for field in a.key_fields:
        fields.setdefault(field.kind, field)
    constexpr = a.kind == "constexpr"
    types = a.types
    typed = (
        types[0]
        if a.triton_typed
        and not a.baked_value
        and a.bind_value is None
        and types is not None
        and len(types) == 1
        and types[0] is not None
        and not types[0].startswith("*")
        else None
    )
    baked_tag = str(a.baked_value[0]) if a.baked_value and not constexpr else None
    baked_bits = ""
    if baked_tag == "float64":
        baked_bits = str(a.baked_value[1])
    elif baked_tag == "int":
        baked_bits = str(abs(int(str(a.baked_value[1]))))
    elif baked_tag == "bool":
        baked_bits = "1" if a.baked_value[1] else "0"
    baked_negative = baked_tag == "int" and int(str(a.baked_value[1])) < 0
    fixed_type = None
    if constexpr and types is None:
        type_from = "constexpr"
    elif constexpr and types is not None and len(types) > 1:
        type_from = "choices"
    elif types is not None and len(types) == 1:
        type_from = "fixed"
        fixed_type = _type_code(types[0], codes)
    else:
        type_from = "infer"
    descriptor = fields.get("descriptor")
    return ParamView(
        name=p.name,
        index=p.index,
        call_index=p.call_index,
        memo=p.memo,
        annotation=a,
        constexpr=constexpr,
        decoded=not a.tuned and (not constexpr or not a.baked_value),
        decoder=_decoder(p, typed),
        typed=typed,
        typed_unsigned=typed is not None and typed.startswith("u"),
        baked_tag=baked_tag,
        baked_kind={
            "none": "NONE",
            "bool": "BOOL",
            "int": "I64",
            "float64": "FP64",
        }.get(baked_tag or "", ""),
        baked_bits=baked_bits,
        baked_negative=baked_negative,
        range_check=(a.pointer_range_32 == "auto" and spec_pointer_range)
        or (a.pointer_range_32 == "assume" and verify),
        bound_range_check=a.pointer_range_32 == "assume" and verify,
        type_from=type_from,
        fixed_type=fixed_type,
        type_choices=tuple(
            _type_code(ty, codes) for ty in sorted(types or (), key=_choice_order)
        )
        if type_from == "choices"
        else (),
        checks=_checks(a, typed, verify, codes),
        bound_checks=_bound_checks(a, verify, codes),
        one_code=a.equal_to_one != "never",
        one_tests_value=a.equal_to_one == "auto",
        align_code=a.aligned_16 == "auto",
        range_code=a.pointer_range_32 == "auto" and spec_pointer_range,
        descriptor=_write(descriptor),
        descriptor_part=KeyWrite.into_part(descriptor.offset) if descriptor else None,
        payload=_write(fields.get("payload")),
        exact=_write(fields.get("exact")),
        exact_kind=_write(fields.get("exact_kind")),
        bound_slot=bound_slot,
        init_arg=None if bound_slot is None else bound_slot + init_extra,
        hidden_slot=hidden_slot,
        meta_slot=meta_slot,
    )


def _write(field: KeyField | None) -> KeyWrite | None:
    return KeyWrite.into_word(field.offset, field.width) if field is not None else None


def _decoder(p: Param, typed: str | None) -> str:
    a = p.annotation
    if a.baked_value:
        return "baked"
    if a.bind_value is not None:
        return f"bound_{a.bind_value}"
    if a.kind == "constexpr":
        return "constexpr_memo" if p.memo is not None else "constexpr"
    if a.tensor_only:
        return "tensor"
    if typed in ("fp32", "fp64"):
        return "typed_float"
    return "typed_int" if typed is not None else "argument"


def _checks(
    a: CanonicalAnnotation,
    typed: str | None,
    verify: bool,
    codes: Mapping[str, int],
) -> tuple[Check, ...]:
    """The checks `verify_annotation` adds to a decoded value, in order."""
    if not verify or a.baked_value or a.bind_value == "pointer":
        return ()
    checks: list[Check] = []
    type_error = "ValueError" if a.power_of_two_or_zero else "TypeError"
    range_error = "ValueError" if a.power_of_two_or_zero else "OverflowError"
    # a Triton-typed value was already converted by Triton's own rules
    if a.types is not None and typed is None:
        if len(a.types) == 1:
            ty = a.types[0]
            code = _type_code(ty, codes)
            if ty is not None and ty.startswith(("i", "u")) and ty != "u1":
                checks += [
                    Check("integer_kind", "type", type_error),
                    Check("valid", "range", range_error, code),
                ]
            elif ty == "fp32":
                checks += [
                    Check("float_kind", "type", "TypeError"),
                    Check("valid", "range", "OverflowError", code),
                ]
            else:
                checks.append(Check("valid", "type", "TypeError", code))
        elif a.kind == "constexpr":
            checks.append(Check("constexpr_type", "type allowlist", type_error))
        else:
            allowed = tuple(
                _type_code(ty, codes)
                for ty in a.types
                if ty is None or not ty.startswith("*") or ty in codes
            )
            checks.append(
                Check("allowlist", "type allowlist", "TypeError", None, allowed)
            )
    if a.power_of_two_or_zero:
        if a.types is None:
            checks.append(Check("integer_kind", "integer type", "ValueError"))
        checks.append(Check("power_of_two", "power_of_two_or_zero", "ValueError"))
    if a.kind == "argument":
        for fact in ("equal_to_one", "aligned_16", "pointer_range_32"):
            if getattr(a, fact) == "assume":
                checks.append(Check(fact, fact, "ValueError"))
    return tuple(checks)


def _bound_checks(
    a: CanonicalAnnotation, verify: bool, codes: Mapping[str, int]
) -> tuple[Check, ...]:
    if not verify or a.bind_value != "pointer" or not a.types:
        return ()
    checks = [Check("bound_type", "type", "TypeError", _type_code(a.types[0], codes))]
    if a.aligned_16 == "assume":
        checks.append(Check("bound_aligned_16", "aligned_16", "ValueError"))
    return tuple(checks)

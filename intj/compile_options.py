"""Compile options: where each comes from, how they merge, and the target's view.

A compile's options are the launcher's `options=`, overlaid by what one call or
candidate sets (parameter-fed, dynamic or autotune-config values), then by the
knobs Triton derives options from (`KNOBS`). `resolve` merges them and
canonicalizes the result for a target; everything that compiles, keys a module
or checks a config goes through it.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any

from . import _imports
from .errors import UnsupportedKernel

if TYPE_CHECKING:
    from .annotation import ResolvedParam
    from .render.key import Param


@dataclasses.dataclass(frozen=True)
class Knob:
    """A Triton knob that sets a compile option.

    Attributes:
        path: `knobs.<group>.<name>`.
        option: The compile option Triton derives from it.
        combine: `"or"`: ORed into the passed value, else the kernel's own
            attribute of that name. `"overwrite"`: replaces any passed value.
        default: The knob's value when it is not read: a declared dynamic one
            outside its miss. None leaves the option as passed.
    """

    path: str
    option: str
    combine: str
    default: object


#: Every knob Triton turns into a compile option. A launcher reads them at its
#: first call, and compat keys its launchers on all of them.
KNOBS = (
    Knob("knobs.runtime.debug", "debug", "or", False),
    Knob(
        "knobs.compilation.instrumentation_mode",
        "instrumentation_mode",
        "overwrite",
        "",
    ),
    Knob(
        "knobs.compilation.fpsan_homomorphic_casts",
        "fpsan_homomorphic_casts",
        "overwrite",
        None,
    ),
)

#: option name -> knob path, for options a knob overwrites: declaring one as a
#: `dynamic_options` entry would key a per-call value that never reaches the
#: compile.
KNOB_OVERWRITES = {k.option: k.path for k in KNOBS if k.combine == "overwrite"}

#: options a launch fixes itself; never `options=` or `dynamic_options`
FORBIDDEN_OPTIONS = ("device", "stream", "device_type", "warp_size")


def live_knobs() -> dict[str, object]:
    """Returns every knob in `KNOBS` by path, as it is now."""
    _imports.ensure_triton_imported()
    values: dict[str, object] = {}
    for knob in KNOBS:
        _, group, name = knob.path.split(".")
        values[knob.path] = getattr(getattr(_imports.knobs, group), name, None)
    return values


def knob_options(
    jit_func: Any, options: Mapping[str, Any], knob_values: Mapping[str, object]
) -> dict[str, Any]:
    """Returns `options` plus what Triton derives from `knob_values`."""
    merged = dict(options)
    for knob in KNOBS:
        value = knob_values.get(knob.path, knob.default)
        if knob.combine == "or":
            passed = options.get(knob.option, getattr(jit_func, knob.option))
            merged[knob.option] = passed or value
        elif value is not None:
            merged[knob.option] = value
    return merged


def canonical_options(target: Any, options: Mapping[str, Any]) -> Any:
    """Runs `options` through the triton compiler backend's `parse_options`.

    That fills in defaults and normalizes the odd fields (`extern_libs`,
    `llvm_fn_attrs`, `warp_size`), so `{}` and `{"num_warps": 4}` reach the same
    rendered module instead of building it twice.  It also rejects what the
    backend rejects when the module is built, not on the first kernel compile.

    `parse_options` ignores keys it does not know, so unknown ones are caught
    here -- otherwise a typo would silently compile with the default.
    """
    _imports.ensure_triton_imported()
    parsed: Any = _imports.make_backend(target).parse_options(dict(options))
    refuse_unknown_options(parsed, options)
    return parsed


def resolve(
    target: Any,
    jit_func: Any,
    base: Mapping[str, Any],
    overlay: Mapping[str, Any],
    knob_values: Mapping[str, object],
) -> Any:
    """Returns the target's options: `overlay` over `base`, then the knobs."""
    return canonical_options(
        target, knob_options(jit_func, {**base, **overlay}, knob_values)
    )


def refuse_unknown_options(parsed: Any, names: Iterable[str]) -> None:
    """Refuses the names that are no field of `parsed` (triton's options)."""
    unknown = set(names) - {f.name for f in dataclasses.fields(parsed)}
    if unknown:
        raise UnsupportedKernel(f"intj: unknown compile option(s) {sorted(unknown)}")


def parameter_options(
    backend: Any, params: Iterable[ResolvedParam | Param]
) -> tuple[str, ...]:
    """Returns the tl.constexpr parameters that double as compile options.

    These are named like one of `backend`'s compile options (`num_warps`,
    `num_stages`, `waves_per_eu`, ...), and every call's value also sets that
    option, as Triton's `k[grid](..., num_warps=8)` does.  Not a key field:
    the parameter's own exact key already tells every value apart.  Options a
    launch fixes itself stay parameters only.
    """
    fields = {f.name for f in dataclasses.fields(backend.parse_options({}))}
    return tuple(
        p.name
        for p in params
        if p.annotation.kind == "constexpr"
        and p.name in fields
        and p.name not in FORBIDDEN_OPTIONS
    )

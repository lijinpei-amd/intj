"""Autotune and heuristics layers between `make_launcher` and a `@triton.jit`.

`analyze` gives every name a role, visiting layers from the outermost inward:

- an *exact key*: a caller var an autotune layer keys on, keyed by its value;
- a *dependent* value: fixed by keys at or below its level, stored in that
  level's cache record;
- a *computed* key: a heuristic reading a caller var nothing keys, lowered to C
  and keyed at the level where its inputs become known.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable
from typing import Any

from .heuristic import Heuristic, HeuristicError, parse_heuristic
from .launcher import UnsupportedKernel


@dataclasses.dataclass(frozen=True)
class Dependent:
    name: str
    level: int  # stored in this level's record


@dataclasses.dataclass(frozen=True)
class Computed:
    name: str
    level: int  # keyed by this level's lookup
    heuristic: Heuristic


@dataclasses.dataclass(frozen=True)
class TuningPlan:
    jit_func: Any = dataclasses.field(compare=False)
    layers: tuple[Any, ...] = dataclasses.field(compare=False)  # outermost first
    exact_keys: tuple[str, ...] = ()
    dependent: tuple[Dependent, ...] = ()
    computed: tuple[Computed, ...] = ()
    levels: int = 1

    @property
    def tuned(self) -> frozenset[str]:
        """Every name a layer assigns: removed from the call, never passed by it."""
        return frozenset(
            [d.name for d in self.dependent] + [c.name for c in self.computed]
        )


def analyze(kernel: Any, fixed: Iterable[str] = ()) -> TuningPlan:
    """Assign roles for `kernel`, an Autotuner/Heuristics chain over a JITFunction.

    `fixed` names baked parameters: their values are in the module, so exact.
    """
    from triton.runtime.autotuner import Autotuner, Heuristics
    from triton.runtime.jit import JITFunction

    layers: list[Any] = []
    inner = kernel
    while type(inner) in (Autotuner, Heuristics):
        layers.append(inner)
        inner = inner.fn
    if not layers:
        raise UnsupportedKernel("intj: expected an autotune or heuristics wrapper")
    if not isinstance(inner, JITFunction):
        raise UnsupportedKernel(
            f"intj: the innermost layer must be a @triton.jit function, got {type(inner).__name__}"
        )
    params = {p.name: p for p in inner.params}
    frozen = frozenset(fixed)
    # name -> (exact, level, known): `level` is the key level of an exact value
    # (or its record level, if dependent); `known` is the lookup count after
    # which C can read it.
    info: dict[str, tuple[bool, int, int]] = {
        name: (bool(p.is_constexpr) or name in frozen, 0, 0)
        for name, p in params.items()
    }
    assigned: set[str] = set()
    exact_keys: list[str] = []
    dependent: list[Dependent] = []
    computed: list[Computed] = []

    def assign(name: str, what: str) -> None:
        if name in assigned:
            raise UnsupportedKernel(f"intj: {name!r} is assigned by two layers")
        if name in params and not params[name].is_constexpr:
            raise UnsupportedKernel(
                f"intj: {what} assigns runtime parameter {name!r}; only tl.constexpr "
                "parameters and compile options can be tuned"
            )
        assigned.add(name)

    for layer in layers:
        if type(layer) is Autotuner:
            level = 0
            for key in layer.keys:
                if key not in params:
                    continue  # Triton filters keys to parameter names too
                exact, _, known = info[key]
                if not exact:
                    info[key] = (True, known, known)
                    exact_keys.append(key)
                level = max(level, info[key][1])
            names: dict[str, None] = {}
            for config in layer.configs:
                for name in config.all_kwargs():
                    names[str(name)] = None
            for name in names:
                assign(name, "autotune")
                info[name] = (True, level, level + 1)
                dependent.append(Dependent(name, level))
            continue
        for name, fn in layer.values.items():
            try:
                heuristic = parse_heuristic(name, fn)
            except HeuristicError as error:
                raise UnsupportedKernel(f"intj: {error}") from error
            for read in heuristic.inputs:
                if read not in info:
                    raise UnsupportedKernel(
                        f"intj: heuristic {name!r} reads {read!r}, which is neither a "
                        "kernel parameter nor assigned by an outer layer"
                    )
            if name in heuristic.inputs:
                raise UnsupportedKernel(
                    f"intj: heuristic {name!r} reads its own result"
                )
            assign(name, "heuristic")
            if all(info[read][0] for read in heuristic.inputs):
                level = max((info[read][1] for read in heuristic.inputs), default=0)
                info[name] = (True, level, level + 1)
                dependent.append(Dependent(name, level))
            else:
                level = max((info[read][2] for read in heuristic.inputs), default=0)
                info[name] = (True, level, level)
                computed.append(Computed(name, level, heuristic))
    levels = 1 + max((c.level for c in computed), default=0)
    return TuningPlan(
        inner,
        tuple(layers),
        tuple(exact_keys),
        tuple(dependent),
        tuple(computed),
        levels,
    )

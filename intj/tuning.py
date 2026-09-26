"""Autotune and heuristics layers between `make_launcher` and a `@triton.jit`.

`analyze` gives every name a role, visiting layers from the outermost inward:

- an *exact key*: a caller var an autotune layer keys on, keyed by its value;
- a *dependent* value: fixed by keys at or below its level, stored in that
  level's cache record;
- a *computed* key: a heuristic reading a caller var nothing keys, lowered to C
  and keyed at the level where its inputs become known.
"""

from __future__ import annotations

import copy
import dataclasses
import inspect
import threading
from collections.abc import Callable, Iterable, Mapping
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


@dataclasses.dataclass(frozen=True)
class TunedGrid:
    """How the miss path hands Triton a grid, per launcher grid mode."""

    mode: str  # "dims" | "cpp" | "py"
    fn: Any = None
    positional: tuple[str, ...] = ()
    extras: tuple[str, ...] = ()


class _Shim:
    """The innermost layer of a private chain: compiles and launches through intj.

    Triton's layers call `run` for every benchmark candidate and last for the
    final launch, so after the outermost `run` returns, `final` is that call.
    """

    def __init__(
        self,
        jit_func: Any,
        compile_kernel: Callable[[dict[str, Any], dict[str, Any]], Any],
    ):
        self.fn = jit_func  # Triton unwraps `.fn` to reach the JITFunction
        self._signature = inspect.signature(jit_func.fn)
        self._compile = compile_kernel
        self.final: tuple[dict[str, Any], dict[str, Any], Any] | None = None

    def reset(self) -> None:
        self.final = None

    def run(self, *args: Any, grid: Any, warmup: bool, **kwargs: Any) -> Any:
        params = {k: v for k, v in kwargs.items() if k in self._signature.parameters}
        options = {k: v for k, v in kwargs.items() if k not in params}
        bound = self._signature.bind(*args, **params)
        bound.apply_defaults()
        values = dict(bound.arguments)
        kernel = self._compile(values, options)
        if not warmup:
            raw: Any = grid(values) if callable(grid) else grid
            dims = tuple(raw)
            dims = dims + (1,) * (3 - len(dims))
            if all(dims):
                kernel[dims](*values.values())
        self.final = (values, options, kernel)
        return kernel


def _private_chain(layers: tuple[Any, ...], shim: _Shim) -> list[Any]:
    """Shallow copies with their own tuner caches, bottoming out in `shim`."""
    from triton.runtime.autotuner import Autotuner

    private = [copy.copy(layer) for layer in layers]
    for outer, inner in zip(private, [*private[1:], shim]):
        outer.fn = inner
    for layer in private:
        if type(layer) is Autotuner:
            layer.cache = {}
            for name in (
                "best_config",
                "bench_time",
                "configs_timings",
                "nargs",
                "restore_copies",
            ):
                layer.__dict__.pop(name, None)
            _rebind_default_hooks(layer)
    return private


def _rebind_default_hooks(tuner: Any) -> None:
    """Triton's default reset_to_zero/restore_value hooks close over the tuner
    they were built for; a copy's hooks would write the user's tuner. Rebuild
    them over `tuner`, as `Autotuner.__init__` does. User hooks stay as given."""
    if not tuner.user_defined_pre_hook and (tuner.reset_to_zero or tuner.restore_value):

        def pre_hook(kwargs: dict[str, Any], reset_only: bool = False) -> None:
            for name in tuner.reset_to_zero:
                if kwargs[name] is not None:
                    kwargs[name].zero_()
            if not reset_only:
                tuner.restore_copies = {
                    name: kwargs[name].clone()
                    for name in tuner.restore_value
                    if kwargs[name] is not None
                }

        tuner.pre_hook = pre_hook
    if not tuner.user_defined_post_hook and tuner.restore_value:

        def post_hook(kwargs: dict[str, Any], exception: Any) -> None:
            del exception
            for name, value in tuner.restore_copies.items():
                kwargs[name].copy_(value)
            tuner.restore_copies = {}

        tuner.post_hook = post_hook


def _copy_back(originals: tuple[Any, ...], private: list[Any]) -> None:
    for original, mine in zip(originals, private):
        for name in ("best_config", "bench_time", "configs_timings"):
            if name in mine.__dict__:
                setattr(original, name, mine.__dict__[name])


def _c_scalar(name: str, value: object) -> object:
    if type(value) is bool or type(value) is int and -(1 << 63) <= value < (1 << 63):
        return value
    raise UnsupportedKernel(
        f"intj: {name!r} is read by a heuristic or grid_cpp but was tuned to {value!r}; "
        "only int and bool are supported"
    )


def make_tuned_callback(
    plan: TuningPlan,
    resolved: tuple[Any, ...],
    options: Mapping[str, Any],
    grid: TunedGrid,
    render: Any,
    return_compiled: bool,
) -> Callable[..., tuple[Any, ...]]:
    """The C miss callback for one bound launcher, which owns its private tuners."""
    from triton.compiler import make_backend

    from .launcher import (
        Param,
        _canonical_options,  # pyright: ignore[reportPrivateUsage]  # launcher internals
        _checked_compile,  # pyright: ignore[reportPrivateUsage]  # launcher internals
        _compiler_input,  # pyright: ignore[reportPrivateUsage]  # launcher internals
        _current_device,  # pyright: ignore[reportPrivateUsage]  # launcher internals
        _current_target,  # pyright: ignore[reportPrivateUsage]  # launcher internals
    )

    del return_compiled  # C decides whether the record keeps the object
    jit_func = plan.jit_func
    target = _current_target()
    backend = make_backend(target)
    # every parameter public: the shim sees the full call Triton makes
    params = tuple(Param(p.name, p.index, p.index, p.annotation) for p in resolved)
    compiled: dict[tuple[Any, str], Any] = {}  # keeps every CompiledKernel alive

    def compile_kernel(values: dict[str, Any], config_options: dict[str, Any]) -> Any:
        canonical = _canonical_options(target, {**options, **config_options})
        compiler_input = _compiler_input(
            jit_func, params, [values[p.name] for p in params], backend
        )
        key = (compiler_input, canonical.hash())
        kernel = compiled.get(key)
        if kernel is None:
            kernel = compiled[key] = _checked_compile(
                jit_func, compiler_input, target, canonical
            )
        return kernel

    shim = _Shim(jit_func, compile_kernel)
    private = _private_chain(plan.layers, shim)
    public = [
        p.name
        for p in resolved
        if not p.annotation.baked_value and not p.annotation.tuned
    ]
    baked = {p.name: p.baked for p in resolved if p.annotation.baked_value}
    lock = threading.RLock()
    running = [False]

    def callback(
        keyblob: bytes,
        nparams: int,
        device: int,
        stream: int,
        controls: Any,
        *args: Any,
    ) -> tuple[Any, ...]:
        del keyblob
        with lock:
            if running[0]:
                raise UnsupportedKernel(
                    "intj: a tuned launcher was re-entered while tuning"
                )
            running[0] = True
            try:
                return tune(nparams, device, stream, controls, args)
            finally:
                running[0] = False

    def tune(
        nparams: int, device: int, stream: int, controls: Any, args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        import torch

        current = _current_device()
        if device != current:
            raise UnsupportedKernel(
                f"intj: launching on device {device} while device {current} is current; "
                "make the target device current before the first launch"
            )
        named = {**dict(zip(public, args)), **baked}
        # Positional up to the first tuned parameter, as a Triton caller would
        # write it: prune functions read `named_args`, which is positional only.
        prefix: list[Any] = []
        for name in jit_func.arg_names:
            if name in plan.tuned:
                break
            prefix.append(named.pop(name))
        if grid.mode == "dims":
            if not all(controls):
                raise ValueError("intj: cannot tune on a zero-volume grid")
            triton_grid: Any = controls
        elif grid.mode == "cpp":
            extras = dict(zip(grid.extras, controls))

            def cpp_grid(meta: dict[str, Any]) -> Any:
                return grid.fn(*[meta[n] for n in grid.positional], **extras)

            triton_grid = cpp_grid
        else:
            triton_grid = grid.fn
        shim.reset()
        with torch.cuda.stream(torch.cuda.ExternalStream(stream, device=device)):
            private[0].run(*prefix, grid=triton_grid, warmup=False, **named)
        _copy_back(plan.layers, private)
        assert shim.final is not None, "Triton finished without a final launch"
        values, config_options, kernel = shim.final
        tuned_values = {**values, **config_options}
        md = kernel.metadata
        expected = sum(1 for ty in kernel.src.signature.values() if ty != "constexpr")
        if expected != nparams:
            raise RuntimeError(
                f"intj: packed {nparams} kernel arguments but triton compiled {expected}; "
                "this is an intj bug"
            )
        return (
            kernel.function,
            md.warp_size * md.num_warps,
            md.shared,
            nparams,
            kernel,
            tuple(_c_scalar(n, tuned_values[n]) for n in render.dep_names),
            tuple(tuned_values[n] for n in render.computed_names),
            tuple(values[n] for n in render.meta_names) if grid.mode == "py" else None,
        )

    return callback

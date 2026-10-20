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
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

from . import _imports
from .heuristic import Heuristic, HeuristicError, parse_heuristic
from .launcher import UnsupportedKernel


@dataclasses.dataclass(frozen=True)
class Dependent:
    """A tuned value fixed by the keys at or below `level`, stored in its record."""

    name: str
    level: int  # stored in this level's record


@dataclasses.dataclass(frozen=True)
class Computed:
    """A heuristic over a caller var nothing keys, lowered to C and keyed at `level`."""

    name: str
    level: int  # keyed by this level's lookup
    heuristic: Heuristic


@dataclasses.dataclass(frozen=True)
class TuningPlan:
    """The roles `analyze` gives every name of an autotune/heuristics chain.

    Attributes:
        jit_func: The innermost `JITFunction`.
        layers: The Autotuner/Heuristics layers, outermost first.
        exact_keys: Caller vars an autotune layer keys on, keyed by value.
        dependent: Tuned values stored in a level's cache record.
        computed: Heuristics lowered to C and keyed at a level.
        levels: The number of lookup levels.
    """

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
    """Assigns roles for `kernel`, an Autotuner/Heuristics chain over a JITFunction.

    Args:
        kernel: The outermost Autotuner or Heuristics layer.
        fixed: Names of baked parameters; their values are in the module, so exact.

    Returns:
        The plan: every name's role and the number of lookup levels.

    Raises:
        UnsupportedKernel: The chain has no tuning layer, does not end in a
            `@triton.jit` function, or has a layer intj cannot lower (a runtime
            parameter tuned, a name assigned twice, a read of a value an inner
            layer assigns, or an unparseable heuristic).
    """
    _imports.ensure_triton_imported()
    layers: list[Any] = []
    inner = kernel
    while type(inner) in (_imports.Autotuner, _imports.Heuristics):
        layers.append(inner)
        inner = inner.fn
    if not layers:
        raise UnsupportedKernel("intj: expected an autotune or heuristics wrapper")
    if not isinstance(inner, _imports.JITFunction):
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

    def assigns(layer: Any) -> set[str]:
        if type(layer) is _imports.Autotuner:
            return {str(n) for config in layer.configs for n in config.all_kwargs()}
        return set(layer.values)

    # inner_assigned[i]: names the layers inside layers[i] assign. An outer
    # layer runs first, so it cannot read them (Triton cannot either).
    inner_assigned: list[set[str]] = [set() for _ in layers]
    for i in range(len(layers) - 2, -1, -1):
        inner_assigned[i] = inner_assigned[i + 1] | assigns(layers[i + 1])

    def check_not_inner(name: str, read: str, i: int) -> None:
        if read in inner_assigned[i]:
            raise UnsupportedKernel(
                f"intj: {name} reads {read!r}, which an inner layer assigns"
            )

    for i, layer in enumerate(layers):
        if type(layer) is _imports.Autotuner:
            level = 0
            for key in layer.keys:
                if key not in params:
                    continue  # Triton filters keys to parameter names too
                check_not_inner("the autotune key", key, i)
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
                check_not_inner(f"heuristic {name!r}", read, i)
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
    """The grid the miss path hands Triton, per launcher grid mode."""

    mode: str  # "dims" | "cpp" | "py"
    fn: Any = None
    positional: tuple[str, ...] = ()
    extras: tuple[str, ...] = ()


class _Shim:
    """The innermost layer of a private chain, compiling and launching through intj.

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
        """Compiles for this call, launches it unless `warmup`, and records it.

        Keyword arguments that are not kernel parameters are compile options.
        A grid with a zero dimension compiles but does not launch.

        Returns:
            The `CompiledKernel`.
        """
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
    """Returns shallow copies of `layers` with their own tuner caches.

    The last copy's `fn` is `shim`; each tuner's default hooks are rebound to it.
    """
    _imports.ensure_triton_imported()
    private = [copy.copy(layer) for layer in layers]
    for outer, inner in zip(private, [*private[1:], shim]):
        outer.fn = inner
    for layer in private:
        if type(layer) is _imports.Autotuner:
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
    """Rebuilds `tuner`'s default reset_to_zero/restore_value hooks over `tuner`.

    Triton's default hooks close over the tuner they were built for; a copy's
    hooks would write the user's tuner. They are rebuilt as `Autotuner.__init__`
    does. User hooks stay as given.
    """
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


def c_scalar(name: str, value: object) -> object:
    """Returns `value` if C can hold it: a bool or an int that fits in int64.

    Raises:
        UnsupportedKernel: `value`, tuned for `name`, is anything else.
    """
    if type(value) is bool or type(value) is int and -(1 << 63) <= value < (1 << 63):
        return value
    raise UnsupportedKernel(
        f"intj: {name!r} is read by a heuristic or grid_cpp but was tuned to {value!r}; "
        "only int and bool are supported"
    )


class TunedCompileState:
    """One launcher's private tuning chain and candidate compile cache."""

    def __init__(
        self,
        plan: TuningPlan,
        resolved: tuple[Any, ...],
        options: Mapping[str, Any],
        grid: TunedGrid,
        render: Any,
        knob_values: Mapping[str, object],
        dynamic: Sequence[str] = (),
    ) -> None:
        from .backend import current_target
        from .launcher import (
            Param,
            _parameter_options,  # pyright: ignore[reportPrivateUsage]  # launcher internals
        )

        self.plan = plan
        self.jit_func = plan.jit_func
        self.target = current_target()
        self.backend = _imports.make_backend(self.target)
        self.options = options
        self.grid = grid
        self.render = render
        self.knob_values = knob_values
        self.dynamic = dynamic
        # Every parameter is public: the shim sees the full Triton call.
        self.params = tuple(
            Param(p.name, p.index, p.index, p.annotation) for p in resolved
        )
        # A CompiledKernel is loaded on one device; keep candidates alive.
        self.compiled: dict[
            tuple[Any, str, tuple[tuple[str, object], ...], int], Any
        ] = {}
        self.call_knobs: dict[str, object] = {}
        self.fed = _parameter_options(self.backend, self.params)
        self.shim = _Shim(self.jit_func, self._compile_kernel)
        self.private = _private_chain(plan.layers, self.shim)
        self.public = [
            p.name
            for p in resolved
            if not p.annotation.baked_value and not p.annotation.tuned
        ]
        self.baked = {p.name: p.baked for p in resolved if p.annotation.baked_value}
        self.lock = threading.RLock()
        self.running = False

    def _compile_kernel(
        self, values: dict[str, Any], config_options: dict[str, Any]
    ) -> Any:
        """Returns the `CompiledKernel` for one candidate, compiling it once."""
        from .backend import canonical_options, current_device
        from .launcher import (
            _checked_compile,  # pyright: ignore[reportPrivateUsage]  # launcher internals
            _compiler_input,  # pyright: ignore[reportPrivateUsage]  # launcher internals
            _knob_options,  # pyright: ignore[reportPrivateUsage]  # launcher internals
        )

        canonical = canonical_options(
            self.target,
            _knob_options(
                self.jit_func,
                {**self.options, **config_options, **{n: values[n] for n in self.fed}},
                {**self.knob_values, **self.call_knobs},
            ),
        )
        compiler_input = _compiler_input(
            self.jit_func,
            self.params,
            [values[p.name] for p in self.params],
            self.backend,
        )
        # knobs feeding no option change the binary too
        key = (
            compiler_input,
            canonical.hash(),
            tuple(sorted(self.call_knobs.items())),
            current_device(),
        )
        kernel = self.compiled.get(key)
        if kernel is None:
            kernel = self.compiled[key] = _checked_compile(
                self.jit_func, compiler_input, self.target, canonical
            )
        return kernel

    def run(
        self,
        keyblob: bytes,
        nparams: int,
        device: int,
        stream: int,
        controls: Any,
        *args: Any,
    ) -> tuple[Any, ...]:
        """Tunes for one miss under `lock`, refusing re-entry."""
        del keyblob
        with self.lock:
            if self.running:
                raise UnsupportedKernel(
                    "intj: a tuned launcher was re-entered while tuning"
                )
            self.running = True
            try:
                return self._tune(nparams, device, stream, controls, args)
            finally:
                self.running = False

    def _tune(
        self,
        nparams: int,
        device: int,
        stream: int,
        controls: Any,
        args: tuple[Any, ...],
    ) -> tuple[Any, ...]:
        """Runs the private chain on `stream` and returns the record for C."""
        from .backend import current_device
        from .launcher import (
            _check_live_knobs,  # pyright: ignore[reportPrivateUsage]  # launcher internals
            _split_dynamic,  # pyright: ignore[reportPrivateUsage]  # launcher internals
        )

        _imports.ensure_torch_imported()
        current = current_device()
        if device != current:
            raise UnsupportedKernel(
                f"intj: launching on device {device} while device {current} is current; "
                "make the target device current before the first launch"
            )
        dyn_values, args = args[: len(self.dynamic)], args[len(self.dynamic) :]
        dyn_options, dyn_knobs = _split_dynamic(self.dynamic, dyn_values)
        _check_live_knobs(dyn_knobs)
        self.call_knobs = dyn_knobs
        named = {**dict(zip(self.public, args)), **self.baked}
        # Positional up to the first tuned parameter, as a Triton caller would
        # write it: prune functions read `named_args`, which is positional only.
        prefix: list[Any] = []
        for name in self.jit_func.arg_names:
            if name in self.plan.tuned:
                break
            prefix.append(named.pop(name))
        if self.grid.mode == "dims":
            if not all(controls):
                raise ValueError("intj: cannot tune on a zero-volume grid")
            triton_grid: Any = controls
        elif self.grid.mode == "cpp":
            extras = dict(zip(self.grid.extras, controls))

            def cpp_grid(meta: dict[str, Any]) -> Any:
                return self.grid.fn(*[meta[n] for n in self.grid.positional], **extras)

            triton_grid = cpp_grid
        else:
            triton_grid = self.grid.fn
        self.shim.reset()
        try:
            # option values reach every config through the shim's kwargs
            cuda = _imports.torch.cuda
            with cuda.stream(cuda.ExternalStream(stream, device=device)):
                self.private[0].run(
                    *prefix, grid=triton_grid, warmup=False, **named, **dyn_options
                )
            _copy_back(self.plan.layers, self.private)
            assert self.shim.final is not None, "Triton finished without a final launch"
            values, config_options, kernel = self.shim.final
            tuned_values = {**values, **config_options}
            md = kernel.metadata
            expected = sum(
                1 for ty in kernel.src.signature.values() if ty != "constexpr"
            )
            if expected != nparams:
                raise RuntimeError(
                    f"intj: packed {nparams} kernel arguments but triton compiled "
                    f"{expected}; this is an intj bug"
                )
            return (
                kernel.function,
                md.warp_size * md.num_warps,
                md.shared,
                nparams,
                kernel,
                tuple(c_scalar(n, tuned_values[n]) for n in self.render.dep_names),
                tuple(tuned_values[n] for n in self.render.computed_names),
                tuple(values[n] for n in self.render.meta_names)
                if self.grid.mode == "py"
                else None,
            )
        finally:
            # Triton's own layers hold per-call scratch (args, tensor clones) on
            # `private` between the start and end of `run`; on the happy path
            # Autotuner.run already clears its `nargs`, but nothing clears it on
            # an exception, and `restore_copies` is only ever cleared by a
            # post_hook that never runs for a single matching config. `private`
            # outlives this call (it is reused for every future miss), so leaving
            # either set keeps this call's tensors alive until the next miss.
            self.shim.final = None
            self.call_knobs = {}
            for layer in self.private:
                if type(layer) is _imports.Autotuner:
                    layer.nargs = None
                    # set by a closure in Autotuner.__init__
                    layer.restore_copies = {}

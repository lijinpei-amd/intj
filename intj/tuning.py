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
import types
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, cast

from . import _imports
from . import backend as backends
from .annotation import DeviceBinding, ResolvedParam
from .heuristic import Heuristic, HeuristicError, Source, lower, parse_heuristic
from .launcher import Launcher, UnsupportedKernel
from .render.key import Param, TuningRender

if TYPE_CHECKING:
    from .grid import GridCode
    from .launcher import _BuildOptions  # pyright: ignore[reportPrivateUsage]  # launcher internals


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
    from .launcher import _is_chain  # pyright: ignore[reportPrivateUsage]  # launcher internals

    _imports.ensure_triton_imported()
    layers: list[Any] = []
    inner = kernel
    while _is_chain(inner):
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
class Tuned:
    """An autotune/heuristics chain's plan, and its render once the grid is known."""

    plan: TuningPlan
    render: TuningRender | None = None  # filled once the grid is known


def plan_tuning(
    chain: Any,
    resolved: tuple[ResolvedParam, ...],
    extra_annotation: Mapping[str, object] | None,
    options: Mapping[str, Any],
    no_gpu: bool,
) -> tuple[Tuned, tuple[ResolvedParam, ...]]:
    """Analyzes an autotune/heuristics `chain` into a `Tuned`, not yet rendered.

    Returns it with `resolved` marked: each parameter's `tuned` and
    `exact_key`.  GPU-free; config options are checked against the target on
    the first call.

    Raises:
        UnsupportedKernel: `no_gpu`, a bound parameter, `extra_annotation` or
            `options` naming a tuned value, or a config with `num_ctas > 1`.
    """
    if no_gpu:
        raise UnsupportedKernel(
            "intj: autotune/heuristics launchers need a GPU; no_gpu=True is not supported"
        )
    plan = analyze(chain, fixed=[p.name for p in resolved if p.annotation.baked_value])
    for p in resolved:
        if p.annotation.bind_value is not None:
            raise UnsupportedKernel(
                f"intj: bound parameter {p.name!r} cannot be tuned over"
            )
    clash = sorted(set(extra_annotation or {}) & plan.tuned)
    if clash:
        raise UnsupportedKernel(f"intj: extra_annotation names tuned value(s) {clash}")
    owned = sorted(set(options) & plan.tuned)
    if owned:
        raise UnsupportedKernel(
            f"intj: options {owned} are set by the autotune configs"
        )
    for config_options in _config_options(plan, resolved):
        if config_options.get("num_ctas", 1) != 1:
            raise UnsupportedKernel("intj: num_ctas > 1 is not supported")
    marked = tuple(
        dataclasses.replace(
            p,
            annotation=dataclasses.replace(
                p.annotation,
                tuned=p.name in plan.tuned,
                exact_key=p.name in plan.exact_keys,
            ),
        )
        for p in resolved
    )
    return Tuned(plan), marked


def _config_options(
    plan: TuningPlan, resolved: tuple[ResolvedParam, ...]
) -> Iterable[dict[str, Any]]:
    """Yields each autotune config's compile options: its kwargs naming no parameter."""
    _imports.ensure_triton_imported()
    names = {p.name for p in resolved}
    for layer in plan.layers:
        if type(layer) is _imports.Autotuner:
            for config in layer.configs:
                yield {
                    str(k): v for k, v in config.all_kwargs().items() if k not in names
                }


def check_tuned_configs(
    plan: TuningPlan,
    resolved: tuple[ResolvedParam, ...],
    options: Mapping[str, Any],
    knob_values: Mapping[str, object],
) -> None:
    """Runs every config's options through the target's `parse_options`.

    Called on the first call, because the target is a GPU query.
    """
    from .launcher import (
        _knob_options,  # pyright: ignore[reportPrivateUsage]  # launcher internals
    )

    target = backends.current_target()
    for config_options in _config_options(plan, resolved):
        backends.canonical_options(
            target,
            _knob_options(plan.jit_func, {**options, **config_options}, knob_values),
        )


def tuning_render(
    plan: TuningPlan,
    resolved: tuple[ResolvedParam, ...],
    grid_cpp: object | None,
    grid_py: object | None,
) -> TuningRender:
    """Returns the dep and comp slots for the grid and lowered heuristics, level-major.

    `computed_fields` is left empty: `render.module._build_module` places it."""
    from .launcher import (
        _render_params,  # pyright: ignore[reportPrivateUsage]  # launcher internals
    )

    _imports.ensure_triton_imported()
    by_name = {p.name: p for p in _render_params(resolved, DeviceBinding.NOT_FIXED)[0]}
    grid_code = getattr(grid_cpp, "__code__", None)
    grid_reads: tuple[str, ...] = (
        grid_code.co_varnames[: grid_code.co_argcount] if grid_code else ()
    )
    levels = plan.levels
    dependent_level: dict[str, int] = {d.name: d.level for d in plan.dependent}
    computed = tuple(sorted(plan.computed, key=lambda c: c.level))
    # A lowered read of a dependent value comes from a record walked before
    # its level's key is computed; `analyze` levels it so.
    for c in computed:
        for n in c.heuristic.inputs:
            if n in dependent_level and dependent_level[n] >= c.level:
                raise UnsupportedKernel(
                    f"intj: heuristic {c.name!r} reads {n!r} before it is known; "
                    "this is an intj bug"
                )
    wanted = [n for c in computed for n in c.heuristic.inputs if n in dependent_level]
    computed_reads = sorted({c.name for c in computed} & set(grid_reads))
    if computed_reads:
        raise UnsupportedKernel(
            f"intj: grid_cpp reads computed heuristic key(s) {computed_reads}; "
            "use grid_py or compute them in the grid"
        )
    wanted += [n for n in grid_reads if n in plan.tuned]
    # stable: level-major, first use within a level
    dep_names = tuple(sorted(dict.fromkeys(wanted), key=lambda n: dependent_level[n]))
    # C reads these; config values are static, so check them now, not on a miss
    for layer in plan.layers:
        if type(layer) is _imports.Autotuner:
            for config in layer.configs:
                for name, value in config.all_kwargs().items():
                    if name in dep_names:
                        c_scalar(str(name), value)
    sources: dict[str, Source] = {}
    for p in resolved:
        if p.annotation.baked_value:
            sources[p.name] = Source("fixed", value=p.baked)
        elif not p.annotation.tuned:
            sources[p.name] = Source("arg", by_name[p.name].call_index or 0)
    for slot, name in enumerate(dep_names):
        sources[name] = Source("dep", slot)
    for slot, c in enumerate(computed):
        sources[c.name] = Source("comp", slot)
    try:
        source = lower([(c.heuristic, c.level) for c in computed], sources, levels)
    except HeuristicError as error:
        raise UnsupportedKernel(f"intj: {error}") from error
    return TuningRender(
        levels=levels,
        computed=tuple(
            sum(c.level == level for c in computed) for level in range(levels)
        ),
        computed_fields=(),
        computed_names=tuple(c.name for c in computed),
        deps=tuple(
            sum(dependent_level[n] == level for n in dep_names)
            for level in range(levels)
        ),
        dep_names=dep_names,
        meta_names=tuple(p.name for p in resolved if p.annotation.tuned)
        if grid_py is not None
        else (),
        source=source,
    )


@dataclasses.dataclass(frozen=True)
class TunedGrid:
    """The grid the miss path hands Triton, per launcher grid mode."""

    mode: str  # "dims" | "cpp" | "py"
    fn: Any = None
    positional: tuple[str, ...] = ()
    extras: tuple[str, ...] = ()


def tuned_grid(
    grid_py: Callable[[dict[str, object]], object] | None,
    grid_cpp: object | None,
    grid_code: GridCode | None,
) -> TunedGrid:
    if grid_py is not None:
        return TunedGrid("py", grid_py)
    if grid_code is not None:
        code = getattr(grid_cpp, "__code__")
        return TunedGrid(
            "cpp",
            grid_cpp,
            tuple(code.co_varnames[: code.co_argcount]),
            grid_code.extras,
        )
    return TunedGrid("dims")


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
        from .launcher import (
            _KernelCompiler,  # pyright: ignore[reportPrivateUsage]  # launcher internals
            _parameter_options,  # pyright: ignore[reportPrivateUsage]  # launcher internals
        )

        self.plan = plan
        self.jit_func = plan.jit_func
        # its own, strong: a CompiledKernel is loaded on one device, and
        # records point into candidates no module sees
        self.compiler = _KernelCompiler(self.jit_func, options, knob_values, weak=False)
        self.grid = grid
        self.render = render
        self.dynamic = dynamic
        # Every parameter is public: the shim sees the full Triton call.
        self.params = tuple(
            Param(p.name, p.index, p.index, p.annotation) for p in resolved
        )
        self.call_knobs: dict[str, object] = {}
        self.fed = _parameter_options(self.compiler.backend, self.params)
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
        from .launcher import (
            _compiler_input,  # pyright: ignore[reportPrivateUsage]  # launcher internals
            _Identity,  # pyright: ignore[reportPrivateUsage]  # launcher internals
        )

        canonical = self.compiler.canonical(
            {**config_options, **{n: values[n] for n in self.fed}}, self.call_knobs
        )
        compiler_input = _compiler_input(
            self.jit_func,
            self.params,
            [values[p.name] for p in self.params],
            self.compiler.backend,
        )
        return self.compiler.compile(
            _Identity.of(compiler_input, canonical, self.call_knobs), canonical
        )

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
        from .launcher import (
            _check_device,  # pyright: ignore[reportPrivateUsage]  # launcher internals
            _record,  # pyright: ignore[reportPrivateUsage]  # launcher internals
            _split_call,  # pyright: ignore[reportPrivateUsage]  # launcher internals
        )

        _imports.ensure_torch_imported()
        _check_device(device)
        dyn_options, self.call_knobs, args = _split_call(self.dynamic, args)
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
            return (
                *_record(kernel, nparams),
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


class TunedLauncher(Launcher):
    """The launcher of an autotune/heuristics chain: `k.__self__` for such a `k`.

    `make_launcher` picks it from the kernel's type.  Its build also checks
    every config against the target, and a miss tunes through a private copy
    of the chain, which launches the winner itself; the rest is `Launcher`'s.
    """

    __slots__ = ()

    def _check_target(self, b: _BuildOptions) -> None:
        assert b.tuned is not None and b.knob_values is not None, (
            "plan_tuning refuses no_gpu"
        )
        check_tuned_configs(
            b.tuned.plan, b.params, dict(b.compile_options), b.knob_values
        )

    def _new_compile_state(
        self, b: _BuildOptions, module: types.ModuleType
    ) -> TunedCompileState:
        """Returns this launcher's private tuning state."""
        del module
        assert (
            b.tuned is not None
            and b.tuned.render is not None
            and b.knob_values is not None
        )
        return TunedCompileState(
            b.tuned.plan,
            b.params,
            dict(b.compile_options),
            tuned_grid(b.grid_py, b.grid_cpp_fn, b.grid_code),
            b.tuned.render,
            b.knob_values,
            b.dynamic,
        )

    def _compile_miss(
        self,
        keyblob: bytes,
        nparams: int,
        device: int,
        stream: int,
        controls: object,
        /,
        *args: Any,
    ) -> Any:
        """Tunes and launches the winner, returning its record for C to cache."""
        state = cast(TunedCompileState, self._compile_state)
        return state.run(keyblob, nparams, device, stream, controls, *args)

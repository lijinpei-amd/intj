"""What a launcher would build: every GPU-free check, and the plan they yield.

`build_plan` runs on `check()` and at the start of every build, before the
target is read. Its stages run in a fixed order, so the first refusal is the
cheapest one that applies and nothing reaches the network before every
refusal has had its chance.
"""

from __future__ import annotations

import dataclasses
import inspect
import os
import warnings
from collections.abc import Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, Any, cast

from . import _imports
from .annotation import (
    DeviceBinding,
    ResolvedParam,
    _canonical_value,  # pyright: ignore[reportPrivateUsage]  # tagged compiler constants
    _merge_annotations,  # pyright: ignore[reportPrivateUsage]  # private merge before target lookup
)
from .compile_options import FORBIDDEN_OPTIONS, KNOB_OVERWRITES, KNOBS, live_knobs
from .errors import UnsupportedKernel
from .kernel_cache import KernelCache
from .options import MakeLauncherOption
from .render import deps as render_deps
from .render import module as render_module
from .render.key import Param, TuningRender
from .specialization import KeyLayout, call_layout, key_layout
from .torch_intf.torch_abi import TorchAccessMode

if TYPE_CHECKING:
    from .grid import GridCode
    from .tuning import Tuned

# triton ships no type information
JitFunction = Any


class ClassGlobalWarning(RuntimeWarning):
    """A class global keyed by its qualified name, so replacing it goes undetected.

    Issued by `make_launcher(..., assume_constant_globals=True)`. Silence with
    `warnings.filterwarnings("ignore", category=ClassGlobalWarning)`.
    """


@dataclasses.dataclass(frozen=True)
class Grid:
    """How a launcher's calls give the grid.

    Attributes:
        mode: `"object"` (one grid argument), `"dims"` (`grid_arg` ints),
            `"cpp"` (`grid_cpp`) or `"py"` (`grid_py`).
        dims: `grid_arg`, in `"dims"` mode.
        code: `grid_cpp` compiled, in `"cpp"` mode.
        fn: The `grid_cpp` def or the `grid_py` callable.
        hidden: `grid_py`'s fixed values: the bound and baked parameters'.
    """

    mode: str
    dims: int | None = None
    code: GridCode | None = None
    fn: Any = None
    hidden: tuple[object, ...] = ()


@dataclasses.dataclass(frozen=True)
class BuildPlan:
    """Everything a build needs from a resolved `MakeLauncherOption`.

    `build_plan` returns it; nothing stores it beyond one build or `check()`.
    The target is not here: reading it is a GPU query, made by the build.
    """

    option: MakeLauncherOption  # `delayed` already called
    jit_func: JitFunction  # the chain unwrapped, checked
    tuned: Tuned | None  # its render complete
    params: tuple[ResolvedParam, ...]  # bound tensors' types resolved
    layout: KeyLayout  # `params` numbered in the call, key fields placed
    compile_options: tuple[tuple[str, Any], ...]  # `options=`, checked and sorted
    dynamic: tuple[str, ...]  # `dynamic_options` that get a slot
    #: `dynamic_options` naming a tl.constexpr parameter: no slot, and only
    #: checked to be a compile option once the target is known
    declared_parameters: tuple[str, ...]
    #: the live knobs but the declared ones; None under no_gpu
    knob_values: Mapping[str, object] | None
    global_values: tuple[tuple[str, str, tuple[object, ...]], ...]
    grid: Grid
    device_binding: DeviceBinding
    args: tuple[object, ...]  # `_init`'s trailing arguments: [device], *bound values
    torch_access_mode: TorchAccessMode
    doc: bytes  # the launcher's `__text_signature__`


def is_chain(kernel: object) -> bool:
    """Returns whether `kernel` is an autotune/heuristics layer: a TunedLauncher's."""
    return type(kernel) in (_imports.Autotuner, _imports.Heuristics)


def build_plan(option: MakeLauncherOption) -> BuildPlan:
    """Runs every GPU-free check on `option` and returns what the build needs.

    In order: the option's own values; the kernel; its annotations and tuning
    plan; `dynamic_options` and `options=`; the grid; the Torch access mode;
    warnings; bound values; then provisioning, which may reach the network.

    Raises:
        UnsupportedKernel: The kernel or an option is outside intj's scope,
            or the kernel cache's library cannot be provisioned.
        TypeError: A malformed option, or bind_device/bind_values that do not
            fit the kernel.
        ValueError: A malformed grid option.

    Warns:
        ClassGlobalWarning: `assume_constant_globals=True` keyed a class
            global by its qualified name.
        RuntimeWarning: `verify_annotation=True` cannot verify a bound
            pointer's range assumption.
    """
    assert option.delayed is None, "resolve delayed first"
    declared = _check_values(option)
    kernel, chain = _checked_kernel(option)
    options = dict(sorted((option.options or {}).items()))
    kernel = _check_kernel(kernel, options, option.assume_constant_globals)
    global_values = _global_values(kernel) if option.assume_constant_globals else ()
    resolved, tuned = _annotate(option, kernel, chain, options)
    _check_dynamic(declared, options, kernel, tuned)
    # a tl.constexpr parameter feeds its option on every call already: no slot
    dynamic = tuple(n for n in declared if n not in kernel.arg_names)
    _check_parameter_clash(options, kernel)
    # unbound: binding below changes no parameter's place in the call
    numbered = call_layout(resolved)
    render: TuningRender | None = None
    deps: Mapping[str, int] | None = None
    if tuned is not None:
        from .tuning import tuning_render

        render = tuning_render(
            tuned.plan, resolved, numbered, option.grid_cpp, option.grid_py
        )
        deps = {n: i for i, n in enumerate(render.dep_names)}
    grid = _grid(option, resolved, numbered, deps, len(dynamic))
    access = render_module._resolve_torch_access_mode(option.torch_access_mode)  # pyright: ignore[reportPrivateUsage]  # shared access checks
    _warn_unverifiable(option, resolved, kernel, global_values)
    params, device_binding, args, hidden = _bind(option, resolved)
    if grid.mode == "py":
        grid = dataclasses.replace(grid, hidden=hidden)
    layout = key_layout(
        call_layout(params),
        device_binding,
        render.computed[0] if render is not None else 0,
        dynamic,
    )
    if tuned is not None and render is not None:
        # level-0 computed keys are placed with the spec fields
        render = dataclasses.replace(render, computed_fields=layout.computed_fields)
        tuned = dataclasses.replace(tuned, render=render)
    doc = _launch_doc(
        layout.params, device_binding is DeviceBinding.FIXED, grid, dynamic
    )
    # a declared knob is per call: leaving its live value out keeps it out
    # of the RenderKey, so it cannot build a module per value
    knob_values = (
        None
        if option.no_gpu
        else {k: v for k, v in live_knobs().items() if k not in dynamic}
    )
    # GPU-free but may reach the network: after every refusal above, never on a build
    render_deps._provision(option.kernel_cache)  # pyright: ignore[reportPrivateUsage]  # checked toolchain
    return BuildPlan(
        option=option,
        jit_func=kernel,
        tuned=tuned,
        params=params,
        layout=layout,
        compile_options=tuple(options.items()),
        dynamic=dynamic,
        declared_parameters=tuple(n for n in declared if n in kernel.arg_names),
        knob_values=knob_values,
        global_values=global_values,
        grid=grid,
        device_binding=device_binding,
        args=args,
        torch_access_mode=access,
        doc=doc,
    )


def _check_values(option: MakeLauncherOption) -> tuple[str, ...]:
    """Checks the option's own values, with no imports, and returns `dynamic_options`."""
    dynamic_options = option.dynamic_options
    grid_arg, grid_cpp, grid_py = option.grid_arg, option.grid_cpp, option.grid_py
    no_gpu = option.no_gpu
    flags: dict[str, object] = {
        "no_gpu": no_gpu,
        "verify_annotation": option.verify_annotation,
        "return_compiled": option.return_compiled,
        "assume_constant_globals": option.assume_constant_globals,
        "ndebug": option.ndebug,
    }
    for name, flag in flags.items():
        if type(flag) is not bool:
            raise TypeError(f"intj: {name} must be a bool, got {flag!r}")
    mappings: dict[str, object] = {
        "extra_annotation": option.extra_annotation,
        "options": option.options,
    }
    for name, mapping in mappings.items():
        if mapping is not None and not (
            isinstance(mapping, Mapping) and all(type(k) is str for k in mapping)
        ):
            raise TypeError(f"intj: {name} must be None or a mapping keyed by str")
    if type(option.kernel_cache) is not KernelCache:
        raise TypeError(
            f"intj: kernel_cache must be a KernelCache, got {option.kernel_cache!r}"
        )
    torch_access_mode = option.torch_access_mode
    if torch_access_mode is not None and type(torch_access_mode) is not TorchAccessMode:
        raise TypeError(
            "intj: torch_access_mode must be None or a TorchAccessMode, "
            f"got {torch_access_mode!r}"
        )
    if isinstance(dynamic_options, str):
        raise TypeError("intj: dynamic_options is a sequence of names, not one string")
    try:
        declared = tuple(dynamic_options)
    except TypeError:
        raise TypeError(
            "intj: dynamic_options must be a sequence of names, "
            f"got {dynamic_options!r}"
        ) from None
    for name in declared:
        if type(name) is not str:
            raise TypeError(f"intj: dynamic_options names must be str, got {name!r}")
    if grid_arg is not None and (
        type(grid_arg) is not int or grid_arg not in (1, 2, 3)
    ):
        raise ValueError("intj: grid_arg must be 1, 2, or 3")
    if sum(value is not None for value in (grid_arg, grid_cpp, grid_py)) > 1:
        raise ValueError("intj: choose only one of grid_arg, grid_cpp and grid_py")
    if grid_py is not None and not callable(grid_py):
        raise TypeError("intj: grid_py must be callable")
    if no_gpu and declared:
        raise UnsupportedKernel(
            "intj: dynamic_options needs GPU mode; no_gpu=True compiles nothing"
        )
    if no_gpu and option.options:
        raise UnsupportedKernel(
            "intj: no_gpu=True supports only default compile options"
        )
    if no_gpu and option.return_compiled:
        raise UnsupportedKernel("intj: return_compiled=True requires GPU mode")
    return declared


def _checked_kernel(option: MakeLauncherOption) -> tuple[JitFunction, Any]:
    """Returns the kernel under any autotune/heuristics chain, and the chain."""
    _imports.ensure_triton_imported()
    render_module._verified_cpython_header()  # pyright: ignore[reportPrivateUsage]  # shared ABI check
    chain = kernel = option.jit_func
    while is_chain(kernel):
        kernel = kernel.fn
    interpreted = _imports.interpreted_function()
    if interpreted is not None and isinstance(kernel, interpreted):
        # `@triton.jit` under `TRITON_INTERPRET=1` returns an InterpretedFunction,
        # not a JITFunction: it has none of the attributes the analysis below
        # needs.  Decoration never gets here, so a module-level `@make_launcher`
        # still imports under the interpreter; this is the first call's refusal.
        raise UnsupportedKernel("intj: TRITON_INTERPRET=1 is not supported")
    return kernel, chain


def _annotate(
    option: MakeLauncherOption,
    kernel: JitFunction,
    chain: Any,
    options: Mapping[str, Any],
) -> tuple[tuple[ResolvedParam, ...], Tuned | None]:
    """Returns the kernel's merged annotations and, for a chain, its tuning plan."""
    resolved = _merge_annotations(kernel, option.extra_annotation)
    tuned: Tuned | None = None
    if chain is not kernel:
        from .tuning import plan_tuning

        tuned, resolved = plan_tuning(
            chain, resolved, option.extra_annotation, options, option.no_gpu
        )
    for p in kernel.params:
        kind = p._param.kind
        if kind not in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            raise UnsupportedKernel(
                f"intj: parameter {p.name!r} is {kind}; only positional parameters are supported"
            )
    return resolved, tuned


def _check_parameter_clash(options: Mapping[str, Any], kernel: JitFunction) -> None:
    constexprs = {p.name for p in kernel.params if p.is_constexpr}
    clash = sorted(n for n in options if n in constexprs)
    if clash:
        raise UnsupportedKernel(
            f"intj: options {clash} name tl.constexpr parameter(s), whose value is "
            "that compile option on every call; pass it as the argument, or bake "
            "it with extra_annotation"
        )


def _grid(
    option: MakeLauncherOption,
    resolved: tuple[ResolvedParam, ...],
    numbered: tuple[Param, ...],
    deps: Mapping[str, int] | None,
    ndynamic: int,
) -> Grid:
    """Returns the launcher's grid, `grid_cpp` compiled; `grid_py` gets its values once bound."""
    if option.grid_py is not None:
        return Grid("py", fn=option.grid_py)
    if option.grid_cpp is None:
        if option.grid_arg is None:
            return Grid("object")
        return Grid("dims", dims=option.grid_arg)
    from .grid import GridError, compile_grid

    try:
        code = compile_grid(
            option.grid_cpp,
            numbered,
            {p.index: p.baked for p in resolved if p.annotation.baked_value},
            deps,
            offset=ndynamic,
        )
    except GridError as error:
        raise UnsupportedKernel(f"intj: {error}") from error
    return Grid("cpp", code=code, fn=option.grid_cpp)


def _warn_unverifiable(
    option: MakeLauncherOption,
    resolved: tuple[ResolvedParam, ...],
    kernel: JitFunction,
    global_values: tuple[tuple[str, str, tuple[object, ...]], ...],
) -> None:
    if option.verify_annotation:
        for p in resolved:
            if (
                p.annotation.bind_value == "pointer"
                and p.annotation.pointer_range_32 == "assume"
            ):
                _warn(
                    f"intj: pointer-range assumption for bound pointer {p.name!r} "
                    "cannot be verified from an address and will be trusted",
                    RuntimeWarning,
                )
    classes = sorted(
        {".".join(c[1:]) for _, _, v in global_values for c in _class_globals(v)}
    )
    if classes:
        _warn(
            f"intj: {kernel.fn.__qualname__} reads class global(s) {classes}; intj "
            "keys a class by its qualified name and will not detect if it is "
            "replaced (assume_constant_globals=True)",
            ClassGlobalWarning,
        )


def _bind(
    option: MakeLauncherOption, resolved: tuple[ResolvedParam, ...]
) -> tuple[
    tuple[ResolvedParam, ...], DeviceBinding, tuple[object, ...], tuple[object, ...]
]:
    """Checks the bound device and values against the kernel.

    Returns:
        The parameters with bound tensors' types resolved, the device binding,
        `_init`'s trailing arguments, and `grid_py`'s fixed values.
    """
    device = option.bind_device
    if type(device) is bool:
        raise TypeError(
            "intj: bind_device takes an int device ordinal, not a flag; "
            "use k.__self__.bind_device(ordinal)"
        )
    if device is not None and (type(device) is not int or not 0 <= device <= 2**31 - 1):
        raise TypeError("intj: device ordinal must be an int in [0, 2**31)")
    values = cast(object, option.bind_values)  # as given: nothing checked it
    if values is None:
        values = {}
    if not (isinstance(values, Mapping) and all(type(k) is str for k in values)):
        raise TypeError("intj: bind_values must be None or a mapping keyed by str")
    names = {p.name for p in resolved if p.annotation.bind_value is not None}
    unknown = values.keys() - names
    if unknown:
        raise TypeError(f"intj: unknown bound parameter(s) {sorted(unknown)}")
    missing = names - values.keys()
    if missing:
        raise TypeError(f"intj: missing bound parameter(s) {sorted(missing)}")
    _imports.ensure_torch_imported()
    _imports.ensure_triton_imported()
    bound_params: list[ResolvedParam] = []
    for p in resolved:
        annotation = p.annotation
        if annotation.bind_value == "tensor":
            value = values[p.name]
            if value is not None and type(value) not in (
                _imports.torch.Tensor,
                _imports.torch.nn.Parameter,
            ):
                raise TypeError(
                    f"intj: bound tensor {p.name!r} must be a tensor or None"
                )
            if annotation.types == (None,) and value is not None:
                raise TypeError(
                    f"intj: bound tensor {p.name!r} must match declared None type"
                )
            if annotation.types is None:
                ty = None
                if value is not None:
                    tensor: Any = value
                    dtype = _imports.type_canonicalisation_dict.get(
                        str(tensor.dtype).split(".")[-1]
                    )
                    if dtype is None:
                        raise TypeError(
                            f"intj: bound tensor {p.name!r} has unsupported dtype {tensor.dtype}"
                        )
                    ty = "*" + dtype
                annotation = dataclasses.replace(annotation, types=(ty,))
        bound_params.append(dataclasses.replace(p, annotation=annotation))
    params = tuple(bound_params)
    bound_values = tuple(values[p.name] for p in params if p.name in names)
    hidden = tuple(
        values[p.name] if p.name in names else p.baked
        for p in params
        if p.annotation.bind_value is not None or p.annotation.baked_value
    )
    device_binding = DeviceBinding.NOT_FIXED if device is None else DeviceBinding.FIXED
    args = (*(() if device is None else (device,)), *bound_values)
    return params, device_binding, args, hidden


def _launch_doc(
    params: Sequence[Param],
    fixed_device: bool,
    grid: Grid,
    dynamic: Sequence[str] = (),
) -> bytes:
    """Returns the launcher's `__text_signature__`.

    Its controls (`device`, `stream`, grid) are renamed away from kernel names.
    """
    public = [p.name for p in params if p.call_index is not None]
    extras = list(grid.code.extras) if grid.code else []
    values = [name.replace(".", "_") for name in dynamic]
    grid_names = {
        "object": ["grid"],
        "dims": ["grid_x", "grid_y", "grid_z"][: grid.dims or 0],
    }.get(grid.mode, [])
    occupied = set(public) | set(extras) | set(values)
    controls: list[str] = []
    for name in (["stream"] if fixed_device else ["device", "stream"]) + grid_names:
        while name in occupied or name in controls:
            name += "_"
        controls.append(name)
    return (
        f"launch({', '.join(controls + extras + values + public)}, /)\n--\n\n".encode()
    )


_PACKAGE = os.path.dirname(__file__) + os.sep


def _warn(message: str, category: type[Warning]) -> None:
    # check() is reached from C, so no fixed stacklevel finds the caller:
    # point at the first frame outside intj instead
    frame, level = inspect.currentframe(), 1
    while frame is not None and frame.f_code.co_filename.startswith(_PACKAGE):
        frame, level = frame.f_back, level + 1
    warnings.warn(message, category, stacklevel=level)


def _check_dynamic(
    dynamic: tuple[str, ...],
    options: Mapping[str, Any],
    kernel: JitFunction,
    tuned: Tuned | None,
) -> None:
    """Refuses bad `dynamic_options` without touching a GPU.

    Unknown compile-option names wait for the target, in
    `render.module._build_module`.

    Raises:
        UnsupportedKernel: A repeated, unknown, forbidden or overwritten name,
            a runtime parameter, a name also in `options`, or one the autotune
            configs set.
    """
    _imports.ensure_triton_imported()
    if len(set(dynamic)) != len(dynamic):
        raise UnsupportedKernel(
            f"intj: dynamic_options repeats a name: {list(dynamic)}"
        )
    for name in dynamic:
        if name.startswith("knobs."):
            parts = name.split(".")
            group = getattr(_imports.knobs, parts[1], None) if len(parts) == 3 else None
            attr = (
                type(group).__dict__.get(parts[2])
                if isinstance(group, _imports.knobs.base_knobs)
                else None
            )
            if not (
                isinstance(attr, _imports.knobs.env_base)
                or type(attr) in (bool, int, float, str)
            ):
                raise UnsupportedKernel(
                    f"intj: unknown knob {name!r}; expected knobs.<group>.<name>, "
                    "e.g. knobs.runtime.debug"
                )
        elif name in FORBIDDEN_OPTIONS:
            raise UnsupportedKernel(f"intj: option {name!r} is not allowed")
        elif name in KNOB_OVERWRITES:
            raise UnsupportedKernel(
                f"intj: dynamic option {name!r} is overwritten by "
                f"{KNOB_OVERWRITES[name]!r}; declare that knob path instead"
            )
        elif name in kernel.arg_names:
            if not kernel.params[kernel.arg_names.index(name)].is_constexpr:
                raise UnsupportedKernel(
                    f"intj: dynamic option {name!r} is a runtime kernel parameter; "
                    "only a tl.constexpr parameter can also be a compile option"
                )
    derived = {k.path: k.option for k in KNOBS}
    both = sorted(n for n in dynamic if n in options or derived.get(n) in options)
    if both:
        raise UnsupportedKernel(
            f"intj: {both} given both in options= and dynamic_options; "
            "a value is fixed or per call, not both"
        )
    owned = sorted(set(dynamic) & tuned.plan.tuned) if tuned is not None else []
    if owned:
        raise UnsupportedKernel(
            f"intj: dynamic_options {owned} are set by the autotune configs"
        )


def _check_kernel(
    jit_func: JitFunction,
    options: Mapping[str, Any],
    assume_constant_globals: bool = False,
) -> JitFunction:
    """Returns `jit_func` once it is a kernel intj can launch.

    Raises:
        UnsupportedKernel: Not a JITFunction, has pre-run hooks, reads globals
            without `assume_constant_globals`, or `options` names an option a
            launch fixes itself.
    """
    _imports.ensure_triton_imported()
    if not isinstance(jit_func, _imports.JITFunction):
        raise UnsupportedKernel(
            "intj: expected a @triton.jit function, optionally wrapped in "
            f"@triton.autotune / @triton.heuristics, got {type(jit_func).__name__}"
        )
    if jit_func.pre_run_hooks:
        raise UnsupportedKernel("intj: kernels with pre-run hooks are not supported")
    jit_func.cache_key  # populates used_global_vals
    if jit_func.used_global_vals and not assume_constant_globals:
        names = sorted({name for name, _ in jit_func.used_global_vals})
        raise UnsupportedKernel(
            f"intj: kernel reads global variable(s) {names}, which intj cannot revalidate; "
            "pass them as arguments instead, or promise they never change with "
            "assume_constant_globals=True"
        )
    for key in FORBIDDEN_OPTIONS:
        if key in options:
            raise UnsupportedKernel(f"intj: option {key!r} is not allowed")
    return jit_func


def _canonical_global(value: object) -> tuple[object, ...]:
    """Canonicalizes one global's value for the `RenderKey`.

    Beyond `_canonical_value`: tl.constexpr wrappers, tuples, enum members (by
    name) and classes (by qualified name).

    Raises:
        ValueError: The value, or one inside it, cannot be canonicalized.
    """
    _imports.ensure_triton_imported()
    if isinstance(value, _imports.tl.constexpr):
        return ("constexpr", _canonical_global(cast(Any, value).value))
    if type(value) is tuple:
        return ("tuple", *(_canonical_global(item) for item in value))
    # a Python or pybind11 enum member (Gluon's `PropagateNan.ALL`), by name
    kind = type(value)
    members = getattr(kind, "__members__", None)
    name = getattr(value, "name", None)
    if isinstance(members, Mapping) and type(name) is str:
        member = cast(Mapping[str, object], members).get(name)
        # `is` covers Python's enum.Enum (members are singletons); `==` also
        # accepts a pybind11 enum member that Triton recorded as a deepcopy,
        # which is `==` but not `is` its live member (deepcopy makes a new
        # object; a pybind11 enum has no `__deepcopy__` to keep it a singleton).
        if (
            member is not None
            and type(member) is kind
            and (member is value or member == value)
        ):
            return ("enum", kind.__module__, kind.__qualname__, name)
    if isinstance(value, type):
        # like a JIT callee, by qualified name: the kernel's cache_key already
        # holds the source using it, and replacing the class is the caller's
        # promise not to (ClassGlobalWarning says so)
        return ("class", value.__module__, value.__qualname__)
    return _canonical_value(value)


def _class_globals(canonical: tuple[object, ...]) -> Iterator[tuple[str, str, str]]:
    """Yields the `("class", module, qualname)` entries of `canonical`.

    `canonical` is a `_canonical_global` result.
    """
    if canonical[0] == "class":
        yield cast(tuple[str, str, str], canonical)
    elif canonical[0] in ("constexpr", "tuple"):
        for item in canonical[1:]:
            yield from _class_globals(cast(tuple[object, ...], item))


def _global_values(
    jit_func: JitFunction,
) -> tuple[tuple[str, str, tuple[object, ...]], ...]:
    """Returns the globals `jit_func` reads, canonicalized for the `RenderKey`.

    The values are triton's snapshot from when it first computed `cache_key`.

    Raises:
        UnsupportedKernel: A global's value cannot be canonicalized.
    """
    values: list[tuple[str, str, tuple[object, ...]]] = []
    for (name, _), (value, scope) in jit_func.used_global_vals.items():
        try:
            canonical = _canonical_global(value)
        except ValueError:
            raise UnsupportedKernel(
                f"intj: global variable {name!r} holds a {type(value).__qualname__} "
                f"({value!r}), which intj cannot canonicalize into a module key "
                "(supported: tl.constexpr, int, float, bool, str, None, tl.dtype, "
                "enum members, classes, and tuples of these)"
            ) from None
        values.append((name, str(scope.get("__name__", "")), canonical))
    return tuple(sorted(values, key=lambda item: (item[0], item[1])))

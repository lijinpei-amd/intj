"""Make the C launcher for a `@triton.jit` kernel; `intj.render` renders and builds it."""

from __future__ import annotations

import dataclasses
import functools
import inspect
import os
import sysconfig
import threading
import types
import warnings
import weakref
from collections.abc import (
    Callable,
    Iterable,
    Iterator,
    Mapping,
    MutableMapping,
    Sequence,
)
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast, overload

from . import _imports, lazy
from . import backend as backends
from ._version import __version__
from .annotation import (
    CanonicalAnnotation,
    DeviceBinding,
    KeyField,
    ResolvedParam,
    _applicable,  # pyright: ignore[reportPrivateUsage]  # canonical specialization facts
    _canonical_value,  # pyright: ignore[reportPrivateUsage]  # tagged compiler constants
    _key_fields,  # pyright: ignore[reportPrivateUsage]  # private key layout
    _layout_fields,  # pyright: ignore[reportPrivateUsage]  # private key layout
    _resolve_annotations,  # pyright: ignore[reportPrivateUsage]  # private merge before target lookup
)
from .kernel_cache import (
    INSTALL_ERRORS,
    KernelCache,
    install,
    toolchain_for,
    unavailable_message,
)
from .options import OPTION_FIELDS, MakeLauncherKwargs, MakeLauncherOption
from .python_intf import cpython_abi
from .render.build import (
    CppExtensionUnavailable,
    compiler_identity,
    compiler_path,
    cxx_lib_dirs,
    cxx_toolchain,
    digest_flags,
    load,
    runtime_headers,
)
from .render.context import ENTRY_TEMPLATE
from .render.key import DynamicSlot, Param, RenderKey
from .torch_intf.torch_abi import (
    TensorABI,
    TorchAccessMode,
    dtype_index_table,
    layout_for,
    live_dtypes,
    supported_versions,
    torch_version,
)

# triton and torch ship no type information, so everything reaching into them is
# typed `Any` on purpose.
JitFunction = Any
CompiledKernel = Any

if TYPE_CHECKING:
    from typing_extensions import Unpack

    from .grid import GridCode
    from .tuning import Tuned, TunedCompileState


class UnsupportedKernel(NotImplementedError):
    """A kernel or option outside intj's (deliberately small) scope."""


class ClassGlobalWarning(RuntimeWarning):
    """A class global keyed by its qualified name, so replacing it goes undetected.

    Issued by `make_launcher(..., assume_constant_globals=True)`. Silence with
    `warnings.filterwarnings("ignore", category=ClassGlobalWarning)`.
    """


def _cache_version() -> tuple[int, int]:
    major, minor, *_ = __version__.split(".")
    return int(major), int(minor)


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


@dataclasses.dataclass(frozen=True)
class _BuildOptions:
    """Everything the build needs from a checked `MakeLauncherOption`.

    `_check_options` returns it; nothing stores it beyond one build.  The
    target is not here: reading it is a GPU query, made by the build.
    """

    jit_func: JitFunction  # the chain unwrapped, checked
    tuned: Tuned | None
    params: tuple[ResolvedParam, ...]  # bound tensors' types resolved
    torch_access_mode: TorchAccessMode
    grid_code: GridCode | None  # `grid_cpp` compiled
    dynamic: tuple[str, ...]
    #: `dynamic_options` naming a tl.constexpr parameter: no slot, and only
    #: checked to be a compile option once the target is known
    declared_parameters: tuple[str, ...]
    global_values: tuple[tuple[str, str, tuple[object, ...]], ...]
    device_binding: DeviceBinding
    args: tuple[object, ...]  # `_init`'s trailing arguments: [device], *bound values
    hidden: tuple[object, ...]  # grid_py's fixed values
    doc: bytes  # the launcher's `__text_signature__`
    # the option's own values, checked and normalized
    compile_options: tuple[tuple[str, Any], ...]
    torch_access_request: TorchAccessMode | None
    kernel_cache: KernelCache
    no_gpu: bool
    verify_annotation: bool
    ndebug: bool
    return_compiled: bool
    grid_arg: int | None
    grid_py: Callable[[dict[str, object]], object] | None
    grid_cpp_fn: object | None  # the `grid_cpp` def itself
    #: the live knobs but the declared ones; None under no_gpu
    knob_values: Mapping[str, object] | None


def _delayed_keywords(delayed: Callable[[], Mapping[str, Any]]) -> dict[str, Any]:
    values = delayed()
    if not isinstance(values, Mapping):  # pyright: ignore[reportUnnecessaryIsInstance]  # user code
        raise TypeError(
            "intj: delayed must return a mapping of make_launcher keywords, "
            f"got {type(values).__name__}"
        )
    values = cast("Mapping[object, object]", values)
    unknown = sorted(
        repr(k) for k in values if type(k) is not str or k not in OPTION_FIELDS
    )
    if unknown:
        raise TypeError(
            "intj: delayed returned unknown make_launcher keyword(s) "
            f"[{', '.join(unknown)}]"
        )
    return cast("dict[str, Any]", dict(values))


def _resolved(option: MakeLauncherOption) -> MakeLauncherOption:
    """Returns `option` with `delayed` called and its keywords replacing the others."""
    if option.delayed is None:
        return option
    return dataclasses.replace(
        option, **_delayed_keywords(option.delayed), delayed=None
    )


def _chained(
    first: Callable[[], Mapping[str, Any]] | None,
    then: Callable[[], Mapping[str, Any]],
) -> Callable[[], Mapping[str, Any]]:
    if first is None:
        return then

    def chained() -> Mapping[str, Any]:
        return {**_delayed_keywords(first), **_delayed_keywords(then)}

    return chained


_PACKAGE = os.path.dirname(__file__) + os.sep


def _warn(message: str, category: type[Warning]) -> None:
    # check() is reached from C, so no fixed stacklevel finds the caller:
    # point at the first frame outside intj instead
    frame, level = inspect.currentframe(), 1
    while frame is not None and frame.f_code.co_filename.startswith(_PACKAGE):
        frame, level = frame.f_back, level + 1
    warnings.warn(message, category, stacklevel=level)


def _check_options(option: MakeLauncherOption) -> _BuildOptions:
    """Runs every GPU-free check on `option` and returns what the build needs.

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
    jit_func = option.jit_func
    dynamic_options = option.dynamic_options
    extra_annotation = option.extra_annotation
    options = option.options
    torch_access_mode = option.torch_access_mode
    kernel_cache = option.kernel_cache
    no_gpu = option.no_gpu
    verify_annotation = option.verify_annotation
    grid_arg = option.grid_arg
    grid_cpp = option.grid_cpp
    grid_py = option.grid_py
    return_compiled = option.return_compiled
    assume_constant_globals = option.assume_constant_globals
    ndebug = option.ndebug
    # Cheap checks on every argument first: no imports, no GPU, no build.
    flags: dict[str, object] = {
        "no_gpu": no_gpu,
        "verify_annotation": verify_annotation,
        "return_compiled": return_compiled,
        "assume_constant_globals": assume_constant_globals,
        "ndebug": ndebug,
    }
    for name, flag in flags.items():
        if type(flag) is not bool:
            raise TypeError(f"intj: {name} must be a bool, got {flag!r}")
    mappings: dict[str, object] = {
        "extra_annotation": extra_annotation,
        "options": options,
    }
    for name, mapping in mappings.items():
        if mapping is not None and not (
            isinstance(mapping, Mapping) and all(type(k) is str for k in mapping)
        ):
            raise TypeError(f"intj: {name} must be None or a mapping keyed by str")
    if type(kernel_cache) is not KernelCache:
        raise TypeError(
            f"intj: kernel_cache must be a KernelCache, got {kernel_cache!r}"
        )
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
    if no_gpu and options:
        raise UnsupportedKernel(
            "intj: no_gpu=True supports only default compile options"
        )
    if no_gpu and return_compiled:
        raise UnsupportedKernel("intj: return_compiled=True requires GPU mode")
    kernel: JitFunction = jit_func
    _imports.ensure_triton_imported()
    _verified_cpython_header()
    options = dict(sorted((options or {}).items()))
    chain = kernel
    while type(kernel) in (_imports.Autotuner, _imports.Heuristics):
        kernel = kernel.fn
    interpreted = _imports.interpreted_function()
    if interpreted is not None and isinstance(kernel, interpreted):
        # `@triton.jit` under `TRITON_INTERPRET=1` returns an InterpretedFunction,
        # not a JITFunction: it has none of the attributes the analysis below
        # needs.  Decoration never gets here, so a module-level `@make_launcher`
        # still imports under the interpreter; this is the first call's refusal.
        raise UnsupportedKernel("intj: TRITON_INTERPRET=1 is not supported")
    kernel = _check_kernel(kernel, options, assume_constant_globals)
    global_values = _global_values(kernel) if assume_constant_globals else ()
    resolved = _resolve_annotations(kernel, extra_annotation)
    tuned: Tuned | None = None
    if chain is not kernel:
        from .tuning import plan_tuning

        tuned, resolved = plan_tuning(
            chain, resolved, extra_annotation, options, no_gpu
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
    _check_dynamic(declared, options, kernel, tuned)
    # a tl.constexpr parameter feeds its option on every call already: no slot
    dynamic = tuple(n for n in declared if n not in kernel.arg_names)
    constexprs = {p.name for p in kernel.params if p.is_constexpr}
    clash = sorted(n for n in options if n in constexprs)
    if clash:
        raise UnsupportedKernel(
            f"intj: options {clash} name tl.constexpr parameter(s), whose value is "
            "that compile option on every call; pass it as the argument, or bake "
            "it with extra_annotation"
        )
    deps: Mapping[str, int] | None = None
    if tuned is not None:
        from .tuning import tuning_render

        render = tuning_render(tuned.plan, resolved, grid_cpp, grid_py)
        tuned = dataclasses.replace(tuned, render=render)
        deps = {n: i for i, n in enumerate(render.dep_names)}
    compiled_grid: GridCode | None = None
    if grid_cpp is not None:
        from .grid import GridError, compile_grid

        grid_params, _, _ = _render_params(resolved, DeviceBinding.NOT_FIXED)
        try:
            compiled_grid = compile_grid(
                grid_cpp,
                grid_params,
                {p.index: p.baked for p in resolved if p.annotation.baked_value},
                deps,
                offset=len(dynamic),
            )
        except GridError as error:
            raise UnsupportedKernel(f"intj: {error}") from error
    access = _resolve_torch_access_mode(torch_access_mode)
    if verify_annotation:
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
    device_binding, params, args, hidden, doc = _check_binding(
        option, resolved, dynamic=dynamic, grid_code=compiled_grid
    )
    # a declared knob is per call: leaving its live value out keeps it out
    # of the RenderKey, so it cannot build a module per value
    knob_values = (
        None if no_gpu else {k: v for k, v in _live_knobs().items() if k not in dynamic}
    )
    # GPU-free but may reach the network: after every refusal above, never on a build
    _provision(kernel_cache)
    return _BuildOptions(
        jit_func=kernel,
        tuned=tuned,
        params=params,
        torch_access_mode=access,
        grid_code=compiled_grid,
        dynamic=dynamic,
        declared_parameters=tuple(n for n in declared if n in kernel.arg_names),
        global_values=global_values,
        device_binding=device_binding,
        args=args,
        hidden=hidden,
        doc=doc,
        compile_options=tuple(options.items()),
        torch_access_request=torch_access_mode,
        kernel_cache=kernel_cache,
        no_gpu=no_gpu,
        verify_annotation=verify_annotation,
        ndebug=ndebug,
        return_compiled=return_compiled,
        grid_arg=grid_arg,
        grid_py=grid_py,
        grid_cpp_fn=grid_cpp,
        knob_values=knob_values,
    )


def _check_binding(
    option: MakeLauncherOption,
    params: tuple[ResolvedParam, ...],
    *,
    dynamic: tuple[str, ...],
    grid_code: GridCode | None,
) -> tuple[
    DeviceBinding,
    tuple[ResolvedParam, ...],
    tuple[object, ...],
    tuple[object, ...],
    bytes,
]:
    """Checks `bind_device` and `bind_values` against the resolved `params`.

    Returns `(device_binding, params, args, hidden, doc)`: `params` with bound
    tensors' types resolved, `_init`'s trailing arguments, grid_py's fixed
    values, and the `__text_signature__`.
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
    names = {p.name for p in params if p.annotation.bind_value is not None}
    unknown = values.keys() - names
    if unknown:
        raise TypeError(f"intj: unknown bound parameter(s) {sorted(unknown)}")
    missing = names - values.keys()
    if missing:
        raise TypeError(f"intj: missing bound parameter(s) {sorted(missing)}")
    _imports.ensure_torch_imported()
    _imports.ensure_triton_imported()
    resolved: list[ResolvedParam] = []
    for p in params:
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
        resolved.append(dataclasses.replace(p, annotation=annotation))
    state = tuple(resolved)
    bound_values = tuple(values[p.name] for p in state if p.name in names)
    hidden = tuple(
        values[p.name] if p.name in names else p.baked
        for p in state
        if p.annotation.bind_value is not None or p.annotation.baked_value
    )
    binding = DeviceBinding.NOT_FIXED if device is None else DeviceBinding.FIXED
    args = (*(() if device is None else (device,)), *bound_values)
    rendered, _, _ = _render_params(state, binding)
    doc = _launch_doc(
        rendered,
        device is not None,
        option.grid_arg,
        grid_code,
        option.grid_py is not None,
        dynamic=dynamic,
    )
    return binding, state, args, hidden, doc


@dataclasses.dataclass
class _CompileState:
    """One launcher's specialization checks and optional private tuning state."""

    seen: dict[bytes, object] = dataclasses.field(default_factory=dict)
    tuned: TunedCompileState | None = None


class Launcher:
    """A launcher's object: `k.__self__` for `k = make_launcher(...)`.

    `k` is a C function whose entry is a build shim until its first call
    succeeds, then the rendered launch.  This object holds that entry, the
    options and, once built, the module's per-launcher state.  The C base
    (`_intj_lazy.Launcher`) defines the `_C_*` attributes and methods, and
    runs `_build_body` on the first call; `check()` and the in-place edits run
    under its build lock through `_C_locked`.  Every edit comes in two forms:
    `foo` returns a new, unbuilt launcher and `_foo` changes this one in place
    until it is built.
    """

    __slots__ = ()

    if TYPE_CHECKING:

        @property
        def _C_options(self) -> MakeLauncherOption: ...
        @property
        def _C_module(self) -> types.ModuleType | None: ...
        @property
        def _C_compile_state(self) -> _CompileState: ...
        @property
        def _C_built(self) -> bool: ...
        def _C_build(self) -> types.ModuleType: ...
        def _C_spec_key(self, *args: object) -> tuple[bytes, int]: ...
        def _C_key_chain(self, *args: object) -> tuple[tuple[bytes, ...], bool]: ...
        def _C_locked(self, fn: Callable[[], object], /) -> object: ...
        def _C_set_options(self, options: MakeLauncherOption, /) -> None: ...

    @property
    def options(self) -> MakeLauncherOption:
        """The launcher's options; once built, the resolved ones (`delayed` None)."""
        return self._C_options

    def check(self) -> None:
        """Runs every check the first call would, without building or storing anything.

        Calls `options.delayed` if set; the first call calls it again.  Does
        nothing once built.
        """
        if not self._C_built:
            self._C_locked(self._dry_run)

    def build(self) -> types.ModuleType:
        """Builds now if the first call has not, and returns the rendered module."""
        return self._C_build()

    def spec_key(self, *args: object) -> tuple[bytes, int]:
        """Returns `(key bytes, nparams)` for `(device, *dynamic, *public_args)`.

        Builds first.
        """
        return self._C_spec_key(*args)

    def key_chain(self, *args: object) -> tuple[tuple[bytes, ...], bool]:
        """Returns `((key bytes, ...), found)` for `(device, *dynamic, *public_args)`.

        Tuned launchers only; builds first.
        """
        return self._C_key_chain(*args)

    def _bind(self, /, **values: object) -> None:
        """Merges `values` into this launcher's `bind_values`; `bind` copies instead.

        Raises:
            RuntimeError: The launcher is built.
        """
        self._edit_options(
            lambda o: dataclasses.replace(
                o, bind_values={**(o.bind_values or {}), **values}
            )
        )

    def bind(self, /, **values: object) -> LaunchFunction:
        """Returns a new launcher with `values` merged into `bind_values`.

        Validates nothing: a missing, unknown or mistyped value fails in the
        new launcher's `check()`.  A bound tensor takes an exact
        `torch.Tensor`, `torch.nn.Parameter`, or None.
        """
        k = self._copy()
        k.__self__._bind(**values)
        return k

    def _bind_device(self, ordinal: int | None, /, **values: object) -> None:
        """Fixes this launcher to device `ordinal` in place; see `bind_device`.

        Raises:
            RuntimeError: The launcher is built.
        """
        self._edit_options(
            lambda o: dataclasses.replace(
                o, bind_device=ordinal, bind_values={**(o.bind_values or {}), **values}
            )
        )

    def bind_device(self, ordinal: int | None, /, **values: object) -> LaunchFunction:
        """Returns a new launcher fixed to device `ordinal`, `values` merged as for `bind`.

        It is called as `(stream, grid, ...)` and owns a separate kernel
        cache, or one kernel when no dynamic specialization key remains.
        None is `unbind_device()`.
        """
        k = self._copy()
        k.__self__._bind_device(ordinal, **values)
        return k

    def _unbind(self) -> None:
        """Drops this launcher's bound values in place; `unbind` copies instead."""
        self._edit_options(lambda o: dataclasses.replace(o, bind_values=None))

    def unbind(self) -> LaunchFunction:
        """Returns a new launcher with no bound values."""
        k = self._copy()
        k.__self__._unbind()
        return k

    def _unbind_device(self) -> None:
        """Makes this launcher take the device per call; `unbind_device` copies instead."""
        self._edit_options(lambda o: dataclasses.replace(o, bind_device=None))

    def unbind_device(self) -> LaunchFunction:
        """Returns a new launcher that takes the device per call."""
        k = self._copy()
        k.__self__._unbind_device()
        return k

    def _replace_delayed(self, fn: Callable[[], Mapping[str, Any]] | None, /) -> None:
        """Sets this launcher's `delayed` to `fn` (None drops it); `replace_delayed` copies instead.

        Raises:
            TypeError: `fn` is neither None nor callable.
            RuntimeError: The launcher is built.
        """
        if fn is not None and not callable(fn):
            raise TypeError(
                "intj: delayed must be None or a callable taking no arguments"
            )
        self._edit_options(lambda o: dataclasses.replace(o, delayed=fn))

    def replace_delayed(
        self, fn: Callable[[], Mapping[str, Any]] | None, /
    ) -> LaunchFunction:
        """Returns a new launcher whose `delayed` is `fn`; None drops it."""
        k = self._copy()
        k.__self__._replace_delayed(fn)
        return k

    def _chain_delayed(self, fn: Callable[[], Mapping[str, Any]], /) -> None:
        """Chains `fn` after this launcher's `delayed`; `chain_delayed` copies instead.

        Raises:
            TypeError: `fn` is not callable.
            RuntimeError: The launcher is built.
        """
        if not callable(fn):
            raise TypeError("intj: chain_delayed takes a callable taking no arguments")
        self._edit_options(
            lambda o: dataclasses.replace(o, delayed=_chained(o.delayed, fn))
        )

    def chain_delayed(self, fn: Callable[[], Mapping[str, Any]], /) -> LaunchFunction:
        """Returns a new launcher whose `delayed` also runs `fn`, whose keywords win."""
        k = self._copy()
        k.__self__._chain_delayed(fn)
        return k

    def _resolve_delayed(self) -> None:
        """Calls `delayed` now and applies its keywords in place; `resolve_delayed` copies instead.

        Does nothing when there is no `delayed`, built or not.
        """
        if self._C_options.delayed is None:
            return
        try:
            self._edit_options(_resolved)
        except RuntimeError:
            # another thread built it after the check above: a built launcher
            # holds resolved options, so there is nothing left to resolve
            if self._C_options.delayed is not None:
                raise

    def resolve_delayed(self) -> LaunchFunction:
        """Returns a new launcher with `delayed` called now and its keywords applied."""
        k = self._copy()
        k.__self__._resolve_delayed()
        return k

    def _copy(self) -> LaunchFunction:
        return _new_launcher(self._C_options)

    def _dry_run(self) -> None:
        # check()'s body, under the build lock: a first call may have built it since
        if not self._C_built:
            _check_options(_resolved(self._C_options))

    def _edit_options(
        self, edit: Callable[[MakeLauncherOption], MakeLauncherOption]
    ) -> None:
        # the `_foo` edits' body: options = edit(options) under the build lock
        def body() -> MakeLauncherOption:
            if self._C_built:
                raise RuntimeError(
                    "intj: a built launcher's options are fixed; the method "
                    "without the leading underscore returns an edited copy"
                )
            old = self._C_options
            self._C_set_options(edit(old))
            # dropped by _C_locked's caller, after the unlock: finalizers run Python
            return old

        self._C_locked(body)

    def _build_body(self) -> tuple[int, MakeLauncherOption, bytes]:
        """Builds this launcher on its first call and returns what `_intj_lazy` publishes.

        `_intj_lazy` calls it by name, under the build lock.

        Resolves `delayed`, checks, then queries the target, renders, compiles,
        loads, and fills this launcher's state block. Returns the rendered
        entry's address, the resolved options and the `__text_signature__`,
        for `_intj_lazy` to publish and swap in.

        `TRITON_INTERPRET=1` is refused here, on the first call: decoration
        never runs a kernel, so a module-level decorator does not make its
        module unimportable under the interpreter.
        Once this call succeeds the swapped-in entry never reads the knob
        again, so a launcher built with the interpreter off keeps launching if
        it is turned on later.
        """
        option = _resolved(self._C_options)
        b = _check_options(option)
        _imports.ensure_triton_imported()
        if _imports.knobs.runtime.interpret:
            raise UnsupportedKernel("intj: TRITON_INTERPRET=1 is not supported")
        if b.tuned is not None:
            from .tuning import check_tuned_configs

            assert b.knob_values is not None, "plan_tuning refuses no_gpu"
            check_tuned_configs(
                b.tuned.plan, b.params, dict(b.compile_options), b.knob_values
            )
        module = _materialize_module(b)
        compile_state = _CompileState()
        if b.tuned is not None:
            from .tuning import TunedCompileState, tuned_grid

            assert b.tuned.render is not None and b.knob_values is not None
            compile_state.tuned = TunedCompileState(
                b.tuned.plan,
                b.params,
                dict(b.compile_options),
                tuned_grid(b.grid_py, b.grid_cpp_fn, b.grid_code),
                b.tuned.render,
                b.knob_values,
                b.dynamic,
            )
        grid_py = () if b.grid_py is None else (b.grid_py, b.hidden)
        address = module._init(self, compile_state, _Interner(), *grid_py, *b.args)
        return address, option, b.doc

    def _compile_miss(self, *call: Any) -> Any:
        """Returns the kernel record for a C cache miss.

        Ordinary misses pass `(keyblob, nparams, device, *args)`; tuned misses
        also pass `stream` and grid controls before `args`. Dynamic values
        precede public kernel arguments. Tuning performs the final launch;
        an ordinary miss leaves that launch to C.
        """
        state = self._C_compile_state
        if state.tuned is not None:
            return state.tuned.run(*call)
        module = self._C_module
        assert module is not None, "compile misses require an initialized launcher"
        # Read per miss so override_compile reaches already built launchers.
        return getattr(module, "_intj_compile")(state.seen, *call)


class LaunchFunction(Protocol):
    """What `make_launcher` returns: a C function whose `__self__` is its `Launcher`."""

    @property
    def __self__(self) -> Launcher: ...
    def __call__(self, *args: Any) -> Any: ...


def _new_launcher(options: MakeLauncherOption) -> LaunchFunction:
    stub, cls = _stub_and_type()
    return stub.new_launcher(cls, options)


@overload
def make_launcher(
    jit_func: JitFunction,
    /,
    *,
    delayed: Callable[[], Mapping[str, Any]] | None = None,
    **kwargs: Unpack[MakeLauncherKwargs],
) -> LaunchFunction: ...


@overload
def make_launcher(
    *,
    delayed: Callable[[], Mapping[str, Any]] | None = None,
    **kwargs: Unpack[MakeLauncherKwargs],
) -> Callable[[JitFunction], LaunchFunction]: ...


#: make_launcher's "no kernel given" default: a positional None is an error, not
#: the decorator-factory form (an optional kernel whose import failed).
_NO_KERNEL: Any = object()


def make_launcher(
    jit_func: JitFunction = _NO_KERNEL,
    /,
    *,
    delayed: Callable[[], Mapping[str, Any]] | None = None,
    **kwargs: Unpack[MakeLauncherKwargs],
) -> Any:
    """Returns a fast launcher for `jit_func`, checked and built on first use.

    Called without `jit_func` (keyword arguments only), this is a decorator
    factory instead: `make_launcher(grid_cpp=grid)(kernel)` builds the same
    launcher `make_launcher(kernel, grid_cpp=grid)` would, so a kernel (plain
    or wrapped in `triton.autotune`/`triton.heuristics`) can be decorated
    directly:

        @intj.make_launcher(grid_cpp=grid)
        @triton.autotune(...)
        @triton.jit
        def k(...): ...

    Decoration only records a `MakeLauncherOption`: no kernel check, no GPU,
    no torch import.  Until one succeeds in this process, a call also imports
    triton and loads `_intj_lazy`, the C stub every launcher starts as, which
    can fail (see Raises); only the Triton refusal is remembered, and any
    other failure there is retried on the next call.  `k.__self__.check()`
    runs every other check now; otherwise the first call does, then builds the
    module (see docs/Usage.md, "Lazy build").

    The returned callable is a C function:

        launcher(device, stream, grid, arg0, arg1, ...)

    `device` is a device index in `[0, 256)` -- one byte of the spec key holds
    it -- and `stream` is a raw stream handle (both ints). The default `grid`
    is an int or a tuple/list of up to 3 ints. Declared `dynamic_options`
    values follow the grid controls. The remaining call arguments are the
    kernel's public parameters, positionally, in declaration order. Baked
    Argument and Constexpr values, and bound parameters, are omitted.

    Args:
        jit_func: A `@triton.jit` kernel, optionally wrapped in
            `triton.autotune`/`triton.heuristics`. Omit it for the decorator
            factory form.
        delayed: A function of no arguments returning more of these
            keywords, as a mapping; they replace the ones passed here. Each
            launcher calls it on its first call (and on every `check()`
            before that); its exception propagates and is retried on the
            next call.
        dynamic_options: Compile options and `knobs.<group>.<name>` paths
            passed per call, right after the grid controls, and keyed.
        extra_annotation: Parameter name -> annotation or shorthand: fixes
            types or specialization facts, bakes values, or marks values for
            binding.
        options: Triton compile options (`num_warps`, `num_stages`, ...)
            baked into every launch.
        torch_access_mode: How the module reads a tensor; None selects
            automatically. See `TorchAccessMode`.
        kernel_cache: The hash map behind the kernel cache; see `KernelCache`.
        no_gpu: Decode and cache on the host without compiling or launching a
            GPU kernel.
        verify_annotation: Check declared types, ranges and assumed facts;
            otherwise these are caller promises.
        bind_device: A non-negative int32 device ordinal: the launcher is
            called as `(stream, grid, ...)` and owns a separate kernel cache,
            or one kernel when no dynamic specialization key remains.
        bind_values: Bound parameter name -> value; see `Launcher.bind`.
        grid_arg: Take that many (1, 2 or 3) separate grid dimensions instead
            of `grid`.
        grid_cpp: An annotated def computing the grid in the extension from
            same-name JIT inputs and keyword-only extra inputs.
        grid_py: A callable given a fresh dict of JIT inputs on every launch,
            returning the grid. `grid_arg`, `grid_cpp` and `grid_py` are
            mutually exclusive.
        return_compiled: Return the cached Triton CompiledKernel after a
            successful launch, including on a zero-volume grid. Requires GPU
            mode.
        assume_constant_globals: Accept a kernel that reads global variables:
            their values when triton first hashed the kernel become part of
            the module identity, and are never checked again. Changing one
            afterwards is unsupported and not detected.
        ndebug: Compile the module with `NDEBUG` defined, so `assert()` in
            CPython's headers and intj's runtime is compiled out. False keeps
            them live, for debugging; it builds a separate module.

    Returns:
        The launcher, a C function whose `__self__` is its `Launcher`
        (`bind`, `bind_device`, `unbind`, `unbind_device`, `replace_delayed`,
        `chain_delayed`, `resolve_delayed`, each with an in-place `_` twin,
        and `check`). Without `jit_func`, a decorator that makes one.

    Raises:
        TypeError: A None kernel, an unknown keyword, or a `delayed` that is
            not callable.
        UnsupportedKernel: Triton is missing or older than 3.7, checked once
            per process: every later call raises the same refusal.  Or the
            host compiler failed to build `_intj_lazy`.
        ImportError: Importing an installed triton failed: a broken install,
            such as a missing dependency or extension module.
        RuntimeError: Triton finds no host C compiler.
        OSError: The directory `_intj_lazy` is built in cannot be created.
    """
    unknown = kwargs.keys() - OPTION_FIELDS
    if unknown:
        raise TypeError(
            "intj: make_launcher() got unexpected keyword argument(s) "
            f"{sorted(unknown)}"
        )
    if delayed is not None and not callable(delayed):
        raise TypeError("intj: delayed must be None or a callable taking no arguments")
    if jit_func is _NO_KERNEL:
        return functools.partial(make_launcher, delayed=delayed, **kwargs)
    if jit_func is None:
        raise TypeError("intj: make_launcher got None instead of a kernel")
    option = MakeLauncherOption(jit_func, delayed=delayed, **kwargs)
    return _new_launcher(option)


def _materialize_module(b: _BuildOptions) -> types.ModuleType:
    """Returns the loaded module for this kernel and configuration.

    Selects the backend and its target (`no_gpu`'s is `HOST`, with none), and
    loads, building if needed, the module for its `RenderKey`.
    `torch_access_request` is what `torch_access_mode` was resolved from, for
    choosing again where a build finds STATIC_COMPILE unavailable after all.

    Raises:
        UnsupportedKernel: The target's backend is not registered, a
            compile option name is unknown to it, or STATIC_COMPILE was
            requested and the module needs a build cpp_extension cannot do.
    """
    options = dict(b.compile_options)
    knob_values = b.knob_values
    tuning = b.tuned.render if b.tuned is not None else None
    backend, target = backends.select(b.no_gpu)
    canonical_options = None
    if target is not None:
        assert knob_values is not None, "GPU builds read knobs at the first call"
        canonical_options = backends.canonical_options(
            target, _knob_options(b.jit_func, options, knob_values)
        )
        # the names only: each value is canonicalized on its own miss
        backends.refuse_unknown_options(
            canonical_options,
            [n for n in b.dynamic if not n.startswith("knobs.")]
            + list(b.declared_parameters),
        )
    params, device_offset, nwords, computed_fields, slots = _spec_key_layout(
        b.params,
        b.device_binding,
        tuning.computed[0] if tuning is not None else 0,
        b.dynamic,
    )
    if tuning is not None:
        # final only now: level-0 computed keys are placed with the spec fields
        tuning = dataclasses.replace(tuning, computed_fields=computed_fields)
    # tuned levels always use a map
    nwords = max(nwords, 1) if tuning is not None else nwords
    # check() provisioned already: this is the idempotent, no-network
    # lookup of the toolchain it installed
    cache_toolchain = _provision(b.kernel_cache)
    # the kernel's own name, unless python allows something C does not
    module_name = b.jit_func.__name__
    if not (module_name.isidentifier() and module_name.isascii()):
        module_name = "kernel"
    # The rendered source is a pure function of the template, the runtime header
    # and the fields it reads, so hash those rather than the render itself.
    key = RenderKey(
        module_name=module_name,
        kernel_repr=get_full_name(b.jit_func),
        params=params,
        nwords=nwords,
        device_binding=b.device_binding,
        device_offset=device_offset,
        verify_annotation=b.verify_annotation,
        max_slots=sum(p.annotation.kind == "argument" for p in params),
        # Where the backend specializes on it, the pointer-range bit is always in
        # the key, even when the knob that emits it is off: a key that cannot tell
        # a > 2 GiB buffer apart would launch a tt.pointer_range=32 binary on it
        # after the knob is flipped back on.
        spec_pointer_range=int(backend.pointer_range),
        driver_path=backend.library_path(),
        launch_symbol=backend.launch_symbol,
        device_symbol=backend.device_symbol,
        error_symbol=backend.error_symbol,
        error_style=backend.error_style,
        # `_for_access` fills in these and `compiler`, `build_flags`
        torch_access_mode="",
        torch_version=None,
        cxx_abi=None,
        kernel_cache=b.kernel_cache.value,
        cache_include_dirs=cache_toolchain["include_dirs"],
        cache_archives=cache_toolchain["archives"],
        python_version=cpython_abi.python_version(),
        free_threaded=bool(sysconfig.get_config_var("Py_GIL_DISABLED")),
        cpython_static_compile_header=_verified_cpython_header(),
        pointer_types=_pointer_types(params),
        grid_arg=b.grid_arg,
        grid_py_mode=b.grid_py is not None,
        grid_cpp_source=b.grid_code.source if b.grid_code else "",
        grid_extra=b.grid_code.extras if b.grid_code else (),
        return_compiled=b.return_compiled,
        ndebug=b.ndebug,
        tuning=tuning,
        dynamic=slots,
        template=ENTRY_TEMPLATE.read_bytes().hex(),
        runtime_header=runtime_headers().hex(),
        cache_key=b.jit_func.cache_key + (":gluon" if b.jit_func.is_gluon() else ""),
        target=(target.backend, target.arch, target.warp_size)
        if target is not None
        else ("host-only",),
        options=canonical_options.hash()
        if canonical_options is not None
        else "host-defaults",
        triton=_triton_identity(),
        compiler=(),
        build_flags=(),
        ext_suffix=sysconfig.get_config_var("EXT_SUFFIX"),
        intj_version=_cache_version(),
        global_values=b.global_values,
    )
    baked_values = {p.index: p.baked for p in b.params if p.annotation.baked_value}

    def compile_function() -> Callable[..., Any]:
        if tuning is not None:
            return _refuse_module_compile
        if target is None:
            return _make_host_compile_callback()
        return _make_compile_callback(
            b.jit_func,
            params,
            options,
            baked_values,
            return_compiled=b.return_compiled,
            knob_values=knob_values or {},
            dynamic=b.dynamic,
        )

    def loaded(mode: TorchAccessMode) -> types.ModuleType:
        # the c++ mode gets it too, not to read from but to check its own
        # compiled-in offset against
        layout = (
            layout_for()
            if mode in (TorchAccessMode.RUNTIME_SHIM, TorchAccessMode.STATIC_COMPILE)
            else None
        )
        return _loaded_module(
            _for_access(key, mode), b.jit_func.__module__, layout, compile_function
        )

    try:
        return loaded(b.torch_access_mode)
    except CppExtensionUnavailable as error:
        # `check()` chose STATIC_COMPILE without importing cpp_extension, and
        # no `.so` was on disk: choose again, as `check()` used to, from
        # cpp_extension itself.  An explicit request is refused.
        try:
            confirmed = _resolve_torch_access_mode(
                b.torch_access_request, confirmed=True
            )
        except UnsupportedKernel as refusal:
            raise refusal from error
        return loaded(confirmed)


def _for_access(key: RenderKey, access: TorchAccessMode) -> RenderKey:
    """Returns `key` with the fields `access` decides filled in."""
    static = access is TorchAccessMode.STATIC_COMPILE
    key = dataclasses.replace(
        key,
        torch_access_mode=access.value,
        # Only the c++ mode compiles anything torch-specific in, so only it has
        # to be rebuilt when torch changes.  Leaving these None for the other
        # two is what keeps their digest -- and so their cached `.so` -- stable
        # across torch versions.
        torch_version=torch_version() if static else None,
        cxx_abi=_cxx_abi() if static else None,
    )
    # `digest_flags` derives these from the fields above
    flags = digest_flags(key)
    return dataclasses.replace(
        key,
        compiler=compiler_identity(flags["language"]),
        build_flags=tuple(flags["ccflags"]),
    )


_INSTALL_FAILED: dict[KernelCache, Exception] = {}


def _provision(cache: KernelCache) -> dict[str, tuple[str, ...]]:
    """Returns the toolchain for `cache`, fetching and building it the first time.

    Only ever reached from `check()`, on the slow path that was already
    going to invoke a compiler.  A failure is remembered: a script that builds
    twenty launchers offline should wait out one connect timeout, not twenty.
    """
    toolchain = toolchain_for(cache)
    if toolchain is not None:
        return toolchain
    if cache not in _INSTALL_FAILED:
        try:
            return install(cache)
        # INSTALL_ERRORS, not Exception: a TypeError from a bug in here is not a
        # provisioning failure, and must not send the reader off to re-run a
        # command that will fail identically.
        except INSTALL_ERRORS as error:
            _INSTALL_FAILED[cache] = error
    failure = _INSTALL_FAILED[cache]
    raise UnsupportedKernel(unavailable_message(cache, failure)) from failure


_stub_loaded: tuple[types.ModuleType, type[Launcher]] | None = None
_STUB_LOCK = threading.Lock()


def _stub_and_type() -> tuple[types.ModuleType, type[Launcher]]:
    """Returns the process's one `_intj_lazy` and the `Launcher` type over it.

    Loaded beside the modules intj renders, once, under a lock: a second load
    has its own `_intj_lazy.Launcher`, and `new_launcher` refuses a type over
    another load's.
    """
    global _stub_loaded
    loaded = _stub_loaded
    if loaded is None:
        with _STUB_LOCK:
            loaded = _stub_loaded
            if loaded is None:
                _imports.ensure_triton_imported()
                stub = lazy.load_stub(
                    Path(_imports.knobs.cache.get_triton_dir("intj")) / "lazy",
                    compiler_path("c"),
                )
                namespace = {"__slots__": (), "__module__": __name__}
                cls = type("Launcher", (Launcher, stub.Launcher), namespace)
                loaded = _stub_loaded = (stub, cast("type[Launcher]", cls))
    return loaded


def _stub() -> types.ModuleType:
    return _stub_and_type()[0]


def module_of(launcher: Callable[..., Any]) -> types.ModuleType:
    """Returns the rendered module behind `launcher`, building it if needed.

    Builds now if the launcher's first call has not.  For tests and debugging;
    `k.__self__.spec_key(...)` builds the same way.
    """
    return getattr(launcher, "__self__").build()


def _launch_doc(
    params: Sequence[Param],
    fixed_device: bool,
    grid_arg: int | None,
    grid_cpp: GridCode | None,
    grid_py: bool,
    dynamic: Sequence[str] = (),
) -> bytes:
    """Returns the launcher's `__text_signature__`.

    Its controls (`device`, `stream`, grid) are renamed away from kernel names.
    """
    public = [p.name for p in params if p.call_index is not None]
    extras = list(grid_cpp.extras) if grid_cpp else []
    values = [name.replace(".", "_") for name in dynamic]
    grid = (
        []
        if grid_cpp or grid_py
        else ["grid"]
        if grid_arg is None
        else ["grid_x", "grid_y", "grid_z"][:grid_arg]
    )
    occupied = set(public) | set(extras) | set(values)
    controls: list[str] = []
    for name in (["stream"] if fixed_device else ["device", "stream"]) + grid:
        while name in occupied or name in controls:
            name += "_"
        controls.append(name)
    return (
        f"launch({', '.join(controls + extras + values + public)}, /)\n--\n\n".encode()
    )


#: knob path -> the compile option triton derives from it
_KNOB_OPTIONS: tuple[tuple[str, str], ...] = (
    ("knobs.runtime.debug", "debug"),
    ("knobs.compilation.instrumentation_mode", "instrumentation_mode"),
    ("knobs.compilation.fpsan_homomorphic_casts", "fpsan_homomorphic_casts"),
)

#: option name -> knob path, for options `_knob_options` replaces outright
#: (unlike "debug", which only ORs a knob's value into a passed one, these
#: overwrite whatever was passed; declaring them as `dynamic_options` would
#: key a per-call value that never reaches the compile).
_KNOB_OVERWRITES: dict[str, str] = {
    option: path for path, option in _KNOB_OPTIONS if option != "debug"
}


def _live_knobs() -> dict[str, object]:
    """Returns the knobs triton turns into compile options, as they are now.

    Read at a launcher's first call and fixed for it.
    """
    _imports.ensure_triton_imported()
    values: dict[str, object] = {}
    for path, _ in _KNOB_OPTIONS:
        _, group, name = path.split(".")
        values[path] = getattr(getattr(_imports.knobs, group), name, None)
    return values


def _knob_options(
    jit_func: JitFunction, options: Mapping[str, Any], knob_values: Mapping[str, object]
) -> dict[str, Any]:
    """Returns `options` plus what triton derives from knobs.

    An explicit `debug` overrides the kernel's default, and the runtime knob
    can still enable it.  A knob missing from `knob_values` (a declared
    dynamic one, outside its miss) takes triton's default.
    """
    merged = dict(options)
    merged["debug"] = options.get("debug", jit_func.debug) or knob_values.get(
        "knobs.runtime.debug", False
    )
    merged["instrumentation_mode"] = knob_values.get(
        "knobs.compilation.instrumentation_mode", ""
    )
    casts = knob_values.get("knobs.compilation.fpsan_homomorphic_casts")
    if casts is not None:
        merged["fpsan_homomorphic_casts"] = casts
    return merged


def _dynamic_key_fields(name: str) -> tuple[KeyField, ...]:
    """Returns one dynamic value's key fields: a kind byte and 8 value bytes.

    Laid out like an exact key.  `name` picks nothing here; the invariant
    tests' mutation checks drop one value's fields by name.
    """
    del name
    return (KeyField("exact_kind", 1), KeyField("exact", 8))


def _check_dynamic(
    dynamic: tuple[str, ...],
    options: Mapping[str, Any],
    kernel: JitFunction,
    tuned: Tuned | None,
) -> None:
    """Refuses bad `dynamic_options` without touching a GPU.

    Unknown compile-option names wait for the target, in `_materialize_module`.

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
        elif name in _FORBIDDEN_OPTIONS:
            raise UnsupportedKernel(f"intj: option {name!r} is not allowed")
        elif name in _KNOB_OVERWRITES:
            raise UnsupportedKernel(
                f"intj: dynamic option {name!r} is overwritten by "
                f"{_KNOB_OVERWRITES[name]!r}; declare that knob path instead"
            )
        elif name in kernel.arg_names:
            if not kernel.params[kernel.arg_names.index(name)].is_constexpr:
                raise UnsupportedKernel(
                    f"intj: dynamic option {name!r} is a runtime kernel parameter; "
                    "only a tl.constexpr parameter can also be a compile option"
                )
    derived = dict(_KNOB_OPTIONS)
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


def _parameter_options(
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
        and p.name not in _FORBIDDEN_OPTIONS
    )


def _split_dynamic(
    names: Sequence[str], values: Sequence[object]
) -> tuple[dict[str, object], dict[str, object]]:
    """Splits one call's dynamic values into (compile options, knob paths)."""
    pairs = list(zip(names, values))
    return (
        {n: v for n, v in pairs if not n.startswith("knobs.")},
        {n: v for n, v in pairs if n.startswith("knobs.")},
    )


def _check_live_knobs(values: Mapping[str, object]) -> None:
    """Checks that each declared knob's passed value is its live one.

    A declared knob only keys: the miss compiles under the live knobs, and
    intj never sets one.

    Raises:
        ValueError: A passed value differs from the knob's current value.
    """
    _imports.ensure_triton_imported()
    for path, passed in values.items():
        _, group, name = path.split(".")
        live = getattr(getattr(_imports.knobs, group), name)
        if passed != live:
            raise ValueError(
                f"intj: dynamic knob {path!r} passed {passed!r} but the current "
                f"value is {live!r}"
            )


def get_full_name(fn: Any) -> str:
    """Returns `fn`'s `<module>.<qualname>`."""
    return f"{fn.__module__}.{fn.__qualname__}"


def _verified_cpython_header() -> str:
    header = cpython_abi.header_for()
    if header is None:
        raise UnsupportedKernel(
            f"intj: no verified CPython layer for python {'%d.%d.%d' % cpython_abi.python_version()} "
            f"(have {', '.join('%d.%d' % v for v in cpython_abi.supported_versions())}); "
            "run `python -m intj.python_intf.check` on it and add it to intj/python_intf/cpython_abi.py"
        )
    return header


def _resolve_torch_access_mode(
    requested: TorchAccessMode | None, *, confirmed: bool = False
) -> TorchAccessMode:
    """Selects a concrete mode from an automatic or explicit request.

    Explicit modes are validated rather than silently downgraded: a caller who
    asked for `RUNTIME_SHIM` because they measured it wants to hear that it is
    unavailable, not to get `INTERPRETER` and wonder where the time went.

    `check()` asks `cxx_lib_dirs`, which spares a first call whose `.so` is on
    disk the cpp_extension import; `confirmed=True` asks `cxx_toolchain`, as a
    build must (`_materialize_module`).  The two differ only where cpp_extension
    is installed but cannot import.
    """

    def cxx() -> bool:
        return (cxx_toolchain() if confirmed else cxx_lib_dirs()) is not None

    if requested is TorchAccessMode.RUNTIME_SHIM and layout_for() is None:
        raise UnsupportedKernel(_unverified_torch_message())
    if requested is TorchAccessMode.STATIC_COMPILE and not cxx():
        raise UnsupportedKernel(
            "intj: STATIC_COMPILE needs a C++ compiler and torch's headers; "
            "set $CXX or pass torch_access_mode=TorchAccessMode.RUNTIME_SHIM"
        )
    if requested is not None:
        return requested
    if cxx():
        return TorchAccessMode.STATIC_COMPILE
    if layout_for() is not None:
        return TorchAccessMode.RUNTIME_SHIM
    return TorchAccessMode.INTERPRETER


def _unverified_torch_message() -> str:
    return (
        f"intj: no verified tensor layout for torch {_torch_version_string()} "
        f"(have {', '.join('%d.%d' % v for v in supported_versions())}), or its "
        "dtypes no longer match the recorded entry, or CPython's object header "
        "makes the tensor offset unrepresentable; pass "
        "torch_access_mode=TorchAccessMode.INTERPRETER, or add an entry to intj/torch_intf/torch_abi.toml "
        "with `python -m intj.torch_intf.abi_detect` on this torch version"
    )


def _torch_version_string() -> str:
    _imports.ensure_torch_imported()
    return str(_imports.torch.__version__)


def _cxx_abi() -> int:
    """Returns torch's `_GLIBCXX_USE_CXX11_ABI`.

    A mismatch here links, then misbehaves at runtime.
    """
    _imports.ensure_torch_imported()
    return int(_imports.torch._C._GLIBCXX_USE_CXX11_ABI)


_LOADED: dict[RenderKey, types.ModuleType] = {}
_LOAD_LOCK = threading.Lock()


def _loaded_module(
    key: RenderKey,
    kernel_module: str,
    layout: TensorABI | None,
    compile_function: Callable[[], Callable[..., Any]],
) -> types.ModuleType:
    """Returns the one module for `key`, loading it once per process.

    `compile_function()` makes the module's compile function (`_intj_compile`),
    called only by the load that installs it.  Reusing the module reuses that
    function, whose compile cache owns the `CompiledKernel`s its launchers'
    records point into -- installing a second one would drop the first,
    freeing kernels whose function handles are still cached. So a hit returns
    the module untouched.

    The lock matters: without it two threads both compile and both load, and the
    loser's module (with its own kernel cache) is silently dropped.
    """
    with _LOAD_LOCK:
        module = _LOADED.get(key)
        if module is None:
            module = load(key, kernel_module)
            if module._header_size() != _stub()._header_size():
                raise ImportError(
                    f"intj: {module.__file__} disagrees with _intj_lazy on the launcher "
                    "header; it was built by another intj -- delete it to rebuild"
                )
            # the module object's dict, written once under _LOAD_LOCK; not C state
            setattr(module, "_intj_compile", compile_function())
            # Pass relative cdata; the compiled setter saves the absolute offset
            # for the torch running now before exposing the module.
            module._set_torch_version(
                torch_version(),
                layout.as_args() if layout else None,
                dtype_index_table(),
            )
            _LOADED[key] = module
        return module


#: options a launch fixes itself; never `options=` or `dynamic_options`
_FORBIDDEN_OPTIONS = ("device", "stream", "device_type", "warp_size")


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
    for key in _FORBIDDEN_OPTIONS:
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


def _spec_key_layout(
    resolved: tuple[ResolvedParam, ...],
    device_binding: DeviceBinding,
    computed0: int = 0,
    dynamic: Sequence[str] = (),
) -> tuple[
    tuple[Param, ...],
    int | None,
    int,
    tuple[tuple[int, int], ...],
    tuple[DynamicSlot, ...],
]:
    """Places the key fields and numbers the arguments in the public call.

    Level-0 computed keys and dynamic values are placed too.

    A computed key is a (value, kind) pair: payload and descriptor, the way a
    constexpr is keyed, so True and 1 key apart.  A dynamic value is keyed
    like an exact key, and its memo follows the parameters'.
    """
    fields = tuple(
        (p.index, field) for p in resolved for field in _key_fields(p.annotation)
    )
    fields += tuple(
        (-1 - j, KeyField(kind, width))
        for j in range(computed0)
        for kind, width in (("payload", 8), ("descriptor", 1))
    )
    fields += tuple(
        (-1 - computed0 - j, field)
        for j, name in enumerate(dynamic)
        for field in _dynamic_key_fields(name)
    )
    layout = _layout_fields(fields, device_binding)
    params: list[Param] = []
    call_index = 0
    memo = 0
    for p in resolved:
        annotation = dataclasses.replace(
            p.annotation,
            key_fields=tuple(
                sorted(
                    (field for index, field in layout.fields if index == p.index),
                    key=lambda field: field.offset,
                )
            ),
        )
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
    placed = {
        (index, field.kind): field.offset for index, field in layout.fields if index < 0
    }
    computed_fields = tuple(
        (placed[(-1 - j, "payload")], placed[(-1 - j, "descriptor")])
        for j in range(computed0)
    )
    slots = tuple(
        DynamicSlot(
            name,
            placed.get((-1 - computed0 - j, "exact")),
            placed.get((-1 - computed0 - j, "exact_kind")),
            memo + j,
        )
        for j, name in enumerate(dynamic)
    )
    return (
        tuple(params),
        layout.device_offset,
        layout.nwords,
        computed_fields,
        slots,
    )


def _render_params(
    resolved: tuple[ResolvedParam, ...], device_binding: DeviceBinding
) -> tuple[tuple[Param, ...], int | None, int]:
    """Places canonical key fields and numbers the arguments in the public call."""
    params, device_offset, nwords, _, _ = _spec_key_layout(resolved, device_binding)
    return params, device_offset, nwords


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


class _Interner:
    """One launcher's object values -> small ids, by canonical value.

    Per launcher, like its kernel cache: an id means one value within one
    launcher and nothing outside it.  Ids live for the process and never enter
    a `RenderKey`.  A JIT function's `cache_key` is read once, when its object
    is first seen, as triton does.  New ids are taken under this table's own
    lock rather than the launcher's C write lock: same effect, and the C side
    never holds a lock while calling Python.

    C asks this table only on a miss in its own object table, which keys
    what it has already answered: a str by content, an instance of
    `dtype_type` by its `name`, anything else by identity.
    """

    def __init__(self) -> None:
        _imports.ensure_triton_imported()
        self._ids: dict[tuple[object, ...], int] = {}
        self._lock = threading.Lock()
        self.dtype_type: type = _imports.tl.dtype  # read once by _init

    def __call__(self, value: object) -> int | None:
        try:
            canonical = _canonical_value(value)
        except ValueError:
            return None  # C raises the TypeError, naming the parameter
        found = self._ids.get(canonical)
        if found is None:
            with self._lock:  # two new values must not both take len(_ids)
                found = self._ids.setdefault(canonical, len(self._ids))
        return found


def _pointer_types(params: Sequence[Param]) -> tuple[tuple[str, int], ...]:
    """Resolves explicit pointer names through the same live dtypes as the ABI table."""
    _imports.ensure_triton_imported()
    wanted = {
        ty[1:]
        for p in params
        for ty in p.annotation.types or ()
        if ty is not None and ty.startswith("*")
    }
    if not wanted:
        return ()
    return tuple(
        sorted(
            ("*" + canonical, code)
            for code, (name, _) in live_dtypes().items()
            if (canonical := _imports.type_canonicalisation_dict.get(name)) in wanted
        )
    )


@functools.lru_cache(maxsize=1)
def _triton_identity() -> tuple[Any, ...]:
    _imports.ensure_triton_imported()
    libtriton = Path(_imports.triton.__file__).parent / "_C" / "libtriton.so"
    stat = libtriton.stat()
    return (_imports.triton.__version__, stat.st_size, stat.st_mtime)


def _triton_specialize(
    value: object, *, backend: Any, is_const: bool, specialize: bool, align: bool
) -> tuple[str, Any]:
    _imports.ensure_triton_imported()
    return _imports.native_specialize_impl(backend, value, is_const, specialize, align)


def _compiler_input(
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


def _checked_compile(
    jit_func: JitFunction,
    compiler_input: CompilerInput,
    target: Any,
    canonical_options: Any,
) -> CompiledKernel:
    """Compiles, loads on the current device, and refuses what intj cannot launch."""
    _imports.ensure_triton_imported()
    # through the module, so a patched `triton.compiler.compile` is the one called
    kernel = _imports.triton.compiler.compile(
        compiler_input.ast_source(jit_func),
        target=target,
        options=canonical_options.__dict__,
    )
    kernel._init_handles()
    md = kernel.metadata
    if md.num_ctas != 1:
        raise UnsupportedKernel("intj: num_ctas > 1 is not supported")
    if md.launch_cooperative_grid:
        raise UnsupportedKernel("intj: launch_cooperative_grid is not supported")
    if getattr(md, "launch_pdl", False):  # nvidia only
        raise UnsupportedKernel("intj: launch_pdl is not supported")
    if getattr(md, "global_scratch_size", 0) or md.profile_scratch_size:
        raise UnsupportedKernel(
            "intj: kernels requiring scratch memory are not supported"
        )
    return kernel


def _refuse_module_compile(*args: Any) -> Any:
    """Refuses a compile; a tuned module's own callback.

    Tuned launches compile through their bound launcher and never read this one.
    """
    del args
    raise RuntimeError("intj: tuned modules compile through their bound launcher")


def override_compile(module: types.ModuleType, fn: Callable[..., Any]) -> Any:
    """Makes `module`'s launchers compile misses with `fn`; for tests and benchmarks.

    `fn` is called as `fn(keyblob, nparams, device, *args)`.

    Returns:
        The previous compile function; restore it with
        `setattr(module, "_intj_compile", previous)`.
    """
    previous = getattr(module, "_intj_compile", None)

    def compile_function(seen: dict[bytes, object], *call: Any) -> Any:
        del seen
        return fn(*call)

    setattr(module, "_intj_compile", compile_function)
    return previous


def _make_compile_callback(
    jit_func: JitFunction,
    params: Sequence[Param],
    options: Mapping[str, Any],
    baked_values: Mapping[int, object] | None = None,
    *,
    return_compiled: bool = False,
    knob_values: Mapping[str, object],
    dynamic: Sequence[str] = (),
) -> Callable[
    ..., tuple[int, int, int, int] | tuple[int, int, int, int, CompiledKernel]
]:
    """Returns the module's compile function.

    It is called on a spec-key miss through the missing launcher's
    `_compile_miss`: `(seen, keyblob, nparams, device, *args)`, `args` being the
    dynamic values, then the public arguments.
    """
    target = backends.current_target()
    backend = _imports.make_backend(target)
    fed = {p.name: p for p in params if p.name in _parameter_options(backend, params)}
    options = {
        **options,
        **{
            n: (baked_values or {})[p.index]
            for n, p in fed.items()
            if p.call_index is None
        },
    }
    per_call = {n: p.call_index for n, p in fed.items() if p.call_index is not None}
    # without per-call values every record has the same options
    fixed_options = (
        None
        if dynamic or per_call
        else backends.canonical_options(
            target, _knob_options(jit_func, options, knob_values)
        )
    )
    # The module's compile cache: a sibling launcher's miss reuses the kernel
    # and only builds its own C record.  With return_compiled the records own
    # their CompiledKernel, so the cache must not keep it alive past them.
    compiled: MutableMapping[tuple[object, ...], CompiledKernel] = (
        weakref.WeakValueDictionary() if return_compiled else {}
    )
    # One compile per input across the module's launchers.  Reentrant: a
    # compile can launch this kernel again (a nested miss on the same thread).
    lock = threading.RLock()

    def compile_callback(
        seen: dict[bytes, object], keyblob: bytes, nparams: int, device: int, *args: Any
    ) -> tuple[int, int, int, int] | tuple[int, int, int, int, CompiledKernel]:
        """Compiles, or reuses, the kernel for one miss.

        `seen` is the calling launcher's own key -> input map.
        """
        current = backends.current_device()
        if device != current:
            # _init_handles() loads the binary on the *current* device;
            # launching that function on another device's stream is a
            # wrong-context launch, so refuse instead.
            raise UnsupportedKernel(
                f"intj: launching on device {device} while device {current} is current; "
                "make the target device current before the first launch"
            )
        values, args = args[: len(dynamic)], args[len(dynamic) :]
        dyn_options, dyn_knobs = _split_dynamic(dynamic, values)
        dyn_options.update({n: args[i] for n, i in per_call.items()})
        _check_live_knobs(dyn_knobs)
        canonical = (
            fixed_options
            if fixed_options is not None
            else backends.canonical_options(
                target,
                _knob_options(
                    jit_func,
                    {**options, **dyn_options},
                    {**knob_values, **dyn_knobs},
                ),
            )
        )
        compiler_input = _compiler_input(
            jit_func, params, args, backend, baked_values=baked_values
        )
        # knobs feeding no option change the binary too
        identity = (compiler_input, canonical.hash(), tuple(sorted(dyn_knobs.items())))
        # b"" is a keyless launcher's one key; seen is per launcher, so that is exact
        if seen.setdefault(keyblob, identity) != identity:
            raise RuntimeError(
                "intj: one spec key maps to two annotated ASTSource inputs; this is an intj bug"
            )
        cache_key = (*identity, current)
        with lock:
            kernel = compiled.get(cache_key)
            if kernel is None:
                fresh = _checked_compile(jit_func, compiler_input, target, canonical)
                # a nested miss may have stored one meanwhile; its record may
                # already be cached, so its kernel is the one that must stay
                kernel = compiled.setdefault(cache_key, fresh)
        md = kernel.metadata
        expected = sum(1 for ty in kernel.src.signature.values() if ty != "constexpr")
        if expected != nparams:
            raise RuntimeError(
                f"intj: packed {nparams} kernel arguments but triton compiled {expected}; "
                "this is an intj bug"
            )
        result = (kernel.function, md.warp_size * md.num_warps, md.shared, nparams)
        return (*result, kernel) if return_compiled else result

    return compile_callback


def _make_host_compile_callback() -> Callable[..., tuple[int, int, int, int]]:
    def compile_callback(
        seen: dict[bytes, object], keyblob: bytes, nparams: int, device: int, *args: Any
    ) -> tuple[int, int, int, int]:
        del seen, keyblob, device, args
        return 0, 1, 0, nparams

    return compile_callback


def triton_specialization(
    jit_func: JitFunction, args: Iterable[Any], options: Mapping[str, Any] | None = None
) -> list[tuple[str, Any]]:
    """Returns the `list[(type_str, key)]` triton would compute for these arguments."""
    binder = jit_func.device_caches[backends.current_device()][4]
    kwargs = _knob_options(jit_func, options or {}, _live_knobs())
    return binder(*args, **kwargs)[1]

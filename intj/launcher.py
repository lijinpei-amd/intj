"""Make the C launcher for a `@triton.jit` kernel; `intj.render` renders and builds it."""

from __future__ import annotations

import dataclasses
import functools
import inspect
import os
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
from .annotation import (
    CanonicalAnnotation,
    DeviceBinding,
    ResolvedParam,
    _applicable,  # pyright: ignore[reportPrivateUsage]  # canonical specialization facts
    _canonical_value,  # pyright: ignore[reportPrivateUsage]  # tagged compiler constants
    _resolve_annotations,  # pyright: ignore[reportPrivateUsage]  # private merge before target lookup
)
from . import kernel_cache as kernel_caches
from .errors import UnsupportedKernel as UnsupportedKernel
from .kernel_cache import KernelCache
from .options import OPTION_FIELDS, MakeLauncherKwargs, MakeLauncherOption
from .render import module as render_module
from .render.build import compiler_path
from .render.key import Param
from .torch_intf.torch_abi import TorchAccessMode

# triton and torch ship no type information, so everything reaching into them is
# typed `Any` on purpose.
JitFunction = Any
CompiledKernel = Any

if TYPE_CHECKING:
    from typing_extensions import Unpack

    from .grid import GridCode
    from .tuning import Tuned


class ClassGlobalWarning(RuntimeWarning):
    """A class global keyed by its qualified name, so replacing it goes undetected.

    Issued by `make_launcher(..., assume_constant_globals=True)`. Silence with
    `warnings.filterwarnings("ignore", category=ClassGlobalWarning)`.
    """


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
    render_module._verified_cpython_header()  # pyright: ignore[reportPrivateUsage]  # shared ABI check
    options = dict(sorted((options or {}).items()))
    chain = kernel
    while _is_chain(kernel):
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

        grid_params, _, _, _, _ = render_module._spec_key_layout(  # pyright: ignore[reportPrivateUsage]  # shared key layout
            resolved, DeviceBinding.NOT_FIXED
        )
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
    access = render_module._resolve_torch_access_mode(torch_access_mode)  # pyright: ignore[reportPrivateUsage]  # shared access checks
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
    rendered, _, _, _, _ = render_module._spec_key_layout(params, device_binding)  # pyright: ignore[reportPrivateUsage]  # shared key layout
    doc = _launch_doc(
        rendered,
        device is not None,
        grid_arg,
        compiled_grid,
        grid_py is not None,
        dynamic=dynamic,
    )
    # a declared knob is per call: leaving its live value out keeps it out
    # of the RenderKey, so it cannot build a module per value
    knob_values = (
        None if no_gpu else {k: v for k, v in _live_knobs().items() if k not in dynamic}
    )
    # GPU-free but may reach the network: after every refusal above, never on a build
    kernel_caches._provision(kernel_cache)  # pyright: ignore[reportPrivateUsage]  # checked toolchain
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
        _compile_state: object

        @property
        def _C_options(self) -> MakeLauncherOption: ...
        @property
        def _C_module(self) -> types.ModuleType | None: ...
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

        One key per lookup level the call walks: an ordinary launcher's chain
        is its spec key alone.  Builds first.
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
        self._check_target(b)
        module = render_module._build_module(b, _stub())  # pyright: ignore[reportPrivateUsage]  # renderer's build entry
        grid_py = () if b.grid_py is None else (b.grid_py, b.hidden)
        state = self._new_compile_state(b)
        address = module._init(self, _Interner(), *grid_py, *b.args)
        self._compile_state = state
        return address, option, b.doc

    def _check_target(self, b: _BuildOptions) -> None:
        """Runs the build's checks that need the target, after every GPU-free one."""
        assert b.tuned is None, "make_launcher gives a chain a TunedLauncher"

    def _new_compile_state(self, b: _BuildOptions) -> object:
        """Returns the Python compile state stored after initialization succeeds."""
        _, target = backends.select(b.no_gpu)
        if target is None:
            return _KernelCompiler(None)
        assert b.knob_values is not None, "GPU builds read knobs at the first call"
        params, _, _, _, _ = render_module._spec_key_layout(b.params, b.device_binding)  # pyright: ignore[reportPrivateUsage]  # shared key layout
        return _KernelCompiler(
            b.jit_func,
            dict(b.compile_options),
            b.knob_values,
            return_compiled=b.return_compiled,
            params=params,
            baked_values={
                p.index: p.baked for p in b.params if p.annotation.baked_value
            },
            dynamic=b.dynamic,
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
        """Returns the record for C to launch on an ordinary cache miss.

        `args` are the dynamic values, then the public kernel arguments.
        """
        del stream, controls
        state = cast(_KernelCompiler, self._compile_state)
        override = state.override
        if override is not None:
            return override(keyblob, nparams, device, *args)
        return state.compile_miss(keyblob, nparams, device, *args)


class LaunchFunction(Protocol):
    """What `make_launcher` returns: a C function whose `__self__` is its `Launcher`."""

    @property
    def __self__(self) -> Launcher: ...
    def __call__(self, *args: Any) -> Any: ...


def _is_chain(kernel: object) -> bool:
    # an autotune/heuristics layer: what make_launcher gives a TunedLauncher
    return type(kernel) in (_imports.Autotuner, _imports.Heuristics)


def _new_launcher(options: MakeLauncherOption) -> LaunchFunction:
    stub, plain = _stub_and_type()
    # `delayed` cannot replace jit_func, so whether it is a chain is fixed now
    chain = _is_chain(options.jit_func)
    return stub.new_launcher(_tuned_type() if chain else plain, options)


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


_stub_loaded: tuple[types.ModuleType, type[Launcher]] | None = None
_tuned_loaded: type[Launcher] | None = None
_STUB_LOCK = threading.Lock()


def _over_stub(base: type[Launcher], stub: types.ModuleType) -> type[Launcher]:
    # Only the concrete class adds storage: both bases must remain layout-compatible.
    return cast(
        "type[Launcher]",
        type(
            base.__name__,
            (base, stub.Launcher),
            {"__slots__": ("_compile_state",), "__module__": base.__module__},
        ),
    )


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
                loaded = _stub_loaded = (stub, _over_stub(Launcher, stub))
    return loaded


def _tuned_type() -> type[Launcher]:
    """Returns `TunedLauncher` over the process's `_intj_lazy`, built on first use."""
    global _tuned_loaded
    tuned = _tuned_loaded
    if tuned is None:
        stub = _stub()
        with _STUB_LOCK:
            tuned = _tuned_loaded
            if tuned is None:
                from .tuning import TunedLauncher  # not at top: tuning imports this

                tuned = _tuned_loaded = _over_stub(TunedLauncher, stub)
    return tuned


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


def _split_call(
    dynamic: Sequence[str], args: tuple[Any, ...]
) -> tuple[dict[str, object], dict[str, object], tuple[Any, ...]]:
    """Splits a miss's arguments into its dynamic options, its knobs, and the rest.

    Raises:
        ValueError: A declared knob was passed other than its live value.
    """
    options, knobs = _split_dynamic(dynamic, args[: len(dynamic)])
    _check_live_knobs(knobs)
    return options, knobs, args[len(dynamic) :]


def _check_device(device: int) -> None:
    # _init_handles() loads the binary on the *current* device; launching that
    # function on another device's stream is a wrong-context launch
    current = backends.current_device()
    if device != current:
        raise UnsupportedKernel(
            f"intj: launching on device {device} while device {current} is current; "
            "make the target device current before the first launch"
        )


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


@dataclasses.dataclass(frozen=True)
class _Identity:
    """What one compile is cached and self-checked by.

    Knobs change the binary even where they feed no option.
    """

    compiler_input: CompilerInput
    options_hash: str
    knobs: tuple[tuple[str, object], ...]

    @classmethod
    def of(
        cls,
        compiler_input: CompilerInput,
        canonical: Any,
        knobs: Mapping[str, object],
    ) -> _Identity:
        return cls(compiler_input, canonical.hash(), tuple(sorted(knobs.items())))


class _KernelCompiler:
    """Compiles `jit_func` for cache misses, once per identity and device.

    An ordinary launcher stores it directly in `_compile_state`; a tuned
    launcher owns one for its candidates. It holds the compiled kernels,
    lock, and ordinary miss metadata, including identity checks and overrides.
    """

    def __init__(
        self,
        jit_func: JitFunction,
        options: Mapping[str, Any] | None = None,
        knob_values: Mapping[str, object] | None = None,
        *,
        return_compiled: bool = False,
        params: Sequence[Param] = (),
        baked_values: Mapping[int, object] | None = None,
        dynamic: Sequence[str] = (),
    ) -> None:
        self.jit_func = jit_func
        self.target = backends.current_target() if jit_func is not None else None
        self.backend = (
            _imports.make_backend(self.target)
            if self.target is not None
            else backends.HOST
        )
        self._options = dict(options or {})
        self._knob_values = dict(knob_values or {})
        # weak where the records own their CompiledKernel (return_compiled):
        # the cache must not keep it alive past them
        self._compiled: MutableMapping[tuple[object, ...], CompiledKernel] = (
            weakref.WeakValueDictionary() if return_compiled else {}
        )
        # Reentrant: a compile can launch this kernel again (a nested miss on
        # the same thread).
        self._lock = threading.RLock()
        self.seen: dict[bytes, object] = {}
        self.override: Callable[..., Any] | None = None
        self._params = params
        self._baked_values = baked_values
        self._return_compiled = return_compiled
        self._dynamic = dynamic
        fed = (
            {
                p.name: p
                for p in params
                if p.name in _parameter_options(self.backend, params)
            }
            if self.target is not None
            else {}
        )
        self._fixed = {
            n: (baked_values or {})[p.index]
            for n, p in fed.items()
            if p.call_index is None
        }
        self._per_call = {
            n: p.call_index for n, p in fed.items() if p.call_index is not None
        }
        # Without per-call values every record has the same options.
        self._fixed_options = (
            self.canonical(self._fixed, {})
            if self.target is not None and not (dynamic or self._per_call)
            else None
        )

    def canonical(self, options: Mapping[str, Any], knobs: Mapping[str, object]) -> Any:
        """Returns the target's options: `options` over the base ones, then knobs."""
        return backends.canonical_options(
            self.target,
            backends._knob_options(  # pyright: ignore[reportPrivateUsage]  # shared compile options
                self.jit_func,
                {**self._options, **options},
                {**self._knob_values, **knobs},
            ),
        )

    def compile(self, identity: _Identity, canonical: Any) -> CompiledKernel:
        """Returns the kernel for `identity` on the current device, compiled once."""
        key = (identity, backends.current_device())
        with self._lock:
            kernel = self._compiled.get(key)
            if kernel is None:
                fresh = _checked_compile(
                    self.jit_func, identity.compiler_input, self.target, canonical
                )
                # a nested miss may have stored one meanwhile; its record may
                # already be cached, so its kernel is the one that must stay
                kernel = self._compiled.setdefault(key, fresh)
        return kernel

    def compile_miss(
        self, keyblob: bytes, nparams: int, device: int, *args: Any
    ) -> Any:
        """Compiles or reuses the kernel for one miss and returns its record."""
        if self.target is None:
            return 0, 1, 0, nparams
        _check_device(device)
        dyn_options, dyn_knobs, args = _split_call(self._dynamic, args)
        canonical = (
            self._fixed_options
            if self._fixed_options is not None
            else self.canonical(
                {
                    **self._fixed,
                    **dyn_options,
                    **{n: args[i] for n, i in self._per_call.items()},
                },
                dyn_knobs,
            )
        )
        compiler_input = _compiler_input(
            self.jit_func,
            self._params,
            args,
            self.backend,
            baked_values=self._baked_values,
        )
        identity = _Identity.of(compiler_input, canonical, dyn_knobs)
        # b"" is a keyless launcher's one key; seen is per launcher.
        if self.seen.setdefault(keyblob, identity) != identity:
            raise RuntimeError(
                "intj: one spec key maps to two annotated ASTSource inputs; this is an intj bug"
            )
        kernel = self.compile(identity, canonical)
        result = _record(kernel, nparams)
        return (*result, kernel) if self._return_compiled else result


def _record(kernel: CompiledKernel, nparams: int) -> tuple[int, int, int, int]:
    """Returns `kernel`'s record for C: function, block_dim, shared, nparams.

    Raises:
        RuntimeError: Triton compiled another argument count than C packed.
    """
    md = kernel.metadata
    expected = sum(1 for ty in kernel.src.signature.values() if ty != "constexpr")
    if expected != nparams:
        raise RuntimeError(
            f"intj: packed {nparams} kernel arguments but triton compiled {expected}; "
            "this is an intj bug"
        )
    return (kernel.function, md.warp_size * md.num_warps, md.shared, nparams)


def override_compile(
    launcher: LaunchFunction, fn: Callable[..., Any]
) -> Callable[[], None]:
    """Overrides one ordinary launcher's misses and returns a restore function.

    For tests and benchmarks. Builds `launcher` if needed. `fn` is called as
    `fn(keyblob, nparams, device, *args)` on subsequent cache misses. Sibling
    launchers are unaffected. The original compiler remains alive so its
    cached kernels' function handles stay valid.

    Returns:
        A function taking no arguments that restores the previous override.

    Raises:
        TypeError: The launcher owns a tuning chain.
    """
    holder = launcher.__self__
    holder.build()
    state = holder._compile_state  # pyright: ignore[reportPrivateUsage]  # launcher internals
    if not isinstance(state, _KernelCompiler):
        raise TypeError("intj: compile overrides require an ordinary launcher")
    previous = state.override
    state.override = fn

    def restore() -> None:
        state.override = previous

    return restore


def triton_specialization(
    jit_func: JitFunction, args: Iterable[Any], options: Mapping[str, Any] | None = None
) -> list[tuple[str, Any]]:
    """Returns the `list[(type_str, key)]` triton would compute for these arguments."""
    binder = jit_func.device_caches[backends.current_device()][4]
    kwargs = backends._knob_options(jit_func, options or {}, _live_knobs())  # pyright: ignore[reportPrivateUsage]  # shared compile options
    return binder(*args, **kwargs)[1]

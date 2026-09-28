"""Render, build and load the C launcher for a `@triton.jit` kernel."""

from __future__ import annotations

import abc
import dataclasses
import functools
import hashlib
import importlib.util
import inspect
import json
import os
import shutil
import subprocess
import sysconfig
import tempfile
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
from typing import TYPE_CHECKING, Any, cast, overload

from . import lazy
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
from .python_intf import cpython_abi
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
    from .grid import GridCode

_RUNTIME = Path(__file__).parent / "runtime"
_ENTRY_TEMPLATE = _RUNTIME / "entry.c.jinja"
_PYTHON_INTF = Path(__file__).parent / "python_intf"


def _runtime_headers() -> bytes:
    """Every header the rendered module can include: the runtime's own, and the
    CPython layer `intj_runtime.h` includes from `python_intf/`."""
    headers = [*sorted(_RUNTIME.glob("*.h")), *sorted(_PYTHON_INTF.glob("*.h"))]
    return b"".join(p.read_bytes() for p in headers)


class UnsupportedKernel(NotImplementedError):
    """Raised for kernels or options outside intj's (deliberately small) scope."""


class ClassGlobalWarning(RuntimeWarning):
    """`make_launcher(..., assume_constant_globals=True)` keyed a class global
    by its qualified name, so replacing that class later goes undetected.

    Silence with `warnings.filterwarnings("ignore", category=ClassGlobalWarning)`.
    """


def _cache_version() -> tuple[int, int]:
    major, minor, *_ = __version__.split(".")
    return int(major), int(minor)


@dataclasses.dataclass(frozen=True)
class Param:
    """One declared kernel parameter, as the template needs it."""

    name: str
    index: int
    call_index: int | None
    annotation: CanonicalAnnotation
    memo: int | None = None  # this slot's intj_memo, if it takes objects


@dataclasses.dataclass(frozen=True)
class TuningRender:
    """What `entry.c.jinja` renders for an autotune/heuristics launcher."""

    levels: int  # lookups per launch
    computed: tuple[int, ...]  # computed keys keyed at each level
    #: level-0 computed keys: (value, kind) byte offsets in the level-0 key
    computed_fields: tuple[tuple[int, int], ...]
    computed_names: tuple[str, ...]  # level-major; the callback's `computed`
    deps: tuple[int, ...]  # dependent values C reads, stored per level
    dep_names: tuple[str, ...]  # level-major; the callback's `deps`
    meta_names: tuple[str, ...]  # grid_py: tuned parameters, declaration order
    source: str  # generated `intj_tuned_level_<n>` functions


@dataclasses.dataclass(frozen=True)
class DynamicSlot:
    """One `dynamic_options` entry as `entry.c.jinja` renders it."""

    name: str  # compile option or `knobs.<group>.<name>`
    value_offset: int | None  # None only under an invariant test's mutation
    kind_offset: int | None
    memo: int  # its intj_memo in the launcher's trailing arrays


@dataclasses.dataclass(frozen=True)
class CompilerInput:
    """The final annotated ASTSource identity, with its original Python constants."""

    signature: tuple[tuple[str, str], ...]
    constants: tuple[tuple[tuple[int, ...], tuple[object, ...]], ...]
    attrs: tuple[tuple[tuple[int, ...], tuple[tuple[str, int], ...]], ...]
    values: tuple[tuple[tuple[int, ...], object], ...] = dataclasses.field(
        compare=False, hash=False, repr=False
    )

    def ast_source(self, jit_func: JitFunction) -> Any:
        from triton.compiler import ASTSource

        source_type = ASTSource
        if jit_func.is_gluon():
            from triton.experimental.gluon._runtime import GluonASTSource

            source_type = GluonASTSource
        return source_type(
            jit_func,
            dict(self.signature),
            dict(self.values),
            {path: [list(attr) for attr in attrs] for path, attrs in self.attrs},
        )


@dataclasses.dataclass(frozen=True)
class RenderContext:
    """Everything `entry.c.jinja` renders from, and nothing else.

    Fixed fields on purpose: a typo fails here instead of silently rendering an
    empty value into C (the template also runs with `StrictUndefined`).
    """

    module_name: str
    kernel_repr: str
    params: tuple[Param, ...]
    nwords: int  # spec-key length, in uint64 words
    device_binding: DeviceBinding
    device_offset: int | None
    verify_annotation: bool
    no_gpu: bool
    max_slots: int  # kernel param slots, upper bound
    spec_pointer_range: int  # 1 if the backend specializes pointers on a 2 GiB range
    driver_path: str
    launch_symbol: str
    device_symbol: str
    error_symbol: str
    error_style: str  # "return" | "outparam"
    torch_access_mode: str  # resolved TorchAccessMode value
    kernel_cache: str  # KernelCache value: which hash map holds the kernels
    #: where that map comes from, when it is not intj's own. In the digest
    #: because the module is built against it.
    cache_include_dirs: tuple[str, ...]
    cache_archives: tuple[str, ...]
    #: STATIC_COMPILE only: what the binary was compiled against. The other modes
    #: discover everything at load, so their `.so` is torch-version-independent
    #: and these stay None -- which is what keeps them out of the digest.
    torch_version: tuple[int, int] | None
    cxx_abi: int | None
    #: The interpreter the module is built for and loaded into.  Every mode reads
    #: CPython internals (see intj/python_intf), so a module is never shared
    #: across versions, patch releases included; the render refuses to compile
    #: against any other `Python.h`.  ABI flags (`t`, `d`) are also in `ext_suffix`.
    python_version: tuple[int, int, int]
    free_threaded: bool
    cpython_static_compile_header: str  # intj/python_intf header for that version
    # Explicit pointer names -> live torch ScalarType codes. Their compact key
    # indices still come from the runtime ABI table, in every access mode.
    pointer_types: tuple[tuple[str, int], ...] = ()
    grid_arg: int | None = None
    grid_py_mode: bool = False
    grid_cpp_source: str = ""
    grid_extra: tuple[str, ...] = ()
    return_compiled: bool = False
    tuning: TuningRender | None = None
    #: `dynamic_options`, in call order: they shape the call and the key
    dynamic: tuple[DynamicSlot, ...] = ()


@dataclasses.dataclass(frozen=True)
class ModuleKey:
    """Everything the rendered `.so` depends on, hashed into its digest.

    Anything that changes the generated code, the compiled binary, or which
    kernel the module's callback compiles has to appear here: two launchers
    whose keys match share one `.so`, so a missing field means a stale module.
    """

    template: str  # entry.c.jinja bytes, hex
    runtime_header: str  # runtime/*.h bytes, hex
    context: RenderContext  # the render is a pure function of this and the template
    cache_key: str  # jit_func.cache_key plus AST language: source and callees
    target: tuple[Any, ...]  # backend, arch, warp size
    options: str  # canonicalized compile options, hashed by triton
    triton: tuple[Any, ...]  # triton version and libtriton identity
    compiler: tuple[str, ...]  # the C compiler triton would invoke
    build_flags: tuple[
        str, ...
    ]  # changes to compiler flags invalidate cached .so files
    ext_suffix: str | None
    intj_version: tuple[int, int]
    #: `assume_constant_globals=True`: the globals the kernel reads, by value.
    #: triton's `cache_key` hashes only the `tl.constexpr` ones; a plain global
    #: (a tuple read by a `constexpr_function`) would otherwise let two
    #: different values share a module -- and its compile callback's kernel.
    global_values: tuple[tuple[str, str, tuple[object, ...]], ...] = ()

    def digest(self) -> str:
        # sort_keys so the digest does not depend on field order
        blob = json.dumps(dataclasses.asdict(self), sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()


@dataclasses.dataclass(frozen=True)
class LauncherFactory:
    jit_func: JitFunction
    params: tuple[ResolvedParam, ...]
    options: tuple[tuple[str, Any], ...]
    torch_access_mode: TorchAccessMode
    kernel_cache: KernelCache
    verify_annotation: bool
    no_gpu: bool
    bind_device_requested: bool
    grid_arg: int | None = None
    grid_py: Callable[[dict[str, object]], object] | None = None
    grid_cpp: GridCode | None = None
    return_compiled: bool = False
    tuned: _Tuned | None = None
    grid_fn: object | None = None
    dynamic: tuple[str, ...] = ()
    global_values: tuple[tuple[str, str, tuple[object, ...]], ...] = ()
    #: `dynamic_options` naming a tl.constexpr parameter: no slot, and only
    #: checked to be a compile option once the target is known
    declared_parameters: tuple[str, ...] = ()

    def bind(self, /, **values: object) -> Callable[..., Any]:
        if self.bind_device_requested:
            raise TypeError("intj: use bind_device(device_ordinal, **values)")
        return self._bind(None, values)

    def bind_device(self, /, *args: object, **values: object) -> Callable[..., Any]:
        if not self.bind_device_requested:
            raise TypeError("intj: bind_device was not requested")
        if len(args) != 1:
            raise TypeError(
                "intj: bind_device requires exactly one positional device ordinal"
            )
        ordinal = args[0]
        # here, not on the first call, so a bad ordinal fails where it is written
        if type(ordinal) is not int or not 0 <= ordinal <= 2**31 - 1:
            raise TypeError("intj: device ordinal must be an int in [0, 2**31)")
        return self._bind(ordinal, values)

    def _binding(self) -> DeviceBinding:
        return (
            DeviceBinding.FIXED
            if self.bind_device_requested
            else DeviceBinding.NOT_FIXED
        )

    def _bind(self, device: object, values: Mapping[str, object]) -> Callable[..., Any]:
        """A lazy launcher.  Everything here is GPU-free; `_build` runs on its first call."""
        names = {p.name for p in self.params if p.annotation.bind_value is not None}
        unknown = values.keys() - names
        if unknown:
            raise TypeError(f"intj: unknown bound parameter(s) {sorted(unknown)}")
        missing = names - values.keys()
        if missing:
            raise TypeError(f"intj: missing bound parameter(s) {sorted(missing)}")
        import torch
        from triton._utils import type_canonicalisation_dict

        resolved: list[ResolvedParam] = []
        for p in self.params:
            annotation = p.annotation
            if annotation.bind_value == "tensor":
                value = values[p.name]
                if value is not None and type(value) not in (
                    torch.Tensor,
                    torch.nn.Parameter,
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
                        tensor = cast(torch.Tensor, value)
                        dtype = type_canonicalisation_dict.get(
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
        args = (*((device,) if self.bind_device_requested else ()), *bound_values)
        params, _, _ = _render_params(state, self._binding())
        doc = _launch_doc(
            params,
            self.bind_device_requested,
            self.grid_arg,
            self.grid_cpp,
            self.grid_py is not None,
            dynamic=self.dynamic,
        )

        def build(header: object) -> int:
            return self._build(header, state, args, hidden)

        nmemo = sum(p.memo is not None for p in params) + len(self.dynamic)
        return _stub().new_launcher(_tail_bytes(nmemo, len(bound_values)), doc, build)

    def _build(
        self,
        header: object,
        resolved: tuple[ResolvedParam, ...],
        args: tuple[object, ...],
        hidden: tuple[object, ...],
    ) -> int:
        """The first call: query the target, render, compile, load, fill `header`.
        Returns the rendered entry's address for `_intj_lazy` to swap in.

        `TRITON_INTERPRET=1` is refused here, not at `make_launcher`, so a
        module-level decorator does not make its module unimportable under the
        interpreter: decoration never runs a kernel, only the first call does.
        Once this call succeeds the swapped-in entry never reads the knob
        again, so a launcher built with the interpreter off keeps launching if
        it is turned on later.
        """
        from triton import knobs

        if knobs.runtime.interpret:
            raise UnsupportedKernel("intj: TRITON_INTERPRET=1 is not supported")
        # a declared knob is per call: leaving its live value out keeps it out
        # of the ModuleKey, so it cannot build a module per value
        knob_values = (
            None
            if self.no_gpu
            else {k: v for k, v in _live_knobs().items() if k not in self.dynamic}
        )
        if self.tuned is not None:
            assert knob_values is not None, "_plan_tuning refuses no_gpu"
            _check_tuned_configs(
                self.tuned.plan, resolved, dict(self.options), knob_values
            )
        module = _materialize_module(
            self.jit_func,
            resolved,
            dict(self.options),
            self.torch_access_mode,
            self.kernel_cache,
            self.no_gpu,
            self.verify_annotation,
            self._binding(),
            grid_arg=self.grid_arg,
            grid_py_mode=self.grid_py is not None,
            grid_cpp=self.grid_cpp,
            return_compiled=self.return_compiled,
            tuning=self.tuned.render if self.tuned else None,
            knob_values=knob_values,
            dynamic=self.dynamic,
            global_values=self.global_values,
            declared_parameters=self.declared_parameters,
        )
        if self.tuned is None:
            callback: Callable[..., Any] = _launcher_compile(module)
        else:
            from .tuning import TunedGrid, make_tuned_callback

            assert self.tuned.render is not None and knob_values is not None
            if self.grid_py is not None:
                grid = TunedGrid("py", self.grid_py)
            elif self.grid_cpp is not None:
                code = getattr(self.grid_fn, "__code__")
                grid = TunedGrid(
                    "cpp",
                    self.grid_fn,
                    tuple(code.co_varnames[: code.co_argcount]),
                    self.grid_cpp.extras,
                )
            else:
                grid = TunedGrid("dims")
            callback = make_tuned_callback(
                self.tuned.plan,
                resolved,
                dict(self.options),
                grid,
                self.tuned.render,
                self.return_compiled,
                knob_values,
                self.dynamic,
            )
        grid_py = () if self.grid_py is None else (self.grid_py, hidden)
        return module.init_bound(header, callback, _Interner(), *grid_py, *args)


def _validate_grid_kwargs(
    dynamic_grid: bool,
    dynamic_options: Sequence[str],
    grid_arg: int | None,
    grid_cpp: object | None,
    grid_py: Callable[[dict[str, object]], object] | None,
    return_compiled: bool,
    no_gpu: bool,
) -> None:
    """The cheap checks on `make_launcher`'s grid/compile kwargs: no imports, no
    GPU, no build. Run eagerly both in the factory form (at decoration time,
    before the kernel exists) and in the direct-call form."""
    if dynamic_grid:
        raise UnsupportedKernel(
            "intj: dynamic_grid is not implemented; pass an int or tuple grid"
        )
    if grid_arg is not None and (
        type(grid_arg) is not int or grid_arg not in (1, 2, 3)
    ):
        raise ValueError("intj: grid_arg must be 1, 2, or 3")
    if sum(value is not None for value in (grid_arg, grid_cpp, grid_py)) > 1:
        raise ValueError("intj: choose only one of grid_arg, grid_cpp and grid_py")
    if grid_py is not None and not callable(grid_py):
        raise TypeError("intj: grid_py must be callable")
    if isinstance(dynamic_options, str):
        raise TypeError("intj: dynamic_options is a sequence of names, not one string")
    if dynamic_options and no_gpu:
        raise UnsupportedKernel(
            "intj: dynamic_options needs GPU mode; no_gpu=True compiles nothing"
        )
    if return_compiled and no_gpu:
        raise UnsupportedKernel("intj: return_compiled=True requires GPU mode")


def _interpreter_deferred(bind_device_requested: bool) -> Any:
    """Stand-in for a launcher over an `InterpretedFunction`.  GPU-free, like a
    real `_bind`: the refusal itself only fires on the first call, so it costs
    nothing beyond that (there is no second call -- this build never succeeds)."""

    def build(header: object) -> int:
        raise UnsupportedKernel("intj: TRITON_INTERPRET=1 is not supported")

    def make() -> Any:
        return _stub().new_launcher(_tail_bytes(0, 0), b"(*args, **kwargs)\x00", build)

    if not bind_device_requested:
        return make()

    class _DeferredFactory:
        def bind(self, /, **values: object) -> Any:
            raise TypeError("intj: use bind_device(device_ordinal, **values)")

        def bind_device(self, /, *args: object, **values: object) -> Any:
            if len(args) != 1:
                raise TypeError(
                    "intj: bind_device requires exactly one positional device ordinal"
                )
            ordinal = args[0]
            if type(ordinal) is not int or not 0 <= ordinal <= 2**31 - 1:
                raise TypeError("intj: device ordinal must be an int in [0, 2**31)")
            return make()

    return _DeferredFactory()


@overload
def make_launcher(
    jit_func: JitFunction,
    *,
    dynamic_grid: bool = False,
    dynamic_options: Sequence[str] = (),
    extra_annotation: Mapping[str, object] | None = None,
    options: Mapping[str, Any] | None = None,
    torch_access_mode: TorchAccessMode | None = None,
    kernel_cache: KernelCache = KernelCache.INTJ,
    no_gpu: bool = False,
    verify_annotation: bool = False,
    bind_device: bool = False,
    grid_arg: int | None = None,
    grid_cpp: object | None = None,
    grid_py: Callable[[dict[str, object]], object] | None = None,
    return_compiled: bool = False,
    assume_constant_globals: bool = False,
) -> Any: ...


@overload
def make_launcher(
    *,
    dynamic_grid: bool = False,
    dynamic_options: Sequence[str] = (),
    extra_annotation: Mapping[str, object] | None = None,
    options: Mapping[str, Any] | None = None,
    torch_access_mode: TorchAccessMode | None = None,
    kernel_cache: KernelCache = KernelCache.INTJ,
    no_gpu: bool = False,
    verify_annotation: bool = False,
    bind_device: bool = False,
    grid_arg: int | None = None,
    grid_cpp: object | None = None,
    grid_py: Callable[[dict[str, object]], object] | None = None,
    return_compiled: bool = False,
    assume_constant_globals: bool = False,
) -> Callable[[Any], Any]: ...


#: make_launcher's "no kernel given" default: a positional None is an error, not
#: the decorator-factory form (an optional kernel whose import failed).
_NO_KERNEL: Any = object()


def make_launcher(
    jit_func: JitFunction = _NO_KERNEL,
    *args: object,
    dynamic_grid: bool = False,
    dynamic_options: Sequence[str] = (),
    extra_annotation: Mapping[str, object] | None = None,
    options: Mapping[str, Any] | None = None,
    torch_access_mode: TorchAccessMode | None = None,
    kernel_cache: KernelCache = KernelCache.INTJ,
    no_gpu: bool = False,
    verify_annotation: bool = False,
    bind_device: bool = False,
    grid_arg: int | None = None,
    grid_cpp: object | None = None,
    grid_py: Callable[[dict[str, object]], object] | None = None,
    return_compiled: bool = False,
    assume_constant_globals: bool = False,
) -> Any:
    """Build a fast launcher for `jit_func`.

    Called without `jit_func` (keyword arguments only), this is a decorator
    factory instead: `make_launcher(grid_cpp=grid)(kernel)` builds the same
    launcher `make_launcher(kernel, grid_cpp=grid)` would, so a kernel (plain
    or wrapped in `triton.autotune`/`triton.heuristics`) can be decorated
    directly:

        @intj.make_launcher(grid_cpp=grid)
        @triton.autotune(...)
        @triton.jit
        def k(...): ...

    The cheap keyword checks below (grid option ranges and exclusivity) run at
    decoration time.  make_launcher does no GPU work: the returned launcher
    builds its module on its first call (see docs/Usage.md, "Lazy build").

    Without bindings, the returned callable is a C function:

        launcher(device, stream, grid, arg0, arg1, ...)

    `device` is a device index in `[0, 256)` -- one byte of the spec key holds
    it -- and `stream` is a raw stream handle (both ints). The default `grid`
    is an int or a tuple/list of up to 3 ints. `grid_arg=1|2|3` takes that many
    separate dimensions; `grid_cpp=annotated_def` computes them in the extension
    from same-name JIT inputs and keyword-only extra inputs; `grid_py=callable`
    calls Python with a fresh dict of JIT inputs on every launch. These three
    keyword-only options are mutually exclusive. The remaining call arguments
    are the kernel's public parameters, positionally, in declaration order.
    Baked Argument and Constexpr values are omitted from the call. With bound
    parameters this returns a non-callable factory; `.bind(**values)` creates the
    native callable and removes those parameters from its signature. `bind_device=True` also
    returns a factory: `.bind_device(ordinal, **values)` fixes a non-negative
    int32 device ordinal and returns `(stream, grid, ...)`. It owns a separate
    kernel cache, or one kernel when no dynamic specialization key remains.
    The return shape depends on this flag and annotations on the JITFunction.

    `options` are triton compile options (`num_warps`, `num_stages`, ...) baked
    into every launch. `torch_access_mode` picks how the module reads a tensor;
    None selects automatically. See `TorchAccessMode`. `kernel_cache` picks the
    hash map behind the kernel cache; see `KernelCache`. `no_gpu=True` decodes
    and caches on the host without compiling or launching a GPU kernel.
    `dynamic_options` names compile options and `knobs.<group>.<name>` paths
    passed per call, right after the grid controls, and keyed; `dynamic_grid`
    is reserved.
    `verify_annotation=True` checks declared types, ranges and assumed facts;
    otherwise these are caller promises.
    `return_compiled=True` returns the cached Triton CompiledKernel after a
    successful launch, including on a zero-volume grid, and requires GPU mode.
    `assume_constant_globals=True` accepts a kernel that reads global
    variables: their values when triton first hashed the kernel become part of
    the module identity, and are never checked again. Changing one afterwards
    is unsupported and not detected.
    """
    if args:
        raise TypeError(
            "intj: make_launcher takes no positional arguments besides the kernel"
        )
    _validate_grid_kwargs(
        dynamic_grid,
        dynamic_options,
        grid_arg,
        grid_cpp,
        grid_py,
        return_compiled,
        no_gpu,
    )
    if jit_func is None:
        raise TypeError("intj: make_launcher got None instead of a kernel")
    if jit_func is _NO_KERNEL:
        return functools.partial(
            make_launcher,
            dynamic_grid=dynamic_grid,
            dynamic_options=dynamic_options,
            extra_annotation=extra_annotation,
            options=options,
            torch_access_mode=torch_access_mode,
            kernel_cache=kernel_cache,
            no_gpu=no_gpu,
            verify_annotation=verify_annotation,
            bind_device=bind_device,
            grid_arg=grid_arg,
            grid_cpp=grid_cpp,
            grid_py=grid_py,
            return_compiled=return_compiled,
            assume_constant_globals=assume_constant_globals,
        )
    kernel: JitFunction = jit_func
    triton_hint = "intj: Triton >=3.7 is required; install `intj[launcher]`"
    try:
        import triton
    except ModuleNotFoundError as e:
        if e.name != "triton":
            raise
        raise UnsupportedKernel(triton_hint) from e
    if tuple(map(int, triton.__version__.split(".")[:2])) < (3, 7):
        raise UnsupportedKernel(f"{triton_hint} (found {triton.__version__})")
    _verified_cpython_header()
    options = dict(sorted((options or {}).items()))
    if no_gpu and options:
        raise UnsupportedKernel(
            "intj: no_gpu=True supports only default compile options"
        )
    from triton.runtime.autotuner import Autotuner, Heuristics
    from triton.runtime.interpreter import InterpretedFunction

    chain = kernel
    while type(kernel) in (Autotuner, Heuristics):
        kernel = kernel.fn
    if isinstance(kernel, InterpretedFunction):
        # `@triton.jit` under `TRITON_INTERPRET=1` returns an InterpretedFunction,
        # not a JITFunction: it has none of the attributes the analysis below
        # needs (`.params`, `.cache_key`, ...), so there is nothing to decorate.
        # Decoration still must not raise -- a module-level `@make_launcher`
        # would make its module unimportable under the interpreter -- so this
        # stands in for the real launcher and defers the actual refusal to the
        # first call, exactly like `_build`'s live `knobs.runtime.interpret`
        # check does for a kernel that decoded to a real JITFunction.
        return _interpreter_deferred(bind_device)
    kernel = _check_kernel(kernel, options, bool(assume_constant_globals))
    global_values = _global_values(kernel) if assume_constant_globals else ()
    resolved = _resolve_annotations(kernel, extra_annotation)
    tuned: _Tuned | None = None
    if chain is not kernel:
        tuned = _plan_tuning(chain, resolved, extra_annotation, options, no_gpu)
        resolved = tuple(
            dataclasses.replace(
                p,
                annotation=dataclasses.replace(
                    p.annotation,
                    tuned=p.name in tuned.plan.tuned,
                    exact_key=p.name in tuned.plan.exact_keys,
                ),
            )
            for p in resolved
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
    declared = tuple(dynamic_options)
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
        render = _tuning_render(tuned.plan, resolved, grid_cpp, grid_py)
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
    needs_binding = bind_device or any(
        p.annotation.bind_value is not None for p in resolved
    )
    if needs_binding and verify_annotation:
        for p in resolved:
            if (
                p.annotation.bind_value == "pointer"
                and p.annotation.pointer_range_32 == "assume"
            ):
                warnings.warn(
                    f"intj: pointer-range assumption for bound pointer {p.name!r} "
                    "cannot be verified from an address and will be trusted",
                    RuntimeWarning,
                    stacklevel=2,
                )
    # GPU-free but may reach the network: here, never on a call
    _provision(kernel_cache)
    factory = LauncherFactory(
        kernel,
        resolved,
        tuple(options.items()),
        access,
        kernel_cache,
        bool(verify_annotation),
        no_gpu,
        bool(bind_device),
        grid_arg,
        grid_py,
        compiled_grid,
        bool(return_compiled),
        tuned,
        grid_cpp,
        dynamic=dynamic,
        global_values=global_values,
        declared_parameters=tuple(n for n in declared if n in kernel.arg_names),
    )
    classes = sorted(
        {".".join(c[1:]) for _, _, v in global_values for c in _class_globals(v)}
    )
    if classes:
        warnings.warn(
            f"intj: {kernel.fn.__qualname__} reads class global(s) {classes}; intj "
            "keys a class by its qualified name and will not detect if it is "
            "replaced (assume_constant_globals=True)",
            ClassGlobalWarning,
            stacklevel=2,
        )
    return factory if needs_binding else factory.bind()


def _materialize_module(
    jit_func: JitFunction,
    resolved: tuple[ResolvedParam, ...],
    options: Mapping[str, Any],
    access: TorchAccessMode,
    kernel_cache: KernelCache,
    no_gpu: bool,
    verify_annotation: bool,
    device_binding: DeviceBinding = DeviceBinding.NOT_FIXED,
    grid_arg: int | None = None,
    grid_py_mode: bool = False,
    grid_cpp: GridCode | None = None,
    return_compiled: bool = False,
    tuning: TuningRender | None = None,
    knob_values: Mapping[str, object] | None = None,
    dynamic: tuple[str, ...] = (),
    global_values: tuple[tuple[str, str, tuple[object, ...]], ...] = (),
    declared_parameters: tuple[str, ...] = (),
) -> types.ModuleType:
    if no_gpu:
        target = None
        backend = None
        canonical_options = None
    else:
        target = _current_target()
        backend = BACKENDS.get(target.backend)
        if backend is None:
            raise UnsupportedKernel(
                f"intj: backend {target.backend!r} is unknown; subclass intj.launcher.Backend and "
                f"register() it (have: {', '.join(sorted(BACKENDS))})"
            )
        assert knob_values is not None, "GPU builds read knobs at the first call"
        canonical_options = _canonical_options(
            target, _knob_options(jit_func, options, knob_values)
        )
        # the names only: each value is canonicalized on its own miss
        _refuse_unknown_options(
            canonical_options,
            [n for n in dynamic if not n.startswith("knobs.")]
            + list(declared_parameters),
        )
    params, device_offset, nwords, computed_fields, slots = _render_key(
        resolved,
        device_binding,
        tuning.computed[0] if tuning is not None else 0,
        dynamic,
    )
    if tuning is not None:
        # final only now: level-0 computed keys are placed with the spec fields
        tuning = dataclasses.replace(tuning, computed_fields=computed_fields)
    # tuned levels always use a map
    nwords = max(nwords, 1) if tuning is not None else nwords
    # make_launcher provisioned already: this is the idempotent, no-network
    # lookup of the toolchain it installed
    cache_toolchain = _provision(kernel_cache)
    # the kernel's own name, unless python allows something C does not
    module_name = jit_func.__name__
    if not (module_name.isidentifier() and module_name.isascii()):
        module_name = "kernel"
    context = RenderContext(
        module_name=module_name,
        kernel_repr=get_full_name(jit_func),
        params=params,
        nwords=nwords,
        device_binding=device_binding,
        device_offset=device_offset,
        verify_annotation=bool(verify_annotation),
        no_gpu=no_gpu,
        max_slots=sum(p.annotation.kind == "argument" for p in params),
        # Where the backend specializes on it, the pointer-range bit is always in
        # the key, even when the knob that emits it is off: a key that cannot tell
        # a > 2 GiB buffer apart would launch a tt.pointer_range=32 binary on it
        # after the knob is flipped back on.
        spec_pointer_range=1 if backend is not None and backend.pointer_range else 0,
        driver_path=backend.library_path() if backend is not None else "",
        launch_symbol=backend.launch_symbol if backend is not None else "",
        device_symbol=backend.device_symbol if backend is not None else "",
        error_symbol=backend.error_symbol if backend is not None else "",
        error_style=backend.error_style if backend is not None else "return",
        torch_access_mode=access.value,
        kernel_cache=kernel_cache.value,
        cache_include_dirs=cache_toolchain["include_dirs"],
        cache_archives=cache_toolchain["archives"],
        # Only the c++ mode compiles anything torch-specific in, so only it has
        # to be rebuilt when torch changes.  Leaving these None for the other
        # two is what keeps their digest -- and so their cached `.so` -- stable
        # across torch versions.
        torch_version=torch_version()
        if access is TorchAccessMode.STATIC_COMPILE
        else None,
        cxx_abi=_cxx_abi() if access is TorchAccessMode.STATIC_COMPILE else None,
        python_version=cpython_abi.python_version(),
        free_threaded=bool(sysconfig.get_config_var("Py_GIL_DISABLED")),
        cpython_static_compile_header=_verified_cpython_header(),
        pointer_types=_pointer_types(params),
        grid_arg=grid_arg,
        grid_py_mode=grid_py_mode,
        grid_cpp_source=grid_cpp.source if grid_cpp else "",
        grid_extra=grid_cpp.extras if grid_cpp else (),
        return_compiled=return_compiled,
        tuning=tuning,
        dynamic=slots,
    )

    # The rendered source is a pure function of the template, the runtime header
    # and the context, so hash those rather than the render itself.
    build = _build_flags(context)
    key = ModuleKey(
        template=_ENTRY_TEMPLATE.read_bytes().hex(),
        runtime_header=_runtime_headers().hex(),
        context=context,
        cache_key=jit_func.cache_key + (":gluon" if jit_func.is_gluon() else ""),
        target=(target.backend, target.arch, target.warp_size)
        if target is not None
        else ("host-only",),
        options=canonical_options.hash()
        if canonical_options is not None
        else "host-defaults",
        triton=_triton_identity(),
        compiler=_compiler_identity(build["language"]),
        build_flags=tuple(build["ccflags"]),
        ext_suffix=sysconfig.get_config_var("EXT_SUFFIX"),
        intj_version=_cache_version(),
        global_values=global_values,
    )
    # the c++ mode gets it too, not to read from but to check its own
    # compiled-in offset against
    layout = (
        layout_for()
        if access in (TorchAccessMode.RUNTIME_SHIM, TorchAccessMode.STATIC_COMPILE)
        else None
    )
    baked_values = {p.index: p.baked for p in resolved if p.annotation.baked_value}
    return _loaded_module(
        key,
        jit_func,
        context,
        params,
        options,
        layout,
        baked_values,
        knob_values,
        dynamic,
    )


_INSTALL_FAILED: dict[KernelCache, Exception] = {}


def _provision(cache: KernelCache) -> dict[str, tuple[str, ...]]:
    """The toolchain for `cache`, fetching and building it the first time.

    Only ever reached from `make_launcher`, on the slow path that was already
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


@functools.lru_cache(maxsize=1)
def _stub() -> types.ModuleType:
    """The process's one `_intj_lazy`, beside the modules intj renders."""
    from triton import knobs

    return lazy.load_stub(
        Path(knobs.cache.get_triton_dir("intj")) / "lazy", _compiler_path("c")
    )


def _tail_bytes(nmemo: int, nbound: int) -> int:
    """sizeof(intj_bound_tail) in intj_runtime.h: 32 for the object table, 16
    per memo and 24 per bound slot, each at least one.  The rendered module
    static_asserts the same."""
    return 32 + 16 * max(nmemo, 1) + 24 * max(nbound, 1)


def module_of(launcher: Callable[..., Any]) -> types.ModuleType:
    """The rendered module behind `launcher`, building it now if its first
    call has not.  For tests and debugging: `module_of(k).spec_key(k, ...)`."""
    return getattr(launcher, "__self__").build()


def _launch_doc(
    params: Sequence[Param],
    fixed_device: bool,
    grid_arg: int | None,
    grid_cpp: GridCode | None,
    grid_py: bool,
    dynamic: Sequence[str] = (),
) -> bytes:
    """The launcher's `__text_signature__`: controls renamed away from kernel names."""
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
    """The knobs triton turns into compile options, as they are now.  Read at
    a launcher's first call and fixed for it."""
    from triton import knobs

    values: dict[str, object] = {}
    for path, _ in _KNOB_OPTIONS:
        _, group, name = path.split(".")
        values[path] = getattr(getattr(knobs, group), name, None)
    return values


def _knob_options(
    jit_func: JitFunction, options: Mapping[str, Any], knob_values: Mapping[str, object]
) -> dict[str, Any]:
    """`options` plus what triton derives from knobs: an explicit `debug`
    overrides the kernel's default, and the runtime knob can still enable it.
    A knob missing from `knob_values` (a declared dynamic one, outside its
    miss) takes triton's default."""
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
    """One dynamic value's key: a kind byte and 8 value bytes, like an exact
    key.  `name` picks nothing here; the invariant tests' mutation checks drop
    one value's fields by name."""
    del name
    return (KeyField("exact_kind", 1), KeyField("exact", 8))


def _check_dynamic(
    dynamic: tuple[str, ...],
    options: Mapping[str, Any],
    kernel: JitFunction,
    tuned: _Tuned | None,
) -> None:
    """GPU-free refusals for `dynamic_options`.  Unknown compile-option names
    wait for the target, in `_materialize_module`."""
    from triton import knobs

    for name in dynamic:
        if type(name) is not str:
            raise TypeError(f"intj: dynamic_options names must be str, got {name!r}")
    if len(set(dynamic)) != len(dynamic):
        raise UnsupportedKernel(
            f"intj: dynamic_options repeats a name: {list(dynamic)}"
        )
    for name in dynamic:
        if name.startswith("knobs."):
            parts = name.split(".")
            group = getattr(knobs, parts[1], None) if len(parts) == 3 else None
            attr = (
                type(group).__dict__.get(parts[2])
                if isinstance(group, knobs.base_knobs)
                else None
            )
            if not (
                isinstance(attr, knobs.env_base)
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
    """The tl.constexpr parameters named like one of `backend`'s compile
    options (`num_warps`, `num_stages`, `waves_per_eu`, ...), which every
    call's value also sets, as Triton's `k[grid](..., num_warps=8)` does.
    Not a key field: the parameter's own exact key already tells every value
    apart.  Options a launch fixes itself stay parameters only."""
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
    """(compile options, knob paths) of one call's dynamic values."""
    pairs = list(zip(names, values))
    return (
        {n: v for n, v in pairs if not n.startswith("knobs.")},
        {n: v for n, v in pairs if n.startswith("knobs.")},
    )


def _check_live_knobs(values: Mapping[str, object]) -> None:
    """A declared knob only keys: its passed value must be the live one,
    because the miss compiles under the live knobs.  intj never sets one."""
    from triton import knobs

    for path, passed in values.items():
        _, group, name = path.split(".")
        live = getattr(getattr(knobs, group), name)
        if passed != live:
            raise ValueError(
                f"intj: dynamic knob {path!r} passed {passed!r} but the current "
                f"value is {live!r}"
            )


def get_full_name(fn: Any) -> str:
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


def _resolve_torch_access_mode(requested: TorchAccessMode | None) -> TorchAccessMode:
    """Select a concrete mode from an automatic or explicit request.

    Explicit modes are validated rather than silently downgraded: a caller who
    asked for `RUNTIME_SHIM` because they measured it wants to hear that it is
    unavailable, not to get `INTERPRETER` and wonder where the time went.
    """
    if requested is TorchAccessMode.RUNTIME_SHIM and layout_for() is None:
        raise UnsupportedKernel(_unverified_torch_message())
    if requested is TorchAccessMode.STATIC_COMPILE and not _cxx_toolchain():
        raise UnsupportedKernel(
            "intj: STATIC_COMPILE needs a C++ compiler and torch's headers; "
            "set $CXX or pass torch_access_mode=TorchAccessMode.RUNTIME_SHIM"
        )
    if requested is not None:
        return requested
    if _cxx_toolchain():
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
    import torch

    return str(torch.__version__)


def _cxx_abi() -> int:
    """How torch was built.  A mismatch here links, then misbehaves at runtime."""
    import torch

    return int(torch._C._GLIBCXX_USE_CXX11_ABI)  # pyright: ignore[reportPrivateUsage]


@functools.lru_cache(maxsize=1)
def _cxx_toolchain() -> tuple[list[str], list[str]] | None:
    """(include dirs, library dirs) for the c++ mode, or None if unavailable.

    Both come from the loaded torch rather than from `torch.__file__`, so a torch
    that reorganises its tree, or one built out of tree, keeps working.
    """
    try:
        _compiler_path("c++")
    except Exception:
        return None
    try:
        from torch.utils import cpp_extension
    except Exception:  # pragma: no cover - torch without the build helpers
        return None
    includes = [p for p in cpp_extension.include_paths() if os.path.isdir(p)]
    libs = [p for p in cpp_extension.library_paths() if os.path.isdir(p)]
    header = "torch/csrc/autograd/python_variable.h"
    if not libs or not any(os.path.exists(os.path.join(d, header)) for d in includes):
        return None
    return (includes, libs)


_LOADED: dict[ModuleKey, types.ModuleType] = {}
_LOAD_LOCK = threading.Lock()


def _loaded_module(
    key: ModuleKey,
    jit_func: JitFunction,
    context: RenderContext,
    params: Sequence[Param],
    options: Mapping[str, Any],
    layout: TensorABI | None,
    baked_values: Mapping[int, object],
    knob_values: Mapping[str, object] | None = None,
    dynamic: tuple[str, ...] = (),
) -> types.ModuleType:
    """One module per `ModuleKey`, for the life of the process.

    Reusing the module reuses its compile function (`_intj_compile`), whose
    compile cache owns the `CompiledKernel`s its launchers' records point
    into -- installing a second one would drop the first, freeing kernels
    whose function handles are still cached. So a hit returns the module
    untouched.

    The lock matters: without it two threads both compile and both load, and the
    loser's module (with its own kernel cache) is silently dropped.
    """
    with _LOAD_LOCK:
        module = _LOADED.get(key)
        if module is None:
            module = _load(key, jit_func, context)
            if module.header_size() != _stub().header_size():
                raise ImportError(
                    f"intj: {module.__file__} disagrees with _intj_lazy on the launcher "
                    "header; it was built by another intj -- delete it to rebuild"
                )
            # the module object's dict, written once under _LOAD_LOCK; not C state
            setattr(
                module,
                "_intj_compile",
                _refuse_module_compile
                if context.tuning is not None
                else _make_host_compile_callback()
                if context.no_gpu
                else _make_compile_callback(
                    jit_func,
                    params,
                    options,
                    baked_values,
                    return_compiled=context.return_compiled,
                    knob_values=knob_values or {},
                    dynamic=dynamic,
                ),
            )
            # Pass relative cdata; the compiled setter saves the absolute offset
            # for the torch running now before exposing the module.
            module.set_torch_version(
                torch_version(),
                layout.as_args() if layout else None,
                dtype_index_table(),
            )
            _LOADED[key] = module
        return module


def _load(
    key: ModuleKey, jit_func: JitFunction, context: RenderContext
) -> types.ModuleType:
    """Load the extension for this key, rendering and building it only if needed.

    The `.so` lands in `$TRITON_HOME/.triton/intj/<digest>/<module>/<kernel>`, so the
    artifact says where the kernel came from and which build it is. Its leaf name
    is the kernel's own name, because CPython derives `PyInit_<leaf>` from the
    last dotted component of the spec name -- that is also what `perf` and
    /proc/<pid>/maps show.

    An existing `.so` is loaded as-is: its path already encodes the digest, so
    rendering the source again would only reproduce what was compiled from it.
    """
    from triton import knobs

    suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
    # a sibling of triton's own cache
    directory = (
        Path(knobs.cache.get_triton_dir("intj")) / key.digest() / jit_func.__module__
    )
    so_path = directory / f"{context.module_name}{suffix}"
    if not so_path.exists():
        _build(so_path, context)

    # Loading by hand, rather than through import_module, keeps the module out of
    # sys.modules.
    spec = importlib.util.spec_from_file_location(context.module_name, so_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"intj: cannot load {so_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # pyright: ignore[reportAttributeAccessIssue]  # 3.8 stubs lack it
    return module


def _build(so_path: Path, context: RenderContext) -> None:
    """Render and compile, then install both artifacts next to each other."""
    import jinja2

    template = jinja2.Template(
        _ENTRY_TEMPLATE.read_text(), undefined=jinja2.StrictUndefined
    )
    src = template.render(**dataclasses.asdict(context))
    flags = _build_flags(context)
    binary = _compile_so_bytes(src, context.module_name, flags)
    so_path.parent.mkdir(parents=True, exist_ok=True)
    # The source is kept next to the binary: it is what you read when a launch
    # misbehaves, and what you recompile by hand to debug it.
    suffix = ".cpp" if flags["language"] == "c++" else ".c"
    _install(src.encode(), so_path.with_name(f"{context.module_name}{suffix}"))
    _install(binary, so_path)


def _compile_so_bytes(src: str, name: str, flags: Mapping[str, Any]) -> bytes:
    """Use Triton's compiler, or its 3.7-era command when the helper is absent."""
    from triton.runtime import build

    compile_so = getattr(build, "compile_so_from_src", None)
    if compile_so is not None:
        return Path(compile_so(src=src, name=name, **flags)).read_bytes()

    from triton import knobs

    language = flags["language"]
    with tempfile.TemporaryDirectory() as directory:
        source = Path(directory) / f"{name}{'.cpp' if language == 'c++' else '.c'}"
        source.write_text(src)
        suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
        binary = Path(directory) / f"{name}{suffix}"
        includes = [
            *flags.get("include_dirs", ()),
            directory,
            lazy.python_include(),
            *knobs.build.backend_dirs,
        ]
        command = [
            _compiler_path(language),
            str(source),
            "-O3",
            "-shared",
            "-fPIC",
            "-Wno-psabi",
            "-o",
            str(binary),
        ]
        command += [f"-l{lib}" for lib in flags.get("libraries", ())]
        command += [f"-L{path}" for path in flags.get("library_dirs", ())]
        command += [f"-I{path}" for path in includes if path is not None]
        command += list(flags.get("ccflags", ()))
        subprocess.check_call(command, stdout=subprocess.DEVNULL)
        return binary.read_bytes()


def _runtime_header_flag() -> str:
    """Make the runtime headers part of what triton's compile cache keys on.

    `_compile_so` keys on the source bytes plus the *names* of the include
    directories, not on the contents of the headers found there.  So an edit to
    a runtime header moves intj's own `ModuleKey` digest, intj re-renders, and
    triton then serves the object it compiled from the previous headers.  Feeding
    the digest in as a define puts it in triton's key too.
    """
    digest = hashlib.sha256(_runtime_headers()).hexdigest()[:16]
    return f"-DINTJ_RUNTIME_HEADER=0x{digest}"


def _build_flags(context: RenderContext) -> dict[str, Any]:
    """Compiler arguments beyond intj's own include directory.

    The c++ access mode needs torch's headers, and `libc10` for the handful of
    error-path symbols `THPVariable_Unpack` pulls in.  Those are the reason it
    cannot be header-only -- torch loads libc10 `RTLD_LOCAL`, so an unlinked
    module builds fine and then fails at import with an undefined
    `throw_data_ptr_access_error`.  libtorch_cpu and libtorch_python are *not*
    needed.

    A kernel cache other than intj's is a C++ map, so it drags the whole module
    into C++ even where the tensor reader would not have.
    """
    torch_static_compile = (
        context.torch_access_mode == TorchAccessMode.STATIC_COMPILE.value
    )
    cxx_cache = context.kernel_cache != KernelCache.INTJ.value
    # No -falign-loops=64 (tests/test_kernel_cache.py uses it): measured
    # 2026-09-27, it leaves the launch path's instruction count unchanged -- the
    # memo-hit path runs no loop -- and moves rows +-1 ns either way by layout
    # alone.
    flags: dict[str, Any] = {
        "include_dirs": [str(_RUNTIME), str(_PYTHON_INTF), *context.cache_include_dirs],
        "ccflags": [
            _runtime_header_flag(),
            f"-DINTJ_CACHE_{context.kernel_cache.upper()}",
        ],
    }
    if context.cache_archives:
        # abseil's own link order is not intj's to encode, so the archives go in
        # a group and the linker sorts it out.
        flags["ccflags"] += [
            "-Wl,--start-group",
            *context.cache_archives,
            "-Wl,--end-group",
        ]
    if not torch_static_compile and not cxx_cache:
        return {"language": "c", **flags}

    flags["language"] = "c++"
    # triton puts its own -std=c++17 early and appends ccflags last, so this
    # wins.  torch >= 2.14 needs c++20 to compile warning-clean.
    flags["ccflags"][:0] = ["-std=c++20", "-fvisibility=hidden"]
    if not torch_static_compile:
        return flags

    toolchain = _cxx_toolchain()
    assert toolchain is not None, "make_launcher validated this"
    includes, libs = toolchain
    flags["include_dirs"] += includes
    flags["library_dirs"] = list(libs)
    flags["libraries"] = ["c10"]
    flags["ccflags"] += [
        # torch's headers make the translation unit ~7x bigger, and g++'s
        # inlining budget is per unit: past a size threshold it stops inlining
        # the tensor reader and PyFloat_AS_DOUBLE into the decode, costing a
        # real call per argument.  Measured: ~5 ns per launch on a 3-tensor
        # kernel.  Nothing here is about the torch code itself.
        f"-D_GLIBCXX_USE_CXX11_ABI={context.cxx_abi}",
        *(f"-Wl,-rpath,{d}" for d in libs),
    ]
    return flags


def _install(content: bytes, path: Path) -> None:
    """Write `content` to `path` atomically, so a racing build cannot be half-read."""
    staged = path.with_name(f".{path.name}.{os.getpid()}")
    staged.write_bytes(content)
    os.replace(staged, path)


class Backend(abc.ABC):
    """How intj reaches one triton backend's driver API.

    Everything backend-specific lives in a subclass; nothing else in intj, and
    nothing in the template, branches on a backend name.  Supporting another
    triton backend is a subclass plus a `register()` call, provided its driver
    exposes a `cuLaunchKernel`-shaped launch entry point.
    """

    #: triton target backend name, i.e. `get_current_target().backend`
    name: str
    #: f(fn, gx,gy,gz, bx,by,bz, shared, stream, params, extra) -> error code
    launch_symbol: str
    #: f(int32_t *device, int32_t ordinal) -> error code
    device_symbol: str
    #: error-string lookup, called as `error_style` says
    error_symbol: str
    #: "return"   const char *f(int)
    #: "outparam" int f(int, const char **)
    error_style: str
    #: backend specializes pointers on a 2 GiB range (AMD's "S" spec bit)
    pointer_range: bool

    @abc.abstractmethod
    def library_path(self) -> str:
        """The driver dylib to dlopen -- the same one triton itself uses."""


class HipBackend(Backend):
    name = "hip"
    launch_symbol = "hipModuleLaunchKernel"
    device_symbol = "hipDeviceGet"
    error_symbol = "hipGetErrorString"
    error_style = "return"
    pointer_range = True

    def library_path(self) -> str:
        # private, but it is the resolution order triton itself launches through
        from triton.backends.amd import driver

        return driver._get_path_to_hip_runtime_dylib()  # pyright: ignore[reportPrivateUsage]


class CudaBackend(Backend):
    name = "cuda"
    launch_symbol = "cuLaunchKernel"
    device_symbol = "cuDeviceGet"
    error_symbol = "cuGetErrorString"
    error_style = "outparam"
    pointer_range = False

    def library_path(self) -> str:
        return "libcuda.so.1"  # already mapped by torch, resolved through the normal search path


BACKENDS: dict[str, Backend] = {}


def register(backend: Backend) -> Backend:
    """Make `backend` usable by `make_launcher`, replacing any same-named one."""
    BACKENDS[backend.name] = backend
    return backend


register(HipBackend())
register(CudaBackend())


def _current_target() -> Any:
    """The active triton target, which triton types as optional."""
    from triton.runtime.driver import driver

    target = driver.active.get_current_target()
    if target is None:
        raise UnsupportedKernel("intj: no active triton target; is a GPU visible?")
    return target


def _current_device() -> int:
    from triton.runtime.driver import driver

    # get_current_device() is on every concrete driver, just not on DriverBase
    return driver.active.get_current_device()  # pyright: ignore[reportAttributeAccessIssue]


def _canonical_options(target: Any, options: Mapping[str, Any]) -> Any:
    """Run `options` through the triton compiler backend's `parse_options`.

    That fills in defaults and normalizes the odd fields (`extern_libs`,
    `llvm_fn_attrs`, `warp_size`), so `{}` and `{"num_warps": 4}` reach the same
    rendered module instead of building it twice.  It also rejects what the
    backend rejects, at `make_launcher` time rather than on the first launch.

    `parse_options` ignores keys it does not know, so unknown ones are caught
    here -- otherwise a typo would silently compile with the default.
    """
    from triton.compiler import make_backend

    parsed: Any = make_backend(target).parse_options(dict(options))
    _refuse_unknown_options(parsed, options)
    return parsed


def _refuse_unknown_options(parsed: Any, names: Iterable[str]) -> None:
    """Refuse the names that are no field of `parsed` (triton's options)."""
    unknown = set(names) - {f.name for f in dataclasses.fields(parsed)}
    if unknown:
        raise UnsupportedKernel(f"intj: unknown compile option(s) {sorted(unknown)}")


#: options a launch fixes itself; never `options=` or `dynamic_options`
_FORBIDDEN_OPTIONS = ("device", "stream", "device_type", "warp_size")


def _check_kernel(
    jit_func: JitFunction,
    options: Mapping[str, Any],
    assume_constant_globals: bool = False,
) -> JitFunction:
    from triton.runtime.jit import JITFunction

    if not isinstance(jit_func, JITFunction):
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
    import triton.language as tl

    if isinstance(value, tl.constexpr):
        return ("constexpr", _canonical_global(value.value))
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
    """The `("class", module, qualname)` entries in a `_canonical_global` result."""
    if canonical[0] == "class":
        yield cast(tuple[str, str, str], canonical)
    elif canonical[0] in ("constexpr", "tuple"):
        for item in canonical[1:]:
            yield from _class_globals(cast(tuple[object, ...], item))


def _global_values(
    jit_func: JitFunction,
) -> tuple[tuple[str, str, tuple[object, ...]], ...]:
    """The globals `jit_func` reads, as triton snapshotted them when it first
    computed `cache_key`, canonicalized for the `ModuleKey`."""
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


@dataclasses.dataclass(frozen=True)
class _Tuned:
    plan: Any  # tuning.TuningPlan
    render: TuningRender | None = None  # filled once the grid is known


def _plan_tuning(
    chain: Any,
    resolved: tuple[ResolvedParam, ...],
    extra_annotation: Mapping[str, object] | None,
    options: Mapping[str, Any],
    no_gpu: bool,
) -> _Tuned:
    from .tuning import analyze

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
    return _Tuned(plan)


def _config_options(
    plan: Any, resolved: tuple[ResolvedParam, ...]
) -> Iterable[dict[str, Any]]:
    """Each autotune config's compile options: its kwargs that name no parameter."""
    from triton.runtime.autotuner import Autotuner

    names = {p.name for p in resolved}
    for layer in plan.layers:
        if type(layer) is Autotuner:
            for config in layer.configs:
                yield {
                    str(k): v for k, v in config.all_kwargs().items() if k not in names
                }


def _check_tuned_configs(
    plan: Any,
    resolved: tuple[ResolvedParam, ...],
    options: Mapping[str, Any],
    knob_values: Mapping[str, object],
) -> None:
    """Every config's options through the target's `parse_options`: on the
    first call, because the target is a GPU query."""
    target = _current_target()
    for config_options in _config_options(plan, resolved):
        _canonical_options(
            target,
            _knob_options(plan.jit_func, {**options, **config_options}, knob_values),
        )


def _tuning_render(
    plan: Any,
    resolved: tuple[ResolvedParam, ...],
    grid_cpp: object | None,
    grid_py: object | None,
) -> TuningRender:
    """Dep and comp slots for the grid and the lowered heuristics, level-major.

    `computed_fields` is left empty: `_materialize_module` places it."""
    from triton.runtime.autotuner import Autotuner

    from .heuristic import HeuristicError, Source, lower
    from .tuning import c_scalar

    by_name = {p.name: p for p in _render_params(resolved, DeviceBinding.NOT_FIXED)[0]}
    grid_code = getattr(grid_cpp, "__code__", None)
    grid_reads: tuple[str, ...] = (
        grid_code.co_varnames[: grid_code.co_argcount] if grid_code else ()
    )
    levels: int = plan.levels
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
        if type(layer) is Autotuner:
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


def _render_key(
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
    """Place key fields, level-0 computed keys and dynamic values included,
    and number the call.

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
    """Place canonical key fields and number the arguments in the public call."""
    params, device_offset, nwords, _, _ = _render_key(resolved, device_binding)
    return params, device_offset, nwords


def _object_capable(annotation: CanonicalAnnotation) -> bool:
    """A public constexpr keyed by descriptor + 8-byte payload takes objects:
    str, tl.dtype and JIT functions key as INTJ_B_CX_OBJECT plus an id."""
    return (
        annotation.kind == "constexpr"
        and annotation.types is None
        and not annotation.power_of_two_or_zero
    )


class _Interner:
    """One launcher's object values -> small ids, by canonical value.

    Per launcher, like its kernel cache: an id means one value within one
    launcher and nothing outside it.  Ids live for the process and never enter
    a `ModuleKey`.  A JIT function's `cache_key` is read once, when its object
    is first seen, as triton does.  New ids are taken under this table's own
    lock rather than the launcher's C write lock: same effect, and the C side
    never holds a lock while calling Python.

    C asks this table only on a miss in its own object table, which keys
    what it has already answered: a str by content, an instance of
    `dtype_type` by its `name`, anything else by identity."""

    def __init__(self) -> None:
        import triton.language as tl

        self._ids: dict[tuple[object, ...], int] = {}
        self._lock = threading.Lock()
        self.dtype_type: type = tl.dtype  # read once by init_bound

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
    """Resolve explicit pointer names through the same live dtypes as the ABI table."""
    from triton._utils import type_canonicalisation_dict

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
            if (canonical := type_canonicalisation_dict.get(name)) in wanted
        )
    )


@functools.lru_cache(maxsize=1)
def _triton_identity() -> tuple[Any, ...]:
    import triton

    libtriton = Path(triton.__file__).parent / "_C" / "libtriton.so"
    stat = libtriton.stat()
    return (triton.__version__, stat.st_size, stat.st_mtime)


@functools.lru_cache(maxsize=2)
def _compiler_path(language: str) -> str:
    from triton.runtime import build

    find_compiler = getattr(build, "_find_compiler", None)
    if find_compiler is not None:
        cc = find_compiler(language)
        return str(cc[0] if isinstance(cc, tuple) else cc)
    variable, candidates = (
        ("CC", ("gcc", "clang")) if language == "c" else ("CXX", ("g++", "clang++"))
    )
    configured = os.environ.get(variable)
    if configured is not None:
        return configured
    for candidate in candidates:
        found = shutil.which(candidate)
        if found is not None:
            return found
    raise RuntimeError(f"Failed to find {language} compiler; set {variable}")


@functools.lru_cache(maxsize=2)
def _compiler_identity(language: str = "c") -> tuple[str, ...]:
    cc = _compiler_path(language)
    try:
        version = subprocess.run(
            [cc, "--version"], capture_output=True, text=True
        ).stdout.splitlines()[0]
    except Exception:  # pragma: no cover - compiler without --version
        version = ""
    return (str(cc), version)


def _triton_specialize(
    value: object, *, backend: Any, is_const: bool, specialize: bool, align: bool
) -> tuple[str, Any]:
    from triton._C.libtriton import native_specialize_impl

    return native_specialize_impl(backend, value, is_const, specialize, align)


def _compiler_input(
    jit_func: JitFunction,
    params: Sequence[Param],
    public_args: Sequence[object],
    backend: Any,
    *,
    baked_values: Mapping[int, object] | None = None,
) -> CompilerInput:
    signature: list[tuple[str, str]] = []
    constants: list[tuple[tuple[int, ...], tuple[object, ...]]] = []
    attrs: list[tuple[tuple[int, ...], tuple[tuple[str, int], ...]]] = []
    values: list[tuple[tuple[int, ...], object]] = []
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
                    from triton.runtime.jit import MockTensor

                    # Both public null spellings have pointer 0 and storage range 0.
                    inference_value = MockTensor(effective[1:])
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
        if ty == "constexpr":
            constants.append((path, _canonical_value(value)))
            values.append((path, value))
        if desc:
            parsed = tuple((name, amount) for name, amount in backend.parse_attr(desc))
            if parsed:
                attrs.append((path, parsed))
    return CompilerInput(
        tuple(signature), tuple(constants), tuple(attrs), tuple(values)
    )


def _checked_compile(
    jit_func: JitFunction,
    compiler_input: CompilerInput,
    target: Any,
    canonical_options: Any,
) -> CompiledKernel:
    """Compile, load on the current device, and refuse what intj cannot launch."""
    from triton.compiler import compile as triton_compile

    kernel = triton_compile(
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
    """A tuned module's own callback: tuned launches compile through their bound
    launcher and never read this one."""
    del args
    raise RuntimeError("intj: tuned modules compile through their bound launcher")


def _launcher_compile(module: types.ModuleType) -> Callable[..., Any]:
    """One launcher's miss callback: its own `seen` map, then the module's
    compile function, looked up per miss so `override_compile` reaches built
    launchers too."""
    seen: dict[bytes, object] = {}

    def callback(*call: Any) -> Any:
        return getattr(module, "_intj_compile")(seen, *call)

    return callback


def override_compile(module: types.ModuleType, fn: Callable[..., Any]) -> Any:
    """Tests and benchmarks: compile misses of `module`'s launchers with
    `fn(keyblob, nparams, device, *args)`.  Returns the previous function;
    restore it with `setattr(module, "_intj_compile", previous)`."""
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
    """The module's compile function, called from C on a spec-key miss through
    the missing launcher's callback: `(seen, keyblob, nparams, device, *args)`,
    `args` being the dynamic values, then the public arguments."""
    from triton.compiler import make_backend

    target = _current_target()
    backend = make_backend(target)
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
        else _canonical_options(target, _knob_options(jit_func, options, knob_values))
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
        """`seen` is the calling launcher's own key -> input map."""
        current = _current_device()
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
            else _canonical_options(
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
    """The `list[(type_str, key)]` triton would compute for these arguments."""
    binder = jit_func.device_caches[_current_device()][4]
    kwargs = _knob_options(jit_func, options or {}, _live_knobs())
    return binder(*args, **kwargs)[1]

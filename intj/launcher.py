"""Render, build and load the C launcher for a `@triton.jit` kernel."""

import abc
import dataclasses
import functools
import hashlib
import importlib.util
import inspect
import json
import os
import subprocess
import sysconfig
import threading
import types
import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from ._version import __version__
from .annotation import (
    CanonicalAnnotation,
    DeviceBinding,
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
from .torch_abi import (
    TensorABI,
    TorchAccess,
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

_RUNTIME = Path(__file__).parent / "runtime"
_ENTRY_TEMPLATE = _RUNTIME / "entry.c.jinja"
_RUNTIME_HEADER = _RUNTIME / "intj_runtime.h"


class UnsupportedKernel(NotImplementedError):
    """Raised for kernels or options outside intj's (deliberately small) scope."""


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

        return ASTSource(
            jit_func, dict(self.signature), dict(self.values),
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
    error_symbol: str
    error_style: str  # "return" | "outparam"
    torch_access: str  # TorchAccess value; never "auto" by this point
    kernel_cache: str  # KernelCache value: which hash map holds the kernels
    #: where that map comes from, when it is not intj's own. In the digest
    #: because the module is built against it.
    cache_include_dirs: tuple[str, ...]
    cache_archives: tuple[str, ...]
    #: `cxx` only: what the binary was compiled against.  The other modes
    #: discover everything at load, so their `.so` is torch-version-independent
    #: and these stay None -- which is what keeps them out of the digest.
    torch_version: tuple[int, int] | None
    cxx_abi: int | None
    # Explicit pointer names -> live torch ScalarType codes. Their compact key
    # indices still come from the runtime ABI table, in every access mode.
    pointer_types: tuple[tuple[str, int], ...] = ()


@dataclasses.dataclass(frozen=True)
class ModuleKey:
    """Everything the rendered `.so` depends on, hashed into its digest.

    Anything that changes the generated code, the compiled binary, or which
    kernel the module's callback compiles has to appear here: two launchers
    whose keys match share one `.so`, so a missing field means a stale module.
    """

    template: str  # entry.c.jinja bytes, hex
    runtime_header: str  # intj_runtime.h bytes, hex
    context: RenderContext  # the render is a pure function of this and the template
    cache_key: str  # jit_func.cache_key: the kernel source and its callees
    target: tuple[Any, ...]  # backend, arch, warp size
    options: str  # canonicalized compile options, hashed by triton
    triton: tuple[Any, ...]  # triton version and libtriton identity
    compiler: tuple[str, ...]  # the C compiler triton would invoke
    ext_suffix: str | None
    intj_version: tuple[int, int]

    def digest(self) -> str:
        # sort_keys so the digest does not depend on field order
        blob = json.dumps(dataclasses.asdict(self), sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()


@dataclasses.dataclass(frozen=True)
class LauncherFactory:
    jit_func: JitFunction
    params: tuple[ResolvedParam, ...]
    options: tuple[tuple[str, Any], ...]
    torch_access: TorchAccess
    kernel_cache: KernelCache
    verify_annotation: bool
    no_gpu: bool
    bind_device_requested: bool

    def bind(self, /, **values: object) -> Callable[..., None]:
        if self.bind_device_requested:
            raise TypeError("intj: use bind_device(device_ordinal, **values)")
        return self._bind(None, values)

    def bind_device(self, /, *args: object, **values: object) -> Callable[..., None]:
        if not self.bind_device_requested:
            raise TypeError("intj: bind_device was not requested")
        if len(args) != 1:
            raise TypeError("intj: bind_device requires exactly one positional device ordinal")
        return self._bind(args[0], values)

    def _bind(self, device: object, values: Mapping[str, object]) -> Callable[..., None]:
        names = {p.name for p in self.params if p.annotation.bind_value is not None}
        unknown = values.keys() - names
        if unknown:
            raise TypeError(f"intj: unknown bound parameter(s) {sorted(unknown)}")
        missing = names - values.keys()
        if missing:
            raise TypeError(f"intj: missing bound parameter(s) {sorted(missing)}")
        if device is not None:
            raise UnsupportedKernel("intj: device binding is not implemented")

        import torch
        from triton._utils import type_canonicalisation_dict

        resolved: list[ResolvedParam] = []
        for p in self.params:
            annotation = p.annotation
            if annotation.bind_value == "tensor":
                value = values[p.name]
                if value is not None and type(value) not in (torch.Tensor, torch.nn.Parameter):
                    raise TypeError(f"intj: bound tensor {p.name!r} must be a tensor or None")
                if annotation.types is None:
                    ty = None
                    if value is not None:
                        tensor = cast(torch.Tensor, value)
                        dtype = type_canonicalisation_dict.get(str(tensor.dtype).split(".")[-1])
                        if dtype is None:
                            raise TypeError(f"intj: bound tensor {p.name!r} has unsupported dtype {tensor.dtype}")
                        ty = "*" + dtype
                    annotation = dataclasses.replace(annotation, types=(ty,))
            resolved.append(dataclasses.replace(p, annotation=annotation))
        module = _materialize_module(
            self.jit_func, tuple(resolved), dict(self.options), self.torch_access,
            self.kernel_cache, self.no_gpu, self.verify_annotation,
        )
        signature = inspect.Signature(tuple(
            inspect.Parameter(name, inspect.Parameter.POSITIONAL_ONLY)
            for name in ("device", "stream", "grid", *(
                p.name for p in resolved if p.annotation.bind_value is None and not p.annotation.baked_value
            ))
        ))
        return module.make_bound(signature, *(values[p.name] for p in resolved if p.name in names))


def make_launcher(
    jit_func: JitFunction,
    dynamic_grid: bool = False,
    dynamic_options: Sequence[str] = (),
    extra_annotation: Mapping[str, object] | None = None,
    options: Mapping[str, Any] | None = None,
    torch_access: TorchAccess = TorchAccess.AUTO,
    kernel_cache: KernelCache = KernelCache.INTJ,
    no_gpu: bool = False,
    verify_annotation: bool = False,
) -> Any:
    """Build a fast launcher for `jit_func`.

    Without bindings, the returned callable is a C function:

        launcher(device, stream, grid, arg0, arg1, ...)

    `device` is a device index in `[0, 256)` -- one byte of the spec key holds
    it -- `stream` a raw stream handle (both ints), `grid` an int or a
    tuple/list of up to 3 ints, and the remaining arguments are the kernel's
    public parameters, positionally, in declaration order. Baked Argument and
    Constexpr values are omitted from the call. With bound parameters this
    returns a non-callable factory; `.bind(**values)` creates the native callable
    and removes those parameters from its signature. The return shape depends
    on the annotations stored on the (untyped) JITFunction.

    `options` are triton compile options (`num_warps`, `num_stages`, ...) baked
    into every launch.  `torch_access` picks how the module reads a tensor; see
    `TorchAccess`.  `kernel_cache` picks the hash map behind the kernel cache;
    see `KernelCache`.  `no_gpu=True` decodes and caches on the host without
    compiling or launching a GPU kernel. `dynamic_grid` and `dynamic_options`
    are reserved and currently unsupported. `verify_annotation=True` checks
    declared types, ranges and assumed facts; otherwise these are caller promises.
    """
    if dynamic_grid:
        raise UnsupportedKernel("intj: dynamic_grid is not implemented; pass an int or tuple grid")
    if dynamic_options:
        raise UnsupportedKernel("intj: dynamic_options is not implemented; pass options=... instead")
    options = dict(sorted((options or {}).items()))
    if no_gpu and options:
        raise UnsupportedKernel("intj: no_gpu=True supports only default compile options")
    jit_func = _check_kernel(jit_func, options)
    resolved = _resolve_annotations(jit_func, extra_annotation)
    for p in jit_func.params:
        kind = p._param.kind
        if kind not in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD):
            raise UnsupportedKernel(
                f"intj: parameter {p.name!r} is {kind}; only positional parameters are supported"
            )
    if not no_gpu:
        from triton import knobs

        options["debug"] = options.get("debug", jit_func.debug) or knobs.runtime.debug
        options["instrumentation_mode"] = knobs.compilation.instrumentation_mode
    access = _resolve_access(torch_access)
    if any(p.annotation.bind_value is not None for p in resolved):
        if not no_gpu:
            _canonical_options(_current_target(), options)
        if verify_annotation:
            for p in resolved:
                if p.annotation.bind_value == "pointer" and p.annotation.pointer_range_32 == "assume":
                    warnings.warn(
                        f"intj: pointer-range assumption for bound pointer {p.name!r} "
                        "cannot be verified from an address and will be trusted",
                        RuntimeWarning,
                        stacklevel=2,
                    )
        return LauncherFactory(jit_func, resolved, tuple(options.items()), access, kernel_cache,
                               bool(verify_annotation), no_gpu, False)
    return _materialize_module(jit_func, resolved, options, access, kernel_cache,
                               no_gpu, bool(verify_annotation)).entry


def _materialize_module(
    jit_func: JitFunction, resolved: tuple[ResolvedParam, ...], options: Mapping[str, Any],
    access: TorchAccess, kernel_cache: KernelCache, no_gpu: bool, verify_annotation: bool,
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
        canonical_options = _canonical_options(target, options)
    device_binding = DeviceBinding.NOT_FIXED
    params, device_offset, nwords = _render_params(resolved, device_binding)

    # last, after every refusal above it: a kernel that is going to be rejected
    # anyway must not pay for a download and an abseil build first
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
        error_symbol=backend.error_symbol if backend is not None else "",
        error_style=backend.error_style if backend is not None else "return",
        torch_access=access.value,
        kernel_cache=kernel_cache.value,
        cache_include_dirs=cache_toolchain["include_dirs"],
        cache_archives=cache_toolchain["archives"],
        # Only the c++ mode compiles anything torch-specific in, so only it has
        # to be rebuilt when torch changes.  Leaving these None for the other
        # two is what keeps their digest -- and so their cached `.so` -- stable
        # across torch versions.
        torch_version=torch_version() if access is TorchAccess.CXX else None,
        cxx_abi=_cxx_abi() if access is TorchAccess.CXX else None,
        pointer_types=_pointer_types(params),
    )

    # The rendered source is a pure function of the template, the runtime header
    # and the context, so hash those rather than the render itself.
    key = ModuleKey(
        template=_ENTRY_TEMPLATE.read_bytes().hex(),
        runtime_header=_RUNTIME_HEADER.read_bytes().hex(),
        context=context,
        cache_key=jit_func.cache_key,
        target=(target.backend, target.arch, target.warp_size) if target is not None else ("host-only",),
        options=canonical_options.hash() if canonical_options is not None else "host-defaults",
        triton=_triton_identity(),
        compiler=_compiler_identity("c++" if access is TorchAccess.CXX else "c"),
        ext_suffix=sysconfig.get_config_var("EXT_SUFFIX"),
        intj_version=_cache_version(),
    )
    # the c++ mode gets it too, not to read from but to check its own
    # compiled-in offset against
    layout = layout_for() if access in (TorchAccess.SHIM, TorchAccess.CXX) else None
    baked_values = {p.index: p.baked for p in resolved if p.annotation.baked_value}
    return _loaded_module(key, jit_func, context, params, options, layout, baked_values)


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


def get_full_name(fn: Any) -> str:
    return f"{fn.__module__}.{fn.__qualname__}"


def _resolve_access(requested: TorchAccess) -> TorchAccess:
    """Turn `AUTO` into a concrete mode, or check the one the caller asked for.

    Explicit modes are validated rather than silently downgraded: a caller who
    asked for `SHIM` because they measured it wants to hear that it is
    unavailable, not to get `CPYTHON` and wonder where the time went.
    """
    if requested is TorchAccess.SHIM and layout_for() is None:
        raise UnsupportedKernel(_unverified_torch_message())
    if requested is TorchAccess.CXX and not _cxx_toolchain():
        raise UnsupportedKernel(
            "intj: the c++ access mode needs a c++ compiler and torch's headers; "
            "set $CXX or pass torch_access=TorchAccess.SHIM"
        )
    if requested is TorchAccess.CXX and layout_for() is None:
        raise UnsupportedKernel(_unverified_torch_message())
    if requested is not TorchAccess.AUTO:
        return requested
    if _cxx_toolchain() and layout_for() is not None:
        return TorchAccess.CXX
    if layout_for() is not None:
        return TorchAccess.SHIM
    return TorchAccess.CPYTHON


def _unverified_torch_message() -> str:
    return (
        f"intj: no verified tensor layout for torch {_torch_version_string()} "
        f"(have {', '.join('%d.%d' % v for v in supported_versions())}), or its "
        "dtypes are no longer the ones the stanza was measured against; pass "
        "torch_access=TorchAccess.CPYTHON, or add a stanza to intj/torch_abi.txt "
        "with `python -m intj.torch_abi` on this torch"
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
    from triton.runtime import build

    try:
        build._find_compiler("c++")  # pyright: ignore[reportPrivateUsage]
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
) -> types.ModuleType:
    """One module per `ModuleKey`, for the life of the process.

    Reusing the module reuses its compile callback, which owns the
    `CompiledKernel`s the C kernel cache points into -- installing a second
    callback would drop the first, freeing kernels whose function handles are
    still in that cache. So a hit returns the module untouched.

    The lock matters: without it two threads both compile and both load, and the
    loser's module (with its own kernel cache) is silently dropped.
    """
    with _LOAD_LOCK:
        module = _LOADED.get(key)
        if module is None:
            module = _load(key, jit_func, context)
            module.set_compile_callback(
                _make_host_compile_callback() if context.no_gpu
                else _make_compile_callback(jit_func, params, options, baked_values)
            )
            # The tensor layout is installed, not compiled in, so it describes the
            # torch running now rather than the one this `.so` was built against.
            module.set_torch_version(
                torch_version(), layout.as_args() if layout else None, dtype_index_table()
            )
            _LOADED[key] = module
        return module


def _load(key: ModuleKey, jit_func: JitFunction, context: RenderContext) -> types.ModuleType:
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
    directory = Path(knobs.cache.get_triton_dir("intj")) / key.digest() / jit_func.__module__
    so_path = directory / f"{context.module_name}{suffix}"
    if not so_path.exists():
        _build(so_path, context)

    # Loading by hand, rather than through import_module, keeps the module out of
    # sys.modules.
    spec = importlib.util.spec_from_file_location(context.module_name, so_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"intj: cannot load {so_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _build(so_path: Path, context: RenderContext) -> None:
    """Render and compile, then install both artifacts next to each other."""
    import jinja2

    from triton.runtime.build import compile_so_from_src

    template = jinja2.Template(_ENTRY_TEMPLATE.read_text(), undefined=jinja2.StrictUndefined)
    src = template.render(**dataclasses.asdict(context))
    built = compile_so_from_src(
        src=src, name=context.module_name, **_build_flags(context)
    )


    so_path.parent.mkdir(parents=True, exist_ok=True)
    # The source is kept next to the binary: it is what you read when a launch
    # misbehaves, and what you recompile by hand to debug it.
    suffix = ".cpp" if _build_flags(context)["language"] == "c++" else ".c"
    _install(src.encode(), so_path.with_name(f"{context.module_name}{suffix}"))
    _install(Path(built).read_bytes(), so_path)


def _runtime_header_flag() -> str:
    """Make the runtime header part of what triton's compile cache keys on.

    `_compile_so` keys on the source bytes plus the *names* of the include
    directories, not on the contents of the headers found there.  So an edit to
    `intj_runtime.h` moves intj's own `ModuleKey` digest, intj re-renders, and
    triton then serves the object it compiled from the previous header.  Feeding
    the digest in as a define puts it in triton's key too.
    """
    digest = hashlib.sha256(_RUNTIME_HEADER.read_bytes()).hexdigest()[:16]
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
    cxx_access = context.torch_access == TorchAccess.CXX.value
    cxx_cache = context.kernel_cache != KernelCache.INTJ.value
    flags: dict[str, Any] = {
        "include_dirs": [str(_RUNTIME), *context.cache_include_dirs],
        "ccflags": [_runtime_header_flag(), f"-DINTJ_CACHE_{context.kernel_cache.upper()}"],
    }
    if context.cache_archives:
        # abseil's own link order is not intj's to encode, so the archives go in
        # a group and the linker sorts it out.
        flags["ccflags"] += ["-Wl,--start-group", *context.cache_archives, "-Wl,--end-group"]
    if not cxx_access and not cxx_cache:
        return {"language": "c", **flags}

    flags["language"] = "c++"
    # triton puts its own -std=c++17 early and appends ccflags last, so this
    # wins.  torch >= 2.14 needs c++20 to compile warning-clean.
    flags["ccflags"].insert(0, "-std=c++20")
    if not cxx_access:
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
    unknown = set(options) - {f.name for f in dataclasses.fields(parsed)}
    if unknown:
        raise UnsupportedKernel(f"intj: unknown compile option(s) {sorted(unknown)}")
    return parsed


def _check_kernel(jit_func: JitFunction, options: Mapping[str, Any]) -> JitFunction:
    from triton import knobs
    from triton.runtime.jit import JITFunction

    if knobs.runtime.interpret:
        raise UnsupportedKernel("intj: TRITON_INTERPRET=1 is not supported")
    if not isinstance(jit_func, JITFunction):
        raise UnsupportedKernel(
            f"intj: expected a @triton.jit function, got {type(jit_func).__name__}; "
            "autotuned and heuristic kernels are not supported"
        )
    if jit_func.pre_run_hooks:
        raise UnsupportedKernel("intj: kernels with pre-run hooks are not supported")
    jit_func.cache_key  # populates used_global_vals
    if jit_func.used_global_vals:
        names = sorted({name for name, _ in jit_func.used_global_vals})
        raise UnsupportedKernel(
            f"intj: kernel reads global variable(s) {names}, which intj cannot revalidate; "
            "pass them as arguments instead"
        )
    for key in ("device", "stream", "device_type", "warp_size"):
        if key in options:
            raise UnsupportedKernel(f"intj: option {key!r} is not allowed")
    return jit_func


def _render_params(
    resolved: tuple[ResolvedParam, ...], device_binding: DeviceBinding
) -> tuple[tuple[Param, ...], int | None, int]:
    """Place canonical key fields and number the arguments in the public call."""
    fields = tuple(
        (p.index, field) for p in resolved for field in _key_fields(p.annotation)
    )
    layout = _layout_fields(fields, device_binding)
    params: list[Param] = []
    call_index = 0
    for p in resolved:
        annotation = dataclasses.replace(
            p.annotation,
            key_fields=tuple(sorted(
                (field for index, field in layout.fields if index == p.index),
                key=lambda field: field.offset,
            )),
        )
        public = not annotation.baked_value and annotation.bind_value is None
        params.append(Param(p.name, p.index, call_index if public else None, annotation))
        if public:
            call_index += 1
    return tuple(params), layout.device_offset, layout.nwords


def _pointer_types(params: Sequence[Param]) -> tuple[tuple[str, int], ...]:
    """Resolve explicit pointer names through the same live dtypes as the ABI table."""
    from triton._utils import type_canonicalisation_dict

    wanted = {ty[1:] for p in params for ty in p.annotation.types or ()
              if ty is not None and ty.startswith("*")}
    if not wanted:
        return ()
    return tuple(sorted(
        ("*" + canonical, code)
        for code, (name, _) in live_dtypes().items()
        if (canonical := type_canonicalisation_dict.get(name)) in wanted
    ))


@functools.lru_cache(maxsize=1)
def _triton_identity() -> tuple[Any, ...]:
    import triton

    libtriton = Path(triton.__file__).parent / "_C" / "libtriton.so"
    stat = libtriton.stat()
    return (triton.__version__, stat.st_size, stat.st_mtime)


@functools.lru_cache(maxsize=2)
def _compiler_identity(language: str = "c") -> tuple[str, ...]:
    from triton.runtime import build

    cc = build._find_compiler(language)  # pyright: ignore[reportPrivateUsage]  # the compiler triton itself picks
    cc = cc[0] if isinstance(cc, tuple) else cc
    try:
        version = subprocess.run([cc, "--version"], capture_output=True, text=True).stdout.splitlines()[0]
    except Exception:  # pragma: no cover - compiler without --version
        version = ""
    return (str(cc), version)


def _triton_specialize(
    value: object, *, backend: Any, is_const: bool, specialize: bool, align: bool
) -> tuple[str, Any]:
    from triton._C.libtriton import native_specialize_impl

    return native_specialize_impl(backend, value, is_const, specialize, align)


def _compiler_input(
    jit_func: JitFunction, params: Sequence[Param], public_args: Sequence[object], backend: Any,
    *, baked_values: Mapping[int, object] | None = None,
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
            assert baked_values is not None, "fixed values must accompany their canonical annotations"
            value = baked_values[param.index]
        else:
            value = public_args[param.call_index]
        ty = "constexpr"
        desc = ""
        if annotation.kind == "argument":
            fixed_type = annotation.types is not None and len(annotation.types) == 1
            effective = annotation.types[0] if fixed_type and annotation.types else None
            modes = (("equal_to_one", annotation.equal_to_one),
                     ("aligned_16", annotation.aligned_16),
                     ("pointer_range_32", annotation.pointer_range_32))
            assumed_one = annotation.equal_to_one == "assume" and _applicable(effective, "equal_to_one")
            inferred_desc = None
            if not assumed_one and (not fixed_type or any(
                mode == "auto" and _applicable(effective, field) for field, mode in modes
            )):
                inference_value = value
                if effective is not None and effective.startswith("*") and (
                    value is None or type(value) is int and value == 0
                ):
                    from triton.runtime.jit import MockTensor

                    # Both public null spellings have pointer 0 and storage range 0.
                    inference_value = MockTensor(effective[1:])
                inferred, inferred_desc = _triton_specialize(
                    inference_value, backend=backend, is_const=jit_func.params[param.index].is_const,
                    specialize=any(mode == "auto" for _, mode in modes),
                    align=annotation.aligned_16 == "auto",
                )
                if not fixed_type:
                    # The native primitive folds only None and integer 1.
                    effective = (None if value is None else "i32") if inferred == "constexpr" else inferred
            ty = effective or "constexpr"
            if _applicable(effective, "equal_to_one") and (
                annotation.equal_to_one == "assume" or
                annotation.equal_to_one == "auto" and value == 1
            ):
                ty = "constexpr"
                if annotation.equal_to_one == "assume":
                    value = 1
            if ty != "constexpr":
                desc = "".join(char for field, mode, char in (
                    ("aligned_16", annotation.aligned_16, "D"),
                    ("pointer_range_32", annotation.pointer_range_32, "S"),
                ) if _applicable(effective, field) and (
                    mode == "assume" or mode == "auto" and
                    isinstance(inferred_desc, str) and char in inferred_desc
                ))
        path = (param.index,)
        signature.append((param.name, ty))
        if ty == "constexpr":
            constants.append((path, _canonical_value(value)))
            values.append((path, value))
        if desc:
            parsed = tuple((name, amount) for name, amount in backend.parse_attr(desc))
            if parsed:
                attrs.append((path, parsed))
    return CompilerInput(tuple(signature), tuple(constants), tuple(attrs), tuple(values))


def _make_compile_callback(
    jit_func: JitFunction, params: Sequence[Param], options: Mapping[str, Any],
    baked_values: Mapping[int, object] | None = None,
) -> Callable[..., tuple[int, int, int, int]]:
    """Called from C on a spec-key miss, with the key blob and the original args."""
    from triton.compiler import compile as triton_compile, make_backend

    target = _current_target()
    backend = make_backend(target)
    canonical_options = _canonical_options(target, options)
    kernels: list[CompiledKernel] = []  # keeps every CompiledKernel, and so its GPU module, alive
    seen: dict[bytes, CompilerInput] = {}
    no_key_input: CompilerInput | None = None

    def compile_callback(
        keyblob: bytes, nparams: int, device: int, *args: Any
    ) -> tuple[int, int, int, int]:
        nonlocal no_key_input
        current = _current_device()
        if device != current:
            # _init_handles() loads the binary on the *current*
            # device; launching that function on another device's stream is a
            # wrong-context launch, so refuse instead.
            raise UnsupportedKernel(
                f"intj: launching on device {device} while device {current} is current; "
                "make the target device current before the first launch"
            )
        compiler_input = _compiler_input(jit_func, params, args, backend, baked_values=baked_values)
        if keyblob:
            previous = seen.setdefault(keyblob, compiler_input)
        else:
            if no_key_input is None:
                no_key_input = compiler_input
            previous = no_key_input
        if previous != compiler_input:
            raise RuntimeError(
                "intj: one spec key maps to two annotated ASTSource inputs; this is an intj bug"
            )
        kernel = triton_compile(compiler_input.ast_source(jit_func), target=target,
                                options=canonical_options.__dict__)
        kernel._init_handles()

        md = kernel.metadata
        if md.num_ctas != 1:
            raise UnsupportedKernel("intj: num_ctas > 1 is not supported")
        if md.launch_cooperative_grid:
            raise UnsupportedKernel("intj: launch_cooperative_grid is not supported")
        if getattr(md, "launch_pdl", False):  # nvidia only
            raise UnsupportedKernel("intj: launch_pdl is not supported")
        if md.global_scratch_size or md.profile_scratch_size:
            raise UnsupportedKernel("intj: kernels requiring scratch memory are not supported")

        expected = sum(1 for ty in kernel.src.signature.values() if ty != "constexpr")
        if expected != nparams:
            raise RuntimeError(
                f"intj: packed {nparams} kernel arguments but triton compiled {expected}; "
                "this is an intj bug"
            )
        kernels.append(kernel)
        return (kernel.function, md.warp_size * md.num_warps, md.shared, nparams)

    return compile_callback


def _make_host_compile_callback() -> Callable[..., tuple[int, int, int, int]]:
    def compile_callback(
        keyblob: bytes, nparams: int, device: int, *args: Any
    ) -> tuple[int, int, int, int]:
        del keyblob, device, args
        return 0, 1, 0, nparams

    return compile_callback


def triton_specialization(
    jit_func: JitFunction, args: Iterable[Any], options: Mapping[str, Any] | None = None
) -> list[tuple[str, Any]]:
    """The `list[(type_str, key)]` triton would compute for these arguments."""
    from triton import knobs

    binder = jit_func.device_caches[_current_device()][4]
    kwargs = dict(options or {})
    kwargs["debug"] = jit_func.debug or knobs.runtime.debug
    kwargs["instrumentation_mode"] = knobs.compilation.instrumentation_mode
    return binder(*args, **kwargs)[1]

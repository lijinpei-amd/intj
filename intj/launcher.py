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
import subprocess
import sysconfig
import threading
import types
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

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
    TorchAccess,
    dtype_index_table,
    layout_for,
    supported_versions,
    torch_version,
)

# triton and torch ship no type information, so everything reaching into them is
# typed `Any` on purpose.
JitFunction = Any
CompiledKernel = Any

SCHEMA_VERSION = 1

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


@dataclasses.dataclass(frozen=True)
class Param:
    """One declared kernel parameter, as the template needs it."""

    name: str
    is_constexpr: bool
    spec: int  # 0 when do_not_specialize
    align: int  # 0 when do_not_specialize_on_alignment
    # Its code byte is at its declaration position, which the template already
    # has as `loop.index0`; only the value word needs carrying.
    cx_word: int | None  # value-word index, None unless is_constexpr


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
    header_words: int  # words the code bytes and the device byte occupy
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
    #: The interpreter the module is built for and loaded into.  Every mode reads
    #: CPython internals (see intj/python_intf), so a module is never shared
    #: across versions, patch releases included; the render refuses to compile
    #: against any other `Python.h`.  ABI flags (`t`, `d`) are also in `ext_suffix`.
    python_version: tuple[int, int, int]
    free_threaded: bool
    python_abi: str  # the intj/python_intf header implementing that version


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
    cache_key: str  # jit_func.cache_key: the kernel source and its callees
    params: tuple[tuple[Any, ...], ...]  # per-parameter decorator state, which cache_key misses
    target: tuple[Any, ...]  # backend, arch, warp size
    options: str  # canonicalized compile options, hashed by triton
    triton: tuple[Any, ...]  # triton version and libtriton identity
    compiler: tuple[str, ...]  # the C compiler triton would invoke
    ext_suffix: str | None
    schema: int = SCHEMA_VERSION

    def digest(self) -> str:
        # sort_keys so the digest does not depend on field order
        blob = json.dumps(dataclasses.asdict(self), sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()


def make_launcher(
    jit_func: JitFunction,
    dynamic_grid: bool = False,
    dynamic_options: Sequence[str] = (),
    extra_annotation: Mapping[str, str] | None = None,
    options: Mapping[str, Any] | None = None,
    torch_access: TorchAccess = TorchAccess.AUTO,
    kernel_cache: KernelCache = KernelCache.INTJ,
) -> Callable[..., None]:
    """Build a fast launcher for `jit_func`.

    The returned callable is a C function:

        launcher(device, stream, grid, arg0, arg1, ...)

    `device` is a device index in `[0, 256)` -- one byte of the spec key holds
    it -- `stream` a raw stream handle (both ints), `grid` an int or a
    tuple/list of up to 3 ints, and the remaining arguments are the kernel's
    parameters, positionally, in declaration order.

    `options` are triton compile options (`num_warps`, `num_stages`, ...) baked
    into every launch.  `torch_access` picks how the module reads a tensor; see
    `TorchAccess`.  `kernel_cache` picks the hash map behind the kernel cache;
    see `KernelCache`.  `dynamic_grid`, `dynamic_options` and `extra_annotation`
    are reserved and currently unsupported.
    """
    if dynamic_grid:
        raise UnsupportedKernel("intj: dynamic_grid is not implemented; pass an int or tuple grid")
    if dynamic_options:
        raise UnsupportedKernel("intj: dynamic_options is not implemented; pass options=... instead")
    if extra_annotation:
        raise UnsupportedKernel("intj: extra_annotation is not implemented")

    options = dict(sorted((options or {}).items()))
    jit_func = _check_kernel(jit_func, options)
    params, nwords = _render_params(jit_func)

    python_abi = cpython_abi.header_for()
    if python_abi is None:
        raise UnsupportedKernel(
            f"intj: no verified CPython layer for python {'%d.%d.%d' % cpython_abi.python_version()} "
            f"(have {', '.join('%d.%d' % v for v in cpython_abi.supported_versions())}); "
            "run `python -m intj.python_intf.check` on it and add it to intj/python_intf/cpython_abi.py"
        )
    access = _resolve_access(torch_access)
    target = _current_target()
    backend = BACKENDS[target.backend]
    canonical_options = _canonical_options(target, options)
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
        header_words=-(-(len(params) + 1) // 8),
        max_slots=sum(0 if p.is_constexpr else 1 for p in params),
        # Where the backend specializes on it, the pointer-range bit is always in
        # the key, even when the knob that emits it is off: a key that cannot tell
        # a > 2 GiB buffer apart would launch a tt.pointer_range=32 binary on it
        # after the knob is flipped back on.
        spec_pointer_range=1 if backend.pointer_range else 0,
        driver_path=backend.library_path(),
        launch_symbol=backend.launch_symbol,
        error_symbol=backend.error_symbol,
        error_style=backend.error_style,
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
        python_version=cpython_abi.python_version(),
        free_threaded=bool(sysconfig.get_config_var("Py_GIL_DISABLED")),
        python_abi=python_abi,
    )

    # The rendered source is a pure function of the template, the runtime header
    # and the context, so hash those rather than the render itself.
    key = ModuleKey(
        template=_ENTRY_TEMPLATE.read_bytes().hex(),
        runtime_header=_runtime_headers().hex(),
        context=context,
        cache_key=jit_func.cache_key,
        params=tuple(
            (p.name, p.annotation, p.is_constexpr, p.is_const, p.do_not_specialize,
             p.do_not_specialize_on_alignment, p.has_default, repr(p.default))
            for p in jit_func.params
        ),
        target=(target.backend, target.arch, target.warp_size),
        options=canonical_options.hash(),
        triton=_triton_identity(),
        compiler=_compiler_identity("c++" if access is TorchAccess.CXX else "c"),
        ext_suffix=sysconfig.get_config_var("EXT_SUFFIX"),
    )
    # the c++ mode gets it too, not to read from but to check its own
    # compiled-in offset against
    layout = layout_for() if access is TorchAccess.SHIM else None
    return _loaded_module(key, jit_func, context, params, options, layout).entry


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
    if requested is not TorchAccess.AUTO:
        return requested
    if _cxx_toolchain():
        return TorchAccess.CXX
    if layout_for() is not None:
        return TorchAccess.SHIM
    return TorchAccess.CPYTHON


def _unverified_torch_message() -> str:
    return (
        f"intj: no verified tensor layout for torch {_torch_version_string()} "
        f"(have {', '.join('%d.%d' % v for v in supported_versions())}), or its "
        "dtypes are no longer the ones the entry was measured against, or this "
        "is a free-threaded build, which no entry covers; pass "
        "torch_access=TorchAccess.CPYTHON, or add an entry to intj/torch_intf/torch_abi.toml "
        "with `python -m intj.torch_intf.abi_detect` on this torch"
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
            module.set_compile_callback(_make_compile_callback(jit_func, params, options))
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
    spec.loader.exec_module(module)  # pyright: ignore[reportAttributeAccessIssue]  # 3.8 stubs lack it
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
    cxx_access = context.torch_access == TorchAccess.CXX.value
    cxx_cache = context.kernel_cache != KernelCache.INTJ.value
    flags: dict[str, Any] = {
        "include_dirs": [str(_RUNTIME), str(_PYTHON_INTF), *context.cache_include_dirs],
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
    target = _current_target()
    if target.backend not in BACKENDS:
        raise UnsupportedKernel(
            f"intj: backend {target.backend!r} is unknown; subclass intj.launcher.Backend and "
            f"register() it (have: {', '.join(sorted(BACKENDS))})"
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


def _render_params(jit_func: JitFunction) -> tuple[tuple[Param, ...], int]:
    """One render-time descriptor per declared kernel parameter, and the key length.

    Byte `i` of the key is parameter `i`, the device byte follows them, and the
    constexpr value words follow the padding.  `nwords` comes back alongside the
    params because it is the same arithmetic: computing it somewhere else is how
    the two drift.
    """
    declared = list(jit_func.params)
    header_words = -(-(len(declared) + 1) // 8)
    params: list[Param] = []
    word = header_words
    for p in declared:
        kind = p._param.kind
        if kind not in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD):
            raise UnsupportedKernel(f"intj: parameter {p.name!r} is {kind}; only positional parameters are supported")
        if not p.is_constexpr and p.annotation:
            raise UnsupportedKernel(
                f"intj: parameter {p.name!r} has annotation {p.annotation!r}; only tl.constexpr annotations "
                "are supported"
            )
        params.append(
            Param(
                name=p.name,
                is_constexpr=p.is_constexpr,
                spec=0 if p.do_not_specialize else 1,
                align=0 if p.do_not_specialize_on_alignment else 1,
                cx_word=word if p.is_constexpr else None,
            )
        )
        if p.is_constexpr:
            word += 1
    return tuple(params), word


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


def _make_compile_callback(
    jit_func: JitFunction, params: Sequence[Param], options: Mapping[str, Any]
) -> Callable[..., tuple[int, int, int, int]]:
    """Called from C on a spec-key miss, with the key blob and the original args."""
    kernels: list[CompiledKernel] = []  # keeps every CompiledKernel, and so its GPU module, alive
    seen: dict[tuple[int, tuple[int, ...]], str] = {}  # key words -> triton's specialization entry

    def compile_callback(
        keyblob: bytes, nparams: int, device: int, *args: Any
    ) -> tuple[int, int, int, int]:
        current = _current_device()
        if device != current:
            # warmup() and _init_handles() both load the binary on the *current*
            # device; launching that function on another device's stream is a
            # wrong-context launch, so refuse instead.
            raise UnsupportedKernel(
                f"intj: launching on device {device} while device {current} is current; "
                "make the target device current before the first launch"
            )
        kernel = jit_func.warmup(*args, grid=None, **options)
        if kernel is None:
            raise RuntimeError("intj: triton returned no kernel (a jit_cache_hook is installed?)")
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
        _validate_spec_key(jit_func, params, options, keyblob, args, seen)

        kernels.append(kernel)
        return (kernel.function, md.warp_size * md.num_warps, md.shared, nparams)

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


def _validate_spec_key(
    jit_func: JitFunction,
    params: Sequence[Param],
    options: Mapping[str, Any],
    keyblob: bytes,
    args: Sequence[Any],
    seen: dict[tuple[int, tuple[int, ...]], str],
) -> None:
    """Assert intj's key is no coarser than triton's specialization.

    Cheap (the miss path is milliseconds) and it turns a classifier drift from a
    silently wrong kernel launch into a loud failure: if the same intj key word
    ever maps to two different triton specializations, the key is too coarse.
    A key that is coarse in *both* directions never misses and so never reaches
    this check -- `tests/test_launcher.py` covers that case with `spec_key`.
    """
    specialization = triton_specialization(jit_func, args, options)

    for i, param in enumerate(params):
        # byte i is parameter i; a constexpr adds its value word
        mine: tuple[int, ...] = (keyblob[i],)
        if param.cx_word is not None:
            at = param.cx_word * 8
            mine += (int.from_bytes(keyblob[at:at + 8], "little"),)
        theirs = repr(specialization[i])
        previous = seen.setdefault((i, mine), theirs)
        if previous != theirs:
            raise RuntimeError(
                f"intj: spec key for parameter {param.name!r} is too coarse: it maps both "
                f"{previous} and {theirs} to the same key; this is an intj bug"
            )

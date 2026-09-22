"""Render, build and load the C launcher for a `@triton.jit` kernel."""

import abc
import dataclasses
import functools
import hashlib
import importlib.util
import inspect
import json
import os
import struct
import subprocess
import sysconfig
import threading
import types
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

# triton and torch ship no type information, so everything reaching into them is
# typed `Any` on purpose.
JitFunction = Any
CompiledKernel = Any

SCHEMA_VERSION = 1

_RUNTIME = Path(__file__).parent / "runtime"
_ENTRY_TEMPLATE = _RUNTIME / "entry.c.jinja"
_RUNTIME_HEADER = _RUNTIME / "intj_runtime.h"


class UnsupportedKernel(NotImplementedError):
    """Raised for kernels or options outside intj's (deliberately small) scope."""


@dataclasses.dataclass(frozen=True)
class Param:
    """One declared kernel parameter, as the template needs it."""

    name: str
    is_constexpr: bool
    spec: int  # 0 when do_not_specialize
    align: int  # 0 when do_not_specialize_on_alignment
    word: int  # index of its first spec-key word


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
    max_slots: int  # kernel param slots, upper bound
    spec_pointer_range: int  # 1 if the backend specializes pointers on a 2 GiB range
    driver_path: str
    launch_symbol: str
    error_symbol: str
    error_style: str  # "return" | "outparam"
    libtorch_path: str


@dataclasses.dataclass(frozen=True)
class ModuleKey:
    """Everything the rendered `.so` depends on, hashed into its digest.

    Anything that changes the generated code, the compiled binary, or which
    kernel the module's callback compiles has to appear here: two launchers
    whose keys match share one `.so`, so a missing field means a stale module.
    """

    template: str  # entry.c.jinja bytes, hex
    runtime_header: str  # intj_runtime.h bytes, hex
    context: RenderContext  # with an empty module name: the digest is what names it
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


def create_launcher(
    jit_func: JitFunction,
    dynamic_grid: bool = False,
    dynamic_options: Sequence[str] = (),
    extra_annotation: Mapping[str, str] | None = None,
    options: Mapping[str, Any] | None = None,
) -> Callable[..., None]:
    """Build a fast launcher for `jit_func`.

    The returned callable is a C function:

        launcher(device, stream, grid, arg0, arg1, ...)

    `device` is a device index, `stream` a raw stream handle (both ints), `grid`
    an int or a tuple/list of up to 3 ints, and the remaining arguments are the
    kernel's parameters, positionally, in declaration order.

    `options` are triton compile options (`num_warps`, `num_stages`, ...) baked
    into every launch.  `dynamic_grid`, `dynamic_options` and `extra_annotation`
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
    params = _render_params(jit_func)

    import torch

    target = _current_target()
    backend = BACKENDS[target.backend]
    canonical_options = _canonical_options(target, options)
    context = RenderContext(
        module_name="",  # the digest below names the module, so it is filled in last
        kernel_repr=get_full_name(jit_func),
        params=params,
        nwords=1 + sum(2 if p.is_constexpr else 1 for p in params),
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
        libtorch_path=os.path.join(os.path.dirname(torch.__file__), "lib", "libtorch_cpu.so"),
    )

    # The rendered source is a pure function of the template, the runtime header
    # and the context, so hash those instead of the render -- otherwise naming the
    # module after the digest would need a throwaway render first.
    key = ModuleKey(
        template=_ENTRY_TEMPLATE.read_bytes().hex(),
        runtime_header=_RUNTIME_HEADER.read_bytes().hex(),
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
        compiler=_compiler_identity(),
        ext_suffix=sysconfig.get_config_var("EXT_SUFFIX"),
    )
    return _loaded_module(key, jit_func, context, params, options).entry


def get_full_name(fn: Any) -> str:
    return f"{fn.__module__}.{fn.__qualname__}"


_LOADED: dict[ModuleKey, types.ModuleType] = {}
_LOAD_LOCK = threading.Lock()


def _loaded_module(
    key: ModuleKey,
    jit_func: JitFunction,
    context: RenderContext,
    params: Sequence[Param],
    options: Mapping[str, Any],
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
            _LOADED[key] = module
        return module


def _load(key: ModuleKey, jit_func: JitFunction, context: RenderContext) -> types.ModuleType:
    """Load the extension for this key, rendering and building it only if needed.

    The `.so` lands in `<cache>/loaded_modules/<digest>/<module>/<kernel>`, so the
    artifact says where the kernel came from and which build it is. Its leaf name
    is the kernel's own name, because CPython derives `PyInit_<leaf>` from the
    last dotted component of the spec name -- that is also what `perf` and
    /proc/<pid>/maps show.

    An existing `.so` is loaded as-is: its path already encodes the digest, so
    rendering the source again would only reproduce what was compiled from it.
    """
    from triton import knobs

    # the kernel's own name, unless python allows something C does not
    module_name = jit_func.__name__
    if not (module_name.isidentifier() and module_name.isascii()):
        module_name = "kernel"

    digest = key.digest()
    suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
    # $TRITON_HOME/.triton/intj, a sibling of triton's own cache
    root = Path(knobs.cache.get_triton_dir("intj")) / "loaded_modules"
    directory = root / digest / jit_func.__module__
    so_path = directory / f"{module_name}{suffix}"
    if not so_path.exists():
        _build(so_path, module_name, dataclasses.replace(context, module_name=module_name))

    # A distinct spec name per digest keeps two builds of one kernel apart.
    spec_name = f"intj.loaded_modules.{get_full_name(jit_func)}.{digest[:16]}.{module_name}"
    spec = importlib.util.spec_from_file_location(spec_name, so_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"intj: cannot load {so_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _build(so_path: Path, module_name: str, context: RenderContext) -> None:
    """Render and compile, then install both artifacts next to each other."""
    import jinja2

    from triton.runtime.build import compile_so_from_src

    template = jinja2.Template(_ENTRY_TEMPLATE.read_text(), undefined=jinja2.StrictUndefined)
    src = template.render(**dataclasses.asdict(context))
    built = compile_so_from_src(src=src, name=module_name, include_dirs=[str(_RUNTIME)], language="c")

    so_path.parent.mkdir(parents=True, exist_ok=True)
    # The source is kept next to the binary: it is what you read when a launch
    # misbehaves, and what you recompile by hand to debug it.
    _install(src.encode(), so_path.with_name(f"{module_name}.c"))
    _install(Path(built).read_bytes(), so_path)


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
    """Make `backend` usable by `create_launcher`, replacing any same-named one."""
    BACKENDS[backend.name] = backend
    return backend


register(HipBackend())
register(CudaBackend())


# --------------------------------------------------------------------- checks


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
    backend rejects, at `create_launcher` time rather than on the first launch.

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


def _render_params(jit_func: JitFunction) -> tuple[Param, ...]:
    """One render-time descriptor per declared kernel parameter."""
    params: list[Param] = []
    word = 1
    for p in jit_func.params:
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
                word=word,
            )
        )
        word += 2 if p.is_constexpr else 1
    return tuple(params)


# -------------------------------------------------------------------- render


@functools.lru_cache(maxsize=1)
def _triton_identity() -> tuple[Any, ...]:
    import triton

    libtriton = Path(triton.__file__).parent / "_C" / "libtriton.so"
    stat = libtriton.stat()
    return (triton.__version__, stat.st_size, stat.st_mtime)


@functools.lru_cache(maxsize=1)
def _compiler_identity() -> tuple[str, ...]:
    from triton.runtime import build

    cc = build._find_compiler("c")  # pyright: ignore[reportPrivateUsage]  # the compiler triton itself picks
    cc = cc[0] if isinstance(cc, tuple) else cc
    try:
        version = subprocess.run([cc, "--version"], capture_output=True, text=True).stdout.splitlines()[0]
    except Exception:  # pragma: no cover - compiler without --version
        version = ""
    return (str(cc), version)


# ------------------------------------------------------------- compile hook


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
    words = struct.unpack(f"<{len(keyblob) // 8}Q", keyblob)
    specialization = triton_specialization(jit_func, args, options)

    for i, param in enumerate(params):
        n = 2 if param.is_constexpr else 1
        mine = words[param.word:param.word + n]
        theirs = repr(specialization[i])
        previous = seen.setdefault((i, mine), theirs)
        if previous != theirs:
            raise RuntimeError(
                f"intj: spec key for parameter {param.name!r} is too coarse: it maps both "
                f"{previous} and {theirs} to the same key; this is an intj bug"
            )

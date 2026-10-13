"""Build a rendered module's `.so`, and load it."""

from __future__ import annotations

import functools
import hashlib
import importlib.util
import os
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import types
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .. import _imports, lazy
from ..kernel_cache import KernelCache
from ..torch_intf.torch_abi import TorchAccessMode
from .context import render
from .key import RenderKey

RUNTIME = Path(__file__).parent.parent / "runtime"
PYTHON_INTF = Path(__file__).parent.parent / "python_intf"


def runtime_headers() -> bytes:
    """Returns every header the rendered module can include, concatenated.

    That is the runtime's own headers, and the CPython layer `intj_runtime.h`
    includes from `python_intf/`.
    """
    headers = [*sorted(RUNTIME.glob("*.h")), *sorted(PYTHON_INTF.glob("*.h"))]
    return b"".join(p.read_bytes() for p in headers)


@functools.lru_cache(maxsize=2)
def compiler_path(language: str) -> str:
    """Returns the `"c"` or `"c++"` compiler triton would invoke.

    Raises:
        RuntimeError: No such compiler is found.
    """
    _imports.ensure_triton_imported()
    find_compiler = getattr(_imports.build, "_find_compiler", None)
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
def compiler_identity(language: str = "c") -> tuple[str, ...]:
    cc = compiler_path(language)
    try:
        version = subprocess.run(
            [cc, "--version"], capture_output=True, text=True
        ).stdout.splitlines()[0]
    except Exception:  # pragma: no cover - compiler without --version
        version = ""
    return (str(cc), version)


_TORCH_HEADER = "torch/csrc/autograd/python_variable.h"


@functools.lru_cache(maxsize=1)
def cxx_toolchain() -> tuple[list[str], list[str]] | None:
    """Returns (include dirs, library dirs) for the c++ mode, or None.

    Both come from the loaded torch rather than from `torch.__file__`, so a torch
    that reorganises its tree, or one built out of tree, keeps working.  Only a
    build needs this; deciding the mode and keying the module need only
    `cxx_lib_dirs`.
    """
    try:
        compiler_path("c++")
    except Exception:
        return None
    try:
        # not cached with torch: importing it costs ~55 ms
        from torch.utils import cpp_extension
    except Exception:  # pragma: no cover - torch without the build helpers
        return None
    includes = [p for p in cpp_extension.include_paths() if os.path.isdir(p)]
    libs = [p for p in cpp_extension.library_paths() if os.path.isdir(p)]
    if not libs or not any(
        os.path.exists(os.path.join(d, _TORCH_HEADER)) for d in includes
    ):
        return None
    return (includes, libs)


@functools.lru_cache(maxsize=1)
def cxx_lib_dirs() -> tuple[str, ...] | None:
    """Returns `cxx_toolchain()`'s library dirs, or None where it returns None.

    The access mode, and the rpath the module's digest records, need only
    these, so a first call whose `.so` is already on disk should not pay ~55 ms
    for `torch.utils.cpp_extension`.  Where `_lib_dirs_without_cpp_extension`
    is certain of the answer it is used; anywhere else this asks
    `cxx_toolchain`.  `build_flags` checks the two agree before any build.
    """
    try:
        compiler_path("c++")
    except Exception:
        return None
    libs = _lib_dirs_without_cpp_extension()
    if libs is not None:
        return libs
    toolchain = cxx_toolchain()
    return None if toolchain is None else tuple(toolchain[1])


def _lib_dirs_without_cpp_extension() -> tuple[str, ...] | None:
    """Returns `cxx_toolchain()`'s library dirs where they are certain without it.

    None means "ask cpp_extension", not "unavailable".  cpp_extension's torch
    root is `dirname(dirname(its own file))` before 2.14 and
    `torch._utils_internal.get_file_path("torch")` from 2.14 on; where the two
    agree, that is its root on either.  On every torch from 2.2 to 2.14 its
    first include dir is `<root>/include` and its only library dir
    `<root>/lib`, so the header in the one and the existence of the other are
    enough to say yes.
    """
    if "torch.utils.cpp_extension" in sys.modules:
        return None  # already imported: asking it is free
    if importlib.util.find_spec("setuptools") is None:
        return None  # cpp_extension imports it: let that import fail
    _imports.ensure_torch_imported()
    spec = importlib.util.find_spec("torch.utils.cpp_extension")
    # torch's own helper, the one cpp_extension calls from 2.14 on
    utils_internal = getattr(_imports.torch, "_utils_internal", None)
    get_file_path = getattr(utils_internal, "get_file_path", None)
    if spec is None or spec.origin is None or get_file_path is None:
        return None
    root = os.path.dirname(os.path.dirname(os.path.abspath(spec.origin)))
    if get_file_path("torch") != root:
        return None
    lib = os.path.join(root, "lib")
    header = os.path.join(root, "include", _TORCH_HEADER)
    if not os.path.isdir(lib) or not os.path.exists(header):
        return None
    return (lib,)


def load(key: RenderKey, kernel_module: str) -> types.ModuleType:
    """Loads the extension for this key, rendering and building it only if needed.

    The `.so` lands in `$TRITON_HOME/.triton/intj/<digest>/<kernel_module>/<kernel>`,
    so the artifact says where the kernel came from and which build it is. Its leaf name
    is the kernel's own name, because CPython derives `PyInit_<leaf>` from the
    last dotted component of the spec name -- that is also what `perf` and
    /proc/<pid>/maps show.

    An existing `.so` is loaded as-is: its path already encodes the digest, so
    rendering the source again would only reproduce what was compiled from it.
    """
    _imports.ensure_triton_imported()
    suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
    # a sibling of triton's own cache
    triton_dir = Path(_imports.knobs.cache.get_triton_dir("intj"))
    directory = triton_dir / key.digest() / kernel_module
    so_path = directory / f"{key.module_name}{suffix}"
    if not so_path.exists():
        build(so_path, key)

    # Loading by hand, rather than through import_module, keeps the module out of
    # sys.modules.
    spec = importlib.util.spec_from_file_location(key.module_name, so_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"intj: cannot load {so_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # pyright: ignore[reportAttributeAccessIssue]  # 3.8 stubs lack it
    return module


def build(so_path: Path, key: RenderKey) -> None:
    """Renders and compiles, then installs both artifacts next to each other."""
    src = render(key)
    flags = build_flags(key)
    binary = _compile_so_bytes(src, key.module_name, flags)
    so_path.parent.mkdir(parents=True, exist_ok=True)
    # The source is kept next to the binary: it is what you read when a launch
    # misbehaves, and what you recompile by hand to debug it.
    suffix = ".cpp" if flags["language"] == "c++" else ".c"
    _install(src.encode(), so_path.with_name(f"{key.module_name}{suffix}"))
    _install(binary, so_path)


def _compile_so_bytes(src: str, name: str, flags: Mapping[str, Any]) -> bytes:
    """Compiles `src` into shared-object bytes with Triton's compiler.

    Falls back to its 3.7-era command when the helper is absent.
    """
    _imports.ensure_triton_imported()
    compile_so = getattr(_imports.build, "compile_so_from_src", None)
    if compile_so is not None:
        return Path(compile_so(src=src, name=name, **flags)).read_bytes()

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
            *_imports.knobs.build.backend_dirs,
        ]
        command = [
            compiler_path(language),
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
    """Returns a define that makes triton's compile cache key on the runtime headers.

    `_compile_so` keys on the source bytes plus the *names* of the include
    directories, not on the contents of the headers found there.  So an edit to
    a runtime header moves intj's own `RenderKey` digest, intj re-renders, and
    triton then serves the object it compiled from the previous headers.  Feeding
    the digest in as a define puts it in triton's key too.
    """
    digest = hashlib.sha256(runtime_headers()).hexdigest()[:16]
    return f"-DINTJRUNTIME_HEADER=0x{digest}"


def build_flags(key: RenderKey) -> dict[str, Any]:
    """Returns the compiler arguments beyond intj's own include directory.

    `digest_flags`, plus torch's include dirs for the c++ access mode: only a
    build reads those, and finding them imports `torch.utils.cpp_extension`.

    Raises:
        UnsupportedKernel: `cxx_lib_dirs` decided the c++ mode, and so the
            digest, without cpp_extension, and cpp_extension disagrees.
    """
    flags = digest_flags(key)
    if key.torch_access_mode != TorchAccessMode.STATIC_COMPILE.value:
        return flags
    toolchain = cxx_toolchain()
    if toolchain is None or tuple(toolchain[1]) != tuple(flags["library_dirs"]):
        from ..launcher import UnsupportedKernel  # not at top: launcher imports this

        found = (
            toolchain[1]
            if toolchain
            else "nothing (it failed to import, or found no torch headers)"
        )
        raise UnsupportedKernel(
            "intj: torch.utils.cpp_extension disagrees with the STATIC_COMPILE "
            f"toolchain intj found without it: library dirs {flags['library_dirs']}, "
            f"cpp_extension found {found}; report it, and pass "
            "torch_access_mode=TorchAccessMode.RUNTIME_SHIM meanwhile"
        )
    flags["include_dirs"] += toolchain[0]
    return flags


def digest_flags(key: RenderKey) -> dict[str, Any]:
    """Returns `build_flags` without torch's include dirs: what the digest records.

    The c++ access mode needs torch's headers, and `libc10` for the handful of
    error-path symbols `THPVariable_Unpack` pulls in.  Those are the reason it
    cannot be header-only -- torch loads libc10 `RTLD_LOCAL`, so an unlinked
    module builds fine and then fails at import with an undefined
    `throw_data_ptr_access_error`.  libtorch_cpu and libtorch_python are *not*
    needed.

    A kernel cache other than intj's is a C++ map, so it drags the whole module
    into C++ even where the tensor reader would not have.
    """
    torch_static_compile = key.torch_access_mode == TorchAccessMode.STATIC_COMPILE.value
    cxx_cache = key.kernel_cache != KernelCache.INTJ.value
    # No -falign-loops=64 (tests/test_kernel_cache.py uses it): measured
    # 2026-09-27, it leaves the launch path's instruction count unchanged -- the
    # memo-hit path runs no loop -- and moves rows +-1 ns either way by layout
    # alone.
    flags: dict[str, Any] = {
        "include_dirs": [str(RUNTIME), str(PYTHON_INTF), *key.cache_include_dirs],
        "ccflags": [
            _runtime_header_flag(),
            f"-DINTJ_CACHE_{key.kernel_cache.upper()}",
        ],
    }
    if key.cache_archives:
        # abseil's own link order is not intj's to encode, so the archives go in
        # a group and the linker sorts it out.
        flags["ccflags"] += [
            "-Wl,--start-group",
            *key.cache_archives,
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

    libs = cxx_lib_dirs()
    assert libs is not None, "check() validated this"
    flags["library_dirs"] = list(libs)
    flags["libraries"] = ["c10"]
    flags["ccflags"] += [
        # torch's headers make the translation unit ~7x bigger, and g++'s
        # inlining budget is per unit: past a size threshold it stops inlining
        # the tensor reader and PyFloat_AS_DOUBLE into the decode, costing a
        # real call per argument.  Measured: ~5 ns per launch on a 3-tensor
        # kernel.  Nothing here is about the torch code itself.
        f"-D_GLIBCXX_USE_CXX11_ABI={key.cxx_abi}",
        *(f"-Wl,-rpath,{d}" for d in libs),
    ]
    return flags


def _install(content: bytes, path: Path) -> None:
    """Writes `content` to `path` atomically, so a racing build cannot be half-read."""
    staged = path.with_name(f".{path.name}.{os.getpid()}")
    staged.write_bytes(content)
    os.replace(staged, path)

"""Builds a kernel's RenderKey and loads its rendered module.

Owns the process module cache, shared key layout, and ABI checks. The launcher
supplies its C stub and keeps Python compile state in its own instance.
"""

from __future__ import annotations

import dataclasses
import functools
import sysconfig
import threading
import types
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .. import _imports
from .. import backend as backends
from .. import kernel_cache as kernel_caches
from .._version import __version__
from ..annotation import (
    CanonicalAnnotation,
    DeviceBinding,
    KeyField,
    ResolvedParam,
    _key_fields,  # pyright: ignore[reportPrivateUsage]  # canonical key fields
    _layout_fields,  # pyright: ignore[reportPrivateUsage]  # shared key layout
)
from ..errors import UnsupportedKernel
from ..python_intf import cpython_abi
from ..torch_intf.torch_abi import (
    TorchAccessMode,
    dtype_index_table,
    layout_for,
    live_dtypes,
    supported_versions,
    torch_version,
)
from .build import (
    CppExtensionUnavailable,
    compiler_identity,
    cxx_lib_dirs,
    cxx_toolchain,
    digest_flags,
    load,
    runtime_headers,
)
from .context import ENTRY_TEMPLATE
from .key import DynamicSlot, Param, RenderKey

if TYPE_CHECKING:
    from ..launcher import _BuildOptions  # pyright: ignore[reportPrivateUsage]  # checked build inputs


def _build_module(b: _BuildOptions, stub: types.ModuleType) -> types.ModuleType:  # pyright: ignore[reportUnusedFunction]  # consumed by launcher
    """Returns the loaded module for this kernel and configuration.

    Selects the backend and its target (`no_gpu`'s is `HOST`, with none), and
    loads, building if needed, the module for its `RenderKey`.
    `torch_access_request` is what `torch_access_mode` was resolved from, for
    choosing again where a build finds STATIC_COMPILE unavailable after all.
    `stub` supplies the exact C launcher type the module must accept.

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
            target,
            backends._knob_options(b.jit_func, options, knob_values),  # pyright: ignore[reportPrivateUsage]  # shared compile options
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
    cache_toolchain = kernel_caches._provision(b.kernel_cache)  # pyright: ignore[reportPrivateUsage]  # checked toolchain
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
        # `loaded` fills in these and `compiler`, `build_flags`
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

    def loaded(mode: TorchAccessMode) -> types.ModuleType:
        """Returns the initialized module for this access mode, cached once."""
        # the c++ mode gets it too, not to read from but to check its own
        # compiled-in offset against
        layout = (
            layout_for()
            if mode in (TorchAccessMode.RUNTIME_SHIM, TorchAccessMode.STATIC_COMPILE)
            else None
        )
        static = mode is TorchAccessMode.STATIC_COMPILE
        access_key = dataclasses.replace(
            key,
            torch_access_mode=mode.value,
            # Only the c++ mode compiles anything torch-specific in, so only it has
            # to be rebuilt when torch changes.  Leaving these None for the other
            # two is what keeps their digest -- and so their cached `.so` -- stable
            # across torch versions.
            torch_version=torch_version() if static else None,
            cxx_abi=_cxx_abi() if static else None,
        )
        # `digest_flags` derives these from the fields above
        flags = digest_flags(access_key)
        access_key = dataclasses.replace(
            access_key,
            compiler=compiler_identity(flags["language"]),
            build_flags=tuple(flags["ccflags"]),
        )
        # Without the lock two threads both compile and load, and the loser's
        # module is silently dropped. Publish only after initialization succeeds.
        with _LOAD_LOCK:
            module = _LOADED.get(access_key)
            if module is None:
                module = load(access_key, b.jit_func.__module__)
                if module._header_size() != stub._header_size():
                    raise ImportError(
                        f"intj: {module.__file__} disagrees with _intj_lazy on the launcher "
                        "header; it was built by another intj -- delete it to rebuild"
                    )
                module._set_launcher_type(stub.Launcher)
                # Pass relative cdata; the compiled setter saves the absolute offset
                # for the torch running now before exposing the module.
                module._set_torch_version(
                    torch_version(),
                    layout.as_args() if layout else None,
                    dtype_index_table(),
                )
                _LOADED[access_key] = module
            return module

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


def _cache_version() -> tuple[int, int]:
    major, minor, *_ = __version__.split(".")
    return int(major), int(minor)


def _cxx_abi() -> int:
    """Returns torch's `_GLIBCXX_USE_CXX11_ABI`.

    A mismatch here links, then misbehaves at runtime.
    """
    _imports.ensure_torch_imported()
    return int(_imports.torch._C._GLIBCXX_USE_CXX11_ABI)


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
    build must (`render.module._build_module`). The two differ only where
    cpp_extension is installed but cannot import.
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


_LOADED: dict[RenderKey, types.ModuleType] = {}
_LOAD_LOCK = threading.Lock()


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


def _dynamic_key_fields(name: str) -> tuple[KeyField, ...]:
    """Returns one dynamic value's key fields: a kind byte and 8 value bytes.

    Laid out like an exact key.  `name` picks nothing here; the invariant
    tests' mutation checks drop one value's fields by name.
    """
    del name
    return (KeyField("exact_kind", 1), KeyField("exact", 8))


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

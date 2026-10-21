"""Builds a kernel's RenderKey and loads its rendered module.

The launcher owns the module cache and compile callbacks. Shared launcher
checks are imported when building, after its module has initialized.
"""

from __future__ import annotations

import dataclasses
import functools
import sysconfig
import types
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .. import _imports
from .. import backend as backends
from .._version import __version__
from ..python_intf import cpython_abi
from ..torch_intf.torch_abi import (
    TorchAccessMode,
    layout_for,
    live_dtypes,
    torch_version,
)
from .build import (
    CppExtensionUnavailable,
    compiler_identity,
    digest_flags,
    runtime_headers,
)
from .context import ENTRY_TEMPLATE
from .key import Param, RenderKey

if TYPE_CHECKING:
    from ..launcher import _BuildOptions  # pyright: ignore[reportPrivateUsage]  # checked build inputs


def _build_module(b: _BuildOptions) -> types.ModuleType:  # pyright: ignore[reportUnusedFunction]  # consumed by launcher
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
    from ..launcher import (
        UnsupportedKernel,
        get_full_name,
        _knob_options,  # pyright: ignore[reportPrivateUsage]  # shared compile options
        _loaded_module,  # pyright: ignore[reportPrivateUsage]  # launcher's module cache
        _make_compile_callback,  # pyright: ignore[reportPrivateUsage]  # shared compiler
        _make_host_compile_callback,  # pyright: ignore[reportPrivateUsage]  # host compiler
        _provision,  # pyright: ignore[reportPrivateUsage]  # checked toolchain
        _resolve_torch_access_mode,  # pyright: ignore[reportPrivateUsage]  # shared checks
        _spec_key_layout,  # pyright: ignore[reportPrivateUsage]  # shared key layout
        _verified_cpython_header,  # pyright: ignore[reportPrivateUsage]  # shared ABI check
    )

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
            _for_access(key, mode),
            b.jit_func.__module__,
            layout,
            compile_function if tuning is None else None,
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

"""The driver APIs intj launches through, and the triton target a build is for."""

from __future__ import annotations

import abc
from typing import Any

from . import _imports
from .errors import UnsupportedKernel


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
        """Returns the driver dylib to dlopen -- the same one triton itself uses."""


class HipBackend(Backend):
    """The HIP driver API, for triton's AMD backend."""

    name = "hip"
    launch_symbol = "hipModuleLaunchKernel"
    device_symbol = "hipDeviceGet"
    error_symbol = "hipGetErrorString"
    error_style = "return"
    pointer_range = True

    def library_path(self) -> str:
        # private, but it is the resolution order triton itself launches through;
        # imported here, as a triton built without the AMD backend lacks it
        from triton.backends.amd import driver

        return driver._get_path_to_hip_runtime_dylib()  # pyright: ignore[reportPrivateUsage]


class CudaBackend(Backend):
    """The CUDA driver API, for triton's NVIDIA backend.

    Compile-checked only; untested on a GPU.
    """

    name = "cuda"
    launch_symbol = "cuLaunchKernel"
    device_symbol = "cuDeviceGet"
    error_symbol = "cuGetErrorString"
    error_style = "outparam"
    pointer_range = False

    def library_path(self) -> str:
        return "libcuda.so.1"  # already mapped by torch, resolved through the normal search path


class HostBackend(Backend):
    """A fake driver on the CPU, for `no_gpu=True`: every launch does nothing.

    Its symbols are `_intj_launcher_base`'s (runtime/intj_launcher_base.c),
    loaded before any rendered module. No triton target names it, so it is
    not registered: only `select(no_gpu=True)` returns it.
    """

    name = "host"
    launch_symbol = "intj_host_launch"
    device_symbol = "intj_host_device_get"
    error_symbol = "intj_host_error_string"
    error_style = "return"
    pointer_range = False

    def library_path(self) -> str:
        # deferred: launcher imports this module
        from .launcher import _stub  # pyright: ignore[reportPrivateUsage]  # the process's C launcher base

        path = _stub().__file__
        assert path is not None, "_intj_launcher_base is loaded from a file"
        return path


HOST = HostBackend()

BACKENDS: dict[str, Backend] = {}


def register(backend: Backend) -> Backend:
    """Makes `backend` usable by `make_launcher`, replacing any same-named one.

    Returns:
        `backend` itself.
    """
    BACKENDS[backend.name] = backend
    return backend


register(HipBackend())
register(CudaBackend())


def select(no_gpu: bool) -> tuple[Backend, Any]:
    """Returns the backend a build launches through, and the triton target it compiles for.

    `HOST` has no target: it compiles nothing.

    Raises:
        UnsupportedKernel: There is no active target, or its backend is not
            registered.
    """
    if no_gpu:
        return HOST, None
    target = current_target()
    backend = BACKENDS.get(target.backend)
    if backend is None:
        raise UnsupportedKernel(
            f"intj: backend {target.backend!r} is unknown; subclass intj.backend.Backend and "
            f"register() it (have: {', '.join(sorted(BACKENDS))})"
        )
    return backend, target


def current_target() -> Any:
    """Returns the active triton target, which triton types as optional.

    Raises:
        UnsupportedKernel: There is no active target (no visible GPU).
    """
    _imports.ensure_triton_imported()
    target = _imports.driver.active.get_current_target()
    if target is None:
        raise UnsupportedKernel("intj: no active triton target; is a GPU visible?")
    return target


def current_device() -> int:
    _imports.ensure_triton_imported()
    # get_current_device() is on every concrete driver, just not on DriverBase
    return _imports.driver.active.get_current_device()

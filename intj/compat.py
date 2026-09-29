"""Routes existing Triton-style host calls through ``make_launcher``.

This keeps the native launcher's positional fast path unchanged.  New hot call
sites should construct and call ``make_launcher`` directly.
"""

from __future__ import annotations

import threading
from functools import lru_cache
from typing import Any

from triton._utils import canonicalize_dtype
from triton import knobs
from triton import language as tl
from triton.runtime.driver import driver
from triton.runtime.jit import JITFunction, TensorWrapper

from .annotation import Argument, Constexpr
from .launcher import UnsupportedKernel, make_launcher

try:
    from triton._compile_warmup_state import is_compile_warmup as _is_compile_warmup  # pyright: ignore[reportMissingImports]  # optional Triton test runtime
except ImportError:  # Triton versions without compile-warmup mode
    _is_compile_warmup = None


def _pointer_type(dtype: Any) -> Any:
    try:
        if not isinstance(dtype, tl.dtype):
            dtype = tl.str_to_ty(canonicalize_dtype(dtype), None)
        return tl.pointer_type(dtype)
    except (KeyError, TypeError, ValueError) as error:
        raise UnsupportedKernel(
            f"intj: unsupported TensorWrapper dtype {dtype!r}"
        ) from error


def _annotations(
    baked: tuple[tuple[str, Any], ...], wrapped_types: tuple[tuple[str, Any], ...]
) -> dict[str, Argument | Constexpr]:
    return {
        **{name: Constexpr(value=value) for name, value in baked},
        **{name: Argument(type=_pointer_type(dtype)) for name, dtype in wrapped_types},
    }


def _make(
    kernel: Any,
    device: int,
    dimensions: int,
    options: tuple[tuple[str, Any], ...],
    baked: tuple[tuple[str, Any], ...],
    wrapped_types: tuple[tuple[str, Any], ...],
    return_compiled: bool,
) -> Any:
    return make_launcher(
        kernel,
        grid_arg=dimensions,
        options=dict(options),
        extra_annotation=_annotations(baked, wrapped_types),
        bind_device=True,
        return_compiled=return_compiled,
    ).bind_device(device)


@lru_cache(maxsize=256)
def _cached(
    kernel: Any,
    source_key: str,
    device: int,
    dimensions: int,
    options: tuple[tuple[str, Any], ...],
    baked: tuple[tuple[str, Any], ...],
    wrapped_types: tuple[tuple[str, Any], ...],
    return_compiled: bool,
    target: tuple[str, str, int],
    debug: object,
    instrumentation: object,
    fpsan_casts: object,
) -> Any:
    del source_key, target, debug, instrumentation, fpsan_casts
    return _make(
        kernel, device, dimensions, options, baked, wrapped_types, return_compiled
    )


#: the grid and TensorWrappers of the `launch` running on this thread
_CALL = threading.local()


def _calling_grid(meta: dict[str, object]) -> object:
    """Evaluates the calling launch's own grid, with its own wrappers.

    This is every cached callable-grid launcher's grid_py, so the launcher keys
    on neither and holds neither (a per-call lambda would miss every call and
    pin its captures).
    """
    return _CALL.grid({**meta, **_CALL.wrapped})


@lru_cache(maxsize=256)
def _cached_grid_py(
    kernel: Any,
    source_key: str,
    options: tuple[tuple[str, Any], ...],
    baked: tuple[tuple[str, Any], ...],
    wrapped_types: tuple[tuple[str, Any], ...],
    return_compiled: bool,
    target: tuple[str, str, int],
    debug: object,
    instrumentation: object,
    fpsan_casts: object,
) -> Any:
    """Returns a callable-grid launcher, reused across calls like `_cached`'s.

    Each launcher owns its kernel cache, so a fresh one per call would miss
    every time.
    """
    del source_key, target, debug, instrumentation, fpsan_casts
    return _make_grid_py(kernel, options, baked, wrapped_types, return_compiled)


def _make_grid_py(
    kernel: Any,
    options: tuple[tuple[str, Any], ...],
    baked: tuple[tuple[str, Any], ...],
    wrapped_types: tuple[tuple[str, Any], ...],
    return_compiled: bool,
) -> Any:
    return make_launcher(
        kernel,
        grid_py=_calling_grid,
        options=dict(options),
        extra_annotation=_annotations(baked, wrapped_types),
        return_compiled=return_compiled,
    )


def launch(
    kernel: Any, grid: Any, /, *args: Any, return_compiled: bool = False, **kwargs: Any
) -> Any:
    """Launches a JIT function with Triton's call spelling, `kernel[grid](...)`.

    This is a migration bridge, not the low-overhead API: it binds Python
    keywords and reads the current device/stream on every call.  Unsupported
    kernels and options still raise rather than falling back to Triton.

    Args:
        kernel: A `@triton.jit` function.
        grid: An int, a tuple or list of 1 to 3 ints, or a callable over the
            kernel's meta-parameters.
        *args: The kernel's arguments, as Triton takes them.
        return_compiled: Whether the launch returns the `CompiledKernel`.
        **kwargs: Kernel arguments by keyword; any other name is a compile option.

    Returns:
        Whatever the underlying launcher returns.

    Raises:
        UnsupportedKernel: `kernel` is not a `JITFunction`, Triton launch hooks
            are active, or the launcher refuses the kernel.
        TypeError: `grid` has an unsupported type.
        ValueError: `grid` has the wrong rank, or a pointer argument is an
            unpinned CPU tensor.
    """
    if _is_compile_warmup is not None and _is_compile_warmup():
        # The test runtime intercepts indexed calls to compile with fake pointers.
        return kernel[grid](*args, **kwargs)
    hooks = (knobs.runtime.launch_enter_hook, knobs.runtime.launch_exit_hook)
    # HookChain is truthy even when no callbacks are registered.
    if any(getattr(hook, "calls", hook) for hook in hooks):
        raise UnsupportedKernel("intj: active Triton launch hooks are not supported")
    if not isinstance(kernel, JITFunction):
        raise UnsupportedKernel(
            f"intj: expected a @triton.jit function, got {type(kernel).__name__}"
        )

    kernel_names = kernel.signature.parameters
    kernel_kwargs = {
        name: value for name, value in kwargs.items() if name in kernel_names
    }
    options = tuple(
        sorted(
            (name, value) for name, value in kwargs.items() if name not in kernel_names
        )
    )
    bound = kernel.signature.bind(*args, **kernel_kwargs)
    bound.apply_defaults()
    baked = tuple(
        (param.name, bound.arguments[param.name])
        for param in kernel.params
        if param.is_constexpr
        and (
            isinstance(bound.arguments[param.name], (tl.dtype, JITFunction))
            or type(bound.arguments[param.name]) is str
        )
    )
    baked_names = {name for name, _ in baked}
    wrapped = {
        param.name: bound.arguments[param.name]
        for param in kernel.params
        if not param.is_constexpr
        and isinstance(bound.arguments[param.name], TensorWrapper)
    }
    wrapped_types = tuple((name, value.dtype) for name, value in wrapped.items())
    values = tuple(
        wrapped[name].base if name in wrapped else value
        for name, value in bound.arguments.items()
        if name not in baked_names
    )

    # Triton's driver rejects ordinary CPU pointers; an empty kernel would
    # otherwise make the invalid intj launch appear to succeed.
    import torch

    for name, value in bound.arguments.items():
        tensor = value.base if isinstance(value, TensorWrapper) else value
        if (
            isinstance(tensor, torch.Tensor)
            and tensor.device.type == "cpu"
            and tensor.data_ptr() != 0
            and not tensor.is_pinned()
        ):
            raise ValueError(
                f"intj: pointer argument {name!r} cannot be accessed from Triton (cpu tensor?)"
            )

    device = driver.active.get_current_device()  # pyright: ignore[reportAttributeAccessIssue]  # concrete drivers provide it
    stream = driver.active.get_current_stream(device)  # pyright: ignore[reportAttributeAccessIssue]  # concrete drivers provide it
    if callable(grid):
        key = (
            kernel,
            kernel.cache_key,
            options,
            baked,
            wrapped_types,
            return_compiled,
            _target_key(),
            *_knob_key(),
        )
        try:
            hash(key)
        except TypeError:
            native = _make_grid_py(
                kernel, options, baked, wrapped_types, return_compiled
            )
        else:
            native = _cached_grid_py(*key)
        # the grid may launch again: restore after, and drop this call's refs
        previous = getattr(_CALL, "grid", None), getattr(_CALL, "wrapped", None)
        _CALL.grid, _CALL.wrapped = grid, wrapped
        try:
            return native(device, stream, *values)
        finally:
            _CALL.grid, _CALL.wrapped = previous
    if type(grid) is int:
        dimensions = (grid,)
    elif type(grid) in (tuple, list):
        dimensions = tuple(grid)
    else:
        raise TypeError("intj: grid must be an int, tuple, list or callable")
    if not 1 <= len(dimensions) <= 3:
        raise ValueError("intj: grid must have between 1 and 3 dimensions")

    cache_key = (
        kernel,
        kernel.cache_key,
        device,
        len(dimensions),
        options,
        baked,
        wrapped_types,
        return_compiled,
        _target_key(),
        *_knob_key(),
    )
    try:
        hash(cache_key)
    except TypeError:
        native = _make(
            kernel,
            device,
            len(dimensions),
            options,
            baked,
            wrapped_types,
            return_compiled,
        )
    else:
        native = _cached(*cache_key)
    return native(stream, *dimensions, *values)


def _target_key() -> tuple[str, str, int]:
    target = driver.active.get_current_target()
    if target is None:
        raise UnsupportedKernel("intj: no active Triton target")
    return (target.backend, repr(target.arch), target.warp_size)


def _knob_key() -> tuple[object, object, object]:
    """Returns the knobs a launcher reads at its first call; each value keys one."""
    return (
        knobs.runtime.debug,
        knobs.compilation.instrumentation_mode,
        getattr(knobs.compilation, "fpsan_homomorphic_casts", None),
    )


def launch_or_interpret(
    kernel: Any, grid: Any, /, *args: Any, return_compiled: bool = False, **kwargs: Any
) -> Any:
    """Launches through Triton while its interpreter is enabled, else via `launch`."""
    if knobs.runtime.interpret:
        return kernel[grid](*args, **kwargs)
    return launch(kernel, grid, *args, return_compiled=return_compiled, **kwargs)

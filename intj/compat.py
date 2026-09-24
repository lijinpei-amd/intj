"""Route existing Triton-style host calls through ``make_launcher``.

This keeps the native launcher's positional fast path unchanged.  New hot call
sites should construct and call ``make_launcher`` directly.
"""

from __future__ import annotations

from copy import copy
from functools import lru_cache
from types import SimpleNamespace
from typing import Any

from triton._utils import canonicalize_dtype
from triton import knobs
from triton import language as tl
from triton.runtime.autotuner import Autotuner, Heuristics
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
        raise UnsupportedKernel(f"intj: unsupported TensorWrapper dtype {dtype!r}") from error


def _annotations(baked: tuple[tuple[str, Any], ...],
                 wrapped_types: tuple[tuple[str, Any], ...]) -> dict[str, Argument | Constexpr]:
    return ({name: Constexpr(value=value) for name, value in baked} |
            {name: Argument(type=_pointer_type(dtype)) for name, dtype in wrapped_types})


def _make(kernel: Any, device: int, dimensions: int,
          options: tuple[tuple[str, Any], ...], baked: tuple[tuple[str, Any], ...],
          wrapped_types: tuple[tuple[str, Any], ...], return_compiled: bool) -> Any:
    return make_launcher(kernel, grid_arg=dimensions, options=dict(options),
                         extra_annotation=_annotations(baked, wrapped_types),
                         bind_device=True, return_compiled=return_compiled).bind_device(device)


@lru_cache(maxsize=256)
def _cached(kernel: Any, source_key: str, device: int, dimensions: int,
            options: tuple[tuple[str, Any], ...], baked: tuple[tuple[str, Any], ...],
            wrapped_types: tuple[tuple[str, Any], ...], return_compiled: bool,
            target: tuple[str, str, int],
            debug: object, instrumentation: object, fpsan_casts: object) -> Any:
    del source_key, target, debug, instrumentation, fpsan_casts
    return _make(kernel, device, dimensions, options, baked, wrapped_types, return_compiled)


def launch(kernel: Any, grid: Any, /, *args: Any, return_compiled: bool = False,
           **kwargs: Any) -> Any:
    """Launch a JIT function or decorated wrapper with Triton's call spelling.

    This is a migration bridge, not the low-overhead API: it binds Python
    keywords and reads the current device/stream on every call.  Unsupported
    kernels and options still raise rather than falling back to Triton.
    """
    if _is_compile_warmup is not None and _is_compile_warmup():
        # The test runtime intercepts indexed calls to compile with fake pointers.
        return kernel[grid](*args, **kwargs)
    hooks = (knobs.runtime.launch_enter_hook, knobs.runtime.launch_exit_hook)
    # HookChain is truthy even when no callbacks are registered.
    if any(getattr(hook, "calls", hook) for hook in hooks):
        raise UnsupportedKernel("intj: active Triton launch hooks are not supported")
    while isinstance(kernel, Heuristics):
        for name, heuristic in kernel.values.items():
            kwargs[name] = heuristic({**dict(zip(kernel.arg_names, args)), **kwargs})
        kernel = kernel.fn
    if isinstance(kernel, Autotuner):
        # Reuse Triton's selection and hooks, redirecting only its inner launch.
        tuned = copy(kernel)

        def run_native(*run_args: Any, **run_kwargs: Any) -> Any:
            return launch(kernel.fn, run_kwargs.pop("grid"), *run_args,
                          return_compiled=return_compiled, **run_kwargs)

        tuned.fn = SimpleNamespace(fn=kernel.fn, run=run_native)
        try:
            result = tuned.run(*args, **kwargs, grid=grid)
        finally:
            for name in ("best_config", "bench_time", "configs_timings"):
                if name in tuned.__dict__:
                    setattr(kernel, name, getattr(tuned, name))
        return result
    if not isinstance(kernel, JITFunction):
        raise UnsupportedKernel(f"intj: expected a @triton.jit function, got {type(kernel).__name__}")

    kernel_names = kernel.signature.parameters
    kernel_kwargs = {name: value for name, value in kwargs.items() if name in kernel_names}
    options = tuple(sorted((name, value) for name, value in kwargs.items()
                           if name not in kernel_names))
    bound = kernel.signature.bind(*args, **kernel_kwargs)
    bound.apply_defaults()
    baked = tuple((param.name, bound.arguments[param.name]) for param in kernel.params
                  if param.is_constexpr and (
                      isinstance(bound.arguments[param.name], (tl.dtype, JITFunction)) or
                      type(bound.arguments[param.name]) is str))
    baked_names = {name for name, _ in baked}
    wrapped = {param.name: bound.arguments[param.name] for param in kernel.params
               if not param.is_constexpr and isinstance(bound.arguments[param.name], TensorWrapper)}
    wrapped_types = tuple((name, value.dtype) for name, value in wrapped.items())
    values = tuple(wrapped[name].base if name in wrapped else value
                   for name, value in bound.arguments.items() if name not in baked_names)

    # Triton's driver rejects ordinary CPU pointers; an empty kernel would
    # otherwise make the invalid intj launch appear to succeed.
    import torch

    for name, value in bound.arguments.items():
        tensor = value.base if isinstance(value, TensorWrapper) else value
        if (isinstance(tensor, torch.Tensor) and tensor.device.type == "cpu" and
                tensor.data_ptr() != 0 and not tensor.is_pinned()):
            raise ValueError(f"intj: pointer argument {name!r} cannot be accessed from Triton (cpu tensor?)")

    device = driver.active.get_current_device()  # pyright: ignore[reportAttributeAccessIssue]  # concrete drivers provide it
    stream = driver.active.get_current_stream(device)  # pyright: ignore[reportAttributeAccessIssue]  # concrete drivers provide it
    if callable(grid):
        def wrapped_grid(meta: dict[str, object]) -> object:
            return grid({**meta, **wrapped})

        grid_py = wrapped_grid if wrapped else grid
        native = make_launcher(kernel, grid_py=grid_py, options=dict(options),
                               extra_annotation=_annotations(baked, wrapped_types),
                               return_compiled=return_compiled)
        return native(device, stream, *values)
    if type(grid) is int:
        dimensions = (grid,)
    elif type(grid) in (tuple, list):
        dimensions = tuple(grid)
    else:
        raise TypeError("intj: grid must be an int, tuple, list or callable")
    if not 1 <= len(dimensions) <= 3:
        raise ValueError("intj: grid must have between 1 and 3 dimensions")

    target = driver.active.get_current_target()
    if target is None:
        raise UnsupportedKernel("intj: no active Triton target")
    target_key = (target.backend, repr(target.arch), target.warp_size)
    cache_key = (kernel, kernel.cache_key, device, len(dimensions), options, baked, wrapped_types,
                 return_compiled,
                 target_key, knobs.runtime.debug, knobs.compilation.instrumentation_mode,
                 getattr(knobs.compilation, "fpsan_homomorphic_casts", None))
    try:
        hash(cache_key)
    except TypeError:
        native = _make(kernel, device, len(dimensions), options, baked, wrapped_types,
                       return_compiled)
    else:
        native = _cached(*cache_key)
    return native(stream, *dimensions, *values)


def launch_or_interpret(kernel: Any, grid: Any, /, *args: Any,
                        return_compiled: bool = False, **kwargs: Any) -> Any:
    """Use Triton's launcher only while its interpreter is enabled."""
    if knobs.runtime.interpret:
        return kernel[grid](*args, **kwargs)
    return launch(kernel, grid, *args, return_compiled=return_compiled, **kwargs)

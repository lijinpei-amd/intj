"""Render, build and load the C launcher for a `@triton.jit` kernel."""

import functools
import hashlib
import inspect
import json
import os
import struct
import subprocess
import sysconfig
from pathlib import Path

SCHEMA_VERSION = 1

_TEMPLATES = Path(__file__).parent / "templates"
_ENTRY_TEMPLATE = _TEMPLATES / "entry.c.jinja"
_RUNTIME_HEADER = _TEMPLATES / "intj_runtime.h"

# Spec-key tags, kept in sync with templates/intj_runtime.h.
T_NONE, T_ONE, T_I32, T_I64, T_U64, T_FP32, T_U1, T_PTR = range(1, 9)
T_CX_NONE, T_CX_BOOL, T_CX_INT, T_CX_UINT, T_CX_FLOAT = range(9, 14)


class UnsupportedKernel(NotImplementedError):
    """Raised for kernels or options outside intj's (deliberately small) scope."""


def create_launcher(jit_func, dynamic_grid=False, dynamic_options=(), extra_annotation=None, options=None):
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
    from triton.backends.amd.driver import _get_path_to_hip_runtime_dylib
    from triton.runtime.build import compile_module_from_src
    from triton.runtime.driver import driver

    target = driver.active.get_current_target()
    context = {
        "kernel_repr": f"{jit_func.module}.{jit_func.__qualname__}",
        "params": params,
        "nwords": 1 + sum(2 if p["is_constexpr"] else 1 for p in params),
        "max_slots": sum(0 if p["is_constexpr"] else 1 for p in params),
        # The pointer-range bit is always part of the key, even when buffer ops
        # are off: the knob is read by triton per specialization, so a key that
        # cannot tell a > 2 GiB buffer apart would launch a tt.pointer_range=32
        # binary on it after the knob is flipped back on.
        "spec_pointer_range": 1,
        "libhip_path": _get_path_to_hip_runtime_dylib(),
        "libtorch_path": os.path.join(os.path.dirname(torch.__file__), "lib", "libtorch_cpu.so"),
    }

    src = _render(context, module_name="intj_module")
    digest = hashlib.sha256(
        json.dumps(
            {
                "schema": SCHEMA_VERSION,
                "src": src,
                "runtime_header": _RUNTIME_HEADER.read_bytes().hex(),
                "cache_key": jit_func.cache_key,
                "params": [
                    [p.name, p.annotation, p.is_constexpr, p.is_const, p.do_not_specialize,
                     p.do_not_specialize_on_alignment, p.has_default, repr(p.default)]
                    for p in jit_func.params
                ],
                "target": [target.backend, target.arch, target.warp_size],
                "options": options,
                "triton": _triton_identity(),
                "compiler": _compiler_identity(),
                "ext_suffix": sysconfig.get_config_var("EXT_SUFFIX"),
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()

    module_name = "intj_" + digest[:32]
    module = compile_module_from_src(
        src=_render(context, module_name=module_name),
        name=module_name,
        include_dirs=[str(_TEMPLATES)],
        language="c",
    )
    module.set_compile_callback(_make_compile_callback(jit_func, params, options))
    return module.entry


# --------------------------------------------------------------------- checks


def _check_kernel(jit_func, options):
    from triton import knobs
    from triton.runtime.driver import driver
    from triton.runtime.jit import JITFunction

    if knobs.runtime.interpret:
        raise UnsupportedKernel("intj: TRITON_INTERPRET=1 is not supported")
    if not isinstance(jit_func, JITFunction):
        raise UnsupportedKernel(
            f"intj: expected a @triton.jit function, got {type(jit_func).__name__}; "
            "autotuned and heuristic kernels are not supported"
        )
    target = driver.active.get_current_target()
    if target.backend != "hip":
        raise UnsupportedKernel(f"intj: only the AMD (hip) backend is supported, got {target.backend!r}")
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


def _render_params(jit_func):
    """One render-time descriptor per declared kernel parameter."""
    params, word = [], 1
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
            {
                "name": p.name,
                "is_constexpr": p.is_constexpr,
                "spec": 0 if p.do_not_specialize else 1,
                "align": 0 if p.do_not_specialize_on_alignment else 1,
                "word": word,
            }
        )
        word += 2 if p.is_constexpr else 1
    return params


# -------------------------------------------------------------------- render


def _render(context, module_name):
    import jinja2

    template = jinja2.Template(_ENTRY_TEMPLATE.read_text(), undefined=jinja2.StrictUndefined)
    return template.render(module_name=module_name, **context)


@functools.lru_cache(maxsize=1)
def _triton_identity():
    import triton

    libtriton = Path(triton.__file__).parent / "_C" / "libtriton.so"
    stat = libtriton.stat()
    return [triton.__version__, stat.st_size, stat.st_mtime]


@functools.lru_cache(maxsize=1)
def _compiler_identity():
    from triton.runtime import build

    cc = build._find_compiler("c")
    cc = cc[0] if isinstance(cc, tuple) else cc
    try:
        version = subprocess.run([cc, "--version"], capture_output=True, text=True).stdout.splitlines()[0]
    except Exception:  # pragma: no cover - compiler without --version
        version = ""
    return [str(cc), version]


# ------------------------------------------------------------- compile hook


def _make_compile_callback(jit_func, params, options):
    """Called from C on a spec-key miss, with the key blob and the original args."""
    kernels = []  # keeps every CompiledKernel (and therefore its hipModule) alive
    seen = {}  # (param index, key word(s)) -> triton's specialization entry

    def compile_callback(keyblob, nparams, device, *args):
        from triton.runtime.driver import driver

        current = driver.active.get_current_device()
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


def triton_specialization(jit_func, args, options=None):
    """The `list[(type_str, key)]` triton would compute for these arguments."""
    from triton import knobs
    from triton.runtime.driver import driver

    device = driver.active.get_current_device()
    binder = jit_func.device_caches[device][4]
    kwargs = dict(options or {})
    kwargs["debug"] = jit_func.debug or knobs.runtime.debug
    kwargs["instrumentation_mode"] = knobs.compilation.instrumentation_mode
    return binder(*args, **kwargs)[1]


def _validate_spec_key(jit_func, params, options, keyblob, args, seen):
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
        n = 2 if param["is_constexpr"] else 1
        mine = words[param["word"]:param["word"] + n]
        theirs = repr(specialization[i])
        previous = seen.setdefault((i, mine), theirs)
        if previous != theirs:
            raise RuntimeError(
                f"intj: spec key for parameter {param['name']!r} is too coarse: it maps both "
                f"{previous} and {theirs} to the same key; this is an intj bug"
            )

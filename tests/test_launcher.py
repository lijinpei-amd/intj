# pyright: standard
"""Differential tests: every launch is compared against triton's own launcher.

Run with `pytest tests` on a machine with an AMD GPU, torch and triton.
"""

from __future__ import annotations

import ctypes
import dataclasses
import dis
import functools
import gc
import importlib.util
import inspect
import itertools
import json
import os
import pathlib
import re
import struct
import subprocess
import sys
import sysconfig
import types
import warnings
import weakref

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib  # pyright: ignore[reportMissingImports]  # dependency on Python <3.11

import pytest
import torch
import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

import intj
from intj import launcher, make_launcher
from intj import (
    AUTO,
    NEVER,
    Aligned,
    Argument,
    Assume,
    BindValue,
    Constexpr,
    EqualTo,
    Exactly,
    PointerRange,
)
from intj.annotation import CanonicalAnnotation
from intj.launcher import (
    UnsupportedKernel,
    module_of,
    override_compile,
    triton_specialization,
)
from knob_helpers import (
    ON_HIP,
    POINTER_RANGE,
    live_knob_values,
    needs_hip,
    needs_pointer_range,
)
from intj.kernel_cache import INSTALL_ERRORS, KernelCache, install
from intj.python_intf import cpython_abi
from intj.render.build import build, compiler_identity
from intj.render.key import DynamicSlot, RenderKey
from intj.torch_intf.abi_detect import probe_layout
from intj.torch_intf.torch_abi import (
    NDTYPES,
    TorchAccessMode,
    dtype_code,
    dtype_index_table,
    itemsize_table,
    layout_for,
)


def _tested_intj_env():
    # A subprocess must import the intj under test: the checkout, or the
    # installed wheel when ci/ runs the suite against one.
    return {**os.environ, "PYTHONPATH": str(pathlib.Path(intj.__file__).parents[1])}


def test_subprocesses_import_the_tested_intj(tmp_path):
    script = tmp_path / "which_intj.py"
    script.write_text("import intj\nprint(intj.__file__)\n")
    run = subprocess.run(
        [sys.executable, str(script)],
        cwd=pathlib.Path(__file__).parents[1],
        env=_tested_intj_env(),
        capture_output=True,
        text=True,
    )
    assert run.returncode == 0, run.stderr
    assert run.stdout.strip() == intj.__file__


@pytest.mark.parametrize("sweep", [False, True])
def test_benchmark_matrix_runs_without_gpu(sweep):
    run = subprocess.run(
        [
            sys.executable,
            "benchmarks/bench_launch.py",
            "--no-gpu",
            *(["--sweep"] if sweep else []),
            "--iters",
            "20",
            "--batches",
            "3",
        ],
        cwd=pathlib.Path(__file__).parents[1],
        env=_tested_intj_env(),
        capture_output=True,
        text=True,
    )
    assert run.returncode == 0, run.stderr
    if sweep:
        assert "mode=sweep; host-only" in run.stdout
        rows = [line.split() for line in run.stdout.splitlines()[2:]]
        assert [(int(count), kind) for count, kind, _ in rows] == [
            (count, kind) for count in (4, 16, 32) for kind in ("int", "tensor")
        ]
        assert all(float(ns) > 0 for _, _, ns in rows)
        return
    for label in (
        "auto map",
        "reduced key",
        "verify off",
        "verify on",
        "baked",
        "bound tensor",
        "bound pointer",
        "fixed device map",
        "fixed device no-map",
    ):
        assert label in run.stdout


def test_readme_benchmark_matrix_runs():
    run = subprocess.run(
        [
            sys.executable,
            "benchmarks/bench_launch.py",
            "--readme",
            "--iters",
            "2",
            "--batches",
            "1",
        ],
        cwd=pathlib.Path(__file__).parents[1],
        env=_tested_intj_env(),
        capture_output=True,
        text=True,
    )
    assert run.returncode == 0, run.stderr
    for label in (
        "grid=(1,)",
        "grid=(0,)",
        "runtime_shim",
        "static_compile",
        "interpreter",
    ):
        assert label in run.stdout


@triton.jit
def axpy(x, y, o, n, a, flag, bias, BLOCK: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = off < n
    v = tl.load(x + off, mask=mask) * a
    if flag:
        v += tl.load(y + off, mask=mask)
    if bias is not None:
        v += bias
    tl.store(o + off, v, mask=mask)


@triton.jit
def axpy_tensor(
    x: torch.Tensor,
    y: tl.tensor,  # pyright: ignore[reportInvalidTypeForm]  # triton's tensor class
    o: torch.Tensor,
    n,
    a,
    flag,
    bias,
    BLOCK: tl.constexpr,
):
    """`axpy` with its pointers annotated as tensors, which Triton ignores."""
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = off < n
    v = tl.load(x + off, mask=mask) * a
    if flag:
        v += tl.load(y + off, mask=mask)
    if bias is not None:
        v += bias
    tl.store(o + off, v, mask=mask)


@triton.jit
def axpy_optional(
    x: torch.Tensor,
    y: torch.Tensor | None,
    o: tl.tensor | None,  # pyright: ignore[reportInvalidTypeForm]  # triton's tensor class
    n,
    a,
    flag,
    bias,
    BLOCK: tl.constexpr,
):
    """`axpy` with `y` and `o` optional tensors, which Triton ignores too."""
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = off < n
    v = tl.load(x + off, mask=mask) * a
    if y is not None:
        if flag:
            v += tl.load(y + off, mask=mask)
    if bias is not None:
        v += bias
    if o is not None:
        tl.store(o + off, v, mask=mask)


@triton.jit
def scale(x, o, n, s, BLOCK: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = off < n
    tl.store(o + off, tl.load(x + off, mask=mask) * s, mask=mask)


@gluon.jit
def gluon_copy(x, o, n, BLOCK: gl.constexpr, WARP: gl.constexpr):
    off = gl.program_id(0) * BLOCK + gl.arange(
        0,
        BLOCK,
        layout=gl.BlockedLayout([1], [WARP], [4], [0]),  # pyright: ignore[reportArgumentType]  # JIT constexpr is an int
    )
    gl.store(o + off, gl.load(x + off, mask=off < n, other=0), mask=off < n)


def _kernel_from_source(tmp_path, name, signature):
    """Returns an empty `@triton.jit` kernel written to and imported from a file."""
    import importlib.util

    path = tmp_path / f"{name}.py"
    path.write_text(
        f"import triton\n\n@triton.jit\ndef {name}({signature}):\n    pass\n"
    )
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # pyright: ignore[reportAttributeAccessIssue]  # 3.8 stubs lack it
    return getattr(module, name)


def _launcher_module(launcher):
    return module_of(launcher)


def _launcher_over(module):
    """A built, binding-free launcher of `module`, for its `spec_key`."""
    from intj.launcher import _Interner, _launcher_compile, _stub

    stub = _stub()
    launch = stub.new_launcher(
        stub.Launcher,
        None,
        None,
        lambda h: module._init(h, _launcher_compile(module), _Interner()),
    )
    module_of(launch)
    return launch


@pytest.fixture
def scalar_kernel(tmp_path):
    return _kernel_from_source(tmp_path, "scalar_kernel", "x")


@pytest.fixture
def power_kernel(tmp_path):
    return _kernel_from_source(tmp_path, "power_kernel", "N")


@pytest.fixture
def bound_kernel(tmp_path):
    return _kernel_from_source(tmp_path, "bound_kernel", "x, p, n")


@pytest.fixture
def device_kernel(tmp_path):
    return _kernel_from_source(tmp_path, "device_kernel", "x")


@pytest.fixture
def pointer_kernel(tmp_path):
    return _kernel_from_source(tmp_path, "pointer_kernel", "x")


def test_bind_device_takes_one_positional_ordinal(device_kernel):
    launch = make_launcher(device_kernel, no_gpu=True)
    with pytest.raises(TypeError):
        launch.__self__.bind_device()  # pyright: ignore[reportCallIssue]
    with pytest.raises(TypeError):
        launch.__self__.bind_device(0, 1)  # pyright: ignore[reportCallIssue]
    with pytest.raises(TypeError):
        launch.__self__.bind_device(device_ordinal=0)  # pyright: ignore[reportCallIssue]
    fixed = launch.__self__.bind_device(0)
    fixed.__self__.check()
    assert tuple(inspect.signature(fixed).parameters) == ("stream", "grid", "x")
    assert type(fixed).__flags__ & (1 << 11)  # Py_TPFLAGS_HAVE_VECTORCALL
    assert fixed(0, 1, 7) is None
    with pytest.raises(TypeError, match="arguments"):
        fixed(0, 0, 1, 7)
    with pytest.raises(TypeError, match="keyword"):
        fixed(0, 1, x=7)  # pyright: ignore[reportCallIssue]


def test_bind_device_validates_exact_nonnegative_int32(device_kernel):
    class IntSubclass(int):
        pass

    factory = make_launcher(
        device_kernel,
        no_gpu=True,
        torch_access_mode=TorchAccessMode.INTERPRETER,
    )
    launch = factory.__self__.bind_device(0)
    # None is unbind_device(): not an error
    for value in (True, False, 0.0, "0", IntSubclass(0), -1, 2**31, 2**65):
        with pytest.raises(TypeError, match="device.*int"):
            factory.__self__.bind_device(value).__self__.check()  # pyright: ignore[reportArgumentType]
    assert launch(0, 1, 7) is None


def test_bind_device_host_ordinals_share_artifact_without_gpu_lookup(
    device_kernel, monkeypatch
):
    from intj.annotation import DeviceBinding
    from intj.launcher import _LOADED

    def forbidden(*args, **kwargs):
        pytest.fail("fixed host binding reached a GPU path")

    for name in (
        "_current_target",
        "_current_device",
        "_canonical_options",
        "_make_compile_callback",
    ):
        monkeypatch.setattr(f"intj.launcher.{name}", forbidden)
    factory = make_launcher(
        device_kernel,
        no_gpu=True,
        extra_annotation={"x": Argument(type=tl.int32, specialize=NEVER)},
        torch_access_mode=TorchAccessMode.INTERPRETER,
    )
    handles = [
        factory.__self__.bind_device(ordinal) for ordinal in (0, 255, 256, 2**31 - 1)
    ]
    module = module_of(handles[0])
    assert all(module_of(handle) is module for handle in handles)
    key = next(key for key, loaded in _LOADED.items() if loaded is module)
    assert key.device_binding is DeviceBinding.FIXED
    assert key.device_offset is None and key.nwords == 0
    calls = []

    def compile_for_ordinal(key, nparams, device, *args):
        calls.append((bytes(key), device))
        return 0, 1, 0, nparams

    override_compile(module, compile_for_ordinal)
    for handle in handles:
        assert handle(0, 1, 7) is None
    assert calls == [(b"", 0), (b"", 255), (b"", 256), (b"", 2**31 - 1)]


def test_bind_device_accepts_every_bound_parameter_name(tmp_path):
    names = ("self", "args", "values", "device", "device_ordinal", "stream", "grid")
    kernel = _kernel_from_source(
        tmp_path, "device_binding_names", ", ".join((*names, "n"))
    )
    factory = make_launcher(
        kernel,
        extra_annotation={
            name: Argument(
                type=tl.pointer_type(tl.float32),
                specialize=NEVER,
                bind_value=BindValue.POINTER,
            )
            for name in names
        },
        no_gpu=True,
        torch_access_mode=TorchAccessMode.INTERPRETER,
    )
    values = dict.fromkeys(names, 0)
    with pytest.raises(TypeError, match="missing"):
        factory.__self__.bind_device(0).__self__.check()
    with pytest.raises(TypeError, match="unknown.*other"):
        factory.__self__.bind_device(0, **values, other=1).__self__.check()
    launch = factory.__self__.bind_device(0, **values)
    launch.__self__.check()  # the signature is set by check()
    assert tuple(inspect.signature(launch).parameters) == ("stream", "grid", "n")
    assert launch(0, 1, 7) is None


def test_bind_device_signature_avoids_public_control_names(tmp_path):
    names = ("self", "device", "stream", "stream_", "grid", "grid_")
    kernel = _kernel_from_source(tmp_path, "fixed_control_names", ", ".join(names))
    launch = make_launcher(
        kernel,
        no_gpu=True,
        torch_access_mode=TorchAccessMode.INTERPRETER,
    ).__self__.bind_device(0)
    launch.__self__.check()  # the signature is set by check()
    signature = inspect.signature(launch)
    assert tuple(signature.parameters) == ("stream__", "grid__", *names)
    assert all(
        p.kind is inspect.Parameter.POSITIONAL_ONLY
        for p in signature.parameters.values()
    )
    assert launch(0, 1, 4, 5, 6, 7, 8, 9) is None


def test_fixed_device_handles_own_independent_caches(device_kernel):
    factory = make_launcher(device_kernel, no_gpu=True)
    first = factory.__self__.bind_device(0)
    second = factory.__self__.bind_device(0)
    calls = []

    def compile_once_per_handle(key, nparams, device, *args):
        calls.append((bytes(key), device))
        return 0, 1, 0, nparams

    override_compile(module_of(first), compile_once_per_handle)
    first(0, 1, 7)
    first(0, 1, 7)
    second(0, 1, 7)
    assert len(calls) == 2, "different fixed-device handles must compile independently"
    assert calls[0] == calls[1]


def test_bind_device_dynamic_constexpr_uses_each_handles_map(device_kernel):
    factory = make_launcher(
        device_kernel,
        no_gpu=True,
        extra_annotation={"x": Constexpr(type=tl.int32)},
        torch_access_mode=TorchAccessMode.INTERPRETER,
    )
    first, second = factory.__self__.bind_device(0), factory.__self__.bind_device(0)
    calls = []

    def compile_each_constant(key, nparams, device, value):
        calls.append((bytes(key), nparams, device, value))
        return 0, 1, 0, nparams

    override_compile(module_of(first), compile_each_constant)
    for handle, value in ((first, 7), (first, 9), (first, 7), (second, 7), (second, 7)):
        handle(0, 1, value)
    assert calls == [(struct.pack("<Q", value), 0, 0, value) for value in (7, 9, 7)], (
        "each fixed-device handle must own its specialization map"
    )
    source = (
        pathlib.Path(str(module_of(first).__file__))
        .with_name("device_kernel.c")
        .read_text()
    )
    assert "intj_cache_init(INTJ_BOUND_CACHE_OF(bound))" in source
    assert "intj_cache_init(&st->cache)" not in source


@pytest.mark.parametrize("bind_device", [False, True])
def test_repeated_key_uses_last_lookup_before_hash(device_kernel, bind_device):
    factory = make_launcher(
        device_kernel,
        no_gpu=True,
        extra_annotation={"x": Constexpr(type=tl.int32)},
        torch_access_mode=TorchAccessMode.INTERPRETER,
    )
    launch = factory.__self__.bind_device(0) if bind_device else factory
    module = module_of(launch)
    compiled = []

    def compile_key(key, nparams, device, value):
        compiled.append(value)
        return 0, 1, 0, nparams

    override_compile(module, compile_key)
    for value in (7, 7, 9, 9, 7):
        if bind_device:
            launch(0, 1, value)
        else:
            launch(0, 0, 1, value)
    assert compiled == [7, 9]

    source = pathlib.Path(str(module.__file__)).with_name("device_kernel.c").read_text()
    call = source[
        source.index("static PyObject *intj_call") : source.index(
            "static PyObject *intj_bound_entry"
        )
    ]
    assert "intj_cache_lookup(INTJ_BOUND_CACHE_OF(bound), key)" in call
    assert "intj_hash(key)" not in call


@pytest.mark.parametrize(
    "mode,cache",
    [
        (TorchAccessMode.INTERPRETER, KernelCache.INTJ),
        (TorchAccessMode.STATIC_COMPILE, KernelCache.INTJ),
        (TorchAccessMode.INTERPRETER, KernelCache.TSL),
        (TorchAccessMode.INTERPRETER, KernelCache.ABSL),
    ],
)
def test_bind_device_no_map_compiles_once_and_has_no_hash_lookup(
    device_kernel, mode, cache
):
    factory = make_launcher(
        device_kernel,
        no_gpu=True,
        extra_annotation={"x": Argument(type=tl.int32, specialize=NEVER)},
        torch_access_mode=mode,
        kernel_cache=cache,
    )
    launch = factory.__self__.bind_device(0)
    calls = []

    def compile_once(key, nparams, device, value):
        calls.append((bytes(key), nparams, device, value))
        return 0, 1, 0, nparams

    override_compile(module_of(launch), compile_once)
    launch(0, 1, 7)
    launch(0, 1, 9)
    assert calls == [(b"", 1, 0, 7)]
    other = factory.__self__.bind_device(0)
    other(0, 1, 9)
    assert calls == [(b"", 1, 0, 7), (b"", 1, 0, 9)], (
        "each no-map handle must own its fixed kernel"
    )
    source_path = pathlib.Path(str(module_of(launch).__file__))
    cpp_path = source_path.with_name("device_kernel.cpp")
    source = (
        cpp_path if cpp_path.exists() else source_path.with_name("device_kernel.c")
    ).read_text()
    call_body = source[
        source.index("static PyObject *intj_call") : source.index(
            "static PyObject *intj_bound_entry"
        )
    ]
    assert "#define INTJ_HAS_KEY 0" in source
    assert "intj_hash(" not in call_body
    assert "intj_cache_get(" not in call_body
    assert "intj_cache_put(" not in call_body
    assert "intj_cache_init(" not in source
    assert "intj_cache_free(" not in source
    assert "bound->fixed_kernel" in call_body


@pytest.mark.parametrize("has_key", [False, True])
def test_bind_device_retry_and_reentrant_miss(device_kernel, has_key):
    annotation = (
        Constexpr(type=tl.int32)
        if has_key
        else Argument(type=tl.int32, specialize=NEVER)
    )
    launch = make_launcher(
        device_kernel,
        no_gpu=True,
        extra_annotation={"x": annotation},
        torch_access_mode=TorchAccessMode.INTERPRETER,
    ).__self__.bind_device(0)
    calls = []

    def compile_reentrant(key, nparams, device, value):
        calls.append(value)
        if len(calls) == 1:
            raise ValueError("compile failed")
        if len(calls) == 2:
            launch(0, 1, value)
        return 0, 1, 0, nparams

    override_compile(module_of(launch), compile_reentrant)
    assert launch(0, 0, 7) is None
    assert calls == []
    with pytest.raises(ValueError, match="compile failed"):
        launch(0, 1, 7)
    assert launch(0, 1, 7) is None
    assert launch(0, 1, 7) is None
    assert calls == [7, 7, 7]
    with pytest.raises(TypeError, match="stream"):
        launch(-1, 1, 7)


def test_bound_handles_each_own_a_cache(pointer_kernel):
    factory = make_launcher(
        pointer_kernel,
        no_gpu=True,
        extra_annotation={
            "x": Argument(
                type=tl.pointer_type(tl.float32),
                specialize=NEVER,
                bind_value=BindValue.POINTER,
            ),
        },
        torch_access_mode=TorchAccessMode.INTERPRETER,
    )
    first, second = factory.__self__.bind(x=4096), factory.__self__.bind(x=8192)
    calls = []

    def compile_for_device(key, nparams, device):
        calls.append((bytes(key), device))
        return 0, 1, 0, nparams

    override_compile(module_of(first), compile_for_device)
    first(0, 0, 1)
    second(0, 0, 1)  # its own cache: a miss on a key `first` has
    second(1, 0, 1)
    second(1, 0, 1)
    key0, key1 = b"\0" * 8, b"\1" + b"\0" * 7
    assert calls == [(key0, 0), (key0, 0), (key1, 1)]


@pytest.mark.parametrize("has_key", [False, True])
@pytest.mark.parametrize(
    "mode", [TorchAccessMode.INTERPRETER, TorchAccessMode.STATIC_COMPILE]
)
def test_bind_device_releases_selected_state_and_owners(bound_kernel, has_key, mode):
    from intj.launcher import _LOADED

    factory = make_launcher(
        bound_kernel,
        no_gpu=True,
        extra_annotation={
            "x": Argument(
                type=tl.pointer_type(tl.float32),
                specialize=NEVER,
                bind_value=BindValue.POINTER,
            ),
            "p": Argument(
                type=tl.pointer_type(tl.float32),
                specialize=NEVER,
                bind_value=BindValue.POINTER,
            ),
            "n": Constexpr(type=tl.int32)
            if has_key
            else Argument(type=tl.int32, specialize=NEVER),
        },
        torch_access_mode=mode,
    )
    owner = _BoundPointer(4096)
    bound = factory.__self__.bind_device(0, x=owner, p=0)
    module = module_of(bound)
    owner_refs = sys.getrefcount(owner)
    for _ in range(3):
        bad = factory.__self__.bind_device(
            0, x=owner, p=2**64
        )  # p fails in C, on the first call
        with pytest.raises(OverflowError, match="p"):
            bad(0, 1, 7)
        del bad
    assert sys.getrefcount(owner) == owner_refs
    assert bound(0, 1, 7) is None
    other = factory.__self__.bind_device(0, x=0, p=0)
    assert other(0, 1, 7) is None
    module_ref, owner_ref = weakref.ref(module), weakref.ref(owner)
    key = next(key for key, loaded in _LOADED.items() if loaded is module)
    _LOADED.pop(key)
    owner.bound = bound
    del owner, bound
    gc.collect()
    assert owner_ref() is None
    assert other(0, 1, 7) is None
    del other, module
    gc.collect()
    assert module_ref() is None


@pytest.mark.parametrize("backend_name", ["hip", "cuda"])
@pytest.mark.parametrize(
    "no_gpu,return_compiled", [(True, False), (False, False), (False, True)]
)
def test_bind_device_cuda_template_backend_branches_compile(
    tmp_path, device_kernel, backend_name, no_gpu, return_compiled
):
    from intj.annotation import DeviceBinding, _resolve_annotations
    from intj.launcher import BACKENDS, _render_params

    backend = BACKENDS[backend_name]
    assert (
        backend.device_symbol
        == {"hip": "hipDeviceGet", "cuda": "cuDeviceGet"}[backend_name]
    )
    params, offset, nwords = _render_params(
        _resolve_annotations(
            device_kernel,
            {
                "x": Argument(type=tl.int32, specialize=NEVER),
            },
        ),
        DeviceBinding.FIXED,
    )
    key = _make_render_key(
        params=params,
        nwords=nwords,
        device_binding=DeviceBinding.FIXED,
        device_offset=offset,
        torch_access_mode="interpreter",
        no_gpu=no_gpu,
        driver_path="/intj-test-no-driver.so",
        launch_symbol=backend.launch_symbol,
        device_symbol=backend.device_symbol,
        error_symbol=backend.error_symbol,
        error_style=backend.error_style,
        return_compiled=return_compiled,
    )
    binary = tmp_path / "m.so"
    build(binary, key)  # Build both real-driver branches, but load neither driver.
    assert binary.is_file()
    source = binary.with_suffix(".c").read_text()
    if no_gpu:
        assert "dlopen(" not in source and "dlsym(" not in source
        assert backend.device_symbol not in source
    else:
        assert f'intj_dlsym(gpu, "{backend.device_symbol}")' in source
        assert "st->device_get(&bound->device_handle," in source


def test_bind_device_resolves_without_changing_current_and_checks_first_compile(
    device_kernel, monkeypatch
):
    current = torch.cuda.current_device()
    ordinal = (current + 1) % torch.cuda.device_count()
    factory = make_launcher(
        device_kernel, torch_access_mode=TorchAccessMode.INTERPRETER
    )
    launch = factory.__self__.bind_device(ordinal)
    assert torch.cuda.current_device() == current
    monkeypatch.setattr("intj.launcher._current_device", lambda: ordinal + 1)
    with pytest.raises(
        UnsupportedKernel,
        match="make the target device current before the first launch",
    ):
        launch(0, 1, 7)


def test_bind_device_reports_driver_lookup_failure(device_kernel):
    # HIP leaves a failed lookup in its last-error slot; keep it out of later GPU tests.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import runpy, sys\n"
            "from intj import make_launcher\n"
            "from intj import TorchAccessMode\n"
            "kernel = runpy.run_path(sys.argv[1])['device_kernel']\n"
            "make_launcher(kernel, torch_access_mode=TorchAccessMode.INTERPRETER).__self__.bind_device(2**31 - 1)(0, 1, 7)\n",
            inspect.getfile(device_kernel.fn),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert any(
        f"RuntimeError: intj: {symbol} failed" in result.stderr
        for symbol in ("hipDeviceGet", "cuDeviceGet")
    ), result.stderr


@pytest.mark.parametrize("has_key", [False, True])
def test_bind_device_launches_annotated_ast_source_on_gpu(compiled_kernels, has_key):
    output = torch.empty(1, dtype=torch.int32, device="cuda")
    factory = make_launcher(
        annotated_store,
        extra_annotation={
            "o": Argument(type=tl.pointer_type(tl.int32), specialize=NEVER),
            "x": Constexpr(type=tl.int32)
            if has_key
            else Argument(type=tl.int32, specialize=NEVER),
        },
        torch_access_mode=TorchAccessMode.INTERPRETER,
    )
    bound = factory.__self__.bind_device(torch.cuda.current_device())
    stream = torch.cuda.current_stream().cuda_stream
    for value in (1, 17, -31, 1):
        bound(stream, 1, output, value)
        torch.cuda.synchronize()
        assert output.item() == value
    assert len(compiled_kernels) == (3 if has_key else 1)


def test_bind_requires_every_bound_name_once(bound_kernel):
    factory = make_launcher(
        bound_kernel,
        extra_annotation={
            "x": Argument(
                type=tl.pointer_type(tl.float32),
                specialize=NEVER,
                bind_value=BindValue.TENSOR,
            ),
            "p": Argument(
                type=tl.pointer_type(tl.float32),
                specialize=NEVER,
                bind_value=BindValue.POINTER,
            ),
        },
        no_gpu=True,
    )
    with pytest.raises(TypeError, match="missing"):
        factory(0, 0, 1, 4)  # the first call checks: nothing bound
    with pytest.raises(TypeError, match="missing.*p"):
        factory.__self__.bind(x=torch.ones(4)).__self__.check()
    with pytest.raises(TypeError, match="unknown.*other"):
        factory.__self__.bind(x=torch.ones(4), p=0, other=1).__self__.check()


def test_bound_launcher_is_fastcall_and_hides_bound_parameters(bound_kernel):
    factory = make_launcher(
        bound_kernel,
        extra_annotation={
            "x": Argument(
                type=tl.pointer_type(tl.float32),
                specialize=NEVER,
                bind_value=BindValue.TENSOR,
            ),
            "p": Argument(
                type=tl.pointer_type(tl.float32),
                specialize=NEVER,
                bind_value=BindValue.POINTER,
            ),
        },
        no_gpu=True,
    )
    launch = factory.__self__.bind(x=torch.ones(4), p=0)
    launch.__self__.check()  # the signature is set by check()
    assert tuple(inspect.signature(launch).parameters) == (
        "device",
        "stream",
        "grid",
        "n",
    )
    assert type(launch).__flags__ & (1 << 11)  # Py_TPFLAGS_HAVE_VECTORCALL
    assert launch(0, 0, 1, 4) is None


@pytest.mark.parametrize("bind_device", [False, True], ids=["bind", "bind_device"])
def test_bound_launchers_are_fastcall_builtins(bound_kernel, bind_device):
    factory = make_launcher(
        bound_kernel,
        extra_annotation={
            name: Argument(
                type=tl.pointer_type(tl.float32),
                specialize=NEVER,
                bind_value=BindValue.POINTER,
            )
            for name in ("x", "p")
        },
        no_gpu=True,
        torch_access_mode=TorchAccessMode.INTERPRETER,
    )
    launch = (
        factory.__self__.bind_device(0, x=0, p=0)
        if bind_device
        else factory.__self__.bind(x=0, p=0)
    )
    assert isinstance(launch, types.BuiltinFunctionType)
    flags = ctypes.pythonapi.PyCFunction_GetFlags
    flags.argtypes = [ctypes.py_object]
    flags.restype = ctypes.c_int
    assert flags(launch) == 0x80  # METH_FASTCALL without METH_KEYWORDS.
    call = (lambda: launch(0, 1, 7)) if bind_device else (lambda: launch(0, 0, 1, 7))
    for _ in range(100):
        assert call() is None
    if sys.version_info[:2] == (3, 12):
        assert "CALL_NO_KW_BUILTIN_FAST" in {
            instruction.opname
            for instruction in dis.get_instructions(call, adaptive=True)  # pyright: ignore[reportCallIssue]  # 3.8 stubs lack it
        }


def test_bind_validates_names_before_loading(bound_kernel, monkeypatch):
    factory = make_launcher(
        bound_kernel,
        extra_annotation={
            "x": Argument(specialize=NEVER, bind_value=BindValue.TENSOR),
            "p": Argument(
                type=tl.pointer_type(tl.float32),
                specialize=NEVER,
                bind_value=BindValue.POINTER,
            ),
        },
        no_gpu=True,
    )
    monkeypatch.setattr(
        "intj.launcher._loaded_module",
        lambda *args: pytest.fail("loaded before validating binding names"),
    )
    for values, error in [
        ({}, "missing.*x.*p|missing.*p.*x"),
        ({"x": None}, "missing.*p"),
        ({"x": None, "p": 0, "n": 4}, "unknown.*n"),
    ]:
        with pytest.raises(TypeError, match=error):
            factory.__self__.bind(**values).__self__.check()
    with pytest.raises(TypeError):
        factory.__self__.bind(None, p=0).__self__.check()  # pyright: ignore[reportCallIssue]
    with pytest.raises(TypeError, match="multiple values"):
        factory.__self__.bind(x=None, **{"x": None, "p": 0}).__self__.check()


def test_bind_uses_declaration_order_and_accepts_method_parameter_names(tmp_path):
    kernel = _kernel_from_source(tmp_path, "binding_names", "self, n, values, args")
    factory = make_launcher(
        kernel,
        extra_annotation={
            name: Argument(
                type=tl.pointer_type(tl.float32),
                specialize=NEVER,
                bind_value=BindValue.POINTER,
            )
            for name in ("self", "values", "args")
        },
        no_gpu=True,
        torch_access_mode=TorchAccessMode.INTERPRETER,
    )
    bound = factory.__self__.bind(args=12288, values=8192, self=4096)
    bound.__self__.check()  # the signature is set by check()
    assert tuple(inspect.signature(bound).parameters) == (
        "device",
        "stream",
        "grid",
        "n",
    )
    assert bound(0, 0, 1, 4) is None
    with pytest.raises(TypeError, match="arguments"):
        bound(0, 0, 1, 4096, 4, 8192, 12288)
    with pytest.raises(TypeError, match="keyword"):
        bound(0, 0, 1, n=4)  # pyright: ignore[reportCallIssue]
    for attr in ("__signature__", "__self__"):
        with pytest.raises(AttributeError):
            setattr(bound, attr, None)
    with pytest.raises(TypeError):
        type(bound)()
    assert gc.is_tracked(bound)


@pytest.mark.parametrize(
    "public_names,control_names",
    [
        (("self",), ("device", "stream", "grid")),
        (("device",), ("device_", "stream", "grid")),
        (("stream",), ("device", "stream_", "grid")),
        (("grid",), ("device", "stream", "grid_")),
        (("grid", "device", "stream"), ("device_", "stream_", "grid_")),
        (
            ("device", "device_", "stream", "stream_", "grid", "grid_"),
            ("device__", "stream__", "grid__"),
        ),
    ],
)
def test_bind_signature_control_names_avoid_kernel_names(
    tmp_path, public_names, control_names
):
    kernel = _kernel_from_source(
        tmp_path, "binding_control_names", ", ".join(("x", *public_names))
    )
    factory = make_launcher(
        kernel,
        extra_annotation={
            "x": Argument(
                type=tl.pointer_type(tl.float32),
                specialize=NEVER,
                bind_value=BindValue.POINTER,
            ),
        },
        no_gpu=True,
        torch_access_mode=TorchAccessMode.INTERPRETER,
    )
    bound = factory.__self__.bind(x=0)
    bound.__self__.check()  # the signature is set by check()
    signature = inspect.signature(bound)
    assert tuple(signature.parameters) == (*control_names, *public_names)
    assert all(
        p.kind is inspect.Parameter.POSITIONAL_ONLY
        for p in signature.parameters.values()
    )
    other = factory.__self__.bind(x=4096)
    other.__self__.check()
    assert inspect.signature(other) == signature
    assert bound(0, 0, 1, *range(4, 4 + len(public_names))) is None


@pytest.mark.parametrize(
    "mode",
    [
        TorchAccessMode.STATIC_COMPILE,
        TorchAccessMode.RUNTIME_SHIM,
        TorchAccessMode.INTERPRETER,
    ],
)
def test_bind_tensor_infers_effective_type_and_shares_modules(pointer_kernel, mode):
    from intj.launcher import _LOADED

    factory = make_launcher(
        pointer_kernel,
        extra_annotation={
            "x": Argument(specialize=NEVER, bind_value=BindValue.TENSOR),
        },
        no_gpu=True,
        torch_access_mode=mode,
    )
    first = factory.__self__.bind(x=torch.empty(4))
    second = factory.__self__.bind(x=torch.empty(16))
    wide = factory.__self__.bind(x=torch.empty(4, dtype=torch.float64))
    assert first is not second
    assert first.__self__ is not second.__self__
    assert module_of(first) is module_of(second)
    assert module_of(first) is not module_of(wide)
    for bound, ty in ((first, "*fp32"), (wide, "*fp64")):
        key = next(key for key, module in _LOADED.items() if module is module_of(bound))
        assert key.params[0].annotation.types == (ty,)
        assert key.params[0].annotation.bind_value == "tensor"
        assert key.params[0].annotation.key_fields == ()
        assert hash(key) and key.digest()
        assert bound(0, 0, 1) is None


@pytest.mark.parametrize(
    "annotation",
    [
        Argument(specialize=NEVER, bind_value=BindValue.TENSOR),
        Argument(type=None, specialize=NEVER, bind_value=BindValue.TENSOR),
        Argument(type=(None,), specialize=NEVER, bind_value=BindValue.TENSOR),
    ],
    ids=["inferred", "explicit", "sequence"],
)
@pytest.mark.parametrize("bind_device", [False, True], ids=["module", "no-map"])
@pytest.mark.parametrize("verify", [False, True])
def test_bind_none_materializes_triton_constexpr(
    pointer_kernel, annotation, bind_device, verify
):
    from intj.launcher import _LOADED, _compiler_input, _current_target
    from triton.compiler import make_backend

    factory = make_launcher(
        pointer_kernel,
        extra_annotation={
            "x": annotation,
        },
        no_gpu=True,
        verify_annotation=verify,
        torch_access_mode=TorchAccessMode.INTERPRETER,
    )
    bind = (
        (lambda **values: factory.__self__.bind_device(0, **values))
        if bind_device
        else factory.__self__.bind
    )
    bound = bind(x=None)
    built = module_of(bound)
    key = next(key for key, module in _LOADED.items() if module is built)
    assert key.params[0].annotation.types == (None,)
    source = _compiler_input(
        pointer_kernel, key.params, (), make_backend(_current_target())
    )
    assert source.signature == (("x", "constexpr"),)
    assert source.constants == (((0,), ("none",)),)
    assert source.ast_source(pointer_kernel).constants == {(0,): None}
    assert triton_specialization(pointer_kernel, (None,)) == [("constexpr", None)]
    assert bound(*((0, 1) if bind_device else (0, 0, 1))) is None
    if annotation.type is None or annotation.type == (None,):
        with pytest.raises(TypeError, match="x.*None"):
            bind(x=torch.empty(4)).__self__.check()


class _BoundPointer:
    """A `data_ptr()` object at a fixed address that counts its reads."""

    def __init__(self, address):
        self.address = address
        self.reads = 0
        self.bound: object = None

    def data_ptr(self):
        self.reads += 1
        return self.address


@pytest.mark.parametrize(
    "mode,value,valid",
    [
        (BindValue.TENSOR, None, True),
        (BindValue.TENSOR, 0, False),
        (BindValue.POINTER, None, True),
        (BindValue.POINTER, 0, True),
        (BindValue.POINTER, 4096, True),
    ],
)
def test_binding_null_and_structural_rules(mode, value, valid, pointer_kernel):
    factory = make_launcher(
        pointer_kernel,
        extra_annotation={
            "x": Argument(
                type=tl.pointer_type(tl.float32),
                specialize=NEVER,
                bind_value=mode,
            )
        },
        verify_annotation=True,
        no_gpu=True,
    )
    if valid:
        factory.__self__.bind(x=value)
    else:
        with pytest.raises(TypeError, match="x"):
            factory.__self__.bind(x=value).__self__.check()


@pytest.mark.parametrize(
    "mode",
    [
        TorchAccessMode.STATIC_COMPILE,
        TorchAccessMode.RUNTIME_SHIM,
        TorchAccessMode.INTERPRETER,
    ],
)
@pytest.mark.parametrize("binding", [BindValue.TENSOR, BindValue.POINTER])
@pytest.mark.parametrize("verify", [False, True])
def test_bind_nulls_and_structural_safety(pointer_kernel, mode, binding, verify):
    factory = make_launcher(
        pointer_kernel,
        extra_annotation={
            "x": Argument(
                type=tl.pointer_type(tl.float32), specialize=NEVER, bind_value=binding
            ),
        },
        no_gpu=True,
        torch_access_mode=mode,
        verify_annotation=verify,
    )
    values = [None, torch.empty(4), torch.nn.Parameter(torch.empty(4))]
    bad = [object(), True, 0.0, 2**64]
    if binding is BindValue.POINTER:
        values += [0, 4096, 2**64 - 1, _BoundPointer(4096)]
        bad += [_BoundPointer(True), _BoundPointer(0.0), _BoundPointer(2**64)]
    else:
        bad += [0, 4096, _BoundPointer(4096)]
    for value in values:
        assert factory.__self__.bind(x=value)(0, 0, 1) is None
    for value in bad:
        with pytest.raises((TypeError, OverflowError), match="x"):
            factory.__self__.bind(x=value)(
                0, 0, 1
            )  # C-side checks run on the first call


@pytest.mark.parametrize(
    "mode",
    [
        TorchAccessMode.STATIC_COMPILE,
        TorchAccessMode.RUNTIME_SHIM,
        TorchAccessMode.INTERPRETER,
    ],
)
@pytest.mark.parametrize("binding", [BindValue.TENSOR, BindValue.POINTER])
def test_bind_checked_dtype_and_unchecked_promises(pointer_kernel, mode, binding):
    annotation = {
        "x": Argument(
            type=tl.pointer_type(tl.float32), specialize=NEVER, bind_value=binding
        )
    }
    wrong = torch.empty(4, dtype=torch.float64)
    unchecked = make_launcher(
        pointer_kernel, extra_annotation=annotation, no_gpu=True, torch_access_mode=mode
    ).__self__.bind(x=wrong)
    assert unchecked(0, 0, 1) is None
    checked = make_launcher(
        pointer_kernel,
        extra_annotation=annotation,
        no_gpu=True,
        verify_annotation=True,
        torch_access_mode=mode,
    )
    with pytest.raises(TypeError, match="x.*type"):
        checked.__self__.bind(x=wrong)(0, 0, 1)


@pytest.mark.parametrize(
    "mode",
    [
        TorchAccessMode.STATIC_COMPILE,
        TorchAccessMode.RUNTIME_SHIM,
        TorchAccessMode.INTERPRETER,
    ],
)
@pytest.mark.parametrize("no_gpu", [True, False], ids=["host", "gpu"])
@pytest.mark.parametrize("bind_device", [False, True], ids=["module", "no-map"])
def test_bind_tensor_rechecks_storage_after_set(
    pointer_kernel, mode, no_gpu, bind_device, monkeypatch
):
    monkeypatch.setattr("intj.launcher._LOADED", {})
    factory = make_launcher(
        pointer_kernel,
        extra_annotation={
            "x": Argument(
                type=tl.pointer_type(tl.float32),
                specialize=Assume(Aligned(16), PointerRange(32)),
                bind_value=BindValue.TENSOR,
            ),
        },
        no_gpu=no_gpu,
        verify_annotation=True,
        torch_access_mode=mode,
    )
    device = "cpu" if no_gpu else "cuda"
    tensor = torch.empty(4, device=device)
    bound = (
        factory.__self__.bind_device(0, x=tensor)
        if bind_device
        else factory.__self__.bind(x=tensor)
    )
    calls = []
    if no_gpu:

        def record(key, slots, device, *args):
            calls.append(key)
            return 0, 1, 0, slots

        override_compile(module_of(bound), record)
    controls = (0, 1) if bind_device else (0, 0, 1)
    assert bound(*controls) is None
    tensor.set_(torch.empty(8, device=device).untyped_storage(), 1, (7,), (1,))
    with pytest.raises(ValueError, match="x.*aligned_16"):
        bound(*controls)
    # The view is small; the constraint is on the entire underlying storage.
    tensor.set_(torch.empty(2**29, device=device).untyped_storage(), 0, (4,), (1,))
    with pytest.raises(ValueError, match="x.*pointer_range_32"):
        bound(*controls)
    tensor.set_(torch.empty(4, device=device).untyped_storage(), 0, (4,), (1,))
    assert bound(*controls) is None
    if no_gpu:
        assert len(calls) == 1  # Both failures preceded even the cached kernel.


@pytest.mark.parametrize("verify", [False, True])
@pytest.mark.parametrize(
    "mode",
    [
        TorchAccessMode.STATIC_COMPILE,
        TorchAccessMode.RUNTIME_SHIM,
        TorchAccessMode.INTERPRETER,
    ],
)
@pytest.mark.parametrize("no_gpu", [True, False], ids=["host", "gpu"])
def test_bind_pointer_checks_once_and_warns_at_check(
    pointer_kernel, verify, mode, no_gpu
):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        factory = make_launcher(
            pointer_kernel,
            extra_annotation={
                "x": Argument(
                    type=tl.pointer_type(tl.float32),
                    specialize=Assume(Aligned(16), PointerRange(32)),
                    bind_value=BindValue.POINTER,
                ),
            },
            no_gpu=no_gpu,
            verify_annotation=verify,
            torch_access_mode=mode,
        )
        assert caught == []  # decoration checks nothing
        factory.__self__.bind(x=4096).__self__.check()
        assert len(caught) == int(verify)
        if verify:
            assert caught[0].category is RuntimeWarning
            assert "'x'" in str(caught[0].message)
            assert "cannot be verified from an address" in str(caught[0].message)
            with pytest.raises(ValueError, match="x.*aligned_16"):
                factory.__self__.bind(x=4100)(0, 0, 1)
        else:
            assert factory.__self__.bind(x=4100)(0, 0, 1) is None
        owner = _BoundPointer(4096)
        bound = factory.__self__.bind(x=owner)
        assert bound(0, 0, 1) is None  # the first call reads the pointer, once
        owner.address = 4100
        assert bound(0, 0, 1) is None
        assert bound(0, 0, 1) is None
        assert owner.reads == 1
        assert len(caught) == int(verify)


@pytest.mark.parametrize("verify", [False, True])
@pytest.mark.parametrize("bind_device", [False, True])
def test_binding_warnings_are_per_parameter_at_check(bound_kernel, verify, bind_device):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        factory = make_launcher(
            bound_kernel,
            extra_annotation={
                name: Argument(
                    type=tl.pointer_type(tl.float32),
                    specialize=Assume(Aligned(16), PointerRange(32)),
                    bind_value=BindValue.POINTER,
                )
                for name in ("x", "p")
            },
            no_gpu=True,
            verify_annotation=verify,
        )
        assert caught == []  # decoration checks nothing
        # Even a tensor's known >2 GiB range is deliberately trusted in POINTER mode.
        large = torch.empty(2**29)
        for _ in range(2):
            bound = (
                factory.__self__.bind_device(0, x=large, p=4096)
                if bind_device
                else factory.__self__.bind(x=large, p=4096)
            )
            controls = (0, 1) if bind_device else (0, 0, 1)
            for n in (7, 9):
                assert bound(*controls, n) is None
        # the analysis warns once per make_launcher call, at the first check
        assert [str(w.message) for w in caught] == (
            [
                f"intj: pointer-range assumption for bound pointer {name!r} "
                "cannot be verified from an address and will be trusted"
                for name in ("x", "p")
            ]
            if verify
            else []
        )
        assert all(
            w.category is RuntimeWarning and w.filename == __file__ for w in caught
        )


@pytest.mark.parametrize(
    "mode",
    [
        TorchAccessMode.STATIC_COMPILE,
        TorchAccessMode.RUNTIME_SHIM,
        TorchAccessMode.INTERPRETER,
    ],
)
@pytest.mark.parametrize("binding", [BindValue.TENSOR, BindValue.POINTER])
def test_binding_unchecked_assumptions_keep_structural_safety(
    pointer_kernel, mode, binding
):
    factory = make_launcher(
        pointer_kernel,
        extra_annotation={
            "x": Argument(
                type=tl.pointer_type(tl.float32),
                specialize=Assume(Aligned(16), PointerRange(32)),
                bind_value=binding,
            ),
        },
        no_gpu=True,
        torch_access_mode=mode,
    )
    large = torch.empty(2**28 + 1, dtype=torch.float64)
    wrong = large[1:2]  # Wrong dtype, unaligned pointer, and >2 GiB storage.
    assert factory.__self__.bind(x=wrong)(0, 0, 1) is None

    class TensorSubclass(torch.Tensor):
        pass

    bad = (object(), True, 0.0, 2**64, wrong.as_subclass(TensorSubclass))
    if binding is BindValue.POINTER:
        # POINTER deliberately accepts data_ptr() owners, including subclasses.
        bad = bad[:-1] + (_BoundPointer(True), _BoundPointer(0.0), _BoundPointer(2**64))
    for value in bad:
        with pytest.raises((TypeError, OverflowError), match="x"):
            factory.__self__.bind(x=value)(0, 0, 1)


@pytest.mark.parametrize(
    "mode", [TorchAccessMode.INTERPRETER, TorchAccessMode.STATIC_COMPILE]
)
@pytest.mark.parametrize("cache_owner", ["module", "handle-map", "no-map"])
def test_binding_checked_source_checks_only_where_values_change(
    bound_kernel, mode, cache_owner
):
    for verify in (True, False):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            factory = make_launcher(
                bound_kernel,
                extra_annotation={
                    "x": Argument(
                        type=tl.pointer_type(tl.float32),
                        specialize=Assume(Aligned(16), PointerRange(32)),
                        bind_value=BindValue.TENSOR,
                    ),
                    "p": Argument(
                        type=tl.pointer_type(tl.float32),
                        specialize=Assume(Aligned(16), PointerRange(32)),
                        bind_value=BindValue.POINTER,
                    ),
                    "n": Argument(
                        type=tl.int32,
                        specialize=NEVER
                        if cache_owner == "no-map"
                        else Assume(Aligned(16)),
                    ),
                },
                no_gpu=True,
                verify_annotation=verify,
                torch_access_mode=mode,
            )
            bound = (
                factory.__self__.bind(x=None, p=0)
                if cache_owner == "module"
                else factory.__self__.bind_device(0, x=None, p=0)
            )
            bound.__self__.check()  # where the ignored warning fires
        suffix = ".cpp" if mode is TorchAccessMode.STATIC_COMPILE else ".c"
        source = (
            pathlib.Path(str(module_of(bound).__file__))
            .with_name(f"bound_kernel{suffix}")
            .read_text()
        )
        pack = source[
            source.index("static INTJ_ALWAYS_INLINE int intj_pack") : source.index(
                "/* Parse one grid"
            )
        ]
        bind = source[
            source.index("static PyObject *intj_init") : source.index(
                "/* Debug/test entry"
            )
        ]
        call = source[
            source.index("static PyObject *intj_call") : source.index(
                "static PyObject *intj_bound_entry"
            )
        ]
        lookup = (
            "kernel = (intj_final *)bound->fixed_kernel"
            if cache_owner == "no-map"
            else "intj_cache_lookup("
        )
        assert call.index("intj_pack(") < call.index(lookup), (
            "argument validation must precede cache lookup"
        )
        for i, body in ((0, pack), (1, bind), (2, pack)):
            branch, assume = f"INTJ_UNLIKELY(!valid_{i})", f"INTJ_ASSUME(valid_{i})"
            assert source.count(branch) == source.count(assume) == int(verify)
            if verify:
                assert body.index(branch) < body.index(assume)
        assert "valid_1" not in pack and "valid_0" not in bind and "valid_2" not in bind
        assert "pointer_range_32" not in bind and "storage_nbytes" not in bind
        assert source.count("intj_decode_bound_tensor(") == 1
        assert source.count("intj_decode_pointer(") == 1
        assert "intj_decode_pointer(" not in pack
        if not verify:
            for expression in (
                "valid_",
                "violates",
                "storage_nbytes <=",
                "INTJ_ASSUME(",
            ):
                assert expression not in source


@pytest.mark.parametrize("bind_device", [False, True])
@pytest.mark.parametrize(
    "annotation,bad,error,constraint",
    [
        (Argument(type=tl.int8, specialize=NEVER), True, TypeError, "type"),
        (Argument(type=tl.int8, specialize=NEVER), 128, OverflowError, "range"),
        (
            Argument(type=tl.int32, specialize=Assume(EqualTo(1))),
            2,
            ValueError,
            "equal_to_one",
        ),
        (
            Argument(type=tl.int32, specialize=Assume(Aligned(16))),
            17,
            ValueError,
            "aligned_16",
        ),
        (
            Constexpr(type=tl.int8, power_of_two_or_zero=True),
            3,
            ValueError,
            "power_of_two_or_zero",
        ),
    ],
)
def test_checked_failures_precede_cold_and_hot_lookup(
    scalar_kernel,
    monkeypatch,
    bind_device,
    annotation,
    bad,
    error,
    constraint,
):
    monkeypatch.setattr("intj.launcher._LOADED", {})
    factory = make_launcher(
        scalar_kernel,
        extra_annotation={"x": annotation},
        no_gpu=True,
        verify_annotation=True,
    )
    bound = factory.__self__.bind_device(0) if bind_device else factory
    calls = []

    def record(key, slots, device, value):
        calls.append(key)
        return 0, 1, 0, slots

    override_compile(_launcher_module(bound), record)
    controls = (0, 1) if bind_device else (0, 0, 1)
    good = 16 if constraint == "aligned_16" else 1
    for warm in (False, True):
        if warm:
            bound(*controls, good)
        with pytest.raises(error, match=f"x.*{constraint}"):
            bound(*controls, bad)
        assert len(calls) == int(warm)


@pytest.mark.parametrize("verify", [False, True])
@pytest.mark.parametrize(
    "annotation,constraint",
    [
        (Argument(type=tl.int8, specialize=NEVER, value=128), "fit type"),
        (
            Argument(type=tl.int32, specialize=Assume(EqualTo(1)), value=2),
            "equal_to_one",
        ),
        (
            Argument(type=tl.int32, specialize=Assume(Aligned(16)), value=17),
            "aligned_16",
        ),
        (Constexpr(type=tl.int8, power_of_two_or_zero=True, value=3), "power of two"),
    ],
)
def test_baked_constraints_fail_before_materialization(
    scalar_kernel, monkeypatch, verify, annotation, constraint
):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid baked value reached module materialization")

    monkeypatch.setattr("intj.launcher._materialize_module", forbidden)
    with pytest.raises(ValueError, match=f"x.*{constraint}"):
        make_launcher(
            scalar_kernel,
            extra_annotation={"x": annotation},
            no_gpu=True,
            verify_annotation=verify,
        ).__self__.check()


@pytest.mark.parametrize(
    "mode",
    [
        TorchAccessMode.STATIC_COMPILE,
        TorchAccessMode.RUNTIME_SHIM,
        TorchAccessMode.INTERPRETER,
    ],
)
@pytest.mark.parametrize("binding", [BindValue.TENSOR, BindValue.POINTER])
def test_bind_gpu_storage_and_owner_lifetime(mode, binding):
    factory = make_launcher(
        scale,
        extra_annotation={
            "x": Argument(
                type=tl.pointer_type(tl.float32), specialize=NEVER, bind_value=binding
            ),
        },
        torch_access_mode=mode,
        verify_annotation=True,
    )
    original = torch.arange(16, device="cuda", dtype=torch.float32)
    tensor = original.detach()
    out = torch.empty_like(tensor)
    native_references = getattr(tensor, "_use_count")()
    bound = factory.__self__.bind(x=tensor)
    owns_native_tensor = (
        mode is TorchAccessMode.STATIC_COMPILE and binding is BindValue.TENSOR
    )
    launch(bound, (1,), out, 16, 2.0, 16)  # the first call binds in C
    torch.testing.assert_close(out, original * 2)
    assert getattr(tensor, "_use_count")() == native_references + int(
        owns_native_tensor
    )
    assert (
        any(obj is tensor for obj in gc.get_referents(bound.__self__))
        != owns_native_tensor
    )
    tensor.set_(torch.full_like(original, 7).untyped_storage(), 0, (16,), (1,))
    # original keeps the snapshotted allocation alive after set_ changes tensor.
    if binding is BindValue.TENSOR:
        expected = torch.full_like(original, 14)
    else:
        expected = original * 2
    reference = weakref.ref(tensor)
    del tensor
    gc.collect()
    # Torch 2.14 itself preserves the Python wrapper while a native Tensor owns
    # its TensorImpl. CXX must not add a direct PyObject owner on top of that.
    if not owns_native_tensor:
        assert reference() is not None
    launch(bound, (1,), out, 16, 2.0, 16)
    torch.testing.assert_close(out, expected)
    del bound
    gc.collect()
    assert reference() is None


def test_bind_pointer_snapshots_data_ptr_object_on_gpu():
    tensor = torch.arange(16, device="cuda", dtype=torch.float32)
    out = torch.empty_like(tensor)
    owner = _BoundPointer(tensor.data_ptr())
    bound = make_launcher(
        scale,
        extra_annotation={
            "x": Argument(
                type=tl.pointer_type(tl.float32),
                specialize=NEVER,
                bind_value=BindValue.POINTER,
            ),
        },
        torch_access_mode=TorchAccessMode.INTERPRETER,
    ).__self__.bind(x=owner)
    launch(bound, (1,), out, 16, 2.0, 16)  # the first call snapshots data_ptr()
    owner.address = 0
    out.zero_()
    launch(bound, (1,), out, 16, 2.0, 16)
    torch.testing.assert_close(out, tensor * 2)
    assert owner.reads == 1


@pytest.mark.parametrize(
    "binding,value",
    [
        (BindValue.TENSOR, None),
        (BindValue.POINTER, None),
        (BindValue.POINTER, 0),
    ],
)
def test_bind_null_pointer_reaches_gpu(binding, value):
    out = torch.zeros(1, device="cuda", dtype=torch.int32)
    bound = make_launcher(
        annotated_null,
        extra_annotation={
            "x": Argument(
                type=tl.pointer_type(tl.float32), specialize=NEVER, bind_value=binding
            ),
        },
        verify_annotation=True,
        torch_access_mode=TorchAccessMode.INTERPRETER,
    ).__self__.bind(x=value)
    launch(bound, (1,), out)
    assert out.item() == 1


def test_bind_inferred_none_reaches_gpu_as_constexpr(compiled_kernels):
    tensor = torch.arange(16, device="cuda", dtype=torch.float32)
    out = torch.empty_like(tensor)
    bound = make_launcher(
        axpy,
        extra_annotation={
            "bias": Argument(specialize=NEVER, bind_value=BindValue.TENSOR),
        },
        verify_annotation=True,
        torch_access_mode=TorchAccessMode.INTERPRETER,
    ).__self__.bind(bias=None)
    launch(bound, (1,), tensor, tensor, out, 16, 2.0, False, 16)
    torch.testing.assert_close(out, tensor * 2)
    assert compiled_kernels[-1].src.signature["bias"] == "constexpr"
    assert compiled_kernels[-1].src.constants[(6,)] is None


def test_bind_mixed_and_baked_values_follow_declaration_order_on_gpu():
    tensor = torch.arange(16, device="cuda", dtype=torch.float32)
    out = torch.empty_like(tensor)
    factory = make_launcher(
        scale,
        extra_annotation={
            "x": Argument(
                type=tl.pointer_type(tl.float32),
                specialize=NEVER,
                bind_value=BindValue.TENSOR,
            ),
            "o": Argument(
                type=tl.pointer_type(tl.float32),
                specialize=NEVER,
                bind_value=BindValue.POINTER,
            ),
            "n": Argument(type=tl.int32, specialize=NEVER, value=16),
            "BLOCK": Constexpr(value=16),
        },
        verify_annotation=True,
        torch_access_mode=TorchAccessMode.STATIC_COMPILE,
    )
    bound = factory.__self__.bind(o=out, x=tensor)
    bound.__self__.check()  # the signature is set by check()
    assert tuple(inspect.signature(bound).parameters) == (
        "device",
        "stream",
        "grid",
        "s",
    )
    for scale_value in (2.0, 3.0):
        launch(bound, (1,), scale_value)
        torch.testing.assert_close(out, tensor * scale_value)


def test_bind_no_gpu_never_uses_target_driver_or_compiler(pointer_kernel, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("binding reached a GPU path")

    for name in (
        "_current_target",
        "_current_device",
        "_canonical_options",
        "_make_compile_callback",
    ):
        monkeypatch.setattr(f"intj.launcher.{name}", forbidden)
    factory = make_launcher(
        pointer_kernel,
        extra_annotation={
            "x": Argument(specialize=NEVER, bind_value=BindValue.TENSOR),
        },
        no_gpu=True,
        torch_access_mode=TorchAccessMode.INTERPRETER,
    )
    for value in (None, torch.empty(4)):
        bound = factory.__self__.bind(x=value)
        assert bound(0, 0, 1) is None


@pytest.mark.parametrize(
    "mode,binding",
    [
        (TorchAccessMode.RUNTIME_SHIM, BindValue.TENSOR),
        (TorchAccessMode.INTERPRETER, BindValue.TENSOR),
        (TorchAccessMode.STATIC_COMPILE, BindValue.POINTER),
        (TorchAccessMode.RUNTIME_SHIM, BindValue.POINTER),
        (TorchAccessMode.INTERPRETER, BindValue.POINTER),
    ],
)
def test_bind_retains_extension_and_collects_owner_cycles(
    pointer_kernel, mode, binding
):
    from intj.launcher import _LOADED

    factory = make_launcher(
        pointer_kernel,
        extra_annotation={
            "x": Argument(
                type=tl.pointer_type(tl.float32), specialize=NEVER, bind_value=binding
            ),
        },
        no_gpu=True,
        torch_access_mode=mode,
    )
    owner = torch.empty(4) if binding is BindValue.TENSOR else _BoundPointer(4096)
    bound = factory.__self__.bind(x=owner)
    other = factory.__self__.bind(
        x=torch.empty(4) if binding is BindValue.TENSOR else 8192
    )
    assert bound is not other and bound.__self__ is not other.__self__
    assert module_of(bound) is module_of(other)
    module = module_of(bound)
    module_ref, owner_ref = weakref.ref(module), weakref.ref(owner)
    key = next(key for key, loaded in _LOADED.items() if loaded is module)
    _LOADED.pop(key)
    del factory, module, other
    gc.collect()
    assert module_ref() is module_of(bound)
    assert bound(0, 0, 1) is None
    setattr(owner, "bound", bound)
    del owner, bound
    gc.collect()
    assert owner_ref() is None
    assert module_ref() is None


@pytest.mark.parametrize(
    "mode",
    [
        TorchAccessMode.STATIC_COMPILE,
        TorchAccessMode.RUNTIME_SHIM,
        TorchAccessMode.INTERPRETER,
    ],
)
def test_bind_native_failure_releases_partial_owners(bound_kernel, mode):
    factory = make_launcher(
        bound_kernel,
        extra_annotation={
            "x": Argument(
                type=tl.pointer_type(tl.float32),
                specialize=NEVER,
                bind_value=BindValue.TENSOR,
            ),
            "p": Argument(
                type=tl.pointer_type(tl.float32),
                specialize=NEVER,
                bind_value=BindValue.POINTER,
            ),
        },
        no_gpu=True,
        torch_access_mode=mode,
    )
    tensor = torch.empty(4)
    bound = factory.__self__.bind(x=tensor, p=2**64)  # p fails in C, on the first call
    bound.__self__.check()  # the plan holds the tensor from the first check on
    references, native = sys.getrefcount(tensor), getattr(tensor, "_use_count")()
    for _ in range(3):
        with pytest.raises(OverflowError, match="p"):
            bound(0, 0, 1, 4)
    assert sys.getrefcount(tensor) == references
    assert getattr(tensor, "_use_count")() == native
    module = module_of(factory.__self__.bind(x=None, p=0))
    with pytest.raises(TypeError, match="arguments"):
        module._init(tensor)


def test_bind_hot_fastcall_creates_no_python_frame(pointer_kernel):
    bound = make_launcher(
        pointer_kernel,
        extra_annotation={
            "x": Argument(
                type=tl.pointer_type(tl.float32),
                specialize=NEVER,
                bind_value=BindValue.POINTER,
            ),
        },
        no_gpu=True,
        torch_access_mode=TorchAccessMode.INTERPRETER,
    ).__self__.bind(x=4096)
    bound(0, 0, 1)
    calls = []

    def profile(frame, event, arg):
        if event == "call":
            calls.append(frame.f_code.co_name)

    previous = sys.getprofile()
    sys.setprofile(profile)
    try:
        bound(0, 0, 1)
    finally:
        sys.setprofile(previous)
    assert calls == []


@pytest.mark.parametrize(
    "value,encoded",
    [
        (0, 0x00),
        (1, 0x01),
        (2, 0x02),
        (2**63, 0x40),
        (-1, 0x81),
        (-2, 0x82),
        (-(2**63), 0xC0),
    ],
)
def test_power_of_two_or_zero_encoding(power_kernel, value, encoded):
    launch = make_launcher(
        power_kernel,
        extra_annotation={
            "N": Constexpr(type=(tl.int64, tl.uint64), power_of_two_or_zero=True)
        },
        verify_annotation=True,
        no_gpu=True,
    )
    module = module_of(launch)
    key, slots = launch.__self__.spec_key(0, value)
    assert key == bytes((encoded,)) + bytes(7)
    assert slots == 0


@pytest.mark.parametrize(
    "annotation",
    [
        Constexpr(power_of_two_or_zero=True),
        Constexpr(type=(tl.int64, tl.uint64), power_of_two_or_zero=True),
    ],
)
def test_power_of_two_full_signed_domain(power_kernel, annotation):
    launch = make_launcher(
        power_kernel,
        extra_annotation={"N": annotation},
        verify_annotation=True,
        no_gpu=True,
    )
    module = module_of(launch)
    values = [0] + [sign * 2**exponent for sign in (1, -1) for exponent in range(64)]
    assert len({launch.__self__.spec_key(0, value) for value in values}) == 129
    for bad in (3, -3, True, 1.0, None):
        with pytest.raises(ValueError, match="N"):
            launch.__self__.spec_key(0, bad)


@pytest.mark.parametrize("signed", [False, True])
@pytest.mark.parametrize("width", [8, 16, 32, 64])
def test_typed_constexpr_integer_widths(power_kernel, signed, width):
    dtype = getattr(tl, f"{'int' if signed else 'uint'}{width}")
    low = -(1 << (width - 1)) if signed else 0
    high = (1 << (width - int(signed))) - 1
    launch = make_launcher(
        power_kernel,
        extra_annotation={"N": Constexpr(type=dtype)},
        verify_annotation=True,
        no_gpu=True,
    )
    module = module_of(launch)
    for value in (low, high):
        key, slots = launch.__self__.spec_key(0, value)
        payload = value.to_bytes(width // 8, "little", signed=signed)
        assert key == payload + bytes(len(key) - len(payload))
        assert slots == 0
        assert launch(0, 0, 1, value) is None
    for value in (low - 1, high + 1, True, 1.0, None):
        with pytest.raises((TypeError, OverflowError), match="N"):
            launch.__self__.spec_key(0, value)
        with pytest.raises((TypeError, OverflowError), match="N"):
            launch(0, 0, 1, value)


@pytest.mark.parametrize("annotation", [tl.constexpr, Constexpr()])
def test_bare_constexpr_preserves_descriptor_and_binary64(power_kernel, annotation):
    launch = make_launcher(
        power_kernel, extra_annotation={"N": annotation}, no_gpu=True
    )
    module = module_of(launch)
    values = [
        (None, 138, bytes(8)),
        (False, 139, bytes(8)),
        (True, 139, (1).to_bytes(8, "little")),
        (-(2**63), 140, (-(2**63)).to_bytes(8, "little", signed=True)),
        (2**64 - 1, 141, (2**64 - 1).to_bytes(8, "little")),
        (-0.0, 142, struct.pack("<d", -0.0)),
        (0.0, 142, struct.pack("<d", 0.0)),
        (1.0 + 2**-52, 142, struct.pack("<d", 1.0 + 2**-52)),
    ]
    for value, descriptor, payload in values:
        assert launch.__self__.spec_key(0, value) == (
            payload + bytes((descriptor,)) + bytes(7),
            0,
        )
    assert len(
        {launch.__self__.spec_key(0, value)[0] for value, _, _ in values}
    ) == len(values)


@pytest.mark.parametrize(
    "choices",
    [
        (tl.uint64, tl.int16, tl.uint8, tl.int8),
        (tl.int8, tl.uint8, tl.int16, tl.uint64),
    ],
)
def test_constexpr_type_list_selects_smallest_then_name(power_kernel, choices):
    launch = make_launcher(
        power_kernel,
        extra_annotation={"N": Constexpr(type=choices)},
        verify_annotation=True,
        no_gpu=True,
    )
    module = module_of(launch)
    # i8 wins the equal-width tie; u8 wins over i16 where i8 no longer fits.
    for value, code in [(-1, 144), (127, 144), (128, 146), (256, 148), (2**63, 132)]:
        payload = (value % 2**64).to_bytes(8, "little")
        assert launch.__self__.spec_key(0, value) == (
            payload + bytes((code,)) + bytes(7),
            0,
        )
    for bad in (True, 1.0, None, -32769):
        with pytest.raises(TypeError, match="N.*type"):
            launch.__self__.spec_key(0, bad)


def test_constexpr_type_list_distinguishes_python_kinds(power_kernel):
    launch = make_launcher(
        power_kernel,
        extra_annotation={"N": Constexpr(type=(None, tl.int1, tl.int8, tl.float64))},
        verify_annotation=True,
        no_gpu=True,
    )
    module = module_of(launch)
    for value, code, payload in [
        (None, 136, bytes(8)),
        (False, 135, bytes(8)),
        (0, 144, bytes(8)),
        (0.0, 143, bytes(8)),
    ]:
        assert launch.__self__.spec_key(0, value) == (
            payload + bytes((code,)) + bytes(7),
            0,
        )


def test_constexpr_byte_list_uses_placed_descriptor_and_payload(power_kernel):
    launch = make_launcher(
        power_kernel,
        extra_annotation={"N": Constexpr(type=(None, tl.int1, tl.int8, tl.uint8))},
        verify_annotation=True,
        no_gpu=True,
    )
    module = module_of(launch)
    for value, code, payload in [
        (None, 136, 0),
        (False, 135, 0),
        (-1, 144, 255),
        (128, 146, 128),
    ]:
        assert launch.__self__.spec_key(0, value) == (
            bytes((code, payload)) + bytes(6),
            0,
        )


def test_constexpr_exact_none_stays_public(power_kernel):
    launch = make_launcher(
        power_kernel,
        extra_annotation={"N": Constexpr(type=None)},
        verify_annotation=True,
        no_gpu=True,
    )
    module = module_of(launch)
    assert launch.__self__.spec_key(0, None) == (bytes(8), 0)
    assert launch(0, 0, 1, None) is None
    with pytest.raises(TypeError, match="arguments"):
        launch(0, 0, 1)
    for bad in (0, False):
        with pytest.raises(TypeError, match="N.*type"):
            launch(0, 0, 1, bad)


@pytest.mark.parametrize("value", [None, True, 17, 2**64 - 1, -0.0])
def test_baked_constexpr_removes_public_value_and_key(bound_kernel, value):
    launch = make_launcher(
        bound_kernel,
        extra_annotation={"p": Constexpr(value=value)},
        verify_annotation=True,
        no_gpu=True,
    )
    assert launch(0, 0, 1, 7, 17) is None
    assert launch.__self__.spec_key(0, 7, 17) == (
        bytes((128, 128)) + bytes(6),
        2,
    )
    with pytest.raises(TypeError, match="arguments"):
        launch(0, 0, 1, 7, value, 17)


@pytest.mark.parametrize(
    "annotation,values",
    [
        (Constexpr(type=tl.int1), (False, True)),
        (Constexpr(type=tl.float64), (-0.0, 0.0, 1.0, 1.0 + 2**-52)),
    ],
)
def test_typed_constexpr_bool_and_float_bits(power_kernel, annotation, values):
    launch = make_launcher(
        power_kernel,
        extra_annotation={"N": annotation},
        verify_annotation=True,
        no_gpu=True,
    )
    module = module_of(launch)
    keys = []
    for value in values:
        key, slots = launch.__self__.spec_key(0, value)
        payload = (
            bytes((int(value),)) if type(value) is bool else struct.pack("<d", value)
        )
        assert key == payload + bytes(len(key) - len(payload))
        assert slots == 0
        keys.append(key)
    assert len(set(keys)) == len(keys)
    with pytest.raises(TypeError, match="N.*type"):
        launch.__self__.spec_key(0, 1)


def test_constexpr_binary64_preserves_nonfinite_bits(power_kernel):
    launch = make_launcher(
        power_kernel,
        extra_annotation={"N": Constexpr(type=tl.float64)},
        verify_annotation=True,
        no_gpu=True,
    )
    module = module_of(launch)
    for bits in (
        0x7FF0000000000000,
        0xFFF0000000000000,
        0x7FF8000000000000,
        0x7FF8000000000001,
    ):
        payload = bits.to_bytes(8, "little")
        (value,) = struct.unpack("<d", payload)
        assert launch.__self__.spec_key(0, value) == (payload + bytes(8), 0)


@pytest.mark.parametrize("name", [name for name in tl.dtype.FP_TYPES if name != "fp64"])
def test_constexpr_rejects_lower_precision_float_types(power_kernel, name):
    with pytest.raises(ValueError, match="unsupported constexpr type"):
        make_launcher(
            power_kernel,
            extra_annotation={"N": Constexpr(type=tl.dtype(name))},
            no_gpu=True,
        ).__self__.check()


@pytest.mark.parametrize(
    "dtype,bad",
    [
        (tl.int8, (3, -3, 128, -256, True, 1.0, None)),
        (tl.uint8, (3, -1, 256, True, 1.0, None)),
        (tl.int64, (3, 2**63, True)),
        (tl.uint64, (3, -(2**63), True)),
    ],
)
def test_power_of_two_checked_before_cache_lookup(power_kernel, dtype, bad):
    launch = make_launcher(
        power_kernel,
        extra_annotation={"N": Constexpr(type=dtype, power_of_two_or_zero=True)},
        verify_annotation=True,
        no_gpu=True,
    )
    assert launch(0, 0, 1, 1) is None  # 3 and True would alias this cached key.
    for value in bad:
        with pytest.raises(ValueError, match="N"):
            launch(0, 0, 1, value)
        with pytest.raises(ValueError, match="N"):
            launch.__self__.spec_key(0, value)


def test_power_of_two_unchecked_can_alias_but_keeps_structural_checks(power_kernel):
    launch = make_launcher(
        power_kernel,
        extra_annotation={"N": Constexpr(type=tl.int8, power_of_two_or_zero=True)},
        no_gpu=True,
    )
    module = module_of(launch)
    assert launch.__self__.spec_key(0, 1) == launch.__self__.spec_key(0, 3)
    for value in (3, -3, 128, True, 1.0, None):
        assert launch(0, 0, 1, value) is None
    for bad in (object(), torch.empty(1), 2**64, -(2**63) - 1):
        with pytest.raises((TypeError, OverflowError), match="N"):
            launch.__self__.spec_key(0, bad)


@pytest.mark.parametrize("verify", [False, True])
@pytest.mark.parametrize(
    "annotation",
    [
        Constexpr(value=2**64),
        Constexpr(value=-(2**63) - 1),
        Constexpr(type=tl.int8, value=128),
        Constexpr(type=tl.uint8, value=-1),
        Constexpr(type=tl.int8, power_of_two_or_zero=True, value=3),
        Constexpr(power_of_two_or_zero=True, value=True),
        Constexpr(power_of_two_or_zero=True, value=2**64),
    ],
)
def test_baked_constexpr_invalid_at_creation(power_kernel, annotation, verify):
    with pytest.raises((ValueError, OverflowError), match="range|fit|power of two"):
        make_launcher(
            power_kernel,
            extra_annotation={"N": annotation},
            verify_annotation=verify,
            no_gpu=True,
        ).__self__.check()


@pytest.mark.parametrize(
    "mode", [TorchAccessMode.INTERPRETER, TorchAccessMode.STATIC_COMPILE]
)
def test_constexpr_checked_source_has_one_predicted_branch(power_kernel, mode):
    for verify in (True, False):
        module = module_of(
            make_launcher(
                power_kernel,
                extra_annotation={
                    "N": Constexpr(type=(tl.int8, tl.uint64), power_of_two_or_zero=True)
                },
                verify_annotation=verify,
                no_gpu=True,
                torch_access_mode=mode,
            )
        )
        suffix = ".cpp" if mode is TorchAccessMode.STATIC_COMPILE else ".c"
        source = (
            pathlib.Path(str(module.__file__))
            .with_name(f"power_kernel{suffix}")
            .read_text()
        )
        assert source.count("INTJ_UNLIKELY(!valid_0)") == int(verify)
        assert source.count("INTJ_ASSUME(valid_0)") == int(verify)
        assert source.count("intj_decode_constexpr(") == 1


def test_constexpr_callback_receives_original_objects(power_kernel):
    launch = make_launcher(
        power_kernel,
        extra_annotation={"N": Constexpr(type=(tl.int8, tl.int64, tl.float64))},
        no_gpu=True,
    )
    module = module_of(launch)
    seen = []

    def capture(key, slots, device, value):
        seen.append(value)
        return 0, 1, 0, slots

    override_compile(module, capture)
    for value in (int("10000000000"), float("1.0000000000000002"), -0.0):
        assert launch(0, 0, 1, value) is None
        assert seen[-1] is value


@pytest.mark.parametrize(
    "annotation,good,bad",
    [
        (Argument(type=tl.int32, specialize=NEVER), 7, 2**31),
        (Argument(type=tl.int1, specialize=NEVER), True, 1),
        (Argument(type=tl.float64, specialize=NEVER), 1.25, 1),
        (Argument(type=None, specialize=NEVER), None, 0),
        (Argument(type=(tl.int32, tl.float32), specialize=NEVER), 7, None),
    ],
)
def test_checked_ordinary_types(annotation, good, bad, scalar_kernel):
    launch = make_launcher(
        scalar_kernel,
        extra_annotation={"x": annotation},
        verify_annotation=True,
        no_gpu=True,
    )
    assert launch(0, 0, 1, good) is None
    with pytest.raises((TypeError, ValueError, OverflowError), match="x"):
        launch(0, 0, 1, bad)


@pytest.mark.parametrize("signed", [False, True])
@pytest.mark.parametrize("width", [8, 16, 32, 64])
def test_checked_ordinary_integer_widths(scalar_kernel, signed, width):
    dtype = getattr(tl, f"{'int' if signed else 'uint'}{width}")
    low = -(1 << (width - 1)) if signed else 0
    high = (1 << (width - int(signed))) - 1
    launch = make_launcher(
        scalar_kernel,
        extra_annotation={"x": Argument(type=dtype, specialize=NEVER)},
        verify_annotation=True,
        no_gpu=True,
    )
    for value in (low, high):
        assert launch(0, 0, 1, value) is None
        assert launch.__self__.spec_key(0, value)[1] == 1
    for value in (low - 1, high + 1, True, 1.0, None):
        with pytest.raises((TypeError, ValueError, OverflowError), match="x"):
            launch(0, 0, 1, value)


def test_checked_ordinary_reports_type_and_range_separately(scalar_kernel):
    launch = make_launcher(
        scalar_kernel,
        extra_annotation={"x": Argument(type=tl.int8, specialize=NEVER)},
        verify_annotation=True,
        no_gpu=True,
    )
    with pytest.raises(TypeError, match="x.*type"):
        launch(0, 0, 1, True)
    with pytest.raises(OverflowError, match="x.*range"):
        launch(0, 0, 1, 128)


def test_checked_ordinary_allowlist_is_not_coercion(scalar_kernel):
    launch = make_launcher(
        scalar_kernel,
        extra_annotation={
            "x": Argument(type=(tl.int8, tl.int64, tl.float64), specialize=NEVER)
        },
        verify_annotation=True,
        no_gpu=True,
    )
    assert launch(0, 0, 1, 2**31) is None
    for value in (7, 1.25):
        with pytest.raises(TypeError, match="x.*allowlist"):
            launch(0, 0, 1, value)


def test_unchecked_ordinary_keeps_only_structural_checks(scalar_kernel):
    launch = make_launcher(
        scalar_kernel,
        extra_annotation={"x": Argument(type=tl.int8, specialize=Assume(Aligned(16)))},
        no_gpu=True,
        torch_access_mode=TorchAccessMode.INTERPRETER,
    )
    for value in (129, 17, None, True):
        assert launch(0, 0, 1, value) is None
    for value in (object(), 2**64):
        with pytest.raises((TypeError, OverflowError), match="x"):
            launch(0, 0, 1, value)


@pytest.mark.parametrize(
    "mode",
    [
        TorchAccessMode.INTERPRETER,
        TorchAccessMode.RUNTIME_SHIM,
        TorchAccessMode.STATIC_COMPILE,
    ],
)
def test_checked_ordinary_pointer_null_and_dtype(pointer_kernel, mode):
    launch = make_launcher(
        pointer_kernel,
        extra_annotation={
            "x": Argument(type=tl.pointer_type(tl.float32), specialize=NEVER)
        },
        verify_annotation=True,
        no_gpu=True,
        torch_access_mode=mode,
    )
    for value in (None, 0, torch.empty(4)):
        assert launch(0, 0, 1, value) is None
        assert launch.__self__.spec_key(0, value)[1] == 1
    for value in (True, 1, -1, 0.0, torch.empty(4, dtype=torch.int32)):
        with pytest.raises((TypeError, ValueError), match="x"):
            launch(0, 0, 1, value)


def test_checked_ordinary_pointer_allowlist_rejects_missing_torch_dtype(pointer_kernel):
    # Triton has fp8e4b15, but torch has no corresponding dtype. It must not
    # accidentally match the first compact dtype index (uint8).
    launch = make_launcher(
        pointer_kernel,
        extra_annotation={
            "x": Argument(
                type=(tl.pointer_type(tl.float8e4b15), tl.int32), specialize=NEVER
            )
        },
        verify_annotation=True,
        no_gpu=True,
        torch_access_mode=TorchAccessMode.INTERPRETER,
    )
    assert launch(0, 0, 1, 7) is None
    with pytest.raises(TypeError, match="x.*allowlist"):
        launch(0, 0, 1, torch.empty(1, dtype=torch.uint8))


def test_ordinary_unannotated_none_and_never(scalar_kernel):
    launch = make_launcher(scalar_kernel, no_gpu=True)
    auto = module_of(launch)
    never_launch = make_launcher(
        scalar_kernel,
        extra_annotation={"x": Argument(specialize=NEVER)},
        no_gpu=True,
    )
    never = module_of(never_launch)
    assert launch.__self__.spec_key(0, None)[1] == 0
    assert launch.__self__.spec_key(0, 0)[1] == 1
    assert len({launch.__self__.spec_key(0, value) for value in (1, 16, 17)}) == 3
    assert len({never_launch.__self__.spec_key(0, value) for value in (1, 16, 17)}) == 1
    tensor = torch.empty(8)
    assert never_launch.__self__.spec_key(0, tensor) == never_launch.__self__.spec_key(
        0, tensor[1:]
    )


def test_ordinary_never_omits_pointer_range(pointer_kernel):
    from intj.launcher import _LOADED, _loaded_module

    # CPU storage is virtual: exercise both sides of the threshold without
    # touching the bytes or requiring a GPU/backend initialization.
    small = torch.empty(2**31 - 1, dtype=torch.uint8)
    large = torch.empty(2**31, dtype=torch.uint8)
    for specialization in (AUTO, NEVER):
        launch = make_launcher(
            pointer_kernel,
            extra_annotation={"x": Argument(specialize=specialization)},
            no_gpu=True,
            torch_access_mode=TorchAccessMode.INTERPRETER,
        )
        baseline = module_of(launch)
        assert launch.__self__.spec_key(0, small) == launch.__self__.spec_key(0, large)
        key = next(key for key, module in _LOADED.items() if module is baseline)
        ranged_key = dataclasses.replace(key, spec_pointer_range=1)
        ranged = _loaded_module(
            ranged_key,
            pointer_kernel,
            ranged_key.params,
            {},
            None,
            {},
        )
        ranged_launch = _launcher_over(ranged)
        small_key = ranged_launch.__self__.spec_key(0, small)
        large_key = ranged_launch.__self__.spec_key(0, large)
        if specialization is AUTO:
            assert small_key != large_key
            assert large_key == launch.__self__.spec_key(0, large)
        else:
            assert small_key == large_key == launch.__self__.spec_key(0, small)


def test_checked_ordinary_assumed_facts(scalar_kernel, pointer_kernel):
    with pytest.raises(ValueError, match="impossible"):
        make_launcher(
            scalar_kernel,
            extra_annotation={
                "x": Argument(specialize=Assume(EqualTo(1), Aligned(16)))
            },
            no_gpu=True,
        ).__self__.check()
    equal = make_launcher(
        scalar_kernel,
        extra_annotation={"x": Argument(type=tl.int32, specialize=Assume(EqualTo(1)))},
        verify_annotation=True,
        no_gpu=True,
    )
    assert equal.__self__.spec_key(0, 1)[1] == 0
    with pytest.raises(ValueError, match="x.*equal_to_one"):
        equal(0, 0, 1, 2)
    pointer = make_launcher(
        pointer_kernel,
        extra_annotation={
            "x": Argument(
                type=tl.pointer_type(tl.float32),
                specialize=Assume(Aligned(16), PointerRange(32)),
            )
        },
        verify_annotation=True,
        no_gpu=True,
    )
    assert pointer.__self__.spec_key(0, torch.empty(8)) == (bytes(8), 1)
    with pytest.raises(ValueError, match="x.*aligned_16"):
        pointer(0, 0, 1, torch.empty(8)[1:])
    # CPU empty storage is virtual; the test never touches these bytes.
    with pytest.raises(ValueError, match="x.*pointer_range_32"):
        pointer(0, 0, 1, torch.empty(2**29, dtype=torch.float32))


@pytest.mark.parametrize("value,slots", [(7, 3), (None, 2), (True, 3), (-0.0, 3)])
def test_baked_argument_removes_public_value_and_key(bound_kernel, value, slots):
    launch = make_launcher(
        bound_kernel,
        extra_annotation={"p": Argument(value=value, specialize=NEVER)},
        no_gpu=True,
    )
    assert launch(0, 0, 1, 7, 17) is None
    blob, count = launch.__self__.spec_key(0, 7, 17)
    assert count == slots
    assert len(blob) == 8
    assert blob[2:] == bytes(6)
    with pytest.raises(TypeError, match="arguments"):
        launch(0, 0, 1, 7, value, 17)


@pytest.mark.parametrize(
    "value,bits", [(-0.0, "8000000000000000"), (1.25, "3ff4000000000000")]
)
def test_baked_argument_preserves_float_bits(scalar_kernel, value, bits):
    launch = make_launcher(
        scalar_kernel,
        extra_annotation={
            "x": Argument(type=tl.float64, value=value, specialize=NEVER)
        },
        no_gpu=True,
        torch_access_mode=TorchAccessMode.INTERPRETER,
    )
    assert launch(0, 0, 1) is None
    assert launch.__self__.spec_key(0) == (bytes(8), 1)
    path = pathlib.Path(str(module_of(launch).__file__))
    assert f"UINT64_C(0x{bits})" in path.with_name("scalar_kernel.c").read_text()


def test_annotation_module_identity(scalar_kernel):
    annotations = [
        Argument(type=(tl.int32, tl.float32), specialize=NEVER),
        Argument(type=(tl.float32, tl.int32, tl.int32), specialize=NEVER),
        Argument(type=tl.int32, specialize=NEVER),
    ]
    paths = [
        module_of(
            make_launcher(scalar_kernel, extra_annotation={"x": a}, no_gpu=True)
        ).__file__
        for a in annotations
    ]
    assert paths[0] == paths[1]
    assert paths[0] != paths[2]
    checked = make_launcher(
        scalar_kernel,
        extra_annotation={"x": annotations[0]},
        verify_annotation=True,
        no_gpu=True,
    )
    assert module_of(checked).__file__ != paths[0]


@pytest.mark.parametrize(
    "mode", [TorchAccessMode.INTERPRETER, TorchAccessMode.STATIC_COMPILE]
)
def test_checked_source_has_one_predicted_branch_per_argument(scalar_kernel, mode):
    sources = []
    for verify in (True, False):
        launch = make_launcher(
            scalar_kernel,
            extra_annotation={
                "x": Argument(type=tl.int32, specialize=Assume(Aligned(16)))
            },
            verify_annotation=verify,
            no_gpu=True,
            torch_access_mode=mode,
        )
        path = pathlib.Path(str(module_of(launch).__file__))
        suffix = ".cpp" if mode is TorchAccessMode.STATIC_COMPILE else ".c"
        sources.append(path.with_name(f"scalar_kernel{suffix}").read_text())
    checked, unchecked = sources
    assert checked.count("INTJ_UNLIKELY(!valid_0)") == 1
    assert checked.count("INTJ_ASSUME(valid_0)") == 1
    assert "valid_0" not in unchecked
    assert "aligned_16" not in unchecked
    assert checked.count("intj_decode_argument(") == 1
    assert unchecked.count("intj_decode_argument(") == 1


def test_no_gpu_skips_target_driver_compile_and_launch(monkeypatch):
    import intj.launcher as launcher

    def forbidden(*args, **kwargs):
        raise AssertionError("GPU path was reached")

    monkeypatch.setattr(launcher, "_current_target", forbidden)
    monkeypatch.setattr(launcher, "_current_device", forbidden)
    monkeypatch.setattr(launcher, "_canonical_options", forbidden)
    monkeypatch.setattr(launcher, "_make_compile_callback", forbidden)
    host = make_launcher(
        scale, no_gpu=True, torch_access_mode=TorchAccessMode.INTERPRETER
    )
    x = torch.ones(8)
    out = torch.zeros(8)
    assert host(0, 0, (1,), x, out, 8, 2.0, 8) is None
    assert host(0, 0, (1,), x, out, 8, 2.0, 8) is None
    assert torch.count_nonzero(out) == 0


def test_no_gpu_rejects_real_backend_options(monkeypatch):
    import intj.launcher as launcher

    def forbidden(*args, **kwargs):
        raise AssertionError("provisioning was reached")

    monkeypatch.setattr(launcher, "_provision", forbidden)
    with pytest.raises(UnsupportedKernel, match="no_gpu.*options"):
        make_launcher(scale, no_gpu=True, options={"num_warps": 8}).__self__.check()


def test_no_gpu_does_not_read_gpu_compilation_knobs(scalar_kernel, monkeypatch):
    from triton import knobs

    def forbidden(*args, **kwargs):
        pytest.fail("host mode read a GPU compilation knob")

    monkeypatch.setattr(type(knobs.runtime), "debug", property(forbidden))
    monkeypatch.setattr(
        type(knobs.compilation), "instrumentation_mode", property(forbidden)
    )
    monkeypatch.setattr("intj.launcher._canonical_options", forbidden)
    host = make_launcher(
        scalar_kernel, no_gpu=True, torch_access_mode=TorchAccessMode.INTERPRETER
    )
    host(0, 0, (1,), 7)


@triton.jit(do_not_specialize=["n"], do_not_specialize_on_alignment=["x"])
def scale_nospec(x, o, n, s, BLOCK: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = off < n
    tl.store(o + off, tl.load(x + off, mask=mask) * s, mask=mask)


def launch(launcher, grid, *args):
    out = launcher(
        torch.cuda.current_device(),
        torch.cuda.current_stream().cuda_stream,
        grid,
        *args,
    )
    assert out is None
    torch.cuda.synchronize()


def triton_ast_source(jit_func, args):
    # what Triton's own jit_func[grid](*args) compiles
    _, _, _, backend, binder = jit_func.device_caches[torch.cuda.current_device()]
    bound, specialization, options = binder(*args)
    _, signature, constexprs, attrs = jit_func._pack_args(
        backend, {}, bound, specialization, options
    )
    return jit_func.ASTSource(jit_func, signature, constexprs, attrs)


def check_same_ast_source(jit_func, source, args):
    ours, theirs = source.ast_source(jit_func), triton_ast_source(jit_func, args)
    # str(attrs) is what the hash reads: order and empty entries count
    assert str(ours.attrs) == str(theirs.attrs), args
    assert ours.hash() == theirs.hash(), args


def check_matches_triton(jit_func, launcher, grid, args, out_index):
    """Run the kernel through triton and through intj, compare the outputs."""
    reference = args[out_index]
    reference.zero_()
    jit_func[grid](*args)
    torch.cuda.synchronize()
    expected = reference.clone()

    reference.zero_()
    launch(launcher, grid, *args)
    torch.testing.assert_close(reference, expected)


def test_bare_decorator_matches_the_call_form():
    """A bare `@make_launcher` (no parens) behaves like `make_launcher(kernel)`."""

    @make_launcher
    @triton.jit
    def deco_scale(x, o, n, s, BLOCK: tl.constexpr):
        off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = off < n
        tl.store(o + off, tl.load(x + off, mask=mask) * s, mask=mask)

    x = torch.arange(8, device="cuda", dtype=torch.float32)
    o = torch.empty_like(x)
    deco_scale(
        torch.cuda.current_device(),
        torch.cuda.current_stream().cuda_stream,
        (1,),
        x,
        o,
        8,
        2.0,
        8,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(o, x * 2.0)


def test_none_kernel_is_rejected_not_a_decorator_factory():
    """An optional kernel whose import failed must not become the factory form."""
    with pytest.raises(TypeError, match="None instead of a kernel"):
        make_launcher(None).__self__.check()
    with pytest.raises(TypeError, match="None instead of a kernel"):
        make_launcher(None, options={"num_warps": 4}).__self__.check()
    assert callable(make_launcher(options={"num_warps": 4}))


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"no_gpu": 1}, "no_gpu must be a bool"),
        ({"assume_constant_globals": None}, "assume_constant_globals must be a bool"),
        ({"options": [("num_warps", 4)]}, "options must be None or a mapping"),
        ({"extra_annotation": {0: int}}, "extra_annotation must be None or a mapping"),
        ({"kernel_cache": "intj"}, "kernel_cache must be a KernelCache"),
        ({"torch_access_mode": "interpreter"}, "torch_access_mode must be None or"),
        ({"dynamic_options": ("num_warps", 4)}, "names must be str"),
        ({"dynamic_options": 4}, "dynamic_options must be a sequence"),
    ],
)
def test_malformed_keyword_arguments_are_rejected_when_applied(kwargs, match):
    decorate = make_launcher(**kwargs)
    with pytest.raises(TypeError, match=match):
        decorate(scale).__self__.check()
    with pytest.raises(TypeError, match=match):
        make_launcher(scale, **kwargs).__self__.check()


def test_positional_arguments_beyond_the_kernel_are_rejected():
    with pytest.raises(TypeError, match="positional argument"):
        make_launcher(scale, False).__self__.check()  # pyright: ignore[reportCallIssue]


@pytest.fixture(scope="module")
def axpy_launcher():
    return make_launcher(axpy)


@pytest.fixture(scope="module")
def scale_launcher():
    return make_launcher(scale)


@pytest.mark.parametrize("n", [1, 17, 4096])
@pytest.mark.parametrize("flag", [True, False])
@pytest.mark.parametrize("bias", [None, 0.5])
def test_matches_triton(axpy_launcher, n, flag, bias):
    x = torch.randn(4096, device="cuda")
    y = torch.randn(4096, device="cuda")
    o = torch.empty(4096, device="cuda")
    grid = (triton.cdiv(4096, 128),)
    check_matches_triton(
        axpy, axpy_launcher, grid, (x, y, o, n, 1.5, flag, bias, 128), 2
    )


def test_gluon_launch_matches_triton():
    n = 257
    block = 256
    target = triton.runtime.driver.active.get_current_target()
    assert target is not None
    warp = target.warp_size
    x = torch.arange(n, device="cuda", dtype=torch.float32)
    o = torch.empty_like(x)
    check_matches_triton(
        gluon_copy,
        make_launcher(gluon_copy),
        (triton.cdiv(n, block),),
        (x, o, n, block, warp),
        1,
    )


def test_gluon_and_triton_share_no_compiled_module():
    plain = triton.jit(
        gluon_copy.fn  # pyright: ignore[reportFunctionMemberAccess]  # Gluon wraps the function at runtime
    )
    gluon_module = module_of(make_launcher(gluon_copy, no_gpu=True))
    triton_module = module_of(make_launcher(plain, no_gpu=True))
    assert gluon_module.__file__ != triton_module.__file__


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float16, torch.bfloat16, torch.int32]
)
def test_dtypes(scale_launcher, dtype):
    x = torch.ones(1024, device="cuda", dtype=dtype)
    o = torch.empty(1024, device="cuda", dtype=dtype)
    s = 3 if dtype is torch.int32 else 3.0
    check_matches_triton(scale, scale_launcher, (8,), (x, o, 1024, s, 128), 1)


def test_unaligned_pointer(scale_launcher):
    """An unaligned view must not reuse the tt.divisibility=16 kernel."""
    base = torch.randn(1025, device="cuda")
    o = torch.empty(1024, device="cuda")
    check_matches_triton(scale, scale_launcher, (8,), (base[1:], o, 1024, 2.0, 128), 1)
    check_matches_triton(
        scale, scale_launcher, (8,), (base[:1024], o, 1024, 2.0, 128), 1
    )


@pytest.mark.parametrize("block", [64, 128, 256])
def test_constexpr_variants(scale_launcher, block):
    x = torch.randn(1024, device="cuda")
    o = torch.empty(1024, device="cuda")
    check_matches_triton(
        scale, scale_launcher, (triton.cdiv(1024, block),), (x, o, 1024, 2.0, block), 1
    )


@pytest.mark.parametrize("value", [1, 16, 17, 2**31, 2**63, 2**64 - 16])
def test_int_specialization(scale_launcher, value):
    """Every int bucket (fold-to-1, divisible, i32/i64/u64) reaches the right kernel."""
    x = torch.randn(1024, device="cuda")
    o = torch.empty(1024, device="cuda")
    n = min(value, 1024)
    check_matches_triton(scale, scale_launcher, (8,), (x, o, value, 2.0, 128), 1)
    assert n <= 1024


def test_do_not_specialize():
    launcher = make_launcher(scale_nospec)
    base = torch.randn(1025, device="cuda")
    o = torch.empty(1024, device="cuda")
    for x in (base[:1024], base[1:]):
        for n in (1, 1024):
            check_matches_triton(scale_nospec, launcher, (8,), (x, o, n, 2.0, 128), 1)


def test_grid_spellings(scale_launcher):
    x = torch.randn(1024, device="cuda")
    o = torch.empty(1024, device="cuda")
    for grid in (8, (8,), (8, 1), [8, 1, 1]):
        o.zero_()
        launch(scale_launcher, grid, x, o, 1024, 2.0, 128)
        torch.testing.assert_close(o, x * 2.0)


@pytest.mark.parametrize("sequence", [tuple, list])
def test_nonzero_grid_product_overflow_does_not_skip_decoding(scalar_kernel, sequence):
    host = make_launcher(
        scalar_kernel, no_gpu=True, torch_access_mode=TorchAccessMode.INTERPRETER
    )
    # The product is 2**64; every dimension is valid and nonzero.
    with pytest.raises(TypeError, match="argument 'x'"):
        host(0, 0, sequence((1 << 22, 1 << 21, 1 << 21)), object())


@pytest.mark.parametrize("sequence", [tuple, list])
def test_grid_sequence_validation_and_zero_dimensions(scalar_kernel, sequence):
    host = make_launcher(
        scalar_kernel, no_gpu=True, torch_access_mode=TorchAccessMode.INTERPRETER
    )
    for size in (1, 2, 3):
        assert host(0, 0, sequence([2**32 - 1] * size), 7) is None
        for dim in range(size):
            grid = [1] * size
            grid[dim] = 0
            assert host(0, 0, sequence(grid), object()) is None
            for bad in (-1, 2**32, 2**64, True, 1.0, None):
                invalid = [bad if i == dim else 0 for i in range(size)]
                with pytest.raises(ValueError, match="grid dimensions"):
                    host(0, 0, sequence(invalid), 7)
    for grid in ((), (1, 1, 1, 1)):
        with pytest.raises(ValueError, match="between 1 and 3 dimensions"):
            host(0, 0, sequence(grid), 7)
    for grid in (True, 1.0, None, range(1)):
        with pytest.raises(TypeError, match="grid must be"):
            host(0, 0, grid, 7)


@triton.jit
def write_program_ids(o, sy, sz, BLOCK: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    idx = tl.program_id(2) * sz + tl.program_id(1) * sy + off
    tl.store(o + idx, tl.program_id(1) * 10 + tl.program_id(2))


def test_grid_y_and_z_dimensions():
    """gy and gz must reach the kernel in the right order."""
    launcher = make_launcher(write_program_ids)
    o = torch.zeros(2 * 3 * 64, device="cuda", dtype=torch.int32)
    check_matches_triton(write_program_ids, launcher, (1, 3, 2), (o, 64, 3 * 64, 64), 0)


@triton.jit
def matmul(a, b, c, M, N, K, BLOCK: tl.constexpr):
    pid_m, pid_n = tl.program_id(0), tl.program_id(1)
    offs_m = pid_m * BLOCK + tl.arange(0, BLOCK)
    offs_n = pid_n * BLOCK + tl.arange(0, BLOCK)
    offs_k = tl.arange(0, BLOCK)
    acc = tl.zeros((BLOCK, BLOCK), dtype=tl.float32)
    for k in range(0, K, BLOCK):
        va = tl.load(a + offs_m[:, None] * K + (k + offs_k)[None, :])
        vb = tl.load(b + (k + offs_k)[:, None] * N + offs_n[None, :])
        acc += tl.dot(va, vb)
    tl.store(c + offs_m[:, None] * N + offs_n[None, :], acc)


def test_dynamic_shared_memory():
    """A kernel with non-zero LDS: `shared` must be plumbed into the launch."""
    launcher = make_launcher(matmul)
    m = n = k = 256
    a = torch.randn(m, k, device="cuda", dtype=torch.float16)
    b = torch.randn(k, n, device="cuda", dtype=torch.float16)
    c = torch.zeros(m, n, device="cuda", dtype=torch.float32)
    assert matmul.warmup(a, b, c, m, n, k, 64, grid=None).metadata.shared > 0
    check_matches_triton(
        matmul, launcher, (m // 64, n // 64), (a, b, c, m, n, k, 64), 2
    )


def test_zero_volume_grid_does_not_launch(scale_launcher):
    x = torch.randn(1024, device="cuda")
    o = torch.zeros(1024, device="cuda")
    launch(scale_launcher, (0,), x, o, 1024, 2.0, 128)
    torch.testing.assert_close(o, torch.zeros_like(o))


@pytest.fixture
def compiled_kernels(monkeypatch):
    """Returns the list every `triton.compiler.compile` result is appended to."""
    import triton.compiler

    compile = triton.compiler.compile
    kernels = []

    def capture(*args, **kwargs):
        kernel = compile(*args, **kwargs)
        kernels.append(kernel)
        return kernel

    monkeypatch.setattr(triton.compiler, "compile", capture)
    return kernels


def test_options_are_baked_in(compiled_kernels):
    launcher = make_launcher(
        scale, options={"num_warps": 8}
    )  # freshly loaded: see below
    x = torch.randn(1024, device="cuda")
    o = torch.empty(1024, device="cuda")
    launch(launcher, (8,), x, o, 1024, 2.0, 128)
    torch.testing.assert_close(o, x * 2.0)
    assert {k.metadata.num_warps for k in compiled_kernels} == {8}


@pytest.mark.parametrize(
    "jit_debug,runtime_debug,explicit_debug,expected",
    [
        (True, False, False, False),
        (False, False, True, True),
        (False, True, False, True),
    ],
)
def test_triton_specialization_preserves_explicit_debug(
    scalar_kernel, monkeypatch, jit_debug, runtime_debug, explicit_debug, expected
):
    from triton import knobs

    monkeypatch.setattr(scalar_kernel, "debug", jit_debug)
    monkeypatch.setattr(knobs.runtime, "debug", runtime_debug)
    device = torch.cuda.current_device()
    cache = scalar_kernel.device_caches[device]
    binder = cache[4]
    captured = {}

    def capture(*args, **kwargs):
        result = binder(*args, **kwargs)
        captured.update(result[2])
        return result

    monkeypatch.setitem(scalar_kernel.device_caches, device, (*cache[:4], capture))
    options = {"debug": explicit_debug}
    assert triton_specialization(scalar_kernel, (7,), options) == [("i32", "")]
    assert captured["debug"] is expected
    assert options == {"debug": explicit_debug}


@pytest.mark.parametrize(
    "jit_debug,runtime_debug,instrumentation,options,expected_debug",
    [
        (True, False, "", {}, True),
        (False, True, "", {}, True),
        (False, False, "fpsan", {}, False),
        (True, False, "", {"debug": False}, False),
        (False, True, "", {"debug": False}, True),
        (False, False, "", {"debug": True}, True),
        (True, False, "", {"debug": None}, False),
        (False, False, "", {"instrumentation_mode": "fpsan"}, False),
    ],
)
def test_annotated_effective_compile_options_and_identity(
    scalar_kernel,
    monkeypatch,
    jit_debug,
    runtime_debug,
    instrumentation,
    options,
    expected_debug,
):
    import intj.launcher as launcher
    import triton.compiler
    from triton import knobs

    monkeypatch.setattr(scalar_kernel, "debug", False)
    monkeypatch.setattr(knobs.runtime, "debug", False)
    monkeypatch.setattr(knobs.compilation, "instrumentation_mode", "")
    baseline = module_of(
        make_launcher(scalar_kernel, torch_access_mode=TorchAccessMode.INTERPRETER)
    )
    baseline_key = next(
        key for key, module in launcher._LOADED.items() if module is baseline
    )
    monkeypatch.setattr(scalar_kernel, "debug", jit_debug)
    monkeypatch.setattr(knobs.runtime, "debug", runtime_debug)
    monkeypatch.setattr(knobs.compilation, "instrumentation_mode", instrumentation)
    captured = {}

    class CompilationCaptured(Exception):
        pass

    class Captured(Exception):
        pass

    def capture_compile(source, *, target, options):
        captured.update(options)
        raise CompilationCaptured

    def capture_module(
        key, fn, params, options, layout, baked_values, knob_values, *rest
    ):
        callback = launcher._make_compile_callback(
            fn, params, options, baked_values, knob_values=knob_values
        )
        with pytest.raises(CompilationCaptured):
            callback({}, b"\x00" * 8, 1, torch.cuda.current_device(), 7)
        assert captured["debug"] is expected_debug
        assert captured["instrumentation_mode"] == instrumentation
        expected_values = {
            "debug": expected_debug,
            "instrumentation_mode": instrumentation,
        }
        fpsan_casts = getattr(knobs.compilation, "fpsan_homomorphic_casts", None)
        if fpsan_casts is not None:
            expected_values["fpsan_homomorphic_casts"] = fpsan_casts
        expected_options = triton.compiler.make_backend(
            launcher._current_target()
        ).parse_options(expected_values)
        assert key.options == getattr(expected_options, "hash")()
        assert (key.options != baseline_key.options) == (
            expected_debug or bool(instrumentation)
        )
        raise Captured

    monkeypatch.setattr(triton.compiler, "compile", capture_compile)
    monkeypatch.setattr(launcher, "_loaded_module", capture_module)
    with pytest.raises(Captured):
        module_of(
            make_launcher(
                scalar_kernel,
                options=options,
                torch_access_mode=TorchAccessMode.INTERPRETER,
            )
        )


def test_newer_triton_fpsan_knob_is_baked_into_compile_options(
    scalar_kernel, monkeypatch
):
    import intj.launcher as launcher
    from triton import knobs

    seen = {}

    class Captured(Exception):
        pass

    def capture(kernel, _resolved, options, *_args, **kwargs):
        seen.update(launcher._knob_options(kernel, options, kwargs["knob_values"]))
        raise Captured

    monkeypatch.setattr(
        knobs.compilation, "fpsan_homomorphic_casts", True, raising=False
    )
    monkeypatch.setattr(launcher, "_materialize_module", capture)
    with pytest.raises(Captured):
        module_of(
            make_launcher(scalar_kernel, torch_access_mode=TorchAccessMode.INTERPRETER)
        )
    assert seen["fpsan_homomorphic_casts"] is True


def test_launchers_of_one_kernel_stay_independent(compiled_kernels):
    """Two launchers of one kernel never share a module.

    They share a C symbol and a module name, so they must not share a module.
    The symbol is the kernel's name, for legible `perf` output; the digest in
    the artifact path is what keeps the two builds apart.
    """
    # num_warps values no other test uses, so both modules are loaded here and
    # their kernel caches are cold; a warm one would not compile anything.
    narrow_launch = make_launcher(scale, options={"num_warps": 2})
    wide_launch = make_launcher(scale, options={"num_warps": 16})
    narrow, wide = module_of(narrow_launch), module_of(wide_launch)
    assert narrow.__name__ == wide.__name__ == "scale"  # one name ...
    assert narrow is not wide  # ... two modules, told apart by their file
    assert narrow.__file__ != wide.__file__

    x = torch.randn(1024, device="cuda")
    o = torch.empty(1024, device="cuda")
    for launcher, num_warps in ((narrow_launch, 2), (wide_launch, 16)):
        compiled_kernels.clear()
        launch(launcher, (8,), x, o, 1024, 2.0, 128)
        torch.testing.assert_close(o, x * 2.0)
        assert {k.metadata.num_warps for k in compiled_kernels} == {num_warps}


def test_one_module_per_key():
    """A repeat `make_launcher` reuses the loaded module, compile function included.

    Installing a second compile function would drop the first, freeing the
    `CompiledKernel`s whose function handles launcher caches still hold.
    """
    first = make_launcher(scale, options={"num_stages": 3})
    second = make_launcher(scale, options={"num_stages": 3})
    compile_function = getattr(module_of(first), "_intj_compile")
    assert module_of(first) is module_of(second)
    assert getattr(module_of(second), "_intj_compile") is compile_function
    assert first is not second  # each launcher owns its cache


def test_cached_build_is_reused_without_rendering(monkeypatch):
    """A dict miss with the .so already on disk must not render or compile again."""
    import intj.launcher as launcher_module
    import intj.render.build as render_build

    module_of(make_launcher(scale, options={"num_stages": 5}))
    launcher_module._LOADED.clear()

    def fail(*args, **kwargs):
        raise AssertionError("rebuilt a module that was already on disk")

    monkeypatch.setattr(render_build, "build", fail)
    module_of(make_launcher(scale, options={"num_stages": 5}))


def test_modules_stay_out_of_the_import_system():
    """intj's dict owns the module: nothing in sys.modules, nothing interpreter-cached.

    Loading through `import_module`, or a single-phase `m_size = -1` template,
    would break both halves and make two launchers share one module's state.
    """
    import gc
    import sys
    import weakref

    import intj.launcher as launcher_module

    before = set(sys.modules)
    first = module_of(make_launcher(scale, options={"num_stages": 6}))
    assert first.__spec__ is not None and first.__spec__.name not in sys.modules
    assert "_intj_lazy" not in sys.modules  # the stub every launcher starts as

    launcher_module._LOADED.clear()
    second = module_of(make_launcher(scale, options={"num_stages": 6}))
    assert second.__file__ == first.__file__  # same .so ...
    assert second is not first  # ... but a module of its own

    reference = weakref.ref(second)
    del second
    launcher_module._LOADED.clear()
    gc.collect()
    assert reference() is None


def test_source_and_binary_are_cached_on_disk():
    launcher = make_launcher(
        scale, options={"num_stages": 4}, torch_access_mode=TorchAccessMode.RUNTIME_SHIM
    )
    so_path = pathlib.Path(str(module_of(launcher).__file__))
    source = so_path.with_name("scale.c")
    assert so_path.exists() and source.exists()
    assert "PyInit_scale" in source.read_text()
    assert so_path.parent.parent.parent.name == "intj"
    # no half-written artifacts left behind by the atomic install
    assert not [p for p in so_path.parent.iterdir() if p.name.startswith(".")]


def test_wrong_argument_count(scale_launcher):
    x = torch.randn(1024, device="cuda")
    with pytest.raises(TypeError, match="takes exactly"):
        launch(scale_launcher, (8,), x, 1024, 2.0, 128)


def test_unsupported_argument_type(scale_launcher):
    x = torch.randn(1024, device="cuda")
    o = torch.empty(1024, device="cuda")
    with pytest.raises(TypeError, match="unsupported argument 'n'"):
        launch(scale_launcher, (8,), x, o, "1024", 2.0, 128)


def test_int_boundaries_match_triton(axpy_launcher):
    """One decode now serves both widths, so every bucket boundary is a risk.

    Covers what the fold-to-1, i32/i64/u64 and too-large branches disagree about:
    two values share an intj key only where triton specializes them alike, and
    intj refuses exactly the values triton refuses.
    """
    module = module_of(axpy_launcher)
    base = torch.randn(64, device="cuda")
    values = [
        0,
        1,
        16,
        17,
        -16,
        -17,
        2**31 - 1,
        2**31,
        2**63 - 1,
        2**63,
        2**64 - 16,
        2**64 - 1,
        -(2**63),
        -(2**63) - 1,
        2**64,
        2**100,
    ]

    seen = {}
    for value in values:
        args = (base, base, base, value, 1.5, True, None, 128)
        try:
            spec = repr(triton_specialization(axpy, args)[3])
        except OverflowError:
            with pytest.raises(OverflowError):
                axpy_launcher.__self__.spec_key(0, *args)
            continue

        key, _ = axpy_launcher.__self__.spec_key(0, *args)
        previous = seen.setdefault(key, (spec, value))
        assert previous[0] == spec, (
            f"{previous[1]} and {value} share a key, {previous[0]} != {spec}"
        )


@pytest.mark.parametrize(
    "bind_device", [False, True], ids=["dynamic_device", "bind_device"]
)
@pytest.mark.parametrize(
    "kernel",
    [
        "plain",
        "tensor_annotated",
        "tensor_generic",
        "optional_annotated",
        "optional_generic",
    ],
)
def test_spec_key_is_never_coarser_than_triton(axpy_launcher, bind_device, kernel):
    """The load-bearing invariant: same intj key => same triton specialization.

    A coarser key is the failure mode that does not crash -- it launches, say, a
    tt.divisibility=16 binary on an unaligned pointer. `tensor_annotated` is
    `axpy` with `torch.Tensor` / `tl.tensor` pointers (tensor-only decode, auto
    path); `tensor_generic` adds an explicit `n` type to force the generic path.
    `optional_*` annotate `y` and `o` `... | None` and also pass None there.
    All must compile exactly what the unannotated `axpy` does, as in Triton.
    """
    jit = {"plain": axpy, "tensor": axpy_tensor, "optional": axpy_optional}[
        kernel.split("_")[0]
    ]
    extra = (
        {"n": Argument(type=(tl.int32, tl.int64, tl.uint64))}
        if kernel.endswith("_generic")
        else None
    )
    if bind_device:
        launch = make_launcher(jit, extra_annotation=extra)
        launch = launch.__self__.bind_device(torch.cuda.current_device())
    else:
        launch = (
            axpy_launcher
            if kernel == "plain"
            else make_launcher(jit, extra_annotation=extra)
        )
    module = module_of(launch)
    from intj.annotation import DeviceBinding, _resolve_annotations
    from intj.launcher import _compiler_input, _current_target, _render_params
    from triton.compiler import make_backend

    params, _, _ = _render_params(
        _resolve_annotations(jit, extra),
        DeviceBinding.FIXED if bind_device else DeviceBinding.NOT_FIXED,
    )
    plain_params, _, _ = _render_params(
        _resolve_annotations(axpy, extra), DeviceBinding.NOT_FIXED
    )
    backend = make_backend(_current_target())
    base = torch.randn(4096, device="cuda")
    tensors = [
        base,  # 16B aligned
        base[1:],  # unaligned
        torch.randn(4096, device="cuda", dtype=torch.float16),
        torch.randn(4096, device="cuda", dtype=torch.bfloat16),
        torch.zeros(4096, device="cuda", dtype=torch.int32),
    ]
    try:
        # > 2 GiB of storage: no tt.pointer_range = 32, i.e. no buffer ops
        tensors.append(torch.empty(2**29 + 8, device="cuda", dtype=torch.float32))
    except torch.OutOfMemoryError:  # pragma: no cover - small GPU
        pass
    ints = [0, 1, 16, 17, 2**31, 2**31 - 1, 2**63, 2**64 - 16]
    cases = []
    for x in tensors:
        cases.append((x, base, base, 4096, 1.5, True, None, 128))
    for n in ints:
        cases.append((base, base, base, n, 1.5, True, None, 128))
    for a in (0.0, 1.5, -2.5):
        for flag in (True, False):
            for bias in (None, 0.0, 1.0):
                for block in (64, 128):
                    cases.append((base, base, base, 4096, a, flag, bias, block))
    if jit is axpy_optional:
        for y, o in ((None, base), (base, None), (None, None), (base[1:], None)):
            cases.append((base, y, o, 4096, 1.5, True, None, 128))

    seen = {}
    for args in cases:
        key, nparams = launch.__self__.spec_key(0, *args)
        source = _compiler_input(jit, params, args, backend)
        assert seen.setdefault(key, source) == source, (
            "intj key collides for annotated ASTSource inputs"
        )
        assert nparams == sum(ty != "constexpr" for _, ty in source.signature)
        # Triton treats tensor annotations as absent: same compiler input
        assert source == _compiler_input(axpy, plain_params, args, backend)
        # and the source hashes as Triton's, so the two share its cache entry
        check_same_ast_source(jit, source, args)


@triton.jit
def python_annotated(
    o,
    n: int,
    w: tl.int64,  # pyright: ignore[reportInvalidTypeForm]  # a triton dtype
    f: float,
    b: bool,
    m,
):
    tl.store(o, (n + w + m).to(tl.float32) + f + b.to(tl.float32))  # pyright: ignore[reportAttributeAccessIssue]


def test_python_annotated_params_compile_like_triton():
    """Type-annotated parameters get triton's `(annotation,) + specialize[1:]`.

    An `int` or `tl.int64` parameter equal to 1 stays a runtime integer (only
    divisibility survives), `float`/`bool` are never specialized, and the
    unannotated `m` still folds 1 -- intj must hand the compiler exactly that.
    """
    from intj.annotation import DeviceBinding, _resolve_annotations
    from intj.launcher import _compiler_input, _current_target, _render_params
    from triton.compiler import make_backend

    launch = make_launcher(python_annotated)
    module = module_of(launch)
    params, _, _ = _render_params(
        _resolve_annotations(python_annotated, None), DeviceBinding.NOT_FIXED
    )
    backend = make_backend(_current_target())
    o = torch.zeros(1, device="cuda")
    seen = {}
    for n, w, f, b, m in itertools.product(
        (1, 16, 17, 0, True, False),
        (1, 32, 3),
        (1.0, 2.5, 3),
        (True, False),
        (1, 16, 7),
    ):
        args = (o, n, w, f, b, m)
        spec = triton_specialization(python_annotated, args)
        expected_attrs = tuple(
            ((i,), tuple(tuple(attr) for attr in backend.parse_attr(desc)))
            for i, (_, desc) in enumerate(spec)
            if isinstance(desc, str) and backend.parse_attr(desc)
        )
        source = _compiler_input(python_annotated, params, args, backend)
        assert [ty for _, ty in source.signature] == [ty for ty, _ in spec], args
        assert source.attrs == expected_attrs, args
        assert dict(source.values) == {
            (i,): args[i] for i, (ty, _) in enumerate(spec) if ty == "constexpr"
        }, args
        # `n` lists no attrs for 1 or a bool, and an empty one for 17
        check_same_ast_source(python_annotated, source, args)
        # ... under one key: an empty entry changes the hash, not the binary
        key, _ = launch.__self__.spec_key(0, *args)
        assert seen.setdefault(key, source) == source, args


@triton.jit
def parity_untyped(o, x, s, f, B: tl.constexpr):
    tl.store(o + s, tl.load(x) * f + B)


@triton.jit
def parity_typed(
    o,
    x,
    s: tl.int32,  # pyright: ignore[reportInvalidTypeForm]  # a triton dtype
    f: float,
    B: tl.constexpr,
):
    tl.store(o + s, tl.load(x) * f + B)


@triton.jit(do_not_specialize=["x", "s"])
def parity_dns(o, x, s, f, B: tl.constexpr):
    tl.store(o + s, tl.load(x) * f + B)


@triton.jit(do_not_specialize_on_alignment=["x", "s"])
def parity_dnsa(o, x, s, f, B: tl.constexpr):
    tl.store(o + s, tl.load(x) * f + B)


@gluon.jit
def parity_gluon(o, x, s, f, B: gl.constexpr):
    gl.store(o + s, gl.load(x) * f + B)


PARITY_KERNELS = {
    "untyped": parity_untyped,
    "typed": parity_typed,
    "dns": parity_dns,
    "dnsa": parity_dnsa,
    "gluon": parity_gluon,
}


def _parity_calls():
    o = torch.zeros(64, device="cuda")
    x = torch.ones(64, device="cuda")
    for s, x_view in itertools.product((0, 1, 3, 16), (x, x[1:])):
        yield (o, x_view, s, 1.5, 4)


def _triton_facts(jit_func, args):
    spec = triton_specialization(jit_func, args)
    facts = {}
    for name in ("x", "s"):
        ty, desc = spec[jit_func.arg_names.index(name)]
        if ty == "constexpr":
            facts[name] = (EqualTo(1),)  # an integer 1, folded
        elif isinstance(desc, str):
            by_char = {"D": Aligned(16), "S": PointerRange(32)}
            facts[name] = tuple(by_char[char] for char in desc)
        else:
            facts[name] = ()
    return facts


def _parity_annotations(facts):
    # AUTO, then the facts Triton gives x and s: assumed, exactly, or NEVER
    # where it gives none
    yield None
    yield {name: Argument(specialize=Assume(*held)) for name, held in facts.items()}
    yield {
        "f": Argument(specialize=Exactly()),
        **{name: Argument(specialize=Exactly(*held)) for name, held in facts.items()},
    }
    unspecialized = [name for name, held in facts.items() if not held]
    if unspecialized:
        yield {
            "f": Argument(specialize=NEVER),
            **{name: Argument(specialize=NEVER) for name in unspecialized},
        }


@pytest.mark.parametrize("kernel", PARITY_KERNELS)
def test_ast_source_hashes_as_triton_launch(kernel):
    """intj's ASTSource hashes as `kernel[grid](...)`'s for every annotation that
    states Triton's facts.

    Triton lists every constexpr and every specialized parameter in its attrs,
    an empty entry included, and the hash reads `str(attrs)`.
    """
    from triton.compiler import make_backend

    from intj.annotation import DeviceBinding, _resolve_annotations
    from intj.launcher import _compiler_input, _current_target, _render_params

    jit = PARITY_KERNELS[kernel]
    backend = make_backend(_current_target())
    for args in _parity_calls():
        for extra in _parity_annotations(_triton_facts(jit, args)):
            params, _, _ = _render_params(
                _resolve_annotations(jit, extra), DeviceBinding.NOT_FIXED
            )
            source = _compiler_input(jit, params, args, backend)
            check_same_ast_source(jit, source, args)


@pytest.mark.parametrize("kernel", PARITY_KERNELS)
def test_compiled_kernel_hash_matches_triton_launch(kernel):
    """A launcher compiles the kernel Triton's launch does, under its cache key.

    So a process using both compiles each kernel once. A Triton-typed `s` of
    1 and of 3 share an intj key (only an empty attrs entry tells them
    apart), so the call that misses decides which of Triton's hashes it gets.
    """
    jit = PARITY_KERNELS[kernel]
    plain = make_launcher(jit, return_compiled=True)
    first = {}
    for args in _parity_calls():
        expected = jit[(1,)](*args)
        compiled = plain(*_gpu_controls(), *args)
        key, _ = plain.__self__.spec_key(torch.cuda.current_device(), *args)
        if key in first:
            assert compiled is first[key], args[2:]
        else:
            first[key] = compiled
            assert compiled.hash == expected.hash, args[2:]
    # one annotated launcher per kernel: the Assume of an aligned x and s = 16
    args = next(a for a in _parity_calls() if a[2] == 16)
    facts = _triton_facts(jit, args)
    annotated = make_launcher(
        jit,
        extra_annotation={n: Argument(specialize=Assume(*f)) for n, f in facts.items()},
        return_compiled=True,
    )
    assert annotated(*_gpu_controls(), *args).hash == jit[(1,)](*args).hash


@triton.jit
def parity_i64(
    o,
    x,
    s: tl.int64,  # pyright: ignore[reportInvalidTypeForm]  # a triton dtype
    f: float,
    B: tl.constexpr,
):
    tl.store(o + s, tl.load(x) * f + B)


@triton.jit
def parity_u8(
    o,
    x,
    s: tl.uint8,  # pyright: ignore[reportInvalidTypeForm]  # a triton dtype
    f: float,
    B: tl.constexpr,
):
    tl.store(o + s, tl.load(x) * f + B)


# Values a baked `s` may take: each fits the kernel's type for `s`.
BAKED_VALUES = {
    "untyped": (0, 1, 3, 16, 32, -16, -1, 2**31, 2**32 + 16, 2**63, 2**64 - 1, True),
    "typed": (0, 1, 3, 16, 32, -16, -1, 2**31 - 16),
    "dns": (0, 1, 3, 16, 2**31),
    "dnsa": (0, 1, 3, 16, 2**31),
    "gluon": (0, 1, 3, 16),
    "i64": (0, 1, 3, 16, 2**40, -(2**63)),
    "u8": (0, 1, 3, 16, 255),
}


def _baked_annotations(jit_func, args):
    # bake s = args[2]: AUTO, then the facts Triton gives it, assumed or exactly
    value, facts = args[2], _triton_facts(jit_func, args)["s"]
    yield Argument(value=value)
    yield Argument(value=value, specialize=Assume(*facts))
    yield Argument(value=value, specialize=Exactly(*facts))


@pytest.mark.parametrize("kernel", BAKED_VALUES)
def test_baked_int_specializes_like_triton(kernel):
    """A baked `s` compiles what Triton's launch passing it compiles.

    Its AUTO facts resolve from the value at `check()`: 1 folds into a
    constexpr unless typed or `do_not_specialize`, and a multiple of 16 is
    divisible unless `do_not_specialize*` -- the hash matches too.
    """
    from triton.compiler import make_backend

    from intj.annotation import DeviceBinding, _resolve_annotations
    from intj.launcher import _compiler_input, _current_target, _render_params

    jit = {**PARITY_KERNELS, "i64": parity_i64, "u8": parity_u8}[kernel]
    backend = make_backend(_current_target())
    o, x = torch.zeros(64, device="cuda"), torch.ones(64, device="cuda")
    for value in BAKED_VALUES[kernel]:
        args = (o, x, value, 1.5, 4)
        for annotation in _baked_annotations(jit, args):
            params, _, _ = _render_params(
                _resolve_annotations(jit, {"s": annotation}), DeviceBinding.NOT_FIXED
            )
            source = _compiler_input(
                jit, params, (o, x, 1.5, 4), backend, baked_values={2: value}
            )
            check_same_ast_source(jit, source, args)


@pytest.mark.parametrize("kernel", ["untyped", "typed", "dns", "dnsa"])
def test_baked_int_launches_like_triton(kernel):
    """A baked AUTO `s` launches Triton's binary for its value, hash and result."""
    jit = PARITY_KERNELS[kernel]
    x = torch.ones(64, device="cuda")
    for value in (0, 1, 3):
        expected = torch.zeros(64, device="cuda")
        actual = torch.zeros_like(expected)
        reference = jit[(1,)](expected, x, value, 1.5, 4)
        launch = make_launcher(
            jit, extra_annotation={"s": Argument(value=value)}, return_compiled=True
        )
        compiled = launch(*_gpu_controls(), actual, x, 1.5, 4)
        torch.cuda.synchronize()
        assert compiled.hash == reference.hash, value
        torch.testing.assert_close(actual, expected)
        assert actual[value] == 5.5


@triton.jit
def store_one(p):
    tl.store(p, 1)


@needs_pointer_range
def test_exactly_compiles_only_its_facts():
    """`Exactly` states what a hand-written ASTSource attrs entry does.

    `Assume(PointerRange(32))` still measures divisibility and finds it on an
    aligned tensor; `Exactly(PointerRange(32))` compiles pointer_range alone,
    for any tensor, under the hand-written source's hash.
    """
    from triton.compiler import ASTSource

    x = torch.zeros(4, dtype=torch.int32, device="cuda")
    extra = {"p": Argument(type=tl.pointer_type(tl.int32))}

    def compiled(specialize, tensor):
        annotation = {"p": dataclasses.replace(extra["p"], specialize=specialize)}
        launch = make_launcher(
            store_one, extra_annotation=annotation, return_compiled=True
        )
        return launch(*_gpu_controls(), tensor)

    exact = ASTSource(store_one, {"p": "*i32"}, {}, {(0,): [["tt.pointer_range", 32]]})
    for tensor in (x, x[1:]):
        kernel = compiled(Exactly(PointerRange(32)), tensor)
        assert kernel.src.attrs == exact.attrs
        assert kernel.src.hash() == exact.hash()
    assert compiled(Assume(PointerRange(32)), x).src.attrs == {
        (0,): [["tt.divisibility", 16], ["tt.pointer_range", 32]]
    }


@triton.jit
def typed_f32(o, x: float):
    tl.store(o, x.to(tl.float64))  # pyright: ignore[reportAttributeAccessIssue]


@triton.jit
def typed_f64(o, x: tl.float64):  # pyright: ignore[reportInvalidTypeForm]
    tl.store(o, x)


@triton.jit
def typed_i32(o, x: int):
    tl.store(o, x.to(tl.float64))  # pyright: ignore[reportAttributeAccessIssue]


@triton.jit
def typed_u64(o, x: tl.uint64):  # pyright: ignore[reportInvalidTypeForm]
    tl.store(o, x.to(tl.float64))


@triton.jit
def typed_i1(o, x: tl.int1):  # pyright: ignore[reportInvalidTypeForm]
    tl.store(o, tl.where(x, 1.0, 2.0).to(tl.float64))


@triton.jit
def untyped_store(o, x):
    tl.store(o, x.to(tl.float64))


@pytest.mark.parametrize(
    "kernel",
    [typed_f32, typed_f64, typed_i32, typed_u64, typed_i1, untyped_store],
    ids=lambda k: k.__name__,
)
def test_annotated_scalar_values_convert_like_triton(kernel):
    """A Triton-annotated scalar takes Triton's launcher conversion.

    `3` into a `float` is 3.0, `2.5` into an `int` is a TypeError, out-of-range
    ints raise OverflowError -- never a reinterpretation of the Python value's
    bits.
    """
    launcher = make_launcher(kernel)
    values = [3, True, False, 2.5, 2.0, -1, 2**31, 2**40, 2**63, 2**64 - 1, 2**70]
    values += [1e300, float("nan"), None]

    def result(call, value):
        o = torch.full((1,), -777.0, device="cuda", dtype=torch.float64)
        try:
            call(o, value)
        except (TypeError, OverflowError) as e:
            return type(e).__name__
        return o.item()

    for value in values:
        if kernel is untyped_store and value is None:
            continue  # `None.to` does not compile, in either
        expected = result(lambda o, v: kernel[(1,)](o, v), value)
        actual = result(lambda o, v: launch(launcher, (1,), o, v), value)
        assert repr(actual) == repr(expected), (kernel.__name__, value)


@triton.jit
def act_store(o, x, ACT: tl.constexpr):
    if ACT == "neg":
        tl.store(o, -x)
    else:
        tl.store(o, x)


@triton.jit
def cast_store(o, x, DT: tl.constexpr):
    tl.store(o, x.to(DT).to(tl.float32))


@triton.jit
def _twice(v):
    return v * 2


@triton.jit
def _thrice(v):
    return v * 3


@triton.jit
def apply_store(o, x, FN: tl.constexpr):
    tl.store(o, FN(x))


def _gpu_controls():
    return torch.cuda.current_device(), torch.cuda.current_stream().cuda_stream, 1


def test_object_constexpr_key_is_never_coarser_than_triton():
    """Interned object constexprs get equal ids only for equal canonical values.

    This is the invariant for interned values, whichever object carries them.
    """
    from intj.annotation import DeviceBinding, _resolve_annotations
    from intj.launcher import _compiler_input, _current_target, _render_params
    from triton.compiler import make_backend

    launch = make_launcher(act_store)
    module = module_of(launch)
    params, _, _ = _render_params(
        _resolve_annotations(act_store, None), DeviceBinding.NOT_FIXED
    )
    backend = make_backend(_current_target())
    o = torch.zeros(1, device="cuda")
    values = [
        "neg",
        "".join(["n", "e", "g"]),
        "pos",
        "",
        0,
        1,
        None,
        True,
        1.0,
        "int32",  # a str spelling a dtype name is still a str
        tl.int32,
        tl.float32,
        tl.dtype("int32"),
        _twice,
        _thrice,
        triton.jit(_twice.fn),
    ]
    seen, keys = {}, set()
    for value in values:
        key, _ = launch.__self__.spec_key(0, o, 2.0, value)
        source = _compiler_input(act_store, params, (o, 2.0, value), backend)
        assert seen.setdefault(key, source) == source, value
        keys.add(key)
    assert len(keys) == len(values) - 3  # "neg", tl.int32 and _twice each twice


def test_str_constexpr_keys_by_value():
    launch = make_launcher(act_store, return_compiled=True)
    o = torch.zeros(1, device="cuda")
    neg = launch(*_gpu_controls(), o, 2.0, "neg")
    torch.cuda.synchronize()
    assert o.item() == -2.0
    other = "".join(["n", "e", "g"])  # equal value, another object
    assert launch(*_gpu_controls(), o, 2.0, other) is neg
    pos = launch(*_gpu_controls(), o, 2.0, "pos")
    torch.cuda.synchronize()
    assert o.item() == 2.0 and pos is not neg


def test_dtype_and_jit_constexprs():
    o = torch.zeros(1, device="cuda")
    cast = make_launcher(cast_store, return_compiled=True)
    as_int = cast(*_gpu_controls(), o, 2.7, tl.int32)
    torch.cuda.synchronize()
    assert o.item() == 2.0
    assert cast(*_gpu_controls(), o, 2.7, tl.float32) is not as_int
    torch.cuda.synchronize()
    assert abs(o.item() - 2.7) < 1e-6
    apply = make_launcher(apply_store, return_compiled=True)
    twice = apply(*_gpu_controls(), o, 3.0, _twice)
    torch.cuda.synchronize()
    assert o.item() == 6.0
    again = triton.jit(_twice.fn)  # another JITFunction, same cache_key
    assert again is not _twice and apply(*_gpu_controls(), o, 3.0, again) is twice
    apply(*_gpu_controls(), o, 3.0, _thrice)
    torch.cuda.synchronize()
    assert o.item() == 9.0


@triton.jit
def act_apply_store(o, x, ACT: tl.constexpr, FN: tl.constexpr):
    v = FN(x)
    if ACT == "neg":
        v = -v
    tl.store(o, v)


def test_grid_py_dynamic_options_with_str_and_jit_constexprs(monkeypatch):
    """grid_py with dynamic options and str/JIT constexprs keeps records apart.

    Final review minor 1: grid_py plus two dynamic_options (a compile option
    and a knob) plus str/JIT constexprs -- correct outputs, a separate record
    per variant, and clean GC after.
    """
    from triton import knobs

    launch = make_launcher(
        act_apply_store,
        grid_py=lambda meta: (1,),
        dynamic_options=("num_warps", "knobs.compilation.disable_line_info"),
        return_compiled=True,
    )
    module = module_of(launch)
    o = torch.zeros(1, device="cuda")
    device, stream = (
        torch.cuda.current_device(),
        torch.cuda.current_stream().cuda_stream,
    )
    monkeypatch.setattr(knobs.compilation, "disable_line_info", False)

    neg = launch(device, stream, 2, False, o, 2.0, "neg", _twice)
    torch.cuda.synchronize()
    assert o.item() == -4.0 and neg.metadata.num_warps == 2

    same_neg = launch(
        device, stream, 2, False, o, 2.0, "".join(["n", "e", "g"]), _twice
    )
    torch.cuda.synchronize()
    assert same_neg is neg and o.item() == -4.0  # a hit: equal value, another object

    pos = launch(device, stream, 4, False, o, 2.0, "pos", _thrice)
    torch.cuda.synchronize()
    assert o.item() == 6.0
    assert pos is not neg and pos.metadata.num_warps == 4

    monkeypatch.setattr(knobs.compilation, "disable_line_info", True)
    neg_lines = launch(device, stream, 2, True, o, 2.0, "neg", _twice)
    torch.cuda.synchronize()
    assert o.item() == -4.0 and neg_lines is not neg  # separate record per knob value

    # `_LOADED` is the module's only owner (AGENTS.md); dropping it there is
    # what makes the rest -- launcher, records, interned constexprs -- collectible.
    from intj.launcher import _LOADED

    key = next(key for key, loaded in _LOADED.items() if loaded is module)
    _LOADED.pop(key)
    module_ref, launch_ref = weakref.ref(module), weakref.ref(launch)
    del launch, module, neg, same_neg, pos, neg_lines
    gc.collect()
    assert module_ref() is None and launch_ref() is None


@triton.jit
def bound_apply_store(o, x, FN: tl.constexpr):
    tl.store(o, FN(tl.load(x)))


def test_bind_device_bound_tensor_dynamic_num_warps_object_constexprs():
    """bind_device with a bound tensor, dynamic num_warps and objects works.

    Final review minor 1: bind_device with a bound tensor, a dynamic num_warps
    and object constexprs -- correct outputs, distinct spec_keys and records
    per num_warps, and clean GC after.
    """
    factory = make_launcher(
        bound_apply_store,
        extra_annotation={
            "x": Argument(
                type=tl.pointer_type(tl.float32),
                specialize=NEVER,
                bind_value=BindValue.TENSOR,
            ),
        },
        dynamic_options=("num_warps",),
        return_compiled=True,
    )
    x = torch.tensor([3.0], device="cuda")
    bound = factory.__self__.bind_device(torch.cuda.current_device(), x=x)
    module = module_of(bound)
    o = torch.zeros(1, device="cuda")
    stream = torch.cuda.current_stream().cuda_stream

    two = bound(stream, 1, 2, o, _twice)
    torch.cuda.synchronize()
    assert o.item() == 6.0 and two.metadata.num_warps == 2

    four = bound(stream, 1, 4, o, _thrice)
    torch.cuda.synchronize()
    assert o.item() == 9.0 and four.metadata.num_warps == 4
    assert two is not four

    key_two, _ = bound.__self__.spec_key(0, 2, o, _twice)
    key_four, _ = bound.__self__.spec_key(0, 4, o, _twice)
    assert key_two != key_four

    # `_LOADED` is the module's only owner (AGENTS.md); dropping it there is
    # what makes the rest -- factory, bound handle, records -- collectible.
    from intj.launcher import _LOADED

    loaded_key = next(key for key, loaded in _LOADED.items() if loaded is module)
    _LOADED.pop(loaded_key)
    factory_ref, bound_ref, module_ref = (
        weakref.ref(factory),
        weakref.ref(bound),
        weakref.ref(module),
    )
    del factory, bound, module, two, four
    gc.collect()
    assert factory_ref() is None and bound_ref() is None and module_ref() is None


def test_sibling_launchers_intern_independently(monkeypatch):
    """Sibling launchers of one module intern object constexprs independently.

    Id 0 is "neg" in one launcher and "pos" in the other; each launches right,
    and the module's compile cache compiles each variant once.
    """
    monkeypatch.setattr(launcher, "_LOADED", {})
    real, compiles = launcher._checked_compile, []

    def counted(*args, **kwargs):
        compiles.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(launcher, "_checked_compile", counted)
    first, second = make_launcher(act_store), make_launcher(act_store)
    o = torch.zeros(1, device="cuda")
    second(*_gpu_controls(), o, 2.0, "neg")  # id 0 = "neg" in second
    first(*_gpu_controls(), o, 2.0, "pos")  # id 0 = "pos" in first
    for launch, act, want in (
        (first, "neg", -2.0),
        (second, "pos", 2.0),
        (first, "pos", 2.0),
        (second, "neg", -2.0),
    ):
        launch(*_gpu_controls(), o, 2.0, act)
        torch.cuda.synchronize()
        assert o.item() == want, (act, want)
    assert (
        module_of(first) is module_of(second) and len(compiles) == 2
    )  # one per variant


def test_unsupported_object_constexpr_names_the_parameter():
    launch = make_launcher(act_store)
    o = torch.zeros(1, device="cuda")

    class Name(str):
        pass

    with pytest.raises(TypeError, match="unsupported argument 'ACT' of type list"):
        launch(*_gpu_controls(), o, 2.0, ["neg"])
    with pytest.raises(TypeError, match="'ACT' of type Name"):
        launch(*_gpu_controls(), o, 2.0, Name("neg"))
    launch(*_gpu_controls(), o, 2.0, "neg")  # still usable
    torch.cuda.synchronize()
    assert o.item() == -2.0


def _count_interner(monkeypatch):
    """Count calls into the Python interner, still answering them."""
    calls = []
    real = launcher._Interner.__call__

    def counted(self, value):
        calls.append(value)
        return real(self, value)

    monkeypatch.setattr(launcher._Interner, "__call__", counted)
    return calls


def test_alternating_objects_skip_the_interner(monkeypatch):
    """A slot switching between known objects is answered by the C object table.

    After warmup there is no interner call, and each value keeps its key.
    """
    calls = _count_interner(monkeypatch)
    act, cast = make_launcher(act_store), make_launcher(cast_store)
    act_module, cast_module = module_of(act), module_of(cast)
    o = torch.zeros(1, device="cuda")
    neg, pos = (act.__self__.spec_key(0, o, 2.0, v)[0] for v in ("neg", "pos"))
    i32, f32 = (cast.__self__.spec_key(0, o, 2.0, v)[0] for v in (tl.int32, tl.float32))
    assert neg != pos and i32 != f32 and len(calls) == 4
    del calls[:]
    for _ in range(5):
        assert act.__self__.spec_key(0, o, 2.0, "neg")[0] == neg
        assert act.__self__.spec_key(0, o, 2.0, "pos")[0] == pos
        assert cast.__self__.spec_key(0, o, 2.0, tl.int32)[0] == i32
        assert cast.__self__.spec_key(0, o, 2.0, tl.float32)[0] == f32
    assert calls == []
    for act_value, want in (("neg", -2.0), ("pos", 2.0), ("neg", -2.0)):
        act(*_gpu_controls(), o, 2.0, act_value)
        torch.cuda.synchronize()
        assert o.item() == want


def test_equal_strs_share_an_id_in_c(monkeypatch):
    """Another str object with equal content hits the object table by content.

    One of equal length and kind but different bytes does not.
    """
    calls = _count_interner(monkeypatch)
    launch = make_launcher(act_store)
    module = module_of(launch)
    o = torch.zeros(1, device="cuda")
    wide = "\u043d\u0435\u0433"  # a 2-byte-kind str
    wider = "\U0001f600x"  # a 4-byte-kind str
    known = ("neg", "pos", wide, wider)
    keys = {v: launch.__self__.spec_key(0, o, 2.0, v)[0] for v in known}
    del calls[:]
    for value in known:
        again = "".join(list(value))  # equal content, another object
        assert again is not value
        assert launch.__self__.spec_key(0, o, 2.0, again)[0] == keys[value]
    assert calls == []
    for other in ("gen", "\u0433\u0435\u043d"):  # same length and kind as a known str
        key = launch.__self__.spec_key(0, o, 2.0, other)[0]
        assert key not in keys.values()
    assert len(calls) == 2


def test_equal_hash_different_content_is_another_str(monkeypatch):
    """Content, not the cached hash, decides a str hit in the object table.

    A str whose cached hash is forged to equal "neg"'s is still another value.
    """
    calls = _count_interner(monkeypatch)
    launch = make_launcher(act_store)
    module = module_of(launch)
    o = torch.zeros(1, device="cuda")
    neg = launch.__self__.spec_key(0, o, 2.0, "neg")[0]
    forged = "".join(["g", "e", "n"])
    hash(forged)  # fill the cache, then overwrite it
    # PyASCIIObject: PyObject_HEAD, Py_ssize_t length, Py_hash_t hash (3.8-3.14,
    # free-threaded too: object.__basicsize__ covers the larger head)
    ctypes.c_ssize_t.from_address(id(forged) + object.__basicsize__ + 8).value = hash(
        "neg"
    )
    assert hash(forged) == hash("neg")
    del calls[:]
    assert launch.__self__.spec_key(0, o, 2.0, forged)[0] != neg
    assert calls == ["gen"]


def test_fresh_dtype_objects_hit_by_name(monkeypatch):
    """A tl.dtype built per call hits the object-table entry for its name.

    There is no interner call, and the table keeps none of those objects.  A
    str spelling the same name is another value.
    """
    calls = _count_interner(monkeypatch)
    launch = make_launcher(act_store)
    module = module_of(launch)
    o = torch.zeros(1, device="cuda")
    fp16 = launch.__self__.spec_key(0, o, 2.0, tl.float16)[0]
    fp32 = launch.__self__.spec_key(0, o, 2.0, tl.float32)[0]
    del calls[:]
    refs = []
    for _ in range(5):
        for name, key in (("fp16", fp16), ("fp32", fp32)):
            fresh = tl.dtype(name)
            assert fresh is not tl.float16 and fresh is not tl.float32
            assert launch.__self__.spec_key(0, o, 2.0, fresh)[0] == key
            refs.append(weakref.ref(fresh))
            del fresh
    assert calls == []
    gc.collect()
    assert sum(r() is not None for r in refs) == 1  # the memo's, nothing else
    assert launch.__self__.spec_key(0, o, 2.0, "fp16")[0] not in (fp16, fp32)
    assert calls == ["fp16"]


def test_object_table_references_die_with_the_launcher(monkeypatch):
    """An object the table holds is released when the launcher is collected.

    The memo no longer holds it.  The object is a fresh tl.dtype, not a JIT
    function: triton keeps every JITFunction in a registry of its own.
    """
    monkeypatch.setattr(launcher, "_LOADED", {})
    launch = make_launcher(cast_store)
    module = module_of(launch)
    o = torch.zeros(1, device="cuda")
    dt = tl.dtype("int32")
    assert dt is not tl.int32
    key = launch.__self__.spec_key(0, o, 2.0, dt)[0]
    launch.__self__.spec_key(0, o, 2.0, tl.float32)  # the memo moves on
    dt_ref, launch_ref = weakref.ref(dt), weakref.ref(launch)
    del dt
    gc.collect()
    assert dt_ref() is not None  # the table holds it
    assert launch.__self__.spec_key(0, o, 2.0, tl.int32)[0] == key
    launcher._LOADED.clear()
    del launch, module
    gc.collect()
    assert launch_ref() is None and dt_ref() is None


@triton.jit
def annotated_store(o, x):
    tl.store(o, x)


@triton.jit
def annotated_null(o, x):
    tl.store(o, x.to(tl.uint64) == 0)


@triton.jit
def annotated_baked(x, BLOCK, o, bias):
    tl.store(o, x + BLOCK + bias)


def test_annotated_cuda_compile_only():
    from intj.annotation import DeviceBinding, _resolve_annotations
    from intj.launcher import _compiler_input, _render_params
    from triton.backends.compiler import GPUTarget
    from triton.compiler import compile as triton_compile, make_backend

    target = GPUTarget("cuda", 80, 32)
    resolved = _resolve_annotations(
        annotated_baked,
        {
            "x": Argument(type=tl.float64, value=2.5, specialize=NEVER),
            "BLOCK": Constexpr(type=tl.int16, value=128),
            "o": Argument(
                type=tl.pointer_type(tl.float64), specialize=Assume(Aligned(16))
            ),
            "bias": Argument(type=tl.int32, specialize=NEVER),
        },
    )
    params, _, _ = _render_params(resolved, DeviceBinding.NOT_FIXED)
    source = _compiler_input(
        annotated_baked,
        params,
        (None, 7),
        make_backend(target),
        baked_values={p.index: p.baked for p in resolved if p.annotation.baked_value},
    )
    compiled = triton_compile(source.ast_source(annotated_baked), target=target)
    assert compiled.asm["ptx"]


@pytest.mark.parametrize(
    "annotation,values,signature,output_dtype",
    [
        (Argument(type=tl.int32, specialize=NEVER), (1, 17, -31), "i32", torch.int64),
        (Argument(type=tl.float64), (1.0 + 2**-40, -0.5), "fp64", torch.float64),
        (Argument(specialize=NEVER), (1, 17), "i32", torch.int64),
        (Argument(), (1,), "constexpr", torch.int64),
        (
            Argument(type=tl.int32, specialize=Assume(EqualTo(1))),
            (1,),
            "constexpr",
            torch.int64,
        ),
        (
            Argument(type=tl.int32, specialize=Assume(Aligned(16))),
            (16, 32),
            "i32",
            torch.int64,
        ),
        (Constexpr(type=tl.int8), (-128, 127), "constexpr", torch.int64),
        (Constexpr(type=tl.uint16), (0, 2**16 - 1), "constexpr", torch.int64),
        (Constexpr(type=tl.int32), (-(2**31), 2**31 - 1), "constexpr", torch.int64),
        (Constexpr(type=tl.int64), (-(2**63), 2**63 - 1), "constexpr", torch.int64),
        (Constexpr(type=tl.float64), (1.0 + 2**-40, -1.5), "constexpr", torch.float64),
        (
            Constexpr(type=tl.int64, power_of_two_or_zero=True),
            (0, 1, -1, 2**32, -(2**32), -(2**63)),
            "constexpr",
            torch.int64,
        ),
    ],
)
def test_annotated_scalar_matches_triton(annotation, values, signature, output_dtype):
    from triton.compiler import ASTSource, compile as triton_compile

    reference = torch.empty(1, device="cuda", dtype=output_dtype)
    actual = torch.empty_like(reference)
    pointer = "*fp64" if output_dtype == torch.float64 else "*i64"
    launcher = make_launcher(
        annotated_store,
        extra_annotation={"x": annotation},
        torch_access_mode=TorchAccessMode.INTERPRETER,
        verify_annotation=True,
    )
    for value in values:
        expected = triton_compile(
            ASTSource(
                annotated_store,
                {"o": pointer, "x": signature},
                {(1,): value} if signature == "constexpr" else {},
            )
        )
        expected[(1, 1, 1)](reference, value)
        launch(launcher, (1,), actual, value)
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)


@pytest.mark.parametrize("null", [None, 0])
def test_annotated_null_pointer_matches_triton(null):
    from triton.compiler import ASTSource, compile as triton_compile

    reference = torch.empty(1, device="cuda", dtype=torch.int32)
    actual = torch.empty_like(reference)
    expected = triton_compile(ASTSource(annotated_null, {"o": "*i32", "x": "*fp32"}))
    expected[(1, 1, 1)](reference, None)
    launcher = make_launcher(
        annotated_null,
        extra_annotation={
            "x": Argument(type=tl.pointer_type(tl.float32), specialize=NEVER),
        },
        torch_access_mode=TorchAccessMode.INTERPRETER,
        verify_annotation=True,
    )
    launch(launcher, (1,), actual, null)
    torch.testing.assert_close(actual, reference)
    assert actual.item() == 1


@pytest.mark.parametrize(
    "specialize", [AUTO, NEVER, Assume(Aligned(16), PointerRange(32))]
)
def test_annotated_tensor_matches_triton(specialize):
    x = torch.randn(128, device="cuda", dtype=torch.float64)
    out = torch.empty_like(x)
    launcher = make_launcher(
        scale,
        extra_annotation={
            "x": Argument(type=tl.pointer_type(tl.float64), specialize=specialize),
            "o": Argument(type=tl.pointer_type(tl.float64), specialize=specialize),
            "s": Argument(type=tl.float64),
        },
        torch_access_mode=TorchAccessMode.INTERPRETER,
        verify_annotation=True,
    )
    check_matches_triton(scale, launcher, (1,), (x, out, 128, 2.0, 128), 1)


def test_annotated_baked_arguments_compile_without_jit_state(monkeypatch):
    from triton.compiler import ASTSource, compile as triton_compile

    expected_out = torch.empty(1, device="cuda", dtype=torch.float64)
    actual = torch.empty_like(expected_out)
    expected = triton_compile(
        ASTSource(
            annotated_baked,
            {"x": "fp64", "BLOCK": "constexpr", "o": "*fp64", "bias": "i32"},
            {(1,): 128},
        )
    )
    expected[(1, 1, 1)](2.5, 128, expected_out, 7)

    def forbidden(*args, **kwargs):
        pytest.fail("intj entered mutable JITFunction compilation")

    monkeypatch.setattr(annotated_baked, "warmup", forbidden)
    monkeypatch.setattr(annotated_baked, "_do_compile", forbidden)
    caches = dict(annotated_baked.device_caches)
    launcher = make_launcher(
        annotated_baked,
        extra_annotation={
            "x": Argument(type=tl.float64, value=2.5, specialize=NEVER),
            "BLOCK": Constexpr(type=tl.int16, value=128),
            "bias": Argument(type=tl.int32, specialize=NEVER),
        },
        torch_access_mode=TorchAccessMode.INTERPRETER,
        verify_annotation=True,
    )
    launch(launcher, (1,), actual, 7)
    torch.testing.assert_close(actual, expected_out)
    assert dict(annotated_baked.device_caches) == caches


@pytest.mark.parametrize(
    "annotation,values",
    [
        (Argument(), (None, False, True, 0, 1, 16, 17, 2**31, 2**63, 1.0)),
        (
            Argument(type=(tl.int32, tl.int64, tl.uint64), specialize=NEVER),
            (1, 2**31, 2**63),
        ),
        (Argument(type=tl.int32), (1, 16, 17)),
        (Argument(type=tl.int32, specialize=NEVER), (1, 16, 17)),
        (Argument(type=tl.int32, specialize=Assume(Aligned(16))), (16, 32)),
        (Argument(type=tl.int32, specialize=Assume(EqualTo(1))), (1,)),
        (Argument(type=tl.int32, specialize=Exactly(Aligned(16))), (16, 32)),
        (Argument(specialize=Exactly(EqualTo(1))), (1, 1.0, None)),
        (Argument(specialize=Exactly()), (None, False, True, 0, 1, 16, 17, 2**63)),
        (Constexpr(), (None, False, True, 0, 1, 0.0, -0.0, 1.0, 2**31, 2**63)),
        (Constexpr(type=(tl.int8, tl.int64)), (-128, 127, 128, 2**32)),
        (Constexpr(type=(tl.int8, tl.uint8)), (-128, 0, 127, 128, 255)),
        (
            Constexpr(type=(tl.int1, tl.int32, tl.float64)),
            (False, 0, 0.0, True, 1, 1.0),
        ),
        (
            Constexpr(type=tl.int64, power_of_two_or_zero=True),
            (0, 1, -1, 2, -2, 2**32, -(2**32)),
        ),
    ],
)
def test_annotated_spec_key_is_never_coarser_than_triton(
    scalar_kernel, annotation, values
):
    from intj.annotation import DeviceBinding, _resolve_annotations
    from intj.launcher import _compiler_input, _current_target, _render_params
    from triton.compiler import make_backend

    extra = {"x": annotation}
    params, _, _ = _render_params(
        _resolve_annotations(scalar_kernel, extra), DeviceBinding.NOT_FIXED
    )
    backend = make_backend(_current_target())
    launch = make_launcher(
        scalar_kernel,
        extra_annotation=extra,
        torch_access_mode=TorchAccessMode.INTERPRETER,
        verify_annotation=True,
    )
    module = module_of(launch)
    seen = {}
    for value in values:
        key, nparams = launch.__self__.spec_key(0, value)
        source = _compiler_input(scalar_kernel, params, (value,), backend)
        assert seen.setdefault(key, source) == source, (
            "intj key collides for annotated ASTSource inputs"
        )
        assert nparams == sum(ty != "constexpr" for _, ty in source.signature)


@pytest.mark.parametrize(
    "specialize",
    [
        AUTO,
        NEVER,
        Assume(Aligned(16), PointerRange(32)),
        Assume(PointerRange(32)),
        Exactly(PointerRange(32)),
        Exactly(Aligned(16)),
    ],
)
def test_annotated_pointer_spec_key_is_never_coarser_than_triton(
    pointer_kernel, specialize
):
    from intj.annotation import DeviceBinding, _resolve_annotations
    from intj.launcher import _compiler_input, _current_target, _render_params
    from triton.compiler import make_backend

    base = torch.empty(16, device="cuda")
    large = torch.empty(2**29 + 8, device="cuda")
    extra = {"x": Argument(type=tl.pointer_type(tl.float32), specialize=specialize)}
    params, _, _ = _render_params(
        _resolve_annotations(pointer_kernel, extra), DeviceBinding.NOT_FIXED
    )
    backend = make_backend(_current_target())
    launch = make_launcher(
        pointer_kernel,
        extra_annotation=extra,
        torch_access_mode=TorchAccessMode.INTERPRETER,
    )
    module = module_of(launch)
    seen = {}
    for value in (None, 0, base, base[1:], large, large[1:]):
        key, nparams = launch.__self__.spec_key(0, value)
        source = _compiler_input(pointer_kernel, params, (value,), backend)
        assert seen.setdefault(key, source) == source, (
            "intj key collides for annotated ASTSource inputs"
        )
        assert nparams == 1


@pytest.mark.parametrize("key", [b"", b"\x00" * 8])
def test_annotated_cold_invariant_checks_final_compiler_input(power_kernel, key):
    from intj.annotation import DeviceBinding, _resolve_annotations
    from intj.launcher import _make_compile_callback, _render_params

    params, _, _ = _render_params(
        _resolve_annotations(power_kernel, {"N": Constexpr()}), DeviceBinding.FIXED
    )
    callback = _make_compile_callback(
        power_kernel, params, {}, knob_values=launcher._live_knobs()
    )
    device = torch.cuda.current_device()
    seen = {}  # one launcher's map: the check runs per launcher
    callback(seen, key, 0, device, 1)
    callback(seen, key, 0, device, 1)
    with pytest.raises(RuntimeError, match="two annotated ASTSource inputs"):
        callback(seen, key, 0, device, True)
    callback({}, key, 0, device, True)  # another launcher's map is its own
    objects = {}
    callback(objects, key, 0, device, "a")  # object values included
    callback(objects, key, 0, device, "a")
    with pytest.raises(RuntimeError, match="two annotated ASTSource inputs"):
        callback(objects, key, 0, device, "b")


def _recording_input_launcher(
    kernel, annotation, *, no_gpu, bind_device, monkeypatch, **bound_values
):
    """Builds a launcher of `kernel` whose cache misses record the compiler input.

    `annotation` annotates parameter `x`.  Each miss appends `(key, source)` to
    the returned list, `source` being the ASTSource Triton would compile; with
    `no_gpu` nothing is compiled.

    Returns:
        A `(launcher, compiler_input, calls)` tuple: the launcher (a bound
        handle when `bind_device`), a function mapping public arguments to
        their ASTSource, and the list of recorded misses.
    """
    from intj.launcher import _compiler_input, _current_target, _make_compile_callback
    from triton.backends.compiler import GPUTarget
    from triton.compiler import make_backend

    monkeypatch.setattr("intj.launcher._LOADED", {})
    factory = make_launcher(
        kernel,
        extra_annotation={"x": annotation},
        no_gpu=no_gpu,
        verify_annotation=True,
        torch_access_mode=TorchAccessMode.INTERPRETER,
    )
    native = factory.__self__.bind_device(0, **bound_values) if bind_device else factory
    module = _launcher_module(native)
    # Use the placed, resolved annotations, including the inferred bound dtype.
    from intj.launcher import _LOADED

    params = next(key.params for key, value in _LOADED.items() if value is module)
    backend = make_backend(
        GPUTarget("hip", "gfx942", 64) if no_gpu else _current_target()
    )
    real_compile = (
        None
        if no_gpu
        else _make_compile_callback(
            kernel, params, {}, knob_values=launcher._live_knobs()
        )
    )
    calls = []
    seen = {}  # the launcher's key -> input self-check map

    def compiler_input(*args):
        return _compiler_input(kernel, params, args, backend)

    def record(key, slots, device, *args):
        source = compiler_input(*args)
        result = (
            real_compile(seen, key, slots, device, *args)
            if real_compile
            else (0, 1, 0, slots)
        )
        calls.append((key, source))
        return result

    override_compile(module, record)
    return native, compiler_input, calls


@pytest.mark.parametrize("no_gpu", [True, False], ids=["host", "gpu"])
@pytest.mark.parametrize("bind_device", [False, True], ids=["module", "handle-map"])
@pytest.mark.parametrize(
    "annotation,groups,field",
    [
        pytest.param(
            Argument(specialize=NEVER),
            ((2, 3), (2**31, 2**31 + 1), (2**63, 2**63 + 1)),
            "signature",
            id="ordinary-descriptor",
        ),
        pytest.param(
            Argument(type=tl.int32), ((1, 1), (2, 3)), "signature", id="equal-one"
        ),
        pytest.param(
            Argument(type=tl.int32),
            ((16, 32), (17, 33)),
            "attrs",
            id="scalar-alignment",
        ),
        pytest.param(
            Constexpr(),
            ((False, False), (0, 0), (0.0, 0.0)),
            "constants",
            id="constexpr-descriptor",
        ),
        pytest.param(
            Constexpr(type=tl.float64),
            ((0.0, 0.0), (-0.0, -0.0)),
            "constants",
            id="float64-sign",
        ),
        *[
            pytest.param(
                Constexpr(type=getattr(tl, f"int{width}")),
                ((1, 1), (2, 2)),
                "constants",
                id=f"payload-{width}",
            )
            for width in (8, 16, 32, 64)
        ],
        pytest.param(
            Constexpr(type=tl.int64, power_of_two_or_zero=True),
            ((0, 0), (1, 1), (-1, -1), (2, 2)),
            "constants",
            id="power-payload",
        ),
    ],
)
def test_runtime_compiler_input_invariant(
    scalar_kernel,
    monkeypatch,
    no_gpu,
    bind_device,
    annotation,
    groups,
    field,
):
    native, compiler_input, calls = _recording_input_launcher(
        scalar_kernel,
        annotation,
        no_gpu=no_gpu,
        bind_device=bind_device,
        monkeypatch=monkeypatch,
    )
    controls = (0, 1) if bind_device else (0, 0, 1)
    saved = {}
    for group_index, group in enumerate(groups):
        for value in group:
            source = compiler_input(value)
            blob, slots = native.__self__.spec_key(0, value)
            assert slots == sum(ty != "constexpr" for _, ty in source.signature)
            assert native(*controls, value) is None
            assert saved.setdefault(blob, source) == source, (
                "cached key has different CompilerInput"
            )
            assert len(calls) == group_index + 1, (
                "distinct CompilerInput must cause a miss"
            )
            assert calls[-1] == (blob, source), (
                "cache hit must retain its CompilerInput"
            )
    assert len({getattr(source, field) for _, source in calls}) == len(groups)
    # Revisit the first entry after all distinct inputs have been installed.
    assert native(*controls, groups[0][0]) is None
    assert len(calls) == len(groups)


@pytest.mark.parametrize("bind_device", [False, True], ids=["module", "handle-map"])
@pytest.mark.parametrize(
    "fact",
    [
        "dtype",
        "alignment",
        # only a backend with pointer_range splits on the 2 GiB storage bound
        pytest.param("range", marks=needs_pointer_range),
    ],
)
def test_runtime_pointer_compiler_input_invariant(
    pointer_kernel, monkeypatch, bind_device, fact
):
    from triton import knobs

    # Keep the range expectation independent of the caller's environment.
    monkeypatch.setattr(knobs.amd, "use_buffer_ops", True)
    base = torch.empty(16, device="cuda")
    if fact == "dtype":
        groups = ((base, base[4:]), (base.to(torch.int32), base.to(torch.int32)))
        field = "signature"
    elif fact == "alignment":
        groups = ((base, base[4:]), (base[1:], base[2:]))
        field = "attrs"
    else:
        large = torch.empty(2**29, device="cuda")
        groups = ((base, base[4:]), (large[:4], large[4:8]))
        field = "attrs"
    test_runtime_compiler_input_invariant(
        pointer_kernel, monkeypatch, False, bind_device, Argument(), groups, field
    )


def _attrs_for_backend(attrs, no_gpu):
    """Drops `tt.pointer_range` where the launcher's backend does not specialize on it.

    An `Assume(PointerRange(...))` is parsed by the triton backend itself, and
    one without `pointer_range` (NVIDIA) does not know the `S` descriptor: the
    annotation is accepted and contributes no attribute. A path left with no
    attributes is omitted, as `_compiler_input` omits it.

    `no_gpu` launchers are keyed against the gfx942 target
    `_recording_input_launcher` pins, so they always carry it.
    """
    if no_gpu or POINTER_RANGE:
        return attrs
    kept = (
        (path, tuple(a for a in parsed if a[0] != "tt.pointer_range"))
        for path, parsed in attrs
    )
    return tuple((path, parsed) for path, parsed in kept if parsed)


@pytest.mark.parametrize("no_gpu", [True, False], ids=["host", "gpu"])
@pytest.mark.parametrize(
    "annotation,values,signature,constants,attrs",
    [
        (Argument(type=tl.int32, specialize=NEVER), (7, 9), "i32", (), ()),
        (Argument(type=tl.float64, specialize=NEVER), (1.25, 2.5), "fp64", (), ()),
        (
            Argument(
                type=tl.pointer_type(tl.float32),
                specialize=Assume(Aligned(16), PointerRange(32)),
            ),
            (None, 0),
            "*fp32",
            (),
            (((0,), (("tt.divisibility", 16), ("tt.pointer_range", 32))),),
        ),
        (Constexpr(type=None), (None, None), "constexpr", (((0,), ("none",)),), ()),
        (
            Argument(
                type=tl.pointer_type(tl.float32),
                specialize=Assume(Aligned(16), PointerRange(32)),
                bind_value=BindValue.TENSOR,
            ),
            (None, None),
            "*fp32",
            (),
            (((0,), (("tt.divisibility", 16), ("tt.pointer_range", 32))),),
        ),
    ],
)
def test_no_map_compiler_input_invariant(
    scalar_kernel,
    monkeypatch,
    no_gpu,
    annotation,
    values,
    signature,
    constants,
    attrs,
):
    from intj.launcher import CompilerInput

    binding = isinstance(annotation, Argument) and annotation.bind_value is not None
    bound_values = (
        {"x": torch.empty(4, device="cpu" if no_gpu else "cuda")} if binding else {}
    )
    native, compiler_input, calls = _recording_input_launcher(
        scalar_kernel,
        annotation,
        no_gpu=no_gpu,
        bind_device=True,
        monkeypatch=monkeypatch,
        **bound_values,
    )
    expected = CompilerInput(
        (("x", signature),), constants, _attrs_for_backend(attrs, no_gpu), ()
    )
    saved = None
    for value in values:
        if bound_values:
            bound_values["x"].set_(
                torch.empty(8, device="cpu" if no_gpu else "cuda").untyped_storage(),
                0,
                (8,),
                (1,),
            )
        args = () if bound_values else (value,)
        assert native(0, 1, *args) is None
        source = compiler_input(
            *args
        )  # Compare EVERY accepted call, including the hit.
        if saved is None:
            saved = calls[0][1]
        assert source == saved == expected, (
            "no-map call changed its saved CompilerInput"
        )
        assert calls == [(b"", expected)], "no-map handle must compile exactly once"


@pytest.mark.parametrize(
    "nparams,nconstexpr", [(1, 0), (1, 1), (6, 0), (7, 0), (8, 0), (7, 2), (8, 3)]
)
def test_render_params_layout_and_call_indexes(tmp_path, nparams, nconstexpr):
    """The 7/8 parameter cases straddle the device byte's word boundary."""
    import importlib.util

    from intj.annotation import DeviceBinding, _resolve_annotations
    from intj.launcher import _render_params

    names = [f"p{i}" for i in range(nparams)]
    cxnames = [f"C{i}" for i in range(nconstexpr)]
    sig = ", ".join(names + [f"{c}: tl.constexpr" for c in cxnames])
    # on disk, not exec'd: @triton.jit calls inspect.getsourcelines on it
    path = tmp_path / f"k_{nparams}_{nconstexpr}.py"
    path.write_text(
        f"import triton\nimport triton.language as tl\n\n\n@triton.jit\ndef k({sig}):\n    pass\n"
    )
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # pyright: ignore[reportAttributeAccessIssue]  # 3.8 stubs lack it

    params, device_offset, nwords = _render_params(
        _resolve_annotations(module.k, None), DeviceBinding.NOT_FIXED
    )

    assert [p.name for p in params] == names + cxnames
    assert [p.index for p in params] == list(range(nparams + nconstexpr))
    assert [p.call_index for p in params] == list(range(nparams + nconstexpr))
    assert [p.annotation.key_fields[0].offset for p in params[:nparams]] == [
        8 * nconstexpr + i for i in range(nparams)
    ]
    assert [p.annotation.key_fields[0].offset for p in params[nparams:]] == [
        8 * i for i in range(nconstexpr)
    ]
    assert device_offset is not None
    assert device_offset == 8 * nconstexpr + nparams + nconstexpr
    assert nwords == (device_offset + 8) // 8


def _make_render_key(**overrides):
    """Returns a valid two-parameter `RenderKey` with `overrides` applied."""
    from intj.annotation import DeviceBinding, ResolvedParam
    from intj.launcher import _render_params

    ordinary = CanonicalAnnotation(
        "argument", None, "auto", "auto", "auto", (), False, None, ()
    )
    constexpr = CanonicalAnnotation(
        "constexpr", None, "auto", "auto", "auto", (), False, None, ()
    )
    params, device_offset, nwords = _render_params(
        (ResolvedParam("x", 0, ordinary), ResolvedParam("BLOCK", 1, constexpr)),
        DeviceBinding.NOT_FIXED,
    )
    fields = dict(
        module_name="m",
        kernel_repr="a.b",
        params=params,
        nwords=nwords,
        device_binding=DeviceBinding.NOT_FIXED,
        device_offset=device_offset,
        verify_annotation=False,
        no_gpu=False,
        max_slots=1,
        spec_pointer_range=1,
        driver_path="/driver.so",
        launch_symbol="launch",
        device_symbol="device_get",
        error_symbol="error",
        error_style="return",
        torch_access_mode="runtime_shim",
        torch_version=None,
        cxx_abi=None,
        kernel_cache="intj",
        cache_include_dirs=(),
        cache_archives=(),
        python_version=cpython_abi.python_version(),
        free_threaded=bool(sysconfig.get_config_var("Py_GIL_DISABLED")),
        cpython_static_compile_header=cpython_abi.header_for() or "",
        pointer_types=(),
        template="aa",
        runtime_header="bb",
        cache_key="c",
        target=("hip", "gfx942", 64),
        options="o",
        triton=("3.8.0", 1, 2.0),
        compiler=("gcc", "13"),
        build_flags=(),
        ext_suffix=".so",
        intj_version=(0, 1),
        global_values=(),
    )
    return RenderKey(**{**fields, **overrides})


def test_package_version_has_one_source():
    data = tomllib.loads(pathlib.Path("pyproject.toml").read_text())
    assert "version" not in data["project"]
    assert data["project"]["dynamic"] == ["version"]
    assert data["tool"]["setuptools"]["dynamic"]["version"] == {
        "attr": "intj._version.__version__"
    }
    assert intj.__version__ == "0.1.0"


def test_render_key_uses_major_minor_version_only(monkeypatch):
    import intj.launcher as launcher

    monkeypatch.setattr(launcher, "__version__", "7.8.9")
    assert launcher._cache_version() == (7, 8)
    monkeypatch.setattr(launcher, "__version__", "7.8.10")
    assert launcher._cache_version() == (7, 8)
    monkeypatch.setattr(launcher, "__version__", "7.9.0")
    assert launcher._cache_version() == (7, 9)


@pytest.mark.parametrize(
    "cache,language",
    [
        (KernelCache.INTJ, "c"),
        (KernelCache.TSL, "c++"),
        (KernelCache.ABSL, "c++"),
    ],
)
def test_render_key_tracks_cache_compiler_language(
    scalar_kernel, monkeypatch, cache, language
):
    import intj.launcher as launcher

    keys = []
    monkeypatch.setattr(
        launcher, "_provision", lambda _: {"include_dirs": (), "archives": ()}
    )

    class Captured(Exception):
        pass

    def capture(key, *rest):
        keys.append(key)
        raise Captured

    monkeypatch.setattr(launcher, "_loaded_module", capture)
    with pytest.raises(Captured):
        module_of(
            make_launcher(
                scalar_kernel,
                no_gpu=True,
                torch_access_mode=TorchAccessMode.INTERPRETER,
                kernel_cache=cache,
            )
        )
    assert len(keys) == 1
    assert keys[0].compiler == compiler_identity(language)
    assert ("-fvisibility=hidden" in keys[0].build_flags) == (language == "c++")


@pytest.mark.parametrize(
    "differing",
    [
        {"nwords": 5},
        {"dynamic": (DynamicSlot("num_warps", 8, 16, 0),)},
        {"cache_key": "d"},
    ],
)
def test_value_types_compare_hash_and_serialize(differing):
    """A `RenderKey` carries only immutable fields, so it behaves as a value.

    A list field anywhere in it would compare fine and raise on hash().
    """
    build = _make_render_key
    one, same, other = build(), build(), build(**differing)
    assert one == same and one != other
    assert hash(one) == hash(same) and hash(one) != hash(other)
    assert len({one: 1, same: 2, other: 3}) == 2
    dump = lambda v: json.dumps(dataclasses.asdict(v), sort_keys=True)  # noqa: E731
    assert dump(one) == dump(same) and dump(one) != dump(other)


def test_render_key_digest_tracks_every_field():
    key = _make_render_key()
    for field in dataclasses.fields(key):
        value = getattr(key, field.name)
        if isinstance(value, bool):
            changed = not value
        elif isinstance(value, str):
            changed = "zz"
        elif isinstance(value, int):
            changed = value + 1
        elif isinstance(value, tuple):
            changed = value + ("zz",)
        else:
            changed = (1, 2)
        assert (
            dataclasses.replace(key, **{field.name: changed}).digest() != key.digest()
        ), field.name


def test_free_threaded_build_changes_module_digest():
    key = _make_render_key()
    assert dataclasses.replace(key, free_threaded=True).digest() != key.digest()


def test_render_rejects_mismatched_free_threaded_headers(tmp_path, capfd):
    key = _make_render_key(
        python_version=cpython_abi.python_version(),
        cpython_static_compile_header=cpython_abi.header_for() or "",
        free_threaded=not bool(sysconfig.get_config_var("Py_GIL_DISABLED")),
    )
    with pytest.raises(subprocess.CalledProcessError):
        build(tmp_path / "m.so", key)
    assert (
        "intj: compiling against headers of the wrong GIL build"
        in capfd.readouterr().err
    )


def test_artifact_layout_and_nested_kernels():
    """The symbol is the kernel's name; the path says which kernel and which build."""

    def make():
        @triton.jit
        def inner(x, o, n, BLOCK: tl.constexpr):
            off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
            mask = off < n
            tl.store(o + off, tl.load(x + off, mask=mask) * 2, mask=mask)

        return inner

    nested = make()  # qualname carries `<locals>`, which a C symbol cannot
    launcher = make_launcher(nested)
    module = module_of(launcher)
    path = pathlib.Path(str(module.__file__))
    assert path.name.startswith("inner.")
    assert path.parent.name == nested.__module__
    assert len(path.parent.parent.name) == 64  # the RenderKey digest

    x = torch.randn(1024, device="cuda")
    o = torch.empty(1024, device="cuda")
    launch(launcher, (8,), x, o, 1024, 128)
    torch.testing.assert_close(o, x * 2.0)


CACHES = list(KernelCache)  # make_launcher provisions whichever is missing


def _provisioned(cache):
    """Install `cache` up front, so an offline run skips rather than fails.

    Narrow on purpose: only a failed *install* skips.  Routing every
    UnsupportedKernel here would hide the refusals these tests exist to check.
    """
    try:
        install(cache)
    except INSTALL_ERRORS as error:
        pytest.skip(f"{cache.value} could not be installed: {error}")
    return cache


@pytest.mark.parametrize("cache", CACHES, ids=[c.value for c in CACHES])
def test_kernel_cache_backends_launch_the_same(cache):
    """Every backend is a kernel cache: same key in, same kernel out.

    tsl and absl are C++ maps, so they also drag a RUNTIME_SHIM module into a C++
    build -- which is the part most likely to break.
    """
    _provisioned(cache)
    launcher = make_launcher(
        scale, torch_access_mode=TorchAccessMode.RUNTIME_SHIM, kernel_cache=cache
    )
    x = torch.randn(1024, device="cuda")
    o = torch.empty(1024, device="cuda")
    for block in (64, 128):  # two specializations, so the cache is used
        check_matches_triton(
            scale, launcher, (triton.cdiv(1024, block),), (x, o, 1024, 2.0, block), 1
        )


@pytest.mark.parametrize("cache", [KernelCache.TSL, KernelCache.ABSL])
def test_cpp_cache_modules_with_different_key_widths(cache, tmp_path):
    """Two loaded C++ caches must not share one key policy across key widths."""
    _provisioned(cache)
    source = tmp_path / "mixed_key_widths.py"
    source.write_text(
        "import triton\nimport triton.language as tl\n"
        "@triton.jit\ndef one(x):\n    pass\n"
        "@triton.jit\ndef two(x, C: tl.constexpr):\n    pass\n"
    )
    script = """
import importlib.util
import sys
from intj import KernelCache, make_launcher
from intj.launcher import module_of
from intj.torch_intf.torch_abi import TorchAccessMode

spec = importlib.util.spec_from_file_location("mixed_key_widths", sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
options = dict(no_gpu=True, kernel_cache=KernelCache(sys.argv[2]),
               torch_access_mode=TorchAccessMode.INTERPRETER)
one = make_launcher(module.one, **options)
two = make_launcher(module.two, **options)
assert len(one.__self__.spec_key(0, 1)[0]) == 8
assert len(two.__self__.spec_key(0, 1, 2)[0]) == 16
for value in range(8):
    one(0, 0, (1,), value)
    two(0, 0, (1,), value, value + 7)
"""
    run = subprocess.run(
        [sys.executable, "-c", script, str(source), cache.value],
        cwd=tmp_path,
        env=_tested_intj_env(),
        capture_output=True,
        text=True,
    )
    assert run.returncode == 0, run.stderr


def test_kernel_cache_choice_reaches_the_digest():
    modules = {
        c: pathlib.Path(
            str(module_of(make_launcher(scale, kernel_cache=_provisioned(c))).__file__)
        )
        for c in CACHES
    }
    digests = {c: path.parent.parent.name for c, path in modules.items()}
    assert len(set(digests.values())) == len(CACHES)


def test_kernel_cache_is_installed_on_demand(monkeypatch):
    """An unprovisioned backend installs itself, and its toolchain is what builds."""
    calls: list[KernelCache] = []
    toolchain = {
        "include_dirs": ("/nowhere/include",),
        "library_dirs": (),
        "archives": (),
    }
    # installed once calls is non-empty, like the real complete-tree check
    monkeypatch.setattr(
        "intj.launcher.toolchain_for", lambda cache: toolchain if calls else None
    )
    monkeypatch.setattr("intj.launcher.install", lambda c: calls.append(c) or toolchain)

    # stop before the real build: the point is what reaches it, not that a
    # fabricated include directory compiles
    keys: list[RenderKey] = []

    class Captured(Exception):
        pass

    def capture(key, *rest):
        keys.append(key)
        raise Captured

    monkeypatch.setattr("intj.launcher._loaded_module", capture)
    launch = make_launcher(scale, kernel_cache=KernelCache.TSL)
    assert calls == []  # decoration provisions nothing
    launch.__self__.check()
    assert calls == [KernelCache.TSL]  # at check(), never on a build
    with pytest.raises(Captured):
        module_of(launch)
    # the requested backend, and what install returned actually reaches the build
    assert calls == [KernelCache.TSL]
    assert keys[-1].cache_include_dirs == toolchain["include_dirs"]


def test_kernel_cache_install_failure_is_reported_once(monkeypatch):
    """A failed install is refused once, with the reason and the retry command.

    The install is not re-tried.
    """
    attempts = []

    def explode(cache):
        attempts.append(cache)
        raise OSError("no network")

    monkeypatch.setattr("intj.launcher.toolchain_for", lambda cache: None)
    monkeypatch.setattr("intj.launcher.install", explode)
    monkeypatch.setattr("intj.launcher._INSTALL_FAILED", {})
    for _ in range(3):
        with pytest.raises(
            UnsupportedKernel,
            match=rf"`{re.escape(pathlib.Path(sys.executable).name)} -m "
            r"intj\.kernel_cache tsl`.*no network",
        ):
            make_launcher(scale, kernel_cache=KernelCache.TSL).__self__.check()
    assert attempts == [KernelCache.TSL]  # remembered, not re-attempted


def test_kernel_cache_install_bug_is_not_dressed_up_as_a_download_failure(monkeypatch):
    monkeypatch.setattr("intj.launcher.toolchain_for", lambda cache: None)
    monkeypatch.setattr("intj.launcher._INSTALL_FAILED", {})
    monkeypatch.setattr("intj.launcher.install", lambda cache: cache.no_such_attribute)
    with pytest.raises(AttributeError):
        make_launcher(scale, kernel_cache=KernelCache.TSL).__self__.check()


def test_kernel_cache_install_is_the_last_refusal(monkeypatch):
    """A kernel that will be rejected anyway must not pay for a download first.

    Only GPU-free refusals can precede it: an unknown `options=` name needs the
    target, so it raises on the first call, after make_launcher provisioned."""
    monkeypatch.setattr("intj.launcher.toolchain_for", lambda cache: None)

    def explode(cache):
        raise AssertionError("installed before refusing a global-reading kernel")

    monkeypatch.setattr("intj.launcher.install", explode)
    with pytest.raises(UnsupportedKernel, match="global variable"):
        make_launcher(uses_global, kernel_cache=KernelCache.ABSL).__self__.check()


def test_refuses_non_jit_function():
    with pytest.raises(UnsupportedKernel):
        make_launcher(lambda: None).__self__.check()


def test_importing_intj_imports_neither_triton_nor_torch():
    """Every module reaches them through `intj._imports`, on first use."""
    code = (
        "import sys, intj, intj.compat, intj.grid, intj.heuristic, intj.tuning, "
        "intj.torch_intf.abi_detect, intj.torch_intf.cpp_detect; "
        "print(sorted({'triton', 'torch'} & set(sys.modules)))"
    )
    run = subprocess.run(
        [sys.executable, "-c", code],
        env={**os.environ, "PYTHONPATH": str(pathlib.Path(__file__).parents[1])},
        capture_output=True,
        text=True,
    )
    assert run.returncode == 0, run.stderr
    assert run.stdout.strip() == "[]"


@pytest.mark.parametrize("version", [None, "3.6.0"])
def test_launcher_requires_triton_extra(monkeypatch, version):
    installed = types.ModuleType("triton")
    setattr(installed, "__version__", version)
    monkeypatch.setitem(sys.modules, "triton", installed if version else None)
    monkeypatch.setattr("intj._imports.triton", None)  # forget the cached import
    find = intj._imports._find_triton  # and the cached lookup, for this test only
    monkeypatch.setattr(
        intj._imports,
        "_find_triton",
        functools.lru_cache(maxsize=None)(find.__wrapped__),
    )

    found = "" if version is None else f" (found {version})"
    with pytest.raises(UnsupportedKernel) as excinfo:
        make_launcher(object()).__self__.check()
    assert str(excinfo.value) == (
        f"intj: Triton >=3.7 is required; install `intj[launcher]`{found}"
    )
    assert isinstance(excinfo.value.__cause__, ModuleNotFoundError) == (version is None)


def test_refuses_kernel_reading_globals():
    with pytest.raises(UnsupportedKernel, match="global variable"):
        make_launcher(uses_global).__self__.check()


@pytest.mark.parametrize("path", ["auto", "generic"])
def test_tensor_annotated_param_takes_only_a_tensor(path):
    """`x: torch.Tensor` / `y: tl.tensor` decode through the tensor-only path.

    A tensor or `nn.Parameter` launches like an unannotated one; anything else,
    `None` included, is a TypeError (Triton would take `None` as a constexpr).
    """
    extra = {"n": Argument(type=tl.int32)} if path == "generic" else None
    launch = make_launcher(axpy_tensor, extra_annotation=extra)
    x = torch.randn(64, device="cuda")
    y = torch.nn.Parameter(torch.randn(64, device="cuda"))
    o = torch.empty(64, device="cuda")
    launch(0, 0, 1, x, y, o, 64, 2.0, True, None, 64)
    torch.testing.assert_close(o, 2.0 * x + y.detach())
    for bad in (None, 0, 1, 16, 2.5, True, x.cpu().numpy()):
        with pytest.raises(TypeError, match="'y' is annotated as a tensor"):
            launch(0, 0, 1, x, bad, o, 64, 2.0, True, None, 64)


@pytest.mark.parametrize("path", ["auto", "generic"])
def test_optional_tensor_param_takes_a_tensor_or_none(path):
    """`y: torch.Tensor | None` / `o: tl.tensor | None` take a tensor or None.

    Each is launched as Triton would; anything else is a TypeError.
    """
    extra = {"n": Argument(type=tl.int32)} if path == "generic" else None
    launch = make_launcher(axpy_optional, extra_annotation=extra)
    x = torch.randn(64, device="cuda")
    y = torch.nn.Parameter(torch.randn(64, device="cuda"))
    o = torch.empty(64, device="cuda")
    launch(0, 0, 1, x, y, o, 64, 2.0, True, None, 64)
    torch.testing.assert_close(o, 2.0 * x + y.detach())
    launch(0, 0, 1, x, None, o, 64, 3.0, True, None, 64)
    torch.testing.assert_close(o, 3.0 * x)
    launch(0, 0, 1, x, y, None, 64, 3.0, True, None, 64)
    for bad in (0, 1, 16, 2.5, True, x.cpu().numpy()):
        with pytest.raises(TypeError, match="'y' is annotated as a tensor or None"):
            launch(0, 0, 1, x, bad, o, 64, 2.0, True, None, 64)
        with pytest.raises(TypeError, match="'o' is annotated as a tensor or None"):
            launch(0, 0, 1, x, y, bad, 64, 2.0, True, None, 64)
    with pytest.raises(TypeError, match="'x' is annotated as a tensor; got NoneType"):
        launch(0, 0, 1, None, y, o, 64, 2.0, True, None, 64)


CONSTEXPR_GLOBAL = tl.constexpr(3)


@triton.jit
def reads_constexpr_global(x, o, n, BLOCK: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = off < n
    tl.store(o + off, tl.load(x + off, mask=mask) * CONSTEXPR_GLOBAL, mask=mask)


def test_constant_globals_are_a_caller_promise():
    """A kernel reading a constexpr global is refused unless the caller promises.

    With `assume_constant_globals=True` it launches with the value triton saw
    when it hashed the kernel, in both call forms.
    """
    with pytest.raises(UnsupportedKernel, match="assume_constant_globals"):
        make_launcher(reads_constexpr_global).__self__.check()
    x = torch.randn(64, device="cuda")
    for launch in (
        make_launcher(reads_constexpr_global, assume_constant_globals=True),
        make_launcher(assume_constant_globals=True)(reads_constexpr_global),
    ):
        o = torch.empty(64, device="cuda")
        launch(0, 0, 1, x, o, 64, 64)
        torch.testing.assert_close(o, 3 * x)


def test_constant_globals_are_not_rechecked_on_a_hit(monkeypatch):
    """No per-launch read: changing the global after the build is not seen.

    Unsupported and undetected (docs/Usage.md); Triton raises instead.
    """
    launch = make_launcher(reads_constexpr_global, assume_constant_globals=True)
    x = torch.randn(64, device="cuda")
    o = torch.empty(64, device="cuda")
    launch(0, 0, 1, x, o, 64, 64)
    monkeypatch.setitem(
        reads_constexpr_global.__globals__, "CONSTEXPR_GLOBAL", tl.constexpr(5)
    )
    o.zero_()
    launch(0, 0, 1, x, o, 64, 64)
    torch.testing.assert_close(o, 3 * x)


_TILES_KERNEL = """
import triton
import triton.language as tl

TILES = {tiles}


@triton.constexpr_function
def ntiles():
    return len(TILES)


@triton.jit
def kernel(o):
    tl.store(o, ntiles())
"""

_CONSTEXPR_KERNEL = """
import triton
import triton.language as tl

TILES = tl.constexpr(len({tiles}))


@triton.jit
def kernel(o):
    tl.store(o, TILES)
"""


def _module_kernel(path, source):
    path.write_text(source)
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # pyright: ignore[reportAttributeAccessIssue]  # 3.8 stubs lack it
    return module.kernel


@pytest.mark.parametrize("kind", ["plain", "constexpr"])
def test_different_global_values_never_share_a_module(tmp_path, kind):
    """Two kernels differing only in a global's value get two modules.

    Triton's `cache_key` hashes a `tl.constexpr` global's value but not a plain
    one's (a tuple read by a `constexpr_function`, like aiter's
    `_SUPPORTED_TILES`), so for `plain` only the values in the `RenderKey`
    keep the second launcher off the first's module.  The binaries are
    Triton's business: its own compile cache keys on `cache_key` too, so for
    `plain` it hands both the first kernel compiled, exactly as `kernel[grid]`
    does -- only `constexpr` can check the outputs.
    """
    template = _TILES_KERNEL if kind == "plain" else _CONSTEXPR_KERNEL
    kernels = []
    for i, tiles in enumerate((((1, 2),), ((1, 2), (3, 4)))):
        # one module name, so nothing but the global tells the two apart
        (tmp_path / str(i)).mkdir()
        path = tmp_path / str(i) / f"globals_{kind}.py"
        kernels.append(_module_kernel(path, template.format(tiles=tiles)))
    assert (kernels[0].cache_key == kernels[1].cache_key) == (kind == "plain")
    launches = [make_launcher(k, assume_constant_globals=True) for k in kernels]
    out = []
    for launch in launches:
        o = torch.zeros(1, device="cuda", dtype=torch.int32)
        launch(0, 0, 1, o)
        out.append(o.item())
    assert module_of(launches[0]) is not module_of(launches[1])
    if kind == "constexpr":
        assert out == [1, 2]


def test_global_values_canonicalize_or_refuse(tmp_path):
    from triton._C.libtriton import ir

    from intj.launcher import _canonical_global

    assert _canonical_global(tl.constexpr(ir.PROPAGATE_NAN.ALL)) == (
        "constexpr",
        ("enum", "triton._C.libtriton.ir", "PROPAGATE_NAN", "ALL"),
    )
    assert _canonical_global(((1, 2), tl.constexpr(True))) == (
        "tuple",
        ("tuple", ("int", "1"), ("int", "2")),
        ("constexpr", ("bool", True)),
    )
    # Triton's `used_global_vals` holds a `deepcopy` of the global (jit.py's
    # `record_reference`), and a pybind11 enum member does not deepcopy to
    # itself, unlike a Python `enum.Enum` member. Canonicalization must not
    # rely on identity with the live member.
    import copy
    import enum

    deepcopied = copy.deepcopy(ir.PROPAGATE_NAN.ALL)
    assert deepcopied is not ir.PROPAGATE_NAN.ALL
    assert _canonical_global(tl.constexpr(deepcopied)) == (
        "constexpr",
        ("enum", "triton._C.libtriton.ir", "PROPAGATE_NAN", "ALL"),
    )

    class _PyEnum(enum.Enum):
        ALL = 1

    py_deepcopied = copy.deepcopy(_PyEnum.ALL)
    assert py_deepcopied is _PyEnum.ALL  # Python enums are singletons
    assert _canonical_global(py_deepcopied) == (
        "enum",
        __name__,
        "test_global_values_canonicalize_or_refuse.<locals>._PyEnum",
        "ALL",
    )
    json.dumps(
        _make_render_key(global_values=(("G", "m", _canonical_global((1,))),)).digest()
    )
    opaque = _module_kernel(
        tmp_path / "opaque_global.py", _TILES_KERNEL.format(tiles="object()")
    )
    with pytest.raises(UnsupportedKernel, match="'TILES' holds a object") as info:
        make_launcher(opaque, assume_constant_globals=True).__self__.check()
    assert "as an argument" not in str(info.value)


_CLASS_KERNEL = """
from typing import NamedTuple

import triton
import triton.language as tl


class {cls}(NamedTuple):
    foo: tl.constexpr


T = {cls}


@triton.jit
def kernel(o):
    tl.store(o, T(3).foo)
"""


def test_class_global_is_keyed_by_name_and_warns_once_per_build(tmp_path):
    """A class global (the NamedTuple repro) is keyed by its qualified name.

    It is refused by default, and launched under `assume_constant_globals=True`
    with one ClassGlobalWarning per `make_launcher` and none on a launch.
    """
    from intj import ClassGlobalWarning

    kernel = _module_kernel(tmp_path / "class_global.py", _CLASS_KERNEL.format(cls="A"))
    with pytest.raises(UnsupportedKernel, match="assume_constant_globals"):
        make_launcher(kernel).__self__.check()
    with pytest.warns(ClassGlobalWarning, match=r"\['class_global\.A'\]") as record:
        launch = make_launcher(kernel, assume_constant_globals=True)
        launch.__self__.check()
    assert len(record) == 1
    assert record[0].filename == __file__  # stacklevel points at the caller
    o = torch.zeros(1, device="cuda", dtype=torch.int32)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        launch(0, 0, 1, o)  # the miss
        launch(0, 0, 1, o)  # a hit
    assert o.item() == 3
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warnings.filterwarnings("ignore", category=ClassGlobalWarning)
        make_launcher(
            kernel, assume_constant_globals=True
        ).__self__.check()  # filterable


def test_different_class_globals_never_share_a_module(tmp_path):
    """Kernels differing only in a class global get two modules.

    Same module name, same kernel source (so the same cache_key): only the
    class's qualified name in the RenderKey tells the two apart.
    """
    kernels = []
    for cls in ("A", "B"):
        (tmp_path / cls).mkdir()
        path = tmp_path / cls / "class_global.py"
        kernels.append(_module_kernel(path, _CLASS_KERNEL.format(cls=cls)))
    assert kernels[0].cache_key == kernels[1].cache_key
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", intj.ClassGlobalWarning)
        launches = [make_launcher(k, assume_constant_globals=True) for k in kernels]
        for launch in launches:
            launch.__self__.check()  # where the ignored warning fires
    assert module_of(launches[0]) is not module_of(launches[1])


_BlockedLayout = gl.BlockedLayout  # a class global, like Triton's layout imports


@gluon.jit
def gluon_class_global(o, BLOCK: gl.constexpr, WARP: gl.constexpr):
    off = gl.arange(
        0,
        BLOCK,
        layout=_BlockedLayout([1], [WARP], [4], [0]),  # pyright: ignore[reportArgumentType]  # JIT constexpr is an int
    )
    gl.store(o + off, off)


def test_gluon_layout_class_global_launches():
    """A Gluon kernel building its layout from a global layout class."""
    target = triton.runtime.driver.active.get_current_target()
    assert target is not None
    warp = target.warp_size
    with pytest.warns(intj.ClassGlobalWarning, match="BlockedLayout"):
        launch = make_launcher(gluon_class_global, assume_constant_globals=True)
        launch.__self__.check()
    o = torch.full((256,), -1, device="cuda", dtype=torch.int32)
    launch(0, 0, 1, o, 256, warp)
    torch.testing.assert_close(o, torch.arange(256, device="cuda", dtype=torch.int32))


def test_decorating_under_triton_interpret_succeeds_and_defers_the_refusal():
    """`@make_launcher` under `TRITON_INTERPRET=1` defers its refusal to the call.

    `@triton.jit` there returns an InterpretedFunction, not a JITFunction, so a
    module-level `@make_launcher` must not raise at decoration -- only its
    first call may (intj issue 1).
    """
    from triton import knobs
    from triton.runtime.interpreter import InterpretedFunction
    from triton.runtime.jit import JITFunction

    with knobs.runtime.scope():
        knobs.runtime.interpret = True

        @triton.jit
        def interp_scale(x, o, n, s, BLOCK: tl.constexpr):
            off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
            mask = off < n
            tl.store(o + off, tl.load(x + off, mask=mask) * s, mask=mask)

        assert isinstance(interp_scale, InterpretedFunction)
        assert not isinstance(interp_scale, JITFunction)

        deferred = make_launcher(interp_scale)  # must not raise
        with pytest.raises(UnsupportedKernel, match="TRITON_INTERPRET=1"):
            module_of(deferred)


def test_fresh_launcher_builds_and_launches_normally_once_interpret_is_off():
    """A kernel decorated with interpret off builds and launches normally.

    This holds after the interpreter-mode launcher above refuses: it goes
    through the usual path.
    """
    from triton import knobs

    with knobs.runtime.scope():
        knobs.runtime.interpret = True

        @triton.jit
        def interp_scale2(x, o, n, s, BLOCK: tl.constexpr):
            off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
            mask = off < n
            tl.store(o + off, tl.load(x + off, mask=mask) * s, mask=mask)

        deferred = make_launcher(interp_scale2)
        with pytest.raises(UnsupportedKernel, match="TRITON_INTERPRET=1"):
            module_of(deferred)

    launch_fn = make_launcher(scale)
    x = torch.arange(8, device="cuda", dtype=torch.float32)
    o = torch.empty_like(x)
    launch(launch_fn, (1,), x, o, 8, 2.0, 8)
    torch.testing.assert_close(o, x * 2.0)


def test_launcher_built_with_interpret_off_keeps_launching_if_turned_on_later(
    scale_launcher,
):
    """A built launcher keeps launching after interpret is turned on.

    The interpreter refusal lives only in `_build` (the first call): once the
    module is swapped in, a later call never reads the knob again.
    """
    from triton import knobs

    module_of(scale_launcher)  # ensure it is already built, interpret off
    x = torch.arange(8, device="cuda", dtype=torch.float32)
    o = torch.empty_like(x)
    with knobs.runtime.scope():
        knobs.runtime.interpret = True
        launch(scale_launcher, (1,), x, o, 8, 2.0, 8)
    torch.testing.assert_close(o, x * 2.0)


def test_accepts_kernel_with_lazy_launch_metadata():
    module_of(make_launcher(uses_launch_metadata, no_gpu=True))


@triton.jit(launch_metadata=lambda *_args: {})
def uses_launch_metadata(x):
    pass


GLOBAL_SCALE = 2.0


@triton.jit
def uses_global(x, o, n, BLOCK: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = off < n
    tl.store(o + off, tl.load(x + off, mask=mask) * GLOBAL_SCALE, mask=mask)


# ---------------------------------------------------------------- torch access

ACCESS_MODES = [
    TorchAccessMode.RUNTIME_SHIM,
    TorchAccessMode.INTERPRETER,
    TorchAccessMode.STATIC_COMPILE,
]


@pytest.fixture(scope="module", params=ACCESS_MODES, ids=lambda m: m.name.lower())
def mode(request):
    return request.param


@pytest.fixture(scope="module")
def scale_by_mode(mode):
    return mode, make_launcher(scale, torch_access_mode=mode)


def _read_corpus():
    """Tensors whose pointer/dtype/storage-size the three modes must agree on.

    The empty views matter most: `Tensor::data_ptr()` returns null for any
    zero-element tensor even when its storage is live and its storage_offset is
    not, so RUNTIME_SHIM must special-case its `data + offset * itemsize`
    arithmetic or it keys differently from the other two modes.
    """
    base = torch.randn(4096, device="cuda")
    half = torch.randn(4096, device="cuda", dtype=torch.float16)
    return [
        base,
        base[1:],  # unaligned
        base[4:],  # aligned view
        half,
        half[3:],
        torch.zeros(0, device="cuda"),  # empty, no storage
        base[4096:],  # empty, live storage, non-zero storage_offset
        half[4096:],
        torch.zeros(4096, device="cuda", dtype=torch.int32),
        torch.nn.Parameter(torch.randn(4096, device="cuda")),
    ]


def test_every_mode_matches_triton_specialization(scale_by_mode):
    """The load-bearing invariant, checked per mode rather than once."""
    _, launcher = scale_by_mode
    module = module_of(launcher)
    o = torch.empty(4096, device="cuda")
    seen = {}
    for x in _read_corpus():
        args = (x, o, 1024, 2.0, 128)
        key = launcher.__self__.spec_key(0, *args)
        spec = repr(triton_specialization(scale, args))
        assert seen.setdefault(key, spec) == spec, (
            f"key collides for {x.dtype} {x.shape}"
        )


def test_modes_agree_on_every_read():
    """The independent-oracle test: INTERPRETER assumes nothing about TensorImpl.

    If torch moves a field, RUNTIME_SHIM keeps reading the old offset and this
    is what notices.
    """
    launchers = {m: make_launcher(scale, torch_access_mode=m) for m in ACCESS_MODES}
    modules = {m: (module_of(k), k) for m, k in launchers.items()}
    o = torch.empty(4096, device="cuda")
    for x in _read_corpus():
        keys = {
            m: k.__self__.spec_key(0, x, o, 1024, 2.0, 128)
            for m, (mod, k) in modules.items()
        }
        distinct = set(keys.values())
        assert len(distinct) == 1, (
            f"modes disagree on {x.dtype} numel={x.numel()}: {keys}"
        )


@pytest.mark.parametrize(
    "dtype_name", ["float32", "float16", "bfloat16", "int32", "uint8"]
)
def test_dtype_code_matches_torch(dtype_name):
    """The THPDtype offset read, and the itemsize table built from it."""
    dtype = getattr(torch, dtype_name)
    code = dtype_code(dtype)
    table = itemsize_table()
    assert 0 <= code < NDTYPES
    assert table is not None, "no dtype row for the torch the suite is running against"
    assert table[code] == torch.empty(0, dtype=dtype).element_size()


def test_recorded_dtypes_match_this_torch():
    """The `dtypes:` row is data, and data goes stale.  Check it against reality.

    A drifted row is not cosmetic: the offsets in the same stanza were measured
    against that set of dtypes, and `itemsize_table` refuses the whole stanza
    when they no longer agree -- so this failing is what a silent fall back to
    another mode on a torch intj thinks it supports looks like before it happens.
    """
    from intj.torch_intf.torch_abi import _DTYPES, live_dtypes, torch_version

    recorded = _DTYPES.get(torch_version())
    assert recorded is not None, "no stanza for the torch the suite is running against"
    assert recorded == live_dtypes(), (
        "intj/torch_intf/torch_abi.toml is stale; regenerate with `python -m intj.torch_intf.abi_detect`"
    )


def test_abi_file_parses_and_refuses_broken_entries():
    """`torch_abi.toml` is pasted by hand, so its parser is a trust boundary."""
    from intj.torch_intf.torch_abi import OFFSETS, _parse_abi

    offsets = "".join(f"{f} = {i}\n" for i, f in enumerate(OFFSETS))
    good = '["2.9"]\n' + offsets + 'dtypes = [["uint8", 1], ["int8", 1]]\n'
    assert _parse_abi(good) == {
        (2, 9): (
            {f: i for i, f in enumerate(OFFSETS)},
            {0: ("uint8", 1), 1: ("int8", 1)},
        )
    }

    for broken in (
        good.replace('["2.9"]', "[2.9"),  # not TOML
        good.replace('"2.9"', '"two.nine"'),  # not a version
        good.replace("cdata = 0\n", ""),  # a missing offset
        good + "extra = 1\n",  # an unknown field
        good.replace("cdata = 0", "cdata = 65536"),  # wraps in the C side's uint16_t
        good.replace("cdata = 0", 'cdata = "0"'),
        good.replace('["int8", 1]', '["int8", "1"]'),
        good.replace('["int8", 1]', '["int8", 1, 2]'),
    ):
        with pytest.raises(ValueError, match="torch_abi.toml"):
            _parse_abi(broken)


def test_drifted_dtypes_refuse_the_whole_stanza(monkeypatch):
    """A torch whose dtypes moved is an unverified torch, offsets and all."""
    from intj.torch_intf import torch_abi as abi

    drifted = dict(abi.live_dtypes())
    drifted.pop(max(drifted))  # torch dropped a dtype since the stanza was taken
    monkeypatch.setattr(abi, "live_dtypes", lambda: drifted)
    abi.itemsize_table.cache_clear()
    abi.layout_for.cache_clear()
    try:
        assert abi.itemsize_table() is None
        assert abi.layout_for() is None
    finally:  # the caches outlive the monkeypatch; every later test reads them
        abi.itemsize_table.cache_clear()
        abi.layout_for.cache_clear()


def test_live_dtypes_have_no_holes():
    """Every `ScalarType` below the highest one torch names has a size.

    `live_dtypes` enumerates `dir(torch)`, so it only sees dtypes the torch
    namespace names.  `ScalarType` is a plain dense enum, so a gap means some
    code in the middle has no attribute to find it by -- and the C side would
    then reject a perfectly ordinary tensor of that dtype as a layout mismatch.
    Density is the check that `dir(torch)` is still an exhaustive enumeration.
    """
    from intj.torch_intf.torch_abi import live_dtypes

    codes = sorted(live_dtypes())
    assert codes, "no dtype was found at all"
    assert codes == list(range(len(codes))), (
        f"dtype codes {sorted(set(range(max(codes))) - set(codes))} are named nowhere"
    )


@pytest.mark.parametrize("nparams", [3, 7, 8, 15, 16])
def test_rendered_key_puts_every_byte_where_python_says(tmp_path, nparams):
    """The generated decoder writes each width at its globally placed offset."""
    import importlib.util

    from intj import launcher as launcher_module

    names = [f"p{i}" for i in range(nparams)]
    path = tmp_path / f"k{nparams}.py"
    path.write_text(
        "import triton\nimport triton.language as tl\n\n\n@triton.jit\n"
        f"def k({', '.join(names + [f'C{bits}: tl.constexpr' for bits in (64, 32, 16, 8)])}):\n"
        "    pass\n"
    )
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    kernel = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kernel)  # pyright: ignore[reportAttributeAccessIssue]  # 3.8 stubs lack it

    annotation = {
        f"C{bits}": intj.Constexpr(type=getattr(tl, f"int{bits}"))
        for bits in (64, 32, 16, 8)
    }
    launch = make_launcher(kernel.k, extra_annotation=annotation)
    module = module_of(launch)
    key = next(
        key for key, loaded in launcher_module._LOADED.items() if loaded is module
    )
    suffix = (
        ".cpp"
        if key.torch_access_mode == TorchAccessMode.STATIC_COMPILE.value
        else ".c"
    )
    source = pathlib.Path(str(module.__file__)).with_name(f"k{suffix}").read_text()
    pack = source[
        source.index("static INTJ_ALWAYS_INLINE int intj_pack") : source.index(
            "/* Parse one grid"
        )
    ]
    assert "((uint8_t *)key)" not in pack
    assert "intj_key_store16" not in pack and "intj_key_store32" not in pack
    for word in range(key.nwords):
        assert pack.count(f"intj_key_store64(key + {word}, key_word_{word});") == 1

    b_i32 = 128  # INTJ_B_I32, +1 when the value is divisible by 16
    args = [3 + 2 * i for i in range(nparams)] + [
        0x0102030405060708,
        0x10203040,
        0x1234,
        0x42,
    ]
    blob, np_ = launch.__self__.spec_key(0, *args)

    assert (len(blob), np_) == (key.nwords * 8, nparams)
    for param in key.params[:nparams]:
        (field,) = param.annotation.key_fields
        assert (field.kind, field.width) == ("descriptor", 1)
        assert blob[field.offset] == b_i32
    for param, value, width in zip(key.params[nparams:], args[nparams:], (8, 4, 2, 1)):
        (field,) = param.annotation.key_fields
        assert (field.kind, field.width) == ("payload", width)
        assert blob[field.offset : field.offset + width] == value.to_bytes(
            width, "little"
        )
    assert key.device_offset is not None
    assert blob[key.device_offset] == 0
    assert blob[key.device_offset + 1 :] == bytes(len(blob) - key.device_offset - 1)

    for i in range(nparams):
        alt = list(args)
        alt[i] = 16
        other, _ = launch.__self__.spec_key(0, *alt)
        offset = key.params[i].annotation.key_fields[0].offset
        assert other[offset] == b_i32 + 1
        assert [b for j, b in enumerate(other) if j != offset] == [
            b for j, b in enumerate(blob) if j != offset
        ]


@pytest.mark.parametrize(
    "mode", [TorchAccessMode.RUNTIME_SHIM, TorchAccessMode.STATIC_COMPILE]
)
def test_unannotated_key_decode_stays_direct(mode):
    suffix = ".cpp" if mode is TorchAccessMode.STATIC_COMPILE else ".c"
    launchers = [
        make_launcher(
            scale, torch_access_mode=mode, no_gpu=True, verify_annotation=verify
        )
        for verify in (False, True)
    ]
    modules = [module_of(k) for k in launchers]
    for module in modules:
        source = (
            pathlib.Path(str(module.__file__)).with_name(f"scale{suffix}").read_text()
        )
        pack = source[
            source.index("static INTJ_ALWAYS_INLINE int intj_pack") : source.index(
                "/* Parse one grid"
            )
        ]

        assert "intj_decoded decoded_" not in pack
        assert "intj_decode_argument" not in pack
        assert "intj_decode_constexpr" not in pack

    args = (torch.empty(8), torch.empty(8), 8, 2.0, 16)
    assert launchers[0].__self__.spec_key(0, *args) == launchers[1].__self__.spec_key(
        0, *args
    )


def test_unsupported_dtype_is_refused_in_every_mode():
    """The path that skipped the dtype check: numel == 0 returns before it.

    RUNTIME_SHIM bails at numel == 0 ahead of its own bound test, and the
    INTERPRETER reader has no bound test at all. Harmless while the dtype had 32
    bits of the key to itself; under a byte code every unmapped dtype would
    alias every other one.  So the check lives in the decode, which all three
    readers pass through.
    """
    o = torch.empty(4096, device="cuda")
    for mode in ACCESS_MODES:
        launch = make_launcher(scale, torch_access_mode=mode)
        module = module_of(launch)
        for x in (
            torch.zeros(4, device="cuda", dtype=torch.complex64),
            torch.zeros(0, device="cuda", dtype=torch.complex64),
        ):
            with pytest.raises(RuntimeError, match="triton does not take"):
                launch.__self__.spec_key(0, x, o, 1024, 2.0, 128)


def test_device_above_255_is_refused(axpy_launcher):
    """One byte holds the device, so 256 would alias 0 -- a wrong-context launch."""
    x = torch.randn(64, device="cuda")
    stream = torch.cuda.current_stream().cuda_stream
    for device in (256, -1):
        with pytest.raises(TypeError, match=r"\[0, 256\)"):
            axpy_launcher(device, stream, (1,), x, x, x, 64, 1.5, True, None, 128)


def test_dtype_index_table_matches_triton():
    """The compact index must be exactly as fine as triton's specialization.

    Two torch dtypes share an index iff triton canonicalizes them to the same
    type -- torch.bool, torch.uint1 and torch.int1 are all `u1`.  Finer would
    cost redundant cache entries; coarser would launch the wrong kernel.
    """
    from triton._utils import type_canonicalisation_dict

    table = dtype_index_table()
    assert len(table) == NDTYPES

    canonical: dict[int, str] = {}
    for name in dir(torch):
        value = getattr(torch, name, None)
        if not isinstance(value, torch.dtype):
            continue
        canon = type_canonicalisation_dict.get(str(value).split(".")[-1])
        index = table[dtype_code(value)]
        if canon is None:
            assert index == 0xFF, f"{value} is not a triton type but got index {index}"
            continue
        assert index < 32, f"{value} got index {index}, which does not fit 5 bits"
        assert canonical.setdefault(index, canon) == canon, (
            f"index {index} maps to both {canonical[index]} and {canon}"
        )

    assert canonical, "no torch dtype canonicalized; the table is empty"
    # the three spellings triton folds into one type must share one index
    # torch's stubs omit uint1 and int1.
    assert table[dtype_code(torch.bool)] == table[dtype_code(getattr(torch, "uint1"))]
    assert table[dtype_code(getattr(torch, "int1"))] == table[dtype_code(torch.bool)]
    # and a dtype triton has never accepted must be refused
    assert table[dtype_code(torch.complex64)] == 0xFF


def test_dtype_index_table_is_deterministic():
    """Same torch, same triton, same table -- a module keys on it."""
    dtype_index_table.cache_clear()
    first = dtype_index_table()
    dtype_index_table.cache_clear()
    assert dtype_index_table() == first


def test_layout_probe_reproduces_torch():
    """Every offset RUNTIME_SHIM reads, checked against torch's own accessors."""
    layout = layout_for()
    assert layout is not None, "probe failed on a torch intj is expected to support"
    assert len(layout.itemsize) == NDTYPES
    for x in _read_corpus() + [torch.arange(9, dtype=torch.float64)[2:]]:
        impl = ctypes.c_size_t.from_address(
            id(x) + cpython_abi.pyobject_size() + layout.cdata
        ).value
        assert impl == x._cdata
        assert ctypes.c_int64.from_address(impl + layout.numel).value == x.numel()
        assert (
            ctypes.c_int64.from_address(impl + layout.storage_offset).value
            == x.storage_offset()
        )
        assert ctypes.c_uint8.from_address(impl + layout.data_type).value == dtype_code(
            x.dtype
        )


def test_launch_correctness_per_mode(scale_by_mode):
    _, launcher = scale_by_mode
    x = torch.randn(1024, device="cuda")
    o = torch.empty(1024, device="cuda")
    check_matches_triton(scale, launcher, (8,), (x, o, 1024, 2.0, 128), 1)
    check_matches_triton(scale, launcher, (8,), (x[1:], o, 1023, 2.0, 128), 1)


def test_default_resolves_and_explicit_modes_validate(monkeypatch):
    from intj import launcher as launcher_mod

    resolve = launcher_mod._resolve_torch_access_mode
    monkeypatch.setattr(launcher_mod, "cxx_lib_dirs", lambda: ("/lib",))
    monkeypatch.setattr(launcher_mod, "layout_for", lambda: object())
    for m in ACCESS_MODES:
        assert resolve(m) is m
    assert resolve(None) is TorchAccessMode.STATIC_COMPILE
    # `confirmed` asks cpp_extension itself, which here cannot import
    monkeypatch.setattr(launcher_mod, "cxx_toolchain", lambda: None)
    with pytest.raises(UnsupportedKernel, match="STATIC_COMPILE needs"):
        resolve(TorchAccessMode.STATIC_COMPILE, confirmed=True)
    assert resolve(None, confirmed=True) is TorchAccessMode.RUNTIME_SHIM
    monkeypatch.setattr(launcher_mod, "cxx_lib_dirs", lambda: None)
    with pytest.raises(UnsupportedKernel, match="STATIC_COMPILE needs"):
        resolve(TorchAccessMode.STATIC_COMPILE)
    assert resolve(None) is TorchAccessMode.RUNTIME_SHIM
    monkeypatch.setattr(launcher_mod, "layout_for", lambda: None)
    assert resolve(None) is TorchAccessMode.INTERPRETER


@pytest.mark.parametrize(
    "mode", [None, TorchAccessMode.STATIC_COMPILE], ids=["auto", "static"]
)
def test_static_compile_cpp_extension_cannot_build_chooses_again(
    tmp_path, monkeypatch, mode
):
    """Where cpp_extension cannot import, a build chooses as `check()` used to.

    `check()` picks STATIC_COMPILE without importing it (`cxx_lib_dirs`), so a
    cpp_extension that is installed but fails to import (setuptools under
    `SETUPTOOLS_USE_DISTUTILS=stdlib` on 3.12) says no only when a build asks:
    the automatic choice moves on, an explicit request is refused.
    """
    from intj import launcher as launcher_mod
    from intj.render import build

    try:
        build.compiler_path("c++")  # the c++ key reads its identity
    except Exception:
        pytest.skip("no C++ compiler")
    missing = (str(tmp_path / "lib"),)  # no `.so` on disk has this rpath
    for module in (launcher_mod, build):
        monkeypatch.setattr(module, "cxx_lib_dirs", lambda: missing)
        monkeypatch.setattr(module, "cxx_toolchain", lambda: None)
    monkeypatch.setattr(launcher_mod, "_LOADED", {})
    kernel = _kernel_from_source(tmp_path, "unconfirmed_kernel", "x, n")
    launch = make_launcher(kernel, no_gpu=True, torch_access_mode=mode)
    launch.__self__.check()
    if mode is not None:
        with pytest.raises(UnsupportedKernel, match="STATIC_COMPILE needs") as refused:
            launch(0, 0, (1,), 0, 1)
        assert isinstance(refused.value.__cause__, build.CppExtensionUnavailable)
        assert not launcher_mod._LOADED
        return
    launch(0, 0, (1,), 0, 1)
    (key,) = launcher_mod._LOADED
    assert launcher_mod._LOADED[key] is module_of(launch)
    expected = (
        TorchAccessMode.RUNTIME_SHIM
        if launcher_mod.layout_for() is not None
        else TorchAccessMode.INTERPRETER
    )
    assert key.torch_access_mode == expected.value


def test_static_compile_on_disk_loads_without_asking_cpp_extension(
    tmp_path, monkeypatch
):
    """A c++ module already built loads even where cpp_extension now fails."""
    from intj import launcher as launcher_mod
    from intj.render import build

    if launcher_mod.cxx_lib_dirs() is None:  # also caches the answer
        pytest.skip("no C++ compiler or torch headers")
    kernel = _kernel_from_source(tmp_path, "on_disk_kernel", "x, n")
    built = make_launcher(
        kernel, no_gpu=True, torch_access_mode=TorchAccessMode.STATIC_COMPILE
    )
    built(0, 0, (1,), 0, 1)

    def unavailable():
        raise AssertionError("asked cpp_extension for a module already on disk")

    for module in (launcher_mod, build):
        monkeypatch.setattr(module, "cxx_toolchain", unavailable)
    monkeypatch.setattr(launcher_mod, "_LOADED", {})
    for mode in (None, TorchAccessMode.STATIC_COMPILE):
        launch = make_launcher(kernel, no_gpu=True, torch_access_mode=mode)
        launch(0, 0, (1,), 0, 1)
        assert module_of(launch).__file__ == module_of(built).__file__


def test_warm_first_call_skips_unneeded_imports(tmp_path):
    """A first call whose `.so` is on disk imports neither of two costly modules.

    `torch.utils.cpp_extension` (~55 ms) only a build reads: the mode and the
    digest are decided without it.  `triton.runtime.interpreter` (~4 ms) only
    `@triton.jit` under `TRITON_INTERPRET=1` needs.  The module is built here,
    so the one fresh process (`import torch` alone can take seconds) only loads
    it, for an automatic and an explicit request.
    """
    import importlib.util

    from intj.render.build import cxx_toolchain

    if cxx_toolchain() is None or importlib.util.find_spec("setuptools") is None:
        pytest.skip("no C++ compiler, torch headers or setuptools")
    kernel = _kernel_from_source(tmp_path, "warm_kernel", "x, n")
    built = make_launcher(
        kernel, no_gpu=True, torch_access_mode=TorchAccessMode.STATIC_COMPILE
    )
    built(0, 0, (1,), 0, 1)
    path = module_of(built).__file__
    assert path is not None
    assert pathlib.Path(path).with_name("warm_kernel.cpp").exists()
    script = """
import importlib.util
import sys
from intj import TorchAccessMode, make_launcher
from intj.launcher import module_of

spec = importlib.util.spec_from_file_location("warm_kernel", sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
for mode in (None, TorchAccessMode.STATIC_COMPILE):
    launch = make_launcher(module.warm_kernel, no_gpu=True, torch_access_mode=mode)
    launch(0, 0, (1,), 0, 1)
    print(module_of(launch).__file__)
print("torch.utils.cpp_extension" in sys.modules)
print("triton.runtime.interpreter" in sys.modules)
"""
    run = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path / "warm_kernel.py")],
        cwd=tmp_path,
        env=_tested_intj_env(),
        capture_output=True,
        text=True,
    )
    assert run.returncode == 0, run.stderr
    assert run.stdout.split() == [path, path, "False", "False"]


def test_public_torch_access_modes():
    assert intj.TorchAccessMode is TorchAccessMode
    assert list(TorchAccessMode) == [
        TorchAccessMode.INTERPRETER,
        TorchAccessMode.RUNTIME_SHIM,
        TorchAccessMode.STATIC_COMPILE,
    ]


def test_hardcoded_layout_matches_this_torch():
    """The table is data, and data goes stale.  Check the entry against reality.

    `probe_layout` finds each offset by matching field values against torch's own
    accessors, so it is an independent oracle for the hardcoded row -- and the
    thing that generates entries in the first place (`python -m intj.torch_intf.abi_detect`).
    """
    table, probed = layout_for(), probe_layout(cpython_abi.pyobject_size())
    assert table is not None, "no entry for the torch the suite is running against"
    assert probed is not None, "probe could not pin this torch's layout"
    assert table == probed


def test_hardcoded_layout_matches_torch_headers():
    """The same entry, checked by the other detector: the compiler places each field.

    Element sizes are compared by code only; the spellings are python's, and the
    headers know none of them.  Codes the headers declare past the recorded ones
    must be sizeless placeholders, or the table is missing a dtype.
    """
    from intj.render.build import cxx_toolchain
    from intj.torch_intf.cpp_detect import detect
    from intj.torch_intf.torch_abi import OFFSETS

    if cxx_toolchain() is None:
        pytest.skip("no C++ compiler and torch headers")
    table = layout_for()
    assert table is not None, "no entry for the torch the suite is running against"
    offsets, sizes = detect()
    assert offsets == {f: getattr(table, f) for f in OFFSETS}
    assert {c: s for c, s in sizes.items() if s} == {
        c: table.itemsize[c] for c in range(NDTYPES) if table.itemsize[c]
    }


def test_runtime_shim_is_refused_rather_than_guessed(monkeypatch):
    """An unverified torch must raise, never fall back to a guessed offset."""
    from intj import launcher as launcher_mod

    monkeypatch.setattr(launcher_mod, "layout_for", lambda *a: None)
    with pytest.raises(UnsupportedKernel, match="tensor layout"):
        make_launcher(
            scale, torch_access_mode=TorchAccessMode.RUNTIME_SHIM
        ).__self__.check()


def test_static_compile_needs_no_table(monkeypatch):
    """The compiler supplies offsets, even for an unverified torch."""
    from intj import launcher as launcher_mod

    monkeypatch.setattr(launcher_mod, "layout_for", lambda *a: None)
    x = torch.arange(128, device="cuda", dtype=torch.float32)
    o = torch.empty_like(x)
    check_matches_triton(
        scale,
        make_launcher(scale, torch_access_mode=TorchAccessMode.STATIC_COMPILE),
        (1,),
        (x, o, 128, 2.0, 128),
        1,
    )


def test_only_static_compile_module_is_keyed_on_the_torch_version(monkeypatch):
    """RUNTIME_SHIM and INTERPRETER bake in no Torch version.

    Their `.so` is reusable across Torch versions, while STATIC_COMPILE compiles
    against Torch's headers and must get a different digest.
    """
    from intj import launcher as launcher_mod

    def digest_dir(m):
        return pathlib.Path(
            str(module_of(make_launcher(scale, torch_access_mode=m)).__file__)
        ).parent.parent.name

    before = {m: digest_dir(m) for m in ACCESS_MODES}
    monkeypatch.setattr(launcher_mod, "torch_version", lambda: (99, 99))
    after = {m: digest_dir(m) for m in ACCESS_MODES}

    assert after[TorchAccessMode.RUNTIME_SHIM] == before[TorchAccessMode.RUNTIME_SHIM]
    assert after[TorchAccessMode.INTERPRETER] == before[TorchAccessMode.INTERPRETER]
    assert (
        after[TorchAccessMode.STATIC_COMPILE] != before[TorchAccessMode.STATIC_COMPILE]
    )


def test_unconfigured_module_refuses_to_launch():
    """`entry` is unreachable before _set_torch_version; prove the guard exists."""
    module = module_of(
        make_launcher(scale, torch_access_mode=TorchAccessMode.RUNTIME_SHIM)
    )
    assert hasattr(module, "_set_torch_version")
    index = dtype_index_table()
    with pytest.raises(ValueError, match="needs a tensor layout"):
        module._set_torch_version((2, 14), None, index)
    layout = layout_for()
    assert layout is not None
    interpreter = module_of(
        make_launcher(scale, torch_access_mode=TorchAccessMode.INTERPRETER)
    )
    with pytest.raises(ValueError, match="takes no tensor layout"):
        interpreter._set_torch_version((2, 14), layout.as_args(), index)


def test_a_failed_set_torch_version_changes_nothing():
    """A rejected call must not leave a half-installed table behind.

    The method is reachable from python on a module that is already configured
    and whose kernel cache already holds entries.  Installing the dtype table
    before validating the rest meant a call that raised still committed it --
    and an all-zero table keys every dtype to index 0, so a float16 binary runs
    on float32 memory with no error at all.  Silent, and the worst failure this
    design has.
    """
    launch = make_launcher(scale, torch_access_mode=TorchAccessMode.RUNTIME_SHIM)
    module = module_of(launch)
    o = torch.empty(64, device="cuda")

    def keys():
        return {
            d: launch.__self__.spec_key(
                0, torch.zeros(64, device="cuda", dtype=d), o, 64, 2.0, 128
            )
            for d in (torch.float32, torch.float16)
        }

    before = keys()
    assert len(set(before.values())) == 2, "two dtypes must key apart to begin with"

    good = layout_for()
    assert good is not None
    index = dtype_index_table()
    for layout, table in (
        (None, bytes(NDTYPES)),  # RUNTIME_SHIM rejects a None layout...
        (None, index),  # ...whichever table comes with it
        (good.as_args(), b"\xff" * (NDTYPES - 1)),  # wrong length
        (good.as_args(), bytes([32]) + index[1:]),  # an index past five bits
    ):
        with pytest.raises(ValueError):
            module._set_torch_version((2, 14), layout, table)
        # per case, not once at the end: each rejection has its own way of
        # committing early, and one masks the next
        assert keys() == before


def test_set_torch_version_needs_the_dtype_index_in_every_mode():
    """The dtype table is not a RUNTIME_SHIM detail: all three readers key on it.

    Installed in only some modes, the others would see an all-zero table and key
    every dtype to index 0 -- coarse in the same way in each, so the cross-mode
    equality test would not catch it.
    """
    layout = layout_for()
    assert layout is not None
    for mode in ACCESS_MODES:
        module = module_of(make_launcher(scale, torch_access_mode=mode))
        args = layout.as_args() if mode is TorchAccessMode.RUNTIME_SHIM else None
        with pytest.raises(TypeError, match="dtype_index"):
            module._set_torch_version((2, 14), args)
        with pytest.raises(ValueError, match="dtype index table"):
            module._set_torch_version((2, 14), args, b"\xff" * (NDTYPES - 1))


def test_modes_agree_on_rejecting_a_storageless_tensor():
    """A sparse tensor has no data pointer at all, so every mode must raise.

    RUNTIME_SHIM reads offsets rather than calling torch, so this is the one place it
    could have handed the kernel a null pointer instead of failing.
    """
    sparse = torch.sparse_coo_tensor(
        torch.tensor([[0, 1]]), torch.tensor([1.0, 2.0]), (4,)
    ).cuda()
    o = torch.empty(1024, device="cuda")
    for m in ACCESS_MODES:
        launch = make_launcher(scale, torch_access_mode=m)
        module = module_of(launch)
        with pytest.raises(RuntimeError):
            launch.__self__.spec_key(0, sparse, o, 1024, 2.0, 128)


def test_modes_agree_on_tensors_with_unusual_impls():
    """RUNTIME_SHIM cannot see the bitfields that make torch's accessors throw.

    `storage_access_should_throw_`, `throw_on_immutable_data_ptr_` and
    `size_bytes_is_heap_allocated_` are invisible to offset arithmetic, so the
    RUNTIME_SHIM could accept a tensor STATIC_COMPILE rejects. In practice the
    impls carrying them belong to `Tensor` *subclasses*, which INTJ_DECODE's
    exact-type test already refuses -- this pins that reasoning.
    """
    launchers = {m: make_launcher(scale, torch_access_mode=m) for m in ACCESS_MODES}
    modules = {m: (module_of(k), k) for m, k in launchers.items()}
    o = torch.empty(1024, device="cuda")

    with torch.inference_mode():
        inference = torch.randn(1024, device="cuda")
    cases = {"inference": inference, "meta": torch.zeros(1024, device="meta")}

    for label, t in cases.items():
        keys = {
            m: k.__self__.spec_key(0, t, o, 1024, 2.0, 128)
            for m, (mod, k) in modules.items()
        }
        assert len(set(keys.values())) == 1, f"{label}: {keys}"

    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode() as fake_mode:
        fake = fake_mode.from_tensor(torch.randn(1024, device="cuda"))
    for m, (mod, k) in modules.items():
        with pytest.raises(TypeError):  # a subclass: refused before any read
            k.__self__.spec_key(0, fake, o, 1024, 2.0, 128)


def test_custom_sizes_policy_tensors_are_a_known_limitation():
    """Nested tensors read as empty, identically in every mode.  Documented, not fixed.

    A nested tensor stores `numel_ == 0` and reports `numel() == 8` through a
    virtual override. RUNTIME_SHIM cannot see the policy bit that says so, and
    STATIC_COMPILE reads the field too rather than diverge from it -- one documented limitation
    beats two modes that disagree.  See `docs/Usage.md`.

    This pins the behaviour so that a torch change, or a decision to start
    detecting these, shows up here instead of silently.
    """
    from intj.torch_intf.abi_detect import _read

    layout = layout_for()
    assert layout is not None
    nested = torch.nested.nested_tensor([torch.randn(3), torch.randn(5)])
    assert nested.numel() == 8 and nested.data_ptr() != 0
    assert _read(layout, nested, cpython_abi.pyobject_size())[0] == 0, (
        "expected the known-wrong null pointer"
    )

    # mkldnn has no storage at all, and that *is* detected
    mkl = torch.randn(4, 4).to_mkldnn()
    with pytest.raises(RuntimeError):
        _read(layout, mkl, cpython_abi.pyobject_size())
    o = torch.empty(16, device="cuda")
    for m in ACCESS_MODES:
        launch = make_launcher(scale, torch_access_mode=m)
        module = module_of(launch)
        with pytest.raises(RuntimeError):
            launch.__self__.spec_key(0, mkl, o, 16, 2.0, 128)


def test_every_launcher_shares_one_fixed_size_object(bound_kernel, device_kernel):
    """Every launcher object is one fixed-size type; its state lives behind `state`."""
    from intj.launcher import _stub

    pointer = Argument(
        type=tl.pointer_type(tl.float32), specialize=NEVER, bind_value=BindValue.POINTER
    )
    two = make_launcher(
        bound_kernel,
        extra_annotation={"x": pointer, "p": pointer},
        no_gpu=True,
        torch_access_mode=TorchAccessMode.INTERPRETER,
    ).__self__.bind(x=0, p=0)
    none = make_launcher(
        device_kernel,
        no_gpu=True,
        torch_access_mode=TorchAccessMode.INTERPRETER,
    ).__self__.bind_device(0)
    assert two(0, 0, 1, 7) is None and none(0, 1, 7) is None
    t2, t0 = type(two.__self__), type(none.__self__)
    assert t2 is t0 and issubclass(t2, _stub().Launcher)
    assert t2.__basicsize__ == _stub()._header_size()
    assert t2.__itemsize__ == 0


def test_sibling_launchers_compile_once(monkeypatch):
    """Each launcher owns its cache; the module's compile cache shares kernels."""
    import intj.launcher as launcher

    monkeypatch.setattr(
        launcher, "_LOADED", {}
    )  # a fresh module: an empty compile cache
    real, compiles = launcher._checked_compile, []

    def counted(*args, **kwargs):
        compiles.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(launcher, "_checked_compile", counted)
    first, second = make_launcher(scale), make_launcher(scale)
    assert first.__self__ is not second.__self__
    x = torch.ones(64, device="cuda")
    o = torch.zeros_like(x)
    device, stream = (
        torch.cuda.current_device(),
        torch.cuda.current_stream().cuda_stream,
    )
    first(device, stream, 1, x, o, 64, 2.0, 64)
    second(device, stream, 1, x, o, 64, 3.0, 64)  # its own miss: a record, no compile
    torch.cuda.synchronize()
    assert o[0].item() == 3.0
    assert module_of(first) is module_of(second) and len(compiles) == 1


def test_decoration_needs_no_gpu(monkeypatch):
    """Decoration, the decorator form, bind_device and check() never query a target."""
    import intj.launcher as launcher

    def forbidden(*args, **kwargs):
        raise AssertionError("decoration touched the GPU")

    with monkeypatch.context() as patch:
        patch.setattr(launcher, "_current_target", forbidden)
        patch.setattr(launcher, "_current_device", forbidden)
        plain = make_launcher(grid_arg=1)(scale)
        fixed = make_launcher(scale).__self__.bind_device(torch.cuda.current_device())
        plain.__self__.check()
        fixed.__self__.check()  # check() is GPU-free too
    x = torch.ones(64, device="cuda")
    o = torch.zeros_like(x)
    device, stream = (
        torch.cuda.current_device(),
        torch.cuda.current_stream().cuda_stream,
    )
    plain(device, stream, 1, x, o, 64, 2.0, 64)
    fixed(stream, (1,), x, o, 64, 3.0, 64)
    torch.cuda.synchronize()
    assert o[0].item() == 3.0


def test_failed_first_call_leaves_the_launcher_unbuilt(scalar_kernel, monkeypatch):
    import intj.launcher as launcher

    real, calls = launcher._materialize_module, []

    def flaky(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise UnsupportedKernel("intj: transient")
        return real(*args, **kwargs)

    monkeypatch.setattr(launcher, "_materialize_module", flaky)
    launch = make_launcher(
        scalar_kernel, no_gpu=True, torch_access_mode=TorchAccessMode.INTERPRETER
    )
    with pytest.raises(UnsupportedKernel, match="transient"):
        launch(0, 0, 1, 7)
    assert launch(0, 0, 1, 7) is None
    assert len(calls) == 2


def test_dynamic_num_warps_keys_separate_records():
    launch = make_launcher(
        act_store, dynamic_options=("num_warps",), return_compiled=True
    )
    o = torch.zeros(1, device="cuda")
    two = launch(*_gpu_controls(), 2, o, 1.0, "pos")
    four = launch(*_gpu_controls(), 4, o, 1.0, "pos")
    assert two.metadata.num_warps == 2 and four.metadata.num_warps == 4
    assert launch(*_gpu_controls(), 2, o, 1.0, "pos") is two
    assert tuple(inspect.signature(launch).parameters) == (
        "device",
        "stream",
        "grid",
        "num_warps",
        "o",
        "x",
        "ACT",
    )
    # Triton's own `assert` in parse_options (not a power of two), raised on
    # this miss; like Triton's, it vanishes under `python -O`
    with pytest.raises(AssertionError):
        launch(*_gpu_controls(), 3, o, 1.0, "pos")


def test_declared_knob_stays_out_of_the_render_key(monkeypatch):
    """The live value of a declared knob is not read into the RenderKey.

    So flipping it before the first call builds no second module.
    """
    from triton import knobs

    def build():
        launch = make_launcher(act_store, dynamic_options=("knobs.runtime.debug",))
        return module_of(launch)

    first = build()
    monkeypatch.setattr(knobs.runtime, "debug", not knobs.runtime.debug)
    assert build() is first


def test_declared_knob_only_keys_and_must_match_the_live_value(monkeypatch):
    """A declared knob only keys records, and a miss must match its live value.

    intj never sets a declared knob.  A miss whose value differs from the live
    knob raises and records nothing; each live value compiles its own record,
    and an old value still hits its record after the knob changed.
    """
    import intj.launcher as launcher
    from triton import knobs

    monkeypatch.setattr(knobs.runtime, "debug", False)
    compiles, real = [], launcher._checked_compile

    def spy(*args, **kwargs):
        compiles.append(knobs.runtime.debug)
        return real(*args, **kwargs)

    monkeypatch.setattr(launcher, "_checked_compile", spy)
    launch = make_launcher(
        act_store,
        dynamic_options=("knobs.runtime.debug",),
        options={"num_stages": 1},  # a module of its own: every value compiles here
        return_compiled=True,
    )
    o = torch.zeros(1, device="cuda")
    with pytest.raises(
        ValueError,
        match="dynamic knob 'knobs.runtime.debug' passed True but the current value is False",
    ):
        launch(*_gpu_controls(), True, o, 1.0, "pos")
    assert knobs.runtime.debug is False and compiles == []
    off = launch(*_gpu_controls(), False, o, 1.0, "pos")
    assert compiles == [False] and knobs.runtime.debug is False
    monkeypatch.setattr(knobs.runtime, "debug", True)
    on = launch(*_gpu_controls(), True, o, 1.0, "pos")  # the refused key, now live
    assert compiles == [False, True] and knobs.runtime.debug is True
    assert on is not off and on.metadata.debug and not off.metadata.debug
    assert launch(*_gpu_controls(), False, o, 1.0, "pos") is off  # a hit: no check
    assert compiles == [False, True]


def test_declared_knob_without_an_option_compiles_per_value(monkeypatch):
    """A declared knob that feeds no compile option still compiles per value.

    knobs.compilation.disable_line_info feeds neither a compile option nor the
    specialization: only the knob values in the compile identity keep its two
    values apart.
    """
    from triton import knobs

    launch = make_launcher(
        act_store,
        dynamic_options=("knobs.compilation.disable_line_info",),
        return_compiled=True,
    )
    o = torch.zeros(1, device="cuda")
    monkeypatch.setattr(knobs.compilation, "disable_line_info", False)
    lines = launch(*_gpu_controls(), False, o, 1.0, "pos")
    monkeypatch.setattr(knobs.compilation, "disable_line_info", True)
    bare = launch(*_gpu_controls(), True, o, 1.0, "pos")
    assert launch(*_gpu_controls(), True, o, 1.0, "pos") is bare
    asm = "amdgcn" if "amdgcn" in lines.asm else "ptx"
    assert ".loc" in lines.asm[asm] and ".loc" not in bare.asm[asm]


@needs_hip  # knobs.amd.use_buffer_ops is a HIP knob
def test_declared_knob_reaches_the_specialization(monkeypatch):
    """A declared knob reaches the specialization's compiler input.

    HIP's pointer specialization reads knobs.amd.use_buffer_ops: with the live
    knob matching the passed value, the compiler input follows it.
    """
    from triton import knobs

    launch = make_launcher(
        act_store, dynamic_options=("knobs.amd.use_buffer_ops",), return_compiled=True
    )
    o = torch.zeros(1, device="cuda")
    monkeypatch.setattr(knobs.amd, "use_buffer_ops", True)
    on = launch(*_gpu_controls(), True, o, 1.0, "pos")
    monkeypatch.setattr(knobs.amd, "use_buffer_ops", False)
    off = launch(*_gpu_controls(), False, o, 1.0, "pos")
    assert "tt.pointer_range" in str(on.src.attrs)
    assert "tt.pointer_range" not in str(off.src.attrs)
    assert launch(*_gpu_controls(), False, o, 1.0, "pos") is off


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"dynamic_options": ("knobs.nope.x",)}, "unknown knob"),
        ({"dynamic_options": ("knobs.runtime.nope",)}, "unknown knob"),
        ({"dynamic_options": ("num_warps",), "options": {"num_warps": 4}}, "both"),
        (
            {"dynamic_options": ("knobs.runtime.debug",), "options": {"debug": True}},
            "both",
        ),
        ({"dynamic_options": ("num_warps", "num_warps")}, "repeats"),
        ({"dynamic_options": ("x",)}, "kernel parameter"),
        ({"dynamic_options": ("stream",)}, "not allowed"),
        ({"dynamic_options": ("num_warps",), "no_gpu": True}, "GPU"),
        # final review minor 2: a knob overwrites these outright, so a
        # per-call value would be keyed but never reach the compile
        (
            {"dynamic_options": ("instrumentation_mode",)},
            r"overwritten by 'knobs\.compilation\.instrumentation_mode'",
        ),
        (
            {"dynamic_options": ("fpsan_homomorphic_casts",)},
            r"overwritten by 'knobs\.compilation\.fpsan_homomorphic_casts'",
        ),
    ],
)
def test_dynamic_option_refusals(kwargs, match):
    with pytest.raises(UnsupportedKernel, match=match):
        make_launcher(act_store, **kwargs).__self__.check()


def test_unknown_dynamic_option_is_refused_on_the_first_call():
    launch = make_launcher(act_store, dynamic_options=("num_warpz",))
    with pytest.raises(UnsupportedKernel, match="unknown compile option"):
        launch(*_gpu_controls(), 4, torch.zeros(1, device="cuda"), 1.0, "pos")


@triton.jit
def dyn_axpy(
    o, x, n, ACT: tl.constexpr, DT: tl.constexpr, FN: tl.constexpr, BLOCK: tl.constexpr
):
    offs = tl.arange(0, BLOCK)
    v = FN(tl.load(x + offs, mask=offs < n)).to(DT)
    if ACT == "neg":
        v = -v
    tl.store(o + offs, v, mask=offs < n)


_DYNAMIC = (
    "num_warps",
    "knobs.runtime.debug",
    "knobs.compilation.disable_line_info",
) + (("knobs.amd.use_buffer_ops",) if ON_HIP else ())


def check_dynamic_invariant():
    """Same key => same annotated ASTSource, same canonical options, same knobs."""
    import itertools

    from triton.compiler import make_backend

    from intj.annotation import DeviceBinding, _resolve_annotations
    from intj.launcher import (
        _canonical_options,
        _compiler_input,
        _current_target,
        _knob_options,
        _live_knobs,
        _render_params,
        _split_dynamic,
    )

    launch = make_launcher(dyn_axpy, dynamic_options=_DYNAMIC)
    module, device = module_of(launch), torch.cuda.current_device()
    params, _, _ = _render_params(
        _resolve_annotations(dyn_axpy, None), DeviceBinding.NOT_FIXED
    )
    target = _current_target()
    backend, knobs_now = make_backend(target), _live_knobs()
    x = torch.randn(64, device="cuda")
    o = torch.empty_like(x)
    seen = {}
    per_value = [(2, 4)] + [(False, True)] * (len(_DYNAMIC) - 1)
    for dyn in itertools.product(*per_value):
        options, dyn_knobs = _split_dynamic(_DYNAMIC, dyn)
        knob_values = {**knobs_now, **dyn_knobs}
        with live_knob_values(dyn_knobs):
            canonical = _canonical_options(
                target, _knob_options(dyn_axpy, options, knob_values)
            ).hash()
            for act in ("neg", "".join(["n", "e", "g"]), "pos"):
                for dt in (tl.float32, tl.float16):
                    for fn in (_twice, _thrice, triton.jit(_twice.fn)):
                        public = (o, x, 64, act, dt, fn, 64)
                        key, _ = launch.__self__.spec_key(device, *dyn, *public)
                        truth = (
                            _compiler_input(dyn_axpy, params, public, backend),
                            canonical,
                            tuple(sorted(knob_values.items())),
                        )
                        assert seen.setdefault(key, truth) == truth, (
                            "intj key collides for dynamic values"
                        )


def test_dynamic_values_are_never_coarser_than_triton():
    check_dynamic_invariant()


@pytest.mark.parametrize("dropped", _DYNAMIC)
def test_dynamic_invariant_catches_a_dropped_value(monkeypatch, dropped):
    original = launcher._dynamic_key_fields
    monkeypatch.setattr(
        launcher,
        "_dynamic_key_fields",
        lambda name: () if name == dropped else original(name),
    )
    with pytest.raises(AssertionError, match="collides"):
        check_dynamic_invariant()


def test_dynamic_invariant_catches_a_dropped_object_id(monkeypatch):
    monkeypatch.setattr(launcher._Interner, "__call__", lambda self, value: 0)
    with pytest.raises(AssertionError, match="collides"):
        check_dynamic_invariant()


@triton.jit
def warps_store(o, num_warps: tl.constexpr, SCALE: tl.constexpr):
    tl.store(o + tl.arange(0, 64), tl.full([64], num_warps * SCALE, tl.int32))


@pytest.mark.parametrize("declared", [(), ("num_warps",)], ids=["plain", "declared"])
def test_constexpr_option_parameter_feeds_the_option(declared):
    """A `num_warps: tl.constexpr` parameter is also the compile option.

    As Triton's `k[grid](o, num_warps=8)` makes it: one record per value, each
    compiled with that many warps.  Declaring it in dynamic_options adds
    nothing.
    """
    launch = make_launcher(warps_store, dynamic_options=declared, return_compiled=True)
    launch.__self__.check()  # the signature is set by check()
    assert tuple(inspect.signature(launch).parameters)[3:] == (
        "o",
        "num_warps",
        "SCALE",
    )
    o = torch.zeros(64, device="cuda", dtype=torch.int32)
    records = {}
    for warps in (2, 4, 8):
        records[warps] = launch(*_gpu_controls(), o, warps, 1)
        assert records[warps].metadata.num_warps == warps
        torch.cuda.synchronize()
        assert (o == warps).all()
    assert len({id(r) for r in records.values()}) == 3
    assert launch(*_gpu_controls(), o, 2, 1) is records[2]


def test_baked_constexpr_option_parameter_feeds_its_value():
    launch = make_launcher(
        warps_store,
        extra_annotation={"num_warps": Constexpr(value=8)},
        return_compiled=True,
    )
    o = torch.zeros(64, device="cuda", dtype=torch.int32)
    assert launch(*_gpu_controls(), o, 3).metadata.num_warps == 8
    torch.cuda.synchronize()
    assert (o == 24).all()


def test_static_option_naming_a_constexpr_parameter_is_refused():
    with pytest.raises(UnsupportedKernel, match=r"options \['num_warps'\] name"):
        make_launcher(warps_store, options={"num_warps": 8}).__self__.check()


def test_dynamic_constexpr_parameter_that_is_no_option_is_refused():
    launch = make_launcher(warps_store, dynamic_options=("SCALE",))
    o = torch.zeros(64, device="cuda", dtype=torch.int32)
    with pytest.raises(
        UnsupportedKernel, match=r"unknown compile option\(s\) \['SCALE'\]"
    ):
        launch(*_gpu_controls(), o, 4, 1)


@triton.jit
def option_axpy(
    o, x, n, num_warps: tl.constexpr, num_stages: tl.constexpr, BLOCK: tl.constexpr
):
    offs = tl.arange(0, BLOCK)
    tl.store(o + offs, tl.load(x + offs, mask=offs < n) * num_stages, mask=offs < n)


def check_option_parameter_invariant():
    """Asserts that option parameters keep the key no coarser than Triton's.

    Same key => same annotated ASTSource and same canonical options, the
    options being what Triton's keyword call makes of the parameters; and the
    launcher really compiles with them.
    """
    from triton.compiler import make_backend

    from intj.annotation import DeviceBinding, _resolve_annotations
    from intj.launcher import (
        _canonical_options,
        _compiler_input,
        _current_target,
        _knob_options,
        _live_knobs,
        _render_params,
    )

    launch = make_launcher(option_axpy, return_compiled=True)
    module, device = module_of(launch), torch.cuda.current_device()
    params, _, _ = _render_params(
        _resolve_annotations(option_axpy, None), DeviceBinding.NOT_FIXED
    )
    target = _current_target()
    backend, knobs_now = make_backend(target), _live_knobs()
    x = torch.randn(64, device="cuda")
    o = torch.empty_like(x)
    seen = {}
    for warps, stages, block in itertools.product((2, 4), (1, 2), (64, 128)):
        public = (o, x, 64, warps, stages, block)
        options = {"num_warps": warps, "num_stages": stages}
        key, _ = launch.__self__.spec_key(device, *public)
        truth = (
            _compiler_input(option_axpy, params, public, backend),
            _canonical_options(
                target, _knob_options(option_axpy, options, knobs_now)
            ).hash(),
        )
        assert seen.setdefault(key, truth) == truth, "intj key collides"
        md = launch(*_gpu_controls(), *public).metadata
        assert (md.num_warps, md.num_stages) == (warps, stages), "option not fed"


def test_option_parameters_are_never_coarser_than_triton():
    check_option_parameter_invariant()


def test_option_parameter_invariant_catches_an_unfed_option(monkeypatch):
    # the key is unchanged, so a fresh module table keeps the loaded module's
    # compile function (built unmutated) out of the way
    monkeypatch.setattr(launcher, "_LOADED", {})
    monkeypatch.setattr(launcher, "_parameter_options", lambda backend, params: ())
    with pytest.raises(AssertionError, match="option not fed"):
        check_option_parameter_invariant()

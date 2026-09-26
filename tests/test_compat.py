# pyright: standard
"""Triton-style calls routed through the native launcher."""

from types import SimpleNamespace

import pytest
import torch
import triton
import triton.language as tl

from intj.compat import launch
from intj.launcher import UnsupportedKernel


@triton.jit
def add_one(x, out, n, BLOCK: tl.constexpr = 64):  # pyright: ignore[reportArgumentType]  # Triton permits constexpr defaults
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(out + i, tl.load(x + i, i < n, other=0) + 1, i < n)


@triton.jit
def unused_pointer(x):
    pass


@triton.jit
def cast_through_dtype(x, out, n, DT: tl.constexpr):
    i = tl.program_id(0) * 32 + tl.arange(0, 32)
    value = tl.load(x + i, i < n, other=0).to(DT).to(tl.float32)
    tl.store(out + i, value, i < n)


@triton.jit
def string_mode(x, out, n, MODE: tl.constexpr):
    i = tl.program_id(0) * 32 + tl.arange(0, 32)
    value = tl.load(x + i, i < n, other=0)
    if MODE == "negate":
        value = -value
    tl.store(out + i, value, i < n)


@triton.jit
def add_two(value):
    return value + 2


@triton.jit
def subtract_one(value):
    return value - 1


@triton.jit
def apply_helper(x, out, n, HELPER: tl.constexpr):
    i = tl.program_id(0) * 32 + tl.arange(0, 32)
    value = tl.load(x + i, i < n, other=0)
    tl.store(out + i, HELPER(value), i < n)


@triton.jit
def python_annotated_int(x, out, n: int):
    i = tl.program_id(0) * 32 + tl.arange(0, 32)
    value = tl.load(x + i, i < n, other=0)
    tl.store(out + i, value + 1, i < n)


def test_launch_binds_keywords_default_and_compile_option():
    x = torch.arange(129, device="cuda", dtype=torch.int32)
    out = torch.empty_like(x)
    launch(add_one, (triton.cdiv(x.numel(), 64),), x, n=x.numel(), out=out, num_warps=4)
    torch.testing.assert_close(out, x + 1)


def test_launch_calls_python_grid_with_current_named_values():
    x = torch.arange(65, device="cuda", dtype=torch.int32)
    out = torch.empty_like(x)
    seen = []

    def grid(meta):
        seen.append((meta["n"], meta["BLOCK"]))
        return (triton.cdiv(meta["n"], meta["BLOCK"]),)

    launch(add_one, grid, x, out, x.numel())
    torch.testing.assert_close(out, x + 1)
    assert seen == [(65, 64)]


def test_launch_bakes_dtype_constexpr():
    x = torch.tensor([1.001, -2.003], device="cuda", dtype=torch.float32)
    out = torch.empty_like(x)
    launch(cast_through_dtype, (1,), x, out, x.numel(), DT=tl.float16)
    torch.testing.assert_close(out, x.to(torch.float16).to(torch.float32))
    first = out.clone()
    launch(cast_through_dtype, (1,), x, out, x.numel(), DT=tl.bfloat16)
    torch.testing.assert_close(out, x.to(torch.bfloat16).to(torch.float32))
    assert not torch.equal(first, out)


def test_launch_bakes_string_constexpr():
    x = torch.tensor([2, 3], device="cuda", dtype=torch.int32)
    out = torch.empty_like(x)
    launch(string_mode, (1,), x, out, x.numel(), MODE="negate")
    torch.testing.assert_close(out, -x)
    launch(string_mode, (1,), x, out, x.numel(), MODE="copy")
    torch.testing.assert_close(out, x)


def test_launch_bakes_jit_function_constexpr():
    x = torch.tensor([2, 3], device="cuda", dtype=torch.int32)
    out = torch.empty_like(x)
    launch(apply_helper, (1,), x, out, x.numel(), HELPER=add_two)
    torch.testing.assert_close(out, x + 2)
    launch(apply_helper, (1,), x, out, x.numel(), HELPER=subtract_one)
    torch.testing.assert_close(out, x - 1)


def test_launch_accepts_python_int_annotation_like_triton():
    x = torch.tensor([2.0, 3.0], device="cuda")
    out = torch.empty_like(x)
    launch(python_annotated_int, (1,), x, out, x.numel())
    torch.testing.assert_close(out, x + 1)


def test_launch_refuses_non_jit_targets_without_fallback():
    try:
        launch(object(), (1,))
    except UnsupportedKernel:
        pass
    else:
        raise AssertionError("a non-JIT target was launched")


def test_launch_or_interpret_only_calls_triton_in_interpreter_mode():
    import intj.compat as compat
    from triton import knobs

    grid = lambda meta: (meta["n"],)

    class TritonOnly:
        def __getitem__(self, actual_grid):
            assert actual_grid is grid
            return lambda *args, **kwargs: (args, kwargs)

    kernel = TritonOnly()
    with knobs.runtime.scope():
        knobs.runtime.interpret = True
        assert compat.launch_or_interpret(kernel, grid, 7, n=3) == ((7,), {"n": 3})
        assert compat.launch_or_interpret(
            kernel, grid, 7, n=3, return_compiled=True
        ) == ((7,), {"n": 3})
        knobs.runtime.interpret = False
        with pytest.raises(UnsupportedKernel, match="expected a @triton.jit function"):
            compat.launch_or_interpret(kernel, grid, 7, n=3)


def test_launch_uses_triton_dispatch_during_compile_warmup(monkeypatch):
    import intj.compat as compat

    monkeypatch.setattr(compat, "_is_compile_warmup", lambda: True)
    grid = (1,)

    class TritonOnly:
        def __getitem__(self, actual_grid):
            assert actual_grid is grid
            return lambda *args, **kwargs: (args, kwargs)

    assert compat.launch(TritonOnly(), grid, 7, n=3, return_compiled=True) == (
        (7,),
        {"n": 3},
    )


def test_launch_or_interpret_runs_real_triton_interpreter():
    import intj.compat as compat
    from triton import knobs
    from triton.runtime.interpreter import InterpretedFunction

    with knobs.runtime.scope():
        knobs.runtime.interpret = True

        @triton.jit
        def interpreted_add_one(x, out, n, BLOCK: tl.constexpr):
            i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
            tl.store(out + i, tl.load(x + i, i < n, other=0) + 1, i < n)

        assert isinstance(interpreted_add_one, InterpretedFunction)
        x = torch.arange(32, dtype=torch.int32)
        out = torch.empty_like(x)
        compat.launch_or_interpret(
            interpreted_add_one, (1,), x, out=out, n=x.numel(), BLOCK=32
        )
        torch.testing.assert_close(out, x + 1)


def test_launch_or_interpret_uses_native_launcher_in_compiled_mode():
    import intj.compat as compat
    from triton import knobs

    x = torch.arange(32, device="cuda", dtype=torch.int32)
    out = torch.empty_like(x)
    with knobs.runtime.scope():
        knobs.runtime.interpret = False
        assert (
            compat.launch_or_interpret(add_one, (1,), x, out=out, n=x.numel()) is None
        )
    torch.testing.assert_close(out, x + 1)


def test_launch_returns_cached_compiled_kernel_when_requested():
    x = torch.arange(32, device="cuda", dtype=torch.int32)
    out = torch.empty_like(x)
    assert launch(add_one, (1,), x, out=out, n=x.numel()) is None

    kernel = launch(add_one, (1,), x, out=out, n=x.numel(), return_compiled=True)
    assert kernel.asm["ttir"]
    assert (
        launch(add_one, (1,), x, out=out, n=x.numel(), return_compiled=True) is kernel
    )
    torch.testing.assert_close(out, x + 1)


def test_launch_returns_compiled_kernel_for_python_grid():
    x = torch.arange(32, device="cuda", dtype=torch.int32)
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(meta["n"], meta["BLOCK"]),)

    kernel = launch(add_one, grid, x, out=out, n=x.numel(), return_compiled=True)
    assert kernel.asm["ttir"]
    torch.testing.assert_close(out, x + 1)


def test_tensor_wrapper_pointer_annotation_uses_triton_global_default():
    from intj.compat import _pointer_type

    assert (
        _pointer_type(tl.int32).address_space == tl.pointer_type(tl.int32).address_space
    )


def test_callable_grid_bridge_reuses_kernel_cache(monkeypatch):
    import intj.compat as compat
    import intj.launcher as launcher_module

    from intj import TorchAccessMode, make_launcher

    real_make_launcher = make_launcher

    def host_launcher(kernel, **kwargs):
        return real_make_launcher(
            kernel, no_gpu=True, torch_access_mode=TorchAccessMode.INTERPRETER, **kwargs
        )

    monkeypatch.setattr(compat, "make_launcher", host_launcher)
    monkeypatch.setattr(
        compat,
        "driver",
        SimpleNamespace(
            active=SimpleNamespace(
                get_current_device=lambda: 0,
                get_current_stream=lambda device: 0,
            )
        ),
    )
    monkeypatch.setattr(launcher_module, "_LOADED", {})

    dimensions = [0]

    def grid(meta):
        return (dimensions[0],)

    compat.launch(unused_pointer, grid, 7)
    (module,) = launcher_module._LOADED.values()
    misses = []

    def compile_once(key, nparams, device, *args):
        misses.append(bytes(key))
        return 0, 1, 0, nparams

    module.set_compile_callback(compile_once)
    dimensions[0] = 1
    compat.launch(unused_pointer, grid, 7)
    compat.launch(unused_pointer, grid, 7)
    assert len(misses) == 1


def test_fpsan_knob_change_reselects_compat_launcher(monkeypatch):
    import intj.compat as compat
    from triton import knobs

    selected = []

    def make_native(*_args):
        selected.append(object())
        return lambda *_call_args: None

    monkeypatch.setattr(compat, "_make", make_native)
    monkeypatch.setattr(
        compat,
        "driver",
        SimpleNamespace(
            active=SimpleNamespace(
                get_current_device=lambda: 0,
                get_current_stream=lambda device: 0,
                get_current_target=lambda: SimpleNamespace(
                    backend="hip", arch="gfx942", warp_size=64
                ),
            )
        ),
    )
    compat._cached.cache_clear()
    try:
        monkeypatch.setattr(
            knobs.compilation, "fpsan_homomorphic_casts", False, raising=False
        )
        compat.launch(unused_pointer, (1,), 7)
        monkeypatch.setattr(knobs.compilation, "fpsan_homomorphic_casts", True)
        compat.launch(unused_pointer, (1,), 7)
        assert len(selected) == 2
    finally:
        compat._cached.cache_clear()


@pytest.mark.parametrize("hook_name", ["launch_enter_hook", "launch_exit_hook"])
def test_cached_launch_refuses_active_triton_hook(monkeypatch, hook_name):
    import intj.compat as compat
    from triton import knobs
    from triton.knobs import HookChain

    monkeypatch.setattr(knobs.runtime, "launch_enter_hook", HookChain())
    monkeypatch.setattr(knobs.runtime, "launch_exit_hook", HookChain(reversed=True))
    compat._cached.cache_clear()
    x = torch.arange(32, device="cuda", dtype=torch.int32)
    out = torch.empty_like(x)
    try:
        compat.launch(add_one, (1,), x, out, 32)
        assert compat._cached.cache_info().currsize == 1
        torch.testing.assert_close(out, x + 1)
        getattr(knobs.runtime, hook_name).add(lambda _metadata: None)
        with pytest.raises(UnsupportedKernel, match="launch hook"):
            compat.launch(add_one, (1,), x, out, 32)
    finally:
        compat._cached.cache_clear()


def test_launch_rejects_unpinned_cpu_tensor_before_gpu_driver_lookup():
    x = torch.empty(1024)
    with pytest.raises(ValueError, match="cannot be accessed from Triton"):
        launch(unused_pointer, (1,), x)


@pytest.mark.parametrize(
    "wrap",
    [
        triton.autotune(configs=[triton.Config({})], key=[]),
        triton.heuristics({}),
    ],
)
def test_launch_refuses_decorated_kernel(wrap):
    with pytest.raises(UnsupportedKernel, match="expected a @triton.jit function"):
        launch(wrap(unused_pointer), (1,), torch.empty(1, device="cuda"))

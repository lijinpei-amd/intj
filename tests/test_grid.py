# pyright: standard
"""Callable and explicit-dimension launch grids."""

import builtins
import dataclasses
import gc
import importlib.util
import inspect
import weakref

import pytest
import torch
import triton
import triton.language as tl

from intj import Argument, BindValue, Constexpr, NEVER, TorchAccess, make_launcher
from intj.launcher import UnsupportedKernel


GLOBAL_GRID_OFFSET = 1


@triton.jit
def write_programs(out, X, Y):
    x = tl.program_id(0)
    y = tl.program_id(1)
    z = tl.program_id(2)
    tl.store(out + x + X * (y + Y * z), 1)


@pytest.mark.parametrize("dimensions,shape", [(1, (3,)), (2, (3, 2)), (3, (3, 2, 4))])
def test_grid_arg_passes_each_requested_dimension(dimensions, shape):
    launcher = make_launcher(write_programs, grid_arg=dimensions)
    out = torch.zeros(24, device="cuda", dtype=torch.int32)
    launcher(torch.cuda.current_device(), torch.cuda.current_stream().cuda_stream,
             *shape, out, 3, 2)
    torch.cuda.synchronize()
    count = 3 if dimensions == 1 else 6 if dimensions == 2 else 24
    torch.testing.assert_close(out[:count], torch.ones_like(out[:count]))
    torch.testing.assert_close(out[count:], torch.zeros_like(out[count:]))


def test_grid_arg_zero_skips_launch_and_argument_decode():
    launcher = make_launcher(write_programs, grid_arg=2, no_gpu=True)
    assert launcher(0, 0, 0, 3, object(), object(), object()) is None
    fixed = make_launcher(write_programs, grid_arg=3, bind_device=True,
                          no_gpu=True).bind_device(0)
    assert tuple(inspect.signature(fixed).parameters) == (
        "stream", "grid_x", "grid_y", "grid_z", "out", "X", "Y"
    )
    assert fixed(0, 1, 0, 3, object(), object(), object()) is None


def test_grid_modes_validate_options_and_keep_separate_modules():
    assert callable(make_launcher(write_programs, False, (), no_gpu=True))
    for value in (0, 4, True, 1.0, "2"):
        with pytest.raises(ValueError, match="grid_arg must be 1, 2, or 3"):
            make_launcher(write_programs, grid_arg=value, no_gpu=True)  # pyright: ignore[reportArgumentType]
    for modes in (
        {"grid_arg": 1, "grid_cpp": compiled_grid},
        {"grid_arg": 2, "grid_py": lambda meta: (1,)},
        {"grid_cpp": compiled_grid, "grid_py": lambda meta: (1,)},
    ):
        with pytest.raises(ValueError, match="choose only one"):
            make_launcher(write_programs, no_gpu=True, **modes)
    with pytest.raises(TypeError, match="grid_py must be callable"):
        make_launcher(write_programs, grid_py=3, no_gpu=True)  # pyright: ignore[reportArgumentType]

    def first_grid(X: int):
        return (1,)

    def second_grid(X: int):
        return (2,)

    first = make_launcher(write_programs, grid_cpp=first_grid, no_gpu=True)
    again = make_launcher(write_programs, grid_cpp=first_grid, no_gpu=True)
    second = make_launcher(write_programs, grid_cpp=second_grid, no_gpu=True)
    one_dim = make_launcher(write_programs, grid_arg=1, no_gpu=True)
    two_dim = make_launcher(write_programs, grid_arg=2, no_gpu=True)
    assert first.__self__ is again.__self__
    assert len({id(launch.__self__) for launch in (first, second, one_dim, two_dim)}) == 4


def test_grid_py_receives_current_meta_on_every_call_and_owns_callback():
    first_seen = []
    second_seen = []

    def first(meta):
        first_seen.append(meta)
        return (1,)

    def second(meta):
        second_seen.append(meta)
        return (1,)

    first_launch = make_launcher(write_programs, grid_py=first, no_gpu=True)
    second_launch = make_launcher(write_programs, grid_py=second, no_gpu=True)
    assert first_launch.__self__ is second_launch.__self__
    compiled = []

    def compile_once(key, nparams, device, *args):
        compiled.append(bytes(key))
        return 0, 1, 0, nparams

    first_launch.__self__.set_compile_callback(compile_once)
    for x in (3, 4):
        assert first_launch(0, 0, 0, x, 2) is None
    misses = len(compiled)
    assert misses >= 1
    assert first_launch(0, 0, 0, 4, 2) is None
    assert second_launch(0, 0, 0, 4, 2) is None
    assert len(compiled) == misses
    assert second_launch(0, 0, 0, 7, 2) is None
    assert [meta["X"] for meta in first_seen] == [3, 4, 4]
    assert [meta["X"] for meta in second_seen] == [4, 7]
    assert all(set(meta) == {"out", "X", "Y"} for meta in first_seen + second_seen)
    assert len({id(meta) for meta in first_seen + second_seen}) == 5


def test_grid_py_callback_cycle_is_collected():
    def make_cycle():
        holder = {}

        def grid(meta):
            holder.get("launcher")
            return (0,)

        holder["launcher"] = make_launcher(write_programs, grid_py=grid, no_gpu=True)
        return weakref.ref(grid)

    callback_ref = make_cycle()
    gc.collect()
    assert callback_ref() is None


def test_grid_py_receives_baked_and_bound_values():
    seen = []

    def grid(meta):
        seen.append(dict(meta))
        return (0,)

    factory = make_launcher(write_programs, grid_py=grid, bind_device=True,
                            no_gpu=True, extra_annotation={
        "out": Argument(type=tl.pointer_type(tl.int32), specialize=NEVER,
                        bind_value=BindValue.POINTER),
        "X": Constexpr(value=3),
    })
    launch = factory.bind_device(0, out=0)
    assert tuple(inspect.signature(launch).parameters) == ("stream", "Y")
    assert launch(0, 2) is None
    assert seen == [{"out": 0, "X": 3, "Y": 2}]


def test_grid_py_gpu_dimensions_and_exception():
    def grid(meta):
        assert isinstance(meta["out"], torch.Tensor)
        if meta["X"] < 0:
            raise RuntimeError("grid error")
        return (meta["X"], meta["Y"], 4)

    launcher = make_launcher(write_programs, grid_py=grid)
    out = torch.zeros(24, device="cuda", dtype=torch.int32)
    launcher(torch.cuda.current_device(), torch.cuda.current_stream().cuda_stream,
             out, 3, 2)
    torch.cuda.synchronize()
    torch.testing.assert_close(out, torch.ones_like(out))
    with pytest.raises(RuntimeError, match="grid error"):
        launcher(torch.cuda.current_device(), torch.cuda.current_stream().cuda_stream,
                 out, -1, 2)


def compiled_grid(X: int, Y: int, *, cap: int, active: bool):
    tiles = triton.cdiv(X, 2)
    return (min(tiles, cap), Y if active else 1)


def test_grid_cpp_uses_reused_and_extra_arguments_without_python_callback():
    launcher = make_launcher(write_programs, grid_cpp=compiled_grid)
    out = torch.zeros(6, device="cuda", dtype=torch.int32)
    device = torch.cuda.current_device()
    stream = torch.cuda.current_stream().cuda_stream
    launcher(device, stream, 1, True, out, 3, 2)
    torch.cuda.synchronize()
    assert out.tolist() == [1, 0, 0, 1, 0, 0]
    out.zero_()
    launcher(device, stream, 2, True, out, 3, 2)
    torch.cuda.synchronize()
    assert out.tolist() == [1, 1, 0, 1, 1, 0]


def test_grid_cpp_checks_python_floor_and_integer_bounds():
    def floor_grid(X: int):
        return (X // -2 + 3,)

    floor = make_launcher(write_programs, grid_cpp=floor_grid, no_gpu=True)
    assert floor(0, 0, object(), 5, object()) is None  # -3 + 3 = 0; no kernel decode

    def overflow_grid(X: int):
        return (X + 1,)

    overflow = make_launcher(write_programs, grid_cpp=overflow_grid, no_gpu=True)
    with pytest.raises(OverflowError):
        overflow(0, 0, 0, 2**63 - 1, 2)
    with pytest.raises(TypeError):
        overflow(0, 0, 0, True, 2)

    def divide_grid(X: int):
        return (X // 0,)

    divide = make_launcher(write_programs, grid_cpp=divide_grid, no_gpu=True)
    with pytest.raises(ZeroDivisionError):
        divide(0, 0, 0, 2, 2)


def test_grid_cpp_validates_final_dimensions_and_checked_arithmetic():
    def identity_grid(X: int):
        return (X,)

    identity = make_launcher(write_programs, grid_cpp=identity_grid, no_gpu=True)
    for value in (-1, 2**32):
        with pytest.raises(ValueError, match="grid dimensions"):
            identity(0, 0, 0, value, 2)
    with pytest.raises(OverflowError):
        identity(0, 0, 0, 2**63, 2)

    def multiply_grid(X: int):
        return (X * 2,)

    multiply = make_launcher(write_programs, grid_cpp=multiply_grid, no_gpu=True)
    with pytest.raises(OverflowError):
        multiply(0, 0, 0, 2**62, 2)

    def floor_grid(X: int):
        return (X // -1,)

    floor = make_launcher(write_programs, grid_cpp=floor_grid, no_gpu=True)
    with pytest.raises(OverflowError):
        floor(0, 0, 0, -(2**63), 2)


def test_grid_cpp_fixed_device_reuses_baked_value_and_prepends_extra():
    def grid(X: int, Y: int, *, cap: int):
        return (X + Y - cap,)

    factory = make_launcher(write_programs, grid_cpp=grid, bind_device=True,
                            no_gpu=True, extra_annotation={
        "out": Argument(type=tl.pointer_type(tl.int32), specialize=NEVER,
                        bind_value=BindValue.POINTER),
        "X": Constexpr(value=3),
    })
    launch = factory.bind_device(0, out=0)
    assert tuple(inspect.signature(launch).parameters) == ("stream", "cap", "Y")
    assert launch(0, 7, 4) is None  # 3 + 4 - 7 = 0
    with pytest.raises(ValueError, match="grid dimensions"):
        launch(0, 8, 4)
    assert launch(0, 6, 4) is None


def test_grid_controls_do_not_change_kernel_specialization():
    def grid(X: int, *, cap: int):
        return (min(X, cap),)

    cpp = make_launcher(write_programs, grid_cpp=grid, no_gpu=True)
    explicit = make_launcher(write_programs, grid_arg=1, no_gpu=True)
    for launcher, calls in ((cpp, (1, 2)), (explicit, (1, 2))):
        misses = []

        def compile_once(key, nparams, device, *args):
            misses.append(bytes(key))
            return 0, 1, 0, nparams

        launcher.__self__.set_compile_callback(compile_once)
        for value in calls:
            assert launcher(0, 0, value, 0, 3, 2) is None
        assert len(misses) == 1


def test_cuda_grid_modes_compile_without_cuda_gpu(tmp_path):
    from intj.launcher import BACKENDS, _LOADED, _build

    launchers = {
        "grid_arg": make_launcher(write_programs, grid_arg=2, no_gpu=True,
                                   torch_access=TorchAccess.CPYTHON),
        "grid_cpp": make_launcher(write_programs, grid_cpp=compiled_grid, no_gpu=True,
                                   torch_access=TorchAccess.CPYTHON),
        "grid_py": make_launcher(write_programs, grid_py=lambda meta: (1,), no_gpu=True,
                                  torch_access=TorchAccess.CPYTHON),
    }
    backend = BACKENDS["cuda"]
    for name, launcher in launchers.items():
        module = launcher.__self__
        context = next(key.context for key, value in _LOADED.items() if value is module)
        context = dataclasses.replace(
            context, no_gpu=False, driver_path="/intj-test-no-driver.so",
            launch_symbol=backend.launch_symbol, device_symbol=backend.device_symbol,
            error_symbol=backend.error_symbol, error_style=backend.error_style,
        )
        binary = tmp_path / f"{name}.so"
        _build(binary, context)
        assert binary.is_file()


def test_grid_cpp_refuses_unannotated_or_captured_names():
    def unannotated(X):
        return (X,)

    with pytest.raises(UnsupportedKernel, match="annotat"):
        make_launcher(write_programs, grid_cpp=unannotated, no_gpu=True)
    unannotated.__annotations__["X"] = int
    with pytest.raises(UnsupportedKernel, match="annotat"):
        make_launcher(write_programs, grid_cpp=unannotated, no_gpu=True)

    captured = 3

    def uses_capture(X: int):
        return (X + captured,)

    with pytest.raises(UnsupportedKernel, match="free variable"):
        make_launcher(write_programs, grid_cpp=uses_capture, no_gpu=True)


def test_grid_cpp_refuses_invalid_source_and_unsupported_syntax():
    def wrong_name(Z: int):
        return (Z,)

    def wrong_extra(*, X: int):
        return (X,)

    def wrong_annotation(X: float):
        return (X,)

    def defaulted(X: int = 1):
        return (X,)

    def global_read(X: int):
        return (X + GLOBAL_GRID_OFFSET,)

    def modulo(X: int):
        return (X % 2,)

    for grid, message in (
        (wrong_name, "must name a kernel parameter"),
        (wrong_extra, "must not name a kernel parameter"),
        (wrong_annotation, "int or bool annotation"),
        (defaulted, "cannot have defaults"),
        (global_read, "unknown or global name"),
        (modulo, "unsupported arithmetic operator"),
        (lambda X: (X,), "one Python def function"),
    ):
        with pytest.raises(UnsupportedKernel, match=message):
            make_launcher(write_programs, grid_cpp=grid, no_gpu=True)

    generated = {}
    exec("def grid(X: int):\n    return (X,)\n", generated)
    with pytest.raises(UnsupportedKernel, match="available Python source"):
        make_launcher(write_programs, grid_cpp=generated["grid"], no_gpu=True)


def test_grid_cpp_refuses_shadowed_helpers(tmp_path):
    def local_min(X: int):
        result = min(X, 2)  # pyright: ignore[reportUnboundVariable]
        min = 1
        return (result,)

    def extra_triton(X: int, *, triton: int):
        return (triton.cdiv(X, 2),)  # pyright: ignore[reportAttributeAccessIssue]

    def local_triton(X: int):
        result = triton.cdiv(X, 2)  # pyright: ignore[reportUnboundVariable, reportAttributeAccessIssue]
        triton = 1
        return (result,)

    for grid, helper in ((local_min, "min"), (extra_triton, "triton"),
                         (local_triton, "triton")):
        with pytest.raises(UnsupportedKernel, match=helper):
            make_launcher(write_programs, grid_cpp=grid, no_gpu=True)

    path = tmp_path / "shadowed_builtin_grid.py"
    path.write_text("def grid(X: int):\n    return (min(X, 2),)\n")
    spec = importlib.util.spec_from_file_location("shadowed_builtin_grid", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    module.__dict__["__builtins__"] = {**vars(builtins), "min": max}
    spec.loader.exec_module(module)
    with pytest.raises(UnsupportedKernel, match="min"):
        make_launcher(write_programs, grid_cpp=module.grid, no_gpu=True)

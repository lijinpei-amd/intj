# pyright: standard
"""The opt-in launcher result is Triton's cached CompiledKernel."""

import gc
import weakref

import pytest
import torch
import triton
import triton.language as tl
from triton.compiler import CompiledKernel

from intj import Argument, BindValue, KernelCache, NEVER, make_launcher
from intj.launcher import UnsupportedKernel


@triton.jit
def store_at_offset(x, offset):
    tl.store(x + offset, offset)


@triton.jit
def store_value(x, value):
    tl.store(x, value)


@triton.jit
def reentrant_store(x, value):
    tl.store(x, value)


@triton.jit
def gc_cycle_store(x, value):
    tl.store(x, value)


def grid_from_value(value: int):
    return (value - value + 1,)


def test_return_compiled_exposes_ttir_and_reuses_object():
    x = torch.empty(4, device="cuda", dtype=torch.int32)
    launch = make_launcher(store_at_offset, return_compiled=True)
    device = torch.cuda.current_device()
    stream = torch.cuda.current_stream().cuda_stream

    kernel = launch(device, stream, (1,), x, 3)
    assert isinstance(kernel, CompiledKernel)
    assert "%offset: i32" in kernel.asm["ttir"]
    assert launch(device, stream, (1,), x, 3) is kernel
    assert x[3].item() == 3


def test_zero_grid_returns_kernel_without_dispatch_and_mode_changes_module():
    x = torch.zeros(1, device="cuda", dtype=torch.int32)
    device = torch.cuda.current_device()
    stream = torch.cuda.current_stream().cuda_stream
    default = make_launcher(store_value)
    returning = make_launcher(store_value, return_compiled=True)

    assert default.__self__.__file__ != returning.__self__.__file__
    assert default.__self__.spec_key(x, 7) == returning.__self__.spec_key(x, 7)
    assert default(device, stream, (0,), x, 7) is None
    kernel = returning(device, stream, (0,), x, 7)
    assert kernel.asm["ttir"]
    assert x.item() == 0
    assert returning(device, stream, (1,), x, 7) is kernel
    assert x.item() == 7
    assert default(device, stream, (1,), x, 8) is None


def test_compiled_grid_returns_cached_kernel():
    x = torch.zeros(1, device="cuda", dtype=torch.int32)
    launch = make_launcher(store_value, grid_cpp=grid_from_value, return_compiled=True)
    device = torch.cuda.current_device()
    stream = torch.cuda.current_stream().cuda_stream

    kernel = launch(device, stream, x, 13)
    assert kernel.asm["ttir"]
    assert launch(device, stream, x, 13) is kernel
    assert x.item() == 13


def test_return_compiled_refuses_host_mode_before_provision(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("host return mode reached provisioning")

    monkeypatch.setattr("intj.launcher._provision", forbidden)
    with pytest.raises(UnsupportedKernel, match="return_compiled.*GPU"):
        make_launcher(store_value, no_gpu=True, return_compiled=True)


def test_bound_no_map_keeps_kernel_alive_until_bound_handle_is_released():
    x = torch.zeros(1, device="cuda", dtype=torch.int32)
    device = torch.cuda.current_device()
    stream = torch.cuda.current_stream().cuda_stream
    factory = make_launcher(
        store_value, bind_device=True, return_compiled=True,
        extra_annotation={
            "x": Argument(type=tl.pointer_type(tl.int32), specialize=NEVER,
                          bind_value=BindValue.TENSOR),
            "value": Argument(type=tl.int32, specialize=NEVER),
        },
    )
    bound = factory.bind_device(device, x=x)
    import intj.launcher as launcher_module

    assert any(key.context.nwords == 0 and module.__file__ == bound.__self__.__file__
               for key, module in launcher_module._LOADED.items())
    kernel = bound(stream, (1,), 11)
    assert bound(stream, (1,), 12) is kernel
    assert x.item() == 12

    reference = weakref.ref(kernel)
    del kernel
    gc.collect()
    assert reference() is not None
    del bound
    gc.collect()
    assert reference() is None


def test_reentrant_cache_miss_returns_winning_record(monkeypatch):
    import triton.compiler

    x = torch.zeros(1, device="cuda", dtype=torch.int32)
    device = torch.cuda.current_device()
    stream = torch.cuda.current_stream().cuda_stream
    real_compile = triton.compiler.compile
    compiled = []
    nested_result = []
    launch = None

    def compile_with_reentry(*args, **kwargs):
        kernel = real_compile(*args, **kwargs)
        compiled.append(kernel)
        if len(compiled) == 1:
            assert launch is not None
            nested_result.append(launch(device, stream, (1,), x, 5))
        return kernel

    monkeypatch.setattr(triton.compiler, "compile", compile_with_reentry)
    launch = make_launcher(reentrant_store, return_compiled=True)
    result = launch(device, stream, (1,), x, 5)

    assert len(compiled) == 2
    assert result is nested_result[0] is compiled[1]
    assert result is not compiled[0]
    assert x.item() == 5


@pytest.mark.parametrize("cache", list(KernelCache), ids=lambda cache: cache.value)
def test_module_cache_compiled_reference_is_gc_traversed(monkeypatch, cache):
    import intj.launcher as launcher_module

    monkeypatch.setattr(launcher_module, "_LOADED", {})
    launch = make_launcher(gc_cycle_store, kernel_cache=cache, return_compiled=True)
    module = launch.__self__
    module_ref = weakref.ref(module)

    class Owner:
        module: object

    owner = Owner()
    owner.module = module
    owner_ref = weakref.ref(owner)

    def compile_fake(_key, nparams, _device, *_args):
        return 0, 1, 0, nparams, owner

    module.set_compile_callback(compile_fake)
    x = torch.zeros(1, device="cuda", dtype=torch.int32)
    result = launch(torch.cuda.current_device(), torch.cuda.current_stream().cuda_stream,
                    (0,), x, 9)
    assert result is owner
    module.set_compile_callback(lambda *_args: None)
    launcher_module._LOADED.clear()
    del result, owner, compile_fake, launch, module

    gc.collect()
    assert module_ref() is None
    assert owner_ref() is None


def test_bound_no_map_compiled_reference_is_gc_traversed():
    import intj.launcher as launcher_module

    x = torch.zeros(1, device="cuda", dtype=torch.int32)
    factory = make_launcher(
        gc_cycle_store, bind_device=True, return_compiled=True,
        extra_annotation={
            "x": Argument(type=tl.pointer_type(tl.int32), specialize=NEVER,
                          bind_value=BindValue.TENSOR),
            "value": Argument(type=tl.int32, specialize=NEVER),
        },
    )
    bound = factory.bind_device(torch.cuda.current_device(), x=x)
    module = next(module for module in launcher_module._LOADED.values()
                  if module.__file__ == bound.__self__.__file__)

    class Owner:
        bound: object

    owner = Owner()
    owner.bound = bound
    owner_ref = weakref.ref(owner)

    def compile_fake(_key, nparams, _device, *_args):
        return 0, 1, 0, nparams, owner

    module.set_compile_callback(compile_fake)
    result = bound(torch.cuda.current_stream().cuda_stream, (0,), 9)
    assert result is owner
    module.set_compile_callback(lambda *_args: None)
    del result, owner, compile_fake, bound

    gc.collect()
    assert owner_ref() is None

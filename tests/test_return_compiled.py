# pyright: standard
"""The opt-in launcher result is Triton's cached CompiledKernel."""

import ctypes
import gc
import subprocess
import sys
import sysconfig
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


@triton.jit
def gc_resize_store(x, value: tl.constexpr):
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
        store_value,
        bind_device=True,
        return_compiled=True,
        extra_annotation={
            "x": Argument(
                type=tl.pointer_type(tl.int32),
                specialize=NEVER,
                bind_value=BindValue.TENSOR,
            ),
            "value": Argument(type=tl.int32, specialize=NEVER),
        },
    )
    bound = factory.bind_device(device, x=x)
    import intj.launcher as launcher_module

    assert any(
        key.context.nwords == 0 and module.__file__ == bound.__self__.__self__.__file__
        for key, module in launcher_module._LOADED.items()
    )
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
    result = launch(
        torch.cuda.current_device(), torch.cuda.current_stream().cuda_stream, (0,), x, 9
    )
    assert result is owner
    module.set_compile_callback(lambda *_args: None)
    launcher_module._LOADED.clear()
    del result, owner, compile_fake, launch, module

    gc.collect()
    assert module_ref() is None
    assert owner_ref() is None


@pytest.fixture(scope="module")
def traversal_helpers(tmp_path_factory):
    path = tmp_path_factory.mktemp("visit")
    source = path / "visit.c"
    library = path / "visit.so"
    source.write_text(
        r"""
#include <Python.h>

typedef struct {
  PyObject *trigger;
  PyObject *callback;
  int seen;
  int inserted;
} visit_context;

static int visit_owner(PyObject *object, void *opaque) {
  visit_context *context = (visit_context *)opaque;
  if (Py_TYPE(object) == Py_TYPE(context->trigger))
    context->seen++;
  if (object == context->trigger && !context->inserted) {
    context->inserted = 1;
    PyObject *result = PyObject_CallObject(context->callback, NULL);
    if (!result)
      return -1;
    Py_DECREF(result);
  }
  return 0;
}

int traverse_count(PyObject *owner, PyObject *trigger, PyObject *callback) {
  visit_context context = {trigger, callback, 0, 0};
  if (Py_TYPE(owner)->tp_traverse(owner, visit_owner, &context))
    return -1;
  if (!context.inserted) {
    PyErr_SetString(PyExc_AssertionError, "trigger was not visited");
    return -1;
  }
  return context.seen;
}

static int visit_callback(PyObject *object, void *opaque) {
  visit_context *context = (visit_context *)opaque;
  if (object != PyWeakref_GetObject(context->trigger))
    return 0;
  context->seen = 1;
  PyObject *result = PyObject_CallObject(context->callback, NULL);
  if (!result)
    return -1;
  Py_DECREF(result);
  context->inserted = PyWeakref_GetObject(context->trigger) != Py_None;
  return 0;
}

int traverse_callback_alive(PyObject *owner, PyObject *weakref, PyObject *callback) {
  visit_context context = {weakref, callback, 0, 0};
  if (Py_TYPE(owner)->tp_traverse(owner, visit_callback, &context))
    return -1;
  if (!context.seen) {
    PyErr_SetString(PyExc_AssertionError, "callback was not visited");
    return -1;
  }
  return context.inserted;
}

static int fail_on_owner(PyObject *object, void *opaque) {
  return Py_TYPE(object) == Py_TYPE((PyObject *)opaque) ? 77 : 0;
}

int traverse_fail(PyObject *owner, PyObject *trigger) {
  return Py_TYPE(owner)->tp_traverse(owner, fail_on_owner, trigger);
}
"""
    )
    subprocess.run(
        [
            "cc",
            "-shared",
            "-fPIC",
            f"-I{sysconfig.get_paths()['include']}",
            str(source),
            "-o",
            str(library),
        ],
        check=True,
    )
    lib = ctypes.PyDLL(str(library))
    for name in ("traverse_count", "traverse_callback_alive"):
        method = getattr(lib, name)
        method.argtypes = [ctypes.py_object] * 3
        method.restype = ctypes.c_int
    lib.traverse_fail.argtypes = [ctypes.py_object] * 2
    lib.traverse_fail.restype = ctypes.c_int
    return lib


@pytest.mark.parametrize("cache", list(KernelCache), ids=lambda cache: cache.value)
@pytest.mark.parametrize("fixed_device", [False, True], ids=["module", "bound"])
def test_cache_traversal_uses_a_snapshot_when_a_visitor_reenters(
    traversal_helpers, fixed_device, cache
):
    x = torch.zeros(1, device="cuda", dtype=torch.int32)
    device = torch.cuda.current_device()
    stream = torch.cuda.current_stream().cuda_stream
    factory = make_launcher(
        gc_resize_store,
        kernel_cache=cache,
        bind_device=fixed_device,
        return_compiled=True,
        extra_annotation=(
            {
                "x": Argument(
                    type=tl.pointer_type(tl.int32),
                    specialize=NEVER,
                    bind_value=BindValue.TENSOR,
                )
            }
            if fixed_device
            else None
        ),
    )
    if fixed_device:
        bound = factory.bind_device(device, x=x)
        owner = bound.__self__
        module = owner.__self__

        def launch(value):
            return bound(stream, (0,), value)

    else:
        module = factory.__self__
        owner = module

        def launch(value):
            return factory(device, stream, (0,), x, value)

    owners = []

    class Owner:
        pass

    def compile_fake(_key, nparams, _device, *_args):
        owner = Owner()
        owners.append(owner)
        return 0, 1, 0, nparams, owner

    module.set_compile_callback(compile_fake)
    for value in range(8):
        assert launch(value) is owners[-1]
    assert sum(type(ref) is Owner for ref in gc.get_referents(owner)) == 8
    refcounts = [sys.getrefcount(value) for value in owners]
    assert traversal_helpers.traverse_fail(owner, owners[0]) == 77
    assert [sys.getrefcount(value) for value in owners] == refcounts

    def grow_cache():
        for value in range(8, 32):
            launch(value)

    assert traversal_helpers.traverse_count(owner, owners[0], grow_cache) == 8
    assert len(owners) == 32


def test_compile_callback_survives_reentrant_traversal(traversal_helpers):
    launch = make_launcher(gc_resize_store, grid_arg=1, return_compiled=True)
    module = launch.__self__

    class OldCallback:
        def __call__(self, *_args):
            pytest.fail("callback should not run")

    old = OldCallback()
    old_ref = weakref.ref(old)
    module.set_compile_callback(old)
    del old
    assert old_ref() is not None

    def replace_callback():
        module.set_compile_callback(lambda *_args: None)

    assert traversal_helpers.traverse_callback_alive(module, old_ref, replace_callback)
    assert old_ref() is None


def test_bound_no_map_compiled_reference_is_gc_traversed():
    import intj.launcher as launcher_module

    x = torch.zeros(1, device="cuda", dtype=torch.int32)
    factory = make_launcher(
        gc_cycle_store,
        bind_device=True,
        return_compiled=True,
        extra_annotation={
            "x": Argument(
                type=tl.pointer_type(tl.int32),
                specialize=NEVER,
                bind_value=BindValue.TENSOR,
            ),
            "value": Argument(type=tl.int32, specialize=NEVER),
        },
    )
    bound = factory.bind_device(torch.cuda.current_device(), x=x)
    module = next(
        module
        for module in launcher_module._LOADED.values()
        if module.__file__ == bound.__self__.__self__.__file__
    )

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

# pyright: standard
"""The rendered module, end to end, without triton or a GPU.

Builds the real `entry.c.jinja` render with the system compiler against a stub
driver that records each launch, so it runs on any CPython intj supports, with
nothing but torch installed.  That is what makes the python-version matrix
testable: triton 3.8 has no wheel below 3.10.  See `tests/run_python_matrix.sh`.

Every Torch access mode this torch supports is built: STATIC_COMPILE when torch
ships its headers and a C++ compiler is found, RUNTIME_SHIM when torch has a
verified layout.  `INTJ_TEST_TORCH_MODES=interpreter,runtime_shim,...` names the
modes instead, and fails on one that is unavailable rather than dropping it.
"""

from __future__ import annotations

import contextlib
import ctypes
import dataclasses
import dis
import gc
import importlib.machinery
import importlib.util
import json
import os
import struct
import subprocess
import sys
import sysconfig
import threading
import time
import weakref
from collections.abc import Callable

import pytest
import torch

from intj import launcher, lazy
from intj.annotation import CanonicalAnnotation, DeviceBinding, ResolvedParam
from intj.render import build
from intj.render import module as render_module
from intj.render.context import render
from intj.render.key import RenderKey
from intj.python_intf import cpython_abi
from intj.torch_intf.torch_abi import (
    NDTYPES,
    TorchAccessMode,
    layout_for,
    torch_version,
)

_STUB = r"""
#include <stdint.h>
#include <string.h>

uint64_t intj_stub_params[16];
uint64_t intj_stub_launch[5]; /* gx, gy, gz, block_dim, stream */
long intj_stub_calls;

int32_t stub_launch(void *f, uint32_t gx, uint32_t gy, uint32_t gz, uint32_t bx,
                    uint32_t by, uint32_t bz, uint32_t shared, void *stream,
                    void **params, void **extra) {
  __atomic_fetch_add(&intj_stub_calls, 1, __ATOMIC_RELAXED);
  if ((uintptr_t)f == 0xbad)
    return 7;
  uint64_t launch[5] = {gx, gy, gz, bx, (uint64_t)(uintptr_t)stream};
  memcpy(intj_stub_launch, launch, sizeof(launch));
  /* the fake handle is 0x1000 + the argument count */
  for (uintptr_t i = 0; i < (uintptr_t)f - 0x1000; i++)
    memcpy(&intj_stub_params[i], params[i], 8);
  return 0;
}

const char *stub_error(int32_t status) { return "stub failure"; }
"""

#: x: tensor, n: int, m: do_not_specialize int, f: float, flag: bool, BLOCK: constexpr
_AUTO = CanonicalAnnotation(
    "argument", None, "auto", "auto", "auto", (), False, None, ()
)
_NEVER = dataclasses.replace(_AUTO, equal_to_one="never", aligned_16="never")
_CONSTEXPR = dataclasses.replace(_AUTO, kind="constexpr")
_PARAMS, _DEVICE_OFFSET, _NWORDS = launcher._render_params(
    (
        ResolvedParam("x", 0, _AUTO),
        ResolvedParam("n", 1, _AUTO),
        ResolvedParam("m", 2, _NEVER),
        ResolvedParam("f", 3, _AUTO),
        ResolvedParam("flag", 4, _AUTO),
        ResolvedParam("BLOCK", 5, _CONSTEXPR),
    ),
    DeviceBinding.NOT_FIXED,
)


def _cc(language: str) -> str:
    return (
        os.environ.get("CXX", "c++")
        if language == "c++"
        else os.environ.get("CC", "cc")
    )


@contextlib.contextmanager
def _test_compiler():
    """Points intj's compiler lookup, which asks triton, at `_cc`."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(build, "compiler_path", _cc)
        build.cxx_toolchain.cache_clear()
        build.cxx_lib_dirs.cache_clear()
        try:
            yield
        finally:
            build.cxx_toolchain.cache_clear()
            build.cxx_lib_dirs.cache_clear()


def _available(mode: TorchAccessMode) -> bool:
    if mode is TorchAccessMode.RUNTIME_SHIM:
        return layout_for() is not None
    if mode is TorchAccessMode.STATIC_COMPILE:
        with _test_compiler():
            return build.cxx_toolchain() is not None
    return True


_REQUIRED = os.environ.get("INTJ_TEST_TORCH_MODES")
_MODES = (
    [TorchAccessMode(m) for m in _REQUIRED.split(",")]
    if _REQUIRED
    else [m for m in TorchAccessMode if _available(m)]
)


@pytest.fixture(scope="module")
def stub(tmp_path_factory):
    out = tmp_path_factory.mktemp("stub")
    src, lib = out / "stub.c", out / "stub.so"
    src.write_text(_STUB)
    subprocess.run(
        [_cc("c"), "-shared", "-fPIC", "-O2", str(src), "-o", str(lib)], check=True
    )
    return ctypes.CDLL(str(lib), mode=ctypes.RTLD_GLOBAL), str(lib)


class Stub:
    """A view of the stub driver's counters and last recorded launch."""

    def __init__(self, lib):
        self.lib = lib

    def calls(self) -> int:
        return ctypes.c_long.in_dll(self.lib, "intj_stub_calls").value

    def last(self, nparams: int):
        """Returns `(grid, block_dim, stream, params)` of the last launch."""
        launch = list((ctypes.c_uint64 * 5).in_dll(self.lib, "intj_stub_launch"))
        params = list((ctypes.c_uint64 * 16).in_dll(self.lib, "intj_stub_params"))
        return tuple(launch[:3]), launch[3], launch[4], params[:nparams]


def _key(mode: TorchAccessMode, driver_path: str = "") -> RenderKey:
    """Returns the key the `built` fixture renders and builds for `mode`."""
    static = mode is TorchAccessMode.STATIC_COMPILE
    return RenderKey(
        module_name=f"rt_{mode.value}",
        kernel_repr="test_runtime.kernel",
        params=_PARAMS,
        nwords=_NWORDS,
        device_binding=DeviceBinding.NOT_FIXED,
        device_offset=_DEVICE_OFFSET,
        verify_annotation=False,
        max_slots=5,
        spec_pointer_range=1,
        driver_path=driver_path,
        launch_symbol="stub_launch",
        device_symbol="",
        error_symbol="stub_error",
        error_style="return",
        torch_access_mode=mode.value,
        kernel_cache="intj",
        cache_include_dirs=(),
        cache_archives=(),
        torch_version=torch_version() if static else None,
        cxx_abi=render_module._cxx_abi() if static else None,
        python_version=cpython_abi.python_version(),
        free_threaded=bool(sysconfig.get_config_var("Py_GIL_DISABLED")),
        cpython_static_compile_header=cpython_abi.header_for() or "",
        # rendered and built here directly, so never hashed or read
        template="",
        runtime_header="",
        cache_key="",
        target=(),
        options="",
        triton=(),
        compiler=(),
        build_flags=(),
        ext_suffix=None,
        intj_version=(0, 0),
        global_values=(),
    )


@pytest.fixture(scope="module", params=_MODES, ids=lambda m: m.value)
def built(request, stub, stub_module, tmp_path_factory):
    """Builds the entry module for one access mode against the stub driver.

    Returns:
        The loaded module, a `Stub` over the driver, and the list of values the
        compile callback was called with.
    """
    lib, lib_path = stub
    mode = request.param
    if not _available(mode):
        pytest.fail(
            f"INTJ_TEST_TORCH_MODES names {mode.value}, unavailable on torch "
            f"{torch.__version__}"
        )
    key = _key(mode, lib_path)
    with _test_compiler():
        flags = build.build_flags(key)
    out = tmp_path_factory.mktemp(mode.value)
    src = out / "entry.c"
    so = out / f"{key.module_name}{sysconfig.get_config_var('EXT_SUFFIX')}"
    src.write_text(render(key))
    subprocess.run(
        [
            _cc(flags["language"]),
            "-shared",
            "-fPIC",
            "-O2",
            "-Wall",
            "-Werror",
            "-Wno-unused-but-set-variable",
            "-x",
            flags["language"],
            *flags["ccflags"],
            *(
                f"-I{d}"
                for d in [*flags["include_dirs"], sysconfig.get_paths()["include"]]
            ),
            str(src),
            "-o",
            str(so),
            *(f"-L{d}" for d in flags.get("library_dirs", [])),
            *(f"-l{lib}" for lib in flags.get("libraries", [])),
        ],
        check=True,
    )
    spec = importlib.util.spec_from_file_location(key.module_name, so)
    module = importlib.util.module_from_spec(spec)  # pyright: ignore[reportArgumentType]
    spec.loader.exec_module(module)  # pyright: ignore
    module._set_launcher_type(stub_module.Launcher)

    compiles = []

    def compile_cb(_key, nparams, _device, *args):
        compiles.append(args[-1])
        return (0xBAD if args[-1] == 999 else 0x1000 + nparams, 128, 0, nparams)

    launcher.override_compile(module, compile_cb)
    layout = layout_for().as_args() if mode is TorchAccessMode.RUNTIME_SHIM else None  # pyright: ignore[reportOptionalMemberAccess]
    # every torch code its own 5-bit index; the real one comes from triton
    module._set_torch_version(
        torch_version(), layout, bytes(c if c < 32 else 0xFF for c in range(NDTYPES))
    )
    return module, Stub(lib), compiles


def _f32(v: float) -> int:
    return struct.unpack("<I", struct.pack("<f", v))[0]


@pytest.mark.parametrize(
    "head, version, expected",
    [
        (16, (2, 9), 8),
        (32, (2, 9), 8),
        (16, (2, 10), 0),
        (32, (2, 10), 0),
        (65535, (2, 9), None),
        (65535, (2, 10), 0),
        (65536, (2, 10), None),
    ],
)
def test_runtime_shim_cdata_is_header_relative(monkeypatch, head, version, expected):
    from intj.torch_intf import torch_abi as abi

    monkeypatch.setattr(abi, "pyobject_size", lambda: head)
    monkeypatch.setattr(abi, "itemsize_table", lambda _version: b"\x01" * NDTYPES)
    abi.layout_for.cache_clear()
    try:
        layout = abi.layout_for(version)
        assert (layout.cdata if layout else None) == expected
    finally:
        abi.layout_for.cache_clear()


def test_runtime_shim_refuses_32_bit_pointer_layout(monkeypatch):
    from intj.torch_intf import torch_abi as abi

    monkeypatch.setattr(abi, "pyobject_size", lambda: 8)
    monkeypatch.setattr(abi.ctypes, "sizeof", lambda _: 4)
    monkeypatch.setattr(abi, "itemsize_table", lambda _version: b"\x01" * NDTYPES)
    abi.layout_for.cache_clear()
    try:
        assert abi.layout_for((2, 10)) is None
    finally:
        abi.layout_for.cache_clear()


def test_runtime_shim_layout_matches_live_torch(monkeypatch):
    from intj.torch_intf import abi_detect

    layout = layout_for()
    if sysconfig.get_config_var("Py_GIL_DISABLED"):
        assert layout is not None, "free-threaded matrix must exercise RUNTIME_SHIM"
    elif layout is None:
        pytest.skip("no verified layout for this torch")
    assert layout is not None

    probes = abi_detect._probes()
    limits = {id(t): type(t).__basicsize__ for t in probes}
    window = abi_detect._window

    def checked_window(address, size):
        if address in limits:
            assert size <= limits[address], "probe read past the tensor object"
        return window(address, size)

    monkeypatch.setattr(abi_detect, "_probes", lambda: probes)
    monkeypatch.setattr(abi_detect, "_window", checked_window)
    assert abi_detect.probe_layout(cpython_abi.pyobject_size()) == layout
    tensor = torch.arange(8, dtype=torch.float32)
    absolute = cpython_abi.pyobject_size() + layout.cdata
    assert ctypes.c_void_p.from_address(id(tensor) + absolute).value == tensor._cdata


def test_runtime_shim_probe_refuses_slot_before_header():
    from intj.torch_intf.abi_detect import probe_layout

    assert probe_layout(torch.Tensor.__basicsize__ + 1) is None


@pytest.fixture
def fresh_lib_dirs():
    build.cxx_lib_dirs.cache_clear()
    yield
    build.cxx_lib_dirs.cache_clear()


def test_cxx_lib_dirs_agree_with_cpp_extension(monkeypatch):
    """Found without cpp_extension, the c++ mode's lib dirs are the ones it gives.

    They decide the access mode and enter the digest as an rpath.
    """
    with _test_compiler():
        toolchain = build.cxx_toolchain()  # imports cpp_extension
        monkeypatch.delitem(sys.modules, "torch.utils.cpp_extension", raising=False)
        found = build._lib_dirs_without_cpp_extension()
    if toolchain is None:
        assert found is None
    elif found is not None:
        assert found == tuple(toolchain[1])


@pytest.mark.parametrize(
    "lib, header, agree, setuptools, imported, certain",
    [
        (True, True, True, True, False, True),
        (False, True, True, True, False, False),
        (True, False, True, True, False, False),
        (True, True, False, True, False, False),
        (True, True, True, False, False, False),
        (True, True, True, True, True, False),
    ],
    ids=["certain", "no-lib", "no-header", "roots-differ", "no-setuptools", "imported"],
)
def test_lib_dirs_without_cpp_extension_answer_only_when_certain(
    tmp_path, monkeypatch, lib, header, agree, setuptools, imported, certain
):
    root = tmp_path / "torch"
    if lib:
        (root / "lib").mkdir(parents=True)
    if header:
        path = root / "include" / "torch/csrc/autograd/python_variable.h"
        path.parent.mkdir(parents=True)
        path.touch()
    real_find_spec = importlib.util.find_spec

    def find_spec(name, *args):
        if name == "setuptools":
            return importlib.machinery.ModuleSpec(name, None) if setuptools else None
        if name == "torch.utils.cpp_extension":
            origin = str(root / "utils" / "cpp_extension.py")
            return importlib.machinery.ModuleSpec(name, None, origin=origin)
        return real_find_spec(name, *args)

    monkeypatch.setattr(importlib.util, "find_spec", find_spec)
    monkeypatch.setattr(
        "torch._utils_internal.get_file_path",
        lambda *parts: str(root if agree else tmp_path / "elsewhere"),
    )
    if imported:
        # any module: only its presence is read
        monkeypatch.setitem(sys.modules, "torch.utils.cpp_extension", sys)
    else:
        monkeypatch.delitem(sys.modules, "torch.utils.cpp_extension", raising=False)
    expected = (str(root / "lib"),) if certain else None
    assert build._lib_dirs_without_cpp_extension() == expected


def test_cxx_lib_dirs_ask_cpp_extension_when_unsure(monkeypatch, fresh_lib_dirs):
    monkeypatch.setattr(build, "compiler_path", _cc)
    monkeypatch.setattr(build, "_lib_dirs_without_cpp_extension", lambda: None)
    for toolchain, expected in [((["/i"], ["/l"]), ("/l",)), (None, None)]:
        monkeypatch.setattr(build, "cxx_toolchain", lambda t=toolchain: t)
        build.cxx_lib_dirs.cache_clear()
        assert build.cxx_lib_dirs() == expected


def test_cxx_lib_dirs_need_a_cxx_compiler(monkeypatch, fresh_lib_dirs):
    def no_compiler(language):
        raise RuntimeError(f"Failed to find {language} compiler")

    monkeypatch.setattr(build, "compiler_path", no_compiler)
    monkeypatch.setattr(build, "_lib_dirs_without_cpp_extension", lambda: ("/l",))
    assert build.cxx_lib_dirs() is None


@pytest.mark.parametrize(
    "toolchain, error, match",
    [
        (None, build.CppExtensionUnavailable, "fails to import"),
        ((["/i"], ["/elsewhere"]), launcher.UnsupportedKernel, "disagrees"),
    ],
    ids=["unavailable", "disagrees"],
)
def test_build_refuses_where_cpp_extension_disagrees(
    monkeypatch, toolchain, error, match
):
    """The mode and digest were decided without cpp_extension: a build checks them.

    Unavailable is the one case the launcher recovers from, by choosing again.
    """
    monkeypatch.setattr(build, "cxx_lib_dirs", lambda: ("/l",))
    monkeypatch.setattr(build, "cxx_toolchain", lambda: toolchain)
    with pytest.raises(error, match=match) as raised:
        build.build_flags(_key(TorchAccessMode.STATIC_COMPILE))
    if toolchain is not None:  # the launcher must not recover from this one
        assert not isinstance(raised.value, build.CppExtensionUnavailable)


def test_build_flags_add_cpp_extension_includes_to_digest_flags(monkeypatch):
    monkeypatch.setattr(build, "cxx_lib_dirs", lambda: ("/l",))
    monkeypatch.setattr(build, "cxx_toolchain", lambda: (["/i"], ["/l"]))
    key = _key(TorchAccessMode.STATIC_COMPILE)
    flags, digest = build.build_flags(key), build.digest_flags(key)
    assert flags["include_dirs"] == [*digest["include_dirs"], "/i"]
    assert {k: v for k, v in flags.items() if k != "include_dirs"} == {
        k: v for k, v in digest.items() if k != "include_dirs"
    }
    assert flags["library_dirs"] == ["/l"]
    assert "-Wl,-rpath,/l" in flags["ccflags"]


def test_runtime_shim_rejects_rebased_cdata_overflow(built, launch):
    module, stub, _ = built
    if module.__name__ != "rt_runtime_shim":
        pytest.skip("only RUNTIME_SHIM installs cdata")
    layout = layout_for()
    assert layout is not None
    x = torch.arange(8, dtype=torch.float32)
    args = (0, 0, 1, x, 5, 0, 0.0, False, 64)
    launch(*args)
    assert stub.last(5)[3][0] == x.data_ptr()

    relative = (1 << 16) - cpython_abi.pyobject_size()
    assert 0 <= relative <= 65535
    bad = (relative, *layout.as_args()[1:])
    index = bytes(c if c < 32 else 0xFF for c in range(NDTYPES))
    with pytest.raises(ValueError, match="16-bit"):
        module._set_torch_version(torch_version(), bad, index)

    launch(*args)
    assert stub.last(5)[3][0] == x.data_ptr()


def test_torch_abi_is_installed_once(built):
    module, _, _ = built
    layout = layout_for() if module.__name__ == "rt_runtime_shim" else None
    index = bytes(c if c < 32 else 0xFF for c in range(NDTYPES))
    with pytest.raises(RuntimeError, match="already installed"):
        module._set_torch_version(
            torch_version(), layout.as_args() if layout else None, index
        )


@pytest.mark.parametrize("oversized_offset", [65536, 2**32])
def test_runtime_shim_rejects_offset_truncation(built, oversized_offset):
    module, _, _ = built
    if module.__name__ != "rt_runtime_shim":
        pytest.skip("only RUNTIME_SHIM reads the offsets")
    layout = layout_for()
    assert layout is not None
    oversized = (oversized_offset, *layout.as_args()[1:])
    index = bytes(c if c < 32 else 0xFF for c in range(NDTYPES))
    with pytest.raises(ValueError, match="16-bit"):
        module._set_torch_version(torch_version(), oversized, index)


def test_gil_stays_off_on_a_free_threaded_build(built):
    if not sysconfig.get_config_var("Py_GIL_DISABLED"):
        pytest.skip("default build")
    assert not sys._is_gil_enabled()  # pyright: ignore[reportAttributeAccessIssue]


@pytest.mark.parametrize("grid", [(2, 3), [2, 3]])
def test_arguments_reach_the_launch(built, launch, grid):
    module, stub, compiles = built
    x = torch.arange(8, dtype=torch.float32)
    launch(0, 77, grid, x, 5, -7, 1.5, True, 64)
    grid, block, stream, params = stub.last(5)
    assert (grid, block, stream) == ((2, 3, 1), 128, 77)
    assert params == [x.data_ptr(), 5, 2**64 - 7, _f32(1.5), 1]
    before = len(compiles)
    launch(0, 0, 4, x, 5, -7, 1.5, True, 64)
    assert len(compiles) == before  # a hit: no second compile
    assert stub.last(0)[0] == (4, 1, 1)


@pytest.mark.parametrize(
    "value",
    [
        0,
        16,
        17,
        -17,
        2**30 - 1,
        2**30,
        -(2**30),
        2**31,
        2**60 + 3,
        2**63 - 1,
        -(2**63),
        2**63,
        2**64 - 1,
        2**64,
        -(2**63) - 1,
        2**90,
        2**100,
    ],
)
def test_int_values(built, launch, value):
    module, stub, _ = built
    x = torch.zeros(4)
    if not -(2**63) <= value < 2**64:
        with pytest.raises(OverflowError):
            launch(0, 0, 1, x, value, 0, 0.0, False, 64)
        return
    launch(0, 0, 1, x, value, value, 0.0, False, 64)
    assert stub.last(3)[3] == [x.data_ptr(), value % 2**64, value % 2**64]


def test_int_one_is_folded_into_the_key(built, launch):
    module, stub, _ = built
    x = torch.zeros(4)
    launch(0, 0, 1, x, 1, 9, 0.0, False, 64)
    assert stub.last(2)[3] == [x.data_ptr(), 9]


def test_spec_key_buckets(built, launch):
    module, _, _ = built
    x = torch.zeros(64)

    def key(*args):
        return launch.__self__._C_spec_key(0, *args)[0]

    ints = [key(x, v, 0, 0.0, False, 64) for v in (17, 16, 2**31, 2**63, 1)]
    assert len(set(ints)) == len(ints)
    assert key(x, 17, 0, 0.0, False, 64) == key(x, 19, 0, 0.0, False, 64)
    assert key(x, 17, 16, 0.0, False, 64) == key(
        x, 17, 17, 0.0, False, 64
    )  # do_not_specialize
    assert key(x, 17, 0, 0.0, False, 64) != key(x.half(), 17, 0, 0.0, False, 64)
    assert key(x, 17, 0, 0.0, False, 64) != key(
        x[1:], 17, 0, 0.0, False, 64
    )  # alignment
    assert key(x, 17, 0, 0.0, False, 64) != key(x, 17, 0, 0.0, False, 128)
    assert key(x, 17, 0, 0.0, False, 2**64 - 1) != key(x, 17, 0, 0.0, False, 2**63 - 1)


def test_bad_arguments(built, launch):
    module, stub, _ = built
    x = torch.zeros(4)
    with pytest.raises(TypeError):
        launch(0, 0, 1, x, 5, 0, 0.0, False)
    with pytest.raises(TypeError, match="unsupported argument 'n'"):
        launch(0, 0, 1, x, "5", 0, 0.0, False, 64)
    with pytest.raises(TypeError, match="device"):
        launch(256, 0, 1, x, 5, 0, 0.0, False, 64)
    for grid in ((1, 1, 1, 1), [], [1, 1, 1, 1], [-1], [2**32], [None]):
        with pytest.raises(ValueError, match="grid"):
            launch(0, 0, grid, x, 5, 0, 0.0, False, 64)
    calls = stub.calls()
    assert launch(0, 0, (4, 0), x, 5, 0, 0.0, False, 64) is None
    assert stub.calls() == calls  # an empty grid launches nothing


@pytest.mark.skipif(
    not sysconfig.get_config_var("Py_GIL_DISABLED"), reason="requires free threading"
)
def test_list_grid_survives_concurrent_resize(built, stub_module):
    """A list resize must not invalidate the grid parser's borrowed item array."""
    module, _, _ = built
    layout = layout_for() if module.__name__ == "rt_runtime_shim" else None
    # Isolate a potential native crash from the rest of the suite.
    code = f"""
import importlib.util
import sys
import threading
import torch
spec = importlib.util.spec_from_file_location({module.__name__!r}, {module.__file__!r})
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module._set_torch_version({torch_version()!r}, {layout.as_args() if layout else None!r},
                         {bytes(c if c < 32 else 0xFF for c in range(NDTYPES))!r})
spec = importlib.util.spec_from_file_location("_intj_lazy", {stub_module.__file__!r})
lazy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lazy)
module._set_launcher_type(lazy.Launcher)
build = lambda header: (module._init(header, print), None, b"")
launch = lazy.new_launcher(type("L", (lazy.Launcher,), {{"__slots__": (), "_build_body": build}}), None)
assert not sys._is_gil_enabled()
grid = [0, 1000, 1000]
stop = threading.Event()
started = threading.Event()
def resize():
    started.set()
    while not stop.is_set():
        grid[:] = []
        grid[:] = [0, 1000, 1000]
thread = threading.Thread(target=resize)
thread.start()
started.wait()
try:
    for _ in range(200_000):
        try:
            # Every valid snapshot has zero volume, so no kernel is needed.
            assert launch(0, 0, grid, None, 5, 0, 0.0, False, 64) is None
        except ValueError as error:
            assert "between 1 and 3 dimensions" in str(error)
finally:
    stop.set()
    thread.join()
"""
    result = subprocess.run(
        [sys.executable, "-X", "faulthandler", "-c", code],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_launch_failure_is_reported(built, launch):
    module, _, _ = built
    with pytest.raises(RuntimeError, match="stub_launch failed: stub failure"):
        launch(0, 0, 1, torch.zeros(4), 5, 0, 0.0, False, 999)


def test_unreadable_tensor_names_the_argument(built, launch):
    """The reader's error is re-raised naming the argument, with its own as the cause.

    Goes through `PyErr_GetRaisedException`, which is a shim below 3.12.
    """
    module, _, _ = built
    if module.__name__ != "rt_interpreter":
        pytest.skip("only the interpreter tensor reader calls into torch")
    sparse = torch.zeros(4).to_sparse()
    with pytest.raises(RuntimeError, match="tensor argument 'x'") as info:
        launch(0, 0, 1, sparse, 5, 0, 0.0, False, 64)
    assert info.value.__cause__ is not None


def test_concurrent_launches(built, launch):
    """Cold fills and hits from many threads at once all reach the launch.

    With the GIL this is interleaving only; on a free-threaded build it is the
    cache lock's test.
    """
    module, stub, _ = built
    x = torch.zeros(4)
    nthreads, iters = 8, 400
    errors = []
    barrier = threading.Barrier(nthreads)

    def work(t):
        try:
            barrier.wait()
            for i in range(iters):
                launch(0, 0, 1, x, 5, t, 0.0, False, 1000 + (i * 7 + t) % 64)
        except BaseException as e:  # noqa: BLE001 - reported below
            errors.append(e)

    before = stub.calls()
    threads = [threading.Thread(target=work, args=(t,)) for t in range(nthreads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert stub.calls() - before == nthreads * iters


def test_cpython_layer_matches_this_python():
    """The header `cpython_abi` selects reads what python itself reports.

    The table in `cpython_abi` lists versions this check passed on; this keeps the
    running one honest.  Other versions: `python -m intj.python_intf.check`.
    """
    from intj.python_intf.check import check

    assert check() == []


def test_verified_python_versions_share_one_header():
    headers = {cpython_abi.header_for((3, minor)) for minor in range(8, 15)}
    assert None not in headers and len(headers) == 1


def test_load_stub_reports_a_missing_compiler_as_unsupported(tmp_path_factory):
    """A missing host compiler raises `UnsupportedKernel`, not a subprocess error.

    Final review minor 3: a decoration-time build must never raise a bare
    subprocess error -- refusals are loud, per AGENTS.md.
    """
    root = tmp_path_factory.mktemp("lazy_missing_cc")
    with pytest.raises(launcher.UnsupportedKernel, match="host compiler") as excinfo:
        lazy.load_stub(root, "intj-test-nonexistent-cc")
    assert isinstance(excinfo.value.__cause__, OSError)


def test_load_stub_reports_a_compile_failure_as_unsupported(
    tmp_path_factory, monkeypatch
):
    def fail(args, **kwargs):
        return subprocess.CompletedProcess(
            args, 1, stdout=b"", stderr=b"intj_lazy.c:1:1: error: bad token\n"
        )

    monkeypatch.setattr(subprocess, "run", fail)
    root = tmp_path_factory.mktemp("lazy_failing_cc")
    with pytest.raises(launcher.UnsupportedKernel, match="bad token") as excinfo:
        lazy.load_stub(root, _cc("c"))
    assert isinstance(excinfo.value.__cause__, subprocess.CalledProcessError)


# Here rather than in test_launcher.py: needing neither triton nor a GPU, it
# runs in ci's test-cpu and on interpreters triton has no wheel for.
_NO_TRITON = '''
class NoTriton:
    """Fails every search for triton, and counts them."""

    searches = 0

    @classmethod
    def find_spec(cls, name, path=None, target=None):
        if name != "triton":
            return None
        cls.searches += 1
        raise ModuleNotFoundError("No module named 'triton'", name="triton")


sys.meta_path.insert(0, NoTriton)
checks = lambda: NoTriton.searches
'''

_OLD_TRITON = """
reads = 0


def __getattr__(name):  # counts the version checks
    global reads
    if name != "__version__":
        raise AttributeError(name)
    reads += 1
    return "3.5.0"
"""


@pytest.mark.parametrize("found", [None, "3.5.0"])
def test_make_launcher_checks_triton_once(tmp_path, found):
    """Without Triton >=3.7 every `make_launcher` call refuses; only the first looks.

    Missing, triton is searched for once, and every refusal chains the
    import's ModuleNotFoundError; old (a stub package), its `__version__` is
    read once.
    """
    if found is None:
        setup = _NO_TRITON
    else:
        (tmp_path / "triton").mkdir()
        (tmp_path / "triton" / "__init__.py").write_text(_OLD_TRITON)
        setup = "checks = lambda: sys.modules['triton'].reads"
    probe = f"""
import json, sys
{setup}
import intj
from intj.launcher import UnsupportedKernel

refusals = []
for _ in range(2):
    try:
        intj.make_launcher(object())
    except UnsupportedKernel as error:
        refusals.append([str(error), type(error.__cause__).__name__])
print(json.dumps([refusals, checks()]))
"""
    package_root = os.path.dirname(os.path.dirname(launcher.__file__))
    run = subprocess.run(
        [sys.executable, "-c", probe],
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join([str(tmp_path), package_root]),
        },
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert run.returncode == 0, run.stderr
    message = "intj: Triton >=3.7 is required; install `intj[launcher]`"
    if found is None:
        refusal = [message, "ModuleNotFoundError"]  # the cause, chained
    else:
        refusal = [f"{message} (found {found})", "NoneType"]
    assert json.loads(run.stdout) == [[refusal, refusal], 1]


@pytest.fixture(scope="module")
def stub_module(tmp_path_factory):
    return lazy.load_stub(tmp_path_factory.mktemp("lazy"), _cc("c"))


_CALL = (0, 0, 1)  # device, stream, grid


def _interner():
    ids = {}
    return lambda v: ids.setdefault(v, len(ids)) if type(v) is str else None


def _built(address, options=None):
    # a `_build_body` result: the entry, the options to publish, no signature
    return address, options, b""


def _init_launcher(module, holder, state, interner):
    address = module._init(holder, interner)
    holder._compile_state = state
    return address


def _new_launcher(stub_module, options, build, *, compile_miss=None):
    # a launcher whose type's `_build_body` is `build(launcher)`
    namespace = {"__slots__": ("_compile_state",), "_build_body": build}
    if compile_miss is not None:
        namespace["_compile_miss"] = compile_miss
    cls = type("L", (launcher.Launcher, stub_module.Launcher), namespace)
    return stub_module.new_launcher(cls, options)


@pytest.fixture(scope="module")
def launch(built, stub_module):
    """A built launcher over the fixture module: what `module.entry` used to be."""
    module, _, _ = built
    fn = _new_launcher(
        stub_module,
        None,
        lambda header: _built(
            _init_launcher(
                module,
                header,
                launcher._CompileState(module._intj_compiler),
                _interner(),
            )
        ),
    )
    fn(*_args())
    return fn


def _args():
    return (*_CALL, torch.zeros(4), 5, -7, 1.5, True, 64)


def test_compile_method_is_called_only_on_misses(built, stub_module, monkeypatch):
    module, driver, _ = built
    launch = _new_launcher(
        stub_module,
        None,
        lambda h: _built(
            _init_launcher(
                module, h, launcher._CompileState(module._intj_compiler), _interner()
            )
        ),
    )
    args = _args()
    launch(*args)
    cls = type(launch.__self__)
    original = getattr(cls, "_compile_miss", None)
    assert original is not None, "the C miss path must dispatch through Launcher"

    def refuse(self, *call):
        raise RuntimeError("compile method refused the miss")

    monkeypatch.setattr(cls, "_compile_miss", refuse)
    calls = driver.calls()
    assert launch(*args) is None
    assert driver.calls() == calls + 1
    missed = (*args[:-1], 128)
    with pytest.raises(RuntimeError, match="compile method refused"):
        launch(*missed)
    assert driver.calls() == calls + 1
    monkeypatch.setattr(cls, "_compile_miss", original)
    assert launch(*missed) is None
    assert driver.last(5)[:3] == ((1, 1, 1), 128, _CALL[1])


def test_compile_state_referent_cycle_is_collected(built, stub_module):
    module, _, _ = built

    class Owner:
        launch: Callable[..., object]

    owner = Owner()
    state = launcher._CompileState(module._intj_compiler, seen={b"owner": owner})
    launch = _new_launcher(
        stub_module,
        None,
        lambda h: _built(_init_launcher(module, h, state, _interner())),
    )
    launch(*_args())
    owner.launch = launch
    assert launch.__self__._compile_state is state
    ref = weakref.ref(owner)
    del owner, state, launch
    gc.collect()
    assert ref() is None


def test_stub_stays_out_of_sys_modules_and_matches_the_header(built, stub_module):
    module, _, _ = built
    assert "_intj_lazy" not in sys.modules
    assert stub_module._header_size() == module._header_size()


def test_first_call_builds_and_swaps_for_old_references(built, stub_module):
    module, _, _ = built
    builds = []

    def build(header):
        builds.append(1)
        return _built(
            _init_launcher(
                module,
                header,
                launcher._CompileState(module._intj_compiler),
                _interner(),
            )
        )

    launch = _new_launcher(stub_module, None, build)
    before = launch  # a reference taken before the build
    function = ctypes.pythonapi.PyCFunction_GetFunction
    function.argtypes, function.restype = [ctypes.py_object], ctypes.c_void_p
    shim = function(launch)
    assert launch.__self__._C_module is None and builds == []
    args = _args()
    assert launch(*args) is None
    assert builds == [1] and function(before) != shim
    assert launch.__self__._C_module is module and launch.__self__._C_build() is module
    a, b, c, d, e, f, g, h, i = args

    def call():
        # explicit positionals: `before(*args)` is CALL_FUNCTION_EX, never specialized
        return before(a, b, c, d, e, f, g, h, i)

    for _ in range(100):
        assert call() is None
    assert builds == [1]
    free_threaded = bool(sysconfig.get_config_var("Py_GIL_DISABLED"))
    if sys.version_info >= (3, 11) and not (
        free_threaded and sys.version_info[:2] == (3, 13)
    ):
        names = {i.opname for i in dis.get_instructions(call, adaptive=True)}  # pyright: ignore[reportCallIssue]
        assert names & {
            "PRECALL_NO_KW_BUILTIN_FAST",
            "CALL_NO_KW_BUILTIN_FAST",
            "CALL_BUILTIN_FAST",
        }


def test_failed_build_raises_and_the_next_call_retries(built, stub_module):
    module, _, _ = built
    attempts = []

    def build(header):
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("first build fails")
        return _built(
            _init_launcher(
                module,
                header,
                launcher._CompileState(module._intj_compiler),
                _interner(),
            )
        )

    launch = _new_launcher(stub_module, None, build)
    with pytest.raises(RuntimeError, match="first build fails"):
        launch(*_args())
    assert launch(*_args()) is None and len(attempts) == 2


def test_concurrent_first_calls_build_once(built, stub_module):
    module, _, _ = built
    builds = []

    def build(header):
        builds.append(1)
        time.sleep(0.05)  # hold the build lock while the others arrive
        return _built(
            _init_launcher(
                module,
                header,
                launcher._CompileState(module._intj_compiler),
                _interner(),
            )
        )

    launch = _new_launcher(stub_module, None, build)
    barrier, results, errors = threading.Barrier(8), [], []

    def worker():
        barrier.wait()
        try:
            results.append(launch(*_args()))
        except BaseException as error:  # noqa: BLE001  report, do not hang
            errors.append(error)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == [] and results == [None] * 8 and builds == [1]


def test_call_during_build_is_refused(built, stub_module):
    module, _, _ = built
    seen = []

    def build(header):
        try:
            launch(*_args())
        except RuntimeError as error:
            seen.append(str(error))
        return _built(
            _init_launcher(
                module,
                header,
                launcher._CompileState(module._intj_compiler),
                _interner(),
            )
        )

    launch = _new_launcher(stub_module, None, build)
    assert launch(*_args()) is None
    assert seen and "while it was being built" in seen[0]


@pytest.mark.parametrize("subclass", [False, True], ids=["base", "subclass"])
def test_unbuilt_and_built_launchers_are_collected(built, stub_module, subclass):
    module, _, _ = built

    class Owner:
        launch: Callable[..., object]

    # the bare base has no `_build_body`, so it is only ever unbuilt
    for call_first in (False, True) if subclass else (False,):
        owner = Owner()

        # Python 3.13t immortalizes classes after the first thread starts.
        # Keep the owner on the instance so the class cannot retain it.
        def build(header):
            return _built(
                _init_launcher(
                    module,
                    header,
                    launcher._CompileState(module._intj_compiler),
                    _interner(),
                ),
                header._C_options,
            )

        # The options hold the owner before and after the build, forming a cycle.
        owner.launch = (
            _new_launcher(stub_module, owner, build)
            if subclass
            else stub_module.new_launcher(stub_module.Launcher, owner)
        )
        if call_first:
            owner.launch(*_args())
        ref = weakref.ref(owner)
        del owner, build
        gc.collect()
        assert ref() is None, f"leaked (called first: {call_first})"


def test_memo_referent_cycle_is_collected(built, stub_module):
    """The memo's object can reach its own launcher; GC must see the memo."""
    module, _, _ = built

    class Value:
        launch: Callable[..., object]

    def compile_cb(_key, nparams, _device, *_args):  # keeps no argument
        return (0x1000 + nparams, 128, 0, nparams)

    value = Value()
    value.launch = _new_launcher(
        stub_module,
        None,
        lambda h: _built(
            _init_launcher(
                module,
                h,
                launcher._CompileState(module._intj_compiler),
                lambda v: 0 if type(v) is Value else None,
            )
        ),
        compile_miss=lambda self, *call: compile_cb(*call),
    )
    value.launch(*_CALL, torch.zeros(4), 5, -7, 1.5, True, value)  # memoized
    ref = weakref.ref(value)
    del value
    gc.collect()
    assert ref() is None


def test_str_constexpr_keys_by_value(built, stub_module):
    module, _, _ = built
    launch = _new_launcher(
        stub_module,
        None,
        lambda h: _built(
            _init_launcher(
                module, h, launcher._CompileState(module._intj_compiler), _interner()
            )
        ),
    )
    launch(*_args())
    x = torch.zeros(4)

    def key(value):
        return launch.__self__._C_spec_key(0, x, 5, -7, 1.5, True, value)[0]

    assert key("act") == key("".join(["a", "c", "t"])) != key("other")
    assert key(0) != key("act")  # an int never shares an object's code


def test_memo_holds_its_object_until_replaced(built, stub_module):
    module, _, _ = built
    launch = _new_launcher(
        stub_module,
        None,
        lambda h: _built(
            _init_launcher(
                module, h, launcher._CompileState(module._intj_compiler), _interner()
            )
        ),
    )
    first, second = "".join(["o", "n", "e"]), "".join(["t", "w", "o"])
    launch(*_CALL, torch.zeros(4), 5, -7, 1.5, True, first)
    held = sys.getrefcount(first)
    launch(*_CALL, torch.zeros(4), 5, -7, 1.5, True, second)
    assert sys.getrefcount(first) == held - 1  # the memo moved and let go
    launch(*_CALL, torch.zeros(4), 5, -7, 1.5, True, first)
    assert sys.getrefcount(first) == held
    with pytest.raises(TypeError, match="unsupported argument 'BLOCK' of type list"):
        launch(*_CALL, torch.zeros(4), 5, -7, 1.5, True, [1])


def test_launchers_come_only_from_new_launcher(stub_module):
    with pytest.raises(TypeError, match="come from make_launcher"):
        stub_module.Launcher()

    class Sub(stub_module.Launcher):
        __slots__ = ()

    with pytest.raises(TypeError, match="come from make_launcher"):
        Sub()
    for cls in (object, int):
        with pytest.raises(TypeError, match="new_launcher"):
            stub_module.new_launcher(cls, None)
    with pytest.raises(TypeError, match="new_launcher"):
        stub_module.new_launcher(Sub, None, print)  # no builder argument
    assert type(stub_module.new_launcher(Sub, None).__self__) is Sub


@pytest.mark.parametrize("slots", [("extra",), None], ids=["slots", "dict"])
def test_launcher_subclass_fields_survive_build_and_are_collected(
    built, stub_module, slots
):
    module, driver, compiles = built

    def build_body(self):
        return _built(
            _init_launcher(
                module, self, launcher._CompileState(module._intj_compiler), _interner()
            )
        )

    namespace: dict[str, object] = {"_build_body": build_body}
    if slots is not None:
        namespace["__slots__"] = ("_compile_state", *slots)
    cls = type("L", (launcher.Launcher, stub_module.Launcher), namespace)
    launch = stub_module.new_launcher(cls, None)

    class Owner:
        launch: Callable[..., object]

    owner = Owner()
    owner.launch = launch
    launch.__self__.extra = owner
    ref = weakref.ref(owner)
    args = _args()
    calls, misses = driver.calls(), len(compiles)
    assert launch(*args) is None
    assert launch(*args) is None
    assert driver.calls() == calls + 2 and len(compiles) == misses + 1
    assert launch.__self__.extra is owner
    assert launch.__self__._C_module is module
    with pytest.raises(TypeError, match="unbuilt launcher"):
        module._init(launch.__self__, _interner())
    del owner, launch
    gc.collect()
    assert ref() is None


def test_init_rejects_objects_with_unrelated_c_layouts(built, stub_module):
    module, _, _ = built
    words = (stub_module._header_size() - object.__basicsize__) // struct.calcsize("P")
    same = type("SameSize", (), {"__slots__": tuple(f"slot{i}" for i in range(words))})
    larger = type("Larger", (same,), {"__slots__": ("extra",)})
    assert same.__basicsize__ == stub_module._header_size()
    for value in (object(), same(), larger()):
        with pytest.raises(TypeError, match="unbuilt launcher"):
            module._init(value, _interner())


def test_launcher_type_cannot_be_replaced(built, tmp_path):
    module, _, _ = built
    other = lazy.load_stub(tmp_path, _cc("c"))
    with pytest.raises(RuntimeError, match="launcher type already installed"):
        module._set_launcher_type(other.Launcher)


def test_init_rejects_a_launcher_from_another_stub(built, tmp_path):
    module, _, _ = built
    other = lazy.load_stub(tmp_path, _cc("c"))
    cls = type("Other", (other.Launcher,), {"__slots__": ()})
    launch = other.new_launcher(cls, None)
    with pytest.raises(TypeError, match="unbuilt launcher"):
        module._init(launch.__self__, _interner())
    assert launch.__self__._C_module is None


def test_locked_runs_python_under_the_build_lock(stub_module):
    launch = _new_launcher(stub_module, "before", print)
    h = launch.__self__
    assert (h._C_options, h._C_module, h._C_built, launch.__text_signature__) == (
        "before",
        None,
        False,
        None,
    )
    with pytest.raises(RuntimeError, match="build lock"):
        h._C_set_options("unlocked")
    assert h._C_locked(lambda: h._C_set_options(h._C_options + "!")) is None
    assert h._C_options == "before!"
    with pytest.raises(KeyError):
        h._C_locked(lambda: {}["x"])  # fn's error, and the lock released
    assert h._C_locked(lambda: 7) == 7
    with pytest.raises(RuntimeError, match="while it was being built"):
        h._C_locked(lambda: h._C_locked(print))  # no re-entry
    assert h._C_options == "before!"


def test_build_body_must_return_entry_options_and_doc(stub_module):
    for result, error in (
        (7, TypeError),
        ((7, "o"), TypeError),
        ((7, "o", "not bytes"), TypeError),
        ((0, "o", b""), RuntimeError),  # a NULL entry
    ):
        launch = _new_launcher(stub_module, "before", lambda h, r=result: r)
        with pytest.raises(error):
            launch()
        assert launch.__self__._C_options == "before"  # nothing published
    bare = stub_module.new_launcher(stub_module.Launcher, "before")
    with pytest.raises(AttributeError, match="_build_body"):
        bare()
    assert not bare.__self__._C_built


def test_build_publishes_options_and_signature_and_refuses_edits(built, stub_module):
    module, _, _ = built
    launch = _new_launcher(
        stub_module,
        "unbuilt",
        lambda h: (
            _init_launcher(
                module, h, launcher._CompileState(module._intj_compiler), _interner()
            ),
            "resolved",
            b"launch(a, /)\n--\n\n",
        ),
    )
    h = launch.__self__
    assert launch.__text_signature__ is None
    launch(*_args())
    assert h._C_options == "resolved" and launch.__text_signature__ == "(a, /)"
    assert h._C_built
    with pytest.raises(RuntimeError, match="fixed"):
        h._C_locked(lambda: h._C_set_options("again"))
    assert h._C_options == "resolved"


def test_options_reads_race_edits_safely(stub_module):
    launchers = [_new_launcher(stub_module, "before", print) for _ in range(200)]
    seen, stop = set(), threading.Event()

    def read():
        while not stop.is_set():
            for k in launchers:
                seen.add(k.__self__._C_options)

    readers = [threading.Thread(target=read) for _ in range(4)]
    for t in readers:
        t.start()
    for k in launchers:
        h = k.__self__
        h._C_locked(lambda: h._C_set_options("after"))
    stop.set()
    for t in readers:
        t.join()
    assert seen <= {"before", "after"}
    assert all(k.__self__._C_options == "after" for k in launchers)


def test_key_chain_of_an_ordinary_launcher_is_its_spec_key(launch):
    """One level: the chain is the spec key, found once that key launched."""
    h = launch.__self__
    public = _args()[len(_CALL) :]
    launch(*_CALL, *public)
    keys, found = h._C_key_chain(_CALL[0], *public)
    assert keys == (h._C_spec_key(_CALL[0], *public)[0],)
    assert found is True


def test_ordinary_miss_gets_stream_and_grid_controls(built, stub_module):
    """Every module misses through one call: stream and grid controls included."""
    module, _, _ = built
    calls = []

    def compile_miss(self, keyblob, nparams, device, stream, controls, *args):
        calls.append((device, stream, controls, len(args)))
        return launcher.Launcher._compile_miss(
            self, keyblob, nparams, device, stream, controls, *args
        )

    launch = _new_launcher(
        stub_module,
        None,
        lambda h: _built(
            _init_launcher(
                module, h, launcher._CompileState(module._intj_compiler), _interner()
            )
        ),
        compile_miss=compile_miss,
    )
    args = _args()
    assert launch(*args) is None
    assert calls == [(_CALL[0], _CALL[1], (1, 1, 1), len(args) - len(_CALL))]

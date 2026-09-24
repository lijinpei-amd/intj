# pyright: standard
"""The rendered module, end to end, without triton or a GPU.

Builds the real `entry.c.jinja` render with the system compiler against a stub
driver that records each launch, so it runs on any CPython intj supports, with
nothing but torch installed.  That is what makes the python-version matrix
testable: triton 3.8 has no wheel below 3.10.  See `tests/run_python_matrix.sh`.

The c++ access mode is left to `test_launcher.py`: its toolchain comes from triton.
"""

from __future__ import annotations

import ctypes
import dataclasses
import importlib.util
import os
import struct
import subprocess
import sys
import sysconfig
import threading

import jinja2
import pytest
import torch

from intj import launcher
from intj.annotation import CanonicalAnnotation, DeviceBinding, ResolvedParam
from intj.launcher import RenderContext
from intj.python_intf import cpython_abi
from intj.torch_intf.torch_abi import NDTYPES, TorchAccessMode, layout_for, torch_version

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
_AUTO = CanonicalAnnotation("argument", None, "auto", "auto", "auto", (), False, None, ())
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

_MODES = [TorchAccessMode.INTERPRETER] + ([TorchAccessMode.RUNTIME_SHIM] if layout_for() else [])


def _cc(language: str) -> str:
    return os.environ.get("CXX", "c++") if language == "c++" else os.environ.get("CC", "cc")


@pytest.fixture(scope="module")
def stub(tmp_path_factory):
    out = tmp_path_factory.mktemp("stub")
    src, lib = out / "stub.c", out / "stub.so"
    src.write_text(_STUB)
    subprocess.run([_cc("c"), "-shared", "-fPIC", "-O2", str(src), "-o", str(lib)], check=True)
    return ctypes.CDLL(str(lib), mode=ctypes.RTLD_GLOBAL), str(lib)


class Stub:
    def __init__(self, lib):
        self.lib = lib

    def calls(self) -> int:
        return ctypes.c_long.in_dll(self.lib, "intj_stub_calls").value

    def last(self, nparams: int):
        """(grid, block_dim, stream, params) of the last launch."""
        launch = list((ctypes.c_uint64 * 5).in_dll(self.lib, "intj_stub_launch"))
        params = list((ctypes.c_uint64 * 16).in_dll(self.lib, "intj_stub_params"))
        return tuple(launch[:3]), launch[3], launch[4], params[:nparams]


@pytest.fixture(scope="module", params=_MODES, ids=lambda m: m.value)
def built(request, stub, tmp_path_factory):
    """(module, stub, compile callback calls) for one access mode."""
    lib, lib_path = stub
    mode = request.param
    context = RenderContext(
        module_name=f"rt_{mode.value}", kernel_repr="test_runtime.kernel", params=_PARAMS,
        nwords=_NWORDS, device_binding=DeviceBinding.NOT_FIXED, device_offset=_DEVICE_OFFSET,
        verify_annotation=False, no_gpu=False,
        max_slots=5, spec_pointer_range=1, driver_path=lib_path,
        launch_symbol="stub_launch", device_symbol="", error_symbol="stub_error", error_style="return",
        torch_access_mode=mode.value, kernel_cache="intj", cache_include_dirs=(), cache_archives=(),
        torch_version=None, cxx_abi=None,
        python_version=cpython_abi.python_version(),
        free_threaded=bool(sysconfig.get_config_var("Py_GIL_DISABLED")),
        cpython_static_compile_header=cpython_abi.header_for() or "",
    )
    flags = launcher._build_flags(context)
    template = jinja2.Template(launcher._ENTRY_TEMPLATE.read_text(), undefined=jinja2.StrictUndefined)
    out = tmp_path_factory.mktemp(mode.value)
    src = out / "entry.c"
    so = out / f"{context.module_name}{sysconfig.get_config_var('EXT_SUFFIX')}"
    src.write_text(template.render(**dataclasses.asdict(context)))
    subprocess.run(
        [
            _cc(flags["language"]), "-shared", "-fPIC", "-O2", "-Wall", "-Werror",
            "-Wno-unused-but-set-variable", "-x", flags["language"], *flags["ccflags"],
            *(f"-I{d}" for d in [*flags["include_dirs"], sysconfig.get_paths()["include"]]),
            str(src), "-o", str(so),
        ],
        check=True,
    )
    spec = importlib.util.spec_from_file_location(context.module_name, so)
    module = importlib.util.module_from_spec(spec)  # pyright: ignore[reportArgumentType]
    spec.loader.exec_module(module)  # pyright: ignore

    compiles = []

    def compile_cb(_key, nparams, _device, *args):
        compiles.append(args[-1])
        return (0xBAD if args[-1] == 999 else 0x1000 + nparams, 128, 0, nparams)

    module.set_compile_callback(compile_cb)
    layout = layout_for().as_args() if mode is TorchAccessMode.RUNTIME_SHIM else None  # pyright: ignore[reportOptionalMemberAccess]
    # every torch code its own 5-bit index; the real one comes from triton
    module.set_torch_version(torch_version(), layout, bytes(c if c < 32 else 0xFF for c in range(NDTYPES)))
    return module, Stub(lib), compiles


def _f32(v: float) -> int:
    return struct.unpack("<I", struct.pack("<f", v))[0]


@pytest.mark.parametrize("head, version, expected", [
    (16, (2, 9), 8),
    (32, (2, 9), 8),
    (16, (2, 10), 0),
    (32, (2, 10), 0),
    (65535, (2, 9), None),
    (65535, (2, 10), 0),
    (65536, (2, 10), None),
])
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


def test_runtime_shim_rejects_rebased_cdata_overflow(built):
    module, stub, _ = built
    if module.__name__ != "rt_runtime_shim":
        pytest.skip("only RUNTIME_SHIM installs cdata")
    layout = layout_for()
    assert layout is not None
    x = torch.arange(8, dtype=torch.float32)
    args = (0, 0, 1, x, 5, 0, 0.0, False, 64)
    module.entry(*args)
    assert stub.last(5)[3][0] == x.data_ptr()

    relative = (1 << 16) - cpython_abi.pyobject_size()
    assert 0 <= relative <= 65535
    bad = (relative, *layout.as_args()[1:])
    index = bytes(c if c < 32 else 0xFF for c in range(NDTYPES))
    with pytest.raises(ValueError, match="16-bit"):
        module.set_torch_version(torch_version(), bad, index)

    module.entry(*args)
    assert stub.last(5)[3][0] == x.data_ptr()


def test_torch_abi_is_installed_once(built):
    module, _, _ = built
    layout = layout_for() if module.__name__ == "rt_runtime_shim" else None
    index = bytes(c if c < 32 else 0xFF for c in range(NDTYPES))
    with pytest.raises(RuntimeError, match="already installed"):
        module.set_torch_version(torch_version(), layout.as_args() if layout else None, index)


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
        module.set_torch_version(torch_version(), oversized, index)


def test_gil_stays_off_on_a_free_threaded_build(built):
    if not sysconfig.get_config_var("Py_GIL_DISABLED"):
        pytest.skip("default build")
    assert not sys._is_gil_enabled()  # pyright: ignore[reportAttributeAccessIssue]


@pytest.mark.parametrize("grid", [(2, 3), [2, 3]])
def test_arguments_reach_the_launch(built, grid):
    module, stub, compiles = built
    x = torch.arange(8, dtype=torch.float32)
    module.entry(0, 77, grid, x, 5, -7, 1.5, True, 64)
    grid, block, stream, params = stub.last(5)
    assert (grid, block, stream) == ((2, 3, 1), 128, 77)
    assert params == [x.data_ptr(), 5, 2**64 - 7, _f32(1.5), 1]
    before = len(compiles)
    module.entry(0, 0, 4, x, 5, -7, 1.5, True, 64)
    assert len(compiles) == before  # a hit: no second compile
    assert stub.last(0)[0] == (4, 1, 1)


@pytest.mark.parametrize("value", [
    0, 16, 17, -17, 2**30 - 1, 2**30, -(2**30), 2**31, 2**60 + 3, 2**63 - 1, -(2**63),
    2**63, 2**64 - 1, 2**64, -(2**63) - 1, 2**90, 2**100,
])
def test_int_values(built, value):
    module, stub, _ = built
    x = torch.zeros(4)
    if not -(2**63) <= value < 2**64:
        with pytest.raises(OverflowError):
            module.entry(0, 0, 1, x, value, 0, 0.0, False, 64)
        return
    module.entry(0, 0, 1, x, value, value, 0.0, False, 64)
    assert stub.last(3)[3] == [x.data_ptr(), value % 2**64, value % 2**64]


def test_int_one_is_folded_into_the_key(built):
    module, stub, _ = built
    x = torch.zeros(4)
    module.entry(0, 0, 1, x, 1, 9, 0.0, False, 64)
    assert stub.last(2)[3] == [x.data_ptr(), 9]


def test_spec_key_buckets(built):
    module, _, _ = built
    x = torch.zeros(64)

    def key(*args):
        return module.spec_key(*args)[0]

    ints = [key(x, v, 0, 0.0, False, 64) for v in (17, 16, 2**31, 2**63, 1)]
    assert len(set(ints)) == len(ints)
    assert key(x, 17, 0, 0.0, False, 64) == key(x, 19, 0, 0.0, False, 64)
    assert key(x, 17, 16, 0.0, False, 64) == key(x, 17, 17, 0.0, False, 64)  # do_not_specialize
    assert key(x, 17, 0, 0.0, False, 64) != key(x.half(), 17, 0, 0.0, False, 64)
    assert key(x, 17, 0, 0.0, False, 64) != key(x[1:], 17, 0, 0.0, False, 64)  # alignment
    assert key(x, 17, 0, 0.0, False, 64) != key(x, 17, 0, 0.0, False, 128)
    assert key(x, 17, 0, 0.0, False, 2**64 - 1) != key(x, 17, 0, 0.0, False, 2**63 - 1)


def test_bad_arguments(built):
    module, stub, _ = built
    x = torch.zeros(4)
    with pytest.raises(TypeError):
        module.entry(0, 0, 1, x, 5, 0, 0.0, False)
    with pytest.raises(TypeError, match="unsupported argument 'n'"):
        module.entry(0, 0, 1, x, "5", 0, 0.0, False, 64)
    with pytest.raises(TypeError, match="device"):
        module.entry(256, 0, 1, x, 5, 0, 0.0, False, 64)
    for grid in ((1, 1, 1, 1), [], [1, 1, 1, 1], [-1], [2**32], [None]):
        with pytest.raises(ValueError, match="grid"):
            module.entry(0, 0, grid, x, 5, 0, 0.0, False, 64)
    calls = stub.calls()
    assert module.entry(0, 0, (4, 0), x, 5, 0, 0.0, False, 64) is None
    assert stub.calls() == calls  # an empty grid launches nothing


@pytest.mark.skipif(not sysconfig.get_config_var("Py_GIL_DISABLED"), reason="requires free threading")
def test_list_grid_survives_concurrent_resize(built):
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
module.set_torch_version({torch_version()!r}, {layout.as_args() if layout else None!r},
                         {bytes(c if c < 32 else 0xFF for c in range(NDTYPES))!r})
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
            assert module.entry(0, 0, grid, None, 5, 0, 0.0, False, 64) is None
        except ValueError as error:
            assert "between 1 and 3 dimensions" in str(error)
finally:
    stop.set()
    thread.join()
"""
    result = subprocess.run([sys.executable, "-X", "faulthandler", "-c", code],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr


def test_launch_failure_is_reported(built):
    module, _, _ = built
    with pytest.raises(RuntimeError, match="stub_launch failed: stub failure"):
        module.entry(0, 0, 1, torch.zeros(4), 5, 0, 0.0, False, 999)


def test_unreadable_tensor_names_the_argument(built):
    """The reader's error is re-raised naming the argument, with its own as the cause.

    Goes through `PyErr_GetRaisedException`, which is a shim below 3.12.
    """
    module, _, _ = built
    if module.__name__ != "rt_interpreter":
        pytest.skip("only the interpreter tensor reader calls into torch")
    sparse = torch.zeros(4).to_sparse()
    with pytest.raises(RuntimeError, match="tensor argument 'x'") as info:
        module.entry(0, 0, 1, sparse, 5, 0, 0.0, False, 64)
    assert info.value.__cause__ is not None


def test_concurrent_launches(built):
    """Cold fills and hits from many threads at once.  With the GIL this is
    interleaving only; on a free-threaded build it is the cache lock's test."""
    module, stub, _ = built
    x = torch.zeros(4)
    nthreads, iters = 8, 400
    errors = []
    barrier = threading.Barrier(nthreads)

    def work(t):
        try:
            barrier.wait()
            for i in range(iters):
                module.entry(0, 0, 1, x, 5, t, 0.0, False, 1000 + (i * 7 + t) % 64)
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

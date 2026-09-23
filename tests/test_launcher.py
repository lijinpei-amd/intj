# pyright: standard
"""Differential tests: every launch is compared against triton's own launcher.

Run with `pytest tests` on a machine with an AMD GPU, torch and triton.
"""

from __future__ import annotations

import ctypes
import dataclasses
import json
import pathlib
import subprocess
import sysconfig
import types

import pytest
import torch
import triton
import triton.language as tl

from intj import launcher, make_launcher
from intj.launcher import (
    ModuleKey,
    Param,
    RenderContext,
    UnsupportedKernel,
    triton_specialization,
)
from intj.kernel_cache import INSTALL_ERRORS, KernelCache, install
from intj.python_intf import cpython_abi
from intj.torch_intf.abi_detect import probe_layout
from intj.torch_intf.torch_abi import (
    NDTYPES,
    TorchAccess,
    dtype_code,
    dtype_index_table,
    itemsize_table,
    layout_for,
)


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
def scale(x, o, n, s, BLOCK: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = off < n
    tl.store(o + off, tl.load(x + off, mask=mask) * s, mask=mask)


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
    check_matches_triton(axpy, axpy_launcher, grid, (x, y, o, n, 1.5, flag, bias, 128), 2)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16, torch.int32])
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
    check_matches_triton(scale, scale_launcher, (8,), (base[:1024], o, 1024, 2.0, 128), 1)


@pytest.mark.parametrize("block", [64, 128, 256])
def test_constexpr_variants(scale_launcher, block):
    x = torch.randn(1024, device="cuda")
    o = torch.empty(1024, device="cuda")
    check_matches_triton(scale, scale_launcher, (triton.cdiv(1024, block),), (x, o, 1024, 2.0, block), 1)


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
    check_matches_triton(matmul, launcher, (m // 64, n // 64), (a, b, c, m, n, k, 64), 2)


def test_zero_volume_grid_does_not_launch(scale_launcher):
    x = torch.randn(1024, device="cuda")
    o = torch.zeros(1024, device="cuda")
    launch(scale_launcher, (0,), x, o, 1024, 2.0, 128)
    torch.testing.assert_close(o, torch.zeros_like(o))


def test_options_are_baked_in():
    launcher = make_launcher(scale, options={"num_warps": 8})  # freshly loaded: see below
    x = torch.randn(1024, device="cuda")
    o = torch.empty(1024, device="cuda")
    kernel_cache = scale.device_caches[torch.cuda.current_device()][0]
    kernel_cache.clear()
    launch(launcher, (8,), x, o, 1024, 2.0, 128)
    torch.testing.assert_close(o, x * 2.0)
    assert {k.metadata.num_warps for k in kernel_cache.values()} == {8}


def test_launchers_of_one_kernel_stay_independent():
    """Two launchers share a C symbol and a module name, so they must not share a module.

    The symbol is the kernel's name, for legible `perf` output; the digest in
    the artifact path is what keeps the two builds apart.
    """
    # num_warps values no other test uses, so both modules are loaded here and
    # their kernel caches are cold; a warm one would not compile anything.
    narrow = getattr(make_launcher(scale, options={"num_warps": 2}), "__self__")
    wide = getattr(make_launcher(scale, options={"num_warps": 16}), "__self__")
    assert narrow.__name__ == wide.__name__ == "scale"  # one name ...
    assert narrow is not wide  # ... two modules, told apart by their file
    assert narrow.__file__ != wide.__file__

    x = torch.randn(1024, device="cuda")
    o = torch.empty(1024, device="cuda")
    kernel_cache = scale.device_caches[torch.cuda.current_device()][0]
    for launcher, num_warps in ((narrow.entry, 2), (wide.entry, 16)):
        kernel_cache.clear()
        launch(launcher, (8,), x, o, 1024, 2.0, 128)
        torch.testing.assert_close(o, x * 2.0)
        assert {k.metadata.num_warps for k in kernel_cache.values()} == {num_warps}


def test_one_module_per_key():
    """A repeat `make_launcher` reuses the loaded module, callback included.

    Installing a second callback would drop the first, freeing the
    `CompiledKernel`s whose function handles the C kernel cache still holds.
    """
    first = make_launcher(scale, options={"num_stages": 3})
    second = make_launcher(scale, options={"num_stages": 3})
    assert getattr(first, "__self__") is getattr(second, "__self__")
    assert first == second


def test_cached_build_is_reused_without_rendering():
    """A dict miss with the .so already on disk must not render or compile again."""
    import intj.launcher as launcher_module

    make_launcher(scale, options={"num_stages": 5})
    launcher_module._LOADED.clear()

    def fail(*args, **kwargs):
        raise AssertionError("rebuilt a module that was already on disk")

    original, launcher_module._build = launcher_module._build, fail
    try:
        make_launcher(scale, options={"num_stages": 5})
    finally:
        launcher_module._build = original


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
    first = getattr(make_launcher(scale, options={"num_stages": 6}), "__self__")
    assert first.__spec__.name not in sys.modules

    launcher_module._LOADED.clear()
    second = getattr(make_launcher(scale, options={"num_stages": 6}), "__self__")
    assert second.__file__ == first.__file__  # same .so ...
    assert second is not first  # ... but a module of its own

    reference = weakref.ref(second)
    del second
    launcher_module._LOADED.clear()
    gc.collect()
    assert reference() is None


def test_source_and_binary_are_cached_on_disk():
    launcher = make_launcher(scale, options={"num_stages": 4}, torch_access=TorchAccess.SHIM)
    so_path = pathlib.Path(getattr(launcher, "__self__").__file__)
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
    module = getattr(axpy_launcher, "__self__")
    base = torch.randn(64, device="cuda")
    values = [0, 1, 16, 17, -16, -17, 2**31 - 1, 2**31, 2**63 - 1, 2**63,
              2**64 - 16, 2**64 - 1, -(2**63), -(2**63) - 1, 2**64, 2**100]

    seen = {}
    for value in values:
        args = (base, base, base, value, 1.5, True, None, 128)
        try:
            spec = repr(triton_specialization(axpy, args)[3])
        except OverflowError:
            with pytest.raises(OverflowError):
                module.spec_key(*args)
            continue

        key, _ = module.spec_key(*args)
        previous = seen.setdefault(key, (spec, value))
        assert previous[0] == spec, f"{previous[1]} and {value} share a key, {previous[0]} != {spec}"


def test_spec_key_is_never_coarser_than_triton(axpy_launcher):
    """The load-bearing invariant: same intj key => same triton specialization.

    A coarser key is the failure mode that does not crash -- it launches, say, a
    tt.divisibility=16 binary on an unaligned pointer.
    """
    module = axpy_launcher.__self__
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

    seen = {}
    for args in cases:
        key, nparams = module.spec_key(*args)
        spec = triton_specialization(axpy, args)
        previous = seen.setdefault((key, nparams), (repr(spec), args))
        assert previous[0] == repr(spec), (
            f"intj key collides for two different triton specializations:\n"
            f"  {previous[1]} -> {previous[0]}\n  {args} -> {spec}"
        )
        assert nparams == sum(1 for ty, _ in spec if ty != "constexpr")


@pytest.mark.parametrize(
    "nparams,nconstexpr", [(1, 0), (1, 1), (6, 0), (7, 0), (8, 0), (7, 2), (8, 3)]
)
def test_render_params_byte_layout(tmp_path, nparams, nconstexpr):
    """Byte i is parameter i; the device byte is last; constexpr words follow.

    The 7-vs-8 cases straddle the header word boundary -- seven parameters plus
    the device byte fill one word exactly, eight need a second.  Every kernel in
    this suite has at most eight parameters, so nothing else here would catch a
    device-byte off-by-one.
    """
    import importlib.util

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

    params, nwords = _render_params(module.k)
    header_words = -(-(nparams + nconstexpr + 1) // 8)

    assert [p.name for p in params] == names + cxnames
    assert [p.cx_word for p in params if not p.is_constexpr] == [None] * nparams
    assert [p.cx_word for p in params if p.is_constexpr] == [
        header_words + i for i in range(nconstexpr)
    ]
    assert nwords == header_words + nconstexpr


def _render_context(**overrides):
    fields = dict(
        module_name="m", kernel_repr="a.b",
        params=(
            Param(name="x", is_constexpr=False, spec=1, align=1, cx_word=None),
            Param(name="BLOCK", is_constexpr=True, spec=1, align=1, cx_word=1),
        ),
        nwords=2, header_words=1, max_slots=1, spec_pointer_range=1, driver_path="/driver.so",
        launch_symbol="launch", error_symbol="error", error_style="return",
        torch_access="shim", torch_version=None, cxx_abi=None,
        kernel_cache="intj", cache_include_dirs=(), cache_archives=(),
        python_version=(3, 12, 3), free_threaded=False, python_abi="cpython_312.h",
    )
    return RenderContext(**{**fields, **overrides})


def _module_key(**overrides):
    fields = dict(
        template="aa", runtime_header="bb", context=_render_context(module_name=""), cache_key="c",
        params=(("x", "", False, False, False, False, False, "None"),),
        target=("hip", "gfx942", 64), options="o", triton=("3.8.0", 1, 2.0),
        compiler=("gcc", "13"), ext_suffix=".so",
    )
    return ModuleKey(**{**fields, **overrides})


@pytest.mark.parametrize(
    "build,differing",
    [(_render_context, {"nwords": 5}), (_module_key, {"cache_key": "d"})],
)
def test_value_types_compare_hash_and_serialize(build, differing):
    """Both carry only immutable fields, so they behave as values.

    A list field anywhere in either would compare fine and raise on hash().
    """
    one, same, other = build(), build(), build(**differing)
    assert one == same and one != other
    assert hash(one) == hash(same) and hash(one) != hash(other)
    assert len({one: 1, same: 2, other: 3}) == 2
    dump = lambda v: json.dumps(dataclasses.asdict(v), sort_keys=True)  # noqa: E731
    assert dump(one) == dump(same) and dump(one) != dump(other)


def test_module_key_digest_tracks_every_field():
    key = _module_key()
    for field in dataclasses.fields(key):
        value = getattr(key, field.name)
        changed = "zz" if isinstance(value, str) else (value + 1 if isinstance(value, int) else None)
        if changed is None:
            continue
        assert dataclasses.replace(key, **{field.name: changed}).digest() != key.digest(), field.name


def test_render_context_digest_tracks_every_field():
    """`context` is one field of ModuleKey, so the loop above never varies its parts."""
    base = _module_key()
    for field in dataclasses.fields(base.context):
        value = getattr(base.context, field.name)
        if isinstance(value, bool):
            changed = not value
        elif isinstance(value, str):
            changed = "zz"
        elif isinstance(value, int):
            changed = value + 1
        else:
            changed = (1, 2)
        if value == changed:
            continue
        other = dataclasses.replace(base, context=dataclasses.replace(base.context, **{field.name: changed}))
        assert other.digest() != base.digest(), field.name


def test_free_threaded_build_changes_module_digest():
    key = _module_key()
    free_threaded = dataclasses.replace(key.context, free_threaded=True)
    assert dataclasses.replace(key, context=free_threaded).digest() != key.digest()


def test_render_rejects_mismatched_free_threaded_headers(tmp_path, capfd):
    context = _render_context(
        python_version=cpython_abi.python_version(),
        python_abi=cpython_abi.header_for() or "",
        free_threaded=not bool(sysconfig.get_config_var("Py_GIL_DISABLED")),
    )
    with pytest.raises(subprocess.CalledProcessError):
        launcher._build(tmp_path / "m.so", context)
    assert "intj: compiling against headers of the wrong GIL build" in capfd.readouterr().err


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
    module = getattr(make_launcher(nested), "__self__")
    path = pathlib.Path(module.__file__)
    assert path.name.startswith("inner.")
    assert path.parent.name == nested.__module__
    assert len(path.parent.parent.name) == 64  # the ModuleKey digest

    x = torch.randn(1024, device="cuda")
    o = torch.empty(1024, device="cuda")
    launch(module.entry, (8,), x, o, 1024, 128)
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

    tsl and absl are C++ maps, so they also drag a shim-mode module into a C++
    build -- which is the part most likely to break.
    """
    _provisioned(cache)
    launcher = make_launcher(scale, torch_access=TorchAccess.SHIM, kernel_cache=cache)
    x = torch.randn(1024, device="cuda")
    o = torch.empty(1024, device="cuda")
    for block in (64, 128):  # two specializations, so the cache is used
        check_matches_triton(scale, launcher, (triton.cdiv(1024, block),), (x, o, 1024, 2.0, block), 1)


def test_kernel_cache_choice_reaches_the_digest():
    modules = {
        c: pathlib.Path(getattr(make_launcher(scale, kernel_cache=_provisioned(c)), "__self__").__file__)
        for c in CACHES
    }
    digests = {c: path.parent.parent.name for c, path in modules.items()}
    assert len(set(digests.values())) == len(CACHES)


def test_kernel_cache_is_installed_on_demand(monkeypatch):
    """An unprovisioned backend installs itself, and its toolchain is what builds."""
    calls: list[KernelCache] = []
    toolchain = {"include_dirs": ("/nowhere/include",), "library_dirs": (), "archives": ()}
    monkeypatch.setattr("intj.launcher.toolchain_for", lambda cache: None)
    monkeypatch.setattr("intj.launcher.install", lambda c: calls.append(c) or toolchain)

    # stop before the real build: the point is what reaches it, not that a
    # fabricated include directory compiles
    contexts: list[RenderContext] = []
    monkeypatch.setattr(
        "intj.launcher._loaded_module",
        lambda key, fn, context, *rest: contexts.append(context) or types.SimpleNamespace(entry=None),
    )
    make_launcher(scale, kernel_cache=KernelCache.TSL)
    # the requested backend, and what install returned actually reaches the build
    assert calls == [KernelCache.TSL]
    assert contexts[-1].cache_include_dirs == toolchain["include_dirs"]


def test_kernel_cache_install_failure_is_reported_once(monkeypatch):
    """A failed install refuses with the reason and the retry command, and is not re-tried."""
    attempts = []

    def explode(cache):
        attempts.append(cache)
        raise OSError("no network")

    monkeypatch.setattr("intj.launcher.toolchain_for", lambda cache: None)
    monkeypatch.setattr("intj.launcher.install", explode)
    monkeypatch.setattr("intj.launcher._INSTALL_FAILED", {})
    for _ in range(3):
        with pytest.raises(UnsupportedKernel, match="python -m intj.kernel_cache tsl.*no network"):
            make_launcher(scale, kernel_cache=KernelCache.TSL)
    assert attempts == [KernelCache.TSL]  # remembered, not re-attempted


def test_kernel_cache_install_bug_is_not_dressed_up_as_a_download_failure(monkeypatch):
    monkeypatch.setattr("intj.launcher.toolchain_for", lambda cache: None)
    monkeypatch.setattr("intj.launcher._INSTALL_FAILED", {})
    monkeypatch.setattr("intj.launcher.install", lambda cache: cache.no_such_attribute)
    with pytest.raises(AttributeError):
        make_launcher(scale, kernel_cache=KernelCache.TSL)


def test_kernel_cache_install_is_the_last_refusal(monkeypatch):
    """A kernel that will be rejected anyway must not pay for a download first."""
    monkeypatch.setattr("intj.launcher.toolchain_for", lambda cache: None)

    def explode(cache):
        raise AssertionError("installed before refusing an unknown option")

    monkeypatch.setattr("intj.launcher.install", explode)
    with pytest.raises(UnsupportedKernel, match="unknown compile option"):
        make_launcher(scale, options={"nonsense": 1}, kernel_cache=KernelCache.ABSL)


def test_refuses_non_jit_function():
    with pytest.raises(UnsupportedKernel):
        make_launcher(lambda: None)


def test_refuses_kernel_reading_globals():
    with pytest.raises(UnsupportedKernel, match="global variable"):
        make_launcher(uses_global)


GLOBAL_SCALE = 2.0


@triton.jit
def uses_global(x, o, n, BLOCK: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = off < n
    tl.store(o + off, tl.load(x + off, mask=mask) * GLOBAL_SCALE, mask=mask)


# ---------------------------------------------------------------- torch access

ACCESS_MODES = [TorchAccess.SHIM, TorchAccess.CPYTHON, TorchAccess.CXX]


@pytest.fixture(scope="module", params=ACCESS_MODES, ids=lambda m: m.name.lower())
def mode(request):
    return request.param


@pytest.fixture(scope="module")
def scale_by_mode(mode):
    return mode, make_launcher(scale, torch_access=mode)


def _read_corpus():
    """Tensors whose pointer/dtype/storage-size the three modes must agree on.

    The empty views matter most: `Tensor::data_ptr()` returns null for any
    zero-element tensor even when its storage is live and its storage_offset is
    not, so the shim's `data + offset * itemsize` arithmetic has to special-case
    them or it keys differently from the other two modes.
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
    module = getattr(launcher, "__self__")
    o = torch.empty(4096, device="cuda")
    seen = {}
    for x in _read_corpus():
        args = (x, o, 1024, 2.0, 128)
        key = module.spec_key(*args)
        spec = repr(triton_specialization(scale, args))
        assert seen.setdefault(key, spec) == spec, f"key collides for {x.dtype} {x.shape}"


def test_modes_agree_on_every_read():
    """The independent-oracle test: cpython assumes nothing about TensorImpl.

    If torch moves a field, the shim mode keeps reading the old offset and this
    is what notices.
    """
    modules = {
        m: getattr(make_launcher(scale, torch_access=m), "__self__") for m in ACCESS_MODES
    }
    o = torch.empty(4096, device="cuda")
    for x in _read_corpus():
        keys = {m: mod.spec_key(x, o, 1024, 2.0, 128) for m, mod in modules.items()}
        distinct = set(keys.values())
        assert len(distinct) == 1, f"modes disagree on {x.dtype} numel={x.numel()}: {keys}"


@pytest.mark.parametrize("dtype_name", ["float32", "float16", "bfloat16", "int32", "uint8"])
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
    `CPYTHON` on a torch intj thinks it supports looks like before it happens.
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
        (2, 9): ({f: i for i, f in enumerate(OFFSETS)}, {0: ("uint8", 1), 1: ("int8", 1)})
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
    """The template's accumulator/shift arithmetic, checked against a real key.

    `test_render_params_byte_layout` pins the python half.  This pins the C
    half: each parameter ORs its code into accumulator `i // 4` at shift
    `(i % 4) * 8`, and the accumulators fold pairwise into one store per word.
    The counts straddle both boundaries -- 4 bytes per accumulator and 8 per
    word -- which is where an off-by-one would hide.
    """
    import importlib.util

    from intj.launcher import _render_params

    names = [f"p{i}" for i in range(nparams)]
    path = tmp_path / f"k{nparams}.py"
    path.write_text(
        "import triton\nimport triton.language as tl\n\n\n@triton.jit\n"
        f"def k({', '.join(names)}, C: tl.constexpr):\n    pass\n"
    )
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    kernel = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kernel)  # pyright: ignore[reportAttributeAccessIssue]  # 3.8 stubs lack it

    params, nwords = _render_params(kernel.k)
    header_words = -(-(nparams + 2) // 8)
    module = getattr(make_launcher(kernel.k), "__self__")

    b_i32 = 128  # INTJ_B_I32, +1 when the value is divisible by 16
    # odd, above 1, not divisible by 16: every byte must be the plain i32 code
    args = [3 + 2 * i for i in range(nparams)] + [256]
    blob, np_ = module.spec_key(*args)

    assert (len(blob), nwords, np_) == (nwords * 8, header_words + 1, nparams)
    assert list(blob[:nparams]) == [b_i32] * nparams
    assert blob[nparams] == 140  # INTJ_B_CX_INT
    assert blob[nparams + 1] == 0  # the device byte; spec_key keys on device 0
    assert set(blob[nparams + 2:header_words * 8]) <= {0}  # padding memcmp reads
    at = header_words * 8
    assert int.from_bytes(blob[at:at + 8], "little") == 256

    # flipping one parameter must move exactly its own byte and nothing else
    for i in range(nparams):
        alt = list(args)
        alt[i] = 16  # divisible by 16 -> tt.divisibility, so +1
        other, _ = module.spec_key(*alt)
        assert other[i] == b_i32 + 1
        assert [b for j, b in enumerate(other) if j != i] == [
            b for j, b in enumerate(blob) if j != i
        ]


def test_unsupported_dtype_is_refused_in_every_mode():
    """The path that skipped the dtype check: numel == 0 returns before it.

    The shim reader bails at numel == 0 ahead of its own bound test, and the
    cpython reader has no bound test at all.  Harmless while the dtype had 32
    bits of the key to itself; under a byte code every unmapped dtype would
    alias every other one.  So the check lives in the decode, which all three
    readers pass through.
    """
    o = torch.empty(4096, device="cuda")
    for mode in ACCESS_MODES:
        module = getattr(make_launcher(scale, torch_access=mode), "__self__")
        for x in (torch.zeros(4, device="cuda", dtype=torch.complex64),
                  torch.zeros(0, device="cuda", dtype=torch.complex64)):
            with pytest.raises(RuntimeError, match="triton does not take"):
                module.spec_key(x, o, 1024, 2.0, 128)


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
    assert table[dtype_code(torch.bool)] == table[dtype_code(torch.uint1)]
    assert table[dtype_code(torch.int1)] == table[dtype_code(torch.bool)]
    # and a dtype triton has never accepted must be refused
    assert table[dtype_code(torch.complex64)] == 0xFF


def test_dtype_index_table_is_deterministic():
    """Same torch, same triton, same table -- a module keys on it."""
    dtype_index_table.cache_clear()
    first = dtype_index_table()
    dtype_index_table.cache_clear()
    assert dtype_index_table() == first


def test_layout_probe_reproduces_torch():
    """Every offset the shim mode reads, checked against torch's own accessors."""
    layout = layout_for()
    assert layout is not None, "probe failed on a torch intj is expected to support"
    assert len(layout.itemsize) == NDTYPES
    for x in _read_corpus() + [torch.arange(9, dtype=torch.float64)[2:]]:
        impl = ctypes.c_size_t.from_address(id(x) + layout.cdata).value
        assert impl == x._cdata
        assert ctypes.c_int64.from_address(impl + layout.numel).value == x.numel()
        assert ctypes.c_int64.from_address(impl + layout.storage_offset).value == x.storage_offset()
        assert ctypes.c_uint8.from_address(impl + layout.data_type).value == dtype_code(x.dtype)


def test_launch_correctness_per_mode(scale_by_mode):
    _, launcher = scale_by_mode
    x = torch.randn(1024, device="cuda")
    o = torch.empty(1024, device="cuda")
    check_matches_triton(scale, launcher, (8,), (x, o, 1024, 2.0, 128), 1)
    check_matches_triton(scale, launcher, (8,), (x[1:], o, 1023, 2.0, 128), 1)


def test_auto_resolves_and_explicit_modes_validate():
    from intj.launcher import _resolve_access

    assert _resolve_access(TorchAccess.AUTO) in set(ACCESS_MODES)
    for m in ACCESS_MODES:
        assert _resolve_access(m) is m


def test_hardcoded_layout_matches_this_torch():
    """The table is data, and data goes stale.  Check the entry against reality.

    `probe_layout` finds each offset by matching field values against torch's own
    accessors, so it is an independent oracle for the hardcoded row -- and the
    thing that generates entries in the first place (`python -m intj.torch_intf.abi_detect`).
    """
    table, probed = layout_for(), probe_layout()
    assert table is not None, "no entry for the torch the suite is running against"
    assert probed is not None, "probe could not pin this torch's layout"
    assert table == probed


def test_hardcoded_layout_matches_torch_headers():
    """The same entry, checked by the other detector: the compiler places each field.

    Element sizes are compared by code only; the spellings are python's, and the
    headers know none of them.  Codes the headers declare past the recorded ones
    must be sizeless placeholders, or the table is missing a dtype.
    """
    from intj.launcher import _cxx_toolchain
    from intj.torch_intf.cpp_detect import detect
    from intj.torch_intf.torch_abi import OFFSETS

    if _cxx_toolchain() is None:
        pytest.skip("no C++ compiler and torch headers")
    table = layout_for()
    assert table is not None, "no entry for the torch the suite is running against"
    offsets, sizes = detect()
    assert offsets == {f: getattr(table, f) for f in OFFSETS}
    assert {c: s for c, s in sizes.items() if s} == {c: table.itemsize[c] for c in range(NDTYPES) if table.itemsize[c]}


def test_shim_is_refused_rather_than_guessed(monkeypatch):
    """An unverified torch must raise, never fall back to a guessed offset."""
    from intj import launcher as launcher_mod

    monkeypatch.setattr(launcher_mod, "layout_for", lambda *a: None)
    with pytest.raises(UnsupportedKernel, match="tensor layout"):
        make_launcher(scale, torch_access=TorchAccess.SHIM)


def test_cxx_needs_no_table(monkeypatch):
    """The compiler supplies CXX's offsets, so an unverified torch still gets it."""
    from intj import launcher as launcher_mod

    monkeypatch.setattr(launcher_mod, "layout_for", lambda *a: None)
    x = torch.arange(128, device="cuda", dtype=torch.float32)
    o = torch.empty_like(x)
    check_matches_triton(scale, make_launcher(scale, torch_access=TorchAccess.CXX), (1,), (x, o, 128, 2.0, 128), 1)


def test_only_the_cxx_module_is_keyed_on_the_torch_version(monkeypatch):
    """shim and cpython bake in nothing torch-specific, so their `.so` is reusable.

    Moving the reported torch version must not move their digest -- and must move
    the c++ one, which really did compile against those headers.
    """
    from intj import launcher as launcher_mod

    def digest_dir(m):
        return pathlib.Path(getattr(make_launcher(scale, torch_access=m), "__self__").__file__).parent.parent.name

    before = {m: digest_dir(m) for m in ACCESS_MODES}
    monkeypatch.setattr(launcher_mod, "torch_version", lambda: (99, 99))
    after = {m: digest_dir(m) for m in ACCESS_MODES}

    assert after[TorchAccess.SHIM] == before[TorchAccess.SHIM]
    assert after[TorchAccess.CPYTHON] == before[TorchAccess.CPYTHON]
    assert after[TorchAccess.CXX] != before[TorchAccess.CXX]


def test_unconfigured_module_refuses_to_launch():
    """`entry` is unreachable before set_torch_version; prove the guard exists."""
    module = getattr(make_launcher(scale, torch_access=TorchAccess.SHIM), "__self__")
    assert hasattr(module, "set_torch_version")
    index = dtype_index_table()
    with pytest.raises(ValueError, match="needs a tensor layout"):
        module.set_torch_version((2, 14), None, index)
    layout = layout_for()
    assert layout is not None
    cpython = getattr(make_launcher(scale, torch_access=TorchAccess.CPYTHON), "__self__")
    with pytest.raises(ValueError, match="takes no tensor layout"):
        cpython.set_torch_version((2, 14), layout.as_args(), index)


def test_a_failed_set_torch_version_changes_nothing():
    """A rejected call must not leave a half-installed table behind.

    The method is reachable from python on a module that is already configured
    and whose kernel cache already holds entries.  Installing the dtype table
    before validating the rest meant a call that raised still committed it --
    and an all-zero table keys every dtype to index 0, so a float16 binary runs
    on float32 memory with no error at all.  Silent, and the worst failure this
    design has.
    """
    module = getattr(make_launcher(scale, torch_access=TorchAccess.SHIM), "__self__")
    o = torch.empty(64, device="cuda")

    def keys():
        return {
            d: module.spec_key(torch.zeros(64, device="cuda", dtype=d), o, 64, 2.0, 128)
            for d in (torch.float32, torch.float16)
        }

    before = keys()
    assert len(set(before.values())) == 2, "two dtypes must key apart to begin with"

    good = layout_for()
    assert good is not None
    index = dtype_index_table()
    for layout, table in (
        (None, bytes(NDTYPES)),  # shim rejects a None layout...
        (None, index),  # ...whichever table comes with it
        (good.as_args(), b"\xff" * (NDTYPES - 1)),  # wrong length
        (good.as_args(), bytes([32]) + index[1:]),  # an index past five bits
    ):
        with pytest.raises(ValueError):
            module.set_torch_version((2, 14), layout, table)
        # per case, not once at the end: each rejection has its own way of
        # committing early, and one masks the next
        assert keys() == before


def test_set_torch_version_needs_the_dtype_index_in_every_mode():
    """The dtype table is not a shim detail: all three readers key on it.

    Installed in only some modes, the others would see an all-zero table and key
    every dtype to index 0 -- coarse in the same way in each, so the cross-mode
    equality test would not catch it.
    """
    layout = layout_for()
    assert layout is not None
    for mode in ACCESS_MODES:
        module = getattr(make_launcher(scale, torch_access=mode), "__self__")
        args = layout.as_args() if mode is TorchAccess.SHIM else None
        with pytest.raises(TypeError, match="dtype_index"):
            module.set_torch_version((2, 14), args)
        with pytest.raises(ValueError, match="dtype index table"):
            module.set_torch_version((2, 14), args, b"\xff" * (NDTYPES - 1))


def test_modes_agree_on_rejecting_a_storageless_tensor():
    """A sparse tensor has no data pointer at all, so every mode must raise.

    The shim reads offsets rather than calling torch, so this is the one place it
    could have handed the kernel a null pointer instead of failing.
    """
    sparse = torch.sparse_coo_tensor(
        torch.tensor([[0, 1]]), torch.tensor([1.0, 2.0]), (4,)
    ).cuda()
    o = torch.empty(1024, device="cuda")
    for m in ACCESS_MODES:
        module = getattr(make_launcher(scale, torch_access=m), "__self__")
        with pytest.raises(RuntimeError):
            module.spec_key(sparse, o, 1024, 2.0, 128)


def test_modes_agree_on_tensors_with_unusual_impls():
    """The shim cannot see the bitfields that make torch's accessors throw.

    `storage_access_should_throw_`, `throw_on_immutable_data_ptr_` and
    `size_bytes_is_heap_allocated_` are invisible to offset arithmetic, so the
    shim could in principle accept a tensor the c++ mode rejects.  In practice
    the impls carrying them belong to `Tensor` *subclasses*, which INTJ_DECODE's
    exact-type test already refuses -- this pins that reasoning.
    """
    modules = {m: getattr(make_launcher(scale, torch_access=m), "__self__") for m in ACCESS_MODES}
    o = torch.empty(1024, device="cuda")

    with torch.inference_mode():
        inference = torch.randn(1024, device="cuda")
    cases = {"inference": inference, "meta": torch.zeros(1024, device="meta")}

    for label, t in cases.items():
        keys = {m: mod.spec_key(t, o, 1024, 2.0, 128) for m, mod in modules.items()}
        assert len(set(keys.values())) == 1, f"{label}: {keys}"

    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode() as fake_mode:
        fake = fake_mode.from_tensor(torch.randn(1024, device="cuda"))
    for m, mod in modules.items():
        with pytest.raises(TypeError):  # a subclass: refused before any read
            mod.spec_key(fake, o, 1024, 2.0, 128)


def test_custom_sizes_policy_tensors_are_a_known_limitation():
    """Nested tensors read as empty, identically in every mode.  Documented, not fixed.

    A nested tensor stores `numel_ == 0` and reports `numel() == 8` through a
    virtual override.  `shim` cannot see the policy bit that says so, and `cxx`
    reads the field too rather than diverge from it -- one documented limitation
    beats two modes that disagree.  See `docs/Usage.md`.

    This pins the behaviour so that a torch change, or a decision to start
    detecting these, shows up here instead of silently.
    """
    from intj.torch_intf.abi_detect import _read

    layout = layout_for()
    assert layout is not None
    nested = torch.nested.nested_tensor([torch.randn(3), torch.randn(5)])
    assert nested.numel() == 8 and nested.data_ptr() != 0
    assert _read(layout, nested)[0] == 0, "expected the known-wrong null pointer"

    # mkldnn has no storage at all, and that *is* detected
    mkl = torch.randn(4, 4).to_mkldnn()
    with pytest.raises(RuntimeError):
        _read(layout, mkl)
    o = torch.empty(16, device="cuda")
    for m in ACCESS_MODES:
        module = getattr(make_launcher(scale, torch_access=m), "__self__")
        with pytest.raises(RuntimeError):
            module.spec_key(mkl, o, 16, 2.0, 128)


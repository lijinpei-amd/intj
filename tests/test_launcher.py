# pyright: standard
"""Differential tests: every launch is compared against triton's own launcher.

Run with `pytest tests` on a machine with an AMD GPU, torch and triton.
"""

import ctypes
import dataclasses
import json
import pathlib

import pytest
import torch
import triton
import triton.language as tl

from intj import create_launcher
from intj.launcher import (
    ModuleKey,
    Param,
    RenderContext,
    UnsupportedKernel,
    triton_specialization,
)
from intj.torch_abi import NDTYPES, TorchAccess, cached_layout, dtype_code, itemsize_table


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
    return create_launcher(axpy)


@pytest.fixture(scope="module")
def scale_launcher():
    return create_launcher(scale)


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
    launcher = create_launcher(scale_nospec)
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
    launcher = create_launcher(write_program_ids)
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
    launcher = create_launcher(matmul)
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
    launcher = create_launcher(scale, options={"num_warps": 8})  # freshly loaded: see below
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
    narrow = getattr(create_launcher(scale, options={"num_warps": 2}), "__self__")
    wide = getattr(create_launcher(scale, options={"num_warps": 16}), "__self__")
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
    """A repeat `create_launcher` reuses the loaded module, callback included.

    Installing a second callback would drop the first, freeing the
    `CompiledKernel`s whose function handles the C kernel cache still holds.
    """
    first = create_launcher(scale, options={"num_stages": 3})
    second = create_launcher(scale, options={"num_stages": 3})
    assert getattr(first, "__self__") is getattr(second, "__self__")
    assert first == second


def test_cached_build_is_reused_without_rendering():
    """A dict miss with the .so already on disk must not render or compile again."""
    import intj.launcher as launcher_module

    create_launcher(scale, options={"num_stages": 5})
    launcher_module._LOADED.clear()

    def fail(*args, **kwargs):
        raise AssertionError("rebuilt a module that was already on disk")

    original, launcher_module._build = launcher_module._build, fail
    try:
        create_launcher(scale, options={"num_stages": 5})
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
    first = getattr(create_launcher(scale, options={"num_stages": 6}), "__self__")
    assert first.__spec__.name not in sys.modules

    launcher_module._LOADED.clear()
    second = getattr(create_launcher(scale, options={"num_stages": 6}), "__self__")
    assert second.__file__ == first.__file__  # same .so ...
    assert second is not first  # ... but a module of its own

    reference = weakref.ref(second)
    del second
    launcher_module._LOADED.clear()
    gc.collect()
    assert reference() is None


def test_source_and_binary_are_cached_on_disk():
    launcher = create_launcher(scale, options={"num_stages": 4}, torch_access=TorchAccess.SHIM)
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


def _render_context(**overrides):
    fields = dict(
        module_name="m", kernel_repr="a.b",
        params=(Param("x", False, 1, 1, 1), Param("BLOCK", True, 1, 1, 2)),
        nwords=4, max_slots=1, spec_pointer_range=1, driver_path="/driver.so",
        launch_symbol="launch", error_symbol="error", error_style="return",
        torch_access="shim", torch_version=None, cxx_abi=None,
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
        changed = "zz" if isinstance(value, str) else (value + 1 if isinstance(value, int) else (1, 2))
        if value == changed:
            continue
        other = dataclasses.replace(base, context=dataclasses.replace(base.context, **{field.name: changed}))
        assert other.digest() != base.digest(), field.name


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
    module = getattr(create_launcher(nested), "__self__")
    path = pathlib.Path(module.__file__)
    assert path.name.startswith("inner.")
    assert path.parent.name == nested.__module__
    assert len(path.parent.parent.name) == 64  # the ModuleKey digest

    x = torch.randn(1024, device="cuda")
    o = torch.empty(1024, device="cuda")
    launch(module.entry, (8,), x, o, 1024, 128)
    torch.testing.assert_close(o, x * 2.0)


def test_refuses_non_jit_function():
    with pytest.raises(UnsupportedKernel):
        create_launcher(lambda: None)


def test_refuses_kernel_reading_globals():
    with pytest.raises(UnsupportedKernel, match="global variable"):
        create_launcher(uses_global)


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
    return mode, create_launcher(scale, torch_access=mode)


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
        m: getattr(create_launcher(scale, torch_access=m), "__self__") for m in ACCESS_MODES
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
    assert 0 <= code < NDTYPES
    assert itemsize_table()[code] == torch.empty(0, dtype=dtype).element_size()


def test_layout_probe_reproduces_torch():
    """Every offset the shim mode reads, checked against torch's own accessors."""
    layout = cached_layout()
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


def test_shim_is_refused_rather_than_guessed(monkeypatch):
    """A layout intj cannot pin must raise, never fall back to a guessed offset."""
    from intj import launcher as launcher_mod

    monkeypatch.setattr(launcher_mod, "cached_layout", lambda: None)
    with pytest.raises(UnsupportedKernel, match="tensor layout"):
        create_launcher(scale, torch_access=TorchAccess.SHIM)


def test_only_the_cxx_module_is_keyed_on_the_torch_version(monkeypatch):
    """shim and cpython bake in nothing torch-specific, so their `.so` is reusable.

    Moving the reported torch version must not move their digest -- and must move
    the c++ one, which really did compile against those headers.
    """
    from intj import launcher as launcher_mod

    def digest_dir(m):
        return pathlib.Path(getattr(create_launcher(scale, torch_access=m), "__self__").__file__).parent.parent.name

    before = {m: digest_dir(m) for m in ACCESS_MODES}
    monkeypatch.setattr(launcher_mod, "torch_version", lambda: (99, 99))
    after = {m: digest_dir(m) for m in ACCESS_MODES}

    assert after[TorchAccess.SHIM] == before[TorchAccess.SHIM]
    assert after[TorchAccess.CPYTHON] == before[TorchAccess.CPYTHON]
    assert after[TorchAccess.CXX] != before[TorchAccess.CXX]


def test_unconfigured_module_refuses_to_launch():
    """`entry` is unreachable before set_torch_version; prove the guard exists."""
    module = getattr(create_launcher(scale, torch_access=TorchAccess.SHIM), "__self__")
    assert hasattr(module, "set_torch_version")
    with pytest.raises(ValueError, match="needs a tensor layout"):
        module.set_torch_version((2, 14), None)
    layout = cached_layout()
    assert layout is not None
    cpython = getattr(create_launcher(scale, torch_access=TorchAccess.CPYTHON), "__self__")
    with pytest.raises(ValueError, match="takes no tensor layout"):
        cpython.set_torch_version((2, 14), layout.as_args())


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
        module = getattr(create_launcher(scale, torch_access=m), "__self__")
        with pytest.raises(RuntimeError):
            module.spec_key(sparse, o, 1024, 2.0, 128)

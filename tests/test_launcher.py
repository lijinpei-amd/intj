# pyright: standard
"""Differential tests: every launch is compared against triton's own launcher.

Run with `pytest tests` on a machine with an AMD GPU, torch and triton.
"""

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
    _symbol_name,
    to_json,
    triton_specialization,
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
    launcher = create_launcher(scale, options={"num_warps": 8})
    x = torch.randn(1024, device="cuda")
    o = torch.empty(1024, device="cuda")
    kernel_cache = scale.device_caches[torch.cuda.current_device()][0]
    kernel_cache.clear()
    launch(launcher, (8,), x, o, 1024, 2.0, 128)
    torch.testing.assert_close(o, x * 2.0)
    assert {k.metadata.num_warps for k in kernel_cache.values()} == {8}


def test_launchers_of_one_kernel_stay_independent():
    """Two launchers share a C symbol, so they must not share a module.

    The symbol is the kernel's name, for legible `perf` output; the digest in
    the path and the spec name is what keeps the two builds apart.
    """
    default = getattr(create_launcher(scale), "__self__")
    wide = getattr(create_launcher(scale, options={"num_warps": 8}), "__self__")
    assert default.__name__.rsplit(".", 1)[-1] == wide.__name__.rsplit(".", 1)[-1] == "scale"
    assert default.__name__ != wide.__name__
    assert default is not wide
    assert default.__file__ != wide.__file__

    x = torch.randn(1024, device="cuda")
    o = torch.empty(1024, device="cuda")
    kernel_cache = scale.device_caches[torch.cuda.current_device()][0]
    for launcher, num_warps in ((default.entry, 4), (wide.entry, 8)):
        kernel_cache.clear()
        launch(launcher, (8,), x, o, 1024, 2.0, 128)
        torch.testing.assert_close(o, x * 2.0)
        assert {k.metadata.num_warps for k in kernel_cache.values()} == {num_warps}


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
        launch_symbol="launch", error_symbol="error", error_style="return", libtorch_path="/torch.so",
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
    assert json.loads(to_json(one)) == json.loads(to_json(same))
    assert json.loads(to_json(one)) != json.loads(to_json(other))


def test_module_key_digest_tracks_every_field():
    key = _module_key()
    for field in dataclasses.fields(key):
        value = getattr(key, field.name)
        changed = "zz" if isinstance(value, str) else (value + 1 if isinstance(value, int) else None)
        if changed is None:
            continue
        assert dataclasses.replace(key, **{field.name: changed}).digest() != key.digest(), field.name


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
    assert _symbol_name(nested) == "inner"

    module = getattr(create_launcher(nested), "__self__")
    path = pathlib.Path(module.__file__)
    assert path.name.startswith("inner.")
    assert path.parent.parent.name == nested.__module__
    assert len(path.parent.name) == 64  # the ModuleKey digest

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

# pyright: standard
"""The spec-key invariant on host builds: same intj key => same compiler input.

Needs Triton, whose specialization rules are the truth, but no GPU: a
`no_gpu=True` module keys CPU tensors the way a GPU module keys device
tensors. Each case runs under two backend pairs: the host driver keying no
pointer range against CUDA's rules, and the host driver keying it (as HIP's
backend does) against HIP's.

`test_launcher.py::test_spec_key_is_never_coarser_than_triton` still runs on a
GPU, against Triton's own per-device binder; dynamic options and tuned chains
need a GPU too (`no_gpu=True` refuses both).
"""

from __future__ import annotations

import dataclasses
import random
from types import SimpleNamespace

import pytest

pytest.importorskip("triton")

import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402
from triton.backends.compiler import GPUTarget  # noqa: E402
from triton.compiler import make_backend  # noqa: E402

from intj import (  # noqa: E402
    AUTO,
    NEVER,
    Argument,
    Assume,
    Constexpr,
    Exactly,
    TorchAccessMode,
    make_launcher,
)
from intj import backend as backends  # noqa: E402
from intj import specialization  # noqa: E402
from intj.annotation import Aligned, EqualTo, PointerRange, _merge_annotations  # noqa: E402
from intj.scalar_types import fits, triton_name  # noqa: E402
from intj.specialization import call_layout, compiler_input  # noqa: E402

REFUSALS = (TypeError, ValueError, OverflowError)


class PointerRangeHost(backends.HostBackend):
    """The host driver, keying the pointer-range bit as HIP's backend does."""

    pointer_range = True


PAIRS = {
    "host-cuda": (backends.HostBackend, GPUTarget("cuda", 89, 32)),
    "range-hip": (PointerRangeHost, GPUTarget("hip", "gfx942", 64)),
}


@pytest.fixture(params=sorted(PAIRS))
def truth(request, monkeypatch):
    """Installs the pair's host backend and returns its Triton compiler backend."""
    host, target = PAIRS[request.param]
    monkeypatch.setattr(backends, "HOST", host())
    return make_backend(target)


@pytest.fixture(scope="module")
def tensors():
    base = torch.randn(4096)
    out = SimpleNamespace(
        base=base,
        values=[
            base,  # 16B aligned
            base[1:],  # unaligned
            torch.randn(4096, dtype=torch.float16),
            torch.zeros(4096, dtype=torch.int32),
        ],
    )
    try:
        # > 2 GiB of storage: no tt.pointer_range = 32. Untouched, so virtual only.
        out.values.append(torch.empty(2**29 + 8))
    except RuntimeError:  # pragma: no cover - no address space
        pass
    return out


def check(jit, extra, cases, truth):
    """Asserts that every case a host launcher of `jit` accepts keys as Triton compiles.

    Returns:
        How many cases were keyed (the rest were refused).
    """
    launch = make_launcher(
        jit,
        extra_annotation=extra,
        no_gpu=True,
        verify_annotation=True,
        torch_access_mode=TorchAccessMode.RUNTIME_SHIM,
    )
    resolved = _merge_annotations(jit, extra)
    params = call_layout(resolved)
    baked = {p.index: p.baked for p in resolved if p.annotation.baked_value}
    seen, keyed = {}, 0
    for args in cases:
        try:
            key, nparams = launch.__self__.spec_key(0, *args)
        except REFUSALS:
            continue  # verify_annotation refused it: nothing launches
        keyed += 1
        source = compiler_input(jit, params, args, truth, baked_values=baked)
        assert seen.setdefault(key, source) == source, f"intj key collides for {args!r}"
        assert nparams == sum(ty != "constexpr" for _, ty in source.signature)
    return keyed


@triton.jit
def axpy(x, y, o, n, a, flag, bias, BLOCK: tl.constexpr):
    pass


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
    pass


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
    pass


def axpy_cases(tensors, optional):
    base = tensors.base
    cases: list[tuple] = [
        (x, base, base, 4096, 1.5, True, None, 128) for x in tensors.values
    ]
    for n in (0, 1, 16, 17, 2**31, 2**31 - 1, 2**63, 2**64 - 16):
        cases.append((base, base, base, n, 1.5, True, None, 128))
    for a in (0.0, 1.5, -2.5):
        for flag in (True, False):
            for bias in (None, 0.0, 1.0):
                for block in (64, 128):
                    cases.append((base, base, base, 4096, a, flag, bias, block))
    if optional:
        for y, o in ((None, base), (base, None), (None, None), (base[1:], None)):
            cases.append((base, y, o, 4096, 1.5, True, None, 128))
    return cases


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
def test_axpy_keys_as_triton_compiles(kernel, tensors, truth):
    """The GPU invariant test's kernels and values, on a host build."""
    jit = {"plain": axpy, "tensor": axpy_tensor, "optional": axpy_optional}[
        kernel.split("_")[0]
    ]
    extra = (
        {"n": Argument(type=(tl.int32, tl.int64, tl.uint64))}
        if kernel.endswith("_generic")
        else None
    )
    cases = axpy_cases(tensors, optional=jit is axpy_optional)
    assert check(jit, extra, cases, truth) == len(cases)


@triton.jit
def mixed(a, b, c, d, e, f, C: tl.constexpr, D: tl.constexpr, E: tl.constexpr):
    pass


_RUNTIME = ("a", "b", "c", "d", "e", "f")
_CONSTEXPR = ("C", "D", "E")
_ARGUMENT_TYPES = (
    tl.int32,
    tl.int64,
    tl.uint64,
    tl.int8,
    tl.uint8,
    tl.float32,
    tl.pointer_type(tl.float32),
    tl.pointer_type(tl.float16),
    None,
    (tl.int32, tl.int64, tl.uint64),
    (tl.int8, tl.int32),
    (tl.pointer_type(tl.float32), None),
    (tl.int32, tl.float32),
    (tl.pointer_type(tl.float32), tl.int32),
)
_POLICIES = (
    AUTO,
    NEVER,
    Assume(Aligned(16)),
    Assume(EqualTo(1)),
    Assume(PointerRange(32)),
    Exactly(Aligned(16)),
    Exactly(),
)
_CONSTEXPR_TYPES = (
    tl.int32,
    (tl.int16, tl.int32),
    (tl.int8, tl.int64),
    None,
    tl.float64,
    (tl.int32, None),
)


def _argument(rng):
    fields = {}
    if rng.random() < 0.7:
        fields["type"] = rng.choice(_ARGUMENT_TYPES)
    if rng.random() < 0.7:
        fields["specialize"] = rng.choice(_POLICIES)
    if rng.random() < 0.1:
        fields["value"] = rng.choice((0, 1, 16, 17, -3))
    return Argument(**fields)


def _constexpr(rng):
    fields = {}
    if rng.random() < 0.6:
        fields["type"] = rng.choice(_CONSTEXPR_TYPES)
    if rng.random() < 0.2:
        fields["power_of_two_or_zero"] = True
    if rng.random() < 0.1:
        fields["value"] = rng.choice((0, 4, 16))
    return Constexpr(**fields)


def random_annotations(rng):
    """Returns one valid random annotation per `mixed` parameter, or none for some."""
    extra = {}
    for name in _RUNTIME + _CONSTEXPR:
        if rng.random() < 0.15:
            continue  # left to Triton's own behavior
        make = _argument if name in _RUNTIME else _constexpr
        for _ in range(20):  # rejection-sample until the parameter alone is valid
            annotation = make(rng)
            try:
                _merge_annotations(mixed, {name: annotation})
            except REFUSALS:
                continue
            extra[name] = annotation
            break
    return extra


def _likely_fits(value, a):
    """Returns whether `value` probably passes `verify_annotation` for `a`.

    A sampling heuristic only: the invariant is checked on whatever intj
    accepts, so a wrong guess costs a refused case, never a missed check.
    """
    tensor = isinstance(value, torch.Tensor)
    if a.types is not None and not any(
        value is None
        if ty is None
        else tensor and triton_name(value.dtype) == ty[1:]
        if ty.startswith("*")
        else not tensor and fits(value, ty, a.kind)
        for ty in a.types
    ):
        return False
    integer = type(value) is int
    if a.equal_to_one == "assume" and integer and value != 1:
        return False
    if a.aligned_16 == "assume" and (
        integer and value % 16 or tensor and value.data_ptr() % 16
    ):
        return False
    if (
        a.pointer_range_32 == "assume"
        and tensor
        and value.untyped_storage().nbytes() >= 2**31
    ):
        return False
    if a.power_of_two_or_zero and not (
        integer and (value == 0 or abs(value).bit_count() == 1)
    ):
        return False
    return True


def random_cases(rng, tensors, extra, count):
    """Draws public calls of `mixed`, each value likely to fit its annotation."""
    runtime = [
        *tensors.values,
        None,
        *(0, 1, 16, 17, -16, 2**31 - 1, 2**31, 2**63 - 1, 2**63, 2**64 - 16),
        *(0.0, 1.5, -2.5, 1e39),
        True,
        False,
    ]
    constexpr = [0, 1, 2, 3, 16, 64, -8, 2**31, 2**63, True, None, 1.5, 0.5, "neg"]
    pools = []
    for p in _merge_annotations(mixed, extra):
        a = p.annotation
        if a.baked_value or a.bind_value is not None:
            continue  # not in the public call
        pool = runtime if a.kind == "argument" else constexpr
        pools.append([v for v in pool if _likely_fits(v, a)] or pool)
    return [tuple(rng.choice(pool) for pool in pools) for _ in range(count)]


@pytest.mark.parametrize("seed", range(4))
def test_random_annotations_key_as_triton_compiles(seed, tensors, truth):
    """Random annotations x random values: every accepted call keys as Triton compiles."""
    rng = random.Random(seed)
    extra = random_annotations(rng)
    # Values drawn at random mostly break some annotation; enough survive.
    assert check(mixed, extra, random_cases(rng, tensors, extra, 400), truth) > 100


def test_host_invariant_catches_overlapping_key_fields(monkeypatch, tensors, truth):
    original = specialization._layout_fields

    def overlapping(fields, device_binding):
        # every descriptor on the first one's byte: the last write wins
        placed = original(fields, device_binding)
        first = next(f.offset for _, f in placed.fields if f.kind == "descriptor")
        return dataclasses.replace(
            placed,
            fields=tuple(
                (i, dataclasses.replace(f, offset=first))
                if f.kind == "descriptor"
                else (i, f)
                for i, f in placed.fields
            ),
        )

    monkeypatch.setattr(specialization, "_layout_fields", overlapping)
    with pytest.raises(AssertionError, match="collides"):
        check(axpy, None, axpy_cases(tensors, optional=False), truth)


def test_host_invariant_catches_an_unkeyed_pointer_range(monkeypatch, tensors):
    if len(tensors.values) < 5:
        pytest.skip("no address space for a > 2 GiB tensor")
    # HIP's rules, but a host driver that does not key the range bit
    monkeypatch.setattr(backends, "HOST", backends.HostBackend())
    hip = make_backend(PAIRS["range-hip"][1])
    with pytest.raises(AssertionError, match="collides"):
        check(axpy, None, axpy_cases(tensors, optional=False), hip)

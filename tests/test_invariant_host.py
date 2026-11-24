# pyright: standard
"""The spec-key invariant on host builds: same intj key => same compiler input.

Needs Triton, whose specialization rules are the truth, but no GPU: a
`no_gpu=True` module keys CPU tensors the way a GPU module keys device
tensors. Each case runs under two backend pairs: the host driver keying no
pointer range against CUDA's rules, and the host driver keying it (as HIP's
backend does) against HIP's.

Most cases check the key against intj's own compiler input, which cannot see
a wrong model of Triton; the `*_as_tritons_binder_*` tests check both the key
and the compiler input against Triton's binder itself.

`test_launcher.py::test_spec_key_is_never_coarser_than_triton` still runs on a
GPU, against Triton's own per-device binder; dynamic options and tuned chains
need a GPU too (`no_gpu=True` refuses both).
"""

from __future__ import annotations

import dataclasses
import importlib.util
import operator
import random
from types import SimpleNamespace

import pytest

pytest.importorskip("triton")

import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402
from triton import knobs  # noqa: E402
from triton.backends.compiler import GPUTarget  # noqa: E402
from triton.compiler import make_backend  # noqa: E402
from triton.runtime.jit import create_function_from_signature  # noqa: E402

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
from intj.annotation import (  # noqa: E402
    Aligned,
    EqualTo,
    PointerRange,
    _merge_annotations,
    canonical_value,
)
from intj.scalar_types import fits, is_pointer, triton_name  # noqa: E402
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


def check(jit, extra, cases, truth, compiled_extra=None):
    """Asserts that every case a host launcher of `jit` accepts keys as Triton compiles.

    `compiled_extra`, if given, replaces `extra` on the compiler side only: a
    mutation of what the launcher keys.

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
    resolved = _merge_annotations(
        jit, extra if compiled_extra is None else compiled_extra
    )
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


def ptr_range_cases(tensors):
    """`axpy` calls whose `x` has a `ptr_range()` disagreeing with its storage."""
    cases = []
    for x in tensors.values:
        for answer in (None, 0, 2**31 - 1, 2**31):
            view = x[:]  # a fresh object, so the attribute stays on it
            if answer is not None:
                view.ptr_range = lambda answer=answer: answer
            cases.append((view, tensors.base, tensors.base, 4096, 1.5, True, None, 128))
    return cases


@pytest.mark.parametrize("on", [False, True])
def test_ptr_range_keys_as_triton_compiles(on, tensors, truth):
    """Off, intj keys and compiles the storage size; on, Triton's `ptr_range()`."""
    cases = ptr_range_cases(tensors)
    assert check(axpy, {"x": Argument(ptr_range=on)}, cases, truth) == len(cases)


@pytest.mark.parametrize("on", [False, True])
def test_host_invariant_catches_an_ignored_ptr_range(on, monkeypatch, tensors):
    """Keying one rule and compiling the other collides, in either direction."""
    monkeypatch.setattr(backends, "HOST", PointerRangeHost())
    hip = make_backend(PAIRS["range-hip"][1])
    with pytest.raises(AssertionError, match="collides"):
        check(
            axpy,
            {"x": Argument(ptr_range=on)},
            ptr_range_cases(tensors),
            hip,
            compiled_extra={"x": Argument(ptr_range=not on)},
        )


@triton.jit
def one_pointer(p):
    pass


def _unreadable(*_):
    raise AssertionError("intj read what it does not key")


def _vanish():
    raise AttributeError("ptr_range")


@pytest.mark.parametrize("mode", list(TorchAccessMode))
@pytest.mark.parametrize(
    "lookups",
    [
        pytest.param((lambda: 0, lambda: 2**31), id="widens"),
        pytest.param((lambda: 2**31, lambda: 0), id="narrows"),
        pytest.param((lambda: 0, _vanish), id="vanishes"),
    ],
)
def test_ptr_range_looks_up_twice_as_triton(lookups, mode, monkeypatch):
    """`hasattr(x, "ptr_range")`, then `x.ptr_range()`: a descriptor sees both.

    Triton's own specialization is the truth, for the C key and for the
    compiler input built without the key's measurement alike.
    """
    monkeypatch.setattr(backends, "HOST", PointerRangeHost())
    monkeypatch.setattr(knobs.amd, "use_buffer_ops", True)
    hip = make_backend(PAIRS["range-hip"][1])
    annotation = Argument(ptr_range=True)
    launch = make_launcher(
        one_pointer,
        extra_annotation={"p": annotation},
        no_gpu=True,
        torch_access_mode=mode,
    )
    params = call_layout(_merge_annotations(one_pointer, {"p": annotation}))
    x = torch.empty(4, dtype=torch.uint8)
    # CPU storage is virtual: the wide tensor's bytes are never touched
    narrow, wide = (
        launch.__self__.spec_key(0, t)
        for t in (x, torch.empty(2**31, dtype=torch.uint8))
    )
    seen = []

    def descriptor(self):
        answer = lookups[len(seen)]
        seen.append(answer)
        return _vanish() if answer is _vanish else answer

    monkeypatch.setattr(torch.Tensor, "ptr_range", property(descriptor), False)

    def run(fn):
        seen.clear()
        try:
            return fn()
        except AttributeError:
            return AttributeError
        finally:
            assert len(seen) == 2, "one hasattr() and one call, as Triton makes"

    truth = run(
        lambda: specialization._triton_specialize(
            x, backend=hip, is_const=False, specialize=True, align=True
        )[1]
    )
    key = run(lambda: launch.__self__.spec_key(0, x))
    source = run(lambda: str(compiler_input(one_pointer, params, [x], hip).attrs))
    if truth is AttributeError:
        assert key is source is AttributeError
    else:
        assert isinstance(truth, str) and isinstance(source, str)
        assert key == (narrow if "S" in truth else wide)
        assert ("tt.pointer_range" in source) == ("S" in truth)


@pytest.mark.parametrize("mode", list(TorchAccessMode))
@pytest.mark.parametrize(
    "case",
    [
        "override skips storage",
        "off skips ptr_range",
        "NEVER skips storage",
        "assumed skips storage",
    ],
)
def test_ptr_range_reads_only_what_it_keys(case, mode, monkeypatch):
    """Neither the C key nor the compiler input touches a tensor's unused range.

    On, a `ptr_range` attribute decides before the storage is read; off, the
    attribute is never looked up; NEVER measures nothing, and an unverified
    assumed range ignores what Triton would measure.
    """
    monkeypatch.setattr(backends, "HOST", PointerRangeHost())
    monkeypatch.setattr(knobs.amd, "use_buffer_ops", True)
    hip = make_backend(PAIRS["range-hip"][1])
    x = torch.empty(4)
    if case == "override skips storage":
        annotation = Argument(ptr_range=True)
        x.ptr_range = lambda: 0  # pyright: ignore[reportAttributeAccessIssue]  # an instance attribute, as Triton reads it
        x.untyped_storage = _unreadable  # pyright: ignore[reportAttributeAccessIssue]  # an instance attribute, as INTERPRETER calls it
    elif case == "off skips ptr_range":
        annotation = Argument()
        monkeypatch.setattr(torch.Tensor, "ptr_range", property(_unreadable), False)
    else:
        annotation = Argument(
            specialize=NEVER
            if case == "NEVER skips storage"
            else Assume(PointerRange(32))
        )
        x.ptr_range = lambda: 0  # pyright: ignore[reportAttributeAccessIssue]  # an instance attribute, as Triton reads it
        x.untyped_storage = _unreadable  # pyright: ignore[reportAttributeAccessIssue]  # an instance attribute, as INTERPRETER calls it
    launch = make_launcher(
        one_pointer,
        extra_annotation={"p": annotation},
        no_gpu=True,
        torch_access_mode=mode,
    )
    key = launch.__self__.spec_key(0, x)
    assert key == launch.__self__.spec_key(0, torch.empty(4))
    params = call_layout(_merge_annotations(one_pointer, {"p": annotation}))
    attrs = str(compiler_input(one_pointer, params, [x], hip).attrs)
    assert ("tt.pointer_range" in attrs) == (case != "NEVER skips storage")


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
        if is_pointer(ty)
        else not tensor and fits(value, ty)
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


# Triton reads a pointer type only from an annotation object, which this
# module's postponed annotations are not: the typed kernels live in a source
# of their own.
TYPED_KERNELS = """
import triton
import triton.language as tl

@triton.jit
def typed_pointer(p: tl.pointer_type(tl.float32)):
    pass

@triton.jit(do_not_specialize_on_alignment=["p"])
def typed_pointer_dnsa(p: tl.pointer_type(tl.float32)):
    pass

@triton.jit
def typed_int(n: int):
    pass

@triton.jit
def typed_i8(n: tl.int8):
    pass

@triton.jit
def typed_u64(n: tl.uint64):
    pass

@triton.jit(do_not_specialize=["n"])
def typed_int_dns(n: int):
    pass

@triton.jit
def typed_bool(b: bool):
    pass

@triton.jit
def typed_float(f: float):
    pass
"""


@pytest.fixture(scope="module")
def typed(tmp_path_factory):
    path = tmp_path_factory.mktemp("typed") / "typed_kernels.py"
    path.write_text(TYPED_KERNELS)
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # pyright: ignore[reportAttributeAccessIssue]  # 3.8 stubs lack it
    return module


def binder_case(typed, case):
    """Returns the kernel, its extra annotation, and the kernel whose binder is the truth.

    `Argument(type=...)` means what Triton's own annotation of that type does.
    """
    if case == "argument_pointer":
        extra = {"p": Argument(type=tl.pointer_type(tl.float32))}
        return one_pointer, extra, typed.typed_pointer
    if case == "untyped":
        return one_pointer, None, one_pointer
    jit = getattr(typed, case)
    return jit, None, jit


BINDER_CASES = (
    "typed_pointer",
    "typed_pointer_dnsa",
    "argument_pointer",
    "typed_int",
    "typed_i8",
    "typed_u64",
    "typed_int_dns",
    "typed_bool",
    "typed_float",
    "untyped",
)


def binder_values(tensors):
    """One-argument calls: tensors, int addresses, scalars and 0-dim tensors."""
    base = tensors.base
    # 0-dim int tensors whose value and address disagree on divisibility
    buf = torch.tensor([16, 17, 16, 17], dtype=torch.int64)
    values = [
        *tensors.values,
        base.data_ptr(),
        base.data_ptr() + 4,
        *(0, 1, 16, 17, -16, 2**31, 2**63, 2**64 - 16),
        None,
        True,
        False,
        *(0.0, 1.5),
        *(buf[i] for i in range(4)),
        torch.tensor(17),
        torch.tensor(1),
    ]
    try:
        import numpy as np
    except ImportError:  # pragma: no cover - torch without numpy
        pass
    else:
        values += [np.int64(16), np.int64(17)]
    return [(value,) for value in values]


INT_NAMES = {f"{sign}{bits}" for sign in "iu" for bits in (8, 16, 32, 64)}


def launched(jit, args):
    """`args` as a launch passes them.

    A Triton-typed integer is converted as its C decode does (`__index__`, or
    `__int__` before Python 3.10). Triton's binder specializes a 0-dim tensor
    there by its address, which it never launches; intj by the integer it does.
    """

    def convert(value):
        if type(value) in (int, bool):
            return value
        return (
            operator.index(value) if hasattr(type(value), "__index__") else int(value)
        )

    return tuple(
        convert(value) if param.annotation_type in INT_NAMES else value
        for param, value in zip(jit.params, args)
    )


def tritons_binder(jit, backend):
    """Returns Triton's specialization of a launch: types, constants and attrs."""
    binder = create_function_from_signature(jit.signature, jit.params, backend)

    def specialize(args):
        _, spec, _ = binder(*launched(jit, args))
        types = tuple(ty for ty, _ in spec)
        constants = tuple(
            ((i,), canonical_value(args[i]))
            for i, ty in enumerate(types)
            if ty == "constexpr"
        )
        attrs = tuple(
            ((i,), tuple((name, amount) for name, amount in backend.parse_attr(desc)))
            for i, (ty, desc) in enumerate(spec)
            if ty != "constexpr" and isinstance(desc, str) and backend.parse_attr(desc)
        )
        return types, constants, attrs

    return specialize


def check_binder(jit, extra, reference, cases, truth):
    """Asserts every case a host launcher of `jit` accepts keys and compiles as Triton.

    Triton's binder for `reference` is the truth: one intj key never covers
    two of its specializations, and intj's compiler input is what it gives.

    Returns:
        How many cases were keyed (the rest were refused).
    """
    launch = make_launcher(
        jit,
        extra_annotation=extra,
        no_gpu=True,
        torch_access_mode=TorchAccessMode.RUNTIME_SHIM,
    )
    params = call_layout(_merge_annotations(jit, extra))
    specialize = tritons_binder(reference, truth)
    seen, keyed = {}, 0
    for args in cases:
        try:
            key, _ = launch.__self__.spec_key(0, *args)
        except REFUSALS:
            continue  # refused: nothing launches
        keyed += 1
        spec = specialize(args)
        assert seen.setdefault(key, spec) == spec, f"intj key collides for {args!r}"
        source = compiler_input(jit, params, args, truth)
        ours = (
            tuple(ty for _, ty in source.signature),
            source.constants,
            source.attrs,
        )
        assert ours == spec, f"intj compiles unlike Triton for {args!r}"
    return keyed


@pytest.mark.parametrize("case", BINDER_CASES)
def test_keys_and_compiles_as_tritons_binder_specializes(case, typed, tensors, truth):
    """Typed pointers and integers given every value Triton's binder takes."""
    jit, extra, reference = binder_case(typed, case)
    assert check_binder(jit, extra, reference, binder_values(tensors), truth) > 10


def test_host_invariant_catches_a_typed_int_compiled_from_its_object(
    monkeypatch, typed, tensors, truth
):
    """A 0-dim tensor compiled by its address, not the integer launched."""
    monkeypatch.setattr(specialization, "_launched_int", lambda value: value)
    jit = typed.typed_int
    with pytest.raises(AssertionError, match="unlike Triton"):
        check_binder(jit, None, jit, binder_values(tensors), truth)


def test_host_invariant_catches_a_wrong_model_of_triton(monkeypatch, typed, tensors):
    """An int address given a pointer range, which Triton never gives one.

    The key is right, so the compiler input agrees with itself under each
    key; only Triton's binder tells it is wrong.
    """
    monkeypatch.setattr(backends, "HOST", PointerRangeHost())
    hip = make_backend(PAIRS["range-hip"][1])
    original = specialization._triton_specialize

    def ranged_ints(value, **kwargs):
        ty, desc = original(value, **kwargs)
        return ty, desc + "S" if type(value) is int and isinstance(desc, str) else desc

    monkeypatch.setattr(specialization, "_triton_specialize", ranged_ints)
    jit = typed.typed_pointer
    with pytest.raises(AssertionError, match="unlike Triton"):
        check_binder(jit, None, jit, binder_values(tensors), hip)


def test_host_invariant_catches_an_unkeyed_range_against_tritons_binder(
    monkeypatch, typed, tensors
):
    if len(tensors.values) < 5:
        pytest.skip("no address space for a > 2 GiB tensor")
    # HIP's rules, but a host driver that does not key the range bit
    monkeypatch.setattr(backends, "HOST", backends.HostBackend())
    hip = make_backend(PAIRS["range-hip"][1])
    jit = typed.typed_pointer
    with pytest.raises(AssertionError, match="collides"):
        check_binder(jit, None, jit, binder_values(tensors), hip)

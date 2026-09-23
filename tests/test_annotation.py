# pyright: standard

from typing import Any, cast

import pytest
import triton.language as tl

import intj
from intj.annotation import UNSET


def test_public_annotation_exports_and_presets():
    assert intj.INT_TYPES == (tl.int32, tl.int64, tl.uint64)
    assert intj.FLOAT_TYPES == (tl.float32,)
    assert "UNSET" not in intj.__all__
    assert intj.Argument().type is UNSET
    assert intj.Argument(type=None).type is None


def test_annotation_sequences_are_immutable():
    arg = intj.Argument(type=[tl.int64, tl.int32, tl.int64])
    assert isinstance(arg.type, tuple)
    assert hash(arg) == hash(intj.Argument(type=(tl.int64, tl.int32, tl.int64)))


@pytest.mark.parametrize(
    "build",
    [
        lambda: intj.Argument(value=[1]),
        lambda: intj.Constexpr(value=[1]),
    ],
)
def test_annotation_values_reject_unhashable_values(build):
    with pytest.raises(TypeError):
        build()


@pytest.mark.parametrize(
    "fact",
    [intj.EqualTo(True), intj.Aligned(cast(Any, 16.0))],
)
def test_assume_fact_values_require_exact_int(fact):
    with pytest.raises(TypeError, match="Fact value must be an int"):
        intj.Assume(fact)


@pytest.mark.parametrize(
    "build,match",
    [
        (lambda: intj.Argument(type=[]), "type sequence must not be empty"),
        (lambda: intj.Argument(bind_value=cast(Any, "tensor")), "bind_value"),
        (
            lambda: intj.Argument(value=1, bind_value=intj.BindValue.POINTER),
            "mutually exclusive",
        ),
        (lambda: intj.Assume(intj.EqualTo(2)), r"EqualTo\(1\)"),
        (lambda: intj.Assume(intj.Aligned(8)), r"Aligned\(16\)"),
        (lambda: intj.Assume(intj.PointerRange(64)), r"PointerRange\(32\)"),
    ],
)
def test_annotation_constructor_errors(build, match):
    with pytest.raises((TypeError, ValueError), match=match):
        build()

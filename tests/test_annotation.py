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

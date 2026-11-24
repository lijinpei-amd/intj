# pyright: standard
"""The scalar type table: kinds, key widths, narrowness and ranges. No triton."""

import pytest

from intj.scalar_types import (
    fact_applies,
    fits,
    inferred_type,
    is_float,
    is_integer,
    is_pointer,
    key_bytes,
    narrowness,
)


def test_integers_are_the_signed_and_unsigned_widths_only():
    assert [n for n in ("u1", "i8", "u64", "fp32", "*i32", None) if is_integer(n)] == [
        "i8",
        "u64",
    ]
    assert [n for n in ("fp32", "fp64", "i32", "*fp32", None) if is_float(n)] == [
        "fp32",
        "fp64",
    ]


def test_key_bytes():
    widths = {None: 0, "u1": 1, "i8": 1, "i16": 2, "u32": 4, "i64": 8, "fp64": 8}
    assert {name: key_bytes(name) for name in widths} == widths


def test_narrowness_keeps_the_rendered_type_test_order():
    # fp32 and pointers sort as width 0, ahead of every integer
    names = ["u64", "fp64", "i32", "fp32", None, "u1", "*fp32", "i8"]
    assert sorted(names, key=narrowness) == [
        None,
        "*fp32",
        "fp32",
        "u1",
        "i8",
        "i32",
        "fp64",
        "u64",
    ]


@pytest.mark.parametrize(
    "value, name, expected",
    [
        (-128, "i8", True),
        (128, "i8", False),
        (-1, "u8", False),
        (255, "u8", True),
        (1, "u1", False),  # a bool only
        (True, "u1", True),
        (True, "i32", False),
        (-(2**63), "i64", True),
        (2**63, "i64", False),
        (0.1, "fp32", True),  # rounds
        (float("inf"), "fp32", True),
        (float("nan"), "fp32", True),
        # beyond fp32's largest, as verify_annotation's range check refuses it
        (1e39, "fp32", False),
        (float.fromhex("0x1.fffffe0000001p+127"), "fp32", False),
        (float.fromhex("0x1.fffffep+127"), "fp32", True),
        (1e39, "fp64", True),
        (1, "fp64", False),
        (0, "*fp32", False),
        (None, None, True),
        (0, None, False),
    ],
)
def test_fits(value, name, expected):
    assert fits(value, name) is expected


@pytest.mark.parametrize(
    "value, expected",
    [
        (None, None),
        (True, "u1"),
        (1e39, "fp32"),  # any float, as Triton's launcher converts it
        (-(2**31), "i32"),
        (2**31, "i64"),
        (-(2**63), "i64"),
        (2**63, "u64"),
        (2**64 - 1, "u64"),
    ],
)
def test_inferred_type(value, expected):
    assert inferred_type(value) == expected


@pytest.mark.parametrize("value", [2**64, -(2**63) - 1, "s"])
def test_inferred_type_refuses_what_triton_cannot_pass(value):
    with pytest.raises(ValueError, match="outside Triton's scalar range"):
        inferred_type(value)


def test_facts_apply_by_kind():
    names = ("u1", "i8", "u64", "fp32", "*fp16", None)
    assert [n for n in names if is_pointer(n)] == ["*fp16"]
    assert {
        fact: [n for n in names if fact_applies(n, fact)]
        for fact in ("equal_to_one", "aligned_16", "pointer_range_32")
    } == {
        "equal_to_one": ["i8", "u64"],
        "aligned_16": ["i8", "u64", "*fp16"],
        "pointer_range_32": ["*fp16"],
    }

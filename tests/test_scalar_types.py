# pyright: standard
"""The scalar type table: kinds, key widths, narrowness and ranges. No triton."""

import pytest

from intj.scalar_types import fits, is_float, is_integer, key_bytes, narrowness


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
    "value, name, kind, expected",
    [
        (-128, "i8", "argument", True),
        (128, "i8", "argument", False),
        (-1, "u8", "argument", False),
        (255, "u8", "argument", True),
        (1, "u1", "argument", False),  # a bool only
        (True, "u1", "argument", True),
        (True, "i32", "argument", False),
        (0.1, "fp32", "argument", True),  # an argument rounds
        (0.1, "fp32", "constexpr", False),  # a constexpr must be exact
        (0.5, "fp32", "constexpr", True),
        (1e39, "fp32", "argument", False),
        (1e39, "fp64", "constexpr", True),
        (1, "fp64", "argument", False),
        (0, "*fp32", "argument", False),
        (None, None, "argument", True),
        (0, None, "argument", False),
    ],
)
def test_fits(value, name, kind, expected):
    assert fits(value, name, kind) is expected

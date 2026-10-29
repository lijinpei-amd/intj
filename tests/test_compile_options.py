# pyright: standard
"""Compile-option merging and the knob table, against Triton's parser, no GPU."""

from types import SimpleNamespace

import pytest

pytest.importorskip("triton")

from triton import knobs  # noqa: E402
from triton.backends.compiler import GPUTarget  # noqa: E402
from triton.compiler import make_backend  # noqa: E402

from intj import compat, compile_options  # noqa: E402
from intj.compile_options import (  # noqa: E402
    Knob,
    knob_options,
    parameter_options,
    resolve,
)
from intj.errors import UnsupportedKernel  # noqa: E402

TARGET = GPUTarget("hip", "gfx942", 64)
DEBUG = "knobs.runtime.debug"
INSTRUMENTATION = "knobs.compilation.instrumentation_mode"
CASTS = "knobs.compilation.fpsan_homomorphic_casts"


def kernel(debug=False):
    return SimpleNamespace(debug=debug)


def test_overlay_wins_over_base():
    resolved = resolve(
        TARGET, kernel(), {"num_warps": 4, "num_stages": 2}, {"num_warps": 8}, {}
    )
    assert (resolved.num_warps, resolved.num_stages) == (8, 2)


@pytest.mark.parametrize(
    "kernel_debug, passed, knob, expected",
    [
        (False, {}, True, True),  # the knob turns it on
        (True, {}, False, True),  # the kernel's own default
        (True, {"debug": False}, False, False),  # a passed value beats the kernel's
        (False, {"debug": False}, True, True),  # the knob still turns it on
    ],
)
def test_debug_is_ored_with_its_knob(kernel_debug, passed, knob, expected):
    merged = knob_options(kernel(kernel_debug), passed, {DEBUG: knob})
    assert merged["debug"] is expected


def test_an_overwriting_knob_replaces_the_passed_value():
    passed = {"instrumentation_mode": "fpsan", "fpsan_homomorphic_casts": True}
    merged = knob_options(kernel(), passed, {INSTRUMENTATION: "", CASTS: False})
    assert merged["instrumentation_mode"] == ""
    assert merged["fpsan_homomorphic_casts"] is False


def test_an_unread_knob_takes_its_default():
    # a declared dynamic knob outside its miss is not in knob_values
    merged = knob_options(kernel(), {"fpsan_homomorphic_casts": True}, {})
    assert merged["debug"] is False
    assert merged["instrumentation_mode"] == ""
    assert merged["fpsan_homomorphic_casts"] is True  # None default: left as passed


def test_unknown_names_are_refused():
    with pytest.raises(UnsupportedKernel, match="num_warpz"):
        resolve(TARGET, kernel(), {}, {"num_warpz": 4}, {})


def test_only_constexpr_parameters_named_like_options_feed_them():
    def param(name, kind):
        return SimpleNamespace(name=name, annotation=SimpleNamespace(kind=kind))

    params = [
        param("num_warps", "constexpr"),
        param("num_stages", "argument"),  # a runtime value cannot set an option
        param("warp_size", "constexpr"),  # a launch fixes it
        param("BLOCK", "constexpr"),
    ]
    assert parameter_options(make_backend(TARGET), params) == ("num_warps",)


def test_compat_keys_on_every_knob_in_the_table(monkeypatch):
    # a new row reaches compat's launcher key with no other edit
    row = Knob("knobs.compilation.always_compile", "always_compile", "overwrite", None)
    monkeypatch.setattr(compile_options, "KNOBS", (*compile_options.KNOBS, row))
    for knob in compile_options.KNOBS:
        _, group, name = knob.path.split(".")
        owner = getattr(knobs, group)
        if name not in type(owner).__dict__:
            continue  # not in this triton
        before = compat._knob_key()
        old = getattr(owner, name)
        monkeypatch.setattr(owner, name, "on" if isinstance(old, str) else not old)
        assert compat._knob_key() != before, knob.path

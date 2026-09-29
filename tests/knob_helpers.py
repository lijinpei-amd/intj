# pyright: standard
"""Helpers shared by test_launcher.py and test_tuned.py (not a test module)."""

import contextlib

import pytest
import triton

ON_HIP = (
    getattr(triton.runtime.driver.active.get_current_target(), "backend", None) == "hip"
)
needs_hip = pytest.mark.skipif(not ON_HIP, reason="needs a HIP target")


@contextlib.contextmanager
def live_knob_values(values):
    """Sets the given live Triton knobs for the duration of the block.

    A declared knob only keys the launch, so the live value must match what the
    call passes. Every knob touched is restored on exit. `values` maps dotted
    paths such as `"knobs.compilation.disable_line_info"` to values.
    """
    from triton import knobs

    before = {}
    try:
        for path, value in values.items():
            _, group, name = path.split(".")
            before.setdefault((group, name), getattr(getattr(knobs, group), name))
            setattr(getattr(knobs, group), name, value)
        yield
    finally:
        for (group, name), value in before.items():
            setattr(getattr(knobs, group), name, value)

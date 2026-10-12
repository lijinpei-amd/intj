# pyright: standard
"""Helpers shared by test_launcher.py and test_tuned.py (not a test module)."""

import contextlib

import pytest
import triton

_BACKEND_NAME = getattr(
    triton.runtime.driver.active.get_current_target(), "backend", None
)
ON_HIP = _BACKEND_NAME == "hip"
needs_hip = pytest.mark.skipif(not ON_HIP, reason="needs a HIP target")


def _specializes_pointer_range():
    """Whether the active backend specializes pointers on a 2 GiB range.

    Read off the registered `Backend`, not the target name: `pointer_range` is
    the field the renderer and the spec key branch on, so a new backend that
    sets it gets these tests without being named here.
    """
    from intj.launcher import BACKENDS

    if _BACKEND_NAME is None:
        return False
    backend = BACKENDS.get(_BACKEND_NAME)
    return backend is not None and backend.pointer_range


POINTER_RANGE = _specializes_pointer_range()
needs_pointer_range = pytest.mark.skipif(
    not POINTER_RANGE,
    reason="backend does not specialize pointers on a 2 GiB range",
)


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

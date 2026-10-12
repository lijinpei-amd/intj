# pyright: standard
"""The launcher object: decoration records options; check() and bind/unbind."""

import subprocess
import sys
import textwrap

import pytest
import triton
import triton.language as tl

from intj import MakeLauncherOption, make_launcher
from intj.annotation import NEVER, Argument, BindValue
from intj.launcher import Launcher
from intj.options import OPTION_FIELDS, MakeLauncherKwargs
from intj.torch_intf.torch_abi import TorchAccessMode


@triton.jit
def store(x, n):
    tl.store(x + tl.arange(0, 1), n)


_POINTER = Argument(
    type=tl.pointer_type(tl.float32), specialize=NEVER, bind_value=BindValue.POINTER
)
_HOST = {"no_gpu": True, "torch_access_mode": TorchAccessMode.INTERPRETER}


def test_decoration_records_and_check_validates():
    launch = make_launcher(store, no_gpu=1)  # pyright: ignore[reportArgumentType]  # no check at decoration
    assert launch.__self__.options == MakeLauncherOption(store, no_gpu=1)  # pyright: ignore[reportArgumentType]
    assert isinstance(launch.__self__, Launcher)
    with pytest.raises(TypeError, match="no_gpu must be a bool"):
        launch.__self__.check()
    with pytest.raises(TypeError, match="no_gpu must be a bool"):
        launch(0, 0, 1, 0, 4)  # the first call checks too, and retries


def test_unknown_keyword_fails_at_decoration():
    with pytest.raises(
        TypeError, match=r"unexpected keyword argument\(s\) \['bogus'\]"
    ):
        make_launcher(store, bogus=1)  # pyright: ignore[reportCallIssue]
    with pytest.raises(TypeError, match="bogus"):
        make_launcher(bogus=1)  # pyright: ignore[reportCallIssue]


def test_decoration_imports_no_torch(tmp_path):
    (tmp_path / "kern.py").write_text(
        textwrap.dedent(
            """
            import triton
            import triton.language as tl

            @triton.jit
            def k(x):
                tl.store(x, 1)
            """
        )
    )
    probe = f"""
import sys
sys.path.insert(0, {str(tmp_path)!r})
import intj, kern
torch_before = "torch" in sys.modules
intj.make_launcher(kern.k, no_gpu=True)
assert torch_before or "torch" not in sys.modules, "make_launcher imported torch"
from intj import _imports
assert _imports.torch is None
"""
    subprocess.run([sys.executable, "-c", probe], check=True)


def test_bind_and_unbind_derive_new_launchers():
    base = make_launcher(store, extra_annotation={"x": _POINTER}, **_HOST)
    bound = base.__self__.bind(x=0)
    fixed = bound.__self__.bind_device(3)
    assert base.__self__.options.bind_values is None
    assert bound.__self__.options.bind_values == {"x": 0}
    assert (fixed.__self__.options.bind_device, fixed.__self__.options.bind_values) == (
        3,
        {"x": 0},
    )
    assert fixed.__self__.unbind_device().__self__.options == bound.__self__.options
    assert bound.__self__.unbind().__self__.options == base.__self__.options
    assert fixed.__self__.bind_device(None).__self__.options.bind_device is None
    assert bound(0, 0, 1, 4) is None and fixed(0, 1, 4) is None
    with pytest.raises(TypeError, match="missing"):
        bound.__self__.unbind()(0, 0, 1, 4)


def test_derivatives_share_one_analysis(monkeypatch):
    import intj.launcher as launcher

    calls = []
    real = launcher._analyze
    monkeypatch.setattr(launcher, "_analyze", lambda o: calls.append(1) or real(o))
    base = make_launcher(store, extra_annotation={"x": _POINTER}, **_HOST)
    for k in (base.__self__.bind(x=0), base.__self__.bind_device(0, x=0)):
        k.__self__.check()
    assert calls == [1]


def test_bind_on_a_built_launcher_returns_an_unbuilt_one():
    base = make_launcher(store, **_HOST)
    assert base(0, 0, 1, 0, 4) is None
    child = base.__self__.bind_device(0)
    assert child.__self__._module is None and child.__self__._plan is None
    assert child(0, 1, 0, 4) is None and base(0, 0, 1, 0, 4) is None


def test_launcher_object_is_not_instantiable():
    launch = make_launcher(store, **_HOST)
    with pytest.raises(TypeError, match="come from make_launcher"):
        type(launch.__self__)()


def test_bind_device_true_points_at_the_method():
    launch = make_launcher(store, bind_device=True, **_HOST)
    with pytest.raises(
        TypeError, match=r"not a flag; use k.__self__.bind_device\(ordinal\)"
    ):
        launch.__self__.check()


def test_kwargs_typeddict_matches_the_dataclass():
    assert set(MakeLauncherKwargs.__annotations__) == OPTION_FIELDS
    assert "jit_func" not in OPTION_FIELDS and "delayed" not in OPTION_FIELDS

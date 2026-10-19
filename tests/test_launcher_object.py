# pyright: standard
"""The launcher object: decoration records options; check() and bind/unbind."""

import dataclasses
import subprocess
import sys
import textwrap
import threading
import time

import pytest
import triton
import triton.language as tl

from intj import MakeLauncherOption, make_launcher
from intj.annotation import NEVER, Argument, BindValue, Constexpr
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


@pytest.fixture
def kern_dir(tmp_path):
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
    return tmp_path


def test_decoration_imports_no_torch(kern_dir):
    tmp_path = kern_dir
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


def _option(launch):
    option = launch.__self__.options
    assert isinstance(option, MakeLauncherOption)
    return option


def test_bind_and_unbind_derive_new_launchers():
    base = make_launcher(store, extra_annotation={"x": _POINTER}, **_HOST)
    bound = base.__self__.bind(x=0)
    fixed = bound.__self__.bind_device(3)
    assert _option(base).bind_values is None
    assert _option(bound).bind_values == {"x": 0}
    assert (_option(fixed).bind_device, _option(fixed).bind_values) == (3, {"x": 0})
    assert _option(fixed.__self__.unbind_device()) == _option(bound)
    assert _option(bound.__self__.unbind()) == _option(base)
    assert _option(fixed.__self__.bind_device(None)).bind_device is None
    assert bound(0, 0, 1, 4) is None and fixed(0, 1, 4) is None
    with pytest.raises(TypeError, match="missing"):
        bound.__self__.unbind()(0, 0, 1, 4)


def test_each_launcher_checks_its_own_options(monkeypatch):
    import intj.launcher as launcher

    calls = []
    real = launcher._check_options
    monkeypatch.setattr(
        launcher, "_check_options", lambda o: calls.append(o.bind_device) or real(o)
    )
    base = make_launcher(store, extra_annotation={"x": _POINTER}, **_HOST)
    for k in (base.__self__.bind(x=0), base.__self__.bind_device(0, x=0)):
        k.__self__.check()
    assert calls == [None, 0]


def test_bind_on_a_built_launcher_returns_an_unbuilt_one():
    base = make_launcher(store, **_HOST)
    assert base(0, 0, 1, 0, 4) is None
    child = base.__self__.bind_device(0)
    assert child.__self__._C_module is None
    assert child(0, 1, 0, 4) is None and base(0, 0, 1, 0, 4) is None


def test_check_is_a_dry_run():
    delayed_calls = []

    def delayed():
        delayed_calls.append(1)
        return _HOST

    launch = make_launcher(store, delayed=delayed)
    launch.__self__.check()
    launch.__self__.check()
    assert delayed_calls == [1, 1]
    assert launch.__self__.options.delayed is delayed  # nothing stored
    assert launch.__text_signature__ is None  # pyright: ignore[reportAttributeAccessIssue]  # a C function's
    assert launch(0, 0, 1, 0, 4) is None and delayed_calls == [1, 1, 1]
    assert launch.__self__.options.delayed is None
    assert launch.__text_signature__ is not None  # pyright: ignore[reportAttributeAccessIssue]  # a C function's
    launch.__self__.check()  # built: nothing runs
    assert delayed_calls == [1, 1, 1]


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


def test_decoration_snapshots_mutable_arguments():
    annotations: dict[str, object] = {"x": _POINTER}
    values, dynamic = {"x": 0}, []
    launch = make_launcher(
        store,
        extra_annotation=annotations,
        bind_values=values,
        dynamic_options=dynamic,
        **_HOST,
    )
    annotations["n"] = Constexpr(value=5)
    values["x"] = 2**64  # would overflow the bound pointer at the first call
    dynamic.append("num_warps")  # would be refused: no_gpu compiles nothing
    assert launch(0, 0, 1, 4) is None
    option = _option(launch)
    assert option.extra_annotation == {"x": _POINTER}
    assert option.bind_values == {"x": 0} and option.dynamic_options == ()


def test_concurrent_first_decorations_share_one_stub(kern_dir):
    probe = f"""
import sys, threading, time
sys.path.insert(0, {str(kern_dir)!r})
import intj.lazy as lazy
real, loads = lazy.load_stub, []
def slow(*args):
    loads.append(1)
    time.sleep(0.2)  # every thread misses the stub before any load returns
    return real(*args)
lazy.load_stub = slow
import intj, kern
barrier, errors = threading.Barrier(8), []
def decorate():
    barrier.wait()
    try:
        intj.make_launcher(kern.k, no_gpu=True)
    except BaseException as error:
        errors.append(repr(error))
threads = [threading.Thread(target=decorate) for _ in range(8)]
for t in threads:
    t.start()
for t in threads:
    t.join()
assert errors == [], errors
assert len(loads) == 1, loads
intj.make_launcher(kern.k, no_gpu=True)  # and every later decoration
"""
    subprocess.run([sys.executable, "-c", probe], check=True)


def test_option_holds_delayed_outside_option_fields():
    def delayed():
        return {}

    option = MakeLauncherOption(store, delayed=delayed)
    assert option.delayed is delayed
    assert MakeLauncherOption(store).delayed is None
    assert "delayed" not in OPTION_FIELDS
    assert "delayed" not in MakeLauncherKwargs.__annotations__


def _first():
    return {"verify_annotation": True}


def _second():
    return {"ndebug": False}


_EDITS = [
    ("bind", (), {"x": 0}),
    ("bind_device", (3,), {"x": 0}),
    ("unbind", (), {}),
    ("unbind_device", (), {}),
    ("replace_delayed", (_second,), {}),
    ("replace_delayed", (None,), {}),
    ("chain_delayed", (_second,), {}),
    ("resolve_delayed", (), {}),
]


def _edit_base():
    return make_launcher(
        store,
        extra_annotation={"x": _POINTER},
        bind_values={"x": 1},
        bind_device=2,
        delayed=_first,
        **_HOST,
    )


def _comparable(option):
    # a chained delayed is a fresh closure each time: compare what it returns
    delayed = option.delayed
    return dataclasses.replace(option, delayed=None), (
        None if delayed is None else delayed()
    )


@pytest.mark.parametrize("name, args, kwargs", _EDITS)
def test_in_place_and_copying_edits_agree(name, args, kwargs):
    base = _edit_base()
    before = _option(base)
    copy = getattr(base.__self__, name)(*args, **kwargs)
    assert _option(base) == before  # the copying form leaves it alone
    in_place = _edit_base()
    assert getattr(in_place.__self__, "_" + name)(*args, **kwargs) is None
    assert _comparable(_option(copy)) == _comparable(_option(in_place))


@pytest.mark.parametrize("name, args, kwargs", _EDITS)
def test_in_place_edits_refused_once_built_allowed_after_check(name, args, kwargs):
    checked = _edit_base()
    checked.__self__.check()
    getattr(checked.__self__, "_" + name)(*args, **kwargs)  # check() fixes nothing
    built = _edit_base()
    built.__self__.build()
    if name == "resolve_delayed":
        built.__self__._resolve_delayed()  # nothing left to resolve: a no-op
        return
    with pytest.raises(RuntimeError, match="options are fixed"):
        getattr(built.__self__, "_" + name)(*args, **kwargs)
    copy = getattr(built.__self__, name)(*args, **kwargs)
    assert copy.__self__._C_module is None  # the copy is unbuilt


def test_resolve_and_chain_delayed():
    k = make_launcher(store, delayed=lambda: _HOST)
    resolved = k.__self__.resolve_delayed()
    assert _option(resolved).delayed is None and _option(resolved).no_gpu is True
    assert _option(k).delayed is not None
    chained = k.__self__.chain_delayed(lambda: {"no_gpu": False, "ndebug": False})
    assert _option(chained.__self__.resolve_delayed()).no_gpu is False  # later wins
    plain = make_launcher(store)
    assert _option(plain.__self__.chain_delayed(_first)).delayed is _first
    assert _option(k.__self__.replace_delayed(None)).delayed is None


def test_delayed_editors_refuse_non_callables():
    k = make_launcher(store, **_HOST)
    with pytest.raises(TypeError, match="callable"):
        k.__self__.replace_delayed(7)  # pyright: ignore[reportArgumentType]
    with pytest.raises(TypeError, match="callable"):
        k.__self__.chain_delayed(None)  # pyright: ignore[reportArgumentType]


def test_chain_delayed_validates_each_function():
    k = make_launcher(store, delayed=lambda: [("no_gpu", True)])  # pyright: ignore[reportArgumentType]
    chained = k.__self__.chain_delayed(lambda: _HOST)
    with pytest.raises(TypeError, match="must return a mapping"):
        chained.__self__.check()


def test_concurrent_in_place_edits_both_land():
    k = make_launcher(store, extra_annotation={"x": _POINTER}, **_HOST)
    barrier = threading.Barrier(2)

    def bind(name):
        barrier.wait()
        for i in range(200):
            k.__self__._bind(**{f"{name}{i}": i})

    threads = [threading.Thread(target=bind, args=(n,)) for n in ("a", "b")]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(_option(k).bind_values or {}) == 400


def test_in_place_edit_racing_the_first_call():
    started = threading.Event()

    def delayed():
        started.set()
        time.sleep(0.2)  # the other thread's edit waits on the build lock
        return _HOST

    k = make_launcher(
        store, extra_annotation={"x": _POINTER}, bind_values={"x": 0}, delayed=delayed
    )
    errors = []

    def edit():
        started.wait()
        try:
            k.__self__._bind(x=8)
        except RuntimeError as error:
            errors.append(str(error))

    t = threading.Thread(target=edit)
    t.start()
    assert k(0, 0, 1, 4) is None
    t.join()
    assert len(errors) == 1 and "options are fixed" in errors[0]
    assert _option(k).bind_values == {"x": 0}  # what the module was built from

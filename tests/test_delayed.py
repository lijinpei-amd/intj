# pyright: standard
"""`make_launcher(delayed=...)`: keywords computed on first use."""

import dis
import gc
import sys
import sysconfig
import threading
import time
import weakref

import pytest
import triton
import triton.language as tl

from intj import make_launcher
from intj.annotation import NEVER, Argument, BindValue
from intj.launcher import module_of
from intj.torch_intf.torch_abi import TorchAccessMode


@triton.jit
def store(x, n):
    tl.store(x + tl.arange(0, 1), n)


_HOST = {"no_gpu": True, "torch_access_mode": TorchAccessMode.INTERPRETER}
_POINTER = Argument(
    type=tl.pointer_type(tl.float32), specialize=NEVER, bind_value=BindValue.POINTER
)


def _counting(values):
    calls = []

    def delayed():
        calls.append(1)
        return values

    return delayed, calls


def test_each_launcher_resolves_its_own_delayed():
    delayed, calls = _counting(_HOST)
    base = make_launcher(store, extra_annotation={"x": _POINTER}, delayed=delayed)
    assert calls == [] and base.__self__.options.delayed is delayed
    assert base.__self__.options.jit_func is store  # known before any call
    children = [base.__self__.bind(x=0), base.__self__.bind_device(0, x=0)]
    assert children[0](0, 0, 1, 4) is None and children[1](0, 1, 4) is None
    assert calls == [1, 1]
    options = children[0].__self__.options
    assert options.delayed is None  # resolved by the build
    assert options.no_gpu is True and options.bind_values == {"x": 0}


def test_decorator_form():
    delayed, calls = _counting(_HOST)
    launch = make_launcher(delayed=delayed)(store)
    assert calls == []
    assert launch(0, 0, 1, 0, 4) is None and calls == [1]


def test_override_and_bad_results():
    # delayed replaces an eager keyword, whole key
    launch = make_launcher(
        store,
        verify_annotation=True,
        delayed=lambda: {**_HOST, "verify_annotation": False},
    )
    assert launch(0, 0, 1, 0, 4) is None
    assert launch.__self__.options.verify_annotation is False
    with pytest.raises(TypeError, match="must return a mapping"):
        make_launcher(store, delayed=lambda: [("no_gpu", True)]).__self__.check()  # pyright: ignore[reportArgumentType]
    for key in ("bogus", "jit_func", "delayed", 1):
        with pytest.raises(TypeError, match="unknown make_launcher keyword"):
            make_launcher(store, delayed=lambda key=key: {key: 1}).__self__.check()  # pyright: ignore[reportArgumentType]
    with pytest.raises(TypeError, match="delayed must be None or a callable"):
        make_launcher(store, delayed=_HOST)  # pyright: ignore[reportArgumentType]


def test_a_failure_is_not_cached():
    calls = []

    def delayed():
        calls.append(1)
        if len(calls) == 1:
            raise KeyError("first")
        return _HOST

    launch = make_launcher(store, delayed=delayed)
    with pytest.raises(KeyError, match="first"):  # unwrapped
        launch(0, 0, 1, 0, 4)
    assert launch(0, 0, 1, 0, 4) is None and len(calls) == 2


@pytest.mark.parametrize("siblings", [False, True])
def test_concurrent_first_calls_run_it_once_per_launcher(siblings):
    delayed, calls = _counting(_HOST)
    base = make_launcher(store, delayed=delayed)
    launchers = [base.__self__.bind() if siblings else base for _ in range(8)]
    barrier, errors = threading.Barrier(8), []

    def worker(k):
        barrier.wait()
        try:
            k(0, 0, 1, 0, 4)
        except BaseException as error:  # noqa: BLE001  report, do not hang
            errors.append(error)

    threads = [threading.Thread(target=worker, args=(k,)) for k in launchers]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == [] and calls == [1] * (8 if siblings else 1)


def test_results_enter_module_identity():
    def launcher(warps):
        return make_launcher(store, delayed=lambda: {"options": {"num_warps": warps}})

    four, eight, again = launcher(4), launcher(8), launcher(4)
    assert module_of(four) is not module_of(eight)
    assert module_of(four) is module_of(again)


def test_calling_itself_or_a_sibling_is_refused():
    seen = []
    box = {}

    def delayed():
        for call in (
            lambda: box["k"](0, 0, 1, 0, 4),
            lambda: box["k"].__self__.check(),
            lambda: box["sibling"].__self__.check(),
        ):
            try:
                call()
            except RuntimeError as error:
                seen.append(str(error))
        return _HOST

    box["k"] = make_launcher(store, delayed=delayed)
    box["sibling"] = box["k"].__self__.bind()
    assert box["k"](0, 0, 1, 0, 4) is None
    # k() and k.check() refused; sibling.check() runs delayed again, where all three are refused
    assert len(seen) == 5 and all("while it was being built" in s for s in seen)
    assert box["sibling"](0, 0, 1, 0, 4) is None  # refused once, not poisoned


@pytest.mark.parametrize("call_first", [False, True])
def test_a_cycle_through_the_closure_is_collected(call_first):
    class Owner:
        launch: object

    owner = Owner()
    # a default, not a free variable: `del owner` must not cut the cycle
    owner.launch = make_launcher(store, delayed=lambda owner=owner: (owner, _HOST)[1])
    if call_first:
        owner.launch(0, 0, 1, 0, 4)
    ref = weakref.ref(owner)
    del owner
    gc.collect()
    assert ref() is None


def test_a_delayed_launcher_keeps_the_builtin_fast_call():
    launch = make_launcher(store, delayed=lambda: _HOST)

    def call():
        return launch(0, 0, 1, 0, 4)

    for _ in range(100):
        assert call() is None
    free_threaded = bool(sysconfig.get_config_var("Py_GIL_DISABLED"))
    if sys.version_info >= (3, 11) and not (
        free_threaded and sys.version_info[:2] == (3, 13)
    ):
        names = {i.opname for i in dis.get_instructions(call, adaptive=True)}  # pyright: ignore[reportCallIssue]
        assert names & {
            "PRECALL_NO_KW_BUILTIN_FAST",
            "CALL_NO_KW_BUILTIN_FAST",
            "CALL_BUILTIN_FAST",
        }


def test_a_racing_first_call_of_a_sibling_is_refused_not_deadlocked():
    """`delayed` calls a sibling while another thread makes that sibling's first call."""
    started, seen, errors = threading.Event(), [], []
    box = {}

    def delayed():
        started.set()
        time.sleep(0.3)  # the other thread is inside k2's first call by now
        try:
            box["k2"].__self__.check()
        except RuntimeError as error:
            seen.append(str(error))
        return _HOST

    box["k1"] = make_launcher(store, delayed=delayed)
    box["k2"] = box["k1"].__self__.bind()

    def run(launch, wait=None):
        if wait is not None:
            wait.wait()
        try:
            launch(0, 0, 1, 0, 4)
        except BaseException as error:  # noqa: BLE001  report, do not hang
            errors.append(error)

    threads = [
        threading.Thread(target=run, args=(box["k1"],), daemon=True),
        threading.Thread(target=run, args=(box["k2"], started), daemon=True),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)
    assert not any(t.is_alive() for t in threads), "deadlocked"
    assert errors == []
    assert len(seen) == 1 and "while it was being built" in seen[0]

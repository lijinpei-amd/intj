# pyright: standard
"""Autotuned and heuristic kernels through make_launcher, on the GPU."""

import threading

import pytest
import torch
import triton
import triton.language as tl

from intj import NEVER, Argument, BindValue, Constexpr, make_launcher
from intj.launcher import UnsupportedKernel, triton_specialization


@triton.jit
def tagged(x, out, n, TAG: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(out + offs, tl.load(x + offs, mask=mask) + TAG, mask=mask)


def configs():
    return [
        triton.Config({"TAG": 1, "BLOCK": 32}),
        triton.Config({"TAG": 2, "BLOCK": 64}),
    ]


class Bench:
    """Deterministic do_bench: reads the TAG the config wrote, scores by table."""

    def __init__(self, out, table):
        self.out, self.table, self.calls = out, table, []
        self.key: object = None

    def __call__(self, kernel_call, quantiles):
        self.out.zero_()
        kernel_call()
        tag = int(self.out[0].item())
        self.calls.append(tag)
        score = self.table[self.key][tag]
        return [score, score, score]


def grid(n: int, BLOCK: int):
    return (triton.cdiv(n, BLOCK),)


def controls():
    return torch.cuda.current_device(), torch.cuda.current_stream().cuda_stream


def tuned_launcher(bench, **kwargs):
    kernel = triton.autotune(configs=configs(), key=["n"], do_bench=bench)(tagged)
    return kernel, make_launcher(kernel, grid_cpp=grid, **kwargs)


def test_tunes_on_miss_and_launches_natively_on_hit():
    x = torch.zeros(256, device="cuda", dtype=torch.int32)
    out = torch.zeros_like(x)
    bench = Bench(out, {16: {1: 1.0, 2: 2.0}, 64: {1: 2.0, 2: 1.0}})
    kernel, launch = tuned_launcher(bench)
    device, stream = controls()
    for n, tag in ((16, 1), (64, 2)):
        bench.key = n
        out.zero_()
        launch(device, stream, x, out, n)
        torch.cuda.synchronize()
        assert int(out[0].item()) == tag
        assert kernel.best_config.kwargs["TAG"] == tag
    assert bench.calls == [1, 2, 1, 2]
    for n, tag in ((16, 1), (64, 2), (16, 1)):
        out.zero_()
        launch(device, stream, x, out, n)
        torch.cuda.synchronize()
        assert int(out[n - 1].item()) == tag
    assert bench.calls == [1, 2, 1, 2], "a hit must not reach Triton"


def test_float_autotune_key_is_exact():
    @triton.jit
    def scaled(x, out, s, TAG: tl.constexpr):
        tl.store(out, TAG + (s * 0).to(tl.int32))

    x = torch.zeros(1, device="cuda", dtype=torch.int32)
    out = torch.zeros_like(x)
    bench = Bench(out, {None: {1: 1.0, 2: 2.0}})
    kernel = triton.autotune(
        configs=[triton.Config({"TAG": 1}), triton.Config({"TAG": 2})],
        key=["s"],
        do_bench=bench,
    )(scaled)
    launch = make_launcher(kernel)
    device, stream = controls()
    for s in (1.0, 1.5, 1.0):
        launch(device, stream, 1, x, out, s)
    assert bench.calls == [1, 2, 1, 2]


def test_heuristic_over_tuned_value_is_stored_not_recomputed():
    @triton.jit
    def half(out, TAG: tl.constexpr, HALF: tl.constexpr):
        tl.store(out, TAG * 10 + HALF)

    out = torch.zeros(1, device="cuda", dtype=torch.int32)
    bench = Bench(out, {None: {10: 2.0, 21: 1.0}})  # TAG * 10 + HALF
    seen = _SEEN
    seen.clear()

    kernel = triton.autotune(
        configs=[triton.Config({"TAG": 1}), triton.Config({"TAG": 2})],
        key=[],
        do_bench=bench,
    )(triton.heuristics({"HALF": pick_half})(half))
    launch = make_launcher(kernel)
    device, stream = controls()
    launch(device, stream, 1, out)
    calls = len(seen)
    launch(device, stream, 1, out)
    torch.cuda.synchronize()
    assert int(out.item()) == 21
    assert len(seen) == calls, "a dependent heuristic runs on misses only"


_SEEN = []


def pick_half(a):
    # a global, not a closure: heuristics may not reference free variables
    return _SEEN.append(a["TAG"]) or a["TAG"] // 2


@pytest.mark.parametrize("mode", ["grid_arg", "default", "grid_py"])
def test_grid_modes(mode):
    x = torch.zeros(256, device="cuda", dtype=torch.int32)
    out = torch.zeros_like(x)
    bench = Bench(out, {None: {1: 2.0, 2: 1.0}})
    kernel = triton.autotune(configs=configs(), key=[], do_bench=bench)(tagged)
    device, stream = controls()
    metas = []
    if mode == "grid_arg":
        launch = make_launcher(kernel, grid_arg=1)
        call = lambda: launch(device, stream, 4, x, out, 256)  # noqa: E731
    elif mode == "default":
        launch = make_launcher(kernel)
        call = lambda: launch(device, stream, (4,), x, out, 256)  # noqa: E731
    else:

        def grid_py(meta):
            metas.append(meta["BLOCK"])
            return (triton.cdiv(meta["n"], meta["BLOCK"]),)

        launch = make_launcher(kernel, grid_py=grid_py)
        call = lambda: launch(device, stream, x, out, 256)  # noqa: E731
    call()
    out.zero_()
    call()
    torch.cuda.synchronize()
    assert int(out[255].item()) == 2
    if mode == "grid_py":
        assert metas[-1] == 64


def test_return_compiled_is_the_selected_kernel():
    x = torch.zeros(64, device="cuda", dtype=torch.int32)
    out = torch.zeros_like(x)
    bench = Bench(out, {None: {1: 2.0, 2: 1.0}})
    kernel = triton.autotune(configs=configs(), key=[], do_bench=bench)(tagged)
    launch = make_launcher(kernel, grid_cpp=grid, return_compiled=True)
    device, stream = controls()
    first = launch(device, stream, x, out, 64)
    assert launch(device, stream, x, out, 64) is first
    assert first.src.constants[(3,)] == 2


def test_tuned_param_before_public_params():
    @triton.jit
    def early(out, TAG: tl.constexpr, n):
        tl.store(out + n - n, TAG)

    out = torch.zeros(1, device="cuda", dtype=torch.int32)
    bench = Bench(out, {None: {1: 2.0, 2: 1.0}})
    kernel = triton.autotune(
        configs=[triton.Config({"TAG": 1}), triton.Config({"TAG": 2})],
        key=["n"],
        do_bench=bench,
    )(early)
    launch = make_launcher(kernel)
    device, stream = controls()
    launch(device, stream, 1, out, 7)
    torch.cuda.synchronize()
    assert int(out.item()) == 2


def test_concurrent_misses_tune_once():
    x = torch.zeros(256, device="cuda", dtype=torch.int32)
    out = torch.zeros_like(x)
    bench = Bench(out, {16: {1: 1.0, 2: 2.0}})
    bench.key = 16
    _, launch = tuned_launcher(bench)
    device, _ = controls()
    errors = []

    def worker():
        try:
            with torch.cuda.stream(torch.cuda.Stream()):
                launch(device, torch.cuda.current_stream().cuda_stream, x, out, 16)
        except Exception as error:  # pragma: no cover - reported below
            errors.append(error)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert bench.calls == [1, 2]


def test_config_compile_options_are_per_record():
    @triton.jit
    def fill(out, n, TAG: tl.constexpr, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        tl.store(out + offs, TAG, mask=offs < n)

    out = torch.zeros(1024, device="cuda", dtype=torch.int32)
    bench = Bench(out, {128: {1: 1.0, 2: 2.0}, 1024: {1: 2.0, 2: 1.0}})
    kernel = triton.autotune(
        configs=[
            triton.Config({"TAG": 1, "BLOCK": 256}, num_warps=1),
            triton.Config({"TAG": 2, "BLOCK": 256}, num_warps=8),
        ],
        key=["n"],
        do_bench=bench,
    )(fill)
    launch = make_launcher(kernel, grid_cpp=lambda_grid, return_compiled=True)
    device, stream = controls()
    for n, warps in ((128, 1), (1024, 8)):
        bench.key = n
        out.zero_()
        compiled = launch(device, stream, out, n)
        torch.cuda.synchronize()
        assert compiled.metadata.num_warps == warps
        assert int(out[:n].min().item()) == int(out[:n].max().item()) != 0


def lambda_grid(n: int, BLOCK: int):
    return (triton.cdiv(n, BLOCK),)


def test_failed_tuning_leaves_no_record():
    x = torch.zeros(64, device="cuda", dtype=torch.int32)
    out = torch.zeros_like(x)
    bench = Bench(out, {None: {1: 2.0, 2: 1.0}})
    failures = [RuntimeError("prune failed")]

    def prune(configs, named_args, **kwargs):
        if failures:
            raise failures.pop()
        return configs

    kernel = triton.autotune(
        configs=configs(),
        key=[],
        do_bench=bench,
        prune_configs_by={"early_config_prune": prune},
    )(tagged)
    launch = make_launcher(kernel, grid_cpp=grid)
    device, stream = controls()
    with pytest.raises(RuntimeError, match="prune failed"):
        launch(device, stream, x, out, 64)
    launch(device, stream, x, out, 64)
    torch.cuda.synchronize()
    assert int(out[0].item()) == 2


def test_autotune_key_refuses_a_tensor_at_call():
    x = torch.zeros(64, device="cuda", dtype=torch.int32)
    kernel = triton.autotune(configs=configs(), key=["n"])(tagged)
    launch = make_launcher(kernel, grid_arg=1)
    device, stream = controls()
    with pytest.raises(TypeError, match="autotune key 'n'"):
        launch(device, stream, 1, x, x, x)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"no_gpu": True}, "no_gpu"),
        ({"options": {"num_warps": 8}}, "num_warps"),
        ({"extra_annotation": {"TAG": Constexpr(value=1)}}, "TAG"),
        (
            {
                "extra_annotation": {
                    "x": Argument(
                        type=tl.pointer_type(tl.int32),
                        specialize=NEVER,
                        bind_value=BindValue.TENSOR,
                    )
                }
            },
            "bound",
        ),
    ],
)
def test_refusals(kwargs, match):
    kernel = triton.autotune(configs=configs(), key=["n"])(tagged)
    with pytest.raises(UnsupportedKernel, match=match):
        make_launcher(kernel, **kwargs)


def test_default_restore_and_reset_hooks_stay_private():
    @triton.jit
    def bump(x, acc, TAG: tl.constexpr):
        tl.store(x, tl.load(x) + TAG)
        tl.store(acc, tl.load(acc) + TAG)

    x = torch.full((1,), 100, device="cuda", dtype=torch.int32)
    acc = torch.zeros_like(x)
    bench = Bench(acc, {None: {1: 2.0, 2: 1.0}})
    kernel = triton.autotune(
        configs=[triton.Config({"TAG": 1}), triton.Config({"TAG": 2})],
        key=[],
        restore_value=["x"],
        reset_to_zero=["acc"],
        do_bench=bench,
    )(bump)
    launch = make_launcher(kernel)
    device, stream = controls()
    launch(device, stream, 1, x, acc)
    torch.cuda.synchronize()
    assert bench.calls == [1, 2]
    assert int(x.item()) == 102, (
        "benchmark runs restore x; only the final launch sticks"
    )
    assert int(acc.item()) == 2, "acc is reset after benchmarking"
    assert "restore_copies" not in kernel.__dict__, (
        "the user's tuner must not be written"
    )


@triton.jit
def aligned_tag(x, out, N, stride, BLOCK: tl.constexpr, ALIGNED: tl.constexpr):
    tl.store(out, BLOCK * 10 + ALIGNED + (N + stride) * 0)


def test_two_level_chain():
    out = torch.zeros(1, device="cuda", dtype=torch.int32)
    x = torch.zeros(1, device="cuda", dtype=torch.int32)

    class ByBlock(Bench):
        def __call__(self, kernel_call, quantiles):
            self.out.zero_()
            kernel_call()
            block = int(self.out.item()) // 10
            self.calls.append(block)
            return [1.0 if block == 64 else 2.0] * 3

    bench = ByBlock(out, None)
    kernel = triton.autotune(
        configs=[triton.Config({"BLOCK": 32}), triton.Config({"BLOCK": 64})],
        key=["N"],
        do_bench=bench,
    )(
        triton.heuristics({"ALIGNED": lambda a: a["stride"] % a["BLOCK"] == 0})(
            aligned_tag
        )
    )
    launch = make_launcher(kernel)
    device, stream = controls()
    for stride, aligned in ((128, 1), (130, 0), (192, 1)):
        launch(device, stream, 1, x, out, 8, stride)
        torch.cuda.synchronize()
        assert int(out.item()) == 640 + aligned
    assert bench.calls == [32, 64], (
        "one tuning run for N=8; strides only add level-1 keys"
    )
    module = launch.__self__.__self__
    keys, found = module.key_chain(launch, device, x, out, 8, 256)
    assert found and len(keys) == 2


@triton.jit
def even_store(out, n, EVEN: tl.constexpr):
    tl.store(out, EVEN + n * 0)


def test_heuristics_only_computed_key_shares_records():
    kernel = triton.heuristics({"EVEN": lambda a: a["n"] % 2 == 0})(even_store)
    launch = make_launcher(kernel, return_compiled=True)
    out = torch.zeros(1, device="cuda", dtype=torch.int32)
    device, stream = controls()
    a = launch(device, stream, 1, out, 2)
    b = launch(device, stream, 1, out, 4)
    c = launch(device, stream, 1, out, 3)
    assert a is b and a is not c
    torch.cuda.synchronize()
    assert int(out.item()) == 0


@triton.jit
def maybe_strided(out, b, B_UNIT: tl.constexpr):
    tl.store(out, B_UNIT)


def test_computed_key_short_circuits_before_tensor_reads():
    kernel = triton.heuristics(
        {"B_UNIT": lambda a: a["b"] is not None and a["b"].stride(0) == 1}
    )(maybe_strided)
    launch = make_launcher(kernel)
    out = torch.zeros(1, device="cuda", dtype=torch.int32)
    device, stream = controls()
    launch(device, stream, 1, out, None)
    torch.cuda.synchronize()
    assert int(out.item()) == 0
    launch(device, stream, 1, out, torch.zeros(4, device="cuda"))
    torch.cuda.synchronize()
    assert int(out.item()) == 1
    launch(device, stream, 1, out, torch.zeros(4, 2, device="cuda").t()[0])
    torch.cuda.synchronize()
    assert int(out.item()) == 0


@triton.jit
def kinds(out, n, K: tl.constexpr):
    tl.store(out, K + n * 0)


def test_computed_key_keeps_bool_and_int_apart():
    # n = 3 and n = 7 share a spec key; K is 1 for one and True for the other.
    # A key holding only the value would launch the int-specialized binary for
    # the bool (Triton compiles them apart).
    kernel = triton.heuristics({"K": lambda a: a["n"] > 0 if a["n"] > 5 else 1})(kinds)
    launch = make_launcher(kernel, return_compiled=True)
    out = torch.zeros(1, device="cuda", dtype=torch.int32)
    device, stream = controls()
    as_int = launch(device, stream, 1, out, 3)
    as_bool = launch(device, stream, 1, out, 7)
    assert as_int is not as_bool


@triton.jit
def lowered(out, x, y, n, m, K: tl.constexpr):
    tl.store(out, K + (n + m) * 0)


# (heuristic, (n, m) cases).  Cases are ordered so that a wrong lowering maps a
# later case onto an earlier case's (value, kind) and hits its record, which
# only the checks here catch: -7 // 2 truncated is -3, the value of -6 // 2;
# 7 % -2 truncated is 1, the value of 7 % 2.
LOWERED_SCALAR = [
    (lambda a: a["n"] // a["m"], [(-6, 2), (-7, 2), (7, 2), (7, -2), (-7, -2), (0, 3)]),
    (lambda a: a["n"] % a["m"], [(7, 2), (7, -2), (-7, 2), (-7, -2), (6, -2), (0, 3)]),
    (
        lambda a: min(a["n"], a["m"]),
        [(1, True), (True, 1), (0, False), (False, 0), (5, 3), (-3, 5)],
    ),
    (lambda a: max(a["n"], a["m"]), [(1, True), (True, 1), (5, 3), (-3, -5)]),
    (
        lambda a: a["n"] and a["m"],
        [(3, 5), (0, 5), (False, 5), (3, True), (3, 0), (True, 0)],
    ),
    (lambda a: a["n"] or a["m"], [(3, 5), (0, 5), (0, False), (False, 0), (True, 7)]),
    (
        lambda a: -a["n"] if a["m"] > 1 else not a["n"],
        [(4, 2), (4, 0), (0, 0), (-3, 5)],
    ),
    (
        lambda a: triton.next_power_of_2(a["n"]) * 1000 + triton.cdiv(a["n"], a["m"]),
        [(5, 2), (0, 1), (1, 1), (-7, 3), (16, 16), (17, 16), (17, -4)],
    ),
    (lambda a: a["n"] < a["m"] or a["n"] == 7, [(1, 2), (2, 1), (7, 1), (2, 2)]),
]


def _check_lowered(h, cases):
    launch = make_launcher(triton.heuristics({"K": h})(lowered), return_compiled=True)
    out = torch.zeros(1, device="cuda", dtype=torch.int32)
    device, stream = controls()
    for x, y, n, m in cases:
        expected = h({"x": x, "y": y, "n": n, "m": m})
        kernel = launch(device, stream, 1, out, x, y, n, m)
        torch.cuda.synchronize()
        got = kernel.src.constants[(5,)]
        assert (got, type(got)) == (expected, type(expected)), (x, y, n, m)
        assert int(out.item()) == int(expected), (x, y, n, m)


@pytest.mark.parametrize("h,cases", LOWERED_SCALAR, ids=range(len(LOWERED_SCALAR)))
def test_lowered_heuristic_matches_python(h, cases):
    x = torch.zeros(1, device="cuda")
    _check_lowered(h, [(x, None, n, m) for n, m in cases])


LOWERED_TENSOR = [
    lambda a: a["x"].shape[0] * 100 + a["x"].stride(0) * 10 + a["x"].dim(),
    lambda a: a["x"].numel() * 10 + a["x"].element_size(),
    lambda a: a["x"].is_contiguous(),
    lambda a: a["x"].dtype == a["y"].dtype,
    lambda a: a["y"] is not None and a["y"].size(0) > 2,
]


@pytest.mark.parametrize("h", LOWERED_TENSOR, ids=range(len(LOWERED_TENSOR)))
def test_lowered_tensor_heuristic_matches_python(h):
    flat = torch.zeros(8, device="cuda")
    strided = torch.zeros(4, 2, device="cuda", dtype=torch.float16).t()[0]
    pairs: list[tuple[torch.Tensor, object]] = [
        (flat, flat),
        (strided, flat),
        (flat, strided),
        (strided, strided),
    ]
    if h is LOWERED_TENSOR[-1]:  # the only one that reads y as optional
        pairs.append((flat, None))
    _check_lowered(h, [(x, y, 1, 2) for x, y in pairs])


def _reference(kernel, args_by_name):
    """What Triton itself decides for these arguments: its tuning keys, the
    heuristic outputs and the final call, from a private chain whose innermost
    layer records instead of launching."""
    import copy

    from triton.runtime.autotuner import Autotuner

    layers, inner = [], kernel
    while not hasattr(inner, "params"):
        layers.append(copy.copy(inner))
        inner = inner.fn
    final = {}

    class Record:
        fn = inner

        def run(self, *args, grid, warmup, **kwargs):
            final.update(zip(inner.arg_names, args))
            final.update(kwargs)

    for outer, nxt in zip(layers, [*layers[1:], Record()]):
        outer.fn = nxt
    tuning_keys = []
    for layer in layers:
        if type(layer) is Autotuner:
            layer.cache = {}
    layers[0].run(grid=(1,), warmup=False, **args_by_name)
    for layer in layers:
        if type(layer) is Autotuner:
            tuning_keys.append(tuple(layer.cache))
    params = [final[n] for n in inner.arg_names]
    options = {k: v for k, v in final.items() if k not in inner.arg_names}
    # only what the layers decided; raw arguments like `stride` are not keyed
    heuristics = tuple(
        sorted((k, repr(v)) for k, v in final.items() if k not in args_by_name)
    )
    return (
        repr(triton_specialization(inner, params, options)),
        tuple(tuning_keys),
        heuristics,
    )


def check_tuned_invariant(make_kernel, cases):
    kernel = make_kernel()
    launch = make_launcher(kernel)
    module = launch.__self__.__self__
    device, stream = controls()
    seen = {}
    for case in cases:
        launch(device, stream, 1, *case.values())
        chain, found = module.key_chain(launch, device, *case.values())
        assert found
        truth = _reference(make_kernel(), case)
        assert seen.setdefault(chain, truth) == truth, (
            "intj key chain collides across Triton decisions"
        )


def _invariant_kernel():
    def bench(call, quantiles):  # first config wins, deterministically
        return [1.0, 1.0, 1.0]

    return triton.autotune(
        configs=[triton.Config({"BLOCK": 32}), triton.Config({"BLOCK": 64})],
        key=["N"],
        do_bench=bench,
    )(
        triton.heuristics({"ALIGNED": lambda a: a["stride"] % a["BLOCK"] == 0})(
            aligned_tag
        )
    )


def _invariant_cases():
    x = torch.zeros(1, device="cuda", dtype=torch.int32)
    out = torch.zeros_like(x)
    half = torch.zeros(1, device="cuda", dtype=torch.float16)
    return [
        {"x": t, "out": out, "N": n, "stride": s}
        for t in (x, half)
        for n in (1, 8, 16, 17, 2**31)
        for s in (0, 32, 33, 64)
    ]


def test_tuned_key_chain_is_never_coarser_than_triton():
    check_tuned_invariant(_invariant_kernel, _invariant_cases())


def test_tuned_invariant_catches_a_dropped_exact_key(monkeypatch):
    from intj import annotation

    original = annotation._key_fields

    def without_exact(a):
        return tuple(f for f in original(a) if f.kind not in ("exact", "exact_kind"))

    monkeypatch.setattr(annotation, "_key_fields", without_exact)
    monkeypatch.setattr("intj.launcher._key_fields", without_exact)
    with pytest.raises(AssertionError, match="collides"):
        check_tuned_invariant(_invariant_kernel, _invariant_cases())

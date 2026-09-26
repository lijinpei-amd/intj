# pyright: standard
"""Autotuned and heuristic kernels through make_launcher, on the GPU."""

import threading

import pytest
import torch
import triton
import triton.language as tl

from intj import NEVER, Argument, BindValue, Constexpr, make_launcher
from intj.launcher import UnsupportedKernel


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


def test_computed_keys_are_refused_until_task_4():
    @triton.jit
    def evens(out, n, EVEN: tl.constexpr):
        tl.store(out, EVEN)

    with pytest.raises(UnsupportedKernel, match="computed"):
        make_launcher(triton.heuristics({"EVEN": lambda a: a["n"] % 2 == 0})(evens))


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

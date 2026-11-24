# pyright: standard
"""Role assignment and refusals for autotune and heuristics layers; nothing here launches."""

import math

import pytest
import torch
import triton
import triton.language as tl

from intj import Argument, make_launcher
from intj.heuristic import HeuristicError, parse_heuristic
from intj.launcher import UnsupportedKernel
from intj.tuning import Computed, Dependent, analyze


@triton.jit
def mm(
    a, b, c, M, N, K, BLOCK_K: tl.constexpr, SPLIT_K: tl.constexpr, EVEN_K: tl.constexpr
):
    pass


@triton.jit
def strided(x, N, stride, BLOCK: tl.constexpr, ALIGNED: tl.constexpr):
    pass


@triton.jit
def evens(x, N, EVEN_N: tl.constexpr):
    pass


BLOCKS = [triton.Config({"BLOCK": 32}), triton.Config({"BLOCK": 64})]

CONFIGS_K = [
    triton.Config({"BLOCK_K": 32, "SPLIT_K": 1}),
    triton.Config({"BLOCK_K": 64, "SPLIT_K": 2}, num_warps=8),
]


def test_matmul_needs_one_lookup():
    kernel = triton.autotune(configs=CONFIGS_K, key=["M", "N", "K"])(
        triton.heuristics(
            {"EVEN_K": lambda a: a["K"] % (a["BLOCK_K"] * a["SPLIT_K"]) == 0}
        )(mm)
    )
    plan = analyze(kernel)
    assert plan.exact_keys == ("M", "N", "K")
    assert plan.levels == 1
    assert plan.computed == ()
    assert set(plan.dependent) == {
        Dependent(name, 0)
        for name in (
            "BLOCK_K",
            "SPLIT_K",
            "num_warps",
            "num_ctas",
            "num_stages",
            "EVEN_K",
        )
    }
    assert plan.tuned >= {"BLOCK_K", "SPLIT_K", "EVEN_K"}


def test_heuristic_over_unkeyed_var_and_tuned_value_is_a_level_one_key():
    kernel = triton.autotune(
        configs=[triton.Config({"BLOCK": 32}), triton.Config({"BLOCK": 64})], key=["N"]
    )(triton.heuristics({"ALIGNED": lambda a: a["stride"] % a["BLOCK"] == 0})(strided))
    plan = analyze(kernel)
    assert plan.levels == 2
    assert [(c.name, c.level) for c in plan.computed] == [("ALIGNED", 1)]
    assert Dependent("BLOCK", 0) in plan.dependent


def test_heuristic_over_caller_var_is_a_level_zero_key():
    plan = analyze(triton.heuristics({"EVEN_N": lambda a: a["N"] % 2 == 0})(evens))
    assert plan.levels == 1
    assert plan.exact_keys == ()
    assert [(c.name, c.level) for c in plan.computed] == [("EVEN_N", 0)]


def test_heuristics_outside_autotune_feed_its_key():
    kernel = triton.heuristics({"EVEN_N": lambda a: a["N"] % 2 == 0})(
        triton.autotune(configs=BLOCKS, key=["EVEN_N"])(strided_even)
    )
    plan = analyze(kernel)
    assert [(c.name, c.level) for c in plan.computed] == [("EVEN_N", 0)]
    assert Dependent("BLOCK", 0) in plan.dependent


@triton.jit
def strided_even(x, N, BLOCK: tl.constexpr, EVEN_N: tl.constexpr):
    pass


def test_baked_var_counts_as_exact():
    plan = analyze(
        triton.heuristics({"EVEN_N": lambda a: a["N"] % 2 == 0})(evens), fixed={"N"}
    )
    assert plan.computed == ()
    assert plan.dependent == (Dependent("EVEN_N", 0),)


@pytest.mark.parametrize(
    "build,match",
    [
        (
            lambda: triton.autotune(configs=[triton.Config({"N": 1})], key=[])(evens),
            "runtime parameter 'N'",
        ),
        (
            lambda: triton.heuristics({"EVEN_N": lambda a: a["EVEN_N"]})(evens),
            "its own result",
        ),
        (lambda: triton.heuristics({"EVEN_N": lambda a: a["nope"]})(evens), "'nope'"),
        (
            lambda: triton.heuristics({"EVEN_N": lambda a: 1})(
                triton.heuristics({"EVEN_N": lambda a: 0})(evens)
            ),
            "two layers",
        ),
        (
            lambda: triton.heuristics({"EVEN_N": lambda a: a.get("N")})(evens),
            "string literal",
        ),
    ],
)
def test_refusals(build, match):
    with pytest.raises(UnsupportedKernel, match=match):
        analyze(build())


def test_parse_finds_the_right_lambda_on_a_shared_line():
    first, second = (lambda a: a["N"] + 1), (lambda a: a["M"] * 2)  # noqa: E731
    assert parse_heuristic("X", first).inputs == ("N",)
    assert parse_heuristic("Y", second).inputs == ("M",)


def test_parse_accepts_a_single_return_def():
    def pick(args):
        """Doc."""
        return args["N"] > 4

    assert parse_heuristic("X", pick).inputs == ("N",)


def test_parse_refuses_free_variables():
    limit = 4
    with pytest.raises(HeuristicError, match="free variables"):
        parse_heuristic("X", lambda a: a["N"] > limit)


from intj.heuristic import Source, lower


@pytest.mark.parametrize(
    "fn,match",
    [
        (lambda a: a["N"] / 2, "operator"),
        (lambda a: a["N"] < a["M"] < 4, "chained"),
        (lambda a: len(a["x"]), "call"),
        (lambda a: a["x"].stride(a["N"]), "literal"),
        (lambda a: a["N"] ** 2, "operator"),
    ],
)
def test_lowering_refusals(fn, match):
    h = parse_heuristic("K", fn)
    sources = {name: Source("arg", i) for i, name in enumerate(h.inputs)}
    with pytest.raises(HeuristicError, match=match):
        lower([(h, 0)], sources, 1)


def test_heuristic_calling_triton_is_located_and_lowered():
    # This file imports triton, so CPython compiles `triton.cdiv(...)` here
    # differently from the lambda's node compiled alone.
    h = parse_heuristic("K", lambda a: triton.cdiv(a["N"], 2))
    assert h.inputs == ("N",)
    assert "intj_grid_cdiv" in lower([(h, 0)], {"N": Source("arg", 0)}, 1)


@pytest.mark.parametrize(
    "build",
    [
        lambda: triton.heuristics({"ALIGNED": lambda a: a["N"] % a["BLOCK"] == 0})(
            triton.autotune(configs=[triton.Config({"BLOCK": 64})], key=["N"])(strided)
        ),
        lambda: triton.autotune(configs=BLOCKS, key=["ALIGNED"])(
            triton.heuristics({"ALIGNED": lambda a: a["N"] % 2 == 0})(strided)
        ),
    ],
    ids=["heuristic", "autotune_key"],
)
def test_refuses_reading_a_value_an_inner_layer_assigns(build):
    with pytest.raises(UnsupportedKernel, match="inner layer"):
        analyze(build())


@triton.jit
def logged(x, N, BLOCK: tl.constexpr, LOG_N: tl.constexpr):
    pass


# C cannot lower it: it runs in Python, so what it reads must be keyed
LOG_N = {"LOG_N": lambda a: math.ceil(math.log2(a["N"]))}


@pytest.mark.parametrize("outer", ["heuristics", "autotune"])
def test_an_autotune_key_is_exact_for_every_layer(outer):
    heuristics = triton.heuristics(LOG_N)
    autotune = triton.autotune(configs=BLOCKS, key=["N"])
    if outer == "heuristics":
        kernel = heuristics(autotune(logged))
    else:
        kernel = autotune(heuristics(logged))
    plan = analyze(kernel)
    assert plan.exact_keys == ("N",)
    assert plan.computed == ()
    assert Dependent("LOG_N", 0) in plan.dependent
    make_launcher(kernel, grid_arg_num=1).__self__.check()


def test_a_single_config_autotuner_keys_nothing():
    # Triton reads the key only to choose among configs
    plan = analyze(triton.autotune(configs=BLOCKS[:1], key=["N"])(strided))
    assert plan.exact_keys == ()
    assert plan.dependent[0] == Dependent("BLOCK", 0)
    inner = triton.heuristics({"ALIGNED": lambda a: a["N"] % 2 == 0})(strided)
    analyze(triton.autotune(configs=BLOCKS[:1], key=["ALIGNED"])(inner))


@pytest.mark.parametrize("others", [0, 1])
def test_a_config_pre_hook_is_refused(others):
    hooked = triton.Config({"BLOCK": 32}, pre_hook=lambda nargs: None)
    configs = [hooked, *BLOCKS[1 : 1 + others]]
    kernel = triton.autotune(configs=configs, key=["N"])(strided)
    with pytest.raises(UnsupportedKernel, match=r"\{BLOCK: 32.*\} has a pre_hook"):
        make_launcher(kernel, grid_arg_num=1).__self__.check()


def test_a_tuner_pre_hook_runs_only_while_benchmarking_so_is_kept():
    kernel = triton.autotune(
        configs=BLOCKS, key=["N"], pre_hook=lambda nargs, reset_only=False: None
    )(strided)
    make_launcher(kernel, grid_arg_num=1).__self__.check()


@pytest.mark.parametrize(
    "annotation",
    [
        Argument(type=tl.pointer_type(tl.int32)),
        Argument(type=(tl.pointer_type(tl.int32), None)),
    ],
    ids=["pointer", "optional_pointer"],
)
def test_an_autotune_key_annotated_a_tensor_is_refused_at_check(annotation):
    kernel = triton.autotune(configs=BLOCKS, key=["x"])(strided)
    with pytest.raises(UnsupportedKernel, match="autotune key 'x' is a tensor"):
        make_launcher(
            kernel, grid_arg_num=1, extra_annotation={"x": annotation}
        ).__self__.check()
    make_launcher(kernel, grid_arg_num=1).__self__.check()  # refused at the call


def test_private_chain_rebinds_triton_default_hooks_to_the_copy():
    # Triton's own reset/restore closures, over the copy: this fails once
    # Triton's default hooks close over anything but their tuner
    from intj.tuning import _private_chain, _Shim

    tuner = triton.autotune(
        configs=BLOCKS, key=["N"], reset_to_zero=["x"], restore_value=["stride"]
    )(strided)
    (copy,) = _private_chain((tuner,), _Shim(strided, lambda values, options: None))
    x, stride = torch.ones(2), torch.full((2,), 7.0)
    copy.pre_hook({"x": x, "stride": stride})
    assert x.tolist() == [0.0, 0.0]
    assert "restore_copies" not in tuner.__dict__
    stride.fill_(1.0)
    copy.post_hook({"x": x, "stride": stride}, None)
    assert stride.tolist() == [7.0, 7.0] and copy.restore_copies == {}

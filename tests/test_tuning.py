# pyright: standard
"""Role assignment for autotune and heuristics layers; nothing here launches."""

import pytest
import triton
import triton.language as tl

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
        triton.autotune(configs=[triton.Config({"BLOCK": 32})], key=["EVEN_N"])(
            strided_even
        )
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
        lambda: triton.autotune(
            configs=[triton.Config({"BLOCK": 64})], key=["ALIGNED"]
        )(triton.heuristics({"ALIGNED": lambda a: a["N"] % 2 == 0})(strided)),
    ],
    ids=["heuristic", "autotune_key"],
)
def test_refuses_reading_a_value_an_inner_layer_assigns(build):
    with pytest.raises(UnsupportedKernel, match="inner layer"):
        analyze(build())

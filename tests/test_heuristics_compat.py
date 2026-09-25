# pyright: standard
"""Triton heuristic decorators in the migration adapter."""

import torch
import triton
import triton.language as tl

from intj.compat import launch


@triton.heuristics({"EVEN_N": lambda meta: meta["N"] % 2 == 0})
@triton.heuristics({"SHIFT": lambda meta: meta["x"].numel() + int(meta["EVEN_N"])})
@triton.jit
def heuristic_add(
    x,
    out,
    N,
    EVEN_N: tl.constexpr,
    SHIFT: tl.constexpr,
    BLOCK: tl.constexpr = 32,  # pyright: ignore[reportArgumentType]  # Triton permits constexpr defaults
):
    i = tl.arange(0, BLOCK)
    tl.store(out + i, tl.load(x + i, i < N, other=0) + SHIFT, i < N)


def test_heuristics_match_triton_and_reach_python_grid():
    x = torch.arange(6, device="cuda", dtype=torch.int32)
    for n in (5, 6):
        expected = torch.full_like(x, -1)
        actual = torch.full_like(x, -1)
        # Triton overwrites a caller-supplied heuristic value before the inner decorator runs.
        heuristic_add[(1,)](x, expected, N=n, EVEN_N=True)
        launch(heuristic_add, (1,), x, actual, N=n, EVEN_N=True)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(actual[:n], x[:n] + 6 + int(n % 2 == 0))

    seen = []

    def grid(meta):
        seen.append((meta["EVEN_N"], meta["SHIFT"], meta["BLOCK"]))
        return (1,)

    out = torch.full_like(x, -1)
    launch(heuristic_add, grid, x, out, N=5)
    assert seen == [(False, 6, 32)]
    torch.testing.assert_close(out[:5], x[:5] + 6)

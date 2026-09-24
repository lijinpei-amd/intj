# pyright: standard
"""Triton's reinterpret wrapper in the migration adapter."""

import torch
import triton
import triton.language as tl

from intj.compat import launch


@triton.jit
def read_reinterpreted(x, out, n):
    i = tl.arange(0, 32)
    value = tl.load(x + i, i < n, other=0).to(tl.float32)
    tl.store(out + i, value, i < n)


def test_reinterpret_matches_triton_and_separates_dtype_cache_keys():
    base = torch.tensor([0x3F800000, 0x40000000, 0x40400000],
                        device="cuda", dtype=torch.int32)
    results = []
    for dtype in (tl.int32, torch.float32, tl.float32, torch.int32):
        wrapped = triton.reinterpret(base, dtype)
        expected = torch.empty_like(base)
        actual = torch.empty_like(expected)
        read_reinterpreted[(1,)](wrapped, triton.reinterpret(expected, tl.float32), base.numel())
        launch(read_reinterpreted, (1,), wrapped,
               triton.reinterpret(actual, tl.float32), base.numel())
        torch.testing.assert_close(actual, expected)
        results.append(actual)
    assert not torch.equal(results[0], results[1])
    torch.testing.assert_close(results[1], results[2])
    torch.testing.assert_close(results[0], results[3])


def test_python_grid_keeps_original_wrapper_in_metadata():
    base = torch.tensor([0x3F800000], device="cuda", dtype=torch.int32)
    wrapped = triton.reinterpret(base, torch.float32)
    out = torch.empty_like(base)
    wrapped_out = triton.reinterpret(out, torch.float32)
    seen = []

    def grid(meta):
        seen.append((meta["x"], meta["out"]))
        return (1,)

    launch(read_reinterpreted, grid, wrapped, wrapped_out, 1)
    assert seen == [(wrapped, wrapped_out)]
    torch.testing.assert_close(out, base)

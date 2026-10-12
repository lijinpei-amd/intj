"""Triton and torch, imported on first use and cached for the whole package.

Importing intj must import neither, so nothing imports them at module top. A
function that needs one calls `ensure_triton_imported()` or
`ensure_torch_imported()`, then reads the names below, which are None until
then. Read them through this module (`_imports.knobs`): `from ._imports import
knobs` copies the None.
"""

from __future__ import annotations

from typing import Any

# triton and torch ship no type information, so every name here is `Any`.
triton: Any = None
knobs: Any = None
tl: Any = None
driver: Any = None
build: Any = None
ASTSource: Any = None
make_backend: Any = None
JITFunction: Any = None
MockTensor: Any = None
TensorWrapper: Any = None
Autotuner: Any = None
Heuristics: Any = None
InterpretedFunction: Any = None
canonicalize_dtype: Any = None
type_canonicalisation_dict: Any = None
native_specialize_impl: Any = None
#: `triton.cdiv` as first imported, so a later replacement can be told apart
cdiv: Any = None
#: stays None on a triton without the compile-warmup test runtime
is_compile_warmup: Any = None

torch: Any = None


def ensure_triton_imported() -> None:
    """Imports triton once and caches the names intj uses from it.

    Raises:
        UnsupportedKernel: Triton is missing or older than 3.7.
    """
    global triton, knobs, tl, driver, build, ASTSource, make_backend
    global JITFunction, MockTensor, TensorWrapper, Autotuner, Heuristics
    global InterpretedFunction, canonicalize_dtype, type_canonicalisation_dict
    global native_specialize_impl, cdiv, is_compile_warmup
    if triton is not None:
        return
    try:
        import triton as found
    except ModuleNotFoundError as e:
        if e.name != "triton":
            raise
        raise _triton_required("") from e
    if tuple(map(int, found.__version__.split(".")[:2])) < (3, 7):
        raise _triton_required(f" (found {found.__version__})")
    import triton.language as tl
    from triton import knobs
    from triton._C.libtriton import native_specialize_impl
    from triton._utils import canonicalize_dtype, type_canonicalisation_dict
    from triton.compiler import ASTSource, make_backend
    from triton.runtime import build
    from triton.runtime.autotuner import Autotuner, Heuristics
    from triton.runtime.driver import driver
    from triton.runtime.interpreter import InterpretedFunction
    from triton.runtime.jit import JITFunction, MockTensor, TensorWrapper

    try:
        from triton._compile_warmup_state import is_compile_warmup  # pyright: ignore[reportMissingImports]  # optional Triton test runtime
    except ImportError:  # Triton versions without compile-warmup mode
        is_compile_warmup = None
    cdiv = found.cdiv
    triton = found  # last: once it is set, so is every name above


def _triton_required(found: str) -> Exception:
    from .launcher import UnsupportedKernel  # not at top: launcher imports this

    return UnsupportedKernel(
        f"intj: Triton >=3.7 is required; install `intj[launcher]`{found}"
    )


def ensure_torch_imported() -> None:
    global torch
    if torch is None:
        import torch

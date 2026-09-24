# Callable grid design

The source survey is [callable-grid-survey.md](../../callable-grid-survey.md).
It measures the language constructs found in existing grids. The API below
uses explicit named parameters instead of Triton's `meta` argument, so the
survey's unchanged-source coverage count does not apply to `grid_cpp`.

## Public API

`make_launcher` gains three mutually exclusive options:

```python
make_launcher(kernel, grid_arg=1)      # launch(device, stream, gx, *kernel_args)
make_launcher(kernel, grid_arg=2)      # launch(device, stream, gx, gy, *kernel_args)
make_launcher(kernel, grid_arg=3)      # launch(device, stream, gx, gy, gz, *kernel_args)
make_launcher(kernel, grid_cpp=fn)     # launch(device, stream, *extra_args, *kernel_args)
make_launcher(kernel, grid_py=fn)      # launch(device, stream, *kernel_args)
```

When all three options are omitted, the existing `launch(device, stream,
grid, *kernel_args)` behavior remains. This preserves callers that pass an
integer or a 1–3 element tuple/list as one grid object. An explicit
`grid_arg=None` has this same meaning. Passing two non-`None` options raises
`ValueError`; `grid_arg` accepts only exact integers 1, 2 or 3. With
`bind_device=True`, omit the device from each launch. Grid controls always
follow device/stream and precede public kernel arguments. All launchers remain
positional only.

`grid_cpp` takes a source-backed Python `def`, with each parameter annotated
as `int` or `bool`. Python does not support lambda parameter annotations, so
there is no lambda form for this option. The positional parameters must name
parameters of the original JIT function, and use their current launch values.
Keyword-only parameters must not name JIT parameters; they are extra scalar
arguments in declaration order. Every parameter is required; defaults,
`*args`, `**kwargs`, and free variables are refused. The return annotation is
optional. The function is never executed by the launcher.

```python
def grid(n: int, BLOCK: int, *, cap: int) -> tuple[int, ...]:
    return (min(cap, triton.cdiv(n, BLOCK)),)

launch = make_launcher(kernel, grid_cpp=grid)
launch(device, stream, cap, output, n, BLOCK)
```

The first compiler supports literal tuple/list results of 1–3 integers;
integer constants; parameter names; `+`, `-`, `*`, `//`; `triton.cdiv` and
two-argument builtin `min`; straight-line local assignments; and conditional
expressions. It recognizes the exact supported helpers, and refuses all other
globals, calls and AST nodes with `UnsupportedKernel`, including `nonlocal`
declarations. Arithmetic uses checked signed 64-bit integers and Python-style
floor division; invalid input type, zero divisor and overflow
raise before launch. Final dimensions use intj's existing `[0, 2**32)` bound.

`grid_py` takes a callable of one argument. On every launch, native code builds
a fresh dict containing every original JIT parameter name and its current
value. Public values come from that launch, baked values from the module, and
bound values from that launcher handle. It calls `fn(meta)` and parses the
returned grid with the existing 1–3 dimension validator. Each handle owns its
own callback and bound values, so sharing a compiled module cannot exchange
two launchers' callbacks. No Python callback runs in `grid_arg` or `grid_cpp`.
The existing requirement to pass every public kernel parameter remains.

## Native layout and identity

Extend the existing rendered entry point. In `grid_arg` mode, parse N exact
integer controls. In `grid_cpp` mode, evaluate a generated C/C++ function over
typed scalar launch inputs before the zero-grid decision. In `grid_py` mode,
use the existing bound vectorcall object to own the callback and hidden
values, even for an otherwise unbound launcher; its C entry point constructs
the dict and calls Python. Kernel argument packing and the GPU specialization
key see only original kernel arguments. Grid controls never enter that key.

The render context contains only immutable grid mode, dimensions, extra names,
and generated native code. Those fields enter `ModuleKey.digest()`. A
`grid_py` callback does not enter the artifact digest: its identity is owned
by the individual handle. The existing module loader, multi-phase module
state and full `EXT_SUFFIX` remain the artifact boundary.

## Checks

- GPU results distinguish 1D, 2D and 3D explicit dimensions and both callback
  modes; zero dimensions skip the launch.
- `grid_cpp` reuses dynamic kernel values and receives extras in declared
  order. Host checks cover source/annotation/name/AST refusals, signed floor
  division, divide-by-zero, overflow and final dimension range.
- Two `grid_py` handles with different callbacks share a module safely; each
  callback receives current public, baked and bound values. Python exceptions
  propagate. Cache hits still call the current callback.
- Existing grid spellings, bound/fixed-device launchers and the specialization
  key invariant stay covered by the repository suite. AMD runs on the available
  GPU; CUDA receives compile checks only, as before.

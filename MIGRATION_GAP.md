# Migration gaps

This file lists what still keeps Triton and aiter launches off intj. It comes
from migrating two trees onto intj `develop`, both on branch
`codex/intj-all-launches`:

- `lijinpei-amd/triton`
- `lijinpei-amd/aiter`

Triton counts are sites as of 2026-09-29; aiter counts are as of aiter
`00df7c810` (2026-10-13), whose list has 11 Triton launches left:
`num_ctas > 1` (3), Iris (3), `torch.compile` (3) and PDL (2). Each tree's
exception list (`intj_launch_exceptions.json` in triton,
`docs/intj_launch_exceptions.md` in aiter) records every site with its
reason. `TODO.md` tracks intj work in general; this file tracks what blocks
migration.

## Gaps that intj work can close

| Gap | What happens today | Sites blocked |
|---|---|---|
| ~~**Class-valued globals**~~ | **Closed** in `3e06c17`. With `assume_constant_globals=True` a class global (`NamedTuple`, Gluon layout classes such as `AMDWMMALayout`) is keyed by `module.qualname`, and each launcher emits a filterable `intj.ClassGlobalWarning` on every `check()` and at its build because replacing the class is not detected. Other values intj cannot canonicalize are still refused, with a message naming the global and its type (see `docs/Usage.md`). | none |
| ~~**Dynamic option with the same name as a constexpr parameter**~~ | **Closed** in `8df1e41` (plain launchers, `compat.launch`) and `0be1991` (autotune/heuristics). A `tl.constexpr` parameter named like a compile option (`num_warps`, `num_stages`, `waves_per_eu`, ...) now also sets that option on every call, passed, baked or tuned; `dynamic_options` naming it is a no-op, and `options=` naming it is refused. Unlike Triton, a positional value sets the option too (see `docs/Usage.md`). | none |
| ~~**No way back from a launcher to its kernel**~~ | **Closed** in `7cebe32`, and for `delayed=` launchers by the eager/delayed rework: `launcher.__self__.options.jit_func` is the `JITFunction` (or tuner chain) it launches, so aiter's AOT compile tools and the tests that read `arg_names`, heuristics or source, or launch a Triton reference, read it from a decorated kernel's launcher. Of aiter's ~13 named launchers, 3 remain because the JIT function must keep its own name: `triton.tools.compile` loads `aiter/mla.py`'s kernel by attribute (2 launchers over it), and `rmsnorm`'s kernel is also called from another kernel. | none |
| **`tl.constexpr` grid dimension** | A `tl.constexpr` value used as a grid dimension is rejected. | triton: a few |
| **`compat.launch` has no `assume_constant_globals`** | Neither `compat.launch` nor `launch_or_interpret` can pass the option, so kernels behind the bridge that read globals stay refused. | triton: bridge sites that read globals |
| **Tuple arguments and tuple constexprs** | `ARG_TUPLE` is not decoded or keyed (see `TODO.md`). aiter's Gluon kernels were rewritten to take scalars instead, and attn_res's sequence layout takes its residuals as 16 pointer parameters (`res0`..`res15`), one launcher per `L2` bucket; more than 16 are stacked into one tensor. | triton: ~41 entries |
| **Gluon layout objects as constexprs** | intj cannot decode layout objects passed as arguments (see `TODO.md`). aiter's gfx1250 kernels were rewritten to build their layouts in-kernel. | triton: ~75 (+6 with tuples) |
| **`TensorWrapper` arguments** | `make_launcher` refuses `TensorWrapper`. | triton: 14 bridge sites |
| **`torch.compile`** | Kernels wrapped with `torch.library.wrap_triton`, or launched inside compiled graphs, must go through Triton (see `TODO.md`). aiter's fav3_sage, compiled with `fullgraph=True`, launches its launchers' `jit_func` through Triton while Dynamo traces. | aiter: 3; triton: some |

## Gaps that intj refuses by design today

| Gap | Sites |
|---|---|
| Parameter defaults: a launcher never applies a `def k(..., X=4)` default; callers pass it explicitly (see `docs/Usage.md`) | triton: 0 alone (6 also hit other gaps) |
| Kernels that need profile or global scratch memory, or a scratch allocator | triton: ~129 |
| `num_ctas > 1` (CGA / multicast) | triton: 82; aiter: 3 |
| Host tensor descriptors and descriptor arguments | triton: ~71 |
| Proton launch hooks and instrumentation | triton: ~79 |
| gsan runtime | triton: 87 |
| PDL (`launch_pdl`) | aiter: 2 |
| Iris multi-GPU helpers | aiter: 3 |

## Not intj gaps

These sites stay on Triton because the site itself tests or needs Triton behaviour:

- tests of Triton's cache, runtime, IR dumps, hooks and diagnostics (triton: ~80);
- compile-warmup interception (triton: 20 bridge sites + 10);
- tests that expect the bridge's own errors;
- precompiled or AOT kernels that are not `JITFunction`s (none left in aiter: its
  metadata-redirect test compiles through a `return_compiled=True` launcher,
  and pa_mqa_logits' Triton < 3.4 loader was deleted);
- code paths only reachable on Triton < 3.7.

## Untested

Several converted sites run on hardware or builds not available here, so they are
untested:

- **gfx1250:** compile-identity checked only.
- **Proton sites.**
- **Free-threaded builds with Triton installed.**

# Migration gaps

This file lists what still keeps Triton and aiter launches off intj. It comes
from migrating two trees onto intj `develop`, both on branch
`codex/intj-all-launches`:

- `lijinpei-amd/triton`
- `lijinpei-amd/aiter`

Counts are sites in those trees as of 2026-09-29. Each tree's exception list
(`intj_launch_exceptions.json` in triton, `docs/intj_launch_exceptions.md` in
aiter) records every site with its reason. `TODO.md` tracks intj work in
general; this file tracks what blocks migration.

## Gaps that intj work can close

| Gap | What happens today | Sites blocked |
|---|---|---|
| **Class-valued globals** | A kernel that reads a global holding a class is refused, with the misleading hint "pass it as an argument". `assume_constant_globals` accepts only values it can canonicalize. Layout classes such as `AMDWMMALayout` are the typical case. | triton: 27 |
| ~~**Dynamic option with the same name as a constexpr parameter**~~ | **Closed** in `8df1e41` (plain launchers, `compat.launch`) and `0be1991` (autotune/heuristics). A `tl.constexpr` parameter named like a compile option (`num_warps`, `num_stages`, `waves_per_eu`, ...) now also sets that option on every call, passed, baked or tuned; `dynamic_options` naming it is a no-op, and `options=` naming it is refused. Unlike Triton, a positional value sets the option too (see `docs/Usage.md`). | none |
| **No way back from a launcher to its kernel** | A launcher is a `PyCFunction` and exposes no `JITFunction`. Callers that need it keep a separate named launcher. Uses include AOT compile by name, reference launches through Triton in tests, heuristics checks, calling the kernel as a device function, and reading its `debug` attribute (FlashKDA). A candidate fix is an `intj.jit_of(launcher)` read from the lazy-state object. | aiter: ~13 named launchers + FlashKDA |
| **Parameter defaults** | Kernel parameter defaults are not applied, so every argument must be passed positionally. | triton: a few |
| **`tl.constexpr` grid dimension** | A `tl.constexpr` value used as a grid dimension is rejected. | triton: a few |
| **`compat.launch` has no `assume_constant_globals`** | Neither `compat.launch` nor `launch_or_interpret` can pass the option, so kernels behind the bridge that read globals stay refused. | triton: bridge sites that read globals |
| **Tuple arguments and tuple constexprs** | `ARG_TUPLE` is not decoded or keyed (see `TODO.md`). aiter's Gluon kernels were rewritten to take scalars instead. | triton: ~41 entries; aiter: `attn_res` (`res`) |
| **Gluon layout objects as constexprs** | intj cannot decode layout objects passed as arguments (see `TODO.md`). aiter's gfx1250 kernels were rewritten to build their layouts in-kernel. | triton: ~75 (+6 with tuples) |
| **`TensorWrapper` arguments** | `make_launcher` refuses `TensorWrapper`. | triton: 14 bridge sites |
| **`torch.compile`** | Kernels wrapped with `torch.library.wrap_triton`, or launched inside compiled graphs, must go through Triton (see `TODO.md`). | aiter: 2; triton: some |

## Gaps that intj refuses by design today

| Gap | Sites |
|---|---|
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
- precompiled or AOT kernels that are not `JITFunction`s;
- code paths only reachable on Triton < 3.7.

## Untested

Several converted sites run on hardware or builds not available here, so they are
untested:

- **gfx1250:** compile-identity checked only.
- **CUDA:** compile-checked only.
- **Proton sites.**
- **Free-threaded builds with Triton installed.**

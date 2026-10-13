# TODO

Ranked. Each line is a known gap in the committed code, not a wishlist.

## Correctness

- **Support non-x86-64 hosts.** `torch_abi.toml` contains x86-64 tensor offsets,
  but `layout_for()` selects them by torch version and `PyObject` size on every
  host. Refuse RUNTIME_SHIM on other host ABIs until their offsets are measured and
  independently checked; add ARM64 layouts and run launcher tests there.
- **Fork safety.** The C kernel cache keeps parent handles across `fork(2)`; a child
  that launches uses a dead context. Triton guards this by pid. Fix with a
  `pthread_atfork` child handler that repoints the cache at an empty sentinel.
- **`knobs.runtime.debug` / `knobs.compilation.instrumentation_mode`** are in triton's
  cache key; intj reads them at a launcher's first call. Flipping one later keeps
  launching the old binary unless it is declared in `dynamic_options`. Same for
  `use_buffer_ops`, which is only stale-but-valid since the `S` bit is
  unconditionally keyed.
- **Free-threaded lazy-launcher publication on non-TSO hosts.** CPython reads
  `ml_meth` with a plain load; the build publishes `state` before a release store
  of `ml_meth`, which x86-64 orders. Recheck (or have the entry fall back to an
  acquire reload of `state`) when ARM64 is supported.
- **Multi-GPU on NVIDIA.** The sm_89 machine has one card, so
  `test_tuned.py`'s two-GPU case is still skipped there; it passes on AMD.
- **`noexcept` at the CPython boundary in C++ builds.** STATIC_COMPILE tensor
  access and non-intj cache backends compile the extension as C++. An
  exception escaping `entry` into CPython's C frames is UB; `INTJ_NOEXCEPT` on the
  functions CPython calls turns that into a deterministic `std::terminate`. This is
  *not* a performance item:
  measured identical codegen with and without, because intj has no non-trivial
  destructors and so no landing pads to elide.
- **The triton half below 3.10.** `tests/run_python_matrix.sh` runs the rendered
  module on 3.8 through 3.14t, but `triton>=3.8` ships no wheel below 3.10, so
  `make_launcher` and the compile callback are only exercised from 3.10 up.
- **Develop a Torch and CPython interface spec.** Define the binary data each
  interface reads, supported versions and build variants, how those facts are
  verified, and when intj refuses an unsupported combination.

## Coverage (all currently refused loudly)

- Parameter annotations (`extra_annotation`), including non-constexpr annotated
  parameters, which change arity (an annotated `== 1` int stays a kernel param).
- **A baked int cannot take Triton's specialization of its value.** On a
  parameter with no inline type annotation, `Argument(value=0)` raises "has
  AUTO specialization with a fixed value" under `AUTO` and under
  `Assume(Aligned(16))` (equal-to-1 stays `AUTO`); only `NEVER` is accepted,
  which drops the divisibility Triton gives 0 or any multiple of 16 (and the
  constexpr it makes of 1), so the binary differs from Triton's. Resolve
  `AUTO` from the baked value.
- **`Assume` cannot turn a fact off.** Facts left out of `Assume` stay `AUTO`
  and are measured per call; only Triton's `do_not_specialize_on_alignment`
  and an inline type annotation turn single facts off, and "assume aligned,
  no pointer range" cannot be written. So a hand-written `ASTSource` attrs set
  is not always reproducible: aiter's pa_mqa_logits launchers compile with
  `tt.divisibility` and `tt.pointer_range` facts its attrs lacked, whenever
  they hold. Needs a per-fact `NEVER`.
- **Tuple / namedtuple arguments and tuple constexprs.** `ARG_TUPLE` recursion in the decoder and
  the key; ~41 Triton-test entries pass tuples. aiter's last one, `attn_res`'s
  `res`, became 16 pointer parameters, one launcher per `L2` bucket.
- **`torch.compile` integration.** Kernels wrapped with `torch.library.wrap_triton` or launched
  inside `torch.compile` graphs still go through Triton (aiter rotary, Triton's constexpr-mutation
  test, and aiter's fav3_sage, which launches its launchers' `jit_func` while Dynamo traces);
  intj launchers are opaque to dynamo -- register them as custom ops or teach
  `wrap_triton` to dispatch to an intj launcher.
- Kernels reading globals: `assume_constant_globals=True` trusts them; still
  refused by default. Revalidating in C (`PyObject_RichCompareBool` per entry)
  would cover the default too.
- Kernels needing global/profile scratch, `num_ctas > 1`, cooperative launches,
  `launch_pdl`.
- **Gluon layout objects as constexpr arguments.** gfx1250 GEMM/MoE/batched-GEMM
  kernels pass SHARED/WMMA layouts via `**layouts`; intj can't decode them.
  Rewrite the kernels to build layouts in-kernel from constexpr ints, or add
  value-keyed layout constexprs. The Triton tree has ~75 such entries
  (`unsupported_layout_constexpr`) plus 6 with tuples.
- **`Autotuner.cache` is not mirrored.** intj copies back `best_config`,
  `bench_time` and `configs_timings` after tuning, but not `cache` itself, so
  code inspecting `len(kernel.cache)` sees 0 even after a tuned launch; either
  mirror entries back per key or warm-start intj's own cache from it.
- **A `delayed=` launcher has no `jit_func` before `check()`.** Its
  `__self__.options` is an `intj.launcher.Delayed` until `check()` runs the
  delayed function, which may query the GPU; code that only wants the kernel
  back must check first or, like aiter's `launch_on_current_stream`, take only
  launchers without `delayed=`. The kernel is known at `make_launcher`; expose
  it on `Delayed`.
- **Keyword arguments at launch.** Launchers are `METH_FASTCALL` (positional
  only); `METH_FASTCALL | METH_KEYWORDS` (vectorcall `kwnames`) would accept
  keywords with no cost for positional calls (`kwnames` `NULL`) -- needs a
  name-to-slot map. E.g. Triton's `test_prune_configs` passes `N=N` and
  expects it in the pruner's kwargs.
- **Cache-invalidating knobs are not keyed.** Triton's cache key includes
  `get_cache_invalidating_env_vars()` (e.g. AMD buffer-ops / pingpong knobs);
  intj's module and spec keys capture only `debug`, instrumentation mode and
  fpsan casts, read at a launcher's first call, so flipping another such knob
  after that keeps launching the binary compiled under the old value -- a
  coarser key than Triton's. Workaround today: declare the knob in
  `dynamic_options`, which keys it per call. Fix: freeze the rest into the
  module identity at the first call.

## Performance

- **32-argument launches lost ~7-10 ns to code placement.** `last_key 32` and
  `sweep 32 int` moved from 99-106 ns to 106-113 ns during the lazy-launcher
  work with identical `intj_call` instructions: cold code grew ahead of it.
  Aligning `intj_call`/`intj_bound_entry` to 64 bytes wins that back but costs
  common shapes ~1.5 ns (journal `2026-09-28_lazy-task5_0_8d95e66.md`); the
  upgrade path is a hot/cold section split (`__attribute__((hot/cold))` or
  `.text.hot`) so cold code cannot move the hit path.
- **Launch calls with more than 30 positional arguments.** CPython 3.10+
  compiles a call site with 31 or more positional arguments to `BUILD_LIST` /
  `LIST_APPEND` + `CALL_FUNCTION_EX` instead of `CALL` (3.8 and 3.9 do not),
  so the launcher loses `CALL_BUILTIN_FAST` and every call builds a list and a
  tuple. Device, stream, grid dimensions and dynamic options all count. On
  gfx942 a 27-parameter kernel went from 4123 ns at 30 positional arguments to
  4514-4566 ns at 31-33; aiter's mla stage-2 launch gained +389-411 ns when
  three dynamic options pushed it to 31. aiter's pa_decode_gluon kernels (54
  parameters) pay this on every launch. The threshold is a property of the
  caller's source, so intj cannot avoid it inside the entry; options are a
  packed-argument call form, guidance to bake or `bind_values` arguments to stay
  at or below 30, or both.
- **Two-entry memo per object slot.** A slot remembers one object; a call site
  alternating two `str`/dtype/JIT objects in one slot pays an interner call
  per launch. Add a second entry if `--dynamic` or a real site shows it.
- **ROCm zero-user-argument launches.** Triton still declares two implicit
  scratch-pointer arguments (16-byte kernarg segment) when neither scratch
  buffer is needed. On gfx942, `hipModuleLaunchKernel` took ~1.55 us with no
  ABI arguments versus ~3.03 us with one or two; TVM FFI launching the same
  Triton kernel took ~3.01 us versus INTJ's ~2.77 us. Investigate whether
  Triton can omit unused scratch arguments or HIP can launch argument-bearing
  kernels faster; packed `extra` arguments did not help in this case.
- Benchmark `hipModuleLaunchKernel` with `extra` parameter-buffer passing against
  the current `kernelParams` pointer array; use it if faster.
- Benchmark CUDA driver `cuLaunchKernel` with `extra` parameter-buffer passing
  against the current `kernelParams` pointer array; use it if faster.
- Benchmark sweep promised in the README: dynamic vs constexpr argument counts and
  tensor counts.
- **Measure compact-int decoding.** Compare the existing inline CPython helpers
  with a direct `lv_tag`/`ob_digit[0]` path on supported builds; use custom
  decoding only if it is faster and passes `check.py`.
- Single-entry inline cache in front of the hash map (~2-3 ns, ~100% hit in a
  steady-state loop).
- Annotated fast paths: pinning a parameter's type skips the generic classifier
  (~185 ns vs ~275 ns measured upstream on a 10-arg kernel).
- Tuned launches decode arguments on the generic path; the auto-decode macros do
  not handle exact keys yet.

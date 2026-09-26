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
  cache key but neither in intj's module digest nor its spec key: flipping one after
  `make_launcher` keeps launching the old binary. Same for `use_buffer_ops`, which
  is only stale-but-valid since the `S` bit is unconditionally keyed.
- **Run the NVIDIA path on an NVIDIA GPU.** It is compile-checked only.
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

- Callable grids (`dynamic_grid`). The expensive part is `ConstexprFunction.__call__`
  (~963 ns/call), not the mapping; a `grid=<spec>` baked to literal C is ~4 ns.
- Per-launch options (`dynamic_options`) — needs a second entry point, not a reserved
  always-`None` slot on the hot path.
- Parameter annotations (`extra_annotation`), including non-constexpr annotated
  parameters, which change arity (an annotated `== 1` int stays a kernel param).
- Tuple / namedtuple arguments: `ARG_TUPLE` recursion in the decoder and the key.
- Kernels reading globals: revalidate in C (`PyObject_RichCompareBool` per entry)
  instead of refusing.
- Kernels needing global/profile scratch, `num_ctas > 1`, cooperative launches,
  `launch_pdl`.
- Parameter defaults: the launcher requires every argument positionally.

## Performance

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

# TODO

Ranked. Each line is a known gap in the committed code, not a wishlist.

## Correctness

- **Fork safety.** The C kernel cache keeps parent handles across `fork(2)`; a child
  that launches uses a dead context. Triton guards this by pid. Fix with a
  `pthread_atfork` child handler that repoints the cache at an empty sentinel.
- **Free-threaded builds.** The kernel cache and the borrow-on-hit discipline assume
  the GIL. Refuse on `Py_GIL_DISABLED`, or lock it.
- **`knobs.runtime.debug` / `knobs.compilation.instrumentation_mode`** are in triton's
  cache key but neither in intj's module digest nor its spec key: flipping one after
  `make_launcher` keeps launching the old binary. Same for `use_buffer_ops`, which
  is only stale-but-valid since the `S` bit is unconditionally keyed.
- **Run the NVIDIA path on an NVIDIA GPU.** It is compile-checked only.
- **`noexcept` at the CPython boundary**, once the C++ access mode lands. An
  exception escaping `entry` into CPython's C frames is UB; `INTJ_NOEXCEPT` on the
  functions CPython calls turns that into a deterministic `std::terminate`. Not a
  gap today (the extension is C, nothing can throw), and *not* a performance item:
  measured identical codegen with and without, because intj has no non-trivial
  destructors and so no landing pads to elide.
- **Python versions other than 3.12.** The runtime is written against one
  interpreter and refuses the rest at compile time (`#error` below `0x030C0000`),
  which is honest but narrow. What is version-dependent today:
  - the non-compact int decode walks `ob_digit` with CPython's layout macros. The
    compact case already uses `PyUnstable_Long_*`; 3.13 adds `PyLong_AsNativeBytes`
    and `PyLong_AsInt64`, which would retire the digit walk entirely.
  - `PyFloat_CheckExact` + a direct `ob_fval` read, and the `PyLongObject` /
    `PyFloatObject` layouts behind both.
  - `PyErr_GetRaisedException` / `PyErr_SetRaisedException` (3.12+).
  - 3.13+ also needs `Py_mod_gil` in the module slots, and free-threaded builds
    need the kernel cache locked (see above).
  A version's layout bets want the same treatment torch's got: a table per
  supported version plus a self-check at load, not an `#if` thicket.

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
- `@triton.autotune` / `@triton.heuristics`: render per config, or pick in python.
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
- Single-entry inline cache in front of the hash map (~2-3 ns, ~100% hit in a
  steady-state loop).
- Annotated fast paths: pinning a parameter's type skips the generic classifier
  (~185 ns vs ~275 ns measured upstream on a 10-arg kernel).

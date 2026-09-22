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
  `create_launcher` keeps launching the old binary. Same for `use_buffer_ops`, which
  is only stale-but-valid since the `S` bit is unconditionally keyed.
- **Run the NVIDIA path on an NVIDIA GPU.** It is compile-checked only.

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

- Benchmark sweep promised in the README: dynamic vs constexpr argument counts and
  tensor counts.
- Single-entry inline cache in front of the hash map (~2-3 ns, ~100% hit in a
  steady-state loop).
- Annotated fast paths: pinning a parameter's type skips the generic classifier
  (~185 ns vs ~275 ns measured upstream on a 10-arg kernel).

# Autotune and Heuristics in `make_launcher`

## Goal

Make the tuned launch path fast. `make_launcher` accepts a JIT function wrapped in
any nesting of `@triton.autotune` and `@triton.heuristics`:

```python
@intj.make_launcher      # must be outermost
@triton.autotune(...)    # any number, any order
@triton.heuristics(...)
@triton.jit              # must be innermost
def kernel(...): ...
```

On a cache miss, Triton tunes, picks heuristic values and launches; intj records the
result. On a hit, intj launches itself and bypasses every wrapper layer.

## Role assignment

`make_launcher` visits the layers from the outermost inward. It assigns every value
an input level and says whether it is **exact**, meaning fixed by the keys of lookups
`0..n`. The result of the innermost layer is final.

- **Caller arguments** are known at level 0. A `tl.constexpr` is exact at level 0,
  because the spec key holds its value. A non-constexpr var is exact only when it
  becomes a key.
- **Autotune layer:**
  - Every `key` name that is not already exact becomes an **independent key** at the
    level where it is known. A caller var is keyed at level 0 by its exact value:
    int64, fp64 bits or bool. It is exact for keying only and still compiles as a
    runtime argument with Triton's normal specialization.
  - Tensor-valued keys are refused.
  - The dtype part of Triton's tuning key is already in the spec key.
  - Each tuned config kwarg is **dependent**. It is stored in the record of the
    highest level among its keys, and is known one level later.
  - A tuned name that an outer layer already assigned is an error.
- **Heuristics layer:** each dict key is an assigned value, and the lambda's inputs
  are the names it reads.
  - If every input is exact, the result is **dependent**. It is stored in the record
    of the highest level among its inputs.
  - Otherwise the result is a **computed independent key**. The lambda is lowered to
    C++ and evaluated before the lookup at level `max(input levels)`. Its inputs are
    never keyed.
  - A result name must not be one of its own inputs.

Example: with `autotune(key=["N"])` outside
`heuristics({"ALIGNED": lambda a: a["stride"] % a["BLOCK"] == 0})`, `N` is a level-0
key and `BLOCK` sits in the level-0 record. `ALIGNED` becomes the level-1 key,
computed from `stride` and the record's `BLOCK`. In a matmul where the heuristic reads
only autotune keys and tuned values (`K % (BLOCK_K * SPLIT_K)`), everything is exact
and one lookup suffices.

### Lambda lowering

Reuse `grid.py`'s restricted AST, and add:

- `%`, comparisons, `and`/`or`/`not`, `max`, `triton.next_power_of_2`
- on tensors, with literal indices only: `.numel()`, `.shape[i]`, `.size(i)`,
  `.stride(i)`, `.dim()`, `.element_size()`, `.is_contiguous()`, `.dtype`
- `x is None`

Results must be `int` or `bool`. Free variables, globals other than the builtins
above, passing `meta` onward, and any other syntax raise `UnsupportedKernel` at
`make_launcher`. A lambda may read a computed key from an outer heuristic.

## Launcher signature

Dependent names are removed from the positional call, as baked parameters are. Triton
also never takes them from the caller: autotune raises on a conflict, and heuristics
overwrite the value. Config compile options (`num_warps`, `num_stages`, ...) are
stored per record. `make_launcher` refuses with `UnsupportedKernel`:

- an `options` entry for a config-owned option
- an `extra_annotation` entry or `.bind()` value that names a dependent value
- `no_gpu=True`

`intj.compat.launch` keeps refusing decorated kernels.

## Cache layout

The number of levels N is known at render time, so the entry template unrolls the
lookup chain.

- **Level 0** keeps today's map, from the swappable `kernel_cache` backend, keyed by
  the spec words plus the exact values of its independent keys.
- **A non-final slot** holds a pointer to a heap record. The record contains:
  - the dependent values that C++ reads later (next-level lambdas, `grid_cpp`)
  - an embedded child map
- **Child maps** are always intj's own map, made width-generic. A child key holds
  only that level's computed keys (usually 1 word) with no parent pointer, and the
  map starts at 4 slots.
- **The final level** stores its record by value in every backend (intj, tsl, absl).
  The record, `intj_final_record`, is `intj_kernel` plus the dependent values that
  `grid_cpp` reads. With `grid_py`, it also holds a dict of every dependent value.
  It is a plain struct, so a rehash moves it by copying its fields.
- **Memo:** every map keeps its own `last` memo, and an insert resets the memo of the
  map it writes to.
- **Freeing:** `intj_cache_free` and GC traversal recurse into child maps.
- **Locking:**
  - Free-threaded builds replace `PyMutex` with `pthread_rwlock_t`.
  - A hit takes the read lock across the whole chain and the launch.
  - A miss releases the lock before calling Python, then takes the write lock to
    recheck and insert.
  - On GIL builds the lock stays a no-op.
- **Benchmark:** `tests/bench_kernel_cache.cpp` moves every backend to by-value final
  records and adds a 1-word child map row.

## Miss path

At `make_launcher`, the wrapper chain is `copy.copy`'d layer by layer into a private
chain, so the launcher owns its tuner caches. The copy's innermost `.fn` becomes a
shim.

- **The shim** is an internal bare-JIT intj launcher with every constexpr public. It
  converts Triton's kwargs to positional arguments and evaluates Triton's grid with
  `meta`.
- **Benchmarking:** benchmark calls and the final call both run through the shim, so
  the benchmarked binary is the launched binary. Different exact keys that share a
  Triton specialization reuse the shim's compiled kernel.
- **Any miss, at any level:** run the private chain's outermost `.run`. Known tuning
  keys hit the private tuner cache, so there is no re-benchmark.
- **Filling records:** the last call the shim receives is the final launch. Its kwargs
  and `CompiledKernel` fill every missing level. C rechecks each level under the
  write lock, as today.
- **Visible state:** `best_config`, `bench_time` and `configs_timings` are copied back
  to the user's tuners.

Pruning, `reset_to_zero`, `restore_value`, `do_bench`, `warmup`/`rep` and
`cache_results` run inside Triton on misses. `Config.pre_hook` is allowed but runs on
misses only. That differs from Triton, which runs it on every launch.

## Grid and return value

- `grid_arg` and the default grid are unchanged. The default grid is caller-supplied,
  so it cannot depend on the config.
- `grid_cpp` may name dependent values, read from the final record after lookup.
- `grid_py`'s `meta` includes the dependent values.
- `return_compiled` returns the final record's `CompiledKernel`.

## Invariant and tests

> same intj key chain ⟹ same Triton specialization, same tuning key, same heuristic outputs

- Extend `test_spec_key_is_never_coarser_than_triton` to cover this invariant.
- Mutation check: remove an autotune var's exact-value word and confirm the test fails.
- Also cover:
  - nesting orders
  - a two-level chain
  - a float key
  - hit versus miss call counts, with no Python on a hit
  - `grid_cpp` reading tuned values
  - copy-back of tuner state
  - each refusal

CUDA is compile-checked only (untested).

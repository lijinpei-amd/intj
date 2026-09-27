# Lazy, Per-Call-Variant `make_launcher`

## Goal

Every launch site now on `intj.compat.launch`, aiter's `launch_tuned`, or
aiter's lazy `intj_handle` can use one `make_launcher` handle, written as a
decorator on the kernel.

Two things block that today:

- **Building needs the GPU.** `make_launcher` queries the GPU target when it
  is called, so decorating at module level forces GPU work at import.
- **Per-call variants.** Compile options, Triton knobs, and `str` / `tl.dtype`
  / JIT-function constexprs are fixed when a launcher is built. A call site
  that varies any of them needs a separate launcher per variant.

This spec makes the build lazy and lets one handle key those variants. The hit
path stays at today's cost. The key invariant is unchanged: same intj key
implies same Triton specialization.

## API and call contract

```python
@intj.make_launcher(grid_cpp=grid, dynamic_options=("num_warps", "knobs.compilation.instrumentation_mode"))
@triton.autotune(...)          # optional, heuristics too
@triton.jit
def k(x, out, n, ACT: tl.constexpr, BLOCK: tl.constexpr): ...

k(device, stream, 8, "", x, out, n, "gelu")   # device (omitted with bind_device), stream, grid controls
                                              # (none for grid_cpp), dynamic values, public args
```

### Lazy build

`make_launcher(...)`, whether used as a decorator or called directly, returns
a launcher right away without touching the GPU. At that point it does only
the GPU-free work:

- validating arguments
- resolving annotations
- analyzing the tuning chain
- lowering the grid
- computing the bound-object layout

The first call builds the module:

1. query the target
2. render
3. compile
4. load
5. bind

After that, the launcher's entry points at the built module. If the build
fails, that call raises and the next call retries.

Why the build needs a GPU at all: `driver.active.get_current_target()` asks
the driver for the visible device's architecture and warp size. The target
decides four things:

- **The backend:** driver library, launch symbol, error style, and the
  pointer-range key bit.
- **Canonical compile options,** through `parse_options` defaults.
- **The `ModuleKey` target field.**
- **Early validation of tuned configs.**

Compiling the C module needs only the host compiler. Loading the `.so` needs
the driver library to be installed, but no visible device.

### `dynamic_options`

`dynamic_options=(name, ...)` takes compile-option names (`"num_warps"`) and
Triton knob paths (`"knobs.<group>.<name>"`, for example
`"knobs.compilation.instrumentation_mode"`, `"knobs.runtime.debug"`,
`"knobs.amd.use_buffer_ops"`).

- **Passing:** each declared entry becomes a positional argument right after
  the grid controls, in the declared order. Its value is part of the key.
- **Knobs only key:** intj never sets a knob. The caller sets it through
  Triton and passes its current value; a declared path is left out of the
  `ModuleKey`, so the live value at the first call builds no extra module.
- **Refused with `UnsupportedKernel`:**
  - an unknown option or knob path;
  - a name that is also given in `options=` (it can't be both fixed and per
    call);
  - a name a tuning layer assigns.

### Object constexprs

`str`, `tl.dtype`, and JIT-function constexprs become ordinary public
arguments, keyed by value. Baking one through `extra_annotation` still works
and removes it from the call.

### Unchanged

- **Grid:** the default grid already takes a tuple of 1–3 elements per call.
  `grid_arg`, `grid_cpp` and `grid_py` behave as today.
- **Also as today:** `bind_device`, `return_compiled`, tuning, and every
  refusal not listed here.
- **Fallback:** `intj.compat.launch` remains for what is still refused: tuple
  arguments, keyword-only calls, and sites that cannot be verified.

## Lazy build in C

### The stub module

`_intj_lazy` is a small generic C extension. It is:

- compiled once per interpreter and intj version, with the host compiler only;
- cached on disk like rendered modules;
- loaded by hand, not through `import`;
- free of kernel code, so building or loading it needs no GPU.

### Uniform bound-launcher layout

`intj_bound_launcher` becomes a fixed header shared by all launchers, followed
by trailing arrays whose length is `NBOUND`: bound owners, pointer bits, and
static-compile tensor slots. `NBOUND` is known without a GPU.

- **Cache storage:** the header holds fixed-size, opaque storage for the level-0
  cache, large enough for every backend (intj map, tsl, absl).
- **Field order:** fields the warmed launch reads sit within 64 bytes after
  `PyObject_HEAD` (not one aligned cache line: GC objects are not 64-aligned):
  - device ordinal and handle
  - the cache storage, including its memo
  - the fixed-kernel pointer
  - (the per-slot dynamic-value memos, sized by the number of dynamic slots,
    live in the trailing arrays right after the header)

  Cold fields come after them: the build lock, the builder, GC hooks, the
  launcher's miss callback (`compile_cb`, the tuned callback when tuned), its
  intern table, its read/write lock, `grid_py`, `grid_hidden`, and the module
  pointer.
  `offsetof` static asserts pin this order.
- **Layout check:** the rendered module `static_assert`s the same header. At
  load it checks the header size the stub reports and raises `ImportError` on
  a mismatch.

### Decoration

`make_launcher` allocates the bound object from the stub type at its final
size. It stores a Python builder closure in the header. It returns a real
`PyCFunction` whose `self` is that bound object, created from a heap-allocated
`PyMethodDef` that the bound object owns:

- flags `METH_FASTCALL`;
- `ml_meth` set to the stub's build shim.

### First call

The shim:

1. takes the header's build lock;
2. calls the builder, which renders and loads the module (or reuses one
   already loaded for the same `ModuleKey`) and calls the rendered module's
   `init_bound(self, ...)`. That call fills the header in place (module
   reference, device handle, cache init, miss callback, intern table,
   `grid_py`, GC hooks) and
   returns the rendered entry function pointer;
3. stores the entry with one release store to `def->ml_meth`;
4. drops the builder;
5. forwards the current call.

Threads that race the first call wait on the lock, then forward. A builder
exception propagates and leaves the launcher unbuilt, so the next call
retries.

### After the build

CPython reads `m_ml->ml_meth` on every call. So every caller reaches the
rendered `intj_bound_entry(self, args, nargs)` directly, including references
taken before the build:

- It takes `self` as its bound launcher, with no indirection.
- `CALL_BUILTIN_FAST` specialization still applies.
- `self` never changes, and the swap is a single pointer store. On
  free-threaded builds a reader cannot pair a new function with an old `self`.

### GC

The stub type's traverse and clear visit the builder and the module reference,
then call the rendered module's hooks once they are installed. The hooks reach
cache records, compiled objects, the miss callback, the intern table, slot
memos, and grid objects. The stub creates the launcher's rwlock with the
header and destroys it in dealloc, never in clear, so a retried build never
re-initializes a live lock.

### Uniform shape

Launchers without bindings return the module's `entry` today, with
`self = module`. They become bound-style lazy launchers too. `.bind()` and
`.bind_device()` return lazy launchers.

All mutable state lives in the launcher (the bound object):

- its level-0 kernel cache and that cache's memo;
- its intern table and per-slot memos;
- its own read/write lock, which guards the cache, memos, and intern writes;
- its key self-check (the map behind "one spec key maps to two compiler
  inputs"), which therefore runs on every call, object values included;
- its miss callback: the tuned callback, or a per-launcher wrapper around the
  module's compile function.

Module state is read-only after load: torch types, the ABI layout and dtype
table, and driver symbols, written once under the load lock before the module
is reachable. The module also carries the Python compile function. That
function keeps a per-module compile cache behind its own Python lock, keyed on
the compiler input, canonical options, declared knob values, and device. A
sibling launcher of the same `ModuleKey` misses once and builds only its C
record, reusing the `CompiledKernel`.

The module's debug `entry`, its shared cache, its compile-callback slot, and
its rwlock go away: every `make_launcher` result is a lazy bound launcher, and
no module lock remains. `module.spec_key(launcher, device, *args)` takes the
launcher, decodes with its intern table, and is the level-0 half of
`key_chain`.

Why: this removes every structure shared across launchers, so every launcher
behaves like a tuned or `bind_device` one, and an intern id never has to mean
the same value in two launchers. The cost is memory per launcher: one map
(16 slots to start), one intern table, and one lock.

Key layout and tuned paths are otherwise unchanged.

## Keying dynamic values

### Scalar values

Dynamic options and knobs, and constexprs, that are `int`, `bool`, or `float`
are keyed like exact autotune keys: a kind byte plus 8 bytes of value (int64,
uint64, or fp64 bits). A `str` option or knob value, such as
`instrumentation_mode`, goes through the intern table below.

### Object values

`str`, `tl.dtype`, and JIT-function values reduce to a canonical tuple, the
same one `_canonical_value` produces for baked values:

- `("str", s)`
- `("dtype", dtype.name)`
- `("jit", module, qualname, cache_key)`

A per-launcher intern table maps each canonical tuple to a small `uint32` id,
and the key carries the id.

- **Per-slot memo:** each argument slot keeps a strong reference to the last
  object it saw, plus that object's id. Holding the reference means the
  object's address cannot be reused while it is memoized. When the same
  object arrives again, one pointer compare returns the id, with no Python
  code and no lock (a free-threaded build takes the read side of the
  launcher's own rwlock). This covers literals, interned strings, dtype singletons,
  and repeated JIT objects.
- **New object:**
  - Python computes the canonical tuple outside any lock. For a JIT function
    this reads `cache_key`.
  - It then probes a plain dict of canonical tuple → id:
    - `str` content hashes use CPython's cached hash, so this is cheap.
    - Equality is by value.
  - A new tuple takes the next id under the launcher's write lock.
  - The memo moves to the new object, and the old reference is released
    outside the lock.
- **Equality:** equal content gets the same id whichever object carries it,
  which matches Triton's constexpr semantics.
- **Scope:** ids are per launcher and per process, because each launcher also
  owns its kernel cache (see Uniform shape). They are never persisted
  and never enter the module key.
- **Limitation:** a JIT function's `cache_key` is read once and cached. Edits
  to a callee's source or globals after first use are not seen. Triton has the
  same limitation.

### Module key

The declared dynamic names and their order enter `RenderContext` and
`ModuleKey`, because they shape the rendered call signature. Per-call values
never do.

Baked values keep today's hashing: their canonical tuple is JSON-serialized
into `ModuleKey` and SHA-256-hashed. A JIT function's `cache_key` is a content
hash, so the on-disk module cache stays valid across processes.

### Miss path

The compile callback receives the dynamic values.

- **Options:** it merges them into `options=` and canonicalizes per record
  with `parse_options`. Today this happens once per module. An invalid value,
  such as `num_warps=3`, raises on that miss.
- **Knobs:** before building the compiler input, it compares each declared
  knob's passed value with the live knob. A mismatch raises `ValueError`
  (`intj: dynamic knob '<path>' passed <v> but the current value is <w>`)
  and records nothing. Otherwise the compile runs under the live knobs, which
  equal the key's values. The knob values stay in both compile-cache keys
  (untuned identity, tuned `compiled` key): a knob that feeds no option, such
  as `knobs.compilation.disable_line_info`, must not reuse another value's
  kernel. A hit only compares keys, so an old value keeps hitting the record
  compiled under it after the knob changes.
  A declared knob must not change while a miss for its launcher is running:
  the check and the compile read it at different times. The check runs per
  launcher, before the module's compile cache, so a sibling launcher's first
  call with an old value raises even when that cache holds the kernel.
- **Object constexprs:** these reach the compiler as the Python objects from
  the call.
- **Tuned launchers:** dynamic values sit in the level-0 key, and the shim
  merges option values with each config's options. Overlap is refused up
  front.

### Undeclared knobs

Knobs that are not declared are read once, at the first call, and fixed for
that launcher. This moves today's `debug` / instrumentation read from
`make_launcher` to the first call. Keying the cache-invalidating environment
knobs stays a TODO.

## Testing

### intj

- **Lazy build:**
  - With `_current_target` patched to raise, decoration still succeeds.
  - The first call builds.
  - A failing build raises, and the next call retries.
- **Swap:**
  - A reference taken before the build (`f = k`) reaches the real entry
    afterwards.
  - After warm-up, `dis` with adaptive specialization shows
    `CALL_BUILTIN_FAST` at the call site.
  - `offsetof` asserts pin the hot fields within 64 bytes after the object head.
- **Per-launcher state:** two launchers of one `ModuleKey` compile a key
  once; the second's first call only builds its C record. The key self-check
  raises for a key that maps to two compiler inputs, object values included.
- **Concurrency:** N threads make the first call together on 3.13t and 3.14t
  (both installed via uv). There is exactly one build, and every call returns
  correctly.
- **Dynamic options and knobs:**
  - Varying `num_warps` gives separate records with the right
    `metadata.num_warps`.
  - A declared knob's value keys the record; a value that differs from the
    live knob raises on a miss and records nothing; intj never modifies a knob.
  - Each refusal raises `UnsupportedKernel`.
- **Object constexprs:**
  - Two equal strings at different addresses share one record, and different
    strings get separate records.
  - `tl.dtype` works.
  - A JIT function works, and the same `cache_key` shares a record.
  - A freed-and-reused address cannot alias another value.
  - Two launchers of one module that intern different strings first each
    launch the right binary, and each variant compiles once.
- **Invariant:**
  - The untuned and tuned invariant tests vary each dynamic option, knob, and
    object constexpr.
  - Each has a mutation check: dropping its key field makes the test fail.

### Benchmarks

The full gate from `benchmarks/AGENTS.md` runs against the baseline and the
latest journal. It adds rows for:

- a warmed lazy launcher;
- a launcher with 2 dynamic options;
- a launcher with a `str` constexpr.

Acceptance: an untuned launcher without dynamic values is within 1 ns of
today on `bench_launch.py --no-gpu`, measured interleaved.

## Migration (follow-up plans, after intj lands)

- **Triton tree:**
  - Move the 24 per-call-variant sites and the 39 fpsan sites to
    `@make_launcher(dynamic_options=...)`.
  - Move existing handles and the `partial` stacks to the decorator.
  - Leave the 107 hardware-gated sites until they can be verified.
- **aiter:**
  - Replace the 228 `intj_handle` handles with `@make_launcher` decorators.
    This is safe at import, because the build is lazy.
  - Move the 47 `launch_tuned` sites and the 134 `compat.launch` sites to the
    decorator, using `dynamic_options` or constexpr arguments.
  - Delete `intj_tuned.py` and `intj_handle.py` once nothing uses them.
- **Both trees:** `compat.launch` remains only for tuple arguments,
  keyword-only calls, and unverified sites.

## Docs

`docs/Usage.md` covers:

- lazy build and what the first call does;
- `dynamic_options`, including the knob paths;
- object constexprs.

Remove the per-call-variant entry from `TODO.md`.

## Out of scope

These are already on `TODO.md`:

- keyword arguments at launch
- mirroring `Autotuner.cache`
- keying cache-invalidating knobs
- tuple arguments

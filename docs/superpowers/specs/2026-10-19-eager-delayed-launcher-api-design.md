# Eager and Delayed Options, In-Place and Copying Launcher Edits

Replaces the `_Origin`/`Delayed`/`_Plan` machinery of
`2026-10-13-delayed-launcher-options-design.md`.

## Goal

- A launcher holds one `MakeLauncherOption`. Its `delayed` field is the
  **delayed** function (`() -> Mapping[str, Any]`, or None); every other
  field is an **eager** option. Nothing is shared between launchers, and
  nothing derived from the options is stored on the launcher.
- Every edit exists twice: `_foo` changes this launcher in place and returns
  None; `foo` returns a new launcher with the edit applied, leaving this one
  untouched.
- The first call is one straight pipeline: resolve `delayed`, check the
  options, build the render key, render, compile, load, call.
- `check()` is a dry run of the first two steps; it stores nothing.

## State

`MakeLauncherOption` gains `delayed: Callable[[], Mapping[str, Any]] | None =
None`. It is every `make_launcher` keyword now, so the "every keyword except
`delayed`" wording in `options.py` and `intj/__init__.py` changes.
`delayed` stays out of `OPTION_FIELDS` (a delayed function cannot return
`delayed`) and out of `MakeLauncherKwargs` (`make_launcher` keeps it as an
explicit keyword).

A launcher's C object holds that one value in its `options` slot; the
`origin` and `plan` slots are deleted. `k.__self__.options` is always a
`MakeLauncherOption`, so `options.jit_func` is readable before the first call
(closes `TODO.md`'s "A `delayed=` launcher has no `jit_func` before
`check()`"), and `options.delayed` shows a pending function.

A launcher is **built** once its `ml_meth` is no longer the shim. That is
the only state transition: before it, options are editable; after it, they
are fixed and `options` is the resolved option the module was built from.

`make_launcher(jit_func, *, delayed=None, **kwargs)` stores
`MakeLauncherOption(jit_func, delayed=delayed, **kwargs)`. It still checks
only what it checks today: unknown keywords, a None kernel, a non-callable
`delayed`.

Two options with equal settings but different `delayed` functions compare
unequal (function identity). Nothing compares or hashes `MakeLauncherOption`;
`RenderKey` is separate.

## API on `k.__self__`

| In place, returns None | New launcher | Effect on `options` |
| --- | --- | --- |
| `_bind(**v)` | `bind(**v)` | `bind_values` merged with `v` |
| `_bind_device(n, **v)` | `bind_device(n, **v)` | `bind_device=n`, `bind_values` merged with `v` |
| `_unbind()` | `unbind()` | `bind_values=None` |
| `_unbind_device()` | `unbind_device()` | `bind_device=None` |
| `_replace_delayed(fn)` | `replace_delayed(fn)` | `delayed=fn`; `fn` may be None |
| `_chain_delayed(fn)` | `chain_delayed(fn)` | `delayed=lambda: {**old(), **fn()}`; just `fn` when `old` is None |
| `_resolve_delayed()` | `resolve_delayed()` | `replace(options, **delayed(), delayed=None)` |

Unchanged names: `options`, `check()`, `build()`, `spec_key()`,
`key_chain()`. They query or build; they do not edit options.

`foo(...)` is defined as: make a fresh unbuilt launcher from this one's
`options` (`_copy()`), call `_foo(...)` on it, return it. Each edit's effect
is written once, in `_foo`. `foo` works on a built launcher too: the copy is
unbuilt and starts from the built one's resolved options.

### Rules for `_foo`

- Is `_C_edit_options(edit)`, with `edit: MakeLauncherOption ->
  MakeLauncherOption`. C takes the build lock, refuses once built, calls
  `edit` on the current options and stores the result. Reading and writing
  under one lock means two threads' `_bind` calls never lose an update, and
  an edit is ordered against a concurrent first call: it applies before the
  build, or raises.
- Raises `RuntimeError("intj: a built launcher's options are fixed; the
  method without the leading underscore returns an edited copy")` once the
  launcher is built (C raises it and cannot name the method). Exception:
  `_resolve_delayed()` with `options.delayed` None is a no-op, built or not.
- Called from the launcher's own build (e.g. from its delayed function), the
  lock refuses it with the existing "called while it was being built" error.
- `_replace_delayed`/`_chain_delayed` raise `TypeError` for an `fn` that is
  not callable (None is allowed for `_replace_delayed` only), before taking
  the lock.
- Bind arguments are not validated here, as today: `check()` or the first
  call reports them.

## Resolving `delayed`

`_resolved(options) -> MakeLauncherOption`, a pure function used by
`_resolve_delayed()`, `check()` and the build:

1. If `options.delayed` is None, return `options`.
2. Call it. Its exception propagates; nothing is stored, so the next call
   retries.
3. The result must be a mapping whose keys are in `OPTION_FIELDS`, else
   `TypeError` (messages unchanged).
4. Return `dataclasses.replace(options, **result, delayed=None)`: **delayed
   replaces whole keys.** A delayed `bind_values` replaces a `bind(x=...)`;
   dicts are not merged. There is no clash error any more.

## Checking

`_check_options(option) -> _BuildOptions` replaces `_analyze` and `_bind`.
It asserts `option.delayed is None`, runs every GPU-free check (including
`_provision(kernel_cache)`), and returns `_BuildOptions`. That is a return
value only; nothing stores it.

`_BuildOptions` is a frozen dataclass holding everything the build needs
except what takes a GPU query (the target, and knob values read at the first
call). The build reads nothing from `option`: `option` holds values as given,
`_BuildOptions` holds them checked and normalized (e.g. `options=None`
becomes `()`).

Computed:

| Field | Computed from |
| --- | --- |
| `jit_func` | `option.jit_func`, the autotune/heuristics chain unwrapped, the kernel checked |
| `tuned` | the tuner chain: configs, tuned and exact-key params, render plan (None if untuned) |
| `params` | per-parameter annotations resolved; bound tensors' types filled in from their dtypes |
| `torch_access_mode` | `option.torch_access_mode`, None resolved |
| `grid_code` | `option.grid_cpp` compiled (`GridCode`), or None |
| `dynamic` | `dynamic_options` that are per-call slots |
| `declared_parameters` | `dynamic_options` naming a `tl.constexpr` parameter |
| `global_values` | the kernel's globals, when `assume_constant_globals` |
| `device_binding` | `bind_device` checked: `FIXED` or `NOT_FIXED` |
| `args` | `module._init`'s trailing arguments: `[device], *bound values` |
| `hidden` | `grid_py`'s fixed values: bound and baked |
| `doc` | the launcher's `__text_signature__` |

Checked copies of the option:

| Field | From | Build use |
| --- | --- | --- |
| `compile_options` | `options`, as a tuple of items | canonicalized against the target; tuned callback |
| `torch_access_request` | `torch_access_mode` as given | re-choosing the mode if `cpp_extension` cannot import |
| `kernel_cache` | `kernel_cache` | `RenderKey` |
| `no_gpu` | `no_gpu` | `RenderKey`; skips the target query |
| `verify_annotation` | `verify_annotation` | `RenderKey` |
| `ndebug` | `ndebug` | `RenderKey` |
| `return_compiled` | `return_compiled` | `RenderKey`; tuned callback |
| `grid_arg` | `grid_arg` | `RenderKey` |
| `grid_py` | `grid_py` | `RenderKey` (mode); `module._init`; tuned grid |
| `grid_cpp_fn` | `grid_cpp`, the function | its `__code__` arg names for the tuned grid |

`grid_code`/`grid_cpp_fn` replace today's `_Analysis.grid_cpp`/`grid_fn`,
whose names were swapped relative to what they held.

Analysis now runs per launcher and per check, so its warnings
(`ClassGlobalWarning`, the pointer-range `RuntimeWarning`) fire per launcher,
and twice for a launcher that runs `check()` before its first call. Measured
`_analyze` cost: ~60 us for a simple kernel with imports warm.

### `check()`

`check()` is `_C_check()`: on a built launcher it returns at once; otherwise
C takes the build lock and calls the Python `_dry_run()`, which is
`_check_options(_resolved(self._C_options))` with the result discarded. The
lock is only there to refuse re-entry: a `delayed` that calls its own
launcher's `check()` gets "called while it was being built" instead of
recursing forever. It stores nothing:

- A launcher with `delayed` calls `delayed()` at `check()` and again at the
  first call. `k.__self__._resolve_delayed()` then `check()` resolves it once.
- `_foo` stays allowed after `check()`.
- `__text_signature__` is not set by `check()`; the build sets it.

## The first call

`intj_lazy_call` -> `intj_lazy_build` -> under the build lock, the builder
(`Launcher._build_body`):

1. `option = _resolved(self.options)`
2. `build = _check_options(option)`
3. Import triton, refuse `TRITON_INTERPRET=1`, query the target, build the
   `RenderKey`/`RenderContext` from `(build, target)`
4. Render, compile (or hit the disk cache), load, `module._init(...)`
5. Return `(entry_address, option, build.doc)`; `option` is used only here,
   to publish the resolved options

C then, still under the lock: stores `option` in the `options` slot, stores
`doc` and points `def.ml_doc` at it (set once, freed only in dealloc, as
today), swaps `ml_meth` to the entry, drops `builder`, unlocks, and calls the
entry with the original arguments. A failure in steps 1-4 stores nothing and
leaves the shim, so the next call retries from step 1.

`build()`, `spec_key()` and `key_chain()` run the same build first, as today.

## Removed

- `_Origin`, `_Once`, `Delayed`, `_Analysis`, `_Binding`, `_Plan`
  (`_BuildOptions` replaces the last three),
  `_merge_delayed`'s clash check and its explicit-names set, `_prepare`.
- C: the `origin` and `plan` slots, `_C_origin`, `_C_plan`, `_C_set_plan`,
  `intj_lazy_prepare`.

## C stub (`runtime/intj_lazy.c`)

- `new_launcher(cls, options, builder)`.
- New `_C_edit_options(edit)`: takes the build lock, refuses once built,
  stores `edit(options)`.
- `_C_check()`: returns if built; else takes the build lock and calls
  `self._dry_run()`.
- The builder returns `(address: int, options, doc: bytes)` instead of an
  address; `intj_lazy_build` publishes `options` and `doc` before swapping
  `ml_meth`.
- `tests/test_runtime.py`'s direct test of the C type (~1086) follows.

## Naming

AGENTS.md's "`foo` if public, `_foo` if private" becomes: a public `foo` that
returns a new launcher has a public in-place twin `_foo`; other private
helpers must not use a public method's name with `_` prepended. Renames:
`_build` -> `_build_body` (the C `builder` argument follows); `_derive`
becomes `_copy()` (a fresh unbuilt launcher with the same `options`); the
`_check` method goes, replaced by the module-level `_check_options`.

## Tests

- `tests/test_delayed.py`: once-per-launcher replaces once-per-call
  (`test_runs_once_on_first_use_across_bind_children`,
  `test_concurrent_first_calls_run_it_once[siblings=True]`); the `no_gpu`
  clash case becomes an override case; `isinstance(options, Delayed)` checks
  become `options.delayed is not None`.
- `tests/test_launcher.py` (~1330): warnings per launcher, not per
  `make_launcher` call.
- Signature tests that call `check()` first (`tests/test_grid.py:84` and
  similar) call `build()` instead.
- `tests/test_launcher_object.py`: `OPTION_FIELDS` and `MakeLauncherKwargs`
  still exclude `delayed`; `_C_plan` assertions go.
- New, per edit: `_foo` and `foo` produce equal options; `foo` leaves the
  original unchanged; `_foo` raises after the build but not after `check()`.
- New: `check()` leaves `options` (including a pending `delayed`) unchanged;
  after the first call `options` is the resolved option with `delayed=None`.
- New: `resolve_delayed()` returns a launcher whose `options` is resolved;
  `chain_delayed` merges with the later function winning;
  `replace_delayed(None)` drops it; a delayed result containing `delayed` is
  a `TypeError`.
- New: `_bind` racing a first call on another thread either lands before the
  build (and the module is built with it) or raises.
- Mutation check: make resolution apply eager over delayed; an override test
  must fail.

## Docs

`make_launcher` docstring, `MakeLauncherOption` docstring, `Launcher.check`
docstring (dry run), README (`delayed` section), `docs/Usage.md`,
`MIGRATION_GAP.md` (the "no way back to the kernel" row: now closed for
`delayed=` too), delete the `TODO.md` item, AGENTS.md naming rule and the
`_intj_lazy` paragraph (no plan; options fixed at build).

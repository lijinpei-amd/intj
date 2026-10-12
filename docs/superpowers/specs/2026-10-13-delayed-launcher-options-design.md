# `delayed=`, `MakeLauncherOption`, and Methods on the Launcher Object

## Goal

- `make_launcher(..., delayed=lambda: {...})` defers `make_launcher`'s own
  arguments to the first call, so the decorator form works for options that
  need the GPU (README's `functools.cache` workaround goes away).
- Every `make_launcher` option except `delayed` lives in one dataclass,
  `MakeLauncherOption`.
- Decoration validates nothing: it only builds the `MakeLauncherOption`. All
  consistency checks run in `check()`, at the latest on the first call.
- The module methods (`init_bound`, `header_size`, `spec_key`,
  `set_torch_version`) and `LauncherFactory.bind`/`bind_device` become methods
  on one launcher object, plus new `unbind`/`unbind_device`.
- The hit path keeps today's cost, and the launcher keeps CPython's
  `CALL_BUILTIN_FAST` specialization.

## The callable and the launcher object

`k = make_launcher(...)` is a plain `PyCFunction`, as today. Its `__self__` is
the launcher object: an instance of `intj.launcher.Launcher`, a Python
subclass (`__slots__ = ()`) of the C type `_intj_lazy.Launcher`.

The swappable entry is `k.__self__.def.ml_meth`, a `PyMethodDef` embedded in
the launcher object: the shim until the first call succeeds, then the rendered
entry. This is the request's "swappable vectorcall pointer", placed where
specialization can see it.

### Why not a custom type with `tp_vectorcall_offset`

`CALL_BUILTIN_FAST` (3.13+; `CALL_NO_KW_BUILTIN_FAST` on 3.12,
`PRECALL_NO_KW_BUILTIN_FAST` on 3.11) requires `Py_TYPE == &PyCFunction_Type`
exactly and `ml_flags == METH_FASTCALL`, then calls `m_ml->ml_meth(m_self,
args, nargs)` directly. A custom type gets `CALL_NON_PY_GENERAL` on 3.13+ and
no specialization on 3.11/3.12; a spike measured +1.4 to 5 ns per call.

## API

### Decoration

```python
@make_launcher(options={"num_warps": 8})                 # MakeLauncherOption, no checks
@make_launcher(delayed=lambda: {"options": pick()})      # Delayed, no checks
k = make_launcher(kernel, bind_device=0, bind_values={"x": t})
```

`make_launcher(jit_func=_NO_KERNEL, *, delayed=None, **kwargs)`:

1. Without `jit_func`, returns the decorator (`functools.partial`), as today.
2. Builds `MakeLauncherOption(jit_func, **kwargs)`. An unknown keyword is a
   `TypeError`. So is a None kernel, or a `delayed` that is neither None nor
   callable.
3. Returns the launcher. No other check, no torch import, no provisioning.

`kwargs` is typed `Unpack[_MakeLauncherKwargs]` (a `total=False` TypedDict),
so pyright still checks call sites, and "passed explicitly" is decided by name,
never by comparing a value against its default. The return type is a
`LaunchFunction` Protocol (`__call__`, `__self__: Launcher`), so
`k.__self__.bind(...)` type-checks.

Decoration still imports triton, as today: loading `_intj_lazy` needs triton's
cache directory and compiler.

### `MakeLauncherOption` (`intj/options.py`, exported from `intj`)

A frozen dataclass. Values are stored as given; nothing is validated here.

| Field | Type | Default |
|---|---|---|
| `jit_func` | `JitFunction` | required |
| `dynamic_options` | `Sequence[str]` | `()` |
| `extra_annotation` | `Mapping[str, object] \| None` | `None` |
| `options` | `Mapping[str, Any] \| None` | `None` |
| `torch_access_mode` | `TorchAccessMode \| None` | `None` |
| `kernel_cache` | `KernelCache` | `KernelCache.INTJ` |
| `no_gpu` | `bool` | `False` |
| `verify_annotation` | `bool` | `False` |
| `bind_device` | `int \| None` | `None` |
| `bind_values` | `Mapping[str, object] \| None` | `None` |
| `grid_arg` | `int \| None` | `None` |
| `grid_cpp` | `object \| None` | `None` |
| `grid_py` | `Callable[[dict[str, object]], object] \| None` | `None` |
| `return_compiled` | `bool` | `False` |
| `assume_constant_globals` | `bool` | `False` |

`bind_device` changes meaning: it is the device ordinal, not a flag. The
`bool` flag and the non-callable `LauncherFactory` are removed;
`make_launcher` always returns a launcher.

### `delayed`

- Takes no arguments and returns a mapping of `make_launcher` keywords (any
  field above except `jit_func`), merged with the explicitly passed ones.
- Runs once per `make_launcher` call, on the first `check()` or first call of
  any launcher derived from it. Its result is shared by those launchers.
- A key also passed explicitly raises `UnsupportedKernel`. A non-mapping
  result, a non-str key, an unknown field, `jit_func` or `delayed` raises
  `TypeError`.
- If it raises, the exception propagates unwrapped and the next call runs it
  again: failures are not cached.

### Methods on `k.__self__`

| Method | Returns / does |
|---|---|
| `check()` | Resolves `delayed`, runs every check `make_launcher` runs today, provisions the kernel cache. `None` or raises. Cached on success. |
| `bind(**values)` | New launcher, `bind_values` merged with `values` |
| `bind_device(ordinal, **values)` | New launcher, `bind_device=ordinal`, `bind_values` merged |
| `unbind()` | New launcher, `bind_values=None` |
| `unbind_device()` | New launcher, `bind_device=None` |
| `build()` | Builds if needed; returns the rendered module (`module_of` uses it) |
| `spec_key(device, *dynamic, *public_args)` | As the module's `spec_key` today, without the launcher argument; builds first |
| `key_chain(device, *dynamic, *public_args)` | Same, tuned launchers only |
| `options` | The stored `MakeLauncherOption`, or the `Delayed` until it is resolved |

`bind`/`bind_device`/`unbind`/`unbind_device` validate nothing: they
`dataclasses.replace` the option (or derive a new `Delayed`) and return a new
launcher. The original is unchanged. A bad ordinal or bound value fails in the
new launcher's `check()`. `bind_device=True` fails there with a message
pointing at `bind_device(ordinal)`, since `True` is an int.

The rendered module keeps only private loader hooks: `_init` (was
`init_bound`), `_set_torch_version`, `_header_size`.

## C object

### Launcher header (`runtime/intj_lazy.h`)

Allocated at decoration, GPU-free, fixed size:

```c
typedef struct intj_launcher {
  PyObject_HEAD
  void *state;              /* hot: the module's per-launcher block; NULL until built / after clear */
  PyMethodDef def;          /* ml_meth: shim -> rendered entry; METH_FASTCALL */
  PyObject *options;        /* MakeLauncherOption, or the Delayed until resolved */
  PyObject *origin;         /* _Origin, shared with every bind/unbind derivative */
  PyObject *plan;           /* check()'s _Plan; NULL until checked */
  PyObject *builder;        /* intj.launcher._build; tests pass their own */
  PyObject *module;
  PyObject *doc;            /* bytes behind def.ml_doc, once checked */
  PyThread_type_lock build_lock;
  unsigned long build_thread;
  const intj_launcher_ops *ops;  /* the module's traverse / clear / free / spec_key / key_chain */
} intj_launcher;
```

`state` stays at `sizeof(PyObject)` (static assert). `options`, `origin` and
`plan` are private members readable from Python; `options` is public
read-only.

`_intj_lazy.new_launcher(cls, options, origin, builder)` allocates `cls` and
returns `PyCFunction_New(&h->def, h)`. `_stub()` creates `intj.launcher.Launcher`
once per process from a typed mixin (`bind`, `bind_device`, `unbind`,
`unbind_device`, `check`); `spec_key`, `key_chain` and `build` stay in C so
the decode-and-key benchmark row gets no Python frame.

### Per-launcher state (`intj_launcher_state`, `runtime/intj_runtime.h`)

`_init` allocates it (`PyMem_RawCalloc`, the module's own size) and fills it:
the module-state pointer, then everything the header and its trailing arrays
hold today -- device ordinal and handle, `cache_ready`, the cache,
`fixed_kernel`, the lock, the interner, `compile_cb`, `grid_py`,
`grid_hidden`, memos, bound slots, the object table. Python no longer mirrors
its size: `_tail_bytes` and its static assert go. `_header_size` stays, as the
stub/module agreement check in `_loaded_module`.

### First call

1. `CALL_BUILTIN_FAST` calls `ml_meth`, the shim. The shim takes
   `build_lock`; the same thread re-entering is refused, other threads wait
   (as today).
2. The shim calls `builder(self)`; see "Builder" below. It ends with
   `module._init(self, ...)`, which fills `state`, sets `ops`, stores `state`
   last, and returns the entry address.
3. The shim release-stores `ml_meth = entry` and calls it.

A failure at any step leaves the shim installed and `plan`/`state` as they
were; the next call retries.

### Cleanup

`tp_traverse`/`tp_clear` visit the header's objects, then `ops->traverse` /
`ops->clear`. Clear sets `state = NULL` first, drops references, then
`ops->free`s the block. Dealloc frees `doc`.

### Cost and fallback

The hot path gains one dependent load (`self -> state -> st`/cache), an L1
hit, ~1 ns. Gate: `bench_launch.py --readme` zero-grid and `--last-key` rows
within +1 ns of `main`. If over budget, apply in order:

1. Keep a module-state pointer in the header next to `state`, loaded in
   parallel.
2. Swap `PyCFunctionObject.m_self` from the header to the state on the first
   call (`CALL_BUILTIN_FAST` reads it live). Rejected as the primary design:
   the `ml_meth`/`m_self` pair is not swapped atomically (safe under the GIL,
   not free-threaded without a type check in both shim and entry); the old
   self must stay alive for in-flight calls; and `k.__self__` would change
   identity after the first call.

## Python pipeline

### Units (`intj/options.py`, `intj/launcher.py`)

- `MakeLauncherOption` -- above.
- `Delayed` -- the callable stored in `options` for a delayed launcher:
  `Delayed()` returns a `MakeLauncherOption`. Holds the `_Origin` plus this
  launcher's bind edits.
- `_Origin` -- one per `make_launcher` call: the explicit option, the names
  passed explicitly, the user's `delayed` function, and two once-cells (the
  `delayed` result, the `_Analysis`). A once-cell is lock-guarded and does not
  cache failures.
- `_Analysis` -- today's `LauncherFactory` without the bind fields.
- `_Binding` -- per launcher: resolved params, `init` args, hidden values,
  doc.
- `_Plan` -- `(_Analysis, _Binding)`, cached in `plan`.

### `check()`

Runs under `build_lock` (the builder, already holding it, calls the unlocked
form):

1. **Resolve.** `Delayed()` runs the user's function once per origin,
   validates the mapping, raises on a conflict with an explicit name, merges,
   then applies the bind edits. The result replaces the `options` slot.
2. **Analyze**, once per origin: today's `make_launcher` body, from the cheap
   type checks through `_verified_cpython_header`, `_check_kernel`,
   annotations, the tuning plan, `_check_dynamic`, the constexpr clash,
   `compile_grid`, `_resolve_torch_access_mode`, `_provision` and
   `ClassGlobalWarning`; plus the `InterpretedFunction` refusal. Bind fields
   are not inputs, so every derivative shares it.
3. **Bind**, per launcher: today's `_bind` -- unknown/missing/mistyped bound
   values, bound-tensor dtypes, the ordinal in `[0, 2**31)`, the bound-pointer
   warning. An unbound bind parameter is a `TypeError` here.
4. Set `doc` and `def.ml_doc`. Before this the launcher has no
   `__text_signature__`.

### Builder (`intj.launcher._build(launcher)`, process-wide)

`check()`, then today's `LauncherFactory._build` body: the
`TRITON_INTERPRET` refusal, live knobs, `_check_tuned_configs`,
`_materialize_module` -> `RenderKey` -> `_loaded_module`, the compile
callback, `module._init(launcher, callback, _Interner(), *grid_py, *args)`.

`RenderKey`, `RenderContext` and `entry.c.jinja` are unchanged in role. The
key is derived from `_Analysis`, `_Binding` and the target, never from the
option object, so delayed values enter module identity exactly as explicit
ones do, and two options that analyze alike share one module.

### Deleted

`LauncherFactory`, `_interpreter_deferred` and `_DeferredFactory`,
`_tail_bytes`.

### Errors

| When | Raises |
|---|---|
| `make_launcher` | `TypeError` only: unknown keyword, None kernel, non-callable `delayed` |
| `check()` / first call | Everything else, same types and messages as today; `delayed`'s own exception unwrapped |
| Retry | A failed check or build leaves `plan` NULL and the shim in place; the next call retries from the failed step. `_provision` keeps its one-failure-per-process memo. |

Warnings (`ClassGlobalWarning`, bound-pointer `RuntimeWarning`) fire at
`check()` or the first call.

## Testing

New, in `tests/test_delayed.py` and `tests/test_launcher.py`:

1. Decoration does no work: a bad option passes `make_launcher` and raises
   from `__self__.check()` with today's message; an unknown keyword is a
   `TypeError` at decoration; decorating imports no torch.
2. `delayed`: not called at decoration; called once across a launcher and its
   `bind` children; conflict -> `UnsupportedKernel`; non-mapping or unknown
   key -> `TypeError`; when it raises, the next call retries (count 2); eight
   threads making the first call together -> one call.
3. Module identity: two delayed launchers returning different `options` get
   different `RenderKey` digests; equal results share one module.
4. Bind/unbind: each returns a new launcher and leaves the original unchanged;
   `unbind()` then a call -> `TypeError`; `bind_device`/`unbind_device`
   round-trip; derivatives run `_Analysis` once; `bind_device=True` ->
   `TypeError` from `check()`.
5. Kept: the `CALL_BUILTIN_FAST` pin
   (`test_first_call_builds_and_swaps_for_old_references`), also run on a
   delayed launcher; the GC/cycle tests; both invariant tests and their
   mutation tests -- only how they reach `spec_key` changes.

Gates: `pyright` 0 errors; `ninja -C ci` and `test-gpu` on gfx942 and sm_89;
the bench gate above.

Untested, as today: publication order of `state` before `ml_meth` on non-TSO
hosts (aarch64).

## Migration

| From | To |
|---|---|
| `module_of(k).spec_key(k, ...)` / `key_chain` | `k.__self__.spec_key(...)` |
| `make_launcher(k, bind_device=True).bind_device(d, **v)` | `make_launcher(k).__self__.bind_device(d, **v)` |
| `factory.bind(**v)` | `k.__self__.bind(**v)` |
| `with pytest.raises(...): make_launcher(...)` (41) | `... make_launcher(...).__self__.check()` |
| `stub.new_launcher(_TAIL, doc, build)` | `stub.new_launcher(stub.Launcher, option, origin, build)` |

~250 call sites: `intj/compat.py`, `benchmarks/bench_launch.py`,
`bench_ffi_compare.py`, `bench_hip_module_launch.py`, `bench_intj_ffi_paths.py`,
and `tests/test_{launcher,runtime,grid,kernel_cache,return_compiled,tuned}.py`.
`module_of` stays for tests that need the module object.

## Docs

- README: a `delayed=` example replaces the `functools.cache` factory and the
  "not implemented" note.
- `docs/Usage.md`: bind, lazy build, error timing, `spec_key`.
- AGENTS.md: refusals happen in `check()`, at the latest on the first call;
  provisioning moves there; `init_bound` -> `_init`, which installs `ops`.
- TODO.md: drop "Deferred `make_launcher` arguments".
- Earlier specs and plans stay as written.

## Commit order

Each commit leaves the suite green:

1. `MakeLauncherOption` and the `check()` split, on today's header.
2. The header/state C change; methods on `__self__`.
3. `delayed`.
4. `unbind`/`unbind_device`.
5. Docs and the bench run.

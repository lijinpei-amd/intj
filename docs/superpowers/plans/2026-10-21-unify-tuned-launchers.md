# Unify Tuned and Ordinary Launchers Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One compile core, one rendered miss interface, and a `TunedLauncher(Launcher)` class, so tuned and ordinary launchers differ only where the work differs.

**Architecture:** Python first: `_KernelCompiler` and four small helpers replace the compile code duplicated between `_make_compile_callback` and `TunedCompileState`, then `TunedLauncher` overrides three hooks of `Launcher` instead of `Launcher` branching on `b.tuned`. Then C: the template renders an ordinary launcher as the one-level case of the tuned walk, so every module calls `_compile_miss` with the same arguments, reads back the same record, and exports `key_chain`.

**Tech Stack:** Python 3.8+ (pyright strict), Jinja2-rendered C (`intj/render/entry.c.jinja`), CPython C API, Triton 3.8, torch 2.14 (ROCm, gfx942).

**Spec:** none written; the design was agreed in conversation (2026-10-21) and is recorded in § Design below.

## Global Constraints

- `pyright --pythonpath $V/bin/python` reports 0 errors.
- Runtime syntax stays Python 3.8 (`requires-python = ">=3.8"`): no `tuple[...]`/`X | Y` outside annotations.
- `import intj` imports neither triton nor torch; `launcher.py` imports `intj.tuning` only inside functions (tuning imports launcher at its top).
- `RenderContext` computes, the template writes C: the template branches on a boolean, a `none` test, or one field against a literal (AGENTS.md).
- No new docstrings on simple private helpers under ~10 lines; public and overridable methods get one.
- Tests monkeypatch `intj.launcher._checked_compile`, `_parameter_options`, `_make_compile_callback`, `_loaded_module`, `_cache_key_fields`: keep these names, and keep `tuning.py` reading launcher internals at call time (`from .launcher import ...` inside the function).
- Every command uses the project venv:

```bash
V=/mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/venv
PYTHONPATH=$PWD VIRTUAL_ENV=$V uv run --active --no-sync python -m pytest tests -q   # ~14 min, baseline 1099 passed, 15 skipped
pyright --pythonpath $V/bin/python
pre-commit run --files <changed files>
```

## Design

Where the two launchers stay different (rendered from `tuned`, `tuned_meta`, `grid_eval`, `levels`): heuristics lowered to C and the extra lookup levels, `dep`/`meta` in the final record, the grid evaluated after the lookup, and who launches on a miss (Triton's autotuner launches the winner, so a tuned miss only stores the record).

What becomes one:

| | Before | After |
|---|---|---|
| Compile | `compile_callback` and `TunedCompileState._compile_kernel` each merge options, build the input, cache, `_checked_compile` | both use `_KernelCompiler` + `_Identity`, `_check_device`, `_split_call`, `_record` |
| Python class | `Launcher._build_body` / `_compile_miss` branch on `b.tuned` / `state.tuned` | `TunedLauncher(Launcher)` overrides `_check_target`, `_new_compile_state`, `_compile_miss` |
| Miss call (C → Python) | ordinary `(keyblob, nparams, device, *args)`; tuned adds `stream, controls` | every module: `(keyblob, nparams, device, stream, controls, *args)` |
| Record (Python → C) | ordinary 4/5-tuple, tuned 8-tuple, parsed by two fills | one `intj_fill` parses `(function, block_dim, shared, nparams[, compiled[, deps, computed[, meta]]])` |
| Lookup | ordinary: inline lookup; tuned: `intj_walk` | `intj_walk` for both (one level when ordinary) |
| `key_chain` op | NULL unless tuned | every module; an ordinary chain is `(spec key,)` |

Out of scope, on purpose:
- `override_compile` stays ordinary-only: its `fn` returns a launch record, and a tuned miss must run Triton's chain, which launches.
- Each `TunedLauncher` keeps its own `_KernelCompiler`, as `TunedCompileState.compiled` is today, so dropping the launcher still frees its candidates.
- `_check_options` keeps calling `plan_tuning`/`tuning_render` where they sit: they interleave with refusals whose order tests pin. `Launcher._check_target` asserts the class choice and `_check_options` agree.

## Review Focus

- The ordinary hot path after its lookup moves into `intj_walk` and `comp`/`dep` become unconditional: expect identical timings (Task 4, Step 9 benchmark gate).
- A keyless bound launcher (`nwords == 0`, `fixed_kernel`): miss, re-walk, and `key_chain` (Task 4 test `test_key_chain_of_a_keyless_launcher_is_one_empty_key`).
- `override_compile` fakes returning 4-tuples keep working; under `return_compiled` a 4-tuple is a clear `TypeError` (Task 4 test `test_return_compiled_record_carries_its_kernel`).
- Copies of a tuned launcher (`bind_device`, `replace_delayed`, ...) stay `TunedLauncher` (Task 3 test).
- Patching `intj.launcher._checked_compile` still reaches both miss paths, now through `_KernelCompiler.compile` (Task 2 test patches that method).

---

### Task 1: Commit the tuning move

The move of `_Tuned`/`_plan_tuning`/`_config_options`/`_check_tuned_configs`/`_tuning_render` into `tuning.py` is in the working tree and passed the full suite (1099 passed, 15 skipped) and pyright.

- [ ] **Step 1: Commit**

```bash
git add intj/launcher.py intj/tuning.py
git commit -m "tuning: move autotune/heuristics planning out of launcher.py

Tuned, plan_tuning (now also marking tuned/exact_key), check_tuned_configs,
tuning_render and tuned_grid live in tuning.py; launcher.py imports them
where it uses them.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: One compile core for both miss paths

**Files:**
- Modify: `intj/launcher.py` (new helpers after `_check_live_knobs`; `_KernelCompiler`, `_Identity`, `_record` before `_make_compile_callback`; rewrite `_make_compile_callback`)
- Modify: `intj/tuning.py` (`TunedCompileState.__init__`, `_compile_kernel`, `_tune`)
- Test: `tests/test_tuned.py`

**Interfaces:**
- Produces (launcher.py, private, used by tuning.py):
  - `_check_device(device: int) -> None` — raises `UnsupportedKernel` unless `device` is current.
  - `_split_call(dynamic: Sequence[str], args: Sequence[Any]) -> tuple[dict[str, object], dict[str, object], Sequence[Any]]` — `(dynamic options, dynamic knobs, rest)`, live knobs checked.
  - `_record(kernel: CompiledKernel, nparams: int) -> tuple[int, int, int, int]`.
  - `_Identity` frozen dataclass `(compiler_input, options_hash, knobs)`, `_Identity.of(compiler_input, canonical, knobs)`.
  - `_KernelCompiler(jit_func, options, knob_values, *, weak: bool)` with `.jit_func`, `.target`, `.backend`, `.canonical(options, knobs) -> Any`, `.compile(identity, canonical) -> CompiledKernel`.
- `_make_compile_callback` keeps its signature (tests call it).

- [ ] **Step 1: Write the failing test** — append to `tests/test_tuned.py`:

```python
def test_tuned_and_ordinary_misses_share_the_kernel_compiler(monkeypatch):
    """Both miss paths reach Triton through `_KernelCompiler.compile`."""
    import intj.launcher as launcher_mod

    real = launcher_mod._KernelCompiler.compile
    calls = []

    def counting(self, identity, canonical):
        calls.append(identity)
        return real(self, identity, canonical)

    monkeypatch.setattr(launcher_mod._KernelCompiler, "compile", counting)
    x = torch.zeros(256, device="cuda", dtype=torch.int32)
    out = torch.zeros_like(x)
    bench = Bench(out, {16: {1: 1.0, 2: 2.0}})
    _, tuned = tuned_launcher(bench)
    device, stream = controls()
    bench.key = 16
    tuned(device, stream, x, out, 16)
    tuned_calls = len(calls)
    assert tuned_calls >= 2, "each autotune config compiles through it"
    plain = make_launcher(tagged)
    plain(device, stream, (1,), x, out, 16, 1, 32)
    torch.cuda.synchronize()
    assert len(calls) == tuned_calls + 1
    assert int(out[0].item()) == 1
```

- [ ] **Step 2: Run it to verify it fails**

Run: `PYTHONPATH=$PWD VIRTUAL_ENV=$V uv run --active --no-sync python -m pytest tests/test_tuned.py -q -k share_the_kernel_compiler`
Expected: FAIL, `AttributeError: module 'intj.launcher' has no attribute '_KernelCompiler'`.

- [ ] **Step 3: Add the helpers to `intj/launcher.py`**

After `_check_live_knobs` (keep `_split_dynamic`: `tests/test_tuned.py` imports it):

```python
def _split_call(
    dynamic: Sequence[str], args: Sequence[Any]
) -> tuple[dict[str, object], dict[str, object], Sequence[Any]]:
    """Splits a miss's arguments into its dynamic options, its knobs, and the rest.

    Raises:
        ValueError: A declared knob was passed other than its live value.
    """
    options, knobs = _split_dynamic(dynamic, args[: len(dynamic)])
    _check_live_knobs(knobs)
    return options, knobs, args[len(dynamic) :]


def _check_device(device: int) -> None:
    # _init_handles() loads the binary on the *current* device; launching that
    # function on another device's stream is a wrong-context launch
    current = backends.current_device()
    if device != current:
        raise UnsupportedKernel(
            f"intj: launching on device {device} while device {current} is current; "
            "make the target device current before the first launch"
        )
```

Immediately before `_refuse_module_compile`:

```python
@dataclasses.dataclass(frozen=True)
class _Identity:
    """What one compile is cached and self-checked by.

    Knobs change the binary even where they feed no option.
    """

    compiler_input: CompilerInput
    options_hash: str
    knobs: tuple[tuple[str, object], ...]

    @classmethod
    def of(
        cls,
        compiler_input: CompilerInput,
        canonical: Any,
        knobs: Mapping[str, object],
    ) -> _Identity:
        return cls(compiler_input, canonical.hash(), tuple(sorted(knobs.items())))


class _KernelCompiler:
    """Compiles `jit_func` for cache misses, once per identity and device.

    A module's ordinary launchers share one through its `_intj_compile`; each
    tuned launcher owns one for its candidates.
    """

    def __init__(
        self,
        jit_func: JitFunction,
        options: Mapping[str, Any],
        knob_values: Mapping[str, object],
        *,
        weak: bool,
    ) -> None:
        self.jit_func = jit_func
        self.target = backends.current_target()
        self.backend = _imports.make_backend(self.target)
        self._options = dict(options)
        self._knob_values = dict(knob_values)
        # weak where the records own their CompiledKernel (return_compiled):
        # the cache must not keep it alive past them
        self._compiled: MutableMapping[tuple[object, ...], CompiledKernel] = (
            weakref.WeakValueDictionary() if weak else {}
        )
        # Reentrant: a compile can launch this kernel again (a nested miss on
        # the same thread).
        self._lock = threading.RLock()

    def canonical(self, options: Mapping[str, Any], knobs: Mapping[str, object]) -> Any:
        """Returns the target's options: `options` over the base ones, then knobs."""
        return backends.canonical_options(
            self.target,
            _knob_options(
                self.jit_func,
                {**self._options, **options},
                {**self._knob_values, **knobs},
            ),
        )

    def compile(self, identity: _Identity, canonical: Any) -> CompiledKernel:
        """Returns the kernel for `identity` on the current device, compiled once."""
        key = (identity, backends.current_device())
        with self._lock:
            kernel = self._compiled.get(key)
            if kernel is None:
                fresh = _checked_compile(
                    self.jit_func, identity.compiler_input, self.target, canonical
                )
                # a nested miss may have stored one meanwhile; its record may
                # already be cached, so its kernel is the one that must stay
                kernel = self._compiled.setdefault(key, fresh)
        return kernel


def _record(kernel: CompiledKernel, nparams: int) -> tuple[int, int, int, int]:
    """Returns `kernel`'s record for C: function, block_dim, shared, nparams.

    Raises:
        RuntimeError: Triton compiled another argument count than C packed.
    """
    md = kernel.metadata
    expected = sum(1 for ty in kernel.src.signature.values() if ty != "constexpr")
    if expected != nparams:
        raise RuntimeError(
            f"intj: packed {nparams} kernel arguments but triton compiled {expected}; "
            "this is an intj bug"
        )
    return (kernel.function, md.warp_size * md.num_warps, md.shared, nparams)
```

- [ ] **Step 4: Rewrite `_make_compile_callback`'s body** (signature and docstring unchanged)

```python
    # The module's compile cache: a sibling launcher's miss reuses the kernel
    # and only builds its own C record.
    compiler = _KernelCompiler(jit_func, options, knob_values, weak=return_compiled)
    fed = {
        p.name: p
        for p in params
        if p.name in _parameter_options(compiler.backend, params)
    }
    fixed = {
        n: (baked_values or {})[p.index] for n, p in fed.items() if p.call_index is None
    }
    per_call = {n: p.call_index for n, p in fed.items() if p.call_index is not None}
    # without per-call values every record has the same options
    fixed_options = None if dynamic or per_call else compiler.canonical(fixed, {})

    def compile_callback(
        seen: dict[bytes, object], keyblob: bytes, nparams: int, device: int, *args: Any
    ) -> tuple[int, int, int, int] | tuple[int, int, int, int, CompiledKernel]:
        """Compiles, or reuses, the kernel for one miss.

        `seen` is the calling launcher's own key -> identity map.
        """
        _check_device(device)
        dyn_options, dyn_knobs, args = _split_call(dynamic, args)
        canonical = (
            fixed_options
            if fixed_options is not None
            else compiler.canonical(
                {**fixed, **dyn_options, **{n: args[i] for n, i in per_call.items()}},
                dyn_knobs,
            )
        )
        compiler_input = _compiler_input(
            jit_func, params, args, compiler.backend, baked_values=baked_values
        )
        identity = _Identity.of(compiler_input, canonical, dyn_knobs)
        # b"" is a keyless launcher's one key; seen is per launcher, so that is exact
        if seen.setdefault(keyblob, identity) != identity:
            raise RuntimeError(
                "intj: one spec key maps to two annotated ASTSource inputs; this is an intj bug"
            )
        kernel = compiler.compile(identity, canonical)
        result = _record(kernel, nparams)
        return (*result, kernel) if return_compiled else result

    return compile_callback
```

Option order is unchanged: base < baked fed < dynamic < per-call fed.

- [ ] **Step 5: Point `TunedCompileState` at the core** (`intj/tuning.py`)

In `__init__`, replace the `_parameter_options` import, `self.target`, `self.backend`, `self.options`, `self.knob_values`, the `self.compiled` dict and its comment, and `self.fed` with:

```python
        from .launcher import (
            _KernelCompiler,  # pyright: ignore[reportPrivateUsage]  # launcher internals
            _parameter_options,  # pyright: ignore[reportPrivateUsage]  # launcher internals
        )

        self.plan = plan
        self.jit_func = plan.jit_func
        # its own, strong: a CompiledKernel is loaded on one device, and
        # records point into candidates no module sees
        self.compiler = _KernelCompiler(self.jit_func, options, knob_values, weak=False)
        self.grid = grid
        self.render = render
        self.dynamic = dynamic
```

…and keep `self.params`, `self.call_knobs`, then `self.fed = _parameter_options(self.compiler.backend, self.params)`, `self.shim`, `self.private`, `self.public`, `self.baked`, `self.lock`, `self.running` as they are.

Replace `_compile_kernel`'s body:

```python
        from .launcher import (
            _compiler_input,  # pyright: ignore[reportPrivateUsage]  # launcher internals
            _Identity,  # pyright: ignore[reportPrivateUsage]  # launcher internals
        )

        canonical = self.compiler.canonical(
            {**config_options, **{n: values[n] for n in self.fed}}, self.call_knobs
        )
        compiler_input = _compiler_input(
            self.jit_func,
            self.params,
            [values[p.name] for p in self.params],
            self.compiler.backend,
        )
        return self.compiler.compile(
            _Identity.of(compiler_input, canonical, self.call_knobs), canonical
        )
```

In `_tune`, replace the imports, the device check and the dynamic split:

```python
        from .launcher import (
            _check_device,  # pyright: ignore[reportPrivateUsage]  # launcher internals
            _record,  # pyright: ignore[reportPrivateUsage]  # launcher internals
            _split_call,  # pyright: ignore[reportPrivateUsage]  # launcher internals
        )

        _imports.ensure_torch_imported()
        _check_device(device)
        dyn_options, self.call_knobs, args = _split_call(self.dynamic, args)
```

…and replace the `md = kernel.metadata` … `return (kernel.function, md.warp_size * md.num_warps, md.shared, nparams, kernel, ...)` block with:

```python
            return (
                *_record(kernel, nparams),
                kernel,
                tuple(c_scalar(n, tuned_values[n]) for n in self.render.dep_names),
                tuple(tuned_values[n] for n in self.render.computed_names),
                tuple(values[n] for n in self.render.meta_names)
                if self.grid.mode == "py"
                else None,
            )
```

Then `grep -n "self\.\(target\|backend\|options\|knob_values\|compiled\)\b" intj/tuning.py` must print nothing, and `grep -rn "\.compiled\b\|_compile_kernel" tests` must show no test reading the removed attributes.

- [ ] **Step 6: Run the new test, the tuned suite, pyright**

Run: `PYTHONPATH=$PWD VIRTUAL_ENV=$V uv run --active --no-sync python -m pytest tests/test_tuned.py tests/test_tuning.py -q && pyright --pythonpath $V/bin/python`
Expected: all pass, 0 errors.

- [ ] **Step 7: Run the full suite**

Run: `PYTHONPATH=$PWD VIRTUAL_ENV=$V uv run --active --no-sync python -m pytest tests -q`
Expected: 1100 passed, 15 skipped.

- [ ] **Step 8: Commit**

```bash
pre-commit run --files intj/launcher.py intj/tuning.py tests/test_tuned.py
git add intj/launcher.py intj/tuning.py tests/test_tuned.py
git commit -m "launcher: one _KernelCompiler for ordinary and tuned misses

_check_device, _split_call, _Identity and _record replace the copies in
compile_callback and TunedCompileState.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: `TunedLauncher(Launcher)`

**Files:**
- Modify: `intj/launcher.py` (`_CompileState`, `Launcher` TYPE_CHECKING block, `_build_body`, new `_check_target`/`_new_compile_state`, `_compile_miss`, `_stub_and_type` → `_stub_and_types`, `_stub`, `_new_launcher`)
- Modify: `intj/tuning.py` (new `TunedLauncher` at the end of the file)
- Modify: `AGENTS.md` (launcher paragraph)
- Test: `tests/test_tuned.py`

**Interfaces:**
- Consumes: `check_tuned_configs`, `tuned_grid`, `TunedCompileState` (tuning.py); `_BuildOptions` (launcher.py, TYPE_CHECKING import in tuning.py).
- Produces:
  - `Launcher._check_target(self, b: _BuildOptions) -> None`, `Launcher._new_compile_state(self, b: _BuildOptions) -> object` (hooks).
  - `intj.tuning.TunedLauncher(Launcher)`, `__slots__ = ()`, overriding `_check_target`, `_new_compile_state`, `_compile_miss`.
  - `_stub_and_types() -> tuple[types.ModuleType, type[Launcher], type[Launcher]]` (stub, ordinary type, tuned type).
  - `_CompileState(seen=...)` without the `tuned` field.

- [ ] **Step 1: Write the failing test** — append to `tests/test_tuned.py`:

```python
def test_a_chain_gets_a_tuned_launcher_and_its_copies_keep_it():
    from intj.launcher import Launcher, _CompileState
    from intj.tuning import TunedCompileState, TunedLauncher

    x = torch.zeros(256, device="cuda", dtype=torch.int32)
    out = torch.zeros_like(x)
    bench = Bench(out, {16: {1: 1.0, 2: 2.0}})
    _, launch = tuned_launcher(bench)
    device, stream = controls()
    copies = [
        launch.__self__.bind_device(device),
        launch.__self__.replace_delayed(None),
    ]
    assert all(isinstance(k.__self__, TunedLauncher) for k in (launch, *copies))
    plain = make_launcher(tagged)
    assert isinstance(plain.__self__, Launcher)
    assert not isinstance(plain.__self__, TunedLauncher)
    bench.key = 16
    launch(device, stream, x, out, 16)
    plain(device, stream, (1,), x, out, 16, 1, 32)
    assert type(launch.__self__._C_compile_state) is TunedCompileState
    assert type(plain.__self__._C_compile_state) is _CompileState
```

- [ ] **Step 2: Run it to verify it fails**

Run: `PYTHONPATH=$PWD VIRTUAL_ENV=$V uv run --active --no-sync python -m pytest tests/test_tuned.py -q -k gets_a_tuned_launcher`
Expected: FAIL, `ImportError: cannot import name 'TunedLauncher'`.

- [ ] **Step 3: `_CompileState` and the TYPE_CHECKING block** (`intj/launcher.py`)

```python
@dataclasses.dataclass
class _CompileState:
    """One ordinary launcher's spec key -> compile identity checks."""

    seen: dict[bytes, object] = dataclasses.field(default_factory=dict)
```

In `if TYPE_CHECKING:` at the top, `from .tuning import Tuned, TunedCompileState` becomes `from .tuning import Tuned`. In `Launcher`'s TYPE_CHECKING block, `def _C_compile_state(self) -> _CompileState: ...` becomes `def _C_compile_state(self) -> object: ...`.

- [ ] **Step 4: Hooks in `Launcher`**

Replace `_build_body`'s body after its docstring (docstring unchanged):

```python
        option = _resolved(self._C_options)
        b = _check_options(option)
        _imports.ensure_triton_imported()
        if _imports.knobs.runtime.interpret:
            raise UnsupportedKernel("intj: TRITON_INTERPRET=1 is not supported")
        self._check_target(b)
        module = _materialize_module(b)
        grid_py = () if b.grid_py is None else (b.grid_py, b.hidden)
        address = module._init(
            self, self._new_compile_state(b), _Interner(), *grid_py, *b.args
        )
        return address, option, b.doc

    def _check_target(self, b: _BuildOptions) -> None:
        """Runs the build's checks that need the target, after every GPU-free one."""
        assert b.tuned is None, "make_launcher gives a chain a TunedLauncher"

    def _new_compile_state(self, b: _BuildOptions) -> object:
        """Returns the Python state `_init` stores and `_compile_miss` reads back."""
        del b
        return _CompileState()
```

Replace `_compile_miss`:

```python
    def _compile_miss(self, *call: Any) -> Any:
        """Returns the kernel record for a C cache miss, which C then launches.

        Called as `(keyblob, nparams, device, *args)`, the dynamic values
        preceding the public kernel arguments.
        """
        module = self._C_module
        assert module is not None, "compile misses require an initialized launcher"
        state = cast(_CompileState, self._C_compile_state)
        # Read per miss so override_compile reaches already built launchers.
        return getattr(module, "_intj_compile")(state.seen, *call)
```

- [ ] **Step 5: Two launcher types over the stub**

Replace `_stub_loaded`, `_stub_and_type`, `_stub` and `_new_launcher`:

```python
_stub_loaded: tuple[types.ModuleType, type[Launcher], type[Launcher]] | None = None
_STUB_LOCK = threading.Lock()


def _stub_and_types() -> tuple[types.ModuleType, type[Launcher], type[Launcher]]:
    """Returns the process's one `_intj_lazy` and the launcher types over it.

    `Launcher` and `TunedLauncher`, each mixed with `_intj_lazy.Launcher`.
    Loaded beside the modules intj renders, once, under a lock: a second load
    has its own `_intj_lazy.Launcher`, and `new_launcher` refuses a type over
    another load's.
    """
    global _stub_loaded
    loaded = _stub_loaded
    if loaded is None:
        with _STUB_LOCK:
            loaded = _stub_loaded
            if loaded is None:
                _imports.ensure_triton_imported()
                from .tuning import TunedLauncher  # not at top: tuning imports this

                stub = lazy.load_stub(
                    Path(_imports.knobs.cache.get_triton_dir("intj")) / "lazy",
                    compiler_path("c"),
                )
                plain, tuned = (
                    cast(
                        "type[Launcher]",
                        type(
                            base.__name__,
                            (base, stub.Launcher),
                            {"__slots__": (), "__module__": base.__module__},
                        ),
                    )
                    for base in (Launcher, TunedLauncher)
                )
                loaded = _stub_loaded = (stub, plain, tuned)
    return loaded


def _stub() -> types.ModuleType:
    return _stub_and_types()[0]
```

```python
def _new_launcher(options: MakeLauncherOption) -> LaunchFunction:
    stub, plain, tuned = _stub_and_types()
    # `delayed` cannot replace jit_func, so whether it is a chain is fixed now
    chain = type(options.jit_func) in (_imports.Autotuner, _imports.Heuristics)
    return stub.new_launcher(tuned if chain else plain, options)
```

- [ ] **Step 6: `TunedLauncher`** — at the end of `intj/tuning.py`. Change the imports: `from typing import TYPE_CHECKING, Any, cast`; `from .launcher import Launcher, UnsupportedKernel`; under `if TYPE_CHECKING:` add `from .launcher import _BuildOptions  # pyright: ignore[reportPrivateUsage]  # launcher internals`.

```python
class TunedLauncher(Launcher):
    """The launcher of an autotune/heuristics chain: `k.__self__` for such a `k`.

    `make_launcher` picks it from the kernel's type.  Its build also checks
    every config against the target, and a miss tunes through a private copy
    of the chain, which launches the winner itself; the rest is `Launcher`'s.
    """

    __slots__ = ()

    def _check_target(self, b: _BuildOptions) -> None:
        assert b.tuned is not None and b.knob_values is not None, (
            "plan_tuning refuses no_gpu"
        )
        check_tuned_configs(
            b.tuned.plan, b.params, dict(b.compile_options), b.knob_values
        )

    def _new_compile_state(self, b: _BuildOptions) -> TunedCompileState:
        assert (
            b.tuned is not None
            and b.tuned.render is not None
            and b.knob_values is not None
        )
        return TunedCompileState(
            b.tuned.plan,
            b.params,
            dict(b.compile_options),
            tuned_grid(b.grid_py, b.grid_cpp_fn, b.grid_code),
            b.tuned.render,
            b.knob_values,
            b.dynamic,
        )

    def _compile_miss(self, *call: Any) -> Any:
        """Tunes for a C cache miss and returns the winner's record, launched."""
        return cast(TunedCompileState, self._C_compile_state).run(*call)
```

`grep -n "_stub_and_type\b\|compile_state.tuned\|state.tuned" intj tests benchmarks -r` must print nothing.

- [ ] **Step 7: AGENTS.md** — in the `_intj_lazy` paragraph replace `(\`intj.launcher.Launcher\`, a \`__slots__ = ()\` subclass of \`_intj_lazy.Launcher\`, one fixed size)` with:

```
(`intj.launcher.Launcher`, a `__slots__ = ()` subclass of `_intj_lazy.Launcher`,
one fixed size; an autotune/heuristics chain gets `intj.tuning.TunedLauncher`,
a `Launcher` subclass `make_launcher` picks from the kernel's type, which
`delayed` cannot change)
```

…and in the next paragraph replace `the per-launcher specialization check map and, when tuned, a private \`TunedCompileState\`` with `a \`_CompileState\` (the per-launcher specialization check map) or, for a \`TunedLauncher\`, its private \`TunedCompileState\``.

- [ ] **Step 8: Run the new test, the tuned and runtime suites, pyright**

Run: `PYTHONPATH=$PWD VIRTUAL_ENV=$V uv run --active --no-sync python -m pytest tests/test_tuned.py tests/test_runtime.py -q && pyright --pythonpath $V/bin/python`
Expected: all pass, 0 errors. (`test_tuned_misses_dispatch_through_launcher_method` now patches `TunedLauncher._compile_miss` on the concrete type; it must still pass.)

- [ ] **Step 9: Full suite, then commit**

Run: `PYTHONPATH=$PWD VIRTUAL_ENV=$V uv run --active --no-sync python -m pytest tests -q` — expected 1101 passed, 15 skipped.

```bash
pre-commit run --files intj/launcher.py intj/tuning.py tests/test_tuned.py AGENTS.md
git add intj/launcher.py intj/tuning.py tests/test_tuned.py AGENTS.md
git commit -m "tuning: TunedLauncher overrides Launcher's build and miss hooks

make_launcher picks it from the kernel's type; Launcher no longer branches
on a tuned plan.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: One rendered miss interface

**Files:**
- Modify: `intj/render/entry.c.jinja` (pack, walk, fill, call, spec_key, key_chain, ops)
- Modify: `intj/render/context.py` (`computed_names` field)
- Modify: `intj/runtime/intj_lazy.c:181-193`, `intj/runtime/intj_lazy.h` (`key_chain` comment)
- Modify: `intj/launcher.py` (`Launcher._compile_miss`, `Launcher.key_chain` docstring)
- Modify: `AGENTS.md` (miss paragraph)
- Test: `tests/test_runtime.py`, `tests/test_launcher.py`, `tests/test_return_compiled.py`, `tests/test_tuned.py` (docstring)

**Interfaces:**
- Consumes: `TunedLauncher._compile_miss(*call)` → `TunedCompileState.run(keyblob, nparams, device, stream, controls, *args)` (unchanged).
- Produces:
  - C → Python, every module: `launcher._compile_miss(keyblob, nparams, device, stream, controls, *dynamic, *public)`; `controls` is `(gx, gy, gz)`, the `grid_cpp` extras tuple, or None for `grid_py`.
  - Python → C, every module: `(function, block_dim, shared, nparams[, compiled[, deps, computed[, meta]]])`; `compiled` required under `return_compiled`, `deps`/`computed` when the render has any, `meta` when `tuned_meta`.
  - `_C_key_chain` on every launcher: `((key bytes, ...), found)`.
- One new `RenderContext` field, `computed_names: tuple[str, ...]` (empty when ordinary): `intj_computed_names` is now rendered for every module, and `tuning` is None for an ordinary one. Everything else branches on existing `tuned`, `tuned_meta`, `grid_eval`, `nwords`, `nlevels`, `levels`.

- [ ] **Step 1: Write the failing tests**

`tests/test_runtime.py` — replace `test_key_chain_is_for_tuned_launchers_only` with:

```python
def test_key_chain_of_an_ordinary_launcher_is_its_spec_key(launch):
    """One level: the chain is the spec key, found once that key launched."""
    h = launch.__self__
    public = _args()[len(_CALL) :]
    launch(*_CALL, *public)
    keys, found = h._C_key_chain(_CALL[0], *public)
    assert keys == (h._C_spec_key(_CALL[0], *public)[0],)
    assert found is True


def test_ordinary_miss_gets_stream_and_grid_controls(built, stub_module):
    """Every module misses through one call: stream and grid controls included."""
    module, _, _ = built
    calls = []

    def compile_miss(self, keyblob, nparams, device, stream, controls, *args):
        calls.append((device, stream, controls, len(args)))
        return launcher.Launcher._compile_miss(
            self, keyblob, nparams, device, stream, controls, *args
        )

    launch = _new_launcher(
        stub_module,
        None,
        lambda h: _built(module._init(h, launcher._CompileState(), _interner())),
        compile_miss=compile_miss,
    )
    args = _args()
    assert launch(*args) is None
    assert calls == [(_CALL[0], _CALL[1], (1, 1, 1), len(args) - len(_CALL))]
```

`tests/test_launcher.py` — after `test_bind_device_handles_share_a_keyless_module` (the test ending `assert calls == [(b"", 0), (b"", 255), ...]`; find it with `grep -n "2\*\*31 - 1)\]" tests/test_launcher.py`):

```python
def test_key_chain_of_a_keyless_launcher_is_one_empty_key(device_kernel):
    handle = make_launcher(
        device_kernel,
        no_gpu=True,
        extra_annotation={"x": Argument(type=tl.int32, specialize=NEVER)},
        torch_access_mode=TorchAccessMode.INTERPRETER,
    ).__self__.bind_device(0)
    assert handle.__self__.key_chain(0, 7) == ((b"",), False)
    assert handle(0, 1, 7) is None
    assert handle.__self__.key_chain(0, 7) == ((b"",), True)
```

`tests/test_return_compiled.py` — append:

```python
def test_return_compiled_record_carries_its_kernel(monkeypatch):
    import intj.launcher as launcher_module

    monkeypatch.setattr(launcher_module, "_LOADED", {})
    launch = make_launcher(store_value, return_compiled=True)
    override_compile(module_of(launch), lambda _key, nparams, *_: (0, 1, 0, nparams))
    x = torch.zeros(1, device="cuda", dtype=torch.int32)
    device = torch.cuda.current_device()
    stream = torch.cuda.current_stream().cuda_stream
    with pytest.raises(TypeError, match="CompiledKernel"):
        launch(device, stream, (1,), x, 3)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `PYTHONPATH=$PWD VIRTUAL_ENV=$V uv run --active --no-sync python -m pytest tests/test_runtime.py tests/test_launcher.py tests/test_return_compiled.py -q -k "key_chain_of or stream_and_grid_controls or carries_its_kernel"`
Expected: 4 FAIL — `TypeError: intj: key_chain is for tuned launchers only` (×2), the controls assertion, and `function takes exactly 5 arguments` not matching `CompiledKernel`.

- [ ] **Step 3: Record a pre-change benchmark** (on Task 3's commit, before editing C)

```bash
PYTHONPATH=$PWD taskset -c 0 $V/bin/python benchmarks/bench_launch.py --no-gpu --iters 100000 --batches 9 > /tmp/bench-before-host.txt
PYTHONPATH=$PWD taskset -c 0 $V/bin/python benchmarks/bench_launch.py --readme --iters 20000 --batches 7 > /tmp/bench-before-readme.txt
PYTHONPATH=$PWD taskset -c 0 $V/bin/python benchmarks/bench_launch.py --tuned --iters 20000 --batches 7 > /tmp/bench-before-tuned.txt
```

- [ ] **Step 4: Template — pack and spec_key always take `comp`**

In `intj_pack`'s signature, `uint32_t device, intj_bound_launcher *bound{{ ', int64_t *comp' if tuned }}) {` becomes `uint32_t device, intj_bound_launcher *bound, int64_t *comp) {`. At its end, `{% elif tuned %}\n  (void)comp;` becomes `{% else %}\n  (void)comp;`.

In `intj_spec_key`, drop the `{% if tuned %}`/`{% endif %}` around `int64_t comp[{{ comp_len }}];` and change `bound{{ ', comp' if tuned }}) != 0)` to `bound, comp) != 0)`.

- [ ] **Step 5: Template — one walk, one fill**

(a) Delete the ordinary fill: from `{% if not tuned %}\n/* Cold path: Python assembles the final canonical ASTSource` through the `  return 0;\n}\n{% endif %}` that closes it.

(b) Remove the `{% if tuned %}` line above `/* The final record with the read lock held`, and its matching `{% endif %}` — the second of the two consecutive `{% endif %}` lines just before `{% if grid_cpp_source %}\n{{ grid_cpp_source }}`.

(c) Replace the walk's comment, signature and level-0 lookup:

```c
/* The final record with the read lock held, or NULL with it released:
 * *missed is then the first level without a record, or -1 if a heuristic
 * raised.  An ordinary launcher has one level, its spec key's.  Non-final
 * records never move, so each is read after its lookup's unlock, and every
 * heuristic runs unlocked.  `dep` receives every walked level's dependent
 * values, `comp` every level's computed keys. */
static INTJ_ALWAYS_INLINE intj_final *intj_walk(intj_state *st, intj_bound_launcher *bound,
                                                PyObject *const *kargs, const uint64_t *key,
                                                int64_t *comp, int64_t *dep, int *missed) {
  (void)st; (void)kargs; (void)key; (void)comp; (void)dep;
{% for l in levels %}
{% if l.n == 0 %}
  INTJ_RDLOCK(&bound->lock);
{% if nwords %}
  {{ rec_type(l) }} *s0 = intj_cache_lookup(INTJ_BOUND_CACHE_OF(bound), key);
{% else %}
  intj_final *s0 = (intj_final *)bound->fixed_kernel;
{% endif %}
{% else %}
```

(the rest of the walk is unchanged; `nwords == 0` only happens with one level, whose record type is `intj_final`).

(d) Replace `intj_tuned_fill` — from `/* Miss: Python runs the private chain` through its closing `}` (the one after `Py_RETURN_NONE;\n{% endif %}`) — with:

```c
/* Miss: Python returns the record -- a tuned launcher's after tuning and
 * launching it -- and every missing level stores it; a present one must
 * agree.  No lock is held while Python runs; another thread may fill this
 * key meanwhile, and the put rechecks.  Returns the call's result, or NULL.
 * An ordinary launcher's caller then launches the stored record. */
static PyObject *intj_fill(intj_state *st, intj_bound_launcher *bound,
                           const uint64_t *key,
                           int np, int64_t device_ordinal, uint64_t stream,
                           uint32_t gx, uint32_t gy, uint32_t gz,
                           PyObject *const *args, int64_t *comp, int64_t *dep,
                           PyObject *launcher) {
  (void)comp; (void)dep;
{% if grid_cpp_source %}
  PyObject *controls = PyTuple_New({{ grid_count }});
  if (!controls)
    return NULL;
  for (int i = 0; i < {{ grid_count }}; i++)
    PyTuple_SET_ITEM(controls, i, Py_NewRef(args[i]));
{% elif grid_py_mode %}
  PyObject *controls = Py_NewRef(Py_None);
{% else %}
  PyObject *controls = Py_BuildValue("(III)", gx, gy, gz);
  if (!controls)
    return NULL;
{% endif %}
  (void)gx; (void)gy; (void)gz;
  PyObject *blob = PyBytes_FromStringAndSize((const char *)key, INTJ_KEY_BYTES);
  PyObject *nparams_obj = blob ? PyLong_FromLong(np) : NULL;
  PyObject *device_obj = nparams_obj ? PyLong_FromLongLong(device_ordinal) : NULL;
  PyObject *stream_obj = device_obj ? PyLong_FromUnsignedLongLong(stream) : NULL;
  if (!stream_obj) {
    Py_XDECREF(blob); Py_XDECREF(nparams_obj); Py_XDECREF(device_obj);
    Py_DECREF(controls);
    return NULL;
  }
  /* after the controls: the dynamic values, then the public kernel arguments */
  PyObject *cb_args[6 + INTJ_NDYN + INTJ_NPARAMS];
  cb_args[0] = launcher;
  cb_args[1] = blob; cb_args[2] = nparams_obj; cb_args[3] = device_obj;
  cb_args[4] = stream_obj; cb_args[5] = controls;
  for (int i = 0; i < INTJ_NDYN + INTJ_NPARAMS; i++)
    cb_args[6 + i] = args[{{ grid_count }} + i];
  PyObject *res = PyObject_VectorcallMethod(st->compile_name, cb_args,
                                            6 + INTJ_NDYN + INTJ_NPARAMS, NULL);
  Py_DECREF(blob); Py_DECREF(nparams_obj);
  Py_DECREF(device_obj); Py_DECREF(stream_obj); Py_DECREF(controls);
  if (!res)
    return NULL;
  /* (function, block_dim, shared, nparams[, compiled[, deps, computed[, meta]]]) */
  unsigned long long function;
  unsigned int block_dim, shared, nparams;
  PyObject *compiled = NULL, *deps = NULL, *computed = NULL, *meta = NULL;
  if (!PyArg_ParseTuple(res, "KIII|OOOO", &function, &block_dim, &shared, &nparams,
                        &compiled, &deps, &computed, &meta)) {
    Py_DECREF(res);
    return NULL;
  }
  if (INTJ_UNLIKELY(block_dim == 0)) { /* marks an empty map slot */
    Py_DECREF(res);
    PyErr_SetString(PyExc_ValueError, "intj: compile callback returned block_dim 0");
    return NULL;
  }
{% if return_compiled %}
  if (!compiled) {
    Py_DECREF(res);
    PyErr_SetString(PyExc_TypeError,
                    "intj: with return_compiled, a record's fifth item is its CompiledKernel");
    return NULL;
  }
{% endif %}
  intj_final fresh;
  memset(&fresh, 0, sizeof(fresh));
  if (deps ? !PyTuple_CheckExact(deps) || PyTuple_GET_SIZE(deps) != {{ ndeps }}
           : {{ ndeps }} != 0) {
    Py_DECREF(res);
    PyErr_SetString(PyExc_RuntimeError, "intj: compile callback returned the wrong deps; this is an intj bug");
    return NULL;
  }
{% if tuned_meta %}
  if (!meta || !PyTuple_CheckExact(meta) || PyTuple_GET_SIZE(meta) != {{ nmeta }}) {
    Py_DECREF(res);
    PyErr_SetString(PyExc_RuntimeError, "intj: compile callback returned the wrong meta; this is an intj bug");
    return NULL;
  }
{% endif %}
  /* Every level's dependent values, in the slots the walk would fill. */
  for (int i = 0; i < {{ ndeps }}; i++)
    if (intj_dep_value(PyTuple_GET_ITEM(deps, i), &dep[2 * i], &dep[2 * i + 1]) != 0) {
      Py_DECREF(res);
      return NULL;
    }
{% if levels[-1].deps %}
  memcpy(fresh.dep, dep + {{ levels[-1].dep_at }}, sizeof(fresh.dep));
{% endif %}
  /* Keys past level 0, from the values this run recorded; no lock is held. */
{% for l in later_levels %}
  if (intj_tuned_level_{{ l.n }}(st, args + {{ grid_count }} + INTJ_NDYN, dep, comp) != 0) {
    Py_DECREF(res);
    return NULL;
  }
{% endfor %}
  /* The lowering must agree with Triton's own evaluation, kind included. */
  if (computed ? !PyTuple_CheckExact(computed) || PyTuple_GET_SIZE(computed) != {{ ncomp }}
               : {{ ncomp }} != 0) {
    Py_DECREF(res);
    PyErr_SetString(PyExc_RuntimeError, "intj: compile callback returned the wrong computed; this is an intj bug");
    return NULL;
  }
  for (int i = 0; i < {{ ncomp }}; i++) {
    PyObject *item = PyTuple_GET_ITEM(computed, i);
    int64_t value = 0;
    int same = item == Py_True || item == Py_False
                   ? comp[2 * i + 1] == 1 && comp[2 * i] == (item == Py_True)
                   : comp[2 * i + 1] == 0 && PyLong_CheckExact(item) &&
                         intj_as_i64(item, &value) == 0 && value == comp[2 * i];
    if (!same) {
      PyErr_Format(PyExc_RuntimeError,
                   "intj: heuristic '%s' lowered to %lld but Triton computed %R; this is an intj bug",
                   intj_computed_names[i], (long long)comp[2 * i], item);
      Py_DECREF(res);
      return NULL;
    }
  }
  fresh.k.function = (void *)(uintptr_t)function;
  fresh.k.block_dim = block_dim;
  fresh.k.shared = shared;
  fresh.k.nparams = nparams;
{% if return_compiled %}
  fresh.k.compiled = Py_NewRef(compiled);
{% endif %}
{% if tuned_meta %}
  fresh.meta = Py_NewRef(meta);
{% endif %}
  Py_DECREF(res);
  int taken = 0, failed = 0, mismatch = 0;
  PyObject *out = NULL;
{% if nwords %}
  intj_cache *cache = INTJ_BOUND_CACHE_OF(bound);
{% endif %}
  INTJ_WRLOCK(&bound->lock);
{% if not nwords %}
  /* keyless: the launcher's one record, on the heap */
  intj_final *stored = (intj_final *)bound->fixed_kernel;
  if (!stored) {
    stored = (intj_final *)PyMem_RawMalloc(sizeof(intj_final));
    if (stored) {
      *stored = fresh;
      bound->fixed_kernel = stored;
      taken = 1;
    } else {
      failed = 1;
    }
  }
{% elif nlevels == 1 %}
  intj_final *stored = intj_cache_lookup(cache, key);
  if (stored) {
{% if tuned %}
    mismatch = memcmp(stored->dep, fresh.dep, sizeof(fresh.dep)) != 0;
{% endif %}
  } else {
    stored = intj_cache_put(cache, key, &fresh);
    taken = stored != NULL;
    failed = stored == NULL;
  }
{% else %}
```

…followed, unchanged, by the multi-level block of `intj_tuned_fill` (from `  {{ rec_type(levels[0]) }} *s0 = intj_cache_lookup(cache, key);` through `  intj_final *stored = s{{ levels[-1].n }};`), then:

```c
{% endif %}
{% if return_compiled %}
  if (stored && !mismatch)
    out = Py_NewRef(stored->k.compiled);
{% endif %}
  (void)stored;
  INTJ_RWUNLOCK(&bound->lock);
  if (!taken)
    intj_final_release(&fresh);
  if (failed)
    return PyErr_NoMemory();
  if (mismatch) {
    PyErr_SetString(PyExc_RuntimeError,
                    "intj: two tuning runs recorded different values for one key; this is an intj bug");
    return NULL;
  }
{% if return_compiled %}
  return out;
{% else %}
  (void)out;
  Py_RETURN_NONE;
{% endif %}
}
```

(e) `{% if grid_py_mode %}\nstatic PyObject *intj_tuned_grid_py_launch` becomes `{% if tuned_meta %}\nstatic PyObject *intj_tuned_grid_py_launch` (an ordinary `grid_py` launcher has no `meta`).

(f) In `intj_computed_names`, `{% for name in tuning.computed_names %}` becomes `{% for name in computed_names %}`. In `intj/render/context.py`, add the field after `computed_names_len: int`:

```python
    computed_names: tuple[str, ...]  # every level's computed keys, for errors
```

and in `from_key`, after `computed_names_len=ncomp + 1,`:

```python
            computed_names=tuning.computed_names if tuning is not None else (),
```

- [ ] **Step 6: Template — `intj_call` walks for both**

Replace from `{% if tuned %}\n  int64_t comp[{{ comp_len }}];` (in `intj_call`) through the `{% endif %}` that closes the ordinary `{% else %}` lookup branch (the line before `{% if not tuned_meta %}`) with:

```c
  int64_t comp[{{ comp_len }}];
  if (intj_pack(st, dyn, kargs, key, vals, &np, (uint32_t)device_ordinal, bound, comp) != 0)
    return NULL;

  int64_t dep[{{ dep_len }}];
  int missed = 0;
  intj_final *kernel = intj_walk(st, bound, kargs, key, comp, dep, &missed);
  if (INTJ_UNLIKELY(!kernel)) {
    if (missed < 0)
      return NULL;
{% if tuned %}
    /* tuning launched it: the record is only stored */
    return intj_fill(st, bound, key, np, device_ordinal, stream,
                     gx, gy, gz, args, comp, dep, launcher);
{% else %}
    PyObject *filled = intj_fill(st, bound, key, np, device_ordinal, stream,
                                 gx, gy, gz, args, comp, dep, launcher);
    if (!filled)
      return NULL;
    Py_DECREF(filled);
    kernel = intj_walk(st, bound, kargs, key, comp, dep, &missed);
{% endif %}
  }
  (void)dep;
{% if tuned_meta %}
  /* grid_py runs Python: leave the lock with a copy of the record. */
  intj_final copy = *kernel;
  Py_XINCREF(copy.meta);
{% if return_compiled %}
  Py_XINCREF(copy.k.compiled);
{% endif %}
  INTJ_RWUNLOCK(&bound->lock);
  PyObject *result = intj_tuned_grid_py_launch(st, bound, &copy, stream, kargs, vals, np);
  Py_XDECREF(copy.meta);
{% if return_compiled %}
  Py_XDECREF(copy.k.compiled);
{% endif %}
  return result;
{% elif grid_eval == 'deferred' %}
  /* ponytail: evaluated under the read lock, so a grid error allocates while
   * holding it; a free-threaded finalizer that re-enters this launcher on a
   * miss would deadlock.  Evaluate from a copied dep array if that is seen. */
  uint32_t dims[3] = {1, 1, 1};
  if (intj_eval_grid(args, dep, dims) != 0) {
    INTJ_RWUNLOCK(&bound->lock);
    return NULL;
  }
  gx = dims[0]; gy = dims[1]; gz = dims[2];
{% endif %}
```

(`grid_eval == 'deferred'` after `tuned_meta` is exactly the tuned `grid_cpp` case.)

- [ ] **Step 7: Template, C stub — `key_chain` on every module**

In the template: drop `{% if tuned %}`/`{% endif %}` around the `intj_key_chain` forward declaration and around its definition; in `intj_ops`, `{{ 'intj_key_chain' if tuned else 'NULL' }}` becomes `intj_key_chain`; in `intj_key_chain`, wrap the `if (!bound->cache_ready) {...}` check in `{% if nwords %}`/`{% endif %}`.

In `intj/runtime/intj_lazy.c`, `intj_lazy_key_chain` becomes:

```c
static PyObject *intj_lazy_key_chain(PyObject *self, PyObject *const *args,
                                     Py_ssize_t nargs) {
  intj_launcher *h = (intj_launcher *)self;
  void *state = intj_lazy_built_state(h);
  return state ? h->ops->key_chain(state, args, nargs) : NULL;
}
```

In `intj/runtime/intj_lazy.h`, the comment `/* NULL unless tuned */` above `key_chain` becomes `/* every level's key a call walks; an ordinary launcher's one */`.

- [ ] **Step 8: Python side of the interface** (`intj/launcher.py`)

```python
    def _compile_miss(
        self,
        keyblob: bytes,
        nparams: int,
        device: int,
        stream: int,
        controls: object,
        /,
        *args: Any,
    ) -> Any:
        """Returns the kernel record for a C cache miss, which C then launches.

        Every module calls it alike; `stream` and the grid `controls` matter
        only to a `TunedLauncher`, which launches its own.  `args` are the
        dynamic values, then the public kernel arguments.
        """
        del stream, controls
        module = self._C_module
        assert module is not None, "compile misses require an initialized launcher"
        state = cast(_CompileState, self._C_compile_state)
        # Read per miss so override_compile reaches already built launchers.
        return getattr(module, "_intj_compile")(
            state.seen, keyblob, nparams, device, *args
        )
```

Positional-only, so `TunedLauncher._compile_miss(self, *call)` stays a valid override for pyright. `Launcher.key_chain`'s docstring becomes:

```python
        """Returns `((key bytes, ...), found)` for `(device, *dynamic, *public_args)`.

        One key per lookup level the call walks: an ordinary launcher's chain
        is its spec key alone.  Builds first.
        """
```

- [ ] **Step 9: Update the tests that slice the rendered C**

In `tests/test_launcher.py`, the keyless-module test that asserts `"bound->fixed_kernel" in call_body` (find with `grep -n '"bound->fixed_kernel" in call_body' tests/test_launcher.py`): after `call_body = ...`, add

```python
    walk_body = source[
        source.index("static INTJ_ALWAYS_INLINE intj_final *intj_walk") : source.index(
            "static const char *const intj_computed_names"
        )
    ]
```

and change the last assertion to `assert "intj_walk(" in call_body and "bound->fixed_kernel" in walk_body`.

In the bound-parameter source test (find with `grep -n "argument validation must precede cache lookup" tests/test_launcher.py`): add the same `walk_body` slice after `call = ...`, change `lookup`'s no-map value from `"kernel = (intj_final *)bound->fixed_kernel"` to `"s0 = (intj_final *)bound->fixed_kernel"` (keep `"intj_cache_lookup("` for the map case), and replace

```python
        assert call.index("intj_pack(") < call.index(lookup), (
            "argument validation must precede cache lookup"
        )
```

with

```python
        assert call.index("intj_pack(") < call.index("intj_walk("), (
            "argument validation must precede cache lookup"
        )
        assert lookup in walk_body
```

In `tests/test_tuned.py::test_tuned_callback_block_dim_zero_is_refused`'s docstring, `intj_tuned_fill` becomes `intj_fill`.

- [ ] **Step 10: AGENTS.md** — replace the paragraph starting `The rendered C miss path calls` with:

```
The rendered C miss path calls `Launcher._compile_miss(keyblob, nparams,
device, stream, controls, *dynamic, *args)` by name through the Python C API,
the same call from every module, and reads back `(function, block_dim, shared,
nparams[, compiled[, deps, computed[, meta]]])`. `_init` stores its Python
state in the launcher's state block, exposed by `_C_compile_state`: a
`_CompileState` (the per-launcher specialization check map) or, for a
`TunedLauncher`, its private `TunedCompileState`. `Launcher._compile_miss`
looks up the module's `_intj_compile` each time, preserving compile overrides
and the shared compiled kernel cache, and C launches the record.
`TunedLauncher._compile_miss` runs its private chain, including the final
launch; C stores the record without launching it again. Both compile through
`_KernelCompiler`. An ordinary launcher is the one-level case of the tuned
walk, so `key_chain` works on both. A cache hit skips this method.
```

- [ ] **Step 11: Run the new tests, the touched suites, pyright**

Run: `PYTHONPATH=$PWD VIRTUAL_ENV=$V uv run --active --no-sync python -m pytest tests/test_runtime.py tests/test_return_compiled.py tests/test_tuned.py tests/test_grid.py tests/test_kernel_cache.py -q && pyright --pythonpath $V/bin/python`
Expected: all pass, 0 errors.

- [ ] **Step 12: Benchmark gate** — rerun Step 3's three commands into `/tmp/bench-after-*.txt` and compare row by row:

Run: `diff <(cat /tmp/bench-before-host.txt) <(cat /tmp/bench-after-host.txt)` (and `-readme`, `-tuned`)
Expected: every ordinary and tuned hit row within ±3 % of before. If an ordinary row regresses, render one ordinary key before and after (`intj.render.context.render(key)`, keys pickled as AGENTS.md describes) and diff `intj_call`/`intj_walk`; the walk must inline to the old single lookup.

- [ ] **Step 13: Full suite, lint, commit**

Run: `PYTHONPATH=$PWD VIRTUAL_ENV=$V uv run --active --no-sync python -m pytest tests -q` — expected 1104 passed, 15 skipped (1101 + 4 new − 1 replaced).

```bash
pre-commit run --files intj/render/entry.c.jinja intj/render/context.py intj/runtime/intj_lazy.c intj/runtime/intj_lazy.h intj/launcher.py AGENTS.md tests/test_runtime.py tests/test_launcher.py tests/test_return_compiled.py tests/test_tuned.py
git add intj/render/entry.c.jinja intj/render/context.py intj/runtime/intj_lazy.c intj/runtime/intj_lazy.h intj/launcher.py AGENTS.md tests/test_runtime.py tests/test_launcher.py tests/test_return_compiled.py tests/test_tuned.py
git commit -m "render: one miss interface for ordinary and tuned modules

An ordinary launcher renders as the one-level walk: every module calls
_compile_miss with stream and grid controls, one intj_fill parses one
record shape, and key_chain is no longer tuned-only.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

# Eager/Delayed Options and In-Place Launcher Edits Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A launcher holds one `MakeLauncherOption` (with `delayed` as a field), every edit has an in-place `_foo` and a copying `foo`, and the first call is a straight pipeline resolve -> check -> render -> build -> call with nothing derived stored on the launcher.

**Architecture:** `_analyze` + `_bind` merge into `_check_options(option) -> _BuildOptions`, a return value only. The C stub (`_intj_lazy`) loses its `origin` and `plan` slots; the builder returns `(entry, resolved options, doc)` and C publishes them at the swap. In-place edits go through `_C_edit_options(edit)`, which applies `edit` under the build lock and refuses once built.

**Tech Stack:** Python 3.8+ (pyright strict on `intj/`), CPython C API (`intj/runtime/intj_lazy.c`), triton 3.8, pytest.

**Spec:** `docs/superpowers/specs/2026-10-19-eager-delayed-launcher-api-design.md`

## Global Constraints

- Python: `PY=/mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/venv/bin/python`; run tests as `PYTHONPATH=$PWD $PY -m pytest ...` from the repo root.
- Typecheck: `pyright --pythonpath $PY` must report 0 errors.
- `intj/` imports neither triton nor torch at module top; use `_imports.ensure_triton_imported()` then `_imports.<name>`.
- The C stub supports CPython 3.8 to 3.14t: no `Py_NewRef`, no `PyObject_CallOneArg`; use `Py_INCREF` and `PyObject_CallFunctionObjArgs`.
- Anything intj cannot do raises `UnsupportedKernel`; never fall back silently.
- `delayed` is in `MakeLauncherOption` but stays out of `OPTION_FIELDS` and `MakeLauncherKwargs`.
- Delayed keywords replace eager ones, whole key (no dict merging, no clash error).
- In-place edits are `_foo` (public, return None); copying edits are `foo` (return a new launcher). No other private helper is named `_` + a public method name (so no `_check`, `_build`, `_bind` helpers).
- Before each commit run `PYTHON=$PY ninja -C ci lint` (pre-commit: formatting and lint) and commit what it reformats; if pre-commit is not installed, say so in your report instead of skipping silently.
- No new docstrings on private functions under ~10 lines. Commit messages: `area: lowercase summary`, ending with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. A `delayed` that calls its own launcher's `check()` or call: must raise "while it was being built", never recurse or deadlock (Task 3, `test_calling_itself_or_a_sibling_is_refused`).
2. Two threads calling `_bind` on one unbuilt launcher: both edits land (Task 4, `test_concurrent_in_place_edits_both_land`).
3. `_bind` racing the first call: either lands before the build or raises; `options` always reports what the module was built from (Task 4, `test_in_place_edit_racing_the_first_call`).
4. A `chain_delayed` whose earlier function returns a non-mapping: the same `TypeError` as an unchained one, not a bare `dict` unpacking error (Task 4, `test_chain_delayed_validates_each_function`).
5. An explicit `check()` then first call on a kernel that warns: the warning fires again at the build; tests that set `simplefilter("error")` after `check()` must account for it (Task 3, class-global test).

---

## File Structure

- `intj/options.py`: `MakeLauncherOption.delayed` field.
- `intj/launcher.py`: `_BuildOptions`, `_check_options`, `_check_binding`, `_resolved`, `_delayed_keywords`, `_chained`, `Launcher` methods, `make_launcher`, `_new_launcher`; delete `_Once`, `_Analysis`, `_Binding`, `_Plan`, `_Origin`, `Delayed`, `_merge_delayed`, `_analyze`, `_bind`.
- `intj/runtime/intj_lazy.h`, `intj/runtime/intj_lazy.c`: drop `origin`/`plan`, builder tuple, `_C_edit_options`, `_C_check` -> `_dry_run`.
- Tests: `tests/test_launcher_object.py`, `tests/test_delayed.py`, `tests/test_runtime.py`, `tests/test_launcher.py`, `tests/test_grid.py`.
- Docs: `README.md`, `docs/Usage.md`, `TODO.md`, `MIGRATION_GAP.md`, `AGENTS.md`, `intj/python_intf/AGENTS.md`.

---

### Task 1: `MakeLauncherOption.delayed`

**Files:**
- Modify: `intj/options.py`
- Modify: `intj/__init__.py:5` (module docstring line)
- Test: `tests/test_launcher_object.py`

**Interfaces:**
- Produces: `MakeLauncherOption(jit_func, ..., delayed: Callable[[], Mapping[str, Any]] | None = None)`; `OPTION_FIELDS` excludes `"jit_func"` and `"delayed"`.

- [ ] **Step 1: Write the failing test** (append to `tests/test_launcher_object.py`)

```python
def test_option_holds_delayed_outside_option_fields():
    def delayed():
        return {}

    option = MakeLauncherOption(store, delayed=delayed)
    assert option.delayed is delayed
    assert MakeLauncherOption(store).delayed is None
    assert "delayed" not in OPTION_FIELDS
    assert "delayed" not in MakeLauncherKwargs.__annotations__
```

- [ ] **Step 2: Run it to verify it fails**

Run: `PYTHONPATH=$PWD $PY -m pytest tests/test_launcher_object.py::test_option_holds_delayed_outside_option_fields -q`
Expected: FAIL, `TypeError: ... unexpected keyword argument 'delayed'`.

- [ ] **Step 3: Implement**

In `intj/options.py`:
- Module docstring: `"""`MakeLauncherOption`: every `make_launcher` keyword, as one value."""`
- Class docstring first line: `"""Every `make_launcher` keyword, stored as given.` and add a paragraph after the existing ones:

```
    `delayed`, if set, is called at the first `check()` or call; its keywords
    replace these.  Once a launcher is built its `options` has `delayed=None`.
```
- Add the field last (after `ndebug: bool = True`):

```python
    delayed: Callable[[], Mapping[str, Any]] | None = None
```
- `OPTION_FIELDS`:

```python
#: the keywords `make_launcher` passes through `**kwargs` and `delayed` may return
OPTION_FIELDS: frozenset[str] = frozenset(
    f.name for f in dataclasses.fields(MakeLauncherOption)
) - {"jit_func", "delayed"}
```
- `MakeLauncherKwargs` docstring: `"""`make_launcher`'s `**kwargs`: `MakeLauncherOption`'s fields but `jit_func` and `delayed`."""`

In `intj/__init__.py:5`, change `MakeLauncherOption: every make_launcher keyword but delayed, as one value.` to `MakeLauncherOption: every make_launcher keyword, as one value.`

- [ ] **Step 4: Run the file's tests**

Run: `PYTHONPATH=$PWD $PY -m pytest tests/test_launcher_object.py -q`
Expected: all PASS (`test_kwargs_typeddict_matches_the_dataclass` still holds).

- [ ] **Step 5: Commit**

```bash
git add intj/options.py intj/__init__.py tests/test_launcher_object.py
git commit -m "options: MakeLauncherOption gains a delayed field, outside OPTION_FIELDS"
```

---

### Task 2: `_BuildOptions` and `_check_options` replace `_Analysis`/`_Binding`

Python only; the C stub is unchanged in this task. `_C_set_plan` now stores a `_BuildOptions` as the plan, and the analysis is no longer shared between launchers.

**Files:**
- Modify: `intj/launcher.py` (`_Analysis` 172-200, `_Binding` 203-210, `_Plan` 214-217, `_Origin.analysis` 228, `_analyze` 300-503, `_bind` 506-578, `Launcher` 596-720)
- Test: `tests/test_launcher_object.py`, `tests/test_launcher.py:1299-1340`

**Interfaces:**
- Produces:
  - `_BuildOptions` (frozen dataclass, fields below).
  - `_check_options(option: MakeLauncherOption) -> _BuildOptions`.
  - `_check_binding(option: MakeLauncherOption, params: tuple[ResolvedParam, ...], *, dynamic: tuple[str, ...], grid_code: GridCode | None) -> tuple[DeviceBinding, tuple[ResolvedParam, ...], tuple[object, ...], tuple[object, ...], bytes]`.

- [ ] **Step 1: Write the failing test** (replace `test_derivatives_share_one_analysis` in `tests/test_launcher_object.py`)

```python
def test_each_launcher_checks_its_own_options(monkeypatch):
    import intj.launcher as launcher

    calls = []
    real = launcher._check_options
    monkeypatch.setattr(
        launcher, "_check_options", lambda o: calls.append(o.bind_device) or real(o)
    )
    base = make_launcher(store, extra_annotation={"x": _POINTER}, **_HOST)
    for k in (base.__self__.bind(x=0), base.__self__.bind_device(0, x=0)):
        k.__self__.check()
    assert calls == [None, 0]
```

- [ ] **Step 2: Run it to verify it fails**

Run: `PYTHONPATH=$PWD $PY -m pytest tests/test_launcher_object.py::test_each_launcher_checks_its_own_options -q`
Expected: FAIL, `AttributeError: ... has no attribute '_check_options'`.

- [ ] **Step 3: Define `_BuildOptions`** in place of `_Analysis`, `_Binding` and `_Plan` (delete those three):

```python
@dataclasses.dataclass(frozen=True)
class _BuildOptions:
    """Everything the build needs from a checked `MakeLauncherOption`.

    `_check_options` returns it; nothing stores it beyond one build.  The
    target and live knob values are not here: reading them is a GPU query,
    made by the build.
    """

    jit_func: JitFunction  # the chain unwrapped, checked
    tuned: _Tuned | None
    params: tuple[ResolvedParam, ...]  # bound tensors' types resolved
    torch_access_mode: TorchAccessMode
    grid_code: GridCode | None  # `grid_cpp` compiled
    dynamic: tuple[str, ...]
    #: `dynamic_options` naming a tl.constexpr parameter: no slot, and only
    #: checked to be a compile option once the target is known
    declared_parameters: tuple[str, ...]
    global_values: tuple[tuple[str, str, tuple[object, ...]], ...]
    device_binding: DeviceBinding
    args: tuple[object, ...]  # `_init`'s trailing arguments: [device], *bound values
    hidden: tuple[object, ...]  # grid_py's fixed values
    doc: bytes  # the launcher's `__text_signature__`
    # the option's own values, checked and normalized
    compile_options: tuple[tuple[str, Any], ...]
    torch_access_request: TorchAccessMode | None
    kernel_cache: KernelCache
    no_gpu: bool
    verify_annotation: bool
    ndebug: bool
    return_compiled: bool
    grid_arg: int | None
    grid_py: Callable[[dict[str, object]], object] | None
    grid_cpp_fn: object | None  # the `grid_cpp` def itself
```

Delete the `analysis` field from `_Origin` (keep `_Origin` itself until Task 3).

- [ ] **Step 4: Turn `_bind` into `_check_binding`**

Rename `def _bind(a: _Analysis, option: MakeLauncherOption) -> _Binding:` to:

```python
def _check_binding(
    option: MakeLauncherOption,
    params: tuple[ResolvedParam, ...],
    *,
    dynamic: tuple[str, ...],
    grid_code: GridCode | None,
) -> tuple[DeviceBinding, tuple[ResolvedParam, ...], tuple[object, ...], tuple[object, ...], bytes]:
    """Checks `bind_device` and `bind_values` against the resolved `params`.

    Returns `(device_binding, params, args, hidden, doc)`: `params` with bound
    tensors' types resolved, `_init`'s trailing arguments, grid_py's fixed
    values, and the `__text_signature__`.
    """
```

In its body replace `a.params` -> `params`, `a.grid_arg` -> `option.grid_arg`, `a.grid_cpp` -> `grid_code`, `a.grid_py is not None` -> `option.grid_py is not None`, `a.dynamic` -> `dynamic`, and the last line `return _Binding(binding, state, args, hidden, doc)` -> `return binding, state, args, hidden, doc`.

- [ ] **Step 5: Turn `_analyze` into `_check_options`**

Rename `def _analyze(option: MakeLauncherOption) -> _Analysis:` to `def _check_options(option: MakeLauncherOption) -> _BuildOptions:`. Docstring first line: `"""Runs every GPU-free check on `option` and returns what the build needs.`; in `Raises:` add `TypeError: A malformed option, or bind_device/bind_values that do not fit the kernel.`; in `Warns:` keep both entries.

Replace its tail, from `analysis = _Analysis(` through `return analysis`, with:

```python
    classes = sorted(
        {".".join(c[1:]) for _, _, v in global_values for c in _class_globals(v)}
    )
    if classes:
        _warn(
            f"intj: {kernel.fn.__qualname__} reads class global(s) {classes}; intj "
            "keys a class by its qualified name and will not detect if it is "
            "replaced (assume_constant_globals=True)",
            ClassGlobalWarning,
        )
    device_binding, params, args, hidden, doc = _check_binding(
        option, resolved, dynamic=dynamic, grid_code=compiled_grid
    )
    return _BuildOptions(
        jit_func=kernel,
        tuned=tuned,
        params=params,
        torch_access_mode=access,
        grid_code=compiled_grid,
        dynamic=dynamic,
        declared_parameters=tuple(n for n in declared if n in kernel.arg_names),
        global_values=global_values,
        device_binding=device_binding,
        args=args,
        hidden=hidden,
        doc=doc,
        compile_options=tuple(options.items()),
        torch_access_request=torch_access_mode,
        kernel_cache=kernel_cache,
        no_gpu=no_gpu,
        verify_annotation=verify_annotation,
        ndebug=ndebug,
        return_compiled=return_compiled,
        grid_arg=grid_arg,
        grid_py=grid_py,
        grid_cpp_fn=grid_cpp,
    )
```

(`_provision(kernel_cache)` stays where it is, before this tail.)

- [ ] **Step 6: Update `Launcher`**

- TYPE_CHECKING stubs: `_C_plan` returns `_BuildOptions | None`; `_C_set_plan(self, options: MakeLauncherOption, plan: _BuildOptions, doc: bytes) -> None`.
- `_resolved` becomes options-only:

```python
    def _resolved(self) -> MakeLauncherOption:
        options = self._C_options
        return options() if isinstance(options, Delayed) else options
```
- `_prepare` body: `self._resolved()` (unchanged text; it now only runs `delayed`).
- `_check`:

```python
    def _check(self) -> None:
        # check()'s body: _C_check calls it under the build lock, as does _build
        if self._C_plan is not None:
            return
        option = self._resolved()
        build = _check_options(option)
        self._C_set_plan(option, build, build.doc)
```
- `_build`: after `self._check()`, use `b = self._C_plan` / `assert b is not None`, delete `a, b = plan.analysis, plan.binding`, and rewrite every `a.<field>` / `b.<field>` read:
  - `a.jit_func` -> `b.jit_func`; `b.params` stays; `dict(a.options)` -> `dict(b.compile_options)` (three places); `a.torch_access_mode` -> `b.torch_access_mode`; `a.torch_access_request` -> `b.torch_access_request`; `a.kernel_cache` / `a.no_gpu` / `a.verify_annotation` / `a.return_compiled` / `a.ndebug` / `a.grid_arg` / `a.grid_py` / `a.dynamic` / `a.global_values` / `a.declared_parameters` / `a.tuned` -> same name on `b`; `b.device_binding` stays.
  - `grid_cpp=a.grid_cpp` -> `grid_cpp=b.grid_code`; `elif a.grid_cpp is not None:` -> `elif b.grid_code is not None:`; `getattr(a.grid_fn, "__code__")` -> `getattr(b.grid_cpp_fn, "__code__")`; `TunedGrid("cpp", a.grid_fn, ..., a.grid_cpp.extras)` -> `TunedGrid("cpp", b.grid_cpp_fn, ..., b.grid_code.extras)`.
  - `grid_py = () if a.grid_py is None else (a.grid_py, b.hidden)` -> `grid_py = () if b.grid_py is None else (b.grid_py, b.hidden)`.

- [ ] **Step 7: Update the warnings test** (`tests/test_launcher.py`, `test_binding_warnings_are_per_parameter_at_check`)

Each of the two loop iterations makes a new launcher, which now checks on its own. Replace the comment and expected list:

```python
        # each launcher checks its own options: one warning per parameter per launcher
        assert [str(w.message) for w in caught] == (
            [
                f"intj: pointer-range assumption for bound pointer {name!r} "
                "cannot be verified from an address and will be trusted"
                for name in ("x", "p")
            ]
            * 2
            if verify
            else []
        )
```

- [ ] **Step 8: Run the tests**

Run: `PYTHONPATH=$PWD $PY -m pytest tests/test_launcher_object.py tests/test_delayed.py tests/test_grid.py tests/test_tuned.py -q`
Expected: PASS. Then `grep -n "_analyze\|_Analysis\|_Binding\|_Plan\b\|grid_fn\|\b_bind(" intj/ tests/ -r` prints nothing.

Run: `PYTHONPATH=$PWD $PY -m pytest tests -q -x` and `pyright --pythonpath $PY`
Expected: PASS; 0 errors.

- [ ] **Step 9: Commit**

```bash
git add intj/launcher.py tests/test_launcher_object.py tests/test_launcher.py
git commit -m "launcher: _check_options returns _BuildOptions, one per launcher"
```

---

### Task 3: One `MakeLauncherOption` per launcher; C stub without origin or plan

**Files:**
- Modify: `intj/runtime/intj_lazy.h:54-68`, `intj/runtime/intj_lazy.c`
- Modify: `intj/launcher.py` (`_Once`, `_Origin`, `Delayed`, `_merge_delayed`, `Launcher`, `_new_launcher`, `make_launcher`)
- Test: `tests/test_runtime.py:1070-1125`, `tests/test_delayed.py`, `tests/test_launcher_object.py`, `tests/test_grid.py` (3 sites), `tests/test_launcher.py` (signature sites, class-global test)

**Interfaces:**
- Consumes: `_check_options`, `_BuildOptions` (Task 2); `MakeLauncherOption.delayed` (Task 1).
- Produces:
  - C: `new_launcher(cls, options, builder)`; builder returns `(address: int, options, doc: bytes)`; `_C_edit_options(edit)` (METH_O); `_C_check()` calls `self._dry_run()`; members `_C_options`, `_C_module`.
  - Python: `_resolved(option) -> MakeLauncherOption`, `_delayed_keywords(fn) -> dict[str, Any]`, `_new_launcher(options: MakeLauncherOption) -> LaunchFunction`, `Launcher._copy() -> LaunchFunction`, `Launcher._dry_run() -> None`, `Launcher._build_body() -> tuple[int, MakeLauncherOption, bytes]`, `Launcher.options -> MakeLauncherOption`.

- [ ] **Step 1: Write the failing C-stub tests** (`tests/test_runtime.py`)

In `test_launchers_come_only_from_new_launcher`, drop one `None` from every `new_launcher(...)` call (4 args -> 3: `(cls, None, print)`, `(stub_module.Launcher, None, 7)`, `(Sub, None, print)`).

Replace `test_set_plan_publishes_options_and_signature_once` and `test_options_reads_race_set_plan_safely` with:

```python
def test_edit_options_applies_under_the_lock(stub_module):
    launch = stub_module.new_launcher(stub_module.Launcher, "before", print)
    h = launch.__self__
    assert (h._C_options, h._C_module, launch.__text_signature__) == (
        "before",
        None,
        None,
    )
    assert h._C_edit_options(lambda o: o + "!") is None
    assert h._C_options == "before!"
    with pytest.raises(KeyError):
        h._C_edit_options(lambda o: {}[o])  # the edit's error, options kept
    assert h._C_options == "before!"


def test_builder_must_return_entry_options_and_doc(stub_module):
    for result, error in (
        (7, TypeError),
        ((7, "o"), TypeError),
        ((7, "o", "not bytes"), TypeError),
        ((0, "o", b""), RuntimeError),  # a NULL entry
    ):
        launch = stub_module.new_launcher(
            stub_module.Launcher, "before", lambda h, r=result: r
        )
        with pytest.raises(error):
            launch()
        assert launch.__self__._C_options == "before"  # nothing published


def test_options_reads_race_edits_safely(stub_module):
    launchers = [
        stub_module.new_launcher(stub_module.Launcher, "before", print)
        for _ in range(200)
    ]
    seen, stop = set(), threading.Event()

    def read():
        while not stop.is_set():
            for k in launchers:
                seen.add(k.__self__._C_options)

    readers = [threading.Thread(target=read) for _ in range(4)]
    for t in readers:
        t.start()
    for k in launchers:
        k.__self__._C_edit_options(lambda o: "after")
    stop.set()
    for t in readers:
        t.join()
    assert seen <= {"before", "after"}
    assert all(k.__self__._C_options == "after" for k in launchers)
```

Also find the test around `tests/test_runtime.py:1120` that calls `k.__self__._C_set_plan("after", None, b"")`; read it, and replace that call with `k.__self__._C_edit_options(lambda o: "after")`, keeping what the test asserts.

- [ ] **Step 2: Run them to verify they fail**

Run: `PYTHONPATH=$PWD $PY -m pytest tests/test_runtime.py -q -k "new_launcher or edit_options or builder_must or race_edits"`
Expected: FAIL (`new_launcher(cls, options, origin, builder)` TypeError; no `_C_edit_options`).

- [ ] **Step 3: C header** (`intj/runtime/intj_lazy.h`)

Replace the four fields `options`, `origin`, `plan`, `builder` and the `doc` comment with:

```c
  PyObject *options; /* MakeLauncherOption; the resolved one once built */
  PyObject *builder; /* until built: callable(launcher) -> (entry, options, doc) */
  PyObject *module;
  PyObject *doc; /* bytes behind def.ml_doc, once built */
```

and change the critical-section comment to `/* Free-threaded builds lock the object around its member reads and writes; before 3.13 these are plain blocks. */`.

- [ ] **Step 4: C stub** (`intj/runtime/intj_lazy.c`)

1. Delete `intj_lazy_prepare` and its comment.
2. Replace `intj_lazy_build` with:

```c
/* Run the builder once: 0, or -1 with an error set.  The builder returns
 * (entry address, resolved options, doc); they are published, then the entry
 * swapped in.  A failure publishes nothing and leaves the shim, so the next
 * call retries. */
static int intj_lazy_build(intj_launcher *h) {
  if (__atomic_load_n(&h->def.ml_meth, __ATOMIC_ACQUIRE) != INTJ_SHIM)
    return 0;
  if (intj_lazy_lock(h) != 0)
    return -1;
  int rc = 0;
  PyObject *done = NULL, *old = NULL;
  if (__atomic_load_n(&h->def.ml_meth, __ATOMIC_ACQUIRE) == INTJ_SHIM) {
    if (!h->builder) {
      PyErr_SetString(PyExc_RuntimeError, "intj: launcher has been cleared");
      rc = -1;
    } else {
      PyObject *result =
          PyObject_CallFunctionObjArgs(h->builder, (PyObject *)h, NULL);
      void *address = NULL;
      if (result &&
          (!PyTuple_CheckExact(result) || PyTuple_GET_SIZE(result) != 3 ||
           !PyBytes_CheckExact(PyTuple_GET_ITEM(result, 2))))
        PyErr_SetString(PyExc_TypeError,
                        "intj: a launcher builder returns (entry, options, "
                        "doc: bytes)");
      else if (result)
        address = PyLong_AsVoidPtr(PyTuple_GET_ITEM(result, 0));
      if (!address) {
        if (!PyErr_Occurred())
          PyErr_SetString(PyExc_RuntimeError,
                          "intj: launcher builder returned no entry");
        rc = -1;
      } else {
        PyObject *options = PyTuple_GET_ITEM(result, 1);
        PyObject *doc = PyTuple_GET_ITEM(result, 2);
        Py_INCREF(options);
        Py_INCREF(doc);
        Py_BEGIN_CRITICAL_SECTION(h);
        old = h->options;
        h->options = options;
        h->doc = doc; /* NULL until now: only the build sets it */
        __atomic_store_n(&h->def.ml_doc, PyBytes_AS_STRING(doc),
                         __ATOMIC_RELEASE);
        Py_END_CRITICAL_SECTION();
        PyCFunction fn;
        memcpy(&fn, &address, sizeof(fn));
        __atomic_store_n(&h->def.ml_meth, fn, __ATOMIC_RELEASE);
        done = h->builder;
        h->builder = NULL;
      }
      Py_XDECREF(result);
    }
  }
  intj_lazy_unlock(h);
  /* outside the lock: finalizers run Python */
  Py_XDECREF(done);
  Py_XDECREF(old);
  return rc;
}
```

3. Replace `intj_lazy_check` with:

```c
/* _C_check(): the subclass's `_dry_run` under the build lock, which refuses
 * a launcher checked from its own build; nothing once built. */
static PyObject *intj_lazy_check(PyObject *self, PyObject *unused) {
  (void)unused;
  intj_launcher *h = (intj_launcher *)self;
  if (__atomic_load_n(&h->def.ml_meth, __ATOMIC_ACQUIRE) != INTJ_SHIM)
    Py_RETURN_NONE;
  if (intj_lazy_lock(h) != 0)
    return NULL;
  PyObject *result;
  if (__atomic_load_n(&h->def.ml_meth, __ATOMIC_ACQUIRE) != INTJ_SHIM) {
    result = Py_None;
    Py_INCREF(result);
  } else if (!h->options) {
    PyErr_SetString(PyExc_RuntimeError, "intj: launcher has been cleared");
    result = NULL;
  } else {
    result = PyObject_CallMethod(self, "_dry_run", NULL);
  }
  intj_lazy_unlock(h);
  return result;
}
```

4. Replace `intj_lazy_set_plan` with:

```c
/* _C_edit_options(edit): options = edit(options) under the build lock, so
 * concurrent edits and a first call are ordered; refused once built. */
static PyObject *intj_lazy_edit_options(PyObject *self, PyObject *edit) {
  intj_launcher *h = (intj_launcher *)self;
  if (intj_lazy_lock(h) != 0)
    return NULL;
  PyObject *old = NULL, *result = NULL;
  if (__atomic_load_n(&h->def.ml_meth, __ATOMIC_ACQUIRE) != INTJ_SHIM) {
    PyErr_SetString(PyExc_RuntimeError,
                    "intj: a built launcher's options are fixed; the method "
                    "without the leading underscore returns an edited copy");
  } else if (!h->options) {
    PyErr_SetString(PyExc_RuntimeError, "intj: launcher has been cleared");
  } else {
    PyObject *current = h->options;
    Py_INCREF(current);
    PyObject *edited = PyObject_CallFunctionObjArgs(edit, current, NULL);
    Py_DECREF(current);
    if (edited) {
      Py_BEGIN_CRITICAL_SECTION(self);
      old = h->options;
      h->options = edited;
      Py_END_CRITICAL_SECTION();
      result = Py_None;
      Py_INCREF(result);
    }
  }
  intj_lazy_unlock(h);
  Py_XDECREF(old); /* outside the lock: finalizers run Python */
  return result;
}
```

5. `intj_lazy_traverse` / `intj_lazy_clear`: delete the `origin` and `plan` lines.
6. Getset table: delete `_C_origin` and `_C_plan`. Method table: replace the `_C_set_plan` entry with `{"_C_edit_options", intj_lazy_edit_options, METH_O, NULL},`. Update the table comment to list `options`, `check()`, `build()`, `spec_key()`, `key_chain()` and the `_foo` edits.
7. `new_launcher`: `nargs == 3`, `PyCallable_Check(args[2])`, error text and comment `new_launcher(cls, options, builder)`, store `h->options = args[1]` and `h->builder = args[2]` (with `Py_INCREF`), delete the `origin` lines, comment on `ml_doc`: `/* the build sets the signature */`. The module method doc string: `"new_launcher(cls, options, builder) -> launcher"`.

- [ ] **Step 5: Run the C-stub tests**

Run: `PYTHONPATH=$PWD $PY -m pytest tests/test_runtime.py -q -k "new_launcher or edit_options or builder_must or race_edits"`
Expected: PASS (the stub is rebuilt automatically: its digest covers `intj_lazy.c`/`.h`). The rest of the suite fails until Step 7.

- [ ] **Step 6: Write the failing Python-side tests**

`tests/test_delayed.py`: remove `Delayed` from the `intj.launcher` import, then:

```python
def test_each_launcher_resolves_its_own_delayed():
    delayed, calls = _counting(_HOST)
    base = make_launcher(store, extra_annotation={"x": _POINTER}, delayed=delayed)
    assert calls == [] and base.__self__.options.delayed is delayed
    assert base.__self__.options.jit_func is store  # known before any call
    children = [base.__self__.bind(x=0), base.__self__.bind_device(0, x=0)]
    assert children[0](0, 0, 1, 4) is None and children[1](0, 1, 4) is None
    assert calls == [1, 1]
    options = children[0].__self__.options
    assert options.delayed is None  # resolved by the build
    assert options.no_gpu is True and options.bind_values == {"x": 0}
```

(replaces `test_runs_once_on_first_use_across_bind_children`.)

In `test_conflict_and_bad_results`, replace the first `pytest.raises(UnsupportedKernel, ...)` block with:

```python
    # delayed replaces an eager keyword, whole key
    launch = make_launcher(store, verify_annotation=True, delayed=lambda: {**_HOST, "verify_annotation": False})
    assert launch(0, 0, 1, 0, 4) is None
    assert launch.__self__.options.verify_annotation is False
```

Rename it `test_override_and_bad_results`. The unknown-keyword loop (including `"delayed"`) and the other `TypeError` cases stay.

In `test_concurrent_first_calls_run_it_once`, change the final assertion to `assert errors == [] and calls == [1] * (8 if siblings else 1)` and rename it `test_concurrent_first_calls_run_it_once_per_launcher`.

In `test_calling_itself_or_a_sibling_is_refused`, the sibling's `check()` now runs the same `delayed` nested, which is refused three more times: change `assert len(seen) == 3 and ...` to `assert len(seen) == 5 and ...` and add a comment `# k() and k.check() refused; sibling.check() runs delayed again, where all three are refused`.

`tests/test_launcher_object.py`:
- `test_bind_on_a_built_launcher_returns_an_unbuilt_one`: `assert child.__self__._C_module is None` (drop `_C_plan`).
- Add:

```python
def test_check_is_a_dry_run():
    delayed_calls = []

    def delayed():
        delayed_calls.append(1)
        return _HOST

    launch = make_launcher(store, delayed=delayed)
    launch.__self__.check()
    launch.__self__.check()
    assert delayed_calls == [1, 1]
    assert launch.__self__.options.delayed is delayed  # nothing stored
    assert launch.__text_signature__ is None
    assert launch(0, 0, 1, 0, 4) is None and delayed_calls == [1, 1, 1]
    assert launch.__self__.options.delayed is None
    assert launch.__text_signature__ is not None
    launch.__self__.check()  # built: nothing runs
    assert delayed_calls == [1, 1, 1]
```

Signature tests: in `tests/test_grid.py` (3 sites) and `tests/test_launcher.py` (5 sites: the `check()` lines just before `inspect.signature(...)` near lines 316, 409, 422, 842, 934), change `<x>.__self__.check()  # the signature is set by check()` to `<x>.__self__.build()  # the build sets the signature` (the one at `test_launcher.py:316` has no comment; add it).

`tests/test_launcher.py`, `test_class_global_is_keyed_by_name_and_warns_once_per_build`: the launch after `check()` builds and warns again. Change the docstring's last line to `with one ClassGlobalWarning per check and per build, and none on a launch.` and the launch block to:

```python
    o = torch.zeros(1, device="cuda", dtype=torch.int32)
    with pytest.warns(ClassGlobalWarning):
        launch(0, 0, 1, o)  # the build checks again, and the miss
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        launch(0, 0, 1, o)  # a hit
```

- [ ] **Step 7: Rework `launcher.py`**

Delete `_Once` (and `T = TypeVar("T")`, `Generic`, `TypeVar` from the imports if `grep -n "\bT\b\|Generic\|TypeVar" intj/launcher.py` shows no other use), `_Origin`, `Delayed`, `_merge_delayed`.

Add, where `_merge_delayed` was:

```python
def _delayed_keywords(delayed: Callable[[], Mapping[str, Any]]) -> dict[str, Any]:
    values = delayed()
    if not isinstance(values, Mapping):
        raise TypeError(
            "intj: delayed must return a mapping of make_launcher keywords, "
            f"got {type(values).__name__}"
        )
    values = cast("Mapping[object, object]", values)
    unknown = sorted(
        repr(k) for k in values if type(k) is not str or k not in OPTION_FIELDS
    )
    if unknown:
        raise TypeError(
            "intj: delayed returned unknown make_launcher keyword(s) "
            f"[{', '.join(unknown)}]"
        )
    return cast("dict[str, Any]", dict(values))


def _resolved(option: MakeLauncherOption) -> MakeLauncherOption:
    """Returns `option` with `delayed` called and its keywords replacing the others."""
    if option.delayed is None:
        return option
    return dataclasses.replace(
        option, **_delayed_keywords(option.delayed), delayed=None
    )
```

At the top of `_check_options`'s body add `assert option.delayed is None, "resolve delayed first"`.

`Launcher`:
- TYPE_CHECKING block becomes:

```python
        @property
        def _C_options(self) -> MakeLauncherOption: ...
        @property
        def _C_module(self) -> types.ModuleType | None: ...
        def _C_check(self) -> None: ...
        def _C_build(self) -> types.ModuleType: ...
        def _C_spec_key(self, *args: object) -> tuple[bytes, int]: ...
        def _C_key_chain(self, *args: object) -> tuple[tuple[bytes, ...], bool]: ...
        def _C_edit_options(
            self, edit: Callable[[MakeLauncherOption], MakeLauncherOption], /
        ) -> None: ...
```
- `options`:

```python
    @property
    def options(self) -> MakeLauncherOption:
        """The launcher's options; once built, the resolved ones (`delayed` None)."""
        return self._C_options
```
- `check`:

```python
    def check(self) -> None:
        """Runs every check the first call would, without building or storing anything.

        Calls `options.delayed` if set; the first call calls it again.  Does
        nothing once built.
        """
        self._C_check()
```
- `bind`, `bind_device`, `unbind`, `unbind_device` keep their docstrings and edits but call `self._derive(...)`, which becomes:

```python
    def _derive(
        self, edit: Callable[[MakeLauncherOption], MakeLauncherOption]
    ) -> LaunchFunction:
        k = self._copy()
        k.__self__._C_edit_options(edit)
        return k

    def _copy(self) -> LaunchFunction:
        return _new_launcher(self._C_options)
```
(Task 4 replaces `_derive` with the `_foo` twins.)
- Delete `_resolved`, `_prepare`, `_check`. Add:

```python
    def _dry_run(self) -> None:
        # check()'s body: _C_check calls it under the build lock
        _check_options(_resolved(self._C_options))
```
- Rename `_build` to `_build_body` with return type `tuple[int, MakeLauncherOption, bytes]`. Docstring: replace "Checks, then queries the target" with "Resolves `delayed`, checks, then queries the target" and "The returned address is the rendered entry, for `_intj_lazy` to swap in." with "Returns the rendered entry's address, the resolved options and the `__text_signature__`, for `_intj_lazy` to publish and swap in." Body start:

```python
        option = _resolved(self._C_options)
        b = _check_options(option)
```
(replacing `self._check()` / `b = self._C_plan` / `assert b is not None`), and the last line:

```python
        address = module._init(self, callback, _Interner(), *grid_py, *b.args)
        return address, option, b.doc
```
- `_new_launcher`:

```python
def _new_launcher(options: MakeLauncherOption) -> LaunchFunction:
    stub, cls = _stub_and_type()
    return stub.new_launcher(cls, options, cls._build_body)  # pyright: ignore[reportPrivateUsage]  # the builder
```
- `make_launcher` tail:

```python
    option = MakeLauncherOption(jit_func, delayed=delayed, **kwargs)
    return _new_launcher(option)
```
- `make_launcher` docstring: `delayed:` arg becomes

```
        delayed: A function of no arguments returning more of these
            keywords, as a mapping; they replace the ones passed here. Each
            launcher calls it on its first call (and on every `check()`
            before that); its exception propagates and is retried on the
            next call.
```
and "Decoration only records a `MakeLauncherOption` (or, with `delayed`, how to make one)" becomes "Decoration only records a `MakeLauncherOption`". In `Returns:` list the methods: `(`bind`, `bind_device`, `unbind`, `unbind_device`, `check`, ...)` unchanged for now.

- [ ] **Step 8: Run the tests**

Run: `PYTHONPATH=$PWD $PY -m pytest tests/test_delayed.py tests/test_launcher_object.py tests/test_runtime.py tests/test_grid.py -q`
Expected: PASS.

Run: `grep -rn "Delayed\|_Origin\|_Once\|_C_plan\|_C_set_plan\|_C_origin\|_prepare\b" intj tests benchmarks`
Expected: no output.

Run: `PYTHONPATH=$PWD $PY -m pytest tests -q` and `pyright --pythonpath $PY`
Expected: PASS; 0 errors.

- [ ] **Step 9: Commit**

```bash
git add intj/runtime/intj_lazy.h intj/runtime/intj_lazy.c intj/launcher.py tests
git commit -m "launcher: one MakeLauncherOption per launcher; the build publishes it, check() is a dry run"
```

---

### Task 4: In-place `_foo` and copying `foo` edits, delayed editors

**Files:**
- Modify: `intj/launcher.py` (`Launcher` edit methods; new `_chained`)
- Test: `tests/test_launcher_object.py`

**Interfaces:**
- Consumes: `Launcher._copy`, `Launcher._C_edit_options`, `_resolved`, `_delayed_keywords` (Task 3).
- Produces on `Launcher`: `_bind(**v)`, `bind(**v)`, `_bind_device(n, /, **v)`, `bind_device(n, /, **v)`, `_unbind()`, `unbind()`, `_unbind_device()`, `unbind_device()`, `_replace_delayed(fn, /)`, `replace_delayed(fn, /)`, `_chain_delayed(fn, /)`, `chain_delayed(fn, /)`, `_resolve_delayed()`, `resolve_delayed()`. `_foo` return None; `foo` return `LaunchFunction`.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_launcher_object.py`; add `import dataclasses`, `import threading`, `import time` at the top)

```python
def _first():
    return {"verify_annotation": True}


def _second():
    return {"ndebug": False}


_EDITS = [
    ("bind", (), {"x": 0}),
    ("bind_device", (3,), {"x": 0}),
    ("unbind", (), {}),
    ("unbind_device", (), {}),
    ("replace_delayed", (_second,), {}),
    ("replace_delayed", (None,), {}),
    ("chain_delayed", (_second,), {}),
    ("resolve_delayed", (), {}),
]


def _edit_base():
    return make_launcher(
        store,
        extra_annotation={"x": _POINTER},
        bind_values={"x": 1},
        bind_device=2,
        delayed=_first,
        **_HOST,
    )


def _comparable(option):
    # a chained delayed is a fresh closure each time: compare what it returns
    delayed = option.delayed
    return dataclasses.replace(option, delayed=None), (
        None if delayed is None else delayed()
    )


@pytest.mark.parametrize("name, args, kwargs", _EDITS)
def test_in_place_and_copying_edits_agree(name, args, kwargs):
    base = _edit_base()
    before = _option(base)
    copy = getattr(base.__self__, name)(*args, **kwargs)
    assert _option(base) == before  # the copying form leaves it alone
    in_place = _edit_base()
    assert getattr(in_place.__self__, "_" + name)(*args, **kwargs) is None
    assert _comparable(_option(copy)) == _comparable(_option(in_place))


@pytest.mark.parametrize("name, args, kwargs", _EDITS)
def test_in_place_edits_refused_once_built_allowed_after_check(name, args, kwargs):
    checked = _edit_base()
    checked.__self__.check()
    getattr(checked.__self__, "_" + name)(*args, **kwargs)  # check() fixes nothing
    built = _edit_base()
    built.__self__.build()
    if name == "resolve_delayed":
        built.__self__._resolve_delayed()  # nothing left to resolve: a no-op
        return
    with pytest.raises(RuntimeError, match="options are fixed"):
        getattr(built.__self__, "_" + name)(*args, **kwargs)
    copy = getattr(built.__self__, name)(*args, **kwargs)
    assert copy.__self__._C_module is None  # the copy is unbuilt


def test_resolve_and_chain_delayed():
    k = make_launcher(store, delayed=lambda: _HOST)
    resolved = k.__self__.resolve_delayed()
    assert _option(resolved).delayed is None and _option(resolved).no_gpu is True
    assert _option(k).delayed is not None
    chained = k.__self__.chain_delayed(lambda: {"no_gpu": False, "ndebug": False})
    assert _option(chained.__self__.resolve_delayed()).no_gpu is False  # later wins
    plain = make_launcher(store)
    assert _option(plain.__self__.chain_delayed(_first)).delayed is _first
    assert _option(k.__self__.replace_delayed(None)).delayed is None


def test_delayed_editors_refuse_non_callables():
    k = make_launcher(store, **_HOST)
    with pytest.raises(TypeError, match="callable"):
        k.__self__.replace_delayed(7)  # pyright: ignore[reportArgumentType]
    with pytest.raises(TypeError, match="callable"):
        k.__self__.chain_delayed(None)  # pyright: ignore[reportArgumentType]


def test_chain_delayed_validates_each_function():
    k = make_launcher(store, delayed=lambda: [("no_gpu", True)])  # pyright: ignore[reportArgumentType]
    chained = k.__self__.chain_delayed(lambda: _HOST)
    with pytest.raises(TypeError, match="must return a mapping"):
        chained.__self__.check()


def test_concurrent_in_place_edits_both_land():
    k = make_launcher(store, extra_annotation={"x": _POINTER}, **_HOST)
    barrier = threading.Barrier(2)

    def bind(name):
        barrier.wait()
        for i in range(200):
            k.__self__._bind(**{f"{name}{i}": i})

    threads = [threading.Thread(target=bind, args=(n,)) for n in ("a", "b")]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(_option(k).bind_values) == 400


def test_in_place_edit_racing_the_first_call():
    started = threading.Event()

    def delayed():
        started.set()
        time.sleep(0.2)  # the other thread's edit waits on the build lock
        return _HOST

    k = make_launcher(
        store, extra_annotation={"x": _POINTER}, bind_values={"x": 0}, delayed=delayed
    )
    errors = []

    def edit():
        started.wait()
        try:
            k.__self__._bind(x=8)
        except RuntimeError as error:
            errors.append(str(error))

    t = threading.Thread(target=edit)
    t.start()
    assert k(0, 0, 1, 4) is None
    t.join()
    assert len(errors) == 1 and "options are fixed" in errors[0]
    assert _option(k).bind_values == {"x": 0}  # what the module was built from
```

- [ ] **Step 2: Run them to verify they fail**

Run: `PYTHONPATH=$PWD $PY -m pytest tests/test_launcher_object.py -q`
Expected: FAIL (`AttributeError: ... '_bind'`, `'replace_delayed'`, ...).

- [ ] **Step 3: Implement**

In `intj/launcher.py`, after `_resolved`:

```python
def _chained(
    first: Callable[[], Mapping[str, Any]] | None,
    then: Callable[[], Mapping[str, Any]],
) -> Callable[[], Mapping[str, Any]]:
    if first is None:
        return then

    def chained() -> Mapping[str, Any]:
        return {**_delayed_keywords(first), **_delayed_keywords(then)}

    return chained
```

In `Launcher`, delete `_derive` and replace the four bind methods with the pairs below:

```python
    def _bind(self, /, **values: object) -> None:
        """Merges `values` into this launcher's `bind_values`; `bind` copies instead.

        Raises:
            RuntimeError: The launcher is built.
        """
        self._C_edit_options(
            lambda o: dataclasses.replace(
                o, bind_values={**(o.bind_values or {}), **values}
            )
        )

    def bind(self, /, **values: object) -> LaunchFunction:
        """Returns a new launcher with `values` merged into `bind_values`.

        Validates nothing: a missing, unknown or mistyped value fails in the
        new launcher's `check()`.  A bound tensor takes an exact
        `torch.Tensor`, `torch.nn.Parameter`, or None.
        """
        k = self._copy()
        k.__self__._bind(**values)
        return k

    def _bind_device(self, ordinal: int | None, /, **values: object) -> None:
        """Fixes this launcher to device `ordinal` in place; see `bind_device`.

        Raises:
            RuntimeError: The launcher is built.
        """
        self._C_edit_options(
            lambda o: dataclasses.replace(
                o, bind_device=ordinal, bind_values={**(o.bind_values or {}), **values}
            )
        )

    def bind_device(self, ordinal: int | None, /, **values: object) -> LaunchFunction:
        """Returns a new launcher fixed to device `ordinal`, `values` merged as for `bind`.

        It is called as `(stream, grid, ...)` and owns a separate kernel
        cache, or one kernel when no dynamic specialization key remains.
        None is `unbind_device()`.
        """
        k = self._copy()
        k.__self__._bind_device(ordinal, **values)
        return k

    def _unbind(self) -> None:
        """Drops this launcher's bound values in place; `unbind` copies instead."""
        self._C_edit_options(lambda o: dataclasses.replace(o, bind_values=None))

    def unbind(self) -> LaunchFunction:
        """Returns a new launcher with no bound values."""
        k = self._copy()
        k.__self__._unbind()
        return k

    def _unbind_device(self) -> None:
        """Makes this launcher take the device per call; `unbind_device` copies instead."""
        self._C_edit_options(lambda o: dataclasses.replace(o, bind_device=None))

    def unbind_device(self) -> LaunchFunction:
        """Returns a new launcher that takes the device per call."""
        k = self._copy()
        k.__self__._unbind_device()
        return k

    def _replace_delayed(self, fn: Callable[[], Mapping[str, Any]] | None, /) -> None:
        """Sets this launcher's `delayed` to `fn` (None drops it); `replace_delayed` copies instead.

        Raises:
            TypeError: `fn` is neither None nor callable.
            RuntimeError: The launcher is built.
        """
        if fn is not None and not callable(fn):
            raise TypeError("intj: delayed must be None or a callable taking no arguments")
        self._C_edit_options(lambda o: dataclasses.replace(o, delayed=fn))

    def replace_delayed(self, fn: Callable[[], Mapping[str, Any]] | None, /) -> LaunchFunction:
        """Returns a new launcher whose `delayed` is `fn`; None drops it."""
        k = self._copy()
        k.__self__._replace_delayed(fn)
        return k

    def _chain_delayed(self, fn: Callable[[], Mapping[str, Any]], /) -> None:
        """Chains `fn` after this launcher's `delayed`; `chain_delayed` copies instead.

        Raises:
            TypeError: `fn` is not callable.
            RuntimeError: The launcher is built.
        """
        if not callable(fn):
            raise TypeError("intj: chain_delayed takes a callable taking no arguments")
        self._C_edit_options(
            lambda o: dataclasses.replace(o, delayed=_chained(o.delayed, fn))
        )

    def chain_delayed(self, fn: Callable[[], Mapping[str, Any]], /) -> LaunchFunction:
        """Returns a new launcher whose `delayed` also runs `fn`, whose keywords win."""
        k = self._copy()
        k.__self__._chain_delayed(fn)
        return k

    def _resolve_delayed(self) -> None:
        """Calls `delayed` now and applies its keywords in place; `resolve_delayed` copies instead.

        Does nothing when there is no `delayed`, built or not.

        Raises:
            RuntimeError: The launcher is built.
        """
        if self._C_options.delayed is not None:
            self._C_edit_options(_resolved)

    def resolve_delayed(self) -> LaunchFunction:
        """Returns a new launcher with `delayed` called now and its keywords applied."""
        k = self._copy()
        k.__self__._resolve_delayed()
        return k
```

Update the `Launcher` class docstring's last sentence to: `Every edit comes in two forms: `foo` returns a new, unbuilt launcher and `_foo` changes this one in place until it is built.` Update `make_launcher`'s `Returns:` method list to `(`bind`, `bind_device`, `unbind`, `unbind_device`, `replace_delayed`, `chain_delayed`, `resolve_delayed`, each with an in-place `_` twin, and `check`)`.

- [ ] **Step 4: Run the tests**

Run: `PYTHONPATH=$PWD $PY -m pytest tests/test_launcher_object.py tests/test_delayed.py -q`
Expected: PASS.

- [ ] **Step 5: Mutation check** (the override must be able to fail)

Temporarily change `_resolved`'s return to let eager values win:

```python
    keywords = _delayed_keywords(option.delayed)
    explicit = {
        f.name for f in dataclasses.fields(option) if getattr(option, f.name) != f.default
    }
    return dataclasses.replace(
        option, **{k: v for k, v in keywords.items() if k not in explicit}, delayed=None
    )
```

Run: `PYTHONPATH=$PWD $PY -m pytest tests/test_delayed.py::test_override_and_bad_results -q`
Expected: FAIL (`verify_annotation` is True). Then `git checkout -p intj/launcher.py` (or re-apply the original two lines) to restore, and re-run: PASS.

- [ ] **Step 6: Full run**

Run: `PYTHONPATH=$PWD $PY -m pytest tests -q` and `pyright --pythonpath $PY`
Expected: PASS; 0 errors.

- [ ] **Step 7: Commit**

```bash
git add intj/launcher.py tests/test_launcher_object.py
git commit -m "launcher: in-place _foo and copying foo edits; replace/chain/resolve_delayed"
```

---

### Task 5: Docs and conventions

**Files:**
- Modify: `README.md` (the `delayed` paragraph, ~lines 84-100), `docs/Usage.md` (signature block ~25, the paragraph after it ~29-31, "Lazy build" ~57-68, `delayed` section ~102-115), `TODO.md:69-74`, `MIGRATION_GAP.md:24`, `AGENTS.md` (`## Modules stay out of the import system`, third paragraph), `intj/python_intf/AGENTS.md:128-130`.

**Interfaces:** none (docs only).

- [ ] **Step 1: README.md** — replace the paragraph starting "`delayed` runs once, on the launcher's first call" with:

```
`delayed` runs on the launcher's first call, and its keywords replace the ones
passed to `make_launcher`. `my_kernel.__self__.check()` is a dry run that calls
it too; `resolve_delayed()` calls it once, now, and returns a launcher without
it.
```

- [ ] **Step 2: docs/Usage.md**
  - Signature block: `delayed=None,          # keyword-only: () -> more of these keywords, replacing them, run on first use`.
  - Paragraph after the block: replace with

```
`make_launcher` always returns a launcher. Its `__self__` edits options two
ways: `bind()`, `bind_device()`, `unbind()`, `unbind_device()`,
`replace_delayed()`, `chain_delayed()` and `resolve_delayed()` return a new,
unbuilt launcher; the same names with a leading underscore (`_bind()`, ...)
change this launcher in place and raise `RuntimeError` once it is built. It
also has `check()`. Call `make_launcher` once, outside any hot loop.
```
  - "Lazy build": "it records its keywords as a `MakeLauncherOption`" stays; replace "`launcher.__self__.check()` does the rest without touching the GPU: it runs `delayed`, ..." through "It runs once; launchers derived by `bind()` and the like share its analysis." with

```
`launcher.__self__.check()` is a dry run of the first call's checks, without
touching the GPU: it calls `delayed`, validates arguments, resolves
annotations, analyzes an autotune/heuristics chain, lowers `grid_cpp`, checks
bound values, and provisions a non-default `kernel_cache`. It stores nothing:
the first call runs the same checks again (and calls `delayed` again), and
each launcher checks its own options.
```
    and "The **first call** checks if `check()` has not, then builds:" -> "The **first call** resolves `delayed`, checks, then builds:"; after "Concurrent first calls build once." add: "Once built, `launcher.__self__.options` is the resolved option (`delayed=None`) and the `__text_signature__` is set."
  - `delayed` section: replace "It runs once per `make_launcher` call, ... raises `UnsupportedKernel`, and a result" with

```
Each launcher calls it on its first call (`check()` calls it too, storing
nothing), and its keywords replace the explicit ones, whole key, entering
module identity the same way. A result
```
    keeping the rest of that sentence ("that is not a mapping of `make_launcher` keywords raises `TypeError`. If it raises, ..."). Add after the section's example: "`replace_delayed(fn)` swaps it, `chain_delayed(fn)` runs `fn` after it with `fn`'s keywords winning, and `resolve_delayed()` calls it now."

- [ ] **Step 3: TODO.md** — delete the bullet "**A `delayed=` launcher has no `jit_func` before `check()`.**" (lines 69-74).

- [ ] **Step 4: MIGRATION_GAP.md:24** — replace "**Closed** in `7cebe32` for launchers without `delayed=`: " with "**Closed** in `7cebe32`, and for `delayed=` launchers by the eager/delayed rework: " and delete the sentence "A `delayed=` launcher's `options` is an `intj.launcher.Delayed` with no `jit_func` until `check()` (see `TODO.md`)."

- [ ] **Step 5: AGENTS.md** — in "## Modules stay out of the import system", third paragraph, replace "Every attribute and method the C type defines is `_C_*`; the Python subclass's own are `foo` if public, `_foo` if private, and a public one backed by C wraps its `_C_foo`." through the end of the paragraph with:

```
Every attribute and method the C type defines is `_C_*`; a public one in the
Python subclass backed by C wraps its `_C_foo`. Each option edit comes in two
forms: `foo` returns a new, unbuilt launcher, and the public `_foo` edits this
one in place (through `_C_edit_options`, under the build lock) until it is
built. No other private helper is named `_` + a public name. The launcher holds
one `MakeLauncherOption`, `delayed` included, and nothing derived from it: the
build resolves `delayed`, checks (`_check_options` -> `_BuildOptions`, a
return value only), renders, and returns `(entry, resolved options, doc)` for
C to publish before it swaps the entry in. The module's `_init` allocates the
launcher's state block behind `state` and installs the `traverse`/`release`
ops GC and dealloc reach it through. All mutable launch state (cache, lock,
miss callback) is the launcher's; the module's C state is written once, at
load.
```

- [ ] **Step 6: intj/python_intf/AGENTS.md:128-130** — replace "The launcher object's members (`_C_options`, `_C_plan`, `_C_module`) are read under its critical section and `_C_set_plan` publishes them once;" with "The launcher object's members (`_C_options`, `_C_module`) are read under its critical section; `_C_edit_options` and the build write `options` under it, and the build publishes `doc` once;".

- [ ] **Step 7: Check for stale references**

Run: `grep -rn "Delayed\|_C_plan\|_C_set_plan\|share its analysis\|once per \`make_launcher\` call\|passed both ways" --include=*.md . | grep -v "docs/superpowers\|benchmarks/journals"`
Expected: no output.

- [ ] **Step 8: Full verification**

Run: `PYTHONPATH=$PWD $PY -m pytest tests -q`, `pyright --pythonpath $PY`, and `PYTHON=$PY ninja -C ci lint`
Expected: PASS; 0 errors; lint clean.

- [ ] **Step 9: Commit**

```bash
git add README.md docs/Usage.md TODO.md MIGRATION_GAP.md AGENTS.md intj/python_intf/AGENTS.md
git commit -m "docs: eager and delayed options, in-place and copying launcher edits"
```

# `delayed=`, `MakeLauncherOption`, and Methods on the Launcher Object: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `make_launcher` only records a `MakeLauncherOption` (or a `delayed` callable); every check runs in `k.__self__.check()` or the first call; module and factory methods become methods on `k.__self__`.

**Architecture:** `k` stays a plain `PyCFunction` over a `PyMethodDef` embedded in its launcher object (`intj.launcher.Launcher`, a `__slots__ = ()` Python subclass of `_intj_lazy.Launcher`). The object is fixed-size; the rendered module's `_init` allocates a separate per-launcher state block behind `state` and installs an ops table. Python splits today's `make_launcher` body into `_analyze` (once per `make_launcher` call) and `_bind` (per launcher), cached as a `_Plan` on the object.

**Tech Stack:** C (CPython API 3.8–3.14t), jinja2-rendered C/C++, Python 3.8+ (pyright strict), pytest, triton 3.8, torch.

**Spec:** `docs/superpowers/specs/2026-10-13-delayed-launcher-options-design.md`

## Global Constraints

- Python floor 3.8 in `intj/` (pyright `pythonVersion = "3.8"`); C must compile on CPython 3.8–3.14 and 3.13t/3.14t.
- `intj/` is pyright-strict: `pyright` reports 0 errors. Tests and benchmarks are `# pyright: standard`.
- `import intj` imports neither triton nor torch; `make_launcher` imports no torch.
- Decoration raises `TypeError` only: unknown keyword, None kernel, non-callable `delayed`.
- Every other refusal keeps today's type and message, raised from `check()` or the first call.
- The template never branches on a backend name; `RenderContext` holds no C.
- No new docstrings on private functions under ~10 lines.
- Hit path: `bench_launch.py --readme` zero-grid row and `--last-key` within +1 ns of `9a84e42`.
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`.
- Commands use the venv: `PY=/mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/venv/bin/python`, run from the worktree root with `PYTHONPATH=$PWD`.

## Review Focus

1. `type(k.__self__)()` or `_intj_lazy.Launcher()` → `TypeError`, never a half-initialized launcher (Task 1 test).
2. `delayed` calling its own launcher, or a sibling from the same `make_launcher`, → `RuntimeError("...being built")`, not a deadlock (Task 3 test).
3. A cycle through a `delayed` closure, `bind_values`, or `options`/`origin` is collected (Tasks 1 and 3 tests).
4. `bind` on a built launcher returns an unbuilt launcher, and the original keeps launching (Task 2 test).
5. On free-threaded builds, `options` reads racing `_set_plan` see the old or the new value only (Task 1 test, run by the Python matrix).

---

## File map

| File | Change |
|---|---|
| `intj/runtime/intj_lazy.h` | `intj_launcher` object + `intj_launcher_ops` (replaces `intj_bound_header`) |
| `intj/runtime/intj_lazy.c` | rewrite: lock helpers, `check`/`build`/`spec_key`/`key_chain`/`_set_plan`, getters, subclassable type |
| `intj/runtime/intj_runtime.h` | `struct intj_launcher_state` replaces `intj_bound_tail`; `INTJ_TAIL(b)->` → `b->` |
| `intj/render/entry.c.jinja` | entry reads `self->state`; `_init`/`intj_release`/`intj_traverse_state`/ops; `spec_key`/`key_chain` take the state |
| `intj/options.py` | new: `MakeLauncherOption`, `MakeLauncherKwargs`, `OPTION_FIELDS` |
| `intj/launcher.py` | `_Once`, `_Origin`, `_Analysis`, `_Binding`, `_Plan`, `_analyze`, `_bind`, `Launcher`, `Delayed`, `make_launcher`; delete `LauncherFactory`, `_interpreter_deferred`, `_tail_bytes` |
| `intj/compat.py`, `intj/__init__.py`, comments in `render/`, `kernel_cache.py`, `torch_intf/` | follow the renames |
| `tests/test_runtime.py`, `tests/test_launcher.py`, other tests, `benchmarks/` | migrate call sites; new tests |
| `tests/test_launcher_object.py`, `tests/test_delayed.py` | new |
| `README.md`, `docs/Usage.md`, `TODO.md`, `AGENTS.md`, `CONCETPS.md`, `intj/python_intf/AGENTS.md` | docs |

---

### Task 1: Fixed-size launcher object, module state block, methods on `__self__`

**Files:**
- Modify: `intj/runtime/intj_lazy.h`, `intj/runtime/intj_lazy.c`, `intj/runtime/intj_runtime.h:303,802,835-849,952,976,993,1005,1030-1031,1091,1112,1115,1127,1142`, `intj/render/entry.c.jinja`, `intj/launcher.py` (`_bind`, `_build`, `_interpreter_deferred`, `_tail_bytes`, `module_of`, `_loaded_module`, `_Interner`), `intj/render/context.py:130,221`, `intj/torch_intf/torch_abi.py:62`, `intj/torch_intf/torch_abi.toml:11`
- Test: `tests/test_runtime.py`, `tests/test_launcher.py`, `tests/test_return_compiled.py`, `tests/test_tuned.py`, other `tests/*.py`, `benchmarks/*.py`

**Interfaces:**
- Produces (C, `_intj_lazy`): `Launcher` type (subclassable, `tp_new` raises); `new_launcher(cls, options, origin, builder) -> PyCFunction`; `_header_size() -> int`.
- Produces (launcher object): `check() -> None` (calls `self._check()` under the build lock unless a plan is set), `build() -> module`, `spec_key(device, *dynamic, *public) -> (bytes, int)`, `key_chain(...)` (tuned only), `_set_plan(options, plan, doc: bytes)` (once), getters `options`, `_origin`, `_plan`, `_module` (None while NULL).
- Produces (rendered module): `_init(launcher, callback, interner, [grid_py, hidden], [device], *bound) -> int`, `_header_size()`, `_set_torch_version(...)`.
- Produces (Python, interim until Task 2): launchers made with `stub.new_launcher(stub.Launcher, None, None, build)` then `launch.__self__._set_plan(None, None, doc)`.

- [ ] **Step 1: Write the failing tests (test_runtime.py)**

Add after `test_memo_holds_its_object_until_replaced`:

```python
def test_launchers_come_only_from_new_launcher(stub_module):
    with pytest.raises(TypeError, match="come from make_launcher"):
        stub_module.Launcher()

    class Sub(stub_module.Launcher):
        __slots__ = ()

    with pytest.raises(TypeError, match="come from make_launcher"):
        Sub()
    wide = type("Wide", (stub_module.Launcher,), {"__slots__": ("extra",)})  # one more word
    for cls in (object, int, wide):
        with pytest.raises(TypeError, match="new_launcher"):
            stub_module.new_launcher(cls, None, None, print)
    with pytest.raises(TypeError, match="new_launcher"):
        stub_module.new_launcher(stub_module.Launcher, None, None, 7)
    assert type(stub_module.new_launcher(Sub, None, None, print).__self__) is Sub


def test_set_plan_publishes_options_and_signature_once(stub_module):
    launch = stub_module.new_launcher(stub_module.Launcher, "delayed", "origin", print)
    h = launch.__self__
    assert (h.options, h._origin, h._plan, h._module) == ("delayed", "origin", None, None)
    assert launch.__text_signature__ is None
    h._set_plan("resolved", "plan", b"launch(a, /)\n--\n\n")
    assert (h.options, h._plan) == ("resolved", "plan")
    assert launch.__text_signature__ == "(a, /)"
    h.check()  # a plan is set: nothing runs
    with pytest.raises(RuntimeError, match="set once"):
        h._set_plan("again", "plan", b"")
    fresh = stub_module.new_launcher(stub_module.Launcher, None, None, print)
    with pytest.raises(TypeError, match="_set_plan"):
        fresh.__self__._set_plan(None, None, "not bytes")


def test_options_reads_race_set_plan_safely(stub_module):
    launchers = [
        stub_module.new_launcher(stub_module.Launcher, "before", None, print)
        for _ in range(200)
    ]
    seen, stop = set(), threading.Event()

    def read():
        while not stop.is_set():
            for k in launchers:
                seen.add(k.__self__.options)

    readers = [threading.Thread(target=read) for _ in range(4)]
    for t in readers:
        t.start()
    for k in launchers:
        k.__self__._set_plan("after", None, b"")
    stop.set()
    for t in readers:
        t.join()
    assert seen <= {"before", "after"}
    assert all(k.__self__.options == "after" for k in launchers)


def test_key_chain_is_for_tuned_launchers_only(launch):
    with pytest.raises(TypeError, match="tuned launchers only"):
        launch.__self__.key_chain(0, torch.zeros(4), 5, -7, 1.5, True, 64)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `PYTHONPATH=$PWD $PY -m pytest tests/test_runtime.py -q -k "come_only or set_plan or race or key_chain_is"`
Expected: FAIL (`AttributeError: module '_intj_lazy' has no attribute 'Launcher'`, wrong `new_launcher` arity).

- [ ] **Step 3: Rewrite `intj/runtime/intj_lazy.h`**

Keep the includes, `intj_fast_fn` and the rwlock block (with its comment) as they are. Replace the header comment and everything from `typedef struct intj_bound_header` to the end with:

```c
/* A launcher's fixed-size object.  `_intj_lazy` allocates it at decoration;
 * a rendered module's `_init` allocates the per-launcher state block behind
 * `state` and installs `ops`.  Both compile this one definition;
 * `_header_size` lets Python check they agree. */
```
(at the top, replacing the old comment), and at the end:

```c
/* Free-threaded builds lock the object around its member reads and
 * `_set_plan`; before 3.13 these are plain blocks. */
#ifndef Py_BEGIN_CRITICAL_SECTION
#define Py_BEGIN_CRITICAL_SECTION(op) {
#define Py_END_CRITICAL_SECTION() }
#endif

/* What the rendered module does with its state block. */
typedef struct intj_launcher_ops {
  int (*traverse)(void *state, visitproc visit, void *arg);
  /* drops the block's references, frees its cache and lock, then the block */
  void (*release)(void *state);
  PyObject *(*spec_key)(void *state, PyObject *const *args, Py_ssize_t nargs);
  /* NULL unless tuned */
  PyObject *(*key_chain)(void *state, PyObject *const *args, Py_ssize_t nargs);
} intj_launcher_ops;

typedef struct intj_launcher {
  PyObject_HEAD
  /* hot: the module's intj_launcher_state; NULL until built, and after clear */
  void *state;
  PyMethodDef def;   /* the launcher PyCFunction's; ml_meth is swapped once */
  PyObject *options; /* MakeLauncherOption, or the Delayed until check() */
  PyObject *origin;  /* _Origin, shared with every bind/unbind derivative */
  PyObject *plan;    /* check()'s _Plan; NULL until checked */
  PyObject *builder; /* until built: callable(launcher) -> entry address */
  PyObject *module;
  PyObject *doc; /* bytes behind def.ml_doc, once checked */
  PyThread_type_lock build_lock;
  unsigned long build_thread;   /* thread holding build_lock, else 0 */
  const intj_launcher_ops *ops; /* the module's, once built */
} intj_launcher;

static_assert(offsetof(intj_launcher, state) == sizeof(PyObject),
              "intj: state leads the object");
static_assert(offsetof(intj_launcher, def) + sizeof(PyMethodDef) <=
                  sizeof(PyObject) + 64,
              "intj: the hot path's state and def sit within 64 bytes after "
              "the object head");
```

- [ ] **Step 4: Rewrite `intj/runtime/intj_lazy.c`**

Full file:

```c
/* `_intj_lazy`: the type every launcher's object comes from, and the shim its
 * PyCFunction starts on.  Generic -- no kernel code, no GPU -- so a launcher
 * exists before its module is rendered.  The first call builds: the builder
 * renders and loads the module, whose `_init` allocates and fills the
 * launcher's state block and returns its entry; the shim swaps def.ml_meth to
 * that entry.  CPython reads m_ml->ml_meth on every call (tests/test_runtime.py
 * pins it, 3.8 to 3.14t), so references taken before the build reach the
 * entry as well. */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <string.h>

#include "intj_lazy.h"

typedef struct {
  PyTypeObject *type;
} intj_lazy_state;

static PyObject *intj_lazy_call(PyObject *self, PyObject *const *args,
                                Py_ssize_t nargs);
#define INTJ_SHIM ((PyCFunction)(void (*)(void))intj_lazy_call)

/* Take the build lock: 0, or -1 with an error set.  The thread holding it is
 * refused, not deadlocked: a launcher called or checked from its own build. */
static int intj_lazy_lock(intj_launcher *h) {
  unsigned long me = PyThread_get_thread_ident();
  if (__atomic_load_n(&h->build_thread, __ATOMIC_RELAXED) == me) {
    PyErr_SetString(PyExc_RuntimeError,
                    "intj: a launcher was called while it was being built");
    return -1;
  }
  if (!PyThread_acquire_lock(h->build_lock, NOWAIT_LOCK)) {
    /* detached while waiting: the builder runs Python, and a free-threaded
     * stop-the-world must not wait on us */
    // clang-format off
    Py_BEGIN_ALLOW_THREADS
    PyThread_acquire_lock(h->build_lock, WAIT_LOCK);
    Py_END_ALLOW_THREADS
    // clang-format on
  }
  __atomic_store_n(&h->build_thread, me, __ATOMIC_RELAXED);
  return 0;
}

static void intj_lazy_unlock(intj_launcher *h) {
  __atomic_store_n(&h->build_thread, 0UL, __ATOMIC_RELAXED);
  PyThread_release_lock(h->build_lock);
}

/* Run the builder once: 0, or -1 with an error set.  A failure leaves the
 * shim in place, so the next call retries. */
static int intj_lazy_build(intj_launcher *h) {
  if (__atomic_load_n(&h->def.ml_meth, __ATOMIC_ACQUIRE) != INTJ_SHIM)
    return 0;
  if (intj_lazy_lock(h) != 0)
    return -1;
  int rc = 0;
  PyObject *done = NULL;
  if (__atomic_load_n(&h->def.ml_meth, __ATOMIC_ACQUIRE) == INTJ_SHIM) {
    if (!h->builder) {
      PyErr_SetString(PyExc_RuntimeError, "intj: launcher has been cleared");
      rc = -1;
    } else {
      PyObject *entry =
          PyObject_CallFunctionObjArgs(h->builder, (PyObject *)h, NULL);
      void *address = entry ? PyLong_AsVoidPtr(entry) : NULL;
      Py_XDECREF(entry);
      if (!address) {
        if (!PyErr_Occurred())
          PyErr_SetString(PyExc_RuntimeError,
                          "intj: launcher builder returned no entry");
        rc = -1;
      } else {
        PyCFunction fn;
        memcpy(&fn, &address, sizeof(fn));
        __atomic_store_n(&h->def.ml_meth, fn, __ATOMIC_RELEASE);
        done = h->builder;
        h->builder = NULL;
      }
    }
  }
  intj_lazy_unlock(h);
  Py_XDECREF(done); /* outside the lock: the closure's finalizers run Python */
  return rc;
}

static PyObject *intj_lazy_call(PyObject *self, PyObject *const *args,
                                Py_ssize_t nargs) {
  intj_launcher *h = (intj_launcher *)self;
  if (intj_lazy_build(h) != 0)
    return NULL;
  PyCFunction entry = __atomic_load_n(&h->def.ml_meth, __ATOMIC_ACQUIRE);
  return ((intj_fast_fn)(void (*)(void))entry)(self, args, nargs);
}

static PyObject *intj_lazy_check(PyObject *self, PyObject *unused) {
  (void)unused;
  intj_launcher *h = (intj_launcher *)self;
  if (__atomic_load_n(&h->plan, __ATOMIC_ACQUIRE))
    Py_RETURN_NONE;
  if (intj_lazy_lock(h) != 0)
    return NULL;
  PyObject *result;
  if (__atomic_load_n(&h->plan, __ATOMIC_ACQUIRE)) {
    result = Py_None;
    Py_INCREF(result);
  } else if (!h->options) {
    PyErr_SetString(PyExc_RuntimeError, "intj: launcher has been cleared");
    result = NULL;
  } else {
    result = PyObject_CallMethod(self, "_check", NULL);
  }
  intj_lazy_unlock(h);
  return result;
}

static PyObject *intj_lazy_build_method(PyObject *self, PyObject *unused) {
  (void)unused;
  intj_launcher *h = (intj_launcher *)self;
  if (intj_lazy_build(h) != 0)
    return NULL;
  PyObject *module;
  Py_BEGIN_CRITICAL_SECTION(self);
  module = h->module;
  Py_XINCREF(module);
  Py_END_CRITICAL_SECTION();
  if (!module)
    PyErr_SetString(PyExc_RuntimeError, "intj: launcher has been cleared");
  return module;
}

/* The built launcher's state block, or NULL with an error set. */
static void *intj_lazy_built_state(intj_launcher *h) {
  if (intj_lazy_build(h) != 0)
    return NULL;
  void *state = __atomic_load_n(&h->state, __ATOMIC_ACQUIRE);
  if (!state)
    PyErr_SetString(PyExc_RuntimeError, "intj: launcher has been cleared");
  return state;
}

static PyObject *intj_lazy_spec_key(PyObject *self, PyObject *const *args,
                                    Py_ssize_t nargs) {
  intj_launcher *h = (intj_launcher *)self;
  void *state = intj_lazy_built_state(h);
  return state ? h->ops->spec_key(state, args, nargs) : NULL;
}

static PyObject *intj_lazy_key_chain(PyObject *self, PyObject *const *args,
                                     Py_ssize_t nargs) {
  intj_launcher *h = (intj_launcher *)self;
  void *state = intj_lazy_built_state(h);
  if (!state)
    return NULL;
  if (!h->ops->key_chain) {
    PyErr_SetString(PyExc_TypeError,
                    "intj: key_chain is for tuned launchers only");
    return NULL;
  }
  return h->ops->key_chain(state, args, nargs);
}

/* _set_plan(options, plan, doc: bytes): check()'s result, published once. */
static PyObject *intj_lazy_set_plan(PyObject *self, PyObject *const *args,
                                    Py_ssize_t nargs) {
  intj_launcher *h = (intj_launcher *)self;
  if (nargs != 3 || !PyBytes_CheckExact(args[2])) {
    PyErr_SetString(PyExc_TypeError,
                    "intj: _set_plan(options, plan, doc: bytes)");
    return NULL;
  }
  int set = 0;
  PyObject *old = NULL;
  Py_BEGIN_CRITICAL_SECTION(self);
  if (!h->plan && !h->doc) {
    old = h->options;
    Py_INCREF(args[0]);
    h->options = args[0];
    Py_INCREF(args[2]);
    h->doc = args[2];
    __atomic_store_n(&h->def.ml_doc, PyBytes_AS_STRING(args[2]),
                     __ATOMIC_RELEASE);
    Py_INCREF(args[1]);
    __atomic_store_n(&h->plan, args[1], __ATOMIC_RELEASE);
    set = 1;
  }
  Py_END_CRITICAL_SECTION();
  if (!set) {
    PyErr_SetString(PyExc_RuntimeError, "intj: a launcher's plan is set once");
    return NULL;
  }
  Py_XDECREF(old);
  Py_RETURN_NONE;
}

#define INTJ_MEMBER(field) ((void *)offsetof(intj_launcher, field))

/* An object member, or None while it is NULL. */
static PyObject *intj_lazy_member(PyObject *self, void *closure) {
  PyObject *value;
  Py_BEGIN_CRITICAL_SECTION(self);
  value = *(PyObject **)((char *)self + (uintptr_t)closure);
  value = value ? value : Py_None;
  Py_INCREF(value);
  Py_END_CRITICAL_SECTION();
  return value;
}

static int intj_lazy_traverse(PyObject *self, visitproc visit, void *arg) {
  intj_launcher *h = (intj_launcher *)self;
#if PY_VERSION_HEX < 0x03090000
  /* 3.8's subtype_traverse visits a Python subclass's type itself */
  if (Py_TYPE(self)->tp_traverse == intj_lazy_traverse)
#endif
    Py_VISIT(Py_TYPE(self));
  Py_VISIT(h->options);
  Py_VISIT(h->origin);
  Py_VISIT(h->plan);
  Py_VISIT(h->builder);
  Py_VISIT(h->module);
  /* the module's hook visits what `_init` stored in the state block */
  return h->state ? h->ops->traverse(h->state, visit, arg) : 0;
}

static int intj_lazy_clear(PyObject *self) {
  intj_launcher *h = (intj_launcher *)self;
  /* unpublish first: release drops owners whose finalizers can reenter Python */
  void *state = h->state;
  h->state = NULL;
  if (state)
    h->ops->release(state);
  Py_CLEAR(h->options);
  Py_CLEAR(h->origin);
  Py_CLEAR(h->plan);
  Py_CLEAR(h->builder);
  Py_CLEAR(h->module);
  return 0;
}

static void intj_lazy_dealloc(PyObject *self) {
  intj_launcher *h = (intj_launcher *)self;
  PyTypeObject *type = Py_TYPE(self);
  PyObject_GC_UnTrack(self);
  intj_lazy_clear(self);
  Py_CLEAR(h->doc); /* only here: a live PyCFunction may still read ml_doc */
  if (h->build_lock)
    PyThread_free_lock(h->build_lock);
  type->tp_free(self);
  Py_DECREF(type);
}

static PyObject *intj_lazy_new(PyTypeObject *type, PyObject *args,
                               PyObject *kwargs) {
  (void)type;
  (void)args;
  (void)kwargs;
  PyErr_SetString(PyExc_TypeError, "intj: launchers come from make_launcher");
  return NULL;
}

static PyGetSetDef intj_lazy_getset[] = {
    {(char *)"options", intj_lazy_member, NULL,
     (char *)"The launcher's MakeLauncherOption; a Delayed until check() "
             "resolves it.",
     INTJ_MEMBER(options)},
    {(char *)"_origin", intj_lazy_member, NULL, NULL, INTJ_MEMBER(origin)},
    {(char *)"_plan", intj_lazy_member, NULL, NULL, INTJ_MEMBER(plan)},
    {(char *)"_module", intj_lazy_member, NULL, NULL, INTJ_MEMBER(module)},
    {NULL, NULL, NULL, NULL, NULL}};

static PyMethodDef intj_lazy_type_methods[] = {
    {"check", intj_lazy_check, METH_NOARGS,
     "Run every check make_launcher defers, once; the first call runs it too."},
    {"build", intj_lazy_build_method, METH_NOARGS,
     "Build now if the first call has not, and return the rendered module."},
    {"spec_key", (PyCFunction)(void (*)(void))intj_lazy_spec_key,
     METH_FASTCALL,
     "spec_key(device, *dynamic, *public_args) -> (key bytes, nparams); "
     "builds first."},
    {"key_chain", (PyCFunction)(void (*)(void))intj_lazy_key_chain,
     METH_FASTCALL,
     "key_chain(device, *dynamic, *public_args) -> ((key bytes, ...), found); "
     "tuned launchers only, builds first."},
    {"_set_plan", (PyCFunction)(void (*)(void))intj_lazy_set_plan,
     METH_FASTCALL, NULL},
    {NULL, NULL, 0, NULL}};

static PyType_Slot intj_lazy_slots[] = {
    {Py_tp_new, (void *)intj_lazy_new},
    {Py_tp_traverse, (void *)intj_lazy_traverse},
    {Py_tp_clear, (void *)intj_lazy_clear},
    {Py_tp_dealloc, (void *)intj_lazy_dealloc},
    {Py_tp_getset, (void *)intj_lazy_getset},
    {Py_tp_methods, (void *)intj_lazy_type_methods},
    {0, NULL}};

static PyType_Spec intj_lazy_spec = {"_intj_lazy.Launcher",
                                     sizeof(intj_launcher), 0,
                                     Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC |
                                         Py_TPFLAGS_BASETYPE
#ifdef Py_TPFLAGS_IMMUTABLETYPE
                                         | Py_TPFLAGS_IMMUTABLETYPE
#endif
                                     ,
                                     intj_lazy_slots};

/* new_launcher(cls, options, origin, builder) -> an unbuilt launcher: a
 * PyCFunction over a `cls` instance, `cls` being Launcher or a subclass that
 * adds no storage. */
static PyObject *new_launcher(PyObject *m, PyObject *const *args,
                              Py_ssize_t nargs) {
  intj_lazy_state *st = (intj_lazy_state *)PyModule_GetState(m);
  PyTypeObject *cls =
      nargs == 4 && PyType_Check(args[0]) ? (PyTypeObject *)args[0] : NULL;
  if (!cls || !PyType_IsSubtype(cls, st->type) ||
      cls->tp_basicsize != (Py_ssize_t)sizeof(intj_launcher) ||
      cls->tp_itemsize != 0 || !PyCallable_Check(args[3])) {
    PyErr_SetString(PyExc_TypeError,
                    "intj: new_launcher(cls, options, origin, builder)");
    return NULL;
  }
  intj_launcher *h = (intj_launcher *)cls->tp_alloc(cls, 0);
  if (!h)
    return NULL;
  h->build_lock = PyThread_allocate_lock();
  if (!h->build_lock) {
    Py_DECREF(h);
    return PyErr_NoMemory();
  }
  Py_INCREF(args[1]);
  h->options = args[1];
  Py_INCREF(args[2]);
  h->origin = args[2];
  Py_INCREF(args[3]);
  h->builder = args[3];
  h->def.ml_name = "launch";
  h->def.ml_meth = INTJ_SHIM;
  h->def.ml_flags = METH_FASTCALL;
  h->def.ml_doc = NULL; /* `_set_plan` sets the signature */
  PyObject *launcher = PyCFunction_New(&h->def, (PyObject *)h);
  Py_DECREF(h);
  return launcher;
}

static PyObject *header_size(PyObject *m, PyObject *unused) {
  (void)m;
  (void)unused;
  return PyLong_FromSize_t(sizeof(intj_launcher));
}

static int intj_lazy_exec(PyObject *m) {
  intj_lazy_state *st = (intj_lazy_state *)PyModule_GetState(m);
#if PY_VERSION_HEX >= 0x03090000
  st->type = (PyTypeObject *)PyType_FromModuleAndSpec(m, &intj_lazy_spec, NULL);
#else
  st->type = (PyTypeObject *)PyType_FromSpec(&intj_lazy_spec);
#endif
  if (!st->type)
    return -1;
  Py_INCREF(st->type);
  if (PyModule_AddObject(m, "Launcher", (PyObject *)st->type) != 0) {
    Py_DECREF(st->type);
    return -1;
  }
  return 0;
}
```

Keep `intj_lazy_module_traverse`, `intj_lazy_module_clear`, the module slots, module def and `PyInit__intj_lazy` as they are. Change the method table to:

```c
static PyMethodDef intj_lazy_methods[] = {
    {"new_launcher", (PyCFunction)(void (*)(void))new_launcher, METH_FASTCALL,
     "new_launcher(cls, options, origin, builder) -> launcher"},
    {"_header_size", header_size, METH_NOARGS, NULL},
    {NULL, NULL, 0, NULL}};
```

- [ ] **Step 5: Move per-launcher storage into a state block (`intj_runtime.h`)**

1. Line 802: `typedef intj_bound_header intj_bound_launcher;` → `typedef struct intj_launcher_state intj_bound_launcher;`
2. Replace the "trailing arrays" comment, `intj_bound_tail`, its static assert and `INTJ_TAIL` (835–849) with:

```c
/* One launcher's state, behind intj_launcher.state: `_init` allocates and
 * fills it, ops->release frees it.  Hot fields first. */
struct intj_launcher_state {
  void *st;               /* the module's intj_state */
  int64_t device_ordinal; /* bind_device only */
  int32_t device_handle;  /* bind_device only */
  int32_t cache_ready;
  uint64_t cache[3]; /* opaque intj_cache storage, only ever accessed as one */
  void *fixed_kernel; /* intj_final *: a keyless fixed-device launcher's kernel */
  /* Cold. */
  /* callable(value) -> id or None: this launcher's intern table */
  PyObject *intern;
  /* this launcher's miss callback (the tuned callback if tuned) */
  PyObject *compile_cb;
  intj_rwlock lock; /* guards this launcher's cache, memos and intern writes */
  PyObject *grid_py;
  PyObject *grid_hidden;
  intj_memo memo[INTJ_MEMO_SLOTS];
  PyObject *owners[INTJ_BOUND_SLOTS];
  uint64_t pointer_bits[INTJ_BOUND_SLOTS];
  void *tensors[INTJ_BOUND_SLOTS]; /* at::Tensor * in STATIC_COMPILE */
  intj_objtab objtab;
};
static_assert(offsetof(struct intj_launcher_state, fixed_kernel) +
                      sizeof(void *) <=
                  64,
              "intj: hot fields share the block's first 64 bytes (a span, "
              "not an aligned line)");
```

3. `sed -i 's/INTJ_TAIL(bound)->/bound->/g' intj/runtime/intj_runtime.h intj/render/entry.c.jinja`
4. Comments: 303 `set_torch_version has run` → `_set_torch_version has run`; 1005 `set by init_bound, dropped only by tp_clear` → `set by _init, dropped only by release`; 1030–1031 → `never NULL: _init sets it before publishing the state block, and only release drops it`; 1091 → `The state block's cache storage` (read the line; keep the rest of the sentence).

- [ ] **Step 6: Update `entry.c.jinja`**

1. Line 98: `sizeof(((intj_bound_header *)0)->cache)` → `sizeof(((intj_bound_launcher *)0)->cache)`.
2. Line 613 comment → `/* set by _init before the state block was published; only release drops it */`.
3. `sed -i 's/intj: set_torch_version has not been called/intj: _set_torch_version has not been called/' intj/render/entry.c.jinja`
4. Entry head (`intj_bound_entry`):

```c
  intj_bound_launcher *bound =
      (intj_bound_launcher *)((intj_launcher *)self)->state;
  if (INTJ_UNLIKELY(!bound)) {
    PyErr_SetString(PyExc_RuntimeError, "intj: bound launcher has been cleared");
    return NULL;
  }
  intj_state *st = (intj_state *)bound->st;
```

5. Replace `intj_bound_traverse` and `intj_bound_clear` with:

```c
static int intj_traverse_state(void *state, visitproc visit, void *arg) {
  intj_bound_launcher *bound = (intj_bound_launcher *)state;
  /* `_intj_lazy` visits the type and the object's members; never twice */
  Py_VISIT(bound->grid_py);
  Py_VISIT(bound->grid_hidden);
  Py_VISIT(bound->compile_cb);
  Py_VISIT(bound->intern);
  for (int k = 0; k < INTJ_NMEMO; k++)
    Py_VISIT(bound->memo[k].obj);
  {
    int err = intj_objtab_traverse(bound, &bound->objtab, visit, arg);
    if (err)
      return err;
  }
  for (int i = 0; i < INTJ_NBOUND; i++)
    Py_VISIT(bound->owners[i]);
{% if traverse_cache %}
  if (bound->cache_ready)
    return intj_cache_traverse(bound, visit, arg);
{% elif traverse_fixed %}
  {
    intj_snapshot s = {NULL, 0, 0};
    INTJ_RDLOCK(&bound->lock);
    int failed = bound->fixed_kernel ? intj_final_collect((intj_final *)bound->fixed_kernel, &s) : 0;
    INTJ_RWUNLOCK(&bound->lock);
    return intj_snapshot_visit(&s, failed, visit, arg);
  }
{% endif %}
  return 0;
}

static void intj_release(void *state) {
  intj_bound_launcher *bound = (intj_bound_launcher *)state;
  Py_CLEAR(bound->grid_py);
  Py_CLEAR(bound->grid_hidden);
  Py_CLEAR(bound->compile_cb);
  Py_CLEAR(bound->intern);
  for (int k = 0; k < INTJ_NMEMO; k++)
    Py_CLEAR(bound->memo[k].obj);
  intj_objtab_clear(&bound->objtab);
  for (int i = 0; i < INTJ_NBOUND; i++) {
    Py_CLEAR(bound->owners[i]);
#if defined(INTJ_TORCH_ACCESS_STATIC_COMPILE)
    delete (at::Tensor *)bound->tensors[i];
    bound->tensors[i] = NULL;
#endif
  }
{% if nwords %}
  if (bound->cache_ready) {
    bound->cache_ready = 0;
    intj_cache_free(INTJ_BOUND_CACHE_OF(bound), INTJ_V0_RELEASE);
  }
{% elif fixed_device %}
  intj_final *kernel = (intj_final *)bound->fixed_kernel;
  bound->fixed_kernel = NULL;
  if (kernel) {
    intj_final_release(kernel);
    PyMem_RawFree(kernel);
  }
{% endif %}
  INTJ_RWLOCK_DESTROY(&bound->lock);
  PyMem_RawFree(bound);
}

static PyObject *intj_spec_key(void *state, PyObject *const *args, Py_ssize_t nargs);
{% if tuned %}
static PyObject *intj_key_chain(void *state, PyObject *const *args, Py_ssize_t nargs);
{% endif %}
static const intj_launcher_ops intj_ops = {intj_traverse_state, intj_release, intj_spec_key,
                                           {{ 'intj_key_chain' if tuned else 'NULL' }}};
```

6. `init_bound` → `intj_init`:
   - Comment: `/* _init(launcher, callback, interner, [grid_py, fixed values], [device], *bound) -> the entry's address.  Allocates and fills the launcher's state block, then publishes it.  On failure the launcher is left unbuilt, so its next call retries. ...` (keep the callback/interner sentences).
   - Signature `static PyObject *intj_init(PyObject *self, PyObject *const *args, Py_ssize_t nargs)`; first line after `st`: `intj_bound_launcher *bound = NULL;`.
   - Every `init_bound` in its messages → `_init`.
   - Replace the type check with:

```c
  PyTypeObject *type = Py_TYPE(args[0]);
  intj_launcher *h = (intj_launcher *)args[0];
  if (type->tp_basicsize != (Py_ssize_t)sizeof(intj_launcher) || type->tp_itemsize != 0 ||
      h->state || h->module) {
    PyErr_SetString(PyExc_TypeError, "intj: _init needs an unbuilt launcher");
    return NULL;
  }
```

   - After the last argument check (the `fixed_device` ordinal block), before `#ifdef __cplusplus try {`:

```c
  bound = (intj_bound_launcher *)PyMem_RawCalloc(1, sizeof *bound);
  if (!bound)
    return PyErr_NoMemory();
  INTJ_RWLOCK_INIT(&bound->lock);
  bound->st = st;
```

   - Delete `bound->module = Py_NewRef(self);`.
   - Replace the publish (`bound->traverse = ...; bound->clear = ...; bound->state = st; ...`) with:

```c
    h->ops = &intj_ops;
    Py_BEGIN_CRITICAL_SECTION((PyObject *)h);
    h->module = Py_NewRef(self);
    Py_END_CRITICAL_SECTION();
    h->state = bound; /* last: the entry reads it first */
    return entry;
```

   - `fail:` body → `if (bound) intj_release(bound); /* drops whatever was filled; the launcher stays unbuilt */ return NULL;`
7. `header_size` returns `sizeof(intj_launcher)`.
8. `spec_key` → keep the "Debug/test entry point" comment right before it (a test slices the source from `static PyObject *intj_init` to `/* Debug/test entry`):

```c
static PyObject *intj_spec_key(void *state, PyObject *const *args, Py_ssize_t nargs) {
  intj_bound_launcher *bound = (intj_bound_launcher *)state;
  intj_state *st = (intj_state *)bound->st;
  int64_t device = 0;
  if (nargs != 1 + INTJ_NDYN + INTJ_NPARAMS || !PyLong_CheckExact(args[0]) ||
      intj_as_i64(args[0], &device) != 0 || device < 0 || device > 255) {
    PyErr_Format(PyExc_TypeError,
                 "intj: spec_key(device, *%d dynamic values, *%d public args)",
                 INTJ_NDYN, INTJ_NPARAMS);
    return NULL;
  }
```
   then the unchanged key/vals/comp declarations, and `intj_pack(st, args + 1, args + 1 + INTJ_NDYN, ...)`.
9. `key_chain` → `static PyObject *intj_key_chain(void *state, PyObject *const *args, Py_ssize_t nargs)`: same head as `intj_spec_key` with message `"intj: key_chain(device, *dynamic, *public_args)"`, then `if (!bound->cache_ready) { PyErr_SetString(PyExc_TypeError, "intj: key_chain needs a live launcher"); return NULL; }`; every `args + 2` → `args + 1`. Comment: `key_chain(device, *dynamic, *public_args) -> ((bytes, ...), found)`.
10. `set_torch_version` → `_set_torch_version` (function name, comment, message).
11. Methods table:

```c
static PyMethodDef intj_methods[] = {
    {"_init", (PyCFunction)(void (*)(void))intj_init, METH_FASTCALL, NULL},
    {"_header_size", header_size, METH_NOARGS, NULL},
    {"_set_torch_version", (PyCFunction)(void (*)(void))_set_torch_version,
     METH_FASTCALL, NULL},
    {NULL, NULL, 0, NULL}};
```

- [ ] **Step 7: Python side (interim, `intj/launcher.py`)**

1. `LauncherFactory._bind`: delete the `nmemo` line; end with

```python
        stub = _stub()
        launch = stub.new_launcher(stub.Launcher, None, None, build)
        launch.__self__._set_plan(None, None, doc)
        return launch
```

2. `LauncherFactory._build`: `module.init_bound(header, ...)` → `module._init(header, ...)`.
3. `_interpreter_deferred.make()`: same three lines with doc `b"(*args, **kwargs)\x00"`.
4. Delete `_tail_bytes`.
5. `module_of` docstring: `` `module_of(k).spec_key(k, ...)` `` → `` `k.__self__.spec_key(...)` builds the same way ``.
6. `_loaded_module`: `module.header_size() != _stub().header_size()` → `module._header_size() != _stub()._header_size()`; `module.set_torch_version(` → `module._set_torch_version(`.
7. `_Interner`: comment `read once by init_bound` → `read once by _init`.
8. `render/context.py:130,221`: `init_bound` → `_init`. `torch_abi.py:62`, `torch_abi.toml:11`: `set_torch_version` → `_set_torch_version`.

- [ ] **Step 8: Migrate the tests and benchmarks**

```bash
sed -i 's/\.set_torch_version(/._set_torch_version(/g; s/\.init_bound(/._init(/g; s/\.header_size()/._header_size()/g' tests/*.py benchmarks/*.py
$PY /tmp/migrate_spec_key.py write tests/*.py benchmarks/*.py   # prints "replaced 90" and LEFT lines
```

`/tmp/migrate_spec_key.py`:

```python
import re, sys, pathlib
# <module expr>.spec_key(<launcher>, ...) -> <launcher>.__self__.spec_key(...)
CALL = re.compile(
    r"(?:\b_launcher_module\(\w+\)|\bmodule_of\(\w+\)|\b\w+\[\d+\]|\b\w+)"
    r"\.(spec_key|key_chain)\(\s*(\w+(?:\[\d+\])?),\s*"
)
total = 0
for path in map(pathlib.Path, sys.argv[2:]):
    text = path.read_text()
    new, n = CALL.subn(lambda m: f"{m[2]}.__self__.{m[1]}(", text)
    total += n
    if sys.argv[1] == "write":
        path.write_text(new)
    left = [l for l in new.splitlines() if re.search(r"\.(spec_key|key_chain)\b", l) and "__self__" not in l]
    for l in left:
        print(f"{path}: LEFT {l.strip()}")
print("replaced", total)
```

Then by hand:
1. `test_runtime.py`: delete `_TAIL = ...`; `python - <<'EOF'` regex `new_launcher\(\s*_TAIL,\s*b"[^"]*",\s*` → `new_launcher(stub_module.Launcher, None, None, ` over the file.
2. `test_runtime.py` subprocess source: `launch = lazy.new_launcher(lazy.Launcher, None, None, lambda header: module._init(header, print, print))`.
3. `test_first_call_builds_and_swaps_for_old_references`: `launch.__self__.__self__ is None` → `launch.__self__._module is None`; `launch.__self__.__self__ is module` → `launch.__self__._module is module`.
4. `test_unbuilt_and_built_launchers_are_collected`: parametrize `subclass` in `(False, True)`; `cls = type("Sub", (stub_module.Launcher,), {"__slots__": ()}) if subclass else stub_module.Launcher`; create with `stub_module.new_launcher(cls, owner, owner, build)` (cycles through options and origin too).
5. `test_launcher.py::_launcher_over`: import `_Interner, _launcher_compile, _stub`; `stub = _stub(); launch = stub.new_launcher(stub.Launcher, None, None, lambda h: module._init(h, _launcher_compile(module), _Interner()))`.
6. `test_launcher.py`: slice `"static PyObject *init_bound"` → `"static PyObject *intj_init"`; `hasattr(module, "set_torch_version")` → `"_set_torch_version"`; docstring `before set_torch_version` → `before _set_torch_version`.
7. Replace `test_bound_handles_share_one_header_layout` with:

```python
def test_every_launcher_shares_one_fixed_size_object(bound_kernel, device_kernel):
    """Every launcher object is one fixed-size type; its state lives behind `state`."""
    from intj.launcher import _stub

    pointer = Argument(
        type=tl.pointer_type(tl.float32), specialize=NEVER, bind_value=BindValue.POINTER
    )
    two = make_launcher(
        bound_kernel,
        extra_annotation={"x": pointer, "p": pointer},
        no_gpu=True,
        torch_access_mode=TorchAccessMode.INTERPRETER,
    ).bind(x=0, p=0)
    none = make_launcher(
        device_kernel,
        bind_device=True,
        no_gpu=True,
        torch_access_mode=TorchAccessMode.INTERPRETER,
    ).bind_device(0)
    assert two(0, 0, 1, 7) is None and none(0, 1, 7) is None
    t2, t0 = type(two.__self__), type(none.__self__)
    assert t2 is t0 and issubclass(t2, _stub().Launcher)
    assert t2.__basicsize__ == _stub()._header_size()
    assert t2.__itemsize__ == 0
```

8. `test_return_compiled.py` (the two-line call the script cannot see): `assert default.__self__.spec_key(0, x, 7) == returning.__self__.spec_key(0, x, 7)`.
9. `bench_ffi_compare.py`: `spec_key = intj_mixed_bound.__self__.spec_key` and `spec_key(0, *mixed_args) == spec_key(0, *other_args) == (b"", 5)`.
10. `bench_launch.py`: `bench(keyed.__self__.spec_key, (0, *args), ...)`; `bench(fn.__self__.spec_key, (device, *dyn, *public), ...)`.
11. Re-run the script in `check` mode; resolve every LEFT line by hand.

- [ ] **Step 9: Run the suite**

Run: `PYTHONPATH=$PWD $PY -m pytest tests -q -x`
Expected: all pass, including the four new tests.

- [ ] **Step 10: Gates and commit**

```bash
~/.local/bin/pyright && ~/.local/bin/pre-commit run --all-files --show-diff-on-failure
bash tests/run_python_matrix.sh 3.8 3.13t 3.14t
git add -A && git commit -m "lazy: a fixed-size launcher object; module state behind it; methods on __self__"
```

---

### Task 2: `MakeLauncherOption`, `check()`, bind/unbind on the launcher object

**Files:**
- Create: `intj/options.py`, `tests/test_launcher_object.py`
- Modify: `intj/launcher.py`, `intj/compat.py:39-56`, `intj/__init__.py`, `intj/render/build.py:241`, `intj/kernel_cache.py:34,165`, `benchmarks/bench_mix.py:154`
- Test: `tests/test_{launcher,runtime,grid,kernel_cache,return_compiled,tuned,annotation}.py`, `benchmarks/{bench_launch,bench_ffi_compare,bench_hip_module_launch,bench_intj_ffi_paths}.py`

**Interfaces:**
- Consumes (Task 1): `new_launcher(cls, options, origin, builder)`, `check`, `build`, `_set_plan`, `options`/`_origin`/`_plan`/`_module`, `module._init(launcher, ...)`.
- Produces: `intj.MakeLauncherOption`; `intj.options.MakeLauncherKwargs`, `OPTION_FIELDS: frozenset[str]`; `intj.launcher.Launcher` with `bind(**values)`, `bind_device(ordinal, /, **values)`, `unbind()`, `unbind_device()` returning `LaunchFunction`; `_Once[T].get(compute)`; `_Origin(explicit, names)`; `_analyze(option) -> _Analysis`; `_bind(analysis, option) -> _Binding`; `_new_launcher(options, origin) -> LaunchFunction`; `Launcher._check()`, `Launcher._derive(edit)`.

- [ ] **Step 1: Write the failing tests (`tests/test_launcher_object.py`)**

```python
# pyright: standard
"""The launcher object: decoration records options; check() and bind/unbind."""

import subprocess
import sys
import textwrap

import pytest
import triton
import triton.language as tl

from intj import MakeLauncherOption, make_launcher
from intj.annotation import Argument, BindValue, NEVER
from intj.launcher import Launcher, UnsupportedKernel
from intj.options import OPTION_FIELDS, MakeLauncherKwargs
from intj.torch_intf.torch_abi import TorchAccessMode


@triton.jit
def store(x, n):
    tl.store(x + tl.arange(0, 1), n)


_POINTER = Argument(
    type=tl.pointer_type(tl.float32), specialize=NEVER, bind_value=BindValue.POINTER
)
_HOST = {"no_gpu": True, "torch_access_mode": TorchAccessMode.INTERPRETER}


def test_decoration_records_and_check_validates():
    launch = make_launcher(store, no_gpu=1)  # no check at decoration
    assert launch.__self__.options == MakeLauncherOption(store, no_gpu=1)
    assert isinstance(launch.__self__, Launcher)
    with pytest.raises(TypeError, match="no_gpu must be a bool"):
        launch.__self__.check()
    with pytest.raises(TypeError, match="no_gpu must be a bool"):
        launch(0, 0, 1, 0, 4)  # the first call checks too, and retries


def test_unknown_keyword_fails_at_decoration():
    with pytest.raises(TypeError, match=r"unexpected keyword argument\(s\) \['bogus'\]"):
        make_launcher(store, bogus=1)
    with pytest.raises(TypeError, match="bogus"):
        make_launcher(bogus=1)


def test_decoration_imports_no_torch(tmp_path):
    source = tmp_path / "kern.py"
    source.write_text(textwrap.dedent("""
        import triton, triton.language as tl
        @triton.jit
        def k(x):
            tl.store(x, 1)
    """))
    probe = f"""
import sys; sys.path.insert(0, {str(tmp_path)!r})
import intj, kern
torch_before = "torch" in sys.modules
intj.make_launcher(kern.k, no_gpu=True)
assert torch_before or "torch" not in sys.modules, "make_launcher imported torch"
from intj import _imports
assert _imports.torch is None
"""
    subprocess.run([sys.executable, "-c", probe], check=True)


def test_bind_and_unbind_derive_new_launchers():
    base = make_launcher(store, extra_annotation={"x": _POINTER}, **_HOST)
    bound = base.__self__.bind(x=0)
    fixed = bound.__self__.bind_device(3)
    assert base.__self__.options.bind_values is None
    assert bound.__self__.options.bind_values == {"x": 0}
    assert (fixed.__self__.options.bind_device, fixed.__self__.options.bind_values) == (3, {"x": 0})
    assert fixed.__self__.unbind_device().__self__.options == bound.__self__.options
    assert bound.__self__.unbind().__self__.options == base.__self__.options
    assert fixed.__self__.bind_device(None).__self__.options.bind_device is None
    assert bound(0, 0, 1, 4) is None and fixed(0, 1, 4) is None
    with pytest.raises(TypeError, match="missing"):
        bound.__self__.unbind()(0, 0, 1, 4)


def test_derivatives_share_one_analysis(monkeypatch):
    import intj.launcher as launcher

    calls = []
    real = launcher._analyze
    monkeypatch.setattr(launcher, "_analyze", lambda o: calls.append(1) or real(o))
    base = make_launcher(store, extra_annotation={"x": _POINTER}, **_HOST)
    for k in (base.__self__.bind(x=0), base.__self__.bind_device(0, x=0)):
        k.__self__.check()
    assert calls == [1]


def test_bind_on_a_built_launcher_returns_an_unbuilt_one():
    base = make_launcher(store, **_HOST)
    assert base(0, 0, 1, 0, 4) is None
    child = base.__self__.bind_device(0)
    assert child.__self__._module is None and child.__self__._plan is None
    assert child(0, 1, 0, 4) is None and base(0, 0, 1, 0, 4) is None


def test_launcher_object_is_not_instantiable():
    launch = make_launcher(store, **_HOST)
    with pytest.raises(TypeError, match="come from make_launcher"):
        type(launch.__self__)()


def test_bind_device_true_points_at_the_method():
    launch = make_launcher(store, bind_device=True, **_HOST)
    with pytest.raises(TypeError, match=r"not a flag; use k.__self__.bind_device\(ordinal\)"):
        launch.__self__.check()


def test_kwargs_typeddict_matches_the_dataclass():
    assert set(MakeLauncherKwargs.__annotations__) == OPTION_FIELDS
    assert "jit_func" not in OPTION_FIELDS and "delayed" not in OPTION_FIELDS
```

(`UnsupportedKernel` stays imported for Task 3's additions to this file; drop the import if ruff flags it now.)

- [ ] **Step 2: Run them to verify they fail**

Run: `PYTHONPATH=$PWD $PY -m pytest tests/test_launcher_object.py -q`
Expected: FAIL at import (`cannot import name 'MakeLauncherOption'`).

- [ ] **Step 3: Create `intj/options.py`**

```python
"""`MakeLauncherOption`: every `make_launcher` keyword but `delayed`, as one value."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping, Sequence
from typing import Any, TypedDict

from .kernel_cache import KernelCache
from .torch_intf.torch_abi import TorchAccessMode


@dataclasses.dataclass(frozen=True)
class MakeLauncherOption:
    """Every `make_launcher` keyword except `delayed`, stored as given.

    Nothing is validated here: `k.__self__.check()`, at the latest the first
    call, checks the values with the errors `make_launcher` used to raise.
    `k.__self__.options` is a launcher's; `dataclasses.replace` makes variants.
    """

    jit_func: Any
    dynamic_options: Sequence[str] = ()
    extra_annotation: Mapping[str, object] | None = None
    options: Mapping[str, Any] | None = None
    torch_access_mode: TorchAccessMode | None = None
    kernel_cache: KernelCache = KernelCache.INTJ
    no_gpu: bool = False
    verify_annotation: bool = False
    bind_device: int | None = None
    bind_values: Mapping[str, object] | None = None
    grid_arg: int | None = None
    grid_cpp: object | None = None
    grid_py: Callable[[dict[str, object]], object] | None = None
    return_compiled: bool = False
    assume_constant_globals: bool = False


class MakeLauncherKwargs(TypedDict, total=False):
    """`make_launcher`'s keywords: `MakeLauncherOption`'s fields but `jit_func`."""

    dynamic_options: Sequence[str]
    extra_annotation: Mapping[str, object] | None
    options: Mapping[str, Any] | None
    torch_access_mode: TorchAccessMode | None
    kernel_cache: KernelCache
    no_gpu: bool
    verify_annotation: bool
    bind_device: int | None
    bind_values: Mapping[str, object] | None
    grid_arg: int | None
    grid_cpp: object | None
    grid_py: Callable[[dict[str, object]], object] | None
    return_compiled: bool
    assume_constant_globals: bool


#: the keywords `make_launcher` (and `delayed`) may set
OPTION_FIELDS: frozenset[str] = frozenset(
    f.name for f in dataclasses.fields(MakeLauncherOption)
) - {"jit_func"}
```

- [ ] **Step 4: Split `make_launcher` in `intj/launcher.py`**

1. Imports: add `os`; `from typing import TYPE_CHECKING, Any, Generic, Protocol, TypeVar, cast, overload`; `from .options import OPTION_FIELDS, MakeLauncherKwargs, MakeLauncherOption`; under `TYPE_CHECKING`: `from typing_extensions import Unpack`.
2. Delete `LauncherFactory`, `_interpreter_deferred` and both `make_launcher` overloads.
3. Add, where `LauncherFactory` was:

```python
T = TypeVar("T")


class _Once(Generic[T]):
    # a value computed once, under a lock; a failure is not kept, and the
    # computing thread re-entering is refused rather than deadlocked
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._running = False
        self._result: tuple[T] | None = None  # one load: all or nothing

    def get(self, compute: Callable[[], T]) -> T:
        result = self._result
        if result is None:
            with self._lock:
                result = self._result
                if result is None:
                    if self._running:
                        raise RuntimeError(
                            "intj: a launcher was called while it was being built"
                        )
                    self._running = True
                    try:
                        result = self._result = (compute(),)
                    finally:
                        self._running = False
        return result[0]


@dataclasses.dataclass(frozen=True)
class _Analysis:
    """What `check()` derives from a `make_launcher` call's non-bind options.

    Computed once per `make_launcher` call and shared by every launcher
    derived from it: bind fields are not inputs.
    """

    jit_func: JitFunction
    params: tuple[ResolvedParam, ...]
    options: tuple[tuple[str, Any], ...]
    torch_access_mode: TorchAccessMode
    kernel_cache: KernelCache
    verify_annotation: bool
    no_gpu: bool
    grid_arg: int | None = None
    grid_py: Callable[[dict[str, object]], object] | None = None
    grid_cpp: GridCode | None = None
    return_compiled: bool = False
    tuned: _Tuned | None = None
    grid_fn: object | None = None
    dynamic: tuple[str, ...] = ()
    global_values: tuple[tuple[str, str, tuple[object, ...]], ...] = ()
    #: `dynamic_options` naming a tl.constexpr parameter: no slot, and only
    #: checked to be a compile option once the target is known
    declared_parameters: tuple[str, ...] = ()


@dataclasses.dataclass(frozen=True, eq=False)
class _Binding:
    """One launcher's checked bind fields."""

    device_binding: DeviceBinding
    params: tuple[ResolvedParam, ...]  # bound tensors' types resolved
    args: tuple[object, ...]  # `_init`'s trailing arguments: [device], *bound values
    hidden: tuple[object, ...]  # grid_py's fixed values
    doc: bytes  # the launcher's `__text_signature__`


@dataclasses.dataclass(frozen=True, eq=False)
class _Plan:
    analysis: _Analysis
    binding: _Binding


@dataclasses.dataclass(frozen=True, eq=False)
class _Origin:
    """One `make_launcher` call, shared by every launcher derived from it."""

    explicit: MakeLauncherOption
    names: frozenset[str]  # the keywords passed explicitly
    analysis: _Once[_Analysis] = dataclasses.field(default_factory=_Once)


_PACKAGE = os.path.dirname(__file__) + os.sep


def _warn(message: str, category: type[Warning]) -> None:
    # check() is reached from C, so no fixed stacklevel finds the caller:
    # point at the first frame outside intj instead
    frame, level = inspect.currentframe(), 1
    while frame is not None and frame.f_code.co_filename.startswith(_PACKAGE):
        frame, level = frame.f_back, level + 1
    warnings.warn(message, category, stacklevel=level)
```

4. `_analyze(option: MakeLauncherOption) -> _Analysis`, docstring: `"""Runs every check on `option`'s non-bind fields: `make_launcher`'s old body.` + Raises/Warns sections moved from the old `make_launcher` docstring. Body: unpack `jit_func, dynamic_options, extra_annotation, options, torch_access_mode, kernel_cache, no_gpu, verify_annotation, grid_arg, grid_cpp, grid_py, return_compiled, assume_constant_globals` from `option`, then the old body from `# Cheap checks on every argument` to the `ClassGlobalWarning` block, with these edits:
   - `flags`: drop the `"bind_device"` entry.
   - The `InterpretedFunction` branch: `raise UnsupportedKernel("intj: TRITON_INTERPRET=1 is not supported")`, comment: `` `@triton.jit` under `TRITON_INTERPRET=1` returns an InterpretedFunction, which has none of the attributes the analysis needs; decoration no longer analyzes, so refusing here is the first call's refusal. ``
   - Bound-pointer warning: `if verify_annotation:` (no `needs_binding`), `_warn(<same message>, RuntimeWarning)`.
   - `factory = LauncherFactory(...)` → `analysis = _Analysis(kernel, resolved, tuple(options.items()), access, kernel_cache, verify_annotation, no_gpu, grid_arg, grid_py, compiled_grid, return_compiled, tuned, grid_cpp, dynamic=..., global_values=..., declared_parameters=...)`.
   - `ClassGlobalWarning` through `_warn(<same message>, ClassGlobalWarning)`; `return analysis`.
5. `_bind(a: _Analysis, option: MakeLauncherOption) -> _Binding`:

```python
def _bind(a: _Analysis, option: MakeLauncherOption) -> _Binding:
    """Checks one launcher's `bind_device` and `bind_values` against its analysis."""
    device = option.bind_device
    if type(device) is bool:
        raise TypeError(
            "intj: bind_device takes an int device ordinal, not a flag; "
            "use k.__self__.bind_device(ordinal)"
        )
    if device is not None and (type(device) is not int or not 0 <= device <= 2**31 - 1):
        raise TypeError("intj: device ordinal must be an int in [0, 2**31)")
    values = option.bind_values
    if values is None:
        values = {}
    elif not (isinstance(values, Mapping) and all(type(k) is str for k in values)):
        raise TypeError("intj: bind_values must be None or a mapping keyed by str")
```
   then the old `LauncherFactory._bind` body with `self.params` → `a.params`, ending:

```python
    binding = DeviceBinding.NOT_FIXED if device is None else DeviceBinding.FIXED
    args = (*(() if device is None else (device,)), *bound_values)
    params, _, _ = _render_params(state, binding)
    doc = _launch_doc(
        params,
        device is not None,
        a.grid_arg,
        a.grid_cpp,
        a.grid_py is not None,
        dynamic=a.dynamic,
    )
    return _Binding(binding, state, args, hidden, doc)
```

6. The mixin, `_launcher_type`, `_new_launcher`, `LaunchFunction`:

```python
class Launcher:
    """A launcher's object: `k.__self__` for `k = make_launcher(...)`.

    `k` is a C function whose entry is a build shim until its first call
    succeeds, then the rendered launch.  This object holds that entry, the
    options and, once built, the module's per-launcher state.  The C base
    (`_intj_lazy.Launcher`) adds `options`, `check()`, `build()`,
    `spec_key()` and `key_chain()`.
    """

    __slots__ = ()

    if TYPE_CHECKING:

        @property
        def options(self) -> MakeLauncherOption: ...
        @property
        def _origin(self) -> _Origin: ...
        @property
        def _plan(self) -> _Plan | None: ...
        @property
        def _module(self) -> types.ModuleType | None: ...
        def check(self) -> None: ...
        def build(self) -> types.ModuleType: ...
        def spec_key(self, *args: object) -> tuple[bytes, int]: ...
        def key_chain(self, *args: object) -> tuple[tuple[bytes, ...], bool]: ...
        def _set_plan(self, options: MakeLauncherOption, plan: _Plan, doc: bytes) -> None: ...

    def bind(self, /, **values: object) -> LaunchFunction:
        """Returns a new launcher with `values` merged into `bind_values`.

        Validates nothing: a missing, unknown or mistyped value fails in the
        new launcher's `check()`.  A bound tensor takes an exact
        `torch.Tensor`, `torch.nn.Parameter`, or None.
        """
        return self._derive(
            lambda o: dataclasses.replace(o, bind_values={**(o.bind_values or {}), **values})
        )

    def bind_device(self, ordinal: int | None, /, **values: object) -> LaunchFunction:
        """Returns a new launcher fixed to device `ordinal`, `values` merged as for `bind`.

        It is called as `(stream, grid, ...)` and owns a separate kernel cache,
        or one kernel when no dynamic specialization key remains.  None is
        `unbind_device()`.
        """
        return self._derive(
            lambda o: dataclasses.replace(
                o, bind_device=ordinal, bind_values={**(o.bind_values or {}), **values}
            )
        )

    def unbind(self) -> LaunchFunction:
        """Returns a new launcher with no bound values."""
        return self._derive(lambda o: dataclasses.replace(o, bind_values=None))

    def unbind_device(self) -> LaunchFunction:
        """Returns a new launcher that takes the device per call."""
        return self._derive(lambda o: dataclasses.replace(o, bind_device=None))

    def _derive(self, edit: Callable[[MakeLauncherOption], MakeLauncherOption]) -> LaunchFunction:
        return _new_launcher(edit(self.options), self._origin)

    def _check(self) -> None:
        # check()'s body: the C method calls it under the build lock, as does _build
        if self._plan is not None:
            return
        option = self.options
        analysis = self._origin.analysis.get(lambda: _analyze(option))
        binding = _bind(analysis, option)
        self._set_plan(option, _Plan(analysis, binding), binding.doc)

    def _build(self) -> int:
        """<the old `LauncherFactory._build` docstring, "fills `header`" → "fills this launcher">"""
        self._check()
        plan = self._plan
        assert plan is not None
        a, b = plan.analysis, plan.binding
        <the old `_build` body: `self.` → `a.` for analysis fields, `resolved` → `b.params`,
         `self._binding()` → `b.device_binding`, `hidden` → `b.hidden`>
        grid_py = () if a.grid_py is None else (a.grid_py, b.hidden)
        return module._init(self, callback, _Interner(), *grid_py, *b.args)


class LaunchFunction(Protocol):
    """What `make_launcher` returns: a C function whose `__self__` is its `Launcher`."""

    @property
    def __self__(self) -> Launcher: ...
    def __call__(self, *args: Any) -> Any: ...


@functools.lru_cache(maxsize=1)
def _launcher_type() -> type[Launcher]:
    """Returns `Launcher` over this process's `_intj_lazy.Launcher`."""
    namespace = {"__slots__": (), "__module__": __name__, "__qualname__": "Launcher"}
    return cast("type[Launcher]", type("Launcher", (Launcher, _stub().Launcher), namespace))


def _new_launcher(options: MakeLauncherOption, origin: _Origin) -> LaunchFunction:
    cls = _launcher_type()
    return _stub().new_launcher(cls, options, origin, cls._build)  # pyright: ignore[reportPrivateUsage]  # the builder
```

7. `make_launcher` (keep `_NO_KERNEL` above it), with two overloads returning `LaunchFunction` (kernel given) and `Callable[[JitFunction], LaunchFunction]` (decorator form):

```python
def make_launcher(
    jit_func: JitFunction = _NO_KERNEL, /, **kwargs: Unpack[MakeLauncherKwargs]
) -> Any:
    """Returns a fast launcher for `jit_func`, checked and built on first use.

    <rewritten docstring: the decorator-factory paragraph as today; "Decoration
    only records a `MakeLauncherOption`: no check, no GPU, no torch import.
    `k.__self__.check()` runs every check now; otherwise the first call does,
    then builds."; the call convention paragraph as today; Args as today with
    `bind_device: A device ordinal: the launcher is called as `(stream, grid,
    ...)` ...` and `bind_values: Bound parameter name -> value; see
    `Launcher.bind`.`; Returns: "The launcher, a C function; `__self__` is its
    `Launcher`. Without `jit_func`, a decorator that makes one."; Raises:
    "TypeError: A None kernel or an unknown keyword. Everything else raises
    from `check()` or the first call.">
    """
    unknown = kwargs.keys() - OPTION_FIELDS
    if unknown:
        raise TypeError(
            f"intj: make_launcher() got unexpected keyword argument(s) {sorted(unknown)}"
        )
    if jit_func is _NO_KERNEL:
        return functools.partial(make_launcher, **kwargs)
    if jit_func is None:
        raise TypeError("intj: make_launcher got None instead of a kernel")
    option = MakeLauncherOption(jit_func, **kwargs)
    return _new_launcher(option, _Origin(option, frozenset(kwargs)))
```

   (`/` keeps `jit_func` positional, so `make_launcher(jit_func=k)` is the unknown-keyword error; if any test passes `jit_func=` by name, drop the `/`.)
8. `_materialize_module` comment `make_launcher provisioned already` → `check() provisioned already`; `_provision` docstring `Only ever reached from `make_launcher`` → `Only ever reached from `check()``.
9. `intj/__init__.py`: import and export `MakeLauncherOption` (docstring Exports line + `__all__`).
10. `compat.py::_make`: `bind_device=True,` → `bind_device=device,` and drop `.bind_device(device)`.
11. `render/build.py:241` `"make_launcher validated this"` → `"check() validated this"`; `kernel_cache.py:34` `` `make_launcher` provisions on demand `` → `` A launcher's `check()` provisions on demand ``; `kernel_cache.py:165` `inside `make_launcher`` → `inside `check()``.
12. `bench_mix.py:154`: `params = inspect.signature(make_launcher).parameters` and `if "no_gpu" not in params and "kwargs" not in params:`.

- [ ] **Step 5: Run the new tests**

Run: `PYTHONPATH=$PWD $PY -m pytest tests/test_launcher_object.py -q`
Expected: PASS.

- [ ] **Step 6: Migrate the remaining call sites**

`/tmp/migrate_bind.py` (KWARG_INLINE's lookahead includes `\n`):

```python
"""Task 2 test migration: bind_device kwarg, bind receivers, check() in raises/warns."""
import ast, re, sys, pathlib

KWARG_LINE = re.compile(r"^[ \t]*bind_device=[^\n]*,[ \t]*\n", re.M)
KWARG_INLINE = re.compile(r",\s*bind_device=[^,)\n]+(?=[,)\n])|\bbind_device=[^,)\n]+,\s*")
RECEIVER = re.compile(r"(\)|(?<![\w.])(?!_?signature\b)\w+)(\s*)\.(bind|bind_device)\(")
CHECKED = ("make_launcher", "bind", "bind_device", "unbind", "unbind_device")

def callee(node):
    f = node.func
    return f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else None

def wants_check(stmt):
    return isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call) and callee(stmt.value) in CHECKED

def raising_blocks(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.With) and any(
            isinstance(i.context_expr, ast.Call)
            and isinstance(i.context_expr.func, ast.Attribute)
            and i.context_expr.func.attr in ("raises", "warns")
            for i in node.items
        ):
            yield node

def add_checks(text):
    lines = text.splitlines(keepends=True)
    ends = sorted(
        {(s.value.end_lineno, s.value.end_col_offset)
         for w in raising_blocks(ast.parse(text)) for s in w.body if wants_check(s)},
        reverse=True,
    )
    for line, col in ends:
        row = lines[line - 1]
        lines[line - 1] = row[:col] + ".__self__.check()" + row[col:]
    return "".join(lines), len(ends)

mode, *paths = sys.argv[1:]
totals = [0, 0, 0]
for path in map(pathlib.Path, paths):
    text = path.read_text()
    text, a = KWARG_LINE.subn("", text)
    text, b = KWARG_INLINE.subn("", text)
    text, c = RECEIVER.subn(lambda m: f"{m[1]}{m[2]}.__self__.{m[3]}(", text)
    text, d = add_checks(text)
    for i, n in enumerate((a + b, c, d)):
        totals[i] += n
    if mode == "write":
        path.write_text(text)
    for n, row in enumerate(text.splitlines(), 1):
        if re.search(r"bind_device\s*=", row):
            print(f"{path}:{n}: LEFT {row.strip()}")
print("kwargs removed %d, receivers %d, checks added %d" % tuple(totals))
```

Before writing, `grep -n "bind_device=" tests/*.py benchmarks/*.py` and confirm every hit is a `make_launcher` keyword (a helper's own `bind_device=` keyword must survive: restore it by hand). Then:

```bash
FILES="tests/test_launcher.py tests/test_runtime.py tests/test_grid.py tests/test_kernel_cache.py tests/test_return_compiled.py tests/test_tuned.py tests/test_annotation.py benchmarks/bench_launch.py benchmarks/bench_ffi_compare.py benchmarks/bench_hip_module_launch.py benchmarks/bench_intj_ffi_paths.py"
$PY /tmp/migrate_bind.py write $FILES
```
Expected: about `kwargs removed 39, receivers 92, checks added 52`; no LEFT. Run `ruff format` (via pre-commit) afterwards.

Manual edits in `tests/test_launcher.py`:
1. Replace `test_bind_device_requires_one_positional_ordinal` with:

```python
def test_bind_device_takes_one_positional_ordinal(device_kernel):
    launch = make_launcher(device_kernel, no_gpu=True)
    with pytest.raises(TypeError):
        launch.__self__.bind_device()
    with pytest.raises(TypeError):
        launch.__self__.bind_device(0, 1)
    with pytest.raises(TypeError):
        launch.__self__.bind_device(device_ordinal=0)
    fixed = launch.__self__.bind_device(0)
    fixed.__self__.check()
    assert tuple(inspect.signature(fixed).parameters) == ("stream", "grid", "x")
    assert type(fixed).__flags__ & (1 << 11)  # Py_TPFLAGS_HAVE_VECTORCALL
    assert fixed(0, 1, 7) is None
    with pytest.raises(TypeError, match="arguments"):
        fixed(0, 0, 1, 7)
    with pytest.raises(TypeError, match="keyword"):
        fixed(0, 1, x=7)
```

2. `test_bind_device_validates_exact_nonnegative_int32`: drop `None` from the value list (`bind_device(None)` is `unbind_device()`).
3. `test_bind_requires_every_bound_name_once`: `match="not callable"` / `factory(0, 0, 1, 4)` → `match="missing"`: an unbound launcher's first call fails its check.
4. `test_bind_validates_names_before_loading`: delete the final `with pytest.raises(TypeError, match="bind_device"):` block (any launcher may now `bind_device`).
5. `test_bind_pointer_checks_once_and_warns_at_creation` → `..._warns_at_check`: right after `make_launcher(...)` add `assert caught == []  # decoration checks nothing` and `factory.__self__.bind(x=4096).__self__.check()`; the existing `assert len(caught) == int(verify)` follows.
6. `test_binding_warnings_are_per_parameter_at_factory_creation` → `..._at_check`: after `make_launcher(...)` add `assert caught == []`; move the message/category/filename assertions after the bind loop; keep `assert len(caught) == 2 * int(verify)` (the analysis warns once per `make_launcher` call).
7. `test_malformed_keyword_arguments_are_rejected_when_applied`: `decorate(scale)` → `decorate(scale).__self__.check()`.
8. The `KernelCache.TSL` provisioning test: `assert calls == []  # decoration provisions nothing`; `launch.__self__.check()`; `assert calls == [KernelCache.TSL]  # at check(), never on a call`.
9. `test_class_global_is_keyed_by_name_and_warns_once_per_build`: inside `pytest.warns`, `launch = make_launcher(...)` then `launch.__self__.check()`; the filterable line → `make_launcher(kernel, assume_constant_globals=True).__self__.check()  # filterable`.
10. `test_gluon_layout_class_global_launches`: `launch.__self__.check()` inside `pytest.warns`.
11. `test_decoration_needs_no_gpu`: inside the monkeypatch context, after `fixed = ...`, add `plain.__self__.check()` and `fixed.__self__.check()  # check() is GPU-free too`.
12. The bound-tensor refcount test: `bound.__self__.check()` before `references, native = ...` (the plan's binding holds the tensor from the first check on).
13. Every `inspect.signature(k)` on a launcher not yet called or checked: add `k.__self__.check()` first (an unchecked launcher has no `__text_signature__`; `inspect` raises `ValueError: no signature found`). Known sites: test_launcher `bind_device_accepts_every_bound_parameter_name`, `bind_device_signature_avoids_public_control_names`, `bound_launcher_is_fastcall_and_hides_bound_parameters`, `bind_uses_declaration_order...`, `bind_signature_control_names_avoid_kernel_names` (both signatures), the `factory.bind(o=out, x=tensor)` STATIC_COMPILE test, `constexpr_option_parameter_feeds_the_option`; test_grid's three signature tests.
14. Any further test that expected a decoration-time error: append `.__self__.check()` to the decoration.

- [ ] **Step 7: Run the suite, pyright and the matrix**

```bash
PYTHONPATH=$PWD $PY -m pytest tests -q
~/.local/bin/pyright
bash tests/run_python_matrix.sh 3.8 3.13t 3.14t
```
Expected: all pass; pyright 0 errors (fix test `reportAttributeAccessIssue` on `LaunchFunction` by adding the attribute to the Protocol only if intj code needs it, else by `getattr` in the test).

- [ ] **Step 8: Commit**

```bash
git add -A && git commit -m "launcher: MakeLauncherOption, check(), bind/unbind on the launcher object"
```

---

### Task 3: `delayed=`

**Files:**
- Modify: `intj/launcher.py` (`_Origin`, `Delayed`, `_merge_delayed`, `Launcher._check`/`_derive`, `make_launcher`)
- Create: `tests/test_delayed.py`

**Interfaces:**
- Consumes (Task 2): `_Origin`, `_Once`, `_new_launcher`, `Launcher._check`, `OPTION_FIELDS`.
- Produces: `make_launcher(..., delayed: Callable[[], Mapping[str, Any]] | None = None)`; `Delayed(origin, edits=())` with `__call__() -> MakeLauncherOption` and `then(edit) -> Delayed`; `_Origin.delayed`, `_Origin.merged: _Once[MakeLauncherOption]`.

- [ ] **Step 1: Write the failing tests (`tests/test_delayed.py`)**

```python
# pyright: standard
"""`make_launcher(delayed=...)`: options computed on first use."""

import dis
import gc
import sys
import sysconfig
import threading
import weakref

import pytest
import torch
import triton
import triton.language as tl

from intj import make_launcher
from intj.annotation import Argument, BindValue, NEVER
from intj.launcher import Delayed, UnsupportedKernel, module_of
from intj.torch_intf.torch_abi import TorchAccessMode


@triton.jit
def store(x, n):
    tl.store(x + tl.arange(0, 1), n)


_HOST = {"no_gpu": True, "torch_access_mode": TorchAccessMode.INTERPRETER}
_POINTER = Argument(
    type=tl.pointer_type(tl.float32), specialize=NEVER, bind_value=BindValue.POINTER
)


def _counting(values):
    calls = []

    def delayed():
        calls.append(1)
        return values

    return delayed, calls


def test_runs_once_on_first_use_across_bind_children():
    delayed, calls = _counting(_HOST)
    base = make_launcher(store, extra_annotation={"x": _POINTER}, delayed=delayed)
    assert calls == [] and isinstance(base.__self__.options, Delayed)
    children = [base.__self__.bind(x=0), base.__self__.bind_device(0, x=0)]
    assert children[0](0, 0, 1, 4) is None and children[1](0, 1, 4) is None
    assert calls == [1]
    assert children[0].__self__.options.no_gpu is True  # resolved in place


def test_decorator_form():
    delayed, calls = _counting(_HOST)
    launch = make_launcher(delayed=delayed)(store)
    assert calls == [] and launch(0, 0, 1, 0, 4) is None and calls == [1]


def test_conflict_and_bad_results():
    with pytest.raises(UnsupportedKernel, match=r"delayed returned \['no_gpu'\]"):
        make_launcher(store, no_gpu=True, delayed=lambda: _HOST).__self__.check()
    with pytest.raises(TypeError, match="must return a mapping"):
        make_launcher(store, delayed=lambda: [("no_gpu", True)]).__self__.check()
    for key in ("bogus", "jit_func", "delayed", 1):
        with pytest.raises(TypeError, match="unknown make_launcher keyword"):
            make_launcher(store, delayed=lambda key=key: {key: 1}).__self__.check()
    with pytest.raises(TypeError, match="delayed must be None or a callable"):
        make_launcher(store, delayed=_HOST)


def test_a_failure_is_not_cached():
    calls = []

    def delayed():
        calls.append(1)
        if len(calls) == 1:
            raise KeyError("first")
        return _HOST

    launch = make_launcher(store, delayed=delayed)
    with pytest.raises(KeyError, match="first"):  # unwrapped
        launch(0, 0, 1, 0, 4)
    assert launch(0, 0, 1, 0, 4) is None and len(calls) == 2


@pytest.mark.parametrize("siblings", [False, True])
def test_concurrent_first_calls_run_it_once(siblings):
    delayed, calls = _counting(_HOST)
    base = make_launcher(store, delayed=delayed)
    launchers = [base.__self__.bind() if siblings else base for _ in range(8)]
    barrier, errors = threading.Barrier(8), []

    def worker(k):
        barrier.wait()
        try:
            k(0, 0, 1, 0, 4)
        except BaseException as error:  # noqa: BLE001  report, do not hang
            errors.append(error)

    threads = [threading.Thread(target=worker, args=(k,)) for k in launchers]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == [] and calls == [1]


def test_results_enter_module_identity():
    def launcher(warps):
        return make_launcher(store, delayed=lambda: {"options": {"num_warps": warps}})

    four, eight, again = launcher(4), launcher(8), launcher(4)
    assert module_of(four) is not module_of(eight)
    assert module_of(four) is module_of(again)


def test_calling_itself_or_a_sibling_is_refused():
    seen = []
    box = {}

    def delayed():
        for call in (
            lambda: box["k"](0, 0, 1, 0, 4),
            lambda: box["k"].__self__.check(),
            lambda: box["sibling"].__self__.check(),
        ):
            try:
                call()
            except RuntimeError as error:
                seen.append(str(error))
        return _HOST

    box["k"] = make_launcher(store, delayed=delayed)
    box["sibling"] = box["k"].__self__.bind()
    assert box["k"](0, 0, 1, 0, 4) is None
    assert len(seen) == 3 and all("while it was being built" in s for s in seen)


@pytest.mark.parametrize("call_first", [False, True])
def test_a_cycle_through_the_closure_is_collected(call_first):
    class Owner:
        launch: object

    owner = Owner()
    # a default, not a free variable: `del owner` must not cut the cycle
    owner.launch = make_launcher(store, delayed=lambda owner=owner: (owner, _HOST)[1])
    if call_first:
        owner.launch(0, 0, 1, 0, 4)
    ref = weakref.ref(owner)
    del owner
    gc.collect()
    assert ref() is None


def test_a_delayed_launcher_keeps_the_builtin_fast_call():
    launch = make_launcher(store, delayed=lambda: _HOST)

    def call():
        return launch(0, 0, 1, 0, 4)

    for _ in range(100):
        assert call() is None
    free_threaded = bool(sysconfig.get_config_var("Py_GIL_DISABLED"))
    if sys.version_info >= (3, 11) and not (
        free_threaded and sys.version_info[:2] == (3, 13)
    ):
        names = {i.opname for i in dis.get_instructions(call, adaptive=True)}  # pyright: ignore[reportCallIssue]
        assert names & {
            "PRECALL_NO_KW_BUILTIN_FAST",
            "CALL_NO_KW_BUILTIN_FAST",
            "CALL_BUILTIN_FAST",
        }
```

(`torch` is imported so the GPU-backed `module_of` test runs where the rest of the suite does; drop it if unused.)

- [ ] **Step 2: Run them to verify they fail**

Run: `PYTHONPATH=$PWD $PY -m pytest tests/test_delayed.py -q`
Expected: FAIL at import (`cannot import name 'Delayed'`).

- [ ] **Step 3: Implement**

1. `_Origin` gains `delayed: Callable[[], object] | None = None` (after `names`) and `merged: _Once[MakeLauncherOption] = dataclasses.field(default_factory=_Once)`.
2. Add after `_Origin`:

```python
class Delayed:
    """A delayed launcher's `options` until `check()` resolves it.

    Calling it runs the `make_launcher(delayed=...)` function, once per
    `make_launcher` call, merges its keywords with the explicit ones, then
    applies this launcher's bind/unbind edits.
    """

    __slots__ = ("_origin", "_edits")

    def __init__(
        self,
        origin: _Origin,
        edits: tuple[Callable[[MakeLauncherOption], MakeLauncherOption], ...] = (),
    ) -> None:
        self._origin = origin
        self._edits = edits

    def __call__(self) -> MakeLauncherOption:
        origin = self._origin
        option = origin.merged.get(lambda: _merge_delayed(origin))
        for edit in self._edits:
            option = edit(option)
        return option

    def then(self, edit: Callable[[MakeLauncherOption], MakeLauncherOption]) -> Delayed:
        """Returns this `Delayed` with `edit` applied after the others."""
        return Delayed(self._origin, (*self._edits, edit))


def _merge_delayed(origin: _Origin) -> MakeLauncherOption:
    assert origin.delayed is not None
    values = origin.delayed()
    if not isinstance(values, Mapping):
        raise TypeError(
            "intj: delayed must return a mapping of make_launcher keywords, "
            f"got {type(values).__name__}"
        )
    values = cast("Mapping[object, object]", values)
    unknown = sorted(repr(k) for k in values if type(k) is not str or k not in OPTION_FIELDS)
    if unknown:
        raise TypeError(
            f"intj: delayed returned unknown make_launcher keyword(s) [{', '.join(unknown)}]"
        )
    keywords = cast("dict[str, Any]", dict(values))
    clash = sorted(k for k in keywords if k in origin.names)
    if clash:
        raise UnsupportedKernel(
            f"intj: delayed returned {clash}, also passed to make_launcher; "
            "pass each keyword once"
        )
    return dataclasses.replace(origin.explicit, **keywords)
```

3. `Launcher`: the `options` stub returns `MakeLauncherOption | Delayed`;

```python
    def _derive(self, edit: Callable[[MakeLauncherOption], MakeLauncherOption]) -> LaunchFunction:
        options = self.options
        derived = options.then(edit) if isinstance(options, Delayed) else edit(options)
        return _new_launcher(derived, self._origin)
```
   and in `_check`: `options = self.options; option = options() if isinstance(options, Delayed) else options`.
4. `_new_launcher(options: MakeLauncherOption | Delayed, origin: _Origin)`.
5. `make_launcher` gains `*, delayed: Callable[[], Mapping[str, Any]] | None = None` after `jit_func, /`; after the unknown-keyword check:

```python
    if delayed is not None and not callable(delayed):
        raise TypeError("intj: delayed must be None or a callable taking no arguments")
    if jit_func is _NO_KERNEL:
        return functools.partial(make_launcher, delayed=delayed, **kwargs)
    ...
    option = MakeLauncherOption(jit_func, **kwargs)
    origin = _Origin(option, frozenset(kwargs), delayed)
    return _new_launcher(option if delayed is None else Delayed(origin), origin)
```
   Docstring Args: `delayed: A function of no arguments returning more keywords, as a mapping. It runs once, on the first check() or call of any launcher this call makes; a keyword also passed here raises UnsupportedKernel.`

- [ ] **Step 4: Run, gate, commit**

```bash
PYTHONPATH=$PWD $PY -m pytest tests/test_delayed.py tests/test_launcher_object.py tests/test_runtime.py -q
PYTHONPATH=$PWD $PY -m pytest tests -q
~/.local/bin/pyright && ~/.local/bin/pre-commit run --all-files --show-diff-on-failure
git add -A && git commit -m "launcher: delayed= computes make_launcher keywords on first use"
```

---

### Task 4: Docs, CI, bench gate

**Files:** `README.md:81-101`, `docs/Usage.md`, `TODO.md:52-57`, `AGENTS.md:69,92,113`, `CONCETPS.md:31`, `intj/python_intf/AGENTS.md:121-123`

- [ ] **Step 1: README** — replace the `functools.cache` factory example and the "not implemented" paragraph with:

````markdown
```python
@make_launcher(delayed=lambda: {
    "options": {"num_warps": {"hip": 4, "cuda": 8}[
        triton.runtime.driver.active.get_current_target().backend]},
})
@triton.jit
def kernel(...): ...
```

`delayed` runs once, on the launcher's first call (or `kernel.__self__.check()`),
and its keywords join the ones passed to `make_launcher`; passing one keyword
both ways raises.
````

- [ ] **Step 2: `docs/Usage.md`**
  - Signature block: `bind_device=None` (an ordinal), add `bind_values=None`, `delayed=None`.
  - "always returns a launcher" replaces the factory sentence.
  - "Lazy build": decoration only records a `MakeLauncherOption`; `k.__self__.check()` runs the checks and provisioning; the first call checks then builds; `k.__self__` is an `intj.launcher.Launcher`; `inspect.signature(k)` needs a check or call first.
  - `With bind_device=True` → `With bind_device=<ordinal>`.
  - Autotune config values are checked at `check()`; `factory.bind` → `k.__self__.bind`.
  - Binding example: `launch = make_launcher(kernel, extra_annotation=...)`, `bound = launch.__self__.bind(x=tensor)`, `fixed = launch.__self__.bind_device(device, x=tensor)`, plus `unbind()`/`unbind_device()`.
  - `set_torch_version` → `_set_torch_version`; spec key "through `k.__self__.spec_key(device, *args)`".
- [ ] **Step 3:** TODO.md: drop the "Deferred `make_launcher` arguments" bullet. AGENTS.md: 69 → "the `traverse`/`release` ops that the module's `_init` installs"; 92 → "a launcher's `check()` provisions on demand"; 113 → "preferably in `check()` (at the latest on the first call)". CONCETPS.md:31 → `_set_torch_version()`. python_intf/AGENTS.md → "`intj_rwlock` (in `runtime/intj_lazy.h`), held in each launcher's state block that the module's `_init` allocates, ...".
- [ ] **Step 4: CI**

```bash
PYTHON=$PY ninja -C ci && PYTHON=$PY ninja -C ci test-gpu
bash tests/run_python_matrix.sh
```

- [ ] **Step 5: Bench gate**

```bash
git worktree add /tmp/intj-base 9a84e42
for tree in /tmp/intj-base $PWD; do
  (cd $tree && taskset -c 0 env PYTHONPATH=$tree $PY benchmarks/bench_launch.py --readme && taskset -c 0 env PYTHONPATH=$tree $PY benchmarks/bench_launch.py --last-key)
done
git worktree remove /tmp/intj-base
```
Expected: zero-grid and last-key rows within +1 ns of the base. If over: first add a module-state pointer to the object next to `state` (loaded in parallel); second, the `m_self` swap (spec, "Cost and fallback").

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "docs: delayed=, check(), and methods on the launcher object"
```

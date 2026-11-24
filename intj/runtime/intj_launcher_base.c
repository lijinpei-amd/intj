/* `_intj_launcher_base`: the type every launcher's object comes from, and
 * the shim its PyCFunction starts on. No kernel code or GPU, so a launcher
 * exists before its module is rendered.  The first call builds: the
 * subclass's `_build_body` renders and loads the module, whose `_init`
 * allocates and fills the launcher's state block and returns its entry; the
 * shim swaps def.ml_meth to that entry.  The rest of the Python side runs
 * under the same build lock through `_C_locked`.  CPython reads
 * m_ml->ml_meth on every call (tests/test_runtime.py pins it, 3.8 to 3.14t),
 * so references taken before the build reach the entry as well. */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <pthread.h>
#include <string.h>

#include "intj_launcher_base.h"

typedef struct {
  PyTypeObject *type;
} intj_launcher_base_state;

static PyObject *intj_launcher_base_call(PyObject *self, PyObject *const *args,
                                         Py_ssize_t nargs);
#define INTJ_SHIM ((PyCFunction)(void (*)(void))intj_launcher_base_call)

/* Build locks this thread holds: a build (`delayed`) runs Python under one,
 * and may reach another launcher. */
static _Thread_local int intj_launcher_base_held;

/* The wait-for graph between build locks, under intj_launcher_base_graph.  A
 * launcher's edge is its `build_thread`.  A thread's is a record on its stack
 * while it waits on a build lock holding another; one holding none closes no
 * cycle.  An edge is only added if it closes no cycle, so the graph stays
 * acyclic and a walk along it ends. */
typedef struct intj_launcher_base_wait {
  unsigned long thread;
  const intj_launcher *on;
  struct intj_launcher_base_wait *next;
} intj_launcher_base_wait;

static pthread_mutex_t intj_launcher_base_graph = PTHREAD_MUTEX_INITIALIZER;
static intj_launcher_base_wait *intj_launcher_base_waits;

/* Under the graph lock: whether `me` waiting on `h` closes a cycle, as when
 * h's builder waits on a lock whose builder ... waits on one `me` holds. */
static int intj_launcher_base_closes_cycle(const intj_launcher *h,
                                           unsigned long me) {
  for (;;) {
    unsigned long owner = h->build_thread;
    if (owner == me)
      return 1;
    const intj_launcher_base_wait *w = intj_launcher_base_waits;
    while (w && w->thread != owner)
      w = w->next;
    if (!w)
      return 0;
    h = w->on;
  }
}

/* Take the build lock: 0, or -1 with an error set.  Two refusals, never a
 * wait that would deadlock: the thread holding this lock (a launcher called or
 * checked from its own build), and a wait that closes a cycle of threads each
 * holding a build lock and waiting on the next one's.  Any other caller
 * waits. */
static int intj_launcher_base_lock(intj_launcher *h) {
  unsigned long me = PyThread_get_thread_ident();
  if (__atomic_load_n(&h->build_thread, __ATOMIC_RELAXED) == me) {
    PyErr_SetString(PyExc_RuntimeError,
                    "intj: a launcher was called while it was being built");
    return -1;
  }
  intj_launcher_base_wait wait = {me, h, NULL};
  int in_graph = 0;
  if (!PyThread_acquire_lock(h->build_lock, NOWAIT_LOCK)) {
    if (intj_launcher_base_held > 0) {
      pthread_mutex_lock(&intj_launcher_base_graph);
      int cycle = intj_launcher_base_closes_cycle(h, me);
      if (!cycle) {
        wait.next = intj_launcher_base_waits;
        intj_launcher_base_waits = &wait;
        in_graph = 1;
      }
      pthread_mutex_unlock(&intj_launcher_base_graph);
      if (cycle) {
        PyErr_SetString(PyExc_RuntimeError,
                        "intj: a launcher's build lock is held by a thread "
                        "waiting on this one's: waiting would deadlock");
        return -1;
      }
    }
    /* detached while waiting: the build runs Python, and a free-threaded
     * stop-the-world must not wait on us */
    // clang-format off
    Py_BEGIN_ALLOW_THREADS
    PyThread_acquire_lock(h->build_lock, WAIT_LOCK);
    Py_END_ALLOW_THREADS
    // clang-format on
  }
  intj_launcher_base_held++;
  pthread_mutex_lock(&intj_launcher_base_graph);
  if (in_graph) {
    intj_launcher_base_wait **p = &intj_launcher_base_waits;
    while (*p != &wait)
      p = &(*p)->next;
    *p = wait.next;
  }
  __atomic_store_n(&h->build_thread, me, __ATOMIC_RELAXED);
  pthread_mutex_unlock(&intj_launcher_base_graph);
  return 0;
}

static void intj_launcher_base_unlock(intj_launcher *h) {
  intj_launcher_base_held--;
  /* cleared before the release, so it never clears the next builder's */
  pthread_mutex_lock(&intj_launcher_base_graph);
  __atomic_store_n(&h->build_thread, 0UL, __ATOMIC_RELAXED);
  pthread_mutex_unlock(&intj_launcher_base_graph);
  PyThread_release_lock(h->build_lock);
}

/* Run `self._build_body()` once: 0, or -1 with an error set.  It returns
 * (entry address, resolved options, doc); they are published, then the entry
 * swapped in.  A failure leaves the shim and unpublishes what `_init` may
 * have published before it (an exception after `_init` returned, a malformed
 * result), so the next call retries from an unbuilt launcher. */
static int intj_launcher_base_build(intj_launcher *h) {
  if (__atomic_load_n(&h->def.ml_meth, __ATOMIC_ACQUIRE) != INTJ_SHIM)
    return 0;
  if (intj_launcher_base_lock(h) != 0)
    return -1;
  int rc = 0;
  PyObject *old = NULL;
  void *state = NULL;
  const intj_launcher_ops *ops = NULL;
  if (__atomic_load_n(&h->def.ml_meth, __ATOMIC_ACQUIRE) == INTJ_SHIM) {
    if (!h->options) {
      PyErr_SetString(PyExc_RuntimeError, "intj: launcher has been cleared");
      rc = -1;
    } else {
      PyObject *result =
          PyObject_CallMethod((PyObject *)h, "_build_body", NULL);
      void *address = NULL;
      if (result &&
          (!PyTuple_CheckExact(result) || PyTuple_GET_SIZE(result) != 3 ||
           !PyBytes_CheckExact(PyTuple_GET_ITEM(result, 2))))
        PyErr_SetString(PyExc_TypeError,
                        "intj: _build_body returns (entry, options, doc: "
                        "bytes)");
      else if (result)
        address = PyLong_AsVoidPtr(PyTuple_GET_ITEM(result, 0));
      if (!address) {
        if (!PyErr_Occurred())
          PyErr_SetString(PyExc_RuntimeError,
                          "intj: _build_body returned no entry");
        rc = -1;
        ops = h->ops;
        state = __atomic_exchange_n(&h->state, NULL, __ATOMIC_ACQ_REL);
        Py_BEGIN_CRITICAL_SECTION(h);
        old = h->module; /* dropped with the state, after the unlock */
        h->module = NULL;
        Py_END_CRITICAL_SECTION();
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
      }
      Py_XDECREF(result);
    }
  }
  intj_launcher_base_unlock(h);
  /* outside the lock: finalizers run Python */
  if (state)
    ops->release(state);
  Py_XDECREF(old);
  return rc;
}

static PyObject *intj_launcher_base_call(PyObject *self, PyObject *const *args,
                                         Py_ssize_t nargs) {
  intj_launcher *h = (intj_launcher *)self;
  if (intj_launcher_base_build(h) != 0)
    return NULL;
  PyCFunction entry = __atomic_load_n(&h->def.ml_meth, __ATOMIC_ACQUIRE);
  return ((intj_fast_fn)(void (*)(void))entry)(self, args, nargs);
}

/* _C_locked(fn): fn() under the build lock, the one a first call builds
 * under; fn's result or error.  The lock refuses fn reaching this launcher's
 * own _C_locked or first call.  fn's result is released after the unlock, so
 * returning what it replaced keeps finalizers outside the lock. */
static PyObject *intj_launcher_base_locked(PyObject *self, PyObject *fn) {
  intj_launcher *h = (intj_launcher *)self;
  if (intj_launcher_base_lock(h) != 0)
    return NULL;
  PyObject *result;
  if (!h->options) {
    PyErr_SetString(PyExc_RuntimeError, "intj: launcher has been cleared");
    result = NULL;
  } else {
    result = PyObject_CallFunctionObjArgs(fn, NULL);
  }
  intj_launcher_base_unlock(h);
  return result;
}

static PyObject *intj_launcher_base_build_method(PyObject *self,
                                                 PyObject *unused) {
  (void)unused;
  intj_launcher *h = (intj_launcher *)self;
  if (intj_launcher_base_build(h) != 0)
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
static void *intj_launcher_base_built_state(intj_launcher *h) {
  if (intj_launcher_base_build(h) != 0)
    return NULL;
  void *state = __atomic_load_n(&h->state, __ATOMIC_ACQUIRE);
  if (!state)
    PyErr_SetString(PyExc_RuntimeError, "intj: launcher has been cleared");
  return state;
}

static PyObject *intj_launcher_base_spec_key(PyObject *self,
                                             PyObject *const *args,
                                             Py_ssize_t nargs) {
  intj_launcher *h = (intj_launcher *)self;
  void *state = intj_launcher_base_built_state(h);
  return state ? h->ops->spec_key(state, args, nargs) : NULL;
}

static PyObject *intj_launcher_base_key_chain(PyObject *self,
                                              PyObject *const *args,
                                              Py_ssize_t nargs) {
  intj_launcher *h = (intj_launcher *)self;
  void *state = intj_launcher_base_built_state(h);
  return state ? h->ops->key_chain(state, args, nargs) : NULL;
}

/* _C_set_options(options): only from inside _C_locked on this thread, so
 * edits and a first call are ordered; refused once built. */
static PyObject *intj_launcher_base_set_options(PyObject *self,
                                                PyObject *options) {
  intj_launcher *h = (intj_launcher *)self;
  if (__atomic_load_n(&h->build_thread, __ATOMIC_RELAXED) !=
      PyThread_get_thread_ident()) {
    PyErr_SetString(PyExc_RuntimeError,
                    "intj: _C_set_options needs this launcher's build lock "
                    "(_C_locked)");
    return NULL;
  }
  if (__atomic_load_n(&h->def.ml_meth, __ATOMIC_ACQUIRE) != INTJ_SHIM) {
    PyErr_SetString(PyExc_RuntimeError,
                    "intj: a built launcher's options are fixed");
    return NULL;
  }
  PyObject *old;
  Py_INCREF(options);
  Py_BEGIN_CRITICAL_SECTION(self);
  old = h->options;
  h->options = options;
  Py_END_CRITICAL_SECTION();
  Py_XDECREF(old); /* _C_locked's fn holds it, to drop after the unlock */
  Py_RETURN_NONE;
}

/* _C_built: whether the first call has swapped the entry in. */
static PyObject *intj_launcher_base_built(PyObject *self, void *closure) {
  (void)closure;
  intj_launcher *h = (intj_launcher *)self;
  return PyBool_FromLong(__atomic_load_n(&h->def.ml_meth, __ATOMIC_ACQUIRE) !=
                         INTJ_SHIM);
}

#define INTJ_MEMBER(field) ((void *)offsetof(intj_launcher, field))

/* An object member, or None while it is NULL. */
static PyObject *intj_launcher_base_member(PyObject *self, void *closure) {
  PyObject *value;
  Py_BEGIN_CRITICAL_SECTION(self);
  value = *(PyObject **)((char *)self + (uintptr_t)closure);
  value = value ? value : Py_None;
  Py_INCREF(value);
  Py_END_CRITICAL_SECTION();
  return value;
}

static int intj_launcher_base_traverse(PyObject *self, visitproc visit,
                                       void *arg) {
  intj_launcher *h = (intj_launcher *)self;
#if PY_VERSION_HEX < 0x03090000
  /* 3.8's subtype_traverse visits a Python subclass's type itself */
  if (Py_TYPE(self)->tp_traverse == intj_launcher_base_traverse)
#endif
    Py_VISIT(Py_TYPE(self));
  Py_VISIT(h->options);
  Py_VISIT(h->module);
  /* the module's hook visits what `_init` stored in the state block */
  return h->state ? h->ops->traverse(h->state, visit, arg) : 0;
}

static int intj_launcher_base_clear(PyObject *self) {
  intj_launcher *h = (intj_launcher *)self;
  /* unpublish first: release drops owners whose finalizers can reenter Python */
  void *state = h->state;
  h->state = NULL;
  if (state)
    h->ops->release(state);
  Py_CLEAR(h->options);
  Py_CLEAR(h->module);
  return 0;
}

static void intj_launcher_base_dealloc(PyObject *self) {
  intj_launcher *h = (intj_launcher *)self;
  PyTypeObject *type = Py_TYPE(self);
  PyObject_GC_UnTrack(self);
  intj_launcher_base_clear(self);
  Py_CLEAR(h->doc); /* only here: a live PyCFunction may still read ml_doc */
  if (h->build_lock)
    PyThread_free_lock(h->build_lock);
  type->tp_free(self);
  Py_DECREF(type);
}

static PyObject *intj_launcher_base_new(PyTypeObject *type, PyObject *args,
                                        PyObject *kwargs) {
  (void)type;
  (void)args;
  (void)kwargs;
  PyErr_SetString(PyExc_TypeError, "intj: launchers come from make_launcher");
  return NULL;
}

/* `_C_` marks every attribute and method this type defines;
 * intj.launcher.Launcher exposes the public ones as `options`, `build()`,
 * `spec_key()` and `key_chain()`, and runs `check()` and its `_foo` edits
 * under `_C_locked`.  It defines `_build_body`, which the first call runs. */
static PyGetSetDef intj_launcher_base_getset[] = {
    {(char *)"_C_options", intj_launcher_base_member, NULL, NULL,
     INTJ_MEMBER(options)},
    {(char *)"_C_module", intj_launcher_base_member, NULL, NULL,
     INTJ_MEMBER(module)},
    {(char *)"_C_built", intj_launcher_base_built, NULL, NULL, NULL},
    {NULL, NULL, NULL, NULL, NULL}};

static PyMethodDef intj_launcher_base_type_methods[] = {
    {"_C_build", intj_launcher_base_build_method, METH_NOARGS, NULL},
    {"_C_spec_key", (PyCFunction)(void (*)(void))intj_launcher_base_spec_key,
     METH_FASTCALL, NULL},
    {"_C_key_chain", (PyCFunction)(void (*)(void))intj_launcher_base_key_chain,
     METH_FASTCALL, NULL},
    {"_C_locked", intj_launcher_base_locked, METH_O, NULL},
    {"_C_set_options", intj_launcher_base_set_options, METH_O, NULL},
    {NULL, NULL, 0, NULL}};

static PyType_Slot intj_launcher_base_slots[] = {
    {Py_tp_new, (void *)intj_launcher_base_new},
    {Py_tp_traverse, (void *)intj_launcher_base_traverse},
    {Py_tp_clear, (void *)intj_launcher_base_clear},
    {Py_tp_dealloc, (void *)intj_launcher_base_dealloc},
    {Py_tp_getset, (void *)intj_launcher_base_getset},
    {Py_tp_methods, (void *)intj_launcher_base_type_methods},
    {0, NULL}};

static PyType_Spec intj_launcher_base_spec = {
    "_intj_launcher_base.Launcher", sizeof(intj_launcher), 0,
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC | Py_TPFLAGS_BASETYPE
#ifdef Py_TPFLAGS_IMMUTABLETYPE
        | Py_TPFLAGS_IMMUTABLETYPE
#endif
    ,
    intj_launcher_base_slots};

/* new_launcher(cls, options) -> an unbuilt launcher: a PyCFunction over a
 * `cls` instance, with intj_launcher as its C base; Python subclasses may add
 * storage.  The first call runs its `_build_body`. */
static PyObject *new_launcher(PyObject *m, PyObject *const *args,
                              Py_ssize_t nargs) {
  intj_launcher_base_state *st =
      (intj_launcher_base_state *)PyModule_GetState(m);
  PyTypeObject *cls =
      nargs == 2 && PyType_Check(args[0]) ? (PyTypeObject *)args[0] : NULL;
  if (!cls || !PyType_IsSubtype(cls, st->type) || cls->tp_itemsize != 0) {
    PyErr_SetString(PyExc_TypeError, "intj: new_launcher(cls, options)");
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
  h->def.ml_name = "launch";
  h->def.ml_meth = INTJ_SHIM;
  h->def.ml_flags = METH_FASTCALL;
  h->def.ml_doc = NULL; /* the build sets the signature */
  PyObject *launcher = PyCFunction_New(&h->def, (PyObject *)h);
  Py_DECREF(h);
  return launcher;
}

/* The host backend's driver (intj.backend.HostBackend), for `no_gpu=True`: a
 * module dlsyms these from this library as it would hipModuleLaunchKernel and
 * hipDeviceGet from libamdhip64.  Every launch succeeds and does nothing. */
#define INTJ_EXPORT __attribute__((visibility("default")))

INTJ_EXPORT int32_t intj_host_launch(void *f, uint32_t gx, uint32_t gy,
                                     uint32_t gz, uint32_t bx, uint32_t by,
                                     uint32_t bz, uint32_t shared, void *stream,
                                     void **params, void **extra) {
  (void)f, (void)gx, (void)gy, (void)gz, (void)bx, (void)by, (void)bz;
  (void)shared, (void)stream, (void)params, (void)extra;
  return 0;
}

INTJ_EXPORT int32_t intj_host_device_get(int32_t *device, int32_t ordinal) {
  *device = ordinal;
  return 0;
}

INTJ_EXPORT const char *intj_host_error_string(int32_t status) {
  (void)status;
  return "intj host backend error";
}

static PyObject *header_size(PyObject *m, PyObject *unused) {
  (void)m;
  (void)unused;
  return PyLong_FromSize_t(sizeof(intj_launcher));
}

static int intj_launcher_base_exec(PyObject *m) {
  intj_launcher_base_state *st =
      (intj_launcher_base_state *)PyModule_GetState(m);
#if PY_VERSION_HEX >= 0x03090000
  st->type = (PyTypeObject *)PyType_FromModuleAndSpec(
      m, &intj_launcher_base_spec, NULL);
#else
  st->type = (PyTypeObject *)PyType_FromSpec(&intj_launcher_base_spec);
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

static int intj_launcher_base_module_traverse(PyObject *m, visitproc visit,
                                              void *arg) {
  Py_VISIT(((intj_launcher_base_state *)PyModule_GetState(m))->type);
  return 0;
}

static int intj_launcher_base_module_clear(PyObject *m) {
  Py_CLEAR(((intj_launcher_base_state *)PyModule_GetState(m))->type);
  return 0;
}

static PyMethodDef intj_launcher_base_methods[] = {
    {"new_launcher", (PyCFunction)(void (*)(void))new_launcher, METH_FASTCALL,
     "new_launcher(cls, options) -> launcher"},
    {"_header_size", header_size, METH_NOARGS, NULL},
    {NULL, NULL, 0, NULL}};

static PyModuleDef_Slot intj_launcher_base_module_slots[] = {
    {Py_mod_exec, (void *)intj_launcher_base_exec},
#ifdef Py_mod_multiple_interpreters
    {Py_mod_multiple_interpreters, Py_MOD_MULTIPLE_INTERPRETERS_NOT_SUPPORTED},
#endif
#ifdef Py_mod_gil
    {Py_mod_gil, Py_MOD_GIL_NOT_USED},
#endif
    {0, NULL}};

static struct PyModuleDef intj_launcher_base_module_def = {
    PyModuleDef_HEAD_INIT,
    "_intj_launcher_base",
    NULL,
    sizeof(intj_launcher_base_state),
    intj_launcher_base_methods,
    intj_launcher_base_module_slots,
    intj_launcher_base_module_traverse,
    intj_launcher_base_module_clear,
    NULL};

PyMODINIT_FUNC PyInit__intj_launcher_base(void) {
  return PyModuleDef_Init(&intj_launcher_base_module_def);
}

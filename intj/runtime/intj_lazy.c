/* `_intj_lazy`: the type every launcher's object comes from, and the shim its
 * PyCFunction starts on.  Generic -- no kernel code, no GPU -- so a launcher
 * exists before its module is rendered.  The first call builds: the
 * subclass's `_build_body` renders and loads the module, whose `_init`
 * allocates and fills the launcher's state block and returns its entry; the
 * shim swaps def.ml_meth to that entry.  The rest of the Python side runs
 * under the same build lock through `_C_locked`.  CPython reads
 * m_ml->ml_meth on every call (tests/test_runtime.py pins it, 3.8 to 3.14t),
 * so references taken before the build reach the entry as well. */
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

/* Build locks this thread holds: a build (`delayed`) runs Python under one,
 * and may reach another launcher. */
static _Thread_local int intj_lazy_held;

/* Take the build lock: 0, or -1 with an error set.  Two refusals, never a
 * wait that could deadlock: the thread holding this lock (a launcher called or
 * checked from its own build), and a thread already holding another launcher's
 * lock when this one is taken (two threads each holding one and waiting on the
 * other's would wait forever).  A thread holding none waits. */
static int intj_lazy_lock(intj_launcher *h) {
  unsigned long me = PyThread_get_thread_ident();
  if (__atomic_load_n(&h->build_thread, __ATOMIC_RELAXED) == me) {
    PyErr_SetString(PyExc_RuntimeError,
                    "intj: a launcher was called while it was being built");
    return -1;
  }
  if (!PyThread_acquire_lock(h->build_lock, NOWAIT_LOCK)) {
    if (intj_lazy_held > 0) {
      PyErr_SetString(PyExc_RuntimeError,
                      "intj: a launcher was called while it was being built");
      return -1;
    }
    /* detached while waiting: the build runs Python, and a free-threaded
     * stop-the-world must not wait on us */
    // clang-format off
    Py_BEGIN_ALLOW_THREADS
    PyThread_acquire_lock(h->build_lock, WAIT_LOCK);
    Py_END_ALLOW_THREADS
    // clang-format on
  }
  intj_lazy_held++;
  __atomic_store_n(&h->build_thread, me, __ATOMIC_RELAXED);
  return 0;
}

static void intj_lazy_unlock(intj_launcher *h) {
  intj_lazy_held--;
  __atomic_store_n(&h->build_thread, 0UL, __ATOMIC_RELAXED);
  PyThread_release_lock(h->build_lock);
}

/* Run `self._build_body()` once: 0, or -1 with an error set.  It returns
 * (entry address, resolved options, doc); they are published, then the entry
 * swapped in.  A failure publishes nothing and leaves the shim, so the next
 * call retries. */
static int intj_lazy_build(intj_launcher *h) {
  if (__atomic_load_n(&h->def.ml_meth, __ATOMIC_ACQUIRE) != INTJ_SHIM)
    return 0;
  if (intj_lazy_lock(h) != 0)
    return -1;
  int rc = 0;
  PyObject *old = NULL;
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
  intj_lazy_unlock(h);
  Py_XDECREF(old); /* outside the lock: finalizers run Python */
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

/* _C_locked(fn): fn() under the build lock, the one a first call builds
 * under; fn's result or error.  The lock refuses fn reaching this launcher's
 * own _C_locked or first call.  fn's result is released after the unlock, so
 * returning what it replaced keeps finalizers outside the lock. */
static PyObject *intj_lazy_locked(PyObject *self, PyObject *fn) {
  intj_launcher *h = (intj_launcher *)self;
  if (intj_lazy_lock(h) != 0)
    return NULL;
  PyObject *result;
  if (!h->options) {
    PyErr_SetString(PyExc_RuntimeError, "intj: launcher has been cleared");
    result = NULL;
  } else {
    result = PyObject_CallFunctionObjArgs(fn, NULL);
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

static PyObject *intj_lazy_compile_state(PyObject *self, void *closure) {
  (void)closure;
  intj_launcher *h = (intj_launcher *)self;
  /* _init publishes this before the build shim swaps the entry.  Reading it
   * from _compile_miss must not try to acquire the build lock again. */
  void *state = __atomic_load_n(&h->state, __ATOMIC_ACQUIRE);
  if (!state) {
    PyErr_SetString(PyExc_RuntimeError, "intj: launcher has no compile state");
    return NULL;
  }
  return h->ops->compile_state(state);
}

/* _C_set_options(options): only from inside _C_locked on this thread, so
 * edits and a first call are ordered; refused once built. */
static PyObject *intj_lazy_set_options(PyObject *self, PyObject *options) {
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
static PyObject *intj_lazy_built(PyObject *self, void *closure) {
  (void)closure;
  intj_launcher *h = (intj_launcher *)self;
  return PyBool_FromLong(__atomic_load_n(&h->def.ml_meth, __ATOMIC_ACQUIRE) !=
                         INTJ_SHIM);
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

/* `_C_` marks every attribute and method this type defines;
 * intj.launcher.Launcher exposes the public ones as `options`, `build()`,
 * `spec_key()` and `key_chain()`, and runs `check()` and its `_foo` edits
 * under `_C_locked`.  It defines `_build_body`, which the first call runs. */
static PyGetSetDef intj_lazy_getset[] = {
    {(char *)"_C_options", intj_lazy_member, NULL, NULL, INTJ_MEMBER(options)},
    {(char *)"_C_module", intj_lazy_member, NULL, NULL, INTJ_MEMBER(module)},
    {(char *)"_C_compile_state", intj_lazy_compile_state, NULL, NULL, NULL},
    {(char *)"_C_built", intj_lazy_built, NULL, NULL, NULL},
    {NULL, NULL, NULL, NULL, NULL}};

static PyMethodDef intj_lazy_type_methods[] = {
    {"_C_build", intj_lazy_build_method, METH_NOARGS, NULL},
    {"_C_spec_key", (PyCFunction)(void (*)(void))intj_lazy_spec_key,
     METH_FASTCALL, NULL},
    {"_C_key_chain", (PyCFunction)(void (*)(void))intj_lazy_key_chain,
     METH_FASTCALL, NULL},
    {"_C_locked", intj_lazy_locked, METH_O, NULL},
    {"_C_set_options", intj_lazy_set_options, METH_O, NULL},
    {NULL, NULL, 0, NULL}};

static PyType_Slot intj_lazy_slots[] = {
    {Py_tp_new, (void *)intj_lazy_new},
    {Py_tp_traverse, (void *)intj_lazy_traverse},
    {Py_tp_clear, (void *)intj_lazy_clear},
    {Py_tp_dealloc, (void *)intj_lazy_dealloc},
    {Py_tp_getset, (void *)intj_lazy_getset},
    {Py_tp_methods, (void *)intj_lazy_type_methods},
    {0, NULL}};

static PyType_Spec intj_lazy_spec = {
    "_intj_lazy.Launcher", sizeof(intj_launcher), 0,
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC | Py_TPFLAGS_BASETYPE
#ifdef Py_TPFLAGS_IMMUTABLETYPE
        | Py_TPFLAGS_IMMUTABLETYPE
#endif
    ,
    intj_lazy_slots};

/* new_launcher(cls, options) -> an unbuilt launcher: a PyCFunction over a
 * `cls` instance, `cls` being Launcher or a subclass that adds no storage; the
 * first call runs its `_build_body`. */
static PyObject *new_launcher(PyObject *m, PyObject *const *args,
                              Py_ssize_t nargs) {
  intj_lazy_state *st = (intj_lazy_state *)PyModule_GetState(m);
  PyTypeObject *cls =
      nargs == 2 && PyType_Check(args[0]) ? (PyTypeObject *)args[0] : NULL;
  if (!cls || !PyType_IsSubtype(cls, st->type) ||
      cls->tp_basicsize != (Py_ssize_t)sizeof(intj_launcher) ||
      cls->tp_itemsize != 0) {
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

static int intj_lazy_module_traverse(PyObject *m, visitproc visit, void *arg) {
  Py_VISIT(((intj_lazy_state *)PyModule_GetState(m))->type);
  return 0;
}

static int intj_lazy_module_clear(PyObject *m) {
  Py_CLEAR(((intj_lazy_state *)PyModule_GetState(m))->type);
  return 0;
}

static PyMethodDef intj_lazy_methods[] = {
    {"new_launcher", (PyCFunction)(void (*)(void))new_launcher, METH_FASTCALL,
     "new_launcher(cls, options) -> launcher"},
    {"_header_size", header_size, METH_NOARGS, NULL},
    {NULL, NULL, 0, NULL}};

static PyModuleDef_Slot intj_lazy_module_slots[] = {
    {Py_mod_exec, (void *)intj_lazy_exec},
#ifdef Py_mod_multiple_interpreters
    {Py_mod_multiple_interpreters, Py_MOD_MULTIPLE_INTERPRETERS_NOT_SUPPORTED},
#endif
#ifdef Py_mod_gil
    {Py_mod_gil, Py_MOD_GIL_NOT_USED},
#endif
    {0, NULL}};

static struct PyModuleDef intj_lazy_module_def = {
    PyModuleDef_HEAD_INIT,     "_intj_lazy",           NULL,
    sizeof(intj_lazy_state),   intj_lazy_methods,      intj_lazy_module_slots,
    intj_lazy_module_traverse, intj_lazy_module_clear, NULL};

PyMODINIT_FUNC PyInit__intj_lazy(void) {
  return PyModuleDef_Init(&intj_lazy_module_def);
}

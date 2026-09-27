/* `_intj_lazy`: the type every launcher is allocated from, and the shim its
 * PyCFunction starts on.  Generic -- no kernel code, no GPU -- so a launcher
 * exists before its module is rendered.  The first call builds: the builder
 * renders and loads the module, whose init_bound fills this header in place
 * and returns its entry; the shim swaps def.ml_meth to that entry.  CPython
 * reads m_ml->ml_meth on every call (tests/test_runtime.py pins it, 3.8 to
 * 3.14t), so references taken before the build reach the entry as well. */
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

/* Run the builder once, under the header's lock: 0, or -1 with an error set.
 * A failure leaves the shim in place, so the next call retries. */
static int intj_lazy_build(intj_bound_header *h) {
  unsigned long me = PyThread_get_thread_ident();
  if (__atomic_load_n(&h->build_thread, __ATOMIC_RELAXED) == me) {
    PyErr_SetString(PyExc_RuntimeError,
                    "intj: a launcher was called while it was being built");
    return -1;
  }
  if (!PyThread_acquire_lock(h->build_lock, NOWAIT_LOCK)) {
    /* detached while waiting: the builder runs Python, and a free-threaded
     * stop-the-world must not wait on us */
    Py_BEGIN_ALLOW_THREADS PyThread_acquire_lock(h->build_lock, WAIT_LOCK);
    Py_END_ALLOW_THREADS
  }
  int rc = 0;
  PyObject *done = NULL;
  if (__atomic_load_n(&h->def.ml_meth, __ATOMIC_ACQUIRE) == INTJ_SHIM) {
    if (!h->builder) {
      PyErr_SetString(PyExc_RuntimeError, "intj: launcher has been cleared");
      rc = -1;
    } else {
      __atomic_store_n(&h->build_thread, me, __ATOMIC_RELAXED);
      PyObject *entry =
          PyObject_CallFunctionObjArgs(h->builder, (PyObject *)h, NULL);
      __atomic_store_n(&h->build_thread, 0UL, __ATOMIC_RELAXED);
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
  PyThread_release_lock(h->build_lock);
  Py_XDECREF(done); /* outside the lock: the closure's finalizers run Python */
  return rc;
}

static PyObject *intj_lazy_call(PyObject *self, PyObject *const *args,
                                Py_ssize_t nargs) {
  intj_bound_header *h = (intj_bound_header *)self;
  if (intj_lazy_build(h) != 0)
    return NULL;
  PyCFunction entry = __atomic_load_n(&h->def.ml_meth, __ATOMIC_ACQUIRE);
  return ((intj_fast_fn)(void (*)(void))entry)(self, args, nargs);
}

static PyObject *intj_lazy_build_method(PyObject *self, PyObject *unused) {
  (void)unused;
  intj_bound_header *h = (intj_bound_header *)self;
  if (intj_lazy_build(h) != 0)
    return NULL;
  if (!h->module) {
    PyErr_SetString(PyExc_RuntimeError, "intj: launcher has been cleared");
    return NULL;
  }
  Py_INCREF(h->module);
  return h->module;
}

static PyObject *intj_lazy_module(PyObject *self, void *closure) {
  (void)closure;
  PyObject *module = ((intj_bound_header *)self)->module;
  module = module ? module : Py_None;
  Py_INCREF(module);
  return module;
}

static int intj_lazy_traverse(PyObject *self, visitproc visit, void *arg) {
  intj_bound_header *h = (intj_bound_header *)self;
  Py_VISIT(Py_TYPE(self));
  Py_VISIT(h->builder);
  Py_VISIT(h->module);
  /* the module's hook visits what init_bound stored; never these two again */
  return h->traverse ? h->traverse(self, visit, arg) : 0;
}

static int intj_lazy_clear(PyObject *self) {
  intj_bound_header *h = (intj_bound_header *)self;
  if (h->clear)
    h->clear(self); /* marks the launcher closed first, then drops its state */
  Py_CLEAR(h->builder);
  Py_CLEAR(h->module);
  return 0;
}

static void intj_lazy_dealloc(PyObject *self) {
  intj_bound_header *h = (intj_bound_header *)self;
  PyTypeObject *type = Py_TYPE(self);
  PyObject_GC_UnTrack(self);
  intj_lazy_clear(self);
  Py_CLEAR(h->doc); /* only here: a live PyCFunction may still read ml_doc */
  if (h->build_lock)
    PyThread_free_lock(h->build_lock);
  INTJ_RWLOCK_DESTROY(&h->lock);
  type->tp_free(self);
  Py_DECREF(type);
}

static PyGetSetDef intj_lazy_getset[] = {
    {(char *)"__self__", intj_lazy_module, NULL, NULL, NULL},
    {NULL, NULL, NULL, NULL, NULL}};

static PyMethodDef intj_lazy_type_methods[] = {
    {"build", intj_lazy_build_method, METH_NOARGS,
     "Build now if the first call has not, and return the rendered module."},
    {NULL, NULL, 0, NULL}};

static PyType_Slot intj_lazy_slots[] = {
    {Py_tp_traverse, (void *)intj_lazy_traverse},
    {Py_tp_clear, (void *)intj_lazy_clear},
    {Py_tp_dealloc, (void *)intj_lazy_dealloc},
    {Py_tp_getset, (void *)intj_lazy_getset},
    {Py_tp_methods, (void *)intj_lazy_type_methods},
    {0, NULL}};

static PyType_Spec intj_lazy_spec = {"_intj_lazy.Launcher",
                                     sizeof(intj_bound_header), 1,
                                     Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC
#ifdef Py_TPFLAGS_IMMUTABLETYPE
                                         | Py_TPFLAGS_IMMUTABLETYPE
#endif
#ifdef Py_TPFLAGS_DISALLOW_INSTANTIATION
                                         | Py_TPFLAGS_DISALLOW_INSTANTIATION
#endif
                                     ,
                                     intj_lazy_slots};

/* new_launcher(tail_bytes, doc, builder) -> an unbuilt launcher */
static PyObject *new_launcher(PyObject *m, PyObject *const *args,
                              Py_ssize_t nargs) {
  intj_lazy_state *st = (intj_lazy_state *)PyModule_GetState(m);
  Py_ssize_t tail = -1;
  if (nargs == 3 && PyLong_CheckExact(args[0]))
    tail = PyLong_AsSsize_t(args[0]);
  if (tail < 0 || !PyBytes_CheckExact(args[1]) || !PyCallable_Check(args[2])) {
    if (!PyErr_Occurred())
      PyErr_SetString(PyExc_TypeError,
                      "intj: new_launcher(tail_bytes, doc, builder)");
    return NULL;
  }
  intj_bound_header *h =
      (intj_bound_header *)st->type->tp_alloc(st->type, tail);
  if (!h)
    return NULL;
  INTJ_RWLOCK_INIT(&h->lock); /* lives as long as the header; see dealloc */
  h->build_lock = PyThread_allocate_lock();
  if (!h->build_lock) {
    Py_DECREF(h);
    return PyErr_NoMemory();
  }
  Py_INCREF(args[1]);
  h->doc = args[1];
  Py_INCREF(args[2]);
  h->builder = args[2];
  h->def.ml_name = "launch";
  h->def.ml_meth = INTJ_SHIM;
  h->def.ml_flags = METH_FASTCALL;
  h->def.ml_doc = PyBytes_AS_STRING(h->doc);
  PyObject *launcher = PyCFunction_New(&h->def, (PyObject *)h);
  Py_DECREF(h);
  return launcher;
}

static PyObject *header_size(PyObject *m, PyObject *unused) {
  (void)m;
  (void)unused;
  return PyLong_FromSize_t(sizeof(intj_bound_header));
}

static int intj_lazy_exec(PyObject *m) {
  intj_lazy_state *st = (intj_lazy_state *)PyModule_GetState(m);
#if PY_VERSION_HEX >= 0x03090000
  st->type = (PyTypeObject *)PyType_FromModuleAndSpec(m, &intj_lazy_spec, NULL);
#else
  st->type = (PyTypeObject *)PyType_FromSpec(&intj_lazy_spec);
#endif
  return st->type ? 0 : -1;
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
     "new_launcher(tail_bytes, doc, builder) -> launcher"},
    {"header_size", header_size, METH_NOARGS, NULL},
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

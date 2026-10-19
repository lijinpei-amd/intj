/* A launcher's fixed-size object.  `_intj_lazy` allocates it at decoration;
 * a rendered module's `_init` allocates the per-launcher state block behind
 * `state` and installs `ops`.  Both compile this one definition;
 * `_header_size` lets Python check they agree. */
#pragma once

#include <Python.h>
#include <assert.h>
#include <stddef.h>
#include <stdint.h>

typedef PyObject *(*intj_fast_fn)(PyObject *, PyObject *const *, Py_ssize_t);

/* A launcher's lock, for what the GIL guards on a default build: its kernel
 * cache, memos and intern writes.  Launches take it shared across their
 * lookups and the launch, because records live in the map and a put can move
 * them; a put takes it exclusive.  Compiled out with the GIL, so a default
 * build pays nothing.  CPython's own rwlock is private, hence pthread.
 * (Moved from python_intf/cpython_abi.h: it is not a CPython internal.) */
#ifdef Py_GIL_DISABLED
#include <pthread.h>
typedef pthread_rwlock_t intj_rwlock;
#define INTJ_RWLOCK_INIT(l) pthread_rwlock_init((l), NULL)
#define INTJ_RWLOCK_DESTROY(l) pthread_rwlock_destroy(l)
#define INTJ_RDLOCK(l) pthread_rwlock_rdlock(l)
#define INTJ_WRLOCK(l) pthread_rwlock_wrlock(l)
#define INTJ_RWUNLOCK(l) pthread_rwlock_unlock(l)
#else
typedef char intj_rwlock;
#define INTJ_RWLOCK_INIT(l) ((void)(l))
#define INTJ_RWLOCK_DESTROY(l) ((void)(l))
#define INTJ_RDLOCK(l) ((void)(l))
#define INTJ_WRLOCK(l) ((void)(l))
#define INTJ_RWUNLOCK(l) ((void)(l))
#endif

/* Free-threaded builds lock the object around its member reads and writes;
 * before 3.13 these are plain blocks. */
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
  PyObject *options; /* MakeLauncherOption; the resolved one once built */
  PyObject *builder; /* until built: (launcher) -> (entry, options, doc) */
  PyObject *module;
  PyObject *doc; /* bytes behind def.ml_doc, once built */
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

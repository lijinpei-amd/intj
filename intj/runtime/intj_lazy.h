/* The bound launcher's fixed header.  `_intj_lazy` (intj_lazy.c) allocates
 * it at its final size; a rendered module's init_bound fills it in place.
 * Both compile this one definition, so the static asserts below pin the same
 * layout on both sides.  Trailing arrays follow the header: their layout is
 * the rendered module's (`intj_bound_tail`, intj_runtime.h), their byte
 * count is ob_size. */
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

typedef struct intj_bound_header {
  PyVarObject ob_base; /* PyObject_VAR_HEAD; ob_size: trailing array bytes */
  /* Hot: a warmed launch reads these and nothing else in the header. */
  /* the module's intj_state; NULL until built, and after clear */
  void *state;
  int64_t device_ordinal; /* bind_device only */
  int32_t device_handle;  /* bind_device only */
  int32_t cache_ready;
  uint64_t cache[3]; /* opaque intj_cache storage, only ever accessed as one */
  /* intj_final *: a keyless fixed-device launcher's kernel */
  void *fixed_kernel;
  /* Cold. */
  PyMethodDef def;   /* the launcher PyCFunction's; ml_meth is swapped once */
  PyObject *doc;     /* bytes behind def.ml_doc */
  PyObject *builder; /* until built: callable(header) -> entry address */
  PyObject *module;
  /* callable(value) -> id or None: this launcher's intern table */
  PyObject *intern;
  /* this launcher's miss callback (the tuned callback if tuned) */
  PyObject *compile_cb;
  intj_rwlock lock; /* guards this launcher's cache, memos and intern writes */
  PyObject *grid_py;
  PyObject *grid_hidden;
  PyThread_type_lock build_lock;
  unsigned long build_thread; /* thread running the builder, else 0 */
  /* the module's, once built */
  int (*traverse)(PyObject *self, visitproc visit, void *arg);
  int (*clear)(PyObject *self);
} intj_bound_header;

static_assert(offsetof(intj_bound_header, state) == sizeof(PyVarObject),
              "intj: state leads the hot fields");
static_assert(offsetof(intj_bound_header, fixed_kernel) + sizeof(void *) <=
                  sizeof(PyVarObject) + 64,
              "intj: hot fields sit within 64 bytes after the object head "
              "(a span, not an aligned line: GC objects are not 64-aligned)");
static_assert(offsetof(intj_bound_header, def) >=
                  offsetof(intj_bound_header, fixed_kernel) + sizeof(void *),
              "intj: cold fields follow the hot ones");
static_assert(sizeof(intj_bound_header) % 8 == 0,
              "intj: trailing arrays start 8-byte aligned");

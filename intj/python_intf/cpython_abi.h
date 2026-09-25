/* CPython internals for the verified 3.8 to 3.14 layouts.  PY_VERSION_HEX
 * selects the int layout at compile time; Py_GIL_DISABLED selects the lock.
 * Needs INTJ_ALWAYS_INLINE / INTJ_LIKELY / INTJ_UNLIKELY defined first.
 */
#pragma once

#include <Python.h>
#include <stdint.h>

#if PY_VERSION_HEX < 0x03080000
#error "intj: unsupported CPython int layout"
#endif
#if PY_VERSION_HEX < 0x030B0000
#include <longintrepr.h> /* Python.h includes it itself from 3.11 */
#endif

#if PY_VERSION_HEX < 0x030C0000
static INTJ_ALWAYS_INLINE int intj_long_compact(PyLongObject *v, int64_t *out) {
  Py_ssize_t size = Py_SIZE(v);
  if (size < -1 || size > 1)
    return 0;
  /* zero may have no digit allocated at all, so it is not read */
  *out = size == 0 ? 0 : (int64_t)size * (int64_t)v->ob_digit[0];
  return 1;
}

static INTJ_ALWAYS_INLINE const digit *intj_long_digits(PyLongObject *v,
                                                         size_t *nd,
                                                         int *negative) {
  Py_ssize_t size = Py_SIZE(v);
  *negative = size < 0;
  *nd = (size_t)(size < 0 ? -size : size);
  return v->ob_digit;
}
#else
/* The compact case has a public reader.  Larger ints use CPython's own
 * lv_tag and digit macros from cpython/longintrepr.h. */
static INTJ_ALWAYS_INLINE int intj_long_compact(PyLongObject *v, int64_t *out) {
  if (!PyUnstable_Long_IsCompact(v))
    return 0;
  *out = (int64_t)PyUnstable_Long_CompactValue(v);
  return 1;
}

static INTJ_ALWAYS_INLINE const digit *intj_long_digits(PyLongObject *v,
                                                         size_t *nd,
                                                         int *negative) {
  uintptr_t tag = v->long_value.lv_tag;
  *nd = (size_t)(tag >> _PyLong_NON_SIZE_BITS);
  *negative = (tag & _PyLong_SIGN_MASK) == 2;
  return v->long_value.ob_digit;
}
#endif

static_assert(PyLong_SHIFT == 30, "intj's int decoder assumes 30-bit digits");

/* The magnitude, if it fits 90 bits: 0 = ok, -1 = larger. */
static INTJ_ALWAYS_INLINE int intj_long_magnitude(PyLongObject *v,
                                                  __uint128_t *acc,
                                                  int *negative) {
  size_t nd;
  const digit *d = intj_long_digits(v, &nd, negative);
  if (INTJ_UNLIKELY(nd > 3)) /* > 90 bits */
    return -1;
  __uint128_t a = 0;
  for (size_t i = nd; i-- > 0;)
    a = (a << PyLong_SHIFT) | d[i];
  *acc = a;
  return 0;
}

/* 0 = ok, -1 = does not fit */
static INTJ_ALWAYS_INLINE int intj_as_i64(PyObject *o, int64_t *out) {
  PyLongObject *v = (PyLongObject *)o;
  if (INTJ_LIKELY(intj_long_compact(v, out)))
    return 0;
  __uint128_t acc;
  int negative;
  if (INTJ_UNLIKELY(intj_long_magnitude(v, &acc, &negative) != 0))
    return -1;
  if (negative) {
    if (acc > ((__uint128_t)1 << 63))
      return -1;
    *out = (int64_t)(-(__int128_t)acc);
  } else {
    if (acc > (__uint128_t)INT64_MAX)
      return -1;
    *out = (int64_t)acc;
  }
  return 0;
}

/* Which of the two integer widths a python int lands in, if either.  Triton
 * buckets an int by what it fits (i32/i64, else u64), so a caller wants both
 * answers from one decode -- asking `intj_as_i64` and then `intj_as_u64` walks
 * the digits twice for every value above INT64_MAX. */
#define INTJ_INT_TOO_BIG 0 /* fits neither: |value| is over 64 bits */
#define INTJ_INT_I64 1     /* *out is the int64, cast back from the bits */
#define INTJ_INT_U64 2     /* *out is the uint64: above INT64_MAX */

static INTJ_ALWAYS_INLINE int intj_as_int(PyObject *o, uint64_t *out) {
  PyLongObject *v = (PyLongObject *)o;
  int64_t small;
  if (INTJ_LIKELY(intj_long_compact(v, &small))) {
    *out = (uint64_t)small;
    return INTJ_INT_I64;
  }
  __uint128_t acc;
  int negative;
  if (INTJ_UNLIKELY(intj_long_magnitude(v, &acc, &negative) != 0))
    return INTJ_INT_TOO_BIG;
  if (negative) { /* int64 or nothing */
    if (acc > ((__uint128_t)1 << 63))
      return INTJ_INT_TOO_BIG;
    *out = (uint64_t)(int64_t)(-(__int128_t)acc);
    return INTJ_INT_I64;
  }
  if (acc > (__uint128_t)UINT64_MAX)
    return INTJ_INT_TOO_BIG;
  *out = (uint64_t)acc;
  return acc > (__uint128_t)INT64_MAX ? INTJ_INT_U64 : INTJ_INT_I64;
}

/* Reads `ob_fval` rather than calling `PyFloat_AS_DOUBLE`: that is a static
 * inline in CPython's headers, and in the c++ mode's much larger translation
 * unit g++ leaves it out of line -- a real call per float argument.  The two
 * are the same field read, in every CPython. */
#define INTJ_FLOAT_VALUE(o) (((PyFloatObject *)(o))->ob_fval)

/* The module's one lock, for what the GIL guards on a default build: the kernel
 * cache and the compile callback.  Compiled out with the GIL, so a default build
 * pays nothing.  PyMutex is zero-initialized unlocked, as module state is.
 * ponytail: every launch takes it on a free-threaded build, so concurrent
 * launchers serialize on the lookup (~20 ns); lock-free readers over an
 * insert-only table, with writers under this lock, if that shows up. */
#ifdef Py_GIL_DISABLED
typedef PyMutex intj_mutex;
#define INTJ_LOCK(m) PyMutex_Lock(m)
#define INTJ_UNLOCK(m) PyMutex_Unlock(m)
#else
typedef char intj_mutex;
#define INTJ_LOCK(m) ((void)(m))
#define INTJ_UNLOCK(m) ((void)(m))
#endif

/* Public calls the older interpreters predate. */
#if PY_VERSION_HEX < 0x030C0000
#if PY_VERSION_HEX < 0x03090000
#define PyObject_Vectorcall _PyObject_Vectorcall
static inline PyObject *PyObject_CallMethodNoArgs(PyObject *self,
                                                  PyObject *name) {
  return PyObject_CallMethodObjArgs(self, name, NULL);
}
#endif
#if PY_VERSION_HEX < 0x030A0000
static inline PyObject *Py_NewRef(PyObject *o) {
  Py_INCREF(o);
  return o;
}
#endif
/* Before 3.12, the error indicator is a (type, value, traceback) triple that
 * must be normalized before returning the raised exception. */
static inline PyObject *PyErr_GetRaisedException(void) {
  PyObject *type, *value, *tb;
  PyErr_Fetch(&type, &value, &tb);
  if (!type)
    return NULL;
  PyErr_NormalizeException(&type, &value, &tb);
  if (tb)
    PyException_SetTraceback(value, tb);
  Py_DECREF(type);
  Py_XDECREF(tb);
  return value;
}
static inline void PyErr_SetRaisedException(PyObject *exc) { /* steals exc */
  PyErr_Restore(Py_NewRef((PyObject *)Py_TYPE(exc)), exc,
                PyException_GetTraceback(exc));
}
#endif

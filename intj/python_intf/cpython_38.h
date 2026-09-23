/* The CPython internals the runtime reads, for the int layout of 3.8 to 3.11:
 * `ob_size` is the digit count, negated for a negative int.  See cpython_312.h
 * for how this file is picked and checked.
 */
#pragma once

#include <Python.h>
#include <stdint.h>

#if PY_VERSION_HEX < 0x03080000 || PY_VERSION_HEX >= 0x030C0000
#error "cpython_38.h is the int layout of CPython 3.8 to 3.11"
#endif
#if PY_VERSION_HEX < 0x030B0000
#include <longintrepr.h> /* Python.h includes it itself from 3.11 */
#endif

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

#include "cpython_common.h"

/* The public calls the runtime uses that these versions predate, in terms of
 * what they had.  Each is guarded by the version that added it. */
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
/* 3.12 keeps only the normalized exception; before it, the error indicator is
 * a (type, value, traceback) triple that has to be normalized first. */
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

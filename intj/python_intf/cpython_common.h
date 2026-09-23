/* What every CPython layer shares: the int and float decoders, over one int
 * layout's two primitives, and the free-threaded build's lock.
 *
 * Included by a cpython_*.h after it defines, always inline:
 *
 *   int intj_long_compact(PyLongObject *v, int64_t *out)
 *     1 and the value when |value| < 2**30, else 0.
 *   const digit *intj_long_digits(PyLongObject *v, size_t *nd, int *negative)
 *     the magnitude's 30-bit digits, least significant first.
 */
#pragma once

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

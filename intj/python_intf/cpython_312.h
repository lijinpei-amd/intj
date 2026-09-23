/* The CPython internals the runtime reads, for the int layout 3.12 introduced
 * (`lv_tag`).  `cpython_abi.header_for` picks this file per interpreter version
 * and the render includes it as INTJ_PYTHON_ABI.
 *
 * Needs INTJ_ALWAYS_INLINE / INTJ_LIKELY / INTJ_UNLIKELY defined first.  Unlike
 * torch's, these layouts need no table: every module is compiled against the
 * running interpreter's own headers.  What the compiler cannot check -- the
 * meaning of a field -- `python -m intj.python_intf.check` does.
 */
#pragma once

#include <Python.h>
#include <stdint.h>

#if PY_VERSION_HEX < 0x030C0000
#error "cpython_312.h is the int layout of CPython 3.12 and later"
#endif

/* The compact case has a public reader (`PyUnstable_Long_*`).  Above it the
 * digits have none before `PyLong_AsNativeBytes` in 3.13, so the rest reads
 * `lv_tag` and `ob_digit` with CPython's own macros from cpython/longintrepr.h. */
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

#include "cpython_common.h"

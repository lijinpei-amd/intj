/* INTJ launcher runtime: helpers shared by every rendered entry module.
 *
 * The rendered module must define INTJ_NWORDS (spec-key length in uint64 words)
 * before including this header.
 */
#pragma once

#include <Python.h>
#include <dlfcn.h>
#include <stdint.h>
#include <string.h>

/* ---------------------------------------------------------------- integers */

#if PY_VERSION_HEX < 0x030C0000
#error "intj requires CPython 3.12 or newer (PyLongObject layout)"
#endif
_Static_assert(PyLong_SHIFT == 30, "intj's int decoder assumes 30-bit digits");

/* CPython 3.12 PyLongObject layout, see cpython/longintrepr.h. */
#define INTJ_SIGN_MASK 3
#define INTJ_NON_SIZE_BITS 3

/* 0 = ok, -1 = does not fit, 1 = not an exact int */
static inline int intj_as_i64(PyObject *o, int64_t *out) {
  PyLongObject *v = (PyLongObject *)o;
  uintptr_t tag = v->long_value.lv_tag;
  if (tag < (2 << INTJ_NON_SIZE_BITS)) { /* |value| < 2**30 */
    int64_t sign = 1 - (int64_t)(tag & INTJ_SIGN_MASK);
    *out = sign * (int64_t)v->long_value.ob_digit[0];
    return 0;
  }
  size_t nd = (size_t)(tag >> INTJ_NON_SIZE_BITS);
  if (nd > 3) /* > 90 bits */
    return -1;
  __uint128_t acc = 0;
  for (size_t i = nd; i-- > 0;)
    acc = (acc << PyLong_SHIFT) | v->long_value.ob_digit[i];
  if ((tag & INTJ_SIGN_MASK) == 2) {
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

/* Only called after intj_as_i64 reported overflow, i.e. for values that do not
 * fit in int64.  Mirrors PyLong_AsUnsignedLongLong. */
static inline int intj_as_u64(PyObject *o, uint64_t *out) {
  PyLongObject *v = (PyLongObject *)o;
  uintptr_t tag = v->long_value.lv_tag;
  if ((tag & INTJ_SIGN_MASK) == 2)
    return -1;
  size_t nd = (size_t)(tag >> INTJ_NON_SIZE_BITS);
  if (nd > 3)
    return -1;
  __uint128_t acc = 0;
  for (size_t i = nd; i-- > 0;)
    acc = (acc << PyLong_SHIFT) | v->long_value.ob_digit[i];
  if (acc > (__uint128_t)UINT64_MAX)
    return -1;
  *out = (uint64_t)acc;
  return 0;
}

/* ------------------------------------------------------------------- hash */

static inline uint64_t intj_mix(uint64_t a, uint64_t b) {
  __uint128_t r = (__uint128_t)a * b;
  return (uint64_t)(r >> 64) ^ (uint64_t)r;
}

static inline uint64_t intj_hash(const uint64_t *w) {
  const uint64_t s0 = 0xa0761d6478bd642full, s1 = 0xe7037ed1a0b428dbull;
  uint64_t h = s0;
  for (int i = 0; i < INTJ_NWORDS; i++)
    h = intj_mix(h ^ s1, w[i] ^ s0);
  return intj_mix(h, INTJ_NWORDS * 8 + s1);
}

/* ------------------------------------------------------------ kernel cache */

typedef struct {
  void *function;     /* hipFunction_t / CUfunction */
  uint32_t block_dim; /* warp_size * num_warps */
  uint32_t shared;    /* dynamic LDS bytes */
  uint32_t nparams;   /* kernel params, scratch slots excluded */
} intj_kernel;

typedef struct {
  uint64_t hash;
  intj_kernel *val; /* NULL => empty slot */
  uint64_t key[INTJ_NWORDS];
} intj_slot;

typedef struct {
  intj_slot *slots;
  uint32_t mask;
  uint32_t used;
} intj_map;

static inline int intj_key_eq(const uint64_t *a, const uint64_t *b) {
  for (int i = 0; i < INTJ_NWORDS; i++)
    if (a[i] != b[i])
      return 0;
  return 1;
}

static inline intj_kernel *intj_map_get(const intj_map *m, const uint64_t *k,
                                        uint64_t h) {
  uint32_t i = (uint32_t)h & m->mask;
  for (;;) {
    const intj_slot *s = &m->slots[i];
    if (!s->val)
      return NULL;
    if (s->hash == h && intj_key_eq(s->key, k))
      return s->val;
    i = (i + 1) & m->mask;
  }
}

static int intj_map_init(intj_map *m, uint32_t cap) {
  m->slots = (intj_slot *)PyMem_RawCalloc(cap, sizeof(intj_slot));
  if (!m->slots)
    return -1;
  m->mask = cap - 1;
  m->used = 0;
  return 0;
}

static void intj_map_insert(intj_map *m, const uint64_t *k, uint64_t h,
                            intj_kernel *val) {
  uint32_t i = (uint32_t)h & m->mask;
  while (m->slots[i].val)
    i = (i + 1) & m->mask;
  m->slots[i].hash = h;
  m->slots[i].val = val;
  memcpy(m->slots[i].key, k, INTJ_NWORDS * sizeof(uint64_t));
  m->used++;
}

/* Returns -1 on allocation failure (with no python error set). */
static int intj_map_put(intj_map *m, const uint64_t *k, uint64_t h,
                        intj_kernel *val) {
  if ((m->used + 1) * 2 > m->mask + 1) {
    uint32_t cap = (m->mask + 1) * 2;
    intj_map grown;
    if (intj_map_init(&grown, cap) != 0)
      return -1;
    for (uint32_t i = 0; i <= m->mask; i++)
      if (m->slots[i].val)
        intj_map_insert(&grown, m->slots[i].key, m->slots[i].hash,
                        m->slots[i].val);
    PyMem_RawFree(m->slots);
    *m = grown;
  }
  intj_map_insert(m, k, h, val);
  return 0;
}

/* ------------------------------------------------------------ torch access */

/* ABI-stable C shims exported by libtorch_cpu.so; see
 * torch/csrc/inductor/aoti_torch/c/shim.h.  THPVariable stores the at::Tensor
 * right after PyObject_HEAD, so the handle is a fixed offset from the object.
 */
typedef void *intj_tensor_handle;
#define INTJ_TENSOR_HANDLE(o) ((intj_tensor_handle)((char *)(o) + sizeof(PyObject)))

typedef int32_t (*intj_get_data_ptr_t)(intj_tensor_handle, void **);
typedef int32_t (*intj_get_storage_size_t)(intj_tensor_handle, int64_t *);
typedef int32_t (*intj_get_dtype_t)(intj_tensor_handle, int32_t *);

/* ----------------------------------------------------------- gpu  runtime */

/* hipModuleLaunchKernel and cuLaunchKernel take the same arguments in the same
 * order.  Their error-string calls do not, hence the two typedefs. */
typedef int32_t (*intj_launch_t)(void *f, uint32_t gx, uint32_t gy, uint32_t gz,
                                 uint32_t bx, uint32_t by, uint32_t bz,
                                 uint32_t shared, void *stream, void **params,
                                 void **extra);
typedef const char *(*intj_error_string_ret_t)(int32_t);
typedef int32_t (*intj_error_string_out_t)(int32_t, const char **);

/* ----------------------------------------------------------- spec-key tags */

#define INTJ_T_NONE 1u     /* ("constexpr", None) */
#define INTJ_T_ONE 2u      /* int 1 folded to ("constexpr", 1) */
#define INTJ_T_I32 3u
#define INTJ_T_I64 4u
#define INTJ_T_U64 5u
#define INTJ_T_FP32 6u
#define INTJ_T_U1 7u
#define INTJ_T_PTR 8u
#define INTJ_T_CX_NONE 9u
#define INTJ_T_CX_BOOL 10u
#define INTJ_T_CX_INT 11u
#define INTJ_T_CX_UINT 12u
#define INTJ_T_CX_FLOAT 13u

#define INTJ_FLAG_D 1u /* tt.divisibility = 16 */
#define INTJ_FLAG_S 2u /* tt.pointer_range = 32 */

#define INTJ_WORD(tag, dtype, flags)                                           \
  (((uint64_t)(tag) << 56) | ((uint64_t)(uint32_t)(dtype) << 8) |              \
   (uint64_t)(flags))

/* --------------------------------------------------------- argument decode */

/* Decode one non-constexpr kernel argument: fills one key word, and appends at
 * most one value slot.  SPEC/ALIGN/SBIT are render-time constants
 * (do_not_specialize, do_not_specialize_on_alignment, and whether the backend
 * specializes pointers on a 2 GiB range).
 *
 * `st` must expose tensor_type/param_type and the three torch shims.
 * On failure it sets a python error and jumps to `error`.
 */
#define INTJ_DECODE(st, o, word, vals, np, SPEC, ALIGN, SBIT, pname)           \
  do {                                                                         \
    PyObject *_o = (o);                                                        \
    PyTypeObject *_t = Py_TYPE(_o);                                            \
    if (_t == (st)->tensor_type || _t == (st)->param_type) {                   \
      intj_tensor_handle _h = INTJ_TENSOR_HANDLE(_o);                          \
      void *_p = NULL;                                                         \
      int32_t _dt = -1;                                                        \
      if ((st)->get_data_ptr(_h, &_p) != 0 || (st)->get_dtype(_h, &_dt) != 0) { \
        PyErr_Format(PyExc_RuntimeError,                                       \
                     "intj: failed to read tensor argument '%s'", pname);      \
        goto error;                                                            \
      }                                                                        \
      uint32_t _flags = 0;                                                     \
      if (SPEC) {                                                              \
        if (ALIGN && (((uintptr_t)_p & 15u) == 0))                             \
          _flags |= INTJ_FLAG_D;                                               \
        if (SBIT) {                                                            \
          int64_t _sz = 0;                                                     \
          if ((st)->get_storage_size(_h, &_sz) != 0) {                         \
            PyErr_Format(PyExc_RuntimeError,                                   \
                         "intj: failed to read storage size of '%s'", pname);  \
            goto error;                                                        \
          }                                                                    \
          if (_sz <= 2147483647LL)                                             \
            _flags |= INTJ_FLAG_S;                                             \
        }                                                                      \
      }                                                                        \
      (word) = INTJ_WORD(INTJ_T_PTR, _dt, _flags);                             \
      (vals)[(np)] = (uint64_t)(uintptr_t)_p;                                  \
      (np)++;                                                                  \
    } else if (_o == Py_True || _o == Py_False) {                              \
      (word) = INTJ_WORD(INTJ_T_U1, 0, 0);                                     \
      (vals)[(np)] = (uint64_t)(_o == Py_True);                                \
      (np)++;                                                                  \
    } else if (PyLong_CheckExact(_o)) {                                        \
      int64_t _v;                                                              \
      int _rc = intj_as_i64(_o, &_v);                                          \
      if (_rc == 0) {                                                          \
        if (SPEC && _v == 1) {                                                 \
          (word) = INTJ_WORD(INTJ_T_ONE, 0, 0);                                \
        } else {                                                               \
          uint32_t _flags =                                                    \
              (SPEC && ALIGN && ((_v & 15) == 0)) ? INTJ_FLAG_D : 0;           \
          uint32_t _tag = (_v >= INT32_MIN && _v <= INT32_MAX) ? INTJ_T_I32    \
                                                               : INTJ_T_I64;   \
          (word) = INTJ_WORD(_tag, 0, _flags);                                 \
          (vals)[(np)] = (uint64_t)_v;                                         \
          (np)++;                                                              \
        }                                                                      \
      } else {                                                                 \
        uint64_t _u;                                                           \
        if (intj_as_u64(_o, &_u) != 0) {                                       \
          PyErr_Format(PyExc_OverflowError,                                    \
                       "intj: integer argument '%s' is too large", pname);     \
          goto error;                                                          \
        }                                                                      \
        uint32_t _flags =                                                      \
            (SPEC && ALIGN && ((_u & 15u) == 0)) ? INTJ_FLAG_D : 0;            \
        (word) = INTJ_WORD(INTJ_T_U64, 0, _flags);                             \
        (vals)[(np)] = _u;                                                     \
        (np)++;                                                                \
      }                                                                        \
    } else if (PyFloat_CheckExact(_o)) {                                       \
      float _f = (float)PyFloat_AS_DOUBLE(_o);                                 \
      uint32_t _bits;                                                          \
      memcpy(&_bits, &_f, 4);                                                  \
      (word) = INTJ_WORD(INTJ_T_FP32, 0, 0);                                   \
      (vals)[(np)] = (uint64_t)_bits;                                          \
      (np)++;                                                                  \
    } else if (_o == Py_None) {                                                \
      (word) = INTJ_WORD(INTJ_T_NONE, 0, 0);                                   \
    } else {                                                                   \
      PyErr_Format(PyExc_TypeError,                                            \
                   "intj: unsupported argument '%s' of type %s; pass a "       \
                   "torch.Tensor, int, float, bool or None",                   \
                   pname, Py_TYPE(_o)->tp_name);                               \
      goto error;                                                              \
    }                                                                          \
  } while (0)

/* Decode a tl.constexpr argument into two key words.  No value slot. */
#define INTJ_DECODE_CONSTEXPR(o, w0, w1, pname)                                \
  do {                                                                         \
    PyObject *_o = (o);                                                        \
    if (_o == Py_None) {                                                       \
      (w0) = INTJ_WORD(INTJ_T_CX_NONE, 0, 0);                                  \
      (w1) = 0;                                                                \
    } else if (_o == Py_True || _o == Py_False) {                              \
      (w0) = INTJ_WORD(INTJ_T_CX_BOOL, 0, 0);                                  \
      (w1) = (uint64_t)(_o == Py_True);                                        \
    } else if (PyLong_CheckExact(_o)) {                                        \
      int64_t _v;                                                              \
      if (intj_as_i64(_o, &_v) == 0) {                                         \
        (w0) = INTJ_WORD(INTJ_T_CX_INT, 0, 0);                                 \
        (w1) = (uint64_t)_v;                                                   \
      } else {                                                                 \
        uint64_t _u;                                                           \
        if (intj_as_u64(_o, &_u) != 0) {                                       \
          PyErr_Format(PyExc_OverflowError,                                    \
                       "intj: constexpr argument '%s' is too large", pname);   \
          goto error;                                                          \
        }                                                                      \
        (w0) = INTJ_WORD(INTJ_T_CX_UINT, 0, 0);                                \
        (w1) = _u;                                                             \
      }                                                                        \
    } else if (PyFloat_CheckExact(_o)) {                                       \
      double _d = PyFloat_AS_DOUBLE(_o);                                       \
      (w0) = INTJ_WORD(INTJ_T_CX_FLOAT, 0, 0);                                 \
      memcpy(&(w1), &_d, 8);                                                   \
    } else {                                                                   \
      PyErr_Format(PyExc_TypeError,                                            \
                   "intj: unsupported constexpr argument '%s' of type %s; "    \
                   "pass an int, float, bool or None",                         \
                   pname, Py_TYPE(_o)->tp_name);                               \
      goto error;                                                              \
    }                                                                          \
  } while (0)

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
/* `static_assert` rather than `_Static_assert`: the former is spelled the same in
 * C11 (via assert.h, which Python.h pulls in) and in C++, and the CXX access mode
 * compiles this header as C++. */
static_assert(PyLong_SHIFT == 30, "intj's int decoder assumes 30-bit digits");

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

/* intj reads three things off a tensor: the data pointer, the dtype (as an
 * opaque int32 discriminator) and, where the backend specializes on pointer
 * range, the storage size.  Exactly one of three strategies is compiled in.
 *
 *   INTJ_ACCESS_SHIM     read torch's structs at offsets supplied at load time
 *   INTJ_ACCESS_CPYTHON  call through the interpreter
 *   INTJ_ACCESS_CXX      C++, against torch's own headers
 *
 * No mode calls libtorch's aoti_torch_* shims, and no mode dlopens libtorch.
 */

#define INTJ_NDTYPES 64

typedef struct {
  /* SHIM: byte offsets into THPVariable / TensorImpl / StorageImpl. Unused, and
   * left zero, in the other two modes. */
  uint16_t cdata;          /* PyObject*   -> TensorImpl**   */
  uint16_t storage;        /* TensorImpl* -> StorageImpl**  */
  uint16_t storage_offset; /* TensorImpl* -> int64_t        */
  uint16_t numel;          /* TensorImpl* -> int64_t        */
  uint16_t data_type;      /* TensorImpl* -> TypeMeta index (low byte)  */
  uint16_t s_data;         /* StorageImpl* -> void*         */
  uint16_t s_nbytes;       /* StorageImpl* -> int64_t       */
  uint8_t itemsize[INTJ_NDTYPES]; /* dtype code -> element size */
  int ready;               /* set_torch_version has run */
} intj_torch_abi;

#ifdef INTJ_ACCESS_CXX
#include <torch/csrc/autograd/python_variable.h>
#endif

/* Interned method names, used only by the CPYTHON reader. Filled by
 * intj_abi_init; never released, like every other module-lifetime singleton. */
static PyObject *intj_str_dtype;
static PyObject *intj_str_data_ptr;
static PyObject *intj_str_untyped_storage;
static PyObject *intj_str_nbytes;

#if defined(INTJ_ACCESS_SHIM)

/* Reads torch's structs directly. `want_size` is a compile-time-ish flag: the
 * storage size is only needed where the backend specializes on pointer range. */
static inline int intj_read_tensor(const intj_torch_abi *abi, PyObject *o,
                                   void **p, int32_t *dt, int want_size,
                                   int64_t *sz) {
  char *ti = *(char **)((char *)o + abi->cdata);
  char *si = *(char **)(ti + abi->storage);
  int64_t numel = *(const int64_t *)(ti + abi->numel);
  uint8_t code = *(const uint8_t *)(ti + abi->data_type);
  *dt = (int32_t)code;
  /* A null storage means the tensor has none at all -- sparse, and anything else
   * with a non-dense impl.  torch raises there, and so must this: handing the
   * kernel a null pointer instead would be a silent wrong launch.  Every dense
   * tensor has a storage, including a zero-element one. */
  if (!si) {
    PyErr_SetString(PyExc_RuntimeError,
                    "intj: cannot access data pointer of a tensor with no "
                    "storage (a sparse tensor?)");
    return -1;
  }
  /* torch's Tensor::data_ptr() returns null for every zero-element tensor, even
   * one whose storage is live and whose storage_offset is not zero (an empty
   * slice at the end of a buffer).  Reproduce that: the pointer feeds the
   * alignment bit of the spec key, so a past-the-end pointer here would key
   * differently from the other two modes for the same arguments. */
  if (numel == 0) {
    *p = NULL;
    if (want_size)
      *sz = *(const int64_t *)(si + abi->s_nbytes);
    return 0;
  }
  if (code >= INTJ_NDTYPES || !abi->itemsize[code]) {
    /* A dtype intj has no element size for means the offsets are wrong, or torch
     * grew a dtype after the table was built.  Either way, refuse rather than
     * compute a pointer from a guess. */
    PyErr_Format(PyExc_RuntimeError,
                 "intj: unknown torch dtype code %d; this build's tensor layout "
                 "does not match the running torch",
                 (int)code);
    return -1;
  }
  char *data = *(char **)(si + abi->s_data);
  int64_t off = *(const int64_t *)(ti + abi->storage_offset);
  *p = data + off * (int64_t)abi->itemsize[code];
  if (want_size)
    *sz = *(const int64_t *)(si + abi->s_nbytes);
  return 0;
}

#elif defined(INTJ_ACCESS_CPYTHON)

/* Calls through the interpreter.  Slowest, but assumes nothing about torch's
 * layout beyond THPDtype, which self-checks at load. */
static inline int intj_read_tensor(const intj_torch_abi *abi, PyObject *o,
                                   void **p, int32_t *dt, int want_size,
                                   int64_t *sz) {
  (void)abi;
  PyObject *d = PyObject_GetAttr(o, intj_str_dtype);
  if (!d)
    return -1;
  /* struct THPDtype { PyObject_HEAD at::ScalarType scalar_type; char name[65]; }
   * ScalarType is `enum class : int8_t`, so this is the first byte of payload.
   * Only reached after INTJ_DECODE's exact-type test, so `o` really is a tensor
   * and `d` really is a THPDtype. */
  *dt = (int32_t) * (const int8_t *)((char *)d + sizeof(PyObject));
  Py_DECREF(d);

  PyObject *v = PyObject_CallMethodNoArgs(o, intj_str_data_ptr);
  if (!v)
    return -1;
  *p = PyLong_AsVoidPtr(v);
  Py_DECREF(v);
  if (!*p && PyErr_Occurred()) /* a real 0 is legal: any zero-element tensor */
    return -1;

  if (want_size) {
    PyObject *st = PyObject_CallMethodNoArgs(o, intj_str_untyped_storage);
    if (!st)
      return -1;
    PyObject *n = PyObject_CallMethodNoArgs(st, intj_str_nbytes);
    Py_DECREF(st);
    if (!n)
      return -1;
    *sz = PyLong_AsLongLong(n);
    Py_DECREF(n);
    if (*sz == -1 && PyErr_Occurred())
      return -1;
  }
  return 0;
}

#elif defined(INTJ_ACCESS_CXX)

static inline int intj_read_tensor(const intj_torch_abi *abi, PyObject *o,
                                   void **p, int32_t *dt, int want_size,
                                   int64_t *sz) {
  (void)abi;
  /* torch throws; an exception reaching CPython's C frames is undefined, so it
   * stops here and becomes a python error like every other decode failure. */
  try {
    const at::Tensor &t = THPVariable_Unpack(o);
    *p = t.data_ptr();
    *dt = static_cast<int32_t>(t.scalar_type());
    if (want_size)
      *sz = static_cast<int64_t>(t.storage().nbytes());
  } catch (const std::exception &e) {
    PyErr_SetString(PyExc_RuntimeError, e.what());
    return -1;
  } catch (...) {
    PyErr_SetString(PyExc_RuntimeError, "intj: unknown c++ exception from torch");
    return -1;
  }
  return 0;
}

#else
#error "intj: define one of INTJ_ACCESS_{SHIM,CPYTHON,CXX}"
#endif

/* Re-raise the reader's error as "which argument", keeping torch's own message
 * as __cause__.  The reader knows what went wrong; only the caller knows which
 * parameter it was reading. */
static inline void intj_note_param(const char *pname) {
  PyObject *exc = PyErr_GetRaisedException();
  PyErr_Format(PyExc_RuntimeError, "intj: cannot read tensor argument '%s'", pname);
  if (exc) {
    PyObject *raised = PyErr_GetRaisedException();
    PyException_SetCause(raised, Py_NewRef(exc));
    PyException_SetContext(raised, exc); /* steals the reference */
    PyErr_SetRaisedException(raised);
  }
}

/* Intern the method names the CPYTHON reader uses.  Interning once and holding
 * the references for the module's life keeps the reader down to a pointer
 * compare in the attribute lookup.  Returns -1 with a python error set. */
static inline int intj_intern_names(void) {
  intj_str_dtype = PyUnicode_InternFromString("dtype");
  intj_str_data_ptr = PyUnicode_InternFromString("data_ptr");
  intj_str_untyped_storage = PyUnicode_InternFromString("untyped_storage");
  intj_str_nbytes = PyUnicode_InternFromString("nbytes");
  return (intj_str_dtype && intj_str_data_ptr && intj_str_untyped_storage &&
          intj_str_nbytes)
             ? 0
             : -1;
}

/* Confirm that `torch.dtype` still starts with the ScalarType byte.
 *
 * This is the one layout bet in the design that can check itself: THPDtype is
 * `{ PyObject_HEAD at::ScalarType scalar_type; char name[65]; }`, so the object
 * carries the name of the dtype it claims to be.  The object is at least 82
 * bytes, so the read cannot fault even if the layout moved.  Returns -1 with a
 * python error set.
 */
static inline int intj_dtype_selfcheck(PyObject *torch) {
  PyObject *f32 = PyObject_GetAttrString(torch, "float32");
  if (!f32)
    return -1;
  const char *payload = (const char *)f32 + sizeof(PyObject);
  int code = (int)*(const int8_t *)payload;
  int ok = code >= 0 && code < INTJ_NDTYPES && strncmp(payload + 1, "float32", 8) == 0;
  Py_DECREF(f32);
  if (!ok) {
    PyErr_SetString(PyExc_RuntimeError,
                    "intj: torch.dtype is not laid out as intj expects "
                    "(THPDtype { PyObject_HEAD ScalarType; char name[] }); "
                    "this torch is too new or too old for the cpython access mode");
    return -1;
  }
  return 0;
}

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
 * `st` must expose tensor_type/param_type and the torch ABI block.
 * On failure it sets a python error and returns NULL from the enclosing
 * function -- which must therefore return PyObject *.  (A `goto` to a shared
 * label would be tidier, but C++ forbids jumping over the initializations that
 * follow in `entry`, and the CXX access mode compiles this as C++.)
 */
#define INTJ_DECODE(st, o, word, vals, np, SPEC, ALIGN, SBIT, pname)           \
  do {                                                                         \
    PyObject *_o = (o);                                                        \
    PyTypeObject *_t = Py_TYPE(_o);                                            \
    if (_t == (st)->tensor_type || _t == (st)->param_type) {                   \
      void *_p = NULL;                                                         \
      int32_t _dt = -1;                                                        \
      int64_t _sz = 0;                                                         \
      int _want = (SPEC) && (SBIT);                                            \
      if (intj_read_tensor(&(st)->abi, _o, &_p, &_dt, _want, &_sz) != 0) {     \
        intj_note_param(pname);                                                \
        return NULL;                                                           \
      }                                                                        \
      uint32_t _flags = 0;                                                     \
      if (SPEC) {                                                              \
        if (ALIGN && (((uintptr_t)_p & 15u) == 0))                             \
          _flags |= INTJ_FLAG_D;                                               \
        if (_want && _sz <= 2147483647LL)                                      \
          _flags |= INTJ_FLAG_S;                                               \
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
          return NULL;                                                         \
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
      return NULL;                                                             \
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
          return NULL;                                                         \
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
      return NULL;                                                             \
    }                                                                          \
  } while (0)

/* INTJ launcher runtime: helpers shared by every rendered entry module.
 *
 * The rendered module must define INTJ_NWORDS (spec-key length in uint64 words)
 * before including this header.
 */
#pragma once

#include <Python.h>
#include <dlfcn.h>
#include <float.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#ifdef __cplusplus
#include <exception>
#include <new>
#endif

#if PY_VERSION_HEX < 0x030C0000
#error "intj requires CPython 3.12 or newer (PyLongObject layout)"
#endif

#if defined(__GNUC__) || defined(__clang__)
#define INTJ_ALWAYS_INLINE __attribute__((always_inline)) inline
/* Only for branches whose outcome is a property of the design, not a guess
 * about the caller: a bad argument, a launch failure, a cold cache.  The
 * argument *types* are deliberately not marked -- the render cannot know
 * whether a parameter is usually a tensor or an int. */
#define INTJ_LIKELY(x) __builtin_expect(!!(x), 1)
#define INTJ_UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
#define INTJ_ALWAYS_INLINE inline
#define INTJ_LIKELY(x) (x)
#define INTJ_UNLIKELY(x) (x)
#endif

#if defined(__clang__)
#define INTJ_ASSUME(x) __builtin_assume(x)
#elif defined(__GNUC__)
#define INTJ_ASSUME(x) do { if (!(x)) __builtin_unreachable(); } while (0)
#else
#define INTJ_ASSUME(x) ((void)0)
#endif

static_assert(PyLong_SHIFT == 30, "intj's int decoder assumes 30-bit digits");

/* The compact case has a public reader (`PyUnstable_Long_*`, 3.12+).  Above it
 * the digits have none before `PyLong_AsNativeBytes` in 3.13, so the rest walks
 * `ob_digit` with CPython's own layout macros from cpython/longintrepr.h.
 */

/* 0 = ok, -1 = does not fit, 1 = not an exact int */
static INTJ_ALWAYS_INLINE int intj_as_i64(PyObject *o, int64_t *out) {
  PyLongObject *v = (PyLongObject *)o;
  if (INTJ_LIKELY(PyUnstable_Long_IsCompact(v))) { /* |value| < 2**30 */
    *out = (int64_t)PyUnstable_Long_CompactValue(v);
    return 0;
  }
  uintptr_t tag = v->long_value.lv_tag;
  size_t nd = (size_t)(tag >> _PyLong_NON_SIZE_BITS);
  if (INTJ_UNLIKELY(nd > 3)) /* > 90 bits */
    return -1;
  __uint128_t acc = 0;
  for (size_t i = nd; i-- > 0;)
    acc = (acc << PyLong_SHIFT) | v->long_value.ob_digit[i];
  if ((tag & _PyLong_SIGN_MASK) == 2) {
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
  if (INTJ_LIKELY(PyUnstable_Long_IsCompact(v))) { /* |value| < 2**30 */
    *out = (uint64_t)(int64_t)PyUnstable_Long_CompactValue(v);
    return INTJ_INT_I64;
  }
  uintptr_t tag = v->long_value.lv_tag;
  size_t nd = (size_t)(tag >> _PyLong_NON_SIZE_BITS);
  if (INTJ_UNLIKELY(nd > 3)) /* > 90 bits */
    return INTJ_INT_TOO_BIG;
  __uint128_t acc = 0;
  for (size_t i = nd; i-- > 0;)
    acc = (acc << PyLong_SHIFT) | v->long_value.ob_digit[i];
  if ((tag & _PyLong_SIGN_MASK) == 2) { /* negative: int64 or nothing */
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

static inline uint64_t intj_mix(uint64_t a, uint64_t b) {
  __uint128_t r = (__uint128_t)a * b;
  return (uint64_t)(r >> 64) ^ (uint64_t)r;
}

/* The renderer knows INTJ_NWORDS, so the shape is chosen rather than looped.
 * The signature is the same at every length: `tests/bench_kernel_cache.cpp` and
 * the entry template both call this, and a per-length signature would fork both
 * callers for nothing.
 *
 * At one word the mix is replaced by a bijection -- xorshift-right and an odd
 * multiply both are -- which makes hash equality key equality and lets the slot
 * drop the key altogether (see intj_slot).  At two, one multiply suffices, the
 * way wyhash handles a short input.  Above that the chain is split across two
 * lanes: a mix is a ~4-cycle multiply and the whole chain sits between the last
 * argument decode and the first table probe, so five words is ~20 cycles of
 * pure latency serially and ~12 in pairs.  Same reason wyhash runs three lanes
 * over a 48-byte block.
 */
#if INTJ_NWORDS == 1
static inline uint64_t intj_hash(const uint64_t *w) {
  uint64_t x = w[0];
  x ^= x >> 30;
  x *= 0xbf58476d1ce4e5b9ull;
  x ^= x >> 27;
  x *= 0x94d049bb133111ebull;
  x ^= x >> 31;
  return x;
}
#elif INTJ_NWORDS == 2
static inline uint64_t intj_hash(const uint64_t *w) {
  const uint64_t s0 = 0xa0761d6478bd642full, s1 = 0xe7037ed1a0b428dbull;
  return intj_mix(intj_mix(w[0] ^ s0, w[1] ^ s1), 2 * 8 + s1);
}
#else
static inline uint64_t intj_hash(const uint64_t *w) {
  const uint64_t s0 = 0xa0761d6478bd642full, s1 = 0xe7037ed1a0b428dbull;
  uint64_t h0 = s0, h1 = s1;
  int i = 0;
  for (; i + 1 < INTJ_NWORDS; i += 2) {
    h0 = intj_mix(h0 ^ s1, w[i] ^ s0);
    h1 = intj_mix(h1 ^ s0, w[i + 1] ^ s1);
  }
  if (i < INTJ_NWORDS)
    h0 = intj_mix(h0 ^ s1, w[i] ^ s0);
  return intj_mix(h0 ^ h1, INTJ_NWORDS * 8 + s1);
}
#endif

typedef struct {
  void *function;     /* hipFunction_t / CUfunction */
  uint32_t block_dim; /* warp_size * num_warps */
  uint32_t shared;    /* dynamic LDS bytes */
  uint32_t nparams;   /* kernel params, scratch slots excluded */
} intj_kernel;

/* At one word `intj_hash` is a bijection, so `hash == h` IS key equality and the
 * slot has no reason to carry the key: 16 bytes, four to a cache line, against
 * 8 + 8 + INTJ_NWORDS * 8.  A probe is a cache miss, and the slot size is what
 * decides how many lines that miss costs.
 *
 * gcc 13.3.0 miscompiles the lookup below at `-O1 -fsanitize=undefined` (or
 * `=null`, or `=alignment`, each alone) when this branch is taken: the inlined
 * `intj_map_get` returns NULL for a key that is present, while an immediately
 * following identical call returns it.  Ruled out as a bug here rather than in
 * intj: UBSan reports no diagnostic at all -- it silently changes behaviour,
 * which is what a wrong-code bug looks like and a real null/alignment violation
 * does not; the keyed slot at INTJ_NWORDS==1 with this same hash is clean under
 * identical flags; -O0, -O2, -O3, every level without a sanitizer, and clang
 * are all clean; and no rephrasing of the loop (hoisting the loads, dropping
 * the INTJ_LIKELY hints) changes it.  It would not reduce to a standalone file,
 * so there is no upstream report yet.
 *
 * Nothing intj builds is affected: triton compiles rendered modules at `-O3`
 * with no sanitizer (`triton/runtime/build.py`), and so does
 * `tests/test_kernel_cache.py`.  It is recorded because the benchmark is the
 * only GPU-free coverage of this code, and someone will eventually build it by
 * hand with sanitizers and go looking for the bug in here.
 */
#if INTJ_NWORDS == 1
typedef struct {
  uint64_t hash;
  intj_kernel *val; /* NULL => empty slot */
} intj_slot;
#else
typedef struct {
  uint64_t hash;
  intj_kernel *val; /* NULL => empty slot */
  uint64_t key[INTJ_NWORDS];
} intj_slot;
#endif

typedef struct {
  intj_slot *slots;
  uint32_t mask;
  uint32_t used;
} intj_map;

/* memcmp, not a word loop: the loop has to exit early, so it compiles to one
 * mov/cmp/jne per word, while memcmp of a compile-time size vectorizes -- three
 * vpxor/vptest cover 88 bytes.  Measured on an 11-word key: 33 instructions and
 * 11 dependent branches down to 12 and 3. */
#if INTJ_NWORDS > 1 /* at one word the hash settles it; see intj_slot */
static inline int intj_key_eq(const uint64_t *a, const uint64_t *b) {
  return memcmp(a, b, INTJ_NWORDS * sizeof(uint64_t)) == 0;
}
#endif

static inline intj_kernel *intj_map_get(const intj_map *m, const uint64_t *k,
                                        uint64_t h) {
#if INTJ_NWORDS == 1
  (void)k;
#endif
  uint32_t i = (uint32_t)h & m->mask;
  for (;;) {
    const intj_slot *s = &m->slots[i];
    if (INTJ_UNLIKELY(!s->val))
      return NULL;
#if INTJ_NWORDS == 1
    if (INTJ_LIKELY(s->hash == h))
#else
    if (INTJ_LIKELY(s->hash == h && intj_key_eq(s->key, k)))
#endif
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
#if INTJ_NWORDS == 1
  (void)k;
#endif
  uint32_t i = (uint32_t)h & m->mask;
  while (m->slots[i].val)
    i = (i + 1) & m->mask;
  m->slots[i].hash = h;
  m->slots[i].val = val;
#if INTJ_NWORDS > 1
  memcpy(m->slots[i].key, k, INTJ_NWORDS * sizeof(uint64_t));
#endif
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
#if INTJ_NWORDS == 1 /* no key to carry over: the hash is the key */
        intj_map_insert(&grown, NULL, m->slots[i].hash, m->slots[i].val);
#else
        intj_map_insert(&grown, m->slots[i].key, m->slots[i].hash,
                        m->slots[i].val);
#endif
    PyMem_RawFree(m->slots);
    *m = grown;
  }
  intj_map_insert(m, k, h, val);
  return 0;
}

/* The kernel cache behind four calls, so the entry template never names an
 * implementation.  INTJ_CACHE_{INTJ,TSL,ABSL} selects one; the last two are
 * C++ and so force the module to be compiled as C++.
 *
 * `intj_cache` is POD in every mode -- it lives in module state, which CPython
 * hands out as zeroed memory, so a member with a constructor would need a
 * placement new the template should not have to know about.  The C++ maps are
 * held by pointer for the same reason.
 *
 * Lookups are handed the hash the caller already computed.  Only tsl can take
 * it; abseil has no such API and re-computes it, which is the measured cost of
 * that option rather than an oversight.
 */
#if defined(INTJ_CACHE_TSL) || defined(INTJ_CACHE_ABSL)

#include <cstring>
#include <new>
#if defined(INTJ_CACHE_TSL)
#include <tsl/robin_map.h>
#else
#include <absl/container/flat_hash_map.h>
#endif

struct intj_key {
  uint64_t w[INTJ_NWORDS];
  bool operator==(const intj_key &o) const {
    return memcmp(w, o.w, sizeof(w)) == 0;
  }
};

struct intj_key_hash {
  using is_avalanching = void; /* wyhash output: no further mixing wanted */
  size_t operator()(const intj_key &k) const { return (size_t)intj_hash(k.w); }
};

#if defined(INTJ_CACHE_TSL)
using intj_cache_map = tsl::robin_map<intj_key, intj_kernel *, intj_key_hash>;
#else
using intj_cache_map = absl::flat_hash_map<intj_key, intj_kernel *, intj_key_hash>;
#endif

typedef struct {
  intj_cache_map *map;
} intj_cache;

static inline int intj_cache_init(intj_cache *c) {
  c->map = new (std::nothrow) intj_cache_map();
  return c->map ? 0 : -1;
}

static inline intj_kernel *intj_cache_get(const intj_cache *c,
                                          const uint64_t *k, uint64_t h) {
  const intj_key *key = (const intj_key *)k;
#if defined(INTJ_CACHE_TSL)
  auto it = c->map->find(*key, (size_t)h);
#else
  (void)h;
  auto it = c->map->find(*key);
#endif
  return it == c->map->end() ? NULL : it->second;
}

/* Returns -1 on allocation failure (with no python error set).  The maps throw
 * where intj returns, and an exception reaching CPython's C frames is
 * std::terminate, so the throw stops here. */
static inline int intj_cache_put(intj_cache *c, const uint64_t *k, uint64_t h,
                                 intj_kernel *val) {
  (void)h;
  try {
    (*c->map)[*(const intj_key *)k] = val;
  } catch (...) {
    return -1;
  }
  return 0;
}

/* Frees intj's own allocations only -- the same contract as the intj backend:
 * the cache owns the kernel records, triton owns the kernels behind them. */
static inline void intj_cache_free(intj_cache *c) {
  if (!c->map)
    return;
  for (auto &entry : *c->map)
    PyMem_RawFree(entry.second);
  delete c->map;
  c->map = NULL;
}

#else /* INTJ_CACHE_INTJ */

typedef intj_map intj_cache;

static inline int intj_cache_init(intj_cache *c) { return intj_map_init(c, 16); }

static inline intj_kernel *intj_cache_get(const intj_cache *c,
                                          const uint64_t *k, uint64_t h) {
  return intj_map_get(c, k, h);
}

static inline int intj_cache_put(intj_cache *c, const uint64_t *k, uint64_t h,
                                 intj_kernel *val) {
  return intj_map_put(c, k, h, val);
}

/* Frees intj's own allocations only.  The kernel records are the cache's; the
 * kernels behind them belong to the callback's CompiledKernels. */
static inline void intj_cache_free(intj_cache *c) {
  if (!c->slots)
    return;
  for (uint32_t i = 0; i <= c->mask; i++)
    PyMem_RawFree(c->slots[i].val);
  PyMem_RawFree(c->slots);
  c->slots = NULL;
}

#endif

/* intj reads three things off a tensor: the data pointer, the dtype (as an
 * opaque int32 discriminator) and, where the backend specializes on pointer
 * range, the storage size.  Exactly one of three strategies is compiled in.
 *
 *   INTJ_ACCESS_SHIM     read torch's structs at offsets installed at load
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
  /* dtype code -> the spec key's 5-bit index, 0xFF where triton takes no such
   * dtype.  Unlike itemsize, every access mode reads this: it is what the key
   * encodes, not how a pointer is computed. */
  uint8_t dtype_index[INTJ_NDTYPES];
  int ready;               /* set_torch_version has run */
} intj_torch_abi;

#ifdef INTJ_ACCESS_CXX
/* Deliberately NOT torch/csrc/autograd/python_variable.h.  That header declares
 * the four-line struct below and nothing else intj needs, but it also pulls in
 * pybind11 -- 79% of the resulting .text, none of it reachable, and 9.5 s of
 * the 11 s build.  These three give byte-identical code for the reader.
 */
#include <ATen/core/Tensor.h>
#include <c10/core/StorageImpl.h>
#include <c10/core/TensorImpl.h>

/* The head of THPVariable.  Declaring it here rather than including torch's
 * copy asserts one thing: that `cdata` is the first member after PyObject_HEAD.
 * That assertion is checked at load against the verified layout for this torch
 * -- see set_torch_version.
 */
struct intj_THPVariable {
  PyObject_HEAD
  at::Tensor cdata;
};
#endif

/* Interned method names, used only by the CPYTHON reader. Filled by
 * intj_abi_init; never released, like every other module-lifetime singleton. */
static PyObject *intj_str_dtype;
static PyObject *intj_str_data_ptr;
static PyObject *intj_str_untyped_storage;
static PyObject *intj_str_nbytes;

#if defined(INTJ_ACCESS_SHIM)

/* Reads torch's structs directly. `want_size` is a compile-time-ish flag: the
 * storage size is only needed where the backend specializes on pointer range.
 */
static inline int intj_read_tensor(const intj_torch_abi *abi, PyObject *o,
                                   void **p, int32_t *dt, int want_size,
                                   int64_t *sz) {
  char *ti = *(char **)((char *)o + abi->cdata);
  char *si = *(char **)(ti + abi->storage);
  int64_t numel = *(const int64_t *)(ti + abi->numel);
  uint8_t code = *(const uint8_t *)(ti + abi->data_type);
  *dt = (int32_t)code;
  /* A null storage means the tensor has none at all -- sparse, and anything
   * else with a non-dense impl.  torch raises there, and so must this: handing
   * the kernel a null pointer instead would be a silent wrong launch.  Every
   * dense tensor has a storage, including a zero-element one.
   */
  if (INTJ_UNLIKELY(!si)) {
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
  if (INTJ_UNLIKELY(numel == 0)) {
    *p = NULL;
    if (want_size)
      *sz = *(const int64_t *)(si + abi->s_nbytes);
    return 0;
  }
  if (INTJ_UNLIKELY(code >= INTJ_NDTYPES || !abi->itemsize[code])) {
    /* A dtype intj has no element size for means the offsets are wrong, or
     * torch grew a dtype after the table was built.  Either way, refuse rather
     * than compute a pointer from a guess.
     */
    PyErr_Format(PyExc_RuntimeError,
                 "intj: unknown torch dtype code %d; this build's tensor "
                 "layout does not match the running torch",
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
  /* struct THPDtype { PyObject_HEAD at::ScalarType scalar_type; char name[65];
   * } ScalarType is `enum class : int8_t`, so this is the first byte of
   * payload. Only reached after intj_decode_argument's exact-type test, so `o` really is
   * a tensor and `d` really is a THPDtype.
   */
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

/* Exactly the loads the shim mode makes, with the compiler supplying the field
 * offsets instead of a probe.
 *
 * The fields are private, so they are reached through the
 * explicit-instantiation trick: [temp.spec]/6 says access checking does not
 * apply to names in an explicit instantiation, so a pointer-to-private-member
 * is legal there,
 * and the friend injected by `Rob` hands it back.  (Johannes Schaub, 2010:
 * bloglitb.blogspot.com/2010/07/access-to-private-members-thats-easy.html)
 *
 * This is deliberate, and it buys two things the public accessors cannot:
 *
 *  - No throw sites at all.  `t.data_ptr()`, `t.storage()`, `storage.nbytes()`
 *    and even `numel()` each hide a TORCH_CHECK, and an exception crossing
 *    into CPython's C frames is `std::terminate` -- measured, a core dump on a
 *    sparse tensor.  Reading the fields makes every failure intj can reach an
 *    explicit branch below, so this path needs no try/catch, which is worth
 *    ~4 ns per decode in inlining and block layout alone.
 *
 *  - No accessor torch declines to inline.  `StorageImpl::nbytes()` is defined
 *    in-class but goes through the PLT in a module this size, and
 *    `sym_nbytes()` materialises a `SymInt` whose refcounting copy and destroy
 *    cost ~129 instructions for a value never heap-allocated here.
 *
 * The cost is that a torch release renaming any of these fields breaks the
 * build.  That is the right failure: a compile error, not a wrong pointer --
 * which is what the shim mode would get, since it cannot see names at all.
 */namespace intj_rob {
template <typename Tag, typename Tag::type M>
struct Rob {
  friend typename Tag::type get(Tag) { return M; }
};
#define INTJ_ROB(NAME, CLASS, MEMBER, ...)                                     \
  struct NAME {                                                               \
    using type = __VA_ARGS__ CLASS::*;                                        \
    friend type get(NAME);                                                    \
  };                                                                          \
  template struct Rob<NAME, &CLASS::MEMBER>;

INTJ_ROB(ti_storage, c10::TensorImpl, storage_, c10::Storage)
INTJ_ROB(ti_numel, c10::TensorImpl, numel_, int64_t)
INTJ_ROB(ti_storage_offset, c10::TensorImpl, storage_offset_, int64_t)
INTJ_ROB(ti_data_type, c10::TensorImpl, data_type_, caffe2::TypeMeta)
INTJ_ROB(si_size_bytes, c10::StorageImpl, size_bytes_, c10::SymInt)
INTJ_ROB(si_data_ptr, c10::StorageImpl, data_ptr_, c10::DataPtr)
#undef INTJ_ROB
}  // namespace intj_rob

static INTJ_ALWAYS_INLINE int intj_read_cxx_tensor(const intj_torch_abi *abi,
                                                   const at::Tensor &t, void **p,
                                                   int32_t *dt, int want_size,
                                                   int64_t *sz) {
  using namespace intj_rob;
  (void)abi;
  c10::TensorImpl *ti = t.unsafeGetTensorImpl();

  /* No storage means no dense data pointer -- sparse, mkldnn.  torch raises
   * here and so does intj, rather than handing the kernel a null. */
  const c10::Storage &storage = ti->*get(ti_storage());
  if (INTJ_UNLIKELY(!storage)) {
    PyErr_SetString(PyExc_RuntimeError,
                    "intj: cannot access data pointer of a tensor with no "
                    "storage (a sparse tensor?)");
    return -1;
  }
  c10::StorageImpl *si = storage.unsafeGetStorageImpl();

  const caffe2::TypeMeta meta = ti->*get(ti_data_type());
  if (INTJ_UNLIKELY(!meta.isScalarType())) { /* also stops toScalarType() throwing */
    PyErr_SetString(PyExc_RuntimeError, "intj: tensor has no scalar dtype");
    return -1;
  }
  *dt = (int32_t)meta.toScalarType();

  if (want_size)
    *sz = (si->*get(si_size_bytes())).as_int_unchecked();

  /* torch returns null for every zero-element tensor, even one whose storage is
   * live and whose storage_offset is not zero; see the shim reader.  Reading
   * `numel_` rather than calling `numel()` also matches the shim mode on a
   * tensor whose impl overrides it -- see docs/Usage.md. */
  const int64_t itemsize = (int64_t)meta.itemsize();
  *p = (ti->*get(ti_numel())) == 0
           ? NULL
           : (void *)((char *)(si->*get(si_data_ptr())).get() +
                      (ti->*get(ti_storage_offset())) * itemsize);
  return 0;
}

static INTJ_ALWAYS_INLINE int intj_read_tensor(const intj_torch_abi *abi,
                                               PyObject *o, void **p,
                                               int32_t *dt, int want_size,
                                               int64_t *sz) {
  return intj_read_cxx_tensor(abi, ((intj_THPVariable *)o)->cdata,
                              p, dt, want_size, sz);
}

#else
#error "intj: define one of INTJ_ACCESS_{SHIM,CPYTHON,CXX}"
#endif

/* Re-raise the reader's error as "which argument", keeping torch's own message
 * as __cause__.  The reader knows what went wrong; only the caller knows which
 * parameter it was reading. */
static inline void intj_note_param(const char *pname) {
  PyObject *exc = PyErr_GetRaisedException();
  PyErr_Format(PyExc_RuntimeError, "intj: cannot read tensor argument '%s'",
               pname);
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
  int ok = code >= 0 && code < INTJ_NDTYPES &&
           strncmp(payload + 1, "float32", 8) == 0;
  Py_DECREF(f32);
  if (!ok) {
    PyErr_SetString(PyExc_RuntimeError,
                    "intj: torch.dtype is not laid out as intj expects "
                    "(THPDtype { PyObject_HEAD ScalarType; char name[] }); "
                    "this torch is too new or too old for the cpython mode");
    return -1;
  }
  return 0;
}

/* hipModuleLaunchKernel and cuLaunchKernel take the same arguments in the same
 * order.  Their error-string calls do not, hence the two typedefs. */
typedef int32_t (*intj_launch_t)(void *f, uint32_t gx, uint32_t gy, uint32_t gz,
                                 uint32_t bx, uint32_t by, uint32_t bz,
                                 uint32_t shared, void *stream, void **params,
                                 void **extra);
typedef const char *(*intj_error_string_ret_t)(int32_t);
typedef int32_t (*intj_error_string_out_t)(int32_t, const char **);

/* One byte per declared parameter.  The pointer range partitions the space, so
 * "is a pointer" needs no flag of its own, and `S` -- the one bit that is only
 * ever set for a pointer -- fits inside the byte rather than a section beside
 * it.  That is what the 5-bit compact dtype index buys: at torch's raw 6-bit
 * code the pointer codes alone would be all 256, leaving the scalars nowhere.
 */
#define INTJ_B_PTR(idx, d, s)                                                  \
  (((uint32_t)(idx) << 2) | ((uint32_t)(d) << 1) | (uint32_t)(s))
#define INTJ_B_I32 128u /* +1 when tt.divisibility = 16 */
#define INTJ_B_I64 130u /* +1 likewise */
#define INTJ_B_U64 132u /* +1 likewise */
#define INTJ_B_FP32 134u
#define INTJ_B_U1 135u
#define INTJ_B_NONE 136u
#define INTJ_B_ONE 137u /* int 1 folded to ("constexpr", 1) */
#define INTJ_B_CX_NONE 138u
#define INTJ_B_CX_BOOL 139u
#define INTJ_B_CX_INT 140u
#define INTJ_B_CX_UINT 141u
#define INTJ_B_CX_FLOAT 142u
#define INTJ_B_FP64 143u
#define INTJ_B_I8 144u
#define INTJ_B_U8 146u
#define INTJ_B_I16 148u
#define INTJ_B_U16 150u
#define INTJ_B_U32 152u
/* 154..255 unused. Integer codes reserve their low bit for alignment.
 * The constexpr codes are disjoint from the rest even though
 * a byte position is always known at render time to be one or the other: it
 * costs nothing out of the spare codes, and it turns a wrong render-time offset
 * into a nonsense code rather than a silent alias. */

/* Structural decoding is independent of template-local state.  It reads each
 * Python object once; rendering owns semantic promises and GPU packing. */
typedef enum {
  INTJ_VALUE_TENSOR,
  INTJ_VALUE_BOOL,
  INTJ_VALUE_I64,
  INTJ_VALUE_U64,
  INTJ_VALUE_FP64,
  INTJ_VALUE_NONE
} intj_value_kind;

typedef struct {
  intj_value_kind kind;
  uint64_t bits;
  void *pointer;
  int64_t storage_nbytes;
  uint8_t dtype_index;
} intj_decoded;

static INTJ_ALWAYS_INLINE int
intj_decode_constexpr(PyObject *o, const char *pname, intj_decoded *out) {
  memset(out, 0, sizeof(*out));
  if (o == Py_True || o == Py_False) {
    out->kind = INTJ_VALUE_BOOL;
    out->bits = (uint64_t)(o == Py_True);
  } else if (PyLong_CheckExact(o)) {
    int kind = intj_as_int(o, &out->bits);
    if (INTJ_UNLIKELY(kind == INTJ_INT_TOO_BIG)) {
      PyErr_Format(PyExc_OverflowError,
                   "intj: integer argument '%s' is too large", pname);
      return -1;
    }
    out->kind = kind == INTJ_INT_U64 ? INTJ_VALUE_U64 : INTJ_VALUE_I64;
  } else if (PyFloat_CheckExact(o)) {
    out->kind = INTJ_VALUE_FP64;
    /* Reading ob_fval directly keeps g++ from leaving an accessor call here. */
    double value = ((PyFloatObject *)o)->ob_fval;
    memcpy(&out->bits, &value, sizeof(value));
  } else if (o == Py_None) {
    out->kind = INTJ_VALUE_NONE;
  } else {
    PyErr_Format(PyExc_TypeError,
                 "intj: unsupported argument '%s' of type %s; pass a "
                 "scalar int, float, bool or None", pname, Py_TYPE(o)->tp_name);
    return -1;
  }
  return 0;
}

static INTJ_ALWAYS_INLINE int
intj_finish_tensor(const intj_torch_abi *abi, int32_t dtype,
                    const char *pname, intj_decoded *out) {
  /* All three readers, including zero-element tensors, share this gate. */
  uint32_t index = (dtype >= 0 && dtype < INTJ_NDTYPES)
                       ? abi->dtype_index[dtype] : 0xFFu;
  if (INTJ_UNLIKELY(index == 0xFFu)) {
    PyErr_Format(PyExc_RuntimeError,
                 "intj: tensor argument '%s' has dtype code %d, which "
                 "triton does not take", pname, (int)dtype);
    return -1;
  }
  out->kind = INTJ_VALUE_TENSOR;
  out->bits = (uint64_t)(uintptr_t)out->pointer;
  out->dtype_index = (uint8_t)index;
  return 0;
}

static INTJ_ALWAYS_INLINE int
intj_decode_argument(const intj_torch_abi *abi, PyTypeObject *tensor_type,
                     PyTypeObject *param_type, PyObject *o, int want_size,
                     const char *pname, intj_decoded *out) {
  PyTypeObject *type = Py_TYPE(o);
  if (type != tensor_type && type != param_type)
    return intj_decode_constexpr(o, pname, out);
  memset(out, 0, sizeof(*out));
  int32_t dtype = -1;
  if (INTJ_UNLIKELY(intj_read_tensor(abi, o, &out->pointer, &dtype,
                                    want_size, &out->storage_nbytes) != 0)) {
    intj_note_param(pname);
    return -1;
  }
  return intj_finish_tensor(abi, dtype, pname, out);
}

#ifndef INTJ_NBOUND
#define INTJ_NBOUND 0
#endif
#define INTJ_BOUND_SLOTS (INTJ_NBOUND ? INTJ_NBOUND : 1)

typedef struct intj_bound_launcher {
  PyObject_HEAD
  PyObject *module;
  PyObject *owners[INTJ_BOUND_SLOTS];
  uint64_t pointer_bits[INTJ_BOUND_SLOTS];
  intj_cache cache;
  int cache_ready;
  intj_kernel *fixed_kernel;
  int fixed_device;
  int64_t device_ordinal;
  int32_t device_handle;
#if defined(INTJ_ACCESS_CXX)
  at::Tensor *tensors[INTJ_BOUND_SLOTS];
#endif
} intj_bound_launcher;

static inline int
intj_bind_tensor(intj_bound_launcher *bound, int index, PyObject *value,
                 PyTypeObject *tensor_type, PyTypeObject *param_type,
                 const char *pname) {
  if (value != Py_None && Py_TYPE(value) != tensor_type && Py_TYPE(value) != param_type) {
    PyErr_Format(PyExc_TypeError, "intj: bound tensor '%s' must be a tensor or None", pname);
    return -1;
  }
#if defined(INTJ_ACCESS_CXX)
  if (value != Py_None) {
    bound->tensors[index] = new (std::nothrow) at::Tensor(((intj_THPVariable *)value)->cdata);
    if (!bound->tensors[index]) {
      PyErr_NoMemory();
      return -1;
    }
  }
#else
  bound->owners[index] = Py_NewRef(value);
#endif
  return 0;
}

static INTJ_ALWAYS_INLINE int
intj_decode_bound_tensor(const intj_torch_abi *abi, PyTypeObject *tensor_type,
                         PyTypeObject *param_type, intj_bound_launcher *bound,
                         int index, int want_size, const char *pname, intj_decoded *out) {
#if defined(INTJ_ACCESS_CXX)
  memset(out, 0, sizeof(*out));
  if (!bound->tensors[index]) {
    out->kind = INTJ_VALUE_NONE;
    return 0;
  }
  int32_t dtype = -1;
  if (INTJ_UNLIKELY(intj_read_cxx_tensor(abi, *bound->tensors[index], &out->pointer,
                                        &dtype, want_size, &out->storage_nbytes) != 0)) {
    intj_note_param(pname);
    return -1;
  }
  return intj_finish_tensor(abi, dtype, pname, out);
#else
  return intj_decode_argument(abi, tensor_type, param_type, bound->owners[index],
                              want_size, pname, out);
#endif
}

/* A POINTER is decoded only during bind. Addresses carry no allocation range. */
static inline int
intj_decode_pointer(const intj_torch_abi *abi, PyTypeObject *tensor_type,
                     PyTypeObject *param_type, PyObject *value,
                     const char *pname, intj_decoded *out) {
  if (Py_TYPE(value) == tensor_type || Py_TYPE(value) == param_type)
    return intj_decode_argument(abi, tensor_type, param_type, value, 0, pname, out);
  if (value == Py_None || PyLong_CheckExact(value))
    return intj_decode_constexpr(value, pname, out);
  PyObject *address = PyObject_CallMethodNoArgs(value, intj_str_data_ptr);
  if (!address) {
    if (PyErr_ExceptionMatches(PyExc_AttributeError))
      PyErr_Format(PyExc_TypeError, "intj: bound pointer '%s' needs a tensor, int, data_ptr() or None", pname);
    return -1;
  }
  if (!PyLong_CheckExact(address)) {
    Py_DECREF(address);
    PyErr_Format(PyExc_TypeError, "intj: bound pointer '%s' data_ptr() must return an exact int", pname);
    return -1;
  }
  int rc = intj_decode_constexpr(address, pname, out);
  Py_DECREF(address);
  return rc;
}

static INTJ_ALWAYS_INLINE double intj_as_double(uint64_t bits) {
  double value;
  memcpy(&value, &bits, sizeof(value));
  return value;
}

static INTJ_ALWAYS_INLINE uint32_t intj_infer_type(const intj_decoded *value) {
  switch (value->kind) {
  case INTJ_VALUE_TENSOR: return INTJ_B_PTR(value->dtype_index, 0, 0);
  case INTJ_VALUE_BOOL: return INTJ_B_U1;
  case INTJ_VALUE_I64:
    return (int64_t)value->bits >= INT32_MIN && (int64_t)value->bits <= INT32_MAX
               ? INTJ_B_I32 : INTJ_B_I64;
  case INTJ_VALUE_U64: return INTJ_B_U64;
  case INTJ_VALUE_FP64: return INTJ_B_FP32;
  case INTJ_VALUE_NONE: return INTJ_B_NONE;
  }
  return INTJ_B_NONE;
}

static INTJ_ALWAYS_INLINE uint32_t intj_constexpr_type(const intj_decoded *value) {
  switch (value->kind) {
  case INTJ_VALUE_BOOL: return INTJ_B_CX_BOOL;
  case INTJ_VALUE_I64: return INTJ_B_CX_INT;
  case INTJ_VALUE_U64: return INTJ_B_CX_UINT;
  case INTJ_VALUE_FP64: return INTJ_B_CX_FLOAT;
  default: return INTJ_B_CX_NONE;
  }
}

/* Magnitude and sign are separate so both +2**63 and -2**63 are representable. */
static INTJ_ALWAYS_INLINE uint8_t
intj_power_of_two_or_zero(uint64_t magnitude, int negative) {
  if (magnitude == 0)
    return 0;
  uint8_t code = (uint8_t)(__builtin_ctzll(magnitude) + 1);
  return negative ? (uint8_t)(0x80u | code) : code;
}

static INTJ_ALWAYS_INLINE int intj_integer_type(uint32_t type) {
  return type == INTJ_B_I32 || type == INTJ_B_I64 || type == INTJ_B_U64 ||
         (type >= INTJ_B_I8 && type <= INTJ_B_U32);
}

static INTJ_ALWAYS_INLINE uint64_t intj_pack_value(uint64_t bits, uint32_t type) {
  if (type == INTJ_B_FP32) {
    float value = (float)intj_as_double(bits);
    uint32_t packed;
    memcpy(&packed, &value, sizeof(packed));
    return packed;
  }
  return bits;
}

/* Keys are little endian, independent of the host's byte order. */
#define INTJ_KEY_STORE(WIDTH)                                                  \
  static INTJ_ALWAYS_INLINE void intj_key_store##WIDTH(                        \
      void *out, uint##WIDTH##_t value) {                                      \
    if (PY_BIG_ENDIAN)                                                        \
      value = __builtin_bswap##WIDTH(value);                                  \
    memcpy(out, &value, sizeof(value));                                        \
  }
INTJ_KEY_STORE(16)
INTJ_KEY_STORE(32)
INTJ_KEY_STORE(64)
#undef INTJ_KEY_STORE

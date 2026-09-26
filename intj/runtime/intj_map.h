/* Kernel-cache maps: the key hash, intj's open-addressed map instantiated per
 * key width and value type, and the swappable level-0 cache.
 *
 * Values live in the map.  A rehash moves them, so a `V *` from a lookup or put
 * is valid until the next put into that map.  A launch holds the module's read
 * lock across its lookups and the launch; a put holds the write lock.  A value
 * that must not move -- a record owning a child map -- is stored by pointer.
 */
#pragma once

#include <Python.h>
#include <stdint.h>
#include <string.h>

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

static inline uint64_t intj_mix(uint64_t a, uint64_t b) {
  __uint128_t r = (__uint128_t)a * b;
  return (uint64_t)(r >> 64) ^ (uint64_t)r;
}

/* `nw` is a compile-time constant at every call site, so each one folds to the
 * shape its width needs.  At one word the mix is a bijection -- xorshift-right
 * and an odd multiply both are -- so a one-word slot keeps the key and
 * recomputes the hash when it needs one.  At two, one multiply suffices, the way wyhash
 * handles a short input.  Above that the chain is split across two lanes: a mix
 * is a ~4-cycle multiply sitting between the last decode and the first probe,
 * so five words is ~20 cycles serially and ~12 in pairs. */
static INTJ_ALWAYS_INLINE uint64_t intj_hash_n(const uint64_t *w, int nw) {
  if (nw == 1) {
    uint64_t x = w[0];
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ull;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebull;
    x ^= x >> 31;
    return x;
  }
  const uint64_t s0 = 0xa0761d6478bd642full, s1 = 0xe7037ed1a0b428dbull;
  if (nw == 2)
    return intj_mix(intj_mix(w[0] ^ s0, w[1] ^ s1), 2 * 8 + s1);
  uint64_t h0 = s0, h1 = s1;
  int i = 0;
  for (; i + 1 < nw; i += 2) {
    h0 = intj_mix(h0 ^ s1, w[i] ^ s0);
    h1 = intj_mix(h1 ^ s0, w[i + 1] ^ s1);
  }
  if (i < nw)
    h0 = intj_mix(h0 ^ s1, w[i] ^ s0);
  return intj_mix(h0 ^ h1, (uint64_t)nw * 8 + s1);
}

#ifdef INTJ_NWORDS
static inline uint64_t intj_hash(const uint64_t *w) {
  return intj_hash_n(w, INTJ_NWORDS);
}
#endif

typedef struct {
  void *function;     /* hipFunction_t / CUfunction */
  uint32_t block_dim; /* warp_size * num_warps */
  uint32_t shared;    /* dynamic LDS bytes */
  uint32_t nparams;   /* kernel params, scratch slots excluded */
#ifdef INTJ_RETURN_COMPILED
  PyObject *compiled; /* owns the CompiledKernel behind function */
#endif
} intj_kernel;

/* `full` marks occupancy because no value field is free to act as a sentinel.
 * A slot stores its key and, above one word, its hash.  At one word the hash is
 * a bijection of the key, so it is left out -- a zero-length array, which gcc
 * and clang accept in C and C++ -- and recomputed by a rehash.  Keeping the key
 * rather than the hash lets the memo compare without hashing, which measured
 * ~1.3 ns on the launch path; keeping both costs a probe ~2 ns on a miss.
 *
 * `last` is the one-entry memo: the slot of the last hit.  Readers under the read lock store `last` with relaxed atomics; a slot cannot
 * move while any reader holds the lock, and a put -- under the write lock --
 * clears it.
 *
 * gcc 13.3.0 miscompiled the old one-word lookup at `-O1 -fsanitize=undefined`
 * (a present key read as absent, with no UBSan diagnostic); -O2/-O3 and clang
 * were clean.  Nothing intj builds uses those flags.  Recheck here first if a
 * sanitizer build of `tests/bench_kernel_cache.cpp` reports a wrong kernel. */
#define INTJ_DEFINE_MAP(P, NW, V, CAP)                                         \
  typedef struct {                                                             \
    uint64_t full;                                                             \
    uint64_t hash[(NW) > 1];                                                   \
    uint64_t key[NW];                                                          \
    V val;                                                                     \
  } P##_slot;                                                                  \
  static INTJ_ALWAYS_INLINE uint64_t P##_slot_hash(const P##_slot *s) {        \
    uint64_t h = 0;                                                            \
    memcpy(&h, s->hash, sizeof(s->hash));                                      \
    return (NW) > 1 ? h : intj_hash_n(s->key, 1);                              \
  }                                                                            \
  typedef struct {                                                             \
    P##_slot *slots;                                                           \
    uint32_t mask;                                                             \
    uint32_t used;                                                             \
    P##_slot *last;                                                            \
  } P;                                                                         \
  static inline int P##_init(P *m) {                                           \
    m->slots = (P##_slot *)PyMem_RawCalloc((CAP), sizeof(P##_slot));           \
    m->mask = (CAP) - 1;                                                       \
    m->used = 0;                                                               \
    m->last = NULL;                                                            \
    return m->slots ? 0 : -1;                                                  \
  }                                                                            \
  static INTJ_ALWAYS_INLINE P##_slot *P##_find(const P *m, const uint64_t *k,  \
                                               uint64_t h) {                   \
    uint32_t i = (uint32_t)h & m->mask;                                        \
    for (;;) {                                                                 \
      P##_slot *s = &m->slots[i];                                              \
      if (INTJ_UNLIKELY(!s->full))                                             \
        return NULL;                                                           \
      if (INTJ_LIKELY(((NW) == 1 || P##_slot_hash(s) == h) &&                  \
                      memcmp(s->key, k, sizeof(s->key)) == 0))                 \
        return s;                                                              \
      i = (i + 1) & m->mask;                                                   \
    }                                                                          \
  }                                                                            \
  static INTJ_ALWAYS_INLINE V *P##_get(const P *m, const uint64_t *k,          \
                                       uint64_t h) {                           \
    P##_slot *s = P##_find(m, k, h);                                           \
    return s ? &s->val : NULL;                                                 \
  }                                                                            \
  static INTJ_ALWAYS_INLINE V *P##_lookup(P *m, const uint64_t *k) {           \
    P##_slot *s = __atomic_load_n(&m->last, __ATOMIC_RELAXED);                 \
    if (s && memcmp(s->key, k, sizeof(s->key)) == 0)                           \
      return &s->val;                                                          \
    uint64_t h = intj_hash_n(k, (NW));                                         \
    s = P##_find(m, k, h);                                                     \
    if (!s)                                                                    \
      return NULL;                                                             \
    __atomic_store_n(&m->last, s, __ATOMIC_RELAXED);                           \
    return &s->val;                                                            \
  }                                                                            \
  static inline V *P##_put(P *m, const uint64_t *k, const V *val) {            \
    if ((m->used + 1) * 2 > m->mask + 1) {                                     \
      uint32_t cap = (m->mask + 1) * 2;                                        \
      P##_slot *grown = (P##_slot *)PyMem_RawCalloc(cap, sizeof(P##_slot));    \
      if (!grown)                                                              \
        return NULL;                                                           \
      for (uint32_t i = 0; i <= m->mask; i++) {                                \
        if (!m->slots[i].full)                                                 \
          continue;                                                            \
        uint32_t j = (uint32_t)P##_slot_hash(&m->slots[i]) & (cap - 1);        \
        while (grown[j].full)                                                  \
          j = (j + 1) & (cap - 1);                                             \
        grown[j] = m->slots[i];                                                \
      }                                                                        \
      PyMem_RawFree(m->slots);                                                 \
      m->slots = grown;                                                        \
      m->mask = cap - 1;                                                       \
    }                                                                          \
    uint64_t h = intj_hash_n(k, (NW));                                         \
    uint32_t i = (uint32_t)h & m->mask;                                        \
    while (m->slots[i].full)                                                   \
      i = (i + 1) & m->mask;                                                   \
    P##_slot *s = &m->slots[i];                                                \
    memcpy(s->hash, &h, sizeof(s->hash));                                      \
    s->full = 1;                                                               \
    memcpy(s->key, k, sizeof(s->key));                                         \
    s->val = *val;                                                             \
    m->used++;                                                                 \
    __atomic_store_n(&m->last, (P##_slot *)NULL, __ATOMIC_RELAXED);            \
    return &s->val;                                                            \
  }                                                                            \
  static inline int P##_each(const P *m, int (*fn)(V *, void *), void *ctx) {  \
    if (!m->slots)                                                             \
      return 0;                                                                \
    for (uint32_t i = 0; i <= m->mask; i++)                                    \
      if (m->slots[i].full) {                                                  \
        int r = fn(&m->slots[i].val, ctx);                                     \
        if (r)                                                                 \
          return r;                                                            \
      }                                                                        \
    return 0;                                                                  \
  }                                                                            \
  /* Detach before releasing: a CompiledKernel finalizer may reenter GC. */    \
  static inline void P##_free(P *m, void (*release)(V *)) {                    \
    P##_slot *slots = m->slots;                                                \
    uint32_t mask = m->mask;                                                   \
    m->slots = NULL;                                                           \
    m->used = 0;                                                               \
    m->last = NULL;                                                            \
    if (!slots)                                                                \
      return;                                                                  \
    for (uint32_t i = 0; i <= mask; i++)                                       \
      if (slots[i].full)                                                       \
        release(&slots[i].val);                                                \
    PyMem_RawFree(slots);                                                      \
  }

/* The level-0 cache behind one interface, so the entry template never names an
 * implementation.  INTJ_CACHE_{INTJ,TSL,ABSL} selects it; the last two are C++
 * and force the module to be compiled as C++.  `intj_cache` is POD in every
 * mode -- it lives in zeroed module state -- so the C++ maps are held by
 * pointer.  Only tsl takes the precomputed hash; abseil re-computes it, which
 * is the measured cost of that option. */
#if defined(INTJ_CACHE_TSL) || defined(INTJ_CACHE_ABSL)

#include <new>
#if defined(INTJ_CACHE_TSL)
#include <tsl/robin_map.h>
#define INTJ_CACHE_MAP(V) tsl::robin_map<intj_key, V, intj_key_hash>
#define INTJ_CACHE_FIND(map, key, h) (map)->find((key), (size_t)(h))
#else
#include <absl/container/flat_hash_map.h>
#define INTJ_CACHE_MAP(V) absl::flat_hash_map<intj_key, V, intj_key_hash>
#define INTJ_CACHE_FIND(map, key, h) ((void)(h), (map)->find(key))
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

/* Entries are const_cast back to V: tsl hands out const pairs so the key
 * cannot change, but the stored object itself is not const. */
#define INTJ_DEFINE_CACHE(V)                                                    \
  typedef INTJ_CACHE_MAP(V) intj_cache_map;                                     \
  typedef intj_cache_map::value_type intj_cache_entry;                          \
  typedef struct {                                                              \
    intj_cache_map *map;                                                        \
    const intj_cache_entry *last;                                               \
  } intj_cache;                                                                 \
  static inline int intj_cache_init(intj_cache *c) {                            \
    c->map = new (std::nothrow) intj_cache_map();                               \
    c->last = NULL;                                                             \
    return c->map ? 0 : -1;                                                     \
  }                                                                             \
  static INTJ_ALWAYS_INLINE V *intj_cache_get(const intj_cache *c,              \
                                              const uint64_t *k, uint64_t h) {  \
    auto it = INTJ_CACHE_FIND(c->map, *(const intj_key *)k, h);                 \
    return it == c->map->end() ? NULL : const_cast<V *>(&it->second);           \
  }                                                                             \
  static INTJ_ALWAYS_INLINE V *intj_cache_lookup(intj_cache *c,                 \
                                                 const uint64_t *k) {           \
    const intj_cache_entry *e = __atomic_load_n(&c->last, __ATOMIC_RELAXED);    \
    if (e && memcmp(e->first.w, k, sizeof(e->first.w)) == 0)                    \
      return const_cast<V *>(&e->second);                                       \
    auto it = INTJ_CACHE_FIND(c->map, *(const intj_key *)k, intj_hash(k));      \
    if (it == c->map->end())                                                    \
      return NULL;                                                              \
    e = &*it;                                                                   \
    __atomic_store_n(&c->last, e, __ATOMIC_RELAXED);                            \
    return const_cast<V *>(&e->second);                                         \
  }                                                                             \
  /* The maps throw where intj returns, and an exception reaching CPython's   \
   * C frames is std::terminate, so the throw stops here. */ \
  static inline V *intj_cache_put(intj_cache *c, const uint64_t *k,             \
                                  const V *val) {                               \
    try {                                                                       \
      auto r = c->map->insert(intj_cache_entry(*(const intj_key *)k, *val));    \
      __atomic_store_n(&c->last, (const intj_cache_entry *)NULL,                \
                       __ATOMIC_RELAXED);                                       \
      return const_cast<V *>(&r.first->second);                                 \
    } catch (...) {                                                             \
      return NULL;                                                              \
    }                                                                           \
  }                                                                             \
  static inline int intj_cache_each(const intj_cache *c,                        \
                                    int (*fn)(V *, void *), void *ctx) {        \
    if (!c->map)                                                                \
      return 0;                                                                 \
    for (const auto &e : *c->map) {                                             \
      int r = fn(const_cast<V *>(&e.second), ctx);                              \
      if (r)                                                                    \
        return r;                                                               \
    }                                                                           \
    return 0;                                                                   \
  }                                                                             \
  static inline void intj_cache_free(intj_cache *c, void (*release)(V *)) {     \
    intj_cache_map *map = c->map;                                               \
    c->map = NULL;                                                              \
    c->last = NULL;                                                             \
    if (!map)                                                                   \
      return;                                                                   \
    for (auto &e : *map)                                                        \
      release(const_cast<V *>(&e.second));                                      \
    delete map;                                                                 \
  }

#else /* INTJ_CACHE_INTJ */

#define INTJ_DEFINE_CACHE(V) INTJ_DEFINE_MAP(intj_cache, INTJ_NWORDS, V, 16)

#endif

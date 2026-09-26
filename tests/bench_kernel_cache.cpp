/* What the hash map behind `kernel_cache=` costs, without the last-key shortcut.
 *
 * Built once per backend, since INTJ_CACHE_* picks the implementation at
 * compile time and all three define `intj_cache`.  `tests/test_kernel_cache.py`
 * does the building and runs whichever backends are available.
 *
 *   g++ -O3 -DNDEBUG -DINTJ_CACHE_INTJ -DINTJ_ACCESS_SHIM -DINTJ_NWORDS=5 \
 *       -I intj/runtime -I $(python3-config --includes) tests/bench_kernel_cache.cpp \
 *       -lbenchmark -lpython3.12 -o bench
 *
 * The workload is a key of INTJ_NWORDS uint64 words, insert-only, ~100% map
 * hit, and the hash already computed by the caller -- which is why the
 * lookup takes it as an argument.  `entries` is how many specializations of one
 * kernel are live; real ones sit at 1-8, 512 is there to show the cache
 * behaviour.
 *
 * It is built at three key lengths, which are the three the packed layout
 * produces and so also the three specializations in the header: 1, where the
 * hash is a bijection and the slot carries no key at all; 2, one multiply; and
 * 5, the loop.  Nothing here forks on INTJ_NWORDS -- `intj_hash` and
 * `intj_cache_*` keep one signature at every length, which is what lets this
 * file measure the underlying map for every key size.
 *
 * Two things to keep in mind reading the output.  `hit/1` and `miss/1` index
 * with `& 0`, so they always touch element 0 and measure a permanently hot
 * line.  And the keys here are uniformly random words, where a real key is a
 * header of small byte codes plus a few value words -- fine for the table, but
 * not a test of the layout's own distribution.
 *
 * If you build this by hand at `-O1` with `-fsanitize=undefined` and
 * `INTJ_NWORDS=1`, every `hit/*` reports "lookup returned the wrong kernel".
 * That is a gcc 13.3.0 wrong-code bug, not a bug in the cache -- see the note
 * above `intj_slot` in intj_runtime.h for what was ruled out.  `-O2` and `-O3`,
 * which is what everything intj actually builds uses, are clean.
 */
#include <benchmark/benchmark.h>

#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

#define INTJ_ACCESS_SHIM /* the cache does not touch the tensor reader */
#include "intj_runtime.h"

#ifndef INTJ_CACHE_NAME
#define INTJ_CACHE_NAME "unknown"
#endif

namespace {

std::vector<std::vector<uint64_t>> make_keys(int n) {
  std::mt19937_64 rng(20260923);
  std::vector<std::vector<uint64_t>> keys(n, std::vector<uint64_t>(INTJ_NWORDS));
  for (auto &key : keys)
    for (auto &word : key)
      word = rng();
  return keys;
}

/* One lookup of a key that is present, the launch-path case. */
void hit(benchmark::State &state) {
  const int n = (int)state.range(0);
  auto keys = make_keys(n);
  std::vector<uint64_t> hashes;
  for (auto &key : keys)
    hashes.push_back(intj_hash(key.data()));
  /* the cache frees these: the records are its, per intj_cache_free */
  std::vector<intj_kernel *> kernels(n);

  intj_cache cache;
  if (intj_cache_init(&cache) != 0) {
    state.SkipWithError("intj_cache_init failed");
    return;
  }
  for (int i = 0; i < n; i++) {
    kernels[i] = (intj_kernel *)PyMem_RawCalloc(1, sizeof(intj_kernel));
    if (intj_cache_put(&cache, keys[i].data(), hashes[i], kernels[i]) != 0) {
      state.SkipWithError("intj_cache_put failed");
      intj_cache_free(&cache);
      return;
    }
  }

  int i = 0;
  for (auto _ : state) {
    const int j = i++ & (n - 1);
    intj_kernel *found = intj_cache_get(&cache, keys[j].data(), hashes[j]);
    benchmark::DoNotOptimize(found);
    if (found != kernels[j])
      state.SkipWithError("lookup returned the wrong kernel");
  }
  intj_cache_free(&cache);
  state.SetLabel(INTJ_CACHE_NAME);
}

/* A key that is absent: what a launch pays before it compiles. */
void miss(benchmark::State &state) {
  const int n = (int)state.range(0);
  auto keys = make_keys(n);
  auto absent = make_keys(n);
  for (auto &key : absent)
    key[0] ^= 0x9e3779b97f4a7c15ull; /* same shape, not in the table */
  std::vector<uint64_t> hashes, absent_hashes;
  for (auto &key : keys)
    hashes.push_back(intj_hash(key.data()));
  for (auto &key : absent)
    absent_hashes.push_back(intj_hash(key.data()));
  intj_cache cache;
  if (intj_cache_init(&cache) != 0) {
    state.SkipWithError("intj_cache_init failed");
    return;
  }
  for (int i = 0; i < n; i++)
    intj_cache_put(&cache, keys[i].data(), hashes[i],
                   (intj_kernel *)PyMem_RawCalloc(1, sizeof(intj_kernel)));

  int i = 0;
  for (auto _ : state) {
    const int j = i++ & (n - 1);
    intj_kernel *found = intj_cache_get(&cache, absent[j].data(), absent_hashes[j]);
    benchmark::DoNotOptimize(found);
    if (found != NULL)
      state.SkipWithError("a key that was never inserted was found");
  }
  intj_cache_free(&cache);
  state.SetLabel(INTJ_CACHE_NAME);
}

/* The hash itself: the launcher computes it on a last-key miss, and only tsl
 * can be handed the result, so this is the floor the other two cannot reach. */
void hash_only(benchmark::State &state) {
  auto keys = make_keys(8);
  int i = 0;
  for (auto _ : state) {
    uint64_t h = intj_hash(keys[i++ & 7].data());
    benchmark::DoNotOptimize(h);
  }
  state.SetLabel(INTJ_CACHE_NAME);
}

}  // namespace

BENCHMARK(hit)->Arg(1)->Arg(8)->Arg(64)->Arg(512);
BENCHMARK(miss)->Arg(1)->Arg(8)->Arg(64)->Arg(512);
BENCHMARK(hash_only);

BENCHMARK_MAIN();

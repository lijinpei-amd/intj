// What TVM FFI's per-argument dispatch costs on its own: one
// std::unordered_map<PyTypeObject*, setter>::find per argument
// (python/tvm_ffi/cython/tvm_ffi_python_helpers.h, SetArgument), against a direct
// type-pointer compare, which is how a decoder specialized to the signature (intj,
// or a schema-typed path) picks the setter.
//
//   g++ -O2 -std=c++17 bench_unordered_map_lookup.cpp -o bench && taskset -c <cpu> ./bench
//
// Keys are addresses of heap blocks the size of a PyTypeObject (~400 B), as type
// objects are; the map is built like TVM FFI's (reserve(32), then emplace). "light"
// holds the 6 types a launch-style call site passes; "heavy" adds 1024 more first.
// Each sample looks up one 4-argument call's types (int, float, tensor, None) in a
// loop; "spread" looks up 4 keys drawn from all live entries, so the touched nodes
// change call to call. Prints one JSON line per case: ns per argument, median of
// 9 samples.
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <random>
#include <unordered_map>
#include <vector>

using Setter = int (*)(void *, void *, void *);
static int setter_a(void *, void *, void *) { return 0; }

struct Fake {
  char bytes[408]; // sizeof(PyTypeObject) on CPython 3.12, x86-64
};

static volatile Setter sink;

template <class F> static double sample_ns(F &&body, long iters) {
  auto t0 = std::chrono::steady_clock::now();
  body(iters);
  auto t1 = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::nano>(t1 - t0).count() / iters;
}

template <class F>
static void report(const char *name, size_t entries, size_t buckets, F &&body,
                   long iters) {
  body(iters / 10); // warm-up
  std::vector<double> s;
  for (int i = 0; i < 9; ++i)
    s.push_back(sample_ns(body, iters));
  std::sort(s.begin(), s.end());
  std::printf("{\"case\": \"%s\", \"entries\": %zu, \"buckets\": %zu, "
              "\"ns_per_arg\": %.3f, \"lo\": %.3f, "
              "\"hi\": %.3f}\n",
              name, entries, buckets, s[4] / 4, s[0] / 4, s[8] / 4);
}

int main() {
  const long iters = 20'000'000;
  std::vector<Fake *> types;
  for (int i = 0; i < 6 + 1024; ++i)
    types.push_back(new Fake());
  std::mt19937_64 rng(0);
  std::shuffle(types.begin() + 6, types.end(),
               rng); // heap order is not map order

  // The call: 4 arguments of 4 of the first 6 types.
  void *call[4] = {types[0], types[1], types[2], types[3]};

  for (int heavy = 0; heavy < 2; ++heavy) {
    std::unordered_map<void *, Setter> map;
    map.reserve(32);
    size_t n = heavy ? types.size() : 6;
    // The heavy map saw the other types first, as a long-running process would.
    for (size_t i = n; i-- > 0;)
      map.emplace(types[i], &setter_a);
    const char *load = heavy ? "heavy" : "light";
    char name[64];

    std::snprintf(name, sizeof name, "unordered_map find, %s", load);
    report(
        name, map.size(), map.bucket_count(),
        [&](long k) {
          for (long i = 0; i < k; ++i)
            for (void *t : call) {
              auto it = map.find(t);
              sink = it->second;
            }
        },
        iters / 4);

    std::vector<void *> spread;
    std::mt19937_64 pick(1);
    for (int i = 0; i < 4096; ++i)
      spread.push_back(types[pick() % n]);
    std::snprintf(name, sizeof name, "unordered_map find, %s, spread keys",
                  load);
    report(
        name, map.size(), map.bucket_count(),
        [&](long k) {
          for (long i = 0; i < k; ++i)
            for (int j = 0; j < 4; ++j) {
              auto it = map.find(spread[(i * 4 + j) & 4095]);
              sink = it->second;
            }
        },
        iters / 4);
  }

  // A decoder specialized to the signature: compare the type pointer it expects.
  report(
      "direct type compare", 0, 0,
      [&](long k) {
        for (long i = 0; i < k; ++i)
          for (int j = 0; j < 4; ++j) {
            void *volatile t = call[j];
            sink = (t == types[j]) ? &setter_a : nullptr;
          }
      },
      iters / 4);
  return 0;
}

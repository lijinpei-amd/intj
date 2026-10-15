#!/bin/bash
# usage: run_ffi_dispatch.sh <out-dir>   from an intj root; env: PY, CPU (and the machine's
# PYTHONPATH with apache-tvm-ffi). Builds bench_unordered_map_lookup.cpp with g++ -O2, then
# three process rounds of it and of bench_ffi_dispatch.py: r<N>_umap.* and r<N>_ffi_dispatch.*.
set -u
OUT=$1; mkdir -p "$OUT"
SCRIPTS=$(cd "$(dirname "$0")" && pwd)
g++ -O2 -std=c++17 "$SCRIPTS/bench_unordered_map_lookup.cpp" -o "$OUT/bench_umap" > "$OUT/build.log" 2>&1 || exit 1
{ g++ --version | head -1; uptime; } > "$OUT/state.txt"
run() {
  local name=$1; shift
  { echo "started=$(date -Iseconds)"; echo "cwd=$PWD"; echo "command=taskset -c $CPU $*"; } > "$OUT/$name.meta"
  taskset -c "$CPU" "$@" > "$OUT/$name.out" 2> "$OUT/$name.err"
  echo "exit_status=$? ended=$(date -Iseconds)" >> "$OUT/$name.meta"
}
for r in 0 1 2; do
  run r${r}_umap "$OUT/bench_umap"
  run r${r}_ffi_dispatch "$PY" "$SCRIPTS/bench_ffi_dispatch.py"
done
date -Iseconds > "$OUT/DONE"

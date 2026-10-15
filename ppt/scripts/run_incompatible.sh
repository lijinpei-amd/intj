#!/bin/bash
# usage: run_incompatible.sh <out-dir>   from an intj root; env: PY, CPU, PYTHONPATH, TRITON_HOME,
# and HIP_VISIBLE_DEVICES or CUDA_VISIBLE_DEVICES. Three process rounds of bench_incompatible.py,
# written as r<N>_incompatible.{meta,out,err}.
set -u
OUT=$1; mkdir -p "$OUT"
SCRIPTS=$(cd "$(dirname "$0")" && pwd)
for r in 0 1 2; do
  name=r${r}_incompatible
  { echo "started=$(date -Iseconds)"; echo "cwd=$PWD"; echo "command=taskset -c $CPU $PY $SCRIPTS/bench_incompatible.py"; } > "$OUT/$name.meta"
  taskset -c "$CPU" "$PY" "$SCRIPTS/bench_incompatible.py" > "$OUT/$name.out" 2> "$OUT/$name.err"
  echo "exit_status=$? ended=$(date -Iseconds)" >> "$OUT/$name.meta"
done
date -Iseconds > "$OUT/DONE"

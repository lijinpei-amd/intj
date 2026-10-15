#!/bin/bash
# usage: run_same_binary.sh <out-dir>   from an intj root; env: PY, CPU, PYTHONPATH,
# and HIP_VISIBLE_DEVICES or CUDA_VISIBLE_DEVICES. Three process rounds of
# bench_same_binary.py, written as r<N>_same_binary_1arg.{meta,out,err}.
set -u
OUT=$1; mkdir -p "$OUT"
SCRIPTS=$(cd "$(dirname "$0")" && pwd)
for r in 0 1 2; do
  name=r${r}_same_binary_1arg
  { echo "started=$(date -Iseconds)"; echo "cwd=$PWD"; echo "command=taskset -c $CPU $PY $SCRIPTS/bench_same_binary.py --iters 1000 --batches 9"; } > "$OUT/$name.meta"
  taskset -c "$CPU" "$PY" "$SCRIPTS/bench_same_binary.py" --iters 1000 --batches 9 > "$OUT/$name.out" 2> "$OUT/$name.err"
  echo "exit_status=$? ended=$(date -Iseconds)" >> "$OUT/$name.meta"
done
date -Iseconds > "$OUT/DONE"

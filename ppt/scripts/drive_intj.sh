#!/bin/bash
# usage: drive_intj.sh <out-dir>
# intj-only benchmarks, 3 process rounds, one process at a time, run from the
# intj root ($ROOT). Env: PY (python), CPU (core to pin), HIP_VISIBLE_DEVICES,
# SCRIPTS (this directory), TRITON_HOME.
set -u
OUT=$1; mkdir -p "$OUT"
cd "$ROOT" || exit 1
run() {
  local name=$1; shift
  { echo "started=$(date -Iseconds)"; echo "cwd=$PWD"; echo "command=taskset -c $CPU $*"; } > "$OUT/$name.meta"
  taskset -c "$CPU" "$@" > "$OUT/$name.out" 2> "$OUT/$name.err"
  echo "exit_status=$? ended=$(date -Iseconds)" >> "$OUT/$name.meta"
}
for r in 0 1 2; do
  run r${r}_readme "$PY" "$SCRIPTS/bench_readme.py" --iters 1000 --batches 9
  for s in args constexpr cache; do
    run r${r}_sweep_$s "$PY" "$SCRIPTS/bench_sweeps.py" $s --tag r$r
  done
  run r${r}_ffi_compare "$PY" benchmarks/bench_ffi_compare.py --iters 1000 --batches 9
  run r${r}_ffi_sweep "$PY" benchmarks/bench_ffi_compare.py --sweep --iters 1000 --batches 9
  # same-HSACO comparison: ROCm only (SKIP_HIP=1 on CUDA)
  [ -z "${SKIP_HIP:-}" ] && run r${r}_hip_module "$PY" benchmarks/bench_hip_module_launch.py --iters 1000 --batches 9
done
date -Iseconds > "$OUT/DONE"

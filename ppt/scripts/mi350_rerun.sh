#!/bin/bash
# MI350 second pass: AITER with the 32-call loop, and the FFI sweep through
# the access-mode-tolerant wrapper (RUNTIME_SHIM is unavailable on torch 2.15).
W=$HOME/development/workspace/intj-ppt
OUT=$1; mkdir -p "$OUT"
. $W/env.sh
export CPU=100 GPU=2
bash $W/intj-aiter-gpu-idle.sh > "$OUT/gpu_idle_before_rerun.txt" 2>&1
bash $W/scripts/drive_aiter.sh "$OUT/aiter"
cd $W/src
for r in 0 1 2; do
  f=$OUT/intj/r${r}_ffi_sweep_tolerant
  { echo "started=$(date -Iseconds)"; echo "cwd=$PWD"; echo "command=taskset -c $CPU $PY $W/scripts/bench_ffi_sweep_tolerant.py --sweep --iters 1000 --batches 9"; } > $f.meta
  taskset -c $CPU $PY $W/scripts/bench_ffi_sweep_tolerant.py --sweep --iters 1000 --batches 9 > $f.out 2> $f.err
  echo "exit_status=$? ended=$(date -Iseconds)" >> $f.meta
done
bash $W/intj-aiter-gpu-idle.sh > "$OUT/gpu_idle_after_rerun.txt" 2>&1
date -Iseconds > "$OUT/RERUN_DONE"

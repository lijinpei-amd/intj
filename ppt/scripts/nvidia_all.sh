#!/bin/bash
# The NVIDIA collection (intj benchmarks only; no AITER, no same-HSACO bench).
# usage: nvidia_all.sh <out-dir>   from an intj root (a checkout or a git archive)
# env: PY (python with torch and triton), EXTRA_PYTHONPATH, CPU (core to pin,
#      default 0), CUDA_VISIBLE_DEVICES (default 0); nvcc on PATH for TVM FFI
OUT=$1; mkdir -p "$OUT"
export ROOT=$PWD SCRIPTS=$(cd "$(dirname "$0")" && pwd)
export CPU=${CPU:-0} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} SKIP_HIP=1
# EXTRA_PYTHONPATH: e.g. a --target dir holding apache-tvm-ffi
export PYTHONPATH=$ROOT${EXTRA_PYTHONPATH:+:$EXTRA_PYTHONPATH}
export TRITON_HOME=${TRITON_HOME:-/tmp/intj-ppt-triton-home}
mkdir -p "$TRITON_HOME"
nvidia-smi > "$OUT/gpu_idle_before.txt" 2>&1
bash $SCRIPTS/env_info.sh $CPU 0 > "$OUT/env.txt" 2>&1
echo "intj commit (microbenchmarks): $(cat $ROOT/COMMIT 2>/dev/null || git -C $ROOT rev-parse --short HEAD)" >> "$OUT/env.txt"
bash $SCRIPTS/drive_intj.sh "$OUT/intj"
nvidia-smi > "$OUT/gpu_idle_after.txt" 2>&1
date -Iseconds > "$OUT/ALL_DONE"

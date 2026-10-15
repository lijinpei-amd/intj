#!/bin/bash
# The MI308X collection (intj benchmarks only), from the intj root.
# usage: mi308_all.sh <out-dir>   (GPU 6, CPU 80: both NUMA 1)
OUT=$1; mkdir -p "$OUT"
export ROOT=$(git rev-parse --show-toplevel) SCRIPTS=$(cd "$(dirname "$0")" && pwd)
export PY=/mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/venv/bin/python
export CPU=80 HIP_VISIBLE_DEVICES=6 PYTHONPATH=$ROOT TRITON_HOME=/tmp/intj-ppt-triton-home
mkdir -p $TRITON_HOME
rocm-smi --showuse > "$OUT/gpu_idle_before.txt" 2>&1
bash $SCRIPTS/env_info.sh $CPU 6 > "$OUT/env.txt" 2>&1
echo "intj commit (microbenchmarks): $(git -C $ROOT rev-parse --short HEAD), tracked tree clean: $(git -C $ROOT status --porcelain --untracked-files=no | wc -l) changes" >> "$OUT/env.txt"
bash $SCRIPTS/drive_intj.sh "$OUT/intj"
rocm-smi --showuse > "$OUT/gpu_idle_after.txt" 2>&1
date -Iseconds > "$OUT/ALL_DONE"

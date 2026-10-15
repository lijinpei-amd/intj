#!/bin/bash
# The whole MI350 collection: env info, intj benchmarks, AITER benchmarks.
# usage: setsid nohup mi350_all.sh <out-dir> &   (GPU 2, CPU 100: both NUMA 0)
W=$HOME/development/workspace/intj-ppt
OUT=$1; mkdir -p "$OUT"
. $W/env.sh
export CPU=100 GPU=2 ROOT=$W/src SCRIPTS=$W/scripts
bash $W/intj-aiter-gpu-idle.sh > "$OUT/gpu_idle_before.txt" 2>&1
bash $SCRIPTS/env_info.sh $CPU 2 > "$OUT/env.txt" 2>&1
echo "intj commit (microbenchmarks): $(cat $ROOT/COMMIT)" >> "$OUT/env.txt"
bash $SCRIPTS/drive_intj.sh "$OUT/intj"
bash $SCRIPTS/drive_aiter.sh "$OUT/aiter"
bash $W/intj-aiter-gpu-idle.sh > "$OUT/gpu_idle_after.txt" 2>&1
date -Iseconds > "$OUT/ALL_DONE"

#!/bin/bash
# MI350 third AITER pass: event-timed eager loop; kernel traces kept from the
# second pass (aiter-v2-wallloop/prof_*).
W=$HOME/development/workspace/intj-ppt
OUT=$1
. $W/env.sh
export CPU=100 GPU=2
bash $W/intj-aiter-gpu-idle.sh > "$OUT/gpu_idle_before_v3.txt" 2>&1
SKIP_PROF=1 bash $W/scripts/drive_aiter.sh "$OUT/aiter"
cp -r "$OUT"/aiter-v2-wallloop/prof_* "$OUT/aiter/"
bash $W/intj-aiter-gpu-idle.sh > "$OUT/gpu_idle_after_v3.txt" 2>&1
date -Iseconds > "$OUT/V3_DONE"

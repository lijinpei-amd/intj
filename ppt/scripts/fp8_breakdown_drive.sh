#!/bin/bash
# usage: fp8_breakdown_drive.sh <out-dir>   on the MI350
# The fp8_mqa_logits host-time breakdown: 3 rounds, side order alternating, one process
# per side; then the clean_logits=False / free-cost extras; machine state before and after.
W=$HOME/development/workspace/intj-ppt
OUT=$1; mkdir -p "$OUT"
. $W/env.sh
state() { date -Iseconds; uptime; rocm-smi --showuse | grep "GPU\[2\]"; ps -eo psr,pcpu,comm | awk '$1==100 && $2>1'; }
state > "$OUT/state_before.txt" 2>&1
for r in 0 1 2; do
  if (( r % 2 )); then o="intj base"; else o="base intj"; fi
  for side in $o; do
    bash $W/scripts/fp8_breakdown_run.sh $side --shapes 128,256,512,1024 --calls 4000 > "$OUT/r${r}_$side.jsonl" 2> "$OUT/r${r}_$side.err"
  done
done
for side in base intj; do
  SCRIPT=$W/scripts/bench_fp8_mqa_breakdown_extra.py bash $W/scripts/fp8_breakdown_run.sh $side > "$OUT/extra_$side.jsonl"
done
state > "$OUT/state_after.txt" 2>&1

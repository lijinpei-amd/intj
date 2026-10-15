#!/bin/bash
# usage: drive_aiter.sh <out-dir>   (on the MI350; env as run_aiter.sh: GPU, CPU)
# 3 rounds of every (op, route) on both sides, side order alternating per
# round, then one rocprofv3 kernel-trace run per (op, route, side).
set -u
OUT=$1; mkdir -p "$OUT"
S=$(cd "$(dirname "$0")" && pwd)
SPECS="kda_decode:default flash_kda:default flash_kda:triton fp8_mqa_logits:default"
args_for() {  # op route side -> env and extra args
  local env= extra=
  [ "$2" = triton ] && env=AITER_FDA_USE_GLUON=0
  [ "$3" = base ] && [ "$1" = flash_kda ] && extra=--bypass
  echo "$env|$extra"
}
run() {
  local name=$1 env=$2; shift 2
  { echo "started=$(date -Iseconds)"; echo "command=${env:+env $env }$*"; } > "$OUT/$name.meta"
  env $env "$@" > "$OUT/$name.jsonl" 2> "$OUT/$name.err"
  echo "exit_status=$? ended=$(date -Iseconds)" >> "$OUT/$name.meta"
}
for r in 0 1 2; do
  if (( r % 2 )); then order="intj base"; else order="base intj"; fi
  for spec in $SPECS; do
    op=${spec%%:*}; route=${spec##*:}
    for side in $order; do
      IFS='|' read -r env extra <<< "$(args_for $op $route $side)"
      run r${r}_${op}_${route}_${side} "$env" "$S/run_aiter.sh" $side $op --tag r$r $extra
    done
  done
done
# SKIP_PROF=1: timings only (the kernel traces do not depend on them)
[ -n "${SKIP_PROF:-}" ] && SPECS=
for spec in $SPECS; do
  op=${spec%%:*}; route=${spec##*:}
  for side in base intj; do
    IFS='|' read -r env _ <<< "$(args_for $op $route $side)"
    name=prof_${op}_${route}_${side}
    rm -rf "$OUT/$name"
    PROF=$OUT/$name run $name "$env" "$S/run_aiter.sh" $side $op --profile 50
    python3 "$S/summarize_rocprof.py" "$OUT/$name/trace_kernel_trace.csv" "$OUT/$name.jsonl" \
      > "$OUT/$name.summary.jsonl" 2>> "$OUT/$name.err"
  done
done
date -Iseconds > "$OUT/DONE"

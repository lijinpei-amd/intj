#!/bin/bash
# usage: fp8_breakdown_run.sh base|intj [script args]   (mirrors run_aiter.sh)
# One process on the MI350: the side's AITER tree (+ intj 20e578f for intj), GPU $GPU,
# pinned to CPU $CPU; runs $SCRIPT (default bench_fp8_mqa_breakdown.py). JSON lines on stdout.
W=$HOME/development/workspace/intj-ppt
TREES=$HOME/development/workspace/intj-aiter/trees
side=$1; shift
case $side in
  base) T=$TREES/aiter-base-874d253dd3; EXTRA= ;;
  intj) T=$TREES/aiter-exactly-5d710e670; EXTRA=:$TREES/intj-20e578f ;;
  *) echo "side must be base|intj" >&2; exit 2 ;;
esac
. $W/env.sh
export AITER_TRITON_ONLY=1 AITER_USE_SYSTEM_TRITON=1
export HIP_VISIBLE_DEVICES=${GPU:-2}
export TRITON_HOME=$W/triton_home_aiter_$side
export PYTHONPATH="$T:$T/aiter/jit/utils$EXTRA"
mkdir -p $TRITON_HOME
cd $W
V=$HOME/development/venv/01/bin
exec taskset -c ${CPU:-100} $V/python ${SCRIPT:-$W/scripts/bench_fp8_mqa_breakdown.py} "$@"

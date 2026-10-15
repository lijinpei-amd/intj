#!/bin/bash
# usage: run_aiter.sh base|intj <op> [bench_aiter_ops.py args...]
# One process on the MI350: the AITER tree of that side (+ intj 20e578f for
# intj), GPU $GPU, main thread pinned to CPU $CPU. JSON lines on stdout.
W=$HOME/development/workspace/intj-ppt
TREES=$HOME/development/workspace/intj-aiter/trees
side=$1; op=$2; shift 2
case $side in
  base) T=$TREES/aiter-base-874d253dd3; EXTRA= ;;
  intj) T=$TREES/aiter-exactly-5d710e670; EXTRA=:$TREES/intj-20e578f ;;
  *) echo "side must be base|intj" >&2; exit 2 ;;
esac
export AITER_TRITON_ONLY=1 AITER_USE_SYSTEM_TRITON=1
export HIP_VISIBLE_DEVICES=${GPU:-2}
export TRITON_HOME=$W/triton_home_aiter_$side
export PYTHONPATH="$T:$T/aiter/jit/utils$EXTRA"
mkdir -p $TRITON_HOME
cd $W
V=$HOME/development/venv/01/bin
# PROF=<dir>: run under rocprofv3 --kernel-trace, CSV into <dir>
exec taskset -c ${CPU:-100} ${PROF:+$V/rocprofv3 --kernel-trace --output-format csv -d $PROF -o trace --} \
  $V/python $W/scripts/bench_aiter_ops.py --op $op --side $side "$@"

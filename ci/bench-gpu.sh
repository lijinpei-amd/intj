# GPU launch timings, INTJ matrix and Triton vs INTJ. Informational: no thresholds.
. "$(dirname "$0")/env.sh"
mkdir -p "$OUT/bench"
taskset -cp "${BENCH_CPU:-0}" $$ >/dev/null
py "$ROOT/benchmarks/bench_launch.py" | tee "$OUT/bench/bench-gpu.txt"
py "$ROOT/benchmarks/bench_launch.py" --readme | tee "$OUT/bench/bench-readme.txt"

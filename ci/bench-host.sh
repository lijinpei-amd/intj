# Host-only launch timings (decode + cache, no GPU). Informational: no thresholds.
. "$(dirname "$0")/env.sh"
mkdir -p "$OUT/bench"
taskset -cp "${BENCH_CPU:-0}" $$ >/dev/null
py "$ROOT/benchmarks/bench_launch.py" --no-gpu | tee "$OUT/bench/bench-host.txt"

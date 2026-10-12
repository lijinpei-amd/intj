# pyright over intj/, tests/ and benchmarks/, resolving imports from $PYTHON.
. "$(dirname "$0")/env.sh"
cd "$ROOT"
pyright --pythonpath "$(command -v "$PYTHON")"

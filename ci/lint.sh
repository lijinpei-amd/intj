# The pre-commit hooks (ruff-format, clang-format) over every file.
. "$(dirname "$0")/env.sh"
cd "$ROOT"
pre-commit run --all-files --show-diff-on-failure

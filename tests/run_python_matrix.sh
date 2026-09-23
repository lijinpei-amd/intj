#!/usr/bin/env bash
# The triton-free tests (tests/test_runtime.py) on every CPython intj supports,
# each in a throwaway uv environment with the newest CPU torch it can install.
#
#   tests/run_python_matrix.sh            # all of them
#   tests/run_python_matrix.sh 3.8 3.13t  # some
cd "$(dirname "$0")/.." || exit 1
versions=("$@")
[ ${#versions[@]} -eq 0 ] && versions=(3.8 3.9 3.10 3.11 3.12 3.13 3.13t 3.14 3.14t)
failed=()
for py in "${versions[@]}"; do
  echo "== python $py"
  PYTHONPATH=$PWD uv run -q --no-project --python "cpython-$py" \
    --index https://download.pytorch.org/whl/cpu \
    --with torch --with pytest --with jinja2 --with "tomli; python_version < '3.11'" \
    python -m pytest -q -p no:cacheprovider tests/test_runtime.py || failed+=("$py")
done
[ ${#failed[@]} -eq 0 ] || { echo "failed: ${failed[*]}"; exit 1; }

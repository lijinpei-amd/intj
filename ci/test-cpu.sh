# The tests that need neither triton nor a GPU.
. "$(dirname "$0")/env.sh"
py -m pytest -q -p no:cacheprovider "$ROOT/tests/test_runtime.py"

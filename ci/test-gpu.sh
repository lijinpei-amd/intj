# The full suite; needs triton and a GPU.
. "$(dirname "$0")/env.sh"
py -m pytest -q -p no:cacheprovider "$ROOT/tests"

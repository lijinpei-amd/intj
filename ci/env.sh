# Sourced by every ci/ script: resolves the install mode, the interpreter and
# the paths, and runs python against the intj that mode selects.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
OUT=$ROOT/ci/out
INTJ_MODE=${INTJ_MODE:-editable}
PYTHON=${PYTHON:-python3}

die() {
  echo "ci: $*" >&2
  exit 1
}

case $INTJ_MODE in
  editable | wheel) ;;
  *) die "INTJ_MODE must be editable or wheel, got '$INTJ_MODE'" ;;
esac
command -v "$PYTHON" >/dev/null || die "PYTHON '$PYTHON' not found"

# pip when the interpreter has it, else uv: uv-created venvs ship without pip.
pip_() {
  if "$PYTHON" -m pip --version >/dev/null 2>&1; then
    "$PYTHON" -m pip "$@"
  else
    uv pip "$1" --python "$PYTHON" "${@:2}"
  fi
}

# Fails unless `import intj` resolves to the interpreter's site-packages.
check_installed() {
  mkdir -p "$OUT/run"
  (cd "$OUT/run" && "$PYTHON" -I -c '
import os, sys, sysconfig
try:
    import intj
except ImportError:
    sys.exit("ci: intj is not installed; run INTJ_MODE=wheel ninja -C ci wheel, then install")
path = os.path.realpath(intj.__file__)
lib = os.path.realpath(sysconfig.get_paths()["purelib"])
if not path.startswith(lib + os.sep):
    sys.exit("ci: intj imports from %s, not from %s" % (path, lib))
print("ci: intj from %s" % path)
')
}

# Runs "$PYTHON" "$@": editable -> the checkout; wheel -> the installed package,
# isolated (-I) from the cwd, the script's directory and PYTHONPATH.
py() {
  if [ "$INTJ_MODE" = editable ]; then
    (cd "$ROOT" && PYTHONPATH=$ROOT "$PYTHON" "$@")
  else
    check_installed
    (cd "$OUT/run" && "$PYTHON" -I "$@")
  fi
}

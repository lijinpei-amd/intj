# Builds the intj wheel into ci/out/dist, replacing any earlier one.
. "$(dirname "$0")/env.sh"
rm -rf "$OUT/dist"
if "$PYTHON" -m pip --version >/dev/null 2>&1; then
  "$PYTHON" -m pip wheel --no-deps -w "$OUT/dist" "$ROOT"
else
  uv build --wheel --python "$PYTHON" --out-dir "$OUT/dist" "$ROOT"
fi
ls "$OUT"/dist/intj-*.whl

# editable: pip install -e the checkout.  wheel: install the one wheel in
# ci/out/dist, replacing an installed intj even at the same version.
# TORCH=<version> first replaces torch with that version from
# $INTJ_TORCH_INDEX (set in the CI images; default the CPU index).
. "$(dirname "$0")/env.sh"
if [ -n "${TORCH:-}" ]; then
  pip_ install --index-url "${INTJ_TORCH_INDEX:-https://download.pytorch.org/whl/cpu}" "torch==$TORCH"
fi
if [ "$INTJ_MODE" = editable ]; then
  pip_ install -e "$ROOT"
  exit
fi
shopt -s nullglob
wheels=("$OUT"/dist/intj-*.whl)
[ ${#wheels[@]} -eq 1 ] || die "expected one wheel in ci/out/dist, found ${#wheels[@]}; run ninja -C ci wheel"
pip_ install --force-reinstall --no-deps "${wheels[0]}"
pip_ install "${wheels[0]}"

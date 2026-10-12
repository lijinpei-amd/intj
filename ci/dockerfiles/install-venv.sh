#!/bin/sh
# Builds /opt/venv from $PY_BIN with everything the ci/ targets need but intj.
set -eux
"$PY_BIN" -m venv /opt/venv
pip=/opt/venv/bin/pip
/opt/venv/bin/python -m pip install -U pip
"$pip" install --index-url "$TORCH_INDEX" torch
if [ -n "$TRITON" ]; then
  # torch's own triton build (pytorch-triton-rocm) is older than intj needs.
  "$pip" uninstall -y triton pytorch-triton-rocm || true
  "$pip" install "triton==$TRITON"
fi
"$pip" install jinja2 "tomli; python_version < '3.11'" pytest ninja pre-commit "pyright[nodejs]"

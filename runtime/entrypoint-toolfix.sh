#!/usr/bin/env bash
set -Eeuo pipefail

PYTHON=/opt/runtime-venv/bin/python
PIP=/opt/runtime-venv/bin/pip
WHEELS=/runtime-wheels
XGRAMMAR=xgrammar-0.2.4-cp312-cp312-manylinux_2_26_aarch64.manylinux_2_28_aarch64.whl
TRANSFORMERS=transformers-5.13.1-py3-none-any.whl

printf '%s  %s\n' \
  'b1d2bef13aec8122dfa6b96e3ece2cc0a42a732685f1d44f6248c305274e8eae' \
  "${WHEELS}/${XGRAMMAR}" \
  '53f0ea8aa397e29244c2377ba981bcaf0c87adcf44fbdd447ef6306522afcacd' \
  "${WHEELS}/${TRANSFORMERS}" | sha256sum -c -

"${PIP}" install -q --no-index --no-deps \
  "${WHEELS}/${XGRAMMAR}" \
  "${WHEELS}/${TRANSFORMERS}"

"${PYTHON}" - <<'PY'
from importlib.metadata import version
expected = {"xgrammar": "0.2.4", "transformers": "5.13.1"}
actual = {name: version(name) for name in expected}
if actual != expected:
    raise SystemExit(f"offline toolfix version mismatch: {actual} != {expected}")
print(f"[toolfix] offline pinned packages verified: {actual}")
PY

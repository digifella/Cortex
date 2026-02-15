#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${1:-$ROOT_DIR/.venv-docling}"

echo "[docling-setup] root: $ROOT_DIR"
echo "[docling-setup] venv: $VENV_DIR"

python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r "$ROOT_DIR/requirements-docling.txt"

python - <<'PY'
from importlib.metadata import version, PackageNotFoundError

pkgs = ["docling", "docling-core", "numpy"]
for pkg in pkgs:
    try:
        print(f"[docling-setup] {pkg}={version(pkg)}")
    except PackageNotFoundError:
        print(f"[docling-setup] {pkg}=not-installed")
PY

echo "[docling-setup] complete"
echo "[docling-setup] activate with: source $VENV_DIR/bin/activate"

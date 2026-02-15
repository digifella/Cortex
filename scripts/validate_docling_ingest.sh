#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DB_PATH="${1:-/tmp/cortex_docling_smoke_db}"
SAMPLE_PATH="${2:-}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

normalize_input_path() {
  local raw="${1:-}"
  if [[ -z "$raw" ]]; then
    echo ""
    return
  fi
  if [[ -e "$raw" ]]; then
    echo "$raw"
    return
  fi
  if command -v wslpath >/dev/null 2>&1; then
    local converted
    converted="$(wslpath "$raw" 2>/dev/null || true)"
    if [[ -n "$converted" && -e "$converted" ]]; then
      echo "$converted"
      return
    fi
  fi
  echo "$raw"
}

if [[ $# -eq 1 ]]; then
  candidate="$(normalize_input_path "$1")"
  if [[ -f "$candidate" ]]; then
    DB_PATH="/tmp/cortex_docling_smoke_db"
    SAMPLE_PATH="$candidate"
  elif [[ "$1" =~ \.(pdf|docx|pptx|txt|md)$ ]]; then
    DB_PATH="/tmp/cortex_docling_smoke_db"
    SAMPLE_PATH="$candidate"
  fi
fi

SAMPLE_PATH="$(normalize_input_path "$SAMPLE_PATH")"

echo "[docling-validate] root: $ROOT_DIR"
echo "[docling-validate] python: $PYTHON_BIN"
echo "[docling-validate] db_path: $DB_PATH"

if [[ -n "$SAMPLE_PATH" && ! -e "$SAMPLE_PATH" ]]; then
  echo "[docling-validate] sample file does not exist: $SAMPLE_PATH" >&2
  echo "[docling-validate] tip: pass a real local file path, e.g. one of:" >&2
  find "$ROOT_DIR" -maxdepth 1 -type f \( -iname "*.pdf" -o -iname "*.docx" -o -iname "*.pptx" \) | head -n 5 >&2 || true
  exit 1
fi

mkdir -p "$DB_PATH"

"$PYTHON_BIN" - <<'PY'
import importlib
import sys

importlib.import_module("docling")
print("[docling-validate] import ok: docling")

try:
    importlib.import_module("llama_index.core")
except Exception as e:
    print(f"[docling-validate] warning: full Cortex ingest deps missing ({e})")
    print("[docling-validate] hint: run validator with your main Cortex venv python")
    sys.exit(2)

mods = ["cortex_engine.docling_reader", "cortex_engine.migration_to_docling"]
for m in mods:
    importlib.import_module(m)
    print(f"[docling-validate] import ok: {m}")
PY

if [[ -n "$SAMPLE_PATH" ]]; then
  echo "[docling-validate] running single-file analyze pass with --ingest-backend docling"
  "$PYTHON_BIN" "$ROOT_DIR/cortex_engine/ingest_cortex.py" \
    --analyze-only \
    --db-path "$DB_PATH" \
    --ingest-backend docling \
    --include "$SAMPLE_PATH"
else
  echo "[docling-validate] no sample file supplied; import checks complete"
fi

echo "[docling-validate] complete"

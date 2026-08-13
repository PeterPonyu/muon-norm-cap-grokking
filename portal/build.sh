#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${1:-"$ROOT/_site"}"
python3 "$ROOT/tools/warehouse_ci.py" build "$DEST"

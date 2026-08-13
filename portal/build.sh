#!/usr/bin/env bash
# Validate FIGURE-INDEX then copy portal/ → _site/. No LaTeX.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

python3 - <<'PY'
import json
from pathlib import Path

import jsonschema

root = Path(".")
schema = json.loads((root / "papers/FIGURE-INDEX.schema.json").read_text(encoding="utf-8"))
index = json.loads((root / "papers/FIGURE-INDEX.json").read_text(encoding="utf-8"))
jsonschema.validate(instance=index, schema=schema)
pdfs = sorted(p for p in (root / "papers").rglob("*.pdf") if p.is_file())
if pdfs:
    raise SystemExit(f"refuse: PDFs under papers/ (do not copy venue flats): {pdfs}")
print("INDEX valid; no papers/**/*.pdf")
PY

rm -rf _site
mkdir -p _site/data/figs
cp -a portal/. _site/
cp papers/FIGURE-INDEX.json _site/data/figures.json
if [[ -d papers/figs/summaries ]]; then
  cp -a papers/figs/summaries _site/data/figs/summaries
fi
if [[ -d papers/figs/previews ]]; then
  cp -a papers/figs/previews _site/data/figs/previews
fi
if [[ -e _site/experiments || -e _site/.omc ]]; then
  echo "I4: experiments or .omc leaked into _site" >&2
  exit 1
fi
echo "built _site/ from portal/ + FIGURE-INDEX"

#!/usr/bin/env bash
# Validate FIGURE-INDEX, Next.js static export → _site/. No LaTeX.
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

cd portal
if [[ -f package-lock.json ]]; then
  npm ci
else
  npm install
fi
npm run build
test -f out/index.html
test -d out/dose
cd "$ROOT"

rm -rf _site
mkdir -p _site/data
cp -a portal/out/. _site/
python3 - <<'PY'
import json
from pathlib import Path
index = json.loads(Path("papers/FIGURE-INDEX.json").read_text(encoding="utf-8"))
stripped = {
    "paper_id": index["paper_id"],
    "github": index["github"],
    "zenodo_concept_doi": index["zenodo_concept_doi"],
    "pipeline": index["pipeline"],
    "figures": [
        {
            "id": fig["id"],
            "generator": fig.get("generator"),
            "summary": fig.get("summary"),
            "preview_svg": fig.get("preview_svg"),
            "tex_build": fig.get("tex_build"),
            "vec_build": fig.get("vec_build"),
        }
        for fig in index["figures"]
    ],
}
Path("_site/data").mkdir(parents=True, exist_ok=True)
Path("_site/data/figures.json").write_text(json.dumps(stripped, indent=2) + "\n", encoding="utf-8")
print("wrote stripped _site/data/figures.json")
PY
# F9: keep the figs/ tree resolvable under _site/data/ (summaries + previews).
mkdir -p _site/data/figs
cp -a papers/figs/summaries _site/data/figs/
if compgen -G "papers/figs/previews/*.svg" > /dev/null; then
  cp -a papers/figs/previews _site/data/figs/
fi
python3 - <<'PY'
import json
from pathlib import Path
index = json.loads(Path("papers/FIGURE-INDEX.json").read_text(encoding="utf-8"))
missing = []
for fig in index["figures"]:
    for key in ("summary", "preview_svg"):
        rel = fig.get(key)
        if rel and not (Path("_site/data") / rel).is_file():
            missing.append(f"{fig['id']}.{key}: {rel}")
if missing:
    raise SystemExit(f"F9: _site/data paths do not resolve: {missing}")
print("F9: every non-null summary/preview resolves under _site/data/")
PY
if [[ -e _site/experiments || -e _site/.omc ]]; then
  echo "I4: experiments or .omc leaked into _site" >&2
  exit 1
fi
echo "exported Next.js out/ → _site/"

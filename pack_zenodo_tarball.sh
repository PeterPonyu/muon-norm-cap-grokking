#!/usr/bin/env bash
# Scientific tarball via git archive (honors .gitattributes export-ignore).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
OUT="${1:-muon-norm-cap-grokking-src.tar.gz}"
git archive --format=tar.gz --prefix=muon-norm-cap-grokking/ -o "$OUT" HEAD
python3 - "$OUT" <<'PY'
import sys
import tarfile

path = sys.argv[1]
forbidden = ("portal/", "_site/", ".github/")
with tarfile.open(path, "r:gz") as tar:
    names = tar.getnames()
hits = [n for n in names if any(part in n.split("/") or n.endswith(part.rstrip("/")) or f"/{part}" in n + "/" or n.startswith(part) for part in ("portal", "_site", ".github"))]
# stricter: any member whose path contains those trees
hits = []
for n in names:
    parts = n.split("/")
    if "portal" in parts or "_site" in parts or ".github" in parts:
        hits.append(n)
if hits:
    raise SystemExit(f"verify_tarball failed: website/Actions members present: {hits[:20]}")
print(f"verify_tarball ok: {path} members={len(names)}")
PY

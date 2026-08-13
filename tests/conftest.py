"""Warehouse-root fixtures for Paper A contract tests."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
INDEX_PATH = REPO_ROOT / "papers" / "FIGURE-INDEX.json"
SCHEMA_PATH = REPO_ROOT / "papers" / "FIGURE-INDEX.schema.json"
PORTAL_DIR = REPO_ROOT / "portal"
PORTAL_INDEX = PORTAL_DIR / "index.html"
WORKFLOWS_DIR = REPO_ROOT / ".github" / "workflows"
CANONICAL_TEX = REPO_ROOT / "papers" / "A" / "main.tex"

CONCEPT_DOI = "10.5281/zenodo.21020291"
VERSION_DOI = "10.5281/zenodo.21020292"
GITHUB_REPO = "PeterPonyu/muon-norm-cap-grokking"
GITHUB_URL = f"https://github.com/{GITHUB_REPO}"
NAV_LABELS = ("Cap", "Dose", "Floor", "LMC", "Boundary", "Reproduce")
ROUTE_FILES = {
    "cap": "index.html",
    "dose": "dose.html",
    "floor": "floor.html",
    "lmc": "lmc.html",
    "boundary": "boundary.html",
    "reproduce": "reproduce.html",
}


def workflow_path(name: str) -> Path:
    return WORKFLOWS_DIR / name

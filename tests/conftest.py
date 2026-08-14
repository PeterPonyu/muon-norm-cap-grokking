"""Warehouse-root fixtures for Paper A contract tests."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
INDEX_PATH = REPO_ROOT / "papers" / "FIGURE-INDEX.json"
SCHEMA_PATH = REPO_ROOT / "papers" / "FIGURE-INDEX.schema.json"
PORTAL_DIR = REPO_ROOT / "portal"
WORKFLOWS_DIR = REPO_ROOT / ".github" / "workflows"
CANONICAL_TEX = REPO_ROOT / "papers" / "A" / "main.tex"

CONCEPT_DOI = "10.5281/zenodo.21020291"
VERSION_DOI = "10.5281/zenodo.21020292"
GITHUB_REPO = "PeterPonyu/muon-norm-cap-grokking"
GITHUB_URL = f"https://github.com/{GITHUB_REPO}"
BASE_PATH = "/muon-norm-cap-grokking"
NAV_LABELS = ("Cap", "Dose", "Floor", "LMC", "Boundary", "Reproduce")
APP_ROUTES = (
    PORTAL_DIR / "app" / "page.tsx",
    PORTAL_DIR / "app" / "dose" / "page.tsx",
    PORTAL_DIR / "app" / "floor" / "page.tsx",
    PORTAL_DIR / "app" / "lmc" / "page.tsx",
    PORTAL_DIR / "app" / "boundary" / "page.tsx",
    PORTAL_DIR / "app" / "reproduce" / "page.tsx",
)
EXPORT_ROUTES = (
    "index.html",
    "dose/index.html",
    "floor/index.html",
    "lmc/index.html",
    "boundary/index.html",
    "reproduce/index.html",
)
FORBIDDEN_UI = (
    "paper",
    "journal",
    "document",
    "manuscript",
    "submission",
    "preprint",
    "publication",
    "venue",
    "peerj",
    "neurocomputing",
    "jmlr",
    "main.tex",
    "figure-index",
    "pipeline.md",
    "warehouse",
)
FILENAME_LEAKS = (
    ".json",
    ".tex",
    ".md",
    ".py",
    ".r",
    "figure-index",
    "pipeline",
    "main.tex",
    "a_gap",
    "a_normctl",
    "a_floor",
    "a_lmc",
    "a_sink",
    "a_plasticity",
    "a_synth",
    "papers/a",
    "warehouse",
)
SCIENCE_TOKENS = (
    "Cap",
    "Dose",
    "Floor",
    "LMC",
    "Boundary",
    "dose response",
    "hidden-norm",
)
LEAK_PATTERNS = (
    "3525",
    "10.8",
    "10.846",
    "7.13",
    "19.5",
    "8/8",
    "0/5",
    "311",
    "12.5",
    "lock-by-8000",
    "S5 GROK",
    "BCa 95%",
    "abstract",
)


def workflow_path(name: str) -> Path:
    return WORKFLOWS_DIR / name

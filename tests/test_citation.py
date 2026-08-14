"""Citation / license contract (C1–C4, G4, I6)."""

from __future__ import annotations

import json

import yaml

from conftest import CONCEPT_DOI, GITHUB_URL, REPO_ROOT, VERSION_DOI


def test_citation_cff_parses() -> None:
    raw = (REPO_ROOT / "CITATION.cff").read_text(encoding="utf-8")
    data = yaml.safe_load(raw)
    assert isinstance(data, dict)
    assert isinstance(data.get("title"), str) and data["title"].strip()


def test_citation_cff_uses_concept_doi() -> None:
    data = yaml.safe_load((REPO_ROOT / "CITATION.cff").read_text(encoding="utf-8"))
    assert data["doi"] == CONCEPT_DOI
    identifiers = data.get("identifiers") or []
    values = {item.get("value") for item in identifiers if isinstance(item, dict)}
    assert VERSION_DOI in values
    assert data["doi"] != VERSION_DOI


def test_citation_cff_is_not_five_paper_bundle() -> None:
    data = yaml.safe_load((REPO_ROOT / "CITATION.cff").read_text(encoding="utf-8"))
    blob = json.dumps(data).lower()
    assert "grokking-clock" not in blob
    assert "architecture-staircase" not in blob
    assert "free-repetition-band" not in blob
    assert "calibration-traps" not in blob


def test_dual_license_notice_preserved() -> None:
    license_text = (REPO_ROOT / "LICENSE").read_text(encoding="utf-8")
    assert "MIT License" in license_text
    assert "Creative Commons Attribution 4.0" in license_text or "CC BY 4.0" in license_text


def test_readme_cites_github() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert GITHUB_URL in readme or "PeterPonyu/muon-norm-cap-grokking" in readme


def test_readme_leads_with_live_door_and_strips_publication_leaks() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    head = "\n".join(readme.strip().splitlines()[:6])
    assert "https://peterponyu.github.io/muon-norm-cap-grokking/" in head
    lower = readme.lower()
    for tok in (
        "warehouse",
        "preprint",
        "journal",
        "manuscript",
        "venue",
        "main.tex",
        "figure-index",
        "pipeline.md",
        "github estimate",
        "stars",
    ):
        assert tok not in lower, f"README still leaks {tok}"
    assert "star" not in lower

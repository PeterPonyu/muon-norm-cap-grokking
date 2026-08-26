"""Pointer contract: public papers/A/main.tex is a short archive pointer."""

from __future__ import annotations

from conftest import CANONICAL_TEX, CONCEPT_DOI, GITHUB_URL, PAPER_TITLE, REPO_ROOT

MAX_LINES = 40


def test_pointer_tex_exists() -> None:
    assert CANONICAL_TEX.is_file()


def test_pointer_tex_is_short() -> None:
    lines = CANONICAL_TEX.read_text(encoding="utf-8").splitlines()
    assert 1 <= len(lines) <= MAX_LINES


def test_pointer_tex_cites_title_doi_and_github() -> None:
    text = CANONICAL_TEX.read_text(encoding="utf-8")
    assert PAPER_TITLE in text
    assert CONCEPT_DOI in text
    assert GITHUB_URL in text


def test_pointer_tex_states_reproduction_archive() -> None:
    text = CANONICAL_TEX.read_text(encoding="utf-8").lower()
    assert "reproduction archive" in text
    assert "submitted separately" in text


def test_pointer_tex_has_no_journal_or_internal_leaks() -> None:
    lower = CANONICAL_TEX.read_text(encoding="utf-8").lower()
    for tok in (
        "bundle a",
        "paper a",
        "paper c",
        "ieee",
        "peerj",
        "elsevier",
        "claude",
    ):
        assert tok not in lower, f"pointer leaks {tok}"


def test_figpreamble_file_is_unmodified_pointer() -> None:
    preamble = (REPO_ROOT / "papers" / "figs" / "figpreamble.tex").read_text(encoding="utf-8")
    assert r"\graphicspath{{../figs/vec/}{../figs/}}" in preamble
    assert r"\figtikz" in preamble

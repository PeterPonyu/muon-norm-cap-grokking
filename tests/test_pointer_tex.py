"""Pointer manuscript contract (F5, F6, F6b)."""

from __future__ import annotations

from conftest import CANONICAL_TEX, REPO_ROOT

MIN_LINES = 2100  # canonical papers/A/main.tex is ~2295; 20-line stubs fail


def test_warehouse_tex_is_full_canonical_not_a_list() -> None:
    assert CANONICAL_TEX.is_file()
    text = CANONICAL_TEX.read_text(encoding="utf-8")
    lines = text.splitlines()
    assert len(lines) >= MIN_LINES, (
        f"F6b: papers/A/main.tex has {len(lines)} lines; expected full manuscript"
    )
    assert "pointer-only" not in text.lower()


def test_figpreamble_include_survives() -> None:
    text = CANONICAL_TEX.read_text(encoding="utf-8")
    assert r"\input{../figs/figpreamble.tex}" in text
    assert r"\graphicspath{{./}}" not in text
    assert "figpreamble severed" not in text
    assert "Figure1.pdf" not in text
    assert "Figure3.pdf" not in text


def test_preamble_routed_heatmap_includes_may_remain() -> None:
    text = CANONICAL_TEX.read_text(encoding="utf-8")
    assert "A_normctl.pdf" in text


def test_tex_does_not_include_previews() -> None:
    text = CANONICAL_TEX.read_text(encoding="utf-8")
    assert "previews/" not in text


def test_figpreamble_file_is_unmodified_pointer() -> None:
    preamble = (REPO_ROOT / "papers" / "figs" / "figpreamble.tex").read_text(encoding="utf-8")
    assert r"\graphicspath{{../figs/vec/}{../figs/}}" in preamble
    assert r"\figtikz" in preamble

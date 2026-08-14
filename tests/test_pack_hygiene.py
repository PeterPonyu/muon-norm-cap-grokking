"""Zenodo / git-archive hygiene: omit portal/, _site/, .github/."""

from __future__ import annotations

from conftest import REPO_ROOT


def test_gitattributes_export_ignore_website_trees() -> None:
    text = (REPO_ROOT / ".gitattributes").read_text(encoding="utf-8")
    assert "portal/ export-ignore" in text
    assert "_site/ export-ignore" in text
    assert ".github/ export-ignore" in text


def test_pack_script_omits_website_trees() -> None:
    script = (REPO_ROOT / "pack_zenodo_tarball.sh").read_text(encoding="utf-8")
    assert "git archive" in script
    assert "portal" in script
    assert "_site" in script
    assert ".github" in script


def test_gitignore_excludes_site_output() -> None:
    gitignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "_site/" in gitignore

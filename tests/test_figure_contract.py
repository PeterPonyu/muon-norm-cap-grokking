"""Figure-pointer contract (F1–F9) for muon-norm-cap-grokking."""

from __future__ import annotations

import json
import subprocess

import jsonschema

from conftest import INDEX_PATH, REPO_ROOT, SCHEMA_PATH


def _index() -> dict:
    assert INDEX_PATH.is_file(), f"F1: missing {INDEX_PATH}"
    return json.loads(INDEX_PATH.read_text(encoding="utf-8"))


def test_figure_index_and_shared_schema_exist() -> None:
    assert INDEX_PATH.is_file()
    assert SCHEMA_PATH.is_file()
    assert SCHEMA_PATH.name == "FIGURE-INDEX.schema.json"
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    paper_id = schema["properties"]["paper_id"]
    assert paper_id.get("const") != "A"
    assert "A" in (paper_id.get("enum") or [])


def test_figure_index_validates_against_schema() -> None:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    jsonschema.validate(instance=_index(), schema=schema)


def test_index_identifies_paper_a_warehouse() -> None:
    data = _index()
    assert data["paper_id"] == "A"
    assert data["github"] == "PeterPonyu/muon-norm-cap-grokking"
    assert data["zenodo_concept_doi"] == "10.5281/zenodo.21020291"
    assert data["pipeline"] == "figs/PIPELINE.md"


def test_index_paths_are_papers_relative() -> None:
    data = _index()
    for fig in data["figures"]:
        for key in ("generator", "summary", "preview_svg", "tex_build", "vec_build"):
            value = fig.get(key)
            if value is None:
                continue
            assert value.startswith("figs/"), f"F9: {fig['id']}.{key} must start with figs/: {value}"
            assert not value.startswith("summaries/"), f"F9: mixed base {value}"


def test_index_ids_have_generators_and_summaries() -> None:
    data = _index()
    assert data["figures"], "INDEX must list figures"
    for fig in data["figures"]:
        gen = fig.get("generator")
        assert gen, f"F2: {fig['id']} needs generator"
        assert (REPO_ROOT / "papers" / gen).is_file(), f"F2: missing generator {gen}"
        summary = fig.get("summary")
        if summary:
            assert (REPO_ROOT / "papers" / summary).is_file(), f"F7: missing {summary}"
        preview = fig.get("preview_svg")
        if preview:
            assert (REPO_ROOT / "papers" / preview).is_file(), f"F9: missing preview {preview}"


def test_no_pdfs_committed_under_papers() -> None:
    proc = subprocess.run(
        ["git", "ls-files", "papers/**/*.pdf"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    tracked = [line for line in proc.stdout.splitlines() if line.strip()]
    assert not tracked, f"F4: committed PDFs forbidden: {tracked}"


def test_gitignore_excludes_compiled_figure_tiers() -> None:
    gitignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "papers/figs/tex/" in gitignore or "figs/tex/" in gitignore
    assert "papers/figs/vec/" in gitignore or "figs/vec/" in gitignore

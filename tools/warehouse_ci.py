"""Warehouse CI checks for the Paper A figure-pointer and portal contract."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import jsonschema

REPO_ROOT = Path(__file__).resolve().parents[1]


class ContractError(ValueError):
    """Raised when the warehouse figure or portal contract is violated."""


def validate_index(index: dict[str, Any], schema: dict[str, Any]) -> None:
    try:
        jsonschema.validate(instance=index, schema=schema)
    except jsonschema.ValidationError as exc:
        raise ContractError(str(exc.message)) from exc


def find_committed_pdfs(papers_dir: Path) -> list[Path]:
    if not papers_dir.exists():
        return []
    return sorted(path for path in papers_dir.rglob("*.pdf") if path.is_file())


def tex_contract_violations(tex: str) -> list[str]:
    violations: list[str] = []
    if "previews/" in tex:
        violations.append("previews/ include is forbidden")
    if re.search(r"Figure\d+\.pdf", tex):
        violations.append("PeerJ FigureN.pdf include is forbidden")
    if re.search(r"\\includegraphics(?:\[[^\]]*\])?\{(?:\./)?A_[^}]+\.pdf\}", tex):
        violations.append("same-directory A_*.pdf include is forbidden")
    if r"\figtikz" not in tex:
        violations.append("missing \\figtikz")
    if r"\input{../figs/figpreamble.tex}" not in tex:
        violations.append("missing figpreamble input")
    return violations


def absolute_asset_hits(text: str) -> list[str]:
    return re.findall(r"""(?:href|src)=["'](/[^"']*)""", text)


def missing_summaries(index: dict[str, Any], summaries_dir: Path) -> list[str]:
    missing: list[str] = []
    figures = index.get("figures") or []
    if not isinstance(figures, list):
        return ["figures is not a list"]
    for figure in figures:
        if not isinstance(figure, dict):
            continue
        summary = figure.get("summary")
        if not summary:
            continue
        name = Path(str(summary)).name
        if not (summaries_dir / name).is_file():
            missing.append(str(summary))
    return missing


def build_site(root: Path, dest: Path) -> None:
    papers = root / "papers"
    index_path = papers / "FIGURE-INDEX.json"
    schema_path = papers / "figure-index.schema.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    if not isinstance(index, dict) or not isinstance(schema, dict):
        raise ContractError("FIGURE-INDEX and schema must be objects")
    validate_index(index, schema)
    pdfs = find_committed_pdfs(papers)
    if pdfs:
        raise ContractError(f"committed PDFs: {pdfs}")
    tex_path = papers / "A" / "main.tex"
    if tex_path.is_file():
        violations = tex_contract_violations(tex_path.read_text(encoding="utf-8"))
        if violations:
            raise ContractError("; ".join(violations))
    missing = missing_summaries(index, papers / "figs" / "summaries")
    if missing:
        raise ContractError(f"missing summaries: {missing}")
    portal = root / "portal"
    html = (portal / "index.html").read_text(encoding="utf-8")
    abs_hits = absolute_asset_hits(html)
    if abs_hits:
        raise ContractError(f"root-absolute portal assets: {abs_hits}")
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    shutil.copytree(portal, dest, dirs_exist_ok=True)
    data = dest / "data"
    data.mkdir(exist_ok=True)
    shutil.copy2(index_path, data / "figures.json")
    summaries_src = papers / "figs" / "summaries"
    if summaries_src.is_dir():
        shutil.copytree(summaries_src, data / "summaries", dirs_exist_ok=True)
    for forbidden in ("experiments", ".omc", ".omx"):
        path = dest / forbidden
        if path.is_dir():
            shutil.rmtree(path)
        elif path.is_file():
            path.unlink()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("validate", "build"))
    parser.add_argument("dest", nargs="?", default="_site")
    args = parser.parse_args(argv)
    if args.command == "validate":
        index = json.loads((REPO_ROOT / "papers" / "FIGURE-INDEX.json").read_text())
        schema = json.loads(
            (REPO_ROOT / "papers" / "figure-index.schema.json").read_text()
        )
        if not isinstance(index, dict) or not isinstance(schema, dict):
            raise ContractError("FIGURE-INDEX and schema must be objects")
        validate_index(index, schema)
        return 0
    dest = Path(args.dest)
    if not dest.is_absolute():
        dest = REPO_ROOT / dest
    build_site(REPO_ROOT, dest)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ContractError as exc:
        print(f"contract error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

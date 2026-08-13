"""Unit tests for the Paper A warehouse contract helpers."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from warehouse_ci import (  # noqa: E402
    ContractError,
    absolute_asset_hits,
    build_site,
    find_committed_pdfs,
    missing_summaries,
    tex_contract_violations,
    validate_index,
)


MINIMAL_SCHEMA = {
    "type": "object",
    "required": [
        "paper_id",
        "github",
        "zenodo_concept_doi",
        "pipeline",
        "figures",
    ],
    "properties": {
        "paper_id": {"type": "string"},
        "github": {"type": "string"},
        "zenodo_concept_doi": {"type": "string"},
        "pipeline": {"type": "string"},
        "figures": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "label", "generator"],
                "properties": {
                    "id": {"type": "string"},
                    "label": {"type": "string"},
                    "generator": {"type": "string"},
                    "summary": {"type": ["string", "null"]},
                },
            },
        },
    },
}


def valid_index() -> dict[str, object]:
    return {
        "paper_id": "A",
        "github": "PeterPonyu/muon-norm-cap-grokking",
        "zenodo_concept_doi": "10.5281/zenodo.21020291",
        "pipeline": "papers/figs/PIPELINE.md",
        "figures": [
            {
                "id": "A_gap_normcap",
                "label": "fig:gap_normcap",
                "generator": "figs/make_gap20260705_figs_r.R",
                "summary": "summaries/A_gap_normcap.json",
            }
        ],
    }


class ValidateIndexTests(unittest.TestCase):
    def test_validate_index_rejects_missing_paper_id(self) -> None:
        payload = valid_index()
        del payload["paper_id"]
        with self.assertRaises(ContractError):
            validate_index(payload, MINIMAL_SCHEMA)

    def test_validate_index_accepts_minimal_valid_index(self) -> None:
        validate_index(valid_index(), MINIMAL_SCHEMA)


class CommittedPdfTests(unittest.TestCase):
    def test_committed_pdfs_reports_pdf_beside_main_tex(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            papers = Path(tmp) / "papers"
            paper_a = papers / "A"
            paper_a.mkdir(parents=True)
            sibling = paper_a / "A_gap_normcap.pdf"
            sibling.write_bytes(b"%PDF-1.4")
            found = find_committed_pdfs(papers)
            self.assertEqual(found, [sibling])

    def test_committed_pdfs_returns_empty_when_papers_has_no_pdfs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            papers = Path(tmp) / "papers"
            (papers / "A").mkdir(parents=True)
            (papers / "A" / "main.tex").write_text("% pointer\n", encoding="utf-8")
            self.assertEqual(find_committed_pdfs(papers), [])


class TexContractTests(unittest.TestCase):
    def test_tex_contract_rejects_previews_include(self) -> None:
        tex = (
            r"\input{../figs/figpreamble.tex}"
            "\n"
            r"\figtikz{A_gap_normcap}"
            "\n"
            r"\includegraphics{previews/A_gap_normcap.svg}"
            "\n"
        )
        violations = tex_contract_violations(tex)
        self.assertTrue(any("previews/" in item for item in violations))

    def test_tex_contract_rejects_peerj_figure_pdf(self) -> None:
        tex = (
            r"\input{../figs/figpreamble.tex}"
            "\n"
            r"\includegraphics{Figure3.pdf}"
            "\n"
        )
        violations = tex_contract_violations(tex)
        self.assertTrue(any("Figure" in item for item in violations))

    def test_tex_contract_requires_figtikz_and_figpreamble(self) -> None:
        violations = tex_contract_violations(r"\includegraphics{A_landscape.pdf}" + "\n")
        joined = " ".join(violations)
        self.assertIn("figtikz", joined)
        self.assertIn("figpreamble", joined)


class PortalAssetTests(unittest.TestCase):
    def test_portal_rejects_root_absolute_assets(self) -> None:
        hits = absolute_asset_hits('<link href="/styles.css">')
        self.assertEqual(hits, ["/styles.css"])

    def test_portal_allows_relative_assets(self) -> None:
        hits = absolute_asset_hits('<link href="instrument-tokens.css">')
        self.assertEqual(hits, [])


class SummaryPresenceTests(unittest.TestCase):
    def test_missing_summaries_reports_absent_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            summaries = Path(tmp)
            missing = missing_summaries(valid_index(), summaries)
            self.assertEqual(missing, ["summaries/A_gap_normcap.json"])

    def test_missing_summaries_returns_empty_when_json_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            summaries = Path(tmp)
            (summaries / "A_gap_normcap.json").write_text("{}", encoding="utf-8")
            self.assertEqual(missing_summaries(valid_index(), summaries), [])


class BuildSiteTests(unittest.TestCase):
    def test_build_site_copies_portal_and_index_not_experiments(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            portal = root / "portal"
            portal.mkdir()
            (portal / "index.html").write_text(
                '<header class="instrument"></header>\n',
                encoding="utf-8",
            )
            papers = root / "papers"
            papers.mkdir()
            index = valid_index()
            (papers / "FIGURE-INDEX.json").write_text(
                __import__("json").dumps(index),
                encoding="utf-8",
            )
            schema_path = papers / "figure-index.schema.json"
            schema_path.write_text(
                __import__("json").dumps(MINIMAL_SCHEMA),
                encoding="utf-8",
            )
            summaries = papers / "figs" / "summaries"
            summaries.mkdir(parents=True)
            (summaries / "A_gap_normcap.json").write_text("{}", encoding="utf-8")
            (root / "experiments").mkdir()
            (root / "experiments" / "secret.log").write_text("nope\n", encoding="utf-8")
            dest = root / "_site"
            build_site(root, dest)
            self.assertTrue((dest / "index.html").is_file())
            self.assertTrue((dest / "data" / "figures.json").is_file())
            self.assertFalse((dest / "experiments").exists())
            self.assertFalse((dest / "secret.log").exists())


if __name__ == "__main__":
    unittest.main()

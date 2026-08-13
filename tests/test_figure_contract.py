"""Integration tests for the Paper A figure-pointer contract (F1–F8)."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from warehouse_ci import (  # noqa: E402
    find_committed_pdfs,
    missing_summaries,
    tex_contract_violations,
    validate_index,
)


MANIFEST_IDS = {
    "A_gap_normlaw",
    "A_gap_normcap",
    "A_normctl",
    "A_normctl_timecourse",
    "A_floor",
    "A_norm_discriminator",
    "A_lmc",
    "A_sink",
    "A_plasticity",
    "A_synth",
}
SCHEMATIC_IDS = {"A_landscape", "A_scheme"}
ALLOWED_IDS = MANIFEST_IDS | SCHEMATIC_IDS


class FigureIndexSchemaTests(unittest.TestCase):
    def test_figure_index_validates_against_schema(self) -> None:
        index_path = ROOT / "papers" / "FIGURE-INDEX.json"
        schema_path = ROOT / "papers" / "figure-index.schema.json"
        index = json.loads(index_path.read_text(encoding="utf-8"))
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        validate_index(index, schema)
        self.assertEqual(index["paper_id"], "A")
        self.assertEqual(index["github"], "PeterPonyu/muon-norm-cap-grokking")
        self.assertEqual(index["zenodo_concept_doi"], "10.5281/zenodo.21020291")


class FigureIdTests(unittest.TestCase):
    def test_figure_ids_match_warehouse_or_documented_schematic(self) -> None:
        index = json.loads(
            (ROOT / "papers" / "FIGURE-INDEX.json").read_text(encoding="utf-8")
        )
        ids = {figure["id"] for figure in index["figures"]}
        unexpected = ids - ALLOWED_IDS
        self.assertEqual(unexpected, set(), f"unknown figure ids: {sorted(unexpected)}")
        missing_qualifying = MANIFEST_IDS - ids
        self.assertEqual(
            missing_qualifying,
            set(),
            f"missing qualifying ids: {sorted(missing_qualifying)}",
        )
        for figure in index["figures"]:
            if figure["id"] in SCHEMATIC_IDS:
                self.assertTrue(
                    figure["id"].endswith("_landscape")
                    or figure["id"].endswith("_scheme"),
                )
                self.assertTrue(figure.get("generator"))


class CommittedArtifactTests(unittest.TestCase):
    def test_no_pdf_is_committed_under_papers(self) -> None:
        papers = ROOT / "papers"
        found = find_committed_pdfs(papers)
        self.assertEqual(found, [], f"committed PDFs: {found}")


class MainTexContractTests(unittest.TestCase):
    def test_main_tex_does_not_reference_previews(self) -> None:
        tex = (ROOT / "papers" / "A" / "main.tex").read_text(encoding="utf-8")
        self.assertNotIn("previews/", tex)

    def test_main_tex_uses_figtikz_and_unmodified_figpreamble_input(self) -> None:
        tex = (ROOT / "papers" / "A" / "main.tex").read_text(encoding="utf-8")
        self.assertIn(r"\input{../figs/figpreamble.tex}", tex)
        self.assertIn(r"\figtikz", tex)
        violations = tex_contract_violations(tex)
        self.assertEqual(violations, [])

    def test_main_tex_does_not_include_venue_flat_pdfs(self) -> None:
        tex = (ROOT / "papers" / "A" / "main.tex").read_text(encoding="utf-8")
        self.assertNotRegex(tex, r"Figure\d+\.pdf")
        self.assertNotRegex(
            tex,
            r"\\includegraphics(?:\[[^\]]*\])?\{(?:\./)?A_[^}]+\.pdf\}",
        )

    def test_figpreamble_keeps_canonical_pointer_paths(self) -> None:
        preamble = (ROOT / "papers" / "figs" / "figpreamble.tex").read_text(
            encoding="utf-8"
        )
        self.assertIn(r"\graphicspath{{../figs/vec/}{../figs/}}", preamble)
        self.assertIn(r"\input{../figs/tex/#2.tex}", preamble)


class SummaryAndGitignoreTests(unittest.TestCase):
    def test_every_manifest_summary_json_is_present(self) -> None:
        index = json.loads(
            (ROOT / "papers" / "FIGURE-INDEX.json").read_text(encoding="utf-8")
        )
        summaries = ROOT / "papers" / "figs" / "summaries"
        missing = missing_summaries(index, summaries)
        self.assertEqual(missing, [])
        for artifact in MANIFEST_IDS:
            path = summaries / f"{artifact}.json"
            self.assertTrue(path.is_file(), f"missing {path}")

    def test_gitignore_excludes_compiled_tex_and_vec_tiers(self) -> None:
        text = (ROOT / ".gitignore").read_text(encoding="utf-8")
        self.assertIn("papers/figs/tex/", text)
        self.assertIn("papers/figs/vec/", text)


if __name__ == "__main__":
    unittest.main()

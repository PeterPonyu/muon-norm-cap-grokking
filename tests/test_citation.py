"""Citation, license, and README contract tests (C1, C3, C4, G4, I6)."""

from __future__ import annotations

import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONCEPT_DOI = "10.5281/zenodo.21020291"
VERSION_DOI = "10.5281/zenodo.21020292"


class CitationTests(unittest.TestCase):
    def test_citation_cff_parses(self) -> None:
        payload = yaml.safe_load((ROOT / "CITATION.cff").read_text(encoding="utf-8"))
        self.assertIsInstance(payload, dict)
        self.assertIn("title", payload)
        self.assertIsInstance(payload["title"], str)

    def test_citation_cff_uses_concept_doi(self) -> None:
        payload = yaml.safe_load((ROOT / "CITATION.cff").read_text(encoding="utf-8"))
        self.assertEqual(payload["doi"], CONCEPT_DOI)

    def test_citation_cff_lists_version_doi_in_identifiers(self) -> None:
        payload = yaml.safe_load((ROOT / "CITATION.cff").read_text(encoding="utf-8"))
        identifiers = payload.get("identifiers") or []
        values = {item.get("value") for item in identifiers if isinstance(item, dict)}
        self.assertIn(VERSION_DOI, values)

    def test_citation_cff_is_not_the_five_paper_bundle(self) -> None:
        text = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
        self.assertNotIn("architecture-staircase", text)
        self.assertNotIn("calibration-traps", text)
        payload = yaml.safe_load(text)
        title = payload["title"]
        self.assertNotIn("five paper", title.lower())


class LicenseAndReadmeTests(unittest.TestCase):
    def test_license_states_mit_and_cc_by_40(self) -> None:
        text = (ROOT / "LICENSE").read_text(encoding="utf-8")
        self.assertIn("MIT License", text)
        self.assertIn("CC BY 4.0", text)

    def test_readme_includes_github_url(self) -> None:
        text = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn(
            "https://github.com/PeterPonyu/muon-norm-cap-grokking",
            text,
        )


if __name__ == "__main__":
    unittest.main()

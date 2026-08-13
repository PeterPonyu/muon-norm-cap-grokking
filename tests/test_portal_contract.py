"""Integration tests for the Paper A portal stub (U1–U6, I1, I4)."""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PORTAL = ROOT / "portal"
EMOJI_RE = re.compile(r"[\U0001F300-\U0001FAFF]")
REQUIRED_NAV = ("Cap", "Dose", "Floor", "LMC")


def _portal_text_files() -> list[Path]:
    files: list[Path] = []
    for path in PORTAL.rglob("*"):
        if path.is_file() and path.suffix in {".html", ".css", ".js", ".sh", ".md"}:
            files.append(path)
    return files


class PortalUniquenessTests(unittest.TestCase):
    def test_portal_has_no_shared_theme_package_directory(self) -> None:
        self.assertFalse((ROOT / "portal-theme").exists())
        self.assertFalse((PORTAL / "tokens.css").exists())

    def test_index_html_uses_instrument_landmark(self) -> None:
        html = (PORTAL / "index.html").read_text(encoding="utf-8")
        self.assertIn("header", html)
        self.assertIn("instrument", html)
        self.assertNotIn("class=\"atlas\"", html)
        self.assertNotIn("class=\"notebook\"", html)

    def test_portal_contains_required_nav_labels(self) -> None:
        html = (PORTAL / "index.html").read_text(encoding="utf-8")
        for label in REQUIRED_NAV:
            self.assertIn(label, html)

    def test_portal_has_no_emoji(self) -> None:
        for path in _portal_text_files():
            text = path.read_text(encoding="utf-8")
            self.assertIsNone(EMOJI_RE.search(text), f"emoji in {path}")

    def test_footer_lists_concept_doi_github_and_dual_license(self) -> None:
        html = (PORTAL / "index.html").read_text(encoding="utf-8")
        self.assertIn("10.5281/zenodo.21020291", html)
        self.assertIn("https://github.com/PeterPonyu/muon-norm-cap-grokking", html)
        self.assertIn("MIT", html)
        self.assertIn("CC BY 4.0", html)

    def test_portal_does_not_host_journal_pdf(self) -> None:
        forbidden = list(PORTAL.rglob("main.pdf")) + list(PORTAL.rglob("manuscript.pdf"))
        self.assertEqual(forbidden, [])

    def test_portal_js_consumes_figure_index(self) -> None:
        js_files = list(PORTAL.glob("*.js"))
        self.assertTrue(js_files, "expected a portal JS stub")
        blob = "\n".join(path.read_text(encoding="utf-8") for path in js_files)
        self.assertTrue(
            "FIGURE-INDEX.json" in blob or "data/figures.json" in blob,
            "portal JS must read FIGURE-INDEX or data/figures.json",
        )
        self.assertNotIn("Figure3.pdf", blob)

    def test_portal_assets_are_relative(self) -> None:
        html = (PORTAL / "index.html").read_text(encoding="utf-8")
        self.assertNotRegex(html, r"""(?:href|src)=["']\/""")


class PortalBuildTests(unittest.TestCase):
    def test_build_script_validates_then_copies_without_latexmk(self) -> None:
        script = PORTAL / "build.sh"
        self.assertTrue(script.is_file())
        text = script.read_text(encoding="utf-8")
        self.assertNotIn("latexmk", text)
        self.assertIn("warehouse_ci.py", text)

    def test_site_artifact_excludes_experiments_and_omc(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "_site"
            result = subprocess.run(
                ["bash", str(PORTAL / "build.sh"), str(dest)],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                result.returncode,
                0,
                msg=result.stdout + "\n" + result.stderr,
            )
            self.assertTrue((dest / "index.html").is_file())
            self.assertTrue((dest / "data" / "figures.json").is_file())
            self.assertFalse((dest / "experiments").exists())
            self.assertFalse((dest / ".omc").exists())
            self.assertFalse((dest / "main.pdf").exists())
            shutil.rmtree(dest, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()

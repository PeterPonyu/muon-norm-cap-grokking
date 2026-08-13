"""Portal uniqueness, copy+validate, relative URLs (U-STUB, U3–U7, I1, P-A)."""

from __future__ import annotations

import re
import subprocess

from conftest import (
    CONCEPT_DOI,
    GITHUB_URL,
    NAV_LABELS,
    PORTAL_DIR,
    PORTAL_INDEX,
    REPO_ROOT,
    ROUTE_FILES,
)

EMOJI = re.compile(r"[\U0001F300-\U0001FAFF]")
STUB = re.compile(
    r"CI stub|instrument stub|Two-probe contract stub|"
    r"waits on a user-approved reference\.png|\bstub\b",
    re.IGNORECASE,
)
LANDMARKS = (
    'class="instrument"',
    'class="status-strip"',
    'class="readout-grid"',
    'class="claim-ledger"',
)
TOKENS = {
    "--ground": "#12151A",
    "--trace-amber": "#E8A838",
    "--cap-cyan": "#3EC8D8",
    "--font-sans": '"IBM Plex Sans"',
    "--font-mono": '"IBM Plex Mono"',
}
FORBIDDEN_SIBLING = (
    "Source Serif 4",
    "Source Sans 3",
    "Literata",
    "STIX Two Text",
    "Fraunces",
    "Atkinson Hyperlegible",
    "JetBrains Mono",
    "Newsreader",
    'class="atlas"',
    'class="notebook"',
    "field-guide",
    'class="console"',
    "#F4F1EA",
)


def _portal_blob() -> str:
    chunks: list[str] = []
    for path in sorted(PORTAL_DIR.rglob("*")):
        if path.is_file() and path.suffix.lower() in {".html", ".css", ".js", ".sh"}:
            chunks.append(path.read_text(encoding="utf-8", errors="replace"))
    return "\n".join(chunks)


def test_portal_routes_exist() -> None:
    assert PORTAL_DIR.is_dir()
    missing = [name for name in ROUTE_FILES.values() if not (PORTAL_DIR / name).is_file()]
    assert missing == [], f"missing routes: {missing}"


def test_index_is_instrument_home() -> None:
    html = PORTAL_INDEX.read_text(encoding="utf-8")
    assert "<header" in html and "instrument" in html
    assert 'role="banner"' in html or 'class="instrument"' in html
    for landmark in LANDMARKS:
        assert landmark in html
    assert "grid-template-columns: repeat(12" in html or "readout-grid" in html
    for key in ("CEILING k", "S5 GROK STEP", "RATIO", "BCa 95%"):
        assert key in html
    for row in ("PRESERVE", "ACCELERATE", "BOUNDARY"):
        assert row in html


def test_horizontal_module_nav() -> None:
    html = PORTAL_INDEX.read_text(encoding="utf-8")
    start = html.index("<nav")
    end = html.find("</nav>", start)
    nav = html[start : end + 6]
    for label in NAV_LABELS:
        assert label in nav
    for filename in ROUTE_FILES.values():
        assert filename in html


def test_anti_stub_and_no_emoji() -> None:
    blob = _portal_blob()
    assert not STUB.search(blob), "U-STUB"
    assert not EMOJI.search(blob), "U3"


def test_footer_identity() -> None:
    html = PORTAL_INDEX.read_text(encoding="utf-8")
    footer = html[html.lower().rfind("<footer") :]
    assert CONCEPT_DOI in footer
    assert GITHUB_URL in footer
    assert "MIT" in footer
    assert "CC BY" in footer or "CC-BY" in footer


def test_consumes_figure_index_not_venue_pdfs() -> None:
    blob = _portal_blob()
    assert "FIGURE-INDEX.json" in blob or "data/figures.json" in blob
    assert "PIPELINE.md" in blob
    for stem in ("main.pdf", "manuscript.pdf", "Figure1.pdf", "Figure3.pdf", "A_landscape.pdf"):
        assert stem not in blob
    assert r"\includegraphics" not in blob
    assert r"\figtikz" not in blob
    assert "graphicspath" not in blob.lower()


def test_relative_asset_urls_only() -> None:
    blob = _portal_blob()
    assert 'href="/' not in blob
    assert 'src="/' not in blob
    assert "url(/" not in blob
    assert "/assets" not in blob


def test_tokens_and_not_sibling_skins() -> None:
    css = (PORTAL_DIR / "assets" / "instrument.css").read_text(encoding="utf-8")
    for name, value in TOKENS.items():
        assert value in css, f"missing token {name} {value}"
    blob = _portal_blob()
    for sig in FORBIDDEN_SIBLING:
        assert sig not in blob
    assert "grokking-clock" not in blob
    assert "architecture-staircase" not in blob
    assert "free-repetition-band" not in blob
    assert "calibration-traps" not in blob


def test_no_journal_pdfs_in_portal_tree() -> None:
    pdfs = list(PORTAL_DIR.rglob("*.pdf"))
    assert pdfs == []


def test_build_script_copy_validate() -> None:
    build = PORTAL_DIR / "build.sh"
    assert build.is_file()
    text = build.read_text(encoding="utf-8")
    assert "latexmk" not in text.lower()
    assert "_site" in text
    assert "FIGURE-INDEX" in text
    assert "experiments/" not in text
    subprocess.run(["bash", str(build)], cwd=REPO_ROOT, check=True)
    site = REPO_ROOT / "_site"
    assert (site / "index.html").is_file()
    assert (site / "data" / "figures.json").is_file()
    assert (site / "data" / "figs" / "summaries" / "A_normctl.json").is_file()
    assert not (site / "experiments").exists()
    assert not list(site.rglob("main.pdf"))
    assert not list(site.rglob("manuscript.pdf"))

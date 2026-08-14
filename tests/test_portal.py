"""Portal: Next.js export, instrument chrome, no paper leak (U-STUB, U3–U7, I1, P-A)."""

from __future__ import annotations

import json
import re
import subprocess

from conftest import (
    APP_ROUTES,
    BASE_PATH,
    CONCEPT_DOI,
    EXPORT_ROUTES,
    FILENAME_LEAKS,
    FORBIDDEN_UI,
    GITHUB_URL,
    LEAK_PATTERNS,
    NAV_LABELS,
    PORTAL_DIR,
    REPO_ROOT,
    SCIENCE_TOKENS,
)

EMOJI = re.compile(r"[\U0001F300-\U0001FAFF]")
STUB = re.compile(
    r"CI stub|instrument stub|Two-probe contract stub|"
    r"waits on a user-approved reference\.png|\bstub\b",
    re.IGNORECASE,
)
LANDMARKS = (
    'class="instrument"',
    "status-strip",
    "readout-grid",
    "claim-ledger",
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
SOURCE_SUFFIXES = {".tsx", ".ts", ".css", ".js", ".json", ".mjs"}


def _portal_source_blob() -> str:
    chunks: list[str] = []
    skip = {"node_modules", ".next", "out"}
    for path in sorted(PORTAL_DIR.rglob("*")):
        if any(part in skip for part in path.parts):
            continue
        if path.is_file() and path.suffix.lower() in SOURCE_SUFFIXES | {".sh"}:
            chunks.append(path.read_text(encoding="utf-8", errors="replace"))
    return "\n".join(chunks)


def test_next_app_routes_exist() -> None:
    missing = [str(p.relative_to(REPO_ROOT)) for p in APP_ROUTES if not p.is_file()]
    assert missing == [], f"missing Next.js app routes: {missing}"
    config = (PORTAL_DIR / "next.config.ts").read_text(encoding="utf-8")
    assert "output: \"export\"" in config or "output: 'export'" in config
    assert 'basePath: "/muon-norm-cap-grokking"' in config or "basePath: '/muon-norm-cap-grokking'" in config
    assert "trailingSlash: true" in config
    layout = (PORTAL_DIR / "app" / "layout.tsx").read_text(encoding="utf-8")
    assert "next/font/google" not in layout, "C8: build must not fetch Google Fonts"
    assert "next/font/local" in layout
    fonts = PORTAL_DIR / "app" / "fonts"
    for stem in ("ibm-plex-sans", "ibm-plex-mono"):
        for weight in ("400", "500", "600"):
            assert (fonts / f"{stem}-latin-{weight}-normal.woff2").is_file(), (
                f"C8: missing self-hosted {stem} {weight} woff2"
            )
    assert (fonts / "OFL.txt").is_file(), "C8: OFL license must ship with the fonts"


def test_module_nav_labels_in_source() -> None:
    nav = (PORTAL_DIR / "components" / "ModuleNav.tsx").read_text(encoding="utf-8")
    for label in NAV_LABELS:
        assert label in nav
    assert 'href: "/dose/"' in nav or 'href: "/dose"' in nav


def test_anti_stub_and_no_emoji() -> None:
    blob = _portal_source_blob()
    assert not STUB.search(blob), "U-STUB"
    assert not EMOJI.search(blob), "U3"


def test_no_leaked_findings_in_portal_source() -> None:
    blob = _portal_source_blob()
    hits = [pat for pat in LEAK_PATTERNS if pat.lower() in blob.lower()]
    assert hits == [], f"portal source leaks findings: {hits}"
    assert r"\includegraphics" not in blob
    assert "graphicspath" not in blob.lower()
    for stem in ("main.pdf", "manuscript.pdf", "Figure1.pdf", "A_landscape.pdf"):
        assert stem not in blob


def test_tokens_and_not_sibling_skins() -> None:
    css = (PORTAL_DIR / "app" / "globals.css").read_text(encoding="utf-8")
    for name, value in TOKENS.items():
        assert value in css, f"missing token {name} {value}"
    blob = _portal_source_blob()
    for sig in FORBIDDEN_SIBLING:
        assert sig not in blob
    assert "grokking-clock" not in blob
    assert "architecture-staircase" not in blob


def test_no_journal_pdfs_in_portal_tree() -> None:
    pdfs = [p for p in PORTAL_DIR.rglob("*.pdf") if "node_modules" not in p.parts]
    assert pdfs == []


def test_build_exports_next_out_and_site() -> None:
    build = PORTAL_DIR / "build.sh"
    assert build.is_file()
    text = build.read_text(encoding="utf-8")
    assert "latexmk" not in text.lower()
    assert "npm run build" in text
    assert "_site" in text
    assert "FIGURE-INDEX" in text
    assert "experiments/" not in text
    subprocess.run(["bash", str(build)], cwd=REPO_ROOT, check=True)
    out = PORTAL_DIR / "out"
    site = REPO_ROOT / "_site"
    assert (out / "index.html").is_file(), "Next.js export missing portal/out/index.html"
    for rel in EXPORT_ROUTES:
        assert (out / rel).is_file(), f"export missing {rel} (trailingSlash refresh)"
        assert (site / rel).is_file(), f"_site missing {rel}"
    html = (site / "index.html").read_text(encoding="utf-8")
    for landmark in LANDMARKS:
        assert landmark in html
    for label in NAV_LABELS:
        assert label in html
    assert CONCEPT_DOI in html
    assert "PeterPonyu/muon-norm-cap-grokking" in html
    assert "MIT" in html
    assert "CC BY" in html
    assert BASE_PATH in html
    assert (site / "data" / "figures.json").is_file()
    data = json.loads((site / "data" / "figures.json").read_text(encoding="utf-8"))
    assert "caption" not in json.dumps(data)
    assert not (site / "experiments").exists()
    assert not list(site.rglob("main.pdf"))


def _visible_html(html: str) -> str:
    stripped = re.sub(r"<script\b[^>]*>.*?</script>", " ", html, flags=re.I | re.S)
    stripped = re.sub(r"<style\b[^>]*>.*?</style>", " ", stripped, flags=re.I | re.S)
    stripped = re.sub(r"<[^>]+>", " ", stripped)
    return stripped


def _forbidden_hits(text: str) -> list[str]:
    lower = text.lower()
    return [word for word in FORBIDDEN_UI if word in lower]


def test_portal_source_has_no_document_framing() -> None:
    hits: list[str] = []
    roots = (
        PORTAL_DIR / "app",
        PORTAL_DIR / "components",
        PORTAL_DIR / "content",
    )
    for root in roots:
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix.lower() in {".tsx", ".ts", ".css", ".json"}:
                found = _forbidden_hits(path.read_text(encoding="utf-8", errors="replace"))
                if found:
                    hits.append(f"{path.relative_to(PORTAL_DIR)}: {found}")
    assert hits == [], f"instrument copy still frames a document: {hits}"


def test_exported_html_has_no_document_framing() -> None:
    site = REPO_ROOT / "_site"
    if not (site / "index.html").is_file():
        subprocess.run(["bash", str(PORTAL_DIR / "build.sh")], cwd=REPO_ROOT, check=True)
    hits: list[str] = []
    for rel in EXPORT_ROUTES:
        visible = _visible_html((site / rel).read_text(encoding="utf-8"))
        found = _forbidden_hits(visible)
        if found:
            hits.append(f"{rel}: {found}")
    assert hits == [], f"exported visible copy still frames a document: {hits}"


def test_exported_html_shows_science_not_filenames() -> None:
    site = REPO_ROOT / "_site"
    if not (site / "index.html").is_file():
        subprocess.run(["bash", str(PORTAL_DIR / "build.sh")], cwd=REPO_ROOT, check=True)
    home = _visible_html((site / "index.html").read_text(encoding="utf-8"))
    missing = [tok for tok in SCIENCE_TOKENS if tok.lower() not in home.lower()]
    assert missing == [], f"home is missing scientific structure: {missing}"
    leaks: list[str] = []
    for rel in EXPORT_ROUTES:
        visible = _visible_html((site / rel).read_text(encoding="utf-8")).lower()
        found = [tok for tok in FILENAME_LEAKS if tok in visible]
        if found:
            leaks.append(f"{rel}: {found}")
    assert leaks == [], f"exported visible copy still prints paths: {leaks}"


def test_exported_html_uses_basepath_not_user_site_assets() -> None:
    site = REPO_ROOT / "_site"
    if not (site / "index.html").is_file():
        subprocess.run(["bash", str(PORTAL_DIR / "build.sh")], cwd=REPO_ROOT, check=True)
    html = (site / "index.html").read_text(encoding="utf-8")
    assert 'href="/assets' not in html
    assert 'src="/assets' not in html
    assert f'href="{BASE_PATH}/' in html or f'"{BASE_PATH}/_next/' in html
    blob = html
    for pat in LEAK_PATTERNS:
        assert pat not in blob, f"exported HTML leaks {pat}"

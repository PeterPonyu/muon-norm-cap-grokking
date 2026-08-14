"""GitHub Actions: required ci.yml; pages.yml I2/I3; optional visual-pixel."""

from __future__ import annotations

import yaml

from conftest import REPO_ROOT, workflow_path


def _load(name: str) -> dict:
    path = workflow_path(name)
    assert path.is_file(), f"missing .github/workflows/{name}"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


def test_ci_workflow_runs_pytest() -> None:
    data = _load("ci.yml")
    jobs = data.get("jobs") or {}
    assert "contract" in jobs
    blob = "\n".join(str(step) for step in jobs["contract"].get("steps") or [])
    assert "pytest" in blob
    assert "latexmk" not in blob.lower()
    on = data.get("on") or data.get(True) or {}
    if isinstance(on, dict):
        push = on.get("push") or {}
        branches = push.get("branches") if isinstance(push, dict) else None
        if branches:
            assert "ci/comprehensive" not in branches


def test_pages_permissions_environment_and_main_only() -> None:
    data = _load("pages.yml")
    perms = data.get("permissions") or {}
    assert perms.get("pages") == "write"
    jobs = data.get("jobs") or {}
    env_names = []
    for job in jobs.values():
        env = job.get("environment")
        if isinstance(env, str):
            env_names.append(env)
        elif isinstance(env, dict):
            env_names.append(env.get("name"))
    assert "github-pages" in env_names
    on = data.get("on") or data.get(True)
    assert isinstance(on, dict)
    assert "workflow_dispatch" in on
    push = on.get("push") or {}
    assert push.get("branches") == ["main"], "I3: branches: [main] only"
    paths = push.get("paths") or []
    joined = " ".join(paths)
    assert "portal/**" in joined
    assert "papers/FIGURE-INDEX.json" in joined
    assert "papers/figs/summaries/**" in joined
    assert "papers/figs/previews/**" in joined
    assert ".github/workflows/pages.yml" in joined
    assert "ci/comprehensive" not in yaml.safe_dump(data)
    assert "deploy" in jobs
    blob = yaml.safe_dump(data)
    assert "latexmk" not in blob.lower()
    assert "pdflatex" not in blob.lower()


def test_visual_pixel_is_optional() -> None:
    path = workflow_path("visual-pixel.yml")
    if not path.is_file():
        return
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    on = data.get("on") or data.get(True)
    assert "workflow_dispatch" in on or (isinstance(on, dict) and "workflow_dispatch" in on)


def test_no_workflow_compiles_latex() -> None:
    workflows = list((REPO_ROOT / ".github" / "workflows").glob("*.yml"))
    assert workflows
    for path in workflows:
        text = path.read_text(encoding="utf-8").lower()
        assert "latexmk" not in text
        assert "pdflatex" not in text
        assert "lualatex" not in text

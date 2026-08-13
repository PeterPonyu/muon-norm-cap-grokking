"""GitHub Actions contract tests: required CI vs gated Pages/Visual Ralph."""

from __future__ import annotations

import unittest
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"


def load_workflow(name: str) -> dict[str, Any]:
    path = WORKFLOWS / name
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise AssertionError(f"{path} is not a mapping")
    return payload


def job_if(job: dict[str, Any]) -> str:
    value = job.get("if", "")
    return str(value).strip().lower()


def workflow_on(payload: dict[str, Any]) -> dict[str, Any]:
    # PyYAML 1.1 parses unquoted `on:` as boolean True.
    raw: Any = payload
    value = raw.get("on")
    if value is None:
        value = raw.get(True)
    if not isinstance(value, dict):
        return {}
    return value


class PagesWorkflowTests(unittest.TestCase):
    def test_pages_workflow_exists_with_pages_write_permission(self) -> None:
        payload = load_workflow("pages.yml")
        permissions = payload.get("permissions") or {}
        self.assertEqual(permissions.get("pages"), "write")
        self.assertEqual(permissions.get("id-token"), "write")

    def test_pages_workflow_declares_github_pages_environment(self) -> None:
        payload = load_workflow("pages.yml")
        jobs = payload.get("jobs") or {}
        environments = []
        for job in jobs.values():
            env = job.get("environment")
            if isinstance(env, dict):
                environments.append(env.get("name"))
            else:
                environments.append(env)
        self.assertIn("github-pages", environments)

    def test_pages_deploy_job_is_gated_off(self) -> None:
        payload = load_workflow("pages.yml")
        jobs = payload.get("jobs") or {}
        self.assertIn("deploy", jobs)
        condition = job_if(jobs["deploy"])
        self.assertIn(condition, {"false", "${{ false }}"})

    def test_pages_push_paths_include_portal_index_summaries_previews(self) -> None:
        payload = load_workflow("pages.yml")
        push = workflow_on(payload).get("push") or {}
        paths = set(push.get("paths") or [])
        self.assertIn("portal/**", paths)
        self.assertIn("papers/FIGURE-INDEX.json", paths)
        self.assertIn("papers/figs/summaries/**", paths)
        self.assertIn("papers/figs/previews/**", paths)
        self.assertIn(".github/workflows/pages.yml", paths)

    def test_pages_workflow_supports_workflow_dispatch(self) -> None:
        payload = load_workflow("pages.yml")
        self.assertIn("workflow_dispatch", workflow_on(payload))

    def test_pages_workflow_does_not_run_latexmk(self) -> None:
        text = (WORKFLOWS / "pages.yml").read_text(encoding="utf-8")
        self.assertNotIn("latexmk", text)
        self.assertNotIn("pdflatex", text)
        self.assertNotIn("lualatex", text)


class RequiredCiWorkflowTests(unittest.TestCase):
    def test_ci_workflow_runs_lint_typecheck_unit_and_integration(self) -> None:
        payload = load_workflow("ci.yml")
        jobs = payload.get("jobs") or {}
        for name in ("lint", "typecheck", "unit", "integration"):
            self.assertIn(name, jobs)
            condition = job_if(jobs[name])
            self.assertNotIn(condition, {"false", "${{ false }}"})

    def test_ci_workflow_invokes_documented_test_commands(self) -> None:
        text = (WORKFLOWS / "ci.yml").read_text(encoding="utf-8")
        self.assertIn("ruff check", text)
        self.assertIn("mypy", text)
        self.assertIn("python -m unittest", text)
        self.assertIn("portal/build.sh", text)


class VisualRalphWorkflowTests(unittest.TestCase):
    def test_visual_ralph_is_a_separate_optional_workflow(self) -> None:
        self.assertTrue((WORKFLOWS / "visual-ralph.yml").is_file())
        self.assertTrue((WORKFLOWS / "ci.yml").is_file())
        ci = load_workflow("ci.yml")
        jobs = ci.get("jobs") or {}
        for _name, job in jobs.items():
            blob = yaml.safe_dump(job)
            self.assertNotIn("score >= 90", blob)
            self.assertNotIn("visual-ralph", blob.lower())

    def test_visual_ralph_job_is_disabled_until_reference_approved(self) -> None:
        payload = load_workflow("visual-ralph.yml")
        jobs = payload.get("jobs") or {}
        self.assertTrue(jobs, "visual-ralph.yml must define a job")
        for job in jobs.values():
            condition = job_if(job)
            self.assertIn(condition, {"false", "${{ false }}"})
        triggers = workflow_on(payload)
        self.assertIn("workflow_dispatch", triggers)
        self.assertNotIn("push", triggers)


if __name__ == "__main__":
    unittest.main()

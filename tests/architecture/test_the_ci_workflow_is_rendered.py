"""The GitHub Actions workflow is RENDERED from TOML (`dev_lab.dev.ciworkflows`): no drift, no hand-written file."""

from __future__ import annotations

from pathlib import Path

from dev_lab.dev.ciworkflows import drift, workflows

ROOT = Path(__file__).resolve().parents[2]


def test_the_repo_runs_the_family_pr_lifecycle() -> None:
    # continue.yml gained 2026-11-08 with dev-lab #320 (a landed branch continues in a new PR).
    # review-prepare.yml left 2026-11-19 with dev-lab #772 (the judge prepares its own bundle, D73);
    # review-again.yml left 2026-11-20 with dev-lab 2adcab2 (no longer declared by the base set).
    assert set(workflows(ROOT)) == {
        'ci.yml',
        'continue.yml',
        'merge-tick.yml',
    }


def test_every_workflow_matches_its_toml() -> None:
    assert drift(ROOT) == []

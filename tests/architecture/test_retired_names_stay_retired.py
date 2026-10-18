"""Retired spellings stay retired: no shim, no alias, no prose that keeps the old name alive.

THE REGISTRY is the table below -- the one place an old spelling may appear -- each with the name
that replaced it. A rename is NOT backward compatible (plan principle), so the scan covers the
engine's Python and Rust sources.
"""

from __future__ import annotations

from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]

#: old spelling -> what replaced it (2026-10-18, Track B3).
RETIRED: dict[str, str] = {
    'linear::dirichlet': 'linear::constrained::fixed_values (pin_rows: linear::constrained::pin_rows)',
    'linear::constraints': 'linear::constrained::master_slave',
    'mod dirichlet': 'mod fixed_values',
    'mod constraints': 'mod master_slave',
    'time_integrators._step.directional_line_search': 'numerics_lab.line_search.directional_line_search',
    'from numerics_lab.time_integrators import directional_line_search': 'numerics_lab.line_search',
}


def live_spellings(text: str) -> set[str]:
    return {old for old in RETIRED if old in text}


def _sources() -> list[Path]:
    files = [*(_ROOT / 'python').rglob('*.py'), *(_ROOT / 'crates').rglob('*.rs'), *(_ROOT / 'tests').rglob('*.py')]
    return [path for path in files if path.resolve() != Path(__file__).resolve()]


def test_the_scan_sees_a_planted_retired_spelling() -> None:
    assert live_spellings('use crate::linear::dirichlet::pin_rows;\nfine') == {'linear::dirichlet'}


def test_no_source_spells_a_retired_name() -> None:
    files = _sources()
    assert len(files) > 100, f'the scan reached only {len(files)} files'
    offenders = {
        path.relative_to(_ROOT).as_posix(): found
        for path in files
        if (found := live_spellings(path.read_text(encoding='utf-8')))
    }
    assert offenders == {}

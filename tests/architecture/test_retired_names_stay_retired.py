"""Retired spellings stay retired: no shim, no alias, no prose that keeps the old name alive.

THE REGISTRY is the table below -- the one place an old spelling may appear -- each with the name
that replaced it, the decision that retired it and its class. A rename is NOT backward compatible (plan principle), so the scan covers the
engine's Python and Rust sources.
"""

from __future__ import annotations

from pathlib import Path

from dev_lab.dev.retirementclass import refusals

_ROOT = Path(__file__).resolve().parents[2]

#: old spelling -> (what replaced it, the decision that retired it, its class -- dev_lab.dev.retirementclass).
_B3 = '2026-10-18, Track B3: the constrained-solve modules are named for what they hold'
_LS = '2026-10-18: the line search is one public module, not an integrator internal'
_SPLIT = "2026-11-26, engine split: optimization strategies and sampling are optimi-lab's; callers import them there"
RETIRED: dict[str, tuple[str, str, str]] = {
    # A bit-exact idempotence test was the eligibility predicate; floating reapplication can drift one bit
    # forever, so it was a defect replaced by tested provenance, not a method.
    'projection_fixed_point': (
        'retained owner-projected trial provenance, or explicit affine admission',
        '2026-11-06: a directional ray inherits only a tested projected iterate',
        'baggage',
    ),
    'linear::dirichlet': (
        'linear::constrained::fixed_values (pin_rows: linear::constrained::pin_rows)',
        _B3,
        'renamed',
    ),
    'linear::constraints': ('linear::constrained::master_slave', _B3, 'renamed'),
    'mod dirichlet': ('mod fixed_values', _B3, 'renamed'),
    'mod constraints': ('mod master_slave', _B3, 'renamed'),
    'time_integrators._step.directional_line_search': (
        'numerics_lab.line_search.directional_line_search',
        _LS,
        'renamed',
    ),
    'from numerics_lab.time_integrators import directional_line_search': ('numerics_lab.line_search', _LS, 'renamed'),
    'material_change': (
        'state_change (AssembledState, _has_converged, _StagnationWatch.observe)',
        'the change watched is any assembled state, not only a material',
        'renamed',
    ),
    'numerics_lab.trust_region': (
        'optimi_lab.operator.trust_region (agreement, update_radius, dogleg)',
        _SPLIT,
        'renamed',
    ),
    'numerics_lab.random_streams': ('optimi_lab.core.random_streams.Stream', _SPLIT, 'renamed'),
    'bfgs_update': ('optimi_lab.operator.quasi_newton.bfgs_update', _SPLIT, 'renamed'),
    'bfgs_inverse_update': ('optimi_lab.operator.quasi_newton.bfgs_inverse_update', _SPLIT, 'renamed'),
    'sr1_update': ('optimi_lab.operator.quasi_newton.sr1_update', _SPLIT, 'renamed'),
}


def live_spellings(text: str) -> set[str]:
    return {old for old in RETIRED if old in text}


def _sources() -> list[Path]:
    files = [*(_ROOT / 'python').rglob('*.py'), *(_ROOT / 'crates').rglob('*.rs'), *(_ROOT / 'tests').rglob('*.py')]
    return [path for path in files if path.resolve() != Path(__file__).resolve()]


def test_the_scan_sees_a_planted_retired_spelling() -> None:
    assert live_spellings('use crate::linear::dirichlet::pin_rows;\nfine') == {'linear::dirichlet'}


def test_the_removed_trial_helper_is_seen_in_each_real_python_spelling() -> None:
    old = next(name for name, replacement in RETIRED.items() if replacement[0].startswith('retained owner-projected'))
    for source in (
        f'from numerics_lab.time_integrators._trials import {old}',
        f'from ._trials import {old}',
        f'_trials.{old}(problem)',
        f'def {old}(problem): pass',
    ):
        assert live_spellings(source) == {old}


def test_no_source_spells_a_retired_name() -> None:
    files = _sources()
    assert len(files) > 100, f'the scan reached only {len(files)} files'
    offenders = {
        path.relative_to(_ROOT).as_posix(): found
        for path in files
        if (found := live_spellings(path.read_text(encoding='utf-8')))
    }
    assert offenders == {}


def test_every_retirement_is_classified() -> None:
    assert refusals(RETIRED) == []


def test_the_classifier_refuses_a_planted_unclassified_row_and_an_unruled_drop() -> None:
    planted = {
        'old_name': ('new_name', 'a decision'),
        'old_method': ('nothing', '2026-11-26: tidied away', 'capability-dropped'),
    }
    assert [line.split(':', 1)[0] for line in refusals(planted)] == ["'old_name'", "'old_method'"]

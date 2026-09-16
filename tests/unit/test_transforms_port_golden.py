"""The Park/Clarke port's golden vectors — one file, read by BOTH languages.

Companion to `tests/unit/core/numerics/test_chart_port_golden.py`; read that module's
docstring for why the comparison goes through a committed data file rather than across the
FFI, and why the Python half of the pair IS the migration plan's freeze protocol made
executable.

Regenerate deliberately::

    MOTRONICS_REGEN_PORT_GOLDENS=1 .venv/Scripts/python.exe -m pytest \
        tests/unit/gauss/test_transforms_port_golden.py
"""

from __future__ import annotations

import os

import numpy as np
import pytest
from tests.port_goldens import dump_golden, load_golden

from motronics.core.utils.config import repo_root
from motronics.gauss.transforms import (
    clarke_forward,
    park_forward,
    park_inverse,
    phase_L_to_dq_L,
)

_GOLDEN = repo_root() / 'rust' / 'crates' / 'gauss' / 'tests' / 'goldens' / 'transforms.json'

#: Sample rotor electrical angles. Deliberately not a uniform sweep: an evenly spaced set can
#: hide a phase-axis/sample-axis transposition, because every phase then sees an angle that
#: some other phase also sees.
_THETA = [0.0, 0.7, 2.5, -1.3, 3.9]

#: Phase counts to cover. Three is the production case; five and six are what proves the
#: `2/N` amplitude-invariant scaling is a function of N and not a hard-coded 2/3.
_N_PHASE = (3, 5, 6)


def _balanced(n_phase: int, amp: float, theta: np.ndarray) -> np.ndarray:
    """A balanced N-phase set of peak `amp`, aligned so d = amp and q = 0."""
    axes = 2.0 * np.pi * np.arange(n_phase, dtype=float) / n_phase
    return amp * np.cos(theta[None, :] - axes[:, None])


def _unbalanced(n_phase: int, theta: np.ndarray) -> np.ndarray:
    """A set with a zero sequence and a q component, so d/q/zero are all exercised.

    A purely balanced fixture leaves `q` and `zero` identically zero, and a port that dropped
    either term would pass against it. The floor test below asserts this fixture does not.
    """
    base = _balanced(n_phase, 2.0, theta + 0.4)
    offset = np.linspace(0.3, 1.1, n_phase)[:, None]
    return base + offset


def _build() -> dict:
    theta = np.asarray(_THETA, dtype=float)
    cases: dict[str, dict] = {}
    for n_phase in _N_PHASE:
        for label, x in (('balanced', _balanced(n_phase, 4.0, theta)), ('unbalanced', _unbalanced(n_phase, theta))):
            d, q, zero = park_forward(x, theta)
            entry: dict = {
                'n_phase': n_phase,
                'theta_e': theta.tolist(),
                'x_phase': x.tolist(),
                'park_forward': {'d': d.tolist(), 'q': q.tolist(), 'zero': zero.tolist()},
                'park_inverse': park_inverse(d, q, zero, theta, n_phase=n_phase).tolist(),
            }
            if n_phase == 3:
                alpha, i_phase_angle = clarke_forward(x)
                entry['clarke_forward'] = {'alpha': alpha.tolist(), 'i_phase_angle': i_phase_angle.tolist()}
            cases[f'{label}_{n_phase}ph'] = entry

    # The dq bridge, on a deliberately ASYMMETRIC L so the (2/n) factor cannot cancel by luck.
    l_phase = np.array([[1.3, -0.2, -0.2], [-0.2, 0.9, -0.15], [-0.2, -0.15, 1.1]])
    bridge = {
        'n_phase': 3,
        'L_phase': l_phase.tolist(),
        'theta_e': theta.tolist(),
        'L_dq': phase_L_to_dq_L(l_phase, theta).tolist(),
    }
    return {
        'note': (
            'Generated from src/motronics/gauss/transforms.py. Read by BOTH '
            'tests/unit/gauss/test_transforms_port_golden.py and '
            'rust/crates/gauss/tests/golden_transforms.rs.'
        ),
        'cases': cases,
        'phase_L_to_dq_L': bridge,
    }


def _load() -> dict:
    # Under regeneration the golden is being REWRITTEN by a sibling test, and xdist runs the two
    # in different workers -- a reader that raced the writer read a truncated file and failed on
    # a coverage assertion that had nothing wrong with it (measured 2026-09-04). A regeneration
    # run is not a verdict about anything, so every reader stands down for it.
    if os.environ.get('MOTRONICS_REGEN_PORT_GOLDENS'):
        pytest.skip('regenerating: the golden is being rewritten by a sibling test')
    if not _GOLDEN.is_file():
        pytest.fail(
            f'{_GOLDEN} is missing. It is a committed artifact, not a cache -- regenerate it '
            f'deliberately with MOTRONICS_REGEN_PORT_GOLDENS=1 and commit the result.'
        )
    return load_golden(_GOLDEN)


def _walk(got: object, want: object, path: str) -> None:
    if isinstance(want, dict):
        assert isinstance(got, dict) and set(got) == set(want), f'{path}: key set moved'
        for key in want:
            _walk(got[key], want[key], f'{path}.{key}')
    elif isinstance(want, list):
        np.testing.assert_allclose(
            np.asarray(got, dtype=float), np.asarray(want, dtype=float), rtol=1e-15, atol=0.0, err_msg=path
        )
    else:
        assert got == want, path


@pytest.mark.unit
def test_the_live_python_still_produces_the_frozen_vectors() -> None:
    """The freeze, enforced: a behaviour change in `transforms.py` reds HERE, not in silence."""
    if os.environ.get('MOTRONICS_REGEN_PORT_GOLDENS'):
        dump_golden(_GOLDEN, _build())
        pytest.skip(f'regenerated {_GOLDEN} -- commit it, then run again without the env var')

    golden = _load()
    live = _build()
    for section in ('cases', 'phase_L_to_dq_L'):
        _walk(live[section], golden[section], section)


@pytest.mark.unit
def test_the_golden_covers_every_transform_and_every_phase_count() -> None:
    """A golden narrower than the thing it freezes is a scan with no floor."""
    golden = _load()
    for n_phase in _N_PHASE:
        assert f'balanced_{n_phase}ph' in golden['cases'], f'{n_phase}-phase case missing'
    covered = {key for case in golden['cases'].values() for key in case}
    assert {'park_forward', 'park_inverse', 'clarke_forward'} <= covered, (
        f'the golden does not exercise every transform; it covers {sorted(covered)}'
    )
    assert golden['phase_L_to_dq_L']['L_dq'], 'the dq bridge is not covered'


@pytest.mark.unit
def test_the_unbalanced_fixture_actually_exercises_q_and_zero() -> None:
    """The floor under the fixtures themselves.

    A balanced set has `q == 0` and `zero == 0` identically, so a port that dropped either
    term would pass against balanced fixtures alone. The unbalanced case must move both, or it
    is not doing the job it was added for.
    """
    golden = _load()
    for name, case in golden['cases'].items():
        if not name.startswith('unbalanced'):
            continue
        for component in ('q', 'zero'):
            values = np.abs(np.asarray(case['park_forward'][component], dtype=float))
            assert values.max() > 1e-6, f'{name}: {component} is identically zero, so it proves nothing'

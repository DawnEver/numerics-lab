"""A repaired direct solve keeps a multi-column RHS's shape: each column is repaired alone (numerics-lab #32)."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from numerics_lab.linear import refinement


def _system(n: int = 40):
    main, off = np.full(n, 2.0), np.full(n - 1, -1.0)
    return sp.csr_matrix(sp.diags([off, main, off], offsets=[-1, 0, 1]))


@pytest.fixture
def repairable(monkeypatch):
    """Open the repair window: the conditioning gate is its own tests'; here it must say yes."""
    monkeypatch.setattr(refinement, 'move_is_this_systems_conditioning', lambda *_a, **_k: True)


@pytest.mark.usefixtures('repairable')
def test_a_repaired_multi_column_solve_returns_every_column() -> None:
    matrix = _system()
    rhs = np.stack([np.ones(40), np.linspace(0.0, 1.0, 40), np.cos(np.arange(40.0))], axis=1)
    exact = sp.linalg.spsolve(sp.csc_matrix(matrix), rhs)
    repaired, moves = refinement.repair_direct_solution(matrix, rhs, exact * 1.05)
    assert repaired.shape == rhs.shape
    assert moves, 'the repair must have fired, or this test measures nothing'
    np.testing.assert_allclose(repaired, exact, rtol=1e-9, atol=1e-12)


@pytest.mark.usefixtures('repairable')
def test_planted_control_a_single_column_rhs_keeps_its_vector_shape() -> None:
    matrix = _system()
    rhs = np.ones(40)
    exact = sp.linalg.spsolve(sp.csc_matrix(matrix), rhs)
    repaired, moves = refinement.repair_direct_solution(matrix, rhs, exact * 1.05)
    assert repaired.shape == (40,)
    assert moves
    np.testing.assert_allclose(repaired, exact, rtol=1e-9, atol=1e-12)

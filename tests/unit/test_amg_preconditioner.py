"""Aggregation multigrid is an actual coarse correction, not a renamed smoother."""

import numpy as np
import pytest
import scipy.sparse as sp

from motronics.__main__ import build_parser
from motronics.cli.legs import _resolve_orchestration_specs
from motronics.gauss.linear import make_preconditioner, solve_linear_system


def poisson(side):
    line = sp.diags([-np.ones(side - 1), 2 * np.ones(side), -np.ones(side - 1)], [-1, 0, 1])
    return (sp.kron(sp.eye(side), line) + sp.kron(line, sp.eye(side))).tocsr()


def test_amg_cg_recovers_the_same_field_as_direct():
    operator = poisson(12)
    rhs = np.ones(operator.shape[0])
    actual = solve_linear_system(operator, rhs, linear_solver='cg', preconditioner='amg')
    expected = solve_linear_system(operator, rhs)
    np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-8)


def test_amg_is_symmetric_positive_and_reduces_true_residual():
    operator = poisson(10)
    preconditioner = make_preconditioner('amg', operator)
    left = np.arange(operator.shape[0], dtype=float)
    right = np.cos(left)
    np.testing.assert_allclose(left @ (preconditioner @ right), right @ (preconditioner @ left), rtol=1e-12)
    assert left @ (preconditioner @ left) > 0
    rhs = np.ones(operator.shape[0])
    assert np.linalg.norm(rhs - operator @ (preconditioner @ rhs)) < np.linalg.norm(rhs)
    assert len(preconditioner.level_sizes) > 1
    assert all(a > b for a, b in zip(preconditioner.level_sizes, preconditioner.level_sizes[1:], strict=False))


def test_amg_refuses_unsupported_operator_types():
    for operator in (sp.csr_matrix([[2.0, 1.0], [0.0, 2.0]]), sp.diags([1.0, -1.0]), sp.eye(2, dtype=complex)):
        with pytest.raises(ValueError, match='AMG'):
            make_preconditioner('amg', operator)


def test_amg_is_reachable_from_the_production_cli_registry():
    args = build_parser().parse_args(['case', 'run', 'raw.FCStd', '--select', 'fea,cg,amg'])
    (spec,) = _resolve_orchestration_specs(args)
    assert spec.linear_solver == 'cg'
    assert spec.preconditioner == 'amg'


def test_amg_operator_accepts_multiple_right_hand_sides():
    operator = poisson(8)
    preconditioner = make_preconditioner('amg', operator)
    rhs = np.ones((operator.shape[0], 2))
    np.testing.assert_allclose((preconditioner @ rhs)[:, 0], preconditioner @ rhs[:, 0])

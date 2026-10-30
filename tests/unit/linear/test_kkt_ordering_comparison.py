"""Bounded ordering characterization, not an application fit or production default."""

import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
import scipy
import scipy.sparse as sp
from tests.unit.linear.test_rate_factor_retires_before_all_row_retry import (
    test_violated_original_rows_retire_square_lu_without_changing_numerics as _lifetime_control,
)
from tests.unit.test_a_field_size_overdetermined_rate_is_solved_in_least_squares import _system

from numerics_lab.linear import initial_rate

ORDERS = (None, 'MMD_AT_PLUS_A')


def _matrix_digest(matrix):
    digest = hashlib.sha256()
    for array in (matrix.data, matrix.indices, matrix.indptr):
        digest.update(array.tobytes())
    return digest.hexdigest()


def _run_order(monkeypatch, call, order):
    assert order in ORDERS
    original_factor, original_kkt = initial_rate.splu, initial_rate._least_squares_rate_values
    record = {'ordering': order or 'default_COLAMD', 'kkt_calls': 0, 'solve_calls': 0}
    inside = False

    class ForwardingLU:
        def __init__(self, factor) -> None:
            self.factor = factor

        def solve(self, rhs):
            record['solve_calls'] += 1
            return self.factor.solve(rhs)

    def factor(operator, *args, **kwargs):
        if not inside:
            return original_factor(operator, *args, **kwargs)
        assert not args and not kwargs, 'original KKT options changed'
        assert operator.shape[0] <= 1400 and operator.nnz < 20000, 'bounded synthetic factors only'
        options = {} if order is None else {'permc_spec': order}
        record.update(
            kkt_sha256=_matrix_digest(operator),
            shape=list(operator.shape),
            nnz=operator.nnz,
            matrix_array_bytes=sum(array.nbytes for array in (operator.data, operator.indices, operator.indptr)),
            factor_options=options,
        )
        result = original_factor(operator, **options)
        assert _matrix_digest(operator) == record['kkt_sha256'], 'factorization changed original coefficients'
        # Public L/U extraction is bounded HERE only, never added to an application observer.
        lower, upper = result.L, result.U
        record['factor_nnz'] = lower.nnz + upper.nnz
        record['public_factor_array_bytes'] = sum(
            array.nbytes for value in (lower, upper) for array in (value.data, value.indices, value.indptr)
        )
        return ForwardingLU(result)

    def kkt(reduced, forcing):
        nonlocal inside
        record['kkt_calls'] += 1
        record['reduced_shape'] = list(reduced.shape)
        record['forcing_sha256'] = hashlib.sha256(forcing.tobytes()).hexdigest()
        inside = True
        try:
            return original_kkt(reduced, forcing)
        finally:
            inside = False

    with monkeypatch.context() as patch:
        patch.setattr(initial_rate, 'splu', factor)
        patch.setattr(initial_rate, '_least_squares_rate_values', kkt)
        try:
            result = call()
        except ValueError as error:
            result = error
            record['exception'] = {'type': type(error).__name__, 'args': error.args}
    return result, record


def _save_evidence(name, records):
    root = Path(__file__).resolve().parents[3]
    directory = root / 'output/logs' / datetime.now(UTC).strftime('%y/%m/%d') / 'kkt-ordering' / str(os.getpid())
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / f'{name}.json').open('x', encoding='utf-8') as stream:
        json.dump(
            {
                'scope': 'bounded synthetic equations only; NOT application fill or memory fit',
                'memory_scope': 'public CSC array storage only; whole-test peak belongs to the verify Box receipt',
                'production_sha256': hashlib.sha256(Path(initial_rate.__file__).read_bytes()).hexdigest(),
                'test_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'numpy': np.__version__,
                'scipy': scipy.__version__,
                'records': records,
            },
            stream,
            indent=2,
            allow_nan=False,
        )


@pytest.mark.parametrize('coupled', [False, True])
def test_sparse_grid_equations_survive_symmetric_pattern_ordering(monkeypatch, coupled):
    width = 26
    line = sp.diags((-np.ones(width - 1), 2 * np.ones(width), -np.ones(width - 1)), (-1, 0, 1))
    grid = sp.kronsum(line, line, format='csr')
    head = sp.csr_matrix([[1.0, 1.0], [1.0, 1.0 + 2 * np.finfo(float).eps], [1.0, -1.0]])
    matrix = sp.block_diag((head, grid), format='lil')
    if coupled:
        matrix[0, 2], matrix[1, 2], matrix[2, -1] = 0.25, 0.25, 1e-30
    matrix = matrix.tocsr()
    weak = sp.csr_matrix(([1e-30], ([0], [matrix.shape[1] - 1])), shape=(1, matrix.shape[1]))
    matrix = sp.vstack((matrix, weak), format='csr')
    expected = np.linspace(0.25, 1.25, matrix.shape[1])
    rhs = matrix @ expected
    reduced, forcing = initial_rate.normalize_sparse_rows(matrix, rhs)
    outputs, evidence = [], []
    for order in ORDERS:
        result, record = _run_order(
            monkeypatch, lambda: initial_rate._least_squares_rate_values(reduced, forcing), order
        )
        assert not isinstance(result, Exception)
        np.testing.assert_allclose(result, expected, rtol=0, atol=1e-10)
        bars = 1e-10 * (abs(matrix) @ abs(result) + abs(rhs))
        assert np.all(abs(matrix @ result - rhs) <= bars)
        assert record['kkt_calls'] == 1 and record['solve_calls'] == 5
        outputs.append(result)
        evidence.append(record)
    np.testing.assert_allclose(outputs[0], outputs[1], rtol=0, atol=1e-10)
    assert evidence[0]['kkt_sha256'] == evidence[1]['kkt_sha256']
    assert evidence[0]['forcing_sha256'] == evidence[1]['forcing_sha256']
    assert evidence[0]['factor_options'] == {} and evidence[1]['factor_options'] == {'permc_spec': 'MMD_AT_PLUS_A'}
    _save_evidence(f'grid-coupled-{coupled}', evidence)


@pytest.mark.parametrize('unavailable', [False, True])
@pytest.mark.parametrize(('contradiction', 'allowance'), [(False, 0.0), (True, 0.0), (True, 2.0)])
def test_real_public_retry_keeps_all_rows_exceptions_and_uniqueness(monkeypatch, unavailable, contradiction, allowance):
    matrix, rhs, expected = _system(contradict=contradiction)
    if unavailable:
        matrix = sp.hstack((matrix, sp.csr_matrix((matrix.shape[0], 1))), format='csr')
        expected = np.r_[expected, 0.0]
    bounds = np.full(len(rhs), allowance)
    outputs, evidence = [], []
    for order in ORDERS:
        result, record = _run_order(
            monkeypatch,
            lambda: initial_rate.sparse_initial_rate(matrix, rhs, residual_allowance=bounds),
            order,
        )
        assert record['kkt_calls'] == 1 and record['solve_calls'] == 5
        if contradiction and not allowance:
            assert isinstance(result, ValueError)
            assert result.args == ('actual rate equations contradict original physical constraints',)
        else:
            rate, unique = result
            bars = 1e-10 * (abs(matrix) @ abs(rate) + abs(rhs)) + bounds
            assert np.all(abs(matrix @ rate - rhs) <= bars)
            np.testing.assert_array_equal(
                unique, np.r_[np.ones(600, dtype=bool), np.zeros(int(unavailable), dtype=bool)]
            )
            if not contradiction:
                np.testing.assert_allclose(rate, expected, rtol=0, atol=1e-12)
        outputs.append(result)
        evidence.append(record)
    if not (contradiction and not allowance):
        np.testing.assert_allclose(outputs[0][0], outputs[1][0], rtol=0, atol=1e-12)
        np.testing.assert_array_equal(outputs[0][1], outputs[1][1])
    assert evidence[0]['kkt_sha256'] == evidence[1]['kkt_sha256']
    assert evidence[0]['forcing_sha256'] == evidence[1]['forcing_sha256']
    _save_evidence(f'public-{unavailable}-{contradiction}-{allowance}', evidence)


def test_rank_deficient_kkt_still_refuses_with_unchanged_pivoting(monkeypatch):
    matrix = sp.block_diag((sp.csr_matrix([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]), sp.eye(598)), format='csr')
    reduced, forcing = initial_rate.normalize_sparse_rows(matrix, np.ones(matrix.shape[0]))
    evidence = []
    for order in ORDERS:
        result, record = _run_order(
            monkeypatch, lambda: initial_rate._least_squares_rate_values(reduced, forcing), order
        )
        assert isinstance(result, ValueError)
        assert result.args == ('actual field-size rate equations lack full column rank',)
        assert record['kkt_calls'] == 1 and record['solve_calls'] == 0
        evidence.append(record)
    _save_evidence('rank-refusal', evidence)


@pytest.mark.parametrize('unavailable', [False, True])
@pytest.mark.parametrize(('contradiction', 'allowance'), [(0.0, 0.0), (1e-3, 0.0), (1e-3, 1e-2)])
def test_original_violated_row_lifetime_control_is_unchanged(monkeypatch, unavailable, contradiction, allowance):
    _lifetime_control(monkeypatch, unavailable, contradiction, allowance)

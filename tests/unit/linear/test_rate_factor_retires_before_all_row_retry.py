"""A real square factor may precondition rows but retires before any KKT fallback."""

import weakref
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

from numerics_lab.linear import initial_rate


@pytest.mark.parametrize('unavailable', [False, True])
@pytest.mark.parametrize(('contradiction', 'allowance'), [(0.0, 0.0), (1e-3, 0.0), (1e-3, 1e-2)])
def test_violated_original_rows_retire_square_lu_without_changing_numerics(
    monkeypatch, unavailable, contradiction, allowance
):
    assert Path(initial_rate.__file__).resolve() == (
        Path(__file__).resolve().parents[3] / 'python/numerics_lab/linear/initial_rate.py'
    )
    # The square matching REALLY factors, but its rounded RHS loses the second
    # coordinate. The independent third row detects that: no planted solve output.
    size = 600
    head = sp.csr_matrix([[1.0, 1.0], [1.0, 1.0 + 2 * np.finfo(float).eps], [1.0, -1.0]])
    matrix = sp.block_diag((head, sp.eye(size - 2)), format='csr')
    expected = np.r_[0.75, 0.25, np.linspace(-1.0, 1.0, size - 2)]
    rhs = matrix @ expected
    rhs[1] += contradiction
    if unavailable:
        matrix = sp.hstack((matrix, sp.csr_matrix((size + 1, 1))), format='csr')
        expected = np.r_[expected, 0.0]
    bounds = np.full(size + 1, allowance)
    original_factor, original_refine = initial_rate.splu, initial_rate._refined_rate_values

    def run(*, require_retired):
        refs, shapes, solves, refined = [], [], [], []

        class ForwardingLU:
            def __init__(self, factor, role) -> None:
                self.factor, self.role = factor, role

            def solve(self, forcing, **kwargs):
                result = self.factor.solve(forcing, **kwargs)
                solves.append((self.role, forcing.copy(), result.copy(), kwargs.copy()))
                return result

        def factor(operator):
            if shapes and require_retired:
                assert refs[0]() is None, 'retired square LU still owned at all-row KKT entry'
            shapes.append(operator.shape)
            proxy = ForwardingLU(original_factor(operator), len(shapes))
            refs.append(weakref.ref(proxy))  # Never retain a strong spy owner or force collection.
            return proxy

        def refine(*args):
            result = original_refine(*args)
            refined.append(result.copy())  # No factor/argument tuple escapes this forwarding call.
            return result

        with monkeypatch.context() as patch:
            patch.setattr(initial_rate, 'splu', factor)
            patch.setattr(initial_rate, '_refined_rate_values', refine)
            try:
                result = initial_rate.sparse_initial_rate(matrix, rhs, residual_allowance=bounds)
            except ValueError as error:
                result = error
        assert shapes[0] == (size, size) and len(shapes) <= 2
        if len(shapes) == 2:
            assert shapes[1] == (2 * size + 1, 2 * size + 1)
        if contradiction and not allowance:
            assert len(shapes) == 2, 'inconsistent rows still require the unchanged augmented fallback'
        assert len(refined) == 1
        first = np.r_[refined[0], 0.0] if unavailable else refined[0]
        original_bar = 1e-10 * (abs(matrix) @ abs(first) + abs(rhs)) + bounds
        assert np.any(abs(matrix @ first - rhs) > original_bar), 'must exercise actual violated-row retry'
        assert all(reference() is None for reference in refs)
        return result, solves, refined

    before, before_solves, before_refined = run(require_retired=False)
    after, after_solves, after_refined = run(require_retired=True)
    assert len(before_solves) == len(after_solves)
    for (role, forcing, value, options), (other_role, other_forcing, other_value, other_options) in zip(
        before_solves, after_solves, strict=True
    ):
        assert role == other_role and options == other_options
        np.testing.assert_array_equal(forcing, other_forcing)
        np.testing.assert_array_equal(value, other_value)
    np.testing.assert_array_equal(before_refined, after_refined)
    if contradiction and not allowance:
        assert isinstance(before, ValueError) and isinstance(after, ValueError)
        assert before.args == after.args == ('actual rate equations contradict original physical constraints',)
    else:
        assert not isinstance(before, Exception) and not isinstance(after, Exception)
        np.testing.assert_array_equal(after[0], before[0])
        np.testing.assert_array_equal(after[1], before[1])
        np.testing.assert_array_equal(
            after[1], np.r_[np.ones(size, dtype=bool), np.zeros(int(unavailable), dtype=bool)]
        )
        bar = 1e-10 * (abs(matrix) @ abs(after[0]) + abs(rhs)) + bounds
        assert np.all(abs(matrix @ after[0] - rhs) <= bar)
        if not contradiction:
            np.testing.assert_allclose(after[0], expected, rtol=0, atol=1e-12)

"""SciPy's PCHIP / piecewise polynomial and NNLS, frozen as the goldens the Rust ports judge themselves by.

Moved from motronics-studio's ``core_num`` (``bh::pchip_slopes``, ``ppoly::PiecewisePoly``) on 2026-11-19, with
``nnls`` new beside them. The reference is SciPy itself, not a Python twin: ``PchipInterpolator`` and
``PPoly.antiderivative`` for ``crates/numerics/tests/golden_interp.rs``, ``scipy.optimize.nnls`` for
``crates/numerics/tests/golden_nnls.rs``. This file asserts the live SciPy still produces both goldens and that each
golden holds the fixtures its Rust test needs to say no.

Regenerate deliberately: ``python -c "from tests.unit.test_interp_and_nnls_port_golden import write; write()"``.
"""

from __future__ import annotations

import itertools
import json

import numpy as np
import pytest
import scipy
from scipy.interpolate import PchipInterpolator, PPoly, make_interp_spline
from scipy.optimize import nnls
from tests.port_goldens import GOLDENS, load_golden, register, write_golden

_INTERP = GOLDENS / 'interp.jsonl'
_NNLS = GOLDENS / 'nnls.jsonl'

#: ``(name, x, y)`` -- every branch of the slope rule reached by at least one table.
_TABLES = (
    ('nonuniform_convex', [0.0, 0.2, 0.5, 0.9, 1.4, 1.6, 1.9], [0.0, 40.0, 120.0, 300.0, 1200.0, 3000.0, 12000.0]),
    ('uniform', [0.0, 0.25, 0.5, 0.75], [0.0, 10.0, 45.0, 200.0]),
    ('parabola_ends', [0.0, 0.3, 1.0], [1.0, 1.87, 6.0]),
    ('edge_sign_clamp', [0.0, 1.0, 1.1], [0.0, 1.0, 100.0]),
    ('edge_three_secant_cap', [0.0, 1.0, 2.0], [0.0, 1.0, -20.0]),
    ('flat_interior', [0.0, 1.0, 2.0, 3.0], [0.0, 5.0, 5.0, 9.0]),
    ('local_maximum', [0.0, 1.0, 2.0], [0.0, 5.0, 1.0]),
    ('two_knots', [0.5, 1.5], [100.0, 400.0]),
    ('decreasing_nonuniform', [-2.0, -1.5, 0.0, 0.1, 3.0], [9.0, 4.0, 3.5, -1.0, -7.0]),
    (
        'steep_saturating',
        [0.0, 0.05, 0.1, 0.2, 0.4, 0.8, 1.2, 1.5, 1.7, 1.9, 2.0],
        [0.0, 17.06, 26.9, 36.82, 49.6, 80.68, 193.67, 1800.49, 6000.0, 30000.0, 90000.0],
    ),
)


def _queries(x: list[float]) -> list[float]:
    """Knots, midpoints, quarter points and both outside ends (the extrapolated end cubics)."""
    lo, hi = x[0], x[-1]
    span = hi - lo
    q = list(x)
    for a, b in itertools.pairwise(x):
        q += [a + 0.25 * (b - a), 0.5 * (a + b), a + 0.8 * (b - a)]
    q += [lo - 0.3 * span, hi + 0.3 * span]
    return sorted(q)


def _poly_entry(p: PPoly, q: list[float]) -> dict:
    anti = p.antiderivative()
    return {
        'c': [[float(v) for v in col] for col in p.c.T],
        'eval': {str(nu): [float(v) for v in p(q, nu)] for nu in range(4)},
        'antiderivative': [float(v) for v in anti(q)],
    }


def _build_interp() -> dict:
    cases = []
    for name, x, y in _TABLES:
        xa, ya, q = np.asarray(x), np.asarray(y), _queries(x)
        pchip = PchipInterpolator(xa, ya)
        linear = PPoly.from_spline(make_interp_spline(xa, ya, k=1))
        # from_spline repeats the end knots; keep the intervals over the declared breakpoints only.
        keep = [i for i in range(len(linear.x) - 1) if linear.x[i] < linear.x[i + 1]]
        linear = PPoly(linear.c[:, keep], np.concatenate([linear.x[keep], linear.x[keep[-1] + 1 : keep[-1] + 2]]))
        cases.append({
            'name': name,
            'x': x,
            'y': y,
            'query': q,
            'slopes': [float(v) for v in pchip(xa, 1)],
            'pchip': _poly_entry(pchip, q),
            'linear': _poly_entry(linear, q),
        })
    return {'note': f'scipy {scipy.__version__}: PchipInterpolator, PPoly.antiderivative, k=1 spline', 'cases': cases}


def _nnls_problems() -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Deterministic problems (integer recurrences, no RNG): bound-active, interior, wide, degenerate."""
    tall = np.array([[((3 * i + 7 * j) % 11) - 4.0 + 0.5 * (i == j) for j in range(5)] for i in range(8)])
    wide = np.array([[((5 * i + 2 * j) % 7) - 2.0 for j in range(6)] for i in range(3)])
    hilbert = np.array([[1.0 / (i + j + 1) for j in range(5)] for i in range(5)])
    dup = tall[:, [2, 2, 3, 4, 0]].copy()
    zero_col = tall.copy()
    zero_col[:, 2] = 0.0
    interior = np.eye(4) + 0.1 * np.ones((4, 4))
    return [
        ('scipy_doc', np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]), np.array([2.0, 1.0, 1.0])),
        ('all_bounds_active', np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]), np.array([-1.0, -1.0, -1.0])),
        ('tall_mixed', tall, np.array([3.0, -1.0, 2.0, 0.5, -2.0, 4.0, 1.0, -0.5])),
        ('wide', wide, np.array([1.0, -2.0, 0.5])),
        ('hilbert', hilbert, np.array([1.0, -0.5, 0.25, 0.1, -0.2])),
        ('duplicate_columns', dup, np.array([3.0, -1.0, 2.0, 0.5, -2.0, 4.0, 1.0, -0.5])),
        ('zero_column', zero_col, np.array([3.0, -1.0, 2.0, 0.5, -2.0, 4.0, 1.0, -0.5])),
        ('interior', interior, np.array([1.0, 2.0, 3.0, 4.0])),
    ]


def _build_nnls() -> dict:
    cases = []
    for name, a, b in _nnls_problems():
        x, rnorm = nnls(a, b)
        cases.append({
            'name': name,
            'm': a.shape[0],
            'n': a.shape[1],
            'a': [float(v) for v in a.ravel()],
            'b': [float(v) for v in b],
            'x': [float(v) for v in x],
            'rnorm': float(rnorm),
        })
    return {'note': f'scipy {scipy.__version__}: scipy.optimize.nnls, A row-major', 'cases': cases}


register(_INTERP, _build_interp)
register(_NNLS, _build_nnls)


def write() -> None:
    """Rewrite both goldens from the live SciPy (a deliberate act; commit the result)."""
    for path, build in ((_INTERP, _build_interp), (_NNLS, _build_nnls)):
        write_golden(path, build())


def _same(got: object, want: object, where: str) -> None:
    if isinstance(want, dict):
        assert isinstance(got, dict)
        assert set(got) == set(want), where
        for k in want:
            _same(got[k], want[k], f'{where}.{k}')
    elif isinstance(want, list) and want and isinstance(want[0], (list, dict)):
        assert isinstance(got, list)
        assert len(got) == len(want), where
        for i, (g, w) in enumerate(zip(got, want, strict=True)):
            _same(g, w, f'{where}[{i}]')
    elif isinstance(want, list):
        np.testing.assert_allclose(got, want, rtol=1e-13, atol=1e-13, err_msg=where)  # the table's own unit
    elif isinstance(want, float):
        assert got == pytest.approx(want, rel=1e-13, abs=1e-13), where  # the table's own unit
    elif where.endswith('.note'):
        return
    else:
        assert got == want, where


@pytest.mark.unit
@pytest.mark.parametrize(('path', 'build'), [(_INTERP, _build_interp), (_NNLS, _build_nnls)], ids=['interp', 'nnls'])
def test_the_live_scipy_still_produces_the_golden(path, build) -> None:
    if not path.is_file():
        pytest.fail(f'{path} is a committed artifact; regenerate it deliberately with write() and commit it')
    _same(build(), load_golden(path), path.name)


@pytest.mark.unit
def test_the_comparison_sees_a_moved_entry() -> None:
    """Planted control: the freeze comparison reds on one moved coefficient."""
    golden = load_golden(_INTERP)
    moved = json.loads(json.dumps(golden))
    moved['cases'][0]['pchip']['c'][1][0] *= 1 + 1e-9
    with pytest.raises(AssertionError):
        _same(moved, golden, 'interp')


@pytest.mark.unit
def test_the_nnls_golden_holds_an_active_bound_an_interior_solution_and_a_degenerate_column() -> None:
    """Floors: a golden with no active bound, or no fully positive solution, cannot fail the active-set logic."""
    cases = {c['name']: c for c in load_golden(_NNLS)['cases']}
    assert any(0.0 in c['x'] and any(v > 0 for v in c['x']) for c in cases.values())
    assert all(v > 0 for v in cases['interior']['x'])
    assert cases['zero_column']['x'][2] == 0.0
    assert cases['duplicate_columns']['x'][0] + cases['duplicate_columns']['x'][1] > 0.0

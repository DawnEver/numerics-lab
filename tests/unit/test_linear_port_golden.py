"""The linear-solve port's golden vectors — one file, read by BOTH languages.

Companion to `tests/unit/test_transforms_port_golden.py`; read that module for why the
comparison goes through a committed data file rather than across the FFI, and why the Python half
of the pair IS the migration plan's freeze protocol made executable.

## What this golden can and cannot promise

`python/numerics_lab/linear/__init__.py` documents its `direct` backend as bit-for-bit
`scipy.sparse.linalg.spsolve` (SuperLU). **No Rust LU can inherit that promise** — bit-identity is
a property of an implementation, not of the mathematics — so the Rust `direct` (faer's sparse LU)
is a DIFFERENT backend that is CHARACTERIZED against this file rather than equated to it. What is
frozen here is therefore the SOLUTION and the residual it achieves, and the Rust side judges
itself at a residual-based tolerance stated in `crates/numerics/tests/golden_linear.rs`.

## The floors this file holds

- every registered backend and every preconditioner appears at least once;
- at least one fixture is REFUSED by the singular guard. A fixture set that only ever succeeds
  cannot test the guard that exists to fail, and that guard is the reason this seam exists.

Regenerate deliberately with `.venv/Scripts/python.exe -m tests.port_goldens`.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
from tests.port_goldens import GOLDENS, load_golden, register

from numerics_lab.linear import (
    LINEAR_SOLVER_REGISTRY,
    PRECONDITIONER_REGISTRY,
    LinearSolveError,
    apply_dirichlet_triplets,
    requires_symmetric,
    solve_linear_system,
)

_GOLDEN = GOLDENS / 'linear.json'

#: The backends both sides register.
_SOLVERS = ('direct', 'cg', 'minres', 'gmres', 'bicgstab', 'qmr')
_PRECONDITIONERS = ('jacobi', 'ilu', 'ssor')


def _spd(n=24):
    """SPD tridiagonal: a 1-D Laplacian plus a varying diagonal.

    The jitter is a fixed integer recurrence, not an RNG draw. A fixture that differs between
    runs cannot anchor a frozen solution, and `default_rng(seed)` ties the file to a numpy
    version's generator rather than to the data.
    """
    main = np.array([2.0 + 0.5 + ((i * 37) % 11) / 10.0 for i in range(n)])
    off = -np.ones(n - 1)
    return sp.diags([off, main, off], offsets=[-1, 0, 1], format='csr')


def _nonsymmetric(n=24):
    """Non-symmetric tridiagonal — the fixture cg/minres are NOT valid on."""
    main = np.array([3.0 + (i % 5) / 4.0 for i in range(n)])
    return sp.diags([-np.ones(n - 1), main, -0.5 * np.ones(n - 1)], offsets=[-1, 0, 1], format='csr')


def _dirichlet_system(n=12):
    """An SPD system with two Dirichlet rows pinned through `apply_dirichlet_triplets`.

    This is the fixture that exercises the pinning path end to end: the triplets go through the
    real function, the pinned rhs is what the solve is handed, and the frozen solution therefore
    depends on the pinning being right.
    """
    a = sp.coo_matrix(_spd(n))
    rhs = np.array([((i % 7) - 3.0) / 2.0 for i in range(n)])
    row, col, val = apply_dirichlet_triplets(a.row, a.col, a.data, rhs, {1.5: [0], -0.5: [n - 1]}, 0.25)
    pinned = sp.coo_matrix((val, (row, col)), shape=(n, n)).tocsr()
    return pinned, rhs, [0, n - 1]


def _singular():
    """Two identical rows — rank-deficient. The direct guard MUST refuse this."""
    return sp.csr_matrix(np.array([[2.0, 1.0, 1.0], [2.0, 1.0, 1.0], [1.0, 1.0, 3.0]])), np.array([1.0, 2.0, 1.0])


def _zero_row():
    """A structurally singular system — an entirely zero row."""
    return sp.csr_matrix(np.array([[1.0, 0.0], [0.0, 0.0]])), np.array([1.0, 1.0])


def _triplets(a) -> dict:
    coo = sp.coo_matrix(a)
    return {
        'n': int(coo.shape[0]),
        'rows': [int(v) for v in coo.row],
        'cols': [int(v) for v in coo.col],
        'values': [float(v) for v in coo.data],
    }


def _entry(a, b, *, refused: bool, pinned: list[int] | None = None) -> dict:
    """One fixture: its matrix, its rhs, and either the frozen solutions or the refusal."""
    entry: dict = {
        'matrix': _triplets(a),
        'rhs': [float(v) for v in b],
        'symmetric': bool((abs(a - a.T) > 1e-12).nnz == 0),
        'refused_by_the_singular_guard': refused,
        'solutions': {},
        'residuals': {},
    }
    if pinned is not None:
        entry['pinned_indices'] = pinned
    if refused:
        # The refusal itself is the frozen datum. A fixture whose only content was a solution
        # could not freeze a guard.
        with pytest.raises(LinearSolveError):
            solve_linear_system(a, b, linear_solver='direct')
        return entry
    for name in _SOLVERS:
        if requires_symmetric(name) and not entry['symmetric']:
            continue  # the tag is a declaration; running the solver anyway would freeze garbage
        x = solve_linear_system(a, b, linear_solver=name)
        entry['solutions'][name] = [float(v) for v in x]
        entry['residuals'][name] = float(np.linalg.norm(a @ x - b) / np.linalg.norm(b))
    for precond in _PRECONDITIONERS:
        x = solve_linear_system(a, b, linear_solver='cg' if entry['symmetric'] else 'gmres', preconditioner=precond)
        entry['solutions'][f'precond:{precond}'] = [float(v) for v in x]
        entry['residuals'][f'precond:{precond}'] = float(np.linalg.norm(a @ x - b) / np.linalg.norm(b))
    return entry


def _build() -> dict:
    spd = _spd()
    ns = _nonsymmetric()
    pinned_a, pinned_b, pinned_idx = _dirichlet_system()
    sing_a, sing_b = _singular()
    zero_a, zero_b = _zero_row()
    return {
        'note': (
            'Generated from python/numerics_lab/linear/. Read by BOTH '
            'tests/unit/test_linear_port_golden.py and '
            'crates/numerics/tests/golden_linear.rs. The Rust `direct` is faer, NOT spsolve: '
            'it is characterized against these solutions at a residual-based tolerance, never '
            'equated to them bit-for-bit.'
        ),
        'registry': {'solvers': list(_SOLVERS), 'preconditioners': list(_PRECONDITIONERS)},
        'requires_symmetric': {name: bool(requires_symmetric(name)) for name in _SOLVERS},
        'cases': {
            'spd': _entry(spd, np.array([((i % 7) - 3.0) / 2.0 for i in range(24)]), refused=False),
            'nonsymmetric': _entry(ns, np.array([1.0 + (i % 3) for i in range(24)]), refused=False),
            'dirichlet_pinned': _entry(pinned_a, pinned_b, refused=False, pinned=pinned_idx),
            'singular_duplicate_row': _entry(sing_a, sing_b, refused=True),
            'singular_zero_row': _entry(zero_a, zero_b, refused=True),
        },
    }


def _load() -> dict:
    if not _GOLDEN.is_file():
        pytest.fail(
            f'{_GOLDEN} is missing. It is a committed artifact, not a cache -- regenerate it '
            f'deliberately with `python -m tests.port_goldens` and commit the result.'
        )
    return load_golden(_GOLDEN)


@pytest.mark.unit
def test_the_live_python_still_produces_the_frozen_solutions() -> None:
    """The freeze, enforced: a behaviour change in `gauss/linear/` reds HERE, not in silence."""
    golden = _load()
    live = _build()
    assert set(live['cases']) == set(golden['cases']), 'the fixture set moved'
    for name, want in golden['cases'].items():
        got = live['cases'][name]
        assert got['refused_by_the_singular_guard'] == want['refused_by_the_singular_guard'], name
        assert set(got['solutions']) == set(want['solutions']), f'{name}: the backend set moved'
        for backend, values in want['solutions'].items():
            np.testing.assert_allclose(
                got['solutions'][backend],
                np.asarray(values, dtype=float),
                rtol=1e-12,
                atol=1e-14,
                err_msg=f'{name}.{backend}',
            )


@pytest.mark.unit
def test_the_golden_covers_every_backend_and_every_preconditioner() -> None:
    """A golden narrower than the thing it freezes is a scan with no floor."""
    golden = _load()
    covered: set[str] = set()
    for case in golden['cases'].values():
        covered |= set(case['solutions'])
    missing = set(_SOLVERS) - covered
    assert not missing, f'no fixture exercises {sorted(missing)}'
    for precond in _PRECONDITIONERS:
        assert f'precond:{precond}' in covered, f'no fixture exercises the {precond} preconditioner'
    assert golden['registry']['solvers'] == list(_SOLVERS), 'the registry declaration drifted'


@pytest.mark.unit
def test_at_least_one_fixture_is_refused_by_the_singular_guard() -> None:
    """The floor under the fixtures themselves.

    A fixture set that only ever succeeds cannot test the guard that exists to FAIL, and that
    guard -- the one that caught finite-but-meaningless SuperLU output shipping as ``status=ok``
    -- is the reason this seam is worth porting carefully.
    """
    golden = _load()
    refused = [n for n, c in golden['cases'].items() if c['refused_by_the_singular_guard']]
    assert refused, 'every fixture succeeds, so the singular guard is never exercised'
    for name in refused:
        assert not golden['cases'][name]['solutions'], f'{name} is refused AND carries a solution'


@pytest.mark.unit
def test_the_symmetry_tag_is_frozen_alongside_the_solutions() -> None:
    """`requires_symmetric` is a DECLARATION the Rust side mirrors; freeze it, not just the numbers."""
    golden = _load()
    assert golden['requires_symmetric'] == {name: name in ('cg', 'minres') for name in _SOLVERS}
    # The committed equivalence corpus remains immutable; the new Python
    # preconditioners have dedicated IC0/AMG mathematical-policy tests.
    assert set(PRECONDITIONER_REGISTRY) == set(_PRECONDITIONERS) | {'ic0', 'amg'}
    assert requires_symmetric('iccg')
    assert not requires_symmetric('auto')
    assert set(_SOLVERS) <= set(LINEAR_SOLVER_REGISTRY)


@pytest.mark.unit
def test_the_non_symmetric_fixture_is_actually_non_symmetric() -> None:
    """Without this, `cg` could be skipped on a fixture that was symmetric all along."""
    golden = _load()
    assert not golden['cases']['nonsymmetric']['symmetric']
    assert golden['cases']['spd']['symmetric']
    assert 'cg' not in golden['cases']['nonsymmetric']['solutions']


register(_GOLDEN, _build)

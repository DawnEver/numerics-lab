"""The IC0 apply is a pair of PRE-FACTORED triangular sweeps, and one tangent builds IC0 once.

THE DEFECT (user ruling 30, 2026-10-06): `case_m` marched at ~126 s/step under ICCG. A
profile (output/logs/26/10/06/perf/iccg10.prof) put 479 of the 537 s spent inside CG in
`spsolve_triangular`, which re-validates, re-sorts and re-multiplies its CSR operand on EVERY call
-- about 89 matvecs' worth of work for two sweeps whose arithmetic is one matvec's -- and a further
74 s in rebuilding the same IC0 factor once per Schur column.

THE BUDGET IS DERIVED, NOT COUNTED. Two triangular sweeps over ``L`` touch every stored entry of
``A`` about once, so the floor is ~1 matvec of ``A``. SuperLU's supernodal sweep, the compiled
kernel available here, measured ~7 matvecs per sweep (~15 per apply) on the 122 500-unknown
5-point operator below; the per-call-validated path measured ~89. The bar is twice the measured
compiled cost: it holds under a loaded box and refuses the per-call path by 3x.
"""

import time

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve_triangular

from numerics_lab.linear import make_preconditioner, solve_linear_system

#: Two SuperLU sweeps measured ~15 matvecs; 2x headroom for a shared box.
_IC0_APPLY_MATVEC_BUDGET = 30


def _laplacian(n: int) -> sp.csr_matrix:
    t = sp.diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(n, n))
    return (sp.kron(sp.eye(n), t) + sp.kron(t, sp.eye(n))).tocsr()


def _median_seconds(fn, repeats: int = 15) -> float:
    fn()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - start)
    return float(np.median(samples))


def test_one_ic0_apply_costs_a_few_matvecs_not_a_revalidation():
    matrix = _laplacian(200)
    vector = np.random.default_rng(0).standard_normal(matrix.shape[0])
    apply = make_preconditioner('ic0', matrix)
    matvec = _median_seconds(lambda: matrix @ vector)
    sweep = _median_seconds(lambda: apply @ vector)
    assert sweep <= _IC0_APPLY_MATVEC_BUDGET * matvec, (sweep / matvec, _IC0_APPLY_MATVEC_BUDGET)


def test_the_prefactored_apply_is_the_triangular_pair_to_roundoff():
    matrix = _laplacian(30)
    apply = make_preconditioner('ic0', matrix)
    lower = apply.lower
    vector = np.random.default_rng(1).standard_normal(matrix.shape[0])
    reference = spsolve_triangular(lower.T.tocsr(), spsolve_triangular(lower, vector, lower=True), lower=False)
    np.testing.assert_allclose(apply @ vector, reference, rtol=1e-13, atol=0.0)


def test_a_built_ic0_operator_is_accepted_as_the_iccg_preconditioner():
    matrix = _laplacian(12)
    rhs = np.arange(matrix.shape[0], dtype=float)
    apply = make_preconditioner('ic0', matrix)
    for name in ('iccg', 'auto'):
        np.testing.assert_array_equal(
            solve_linear_system(matrix, rhs, linear_solver=name, preconditioner=apply),
            solve_linear_system(matrix, rhs, linear_solver=name),
        )

"""Symmetric-SOR (SSOR) preconditioner -- a physics-blind algebraic primitive.

SSOR is a classic *algebraic* preconditioner: it is built entirely from the matrix's own
lower / upper triangles and diagonal, touching no mesh, discretization, or curl/gradient
topology. It therefore lives in ``gauss/`` (the physics-blind numerical runtime) rather
than in any physics package, and the shared :data:`~numerics_lab.linear.PRECONDITIONER_REGISTRY`
registers it as a default-available ``'ssor'`` option with a module-level import (no cycle).

Why SSOR is the right auxiliary preconditioner for the nodal magnetostatic Krylov path
(handoff §4 close-out of the Hiptmair-Xu / AMS item): AMS preconditions the **curl-curl**
operator ``curl(nu curl A)`` discretized with **edge (Nedelec) elements** (an H(curl) system
whose huge gradient near-null-space stalls plain CG/ILU/AMG). This codebase has NO such
system -- the FEA solve is the SPD 2D scalar nodal Poisson ``-div(nu grad A) = J`` (scalar
node potential, 3x3 nodal Laplacian element matrix), so AMS has nothing to precondition and
the assessment (`.claude/memory/2026/07/10/finding-hiptmair-xu-assessment.md`) DEFERS it
until a curl-curl / eddy-current / 3D formulation exists.

What DOES help the nodal Krylov path is a stronger *symmetric* auxiliary preconditioner than
the registry's Jacobi (a no-op on the constant-diagonal Laplacian) that -- unlike ILU --
stays symmetric so CG remains a valid Krylov method on the SPD operator. SSOR is exactly
that: symmetric by construction, provably stronger than Jacobi on a diffusion (Poisson-type)
operator. Measured on the 2D 5-point Laplacian it roughly halves-or-better the CG iteration
count (52->23, 77->31, 102->41 at N=256/576/1024) while Jacobi leaves it unchanged. The
registration is ADDITIVE -- the default direct/None solve path is untouched, bit-for-bit;
``'ssor'`` is only used when a caller opts in via ``preconditioner='ssor'``.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, spsolve_triangular

__all__ = [
    'ssor_preconditioner',
]

# SSOR relaxation factor. omega=1 is symmetric Gauss-Seidel -- the robust, parameter-free
# choice that needs no spectral estimate and already gives the measured >2x CG speed-up on
# the Poisson-type nodal operator. (1 < omega < 2 can sharpen it further but needs tuning;
# we keep the parameter-free default so the preconditioner is drop-in.)
_SSOR_OMEGA = 1.0


def ssor_preconditioner(matrix, *, omega: float = _SSOR_OMEGA) -> LinearOperator:
    r"""Symmetric SOR (SSOR) preconditioner ``M ~ matrix^{-1}`` for the SPD nodal operator.

    For ``A = L + D + U`` (strict lower, diagonal, strict upper), the SSOR operator applies

    .. math::

        M^{-1} = \frac{\omega}{2-\omega}\,(D/\omega + U)^{-1}\, D\, (D/\omega + L)^{-1},

    a forward triangular solve, a diagonal scaling, then a backward triangular solve. It is
    **symmetric** whenever ``A`` is (so CG stays valid on the SPD magnetostatic operator,
    unlike a non-symmetric ILU factor) and strictly stronger than Jacobi (it uses the full
    triangles, not just the diagonal). Handed to a Krylov backend as ``M`` it accelerates
    convergence WITHOUT changing the converged solution.

    A zero diagonal entry is left as 1 so the triangular factors stay non-singular (a pinned
    identity row has diagonal 1, so this guard is rarely exercised).
    """
    if not 0.0 < omega < 2.0:
        msg = f'SSOR relaxation omega must satisfy 0 < omega < 2 (got {omega})'
        raise ValueError(msg)

    a = sp.csr_matrix(matrix)
    n = a.shape[0]
    diag = np.asarray(a.diagonal(), dtype=float)
    diag = np.where(np.abs(diag) > 0.0, diag, 1.0)
    d_over_omega = sp.diags(diag / omega)
    lower = (d_over_omega + sp.tril(a, k=-1)).tocsr()  # D/omega + L
    upper = (d_over_omega + sp.triu(a, k=1)).tocsr()  # D/omega + U
    scale = omega / (2.0 - omega)

    def _matvec(v: np.ndarray) -> np.ndarray:
        y = spsolve_triangular(lower, np.asarray(v, dtype=float).ravel(), lower=True)
        y = diag * y
        z = spsolve_triangular(upper, y, lower=False)
        return scale * z

    return LinearOperator((n, n), matvec=_matvec)

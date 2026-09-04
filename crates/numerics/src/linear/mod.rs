//! The pluggable INNER linear-solve seam — the Rust mirror of `src/motronics/gauss/linear/`.
//!
//! One registry of `solve(matrix, rhs) -> x` backends, so a solve can pick a Krylov method
//! without the equation code changing. It touches no physics and no mesh: it only solves
//! `matrix @ x = rhs`.
//!
//! First principle (`AGENTS.md`): IMPLEMENT every method and characterize it. Every backend the
//! Python registers is here, each of which must reach the SAME solution as `direct` within its
//! tolerance — the linear analogue of the nonlinear "converge-to-reference" contract.
//!
//! # Where this port DIVERGES from the Python, deliberately and on the record
//!
//! 1. **`direct` is not `spsolve`.** The Python calls its `direct` backend bit-for-bit
//!    `scipy.sparse.linalg.spsolve` (SuperLU), "so every golden is unchanged". No Rust LU can
//!    inherit that — bit-identity is a property of an implementation. This `direct` is `faer`'s
//!    sparse LU, a DIFFERENT backend, CHARACTERIZED against the Python at a residual-based
//!    tolerance rather than equated to it. See [`direct`] and `tests/golden_linear.rs`.
//! 2. **`ilu` is ILU(0), not scipy's `spilu` (ILUTP).** Different iteration counts, the same
//!    converged solution. See [`precond`].
//! 3. **`pardiso` is not registered.** The Python's entry appears only when the optional
//!    `pypardiso` (Intel MKL) is installed, and it is not installed on this box — so there is no
//!    baseline to characterize against and nothing to port. It is an ABSENCE with a reason, not
//!    an omission. When it arrives it joins as a peer of `direct`, under the same residual guard.
//! 4. **The complex Dirichlet path** is not portable to an `f64` operator; see [`dirichlet`].
//!
//! # The two-sided singular guard
//!
//! A singular direct solve is the defect this seam exists to catch. The BELT is the
//! factorisation's own refusal; the SUSPENDERS are [`DIRECT_RESIDUAL_RTOL`], and it is the
//! suspenders that catch the case that actually shipped garbage — a FINITE but meaningless
//! vector that passes every finiteness check. Both apply to `faer`'s LU exactly as they applied
//! to SuperLU's.

pub mod direct;
pub mod dirichlet;
pub mod krylov;
pub mod precond;

#[cfg(test)]
mod tests;

use contracts::sparse::{Csr, SparseError};

pub use dirichlet::{apply_dirichlet_triplets, pin_rows, DirichletGroup};
pub use krylov::{
    KrylovFailure, KrylovOutcome, Method, GMRES_RESTART, KRYLOV_ATOL, KRYLOV_MAXITER_FACTOR,
    KRYLOV_RTOL,
};
pub use precond::{Preconditioner, PreconditionerError, KNOWN_PRECONDITIONERS, SSOR_OMEGA};

/// Direct-solve singular guard: the largest relative residual a direct factorisation may report.
///
/// Even when the factorisation returns a FINITE vector for an (exactly or numerically) singular
/// system, `||A x - b|| / ||b||` is large. Rejecting above this tolerance is what makes a
/// rank-deficient reluctance network fail honestly instead of shipping garbage as `status=ok`.
/// The bound is loose (1e-6) so a well-conditioned factorisation — residual ~ machine eps — always
/// passes. Verbatim from the Python's `_DIRECT_RESIDUAL_RTOL`.
pub const DIRECT_RESIDUAL_RTOL: f64 = 1e-6;

/// The registry's spellings, in one place so the resolver and its error cannot drift.
///
/// `direct` is the default. `cg`/`minres` assume a symmetric/SPD operator (the magnetostatic
/// secant stiffness is SPD-ish); `gmres`/`bicgstab`/`qmr` handle general operators.
pub const KNOWN_LINEAR_SOLVERS: [&str; 6] = ["bicgstab", "cg", "direct", "gmres", "minres", "qmr"];

/// A linear backend failed to produce a usable solution.
///
/// Raised instead of returning silent NaNs, mirroring the Python's `LinearSolveError` and the MEC
/// singular-system guard: a reference-free network or a Krylov breakdown is a hard failure the
/// caller must SEE.
#[derive(Debug, Clone, PartialEq)]
pub enum LinearSolveError {
    /// An unregistered backend name.
    UnknownSolver {
        /// What was asked for.
        name: String,
    },
    /// An unregistered preconditioner name.
    UnknownPreconditioner(PreconditionerError),
    /// The direct factorisation refused (the BELT half of the singular guard).
    Direct(direct::DirectFailure),
    /// A Krylov method did not converge or broke down.
    Krylov {
        /// Which method.
        method: &'static str,
        /// Why.
        cause: KrylovFailure,
    },
    /// The solution contains a NaN or an infinity.
    NonFinite {
        /// Which backend produced it.
        backend: String,
    },
    /// The SUSPENDERS half: a finite solution that does not satisfy the system.
    ///
    /// This is the one that catches what actually shipped — a rank-deficient factorisation
    /// returning finite-but-meaningless numbers past every finiteness check.
    DirectResidual {
        /// The measured `||A x - b|| / ||b||`.
        residual: f64,
        /// The bound it exceeded.
        bound: f64,
    },
    /// The operator or the vectors do not agree on a shape.
    Shape(SparseError),
}

impl std::fmt::Display for LinearSolveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnknownSolver { name } => write!(
                f,
                "unknown linear solver {name:?}; known: {KNOWN_LINEAR_SOLVERS:?}. ('pardiso' is \
                 NOT registered in the Rust port: the Python entry exists only when the optional \
                 pypardiso is installed, and there is no installation here to characterize \
                 against.)"
            ),
            Self::UnknownPreconditioner(e) => write!(f, "{e}"),
            Self::Direct(e) => write!(
                f,
                "direct solver reports the matrix is (near-)singular ({e:?}); the linear system \
                 is rank-deficient -- is a gauge/reference set, and is the (loop) basis full-rank \
                 at this mesh resolution?"
            ),
            Self::Krylov { method, cause } => {
                write!(f, "Krylov solver {method:?} failed: {cause:?}")
            }
            Self::NonFinite { backend } => write!(
                f,
                "linear solver {backend:?} produced a non-finite solution (singular system or \
                 breakdown); is a Dirichlet boundary set / is the matrix well-conditioned?"
            ),
            Self::DirectResidual { residual, bound } => write!(
                f,
                "direct solve residual ||A x - b|| / ||b|| = {residual:.3e} exceeds {bound:.0e}; \
                 the system is singular / rank-deficient (the factorization returned a finite but \
                 meaningless solution)"
            ),
            Self::Shape(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for LinearSolveError {}

impl From<SparseError> for LinearSolveError {
    fn from(e: SparseError) -> Self {
        Self::Shape(e)
    }
}

/// A resolved backend: the direct factorisation or one of the Krylov peers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinearSolver {
    /// Sparse LU.
    Direct,
    /// One of the Krylov family.
    Krylov(Method),
}

impl LinearSolver {
    /// Resolve a backend name; `None` -> the default `direct`.
    ///
    /// # Errors
    /// [`LinearSolveError::UnknownSolver`].
    pub fn resolve(name: Option<&str>) -> Result<Self, LinearSolveError> {
        match name {
            None | Some("direct") => Ok(Self::Direct),
            Some(other) => Method::from_name(other).map(Self::Krylov).ok_or_else(|| {
                LinearSolveError::UnknownSolver {
                    name: other.to_owned(),
                }
            }),
        }
    }

    /// The registry spelling.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Direct => "direct",
            Self::Krylov(m) => m.name(),
        }
    }

    /// Does this backend assume a symmetric / SPD operator?
    #[must_use]
    pub const fn requires_symmetric(self) -> bool {
        match self {
            Self::Direct => false,
            Self::Krylov(m) => m.requires_symmetric(),
        }
    }
}

/// Whether the named linear backend assumes a symmetric/SPD operator (`cg` / `minres`).
///
/// Mirrors the Python's `requires_symmetric`. An unknown name is `false` rather than an error,
/// matching the Python's `getattr(..., False)` — this is a QUERY about a tag, and the resolver is
/// where an unknown name is refused.
#[must_use]
pub fn requires_symmetric(name: &str) -> bool {
    LinearSolver::resolve(Some(name)).is_ok_and(LinearSolver::requires_symmetric)
}

/// Is this solver being run on an operator it assumes something FALSE about?
///
/// `contracts::sparse::Csr::is_symmetric` says what a matrix IS; [`requires_symmetric`] says what
/// a solver ASSUMES. Neither alone can catch the failure that matters — a symmetric-only method
/// on an asymmetric operator converges quietly to the wrong answer — so the comparison is given a
/// name and a caller instead of living in two docstrings.
#[must_use]
pub fn assumption_violated(solver: LinearSolver, matrix: &Csr, tol: f64) -> bool {
    solver.requires_symmetric() && !matrix.is_symmetric(tol)
}

/// Solve `matrix @ x = rhs` with the named backend, guarding a non-finite or wrong result.
///
/// `linear_solver = None` is the default `direct`. A Krylov backend may take an optional
/// `preconditioner` (`jacobi` / `ilu` / `ssor`) and a warm-start `x0`. The finite-solution guard
/// is uniform across backends, and the direct residual guard rides on top of it.
///
/// The residual guard is applied ONLY to `direct`: the Krylov backends already refuse on their
/// own convergence test, and asserting a second (looser) residual on top of theirs would clash
/// with their stop criterion.
///
/// # Errors
/// [`LinearSolveError`] — never a wrong `x` returned as a success.
pub fn solve_linear_system(
    matrix: &Csr,
    rhs: &[f64],
    linear_solver: Option<&str>,
    preconditioner: Option<&str>,
    x0: Option<&[f64]>,
) -> Result<Vec<f64>, LinearSolveError> {
    let backend = LinearSolver::resolve(linear_solver)?;
    let x = match backend {
        LinearSolver::Direct => {
            // The Python builds no preconditioner for the direct path; an unknown NAME on that
            // path therefore goes unreported there. Resolving it here anyway means a typo is
            // refused rather than silently ignored, which is the same defect class the rest of
            // this module exists to close.
            if let Some(name) = preconditioner {
                Preconditioner::build(name, matrix)
                    .map_err(LinearSolveError::UnknownPreconditioner)?;
            }
            direct::solve(matrix, rhs).map_err(LinearSolveError::Direct)?
        }
        LinearSolver::Krylov(method) => {
            let m = match preconditioner {
                Some(name) => Some(
                    Preconditioner::build(name, matrix)
                        .map_err(LinearSolveError::UnknownPreconditioner)?,
                ),
                None => None,
            };
            method
                .solve(matrix, rhs, x0, m.as_ref())
                .map_err(|cause| LinearSolveError::Krylov {
                    method: method.name(),
                    cause,
                })?
                .x
        }
    };
    check_finite(&x, backend.name())?;
    if matches!(backend, LinearSolver::Direct) {
        check_direct_residual(matrix, rhs, &x)?;
    }
    Ok(x)
}

/// The finiteness half of the guard.
///
/// # Errors
/// [`LinearSolveError::NonFinite`].
pub fn check_finite(x: &[f64], backend: &str) -> Result<(), LinearSolveError> {
    if x.iter().all(|v| v.is_finite()) {
        Ok(())
    } else {
        Err(LinearSolveError::NonFinite {
            backend: backend.to_owned(),
        })
    }
}

/// The SUSPENDERS: reject a direct solution whose relative residual exceeds
/// [`DIRECT_RESIDUAL_RTOL`].
///
/// Public and separate from [`solve_linear_system`] on purpose. The Python tests this half by
/// monkeypatching the registry to return finite garbage, which is the only way to reach the
/// branch — a real factorisation rarely produces it on demand. Exposing the check makes the same
/// test possible here WITHOUT a mutable registry, and makes the guard a thing a caller with its
/// own factorisation can reuse.
///
/// A zero-norm rhs is not judged: `||b|| = 0` has no relative residual, and `x = 0` is the answer.
///
/// # Errors
/// [`LinearSolveError::DirectResidual`].
pub fn check_direct_residual(matrix: &Csr, rhs: &[f64], x: &[f64]) -> Result<(), LinearSolveError> {
    let b_norm = rhs.iter().map(|v| v * v).sum::<f64>().sqrt();
    if b_norm <= 0.0 {
        return Ok(());
    }
    let ax = matrix.matvec(x)?;
    let residual = ax
        .iter()
        .zip(rhs)
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f64>()
        .sqrt()
        / b_norm;
    if residual > DIRECT_RESIDUAL_RTOL {
        return Err(LinearSolveError::DirectResidual {
            residual,
            bound: DIRECT_RESIDUAL_RTOL,
        });
    }
    Ok(())
}

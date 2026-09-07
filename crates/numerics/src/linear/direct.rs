//! The DIRECT sparse LU backend — `faer`'s sparse LU, guarded.
//!
//! # This is NOT `spsolve`, and the port cannot inherit that promise
//!
//! `src/motronics/gauss/linear/__init__.py` documents its `direct` backend as
//! `scipy.sparse.linalg.spsolve` VERBATIM — "bit-for-bit default, so every golden is unchanged".
//! That claim is about SuperLU's exact pivot sequence, its column ordering, and the order in
//! which it accumulates each update. **No other LU can inherit it** — not `faer`, not a
//! hand-written one — because bit-identity is a property of an implementation, not of the
//! mathematics.
//!
//! So this backend is a DIFFERENT backend and is CHARACTERIZED rather than equated: the golden
//! pins the Python's solution and this side is judged against it at a RESIDUAL-based tolerance,
//! not by bit equality. See `rust/crates/gauss/tests/golden_linear.rs` for the tolerance and its
//! reasoning. Nothing here claims to match `spsolve`.
//!
//! # The singular guard, both halves
//!
//! SuperLU emits a `MatrixRankWarning` on a zero pivot and then returns EITHER NaNs OR a
//! FINITE-but-meaningless vector. The finite case slipped past a caller's finiteness check and
//! shipped as `status=ok` garbage — which is why the Python has two guards, not one:
//!
//! * the BELT — promote the warning to a hard error. Here the equivalent is `faer`'s typed
//!   `LuError`, which for a structurally singular matrix refuses to factorise at all.
//! * the SUSPENDERS — reject a relative residual above `DIRECT_RESIDUAL_RTOL`. This is the half
//!   that catches finite garbage, it lives in [`super::solve_linear_system`], and it applies to
//!   this LU exactly as it applied to SuperLU's. A numerically (rather than structurally)
//!   singular system factorises here and produces non-finite or wildly wrong values, so the
//!   suspenders are not decoration.

use contracts::sparse::{Csr, SparseError};
use faer::linalg::solvers::Solve;
use faer::sparse::linalg::solvers::{Lu, SymbolicLu};
use faer::sparse::{SparseColMat, SymbolicSparseColMat, Triplet};
use faer::Mat;

/// Why the direct factorisation refused.
#[derive(Debug, Clone, PartialEq)]
pub enum DirectFailure {
    /// The matrix is (near-)singular — `faer` refused to factorise it.
    ///
    /// The Python's counterpart is `spsolve` raising on the promoted `MatrixRankWarning`, and the
    /// question it should prompt is the same: is a gauge / reference set, and is the (loop) basis
    /// full-rank at this mesh resolution?
    Singular {
        /// `faer`'s own description of the failure.
        detail: String,
    },
    /// The operator or the vectors do not agree on a shape.
    Shape(SparseError),
    /// A cached symbolic factorisation was offered a matrix with a DIFFERENT sparsity pattern.
    ///
    /// Refused rather than repaired: reusing a symbolic half across a pattern change factorises
    /// the wrong structure and returns a plausible-looking vector, which is worse than raising.
    /// The caller re-analyses with [`DirectFactorization::new`].
    PatternChanged,
}

impl From<SparseError> for DirectFailure {
    fn from(e: SparseError) -> Self {
        Self::Shape(e)
    }
}

/// Solve `A x = b` by sparse LU.
///
/// # Errors
/// [`DirectFailure::Singular`] when the factorisation refuses, [`DirectFailure::Shape`] when the
/// matrix is not square or `b` is the wrong length. The residual half of the guard is applied by
/// the caller, [`super::solve_linear_system`].
pub fn solve(a: &Csr, b: &[f64]) -> Result<Vec<f64>, DirectFailure> {
    let (n_rows, n_cols) = a.shape();
    if n_rows != n_cols {
        return Err(DirectFailure::Shape(SparseError::NotSquare {
            n_rows,
            n_cols,
        }));
    }
    if b.len() != n_rows {
        return Err(DirectFailure::Shape(SparseError::ShapeMismatch {
            expected: n_rows,
            got: b.len(),
        }));
    }

    let mut triplets = Vec::with_capacity(a.nnz());
    let (indptr, indices, values) = (a.indptr(), a.indices(), a.values());
    for r in 0..n_rows {
        for i in indptr[r] as usize..indptr[r + 1] as usize {
            triplets.push(Triplet::new(r, indices[i] as usize, values[i]));
        }
    }
    let mat = SparseColMat::<usize, f64>::try_new_from_triplets(n_rows, n_cols, &triplets)
        .map_err(|e| DirectFailure::Singular {
            detail: format!("{e:?}"),
        })?;
    let symbolic = SymbolicLu::try_new(mat.symbolic()).map_err(|e| DirectFailure::Singular {
        detail: format!("symbolic: {e:?}"),
    })?;
    let lu =
        Lu::try_new_with_symbolic(symbolic, mat.as_ref()).map_err(|e| DirectFailure::Singular {
            detail: format!("numeric: {e:?}"),
        })?;
    let rhs = Mat::from_fn(n_rows, 1, |i, _| b[i]);
    let x = lu.solve(&rhs);
    Ok((0..n_rows).map(|i| x[(i, 0)]).collect())
}

/// A sparse LU whose SYMBOLIC half is computed once and reused while only the VALUES change.
///
/// # Why this exists, measured rather than argued
///
/// A Newton iteration reassembles the same mesh: the sparsity pattern is CONSTANT across every
/// iteration and only the entries move. Measured on the real production systems dumped from
/// `python -m motronics run <case> --select fea,tri` (six consecutive Newton systems per case,
/// byte-compared `indptr`/`indices` -- one distinct pattern each):
///
/// | phase (faer, one call)         | `lipo_inductor` n=5355 | `pmasynrm_uv` n=32572 |
/// |--------------------------------|------------------------|-----------------------|
/// | marshal CSR -> triplets -> CSC |  1.14 ms               |  13.26 ms             |
/// | `SymbolicLu` (colamd+symbolic) | 10.01 ms               |  86.34 ms             |
/// | numeric factorisation          | 32.76 ms               | 104.40 ms             |
/// | triangular solve               |  0.51 ms               |   5.48 ms             |
///
/// [`solve`] pays every row on every call, and the first two rows are 34% and 46% of its total
/// while being IDENTICAL between iterations. That is where the "4% slower than SuperLU" verdict
/// came from: SciPy `spsolve` on the same systems measures 35.3 ms and 266.6 ms, so faer's
/// NUMERIC half already wins `pmasynrm_uv` by 2.6x and only its recomputed symbolic half gave
/// the margin back.
///
/// So this type splits the two. [`DirectFactorization::new`] pays colamd, the symbolic
/// factorisation and the CSR->CSC value permutation ONCE; [`DirectFactorization::solve`] pays the
/// numeric factorisation and the triangular solve, and nothing else.
///
/// # The pattern is CHECKED, never assumed
///
/// A cached symbolic that silently accepts a different pattern is the declaration-that-lies
/// defect in its purest form -- it would factorise the wrong structure and return a plausible
/// vector. [`DirectFactorization::solve`] compares the caller's `indptr`/`indices` against the
/// pattern it was built from and returns [`DirectFailure::PatternChanged`] on any difference.
/// The comparison is two slice equalities over `nnz` -- `O(nnz)` against an `O(nnz^1.5)`
/// factorisation, and cheaper than the marshalling it replaces.
pub struct DirectFactorization {
    /// The CSC pattern the symbolic half was computed from.
    pattern: SymbolicSparseColMat<usize>,
    /// The reusable symbolic factorisation (column ordering + structure).
    symbolic: SymbolicLu<usize>,
    /// `csc_slot -> csr_value_index`, so a refactorisation is a scatter rather than a re-sort.
    csr_of_csc: Vec<usize>,
    /// The CSR pattern, kept verbatim so `solve` can REFUSE a caller whose pattern moved.
    indptr: Vec<u32>,
    /// The CSR column indices, kept for the same refusal.
    indices: Vec<u32>,
}

impl DirectFactorization {
    /// Analyse `a`'s pattern once: column ordering, symbolic factorisation, value permutation.
    ///
    /// # Errors
    /// [`DirectFailure::Shape`] when `a` is not square, [`DirectFailure::Singular`] when the
    /// pattern itself is structurally singular and `faer` refuses to analyse it.
    pub fn new(a: &Csr) -> Result<Self, DirectFailure> {
        let (n_rows, n_cols) = a.shape();
        if n_rows != n_cols {
            return Err(DirectFailure::Shape(SparseError::NotSquare {
                n_rows,
                n_cols,
            }));
        }
        let (indptr, indices) = (a.indptr(), a.indices());
        let mut triplets = Vec::with_capacity(a.nnz());
        for r in 0..n_rows {
            let (lo, hi) = (indptr[r] as usize, indptr[r + 1] as usize);
            for (offset, &col) in indices[lo..hi].iter().enumerate() {
                // The VALUE carried through the sort is the CSR slot index, so the sorted result
                // IS the permutation. An index is exact in `f64` far beyond any nnz that fits in
                // memory, so this is a relabelling and not a rounding.
                #[allow(clippy::cast_precision_loss)]
                triplets.push(Triplet::new(r, col as usize, (lo + offset) as f64));
            }
        }
        let labelled = SparseColMat::<usize, f64>::try_new_from_triplets(n_rows, n_cols, &triplets)
            .map_err(|e| DirectFailure::Singular {
                detail: format!("{e:?}"),
            })?;
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let csr_of_csc: Vec<usize> = labelled.val().iter().map(|v| *v as usize).collect();
        let pattern = labelled
            .symbolic()
            .to_owned()
            .map_err(|e| DirectFailure::Singular {
                detail: format!("pattern: {e:?}"),
            })?;
        let symbolic =
            SymbolicLu::try_new(pattern.as_ref()).map_err(|e| DirectFailure::Singular {
                detail: format!("symbolic: {e:?}"),
            })?;
        Ok(Self {
            pattern,
            symbolic,
            csr_of_csc,
            indptr: indptr.to_vec(),
            indices: indices.to_vec(),
        })
    }

    /// The order of the analysed system.
    #[must_use]
    pub fn n(&self) -> usize {
        self.indptr.len().saturating_sub(1)
    }

    /// Solve `A x = b` reusing the cached symbolic half. `a` must carry the analysed pattern.
    ///
    /// # Errors
    /// [`DirectFailure::PatternChanged`] when `a`'s pattern differs from the analysed one,
    /// [`DirectFailure::Shape`] when `b` is the wrong length, [`DirectFailure::Singular`] when
    /// the numeric factorisation refuses. The residual half of the guard is the caller's, exactly
    /// as it is for [`solve`].
    pub fn solve(&self, a: &Csr, b: &[f64]) -> Result<Vec<f64>, DirectFailure> {
        let n = self.n();
        if a.indptr() != self.indptr.as_slice() || a.indices() != self.indices.as_slice() {
            return Err(DirectFailure::PatternChanged);
        }
        if b.len() != n {
            return Err(DirectFailure::Shape(SparseError::ShapeMismatch {
                expected: n,
                got: b.len(),
            }));
        }
        let values = a.values();
        let val: Vec<f64> = self.csr_of_csc.iter().map(|&i| values[i]).collect();
        let mat = SparseColMat::new(self.pattern.clone(), val);
        let lu = Lu::try_new_with_symbolic(self.symbolic.clone(), mat.as_ref()).map_err(|e| {
            DirectFailure::Singular {
                detail: format!("numeric: {e:?}"),
            }
        })?;
        let rhs = Mat::from_fn(n, 1, |i, _| b[i]);
        let x = lu.solve(&rhs);
        Ok((0..n).map(|i| x[(i, 0)]).collect())
    }
}

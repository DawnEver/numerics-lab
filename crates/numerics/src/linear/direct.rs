//! The DIRECT sparse backend — `faer`'s supernodal LU, or its supernodal Cholesky when the
//! operator is exactly symmetric and positive definite; guarded either way.
//!
//! # This is NOT `spsolve`, and the port cannot inherit that promise
//!
//! `src/motronics/gauss/linear/__init__.py` documents its SciPy `direct` backend as
//! `scipy.sparse.linalg.spsolve` — "bit-for-bit" with the SuperLU the goldens were frozen on.
//! That claim is about SuperLU's exact pivot sequence, its column ordering, and the order in
//! which it accumulates each update. **No other factorisation can inherit it** — not `faer`, not a
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
//!
//! # Three choices, each MEASURED (2026-09-26, Xeon W-2223, sequential, median of 5-15)
//!
//! `.claude/memory/2026/09/26/faer-loses-small-systems-because-auto-picks-simplicial.md` has the
//! whole sweep; the three levers it found are the three this file pulls.
//!
//! 1. **Supernodal is FORCED.** `SupernodalThreshold::AUTO` picks the SIMPLICIAL kernel at 5k and
//!    36k unknowns, and that — not Rust against C — is why faer lost small systems to SuperLU.
//!    Forced supernodal: `lipo_inductor` (n=5355) numeric 27.0 -> 8.5 ms (SuperLU 19.7 ms), the
//!    `ipm_uv` chopper (n=36k) 160 -> 49 ms (SuperLU 121 ms). At 111k/129k AUTO already chose
//!    supernodal. No size is measured at which simplicial wins, so there is no threshold.
//! 2. **One numeric factorisation is REUSED.** The previous revision built a fresh `Lu` and cloned
//!    the symbolic on every call: 0.486 s against 0.259 s reused at 111k, 0.54 against 0.42 s at
//!    129k. [`DirectFactorization`] holds one `NumericLu` (and one Cholesky value buffer) and
//!    refactorises into it.
//! 3. **Cholesky when the operator IS symmetric.** The MEC loop matrix is exactly symmetric and SPD
//!    (smallest eigenvalue 2.96); supernodal LLT on it is 0.086 s numeric + 0.020 s solve against
//!    0.34 s for LU. Symmetry is VERIFIED — pattern at analysis, VALUES bit-for-bit on every
//!    refactor — and never declared: a caller's "this is symmetric" is a declaration that can lie,
//!    an `O(nnz)` comparison cannot. A symmetric operator that is not positive definite is refused
//!    by faer's LLT and falls to LU, and the refusal is remembered so it is not paid twice.
//!
//! Threading (`Par::rayon`) was measured to HURT at every size here and is not used.

use contracts::sparse::{Csr, SparseError};
use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::cholesky::llt::factor::LltRegularization;
use faer::sparse::linalg::cholesky::{
    factorize_symbolic_cholesky, CholeskySymbolicParams, LltRef, SymbolicCholesky,
    SymmetricOrdering,
};
use faer::sparse::linalg::lu::{
    factorize_symbolic_lu, LuRef, LuSymbolicParams, NumericLu, SymbolicLu,
};
use faer::sparse::linalg::SupernodalThreshold;
use faer::sparse::{SparseColMat, SparseColMatRef, SymbolicSparseColMat, Triplet};
use faer::{Conj, MatMut, Par, Side};

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
    /// A back-substitution was asked for before any numeric factorisation existed (or after one
    /// failed). Refused rather than answered from a factor of some OTHER matrix.
    NotFactored,
}

impl From<SparseError> for DirectFailure {
    fn from(e: SparseError) -> Self {
        Self::Shape(e)
    }
}

/// Which factorisation the last [`DirectFactorization::refactor`] produced.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FactorKind {
    /// Supernodal LU with partial pivoting (COLAMD column ordering).
    Lu,
    /// Supernodal `L L^T` (AMD ordering) — only for an operator verified exactly symmetric, and
    /// only once faer's LLT has accepted it as positive definite.
    Cholesky,
}

impl FactorKind {
    /// The wire spelling (`"lu"` / `"cholesky"`) the Python labels its solve with.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Lu => "lu",
            Self::Cholesky => "cholesky",
        }
    }
}

/// Solve `A x = b` once: analyse, factorise, back-substitute.
///
/// # Errors
/// [`DirectFailure::Singular`] when the factorisation refuses, [`DirectFailure::Shape`] when the
/// matrix is not square or `b` is the wrong length. The residual half of the guard is applied by
/// the caller, [`super::solve_linear_system`].
pub fn solve(a: &Csr, b: &[f64]) -> Result<Vec<f64>, DirectFailure> {
    DirectFactorization::new(a)?.solve(a, b)
}

/// The supernodal threshold every analysis here uses — see the module docs, lever 1.
const THRESHOLD: SupernodalThreshold = SupernodalThreshold::FORCE_SUPERNODAL;

fn singular(stage: &str, e: impl core::fmt::Debug) -> DirectFailure {
    DirectFailure::Singular {
        detail: format!("{stage}: {e:?}"),
    }
}

/// The symmetric half: analysed lazily, on the first refactor whose values are symmetric.
struct Cholesky {
    symbolic: SymbolicCholesky<usize>,
    /// The `L` values, REUSED across refactors (lever 2).
    l_values: Vec<f64>,
}

/// A sparse direct factorisation whose SYMBOLIC half is computed once and whose NUMERIC storage
/// is reused while only the VALUES change.
///
/// # Why this exists, measured rather than argued
///
/// A Newton iteration reassembles the same mesh: the sparsity pattern is CONSTANT across every
/// iteration and only the entries move (byte-compared `indptr`/`indices` on six consecutive
/// production Newton systems per case — one distinct pattern each). [`DirectFactorization::new`]
/// pays the CSR->CSC value permutation and the transpose map ONCE; each ordering + symbolic
/// analysis is paid once, on first use; [`DirectFactorization::refactor`] pays the numeric
/// factorisation into storage it already owns, and [`DirectFactorization::solve_in_place`] the
/// triangular solves.
///
/// # The pattern is CHECKED, never assumed
///
/// A cached symbolic that silently accepts a different pattern is the declaration-that-lies
/// defect in its purest form -- it would factorise the wrong structure and return a plausible
/// vector. [`DirectFactorization::refactor`] compares the caller's `indptr`/`indices` against the
/// pattern it was built from and returns [`DirectFailure::PatternChanged`] on any difference.
/// The comparison is two slice equalities over `nnz` -- `O(nnz)` against an `O(nnz^1.5)`
/// factorisation.
pub struct DirectFactorization {
    /// The CSC pattern the symbolic halves are computed from.
    pattern: SymbolicSparseColMat<usize>,
    /// `csc_slot -> csr_value_index`, so a refactorisation is a scatter rather than a re-sort.
    csr_of_csc: Vec<usize>,
    /// `csr_slot -> csr slot of its transpose`, or `None` when the PATTERN is unsymmetric (then
    /// no value set can be symmetric and the check is never paid).
    transpose_of: Option<Vec<usize>>,
    /// The CSR pattern, kept verbatim so `refactor` can REFUSE a caller whose pattern moved.
    indptr: Vec<u32>,
    /// The CSR column indices, kept for the same refusal.
    indices: Vec<u32>,
    /// The CSC values of the operator last refactorised (reused buffer).
    csc_values: Vec<f64>,
    /// COLAMD + supernodal LU symbolic, analysed on first LU refactor.
    lu_symbolic: Option<SymbolicLu<usize>>,
    /// The ONE numeric LU, refactorised in place (lever 2).
    lu_numeric: NumericLu<usize, f64>,
    /// AMD + supernodal Cholesky, analysed on first symmetric refactor.
    cholesky: Option<Cholesky>,
    /// Set once faer's LLT has refused this operator as not positive definite.
    cholesky_refused: bool,
    /// What the last successful refactor produced; `None` before one / after a failed one.
    current: Option<FactorKind>,
}

impl DirectFactorization {
    /// Analyse `a`'s pattern once: value permutation and transpose map.
    ///
    /// # Errors
    /// [`DirectFailure::Shape`] when `a` is not square, [`DirectFailure::Singular`] when `faer`
    /// refuses the pattern.
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
            .map_err(|e| singular("pattern", e))?;
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let csr_of_csc: Vec<usize> = labelled.val().iter().map(|v| *v as usize).collect();
        let pattern = labelled
            .symbolic()
            .to_owned()
            .map_err(|e| singular("pattern", e))?;
        Ok(Self {
            pattern,
            transpose_of: transpose_map(n_rows, indptr, indices),
            csc_values: vec![0.0; csr_of_csc.len()],
            csr_of_csc,
            indptr: indptr.to_vec(),
            indices: indices.to_vec(),
            lu_symbolic: None,
            lu_numeric: NumericLu::new(),
            cholesky: None,
            cholesky_refused: false,
            current: None,
        })
    }

    /// The order of the analysed system.
    #[must_use]
    pub fn n(&self) -> usize {
        self.indptr.len().saturating_sub(1)
    }

    /// What the last successful [`Self::refactor`] produced, if any.
    #[must_use]
    pub fn kind(&self) -> Option<FactorKind> {
        self.current
    }

    /// Whether `values` (in the analysed CSR slot order) are EXACTLY symmetric — bitwise equal to
    /// their transposes. A NaN is unequal to itself and so is never symmetric, which routes it to
    /// the LU whose refusal names it.
    fn values_symmetric(&self, values: &[f64]) -> bool {
        self.transpose_of.as_ref().is_some_and(|t| {
            t.iter()
                .enumerate()
                .all(|(k, &kt)| values[k].to_bits() == values[kt].to_bits())
        })
    }

    /// Numerically factorise `a` (which must carry the analysed pattern) into the held storage.
    ///
    /// # Errors
    /// [`DirectFailure::PatternChanged`] when `a`'s pattern differs from the analysed one,
    /// [`DirectFailure::Singular`] when the LU refuses. After an error no factor is held and
    /// [`Self::solve_in_place`] answers [`DirectFailure::NotFactored`].
    pub fn refactor(&mut self, a: &Csr) -> Result<FactorKind, DirectFailure> {
        if a.indptr() != self.indptr.as_slice() || a.indices() != self.indices.as_slice() {
            return Err(DirectFailure::PatternChanged);
        }
        self.current = None;
        let values = a.values();
        for (dst, &src) in self.csc_values.iter_mut().zip(&self.csr_of_csc) {
            *dst = values[src];
        }
        if !self.cholesky_refused && self.values_symmetric(values) {
            if self.try_cholesky()? {
                self.current = Some(FactorKind::Cholesky);
                return Ok(FactorKind::Cholesky);
            }
            self.cholesky_refused = true;
        }
        self.factor_lu()?;
        self.current = Some(FactorKind::Lu);
        Ok(FactorKind::Lu)
    }

    /// `Ok(true)` when LLT succeeded, `Ok(false)` when faer refused it as not positive definite.
    fn try_cholesky(&mut self) -> Result<bool, DirectFailure> {
        if self.cholesky.is_none() {
            let symbolic = factorize_symbolic_cholesky(
                self.pattern.as_ref(),
                Side::Lower,
                SymmetricOrdering::Amd,
                CholeskySymbolicParams {
                    supernodal_flop_ratio_threshold: THRESHOLD,
                    ..Default::default()
                },
            )
            .map_err(|e| singular("cholesky symbolic", e))?;
            let l_values = vec![0.0; symbolic.len_val()];
            self.cholesky = Some(Cholesky { symbolic, l_values });
        }
        let Some(chol) = self.cholesky.as_mut() else {
            return Ok(false);
        };
        let mat = SparseColMatRef::new(self.pattern.as_ref(), &self.csc_values);
        let mut buf = MemBuffer::new(
            chol.symbolic
                .factorize_numeric_llt_scratch::<f64>(Par::Seq, Default::default()),
        );
        Ok(chol
            .symbolic
            .factorize_numeric_llt::<f64>(
                &mut chol.l_values,
                mat,
                Side::Lower,
                LltRegularization::default(),
                Par::Seq,
                MemStack::new(&mut buf),
                Default::default(),
            )
            .is_ok())
    }

    fn factor_lu(&mut self) -> Result<(), DirectFailure> {
        if self.lu_symbolic.is_none() {
            let symbolic = factorize_symbolic_lu(
                self.pattern.as_ref(),
                LuSymbolicParams {
                    supernodal_flop_ratio_threshold: THRESHOLD,
                    ..Default::default()
                },
            )
            .map_err(|e| singular("symbolic", e))?;
            self.lu_symbolic = Some(symbolic);
        }
        let Some(symbolic) = self.lu_symbolic.as_ref() else {
            return Err(DirectFailure::NotFactored);
        };
        let mat = SparseColMatRef::new(self.pattern.as_ref(), &self.csc_values);
        let mut buf = MemBuffer::new(
            symbolic.factorize_numeric_lu_scratch::<f64>(Par::Seq, Default::default()),
        );
        symbolic
            .factorize_numeric_lu(
                &mut self.lu_numeric,
                mat,
                Par::Seq,
                MemStack::new(&mut buf),
                Default::default(),
            )
            .map_err(|e| singular("numeric", e))?;
        Ok(())
    }

    /// Back-substitute `nrhs` right-hand sides, stored COLUMN-MAJOR in `b`, against the factor the
    /// last [`Self::refactor`] produced. `b` is overwritten with the solutions.
    ///
    /// # Errors
    /// [`DirectFailure::NotFactored`] when no factor is held, [`DirectFailure::Shape`] when
    /// `b.len() != n * nrhs`.
    pub fn solve_in_place(&self, b: &mut [f64], nrhs: usize) -> Result<(), DirectFailure> {
        let n = self.n();
        if b.len() != n * nrhs {
            return Err(DirectFailure::Shape(SparseError::ShapeMismatch {
                expected: n * nrhs,
                got: b.len(),
            }));
        }
        let rhs = MatMut::from_column_major_slice_mut(b, n, nrhs);
        match self.current {
            Some(FactorKind::Cholesky) => {
                let Some(chol) = self.cholesky.as_ref() else {
                    return Err(DirectFailure::NotFactored);
                };
                let mut buf = MemBuffer::new(chol.symbolic.solve_in_place_scratch::<f64>(nrhs, Par::Seq));
                LltRef::new(&chol.symbolic, &chol.l_values).solve_in_place_with_conj(
                    Conj::No,
                    rhs,
                    Par::Seq,
                    MemStack::new(&mut buf),
                );
            }
            Some(FactorKind::Lu) => {
                let Some(symbolic) = self.lu_symbolic.as_ref() else {
                    return Err(DirectFailure::NotFactored);
                };
                let mut buf = MemBuffer::new(symbolic.solve_in_place_scratch::<f64>(nrhs, Par::Seq));
                LuRef::new_unchecked(symbolic, &self.lu_numeric).solve_in_place_with_conj(
                    Conj::No,
                    rhs,
                    Par::Seq,
                    MemStack::new(&mut buf),
                );
            }
            None => return Err(DirectFailure::NotFactored),
        }
        Ok(())
    }

    /// Refactorise against `a` and solve one right-hand side — [`Self::refactor`] then
    /// [`Self::solve_in_place`].
    ///
    /// # Errors
    /// As [`Self::refactor`], plus [`DirectFailure::Shape`] when `b` is the wrong length (checked
    /// BEFORE the factorisation is paid for).
    pub fn solve(&mut self, a: &Csr, b: &[f64]) -> Result<Vec<f64>, DirectFailure> {
        if b.len() != self.n() {
            return Err(DirectFailure::Shape(SparseError::ShapeMismatch {
                expected: self.n(),
                got: b.len(),
            }));
        }
        self.refactor(a)?;
        let mut x = b.to_vec();
        self.solve_in_place(&mut x, 1)?;
        Ok(x)
    }
}


/// `csr_slot -> slot of (col, row)` when every stored `(row, col)` has a stored transpose, else
/// `None`. Columns within a canonical CSR row are ascending, so each lookup is a binary search.
fn transpose_map(n: usize, indptr: &[u32], indices: &[u32]) -> Option<Vec<usize>> {
    let mut out = Vec::with_capacity(indices.len());
    for r in 0..n {
        for &c in &indices[indptr[r] as usize..indptr[r + 1] as usize] {
            let (lo, hi) = (indptr[c as usize] as usize, indptr[c as usize + 1] as usize);
            let row = u32::try_from(r).ok()?;
            let at = indices[lo..hi].binary_search(&row).ok()?;
            out.push(lo + at);
        }
    }
    Some(out)
}

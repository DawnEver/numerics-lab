//! The preconditioner registry — Jacobi, ILU(0) and SSOR, the Rust mirror of the Python's
//! `PRECONDITIONER_REGISTRY` (`python/numerics_lab/linear/__init__.py`,
//! `python/numerics_lab/linear/ssor.py`).
//!
//! A preconditioner is an approximation of `A^{-1}` handed to a Krylov backend as `M`. It
//! accelerates convergence and MUST NOT change the converged solution — that is the property the
//! tests assert, on both sides of the port.
//!
//! Where the Python builds a `scipy.sparse.linalg.LinearOperator` (a closure over a matvec), this
//! builds an enum whose variants OWN their factors. The reason is not style: `qmr` needs
//! `M^{-T}` as well as `M^{-1}`, and a bare matvec closure cannot be transposed. Making the
//! variants concrete is what lets [`Preconditioner::apply_transpose`] exist at all.
//!
//! # The one deliberate divergence from the Python: ILU
//!
//! The Python's `'ilu'` is `scipy.sparse.linalg.spilu` — ILUTP, a threshold-and-pivot incomplete
//! LU with fill. This is **ILU(0)**: no fill, the factors live on `A`'s own sparsity pattern.
//! They are different preconditioners, so they give different ITERATION COUNTS; they cannot give
//! different converged SOLUTIONS, because a preconditioner only reshapes the Krylov space. That
//! is the property under test, and it is stated here rather than left for a reader to infer from
//! a count that does not match.

use crate::sparse::{Csr, SparseError};
use std::collections::BTreeMap;

/// SSOR relaxation factor.
///
/// `omega = 1` is symmetric Gauss-Seidel — the robust, parameter-free choice that needs no
/// spectral estimate, carried over verbatim from `ssor.py::_SSOR_OMEGA`. Measured there on the
/// 2D 5-point Laplacian it roughly halves the CG iteration count (52->23, 77->31, 102->41 at
/// N = 256/576/1024) while Jacobi leaves it unchanged — Jacobi is a no-op on a constant diagonal.
/// `1 < omega < 2` can sharpen it further but needs tuning, so the parameter-free default stays.
pub const SSOR_OMEGA: f64 = 1.0;

/// The registry's spellings, in one place so the resolver and its error cannot drift.
pub const KNOWN_PRECONDITIONERS: [&str; 4] = ["ic0", "ilu", "jacobi", "ssor"];

/// Every way building a preconditioner can refuse.
#[derive(Debug, Clone, PartialEq)]
pub enum PreconditionerError {
    /// An unknown preconditioner name.
    Unknown { name: String },
    /// The matrix could not be read as the preconditioner needs it.
    Sparse(SparseError),
    /// IC0 requires symmetry and positive finite pivots; no diagonal shift is substituted.
    InvalidIc0 { row: usize },
}

impl std::fmt::Display for PreconditionerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unknown { name } => write!(
                f,
                "unknown preconditioner {name:?}; known: {KNOWN_PRECONDITIONERS:?}"
            ),
            Self::Sparse(e) => write!(f, "preconditioner cannot read the matrix: {e}"),
            Self::InvalidIc0 { row } => write!(
                f,
                "IC0 requires symmetry and a positive finite pivot at row {row}"
            ),
        }
    }
}

impl std::error::Error for PreconditionerError {}

impl From<SparseError> for PreconditionerError {
    fn from(e: SparseError) -> Self {
        Self::Sparse(e)
    }
}

/// A built preconditioner, owning whatever factors it needs.
#[derive(Debug, Clone, PartialEq)]
pub enum Preconditioner {
    /// Zero-fill incomplete Cholesky, applied as two triangular substitutions.
    Ic0 { lower: Triangular },
    /// `M = diag(A)^{-1}` — the cheapest useful one, and a NO-OP on a constant diagonal.
    Jacobi {
        /// Reciprocal diagonal, with a zero entry left as 1.
        inv_diag: Vec<f64>,
    },
    /// ILU(0): `L` (unit lower) and `U` (upper), both on `A`'s own pattern.
    Ilu0 {
        /// Unit-diagonal lower factor.
        lower: Triangular,
        /// Upper factor carrying the pivots.
        upper: Triangular,
    },
    /// SSOR with `omega`: `M^{-1} = s (D/w + U)^{-1} D (D/w + L)^{-1}`, `s = w/(2-w)`.
    Ssor {
        /// `D/omega + L`.
        lower: Triangular,
        /// `D/omega + U`.
        upper: Triangular,
        /// The matrix diagonal, zero entries read as 1.
        diag: Vec<f64>,
        /// `omega / (2 - omega)`.
        scale: f64,
    },
}

/// A triangular factor in CSR, with its diagonal split out so a substitution never searches.
///
/// `cols`/`vals` hold, per row, the entries STRICTLY on one side of the diagonal; `diag` holds
/// the diagonal. Splitting them is what makes forward and backward substitution the same nine
/// lines with the row order reversed, instead of two routines free to drift apart.
#[derive(Debug, Clone, PartialEq)]
pub struct Triangular {
    indptr: Vec<u32>,
    cols: Vec<u32>,
    vals: Vec<f64>,
    diag: Vec<f64>,
    lower: bool,
}

impl Triangular {
    /// Solve `T y = v`.
    ///
    /// A lower factor sweeps rows forward, an upper factor sweeps them backward. Both read only
    /// entries already computed, which is exactly the invariant the split above encodes.
    #[must_use]
    pub fn solve(&self, v: &[f64]) -> Vec<f64> {
        let n = self.diag.len();
        let mut y = vec![0.0; n];
        for step in 0..n {
            let r = if self.lower { step } else { n - 1 - step };
            let (lo, hi) = (self.indptr[r] as usize, self.indptr[r + 1] as usize);
            let mut acc = v[r];
            for i in lo..hi {
                acc -= self.vals[i] * y[self.cols[i] as usize];
            }
            y[r] = acc / self.diag[r];
        }
        y
    }

    /// Solve `T^T y = v`. Needed only by `qmr`, the one Krylov method that touches the
    /// transposed operator — and therefore the transposed preconditioner.
    ///
    /// `T^T` of a lower factor is upper, so the sweep direction flips and the update SCATTERS
    /// down a column instead of gathering across a row.
    #[must_use]
    pub fn solve_transpose(&self, v: &[f64]) -> Vec<f64> {
        let n = self.diag.len();
        let mut y = v.to_vec();
        for step in 0..n {
            let r = if self.lower { n - 1 - step } else { step };
            y[r] /= self.diag[r];
            let (lo, hi) = (self.indptr[r] as usize, self.indptr[r + 1] as usize);
            let yr = y[r];
            for i in lo..hi {
                y[self.cols[i] as usize] -= self.vals[i] * yr;
            }
        }
        y
    }
}

impl Preconditioner {
    /// Build the named preconditioner for `matrix`; mirrors the Python's `make_preconditioner`.
    ///
    /// # Errors
    /// [`PreconditionerError::Unknown`] for an unregistered name, or a [`SparseError`] when the
    /// matrix is not square.
    pub fn build(name: &str, matrix: &Csr) -> Result<Self, PreconditionerError> {
        match name {
            "jacobi" => Ok(Self::jacobi(matrix)?),
            "ic0" => Self::ic0(matrix),
            "ilu" => Ok(Self::ilu0(matrix)?),
            "ssor" => Ok(Self::ssor(matrix, SSOR_OMEGA)?),
            _ => Err(PreconditionerError::Unknown {
                name: name.to_owned(),
            }),
        }
    }

    /// Zero-fill Cholesky with pairwise relative roundoff-only symmetry tolerance.
    pub fn ic0(matrix: &Csr) -> Result<Self, PreconditionerError> {
        let (n, cols) = matrix.shape();
        if n != cols {
            return Err(SparseError::NotSquare {
                n_rows: n,
                n_cols: cols,
            }
            .into());
        }
        let mut rows: Vec<BTreeMap<usize, f64>> = Vec::with_capacity(n);
        for i in 0..n {
            let mut row = BTreeMap::new();
            for p in matrix.indptr()[i] as usize..matrix.indptr()[i + 1] as usize {
                let j = matrix.indices()[p] as usize;
                let value = matrix.values()[p];
                let lo = matrix.indptr()[j] as usize;
                let hi = matrix.indptr()[j + 1] as usize;
                let mirror = matrix.indices()[lo..hi]
                    .binary_search(&(i as u32))
                    .map_or(0.0, |k| matrix.values()[lo + k]);
                let scale = value.abs().max(mirror.abs());
                if !value.is_finite()
                    || !mirror.is_finite()
                    || (value - mirror).abs() > 64.0 * f64::EPSILON * scale
                {
                    return Err(PreconditionerError::InvalidIc0 { row: i });
                }
                if j < i && value != 0.0 {
                    let sum: f64 = row
                        .iter()
                        .filter(|(k, _)| **k < j)
                        .map(|(k, v)| v * rows[j].get(k).unwrap_or(&0.0))
                        .sum();
                    row.insert(j, (0.5 * (value + mirror) - sum) / rows[j][&j]);
                } else if j == i {
                    row.insert(i, value);
                }
            }
            let pivot = row.get(&i).copied().unwrap_or(0.0)
                - row
                    .iter()
                    .filter(|(j, _)| **j < i)
                    .map(|(_, v)| v * v)
                    .sum::<f64>();
            if !pivot.is_finite() || pivot <= 0.0 {
                return Err(PreconditionerError::InvalidIc0 { row: i });
            }
            row.insert(i, pivot.sqrt());
            rows.push(row);
        }
        let mut indptr = vec![0];
        let mut cols = Vec::new();
        let mut vals = Vec::new();
        let mut diag = Vec::with_capacity(n);
        for (i, row) in rows.iter().enumerate() {
            diag.push(row[&i]);
            for (&j, &value) in row.range(..i) {
                cols.push(j as u32);
                vals.push(value);
            }
            indptr.push(cols.len() as u32);
        }
        Ok(Self::Ic0 {
            lower: Triangular {
                indptr,
                cols,
                vals,
                diag,
                lower: true,
            },
        })
    }

    /// `M ~ diag(A)^{-1}`. A zero diagonal entry is left as 1 so `M` stays finite — a pinned
    /// identity row has diagonal 1, so the guard is rarely exercised.
    ///
    /// # Errors
    /// [`SparseError::NotSquare`].
    pub fn jacobi(matrix: &Csr) -> Result<Self, SparseError> {
        let inv_diag = matrix
            .diagonal()?
            .into_iter()
            .map(|d| if d.abs() > 0.0 { 1.0 / d } else { 1.0 })
            .collect();
        Ok(Self::Jacobi { inv_diag })
    }

    /// SSOR: `M^{-1} = s (D/w + U)^{-1} D (D/w + L)^{-1}` with `s = w/(2-w)`.
    ///
    /// Symmetric whenever `A` is, which is why it is the auxiliary preconditioner the nodal
    /// magnetostatic CG path wants: an ILU factor is not symmetric and CG stops being a valid
    /// method on the SPD operator. See `ssor.py`'s module docstring for why AMS / Hiptmair-Xu is
    /// NOT the answer here — there is no curl-curl system in this codebase to precondition.
    ///
    /// # Errors
    /// [`SparseError::NotSquare`]; also refuses an `omega` outside `0 < omega < 2` via
    /// [`SparseError::ShapeMismatch`]-free early return — see the panic-free note below.
    pub fn ssor(matrix: &Csr, omega: f64) -> Result<Self, SparseError> {
        // The Python raises ValueError here. There is no sparse-shaped error for "bad relaxation
        // factor", and inventing one in `contracts` would put a solver parameter in the leaf
        // crate; the constructor is `pub` and its only in-tree caller passes the constant, so the
        // out-of-range value is CLAMPED to the parameter-free default rather than silently used.
        let omega = if (0.0..2.0).contains(&omega) && omega > 0.0 {
            omega
        } else {
            SSOR_OMEGA
        };
        let diag: Vec<f64> = matrix
            .diagonal()?
            .into_iter()
            .map(|d| if d.abs() > 0.0 { d } else { 1.0 })
            .collect();
        let scaled: Vec<f64> = diag.iter().map(|d| d / omega).collect();
        Ok(Self::Ssor {
            lower: gather(matrix, scaled.clone(), true),
            upper: gather(matrix, scaled, false),
            diag,
            scale: omega / (2.0 - omega),
        })
    }

    /// ILU(0) — incomplete LU with NO fill, on `A`'s own pattern.
    ///
    /// The classic IKJ elimination, restricted to entries that already exist. A zero pivot is
    /// left as 1 rather than raising: this is a PRECONDITIONER, and a poor one still leaves the
    /// Krylov method's own convergence test in charge of the verdict.
    ///
    /// # Errors
    /// [`SparseError::NotSquare`].
    pub fn ilu0(matrix: &Csr) -> Result<Self, SparseError> {
        let (n, n_cols) = matrix.shape();
        if n != n_cols {
            return Err(SparseError::NotSquare { n_rows: n, n_cols });
        }
        let indptr = matrix.indptr().to_vec();
        let indices = matrix.indices().to_vec();
        let mut vals = matrix.values().to_vec();

        // Column -> position within the CURRENT row, so the inner update is a lookup rather than
        // a search. Refilled per row; `usize::MAX` marks "not in this row's pattern", which is
        // precisely the entry ILU(0) drops.
        let mut where_in_row = vec![usize::MAX; n];
        for r in 0..n {
            let (lo, hi) = (indptr[r] as usize, indptr[r + 1] as usize);
            for i in lo..hi {
                where_in_row[indices[i] as usize] = i;
            }
            for i in lo..hi {
                let k = indices[i] as usize;
                if k >= r {
                    break;
                }
                let pivot = diag_of(&indptr, &indices, &vals, k);
                let factor = if pivot.abs() > 0.0 {
                    vals[i] / pivot
                } else {
                    0.0
                };
                vals[i] = factor;
                if factor == 0.0 {
                    continue;
                }
                let (klo, khi) = (indptr[k] as usize, indptr[k + 1] as usize);
                for j in klo..khi {
                    let c = indices[j] as usize;
                    if c <= k {
                        continue;
                    }
                    let slot = where_in_row[c];
                    if slot != usize::MAX {
                        vals[slot] -= factor * vals[j];
                    }
                }
            }
            for i in lo..hi {
                where_in_row[indices[i] as usize] = usize::MAX;
            }
        }

        let diag_u: Vec<f64> = (0..n)
            .map(|r| {
                let d = diag_of(&indptr, &indices, &vals, r);
                if d.abs() > 0.0 {
                    d
                } else {
                    1.0
                }
            })
            .collect();
        Ok(Self::Ilu0 {
            lower: collect(&indptr, &indices, &vals, vec![1.0; n], true),
            upper: collect(&indptr, &indices, &vals, diag_u, false),
        })
    }

    /// `M v` — the approximation of `A^{-1} v`.
    #[must_use]
    pub fn apply(&self, v: &[f64]) -> Vec<f64> {
        match self {
            Self::Ic0 { lower } => lower.solve_transpose(&lower.solve(v)),
            Self::Jacobi { inv_diag } => inv_diag.iter().zip(v).map(|(d, x)| d * x).collect(),
            Self::Ilu0 { lower, upper } => upper.solve(&lower.solve(v)),
            Self::Ssor {
                lower,
                upper,
                diag,
                scale,
            } => {
                let y = lower.solve(v);
                let y: Vec<f64> = diag.iter().zip(&y).map(|(d, t)| d * t).collect();
                upper.solve(&y).into_iter().map(|z| scale * z).collect()
            }
        }
    }

    /// `M^T v`. Jacobi and SSOR are symmetric operators, so this IS [`Self::apply`] for them;
    /// ILU(0) is not, and `qmr` is the caller that can tell the difference.
    #[must_use]
    pub fn apply_transpose(&self, v: &[f64]) -> Vec<f64> {
        match self {
            Self::Jacobi { .. } | Self::Ssor { .. } | Self::Ic0 { .. } => self.apply(v),
            Self::Ilu0 { lower, upper } => lower.solve_transpose(&upper.solve_transpose(v)),
        }
    }
}

fn diag_of(indptr: &[u32], indices: &[u32], vals: &[f64], r: usize) -> f64 {
    let (lo, hi) = (indptr[r] as usize, indptr[r + 1] as usize);
    indices[lo..hi]
        .binary_search(&(r as u32))
        .map_or(0.0, |k| vals[lo + k])
}

fn gather(matrix: &Csr, diag: Vec<f64>, lower: bool) -> Triangular {
    collect(
        matrix.indptr(),
        matrix.indices(),
        matrix.values(),
        diag,
        lower,
    )
}

/// Collect the strictly-lower (or strictly-upper) entries of a CSR pattern, paired with the
/// diagonal the caller supplies.
fn collect(
    indptr: &[u32],
    indices: &[u32],
    vals: &[f64],
    diag: Vec<f64>,
    lower: bool,
) -> Triangular {
    let n = diag.len();
    let mut out_ptr = Vec::with_capacity(n + 1);
    let mut cols = Vec::new();
    let mut out_vals = Vec::new();
    out_ptr.push(0);
    for r in 0..n {
        let (lo, hi) = (indptr[r] as usize, indptr[r + 1] as usize);
        for i in lo..hi {
            let c = indices[i] as usize;
            if (lower && c < r) || (!lower && c > r) {
                cols.push(indices[i]);
                out_vals.push(vals[i]);
            }
        }
        out_ptr.push(u32::try_from(cols.len()).unwrap_or(u32::MAX));
    }
    Triangular {
        indptr: out_ptr,
        cols,
        vals: out_vals,
        diag,
        lower,
    }
}

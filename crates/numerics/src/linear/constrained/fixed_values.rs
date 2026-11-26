//! Fixed values (Dirichlet pinning) — pure algebra, no physics, no mesh.
//!
//! The Rust mirror of `python/numerics_lab/linear/constrained/fixed_values.py`. It pins an
//! assembly's COO stream ([`apply_dirichlet_triplets`]); [`super::pin_rows`] pins a matrix that
//! is already built.
//!
//! # Both scalar types, one implementation
//!
//! The Python's `apply_dirichlet_triplets` is dtype-agnostic on purpose: a complex system pins a
//! complex value through it, and a real-only implementation would silently drop the imaginary
//! part. [`apply_dirichlet_triplets`] is generic over [`Scalar`] for the same reason, so a `c64`
//! pin keeps both components and the pinned triplets feed a `Coo<c64>` directly.

use crate::sparse::{Coo, Csr, Scalar, SparseError};

/// A Dirichlet group: one prescribed value and the indices it is imposed on.
///
/// The Python keys a `dict` by VALUE (`{value: index_ids}`) because that is the shape the
/// FEA/harmonic boundary maps already use. A `f64` cannot be a hash key in Rust, and it should
/// not be one in Python either — two groups whose values differ by an ulp are different keys —
/// so the port carries the pairs as a SEQUENCE, which is also what `dict` iteration order gives.
#[derive(Debug, Clone, PartialEq)]
pub struct DirichletGroup<T = f64> {
    /// The prescribed value.
    pub value: T,
    /// The indices it is imposed on. Duplicates are collapsed, mirroring the Python's
    /// `np.unique`.
    pub indices: Vec<u32>,
}

/// Exact identity equations may be lifted without changing the free operator.
/// This undoes row-only pinning's artificial asymmetry, not physical asymmetry.
pub fn lift_identity_rows(
    matrix: &Csr,
    rhs: &[f64],
) -> Result<(Csr, Vec<f64>, Vec<usize>), SparseError> {
    let (n, cols) = matrix.shape();
    if n != cols {
        return Err(SparseError::NotSquare {
            n_rows: n,
            n_cols: cols,
        });
    }
    if rhs.len() != n {
        return Err(SparseError::ShapeMismatch {
            expected: n,
            got: rhs.len(),
        });
    }
    let mut pinned = vec![false; n];
    for (row, pin) in pinned.iter_mut().enumerate() {
        let mut diagonal = false;
        let mut identity = true;
        for k in matrix.indptr()[row] as usize..matrix.indptr()[row + 1] as usize {
            let col = matrix.indices()[k] as usize;
            let value = matrix.values()[k];
            if col == row {
                diagonal = value == 1.0;
            } else if value != 0.0 {
                identity = false;
            }
        }
        *pin = diagonal && identity;
    }
    let ids: Vec<_> = pinned
        .iter()
        .enumerate()
        .filter_map(|(i, &pin)| pin.then_some(i))
        .collect();
    if ids.is_empty() {
        return Ok((matrix.clone(), rhs.to_vec(), ids));
    }
    let mut lifted = rhs.to_vec();
    let mut coo = Coo::new(n, n);
    for (row, lifted_value) in lifted.iter_mut().enumerate() {
        for k in matrix.indptr()[row] as usize..matrix.indptr()[row + 1] as usize {
            let col = matrix.indices()[k] as usize;
            let value = matrix.values()[k];
            if pinned[col] && row != col {
                *lifted_value -= value * rhs[col];
            } else {
                coo.push(row as u32, col as u32, value)?;
            }
        }
    }
    Ok((coo.to_csr()?, lifted, ids))
}

/// Pin `rhs[i] = value + bias` for every index in every group, and turn each pinned row of the
/// COO triplets into the identity row.
///
/// Existing entries in a pinned row are ZEROED rather than removed, so the returned arrays stay
/// valid COO input — exactly the Python's choice, and the reason the appended unit diagonal does
/// not collide with anything: `Coo::to_csr` SUMS duplicates, and `0 + 1 = 1`.
///
/// The COLUMN of a pinned index is deliberately NOT eliminated. This is row-pinning, not static
/// condensation, so an unpinned row keeps its coupling to a pinned unknown.
///
/// `rhs` is modified in place at the pinned indices; the triplets are consumed and returned
/// extended.
#[must_use]
pub fn apply_dirichlet_triplets<T: Scalar>(
    mut rows: Vec<u32>,
    mut cols: Vec<u32>,
    mut values: Vec<T>,
    rhs: &mut [T],
    groups: &[DirichletGroup<T>],
    bias: T,
) -> (Vec<u32>, Vec<u32>, Vec<T>) {
    for group in groups {
        let mut ids = group.indices.clone();
        ids.sort_unstable();
        ids.dedup();
        if ids.is_empty() {
            continue;
        }
        for &i in &ids {
            rhs[i as usize] = group.value + bias;
        }
        for (k, &r) in rows.iter().enumerate() {
            if ids.binary_search(&r).is_ok() {
                values[k] = T::ZERO;
            }
        }
        for &i in &ids {
            rows.push(i);
            cols.push(i);
            values.push(T::ONE);
        }
    }
    (rows, cols, values)
}

//! Master-slave constraint chains, resolved so no master is itself constrained.
//!
//! Port of `resolve_constraint_chains` in `python/numerics_lab/linear/constrained/master_slave.py`, and it
//! lives HERE because that is where the Python has it.
//!
//! # It arrived here as a MOVE, which is the point
//!
//! The first Rust caller was the MEC boundary-fold reduction, and the port landed in
//! `hamilton_mec::constraints` with its own docstring recording the placement as DEBT: "when a
//! second Rust caller appears it MOVES to the engine (`numerics::linear::constrained::master_slave`) rather than being copied,
//! because a second copy is the exact defect the Python consolidation removed." The second
//! caller is the FEA hanging-node fold (`hamilton_fem::solve`), so the move happened rather than
//! the copy, and `hamilton_mec` re-exports from here. That is the whole reason the debt was
//! written down as a CONDITION rather than as a wish.
//!
//! # Why it exists at all
//!
//! Both solvers used to eliminate one constraint at a time in DECLARATION ORDER. That resolves
//! only if the list happens to be topologically sorted, and nothing produces it that way.
//! Measured on a 5x5 unit-quad grid carrying `A = 0.3x - 0.7y` (exactly representable, so a
//! correct fold cannot move the solution): independent constraints left `max|A - exact|` at
//! 4e-16, while the chain `12 <- {11, 13}` with `11 <- {6, 16}` gave 5.500e-01 declared one way
//! and 2.983e-07 the other. Two different wrong answers for one problem is what identifies the
//! elimination rather than the physics.
//!
//! # A cycle is NOT automatically contradictory
//!
//! Refusing one was the Python's own earlier mistake. With `A_c = W_cc A_c + W_cf A_f` the
//! resolved rows are `A_c = (I - W_cc)^-1 W_cf A_f`, unique whenever `I - W_cc` is invertible.
//! Only a SINGULAR `I - W_cc` is genuinely unsolvable, and that alone is refused.
//! case_a under `--set mesh.max_aspect=4` carries exactly this shape: of 3892 hanging nodes 4
//! are chained, and those 4 form two mutual pairs.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

/// One `(slave, [(master, weight), ...])` relation.
pub type Constraint = (usize, Vec<(usize, f64)>);

/// The one refusal: a constraint set no fold can satisfy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConstraintError {
    /// `I - W_cc` is singular over the named closure.
    Singular { nodes: Vec<usize> },
}

impl fmt::Display for ConstraintError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Singular { nodes } => {
                let loop_ = nodes
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
                    .join(" -> ");
                write!(
                    f,
                    "hanging constraints form a SINGULAR system over nodes ({loop_}): the \
                     relations are mutually dependent with no solution in terms of free DOFs, so \
                     no fold can satisfy them. This is not a chain (chains resolve); it is a \
                     contradictory constraint set and the mesh that produced it is wrong."
                )
            }
        }
    }
}

impl std::error::Error for ConstraintError {}

/// Rewrite every constraint in terms of UNCONSTRAINED targets only, chains and cycles alike.
///
/// A constraint set with no chains at all is returned unchanged, so meshes that never had the
/// defect stay bit-for-bit. Only the nodes actually entangled are solved; every other row comes
/// back untouched.
///
/// # Errors
/// [`ConstraintError::Singular`].
pub fn resolve_constraint_chains(
    constraints: &[Constraint],
) -> Result<Vec<Constraint>, ConstraintError> {
    // The Python builds a DICT keyed by slave, which silently keeps only the LAST relation for a
    // repeated slave -- but only on the entangled path, because the un-entangled path returns the
    // input list untouched. Both halves are reproduced: `declared` is keyed and ordered by first
    // appearance, and the early return below hands back the original slice.
    let mut order_of_slave: Vec<usize> = Vec::new();
    let mut declared: BTreeMap<usize, Vec<(usize, f64)>> = BTreeMap::new();
    for (h, masters) in constraints {
        if declared.insert(*h, masters.clone()).is_none() {
            order_of_slave.push(*h);
        }
    }

    let mut entangled: BTreeSet<usize> = declared
        .iter()
        .filter(|(_, ms)| ms.iter().any(|(m, _)| declared.contains_key(m)))
        .map(|(h, _)| *h)
        .collect();
    if entangled.is_empty() {
        return Ok(constraints.to_vec());
    }

    // Grow to the closure: a node the entangled set depends on must be solved with it.
    let mut frontier: Vec<usize> = entangled.iter().copied().collect();
    while let Some(h) = frontier.pop() {
        for (m, _) in &declared[&h] {
            if declared.contains_key(m) && entangled.insert(*m) {
                frontier.push(*m);
            }
        }
    }

    let solved: Vec<usize> = entangled.iter().copied().collect(); // BTreeSet == sorted
    let pos: BTreeMap<usize, usize> = solved.iter().enumerate().map(|(i, h)| (*h, i)).collect();
    let n = solved.len();
    let free_targets: Vec<usize> = solved
        .iter()
        .flat_map(|h| declared[h].iter())
        .filter(|(m, _)| !declared.contains_key(m))
        .map(|(m, _)| *m)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    let fpos: BTreeMap<usize, usize> = free_targets
        .iter()
        .enumerate()
        .map(|(j, m)| (*m, j))
        .collect();

    let nf = free_targets.len();
    let mut a_mat = vec![vec![0.0_f64; n]; n];
    for (i, row) in a_mat.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    let mut b_mat = vec![vec![0.0_f64; nf]; n];
    for h in &solved {
        let i = pos[h];
        for (m, w) in &declared[h] {
            if let Some(&j) = pos.get(m) {
                a_mat[i][j] -= w;
            } else {
                // A master that is constrained but outside the closure cannot happen -- the
                // closure grew to include every constrained dependency above.
                b_mat[i][fpos[m]] += w;
            }
        }
    }

    let x_mat = solve_dense(&mut a_mat, &mut b_mat).ok_or_else(|| ConstraintError::Singular {
        nodes: solved.clone(),
    })?;

    let mut out = Vec::with_capacity(order_of_slave.len());
    for h in &order_of_slave {
        match pos.get(h) {
            Some(&i) => {
                // `np.flatnonzero` drops EXACT zeros, so a resolved weight that cancelled to 0.0
                // leaves no entry at all. Reproduced: a master with a zero weight is not a master.
                let row = (0..nf)
                    .filter(|&j| x_mat[i][j] != 0.0)
                    .map(|j| (free_targets[j], x_mat[i][j]))
                    .collect();
                out.push((*h, row));
            }
            None => out.push((*h, declared[h].clone())),
        }
    }
    Ok(out)
}

/// Solve `A X = B` in place by Gaussian elimination with partial pivoting; `None` if singular.
///
/// LAPACK's `gesv` -- what `numpy.linalg.solve` calls -- is the same algorithm with the same
/// pivot rule, and declares singularity on an EXACTLY zero pivot. The criterion is copied rather
/// than softened to a tolerance: a near-singular system that numpy solves must be solved here
/// too, or the port would refuse meshes the Python accepts.
fn solve_dense(a: &mut [Vec<f64>], b: &mut [Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let n = a.len();
    let nf = if n == 0 { 0 } else { b[0].len() };
    for col in 0..n {
        let (piv, best) = (col..n).fold((col, 0.0_f64), |(pi, pv), r| {
            let v = a[r][col].abs();
            if v > pv {
                (r, v)
            } else {
                (pi, pv)
            }
        });
        if best == 0.0 {
            return None;
        }
        a.swap(col, piv);
        b.swap(col, piv);
        for r in (col + 1)..n {
            let f = a[r][col] / a[col][col];
            if f == 0.0 {
                continue;
            }
            let (pivot_row, target) = a.split_at_mut(r);
            for (dst, src) in target[0][col..n].iter_mut().zip(&pivot_row[col][col..n]) {
                *dst -= f * src;
            }
            let (pivot_b, target_b) = b.split_at_mut(r);
            for (dst, src) in target_b[0][..nf].iter_mut().zip(&pivot_b[col][..nf]) {
                *dst -= f * src;
            }
        }
    }
    let mut x = vec![vec![0.0_f64; nf]; n];
    for col in (0..n).rev() {
        for c in 0..nf {
            let mut acc = b[col][c];
            for k in (col + 1)..n {
                acc -= a[col][k] * x[k][c];
            }
            x[col][c] = acc / a[col][col];
        }
    }
    Some(x)
}

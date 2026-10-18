//! Fixed values, mirroring `tests/unit/test_dirichlet_triplets.py`.

use crate::linear::constrained::fixed_values::{apply_dirichlet_triplets, DirichletGroup};
use crate::sparse::Coo;

fn dense(rows: &[u32], cols: &[u32], vals: &[f64], n: usize) -> Vec<Vec<f64>> {
    let coo = Coo::from_triplets(n, n, rows.to_vec(), cols.to_vec(), vals.to_vec()).unwrap();
    let csr = coo.to_csr().unwrap();
    (0..n)
        .map(|r| {
            (0..n)
                .map(|c| {
                    let (lo, hi) = (csr.indptr()[r] as usize, csr.indptr()[r + 1] as usize);
                    csr.indices()[lo..hi]
                        .binary_search(&(c as u32))
                        .map_or(0.0, |k| csr.values()[lo + k])
                })
                .collect()
        })
        .collect()
}

#[test]
fn a_pinned_row_becomes_the_identity_row_and_the_rhs_takes_the_value() {
    let rows = vec![0, 0, 1, 1, 2];
    let cols = vec![0, 1, 0, 1, 2];
    let vals = vec![2.0, -1.0, -1.0, 2.0, 3.0];
    let mut rhs = vec![0.0; 3];
    let (r, c, v) = apply_dirichlet_triplets(
        rows,
        cols,
        vals,
        &mut rhs,
        &[DirichletGroup {
            value: 5.0,
            indices: vec![1],
        }],
        0.0,
    );
    let d = dense(&r, &c, &v, 3);
    assert_eq!(d[1], vec![0.0, 1.0, 0.0], "pinned row is not the identity");
    assert_eq!(rhs[1], 5.0);
    // The COLUMN of a pinned index is deliberately NOT eliminated -- this is row-pinning, not
    // static condensation, so row 0 keeps its -1 coupling.
    assert_eq!(d[0], vec![2.0, -1.0, 0.0]);
}

#[test]
fn the_bias_is_added_to_every_prescribed_value() {
    let mut rhs = vec![0.0; 3];
    let _ = apply_dirichlet_triplets(
        vec![0],
        vec![0],
        vec![1.0],
        &mut rhs,
        &[DirichletGroup {
            value: 2.0,
            indices: vec![0, 2],
        }],
        0.5,
    );
    assert_eq!(rhs[0], 2.5);
    assert_eq!(rhs[2], 2.5);
}

#[test]
fn an_empty_index_group_is_a_no_op() {
    let mut rhs = vec![7.0];
    let (r, c, v) = apply_dirichlet_triplets(
        vec![0],
        vec![0],
        vec![1.0],
        &mut rhs,
        &[DirichletGroup {
            value: 3.0,
            indices: vec![],
        }],
        0.0,
    );
    assert_eq!((r, c, v), (vec![0], vec![0], vec![1.0]));
    assert_eq!(rhs, vec![7.0], "an empty group wrote to the rhs");
}

#[test]
fn duplicate_indices_are_collapsed() {
    // The Python's `np.unique`. Without it a repeated index appends the unit diagonal twice and
    // `to_csr` SUMS duplicates, so the pinned row would read 2 instead of 1.
    let mut rhs = vec![0.0; 2];
    let (r, c, v) = apply_dirichlet_triplets(
        vec![0],
        vec![0],
        vec![4.0],
        &mut rhs,
        &[DirichletGroup {
            value: 1.0,
            indices: vec![0, 0, 0],
        }],
        0.0,
    );
    assert_eq!(dense(&r, &c, &v, 2)[0], vec![1.0, 0.0]);
}

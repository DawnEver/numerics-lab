//! Pinned rows, mirroring `tests/unit/test_dirichlet_triplets.py`'s pin_rows half.

use crate::linear::constrained::pin_rows::pin_rows;
use crate::sparse::Coo;

#[test]
fn pin_rows_replaces_each_pinned_row_by_the_identity_row() {
    let mut coo = Coo::new(3, 3);
    for (r, row) in [[2.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 2.0]]
        .iter()
        .enumerate()
    {
        for (c, val) in row.iter().enumerate() {
            if *val != 0.0 {
                coo.push(r as u32, c as u32, *val).unwrap();
            }
        }
    }
    let pinned = pin_rows(&coo.to_csr().unwrap(), &[0, 2]).unwrap();
    let e = |i: usize| {
        let mut v = vec![0.0; 3];
        v[i] = 1.0;
        v
    };
    assert_eq!(pinned.matvec(&e(0)).unwrap(), vec![1.0, -1.0, 0.0]);
    assert_eq!(pinned.matvec(&e(1)).unwrap(), vec![0.0, 2.0, 0.0]);
    assert_eq!(pinned.matvec(&e(2)).unwrap(), vec![0.0, -1.0, 1.0]);
    assert!(
        pinned.check_invariants(),
        "pin_rows broke the CSR invariants"
    );
}

#[test]
fn pin_rows_refuses_an_index_outside_the_matrix() {
    let (a, _) = super::spd_system(4);
    assert!(
        pin_rows(&a, &[9]).is_err(),
        "an out-of-range pin was accepted"
    );
}

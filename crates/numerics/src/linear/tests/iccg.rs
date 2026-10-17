use super::{assert_close, nonsymmetric_system, spd_system};
use crate::linear::{direct, solve_linear_system, Preconditioner};
use crate::sparse::Coo;

#[test]
fn iccg_and_auto_recover_the_direct_solution() {
    let (a, b) = spd_system(40);
    let exact = direct::solve(&a, &b).unwrap();
    for name in ["iccg", "auto"] {
        let actual = solve_linear_system(&a, &b, Some(name), None, None).unwrap();
        assert_close(&actual, &exact, 1e-10, name);
    }
    let m = Preconditioner::build("ic0", &a).unwrap();
    assert_close(&m.apply(&b), &exact, 1e-12, "tridiagonal IC0 exact");
}

#[test]
fn auto_falls_back_for_nonsymmetric_operator() {
    let (a, b) = nonsymmetric_system(20);
    assert!(Preconditioner::build("ic0", &a).is_err());
    let exact = direct::solve(&a, &b).unwrap();
    let actual = solve_linear_system(&a, &b, Some("auto"), None, None).unwrap();
    assert_close(&actual, &exact, 1e-12, "auto direct fallback");
}

#[test]
fn ic0_rejects_indefinite_and_accepts_only_relative_roundoff() {
    for scale in [1e-20, 1.0, 1e20] {
        for asymmetry in [1e-15, 1e-8] {
            let mut coo = Coo::new(2, 2);
            for (r, c, v) in [
                (0, 0, 3.0),
                (1, 1, 3.0),
                (0, 1, -1.0),
                (1, 0, -1.0 + asymmetry),
            ] {
                coo.push(r, c, v * scale).unwrap();
            }
            assert_eq!(
                Preconditioner::build("ic0", &coo.to_csr().unwrap()).is_ok(),
                asymmetry < 1e-14
            );
        }
    }
    let mut coo = Coo::new(2, 2);
    coo.push(0, 0, 1.0).unwrap();
    coo.push(1, 1, -1.0).unwrap();
    assert!(Preconditioner::build("ic0", &coo.to_csr().unwrap()).is_err());
}

#[test]
fn explicit_absolute_tolerance_can_stop_a_warm_seed() {
    let (a, b) = spd_system(20);
    let initial = vec![0.0; 20];
    let outcome = crate::linear::Method::Cg
        .solve_with_tolerances(&a, &b, Some(&initial), None, 1e-10, 100.)
        .unwrap();
    assert_eq!(outcome.iterations, 0);
    assert!(crate::linear::Method::Cg
        .solve_with_tolerances(&a, &b, None, None, -1.0, 0.0)
        .is_err());
}

#[test]
fn iccg_lifts_nonzero_identity_rows_without_altering_the_free_operator() {
    let mut coo = Coo::new(3, 3);
    for (r, c, v) in [
        (0, 0, 1.),
        (1, 0, -2.),
        (1, 1, 4.),
        (1, 2, -1.),
        (2, 0, 3.),
        (2, 1, -1.),
        (2, 2, 5.),
    ] {
        coo.push(r, c, v).unwrap();
    }
    let matrix = coo.to_csr().unwrap();
    let rhs = [2., 1., 3.];
    let expected = direct::solve(&matrix, &rhs).unwrap();
    let actual = solve_linear_system(&matrix, &rhs, Some("iccg"), None, None).unwrap();
    assert_close(&actual, &expected, 1e-12, "nonzero identity-row lifting");
}

//! The preconditioners: they accelerate and they do NOT move the answer.
//!
//! The SSOR iteration-count assertions are the interesting ones. `ssor.py` claims for itself that
//! SSOR "roughly halves-or-better the CG iteration count (52->23, 77->31, 102->41 at
//! N = 256/576/1024) while Jacobi leaves it unchanged". That is a property of the PRECONDITIONER,
//! not a timing, so it is safe to assert on a shared box — the same fixture gives the same count
//! on any machine.

use super::{assert_close, laplacian_2d, spd_system};
use crate::linear::krylov::Method;
use crate::linear::precond::Preconditioner;
use crate::linear::{direct, solve_linear_system};

const PRECONDS: [&str; 3] = ["jacobi", "ilu", "ssor"];

#[test]
fn no_preconditioner_changes_the_converged_solution() {
    let (a, b) = spd_system(60);
    let reference = direct::solve(&a, &b).unwrap();
    for solver in ["cg", "gmres"] {
        for p in PRECONDS {
            let x = solve_linear_system(&a, &b, Some(solver), Some(p), None)
                .unwrap_or_else(|e| panic!("{solver}+{p}: {e}"));
            assert_close(&x, &reference, 1e-7, &format!("{solver}+{p}"));
        }
    }
}

#[test]
fn qmr_works_with_a_non_symmetric_preconditioner() {
    // ILU(0) is the one preconditioner whose transpose differs from itself, and `qmr` is the one
    // method that applies it. Without `apply_transpose` this converges to the wrong answer or not
    // at all, and nothing else in the suite would notice.
    let (a, b) = super::nonsymmetric_system(40);
    let reference = direct::solve(&a, &b).unwrap();
    let x = solve_linear_system(&a, &b, Some("qmr"), Some("ilu"), None).expect("qmr + ilu");
    assert_close(&x, &reference, 1e-7, "qmr+ilu");
}

fn cg_iterations(side: usize, preconditioner: Option<&str>) -> usize {
    let (a, b) = laplacian_2d(side);
    let m = preconditioner.map(|name| Preconditioner::build(name, &a).unwrap());
    Method::Cg
        .solve(&a, &b, None, m.as_ref())
        .expect("the 2D Laplacian is SPD")
        .iterations
}

#[test]
fn ssor_roughly_halves_the_cg_count_on_the_2d_laplacian() {
    // `ssor.py` measured 52->23, 77->31, 102->41 at N = 256/576/1024. This port measures
    // 57->26, 85->36, 113->47 on the same operator — the same ratios (2.19 / 2.36 / 2.40 against
    // the Python's 2.26 / 2.48 / 2.49), with every count ~10% higher because the stop here is the
    // TRUE residual where scipy carries a recurrence that drifts slightly optimistic.
    //
    // What is ASSERTED is the ratio, not the count. The count is a function of the stopping rule
    // and would have to be re-frozen every time that rule is touched; the ratio is the claim the
    // preconditioner actually makes for itself, and it is what a regression in SSOR would break.
    for side in [16usize, 24, 32] {
        let plain = cg_iterations(side, None);
        let ssor = cg_iterations(side, Some("ssor"));
        assert!(
            (plain as f64) / (ssor as f64) >= 1.8,
            "N = {}: SSOR took {ssor} against plain {plain}; the claimed >=2x acceleration is gone",
            side * side
        );
    }
    assert!(
        cg_iterations(16, None) < cg_iterations(32, None),
        "the CG count does not grow with the problem size, so the fixture is not the Laplacian"
    );
}

#[test]
fn jacobi_is_a_no_op_on_the_constant_diagonal_laplacian() {
    // The control for the test above. Jacobi scales every row by the same number on this
    // operator, so the Krylov space is identical and the count cannot move. If it DOES move, the
    // "SSOR beats Jacobi" claim would be measuring something other than the preconditioner.
    for side in [16usize, 24, 32] {
        assert_eq!(
            cg_iterations(side, Some("jacobi")),
            cg_iterations(side, None),
            "N = {}: Jacobi moved the count on a constant diagonal",
            side * side
        );
    }
}

#[test]
fn ilu0_also_accelerates_the_laplacian() {
    for side in [16usize, 24, 32] {
        let plain = cg_iterations(side, None);
        let ilu = cg_iterations(side, Some("ilu"));
        assert!(
            ilu < plain,
            "N = {}: ILU(0) took {ilu} against plain {plain}",
            side * side
        );
    }
}

#[test]
fn ssor_is_a_symmetric_operator() {
    // The property that makes SSOR the right auxiliary preconditioner for CG, and the one ILU
    // does not have: M must be symmetric or CG stops being a valid method on the SPD operator.
    let (a, _) = laplacian_2d(6);
    let n = a.shape().0;
    let m = Preconditioner::ssor(&a, 1.0).unwrap();
    for i in 0..n {
        for j in 0..n {
            let mut ei = vec![0.0; n];
            ei[i] = 1.0;
            let mut ej = vec![0.0; n];
            ej[j] = 1.0;
            let mij = m.apply(&ej)[i];
            let mji = m.apply(&ei)[j];
            assert!(
                (mij - mji).abs() <= 1e-12 * mij.abs().max(1.0),
                "M[{i},{j}] = {mij} but M[{j},{i}] = {mji}"
            );
        }
    }
}

#[test]
fn the_ilu0_transpose_apply_is_the_transpose_of_its_apply() {
    // `apply_transpose` is only exercised by qmr, so it needs its own floor: `<M^T u, v>` must
    // equal `<u, M v>`.
    let (a, _) = super::nonsymmetric_system(12);
    let n = a.shape().0;
    let m = Preconditioner::ilu0(&a).unwrap();
    let u: Vec<f64> = (0..n).map(|i| 1.0 + (i as f64) * 0.3).collect();
    let v: Vec<f64> = (0..n).map(|i| 2.0 - (i as f64) * 0.1).collect();
    let left: f64 = m
        .apply_transpose(&u)
        .iter()
        .zip(&v)
        .map(|(a, b)| a * b)
        .sum();
    let right: f64 = u.iter().zip(m.apply(&v)).map(|(a, b)| a * b).sum();
    assert!(
        (left - right).abs() <= 1e-10 * left.abs().max(1.0),
        "<M^T u, v> = {left} but <u, M v> = {right}"
    );
}

#[test]
fn an_unknown_preconditioner_is_refused_by_name() {
    let (a, b) = spd_system(8);
    let err = solve_linear_system(&a, &b, Some("cg"), Some("nope"), None)
        .expect_err("an unknown preconditioner must be refused");
    assert!(
        err.to_string().contains("unknown preconditioner"),
        "the refusal does not name the problem: {err}"
    );
}

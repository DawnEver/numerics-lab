//! Every Krylov peer reaches the SAME solution as `direct` — the linear analogue of the
//! nonlinear "converge-to-reference" contract, mirroring `test_linear_solvers.py`.

use super::{assert_close, nonsymmetric_system, spd_system};
use crate::linear::krylov::{Method, KRYLOV_MAXITER_FACTOR, KRYLOV_RTOL};
use crate::linear::{direct, solve_linear_system, LinearSolveError};

const ALL: [&str; 5] = ["cg", "minres", "gmres", "bicgstab", "qmr"];
const GENERAL: [&str; 3] = ["gmres", "bicgstab", "qmr"];

#[test]
fn every_krylov_backend_reaches_the_direct_solution_on_the_spd_system() {
    let (a, b) = spd_system(60);
    let reference = direct::solve(&a, &b).expect("the SPD system factorises");
    for name in ALL {
        let x = solve_linear_system(&a, &b, Some(name), None, None)
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_close(&x, &reference, 1e-7, name);
    }
}

#[test]
fn the_general_backends_reach_the_direct_solution_on_a_non_symmetric_system() {
    let (a, b) = nonsymmetric_system(60);
    let reference = direct::solve(&a, &b).expect("the non-symmetric system factorises");
    for name in GENERAL {
        let x = solve_linear_system(&a, &b, Some(name), None, None)
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_close(&x, &reference, 1e-7, name);
    }
}

#[test]
fn a_warm_start_still_converges_to_the_direct_solution() {
    let (a, b) = spd_system(60);
    let reference = direct::solve(&a, &b).unwrap();
    let seed: Vec<f64> = reference.iter().map(|v| v + 1e-3).collect();
    let x = solve_linear_system(&a, &b, Some("cg"), None, Some(&seed)).unwrap();
    assert_close(&x, &reference, 1e-6, "cg warm start");
}

#[test]
fn a_warm_start_at_the_answer_converges_in_zero_iterations() {
    // The floor under the warm start itself: if `x0` were ignored, this would still pass on the
    // solution but NOT on the count.
    let (a, b) = spd_system(40);
    let reference = direct::solve(&a, &b).unwrap();
    let out = Method::Cg
        .solve(&a, &b, Some(&reference), None)
        .expect("an exact start is converged");
    assert_eq!(out.iterations, 0, "x0 was ignored");
}

#[test]
fn the_achieved_residual_meets_the_declared_rtol() {
    let (a, b) = spd_system(60);
    for name in ALL {
        let m = Method::from_name(name).unwrap();
        let out = m
            .solve(&a, &b, None, None)
            .unwrap_or_else(|e| panic!("{name}: {e:?}"));
        assert!(
            out.residual <= KRYLOV_RTOL * 10.0,
            "{name}: residual {:e} misses the declared rtol {KRYLOV_RTOL:e}",
            out.residual
        );
    }
}

#[test]
fn a_zero_rhs_converges_immediately_rather_than_burning_the_ceiling() {
    // atol = 0 makes the stop purely relative, so a zero rhs has a zero target — and x = 0 meets
    // it exactly. This is the boundary that an absolute floor would have blurred.
    let (a, _) = spd_system(20);
    let b = vec![0.0; 20];
    let out = Method::Cg.solve(&a, &b, None, None).unwrap();
    assert_eq!(out.iterations, 0);
    assert!(out.x.iter().all(|v| *v == 0.0));
}

#[test]
fn cg_and_minres_are_the_symmetric_assuming_backends() {
    for name in ALL {
        let m = Method::from_name(name).unwrap();
        assert_eq!(
            m.requires_symmetric(),
            matches!(m, Method::Cg | Method::Minres),
            "{name} carries the wrong symmetry tag"
        );
    }
}

#[test]
fn a_non_converging_krylov_solve_raises_rather_than_returning_a_wrong_x() {
    // A deliberately hostile operator: a 2x2 rotation-like block CG cannot handle. The point is
    // the SHAPE of the failure, not the matrix — a wrong `x` returned as a success is the defect
    // this whole seam exists to refuse.
    let mut coo = crate::sparse::Coo::new(2, 2);
    coo.push(0, 1, 1.0).unwrap();
    coo.push(1, 0, -1.0).unwrap();
    let a = coo.to_csr().unwrap();
    let b = vec![1.0, 1.0];
    let err = solve_linear_system(&a, &b, Some("cg"), None, None)
        .expect_err("cg must not certify a solution it never reached");
    match err {
        LinearSolveError::Krylov { method, .. } => assert_eq!(method, "cg"),
        other => panic!("expected a Krylov failure, got {other}"),
    }
}

#[test]
fn the_iteration_ceiling_is_four_n() {
    let (a, b) = spd_system(8);
    let out = Method::Gmres.solve(&a, &b, None, None).unwrap();
    assert!(
        out.iterations <= KRYLOV_MAXITER_FACTOR * 8,
        "a converged solve exceeded its own ceiling"
    );
}

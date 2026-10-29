//! The seam itself: the registry, the resolver, and BOTH halves of the singular guard.

use super::{assert_close, duplicate_row_system, spd_system, zero_row_system};
use crate::linear::{
    assumption_violated, check_direct_residual, requires_symmetric, solve_linear_system,
    LinearSolveError, LinearSolver, DIRECT_RESIDUAL_RTOL, KNOWN_LINEAR_SOLVERS,
    KNOWN_PRECONDITIONERS,
};

#[test]
fn the_registry_covers_the_roadmap_backends() {
    // The Python's floor, carried over: the registry must never LOSE a roadmap backend.
    // 'pardiso' is deliberately absent here — see the module docs for why that is an absence
    // with a reason rather than an omission.
    for name in ["direct", "cg", "minres", "gmres", "bicgstab", "qmr"] {
        assert!(
            KNOWN_LINEAR_SOLVERS.contains(&name),
            "{name} left the registry"
        );
        LinearSolver::resolve(Some(name)).unwrap_or_else(|e| panic!("{name}: {e}"));
    }
    assert_eq!(KNOWN_PRECONDITIONERS, ["ic0", "ilu", "jacobi", "ssor"]);
}

#[test]
fn none_resolves_to_direct() {
    assert_eq!(LinearSolver::resolve(None).unwrap(), LinearSolver::Direct);
}

#[test]
fn an_unknown_backend_is_refused_by_name() {
    let err = LinearSolver::resolve(Some("nope")).expect_err("unknown backends are refused");
    assert!(
        err.to_string().contains("unknown linear solver"),
        "the refusal does not name the problem: {err}"
    );
    // The absence of 'pardiso' is REPORTED, not silent — the Python hints the install, and this
    // side hints why the name does not exist at all.
    let err = LinearSolver::resolve(Some("pardiso")).unwrap_err();
    assert!(err.to_string().contains("pypardiso"), "{err}");
}

#[test]
fn requires_symmetric_tags_cg_and_minres_and_nothing_else() {
    for name in KNOWN_LINEAR_SOLVERS {
        assert_eq!(
            requires_symmetric(name),
            name == "cg" || name == "iccg" || name == "minres",
            "{name} carries the wrong symmetry tag"
        );
    }
    assert!(!requires_symmetric("nope"), "an unknown name is not a tag");
}

#[test]
fn a_symmetric_only_solver_on_an_asymmetric_operator_is_detectable() {
    // The failure this pairing exists to catch is SILENT: cg on a non-symmetric operator
    // converges to the wrong answer rather than refusing. Making the mismatch askable is the
    // whole point of keeping the tag next to `Csr::is_symmetric`.
    let (spd, _) = spd_system(20);
    let (asym, _) = super::nonsymmetric_system(20);
    let cg = LinearSolver::resolve(Some("cg")).unwrap();
    let gmres = LinearSolver::resolve(Some("gmres")).unwrap();
    assert!(assumption_violated(cg, &asym, 1e-12));
    assert!(!assumption_violated(cg, &spd, 1e-12));
    assert!(
        !assumption_violated(gmres, &asym, 1e-12),
        "gmres assumes nothing about symmetry, so nothing can be violated"
    );
}

#[test]
fn the_direct_backend_solves_the_spd_system() {
    let (a, b) = spd_system(60);
    let x = solve_linear_system(&a, &b, None, None, None).expect("direct");
    let ax = a.matvec(&x).unwrap();
    assert_close(&ax, &b, 1e-10, "A x vs b");
    assert_eq!(
        x,
        solve_linear_system(&a, &b, Some("direct"), None, None).unwrap(),
        "None and 'direct' are not the same route"
    );
}

#[test]
fn a_structurally_singular_system_raises_rather_than_returning_nans() {
    let (a, b) = zero_row_system();
    let err = solve_linear_system(&a, &b, Some("direct"), None, None)
        .expect_err("a zero row must be refused");
    assert!(
        matches!(
            err,
            LinearSolveError::Direct(_)
                | LinearSolveError::NonFinite { .. }
                | LinearSolveError::DirectResidual { .. }
        ),
        "the refusal is the wrong shape: {err}"
    );
}

#[test]
fn a_rank_deficient_system_is_refused_by_one_of_the_two_halves() {
    // Two identical rows. The Python's SuperLU emits a MatrixRankWarning and may return NaNs OR
    // finite garbage; faer factorises and produces non-finite values. Either way the seam must
    // refuse — which half fires is an implementation detail, that it fires is not.
    let (a, b) = duplicate_row_system();
    for route in [None, Some("direct")] {
        let err = solve_linear_system(&a, &b, route, None, None)
            .expect_err("a rank-deficient system must be refused");
        assert!(
            !matches!(err, LinearSolveError::UnknownSolver { .. }),
            "{err}"
        );
    }
}

#[test]
fn a_finite_but_wrong_direct_solution_is_rejected_by_the_residual_guard() {
    // THE SUSPENDERS, alone. This is the case that actually shipped as `status=ok` garbage: a
    // vector that passes every finiteness check and does not satisfy the system. The Python
    // reaches it by monkeypatching the registry; here the guard is public, so the fabricated
    // solution goes straight to it.
    let (a, b) = spd_system(8);
    let garbage = vec![1.0; 8];
    let err = check_direct_residual(&a, &b, &garbage).expect_err("finite garbage must be rejected");
    match err {
        LinearSolveError::DirectResidual { residual, bound } => {
            assert!(residual > bound);
            assert_eq!(bound, DIRECT_RESIDUAL_RTOL);
        }
        other => panic!("expected a residual refusal, got {other}"),
    }
    // The floor under the guard itself: the true solution must PASS, or the bound is meaningless.
    let good = crate::linear::direct::solve(&a, &b).unwrap();
    check_direct_residual(&a, &b, &good).expect("a well-conditioned solve clears the bound");
}

#[test]
fn a_zero_rhs_is_not_judged_by_the_relative_residual() {
    // ||b|| = 0 has no relative residual to compare against. Dividing anyway would make every
    // zero-rhs direct solve a refusal.
    let (a, _) = spd_system(8);
    check_direct_residual(&a, &[0.0; 8], &[0.0; 8]).expect("a zero rhs is not a failure");
}

#[test]
fn the_residual_guard_does_not_fire_on_a_krylov_backend() {
    // Deliberate: the Krylov backends already refuse on their own convergence test, and layering
    // a second (looser) residual on top would clash with their stop criterion.
    let (a, b) = spd_system(30);
    for name in ["cg", "gmres", "bicgstab", "minres", "qmr"] {
        solve_linear_system(&a, &b, Some(name), None, None)
            .unwrap_or_else(|e| panic!("{name}: {e}"));
    }
}

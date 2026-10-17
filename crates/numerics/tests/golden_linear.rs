//! The linear-solve port judged against the PYTHON, through the committed golden vectors.
//!
//! The golden is written by `tests/unit/test_linear_port_golden.py`, which separately
//! asserts the live Python still produces it — that half is the freeze, this half is the port.
//!
//! # The one claim this file does NOT make
//!
//! The Python's `direct` backend is documented as bit-for-bit `scipy.sparse.linalg.spsolve`
//! (SuperLU). The Rust `direct` is `faer`'s sparse LU. **They are different backends**, and no
//! amount of care makes one bit-identical to the other: the pivot sequence, the column ordering
//! and the accumulation order all differ, and bit-identity is a property of an implementation
//! rather than of the mathematics. So this file CHARACTERIZES rather than equates, on two axes:
//!
//! * [`DIRECT_RESIDUAL_BAR`] — the port's own solution must satisfy `A x = b`. This is the
//!   axis that matters, because it is the property a user of the solve depends on, and it is
//!   checkable without reference to any other implementation.
//! * [`DIRECT_AGREEMENT_RTOL`] — and it must AGREE with the Python's answer. The fixtures are
//!   well-conditioned (`kappa ~ 1e1..1e2`), so two backward-stable factorisations agree to about
//!   `kappa * eps ~ 1e-14`. The bar is set at `1e-8`, six orders looser than that so a pivot-order
//!   difference can never red it, and still six orders tighter than the `1e-6` residual guard —
//!   which is to say, far too tight for a wrong answer to slip through.
//!
//! THE 2.3e-10 QUESTION IS SETTLED, and settling it changed nothing here. A 2026-09-04 sweep
//! measured `faer` departing from `spsolve` by up to **2.3e-10** on production-sized systems and
//! flagged it as above "the 1e-10 that was assumed". That 1e-10 was never this file's bar — it was
//! the assumed SIZE of the departure, and the bar has stood at 1e-8 throughout, so the measured
//! 2.3e-10 was inside it by two orders the whole time. The user accepted the departure on
//! 2026-09-05 ("accept for now"), which removes it as an objection and leaves the constant untouched.
//!
//! The wiring decision it did not reopen was reopened on 2026-09-26 by a SPEED finding: the old
//! crossover (31,901 -> 0.81x, 32,572 -> 1.70x, 35,430 -> 0.78x) was `SupernodalThreshold::AUTO`
//! picking the simplicial kernel. With supernodal forced, one reused numeric factor, Cholesky on an
//! exactly symmetric operator, one refinement step and a Schur border for dense rows,
//! `numerics::linear::direct::DirectFactorization` is the Python `direct` backend's default through
//! `gauss.linear.native_direct` (SciPy stays the logged fallback). The refinement step makes this
//! side agree with the Python BETTER than before, so the bars below are unchanged.

use numerics::sparse::{Coo, Csr};
use numerics::linear::{
    check_direct_residual, direct, solve_linear_system, LinearSolveError, LinearSolver,
    DIRECT_RESIDUAL_RTOL,
};
use serde_json::Value;
use std::path::PathBuf;

/// The port's own solution must satisfy the system to the direct guard's own bound.
const DIRECT_RESIDUAL_BAR: f64 = DIRECT_RESIDUAL_RTOL;

/// Agreement with the Python's `spsolve` answer. See the module docs for the arithmetic.
const DIRECT_AGREEMENT_RTOL: f64 = 1e-8;

/// Agreement for the Krylov peers. Both sides stop at a relative residual of `1e-10` but through
/// DIFFERENT Krylov spaces (scipy's recurrence-carried residual against this port's true one), so
/// the solutions agree to roughly `kappa * 1e-10` rather than exactly.
const KRYLOV_AGREEMENT_RTOL: f64 = 1e-7;

fn golden() -> Value {
    let path: PathBuf = [
        env!("CARGO_MANIFEST_DIR"),
        "tests",
        "goldens",
        "linear.json",
    ]
    .iter()
    .collect();
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "{} is a committed artifact, not a cache. Regenerate it deliberately with \
             `python -m tests.port_goldens` on the Python side and commit it. ({e})",
            path.display()
        )
    });
    serde_json::from_str(&text).expect("goldens/linear.json is valid JSON")
}

fn floats(v: &Value) -> Vec<f64> {
    v.as_array()
        .expect("an array")
        .iter()
        .map(|x| x.as_f64().expect("a number"))
        .collect()
}

fn matrix_of(case: &Value) -> Csr {
    let m = &case["matrix"];
    let n = m["n"].as_u64().expect("n") as usize;
    let rows = m["rows"]
        .as_array()
        .expect("rows")
        .iter()
        .map(|v| v.as_u64().expect("row") as u32)
        .collect();
    let cols = m["cols"]
        .as_array()
        .expect("cols")
        .iter()
        .map(|v| v.as_u64().expect("col") as u32)
        .collect();
    Coo::from_triplets(n, n, rows, cols, floats(&m["values"]))
        .expect("the golden's triplets are in range")
        .to_csr()
        .expect("the golden's triplets convert")
}

fn agree(got: &[f64], want: &[f64], rtol: f64, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let tol = rtol * w.abs().max(1.0);
        assert!(
            (g - w).abs() <= tol,
            "{what}[{i}]: rust {g} vs python {w} (|d| = {:e}, tol {tol:e})",
            (g - w).abs()
        );
    }
}

#[test]
fn every_solved_fixture_agrees_with_the_python_on_every_backend() {
    let g = golden();
    let cases = g["cases"].as_object().expect("cases is an object");
    assert!(!cases.is_empty(), "the golden covers no case at all");

    for (name, case) in cases {
        if case["refused_by_the_singular_guard"] == Value::Bool(true) {
            continue;
        }
        let a = matrix_of(case);
        let b = floats(&case["rhs"]);
        for (backend, want) in case["solutions"].as_object().expect("solutions") {
            let want = floats(want);
            let (solver, precond) = match backend.strip_prefix("precond:") {
                // The generator runs every preconditioner through cg on a symmetric fixture and
                // gmres on a non-symmetric one; mirroring that choice here is what makes the two
                // sides comparable at all.
                Some(p) => {
                    let symmetric = case["symmetric"] == Value::Bool(true);
                    (if symmetric { "cg" } else { "gmres" }, Some(p))
                }
                None => (backend.as_str(), None),
            };
            let got = solve_linear_system(&a, &b, Some(solver), precond, None)
                .unwrap_or_else(|e| panic!("{name}.{backend}: {e}"));
            let rtol = if solver == "direct" {
                DIRECT_AGREEMENT_RTOL
            } else {
                KRYLOV_AGREEMENT_RTOL
            };
            agree(&got, &want, rtol, &format!("{name}.{backend}"));
        }
    }
}

#[test]
fn iccg_and_auto_agree_with_committed_python_direct_answers() {
    // This compares solutions, not a live Python ICCG trajectory or a JMAG result.
    let g = golden();
    let mut iccg_cases = 0;
    for (name, case) in g["cases"].as_object().expect("cases") {
        if case["refused_by_the_singular_guard"] == Value::Bool(true) {
            continue;
        }
        let a = matrix_of(case);
        let b = floats(&case["rhs"]);
        let want = floats(&case["solutions"]["direct"]);
        let auto = solve_linear_system(&a, &b, Some("auto"), None, None).unwrap();
        agree(&auto, &want, KRYLOV_AGREEMENT_RTOL, &format!("{name}.auto"));
        if numerics::linear::Preconditioner::build("ic0", &a).is_ok() {
            let iccg = solve_linear_system(&a, &b, Some("iccg"), None, None).unwrap();
            agree(&iccg, &want, KRYLOV_AGREEMENT_RTOL, &format!("{name}.iccg"));
            iccg_cases += 1;
        }
    }
    assert!(iccg_cases > 0, "no committed Python operator exercises IC0");
}

#[test]
fn the_ported_direct_backend_satisfies_the_system_it_was_given() {
    // The axis that does NOT reference the Python at all: whatever faer's pivots do, `A x = b`
    // must hold. This is the property a caller depends on, and the one the Python's `direct`
    // bit-identity claim was only ever a proxy for.
    let g = golden();
    for (name, case) in g["cases"].as_object().expect("cases") {
        if case["refused_by_the_singular_guard"] == Value::Bool(true) {
            continue;
        }
        let a = matrix_of(case);
        let b = floats(&case["rhs"]);
        let x = direct::solve(&a, &b).unwrap_or_else(|e| panic!("{name}: {e:?}"));
        check_direct_residual(&a, &b, &x).unwrap_or_else(|e| {
            panic!("{name}: the port's own LU misses {DIRECT_RESIDUAL_BAR:e}: {e}")
        });
    }
}

#[test]
fn every_fixture_the_python_refused_is_refused_here_too() {
    // The guard is the reason this seam exists. A port that solved what the Python refuses would
    // be shipping exactly the `status=ok` garbage the guard was written to stop.
    let g = golden();
    let mut refused_any = false;
    for (name, case) in g["cases"].as_object().expect("cases") {
        if case["refused_by_the_singular_guard"] != Value::Bool(true) {
            continue;
        }
        refused_any = true;
        let a = matrix_of(case);
        let b = floats(&case["rhs"]);
        for route in [None, Some("direct")] {
            let err = solve_linear_system(&a, &b, route, None, None).expect_err(&format!(
                "{name}: the port SOLVED a system the Python refuses"
            ));
            assert!(
                !matches!(err, LinearSolveError::UnknownSolver { .. }),
                "{name}: refused for the wrong reason: {err}"
            );
        }
    }
    assert!(
        refused_any,
        "no fixture is refused, so the singular guard is never exercised on this side either"
    );
}

#[test]
fn the_symmetry_tags_match_the_python() {
    let g = golden();
    for (name, want) in g["requires_symmetric"].as_object().expect("the tag map") {
        let solver = LinearSolver::resolve(Some(name)).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(
            solver.requires_symmetric(),
            want.as_bool().expect("a bool"),
            "{name}: the symmetry tag diverged from the Python"
        );
    }
}

#[test]
fn the_golden_covers_every_registered_backend_and_preconditioner() {
    // The Rust-side floor, mirroring the Python's. A golden narrower than the registry cannot
    // notice a backend that stopped working.
    let g = golden();
    let mut covered = std::collections::BTreeSet::new();
    for case in g["cases"].as_object().expect("cases").values() {
        for key in case["solutions"].as_object().expect("solutions").keys() {
            covered.insert(key.clone());
        }
    }
    for solver in ["direct", "cg", "minres", "gmres", "bicgstab", "qmr"] {
        assert!(covered.contains(solver), "no fixture exercises {solver}");
    }
    for precond in ["jacobi", "ilu", "ssor"] {
        assert!(
            covered.contains(&format!("precond:{precond}")),
            "no fixture exercises the {precond} preconditioner"
        );
    }
}

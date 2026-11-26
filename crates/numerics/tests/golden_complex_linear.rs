//! The complex direct solve judged against SciPy SuperLU, through the committed golden.
//!
//! Written by `tests/unit/test_complex_linear_port_golden.py`, which separately asserts the live
//! Python still produces it. As in `golden_linear.rs`, faer is a DIFFERENT factorisation from
//! SuperLU, so this file CHARACTERIZES: the port's solution must satisfy the system to the guard's
//! bound, and agree with SuperLU's to [`AGREEMENT_RTOL`] (the fixtures have `kappa ~ 1e1`, so two
//! backward-stable factorisations agree to `~1e-15`; the bar is the real golden's `1e-8`).

use numerics::linear::{solve_direct, solve_equilibrated, LinearSolveError};
use numerics::sparse::{c64, Coo, Csr, Scalar};
use serde_json::Value;
use std::path::PathBuf;

/// Agreement with SuperLU, relative to each reference entry's modulus.
const AGREEMENT_RTOL: f64 = 1e-8;
/// The absolute leg of the agreement, in the unit of the solution entries.
const AGREEMENT_ATOL: f64 = 1e-14;

fn golden() -> Value {
    let path: PathBuf = [
        env!("CARGO_MANIFEST_DIR"),
        "tests",
        "goldens",
        "linear_complex.json",
    ]
    .iter()
    .collect();
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "{} is a committed artifact; regenerate it deliberately with \
             tests/unit/test_complex_linear_port_golden.py::write and commit it. ({e})",
            path.display()
        )
    });
    serde_json::from_str(&text).expect("goldens/linear_complex.json is valid JSON")
}

fn complexes(v: &Value) -> Vec<c64> {
    let part = |k: &str| -> Vec<f64> {
        v[k].as_array()
            .expect("an array")
            .iter()
            .map(|x| x.as_f64().expect("a number"))
            .collect()
    };
    let (re, im) = (part("re"), part("im"));
    assert_eq!(re.len(), im.len());
    re.into_iter()
        .zip(im)
        .map(|(r, i)| c64::new(r, i))
        .collect()
}

fn indices(v: &Value) -> Vec<u32> {
    v.as_array()
        .expect("an array")
        .iter()
        .map(|x| u32::try_from(x.as_u64().expect("an index")).expect("fits u32"))
        .collect()
}

fn matrix_of(case: &Value) -> Csr<c64> {
    let m = &case["matrix"];
    let n = usize::try_from(m["n"].as_u64().expect("n")).expect("fits usize");
    Coo::from_triplets(
        n,
        n,
        indices(&m["rows"]),
        indices(&m["cols"]),
        complexes(&m["values"]),
    )
    .expect("the golden's triplets are in range")
    .to_csr()
    .expect("the golden's triplets convert")
}

#[test]
fn every_solved_complex_fixture_agrees_with_superlu() {
    let g = golden();
    let mut solved = 0;
    for (name, case) in g["cases"].as_object().expect("cases") {
        if case["refused_by_the_singular_guard"] == Value::Bool(true) {
            continue;
        }
        let (a, b) = (matrix_of(case), complexes(&case["rhs"]));
        let got = match case["solve"].as_str() {
            Some("equilibrated") => solve_equilibrated(&a, &b),
            Some("direct") => solve_direct(&a, &b),
            other => panic!("{name}: unknown solve {other:?}"),
        }
        .unwrap_or_else(|e| panic!("{name}: {e}"));
        let want = complexes(&case["solution"]);
        assert_eq!(got.len(), want.len(), "{name}: length");
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            let d = (*g - *w).modulus();
            assert!(
                d <= AGREEMENT_RTOL * w.modulus() + AGREEMENT_ATOL,
                "{name}[{i}]: rust {g:?} vs superlu {w:?} (|d| = {d:e})"
            );
        }
        solved += 1;
    }
    assert!(solved >= 4, "the golden lost fixtures: {solved} solved");
}

#[test]
fn every_complex_fixture_superlu_refused_is_refused_here_too() {
    let g = golden();
    let mut refused = 0;
    for (name, case) in g["cases"].as_object().expect("cases") {
        if case["refused_by_the_singular_guard"] != Value::Bool(true) {
            continue;
        }
        let err = solve_direct(&matrix_of(case), &complexes(&case["rhs"]))
            .expect_err(&format!("{name}: solved what the Python refused"));
        assert!(
            matches!(
                err,
                LinearSolveError::Direct(_)
                    | LinearSolveError::DirectResidual { .. }
                    | LinearSolveError::NonFinite { .. }
            ),
            "{name}: {err:?}"
        );
        refused += 1;
    }
    assert!(refused > 0, "no fixture exercises the singular guard");
}

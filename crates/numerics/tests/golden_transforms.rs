//! The Park/Clarke port judged against the PYTHON, through the committed golden vectors.
//!
//! Companion to `crates/core_num/tests/golden_chart.rs`; see that file for why the comparison
//! goes through a data file rather than the FFI. The golden is written by
//! `tests/unit/gauss/test_transforms_port_golden.py`, which separately asserts the live Python
//! still produces it — that half is the freeze, this half is the port.

use gauss::transforms::{clarke_forward, park_forward, park_inverse, phase_l_to_dq_l};
use serde_json::Value;
use std::path::PathBuf;

/// Both sides evaluate the same closed-form trigonometry in f64. The only admissible
/// difference is operation ORDER — numpy reaches the per-phase sum through a vectorised
/// reduction where the port writes the loop out — so this is a floor under reassociation, not
/// an accuracy budget.
const RTOL: f64 = 1e-13;

fn golden() -> Value {
    let path: PathBuf = [
        env!("CARGO_MANIFEST_DIR"),
        "tests",
        "goldens",
        "transforms.json",
    ]
    .iter()
    .collect();
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "{} is a committed artifact, not a cache. Regenerate it deliberately with \
             MOTRONICS_REGEN_PORT_GOLDENS=1 on the Python side and commit it. ({e})",
            path.display()
        )
    });
    serde_json::from_str(&text).expect("goldens/transforms.json is valid JSON")
}

fn flat(v: &Value) -> Vec<f64> {
    match v {
        Value::Number(n) => vec![n.as_f64().expect("finite number")],
        Value::Array(items) => items.iter().flat_map(flat).collect(),
        other => panic!("expected a number or an array, got {other}"),
    }
}

fn close(got: &[f64], want: &[f64], what: &str) {
    assert_eq!(
        got.len(),
        want.len(),
        "{what}: length {} vs golden {}",
        got.len(),
        want.len()
    );
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let tol = RTOL * w.abs().max(1.0);
        assert!(
            (g - w).abs() <= tol,
            "{what}[{i}]: rust {g} vs python {w} (|d| = {:e}, tol {tol:e})",
            (g - w).abs()
        );
    }
}

#[test]
fn every_case_in_the_golden_matches_the_port() {
    let g = golden();
    let cases = g["cases"].as_object().expect("cases is an object");
    assert!(!cases.is_empty(), "the golden covers no case at all");

    for (name, case) in cases {
        let n_phase = case["n_phase"].as_u64().expect("n_phase") as usize;
        let theta = flat(&case["theta_e"]);
        let x = flat(&case["x_phase"]);

        let dq0 = park_forward(&x, n_phase, &theta).unwrap();
        let want = &case["park_forward"];
        close(&dq0.d, &flat(&want["d"]), &format!("{name}.park_forward.d"));
        close(&dq0.q, &flat(&want["q"]), &format!("{name}.park_forward.q"));
        close(
            &dq0.zero,
            &flat(&want["zero"]),
            &format!("{name}.park_forward.zero"),
        );

        let back = park_inverse(&dq0, &theta, n_phase).unwrap();
        close(
            &back,
            &flat(&case["park_inverse"]),
            &format!("{name}.park_inverse"),
        );

        if let Some(cl) = case.get("clarke_forward") {
            let (alpha, beta) = clarke_forward(&x, theta.len()).unwrap();
            close(&alpha, &flat(&cl["alpha"]), &format!("{name}.clarke.alpha"));
            close(&beta, &flat(&cl["beta"]), &format!("{name}.clarke.beta"));
        }
    }
}

#[test]
fn the_dq_bridge_matches_the_python() {
    let g = golden();
    let b = &g["phase_L_to_dq_L"];
    let n_phase = b["n_phase"].as_u64().expect("n_phase") as usize;
    let theta = flat(&b["theta_e"]);
    let l = flat(&b["L_phase"]);
    let got = phase_l_to_dq_l(&l, n_phase, &theta).unwrap();
    close(&got, &flat(&b["L_dq"]), "phase_L_to_dq_L");
}

#[test]
fn the_golden_exercises_q_and_zero_and_more_than_three_phases() {
    // The floor, mirroring the Python's. A balanced-only fixture set leaves q and zero
    // identically zero, so a port that dropped either term would pass against it; and a
    // 3-phase-only set cannot tell a `2/N` scaling from a hard-coded `2/3`.
    let g = golden();
    let cases = g["cases"].as_object().expect("cases is an object");
    let mut moved_q = false;
    let mut moved_zero = false;
    let mut phase_counts = std::collections::BTreeSet::new();
    for case in cases.values() {
        phase_counts.insert(case["n_phase"].as_u64().expect("n_phase"));
        for (component, flag) in [("q", &mut moved_q), ("zero", &mut moved_zero)] {
            if flat(&case["park_forward"][component])
                .iter()
                .any(|v| v.abs() > 1e-6)
            {
                *flag = true;
            }
        }
    }
    assert!(moved_q, "no fixture moves the q axis");
    assert!(moved_zero, "no fixture moves the zero sequence");
    assert!(
        phase_counts.len() > 1,
        "every fixture has the same phase count {phase_counts:?}, so the 2/N scaling is untested"
    );
}

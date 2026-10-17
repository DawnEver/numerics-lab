//! The nonlinear port judged against the PYTHON, through the committed golden TRAJECTORIES.
//!
//! `goldens/nonlinear.json` is generated from `python/numerics_lab/nonlinear/` and is read by
//! both languages: `tests/unit/test_nonlinear_port_golden.py` asserts the live Python still
//! produces it (that is the freeze protocol, made executable), and this file asserts the Rust port
//! produces it too. Neither half alone is the proof.
//!
//! Why a data file rather than calling Rust from Python across the FFI: a second native extension
//! cannot be installed into MAIN's venv from a lane. The trade turns out to be a gain — the golden
//! pins the Python at a sha, which is exactly what the migration's freeze protocol needs and what
//! an FFI round-trip would NOT give.
//!
//! # Why the per-ITERATION triples are the assertion
//!
//! Every strategy converges to the same fixed point, so a golden that pinned only the final `x`
//! would pass against a port that reached it by a different path — a broken alpha ratchet, a dead
//! stagnation break, a Newton silently falling back to Picard. The `(alpha, err, increment)`
//! sequence is what those defects move, and it is what this file compares.
//!
//! Tolerance is `1e-12` relative. That is not an accuracy budget — both sides evaluate the same
//! closed-form arithmetic in f64 — it is a floor under the operation-ordering differences a hand
//! port makes (numpy reaching a length-3 dot through BLAS where the Rust writes the loop out).

use numerics::nonlinear::{
    get_strategy, AssembledState, IterationStrategy, NonlinearError, NonlinearSystem, RawState,
    SolveOptions, StateOf, KNOWN_STRATEGIES,
};
use serde_json::Value;
use std::path::PathBuf;

const RTOL: f64 = 1e-12;

fn golden() -> Value {
    let path: PathBuf = [
        env!("CARGO_MANIFEST_DIR"),
        "tests",
        "goldens",
        "nonlinear.json",
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
    serde_json::from_str(&text).expect("goldens/nonlinear.json is valid JSON")
}

fn floats(v: &Value) -> Vec<f64> {
    v.as_array()
        .expect("an array of numbers")
        .iter()
        .map(|n| n.as_f64().expect("a finite number"))
        .collect()
}

/// One recorded number. `null` means "no predecessor" (the first pass's infinite energy error) and
/// must not silently compare equal to a finite value.
fn close_opt(got: f64, want: &Value, what: &str) {
    if want.is_null() {
        assert!(
            !got.is_finite(),
            "{what}: rust {got} where the python had no predecessor at all"
        );
        return;
    }
    let w = want.as_f64().expect("a number or null");
    assert!(got.is_finite(), "{what}: rust {got}, python {w}");
    let tol = RTOL * w.abs().max(1.0);
    assert!(
        (got - w).abs() <= tol,
        "{what}: rust {got} vs python {w} (|d| = {:e}, tol {tol:e})",
        (got - w).abs()
    );
}

/// The golden's `_AffineSystem`: `g(x)_i = slope_i x_i + lam b_i`, or the constant target when the
/// system declares itself linear.
///
/// The tangent is the diagonal of `r(x) = x - g(x)` and the secant operator is the identity, which
/// is the `H0 = I` seed the quasi-Newton family is given on the Python side too.
struct Affine {
    slope: Vec<f64>,
    b: Vec<f64>,
    linear: bool,
    lam: f64,
    /// An energy sequence INDEPENDENT of the iterate, held at its last value once exhausted.
    ///
    /// `sum(x^2)` on a contraction is monotone, so it can never drive alpha to its floor — and a
    /// golden that never reaches the floor cannot detect the floor moving. Measured: perturbing
    /// `ALPHA_MIN` by 1e-7 relative left every trajectory in the first three fixtures unchanged.
    energy_schedule: Option<Vec<f64>>,
    assembled: usize,
    trace: Vec<(f64, f64, f64)>,
}

impl Affine {
    fn new(slope: Vec<f64>, b: Vec<f64>, linear: bool, energy_schedule: Option<Vec<f64>>) -> Self {
        Self {
            slope,
            b,
            linear,
            lam: 1.0,
            energy_schedule,
            assembled: 0,
            trace: Vec::new(),
        }
    }

    fn g(&self, x: &[f64]) -> Vec<f64> {
        if self.linear {
            return self.b.iter().map(|v| self.lam * v).collect();
        }
        x.iter()
            .zip(&self.slope)
            .zip(&self.b)
            .map(|((v, s), c)| s * v + self.lam * c)
            .collect()
    }
}

impl NonlinearSystem for Affine {
    type Matrix = Vec<f64>;
    type Payload = Vec<f64>;

    fn is_linear(&self) -> bool {
        self.linear
    }

    fn initial_guess(&self) -> Vec<f64> {
        vec![0.0; self.b.len()]
    }

    fn assemble(&mut self, x: &[f64]) -> Result<StateOf<Self>, NonlinearError> {
        let total_energy = match &self.energy_schedule {
            None => x.iter().map(|v| v * v).sum(),
            Some(s) => s[self.assembled.min(s.len() - 1)],
        };
        self.assembled += 1;
        Ok(AssembledState {
            // The identity secant, so the trait's default `solve_secant` reproduces the Python
            // fixture's `solve_secant(state, rhs) -> rhs`.
            system_matrix: Some(vec![1.0; x.len()]),
            total_energy,
            payload: Some(x.to_vec()),
        })
    }

    fn solve_linear(&mut self, state: &StateOf<Self>) -> Result<Vec<f64>, NonlinearError> {
        Ok(self.g(state.payload.as_ref().expect("always set")))
    }

    fn residual(&mut self, x: &[f64], _state: &StateOf<Self>) -> Result<Vec<f64>, NonlinearError> {
        let g = self.g(x);
        Ok(x.iter().zip(&g).map(|(a, b)| a - b).collect())
    }

    fn tangent(
        &mut self,
        _x: &[f64],
        _state: &StateOf<Self>,
    ) -> Result<Option<Self::Matrix>, NonlinearError> {
        // ALWAYS a tangent, mirroring the Python fixture exactly -- see its `tangent` comment.
        // A `None` here would send `newton-krylov` down the Jacobian-free branch, which the
        // Python solves with GMRES and this port does not implement.
        Ok(Some(self.slope.iter().map(|s| 1.0 - s).collect()))
    }

    fn solve_system(
        &mut self,
        matrix: &Self::Matrix,
        rhs: &[f64],
    ) -> Result<Vec<f64>, NonlinearError> {
        Ok(matrix.iter().zip(rhs).map(|(d, r)| r / d).collect())
    }

    fn scale_source(&mut self, lam: f64) -> Result<(), NonlinearError> {
        self.lam = lam;
        Ok(())
    }

    fn on_iteration(
        &mut self,
        _i: usize,
        _state: &StateOf<Self>,
        err: f64,
        alpha: f64,
        increment: f64,
    ) {
        self.trace.push((alpha, err, increment));
    }
}

/// The golden's `_NuMixSystem`: raw arrays linear in one scalar per call.
struct NuMix {
    k_seq: Vec<f64>,
    calls: usize,
    folded: Vec<(f64, f64)>,
    trace: Vec<(f64, f64, f64)>,
}

impl NonlinearSystem for NuMix {
    type Matrix = Vec<f64>;
    type Payload = ();

    fn initial_guess(&self) -> Vec<f64> {
        vec![0.0]
    }
    fn assemble(&mut self, _x: &[f64]) -> Result<StateOf<Self>, NonlinearError> {
        unreachable!("coefficient-damped drives raw_state/fold_raw, never assemble")
    }
    fn solve_linear(&mut self, _state: &StateOf<Self>) -> Result<Vec<f64>, NonlinearError> {
        Ok(vec![1.0])
    }
    fn residual(&mut self, x: &[f64], _state: &StateOf<Self>) -> Result<Vec<f64>, NonlinearError> {
        Ok(x.to_vec())
    }
    fn solve_system(
        &mut self,
        _matrix: &Self::Matrix,
        rhs: &[f64],
    ) -> Result<Vec<f64>, NonlinearError> {
        Ok(rhs.to_vec())
    }
    fn supports_raw_mixing(&self) -> bool {
        true
    }
    fn raw_state(&mut self, _x: &[f64]) -> Result<Option<RawState>, NonlinearError> {
        let k = self.k_seq[self.calls.min(self.k_seq.len() - 1)];
        self.calls += 1;
        Ok(Some(RawState {
            rows: vec![0],
            cols: vec![0],
            vals: vec![k],
            source: vec![2.0 * k],
            total_energy: k * k,
        }))
    }
    fn fold_raw(&mut self, raw: &RawState) -> Result<Option<StateOf<Self>>, NonlinearError> {
        self.folded.push((raw.vals[0], raw.source[0]));
        Ok(Some(AssembledState {
            system_matrix: Some(raw.vals.clone()),
            total_energy: raw.total_energy,
            payload: None,
        }))
    }
    fn on_iteration(
        &mut self,
        _i: usize,
        _state: &StateOf<Self>,
        err: f64,
        alpha: f64,
        increment: f64,
    ) {
        self.trace.push((alpha, err, increment));
    }
}

#[test]
fn every_strategy_walks_the_python_trajectory_on_every_fixture() {
    let g = golden();
    let rtol_pct = g["rtol_energy_pct"].as_f64().expect("rtol");
    let fixtures = g["fixtures"].as_object().expect("fixtures is an object");
    assert!(!fixtures.is_empty(), "the golden covers no fixture at all");

    for (fixture_name, fixture) in fixtures {
        let slope = floats(&fixture["slope"]);
        let b = floats(&fixture["b"]);
        let linear = fixture["is_linear"].as_bool().expect("is_linear");
        let max_iter = fixture["max_iter"].as_u64().expect("max_iter") as usize;
        let strategies = fixture["strategies"]
            .as_object()
            .expect("strategies is an object");

        let schedule = fixture["energy_schedule"]
            .as_array()
            .map(|_| floats(&fixture["energy_schedule"]));

        for (name, want) in strategies {
            let mut sys = Affine::new(slope.clone(), b.clone(), linear, schedule.clone());
            let opts = SolveOptions {
                max_iter,
                rtol_energy_pct: rtol_pct,
                seed: None,
            };
            let strategy: Box<dyn IterationStrategy<Affine>> =
                get_strategy(Some(name)).unwrap_or_else(|e| panic!("{fixture_name}/{name}: {e}"));
            let res = strategy
                .solve(&mut sys, &opts)
                .unwrap_or_else(|e| panic!("{fixture_name}/{name} refused: {e}"));
            let where_ = format!("{fixture_name}/{name}");

            assert_eq!(
                res.iterations,
                want["iterations"].as_u64().expect("iterations") as usize,
                "{where_}: iteration count -- the port took a different number of steps"
            );
            assert_eq!(
                res.converged,
                want["converged"].as_bool().expect("converged"),
                "{where_}: converged"
            );

            let want_trace = want["trace"].as_array().expect("trace is an array");
            assert_eq!(
                sys.trace.len(),
                want_trace.len(),
                "{where_}: the port reported {} iterations to its observer, the python {}",
                sys.trace.len(),
                want_trace.len()
            );
            for (k, ((alpha, err, inc), w)) in sys.trace.iter().zip(want_trace).enumerate() {
                close_opt(*alpha, &w["alpha"], &format!("{where_}.trace[{k}].alpha"));
                close_opt(*err, &w["err"], &format!("{where_}.trace[{k}].err"));
                close_opt(
                    *inc,
                    &w["increment"],
                    &format!("{where_}.trace[{k}].increment"),
                );
            }

            let want_x = want["x"].as_array().expect("x is an array");
            assert_eq!(res.x.len(), want_x.len(), "{where_}: x length");
            for (k, (got, w)) in res.x.iter().zip(want_x).enumerate() {
                close_opt(*got, w, &format!("{where_}.x[{k}]"));
            }
        }
    }
}

#[test]
fn coefficient_damped_mixes_the_raw_arrays_the_way_the_python_does() {
    let g = golden();
    let nu = &g["coefficient_damped"];
    let mut sys = NuMix {
        k_seq: floats(&nu["k_seq"]),
        calls: 0,
        folded: Vec::new(),
        trace: Vec::new(),
    };
    let opts = SolveOptions {
        max_iter: nu["max_iter"].as_u64().expect("max_iter") as usize,
        rtol_energy_pct: g["rtol_energy_pct"].as_f64().expect("rtol"),
        seed: None,
    };
    let strategy: Box<dyn IterationStrategy<NuMix>> = get_strategy(Some("coefficient-damped")).unwrap();
    let res = strategy.solve(&mut sys, &opts).expect("coefficient-damped runs");

    let want = nu["folded"].as_array().expect("folded is an array");
    assert_eq!(sys.folded.len(), want.len(), "coefficient_damped: fold count");
    for (k, ((vals, source), w)) in sys.folded.iter().zip(want).enumerate() {
        close_opt(*vals, &w[0], &format!("coefficient_damped.folded[{k}].vals"));
        close_opt(*source, &w[1], &format!("coefficient_damped.folded[{k}].source"));
    }
    let want_trace = nu["trace"].as_array().expect("trace is an array");
    assert_eq!(sys.trace.len(), want_trace.len(), "coefficient_damped: trace length");
    for (k, ((alpha, err, inc), w)) in sys.trace.iter().zip(want_trace).enumerate() {
        close_opt(*alpha, &w["alpha"], &format!("coefficient_damped.trace[{k}].alpha"));
        close_opt(*err, &w["err"], &format!("coefficient_damped.trace[{k}].err"));
        close_opt(
            *inc,
            &w["increment"],
            &format!("coefficient_damped.trace[{k}].increment"),
        );
    }
    assert_eq!(
        res.iterations,
        nu["iterations"].as_u64().expect("iterations") as usize
    );
    assert_eq!(res.converged, nu["converged"].as_bool().expect("converged"));
}

#[test]
fn the_golden_covers_every_strategy_the_port_registers() {
    // The mirror of the Python's coverage test. Without it, a strategy added to the registry and
    // not to the golden silently stops being compared at all.
    let g = golden();
    let generic = g["fixtures"]["contraction"]["strategies"]
        .as_object()
        .expect("the contraction fixture drives the generic strategies");
    for name in KNOWN_STRATEGIES {
        let covered = generic.contains_key(name) || name == "coefficient-damped";
        assert!(
            covered,
            "the port registers {name} but the golden does not cover it -- regenerate on the \
             Python side and commit"
        );
    }
    assert!(
        g["coefficient_damped"].is_object(),
        "coefficient-damped needs the raw pre-fold pair, so it has its own fixture -- and it is missing"
    );
}

#[test]
fn the_fixtures_exercise_both_damping_answers() {
    // The floor. A fixture set where alpha never left 1.0 would let a port that ignored every
    // ratchet -- the `meaningful` guard, the cap, the recovery leg -- pass a golden that still
    // looked complete. The measured defects this module exists to prevent are ALL in that half.
    let g = golden();
    let mut damped = std::collections::BTreeSet::new();
    for (name, fixture) in g["fixtures"].as_object().expect("fixtures") {
        let trace = fixture["strategies"]["picard-damped"]["trace"]
            .as_array()
            .expect("picard-damped trace");
        let lowest = trace
            .iter()
            .filter_map(|s| s["alpha"].as_f64())
            .fold(f64::INFINITY, f64::min);
        assert!(
            lowest.is_finite(),
            "{name}: picard-damped recorded no alpha"
        );
        damped.insert(lowest < 1.0);
    }
    assert_eq!(
        damped,
        [false, true].into_iter().collect(),
        "every fixture answers the same way on damping, so the ratchet logic is untested"
    );
}

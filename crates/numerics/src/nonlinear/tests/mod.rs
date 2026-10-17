//! The port's own regressions, mirroring the Python's assertions one for one.
//!
//! Every test here corresponds to a named Python file under `tests/unit/`, because each of
//! those files encodes a MEASURED failure the code exists to prevent. A port that kept the
//! arithmetic and dropped the tests would silently reintroduce them.

use super::*;

/// `(x, call_index) -> total_energy`, the only thing a strategy reads off an assembly.
type EnergyRule = Box<dyn Fn(&[f64], usize) -> f64>;
/// `(x, call_index) -> g(x)`, the Picard image.
type PicardMap = Box<dyn Fn(&[f64], usize) -> Vec<f64>>;

/// A synthetic [`NonlinearSystem`] driven by two closures: the energy it reports and the Picard
/// image it returns. Everything real physics supplies, reduced to the two things a strategy reads.
///
/// `Matrix` is a DIAGONAL, which is enough for every synthetic here and keeps the tests reading as
/// arithmetic rather than as a linear-algebra exercise. The secant operator is the identity, the
/// same `H0 = I` the Python fixtures hand the quasi-Newton family.
struct Synth {
    local_fields: bool,
    linear_tolerances: Vec<f64>,
    n: usize,
    x0: Vec<f64>,
    linear: bool,
    /// `(x, call_index) -> total_energy`.
    energy: EnergyRule,
    /// `(x, call_index) -> g(x)`.
    map: PicardMap,
    /// Diagonal Newton tangent; `None` makes the Newton family fall back to Picard, which is what
    /// most of the Python fixtures do.
    tangent_diag: Option<Vec<f64>>,
    assembled: usize,
    solved: usize,
    alphas: Vec<f64>,
    increments: Vec<f64>,
}

impl Synth {
    fn new(
        x0: Vec<f64>,
        energy: impl Fn(&[f64], usize) -> f64 + 'static,
        map: impl Fn(&[f64], usize) -> Vec<f64> + 'static,
    ) -> Self {
        Self {
            local_fields: false,
            linear_tolerances: Vec::new(),
            n: x0.len(),
            x0,
            linear: false,
            energy: Box::new(energy),
            map: Box::new(map),
            tangent_diag: None,
            assembled: 0,
            solved: 0,
            alphas: Vec::new(),
            increments: Vec::new(),
        }
    }

    /// `g(x) = slope * x + b`, elementwise — the Python's `_FixedPointSystem`, with the slope
    /// allowed to differ per component so a multi-secant method has something to find.
    fn affine(slope: Vec<f64>, b: Vec<f64>) -> Self {
        let (s, c) = (slope.clone(), b.clone());
        let mut sys = Self::new(
            vec![0.0; b.len()],
            |x, _| x.iter().map(|v| v * v).sum(),
            move |x, _| {
                x.iter()
                    .zip(&s)
                    .zip(&c)
                    .map(|((v, m), k)| m * v + k)
                    .collect()
            },
        );
        // r(x) = x - g(x) = (1 - slope) x - b, so the tangent is diag(1 - slope).
        sys.tangent_diag = Some(slope.iter().map(|m| 1.0 - m).collect());
        sys
    }

    fn min_alpha(&self) -> f64 {
        self.alphas.iter().copied().fold(f64::INFINITY, f64::min)
    }
}

impl NonlinearSystem for Synth {
    type Matrix = Vec<f64>;
    type Payload = Vec<f64>;

    fn local_values(&self, state: &StateOf<Self>) -> Result<LocalFields, NonlinearError> {
        Ok(if self.local_fields {
            vec![state
                .payload
                .as_ref()
                .unwrap()
                .iter()
                .map(|&x| vec![x])
                .collect()]
        } else {
            Vec::new()
        })
    }

    fn set_linear_tolerance(&mut self, atol: f64) -> Result<(), NonlinearError> {
        self.linear_tolerances.push(atol);
        Ok(())
    }

    fn is_linear(&self) -> bool {
        self.linear
    }

    fn initial_guess(&self) -> Vec<f64> {
        self.x0.clone()
    }

    fn assemble(&mut self, x: &[f64]) -> Result<StateOf<Self>, NonlinearError> {
        let total_energy = (self.energy)(x, self.assembled);
        self.assembled += 1;
        Ok(AssembledState {
            system_matrix: Some(vec![1.0; self.n]),
            total_energy,
            payload: Some(x.to_vec()),
        })
    }

    fn solve_linear(&mut self, state: &StateOf<Self>) -> Result<Vec<f64>, NonlinearError> {
        let x = state.payload.as_ref().expect("the fixture always sets it");
        let out = (self.map)(x, self.solved);
        self.solved += 1;
        Ok(out)
    }

    fn residual(&mut self, x: &[f64], state: &StateOf<Self>) -> Result<Vec<f64>, NonlinearError> {
        // r(x) = x - g(x): its root IS the fixed point, so Newton and Picard chase the same thing.
        let _ = state;
        let g = (self.map)(x, self.solved);
        Ok(x.iter().zip(&g).map(|(a, b)| a - b).collect())
    }

    fn tangent(
        &mut self,
        _x: &[f64],
        _state: &StateOf<Self>,
    ) -> Result<Option<Self::Matrix>, NonlinearError> {
        Ok(self.tangent_diag.clone())
    }

    fn solve_system(
        &mut self,
        matrix: &Self::Matrix,
        rhs: &[f64],
    ) -> Result<Vec<f64>, NonlinearError> {
        if matrix.len() != rhs.len() {
            return Err(NonlinearError::ShapeMismatch {
                expected: matrix.len(),
                got: rhs.len(),
            });
        }
        Ok(matrix.iter().zip(rhs).map(|(d, r)| r / d).collect())
    }

    fn on_iteration(
        &mut self,
        _i: usize,
        _state: &StateOf<Self>,
        _err: f64,
        alpha: f64,
        increment: f64,
    ) {
        self.alphas.push(alpha);
        self.increments.push(increment);
    }
}

fn opts(max_iter: usize) -> SolveOptions {
    SolveOptions::new(max_iter)
}

// -- test_nonlinear.py ------------------------------------------------------------------------

/// A genuinely linear system: `solve_linear` returns a fixed target regardless of `x`.
fn linear_system(target: Vec<f64>) -> Synth {
    let t = target.clone();
    let mut s = Synth::new(
        vec![0.0; target.len()],
        |x, _| x.iter().map(|v| v * v).sum(),
        move |_, _| t.clone(),
    );
    s.linear = true;
    s
}

#[test]
fn a_linear_system_converges_to_target_in_one_substitution() {
    for name in ["picard-damped", "picard"] {
        let mut sys = linear_system(vec![3.0, -1.5]);
        let strategy = get_strategy::<Synth>(Some(name)).unwrap();
        let res = strategy.solve(&mut sys, &opts(50)).unwrap();
        assert!(res.converged, "{name}");
        assert_eq!(res.x, vec![3.0, -1.5], "{name}");
        // The final assembly is at the CONVERGED x, not at the one before it.
        let payload = res.state.unwrap().payload.unwrap();
        assert_eq!(payload, vec![3.0, -1.5], "{name}: delayed break");
    }
}

#[test]
fn a_contraction_converges_to_its_fixed_point() {
    for name in ["picard-damped", "picard"] {
        let mut sys = Synth::affine(vec![0.5, 0.5], vec![2.0, 4.0]);
        let res = get_strategy::<Synth>(Some(name))
            .unwrap()
            .solve(&mut sys, &opts(500))
            .unwrap();
        assert!(res.converged, "{name}");
        for (got, want) in res.x.iter().zip([4.0, 8.0]) {
            assert!((got - want).abs() < 1e-3, "{name}: {got} vs {want}");
        }
    }
}

#[test]
fn plain_picard_diverges_on_an_expansive_map_and_says_so() {
    // First principle: the failure is OBSERVABLE (`converged: false`, the iterate blows up), not
    // hidden. It is the datum a convergence matrix would `xfail`, never a silent exclusion.
    let mut sys = Synth::affine(vec![2.0], vec![1.0]);
    let res = PicardStrategy.solve(&mut sys, &opts(30)).unwrap();
    assert!(!res.converged);
    assert!(res.x[0].abs() > 1e3, "it blew up, plainly: {}", res.x[0]);
}

#[test]
fn damped_picard_engages_under_relaxation_where_plain_picard_does_not() {
    let mut damped = Synth::affine(vec![-1.5], vec![1.0]);
    DampedPicardStrategy.solve(&mut damped, &opts(25)).unwrap();
    assert!(damped.min_alpha() < 1.0, "under-relaxation never activated");

    let mut plain = Synth::affine(vec![-1.5], vec![1.0]);
    PicardStrategy.solve(&mut plain, &opts(25)).unwrap();
    assert!(
        plain.alphas.iter().all(|a| *a == 1.0),
        "undamped Picard must always take the full step"
    );
}

#[test]
fn max_iter_zero_returns_the_initial_guess_and_no_state() {
    let mut sys = linear_system(vec![1.0, 2.0]);
    let res = DampedPicardStrategy.solve(&mut sys, &opts(0)).unwrap();
    assert_eq!(res.iterations, 0);
    assert!(res.state.is_none());
    assert_eq!(res.x, vec![0.0, 0.0]);
}

#[test]
fn the_registry_resolves_every_declared_name_and_refuses_the_rest() {
    for name in KNOWN_STRATEGIES {
        let s = get_strategy::<Synth>(Some(name)).unwrap();
        assert_eq!(s.name(), name, "the registry key and the strategy disagree");
    }
    // `None` is Newton, the default in every family since 2026-09-25 (mirrors the Python).
    assert_eq!(get_strategy::<Synth>(None).unwrap().name(), "newton");
    assert!(matches!(
        get_strategy::<Synth>(Some("nonesuch")),
        Err(NonlinearError::UnknownStrategy { .. })
    ));
    // The requirement is DATA, askable without running anything.
    assert_eq!(strategy_requires("coefficient-damped").unwrap().len(), 2);
    assert!(strategy_requires("picard").unwrap().is_empty());
    let sys = linear_system(vec![1.0]);
    assert!(!strategy_runs_on("coefficient-damped", &sys).unwrap());
    assert!(strategy_runs_on("picard", &sys).unwrap());
}

// -- The declared stop rule and the per-rule default bar (the Python split, 2026-09-26) --------

#[test]
fn the_newton_family_declares_the_residual_stop_and_every_other_strategy_the_energy_one() {
    let residual: Vec<&str> = KNOWN_STRATEGIES
        .iter()
        .copied()
        .filter(|name| strategy_stops_on_residual(name).unwrap())
        .collect();
    assert_eq!(
        residual,
        ["newton", "newton-held-factor", "newton-linesearch"]
    );
}

#[test]
fn an_unknown_strategy_has_no_stop_rule_to_answer_with() {
    assert!(matches!(
        strategy_stops_on_residual("no-such-strategy"),
        Err(NonlinearError::UnknownStrategy { .. })
    ));
}

#[test]
fn the_default_bar_is_newton_rtol_for_a_residual_stop_and_the_fixed_point_one_otherwise() {
    assert!((NEWTON_RTOL_PCT - 1e-4).abs() < f64::EPSILON); // 1e-6 relative
    assert!((RTOL_ENERGY_PCT - 0.01).abs() < f64::EPSILON); // 1e-4 relative
    for name in KNOWN_STRATEGIES {
        let want = if strategy_stops_on_residual(name).unwrap() {
            NEWTON_RTOL_PCT
        } else {
            RTOL_ENERGY_PCT
        };
        assert_eq!(
            default_rtol_energy_pct(name).unwrap().to_bits(),
            want.to_bits(),
            "{name}"
        );
        let opts = SolveOptions::for_strategy(name, 7).unwrap();
        assert_eq!(opts.rtol_energy_pct.to_bits(), want.to_bits(), "{name}");
        assert_eq!(opts.max_iter, 7);
    }
}

mod local;
mod regressions;

//! The alpha-ratchet, plateau, stagnation, budget and continuation regressions.
//!
//! Split from `tests/mod.rs`, which holds the shared `Synth` fixture and the port of
//! `test_nonlinear.py`; this file ports the five named regression files beside it.

use super::*;

// -- test_energy_noise_is_not_a_limit_cycle.py -------------------------------------------------

/// A contraction toward `x* = 10` whose energy follows a prescribed schedule.
fn contraction_with_energy(energy: impl Fn(&[f64], usize) -> f64 + 'static) -> Synth {
    Synth::new(vec![0.0], energy, |x, _| vec![x[0] + 0.5 * (10.0 - x[0])])
}

#[test]
fn energy_noise_does_not_drive_alpha_to_the_floor() {
    // MEASURED RED before the `meaningful` guard: alpha reached 0.1. An energy oscillation of
    // relative size 2e-12 is twelve orders of magnitude inside the 0.01% tolerance -- the float
    // noise floor, not a limit cycle, and damping for it cost ~10x the iterations at a tenth of
    // the step size. From outside the process that is indistinguishable from a hang.
    let mut sys = contraction_with_energy(|_, k| 500.0 + if k % 2 == 1 { 1e-11 } else { -1e-11 });
    DampedPicardStrategy.solve(&mut sys, &opts(25)).unwrap();
    assert_eq!(
        sys.min_alpha(),
        1.0,
        "alpha was ratcheted by the energy's own noise floor"
    );
}

#[test]
fn a_real_limit_cycle_still_damps_alpha() {
    // The converse. Without it, deleting the sign-flip ratchet entirely would pass the test above.
    let mut sys = contraction_with_energy(|_, k| 500.0 * if k % 2 == 1 { 1.2 } else { 0.8 });
    DampedPicardStrategy.solve(&mut sys, &opts(25)).unwrap();
    assert!(
        sys.min_alpha() < 1.0,
        "a 20%-amplitude period-2 oscillation did not damp alpha at all"
    );
}

// -- test_alpha_recovers_after_a_disturbance_at_the_floor.py -----------------------------------

/// Oscillate ABOVE tolerance for `disturbed` steps, then hold the energy exactly flat.
///
/// `solve_linear` holds the undamped increment CONSTANT so the run cannot converge out from under
/// the question: what is observed is purely whether alpha recovers once nothing disturbs it.
fn oscillate_then_settle(disturbed: usize) -> Synth {
    Synth::new(
        vec![0.0, 0.0, 0.0],
        move |_, k| {
            if k < disturbed {
                100.0 + if k % 2 == 0 { 5.0 } else { -5.0 }
            } else {
                100.0
            }
        },
        |x, _| x.iter().map(|v| v + 10.0).collect(),
    )
}

#[test]
fn alpha_reaches_the_floor_when_the_energy_genuinely_oscillates() {
    // THE FLOOR of the next test: if alpha never got damped, the recovery assertion is vacuous.
    let mut sys = oscillate_then_settle(8);
    DampedPicardStrategy.solve(&mut sys, &opts(40)).unwrap();
    assert!(
        (sys.min_alpha() - ALPHA_MIN).abs() < 1e-12,
        "the fixture no longer reproduces the condition: min alpha {}",
        sys.min_alpha()
    );
}

#[test]
fn alpha_recovers_once_the_energy_settles() {
    // THE DEFECT: a disturbance arriving AT the floor used to kill recovery for the rest of the
    // solve -- `max(ALPHA_MIN, alpha * 0.6)` is a no-op there, so recording the cap anyway set
    // `alpha_cap = ALPHA_MIN` and the gate `alpha < alpha_cap` read `0.1 < 0.1` forever. Measured
    // on case_h: 557 of 605 iterations pinned at 0.100 with the energy error at 0.00%.
    let mut sys = oscillate_then_settle(8);
    DampedPicardStrategy.solve(&mut sys, &opts(40)).unwrap();
    let settled = &sys.alphas[9..];
    let best = settled.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    assert!(
        best > ALPHA_MIN,
        "alpha stayed pinned at {ALPHA_MIN} for all {} undisturbed steps",
        settled.len()
    );
}

#[test]
fn the_cap_still_forbids_retrying_a_step_size_that_misbehaved() {
    // THE OTHER SIDE OF THE RATCHET, so the fix cannot be "delete the cap": recovery must stay
    // bounded by the largest step size that actually misbehaved.
    let mut sys = oscillate_then_settle(8);
    DampedPicardStrategy.solve(&mut sys, &opts(40)).unwrap();
    let disturbed_max = sys.alphas[..9].iter().copied().fold(f64::MIN, f64::max);
    let settled_max = sys.alphas[9..].iter().copied().fold(f64::MIN, f64::max);
    assert!(
        settled_max <= disturbed_max,
        "alpha recovered to {settled_max}, above the {disturbed_max} that misbehaved"
    );
}

#[test]
fn the_cap_itself_releases_once_the_solve_has_settled() {
    // THE SECOND RATCHET. A step size that misbehaved FAR FROM THE SOLUTION must not be forbidden
    // for the whole solve. Measured on case_h after releasing only the FLOOR: 557 iterations
    // pinned at 0.100 became 447 at 0.108 -- recovery fired, and 0.108 was now the ceiling.
    let mut sys = oscillate_then_settle(8);
    DampedPicardStrategy.solve(&mut sys, &opts(120)).unwrap();
    let best = sys.alphas[10..].iter().copied().fold(f64::MIN, f64::max);
    assert!(
        best > 0.2,
        "alpha never climbed above {best}: the cap is still a one-way ratchet"
    );
}

#[test]
fn the_released_cap_still_backs_off_when_the_energy_misbehaves_again() {
    // THE OTHER SIDE: releasing the cap must not disarm the globaliser, or the release has traded
    // robustness for speed -- the exact trade the cap existed to prevent.
    let mut sys = oscillate_then_settle(200); // never settles
    DampedPicardStrategy.solve(&mut sys, &opts(60)).unwrap();
    let best = sys.alphas[20..].iter().copied().fold(f64::MIN, f64::max);
    assert!(
        best <= 0.7,
        "alpha stayed at {best} while the energy oscillated the whole time"
    );
}

// -- test_a_flat_energy_plateau_is_not_convergence.py ------------------------------------------

/// A contraction whose energy is FLAT over the stretch the iterate is still crossing.
///
/// `g(x) = x + 0.5 (x* - x)` converges to `x* = 10` from `x0 = 0` as 0, 5, 7.5, 8.75 ... Energy is
/// a staircase pinned at one value until the iterate is nearly there, so two CONSECUTIVE iterates
/// report a 0.00% energy change while the iterate is still 5.0 away. Nothing about this is exotic:
/// it is the measured saturating-BH plateau, reduced to the one property that produces the wrong
/// answer.
fn plateau_system(x0: f64) -> Synth {
    let mut s = Synth::new(
        vec![x0],
        |x, _| {
            if x[0] < 9.0 {
                100.0
            } else {
                100.0 + (x[0] - 9.0)
            }
        },
        |x, _| vec![x[0] + 0.5 * (10.0 - x[0])],
    );
    // No tangent offered, so the Newton family falls back to the Picard substitution and the
    // quasi-Newton family drives the identity secant -- exactly what the Python fixture does.
    s.tangent_diag = None;
    s
}

/// Every registered strategy except five, EXCLUDED BY NAME with their reasons rather than by a
/// silent filter: `continuation` wraps another strategy and is driven through `scale_source`,
/// `coefficient-damped` needs the raw pre-fold pair and refuses a system without it (it runs the same loop
/// and the same `has_converged` as `picard-damped`), and `newton` / `newton-linesearch` /
/// `newton-held-factor` REFUSE a system with no tangent (2026-09-25) -- they are held to the
/// plateau on a tangent-bearing fixture below.
const PLATEAU_DRIVABLE: [&str; 8] = [
    "anderson",
    "bfgs",
    "l-bfgs",
    "modified-newton",
    "newton-krylov",
    "picard",
    "picard-damped",
    "quasi-newton",
];

/// The three registry names of the ONE Newton, which judges the RESIDUAL and needs a tangent.
const NEWTON_NAMES: [&str; 3] = ["newton", "newton-linesearch", "newton-held-factor"];

#[test]
fn newton_refuses_a_system_with_no_tangent() {
    // UNSUPPORTED-RAISES: the Picard step Newton used to take here ran another algorithm under
    // Newton's name.
    for name in NEWTON_NAMES {
        let mut sys = plateau_system(0.0);
        let res = get_strategy::<Synth>(Some(name))
            .unwrap()
            .solve(&mut sys, &opts(3));
        assert!(
            matches!(res, Err(NonlinearError::UnsupportedSystem { .. })),
            "{name} ran on a system with no tangent"
        );
    }
}

#[test]
fn newton_is_not_fooled_by_a_flat_energy_and_still_converges_at_the_root() {
    // The plateau map with its tangent (r = x - g(x) = 0.5 (x - 10), so J = 0.5): the residual,
    // not the flat energy, decides, so a capped solve short of 10 cannot report convergence and a
    // solve allowed to run lands on it.
    for name in NEWTON_NAMES {
        let mut sys = plateau_system(0.0);
        sys.tangent_diag = Some(vec![0.5]);
        let res = get_strategy::<Synth>(Some(name))
            .unwrap()
            .solve(&mut sys, &opts(1))
            .unwrap();
        assert!(!(res.converged && (res.x[0] - 10.0).abs() > 1e-3), "{name}");
        let mut sys = plateau_system(0.0);
        sys.tangent_diag = Some(vec![0.5]);
        let res = get_strategy::<Synth>(Some(name))
            .unwrap()
            .solve(&mut sys, &opts(10))
            .unwrap();
        assert!(
            res.converged,
            "{name} did not converge on a linear residual"
        );
        assert!((res.x[0] - 10.0).abs() < 1e-9, "{name}: {}", res.x[0]);
    }
}

#[test]
fn a_constant_energy_is_not_reported_as_convergence() {
    for name in PLATEAU_DRIVABLE {
        let mut sys = plateau_system(0.0);
        let res = get_strategy::<Synth>(Some(name))
            .unwrap()
            .solve(&mut sys, &opts(3))
            .unwrap();
        let x = res.x[0];
        assert!(
            !(res.converged && (x - 10.0).abs() > 1e-3),
            "{name} reported converged at x={x}, {} from the fixed point -- it read a flat \
             total_energy as a solution",
            (x - 10.0).abs()
        );
    }
}

#[test]
fn a_genuine_fixed_point_is_still_reported_as_converged() {
    // The converse. Without it, a strategy that NEVER converges would pass the test above.
    for name in PLATEAU_DRIVABLE {
        let mut sys = plateau_system(10.0);
        let res = get_strategy::<Synth>(Some(name))
            .unwrap()
            .solve(&mut sys, &opts(10))
            .unwrap();
        assert!(res.converged, "{name} did not converge AT its fixed point");
        assert!((res.x[0] - 10.0).abs() < 1e-9, "{name}");
    }
}

// -- test_a_stationary_increment_breaks_the_stagnation.py --------------------------------------

/// A contraction toward 0 whose RELATIVE step is pinned at `steps` (cycled) and whose energy is
/// flat. `g(x) = x (1 - step)` gives `increment_ratio(x, g(x)) == step` exactly.
fn creeping_system(steps: Vec<f64>) -> Synth {
    Synth::new(
        vec![1.0],
        |_, _| 100.0,
        move |x, k| vec![x[0] * (1.0 - steps[k % steps.len()])],
    )
}

/// The strategies whose step IS the Picard map's, so a synthetic map can pin their increment.
/// Anderson and the quasi-Newton family mix their own steps from history, and `newton` left this
/// set on 2026-09-25: it no longer steps the Picard map of a tangent-less system, it refuses it.
const CREEP_DRIVABLE: [&str; 2] = ["picard", "picard-damped"];

#[test]
fn a_stationary_increment_under_a_passing_energy_converges_before_the_cap() {
    // The case_b newton shape: energy passes rtol from the second step and the increment pins at
    // 3.5e-4 -- 3.5x above the fast break's rtol/100 floor -- so without the stagnation break the
    // solve burns the whole 100-step cap: ~85 wasted iterations, ~123 s.
    for name in CREEP_DRIVABLE {
        let mut sys = creeping_system(vec![3.5e-4]);
        let res = get_strategy::<Synth>(Some(name))
            .unwrap()
            .solve(&mut sys, &opts(100))
            .unwrap();
        assert!(res.converged, "{name} ran a stationary tail to the cap");
        assert!(
            res.iterations <= STAGNATION_WINDOW + 2,
            "{name} took {} iterations; the window is {STAGNATION_WINDOW} plus the first \
             (err = inf) pass and the delayed-break pass",
            res.iterations
        );
    }
}

#[test]
fn a_wandering_increment_on_a_flat_energy_never_stagnation_breaks() {
    // The documented false-plateau hazard: energy 0.00% for the whole run, increment WANDERING 4x
    // across the window. The energy half of the rule alone WOULD have fired here, which is exactly
    // what the stationarity half exists for.
    let cap = STAGNATION_WINDOW + 20;
    for name in CREEP_DRIVABLE {
        let mut sys = creeping_system(vec![2.0e-4, 8.0e-4]);
        let res = get_strategy::<Synth>(Some(name))
            .unwrap()
            .solve(&mut sys, &opts(cap))
            .unwrap();
        assert!(
            !res.converged,
            "{name} called a wandering plateau converged"
        );
        assert_eq!(res.iterations, cap, "{name}");
    }
}

#[test]
fn the_fast_break_still_fires_first() {
    // A system already at its fixed point converges in a few iterations -- long before the window
    // can fill -- so the added break has not displaced or weakened the fast one.
    for name in CREEP_DRIVABLE {
        let mut sys = creeping_system(vec![0.0]);
        let res = get_strategy::<Synth>(Some(name))
            .unwrap()
            .solve(&mut sys, &opts(100))
            .unwrap();
        assert!(res.converged, "{name}");
        assert!(res.iterations <= 5, "{name} took {}", res.iterations);
    }
}

// -- coefficient-damped -------------------------------------------------------------------------------

/// Minimal stand-in for the FEA `raw_state`/`fold_raw` contract.
///
/// Raw stiffness/source are linear in a single scalar `k` per call -- `k` stands in for the
/// per-element reluctivity target, since every real FEA kernel's raw stiffness AND magnet-source
/// entries are linear in it. Each call returns `vals = [k]`, `source = [2k]`,
/// `total_energy = k^2`, driven by a `k` sequence chosen to force a KNOWN alpha ratchet at a KNOWN
/// iteration, so the mixed value can be hand-computed and compared exactly.
struct NuMix {
    k_seq: Vec<f64>,
    calls: usize,
    folded: Vec<(f64, f64)>,
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
        // The value is irrelevant here: `raw_state` ignores x and follows `k_seq`.
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
}

#[test]
fn coefficient_damped_mixes_the_raw_arrays_exactly_like_the_deleted_per_element_mix() {
    // `k_seq = [1, 3, 2, 5]` (energy k^2) gives deltas +8 (sign +1, no ratchet yet -- there is no
    // PRIOR sign to flip against), then -5 (sign -1, FLIPS vs +1 -> alpha ratchets to 0.6 for the
    // NEXT mix). So the 4th call's mixed reluctivity is 0.4*2 + 0.6*5 = 3.8, computed here
    // independently of the strategy.
    let mut sys = NuMix {
        k_seq: vec![1.0, 3.0, 2.0, 5.0],
        calls: 0,
        folded: Vec::new(),
    };
    CoefficientDampedPicardStrategy
        .solve(&mut sys, &opts(4))
        .unwrap();
    assert_eq!(sys.folded.len(), 4);
    assert_eq!(sys.folded[0], (1.0, 2.0));
    assert_eq!(sys.folded[1], (3.0, 6.0));
    assert_eq!(sys.folded[2], (2.0, 4.0));
    let expected = 0.4 * 2.0 + 0.6 * 5.0;
    assert!((sys.folded[3].0 - expected).abs() < 1e-12);
    assert!((sys.folded[3].1 - 2.0 * expected).abs() < 1e-12);
}

#[test]
fn coefficient_damped_refuses_a_system_without_the_raw_pair() {
    // The refusal is a declared capability boundary, not a discovered TypeError -- and it names
    // what is missing rather than naming a wrapper.
    let mut sys = linear_system(vec![1.0]);
    let err = CoefficientDampedPicardStrategy
        .solve(&mut sys, &opts(5))
        .unwrap_err();
    assert!(
        format!("{err}").contains("raw_state/fold_raw"),
        "the refusal must name the missing pair: {err}"
    );
}

// -- the time budget ---------------------------------------------------------------------------

#[test]
fn a_spent_budget_stops_the_solve_and_names_the_call() {
    use std::cell::Cell;
    // A clock that jumps past the budget on its second reading: the first `assemble` is admitted,
    // the next expensive call is refused. Overshoot is bounded by ONE call, which is the claim the
    // wrapper makes and the most it can honestly make.
    let ticks = Cell::new(0.0_f64);
    let sys = Synth::affine(vec![0.5], vec![1.0]);
    let mut budgeted = with_time_budget(sys, 1.0, move || {
        let t = ticks.get();
        ticks.set(t + 1.0);
        t
    });
    let err = DampedPicardStrategy
        .solve(&mut budgeted, &opts(10))
        .unwrap_err();
    match err {
        NonlinearError::BudgetExceeded { at, budget_s, .. } => {
            assert!(BUDGETED_CALLS.contains(&at), "{at} is not a budgeted call");
            assert_eq!(budget_s, 1.0);
        }
        other => panic!("expected a budget refusal, got {other}"),
    }
}

#[test]
fn the_budgeted_set_is_every_expensive_entry_point() {
    // The scope claim in `TimeBudgetedSystem`'s docs, held to its own data. Leaving the
    // Newton-family calls unbudgeted would put a hole in the guard exactly where a caller who
    // switched strategy would not think to look.
    for call in [
        "assemble",
        "solve_linear",
        "solve_system",
        "solve_secant",
        "tangent",
    ] {
        assert!(BUDGETED_CALLS.contains(&call), "{call} is unbudgeted");
    }
}

#[test]
fn a_system_with_no_budget_is_the_system_itself() {
    // Opting out must be indistinguishable from never having opted in. Rust says it with the type:
    // no budget means no wrapper at all.
    let mut sys = Synth::affine(vec![0.5], vec![1.0]);
    let res = DampedPicardStrategy.solve(&mut sys, &opts(200)).unwrap();
    assert!(res.converged);
}

// -- continuation ------------------------------------------------------------------------------

/// The affine system with a scalable source, so `continuation` can ramp it.
struct Ramped {
    lam: std::rc::Rc<std::cell::Cell<f64>>,
    inner: Synth,
}

impl Ramped {
    fn new() -> Self {
        let lam = std::rc::Rc::new(std::cell::Cell::new(1.0));
        let l = lam.clone();
        let mut inner = Synth::new(
            vec![0.0],
            |x, _| x[0] * x[0],
            move |x, _| vec![0.5 * x[0] + 4.0 * l.get()],
        );
        inner.tangent_diag = Some(vec![0.5]);
        Self { lam, inner }
    }
}

impl NonlinearSystem for Ramped {
    type Matrix = Vec<f64>;
    type Payload = Vec<f64>;

    fn initial_guess(&self) -> Vec<f64> {
        self.inner.initial_guess()
    }
    fn assemble(&mut self, x: &[f64]) -> Result<StateOf<Self>, NonlinearError> {
        self.inner.assemble(x)
    }
    fn solve_linear(&mut self, state: &StateOf<Self>) -> Result<Vec<f64>, NonlinearError> {
        self.inner.solve_linear(state)
    }
    fn residual(&mut self, x: &[f64], state: &StateOf<Self>) -> Result<Vec<f64>, NonlinearError> {
        self.inner.residual(x, state)
    }
    fn solve_system(
        &mut self,
        matrix: &Self::Matrix,
        rhs: &[f64],
    ) -> Result<Vec<f64>, NonlinearError> {
        self.inner.solve_system(matrix, rhs)
    }
    fn scale_source(&mut self, lam: f64) -> Result<(), NonlinearError> {
        self.lam.set(lam);
        Ok(())
    }
}

#[test]
fn continuation_ends_at_the_full_source_and_leaves_the_system_unscaled() {
    let mut sys = Ramped::new();
    let res = ContinuationStrategy::<Ramped>::default()
        .solve(&mut sys, &opts(200))
        .unwrap();
    // Fixed point of the FULL problem: x = 0.5x + 4 -> x = 8.
    assert!((res.x[0] - 8.0).abs() < 1e-3, "{}", res.x[0]);
    assert_eq!(sys.lam.get(), 1.0, "the system was left scaled");
}

#[test]
fn continuation_degrades_to_a_single_solve_when_the_source_cannot_be_scaled() {
    // The Python's `NotImplementedError` path, typed. The equation says it cannot rescale; the
    // ramp is abandoned, not faked.
    let mut sys = Synth::affine(vec![0.5], vec![4.0]);
    let res = ContinuationStrategy::<Synth>::default()
        .solve(&mut sys, &opts(200))
        .unwrap();
    assert!(res.converged);
    assert!((res.x[0] - 8.0).abs() < 1e-3);
}

// -- the helpers themselves --------------------------------------------------------------------

#[test]
fn the_first_pass_has_no_energy_predecessor_and_says_infinity() {
    assert!(energy_error_pct(1.0, None).is_infinite());
    assert!((energy_error_pct(101.0, Some(100.0)) - 1.0 / 101.0 * 100.0).abs() < 1e-12);
    // The denominator floors at 1e-12, so a zero-energy pair is a 0% change and not a NaN.
    assert_eq!(energy_error_pct(0.0, Some(0.0)), 0.0);
}

#[test]
fn the_increment_is_relative_and_floors_its_own_scale() {
    assert!((increment_ratio(&[3.0, 4.0], &[3.0, 4.0])).abs() < 1e-300);
    assert!((increment_ratio(&[3.0, 4.0], &[6.0, 8.0]) - 1.0).abs() < 1e-12);
    // ||x|| = 0 would divide by zero; the 1e-12 floor makes it a large finite number instead.
    assert!(increment_ratio(&[0.0], &[1.0]).is_finite());
}

#[test]
fn both_halves_of_the_convergence_rule_are_required() {
    let rtol = 0.01;
    // Energy settled but the iterate still moving: NOT converged. That is the flat-plateau hazard.
    assert!(!has_converged(0.0, &[1.0], &[2.0], rtol));
    // Iterate settled but the energy still moving: NOT converged either.
    assert!(!has_converged(5.0, &[1.0], &[1.0], rtol));
    assert!(has_converged(0.0, &[1.0], &[1.0], rtol));
}

#[test]
fn the_stagnation_window_needs_every_step_settled_and_stationary() {
    let mut w = StagnationWatch::with_window(0.01, 3);
    assert!(!w.observe(0.001, 1e-4));
    assert!(!w.observe(0.001, 1e-4));
    assert!(w.observe(0.001, 1e-4), "three settled stationary steps");
    // One step over rtol sits INSIDE the window and blocks the break until it has aged out.
    assert!(!w.observe(0.5, 1e-4));
    assert!(!w.observe(0.001, 1e-4));
    assert!(!w.observe(0.001, 1e-4));
    assert!(w.observe(0.001, 1e-4));
    // A wandering increment past the 2x band is refused however settled the energy is.
    let mut w = StagnationWatch::with_window(0.01, 2);
    assert!(!w.observe(0.001, 1e-4));
    assert!(!w.observe(0.001, 8e-4), "4x wander must not break");
}

#[test]
fn the_error_display_names_the_budget_and_the_known_strategies() {
    let e = NonlinearError::BudgetExceeded {
        budget_s: 500.0,
        elapsed_s: 501.0,
        at: "assemble",
    };
    let text = format!("{e}");
    assert!(text.contains("500 s budget"), "{text}");
    assert!(text.contains("INCONCLUSIVE"), "{text}");
    let e = NonlinearError::UnknownStrategy {
        name: "nope".to_owned(),
    };
    assert!(format!("{e}").contains("picard-damped"));
}

/// The increment channel, both directions plus its floor - see `IncrementTrend` for the measurement.
///
/// The pair matters, not either half: a gate that never permitted growth would pass the refusal
/// assertions alone while re-introducing the crawl `ALPHA_CALM_STEPS` exists to remove.
#[test]
fn the_increment_trend_refuses_growth_on_a_rise_and_permits_it_on_a_contraction() {
    let mut t = IncrementTrend::new();
    // No history is permission: refusing here would throttle every solve's first steps.
    assert!(t.permits_growth());
    assert!(!t.claim_damping());

    // Four observations, because what counts is the RISES BETWEEN them: 3 = INCREMENT_RISE_STEPS.
    for inc in [1.0e-3, 1.1e-3, 1.2e-3, 1.3e-3] {
        assert!(
            !t.claim_damping(),
            "fewer rises than INCREMENT_RISE_STEPS must not damp"
        );
        t.observe(inc);
    }
    assert!(
        !t.permits_growth(),
        "the iterate moved MORE than the step before"
    );
    assert!(
        t.claim_damping(),
        "three consecutive rises must force a shrink"
    );
    assert!(
        !t.claim_damping(),
        "the demand is CONSUMED: one sustained rise costs one shrink, not one per step"
    );

    // THE FLOOR. A contracting march recovers immediately -- a falling increment resets the run.
    t.observe(9.0e-4);
    assert!(t.permits_growth());
    // Equal increments are not a rise: a stationary tail must not be damped for standing still.
    t.observe(9.0e-4);
    assert!(t.permits_growth());
}

/// A creeping rise INSIDE the 2x band is not stationarity, however settled the energy is.
///
/// MEASURED 2026-09-06 on `case_a mec:quad`: 1.91e-3 -> 2.05e-3 over ten steps, band ratio
/// 1.07, and the break fired at 20x the increment bar. The floor is the case above it, unchanged:
/// a constant increment still breaks.
#[test]
fn a_creeping_rise_inside_the_band_does_not_stagnation_break() {
    let mut w = StagnationWatch::with_window(0.01, 4);
    let mut inc = 1.0e-3_f64;
    for _ in 0..12 {
        assert!(!w.observe(0.001, inc), "a monotone rise is not stationary");
        inc *= 1.007;
    }
}

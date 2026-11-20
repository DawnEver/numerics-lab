//! The three optional engine hooks: a reported `state_change` the convergence rule requires
//! settled, a recoverable `TrialNotEvaluable`, and a held factor the modified Newton reuses.
//! Each is proven twice: it acts when used, and an unused hook leaves the result bit-identical.

use super::*;

/// Every strategy the synthetic can run (`coefficient-damped` needs the raw pre-fold pair).
fn strategies() -> Vec<&'static str> {
    KNOWN_STRATEGIES
        .iter()
        .copied()
        .filter(|n| *n != "coefficient-damped")
        .collect()
}

fn affine() -> Synth {
    Synth::affine(vec![0.5, 0.25], vec![1.0, -2.0])
}

fn run(name: &str, sys: &mut Synth, max_iter: usize) -> Result<ResultOf<Synth>, NonlinearError> {
    get_strategy::<Synth>(Some(name))?.solve(sys, &opts(max_iter))
}

// -- 1. state_change ---------------------------------------------------------------------------

#[test]
fn a_state_change_above_the_bar_blocks_every_strategy_from_converging() {
    for name in strategies() {
        let mut sys = affine();
        sys.state_change = Box::new(|_| Some(1.0));
        let res = run(name, &mut sys, 60).unwrap();
        assert!(
            !res.converged,
            "{name} converged while its state still moved"
        );
    }
}

#[test]
fn a_settled_state_change_reproduces_the_unreported_run_exactly() {
    for name in strategies() {
        let mut plain = affine();
        let base = run(name, &mut plain, 60).unwrap();
        // PLANTED CONTROL: the baseline converges, so the blocking test above can say no.
        assert!(base.converged, "{name} baseline did not converge");
        let mut settled = affine();
        settled.state_change = Box::new(|_| Some(0.0));
        let res = run(name, &mut settled, 60).unwrap();
        assert_eq!(res.x, base.x, "{name}");
        assert_eq!(res.iterations, base.iterations, "{name}");
        assert!(res.converged, "{name}");
    }
}

#[test]
fn convergence_waits_for_the_state_change_to_settle() {
    for name in strategies() {
        let mut plain = affine();
        let base = run(name, &mut plain, 200).unwrap();
        let settle_at = base.iterations * 3 + 5;
        let mut late = affine();
        late.state_change = Box::new(move |k| Some(if k < settle_at { 1.0 } else { 0.0 }));
        let res = run(name, &mut late, 200).unwrap();
        assert!(
            res.converged,
            "{name} never converged after the state settled"
        );
        assert!(
            res.iterations > base.iterations,
            "{name}: {} <= {}",
            res.iterations,
            base.iterations
        );
    }
}

// -- 2. TrialNotEvaluable ----------------------------------------------------------------------

/// `g(x) = x/2 + 1` (root 2) driven by a tangent `0.25` instead of `0.5`, so the full Newton step
/// from 0 overshoots to 4; anything above 3 cannot be evaluated.
fn overshooting() -> Synth {
    let mut sys = Synth::affine(vec![0.5], vec![1.0]);
    sys.tangent_diag = Some(vec![0.25]);
    sys.unevaluable = Box::new(|x| x[0] > 3.0);
    sys
}

const LINE_SEARCHED: [&str; 4] = [
    "newton",
    "newton-linesearch",
    "newton-held-factor",
    "modified-newton",
];

#[test]
fn a_line_search_shrinks_past_an_unevaluable_trial() {
    for name in LINE_SEARCHED {
        let mut sys = overshooting();
        let res = run(name, &mut sys, 50).unwrap();
        assert!(res.converged, "{name}");
        assert!((res.x[0] - 2.0).abs() < 1e-6, "{name}: {:?}", res.x);
    }
}

#[test]
fn an_unevaluable_trial_is_actually_hit() {
    // PLANTED CONTROL: without the refusal the full step lands at 4, beyond the refused region,
    // so the recovery above is exercised, not bypassed.
    let mut sys = overshooting();
    sys.unevaluable = Box::new(|_| false);
    sys.max_seen = f64::NEG_INFINITY;
    let _ = run("newton", &mut sys, 50).unwrap();
    assert!(sys.max_seen > 3.0, "{}", sys.max_seen);
}

#[test]
fn a_line_search_with_no_evaluable_trial_refuses() {
    for name in LINE_SEARCHED {
        let mut sys = overshooting();
        sys.unevaluable = Box::new(|x| x[0] > 1e-300);
        let err = run(name, &mut sys, 50).unwrap_err();
        assert!(
            matches!(err, NonlinearError::TrialNotEvaluable(_)),
            "{name}: {err:?}"
        );
    }
}

#[test]
fn a_strategy_without_a_line_search_aborts_on_an_unevaluable_trial() {
    for name in strategies()
        .into_iter()
        .filter(|n| !LINE_SEARCHED.contains(n))
    {
        let mut sys = overshooting();
        sys.unevaluable = Box::new(|x| x[0] > 0.5);
        let err = run(name, &mut sys, 50).unwrap_err();
        assert!(
            matches!(err, NonlinearError::TrialNotEvaluable(_)),
            "{name}: {err:?}"
        );
    }
}

// -- 3. held factor ----------------------------------------------------------------------------

/// A tangent 20% off (0.4 vs 0.5) makes every step contract linearly, so a held factor serves
/// several iterations.
fn modified() -> Synth {
    let mut sys = Synth::affine(vec![0.5], vec![1.0]);
    sys.tangent_diag = Some(vec![0.4]);
    sys
}

const HELD: [&str; 2] = ["newton-held-factor", "modified-newton"];

#[test]
fn a_held_factor_is_factorized_once_and_reused() {
    for name in HELD {
        let mut sys = modified();
        sys.holds_factor = true;
        let res = run(name, &mut sys, 50).unwrap();
        assert!(res.converged, "{name}");
        assert_eq!(
            sys.factorized, sys.tangents,
            "{name}: one factor per tangent"
        );
        assert!(
            sys.held_solves > sys.factorized,
            "{name}: {} solves on {} factors",
            sys.held_solves,
            sys.factorized
        );
        assert_eq!(sys.system_solves, 0, "{name} bypassed the held factor");
    }
}

#[test]
fn a_held_factor_matches_refactorisation() {
    for name in HELD {
        let mut plain = modified();
        let base = run(name, &mut plain, 50).unwrap();
        // PLANTED CONTROL: the plain run never touches the hook.
        assert_eq!(plain.factorized + plain.held_solves, 0, "{name}");
        assert!(plain.system_solves > 0, "{name}");
        let mut held = modified();
        held.holds_factor = true;
        let res = run(name, &mut held, 50).unwrap();
        assert_eq!(res.iterations, base.iterations, "{name}");
        for (a, b) in res.x.iter().zip(&base.x) {
            assert!((a - b).abs() <= 1e-12, "{name}: {a} vs {b}");
        }
    }
}

#[test]
fn full_newton_does_not_hold_a_factor() {
    let mut sys = modified();
    sys.holds_factor = true;
    let _ = run("newton", &mut sys, 50).unwrap();
    assert_eq!(sys.factorized, 0);
}

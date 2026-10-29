use super::*;
use crate::integrators::{Bdf1, Bdf2, Newmark, TimeIntegrator};

const TOL: f64 = 1e-12;

fn state_with(us: Vec<Vec<f64>>, t: f64, h: Option<f64>) -> IntegratorState {
    IntegratorState {
        us,
        t,
        v: None,
        a: None,
        h,
    }
}

/// The property the whole variable form exists for: at equal spacing it must BE the uniform
/// BDF2, or an event-free march is no longer bit-identical to a plain one.
#[test]
fn the_variable_form_reduces_to_uniform_bdf2_at_equal_spacing() {
    let dt = 0.1;
    let spaced = state_with(vec![vec![3.0, -1.0], vec![5.0, 2.0]], 0.2, Some(dt));
    let unspaced = state_with(vec![vec![3.0, -1.0], vec![5.0, 2.0]], 0.2, None);
    let variable = Bdf2.prepare(&spaced, dt, 0.3).unwrap();
    let uniform = Bdf2.prepare(&unspaced, dt, 0.3).unwrap();
    assert!((variable.weights.alpha1.re - uniform.weights.alpha1.re).abs() < TOL);
    for (a, b) in variable.h_dot.iter().zip(&uniform.h_dot) {
        assert!((a - b).abs() < TOL, "h_dot {a} vs {b}");
    }
    assert_eq!(variable.t_eval, uniform.t_eval);
}

/// The trap itself, made numerical. Against a QUADRATIC in t, BDF2 reading the spacing off the
/// state must recover the exact derivative at unequal spacing; the uniform weights (no spacing
/// on the state) cannot, and the gap is O(1) rather than a small order loss.
#[test]
fn the_variable_form_differentiates_a_quadratic_exactly_at_unequal_spacing() {
    // u(t) = 1 + 2t + 3t^2, sampled at t = 0, h, h + dt with h != dt.
    let (h, dt) = (0.4_f64, 0.1_f64);
    let u = |t: f64| 1.0 + 2.0 * t + 3.0 * t * t;
    let (t_nm1, t_n, t_new) = (0.0, h, h + dt);
    let state = state_with(vec![vec![u(t_nm1)], vec![u(t_n)]], t_n, Some(h));

    let form = Bdf2.prepare(&state, dt, t_new).unwrap();
    // u_dot = alpha1 * u_new + h_dot, evaluated at the true u(t_new)
    let got = form.weights.alpha1.re * u(t_new) + form.h_dot[0];
    let exact = 2.0 + 6.0 * t_new;
    assert!(
        (got - exact).abs() < 1e-10,
        "variable form {got} vs exact {exact}"
    );

    // ...and the uniform weights, at the SAME unequal spacing, are wrong by an O(1) factor.
    let blind = state_with(state.us.clone(), t_n, None);
    let wrong = Bdf2.prepare(&blind, dt, t_new).unwrap();
    let got_wrong = wrong.weights.alpha1.re * u(t_new) + wrong.h_dot[0];
    let relative = (got_wrong - exact).abs() / exact.abs();
    assert!(
        relative > 0.1,
        "the uniform form should be badly wrong here, but was off by only {relative}"
    );
}

/// A spacing on a one-state history is the bootstrap, not a variable step: there is no
/// quadratic through two points, so BDF2 takes its single BDF1 step.
#[test]
fn a_spacing_on_a_one_state_history_still_bootstraps_bdf1() {
    let state = state_with(vec![vec![1.0]], 0.0, Some(0.3));
    assert_eq!(
        Bdf2.prepare(&state, 0.1, 0.1).unwrap(),
        Bdf1.prepare(&state, 0.1, 0.1).unwrap()
    );
}

#[test]
fn the_variable_form_refuses_a_non_positive_spacing() {
    for (dt, h) in [(0.0, 0.1), (0.1, 0.0), (-0.1, 0.1), (0.1, f64::NAN)] {
        let state = state_with(vec![vec![1.0], vec![2.0]], 0.0, Some(h));
        assert!(
            matches!(
                Bdf2.prepare(&state, dt, 0.1),
                Err(IntegratorError::NonPositiveStep { .. })
            ),
            "dt={dt} h={h} was accepted"
        );
    }
}

/// Every step records its own spacing, so the NEXT step can read it; a start records none.
#[test]
fn an_advance_records_the_spacing_it_took() {
    let start = Bdf2.start(vec![1.0], 0.0, None, None).unwrap();
    assert_eq!(start.h, None);
    let one = Bdf2.advance(&start, vec![2.0], 0.1, 0.1).unwrap();
    assert_eq!(one.h, Some(0.1));
    let two = Bdf1.advance(&one, vec![3.0], 0.25, 0.35).unwrap();
    assert_eq!(two.h, Some(0.25));
}

// --- checkpoint / rollback -----------------------------------------------------------------

#[test]
fn a_checkpoint_survives_the_history_moving_underneath_it() {
    let mut state = state_with(vec![vec![1.0, 2.0]], 0.0, Some(0.1));
    let cp = Checkpoint::take(&state, 7);
    // The march moves on...
    state.us.push(vec![9.0, 9.0]);
    state.t = 0.1;
    state.h = Some(0.2);
    let restored = cp.restore();
    assert_eq!(restored.us, vec![vec![1.0, 2.0]]);
    assert_eq!(restored.t, 0.0);
    assert_eq!(restored.h, Some(0.1));
    assert_eq!(cp.n_points(), 7);
}

#[test]
fn a_checkpoint_carries_the_previous_spacing_because_bdf2_reads_it() {
    // A rollback that lost the spacing would restore the right numbers against the wrong
    // spacing, which the variable form would then silently apply. The spacing is IN the state.
    let none = state_with(vec![vec![1.0], vec![2.0]], 0.3, None);
    assert_eq!(Checkpoint::take(&none, 0).restore().h, None);
    let spaced = state_with(vec![vec![1.0], vec![2.0]], 0.3, Some(0.25));
    assert_eq!(Checkpoint::take(&spaced, 0).restore().h, Some(0.25));
}

#[test]
fn a_second_order_checkpoint_round_trips_v_and_a() {
    let state = IntegratorState {
        us: vec![vec![1.0]],
        t: 0.0,
        v: Some(vec![0.5]),
        a: Some(vec![-0.25]),
        h: None,
    };
    let restored = Checkpoint::take(&state, 0).restore();
    assert_eq!(restored.v, Some(vec![0.5]));
    assert_eq!(restored.a, Some(vec![-0.25]));
}

// --- the sign bit and the locator ----------------------------------------------------------

#[test]
fn a_landing_exactly_on_zero_reads_as_the_far_side() {
    // Otherwise a probe that lands precisely on the crossing bisects forever.
    assert!(event_sign(0.0));
    assert!(event_sign(1e-300));
    assert!(!event_sign(-1e-300));
}

/// The locator on an exactly-known crossing: `u(t) = t`, `g = u - 0.3`, one step of 1.0.
#[test]
fn the_locator_finds_a_linear_crossing_to_tolerance() {
    let t0 = 0.0;
    let g_start = [-0.3];
    let found = locate_event(
        t0,
        1.0,
        &g_start,
        |dt| Ok(vec![t0 + dt]),
        |u, _t, _i| u[0] - 0.3,
        1,
        Bisection {
            tol: 1e-9,
            max_probes: 200,
        },
    )
    .unwrap()
    .expect("the crossing is inside the step");
    assert_eq!(found.function, 0);
    assert!((found.time - 0.3).abs() < 1e-8, "located at {}", found.time);
}

#[test]
fn no_sign_change_across_the_step_is_no_event() {
    let found = locate_event(
        0.0,
        1.0,
        &[-5.0],
        |dt| Ok(vec![dt]),
        |u, _t, _i| u[0] - 9.0,
        1,
        Bisection {
            tol: 1e-9,
            max_probes: 200,
        },
    )
    .unwrap();
    assert!(found.is_none());
}

/// Two functions flipping in one step: only the EARLIER is a step the march may take.
#[test]
fn the_earliest_crossing_wins_when_two_functions_flip() {
    let g_start = [-0.2, -0.8];
    let found = locate_event(
        0.0,
        1.0,
        &g_start,
        |dt| Ok(vec![dt]),
        |u, _t, i| if i == 0 { u[0] - 0.2 } else { u[0] - 0.8 },
        2,
        Bisection {
            tol: 1e-9,
            max_probes: 200,
        },
    )
    .unwrap()
    .expect("both cross");
    assert_eq!(found.function, 0, "the later function was returned");
    assert!((found.time - 0.2).abs() < 1e-8);
}

#[test]
fn the_locator_bisects_the_step_size_not_an_interpolant() {
    // The probe sizes it asks for must be step sizes it could actually march, and the state it
    // reports must come from `step_to` -- never from blending the two endpoints. Recording the
    // requests is how that stays true under a later refactor.
    let mut asked: Vec<f64> = Vec::new();
    let found = locate_event(
        0.0,
        1.0,
        &[-0.5],
        |dt| {
            asked.push(dt);
            Ok(vec![dt])
        },
        |u, _t, _i| u[0] - 0.5,
        1,
        Bisection {
            tol: 1e-6,
            max_probes: 200,
        },
    )
    .unwrap()
    .unwrap();
    assert!((found.time - 0.5).abs() < 1e-5);
    assert!(asked.len() > 3, "no bisection happened: {asked:?}");
    assert!(
        asked.iter().all(|&dt| (0.0..=1.0).contains(&dt)),
        "a probe asked for a step outside the bracket: {asked:?}"
    );
}

#[test]
fn a_bracket_that_cannot_narrow_in_the_allowed_probes_is_an_error_rather_than_a_hang() {
    // A real crossing, but a tolerance the allowed probe count cannot reach: bisection halves
    // the bracket each time, so 8 probes leave 1/256 -- far above 1e-30. Saying "not locatable
    // to that tolerance" is the honest answer; looping until it is would be a hang wearing a
    // computation's face.
    let err = locate_event(
        0.0,
        1.0,
        &[-0.5],
        |dt| Ok(vec![dt]),
        |u, _t, _i| u[0] - 0.5,
        1,
        Bisection {
            tol: 1e-30,
            max_probes: 8,
        },
    );
    match err {
        Err(MarchError::BisectionExhausted {
            function,
            iterations,
        }) => assert_eq!((function, iterations), (0, 8)),
        other => panic!("expected BisectionExhausted, got {other:?}"),
    }
}

// --- the chatter guard ---------------------------------------------------------------------

#[test]
fn the_chatter_guard_admits_a_dense_but_genuine_event_stream() {
    // The continuous-PI reproduction samples ~200 times per nominal cell. That is dense and
    // legitimate, so it must NOT trip.
    let mut guard = ChatterGuard::default();
    for _ in 0..200 {
        guard.observe(0, 0).unwrap();
    }
    assert_eq!(guard.seen_in_cell(), 200);
}

#[test]
fn the_chatter_guard_refuses_a_function_retriggering_on_its_own_zero() {
    let mut guard = ChatterGuard::new(4);
    for _ in 0..4 {
        guard.observe(0, 2).unwrap();
    }
    let err = guard.observe(0, 2).unwrap_err();
    match err {
        MarchError::Chattering {
            function,
            events_in_cell,
            ceiling,
        } => {
            assert_eq!((function, events_in_cell, ceiling), (2, 5, 4));
        }
        other => panic!("expected Chattering, got {other}"),
    }
    assert!(err.to_string().contains("hysteresis"));
}

#[test]
fn the_count_is_per_cell_so_a_new_cell_starts_clean() {
    let mut guard = ChatterGuard::new(3);
    for _ in 0..3 {
        guard.observe(0, 0).unwrap();
    }
    guard.observe(1, 0).unwrap(); // a new cell -- not the fourth strike
    assert_eq!(guard.seen_in_cell(), 1);
}

// --- the restart -----------------------------------------------------------------------------

#[test]
fn the_restart_drops_the_derivative_history_and_keeps_the_state() {
    // The companion trap: after the restart the next step must be BDF1, carrying no pre-event
    // slope. A two-state history would let the pre-toggle quadratic reach across the event.
    let before = IntegratorState {
        us: vec![vec![1.0], vec![2.0]],
        t: 0.7,
        v: None,
        a: None,
        h: Some(0.05),
    };
    let after = restart_at_event(&Bdf2, &before, 0.75).unwrap();
    assert_eq!(after.us, vec![vec![2.0]], "the older state must be dropped");
    assert_eq!(after.t, 0.75);
    assert_eq!(after.h, None, "a restart carries no spacing: the history it describes is gone");
    // ...and the proof that it is BDF1: Bdf2's own prepare bootstraps on a one-state history.
    let restarted = Bdf2.prepare(&after, 0.05, 0.8).unwrap();
    let bdf1 = Bdf1.prepare(&after, 0.05, 0.8).unwrap();
    assert_eq!(restarted, bdf1);
}

#[test]
fn a_second_order_restart_preserves_continuous_state() {
    // The physical state did not jump -- only the model did -- so v and a are CARRIED.
    let integ = Newmark::default();
    let before = IntegratorState {
        us: vec![vec![1.0, -1.0]],
        t: 0.2,
        v: Some(vec![0.5, 0.25]),
        a: Some(vec![-0.1, 0.0]),
        h: Some(0.05),
    };
    let after = restart_at_event(&integ, &before, 0.25).unwrap();
    assert_eq!(after.v, before.v);
    assert_eq!(after.a, before.a);
    assert_eq!(after.us, vec![vec![1.0, -1.0]]);
    assert_eq!(after.t, 0.25);
}

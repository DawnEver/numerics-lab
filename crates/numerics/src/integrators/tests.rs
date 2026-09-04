use super::*;

const TOL: f64 = 1e-12;

/// March the scalar decay `u' = -lambda u` and return the final value.
///
/// The affine contract makes the step solvable in closed form for this scalar problem:
/// the residual is `u_dot + lambda*u_arg = 0`, and with `u_dot = alpha1*u + h_dot`,
/// `u_arg = alpha0*u + h_u` that is `(alpha1 + lambda*alpha0) u = -(h_dot + lambda*h_u)`.
/// So this exercises the SAME numbers a real driver would hand a linear solver.
fn march_decay(integ: &dyn TimeIntegrator, lambda: f64, t_end: f64, n_step: usize) -> f64 {
    let dt = t_end / n_step as f64;
    let mut state = integ
        .start(vec![1.0], 0.0, Some(vec![-lambda]), None)
        .unwrap();
    for k in 0..n_step {
        let t_new = (k + 1) as f64 * dt;
        let form = integ.prepare(&state, dt, t_new).unwrap();
        let a0 = form.weights.alpha0.re;
        let a1 = form.weights.alpha1.re;
        let u = -(form.h_dot[0] + lambda * form.h_u[0]) / (a1 + lambda * a0);
        state = integ.advance(&state, vec![u], dt, t_new).unwrap();
    }
    state.last_u()[0]
}

#[test]
fn bdf1_is_first_order_and_bdf2_is_second_on_the_analytic_decay() {
    let exact = (-1.0_f64).exp(); // u' = -u, t_end = 1
    let err = |integ: &dyn TimeIntegrator, n| (march_decay(integ, 1.0, 1.0, n) - exact).abs();
    // Halving dt must cut the error by ~2 for a first-order scheme and ~4 for second order.
    let r1 = err(&Bdf1, 40) / err(&Bdf1, 80);
    let r2 = err(&Bdf2, 40) / err(&Bdf2, 80);
    assert!((1.7..2.4).contains(&r1), "bdf1 order ratio {r1}");
    assert!((3.2..4.8).contains(&r2), "bdf2 order ratio {r2}");
}

#[test]
fn bdf2_bootstraps_with_a_bdf1_step_when_it_has_only_one_state() {
    let state = Bdf2.start(vec![1.0, 2.0], 0.0, None, None).unwrap();
    let boot = Bdf2.prepare(&state, 0.1, 0.1).unwrap();
    let bdf1 = Bdf1.prepare(&state, 0.1, 0.1).unwrap();
    assert_eq!(boot, bdf1, "the first BDF2 step must BE a BDF1 step");
    // ...and once it has two, it is no longer BDF1.
    let two = Bdf2.advance(&state, vec![0.9, 1.8], 0.1, 0.1).unwrap();
    let real = Bdf2.prepare(&two, 0.1, 0.2).unwrap();
    assert_ne!(real.weights, bdf1.weights);
    assert!((real.weights.alpha1.re - 15.0).abs() < TOL); // 1.5/0.1
}

#[test]
fn bdf2_keeps_exactly_two_past_states() {
    let mut state = Bdf2.start(vec![0.0], 0.0, None, None).unwrap();
    for k in 1..6 {
        state = Bdf2
            .advance(&state, vec![f64::from(k)], 0.1, f64::from(k) * 0.1)
            .unwrap();
        assert!(state.us.len() <= 2, "history grew to {}", state.us.len());
    }
    assert_eq!(state.us, vec![vec![4.0], vec![5.0]]);
}

#[test]
fn the_newmark_defaults_give_the_average_acceleration_weights() {
    let w = Newmark::default().weights(0.5).unwrap();
    assert!((w.alpha1.re - 2.0 / 0.5).abs() < TOL);
    assert!((w.alpha2.re - 4.0 / (0.5 * 0.5)).abs() < TOL);
}

/// The property the affine contract exists to have: `advance` must reproduce exactly the
/// `v` and `a` that `prepare`'s offsets imply, or the history and the step form disagree.
#[test]
fn newmark_advance_agrees_with_its_own_prepared_form() {
    let integ = Newmark::default();
    let state = integ
        .start(
            vec![1.0, -2.0],
            0.0,
            Some(vec![0.5, 0.0]),
            Some(vec![0.0, 0.3]),
        )
        .unwrap();
    let dt = 0.05;
    let form = integ.prepare(&state, dt, dt).unwrap();
    let u_new = vec![1.1, -1.8];
    let next = integ.advance(&state, u_new.clone(), dt, dt).unwrap();
    let w = integ.weights(dt).unwrap();
    for (i, &u) in u_new.iter().enumerate() {
        let a_expected = w.alpha2.re * u + form.h_ddot[i];
        let v_expected = w.alpha1.re * u + form.h_dot[i];
        assert!((next.a.as_ref().unwrap()[i] - a_expected).abs() < TOL);
        assert!((next.v.as_ref().unwrap()[i] - v_expected).abs() < TOL);
    }
}

#[test]
fn generalized_alpha_at_rho_one_recovers_undamped_newmark_weights() {
    // rho_inf = 1 gives am = 1/2, af = 1/2, so beta = 1/4 and gamma = 1/2 -- the
    // average-acceleration pair, with alpha0 = alpha2-scaling both halved by (1 - a).
    let ga = GeneralizedAlpha::new(1.0).unwrap();
    assert!((ga.beta - NEWMARK_BETA).abs() < TOL, "beta {}", ga.beta);
    assert!((ga.gamma - NEWMARK_GAMMA).abs() < TOL, "gamma {}", ga.gamma);
    let dt = 0.25;
    let g = ga.weights(dt).unwrap();
    let n = Newmark::default().weights(dt).unwrap();
    // Every weight is exactly half the Newmark one at af = am = 1/2.
    assert!((g.alpha0.re - 0.5).abs() < TOL);
    assert!((g.alpha1.re - 0.5 * n.alpha1.re).abs() < TOL);
    assert!((g.alpha2.re - 0.5 * n.alpha2.re).abs() < TOL);
}

#[test]
fn generalized_alpha_is_the_only_scheme_that_shifts_the_evaluation_time() {
    let state = |i: &dyn TimeIntegrator| {
        i.start(vec![0.0], 0.0, Some(vec![0.0]), Some(vec![0.0]))
            .unwrap()
    };
    for integ in [
        Box::new(Bdf1) as Box<dyn TimeIntegrator>,
        Box::new(Bdf2),
        Box::new(Newmark::default()),
    ] {
        let s = state(integ.as_ref());
        assert!((integ.prepare(&s, 0.1, 0.1).unwrap().t_eval - 0.1).abs() < TOL);
    }
    let ga = GeneralizedAlpha::new(0.8).unwrap();
    let s = state(&ga);
    let form = ga.prepare(&s, 0.1, 0.1).unwrap();
    assert!(
        form.t_eval < 0.1,
        "t_eval {} was not shifted back",
        form.t_eval
    );
    // ...and h_u is nonzero only here, which is why StepForm carries it.
    let s2 = ga.start(vec![3.0], 0.0, None, None).unwrap();
    assert!(ga.prepare(&s2, 0.1, 0.1).unwrap().h_u[0].abs() > 0.0);
    assert_eq!(Bdf1.prepare(&s2, 0.1, 0.1).unwrap().h_u, vec![0.0]);
}

#[test]
fn a_spectral_radius_outside_the_unit_interval_is_refused() {
    for bad in [-0.1, 1.1, f64::NAN] {
        assert!(matches!(
            GeneralizedAlpha::new(bad).unwrap_err(),
            IntegratorError::SpectralRadiusOutOfRange { .. }
        ));
    }
}

#[test]
fn a_non_positive_step_is_refused_by_every_scheme() {
    for integ in [
        Box::new(Bdf1) as Box<dyn TimeIntegrator>,
        Box::new(Bdf2),
        Box::new(Newmark::default()),
        Box::new(GeneralizedAlpha::default()),
    ] {
        for dt in [0.0, -1.0] {
            let refused = matches!(
                integ.weights(dt),
                Err(IntegratorError::NonPositiveStep { .. })
            );
            assert!(refused, "{} accepted dt={dt}", integ.name());
        }
    }
}

#[test]
fn a_second_order_scheme_refuses_a_state_with_no_velocity_history() {
    // A first-order state handed to Newmark. Defaulting v/a to zero here would silently
    // change the trajectory, so it is a refusal.
    let first_order = Bdf1.start(vec![1.0], 0.0, None, None).unwrap();
    for integ in [
        Box::new(Newmark::default()) as Box<dyn TimeIntegrator>,
        Box::new(GeneralizedAlpha::default()),
    ] {
        assert!(matches!(
            integ.prepare(&first_order, 0.1, 0.1),
            Err(IntegratorError::MissingSecondOrderHistory { .. })
        ));
    }
}

#[test]
fn every_known_name_resolves_and_an_unknown_one_is_refused_by_name() {
    for name in KNOWN_INTEGRATORS {
        assert_eq!(get_integrator(name).unwrap().name(), name);
    }
    let err = get_integrator("rk4").unwrap_err();
    assert!(matches!(err, IntegratorError::UnknownKind { .. }));
    assert!(err.to_string().contains("rk4"));
}

#[test]
fn a_mismatched_initial_velocity_is_refused_rather_than_broadcast() {
    assert!(matches!(
        Newmark::default().start(vec![1.0, 2.0], 0.0, Some(vec![1.0]), None),
        Err(IntegratorError::ShapeMismatch { .. })
    ));
}

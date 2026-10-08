//! Local element convergence and original-source inexact Newton policy.

use super::*;

#[test]
fn a_small_element_zero_crossing_is_not_hidden_by_the_bulk() {
    let old = vec![vec![vec![100., 0.], vec![0., 0.]]];
    let new = vec![vec![vec![100., 0.], vec![0.001, 0.]]];
    assert_eq!(local_change(&old, &new).unwrap(), 1.);
    assert_eq!(local_change(&old, &old).unwrap(), 0.);
}

#[test]
fn independently_scaled_fields_preserve_units_and_refuse_shape_changes() {
    let old = vec![vec![vec![1e9]], vec![vec![1e-9]]];
    let new = vec![vec![vec![1e9]], vec![vec![2e-9]]];
    assert_eq!(local_change(&old, &new).unwrap(), 0.5);
    assert!(local_change(&old, &[]).is_err());
    assert!(local_change(&[vec![vec![0.]]], &[vec![vec![f64::NAN]]])
        .unwrap()
        .is_infinite());
}

#[test]
fn original_rhs_tolerance_tightens_and_rejects_nonfinite_ratios() {
    assert_eq!(newton_linear_tolerance(1.).unwrap(), 1e-2);
    assert!((newton_linear_tolerance(1e-4).unwrap() - 1e-7).abs() < 1e-20);
    assert_eq!(newton_linear_tolerance(0.).unwrap(), 1e-10);
    assert!(newton_linear_tolerance(f64::NAN).is_err());
    assert!(newton_linear_tolerance(-1.).is_err());
}

#[test]
fn local_unit_scaling_does_not_underflow_or_overflow() {
    for scale in [1e-200, 1e200] {
        let old = vec![vec![vec![scale, scale]]];
        let new = vec![vec![vec![2. * scale, 2. * scale]]];
        assert!((local_change(&old, &new).unwrap() - 0.5).abs() < 1e-14);
    }
}

#[test]
fn budgeted_newton_obeys_small_element_convergence_and_forwards_tolerances() {
    let mut sys = Synth::affine(vec![0., 0.], vec![1e3, 1e-3]);
    sys.local_fields = true;
    sys.tangent_diag = Some(vec![1., 2.]);
    let mut wrapped = with_time_budget(sys, 100., || 0.);
    let result = NewtonStrategy::default()
        .solve(&mut wrapped, &opts(30))
        .unwrap();
    assert!(result.converged);
    assert!(
        result.iterations > 10,
        "global residual must not hide the small element"
    );
    assert!((result.x[1] / 1e-3 - 1.).abs() < 1e-4);
    let recorded = &wrapped.unwrapped().linear_tolerances;
    assert_eq!(recorded.len(), result.iterations);
    let reference = norm(&[1e3, 1e-3]);
    assert_eq!(recorded[0], reference * 1e-2);
    assert!(recorded.windows(2).all(|pair| pair[1] <= pair[0]));
    assert!(recorded.iter().all(|&x| x >= reference * 1e-10));
}

#[test]
fn a_converged_fields_roundoff_is_not_a_change() {
    let old = vec![vec![vec![1.82], vec![1.084_202_172_485_502_9e-15]]];
    let new = vec![vec![vec![1.82], vec![-1.084_202_172_485_502_9e-15]]];
    assert!(local_change(&old, &new).unwrap() < 1e-6);
}

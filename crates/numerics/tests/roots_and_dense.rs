//! The homes the audit of #59 asked for: `brentq` under `nonlinear::roots`, the row-major matrix under `dense`.

use numerics::dense::matrix::{vecmat, Dense, ShapeError};
use numerics::nonlinear::roots::{brentq, BrentError};
use std::convert::Infallible;

#[test]
fn brentq_matches_scipy_on_a_cubic() {
    // scipy.optimize.brentq(lambda x: x**3 - 2, 0, 2, xtol=1e-12): 1.2599210498948732
    let (root, converged) = brentq::<_, Infallible>(
        |x| Ok(x * x * x - 2.0),
        0.0,
        2.0,
        1e-12,
        4.0 * f64::EPSILON,
        100,
    )
    .unwrap();
    assert!(converged);
    assert!((root - 2f64.cbrt()).abs() <= 1e-12, "{root}");
}

#[test]
fn a_bracket_without_a_sign_change_is_refused_and_a_function_error_passes_through() {
    let e = brentq::<_, Infallible>(|x| Ok(x * x + 1.0), -1.0, 1.0, 1e-12, 1e-15, 100).unwrap_err();
    assert!(matches!(e, BrentError::Bracket { .. }));
    assert_eq!(
        e.to_string(),
        "f(a) and f(b) must have different signs (a = -1e0, b = 1e0)"
    );
    let e = brentq(
        |_| Err::<f64, &str>("refused"),
        -1.0,
        1.0,
        1e-12,
        1e-15,
        100,
    )
    .unwrap_err();
    assert_eq!(e, BrentError::Function("refused"));
}

#[test]
fn a_dense_product_and_its_shape_refusals() {
    let a = Dense::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    assert_eq!(a.mul(&a).unwrap().data, vec![7.0, 10.0, 15.0, 22.0]);
    assert_eq!(a.apply(&[1.0, 1.0]), vec![3.0, 7.0]);
    assert_eq!(vecmat(&[1.0, 1.0], &a), vec![4.0, 6.0]);
    assert!(matches!(Dense::new(2, 2, vec![1.0]), Err(ShapeError(_))));
    assert!(a.mul(&Dense::zeros(3, 1)).is_err());
}

//! Independently scaled local convergence: magnetic, voltage and current units never mix.

use super::NonlinearError;

/// Fields -> elements -> vector components. Each field has its own rounding floor.
pub type LocalFields = Vec<Vec<Vec<f64>>>;

/// Worst element-relative change, using the Python numerical policy unchanged.
pub fn local_change(
    previous: &[Vec<Vec<f64>>],
    current: &[Vec<Vec<f64>>],
) -> Result<f64, NonlinearError> {
    if previous.len() != current.len() {
        return Err(NonlinearError::ShapeMismatch {
            expected: previous.len(),
            got: current.len(),
        });
    }
    let mut maximum = 0.0_f64;
    let norm = |values: &[f64]| values.iter().fold(0.0_f64, |sum, &x| sum.hypot(x));
    for (old, new) in previous.iter().zip(current) {
        if old.len() != new.len() {
            return Err(NonlinearError::ShapeMismatch {
                expected: old.len(),
                got: new.len(),
            });
        }
        let mut changes = Vec::with_capacity(old.len());
        let mut scales = Vec::with_capacity(old.len());
        for (a, b) in old.iter().zip(new) {
            if a.len() != b.len() {
                return Err(NonlinearError::ShapeMismatch {
                    expected: a.len(),
                    got: b.len(),
                });
            }
            if !a.iter().chain(b).all(|x| x.is_finite()) {
                return Ok(f64::INFINITY);
            }
            changes.push(norm(
                &a.iter().zip(b).map(|(x, y)| y - x).collect::<Vec<_>>(),
            ));
            scales.push(norm(a).max(norm(b)));
        }
        if !changes.iter().chain(&scales).all(|x| x.is_finite()) {
            return Ok(f64::INFINITY);
        }
        // sqrt(eps) of the field's scale, as the Python: a solved field's roundoff is a few ulps,
        // and an element at that noise flips sign between two converged iterates.
        let floor = f64::EPSILON.sqrt() * scales.iter().copied().fold(0.0_f64, f64::max);
        for (change, scale) in changes.into_iter().zip(scales) {
            if scale > 0.0 {
                maximum = maximum.max(change / scale.max(floor));
            }
        }
    }
    Ok(maximum)
}

/// Absolute tolerance fraction relative to the original source, never the new Newton RHS.
pub fn newton_linear_tolerance(residual_ratio: f64) -> Result<f64, NonlinearError> {
    if !residual_ratio.is_finite() || residual_ratio < 0.0 {
        return Err(NonlinearError::Equation(
            "invalid Newton tolerance residual ratio".to_owned(),
        ));
    }
    Ok((0.1 * residual_ratio.powf(1.5)).clamp(1e-10, 1e-2))
}

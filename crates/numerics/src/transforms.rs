//! Park / Clarke coordinate transforms — the Rust mirror of `src/motronics/gauss/transforms.py`.
//!
//! The Park family is ORTHOGONAL COORDINATE ROTATION, not physics: rotating a balanced N-phase
//! set into the rotor frame is the same linear algebra whoever asks for it (a field
//! post-processing route, a control law, a drive signal chain). That is why it lives in the
//! physics-free runtime and why both platforms above can reach it.
//!
//! All transforms are AMPLITUDE-INVARIANT: a balanced set of peak amplitude `I` maps to a
//! `(d, q)` vector of magnitude `I`. That convention is load-bearing — [`phase_l_to_dq_l`]
//! below is wrong by a factor of `2/n` if it is forgotten, and the Python carries a measured
//! counterexample saying so.
//!
//! # Layout
//!
//! `x_phase` is flat row-major `(n_phase, n_theta)`: phase-major, so phase `k`'s samples are
//! contiguous. That is the layout numpy already hands over for a C-contiguous array of this
//! shape, and it is the one the sums below walk with unit stride.
//!
//! # What numpy's broadcasting bought, and what replaces it
//!
//! The Python accepts scalars, `(n,)` and `(n_phase, n)` and reconciles them by broadcasting.
//! Two of its guards exist purely because that flexibility can silently consume the wrong axis
//! — a `(3, n)` `theta_e` whose leading axis happens to equal `n_phase`, and a `(3,)` command
//! vector with a scalar angle that means three SAMPLES rather than three phases. Both are
//! recorded there as fixed defects (SR-031, SR-024).
//!
//! Rust has no broadcasting, so those two bugs are not reachable here: the sample count is an
//! explicit argument and the phase axis is never inferred. The guards are therefore not
//! ported — but the REASON is, in this paragraph, so that a later reader does not restore the
//! flexibility and with it the defect.

use std::fmt;

/// Every way a transform can be refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TransformError {
    /// A phase count below three cannot carry a `(d, q, zero)` decomposition.
    TooFewPhases { n_phase: usize },
    /// A flat buffer whose length contradicts the declared `(n_phase, n_theta)` shape.
    ShapeMismatch {
        what: &'static str,
        expected: usize,
        got: usize,
    },
    /// A square matrix argument that is not square.
    NotSquare { rows: usize, len: usize },
}

impl fmt::Display for TransformError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooFewPhases { n_phase } => write!(
                f,
                "a symmetric set needs at least 3 phases to carry (d, q, zero); got {n_phase}"
            ),
            Self::ShapeMismatch {
                what,
                expected,
                got,
            } => write!(f, "{what} must hold {expected} values; got {got}"),
            Self::NotSquare { rows, len } => write!(
                f,
                "L_phase must be square: {rows} rows implies {} entries, got {len}",
                rows * rows
            ),
        }
    }
}

impl std::error::Error for TransformError {}

/// Electrical axis angle of each of `n_phase` symmetric phases: `2*pi*k/n_phase`.
fn phase_axes(n_phase: usize) -> Vec<f64> {
    (0..n_phase)
        .map(|k| 2.0 * std::f64::consts::PI * k as f64 / n_phase as f64)
        .collect()
}

/// The rotor-frame components of an N-phase set.
#[derive(Debug, Clone, PartialEq)]
pub struct Dq0 {
    /// Direct axis, one per sample.
    pub d: Vec<f64>,
    /// Quadrature axis, one per sample.
    pub q: Vec<f64>,
    /// Zero sequence — the phase mean, one per sample.
    pub zero: Vec<f64>,
}

/// Park transform: N-phase stator quantities -> rotor-frame `(d, q, zero)`.
///
/// Projects the symmetric N-phase set (phase `k` on electrical axis `2*pi*k/N`) onto the
/// rotor-aligned direct and quadrature axes. Amplitude-invariant via the `2/N` scaling, so a
/// balanced set of peak amplitude `I` aligned with the rotor gives `d = I, q = 0` — the d-q
/// amplitudes equal the phase amplitude. The zero sequence is the phase mean.
///
/// `x_phase` is flat `(n_phase, n_theta)`, `theta_e` is the rotor ELECTRICAL angle per sample.
///
/// # Errors
/// [`TransformError::TooFewPhases`] or [`TransformError::ShapeMismatch`].
pub fn park_forward(
    x_phase: &[f64],
    n_phase: usize,
    theta_e: &[f64],
) -> Result<Dq0, TransformError> {
    if n_phase < 3 {
        return Err(TransformError::TooFewPhases { n_phase });
    }
    let n_theta = theta_e.len();
    if x_phase.len() != n_phase * n_theta {
        return Err(TransformError::ShapeMismatch {
            what: "x_phase (n_phase, n_theta)",
            expected: n_phase * n_theta,
            got: x_phase.len(),
        });
    }
    let axes = phase_axes(n_phase);
    let scale = 2.0 / n_phase as f64;
    let mut out = Dq0 {
        d: Vec::with_capacity(n_theta),
        q: Vec::with_capacity(n_theta),
        zero: Vec::with_capacity(n_theta),
    };
    for (t, &theta) in theta_e.iter().enumerate() {
        let (mut d, mut q, mut sum) = (0.0, 0.0, 0.0);
        for (k, &ax) in axes.iter().enumerate() {
            let x = x_phase[k * n_theta + t];
            let angle = theta - ax;
            d += x * angle.cos();
            q -= x * angle.sin();
            sum += x;
        }
        out.d.push(scale * d);
        out.q.push(scale * q);
        out.zero.push(sum / n_phase as f64);
    }
    Ok(out)
}

/// Inverse Park: rotor-frame `(d, q, zero)` -> flat `(n_phase, n_theta)` phase quantities.
///
/// The exact inverse of [`park_forward`] for the amplitude-invariant convention: phase `k` is
/// `d*cos(theta_e - axis_k) - q*sin(theta_e - axis_k) + zero`. It round-trips to machine
/// precision for any `n_phase >= 3` — the forward map has a d, a q and a zero degree of
/// freedom, and the remaining `n_phase - 3` phase harmonics vanish for a symmetric set, which
/// is the standard balanced-machine assumption.
///
/// # Errors
/// [`TransformError::TooFewPhases`] or [`TransformError::ShapeMismatch`] when `d`, `q`, `zero`
/// and `theta_e` do not all carry the same sample count.
pub fn park_inverse(
    dq0: &Dq0,
    theta_e: &[f64],
    n_phase: usize,
) -> Result<Vec<f64>, TransformError> {
    if n_phase < 3 {
        return Err(TransformError::TooFewPhases { n_phase });
    }
    let n_theta = theta_e.len();
    for (what, got) in [
        ("d", dq0.d.len()),
        ("q", dq0.q.len()),
        ("zero", dq0.zero.len()),
    ] {
        if got != n_theta {
            return Err(TransformError::ShapeMismatch {
                what: match what {
                    "d" => "d (n_theta)",
                    "q" => "q (n_theta)",
                    _ => "zero (n_theta)",
                },
                expected: n_theta,
                got,
            });
        }
    }
    let axes = phase_axes(n_phase);
    let mut out = vec![0.0; n_phase * n_theta];
    for (k, &ax) in axes.iter().enumerate() {
        for t in 0..n_theta {
            let angle = theta_e[t] - ax;
            out[k * n_theta + t] = dq0.d[t] * angle.cos() - dq0.q[t] * angle.sin() + dq0.zero[t];
        }
    }
    Ok(out)
}

/// Amplitude-invariant Clarke transform: 3-phase `(a, b, c)` -> stationary `(alpha, beta)`.
///
/// `alpha = (2/3)*(a - b/2 - c/2)`, `beta = (2/3)*(sqrt(3)/2)*(b - c)`. For a balanced set of
/// peak `I` the `(alpha, beta)` vector has magnitude `I`, matching [`park_forward`]'s
/// convention. `x_phase` is flat `(3, n_theta)`.
///
/// # Errors
/// [`TransformError::ShapeMismatch`] when `x_phase` is not `3 * n_theta` long.
pub fn clarke_forward(
    x_phase: &[f64],
    n_theta: usize,
) -> Result<(Vec<f64>, Vec<f64>), TransformError> {
    if x_phase.len() != 3 * n_theta {
        return Err(TransformError::ShapeMismatch {
            what: "x_phase (3, n_theta)",
            expected: 3 * n_theta,
            got: x_phase.len(),
        });
    }
    let k = 2.0 / 3.0;
    let kb = k * (3.0_f64.sqrt() / 2.0);
    let mut alpha = Vec::with_capacity(n_theta);
    let mut beta = Vec::with_capacity(n_theta);
    for t in 0..n_theta {
        let (a, b, c) = (x_phase[t], x_phase[n_theta + t], x_phase[2 * n_theta + t]);
        alpha.push(k * (a - 0.5 * b - 0.5 * c));
        beta.push(kb * (b - c));
    }
    Ok((alpha, beta))
}

/// The Park bridge: a phase-basis inductance matrix -> the dq-basis matrix at `theta_e`.
///
/// The phase-basis `L_phase` (`(n, n)`) and the dq-basis `L_dq` (`(2, 2)`) are the SAME
/// inductance tensor in two bases — the dq report is not a second measurement protocol, it is
/// this rotation. The amplitude-invariant relation is `L_dq = (2/n) M L M^T` where `M` is the
/// `2 x n` forward-Park matrix with rows `[cos(theta_e - axis_k); -sin(theta_e - axis_k)]`.
///
/// **Why the `(2/n)` and not a bare `T L T^T`.** The full forward-Park matrix already carries
/// the amplitude-invariant `2/n` in each of its d and q rows, so a bare `T L T^T` applies it
/// TWICE and maps a balanced diagonal `L_phase` to dq entries `2/n` too small. The Python
/// records the measurement: `L_phase = diag(1, 1, 1)` with balanced mutuals gives `Ld = 1.2`
/// through the direct injection and `0.8` through `T L T^T`. This bridge uses the
/// UN-normalised `M` and a single `2/n`, so it reproduces the direct dq-injection route.
///
/// Returns flat `(n_theta, 2, 2)`.
///
/// # Errors
/// [`TransformError::NotSquare`] or [`TransformError::TooFewPhases`].
pub fn phase_l_to_dq_l(
    l_phase: &[f64],
    n_phase: usize,
    theta_e: &[f64],
) -> Result<Vec<f64>, TransformError> {
    if n_phase < 3 {
        return Err(TransformError::TooFewPhases { n_phase });
    }
    if l_phase.len() != n_phase * n_phase {
        return Err(TransformError::NotSquare {
            rows: n_phase,
            len: l_phase.len(),
        });
    }
    let axes = phase_axes(n_phase);
    let scale = 2.0 / n_phase as f64;
    let mut out = vec![0.0; theta_e.len() * 4];
    for (t, &theta) in theta_e.iter().enumerate() {
        // M is 2 x n: row 0 the d projection, row 1 the q projection.
        let m: Vec<[f64; 2]> = axes
            .iter()
            .map(|&ax| [(theta - ax).cos(), -(theta - ax).sin()])
            .collect();
        // (M L) is 2 x n, then (M L) M^T is 2 x 2.
        for r in 0..2 {
            let ml: Vec<f64> = (0..n_phase)
                .map(|j| {
                    (0..n_phase)
                        .map(|i| m[i][r] * l_phase[i * n_phase + j])
                        .sum()
                })
                .collect();
            for c in 0..2 {
                let v: f64 = (0..n_phase).map(|j| ml[j] * m[j][c]).sum();
                out[t * 4 + r * 2 + c] = scale * v;
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-12;

    /// A balanced N-phase set of peak `amp`, aligned so that `d = amp, q = 0` at `theta`.
    fn balanced(n_phase: usize, amp: f64, theta: &[f64]) -> Vec<f64> {
        let axes = phase_axes(n_phase);
        let mut x = vec![0.0; n_phase * theta.len()];
        for (k, &ax) in axes.iter().enumerate() {
            for (t, &th) in theta.iter().enumerate() {
                x[k * theta.len() + t] = amp * (th - ax).cos();
            }
        }
        x
    }

    #[test]
    fn a_balanced_set_aligned_with_the_rotor_is_all_d_and_amplitude_invariant() {
        let theta = [0.0, 0.7, 2.5, -1.3];
        for n in [3, 5, 6] {
            let x = balanced(n, 4.0, &theta);
            let r = park_forward(&x, n, &theta).unwrap();
            for t in 0..theta.len() {
                assert!((r.d[t] - 4.0).abs() < TOL, "n={n} d={}", r.d[t]);
                assert!(r.q[t].abs() < TOL, "n={n} q={}", r.q[t]);
                assert!(r.zero[t].abs() < TOL, "n={n} zero={}", r.zero[t]);
            }
        }
    }

    #[test]
    fn the_inverse_round_trips_the_forward_to_machine_precision() {
        let theta = [0.0, 0.7, 2.5, -1.3];
        for n in [3, 5, 6] {
            let x = balanced(n, 2.5, &theta);
            let dq0 = park_forward(&x, n, &theta).unwrap();
            let back = park_inverse(&dq0, &theta, n).unwrap();
            for (i, (&a, &b)) in back.iter().zip(&x).enumerate() {
                assert!((a - b).abs() < 1e-12, "n={n} phase sample {i}: {a} vs {b}");
            }
        }
    }

    #[test]
    fn the_zero_sequence_is_the_phase_mean() {
        let theta = [0.4];
        // a, b, c with a deliberate common offset
        let x = vec![1.0, 2.0, 6.0];
        let r = park_forward(&x, 3, &theta).unwrap();
        assert!((r.zero[0] - 3.0).abs() < TOL);
    }

    #[test]
    fn clarke_is_amplitude_invariant_on_a_balanced_set() {
        // A balanced three-phase set of peak 3 at electrical angle 0.
        let theta = [0.0, 1.1, -2.0];
        let x = balanced(3, 3.0, &theta);
        let (alpha, beta) = clarke_forward(&x, theta.len()).unwrap();
        for t in 0..theta.len() {
            assert!(
                (alpha[t].hypot(beta[t]) - 3.0).abs() < TOL,
                "|(alpha, beta)| = {}",
                alpha[t].hypot(beta[t])
            );
        }
    }

    /// The measured counterexample the Python records: a bare `T L T^T` would give 0.8 here.
    #[test]
    fn the_dq_bridge_carries_the_two_over_n_exactly_once() {
        let l: Vec<f64> = vec![
            1.0, -0.1, -0.1, //
            -0.1, 1.0, -0.1, //
            -0.1, -0.1, 1.0,
        ];
        let out = phase_l_to_dq_l(&l, 3, &[0.0]).unwrap();
        // Ld = (2/3) * M L M^T with the un-normalised M. For this balanced L that is
        // 1.0 - (-0.1) = 1.1 on both diagonal entries, and zero off-diagonal.
        assert!((out[0] - 1.1).abs() < 1e-12, "Ld = {}", out[0]);
        assert!((out[3] - 1.1).abs() < 1e-12, "Lq = {}", out[3]);
        assert!(out[1].abs() < 1e-12 && out[2].abs() < 1e-12);
        // And it is emphatically NOT the double-scaled value a bare T L T^T gives.
        assert!((out[0] - (2.0 / 3.0) * 1.1).abs() > 0.1);
    }

    #[test]
    fn the_dq_bridge_is_the_direct_injection_route() {
        // Inject a unit d current, read the dq flux back through the phase matrix, and check
        // the bridge agrees. This is the property the bridge exists to have.
        let l: Vec<f64> = vec![1.3, -0.2, -0.2, -0.2, 0.9, -0.15, -0.2, -0.15, 1.1];
        for &theta in &[0.0, 0.9, -2.2] {
            let bridge = phase_l_to_dq_l(&l, 3, &[theta]).unwrap();
            for (col, unit) in [(0, (1.0, 0.0)), (1, (0.0, 1.0))] {
                let dq0 = Dq0 {
                    d: vec![unit.0],
                    q: vec![unit.1],
                    zero: vec![0.0],
                };
                let i_phase = park_inverse(&dq0, &[theta], 3).unwrap();
                let flux: Vec<f64> = (0..3)
                    .map(|r| (0..3).map(|c| l[r * 3 + c] * i_phase[c]).sum())
                    .collect();
                let out = park_forward(&flux, 3, &[theta]).unwrap();
                assert!(
                    (out.d[0] - bridge[col]).abs() < 1e-12,
                    "theta={theta} col={col}: direct {} vs bridge {}",
                    out.d[0],
                    bridge[col]
                );
                assert!((out.q[0] - bridge[2 + col]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn a_shape_that_contradicts_the_phase_count_is_refused_rather_than_read_short() {
        assert!(matches!(
            park_forward(&[1.0, 2.0], 3, &[0.0]).unwrap_err(),
            TransformError::ShapeMismatch { .. }
        ));
        assert!(matches!(
            clarke_forward(&[1.0, 2.0], 1).unwrap_err(),
            TransformError::ShapeMismatch { .. }
        ));
        assert!(matches!(
            phase_l_to_dq_l(&[1.0; 8], 3, &[0.0]).unwrap_err(),
            TransformError::NotSquare { .. }
        ));
    }

    #[test]
    fn fewer_than_three_phases_is_refused() {
        for n in [0, 1, 2] {
            assert!(matches!(
                park_forward(&[], n, &[]).unwrap_err(),
                TransformError::TooFewPhases { .. }
            ));
        }
    }
}

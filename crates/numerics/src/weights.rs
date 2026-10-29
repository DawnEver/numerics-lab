//! `DerivativeWeights` — the Rust mirror of the Python `formulation` module.
//!
//! The semi-discrete operator a jacobian assembles is `alpha0*K + alpha1*C + alpha2*M`
//! (stiffness / damping / mass). A study supplies ONLY these three numbers; the operators
//! themselves come from the compiled equation.
//!
//! The integrators PRODUCE the triple for a time step and the physics kernels CONSUME it when
//! assembling the tangent, so the numerics leaf both depend on owns it.
//!
//! # Why the weights are complex
//!
//! The harmonic study's `alpha1 = j*omega` has to be expressible, and `omega = 0` must then
//! collapse the harmonic operator onto the stationary one — a property the Python proves
//! bit-for-bit against its production assemblers. A real-only weight type would make that
//! collapse unstateable, so the complex form is carried even though every time integrator
//! produces real values.
//!
//! `Complex64` is hand-written rather than pulled from `num-complex`: what is needed here is
//! addition, scaling and a real part.

use std::fmt;
use std::ops::{Add, Mul};

/// Newmark's average-acceleration defaults — unconditionally stable and second order.
///
/// Owned here, not re-declared per integrator. One home, one edit.
pub const NEWMARK_BETA: f64 = 0.25;
/// See [`NEWMARK_BETA`].
pub const NEWMARK_GAMMA: f64 = 0.5;

/// The minimal complex number this crate needs.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Complex64 {
    pub re: f64,
    pub im: f64,
}

impl Complex64 {
    #[must_use]
    pub const fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    /// A purely real weight — what every time integrator produces.
    #[must_use]
    pub const fn real(re: f64) -> Self {
        Self { re, im: 0.0 }
    }

    /// A purely imaginary weight — the harmonic `j*omega`.
    #[must_use]
    pub const fn imag(im: f64) -> Self {
        Self { re: 0.0, im }
    }

    #[must_use]
    pub const fn is_zero(self) -> bool {
        self.re == 0.0 && self.im == 0.0
    }

    #[must_use]
    pub fn norm(self) -> f64 {
        self.re.hypot(self.im)
    }
}

impl Add for Complex64 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self::new(self.re + rhs.re, self.im + rhs.im)
    }
}

impl Mul for Complex64 {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Self::new(
            self.re * rhs.re - self.im * rhs.im,
            self.re * rhs.im + self.im * rhs.re,
        )
    }
}

impl Mul<f64> for Complex64 {
    type Output = Self;
    fn mul(self, rhs: f64) -> Self {
        Self::new(self.re * rhs, self.im * rhs)
    }
}

impl From<f64> for Complex64 {
    fn from(re: f64) -> Self {
        Self::real(re)
    }
}

impl fmt::Display for Complex64 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.im == 0.0 {
            write!(f, "{}", self.re)
        } else {
            write!(f, "{}{:+}j", self.re, self.im)
        }
    }
}

/// Weights `(alpha0, alpha1, alpha2)` on the `u` / `u_dot` / `u_ddot` operators.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DerivativeWeights {
    pub alpha0: Complex64,
    pub alpha1: Complex64,
    pub alpha2: Complex64,
}

impl Default for DerivativeWeights {
    /// The stationary triple: stiffness only.
    fn default() -> Self {
        Self {
            alpha0: Complex64::real(1.0),
            alpha1: Complex64::new(0.0, 0.0),
            alpha2: Complex64::new(0.0, 0.0),
        }
    }
}

impl DerivativeWeights {
    /// Real-valued weights — the transient case.
    #[must_use]
    pub const fn real(alpha0: f64, alpha1: f64, alpha2: f64) -> Self {
        Self {
            alpha0: Complex64::real(alpha0),
            alpha1: Complex64::real(alpha1),
            alpha2: Complex64::real(alpha2),
        }
    }

    /// The stationary triple `(1, 0, 0)`.
    #[must_use]
    pub const fn stationary() -> Self {
        Self::real(1.0, 0.0, 0.0)
    }

    /// The harmonic triple `(1, j*omega, 0)` — the phasor `d/dt -> j*omega` mapping.
    ///
    /// At `omega = 0` this IS [`Self::stationary`], which is the collapse the Python proves
    /// against its production assemblers and the reason the type is complex at all.
    #[must_use]
    pub const fn harmonic(omega: f64) -> Self {
        Self {
            alpha0: Complex64::real(1.0),
            alpha1: Complex64::imag(omega),
            alpha2: Complex64::new(0.0, 0.0),
        }
    }

    /// True when only `alpha0` is nonzero — the equation reduces to the stiffness `K`.
    #[must_use]
    pub const fn collapses_to_stationary(&self) -> bool {
        self.alpha1.is_zero() && self.alpha2.is_zero()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_default_is_the_stationary_triple() {
        assert_eq!(
            DerivativeWeights::default(),
            DerivativeWeights::stationary()
        );
        assert!(DerivativeWeights::default().collapses_to_stationary());
    }

    #[test]
    fn a_harmonic_study_at_zero_frequency_is_exactly_the_stationary_one() {
        // The orthogonality gate the Python states: HarmonicStudy(omega=0) must collapse onto
        // the magnetostatic operator, not merely approximate it.
        assert_eq!(
            DerivativeWeights::harmonic(0.0),
            DerivativeWeights::stationary()
        );
        assert!(DerivativeWeights::harmonic(0.0).collapses_to_stationary());
    }

    #[test]
    fn a_nonzero_frequency_does_not_collapse() {
        assert!(!DerivativeWeights::harmonic(50.0).collapses_to_stationary());
        assert_eq!(
            DerivativeWeights::harmonic(50.0).alpha1,
            Complex64::new(0.0, 50.0)
        );
    }

    #[test]
    fn a_transient_triple_with_a_damping_term_does_not_collapse() {
        assert!(!DerivativeWeights::real(1.0, 100.0, 0.0).collapses_to_stationary());
    }

    #[test]
    fn complex_arithmetic_is_the_ordinary_one() {
        let a = Complex64::new(1.0, 2.0);
        let b = Complex64::new(3.0, -1.0);
        assert_eq!(a + b, Complex64::new(4.0, 1.0));
        assert_eq!(a * b, Complex64::new(5.0, 5.0));
        assert_eq!(a * 2.0, Complex64::new(2.0, 4.0));
        assert!((Complex64::new(3.0, 4.0).norm() - 5.0).abs() < 1e-15);
    }

    #[test]
    fn the_newmark_defaults_are_the_average_acceleration_pair() {
        assert_eq!((NEWMARK_BETA, NEWMARK_GAMMA), (0.25, 0.5));
    }
}

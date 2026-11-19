//! One-dimensional interpolation: the monotone PCHIP slope rule and the piecewise polynomial built on it.
//!
//! * [`pchip_slopes`]: scipy `PchipInterpolator`'s knot slopes (Fritsch-Carlson interior, Moler's edge rule).
//! * [`PiecewisePoly`]: scipy `PPoly` in the local power basis -- evaluation, derivatives, exact antiderivative.
//!
//! Both reproduce SciPy to `1e-12` (`tests/golden_interp.rs`). Moved here from motronics-studio's `core_num`
//! crate on 2026-11-19 (`bh::pchip_slopes`, `ppoly`), with no unit in the interface.

pub mod pchip;
pub mod ppoly;

pub use pchip::pchip_slopes;
pub use ppoly::PiecewisePoly;

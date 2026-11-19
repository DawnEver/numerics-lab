//! Events on the EXACT trajectory of a linear time-invariant segment `z' = M z`, read through `expm(M s)`.
//!
//! * [`guard_scan`]: armed threshold arrivals of affine guard reads, tied to an instant `tie`, with `brentq`.
//! * [`first_crossing`]: the first downward zero of affine rows, bracketed on a spectrum lattice and polished.
//!
//! Moved from motronics-studio's `laplace_net::control` on 2026-11-19 (user ruling: all numerics in the engine).

pub mod first_crossing;
pub mod guard_scan;

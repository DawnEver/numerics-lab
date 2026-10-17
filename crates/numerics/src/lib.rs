//! Physics-free numerical runtime — the Rust mirror of `python/numerics_lab/`.
//!
//! Nonlinear iteration, linear solves and time integration. It touches
//! no physics and no mesh: everything here is "given a system, drive the iterate".
//!
//! Wave 2 of `.claude/memory/2026/09/03/plan-rust-migration-execution.md`. This is where D4
//! (scipy may be replaced) is cashed, so the one thing to keep in view while reading is that
//! the Python's `direct` backend is documented as bit-for-bit `scipy.sparse.linalg.spsolve`,
//! with every golden unchanged — a promise the port inherits and cannot quietly renegotiate.

#![forbid(unsafe_code)]

pub mod integrators;
pub mod linear;
pub mod marching;
pub mod nonlinear;
pub mod sparse;
pub mod weights;

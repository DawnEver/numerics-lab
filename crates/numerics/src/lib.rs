//! Physics-free numerical runtime — the Rust mirror of `src/motronics/gauss/`.
//!
//! Nonlinear iteration, linear solves, time integration and the phase transforms. It touches
//! no physics and no mesh: everything here is "given a system, drive the iterate".
//!
//! Wave 2 of `git show d212b178f^:.claude/memory/2026/09/03/plan-rust-migration-execution.md`. This is where D4
//! (scipy may be replaced) is cashed, so the one thing to keep in view while reading is that
//! the Python's `direct` backend is documented as bit-for-bit `scipy.sparse.linalg.spsolve`,
//! with every golden unchanged — a promise the port inherits and cannot quietly renegotiate.

#![forbid(unsafe_code)]

pub mod integrators;
pub mod linear;
pub mod marching;
pub mod nonlinear;
pub mod transforms;

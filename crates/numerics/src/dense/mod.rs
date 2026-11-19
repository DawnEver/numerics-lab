//! Small DENSE kernels transcribed bit for bit from the BLAS/LAPACK that SciPy and numpy link (OpenBLAS,
//! SKYLAKEX), so a Rust caller reproduces a Python march to the last bit.
//!
//! * [`expm`]: `scipy.linalg.expm` (Al-Mohy & Higham 2009), bit-identical up to [`expm::MAX_N`], round-off past it.
//! * [`lapack`]: `np.linalg.solve` (`dgesv`) and `np.linalg.cond` (`dgesdd`, `n <= COND_MAX_N`).
//!
//! Moved here from motronics-studio's `core_num` crate on 2026-11-19 (the family's one numerics engine).

pub mod expm;
pub mod lapack;
pub mod matrix;

pub use expm::{expm, ExpmError, MAX_N as EXPM_BIT_IDENTICAL_MAX_N};
pub use lapack::{cond, solve, CondError, COND_MAX_N};

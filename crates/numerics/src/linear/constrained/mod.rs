//! Constrained linear systems — the Rust mirror of `python/numerics_lab/linear/constrained/`.
//!
//! [`fixed_values`] prescribes values on an assembly's COO stream, [`pin_rows`] replaces rows of a
//! built matrix by identity rows, and [`master_slave`] resolves master-slave constraint chains.
//! The Python's `null_space_pinning` has no Rust caller yet, so no port.

pub mod fixed_values;
pub mod master_slave;
pub mod pin_rows;

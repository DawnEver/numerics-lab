//! `motronics_native.gauss` — the physics-free numerical runtime's Python surface.
//!
//! Thin glue only: NumPy arrays in, NumPy arrays out, calling [`gauss`]. No numerics live
//! here. The sparse operator arrives as COO triplets because that is the shape
//! `scipy.sparse.coo_matrix` hands out on the Python side, and `contracts` owns the
//! COO -> CSR conversion.

use contracts::sparse::Coo;
use gauss::linear::{requires_symmetric, solve_linear_system};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// `solve_linear_system(n_rows, n_cols, rows, cols, values, rhs, solver=None, preconditioner=None, x0=None)`
///
/// Solve `A @ x = rhs` where `A` is given as COO triplets. `solver` is `None` (the default
/// `direct` sparse LU) or a Krylov name; `preconditioner` is `jacobi` / `ilu` / `ssor`;
/// `x0` is an optional Krylov warm start. Returns `x` as `f64[n_rows]`.
///
/// Errors from the guard (a non-finite solution, a direct-solve residual that does not
/// verify, an unknown backend name) come back as `ValueError` — never a wrong `x` returned
/// as a success.
#[pyfunction]
#[pyo3(
    name = "solve_linear_system",
    signature = (n_rows, n_cols, rows, cols, values, rhs, solver=None, preconditioner=None, x0=None)
)]
#[allow(clippy::too_many_arguments)]
pub fn py_solve_linear_system<'py>(
    py: Python<'py>,
    n_rows: usize,
    n_cols: usize,
    rows: PyReadonlyArray1<'py, i64>,
    cols: PyReadonlyArray1<'py, i64>,
    values: PyReadonlyArray1<'py, f64>,
    rhs: PyReadonlyArray1<'py, f64>,
    solver: Option<&str>,
    preconditioner: Option<&str>,
    x0: Option<PyReadonlyArray1<'py, f64>>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let r: Vec<u32> = rows.as_array().iter().map(|&i| i as u32).collect();
    let c: Vec<u32> = cols.as_array().iter().map(|&i| i as u32).collect();
    let v: Vec<f64> = values.as_array().to_vec();
    let coo = Coo::from_triplets(n_rows, n_cols, r, c, v)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let csr = coo
        .to_csr()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let b = rhs.as_array().to_vec();
    let warm = x0.map(|a| a.as_array().to_vec());
    let x = solve_linear_system(&csr, &b, solver, preconditioner, warm.as_deref())
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(x.into_pyarray(py))
}

/// `requires_symmetric(name)` — does the named backend assume a symmetric/SPD operator?
///
/// An unknown name is `False` rather than an error: this is a QUERY about a tag, and
/// `solve_linear_system` is where an unknown name is refused.
#[pyfunction]
#[pyo3(name = "requires_symmetric")]
pub fn py_requires_symmetric(name: &str) -> bool {
    requires_symmetric(name)
}

/// Build and populate the `gauss` submodule.
///
/// # Errors
/// Propagates any PyO3 registration failure.
pub fn register<'py>(parent: &Bound<'py, PyModule>) -> PyResult<Bound<'py, PyModule>> {
    let m = PyModule::new(parent.py(), "gauss")?;
    m.add_function(wrap_pyfunction!(py_solve_linear_system, &m)?)?;
    m.add_function(wrap_pyfunction!(py_requires_symmetric, &m)?)?;
    m.add(
        "__doc__",
        "Physics-free numerical runtime: sparse linear solves (direct LU + Krylov).",
    )?;
    parent.add_submodule(&m)?;
    Ok(m)
}

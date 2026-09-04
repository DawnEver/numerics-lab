//! `motronics_native.gauss` — the physics-free numerical runtime's Python surface.
//!
//! Thin glue only: NumPy arrays in, NumPy arrays out, calling [`gauss`]. No numerics live
//! here. The sparse operator arrives as COO triplets because that is the shape
//! `scipy.sparse.coo_matrix` hands out on the Python side, and `contracts` owns the
//! COO -> CSR conversion.

use contracts::sparse::{Coo, Csr};
use gauss::linear::direct::DirectFactorization;
use gauss::linear::{check_direct_residual, requires_symmetric, solve_linear_system};
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

/// `DirectFactorization(n_rows, n_cols, indptr, indices, values)` — the CACHED-symbolic direct LU.
///
/// [`py_solve_linear_system`] pays colamd and the symbolic factorisation on every call, which is
/// 34% / 46% of a faer solve at the two production sizes and is IDENTICAL between Newton
/// iterations. This class is that half held: the constructor analyses the pattern once and
/// [`PyDirectFactorization::solve`] pays only the numeric factorisation and the triangular solve.
///
/// It exists because the comparison that decides whether faer replaces SuperLU is only meaningful
/// in this shape — a wiring would hold the factorisation across the loop, so a benchmark that
/// re-analysed per call would be measuring an implementation nobody proposes.
///
/// The pattern is CHECKED, never assumed: the constructor refuses a non-canonical CSR (columns not
/// ascending / not unique, which `Coo::to_csr` would silently reorder into a different value
/// order), and `gauss`'s own `DirectFactorization::solve` refuses a changed pattern.
#[pyclass(name = "DirectFactorization")]
pub struct PyDirectFactorization {
    /// The analysed operator; only its VALUES move between solves.
    csr: Csr,
    factorization: DirectFactorization,
}

#[pymethods]
impl PyDirectFactorization {
    #[new]
    #[pyo3(signature = (n_rows, n_cols, indptr, indices, values))]
    fn new<'py>(
        n_rows: usize,
        n_cols: usize,
        indptr: PyReadonlyArray1<'py, i64>,
        indices: PyReadonlyArray1<'py, i64>,
        values: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Self> {
        let ptr: Vec<usize> = indptr.as_array().iter().map(|&i| i as usize).collect();
        let idx: Vec<u32> = indices.as_array().iter().map(|&i| i as u32).collect();
        let val: Vec<f64> = values.as_array().to_vec();
        if ptr.len() != n_rows + 1 || ptr.last().copied().unwrap_or(0) != idx.len() {
            return Err(PyValueError::new_err(
                "indptr must be n_rows+1 long and end at nnz",
            ));
        }
        let mut rows: Vec<u32> = Vec::with_capacity(idx.len());
        for r in 0..n_rows {
            for _ in ptr[r]..ptr[r + 1] {
                rows.push(r as u32);
            }
        }
        let coo = Coo::from_triplets(n_rows, n_cols, rows, idx.clone(), val)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let csr = coo
            .to_csr()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        if csr.indices() != idx.as_slice() {
            return Err(PyValueError::new_err(
                "the CSR offered is not canonical (columns must be ascending and unique); \
                 call sort_indices()/sum_duplicates() before handing it over, because a \
                 reordering here would silently misplace every later value update",
            ));
        }
        let factorization =
            DirectFactorization::new(&csr).map_err(|e| PyValueError::new_err(format!("{e:?}")))?;
        Ok(Self { csr, factorization })
    }

    /// `solve(values, rhs)` — a new value vector on the SAME pattern, solved against the cache.
    ///
    /// The residual guard is applied here exactly as `solve_linear_system` applies it to the
    /// one-shot path: a finite-but-meaningless solution is refused rather than returned.
    fn solve<'py>(
        &mut self,
        py: Python<'py>,
        values: PyReadonlyArray1<'py, f64>,
        rhs: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let new_values = values.as_array();
        let slot = self.csr.values_mut();
        if new_values.len() != slot.len() {
            return Err(PyValueError::new_err(
                "values must carry the analysed pattern's nnz",
            ));
        }
        for (dst, &src) in slot.iter_mut().zip(new_values.iter()) {
            *dst = src;
        }
        let b = rhs.as_array().to_vec();
        let x = self
            .factorization
            .solve(&self.csr, &b)
            .map_err(|e| PyValueError::new_err(format!("{e:?}")))?;
        check_direct_residual(&self.csr, &b, &x).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(x.into_pyarray(py))
    }

    /// The order of the analysed system.
    #[getter]
    fn n(&self) -> usize {
        self.factorization.n()
    }
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
    m.add_class::<PyDirectFactorization>()?;
    m.add(
        "__doc__",
        "Physics-free numerical runtime: sparse linear solves (direct LU + Krylov).",
    )?;
    parent.add_submodule(&m)?;
    Ok(m)
}

//! `numerics_lab._native.AmgHierarchy` and the AMG graph queries -- glue over `numerics::linear::amg`.

use numerics::linear::amg::{
    aggregate_labels_of, strong_connections_of, Amg, Coarsening, Smoothing,
};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::canonical_csr;

type Ints<'py> = Bound<'py, PyArray1<i64>>;

fn value_error(e: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(e.to_string())
}

#[allow(clippy::cast_possible_wrap)]
fn to_i64(values: &[usize]) -> Vec<i64> {
    values.iter().map(|&v| v as i64).collect()
}

/// `AmgHierarchy(n, indptr, indices, values, coarsening, theta, smoothing)` -- a built V-cycle.
#[pyclass(name = "AmgHierarchy")]
pub struct PyAmgHierarchy(Amg);

#[pymethods]
impl PyAmgHierarchy {
    #[new]
    #[pyo3(signature = (n, indptr, indices, values, coarsening, theta, smoothing))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        n: usize,
        indptr: PyReadonlyArray1<'_, i64>,
        indices: PyReadonlyArray1<'_, i64>,
        values: PyReadonlyArray1<'_, f64>,
        coarsening: &str,
        theta: f64,
        smoothing: &str,
    ) -> PyResult<Self> {
        let csr = canonical_csr(n, n, &indptr, &indices, &values)?;
        let coarsening = Coarsening::parse(coarsening).map_err(value_error)?;
        let smoothing = Smoothing::parse(smoothing).map_err(value_error)?;
        Amg::new(&csr, coarsening, theta, smoothing)
            .map(Self)
            .map_err(value_error)
    }

    /// `apply(rhs)` -- one V-cycle, `M^-1 rhs`.
    fn apply<'py>(
        &self,
        py: Python<'py>,
        rhs: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let rhs = rhs.as_array().to_vec();
        let out = py.detach(|| self.0.apply(&rhs)).map_err(value_error)?;
        Ok(out.into_pyarray(py))
    }

    #[getter]
    fn level_sizes(&self) -> Vec<usize> {
        self.0.level_sizes().to_vec()
    }

    #[getter]
    fn operator_complexity(&self) -> f64 {
        self.0.operator_complexity()
    }
}

/// `amg_strong_connections(n, indptr, indices, values, theta) -> (indptr, indices, values)`.
#[pyfunction]
#[pyo3(name = "amg_strong_connections")]
fn py_amg_strong_connections<'py>(
    py: Python<'py>,
    n: usize,
    indptr: PyReadonlyArray1<'py, i64>,
    indices: PyReadonlyArray1<'py, i64>,
    values: PyReadonlyArray1<'py, f64>,
    theta: f64,
) -> PyResult<(Ints<'py>, Ints<'py>, Bound<'py, PyArray1<f64>>)> {
    let strong = strong_connections_of(&canonical_csr(n, n, &indptr, &indices, &values)?, theta);
    Ok((
        to_i64(&strong.ptr).into_pyarray(py),
        to_i64(&strong.idx).into_pyarray(py),
        strong.val.into_pyarray(py),
    ))
}

/// `amg_aggregates(n, indptr, indices, values, coarsening, theta) -> (labels, count)`.
#[pyfunction]
#[pyo3(name = "amg_aggregates")]
fn py_amg_aggregates<'py>(
    py: Python<'py>,
    n: usize,
    indptr: PyReadonlyArray1<'py, i64>,
    indices: PyReadonlyArray1<'py, i64>,
    values: PyReadonlyArray1<'py, f64>,
    coarsening: &str,
    theta: f64,
) -> PyResult<(Ints<'py>, usize)> {
    let csr = canonical_csr(n, n, &indptr, &indices, &values)?;
    let (labels, count) = aggregate_labels_of(
        &csr,
        Coarsening::parse(coarsening).map_err(value_error)?,
        theta,
    );
    Ok((to_i64(&labels).into_pyarray(py), count))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyAmgHierarchy>()?;
    m.add_function(wrap_pyfunction!(py_amg_strong_connections, m)?)?;
    m.add_function(wrap_pyfunction!(py_amg_aggregates, m)?)?;
    Ok(())
}

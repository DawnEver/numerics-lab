//! `numerics::structural` on `_native`: index arrays in (`int64`), index arrays out (`int64`, `-1` for "none").

use numerics::structural::{self as st, Bipartite, Digraph, StructureError};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

type Ix<'py> = Bound<'py, PyArray1<i64>>;

fn refuse(e: &StructureError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// `i64` indices to `usize`, a negative one refused with the caller's number (a cast would wrap it).
fn indices(a: &PyReadonlyArray1<'_, i64>, axis: &str) -> PyResult<Vec<usize>> {
    a.as_array()
        .iter()
        .map(|&v| {
            usize::try_from(v)
                .map_err(|_| PyValueError::new_err(format!("{axis} index {v} is negative")))
        })
        .collect()
}

fn to_py(v: impl IntoIterator<Item = usize>) -> Vec<i64> {
    v.into_iter()
        .map(|x| i64::try_from(x).expect("an index fits i64"))
        .collect()
}

fn opt_to_py(v: &[Option<usize>]) -> Vec<i64> {
    v.iter()
        .map(|x| x.map_or(-1, |x| i64::try_from(x).expect("an index fits i64")))
        .collect()
}

fn bipartite(
    n_rows: usize,
    n_cols: usize,
    rows: &PyReadonlyArray1<'_, i64>,
    cols: &PyReadonlyArray1<'_, i64>,
) -> PyResult<Bipartite> {
    Bipartite::from_incidence(
        n_rows,
        n_cols,
        &indices(rows, "row")?,
        &indices(cols, "column")?,
    )
    .map_err(|e| refuse(&e))
}

fn digraph(
    n: usize,
    sources: &PyReadonlyArray1<'_, i64>,
    targets: &PyReadonlyArray1<'_, i64>,
) -> PyResult<Digraph> {
    Digraph::from_edges(
        n,
        &indices(sources, "source")?,
        &indices(targets, "target")?,
    )
    .map_err(|e| refuse(&e))
}

/// `maximum_matching(n_rows, n_cols, rows, cols)` -> the column matched to each row, `-1` when unmatched
/// (the shape of `scipy.sparse.csgraph.maximum_bipartite_matching(..., perm_type="column")`).
#[pyfunction]
#[pyo3(name = "maximum_matching")]
pub fn py_maximum_matching<'py>(
    py: Python<'py>,
    n_rows: usize,
    n_cols: usize,
    rows: PyReadonlyArray1<'py, i64>,
    cols: PyReadonlyArray1<'py, i64>,
) -> PyResult<Ix<'py>> {
    let g = bipartite(n_rows, n_cols, &rows, &cols)?;
    let m = py.detach(|| st::maximum_matching(&g));
    Ok(opt_to_py(&m.col_of_row).into_pyarray(py))
}

/// `strongly_connected_components(n, sources, targets)` -> `(count, labels)`, labels dependency-first:
/// a component reachable from component `k` has a label `<= k`.
#[pyfunction]
#[pyo3(name = "strongly_connected_components")]
pub fn py_strongly_connected_components<'py>(
    py: Python<'py>,
    n: usize,
    sources: PyReadonlyArray1<'py, i64>,
    targets: PyReadonlyArray1<'py, i64>,
) -> PyResult<(usize, Ix<'py>)> {
    let g = digraph(n, &sources, &targets)?;
    let c = py.detach(|| st::strongly_connected_components(&g));
    Ok((c.count, to_py(c.label).into_pyarray(py)))
}

/// `dulmage_mendelsohn(n_rows, n_cols, rows, cols)` -> `(row_order, col_order, row_bounds, col_bounds)`:
/// block 0 the over-determined part, the last block the under-determined part (either may be empty), the
/// square part's fine blocks between; block lower triangular.
#[pyfunction]
#[pyo3(name = "dulmage_mendelsohn")]
pub fn py_dulmage_mendelsohn<'py>(
    py: Python<'py>,
    n_rows: usize,
    n_cols: usize,
    rows: PyReadonlyArray1<'py, i64>,
    cols: PyReadonlyArray1<'py, i64>,
) -> PyResult<(Ix<'py>, Ix<'py>, Ix<'py>, Ix<'py>)> {
    let g = bipartite(n_rows, n_cols, &rows, &cols)?;
    let d = py.detach(|| st::dulmage_mendelsohn(&g));
    Ok((
        to_py(d.row_order).into_pyarray(py),
        to_py(d.col_order).into_pyarray(py),
        to_py(d.row_bounds).into_pyarray(py),
        to_py(d.col_bounds).into_pyarray(py),
    ))
}

/// `block_triangular(n, rows, cols)` -> `(row_order, col_order, bounds)` of a structurally nonsingular
/// square pattern; `ValueError` naming the structural rank otherwise.
#[pyfunction]
#[pyo3(name = "block_triangular")]
pub fn py_block_triangular<'py>(
    py: Python<'py>,
    n: usize,
    rows: PyReadonlyArray1<'py, i64>,
    cols: PyReadonlyArray1<'py, i64>,
) -> PyResult<(Ix<'py>, Ix<'py>, Ix<'py>)> {
    let g = bipartite(n, n, &rows, &cols)?;
    let b = py
        .detach(|| st::block_triangular(&g))
        .map_err(|e| refuse(&e))?;
    Ok((
        to_py(b.row_order).into_pyarray(py),
        to_py(b.col_order).into_pyarray(py),
        to_py(b.bounds).into_pyarray(py),
    ))
}

/// `tear(n, sources, targets)` -> `(tears, order)` for a digraph whose edge `u -> v` reads "`u` needs `v`".
#[pyfunction]
#[pyo3(name = "tear")]
pub fn py_tear<'py>(
    py: Python<'py>,
    n: usize,
    sources: PyReadonlyArray1<'py, i64>,
    targets: PyReadonlyArray1<'py, i64>,
) -> PyResult<(Ix<'py>, Ix<'py>)> {
    let g = digraph(n, &sources, &targets)?;
    let t = py.detach(|| st::tear(&g));
    Ok((
        to_py(t.tears).into_pyarray(py),
        to_py(t.order).into_pyarray(py),
    ))
}

/// `pantelides(n_eq, n_var, rows, cols, derivative_of)` ->
/// `(eq_rows, eq_cols, equation_derivative, variable_derivative, assignment)`: the extended incidence as
/// triplets, and three maps with `-1` for "none". `derivative_of[j]` is the variable that is `d/dt` of `j`, or `-1`.
#[pyfunction]
#[pyo3(name = "pantelides")]
#[allow(clippy::type_complexity)]
pub fn py_pantelides<'py>(
    py: Python<'py>,
    n_eq: usize,
    n_var: usize,
    rows: PyReadonlyArray1<'py, i64>,
    cols: PyReadonlyArray1<'py, i64>,
    derivative_of: PyReadonlyArray1<'py, i64>,
) -> PyResult<(Ix<'py>, Ix<'py>, Ix<'py>, Ix<'py>, Ix<'py>)> {
    let g = bipartite(n_eq, n_var, &rows, &cols)?;
    let der: Vec<Option<usize>> = derivative_of
        .as_array()
        .iter()
        .map(|&v| match v {
            -1 => Ok(None),
            v => usize::try_from(v).map(Some).map_err(|_| {
                PyValueError::new_err(format!(
                    "derivative_of entry {v} is neither -1 nor an index"
                ))
            }),
        })
        .collect::<PyResult<_>>()?;
    let r = py
        .detach(|| st::pantelides(&g, &der))
        .map_err(|e| refuse(&e))?;
    let (mut er, mut ec) = (Vec::new(), Vec::new());
    for (i, vars) in r.equations.iter().enumerate() {
        er.extend(std::iter::repeat_n(i, vars.len()));
        ec.extend(vars.iter().copied());
    }
    Ok((
        to_py(er).into_pyarray(py),
        to_py(ec).into_pyarray(py),
        opt_to_py(&r.equation_derivative).into_pyarray(py),
        opt_to_py(&r.variable_derivative).into_pyarray(py),
        opt_to_py(&r.assignment).into_pyarray(py),
    ))
}

/// Register this file's functions on `_native`.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_maximum_matching, m)?)?;
    m.add_function(wrap_pyfunction!(py_strongly_connected_components, m)?)?;
    m.add_function(wrap_pyfunction!(py_dulmage_mendelsohn, m)?)?;
    m.add_function(wrap_pyfunction!(py_block_triangular, m)?)?;
    m.add_function(wrap_pyfunction!(py_tear, m)?)?;
    m.add_function(wrap_pyfunction!(py_pantelides, m)?)?;
    Ok(())
}

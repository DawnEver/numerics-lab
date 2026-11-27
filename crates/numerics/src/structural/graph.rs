//! The two graph shapes: a [`Bipartite`] incidence pattern and a [`Digraph`], both compressed adjacency.

use std::fmt;

/// Why a structural request was refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StructureError {
    /// Two parallel index arrays of different lengths.
    LengthMismatch { left: usize, right: usize },
    /// An index outside its axis.
    IndexOutOfRange {
        axis: &'static str,
        index: usize,
        bound: usize,
    },
    /// A square pattern was required.
    NotSquare { rows: usize, cols: usize },
    /// The pattern has no perfect matching: `rank` of `n` rows can be matched.
    StructurallySingular { rank: usize, n: usize },
    /// The derivative map is not a set of disjoint chains (a variable is the derivative of two, or of itself).
    DerivativeNotAChain { variable: usize },
    /// Pantelides differentiated `equation` more than `bound` times: the DAE pattern is structurally singular.
    IndexBoundExceeded { equation: usize, bound: usize },
}

impl fmt::Display for StructureError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LengthMismatch { left, right } => write!(
                f,
                "index arrays differ in length ({left} vs {right}); one entry is one (row, col) pair"
            ),
            Self::IndexOutOfRange { axis, index, bound } => {
                write!(f, "{axis} index {index} is outside [0, {bound})")
            }
            Self::NotSquare { rows, cols } => write!(
                f,
                "a {rows} x {cols} pattern is not square; dulmage_mendelsohn decomposes a rectangular one"
            ),
            Self::StructurallySingular { rank, n } => write!(
                f,
                "structurally singular: a maximum matching covers {rank} of {n} rows; \
                 dulmage_mendelsohn names the over- and under-determined parts"
            ),
            Self::DerivativeNotAChain { variable } => write!(
                f,
                "variable {variable} is the derivative of two variables or of itself; \
                 derivative_of must map each variable to at most one other, without cycles"
            ),
            Self::IndexBoundExceeded { equation, bound } => write!(
                f,
                "equation {equation} was differentiated more than {bound} times: the DAE pattern is \
                 structurally singular (no differentiation makes it matchable)"
            ),
        }
    }
}

impl std::error::Error for StructureError {}

fn check(axis: &'static str, index: usize, bound: usize) -> Result<(), StructureError> {
    if index < bound {
        Ok(())
    } else {
        Err(StructureError::IndexOutOfRange { axis, index, bound })
    }
}

/// Compressed adjacency `ptr`/`adj` from `(source, target)` pairs: sorted, duplicates dropped.
fn compress(n: usize, sources: &[usize], targets: &[usize]) -> (Vec<usize>, Vec<usize>) {
    let mut lists = vec![Vec::new(); n];
    for (&s, &t) in sources.iter().zip(targets) {
        lists[s].push(t);
    }
    let mut ptr = Vec::with_capacity(n + 1);
    let mut adj = Vec::with_capacity(sources.len());
    ptr.push(0);
    for mut list in lists {
        list.sort_unstable();
        list.dedup();
        adj.extend(list);
        ptr.push(adj.len());
    }
    (ptr, adj)
}

/// A bipartite incidence pattern: `n_rows` x `n_cols`, row `r` adjacent to the columns of [`Bipartite::row`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Bipartite {
    n_rows: usize,
    n_cols: usize,
    ptr: Vec<usize>,
    adj: Vec<usize>,
}

impl Bipartite {
    /// The pattern holding an entry at every `(rows[k], cols[k])`; repeated entries count once.
    ///
    /// # Errors
    /// [`StructureError::LengthMismatch`] or [`StructureError::IndexOutOfRange`].
    pub fn from_incidence(
        n_rows: usize,
        n_cols: usize,
        rows: &[usize],
        cols: &[usize],
    ) -> Result<Self, StructureError> {
        if rows.len() != cols.len() {
            return Err(StructureError::LengthMismatch {
                left: rows.len(),
                right: cols.len(),
            });
        }
        for (&r, &c) in rows.iter().zip(cols) {
            check("row", r, n_rows)?;
            check("column", c, n_cols)?;
        }
        let (ptr, adj) = compress(n_rows, rows, cols);
        Ok(Self {
            n_rows,
            n_cols,
            ptr,
            adj,
        })
    }

    /// Number of rows.
    #[must_use]
    pub fn n_rows(&self) -> usize {
        self.n_rows
    }

    /// Number of columns.
    #[must_use]
    pub fn n_cols(&self) -> usize {
        self.n_cols
    }

    /// The columns row `r` touches, ascending.
    #[must_use]
    pub fn row(&self, r: usize) -> &[usize] {
        &self.adj[self.ptr[r]..self.ptr[r + 1]]
    }

    /// The same pattern seen from the columns: row `c` of the result lists the rows column `c` touches.
    #[must_use]
    pub fn transpose(&self) -> Self {
        let mut rows = Vec::with_capacity(self.adj.len());
        for r in 0..self.n_rows {
            rows.extend(std::iter::repeat_n(r, self.row(r).len()));
        }
        let (ptr, adj) = compress(self.n_cols, &self.adj, &rows);
        Self {
            n_rows: self.n_cols,
            n_cols: self.n_rows,
            ptr,
            adj,
        }
    }
}

/// A directed graph on `n` nodes; [`Digraph::successors`] of `v` are the targets of its edges.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Digraph {
    ptr: Vec<usize>,
    adj: Vec<usize>,
}

impl Digraph {
    /// The digraph with an edge `sources[k] -> targets[k]` for every `k`; repeated edges count once.
    ///
    /// # Errors
    /// [`StructureError::LengthMismatch`] or [`StructureError::IndexOutOfRange`].
    pub fn from_edges(
        n: usize,
        sources: &[usize],
        targets: &[usize],
    ) -> Result<Self, StructureError> {
        if sources.len() != targets.len() {
            return Err(StructureError::LengthMismatch {
                left: sources.len(),
                right: targets.len(),
            });
        }
        for (&s, &t) in sources.iter().zip(targets) {
            check("source", s, n)?;
            check("target", t, n)?;
        }
        let (ptr, adj) = compress(n, sources, targets);
        Ok(Self { ptr, adj })
    }

    /// Number of nodes.
    #[must_use]
    pub fn n(&self) -> usize {
        self.ptr.len() - 1
    }

    /// The targets of `v`'s edges, ascending.
    #[must_use]
    pub fn successors(&self, v: usize) -> &[usize] {
        &self.adj[self.ptr[v]..self.ptr[v + 1]]
    }

    /// Every edge reversed.
    #[must_use]
    pub fn reversed(&self) -> Self {
        let n = self.n();
        let mut sources = Vec::with_capacity(self.adj.len());
        for v in 0..n {
            sources.extend(std::iter::repeat_n(v, self.successors(v).len()));
        }
        let (ptr, adj) = compress(n, &self.adj, &sources);
        Self { ptr, adj }
    }
}

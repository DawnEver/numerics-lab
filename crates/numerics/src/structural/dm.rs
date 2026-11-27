//! Dulmage-Mendelsohn decomposition and the block lower triangular form.
//!
//! Layout (LOWER block triangular throughout): the over-determined part first, the square part's
//! fine blocks next in dependency order, the under-determined part last. A row then touches only
//! columns of its own block or of blocks before it:
//!
//! ```text
//!            C_over  C_square  C_under
//! R_over   [  V        0         0   ]
//! R_square [  *        S         0   ]   S itself block lower triangular
//! R_under  [  *        *         H   ]
//! ```

use super::graph::{Bipartite, Digraph, StructureError};
use super::matching::{maximum_matching, Matching};
use super::scc::strongly_connected_components;

/// The Dulmage-Mendelsohn decomposition of a rectangular pattern.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Decomposition {
    /// Rows in block order.
    pub row_order: Vec<usize>,
    /// Columns in block order.
    pub col_order: Vec<usize>,
    /// Block `k` is rows `row_order[row_bounds[k]..row_bounds[k + 1]]`. Block 0 is the over-determined
    /// part (more rows than columns), the last block the under-determined part (more columns than rows),
    /// either possibly empty; the blocks between are the square part's fine blocks.
    pub row_bounds: Vec<usize>,
    /// Block `k`'s columns, as [`Self::row_bounds`].
    pub col_bounds: Vec<usize>,
    /// The maximum matching the decomposition was read from.
    pub matching: Matching,
}

impl Decomposition {
    /// The structural rank: the maximum matching's size.
    #[must_use]
    pub fn structural_rank(&self) -> usize {
        self.matching.size()
    }
}

/// The block lower triangular form of a structurally nonsingular square pattern.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlockTriangular {
    /// Rows in block order.
    pub row_order: Vec<usize>,
    /// Columns in block order; `col_order[k]` is the column matched to `row_order[k]`.
    pub col_order: Vec<usize>,
    /// Block `k` is positions `bounds[k]..bounds[k + 1]` of both orders.
    pub bounds: Vec<usize>,
}

/// Mark what alternating paths reach from the free vertices of one side: `from` free side's adjacency,
/// `matched` the other side's partner. Returns (marks on the start side, marks on the other side).
fn alternating_reach(
    adj: &Bipartite,
    start_free: &[Option<usize>],
    partner_of_other: &[Option<usize>],
) -> (Vec<bool>, Vec<bool>) {
    let mut start = vec![false; adj.n_rows()];
    let mut other = vec![false; adj.n_cols()];
    let mut queue: Vec<usize> = (0..adj.n_rows())
        .filter(|&v| start_free[v].is_none())
        .collect();
    for &v in &queue {
        start[v] = true;
    }
    while let Some(v) = queue.pop() {
        for &w in adj.row(v) {
            if !other[w] {
                other[w] = true;
                if let Some(p) = partner_of_other[w] {
                    if !start[p] {
                        start[p] = true;
                        queue.push(p);
                    }
                }
            }
        }
    }
    (start, other)
}

/// The square, perfectly matched rows `rows` split into strongly connected blocks, dependency-first:
/// row `r` depends on row `r'` when it touches the column matched to `r'`.
fn fine_blocks(
    g: &Bipartite,
    rows: &[usize],
    matching: &Matching,
    in_square_col: &[bool],
) -> (Vec<usize>, Vec<usize>, Vec<usize>) {
    let mut local = vec![usize::MAX; g.n_rows()];
    for (k, &r) in rows.iter().enumerate() {
        local[r] = k;
    }
    let (mut sources, mut targets) = (Vec::new(), Vec::new());
    for (k, &r) in rows.iter().enumerate() {
        for &c in g.row(r) {
            if in_square_col[c] {
                if let Some(r2) = matching.row_of_col[c] {
                    if r2 != r {
                        sources.push(k);
                        targets.push(local[r2]);
                    }
                }
            }
        }
    }
    let dg = Digraph::from_edges(rows.len(), &sources, &targets)
        .expect("edges are built from in-range local indices");
    let comps = strongly_connected_components(&dg);
    let mut by_block: Vec<Vec<usize>> = vec![Vec::new(); comps.count];
    for (k, &r) in rows.iter().enumerate() {
        by_block[comps.label[k]].push(r);
    }
    let mut row_order = Vec::with_capacity(rows.len());
    let mut bounds = vec![0];
    for block in by_block {
        row_order.extend(block);
        bounds.push(row_order.len());
    }
    let col_order = row_order
        .iter()
        .map(|&r| matching.col_of_row[r].expect("a square-part row is matched"))
        .collect();
    (row_order, col_order, bounds)
}

/// The Dulmage-Mendelsohn decomposition of `g` (Dulmage & Mendelsohn 1958; Pothen & Fan 1990):
/// the over-determined part is what alternating paths reach from the unmatched rows, the
/// under-determined part what they reach from the unmatched columns, and the square remainder
/// splits into its strongly connected blocks.
#[must_use]
pub fn dulmage_mendelsohn(g: &Bipartite) -> Decomposition {
    let matching = maximum_matching(g);
    let gt = g.transpose();
    let (over_rows, over_cols) = alternating_reach(g, &matching.col_of_row, &matching.row_of_col);
    let (under_cols, under_rows) =
        alternating_reach(&gt, &matching.row_of_col, &matching.col_of_row);
    let square_rows: Vec<usize> = (0..g.n_rows())
        .filter(|&r| !over_rows[r] && !under_rows[r])
        .collect();
    let in_square_col: Vec<bool> = (0..g.n_cols())
        .map(|c| !over_cols[c] && !under_cols[c])
        .collect();
    let (sq_rows, sq_cols, sq_bounds) = fine_blocks(g, &square_rows, &matching, &in_square_col);

    let mut row_order: Vec<usize> = (0..g.n_rows()).filter(|&r| over_rows[r]).collect();
    let mut col_order: Vec<usize> = (0..g.n_cols()).filter(|&c| over_cols[c]).collect();
    let mut row_bounds = vec![0, row_order.len()];
    let mut col_bounds = vec![0, col_order.len()];
    for &b in &sq_bounds[1..] {
        row_bounds.push(row_order.len() + b);
        col_bounds.push(col_order.len() + b);
    }
    row_order.extend(sq_rows);
    col_order.extend(sq_cols);
    row_order.extend((0..g.n_rows()).filter(|&r| under_rows[r]));
    col_order.extend((0..g.n_cols()).filter(|&c| under_cols[c]));
    row_bounds.push(row_order.len());
    col_bounds.push(col_order.len());
    Decomposition {
        row_order,
        col_order,
        row_bounds,
        col_bounds,
        matching,
    }
}

/// The block lower triangular form of a square pattern: a perfect matching assigns each row its column,
/// and the strongly connected components of the row dependency graph, dependency-first, are the blocks.
///
/// # Errors
/// [`StructureError::NotSquare`] or, without a perfect matching, [`StructureError::StructurallySingular`].
pub fn block_triangular(g: &Bipartite) -> Result<BlockTriangular, StructureError> {
    let n = g.n_rows();
    if g.n_cols() != n {
        return Err(StructureError::NotSquare {
            rows: n,
            cols: g.n_cols(),
        });
    }
    let matching = maximum_matching(g);
    let rank = matching.size();
    if rank != n {
        return Err(StructureError::StructurallySingular { rank, n });
    }
    let rows: Vec<usize> = (0..n).collect();
    let (row_order, col_order, bounds) = fine_blocks(g, &rows, &matching, &vec![true; n]);
    Ok(BlockTriangular {
        row_order,
        col_order,
        bounds,
    })
}

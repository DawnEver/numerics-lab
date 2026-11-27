//! Hopcroft-Karp maximum bipartite matching.

use super::graph::Bipartite;

/// A matching: each row to at most one column and back.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Matching {
    /// The column row `r` is matched to.
    pub col_of_row: Vec<Option<usize>>,
    /// The row column `c` is matched to.
    pub row_of_col: Vec<Option<usize>>,
}

impl Matching {
    /// Number of matched pairs -- the structural rank when the matching is maximum.
    #[must_use]
    pub fn size(&self) -> usize {
        self.col_of_row.iter().flatten().count()
    }
}

/// A maximum-cardinality matching of `g` (Hopcroft-Karp 1973: BFS layers from the free rows, then
/// vertex-disjoint shortest augmenting paths by an iterative DFS, until no augmenting path remains).
#[must_use]
pub fn maximum_matching(g: &Bipartite) -> Matching {
    const UNSEEN: usize = usize::MAX;
    let (nr, nc) = (g.n_rows(), g.n_cols());
    let mut col_of_row: Vec<Option<usize>> = vec![None; nr];
    let mut row_of_col: Vec<Option<usize>> = vec![None; nc];
    let mut dist = vec![UNSEEN; nr];
    let mut queue = Vec::with_capacity(nr);
    let mut cursor = vec![0usize; nr];
    let mut stack: Vec<usize> = Vec::new();
    loop {
        queue.clear();
        for r in 0..nr {
            if col_of_row[r].is_none() {
                dist[r] = 0;
                queue.push(r);
            } else {
                dist[r] = UNSEEN;
            }
        }
        let mut found = false;
        let mut head = 0;
        while head < queue.len() {
            let r = queue[head];
            head += 1;
            for &c in g.row(r) {
                match row_of_col[c] {
                    None => found = true,
                    Some(r2) if dist[r2] == UNSEEN => {
                        dist[r2] = dist[r] + 1;
                        queue.push(r2);
                    }
                    Some(_) => {}
                }
            }
        }
        if !found {
            break;
        }
        cursor.fill(0);
        for r0 in 0..nr {
            if col_of_row[r0].is_some() {
                continue;
            }
            stack.clear();
            stack.push(r0);
            while let Some(&r) = stack.last() {
                let row = g.row(r);
                if cursor[r] == row.len() {
                    dist[r] = UNSEEN; // a dead end for the rest of this phase
                    stack.pop();
                    continue;
                }
                let c = row[cursor[r]];
                cursor[r] += 1;
                match row_of_col[c] {
                    None => {
                        // Augment: every row on the stack moves to the column it last stepped through.
                        for &ri in &stack {
                            let ci = g.row(ri)[cursor[ri] - 1];
                            col_of_row[ri] = Some(ci);
                            row_of_col[ci] = Some(ri);
                        }
                        break;
                    }
                    Some(r2) if dist[r2] != UNSEEN && dist[r2] == dist[r] + 1 => stack.push(r2),
                    Some(_) => {}
                }
            }
        }
    }
    Matching {
        col_of_row,
        row_of_col,
    }
}

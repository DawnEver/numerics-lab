//! Tarjan's strongly connected components, iterative.

use super::graph::Digraph;

/// A partition of the nodes into strongly connected components.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Components {
    /// The component of each node, numbered DEPENDENCY-FIRST: every component reachable from
    /// component `k` along edges has a label `<= k`, so evaluating `0, 1, ...` in order meets every
    /// edge's target before its source.
    pub label: Vec<usize>,
    /// Number of components.
    pub count: usize,
}

/// The strongly connected components of `g` (Tarjan 1972: one DFS with low-links; a component closes
/// when its root's low-link equals its index, after every component it reaches has closed).
#[must_use]
pub fn strongly_connected_components(g: &Digraph) -> Components {
    const UNSEEN: usize = usize::MAX;
    let n = g.n();
    let mut index = vec![UNSEEN; n];
    let mut low = vec![0usize; n];
    let mut on_stack = vec![false; n];
    let mut stack = Vec::new();
    let mut label = vec![0usize; n];
    let mut count = 0;
    let mut next = 0;
    let mut call: Vec<(usize, usize)> = Vec::new();
    for s in 0..n {
        if index[s] != UNSEEN {
            continue;
        }
        index[s] = next;
        low[s] = next;
        next += 1;
        stack.push(s);
        on_stack[s] = true;
        call.push((s, 0));
        while let Some(top) = call.len().checked_sub(1) {
            let (v, k) = call[top];
            if let Some(&w) = g.successors(v).get(k) {
                call[top].1 += 1;
                if index[w] == UNSEEN {
                    index[w] = next;
                    low[w] = next;
                    next += 1;
                    stack.push(w);
                    on_stack[w] = true;
                    call.push((w, 0));
                } else if on_stack[w] {
                    low[v] = low[v].min(index[w]);
                }
                continue;
            }
            call.pop();
            if let Some(&(u, _)) = call.last() {
                low[u] = low[u].min(low[v]);
            }
            if low[v] == index[v] {
                while let Some(w) = stack.pop() {
                    on_stack[w] = false;
                    label[w] = count;
                    if w == v {
                        break;
                    }
                }
                count += 1;
            }
        }
    }
    Components { label, count }
}

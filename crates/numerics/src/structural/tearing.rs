//! Tearing: the nodes whose values are guessed so the rest of a cyclic dependency graph evaluates in order.

use super::graph::Digraph;

/// A tearing of a digraph whose edge `u -> v` reads "`u` needs `v` first".
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Tearing {
    /// The torn nodes, in the order they were chosen. With their values given, the rest is acyclic.
    pub tears: Vec<usize>,
    /// Every other node in an evaluation order: each one's successors are torn or earlier in this list.
    pub order: Vec<usize>,
}

/// Tear `g` greedily: evaluate every node whose dependencies are all met; when none is, tear the
/// remaining node with the most unmet dependencies (lowest index on a tie) and continue.
///
/// The result is valid (removing `tears` leaves `order` a topological order) for every digraph,
/// a self-loop included; it is a heuristic, not a minimum feedback vertex set (that problem is NP-hard).
///
/// # Panics
/// Never: a tear is chosen only while a node is left.
#[must_use]
pub fn tear(g: &Digraph) -> Tearing {
    let n = g.n();
    let preds = g.reversed();
    let mut unmet: Vec<usize> = (0..n).map(|v| g.successors(v).len()).collect();
    let mut done = vec![false; n];
    let mut ready: Vec<usize> = (0..n).rev().filter(|&v| unmet[v] == 0).collect();
    let mut tears = Vec::new();
    let mut order = Vec::with_capacity(n);
    let mut left = n;
    let release = |v: usize, done: &[bool], unmet: &mut [usize], ready: &mut Vec<usize>| {
        for &p in preds.successors(v) {
            if !done[p] {
                unmet[p] -= 1;
                if unmet[p] == 0 {
                    ready.push(p);
                }
            }
        }
    };
    while left > 0 {
        if let Some(v) = ready.pop() {
            if done[v] {
                continue;
            }
            done[v] = true;
            left -= 1;
            order.push(v);
            release(v, &done, &mut unmet, &mut ready);
            continue;
        }
        let t = (0..n)
            .filter(|&v| !done[v])
            .max_by(|&a, &b| unmet[a].cmp(&unmet[b]).then(b.cmp(&a)))
            .expect("a node is left");
        done[t] = true;
        left -= 1;
        tears.push(t);
        release(t, &done, &mut unmet, &mut ready);
    }
    Tearing { tears, order }
}

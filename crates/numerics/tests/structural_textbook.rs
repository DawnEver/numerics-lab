//! `numerics::structural` on textbook cases whose answers are known by hand, each with a planted control
//! that the check can say no to.

use numerics::structural::{
    block_triangular, dulmage_mendelsohn, maximum_matching, pantelides,
    strongly_connected_components, tear, Bipartite, Digraph, StructureError,
};

fn bip(n_rows: usize, n_cols: usize, entries: &[(usize, usize)]) -> Bipartite {
    let (r, c): (Vec<usize>, Vec<usize>) = entries.iter().copied().unzip();
    Bipartite::from_incidence(n_rows, n_cols, &r, &c).expect("a valid pattern")
}

fn dig(n: usize, edges: &[(usize, usize)]) -> Digraph {
    let (s, t): (Vec<usize>, Vec<usize>) = edges.iter().copied().unzip();
    Digraph::from_edges(n, &s, &t).expect("a valid digraph")
}

/// Every row in `order[bounds[k]..]` touches only columns placed at or before its block's end.
fn is_block_lower(
    g: &Bipartite,
    rows: &[usize],
    cols: &[usize],
    rb: &[usize],
    cb: &[usize],
) -> bool {
    let mut col_block = vec![0; g.n_cols()];
    for k in 0..cb.len() - 1 {
        for &c in &cols[cb[k]..cb[k + 1]] {
            col_block[c] = k;
        }
    }
    (0..rb.len() - 1).all(|k| {
        rows[rb[k]..rb[k + 1]]
            .iter()
            .all(|&r| g.row(r).iter().all(|&c| col_block[c] <= k))
    })
}

#[test]
fn hopcroft_karp_finds_the_augmenting_path_a_greedy_matching_misses() {
    // Greedy row-by-row takes (0,0) and strands row 1; the maximum matches both.
    let g = bip(2, 2, &[(0, 0), (0, 1), (1, 0)]);
    let m = maximum_matching(&g);
    assert_eq!(m.col_of_row, vec![Some(1), Some(0)]);
    assert_eq!(m.size(), 2);
    // Planted: a pattern of rank 1 is reported as rank 1.
    assert_eq!(maximum_matching(&bip(2, 2, &[(0, 0), (1, 0)])).size(), 1);
}

#[test]
fn tarjan_numbers_components_dependency_first() {
    // 0 -> 1 -> 2 -> 1, 2 -> 3: components {3}, {1, 2}, {0}, in that order.
    let c = strongly_connected_components(&dig(4, &[(0, 1), (1, 2), (2, 1), (2, 3)]));
    assert_eq!(c.count, 3);
    assert_eq!(c.label, vec![2, 1, 1, 0]);
    // Planted: breaking the cycle splits {1, 2}.
    assert_eq!(
        strongly_connected_components(&dig(4, &[(0, 1), (1, 2), (2, 3)])).count,
        4
    );
}

#[test]
fn a_long_chain_does_not_overflow_the_stack() {
    let n = 200_000;
    let edges: Vec<(usize, usize)> = (0..n - 1).map(|v| (v, v + 1)).chain([(n - 1, 0)]).collect();
    assert_eq!(strongly_connected_components(&dig(n, &edges)).count, 1);
    let path: Vec<(usize, usize)> = (0..n).flat_map(|r| [(r, r), (r, (r + 1) % n)]).collect();
    assert_eq!(maximum_matching(&bip(n, n, &path)).size(), n);
}

#[test]
fn blt_orders_a_triangular_system_with_a_two_by_two_loop() {
    // x0 = f(); x1, x2 coupled and using x0; x3 uses x2.
    let g = bip(
        4,
        4,
        &[
            (0, 0),
            (1, 1),
            (1, 2),
            (1, 0),
            (2, 1),
            (2, 2),
            (3, 3),
            (3, 2),
        ],
    );
    let b = block_triangular(&g).expect("nonsingular");
    assert_eq!(b.bounds, vec![0, 1, 3, 4]);
    assert_eq!(b.row_order[0], 0);
    assert_eq!(b.row_order[3], 3);
    assert!(is_block_lower(
        &g,
        &b.row_order,
        &b.col_order,
        &b.bounds,
        &b.bounds
    ));
    // Planted: the reversed order is NOT block lower.
    let rev_r: Vec<usize> = b.row_order.iter().rev().copied().collect();
    let rev_c: Vec<usize> = b.col_order.iter().rev().copied().collect();
    assert!(!is_block_lower(&g, &rev_r, &rev_c, &b.bounds, &b.bounds));
    assert_eq!(
        block_triangular(&bip(2, 2, &[(0, 0), (1, 0)])),
        Err(StructureError::StructurallySingular { rank: 1, n: 2 })
    );
    assert_eq!(
        block_triangular(&bip(2, 3, &[])),
        Err(StructureError::NotSquare { rows: 2, cols: 3 })
    );
}

#[test]
fn dulmage_mendelsohn_separates_over_square_and_under_parts() {
    // Rows 0,1 both use only column 0 (over-determined); row 2 uses columns 1 (square, alone);
    // row 3 uses columns 2 and 3 (under-determined).
    let g = bip(
        4,
        4,
        &[(0, 0), (1, 0), (2, 1), (2, 0), (3, 2), (3, 3), (3, 1)],
    );
    let d = dulmage_mendelsohn(&g);
    assert_eq!(d.structural_rank(), 3);
    let rb = &d.row_bounds;
    let cb = &d.col_bounds;
    assert_eq!(&d.row_order[rb[0]..rb[1]], &[0, 1]);
    assert_eq!(&d.col_order[cb[0]..cb[1]], &[0]);
    assert_eq!(&d.row_order[rb[1]..rb[2]], &[2]);
    let last = rb.len() - 2;
    assert_eq!(&d.row_order[rb[last]..rb[last + 1]], &[3]);
    assert_eq!(&d.col_order[cb[last]..cb[last + 1]], &[2, 3]);
    assert!(is_block_lower(&g, &d.row_order, &d.col_order, rb, cb));
    // A nonsingular square pattern has empty over and under parts.
    let sq = dulmage_mendelsohn(&bip(2, 2, &[(0, 0), (1, 1), (1, 0)]));
    assert_eq!(sq.row_bounds, vec![0, 0, 1, 2, 2]);
}

#[test]
fn tearing_breaks_every_cycle_and_orders_the_rest() {
    // 0 <-> 1 <-> 2, 3 needs 2: tearing the most-dependent node (1) frees the rest.
    let g = dig(4, &[(0, 1), (1, 0), (1, 2), (2, 1), (3, 2)]);
    let t = tear(&g);
    assert_eq!(t.tears, vec![1]);
    assert_eq!(t.order.len(), 3);
    let mut pos = [usize::MAX; 4];
    for (k, &v) in t.order.iter().enumerate() {
        pos[v] = k;
    }
    for v in 0..4 {
        for &w in g.successors(v) {
            assert!(
                t.tears.contains(&w) || t.tears.contains(&v) || pos[w] < pos[v],
                "{v} -> {w}"
            );
        }
    }
    // A self-loop must be torn; an acyclic graph needs no tear.
    assert_eq!(tear(&dig(1, &[(0, 0)])).tears, vec![0]);
    assert!(tear(&dig(3, &[(2, 1), (1, 0)])).tears.is_empty());
    assert_eq!(tear(&dig(3, &[(2, 1), (1, 0)])).order, vec![0, 1, 2]);
}

/// The planar pendulum (Pantelides 1988; Hairer & Wanner): index 3, so the length constraint is
/// differentiated twice and the two kinematic equations once each.
#[test]
fn pantelides_reduces_the_pendulum() {
    // vars: 0 x, 1 y, 2 u, 3 v, 4 lambda, 5 x', 6 y', 7 u', 8 v'
    // eqs: x' = u; y' = v; u' = -lambda x; v' = -lambda y - g; x^2 + y^2 = L^2
    let g = bip(
        5,
        9,
        &[
            (0, 5),
            (0, 2),
            (1, 6),
            (1, 3),
            (2, 7),
            (2, 4),
            (2, 0),
            (3, 8),
            (3, 4),
            (3, 1),
            (4, 0),
            (4, 1),
        ],
    );
    let der = [
        Some(5),
        Some(6),
        Some(7),
        Some(8),
        None,
        None,
        None,
        None,
        None,
    ];
    let r = pantelides(&g, &der).expect("the pendulum is structurally regular");
    assert_eq!(r.differentiations(5), vec![1, 1, 0, 0, 2]);
    assert_eq!(r.equations.len(), 9);
    // Every highest-derivative variable is matched to a distinct equation.
    let highest: Vec<usize> = (0..r.variable_derivative.len())
        .filter(|&j| r.variable_derivative[j].is_none())
        .collect();
    let mut eqs: Vec<usize> = highest
        .iter()
        .map(|&j| r.assignment[j].expect("matched"))
        .collect();
    eqs.sort_unstable();
    eqs.dedup();
    assert_eq!(eqs.len(), highest.len());
    // Planted: an index-1 system (the constraint replaced by one on lambda) needs no differentiation.
    let easy = bip(
        5,
        9,
        &[
            (0, 5),
            (0, 2),
            (1, 6),
            (1, 3),
            (2, 7),
            (2, 4),
            (2, 0),
            (3, 8),
            (3, 4),
            (3, 1),
            (4, 4),
            (4, 0),
        ],
    );
    assert_eq!(
        pantelides(&easy, &der)
            .expect("index 1")
            .differentiations(5),
        vec![0; 5]
    );
}

#[test]
fn pantelides_refuses_a_structurally_singular_dae_and_a_bad_derivative_map() {
    // Two equations in x alone, x' only through its derivative: no differentiation helps.
    let g = bip(2, 2, &[(0, 0), (1, 0)]);
    assert!(matches!(
        pantelides(&g, &[Some(1), None]),
        Err(StructureError::IndexBoundExceeded { .. })
    ));
    assert_eq!(
        pantelides(&bip(1, 2, &[(0, 0)]), &[Some(1), Some(0)]),
        Err(StructureError::DerivativeNotAChain { variable: 0 })
    );
    assert_eq!(
        pantelides(&bip(1, 1, &[(0, 0)]), &[Some(0)]),
        Err(StructureError::DerivativeNotAChain { variable: 0 })
    );
}

#[test]
fn an_out_of_range_index_is_refused_by_name() {
    assert_eq!(
        Bipartite::from_incidence(2, 2, &[0, 2], &[0, 0]),
        Err(StructureError::IndexOutOfRange {
            axis: "row",
            index: 2,
            bound: 2
        })
    );
    assert_eq!(
        Digraph::from_edges(2, &[0], &[0, 1]),
        Err(StructureError::LengthMismatch { left: 1, right: 2 })
    );
}

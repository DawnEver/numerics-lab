//! `numerics::structural` against `scipy.sparse.csgraph` through the committed `goldens/structural.jsonl` that
//! `tests/unit/test_structural_graph_port_golden.py` freezes from live SciPy.
//!
//! A maximum matching is not unique, so its SIZE is compared (and the matching checked valid); strong components
//! are compared as a PARTITION (labels are arbitrary up to renaming). Planted: a wrong rank and a merged
//! partition are seen.

use numerics::structural::{maximum_matching, strongly_connected_components, Bipartite, Digraph};
use serde_json::Value;

fn ids(v: &Value) -> Vec<usize> {
    v.as_array()
        .expect("an array")
        .iter()
        .map(|x| usize::try_from(x.as_u64().expect("an index")).expect("fits"))
        .collect()
}

fn golden() -> Value {
    serde_json::from_str(include_str!("goldens/structural.jsonl"))
        .expect("goldens/structural.jsonl")
}

fn same_partition(a: &[usize], b: &[usize]) -> bool {
    let pairs: std::collections::BTreeSet<(usize, usize)> =
        a.iter().copied().zip(b.iter().copied()).collect();
    let left: std::collections::BTreeSet<usize> = pairs.iter().map(|p| p.0).collect();
    let right: std::collections::BTreeSet<usize> = pairs.iter().map(|p| p.1).collect();
    a.len() == b.len() && pairs.len() == left.len() && pairs.len() == right.len()
}

#[test]
fn the_matching_reaches_scipys_structural_rank() {
    let g = golden();
    let cases = g["matching"].as_array().expect("matching");
    assert!(cases.len() >= 6, "the golden lost a pattern");
    for c in cases {
        let (rows, cols) = (ids(&c["rows"]), ids(&c["cols"]));
        let n_rows = usize::try_from(c["n_rows"].as_u64().expect("n_rows")).expect("fits");
        let n_cols = usize::try_from(c["n_cols"].as_u64().expect("n_cols")).expect("fits");
        let bp = Bipartite::from_incidence(n_rows, n_cols, &rows, &cols).expect("valid");
        let m = maximum_matching(&bp);
        let rank = usize::try_from(c["rank"].as_u64().expect("rank")).expect("fits");
        assert_eq!(m.size(), rank);
        assert_ne!(m.size(), rank + 1, "planted: a wrong rank is seen");
        for (r, c) in m.col_of_row.iter().enumerate() {
            if let Some(c) = *c {
                assert!(bp.row(r).contains(&c));
                assert_eq!(m.row_of_col[c], Some(r));
            }
        }
    }
}

#[test]
fn the_components_are_scipys_partition() {
    let g = golden();
    let cases = g["scc"].as_array().expect("scc");
    assert!(cases.len() >= 4, "the golden lost a digraph");
    for c in cases {
        let n = usize::try_from(c["n"].as_u64().expect("n")).expect("fits");
        let dg = Digraph::from_edges(n, &ids(&c["sources"]), &ids(&c["targets"])).expect("valid");
        let got = strongly_connected_components(&dg);
        let want = ids(&c["labels"]);
        assert!(same_partition(&got.label, &want));
        for v in 0..n {
            for &w in dg.successors(v) {
                assert!(got.label[w] <= got.label[v], "dependency-first numbering");
            }
        }
    }
    // Planted: merging two components is seen.
    assert!(!same_partition(&[0, 1, 2], &[0, 0, 1]));
}

use crate::linear::constrained::master_slave::{resolve_constraint_chains, ConstraintError};

#[test]
fn a_set_with_no_chain_comes_back_untouched() {
    let cs = vec![(5, vec![(1, 0.5), (2, 0.5)]), (7, vec![(3, 1.0)])];
    assert_eq!(resolve_constraint_chains(&cs).unwrap(), cs);
}

#[test]
fn a_chain_is_substituted_out_so_no_master_is_itself_a_slave() {
    // 12 <- {11, 13}, 11 <- {6, 16}: the exact shape the Python's docstring measures.
    let cs = vec![
        (12, vec![(11, 0.5), (13, 0.5)]),
        (11, vec![(6, 0.5), (16, 0.5)]),
    ];
    let out = resolve_constraint_chains(&cs).unwrap();
    let slaves: Vec<usize> = out.iter().map(|(h, _)| *h).collect();
    assert_eq!(slaves, vec![12, 11], "the declaration order must survive");
    for (_, masters) in &out {
        for (m, _) in masters {
            assert!(!slaves.contains(m), "master {m} is still a slave: {out:?}");
        }
    }
    let twelve = &out[0].1;
    let weight = |t: usize| {
        twelve
            .iter()
            .find(|(m, _)| *m == t)
            .map_or(0.0, |(_, w)| *w)
    };
    assert!((weight(6) - 0.25).abs() < 1e-15, "{twelve:?}");
    assert!((weight(16) - 0.25).abs() < 1e-15, "{twelve:?}");
    assert!((weight(13) - 0.5).abs() < 1e-15, "{twelve:?}");
}

/// A CYCLE IS NOT AUTOMATICALLY CONTRADICTORY, and refusing one was the Python's own earlier
/// mistake. `P = 0.5A + 0.5B`, `A = 0.5P + 0.5C` has the unique solution
/// `P = (0.25 C + 0.5 B) / 0.75`.
#[test]
fn a_mutual_pair_resolves_rather_than_being_refused() {
    let cs = vec![(0, vec![(1, 0.5), (2, 0.5)]), (1, vec![(0, 0.5), (3, 0.5)])];
    let out = resolve_constraint_chains(&cs).unwrap();
    let p = &out[0].1;
    let w = |t: usize| p.iter().find(|(m, _)| *m == t).map_or(0.0, |(_, x)| *x);
    assert!((w(2) - 0.5 / 0.75).abs() < 1e-14, "{p:?}");
    assert!((w(3) - 0.25 / 0.75).abs() < 1e-14, "{p:?}");
}

#[test]
fn a_singular_cycle_is_refused_and_names_its_nodes() {
    // A = B and B = A: `I - W_cc` is exactly singular.
    let cs = vec![(0, vec![(1, 1.0)]), (1, vec![(0, 1.0)])];
    assert_eq!(
        resolve_constraint_chains(&cs),
        Err(ConstraintError::Singular { nodes: vec![0, 1] })
    );
}

#[test]
fn a_resolved_weight_that_cancels_to_zero_leaves_no_master() {
    // 0 <- {1: +1, 2: +1}; 1 <- {2: -1, 3: +1}. Substituting gives 0 <- {2: 0, 3: 1}, and
    // `np.flatnonzero` drops the zero rather than carrying a master with no weight.
    let cs = vec![
        (0, vec![(1, 1.0), (2, 1.0)]),
        (1, vec![(2, -1.0), (3, 1.0)]),
    ];
    let out = resolve_constraint_chains(&cs).unwrap();
    assert_eq!(out[0].0, 0);
    assert_eq!(
        out[0].1,
        vec![(3, 1.0)],
        "the cancelled master must not survive"
    );
}

#[test]
fn only_the_entangled_closure_is_solved_and_the_rest_pass_through() {
    let cs = vec![
        (10, vec![(1, 1.0)]),  // independent
        (20, vec![(21, 1.0)]), // chained
        (21, vec![(22, 0.25)]),
    ];
    let out = resolve_constraint_chains(&cs).unwrap();
    assert_eq!(out[0], (10, vec![(1, 1.0)]));
    assert_eq!(out[1], (20, vec![(22, 0.25)]));
    assert_eq!(out[2], (21, vec![(22, 0.25)]));
}

#[test]
fn an_empty_set_is_returned_empty() {
    assert!(resolve_constraint_chains(&[]).unwrap().is_empty());
}

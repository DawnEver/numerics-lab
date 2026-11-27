//! Structural graph algorithms on sparse bipartite and directed graphs -- domain-free.
//!
//! An incidence pattern (rows x columns, e.g. equations x unknowns) is a [`Bipartite`] graph; a
//! dependency relation is a [`Digraph`]. Nothing here knows what a row or a column MEANS: the caller
//! builds the graph and interprets the answer.
//!
//! * [`maximum_matching`] -- Hopcroft-Karp, `O(E sqrt(V))`.
//! * [`dulmage_mendelsohn`] -- the coarse (over / square / under-determined) and fine decomposition.
//! * [`block_triangular`] -- the block lower triangular (BLT) form of a structurally nonsingular square pattern.
//! * [`strongly_connected_components`] -- Tarjan, components numbered dependency-first.
//! * [`tear`] -- tearing nodes that make a digraph acyclic, and the evaluation order of the rest.
//! * [`pantelides`] -- Pantelides structural index reduction of a DAE pattern.
//!
//! Every search is iterative: a deep alternating path or a long dependency chain cannot overflow the stack.

mod dm;
mod graph;
mod matching;
mod pantelides;
mod scc;
mod tearing;

pub use dm::{block_triangular, dulmage_mendelsohn, BlockTriangular, Decomposition};
pub use graph::{Bipartite, Digraph, StructureError};
pub use matching::{maximum_matching, Matching};
pub use pantelides::{pantelides, IndexReduction};
pub use scc::{strongly_connected_components, Components};
pub use tearing::{tear, Tearing};

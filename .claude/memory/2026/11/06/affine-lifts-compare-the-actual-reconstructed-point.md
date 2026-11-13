---
name: affine-lifts-compare-the-actual-reconstructed-point
description: Preserve legacy correction arithmetic but recover projected points lost against huge proposals
created: 2026-11-06
metadata:
  type: project
---

# Affine lift recovery uses its actual map, not a guessed magnitude cutoff

Execution client: Codex. Prior art: _lift distinguishes incremental affine
projection from absolute limiter lifting, and the existing real StepForm
control preserves untouched endpoint rows and nonbinary-weight arithmetic.
Review F-843bd6485180 identifies another valid source counterexample: with
zero history and alpha=1, a proposal 1e20 projected to 1 loses that one in
the subtraction/addition. Opposite finite extremes can overflow their delta.
These are analytic source counterexamples, not locally executed failures.

Do not unconditionally replace incremental arithmetic with absolute lifting:
that would undo the earlier history-offset correction and gratuitously change
ordinary affine trials. First form the existing incremental result. Only if
its actual StepForm.u_arg reconstruction misses a changed projected row,
consider the absolute inverse as another representable lift. Select it only
if it is finite and its reconstructed physical point is strictly closer, or
the incremental endpoint is nonfinite. Retain incremental arithmetic on ties
and preserve every untouched endpoint bit. There is no numerical acceptance
floor, changed residual policy, extra projection callback or new solver here.

Five authored UNRUN real StepForm controls cover large proposal/small point
with history and nonunit weights, untouched rows, and opposite finite extremes
under strict overflow/invalid reporting. Existing correction/limiter controls
remain unchanged. All runtime validation must occur remotely at the new head;
parent test receipts cannot be reused. Other review findings remain open.

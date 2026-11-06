# A bounded endpoint owns its direction without losing small terms

UNVERIFIED: all local tests/imports/numerical execution remain UNRUN.

PR9 finding F-981c3a9da722 identified duplicate caller/callee endpoint subtraction
and rejection of a legitimate finite endpoint when its difference overflows.
The explicit endpoint is now the sole direction authority: its caller supplies
step=None, and supplying both is refused, with no alias/compatibility branch.
Ordinary step rays remain available.

Ordinary finite np.dot and norm arithmetic is unchanged. Only an unrepresentable
endpoint difference or nonfinite dot/norm result engages finite-input exact
Fraction arithmetic. Per-coordinate differences and products remain intact;
normalizing a huge direction would silently erase a meaningful tiny component.
Fraction slopes are never converted to float. Only a known bounded interval
ratio returns to float for the SAME Illinois search; curvature comparisons and
halving preserve exact fallback values. An out-of-interior secant uses that
search's midpoint, not an unbounded parameter. Normal finite rounding is NOT
claimed to be globally exact.

Non-descent still takes a full step when it evaluates, but no longer returns an
unevaluated point. TrialNotEvaluable/nonfinite residuals close the existing
bracket; no evaluable trial still refuses. Nonfinite callback residuals never
reach Fraction conversion or become a successful endpoint.

Authored controls cover extreme endpoint differences with cancelling huge terms
and a meaningful tiny component, ordinary mixed-magnitude finite dots, overflowed
products/norms, explicit dual-authority refusal, and non-descent invalid domains,
nonfinite residuals and complete refusal. Controls remain UNRUN. Ruff, format and
whitespace passed; remote receipts for ebabbba do not validate this candidate.
No physical law, acceptance bar, limiter or directional capability is removed.

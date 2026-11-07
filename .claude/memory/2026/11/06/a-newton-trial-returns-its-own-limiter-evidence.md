# A Newton trial returns its own limiter evidence

UNVERIFIED: local tests/imports/numerical execution are forbidden in this session.

PR9 review F-1348fed86fc8 identified a mutable one-element list side channel
between the physical trial owner and Newton/continuation callers. The shared
operation now returns (endpoint, limited) directly. Both production callers and
all direct controls consume that result; rejected probe metadata cannot overwrite
another retained trial's evidence through a shared cell. No alias or old optional
out-parameter remains, and no residual or acceptance standard changed.

Existing physical limiter controls now assert changed evidence; affine projection
with no limiter or an identity limiter asserts unchanged evidence. These controls
are authored UNRUN. Only Ruff, formatting and whitespace checks passed. This slice
does not fix the separate directional feasibility and overflow review findings.

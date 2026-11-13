# A directional ray inherits only a tested projected iterate

UNVERIFIED: all local tests/imports/numerical execution remain forbidden and UNRUN.

PR9 findings F-1a2145ae45b8 and F-18032f2d8d05 share the initial off-plane
directional recovery boundary. A projector's bit-exact idempotence is not its
physical admission certificate; floating reapplication can drift one bit forever.
Nor may a projected/limited endpoint replace the retained iterate without an
actual residual and the existing merit check.

Recovery now uses the SAME existing backtracking trial/accept predicates, with
the configured budget (one checked full trial when cap is zero). Refused or
unevaluable recovery retains the old point, never the untried fallback. Accepted
projected trials carry point-local provenance: the projector owns its actual
affine map and the limiter contract preserves it. Without an admission protocol,
that provenance permits the next actual directional ray without reprojecting for
bitwise equality. An explicit admission certificate remains authoritative. Bare
unassessed assignments do not gain provenance; discarded probes cannot change
the retained point's evidence. No tolerance, residual, law or convergence bar changed.

Authored controls cover a non-idempotent-at-last-bit projector that reaches actual
directional search after checked recovery, invalid-domain and worse-merit recovery
at small budgets/cap zero, and ordinary errors plus four typed cancellation objects
and their original causes. The removed private fixed-point helper is qualified-
retired with no alias. Only Ruff/format/whitespace passed. Remote base receipts for
ebabbba do not validate these new source slices; directional overflow remains open.

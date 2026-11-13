# Affine projection keeps its correction and untouched rows

UNVERIFIED: local tests, numerical imports and solver execution are forbidden
for this session. The new controls are authored, not run.

PR #9 review at a71c374c34 reported F-f1173941188c and F-7d44e5120a54.
Source confirms that shared bounded-point lifting rebuilt affine projections
from their full history offset even without a limiter. That unnecessarily
replaced the prior correction-only arithmetic and could perturb a coordinate
the physical projector did not change.

The shared lift now distinguishes an affine correction from a limiter-owned
physical bound. Projection uses the actual physical delta over the state
weight; limiting still reconstructs the bounded point, avoiding cancellation
against the discarded huge proposal. Both paths copy untouched endpoint
coordinates exactly. An identity transformation retains its fast path.

Controls cover a nonbinary 0.3 state weight and 0.7 history offset, with and
without an identity limiter. Existing huge-proposal limiter controls remain
unchanged. No acceptance tolerance or constitutive law changed. The remaining
directional-search and metadata review findings remain open.

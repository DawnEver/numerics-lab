---
name: thin-joint-response-columns-own-column-not-row-pointers
description: Assemble wide joint responses from CSC thin columns, retaining exact support and original numerical acceptance without field-sized CSR pointers per column.
metadata:
  type: project
---

# Column storage for thin joint responses

The application's second four-GiB full initialization still terminated for
memory after the all-row rate retry returned true and the joint constructor
recorded its 78842 and 179726 factors. No complete initial pair or integration
was reached. The precise terminal allocation was not captured.

Source inspection found a definite avoidable representation cost: component
mode retained one field-sized CSR row-pointer array for EVERY thin response
column, including empty columns, before hstack. The error columns repeated it.
For 517136 rows and 629 columns, int32 pointers alone would cost about 1.3 GB
per assembly. Those dimensions illustrate the mechanism, not a captured column
count from the failed application's joint response.

Thin intermediates now use CSC, hence two pointers per column; the assembled
result remains CSR. An additional CSC coupling view supplies column support
and forcing entries directly. Only structurally empty forcing skips computation;
even 1e-300 support remains. Original solve order, original equations, row bars,
allowance propagation, uniqueness and singularity refusal remain unchanged.

TDD: five RED tests, including two actual 1200-row SPD-arrow joint systems
whose original numerical assertions passed before the storage assertion failed.
Initial test construction errors and a decaying-chain underflow fixture were
retained in earlier RED logs, then corrected without changing solver bars.
Final canonical verification: 59 passed; whole-repo Ruff and 264 format checks
pass. CPU1/1GiB/300s, peak249147392 bytes, wall7.169286 seconds.
Receipts: output/logs/26/10/31/wide-joint-columns-{red45-representable,green45}.

This proves the representation change and original numerical regressions only.
Application memory fit, complete initialization and external physics accuracy
remain unmeasured with this source. A fresh unchanged-budget native admission
is required; do not raise the cap or discard response coefficients.

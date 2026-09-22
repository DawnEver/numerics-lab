"""A multi-operand contraction must reach BLAS, and every assembly site must go through the one
helper that guarantees it.

THE MEASUREMENT THAT PUT THIS HERE, taken 2026-09-16 on ``cases/field/pm/spm_slotted`` at
``SPLINE_REFINE = (2, 2)`` with ``compile_spline_magnetostatic``. Every ``np.einsum`` on the
assembly path was timed by subscript, and the same call priced with ``optimize=True``:

===========================  =====  ==========  =========  ========
subscripts                       n  naive (s)   opt (s)     speedup
===========================  =====  ==========  =========  ========
``qid,qjd,q->ij``               38       82.50       2.33     35.5x
``qid,qde->qie``               812       26.89       1.47     18.3x
``qid,ic->qcd``                850        9.12       3.61      2.5x
``qi,qj,q->ij``                136        0.71       0.06     11.5x
TOTAL                         1972      119.23       7.48     15.9x
===========================  =====  ==========  =========  ========

119.2 s of a 136.5 s compile -- **87%** -- was one defect: ``numpy.einsum`` defaults to
``optimize=False``, which runs its own naive C loop instead of factoring the contraction into
pairwise ``tensordot`` calls that land on ``dgemm``. Both languages are compiled; what differed
was the ALGORITHM, not the implementation.

Note WHICH site dominated, because the lane's standing suspicion was the other one: the 82.5 s is
the VOLUME stiffness (``spline.py``), while ``trim_nitsche_blocks`` -- the 68 cut interfaces that
were the prime suspect for the assembly blow-up -- account for the 0.71 s row, 0.6% of the budget.

WHY A HELPER RATHER THAN ``optimize=True`` AT EACH SITE: the defect is a DEFAULT, so any site that
forgets the keyword silently re-acquires it and nothing fails. The scan below is what makes the
declaration one the code must consult.
"""

from __future__ import annotations

import ast
import time
from pathlib import Path

import numpy as np
import pytest

import motronics
from motronics.gauss.contract import contract

#: The module GLOBS this scan reads. A GLOB AND NOT A HAND-WRITTEN LIST, and that is the whole
#: repair (2026-09-22). The list that stood here named four files:
#:
#:     gauss/quadrature.py, hamilton/discretizations/spline{,_interface,_nonlinear}.py
#:
#: and all four still EXIST, so the `path.is_file()` guard below stayed green while the code walked
#: out from under it. `spline.py` was split into `spline_estimate`, `spline_stabilise`,
#: `spline_structural`, `spline_thermal`, `spline_trim_interface`, `spline_airgap` and
#: `spline_electrostatic`; `gauss/quadrature.py`'s contractions moved to `gauss/spline_basis.py`.
#: The four named files were left holding 7 contractions against a floor of 12 -- which is the
#: floor doing exactly its job, and the reason this was found at all.
#:
#: A SPLIT CANNOT EMPTY A GLOB. That is the property being bought: the failure mode here was never
#: "someone deleted a module", it was "someone moved the code and the list still resolved".
_ASSEMBLY_MODULE_GLOBS = (
    'gauss/quadrature.py',
    'gauss/spline_basis.py',
    'hamilton/discretizations/spline*.py',
)

#: Below this, the scan is not looking at the code that was measured. A scan that finds NOTHING is
#: vacuous rather than green, so it carries a floor as well as a verdict.
#:
#: RE-MEASURED 2026-09-22 after the globs above replaced the stale list: 37 multi-operand
#: contractions across 18 modules, against the 7 the four named files still held. The floor sits
#: below the census rather than at it, so adding an assembly site does not red this.
_MINIMUM_CONTRACTIONS_SCANNED = 30

#: The modules that STILL take numpy's naive path, by name, and this set MAY ONLY SHRINK.
#:
#: WHY THEY ARE RECORDED RATHER THAN CONVERTED HERE. The scan was blind for as long as the split
#: has existed, and 18 naive multi-operand contractions accumulated in the new siblings behind it.
#: Converting them is not a no-op: `optimize=` changes the CONTRACTION ORDER, which changes the
#: floating-point associativity of a structural, thermal and electrostatic assembly. That is a
#: numerics change owed a measurement against this repo's goldens, and bundling it into the repair
#: of the INSTRUMENT would land it unmeasured -- so the debt is named and frozen instead.
#:
#: NAMED MODULES AND NOT A COUNT OR A LINE LIST: a count is blind to a swap (one converted, one
#: added, still 18) and line numbers move under any edit. A module leaving this set is a
#: conversion; a module ARRIVING is a new naive contraction in code that was clean, and the
#: equality below reds on both -- which is what makes it a ratchet rather than an allowlist.
_NAIVE_EINSUM_DEBT = frozenset({
    'hamilton/discretizations/spline_airgap.py',
    'hamilton/discretizations/spline_electrostatic.py',
    'hamilton/discretizations/spline_estimate.py',
    'hamilton/discretizations/spline_stabilise.py',
    'hamilton/discretizations/spline_structural.py',
    'hamilton/discretizations/spline_thermal.py',
})


def _source_root() -> Path:
    return Path(motronics.__file__).parent


@pytest.mark.parametrize(
    ('subscripts', 'shapes'),
    [
        # The four subscripts the profile above actually measured, at their measured shapes.
        ('qid,qjd,q->ij', ((400, 105, 2), (400, 105, 2), (400,))),
        ('qid,qde->qie', ((400, 105, 2), (400, 2, 2))),
        ('qid,ic->qcd', ((400, 105, 2), (105, 3))),
        ('qi,qj,q->ij', ((256, 210), (256, 210), (256,))),
    ],
)
def test_the_HELPER_AGREES_WITH_THE_NAIVE_CONTRACTION_it_replaces(subscripts, shapes) -> None:
    """Same contraction, different summation order -- so agreement is RELATIVE, not bitwise.

    BLAS blocks its accumulation, so the result is not bit-identical to numpy's naive loop and
    cannot be: reassociating a floating-point sum changes its rounding. The bar is therefore the
    only honest one, a relative agreement at the level reassociation can move a double, and it is
    stated against the operand scale rather than against the entries -- a stiffness block's entries
    span many orders of magnitude and a per-entry relative bar would be a statement about the
    smallest one.
    """
    rng = np.random.default_rng(20260916)
    operands = [rng.random(shape) for shape in shapes]
    naive = np.einsum(subscripts, *operands)
    fast = contract(subscripts, *operands)
    assert fast.shape == naive.shape
    scale = float(np.abs(naive).max())
    assert np.abs(fast - naive).max() <= 1.0e-12 * scale


def test_the_HELPER_IS_FASTER_THAN_THE_NAIVE_PATH_BY_A_STATED_RATIO() -> None:
    """The point of the helper, pinned as a RATIO between two ways of taking the SAME contraction.

    A wall time is a property of the box; a ratio measured back-to-back on one box is a property of
    the algorithm. The floor is 5x against a measured 35.5x, so the pin survives a busy box by a
    factor of seven and still cannot pass if the helper stops reaching BLAS.
    """
    rng = np.random.default_rng(20260916)
    gradients = rng.random((400, 105, 2))
    measure = rng.random(400)

    def timed(call) -> float:
        call()  # warm the path-cache and the BLAS thread pool out of the measurement
        best = float('inf')
        for _ in range(3):
            start = time.perf_counter()
            call()
            best = min(best, time.perf_counter() - start)
        return best

    naive = timed(lambda: np.einsum('qid,qjd,q->ij', gradients, gradients, measure))
    fast = timed(lambda: contract('qid,qjd,q->ij', gradients, gradients, measure))
    assert fast > 0.0
    assert naive / fast >= 5.0, f'the helper is only {naive / fast:.1f}x the naive path, not the 35.5x measured'


def test_NO_ASSEMBLY_SITE_TAKES_THE_NAIVE_MULTI_OPERAND_EINSUM() -> None:
    """The scan that makes the helper a declaration the code must consult.

    A single-operand ``einsum`` is a transpose-and-trace and has no pairwise path to find, so it is
    left alone; two or more operands is a contraction ORDER, which is exactly the decision the
    default gets wrong.
    """
    root = _source_root()
    scanned = 0
    offenders: list[str] = []
    paths = sorted({q for pattern in _ASSEMBLY_MODULE_GLOBS for q in root.glob(pattern)})
    assert paths, 'the globs matched no module at all, so this scan reads nothing'
    for path in paths:
        relative = path.relative_to(root).as_posix()
        tree = ast.parse(path.read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            target = node.func
            name = target.attr if isinstance(target, ast.Attribute) else getattr(target, 'id', '')
            # args[0] is the subscript string, so two or more OPERANDS is len(args) >= 3.
            if name not in {'einsum', 'contract'} or len(node.args) < 3:
                continue
            # BOTH spellings count towards the floor. Counting only the offending one would let a
            # complete conversion empty the scan and pass it vacuously -- the floor exists to say
            # the scan is still reading the measured code, not that offenders remain.
            scanned += 1
            if name == 'einsum' and not any(keyword.arg == 'optimize' for keyword in node.keywords):
                offenders.append(f'{relative}:{node.lineno}')
    assert scanned >= _MINIMUM_CONTRACTIONS_SCANNED, (
        f'the scan found only {scanned} multi-operand contractions across {len(paths)} '
        f'modules, below the floor {_MINIMUM_CONTRACTIONS_SCANNED} -- it is no longer reading the '
        'code that was measured, so finding no offender proves nothing'
    )
    owing = {site.split(':')[0] for site in offenders}
    assert owing == _NAIVE_EINSUM_DEBT, (
        "the set of modules taking numpy's naive multi-operand path moved. Modules that appeared "
        f'(a NEW naive contraction where the code was clean -- call motronics.gauss.contract.contract '
        f'instead): {sorted(owing - _NAIVE_EINSUM_DEBT)}. Modules that left (converted -- delete them '
        f'from `_NAIVE_EINSUM_DEBT`, the set may only shrink): {sorted(_NAIVE_EINSUM_DEBT - owing)}. '
        f'The measured cost of the naive path is 15.9x across the assembly. Sites: {sorted(offenders)}'
    )

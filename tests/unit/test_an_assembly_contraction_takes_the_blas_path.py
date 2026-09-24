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

#: Every module on the ISOGEOMETRIC assembly path that contracts basis values against a measure.
#:
#: WIDENED 2026-09-24, and the old four-entry list is why this comment exists. It named
#: `spline.py` for the volume stiffness measured at 82.5 s -- and that assembly has since been
#: split across `spline_structural.py`, `spline_stabilise.py`, `spline_estimate.py`,
#: `spline_thermal.py`, `spline_trim_interface.py` and `spline_airgap.py`, leaving `spline.py`
#: itself with ONE contraction. The scan went on reading four modules, found 8 sites against its
#: own floor of 12, and FAILED ON THE FLOOR -- which short-circuits the offender assertion below
#: it, so the naive four-operand `np.einsum` at `gauss/quadrature.py:219` sat inside a scanned
#: module, uncaught, for as long as the list was stale. A floor that fires is not a floor that
#: protects: it stopped the scan before the scan's actual job.
_ASSEMBLY_MODULES = (
    'gauss/quadrature.py',
    'gauss/spline_basis.py',
    'hamilton/coupling/spline_transfer.py',
    'hamilton/discretizations/spline.py',
    'hamilton/discretizations/spline_airgap.py',
    'hamilton/discretizations/spline_estimate.py',
    'hamilton/discretizations/spline_interface.py',
    'hamilton/discretizations/spline_nonlinear.py',
    'hamilton/discretizations/spline_stabilise.py',
    'hamilton/discretizations/spline_structural.py',
    'hamilton/discretizations/spline_thermal.py',
    'hamilton/discretizations/spline_trim_interface.py',
)

#: NOT COVERED, BY NAME, so that "the scan is green" cannot be read as "the tree is converted".
#: The ELEMENT (FEM) assembly path carries multi-operand contractions of the same class --
#: `fem/_assemble_quad.py`, `fem/_kernels.py`, `fem/_newton.py`, `hcurl/_nonlinear.py`,
#: `physics/structural/backend.py` -- and converting them is a separate decision, not an
#: oversight: those kernels are pinned by BIT-FOR-BIT port goldens
#: (`test_solve_port_golden`, `test_newton_port_golden`), and reassociating a floating-point sum
#: is precisely what a bit-for-bit golden refuses. The measurement that justifies the conversion
#: there has to be taken on that path and against those goldens first.
_NOT_SCANNED_BY_DESIGN = (
    'hamilton/discretizations/fem/_assemble_quad.py',
    'hamilton/discretizations/fem/_kernels.py',
    'hamilton/discretizations/fem/_newton.py',
    'hamilton/discretizations/hcurl/_nonlinear.py',
    'hamilton/physics/structural/backend.py',
)

#: Below this, the scan is not looking at the code that was measured. A scan that finds NOTHING is
#: vacuous rather than green, so it carries a floor as well as a verdict. Raised 12 -> 30 with the
#: widened list, against 39 found: `RATCHET-TWO-SIDES` -- the number may only move with the list it
#: measures, and it is a floor on the SCAN's reach, not a budget for offenders.
_MINIMUM_CONTRACTIONS_SCANNED = 30


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


def test_THE_UNCOVERED_ELEMENT_PATH_IS_NAMED_AND_STILL_UNCOVERED() -> None:
    """The exclusion is two-sided, so it cannot rot in either direction.

    A named exclusion that nobody checks is the same failure as a stale module list: the file it
    names gets deleted or converted and the prose goes on claiming a gap. This asserts each
    excluded module still EXISTS, still carries at least one multi-operand contraction (so the
    gap is real and not already closed), and is not also in the scanned list (so the two
    declarations cannot contradict each other).
    """
    root = _source_root()
    for relative in _NOT_SCANNED_BY_DESIGN:
        path = root / relative
        assert path.is_file(), f'{relative} is named as an uncovered module but is not in the tree'
        assert relative not in _ASSEMBLY_MODULES, f'{relative} is declared both scanned and uncovered'
        tree = ast.parse(path.read_text(encoding='utf-8'))
        naive = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == 'einsum'
            and len(node.args) >= 3
            and not any(keyword.arg == 'optimize' for keyword in node.keywords)
        ]
        assert naive, (
            f'{relative} is named as an uncovered element-path module but carries no naive '
            'multi-operand einsum any more -- the gap is closed, so move it into _ASSEMBLY_MODULES.'
        )


def test_NO_ASSEMBLY_SITE_TAKES_THE_NAIVE_MULTI_OPERAND_EINSUM() -> None:
    """The scan that makes the helper a declaration the code must consult.

    A single-operand ``einsum`` is a transpose-and-trace and has no pairwise path to find, so it is
    left alone; two or more operands is a contraction ORDER, which is exactly the decision the
    default gets wrong.
    """
    root = _source_root()
    scanned = 0
    offenders: list[str] = []
    silent: list[str] = []
    for relative in _ASSEMBLY_MODULES:
        path = root / relative
        assert path.is_file(), f'{relative} is not in the source tree; the scan names a module that moved'
        before = scanned
        tree = ast.parse(path.read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            target = node.func
            name = (
                target.attr if isinstance(target, ast.Attribute) else target.id if isinstance(target, ast.Name) else ''
            )
            # args[0] is the subscript string, so two or more OPERANDS is len(args) >= 3.
            if name not in {'einsum', 'contract'} or len(node.args) < 3:
                continue
            # BOTH spellings count towards the floor. Counting only the offending one would let a
            # complete conversion empty the scan and pass it vacuously -- the floor exists to say
            # the scan is still reading the measured code, not that offenders remain.
            scanned += 1
            if name == 'einsum' and not any(keyword.arg == 'optimize' for keyword in node.keywords):
                offenders.append(f'{relative}:{node.lineno}')
        if scanned == before:
            silent.append(relative)
    # A NAMED SET BEATS THE COUNT, and the count alone is what let this scan go stale: a module
    # whose contractions moved out contributes nothing and the total is made up elsewhere, so the
    # list stops describing the tree without the number ever falling. `NAMED-SETS-NOT-COUNTS`.
    assert not silent, (
        f'{silent} are declared assembly modules but contribute NO multi-operand contraction, so '
        'the list no longer describes where the assembly is. Move the entry, do not keep it.'
    )
    assert scanned >= _MINIMUM_CONTRACTIONS_SCANNED, (
        f'the scan found only {scanned} multi-operand contractions across {len(_ASSEMBLY_MODULES)} '
        f'modules, below the floor {_MINIMUM_CONTRACTIONS_SCANNED} -- it is no longer reading the '
        'code that was measured, so finding no offender proves nothing'
    )
    assert not offenders, (
        "these multi-operand contractions still take numpy's naive path, which measured 15.9x "
        f'slower across the assembly: {offenders}. Call motronics.gauss.contract.contract instead.'
    )

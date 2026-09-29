"""A multi-operand contraction must reach BLAS, and every assembly site must go through the one
helper that guarantees it.

THE MEASUREMENT THAT PUT THIS HERE, taken 2026-09-16 on ``cases/field/pm/spm_slotted`` at
``SPLINE_REFINE = (2, 2)`` with ``compile_iga_magnetostatic``. Every ``np.einsum`` on the
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
the VOLUME stiffness (``iga.py``), while ``trim_nitsche_blocks`` -- the 68 cut interfaces that
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

#: Every module that assembles an operator by contracting basis values against a measure: DERIVED
#: from the two trees that hold assembly, never listed -- a hand list went stale the day a new
#: assembler (the coupled SDM subdomain) landed outside it.
_ASSEMBLY_TREES = ('gauss', 'hamilton/discretizations')


def _assembly_modules() -> list[str]:
    root = _source_root()
    return sorted(path.relative_to(root).as_posix() for tree in _ASSEMBLY_TREES for path in (root / tree).rglob('*.py'))


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
_NAIVE_EINSUM_DEBT: frozenset[str] = frozenset()

#: NOT COVERED BY THE TREE SCAN ABOVE, BY NAME, so that "the scan is green" cannot be read as "the
#: tree is converted". The element (FEM) path inside `hamilton/discretizations` IS scanned now --
#: `feat/sdm` converted it (`fix(fem): every assembly contraction takes the BLAS path`) and the
#: port goldens were re-measured with it -- so what remains outside is the structural backend,
#: which assembles in `hamilton/physics` and still carries naive multi-operand contractions.
_NOT_SCANNED_BY_DESIGN = ('hamilton/physics/structural/backend.py',)


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
        assert relative not in _assembly_modules(), f'{relative} is declared uncovered but the tree scan reads it'
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
            'multi-operand einsum any more -- the gap is closed, so widen _ASSEMBLY_TREES to cover it.'
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
    modules = _assembly_modules()
    for relative in modules:
        path = root / relative
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
    assert scanned >= _MINIMUM_CONTRACTIONS_SCANNED, (
        f'the scan found only {scanned} multi-operand contractions across {len(modules)} '
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

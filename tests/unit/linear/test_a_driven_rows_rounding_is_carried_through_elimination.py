"""A driven eliminated row's certified rounding reaches the retained row it is eliminated into.

The case_a step-1 body rows are a ~1000-sized cancellation against the conducting field rows,
which are DRIVEN (their forcing far outside their own allowance), so the slack image carried
nothing of their rounding and the anchor refused at 1.15-1.44x
(a downstream memory entry).
The image is ``|left B^-1| . bound``, checked here against exact rational arithmetic.
"""

from fractions import Fraction

import numpy as np
import scipy.sparse as sp

from numerics_lab.linear.initial_storage_border import _driven_allowance_image


def _exact_adjoint(block, row):
    """Solve ``B^T y = row`` by rational Gauss-Jordan."""
    size = block.shape[0]
    augmented = [[Fraction(float(block[j, i])) for j in range(size)] + [Fraction(float(row[i]))] for i in range(size)]
    for column in range(size):
        pivot = next(r for r in range(column, size) if augmented[r][column] != 0)
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        for r in range(size):
            if r != column and augmented[r][column] != 0:
                factor = augmented[r][column] / augmented[column][column]
                augmented[r] = [x - factor * y for x, y in zip(augmented[r], augmented[column], strict=True)]
    return [augmented[i][size] / augmented[i][i] for i in range(size)]


def test_the_image_is_the_exact_adjoint_weighting_of_the_driven_bounds() -> None:
    block = sp.csr_matrix(np.array([[4.0, 1.0, 0.0], [1e3, 3e3, 1e3], [0.0, 2.0, 5.0]]))
    left = sp.csr_matrix(np.array([[1.0, -1.0, 1.0], [0.0, 0.0, 0.0]]))
    bound = np.array([3e-13, 0.0, 7e-12])
    image = _driven_allowance_image(block, left, bound, np.array([True, True]))
    exact = sum(
        abs(y) * Fraction(float(b))
        for y, b in zip(_exact_adjoint(block.toarray(), [1.0, -1.0, 1.0]), bound, strict=True)
    )
    assert image[1] == 0.0, 'a row with no elimination support inherits nothing'
    np.testing.assert_allclose(image[0], float(exact), rtol=1e-12)


def test_a_row_that_does_not_need_it_and_a_singular_elimination_inherit_nothing() -> None:
    left = sp.csr_matrix(np.array([[1.0, 1.0]]))
    bound = np.array([1e-12, 1e-12])
    regular = sp.csr_matrix(np.array([[2.0, 0.0], [0.0, 3.0]]))
    assert _driven_allowance_image(regular, left, bound, np.array([False]))[0] == 0.0
    # PLANTED: an elimination whose adjoint does not exist cannot grant an allowance.
    singular = sp.csr_matrix(np.array([[1.0, 1.0], [1.0, 1.0]]))
    assert _driven_allowance_image(singular, left, bound, np.array([True]))[0] == 0.0

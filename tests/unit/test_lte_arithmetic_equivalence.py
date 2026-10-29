"""LTE hot-path changes preserve the recorded floating-point arithmetic exactly."""

import math

import numpy as np

from numerics_lab.time_integrators.march import StepTolerance, _lte_ratio


def test_lte_ratio_is_bit_identical_on_irregular_per_variable_histories():
    rng = np.random.default_rng(271828)
    for order in (1, 2, 3, 4):
        for _ in range(40):
            times = np.cumsum(rng.uniform(1e-7, 1e-4, order + 2))
            states = rng.normal(size=(order + 2, 7)) * np.logspace(-6, 6, 7)
            history = list(zip(times[:-1], states[:-1], strict=True))
            tolerance = StepTolerance(rtol=np.logspace(-10, -4, 7), atol=np.logspace(-14, -8, 7))
            pred = np.zeros_like(states[-1])
            for j, (tj, uj) in enumerate(history):
                weight = 1.0
                for m, tm in enumerate(times[:-1]):
                    if m != j:
                        weight *= (times[-1] - tm) / (tj - tm)
                pred = pred + weight * uj
            span = float(np.prod([times[-1] - t for t in times[:-1]]))
            h = times[-1] - times[-2]
            lte = 0.25 * math.factorial(order + 1) * h ** (order + 1) / span * (states[-1] - pred)
            scale = tolerance.atol + tolerance.rtol * np.maximum(np.abs(states[-2]), np.abs(states[-1]))
            expected = float(np.max(np.abs(lte) / scale))
            assert _lte_ratio(history, times[-1], states[-1], order, 0.25, tolerance).hex() == expected.hex()

"""The exact zero-order-hold march of a linear plant under sampled feedback is gauss's, once.

:func:`~numerics_lab.time_integrators.zoh.march_zoh` closes each held-input interval in closed
form, so a scalar decay ``dx/dt = -a x + b u`` with a constant held ``u`` lands on the analytic
solution at every sample to roundoff, and a feedback law reads the state AT the sample it acts on.
"""

from __future__ import annotations

import numpy as np

from numerics_lab.time_integrators.zoh import march_zoh, zoh_discretize


def test_a_held_constant_input_lands_on_the_analytic_step_response() -> None:
    a, b, u, dt = 3.0, 2.0, 5.0, 0.1
    ad, (bd,) = zoh_discretize(np.array([[-a]]), (np.array([b]),), dt)
    times, states = march_zoh(ad, (bd,), np.array([0.0]), t0=0.0, dt=dt, n_steps=20, control=lambda _t, _x: (u,))
    np.testing.assert_allclose(times, dt * np.arange(21), rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(states[:, 0], b * u / a * (1.0 - np.exp(-a * times)), rtol=1e-12, atol=1e-14)


def test_the_feedback_law_reads_the_state_it_acts_on_including_the_last_sample() -> None:
    seen: list[tuple[float, float]] = []

    def control(t: float, x: np.ndarray) -> tuple[float]:
        seen.append((t, float(x[0])))
        return (-x[0],)

    ad, columns = zoh_discretize(np.array([[-1.0]]), (np.array([1.0]),), 0.5)
    times, states = march_zoh(ad, columns, np.array([1.0]), t0=1.0, dt=0.5, n_steps=4, control=control)
    assert [t for t, _ in seen] == list(times)
    assert [x for _, x in seen] == list(states[:, 0])

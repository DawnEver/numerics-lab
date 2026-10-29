"""A failed event landing must never advance a physical controller."""

from importlib import import_module

import numpy as np
import pytest
from tests.unit._helpers import _RampWithToggle

from numerics_lab.formulation import IntegratorKind, TimeWindow
from numerics_lab.time_integrators import _step as _step_mod


@pytest.mark.parametrize('adaptive', [False, True])
def test_failed_landing_cannot_call_the_physical_handler(monkeypatch, adaptive):
    events = import_module('numerics_lab.time_integrators.march')
    original = _step_mod._solve_step
    calls = []

    def solve_step(*args, **kwargs):
        result = original(*args, **kwargs)
        calls.append(result)
        # First full step is a converged crossing probe. The locator is pinned
        # to the known clock below, so the second solve is the actual landing.
        if len(calls) == 2:
            u, _ok, iterations, residual, floor = result
            return u, False, iterations, residual, floor
        return result

    monkeypatch.setattr(_step_mod, '_solve_step', solve_step)
    monkeypatch.setattr(events, '_locate_bracket', lambda *_args, **_kwargs: (0.5, 0.5))
    physical_samples = []
    problem = _RampWithToggle(0.5)
    study = TimeWindow(t0=0.0, t1=1.0, dt=1.0, integrator=IntegratorKind.BDF1)
    with pytest.raises(RuntimeError, match='event landing.*did not converge'):
        events.integrate_transient(
            problem,
            study,
            np.array([0.0]),
            event_fns=(lambda _u, t: t - 0.5,),
            on_event=lambda _index, t, u: physical_samples.append((t, u.copy())),
            adaptive=adaptive,
        )
    assert len(calls) == 2
    assert physical_samples == []

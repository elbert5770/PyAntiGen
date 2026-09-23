"""Every simulation starts from the same parameters, whatever the last one did.

RoadRunner's reset() does not undo a global parameter an event assigned.
An arm whose events switch a rate on and leave it on would otherwise start
every run after the first with that rate already on, and the objective would
depend on evaluation history. See Optimize.restore_parameter_baseline.

Run from the repository root: python -m pytest tests/engine
"""
import numpy as np
import pytest

te = pytest.importorskip("tellurium")

from pyantigen.engine.Optimize import (remember_parameter_baseline,  # noqa: E402
                                       restore_parameter_baseline, run_all,
                                       set_parameters_from_dict)

# A decays at k_on, which an event switches on at t = 5 and nothing switches
# off. "gain" is fitted; "twice" is an assignment rule and cannot be set.
MODEL = """
model m
  species A = 1
  A' = gain - k_on*A
  gain = 0.1
  k_on = 0
  twice := 2*gain
  at (time >= 5): k_on = 0.5
end
"""


def _replicate():
    return {
        "Label": "arm",
        "Update_parameters": lambda r, rep: None,
        "Observed_species": lambda r: ["time", "[A]"],
        "Solver_settings": lambda rep: {
            "integrator": "cvode", "absolute_tolerance": 1e-10,
            "relative_tolerance": 1e-10, "maximum_num_steps": 20000,
            "stiff": True, "variable_step_size": True,
            "simulation_blocks": {"b": {"start": 0, "end": 10, "n_points": 101,
                                        "abs_tol": 1e-12, "rel_tol": 1e-12}}},
    }


def _run(r, gain):
    res = run_all(r, "arm", _replicate(), {}, set_parameters=set_parameters_from_dict,
                  parameters={"gain": gain})
    return np.asarray(res["arm"]["results"]["[A]"], dtype=float)


def test_reset_alone_keeps_the_event_assignment():
    """The RoadRunner behaviour the fix exists for; if this ever fails, the
    fix has become unnecessary rather than wrong."""
    r = te.loada(MODEL)
    r.simulate(0, 10, 11)
    r.reset()
    assert r["k_on"] == 0.5


def test_repeated_runs_at_one_point_are_identical():
    r = te.loada(MODEL)
    remember_parameter_baseline(r)
    first = _run(r, 0.1)
    for _ in range(3):
        assert np.array_equal(_run(r, 0.1), first)
    assert first[20] > first[-1]          # the event did fire in each run


def test_the_objective_is_history_free():
    """Run B after A equals B run first on a fresh model."""
    r = te.loada(MODEL)
    remember_parameter_baseline(r)
    _run(r, 0.3)
    after_a = _run(r, 0.1)
    fresh = te.loada(MODEL)
    remember_parameter_baseline(fresh)
    assert np.array_equal(after_a, _run(fresh, 0.1))


def test_restore_touches_only_what_changed_and_skips_rules():
    r = te.loada(MODEL)
    remember_parameter_baseline(r)
    r.simulate(0, 10, 11)
    r.reset()
    assert restore_parameter_baseline(r) == ["k_on"]
    assert r["k_on"] == 0.0
    assert restore_parameter_baseline(r) == []


def test_a_model_run_before_being_recorded_is_recorded_on_first_use():
    r = te.loada(MODEL)
    assert not hasattr(r, "_pyantigen_param_baseline")
    first = _run(r, 0.1)
    assert hasattr(r, "_pyantigen_param_baseline")
    assert np.array_equal(_run(r, 0.1), first)

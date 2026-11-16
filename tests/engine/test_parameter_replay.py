"""A fitted parameter vector is applied to a model the way the fit applies it (pyantigen.engine.Parameter_replay).

The prediction tools give a stored vector (``{name: linear value}``) to a RoadRunner model through ``Model_simulate.apply_member``. It must mean what it
meant in the fit: ``Update_parameters`` first, the values over it, then the replicate's hooks; a name that only a hook understands is legitimate; a
name that changes nothing is an error, because the fit's own setter ignores unknown names and a prediction would silently equal the optimum.
"""
import subprocess
import sys

import numpy as np
import pytest

tellurium = pytest.importorskip("tellurium")

from pyantigen.engine.Parameter_replay import ParameterReplay, unreached  # noqa: E402
from pyantigen.engine.Model_simulate import apply_member  # noqa: E402

MODEL = """
model m
  A = 1; B = 0
  k = 0.5; c = 1.0; kr = 2.0; kf = 1.0
  J1: A -> B; k*A
  J2: B -> A; kf*B
end
"""


def _model():
    r = tellurium.loada(MODEL)
    from pyantigen.engine.Optimize import remember_parameter_baseline
    remember_parameter_baseline(r)
    return r


def _replicate(calls=None):
    def update(r, rep):
        if calls is not None:
            calls.append("update")
        r["c"] = 3.0                         # what the project sets at build time; a fitted value must not undo it unless it is c itself

    def hook(r, params):
        if calls is not None:
            calls.append("hook")
        if "Kd" in params:                   # a fitted dissociation constant that the model only knows as a rate constant
            r["kf"] = r["kr"] / params["Kd"]

    return {"Label": "x", "Update_parameters": update, "parameter_hooks": [hook]}


def test_the_values_go_in_after_update_parameters_and_the_hooks_run_last():
    r, calls = _model(), []
    rep = _replicate(calls)
    ParameterReplay(r, rep, ["k"], []).apply([2.0])
    assert calls == ["update", "hook"]
    assert r["k"] == 2.0 and r["c"] == 3.0          # the fitted value over the model's, Update_parameters' own settings kept


def test_a_name_only_a_hook_reads_is_reached_and_a_misspelt_one_is_not():
    r = _model()
    replay = ParameterReplay(r, _replicate(), ["k", "Kd", "kk"], [])
    assert unreached(replay, [0.5, 4.0, 1.0]) == ["kk"]      # k is a model parameter, Kd moves kf through the hook, kk does nothing


def test_the_hooked_name_changes_the_model():
    r = _model()
    ParameterReplay(r, _replicate(), ["Kd"], []).apply([4.0])
    assert r["kf"] == pytest.approx(0.5)               # kr / Kd = 2 / 4


def test_apply_member_refuses_a_name_that_changes_nothing_and_accepts_a_good_one():
    r, rep = _model(), _replicate()
    with pytest.raises(ValueError, match="kk"):
        apply_member(r, rep, {"k": 1.0, "kk": 2.0})
    r2, rep2 = _model(), _replicate()
    apply_member(r2, rep2, {"k": 1.5, "Kd": 8.0})
    assert r2["k"] == 1.5 and r2["kf"] == pytest.approx(0.25)


def test_a_member_changes_the_simulated_curve_relative_to_the_optimum():
    r, rep = _model(), _replicate()
    apply_member(r, rep, {"k": 0.5})
    base = np.asarray(r.simulate(0, 4, 5, selections=["time", "B"]))[:, 1]
    apply_member(r, rep, {"k": 1.0})
    moved = np.asarray(r.simulate(0, 4, 5, selections=["time", "B"]))[:, 1]
    assert np.max(np.abs(moved - base)) > 1e-3


def test_the_old_import_path_still_works_and_the_module_needs_no_amici_or_fcntl():
    from pyantigen.engine.Analytic_gradient import ParameterReplay as old
    assert old is ParameterReplay
    code = "import sys; import pyantigen.engine.Parameter_replay; bad = [m for m in ('amici', 'fcntl') if m in sys.modules]; sys.exit(1 if bad else 0)"
    import os
    import pyantigen
    root = os.path.dirname(os.path.dirname(os.path.abspath(pyantigen.__file__)))
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=root, env={**os.environ, "PYTHONPATH": root})
    assert out.returncode == 0, out.stderr

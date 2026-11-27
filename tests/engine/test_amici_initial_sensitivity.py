"""A fitted parameter that feeds a species' initial amount is differentiated (pyantigen.engine.Scaled_amici).

``AmiciSegmentRunner`` used to take the initial state sensitivities from ``model.get_initial_state_sensitivities()``, which in AMICI 1.1.0
puts the entries of every listed parameter in the FIRST parameter's slot once more than one is listed. A receptor number
``R2 = a * b`` then had sensitivity zero (or another parameter's) with respect to ``a`` and ``b``, and the gradient of a fit that moved them
was wrong without any error (Schwen_PONE2014 in Petab2PyAntiGen: ini_R1, ini_R2fold). The runner now reads them from a run that stops at t0.

Run from the repository root: python -m pytest tests/engine/test_amici_initial_sensitivity.py
Skipped unless AMICI >= 1.1 and tellurium import (on Windows: under WSL).
"""
import numpy as np
import pytest

pytest.importorskip("amici")
te = pytest.importorskip("tellurium")

from pyantigen.engine.Scaled_amici import AmiciSegmentRunner, compile_model  # noqa: E402
from pyantigen.generate.input_schedule import InputSchedule  # noqa: E402
from pyantigen.generate.scaled_time import scale_model  # noqa: E402

MODEL = """
model initial_amounts
  compartment C = 1
  substanceOnly species R1 in C, R2 in C
  a = 3; b = 2; k1 = 0.4; k2 = 0.7
  R1 = a
  R2 = a * b
  R1 -> ; k1 * R1
  R2 -> ; k2 * R2
end
"""
THETA = {"a": 3.0, "b": 2.0, "k1": 0.4, "k2": 0.7}
TIMES = [0.5, 1.0, 2.0]


@pytest.fixture(scope="module")
def runner(tmp_path_factory):
    sched = InputSchedule(THETA)
    scaled = scale_model(te.antimonyToSBML(MODEL), sched)
    module = compile_model(scaled, "initial_amounts_scaled", tmp_path_factory.mktemp("amici_initial"))
    return AmiciSegmentRunner(scaled, sched, module, states=["R1", "R2"])


@pytest.mark.parametrize("wrt", [["a", "b"], ["k1", "a", "b"], ["b", "k2", "a", "k1"]])
def test_initial_amounts_have_their_sensitivities(runner, wrt):
    """d R1/d a = exp(-k1 t), d R2/d a = b exp(-k2 t), d R2/d b = a exp(-k2 t), and the rate constants' own, whatever the order of ``wrt``."""
    x, s = runner.run(THETA, 0, 2.0, TIMES, wrt=wrt)
    t = np.array(TIMES)
    a, b, k1, k2 = THETA["a"], THETA["b"], THETA["k1"], THETA["k2"]
    e1, e2 = np.exp(-k1 * t), np.exp(-k2 * t)
    np.testing.assert_allclose(x[:, 0], a * e1, rtol=1e-7)
    np.testing.assert_allclose(x[:, 1], a * b * e2, rtol=1e-7)
    want = {"a": (e1, b * e2), "b": (0 * t, a * e2), "k1": (-t * a * e1, 0 * t), "k2": (0 * t, -t * a * b * e2)}
    for j, name in enumerate(wrt):
        np.testing.assert_allclose(s[:, 0, j], want[name][0], rtol=1e-6, atol=1e-9, err_msg=f"dR1/d{name}")
        np.testing.assert_allclose(s[:, 1, j], want[name][1], rtol=1e-6, atol=1e-9, err_msg=f"dR2/d{name}")

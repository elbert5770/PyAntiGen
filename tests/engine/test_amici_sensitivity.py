"""AMICI forward sensitivities through the schedule, with no events (pyantigen.engine.Scaled_amici).

Run from the repository root: python -m pytest tests/engine/test_amici_sensitivity.py
Skipped unless AMICI >= 1.1 and tellurium import. On Windows run it under WSL:
there is no AMICI wheel and no compiler there.

The model is compiled once, scaled in time (``scale_model``). Changing Q_CSF
changes only the numbers handed to AMICI: the segment constants, the input
states and their sensitivities. The results are compared with the scipy
reference of ``test_segment_sensitivity.py`` (itself checked against finite
differences, an independent original-time formulation and the RoadRunner
segment runner).
"""
import importlib.util
import os

import numpy as np
import pytest

pytest.importorskip("amici")
pytest.importorskip("tellurium")

from pyantigen.engine.Scaled_amici import AmiciSegmentRunner, compile_model  # noqa: E402
from pyantigen.generate.input_schedule import ScheduleError  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ref = _load("test_segment_sensitivity.py", "_sens_reference")
gen = _load(os.path.join("..", "generate", "test_gen_scaled_time.py"), "_scaled_gen")


@pytest.fixture(scope="module")
def runner(tmp_path_factory):
    sched, scaled = gen.scaled(gen.NOMINAL)
    module = compile_model(scaled, "toy_scaled", tmp_path_factory.mktemp("amici_toy"))
    return AmiciSegmentRunner(scaled, sched, module, states=ref.STATES)


def close(a, b, rtol):
    scale = float(np.max(np.abs(b))) + 1e-12
    np.testing.assert_allclose(a, b, rtol=rtol, atol=rtol * scale)


@pytest.mark.parametrize("q", [17.39500698, 12.0, 25.0, 9.0, 40.0])
def test_amici_matches_the_reference_on_one_compiled_model(runner, q):
    theta = ref.at(q)
    x, s = runner.run(theta, 0, 8, ref.OBS, wrt=ref.THETA)
    np.testing.assert_allclose(x, ref.forward(ref.schedule(), theta), rtol=1e-7, atol=1e-9)
    close(s, ref.sens_scaled(ref.schedule(), theta), rtol=1e-6)


def test_amici_matches_finite_differences_of_its_own_states(runner):
    theta = ref.at(17.39500698)
    _x, s = runner.run(theta, 0, 8, ref.OBS, wrt=ref.THETA)
    fd = ref.richardson_fd(lambda th: runner.run(th, 0, 8, ref.OBS, wrt=[])[0], theta)
    close(s, fd, rtol=1e-6)


def test_the_moving_switches_are_in_the_gradient(runner):
    theta = ref.at(17.39500698)
    _x, s = runner.run(theta, 0, 8, ref.OBS, wrt=ref.THETA)
    assert np.abs(s[..., ref.THETA.index("V_LP")]).max() > 0.1
    assert np.abs(s[..., ref.THETA.index("Q_CSF")]).max() > 0.1


def test_infeasible_parameters_are_refused_before_integration(runner):
    with pytest.raises(ScheduleError):
        runner.run(ref.at(5.0), 0, 8, ref.OBS, wrt=ref.THETA)


def test_a_parameter_the_model_does_not_have_is_refused(runner):
    with pytest.raises(KeyError):
        runner.run(ref.at(17.39500698), 0, 8, ref.OBS, wrt=["Q_CSF", "not_a_parameter"])


def test_the_pre_history_before_the_first_switch_is_differentiated_too(runner):
    # from t = -5 the inputs hold the model's initial values (Q_SN = Q_CSF*f_SN), so the
    # state at t = 0 depends on Q_CSF and f_SN before any switch fires
    theta = ref.at(17.39500698)
    obs = [0.3, 2.2, 8.0]
    _x, s = runner.run(theta, -5, 8, obs, wrt=ref.THETA)
    fd = ref.richardson_fd(lambda th: runner.run(th, -5, 8, obs, wrt=[])[0], theta)
    close(s, fd, rtol=1e-6)
    # ... and that dependence is really there: starting at 0 instead gives a different gradient
    _x0, s0 = runner.run(theta, 0, 8, obs, wrt=ref.THETA)
    assert np.abs(s - s0).max() > 1e-3 * np.abs(s).max()

"""AMICI forward sensitivities through boluses (steps that assign a state), no events.

Run from the repository root under WSL: python -m pytest tests/engine/test_amici_jumps.py
Skipped unless AMICI >= 1.1 and tellurium import. The reference is the scipy
integration in ``test_segment_jumps.py`` (finite differences, an independent
original-time formulation, and the RoadRunner segment runner all agree with it).
"""
import importlib.util
import os

import numpy as np
import pytest

pytest.importorskip("amici")
pytest.importorskip("tellurium")

from pyantigen.engine.Scaled_amici import AmiciSegmentRunner, compile_model  # noqa: E402
from pyantigen.generate.input_schedule import InputSchedule  # noqa: E402
from pyantigen.generate.scaled_time import scale_model  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("_jumps_reference", os.path.join(HERE, "test_segment_jumps.py"))
ref = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ref)
te = ref.te


def build(sched):
    return scale_model(te.antimonyToSBML(ref.MODEL.format(**ref.NOMINAL) + sched.to_antimony(events=False)), sched)


@pytest.fixture(scope="module")
def runner(tmp_path_factory):
    sched = ref.schedule()
    scaled = build(sched)
    module = compile_model(scaled, "bolus_scaled", tmp_path_factory.mktemp("amici_bolus"))
    return AmiciSegmentRunner(scaled, sched, module, states=ref.STATES)


@pytest.mark.parametrize("variant", [{}, {"R_inf": 3.2}, {"R_inf": 4.5, "frac": 0.7}, {"k_a": 0.5, "D_tot": 7.0}])
def test_amici_matches_the_reference_with_boluses(runner, variant):
    theta = ref.at(**variant)
    x, s = runner.run(theta, 0, 14, ref.OBS, wrt=ref.THETA)
    np.testing.assert_allclose(x, ref.forward(ref.schedule(), theta), rtol=1e-7, atol=1e-9)
    ref.close(s, ref.sens_scaled(ref.schedule(), theta), rtol=1e-6)


def test_amici_matches_finite_differences_of_its_own_states(runner):
    theta = ref.at()
    _x, s = runner.run(theta, 0, 14, ref.OBS, wrt=ref.THETA)
    fd = ref.richardson_fd(lambda th: runner.run(th, 0, 14, ref.OBS, wrt=[])[0], theta)
    ref.close(s, fd, rtol=1e-6)


def test_a_schedule_read_from_event_text_gives_the_same_gradient(tmp_path):
    theta = ref.at()
    parsed = InputSchedule.from_antimony_events(ref.schedule().to_antimony(events=True), theta)
    scaled = build(parsed)
    runner = AmiciSegmentRunner(scaled, parsed, compile_model(scaled, "bolus_parsed", tmp_path), states=ref.STATES)
    x, s = runner.run(theta, 0, 14, ref.OBS, wrt=ref.THETA)
    ref.close(s, ref.sens_scaled(ref.schedule(), theta), rtol=1e-6)
    np.testing.assert_allclose(x, ref.forward(ref.schedule(), theta), rtol=1e-7, atol=1e-9)

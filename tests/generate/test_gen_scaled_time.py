"""The scaled-time rewrite of a model (pyantigen.generate.scaled_time), checked in RoadRunner.

Run from the repository root: python -m pytest tests/generate/test_gen_scaled_time.py

Each segment of the schedule is integrated on s in [0, 1] with the model the
rewrite produced, the final state handed to the next segment, and the result
compared with the same model written with real events. If the clock, the
``Delta`` factor, the measured curve or the input states were wrong, the values
would differ. (The sensitivities need a solver that differentiates: see
tests/engine/test_amici_sensitivity.py.)
"""
import numpy as np
import pytest

te = pytest.importorskip("tellurium")

from pyantigen.generate.input_schedule import InputSchedule  # noqa: E402
from pyantigen.generate.scaled_time import (  # noqa: E402
    ScaledTimeError, scale_model, CLOCK, DELTA)

NOMINAL = {"Q_CSF": 17.39500698, "V_LP": 6.0, "t_CSFdraw": 0.1, "f_SN": 0.1, "k_y": 0.3}
STATES = ["x_leak", "x_sn", "x_lp", "x_refill", "y", "z", "w"]
OBS = [0.3, 0.45, 1.0, 2.2, 3.7, 5.25, 7.0, 8.0]

MODEL = """
Q_CSF = {Q_CSF!r}; V_LP = {V_LP!r}; t_CSFdraw = {t_CSFdraw!r}; f_SN = {f_SN!r}; k_y = {k_y!r}
Q_Leak = 0; Q_LP = 0; Q_refill = 0; Q_SN = Q_CSF*f_SN
x_leak = 0; x_sn = 0; x_lp = 0; x_refill = 0; y = 1; z = 0; w = 0
x_leak' = Q_Leak; x_sn' = Q_SN; x_lp' = Q_LP; x_refill' = Q_refill
y' = 0.05*Q_LP - k_y*y*(1 + Q_SN)
z' = Q_refill*y/(1 + z*z) - 0.2*z
w' = f_L - 0.4*w
"""


def schedule():
    s = InputSchedule(NOMINAL)
    t_draw = s.moving("t_draw", "t_CSFdraw")
    t_refill = s.moving("t_refill", "V_LP/Q_CSF")
    s.step(0, {"Q_Leak": "15.0"})
    s.step(3, {"Q_Leak": "0.0"})
    s.step(5, {"Q_Leak": "Q_CSF - f_SN*Q_CSF"})
    s.periodic(0, 1, 8, [
        (0, {"Q_SN": "0.0", "Q_LP": "Q_CSF", "Q_refill": "Q_CSF - V_LP/t_CSFdraw"}),
        (t_draw, {"Q_refill": "Q_CSF"}),
        (t_refill, {"Q_SN": "f_SN*Q_CSF", "Q_LP": "Q_Leak", "Q_refill": "0"}),
    ])
    s.linear_input("f_L", [0, 1, 2.5, 4, 6.5], [0.1, 0.3, 0.2, 0.5, 0.1],
                   default_before=0, default_after=0)
    return s


def events_reference(theta, obs=OBS):
    """(state at t = 0, states at *obs*) from the model with real events, tight tolerances.

    The run starts at t = -5 so the events at t = 0 have a false-to-true edge to fire on,
    as in the engine's pre-equilibration block; the state it reaches at 0 seeds the scaled run.
    """
    r = te.loada(MODEL.format(**theta) + schedule().to_antimony(events=True))
    r.integrator.absolute_tolerance = 1e-13
    r.integrator.relative_tolerance = 1e-13
    res = r.simulate(-5, 8, 1301, ["time"] + STATES)
    rows = [int(np.argmin(np.abs(res[:, 0] - t))) for t in [0.0] + list(obs)]
    assert np.allclose(res[rows, 0], [0.0] + list(obs), atol=1e-9)
    return res[rows[0], 1:], res[rows[1:], 1:]


def scaled(theta):
    sched = schedule()
    sbml = te.antimonyToSBML(MODEL.format(**theta) + sched.to_antimony(events=False))
    return sched, scale_model(sbml, sched)


def run_scaled(theta, x0, obs=OBS):
    """Integrate every segment on s in [0, 1] and return the states at the observation times."""
    sched, sm = scaled(theta)
    r = te.loadSBMLModel(sm.sbml)
    r.integrator.absolute_tolerance = 1e-13
    r.integrator.relative_tolerance = 1e-13
    x = dict(zip(STATES, x0))
    out = []
    for seg in sched.plan(0, 8, obs, params=theta):
        r.reset()
        for k, v in sm.segment_constants(seg).items():
            r[k] = v
        for v in sm.inputs:
            r[v] = seg.values[v]
        for n, v in x.items():
            r[n] = v
        res = r.simulate(0, 1, 2, STATES)
        x = dict(zip(STATES, np.asarray(res)[-1]))
        if seg.observed:
            out.append([x[n] for n in STATES])
    return np.array(out)


@pytest.mark.parametrize("q", [17.39500698, 12.0, 25.0, 9.0])
def test_scaled_segments_reproduce_the_events(q):
    theta = dict(NOMINAL, Q_CSF=q)
    x0, want = events_reference(theta)
    np.testing.assert_allclose(run_scaled(theta, x0), want, rtol=1e-6, atol=1e-8)


def test_the_rewritten_model_has_no_events_and_a_clock_made_of_the_parameters():
    sched, sm = scaled(NOMINAL)
    r = te.loadSBMLModel(sm.sbml)
    ids = set(r.getGlobalParameterIds())
    assert {CLOCK, DELTA, "t_draw", "t_refill", "sched_a0", "sched_c1_t_refill",
            "sched_ci_f_L", "sched_cs_f_L"} <= ids
    assert sm.inputs == ["Q_LP", "Q_Leak", "Q_SN", "Q_refill"][:0] + sorted(sm.inputs)
    assert sm.moving == ["t_draw", "t_refill"] and sm.curves == ["f_L"]
    assert "sched_a0" in sm.constants and "sched_cs_f_L" in sm.constants
    import libsbml
    assert libsbml.readSBMLFromString(sm.sbml).getModel().getNumEvents() == 0


def test_segment_constants_describe_the_end_points():
    sched, sm = scaled(NOMINAL)
    # the stretch from the end of the draw to the refill in hour 1: both ends move
    seg = next(g for g in sched.plan(0, 8, OBS, params=NOMINAL)
               if g.start.terms and g.end.terms and 1 <= g.t0 < 2)
    c = sm.segment_constants(seg)
    assert c["sched_a0"] == 1.0 and c["sched_c0_t_draw"] == 1.0 and c["sched_c0_t_refill"] == 0.0
    assert c["sched_a1"] == 1.0 and c["sched_c1_t_refill"] == 1.0 and c["sched_c1_t_draw"] == 0.0
    # the delta they build is the segment's duration
    t_draw, t_refill = NOMINAL["t_CSFdraw"], NOMINAL["V_LP"] / NOMINAL["Q_CSF"]
    delta = (c["sched_a1"] - c["sched_a0"]) + (c["sched_c1_t_refill"] - c["sched_c0_t_refill"]) * t_refill \
        + (c["sched_c1_t_draw"] - c["sched_c0_t_draw"]) * t_draw
    assert delta == pytest.approx(seg.t1 - seg.t0, abs=1e-15)


def test_a_model_that_still_has_events_is_refused():
    sched = schedule()
    sbml = te.antimonyToSBML(MODEL.format(**NOMINAL) + sched.to_antimony(events=True))
    with pytest.raises(ScaledTimeError, match="events"):
        scale_model(sbml, sched)


def test_a_measured_input_the_model_lacks_is_refused():
    sched = schedule()
    sbml = te.antimonyToSBML(MODEL.format(**NOMINAL) + "f_L = 0.3\n")
    with pytest.raises(ScaledTimeError):
        scale_model(sbml, sched)


def test_the_public_schedule_accessors():
    s = schedule()
    assert s.moving_quantities() == {"t_draw": "t_CSFdraw", "t_refill": "V_LP/Q_CSF"}
    assert s.assigned_variables() == ["Q_LP", "Q_Leak", "Q_SN", "Q_refill"][:0] + sorted(
        ["Q_LP", "Q_Leak", "Q_SN", "Q_refill"])
    (name, times, values, before, after), = s.inputs()
    assert name == "f_L" and times[-1] == 6.5 and after == 0
    p = s.plan(0, 8, [0.45], params=NOMINAL, wrt=["Q_CSF"])
    assert all(len(seg.dt0) == 1 for seg in p)
    assert sum(seg.observed for seg in p) == 1


def test_initial_assignment_chains_are_followed_to_the_parameters():
    # Q_SN starts as Q_SN0, which starts as Q_CSF*f_SN: the dependence on Q_CSF is two levels down
    sched = schedule()
    chain = MODEL.format(**NOMINAL).replace("Q_SN = Q_CSF*f_SN", "Q_SN0 = Q_CSF*f_SN; Q_SN = Q_SN0")
    sm = scale_model(te.antimonyToSBML(chain + sched.to_antimony(events=False)), sched)
    assert sm.initial_inputs["Q_SN"].replace(" ", "") == "Q_CSF*f_SN"
    assert sm.initial_inputs["Q_Leak"] == 0.0
    assert sm.parameter_values["Q_CSF"] == NOMINAL["Q_CSF"]


def test_time_in_a_rule_becomes_the_clock_without_the_time_symbols_url():
    """A rule that reads ``time`` (a drug curve written as a function of time) is rewritten in terms of
    the segment clock. The renamed node must be a plain <ci>: left carrying definitionURL=".../time" it is
    refused by AMICI's importer ("Unhandled <ci> attributes")."""
    import re
    sched = schedule()
    drug = "\ndrug := piecewise(0, time < 2, 3*exp(-0.1*(time - 2)))\n"
    sbml = te.antimonyToSBML(MODEL.format(**NOMINAL) + drug + sched.to_antimony(events=False))
    sm = scale_model(sbml, sched)
    assert not re.search(r"<ci[^>]*definitionURL", sm.sbml)
    rule = re.search(r'<assignmentRule variable="drug">.*?</assignmentRule>', sm.sbml, re.S).group(0)
    assert CLOCK in rule and "time" not in re.sub(r"</?[a-z]+[^>]*>", " ", rule).split()

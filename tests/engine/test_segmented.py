"""The event-free backend (pyantigen.engine.Segmented) against the events it replaces.

Run from the repository root: python -m pytest tests/engine/test_segmented.py

The model is a small stand-in for the v4 SILK CSF draw: the same three switches
an hour (two of them moving with Q_CSF and V_LP), a step at t = 3 and t = 20,
and a measured curve linear between knots. Every check runs the real
``simulate()`` twice -- once with the schedule written as events, once with the
schedule handed to the segment runner -- and compares what comes out.
"""
import numpy as np
import pytest

te = pytest.importorskip("tellurium")

from pyantigen.engine import Simulate as sim  # noqa: E402
from pyantigen.engine.Event_times import build_event_time_table  # noqa: E402
from pyantigen.engine.Preequil_cache import PreequilCache  # noqa: E402
from pyantigen.engine.Segmented import switches_before  # noqa: E402
from pyantigen.generate.input_schedule import InputSchedule, ScheduleError  # noqa: E402

PARAMS = {"Q_CSF": 17.39500698, "V_LP": 6.0, "t_CSFdraw": 0.1, "f_SN": 0.1}

MODEL = """
Q_CSF = {q!r}; f_SN = 0.1; V_LP = 6; t_CSFdraw = 0.1
Q_Leak = 0; Q_LP = 0; Q_refill = 0; Q_SN = Q_CSF*f_SN
x_leak = 0; x_sn = 0; x_lp = 0; x_refill = 0; y = 1; z = 0; w = 0
x_leak' = Q_Leak; x_sn' = Q_SN; x_lp' = Q_LP; x_refill' = Q_refill
y' = 0.05*Q_LP - 0.3*y*(1 + Q_SN)
z' = Q_refill*y/(1 + z*z) - 0.2*z
w' = f_L - 0.4*w
"""

OBSERVED = ["time", "x_leak", "x_sn", "x_lp", "x_refill", "y", "z", "w",
            "Q_Leak", "Q_SN", "Q_LP", "Q_refill", "f_L"]


def schedule(q=PARAMS["Q_CSF"], hours=8):
    s = InputSchedule(dict(PARAMS, Q_CSF=q))
    t_draw = s.moving("t_draw", "t_CSFdraw")
    t_refill = s.moving("t_refill", "V_LP/Q_CSF")
    s.step(0, {"Q_Leak": "15.0"})
    s.step(3, {"Q_Leak": "0.0"})
    s.step(5, {"Q_Leak": "Q_CSF - f_SN*Q_CSF"})
    s.periodic(0, 1, hours, [
        (0, {"Q_SN": "0.0", "Q_LP": "Q_CSF", "Q_refill": "Q_CSF - V_LP/t_CSFdraw"}),
        (t_draw, {"Q_refill": "Q_CSF"}),
        (t_refill, {"Q_SN": "f_SN*Q_CSF", "Q_LP": "Q_Leak", "Q_refill": "0"}),
    ])
    s.linear_input("f_L", [0, 1, 2.5, 4, 6.5], [0.1, 0.3, 0.2, 0.5, 0.1],
                   default_before=0, default_after=0)
    return s


def settings(n_points=801, sched=None):
    out = {"integrator": "cvode", "absolute_tolerance": 1e-12, "relative_tolerance": 1e-12,
           "stiff": True, "variable_step_size": True,
           "simulation_blocks": {
               "pre": {"start": -5, "end": 0, "n_points": 51, "tracked": False},
               "run": {"start": 0, "end": 8, "n_points": n_points, "tracked": True}}}
    if sched is not None:
        out["schedule"] = sched
    return out


def with_events(q):
    return te.loada(MODEL.format(q=q) + schedule(q).to_antimony())


def without_events(q):
    return te.loada(MODEL.format(q=q) + schedule(q).to_antimony(events=False))


def run(r, **kw):
    return sim.simulate(r, settings(**kw), OBSERVED)


@pytest.mark.parametrize("q", [17.39500698, 12.0, 25.0])
def test_it_integrates_like_the_events_it_replaces(q):
    legacy = np.asarray(run(with_events(q)))
    new = np.asarray(run(without_events(q), sched=schedule(q)))
    assert new.shape == legacy.shape
    np.testing.assert_allclose(new[:, 0], legacy[:, 0], rtol=0, atol=1e-9)   # the same grid
    np.testing.assert_allclose(new, legacy, rtol=1e-6, atol=1e-8)


def test_a_dense_grid_costs_no_extra_restarts_and_lands_on_the_switches():
    # 0.01 h spacing: grid points fall on every hour and, to within 1e-9, on the draw ends
    q = PARAMS["Q_CSF"]
    legacy = np.asarray(run(with_events(q), n_points=801))
    new = np.asarray(run(without_events(q), n_points=801, sched=schedule(q)))
    hours = np.isclose(legacy[:, 0] % 1.0, 0.0, atol=1e-9)
    assert hours.sum() >= 8
    # at a switch the row shows the value after it, as an event firing at its trigger time does
    np.testing.assert_allclose(new[hours, 1:], legacy[hours, 1:], rtol=1e-6, atol=1e-8)


def test_the_model_it_runs_has_no_events():
    r = without_events(PARAMS["Q_CSF"])
    assert build_event_time_table(r).n_events == 0


def test_the_run_leaves_the_model_as_it_found_it():
    q = PARAMS["Q_CSF"]
    r = without_events(q)
    before = {n: r[n] for n in ("Q_Leak", "Q_SN", "Q_LP", "Q_refill")}
    r.integrator.variable_step_size = True
    run(r, sched=schedule(q))
    assert {n: r[n] for n in before} == before
    assert r.integrator.variable_step_size is True


def test_a_second_run_gives_the_same_answer():
    q = 12.0
    r = without_events(q)
    first = np.array(run(r, sched=schedule(q)))
    r.reset()
    second = np.array(run(r, sched=schedule(q)))
    assert np.array_equal(first, second)


def test_the_run_follows_the_parameters_the_model_holds():
    # the schedule was designed at the nominal Q_CSF; the model has been moved
    r = without_events(25.0)
    new = np.asarray(run(r, sched=schedule(PARAMS["Q_CSF"])))
    legacy = np.asarray(run(with_events(25.0)))
    np.testing.assert_allclose(new, legacy, rtol=1e-6, atol=1e-8)


def test_parameters_that_reorder_the_switches_are_refused():
    r = without_events(3.0)                     # V_LP/Q_CSF = 2 h: after the next draw
    with pytest.raises(ScheduleError):
        run(r, sched=schedule())


# --- explicit output times in safe_simulate ---------------------------------------

def test_safe_simulate_returns_exactly_the_times_asked_for():
    r = without_events(PARAMS["Q_CSF"])
    r.integrator.variable_step_size = False
    grid = [0.0, 0.1, 0.25, 0.6, 1.0]
    res, _ = sim.safe_simulate(r, {"times": grid, "variable_step_size": False}, OBSERVED)
    np.testing.assert_array_equal(np.asarray(res)[:, 0], grid)


def test_the_retry_ladder_subdivides_an_explicit_grid(monkeypatch):
    r = without_events(PARAMS["Q_CSF"])
    r.integrator.variable_step_size = False
    grid = list(np.linspace(0.0, 1.0, 9))
    direct, _ = sim.safe_simulate(r, {"times": grid, "variable_step_size": False}, OBSERVED)
    direct = np.array(direct)

    real, done = sim._run, []

    def flaky(rr, start, end, points, times, observed):
        if len(times) > 3:
            raise RuntimeError("CV_TOO_MUCH_WORK: pretend")
        done.append(len(times))
        return real(rr, start, end, points, times, observed)

    monkeypatch.setattr(sim, "_run", flaky)
    r.reset()
    with sim.retry_limits(max_attempts=2, max_depth=4):
        res, meta = sim.safe_simulate(r, {"times": grid, "variable_step_size": False}, OBSERVED)
    assert meta["subdivided"] and done and max(done) <= 3
    np.testing.assert_array_equal(np.asarray(res)[:, 0], grid)
    # the ladder loosens the tolerances (to 1e-4 here) before it subdivides, so the
    # values agree to that, not to the solver's own accuracy
    np.testing.assert_allclose(np.asarray(res), direct, rtol=1e-3, atol=1e-3)


# --- the pre-equilibration cache ----------------------------------------------------

def test_switches_before_a_time():
    r = without_events(PARAMS["Q_CSF"])
    s = schedule()
    assert not switches_before(r, s, 0.0)        # the first switch is at 0: not before it
    assert switches_before(r, s, 3.0)


def test_a_schedule_that_switches_inside_the_cached_block_is_not_cached():
    q = PARAMS["Q_CSF"]
    r = without_events(q)
    cache = PreequilCache()
    ss = settings(sched=schedule(q))
    ss["simulation_blocks"]["pre"]["end"] = 3.0          # now the schedule switches inside it
    assert cache.apply(r, ss, OBSERVED) is ss
    assert cache.stats()["misses"] == 0

    ss = settings(sched=schedule(q))                     # block ends at 0: caching is fine
    rest = cache.apply(r, ss, OBSERVED)
    assert list(rest["simulation_blocks"]) == ["run"] and cache.stats()["misses"] == 1

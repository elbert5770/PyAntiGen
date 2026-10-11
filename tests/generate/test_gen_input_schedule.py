"""Checks for pyantigen.generate.input_schedule.

Run from the repository root:
    python -m pytest tests/generate/test_gen_input_schedule.py

The schedule here is the v4 SILK CSF draw (three switches an hour, two of them
moving with Q_CSF and V_LP), small enough to check by hand. The last tests
compile the schedule's Antimony in RoadRunner and compare it with the
hand-written events the v4 project used before.
"""
import numpy as np
import pytest

from pyantigen.generate.input_schedule import (
    Breakpoint,
    InputSchedule,
    ScheduleError,
)

PARAMS = {"Q_CSF": 17.39500698, "V_LP": 6.0, "t_CSFdraw": 0.1, "f_SN": 0.1}


def silk_like(params=None, hours=48):
    s = InputSchedule(params or PARAMS)
    t_draw = s.moving("t_draw", "t_CSFdraw")
    t_refill = s.moving("t_refill", "V_LP/Q_CSF")
    s.step(0, {"Q_Leak": "15.0"})
    s.step(3, {"Q_Leak": "0.0"})
    s.step(20, {"Q_Leak": "Q_CSF - f_SN*Q_CSF"})
    s.step(30, {"Q_Leak": "0.0"})
    s.step(48, {"Q_Leak": "0.0"})
    s.periodic(0, 1, hours, [
        (0, {"Q_SN": "0.0", "Q_LP": "Q_CSF", "Q_refill": "Q_CSF - V_LP/t_CSFdraw"}),
        (t_draw, {"Q_refill": "Q_CSF"}),
        (t_refill, {"Q_SN": "f_SN*Q_CSF", "Q_LP": "Q_Leak", "Q_refill": "0"}),
    ])
    return s


# --- exact switch times -----------------------------------------------------

def test_breakpoints_compare_exactly():
    a = Breakpoint(1) + Breakpoint(0, {"m": 1})
    b = Breakpoint(0, {"m": 1}) + 1
    assert a == b and hash(a) == hash(b)
    assert a != Breakpoint(1)
    assert (a - a) == Breakpoint(0)               # the moving term cancels exactly
    assert (a * 2).anchor == 2 and (a * 2).terms == (("m", 2),)


def test_a_float_is_read_by_its_decimal_form():
    assert Breakpoint(0.1).anchor * 10 == 1       # one tenth, not the binary neighbour of it
    assert float(Breakpoint(0.1).anchor) == 0.1


def test_coincident_switches_merge_into_one_event():
    s = silk_like()
    ev = dict(s.events())
    # 5 Q_Leak steps + 144 hourly = 149 steps; t = 0, 3, 20, 30 coincide
    assert len(s.events()) == 145
    assert set(ev[Breakpoint(20)]) == {"Q_Leak", "Q_SN", "Q_LP", "Q_refill"}
    assert len(s.times()) == 145


def test_two_values_for_one_variable_at_one_switch_is_an_error():
    s = InputSchedule(PARAMS)
    s.step(5, {"Q_LP": "1"})
    s.step(5, {"Q_LP": "2"})
    with pytest.raises(ScheduleError):
        s.events()


def test_times_match_the_expressions_they_are_written_as():
    s = silk_like()
    t = s.time(Breakpoint(7, {"t_refill": 1}))
    assert t == 7 + 6.0 / 17.39500698             # same doubles, same operations


def test_moving_quantities_accept_arithmetic_only():
    s = InputSchedule(PARAMS)
    with pytest.raises(ScheduleError):
        s.moving("bad", "__import__('os').getcwd()")
    with pytest.raises(ScheduleError):
        s.moving("bad", "V_LP/unknown")


# --- gradients ----------------------------------------------------------------

def test_gradient_is_exact():
    s = silk_like()
    bp = Breakpoint(5, {"t_refill": 1})
    g = s.gradient(bp, wrt=["Q_CSF", "V_LP", "t_CSFdraw", "f_SN"])
    q, v = PARAMS["Q_CSF"], PARAMS["V_LP"]
    assert g["Q_CSF"] == pytest.approx(-v / q**2, rel=1e-14)
    assert g["V_LP"] == pytest.approx(1 / q, rel=1e-14)
    assert g["t_CSFdraw"] == 0.0 and g["f_SN"] == 0.0


def test_gradient_matches_a_finite_difference():
    s = silk_like()
    bp = Breakpoint(5) + Breakpoint(0, {"t_refill": 1}) - Breakpoint(0, {"t_draw": 1})
    h = 1e-6
    up, dn = dict(PARAMS, Q_CSF=PARAMS["Q_CSF"] + h), dict(PARAMS, Q_CSF=PARAMS["Q_CSF"] - h)
    fd = (s.time(bp, up) - s.time(bp, dn)) / (2 * h)
    assert s.gradient(bp)["Q_CSF"] == pytest.approx(fd, rel=1e-8)


def test_durations_in_one_hour_sum_to_the_period_for_every_q_csf():
    s = silk_like()
    for q in (8.0, 17.39500698, 40.0, 59.0):
        p = dict(PARAMS, Q_CSF=q)
        seg = [x for x in s.segments(0, 1, params=p)]
        total = sum(s.time(x.duration, p) for x in seg)
        assert total == pytest.approx(1.0, abs=1e-12)
        assert sum(s.gradient(x.duration, p)["Q_CSF"] for x in seg) == pytest.approx(0.0, abs=1e-14)


# --- feasibility --------------------------------------------------------------

def test_nominal_parameters_validate_without_warnings():
    assert silk_like().validate() == []


@pytest.mark.parametrize("q", [3.0, 5.99, 60.0, 70.0])
def test_q_csf_outside_6_to_60_changes_the_order(q):
    # the refill ends after the hour (q < 6) or before the draw ends (q > 60)
    with pytest.raises(ScheduleError):
        silk_like().validate(dict(PARAMS, Q_CSF=q))


@pytest.mark.parametrize("q", [6.5, 17.39500698, 59.0])
def test_q_csf_inside_6_to_60_is_feasible(q):
    silk_like().validate(dict(PARAMS, Q_CSF=q))


def test_distinct_switches_too_close_to_separate_are_refused():
    s = InputSchedule({"a": 1.0})
    s.moving("m", "a")
    s.step(Breakpoint(2, {"m": 1}), {"x": "1"})             # 3 at a = 1
    s.step(3, {"x": "2"})
    with pytest.raises(ScheduleError):
        s.validate()                                          # the same instant, written two ways


def test_a_read_of_a_variable_assigned_at_the_same_switch_is_reported():
    s = InputSchedule(PARAMS)
    s.step(20, {"Q_Leak": "Q_CSF - Q_SN"})
    s.step(20, {"Q_SN": "0.0"})
    warnings = s.validate()
    assert len(warnings) == 1 and "Q_SN" in warnings[0]


# --- segments -----------------------------------------------------------------

def test_observations_become_fixed_segment_boundaries():
    s = silk_like()
    seg = s.segments(0, 2, observation_times=[0.25, 1.5, 2])
    ends = [x.end for x in seg]
    assert Breakpoint(0.25) in ends and Breakpoint(1.5) in ends
    assert seg[0].start == Breakpoint(0) and seg[-1].end == Breakpoint(2)
    assert all(a.end == b.start for a, b in zip(seg, seg[1:]))
    # 0.25 lies between the draw end (0.1) and the refill end (0.345): that segment
    # runs from a moving switch to a fixed one
    stretch = next(x for x in seg if x.end == Breakpoint(0.25))
    assert stretch.start == Breakpoint(0, {"t_draw": 1}) and stretch.end.fixed


def test_segments_cover_the_run_in_order():
    s = silk_like()
    seg = s.segments(0, 48)
    assert len(seg) == 144                                    # three an hour
    times = [s.time(x.start) for x in seg]
    assert times == sorted(times)


# --- measured curves ----------------------------------------------------------

def test_a_measured_curve_adds_fixed_knots_and_text():
    s = InputSchedule(PARAMS)
    s.linear_input("f_L", [0, 1, 2.5], [0.1, 0.2, 0.15], default_before=0, default_after=0)
    assert s.breakpoints() == {Breakpoint(0), Breakpoint(1), Breakpoint(2.5)}
    assert "f_L := piecewise" in s.to_antimony()


def test_a_measured_curve_with_close_knots_is_refused():
    s = InputSchedule(PARAMS)
    with pytest.raises(ScheduleError):
        s.linear_input("f_L", [0, 0.005, 1], [0, 1, 2])


# --- what an event-free backend needs ---------------------------------------------

def test_ordered_events_are_the_steps_in_time_order_with_the_same_doubles():
    s = silk_like()
    s.linear_input("f_L", [0.5, 1.5], [1, 2])               # knots carry no assignments
    steps = s.ordered_events()
    assert len(steps) == 145
    assert [t for t, _a in steps] == [t for t in s.times() if t not in (0.5, 1.5)]
    assert steps[0][1]["Q_Leak"] == "15.0"


def test_a_right_hand_side_is_evaluated_from_the_values_it_is_given():
    values = {"Q_CSF": 17.5, "V_LP": 6.0, "f_SN": 0.1}
    assert InputSchedule.evaluate("Q_CSF - V_LP/0.1", values.__getitem__) == 17.5 - 6.0 / 0.1
    assert InputSchedule.evaluate("f_SN*Q_CSF", values.__getitem__) == 0.1 * 17.5
    assert InputSchedule.evaluate("2**3 - -1", values.__getitem__) == 9
    with pytest.raises(ScheduleError):
        InputSchedule.evaluate("sin(Q_CSF)", values.__getitem__)


def test_antimony_without_events_keeps_only_the_measured_curve():
    s = silk_like()
    s.linear_input("f_L", [0, 1, 2.5], [0.1, 0.2, 0.15], default_before=0, default_after=0)
    text = s.to_antimony(events=False)
    assert "at (" not in text and "f_L := piecewise" in text


def test_a_moving_step_may_land_on_a_measured_curve_knot():
    # 2 + V_LP/Q_CSF is 2.5 at Q_CSF = 12: the refill falls exactly on the knot
    s = silk_like()
    s.linear_input("f_L", [0, 1, 2.5], [0.1, 0.2, 0.15], default_before=0, default_after=0)
    p = dict(PARAMS, Q_CSF=12.0)
    s.validate(p)
    assert s.time(Breakpoint(2, {"t_refill": 1}), p) == 2.5
    # ... and so may an observation time, which is how a data point becomes a cut
    s.segments(0, 4, observation_times=[2.5], params=p)


def test_a_knot_crossing_a_step_is_not_a_change_of_order():
    s = silk_like()
    s.linear_input("f_L", [0, 1, 2.4, 5], [0.1, 0.2, 0.15, 0.3], default_before=0, default_after=0)
    s.validate(dict(PARAMS, Q_CSF=15.0))                    # 2 + 6/15 = 2.4: on the knot
    s.validate(dict(PARAMS, Q_CSF=14.0))                    # 2.43: the far side of it


def test_a_fixed_time_on_a_moving_switch_comes_after_it_however_the_switch_is_written():
    # 4 - x is 3 at a = 1, and the text '4 - ...' sorts after '3': the tie must not go by spelling
    s = InputSchedule({"a": 1.0})
    x = s.moving("x", "a")
    moving = 4 - x
    s.step(moving, {"u": "1"})
    segs = s.segments(0, 5, observation_times=[3], params={"a": 1.0})
    assert [(g.start, g.end) for g in segs][1] == (moving, Breakpoint(3))
    assert s.time(moving) == s.time(Breakpoint(3)) == 3.0


def test_fixed_times_too_close_to_tell_apart_are_still_refused():
    s = InputSchedule({"a": 1.0})
    s.step(3, {"x": "1"})
    s.linear_input("f_L", [0, 3.0 + 1e-12, 5], [0, 1, 0])
    with pytest.raises(ScheduleError):
        s.validate()


# --- against RoadRunner and the events the project used to write ------------------

LEGACY_Q_LEAK = ("at (time >= 0.0 ): Q_Leak = 15.0\n"
                 "at (time >= 3.0 ): Q_Leak = 0.0\n"
                 "at (time > 20.0): Q_Leak = Q_CSF-Q_SN\n"
                 "at (time >= 30.0): Q_Leak = 0.0\n"
                 "at (time >= 48.0): Q_Leak = 0.0\n")


def legacy_hourly():
    out = ""
    for i in range(48):
        out += f"at (time >= {i}): Q_SN = 0.0, Q_LP = Q_CSF, Q_refill = Q_CSF - V_LP/t_CSFdraw\n"
        out += f"at (time >= {i} + t_CSFdraw):  Q_refill = Q_CSF\n"
        out += f"at (time >= {i} + V_LP/Q_CSF): Q_SN = f_SN*Q_CSF, Q_LP = Q_Leak, Q_refill = 0\n"
    return out


TOY = """
Q_CSF = {q!r}; f_SN = 0.1; V_LP = 6; t_CSFdraw = 0.1
Q_Leak = 0; Q_LP = 0; Q_refill = 0; Q_SN = Q_CSF*f_SN
x_leak = 0; x_sn = 0; x_lp = 0; x_refill = 0
x_leak' = Q_Leak; x_sn' = Q_SN; x_lp' = Q_LP; x_refill' = Q_refill
"""


def _integrals(q, events):
    te = pytest.importorskip("tellurium")
    r = te.loada(TOY.format(q=q) + events)
    r.integrator.absolute_tolerance = 1e-14
    r.integrator.relative_tolerance = 1e-12
    res = r.simulate(-1, 48, 4801, selections=["time", "x_leak", "x_sn", "x_lp", "x_refill"])
    return r, np.asarray(res)


@pytest.mark.parametrize("q", [17.39500698, 12.0, 25.0])
def test_the_events_it_writes_integrate_like_the_old_hand_written_ones(q):
    s = silk_like(dict(PARAMS, Q_CSF=q))
    _r, new = _integrals(q, s.to_antimony())
    _r, old = _integrals(q, LEGACY_Q_LEAK + legacy_hourly())
    np.testing.assert_allclose(new, old, rtol=1e-12, atol=1e-12)


def test_schedule_times_equal_the_times_the_engine_reads_from_the_model():
    pytest.importorskip("tellurium")
    from pyantigen.engine.Event_times import build_event_time_table
    s = silk_like()
    r, _ = _integrals(PARAMS["Q_CSF"], s.to_antimony())
    assert build_event_time_table(r).times() == s.times()

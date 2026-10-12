"""The fit's box against the region a schedule can be integrated in.

Run from the repository root: python -m pytest tests/engine/test_schedule_bounds.py

A schedule's switches must keep their order (``InputSchedule.validate``); a parameter set where
they do not is a failed evaluation, which a gradient method reads as a cliff. ``check_box`` and
``shrink_box`` find that region's edge for a box of bounds, and the optimizer
(``Optimize._apply_schedule_bounds``) uses them before the first evaluation.

The schedule is the SILK hour: draw at 0, end of draw at t_CSFdraw, refill at V_LP/Q_CSF, the next
hour at 1. The order 0 < t_CSFdraw < V_LP/Q_CSF < 1 is the feasible region; with V_LP = 6 and
t_CSFdraw = 0.1 that is 6 < Q_CSF < 60.
"""
import pytest

from pyantigen.engine.Optimize import _apply_schedule_bounds
from pyantigen.generate.input_schedule import InputSchedule, ScheduleError

NOMINAL = {"Q_CSF": 17.39500698, "V_LP": 6.0, "t_CSFdraw": 0.1, "f_SN": 0.1}


def schedule():
    s = InputSchedule(NOMINAL)
    t_draw = s.moving("t_draw", "t_CSFdraw")
    t_refill = s.moving("t_refill", "V_LP/Q_CSF")
    s.periodic(0, 1, 4, [
        (0, {"Q_SN": "0.0", "Q_LP": "Q_CSF"}),
        (t_draw, {"Q_refill": "Q_CSF"}),
        (t_refill, {"Q_SN": "f_SN*Q_CSF", "Q_refill": "0"}),
    ])
    return s


FEASIBLE = {"Q_CSF": (10.0, 50.0), "V_LP": (3.0, 6.5), "t_CSFdraw": (0.02, 0.055)}
TOO_WIDE = {"Q_CSF": (8.0, 50.0), "V_LP": (3.0, 12.0), "t_CSFdraw": (0.02, 0.2)}


# --- InputSchedule.check_box / shrink_box -----------------------------------------------------------

def test_only_the_parameters_that_move_a_switch_matter():
    assert schedule().switch_parameters() == {"Q_CSF", "V_LP", "t_CSFdraw"}      # not f_SN


def test_a_box_inside_the_region_is_feasible():
    out = schedule().check_box(FEASIBLE)
    assert out["feasible"] and not out["failures"] and out["tested"] >= 8        # at least the corners
    assert out["intervals"] == {k: pytest.approx(v) for k, v in FEASIBLE.items()}


def test_a_box_that_reaches_swapped_switches_says_where():
    out = schedule().check_box(TOO_WIDE)
    assert not out["feasible"]
    point, message = out["failures"][0]
    assert "order of the switches" in message
    # the corner it names really is infeasible, and the nominal point is not
    with pytest.raises(ScheduleError):
        schedule().validate({**NOMINAL, **point})
    schedule().validate(NOMINAL)


def test_intervals_are_each_bound_cut_alone_around_the_nominal_value():
    out = schedule().check_box({"Q_CSF": (2.0, 50.0), "V_LP": FEASIBLE["V_LP"], "t_CSFdraw": (0.02, 0.055)})
    lo, hi = out["intervals"]["Q_CSF"]
    assert hi == pytest.approx(50.0)
    # below Q_CSF = V_LP/1 h = 6 the refill is later than the next draw
    assert lo == pytest.approx(6.0, rel=1e-6)


def test_the_box_is_sampled_inside_as_well_as_at_the_corners():
    s = schedule()
    corners_only = s.check_box(TOO_WIDE, interior=0)
    assert corners_only["tested"] == 8
    assert s.check_box(TOO_WIDE, interior=20)["tested"] == 28
    # the same seed tests the same points
    assert s.check_box(TOO_WIDE, seed=3) == s.check_box(TOO_WIDE, seed=3)


def test_unbounded_and_irrelevant_parameters_are_left_alone():
    out = schedule().check_box({"Q_CSF": (10.0, None), "f_SN": (0.0, 5.0), "V_LP": FEASIBLE["V_LP"]})
    assert out["feasible"] and set(out["intervals"]) == {"V_LP"}


def test_an_infeasible_nominal_point_or_reversed_bounds_raise():
    with pytest.raises(ScheduleError):
        schedule().check_box(FEASIBLE, params={**NOMINAL, "Q_CSF": 3.0})
    with pytest.raises(ScheduleError, match="reversed"):
        schedule().check_box({"Q_CSF": (50.0, 10.0)})


def test_shrinking_a_box_toward_the_nominal_point_makes_it_feasible():
    s = schedule()
    cut, alpha = s.shrink_box(TOO_WIDE)
    assert 0.0 < alpha < 1.0
    assert s.check_box(cut)["feasible"]
    # and it is close to the largest such box: a bit more is not feasible
    bigger = {n: (NOMINAL[n] + (alpha * 1.05) * (lo - NOMINAL[n]), NOMINAL[n] + (alpha * 1.05) * (hi - NOMINAL[n]))
              for n, (lo, hi) in TOO_WIDE.items()}
    assert not s.check_box(bigger, interior=16)["feasible"]
    # a box that needs no cut comes back whole
    whole, a1 = s.shrink_box(FEASIBLE)
    assert a1 == 1.0 and whole == FEASIBLE


# --- the optimizer's check ------------------------------------------------------------------------

te = pytest.importorskip("tellurium")


def _problem(with_schedule=True):
    r = te.loada("Q_CSF = 17.39500698; V_LP = 6; t_CSFdraw = 0.1; f_SN = 0.1; x = 0; x' = Q_CSF*f_SN")
    rep = {"Solver_settings": lambda rep: ({"schedule": schedule()} if with_schedule else {})}
    return {"sim": rep}, {"sim": {"r": r}}


NAMES = ["Q_CSF", "V_LP", "t_CSFdraw", "f_SN"]
X0 = [17.39500698, 6.0, 0.05, 0.1]
WIDE = [(8.0, 50.0), (3.0, 12.0), (0.02, 0.2), (0.01, 0.5)]
OK = [(10.0, 50.0), (3.0, 6.5), (0.02, 0.055), (0.01, 0.5)]


def test_a_feasible_box_passes_untouched_in_every_mode(capsys):
    reps, models = _problem()
    for mode in ("warn", "error", "clip", "ignore"):
        assert _apply_schedule_bounds(reps, models, NAMES, X0, OK, mode) == [tuple(b) for b in OK] \
            or mode == "ignore"
    assert "WARNING" not in capsys.readouterr().out


def test_warn_prints_where_the_box_goes_wrong_and_changes_nothing(capsys):
    reps, models = _problem()
    out = _apply_schedule_bounds(reps, models, NAMES, X0, WIDE, "warn")
    assert out == [tuple(b) for b in WIDE]
    text = capsys.readouterr().out
    assert "WARNING" in text and "change order" in text and "schedule_bounds" in text


def test_error_raises_with_the_same_message():
    reps, models = _problem()
    with pytest.raises(ScheduleError, match="change order"):
        _apply_schedule_bounds(reps, models, NAMES, X0, WIDE, "error")


def test_clip_returns_bounds_the_schedule_can_be_integrated_in(capsys):
    reps, models = _problem()
    out = _apply_schedule_bounds(reps, models, NAMES, X0, WIDE, "clip")
    assert "were cut" in capsys.readouterr().out
    cut = {n: b for n, b in zip(NAMES, out)}
    assert schedule().check_box({n: cut[n] for n in ("Q_CSF", "V_LP", "t_CSFdraw")},
                                params={**NOMINAL, "t_CSFdraw": 0.05})["feasible"]
    assert cut["f_SN"] == (0.01, 0.5)                          # a parameter that moves no switch keeps its bounds
    for (lo, hi), (clo, chi), x in zip(WIDE, out, X0):
        assert lo <= clo <= x <= chi <= hi                      # inside the old box, around the start


def test_a_starting_point_outside_the_region_stops_the_fit_unless_ignored():
    reps, models = _problem()
    bad_x0 = [17.39500698, 6.0, 0.9, 0.1]                       # t_CSFdraw after the refill
    for mode in ("warn", "error", "clip"):
        with pytest.raises(ScheduleError, match="starting point"):
            _apply_schedule_bounds(reps, models, NAMES, bad_x0, WIDE, mode)
    assert _apply_schedule_bounds(reps, models, NAMES, bad_x0, WIDE, "ignore") == WIDE


def test_simulations_without_a_schedule_and_unknown_modes():
    reps, models = _problem(with_schedule=False)
    assert _apply_schedule_bounds(reps, models, NAMES, X0, WIDE, "error") == [tuple(b) for b in WIDE]
    with pytest.raises(ValueError, match="schedule_bounds"):
        _apply_schedule_bounds(reps, models, NAMES, X0, WIDE, "sometimes")
    assert _apply_schedule_bounds(reps, models, NAMES, X0, None, "error") is None

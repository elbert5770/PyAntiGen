"""The re-walk pass of the profile (``pyantigen.engine.Profile_rewalk``): which sides it takes up, and the forced start.

Run from the repository root:
    python -m pytest tests/engine/test_profile_rewalk.py
"""
import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from pyantigen.engine import Profile_rewalk as RW   # noqa: E402

LEVEL = 1.9207


def pt(x, dnll, box=False):
    return SimpleNamespace(x_fixed=x, dnll=dnll, box_limited=box)


def test_closely_bracketed_side_is_left_alone():
    best = [pt(0.0, 0.0), pt(0.5, 1.0), pt(1.0, 1.85), pt(1.1, 2.05)]       # 0.07 under, 0.13 over the level
    assert RW.unresolved(best, 1, LEVEL) is None


def test_loose_crossing_is_taken_up_from_the_farthest_inside_point():
    best = [pt(0.0, 0.0), pt(0.5, 1.0), pt(1.5, 3.0)]                       # 0.92 under, 1.08 over
    start, beyond, why = RW.unresolved(best, 1, LEVEL)
    assert start.x_fixed == 0.5 and [q.x_fixed for q in beyond] == [1.5]
    assert "not closely bracketed" in why


def test_box_limited_crossing_starts_before_the_run_of_limited_points():
    best = [pt(0.0, 0.0), pt(0.3, 0.8), pt(0.6, 1.2, True), pt(0.9, 1.85, True), pt(1.0, 2.0, True)]
    start, beyond, why = RW.unresolved(best, 1, LEVEL)
    assert start.x_fixed == 0.3 and "bound" in why


def test_down_side_orders_by_direction():
    best = [pt(0.0, 0.0), pt(-0.5, 1.0), pt(-1.5, 3.0)]
    start, beyond, _ = RW.unresolved(sorted(best, key=lambda q: -1 * q.x_fixed), -1, LEVEL)
    assert start.x_fixed == -0.5 and [q.x_fixed for q in beyond] == [-1.5]


def test_open_or_empty_side_is_left_alone():
    assert RW.unresolved([pt(0.0, 0.0), pt(1.0, 0.5)], 1, LEVEL) is None     # nothing past the level
    assert RW.unresolved([], 1, LEVEL) is None


def test_forced_start_picks_the_named_point_for_the_named_side_only():
    best = [pt(1.055992, 0.0), pt(1.148, 0.09), pt(1.493, 1.47)]
    spec = "1:1:1.055992,2:1:2.174526"
    assert RW.forced_start(1, 1, best, spec).x_fixed == 1.055992
    assert RW.forced_start(1, -1, best, spec) is None                         # the other direction
    assert RW.forced_start(3, 1, best, spec) is None                          # another parameter
    assert RW.forced_start(1, 1, best, "") is None


def test_forced_start_refuses_a_value_without_a_recorded_point():
    best = [pt(1.055992, 0.0), pt(1.148, 0.09)]
    with pytest.raises(ValueError, match="no recorded point"):
        RW.forced_start(1, 1, best, "1:1:1.3")


def test_forced_start_reads_the_environment(monkeypatch):
    best = [pt(2.174526, 0.0)]
    monkeypatch.setenv("REWALK_FORCE_FROM", "2:1:2.174526")
    assert RW.forced_start(2, 1, best).x_fixed == 2.174526
    monkeypatch.delenv("REWALK_FORCE_FROM")
    assert RW.forced_start(2, 1, best) is None


def test_settings_are_read_at_call_time(monkeypatch):
    monkeypatch.delenv("REWALK_EVALS", raising=False)
    assert RW.settings() == {"evals": 200, "hmax": 0.02, "maxpts": 120}
    monkeypatch.setenv("REWALK_HMAX", "0.05")
    assert RW.settings()["hmax"] == 0.05


def test_recheck_dispatches_to_the_rewalk(monkeypatch):
    from pyantigen.engine import Profile
    calls = []
    monkeypatch.setattr(RW, "rewalk", lambda *a: calls.append(a) or "rewalked")
    monkeypatch.setenv("PYANTIGEN_PROFILE_RECHECK", "rewalk")
    assert Profile.recheck("pr", 1, 1, None, "store", None, None, "profile") == "rewalked"
    assert calls and calls[0][1:3] == (1, 1)

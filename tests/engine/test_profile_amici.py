"""The AMICI profile stage (Profile_amici) on a linear-residual stand-in for the AMICI provider."""
import math

import numpy as np
import pytest

from pyantigen.engine import Profile_amici as PA
from pyantigen.engine.Profile_record import crossings


class LinearProvider:
    """r = L p - y, so f = 0.5 r.r is a quadratic whose profile in p_k is 0.5 (p_k - p*_k)^2 / cov_kk."""

    def __init__(self):
        rng = np.random.default_rng(3)
        self.L = rng.normal(size=(12, 3))
        self.y = self.L @ np.array([1.0, -0.5, 2.0]) + 0.05 * rng.normal(size=12)
        self.calls = 0

    def linear_residual_jacobian(self, p):
        self.calls += 1
        return self.L @ np.asarray(p, float) - self.y, self.L.copy(), 0.0


NAMES = ["a", "b", "c"]
SCALES = ["lin", "lin", "lin"]
BOUNDS = [(-20.0, 20.0)] * 3


def _optimum(p):
    x, *_ = np.linalg.lstsq(p.L, p.y, rcond=None)
    return x


def _expected(p, k):
    cov = np.linalg.inv(p.L.T @ p.L)
    return _optimum(p)[k] + np.array([-1.0, 1.0]) * math.sqrt(2 * 1.9207 * cov[k, k])


def test_side_from_env():
    assert PA.side_from_env(3, {}) is None
    assert PA.side_from_env(3, {PA.SIDE_ENV: ""}) is None
    assert PA.side_from_env(3, {PA.SIDE_ENV: "0"}) == (0, 1)
    assert PA.side_from_env(3, {PA.SIDE_ENV: "5"}) == (2, -1)
    with pytest.raises(ValueError):
        PA.side_from_env(3, {PA.SIDE_ENV: "6"})


def test_every_side_in_turn_finds_the_analytic_crossings(tmp_path):
    p = LinearProvider()
    xs = _optimum(p)
    out = PA.run(p, NAMES, BOUNDS, SCALES, xs, str(tmp_path), log=lambda m: None)
    assert len(out["ran"]) == 6
    assert PA.sides_finished(str(tmp_path), NAMES) == [True] * 6
    by = PA.read_sides(str(tmp_path), NAMES)
    for k, name in enumerate(NAMES):
        lo, hi = crossings(by[name], xs[k])
        want = _expected(p, k)
        assert lo == pytest.approx(want[0], abs=0.01) and hi == pytest.approx(want[1], abs=0.01)


def test_one_side_per_process_then_collect(tmp_path):
    """A Slurm array: each task runs one side; a last run with no side finds everything done and walks nothing."""
    p = LinearProvider()
    xs = _optimum(p)
    for task in range(6):
        PA.run(p, NAMES, BOUNDS, SCALES, xs, str(tmp_path), side=PA.side_from_env(3, {PA.SIDE_ENV: str(task)}),
               log=lambda m: None)
    before = p.calls
    out = PA.run(p, NAMES, BOUNDS, SCALES, xs, str(tmp_path), log=lambda m: None)
    assert p.calls - before == 1            # only the evaluation at the optimum: every side was already finished
    assert PA.sides_finished(str(tmp_path), NAMES) == [True] * 6
    completed = PA.completed_from_store(str(tmp_path), NAMES, SCALES, lambda x: round(x, 12))
    pts = completed["b"].values()
    assert all(r["status"] == "ok" and r["backend"] == "amici" for r in pts)
    assert min(r["dnll"] for r in pts) == pytest.approx(0.0, abs=1e-9)
    assert out["floors"]


def test_all_tasks_share_one_floors_file(tmp_path):
    p = LinearProvider()
    xs = _optimum(p)
    a = PA.run(p, NAMES, BOUNDS, SCALES, xs, str(tmp_path), side=(0, 1), log=lambda m: None)
    b = PA.run(p, NAMES, BOUNDS, SCALES, xs, str(tmp_path), side=(1, -1), log=lambda m: None)
    assert a["floors"] == b["floors"]
    recs = PA.read_sides(str(tmp_path), NAMES)
    assert {r.floors_sha256 for rs in recs.values() for r in rs} == {a["floors"]}


def test_record_to_raw_carries_what_the_reports_read(tmp_path):
    p = LinearProvider()
    xs = _optimum(p)
    PA.run(p, NAMES, BOUNDS, SCALES, xs, str(tmp_path), side=(2, 1), log=lambda m: None)
    rec = [r for r in PA.read_sides(str(tmp_path), NAMES)["c"] if r.dnll and r.dnll > 0][0]
    raw = PA.record_to_raw(rec, NAMES, SCALES)
    assert raw["x_fixed"] == rec.x_fixed and raw["x_fixed_linear"] == rec.x_fixed
    assert len(raw["nuisance_x"]) == 2 and raw["converged"] and not raw["interrupted"]


def test_mode_and_pins_come_from_the_environment():
    from pyantigen.engine import Profile_amici as PA
    mode, st = PA.settings_from_env({})
    assert mode == "walk" and st.pins[0] == 0.03
    mode, st = PA.settings_from_env({"PYANTIGEN_PROFILE_MODE": "pins", "PYANTIGEN_PROFILE_PINS": "0.5, 2,10"})
    assert mode == "pins" and st.pins == (0.5, 2.0, 10.0)
    import pytest
    with pytest.raises(ValueError):
        PA.settings_from_env({"PYANTIGEN_PROFILE_MODE": "grid"})


def test_pins_mode_has_its_own_files_and_agrees_with_the_walk(tmp_path):
    p = LinearProvider()
    xs = _optimum(p)
    PA.run(p, NAMES, BOUNDS, SCALES, xs, str(tmp_path), log=lambda m: None)
    walk_files = sorted(f for f in __import__("os").listdir(tmp_path))
    st = PA.PR.WalkSettings(pins=(0.5, 0.8, 1.25, 2.0, 4.0))
    out = PA.run(p, NAMES, BOUNDS, SCALES, xs, str(tmp_path), settings=st, mode="pins", log=lambda m: None)
    assert len(out["ran"]) == 6 and all(t.startswith("p") for t in out["ran"])
    assert set(walk_files) <= set(__import__("os").listdir(tmp_path))            # the walk's files were not touched
    assert PA.sides_finished(str(tmp_path), NAMES, mode="pins") == [True] * 6
    by = PA.read_sides(str(tmp_path), NAMES, mode="pins")
    for k, name in enumerate(NAMES):
        lo, hi = crossings(by[name], xs[k])
        want = _expected(p, k)
        assert lo == pytest.approx(want[0], abs=0.01) and hi == pytest.approx(want[1], abs=0.01)


def test_build_lock_serialises_processes(tmp_path):
    import multiprocessing as mp
    import time
    from pyantigen.engine.Analytic_gradient import _build_lock
    ctx = mp.get_context("fork")
    spans = ctx.Queue()

    def work():
        with _build_lock(str(tmp_path / "model_x")):
            t0 = time.time()
            time.sleep(0.2)
            spans.put((t0, time.time()))
    ps = [ctx.Process(target=work) for _ in range(3)]
    [q.start() for q in ps]
    got = sorted(spans.get(timeout=30) for _ in range(3))
    [q.join() for q in ps]
    assert all(a[1] <= b[0] for a, b in zip(got, got[1:]))                    # no two holders at once


class CurvedProvider:
    """r = (a - 1, b - a^2, c - 3): the profile of a is 0.5 (a - 1)^2, but b must follow a^2 past its bound."""

    def linear_residual_jacobian(self, p):
        a, b, c = p
        return (np.array([a - 1.0, b - a ** 2, c - 3.0]),
                np.array([[1.0, 0, 0], [-2.0 * a, 1.0, 0], [0, 0, 1.0]]), 0.0)


def _curved_report(tmp_path, relax):
    bounds = [(-9.0, 9.0), (-99.0, 3.0), (-9.0, 9.0)]
    st = PA.PR.WalkSettings(relax=relax)
    PA.run(CurvedProvider(), NAMES, bounds, SCALES, np.array([1.0, 1.0, 3.0]), str(tmp_path), settings=st,
           log=lambda m: None)
    return PA.side_report(PA.read_sides(str(tmp_path), NAMES), NAMES, bounds)


def test_side_report_names_the_bound_that_limits_a_side(tmp_path):
    rep = _curved_report(tmp_path, relax=0.0)
    assert rep["a"]["above"]["verdict"] == "box-limited" and rep["a"]["above"]["bound_active"] == ["b"]
    assert rep["a"]["below"]["verdict"] == "box-limited"            # b = a^2 reaches its bound on this side too
    assert rep["c"]["above"]["verdict"] == "crossed"
    text = "\n".join(PA.format_side_report(rep))
    assert "a above: BOX-LIMITED" in text and "on a bound at the crossing: b" in text and "c above" not in text


def test_side_report_says_when_relaxing_needed_a_wider_box(tmp_path):
    rep = _curved_report(tmp_path, relax=1.0)
    above = rep["a"]["above"]
    assert above["verdict"] == "crossed" and above["bound_active"] == []
    assert above["needs_wider"]["b"][1] > 3.0
    assert "widen its bound" in "\n".join(PA.format_side_report(rep))

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

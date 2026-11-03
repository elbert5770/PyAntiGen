"""Multi-start on AMICI (Multistart_amici): one LM fit per start, then a collect, on a residual stand-in for the provider."""
import json
import os

import numpy as np
import pytest

from pyantigen.engine import Multistart_amici as MA

NAMES = ["a", "b"]
SCALES = ["lin", "lin"]
BOUNDS = [(-10.0, 10.0), (-10.0, 10.0)]


class Bowl:
    """r = (a - 1, b + 2): one minimum, f = 0 there."""

    def linear_residual_jacobian(self, p):
        return np.array([p[0] - 1.0, p[1] + 2.0]), np.eye(2), 0.0


class TwoValleys:
    """r = (a^2 - 4, b): minima at a = +-2, the one at +2 shifted up by 3 nats through a constant."""

    def linear_residual_jacobian(self, p):
        r = np.array([p[0] ** 2 - 4.0, p[1]])
        J = np.array([[2.0 * p[0], 0.0], [0.0, 1.0]])
        return r, J, (3.0 if p[0] > 0 else 0.0)


POINTS = [np.array([5.0, 5.0]), np.array([-5.0, 3.0]), np.array([4.0, -4.0])]


def test_task_from_env():
    assert MA.task_from_env(3, {}) is None and MA.task_from_env(3, {MA.ENV: ""}) is None
    assert MA.task_from_env(3, {MA.ENV: "2"}) == 2 and MA.task_from_env(3, {MA.ENV: "Collect"}) == "collect"
    with pytest.raises(ValueError):
        MA.task_from_env(3, {MA.ENV: "3"})


def test_a_start_is_fitted_written_and_not_run_again(tmp_path):
    d = str(tmp_path)
    rec = MA.run_task(0, POINTS, Bowl(), BOUNDS, SCALES, d, log=lambda m: None)
    assert rec["converged"] and rec["f"] == pytest.approx(0.0, abs=1e-9) and rec["x_lin"] == pytest.approx([1.0, -2.0], abs=1e-4)
    again = MA.run_task(0, POINTS, None, BOUNDS, SCALES, d, log=lambda m: None)          # no provider: it must not need one
    assert again == rec


def test_an_unconverged_start_is_continued_from_where_it_stopped(tmp_path):
    d = str(tmp_path)
    short = MA.run_task(1, POINTS, Bowl(), BOUNDS, SCALES, d, options={"maxfev": 1}, log=lambda m: None)
    assert not short["converged"]
    full = MA.run_task(1, POINTS, Bowl(), BOUNDS, SCALES, d, log=lambda m: None)
    assert full["converged"] and full["evals"] > short["evals"] and full["x_start"] == short["x_start"]


def test_collect_lists_unfinished_starts_and_picks_the_lowest_converged(tmp_path):
    d = str(tmp_path)
    for i in (0, 1):
        MA.run_task(i, POINTS, TwoValleys(), BOUNDS, SCALES, d, log=lambda m: None)
    lines = []
    rep = MA.collect(d, NAMES, 3, log=lines.append)
    assert rep["unfinished"] == [2] and open(os.path.join(d, "unfinished.txt")).read() == "2\n"
    assert any("--array=2" in m for m in lines)
    assert rep["best"] == 1 and rep["better_than_start_0"] and rep["gain_over_start_0"] == pytest.approx(3.0, abs=1e-6)
    assert rep["verdict"] == "multimodal"
    best = json.load(open(os.path.join(d, "best.json")))["complete"]
    assert best["parameters"]["a"] == pytest.approx(-2.0, abs=1e-3) and best["fun"] == pytest.approx(0.0, abs=1e-5)
    assert any("best.json" in m for m in lines)


def test_collect_says_start_0_stands_when_nothing_beats_it(tmp_path):
    d = str(tmp_path)
    for i in range(3):
        MA.run_task(i, POINTS, Bowl(), BOUNDS, SCALES, d, log=lambda m: None)
    lines = []
    rep = MA.collect(d, NAMES, 3, log=lines.append)
    assert rep["unfinished"] == [] and open(os.path.join(d, "unfinished.txt")).read() == ""
    assert not rep["better_than_start_0"] and rep["verdict"] == "unimodal"
    assert any("start 0 stands" in m for m in lines)


def test_collect_with_nothing_run_reports_that(tmp_path):
    rep = MA.collect(str(tmp_path), NAMES, 2, log=lambda m: None)
    assert rep["best"] is None and rep["unfinished"] == [0, 1]


def test_unfinished_list_is_also_written_where_the_submitter_asks(tmp_path, monkeypatch):
    mirror = tmp_path / "state" / "ms.txt"
    monkeypatch.setenv("PYANTIGEN_UNFINISHED_FILE", str(mirror))
    d = str(tmp_path / "ms")
    MA.run_task(0, POINTS, Bowl(), BOUNDS, SCALES, d, log=lambda m: None)
    MA.collect(d, NAMES, 3, log=lambda m: None)
    assert mirror.read_text() == "1,2\n" and open(os.path.join(d, "unfinished.txt")).read() == "1,2\n"


def test_a_multistart_stage_is_recognised_so_the_fit_cache_cannot_short_circuit_it():
    assert MA.requested({"PYANTIGEN_MULTISTART": "3"}) and MA.requested({"PYANTIGEN_MULTISTART": "collect"})
    assert not MA.requested({}) and not MA.requested({"PYANTIGEN_MULTISTART": "  "})


# ---- one objective for every start: the noise floors are calibrated once, at start 0, and shared ----

class Floored:
    """r = (p - target) / sigma, with sigma calibrated on the FIRST point evaluated and cached (as Noise_floor does)."""

    def linear_residual_jacobian(self, p):
        from pyantigen.engine import Noise_floor
        sigma = Noise_floor._FLOOR_CACHE.setdefault(("sigma",), abs(float(p[0])) + 1.0)
        return (np.asarray(p) - np.array([1.0, -2.0])) / sigma, np.eye(2) / sigma, 0.0


@pytest.fixture
def fresh_floors():
    from pyantigen.engine import Noise_floor
    Noise_floor.clear_cache()
    yield Noise_floor
    Noise_floor.clear_cache()


def test_every_start_is_scored_under_the_floors_calibrated_at_start_0(tmp_path, fresh_floors):
    d = str(tmp_path)
    r1 = MA.run_task(1, POINTS, Floored(), BOUNDS, SCALES, d, log=lambda m: None)      # start 1 first: it must still calibrate at start 0
    assert fresh_floors._FLOOR_CACHE[("sigma",)] == pytest.approx(6.0)                  # |POINTS[0][0]| + 1, not |POINTS[1][0]| + 1
    fresh_floors.clear_cache()                                                          # a new task is a new process
    r0 = MA.run_task(0, POINTS, Floored(), BOUNDS, SCALES, d, log=lambda m: None)
    assert fresh_floors._FLOOR_CACHE[("sigma",)] == pytest.approx(6.0)
    assert r0["floors"] == r1["floors"] == MA.floors_stamp(d) is not None


def test_a_later_task_loads_the_shared_file_and_does_not_calibrate(tmp_path, fresh_floors):
    d = str(tmp_path)
    MA.run_task(0, POINTS, Floored(), BOUNDS, SCALES, d, log=lambda m: None)
    fresh_floors.clear_cache()
    lines = []
    MA.run_task(2, POINTS, Floored(), BOUNDS, SCALES, d, log=lines.append)
    assert not any("calibrating" in m for m in lines) and any("floors " in m for m in lines)


def test_a_start_scored_under_other_floors_is_fitted_again_and_not_ranked(tmp_path, fresh_floors):
    d = str(tmp_path)
    MA.run_task(0, POINTS, Floored(), BOUNDS, SCALES, d, log=lambda m: None)
    old = {"start": 1, "x_start": [-5.0, 3.0], "x": [1.0, -2.0], "x_lin": [1.0, -2.0], "f": -99.0, "converged": True,
           "verdict": "converged", "pg": 0.0, "evals": 4, "saved": "then"}                # no floors stamp: from before they were shared
    MA._write(MA._path(d, 1), old)
    lines = []
    rep = MA.collect(d, NAMES, 2, log=lines.append)
    assert rep["unfinished"] == [1] and rep["best"] == 0 and any("other noise floors" in m for m in lines)
    fresh_floors.clear_cache()
    new = MA.run_task(1, POINTS, Floored(), BOUNDS, SCALES, d, log=lines.append)
    assert new["floors"] == MA.floors_stamp(d) and new["f"] != -99.0 and new["x_start"] == [-5.0, 3.0]
    assert any("fitting again" in m for m in lines)
    assert MA.collect(d, NAMES, 2, log=lambda m: None)["unfinished"] == []

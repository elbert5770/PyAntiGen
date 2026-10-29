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
    assert any("registry_update.py" in m for m in lines)


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


def test_best_json_is_what_registry_update_reads(tmp_path):
    import importlib.util
    d = str(tmp_path)
    MA.run_task(0, POINTS, Bowl(), BOUNDS, SCALES, d, log=lambda m: None)
    MA.collect(d, NAMES, 1, log=lambda m: None)
    here = os.path.dirname(os.path.abspath(__file__))
    tool = os.path.normpath(os.path.join(here, "..", "..", "..", "Elbert_Esguerra_20260923", "Projects",
                                          "Elbert_Esguerra_20260923", "tools", "registry_update.py"))
    if not os.path.exists(tool):
        pytest.skip("registry_update.py is a project tool, not in this checkout")
    spec = importlib.util.spec_from_file_location("registry_update", tool)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    params, info = mod.read_fit(os.path.join(d, "best.json"))
    assert params == {"a": pytest.approx(1.0, abs=1e-4), "b": pytest.approx(-2.0, abs=1e-4)} and info["success"]


def test_unfinished_list_is_also_written_where_the_submitter_asks(tmp_path, monkeypatch):
    mirror = tmp_path / "state" / "ms.txt"
    monkeypatch.setenv("PYANTIGEN_UNFINISHED_FILE", str(mirror))
    d = str(tmp_path / "ms")
    MA.run_task(0, POINTS, Bowl(), BOUNDS, SCALES, d, log=lambda m: None)
    MA.collect(d, NAMES, 3, log=lambda m: None)
    assert mirror.read_text() == "1,2\n" and open(os.path.join(d, "unfinished.txt")).read() == "1,2\n"

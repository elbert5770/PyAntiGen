"""Every candidate's score and fate is kept, so a waterfall plot can be drawn."""
import glob
import json
import os

import numpy as np
import pytest

from pyantigen.engine import Multistart as M
from pyantigen.engine.Fit_cache import FitCache
from pyantigen.engine.Results import log_optimization_results
from pyantigen.engine.Optimize import (
    _multistart_points,
    _resolve_multistart_limits,
    _run_multistart_stage,
)

FIELDS = {"candidate", "x", "triage_nll", "rank", "relaxed", "feasible", "kept", "cluster",
          "leader", "fitted", "fit_nll"}
SCALES = ["log10", "log10"]
BOUNDS = [(-6.0, 6.0)] * 2


def _two_valley(x_opt):
    x = np.asarray(x_opt, float)
    if x[0] > 1.5:
        return 1e10
    return min(np.sum((x - 1.0) ** 2), 5.0 + np.sum((x + 2.0) ** 2))


def _pts(*cols):
    return 10.0 ** np.array(cols, dtype=float).reshape(-1, 1)


# --- the pure table -------------------------------------------------------------------------

def test_triage_says_which_niche_each_kept_candidate_is_in():
    pos = [0.0, 1.00, 1.05, 2.0, 2.02, 3.0, 4.0, 5.0]
    scores = [50., 10., 11., 12., np.inf, 90., 80., 70.]
    r = M.triage(_pts(*pos), scores, keep_fraction=0.5, cluster_radius=0.2)
    assert r["kept"] == [1, 2, 3] and len(r["cluster_labels"]) == 3
    labels = dict(zip(r["kept"], r["cluster_labels"]))
    assert labels[1] == labels[2] != labels[3]                  # {1, 2} share a niche


def test_the_table_has_one_row_per_candidate_with_every_field():
    pos = [0.0, 1.00, 1.05, 2.0, 2.02, 3.0, 4.0, 5.0]
    scores = np.array([50., 10., 11., 12., np.inf, 90., 80., 70.])
    pts = _pts(*pos)
    rep = M.triage(pts, scores, keep_fraction=0.5, cluster_radius=0.2)
    rows = M.candidate_table(pts, scores, rep, {0: 49.5, 1: 9.0, 3: 11.5})
    assert [r["candidate"] for r in rows] == list(range(8))
    assert all(set(r) == FIELDS for r in rows)
    by = {r["candidate"]: r for r in rows}
    assert by[4]["feasible"] is False and by[4]["triage_nll"] is None and by[4]["rank"] is None
    assert [by[i]["rank"] for i in (1, 2, 3, 0, 7, 6, 5)] == [1, 2, 3, 4, 5, 6, 7]
    assert [i for i in by if by[i]["kept"]] == [1, 2, 3]
    assert by[1]["leader"] and not by[2]["leader"] and by[3]["leader"]
    assert [i for i in by if by[i]["fitted"]] == [0, 1, 3]
    assert (by[0]["fit_nll"], by[1]["fit_nll"], by[3]["fit_nll"]) == (49.5, 9.0, 11.5)
    assert by[2]["fit_nll"] is None and by[2]["cluster"] == by[1]["cluster"]
    assert by[0]["relaxed"] is False and by[1]["relaxed"] is True
    assert by[1]["x"] == [10.0 ** 1.00]


def test_x0_is_never_marked_kept_but_is_always_fitted():
    pts = _pts(0.0, 1.0, 2.0)
    rep = M.triage(pts, [5.0, 1.0, 2.0], keep_fraction=0.5, cluster_radius=0.1)
    x0 = M.candidate_table(pts, [5.0, 1.0, 2.0], rep)[0]
    assert x0["kept"] is False and x0["fitted"] is True and x0["cluster"] is None


def test_a_table_with_every_candidate_failed_still_has_all_rows():
    pts = _pts(0.0, 1.0, 2.0)
    scores = [3.0, np.inf, np.inf]
    rep = M.triage(pts, scores, keep_fraction=0.2, cluster_radius=0.1)
    rows = M.candidate_table(pts, scores, rep)
    assert len(rows) == 3 and [r["feasible"] for r in rows] == [True, False, False]
    assert [r["rank"] for r in rows] == [1, None, None]


# --- from the stage -----------------------------------------------------------------------

def _stage(cache=None, objective=_two_valley, n_candidates=16):
    x0 = np.array([-2.0, -2.0])
    cands = _multistart_points(x0, BOUNDS, SCALES, n_candidates + 1, search_decades=4.0,
                               seed=3, verbose=False, sampler="sobol")
    tri = {**M.TRIAGE_DEFAULTS, "n_candidates": n_candidates, "keep_fraction": 0.5,
           "cluster_radius": 0.3, "parallel": False}
    return cands, _run_multistart_stage(
        objective, cands, x0, BOUNDS, SCALES, "L-BFGS-B", {"options": {"maxiter": 60}}, tri,
        _resolve_multistart_limits({}), n_fits=4, method="Nelder-Mead", verbose=False,
        fit_cache=cache)


def test_the_stage_reports_every_candidate():
    cands, (start, records, report) = _stage()
    rows = report["candidates"]
    assert len(rows) == 17 and all(set(r) == FIELDS for r in rows)
    assert sum(r["feasible"] for r in rows) == report["n_feasible"] + 1      # + x0
    assert sum(not r["feasible"] for r in rows) == report["n_pruned"] > 0    # the wall pruned some
    assert sum(r["kept"] for r in rows) == report["n_kept"]
    assert sorted(r["candidate"] for r in rows if r["fitted"]) == \
        sorted(e["candidate"] for e in report["reference_set"])
    assert report["kept_threshold_nll"] == max(r["triage_nll"] for r in rows if r["kept"])


def test_each_fitted_rows_fit_nll_is_what_its_fit_reached():
    _, (start, records, report) = _stage()
    by = {r["candidate"]: r for r in report["candidates"]}
    for rec in records:
        assert by[rec["candidate"]]["fit_nll"] == rec["fun"]
    assert min(r["fit_nll"] for r in by.values() if r["fit_nll"] is not None) == \
        pytest.approx(report["best_fun"])


def test_the_candidate_positions_are_the_linear_values_the_stage_scored():
    cands, (_, _, report) = _stage()
    for row, c in zip(report["candidates"], cands):
        assert row["x"] == pytest.approx([10.0 ** v for v in c])


def test_the_table_is_plain_json_even_with_failed_candidates():
    _, (_, _, report) = _stage()
    text = json.dumps(report["candidates"], allow_nan=False)     # what the snapshot does
    assert "Infinity" not in text and "NaN" not in text


class _Killed(BaseException):
    pass


def test_an_interrupted_and_resumed_stage_gives_the_same_table(tmp_path):
    cold = _stage()[1][2]["candidates"]

    class Dying:
        def __init__(self):
            self.n = 0

        def __call__(self, x):
            if self.n >= 9:
                raise _Killed()
            self.n += 1
            return _two_valley(x)

    cache = FitCache(str(tmp_path), "tag", "m" * 16, "f" * 16, n_params=2)
    with pytest.raises(_Killed):
        _stage(cache, Dying())
    resumed = _stage(cache)[1][2]
    assert resumed["candidates"] == cold and resumed["launches"] == 2


# --- in the snapshot, and what can be drawn from it --------------------------------------------

def _snapshot(tmp_path, report, starts):
    opt = {"x": np.array([1.0, 2.0]), "fun": 5.0, "success": True, "message": "ok",
           "stats": {}, "x0": [1.0, 2.0], "n_starts": 5, "multistart": report, "starts": starts}
    log_optimization_results(opt, ["a", "b"], str(tmp_path / "res.csv"), model_name="m",
                             experiment_id="e", method="Nelder-Mead")
    (path,) = glob.glob(str(tmp_path / "res_*.json"))
    return json.load(open(path, encoding="utf-8"))


def test_the_snapshot_holds_every_candidate_and_the_parameter_names(tmp_path):
    _, (_, records, report) = _stage()
    snap = _snapshot(tmp_path, report, records)
    ms = snap["multistart"]
    assert ms["param_names"] == ["a", "b"]
    assert len(ms["candidates"]) == 17
    assert ms["candidates"] == json.loads(json.dumps(report["candidates"]))
    assert not os.path.exists(tmp_path / "res.csv")              # still no CSV row


def test_a_waterfall_can_be_drawn_from_the_saved_rows(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _, (_, records, report) = _stage()
    ms = _snapshot(tmp_path, report, records)["multistart"]

    rows = [r for r in ms["candidates"] if r["feasible"]]
    rows.sort(key=lambda r: r["rank"])                           # best to worst
    y = [r["triage_nll"] for r in rows]
    assert y == sorted(y)                                        # the waterfall is monotone
    colour = lambda r: ("tab:red" if r["fitted"] else "tab:orange" if r["kept"] else "0.6")  # noqa: E731
    fig, ax = plt.subplots()
    ax.bar(range(len(rows)), y, color=[colour(r) for r in rows])
    if ms["kept_threshold_nll"] is not None:
        ax.axhline(ms["kept_threshold_nll"], ls="--")
    out = tmp_path / "waterfall.png"
    fig.savefig(out)
    plt.close(fig)
    assert out.stat().st_size > 1000
    # and the pruned ones are there to be shown separately, not silently absent
    assert len(ms["candidates"]) - len(rows) == ms["n_pruned"]

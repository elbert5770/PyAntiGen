"""The multi-start triage: pure steps, tolerance relaxation, and the stage."""
import numpy as np
import pytest

from pyantigen.engine import Multistart as M
from pyantigen.engine import Simulate
from pyantigen.engine.Optimize import (
    _ENGINE_ONLY_OPTIMIZER_KEYS,
    _multistart_points,
    _run_multistart_stage,
)
from pyantigen.engine.Preequil_cache import _digest


# --- settings -----------------------------------------------------------------

def test_defaults_and_validation():
    assert M.resolve_triage({}) == M.TRIAGE_DEFAULTS
    out = M.resolve_triage({"multistart_triage": {"keep_fraction": 0.5}})
    assert out["keep_fraction"] == 0.5 and out["n_candidates"] == 64
    for bad in ({"keep_fracton": 0.5}, {"keep_fraction": 0.0}, {"keep_fraction": 1.5},
                {"n_candidates": 0}, {"cluster_radius": -1}, {"tolerance_factor": 0.5}):
        with pytest.raises(ValueError):
            M.resolve_triage({"multistart_triage": bad})
    assert "multistart_triage" in _ENGINE_ONLY_OPTIMIZER_KEYS


# --- coordinates and clustering ------------------------------------------------

def test_distances_are_in_log10_where_positive_and_raw_otherwise():
    pts = [[1.0, -2.0], [100.0, 3.0], [10.0, 0.0]]
    c = M.log10_coordinates(pts)
    assert np.allclose(c[:, 0], [0.0, 2.0, 1.0])       # log10 of a positive column
    assert np.allclose(c[:, 1], [-2.0, 3.0, 0.0])      # a log-odds is left alone


def test_complete_linkage_does_not_chain():
    # Evenly spaced 0.1 apart: single linkage at 0.15 would make ONE cluster.
    coords = np.arange(0.0, 0.31, 0.1).reshape(-1, 1)
    labels = M.cluster_labels(coords, 0.15)
    assert len(set(labels)) >= 2
    for lab in set(labels):
        span = np.ptp(coords[labels == lab])
        assert span <= 0.15 + 1e-12


def test_cluster_radius_is_rms_per_parameter():
    # Two points 0.3 decades apart in each of 4 parameters: Euclidean 0.6,
    # RMS per parameter 0.3. Together at radius 0.3, apart at 0.29.
    a, b = np.zeros(4), np.full(4, 0.3)
    assert len(set(M.cluster_labels([a, b], 0.30))) == 1
    assert len(set(M.cluster_labels([a, b], 0.29))) == 2


# --- triage --------------------------------------------------------------------

def _pts(*cols):
    """Candidate rows from 1-D log10 positions; x0 is the first."""
    return 10.0 ** np.array(cols, dtype=float).reshape(-1, 1)


def test_triage_prunes_ranks_clusters_and_orders():
    #            x0    a     b     c     d     e     f     g
    pos    = [0.0, 1.00, 1.05, 2.0, 2.02, 3.0, 4.0, 5.0]
    scores = [50., 10., 11., 12., np.inf, 90., 80., 70.]
    r = M.triage(_pts(*pos), scores, keep_fraction=0.5, cluster_radius=0.2)
    assert r["n_candidates"] == 7 and r["n_pruned"] == 1 and r["n_feasible"] == 6
    assert r["n_kept"] == 3                       # best half of 6 feasible
    assert r["kept"] == [1, 2, 3]                 # a, b, c by score
    assert r["n_clusters"] == 2                   # {a, b} share a niche; c alone
    assert r["leaders"] == [1, 3]                 # lowest score of each niche
    assert r["starts"] == [0, 1, 3]               # x0 first, then best first


def test_x0_is_never_pruned_or_ranked_away():
    r = M.triage(_pts(0.0, 1.0, 2.0), [np.inf, 5.0, 6.0], keep_fraction=0.5,
                 cluster_radius=0.1)
    assert r["starts"][0] == 0
    r = M.triage(_pts(0.0, 1.0, 2.0), [999.0, 5.0, 6.0], keep_fraction=0.5,
                 cluster_radius=0.1)
    assert r["starts"][0] == 0


def test_a_leader_inside_x0s_niche_is_not_fitted_twice():
    # Candidate 1 sits 0.05 decade from x0: a duplicate start.
    r = M.triage(_pts(0.0, 0.05, 2.0), [10.0, 5.0, 6.0], keep_fraction=1.0,
                 cluster_radius=0.2)
    assert r["starts"] == [0, 2] and r["n_duplicates_of_x0"] == 1


def test_max_fits_caps_the_reference_set_with_x0_inside_it():
    pos = [0.0] + list(np.arange(1.0, 9.0))
    r = M.triage(_pts(*pos), [50.0] + list(range(1, 9)), keep_fraction=1.0,
                 cluster_radius=0.1, max_fits=3)
    assert r["starts"] == [0, 1, 2]


def test_every_candidate_failing_leaves_just_x0():
    r = M.triage(_pts(0.0, 1.0, 2.0), [1.0, np.inf, np.inf], keep_fraction=0.2,
                 cluster_radius=0.1)
    assert r["starts"] == [0] and r["n_feasible"] == 0 and r["n_kept"] == 0


# --- basins --------------------------------------------------------------------

def test_loose_stops_in_one_valley_are_one_basin_not_multimodal():
    # The case the old rounded-NLL message got wrong: same place, NLL 0.001 apart.
    b, verdict, _ = M.basins([[1.00], [1.01], [1.02]], [184.174, 184.175, 184.260], 0.15)
    assert verdict == "unimodal" and len(b) == 1 and b[0]["members"] == [0, 1, 2]


def test_separate_places_with_equal_nll_are_degenerate_not_multimodal():
    b, verdict, text = M.basins([[1.0], [100.0]], [184.17, 184.30], 0.15)
    assert verdict == "degenerate" and len(b) == 2 and "degenerate" in text


def test_a_clearly_worse_separate_basin_is_multimodal():
    b, verdict, _ = M.basins([[1.0], [100.0]], [184.17, 190.0], 0.15)
    assert verdict == "multimodal" and [x["nll"] for x in b] == [184.17, 190.0]


def test_basins_with_no_finished_fit():
    assert M.basins(np.zeros((0, 1)), [], 0.15)[1] == "none"


# --- tolerance relaxation --------------------------------------------------------

def test_effective_tolerances_default_is_untouched():
    assert Simulate.effective_tolerances({"absolute_tolerance": 1e-8,
                                          "relative_tolerance": 1e-9}) == (1e-8, 1e-9)


def test_relaxation_scales_caps_and_never_tightens():
    s = {"absolute_tolerance": 1e-8, "relative_tolerance": 1e-8}
    with Simulate.relaxed_tolerances(100):
        assert Simulate.effective_tolerances(s) == pytest.approx((1e-6, 1e-6))
    with Simulate.relaxed_tolerances(1e9):
        assert Simulate.effective_tolerances(s) == pytest.approx((1e-4, 1e-4))   # capped
    loose = {"absolute_tolerance": 1e-3, "relative_tolerance": 1e-3}
    with Simulate.relaxed_tolerances(100):
        assert Simulate.effective_tolerances(loose) == (1e-3, 1e-3)              # not tightened


def test_relaxation_restores_and_validates():
    assert Simulate._TOL_RELAX is None
    with Simulate.relaxed_tolerances(10):
        with Simulate.relaxed_tolerances(50):
            assert Simulate._TOL_RELAX[0] == 50
        assert Simulate._TOL_RELAX[0] == 10
    assert Simulate._TOL_RELAX is None
    with pytest.raises(ValueError):
        Simulate.set_tolerance_relaxation(0.5)


def test_configure_integrator_applies_the_effective_tolerances():
    class RR:
        class integrator:                 # noqa: N801 (a stand-in attribute)
            absolute_tolerance = relative_tolerance = None
            variable_step_size = None

            @staticmethod
            def setValue(*a):
                pass

        def setIntegrator(self, name):
            pass

    r = RR()
    s = {"absolute_tolerance": 1e-8, "relative_tolerance": 1e-8}
    with Simulate.relaxed_tolerances(100):
        Simulate.configure_integrator(r, s)
    assert r.integrator.absolute_tolerance == pytest.approx(1e-6)
    assert s["absolute_tolerance"] == 1e-8      # the configured value is not rewritten
    Simulate.configure_integrator(r, s)
    assert r.integrator.absolute_tolerance == pytest.approx(1e-8)


def test_predose_cache_key_separates_relaxed_from_configured():
    state = {"keys": ["A", "B"], "values": [1.0, 2.0]}
    block = {"start": 0, "end": 1, "n_points": 2}
    s = {"integrator": "cvode", "absolute_tolerance": 1e-8, "relative_tolerance": 1e-8,
         "stiff": True, "maximum_num_steps": 20000}
    normal = _digest(state, block, s)
    with Simulate.relaxed_tolerances(100):
        assert _digest(state, block, s) != normal
    assert _digest(state, block, s) == normal


# --- the stage -------------------------------------------------------------------

SCALES = ["log10", "log10"]
BOUNDS = [(-6.0, 6.0)] * 2


def _two_valley(x_opt):
    """Opt-space objective: the global valley at (1, 1) is deeper than the one
    at (-2, -2); a wall (model 'does not integrate') where the first coordinate
    exceeds 1.5, inside the sampled box."""
    x = np.asarray(x_opt, float)
    if x[0] > 1.5:
        return 1e10
    return min(np.sum((x - 1.0) ** 2), 5.0 + np.sum((x + 2.0) ** 2))


def _stage(n_candidates=32, n_fits=4, **tri_over):
    from pyantigen.engine.Optimize import _resolve_multistart_limits
    x0 = np.array([-2.0, -2.0])              # starts in the WORSE valley
    cands = _multistart_points(x0, BOUNDS, SCALES, n_candidates + 1, search_decades=4.0,
                               seed=3, verbose=False, sampler="sobol")
    tri = {**M.TRIAGE_DEFAULTS, "n_candidates": n_candidates, "keep_fraction": 0.25,
           "cluster_radius": 0.5, **tri_over}
    return _run_multistart_stage(
        _two_valley, cands, x0, BOUNDS, SCALES, "L-BFGS-B",
        {"options": {"maxiter": 60}}, tri, _resolve_multistart_limits({}),
        n_fits=n_fits, method="Nelder-Mead", verbose=False)


def test_stage_escapes_the_worse_valley_x0_is_in():
    start, records, report = _stage()
    assert start is not None
    assert np.allclose(start, [1.0, 1.0], atol=0.05)      # the global valley
    assert _two_valley(start) < 1e-3


def test_stage_report_and_records():
    start, records, report = _stage()
    assert report["mode"] == "triage"
    assert report["n_candidates"] == 32
    assert report["n_pruned"] > 0                          # the wall pruned some
    assert report["n_feasible"] == 32 - report["n_pruned"]
    assert report["n_kept"] <= report["n_feasible"]
    assert len(records) == len(report["reference_set"]) <= 4
    assert records[0]["candidate"] == 0                     # x0 is always fitted from
    assert all("triage_nll" in r and r["x"] is not None for r in records)
    assert report["verdict"] in {"unimodal", "degenerate", "multimodal"}
    assert report["verdict"] == "multimodal"               # the two valleys differ by 5 nll
    assert {"triage", "fits"} == set(report["seconds"])


def test_stage_fits_no_more_than_n_fits():
    _, records, report = _stage(n_fits=2)
    assert len(records) <= 2


# --- the fast pass on the worker pool ----------------------------------------------

from pyantigen.engine import Evaluator as E  # noqa: E402


def test_eval_mode_is_applied_around_one_task_and_undone(monkeypatch):
    seen = {}

    def fake_nll(x, frozen_sigmas=None):
        seen["limits"], seen["relax"] = Simulate._RETRY_LIMITS, Simulate._TOL_RELAX
        return 3.5

    monkeypatch.setitem(E._WORKER, "spec", object())
    monkeypatch.setattr(E, "_worker_nll", fake_nll)
    val, status, _ = E._eval_task([0.0], None, {"retry_limits": (4, 0), "relax": 100.0})
    assert (val, status) == (3.5, "ok")
    assert seen["limits"] == (4, 0) and seen["relax"][0] == 100.0
    # The worker goes on to serve profile points: nothing may leak out of the task.
    assert Simulate._RETRY_LIMITS is None and Simulate._TOL_RELAX is None
    E._eval_task([0.0])                                    # no mode: nothing applied
    assert seen["limits"] is None and seen["relax"] is None


def test_eval_mode_is_undone_even_when_the_evaluation_raises(monkeypatch):
    def boom(x, frozen_sigmas=None):
        raise RuntimeError("CVODE")

    monkeypatch.setitem(E._WORKER, "spec", object())
    monkeypatch.setattr(E, "_worker_nll", boom)
    val, status, _ = E._eval_task([0.0], None, {"retry_limits": (4, 0), "relax": 100.0})
    assert val == E.FAILURE_VALUE and status.startswith("error")
    assert Simulate._RETRY_LIMITS is None and Simulate._TOL_RELAX is None


class _FakePool:
    """Stands in for ParallelEvaluator: scores by the objective, records the call."""
    n_workers = 7

    def __init__(self, fail=False):
        self.calls, self.closed, self.fail = [], False, fail

    def evaluate_batch(self, xs, label=None, eval_mode=None, **kw):
        if self.fail:
            raise RuntimeError("worker pool broke")
        self.calls.append((list(xs), label, eval_mode))
        return [_two_valley(x) for x in xs]

    def shutdown(self):
        self.closed = True


def _stage_with(pool_builder, n_candidates=32, parallel=True, **kw):
    from pyantigen.engine.Optimize import _resolve_multistart_limits
    x0 = np.array([-2.0, -2.0])
    cands = _multistart_points(x0, BOUNDS, SCALES, n_candidates + 1, search_decades=4.0,
                               seed=3, verbose=False, sampler="sobol")
    tri = {**M.TRIAGE_DEFAULTS, "n_candidates": n_candidates, "keep_fraction": 0.25,
           "cluster_radius": 0.5, "parallel": parallel}
    return cands, _run_multistart_stage(
        kw.get("objective", _two_valley), cands, x0, BOUNDS, SCALES, "L-BFGS-B",
        {"options": {"maxiter": 60}}, tri, _resolve_multistart_limits({}), n_fits=4,
        method="Nelder-Mead", verbose=False, pool_builder=pool_builder,
        n_workers_hint=kw.get("n_workers_hint", 8))


def test_pool_gets_every_candidate_but_x0_with_the_fast_pass_settings():
    pool = _FakePool()
    cands, (start, records, report) = _stage_with(lambda: pool)
    xs = [x for call in pool.calls for x in call[0]]       # the pool is fed in chunks
    assert len(xs) == 32 and not any(np.array_equal(x, cands[0]) for x in xs)
    assert all(call[2] == {"retry_limits": (4, 0), "relax": 100.0} for call in pool.calls)
    assert len(pool.calls) > 1 and all(len(call[0]) <= 14 for call in pool.calls)
    assert pool.closed                                    # closed as soon as it is done
    assert report["triage_workers"] == 7
    assert np.allclose(start, [1.0, 1.0], atol=0.05)      # same answer as serial


def test_pool_and_serial_fast_passes_agree():
    _, (s_pool, r_pool, rep_pool) = _stage_with(lambda: _FakePool())
    _, (s_ser, r_ser, rep_ser) = _stage_with(None)
    assert [r["candidate"] for r in r_pool] == [r["candidate"] for r in r_ser]
    assert rep_pool["n_pruned"] == rep_ser["n_pruned"] and rep_ser["triage_workers"] == 1


def test_a_broken_pool_falls_back_to_serial_and_is_still_closed():
    pool = _FakePool(fail=True)
    _, (start, records, report) = _stage_with(lambda: pool)
    assert pool.closed and report["triage_workers"] == 1
    assert np.allclose(start, [1.0, 1.0], atol=0.05)


def test_no_pool_available_is_just_serial():
    _, (start, records, report) = _stage_with(lambda: None)
    assert report["triage_workers"] == 1 and start is not None


def test_x0_is_scored_before_the_pool_is_built():
    order = []

    def objective(x):
        order.append("objective")
        return _two_valley(x)

    def builder():
        order.append("pool built")
        return _FakePool()

    from pyantigen.engine.Optimize import _resolve_multistart_limits
    x0 = np.array([-2.0, -2.0])
    cands = _multistart_points(x0, BOUNDS, SCALES, 9, search_decades=4.0, seed=3,
                               verbose=False, sampler="sobol")
    tri = {**M.TRIAGE_DEFAULTS, "n_candidates": 8, "parallel": True}
    _run_multistart_stage(objective, cands, x0, BOUNDS, SCALES, "L-BFGS-B",
                          {"options": {"maxiter": 5}}, tri, _resolve_multistart_limits({}),
                          n_fits=2, method="Nelder-Mead", verbose=False, pool_builder=builder)
    assert order[0] == "objective" and order[1] == "pool built"


def test_parallel_setting_is_validated():
    assert M.resolve_triage({})["parallel"] == "auto"
    assert M.resolve_triage({"multistart_triage": {"parallel": 0}})["parallel"] is False
    assert M.resolve_triage({"multistart_triage": {"parallel": True}})["parallel"] is True
    assert M.resolve_triage({"multistart_triage": {"parallel": "AUTO"}})["parallel"] == "auto"
    with pytest.raises(ValueError, match="auto"):
        M.resolve_triage({"multistart_triage": {"parallel": "sometimes"}})


# --- choosing the pool automatically ---------------------------------------------------

def test_prefer_pool_follows_the_measured_cases():
    # PK Aducanumab, 8 workers: ~6 s warm, x0's first evaluation ~60 s cold.
    assert M.prefer_pool(31, 60.0, 6.0, 8)[0] is False        # 32 candidates: near tie
    assert M.prefer_pool(7, 60.0, 6.0, 8)[0] is False         # 8 candidates: pool loses
    # silk_appfull-like: ~21 s warm, a long cold start, many candidates.
    use, serial, pool = M.prefer_pool(63, 120.0, 21.0, 24)
    assert use is True and serial > 1.5 * pool
    assert M.prefer_pool(63, 120.0, 21.0, 1)[0] is False      # no workers to use
    assert M.prefer_pool(0, 60.0, 6.0, 8) == (False, 0.0, 0.0)


def test_prefer_pool_charges_a_start_up_floor_for_a_warm_parent():
    _, _, pool = M.prefer_pool(100, 0.1, 1.0, 100)
    assert pool >= M.POOL_STARTUP_FLOOR_S


def _clocked(per_eval, cold):
    """An objective that advances a fake clock: the first call is `cold` seconds,
    every later one `per_eval`."""
    state = {"t": 0.0, "n": 0}

    def clock():
        return state["t"]

    def objective(x):
        state["t"] += cold if state["n"] == 0 else per_eval
        state["n"] += 1
        return _two_valley(x)

    return clock, objective


def test_auto_picks_serial_when_evaluations_are_cheap(monkeypatch):
    from pyantigen.engine import Optimize
    clock, objective = _clocked(per_eval=1.0, cold=10.0)
    monkeypatch.setattr(Optimize, "_now", clock)
    pool = _FakePool()
    _, (start, records, report) = _stage_with(lambda: pool, parallel="auto",
                                              objective=objective)
    assert pool.calls == [] and report["triage_workers"] == 1
    d = report["triage_decision"]
    assert d["parallel"] == "auto" and d["used_pool"] is False
    assert d["t_x0_s"] == 10.0 and d["t_eval_s"] == 1.0
    assert d["serial_estimate_s"] < 1.5 * d["pool_estimate_s"]


def test_auto_picks_the_pool_when_evaluations_are_expensive(monkeypatch):
    from pyantigen.engine import Optimize
    clock, objective = _clocked(per_eval=40.0, cold=60.0)
    monkeypatch.setattr(Optimize, "_now", clock)
    pool = _FakePool()
    cands, (start, records, report) = _stage_with(lambda: pool, parallel="auto",
                                                  objective=objective)
    xs = [x for call in pool.calls for x in call[0]]
    # x0 and ONE warm candidate were scored here; the pool gets the other 31.
    assert len(xs) == 31
    assert not any(np.array_equal(x, cands[0]) for x in xs)
    assert not any(np.array_equal(x, cands[1]) for x in xs)
    assert report["triage_workers"] == 7 and report["triage_decision"]["used_pool"] is True
    assert pool.closed
    assert np.allclose(start, [1.0, 1.0], atol=0.05)


def test_auto_scores_every_candidate_exactly_once_either_way(monkeypatch):
    from pyantigen.engine import Optimize
    for per_eval in (1.0, 40.0):
        clock, objective = _clocked(per_eval=per_eval, cold=60.0)
        monkeypatch.setattr(Optimize, "_now", clock)
        _, (start, records, report) = _stage_with(lambda: _FakePool(), parallel="auto",
                                                  objective=objective)
        assert report["n_candidates"] == 32
        assert report["n_feasible"] + report["n_pruned"] == 32


def test_false_never_builds_a_pool_and_true_skips_the_timing(monkeypatch):
    built = []
    _stage_with(lambda: built.append(1) or _FakePool(), parallel=False)
    assert built == []
    _, (_, _, report) = _stage_with(lambda: _FakePool(), parallel=True)
    assert "t_eval_s" not in report["triage_decision"]        # no timing probe when forced


# --- what is kept: the fit cache and the result snapshot -------------------------------

import glob  # noqa: E402
import json  # noqa: E402

from pyantigen.engine.Fit_cache import FitCache  # noqa: E402
from pyantigen.engine.Optimize import _load_saved_multistart  # noqa: E402


def test_json_safe_makes_numpy_and_nonfinite_into_plain_json():
    raw = {"a": np.float64(1.5), "b": np.arange(3), "c": (1, np.int64(2)),
           "d": float("inf"), "e": [float("nan"), None, "x"], 1: {"f": np.bool_(True)}}
    safe = M.json_safe(raw)
    assert safe == {"a": 1.5, "b": [0, 1, 2], "c": [1, 2], "d": None,
                    "e": [None, None, "x"], "1": {"f": True}}
    json.dumps(safe, allow_nan=False)                      # what the snapshot does


def _cache(tmp_path):
    return FitCache(str(tmp_path), "tag", "m" * 16, "f" * 16, n_params=2)


def test_the_report_round_trips_through_the_fit_cache(tmp_path):
    c = _cache(tmp_path)
    assert c.load_multistart() is None
    report = {"mode": "triage", "n_candidates": 32, "starts": [{"fun": 1.0, "x": [1, 2]}]}
    assert c.save_multistart(report) is True
    assert _cache(tmp_path).load_multistart() == report    # a fresh process sees it


def test_the_report_survives_the_cache_records_written_after_it(tmp_path):
    c = _cache(tmp_path)
    c.save_multistart({"mode": "triage"})
    c.save_partial([1.0, 2.0], 5.0, n_evals=10, force=True)
    assert c.load_multistart() == {"mode": "triage"}
    c.save_complete([1.0, 2.0], 5.0, param_names=["a", "b"])
    assert c.load_multistart() == {"mode": "triage"}
    assert c.load_complete() is not None and c.load_partial() is None


def test_a_cache_holding_only_a_report_is_not_a_finished_fit(tmp_path):
    c = _cache(tmp_path)
    c.save_multistart({"mode": "triage"})
    assert c.load_complete() is None and c.load_partial() is None


def test_a_disabled_cache_keeps_nothing():
    c = FitCache(None, "tag", "m" * 16, "f" * 16, n_params=2, enabled=False)
    assert c.save_multistart({"x": 1}) is False and c.load_multistart() is None


def test_a_saved_report_comes_back_marked_as_not_from_this_process(tmp_path):
    c = _cache(tmp_path)
    c.save_multistart({"mode": "triage", "verdict": "unimodal", "starts": [{"fun": 2.0}]})
    report, records = _load_saved_multistart(_cache(tmp_path))
    assert report["from_cache"] is True and report["verdict"] == "unimodal"
    assert "starts" not in report and records == [{"fun": 2.0}]
    assert _load_saved_multistart(None) == (None, None)
    assert _load_saved_multistart(_cache(tmp_path / "empty")) == (None, None)


def _snapshot(tmp_path, **opt_extra):
    from pyantigen.engine.Results import log_optimization_results
    opt = {"x": np.array([1.0, 2.0]), "fun": 5.0, "success": True, "message": "ok",
           "stats": {}, "x0": [1.0, 2.0], **opt_extra}
    # A folder per call: the snapshot's file name has one-second resolution, so
    # two calls in one folder can leave two files and make the glob ambiguous.
    here = tmp_path / f"run{len(list(tmp_path.iterdir()))}"
    here.mkdir()
    csv = here / "res.csv"
    log_optimization_results(opt, ["a", "b"], str(csv), model_name="m",
                             experiment_id="e", method="Nelder-Mead")
    (path,) = glob.glob(str(here / "res_*.json"))
    return json.load(open(path, encoding="utf-8"))


def test_the_snapshot_carries_the_multistart_report_and_its_fits(tmp_path):
    report = {"mode": "triage", "n_candidates": 32, "n_pruned": 2, "verdict": "unimodal",
              "basins": [{"nll": 1.0, "members": [0, 1], "x": [1.0, 2.0]}],
              "triage_decision": {"parallel": "auto", "used_pool": False}}
    starts = [{"fun": 5.0, "x": [1.0, 2.0], "candidate": 0, "triage_nll": None},
              {"fun": None, "x": None, "candidate": 7, "triage_nll": float("inf")}]
    snap = _snapshot(tmp_path, multistart=report, starts=starts)
    ms = snap["multistart"]
    assert ms["n_pruned"] == 2 and ms["verdict"] == "unimodal"
    assert ms["triage_decision"]["used_pool"] is False
    assert [s["candidate"] for s in ms["starts"]] == [0, 7]
    assert ms["starts"][1]["triage_nll"] is None           # inf became null, file still valid


def test_a_single_start_snapshot_has_no_multistart_block(tmp_path):
    assert "multistart" not in _snapshot(tmp_path)
    assert "multistart" not in _snapshot(tmp_path, multistart=None, starts=None)


# --- what survives being killed ----------------------------------------------------------

from pyantigen.engine.Optimize import _persist_stage_result  # noqa: E402


def test_the_stage_reports_its_best_point_for_the_caller_to_keep():
    start, records, report = _stage()
    assert report["best_fun"] == pytest.approx(_two_valley(start), abs=1e-6)
    assert np.allclose(report["best_x"], 10.0 ** np.asarray(start))     # linear units


def test_persisting_the_stage_makes_its_best_the_fits_start_and_x0_the_calibration(tmp_path):
    c = _cache(tmp_path)
    report = {"mode": "triage", "best_fun": 184.17, "verdict": "unimodal"}
    staged_opt = np.array([0.3, -1.0])                  # opt space; both log10
    x0_lin = np.array([2.0, 5.0])
    _persist_stage_result(c, report, [{"fun": 184.17}], staged_opt, x0_lin, ["log10", "log10"])
    part = c.load_partial()
    assert np.allclose(part["x_lin"], 10.0 ** staged_opt) and part["fun"] == 184.17
    assert part["n_evals"] == 0 and part["nm_state"] is None
    # Noise floors were calibrated at x0 -- the first thing the stage scores.
    assert np.allclose(part["cal_x_lin"], x0_lin)
    assert c.load_multistart()["verdict"] == "unimodal"
    assert c.load_multistart()["starts"] == [{"fun": 184.17}]
    assert c.load_complete() is None


def test_a_relaunch_after_the_stage_resumes_from_its_best_not_from_x0(tmp_path):
    c = _cache(tmp_path)
    _persist_stage_result(c, {"best_fun": 5.0}, [], np.array([0.0, 0.0]),
                          np.array([9.0, 9.0]), ["log10", "log10"])
    fresh = _cache(tmp_path)                             # the relaunched process
    assert fresh.load_partial() is not None              # -> the stage is skipped
    assert np.allclose(fresh.load_partial()["x_lin"], [1.0, 1.0])
    report, records = _load_saved_multistart(fresh)
    assert report["from_cache"] is True and report["best_fun"] == 5.0


def test_persisting_without_a_best_point_keeps_only_the_report(tmp_path):
    c = _cache(tmp_path)
    _persist_stage_result(c, {"best_fun": None}, [], None, np.array([1.0, 1.0]), ["lin", "lin"])
    assert c.load_partial() is None and c.load_multistart() is not None
    _persist_stage_result(c, None, None, np.array([0.0, 0.0]), np.array([1.0, 1.0]), ["lin", "lin"])
    _persist_stage_result(None, {"best_fun": 1.0}, [], np.array([0.0]), np.array([1.0]), ["lin"])


def test_the_first_calibration_point_survives_the_fits_later_writes(tmp_path):
    c = _cache(tmp_path)
    _persist_stage_result(c, {"best_fun": 5.0}, [], np.array([0.0, 0.0]),
                          np.array([7.0, 8.0]), ["log10", "log10"])
    c.save_partial([2.0, 2.0], 4.0, n_evals=10, force=True, cal_x_lin=[2.0, 2.0])
    assert np.allclose(c.load_partial()["cal_x_lin"], [7.0, 8.0])        # first writer wins

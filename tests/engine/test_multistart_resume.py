"""The multi-start stage survives being killed: what it keeps, and what a relaunch skips."""
import numpy as np
import pytest

from pyantigen.engine import Multistart as M
from pyantigen.engine.Deadline import DeadlineReached, RunBudget
from pyantigen.engine.Fit_cache import FitCache
from pyantigen.engine.Optimize import (
    _multistart_points,
    _persist_stage_result,
    _resolve_multistart_limits,
    _run_multistart,
    _run_multistart_stage,
)

SCALES = ["log10", "log10"]
BOUNDS = [(-6.0, 6.0)] * 2


def _two_valley(x_opt):
    """Global valley (1, 1), a worse one at (-2, -2), a wall past x[0] = 1.5."""
    x = np.asarray(x_opt, float)
    if x[0] > 1.5:
        return 1e10
    return min(np.sum((x - 1.0) ** 2), 5.0 + np.sum((x + 2.0) ** 2))


def _cache(tmp_path):
    return FitCache(str(tmp_path), "tag", "m" * 16, "f" * 16, n_params=2)


class _Killed(BaseException):
    """A kill. BaseException so that nothing in the stage's own ``except Exception``
    handling can swallow it, as a SIGKILL cannot be."""


class _Counting:
    """_two_valley, counting calls, and optionally dying after ``die_after`` of them."""

    def __init__(self, die_after=None):
        self.calls, self.die_after = 0, die_after

    def __call__(self, x):
        if self.die_after is not None and self.calls >= self.die_after:
            raise _Killed()
        self.calls += 1
        return _two_valley(x)


class _FakePool:
    n_workers = 7

    def __init__(self):
        self.calls, self.closed = [], False

    def evaluate_batch(self, xs, label=None, eval_mode=None, **kw):
        self.calls.append((list(xs), label, eval_mode))
        return [_two_valley(x) for x in xs]

    def shutdown(self):
        self.closed = True


def _stage(cache, objective, pool_builder=None, budget=None, n_candidates=16,
           parallel=False, n_fits=4):
    x0 = np.array([-2.0, -2.0])
    cands = _multistart_points(x0, BOUNDS, SCALES, n_candidates + 1, search_decades=4.0,
                               seed=3, verbose=False, sampler="sobol")
    tri = {**M.TRIAGE_DEFAULTS, "n_candidates": n_candidates, "keep_fraction": 0.5,
           "cluster_radius": 0.3, "parallel": parallel}
    return _run_multistart_stage(
        objective, cands, x0, BOUNDS, SCALES, "L-BFGS-B", {"options": {"maxiter": 60}}, tri,
        _resolve_multistart_limits({}), n_fits=n_fits, method="Nelder-Mead", verbose=False,
        pool_builder=pool_builder, fit_cache=cache, budget=budget)


def _outcome(result):
    start, records, report = result
    return (np.round(start, 9).tolist(),
            [(r["candidate"], r["fun"], r["x"]) for r in records],
            report["reference_set"], report["n_pruned"], report["verdict"])


def _fits_start_after(n_evals):
    """How many objective calls the fast pass takes, to aim a kill at the fits."""
    c = _Counting()
    _stage(None, c, n_candidates=16)
    return c.calls


# --- the checkpoint ---------------------------------------------------------------------

def test_a_stage_with_a_cache_gives_the_same_answer_as_one_without(tmp_path):
    assert _outcome(_stage(None, _Counting())) == _outcome(_stage(_cache(tmp_path), _Counting()))


def test_the_stage_keeps_a_checkpoint_while_it_runs(tmp_path):
    c = _cache(tmp_path)
    with pytest.raises(_Killed):
        _stage(c, _Counting(die_after=9))                       # x0 and 8 candidates
    st = c.load_stage()
    assert st["version"] == 1 and st["launches"] == 1
    assert set(st["scores"]) == {str(i) for i in range(9)}
    assert st["fits"] == {}


def test_the_stage_never_writes_the_fits_partial(tmp_path):
    c = _cache(tmp_path)
    with pytest.raises(_Killed):
        _stage(c, _Counting(die_after=40))
    assert c.load_partial() is None and c.load_complete() is None


# --- what a relaunch skips ----------------------------------------------------------------

def test_a_kill_in_the_fast_pass_costs_only_what_was_in_flight(tmp_path):
    c = _cache(tmp_path)
    ref = _outcome(_stage(None, _Counting()))
    with pytest.raises(_Killed):
        _stage(c, _Counting(die_after=9))
    again = _Counting()
    resumed = _stage(c, again)
    assert _outcome(resumed) == ref                              # the same answer
    full = _Counting()
    _stage(None, full)
    assert again.calls == full.calls - 8                         # not the 8 already scored
    assert resumed[2]["launches"] == 2
    assert resumed[2]["resumed"]["candidates_scored_in_earlier_launches"] == 8


def test_a_kill_between_fits_resumes_with_finished_fits_as_they_were(tmp_path):
    c = _cache(tmp_path)
    ref = _outcome(_stage(None, _Counting()))
    n_fast = 1 + 16                                              # x0 + every candidate
    with pytest.raises(_Killed):
        _stage(c, _Counting(die_after=n_fast + 25))              # a little way into the fits
    st = c.load_stage()
    assert set(st["scores"]) == {str(i) for i in range(17)}      # the fast pass had finished
    done = len(st["fits"])
    resumed = _stage(c, _Counting())
    assert _outcome(resumed) == ref
    assert resumed[2]["resumed"]["fits_finished_in_earlier_launches"] == done


def test_x0_is_scored_first_in_the_new_process_even_when_recorded(tmp_path):
    c = _cache(tmp_path)
    with pytest.raises(_Killed):
        _stage(c, _Counting(die_after=4))
    order = []

    def objective(x):
        order.append(tuple(np.round(x, 6)))
        return _two_valley(x)

    _stage(c, objective)
    assert order[0] == (-2.0, -2.0)


def test_a_changed_x0_score_discards_the_checkpoint(tmp_path):
    c = _cache(tmp_path)
    with pytest.raises(_Killed):
        _stage(c, _Counting(die_after=6))
    shifted = lambda x: _two_valley(x) + 5.0                     # noqa: E731
    _, _, report = _stage(c, shifted)
    assert report["launches"] == 1 and "resumed" not in report   # started over


def test_a_checkpoint_for_a_different_stage_is_not_used(tmp_path):
    c = _cache(tmp_path)
    with pytest.raises(_Killed):
        _stage(c, _Counting(die_after=6), n_candidates=16)
    _, _, report = _stage(c, _Counting(), n_candidates=8)        # other candidates
    assert report["launches"] == 1 and "resumed" not in report


def test_nothing_left_to_score_builds_no_pool(tmp_path):
    c = _cache(tmp_path)
    with pytest.raises(_Killed):
        _stage(c, _Counting(die_after=1 + 16 + 5))               # the fast pass is done
    built = []
    _, _, report = _stage(c, _Counting(), parallel=True,
                          pool_builder=lambda: built.append(1) or _FakePool())
    assert built == [] and "skipped" in report["triage_decision"]


# --- the pool ------------------------------------------------------------------------------

def test_a_pooled_fast_pass_resumes_chunk_by_chunk(tmp_path):
    c = _cache(tmp_path)

    class DyingPool(_FakePool):
        def evaluate_batch(self, xs, label=None, eval_mode=None, **kw):
            if len(self.calls) == 1:
                raise _Killed()                                  # the second chunk dies
            return super().evaluate_batch(xs, label=label, eval_mode=eval_mode)

    pool = DyingPool()
    with pytest.raises(_Killed):
        _stage(c, _Counting(), pool_builder=lambda: pool, parallel=True, n_candidates=32)
    first = len(pool.calls[0][0])
    assert len(c.load_stage()["scores"]) == 1 + first            # x0 + the first chunk

    pool2 = _FakePool()
    _, _, report = _stage(c, _Counting(), pool_builder=lambda: pool2, parallel=True,
                          n_candidates=32)
    assert sum(len(call[0]) for call in pool2.calls) == 32 - first
    assert report["resumed"]["candidates_scored_in_earlier_launches"] == first


# --- stop times ----------------------------------------------------------------------------

def test_a_stop_time_saves_and_raises_and_the_next_launch_continues(tmp_path):
    c = _cache(tmp_path)
    expired = RunBudget(deadline=1.0, margin_s=0.0)              # long past
    with pytest.raises(DeadlineReached) as exc:
        _stage(c, _Counting(), budget=expired)
    assert exc.value.label == "multi-start fast pass"
    assert set(c.load_stage()["scores"]) == {"0"}                # x0 is scored; then it stops
    _, _, report = _stage(c, _Counting(), budget=RunBudget())
    assert report["launches"] == 2


def test_a_stop_time_between_fits_keeps_the_fits_already_finished(tmp_path):
    c = _cache(tmp_path)

    class OutOfTimeOnceAFitIsKept(RunBudget):
        def __init__(self):
            super().__init__(deadline=None)

        @property
        def is_limited(self):
            st = c.load_stage()
            return bool(st and st["fits"])

        def work_deadline(self):
            return 0.0

    with pytest.raises(DeadlineReached) as exc:
        _stage(c, _Counting(), budget=OutOfTimeOnceAFitIsKept())
    assert exc.value.label == "multi-start fits"
    assert len(c.load_stage()["fits"]) == 1
    _, records, report = _stage(c, _Counting())
    assert report["resumed"]["fits_finished_in_earlier_launches"] == 1 and len(records) >= 2


# --- _run_multistart, and what the caller keeps ------------------------------------------------

def test_run_multistart_reuses_recorded_fits_and_reports_each_new_one():
    starts = [np.array([-2.0, -2.0]), np.array([0.5, 0.5])]
    kw = dict(screen=False, verbose=False)
    opts = {"options": {"maxiter": 60}}
    full = _Counting()
    best, recs, _ = _run_multistart(full, starts, "L-BFGS-B", BOUNDS, opts, SCALES, **kw)
    seen, partial = [], _Counting()
    best2, recs2, _ = _run_multistart(partial, starts, "L-BFGS-B", BOUNDS, opts, SCALES,
                                      resume=[recs[0], None],
                                      on_fit=lambda i, r: seen.append(i), **kw)
    assert seen == [1]                                           # only the new fit is reported
    assert recs2[0] == recs[0] and partial.calls < full.calls
    assert best2.fun == pytest.approx(best.fun)


def test_the_stage_checkpoint_is_cleared_once_its_result_is_on_record(tmp_path):
    c = _cache(tmp_path)
    start, records, report = _stage(c, _Counting())
    assert c.load_stage() is not None                            # kept until persisted
    _persist_stage_result(c, report, records, start, np.array([0.01, 0.01]), SCALES)
    assert c.load_stage() is None and c.load_partial() is not None
    assert c.load_multistart()["verdict"] == report["verdict"]


def test_the_checkpoint_stays_when_no_fit_finished(tmp_path):
    c = _cache(tmp_path)
    c.save_stage({"version": 1, "scores": {}, "fits": {}})
    _persist_stage_result(c, {"best_fun": None}, [], None, np.array([1.0, 1.0]), ["lin", "lin"])
    assert c.load_stage() is not None


def test_stage_checkpoint_round_trips_and_survives_other_cache_writes(tmp_path):
    c = _cache(tmp_path)
    assert c.load_stage() is None and c.clear_stage() is False
    c.save_stage({"version": 1, "x": [1, 2]})
    c.save_multistart({"mode": "triage"})
    assert _cache(tmp_path).load_stage() == {"version": 1, "x": [1, 2]}
    assert c.clear_stage() is True and c.load_stage() is None
    assert c.load_multistart() == {"mode": "triage"}


def test_seconds_accumulate_across_launches(tmp_path):
    c = _cache(tmp_path)
    with pytest.raises(_Killed):
        _stage(c, _Counting(die_after=40))
    first = c.load_stage()["seconds"]
    _, _, report = _stage(c, _Counting())
    assert report["seconds"]["triage"] >= round(first["triage"], 1)
    assert set(report["seconds"]) == {"triage", "fits"}

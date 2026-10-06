"""The local fits' gradient: finite differences taken as ONE batch, on the pool when it pays."""
import numpy as np
import pytest
from scipy.optimize import minimize

from pyantigen.engine import Multistart as M
from pyantigen.engine.Fit_cache import FitCache, fit_fingerprint
from pyantigen.engine.Optimize import (
    _ENGINE_ONLY_OPTIMIZER_KEYS,
    _multistart_points,
    _resolve_multistart_limits,
    _run_multistart_stage,
)

FD = M.FDObjective


def _quad(x):
    """f = sum (x_i - i)^2 ; gradient 2 (x_i - i)."""
    x = np.asarray(x, float)
    return float(np.sum((x - np.arange(len(x))) ** 2))


def _true_grad(x):
    x = np.asarray(x, float)
    return 2.0 * (x - np.arange(len(x)))


class _Batches:
    """A batch evaluator that records the batches it is handed."""

    def __init__(self, fn=_quad):
        self.fn, self.batches = fn, []

    def __call__(self, pts):
        self.batches.append([np.array(p) for p in pts])
        return [self.fn(p) for p in pts]


# --- settings -------------------------------------------------------------------------------

def test_gradient_defaults_and_validation():
    assert M.resolve_gradient({}) == M.GRADIENT_DEFAULTS
    out = M.resolve_gradient({"multistart_gradient": {"step": 1e-2, "scheme": "CENTRAL", "parallel": 0}})
    assert out == {"parallel": False, "step": 1e-2, "scheme": "central"}
    for bad in ({"stp": 1e-3}, {"step": 0}, {"step": -1}, {"scheme": "backward"}, {"parallel": "maybe"}):
        with pytest.raises(ValueError):
            M.resolve_gradient({"multistart_gradient": bad})


def test_gradient_key_is_engine_only_and_leaves_a_one_start_fingerprint_alone():
    assert "multistart_gradient" in _ENGINE_ONLY_OPTIMIZER_KEYS
    kw = {"options": {"maxiter": 5}}
    g = {"multistart_gradient": {"step": 1e-2}}

    def fp(extra, n):
        return fit_fingerprint(["a"], [1.0], [(0.1, 10.0)], ["log10"], {"A": {}}, "m",
                               "Nelder-Mead", {**kw, **extra}, n_starts=n)

    assert fp({}, 1) == fp(g, 1)
    assert fp({}, 5) != fp(g, 5)


def test_only_gradient_methods_are_in_the_set():
    assert "l-bfgs-b" in M.GRADIENT_METHODS
    assert "nelder-mead" not in M.GRADIENT_METHODS and "powell" not in M.GRADIENT_METHODS


# --- FDObjective: the maths -----------------------------------------------------------------------

def test_forward_difference_is_one_batch_of_k_plus_one_points_x_first():
    b = _Batches()
    f, g = FD(b, [(-9, 9)] * 3, ["log10"] * 3, step=1e-3)(np.array([0.5, 1.5, 2.5]))
    assert len(b.batches) == 1 and len(b.batches[0]) == 4
    assert np.allclose(b.batches[0][0], [0.5, 1.5, 2.5])             # x itself first
    assert f == pytest.approx(_quad([0.5, 1.5, 2.5]))
    assert np.allclose(g, _true_grad([0.5, 1.5, 2.5]), atol=2e-3)      # O(step) error


def test_central_difference_is_exact_on_a_quadratic_in_two_k_plus_one_points():
    b = _Batches()
    f, g = FD(b, [(-9, 9)] * 3, ["log10"] * 3, step=1e-2, scheme="central")(np.array([0.5, 1.5, 2.5]))
    assert len(b.batches) == 1 and len(b.batches[0]) == 7
    assert np.allclose(g, _true_grad([0.5, 1.5, 2.5]), atol=1e-9)


def test_points_per_call():
    assert FD(_Batches(), [(0, 1)] * 5, ["lin"] * 5).points_per_call() == 6
    assert FD(_Batches(), [(0, 1)] * 5, ["lin"] * 5, scheme="central").points_per_call() == 11


def test_a_log10_parameter_steps_by_decades_and_a_linear_one_relatively():
    b = _Batches()
    FD(b, [(-1e9, 1e9)] * 2, ["log10", "lin"], step=1e-3)(np.array([3.0, 100.0]))
    pts = b.batches[0]
    assert pts[1][0] - 3.0 == pytest.approx(1e-3) and pts[1][1] == 100.0     # decades, absolute
    assert pts[2][1] - 100.0 == pytest.approx(1e-3 * 100.0)                   # |x| = 100, relative
    b2 = _Batches()
    FD(b2, [(-1e9, 1e9)], ["lin"], step=1e-3)(np.array([0.0]))
    assert b2.batches[0][1][0] == pytest.approx(1e-3)                         # max(|x|, 1)


def test_a_step_that_would_leave_the_bounds_is_taken_the_other_way():
    b = _Batches()
    x = np.array([2.0])
    f, g = FD(b, [(0.0, 2.0)], ["log10"], step=1e-3)(x)                  # x at the upper bound
    assert b.batches[0][1][0] == pytest.approx(2.0 - 1e-3)               # backward
    assert g[0] == pytest.approx(2.0 * (2.0 - 0.0), abs=2e-3)            # d/dx (x-0)^2 = 4: right sign
    assert all(0.0 <= p[0] <= 2.0 for p in b.batches[0])


def test_central_at_a_bound_falls_back_to_the_one_side_inside():
    b = _Batches()
    f, g = FD(b, [(0.0, 2.0)], ["log10"], step=1e-3, scheme="central")(np.array([2.0]))
    assert len(b.batches[0]) == 2 and g[0] == pytest.approx(4.0, abs=2e-3)


def test_a_failed_perturbed_point_is_retried_on_the_other_side():
    def fn(p):
        return 1e10 if p[0] > 1.0005 else _quad(p)                        # a wall just above x = 1

    b = _Batches(fn)
    fd = FD(b, [(-9, 9)], ["log10"], step=1e-3)
    f, g = fd(np.array([1.0 + 1e-4]))
    assert len(b.batches) == 2 and len(b.batches[1]) == 1                  # one retry batch
    assert b.batches[1][0][0] < 1.0 + 1e-4                                  # on the other side
    assert g[0] == pytest.approx(_true_grad([1.0 + 1e-4])[0], abs=2e-3)    # not a cliff
    assert fd.n_retried == 1 and fd.n_zeroed == 0
    assert abs(g[0]) < 10.0                                                 # certainly not ~1e13


def test_when_neither_side_evaluates_the_component_is_zero():
    def fn(p):
        return _quad(p) if abs(p[0] - 1.0) < 1e-12 else 1e10                # only x itself integrates

    b = _Batches(fn)
    fd = FD(b, [(-9, 9)], ["log10"], step=1e-3)
    f, g = fd(np.array([1.0]))
    assert f == pytest.approx(_quad([1.0])) and g[0] == 0.0 and fd.n_zeroed == 1


def test_a_failed_x_returns_the_penalty_and_a_zero_gradient_without_a_retry_batch():
    b = _Batches(lambda p: 1e10)
    fd = FD(b, [(-9, 9)] * 2, ["log10"] * 2, penalty=1234.5)
    f, g = fd(np.array([0.0, 0.0]))
    assert f == 1234.5 and np.all(g == 0.0) and len(b.batches) == 1
    f2, _ = FD(_Batches(lambda p: 1e10), [(-9, 9)], ["log10"])(np.array([0.0]))
    assert f2 == 1e10                                                       # no penalty set


def test_non_finite_values_count_as_failures():
    b = _Batches(lambda p: float("nan") if p[0] > 0.0005 else _quad(p))
    f, g = FD(b, [(-9, 9)], ["log10"], step=1e-3)(np.array([0.0]))
    assert np.isfinite(g[0])


def test_counters():
    b = _Batches()
    fd = FD(b, [(-9, 9)] * 4, ["log10"] * 4)
    fd(np.zeros(4))
    fd(np.ones(4))
    assert fd.n_calls == 2 and fd.n_evals == 10 and fd.last_f == pytest.approx(_quad(np.ones(4)))


def test_a_noise_floor_swamps_scipys_default_step_and_not_ours():
    """Why the step matters: solver noise ~1e-7 divided by 1e-8 is a gradient of ~10
    where the truth is ~0.2; divided by 1e-3 it is ~1e-4."""
    def noisy(p):
        p = np.asarray(p, float)
        return float(np.sum((p - 1.0) ** 2) + 2e-7 * np.sin(3e9 * np.sum(p)))

    x = np.array([1.05, 0.95, 1.1])
    truth = 2.0 * (x - 1.0)
    naive = np.array([(noisy(x + 1e-8 * e) - noisy(x)) / 1e-8 for e in np.eye(3)])
    ours = FD(lambda pts: [noisy(p) for p in pts], [(-9, 9)] * 3, ["log10"] * 3, step=1e-3)(x)[1]
    assert np.max(np.abs(naive - truth)) > 1.0
    assert np.max(np.abs(ours - truth)) < 0.02


def test_lbfgsb_with_the_batch_gradient_reaches_the_optimum_through_noise():
    def noisy(p):
        p = np.asarray(p, float)
        return float(np.sum((p - np.array([1.0, -2.0, 0.5])) ** 2) + 1e-7 * np.sin(5e8 * np.sum(p)))

    fdo = FD(lambda pts: [noisy(p) for p in pts], [(-6, 6)] * 3, ["log10"] * 3, step=1e-3)
    r = minimize(fdo, np.array([0.0, 0.0, 0.0]), jac=True, method="L-BFGS-B",
                 bounds=[(-6, 6)] * 3, options={"maxiter": 50})
    assert np.allclose(r.x, [1.0, -2.0, 0.5], atol=5e-3) and r.fun < 1e-4


# --- pool or serial -----------------------------------------------------------------------------------

def test_prefer_pool_gradient_follows_the_measured_cases():
    # silk_appfull: 18 parameters, 22 s an evaluation, 24 workers, 40 iterations.
    use, serial, pool = M.prefer_pool_gradient(40, 19, 80.0, 22.0, 23)
    assert use and serial == pytest.approx(40 * 19 * 22.0) and pool == pytest.approx(160.0 + 40 * 22.0)
    # cook_gsi_ki: one parameter, a gradient is two points wide, so the pool cannot win.
    assert M.prefer_pool_gradient(12, 2, 18.0, 6.0, 23)[0] is False
    # a 3-parameter fit on 8 workers: width 4, a 4x ceiling, but the start-up eats it.
    assert M.prefer_pool_gradient(3, 4, 60.0, 6.0, 8)[0] is False


def test_prefer_pool_gradient_charges_no_start_up_for_a_pool_that_is_up():
    cold = M.prefer_pool_gradient(5, 19, 80.0, 22.0, 23)
    ready = M.prefer_pool_gradient(5, 19, 80.0, 22.0, 23, pool_ready=True)
    assert ready[2] == pytest.approx(5 * 22.0) and ready[2] < cold[2]


def test_prefer_pool_gradient_with_nothing_to_gain():
    assert M.prefer_pool_gradient(0, 19, 80.0, 22.0, 23) == (False, 0.0, 0.0)
    assert M.prefer_pool_gradient(40, 19, 80.0, 22.0, 1)[0] is False
    assert M.prefer_pool_gradient(40, 1, 80.0, 22.0, 23)[0] is False


def test_the_pool_ceiling_is_the_gradient_width():
    # 19 points on 4 workers is 5 rounds, so at best ~3.8x however many iterations.
    _, serial, pool = M.prefer_pool_gradient(1000, 19, 80.0, 22.0, 4, pool_ready=True)
    assert serial / pool == pytest.approx(19 / 5)


# --- the stage -------------------------------------------------------------------------------------------

SCALES = ["log10", "log10"]
BOUNDS = [(-6.0, 6.0)] * 2


def _two_valley(x_opt):
    x = np.asarray(x_opt, float)
    if x[0] > 1.5:
        return 1e10
    return min(np.sum((x - 1.0) ** 2), 5.0 + np.sum((x + 2.0) ** 2))


class _Pool:
    n_workers = 7

    def __init__(self):
        self.calls, self.closed = [], False

    def evaluate_batch(self, xs, label=None, eval_mode=None, **kw):
        self.calls.append((list(xs), label, eval_mode))
        return [_two_valley(x) for x in xs]

    def shutdown(self):
        self.closed = True


def _stage(gradient, pool_builder=None, parallel=False, cache=None, objective=_two_valley,
           method="L-BFGS-B", n_candidates=16, n_fits=3, maxiter=25):
    x0 = np.array([-2.0, -2.0])
    cands = _multistart_points(x0, BOUNDS, SCALES, n_candidates + 1, search_decades=4.0, seed=3,
                               verbose=False, sampler="sobol")
    tri = {**M.TRIAGE_DEFAULTS, "n_candidates": n_candidates, "keep_fraction": 0.5,
           "cluster_radius": 0.3, "parallel": parallel}
    return _run_multistart_stage(
        objective, cands, x0, BOUNDS, SCALES, method, {"options": {"maxiter": maxiter}}, tri,
        _resolve_multistart_limits({}), n_fits=n_fits, method="Nelder-Mead", verbose=False,
        pool_builder=pool_builder, fit_cache=cache, gradient=gradient)


def _g(**over):
    return {**M.GRADIENT_DEFAULTS, **over}


def _outcome(result):
    start, records, report = result
    return (np.round(start, 7).tolist(), [(r["candidate"], round(r["fun"], 7)) for r in records])


def test_the_gradients_go_to_the_pool_as_batches_one_gradient_wide():
    pool = _Pool()
    start, records, report = _stage(_g(parallel=True), pool_builder=lambda: pool)
    grad_calls = [c for c in pool.calls if c[1] == "multistart gradient"]
    assert grad_calls and all(len(c[0]) == 3 for c in grad_calls)          # k + 1 = 3
    assert all(c[2] == {"retry_limits": (4, 0)} for c in grad_calls)       # limited ladder, NOT relaxed
    assert all("relax" not in c[2] for c in grad_calls)
    assert report["gradient"]["used_pool"] is True and report["gradient"]["points_per_gradient"] == 3
    assert all(r["gradient"]["where"] == "pool" for r in records)
    assert np.allclose(start, [1.0, 1.0], atol=0.05)


def test_one_pool_serves_the_fast_pass_and_the_gradients_and_closes_once():
    built, pool = [], _Pool()

    def builder():
        built.append(1)
        return pool

    _stage(_g(parallel=True), pool_builder=builder, parallel=True, n_candidates=32)
    assert len(built) == 1 and pool.closed
    labels = {c[1] for c in pool.calls}
    assert labels == {"multistart triage", "multistart gradient"}


def test_serial_gradients_give_the_same_fits_as_pooled_ones():
    pooled = _stage(_g(parallel=True), pool_builder=lambda: _Pool())
    serial = _stage(_g(parallel=False), pool_builder=lambda: _Pool())
    assert _outcome(pooled) == _outcome(serial)
    assert all(r["gradient"]["where"] == "serial" for r in serial[1])


def test_parallel_false_never_builds_a_pool_for_the_gradients():
    built = []
    _stage(_g(parallel=False), pool_builder=lambda: built.append(1) or _Pool(), parallel=False)
    assert built == []


def test_the_pool_is_built_for_the_fits_even_when_the_fast_pass_was_all_recorded(tmp_path):
    cache = FitCache(str(tmp_path), "tag", "m" * 16, "f" * 16, n_params=2)
    cache.save_stage({})                                                    # harmless: wrong version
    # score everything once (serial), kill before the fits by raising in the first fit
    class Killed(BaseException):
        pass

    n = {"calls": 0}

    def dying(x):
        n["calls"] += 1
        if n["calls"] > 1 + 16 + 3:                                         # x0, 16 candidates, 3 gradient evals
            raise Killed()
        return _two_valley(x)

    with pytest.raises(Killed):
        _stage(_g(parallel=False), cache=cache, objective=dying)
    assert len(cache.load_stage()["scores"]) == 17
    built, pool = [], _Pool()
    _stage(_g(parallel=True), pool_builder=lambda: built.append(1) or pool, cache=cache)
    assert built == [1] and any(c[1] == "multistart gradient" for c in pool.calls)
    assert not any(c[1] == "multistart triage" for c in pool.calls)         # nothing left to score


def test_a_pool_that_fails_mid_fit_falls_back_to_serial_and_the_fit_still_finishes():
    class Breaking(_Pool):
        def evaluate_batch(self, xs, label=None, eval_mode=None, **kw):
            if len(self.calls) >= 3:
                raise RuntimeError("worker pool broke")
            return super().evaluate_batch(xs, label=label, eval_mode=eval_mode)

    pool = Breaking()
    start, records, report = _stage(_g(parallel=True), pool_builder=lambda: pool)
    assert pool.closed and np.allclose(start, [1.0, 1.0], atol=0.05)


def test_no_pool_available_just_means_serial():
    start, records, report = _stage(_g(parallel=True), pool_builder=lambda: None)
    assert report["gradient"]["used_pool"] is False
    assert all(r["gradient"]["where"] == "serial" for r in records)


def test_the_scipy_scheme_is_the_old_serial_behaviour():
    start, records, report = _stage(_g(scheme="scipy"), pool_builder=lambda: _Pool())
    assert all("gradient" not in r for r in records)
    assert report["gradient"]["scheme"] == "scipy" and report["gradient"]["used_pool"] is False


def test_a_derivative_free_method_takes_no_gradient_at_all():
    start, records, report = _stage(_g(parallel=True), pool_builder=lambda: _Pool(), method="Nelder-Mead")
    assert all("gradient" not in r for r in records) and report["gradient"]["used_pool"] is False


def test_without_gradient_settings_the_stage_behaves_as_before():
    _, records, report = _stage(None)
    assert all("gradient" not in r for r in records)


def test_records_say_how_the_gradient_was_taken_and_nfev_counts_evaluations():
    _, records, _ = _stage(_g(parallel=False))
    r = records[0]
    assert r["gradient"]["scheme"] == "forward" and r["gradient"]["step"] == 1e-3
    assert r["nfev"] == 3 * r["gradient"]["n_gradients"] + r["gradient"]["n_retried"]
    # x0 sits exactly at the second valley's minimum, so ITS fit stops at iteration 0;
    # the fits from the other starts iterate.
    assert all(isinstance(rec["nit"], int) for rec in records)
    assert max(rec["nit"] for rec in records) >= 1


def _clocked(per_eval, cold):
    state = {"t": 0.0, "n": 0}

    def objective(x):
        state["t"] += cold if state["n"] == 0 else per_eval
        state["n"] += 1
        return _two_valley(x)

    return (lambda: state["t"]), objective


def test_auto_picks_serial_for_a_cheap_two_parameter_gradient(monkeypatch):
    from pyantigen.engine import Optimize
    clock, obj = _clocked(per_eval=0.1, cold=1.0)
    monkeypatch.setattr(Optimize, "_now", clock)
    pool = _Pool()
    _, _, report = _stage(_g(parallel="auto"), pool_builder=lambda: pool, objective=obj)
    assert report["gradient"]["used_pool"] is False
    assert report["gradient"]["serial_estimate_s"] <= 1.5 * report["gradient"]["pool_estimate_s"]
    assert not any(c[1] == "multistart gradient" for c in pool.calls)


def test_auto_picks_the_pool_when_the_fits_are_long_and_evaluations_slow(monkeypatch):
    # k = 2 so a gradient is only 3 wide: the ceiling is 3x. 400 iterations makes that count.
    from pyantigen.engine import Optimize
    clock, obj = _clocked(per_eval=60.0, cold=90.0)
    monkeypatch.setattr(Optimize, "_now", clock)
    pool = _Pool()
    _, _, report = _stage(_g(parallel="auto"), pool_builder=lambda: pool, objective=obj, maxiter=400)
    assert report["gradient"]["used_pool"] is True
    assert any(c[1] == "multistart gradient" for c in pool.calls)
    assert report["gradient"]["iteration_cap"] == 400


def test_the_gradient_settings_are_part_of_what_a_checkpoint_is_for(tmp_path):
    cache = FitCache(str(tmp_path), "tag", "m" * 16, "f" * 16, n_params=2)

    class Killed(BaseException):
        pass

    n = {"c": 0}

    def dying(x):
        n["c"] += 1
        if n["c"] > 8:
            raise Killed()
        return _two_valley(x)

    with pytest.raises(Killed):
        _stage(_g(step=1e-3), cache=cache, objective=dying)
    _, _, report = _stage(_g(step=1e-2), cache=cache)                      # a different step
    assert report["launches"] == 1 and "resumed" not in report              # started over

    # WHERE the points are evaluated is not part of it: same step, a different place, resumes.
    cache2 = FitCache(str(tmp_path / "second"), "tag", "m" * 16, "f" * 16, n_params=2)
    n["c"] = 0
    with pytest.raises(Killed):
        _stage(_g(step=1e-2, parallel=False), cache=cache2, objective=dying)
    _, _, again = _stage(_g(step=1e-2, parallel=True), cache=cache2, pool_builder=lambda: _Pool())
    assert again["resumed"]["candidates_scored_in_earlier_launches"] > 0

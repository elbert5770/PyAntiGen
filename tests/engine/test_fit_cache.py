"""Checks on the fit cache.

The cache exists so that a job which fits and then profiles survives being
restarted: the fitted optimum is written the moment the fit returns and a
relaunch of the same problem takes it rather than refitting. Its danger, like
the Hessian cache's, is not a miss but a wrong hit -- an optimum belonging to
a different problem would anchor the whole profile in the wrong place -- so
most of what follows is about invalidation.

Run from the repository root:
    python tests/engine/test_fit_cache.py
"""
import json
import os
import shutil
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.getcwd())

import pyantigen.engine.Fit_cache as fc                                     # noqa: E402
from pyantigen.engine.Fit_cache import (                                    # noqa: E402
    FitCache,
    data_fingerprint,
    fit_fingerprint,
)

failures = []


def check(name, cond, detail=""):
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}  {detail}")
        failures.append(name)


K = 4
NAMES = [f"p{i}" for i in range(K)]
X0 = [1.0, 2.0, 3.0, 4.0]
BOUNDS = [(0.1, 10.0)] * K
SCALES = ["log10"] * K
GROUPS = {"A": {}, "B": {}}
MODEL = "model A -> B; k1*A;"
KW = {"options": {"maxiter": 500}, "events_depend_on_opt_param": False,
      "profile_optimizer_kwargs": {"options": {"maxfev": 1500}}}


def _fp(**over):
    args = dict(param_names=NAMES, x0_lin=X0, bounds_lin=BOUNDS, scales=SCALES,
                groups=GROUPS, model_text=MODEL, method="Nelder-Mead",
                optimizer_kwargs=KW, n_starts=1, start_seed=None,
                search_decades=None, solver_hash="s0", data_hash="d0")
    args.update(over)
    return fit_fingerprint(**args)


def _cache(root, mh=None, fh=None, enabled=True):
    m, f = _fp()
    return FitCache(root, "A_B", mh or m, fh or f, K, enabled=enabled)


def test_round_trip():
    print("\n[round trip]")
    root = tempfile.mkdtemp()
    try:
        c = _cache(root)
        check("empty cache misses", c.load_complete() is None)
        x = np.array([1.5, 2.5, 3.5, 4.5])
        check("save complete", c.save_complete(x, 12.5, param_names=NAMES,
                                               nit=40, nfev=70))
        got = _cache(root).load_complete()
        check("complete comes back", got is not None)
        if got is not None:
            check("x round-trips", np.allclose(got["x_lin"], x))
            check("fun round-trips", got["fun"] == 12.5)
            check("nfev round-trips", got.get("nfev") == 70)
            check("registry-shaped parameters",
                  got.get("parameters", {}).get("p2") == 3.5)
            check("path reported", got["path"] == c.path)
        check("no temp file left behind",
              not any(n.endswith(".tmp") for n in os.listdir(c.dir)),
              str(os.listdir(c.dir)))
        check("lives under fits/", os.path.basename(c.dir) == "fits")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_every_input_invalidates():
    print("\n[every input to the fingerprint invalidates]")
    base_m, base_f = _fp()
    cases = {
        "x0": dict(x0_lin=[1.0, 2.0, 3.0, 4.0 + 1e-6]),
        "bounds": dict(bounds_lin=[(0.1, 10.0)] * 3 + [(0.1, 20.0)]),
        "scales": dict(scales=["lin"] * K),
        "groups": dict(groups={"A": {}}),
        "method": dict(method="Powell"),
        "optimizer options": dict(optimizer_kwargs={"options": {"maxiter": 200}}),
        "n_starts": dict(n_starts=4),
        "start_seed": dict(start_seed=7),
        "search_decades": dict(search_decades=2.0),
        "solver": dict(solver_hash="s1"),
        "data": dict(data_hash="d1"),
        "param names": dict(param_names=["q0", "p1", "p2", "p3"]),
    }
    for label, over in cases.items():
        m, f = _fp(**over)
        check(f"{label} changes the fit hash", f != base_f)
        check(f"{label} leaves the model hash", m == base_m)
    m, f = _fp(model_text=MODEL + " k2*B;")
    check("model text changes the model hash", m != base_m)

    # Profile-only settings decide what is done with the fit, not the fit.
    kw2 = dict(KW)
    kw2["profile_optimizer_kwargs"] = {"options": {"maxfev": 9999}}
    kw2["profile_grid"] = {"n_grid": 9}
    _, f2 = _fp(optimizer_kwargs=kw2)
    check("profile-only optimizer keys do not change the hash", f2 == base_f)

    # x0 within float noise is the same problem.
    _, f3 = _fp(x0_lin=[1.0, 2.0, 3.0, 4.0 + 1e-14])
    check("x0 float noise is the same problem", f3 == base_f)

    root = tempfile.mkdtemp()
    try:
        c = _cache(root)
        c.save_complete(np.ones(K), 1.0)
        _, f_other = _fp(x0_lin=[1.0, 2.0, 3.0, 5.0])
        other = FitCache(root, "A_B", base_m, f_other, K)
        check("a different problem misses the stored fit",
              other.load_complete() is None)
        other_model = FitCache(root, "A_B", m, base_f, K)
        check("a different model misses the stored fit",
              other_model.load_complete() is None)
        wrong_n = FitCache(root, "A_B", base_m, base_f, K + 1)
        check("a different parameter count misses",
              wrong_n.load_complete() is None)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_data_fingerprint():
    print("\n[data fingerprint]")
    try:
        import pandas as pd
    except ImportError:
        print("  SKIP  pandas not installed")
        return
    df = pd.DataFrame({"t": [0.0, 1.0, 2.0], "y": [1.0, 0.5, 0.25]})
    models = {"arm": {"df_dict": {"silk": df, "f_plateau": 0.3, "missing": None}}}
    h0 = data_fingerprint(models)
    check("deterministic", data_fingerprint(models) == h0)
    df2 = df.copy()
    df2.loc[1, "y"] = 0.51
    check("one edited value changes it",
          data_fingerprint({"arm": {"df_dict": {"silk": df2, "f_plateau": 0.3,
                                                 "missing": None}}}) != h0)
    check("a scalar change changes it",
          data_fingerprint({"arm": {"df_dict": {"silk": df, "f_plateau": 0.31,
                                                 "missing": None}}}) != h0)
    check("an unhashable value is tolerated",
          isinstance(data_fingerprint({"arm": {"df_dict": {"x": object()}}}), str))
    check("empty is tolerated", isinstance(data_fingerprint({}), str))


def test_partial_record():
    print("\n[partial record]")
    root = tempfile.mkdtemp()
    old = fc.PARTIAL_SAVE_INTERVAL_S
    try:
        c = _cache(root)
        check("no partial on an empty cache", c.load_partial() is None)
        x1 = np.array([1.1, 2.1, 3.1, 4.1])
        check("first partial writes", c.save_partial(x1, 20.0, n_evals=5))
        got = _cache(root).load_partial()
        check("partial comes back", got is not None
              and np.allclose(got["x_lin"], x1) and got.get("n_evals") == 5)

        # Throttled: a second write inside the interval is refused, but a
        # forced one goes through.
        x2 = np.array([1.2, 2.2, 3.2, 4.2])
        check("second partial inside the interval is throttled",
              not c.save_partial(x2, 19.0, n_evals=6))
        check("forced partial writes", c.save_partial(x2, 19.0, n_evals=6,
                                                      force=True))
        got = _cache(root).load_partial()
        check("forced partial is the one on disk",
              got is not None and np.allclose(got["x_lin"], x2))

        # A partial never displaces or accompanies a complete record.
        check("complete still absent", c.load_complete() is None)
        check("complete writes over a partial", c.save_complete(x2, 18.0))
        check("partial cleared by complete", _cache(root).load_partial() is None)
        check("a later partial cannot demote a complete",
              not c.save_partial(x1, 25.0, force=True))
        check("complete survives", _cache(root).load_complete() is not None)

        # The interval is honoured against wall time, not call count.
        fc.PARTIAL_SAVE_INTERVAL_S = 0.05
        d = _cache(root)
        # fresh dir so there is no complete record blocking partials
        d = FitCache(os.path.join(root, "fresh"), "A_B", d.model_hash, d.fit_hash, K)
        check("first write ok", d.save_partial(x1, 1.0))
        check("immediate second refused", not d.save_partial(x1, 1.0))
        time.sleep(0.1)
        check("write after the interval ok", d.save_partial(x1, 1.0))
    finally:
        fc.PARTIAL_SAVE_INTERVAL_S = old
        shutil.rmtree(root, ignore_errors=True)


def test_damage_is_a_miss_not_a_crash():
    print("\n[damage is a miss, not a crash]")
    root = tempfile.mkdtemp()
    try:
        c = _cache(root)
        c.save_complete(np.ones(K), 1.0)

        with open(c.path, "w") as fh:
            fh.write('{"format": "fit-v1", "complete": {"x_lin": [1, 2')
        check("truncated file misses", _cache(root).load_complete() is None)

        c.save_complete(np.ones(K), 1.0)
        with open(c.path) as fh:
            data = json.load(fh)
        data["complete"]["x_lin"] = [1.0, 2.0, 3.0]
        with open(c.path, "w") as fh:
            json.dump(data, fh)
        check("wrong-length vector misses", _cache(root).load_complete() is None)

        data["complete"]["x_lin"] = [1.0, None, 3.0, 4.0]
        with open(c.path, "w") as fh:
            json.dump(data, fh)
        check("non-finite vector misses", _cache(root).load_complete() is None)

        data["complete"]["x_lin"] = [1.0, 2.0, 3.0, 4.0]
        data["format"] = "fit-v0"
        with open(c.path, "w") as fh:
            json.dump(data, fh)
        check("old format misses", _cache(root).load_complete() is None)

        with open(c.path, "w") as fh:
            fh.write("[1, 2, 3]")
        check("wrong top-level type misses", _cache(root).load_complete() is None)
        check("a miss still allows a fresh save",
              c.save_complete(np.ones(K), 1.0)
              and _cache(root).load_complete() is not None)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_disabled_cache_is_inert():
    print("\n[disabled cache is inert]")
    root = tempfile.mkdtemp()
    try:
        c = _cache(root, enabled=False)
        check("disabled save is a no-op", not c.save_complete(np.ones(K), 1.0))
        check("disabled partial is a no-op", not c.save_partial(np.ones(K), 1.0))
        check("disabled load misses", c.load_complete() is None)
        check("no directory created", not os.path.exists(os.path.join(root, "fits")))
        n = FitCache(None, "A_B", "m", "f", K)
        check("no root is disabled", not n.enabled and n.path is None)
        check("no root load misses", n.load_complete() is None)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_fingerprint_survives_odd_kwargs():
    print("\n[fingerprint tolerates non-JSON optimizer kwargs]")
    kw = dict(KW)
    kw["callback"] = lambda x: None
    kw["x_scale"] = np.array([1.0, 2.0])
    try:
        m, f = _fp(optimizer_kwargs=kw)
        ok = isinstance(f, str) and len(f) == 16
    except Exception as exc:  # noqa: BLE001
        ok = False
        print("   ", exc)
    check("callables and arrays are hashable", ok)


def test_partial_carries_the_optimizer_state():
    print("\n[partial record: optimizer state]")
    from pyantigen.engine.Nelder_mead import nelder_mead, new_state, state_to_json
    root = tempfile.mkdtemp()
    try:
        c = _cache(root)
        st = new_state(np.array([1.0, 2.0, 3.0, 4.0]))
        nelder_mead(lambda x: float(np.sum(x ** 2)), st,
                    should_stop=lambda: st["nfev"] >= 7)
        blob = state_to_json(st)
        x1 = np.array([1.1, 2.1, 3.1, 4.1])
        check("a partial with a state writes",
              c.save_partial(x1, 20.0, n_evals=7, nm_state=blob, force=True))
        got = _cache(root).load_partial()
        check("the state comes back validated, as a state",
              got is not None and got["nm_state"] is not None
              and got["nm_state"]["nfev"] == st["nfev"]
              and np.array_equal(got["nm_state"]["sim"], st["sim"])
              and np.array_equal(got["nm_state"]["fsim"], st["fsim"]))
        check("no legacy simplex is reported for it", got["nm_simplex"] is None)

        # Two writers of one record must not undo each other.
        check("a best-point-only write goes through",
              c.save_partial(x1, 19.0, n_evals=9, force=True))
        got = _cache(root).load_partial()
        check("and leaves the saved state alone rather than erasing it",
              got["nm_state"] is not None and got["fun"] == 19.0)

        # The state is written more often than the best point: its own interval.
        old = fc.PARTIAL_SAVE_INTERVAL_S
        fc.PARTIAL_SAVE_INTERVAL_S = 3600.0
        try:
            d = FitCache(os.path.join(root, "fresh"), "A_B", c.model_hash,
                         c.fit_hash, K)
            check("first write", d.save_partial(x1, 1.0, nm_state=blob))
            check("a second inside the default interval is refused",
                  not d.save_partial(x1, 1.0, nm_state=blob))
            check("but an explicit short interval is honoured",
                  d.save_partial(x1, 1.0, nm_state=blob, interval=0.0))
        finally:
            fc.PARTIAL_SAVE_INTERVAL_S = old

        # Damage costs the exact resume, never the best point.
        with open(c.path) as fh:
            data = json.load(fh)
        data["partial"]["nm_state"]["sim"] = [[1.0, 2.0]]
        with open(c.path, "w") as fh:
            json.dump(data, fh)
        got = _cache(root).load_partial()
        check("a damaged state is a miss but the best point survives",
              got is not None and got["nm_state"] is None
              and np.allclose(got["x_lin"], x1))

        # A record written before states were saved carried a bare simplex.
        data["partial"].pop("nm_state")
        data["partial"]["nm_simplex"] = np.eye(K + 1, K).tolist()
        with open(c.path, "w") as fh:
            json.dump(data, fh)
        got = _cache(root).load_partial()
        check("a legacy simplex is still read",
              got["nm_state"] is None and got["nm_simplex"] is not None
              and got["nm_simplex"].shape == (K + 1, K))
        data["partial"]["nm_simplex"] = [[1.0, 2.0]]
        with open(c.path, "w") as fh:
            json.dump(data, fh)
        check("a wrong-shaped legacy simplex is a miss",
              _cache(root).load_partial()["nm_simplex"] is None)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _fit_problem(counter, sleep=0.0, kill_at=None):
    """A 4-parameter ill-conditioned bowl in opt space, and how to run a fit."""
    from scipy.optimize import minimize
    rng = np.random.default_rng(5)
    Q = np.linalg.qr(rng.standard_normal((K, K)))[0]
    A = Q @ np.diag(np.geomspace(1.0, 200.0, K)) @ Q.T
    target = np.array([0.4, -0.2, 0.9, 0.1])

    class Killed(BaseException):
        pass

    def objective(x):
        counter["n"] += 1
        if kill_at is not None and counter["n"] == kill_at:
            raise Killed
        if sleep:
            time.sleep(sleep)
        d = np.asarray(x, dtype=float) - target
        return float(d @ A @ d)

    return objective, Killed, minimize


def test_the_fit_driver_is_scipys_fit_and_survives_stops_and_kills():
    print("\n[fit driver]")
    from pyantigen.engine.Deadline import DeadlineReached, RunBudget
    from pyantigen.engine.Optimize import _run_fit_nelder_mead

    scales = ["lin"] * K
    bounds = [(-3.0, 3.0)] * K
    opt_kw = {"options": {"maxiter": 900}, "tol": 1e-7}
    x_start = np.array([1.5, 1.0, -1.0, 2.0])
    old_interval = fc.NM_STATE_SAVE_INTERVAL_S
    fc.NM_STATE_SAVE_INTERVAL_S = 0.0       # write at every consistent point
    root = tempfile.mkdtemp()
    try:
        counter = {"n": 0}
        objective, _Killed, minimize = _fit_problem(counter)
        ref = minimize(objective, x_start, method="Nelder-Mead", bounds=bounds,
                       **opt_kw)
        counter["n"] = 0
        cache = _cache(os.path.join(root, "whole"))
        got = _run_fit_nelder_mead(objective, x_start, bounds, opt_kw, scales,
                                   cache, None, RunBudget(deadline=None))
        evals_whole = counter["n"]
        check("the driver returns scipy's fit exactly",
              np.array_equal(got.x, ref.x) and got.fun == ref.fun
              and got.nfev == ref.nfev == evals_whole and got.nit == ref.nit
              and got.success == ref.success, f"{got.nfev} vs {ref.nfev}")
        check("it left a partial record behind for a kill",
              cache.load_partial() is not None
              and cache.load_partial()["nm_state"] is not None)

        # -- stopped by the run's stop time, over and over -------------------
        counter["n"] = 0
        objective, _Killed, _m = _fit_problem(counter, sleep=0.001)
        cache = _cache(os.path.join(root, "stops"))
        stops = 0
        for _ in range(400):
            budget = RunBudget(deadline=time.time() + 600.0 + 0.03,
                               margin_s=600.0)
            try:
                res = _run_fit_nelder_mead(objective, x_start, bounds, opt_kw,
                                           scales, cache, cache.load_partial(),
                                           budget)
                break
            except DeadlineReached:
                stops += 1
        check("the fit really was stopped several times", stops >= 3,
              f"{stops} stop(s)")
        check("stopped and resumed, it is scipy's fit exactly",
              np.array_equal(res.x, ref.x) and res.fun == ref.fun
              and res.nfev == ref.nfev, f"{res.nfev} vs {ref.nfev}")
        # Each resume spends exactly one evaluation putting the noise floors back
        # where the first launch had them (pyantigen.engine.Optimize._restore_calibration);
        # nothing else is repeated.
        check("no evaluation was repeated across the stops beyond one "
              "calibration evaluation per resume",
              counter["n"] == evals_whole + stops,
              f"{counter['n']} vs {evals_whole} + {stops}")

        # -- killed outright, at several points ------------------------------
        for kill in (3, 20, 70):
            counter["n"] = 0
            objective, Killed, _m = _fit_problem(counter, kill_at=kill)
            cache = _cache(os.path.join(root, f"kill{kill}"))
            try:
                _run_fit_nelder_mead(objective, x_start, bounds, opt_kw, scales,
                                     cache, None, RunBudget(deadline=None))
                died = False
            except Killed:
                died = True
            before = counter["n"]
            resume = cache.load_partial()
            counter["n"] = 0
            objective, _K2, _m = _fit_problem(counter)
            res = _run_fit_nelder_mead(objective, x_start, bounds, opt_kw,
                                       scales, cache, resume,
                                       RunBudget(deadline=None))
            repeated = before + counter["n"] - evals_whole
            check(f"kill at evaluation {kill}: died, resumed, and got scipy's "
                  f"fit exactly",
                  died and np.array_equal(res.x, ref.x) and res.fun == ref.fun
                  and res.nfev == ref.nfev, f"died={died} {res.nfev}")
            # One evaluation is the resume's calibration; the rest is what the
            # kill lost.
            check(f"kill at evaluation {kill}: at most one evaluation was lost "
                  f"(plus the resume's calibration evaluation)",
                  0 <= repeated <= 3, f"{repeated} repeated")

        # -- a record from before states were saved ---------------------------
        counter["n"] = 0
        objective, _Killed, _m = _fit_problem(counter)
        cache = _cache(os.path.join(root, "legacy"))
        sim = x_start + 0.3 * np.vstack([np.zeros(K), np.eye(K)])
        cache.save_partial(x_start, 99.0, n_evals=5, force=True)
        with open(cache.path) as fh:
            data = json.load(fh)
        data["partial"]["nm_simplex"] = sim.tolist()
        data["partial"]["nfev_total"] = 5
        data["partial"]["nit_total"] = 1
        with open(cache.path, "w") as fh:
            json.dump(data, fh)
        partial = cache.load_partial()
        res = _run_fit_nelder_mead(objective, x_start, bounds, opt_kw, scales,
                                   cache, partial, RunBudget(deadline=None))
        check("a legacy simplex record resumes and converges",
              res.success and res.fun < 1e-6, f"fun={res.fun}")
        check("its recorded spend counts against the allowance",
              res.nfev >= 5 + K + 1, str(res.nfev))
    finally:
        fc.NM_STATE_SAVE_INTERVAL_S = old_interval
        shutil.rmtree(root, ignore_errors=True)


def test_the_driver_is_not_used_for_options_it_does_not_implement():
    print("\n[fit driver: what it declines]")
    from pyantigen.engine.Nelder_mead import can_run
    check("the spec's usual options are covered",
          can_run({"maxiter": 1500, "maxfev": 1500, "fatol": 1e-4, "xatol": 1e-4},
                  {"tol": 1e-3}))
    check("an option it lacks sends the fit back to scipy",
          not can_run({"maxiter": 5, "return_all_the_things": True}))
    check("so does an extra scipy argument", not can_run({}, {"callback": 1}))


def test_partial_carries_the_search_state_beside_the_polish_state():
    print("\n[partial record: search + polish states]")
    from pyantigen.engine.Differential_evolution import (
        differential_evolution, new_state as de_new, state_to_json as de_json,
    )
    from pyantigen.engine.Nelder_mead import (
        nelder_mead, new_state as nm_new, state_to_json as nm_json,
    )
    root = tempfile.mkdtemp()
    try:
        c = _cache(root)
        de = de_new([(-3.0, 3.0)] * K, popsize=3, seed=1)
        differential_evolution(lambda X: np.sum(np.asarray(X) ** 2, axis=1), de,
                               maxiter=2, stall_gens=0)
        nm = nm_new(np.ones(K))
        nelder_mead(lambda x: float(np.sum(x ** 2)), nm,
                    should_stop=lambda: nm["nfev"] >= 4)
        x = np.ones(K)

        check("a search state writes", c.save_partial(x, 5.0, de_state=de_json(de),
                                                       force=True))
        got = _cache(root).load_partial()
        check("it comes back validated, with no polish state yet",
              got["de_state"] is not None and got["nm_state"] is None
              and got["de_state"]["gen"] == de["gen"]
              and np.array_equal(got["de_state"]["pop"], de["pop"]))

        check("the polish then writes its own", c.save_partial(
            x, 4.0, nm_state=nm_json(nm), force=True))
        got = _cache(root).load_partial()
        check("and the search state survives it, so 'a polish state exists' "
              "still means the search finished",
              got["de_state"] is not None and got["nm_state"] is not None)

        check("a best-point-only write erases neither",
              c.save_partial(x, 3.0, force=True))
        got = _cache(root).load_partial()
        check("both are still there", got["de_state"] is not None
              and got["nm_state"] is not None and got["fun"] == 3.0)

        # A search state for another problem size is a miss, not a crash.
        import json as _json
        with open(c.path) as fh:
            data = _json.load(fh)
        data["partial"]["de_state"]["pop"] = [[1.0, 2.0]] * 5
        with open(c.path, "w") as fh:
            _json.dump(data, fh)
        got = _cache(root).load_partial()
        check("a damaged search state is a miss and costs nothing else",
              got["de_state"] is None and got["nm_state"] is not None
              and np.allclose(got["x_lin"], x))
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_the_calibration_point_is_recorded_once_and_kept():
    print("\n[partial record: noise-floor calibration point]")
    root = tempfile.mkdtemp()
    try:
        c = _cache(root)
        first, later = np.arange(1.0, K + 1.0), np.full(K, 9.0)
        c.save_partial(np.ones(K), 5.0, force=True, cal_x_lin=first)
        check("it comes back", np.array_equal(_cache(root).load_partial()["cal_x_lin"],
                                             first))
        c.save_partial(np.ones(K), 4.0, force=True, cal_x_lin=later)
        check("the first writer wins: a later write cannot move it",
              np.array_equal(_cache(root).load_partial()["cal_x_lin"], first))
        c.save_partial(np.ones(K), 3.0, force=True)
        check("and a write that does not mention it keeps it",
              np.array_equal(_cache(root).load_partial()["cal_x_lin"], first))

        fresh = _cache(os.path.join(root, "old"))
        fresh.save_partial(np.ones(K), 5.0, force=True)
        check("a record with none reports none (older records keep working)",
              _cache(os.path.join(root, "old")).load_partial()["cal_x_lin"] is None)

        with open(c.path) as fh:
            data = json.load(fh)
        data["partial"]["cal_x_lin"] = [1.0, 2.0]
        with open(c.path, "w") as fh:
            json.dump(data, fh)
        got = _cache(root).load_partial()
        check("a wrong-length one is a miss and costs nothing else",
              got["cal_x_lin"] is None and np.allclose(got["x_lin"], np.ones(K)))
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _calibrating_objective():
    """A stand-in for the real objective's one nasty property.

    pyantigen.engine.Noise_floor sets a floored observable's sigma on the first call a
    process makes and freezes it, so what a vector scores depends on which vector
    the process happened to evaluate first. Here that is a constant offset. A new
    process is a new call to this factory.
    """
    cal = {"c": None}
    rng = np.random.default_rng(5)
    Q = np.linalg.qr(rng.standard_normal((K, K)))[0]
    A = Q @ np.diag(np.geomspace(1.0, 200.0, K)) @ Q.T
    target = np.array([0.4, -0.2, 0.9, 0.1])

    def objective(x):
        x = np.asarray(x, dtype=float)
        if cal["c"] is None:
            cal["c"] = float(x[0])
        d = x - target
        return float(d @ A @ d) + 10.0 * cal["c"]
    return objective


def test_a_resumed_nelder_mead_fit_scores_against_the_original_floors():
    print("\n[fit driver: noise-floor calibration across a resume]")
    import pyantigen.engine.Optimize as O
    from pyantigen.engine.Deadline import RunBudget
    scales = ["lin"] * K
    bounds = [(-3.0, 3.0)] * K
    opt_kw = {"options": {"maxiter": 200}, "tol": 1e-7}
    x_start = np.array([1.5, 1.0, -1.0, 2.0])
    old = fc.NM_STATE_SAVE_INTERVAL_S
    fc.NM_STATE_SAVE_INTERVAL_S = 0.0
    root = tempfile.mkdtemp()

    class Killed(BaseException):
        pass

    def run(cache, objective, resume):
        return O._run_fit_nelder_mead(objective, x_start, bounds, opt_kw, scales,
                                      cache, resume, RunBudget(deadline=None))

    def killed_then_resumed(name, restore=True):
        cache = _cache(os.path.join(root, name))
        base = _calibrating_objective()
        n = {"n": 0}

        def dying(x):
            n["n"] += 1
            if n["n"] == 30:
                raise Killed
            return base(x)
        try:
            run(cache, dying, None)
        except Killed:
            pass
        # A new process: a fresh objective that has calibrated on nothing yet.
        return run(cache, _calibrating_objective(), cache.load_partial())

    try:
        ref = run(_cache(os.path.join(root, "ref")), _calibrating_objective(), None)
        res = killed_then_resumed("with")
        check("killed and resumed in a new process, the fit is identical to an "
              "uninterrupted one",
              np.array_equal(res.x, ref.x) and res.fun == ref.fun,
              f"{res.fun!r} vs {ref.fun!r}")

        saved = O._restore_calibration
        O._restore_calibration = lambda *a, **k: False
        try:
            broken = killed_then_resumed("without")
        finally:
            O._restore_calibration = saved
        check("(control) without the restore it is NOT -- the test has teeth",
              broken.fun != ref.fun, f"{broken.fun!r} vs {ref.fun!r}")
    finally:
        fc.NM_STATE_SAVE_INTERVAL_S = old
        shutil.rmtree(root, ignore_errors=True)


def test_failures():
    """``check`` records rather than raises, so pytest needs this to see them."""
    assert not failures, "failed checks: " + ", ".join(failures)


def main():
    test_round_trip()
    test_every_input_invalidates()
    test_data_fingerprint()
    test_partial_record()
    test_damage_is_a_miss_not_a_crash()
    test_disabled_cache_is_inert()
    test_fingerprint_survives_odd_kwargs()
    test_partial_carries_the_optimizer_state()
    test_the_fit_driver_is_scipys_fit_and_survives_stops_and_kills()
    test_the_driver_is_not_used_for_options_it_does_not_implement()
    test_partial_carries_the_search_state_beside_the_polish_state()
    test_the_calibration_point_is_recorded_once_and_kept()
    test_a_resumed_nelder_mead_fit_scores_against_the_original_floors()

    print("\n" + "=" * 72)
    if failures:
        print(f"FAILED ({len(failures)}):")
        for f in failures:
            print("  -", f)
        return 1
    print("ALL FIT CACHE CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())

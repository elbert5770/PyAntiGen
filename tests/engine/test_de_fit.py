"""The differential-evolution fit driver: search, polish, and surviving being killed.

Run from the repository root:
    python tests/engine/test_de_fit.py

pyantigen.engine.Optimize._run_fit_differential_evolution glues three things together --
the search (pyantigen.engine.Differential_evolution), a worker pool that evaluates a
generation at a time in solver search mode, and the Nelder-Mead polish -- and
keeps the whole thing recoverable through the fit cache. The pool is a stub here,
so what is being tested is the glue: that the answer does not depend on how many
times the run was stopped or killed, that a kill in the search and a kill in the
polish each resume at the right stage, and that a bad step cap or a dead pool is
reported or survived rather than silently producing garbage.
"""
import os
import shutil
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.getcwd())

import pyantigen.engine.Simulate as S                                          # noqa: E402
from pyantigen.engine.Deadline import DeadlineReached, RunBudget               # noqa: E402
import pyantigen.engine.Fit_cache as fc                                         # noqa: E402
from pyantigen.engine.Fit_cache import FitCache                                # noqa: E402
from pyantigen.engine.Optimize import (                                        # noqa: E402
    _de_options, _resolve_profile_optimizer, _run_fit_differential_evolution,
)

failures = []


def check(name, cond, detail=""):
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}  {detail}")
        failures.append(name)


# The fit cache throttles state writes to one per NM_STATE_SAVE_INTERVAL_S, which
# is right when an evaluation takes seconds and means a test objective that
# finishes in milliseconds never writes the polish state at all.
fc.NM_STATE_SAVE_INTERVAL_S = 0.0

K = 3
SCALES = ["lin"] * K
BOUNDS = [(-5.0, 5.0)] * K
X0 = np.array([3.0, -3.0, 2.5])
TARGET = np.array([0.4, -0.7, 1.1])
OPTIONS = {"popsize": 4, "maxiter": 60, "search_max_steps": 5000,
           "stall_gens": 12, "stall_tol": 1e-6, "seed": 5,
           "polish_options": {"maxiter": 400}}


class Killed(BaseException):
    """Goes straight through ``except Exception``, like a SIGKILL."""


def _cache(root):
    return FitCache(root, "t", "m" * 16, "f" * 16, K)


def _objective_factory(counter=None, kill_at=None, sleep=0.0, seen_caps=None):
    def objective(x):
        if counter is not None:
            counter["n"] += 1
            if kill_at is not None and counter["n"] == kill_at:
                raise Killed
        if seen_caps is not None:
            seen_caps.append(S._SEARCH_MAX_STEPS)
        if sleep:
            time.sleep(sleep)
        x = np.asarray(x, dtype=float)
        if x[0] > 4.5:                      # a corner the "integrator" cannot solve
            return 1e10
        return float(np.sum((x - TARGET) ** 2))
    return objective


class StubPool:
    """Evaluates a batch the way ParallelEvaluator does, in this process."""

    def __init__(self, objective, cap, batch_counter=None, kill_batch=None,
                 fail_after=None, fail_x0=False, sleep=0.0):
        self.objective, self.cap = objective, cap
        self.n_workers = 4
        self.batches = batch_counter if batch_counter is not None else {"n": 0}
        self.kill_batch, self.fail_after, self.fail_x0 = kill_batch, fail_after, fail_x0
        self.sleep = sleep
        self.shut = False
        self.caps_seen = []

    def evaluate_batch(self, xs, label=None):
        self.batches["n"] += 1
        if self.kill_batch is not None and self.batches["n"] == self.kill_batch:
            raise Killed
        if self.fail_after is not None and self.batches["n"] > self.fail_after:
            raise RuntimeError("the pool broke")
        if self.sleep:
            time.sleep(self.sleep)
        out = []
        with S.search_mode(self.cap):
            for x in xs:
                self.caps_seen.append(S._SEARCH_MAX_STEPS)
                out.append(1e10 if self.fail_x0 else self.objective(x))
        return out

    def shutdown(self):
        self.shut = True


def _run(root, objective=None, pool_kw=None, options=None, budget=None,
         resume=True, pools=None, cache=None, pool_objective=None):
    objective = objective or _objective_factory()
    pool_objective = pool_objective or objective
    cache = cache or _cache(root)
    pools = pools if pools is not None else []

    def build_pool(cap):
        pool = StubPool(pool_objective, cap, **(pool_kw or {}))
        pools.append(pool)
        return pool

    opts = dict(OPTIONS)
    opts.update(options or {})
    partial = cache.load_partial() if resume else None
    x0 = X0 if partial is None else np.asarray(partial["x_lin"], dtype=float)
    res = _run_fit_differential_evolution(
        objective, x0, BOUNDS, opts, SCALES, cache, partial,
        budget or RunBudget(deadline=None), build_pool)
    return res, cache, pools


def _quiet(fn, *a, **k):
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()) as buf:
        try:
            return fn(*a, **k), buf.getvalue()
        finally:
            pass


# ---------------------------------------------------------------------------

def test_search_then_polish_reaches_the_optimum():
    print("\nSearch and polish:")
    root = tempfile.mkdtemp(prefix="defit_")
    try:
        (out, log) = _quiet(_run, root)
        res, cache, pools = out
        check("the polished fit lands on the optimum", res.fun < 1e-6
              and np.allclose(res.x, TARGET, atol=1e-3), f"f={res.fun:.3g} x={res.x}")
        check("the search ran and reported its own cost",
              res.de_generations > 0 and res.de_nfev > 0)
        check("the pool was built once, in search mode, and shut down",
              len(pools) == 1 and pools[0].cap == 5000 and pools[0].shut)
        check("every pool evaluation ran under the cap",
              set(pools[0].caps_seen) == {5000})
        check("the search mode does not leak out of the run", S._SEARCH_MAX_STEPS is None)
        part = cache.load_partial()
        check("the partial record holds both the search and the polish state",
              part["de_state"] is not None and part["nm_state"] is not None)
        check("the log says what it did", "[de] starting point" in log
              and "search finished" in log and "polishing" in log, log[-300:])
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_polish_can_be_switched_off():
    print("\nNo polish:")
    root = tempfile.mkdtemp(prefix="defit_")
    try:
        (out, _l) = _quiet(_run, root, options={"polish": False})
        res, cache, _p = out
        check("the search's own result comes back", hasattr(res, "population")
              and res.fun < 1.0, f"f={res.fun}")
        check("and no polish state was written", cache.load_partial()["nm_state"] is None)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _reference(root):
    (out, _l) = _quiet(_run, root)
    return out[0]


def test_stopped_over_and_over_it_is_the_same_fit():
    print("\nStopped by the run's stop time, repeatedly:")
    ref_root = tempfile.mkdtemp(prefix="defit_ref_")
    root = tempfile.mkdtemp(prefix="defit_")
    try:
        ref = _reference(ref_root)
        stops = 0
        res = None
        for _ in range(400):
            budget = RunBudget(deadline=time.time() + 600.0 + 0.04, margin_s=600.0)
            try:
                (out, _l) = _quiet(_run, root, objective=_objective_factory(sleep=0.001),
                                   pool_kw={"sleep": 0.002}, budget=budget)
                res = out[0]
                break
            except DeadlineReached:
                stops += 1
        check("it really was stopped several times", stops >= 3, f"{stops}")
        check("stopped and resumed, the fit is bit-identical to an uninterrupted one",
              res is not None and np.array_equal(res.x, ref.x) and res.fun == ref.fun,
              f"{None if res is None else res.fun!r} vs {ref.fun!r}")
    finally:
        shutil.rmtree(ref_root, ignore_errors=True)
        shutil.rmtree(root, ignore_errors=True)


def test_a_kill_in_the_search_costs_one_generation():
    print("\nKilled during the search:")
    ref_root = tempfile.mkdtemp(prefix="defit_ref_")
    try:
        ref_counter = {"n": 0}
        pools = []
        (out, _l) = _quiet(_run, ref_root, pool_kw={"batch_counter": ref_counter},
                           pools=pools)
        ref, batches_ref = out[0], ref_counter["n"]
        for kill_batch in (2, 5, 9):
            root = tempfile.mkdtemp(prefix="defit_")
            try:
                counter = {"n": 0}
                died = False
                try:
                    _quiet(_run, root, pool_kw={"batch_counter": counter,
                                                "kill_batch": kill_batch})
                except Killed:
                    died = True
                done_before = counter["n"] - 1        # the killed batch never finished
                counter2 = {"n": 0}
                (out, _l) = _quiet(_run, root, pool_kw={"batch_counter": counter2})
                res = out[0]
                check(f"kill in batch {kill_batch}: died, resumed, identical fit",
                      died and np.array_equal(res.x, ref.x) and res.fun == ref.fun,
                      f"died={died}")
                lost = done_before + counter2["n"] - batches_ref
                check(f"kill in batch {kill_batch}: at most one batch of work repeated",
                      0 <= lost <= 1, f"{lost}")
            finally:
                shutil.rmtree(root, ignore_errors=True)
    finally:
        shutil.rmtree(ref_root, ignore_errors=True)


def test_a_kill_in_the_polish_does_not_redo_the_search():
    print("\nKilled during the polish:")
    ref_root = tempfile.mkdtemp(prefix="defit_ref_")
    root = tempfile.mkdtemp(prefix="defit_")
    try:
        ref = _reference(ref_root)
        # The pool gets its own, uncounted objective: only the parent-side one is
        # called once for f0 and then by the polish, so call 30 is inside the
        # polish (which makes ~67 evaluations here).
        counter = {"n": 0}
        died = False
        try:
            _quiet(_run, root, objective=_objective_factory(counter, kill_at=30),
                   pool_objective=_objective_factory())
        except Killed:
            died = True
        part = _cache(root).load_partial()
        check("(setup) the kill landed in the polish, after the search finished",
              died and part["de_state"] is not None and part["nm_state"] is not None)
        pools = []
        (out, _l) = _quiet(_run, root, pools=pools)
        res = out[0]
        check("the resumed run did not build a pool or rerun the search",
              len(pools) == 0, f"{len(pools)} pool(s) built")
        check("and reaches the uninterrupted fit exactly",
              np.array_equal(res.x, ref.x) and res.fun == ref.fun,
              f"{res.fun!r} vs {ref.fun!r}")
    finally:
        shutil.rmtree(ref_root, ignore_errors=True)
        shutil.rmtree(root, ignore_errors=True)


def _calibrating_objective():
    """The real objective's nasty property: what a vector scores depends on which
    vector this process evaluated first (pyantigen.engine.Noise_floor calibrates on the first
    call and freezes it). A new process is a new call to this factory."""
    cal = {"c": None}

    def objective(x):
        x = np.asarray(x, dtype=float)
        if cal["c"] is None:
            cal["c"] = float(x[0])
        if x[0] > 4.5:
            return 1e10
        return float(np.sum((x - TARGET) ** 2)) + 10.0 * cal["c"]
    return objective


def test_a_resumed_fit_scores_against_the_original_noise_floors():
    """Measured on a 16-parameter aggregation fit: the same vector scored 2,319 in the first
    launch and -59.4 in the resumed one, because the resumed process calibrated its
    noise floors at the best point instead of at the start. Saved energies and new
    trials must be scores of the same function."""
    print("\nNoise-floor calibration across a resume:")
    import pyantigen.engine.Optimize as O
    ref_root = tempfile.mkdtemp(prefix="defit_ref_")
    try:
        (out, _l) = _quiet(_run, ref_root, objective=_calibrating_objective())
        ref = out[0]
        for where, kw in (("in the search", {"kill_batch": 5}),
                          ("in the polish", None)):
            root = tempfile.mkdtemp(prefix="defit_")
            try:
                died = False
                try:
                    if kw is not None:
                        _quiet(_run, root, objective=_calibrating_objective(),
                               pool_kw=kw)
                    else:
                        base = _calibrating_objective()
                        n = {"n": 0}

                        def dying(x):
                            n["n"] += 1
                            if n["n"] == 30:
                                raise Killed
                            return base(x)
                        _quiet(_run, root, objective=dying,
                               pool_objective=base)
                except Killed:
                    died = True
                # a new process: an objective that has calibrated on nothing yet
                (out, _l) = _quiet(_run, root, objective=_calibrating_objective())
                res = out[0]
                check(f"killed {where} and resumed in a new process: identical to "
                      f"an uninterrupted fit", died and np.array_equal(res.x, ref.x)
                      and res.fun == ref.fun, f"died={died} {res.fun!r} vs {ref.fun!r}")
            finally:
                shutil.rmtree(root, ignore_errors=True)

        root = tempfile.mkdtemp(prefix="defit_")
        saved = O._restore_calibration
        O._restore_calibration = lambda *a, **k: False
        try:
            try:
                _quiet(_run, root, objective=_calibrating_objective(),
                       pool_kw={"kill_batch": 5})
            except Killed:
                pass
            (out, _l) = _quiet(_run, root, objective=_calibrating_objective())
            check("(control) without the restore it is NOT -- the test has teeth",
                  out[0].fun != ref.fun, f"{out[0].fun!r} vs {ref.fun!r}")
        finally:
            O._restore_calibration = saved
            shutil.rmtree(root, ignore_errors=True)
    finally:
        shutil.rmtree(ref_root, ignore_errors=True)


def test_a_step_cap_that_fails_the_start_is_refused():
    print("\nA cap that is too small:")
    root = tempfile.mkdtemp(prefix="defit_")
    try:
        try:
            _quiet(_run, root, pool_kw={"fail_x0": True})
            msg = None
        except ValueError as exc:
            msg = str(exc)
        check("the run refuses to start", msg is not None)
        check("and says which setting to look at",
              msg is not None and "search_max_steps" in msg and "starting point" in msg,
              str(msg))
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_a_dead_pool_falls_back_to_serial_in_search_mode():
    print("\nThe pool breaks mid-search:")
    root = tempfile.mkdtemp(prefix="defit_")
    try:
        seen = []
        pools = []
        (out, log) = _quiet(_run, root, objective=_objective_factory(seen_caps=seen),
                            pool_kw={"fail_after": 3}, pools=pools)
        res = out[0]
        check("it says so and carries on", "evaluating serially" in log)
        check("the serial evaluations ran in search mode too",
              5000 in seen, f"caps seen by the parent: {sorted(set(seen), key=str)}")
        check("the fit still completes", res.fun < 1e-6, f"f={res.fun}")
        check("and the broken pool was shut down", pools[0].shut)
        check("search mode is off again afterwards", S._SEARCH_MAX_STEPS is None)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_no_pool_at_all_runs_serially():
    print("\nNo pool available:")
    root = tempfile.mkdtemp(prefix="defit_")
    try:
        seen = []
        objective = _objective_factory(seen_caps=seen)
        cache = _cache(root)
        opts = dict(OPTIONS)
        (res, _l) = _quiet(_run_fit_differential_evolution, objective, X0, BOUNDS,
                           opts, SCALES, cache, None, RunBudget(deadline=None),
                           lambda cap: None)
        check("a search with no pool still finds the optimum", res.fun < 1e-6,
              f"f={res.fun}")
        check("and its serial evaluations were in search mode", 5000 in seen)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_options_are_closed():
    print("\nOptions:")
    check("defaults are filled in", _de_options({})["popsize"] == 10
          and _de_options({})["search_max_steps"] == 10000)
    check("scipy's tol is accepted and ignored", "tol" not in _de_options({"tol": 1e-3}))
    try:
        _de_options({"strategy": "rand1bin", "atol": 0})
        msg = None
    except ValueError as exc:
        msg = str(exc)
    check("a scipy option this search lacks is refused, by name",
          msg is not None and "strategy" in msg and "atol" in msg, str(msg))
    check("the cap can be switched off", _de_options({"search_max_steps": None})
          ["search_max_steps"] is None)


def test_a_profile_never_inherits_the_global_method():
    print("\nProfile method:")
    check("a spec fitted by DE profiles with Nelder-Mead",
          _resolve_profile_optimizer("differential_evolution", {})[0] == "Nelder-Mead")
    check("an explicit profile_method is still respected",
          _resolve_profile_optimizer("differential_evolution",
                                     {"profile_method": "Powell"})[0] == "Powell")
    check("local methods are inherited exactly as before",
          _resolve_profile_optimizer("Powell", {})[0] == "Powell")
    check("and the profile's own kwargs come through",
          _resolve_profile_optimizer("differential_evolution",
                                     {"profile_optimizer_kwargs": {"a": 1}})[1] == {"a": 1})


def test_zz_every_check_passed():
    """``check`` records failures rather than raising, so pytest needs this."""
    assert not failures, "failed checks: " + ", ".join(failures)


if __name__ == "__main__":
    test_search_then_polish_reaches_the_optimum()
    test_polish_can_be_switched_off()
    test_stopped_over_and_over_it_is_the_same_fit()
    test_a_kill_in_the_search_costs_one_generation()
    test_a_kill_in_the_polish_does_not_redo_the_search()
    test_a_resumed_fit_scores_against_the_original_noise_floors()
    test_a_step_cap_that_fails_the_start_is_refused()
    test_a_dead_pool_falls_back_to_serial_in_search_mode()
    test_no_pool_at_all_runs_serially()
    test_options_are_closed()
    test_a_profile_never_inherits_the_global_method()
    print()
    if failures:
        print(f"{len(failures)} FAILED: {failures}")
        sys.exit(1)
    print("all checks passed")

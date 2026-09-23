"""pyantigen.engine.Nelder_mead against scipy, and its stop/resume guarantees.

Run from the repository root:
    python tests/engine/test_nelder_mead.py

Two claims carry the whole design, and both are checked as hard equalities, not
as "close enough":

1. **It is scipy's optimizer.** Given the same problem it evaluates the same
   points in the same order and returns the same result. Nothing computed under
   ``scipy.optimize.minimize`` is invalidated by using this instead.
2. **Stopping costs nothing.** A run stopped anywhere, written to JSON, read
   back and continued makes exactly the evaluations an uninterrupted run would
   have made -- none repeated, none skipped -- and a hard kill loses at most the
   iteration in flight.
"""
import json
import os
import sys
import warnings

import numpy as np
from scipy.optimize import Bounds, minimize

sys.path.insert(0, os.getcwd())

from pyantigen.engine.Nelder_mead import (                                    # noqa: E402
    STATUS_STOPPED, best_of, bounds_arrays, can_run, minimize_nelder_mead,
    nelder_mead, new_state, state_from_json, state_to_json,
)

failures = []


def check(name, cond, detail=""):
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}  {detail}")
        failures.append(name)


# ---------------------------------------------------------------------------
# Objectives chosen to be awkward
# ---------------------------------------------------------------------------

def _rosen(n):
    return lambda x: float(np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2
                                  + (1 - x[:-1]) ** 2))


def _ties(n):
    """Plateaus: exact ties in f, which is where a different sort would show."""
    r = _rosen(n)
    return lambda x: round(r(x), 1)


def _sphere(n):
    c = np.linspace(-2, 3, n)
    return lambda x: float(np.sum((x - c) ** 2))


def _cases():
    out = []
    for name, mk in (("rosen", _rosen), ("ties", _ties), ("sphere", _sphere)):
        for n in (2, 5, 12):
            for adaptive in (False, True):
                for bounded in (False, True):
                    for opts in ({}, {"maxfev": 60}, {"maxiter": 25},
                                 {"xatol": 1e-8, "fatol": 1e-8,
                                  "maxfev": 900, "maxiter": 900}):
                        out.append((name, mk, n, adaptive, bounded, opts))
    return out


def _problem(rng, n, bounded):
    x0 = rng.uniform(-1.5, 2.0, n)
    lb = np.full(n, -1.0) if bounded else None
    ub = np.full(n, 1.5) if bounded else None
    return x0, lb, ub


def _recorder(f):
    seq = []

    def g(x):
        seq.append(np.array(x, copy=True))
        return f(x)
    return g, seq


def _same_seq(a, b):
    return len(a) == len(b) and all(np.array_equal(u, v) for u, v in zip(a, b))


def _run_scipy(f, x0, lb, ub, adaptive, opts):
    g, seq = _recorder(f)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = minimize(g, x0, method="Nelder-Mead",
                       bounds=None if lb is None else Bounds(lb, ub),
                       options={"adaptive": adaptive, **opts})
    return res, seq


# ---------------------------------------------------------------------------
# 1. Same optimizer as scipy
# ---------------------------------------------------------------------------

def test_matches_scipy_call_for_call():
    print("\nAgainst scipy:")
    rng = np.random.default_rng(0)
    bad = []
    cases = _cases()
    for name, mk, n, adaptive, bounded, opts in cases:
        f = mk(n)
        x0, lb, ub = _problem(rng, n, bounded)
        rs, sa = _run_scipy(f, x0, lb, ub, adaptive, opts)
        g, sb = _recorder(f)
        st = new_state(x0, lb, ub)
        rm = nelder_mead(g, st, lb=lb, ub=ub, adaptive=adaptive, **opts)
        same = (_same_seq(sa, sb) and rs.nfev == rm.nfev and rs.nit == rm.nit
                and rs.status == rm.status and rs.message == rm.message
                and np.array_equal(rs.x, rm.x) and rs.fun == rm.fun)
        if not same:
            bad.append((name, n, adaptive, bounded, opts))
    check(f"{len(cases)} problems: identical evaluation sequence, nfev, nit, "
          f"status, message, x and fun", not bad, str(bad[:3]))


def test_the_drop_in_takes_minimizes_arguments():
    print("\nThe minimize() drop-in:")
    n = 6
    f = _rosen(n)
    x0 = np.linspace(-0.5, 1.4, n)
    # A list of pairs with None entries is what the engine passes.
    bounds = [(-1.0, 2.0), (None, 2.0), (-1.0, None), (None, None),
              (-1.0, 2.0), (-1.0, 2.0)]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        g, sa = _recorder(f)
        rs = minimize(g, x0, method="Nelder-Mead", bounds=bounds, tol=1e-6,
                      options={"maxiter": 300})
    g, sb = _recorder(f)
    rm, _ = minimize_nelder_mead(g, x0, bounds=bounds, tol=1e-6,
                                 options={"maxiter": 300})
    check("bounds as a list of pairs with None, plus tol, match scipy",
          _same_seq(sa, sb) and np.array_equal(rs.x, rm.x)
          and rs.fun == rm.fun and rs.nfev == rm.nfev, f"{rs.nfev} vs {rm.nfev}")

    # extra args are forwarded the way minimize forwards them
    def shifted(x, c):
        return float(np.sum((x - c) ** 2))
    rs = minimize(shifted, x0, args=(0.7,), method="Nelder-Mead")
    rm, _ = minimize_nelder_mead(shifted, x0, args=(0.7,))
    check("args are forwarded", np.array_equal(rs.x, rm.x), str(rm.x))

    # a caller-supplied initial simplex
    sim = np.vstack([x0, x0 + np.eye(n) * 0.3])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        g, sa = _recorder(f)
        rs = minimize(g, x0, method="Nelder-Mead",
                      options={"initial_simplex": sim, "maxfev": 200})
    g, sb = _recorder(f)
    rm, _ = minimize_nelder_mead(g, x0, options={"initial_simplex": sim,
                                                 "maxfev": 200})
    check("an initial_simplex is honoured identically",
          _same_seq(sa, sb) and np.array_equal(rs.x, rm.x))

    lb, ub = bounds_arrays([(None, None)] * 3, 3)
    check("all-unbounded collapses to no bounds", lb is None and ub is None)
    check("options this does not implement are declined",
          can_run({"maxiter": 5, "xatol": 1e-3}) and not can_run({"ftol": 1})
          and can_run({}, {"tol": 1e-3}) and not can_run({}, {"jac": 1}))


# ---------------------------------------------------------------------------
# 2. Stopping is free
# ---------------------------------------------------------------------------

def test_a_clean_stop_and_resume_repeats_nothing():
    print("\nStop, write, read, continue:")
    rng = np.random.default_rng(1)
    total = ok = 0
    bad = []
    for name, mk, n, adaptive, bounded, opts in _cases()[::7]:
        f = mk(n)
        x0, lb, ub = _problem(rng, n, bounded)
        rs, sa = _run_scipy(f, x0, lb, ub, adaptive, opts)
        kw = dict(lb=lb, ub=ub, adaptive=adaptive, **opts)

        points = []
        nelder_mead(f, new_state(x0, lb, ub), on_step=lambda s: points.append(1),
                    **kw)
        step = max(1, len(points) // 25)
        for stop_at in range(1, len(points) + 1, step):
            g, seq = _recorder(f)
            st = new_state(x0, lb, ub)
            seen = [0]

            def on_step(_s):
                seen[0] += 1
            res = nelder_mead(g, st, on_step=on_step,
                              should_stop=lambda: seen[0] >= stop_at, **kw)
            if res.status != STATUS_STOPPED:
                continue
            # A real file round trip, not a copy of the dict.
            st2 = state_from_json(json.loads(json.dumps(state_to_json(st))))
            res2 = nelder_mead(g, st2, **kw)
            total += 1
            same = (_same_seq(seq, sa) and rs.nfev == res2.nfev
                    and rs.nit == res2.nit and rs.status == res2.status
                    and np.array_equal(rs.x, res2.x))
            ok += bool(same)
            if not same:
                bad.append((name, n, adaptive, bounded, opts, stop_at))
    check(f"{total} stops: the joined run is scipy's run -- same evaluations, "
          f"none repeated", total > 300 and ok == total, f"{ok}/{total} {bad[:2]}")


def test_a_hard_kill_loses_at_most_the_iteration_in_flight():
    print("\nA kill mid-iteration:")
    rng = np.random.default_rng(2)

    class Killed(Exception):
        pass

    total = ok = 0
    lost = []
    for name, mk, n, adaptive, bounded, opts in _cases()[::11]:
        f = mk(n)
        x0, lb, ub = _problem(rng, n, bounded)
        kw = dict(lb=lb, ub=ub, adaptive=adaptive, **opts)
        ref = nelder_mead(f, new_state(x0, lb, ub), **kw)
        for die_at in range(3, ref.nfev, max(1, ref.nfev // 20)):
            calls = [0]

            def g(x):
                calls[0] += 1
                if calls[0] == die_at:
                    raise Killed
                return f(x)
            st = new_state(x0, lb, ub)
            disk = {}

            def on_step(s):
                disk["s"] = json.dumps(state_to_json(s))
            try:
                nelder_mead(g, st, on_step=on_step, **kw)
                continue
            except Killed:
                pass
            st2 = (state_from_json(json.loads(disk["s"])) if disk
                   else new_state(x0, lb, ub))
            lost.append(die_at - 1 - st2["nfev"])
            res2 = nelder_mead(f, st2, **kw)
            total += 1
            ok += bool(ref.nfev == res2.nfev and ref.nit == res2.nit
                       and ref.status == res2.status
                       and np.array_equal(ref.x, res2.x) and ref.fun == res2.fun)
    check(f"{total} kills: every resume reaches the identical final result",
          total > 150 and ok == total, f"{ok}/{total}")
    check("no more than one evaluation was ever lost to a kill",
          max(lost) <= 1, f"max {max(lost)}")


def test_stopping_before_any_evaluation_and_mid_build():
    print("\nStopping early in the build:")
    f = _sphere(4)
    x0 = np.zeros(4)
    st = new_state(x0)
    res = nelder_mead(f, st, should_stop=lambda: True)
    check("a stop before the first evaluation returns cleanly",
          res.status == STATUS_STOPPED and st["nfev"] == 0)
    x, fx = best_of(st)
    check("and reports nothing evaluated rather than a fake best",
          x is None and fx == np.inf)

    st = new_state(x0)
    n_calls = [0]

    def stop_after_three():
        return n_calls[0] >= 3

    def g(x):
        n_calls[0] += 1
        return f(x)
    res = nelder_mead(g, st, should_stop=stop_after_three)
    check("a stop mid-build keeps the vertices already evaluated",
          res.status == STATUS_STOPPED and st["k"] == 3
          and np.isfinite(st["fsim"][:3]).all()
          and np.isinf(st["fsim"][3:]).all(), str(st["fsim"]))
    res2 = nelder_mead(f, state_from_json(json.loads(
        json.dumps(state_to_json(st)))))
    ref = nelder_mead(f, new_state(x0))
    check("and finishing from there gives the uninterrupted answer",
          np.array_equal(res2.x, ref.x) and res2.nfev == ref.nfev)


def test_one_allowance_across_launches():
    print("\nCounters carry across a resume:")
    f = _rosen(4)
    x0 = np.array([-1.0, 0.5, 0.2, 1.4])
    whole = nelder_mead(f, new_state(x0), maxfev=120, maxiter=120)

    st = new_state(x0)
    first = nelder_mead(f, st, maxfev=50, maxiter=120)
    check("the first launch stops on its evaluation cap",
          first.status == 1 and first.nfev == 50, f"{first.status} {first.nfev}")
    # The cap is a total across launches: the second is given the same number.
    second = nelder_mead(f, state_from_json(json.loads(
        json.dumps(state_to_json(st)))), maxfev=120, maxiter=120)
    check("a resume spends only what is left of the total",
          second.nfev == whole.nfev == 120, f"{second.nfev} vs {whole.nfev}")

    seeded = new_state(x0, nfev=118)
    res = nelder_mead(f, seeded, maxfev=120)
    check("counters seeded from a record are honoured",
          res.nfev == 120 and res.status == 1, f"{res.nfev} {res.status}")


# ---------------------------------------------------------------------------
# 3. A bad file is a miss
# ---------------------------------------------------------------------------

def test_a_damaged_state_is_a_miss_not_a_crash():
    print("\nDamaged state files:")
    f = _sphere(3)
    st = new_state(np.zeros(3))
    nelder_mead(f, st, should_stop=lambda: st["nfev"] >= 2)
    good = state_to_json(st)
    check("the state we wrote reads back", state_from_json(good, 3) is not None)
    check("a state for a different problem size is refused",
          state_from_json(good, 4) is None)
    for label, mutate in (
            ("wrong format", lambda d: d.update(format="nm-v0")),
            ("missing sim", lambda d: d.pop("sim")),
            ("ragged sim", lambda d: d.update(sim=[[0.0, 0.0], [1.0]])),
            ("nan in sim", lambda d: d["sim"][0].__setitem__(0, float("nan"))),
            ("short fsim", lambda d: d.update(fsim=[0.0])),
            ("bad phase", lambda d: d.update(phase="later")),
            ("bad k", lambda d: d.update(k=99)),
            ("negative nfev", lambda d: d.update(nfev=-1))):
        d = json.loads(json.dumps(good))
        mutate(d)
        check(f"{label} is refused", state_from_json(d, 3) is None)
    check("non-dict input is refused", state_from_json("nope") is None
          and state_from_json(None) is None)


def test_zz_every_check_passed():
    """``check`` records failures rather than raising, so pytest needs this."""
    assert not failures, "failed checks: " + ", ".join(failures)


if __name__ == "__main__":
    test_matches_scipy_call_for_call()
    test_the_drop_in_takes_minimizes_arguments()
    test_a_clean_stop_and_resume_repeats_nothing()
    test_a_hard_kill_loses_at_most_the_iteration_in_flight()
    test_stopping_before_any_evaluation_and_mid_build()
    test_one_allowance_across_launches()
    test_a_damaged_state_is_a_miss_not_a_crash()
    print()
    if failures:
        print(f"{len(failures)} FAILED: {failures}")
        sys.exit(1)
    print("all checks passed")

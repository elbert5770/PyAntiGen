"""pyantigen.engine.Differential_evolution: it searches, it survives failure, it resumes.

Run from the repository root:
    python tests/engine/test_differential_evolution.py

The claims that matter are the ones scipy's version cannot make here:

1. **Resuming is exact.** A search stopped between any two generations (or part
   way through evaluating the initial population), written to JSON and read
   back, continues to the very same population, energies, RNG state and sequence
   of evaluated vectors as an uninterrupted search.
2. **Failure is ordinary.** Vectors the objective cannot score -- NaN, inf, or
   its failure value, which on the real model is most of the search box -- never
   crash the search, never win, and are counted.
3. **Bounds are walls, not sticky.** No trial ever leaves the box, and an escape
   is redrawn rather than clipped, so members do not pile onto the boundary.
4. It is differential evolution: it finds the basin of ordinary and multimodal
   test problems, and the starting point it is given is never lost.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.getcwd())

from pyantigen.engine.Differential_evolution import (                         # noqa: E402
    STATUS_CONVERGED, STATUS_MAXITER, STATUS_STOPPED, best_of,
    differential_evolution, new_state, state_from_json, state_to_json,
)

failures = []


def check(name, cond, detail=""):
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}  {detail}")
        failures.append(name)


# ---------------------------------------------------------------------------
# Objectives
# ---------------------------------------------------------------------------

def sphere(x):
    return float(np.sum((x - np.linspace(-1.0, 1.5, len(x))) ** 2))


def rosen(x):
    return float(np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2))


def rastrigin(x):
    return float(10 * len(x) + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x)))


def batch(f, log=None):
    def fn(X):
        if log is not None:
            log.append(np.array(X, copy=True))
        return np.array([f(x) for x in X])
    return fn


BOX4 = [(-5.0, 5.0)] * 4
BOX3 = [(-5.12, 5.12)] * 3


# ---------------------------------------------------------------------------
# 4. It is differential evolution
# ---------------------------------------------------------------------------

def test_it_finds_the_basin():
    print("\nSearching:")
    for name, f, box, tol in (("sphere", sphere, BOX4, 1e-3),
                              ("rosenbrock", rosen, BOX4, 1.0),
                              ("rastrigin", rastrigin, BOX3, 1.5)):
        st = new_state(box, popsize=15, seed=1)
        res = differential_evolution(batch(f), st, maxiter=400, stall_gens=40,
                                     stall_tol=1e-9)
        check(f"{name}: reaches the basin", res.fun < tol,
              f"f={res.fun:.4g} after {res.nit} generations")
    st = new_state(BOX4, popsize=10, seed=2)
    res = differential_evolution(batch(sphere), st, maxiter=500, stall_gens=25,
                                 stall_tol=1e-8)
    check("it stops on its own when the best value stops improving",
          res.status == STATUS_CONVERGED and res.nit < 500,
          f"status {res.status}, {res.nit} generations")
    st = new_state(BOX4, popsize=5, seed=2)
    res = differential_evolution(batch(rosen), st, maxiter=3)
    check("and reports running out of generations when it does",
          res.status == STATUS_MAXITER and res.nit == 3 and not res.success)


def test_the_starting_point_is_never_lost():
    print("\nThe caller's starting point:")
    x0 = np.array([0.5, 0.5, 0.5, 0.5])
    st = new_state(BOX4, popsize=8, x0=x0, seed=3)
    check("it is member 0 of the initial population", np.array_equal(st["pop"][0], x0))
    res = differential_evolution(batch(rosen), st, maxiter=5)
    check("the result is never worse than the start", res.fun <= rosen(x0) + 1e-12,
          f"{res.fun} vs {rosen(x0)}")
    outside = np.array([9.0, -9.0, 0.0, 0.0])
    st = new_state(BOX4, popsize=5, x0=outside, seed=3)
    check("a start outside the box is clipped into it",
          np.array_equal(st["pop"][0], np.clip(outside, -5, 5)))

    st = new_state(BOX4, popsize=10, x0=x0, seed=4, init_radius=0.5)
    near = np.abs(st["pop"][1:] - x0)
    check("with an init radius the rest of the population starts near it",
          near.max() <= 0.5 + 1e-12, f"{near.max()}")
    st = new_state([(0.0, 5.0)] * 4, popsize=10, x0=np.full(4, 0.1), seed=4,
                   init_radius=0.5)
    check("and that cloud is still clipped to the bounds",
          st["pop"].min() >= 0.0 and st["pop"].max() <= 0.6 + 1e-12)


def test_the_same_seed_is_the_same_search():
    print("\nSeeding:")
    outs = []
    for seed in (5, 5, 6):
        st = new_state(BOX4, popsize=5, seed=seed)
        differential_evolution(batch(rosen), st, maxiter=6)
        outs.append(st["pop"].copy())
    check("the same seed reproduces the population exactly",
          np.array_equal(outs[0], outs[1]))
    check("a different seed does not", not np.array_equal(outs[0], outs[2]))


# ---------------------------------------------------------------------------
# 3. Bounds are walls
# ---------------------------------------------------------------------------

def test_no_trial_leaves_the_box_and_none_is_clipped_onto_it():
    print("\nBounds:")
    log = []
    box = [(0.0, 1.0)] * 4
    st = new_state(box, popsize=10, seed=7)
    # A rugged objective whose optimum sits ON a wall, where clipping would pile
    # members up: this is the geometry that hurt on the real model.
    differential_evolution(batch(lambda x: float(np.sum(x))), st, maxiter=30,
                           stall_gens=0)
    log.clear()
    st = new_state(box, popsize=10, seed=7)
    differential_evolution(batch(lambda x: float(np.sum(x)), log), st, maxiter=30,
                           stall_gens=0)
    allx = np.vstack(log)
    check("every vector ever evaluated is inside the box",
          allx.min() >= 0.0 and allx.max() <= 1.0, f"{allx.min()} {allx.max()}")
    trials = np.vstack(log[1:])       # log[0] is the initial population
    exactly_on = np.mean((trials == 0.0) | (trials == 1.0))
    check("an escaped component is redrawn, not clipped onto the wall",
          exactly_on < 0.001, f"{100 * exactly_on:.3f}% of components on a wall")


# ---------------------------------------------------------------------------
# 2. Failure is ordinary
# ---------------------------------------------------------------------------

def test_failures_never_win_and_never_crash():
    print("\nFailure:")

    def half_infeasible(x):
        if x[0] > 1.0:
            return float("nan")
        if x[1] > 2.0:
            return float("inf")
        if x[2] > 3.0:
            return 1e10
        return sphere(x)

    st = new_state(BOX4, popsize=12, seed=8)
    infos = []
    res = differential_evolution(batch(half_infeasible), st, maxiter=200,
                                 stall_gens=30, stall_tol=1e-9,
                                 on_generation=lambda s, i: infos.append(i))
    check("it still finds the feasible optimum", res.fun < 0.05,
          f"f={res.fun:.4g}")
    check("no member holds a non-finite energy",
          np.all(np.isfinite(st["energies"])))
    check("the failed vectors were counted, generation by generation",
          any(i["n_failed"] > 0 for i in infos)
          and all("n_feasible" in i for i in infos))
    check("the result is never a failure", res.fun < 1e10)

    def all_fail(x):
        return float("nan")
    st = new_state(BOX4, popsize=5, seed=8)
    res = differential_evolution(batch(all_fail), st, maxiter=3, stall_gens=0)
    check("even a search where everything fails returns cleanly",
          res.fun == 1e10 and res.nit == 3, f"{res.fun} {res.nit}")


def test_a_plateau_does_not_freeze_the_population():
    """Ties replace, so a member on a flat region can drift instead of sticking."""
    print("\nPlateaus:")
    st = new_state(BOX4, popsize=6, seed=9)
    differential_evolution(batch(lambda x: 7.0), st, maxiter=1, stall_gens=0)
    initial = new_state(BOX4, popsize=6, seed=9)["pop"]
    moved = np.mean(np.any(st["pop"] != initial, axis=1))
    check("on a constant objective the population moves", moved > 0.8,
          f"{100 * moved:.0f}% of members moved")


# ---------------------------------------------------------------------------
# 1. Resuming is exact
# ---------------------------------------------------------------------------

def _run(stop_after=None, maxiter=25, seed=11, init_chunk=None, log=None,
         state=None):
    st = state or new_state(BOX4, popsize=6, seed=seed)
    counter = {"g": 0, "chunks": 0}

    def on_gen(_s, _i):
        counter["g"] += 1

    def on_chunk(_s):
        counter["chunks"] += 1

    def should_stop():
        if stop_after is None:
            return False
        return counter["g"] + counter["chunks"] >= stop_after

    res = differential_evolution(batch(rosen, log), st, maxiter=maxiter,
                                 stall_gens=0, init_chunk=init_chunk,
                                 on_generation=on_gen, on_init_chunk=on_chunk,
                                 should_stop=should_stop)
    return res, st


def _same(a, b):
    return (np.array_equal(a["pop"], b["pop"])
            and np.array_equal(a["energies"], b["energies"])
            and a["gen"] == b["gen"] and a["nfev"] == b["nfev"]
            and a["best_history"] == b["best_history"]
            and a["rng"] == b["rng"])


def test_a_stopped_search_continues_exactly():
    print("\nStop, write, read, continue:")
    log_ref = []
    _r, ref = _run(log=log_ref)
    bad = []
    n_stops = 0
    for stop_after in range(1, 25):
        log_a = []
        res, st = _run(stop_after=stop_after, log=log_a)
        if res.status != STATUS_STOPPED:
            continue
        n_stops += 1
        # A real file round trip, not a copy of the dict.
        st2 = state_from_json(json.loads(json.dumps(state_to_json(st))))
        log_b = []
        _r2, st2 = _run(state=st2, log=log_b)
        joined = log_a + log_b
        same_calls = (len(joined) == len(log_ref)
                      and all(np.array_equal(u, v)
                              for u, v in zip(joined, log_ref)))
        if not (_same(st2, ref) and same_calls):
            bad.append(stop_after)
    check(f"{n_stops} stops: the joined search IS the uninterrupted search "
          f"(population, energies, RNG, and every batch evaluated)",
          n_stops >= 15 and not bad, f"{n_stops} stops, mismatches at {bad[:5]}")


def test_a_stop_while_the_initial_population_is_still_evaluating():
    print("\nStopped mid-initialisation:")
    _r, ref = _run(init_chunk=7)
    res, st = _run(init_chunk=7, stop_after=2)
    check("the stop landed inside the initial population",
          res.status == STATUS_STOPPED and st["phase"] == "init"
          and 0 < np.sum(np.isfinite(st["energies"])) < len(st["pop"]),
          f"phase={st['phase']}")
    st2 = state_from_json(json.loads(json.dumps(state_to_json(st))))
    log = []
    _r2, st2 = _run(state=st2, init_chunk=7, log=log)
    check("finishing from there reaches the uninterrupted result",
          _same(st2, ref))
    n_done = int(np.sum(np.isfinite(st["energies"])))
    check("and only the members not yet evaluated were evaluated again",
          sum(len(x) for x in log[:1]) <= len(st["pop"]) - n_done + 7,
          str([len(x) for x in log[:3]]))


def test_evaluation_counts_and_batch_shapes():
    print("\nBatches:")
    log = []
    st = new_state(BOX4, popsize=5, seed=12)
    m = len(st["pop"])
    res = differential_evolution(batch(sphere, log), st, maxiter=4, stall_gens=0)
    check("the initial population is one batch of the full size",
          log[0].shape == (m, 4), str(log[0].shape))
    check("every generation is one batch of one trial per member",
          all(x.shape == (m, 4) for x in log[1:]) and len(log) == 1 + 4,
          str([x.shape for x in log]))
    check("nfev counts every evaluation", res.nfev == m * 5, f"{res.nfev} vs {m * 5}")


# ---------------------------------------------------------------------------
# A damaged state is a miss
# ---------------------------------------------------------------------------

def test_a_damaged_state_is_a_miss_not_a_crash():
    print("\nDamaged state:")
    st = new_state(BOX4, popsize=5, seed=13)
    differential_evolution(batch(sphere), st, maxiter=2, stall_gens=0)
    good = state_to_json(st)
    check("the state we wrote reads back", state_from_json(good, 4, len(st["pop"]))
          is not None)
    check("a state for another problem size is refused",
          state_from_json(good, 5) is None
          and state_from_json(good, 4, len(st["pop"]) + 1) is None)
    for label, mutate in (
            ("wrong format", lambda d: d.update(format="de-v0")),
            ("missing pop", lambda d: d.pop("pop")),
            ("ragged pop", lambda d: d.update(pop=[[1.0], [1.0, 2.0]])),
            ("nan in pop", lambda d: d["pop"][0].__setitem__(0, float("nan"))),
            ("short energies", lambda d: d.update(energies=[1.0])),
            ("bad phase", lambda d: d.update(phase="later")),
            ("negative gen", lambda d: d.update(gen=-1)),
            ("bad bounds", lambda d: d.update(bounds=[[0.0, 1.0]])),
            ("broken rng", lambda d: d.update(rng={"bit_generator": "nope"}))):
        d = json.loads(json.dumps(good))
        mutate(d)
        check(f"{label} is refused", state_from_json(d, 4) is None)
    check("non-dict input is refused",
          state_from_json(None) is None and state_from_json("x") is None)


def test_bad_bounds_are_refused_up_front():
    print("\nBounds validation:")
    for label, box in (("infinite", [(-np.inf, 1.0)] * 2),
                       ("inverted", [(1.0, 0.0)] * 2),
                       ("degenerate", [(1.0, 1.0)] * 2)):
        try:
            new_state(box)
            ok = False
        except ValueError:
            ok = True
        check(f"{label} bounds raise", ok)


def test_zz_every_check_passed():
    """``check`` records failures rather than raising, so pytest needs this."""
    assert not failures, "failed checks: " + ", ".join(failures)


if __name__ == "__main__":
    test_it_finds_the_basin()
    test_the_starting_point_is_never_lost()
    test_the_same_seed_is_the_same_search()
    test_no_trial_leaves_the_box_and_none_is_clipped_onto_it()
    test_failures_never_win_and_never_crash()
    test_a_plateau_does_not_freeze_the_population()
    test_a_stopped_search_continues_exactly()
    test_a_stop_while_the_initial_population_is_still_evaluating()
    test_evaluation_counts_and_batch_shapes()
    test_a_damaged_state_is_a_miss_not_a_crash()
    test_bad_bounds_are_refused_up_front()
    print()
    if failures:
        print(f"{len(failures)} FAILED: {failures}")
        sys.exit(1)
    print("all checks passed")

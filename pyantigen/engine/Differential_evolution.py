"""Differential evolution for an objective that is expensive and often fails.

Why this is not ``scipy.optimize.differential_evolution``. Three things this
project needs are not things scipy exposes:

* **Resumability.** A generation of the aggregation_silk search is minutes of
  work on a 40-core node, a search is hours, and the node can be taken away
  without notice. scipy keeps its population and random-number stream inside the
  solver and hands neither back, so a killed search restarts from nothing. Here
  the whole state -- population, energies, generation counter and the RNG's own
  state -- is a JSON-able dict written after every generation, and resuming from
  it continues the *same* search: the run stopped anywhere and continued makes
  exactly the draws an uninterrupted one would (checked in
  ``tests/test_differential_evolution.py``).
* **A batch per generation.** The objective is evaluated on a process pool, and
  the pool wants a whole generation at once. The algorithm here is the deferred
  variant -- every trial vector is built from the population as it stood at the
  start of the generation, then all are evaluated, then all are accepted or
  rejected -- which is exactly the shape ``ParallelEvaluator.evaluate_batch``
  takes. (scipy's ``updating="deferred"`` + ``vectorized=True`` can do this too,
  but only without the first bullet.)
* **Failure as a first-class outcome.** On this model most of a six-decade
  search box is either astronomically bad (NLL 1e6-1e13 against ~500 at the
  optimum) or a vector the integrator cannot solve, which the objective reports
  as the failure value. Both are ordinary here, so both are counted and logged
  per generation rather than left to look like a hang.

The algorithm is scipy's default -- ``best1bin``, mutation dithered per
generation in [0.5, 1), binomial crossover at 0.7, an out-of-bounds component
redrawn uniformly inside the bounds -- so its behaviour is the familiar one. Two
deliberate differences: a trial that TIES its target replaces it, because the
objective has large exact plateaus (a model with no plaque scores the same NLL to
every digit however the other parameters are set) and a strict ``<`` freezes
every member sitting on one; and one member of the initial population is the
caller's own starting point, so the search can only match or beat it.

It finds a basin, not a minimum. Convergence to the tolerance the profile needs is
Nelder-Mead's job, and the caller polishes with it afterwards.
"""

import numpy as np
from scipy.optimize import OptimizeResult

STATE_FORMAT = "de-v1"

STATUS_CONVERGED = 0     # stalled or collapsed: nothing more to find
STATUS_MAXITER = 1       # ran out of generations
STATUS_STOPPED = 3       # asked to stop; resumable (matches Nelder_mead's code)

_MESSAGES = {
    STATUS_CONVERGED: "Search converged: the best value stopped improving.",
    STATUS_MAXITER: "Maximum number of generations has been exceeded.",
    STATUS_STOPPED: "Stopped on request; resumable.",
}

DEFAULT_FAILURE_VALUE = 1e10


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

def _latin_hypercube(rng, m, lo, hi):
    """*m* points stratified in every dimension, columns independently shuffled."""
    n = len(lo)
    u = (np.arange(m)[:, None] + rng.random((m, n))) / m
    u = rng.permuted(u, axis=0)
    return lo + u * (hi - lo)


def new_state(bounds, popsize=15, x0=None, seed=None, init_radius=None):
    """A fresh, unevaluated population.

    *popsize* is a multiplier, as in scipy: the population holds ``popsize * n``
    members (at least 5). Member 0 is *x0* clipped into the bounds when given.
    *init_radius*, when given, draws the rest from a Latin hypercube within that
    distance of *x0* (in optimizer units -- decades, for a log10 parameter),
    clipped to the bounds, instead of over the whole box; the bounds remain the
    wall, and mutation is free to leave the initial cloud.
    """
    bounds = np.asarray(bounds, dtype=float)
    if bounds.ndim != 2 or bounds.shape[1] != 2:
        raise ValueError("bounds must be a sequence of (min, max) pairs")
    lo, hi = bounds[:, 0], bounds[:, 1]
    if not (np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))):
        raise ValueError("differential evolution needs finite bounds on every "
                         "parameter")
    if np.any(lo >= hi):
        raise ValueError("every lower bound must be below its upper bound")
    n = len(lo)
    m = max(5, int(popsize) * n)
    rng = np.random.default_rng(seed)

    if x0 is not None and init_radius is not None:
        x0c = np.clip(np.asarray(x0, dtype=float), lo, hi)
        box_lo = np.maximum(lo, x0c - float(init_radius))
        box_hi = np.minimum(hi, x0c + float(init_radius))
        pop = _latin_hypercube(rng, m, box_lo, box_hi)
    else:
        pop = _latin_hypercube(rng, m, lo, hi)
    if x0 is not None:
        pop[0] = np.clip(np.asarray(x0, dtype=float), lo, hi)

    return {"pop": pop, "energies": np.full(m, np.inf), "bounds": bounds,
            "gen": 0, "nfev": 0, "phase": "init", "best_history": [],
            "rng": rng.bit_generator.state}


def state_to_json(state):
    """Plain lists and numbers, safe for ``json.dump``.

    Unevaluated members (energy ``inf``) are written as ``null``. The RNG state
    holds integers wider than 64 bits, which Python's json round-trips exactly.
    """
    return {
        "format": STATE_FORMAT,
        "pop": np.asarray(state["pop"], dtype=float).tolist(),
        "energies": [None if not np.isfinite(v) else float(v)
                     for v in state["energies"]],
        "bounds": np.asarray(state["bounds"], dtype=float).tolist(),
        "gen": int(state["gen"]), "nfev": int(state["nfev"]),
        "phase": state["phase"],
        "best_history": [float(v) for v in state["best_history"]],
        "rng": state["rng"],
    }


def state_from_json(data, n=None, m=None):
    """The state in *data*, or None if it is not a usable one.

    Every failure is a miss, as everywhere in the checkpoint code: a damaged
    file costs the resume it would have given, never the run. *n* and *m* are
    the parameter count and population size the caller expects.
    """
    try:
        if not isinstance(data, dict) or data.get("format") != STATE_FORMAT:
            return None
        pop = np.array(data["pop"], dtype=float)
        energies = np.array([np.inf if v is None else v
                             for v in data["energies"]], dtype=float)
        bounds = np.array(data["bounds"], dtype=float)
        if pop.ndim != 2 or bounds.shape != (pop.shape[1], 2):
            return None
        if n is not None and pop.shape[1] != n:
            return None
        if m is not None and pop.shape[0] != m:
            return None
        if energies.shape != (pop.shape[0],) or np.any(np.isnan(energies)):
            return None
        if not (np.all(np.isfinite(pop)) and np.all(np.isfinite(bounds))):
            return None
        phase = data["phase"]
        if phase not in ("init", "evolve"):
            return None
        gen, nfev = int(data["gen"]), int(data["nfev"])
        if gen < 0 or nfev < 0:
            return None
        history = [float(v) for v in data["best_history"]]
        rng_state = data["rng"]
        # Prove it restores, rather than discovering it mid-run.
        probe = np.random.default_rng()
        probe.bit_generator.state = rng_state
    except (KeyError, TypeError, ValueError):
        return None
    return {"pop": pop, "energies": energies, "bounds": bounds, "gen": gen,
            "nfev": nfev, "phase": phase, "best_history": history,
            "rng": rng_state}


def best_of(state):
    """``(x, f)`` of the best evaluated member, or ``(None, inf)``."""
    e = np.asarray(state["energies"], dtype=float)
    if not np.any(np.isfinite(e)):
        return None, float("inf")
    i = int(np.argmin(e))
    return np.array(state["pop"][i], dtype=float), float(e[i])


# ---------------------------------------------------------------------------
# The search
# ---------------------------------------------------------------------------

def _trials(rng, pop, energies, bounds, scale, recombination):
    """One trial vector per member, all from the population as it now stands."""
    m, n = pop.shape
    lo, hi = bounds[:, 0], bounds[:, 1]
    best = pop[int(np.argmin(energies))]

    # Two distinct partners, neither the member itself, per member.
    r1 = np.empty(m, dtype=int)
    r2 = np.empty(m, dtype=int)
    for i in range(m):
        pick = rng.choice(m - 1, size=2, replace=False)
        pick = np.where(pick >= i, pick + 1, pick)
        r1[i], r2[i] = pick

    mutant = best + scale * (pop[r1] - pop[r2])
    cross = rng.random((m, n)) < recombination
    cross[np.arange(m), rng.integers(0, n, size=m)] = True   # at least one gene
    trial = np.where(cross, mutant, pop)

    # A component that left the box is redrawn inside it, as scipy does, rather
    # than clipped to the wall: clipping would pile members onto the boundary,
    # which is where this model's integrator is at its worst.
    outside = (trial < lo) | (trial > hi)
    if np.any(outside):
        redraw = lo + rng.random((m, n)) * (hi - lo)
        trial = np.where(outside, redraw, trial)
    return trial


def _values(batch_fn, xs, failure_value):
    """Objective values for the rows of *xs*, with anything non-finite failed."""
    v = np.asarray(batch_fn(np.asarray(xs, dtype=float)), dtype=float).reshape(-1)
    if v.shape[0] != len(xs):
        raise ValueError(f"the batch objective returned {v.shape[0]} value(s) "
                         f"for {len(xs)} vector(s)")
    return np.where(np.isfinite(v), v, failure_value)


def differential_evolution(batch_fn, state, *, maxiter=300,
                           mutation=(0.5, 1.0), recombination=0.7,
                           stall_gens=25, stall_tol=1e-2, xtol=1e-3,
                           failure_value=DEFAULT_FAILURE_VALUE, init_chunk=None,
                           on_generation=None, on_init_chunk=None,
                           should_stop=None):
    """Run from *state*, which is updated in place.

    *batch_fn(X)* takes an ``(k, n)`` array and returns *k* objective values.
    Anything non-finite is scored *failure_value*.

    Stops when the best value has improved by no more than *stall_tol* over the
    last *stall_gens* generations, or the population has collapsed to within
    *xtol* in every parameter (status 0); when *maxiter* generations have run
    (status 1); or when *should_stop()* is true, checked between generations and
    between initial chunks (status 3, resumable). A stop mid-generation is not
    offered: the generation in flight is redone from the last saved state.

    *on_generation(state, info)* is called after each generation with a dict of
    ``gen, best, median_feasible, n_failed, n_improved, nfev, spread``, and is
    where the caller checkpoints. *on_init_chunk(state)* does the same while the
    initial population is still being evaluated. Returns an ``OptimizeResult``.
    """
    pop, energies = state["pop"], state["energies"]
    bounds = np.asarray(state["bounds"], dtype=float)
    m, n = pop.shape
    rng = np.random.default_rng()
    rng.bit_generator.state = state["rng"]

    def save_rng():
        state["rng"] = rng.bit_generator.state

    def finish(status):
        x, f = best_of(state)
        return OptimizeResult(
            x=x if x is not None else np.array(pop[0], dtype=float), fun=f,
            nit=state["gen"], nfev=state["nfev"], status=status,
            success=(status == STATUS_CONVERGED), message=_MESSAGES[status],
            population=pop, population_energies=energies)

    # -- evaluate the initial population, in chunks that can be checkpointed ----
    if state["phase"] == "init":
        step = int(init_chunk) if init_chunk else m
        while True:
            pending = np.flatnonzero(~np.isfinite(energies))
            if pending.size == 0:
                break
            if should_stop is not None and should_stop():
                return finish(STATUS_STOPPED)
            take = pending[:step]
            energies[take] = _values(batch_fn, pop[take], failure_value)
            state["nfev"] += take.size
            if on_init_chunk is not None:
                on_init_chunk(state)
        state["phase"] = "evolve"
        _note_best(state)
        if on_generation is not None:
            on_generation(state, _info(state, 0, 0, 0.0, failure_value))

    # -- generations -----------------------------------------------------------
    while state["gen"] < maxiter:
        if _converged(state, stall_gens, stall_tol, xtol):
            return finish(STATUS_CONVERGED)
        if should_stop is not None and should_stop():
            return finish(STATUS_STOPPED)

        lo_m, hi_m = mutation
        scale = float(rng.uniform(lo_m, hi_m))          # dither, per generation
        trial = _trials(rng, pop, energies, bounds, scale, recombination)
        values = _values(batch_fn, trial, failure_value)
        state["nfev"] += m

        better = values <= energies                     # ties replace: plateaus
        n_improved = int(np.sum(values < energies))
        pop[better] = trial[better]
        energies[better] = values[better]
        n_failed = int(np.sum(values >= failure_value))

        state["gen"] += 1
        _note_best(state)
        save_rng()
        if on_generation is not None:
            on_generation(state, _info(state, n_failed, n_improved, scale,
                                       failure_value))

    if _converged(state, stall_gens, stall_tol, xtol):
        return finish(STATUS_CONVERGED)
    return finish(STATUS_MAXITER)


def _note_best(state):
    _x, f = best_of(state)
    state["best_history"].append(float(f))


def _ranges(state):
    """The population's extent in each parameter."""
    pop = np.asarray(state["pop"], dtype=float)
    return pop.max(axis=0) - pop.min(axis=0)


def _spread(state):
    return float(np.max(_ranges(state)))


def _converged(state, stall_gens, stall_tol, xtol):
    hist = state["best_history"]
    # Population collapse needs no history: the search has nowhere left to go.
    if state["phase"] == "evolve" and state["gen"] > 0 and _spread(state) <= xtol:
        return True
    if stall_gens and len(hist) > stall_gens:
        return (hist[-stall_gens - 1] - hist[-1]) <= stall_tol
    return False


def _info(state, n_failed, n_improved, scale, failure_value):
    e = np.asarray(state["energies"], dtype=float)
    ok = e[e < failure_value]
    return {"gen": state["gen"], "best": best_of(state)[1],
            "median_feasible": float(np.median(ok)) if ok.size else float("nan"),
            "n_feasible": int(ok.size), "n_failed": int(n_failed),
            "n_improved": int(n_improved), "nfev": state["nfev"],
            # Max over parameters is what collapse is tested against, but one
            # unidentifiable parameter keeps it at the full box width for ever, so
            # the median says far more about whether the search is contracting.
            "spread": _spread(state),
            "spread_median": float(np.median(_ranges(state))),
            "scale": float(scale)}

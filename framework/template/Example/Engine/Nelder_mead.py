"""Nelder-Mead whose whole state is a small JSON-able dict.

Why this exists. On a preemptible partition a job is killed without notice, so
an optimizer has to be resumable from what is on disk. For Nelder-Mead the
optimizer's *entire* state is the simplex and its function values plus two
counters -- but ``scipy.optimize.minimize`` keeps them in local variables and
hands the simplex back only when it returns. The earlier answer was to run scipy
in short slices and feed each slice's ``final_simplex`` into the next, which
worked but needed a great deal of machinery around it: slice sizing, an opening
probe evaluation, a floor below which a slice could not make progress, an
evaluation cache to hide the cost of re-evaluating the simplex at every slice
boundary, and a wall-clock deadline that had to be predicted well enough to
finish a slice before it arrived. Every one of those was a place for a bug, and
the boundary itself cost n+1 evaluations each time it was crossed.

This is scipy's ``_minimize_neldermead`` -- same reflection, expansion,
contraction and shrink, same tolerances, same tie-breaking sort, same
termination order -- restructured only so the loop can be stopped and resumed at
any *consistent point*: after each initial vertex, after each shrink vertex and
after each completed iteration. The search path is unchanged, evaluation for
evaluation, so nothing computed under scipy is invalidated by using this
instead; ``tests/test_nelder_mead.py`` checks that call by call against scipy.

What that buys:

* A checkpoint is ``state_to_json(state)``. Written every few evaluations, a hard
  kill costs the evaluations since the last write -- at most one iteration's
  worth, one or two evaluations -- instead of a slice.
* A resume re-evaluates nothing. Resuming from a saved state reaches
  bit-for-bit the same result as an uninterrupted run.
* Nothing needs to know when the job will end. "Stop now" is a callable that is
  checked between evaluations; being wrong about it costs at most the evaluation
  in flight.

The state, and the consistent points
------------------------------------
``sim`` (n+1 x n) and ``fsim`` (n+1) are the simplex and its values. ``phase``
says where in the algorithm the state sits:

``"init"``
    building the first simplex; vertices ``0..k-1`` have been evaluated and
    ``fsim[k:]`` is still ``inf``.
``"iter"``
    at the top of an iteration, ``sim`` sorted by ``fsim``.
``"shrink"``
    partway through a shrink; vertices ``1..k-1`` have been contracted toward
    ``sim[0]`` and re-evaluated, ``k..n`` have not been touched.

A stop in the middle of an iteration -- between the reflection and the expansion
it might lead to -- is not a consistent point and is not offered: the state on
disk is the one from the start of that iteration, and the iteration is simply
redone. That costs a reflection at most.
"""

import warnings

import numpy as np
from scipy.optimize import OptimizeResult

STATE_FORMAT = "nm-v1"

_MESSAGES = {
    0: "Optimization terminated successfully.",
    1: "Maximum number of function evaluations has been exceeded.",
    2: "Maximum number of iterations has been exceeded.",
    3: "Stopped on request; resumable.",
}

# ``status`` on the result. 0-2 are scipy's own; 3 is ours and means the loop
# was asked to stop, which is a normal, resumable outcome and not a failure.
STATUS_STOPPED = 3

# The options this implements. Anything else in ``options`` is refused rather
# than silently ignored, so a caller relying on one that is not here finds out.
SUPPORTED_OPTIONS = frozenset({
    "maxiter", "maxfev", "xatol", "fatol", "adaptive", "initial_simplex",
    "disp", "return_all",
})


class _MaxFev(Exception):
    """Raised inside an iteration when the evaluation budget runs out."""


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

def new_state(x0, lb=None, ub=None, initial_simplex=None, nfev=0, nit=1):
    """A fresh state: an unevaluated simplex built as scipy builds it.

    *nfev* and *nit* seed the counters, which is how a point resumed from a
    record that carries only its spend (and a simplex) shares one allowance
    across launches: the caps are compared with these running totals.
    """
    x0 = np.atleast_1d(np.asarray(x0, dtype=float)).flatten()
    if lb is not None:
        x0 = np.clip(x0, lb, ub)
    n = len(x0)
    if initial_simplex is None:
        sim = np.empty((n + 1, n))
        sim[0] = x0
        for k in range(n):
            y = np.array(x0, copy=True)
            y[k] = 1.05 * y[k] if y[k] != 0 else 0.00025
            sim[k + 1] = y
    else:
        sim = np.atleast_2d(np.array(initial_simplex, dtype=float))
        if sim.ndim != 2 or sim.shape[0] != sim.shape[1] + 1:
            raise ValueError("`initial_simplex` should be an array of shape (N+1,N)")
        if len(x0) != sim.shape[1]:
            raise ValueError("Size of `initial_simplex` is not consistent with `x0`")
    if lb is not None:
        # Reflect into the interior rather than clipping, so a simplex built
        # against an upper bound does not collapse onto it (scipy does this).
        sim = np.where(sim > ub, 2 * ub - sim, sim)
        sim = np.clip(sim, lb, ub)
    return {"sim": sim, "fsim": np.full(n + 1, np.inf), "nfev": int(nfev),
            "nit": int(nit), "phase": "init", "k": 0}


def state_to_json(state):
    """Plain lists and numbers, safe for ``json.dump``.

    Infinite values (vertices not yet evaluated) are written as ``null``, since
    ``Infinity`` is not JSON. Python's ``repr`` round-trips a float exactly, so
    a state written and read back is identical to the one that was saved.
    """
    fsim = [None if not np.isfinite(v) else float(v) for v in state["fsim"]]
    return {"format": STATE_FORMAT,
            "sim": np.asarray(state["sim"], dtype=float).tolist(),
            "fsim": fsim, "nfev": int(state["nfev"]), "nit": int(state["nit"]),
            "phase": state["phase"], "k": int(state["k"])}


def state_from_json(data, n=None):
    """The state in *data*, or None if it is not a usable one.

    Every failure is a miss: a corrupt or hand-edited file must cost the resume
    it was meant to provide, never the run. *n* is the number of parameters the
    caller expects; a state for a different problem is rejected.
    """
    try:
        if not isinstance(data, dict) or data.get("format") != STATE_FORMAT:
            return None
        sim = np.array(data["sim"], dtype=float)
        fsim = np.array([np.inf if v is None else v for v in data["fsim"]],
                        dtype=float)
        if sim.ndim != 2 or sim.shape[0] != sim.shape[1] + 1:
            return None
        if n is not None and sim.shape[1] != n:
            return None
        if fsim.shape != (sim.shape[0],) or not np.all(np.isfinite(sim)):
            return None
        if np.any(np.isnan(fsim)):
            return None
        phase, k = data["phase"], int(data["k"])
        if phase not in ("init", "iter", "shrink") or not 0 <= k <= sim.shape[0]:
            return None
        nfev, nit = int(data["nfev"]), int(data["nit"])
        if nfev < 0 or nit < 1:
            return None
    except (KeyError, TypeError, ValueError):
        return None
    return {"sim": sim, "fsim": fsim, "nfev": nfev, "nit": nit,
            "phase": phase, "k": k}


def best_of(state):
    """``(x, f)`` of the best vertex evaluated so far, or ``(None, inf)``.

    Not simply ``sim[0]``: while the first simplex is still being built the
    vertices are unsorted and some have not been evaluated at all.
    """
    fsim = np.asarray(state["fsim"], dtype=float)
    if not np.any(np.isfinite(fsim)):
        return None, float("inf")
    i = int(np.argmin(fsim))
    return np.array(state["sim"][i], dtype=float), float(fsim[i])


# ---------------------------------------------------------------------------
# The optimizer
# ---------------------------------------------------------------------------

def nelder_mead(func, state, *, lb=None, ub=None, adaptive=False, xatol=1e-4,
                fatol=1e-4, maxfev=None, maxiter=None, on_step=None,
                should_stop=None):
    """Run Nelder-Mead from *state*, which is updated in place.

    *func* takes one parameter vector. *on_step(state)* is called at every
    consistent point -- the caller decides whether to write it down, and
    should copy anything it keeps, because the state is mutated. *should_stop()*
    is checked before each evaluation that begins a unit of work; when it
    returns true the loop returns with ``status == 3`` and *state* exactly as a
    later call needs it.

    Returns an ``OptimizeResult`` shaped like scipy's, with ``final_simplex``.
    ``nfev`` and ``nit`` are running totals including whatever the state
    already carried, and *maxfev*/*maxiter* are compared against those totals.
    """
    N = np.asarray(state["sim"]).shape[1]
    if adaptive:
        dim = float(N)
        rho, chi, psi, sigma = 1, 1 + 2 / dim, 0.75 - 1 / (2 * dim), 1 - 1 / dim
    else:
        rho, chi, psi, sigma = 1, 2, 0.5, 0.5

    # scipy's defaulting of the two caps, so a caller giving only one gets what
    # it would have got from minimize().
    if maxiter is None and maxfev is None:
        maxiter = maxfev = N * 200
    elif maxiter is None:
        maxiter = N * 200 if maxfev == np.inf else np.inf
    elif maxfev is None:
        maxfev = N * 200 if maxiter == np.inf else np.inf

    bounded = lb is not None

    def f(x):
        if state["nfev"] >= maxfev:
            raise _MaxFev
        state["nfev"] += 1
        return func(np.copy(x))

    def clip(x):
        return np.clip(x, lb, ub) if bounded else x

    def checkpoint():
        if on_step is not None:
            on_step(state)

    def halted():
        return should_stop is not None and should_stop()

    def sort():
        ind = np.argsort(state["fsim"])
        state["sim"] = np.take(state["sim"], ind, 0)
        state["fsim"] = np.take(state["fsim"], ind, 0)

    def finish(status):
        sim, fsim = state["sim"], state["fsim"]
        if status == STATUS_STOPPED:
            # Possibly mid-build, with the vertices unsorted: report the best
            # one actually evaluated, not whatever sits in row 0.
            x_best, f_best = best_of(state)
            x = x_best if x_best is not None else np.array(sim[0], dtype=float)
            fun = f_best
        else:
            x, fun = sim[0], np.min(fsim)
        return OptimizeResult(fun=fun, nit=state["nit"], nfev=state["nfev"],
                              status=status, success=(status == 0),
                              message=_MESSAGES[status], x=x,
                              final_simplex=(sim, fsim))

    try:
        # -- building the first simplex, one vertex at a time ----------------
        if state["phase"] == "init":
            try:
                while state["k"] < N + 1:
                    if halted():
                        return finish(STATUS_STOPPED)
                    k = state["k"]
                    state["fsim"][k] = f(state["sim"][k])
                    state["k"] = k + 1
                    checkpoint()
            except _MaxFev:
                pass
            sort()
            state["phase"] = "iter"
            checkpoint()

        # -- iterations ------------------------------------------------------
        while state["nfev"] < maxfev and state["nit"] < maxiter:
            if halted():
                return finish(STATUS_STOPPED)
            sim, fsim = state["sim"], state["fsim"]

            if state["phase"] == "iter":
                if (np.max(np.ravel(np.abs(sim[1:] - sim[0]))) <= xatol and
                        np.max(np.abs(fsim[0] - fsim[1:])) <= fatol):
                    break
                try:
                    xbar = np.add.reduce(sim[:-1], 0) / N
                    xr = clip((1 + rho) * xbar - rho * sim[-1])
                    fxr = f(xr)
                    doshrink = 0

                    if fxr < fsim[0]:
                        xe = clip((1 + rho * chi) * xbar - rho * chi * sim[-1])
                        fxe = f(xe)
                        if fxe < fxr:
                            sim[-1], fsim[-1] = xe, fxe
                        else:
                            sim[-1], fsim[-1] = xr, fxr
                    else:  # fsim[0] <= fxr
                        if fxr < fsim[-2]:
                            sim[-1], fsim[-1] = xr, fxr
                        else:  # fxr >= fsim[-2]
                            if fxr < fsim[-1]:
                                # Outside contraction.
                                xc = clip((1 + psi * rho) * xbar
                                          - psi * rho * sim[-1])
                                fxc = f(xc)
                                if fxc <= fxr:
                                    sim[-1], fsim[-1] = xc, fxc
                                else:
                                    doshrink = 1
                            else:
                                # Inside contraction.
                                xcc = clip((1 - psi) * xbar + psi * sim[-1])
                                fxcc = f(xcc)
                                if fxcc < fsim[-1]:
                                    sim[-1], fsim[-1] = xcc, fxcc
                                else:
                                    doshrink = 1
                            if doshrink:
                                state["phase"], state["k"] = "shrink", 1
                                checkpoint()

                    if not doshrink:
                        state["nit"] += 1
                except _MaxFev:
                    pass

            if state["phase"] == "shrink":
                try:
                    while state["k"] <= N:
                        if halted():
                            return finish(STATUS_STOPPED)
                        j = state["k"]
                        sim[j] = clip(sim[0] + sigma * (sim[j] - sim[0]))
                        fsim[j] = f(sim[j])
                        state["k"] = j + 1
                        checkpoint()
                    state["phase"] = "iter"
                    state["nit"] += 1
                except _MaxFev:
                    pass

            if state["phase"] == "iter":
                sort()
                checkpoint()
            elif state["phase"] == "shrink":
                # Only reachable through _MaxFev partway through a shrink, where
                # scipy also sorts what it has and stops.
                sort()
                state["phase"] = "iter"
    except _MaxFev:
        pass

    if state["nfev"] >= maxfev:
        return finish(1)
    if state["nit"] >= maxiter:
        return finish(2)
    return finish(0)


# ---------------------------------------------------------------------------
# A drop-in for scipy.optimize.minimize(method="Nelder-Mead")
# ---------------------------------------------------------------------------

def bounds_arrays(bounds, n):
    """``(lb, ub)`` arrays from any bounds form ``minimize`` accepts, or None.

    ``None`` entries in a list of pairs mean unbounded on that side. Returns
    ``(None, None)`` when nothing is bounded at all, so the loop skips clipping
    rather than clipping to infinities.
    """
    if bounds is None:
        return None, None
    if hasattr(bounds, "lb") and hasattr(bounds, "ub"):
        lb = np.broadcast_to(np.asarray(bounds.lb, dtype=float), (n,)).copy()
        ub = np.broadcast_to(np.asarray(bounds.ub, dtype=float), (n,)).copy()
    else:
        pairs = list(bounds)
        if len(pairs) != n:
            raise ValueError("bounds must have one (min, max) pair per parameter")
        lb = np.array([-np.inf if p is None or p[0] is None else p[0]
                       for p in pairs], dtype=float)
        ub = np.array([np.inf if p is None or p[1] is None else p[1]
                       for p in pairs], dtype=float)
    if np.any(lb > ub):
        raise ValueError("Nelder Mead - one of the lower bounds is greater "
                         "than an upper bound.")
    if np.all(np.isinf(lb)) and np.all(np.isinf(ub)):
        return None, None
    return lb, ub


def can_run(options=None, extra=None):
    """Whether ``minimize_nelder_mead`` implements everything asked of it.

    *options* is the scipy ``options`` dict and *extra* any other keyword
    arguments that would go to ``minimize`` (``tol`` is the only one supported).
    A caller falls back to scipy when this is false, so a spec using an option
    this does not implement keeps its old behaviour instead of silently losing
    it.
    """
    if set(options or ()) - SUPPORTED_OPTIONS:
        return False
    if set(extra or ()) - {"tol"}:
        return False
    return True


def minimize_nelder_mead(fun, x0, args=(), bounds=None, options=None, tol=None,
                         state=None, on_step=None, should_stop=None, nfev=0,
                         nit=1):
    """``scipy.optimize.minimize(method="Nelder-Mead")``, but resumable.

    Takes the same ``bounds``, ``options`` and ``tol`` and applies them the same
    way, including ``tol`` seeding ``xatol`` and ``fatol``. Returns
    ``(result, state)``. Pass the returned *state* -- or one loaded with
    :func:`state_from_json` -- back in to continue. When *state* is given,
    *x0*, ``initial_simplex``, *nfev* and *nit* are ignored: the state already
    says where the search is.
    """
    options = dict(options or {})
    unknown = set(options) - SUPPORTED_OPTIONS
    if unknown:
        warnings.warn(f"Unknown solver options: {', '.join(sorted(unknown))}",
                      RuntimeWarning, stacklevel=2)
    if tol is not None:
        options.setdefault("xatol", tol)
        options.setdefault("fatol", tol)

    if args and not isinstance(args, tuple):
        args = (args,)
    x0 = np.atleast_1d(np.asarray(x0, dtype=float)).flatten()
    n = len(x0) if state is None else np.asarray(state["sim"]).shape[1]
    lb, ub = bounds_arrays(bounds, n)
    if state is None:
        state = new_state(x0, lb, ub, options.get("initial_simplex"),
                          nfev=nfev, nit=nit)

    res = nelder_mead(
        (lambda x: fun(x, *args)) if args else fun, state, lb=lb, ub=ub,
        adaptive=bool(options.get("adaptive", False)),
        xatol=options.get("xatol", 1e-4), fatol=options.get("fatol", 1e-4),
        maxfev=options.get("maxfev"), maxiter=options.get("maxiter"),
        on_step=on_step, should_stop=should_stop)
    return res, state

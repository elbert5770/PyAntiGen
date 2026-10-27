"""The standard fit as bounded Levenberg-Marquardt on the AMICI residuals (``method="LM"``).

With fixed sigmas (``PYANTIGEN_SIGMA_FLOOR=fixed``) the loss is ``const + 0.5 r.r``, and the AMICI provider returns the
residuals and their Jacobian in one pass, so the fit uses the stepping core the profile stage already runs
(:func:`pyantigen.engine.Profile_lm.minimize`): a Marquardt step on ``J^T J`` with the bounds handled by an active set,
each ``|s_i|`` capped, and a failed integration treated as a rejected step that shrinks the trust region. L-BFGS-B has
no such guard; its first trial point can land on a corner of the box, where the integration may never come back.

The verdict is the projected gradient of the accepted point, not an optimizer flag. A fit that stops at its evaluation
budget or its deadline returns the best finite point with ``success=False``, and a relaunch continues from the partial
record in the fit cache.

Options (``standard["options"]``): ``maxfev`` (evaluations including rejected steps, default 120), ``pg_tol`` (1e-2),
``step0`` / ``step_max`` (initial and largest ``|s_i|``, in the optimizer's units: decades for a log10 parameter),
``lam0``.
"""
import math
import time

import numpy as np

LM_METHODS = frozenset({"lm", "levenberg-marquardt"})

DEFAULT_MAX_EVALS = 120


def is_lm(method):
    return str(method).lower() in LM_METHODS


def _limits(bounds, n):
    lb, ub = np.full(n, -np.inf), np.full(n, np.inf)
    for i, b in enumerate(bounds or []):
        if b is None:
            continue
        lo, hi = b
        if lo is not None:
            lb[i] = lo
        if hi is not None:
            ub[i] = hi
    return lb, ub


def settings_from(options):
    from pyantigen.engine.Profile_lm import LmSettings
    o = options or {}
    kw = {"max_evals": int(o.get("maxfev") or o.get("maxiter") or DEFAULT_MAX_EVALS)}
    for key in ("pg_tol", "step0", "step_max", "lam0"):
        if o.get(key) is not None:
            kw[key] = float(o[key])
    return LmSettings(**kw)


def lm_fit(provider, x_start, bounds, scales, options=None, deadline=None, on_best=None, log=print):
    """Fit from *x_start* (optimizer space). Returns a scipy-style ``OptimizeResult`` with ``x``, ``fun``, ``success``,
    ``message``, ``nit``, ``nfev`` and the LM ``verdict`` and projected gradient ``pg``.

    *on_best(x, f, n_evals)* is called whenever an evaluation improves the best finite value (for the fit cache).
    """
    from scipy.optimize import OptimizeResult
    from pyantigen.engine.Profile_lm import OutOfTime, amici_evaluate, minimize

    if not hasattr(provider, "linear_residual_jacobian"):
        raise RuntimeError("method LM needs the AMICI residual Jacobian: run with --backend amici and "
                           "PYANTIGEN_SIGMA_FLOOR=fixed")
    x_start = np.asarray(x_start, float)
    lb, ub = _limits(bounds, len(x_start))
    st = settings_from(options)
    inner = amici_evaluate(provider, scales)
    best = {"f": math.inf, "x": x_start.copy(), "n": 0}

    def evaluate(x):
        best["n"] += 1
        out = inner(x)
        if out is not None and out[0] < best["f"]:
            best["f"], best["x"] = float(out[0]), np.array(x, float)
            if on_best is not None:
                on_best(best["x"], best["f"], best["n"])
        return out

    t0 = time.time()
    try:
        res = minimize(evaluate, np.clip(x_start, lb, ub), lb, ub, np.arange(len(x_start)), st,
                       deadline=deadline, log=log)
    except OutOfTime:
        log(f"[lm fit] stopped at the deadline after {best['n']} evaluation(s); best f {best['f']:.6f}")
        return OptimizeResult(x=best["x"], fun=best["f"], success=False, message="deadline reached", nit=best["n"],
                              nfev=best["n"], verdict="not_converged", pg=None, seconds=time.time() - t0)
    ok = res.verdict == "converged"
    log(f"[lm fit] {res.verdict}: f {res.f:.6f}, {res.reason}; {res.evals} evaluation(s), "
        f"{time.time() - t0:.0f} s")
    return OptimizeResult(x=res.x, fun=res.f, success=ok, message=f"{res.verdict}: {res.reason}", nit=res.evals,
                          nfev=res.evals, verdict=res.verdict, pg=res.pg, seconds=time.time() - t0)

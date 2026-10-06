"""Multi-start as a triage pipeline.

A multi-start fit is only as good as the starts it is given. Fitting from every
sampled point spends most of the budget polishing points that were never going to
win, and reports a spread that says little. The stage here narrows the field
first, cheaply, and fits only from what survives:

  1. Fast pass.   Score every Sobol candidate once, at RELAXED integrator
                  tolerances. The aim is not accuracy but telling mountains
                  (terrible fits) from valleys (promising ones).
  2. Feasibility. Drop every candidate the solver fails or stalls on -- stiff
                  models have parameter combinations that need impossibly small
                  steps, and handing one to a local optimizer wastes it or
                  crashes it.
  3. Threshold.   Rank the survivors and keep only the best ``keep_fraction``.
  4. Niching.     Cluster what is left by distance in log10 space, so that many
                  points in one valley count once.
  5. Reference set. The lowest-scoring point of each cluster becomes a start.

This module is the pure part -- arrays in, arrays out, no model -- so every step
can be tested on its own. The orchestration, which owns the objective and the
solver settings, is in ``pyantigen.engine.Optimize``.

Distances are Euclidean in log10 of the LINEAR parameter value, divided by
sqrt(k): ``cluster_radius`` is therefore an RMS distance per parameter in
decades, which means the same thing for 3 parameters as for 30. A parameter that
is not strictly positive in every point (a log-odds, an exact zero) has no
decades; its raw value is used.
"""
import math

import numpy as np

TRIAGE_DEFAULTS = {
    "n_candidates": 64,       # Sobol points scored, x0 not counted; a power of 2
    "keep_fraction": 0.2,     # keep the best 20% of the feasible ones
    "cluster_radius": 0.15,   # RMS decades per parameter
    "tolerance_factor": 100.0,  # relaxation of atol and rtol in the fast pass
    # Where the candidates are scored: "auto" decides from timings (see
    # prefer_pool), True always uses the worker pool, False never does.
    "parallel": "auto",
}


def resolve_triage(optimizer_kwargs):
    """The triage settings: defaults overlaid by ``multistart_triage``.

    An unknown key is an error rather than a silent no-op -- a typo there would
    look exactly like a setting that did not help.
    """
    given = dict((optimizer_kwargs or {}).get("multistart_triage") or {})
    unknown = sorted(set(given) - set(TRIAGE_DEFAULTS))
    if unknown:
        raise ValueError(f"unknown multistart_triage key(s) {unknown}; expected a "
                         f"subset of {sorted(TRIAGE_DEFAULTS)}")
    out = {**TRIAGE_DEFAULTS, **given}
    if int(out["n_candidates"]) < 1:
        raise ValueError("multistart_triage['n_candidates'] must be >= 1")
    if not 0.0 < float(out["keep_fraction"]) <= 1.0:
        raise ValueError("multistart_triage['keep_fraction'] must be in (0, 1]")
    if not float(out["cluster_radius"]) >= 0.0:
        raise ValueError("multistart_triage['cluster_radius'] must be >= 0")
    if not float(out["tolerance_factor"]) >= 1.0:
        raise ValueError("multistart_triage['tolerance_factor'] must be >= 1")
    out["n_candidates"] = int(out["n_candidates"])
    par = out["parallel"]
    if isinstance(par, str):
        if par.lower() != "auto":
            raise ValueError("multistart_triage['parallel'] must be 'auto', True or False")
        out["parallel"] = "auto"
    else:
        out["parallel"] = bool(par)
    return out


# -- the gradient of the local fits ------------------------------------------------------
#
# The stage's local fits use a gradient method (L-BFGS-B). Left alone, scipy takes
# the gradient by forward differences with a step of 1e-8: one evaluation per
# parameter, one after another, so an 18-parameter fit pays 19 evaluations -- 7
# minutes at 22 s each -- per iteration, with the worker pool idle. And 1e-8 is
# far below the noise of this objective (its integrator tolerances are 1e-8
# relative, so values wobble at roughly 1e-6 to 1e-3): dividing that noise by
# 1e-8 gives a gradient that is mostly noise, and the line search fails on it
# ("ABNORMAL" termination).
#
# FDObjective evaluates x and every perturbed point TOGETHER, as one batch. On the
# pool that is one round of evaluations per iteration (~22 s) instead of 19, and
# with a step the noise cannot swamp (1e-3) the gradient is usable. The same code
# runs serially, one point after another, when there is no pool: it is the step,
# not the parallelism, that makes the gradient trustworthy.
GRADIENT_DEFAULTS = {
    # Where each gradient's points are evaluated: "auto" decides from timings (see
    # prefer_pool_gradient), True always uses the worker pool, False never does.
    "parallel": "auto",
    # Finite-difference step in optimizer units: decades for a log10 parameter,
    # and relative (scaled by max(|x|, 1)) for a linear one.
    "step": 1e-3,
    # "forward", "central" (twice the evaluations, second-order accurate) or
    # "scipy" (scipy's own serial forward differences at 1e-8: the old behaviour).
    "scheme": "forward",
}

# Methods that take a gradient. Nelder-Mead and Powell do not: none of this applies.
GRADIENT_METHODS = frozenset({"l-bfgs-b", "bfgs", "cg", "tnc", "slsqp"})


def resolve_gradient(optimizer_kwargs):
    """The local fits' gradient settings: defaults overlaid by ``multistart_gradient``.
    An unknown key is an error rather than a silent no-op."""
    given = dict((optimizer_kwargs or {}).get("multistart_gradient") or {})
    unknown = sorted(set(given) - set(GRADIENT_DEFAULTS))
    if unknown:
        raise ValueError(f"unknown multistart_gradient key(s) {unknown}; expected a "
                         f"subset of {sorted(GRADIENT_DEFAULTS)}")
    out = {**GRADIENT_DEFAULTS, **given}
    if not float(out["step"]) > 0.0:
        raise ValueError("multistart_gradient['step'] must be positive")
    out["step"] = float(out["step"])
    scheme = str(out["scheme"]).lower()
    if scheme not in ("forward", "central", "scipy"):
        raise ValueError("multistart_gradient['scheme'] must be 'forward', 'central' or 'scipy'")
    out["scheme"] = scheme
    par = out["parallel"]
    if isinstance(par, str):
        if par.lower() != "auto":
            raise ValueError("multistart_gradient['parallel'] must be 'auto', True or False")
        out["parallel"] = "auto"
    else:
        out["parallel"] = bool(par)
    return out


class FDObjective:
    """``x -> (f, gradient)`` for scipy's ``jac=True``, with the points of one
    gradient evaluated as ONE batch.

    ``batch_eval(points) -> values`` is whatever evaluates a list of opt-space
    points: the worker pool, or a loop in this process. Values that are not finite
    or reach *failure_value* are failures.

    A failed PERTURBED point is never scored as a cliff: it is retried on the
    opposite side of x (one more batch, rare), and if that fails too the
    component is 0. A step that would leave the bounds is taken the other way.
    If f(x) itself fails the call returns *penalty* (or the failure value) and a
    zero gradient -- the plateau semantics of the failure wall -- without using
    the perturbed points.
    """

    def __init__(self, batch_eval, bounds, scales, *, step=1e-3, scheme="forward",
                 failure_value=1e10, penalty=None):
        if scheme not in ("forward", "central"):
            raise ValueError(f"FDObjective scheme must be 'forward' or 'central', got {scheme!r}")
        self.batch_eval = batch_eval
        self.scales = list(scales)
        self.k = len(self.scales)
        bounds = list(bounds) if bounds else [(None, None)] * self.k
        self.lb = np.array([-np.inf if b[0] is None else float(b[0]) for b in bounds])
        self.ub = np.array([np.inf if b[1] is None else float(b[1]) for b in bounds])
        self.step, self.scheme = float(step), scheme
        self.failure_value, self.penalty = failure_value, penalty
        self.last_f = float("nan")  # f at the most recent x whose value evaluated
        self.n_calls = 0          # gradients taken
        self.n_evals = 0          # objective evaluations, all of them
        self.n_retried = 0        # perturbed points retried on the other side
        self.n_zeroed = 0         # components set to 0 because no side evaluated

    def points_per_call(self):
        """Evaluations in one gradient's first batch."""
        return 1 + self.k * (2 if self.scheme == "central" else 1)

    def _ok(self, v):
        return bool(np.isfinite(v)) and v < self.failure_value

    def _h(self, x):
        return np.array([self.step if sc == "log10" else self.step * max(abs(x[i]), 1.0)
                         for i, sc in enumerate(self.scales)])

    def __call__(self, x):
        x = np.asarray(x, dtype=float)
        h = self._h(x)
        pts, plan = [x.copy()], []              # plan[i]: {sign: index into pts}
        for i in range(self.k):
            signs = (1, -1) if self.scheme == "central" else \
                ((1,) if x[i] + h[i] <= self.ub[i] else (-1,))
            slot = {}
            for s in signs:
                xi = x[i] + s * h[i]
                if self.lb[i] <= xi <= self.ub[i]:
                    p = x.copy()
                    p[i] = xi
                    pts.append(p)
                    slot[s] = len(pts) - 1
            plan.append(slot)
        vals = [float(v) for v in self.batch_eval(pts)]
        self.n_evals += len(pts)
        self.n_calls += 1
        f0 = vals[0]
        if not self._ok(f0):
            fail = self.penalty if self.penalty is not None else (
                f0 if np.isfinite(f0) else self.failure_value)
            return float(fail), np.zeros(self.k)
        self.last_f = f0

        # One-sided scheme: a perturbed point that failed is retried on the other side.
        retry = []
        for i, slot in enumerate(plan):
            if self.scheme == "forward" and slot and not self._ok(vals[next(iter(slot.values()))]):
                s_try = -next(iter(slot))
                xi = x[i] + s_try * h[i]
                if self.lb[i] <= xi <= self.ub[i]:
                    p = x.copy()
                    p[i] = xi
                    retry.append((i, s_try, p))
        if retry:
            extra = [float(v) for v in self.batch_eval([p for _, _, p in retry])]
            self.n_evals += len(retry)
            self.n_retried += len(retry)
            for (i, s_try, _), v in zip(retry, extra):
                vals.append(v)
                plan[i][s_try] = len(vals) - 1

        g = np.zeros(self.k)
        for i, slot in enumerate(plan):
            good = {s: vals[j] for s, j in slot.items() if self._ok(vals[j])}
            if self.scheme == "central" and len(good) == 2:
                g[i] = (good[1] - good[-1]) / (2.0 * h[i])
            elif good:
                s = next(iter(good))                      # one-sided from whichever side worked
                g[i] = s * (good[s] - f0) / h[i]
            else:
                self.n_zeroed += 1
        return f0, g


def json_safe(obj):
    """*obj* as plain JSON data: numpy scalars and arrays become Python ones, a
    non-finite float becomes None, tuples become lists. The result snapshot is
    written with allow_nan=False, so a single stray inf would lose the file."""
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return json_safe(obj.tolist())
    if isinstance(obj, np.generic):
        return json_safe(obj.item())
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    return obj


# What it costs to bring a worker pool up, in seconds, at the least. Every worker
# is spawned, re-imports the engine and compiles every model; measured at about
# twice the time the first evaluation takes in the parent, which pays the same
# compile once. The floor covers a parent that was already warm.
POOL_STARTUP_FLOOR_S = 30.0


def prefer_pool(n_remaining, t_first, t_eval, n_workers, margin=1.5):
    """Whether scoring *n_remaining* candidates on a pool beats doing it here.

    t_first   seconds the FIRST evaluation took in this process. It includes the
              cold start, so it stands in for what each worker pays before its
              first result.
    t_eval    seconds a WARM evaluation takes here (a later one).
    n_workers workers the pool would have.
    margin    the pool must win by this factor. The estimate is rough, a pool
              that loses wastes a full start-up, and a serial pass has no
              failure modes of its own, so a near tie goes to serial.

    Measured on the PK Aducanumab spec, 8 workers: 32 candidates cost 161 s
    pooled against about 190 s serial, and 8 cost 132 s against 49 s. The pool
    pays for evaluations that are expensive and numerous, not for many cheap
    ones.

    Returns ``(use_pool, serial_estimate_s, pool_estimate_s)``.
    """
    n_remaining, n_workers = int(n_remaining), int(n_workers)
    if n_remaining <= 0 or n_workers <= 1:
        return False, 0.0, 0.0
    serial = n_remaining * float(t_eval)
    startup = max(2.0 * float(t_first), POOL_STARTUP_FLOOR_S)
    pool = startup + math.ceil(n_remaining / n_workers) * float(t_eval)
    return serial > float(margin) * pool, serial, pool


def prefer_pool_gradient(n_iterations, n_points, t_first, t_eval, n_workers, *,
                         pool_ready=False, margin=1.5):
    """Whether the local fits' gradients are worth the worker pool.

    Not the same question as :func:`prefer_pool`. A gradient's points are a batch
    only ``n_points`` wide (k + 1 for forward differences) and the iterations that
    follow are sequential, so the pool can win at most a factor of
    ``n_points / ceil(n_points / n_workers)`` per iteration -- nothing at all for a
    1-parameter fit, ~19x for 18 parameters on 24 workers.

        serial   n_iterations * n_points * t_eval
        pool     start-up + n_iterations * ceil(n_points / n_workers) * t_eval

    *pool_ready*: the pool is already up (the fast pass built it), so there is no
    start-up to pay. *n_iterations* is an upper bound (the optimizer's iteration
    cap, summed over the fits still to run), which flatters the pool a little; the
    margin covers that.

    Returns ``(use_pool, serial_estimate_s, pool_estimate_s)``.
    """
    n_iterations, n_points, n_workers = int(n_iterations), int(n_points), int(n_workers)
    if n_iterations <= 0 or n_workers <= 1 or n_points <= 1:
        return False, 0.0, 0.0
    serial = n_iterations * n_points * float(t_eval)
    startup = 0.0 if pool_ready else max(2.0 * float(t_first), POOL_STARTUP_FLOOR_S)
    pool = startup + n_iterations * math.ceil(n_points / n_workers) * float(t_eval)
    return serial > float(margin) * pool, serial, pool


def log10_coordinates(points_lin):
    """(n, k) array of the coordinates distances are measured in."""
    p = np.atleast_2d(np.asarray(points_lin, dtype=float))
    out = p.copy()
    for j in range(p.shape[1]):
        col = p[:, j]
        if np.all(np.isfinite(col)) and np.all(col > 0):
            out[:, j] = np.log10(col)
    return out


def _cutoff(cluster_radius, k):
    return float(cluster_radius) * math.sqrt(max(int(k), 1))


def cluster_labels(coords, cluster_radius):
    """Complete-linkage clusters of *coords*: no two points in a cluster are
    further apart than the cutoff, so a cluster cannot chain across a valley.
    Returns an int label per row, 0-based."""
    coords = np.atleast_2d(np.asarray(coords, dtype=float))
    n = len(coords)
    if n == 0:
        return np.zeros(0, dtype=int)
    if n == 1:
        return np.zeros(1, dtype=int)
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import pdist
    Z = linkage(pdist(coords), method="complete")
    labels = fcluster(Z, t=_cutoff(cluster_radius, coords.shape[1]), criterion="distance")
    return np.asarray(labels, dtype=int) - 1


def triage(points_lin, scores, *, keep_fraction, cluster_radius, max_fits=None):
    """Steps 2-5 on scored candidates. Index 0 is the spec's x0.

    points_lin  (n, k) linear parameter values, row 0 = x0.
    scores      (n,) fast-pass NLL; non-finite means the solver failed there.
    max_fits    cap on the reference set, x0 included.

    x0 is never pruned or ranked away: it is the one start that guarantees the
    stage cannot come back worse than the single fit it precedes. It is always
    in the reference set, and a niche leader within the cluster cutoff of it is
    dropped as a duplicate rather than fitted twice.

    Returns a dict with the counts of each step and ``starts``, the row indices
    to fit from, x0 first and the rest best score first.
    """
    pts = np.atleast_2d(np.asarray(points_lin, dtype=float))
    sc = np.asarray(scores, dtype=float)
    n, k = pts.shape
    coords = log10_coordinates(pts)

    cand = np.arange(1, n)                         # x0 is handled separately
    feasible = cand[np.isfinite(sc[cand])]         # step 2
    pruned = len(cand) - len(feasible)

    order = feasible[np.argsort(sc[feasible], kind="stable")]
    n_keep = min(len(order), max(1, math.ceil(float(keep_fraction) * len(order)))) \
        if len(order) else 0
    kept = order[:n_keep]                          # step 3

    labels = cluster_labels(coords[kept], cluster_radius)   # step 4
    leaders = []
    for lab in np.unique(labels):                  # step 5
        members = kept[labels == lab]
        leaders.append(int(members[np.argmin(sc[members])]))
    leaders.sort(key=lambda i: sc[i])

    cutoff = _cutoff(cluster_radius, k)
    fresh = [i for i in leaders
             if np.linalg.norm(coords[i] - coords[0]) > cutoff]
    starts = [0] + fresh
    if max_fits is not None:
        starts = starts[:max(1, int(max_fits))]

    return {
        "n_candidates": int(len(cand)),
        "n_pruned": int(pruned),
        "n_feasible": int(len(feasible)),
        "n_kept": int(len(kept)),
        "n_clusters": int(len(leaders)),
        "kept": [int(i) for i in kept],
        # The niche each kept candidate fell in, aligned with "kept".
        "cluster_labels": [int(c) for c in labels],
        "leaders": leaders,
        "n_duplicates_of_x0": int(len(leaders) - len(fresh)),
        "starts": [int(i) for i in starts],
        "cutoff": cutoff,
    }


def candidate_table(points_lin, scores, rep, fit_nll=None):
    """One row per candidate, x0 included: everything a waterfall plot needs.

    A waterfall plot lays the candidates out best to worst and colours each by
    what became of it. The fields that make that possible:

      candidate   index into the Sobol draw; 0 is the spec's x0.
      x           linear parameter values, in the parameter order of the report.
      triage_nll  the fast-pass NLL, or None where the solver failed there.
      rank        1 = best triage_nll of every feasible candidate, x0 included;
                  None where it failed. Sorting by it gives the waterfall.
      relaxed     True where triage_nll was scored at relaxed tolerances. x0 is
                  scored at the configured ones, so its bar is not on exactly
                  the same scale as the others.
      feasible    False where the solver failed (the pruned candidates).
      kept        passed the ranked threshold (x0 is never ranked: False).
      cluster     the niche it fell in among the kept ones, else None.
      leader      lowest-scoring member of its niche, so a candidate for a fit.
      fitted      a local fit was started from it (x0 always is).
      fit_nll     where that fit ended, or None.

    *fit_nll* maps candidate index to the NLL its fit reached.
    """
    pts = np.atleast_2d(np.asarray(points_lin, dtype=float))
    sc = np.asarray(scores, dtype=float)
    fit_nll = fit_nll or {}
    feasible = [i for i in range(len(sc)) if np.isfinite(sc[i])]
    rank = {i: r for r, i in enumerate(sorted(feasible, key=lambda i: sc[i]), start=1)}
    kept = set(rep["kept"])
    leaders = set(rep["leaders"])
    cluster = dict(zip(rep["kept"], rep["cluster_labels"]))
    fitted = set(rep["starts"])
    rows = []
    for i in range(len(sc)):
        ok = bool(np.isfinite(sc[i]))
        rows.append({
            "candidate": i,
            "x": [float(v) for v in pts[i]],
            "triage_nll": float(sc[i]) if ok else None,
            "rank": rank.get(i),
            "relaxed": i != 0,
            "feasible": ok,
            "kept": i in kept,
            "cluster": cluster.get(i),
            "leader": i in leaders,
            "fitted": i in fitted,
            "fit_nll": fit_nll.get(i),
        })
    return rows


def basins(finals_lin, funs, cluster_radius, report_dnll=1.0):
    """Group the fits' END points into basins, and say what the grouping means.

    Two fits that end near each other in log10 space found the same basin,
    whatever small differences their NLL has: a loose, capped optimizer stops at
    slightly different values of the same valley, and that is not multimodality.
    What the report needs is whether fits ended in DIFFERENT places, and if so
    whether one place is actually better.

    Returns ``(basin_list, verdict, text)``. Each basin is
    ``{"nll", "members", "x"}`` (best member's NLL and linear x), best first.
    verdict is one of "unimodal", "degenerate" (separate places, NLL within
    *report_dnll*: the data do not pin these parameters apart) or "multimodal"
    (a worse basin is at least *report_dnll* above the best).
    """
    finals = np.atleast_2d(np.asarray(finals_lin, dtype=float))
    f = np.asarray(funs, dtype=float)
    ok = np.isfinite(f)
    idx = np.flatnonzero(ok)
    if len(idx) == 0:
        return [], "none", "no multi-start fit finished."
    labels = cluster_labels(log10_coordinates(finals[idx]), cluster_radius)
    out = []
    for lab in np.unique(labels):
        mem = idx[labels == lab]
        best = mem[np.argmin(f[mem])]
        out.append({"nll": float(f[best]), "members": [int(m) for m in mem],
                    "x": [float(v) for v in finals[best]]})
    out.sort(key=lambda b: b["nll"])

    if len(out) == 1:
        text = (f"all {len(idx)} fit(s) ended in one basin (best nll "
                f"{out[0]['nll']:.6g}); unimodal here.")
        return out, "unimodal", text
    gap = out[-1]["nll"] - out[0]["nll"]
    if gap < float(report_dnll):
        text = (f"{len(out)} separate end points within {gap:.3g} nll of each "
                f"other: a flat or degenerate valley, so the data do not pin "
                f"these parameters apart. Not evidence of a better optimum.")
        return out, "degenerate", text
    text = (f"{len(out)} basins; the best is {out[0]['nll']:.6g} and the worst "
            f"{out[-1]['nll']:.6g} ({gap:.3g} nll worse). A profile anchored on "
            f"one fit would describe whichever basin it happened to reach.")
    return out, "multimodal", text

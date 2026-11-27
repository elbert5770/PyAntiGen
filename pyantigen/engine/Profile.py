"""The profile-likelihood stepping protocol, independent of the solver that produces each point.

It replaces the tiered ``Profile_plan`` / ``Stop_rules`` protocol. Everything here talks to a
:class:`~Profile_record.NuisanceSolver` and reads and writes :class:`~Profile_record.PointRecord`, so the same protocol runs
on the LM/AMICI solver (:mod:`Profile_lm`) or on the RoadRunner Nelder-Mead worker (``RoadRunnerSolver``).

Passes, each resumable from its JSONL file and run in this order for one side (a parameter and a direction):

* :func:`walk` leaves the optimum in one direction. The step follows the profile's own curvature: from the last point
  (``dnll`` at distance ``d``) it assumes ``dnll = c d^2`` and takes the step that raises ``dnll`` by ``df_step``, capped by
  ``h_max`` and by three times the previous step. The nuisance parameters start from :func:`predict_start`: the secant
  of the last two usable points, and for the first hop (there is nothing to extrapolate along yet) the tangent of the
  profile at the optimum, :func:`tangent_matrix`, from the Gauss-Newton Hessian of the optimum (the AMICI profile
  computes it from the Jacobian it already evaluates there). Before 2026-11-20 the first hop started from the optimum's own
  vector, which on a ridge lands far from the valley and, with the 40-evaluation budget, on a worse branch that every
  later step was chained from. A point that stopped far from stationary (projected gradient above ``predict_pg_ok``) is not
  used as a start. A nuisance on a bound is widened (``widen`` decades at a time, kept per side) and the point re-solved. The
  walk stops once ``dnll > df_stop`` (``level + stop_margin``, so the crossing is bracketed), when the profiled parameter
  has travelled ``max_decades`` from the optimum without a crossing (the side is open), or when a point cannot be
  evaluated. A hop whose point does not settle (converged, or within the gradient's noise floor: :meth:`PointRecord.settled`) is not
  extended from: the walk goes back to the last settled point and halves the hop, up to ``max_halvings`` times, and ends with "no converged
  point" when none settles, so a point over the level that did not converge never brackets a crossing (``adaptive_hop``; its record is kept
  but is not a start, a neighbour for :func:`fill` or a bracket end). The profiled parameter is not held to the declared box. A last point over the stop level is first re-fitted
  from a start half-way back (:func:`_confirm_end`): a big hop can land in a worse minimum of the nuisance parameters, and
  if the re-fit is lower the walk goes on.
* :func:`fill` inserts points where the profile crosses ``level``: a quadratic through the nearest points proposes the
  value, the point is fitted from both bracketing points' nuisance vectors (strict convergence), the lower is kept, and
  the bracket shrinks until ``|dnll - level| <= tol``.
* :func:`densify` bisects the gaps between the optimum and the crossing that are wider than ``max_gap`` (in the parameter)
  or ``max_dgap`` (in dnll); each new point starts from :func:`predict_start` (the mean of its neighbours when both are
  usable).
* :func:`recheck` (optional) refits each point from its neighbours' vectors, forward then backward, keeping the lowest. A
  first pass warm-started into the wrong valley shows up as a decrease.
* :func:`backfill` (optional, ``backfill_points`` > 0) adds warm-started points in the widest gaps until the side has that
  many converged points between the optimum and the crossing (or where the walk gave up).

:func:`pins` and :func:`relax` are the older grid route: a fixed list of values, with nuisance bounds widened afterwards.

A point the solver gave up on ("no model decrease available") at a projected gradient below ``stall_pg`` nats is *stalled*
(:meth:`Profile_record.PointRecord.stalled`): its dnll is at most that far above the profile and more evaluations reproduce
it. It is ``not_converged`` for the strict test, but the reports count it apart from the points that ran out of evaluations.

Every record carries its verdict, its ``lower`` flag (better than the "optimum": the optimum was not global) and, from
the solver, ``bound_active`` (see :func:`Profile_record.flag_bounds`); :func:`Profile_record.side_verdicts` turns the
points into crossed / box-limited / open.
"""
import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from pyantigen.engine.Profile_lm import OutOfTime
from pyantigen.engine.Profile_record import (BOUND_TOL, DNLL_95, STALL_PG, PointRecord, PointRequest, curve, read_jsonl, reproduces,
                                             validate_result)


@dataclass
class WalkSettings:
    df_step: float = 0.4            # target rise of dnll per point
    stop_margin: float = 1.5        # stop once dnll exceeds level + stop_margin: the crossing is bracketed with room to spare
    h_min: float = 0.002            # optimizer-space step limits (decades for a log parameter)
    h_max: float = 0.5
    max_decades: float = 4.0        # how far the profiled parameter may travel from the optimum, and how far a bound may be widened
    widen: float = 1.0              # decades a bound is widened by each time a nuisance parameter is found on it
    no_widen: tuple = ()            # bounds that are never widened: "name" (both) or "name:lower" / "name:upper"; a nuisance on such a bound stays flagged
    level: float = DNLL_95
    fill_tol: float = 0.03          # |dnll - level| at which a bracket end is close enough
    fill_rounds: int = 6
    point_max_evals: int = 40
    pins: tuple = (0.03, 0.1, 0.25, 0.5, 0.7, 0.85, 1.2, 1.5, 2.0, 3.0, 6.0, 15.0)   # relative pins (x the best fit)
    pins_full: bool = False         # solve the whole grid; else stop a side once a converged pin is past df_stop
    pin_rel: Optional[float] = None  # None: pins held exactly; else the pin may move in a +-pin_rel box (solvers that support it)
    relax: float = 1.0              # pins mode only: decades a nuisance bound may be widened to re-solve a box-limited point (0: never)
    max_gap: float = 0.25           # densify: no two neighbouring points further apart than this (decades for a log parameter) ...
    max_dgap: float = 0.5           # ... or than this in dnll, up to the crossing
    dense_rounds: int = 12
    confirm: bool = True            # a walk that ends over the stop level re-fits its last point from a start half-way back
    confirm_tol: float = 0.05       # ... and keeps the re-fit, and goes on walking, when it is lower by more than this (nats)
    backfill_points: int = 0        # backfill: converged points wanted between the optimum and the crossing (0: no backfill pass)
    backfill_rounds: int = 12
    use_tangent: bool = True        # start the first hop (and any step with no pair of usable points to extrapolate from) along the tangent of the profile at the optimum
    tangent_damping: float = 0.5    # ... moving each nuisance this fraction of the tangent's displacement ...
    tangent_clip: float = 0.5       # ... and by at most this many "decades" (``_unit``): the quadratic model at the optimum is only trustworthy for small hops (stored microglia profile: see docs)
    first_hop_max: Optional[float] = None   # cap on the first hop, in "decades" (``_unit``); None: 4 x the Wald sd as before. The stored microglia sides say 0.15
    predict_pg_ok: float = 1.0      # a point that did not converge is a usable start only when its projected gradient (nats) is below this: wrong-branch points are not
    stall_pg: float = STALL_PG      # a point the solver gave up on below this projected gradient (nats) is "stalled", not "capped" (see PointRecord.stalled)
    plateau_window: int = 0         # 0: off. The nuisance LM stops a point that gained less than ``plateau_tol`` nats in the last ``plateau_window`` evaluations ...
    plateau_tol: float = 0.005
    plateau_pg: float = 0.05        # ... with a projected gradient under this (nats): stalled after the window instead of after ``point_max_evals`` (see Profile_lm.plateau_reached)
    adaptive_hop: bool = True       # a hop whose point does not settle is retried from the last settled point at half the hop; a point that did not settle is never a start (False: the old walk)
    max_halvings: int = 5           # ... at most this many times, then the walk ends: the crossing is not bracketed by converged points
    noise_floor: Optional[float] = None     # nats of projected gradient the gradient's own error hides (measured by Profile_amici.run, or given); None: ``stall_pg`` stands in
    noise_floor_mode: str = "measure"       # "measure": Profile_amici.run measures the floor at the optimum (two evaluations at different solver tolerances); "off": it does not
    polish_evals: int = 0           # 0: off. Evaluations of a tighter-tolerance polish of a point that stopped under ``polish_pg`` above the noise floor (Profile_lm.LmSettings.polish_evals)
    polish_pg: float = 0.5
    weak_defaults: bool = True      # Profile_amici.run: a side whose parameter loads on a weak FIM direction gets ``weak_first_hop`` and ``weak_plateau_window`` unless set
    weak_eig: float = 1e-6          # ... "weak": an eigenvalue of J'J under this times the largest
    weak_load: float = 0.1          # ... "loads": the parameter's squared component on those eigenvectors, summed, is at least this
    weak_first_hop: float = 0.15    # ... the cap on the first hop (decades) for such a side when ``first_hop_max`` is None
    weak_plateau_window: int = 20   # ... the plateau window for such a side when ``plateau_window`` is 0
    fill_seconds: Optional[float] = None    # fill: stop inserting after this many seconds (None: the task's own deadline only)

    @property
    def df_stop(self):
        return self.level + self.stop_margin

    @property
    def settle_pg(self):
        """The projected gradient (nats) at or under which a point that did not converge still counts as settled: the measured noise floor, else ``stall_pg``."""
        return self.noise_floor if self.noise_floor is not None else self.stall_pg


@dataclass
class ProfileProblem:
    """What the protocol needs to know about the model; the solver does the fitting."""
    names: list
    lb: np.ndarray                  # declared bounds, optimizer space
    ub: np.ndarray
    best_x: np.ndarray              # the optimum, full vector
    anchor_nll: float               # f at the optimum
    solver: object                  # a NuisanceSolver
    best_value: Callable = None     # k -> linear best-fit value of parameter k
    sd_first: Optional[np.ndarray] = None   # per-parameter scale of the first step (e.g. the Wald lower bound)
    floors_sha256: Optional[str] = None
    tangent: Optional[np.ndarray] = None    # n x n: column k is d(x_i)/d(x_k) along the profile of parameter k at the optimum (see :func:`tangent_matrix`)

    def request(self, k, x_fixed, x_start, direction, **kw):
        bv = self.best_value(k) if self.best_value else None
        return PointRequest(param=self.names[k], param_idx=k, x_fixed=float(x_fixed), x_start=tuple(float(v) for v in x_start),
                            anchor_nll=self.anchor_nll, best_value=bv, direction=direction, **kw)


class JsonlStore:
    """One fsynced JSONL line per point, a ``.done`` file when a pass finishes. A killed writer's partial line is ignored."""

    def __init__(self, directory, tag):
        self.dir, self.tag = directory, tag
        os.makedirs(directory, exist_ok=True)

    def path(self, kind, ext="jsonl"):
        return os.path.join(self.dir, f"{kind}_{self.tag}.{ext}")

    def read(self, kind):
        return [PointRecord.from_dict(d) for d in read_jsonl(self.path(kind))]

    def append(self, kind, rec):
        with open(self.path(kind), "a") as fh:
            fh.write(json.dumps(rec.to_dict()) + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    def done(self, kind):
        p = self.path(kind, "done")
        return open(p).read().strip() if os.path.exists(p) else None

    def state(self, kind):
        p = self.path(kind, "json")
        return json.load(open(p)) if os.path.exists(p) else None

    def set_state(self, kind, obj):
        tmp = f"{self.path(kind, 'json')}.{os.getpid()}.tmp"
        with open(tmp, "w") as fh:
            json.dump(obj, fh)
        os.replace(tmp, self.path(kind, "json"))

    def finish(self, kind, text):
        with open(self.path(kind, "done"), "w") as fh:
            fh.write(text + "\n")


class MemoryStore:
    """Same interface, in memory (tests, and short runs that keep no checkpoint)."""

    def __init__(self):
        self.rows, self.flags, self.states = {}, {}, {}

    def state(self, kind):
        return self.states.get(kind)

    def set_state(self, kind, obj):
        self.states[kind] = obj

    def read(self, kind):
        return list(self.rows.get(kind, []))

    def append(self, kind, rec):
        self.rows.setdefault(kind, []).append(rec)

    def done(self, kind):
        return self.flags.get(kind)

    def finish(self, kind, text):
        self.flags[kind] = text


def plan_step(records, best_xk, sd_first, st: WalkSettings, unit=1.0):
    """Size of the next step (optimizer space, positive) from the points so far (index 0 is the optimum)."""
    last = records[-1]
    d = abs(last.x_fixed - best_xk)
    if len(records) == 1:
        h = float(np.clip(4.0 * (sd_first if sd_first is not None else st.h_min), st.h_min, st.h_max))
        return min(h, st.first_hop_max * unit) if st.first_hop_max is not None else h
    c = max(last.dnll, 1e-6) / d ** 2
    h = math.sqrt((max(last.dnll, 0.0) + st.df_step) / c) - d
    d_prev = abs(records[-2].x_fixed - best_xk) if len(records) > 1 else 0.0
    return float(np.clip(h, st.h_min, min(st.h_max, 3.0 * max(d - d_prev, st.h_min))))


def tangent_matrix(J, x, lb, ub, ridge=1e-9, tol=BOUND_TOL):
    """Direction in which the nuisance parameters move when parameter *k* is moved away from the optimum, for every k: an n x n matrix T, column k.

    From the Gauss-Newton Hessian H = J'J at the optimum (J the residual Jacobian in the optimizer space): minimizing 0.5 d'H d over the nuisance
    part d_N with d_k = 1 gives d_N = -H_NN^-1 H_Nk (the implicit function theorem for the constrained minimum), T[k, k] = 1. A nuisance on a bound at the
    optimum is held there (a zero row). H_NN is regularized by ``ridge`` x its mean diagonal, which keeps a parameter that is flat on its own
    from blowing the step up; a ridge that involves k itself (the two clearance rates and the gain of the microglia fit enter as products, so the
    profile of one moves the others 1:1 along it) is exactly what this recovers, because H_NN is regular once k is taken out of it.
    """
    J = np.asarray(J, float)
    x = np.asarray(x, float)
    H = J.T @ J
    n = len(x)
    free = ~(((x - np.asarray(lb, float)) <= tol) | ((np.asarray(ub, float) - x) <= tol))
    T = np.zeros((n, n))
    for k in range(n):
        T[k, k] = 1.0
        N = [i for i in range(n) if i != k and free[i]]
        if not N:
            continue
        Hnn = H[np.ix_(N, N)]
        lam = ridge * float(np.mean(np.diag(Hnn))) if len(N) else 0.0
        T[N, k] = np.linalg.solve(Hnn + lam * np.eye(len(N)), -H[N, k])
    return T


def _usable_start(rec, st):
    """A record that can seed a neighbour: it has a vector and a finite dnll and either converged, is the optimum, has no projected gradient to judge by, or is close to stationary."""
    if rec.x is None or not rec.usable:
        return False
    if rec.verdict == "converged" or rec.source == "optimum" or rec.pg is None:
        return True
    return float(rec.pg) < st.predict_pg_ok


def predict_start(pr: ProfileProblem, k, val, pts, direction, st: WalkSettings = None):
    """Start vector for the nuisance fit at ``x_k = val``, from the points so far (*pts*, the optimum included).

    * a usable point at *val*: its own vector;
    * *val* between two usable points: linear interpolation between the two nearest;
    * beyond the last usable point: the secant of the last two when they are apart, else the tangent of the profile at the optimum (``pr.tangent``) from the
      last one, damped by ``tangent_damping`` and clipped to ``tangent_clip`` "decades" per nuisance, else the last vector as it is (what the walk did before).
      The damping and the clip come from the stored microglia profile (193 points, every side, against the FIM at the optimum): the quadratic model is right for
      the parameters on the ridge (two clearance rates and the gain: 1:1 over more than a decade) and overshoots the ones that saturate (the dwell time: -3.8 per
      decade of the profiled parameter against -1.9 observed), so the full tangent is worse than the optimum's own vector for hops beyond 0.3 decades; half of it,
      clipped to 0.5, cuts the median error of the start by 3x for hops under 0.3 and 1.4x up to 1.3.
    A point that stopped far from stationary (``predict_pg_ok``) is not a start: it is on a worse branch of the nuisance fit and everything chained from it
    inherits that. Entry k is *val*; the others are clipped into the bounds.
    """
    st = st or WalkSettings()
    use = sorted((p for p in pts if _usable_start(p, st)), key=lambda p: direction * p.x_fixed)
    if not use:
        raise ValueError("no usable point to start from")
    before = [p for p in use if direction * p.x_fixed < direction * val - 1e-12]
    after = [p for p in use if direction * p.x_fixed > direction * val + 1e-12]
    at = [p for p in use if abs(p.x_fixed - val) <= 1e-12]
    if at:
        x = np.array(at[0].x, float)
    elif before and after:
        a, b = before[-1], after[0]
        w = (val - a.x_fixed) / (b.x_fixed - a.x_fixed)
        x = (1.0 - w) * np.array(a.x, float) + w * np.array(b.x, float)
    elif before:
        a = before[-1]
        x = np.array(a.x, float)
        if len(before) >= 2 and abs(a.x_fixed - before[-2].x_fixed) > 0:
            x = x + (x - np.array(before[-2].x, float)) / (a.x_fixed - before[-2].x_fixed) * (val - a.x_fixed)
        elif st.use_tangent and pr.tangent is not None:
            disp = st.tangent_damping * np.asarray(pr.tangent, float)[:, k] * (val - a.x_fixed)
            cap = st.tangent_clip * np.array([_unit(pr, i) for i in range(len(x))])
            x = x + np.clip(disp, -cap, cap)
    else:
        x = np.array(after[0].x, float)
    x = np.clip(x, pr.lb, pr.ub)
    x[k] = val                              # the profiled parameter is not held to the box
    return x


def _optimum_record(pr: ProfileProblem, k):
    return PointRecord(param=pr.names[k], x_fixed=float(pr.best_x[k]), nll=pr.anchor_nll, dnll=0.0, verdict="converged",
                       param_idx=k, x=[float(v) for v in pr.best_x], evals=0, reason="optimum", source="optimum",
                       solver=getattr(pr.solver, "name", None), backend=getattr(pr.solver, "backend", None),
                       floors_sha256=pr.floors_sha256, value=pr.best_value(k) if pr.best_value else None)


def _unit(pr: ProfileProblem, k):
    """Size of one "decade" for parameter *k*: 1 in a log10 space, the declared span in a linear one."""
    scales = getattr(pr.solver, "scales", None)
    if scales is not None and scales[k] == "log10":
        return 1.0
    span = float(pr.ub[k] - pr.lb[k])
    return span if math.isfinite(span) and span > 0 else 1.0


def _with_bounds(pr: ProfileProblem, lb, ub):
    """*pr* with the nuisance bounds *lb*, *ub* (a solver without ``with_bounds`` keeps its own)."""
    solver = pr.solver.with_bounds(lb, ub) if hasattr(pr.solver, "with_bounds") else pr.solver
    return ProfileProblem(**{**pr.__dict__, "lb": np.asarray(lb, float), "ub": np.asarray(ub, float), "solver": solver})


def side_problem(pr: ProfileProblem, store):
    """*pr* under the bounds this side has widened to so far (``bounds_<side>.json``), the declared ones if none."""
    saved = store.state("bounds")
    if not saved:
        return pr
    return _with_bounds(pr, [-math.inf if v is None else v for v in saved["lb"]],
                        [math.inf if v is None else v for v in saved["ub"]])


def _widen(pr: ProfileProblem, cur: ProfileProblem, rec, st: WalkSettings):
    """Bounds of *cur* widened by ``st.widen`` for every nuisance *rec* has on one, never past ``st.max_decades`` beyond the
    declared box of *pr* and never on a bound named in ``st.no_widen``; None if nothing can move."""
    lb, ub, moved = cur.lb.copy(), cur.ub.copy(), False
    fixed = {str(v) for v in st.no_widen}
    for name in rec.bound_active:
        i = pr.names.index(name)
        u = _unit(pr, i)
        if rec.x[i] - cur.lb[i] <= BOUND_TOL and cur.lb[i] > pr.lb[i] - st.max_decades * u:
            if name in fixed or f"{name}:lower" in fixed:
                continue
            lb[i], moved = max(cur.lb[i] - st.widen * u, pr.lb[i] - st.max_decades * u), True
        elif cur.ub[i] - rec.x[i] <= BOUND_TOL and cur.ub[i] < pr.ub[i] + st.max_decades * u:
            if name in fixed or f"{name}:upper" in fixed:
                continue
            ub[i], moved = min(cur.ub[i] + st.widen * u, pr.ub[i] + st.max_decades * u), True
    return (lb, ub) if moved else None


def _save_bounds(store, cur: ProfileProblem):
    fin = lambda v: float(v) if math.isfinite(v) else None
    store.set_state("bounds", {"lb": [fin(v) for v in cur.lb], "ub": [fin(v) for v in cur.ub]})


def _solve_widening(cur: ProfileProblem, pr: ProfileProblem, req, store, st: WalkSettings, say):
    """Solve *req*; while a nuisance sits on a bound, widen that bound and re-solve from where it stopped.

    Returns ``(record, problem)``: *problem* carries the widened bounds, which the side keeps (and stores) from here on, so
    the next point starts under them. A point still on a bound once ``max_decades`` is used up is left flagged.
    """
    rec = cur.solver.solve(req)
    n = 0
    while rec.usable and rec.bound_active and st.widen > 0 and hasattr(cur.solver, "with_bounds"):
        nb = _widen(pr, cur, rec, st)
        if nb is None:
            break
        say(f"{', '.join(rec.bound_active)} on a bound: widening it by {st.widen:g} decade(s) and re-solving")
        cur = _with_bounds(cur, *nb)
        _save_bounds(store, cur)
        n += 1
        req = cur.request(req.param_idx, req.x_fixed, rec.x, req.direction, anchor_x=req.anchor_x, deadline=req.deadline,
                          max_evals=req.max_evals, strict=True, source=f"{req.source}:wide{n}")
        rec = cur.solver.solve(req)
    return rec, cur


def _walk_points(store, kind):
    """The walk's records, each replaced by the lowest ``confirm`` record at its value (the walk's order is kept)."""
    better = {}
    for c in store.read("confirm"):
        if c.usable:
            key = round(c.x_fixed, 12)
            if key not in better or c.dnll < better[key].dnll:
                better[key] = c
    out = []
    for r in store.read(kind):
        c = better.get(round(r.x_fixed, 12))
        out.append(c if c is not None and r.usable and c.dnll < r.dnll else r)
    return out


def _confirm_end(cur: ProfileProblem, base: ProfileProblem, k, direction, pts, st: WalkSettings, store, deadline, say):
    """Re-fit the point that ended the walk over the stop level from a start half-way back; ``(record or None, problem)``.

    A big hop can land a fit in a worse minimum of the nuisance parameters (a ridge, a bound widened mid-solve) and end the walk
    on a dnll the profile does not have: fitted again from a point half-way, the same value comes out lower. The half-way point
    is fitted first from the previous point, then the last value from it, both strictly; both are kept in ``confirm`` records.
    Returns the lower record when it is lower by more than ``st.confirm_tol``, else None.
    """
    last, prev = pts[-1], pts[-2]
    key = round(last.x_fixed, 12)
    if any(round(c.x_fixed, 12) == key for c in store.read("confirm")):
        return None, cur
    n = len(pts) - 1
    xm = 0.5 * (prev.x_fixed + last.x_fixed)
    x_init = np.clip(np.array(prev.x, float), cur.lb, cur.ub)
    x_init[k] = xm
    say(f"confirming the end point x = {last.x_fixed:.5g} (dnll {last.dnll:.4g}) from a start at x = {xm:.5g}")
    req = cur.request(k, xm, x_init, direction, anchor_x=tuple(prev.x), deadline=deadline, max_evals=st.point_max_evals,
                      strict=True, source=f"confirm:{n}:mid")
    mid, cur = _solve_widening(cur, base, req, store, st, say)
    _check(req, mid)
    store.append("confirm", mid)
    if not mid.usable:
        return None, cur
    x_init = np.clip(np.array(mid.x, float), cur.lb, cur.ub)
    x_init[k] = last.x_fixed
    req = cur.request(k, last.x_fixed, x_init, direction, anchor_x=tuple(mid.x), deadline=deadline, max_evals=st.point_max_evals,
                      strict=True, source=f"confirm:{n}")
    again, cur = _solve_widening(cur, base, req, store, st, say)
    _check(req, again)
    store.append("confirm", again)
    if again.usable and again.dnll < last.dnll - st.confirm_tol:
        say(f"the end point was not on the profile: dnll {again.dnll:.4g} from the half-way start against {last.dnll:.4g}")
        return again, cur
    return None, cur


def _settled(rec, st):
    return rec.source == "optimum" or rec.settled(st.settle_pg)


def _chain(pts, st):
    """The points the walk steps from: the optimum and every point that settled. All of them when ``adaptive_hop`` is off."""
    if not st.adaptive_hop:
        return list(pts)
    return [p for p in pts if _settled(p, st)] or [pts[0]]


def _tries(pts, last, st):
    """The records after *last* (the last settled point) in the walk's order: hops that did not settle, oldest first."""
    if not st.adaptive_hop:
        return []
    i = max(j for j, p in enumerate(pts) if p is last)
    return pts[i + 1:]


def walk(pr: ProfileProblem, k, direction, st: WalkSettings = None, store=None, deadline=None, log=None, kind="profile"):
    """Walk from the optimum in *direction* (+1 / -1). Returns ``(records, reason)``; reason None means out of time.

    Every record is kept, but the walk steps only from settled points (:func:`_chain`): a hop that lands on a point that did not settle is
    followed by a hop half as long from the last settled point, so the walk never extends an unconverged tail and never takes a point
    over the level that did not converge for the crossing. The hops tried since the last settled point are read from the store, so a
    relaunch continues the halving where it stopped.
    """
    st = st or WalkSettings()
    store = store or MemoryStore()
    say = log or (lambda *_: None)
    if store.done(kind):
        return store.read(kind), store.done(kind)
    pts = _walk_points(store, kind)
    if not pts:
        pts = [_optimum_record(pr, k)]
        store.append(kind, pts[0])
    bk = float(pr.best_x[k])
    reach = st.max_decades * _unit(pr, k)
    stop_at = bk + direction * reach
    base, pr = pr, side_problem(pr, store)
    sd = None if pr.sd_first is None else float(pr.sd_first[k])
    while True:
        chain = _chain(pts, st)
        last = chain[-1]
        tries = _tries(pts, last, st)
        reason = None
        if st.confirm and len(chain) > 1 and last.dnll is not None and last.dnll > st.df_stop and last.usable:
            try:
                better, pr = _confirm_end(pr, base, k, direction, chain, st, store, deadline, say)
            except OutOfTime:
                say("deadline reached; exiting to be continued")
                return pts, None
            if better is not None:
                pts[max(j for j, p in enumerate(pts) if p is last)] = better
                continue
        if last.dnll is not None and last.dnll > st.df_stop:
            reason = f"dnll {last.dnll:.3g} > {st.df_stop}: threshold crossed"
        elif not st.adaptive_hop and last.verdict == "failed":
            reason = "point could not be evaluated"
        elif len(chain) > 1 and abs(last.x_fixed - stop_at) < 1e-9:
            reason = f"open: {st.max_decades:g} decades from the optimum reached without a crossing"
        elif tries and (len(tries) > st.max_halvings or 0.5 * abs(tries[-1].x_fixed - last.x_fixed) < st.h_min):
            if not tries[-1].usable:
                reason = "point could not be evaluated"
            else:
                reason = (f"no converged point within {len(tries) - 1} halving(s) of the hop from x = {last.x_fixed:.5g} "
                          f"(last try {tries[-1].status(st.settle_pg)}, dnll {tries[-1].dnll:.3g}): the crossing is not bracketed by converged points")
        if reason:
            store.finish(kind, reason)
            say(f"{pr.names[k]} {'up' if direction > 0 else 'down'}: finished: {reason}")
            return pts, reason
        h = plan_step(chain, bk, sd, st, _unit(pr, k))
        if tries:
            h = min(h, 0.5 * abs(tries[-1].x_fixed - last.x_fixed))
        val = float(np.clip(last.x_fixed + direction * h, min(bk, stop_at), max(bk, stop_at)))
        x_prev = np.array(last.x, float)
        try:
            x_init = predict_start(pr, k, val, chain, direction, st)
        except ValueError:                    # nothing usable (cannot happen: the optimum is one): the last vector
            x_init = np.clip(x_prev, pr.lb, pr.ub)
            x_init[k] = val
        req = pr.request(k, val, x_init, direction, anchor_x=tuple(x_prev), deadline=deadline,
                         max_evals=st.point_max_evals, source=f"walk:{len(pts)}")
        say(f"{pr.names[k]} point {len(pts)}: x = {val:.5g} (step {val - last.x_fixed:+.4g})")
        try:
            rec, pr = _solve_widening(pr, base, req, store, st, say)
        except OutOfTime:
            say("deadline reached; exiting to be continued")
            return pts, None
        _check(req, rec)
        store.append(kind, rec)
        pts.append(rec)
        say(f"{pr.names[k]} point {len(pts) - 1}: dnll {rec.dnll if rec.dnll is not None else float('nan'):.4f} "
            f"{rec.verdict} ({rec.reason}); {rec.evals} evals")
        if st.adaptive_hop and not _settled(rec, st):
            say(f"{pr.names[k]} point {len(pts) - 1} did not settle ({rec.status(st.settle_pg)}): "
                f"the next hop is half of {abs(val - last.x_fixed):.4g}, from x = {last.x_fixed:.5g}")
        if rec.lower:
            say(f"*** LOWER than the 'optimum' by {-rec.dnll:.4g}: the optimum was not global ***")


def pin_x(pr: ProfileProblem, k, rel):
    """Optimizer-space value of the pin ``rel x best fit`` for parameter *k* (None if the best fit is not positive)."""
    best = pr.best_value(k) if pr.best_value else None
    if best is None or not best > 0 or not rel > 0:
        return None
    log = getattr(pr.solver, "scales", None) and pr.solver.scales[k] == "log10"
    return float(math.log10(best * rel)) if log else float(best * rel)


def pins(pr: ProfileProblem, k, direction, st: WalkSettings = None, store=None, deadline=None, log=None):
    """The profile of parameter *k* on one side from a grid of relative pins, then the crossing filled in.

    Pins on the side (``rel > 1`` up, ``rel < 1`` down) are solved outward, each warm-started from the nearest converged
    pin; a pin that does not converge is continued once from where its budget ran out. Once a converged pin is
    past ``df_stop`` the rest are skipped unless ``pins_full``. If the grid does not bracket the level, the serial walk
    continues outward from the last pin; then :func:`fill` places points at the crossing. Records go to the ``pins`` kind
    (the optimum first), so :func:`fill` and :func:`Profile_record.side_verdicts` read them like walk points.
    Returns ``(records, reason)``; reason None means out of time.
    """
    st = st or WalkSettings()
    store = store or MemoryStore()
    say = log or (lambda *_: None)
    if store.done("pins"):
        return store.read("pins"), store.done("pins")
    pts = store.read("pins")
    if not pts:
        pts = [_optimum_record(pr, k)]
        store.append("pins", pts[0])
    lb, ub = pr.lb, pr.ub
    reach = st.max_decades * _unit(pr, k)
    lo_x, hi_x = float(pr.best_x[k]) - reach, float(pr.best_x[k]) + reach
    grid = sorted((r for r in st.pins if (r > 1) == (direction > 0) and r != 1), key=lambda r: direction * math.log(r))
    seen = {r.source for r in pts}
    box = st.pin_rel if getattr(pr.solver, "supports_pin_box", False) else None

    def solve(val, start, tag, strict=False):
        x_init = np.array(start, float)
        x_init[k] = val
        req = pr.request(k, val, np.clip(x_init, lb, ub), direction, anchor_x=tuple(start), deadline=deadline,
                         max_evals=st.point_max_evals, strict=strict, pin_rel=box, source=tag)
        rec = pr.solver.solve(req)
        _check(req, rec)
        store.append("pins", rec)
        pts.append(rec)
        say(f"{pr.names[k]} pin {tag}: x = {val:.5g}: dnll {rec.dnll if rec.dnll is not None else float('nan'):.4f} {rec.verdict}")
        return rec

    def neighbour(val, only_converged=True):
        ok = [q for q in pts if q.x is not None and (q.verdict == "converged" or not only_converged)]
        return min(ok, key=lambda q: abs(q.x_fixed - val)) if ok else pts[0]

    try:
        for rel in grid:
            tag = f"pin:{rel:g}"
            val = pin_x(pr, k, rel)
            if val is None or not lo_x <= val <= hi_x or tag in seen:
                continue
            past = [q for q in pts[1:] if q.verdict == "converged" and q.dnll is not None and q.dnll > st.df_stop]
            if past and not st.pins_full:
                say(f"{pr.names[k]}: pin {rel:g} skipped; a converged pin is past {st.df_stop}")
                continue
            rec = solve(val, neighbour(val).x, tag)
            if rec.verdict != "converged" and rec.usable:
                solve(val, rec.x, f"{tag}:restart")           # continue from where the budget ran out
        for q in [q for q in pts[1:] if q.verdict != "converged" and q.usable and not q.source.endswith(":restart")
                  and f"{q.source}:restart" not in {r.source for r in pts}]:
            solve(q.x_fixed, q.x, f"{q.source}:restart")
        if _bracket(ordered_pins(pts, direction), direction, st.level) is None:
            say(f"{pr.names[k]}: the grid does not bracket the level; walking on from the last pin")
            walk(pr, k, direction, st, store, deadline, say, kind="pins")
            if store.done("pins") is None:
                return pts, None
        else:
            store.finish("pins", f"{len(pts) - 1} pin(s)")
    except OutOfTime:
        say("deadline reached; exiting to be continued")
        return pts, None
    return store.read("pins"), store.done("pins")


def ordered_pins(pts, direction):
    return sorted(curve(pts), key=lambda q: direction * q.x_fixed)


def relax(pr: ProfileProblem, k, direction, st: WalkSettings = None, store=None, deadline=None, log=None, walk_kind="profile"):
    """Re-solve the walk's box-limited points with the nuisance bounds widened by ``st.relax`` decades.

    A point with a nuisance on a declared bound is only an upper bound on the profile there. Its relaxed record is stored
    as ``relax``; its ``bound_active`` is judged against the widened bounds, so a nuisance that needed the room is no longer
    flagged and one that still sits on the wider bound still is. :func:`Profile_record.curve` keeps the lower of the two
    records at each value. Returns the relaxed records; solvers without ``relaxed`` leave the walk as it is.
    """
    st = st or WalkSettings()
    store = store or MemoryStore()
    say = log or (lambda *_: None)
    done = store.read("relax")
    if store.done("relax") or st.relax <= 0 or not hasattr(pr.solver, "relaxed"):
        return done
    wide = ProfileProblem(**{**pr.__dict__, "solver": pr.solver.relaxed(st.relax)})
    seen = {r.source for r in done}
    for i, p in enumerate(store.read(walk_kind)):
        tag = f"relax:{i}"
        if i == 0 or not p.usable or not p.box_limited or tag in seen:
            continue
        start = p.x
        last_dnll = p.dnll
        for attempt in range(4):                     # a far point may need more than one budget: continue where it stopped
            src = tag if attempt == 0 else f"{tag}:cont{attempt}"
            if src in seen:
                prev = next(r for r in done if r.source == src)
                if prev.verdict == "converged" or not prev.usable or prev.reproduced:
                    break
                start, last_dnll = prev.x, prev.dnll
                continue
            req = wide.request(k, p.x_fixed, start, direction, anchor_x=tuple(p.x), deadline=deadline,
                               max_evals=st.point_max_evals, strict=True, source=src)
            try:
                rec = wide.solver.solve(req)
            except OutOfTime:
                return done
            _check(req, rec)
            rec.reproduced = rec.verdict != "converged" and reproduces(last_dnll, rec.dnll)   # the same dnll again: another budget buys nothing
            store.append("relax", rec)
            done.append(rec)
            seen.add(src)
            say(f"relax point {i} ({', '.join(p.bound_active)}): dnll {p.dnll:.4f} -> "
                f"{rec.dnll if rec.dnll is not None else float('nan'):.4f} {rec.verdict}; "
                f"still on a bound: {rec.bound_active or 'none'}")
            if rec.verdict == "converged" or not rec.usable or rec.reproduced:
                break
            start, last_dnll = rec.x, rec.dnll
    store.finish("relax", f"{len(done)} point(s) re-solved with bounds widened {st.relax} decade(s)")
    return done


def _check(req, rec):
    bad = validate_result(req, rec)
    if bad:
        raise ValueError("solver broke the contract: " + "; ".join(bad))


def _bracket(pts, direction, level):
    """Indices (lo, hi) into pts sorted outward: the last point at or under *level* before the first over it."""
    hi = next((i for i, q in enumerate(pts) if i > 0 and q.dnll > level), None)
    if hi is None:
        return None
    return max(i for i in range(hi) if pts[i].dnll <= level), hi


def propose(pts, lo, hi, level):
    """Value at which dnll reaches *level* between points lo and hi: a quadratic through the nearest third point, else linear.

    The proposal is kept inside the middle of the bracket (10 % to 90 % of it), and a bracket whose far end is far above the level
    (more than 100 x) is bisected: interpolating towards a dnll of 3e5 puts the point 1e-5 of the way along
    and each fill round then shrinks the bracket by almost nothing (a flat profile that ends in a cliff did this for 12 h).
    """
    a, b = pts[lo], pts[hi]
    others = sorted((q for q in pts if q is not a and q is not b), key=lambda q: abs(q.x_fixed - a.x_fixed))[:1]
    v = None
    if others:
        xs = np.array([others[0].x_fixed, a.x_fixed, b.x_fixed])
        y = np.array([others[0].dnll, a.dnll, b.dnll])
        if len(set(xs)) == 3:
            roots = [r.real for r in np.roots(np.polyfit(xs, y, 2) - np.array([0, 0, level]))
                     if abs(r.imag) < 1e-12 and min(a.x_fixed, b.x_fixed) < r.real < max(a.x_fixed, b.x_fixed)]
            if roots:
                v = float(roots[0])
    if v is None:
        v = a.x_fixed + (level - a.dnll) * (b.x_fixed - a.x_fixed) / (b.dnll - a.dnll)
    span = b.x_fixed - a.x_fixed
    far = max(abs(a.dnll - level), abs(b.dnll - level))
    t = 0.5 if far > 100.0 * abs(level) else min(max((v - a.x_fixed) / span, 0.1), 0.9)
    return a.x_fixed + t * span


def fill(pr: ProfileProblem, k, direction, st: WalkSettings = None, store=None, deadline=None, log=None, walk_kind="profile"):
    """Insert points at the *level* crossing until the closest bracket end is within ``fill_tol``. Returns the crossing or None."""
    st = st or WalkSettings()
    store = store or MemoryStore()
    say = log or (lambda *_: None)
    extra = store.read("fill")
    pr = side_problem(pr, store)
    if store.read("relax") and hasattr(pr.solver, "relaxed"):       # the walk needed room: insert points under the same bounds
        pr = ProfileProblem(**{**pr.__dict__, "solver": pr.solver.relaxed(st.relax)})

    def every():
        return store.read(walk_kind) + store.read("confirm") + store.read("relax") + store.read("fill")

    def ordered():
        # the bracket ends are settled points: the dnll of one that did not converge is only an upper bound of the profile
        return sorted(curve([r for r in every() if _settled(r, st)]), key=lambda q: direction * q.x_fixed)

    def no_bracket():
        seen_over = _bracket(sorted(curve(every()), key=lambda q: direction * q.x_fixed), direction, st.level) is not None
        store.finish("fill", "crossing not bracketed by converged points" if seen_over else "no crossing in the walk")

    t0 = time.time()
    for rnd in range(len(extra), st.fill_rounds):
        pts = ordered()
        br = _bracket(pts, direction, st.level)
        if br is None:
            no_bracket()
            return None
        if st.fill_seconds is not None and time.time() - t0 > st.fill_seconds:
            say(f"fill: {st.fill_seconds:g} s spent; the crossing stays as bracketed")
            break
        lo, hi = br
        if min(abs(pts[lo].dnll - st.level), abs(pts[hi].dnll - st.level)) <= st.fill_tol:
            break
        val = propose(pts, lo, hi, st.level)
        best = None
        for name, src in (("lower", pts[lo]), ("upper", pts[hi])):
            x_init = np.array(src.x, float)
            x_init[k] = val
            req = pr.request(k, val, np.clip(x_init, pr.lb, pr.ub), direction, anchor_x=tuple(pts[lo].x), deadline=deadline,
                             max_evals=st.point_max_evals, strict=True, source=f"fill:{rnd}:{name}")
            try:
                rec = pr.solver.solve(req)
            except OutOfTime:
                return None
            _check(req, rec)
            say(f"fill round {rnd} from {name}: x = {val:.5g}: dnll {rec.dnll if rec.dnll is not None else float('nan'):.5f} {rec.verdict}")
            if rec.usable and (best is None or (_settled(rec, st), -rec.nll) > (_settled(best, st), -best.nll)):
                best = rec
        if best is None:
            store.finish("fill", "no insert could be evaluated")
            return None
        store.append("fill", best)
        extra.append(best)
        if not _settled(best, st):          # neither start settled: the same bracket would propose the same value again
            say(f"fill round {rnd}: the insert did not settle ({best.status(st.settle_pg)}); the bracket stays as it is")
            break
    pts = ordered()
    br = _bracket(pts, direction, st.level)
    if br is None:
        no_bracket()
        return None
    a, c = pts[br[0]], pts[br[1]]
    cross = a.x_fixed + (st.level - a.dnll) * (c.x_fixed - a.x_fixed) / (c.dnll - a.dnll)
    store.finish("fill", json.dumps({"crossing": cross, "bracket": [a.x_fixed, c.x_fixed],
                                     "dnll_bracket": [a.dnll, c.dnll], "rounds": len(extra)}))
    return cross


def densify(pr: ProfileProblem, k, direction, st: WalkSettings = None, store=None, deadline=None, log=None, walk_kind="profile"):
    """Fill the curve between the optimum and the crossing: bisect the widest gap until none is wider than ``max_gap`` in
    the parameter or ``max_dgap`` in dnll. Returns the inserted records; a side with no crossing is left as it is.

    The walk steps by curvature and the fill places points at the level, so a flat stretch or a step that took a large
    jump leaves holes. Each new point is fitted once, from the mean of its two neighbours' vectors (strict convergence).
    The range covered ends at the first point over the level.
    """
    st = st or WalkSettings()
    store = store or MemoryStore()
    say = log or (lambda *_: None)
    done = store.read("dense")
    if store.done("dense"):
        return done
    pr = side_problem(pr, store)
    if store.read("relax") and hasattr(pr.solver, "relaxed"):
        pr = ProfileProblem(**{**pr.__dict__, "solver": pr.solver.relaxed(st.relax)})
    unit = _unit(pr, k)

    def ordered():
        return sorted(curve(store.read(walk_kind) + store.read("confirm") + store.read("relax") + store.read("fill") + store.read("dense")),
                      key=lambda q: direction * q.x_fixed)

    def widest(pts):
        end = next((i for i, q in enumerate(pts) if i > 0 and q.dnll > st.level), None)
        if end is None:
            return None
        # A gap too narrow to subdivide usefully is a jump in dnll between neighbouring values, not a hole: bisecting it only
        # converges on its end. The jump is _confirm_end's business; here it is left alone.
        gaps = [(max(abs(b.x_fixed - a.x_fixed) / (st.max_gap * unit), abs(b.dnll - a.dnll) / st.max_dgap), i)
                for i, (a, b) in enumerate(zip(pts[:end], pts[1:end + 1]))
                if abs(b.x_fixed - a.x_fixed) > 0.02 * st.max_gap * unit]
        if not gaps:
            return None
        worst = max(gaps)
        return worst[1] if worst[0] > 1.0 else None

    try:
        for rnd in range(len(done), st.dense_rounds):
            pts = ordered()
            i = widest(pts)
            if i is None:
                break
            a, b = pts[i], pts[i + 1]
            val = 0.5 * (a.x_fixed + b.x_fixed)
            try:
                x_init = predict_start(pr, k, val, pts, direction, st)     # between its two neighbours, or between the nearest usable ones when a neighbour is a wrong-branch point
            except ValueError:
                x_init = 0.5 * (np.array(a.x, float) + np.array(b.x, float))
            x_init[k] = val
            req = pr.request(k, val, np.clip(x_init, pr.lb, pr.ub), direction, anchor_x=tuple(a.x), deadline=deadline,
                             max_evals=st.point_max_evals, strict=True, source=f"dense:{rnd}")
            rec = pr.solver.solve(req)
            _check(req, rec)
            store.append("dense", rec)
            done.append(rec)
            say(f"densify round {rnd}: x = {val:.5g}: dnll {rec.dnll if rec.dnll is not None else float('nan'):.4f} {rec.verdict}")
    except OutOfTime:
        return done
    store.finish("dense", f"{len(done)} point(s) inserted")
    return done


def backfill(pr: ProfileProblem, k, direction, st: WalkSettings = None, store=None, deadline=None, log=None, walk_kind="profile"):
    """Last pass on a side: at least ``st.backfill_points`` converged points between the optimum and where the curve
    crosses the level, or, if it never does, where the walk gave up. Returns the inserted records.

    It is optional: with ``backfill_points`` 0 (the default) nothing runs and no flag is written, so a profile made without it
    is complete, and one run later with a target adds the points to the files the profile already has.
    The points are placed in the widest gap of the parameter (in decades), each fitted once and strictly, from the
    neighbours' vectors interpolated to its value. A gap that ends in the crossing is cut at the interpolated crossing, so
    a new point lies inside the level. A point that comes out over the level moves the crossing in, and a point that does
    not converge does not count and its value is not tried again. The pass runs after every other, so what it adds
    is only what they left sparse.
    """
    st = st or WalkSettings()
    store = store or MemoryStore()
    say = log or (lambda *_: None)
    target = int(st.backfill_points)
    if target <= 0:
        return store.read("backfill")
    tag = f"target {target}:"
    flag = store.done("backfill")
    if flag is not None and flag.startswith(tag):
        return store.read("backfill")
    done = store.read("backfill")
    pr = side_problem(pr, store)
    if store.read("relax") and hasattr(pr.solver, "relaxed"):
        pr = ProfileProblem(**{**pr.__dict__, "solver": pr.solver.relaxed(st.relax)})
    unit = _unit(pr, k)
    tried = set()

    def ordered():
        return sorted(curve(store.read(walk_kind) + store.read("confirm") + store.read("relax") + store.read("fill") + store.read("dense")
                            + store.read("backfill")), key=lambda q: direction * q.x_fixed)

    def span(pts):
        """``(inner, nodes)``: the points inside the level, and ``(x, left, right)`` per gap, left to right.

        *left* is the record the gap starts at and *right* the one it ends at, which for a gap that ends in the crossing is
        the first point over the level (its end in x is the interpolated crossing)."""
        end = next((i for i, q in enumerate(pts) if i > 0 and q.dnll > st.level), None)
        if end is None:                                  # open: the farthest point is where the walk gave up
            inner, last = pts[1:-1], pts[-1]
            x_end = last.x_fixed
        else:
            inner, last = pts[1:end], pts[end]
            a = pts[end - 1]
            x_end = a.x_fixed + (st.level - a.dnll) / (last.dnll - a.dnll) * (last.x_fixed - a.x_fixed)
        left = [pts[0]] + inner
        xs = [q.x_fixed for q in left] + [x_end]
        return inner, [(xs[j], xs[j + 1], left[j], left[j + 1] if j + 1 < len(left) else last) for j in range(len(left))]

    try:
        for rnd in range(len(done), st.backfill_rounds):
            pts = ordered()
            if len(pts) < 2:
                break
            inner, gaps = span(pts)
            if sum(1 for q in inner if q.verdict == "converged") >= target:
                break
            order = sorted(gaps, key=lambda g: -abs(g[1] - g[0]))
            gap = next((g for g in order if round(0.5 * (g[0] + g[1]), 9) not in tried), None)
            if gap is None:
                break
            lo, hi, a, b = gap
            val = 0.5 * (lo + hi)
            tried.add(round(val, 9))
            t = (val - a.x_fixed) / (b.x_fixed - a.x_fixed) if b.x_fixed != a.x_fixed else 0.5
            x_init = np.array(a.x, float) + t * (np.array(b.x, float) - np.array(a.x, float))
            x_init[k] = val
            req = pr.request(k, val, np.clip(x_init, pr.lb, pr.ub), direction, anchor_x=tuple(a.x), deadline=deadline,
                             max_evals=st.point_max_evals, strict=True, source=f"backfill:{rnd}")
            rec = pr.solver.solve(req)
            _check(req, rec)
            store.append("backfill", rec)
            done.append(rec)
            say(f"backfill round {rnd}: x = {val:.5g}: dnll {rec.dnll if rec.dnll is not None else float('nan'):.4f} {rec.verdict}")
    except OutOfTime:
        return done
    pts = ordered()
    inside = sum(1 for q in span(pts)[0] if q.verdict == "converged") if len(pts) > 1 else 0
    if inside < target:
        say(f"backfill: {inside} converged point(s) inside the level, short of {target}")
    store.finish("backfill", f"{tag} {len(done)} point(s) added, {inside} converged inside the level")
    return done


def recheck(pr: ProfileProblem, k, direction, st: WalkSettings = None, store=None, deadline=None, log=None, walk_kind="profile"):
    """Refit every walk point from other starts (itself, the previous, then the next point), keeping the lowest nll.

    With ``PYANTIGEN_PROFILE_RECHECK=rewalk`` the pass is :func:`pyantigen.engine.Profile_rewalk.rewalk` instead: a warm re-walk in small steps of a
    side whose crossing the standard passes left unresolved, box-limited or behind a gap.
    """
    if os.environ.get("PYANTIGEN_PROFILE_RECHECK") == "rewalk":
        from pyantigen.engine.Profile_rewalk import rewalk
        return rewalk(pr, k, direction, st, store, deadline, log, walk_kind)
    st = st or WalkSettings()
    store = store or MemoryStore()
    say = log or (lambda *_: None)
    first = store.read(walk_kind)
    if store.done("recheck") or len(first) < 2:
        return store.read("recheck")
    done = store.read("recheck")
    seen = {r.source for r in done}
    best = {i: p for i, p in enumerate(first)}
    for r in done:
        i = int(r.source.split(":")[1])
        if r.usable and r.nll < best[i].nll:
            best[i] = r

    def attempt(i, start, x_init):
        tag = f"recheck:{i}:{start}"
        if tag in seen:
            return
        x0 = np.clip(np.array(x_init, float), pr.lb, pr.ub)
        x0[k] = first[i].x_fixed
        req = pr.request(k, first[i].x_fixed, x0, direction, anchor_x=tuple(first[i - 1].x), deadline=deadline,
                         max_evals=st.point_max_evals, strict=True, source=tag)
        rec = pr.solver.solve(req)
        _check(req, rec)
        store.append("recheck", rec)
        seen.add(tag)
        if rec.usable and rec.nll < best[i].nll:
            best[i] = rec
        say(f"recheck point {i} from {start}: dnll {rec.dnll if rec.dnll is not None else float('nan'):.5f} "
            f"(first {first[i].dnll:.5f}) {rec.verdict}")

    try:
        n = len(first) - 1
        for i in range(1, n + 1):
            attempt(i, "self", first[i].x)
            attempt(i, "prev", best[i - 1].x)
        for i in range(n - 1, 0, -1):
            attempt(i, "next", best[i + 1].x)
    except OutOfTime:
        return store.read("recheck")
    store.finish("recheck", "all points rechecked")
    return store.read("recheck")

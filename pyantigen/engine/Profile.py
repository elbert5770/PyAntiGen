"""The profile-likelihood stepping protocol, independent of the solver that produces each point.

It replaces the tiered ``Profile_plan`` / ``Stop_rules`` protocol. Everything here talks to a
:class:`~Profile_record.NuisanceSolver` and reads and writes :class:`~Profile_record.PointRecord`, so the same protocol runs
on the LM/AMICI solver (:mod:`Profile_lm`) or on the RoadRunner Nelder-Mead worker (``RoadRunnerSolver``).

Passes, each resumable from its JSONL file and run in this order for one side (a parameter and a direction):

* :func:`walk` leaves the optimum in one direction. The step follows the profile's own curvature: from the last point
  (``dnll`` at distance ``d``) it assumes ``dnll = c d^2`` and takes the step that raises ``dnll`` by ``df_step``, capped by
  ``h_max`` and by three times the previous step. The nuisance parameters start from a linear predictor through the last
  two points. A nuisance on a bound is widened (``widen`` decades at a time, kept per side) and the point re-solved. The
  walk stops once ``dnll > df_stop`` (``level + stop_margin``, so the crossing is bracketed), when the profiled parameter
  has travelled ``max_decades`` from the optimum without a crossing (the side is open), or when a point cannot be
  evaluated. The profiled parameter is not held to the declared box. A last point over the stop level is first re-fitted
  from a start half-way back (:func:`_confirm_end`): a big hop can land in a worse minimum of the nuisance parameters, and
  if the re-fit is lower the walk goes on.
* :func:`fill` inserts points where the profile crosses ``level``: a quadratic through the nearest points proposes the
  value, the point is fitted from both bracketing points' nuisance vectors (strict convergence), the lower is kept, and
  the bracket shrinks until ``|dnll - level| <= tol``.
* :func:`densify` bisects the gaps between the optimum and the crossing that are wider than ``max_gap`` (in the parameter)
  or ``max_dgap`` (in dnll).
* :func:`recheck` (optional) refits each point from its neighbours' vectors, forward then backward, keeping the lowest. A
  first pass warm-started into the wrong valley shows up as a decrease.
* :func:`backfill` (optional, ``backfill_points`` > 0) adds warm-started points in the widest gaps until the side has that
  many converged points between the optimum and the crossing (or where the walk gave up).

:func:`pins` and :func:`relax` are the older grid route: a fixed list of values, with nuisance bounds widened afterwards.

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
from pyantigen.engine.Profile_record import BOUND_TOL, DNLL_95, PointRecord, PointRequest, curve, read_jsonl, validate_result


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

    @property
    def df_stop(self):
        return self.level + self.stop_margin


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


def plan_step(records, best_xk, sd_first, st: WalkSettings):
    """Size of the next step (optimizer space, positive) from the points so far (index 0 is the optimum)."""
    last = records[-1]
    d = abs(last.x_fixed - best_xk)
    if len(records) == 1:
        return float(np.clip(4.0 * (sd_first if sd_first is not None else st.h_min), st.h_min, st.h_max))
    c = max(last.dnll, 1e-6) / d ** 2
    h = math.sqrt((max(last.dnll, 0.0) + st.df_step) / c) - d
    d_prev = abs(records[-2].x_fixed - best_xk) if len(records) > 1 else 0.0
    return float(np.clip(h, st.h_min, min(st.h_max, 3.0 * max(d - d_prev, st.h_min))))


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


def walk(pr: ProfileProblem, k, direction, st: WalkSettings = None, store=None, deadline=None, log=None, kind="profile"):
    """Walk from the optimum in *direction* (+1 / -1). Returns ``(records, reason)``; reason None means out of time."""
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
        last = pts[-1]
        reason = None
        if st.confirm and len(pts) > 1 and last.dnll is not None and last.dnll > st.df_stop and last.usable:
            try:
                better, pr = _confirm_end(pr, base, k, direction, pts, st, store, deadline, say)
            except OutOfTime:
                say("deadline reached; exiting to be continued")
                return pts, None
            if better is not None:
                pts[-1] = last = better
        if last.dnll is not None and last.dnll > st.df_stop:
            reason = f"dnll {last.dnll:.3g} > {st.df_stop}: threshold crossed"
        elif last.verdict == "failed":
            reason = "point could not be evaluated"
        elif len(pts) > 1 and abs(last.x_fixed - stop_at) < 1e-9:
            reason = f"open: {st.max_decades:g} decades from the optimum reached without a crossing"
        if reason:
            store.finish(kind, reason)
            say(f"{pr.names[k]} {'up' if direction > 0 else 'down'}: finished: {reason}")
            return pts, reason
        h = plan_step(pts, bk, sd, st)
        val = float(np.clip(last.x_fixed + direction * h, min(bk, stop_at), max(bk, stop_at)))
        x_prev = np.array(last.x, float)
        x_init = x_prev.copy()
        if len(pts) > 1 and abs(last.x_fixed - pts[-2].x_fixed) > 0:
            slope = (x_prev - np.array(pts[-2].x, float)) / (last.x_fixed - pts[-2].x_fixed)
            x_init = x_prev + slope * (val - last.x_fixed)
        x_init = np.clip(x_init, pr.lb, pr.ub)
        x_init[k] = val                       # the profiled parameter is not held to the box
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
        for attempt in range(4):                     # a far point may need more than one budget: continue where it stopped
            src = tag if attempt == 0 else f"{tag}:cont{attempt}"
            if src in seen:
                prev = next(r for r in done if r.source == src)
                if prev.verdict == "converged" or not prev.usable:
                    break
                start = prev.x
                continue
            req = wide.request(k, p.x_fixed, start, direction, anchor_x=tuple(p.x), deadline=deadline,
                               max_evals=st.point_max_evals, strict=True, source=src)
            try:
                rec = wide.solver.solve(req)
            except OutOfTime:
                return done
            _check(req, rec)
            store.append("relax", rec)
            done.append(rec)
            seen.add(src)
            say(f"relax point {i} ({', '.join(p.bound_active)}): dnll {p.dnll:.4f} -> "
                f"{rec.dnll if rec.dnll is not None else float('nan'):.4f} {rec.verdict}; "
                f"still on a bound: {rec.bound_active or 'none'}")
            if rec.verdict == "converged" or not rec.usable:
                break
            start = rec.x
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

    def ordered():
        return sorted(curve(store.read(walk_kind) + store.read("confirm") + store.read("relax") + store.read("fill")), key=lambda q: direction * q.x_fixed)

    for rnd in range(len(extra), st.fill_rounds):
        pts = ordered()
        br = _bracket(pts, direction, st.level)
        if br is None:
            store.finish("fill", "no crossing in the walk")
            return None
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
            if rec.usable and (best is None or rec.nll < best.nll):
                best = rec
        if best is None:
            store.finish("fill", "no insert could be evaluated")
            return None
        store.append("fill", best)
        extra.append(best)
    pts = ordered()
    br = _bracket(pts, direction, st.level)
    if br is None:
        store.finish("fill", "no crossing in the walk")
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
    """Refit every walk point from other starts (itself, the previous, then the next point), keeping the lowest nll."""
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

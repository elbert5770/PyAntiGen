"""The profile-likelihood stepping protocol, independent of the solver that produces each point.

Ported from the AMICI route (``tools/pypesto_silk_appfull_{profile,fill,recheck}.py``), which replaces the tiered
``Profile_plan`` / ``Stop_rules`` protocol. Everything here talks to a :class:`~Profile_record.NuisanceSolver` and
reads and writes :class:`~Profile_record.PointRecord`, so the same protocol runs on the LM/AMICI solver
(:mod:`Profile_lm`) or on the RoadRunner Nelder-Mead worker (``RoadRunnerSolver``).

Three passes, each resumable from its JSONL file:

* :func:`walk` leaves the optimum in one direction. The step follows the profile's own curvature: from the last point
  (``dnll`` at distance ``d``) it assumes ``dnll = c d^2`` and takes the step that raises ``dnll`` by ``df_step``, capped by
  ``h_max`` and by three times the previous step. The nuisance parameters start from a linear predictor through the last
  two points. It stops once ``dnll > df_stop`` (the crossing is bracketed), when the profiled parameter has left its
  declared box by ``extend`` decades, or when a point cannot be evaluated.
* :func:`fill` inserts points where the profile crosses ``level``: a quadratic through the nearest points proposes the
  value, the point is fitted from both bracketing points' nuisance vectors (strict convergence), the lower is kept, and
  the bracket shrinks until ``|dnll - level| <= tol``.
* :func:`recheck` refits each point from its neighbours' vectors, forward then backward, keeping the lowest. A first pass
  warm-started into the wrong valley shows up as a decrease.

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
from pyantigen.engine.Profile_record import DNLL_95, PointRecord, PointRequest, curve, read_jsonl, validate_result


@dataclass
class WalkSettings:
    df_step: float = 0.4            # target rise of dnll per point
    df_stop: float = 2.5            # stop once dnll exceeds this: the crossing is bracketed
    h_min: float = 0.002            # optimizer-space step limits (decades for a log parameter)
    h_max: float = 0.5
    extend: float = 1.0             # decades the profiled parameter may leave its declared box
    level: float = DNLL_95
    fill_tol: float = 0.03          # |dnll - level| at which a bracket end is close enough
    fill_rounds: int = 6
    point_max_evals: int = 40
    pins: tuple = (0.03, 0.1, 0.25, 0.5, 0.7, 0.85, 1.2, 1.5, 2.0, 3.0, 6.0, 15.0)   # relative pins (x the best fit)
    pins_full: bool = False         # solve the whole grid; else stop a side once a converged pin is past df_stop
    pin_rel: Optional[float] = None  # None: pins held exactly; else the pin may move in a +-pin_rel box (solvers that support it)
    relax: float = 1.0              # decades a nuisance bound may be widened to re-solve a box-limited point (0: never)


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

    def finish(self, kind, text):
        with open(self.path(kind, "done"), "w") as fh:
            fh.write(text + "\n")


class MemoryStore:
    """Same interface, in memory (tests, and short runs that keep no checkpoint)."""

    def __init__(self):
        self.rows, self.flags = {}, {}

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
        return max(4.0 * (sd_first if sd_first is not None else st.h_min), st.h_min)
    c = max(last.dnll, 1e-6) / d ** 2
    h = math.sqrt((max(last.dnll, 0.0) + st.df_step) / c) - d
    d_prev = abs(records[-2].x_fixed - best_xk) if len(records) > 1 else 0.0
    return float(np.clip(h, st.h_min, min(st.h_max, 3.0 * max(d - d_prev, st.h_min))))


def _optimum_record(pr: ProfileProblem, k):
    return PointRecord(param=pr.names[k], x_fixed=float(pr.best_x[k]), nll=pr.anchor_nll, dnll=0.0, verdict="converged",
                       param_idx=k, x=[float(v) for v in pr.best_x], evals=0, reason="optimum", source="optimum",
                       solver=getattr(pr.solver, "name", None), backend=getattr(pr.solver, "backend", None),
                       floors_sha256=pr.floors_sha256, value=pr.best_value(k) if pr.best_value else None)


def walk(pr: ProfileProblem, k, direction, st: WalkSettings = None, store=None, deadline=None, log=None, kind="profile"):
    """Walk from the optimum in *direction* (+1 / -1). Returns ``(records, reason)``; reason None means out of time."""
    st = st or WalkSettings()
    store = store or MemoryStore()
    say = log or (lambda *_: None)
    if store.done(kind):
        return store.read(kind), store.done(kind)
    pts = store.read(kind)
    if not pts:
        pts = [_optimum_record(pr, k)]
        store.append(kind, pts[0])
    lb, ub = pr.lb, pr.ub
    stop_at = ub[k] + st.extend if direction > 0 else lb[k] - st.extend
    bk = float(pr.best_x[k])
    sd = None if pr.sd_first is None else float(pr.sd_first[k])
    while True:
        last = pts[-1]
        reason = None
        if last.dnll is not None and last.dnll > st.df_stop:
            reason = f"dnll {last.dnll:.3g} > {st.df_stop}: threshold crossed"
        elif last.verdict == "failed":
            reason = "point could not be evaluated"
        elif len(pts) > 1 and abs(last.x_fixed - stop_at) < 1e-9:
            reason = f"extension limit reached ({st.extend} decades beyond the declared bound): crossing unresolved"
        if reason:
            store.finish(kind, reason)
            say(f"{pr.names[k]} {'up' if direction > 0 else 'down'}: finished: {reason}")
            return pts, reason
        h = plan_step(pts, bk, sd, st)
        val = float(np.clip(last.x_fixed + direction * h, lb[k] - st.extend, ub[k] + st.extend))
        x_prev = np.array(last.x, float)
        x_init = x_prev.copy()
        if len(pts) > 1 and abs(last.x_fixed - pts[-2].x_fixed) > 0:
            slope = (x_prev - np.array(pts[-2].x, float)) / (last.x_fixed - pts[-2].x_fixed)
            x_init = x_prev + slope * (val - last.x_fixed)
        x_init = np.clip(x_init, lb, ub)
        x_init[k] = val                       # only the profiled parameter may leave its box
        req = pr.request(k, val, x_init, direction, anchor_x=tuple(x_prev), deadline=deadline,
                         max_evals=st.point_max_evals, source=f"walk:{len(pts)}")
        say(f"{pr.names[k]} point {len(pts)}: x = {val:.5g} (step {val - last.x_fixed:+.4g})")
        try:
            rec = pr.solver.solve(req)
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
    lo_x, hi_x = lb[k] - st.extend, ub[k] + st.extend
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
        req = wide.request(k, p.x_fixed, p.x, direction, anchor_x=tuple(p.x), deadline=deadline, max_evals=st.point_max_evals,
                           strict=True, source=tag)
        try:
            rec = wide.solver.solve(req)
        except OutOfTime:
            return done
        _check(req, rec)
        store.append("relax", rec)
        done.append(rec)
        say(f"relax point {i} ({', '.join(p.bound_active)}): dnll {p.dnll:.4f} -> "
            f"{rec.dnll if rec.dnll is not None else float('nan'):.4f} {rec.verdict}; still on a bound: {rec.bound_active or 'none'}")
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
    """Value at which dnll reaches *level* between points lo and hi: a quadratic through the nearest third point, else linear."""
    a, b = pts[lo], pts[hi]
    others = sorted((q for q in pts if q is not a and q is not b), key=lambda q: abs(q.x_fixed - a.x_fixed))[:1]
    if others:
        v = np.array([others[0].x_fixed, a.x_fixed, b.x_fixed])
        y = np.array([others[0].dnll, a.dnll, b.dnll])
        if len(set(v)) == 3:
            roots = [r.real for r in np.roots(np.polyfit(v, y, 2) - np.array([0, 0, level]))
                     if abs(r.imag) < 1e-12 and min(a.x_fixed, b.x_fixed) < r.real < max(a.x_fixed, b.x_fixed)]
            if roots:
                return float(roots[0])
    return a.x_fixed + (level - a.dnll) * (b.x_fixed - a.x_fixed) / (b.dnll - a.dnll)


def fill(pr: ProfileProblem, k, direction, st: WalkSettings = None, store=None, deadline=None, log=None, walk_kind="profile"):
    """Insert points at the *level* crossing until the closest bracket end is within ``fill_tol``. Returns the crossing or None."""
    st = st or WalkSettings()
    store = store or MemoryStore()
    say = log or (lambda *_: None)
    extra = store.read("fill")
    if store.read("relax") and hasattr(pr.solver, "relaxed"):       # the walk needed room: insert points under the same bounds
        pr = ProfileProblem(**{**pr.__dict__, "solver": pr.solver.relaxed(st.relax)})

    def ordered():
        return sorted(curve(store.read(walk_kind) + store.read("relax") + store.read("fill")), key=lambda q: direction * q.x_fixed)

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

"""Re-walk a profile side outward, warm and in small steps, where the standard passes left its crossing unresolved or box-limited.

    PYANTIGEN_PROFILE_RECHECK=rewalk      the engine's ``Profile.recheck`` runs :func:`rewalk` (one side per process, as the AMICI profile array tasks do)

WHY. The engine's walk (pyantigen.engine.Profile.walk) takes its first step from the optimum as 4 x the Fisher SD, and every later step
from a quadratic model of the points so far. When that first hop lands on a worse minimum of the nuisance parameters (a ridge), the walk
records dnll values the profile does not have, and everything after is chained from them. The engine has three remedies, and none reaches
the low branch beyond its last inside point:

* ``confirm`` refits only the walk's last point, from half-way back along the walk itself, which is the same wrong branch;
* ``fill`` bisects the bracket between the last inside point and the first point over the level, from both neighbours. It finds the low
  branch inside the bracket (the "from lower" fits) but never leaves it, so the crossing stays pinned to the bracket's far end;
* ``recheck`` refits each walk point from itself, the previous walk point and the next one, all on the same branch.

The k_oligo1_AB42 upper side of aggregation_silk_amici_widebounds (2026-11-14) is the case: the walk hopped 0.2975 decades to dnll 3.18, the
fill rounds then found 1.52 at 0.004 decades short of it, and nothing walked on from the 1.52 point.

WHAT (version 2, 2026-11-14). A side is taken up when either
  (a) its crossing is not closely bracketed: the farthest point inside the level is more than RESOLVED_BELOW nats under it AND the first point
      past it is more than RESOLVED_ABOVE nats over it; or
  (b) the farthest point inside the level, or the first past it, has a nuisance parameter on a bound (its dnll is then only an upper bound).
It walks on from a start point, each step warm-started from the last (slope-extrapolated, like the engine's walk), with the engine's own step
rule (``plan_step``: the 0.4 nat target and the 3x growth cap) capped at REWALK_HMAX decades (0.02). The start point is the farthest inside
point for (a), and for (b) the last point before the unbroken run of box-limited points that ends at the crossing. The steps land exactly on
the x values the standard passes already have past the start, so ``curve`` (the lowest dnll at each value) replaces them. Every solve goes
through the engine's ``_solve_widening``: a nuisance found on a bound has that bound widened (a decade at a time, at most ``max_decades``
beyond the declared box, never a bound in ``no_widen``) and the point is solved again; the widened bounds are saved in the side's
``bounds_<k>_<dir>.json`` and kept for the rest of the side, as the engine's walk does. When a step ends over the level the crossing is
bracketed on the warm branch and bisected, from the inside end, until it is within ``fill_tol`` of the level. A side that is open or already
resolved on both counts is left alone.

Version 1 had no widening: its k_oligo1_AB42 re-walk ran into k_denseplaque's declared bound from x = -8.9684 on, and its crossing
(1.479e-9) was box-limited. Its records (source ``recheck:rewalk:<x>``) are kept; version 2 writes ``recheck:rewalk2:<x>``.

FORCED START (2026-11-19). ``REWALK_FORCE_FROM=k:direction:x[,...]`` walks the named sides on from the recorded point at x although ``unresolved`` calls them resolved: for a
gap the standard walk's first hop left between the optimum and a closely bracketed crossing (microglia_activation_adubar_alphafree_amici, tasks 2 and 4: ``1:1:1.055992,2:1:2.174526``).

WHERE. Records go to the side's ``recheck`` file; the engine's collect with PYANTIGEN_PROFILE_RECHECK=1 reads them with the rest. This pass does
not write the ``recheck`` done flag (a project's continue tool does, and returns at once when it finds it), it writes its own ``rewalk2`` flag,
so a side that continue has finished can still be re-walked, and a side that is re-walked can still be continued, but not at the same time. A
killed task is relaunched with the same command: the chain is rebuilt from the store.

    REWALK_EVALS   evaluations per point (200)       REWALK_HMAX   largest step, decades (0.02)       REWALK_MAXPTS   points before giving up (120)
    REWALK_FORCE_FROM   k:direction:x[,...]  (see FORCED START)

Moved from a project's tools/profile_rewalk.py into the engine (2026-11-19); the project script is now a thin launcher that sets PYANTIGEN_PROFILE_RECHECK=rewalk.
"""
import os

import numpy as np

FLAG = "rewalk2"
TAG = "recheck:rewalk2:"
BISECT_ROUNDS = 8
MIN_GAP = 1e-3                      # decades: the bisection stops here
RESOLVED_BELOW = 0.1                # nats the farthest inside point may be under the level and still count as the crossing
RESOLVED_ABOVE = 0.3                # nats the first point past it may be over the level and still count as the crossing
KINDS = ("profile", "pins", "relax", "confirm", "fill", "dense", "recheck", "backfill")


def settings():
    """``{"evals", "hmax", "maxpts"}`` from REWALK_EVALS, REWALK_HMAX and REWALK_MAXPTS (read at call time)."""
    return {"evals": int(os.environ.get("REWALK_EVALS", "200")),
            "hmax": float(os.environ.get("REWALK_HMAX", "0.02")),
            "maxpts": int(os.environ.get("REWALK_MAXPTS", "120"))}


def _best(store, direction):
    from pyantigen.engine.Profile_record import curve
    records = [r for kind in KINDS for r in store.read(kind)]
    return sorted(curve(records), key=lambda q: direction * q.x_fixed)


def unresolved(best, direction, level):
    """``(start point, [points past it], why)`` when the side should be re-walked, else None (resolved, open or empty)."""
    inside = [q for q in best if q.dnll <= level]
    if not inside:
        return None
    a = inside[-1]
    after = [q for q in best if direction * q.x_fixed > direction * a.x_fixed]
    if not after:
        return None
    loose = (level - a.dnll) > RESOLVED_BELOW and (after[0].dnll - level) > RESOLVED_ABOVE
    limited = bool(a.box_limited or after[0].box_limited)
    if not (loose or limited):
        return None
    start = a
    if limited:
        i = best.index(a)
        while i > 0 and best[i].box_limited:          # back over the unbroken run of box-limited points that ends at the crossing
            i -= 1
        start = best[i]
    beyond = [q for q in best if direction * q.x_fixed > direction * start.x_fixed]
    why = "a crossing that is not closely bracketed" if loose else "a crossing with a nuisance on a bound"
    if loose and limited:
        why += " and with a nuisance on a bound"
    return start, beyond, why


def forced_start(k, direction, best, spec=None):
    """The walk's start point forced by ``REWALK_FORCE_FROM`` (``k:direction:x`` entries, comma separated, direction 1 up or -1 down; e.g. ``1:1:1.055992,2:1:2.174526``), or None.

    For a gap the standard walk left inside the level: its first hop skipped a stretch and landed on a worse branch, so the side's crossing is closely bracketed
    (``unresolved`` says resolved) while the points between the optimum and the crossing are missing. The start is the recorded point at x (it must exist within 1e-3
    decades); the walk then goes on from it exactly as for an unresolved side, and the points it lands on replace those the standard pass has past the start.
    """
    spec = os.environ.get("REWALK_FORCE_FROM", "") if spec is None else spec
    for item in [t for t in spec.split(",") if t.strip()]:
        kk, dd, xx = item.split(":")
        if int(kk) == k and int(dd) == direction:
            q = min(best, key=lambda q: abs(q.x_fixed - float(xx)))
            if abs(q.x_fixed - float(xx)) > 1e-3:
                raise ValueError(f"REWALK_FORCE_FROM {item}: no recorded point within 1e-3 decades of x = {xx} (nearest {q.x_fixed:.6g})")
            return q
    return None


def rewalk(pr, k, direction, st=None, store=None, deadline=None, log=None, walk_kind="profile"):
    from pyantigen.engine import Profile as PR

    cfg = settings()
    EVALS, HMAX, MAXPTS = cfg["evals"], cfg["hmax"], cfg["maxpts"]
    st = st or PR.WalkSettings()
    store = store or PR.MemoryStore()
    say = log or (lambda *_: None)
    if store.done(FLAG):
        return store.read("recheck")
    base = pr                                     # the declared bounds: widening is measured from them
    cur = PR.side_problem(pr, store)              # the bounds this side's walk had widened
    sd = None if pr.sd_first is None else float(pr.sd_first[k])
    bk = float(pr.best_x[k])
    done = {r.source: r for r in store.read("recheck")}

    def solve_once(val, start, src, anchor, x_start):
        nonlocal cur
        x0 = np.clip(np.array(x_start, float), cur.lb, cur.ub)
        x0[k] = val
        req = cur.request(k, val, x0, direction, anchor_x=tuple(anchor), deadline=deadline, max_evals=EVALS, strict=True, source=src)
        rec, cur = PR._solve_widening(cur, base, req, store, st, say)
        PR._check(req, rec)
        store.append("recheck", rec)
        done[src] = rec
        return rec

    def solve(val, start, src):
        rec = done.get(src) or solve_once(val, start, src, start.x, start.x)
        if rec.usable and rec.verdict != "converged":            # continue once from where the budget ran out
            src2 = src + ":c1"
            rec2 = done.get(src2) or solve_once(val, start, src2, start.x, rec.x)
            if rec2.usable and rec2.dnll <= rec.dnll:
                rec = rec2
        return rec

    def lowest_at(val, fallback):
        at = [q for q in _best(store, direction) if abs(q.x_fixed - val) < 1e-9]
        return at[0] if at else fallback

    try:
        found = unresolved(_best(store, direction), direction, st.level)
        _best_now = _best(store, direction)
        _forced = forced_start(k, direction, _best_now)
        if _forced is not None:
            _beyond = [q for q in _best_now if direction * q.x_fixed > direction * _forced.x_fixed]
            found = (_forced, _beyond, "a gap in the walk (start forced by REWALK_FORCE_FROM)") if _beyond else None
        if found is None:
            store.finish(FLAG, "crossing closely bracketed and free of bounds, or open, or empty; nothing to re-walk")
            say("rewalk: the crossing is closely bracketed and free of bounds, or open, or empty; nothing to re-walk")
            return store.read("recheck")
        last, beyond, why = found
        existing = [q.x_fixed for q in beyond]
        plan = [q for q in _best(store, direction) if direction * q.x_fixed <= direction * last.x_fixed]
        say(f"rewalk: {why}; walking on from x = {last.x_fixed:.5g} (dnll {last.dnll:.4f}), the next point on is x = {existing[0]:.5g} "
            f"(dnll {beyond[0].dnll:.4f}); steps up to {HMAX:g}")
        reason, n = None, 0
        while n < MAXPTS:
            h = min(float(PR.plan_step(plan, bk, sd, st)) if len(plan) > 1 else st.h_min, HMAX)
            val = last.x_fixed + direction * h
            nxt = next((x for x in existing if direction * x > direction * last.x_fixed + 1e-12), None)
            if nxt is not None and direction * (nxt - last.x_fixed) <= 1.5 * h:
                val = nxt                                           # land on the value the standard walk has, so curve replaces it
            if abs(val - bk) > st.max_decades:
                reason = f"open: {st.max_decades:g} decades from the optimum reached without a crossing"
                break
            rec = solve(val, last, f"{TAG}{val:.9g}")
            n += 1
            if not rec.usable:
                reason = "point could not be evaluated"
                break
            cur_pt = lowest_at(val, rec)
            say(f"rewalk x = {val:.5g}: dnll {rec.dnll:.4f} ({rec.verdict}{', on ' + ','.join(rec.bound_active) if rec.bound_active else ''}); "
                f"lowest here {cur_pt.dnll:.4f}")
            if cur_pt.dnll > st.level:
                lo, hi = last, cur_pt                               # bracketed on the warm branch: bisect from the inside end
                for _ in range(BISECT_ROUNDS):
                    if abs(hi.x_fixed - lo.x_fixed) < MIN_GAP:
                        break
                    xm = 0.5 * (lo.x_fixed + hi.x_fixed)
                    mid = lowest_at(xm, solve(xm, lo, f"{TAG}{xm:.9g}"))
                    say(f"rewalk bisect x = {xm:.5g}: dnll {mid.dnll:.4f}")
                    if mid.dnll <= st.level:
                        lo = mid
                    else:
                        hi = mid
                    if abs(mid.dnll - st.level) < st.fill_tol:
                        break
                reason = f"crossing bracketed on the warm branch between x = {lo.x_fixed:.5g} and {hi.x_fixed:.5g}"
                break
            last = cur_pt
            plan.append(cur_pt)
        else:
            reason = f"{MAXPTS} points without a crossing"
    except PR.OutOfTime:
        say("rewalk: deadline reached; exiting to be continued")
        return store.read("recheck")
    store.finish(FLAG, reason)
    say(f"rewalk: finished: {reason}")
    return store.read("recheck")

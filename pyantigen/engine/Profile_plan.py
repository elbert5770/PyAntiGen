"""Where the next profile points should go, planned from everything measured so far.

The grid pass places every point of a pass before any of them has run, so the
opening guess -- a Wald SE, taken at a point that may not be the optimum -- sets
the whole round, and on a model whose points take hours that is a round spent.
This module plans the other way: ask for as many points as there are free
workers, from the evidence in hand, every time a worker frees.

``plan_next_points`` is a pure function of the stored records, the points still
running and the settings. It touches no pool and no clock, so it can be tested
against analytic profiles, and a resumed run plans from the same records the
interrupted one left (the order points land in can differ between runs; the
points already stored are never repeated).

What it reads, per side of each parameter, in the optimizer's own space:

* a side with nothing on it gets a first probe at the further of the capped
  Wald distance (1.96 SE) and the slice screen's own crossing -- the profile
  cannot cross inside the slice, so the slice crossing is a floor, and the Wald
  estimate is the guess;
* a side whose points are all below the threshold is extended outward by the
  curve-fitted step (``_extension_growth``), capped at ``max_step_decades`` per
  step so a flat curve cannot launch a jump of several decades;
* the innermost point past the threshold is the bracket's outer end only if it
  is firm: it converged from a warm start, or it has been restarted once. Any
  other value is an upper bound with an unknown excess -- a warm restart from a
  nearer neighbour only lowers it, and the crossing may then lie beyond it -- so
  it is restarted at once (the nearest two such points), alongside the probes
  that narrow the bracket, not after them. The optimizer is deterministic, so a
  restart is planned only from a start the point did not have: a nearer inner
  neighbour, or its own end state if it did not converge. The probe goes between
  the innermost point at or below the threshold and the outer end (sqrt(dNLL)
  interpolation), warm from the former;
* a bracketed side is narrowed by sqrt(dNLL) interpolation until the bracket is
  tight, then filled in toward ``n_grid`` points so the curve has a shape;
* workers left over once every side has its one probe are given extra probes
  inside open brackets (see ``_SPEC_STEP``) and extra rungs beyond the next one
  on a side still below the threshold (``_SPEC_MAX_EXT``), so a side advancing
  one point per round does not hold the pool at a handful of cores.

Below-threshold points count whether or not they converged: an upper bound at or
under the threshold is still under it.
"""
import numpy as np

# Opening probe in Wald standard errors: where a quadratic profile crosses.
FIRST_PROBE_SE = 1.96

# How the first probe on a side is placed.
#   max           the further of 1.96 Wald SE (capped) and the slice crossing
#   slice         at the slice crossing, ignoring Wald
#   slice_ladder  a log-spaced ladder from the slice crossing out to the
#                 ``open_decades`` cap (100x the slice distance if there is
#                 none), middle rung first. No Wald SE is read at all.
# A side the screen did not cross has no slice crossing to use and falls back
# to "max" in every mode.
FIRST_PROBE_MODES = ("max", "slice", "slice_ladder")

# Points on one side, beyond which the planner stops adding to it. The caller's
# max_extend and n_grid set the real budget; this is the ceiling that makes
# termination independent of them.
_SIDE_CEILING_EXTRA = 4

# Extra probes inside an open bracket, for workers that would otherwise idle. A
# bracket is narrowed by one probe per round, and a round is hours, so a side
# with one open bracket holds a single worker for as long as it takes; the rest
# of the pool drains as the other sides finish. The extras are spaced
# ``_SPEC_STEP`` of the bracket apart around the interpolated guess, nearest the
# guess first, and are only ever planned after every side has had its own
# primary probe (tier ``3 + j`` for the j-th extra), so they use cores that
# nothing else wanted.
_SPEC_STEP = 0.15
_SPEC_MAX = 4
# The same for a side still below the threshold: rungs beyond the next one.
_SPEC_MAX_EXT = 3
# Provisional points past the threshold restarted at once, beyond the nearest.
_SPEC_RERUN = 1


def _bracket_fracs(frac, n_extra=_SPEC_MAX, step=_SPEC_STEP):
    """Fractions of a bracket to probe: the guess, then alternately either side."""
    out = [float(frac)]
    for k in range(1, n_extra + 1):
        off = step * ((k + 1) // 2) * (1 if k % 2 else -1)
        f = float(min(max(frac + off, 0.05), 0.95))
        if all(abs(f - g) > 1e-9 for g in out):
            out.append(f)
    return out


def _converged(rec):
    return bool(rec.get("converged")) and not rec.get("interrupted")


def _rerun_count(rec):
    """How many warm restarts this point has had (the planner's, not the grid's)."""
    n = rec.get("reruns_tried") or 0
    if str(rec.get("pass_label", "")).endswith("rerun"):
        n = max(int(n), 1)
    return int(n)


def _is_firm(rec):
    """Whether a point past the threshold can be trusted as a bracket end.

    A value is firm once it has been through a warm start: it converged from a
    neighbour's solution, or it has been restarted once and is accepted as it
    stands. A cold or unconverged value is only an upper bound with an unknown
    excess, so the crossing is not read from it until it has been restarted.
    """
    return _rerun_count(rec) >= 1 or (_converged(rec) and bool(rec.get("warm_seeded")))


def _rerun_seed(rec, inner):
    """Where to restart *rec* from, or None if that would repeat it exactly.

    The optimizer is deterministic, so a restart from the start the point
    already had reproduces it to the last digit. From a nearer inner neighbour
    it is new information; failing that, an unconverged point is carried on from
    where it stopped.
    """
    if inner is not None:
        xs, nx = rec.get("x_start"), inner.get("nuisance_x")
        if (xs is None or nx is None or len(xs) != len(nx)
                or not np.allclose(xs, nx, rtol=0.0, atol=1e-9)):
            return inner
    if not _converged(rec) and rec.get("nuisance_x"):
        return rec
    return None


def _near(a, b, d_ref, rel=0.02):
    """Whether two opt-space values are the same point for planning purposes."""
    return abs(a - b) <= rel * max(abs(d_ref), 1e-9)


def _wald_target(p_opt, sign, is_log, lb, ub, se, range_factor, open_decades,
                 n_se=FIRST_PROBE_SE):
    """Opt-space value ``n_se`` Wald SEs out, capped, or None if undefined."""
    if se is not None:
        x = p_opt + sign * n_se * se
    elif is_log:
        x = p_opt + sign * np.log10(range_factor)
    elif p_opt > 0:
        x = p_opt * range_factor if sign > 0 else p_opt / range_factor
    else:
        return None
    if open_decades is not None:
        cap = float(open_decades)
        if is_log:
            x = max(x, p_opt - cap) if sign < 0 else min(x, p_opt + cap)
        elif p_opt > 0:
            lo, hi = p_opt * 10.0 ** -cap, p_opt * 10.0 ** cap
            x = max(x, lo) if sign < 0 else min(x, hi)
    return float(x)


def _clip(x, lb, ub, sign):
    bound = lb if sign < 0 else ub
    if bound is not None and np.isfinite(bound):
        x = max(x, bound) if sign < 0 else min(x, bound)
    return float(x)


def _slice_floor(screen, name, side_name):
    """Opt-space value of the slice's located crossing for one side, or None."""
    try:
        cr = screen["parameters"][name][side_name].get("crossing")
    except (KeyError, TypeError, AttributeError):
        return None
    if not cr or cr.get("x") is None:
        return None
    return float(cr["x"])


def _value_at_distance(p_opt, d, sign, is_log):
    """Opt-space value *d* from the optimum, in the walk's own measure."""
    if is_log or p_opt <= 0:
        return float(p_opt + sign * d)
    return float(p_opt * 10.0 ** (sign * d))


def _slice_rungs(p_opt, x_s, sign, is_log, lb, ub, open_decades, n_rungs,
                 ext_distance):
    """Log-spaced values from the slice crossing out to the cap, nearest first."""
    d_s = ext_distance(p_opt, x_s, is_log)
    if not np.isfinite(d_s) or d_s <= 0:
        return []
    outer = float(open_decades) if open_decades else 100.0 * d_s
    n = max(int(n_rungs), 2)
    if outer <= d_s:
        ds = [d_s]
    else:
        ds = [d_s * (outer / d_s) ** (k / (n - 1)) for k in range(n)]
    out = []
    for d in ds:
        x = _clip(_value_at_distance(p_opt, d, sign, is_log), lb, ub, sign)
        if x != p_opt and not any(abs(x - o) <= 1e-12 * max(abs(x), 1.0)
                                  for o in out):
            out.append(x)
    return out


def _geometric_ladder(p_opt, x1, n):
    """``n`` values from the optimum out to ``x1``, halving the distance inward."""
    return [p_opt + (0.5 ** k) * (x1 - p_opt) for k in range(n)]


def plan_next_points(n_free, *, param_names, completed, in_flight, res_x,
                     bounds, scales, wald_se, n_grid, range_factor,
                     open_decades, screen, threshold, anchor, max_extend,
                     extend_growth, bracket_rtol, max_step_decades=0.7,
                     attempted=None, first_probe="max"):
    """Up to *n_free* points to run next, best first.

    Returns a list of dicts ``{"i", "name", "x", "seed", "phase", "kind",
    "tier"}``: the parameter index, the fixed value in optimizer space, the
    record to warm-start from (or None for a cold start from the optimum), the
    phase number the job should carry, why it was chosen, and its priority
    (0 first probe, 1 stepping or re-running, 2 filling a bracket, 3 spare
    ladder rung, 4 and up extra probes inside an open bracket).

    *in_flight* is the list of job dicts currently running. Their positions
    count as occupied: nothing is planned on top of one, and a side with a first
    probe still running is waited on rather than given a second guess.
    """
    from pyantigen.engine.Optimize import (
        _param_bounds, _side_points, _extension_growth, _next_extension_value,
        _ext_distance, _bracket_is_tight, ProfileCheckpointKey,
    )

    if first_probe not in FIRST_PROBE_MODES:
        raise ValueError(f"first_probe {first_probe!r} is not one of "
                         f"{list(FIRST_PROBE_MODES)}")
    if n_free <= 0:
        return []
    attempted = attempted if attempted is not None else set()
    thr = float(threshold)
    cands = []

    def dn(r):
        return float(r["dnll"]) - anchor

    for i, name in enumerate(param_names):
        p_opt = float(res_x[i])
        is_log = scales[i] == "log10"
        lb, ub = _param_bounds(bounds, i)
        se = None
        if wald_se is not None:
            try:
                c = float(np.atleast_1d(wald_se)[i])
                se = c if np.isfinite(c) and c > 0 else None
            except (IndexError, TypeError, ValueError):
                se = None

        for sign, side_name in ((-1, "lower"), (1, "upper")):
            pts = [r for r in _side_points(completed, name, p_opt, sign)
                   if r.get("dnll") is not None and np.isfinite(r["dnll"])]
            fl = [float(j["x_fixed"]) for j in in_flight
                  if j.get("param_idx") == i
                  and ((float(j["x_fixed"]) < p_opt) if sign < 0
                       else (float(j["x_fixed"]) > p_opt))]
            occupied = [float(r["x_fixed"]) for r in pts] + fl

            def free(x, d_ref):
                return not any(_near(x, o, d_ref) for o in occupied)

            def probe_bracket(x_in, x_out, frac, seed, kind, d_ref):
                """The guess inside (x_in, x_out), then spare-core extras.

                A guess that lands on a point already measured means the
                crossing is located by that point, so nothing is added: the
                extras are for a side still being narrowed, not a licence to
                keep probing one that has an answer.
                """
                if any(_near(x_in + frac * (x_out - x_in),
                             float(r["x_fixed"]), d_ref) for r in pts):
                    return
                for j, fr in enumerate(_bracket_fracs(frac)):
                    x = x_in + fr * (x_out - x_in)
                    if free(x, d_ref) and \
                            ProfileCheckpointKey(x) not in completed.get(name, {}):
                        cands.append({"i": i, "name": name, "x": float(x),
                                      "seed": seed, "phase": 1, "kind": kind,
                                      "tier": 1 if j == 0 else 3 + j,
                                      "sign": sign})

            ceiling = n_grid + max_extend + _SIDE_CEILING_EXTRA
            if len(pts) + len(fl) >= ceiling:
                continue

            # ── nothing measured on this side yet ──────────────────────────
            if not pts:
                x_s = _slice_floor(screen, name, side_name)
                if first_probe != "max" and x_s is not None and x_s != p_opt:
                    rungs = _slice_rungs(
                        p_opt, x_s, sign, is_log, lb, ub, open_decades,
                        n_grid if first_probe == "slice_ladder" else 1,
                        _ext_distance)
                    if first_probe == "slice":
                        rungs = rungs[:1]
                    if rungs:
                        d_ref = abs(rungs[-1] - p_opt)
                        # The middle rung goes first: the slice crossing is a
                        # floor the profile cannot cross inside, so a probe
                        # there is nearly certain to read below the threshold
                        # and says little, while the middle of the span halves
                        # the uncertainty about where the crossing is.
                        lead = len(rungs) // 2
                        for k, x in enumerate(rungs):
                            if not free(x, d_ref):
                                continue
                            cands.append({
                                "i": i, "name": name, "x": float(x),
                                "seed": None, "phase": 1,
                                "kind": ("slice probe" if first_probe == "slice"
                                         else "slice ladder"),
                                "tier": 0 if (k == lead and not fl) else 3,
                                "sign": sign})
                        continue
                x_w = _wald_target(p_opt, sign, is_log, lb, ub, se,
                                   range_factor, open_decades)
                xs = [v for v in (x_w, x_s) if v is not None]
                if not xs:
                    continue
                x1 = max(xs) if sign > 0 else min(xs)
                x1 = _clip(x1, lb, ub, sign)
                if x1 == p_opt:
                    continue
                d1 = abs(x1 - p_opt)
                for k, x in enumerate(_geometric_ladder(p_opt, x1, n_grid)):
                    if not free(x, d1):
                        continue
                    first = (k == 0 and not fl)
                    cands.append({"i": i, "name": name, "x": float(x),
                                  "seed": None, "phase": 1,
                                  "kind": "first probe" if k == 0 else "ladder",
                                  "tier": 0 if first else 3, "sign": sign})
                continue

            below = [r for r in pts if dn(r) <= thr]
            above = [r for r in pts if dn(r) > thr]
            outer_pt = pts[-1]

            if above:
                # The innermost point past the threshold is the outer end of the
                # bracket -- but only if it is a real measurement. A cold or
                # unconverged value is an upper bound with an unknown excess
                # (observed: 0.05, 0.3 and 3.8 nats), and a warm restart from a
                # nearer neighbour only ever lowers it, so the crossing may lie
                # beyond it and every probe planned inside the bracket would
                # read "below". Such a point is provisional: it is restarted
                # warm at once, alongside the probes, instead of last.
                tgt = above[0]
                x_t = tgt["x_fixed"]
                d_t = _ext_distance(p_opt, x_t, is_log)

                def inner_of(t):
                    d_k = _ext_distance(p_opt, t["x_fixed"], is_log)
                    c = [r for r in below
                         if _ext_distance(p_opt, r["x_fixed"], is_log) < d_k]
                    return c[-1] if c else None

                inner = inner_of(tgt)
                x_in = inner["x_fixed"] if inner else p_opt

                # Only points before the first firm one can move the crossing;
                # anything past a firm end is in the tail.
                prov = []
                for r in above:
                    if _is_firm(r):
                        break
                    prov.append(r)
                prov = prov[:1 + _SPEC_RERUN]
                for rank, t in enumerate(prov):
                    key = (name, ProfileCheckpointKey(t["x_fixed"]))
                    d_r = _ext_distance(p_opt, t["x_fixed"], is_log)
                    if key in attempted or _rerun_count(t) >= 1:
                        continue
                    if any(_near(f, t["x_fixed"], d_r) for f in fl):
                        continue
                    seed = _rerun_seed(t, inner_of(t))
                    if seed is None:
                        continue
                    cands.append({"i": i, "name": name, "x": float(t["x_fixed"]),
                                  "seed": seed, "phase": 3, "kind": "rerun",
                                  "tier": 1 if rank == 0 else 3 + rank,
                                  "sign": sign, "attempt_key": key})

                tight = _bracket_is_tight(x_in, x_t, p_opt, is_log, bracket_rtol)
                if not tight:
                    d_in_nll = dn(inner) if inner else 0.0
                    r_in, r_out = np.sqrt(max(d_in_nll, 0.0)), np.sqrt(dn(tgt))
                    frac = ((np.sqrt(thr) - r_in) / (r_out - r_in)
                            if r_out > r_in + 1e-12 else 0.5)
                    frac = float(min(max(frac, 0.05), 0.95))
                    probe_bracket(x_in, x_t, frac, inner,
                                  "bracket" if _is_firm(tgt) else "narrow",
                                  abs(x_t - p_opt))
                    continue
                if _is_firm(tgt) and len(pts) + len(fl) < n_grid:
                    # Fill the curve's shape: halve the widest relative gap
                    # between the optimum and the crossing.
                    ds = sorted({0.0} | {
                        _ext_distance(p_opt, r["x_fixed"], is_log)
                        for r in pts if _ext_distance(
                            p_opt, r["x_fixed"], is_log) <= d_t})
                    gaps = []
                    for g_a, g_b in zip(ds[:-1], ds[1:]):
                        gaps.append((g_b / g_a if g_a > 0 else float("inf"),
                                     g_a, g_b))
                    if gaps:
                        _, g_a, g_b = max(gaps, key=lambda g: g[0])
                        d_new = g_b / 2.0 if g_a == 0 else float(np.sqrt(g_a * g_b))
                        x = (p_opt + sign * d_new if (is_log or p_opt <= 0)
                             else p_opt * 10.0 ** (sign * d_new))
                        ref = min((r for r in pts
                                   if _ext_distance(p_opt, r["x_fixed"],
                                                    is_log) <= d_new),
                                  key=lambda r: -_ext_distance(
                                      p_opt, r["x_fixed"], is_log),
                                  default=None)
                        if free(x, abs(x_t - p_opt)):
                            cands.append({"i": i, "name": name, "x": float(x),
                                          "seed": ref, "phase": 1,
                                          "kind": "fill", "tier": 2,
                                          "sign": sign})
                continue


            # ── below the threshold everywhere so far: step outward ────────
            # One rung per round would hold a single worker for as long as the
            # side stays open, and an open side says nothing about its tail
            # (a nonlinear profile can flatten and then climb), so it is not
            # assumed to be flat. Instead the next rungs go out together:
            # up to ``_SPEC_MAX_EXT`` beyond the first, each stepping on from
            # the farthest rung already running or planned, so a relaunch or a
            # rung landing continues the ladder instead of restarting it.
            far_fl = [f for f in fl
                      if abs(f - p_opt) > abs(outer_pt["x_fixed"] - p_opt)]
            need = min(1 + _SPEC_MAX_EXT - len(far_fl),
                       ceiling - len(pts) - len(fl))
            if need <= 0:
                continue
            d_out = _ext_distance(p_opt, outer_pt["x_fixed"], is_log)
            growth = float(extend_growth)
            if d_out > 0 and max_step_decades:
                growth = min(growth, 1.0 + float(max_step_decades) / d_out)
            growth = max(growth, 1.05)
            g = _extension_growth(pts, p_opt, is_log, thr, anchor, growth)
            g = float(min(max(g, 1.05), growth))
            x_ref = (max(far_fl) if sign > 0 else min(far_fl)) if far_fl \
                else outer_pt["x_fixed"]
            for j in range(need):
                d_ref = _ext_distance(p_opt, x_ref, is_log)
                g_j = g
                if d_ref > 0 and max_step_decades:
                    g_j = min(g, max(1.0 + float(max_step_decades) / d_ref, 1.05))
                x = _next_extension_value(p_opt, x_ref, lb, ub, sign, is_log, g_j)
                if x is None:
                    break
                x_ref = x
                if free(x, abs(outer_pt["x_fixed"] - p_opt)) and \
                        ProfileCheckpointKey(x) not in completed.get(name, {}):
                    cands.append({"i": i, "name": name, "x": float(x),
                                  "seed": outer_pt, "phase": 1, "kind": "extend",
                                  "tier": 1 if (j == 0 and not far_fl)
                                  else 3 + j + (1 if far_fl else 0),
                                  "sign": sign})

    cands.sort(key=lambda c: (c["tier"], c["i"], c["sign"]))
    return cands[:n_free]

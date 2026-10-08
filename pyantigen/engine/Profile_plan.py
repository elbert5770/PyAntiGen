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
* a side that has crossed only on points that did not converge is narrowed, not
  repeated: an unconverged dNLL is an upper bound, so a crossing read from one
  is not a crossing, and re-running the same far point from the same start
  reproduces it exactly. The probe goes between the innermost converged point
  below the threshold and the innermost unconverged one above it (sqrt(dNLL)
  interpolation), warm from the former, and waits while anything is still
  running inside that bracket. Only once the bracket is tight is the outer
  point itself re-run;
* a bracketed side is narrowed by sqrt(dNLL) interpolation until the bracket is
  tight, then filled in toward ``n_grid`` points so the curve has a shape.

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


def _converged(rec):
    return bool(rec.get("converged")) and not rec.get("interrupted")


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
    (0 first probe, 1 stepping or re-running, 2 filling a bracket, 3 spare).

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
            above_c = [r for r in pts if dn(r) > thr and _converged(r)]
            above_t = [r for r in pts if dn(r) > thr and not _converged(r)]
            outer_pt = pts[-1]

            if above_c:
                out = above_c[0]
                d_out = _ext_distance(p_opt, out["x_fixed"], is_log)
                inner_c = [r for r in below
                           if _ext_distance(p_opt, r["x_fixed"], is_log) < d_out]
                inner = inner_c[-1] if inner_c else None
                x_in = inner["x_fixed"] if inner else p_opt
                d_in_nll = dn(inner) if inner else 0.0
                tight = _bracket_is_tight(x_in, out["x_fixed"], p_opt, is_log,
                                          bracket_rtol)
                if not tight:
                    r_in, r_out = np.sqrt(max(d_in_nll, 0.0)), np.sqrt(dn(out))
                    frac = ((np.sqrt(thr) - r_in) / (r_out - r_in)
                            if r_out > r_in + 1e-12 else 0.5)
                    frac = float(min(max(frac, 0.05), 0.95))
                    x = x_in + frac * (out["x_fixed"] - x_in)
                    if free(x, abs(out["x_fixed"] - p_opt)) and \
                            ProfileCheckpointKey(x) not in completed.get(name, {}):
                        cands.append({"i": i, "name": name, "x": float(x),
                                      "seed": inner, "phase": 1,
                                      "kind": "bracket", "tier": 1,
                                      "sign": sign})
                    continue
                if len(pts) + len(fl) < n_grid:
                    # Fill the curve's shape: halve the widest relative gap
                    # between the optimum and the crossing.
                    ds = sorted({0.0} | {
                        _ext_distance(p_opt, r["x_fixed"], is_log)
                        for r in pts if _ext_distance(
                            p_opt, r["x_fixed"], is_log) <= d_out})
                    gaps = []
                    for a, b in zip(ds[:-1], ds[1:]):
                        gaps.append((b / a if a > 0 else float("inf"), a, b))
                    if gaps:
                        _, a, b = max(gaps, key=lambda g: g[0])
                        d_new = b / 2.0 if a == 0 else float(np.sqrt(a * b))
                        x = (p_opt + sign * d_new if (is_log or p_opt <= 0)
                             else p_opt * 10.0 ** (sign * d_new))
                        ref = min((r for r in pts
                                   if _ext_distance(p_opt, r["x_fixed"],
                                                    is_log) <= d_new),
                                  key=lambda r: -_ext_distance(
                                      p_opt, r["x_fixed"], is_log),
                                  default=None)
                        if free(x, abs(out["x_fixed"] - p_opt)):
                            cands.append({"i": i, "name": name, "x": float(x),
                                          "seed": ref, "phase": 1,
                                          "kind": "fill", "tier": 2,
                                          "sign": sign})
                continue

            if above_t:
                # Crossed only on points that did not converge. Each is an
                # upper bound, so the crossing lies somewhere inside the
                # bracket (nearest converged-below point, innermost of them).
                tgt = above_t[0]
                x_t = tgt["x_fixed"]
                d_t = _ext_distance(p_opt, x_t, is_log)
                inner_c = [r for r in below
                           if _ext_distance(p_opt, r["x_fixed"], is_log) < d_t]
                inner = inner_c[-1] if inner_c else None
                x_in = inner["x_fixed"] if inner else p_opt
                # Anything still running between the optimum and the failed
                # point is about to become a better inner neighbour, or to
                # replace the point we would plan. Planning now would seed from
                # whatever has landed so far -- often nothing -- and a cold
                # restart of a deterministic optimizer from the same start
                # reproduces the failed point to the last digit.
                if any(_ext_distance(p_opt, f, is_log) < d_t for f in fl):
                    continue
                tight = _bracket_is_tight(x_in, x_t, p_opt, is_log,
                                          bracket_rtol)
                if not tight:
                    # Narrow the bracket toward the threshold instead of
                    # repeating the far point: a probe between the inner
                    # neighbour and the failed point is a short, warm
                    # continuation and so can converge where the long one
                    # could not. dnll at the failed point is too high, which
                    # puts the probe on the near side of the crossing; the
                    # next round narrows from there.
                    d_in_nll = dn(inner) if inner else 0.0
                    r_in, r_out = np.sqrt(max(d_in_nll, 0.0)), np.sqrt(dn(tgt))
                    frac = ((np.sqrt(thr) - r_in) / (r_out - r_in)
                            if r_out > r_in + 1e-12 else 0.5)
                    frac = float(min(max(frac, 0.05), 0.95))
                    x = x_in + frac * (x_t - x_in)
                    if free(x, abs(x_t - p_opt)) and \
                            ProfileCheckpointKey(x) not in completed.get(name, {}):
                        cands.append({"i": i, "name": name, "x": float(x),
                                      "seed": inner, "phase": 1,
                                      "kind": "narrow", "tier": 1,
                                      "sign": sign})
                    continue
                # The bracket is tight and the outer end still has not
                # converged: give it one more go, warm from the inner point.
                # With no inner point and a cold original this would be the
                # same computation again, so it is not planned.
                key = (name, ProfileCheckpointKey(x_t))
                if key not in attempted and (
                        inner is not None or tgt.get("warm_seeded")):
                    cands.append({"i": i, "name": name, "x": float(x_t),
                                  "seed": inner, "phase": 3, "kind": "rerun",
                                  "tier": 1, "sign": sign, "attempt_key": key})
                continue

            # ── below the threshold everywhere so far: step outward ────────
            far_fl = [f for f in fl
                      if abs(f - p_opt) > abs(outer_pt["x_fixed"] - p_opt)]
            if far_fl:
                continue
            d_out = _ext_distance(p_opt, outer_pt["x_fixed"], is_log)
            growth = float(extend_growth)
            if d_out > 0 and max_step_decades:
                growth = min(growth, 1.0 + float(max_step_decades) / d_out)
            growth = max(growth, 1.05)
            g = _extension_growth(pts, p_opt, is_log, thr, anchor, growth)
            g = float(min(max(g, 1.05), growth))
            x = _next_extension_value(p_opt, outer_pt["x_fixed"], lb, ub,
                                      sign, is_log, g)
            if x is None:
                continue
            if free(x, abs(outer_pt["x_fixed"] - p_opt)) and \
                    ProfileCheckpointKey(x) not in completed.get(name, {}):
                cands.append({"i": i, "name": name, "x": float(x),
                              "seed": outer_pt, "phase": 1, "kind": "extend",
                              "tier": 1, "sign": sign})

    cands.sort(key=lambda c: (c["tier"], c["i"], c["sign"]))
    return cands[:n_free]

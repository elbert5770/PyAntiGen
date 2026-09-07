"""The fast profile: one capped profile point per side, at the slice crossing.

A full profile finds where dNLL crosses 1.9207 for every parameter, and on a
QSP model that is days. Most of that time is spent on the parameters that
turn out to be poorly determined, and for those the answer is already visible
much closer in. This pass asks a cheaper question at a single, well-chosen
point and reports what can be said from it.

The point
---------
The slice screen (:mod:`Engine.Identifiability`) has already walked each
side with the nuisance parameters held at their fitted values, so the ladder
point where the *slice* first sits above the threshold is known, together
with the slice's value there. That point, not an interpolated crossing, is
where the nuisance minimization is run: its slice value is a measured number,
so how much of it the other parameters absorb is measured too, and nothing
rests on an interpolation whose bias would otherwise leak into every verdict.
The profile crossing lies at or beyond the slice crossing, because the slice
is an upper bound on the profile. At that one point a nuisance minimization
is run, in a few short rounds, and the amount by which dNLL falls says how
much the other parameters can compensate:

* It stays at the threshold: nothing compensates. The profile and the slice
  coincide here, the interval closes at the slice crossing, and the parameter
  is as well determined as the slice suggests.
* It falls to near zero: the whole displacement is absorbed at no cost in
  likelihood. There is a direction through parameter space along which the
  data say nothing over at least this range, and the profile crossing, if it
  exists at all, is far out.
* Somewhere in between: the interval is wider than the slice's by a factor
  that can be estimated, and it is likely, not certainly, closed.

What is proof and what is not
-----------------------------
Every evaluation of the nuisance objective is an upper bound on the profile.
So a value *below* the threshold is a proof that the profile there is below
the threshold too, and a small value is a proof that the profile is at least
that small. Those are the one-directional statements this pass can make with
certainty, and they all widen intervals.

A value that *stays high* proves nothing on its own: Nelder-Mead in fourteen
nuisance dimensions can sit far above the minimum for a long time. "Proven
true" below therefore rests on the optimizer reporting convergence, which is
the same standard the full profile applies when it declares a crossing, and
no stronger. A parameter whose descent stalled without converging is reported
as likely, never proven.

Widths are extrapolated under a locally quadratic profile: if the profile
sits at dNLL d at the slice crossing, a distance s from the optimum, its own
crossing is at least s * sqrt(threshold / d) out. Because d is an upper bound
the estimate is a lower bound on the width, not a guess at it.

Verdicts
--------
Each side of each parameter is judged on the statement "the 95% interval is
closed on this side":

    proven false   the screen proved the slice, and so the profile, still
                   below the threshold decades out. The side is open.
    unlikely true  dNLL at the slice crossing fell to within a small fraction
                   of zero. The nuisance set compensates almost fully; the
                   extrapolated crossing is several times further out and
                   usually beyond where the screen walked.
    likely true    dNLL fell but stayed clear of zero, or stayed above the
                   threshold without the optimizer converging.
    proven true    the nuisance minimization converged above the threshold at
                   the slice crossing: the interval closes at or inside it.
    no verdict     the side could not be screened, the point could not be
                   evaluated, or a better optimum than the fit was found.

Every point computed here is a legitimate profile record and is written to
the same checkpoint the full profile reads, so a full profile run afterwards
starts with these points in hand and nothing is spent twice.
"""

import json
import os
import time
from datetime import datetime

import numpy as np

from Engine.Evaluator import FAILURE_VALUE
from Engine.Identifiability import (
    MIN_REACH_DECADES,
    SPAN_DECADES,
    THRESHOLD,
    _INCONCLUSIVE_STATES,
    decades_from,
    load_screen,
    print_screen_report,
    run_slice_screen,
    save_screen,
)

REPORT_FILENAME = "fast_profile.json"

# Evaluations per round. Nelder-Mead needs n+1 to build its simplex, so a
# round has to be comfortably more than that to say anything; at ~21 s per
# evaluation on the SILK spec a round of 150 is about 50 minutes.
DEFAULT_ROUND_EVALS = 150
DEFAULT_ROUNDS = 3

# dNLL at or below this fraction of the threshold counts as "near zero".
# 0.1 puts the extrapolated crossing at least sqrt(10) ~ 3.2 times further out
# than the slice crossing.
DEFAULT_NEAR_ZERO_FRAC = 0.10

# A round that lowers dNLL by less than this fraction of the threshold has
# stalled. Used only to stop spending rounds on a point that is going nowhere;
# it never upgrades a verdict.
STALL_FRAC = 0.05

# dNLL below this is a better optimum than the fit, not a flat direction.
NEGATIVE_TOL = 1e-3

VERDICTS = ("proven true", "likely true", "unlikely true", "proven false",
            "no verdict")


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def slice_crossing(points, p_opt, threshold):
    """Where the slice crosses the threshold, from the screen's ladder.

    Returns ``(x_cross, x_inner, x_outer)`` in optimizer space: the linear
    interpolation of the crossing, and the two ladder points it lies between.
    The optimum itself, at dNLL 0, is the inner point when the first ladder
    point is already above the threshold. None if the slice never crossed.

    Linear interpolation is chosen for its bias. A convex curve lies below its
    chord, so the chord reaches the threshold first and the interpolated
    crossing sits *inside* the true one. Every width this pass reports is a
    lower bound, and placing the point inside keeps it one.
    """
    prev_x, prev_d = float(p_opt), 0.0
    for p in points:
        d = p.get("dnll")
        nll = p.get("nll")
        if d is None or not np.isfinite(d) or nll is None or nll >= FAILURE_VALUE:
            continue
        x = float(p["x"])
        if d > threshold:
            if d == prev_d:
                return x, prev_x, x
            t = (threshold - prev_d) / (d - prev_d)
            return prev_x + t * (x - prev_x), prev_x, x
        prev_x, prev_d = x, float(d)
    return None


def to_linear(x, is_log):
    return float(10.0 ** x) if is_log else float(x)


def local_compensation(wald_cov, i):
    """Marginal over conditional standard error for parameter *i*, or None.

    The conditional SE, 1/sqrt(H_ii), is the slice's own curvature; the
    marginal SE, sqrt((H^-1)_ii), is the Wald interval's. Their ratio is the
    square root of the variance inflation factor: how much wider the interval
    is once the other parameters are free to move, in the quadratic
    approximation. It is the local, linearised version of the question this
    pass asks at the slice crossing, and it comes free with the Hessian.
    """
    if wald_cov is None:
        return None
    try:
        cov = np.asarray(wald_cov, dtype=float)
        if cov.ndim != 2 or cov.shape[0] != cov.shape[1] or i >= cov.shape[0]:
            return None
        marginal = float(np.sqrt(cov[i, i]))
        hess = np.linalg.inv(cov)
        conditional = float(1.0 / np.sqrt(hess[i, i]))
    except (np.linalg.LinAlgError, ValueError, FloatingPointError):
        return None
    if not (np.isfinite(marginal) and np.isfinite(conditional)) or conditional <= 0:
        return None
    return marginal / conditional


def width_factor(dnll, threshold):
    """Lower bound on (profile crossing distance) / (slice crossing distance).

    Under a locally quadratic profile and with *dnll* an upper bound on the
    profile at the slice crossing. Infinite when dnll is not positive: the
    profile is flat to within resolution.
    """
    if dnll is None or not np.isfinite(dnll):
        return None
    if dnll <= 0:
        return float("inf")
    return float(np.sqrt(threshold / dnll))


# ---------------------------------------------------------------------------
# Verdicts
# ---------------------------------------------------------------------------

def classify(side, threshold, near_zero_frac=DEFAULT_NEAR_ZERO_FRAC):
    """Verdict and one-line reason for one side's record. Pure."""
    state = side.get("screen_state")
    if state == "open":
        reach = side.get("reach_decades")
        how_far = (f"{reach:.2g} decade(s) out" if reach is not None
                   else f"out at {_g(side.get('reach_linear'))}")
        return ("proven false",
                f"the slice stays below {threshold:.4g} {how_far}, so the "
                f"profile does too; the data do not exclude that value")
    if state in _INCONCLUSIVE_STATES or state == "empty":
        return ("no verdict", "the screen could not walk this side")
    if side.get("x_point") is None:
        return ("no verdict", "the slice never crossed within the walk")
    if side.get("status") != "ok" or side.get("dnll") is None:
        return ("no verdict",
                f"the point just outside the slice crossing could not be "
                f"evaluated ({side.get('status')})")

    d = float(side["dnll"])
    conv = bool(side.get("converged"))
    nfev = side.get("nfev_total")
    factor = width_factor(d, threshold)
    absorbed = side.get("absorbed")
    abs_txt = (f"{100 * absorbed:.0f}% of the slice's dNLL absorbed"
               if absorbed is not None else "slice dNLL unknown")
    at = f"{side.get('x_point_linear'):.4g}"
    if d < -NEGATIVE_TOL:
        return ("no verdict",
                f"dNLL {d:.4g} at {at} is below zero: a better optimum than "
                f"the fit exists there, so the anchor is wrong and nothing "
                f"here can be read until the fit is redone from it")
    if d <= near_zero_frac * threshold:
        where = (f"; the crossing is at least {factor:.3g}x further out than "
                 f"{at}" if np.isfinite(factor) else
                 "; flat to within resolution")
        beyond = (", beyond where the screen walked"
                  if side.get("est_beyond_reach") else "")
        return ("unlikely true",
                f"the other parameters absorb the displacement almost "
                f"entirely at {at} ({abs_txt}, dNLL {d:.3g} of "
                f"{threshold:.4g}){where}{beyond}")
    if d < threshold:
        tail = ("converged" if conv else
                f"not converged after {nfev} evaluation(s), so it could fall "
                f"further")
        return ("likely true",
                f"the profile is below the threshold at {at} ({abs_txt}, "
                f"dNLL {d:.3g}), so the crossing is at least {factor:.3g}x "
                f"further out; {tail}")
    if conv:
        return ("proven true",
                f"the nuisance minimization converged above the threshold at "
                f"{at} ({abs_txt}, dNLL {d:.3g}), so the interval closes at or "
                f"inside it")
    how = "stalled" if side.get("stalled") else "still descending"
    return ("likely true",
            f"dNLL {d:.3g} at {at} is still above the threshold after {nfev} "
            f"evaluation(s) without converging ({how}, {abs_txt}); a flat "
            f"direction has not been found, but the search is not finished")


# ---------------------------------------------------------------------------
# The pass
# ---------------------------------------------------------------------------

def _capped_kwargs(optimizer_kwargs, n_evals):
    kw = dict(optimizer_kwargs or {})
    opts = dict(kw.get("options") or {})
    opts["maxfev"] = int(n_evals)
    opts["maxiter"] = int(n_evals)
    kw["options"] = opts
    return kw


def run_fast_profile(batch, nll_batch, res_x, nll_at_optimum, param_names,
                     bounds, scales, method="Nelder-Mead", optimizer_kwargs=None,
                     wald_se=None, wald_cov=None, checkpoint=None, ckpt_dir=None,
                     threshold=THRESHOLD, round_evals=DEFAULT_ROUND_EVALS,
                     n_rounds=DEFAULT_ROUNDS,
                     near_zero_frac=DEFAULT_NEAR_ZERO_FRAC,
                     span_decades=SPAN_DECADES,
                     min_reach_decades=MIN_REACH_DECADES, verbose=True):
    """Screen, then one capped profile point per crossed side, in rounds.

    *batch* has the profile_batch signature: ``batch(jobs, on_result, label)``.
    *nll_batch* evaluates a list of full parameter vectors, for the screen.
    Returns the report dict; :func:`print_fast_profile_report` renders it and
    :func:`fast_profile_summary` shrinks it for the results snapshot.
    """
    from Engine.Optimize import _cold_simplex, _param_bounds

    res_x = np.asarray(res_x, dtype=float)
    n = len(param_names)
    scales = list(scales) if scales is not None else ["lin"] * n
    t_start = time.time()

    # ── The screen, reused when this fit already has one ──────────────────
    screen = load_screen(ckpt_dir, param_names, res_x, threshold,
                         span_decades, min_reach_decades)
    if screen is None:
        screen = run_slice_screen(
            nll_batch, res_x, nll_at_optimum, param_names, bounds,
            scales=scales, wald_se=wald_se, threshold=threshold,
            span_decades=span_decades, min_reach_decades=min_reach_decades,
            verbose=verbose)
        save_screen(screen, ckpt_dir)
    elif verbose:
        print(f"\n[fast profile] reusing the slice screen already run for this "
              f"fit ({screen.get('n_evaluations', 0)} evaluation(s)).",
              flush=True)
    if verbose:
        print_screen_report(screen)

    # ── One side record per parameter per side ────────────────────────────
    sides = {}
    jobs = []
    for i, name in enumerate(param_names):
        is_log = scales[i] == "log10"
        p_opt = float(res_x[i])
        comp = local_compensation(wald_cov, i)
        for sign, side_name in ((-1, "lower"), (1, "upper")):
            rec = screen.get("parameters", {}).get(name, {}).get(side_name, {})
            side = {
                "param_idx": i, "name": name, "side": side_name, "sign": sign,
                "is_log": is_log, "p_opt": p_opt, "p_opt_linear": to_linear(p_opt, is_log),
                "screen_state": rec.get("state", "empty"),
                "reach_decades": rec.get("reach_decades"),
                "reach_linear": rec.get("reach"),
                "certified_inner_linear": rec.get("inner_bracket"),
                "local_compensation": comp,
                # The interpolated slice crossing, for information; the point
                # actually profiled is the ladder point just outside it.
                "x_slice_linear": None,
                "x_point": None, "x_point_linear": None, "point_decades": None,
                "slice_dnll_at_point": None, "absorbed": None,
                "dnll": None, "status": None, "converged": None,
                "nfev_total": None, "rounds": [], "stalled": False,
                "width_factor": None, "est_crossing_linear": None,
                "est_decades": None, "est_beyond_reach": None,
            }
            sides[(i, side_name)] = side
            if rec.get("state") != "crossed":
                continue
            cross = slice_crossing(rec.get("points", []), p_opt, threshold)
            if cross is None:
                continue
            x_cross, _x_inner, x_point = (float(v) for v in cross)
            side["x_slice_linear"] = to_linear(x_cross, is_log)
            side["x_point"] = x_point
            side["x_point_linear"] = to_linear(x_point, is_log)
            d = decades_from(p_opt, x_point, is_log)
            side["point_decades"] = float(d) if np.isfinite(d) else None
            for p in rec.get("points", []):
                if abs(float(p["x"]) - x_point) <= 1e-12 * max(1.0, abs(x_point)):
                    side["slice_dnll_at_point"] = float(p["dnll"])
                    break
            x_slice = x_point

            nb = None
            if bounds is not None:
                nb = [list(b) if b is not None else None
                      for b in (list(bounds[:i]) + list(bounds[i + 1:]))]
            start = np.delete(res_x, i)
            se_nuis = None
            if wald_se is not None:
                try:
                    se_arr = np.atleast_1d(np.asarray(wald_se, dtype=float))
                    if se_arr.size == n:
                        se_nuis = np.delete(se_arr, i)
                except (TypeError, ValueError):
                    se_nuis = None
            job = {
                "param_idx": i,
                "param_name": name,
                "x_fixed": x_slice,
                "x_fixed_linear": side["x_point_linear"],
                "x_start": start.tolist(),
                "warm_seeded": False,
                "nuisance_bounds": nb,
                "method": method,
                "optimizer_kwargs": _capped_kwargs(optimizer_kwargs, round_evals),
                # A cold grid point as far as the full profile is concerned:
                # its warm pass owes this point a sweep like any other.
                "phase": 1,
                "direction": 0,
                "x_step": abs(x_slice - p_opt),
                "fast_profile": True,
                "_side": side_name,
            }
            sim = _cold_simplex(start, se_nuis, bounds=nb)
            if sim is not None:
                job["initial_simplex"] = sim
            jobs.append(job)

    if verbose:
        n_crossed = sum(1 for s in sides.values() if s["screen_state"] == "crossed")
        print(f"\n[fast profile] {len(jobs)} side(s) crossed the slice threshold "
              f"and get one capped profile point each at the slice crossing, "
              f"up to {n_rounds} round(s) of {round_evals} evaluation(s); "
              f"{2 * n - n_crossed} side(s) were settled or blocked by the "
              f"screen.", flush=True)

    # ── Rounds ────────────────────────────────────────────────────────────
    def record(res):
        i, side_name = int(res["param_idx"]), res.get("_side")
        side = sides.get((i, side_name))
        if side is None:
            return
        nll = res.get("nll")
        ok = (res.get("status") == "ok" and nll is not None
              and np.isfinite(nll) and nll < FAILURE_VALUE)
        d = float(nll) - float(nll_at_optimum) if ok else None
        prev = side["dnll"]
        side["status"] = res.get("status")
        side["rounds"].append({
            "dnll": d, "nfev_total": res.get("nfev_total"),
            "converged": bool(res.get("converged")),
            "status": res.get("status"),
        })
        if ok and (prev is None or d <= prev):
            side["dnll"] = d
            side["converged"] = bool(res.get("converged"))
            side["nfev_total"] = res.get("nfev_total")
            side["_last"] = res
            if checkpoint is not None:
                keep = dict(res)
                keep.pop("_side", None)
                keep["dnll"] = d
                keep.setdefault("warm_refined", 0)
                checkpoint.append(keep)
        elif ok:
            # Not an improvement, but the spend and the simplex still move.
            side["_last"] = dict(side.get("_last") or res, **{
                k: res[k] for k in ("nfev_total", "nit_total", "nm_simplex",
                                     "converged", "nuisance_x")
                if k in res})
            side["nfev_total"] = res.get("nfev_total")
            side["converged"] = bool(res.get("converged"))
        if prev is not None and d is not None:
            side["stalled"] = (prev - d) < STALL_FRAC * threshold

    deadline_hit = None
    active = list(jobs)
    for r in range(int(n_rounds)):
        if not active:
            break
        label = f"fast-profile r{r + 1}"
        if verbose and r:
            print(f"\n[fast profile] round {r + 1}: {len(active)} point(s) "
                  f"continue.", flush=True)
        try:
            batch(active, on_result=record, label=label)
        except Exception as exc:  # DeadlineReached, or a pool failure
            if type(exc).__name__ != "DeadlineReached":
                raise
            deadline_hit = str(exc)
            if verbose:
                print(f"[fast profile] {exc}; classifying what landed.",
                      flush=True)
            break

        nxt = []
        for job in active:
            side = sides[(job["param_idx"], job["_side"])]
            d = side["dnll"]
            last = side.get("_last")
            if last is None or side["status"] != "ok" or d is None:
                continue                         # failed: no more rounds
            if d < -NEGATIVE_TOL or d <= near_zero_frac * threshold:
                continue                         # verdict reached
            if side["converged"]:
                continue
            if side["stalled"] and d >= threshold:
                continue                         # going nowhere above the line
            cont = dict(job)
            cont["x_start"] = list(last.get("nuisance_x") or job["x_start"])
            cont["optimizer_kwargs"] = _capped_kwargs(optimizer_kwargs,
                                                      round_evals * (r + 2))
            cont["nfev_used"] = int(last.get("nfev_total") or 0)
            cont["nit_used"] = int(last.get("nit_total") or 0)
            cont["nll_so_far"] = last.get("nll")
            cont["resumed"] = True
            sim = last.get("nm_simplex")
            if sim is not None:
                cont["initial_simplex"] = sim
            nxt.append(cont)
        active = nxt

    # ── Extrapolation and verdicts ────────────────────────────────────────
    counts = {v: 0 for v in VERDICTS}
    for side in sides.values():
        side.pop("_last", None)
        d = side["dnll"]
        if d is not None and side["x_point"] is not None:
            s_here = side.get("slice_dnll_at_point")
            if s_here is not None and s_here > 0:
                side["absorbed"] = float(np.clip(1.0 - d / s_here, 0.0, 1.0))
            f = width_factor(d, threshold)
            side["width_factor"] = f
            if f is not None and np.isfinite(f):
                x_est = side["p_opt"] + side["sign"] * f * abs(side["x_point"] - side["p_opt"])
                side["est_crossing_linear"] = to_linear(x_est, side["is_log"])
                dec = decades_from(side["p_opt"], x_est, side["is_log"])
                side["est_decades"] = float(dec) if np.isfinite(dec) else None
                reach = side.get("reach_decades")
                if reach is not None and np.isfinite(dec):
                    side["est_beyond_reach"] = bool(dec > reach)
                elif side.get("reach_linear") is not None:
                    # No decades for a linear parameter through zero; compare
                    # distances in linear units instead.
                    p_lin = side["p_opt_linear"]
                    side["est_beyond_reach"] = bool(
                        abs(side["est_crossing_linear"] - p_lin)
                        > abs(float(side["reach_linear"]) - p_lin))
                else:
                    side["est_beyond_reach"] = None
            elif f is not None:
                side["est_beyond_reach"] = True
        verdict, reason = classify(side, threshold, near_zero_frac)
        side["verdict"] = verdict
        side["reason"] = reason
        counts[verdict] += 1

    report = {
        "threshold": float(threshold),
        "anchor": float(nll_at_optimum),
        "res_x": [float(v) for v in res_x],
        "param_names": list(param_names),
        "round_evals": int(round_evals),
        "n_rounds": int(n_rounds),
        "near_zero_frac": float(near_zero_frac),
        "n_points": len(jobs),
        "n_evaluations": int(sum(
            (s.get("nfev_total") or 0) for s in sides.values())),
        "wall_s": time.time() - t_start,
        "deadline": deadline_hit,
        "counts": counts,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "parameters": {},
    }
    for (i, side_name), side in sides.items():
        report["parameters"].setdefault(side["name"], {})[side_name] = {
            k: v for k, v in side.items() if k not in ("param_idx", "name",
                                                       "side", "sign")}
    return report


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def save_report(report, ckpt_dir):
    if not ckpt_dir:
        return None
    path = os.path.join(ckpt_dir, REPORT_FILENAME)
    tmp = f"{path}.{os.getpid()}.tmp"
    try:
        os.makedirs(ckpt_dir, exist_ok=True)
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(_jsonable(report), fh, indent=1)
        os.replace(tmp, path)
    except OSError:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        return None
    return path


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return v if np.isfinite(v) else None
    return obj


def _g(v, prec=4):
    if v is None:
        return "-"
    try:
        v = float(v)
    except (TypeError, ValueError):
        return str(v)
    if not np.isfinite(v):
        return "inf" if v > 0 else "-inf"
    return f"{v:.{prec}g}"


def format_summary_table(report):
    """The end-of-run table, as text."""
    thr = report["threshold"]
    rows = []
    for name, sides in report["parameters"].items():
        for side_name in ("lower", "upper"):
            s = sides.get(side_name)
            if s is None:
                continue
            factor = s.get("width_factor")
            factor_txt = ("flat" if factor is not None and not np.isfinite(factor)
                          else (f"x{factor:.2g}" if factor is not None else "-"))
            absorbed = s.get("absorbed")
            rows.append((
                name, side_name,
                _g(s.get("x_point_linear")),
                _g(s.get("slice_dnll_at_point"), 3),
                _g(s.get("dnll"), 3),
                (f"{100 * absorbed:.0f}%" if absorbed is not None else "-"),
                str(s.get("nfev_total") if s.get("nfev_total") is not None else "-"),
                ("yes" if s.get("converged") else
                 ("no" if s.get("converged") is not None else "-")),
                factor_txt,
                _g(s.get("est_crossing_linear")),
                _g(s.get("local_compensation"), 2),
                s.get("verdict", "-"),
            ))
    heads = ("parameter", "side", "point at", "slice dNLL", "profile dNLL",
             "absorbed", "evals", "conv", "width >=", "crossing >=", "VIF^0.5",
             "interval closed?")
    widths = [max(len(h), *(len(r[c]) for r in rows)) if rows else len(h)
              for c, h in enumerate(heads)]
    line = "  ".join(h.ljust(w) for h, w in zip(heads, widths))
    out = ["", f"[fast profile] summary (threshold dNLL = {thr:.4g}):", "",
           "  " + line, "  " + "-" * len(line)]
    for r in rows:
        out.append("  " + "  ".join(v.ljust(w) for v, w in zip(r, widths)))
    c = report["counts"]
    out.append("")
    out.append("  " + "   ".join(f"{k}: {c.get(k, 0)}" for k in VERDICTS))
    out.append("")
    out.append("  Statement judged: 'the 95% interval is closed on this side'.")
    out.append("    proven false   the slice, and so the profile, stays below the "
               "threshold decades out (a proof; the side is open).")
    out.append("    unlikely true  dNLL at the slice crossing fell to within "
               f"{100 * report['near_zero_frac']:.0f}% of zero: the other "
               "parameters absorb the displacement; 'crossing >=' is a lower "
               "bound on where the interval could close.")
    out.append("    likely true    the profile fell below the threshold there "
               "but stayed clear of zero, or stayed above it without converging.")
    out.append("    proven true    the nuisance minimization converged above the "
               "threshold at the slice crossing (proof to optimizer tolerance, "
               "the same standard the full profile uses).")
    out.append("    no verdict     unscreened, unevaluable, or a better optimum "
               "than the fit was found.")
    out.append("  'width >=' is (profile crossing distance)/(slice crossing "
               "distance) under a quadratic profile; since the point is an upper "
               "bound it is a floor, not an estimate.")
    out.append("  'VIF^0.5' is Wald marginal SE over conditional SE from the "
               "Hessian: the same compensation question, linearised at the "
               "optimum.")
    if report.get("deadline"):
        out.append(f"  Stopped on the wall clock: {report['deadline']}")
    out.append("")
    return "\n".join(out)


def print_fast_profile_report(report):
    print(format_summary_table(report), flush=True)
    for name, sides in report["parameters"].items():
        for side_name in ("lower", "upper"):
            s = sides.get(side_name)
            if s is None:
                continue
            print(f"    {name} ({side_name}): {s.get('verdict')} -- {s.get('reason')}")
    print(flush=True)


def fast_profile_summary(report):
    """The report without its per-round detail, for the results snapshot."""
    if not report:
        return None
    keep = ("verdict", "reason", "screen_state", "x_slice_linear",
            "x_point_linear", "slice_dnll_at_point", "dnll", "absorbed",
            "converged", "nfev_total", "width_factor", "est_crossing_linear",
            "est_decades", "est_beyond_reach", "local_compensation",
            "certified_inner_linear", "reach_decades")
    return {
        "threshold": report.get("threshold"),
        "counts": report.get("counts"),
        "n_points": report.get("n_points"),
        "n_evaluations": report.get("n_evaluations"),
        "round_evals": report.get("round_evals"),
        "n_rounds": report.get("n_rounds"),
        "near_zero_frac": report.get("near_zero_frac"),
        "deadline": report.get("deadline"),
        "parameters": {
            name: {side: {k: _jsonable(rec.get(k)) for k in keep}
                   for side, rec in sides.items()}
            for name, sides in report.get("parameters", {}).items()
        },
    }

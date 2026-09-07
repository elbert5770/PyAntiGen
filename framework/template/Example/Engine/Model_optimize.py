"""
High-level optimization setup.  Two entry points:

  setup_optimization(settings, optimization_settings, experiment_dict)
      Original flat-dict path: experiments is a plain dict of treatment dicts.

  setup_optimization_from_groups(settings, optimization_settings, EXPERIMENT_dict)
      Group-aware path: uses Experiment.opt_groups (derived from each replicate's
      Opt_group key) to define which replicates contribute to the objective.
"""
import os
import pandas as pd
import AntiGen_paths

REPO_ROOT = AntiGen_paths.REPO_ROOT

from framework.AntimonyGen import AntimonyGen
from framework.TelluriumGen import TelluriumGen  # used by _run_steady_state
from Modules.Experiment import *
from Engine.Optimize import (
    run_optimization,
    run_optimization_from_groups,
    _extract_profile_ci,
    profile_ci_status,
)
from Engine.Identifiability import UnidentifiableParameters
from Modules.Plots import *
from Engine.Results import log_optimization_results
from Engine.Petab_export import export_petab


# ---------------------------------------------------------------------------
# Diagnostics Presets
# ---------------------------------------------------------------------------

_FULL_DIAGNOSTICS = {
    "wald_analysis": True,
    "slice_analysis": True,
    "profile_likelihood_analysis": True,
    "sobol_analysis": True,
    "sobol_N": 128,
    "profile_se_span": 10,
}

_SLICE_ONLY = {
    "wald_analysis": False,
    "slice_analysis": True,
    "profile_likelihood_analysis": False,
    "sobol_analysis": False,
}

_PROFILE_ONLY = {
    "wald_analysis": True,
    "slice_analysis": False,
    "profile_likelihood_analysis": True,
    "sobol_analysis": False,
    # 4 SE, which is run_parallel_profile's own default; this preset used to
    # override it to 10. The crossing being looked for sits at 1.96 SE for a
    # locally quadratic profile, so a span of 10 spread across n_grid=5 puts
    # points at 2, 4, 6, 8 and 10 SE -- dNLL of roughly 2, 8, 18, 32 and 50,
    # with four of the five deep in a tail whose height nobody needs. At 4 the
    # same five points land at 0.8, 1.6, 2.4, 3.2 and 4.0 SE (dNLL 0.32, 1.28,
    # 2.88, 5.12, 8.0), straddling the threshold with two below and three
    # above, which is what pass 3 refines between.
    #
    # Widening the span was never the right lever for a parameter whose
    # crossing lies further out than Wald predicts: it spends points on every
    # parameter at once, including the ones already answered. max_extend walks
    # outward only where the threshold has not been reached, which is why it
    # exists and why tightening the span here is safe.
    "profile_se_span": 4,
}

_SOBOL_ONLY = {
    "wald_analysis": False,
    "slice_analysis": False,
    "profile_likelihood_analysis": False,
    "sobol_analysis": True,
}

# The Hessian alone: O(k^2) evaluations, minutes rather than days. Worth running
# before any profile, because a Wald SE that is None says the fit is singular in
# that direction and the profile will spend its whole budget discovering the
# same thing -- and an SE wider than the distance to the parameter's bound says
# the box will truncate the interval before the data does.
_WALD_ONLY = {
    "wald_analysis": True,
    "slice_analysis": False,
    "profile_likelihood_analysis": False,
    "sobol_analysis": False,
}

# One capped profile point per side at the slice crossing, classified into
# "proven true / likely true / unlikely true / proven false" on the statement
# "the interval is closed on this side". Hours rather than days, and every
# point lands in the checkpoint the full profile reads. See Engine.Fast_profile.
# The Hessian is needed: the screen places its ladder from the Wald SE, and
# the local compensation factor in the table comes from the covariance.
_FAST_PROFILE_ONLY = {
    "wald_analysis": True,
    "slice_analysis": False,
    "profile_likelihood_analysis": False,
    "fast_profile_likelihood_analysis": True,
    "sobol_analysis": False,
    # Evaluations per round and rounds per point; None takes the module
    # defaults (150 x 3). A point stops early once it has a verdict.
    "fast_profile_round_evals": None,
    "fast_profile_rounds": None,
    # dNLL at or below this fraction of the threshold is "near zero".
    "fast_profile_near_zero_frac": None,
}

_NO_DIAGNOSTICS = {
    "wald_analysis": False,
    "slice_analysis": False,
    "profile_likelihood_analysis": False,
    "sobol_analysis": False,
}

DIAGNOSTICS_PRESETS = {
    "_NO_DIAGNOSTICS": _NO_DIAGNOSTICS,
    "_FULL_DIAGNOSTICS": _FULL_DIAGNOSTICS,
    "_SLICE_ONLY": _SLICE_ONLY,
    "_PROFILE_ONLY": _PROFILE_ONLY,
    "_SOBOL_ONLY": _SOBOL_ONLY,
    "_WALD_ONLY": _WALD_ONLY,
    "WALD_ONLY": _WALD_ONLY,
    "WALD": _WALD_ONLY,
    "_FAST_PROFILE_ONLY": _FAST_PROFILE_ONLY,
    "NO_DIAGNOSTICS": _NO_DIAGNOSTICS,
    "FULL_DIAGNOSTICS": _FULL_DIAGNOSTICS,
    "SLICE_ONLY": _SLICE_ONLY,
    "PROFILE_ONLY": _PROFILE_ONLY,
    "SOBOL_ONLY": _SOBOL_ONLY,
    "FAST_PROFILE_ONLY": _FAST_PROFILE_ONLY,
    "FULL": _FULL_DIAGNOSTICS,
    "SLICE": _SLICE_ONLY,
    "PROFILE": _PROFILE_ONLY,
    "SOBOL": _SOBOL_ONLY,
    "FAST_PROFILE": _FAST_PROFILE_ONLY,
    "NO": _NO_DIAGNOSTICS,
}



# ---------------------------------------------------------------------------
# Fork-safe worker (module-level so it is accessible in forked child memory)
# ---------------------------------------------------------------------------

def _profile_fork_worker(result_queue, param_idx, profile_func):
    """Worker target for fork-based parallel profile on Linux/Mac."""
    try:
        pv, nr = profile_func(param_idx)
        result_queue.put((param_idx, pv, nr, None))
    except Exception as exc:
        result_queue.put((param_idx, None, None, str(exc)))


# ---------------------------------------------------------------------------
# Shared steady-state helper
# ---------------------------------------------------------------------------

def _run_steady_state(model_text, paths, settings):
    """Compute steady state and update InitialConditions CSV."""
    ic_path = os.path.join(
        paths["repo_root"], "antimony_models",
        paths["MODEL_NAME"], f"{paths['MODEL_NAME']}_InitialConditions.csv",
    )
    rss = TelluriumGen(model_text, paths, settings)
    print("Steady state:", rss.steadyState())
    if os.path.exists(ic_path):
        df_ic = pd.read_csv(ic_path)
        if 'Species' in df_ic.columns:
            max_val, vals = 0.0, {}
            for idx, row in df_ic.iterrows():
                try:
                    val = rss[row['Species']]
                    vals[idx] = val
                    if val > max_val:
                        max_val = val
                except RuntimeError:
                    pass
            for idx, val in vals.items():
                df_ic.at[idx, 'InitialCondition'] = 0.0 if val < 1e-10 * max_val else val
            df_ic.to_csv(ic_path, index=False)
            print(f"Updated InitialConditions in {ic_path}")
    return rss


_PROFILE_CI_THRESHOLD = 1.9207  # chi2(df=1, p=0.95) / 2

# A profile flatter than this over its whole grid carries no information about
# the parameter: it is structurally non-identifiable, not merely poorly bounded.
# Well below the threshold, well above nuisance-optimizer noise.
_PROFILE_FLAT_TOL = 1e-3

# How far a slice may sit from zero at the fitted point, or below zero anywhere,
# before it is called out. Same magnitude as the profile's anchor rule, so the
# two diagnostics agree on what counts as "below the reported optimum", and far
# enough above solver noise that a converged fit stays quiet.
_SLICE_DNLL_TOL = 1e-3


def _shutdown_evaluator(opt):
    """Close the worker pool once the diagnostic closures have been consumed.

    run_optimization_from_groups cannot close it itself: the profile/slice
    closures it returns are called from here, after it returns.
    """
    if not isinstance(opt, dict):
        return
    ev = opt.get("stats", {}).pop("_evaluator", None)
    if ev is None:
        return
    try:
        print(f"[pool] shutting down after {ev.n_evals} evaluation(s)"
              + (f", {ev.n_failures} failed" if ev.n_failures else ""))
        ev.shutdown()
    except Exception as exc:
        print(f"[pool] shutdown warning: {exc}")


# Grid controls for the parallel profile, and the settings key that reaches each.
# The engine holds the defaults; only keys a run actually sets are forwarded, so
# there is one place to change a default rather than two.
_PROFILE_GRID_SETTINGS = {
    "profile_se_span":       "se_span",
    "profile_n_grid":        "n_grid",
    "profile_n_refine":      "n_refine",
    "profile_range_factor":  "range_factor",
    "profile_warm_passes":   "warm_passes",
    "profile_max_extend":    "max_extend",
    "profile_extend_growth": "extend_growth",
    "profile_bracket_rtol":  "bracket_rtol",
}


def _profile_kwargs(settings, optimization_spec=None):
    """Profile grid arguments for this run, from the settings and the spec.

    Two sources, and the spec wins. A run's diagnostics preset says what the
    operator asked for today; grid density is a property of how expensive one
    spec's evaluations are, and the two specs sharing ``_PROFILE_ONLY`` differ
    by an order of magnitude in that cost. A spec states its own under
    ``optimizer_kwargs["profile_grid"]``, keyed by the engine's own argument
    names (``n_grid``, ``se_span``, ...). An unknown key is an error rather
    than a silent no-op: a typo there would otherwise look exactly like a
    setting that did not help.

    ``max_extend`` is the one to reach for when a CI comes back open. The
    opening grid is placed from the Wald SE (``se_span``) or, when the Hessian
    gave no SE, from ``range_factor``; whichever it is, it is a guess about
    where dNLL reaches 1.9207, and ``max_extend`` bounds how many times a side
    that guessed short may step further out before giving up. Raising
    ``se_span`` widens the opening guess for *every* parameter at once, which
    spends points on the ones already answered; raising ``max_extend`` spends
    them only where the answer is still missing.

    ``bracket_rtol`` is the opposite control: how precisely a crossing that has
    been bracketed needs to be located before probing stops. It is a relative
    precision on the confidence bound, so 0.05 means "to within 5%" -- far finer
    than identifiability needs, and the budget it releases goes to the sides
    that have not reached the threshold at all.
    """
    out = {}
    for key, arg in _PROFILE_GRID_SETTINGS.items():
        value = (settings or {}).get(key)
        if value is not None:
            out[arg] = value

    spec_kwargs = getattr(optimization_spec, "optimizer_kwargs", None) or {}
    known = set(_PROFILE_GRID_SETTINGS.values())
    for arg, value in (spec_kwargs.get("profile_grid") or {}).items():
        if arg not in known:
            raise ValueError(
                f"profile_grid key {arg!r} is not a profile grid argument; "
                f"expected one of {sorted(known)}"
            )
        if value is not None:
            out[arg] = value
    return out


_FAST_PROFILE_SETTINGS = {
    "fast_profile_round_evals":    "round_evals",
    "fast_profile_rounds":         "n_rounds",
    "fast_profile_near_zero_frac": "near_zero_frac",
}


def _fast_profile_kwargs(settings, optimization_spec=None):
    """Arguments for the fast profile pass, from the settings and the spec.

    The round sizes come from the run settings only. The screen's reach comes
    from the spec's ``profile_grid`` where it states one, so the fast pass and
    the full profile walk the same distance and their screens are the same
    file.
    """
    out = {}
    for key, arg in _FAST_PROFILE_SETTINGS.items():
        value = (settings or {}).get(key)
        if value is not None:
            out[arg] = value
    spec_kwargs = getattr(optimization_spec, "optimizer_kwargs", None) or {}
    grid = spec_kwargs.get("profile_grid") or {}
    for arg in ("screen_span_decades", "screen_min_reach_decades"):
        if grid.get(arg) is not None:
            out[arg] = grid[arg]
    return out


def _run_fast_profile_report(opt, settings, optimization_spec=None):
    """Run the fast pass, print its summary table, keep it for the snapshot."""
    from Engine.Fast_profile import fast_profile_summary, print_fast_profile_report

    fast_all = opt.get("stats", {}).get("fast_profile_all")
    if fast_all is None:
        return None
    try:
        report = fast_all(**_fast_profile_kwargs(settings, optimization_spec))
    except Exception as exc:
        import traceback
        print(f"  fast profile failed ({exc}).")
        traceback.print_exc()
        return None
    print_fast_profile_report(report)
    opt["stats"]["fast_profile"] = fast_profile_summary(report)
    return report


# How far past the threshold crossing the profile figure extends, as a fraction
# of the crossing's own distance from the optimum. Enough to show the curve
# continuing past the bound without letting a grid that was clipped to a
# far-away parameter bound dictate the scale.
_PROFILE_PLOT_MARGIN = 0.25


def _profile_plot_x(cache, params_estimated, profile_ci=None,
                    margin=_PROFILE_PLOT_MARGIN):
    """The x window for the profile figure, and whether a log axis is usable.

    Returns ``((lo, hi), use_log)``, or ``(None, use_log)`` when there is
    nothing to plot.

    The window is anchored on the threshold crossings -- the confidence bounds --
    and extends *margin* beyond them, measured as a fraction of each crossing's
    own distance from the optimum. That distance is multiplicative on a log
    axis, so the margin is too.

    Anchoring on the crossing rather than on the data is what keeps the figure
    readable. Two earlier rules both failed, in opposite directions. The
    original fixed window of [0.2, 3.5] was too narrow: it pre-dated the outward
    extension pass, so a crossing at a fifth of the optimum fell off the left
    edge. Spanning every computed point instead was too wide: a grid clipped to
    a parameter bound a thousand-fold below the optimum -- which is what the
    GantenerumabIV and Donanemab specs produce -- pushed the whole informative
    region, crossings included, into a sliver at the right-hand edge.

    A side with no crossing has no anchor, so it falls back to how far it was
    actually walked: that a profile ran three decades without reaching the
    threshold is the finding for that parameter, and cropping it would hide the
    one thing worth seeing. The window never extends past the computed data
    either, so it cannot imply points that were never evaluated.

    A log axis is used whenever every plotted ratio is positive, which is nearly
    always -- these are rate constants, volumes and clearances, and extension
    walks them over decades, which a linear axis cannot show at all.
    """
    import numpy as np

    ci = list(profile_ci or [])
    lo_all, hi_all = np.inf, -np.inf
    all_positive = True
    ratios = {}
    for idx, cached in cache.items():
        pv, _nr = cached
        r = np.asarray(pv, dtype=float) / params_estimated[idx]
        r = r[np.isfinite(r)]
        if r.size == 0:
            continue
        ratios[idx] = r
        if np.any(r <= 0):
            all_positive = False
        lo_all, hi_all = min(lo_all, r.min()), max(hi_all, r.max())

    if not ratios or not (np.isfinite(lo_all) and np.isfinite(hi_all)):
        return None, False
    use_log = all_positive and lo_all > 0

    def dist(ratio, sign):
        """How far *ratio* lies from the optimum on side *sign*, or None."""
        if use_log:
            if ratio <= 0:
                return None
            d = -np.log10(ratio) if sign < 0 else np.log10(ratio)
        else:
            d = (1.0 - ratio) if sign < 0 else (ratio - 1.0)
        return float(d) if d > 0 else None

    reach = {-1: 0.0, +1: 0.0}
    for idx, r in ratios.items():
        opt_v = float(params_estimated[idx])
        bounds = ci[idx] if idx < len(ci) else (np.nan, np.nan)
        for sign, ci_val in ((-1, bounds[0]), (+1, bounds[1])):
            side = r[r < 1.0] if sign < 0 else r[r > 1.0]
            d_side = [dist(v, sign) for v in side]
            d_data = max([d for d in d_side if d is not None], default=0.0)
            d_ci = None
            try:
                if opt_v != 0 and np.isfinite(float(ci_val)):
                    d_ci = dist(float(ci_val) / opt_v, sign)
            except (TypeError, ValueError):
                d_ci = None
            # Never draw further out than the profile was actually computed.
            d = min(d_ci * (1.0 + margin), d_data) if d_ci else d_data
            reach[sign] = max(reach[sign], d)

    if reach[-1] <= 0 and reach[+1] <= 0:
        # Everything sits on the optimum: give it an interval to be drawn in.
        return ((lo_all / 1.5, hi_all * 1.5), True) if use_log \
            else ((lo_all - 0.5, hi_all + 0.5), False)

    if use_log:
        lo, hi = 10.0 ** -reach[-1], 10.0 ** reach[+1]
        pad = max((hi / lo) ** 0.03, 1.05)
        return (lo / pad, hi * pad), True
    lo, hi = 1.0 - reach[-1], 1.0 + reach[+1]
    span = max(hi - lo, 1e-12)
    return (lo - 0.03 * span, hi + 0.03 * span), False


def _fmt_g(x, width=0, prec=6):
    """Format a number for the report, or "n/a" when it is missing.

    The report is printed to a Windows console as well as written to a UTF-8
    file, and that console is routinely on a codepage where en/em dashes come
    out as replacement characters, so this whole report stays ASCII.
    """
    import numpy as np
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "n/a".rjust(width)
    if not np.isfinite(v):
        return "n/a".rjust(width)
    return f"{v:.{prec}g}".rjust(width)


def _profile_report_text(opt, param_names, params_estimated, profile_ci,
                         profile_ci_state, cache, model_name, tag, stamp,
                         plot_files=()):
    """Consolidated profile-likelihood summary: the intervals and what backs them.

    The console reports each interval as it is extracted, interleaved with that
    parameter's advice. That is the right shape while a run is in progress and
    the wrong shape afterwards: the question a profile exists to answer -- which
    parameters are identifiable, and how far each bound can be trusted -- has to
    be reassembled by eye from lines scattered through thousands of solver
    messages. This gathers it in one place and writes it next to the figures, so
    a run's conclusion outlives its terminal scrollback.

    Every number here is one the reader would otherwise have to hunt for or
    recompute: the interval as a multiple of the optimum (these are rate
    constants and clearances, where a factor is the meaningful unit, not a
    difference), how far each side was actually walked and how high it got,
    whether a missing bound is a result or an unfinished search, and how much of
    the curve rests on nuisance optimizations that never converged.
    """
    import numpy as np

    stats = opt.get("stats", {}) or {}
    conv = stats.get("profile_convergence") or {}
    reach = conv.get("reach") or {}
    per_param = conv.get("per_param") or {}
    warm = conv.get("warm") or {}
    anchor = float(stats.get("profile_anchor_gap", 0.0) or 0.0)

    se_raw = stats.get("wald_se")
    se_arr = (np.atleast_1d(se_raw) if se_raw is not None
              else np.full(len(param_names), np.nan))
    wald_ci = stats.get("wald_ci") or [(np.nan, np.nan)] * len(param_names)

    _reach_word = {"crossed": "crossed the threshold",
                   "bound": "stopped at the parameter bound",
                   "budget": "ran out of extension steps",
                   "empty": "no points",
                   "missing": "not computed"}

    L = []
    rule = "=" * 78
    L.append(rule)
    L.append(f"PROFILE LIKELIHOOD SUMMARY - {model_name}  [{tag}]")
    L.append(f"generated {stamp}")
    L.append(rule)
    L.append("")
    L.append(f"Threshold          dNLL = {_PROFILE_CI_THRESHOLD:.4f}  "
             f"(chi2(df=1, p=0.95) / 2)")
    _running = conv.get("n_interrupted", 0)
    L.append(f"Profile points     {conv.get('n_points', 0)} total, "
             f"{conv.get('n_not_converged', 0)} hit the optimizer cap, "
             f"{conv.get('n_unknown', 0)} unknown"
             + (f", {_running} still in progress" if _running else ""))
    if _running:
        L.append(f"*** INCOMPLETE: {_running} point(s) were stopped by the "
                 f"wall clock with their state saved, not by the optimizer.")
        L.append("    Their dNLL is an upper bound, so every interval below is "
                 "provisional and too narrow. Relaunch to continue.")
    if warm.get("n_attempted"):
        L.append(f"Warm continuation  {warm.get('n_improved', 0)} of "
                 f"{warm['n_attempted']} point(s) improved, "
                 f"{warm.get('nats_recovered', 0.0):.4g} nats recovered")
    if anchor < -1e-3:
        L.append(f"*** The profile found a point {abs(anchor):.4g} nats BELOW "
                 f"the reported optimum: the fit has not converged, and the")
        L.append(f"    parameter values above are not the MLE. Refit before "
                 f"quoting anything here.")
        bp = conv.get("better_point") or {}
        if bp:
            L.append("")
            L.append(f"    lowest NLL found   {bp['nll']:.10g}")
            L.append(f"    found while        profiling {bp['parameter']} "
                     f"= {bp['value']:.8g}")
            xs, names = bp.get("x"), bp.get("param_names")
            if xs and names and len(xs) == len(names):
                L.append("    restart the fit from these values:")
                width = max(len(n) for n in names)
                for n, v in zip(names, xs):
                    L.append(f"      {n:<{width}}  {v:.8e}")
    else:
        L.append("Anchor             the fit sits at the profile minimum")
    L.append("")

    L.append("-" * 78)
    L.append("PARAMETERS")
    L.append("-" * 78)
    for i, pname in enumerate(param_names):
        opt_val = float(params_estimated[i])
        lo, hi = profile_ci[i] if i < len(profile_ci) else (np.nan, np.nan)
        status = (profile_ci_state[i] if i < len(profile_ci_state)
                  else "missing")
        sides = reach.get(pname) or {}
        pp = per_param.get(pname) or {}

        L.append("")
        L.append(f"{pname}")
        L.append(f"    optimum          {_fmt_g(opt_val)}")
        L.append(f"    profile 95% CI   [{_fmt_g(lo)}, {_fmt_g(hi)}]"
                 f"      status: {status}")
        # A multiplicative read of the interval. For rate constants and
        # clearances "between 0.76x and 1.7x of the fitted value" is the
        # statement a modeller can act on; the absolute bounds above are not.
        if opt_val > 0 and np.isfinite(lo) and np.isfinite(hi) and lo > 0:
            L.append(f"    as a factor      [{lo / opt_val:.4g}x, "
                     f"{hi / opt_val:.4g}x] of the optimum "
                     f"(spans {hi / lo:.4g}x)")
        for key in ("lower", "upper"):
            d = sides.get(key)
            if not d:
                continue
            word = _reach_word.get(d["state"], d["state"])
            L.append(f"    {key + ' side':<16} {word}; walked to "
                     f"{_fmt_g(d.get('reach'))}, highest dNLL "
                     f"{_fmt_g(d.get('max_dnll'), prec=4)}")
        se_i = float(se_arr[i]) if i < len(se_arr) else np.nan
        wlo, whi = (wald_ci[i] if i < len(wald_ci) else (np.nan, np.nan))
        if np.isfinite(se_i):
            L.append(f"    Wald SE          {_fmt_g(se_i)}      "
                     f"Wald 95% CI [{_fmt_g(wlo)}, {_fmt_g(whi)}]")
        else:
            # Worth saying explicitly: no SE means the Hessian was singular in
            # this direction, which is also why the opening grid for this
            # parameter came from range_factor rather than from a curvature
            # estimate.
            L.append(f"    Wald SE          none - the Hessian was singular in "
                     f"this direction")
        if pp:
            capped = pp.get("n_not_converged", 0)
            running = pp.get("n_interrupted", 0)
            parts = []
            if capped:
                # Hitting the evaluation cap is not by itself evidence of
                # anything. Measured on the engine's own 15-parameter
                # ill-conditioned quadratic, 79% of points stop on a cap of
                # 1500 while the intervals they produce are within 0.02% of the
                # analytic answer: the search had reached the minimum and was
                # polishing. Reporting "this interval is too narrow" on every
                # such point cried wolf on healthy runs and buried the real
                # ones.
                #
                # What does measure it is how far the warm pass moved the
                # point. That is a direct observation of how much the earlier
                # value was above the profile, so it is what the verdict is
                # based on when it is available.
                near = pp.get("warm_gain_near")
                measured = pp.get("n_warm_measured", 0)
                if not measured:
                    verdict = ("not yet re-run warm, so the bias is "
                               "unmeasured")
                elif near is None or not np.isfinite(near):
                    verdict = "bias unmeasured"
                elif near > 0.1 * _PROFILE_CI_THRESHOLD:
                    verdict = (f"the warm pass still lowered points near the "
                               f"crossing by up to {near:.3g} nats, so this "
                               f"interval is too narrow")
                elif near > 0:
                    verdict = (f"but the warm pass moved them by at most "
                               f"{near:.3g} nats near the crossing, which does "
                               f"not move the bound")
                else:
                    verdict = ("and the warm pass did not lower them, so the "
                               "cap was polishing a minimum already reached")
                parts.append(f"{capped} hit the optimizer cap ({verdict})")
            if running:
                parts.append(f"{running} still in progress")
            tail = (", " + ", ".join(parts)) if parts else ""
            L.append(f"    points           {pp.get('n', 0)}{tail}")
        if status == "flat":
            L.append("    NOTE             the profile is flat: this parameter "
                     "is structurally non-identifiable")

    # ── The verdict, which is the reason anyone opens this file ──────────
    n_ok = sum(1 for s in profile_ci_state if s == "ok")
    n_one = sum(1 for s in profile_ci_state if s in ("open_lower", "open_upper"))
    n_open = sum(1 for s in profile_ci_state if s == "open")
    n_flat = sum(1 for s in profile_ci_state if s == "flat")
    at_bound, at_budget = [], []
    for pname, sides in reach.items():
        for key, d in sides.items():
            if d.get("state") == "bound":
                at_bound.append(f"{pname} ({key})")
            elif d.get("state") == "budget":
                at_budget.append(f"{pname} ({key})")

    L.append("")
    L.append("-" * 78)
    L.append("IDENTIFIABILITY")
    L.append("-" * 78)
    L.append(f"    both bounds found          {n_ok} of {len(param_names)}")
    if n_one:
        L.append(f"    one bound only             {n_one}")
    if n_open:
        L.append(f"    neither bound found        {n_open}")
    if n_flat:
        L.append(f"    flat (non-identifiable)    {n_flat}")

    if at_bound:
        L.append("")
        L.append("    Walked to the parameter's own bound without reaching the")
        L.append("    threshold - not identifiable anywhere it is allowed to go.")
        L.append("    Widen that bound only if wider values are physical:")
        for s in at_bound[:20]:
            L.append(f"      {s}")
        if len(at_bound) > 20:
            L.append(f"      ... and {len(at_bound) - 20} more")
    if at_budget:
        L.append("")
        L.append("    Ran out of extension steps - nothing established either")
        L.append("    way. Raise profile_max_extend and re-run; the checkpoint")
        L.append("    keeps the points already computed:")
        for s in at_budget[:20]:
            L.append(f"      {s}")
        if len(at_budget) > 20:
            L.append(f"      ... and {len(at_budget) - 20} more")

    if plot_files:
        L.append("")
        L.append("-" * 78)
        L.append("FIGURES")
        L.append("-" * 78)
        for p in plot_files:
            L.append(f"    {os.path.basename(p)}")

    L.append("")
    L.append(rule)
    return "\n".join(L)


def _save_profile_likelihood_plot(opt, param_names, plot_path, model_name,
                                  tag="ALL", profile_kwargs=None):
    """Run profile likelihood once per parameter, extract CIs, and save plot.

    Parallelises across parameters on Linux / macOS via fork-based
    multiprocessing.Process (inherits the closure, including non-picklable
    RoadRunner objects, via copy-on-write — no serialisation needed). Windows
    lacks fork, so it falls back to the sequential loop below.

    Stores profile_ci into opt['stats']['profile_ci'] so log_optimization_results
    can include it in the CSV when called afterwards.
    """
    import sys
    from datetime import datetime
    import numpy as np
    import matplotlib.pyplot as plt

    profile_func = opt.get("stats", {}).get("profile_likelihood")
    if not profile_func:
        print("Warning: no profile_likelihood closure in opt['stats'] — skipping plot.")
        return

    # One stamp for every artefact this call produces, so the three figures and
    # the report are identifiable as one run at a glance. Successive runs used
    # to overwrite each other's figures, which made comparing a re-run against
    # the run that motivated it impossible -- the evidence was already gone.
    stamp = datetime.now()
    ts = stamp.strftime("%Y%m%d_%H%M%S")

    colors  = ['blue', 'green', 'red', 'orange', 'purple', 'brown']
    markers = ['o', 's', '^', 'D', 'v']
    params_estimated = np.asarray(opt["x"])
    n_params = len(param_names)

    print("\nProfile likelihood:")
    cache = {}

    # Preferred path: the parallel, checkpointed grid. It replaces both the
    # fork-only branch below (which never ran on Windows) and the sequential
    # fallback, and it is resumable.
    profile_all = opt.get("stats", {}).get("profile_likelihood_all")
    if profile_all is not None:
        try:
            traces, anchor, where, convergence = profile_all(
                **(profile_kwargs or {}))
            opt["stats"]["profile_anchor_gap"] = float(anchor)
            opt["stats"]["profile_convergence"] = convergence
            if where is not None and anchor < -1e-3:
                # The engine assembles the full linear vector; keep the raw
                # nuisance solution beside it for anyone re-entering the
                # optimizer in its own space.
                better = dict(convergence.get("better_point") or {})
                better.setdefault("parameter", where[0])
                better.setdefault("value", where[1])
                better["nuisance_x"] = where[2]
                opt["stats"]["profile_better_point"] = better
            for i, pname in enumerate(param_names):
                if pname not in traces:
                    continue
                pv, nr = traces[pname]
                if len(pv) < 2:
                    print(f"  {pname}: too few points to profile")
                    continue
                cache[i] = (np.asarray(pv), np.asarray(nr))
                print(f"  {pname}: dNLL range [{nr.min():.4g}, {nr.max():.4g}]")
                # Range, not distance from zero: the optimum is no longer
                # spliced in as a hardcoded 0.0, so a genuinely flat profile
                # really does have zero range and is detected. The old test
                # could never fire.
                if np.ptp(nr) < _PROFILE_FLAT_TOL:
                    print(f"    *** FLAT: {pname} is structurally non-identifiable "
                          f"(profile varies by {np.ptp(nr):.3g} over the whole grid) ***")
        except UnidentifiableParameters:
            # Not a failure of the parallel path, so there is nothing to fall
            # back to: the sequential profile would spend days reaching the
            # same verdict the screen has already proved. It has to travel past
            # this handler intact or the run would quietly continue.
            raise
        except Exception as exc:
            import traceback
            print(f"  parallel profile failed ({exc}); falling back.")
            traceback.print_exc()
            cache = {}

    _is_posix   = sys.platform != 'win32'

    if cache:
        pass  # parallel grid already produced every trace
    elif n_params > 1 and _is_posix:
        # ── Linux / macOS: fork each parameter into its own process ──────────
        # Forked children inherit the closure (including non-picklable RoadRunner
        # objects) directly via copy-on-write — no serialisation needed.
        import multiprocessing as mp
        ctx = mp.get_context('fork')
        q   = ctx.Queue()
        procs = [
            ctx.Process(target=_profile_fork_worker, args=(q, i, profile_func))
            for i in range(n_params)
        ]
        print(f"  [parallel] forking {n_params} processes (one per parameter) …")
        for p in procs:
            p.start()
        for p in procs:
            p.join()
        for _ in range(n_params):
            i, pv, nr, err = q.get()
            if err is None:
                cache[i] = (pv, nr)
                print(f"  {param_names[i]}: dNLL range [{nr.min():.4g}, {nr.max():.4g}]")
                if np.all(np.abs(nr) < 1e-10):
                    print(f"    *** FLAT: model may not respond to {param_names[i]} ***")
            else:
                print(f"  {param_names[i]}: error — {err}")

    else:
        # ── Sequential fallback (single param, or non-POSIX platforms) ────────
        for i, pname in enumerate(param_names):
            try:
                pv, nr = profile_func(i)
                cache[i] = (pv, nr)
                print(f"  {pname}: dNLL range [{nr.min():.4g}, {nr.max():.4g}]")
                if np.all(np.abs(nr) < 1e-10):
                    print(f"    *** FLAT: model may not respond to {pname} ***")
            except Exception as exc:
                print(f"  {pname}: error — {exc}")

    # Extract profile CIs and store so Results.py can write them to CSV
    profile_ci = []
    profile_ci_state = []
    profile_traces = {}
    print("\n  Profile likelihood 95% CIs:")
    _status_note = {
        "flat":       "structurally non-identifiable — profile is flat",
        "open":       "no crossing either side",
        "open_lower": "no lower crossing",
        "open_upper": "no upper crossing",
        "empty":      "no profile points",
        "missing":    "profile not computed",
    }
    # Which sides a status leaves unresolved, so the advice can name what
    # actually stopped each one rather than always blaming the grid width.
    _status_sides = {"open": ("lower", "upper"), "open_lower": ("lower",),
                     "open_upper": ("upper",)}
    _reach = (opt.get("stats", {}).get("profile_convergence") or {}).get("reach", {})

    def _reach_advice(pname, status):
        """Why the missing bound is missing: a real answer, or an exhausted budget.

        These call for opposite responses. A side that walked to the parameter's
        own bound without dNLL reaching 1.9207 has answered the question -- the
        parameter is not identifiable anywhere it is allowed to go -- and
        re-running with a wider grid would change nothing, because the bound,
        not the grid, is the constraint. A side that merely ran out of extension
        steps has answered nothing and does want a re-run.
        """
        sides = _reach.get(pname) or {}
        notes = []
        for key in _status_sides.get(status, ()):
            d = sides.get(key)
            if not d:
                continue
            if d["state"] == "bound":
                notes.append(f"{key} side reached the parameter bound "
                             f"({d['reach']:.4g}) at dNLL {d['max_dnll']:.3g} — "
                             f"not identifiable within its declared bounds")
            elif d["state"] == "budget":
                notes.append(f"{key} side stopped at {d['reach']:.4g} with dNLL "
                             f"{d['max_dnll']:.3g} — raise profile_max_extend")
        return notes

    for i, pname in enumerate(param_names):
        cached = cache.get(i)
        if cached is not None:
            lo, hi = _extract_profile_ci(cached[0], cached[1], threshold=_PROFILE_CI_THRESHOLD)
            status = profile_ci_status(cached[0], cached[1], lo, hi,
                                       flat_tol=_PROFILE_FLAT_TOL)
            profile_traces[pname] = {"x": cached[0].tolist(), "y": cached[1].tolist()}
        else:
            lo, hi, status = float('nan'), float('nan'), "missing"
        profile_ci.append((lo, hi))
        profile_ci_state.append(status)
        note = f"   [{status}: {_status_note[status]}]" if status != "ok" else ""
        for advice in _reach_advice(pname, status):
            note += f"\n        {advice}"
        # A CI is only as trustworthy as the nuisance optimizations underneath
        # it, so the caveat belongs on the interval itself rather than only in a
        # summary further up the log.
        conv = (opt.get("stats", {}).get("profile_convergence") or {}) \
            .get("per_param", {}).get(pname)
        if conv and conv.get("n_not_converged"):
            note += (f"   [{conv['n_not_converged']}/{conv['n']} point(s) hit "
                     f"the optimizer cap — interval is too narrow]")
        print(f"    {pname}: [{lo:.4g}, {hi:.4g}]{note}")
    opt["stats"]["profile_ci"] = profile_ci
    opt["stats"]["profile_ci_status"] = profile_ci_state
    opt["stats"]["profile_traces"] = profile_traces

    fig, ax = plt.subplots(figsize=(8, 6))
    for idx, pname in enumerate(param_names):
        cached = cache.get(idx)
        if cached is None:
            continue
        pv, nr = cached
        ax.plot(pv / params_estimated[idx], nr,
                marker=markers[(idx // len(colors)) % len(markers)], linestyle='-',
                label=pname, linewidth=2, color=colors[idx % len(colors)])

    ax.axhline(_PROFILE_CI_THRESHOLD, color='gray', linestyle=':', alpha=0.7, linewidth=1.5,
               label=f'95% CI (Δ NLL = {_PROFILE_CI_THRESHOLD:.2f})')
    ax.axvline(1.0, color='red', linestyle='--', alpha=0.5, linewidth=1.5, label='Optimal')
    ax.set_xlabel('Parameter Value (relative to optimal)')
    ax.set_ylabel('Δ NLL (relative to profile minimum)')
    _gap = opt.get("stats", {}).get("profile_anchor_gap", 0.0)
    if _gap < -1e-3:
        # Say it on the figure too — a plot whose optimum line sits off the
        # minimum is the single most misleading output this code can produce.
        ax.set_title('Profile Likelihood — WARNING: fit is '
                     f'{abs(_gap):.3g} nats above the profile minimum')
    else:
        ax.set_title('Profile Likelihood (Identifiability Check)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    _xwin, _use_log = _profile_plot_x(cache, params_estimated, profile_ci)
    if _use_log:
        ax.set_xscale('log')
    if _xwin:
        ax.set_xlim(*_xwin)
    plt.tight_layout()
    _base = os.path.join(plot_path, f"{model_name}_{tag}_profile_likelihood")
    out_path = f"{_base}_{ts}.png"
    plt.savefig(out_path, bbox_inches="tight")
    print(f"Profile likelihood saved to: {out_path}")

    zoom_half = _PROFILE_CI_THRESHOLD * 0.10
    ax.set_ylim(-zoom_half, zoom_half)
    ax.set_title(f'Profile Likelihood (Identifiability Check, y-axis ±10% CI)')
    out_path_zoom = f"{_base}_zoom_{ts}.png"
    plt.savefig(out_path_zoom, bbox_inches="tight")
    print(f"Profile likelihood (zoomed) saved to: {out_path_zoom}")

    ax.set_ylim(-_PROFILE_CI_THRESHOLD * 0.10, _PROFILE_CI_THRESHOLD)
    ax.set_title(f'Profile Likelihood (Identifiability Check, y-axis -10% to +100% CI)')
    out_path_zoom2 = f"{_base}_zoom2_{ts}.png"
    plt.savefig(out_path_zoom2, bbox_inches="tight")
    print(f"Profile likelihood (zoomed2) saved to: {out_path_zoom2}")
    plt.close(fig)

    # ── Summary report ───────────────────────────────────────────────────
    report = _profile_report_text(
        opt, param_names, params_estimated, profile_ci, profile_ci_state,
        cache, model_name, tag, stamp.strftime("%Y-%m-%d %H:%M:%S"),
        plot_files=(out_path, out_path_zoom, out_path_zoom2),
    )
    report_path = os.path.join(
        plot_path, f"{model_name}_{tag}_profile_summary_{ts}.txt")
    try:
        with open(report_path, "w", encoding="utf-8") as fh:
            fh.write(report + "\n")
        print(f"\n{report}")
        print(f"\nProfile summary saved to: {report_path}")
    except OSError as exc:
        # The report is a convenience; losing it must not cost the run the
        # diagnostics it just spent hours computing.
        print(f"\n{report}")
        print(f"\n  [profile] could not write {report_path}: {exc}")
    opt["stats"]["profile_report"] = report
    opt["stats"]["profile_report_path"] = report_path


def _slice_dnll_at_optimum(param_vals, dnll, opt_val, rtol=1e-6):
    """The slice's own dNLL at the fitted value, or None if the grid missed it.

    The grid is built to contain the optimum, so this is normally a lookup. It
    can still come back None for a parameter whose fitted value is zero on a
    linear scale, or one whose slice errored, and the caller has to survive
    both.
    """
    import numpy as np

    pv = np.asarray(param_vals, dtype=float)
    y = np.asarray(dnll, dtype=float)
    if pv.size == 0 or opt_val == 0:
        return None
    ratio = np.abs(pv / opt_val - 1.0)
    i = int(np.argmin(ratio))
    if ratio[i] > rtol:
        return None
    return float(y[i])


def _save_likelihood_slice_plot(opt, param_names, plot_path, model_name, tag="ALL"):
    """Run likelihood slice once per parameter and save plot."""
    from datetime import datetime
    import numpy as np
    import matplotlib.pyplot as plt

    slice_func = opt.get("stats", {}).get("likelihood_slice")
    if not slice_func:
        print("Warning: no likelihood_slice closure in opt['stats'] — skipping plot.")
        return

    # Stamped like the profile figures, and for the same reason: successive runs
    # used to overwrite each other, so the slice that motivated a change was
    # gone by the time there was anything to compare it against.
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    colors  = ['blue', 'green', 'red', 'orange', 'purple', 'brown']
    markers = ['o', 's', '^', 'D', 'v']
    params_estimated = np.asarray(opt["x"])

    print("\nLikelihood slice:")
    cache = {}
    slice_traces = {}

    # Prefer the all-parameter form: it submits k x n_points evaluations as a
    # single batch, so a 24-core pool is fully fed. Going parameter-by-parameter
    # caps the batch at n_points (20), leaving most workers idle.
    slice_all = opt.get("stats", {}).get("likelihood_slice_all")
    results = None
    if slice_all is not None:
        try:
            results = slice_all(n_points=20, range_factor=2.0)
        except Exception as exc:
            print(f"  batched slice failed ({exc}); falling back per parameter.")

    dnll_at_opt = {}
    for i, pname in enumerate(param_names):
        try:
            if results is not None and pname in results:
                pv, nr = results[pname]
            else:
                pv, nr = slice_func(i, n_points=20, range_factor=2.0)
            cache[i] = (pv, nr)
            slice_traces[pname] = {"x": np.asarray(pv).tolist(),
                                   "y": np.asarray(nr).tolist()}
            # The optimum is a sampled point, not an assumption, so its dNLL is
            # worth stating: it should be 0, and anything else means the
            # diagnostic evaluator and the fit disagree about the fitted point.
            d0 = _slice_dnll_at_optimum(pv, nr, float(params_estimated[i]))
            if d0 is not None:
                dnll_at_opt[pname] = d0
            at_opt = (f", dNLL at optimum {d0:.4g}" if d0 is not None
                      else ", optimum not on the grid")
            print(f"  {pname}: dNLL range [{nr.min():.4g}, {nr.max():.4g}]{at_opt}")
            if np.all(np.abs(nr) < 1e-10):
                print(f"    *** FLAT: model may not respond to {pname} ***")
            # A slice does not re-optimize the nuisances, so a point below the
            # optimum is not a definition mismatch the way it can be in a
            # profile: the fit is beatable by moving this parameter alone.
            if nr.min() < -_SLICE_DNLL_TOL:
                j = int(np.argmin(nr))
                print(f"    *** WARNING: {pname}={pv[j]:.6g} scores "
                      f"{abs(nr.min()):.4g} nats BELOW the reported optimum "
                      f"with every other parameter held fixed - the fit has "
                      f"not converged ***")
        except Exception as exc:
            print(f"  {pname}: error — {exc}")
    opt["stats"]["slice_traces"] = slice_traces
    opt["stats"]["slice_dnll_at_optimum"] = dnll_at_opt
    _worst_at_opt = max((abs(v) for v in dnll_at_opt.values()), default=0.0)
    if _worst_at_opt > _SLICE_DNLL_TOL:
        print(f"  *** WARNING: the slice scores the fitted point "
              f"{_worst_at_opt:.4g} nats away from the value the fit reported "
              f"there - every dNLL above is measured from the wrong place ***")

    fig, ax = plt.subplots(figsize=(8, 6))
    for idx, pname in enumerate(param_names):
        cached = cache.get(idx)
        if cached is None:
            continue
        pv, nr = cached
        ax.plot(pv / params_estimated[idx], nr,
                marker=markers[(idx // len(colors)) % len(markers)], linestyle='-',
                label=pname, linewidth=2, color=colors[idx % len(colors)])

    # The fitted point itself, drawn on top of the curves. It is where the
    # curves now meet, and on the zoomed figures it is often the only point of a
    # steep slice still on screen, so it needs to be identifiable as such.
    _opt_ys = [dnll_at_opt[p] for p in param_names if p in dnll_at_opt]
    if _opt_ys:
        ax.plot([1.0] * len(_opt_ys), _opt_ys, linestyle='none', marker='x',
                color='black', markersize=8, markeredgewidth=1.5, zorder=5,
                label=f'Optimum (Δ NLL = {max(_opt_ys, key=abs):.3g})')

    ax.axhline(_PROFILE_CI_THRESHOLD, color='gray', linestyle=':', alpha=0.7, linewidth=1.5,
               label=f'95% CI threshold (Δ NLL = {_PROFILE_CI_THRESHOLD:.2f})')
    ax.axvline(1.0, color='red', linestyle='--', alpha=0.5, linewidth=1.5, label='Optimal')
    ax.set_xlabel('Parameter Value (relative to optimal)')
    ax.set_ylabel('Δ NLL (relative to minimum)')
    if _worst_at_opt > _SLICE_DNLL_TOL:
        ax.set_title('Likelihood Slice — WARNING: slice is '
                     f'{_worst_at_opt:.3g} nats off the fit at the optimum')
    else:
        ax.set_title('Likelihood Slice')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.2, 3.5)
    plt.tight_layout()
    _base = os.path.join(plot_path, f"{model_name}_{tag}_likelihood_slice")
    out_path = f"{_base}_{ts}.png"
    plt.savefig(out_path, bbox_inches="tight")
    print(f"Likelihood slice saved to: {out_path}")

    zoom_half = _PROFILE_CI_THRESHOLD * 0.10
    ax.set_ylim(-zoom_half, zoom_half)
    ax.set_title(f'Likelihood Slice (y-axis ±10% CI)')
    out_path_zoom = f"{_base}_zoom_{ts}.png"
    plt.savefig(out_path_zoom, bbox_inches="tight")
    print(f"Likelihood slice (zoomed) saved to: {out_path_zoom}")

    ax.set_ylim(-_PROFILE_CI_THRESHOLD * 0.10, _PROFILE_CI_THRESHOLD)
    ax.set_title(f'Likelihood Slice (y-axis -10% to +100% CI)')
    out_path_zoom2 = f"{_base}_zoom2_{ts}.png"
    plt.savefig(out_path_zoom2, bbox_inches="tight")
    print(f"Likelihood slice (zoomed2) saved to: {out_path_zoom2}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Original flat-dict entry point
# ---------------------------------------------------------------------------

def setup_optimization(settings, optimization_settings, experiment_dict):
    """
    Run parameter optimization using a flat dict of experiment treatments.

    experiment_dict keys
    --------------------
    "experiment" : dict mapping experiment id -> treatment dict
    "plot"       : callable(paths, results_dict)
    """
    MODEL_NAME = settings.get("MODEL_NAME", AntiGen_paths.MODEL_NAME)
    model_text, paths = AntimonyGen(MODEL_NAME, repo_root=REPO_ROOT)

    if settings.get("run_steady_state_first"):
        _run_steady_state(model_text, paths, settings)
        model_text, paths = AntimonyGen(MODEL_NAME, repo_root=REPO_ROOT)

    param_names = optimization_settings["param_names"]
    x0          = optimization_settings["x0"]
    bounds      = optimization_settings.get("bounds")
    method      = optimization_settings.get("method", "Nelder-Mead")
    opt_kwargs  = optimization_settings.get("optimizer_kwargs", {})

    if not param_names:
        print("Error: No parameters to optimize. Set param_names and x0 in optimization_settings.")
        return

    experiments   = experiment_dict['experiment']
    plot_function = experiment_dict["plot"]

    opt = run_optimization(
        model_text, paths, experiments,
        param_names=param_names,
        x0=x0,
        bounds=bounds,
        loss_config={"observables": optimization_settings["observables"]},
        wald_analysis=optimization_settings.get("wald_analysis", False),
        slice_analysis=optimization_settings.get("slice_analysis", False),
        profile_likelihood_analysis=optimization_settings.get("profile_likelihood_analysis", False),
        fast_profile_likelihood_analysis=optimization_settings.get("fast_profile_likelihood_analysis", False),
        sobol_analysis=optimization_settings.get("sobol_analysis", False),
        sobol_kwargs={"N": optimization_settings.get("sobol_N", 128), "mode": optimization_settings.get("sobol_mode", "loss")},
        method=method,
        optimizer_kwargs=opt_kwargs,
        fit_mode=settings.get("fit_mode"),
    )
    print(f"Optimization success: {opt['success']}  loss: {opt['fun']:.6g}")

    csv_path = os.path.join(paths["plot_path"], f"{MODEL_NAME}_optimization_results.csv")
    log_optimization_results(opt, param_names, csv_path,
                             model_name=MODEL_NAME, experiment_id="ALL", method=method)

    if opt.get("results_dict") is not None:
        plot_function(paths, opt["results_dict"])

    if optimization_settings.get("slice_analysis") and opt["stats"].get("likelihood_slice"):
        _save_likelihood_slice_plot(opt, param_names, paths["plot_path"], MODEL_NAME)
    if (optimization_settings.get("profile_likelihood_analysis") or optimization_settings.get("fast_profile_likelihood_analysis")) and opt["stats"].get("profile_likelihood"):
        _save_profile_likelihood_plot(
            opt, param_names, paths["plot_path"], MODEL_NAME,
            profile_kwargs=_profile_kwargs(optimization_settings))
    if optimization_settings.get("sobol_analysis") and opt["stats"].get("sobol"):
        from Engine.Sensitivity_analysis import save_sobol_plot
        save_sobol_plot(opt["stats"]["sobol"], paths["plot_path"], MODEL_NAME)


# ---------------------------------------------------------------------------
# Group-aware entry point
# ---------------------------------------------------------------------------

def _is_per_group_settings(optimization_settings):
    """Return True when optimization_settings contains per-group sub-dicts
    (e.g. the PK block keyed by drug name), False for a flat shared dict."""
    return any(
        isinstance(v, dict) and "param_names" in v
        for v in optimization_settings.values()
    )


def setup_optimization_from_groups(settings, optimization_settings, EXPERIMENT_dict):
    """
    Run parameter optimization using Experiment.opt_groups.

    Loss_config is read from each replicate's ``Loss_config`` key.
    Replicates whose Loss_config is ``no_optimization()`` are simulated at the
    end with the optimal parameters for use by the plot function.

    Flat mode  — optimization_settings has top-level param_names/x0/bounds:
        one optimization is run, summing NLL across all active groups.

    Per-group mode — optimization_settings has per-group sub-dicts each
        containing param_names/x0/bounds (e.g. the PK block keyed by drug):
        one independent optimization is run per group, results accumulated,
        then plot_function is called once.
    """
    MODEL_NAME = settings.get("MODEL_NAME", AntiGen_paths.MODEL_NAME)
    model_text, paths = AntimonyGen(MODEL_NAME, repo_root=REPO_ROOT)

    if settings.get("run_steady_state_first"):
        _run_steady_state(model_text, paths, settings)
        model_text, paths = AntimonyGen(MODEL_NAME, repo_root=REPO_ROOT)

    experiment    = EXPERIMENT_dict["EXPERIMENT"]
    plot_function = EXPERIMENT_dict["plot"]

    # Support for the new decoupled, nested Optimization spec
    from Modules.Optimizer_settings import Optimization
    if isinstance(optimization_settings, Optimization):
        opt = run_optimization_from_groups(
            model_text, paths, experiment,
            param_names=optimization_settings.param_names,
            x0=optimization_settings.x0,
            bounds=optimization_settings.bounds,
            method=optimization_settings.method,
            optimizer_kwargs=optimization_settings.optimizer_kwargs,
            wald_analysis=settings.get("wald_analysis", False),
            slice_analysis=settings.get("slice_analysis", False),
            profile_likelihood_analysis=settings.get("profile_likelihood_analysis", False),
            fast_profile_likelihood_analysis=settings.get("fast_profile_likelihood_analysis", False),
            sobol_analysis=settings.get("sobol_analysis", False),
            sobol_kwargs={"N": settings.get("sobol_N", 128), "mode": settings.get("sobol_mode", "loss")},
            optimization_spec=optimization_settings,
            fit_mode=settings.get("fit_mode"),
            n_workers=settings.get("n_workers"),
            profile_checkpoint=settings.get("profile_checkpoint", True),
            preequil_cache=settings.get("preequil_cache", True),
            reuse_fit=settings.get("reuse_fit", True),
        )

        groups_tag = "_".join(opt.get("groups", ["ALL"]))

        if settings.get("slice_analysis") and opt.get("stats", {}).get("likelihood_slice"):
            _save_likelihood_slice_plot(opt, optimization_settings.param_names, paths["plot_path"],
                                        MODEL_NAME, tag=groups_tag)
        if (settings.get("profile_likelihood_analysis") or settings.get("fast_profile_likelihood_analysis")) and opt.get("stats", {}).get("profile_likelihood"):
            _save_profile_likelihood_plot(
                opt, optimization_settings.param_names, paths["plot_path"],
                MODEL_NAME, tag=groups_tag,
                profile_kwargs=_profile_kwargs(settings, optimization_settings))
        if settings.get("fast_profile_likelihood_analysis") and opt.get("stats", {}).get("fast_profile_all"):
            _run_fast_profile_report(opt, settings, optimization_settings)
        if settings.get("sobol_analysis") and opt.get("stats", {}).get("sobol"):
            from Engine.Sensitivity_analysis import save_sobol_plot
            save_sobol_plot(opt["stats"]["sobol"], paths["plot_path"], MODEL_NAME, tag=groups_tag)

        csv_path = os.path.join(
            paths["plot_path"],
            f"{MODEL_NAME}_{groups_tag}_optimization_results.csv",
        )
        log_optimization_results(opt, optimization_settings.param_names, csv_path,
                                 model_name=MODEL_NAME, experiment_id=groups_tag, method=optimization_settings.method)

        if opt.get("results_dict") is not None and plot_function:
            plot_function(paths, opt["results_dict"])
        _shutdown_evaluator(opt)
        return opt

    if _is_per_group_settings(optimization_settings):
        # ── Per-group mode ────────────────────────────────────────────────
        all_results = {}
        group_optimizations = {}
        allowed_groups = optimization_settings.get("group_names")  # None = run all
        # Replicates whose Opt_group is not among any per-group sub-dict key
        # are treated as passive (plot-only) and kept in all_results regardless
        # of which sub-group is being optimized.
        known_opt_groups = {
            name for name, val in optimization_settings.items()
            if isinstance(val, dict) and val.get("param_names")
        }
        for group_name, group_settings in optimization_settings.items():
            if not isinstance(group_settings, dict) or not group_settings.get("param_names"):
                continue
            if allowed_groups is not None and group_name not in allowed_groups:
                continue
            param_names = group_settings["param_names"]
            method      = group_settings.get("method", "Nelder-Mead")

            try:
                opt = run_optimization_from_groups(
                    model_text, paths, experiment,
                    param_names=param_names,
                    x0=group_settings["x0"],
                    bounds=group_settings.get("bounds"),
                    group_names=[group_name],
                    method=method,
                    optimizer_kwargs=group_settings.get("optimizer_kwargs", {}),
                    wald_analysis=group_settings.get("wald_analysis", False),
                    slice_analysis=group_settings.get("slice_analysis", False),
                    profile_likelihood_analysis=group_settings.get("profile_likelihood_analysis", False),
                    fast_profile_likelihood_analysis=group_settings.get("fast_profile_likelihood_analysis", False),
                    sobol_analysis=group_settings.get("sobol_analysis", False),
                    sobol_kwargs={"N": group_settings.get("sobol_N", 128), "mode": group_settings.get("sobol_mode", "loss")},
                    fit_mode=group_settings.get("fit_mode", settings.get("fit_mode")),
                    n_workers=group_settings.get("n_workers", settings.get("n_workers")),
                )
            except Exception as e:
                print(f"Warning: optimization for '{group_name}' failed: {e}")
                continue

            if opt.get("results_dict"):
                filtered_results = {}
                for req_id, item in opt["results_dict"].items():
                    item_group = item.get("replicate", {}).get("Opt_group")
                    if item_group == group_name or item_group not in known_opt_groups:
                        filtered_results[req_id] = item
                all_results.update(filtered_results)

            group_optimizations[group_name] = opt

            # Run analysis plots first so profile_ci is populated before CSV write
            if group_settings.get("slice_analysis") and opt.get("stats", {}).get("likelihood_slice"):
                _save_likelihood_slice_plot(opt, param_names, paths["plot_path"],
                                            MODEL_NAME, tag=group_name)
            if (group_settings.get("profile_likelihood_analysis") or group_settings.get("fast_profile_likelihood_analysis")) and opt.get("stats", {}).get("profile_likelihood"):
                _save_profile_likelihood_plot(
                    opt, param_names, paths["plot_path"], MODEL_NAME,
                    tag=group_name,
                    profile_kwargs=_profile_kwargs(group_settings))
            if group_settings.get("sobol_analysis") and opt.get("stats", {}).get("sobol"):
                from Engine.Sensitivity_analysis import save_sobol_plot
                save_sobol_plot(opt.get("stats", {}).get("sobol"), paths["plot_path"],
                                MODEL_NAME, tag=group_name)

            print(f"\nOptimization Summary for {group_name}:")
            print("-" * 85)
            print(f"{'Parameter':<30} | {'Optimized Value':<15} | {'Wald SE':<15} | {'Wald 95% CI':<20}")
            print("-" * 85)
            x_vals = opt.get("x", [])
            se = opt.get("stats", {}).get("wald_se")
            ci = opt.get("stats", {}).get("wald_ci")
            for i, p_name in enumerate(param_names):
                val = x_vals[i] if i < len(x_vals) else float('nan')
                std_err = se[i] if se is not None and i < len(se) else "N/A"
                std_err_str = f"{std_err:.4g}" if isinstance(std_err, (int, float)) else std_err
                conf_int = ci[i] if ci is not None and i < len(ci) else ("N/A", "N/A")
                if isinstance(conf_int, tuple) and len(conf_int) == 2:
                    if isinstance(conf_int[0], (int, float)) and isinstance(conf_int[1], (int, float)):
                        conf_int_str = f"[{conf_int[0]:.4g}, {conf_int[1]:.4g}]"
                    else:
                        conf_int_str = f"[{conf_int[0]}, {conf_int[1]}]"
                else:
                    conf_int_str = str(conf_int)
                print(f"{p_name:<30} | {val:<15.4g} | {std_err_str:<15} | {conf_int_str:<20}")
            print("-" * 85)
            print(f"Final Objective Value (NLL): {opt.get('fun', 'N/A'):.6g}\n")

            corr_matrix = opt.get("stats", {}).get("wald_correlation")
            if corr_matrix is not None:
                print(f"\nWald Correlation Matrix:")
                print("-" * 85)
                header_str = f"{'':<25} | " + " | ".join(f"{p[:10]:<10}" for p in param_names)
                print(header_str)
                print("-" * 85)
                for i, p_row in enumerate(param_names):
                    row_str = f"{p_row[:25]:<25} | "
                    row_vals = []
                    for j in range(len(param_names)):
                        if i < len(corr_matrix) and j < len(corr_matrix[i]):
                            row_vals.append(f"{corr_matrix[i][j]:<10.4g}")
                        else:
                            row_vals.append(f"{'N/A':<10}")
                    row_str += " | ".join(row_vals)
                    print(row_str)
                print("-" * 85)
                print()

            csv_path = os.path.join(
                paths["plot_path"],
                f"{MODEL_NAME}_{group_name}_optimization_results.csv",
            )
            log_optimization_results(opt, param_names, csv_path,
                                     model_name=MODEL_NAME, experiment_id=group_name,
                                     method=method)

        if optimization_settings.get("petab_export"):
            _write_petab_archive(paths, MODEL_NAME, experiment,
                                 optimization_settings, group_optimizations)

        if all_results and plot_function:
            plot_function(paths, all_results)
        for _o in group_optimizations.values():
            _shutdown_evaluator(_o)
        return group_optimizations

    else:
        # ── Flat (shared) mode ────────────────────────────────────────────
        param_names = optimization_settings["param_names"]
        x0          = optimization_settings["x0"]
        bounds      = optimization_settings.get("bounds")
        method      = optimization_settings.get("method", "Nelder-Mead")
        opt_kwargs  = optimization_settings.get("optimizer_kwargs", {})
        group_names = optimization_settings.get("group_names", None)

        if not param_names:
            print("Error: No parameters to optimize. Set param_names and x0 in optimization_settings.")
            return

        opt = run_optimization_from_groups(
            model_text, paths, experiment,
            param_names=param_names,
            x0=x0,
            bounds=bounds,
            group_names=group_names,
            method=method,
            optimizer_kwargs=opt_kwargs,
            wald_analysis=optimization_settings.get("wald_analysis", False),
            slice_analysis=optimization_settings.get("slice_analysis", False),
            profile_likelihood_analysis=optimization_settings.get("profile_likelihood_analysis", False),
            fast_profile_likelihood_analysis=optimization_settings.get("fast_profile_likelihood_analysis", False),
            sobol_analysis=optimization_settings.get("sobol_analysis", False),
            sobol_kwargs={"N": optimization_settings.get("sobol_N", 128), "mode": optimization_settings.get("sobol_mode", "loss")},
            fit_mode=optimization_settings.get("fit_mode", settings.get("fit_mode")),
            n_workers=optimization_settings.get("n_workers", settings.get("n_workers")),
        )

        groups_tag = "_".join(opt.get("groups", ["ALL"]))

        # Run analysis plots first so profile_ci is populated before CSV write
        if optimization_settings.get("slice_analysis") and opt["stats"].get("likelihood_slice"):
            _save_likelihood_slice_plot(opt, param_names, paths["plot_path"],
                                        MODEL_NAME, tag=groups_tag)
        if (optimization_settings.get("profile_likelihood_analysis") or optimization_settings.get("fast_profile_likelihood_analysis")) and opt["stats"].get("profile_likelihood"):
            _save_profile_likelihood_plot(
                opt, param_names, paths["plot_path"], MODEL_NAME,
                tag=groups_tag,
                profile_kwargs=_profile_kwargs(optimization_settings))
        if optimization_settings.get("sobol_analysis") and opt["stats"].get("sobol"):
            from Engine.Sensitivity_analysis import save_sobol_plot
            save_sobol_plot(opt["stats"]["sobol"], paths["plot_path"],
                            MODEL_NAME, tag=groups_tag)

        csv_path = os.path.join(
            paths["plot_path"],
            f"{MODEL_NAME}_{groups_tag}_optimization_results.csv",
        )
        log_optimization_results(opt, param_names, csv_path,
                                 model_name=MODEL_NAME, experiment_id=groups_tag, method=method)

        if optimization_settings.get("petab_export"):
            _write_petab_archive(paths, MODEL_NAME, experiment,
                                 optimization_settings, {"__flat__": opt})

        if opt.get("results_dict") is not None and plot_function:
            plot_function(paths, opt["results_dict"])
        _shutdown_evaluator(opt)
        return opt


def _write_petab_archive(paths, model_name, experiment,
                         optimization_settings, group_optimizations):
    """Write a PEtab v2 archive to results/<model>/petab/<expid>/.

    ``expid`` is built from the sorted union of optimized group names so
    successive runs against different groups land in distinct subdirs.
    """
    group_keys = sorted(group_optimizations.keys()) or ["ALL"]
    expid = "_".join(_sanitize_petab_id(g) for g in group_keys)
    out_dir = os.path.join(paths["plot_path"], "petab", expid)

    model_file_abs = os.path.join(
        paths["repo_root"], "antimony_models", model_name, f"{model_name}.txt",
    )
    if not os.path.exists(model_file_abs):
        model_file_rel = f"{model_name}.txt"
    else:
        model_file_rel = os.path.relpath(model_file_abs, out_dir).replace("\\", "/")

    try:
        export_petab(
            out_dir=out_dir,
            model_name=model_name,
            experiment=experiment,
            optimization_settings=optimization_settings,
            group_optimizations=group_optimizations,
            data_path=paths["data_path"],
            model_file_rel=model_file_rel,
        )
    except Exception as exc:
        print(f"[petab] Export failed: {exc}")


def _sanitize_petab_id(s):
    import re as _re
    out = _re.sub(r'[^A-Za-z0-9_]', '_', str(s))
    return out or "id"

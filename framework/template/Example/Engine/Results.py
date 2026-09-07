"""
Append optimization results as a single row to a CSV file, and write a
per-run JSON snapshot alongside it.

CSV columns written:
  timestamp, model_name, experiment_id, method, success, optimizer_message,
  total_loss, aic, bic,
  {param}                    – optimized value for each parameter
  wald_SE_{param}            – Wald standard error (NaN when Hessian is not PD)
  wald_CI95_lower_{param}    – lower bound of Wald 95% CI
  wald_CI95_upper_{param}    – upper bound of Wald 95% CI
  wald_corr_{a}_{b}          – Wald off-diagonal correlation for every pair a < b
  profile_CI95_lower_{param} – lower bound of profile likelihood 95% CI
  profile_CI95_upper_{param} – upper bound of profile likelihood 95% CI
  profile_CI95_status_{param} – ok / flat / open / open_lower / open_upper
  profile_reach_{side}_{param} – why that side stopped: "crossed" (a bound was
                               found), "bound" (walked to the parameter's own
                               limit without dNLL reaching 1.9207, so it is
                               unidentifiable everywhere it is allowed to go) or
                               "budget" (the outward extension ran out of steps,
                               so nothing was established either way)
  profile_maxdNLL_{side}_{param} – highest dNLL reached on that side
  profile_capped_{param}     – profile points whose nuisance optimization hit
                               the iteration cap; non-zero means that CI is
                               too narrow
  profile_points_not_converged – the same count summed over all parameters
  profile_warm_improved      – points the warm-started continuation pass
                               lowered; non-zero means the cold grid alone
                               would have given narrower intervals
  profile_warm_nats_recovered – total NLL recovered by that pass

JSON file (one per run, timestamped to avoid overwrite):
  {base}_{YYYYMMDD_HHMMSS}.json   alongside the CSV, with
    metadata          – timestamp, model_name, experiment_id, method, success,
                        message, total_loss, aic, bic, n_iter, n_fev
    parameters        – {name: value} pairs, registry-shaped for direct copy
                        into Modules/utils/*_registry.py
    wald_se           – {name: SE}
    wald_ci95         – {name: [lo, hi]}
    profile_ci95      – {name: [lo, hi]}
    wald_correlation  – {"a|b": corr} for every off-diagonal pair (a < b)
"""

import json
import os
from datetime import datetime

import numpy as np
import pandas as pd


def _finite_or_none(x):
    """Convert NaN/inf to None so the value is valid strict JSON."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def _write_results_row(df_row, csv_path):
    """Append one results row, reconciling the header instead of assuming it.

    A plain ``mode="a"`` append writes the row's values in the row's own column
    order under whatever header the file already carries, so the moment the two
    disagree every value lands in whichever column happens to occupy its
    position. That is not hypothetical: a Lecanemab CSV whose header still held
    an older parameter set stored CLrecycle_Tissue's confidence interval under
    ``profile_CI95_lower_CLup_Tissue``, and nothing in the file said so. Any run
    that changes the parameter list -- or adds a diagnostic column, as the
    profile reach columns do -- hits this.

    When the columns match exactly the append is unchanged. When they do not,
    the file is rewritten over the union: older rows keep their own columns,
    this row keeps its, and the gaps are empty. Empty is honest; a positional
    append made them wrong instead.
    """
    if not os.path.exists(csv_path):
        df_row.to_csv(csv_path, index=False)
        return

    try:
        df_old = pd.read_csv(csv_path)
    except Exception as exc:
        # Unreadable is not a reason to overwrite: whatever is in there is the
        # only copy of the earlier runs, and a file this far gone cannot be
        # repaired without guessing which column each orphaned value belonged
        # to. Append so nothing is lost, and say plainly what is wrong -- rows
        # with differing field counts are what earlier appends under a stale
        # header produced, and only a fresh file gets out of that state.
        print(f"  [results] WARNING: {os.path.basename(csv_path)} cannot be "
              f"parsed ({exc}).")
        print(f"    Its rows do not all have the same number of columns, which "
              f"is what appending under a stale header produces. The values "
              f"already in it may be filed under the wrong column names.")
        print(f"    This row is being appended so nothing is lost, but rename "
              f"or archive that file to start a clean one.")
        df_row.to_csv(csv_path, mode="a", header=False, index=False)
        return

    if list(df_old.columns) == list(df_row.columns):
        df_row.to_csv(csv_path, mode="a", header=False, index=False)
        return

    added = [c for c in df_row.columns if c not in df_old.columns]
    dropped = [c for c in df_old.columns if c not in df_row.columns]
    print(f"  [results] column set changed ({len(added)} added, "
          f"{len(dropped)} no longer written); rewriting "
          f"{os.path.basename(csv_path)} so earlier rows keep their columns.")
    pd.concat([df_old, df_row], ignore_index=True).to_csv(csv_path, index=False)


def log_optimization_results(
    opt,
    param_names,
    csv_path,
    model_name="",
    experiment_id="",
    method="",
):
    """
    Append one row of optimization results to *csv_path*.

    Parameters
    ----------
    opt : dict
        Return value of ``run_optimization()``.  Keys used:
        ``x``, ``fun``, ``success``, ``message``, ``stats``.
    param_names : list[str]
        Parameter names in the same order as ``opt["x"]``.
    csv_path : str
        Absolute path to the target CSV file.  Created with a header on the
        first call; subsequent calls append without writing the header again.
    model_name : str, optional
    experiment_id : str, optional
    method : str, optional
    """
    stats = opt.get("stats", {})

    # ------------------------------------------------------------------ #
    # Build the row as an ordered dict so column order is deterministic.  #
    # ------------------------------------------------------------------ #
    row = {}

    # --- bookkeeping --------------------------------------------------- #
    row["timestamp"]          = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row["model_name"]         = model_name
    row["experiment_id"]      = experiment_id
    row["method"]             = method

    # --- core optimization results ------------------------------------- #
    row["success"]            = opt.get("success", False)
    row["optimizer_message"]  = str(opt.get("message", ""))
    row["total_loss"]         = float(opt.get("fun", float("nan")))
    row["nll_proper"]         = float(stats.get("nll_proper", float("nan")))

    # --- information criteria ------------------------------------------ #
    row["aic"] = float(stats.get("aic", float("nan")))
    row["bic"] = float(stats.get("bic", float("nan")))

    # --- parameter values ---------------------------------------------- #
    x_opt = np.atleast_1d(opt.get("x", []))
    for name, val in zip(param_names, x_opt):
        row[name] = float(val)

    # --- Wald standard errors ------------------------------------------ #
    se = stats.get("wald_se")
    se_arr = np.atleast_1d(se) if se is not None else [float("nan")] * len(param_names)
    for name, s in zip(param_names, se_arr):
        row[f"wald_SE_{name}"] = float(s) if s is not None and not np.isnan(float(s)) else float("nan")

    # --- Wald 95% confidence intervals --------------------------------- #
    ci = stats.get("wald_ci")
    if ci is None:
        ci = [(float("nan"), float("nan"))] * len(param_names)
    for name, (lo, hi) in zip(param_names, ci):
        row[f"wald_CI95_lower_{name}"] = float(lo)
        row[f"wald_CI95_upper_{name}"] = float(hi)

    # --- Wald off-diagonal correlations -------------------------------- #
    corr = stats.get("wald_correlation")
    if corr is not None:
        corr = np.atleast_2d(corr)
        for i, a in enumerate(param_names):
            for j, b in enumerate(param_names):
                if j > i:
                    val = corr[i, j]
                    row[f"wald_corr_{a}_{b}"] = float(val) if np.isfinite(val) else float("nan")
    else:
        for i, a in enumerate(param_names):
            for j, b in enumerate(param_names):
                if j > i:
                    row[f"wald_corr_{a}_{b}"] = float("nan")

    # --- Profile likelihood 95% confidence intervals ------------------- #
    profile_ci = stats.get("profile_ci")
    if profile_ci is None:
        profile_ci = [(float("nan"), float("nan"))] * len(param_names)
    ci_status = stats.get("profile_ci_status") or ["missing"] * len(param_names)
    for name, (lo, hi) in zip(param_names, profile_ci):
        row[f"profile_CI95_lower_{name}"] = float(lo)
        row[f"profile_CI95_upper_{name}"] = float(hi)
    # A bare nan cannot distinguish "flat / non-identifiable" from "grid too
    # narrow", and those call for opposite fixes.
    for name, st in zip(param_names, ci_status):
        row[f"profile_CI95_status_{name}"] = st

    # Why each side stopped: "crossed" (a bound was found), "bound" (the profile
    # walked to the parameter's own limit without dNLL reaching 1.9207, so it is
    # unidentifiable everywhere it is allowed to go), or "budget" (the outward
    # extension ran out of steps, so nothing has been established either way).
    # The middle case is a result and the last is an unfinished run; a status of
    # "open" alone cannot tell them apart, and only one of them is worth
    # re-running with a wider grid.
    _reach = (stats.get("profile_convergence") or {}).get("reach") or {}
    for name in param_names:
        sides = _reach.get(name) or {}
        for key in ("lower", "upper"):
            d = sides.get(key) or {}
            row[f"profile_reach_{key}_{name}"] = d.get("state", "missing")
            row[f"profile_maxdNLL_{key}_{name}"] = float(
                d.get("max_dnll") if d.get("max_dnll") is not None else float("nan"))

    # Nuisance optimizations that stopped on the iteration cap rather than
    # converging. Those points overstate the profile, so the CI above them is
    # too narrow -- the caveat has to travel with the interval into the CSV, not
    # live only in the console log of the run that produced it.
    _conv = stats.get("profile_convergence") or {}
    _per_param = _conv.get("per_param", {})
    row["profile_points_not_converged"] = float(_conv.get("n_not_converged", 0))

    # What the warm-started continuation pass recovered. Non-zero means the
    # cold-started grid alone would have reported narrower intervals than these,
    # which is the bias that pass measures and removes.
    _warm = _conv.get("warm") or {}
    row["profile_warm_improved"] = float(_warm.get("n_improved", 0))
    row["profile_warm_nats_recovered"] = float(_warm.get("nats_recovered", 0.0))
    for name in param_names:
        row[f"profile_capped_{name}"] = float(
            (_per_param.get(name) or {}).get("n_not_converged", 0)
        )

    # How far below the reported optimum the profile got. With one shared
    # objective anything materially negative is a fit convergence failure.
    row["profile_anchor_gap"] = float(stats.get("profile_anchor_gap", 0.0))
    row["k_effective"]        = float(stats.get("k_effective", float("nan")))
    row["n_data_points"]      = float(stats.get("n_data_points", float("nan")))

    # ------------------------------------------------------------------ #
    # Console summary                                                      #
    # ------------------------------------------------------------------ #
    nit  = opt.get("nit")
    nfev = opt.get("nfev")
    iter_str = (f"  Iterations: {nit}  |  Func evals: {nfev}"
                if nit is not None else "")
    print(f'\n{"=" * 80}')
    print(f'OPTIMIZATION COMPLETE  [{experiment_id}]')
    print(f'{"=" * 80}')
    print(f'  Model:      {model_name}')
    print(f'  Method:     {method}')
    print(f'  Success:    {opt.get("success", False)}')
    print(f'  Message:    {opt.get("message", "")}')
    print(f'  Final loss: {float(opt.get("fun", float("nan"))):.6e}')
    if iter_str:
        print(iter_str)
    if param_names and len(x_opt) == len(param_names):
        print(f'\n  {"Parameter":<45} {"Value":>18}')
        print(f'  {"-" * 63}')
        for name, val in zip(param_names, x_opt):
            print(f'  {name:<45} {float(val):>18.8e}')
    print(f'{"=" * 80}\n')

    # ------------------------------------------------------------------ #
    # Append to CSV.                                                       #
    # ------------------------------------------------------------------ #
    df_row = pd.DataFrame([row])
    _write_results_row(df_row, csv_path)
    print(f"Optimization results appended to: {csv_path}")

    # ------------------------------------------------------------------ #
    # Per-run JSON snapshot. The "parameters" block is shaped like the    #
    # INDEPENDENT_*_REGISTRY dicts in Modules/utils/ so it can be copied   #
    # directly into the registry source.                                   #
    # ------------------------------------------------------------------ #
    ts_compact = datetime.now().strftime("%Y%m%d_%H%M%S")
    base, _ = os.path.splitext(csv_path)
    json_path = f"{base}_{ts_compact}.json"

    parameters_dict = {
        name: _finite_or_none(val) for name, val in zip(param_names, x_opt)
    }

    wald_se_dict = {
        name: _finite_or_none(s) for name, s in zip(param_names, se_arr)
    }
    wald_ci_dict = {
        name: [_finite_or_none(lo), _finite_or_none(hi)]
        for name, (lo, hi) in zip(param_names, ci)
    }
    profile_ci_dict = {
        name: [_finite_or_none(lo), _finite_or_none(hi)]
        for name, (lo, hi) in zip(param_names, profile_ci)
    }

    wald_corr_dict = {}
    if corr is not None:
        for i, a in enumerate(param_names):
            for j, b in enumerate(param_names):
                if j > i:
                    wald_corr_dict[f"{a}|{b}"] = _finite_or_none(corr[i, j])

    snapshot = {
        "metadata": {
            "timestamp":     row["timestamp"],
            "model_name":    model_name,
            "experiment_id": experiment_id,
            "method":        method,
            "success":       bool(opt.get("success", False)),
            "message":       str(opt.get("message", "")),
            # objective_kind names the function 'total_loss' and every dNLL
            # refer to. Archived runs where it is absent used a weighted
            # chi-square for the fit and a different frozen-sigma NLL for the
            # diagnostics, so their numbers are not comparable with these.
            "objective_kind": "concentrated_gaussian_nll",
            "total_loss":    _finite_or_none(opt.get("fun")),
            # Same objective as total_loss, plus the (n/2)(1+log 2pi) constant
            # the fit drops. Absolute, so AIC/BIC need it; it cancels from dNLL.
            "nll_proper":    _finite_or_none(stats.get("nll_proper")),
            "aic":           _finite_or_none(stats.get("aic")),
            "bic":           _finite_or_none(stats.get("bic")),
            # Parameters + one profiled-out sigma per block.
            "k_effective":   stats.get("k_effective"),
            "n_data_points": _finite_or_none(stats.get("n_data_points")),
            "profile_anchor_gap": _finite_or_none(stats.get("profile_anchor_gap")),
            "n_iterations":  opt.get("nit"),
            "n_fevals":      opt.get("nfev"),
            # Recorded so a run is reproducible: fit_mode says whether the
            # optimizer ran at all, and x0 pins down a randomized multi-start.
            "fit_mode":      opt.get("fit_mode"),
            # Path of the cached fit the optimum was reused from (a relaunch
            # of the same problem), or null when the optimizer ran in this
            # process.
            "fit_source":    opt.get("fit_source"),
        },
        "parameters":       parameters_dict,
        "wald_se":          wald_se_dict,
        "wald_ci95":        wald_ci_dict,
        "profile_ci95":     profile_ci_dict,
        "wald_correlation": wald_corr_dict,
    }

    if opt.get("x0") is not None:
        snapshot["x0"] = {
            name: _finite_or_none(val)
            for name, val in zip(param_names, np.atleast_1d(opt["x0"]))
        }
    if opt.get("parameter_scale") is not None:
        snapshot["parameter_scale"] = dict(zip(param_names, opt["parameter_scale"]))
    if stats.get("curvature_se"):
        snapshot["curvature_se"] = {
            name: _finite_or_none(val)
            for name, val in stats["curvature_se"].items()
        }

    if stats.get("profile_ci_status"):
        snapshot["profile_ci95_status"] = dict(
            zip(param_names, stats["profile_ci_status"])
        )
    if stats.get("block_sigmas"):
        # Fitted noise level per block, with its point count. Under the
        # concentrated likelihood a block's entire contribution is
        # (n/2)log(sigma^2), so these two numbers say which data the fit is
        # actually being driven by — and an outlying sigma is how a block the
        # model cannot fit announces itself.
        block_n = stats.get("block_n") or {}
        snapshot["block_sigmas"] = {
            k: {"sigma": _finite_or_none(v), "n": block_n.get(k)}
            for k, v in stats["block_sigmas"].items()
        }
    if stats.get("profile_convergence"):
        snapshot["profile_convergence"] = stats["profile_convergence"]
    if stats.get("profile_better_point"):
        snapshot["profile_better_point"] = stats["profile_better_point"]
    if stats.get("fast_profile"):
        # Per-side verdicts on "the interval is closed on this side", from
        # one capped profile point at the slice crossing. See Engine.Fast_profile.
        snapshot["fast_profile"] = stats["fast_profile"]
    if "profile_traces" in stats:
        snapshot["profile_traces"] = stats["profile_traces"]
    if "slice_traces" in stats:
        snapshot["slice_traces"] = stats["slice_traces"]

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2, allow_nan=False)
    print(f"Optimization snapshot written to: {json_path}")

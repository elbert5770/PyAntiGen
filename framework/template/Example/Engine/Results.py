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
    for name, (lo, hi) in zip(param_names, profile_ci):
        row[f"profile_CI95_lower_{name}"] = float(lo)
        row[f"profile_CI95_upper_{name}"] = float(hi)

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
        def _fmt_ci(pair):
            lo, hi = pair
            if np.isnan(lo) and np.isnan(hi):
                return "---"
            return f"[{lo:.4g}, {hi:.4g}]"

        print(f'\n  {"Parameter":<30} {"Value":>14} {"Wald SE":>10} '
              f'{"Wald 95% CI":>22} {"Profile 95% CI":>22}')
        print(f'  {"-" * 102}')
        for i, (name, val) in enumerate(zip(param_names, x_opt)):
            s = se_arr[i] if i < len(se_arr) else float("nan")
            se_str = f"{float(s):.4g}" if s is not None and np.isfinite(float(s)) else "---"
            wald_str = _fmt_ci(ci[i]) if i < len(ci) else "---"
            prof_str = _fmt_ci(profile_ci[i]) if i < len(profile_ci) else "---"
            print(f'  {name:<30} {float(val):>14.6g} {se_str:>10} '
                  f'{wald_str:>22} {prof_str:>22}')
        if any(np.isnan(lo) and np.isnan(hi) for lo, hi in profile_ci):
            has_profiles = "profile_traces" in stats
            if has_profiles:
                print("\n  NOTE: a profile 95% CI of '---' here means the profile never "
                      "crossed the\n  threshold on both sides — the parameter is not "
                      "identifiable from this fit.")
    print(f'{"=" * 80}\n')

    # ------------------------------------------------------------------ #
    # Append to CSV.                                                       #
    # ------------------------------------------------------------------ #
    df_row = pd.DataFrame([row])
    write_header = not os.path.exists(csv_path)
    df_row.to_csv(csv_path, mode="a", header=write_header, index=False)
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
            "total_loss":    _finite_or_none(opt.get("fun")),
            "nll_proper":    _finite_or_none(stats.get("nll_proper")),
            "aic":           _finite_or_none(stats.get("aic")),
            "bic":           _finite_or_none(stats.get("bic")),
            "n_iterations":  opt.get("nit"),
            "n_fevals":      opt.get("nfev"),
            # Recorded so a run is reproducible: fit_mode says whether the
            # optimizer ran at all, and x0 pins down a randomized multi-start.
            "fit_mode":      opt.get("fit_mode"),
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

    if "profile_traces" in stats:
        snapshot["profile_traces"] = stats["profile_traces"]
    if "slice_traces" in stats:
        snapshot["slice_traces"] = stats["slice_traces"]

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2, allow_nan=False)
    print(f"Optimization snapshot written to: {json_path}")

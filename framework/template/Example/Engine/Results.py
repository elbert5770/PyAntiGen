"""
Append optimization results as a single row to a CSV file.

Columns written:
  timestamp, model_name, experiment_id, method, success, optimizer_message,
  total_loss, aic, bic,
  {param}            – optimized value for each parameter
  SE_{param}         – standard error (NaN when Hessian is not PD)
  CI95_lower_{param} – lower bound of 95% confidence interval
  CI95_upper_{param} – upper bound of 95% confidence interval
  corr_{a}_{b}       – off-diagonal correlation for every pair a < b
"""

import os
from datetime import datetime

import numpy as np
import pandas as pd


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

    # --- information criteria ------------------------------------------ #
    row["aic"] = float(stats.get("aic", float("nan")))
    row["bic"] = float(stats.get("bic", float("nan")))

    # --- parameter values ---------------------------------------------- #
    x_opt = np.atleast_1d(opt.get("x", []))
    for name, val in zip(param_names, x_opt):
        row[name] = float(val)

    # --- standard errors ----------------------------------------------- #
    se = stats.get("standard_errors")
    se_arr = np.atleast_1d(se) if se is not None else [float("nan")] * len(param_names)
    for name, s in zip(param_names, se_arr):
        row[f"SE_{name}"] = float(s) if s is not None and not np.isnan(float(s)) else float("nan")

    # --- 95% confidence intervals -------------------------------------- #
    ci = stats.get("confidence_intervals")  # list of (lower, upper)
    if ci is None:
        ci = [(float("nan"), float("nan"))] * len(param_names)
    for name, (lo, hi) in zip(param_names, ci):
        row[f"CI95_lower_{name}"] = float(lo)
        row[f"CI95_upper_{name}"] = float(hi)

    # --- off-diagonal correlations ------------------------------------- #
    corr = stats.get("correlation_matrix")
    if corr is not None:
        corr = np.atleast_2d(corr)
        for i, a in enumerate(param_names):
            for j, b in enumerate(param_names):
                if j > i:
                    val = corr[i, j]
                    row[f"corr_{a}_{b}"] = float(val) if np.isfinite(val) else float("nan")
    else:
        for i, a in enumerate(param_names):
            for j, b in enumerate(param_names):
                if j > i:
                    row[f"corr_{a}_{b}"] = float("nan")

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
    write_header = not os.path.exists(csv_path)
    df_row.to_csv(csv_path, mode="a", header=write_header, index=False)
    print(f"Optimization results appended to: {csv_path}")

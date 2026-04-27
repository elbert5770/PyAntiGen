"""
High-level optimization setup.  Two entry points:

  setup_optimization(settings, optimization_settings, experiment_dict)
      Original flat-dict path: experiments is a plain dict of treatment dicts.

  setup_optimization_from_groups(settings, optimization_settings, EXPERIMENT_dict)
      New group-aware path: uses Experiment.optimization_groups to define which
      treatments contribute to the objective, with one model pre-built per
      treatment and Update_parameters applied before the optimizer starts.
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
    _plot_profile_likelihood,
)
from Modules.Plots import *
from Engine.Results import log_optimization_results


# ---------------------------------------------------------------------------
# Shared steady-state helper
# ---------------------------------------------------------------------------

def _run_steady_state(model_text, paths):
    """Compute steady state and update InitialConditions CSV."""
    ic_path = os.path.join(
        paths["repo_root"], "antimony_models",
        paths["MODEL_NAME"], f"{paths['MODEL_NAME']}_InitialConditions.csv",
    )
    rss = TelluriumGen(model_text, paths)
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


def _save_profile_likelihood_plot(opt, param_names, plot_path, model_name, tag="ALL"):
    """Plot and save profile likelihood if the closure is available."""
    import numpy as np
    import matplotlib.pyplot as plt

    profile_func = opt.get("stats", {}).get("profile_likelihood")
    if not profile_func:
        print("Warning: no profile_likelihood closure in opt['stats'] — skipping plot.")
        return

    # Diagnostic: print delta-NLL range for each parameter
    stats_dict   = opt.get("stats", {})
    nll_opt      = stats_dict.get("nll_at_optimum")
    fixed_sigmas = stats_dict.get("fixed_sigmas", {})

    print("Profile likelihood diagnostics:")
    if nll_opt is not None:
        print(f"  NLL at optimum (fixed-sigma): {nll_opt:.6g}")
    if not fixed_sigmas:
        print("  WARNING: fixed_sigmas is empty — no observables were resolved to data.")
    else:
        print(f"  fixed_sigmas ({len(fixed_sigmas)} entries):")
        for k_fs, v_fs in fixed_sigmas.items():
            print(f"    sigma[{k_fs}] = {v_fs:.4g}")

    for i, pname in enumerate(param_names):
        try:
            param_vals, nll_rel = profile_func(i, n_points=5, range_factor=2.0)
            abs_nll = nll_rel + nll_opt if nll_opt is not None else nll_rel
            print(f"  {pname}:")
            print(f"    param range : [{param_vals[0]:.4g}, {param_vals[-1]:.4g}]")
            print(f"    delta-NLL   : [{nll_rel.min():.4g}, {nll_rel.max():.4g}]")
            if nll_opt is not None:
                print(f"    abs-NLL     : [{abs_nll.min():.6g}, {abs_nll.max():.6g}]")
            if np.all(np.abs(nll_rel) < 1e-10):
                print(f"    *** FLAT: model may not respond to {pname} in nll_func_fixed ***")
        except Exception as exc:
            print(f"  {pname}: error — {exc}")

    fig, ax = plt.subplots(figsize=(8, 6))
    _plot_profile_likelihood(
        ax,
        {"parameters": param_names, "n_points": 20, "range_factor": 2.0},
        opt, opt["x"], param_names,
    )
    plt.tight_layout()
    out_path = os.path.join(plot_path, f"{model_name}_{tag}_profile_likelihood.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Profile likelihood saved to: {out_path}")


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
        _run_steady_state(model_text, paths)
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
        likelihood_analysis=optimization_settings.get("likelihood_analysis", False),
        method=method,
        optimizer_kwargs=opt_kwargs,
    )
    print(f"Optimization success: {opt['success']}  loss: {opt['fun']:.6g}")

    csv_path = os.path.join(paths["plot_path"], f"{MODEL_NAME}_optimization_results.csv")
    log_optimization_results(opt, param_names, csv_path,
                             model_name=MODEL_NAME, experiment_id="ALL", method=method)

    if opt.get("results_dict") is not None:
        plot_function(paths, opt["results_dict"])

    if optimization_settings.get("likelihood_analysis") and opt["stats"].get("profile_likelihood"):
        _save_profile_likelihood_plot(opt, param_names, paths["plot_path"], MODEL_NAME)


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
    Run parameter optimization using Experiment.optimization_groups.

    Loss_config is read from each group's ``Loss_config`` key in the
    experiment structure; it is no longer passed via optimization_settings.
    Groups whose Loss_config is ``no_optimization()`` are simulated at the
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
        _run_steady_state(model_text, paths)
        model_text, paths = AntimonyGen(MODEL_NAME, repo_root=REPO_ROOT)

    experiment    = EXPERIMENT_dict["EXPERIMENT"]
    plot_function = EXPERIMENT_dict["plot"]

    if _is_per_group_settings(optimization_settings):
        # ── Per-group mode ────────────────────────────────────────────────
        all_results = {}
        for group_name, group_settings in optimization_settings.items():
            if not isinstance(group_settings, dict) or not group_settings.get("param_names"):
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
                    likelihood_analysis=group_settings.get("likelihood_analysis", False),
                )
            except Exception as e:
                print(f"Warning: optimization for '{group_name}' failed: {e}")
                continue

            csv_path = os.path.join(
                paths["plot_path"],
                f"{MODEL_NAME}_{group_name}_optimization_results.csv",
            )
            log_optimization_results(opt, param_names, csv_path,
                                     model_name=MODEL_NAME, experiment_id=group_name,
                                     method=method)
            if opt.get("results_dict"):
                filtered_results = {}
                for req_id, item in opt["results_dict"].items():
                    if item.get("treatment", {}).get("Drug") == group_name:
                        filtered_results[req_id] = item
                all_results.update(filtered_results)
            
            print(f"\nOptimization Summary for {group_name}:")
            print("-" * 85)
            print(f"{'Parameter':<30} | {'Optimized Value':<15} | {'Std Error':<15} | {'95% CI':<20}")
            print("-" * 85)
            x_vals = opt.get("x", [])
            se = opt.get("stats", {}).get("standard_errors")
            ci = opt.get("stats", {}).get("confidence_intervals")
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
            
            corr_matrix = opt.get("stats", {}).get("correlation_matrix")
            if corr_matrix is not None:
                print(f"\nCorrelation Matrix:")
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
            
            if group_settings.get("likelihood_analysis") and opt.get("stats", {}).get("profile_likelihood"):
                _save_profile_likelihood_plot(opt, param_names, paths["plot_path"],
                                              MODEL_NAME, tag=group_name)

        if all_results:
            plot_function(paths, all_results)

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
            likelihood_analysis=optimization_settings.get("likelihood_analysis", False),
        )

        groups_tag = "_".join(opt.get("groups", ["ALL"]))
        csv_path = os.path.join(
            paths["plot_path"],
            f"{MODEL_NAME}_{groups_tag}_optimization_results.csv",
        )
        log_optimization_results(opt, param_names, csv_path,
                                 model_name=MODEL_NAME, experiment_id=groups_tag, method=method)

        if opt.get("results_dict") is not None:
            plot_function(paths, opt["results_dict"])

        if optimization_settings.get("likelihood_analysis") and opt["stats"].get("profile_likelihood"):
            _save_profile_likelihood_plot(opt, param_names, paths["plot_path"],
                                          MODEL_NAME, tag=groups_tag)



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


_PROFILE_CI_THRESHOLD = 1.9207  # chi2(df=1, p=0.95) / 2


def _save_profile_likelihood_plot(opt, param_names, plot_path, model_name, tag="ALL"):
    """Run profile likelihood once per parameter, extract CIs, and save plot.

    Stores profile_ci into opt['stats']['profile_ci'] so log_optimization_results
    can include it in the CSV when called afterwards.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    profile_func = opt.get("stats", {}).get("profile_likelihood")
    if not profile_func:
        print("Warning: no profile_likelihood closure in opt['stats'] — skipping plot.")
        return

    colors  = ['blue', 'green', 'red', 'orange', 'purple']
    markers = ['o', 's', '^', 'D', 'v']
    params_estimated = np.asarray(opt["x"])

    print("\nProfile likelihood:")
    cache = {}
    for i, pname in enumerate(param_names):
        try:
            pv, nr = profile_func(i)
            cache[i] = (pv, nr)
            print(f"  {pname}: Δ NLL range [{nr.min():.4g}, {nr.max():.4g}]")
            if np.all(np.abs(nr) < 1e-10):
                print(f"    *** FLAT: model may not respond to {pname} ***")
        except Exception as exc:
            print(f"  {pname}: error — {exc}")

    # Extract profile CIs and store so Results.py can write them to CSV
    profile_ci = []
    print("\n  Profile likelihood 95% CIs:")
    for i, pname in enumerate(param_names):
        cached = cache.get(i)
        if cached is not None:
            lo, hi = _extract_profile_ci(cached[0], cached[1], threshold=_PROFILE_CI_THRESHOLD)
        else:
            lo, hi = float('nan'), float('nan')
        profile_ci.append((lo, hi))
        print(f"    {pname}: [{lo:.4g}, {hi:.4g}]")
    opt["stats"]["profile_ci"] = profile_ci

    fig, ax = plt.subplots(figsize=(8, 6))
    for idx, pname in enumerate(param_names):
        cached = cache.get(idx)
        if cached is None:
            continue
        pv, nr = cached
        ax.plot(pv / params_estimated[idx], nr,
                marker=markers[idx % len(markers)], linestyle='-',
                label=pname, linewidth=2, color=colors[idx % len(colors)])

    ax.axhline(_PROFILE_CI_THRESHOLD, color='gray', linestyle=':', alpha=0.7, linewidth=1.5,
               label=f'95% CI (Δ NLL = {_PROFILE_CI_THRESHOLD:.2f})')
    ax.axvline(1.0, color='red', linestyle='--', alpha=0.5, linewidth=1.5, label='Optimal')
    ax.set_xlabel('Parameter Value (relative to optimal)')
    ax.set_ylabel('Δ NLL (relative to minimum)')
    ax.set_title('Profile Likelihood (Identifiability Check)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.2, 3.5)
    plt.tight_layout()
    out_path = os.path.join(plot_path, f"{model_name}_{tag}_profile_likelihood.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Profile likelihood saved to: {out_path}")


def _save_likelihood_slice_plot(opt, param_names, plot_path, model_name, tag="ALL"):
    """Run likelihood slice once per parameter and save plot."""
    import numpy as np
    import matplotlib.pyplot as plt

    slice_func = opt.get("stats", {}).get("likelihood_slice")
    if not slice_func:
        print("Warning: no likelihood_slice closure in opt['stats'] — skipping plot.")
        return

    colors  = ['blue', 'green', 'red', 'orange', 'purple']
    markers = ['o', 's', '^', 'D', 'v']
    params_estimated = np.asarray(opt["x"])

    print("\nLikelihood slice:")
    cache = {}
    for i, pname in enumerate(param_names):
        try:
            pv, nr = slice_func(i, n_points=20, range_factor=2.0)
            cache[i] = (pv, nr)
            print(f"  {pname}: Δ NLL range [{nr.min():.4g}, {nr.max():.4g}]")
            if np.all(np.abs(nr) < 1e-10):
                print(f"    *** FLAT: model may not respond to {pname} ***")
        except Exception as exc:
            print(f"  {pname}: error — {exc}")

    fig, ax = plt.subplots(figsize=(8, 6))
    for idx, pname in enumerate(param_names):
        cached = cache.get(idx)
        if cached is None:
            continue
        pv, nr = cached
        ax.plot(pv / params_estimated[idx], nr,
                marker=markers[idx % len(markers)], linestyle='-',
                label=pname, linewidth=2, color=colors[idx % len(colors)])

    ax.axhline(_PROFILE_CI_THRESHOLD, color='gray', linestyle=':', alpha=0.7, linewidth=1.5,
               label=f'95% CI threshold (Δ NLL = {_PROFILE_CI_THRESHOLD:.2f})')
    ax.axvline(1.0, color='red', linestyle='--', alpha=0.5, linewidth=1.5, label='Optimal')
    ax.set_xlabel('Parameter Value (relative to optimal)')
    ax.set_ylabel('Δ NLL (relative to minimum)')
    ax.set_title('Likelihood Slice')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.2, 3.5)
    plt.tight_layout()
    out_path = os.path.join(plot_path, f"{model_name}_{tag}_likelihood_slice.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Likelihood slice saved to: {out_path}")


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
        wald_analysis=optimization_settings.get("wald_analysis", False),
        slice_analysis=optimization_settings.get("slice_analysis", False),
        profile_likelihood_analysis=optimization_settings.get("profile_likelihood_analysis", False),
        method=method,
        optimizer_kwargs=opt_kwargs,
    )
    print(f"Optimization success: {opt['success']}  loss: {opt['fun']:.6g}")

    csv_path = os.path.join(paths["plot_path"], f"{MODEL_NAME}_optimization_results.csv")
    log_optimization_results(opt, param_names, csv_path,
                             model_name=MODEL_NAME, experiment_id="ALL", method=method)

    if opt.get("results_dict") is not None:
        plot_function(paths, opt["results_dict"])

    if optimization_settings.get("slice_analysis") and opt["stats"].get("likelihood_slice"):
        _save_likelihood_slice_plot(opt, param_names, paths["plot_path"], MODEL_NAME)
    if optimization_settings.get("profile_likelihood_analysis") and opt["stats"].get("profile_likelihood"):
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
                    wald_analysis=group_settings.get("wald_analysis", False),
                    slice_analysis=group_settings.get("slice_analysis", False),
                    profile_likelihood_analysis=group_settings.get("profile_likelihood_analysis", False),
                )
            except Exception as e:
                print(f"Warning: optimization for '{group_name}' failed: {e}")
                continue

            if opt.get("results_dict"):
                filtered_results = {}
                for req_id, item in opt["results_dict"].items():
                    if item.get("replicate", {}).get("Opt_group") == group_name:
                        filtered_results[req_id] = item
                all_results.update(filtered_results)

            # Run analysis plots first so profile_ci is populated before CSV write
            if group_settings.get("slice_analysis") and opt.get("stats", {}).get("likelihood_slice"):
                _save_likelihood_slice_plot(opt, param_names, paths["plot_path"],
                                            MODEL_NAME, tag=group_name)
            if group_settings.get("profile_likelihood_analysis") and opt.get("stats", {}).get("profile_likelihood"):
                _save_profile_likelihood_plot(opt, param_names, paths["plot_path"],
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
            wald_analysis=optimization_settings.get("wald_analysis", False),
            slice_analysis=optimization_settings.get("slice_analysis", False),
            profile_likelihood_analysis=optimization_settings.get("profile_likelihood_analysis", False),
        )

        groups_tag = "_".join(opt.get("groups", ["ALL"]))

        # Run analysis plots first so profile_ci is populated before CSV write
        if optimization_settings.get("slice_analysis") and opt["stats"].get("likelihood_slice"):
            _save_likelihood_slice_plot(opt, param_names, paths["plot_path"],
                                        MODEL_NAME, tag=groups_tag)
        if optimization_settings.get("profile_likelihood_analysis") and opt["stats"].get("profile_likelihood"):
            _save_profile_likelihood_plot(opt, param_names, paths["plot_path"],
                                          MODEL_NAME, tag=groups_tag)

        csv_path = os.path.join(
            paths["plot_path"],
            f"{MODEL_NAME}_{groups_tag}_optimization_results.csv",
        )
        log_optimization_results(opt, param_names, csv_path,
                                 model_name=MODEL_NAME, experiment_id=groups_tag, method=method)

        if opt.get("results_dict") is not None:
            plot_function(paths, opt["results_dict"])



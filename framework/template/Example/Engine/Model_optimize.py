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
from Engine.Optimize import _run_pypesto_profile_single
from Engine.Results import log_optimization_results
from Engine.Petab_export import export_petab


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


def _save_profile_likelihood_plot(opt, param_names, plot_path, model_name, tag="ALL"):
    """Run profile likelihood once per parameter, extract CIs, and save plot.

    Parallelises across parameters automatically:
      - Linux / macOS : fork-based multiprocessing.Process (inherits closure, no pickling)
      - Windows       : ThreadPoolExecutor with per-thread independently-built models

    Stores profile_ci into opt['stats']['profile_ci'] so log_optimization_results
    can include it in the CSV when called afterwards.
    """
    import sys
    import numpy as np
    import matplotlib.pyplot as plt

    profile_func = opt.get("stats", {}).get("profile_likelihood")
    if not profile_func:
        print("Warning: no profile_likelihood closure in opt['stats'] — skipping plot.")
        return

    colors  = ['blue', 'green', 'red', 'orange', 'purple', 'brown']
    markers = ['o', 's', '^', 'D', 'v']
    params_estimated = np.asarray(opt["x"])
    n_params = len(param_names)

    print("\nProfile likelihood:")
    cache = {}

    _is_posix   = sys.platform != 'win32'
    _factory    = opt.get("stats", {}).get("nll_fixed_factory")
    _pstate     = opt.get("stats", {}).get("profile_state")
    _can_par    = n_params > 1

    if _can_par and _is_posix:
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
                print(f"  {param_names[i]}: Δ NLL range [{nr.min():.4g}, {nr.max():.4g}]")
                if np.all(np.abs(nr) < 1e-10):
                    print(f"    *** FLAT: model may not respond to {param_names[i]} ***")
            else:
                print(f"  {param_names[i]}: error — {err}")

    elif _can_par and _factory and _pstate:
        # ── Windows: thread pool with per-thread rebuilt model instances ──────
        # RoadRunner objects are not thread-safe; each thread calls _factory()
        # to get an independent set of models.  Model construction is sequential
        # (safe), only the pypesto optimisation runs concurrently.
        from concurrent.futures import ThreadPoolExecutor, as_completed

        res_x        = _pstate["res_x"]
        bounds       = _pstate["bounds"]
        nll_at_opt   = _pstate["nll_at_optimum"]

        print(f"  [parallel] building {n_params} model copies for thread pool …", flush=True)
        nll_funcs = [_factory() for _ in range(n_params)]   # sequential build

        import time as _time
        _t0 = _time.time()

        def _thread_worker(i):
            print(f"  [thread {i}] starting {param_names[i]} …", flush=True)
            result = _run_pypesto_profile_single(
                i, nll_funcs[i], bounds, res_x, nll_at_opt, param_names)
            elapsed = _time.time() - _t0
            print(f"  [thread {i}] done    {param_names[i]}  ({elapsed:.0f}s elapsed)", flush=True)
            return i, result

        print(f"  [parallel] launching {n_params} profile threads …", flush=True)
        with ThreadPoolExecutor(max_workers=n_params) as ex:
            futs = {ex.submit(_thread_worker, i): i for i in range(n_params)}
            for fut in as_completed(futs):
                try:
                    i, (pv, nr) = fut.result()
                    cache[i] = (pv, nr)
                    print(f"  {param_names[i]}: Δ NLL range [{nr.min():.4g}, {nr.max():.4g}]", flush=True)
                    if np.all(np.abs(nr) < 1e-10):
                        print(f"    *** FLAT: model may not respond to {param_names[i]} ***", flush=True)
                except Exception as exc:
                    print(f"  {param_names[futs[fut]]}: error — {exc}", flush=True)

    else:
        # ── Sequential fallback (single param, or Windows without factory) ────
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
    profile_traces = {}
    print("\n  Profile likelihood 95% CIs:")
    for i, pname in enumerate(param_names):
        cached = cache.get(i)
        if cached is not None:
            lo, hi = _extract_profile_ci(cached[0], cached[1], threshold=_PROFILE_CI_THRESHOLD)
            profile_traces[pname] = {"x": cached[0].tolist(), "y": cached[1].tolist()}
        else:
            lo, hi = float('nan'), float('nan')
        profile_ci.append((lo, hi))
        print(f"    {pname}: [{lo:.4g}, {hi:.4g}]")
    opt["stats"]["profile_ci"] = profile_ci
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
    ax.set_ylabel('Δ NLL (relative to minimum)')
    ax.set_title('Profile Likelihood (Identifiability Check)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.2, 3.5)
    plt.tight_layout()
    out_path = os.path.join(plot_path, f"{model_name}_{tag}_profile_likelihood.png")
    plt.savefig(out_path, bbox_inches="tight")
    print(f"Profile likelihood saved to: {out_path}")

    zoom_half = _PROFILE_CI_THRESHOLD * 0.10
    ax.set_ylim(-zoom_half, zoom_half)
    ax.set_title(f'Profile Likelihood (Identifiability Check, y-axis ±10% CI)')
    out_path_zoom = os.path.join(plot_path, f"{model_name}_{tag}_profile_likelihood_zoom.png")
    plt.savefig(out_path_zoom, bbox_inches="tight")
    print(f"Profile likelihood (zoomed) saved to: {out_path_zoom}")

    ax.set_ylim(-_PROFILE_CI_THRESHOLD * 0.10, _PROFILE_CI_THRESHOLD)
    ax.set_title(f'Profile Likelihood (Identifiability Check, y-axis -10% to +100% CI)')
    out_path_zoom2 = os.path.join(plot_path, f"{model_name}_{tag}_profile_likelihood_zoom2.png")
    plt.savefig(out_path_zoom2, bbox_inches="tight")
    print(f"Profile likelihood (zoomed2) saved to: {out_path_zoom2}")
    plt.close(fig)


def _save_likelihood_slice_plot(opt, param_names, plot_path, model_name, tag="ALL"):
    """Run likelihood slice once per parameter and save plot."""
    import numpy as np
    import matplotlib.pyplot as plt

    slice_func = opt.get("stats", {}).get("likelihood_slice")
    if not slice_func:
        print("Warning: no likelihood_slice closure in opt['stats'] — skipping plot.")
        return

    colors  = ['blue', 'green', 'red', 'orange', 'purple', 'brown']
    markers = ['o', 's', '^', 'D', 'v']
    params_estimated = np.asarray(opt["x"])

    print("\nLikelihood slice:")
    cache = {}
    slice_traces = {}
    for i, pname in enumerate(param_names):
        try:
            pv, nr = slice_func(i, n_points=20, range_factor=2.0)
            cache[i] = (pv, nr)
            slice_traces[pname] = {"x": pv.tolist(), "y": nr.tolist()}
            print(f"  {pname}: Δ NLL range [{nr.min():.4g}, {nr.max():.4g}]")
            if np.all(np.abs(nr) < 1e-10):
                print(f"    *** FLAT: model may not respond to {pname} ***")
        except Exception as exc:
            print(f"  {pname}: error — {exc}")
    opt["stats"]["slice_traces"] = slice_traces

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
    print(f"Likelihood slice saved to: {out_path}")

    zoom_half = _PROFILE_CI_THRESHOLD * 0.10
    ax.set_ylim(-zoom_half, zoom_half)
    ax.set_title(f'Likelihood Slice (y-axis ±10% CI)')
    out_path_zoom = os.path.join(plot_path, f"{model_name}_{tag}_likelihood_slice_zoom.png")
    plt.savefig(out_path_zoom, bbox_inches="tight")
    print(f"Likelihood slice (zoomed) saved to: {out_path_zoom}")

    ax.set_ylim(-_PROFILE_CI_THRESHOLD * 0.10, _PROFILE_CI_THRESHOLD)
    ax.set_title(f'Likelihood Slice (y-axis -10% to +100% CI)')
    out_path_zoom2 = os.path.join(plot_path, f"{model_name}_{tag}_likelihood_slice_zoom2.png")
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
        sobol_analysis=optimization_settings.get("sobol_analysis", False),
        sobol_kwargs={"N": optimization_settings.get("sobol_N", 128), "mode": optimization_settings.get("sobol_mode", "loss")},
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
            optimization_spec=optimization_settings,
        )

        groups_tag = "_".join(opt.get("groups", ["ALL"]))

        if settings.get("slice_analysis") and opt.get("stats", {}).get("likelihood_slice"):
            _save_likelihood_slice_plot(opt, optimization_settings.param_names, paths["plot_path"],
                                        MODEL_NAME, tag=groups_tag)
        if settings.get("profile_likelihood_analysis") and opt.get("stats", {}).get("profile_likelihood"):
            _save_profile_likelihood_plot(opt, optimization_settings.param_names, paths["plot_path"],
                                          MODEL_NAME, tag=groups_tag)

        csv_path = os.path.join(
            paths["plot_path"],
            f"{MODEL_NAME}_{groups_tag}_optimization_results.csv",
        )
        log_optimization_results(opt, optimization_settings.param_names, csv_path,
                                 model_name=MODEL_NAME, experiment_id=groups_tag, method=optimization_settings.method)

        if opt.get("results_dict") is not None and plot_function:
            plot_function(paths, opt["results_dict"])
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
                    sobol_analysis=group_settings.get("sobol_analysis", False),
                    sobol_kwargs={"N": group_settings.get("sobol_N", 128), "mode": group_settings.get("sobol_mode", "loss")},
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
            if group_settings.get("profile_likelihood_analysis") and opt.get("stats", {}).get("profile_likelihood"):
                _save_profile_likelihood_plot(opt, param_names, paths["plot_path"],
                                              MODEL_NAME, tag=group_name)
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

        if all_results:
            plot_function(paths, all_results)
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
            sobol_analysis=optimization_settings.get("sobol_analysis", False),
            sobol_kwargs={"N": optimization_settings.get("sobol_N", 128), "mode": optimization_settings.get("sobol_mode", "loss")},
        )

        groups_tag = "_".join(opt.get("groups", ["ALL"]))

        # Run analysis plots first so profile_ci is populated before CSV write
        if optimization_settings.get("slice_analysis") and opt["stats"].get("likelihood_slice"):
            _save_likelihood_slice_plot(opt, param_names, paths["plot_path"],
                                        MODEL_NAME, tag=groups_tag)
        if optimization_settings.get("profile_likelihood_analysis") and opt["stats"].get("profile_likelihood"):
            _save_profile_likelihood_plot(opt, param_names, paths["plot_path"],
                                          MODEL_NAME, tag=groups_tag)
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

        if opt.get("results_dict") is not None:
            plot_function(paths, opt["results_dict"])
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

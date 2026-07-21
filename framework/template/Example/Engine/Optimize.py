"""
Parameter optimization over multiple experiments: run each experiment with a
given parameter set, compare to data, return a scalar loss for the optimizer.
Also provides run_all for running experiments (with optional parameter injection).
"""

import os
import time
import numpy as np
import warnings
from scipy import stats
import numdifftools as nd

from framework.TelluriumGen import TelluriumGen
from Engine.Simulate import simulate
from Modules.Loss_config import no_optimization


# Persistent state for the live optimization-progress overlay plot. The figure
# is built lazily on the first trigger and reused across calls so the same
# image file is overwritten as the run proceeds.
_progress_overlay_state = {
    "fig":         None,
    "axes":        None,
    "keys":        None,
    "ncols":       None,
    "nrows":       None,
    "last_render": 0.0,
    "timestamp":   None,
}


def _render_progress_overlay(trace_collector, total_loss, best_loss, eval_n,
                             plot_path, min_interval_s=2.0,
                             param_names=None, param_values=None,
                             model_name=None, experiment_id=None, method=None):
    """Save a multi-panel overlay (model curve + data points) per (replicate,
    observable) traced during an optimization eval. Reuses one Figure and
    overwrites a single PNG so the file refreshes in place.
    """
    if not trace_collector or not plot_path:
        return

    now = time.time()
    is_improved = (best_loss is not None) and (total_loss <= best_loss)
    if not is_improved and (now - _progress_overlay_state["last_render"]) < min_interval_s:
        return

    opt_timestamp = _progress_overlay_state.get("timestamp")
    if not opt_timestamp:
        from datetime import datetime
        opt_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        _progress_overlay_state["timestamp"] = opt_timestamp

    if is_improved and param_names is not None and param_values is not None:
        import json
        from datetime import datetime
        
        parameters_dict = {}
        for name, val in zip(param_names, np.atleast_1d(param_values).tolist()):
            try:
                v = float(val)
                parameters_dict[name] = v if np.isfinite(v) else None
            except (TypeError, ValueError):
                parameters_dict[name] = None
        
        progress_json = {
            "metadata": {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "model_name": model_name or "",
                "experiment_id": experiment_id or "",
                "method": method or "",
                "success": False,
                "message": "Optimization in progress",
                "total_loss": float(total_loss) if np.isfinite(total_loss) else None,
                "nll_proper": None,
                "aic": None,
                "bic": None,
                "n_iterations": None,
                "n_fevals": int(eval_n),
            },
            "parameters": parameters_dict
        }
        
        json_out = os.path.join(plot_path, f"optimization_progress_{opt_timestamp}.json")
        try:
            with open(json_out, "w", encoding="utf-8") as f:
                json.dump(progress_json, f, indent=2)
            print(f"  [opt] Saved progress JSON to {json_out}")
        except Exception as e:
            print(f"  [opt] Warning: failed to save progress JSON: {e}")


    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    keys = sorted(trace_collector.keys(), key=lambda k: (str(k[0]), str(k[1])))
    n = len(keys)

    rebuild = (
        _progress_overlay_state["fig"] is None
        or _progress_overlay_state["keys"] != keys
    )
    if rebuild:
        ncols = min(3, max(1, n))
        nrows = (n + ncols - 1) // ncols
        fig = Figure(figsize=(5.0 * ncols, 3.4 * nrows))
        FigureCanvasAgg(fig)
        axes = fig.subplots(nrows, ncols, squeeze=False)
        _progress_overlay_state.update({
            "fig": fig, "axes": axes, "keys": keys,
            "ncols": ncols, "nrows": nrows,
        })

    fig   = _progress_overlay_state["fig"]
    axes  = _progress_overlay_state["axes"]
    ncols = _progress_overlay_state["ncols"]
    nrows = _progress_overlay_state["nrows"]

    for i, key in enumerate(keys):
        ax = axes[i // ncols, i % ncols]
        ax.clear()
        tr = trace_collector[key]
        ax.plot(tr["t_sim"], tr["y_sim"], color="C0", lw=1.2, label="model")
        ax.scatter(tr["t_data"], tr["y_data"], facecolors="none", edgecolors="black", s=10,
                   zorder=5, label="data")

        # Scale axes to the data extent (not the full simulation), so the model
        # curve is visible only over the windows where data exist.
        td = np.asarray(tr["t_data"])
        yd = np.asarray(tr["y_data"])
        td = td[np.isfinite(td)]
        yd = yd[np.isfinite(yd)]
        if td.size:
            tlo, thi = float(td.min()), float(td.max())
            pad = 0.05 * (thi - tlo) if thi > tlo else max(abs(thi), 1.0) * 0.05
            ax.set_xlim(tlo - pad, thi + pad)
        if yd.size:
            ylo, yhi = float(yd.min()), float(yd.max())
            pad = 0.10 * (yhi - ylo) if yhi > ylo else max(abs(yhi), 1.0) * 0.10
            ax.set_ylim(ylo - pad, yhi + pad)

        rep, obs = key
        ax.set_title(f"{rep} · {obs}\ncontrib={tr['contrib']:.4g}", fontsize=9)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=7, loc="best")

    for j in range(n, nrows * ncols):
        axes[j // ncols, j % ncols].set_visible(False)

    fig.suptitle(
        f"eval #{eval_n}  total_loss={total_loss:.5g}  best={best_loss:.5g}",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(plot_path, f"optimization_progress_{opt_timestamp}.png")
    fig.savefig(out, dpi=100, bbox_inches="tight")
    _progress_overlay_state["last_render"] = now


class OptRoadRunnerProxy:
    def __init__(self, r, opt_param_names):
        self._r = r
        self._opt_param_names = set(opt_param_names)

    def __setitem__(self, key, value):
        if key in self._opt_param_names:
            return
        self._r[key] = value

    def __getitem__(self, key):
        return self._r[key]

    def __getattr__(self, name):
        return getattr(self._r, name)

    def __setattr__(self, name, value):
        if name in ["_r", "_opt_param_names"]:
            super().__setattr__(name, value)
        elif name in self._opt_param_names:
            return
        else:
            setattr(self._r, name, value)



# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------

def _resolve_obs_df(df_dict, obs_cfg):
    """
    Return the DataFrame for one observable config entry, or None.

    Lookup order:
      1. If df_dict is a plain DataFrame, return it directly.
      2. If obs_cfg has "data_dict_key", use it as the dict key.
      3. Fall back to the first DataFrame whose columns contain both
         data_column and time_column.
    """
    d_col = obs_cfg["data_column"]
    t_col = obs_cfg["time_column"]
    if not isinstance(df_dict, dict):
        return df_dict if hasattr(df_dict, 'columns') else None
    data_key = obs_cfg.get("data_dict_key")
    if data_key is not None:
        return df_dict.get(data_key)
    return next(
        (v for v in df_dict.values()
         if hasattr(v, 'columns') and d_col in v.columns and t_col in v.columns),
        None,
    )


def _count_replicate_data_points(df_dict, effective_lc):
    """Count finite data points across all observables in a replicate's loss config.

    Used to normalize per-replicate loss contributions so that replicates with
    different numbers of time points contribute equally to the total objective.
    Returns at least 1 to avoid division by zero.
    """
    total = 0
    for obs_cfg in effective_lc.get("observables", []):
        obs_df = _resolve_obs_df(df_dict, obs_cfg)
        if obs_df is None:
            continue
        d_col = obs_cfg["data_column"]
        if d_col not in obs_df.columns:
            continue
        total += int(np.sum(np.isfinite(np.asarray(obs_df[d_col]))))
    return max(total, 1)


# ---------------------------------------------------------------------------
# Core run / loss
# ---------------------------------------------------------------------------

def run_all(r, exp_num, experiment, df_dict, set_parameters=None, parameters=None):
    """
    Run one simulation for *experiment* using the pre-built Tellurium model *r*
    and pre-loaded *df_dict*, optionally applying *parameters* first.

    Returns a results dict keyed by treatment label.
    """
    r.reset()

    # Re-apply treatment-specific parameters which were wiped out by r.reset()
    update_params = experiment.get("Update_parameters")
    if update_params is not None:
        update_params(r, experiment)

    if set_parameters is not None and parameters is not None:
        set_parameters(r, parameters)
        for hook in experiment.get("parameter_hooks", []):
            hook(r, parameters)

    # Propagate optimizer-dependent parameter relationships (e.g. a steady-state
    # coupling where two model parameters must move together but only one is
    # optimized). Update_parameters cannot do this — it runs at model-build
    # time, before the optimizer has chosen values, and it also sets initial
    # conditions we do not want to retouch every eval.
    update_opt = experiment.get("Update_opt_parameters")
    if update_opt is not None:
        # print("update_opt")
        update_opt(r, experiment, parameters)

    solver_settings  = experiment["Solver_settings"](experiment)
    observed_species = experiment["Observed_species"](r)
    label            = experiment.get("Label")
    results          = simulate(r, solver_settings, observed_species, label=label)

    return {label: {"results": results, "data": df_dict, "replicate": experiment}}


def set_parameters_from_dict(r, params):
    """Apply a name -> value dict to a Tellurium model."""
    for name, value in params.items():
        try:
            r[name] = value
        except Exception:
            pass


def loss_function_evaluated(
    param_dict,
    results_dict,
    param_names,
    loss_config=None,
    fixed_sigmas=None,
    debug=False,
    trace_collector=None,
):
    """
    Evaluate the loss for already simulated results.
    """
    loss_config = loss_config or {}
    observables_config = loss_config.get("observables", [])
    if not observables_config:
        raise ValueError(
            "loss_config must provide an 'observables' list specifying how to "
            "map Tellurium output to data columns.\n"
        )

    total_loss = 0.0
    for label, item in results_dict.items():
        result   = item["results"]
        item_df  = item["data"]
        exp_id   = label

        t_sim = np.asarray(result["time"])
        _eval_cache = {"local_dict": None, "cols": None}

        def _ensure_eval_context():
            if _eval_cache["local_dict"] is not None:
                return _eval_cache["local_dict"], _eval_cache["cols"]
            local_dict = {"np": np, "time": t_sim}
            if hasattr(result, "colnames"):
                cols = result.colnames
            elif hasattr(result, "dtype") and result.dtype.names:
                cols = result.dtype.names
            else:
                cols = []
            for col in cols:
                local_dict[col] = np.asarray(result[col])
                if col.startswith('[') and col.endswith(']'):
                    local_dict[col[1:-1]] = np.asarray(result[col])
            local_dict.update(param_dict)
            _eval_cache["local_dict"] = local_dict
            _eval_cache["cols"] = cols
            return local_dict, cols

        for obs_cfg in observables_config:
            obs   = obs_cfg["observed_variable"]
            d_col = obs_cfg["data_column"]
            t_col = obs_cfg["time_column"]

            obs_df = _resolve_obs_df(item_df, obs_cfg)
            if obs_df is None:
                continue

            if d_col not in obs_df.columns or t_col not in obs_df.columns:
                continue

            y_data = np.asarray(obs_df[d_col])
            t_data = np.asarray(obs_df[t_col])

            try:
                if callable(obs):
                    y_sim = np.asarray(obs(result))
                elif isinstance(obs, str):
                    local_dict, cols = _ensure_eval_context()
                    if obs in cols:
                        y_sim = np.asarray(result[obs])
                    else:
                        eval_obs = str(obs)
                        for col in cols:
                            if col.startswith('[') and col.endswith(']'):
                                eval_obs = eval_obs.replace(col, col[1:-1])
                        y_sim = np.asarray(eval(eval_obs, {}, local_dict))
                else:
                    raise ValueError(f"Invalid observable type: {type(obs)}")
            except Exception as e:
                raise RuntimeError(f"Failed to evaluate observable '{obs}': {e}") from e

            y_pred = np.interp(t_data, t_sim, y_sim)

            w_col = obs_cfg.get("weight_column")
            if w_col and w_col in obs_df.columns:
                obs_weights = np.asarray(obs_df[w_col], dtype=float)
            else:
                obs_weights = np.ones(len(y_data))

            valid = np.isfinite(y_pred) & np.isfinite(y_data)
            if not valid.any():
                continue
            y_data_v      = y_data[valid]
            y_pred_v      = y_pred[valid]
            obs_weights_v = obs_weights[valid]
            residuals     = y_data_v - y_pred_v

            n_eff = obs_weights_v.sum()
            loss_type = obs_cfg.get("loss_type", "nll")

            if loss_type == "ssr":
                contrib = np.dot(obs_weights_v, residuals ** 2) / n_eff
            elif fixed_sigmas is not None and (exp_id, obs) in fixed_sigmas:
                sigma = fixed_sigmas[(exp_id, obs)]
                contrib = -np.sum(obs_weights_v * stats.norm.logpdf(y_data_v, loc=y_pred_v, scale=sigma))
            else:
                # priority 1: per-datapoint sigma from data column (PEtab measurement-table style)
                noise_column = obs_cfg.get("noise_column")
                if noise_column and noise_column in obs_df.columns:
                    sigma_arr = np.maximum(
                        np.asarray(obs_df[noise_column], dtype=float)[valid], 1e-6
                    )
                    contrib = 0.5 * np.dot(obs_weights_v, (residuals / sigma_arr) ** 2) / n_eff
                    sigma   = float(np.mean(sigma_arr))
                else:
                    # priority 2: noise_formula (model-variable sigma, existing path)
                    sigma_config = obs_cfg.get("noise_formula", None)
                    sigma_ns = _ensure_eval_context()[0] if sigma_config else None
                    if sigma_config and sigma_ns is not None and sigma_config in sigma_ns:
                        sigma = float(sigma_ns[sigma_config])
                        contrib = -np.sum(obs_weights_v * stats.norm.logpdf(y_data_v, loc=y_pred_v, scale=sigma)) / n_eff
                    else:
                        # priority 3: sigma_method (dataset-level heuristic)
                        n_pts = y_data_v.size
                        sigma_method = obs_cfg.get("sigma_method", "max_mean_std")
                        sigma_arr = None
                        if sigma_method == "mean_y_data":
                            sigma = max(float(np.mean(np.abs(y_data_v))) if n_pts else 0.0, 1e-6)
                        elif sigma_method == "std_y_data":
                            sigma = max(
                                float(np.std(y_data_v)) if n_pts > 1 else
                                float(np.abs(y_data_v[0])) if n_pts == 1 else 0.0,
                                1e-6,
                            )
                        elif sigma_method == "fixed":
                            sigma = max(float(obs_cfg.get("sigma_value", 1.0)), 1e-6)
                        elif sigma_method == "relative":
                            sigma_frac = float(obs_cfg.get("sigma_value", 0.1))
                            sigma_arr  = np.maximum(np.abs(y_pred_v) * sigma_frac, 1e-6)
                            sigma      = float(np.mean(sigma_arr))
                        else:  # "max_mean_std" — current default
                            mean_abs = float(np.mean(np.abs(y_data_v))) if n_pts else 0.0
                            std_val  = float(np.std(y_data_v)) if n_pts > 1 else 0.0
                            sigma    = max(mean_abs, std_val, 1e-6)
                        s = sigma_arr if sigma_arr is not None else sigma
                        contrib = 0.5 * np.dot(obs_weights_v, (residuals / s) ** 2) / n_eff

            total_loss += contrib

            if trace_collector is not None:
                obs_label = getattr(obs, '__name__', str(obs))
                trace_collector[(str(exp_id), obs_label)] = {
                    "t_data":  np.asarray(t_data),
                    "y_data":  np.asarray(y_data),
                    "t_sim":   np.asarray(t_sim),
                    "y_sim":   np.asarray(y_sim),
                    "contrib": float(contrib),
                }

            if debug:
                obs_label = getattr(obs, '__name__', str(obs))
                print(f"  [loss] rep={exp_id}  obs='{obs_label}'")
                print(f" Params: {param_dict}")
                print(f"    t_sim : [{t_sim.min():.6g}, {t_sim.max():.6g}]  n={len(t_sim)}")
                print(f"    t_data: {np.array2string(t_data, precision=6, max_line_width=120)}")
                print(f"    y_data (valid): {np.array2string(y_data_v, precision=5, max_line_width=120)}")
                print(f"    y_pred (valid): {np.array2string(y_pred_v, precision=5, max_line_width=120)}")
                y_sim_finite = y_sim[np.isfinite(y_sim)]
                sim_range = (f"[{y_sim_finite.min():.4g}, {y_sim_finite.max():.4g}]"
                             if len(y_sim_finite) > 0 else "[all NaN/inf]")
                print(f"    y_sim range {sim_range}  ({valid.sum()}/{len(valid)} points valid)")
                if loss_type == "ssr":
                    print(f"    loss_type=ssr  contrib={contrib:.4g}")
                else:
                    sigma_tag = obs_cfg.get("noise_column") or obs_cfg.get("sigma_method", "max_mean_std")
                    print(f"    sigma={sigma:.4g}  [{sigma_tag}]  nll={contrib:.4g}")

    return float(total_loss)


def loss_function_composite(
    params,
    results_list,
    df_dict,
    exp_id,
    composite_elem,
    param_names,
    loss_config=None,
    fixed_sigmas=None,
    debug=False,
    trace_collector=None,
):
    """
    Evaluate aggregated composite loss across multiple simulated results.
    """
    loss_config = loss_config or {}
    observables_config = loss_config.get("observables", [])
    if not observables_config:
        raise ValueError(
            "loss_config must provide an 'observables' list specifying how to "
            "map Tellurium output to data columns.\n"
        )

    if isinstance(params, dict):
        param_dict = params
    else:
        param_dict = dict(zip(param_names, np.atleast_1d(params).tolist()))

    total_loss = 0.0

    for obs_cfg in observables_config:
        obs   = obs_cfg["observed_variable"]
        d_col = obs_cfg["data_column"]
        t_col = obs_cfg["time_column"]

        obs_df = _resolve_obs_df(df_dict, obs_cfg)
        if obs_df is None:
            continue

        if d_col not in obs_df.columns or t_col not in obs_df.columns:
            continue

        y_data = np.asarray(obs_df[d_col])
        t_data = np.asarray(obs_df[t_col])

        y_pred_subs = []
        for item in results_list:
            if isinstance(item, dict) and "results" in item:
                result = item["results"]
                rep = item["replicate"]
                df_dict_s = item["df_dict"]
            else:
                result = item
                rep = None
                df_dict_s = df_dict

            t_sim = np.asarray(result["time"])

            lc_fn = composite_elem.get("loss_config")
            if callable(lc_fn) and rep is not None:
                lc_s = lc_fn(rep)
                obs_cfg_s = None
                for oc in lc_s.get("observables", []):
                    if oc.get("data_column") == d_col:
                        obs_cfg_s = oc
                        break
                if obs_cfg_s is None:
                    obs_cfg_s = obs_cfg
            else:
                obs_cfg_s = obs_cfg

            obs_s = obs_cfg_s["observed_variable"]
            obs_df_s = _resolve_obs_df(df_dict_s, obs_cfg_s)
            if obs_df_s is None or d_col not in obs_df_s.columns or t_col not in obs_df_s.columns:
                continue

            t_data_s = np.asarray(obs_df_s[t_col])
            _eval_cache = {"local_dict": None, "cols": None}

            def _ensure_eval_context():
                if _eval_cache["local_dict"] is not None:
                    return _eval_cache["local_dict"], _eval_cache["cols"]
                local_dict = {"np": np, "time": t_sim}
                if hasattr(result, "colnames"):
                    cols = result.colnames
                elif hasattr(result, "dtype") and result.dtype.names:
                    cols = result.dtype.names
                else:
                    cols = []
                for col in cols:
                    local_dict[col] = np.asarray(result[col])
                    if col.startswith('[') and col.endswith(']'):
                        local_dict[col[1:-1]] = np.asarray(result[col])
                local_dict.update(param_dict)
                _eval_cache["local_dict"] = local_dict
                _eval_cache["cols"] = cols
                return local_dict, cols

            try:
                if callable(obs_s):
                    y_sim = np.asarray(obs_s(result))
                elif isinstance(obs_s, str):
                    local_dict, cols = _ensure_eval_context()
                    if obs_s in cols:
                        y_sim = np.asarray(result[obs_s])
                    else:
                        eval_obs = str(obs_s)
                        for col in cols:
                            if col.startswith('[') and col.endswith(']'):
                                eval_obs = eval_obs.replace(col, col[1:-1])
                        y_sim = np.asarray(eval(eval_obs, {}, local_dict))
                else:
                    raise ValueError(f"Invalid observable type: {type(obs_s)}")
            except Exception as e:
                raise RuntimeError(f"Failed to evaluate observable '{obs_s}': {e}") from e

            y_pred_sub = np.interp(t_data_s, t_sim, y_sim)
            y_pred_subs.append(y_pred_sub)

        if not y_pred_subs:
            continue

        aggregation = composite_elem.get("aggregation", "mean")
        if aggregation == "mean":
            y_pred = np.mean(y_pred_subs, axis=0)
        elif aggregation == "median":
            y_pred = np.median(y_pred_subs, axis=0)
        elif aggregation == "sum":
            y_pred = np.sum(y_pred_subs, axis=0)
        elif callable(aggregation):
            y_pred = aggregation(y_pred_subs)
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation}")

        w_col = obs_cfg.get("weight_column")
        if w_col and w_col in obs_df.columns:
            obs_weights = np.asarray(obs_df[w_col], dtype=float)
        else:
            obs_weights = np.ones(len(y_data))

        valid = np.isfinite(y_pred) & np.isfinite(y_data)
        if not valid.any():
            continue
        y_data_v      = y_data[valid]
        y_pred_v      = y_pred[valid]
        obs_weights_v = obs_weights[valid]
        residuals     = y_data_v - y_pred_v

        n_eff = obs_weights_v.sum()
        loss_type = obs_cfg.get("loss_type", "nll")

        if loss_type == "ssr":
            contrib = np.dot(obs_weights_v, residuals ** 2) / n_eff
        elif fixed_sigmas is not None and (exp_id, obs) in fixed_sigmas:
            sigma = fixed_sigmas[(exp_id, obs)]
            contrib = -np.sum(obs_weights_v * stats.norm.logpdf(y_data_v, loc=y_pred_v, scale=sigma))
        else:
            # priority 1: per-datapoint sigma from data column (PEtab measurement-table style)
            noise_column = obs_cfg.get("noise_column")
            if noise_column and noise_column in obs_df.columns:
                sigma_arr = np.maximum(
                    np.asarray(obs_df[noise_column], dtype=float)[valid], 1e-6
                )
                contrib = 0.5 * np.dot(obs_weights_v, (residuals / sigma_arr) ** 2) / n_eff
                sigma   = float(np.mean(sigma_arr))
            else:
                # priority 2: noise_formula (model-variable sigma, existing path)
                sigma_config = obs_cfg.get("noise_formula", None)
                sigma_ns = None
                if sigma_config and len(results_list) > 0:
                    result_first = results_list[0]["results"] if isinstance(results_list[0], dict) else results_list[0]
                    t_sim_first = np.asarray(result_first["time"])
                    local_dict_first = {"np": np, "time": t_sim_first}
                    cols_first = (result_first.colnames if hasattr(result_first, "colnames")
                                  else (result_first.dtype.names if hasattr(result_first, "dtype") else []))
                    for col in cols_first:
                        local_dict_first[col] = np.asarray(result_first[col])
                        if col.startswith('[') and col.endswith(']'):
                            local_dict_first[col[1:-1]] = np.asarray(result_first[col])
                    local_dict_first.update(param_dict)
                    sigma_ns = local_dict_first

                if sigma_config and sigma_ns is not None and sigma_config in sigma_ns:
                    sigma = float(sigma_ns[sigma_config])
                    contrib = -np.sum(obs_weights_v * stats.norm.logpdf(y_data_v, loc=y_pred_v, scale=sigma)) / n_eff
                else:
                    # priority 3: sigma_method (dataset-level heuristic)
                    n_pts = y_data_v.size
                    sigma_method = obs_cfg.get("sigma_method", "max_mean_std")
                    sigma_arr = None
                    if sigma_method == "mean_y_data":
                        sigma = max(float(np.mean(np.abs(y_data_v))) if n_pts else 0.0, 1e-6)
                    elif sigma_method == "std_y_data":
                        sigma = max(
                            float(np.std(y_data_v)) if n_pts > 1 else
                            float(np.abs(y_data_v[0])) if n_pts == 1 else 0.0,
                            1e-6,
                        )
                    elif sigma_method == "fixed":
                        sigma = max(float(obs_cfg.get("sigma_value", 1.0)), 1e-6)
                    elif sigma_method == "relative":
                        sigma_frac = float(obs_cfg.get("sigma_value", 0.1))
                        sigma_arr  = np.maximum(np.abs(y_pred_v) * sigma_frac, 1e-6)
                        sigma      = float(np.mean(sigma_arr))
                    else:  # "max_mean_std" — current default
                        mean_abs = float(np.mean(np.abs(y_data_v))) if n_pts else 0.0
                        std_val  = float(np.std(y_data_v)) if n_pts > 1 else 0.0
                        sigma    = max(mean_abs, std_val, 1e-6)
                    s = sigma_arr if sigma_arr is not None else sigma
                    contrib = 0.5 * np.dot(obs_weights_v, (residuals / s) ** 2) / n_eff

        total_loss += contrib

        if trace_collector is not None:
            obs_label = getattr(obs, '__name__', str(obs))
            res_0 = results_list[0]["results"] if results_list and isinstance(results_list[0], dict) else (results_list[0] if results_list else None)
            t_sim_first = np.asarray(res_0["time"]) if res_0 is not None else t_data
            trace_collector[(str(exp_id), obs_label)] = {
                "t_data":  np.asarray(t_data),
                "y_data":  np.asarray(y_data),
                "t_sim":   np.asarray(t_sim_first),
                "y_sim":   np.interp(t_sim_first, t_data, y_pred),
                "contrib": float(contrib),
            }

        if debug:
            obs_label = getattr(obs, '__name__', str(obs))
            print(f"  [composite loss] group={exp_id}  obs='{obs_label}'")
            print(f" Params: {param_dict}")
            print(f"    t_data: {np.array2string(t_data, precision=6, max_line_width=120)}")
            print(f"    y_data (valid): {np.array2string(y_data_v, precision=5, max_line_width=120)}")
            print(f"    y_pred (valid): {np.array2string(y_pred_v, precision=5, max_line_width=120)}")
            sigma_tag = obs_cfg.get("noise_column") or obs_cfg.get("sigma_method", "max_mean_std")
            print(f"    sigma={sigma:.4g}  [{sigma_tag}]  loss={contrib:.4g}")

    return float(total_loss)


def loss_function(
    params,
    r,
    exp_num,
    experiment,
    df_dict,
    param_names,
    loss_config=None,
    fixed_sigmas=None,
    debug=False,
    trace_collector=None,
):
    """
    Run *experiment* with the given parameters and return a scalar NLL loss.
    """
    if isinstance(params, dict):
        param_dict = params
    else:
        param_dict = dict(zip(param_names, np.atleast_1d(params).tolist()))

    try:
        p_vals = np.asarray(list(param_dict.values()))
        if np.any(p_vals <= 0):
            return 1e10
    except Exception:
        pass

    def set_params(r, p):
        set_parameters_from_dict(r, p)

    try:
        results_dict = run_all(r, exp_num, experiment, df_dict,
                               set_parameters=set_params, parameters=param_dict)
    except Exception as e:
        print(f"    [loss_function] Warning: Simulation failed for replicate '{exp_num}' with parameters {param_dict}: {e}")
        return 1e10

    return loss_function_evaluated(
        param_dict,
        results_dict,
        param_names,
        loss_config=loss_config,
        fixed_sigmas=fixed_sigmas,
        debug=debug,
        trace_collector=trace_collector,
    )


# ---------------------------------------------------------------------------
# Hessian utilities
# ---------------------------------------------------------------------------

def compute_hessian_numdifftools(func, params):
    try:
        hessian_func = nd.Hessian(func, method='central', step=1e-5)
        return hessian_func(params)
    except Exception as e:
        warnings.warn(f"Error computing Hessian with numdifftools: {e}")
        return compute_hessian_manual(func, params)

def compute_hessian_manual(func, params, epsilon=1e-5):
    n = len(params)
    hessian = np.zeros((n, n))
    f0 = func(params)
    for i in range(n):
        params_plus  = np.array(params, copy=True)
        params_minus = np.array(params, copy=True)
        step = epsilon * max(abs(params[i]), 1.0)
        params_plus[i]  += step
        params_minus[i] -= step
        hessian[i, i] = (func(params_plus) - 2*f0 + func(params_minus)) / (step**2)
    for i in range(n):
        for j in range(i+1, n):
            pp = np.array(params, copy=True); pm = np.array(params, copy=True)
            mp = np.array(params, copy=True); mm = np.array(params, copy=True)
            si = epsilon * max(abs(params[i]), 1.0)
            sj = epsilon * max(abs(params[j]), 1.0)
            pp[i] += si; pp[j] += sj
            pm[i] += si; pm[j] -= sj
            mp[i] -= si; mp[j] += sj
            mm[i] -= si; mm[j] -= sj
            hessian[i, j] = (func(pp) - func(pm) - func(mp) + func(mm)) / (4 * si * sj)
            hessian[j, i] = hessian[i, j]
    return hessian

def compute_parameter_correlations(cov_matrix):
    if cov_matrix is None:
        return None
    try:
        std_devs = np.sqrt(np.diag(cov_matrix))
        return cov_matrix / np.outer(std_devs, std_devs)
    except Exception as e:
        print(f"Error computing correlation matrix: {e}")
        return None


# ---------------------------------------------------------------------------
# Global optimization dispatcher (Enhancement 8)
# ---------------------------------------------------------------------------

_GLOBAL_METHODS = frozenset({
    "differential_evolution",
    "basin_hopping",
    "dual_annealing",
    "shgo",
})


def _prepare_optimizer_kwargs(method, optimizer_kwargs, fast, maxiter, tol):
    """Integrate fast, maxiter, and tol settings into optimizer_kwargs."""
    kwargs = dict(optimizer_kwargs or {})
    kwargs.pop("events_depend_on_opt_param", None)  # Remove our custom parameter so minimize() doesn't fail
    kwargs.pop("profile_without_opt", None)  # Remove custom parameter for skipping optimization
    m = method.lower()
    
    if fast:
        maxi = maxiter if maxiter is not None else 50
        t = tol if tol is not None else 1e-2
    else:
        maxi = maxiter
        t = tol
        
    if m in _GLOBAL_METHODS:
        if "options" in kwargs:
            opts = kwargs.pop("options")
            for k, v in opts.items():
                if k not in kwargs:
                    kwargs[k] = v
        if t is not None:
            kwargs["tol"] = t
        if maxi is not None:
            if m == "basin_hopping":
                kwargs["niter"] = maxi
            else:
                kwargs["maxiter"] = maxi
    else:
        if t is not None:
            kwargs["tol"] = t
        if maxi is not None:
            opts = dict(kwargs.get("options", {}))
            opts["maxiter"] = maxi
            kwargs["options"] = opts
            
    return kwargs


def _run_global_optimization(objective, x0, bounds, method, kwargs):
    """
    Dispatch to a scipy global optimizer and return a result compatible with
    scipy.optimize.OptimizeResult (has .x, .fun, .success, .message).

    Parameters
    ----------
    objective : callable(x) -> float
    x0        : initial parameter vector (used by basin_hopping / dual_annealing)
    bounds    : sequence of (lo, hi) pairs; required for DE / dual_annealing / shgo
    method    : one of _GLOBAL_METHODS
    kwargs    : extra keyword args forwarded to the chosen optimizer
    """
    kwargs = dict(kwargs or {})
    m = method.lower()

    if m == "differential_evolution":
        from scipy.optimize import differential_evolution
        if not bounds:
            raise ValueError("differential_evolution requires bounds")
        kwargs.setdefault("seed", 42)
        kwargs.setdefault("popsize", 15)
        kwargs.setdefault("tol", 1e-6)
        kwargs.setdefault("maxiter", 1000)
        kwargs.setdefault("workers", 1)
        return differential_evolution(objective, bounds=bounds, **kwargs)

    elif m == "basin_hopping":
        from scipy.optimize import basinhopping
        kwargs.setdefault("niter", 100)
        kwargs.setdefault("T", 1.0)
        kwargs.setdefault("stepsize", 0.5)
        minimizer_kw = kwargs.pop(
            "minimizer_kwargs",
            {"method": "L-BFGS-B", "bounds": bounds or []},
        )
        return basinhopping(objective, x0, minimizer_kwargs=minimizer_kw, **kwargs)

    elif m == "dual_annealing":
        from scipy.optimize import dual_annealing
        if not bounds:
            raise ValueError("dual_annealing requires bounds")
        kwargs.setdefault("maxiter", 1000)
        kwargs.setdefault("seed", 42)
        return dual_annealing(objective, bounds=bounds, x0=x0, **kwargs)

    elif m == "shgo":
        from scipy.optimize import shgo
        if not bounds:
            raise ValueError("shgo requires bounds")
        return shgo(objective, bounds=bounds, **kwargs)

    else:
        raise ValueError(f"Unknown global optimization method: {method!r}")


# ---------------------------------------------------------------------------
# Profile-likelihood helpers (module-level so fork/thread workers can call them)
# ---------------------------------------------------------------------------

def _old_run_pypesto_profile_single(
    param_idx, nll_func, bounds, res_x, nll_at_optimum, param_names,
    n_points=20, range_factor=2.0, fallback_func=None, wald_se_val=None
):
    """
    PRESERVED AS REQUESTED:
    Run an adaptive true profile likelihood for one parameter; return (param_vals, nll_vals_rel).
    """
    import scipy.optimize as opt
    import numpy as np

    pname = param_names[param_idx]
    p_opt = res_x[param_idx]
    
    if bounds is not None:
        lb, ub = bounds[param_idx]
    else:
        lb = p_opt / range_factor
        ub = p_opt * range_factor

    print(f"\n[true profile] {pname}  (adaptive walking, re-optimizing nuisance params)", flush=True)
    
    def nuisance_objective(x_nuisance, fixed_val):
        x_full = np.zeros(len(param_names))
        idx_nuisance = 0
        for i in range(len(param_names)):
            if i == param_idx:
                x_full[i] = fixed_val
            else:
                x_full[i] = x_nuisance[idx_nuisance]
                idx_nuisance += 1
        return nll_func(x_full)

    if bounds is not None:
        nuisance_bounds = bounds[:param_idx] + bounds[param_idx+1:]
    else:
        nuisance_bounds = None

    max_nll = 5.0  # Stop when we exceed 95% CI (1.92) by a safe margin
    
    def walk_profile(direction_sign, bound):
        vals = []
        nlls = []
        
        if wald_se_val is not None and wald_se_val > 0:
            step = max(wald_se_val / 4.0, 1e-8)
        else:
            step = max(abs(p_opt) * 0.01, 1e-4)
            
        current_val = p_opt
        x_nuisance = np.delete(res_x, param_idx)
        
        while True:
            test_val = current_val + direction_sign * step
            
            hit_bound = False
            if direction_sign == -1 and test_val <= bound:
                test_val = bound
                hit_bound = True
            if direction_sign == 1 and test_val >= bound:
                test_val = bound
                hit_bound = True
                
            res = opt.minimize(
                nuisance_objective, 
                x_nuisance, 
                args=(test_val,),
                method='L-BFGS-B', 
                bounds=nuisance_bounds,
                options={'maxiter': 50, 'ftol': 1e-4}
            )
            
            nll_rel = res.fun - nll_at_optimum
            
            prev_nll = nlls[-1] if len(nlls) > 0 else 0.0
            dnll = nll_rel - prev_nll
            
            if dnll > 1.5:
                if step > 1e-6 * max(abs(p_opt), 1e-4):
                    factor = 0.5 / max(dnll, 1e-4)
                    factor = max(factor, 0.1)  # shrink by at most 10x
                    step *= factor
                    continue
            
            vals.append(test_val)
            nlls.append(nll_rel)
            
            print(f"  [{'left ' if direction_sign==-1 else 'right'}]  {pname}={test_val:.4g}  dNLL={nll_rel:.6g}", flush=True)
            
            if nll_rel > max_nll:
                break
            if hit_bound:
                break
                
            current_val = test_val
            x_nuisance = res.x
            
            dnll_eff = max(dnll, 1e-4)
            factor = 0.5 / dnll_eff
            factor = min(max(factor, 0.5), 2.0)
            step *= factor
            
            if len(vals) > 50:
                break
                
        return vals, nlls

    left_vals, left_nlls = walk_profile(-1, lb)
    right_vals, right_nlls = walk_profile(1, ub)
    
    all_vals = np.array(left_vals[::-1] + [p_opt] + right_vals)
    all_nlls = np.array(left_nlls[::-1] + [0.0] + right_nlls)
    
    return all_vals, all_nlls


def _run_pypesto_profile_single(
    param_idx, nll_func, bounds, res_x, nll_at_optimum, param_names,
    n_points=20, range_factor=2.0, fallback_func=None, wald_se_val=None
):
    """Run an adaptive true profile likelihood for one parameter; return (param_vals, nll_vals_rel)."""
    import scipy.optimize as opt
    import numpy as np

    pname = param_names[param_idx]
    p_opt = res_x[param_idx]
    
    if bounds is not None:
        lb_bound, ub_bound = bounds[param_idx]
    else:
        lb_bound = -np.inf
        ub_bound = np.inf

    target_1 = p_opt / range_factor
    target_2 = p_opt * range_factor
    lb_target = min(target_1, target_2)
    ub_target = max(target_1, target_2)

    lb = max(lb_target, lb_bound)
    ub = min(ub_target, ub_bound)

    print(f"\n[true profile] {pname}  (adaptive stepping between {lb:.4g} and {ub:.4g}, re-optimizing nuisance params)", flush=True)
    
    def nuisance_objective(x_nuisance, fixed_val):
        x_full = np.zeros(len(param_names))
        idx_nuisance = 0
        for i in range(len(param_names)):
            if i == param_idx:
                x_full[i] = fixed_val
            else:
                x_full[i] = x_nuisance[idx_nuisance]
                idx_nuisance += 1
        return nll_func(x_full)

    if bounds is not None:
        nuisance_bounds = bounds[:param_idx] + bounds[param_idx+1:]
    else:
        nuisance_bounds = None

    def walk_profile(direction_sign, bound):
        if (direction_sign == 1 and bound <= p_opt) or (direction_sign == -1 and bound >= p_opt):
            return [], []
            
        evaluated = [(p_opt, 0.0, np.delete(res_x, param_idx))]
        
        def evaluate_pt(x_target, x_nuisance_guess):
            if len(x_nuisance_guess) == 0:
                # No nuisance parameters to re-optimize (single-parameter fit):
                # the profile value at x_target is just the objective itself.
                # scipy.optimize.minimize errors on a length-0 x0, so skip it.
                nll_rel = nuisance_objective(x_nuisance_guess, x_target) - nll_at_optimum
                print(f"  [{'left ' if direction_sign==-1 else 'right'}]  {pname}={x_target:.4g}  dNLL={nll_rel:.6g}", flush=True)
                return nll_rel, x_nuisance_guess
            res = opt.minimize(
                nuisance_objective,
                x_nuisance_guess,
                args=(x_target,),
                method='L-BFGS-B',
                bounds=nuisance_bounds,
                options={'maxiter': 50, 'ftol': 1e-4}
            )
            nll_rel = res.fun - nll_at_optimum
            print(f"  [{'left ' if direction_sign==-1 else 'right'}]  {pname}={x_target:.4g}  dNLL={nll_rel:.6g}", flush=True)
            return nll_rel, res.x
            
        coarse_steps = max(3, n_points // 4)
        coarse_xs = np.linspace(p_opt, bound, coarse_steps + 1)[1:]
        
        crossed = False
        for x_target in coarse_xs:
            last_x, last_nll, last_nuisance = evaluated[-1]
            nll_rel, x_nuisance = evaluate_pt(x_target, last_nuisance)
            evaluated.append((x_target, nll_rel, x_nuisance))
            
            if nll_rel > 1.92:
                crossed = True
                print(f"  Reached 95% CI threshold. Bracketing first crossing...", flush=True)
                break
                
        if crossed:
            x_in, nll_in, nuisance_in = evaluated[-2]
            x_out, nll_out, nuisance_out = evaluated[-1]
            
            for _ in range(3):
                x_mid = (x_in + x_out) / 2.0
                nll_mid, nuisance_mid = evaluate_pt(x_mid, nuisance_in)
                evaluated.append((x_mid, nll_mid, nuisance_mid))
                
                if nll_mid > 1.92:
                    x_out, nll_out, nuisance_out = x_mid, nll_mid, nuisance_mid
                else:
                    x_in, nll_in, nuisance_in = x_mid, nll_mid, nuisance_mid
                    
            range_end = x_out
        else:
            range_end = bound
            
        budget = max(10, n_points // 2)
        
        while len(evaluated) - 1 < budget:
            if direction_sign == 1:
                valid_pts = sorted([p for p in evaluated if p_opt <= p[0] <= range_end + 1e-12], key=lambda p: p[0])
            else:
                valid_pts = sorted([p for p in evaluated if p_opt >= p[0] >= range_end - 1e-12], key=lambda p: -p[0])
                
            max_score = -1
            best_i = -1
            
            range_width = abs(range_end - p_opt)
            if range_width < 1e-12:
                break
                
            for i in range(len(valid_pts) - 1):
                gap_x = abs(valid_pts[i][0] - valid_pts[i+1][0])
                gap_nll = abs(valid_pts[i][1] - valid_pts[i+1][1])
                
                score = gap_x / range_width
                if gap_nll > 0.5:
                    score += min(gap_nll, 5.0) / 1.92
                    
                if score > max_score and gap_x > 1e-8:
                    max_score = score
                    best_i = i
                    
            if best_i == -1:
                break
                
            x_left, nll_left, nuisance_left = valid_pts[best_i]
            x_right, nll_right, nuisance_right = valid_pts[best_i+1]
            
            x_mid = (x_left + x_right) / 2.0
            nll_mid, nuisance_mid = evaluate_pt(x_mid, nuisance_left)
            evaluated.append((x_mid, nll_mid, nuisance_mid))
            
        if direction_sign == 1:
            final_pts = sorted([p for p in evaluated if p[0] > p_opt + 1e-12], key=lambda p: p[0])
        else:
            final_pts = sorted([p for p in evaluated if p[0] < p_opt - 1e-12], key=lambda p: -p[0])
            
        vals = [p[0] for p in final_pts]
        nlls = [p[1] for p in final_pts]
        
        return vals, nlls

    left_vals, left_nlls = walk_profile(-1, lb)
    right_vals, right_nlls = walk_profile(1, ub)
    
    all_vals = np.array(left_vals[::-1] + [p_opt] + right_vals)
    all_nlls = np.array(left_nlls[::-1] + [0.0] + right_nlls)
    
    return all_vals, all_nlls


# ---------------------------------------------------------------------------
# High-level optimization entry point
# ---------------------------------------------------------------------------

def run_optimization(
    model_text,
    paths,
    experiments,
    param_names,
    x0,
    bounds=None,
    loss_config=None,
    wald_analysis=False,
    slice_analysis=False,
    profile_likelihood_analysis=False,
    sobol_analysis=False,
    sobol_kwargs=None,
    method="Nelder-Mead",
    optimizer_kwargs=None,
    fast=False,
    maxiter=None,
    tol=None,
):
    if profile_likelihood_analysis and not wald_analysis:
        print("Note: profile_likelihood_analysis is True, automatically enabling wald_analysis to seed step sizes.")
        wald_analysis = True

    from datetime import datetime
    _progress_overlay_state["timestamp"] = datetime.now().strftime("%Y%m%d_%H%M%S")

    try:
        from scipy.optimize import minimize
    except ImportError:
        raise ImportError("scipy is required for run_optimization") from None

    data_path = paths["data_path"]

    # r_ic is only used when events depend on optimizer parameters (dynamic
    # event rebuild path). Skip the second model compile when it is not needed.
    _events_dynamic = (optimizer_kwargs.get("events_depend_on_opt_param", False)
                       if optimizer_kwargs else False)

    # Pre-build one Tellurium model per experiment and load its data once.
    models = {}
    for exp_num, experiment in experiments.items():
        df_dict    = experiment["Data"](experiment, data_path)

        if _events_dynamic:
            r_ic       = TelluriumGen(model_text, paths)
            r_ic_proxy = OptRoadRunnerProxy(r_ic, param_names)
            experiment["Update_parameters"](r_ic_proxy, experiment)
            try:
                events_str = experiment["Events"](experiment, df_dict, r_ic=r_ic)
            except TypeError:
                events_str = experiment["Events"](experiment, df_dict)
        else:
            r_ic = None
            try:
                events_str = experiment["Events"](experiment, df_dict, r_ic=None)
            except TypeError:
                events_str = experiment["Events"](experiment, df_dict)

        r          = TelluriumGen(model_text + "\n" + events_str, paths)
        r_proxy    = OptRoadRunnerProxy(r, param_names)
        experiment["Update_parameters"](r_proxy, experiment)

        models[exp_num] = {"r_ic": r_ic, "r": r, "df_dict": df_dict}

    def objective(x):
        x_dict = dict(zip(param_names, np.atleast_1d(x).tolist()))
        events_dynamic = optimizer_kwargs.get("events_depend_on_opt_param", False) if optimizer_kwargs else False
        
        if events_dynamic:
            total_loss = 0.0
            for exp_num, experiment in experiments.items():
                m = models[exp_num]
                r_ic = m["r_ic"]
                r_ic.reset()
                set_parameters_from_dict(r_ic, x_dict)
                try:
                    events_str = experiment["Events"](experiment, m["df_dict"], r_ic=r_ic)
                except TypeError:
                    events_str = experiment["Events"](experiment, m["df_dict"])
                
                r_new = TelluriumGen(model_text + "\n" + events_str, paths)
                r_proxy = OptRoadRunnerProxy(r_new, param_names)
                experiment["Update_parameters"](r_proxy, experiment)
                loss_val = loss_function(
                    x, r_new, exp_num, experiment, m["df_dict"],
                    param_names, loss_config=loss_config,
                )
                if loss_val >= 1e10:
                    return 1e10
                total_loss += loss_val
            return total_loss
        
        total_loss = 0.0
        for exp_num, experiment in experiments.items():
            m = models[exp_num]
            try:
                loss_val = loss_function(
                    x, m["r"], exp_num, experiment, m["df_dict"],
                    param_names, loss_config=loss_config,
                )
            except Exception as e:
                print(f"Error evaluating experiment {exp_num}: {e}")
                return 1e10
            if loss_val >= 1e10:
                return 1e10
            total_loss += loss_val

        return total_loss

    opt_kw = _prepare_optimizer_kwargs(method, optimizer_kwargs, fast, maxiter, tol)
    
    profile_without_opt = optimizer_kwargs.get("profile_without_opt", False) if optimizer_kwargs else False
    if profile_without_opt:
        from scipy.optimize import OptimizeResult
        x0_arr = np.array(x0)
        res = OptimizeResult(x=x0_arr, fun=objective(x0_arr), success=True, message="Optimization bypassed", nit=0, nfev=1)
    elif method.lower() in _GLOBAL_METHODS:
        res = _run_global_optimization(objective, x0, bounds, method, opt_kw)
    else:
        res = minimize(objective, x0, method=method, bounds=bounds or [], **opt_kw)
        
    out = {"x": res.x, "fun": res.fun, "success": res.success,
           "message": res.message, "stats": {},
           "timestamp": _progress_overlay_state.get("timestamp")}

    if res.success:
        param_dict = dict(zip(param_names, res.x.tolist()))

        def set_params(r, p):
            set_parameters_from_dict(r, p)

        # Run at optimal params to get best-fit results and empirical sigmas.
        best_results = {}
        fixed_sigmas = {}
        total_n      = 0
        k            = len(param_names)
        loss_config_safe    = loss_config or {}
        observables_config  = loss_config_safe.get("observables", [])

        for exp_num, experiment in experiments.items():
            m       = models[exp_num]
            res_dict = run_all(m["r"], exp_num, experiment, m["df_dict"],
                               set_parameters=set_params, parameters=param_dict)
            best_results.update(res_dict)

            for i, (label, item) in enumerate(res_dict.items()):
                result   = item["results"]
                item_df  = item["data"]
                exp_id   = exp_num
                t_sim    = np.asarray(result["time"])

                local_dict = {"np": np, "time": t_sim}
                cols = (result.colnames if hasattr(result, "colnames")
                        else (result.dtype.names if hasattr(result, "dtype") else []))
                for c in cols:
                    local_dict[c] = np.asarray(result[c])
                    if c.startswith('[') and c.endswith(']'):
                        local_dict[c[1:-1]] = np.asarray(result[c])
                local_dict.update(param_dict)

                for obs_cfg in observables_config:
                    obs   = obs_cfg["observed_variable"]
                    d_col = obs_cfg["data_column"]
                    t_col = obs_cfg["time_column"]

                    obs_df = _resolve_obs_df(item_df, obs_cfg)
                    if obs_df is None:
                        continue
                    if d_col not in obs_df.columns or t_col not in obs_df.columns:
                        continue

                    y_data = np.asarray(obs_df[d_col])
                    t_data = np.asarray(obs_df[t_col])

                    if callable(obs):
                        y_sim = np.asarray(obs(result))
                    elif isinstance(obs, str) and obs in cols:
                        y_sim = np.asarray(result[obs])
                    elif isinstance(obs, str):
                        eval_obs = str(obs)
                        for c in cols:
                            if c.startswith('[') and c.endswith(']'):
                                eval_obs = eval_obs.replace(c, c[1:-1])
                        y_sim = np.asarray(eval(eval_obs, {}, local_dict))

                    y_pred = np.interp(t_data, t_sim, y_sim)
                    valid = np.isfinite(y_pred) & np.isfinite(y_data)
                    if not valid.any():
                        continue
                    y_data_v  = y_data[valid]
                    y_pred_v  = y_pred[valid]
                    residuals = y_data_v - y_pred_v

                    sigma_config = obs_cfg.get("noise_formula", None)
                    if sigma_config and sigma_config in local_dict:
                        sigma = float(local_dict[sigma_config])
                    else:
                        n_block = len(residuals)
                        if n_block > 1:
                            sigma = np.sqrt(np.sum(residuals**2) /
                                             max(1, n_block - k / max(1, len(observables_config))))
                        elif n_block == 1:
                            sigma = max(np.abs(y_data_v[0]) * 0.1, 1e-6)
                        else:
                            sigma = 1e-6
                    fixed_sigmas[(exp_id, obs)] = sigma
                    total_n += len(residuals)

        def nll_func_fixed(p):
            p_dict = dict(zip(param_names, np.atleast_1d(p).tolist()))
            events_dynamic = optimizer_kwargs.get("events_depend_on_opt_param", False) if optimizer_kwargs else False
            
            if events_dynamic:
                total_nll = 0.0
                for exp_num, experiment in experiments.items():
                    m = models[exp_num]
                    r_ic = m["r_ic"]
                    r_ic.reset()
                    set_parameters_from_dict(r_ic, p_dict)
                    try:
                        events_str = experiment["Events"](experiment, m["df_dict"], r_ic=r_ic)
                    except TypeError:
                        events_str = experiment["Events"](experiment, m["df_dict"])
                    
                    r_new = TelluriumGen(model_text + "\n" + events_str, paths)
                    r_proxy = OptRoadRunnerProxy(r_new, param_names)
                    experiment["Update_parameters"](r_proxy, experiment)
                    total_nll += loss_function(
                        p, r_new, exp_num, experiment, m["df_dict"],
                        param_names, loss_config, fixed_sigmas=fixed_sigmas,
                    )
                return total_nll

            total_nll = 0.0
            for exp_num, experiment in experiments.items():
                m = models[exp_num]
                try:
                    total_nll += loss_function(
                        p, m["r"], exp_num, experiment, m["df_dict"],
                        param_names, loss_config, fixed_sigmas=fixed_sigmas,
                    )
                except Exception as e:
                    print(f"Error evaluating fixed experiment {exp_num}: {e}")
                    return 1e10

            return total_nll

        out["results_dict"] = best_results

        if wald_analysis or slice_analysis or profile_likelihood_analysis or sobol_analysis:
            # AIC/BIC need a proper joint NLL, not the z-score χ² objective
            # the optimizer minimized. nll_func_fixed reuses fixed_sigmas
            # captured at the optimum and returns the joint NLL.
            nll_proper = nll_func_fixed(res.x)
            aic = 2 * k + 2 * nll_proper
            bic = k * np.log(total_n) + 2 * nll_proper
            out["stats"]["aic"] = aic
            out["stats"]["bic"] = bic
            out["stats"]["nll_proper"] = nll_proper

        if wald_analysis:
            try:
                print(f"\n[Wald] Computing Hessian matrix for {k} parameters... (this requires many silent evaluations)")
                hessian_raw  = compute_hessian_numdifftools(nll_func_fixed, res.x)
                fisher_info  = (hessian_raw + hessian_raw.T) / 2.0
                eigenvalues  = np.linalg.eigvals(fisher_info)
                if np.any(eigenvalues <= 1e-10):
                    print("Warning: Hessian is NOT positive definite.")
                    cov_matrix       = None
                    standard_errors  = None
                else:
                    cov_matrix      = np.linalg.inv(fisher_info)
                    diag_elements   = np.diag(cov_matrix)
                    standard_errors = (None if np.any(diag_elements < 0)
                                       else np.sqrt(diag_elements))
                out["stats"]["wald_fisher_info"]  = fisher_info
                out["stats"]["wald_covariance"]   = cov_matrix
                out["stats"]["wald_se"]           = standard_errors
                if standard_errors is not None:
                    cv = stats.norm.ppf(1 - 0.05/2)
                    out["stats"]["wald_ci"] = [
                        (p_val - cv * se, p_val + cv * se)
                        for p_val, se in zip(res.x, standard_errors)
                    ]
                if cov_matrix is not None:
                    out["stats"]["wald_correlation"] = compute_parameter_correlations(cov_matrix)
            except Exception as e:
                print(f"Error computing Wald statistics: {e}")

        if slice_analysis or profile_likelihood_analysis:
            nll_at_optimum = nll_func_fixed(res.x)
            out["stats"]["nll_at_optimum"] = nll_at_optimum

            def likelihood_slice_func(param_idx, n_points=20, range_factor=2.0):
                p_val      = res.x[param_idx]
                pname      = param_names[param_idx]
                param_vals = np.linspace(p_val / range_factor, p_val * range_factor, n_points)
                width      = len(str(n_points))
                print(f"\n[slice] {pname}  ({n_points} points, x{range_factor} range)")
                nll_vals = []
                for i, val in enumerate(param_vals):
                    nll = nll_func_fixed(np.where(np.arange(len(res.x)) == param_idx, val, res.x))
                    print(f"  [{i+1:{width}d}/{n_points}]  {pname}={val:.4g}  nll={nll:.6g}")
                    nll_vals.append(nll)
                return param_vals, np.array(nll_vals) - nll_at_optimum

            if slice_analysis:
                out["stats"]["likelihood_slice"] = likelihood_slice_func

            if profile_likelihood_analysis:
                def true_profile_likelihood_func(param_idx, n_points=20, range_factor=2.0):
                    se_array = out["stats"].get("wald_se")
                    wald_se_val = se_array[param_idx] if se_array is not None else None
                    return _run_pypesto_profile_single(
                        param_idx, nll_func_fixed, bounds, res.x, nll_at_optimum,
                        param_names, n_points=n_points, range_factor=range_factor,
                        fallback_func=likelihood_slice_func, wald_se_val=wald_se_val
                    )

                out["stats"]["profile_likelihood"]  = true_profile_likelihood_func

            if sobol_analysis:
                from Engine.Sensitivity_analysis import run_sobol_analysis
                skwargs = sobol_kwargs or {}
                out["stats"]["sobol"] = run_sobol_analysis(
                    nll_func_fixed, param_names, bounds, res.x, **skwargs
                )

    out["r"] = list(models.values())[0]["r"] if models else None
    return out


# ---------------------------------------------------------------------------
# Group-aware optimization entry point
# ---------------------------------------------------------------------------

def run_optimization_from_groups(
    model_text,
    paths,
    experiment,
    param_names,
    x0,
    bounds=None,
    group_names=None,
    method="Nelder-Mead",
    optimizer_kwargs=None,
    wald_analysis=False,
    slice_analysis=False,
    profile_likelihood_analysis=False,
    sobol_analysis=False,
    sobol_kwargs=None,
    fast=False,
    maxiter=None,
    tol=None,
    optimization_spec=None,
):
    """
    Optimize shared parameters using ``experiment.opt_groups`` or ``optimization_spec``.
    """
    if profile_likelihood_analysis and not wald_analysis:
        print("Note: profile_likelihood_analysis is True, automatically enabling wald_analysis to seed step sizes.")
        wald_analysis = True

    from datetime import datetime
    _progress_overlay_state["timestamp"] = datetime.now().strftime("%Y%m%d_%H%M%S")

    try:
        from scipy.optimize import minimize
    except ImportError:
        raise ImportError("scipy is required for run_optimization_from_groups") from None

    data_path = paths["data_path"]

    # =========================================================================
    # DECOUPLED NESTED SPEC ROUTE
    # =========================================================================
    if optimization_spec is not None:
        param_names = optimization_spec.param_names
        x0 = optimization_spec.x0
        bounds = optimization_spec.bounds
        method = optimization_spec.method
        optimizer_kwargs = optimization_spec.optimizer_kwargs or {}
        selected_group_names = set(optimization_spec.groups.keys())
        groups_tag = "_".join(sorted(selected_group_names))

        unique_sim_names = set()
        for g_name, g_config in optimization_spec.groups.items():
            for elem in g_config.get("loss_elements", []):
                is_composite = elem.get("type") == "composite" or "simulations" in elem
                if is_composite:
                    sub_sims = elem.get("simulations", [])
                    unique_sim_names.update(sub_sims)
                    data_sim = elem.get("data_simulation") or (sub_sims[0] if sub_sims else None)
                    if data_sim:
                        unique_sim_names.add(data_sim)
                else:
                    unique_sim_names.add(elem.get("simulation"))
        unique_sim_names.update(optimization_spec.passive_simulations)

        _events_dynamic = (optimizer_kwargs.get("events_depend_on_opt_param", False)
                           if optimizer_kwargs else False)

        models = {}
        replicates = {}
        for sim_name in unique_sim_names:
            if sim_name not in experiment.replicates:
                print(f"Warning: Simulation '{sim_name}' not found in Experiment replicates.")
                continue
            replicate = experiment.replicates[sim_name]
            df_dict = replicate["Data"](replicate, data_path)

            if _events_dynamic:
                r_ic = TelluriumGen(model_text, paths)
                r_ic_proxy = OptRoadRunnerProxy(r_ic, param_names)
                replicate["Update_parameters"](r_ic_proxy, replicate)
                try:
                    events_str = replicate["Events"](replicate, df_dict, r_ic=r_ic)
                except TypeError:
                    events_str = replicate["Events"](replicate, df_dict)
            else:
                r_ic = None
                try:
                    events_str = replicate["Events"](replicate, df_dict, r_ic=None)
                except TypeError:
                    events_str = replicate["Events"](replicate, df_dict)

            r = TelluriumGen(model_text + "\n" + events_str, paths)
            r_proxy = OptRoadRunnerProxy(r, param_names)
            replicate["Update_parameters"](r_proxy, replicate)

            models[sim_name] = {"r_ic": r_ic, "r": r, "df_dict": df_dict}
            replicates[sim_name] = replicate

        n_points_by_key = {}
        for g_name, g_config in optimization_spec.groups.items():
            for elem in g_config.get("loss_elements", []):
                lc_fn = elem.get("loss_config")
                is_composite = elem.get("type") == "composite" or "simulations" in elem
                if is_composite:
                    sub_sims = elem.get("simulations", [])
                    data_sim = elem.get("data_simulation") or (sub_sims[0] if sub_sims else None)
                    if data_sim in models:
                        lc = lc_fn(replicates[data_sim]) if callable(lc_fn) else lc_fn
                        print(f"[debug_n_points] composite g_name={g_name} data_sim={data_sim} type(lc)={type(lc)} lc={lc}")
                        n_points_by_key[g_name + "_composite"] = _count_replicate_data_points(
                            models[data_sim]["df_dict"], lc
                        )
                else:
                    sim = elem.get("simulation")
                    if sim in models:
                        lc = lc_fn(replicates[sim]) if callable(lc_fn) else lc_fn
                        print(f"[debug_n_points] sim={sim} type(lc)={type(lc)} lc={lc} lc_fn={lc_fn}")
                        n_points_by_key[sim] = _count_replicate_data_points(
                            models[sim]["df_dict"], lc
                        )

        _debug_calls = [0]
        _progress = {"best": float("inf"), "t0": None}

        def set_params(r, p):
            set_parameters_from_dict(r, p)

        def objective(x):
            call_n = _debug_calls[0]
            do_debug = call_n < 3
            if call_n == 0:
                _progress["t0"] = time.time()
            if do_debug:
                print(f"\n[opt debug] call #{call_n + 1}  "
                      + "  ".join(f"{n}={v:.4g}" for n, v in zip(param_names, x.tolist())))

            x_dict = dict(zip(param_names, np.atleast_1d(x).tolist()))
            events_dynamic = optimizer_kwargs.get("events_depend_on_opt_param", False) if optimizer_kwargs else False

            sim_results = {}
            for sim_name, replicate in replicates.items():
                m = models[sim_name]
                if events_dynamic:
                    r_ic = m["r_ic"]
                    r_ic.reset()
                    set_parameters_from_dict(r_ic, x_dict)
                    try:
                        events_str = replicate["Events"](replicate, m["df_dict"], r_ic=r_ic)
                    except TypeError:
                        events_str = replicate["Events"](replicate, m["df_dict"])
                    r_new = TelluriumGen(model_text + "\n" + events_str, paths)
                    r_proxy = OptRoadRunnerProxy(r_new, param_names)
                    replicate["Update_parameters"](r_proxy, replicate)
                    r_to_use = r_new
                else:
                    m["r"].reset()
                    r_to_use = m["r"]

                try:
                    run_res = run_all(r_to_use, sim_name, replicate, m["df_dict"],
                                      set_parameters=set_params, parameters=x_dict)
                    sim_results[sim_name] = run_res[sim_name]
                except Exception as e:
                    print(f"Error simulating '{sim_name}': {e}")
                    return 1e10

            total_loss = 0.0
            loss_components = {}
            trace_collector = {}

            for g_name, g_config in optimization_spec.groups.items():
                g_loss_sum = 0.0
                g_weight_sum = 0.0

                for idx, elem in enumerate(g_config.get("loss_elements", [])):
                    elem_weight = elem.get("weight", 1.0)
                    lc_fn = elem.get("loss_config")

                    is_composite = elem.get("type") == "composite" or "simulations" in elem
                    if is_composite:
                        sub_sims = elem.get("simulations", [])
                        sub_results = [
                            {
                                "results": sim_results[s]["results"],
                                "replicate": replicates[s],
                                "df_dict": sim_results[s]["data"]
                            }
                            for s in sub_sims if s in sim_results
                        ]
                        data_sim = elem.get("data_simulation") or (sub_sims[0] if sub_sims else None)
                        if not sub_results or data_sim not in sim_results:
                            continue

                        lc = lc_fn(replicates[data_sim]) if callable(lc_fn) else lc_fn
                        loss_val = loss_function_composite(
                            x, sub_results, sim_results[data_sim]["data"],
                            f"{g_name}_composite_{idx}", elem, param_names,
                            loss_config=lc, trace_collector=trace_collector
                        )
                        loss_components[f"{g_name}_composite_{idx}"] = loss_val * elem_weight
                    else:
                        sim = elem.get("simulation")
                        if sim not in sim_results:
                            continue

                        lc = lc_fn(replicates[sim]) if callable(lc_fn) else lc_fn
                        loss_val = loss_function_evaluated(
                            x_dict, {f"{g_name} · {sim}": sim_results[sim]}, param_names,
                            loss_config=lc, trace_collector=trace_collector
                        )
                        loss_components[f"{g_name} · {sim}"] = loss_val * elem_weight

                    if loss_val >= 1e10:
                        return 1e10

                    g_loss_sum += loss_val * elem_weight
                    g_weight_sum += elem_weight

                if g_weight_sum > 0:
                    if optimization_spec.group_normalization == "mean_over_groups":
                        g_loss = g_loss_sum / g_weight_sum
                    else:
                        g_loss = g_loss_sum
                else:
                    g_loss = 0.0

                total_loss += g_loss * g_config.get("group_weight", 1.0)

            if do_debug:
                print(f"  -> total_loss = {total_loss:.6g}")
                if loss_components:
                    comp_str = "  ".join(f"{k}={v:.4g}" for k, v in loss_components.items())
                    print(f"    components: {comp_str}")
            else:
                n = call_n + 1
                new_best = total_loss < _progress["best"]
                if new_best:
                    _progress["best"] = total_loss
                if n % 10 == 0 or new_best:
                    elapsed = time.time() - _progress["t0"]
                    rate = n / elapsed if elapsed > 0 else 0.0
                    tag = "*" if new_best else " "
                    print(
                        f"  [opt]{tag}eval {n:5d}  loss={total_loss:.5g}"
                        f"  best={_progress['best']:.5g}"
                        f"  {elapsed:6.0f}s  ({rate:.1f} eval/s)",
                        flush=True,
                    )
                    if loss_components:
                        comp_str = "  ".join(f"{k}={v:.4g}" for k, v in loss_components.items())
                        print(f"         components: {comp_str}", flush=True)
                    _render_progress_overlay(
                        trace_collector, total_loss, _progress["best"],
                        n, paths.get("plot_path"),
                        param_names=param_names, param_values=x,
                        model_name=paths.get("MODEL_NAME", ""),
                        experiment_id=groups_tag, method=method
                    )
            _debug_calls[0] += 1
            return total_loss

        opt_kw = _prepare_optimizer_kwargs(method, optimizer_kwargs, fast, maxiter, tol)
        profile_without_opt = optimizer_kwargs.get("profile_without_opt", False) if optimizer_kwargs else False
        if profile_without_opt:
            from scipy.optimize import OptimizeResult
            x0_arr = np.array(x0)
            res = OptimizeResult(x=x0_arr, fun=objective(x0_arr), success=True, message="Optimization bypassed", nit=0, nfev=1)
        elif method.lower() in _GLOBAL_METHODS:
            res = _run_global_optimization(objective, x0, bounds, method, opt_kw)
        else:
            res = minimize(objective, x0, method=method, bounds=bounds or [], **opt_kw)

        if _debug_calls[0] > 3:
            elapsed = time.time() - _progress["t0"]
            print(f"\n  [opt] done — {_debug_calls[0]} evals in {elapsed:.0f}s"
                  f"  best={_progress['best']:.5g}")

        out = {
            "x": res.x, "fun": res.fun, "success": res.success,
            "message": res.message, "stats": {},
            "groups": sorted(selected_group_names),
            "nit":  getattr(res, "nit",  None),
            "nfev": getattr(res, "nfev", None),
            "timestamp": _progress_overlay_state.get("timestamp"),
        }

        param_dict = dict(zip(param_names, res.x.tolist()))
        best_results = {}
        fixed_sigmas = {}
        total_n = 0
        k = len(param_names)

        # Build passive simulations in results for figure drawing
        for sim_name, replicate in replicates.items():
            m = models[sim_name]
            res_dict = run_all(m["r"], sim_name, replicate, m["df_dict"],
                               set_parameters=set_params, parameters=param_dict)
            best_results.update(res_dict)

        for g_name, g_config in optimization_spec.groups.items():
            for idx, elem in enumerate(g_config.get("loss_elements", [])):
                lc_fn = elem.get("loss_config")

                is_composite = elem.get("type") == "composite" or "simulations" in elem
                if is_composite:
                    sub_sims = elem.get("simulations", [])
                    sub_results = [
                        {
                            "results": best_results[s]["results"],
                            "replicate": replicates[s],
                            "df_dict": best_results[s]["data"]
                        }
                        for s in sub_sims if s in best_results
                    ]
                    data_sim = elem.get("data_simulation") or (sub_sims[0] if sub_sims else None)
                    if not sub_results or data_sim not in best_results:
                        continue
                    item_df = best_results[data_sim]["data"]

                    lc_fn = elem.get("loss_config")
                    lc = lc_fn(replicates[data_sim]) if callable(lc_fn) else lc_fn
                    observables = lc.get("observables", []) if lc else []

                    for obs_cfg in observables:
                        obs = obs_cfg["observed_variable"]
                        d_col = obs_cfg["data_column"]
                        t_col = obs_cfg["time_column"]
                        obs_df = _resolve_obs_df(item_df, obs_cfg)
                        if obs_df is None or d_col not in obs_df.columns or t_col not in obs_df.columns:
                            continue

                        y_data = np.asarray(obs_df[d_col])
                        t_data = np.asarray(obs_df[t_col])

                        y_pred_subs = []
                        for item in sub_results:
                            result = item["results"]
                            rep = item["replicate"]
                            df_dict_s = item["df_dict"]
                            t_sim = np.asarray(result["time"])

                            if callable(lc_fn) and rep is not None:
                                lc_s = lc_fn(rep)
                                obs_cfg_s = None
                                for oc in lc_s.get("observables", []):
                                    if oc.get("data_column") == d_col:
                                        obs_cfg_s = oc
                                        break
                                if obs_cfg_s is None:
                                    obs_cfg_s = obs_cfg
                            else:
                                obs_cfg_s = obs_cfg

                            obs_s = obs_cfg_s["observed_variable"]
                            obs_df_s = _resolve_obs_df(df_dict_s, obs_cfg_s)
                            if obs_df_s is None or d_col not in obs_df_s.columns or t_col not in obs_df_s.columns:
                                continue

                            t_data_s = np.asarray(obs_df_s[t_col])

                            if callable(obs_s):
                                y_sim = np.asarray(obs_s(result))
                            elif isinstance(obs_s, str):
                                cols = (result.colnames if hasattr(result, "colnames")
                                        else (result.dtype.names if hasattr(result, "dtype") else []))
                                if obs_s in cols:
                                    y_sim = np.asarray(result[obs_s])
                                else:
                                    y_sim = np.zeros_like(t_sim)
                            else:
                                continue
                            y_pred_subs.append(np.interp(t_data_s, t_sim, y_sim))

                        if not y_pred_subs:
                            continue
                        y_pred = np.mean(y_pred_subs, axis=0)
                        valid = np.isfinite(y_pred) & np.isfinite(y_data)
                        if not valid.any():
                            continue
                        residuals = y_data[valid] - y_pred[valid]
                        sigma = np.sqrt(np.sum(residuals**2) / max(1, len(residuals) - k/max(1, len(observables))))
                        fixed_sigmas[(f"{g_name}_composite_{idx}", obs)] = max(sigma, 1e-6)
                        total_n += len(residuals)
                else:
                    sim = elem.get("simulation")
                    if sim not in best_results:
                        continue
                    item = best_results[sim]
                    result = item["results"]
                    item_df = item["data"]
                    t_sim = np.asarray(result["time"])

                    lc = lc_fn(replicates[sim]) if callable(lc_fn) else lc_fn
                    observables = lc.get("observables", []) if lc else []

                    for obs_cfg in observables:
                        obs = obs_cfg["observed_variable"]
                        d_col = obs_cfg["data_column"]
                        t_col = obs_cfg["time_column"]
                        obs_df = _resolve_obs_df(item_df, obs_cfg)
                        if obs_df is None or d_col not in obs_df.columns or t_col not in obs_df.columns:
                            continue
                        y_data = np.asarray(obs_df[d_col])
                        t_data = np.asarray(obs_df[t_col])
                        if callable(obs):
                            y_sim = np.asarray(obs(result))
                        elif isinstance(obs, str):
                            cols = (result.colnames if hasattr(result, "colnames")
                                    else (result.dtype.names if hasattr(result, "dtype") else []))
                            if obs in cols:
                                y_sim = np.asarray(result[obs])
                            else:
                                y_sim = np.zeros_like(t_sim)
                        else:
                            continue
                        y_pred = np.interp(t_data, t_sim, y_sim)
                        valid = np.isfinite(y_pred) & np.isfinite(y_data)
                        if not valid.any():
                            continue
                        residuals = y_data[valid] - y_pred[valid]
                        sigma = np.sqrt(np.sum(residuals**2) / max(1, len(residuals) - k/max(1, len(observables))))
                        fixed_sigmas[(sim, obs)] = max(sigma, 1e-6)
                        total_n += len(residuals)

        def nll_func_fixed(p):
            p_dict = dict(zip(param_names, np.atleast_1d(p).tolist()))
            fixed_results = {}
            for sim_name, replicate in replicates.items():
                m = models[sim_name]
                m["r"].reset()
                run_res = run_all(m["r"], sim_name, replicate, m["df_dict"],
                                  set_parameters=set_params, parameters=p_dict)
                fixed_results[sim_name] = run_res[sim_name]

            total_nll = 0.0
            for g_name, g_config in optimization_spec.groups.items():
                g_loss_sum = 0.0
                g_weight_sum = 0.0
                for idx, elem in enumerate(g_config.get("loss_elements", [])):
                    elem_weight = elem.get("weight", 1.0)
                    lc_fn = elem.get("loss_config")

                    is_composite = elem.get("type") == "composite" or "simulations" in elem
                    if is_composite:
                        sub_sims = elem.get("simulations", [])
                        sub_results = [
                            {
                                "results": fixed_results[s]["results"],
                                "replicate": replicates[s],
                                "df_dict": fixed_results[s]["data"]
                            }
                            for s in sub_sims if s in fixed_results
                        ]
                        data_sim = elem.get("data_simulation") or (sub_sims[0] if sub_sims else None)
                        if not sub_results or data_sim not in fixed_results:
                            continue
                        lc = lc_fn(replicates[data_sim]) if callable(lc_fn) else lc_fn
                        loss_val = loss_function_composite(
                            p, sub_results, fixed_results[data_sim]["data"],
                            f"{g_name}_composite_{idx}", elem, param_names,
                            loss_config=lc, fixed_sigmas=fixed_sigmas
                        )
                    else:
                        sim = elem.get("simulation")
                        if sim not in fixed_results:
                            continue
                        lc = lc_fn(replicates[sim]) if callable(lc_fn) else lc_fn
                        loss_val = loss_function_evaluated(
                            p_dict, {sim: fixed_results[sim]}, param_names,
                            loss_config=lc, fixed_sigmas=fixed_sigmas
                        )
                    g_loss_sum += loss_val * elem_weight
                    g_weight_sum += elem_weight

                if g_weight_sum > 0:
                    if optimization_spec.group_normalization == "mean_over_groups":
                        g_loss = g_loss_sum / g_weight_sum
                    else:
                        g_loss = g_loss_sum
                else:
                    g_loss = 0.0
                total_nll += g_loss * g_config.get("group_weight", 1.0)
            return total_nll

        out["results_dict"] = best_results

        if wald_analysis or slice_analysis or profile_likelihood_analysis or sobol_analysis:
            nll_proper = nll_func_fixed(res.x)
            aic = 2 * k + 2 * nll_proper
            bic = k * np.log(max(total_n, 1)) + 2 * nll_proper
            out["stats"]["aic"] = aic
            out["stats"]["bic"] = bic
            out["stats"]["nll_proper"] = nll_proper

        if wald_analysis:
            try:
                print(f"\n[Wald] Computing Hessian matrix for {k} parameters... (this requires many silent evaluations)")
                hessian_raw = compute_hessian_numdifftools(nll_func_fixed, res.x)
                fisher_info = (hessian_raw + hessian_raw.T) / 2.0
                eigenvalues = np.linalg.eigvals(fisher_info)
                if np.any(eigenvalues <= 1e-10):
                    print("Warning: Hessian is NOT positive definite.")
                    cov_matrix      = None
                    standard_errors = None
                else:
                    cov_matrix      = np.linalg.inv(fisher_info)
                    diag_elements   = np.diag(cov_matrix)
                    standard_errors = (None if np.any(diag_elements < 0)
                                       else np.sqrt(diag_elements))
                out["stats"]["wald_fisher_info"] = fisher_info
                out["stats"]["wald_covariance"]  = cov_matrix
                out["stats"]["wald_se"]          = standard_errors
                if standard_errors is not None:
                    cv = stats.norm.ppf(1 - 0.05 / 2)
                    out["stats"]["wald_ci"] = [
                        (p_val - cv * se, p_val + cv * se)
                        for p_val, se in zip(res.x, standard_errors)
                    ]
                if cov_matrix is not None:
                    out["stats"]["wald_correlation"] = compute_parameter_correlations(cov_matrix)
            except Exception as e:
                print(f"Error computing Wald statistics: {e}")

        if slice_analysis or profile_likelihood_analysis:
            try:
                nll_at_optimum = nll_func_fixed(res.x)
                out["stats"]["nll_at_optimum"] = nll_at_optimum
                out["stats"]["fixed_sigmas"]   = fixed_sigmas

                def likelihood_slice_func(param_idx, n_points=20, range_factor=2.0):
                    p_val      = res.x[param_idx]
                    pname      = param_names[param_idx]
                    param_vals = np.linspace(p_val / range_factor, p_val * range_factor, n_points)
                    width      = len(str(n_points))
                    print(f"\n[slice] {pname}  ({n_points} points, x{range_factor} range)")
                    nll_vals = []
                    for i, val in enumerate(param_vals):
                        nll = nll_func_fixed(np.where(np.arange(len(res.x)) == param_idx, val, res.x))
                        print(f"  [{i+1:{width}d}/{n_points}]  {pname}={val:.4g}  nll={nll:.6g}")
                        nll_vals.append(nll)
                    return param_vals, np.array(nll_vals) - nll_at_optimum

                if slice_analysis:
                    out["stats"]["likelihood_slice"] = likelihood_slice_func

                if profile_likelihood_analysis:
                    def true_profile_likelihood_func(param_idx, n_points=20, range_factor=2.0):
                        se_array = out["stats"].get("wald_se")
                        wald_se_val = se_array[param_idx] if se_array is not None else None
                        return _run_pypesto_profile_single(
                            param_idx, nll_func_fixed, bounds, res.x, nll_at_optimum,
                            param_names, n_points=n_points, range_factor=range_factor,
                            fallback_func=likelihood_slice_func, wald_se_val=wald_se_val
                        )
                    out["stats"]["profile_likelihood"]  = true_profile_likelihood_func
            except Exception as e:
                print(f"Warning: likelihood analysis setup failed: {e}")

        if sobol_analysis:
            from Engine.Sensitivity_analysis import run_sobol_analysis
            skwargs = sobol_kwargs or {}
            out["stats"]["sobol"] = run_sobol_analysis(
                nll_func_fixed, param_names, bounds, res.x, **skwargs
            )

        out["r"] = list(models.values())[0]["r"] if models else None
        return out

    # =========================================================================
    # LEGACY / FLAT REPLICATE ROUTE
    # =========================================================================

    opt_groups = experiment.opt_groups  # {opt_group: [key, ...]}
    if group_names is None:
        selected_group_names = set(opt_groups.keys())
    else:
        selected_group_names = set(group_names)
        missing = selected_group_names - set(opt_groups.keys())
        if missing:
            raise ValueError(
                f"Optimization groups not found: {sorted(missing)}. "
                f"Available: {sorted(opt_groups.keys())}"
            )
    groups_tag = "_".join(sorted(selected_group_names))

    def _effective_loss_cfg(treatment):
        lc = treatment.get("Loss_config", no_optimization)
        return lc(treatment)

    # r_ic is only used when events depend on optimizer parameters (dynamic
    # event rebuild path). Skip the second model compile when it is not needed.
    _events_dynamic = (optimizer_kwargs.get("events_depend_on_opt_param", False)
                       if optimizer_kwargs else False)

    # Pre-build one Tellurium model per replicate in selected groups.
    models     = {}
    replicates = {}
    for key, replicate in experiment.replicates.items():
        if replicate.get("Opt_group") not in selected_group_names:
            continue
        df_dict    = replicate["Data"](replicate, data_path)

        if _events_dynamic:
            r_ic       = TelluriumGen(model_text, paths)
            r_ic_proxy = OptRoadRunnerProxy(r_ic, param_names)
            replicate["Update_parameters"](r_ic_proxy, replicate)
            try:
                events_str = replicate["Events"](replicate, df_dict, r_ic=r_ic)
            except TypeError:
                events_str = replicate["Events"](replicate, df_dict)
        else:
            r_ic = None
            try:
                events_str = replicate["Events"](replicate, df_dict, r_ic=None)
            except TypeError:
                events_str = replicate["Events"](replicate, df_dict)

        r          = TelluriumGen(model_text + "\n" + events_str, paths)
        r_proxy    = OptRoadRunnerProxy(r, param_names)
        replicate["Update_parameters"](r_proxy, replicate)

        models[key]     = {"r_ic": r_ic, "r": r, "df_dict": df_dict}
        replicates[key] = replicate

    if not models:
        raise ValueError("No valid replicates found across selected groups.")

    # Pre-compute finite data point counts per active replicate (informational
    # only — loss_function already per-point-averages each observable via /n_eff
    # so the objective sums replicate contributions directly without further
    # /n_pts normalization).
    n_points_by_key = {}
    for key, replicate in replicates.items():
        effective_lc = _effective_loss_cfg(replicate)
        if effective_lc and effective_lc.get("observables"):
            n_points_by_key[key] = _count_replicate_data_points(
                models[key]["df_dict"], effective_lc
            )
    print("[opt] Data point counts per replicate:")
    for key, n_pts in n_points_by_key.items():
        print(f"  {key}: {n_pts} points")

    # Objective: sum NLL over active replicates only.
    # First 3 calls print diagnostics to help identify time-alignment issues.
    # Subsequent calls emit a single overwritten progress line every 10 evals.
    _debug_calls = [0]
    _progress    = {"best": float("inf"), "t0": None}

    def objective(x):
        call_n   = _debug_calls[0]
        do_debug = call_n < 3
        if call_n == 0:
            _progress["t0"] = time.time()
        if do_debug:
            print(f"\n[opt debug] call #{call_n + 1}  "
                  + "  ".join(f"{n}={v:.4g}" for n, v in zip(param_names, x.tolist())))
        total_loss = 0.0
        loss_components = {}
        trace_collector = {}
        x_dict = dict(zip(param_names, np.atleast_1d(x).tolist()))
        events_dynamic = optimizer_kwargs.get("events_depend_on_opt_param", False) if optimizer_kwargs else False

        # Gather active tasks
        tasks = []
        for key, replicate in replicates.items():
            effective_lc = _effective_loss_cfg(replicate)
            if not effective_lc or not effective_lc.get("observables"):
                continue
            tasks.append((key, replicate, effective_lc))

        if not tasks:
            _debug_calls[0] += 1
            return 0.0

        if events_dynamic:
            for key, replicate, effective_lc in tasks:
                m = models[key]
                r_ic = m["r_ic"]
                r_ic.reset()
                set_parameters_from_dict(r_ic, x_dict)
                try:
                    events_str = replicate["Events"](replicate, m["df_dict"], r_ic=r_ic)
                except TypeError:
                    events_str = replicate["Events"](replicate, m["df_dict"])

                r_new = TelluriumGen(model_text + "\n" + events_str, paths)
                r_proxy = OptRoadRunnerProxy(r_new, param_names)
                replicate["Update_parameters"](r_proxy, replicate)
                
                loss_val = loss_function(
                    x, r_new, key, replicate,
                    m["df_dict"], param_names,
                    loss_config=effective_lc,
                    trace_collector=trace_collector,
                )
                if loss_val >= 1e10:
                    _debug_calls[0] += 1
                    return 1e10
                rep_weight = effective_lc.get("replicate_weight", 1.0)
                loss_components[key] = loss_val * rep_weight
                total_loss += loss_val * rep_weight
        else:
            for key, replicate, effective_lc in tasks:
                m = models[key]
                try:
                    loss_val = loss_function(
                        x, m["r"], key, replicate,
                        m["df_dict"], param_names,
                        loss_config=effective_lc,
                        trace_collector=trace_collector,
                    )
                except Exception as e:
                    print(f"Error evaluating replicate {key}: {e}")
                    _debug_calls[0] += 1
                    return 1e10
                if loss_val >= 1e10:
                    _debug_calls[0] += 1
                    return 1e10
                rep_weight = effective_lc.get("replicate_weight", 1.0)
                weighted_loss = loss_val * rep_weight
                loss_components[key] = weighted_loss
                total_loss += weighted_loss
        if do_debug:
            print(f"  -> total_loss = {total_loss:.6g}")
            if loss_components:
                comp_str = "  ".join(f"{k}={v:.4g}" for k, v in loss_components.items())
                print(f"    components: {comp_str}")
        else:
            n        = call_n + 1
            new_best = total_loss < _progress["best"]
            if new_best:
                _progress["best"] = total_loss
            if n % 10 == 0 or new_best:
                elapsed = time.time() - _progress["t0"]
                rate    = n / elapsed if elapsed > 0 else 0.0
                tag = "*" if new_best else " "
                print(
                    f"  [opt]{tag}eval {n:5d}  loss={total_loss:.5g}"
                    f"  best={_progress['best']:.5g}"
                    f"  {elapsed:6.0f}s  ({rate:.1f} eval/s)",
                    flush=True,
                )
                if loss_components:
                    comp_str = "  ".join(f"{k}={v:.4g}" for k, v in loss_components.items())
                    print(f"         components: {comp_str}", flush=True)
                _render_progress_overlay(
                    trace_collector, total_loss, _progress["best"],
                    n, paths.get("plot_path"),
                    param_names=param_names, param_values=x,
                    model_name=paths.get("MODEL_NAME", ""),
                    experiment_id=groups_tag, method=method
                )
        _debug_calls[0] += 1
        return total_loss

    opt_kw = _prepare_optimizer_kwargs(method, optimizer_kwargs, fast, maxiter, tol)
    
    profile_without_opt = optimizer_kwargs.get("profile_without_opt", False) if optimizer_kwargs else False
    if profile_without_opt:
        from scipy.optimize import OptimizeResult
        x0_arr = np.array(x0)
        res = OptimizeResult(x=x0_arr, fun=objective(x0_arr), success=True, message="Optimization bypassed", nit=0, nfev=1)
    elif method.lower() in _GLOBAL_METHODS:
        res = _run_global_optimization(objective, x0, bounds, method, opt_kw)
    else:
        res = minimize(objective, x0, method=method, bounds=bounds or [], **opt_kw)
        
    if _debug_calls[0] > 3:
        elapsed = time.time() - _progress["t0"]
        print(f"\n  [opt] done — {_debug_calls[0]} evals in {elapsed:.0f}s"
              f"  best={_progress['best']:.5g}")

    out = {
        "x": res.x, "fun": res.fun, "success": res.success,
        "message": res.message, "stats": {},
        "groups": sorted(selected_group_names),
        "nit":  getattr(res, "nit",  None),
        "nfev": getattr(res, "nfev", None),
        "timestamp": _progress_overlay_state.get("timestamp"),
    }

    if not res.success:
        print(f"Warning: optimizer reported non-convergence — {res.message}")
        print("Proceeding with best-found parameters for results and profile likelihood.")

    param_dict = dict(zip(param_names, res.x.tolist()))

    def set_params(r, p):
        set_parameters_from_dict(r, p)

    # The optimizer minimized a z-score χ² objective for balanced dataset
    # weighting; the proper joint NLL for AIC/BIC and Hessian-based CIs is
    # computed below from nll_func_fixed(res.x) once fixed_sigmas exists.

    # Run selected replicates at optimal params; passive ones produce plot data only.
    best_results = {}
    fixed_sigmas = {}
    total_n      = 0
    k            = len(param_names)

    for key, replicate in replicates.items():
        m        = models[key]
        res_dict = run_all(m["r"], key, replicate, m["df_dict"],
                           set_parameters=set_params, parameters=param_dict)
        best_results.update(res_dict)

        effective_lc       = _effective_loss_cfg(replicate)
        observables_config = effective_lc.get("observables", [])
        if not observables_config:
            continue

        for _, item in res_dict.items():
            result  = item["results"]
            item_df = item["data"]
            t_sim   = np.asarray(result["time"])

            local_dict = {"np": np, "time": t_sim}
            cols = (result.colnames if hasattr(result, "colnames") else
                    (result.dtype.names if hasattr(result, "dtype") else []))
            for c in cols:
                local_dict[c] = np.asarray(result[c])
                if c.startswith('[') and c.endswith(']'):
                    local_dict[c[1:-1]] = np.asarray(result[c])
            local_dict.update(param_dict)

            for obs_cfg in observables_config:
                obs   = obs_cfg["observed_variable"]
                d_col = obs_cfg["data_column"]
                t_col = obs_cfg["time_column"]
                obs_df = _resolve_obs_df(item_df, obs_cfg)
                if obs_df is None or d_col not in obs_df.columns or t_col not in obs_df.columns:
                    continue

                y_data = np.asarray(obs_df[d_col])
                t_data = np.asarray(obs_df[t_col])

                try:
                    if callable(obs):
                        y_sim = np.asarray(obs(result))
                    elif isinstance(obs, str) and obs in cols:
                        y_sim = np.asarray(result[obs])
                    elif isinstance(obs, str):
                        eval_obs = str(obs)
                        for c in cols:
                            if c.startswith('[') and c.endswith(']'):
                                eval_obs = eval_obs.replace(c, c[1:-1])
                        y_sim = np.asarray(eval(eval_obs, {}, local_dict))
                    else:
                        continue
                except Exception:
                    continue

                y_pred = np.interp(t_data, t_sim, y_sim)
                valid = np.isfinite(y_pred) & np.isfinite(y_data)
                if not valid.any():
                    continue
                y_data_v  = y_data[valid]
                y_pred_v  = y_pred[valid]
                residuals = y_data_v - y_pred_v

                sigma_config = obs_cfg.get("noise_formula", None)
                if sigma_config and sigma_config in local_dict:
                    sigma = float(local_dict[sigma_config])
                else:
                    n_block = len(residuals)
                    if n_block > 1:
                        sigma = np.sqrt(np.sum(residuals**2) /
                                         max(1, n_block - k / max(1, len(observables_config))))
                    elif n_block == 1:
                        sigma = max(np.abs(y_data_v[0]) * 0.1, 1e-6)
                    else:
                        sigma = 1e-6
                fixed_sigmas[(key, obs)] = sigma
                total_n += len(residuals)

    # Simulate replicates NOT in selected groups at optimal params so that
    # plot functions receive a complete results_dict.
    for key, replicate in experiment.replicates.items():
        if replicate.get("Opt_group") in selected_group_names:
            continue
        df_dict    = replicate["Data"](replicate, data_path)

        if _events_dynamic:
            r_ic       = TelluriumGen(model_text, paths)
            r_ic_proxy = OptRoadRunnerProxy(r_ic, param_names)
            replicate["Update_parameters"](r_ic_proxy, replicate)
            try:
                events_str = replicate["Events"](replicate, df_dict, r_ic=r_ic)
            except TypeError:
                events_str = replicate["Events"](replicate, df_dict)
        else:
            try:
                events_str = replicate["Events"](replicate, df_dict, r_ic=None)
            except TypeError:
                events_str = replicate["Events"](replicate, df_dict)

        r          = TelluriumGen(model_text + "\n" + events_str, paths)
        r_proxy    = OptRoadRunnerProxy(r, param_names)
        replicate["Update_parameters"](r_proxy, replicate)

        res_dict   = run_all(r, key, replicate, df_dict,
                             set_parameters=set_params, parameters=param_dict)
        best_results.update(res_dict)

    def nll_func_fixed(p):
        p_dict = dict(zip(param_names, np.atleast_1d(p).tolist()))
        events_dynamic = optimizer_kwargs.get("events_depend_on_opt_param", False) if optimizer_kwargs else False

        # Gather active tasks
        tasks = []
        for key, replicate in replicates.items():
            effective_lc = _effective_loss_cfg(replicate)
            if not effective_lc or not effective_lc.get("observables"):
                continue
            tasks.append((key, replicate, effective_lc))

        if not tasks:
            return 0.0

        total_nll = 0.0

        if events_dynamic:
            for key, replicate, effective_lc in tasks:
                m = models[key]
                r_ic = m["r_ic"]
                r_ic.reset()
                set_parameters_from_dict(r_ic, p_dict)
                try:
                    events_str = replicate["Events"](replicate, m["df_dict"], r_ic=r_ic)
                except TypeError:
                    events_str = replicate["Events"](replicate, m["df_dict"])

                r_new = TelluriumGen(model_text + "\n" + events_str, paths)
                r_proxy = OptRoadRunnerProxy(r_new, param_names)
                replicate["Update_parameters"](r_proxy, replicate)
                
                rep_weight = effective_lc.get("replicate_weight", 1.0)
                total_nll += rep_weight * loss_function(
                    p, r_new, key, replicate,
                    m["df_dict"], param_names,
                    effective_lc, fixed_sigmas=fixed_sigmas,
                )
        else:
            for key, replicate, effective_lc in tasks:
                m = models[key]
                try:
                    loss_val = loss_function(
                        p, m["r"], key, replicate,
                        m["df_dict"], param_names,
                        effective_lc, fixed_sigmas=fixed_sigmas,
                    )
                except Exception as e:
                    print(f"Error evaluating fixed replicate {key}: {e}")
                    return 1e10
                rep_weight = effective_lc.get("replicate_weight", 1.0)
                total_nll += loss_val * rep_weight

        return total_nll

    out["results_dict"] = best_results

    if wald_analysis or slice_analysis or profile_likelihood_analysis or sobol_analysis:
        nll_proper = nll_func_fixed(res.x)
        aic = 2 * k + 2 * nll_proper
        bic = k * np.log(max(total_n, 1)) + 2 * nll_proper
        out["stats"]["aic"] = aic
        out["stats"]["bic"] = bic
        out["stats"]["nll_proper"] = nll_proper

    if wald_analysis:
        try:
            print(f"\n[Wald] Computing Hessian matrix for {k} parameters... (this requires many silent evaluations)")
            hessian_raw = compute_hessian_numdifftools(nll_func_fixed, res.x)
            fisher_info = (hessian_raw + hessian_raw.T) / 2.0
            eigenvalues = np.linalg.eigvals(fisher_info)
            if np.any(eigenvalues <= 1e-10):
                print("Warning: Hessian is NOT positive definite.")
                cov_matrix      = None
                standard_errors = None
            else:
                cov_matrix      = np.linalg.inv(fisher_info)
                diag_elements   = np.diag(cov_matrix)
                standard_errors = (None if np.any(diag_elements < 0)
                                   else np.sqrt(diag_elements))
            out["stats"]["wald_fisher_info"] = fisher_info
            out["stats"]["wald_covariance"]  = cov_matrix
            out["stats"]["wald_se"]          = standard_errors
            if standard_errors is not None:
                cv = stats.norm.ppf(1 - 0.05 / 2)
                out["stats"]["wald_ci"] = [
                    (p_val - cv * se, p_val + cv * se)
                    for p_val, se in zip(res.x, standard_errors)
                ]
            if cov_matrix is not None:
                out["stats"]["wald_correlation"] = compute_parameter_correlations(cov_matrix)
        except Exception as e:
            print(f"Error computing Wald statistics: {e}")

    if slice_analysis or profile_likelihood_analysis:
        try:
            nll_at_optimum = nll_func_fixed(res.x)
            out["stats"]["nll_at_optimum"] = nll_at_optimum
            out["stats"]["fixed_sigmas"]   = fixed_sigmas

            print(f"\nLikelihood analysis setup diagnostics:")
            print(f"  NLL at optimum (fixed-sigma): {nll_at_optimum:.6g}")
            print(f"  fixed_sigmas populated: {len(fixed_sigmas)} entries")
            if not fixed_sigmas:
                print("  WARNING: fixed_sigmas is EMPTY — observables may not be resolving to data.")
            else:
                for k_fs, v_fs in fixed_sigmas.items():
                    print(f"    sigma[{k_fs}] = {v_fs:.4g}")
            print(f"  Per-parameter NLL sensitivity (1.5x perturbation):")
            for _i, _pname in enumerate(param_names):
                _x_test = res.x.copy()
                _x_test[_i] *= 1.5
                _nll_test = nll_func_fixed(_x_test)
                print(f"    {_pname}: {nll_at_optimum:.6g} -> {_nll_test:.6g}  (delta={_nll_test - nll_at_optimum:+.6g})")

            def likelihood_slice_func(param_idx, n_points=20, range_factor=2.0):
                p_val      = res.x[param_idx]
                pname      = param_names[param_idx]
                param_vals = np.linspace(p_val / range_factor, p_val * range_factor, n_points)
                width      = len(str(n_points))
                print(f"\n[slice] {pname}  ({n_points} points, x{range_factor} range)")
                nll_vals = []
                for i, val in enumerate(param_vals):
                    nll = nll_func_fixed(np.where(np.arange(len(res.x)) == param_idx, val, res.x))
                    print(f"  [{i+1:{width}d}/{n_points}]  {pname}={val:.4g}  nll={nll:.6g}")
                    nll_vals.append(nll)
                return param_vals, np.array(nll_vals) - nll_at_optimum

            if slice_analysis:
                out["stats"]["likelihood_slice"] = likelihood_slice_func

            if profile_likelihood_analysis:
                def true_profile_likelihood_func(param_idx, n_points=20, range_factor=2.0):
                    se_array = out["stats"].get("wald_se")
                    wald_se_val = se_array[param_idx] if se_array is not None else None
                    return _run_pypesto_profile_single(
                        param_idx, nll_func_fixed, bounds, res.x, nll_at_optimum,
                        param_names, n_points=n_points, range_factor=range_factor,
                        fallback_func=likelihood_slice_func, wald_se_val=wald_se_val
                    )

                out["stats"]["profile_likelihood"]  = true_profile_likelihood_func
        except Exception as e:
            print(f"Warning: likelihood analysis setup failed: {e}")

    if sobol_analysis:
        from Engine.Sensitivity_analysis import run_sobol_analysis
        skwargs = sobol_kwargs or {}
        out["stats"]["sobol"] = run_sobol_analysis(
            nll_func_fixed, param_names, bounds, res.x, **skwargs
        )

    out["r"] = list(models.values())[0]["r"] if models else None
    return out


# ---------------------------------------------------------------------------
# Legacy / standalone helpers (kept for backward compatibility)
# ---------------------------------------------------------------------------

def predict_concentrations(params, observable_data, model, param_names, observable_defs=None):
    """
    Simulate the model with given parameters and return predicted concentrations
    for multiple observables.
    """
    model.reset()
    for name, value in zip(param_names, params):
        model[name] = value

    try:
        model.setIntegrator('cvode')
        model.integrator.absolute_tolerance = 1e-8
        model.integrator.relative_tolerance = 1e-8
        model.integrator.setValue('stiff', True)
        model.integrator.variable_step_size = True
        model.integrator.setValue('initial_time_step', 1e-6)
        model.integrator.setValue('maximum_num_steps', 100000)
    except Exception:
        pass

    if isinstance(observable_data, dict):
        predictions    = {}
        all_observables = list(observable_data.keys())
        all_times       = [obs_data['times'] for obs_data in observable_data.values()]
        if not all_times or all(len(t) == 0 for t in all_times):
            return {obs_id: np.array([]) for obs_id in all_observables}

        max_time = float(max(t.max() for t in all_times if len(t) > 0))

        try:
            observed_species = (['time'] + list(model.getFloatingSpeciesIds())
                                + list(model.getBoundarySpeciesIds())
                                + list(model.getAssignmentRuleIds()))
            from Engine.Simulate import safe_simulate
            block = {
                "start": 0.0,
                "end": 1.0,
                "n_points": 2,
                "variable_step_size": True
            }
            test_result, _ = safe_simulate(model, block, observed_species)
            available_variables = set(test_result.colnames) - {'time'}
        except Exception:
            try:
                available_variables = set(
                    list(model.getFloatingSpeciesIds())
                    + list(model.getBoundarySpeciesIds())
                    + list(model.getAssignmentRuleIds())
                )
            except Exception:
                available_variables = set()
                warnings.warn("Could not determine available variables from model")

        variables_to_simulate  = ['time']
        formulas_to_evaluate   = {}
        parameters_in_formulas = set()

        import re
        for obs_id in all_observables:
            formula = (observable_defs[obs_id].get('observableFormula', obs_id)
                       if observable_defs and obs_id in observable_defs else obs_id)
            if obs_id in available_variables:
                if obs_id not in variables_to_simulate:
                    variables_to_simulate.append(obs_id)
            else:
                formulas_to_evaluate[obs_id] = formula
                for var in re.findall(r'\b([A-Za-z_][A-Za-z0-9_]*)\b', formula):
                    if var not in ('time', 'and', 'or', 'not', 'if', 'else',
                                   'True', 'False', 'None', 'np', 'numpy'):
                        if var in available_variables:
                            if var not in variables_to_simulate:
                                variables_to_simulate.append(var)
                        else:
                            parameters_in_formulas.add(var)

        n_points = max(10000, sum(len(t) for t in all_times) * 100)
        try:
            from Engine.Simulate import safe_simulate
            block = {
                "start": 0.0,
                "end": max_time,
                "n_points": n_points,
                "variable_step_size": True
            }
            result, _ = safe_simulate(model, block, variables_to_simulate)
        except Exception as e:
            warnings.warn(f"Simulation error: {e}")
            for obs_id, obs_data in observable_data.items():
                predictions[obs_id] = np.zeros_like(obs_data['times'])
            return predictions

        time_sim    = result['time']
        result_cols = set(getattr(result, 'colnames',
                                  result.keys() if hasattr(result, 'keys') else []))

        for obs_id, obs_data in observable_data.items():
            times = obs_data['times']
            if obs_id in formulas_to_evaluate:
                formula = formulas_to_evaluate[obs_id]
                try:
                    cols      = getattr(result, 'colnames', result.keys())
                    namespace = {col: result[col] for col in cols}
                    for pn in list(param_names) + list(parameters_in_formulas):
                        if pn not in namespace:
                            try:
                                namespace[pn] = model[pn]
                            except Exception:
                                pass
                    try:
                        for pid in model.getGlobalParameterIds():
                            if pid not in namespace:
                                try:
                                    namespace[pid] = model[pid]
                                except Exception:
                                    pass
                    except Exception:
                        pass
                    namespace.update({'np': np, 'numpy': np})
                    obs_values = eval(formula, namespace, {})
                    if np.isscalar(obs_values):
                        obs_values = np.full_like(time_sim, obs_values)
                    predictions[obs_id] = np.interp(times, time_sim, obs_values)
                except Exception as e:
                    warnings.warn(f"Could not evaluate formula for {obs_id}: {formula}. Error: {e}")
                    predictions[obs_id] = np.zeros_like(times)
            elif obs_id in result_cols:
                predictions[obs_id] = np.interp(times, time_sim, result[obs_id])
            else:
                warnings.warn(f"Observable {obs_id} not found in simulation results")
                predictions[obs_id] = np.zeros_like(times)

        return predictions

    # Legacy single-observable path
    if isinstance(observable_data, (list, tuple)) and len(observable_data) == 2:
        observed_var, times = observable_data
    else:
        raise ValueError("Legacy format requires (observed_var, times) tuple")
    from Engine.Simulate import safe_simulate
    block = {
        "start": 0.0,
        "end": times[-1],
        "n_points": len(times)*1000,
        "variable_step_size": True
    }
    result, _ = safe_simulate(model, block, ['time', observed_var])
    return np.interp(times, result['time'], result[observed_var])


def neg_log_likelihood(params, observable_data, model, param_names, observable_defs=None):
    if np.any(params <= 0):
        return 1e10
    try:
        predictions = predict_concentrations(params, observable_data, model, param_names, observable_defs)
        total_nll = 0.0
        for obs_id, obs_data in observable_data.items():
            pred      = predictions[obs_id]
            residuals = obs_data['values'] - pred
            sigma2    = max(np.var(residuals), 1e-6)
            total_nll += -np.sum(stats.norm.logpdf(obs_data['values'], loc=pred, scale=np.sqrt(sigma2)))
        return total_nll
    except Exception as e:
        warnings.warn(f"Error in neg_log_likelihood: {e}")
        return 1e10


def neg_log_likelihood_fixed_sigma(params, observable_data, model, param_names,
                                   sigmas, observable_defs=None):
    if np.any(params <= 0):
        return 1e10
    try:
        predictions = predict_concentrations(params, observable_data, model, param_names, observable_defs)
        total_nll = 0.0
        for obs_id, obs_data in observable_data.items():
            sigma     = sigmas.get(obs_id, 1e-6) if isinstance(sigmas, dict) else sigmas
            total_nll += -np.sum(stats.norm.logpdf(
                obs_data['values'], loc=predictions[obs_id], scale=sigma))
        return total_nll
    except Exception as e:
        warnings.warn(f"Error in neg_log_likelihood_fixed_sigma: {e}")
        return 1e10


def likelihood_slice(param_idx, params_estimated, observable_data, model,
                     param_names, sigmas, observable_defs=None,
                     n_points=20, range_factor=2.0):
    param_values = np.linspace(
        params_estimated[param_idx] / range_factor,
        params_estimated[param_idx] * range_factor,
        n_points,
    )
    nll_values = []
    for val in param_values:
        p_test             = params_estimated.copy()
        p_test[param_idx]  = val
        nll_values.append(neg_log_likelihood_fixed_sigma(
            p_test, observable_data, model, param_names, sigmas, observable_defs))
    return param_values, np.array(nll_values)


def _plot_profile_likelihood(ax, plot_config, opt, params_estimated, param_names):
    """Plot profile likelihood for parameters."""
    params_to_profile = plot_config.get('parameters', param_names)
    n_points          = plot_config.get('n_points', 30)
    range_factor      = plot_config.get('range_factor', 3.0)

    colors  = ['blue', 'green', 'red', 'orange', 'purple']
    markers = ['o', 's', '^', 'D', 'v']

    profile_func = opt.get("stats", {}).get("profile_likelihood")
    if not profile_func:
        print("Warning: no profile_likelihood closure found in opt['stats']")
        return

    for idx, param_name in enumerate(params_to_profile):
        if param_name not in param_names:
            continue
        param_idx = param_names.index(param_name)
        param_vals, nll_vals_rel = profile_func(param_idx, n_points=n_points, range_factor=range_factor)

        param_vals_normalized = param_vals / params_estimated[param_idx]

        color  = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
        ax.plot(param_vals_normalized, nll_vals_rel,
                marker=marker, linestyle='-', label=param_name, linewidth=2, color=color)

    ax.axvline(1.0, color='red', linestyle='--', alpha=0.5, linewidth=1.5, label='Optimal')
    ax.set_xlabel(plot_config.get('xlabel', 'Parameter Value (relative to optimal)'))
    ax.set_ylabel(plot_config.get('ylabel', 'Δ NLL (relative to minimum)'))
    ax.set_title(plot_config.get('title', 'Profile Likelihood (Identifiability Check)'))
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    if 'xlim' in plot_config:
        ax.set_xlim(plot_config['xlim'])
    else:
        ax.set_xlim(0.2, 3.5)


def _extract_profile_ci(param_vals, nll_vals_rel, threshold=1.9207):
    """Interpolate lower/upper CI bounds where ΔNLL crosses threshold (default 95%)."""
    mle_idx = int(np.argmin(nll_vals_rel))
    lo, hi = float('nan'), float('nan')
    for i in range(mle_idx - 1, -1, -1):
        if nll_vals_rel[i] >= threshold:
            x0, x1 = param_vals[i], param_vals[i + 1]
            y0, y1 = nll_vals_rel[i], nll_vals_rel[i + 1]
            if y1 != y0:
                lo = x0 + (threshold - y0) * (x1 - x0) / (y1 - y0)
            break
    for i in range(mle_idx + 1, len(nll_vals_rel)):
        if nll_vals_rel[i] >= threshold:
            x0, x1 = param_vals[i - 1], param_vals[i]
            y0, y1 = nll_vals_rel[i - 1], nll_vals_rel[i]
            if y1 != y0:
                hi = x0 + (threshold - y0) * (x1 - x0) / (y1 - y0)
            break
    return lo, hi


def _plot_likelihood_slice(ax, plot_config, opt, params_estimated, param_names):
    """Plot likelihood slice for parameters."""
    params_to_profile = plot_config.get('parameters', param_names)
    n_points          = plot_config.get('n_points', 30)
    range_factor      = plot_config.get('range_factor', 3.0)

    colors  = ['blue', 'green', 'red', 'orange', 'purple']
    markers = ['o', 's', '^', 'D', 'v']

    slice_func = opt.get("stats", {}).get("likelihood_slice")
    if not slice_func:
        print("Warning: no likelihood_slice closure found in opt['stats']")
        return

    for idx, param_name in enumerate(params_to_profile):
        if param_name not in param_names:
            continue
        param_idx = param_names.index(param_name)
        param_vals, nll_vals_rel = slice_func(param_idx, n_points=n_points, range_factor=range_factor)

        param_vals_normalized = param_vals / params_estimated[param_idx]

        color  = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
        ax.plot(param_vals_normalized, nll_vals_rel,
                marker=marker, linestyle='-', label=param_name, linewidth=2, color=color)

    ax.axvline(1.0, color='red', linestyle='--', alpha=0.5, linewidth=1.5, label='Optimal')
    ax.set_xlabel(plot_config.get('xlabel', 'Parameter Value (relative to optimal)'))
    ax.set_ylabel(plot_config.get('ylabel', 'Δ NLL (relative to minimum)'))
    ax.set_title(plot_config.get('title', 'Likelihood Slice'))
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    if 'xlim' in plot_config:
        ax.set_xlim(plot_config['xlim'])
    else:
        ax.set_xlim(0.2, 3.5)

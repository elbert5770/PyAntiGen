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
        ax.set_title(f"{_short_obs_label(rep, 28)} · {_short_obs_label(obs, 28)}"
                     f"\ncontrib={tr['contrib']:.4g}", fontsize=9)
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


def evaluate_observable(obs, result, param_dict=None, on_error="raise"):
    """Evaluate an observable against a simulation result.

    ``obs`` may be a callable, a bare output name ("MFL42", "[AB42_SP3]"), or a
    Python **expression** over the output columns
    (e.g. "np.log10(Total_Plasma_Antibody/V_Plasma)").

    This exists because the expression case was previously handled in the loss
    but *not* in the sigma-estimation block, which silently substituted zeros
    for anything that was not a literal column name. Sigma was then the RMS of
    the data rather than of the residuals -- inflating it by ~100x for log-scale
    PK data, which suppressed every dNLL built on it by ~10^4 and made profile
    likelihood useless for expression-valued observables. One resolver, used
    everywhere, is what keeps that from recurring.

    ``on_error="zeros"`` returns zeros instead of raising, for callers that must
    not fail; it logs, because silently returning zeros is what caused the bug.

    **Not used by the loss functions on purpose.** ``loss_function_evaluated``
    and ``loss_function_composite`` build the same namespace once per result and
    reuse it across observables (``_ensure_eval_context``). They run millions of
    times per fit, so rebuilding the column namespace per call there would cost
    real time. This resolver is for the once-per-fit paths -- sigma estimation
    and diagnostics -- where clarity matters and the cost does not. If the two
    ever need to converge, cache inside this function rather than removing the
    caching from the hot loop.
    """
    if callable(obs):
        return np.asarray(obs(result))

    if not isinstance(obs, str):
        raise ValueError(f"Invalid observable type: {type(obs)}")

    cols = (result.colnames if hasattr(result, "colnames")
            else (result.dtype.names if hasattr(result, "dtype") and result.dtype.names
                  else []))
    if obs in cols:
        return np.asarray(result[obs])

    # Expression: build a namespace of every output column, with [X] aliased to X.
    local_dict = {"np": np, "numpy": np, "time": np.asarray(result["time"])}
    for col in cols:
        local_dict[col] = np.asarray(result[col])
        if col.startswith("[") and col.endswith("]"):
            local_dict[col[1:-1]] = np.asarray(result[col])
    if param_dict:
        local_dict.update(param_dict)

    eval_obs = str(obs)
    for col in cols:
        if col.startswith("[") and col.endswith("]"):
            eval_obs = eval_obs.replace(col, col[1:-1])
    try:
        return np.asarray(eval(eval_obs, {"__builtins__": {}}, local_dict))
    except Exception as exc:
        if on_error == "zeros":
            print(f"  [observable] could not evaluate {_short_obs_label(obs)!r}: "
                  f"{exc}. Returning zeros — any sigma or loss derived from this "
                  f"will be meaningless.")
            return np.zeros_like(np.asarray(result["time"]))
        raise RuntimeError(f"Failed to evaluate observable '{obs}': {exc}") from exc


def _short_obs_label(obs, max_len=32):
    """Compact, stable display name for an observable.

    Observables may be bare column names, callables, or long inline
    expressions. Plot titles and checkpoint records want something readable, so
    long expressions are elided in the middle -- keeping both ends, which is
    where the distinguishing part of an expression usually lives.
    """
    name = getattr(obs, "__name__", None)
    if name:
        return name
    text = str(obs).strip()
    if len(text) <= max_len:
        return text
    keep = (max_len - 3) // 2
    return f"{text[:keep]}...{text[-keep:]}"


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
                obs_label = _short_obs_label(obs)
                trace_collector[(str(exp_id), obs_label)] = {
                    "t_data":  np.asarray(t_data),
                    "y_data":  np.asarray(y_data),
                    "t_sim":   np.asarray(t_sim),
                    "y_sim":   np.asarray(y_sim),
                    "contrib": float(contrib),
                }

            if debug:
                obs_label = _short_obs_label(obs)
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
            obs_label = _short_obs_label(obs)
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
            obs_label = _short_obs_label(obs)
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
# Group-structured NLL (module level so worker processes can call it)
# ---------------------------------------------------------------------------

def simulate_active_replicates(
    p_dict, models, replicates, param_names,
    model_text=None, paths=None, events_dynamic=False,
):
    """Run every replicate in *replicates* at *p_dict*; return {name: result item}.

    Returns None if any simulation fails, which callers translate into the
    failure sentinel. Kept module level so both the in-process path and pool
    workers run byte-identical code -- if these ever diverged, a parallel run
    would silently disagree with a serial one.
    """
    results = {}
    for sim_name, replicate in replicates.items():
        m = models[sim_name]
        try:
            if events_dynamic:
                r_ic = m["r_ic"]
                r_ic.reset()
                set_parameters_from_dict(r_ic, p_dict)
                try:
                    events_str = replicate["Events"](replicate, m["df_dict"], r_ic=r_ic)
                except TypeError:
                    events_str = replicate["Events"](replicate, m["df_dict"])
                r_to_use = TelluriumGen(model_text + "\n" + events_str, paths)
                r_proxy = OptRoadRunnerProxy(r_to_use, param_names)
                replicate["Update_parameters"](r_proxy, replicate)
            else:
                m["r"].reset()
                r_to_use = m["r"]
            run_res = run_all(r_to_use, sim_name, replicate, m["df_dict"],
                              set_parameters=set_parameters_from_dict,
                              parameters=p_dict)
            results[sim_name] = run_res[sim_name]
        except Exception as e:
            print(f"Error evaluating fixed simulation '{sim_name}': {e}")
            return None
    return results


def accumulate_group_nll(
    sim_results, groups, group_normalization, replicates, param_names,
    p_lin, p_dict, fixed_sigmas=None, trace_collector=None, loss_components=None,
    use_weights=True,
):
    """Combine per-simulation losses into a single scalar.

    Two different quantities are built here, and conflating them was a real bug:

    * **The fitting objective** (``use_weights=True`` with the spec's
      ``group_normalization``). Averaging over loss elements stops the optimizer
      chasing whichever condition contributed the most replicates, and the
      element/group weights express the author's judgement about relative
      importance. Both are legitimate ways to shape a fit.
    * **The joint log-likelihood** (``use_weights=False`` and
      ``group_normalization="sum_over_groups"``). Inference needs the plain sum
      of per-observable NLL terms. Averaging divides every dNLL by the number of
      loss elements, and weights scale it arbitrarily, so a dNLL built from the
      objective cannot be compared with the chi-square threshold of 1.9207 --
      the confidence intervals come out too wide by roughly the square root of
      that factor, and AIC/BIC and the Wald standard errors are shifted with them.

    Use :func:`evaluate_nll_fixed` with ``for_inference=True`` to get the second.
    """
    total = 0.0
    for g_name, g_config in groups.items():
        g_loss_sum = 0.0
        g_weight_sum = 0.0
        for idx, elem in enumerate(g_config.get("loss_elements", [])):
            elem_weight = elem.get("weight", 1.0) if use_weights else 1.0
            lc_fn = elem.get("loss_config")

            is_composite = elem.get("type") == "composite" or "simulations" in elem
            if is_composite:
                sub_sims = elem.get("simulations", [])
                sub_results = [
                    {
                        "results": sim_results[s]["results"],
                        "replicate": replicates[s],
                        "df_dict": sim_results[s]["data"],
                    }
                    for s in sub_sims if s in sim_results
                ]
                data_sim = elem.get("data_simulation") or (sub_sims[0] if sub_sims else None)
                if not sub_results or data_sim not in sim_results:
                    continue
                lc = lc_fn(replicates[data_sim]) if callable(lc_fn) else lc_fn
                key = f"{g_name}_composite_{idx}"
                loss_val = loss_function_composite(
                    p_lin, sub_results, sim_results[data_sim]["data"],
                    key, elem, param_names,
                    loss_config=lc, fixed_sigmas=fixed_sigmas,
                    trace_collector=trace_collector,
                )
            else:
                sim = elem.get("simulation")
                if sim not in sim_results:
                    continue
                lc = lc_fn(replicates[sim]) if callable(lc_fn) else lc_fn
                key = sim
                loss_val = loss_function_evaluated(
                    p_dict, {sim: sim_results[sim]}, param_names,
                    loss_config=lc, fixed_sigmas=fixed_sigmas,
                    trace_collector=trace_collector,
                )

            if loss_components is not None:
                loss_components[key] = loss_val * elem_weight
            g_loss_sum += loss_val * elem_weight
            g_weight_sum += elem_weight

        if g_weight_sum > 0:
            g_loss = (g_loss_sum / g_weight_sum
                      if group_normalization == "mean_over_groups" else g_loss_sum)
        else:
            g_loss = 0.0
        total += g_loss * (g_config.get("group_weight", 1.0) if use_weights else 1.0)
    return total


def evaluate_nll_fixed(
    p, models, replicates, param_names, scales, groups, group_normalization,
    fixed_sigmas, model_text=None, paths=None, events_dynamic=False,
    failure_value=1e10, for_inference=True,
):
    """Joint NLL at *p* (optimizer space) with sigmas frozen at the optimum.

    This is the function every diagnostic consumes -- Wald, slice, profile,
    Sobol, AIC/BIC -- and the one the parallel pool evaluates in workers.

    ``for_inference=True`` (the default) forces the plain sum over loss elements
    with unit weights, i.e. the actual joint log-likelihood, regardless of the
    ``group_normalization`` and weights the *fit* used. Those are objective-
    shaping choices; letting them through to inference divides every dNLL by the
    number of loss elements per group and scales it by the weights, which makes
    a dNLL of 1.9207 mean something other than a 95% interval.

    Pass ``for_inference=False`` to reproduce the objective's own scaling.
    """
    p_lin = _to_linear(p, scales)
    p_dict = dict(zip(param_names, p_lin.tolist()))
    sim_results = simulate_active_replicates(
        p_dict, models, replicates, param_names,
        model_text=model_text, paths=paths, events_dynamic=events_dynamic,
    )
    if sim_results is None:
        return failure_value
    if for_inference:
        group_normalization = "sum_over_groups"
    return accumulate_group_nll(
        sim_results, groups, group_normalization, replicates, param_names,
        p_lin, p_dict, fixed_sigmas=fixed_sigmas,
        use_weights=not for_inference,
    )


# ---------------------------------------------------------------------------
# NLL decomposition (what is dNLL actually made of?)
# ---------------------------------------------------------------------------

def describe_nll_terms(p_vec, models, replicates, param_names, scales, groups,
                       group_normalization, fixed_sigmas):
    """Per-observable breakdown of the joint NLL at *p_vec*.

    Answers the three questions that matter when a dNLL looks wrong:

    * **Which branch fired?** ``sigma_source`` is "fixed" only when the
      (simulation, observable) key was found in ``fixed_sigmas``. Anything else
      means the proper likelihood was skipped and a data-scale heuristic was
      used instead -- with a further division by the point count -- so the
      number is a normalized residual, not a log-likelihood.
    * **What sigma was actually used?** Reported per observable, so an inferred
      sigma can be checked rather than assumed.
    * **Where does the change live?** Call at two parameter vectors and diff the
      ``sum_sq`` / ``contrib`` columns to see which observables move.

    Returns a list of dicts, one per (simulation, observable).
    """
    p_lin = _to_linear(p_vec, scales)
    p_dict = dict(zip(param_names, p_lin.tolist()))
    sim_results = simulate_active_replicates(p_dict, models, replicates, param_names)
    if sim_results is None:
        return []

    rows = []
    for g_name, g_cfg in groups.items():
        for idx, elem in enumerate(g_cfg.get("loss_elements", [])):
            sim = elem.get("simulation")
            if sim is None or sim not in sim_results:
                continue
            lc_fn = elem.get("loss_config")
            lc = lc_fn(replicates[sim]) if callable(lc_fn) else lc_fn
            item = sim_results[sim]
            result, item_df = item["results"], item["data"]
            t_sim = np.asarray(result["time"])

            for obs_cfg in (lc or {}).get("observables", []):
                obs = obs_cfg["observed_variable"]
                d_col, t_col = obs_cfg["data_column"], obs_cfg["time_column"]
                obs_df = _resolve_obs_df(item_df, obs_cfg)
                if obs_df is None or d_col not in obs_df.columns:
                    rows.append({"group": g_name, "sim": sim,
                                 "obs": _short_obs_label(obs), "n": 0,
                                 "sigma_source": "unresolved-data"})
                    continue
                y_data = np.asarray(obs_df[d_col])
                t_data = np.asarray(obs_df[t_col])
                try:
                    # Shared resolver: a diagnostic that silently measured the
                    # data against zeros would report the very error it exists
                    # to detect.
                    y_sim = evaluate_observable(obs, result, p_dict)
                except Exception as exc:
                    rows.append({"group": g_name, "sim": sim,
                                 "obs": _short_obs_label(obs), "n": 0,
                                 "sigma_source": f"unevaluable: {exc}"})
                    continue
                y_pred = np.interp(t_data, t_sim, y_sim)
                valid = np.isfinite(y_pred) & np.isfinite(y_data)
                if not valid.any():
                    continue
                res = y_data[valid] - y_pred[valid]
                n = int(valid.sum())

                key_hit = (fixed_sigmas is not None
                           and (sim, obs) in fixed_sigmas)
                if key_hit:
                    sigma = float(fixed_sigmas[(sim, obs)])
                    source = "fixed"
                    contrib = float(-np.sum(stats.norm.logpdf(
                        y_data[valid], loc=y_pred[valid], scale=sigma)))
                else:
                    sm = obs_cfg.get("sigma_method", "max_mean_std")
                    yv = y_data[valid]
                    if sm == "mean_y_data":
                        sigma = max(float(np.mean(np.abs(yv))), 1e-6)
                    elif sm == "std_y_data":
                        sigma = max(float(np.std(yv)) if n > 1 else abs(float(yv[0])), 1e-6)
                    elif sm == "fixed":
                        sigma = max(float(obs_cfg.get("sigma_value", 1.0)), 1e-6)
                    else:
                        sigma = max(float(np.mean(np.abs(yv))),
                                    float(np.std(yv)) if n > 1 else 0.0, 1e-6)
                    source = f"heuristic:{sm}"
                    contrib = float(0.5 * np.sum((res / sigma) ** 2) / n)

                rows.append({
                    "group": g_name, "sim": sim, "obs": _short_obs_label(obs),
                    "n": n, "sigma": sigma, "sigma_source": source,
                    "sum_sq": float(np.sum(res ** 2)),
                    "mean_sq": float(np.mean(res ** 2)),
                    "rms": float(np.sqrt(np.mean(res ** 2))),
                    "contrib": contrib,
                    "elem_weight": float(elem.get("weight", 1.0)),
                    "group_weight": float(g_cfg.get("group_weight", 1.0)),
                    "n_elems_in_group": len(g_cfg.get("loss_elements", [])),
                })
    return rows


def print_nll_decomposition(rows, title="NLL decomposition"):
    """Human-readable table from :func:`describe_nll_terms`."""
    print("\n" + "=" * 108)
    print(title)
    print("=" * 108)
    hdr = (f"{'group':<12} {'simulation':<22} {'observable':<20} {'n':>4} "
           f"{'sigma':>11} {'source':<20} {'sum r^2':>11} {'contrib':>11}")
    print(hdr)
    print("-" * 108)
    n_fixed = n_other = 0
    for r in rows:
        if r.get("n", 0) == 0:
            print(f"{r['group']:<12} {r['sim']:<22} {r['obs']:<20} "
                  f"{'--':>4} {'--':>11} {r['sigma_source']:<20}")
            n_other += 1
            continue
        if r["sigma_source"] == "fixed":
            n_fixed += 1
        else:
            n_other += 1
        print(f"{r['group']:<12} {r['sim']:<22} {r['obs']:<20} {r['n']:>4} "
              f"{r['sigma']:>11.4g} {r['sigma_source']:<20} "
              f"{r['sum_sq']:>11.4g} {r['contrib']:>11.4g}")
    print("-" * 108)
    if n_other:
        print(f"WARNING: {n_other} observable(s) did NOT use the fixed-sigma "
              f"likelihood. Their contributions are point-averaged normalized "
              f"residuals, not log-likelihood terms, so any dNLL built from them "
              f"cannot be compared against the chi-square threshold of 1.9207.")
    else:
        print(f"All {n_fixed} observable(s) used the fixed-sigma likelihood.")


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

def _finite_difference_steps(params, epsilon=1e-4, abs_floor=None):
    """Per-parameter step for central differences.

    **Relative**, not absolute. The obvious rule -- ``eps * max(|p|, 1.0)``,
    which compute_hessian_manual uses -- degenerates into a fixed 1e-5 step for
    every parameter below 1.0. PBPK rate constants routinely sit at 1e-8 or
    smaller (AGGREGATION_BOUNDS reaches 1e-12), so that step is many orders of
    magnitude larger than the parameter: it pushes the value negative, trips the
    ``p <= 0`` guard, and the stencil comes back as a wall of 1e10 sentinels --
    a garbage Hessian, or a very slow one as safe_simulate fights to integrate
    absurd parameter sets.

    epsilon defaults to 1e-4, near the optimum for second differences
    (round-off ~ macheps/h^2 balanced against truncation ~ h^2).

    Fitting in log space makes this moot -- every parameter is then O(1) -- which
    is one more reason to prefer parameter_scale="log10" for rate constants.
    """
    params = np.atleast_1d(np.asarray(params, dtype=float))
    if abs_floor is None:
        # Only a guard against a literally zero step -- it must never exceed a
        # relative step, or it reintroduces the absolute-step bug it replaced.
        abs_floor = np.finfo(float).tiny
    steps = epsilon * np.abs(params)
    # Parameters sitting exactly at zero have no scale of their own; fall back
    # to the typical magnitude of the rest of the vector.
    nonzero = np.abs(params[np.abs(params) > 0])
    fallback = epsilon * (float(np.median(nonzero)) if nonzero.size else 1.0)
    steps[steps <= 0] = max(fallback, abs_floor)
    return np.maximum(steps, abs_floor)


def compute_hessian_batched(nll_batch, params, epsilon=1e-4):
    """Central-difference Hessian evaluated as one batch.

    The stencil is fixed in advance -- 1 centre, 2k diagonal points and 4 points
    per off-diagonal pair -- so every evaluation can be submitted at once
    instead of trickling through numdifftools one call at a time. That is
    2k^2 + 1 evaluations with no dependencies, which is exactly what the pool
    is for.
    """
    params = np.atleast_1d(np.asarray(params, dtype=float))
    n = params.size
    steps = _finite_difference_steps(params, epsilon)

    points = [params.copy()]           # index 0: centre
    index = {}

    for i in range(n):
        p_plus = params.copy(); p_plus[i] += steps[i]
        p_minus = params.copy(); p_minus[i] -= steps[i]
        index[("d", i)] = (len(points), len(points) + 1)
        points += [p_plus, p_minus]

    for i in range(n):
        for j in range(i + 1, n):
            pp = params.copy(); pp[i] += steps[i]; pp[j] += steps[j]
            pm = params.copy(); pm[i] += steps[i]; pm[j] -= steps[j]
            mp = params.copy(); mp[i] -= steps[i]; mp[j] += steps[j]
            mm = params.copy(); mm[i] -= steps[i]; mm[j] -= steps[j]
            index[("o", i, j)] = tuple(range(len(points), len(points) + 4))
            points += [pp, pm, mp, mm]

    vals = np.asarray(nll_batch(points, label="hessian"), dtype=float)
    f0 = vals[0]

    hessian = np.zeros((n, n))
    for i in range(n):
        a, b = index[("d", i)]
        hessian[i, i] = (vals[a] - 2.0 * f0 + vals[b]) / (steps[i] ** 2)
    for i in range(n):
        for j in range(i + 1, n):
            a, b, c, d = index[("o", i, j)]
            val = (vals[a] - vals[b] - vals[c] + vals[d]) / (4.0 * steps[i] * steps[j])
            hessian[i, j] = hessian[j, i] = val
    return hessian


def compute_hessian_manual(func, params, epsilon=1e-4):
    n = len(params)
    hessian = np.zeros((n, n))
    f0 = func(params)
    steps = _finite_difference_steps(params, epsilon)
    for i in range(n):
        params_plus  = np.array(params, copy=True)
        params_minus = np.array(params, copy=True)
        step = steps[i]
        params_plus[i]  += step
        params_minus[i] -= step
        hessian[i, i] = (func(params_plus) - 2*f0 + func(params_minus)) / (step**2)
    for i in range(n):
        for j in range(i+1, n):
            pp = np.array(params, copy=True); pm = np.array(params, copy=True)
            mp = np.array(params, copy=True); mm = np.array(params, copy=True)
            si = steps[i]
            sj = steps[j]
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


def compute_wald_uncertainty(nll_func, x, bounds=None, loss_scale=1.0, alpha=0.05,
                             nll_batch=None):
    """
    Wald standard errors and confidence intervals from the numerically
    differentiated Hessian of *nll_func* at *x*.

    The Hessian of the negative log-likelihood is the observed Fisher
    information, so the covariance matrix is its inverse.  A non-positive-
    definite Hessian means at least one direction in parameter space is flat or
    confounded; rather than giving up entirely we fall back to a pseudo-inverse,
    which keeps usable SEs for the well-determined parameters and produces huge
    or NaN SEs for the degenerate ones.  That is the informative answer — a
    parameter whose SE cannot be computed is itself the finding.

    Parameters
    ----------
    nll_func : callable(array) -> float
        Proper joint NLL, i.e. ``nll_func_fixed`` with sigmas frozen at the
        optimum.  Passing the optimizer's rescaled objective instead would give
        SEs in the wrong units.
    x : array
        Parameter vector at the optimum, in whatever space *nll_func* expects.
    bounds : sequence of (lo, hi) or None
        Used only to clip the reported CIs; a clipped bound is reported so the
        caller can tell a genuinely tight interval from a truncated one.
    loss_scale : float
        Divides the Fisher information.  Leave at 1.0 when *nll_func* is a true
        NLL; set it when the objective is a known multiple of the NLL.
    alpha : float
        1 - confidence level.  0.05 gives the 95% interval.

    Returns
    -------
    (cov, se, ci)
        ``cov`` is the (k, k) covariance matrix or None; ``se`` is a length-k
        array (entries may be NaN) or None; ``ci`` is always a length-k list of
        (lo, hi) tuples, NaN-filled where the SE is unavailable.
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    k = x.size
    nan_ci = [(float("nan"), float("nan"))] * k

    n_evals = 2 * k * k + 2 * k + 1
    print(f"\n[Wald] Computing Hessian for {k} parameter(s) "
          f"(~{n_evals} silent NLL evaluations)...")

    try:
        if nll_batch is not None:
            hessian_raw = compute_hessian_batched(nll_batch, x)
        else:
            hessian_raw = compute_hessian_numdifftools(nll_func, x)
    except Exception as exc:
        print(f"[Wald] Hessian computation failed: {exc}")
        return None, None, nan_ci

    hessian = np.asarray(hessian_raw, dtype=float)
    if hessian.shape != (k, k):
        print(f"[Wald] Hessian has unexpected shape {hessian.shape}, expected {(k, k)}.")
        return None, None, nan_ci
    if not np.all(np.isfinite(hessian)):
        print("[Wald] Hessian contains non-finite entries — the NLL is probably "
              "hitting a failure sentinel near the optimum. Cannot compute SEs.")
        return None, None, nan_ci

    # Symmetrize: finite differences make H slightly asymmetric.
    fisher = 0.5 * (hessian + hessian.T) / float(loss_scale)

    eigvals, eigvecs = np.linalg.eigh(fisher)
    scale_ref = max(float(np.max(np.abs(eigvals))), 1e-300)
    # Directions whose curvature is negligible relative to the stiffest one are
    # unconstrained by the data: flat or exactly confounded.
    null_mask = eigvals <= 1e-8 * scale_ref
    unconstrained = np.zeros(k, dtype=bool)

    if np.any(null_mask):
        print(f"[Wald] Hessian is NOT positive definite / is rank deficient "
              f"(min eigenvalue {eigvals.min():.4g}, largest {scale_ref:.4g}) — "
              f"{int(null_mask.sum())} of {k} direction(s) are flat or confounded.")
        # A pseudo-inverse would hand back a finite, minimum-norm variance for
        # those directions, which reads as a tight CI for a parameter the data
        # cannot pin down at all. Identify which parameters participate in the
        # null space and report no SE for them instead.
        involvement = np.sqrt(np.sum(eigvecs[:, null_mask] ** 2, axis=1))
        unconstrained = involvement > 0.1
        cov = np.linalg.pinv(fisher)
    else:
        try:
            cov = np.linalg.inv(fisher)
        except np.linalg.LinAlgError as exc:
            print(f"[Wald] Hessian inversion failed ({exc}); using pseudo-inverse.")
            cov = np.linalg.pinv(fisher)

    diag = np.asarray(np.diag(cov), dtype=float)
    se = np.full(k, np.nan)
    negative = diag < 0
    if np.any(negative):
        print(f"[Wald] Negative variance for parameter index/indices "
              f"{np.flatnonzero(negative).tolist()} — no SE for those.")
    if np.any(unconstrained):
        print(f"[Wald] Parameter index/indices {np.flatnonzero(unconstrained).tolist()} "
              f"lie in a flat/confounded direction — reporting no SE rather than a "
              f"pseudo-inverse value that would look deceptively tight.")
    ok = ~negative & ~unconstrained & np.isfinite(diag)
    se[ok] = np.sqrt(diag[ok])

    cv = stats.norm.ppf(1.0 - alpha / 2.0)
    ci = []
    clipped = []
    for i in range(k):
        if not np.isfinite(se[i]):
            ci.append((float("nan"), float("nan")))
            continue
        lo = x[i] - cv * se[i]
        hi = x[i] + cv * se[i]
        if bounds is not None and i < len(bounds) and bounds[i] is not None:
            b_lo, b_hi = bounds[i]
            if b_lo is not None and lo < b_lo:
                lo = b_lo
                clipped.append(i)
            if b_hi is not None and hi > b_hi:
                hi = b_hi
                clipped.append(i)
        ci.append((float(lo), float(hi)))

    if clipped:
        print(f"[Wald] CI clipped at the declared bounds for parameter "
              f"index/indices {sorted(set(clipped))} — the interval is at least "
              f"this wide, so widen the bounds if you need the true extent.")

    return cov, se, ci


def _attach_wald_stats(out, nll_func, x, bounds, param_names=None, scales=None,
                       nll_batch=None):
    """Compute Wald statistics and store them in ``out["stats"]``.

    Shared by all optimization routes so they cannot drift apart again.  When
    *scales* is given the Hessian is taken in the optimizer's (possibly log10)
    space and the results are converted back to linear units for reporting —
    see :func:`_transform_wald_to_linear`.
    """
    try:
        cov, se, ci = compute_wald_uncertainty(nll_func, x, bounds=bounds,
                                               nll_batch=nll_batch)
        corr = compute_parameter_correlations(cov) if cov is not None else None
        if scales is not None:
            se, ci = _transform_wald_to_linear(se, ci, x, scales)
        out["stats"]["wald_cov"] = cov
        out["stats"]["wald_se"] = se
        out["stats"]["wald_ci"] = ci
        if corr is not None:
            out["stats"]["wald_correlation"] = corr
        if se is not None and param_names is not None:
            n_bad = int(np.sum(~np.isfinite(np.asarray(se, dtype=float))))
            if n_bad:
                bad_names = [
                    p for p, s in zip(param_names, np.asarray(se, dtype=float))
                    if not np.isfinite(s)
                ]
                print(f"[Wald] No usable SE for {n_bad} parameter(s): {bad_names}")
    except Exception as e:
        print(f"Error computing Wald statistics: {e}")


# ---------------------------------------------------------------------------
# Parameter scaling (linear / log10)
# ---------------------------------------------------------------------------
#
# PBPK/QSP rate constants routinely span many orders of magnitude, and the
# bounds in Optimizer_settings.py are multiplicative (val/10, val*10).  Fitting
# log10(p) instead of p makes those bounds symmetric, conditions the problem so
# Nelder-Mead and L-BFGS-B both behave, makes a "+/-10%" probe scale-free, and
# removes the p <= 0 cliff entirely because 10**q is positive by construction.
#
# The optimizer, the Hessian, the slice and the profile walkers all work in
# "opt space"; everything reported to the user -- opt["x"], plots, CIs, CSV and
# JSON -- is converted back to linear units first.  Default is "lin" so specs
# that say nothing behave exactly as before.

_VALID_SCALES = ("lin", "log10")
_LN10 = np.log(10.0)

_VALID_FIT_MODES = ("optimize", "evaluate_x0")

# Keys authors may place in optimizer_kwargs that configure the engine rather
# than scipy. They must be stripped before any call to scipy.optimize.
_ENGINE_ONLY_OPTIMIZER_KEYS = (
    "events_depend_on_opt_param",
    "profile_without_opt",      # deprecated, superseded by run_settings["fit_mode"]
    "profile_method",
    "profile_optimizer_kwargs",
)


def _resolve_fit_mode(fit_mode, optimizer_kwargs):
    """Decide whether to run the optimizer or just evaluate the starting point.

    ``fit_mode`` belongs to run_settings -- it is a per-run decision ("today,
    skip the fit and just profile the stored parameters"), not a property of the
    optimization spec, and it is not a scipy argument.  The old
    ``profile_without_opt`` key inside optimizer_kwargs is still honoured, with a
    notice, so existing specs keep working.
    """
    legacy = bool((optimizer_kwargs or {}).get("profile_without_opt", False))
    if fit_mode is None:
        if legacy:
            print("[opt] NOTE: 'profile_without_opt' in optimizer_kwargs is deprecated. "
                  "Set run_settings[\"fit_mode\"] = \"evaluate_x0\", or pass --no-fit, "
                  "so the choice lives with the run rather than the spec.")
            return "evaluate_x0"
        return "optimize"
    if fit_mode not in _VALID_FIT_MODES:
        raise ValueError(
            f"Unknown fit_mode {fit_mode!r}; expected one of {_VALID_FIT_MODES}"
        )
    if legacy and fit_mode == "optimize":
        print("[opt] NOTE: fit_mode='optimize' from run_settings overrides the "
              "deprecated profile_without_opt=True in this spec.")
    return fit_mode


def _resolve_profile_optimizer(method, optimizer_kwargs):
    """Method and kwargs for the profile-likelihood nuisance re-optimization.

    Defaults to the spec's own ``method`` rather than a hardcoded L-BFGS-B: a
    gradient method reads the 1e10 failure sentinel as a cliff and stalls at the
    start point, which looks exactly like an unidentifiable parameter.  Override
    per spec with ``profile_method`` / ``profile_optimizer_kwargs`` inside
    optimizer_kwargs.
    """
    kw = optimizer_kwargs or {}
    profile_method = kw.get("profile_method") or method or "Nelder-Mead"
    profile_kwargs = kw.get("profile_optimizer_kwargs") or {}
    return profile_method, profile_kwargs


def _resolve_scales(parameter_scale, param_names):
    """Normalize a ``parameter_scale`` spec to a list of per-parameter scales.

    Accepts None (all linear), a single string applied to every parameter, a
    {name: scale} dict (unlisted names default to "lin"), or an explicit
    per-parameter sequence.
    """
    k = len(param_names)
    if parameter_scale is None:
        return ["lin"] * k
    if isinstance(parameter_scale, str):
        scales = [parameter_scale] * k
    elif isinstance(parameter_scale, dict):
        unknown = set(parameter_scale) - set(param_names)
        if unknown:
            raise ValueError(
                f"parameter_scale names not in param_names: {sorted(unknown)}"
            )
        scales = [parameter_scale.get(name, "lin") for name in param_names]
    else:
        scales = list(parameter_scale)
        if len(scales) != k:
            raise ValueError(
                f"parameter_scale has {len(scales)} entries but there are "
                f"{k} parameters"
            )
    bad = [s for s in scales if s not in _VALID_SCALES]
    if bad:
        raise ValueError(
            f"Unknown parameter scale(s) {bad}; valid options are {_VALID_SCALES}"
        )
    return scales


def _any_log(scales):
    return any(s == "log10" for s in scales)


def _to_opt_space(x_lin, scales):
    """Linear parameter values -> optimizer space."""
    x_lin = np.atleast_1d(np.asarray(x_lin, dtype=float))
    out = np.array(x_lin, dtype=float, copy=True)
    for i, s in enumerate(scales):
        if s == "log10":
            if x_lin[i] <= 0:
                raise ValueError(
                    f"Parameter index {i} has value {x_lin[i]!r}, which cannot be "
                    f"fitted on a log10 scale. Use scale 'lin' for parameters "
                    f"that can reach zero or go negative."
                )
            out[i] = np.log10(x_lin[i])
    return out


def _to_linear(x_opt, scales):
    """Optimizer space -> linear parameter values."""
    x_opt = np.atleast_1d(np.asarray(x_opt, dtype=float))
    out = np.array(x_opt, dtype=float, copy=True)
    for i, s in enumerate(scales):
        if s == "log10":
            out[i] = 10.0 ** x_opt[i]
    return out


def _bounds_to_opt_space(bounds, scales):
    """Transform a scipy-style bounds list into optimizer space."""
    if bounds is None:
        return None
    out = []
    for i, b in enumerate(bounds):
        if b is None:
            out.append(None)
            continue
        lo, hi = b
        if scales[i] == "log10":
            if lo is not None and lo <= 0:
                raise ValueError(
                    f"Parameter index {i} has lower bound {lo!r}, which is invalid "
                    f"on a log10 scale. Raise the bound above zero or use 'lin'."
                )
            lo = None if lo is None else np.log10(lo)
            hi = None if hi is None else np.log10(hi)
        out.append((lo, hi))
    return out


def _transform_wald_to_linear(se_opt, ci_opt, x_opt, scales):
    """Convert Wald SEs and CIs from optimizer space to linear units.

    SEs use the delta method: for q = log10(p), dp/dq = p * ln(10), so
    se_p = se_q * p * ln(10).  CI endpoints are transformed directly (10**q),
    which is exact rather than a local approximation and keeps the interval
    positive and asymmetric as it should be on a log scale.
    """
    if not _any_log(scales):
        return se_opt, ci_opt

    x_lin = _to_linear(x_opt, scales)

    se_lin = None
    if se_opt is not None:
        se_lin = np.array(np.asarray(se_opt, dtype=float), copy=True)
        for i, s in enumerate(scales):
            if s == "log10" and np.isfinite(se_lin[i]):
                se_lin[i] = se_lin[i] * x_lin[i] * _LN10

    ci_lin = None
    if ci_opt is not None:
        ci_lin = []
        for i, (lo, hi) in enumerate(ci_opt):
            if scales[i] == "log10":
                lo = 10.0 ** lo if np.isfinite(lo) else lo
                hi = 10.0 ** hi if np.isfinite(hi) else hi
            ci_lin.append((float(lo), float(hi)))

    return se_lin, ci_lin


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
    # Engine-level keys live in the same dict for author convenience but are not
    # scipy arguments, so strip them all in one place.
    for _engine_key in _ENGINE_ONLY_OPTIMIZER_KEYS:
        kwargs.pop(_engine_key, None)
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

_PROFILE_THRESHOLD = 1.9207  # chi2(df=1, p=0.95) / 2


def _profile_nuisance_defaults(method):
    """Convergence options appropriate to *method*.

    scipy rejects unknown option keys per method, so "ftol" cannot simply be
    handed to Nelder-Mead.  Gradient-free methods also need a larger iteration
    budget to reach comparable accuracy.
    """
    m = (method or "").lower()
    if m in ("l-bfgs-b", "tnc", "slsqp"):
        return {"maxiter": 50, "ftol": 1e-4}
    if m == "nelder-mead":
        return {"maxiter": 200, "fatol": 1e-4, "xatol": 1e-4}
    if m == "powell":
        return {"maxiter": 200, "ftol": 1e-4, "xtol": 1e-4}
    return {"maxiter": 100}


def _minimize_nuisance(fun, x0, args, method, bounds, optimizer_kwargs=None):
    """Minimize over the nuisance parameters with the caller's chosen method.

    Falls back to Nelder-Mead if *method* cannot handle the problem (for
    example a gradient method that scipy refuses for these bounds), so a
    profile run degrades rather than dying.
    """
    import scipy.optimize as opt

    kwargs = dict(optimizer_kwargs or {})
    options = dict(_profile_nuisance_defaults(method))
    options.update(kwargs.pop("options", {}) or {})
    for key in _ENGINE_ONLY_OPTIMIZER_KEYS:
        kwargs.pop(key, None)

    try:
        return opt.minimize(fun, x0, args=args, method=method,
                            bounds=bounds, options=options, **kwargs)
    except (ValueError, TypeError) as exc:
        print(f"    [profile] method {method!r} failed ({exc}); "
              f"retrying with Nelder-Mead.", flush=True)
        return opt.minimize(fun, x0, args=args, method="Nelder-Mead",
                            bounds=bounds,
                            options=_profile_nuisance_defaults("Nelder-Mead"))


def _try_build_evaluator(model_text, paths, models, active_replicates, param_names,
                         scales, optimization_spec, fixed_sigmas, events_dynamic,
                         n_workers):
    """Build a ParallelEvaluator, or return None with the reason printed.

    Every failure mode here is recoverable by running serially, so this never
    raises -- a diagnostics run that cannot parallelize should still produce
    numbers, just more slowly.
    """
    try:
        from Engine.Evaluator import (
            build_eval_spec, check_spec_serializable, ParallelEvaluator,
            default_worker_count,
        )
    except Exception as exc:
        print(f"[pool] Evaluator unavailable ({exc}); evaluating serially.")
        return None

    n = default_worker_count(n_workers)
    if n <= 1:
        print("[pool] one worker requested; evaluating serially.")
        return None

    try:
        spec = build_eval_spec(
            model_text=model_text,
            paths=paths,
            events={name: models[name].get("events", "") for name in active_replicates},
            replicates=active_replicates,
            param_names=param_names,
            scales=scales,
            groups=optimization_spec.groups,
            group_normalization=optimization_spec.group_normalization,
            fixed_sigmas=fixed_sigmas,
            events_dynamic=events_dynamic,
        )
    except Exception as exc:
        print(f"[pool] could not build an eval spec ({exc}); evaluating serially.")
        return None

    ok, _size, _msg = check_spec_serializable(spec)
    if not ok:
        return None

    try:
        return ParallelEvaluator(spec, n_workers=n).start()
    except Exception as exc:
        print(f"[pool] failed to start workers ({exc}); evaluating serially.")
        return None


def _slice_grid(param_idx, res_x, param_names, n_points, range_factor, scales):
    """Parameter values to sample for one slice, in optimizer space."""
    is_log = scales[param_idx] == "log10"
    p_val = res_x[param_idx]
    if is_log:
        offset = np.log10(range_factor)
        return np.linspace(p_val - offset, p_val + offset, n_points), is_log
    return np.linspace(p_val / range_factor, p_val * range_factor, n_points), is_log


def _run_likelihood_slice_all(
    nll_batch, res_x, nll_at_optimum, param_names,
    n_points=20, range_factor=2.0, scales=None
):
    """Every parameter's slice as a single batch.

    A slice has no dependencies between points, so all k x n_points evaluations
    can go out at once -- this is the cheapest possible use of the pool and the
    reason slice analysis was worth parallelizing first.
    Returns {param_name: (param_vals_linear, dnll)}.
    """
    scales = scales if scales is not None else ["lin"] * len(param_names)
    grids, is_logs, xs = [], [], []
    for i in range(len(param_names)):
        grid, is_log = _slice_grid(i, res_x, param_names, n_points, range_factor, scales)
        grids.append(grid)
        is_logs.append(is_log)
        mask = np.arange(len(res_x)) == i
        xs.extend(np.where(mask, v, res_x) for v in grid)

    print(f"\n[slice] {len(param_names)} parameter(s) x {n_points} points "
          f"= {len(xs)} evaluations, submitted as one batch")
    vals = nll_batch(xs, label="slice")

    out = {}
    pos = 0
    for i, name in enumerate(param_names):
        chunk = np.asarray(vals[pos:pos + n_points], dtype=float)
        pos += n_points
        grid = 10.0 ** grids[i] if is_logs[i] else grids[i]
        out[name] = (grid, chunk - nll_at_optimum)
    return out


def _run_likelihood_slice_single(
    param_idx, nll_func, res_x, nll_at_optimum, param_names,
    n_points=20, range_factor=2.0, scales=None, nll_batch=None
):
    """Likelihood slice for one parameter: vary it, hold the others fixed.

    No nuisance re-optimization, so this is a cross-section rather than a
    profile -- cheap, and a useful sanity check that the NLL responds to the
    parameter at all.  Returns parameter values in linear units.
    """
    scales = scales if scales is not None else ["lin"] * len(param_names)
    pname = param_names[param_idx]
    param_vals, is_log = _slice_grid(
        param_idx, res_x, param_names, n_points, range_factor, scales
    )
    idx_mask = np.arange(len(res_x)) == param_idx
    xs = [np.where(idx_mask, v, res_x) for v in param_vals]

    if nll_batch is not None:
        print(f"\n[slice] {pname}  ({n_points} points, x{range_factor} range)")
        nll_vals = nll_batch(xs, label=f"slice:{pname}")
    else:
        width = len(str(n_points))
        print(f"\n[slice] {pname}  ({n_points} points, x{range_factor} range)")
        nll_vals = []
        for i, (val, x) in enumerate(zip(param_vals, xs)):
            nll = nll_func(x)
            shown = 10.0 ** val if is_log else val
            print(f"  [{i+1:{width}d}/{n_points}]  {pname}={shown:.4g}  nll={nll:.6g}")
            nll_vals.append(nll)

    if is_log:
        param_vals = 10.0 ** param_vals

    return param_vals, np.array(nll_vals, dtype=float) - nll_at_optimum


def _make_nuisance_objective(nll_func, param_idx, n_params):
    """Build f(x_nuisance, fixed_val) -> NLL with parameter *param_idx* pinned."""
    def nuisance_objective(x_nuisance, fixed_val):
        x_full = np.empty(n_params)
        idx = 0
        for i in range(n_params):
            if i == param_idx:
                x_full[i] = fixed_val
            else:
                x_full[i] = x_nuisance[idx]
                idx += 1
        return nll_func(x_full)
    return nuisance_objective

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
    n_points=20, range_factor=2.0, fallback_func=None, wald_se_val=None,
    method="L-BFGS-B", optimizer_kwargs=None, scales=None
):
    """Run an adaptive true profile likelihood for one parameter.

    Works in whatever space *nll_func* expects (opt space, which may be log10)
    but returns parameter values in **linear** units so callers can plot and
    interpolate CIs consistently.  The nuisance re-optimization uses *method*,
    defaulting to the spec's own optimizer rather than a hardcoded L-BFGS-B.
    """
    pname = param_names[param_idx]
    p_opt = res_x[param_idx]
    scales = scales if scales is not None else ["lin"] * len(param_names)
    is_log = scales[param_idx] == "log10"

    if bounds is not None and bounds[param_idx] is not None:
        lb_bound, ub_bound = bounds[param_idx]
        lb_bound = -np.inf if lb_bound is None else lb_bound
        ub_bound = np.inf if ub_bound is None else ub_bound
    else:
        lb_bound = -np.inf
        ub_bound = np.inf

    # range_factor is multiplicative in linear space, which is an additive
    # offset of log10(range_factor) once the parameter is fitted in log space.
    if is_log:
        offset = np.log10(range_factor)
        lb_target, ub_target = p_opt - offset, p_opt + offset
    else:
        lb_target = min(p_opt / range_factor, p_opt * range_factor)
        ub_target = max(p_opt / range_factor, p_opt * range_factor)

    lb = max(lb_target, lb_bound)
    ub = min(ub_target, ub_bound)

    def _lin(v):
        return 10.0 ** v if is_log else v

    print(f"\n[true profile] {pname}  (adaptive stepping between {_lin(lb):.4g} and "
          f"{_lin(ub):.4g}, re-optimizing nuisance params with {method})", flush=True)

    nuisance_objective = _make_nuisance_objective(nll_func, param_idx, len(param_names))

    if bounds is not None:
        nuisance_bounds = bounds[:param_idx] + bounds[param_idx+1:]
    else:
        nuisance_bounds = None

    def walk_profile(direction_sign, bound):
        if (direction_sign == 1 and bound <= p_opt) or (direction_sign == -1 and bound >= p_opt):
            return [], []
            
        evaluated = [(p_opt, 0.0, np.delete(res_x, param_idx))]
        
        def evaluate_pt(x_target, x_nuisance_guess):
            tag = 'left ' if direction_sign == -1 else 'right'
            if len(x_nuisance_guess) == 0:
                # No nuisance parameters to re-optimize (single-parameter fit):
                # the profile value at x_target is just the objective itself.
                # scipy.optimize.minimize errors on a length-0 x0, so skip it.
                nll_rel = nuisance_objective(x_nuisance_guess, x_target) - nll_at_optimum
                print(f"  [{tag}]  {pname}={_lin(x_target):.4g}  dNLL={nll_rel:.6g}", flush=True)
                return nll_rel, x_nuisance_guess
            res = _minimize_nuisance(
                nuisance_objective, x_nuisance_guess, (x_target,),
                method, nuisance_bounds, optimizer_kwargs,
            )
            nll_rel = res.fun - nll_at_optimum
            print(f"  [{tag}]  {pname}={_lin(x_target):.4g}  dNLL={nll_rel:.6g}", flush=True)
            return nll_rel, res.x
            
        coarse_steps = max(3, n_points // 4)
        coarse_xs = np.linspace(p_opt, bound, coarse_steps + 1)[1:]
        
        crossed = False
        for x_target in coarse_xs:
            last_x, last_nll, last_nuisance = evaluated[-1]
            nll_rel, x_nuisance = evaluate_pt(x_target, last_nuisance)
            evaluated.append((x_target, nll_rel, x_nuisance))
            
            if nll_rel > _PROFILE_THRESHOLD:
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
                
                if nll_mid > _PROFILE_THRESHOLD:
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
                    score += min(gap_nll, 5.0) / _PROFILE_THRESHOLD
                    
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

    if is_log:
        all_vals = 10.0 ** all_vals

    return all_vals, all_nlls


def _profile_grid_for(param_idx, res_x, bounds, scales, wald_se, n_grid,
                      range_factor, se_span):
    """Grid of fixed values for one parameter, both directions, in opt space.

    Seeded from the Wald standard error when one is available: the profile
    crossing sits near 1.96 SE, so spanning a few SE puts most points where the
    threshold actually is instead of spreading them over a range_factor window
    that may be far too wide or far too narrow. Falls back to the multiplicative
    range_factor when no SE exists (a flat or confounded direction).
    """
    p_opt = res_x[param_idx]
    is_log = scales[param_idx] == "log10"

    if bounds is not None and param_idx < len(bounds) and bounds[param_idx] is not None:
        lb, ub = bounds[param_idx]
        lb = -np.inf if lb is None else lb
        ub = np.inf if ub is None else ub
    else:
        lb, ub = -np.inf, np.inf

    se = None
    if wald_se is not None:
        try:
            cand = float(np.atleast_1d(wald_se)[param_idx])
            if np.isfinite(cand) and cand > 0:
                se = cand
        except (IndexError, TypeError, ValueError):
            se = None

    if se is not None:
        half = se_span * se
        lo_target, hi_target = p_opt - half, p_opt + half
    elif is_log:
        off = np.log10(range_factor)
        lo_target, hi_target = p_opt - off, p_opt + off
    else:
        lo_target = min(p_opt / range_factor, p_opt * range_factor)
        hi_target = max(p_opt / range_factor, p_opt * range_factor)

    lo = max(lo_target, lb)
    hi = min(hi_target, ub)

    left = [v for v in np.linspace(p_opt, lo, n_grid + 1)[1:] if v < p_opt]
    right = [v for v in np.linspace(p_opt, hi, n_grid + 1)[1:] if v > p_opt]
    return left, right, (lb, ub), is_log


def run_parallel_profile(
    nll_batch_profile, res_x, nll_at_optimum, param_names, bounds, scales,
    method="Nelder-Mead", optimizer_kwargs=None, wald_se=None,
    n_grid=5, range_factor=2.0, se_span=4.0, n_refine=2,
    checkpoint=None, threshold=_PROFILE_THRESHOLD,
):
    """Profile likelihood for every parameter as parallel batches.

    The sequential adaptive walker cannot be parallelized: each point warm-starts
    from the previous one. This trades that dependency for width -- a
    predetermined grid whose points are all independent, so 2k x n_grid nuisance
    optimizations go out at once (144 for a 12-parameter fit at n_grid=6).

    Two passes:
      1. coarse grid, cold-started from the optimum, fully parallel;
      2. refinement only in the bracket that straddles the threshold, probing
         the predicted crossing via sqrt(dNLL) interpolation (which is linear in
         distance for a locally quadratic NLL) rather than bisecting blindly.

    Cold-starting pass 1 costs more iterations per point than warm-starting
    would, and that is the deliberate trade: on 40 cores, width beats per-point
    efficiency.

    Returns {param_name: (param_vals_linear, dnll)}, matching the sequential
    walkers so plotting and CI extraction are unchanged.
    """
    n_params = len(param_names)
    completed = checkpoint.load(param_names) if checkpoint is not None else \
        {n: {} for n in param_names}

    def _lin(v, is_log):
        return 10.0 ** v if is_log else v

    def make_job(i, x_fixed, is_log, nb, phase=1, direction=0):
        return {
            "param_idx": i,
            "param_name": param_names[i],
            "x_fixed": float(x_fixed),
            "x_fixed_linear": float(_lin(x_fixed, is_log)),
            "x_start": np.delete(res_x, i).tolist(),
            "nuisance_bounds": nb,
            "method": method,
            "optimizer_kwargs": optimizer_kwargs,
            # phase/direction are recorded so a resume can tell how much
            # refinement a side already has and stop, instead of adding another
            # probe on every launch.
            "phase": phase,
            "direction": direction,
        }

    def nuisance_bounds_for(i):
        if bounds is None:
            return None
        return [list(b) if b is not None else None
                for b in (list(bounds[:i]) + list(bounds[i + 1:]))]

    # ── Pass 1: coarse grid ───────────────────────────────────────────────
    jobs = []
    meta = {}
    for i, name in enumerate(param_names):
        left, right, _b, is_log = _profile_grid_for(
            i, res_x, bounds, scales, wald_se, n_grid, range_factor, se_span
        )
        meta[i] = {"is_log": is_log}
        nb = nuisance_bounds_for(i)
        for v in left + right:
            if ProfileCheckpointKey(v) in completed.get(name, {}):
                continue
            jobs.append(make_job(i, v, is_log, nb))

    n_cached = sum(len(v) for v in completed.values())
    if n_cached:
        print(f"\n[profile] resuming: {n_cached} point(s) already computed, "
              f"{len(jobs)} to run")
    print(f"\n[profile] pass 1: {len(jobs)} independent nuisance optimizations "
          f"across {n_params} parameter(s)")

    def record(res):
        res["dnll"] = (float(res["nll"]) - nll_at_optimum
                       if res.get("nll") is not None else float("nan"))
        if checkpoint is not None and res.get("status") == "ok":
            checkpoint.append(res)
        name = res["param_name"]
        completed.setdefault(name, {})[ProfileCheckpointKey(res["x_fixed"])] = res

    if jobs:
        nll_batch_profile(jobs, on_result=record, label="profile-pass1")

    # ── Pass 2: refine the threshold crossing ─────────────────────────────
    # n_refine rounds, each a single wide batch across every parameter and
    # direction that still needs work. Rounds are sequential because each probe
    # depends on the previous bracket, but every round is still ~2k wide, so the
    # pool stays busy. Doing all rounds here (rather than one per launch) means
    # a completed run resumes as a genuine no-op.
    for _round in range(max(int(n_refine), 0)):
        refine_jobs = _build_refinement_jobs(
            param_names, completed, res_x, meta, nuisance_bounds_for, make_job,
            threshold, n_refine,
        )
        if not refine_jobs:
            break
        print(f"\n[profile] pass 2 (round {_round + 1}): refining "
              f"{len(refine_jobs)} threshold crossing(s)")
        nll_batch_profile(refine_jobs, on_result=record,
                          label=f"profile-refine{_round + 1}")

    return _assemble_profile_traces(param_names, completed, res_x, meta)


def _build_refinement_jobs(param_names, completed, res_x, meta,
                           nuisance_bounds_for, make_job, threshold, n_refine):
    """Probes at the predicted threshold crossing for each parameter/direction."""
    jobs = []
    for i, name in enumerate(param_names):
        pts = sorted(completed.get(name, {}).values(), key=lambda r: r["x_fixed"])
        if not pts:
            continue
        is_log = meta[i]["is_log"]
        nb = nuisance_bounds_for(i)
        p_opt = res_x[i]
        for sign in (-1, +1):
            side = [r for r in pts
                    if (r["x_fixed"] < p_opt if sign < 0 else r["x_fixed"] > p_opt)]
            # n_refine caps the total refinement per side across all launches,
            # so resuming a finished run is a no-op rather than adding a probe
            # every time.
            n_done = sum(1 for r in side
                         if int(r.get("phase", 1)) == 2
                         and int(r.get("direction", 0)) == sign)
            if n_done >= n_refine:
                continue
            side.sort(key=lambda r: abs(r["x_fixed"] - p_opt))
            inner_x, inner_d = p_opt, 0.0
            for r in side:
                d = r.get("dnll")
                if d is None or not np.isfinite(d):
                    continue
                if d > threshold:
                    # Bracket found: probe the predicted crossing. One probe per
                    # launch keeps pass 2 a single wide batch -- iterating here
                    # would serialize what we just made parallel.
                    r_in, r_out = np.sqrt(max(inner_d, 0.0)), np.sqrt(d)
                    x_in, x_out = inner_x, r["x_fixed"]
                    if r_out > r_in + 1e-12:
                        frac = (np.sqrt(threshold) - r_in) / (r_out - r_in)
                        frac = min(max(frac, 0.05), 0.95)
                    else:
                        frac = 0.5
                    x_probe = x_in + frac * (x_out - x_in)
                    if ProfileCheckpointKey(x_probe) not in completed.get(name, {}):
                        jobs.append(
                            make_job(i, x_probe, is_log, nb, phase=2, direction=sign)
                        )
                    break
                inner_x, inner_d = r["x_fixed"], d
    return jobs


def _assemble_profile_traces(param_names, completed, res_x, meta):
    """Turn completed points into {name: (param_vals_linear, dnll)} traces."""
    out = {}
    for i, name in enumerate(param_names):
        is_log = meta.get(i, {}).get("is_log", False)
        pts = [r for r in completed.get(name, {}).values()
               if r.get("dnll") is not None and np.isfinite(r["dnll"])]
        xs = [res_x[i]] + [r["x_fixed"] for r in pts]
        ys = [0.0] + [float(r["dnll"]) for r in pts]
        order = np.argsort(xs)
        xs = np.asarray(xs, dtype=float)[order]
        ys = np.asarray(ys, dtype=float)[order]
        if is_log:
            xs = 10.0 ** xs
        out[name] = (xs, ys)
    return out


def ProfileCheckpointKey(x):
    """Grid-point identity, rounded so float noise cannot duplicate a point."""
    try:
        return round(float(x), 12)
    except (TypeError, ValueError):
        return None


def _run_parallel_profile_with_checkpoint(
    evaluator, res_x, nll_at_optimum, param_names, bounds, scales, groups,
    model_text, paths, method, optimizer_kwargs, wald_se,
    n_grid, range_factor, se_span, n_refine, run_id, checkpoint_enabled=True,
    fixed_sigmas=None,
):
    """Wire the pool, the checkpoint store and run_parallel_profile together."""
    from Engine.Profile_checkpoint import (
        ProfileCheckpoint, spec_fingerprint, default_run_id,
    )

    model_hash, spec_hash = spec_fingerprint(
        param_names, res_x, groups, scales, model_text, fixed_sigmas=fixed_sigmas
    )
    # run_id must be stable across launches or resuming can never happen.
    run_id = default_run_id(run_id, model_hash, spec_hash)
    root = paths.get("plot_path") if checkpoint_enabled else None
    ckpt = ProfileCheckpoint(root, run_id, model_hash, spec_hash,
                             enabled=bool(root))
    if ckpt.dir:
        print(f"\n[profile] checkpointing to {ckpt.dir}")

    try:
        traces = run_parallel_profile(
            evaluator.profile_batch, res_x, nll_at_optimum, param_names,
            bounds, scales, method=method, optimizer_kwargs=optimizer_kwargs,
            wald_se=wald_se, n_grid=n_grid, range_factor=range_factor,
            se_span=se_span, n_refine=n_refine, checkpoint=ckpt,
        )
        if ckpt.n_skipped_stale:
            print(f"[profile] ignored {ckpt.n_skipped_stale} checkpoint record(s) "
                  f"from a different model or spec")
        return traces
    finally:
        ckpt.close()


def _run_fast_profile_likelihood_single(
    param_idx, nll_func, bounds, res_x, nll_at_optimum, param_names,
    variation_pct=0.10, method="L-BFGS-B", optimizer_kwargs=None, scales=None,
    max_points_per_side=6, growth=2.0, n_bisections=2, stats_out=None
):
    """
    Fast profile likelihood for one parameter: step outward from the optimum,
    growing the step geometrically until dNLL brackets the 95% threshold.

    The previous implementation probed a fixed -10%/+10% and could therefore
    never cross dNLL = 1.92, which made the extracted CI (nan, nan) by
    construction.  Growing the step means a genuine CI comes out of typically
    4-6 nuisance optimizations per side rather than the ~13 the full adaptive
    walker uses -- which is what makes this the cheap baseline it was meant to be.

    *variation_pct* sets the first step only.  If *stats_out* is a dict, a
    curvature-based standard error estimated from the innermost points is stored
    under "curvature_se" -- useful as a Wald cross-check and as a step-size seed.

    Returns (param_vals, dnll_vals) with parameter values in linear units.
    """
    pname = param_names[param_idx]
    p_opt = res_x[param_idx]
    scales = scales if scales is not None else ["lin"] * len(param_names)
    is_log = scales[param_idx] == "log10"

    if bounds is not None and bounds[param_idx] is not None:
        lb_bound, ub_bound = bounds[param_idx]
        lb_bound = -np.inf if lb_bound is None else lb_bound
        ub_bound = np.inf if ub_bound is None else ub_bound
    else:
        lb_bound = -np.inf
        ub_bound = np.inf

    def _lin(v):
        return 10.0 ** v if is_log else v

    # First step: variation_pct is multiplicative in linear space, so in log
    # space it becomes the additive offset log10(1 + variation_pct).
    if is_log:
        step0 = np.log10(1.0 + variation_pct)
    else:
        step0 = abs(p_opt) * variation_pct
        if step0 <= 0:
            step0 = max(variation_pct, 1e-8)

    pct_str = f"{int(round(variation_pct * 100))}%"
    print(f"\n[fast profile] {pname} (adaptive: first step {pct_str}, growing x{growth} "
          f"until dNLL > {_PROFILE_THRESHOLD:.3g}, nuisance method {method})", flush=True)

    nuisance_objective = _make_nuisance_objective(nll_func, param_idx, len(param_names))

    if bounds is not None:
        nuisance_bounds = bounds[:param_idx] + bounds[param_idx+1:]
    else:
        nuisance_bounds = None

    x_nuisance_initial = np.delete(res_x, param_idx)

    def evaluate(x_target):
        if len(x_nuisance_initial) == 0:
            return nuisance_objective(x_nuisance_initial, x_target) - nll_at_optimum
        res = _minimize_nuisance(
            nuisance_objective, x_nuisance_initial, (x_target,),
            method, nuisance_bounds, optimizer_kwargs,
        )
        return res.fun - nll_at_optimum

    def walk(direction_sign, bound):
        vals, dnlls = [], []
        step = step0
        tag = "left " if direction_sign < 0 else "right"
        crossed = False
        for _ in range(max_points_per_side):
            x_target = p_opt + direction_sign * step
            hit_bound = False
            if direction_sign < 0 and x_target <= bound:
                x_target, hit_bound = bound, True
            elif direction_sign > 0 and x_target >= bound:
                x_target, hit_bound = bound, True
            if not np.isfinite(x_target):
                break

            dnll = evaluate(x_target)
            print(f"  [{tag}]  {pname}={_lin(x_target):.4g}  dNLL={dnll:.6g}", flush=True)
            vals.append(x_target)
            dnlls.append(dnll)

            if dnll > _PROFILE_THRESHOLD:
                crossed = True
                break
            if hit_bound:
                break
            step *= growth

        # Refine the crossing. Without this the CI is interpolated linearly
        # across one geometric step, which overstates the curvature near the
        # optimum and yields an interval that is too narrow -- an under-covering
        # CI is worse than an obviously missing one.
        #
        # Probe at the predicted crossing rather than the midpoint: for a locally
        # quadratic NLL, sqrt(dNLL) is linear in the distance from the optimum,
        # so interpolating there lands near the threshold in one or two steps
        # even when the first geometric step overshot badly (dNLL in the hundreds).
        if crossed and n_bisections > 0:
            x_in = vals[-2] if len(vals) >= 2 else p_opt
            d_in = dnlls[-2] if len(dnlls) >= 2 else 0.0
            x_out, d_out = vals[-1], dnlls[-1]
            root_target = np.sqrt(_PROFILE_THRESHOLD)
            for _ in range(n_bisections):
                r_in = np.sqrt(max(d_in, 0.0))
                r_out = np.sqrt(max(d_out, 0.0))
                if r_out > r_in + 1e-12:
                    frac = (root_target - r_in) / (r_out - r_in)
                    frac = min(max(frac, 0.05), 0.95)  # stay strictly inside
                else:
                    frac = 0.5
                x_probe = x_in + frac * (x_out - x_in)
                d_probe = evaluate(x_probe)
                print(f"  [{tag}*] {pname}={_lin(x_probe):.4g}  dNLL={d_probe:.6g}", flush=True)
                vals.append(x_probe)
                dnlls.append(d_probe)
                if d_probe > _PROFILE_THRESHOLD:
                    x_out, d_out = x_probe, d_probe
                else:
                    x_in, d_in = x_probe, d_probe

            # Keep the returned trace monotonic in x so _extract_profile_ci can
            # walk outward from the optimum correctly.
            order = np.argsort([direction_sign * v for v in vals])
            vals = [vals[i] for i in order]
            dnlls = [dnlls[i] for i in order]

        return vals, dnlls

    left_vals, left_dnlls = walk(-1, lb_bound)
    right_vals, right_dnlls = walk(+1, ub_bound)

    param_vals = np.array(left_vals[::-1] + [p_opt] + right_vals)
    dnll_vals = np.array(left_dnlls[::-1] + [0.0] + right_dnlls)

    # Curvature SE from the innermost point either side: dNLL ~ (dp)^2 / (2*se^2)
    # so se = dp / sqrt(2*dNLL).  Reported in linear units.
    if stats_out is not None:
        se_est = float("nan")
        candidates = []
        if left_vals and left_dnlls[0] > 0:
            candidates.append((abs(p_opt - left_vals[0]), left_dnlls[0]))
        if right_vals and right_dnlls[0] > 0:
            candidates.append((abs(right_vals[0] - p_opt), right_dnlls[0]))
        if candidates:
            ses = [dp / np.sqrt(2.0 * d) for dp, d in candidates]
            se_est = float(np.mean(ses))
            if is_log:
                # delta method back to linear units
                se_est = se_est * _lin(p_opt) * _LN10
        # Keyed by parameter: this helper runs once per parameter, so a scalar
        # would just be overwritten by the last one.
        stats_out.setdefault("curvature_se", {})[pname] = se_est
        print(f"  curvature SE estimate for {pname}: {se_est:.4g}", flush=True)

    if is_log:
        param_vals = 10.0 ** param_vals

    return param_vals, dnll_vals


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
    fast_profile_likelihood_analysis=False,
    sobol_analysis=False,
    sobol_kwargs=None,
    method="Nelder-Mead",
    optimizer_kwargs=None,
    fast=False,
    maxiter=None,
    tol=None,
    fit_mode=None,
    n_workers=None,
):
    if (profile_likelihood_analysis or fast_profile_likelihood_analysis) and not wald_analysis:
        print("Note: profile likelihood requested — also enabling wald_analysis so the "
              "Hessian-based CI is available as an independent cross-check.")
        wald_analysis = True

    from datetime import datetime
    _progress_overlay_state["timestamp"] = datetime.now().strftime("%Y%m%d_%H%M%S")

    try:
        from scipy.optimize import minimize
    except ImportError:
        raise ImportError("scipy is required for run_optimization") from None

    data_path = paths["data_path"]

    # This route takes a flat settings dict with no parameter_scale field, so it
    # is linear-only. Log-space fitting is available via the nested Optimization
    # spec (run_optimization_from_groups).
    scales = ["lin"] * len(param_names)
    fit_mode = _resolve_fit_mode(fit_mode, optimizer_kwargs)
    profile_method, profile_opt_kwargs = _resolve_profile_optimizer(method, optimizer_kwargs)

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
    
    if fit_mode == "evaluate_x0":
        from scipy.optimize import OptimizeResult
        print("[opt] fit_mode='evaluate_x0' — skipping the fit and evaluating x0 "
              "so diagnostics run against the supplied parameters.")
        x0_arr = np.array(x0)
        res = OptimizeResult(x=x0_arr, fun=objective(x0_arr), success=True,
                             message="Optimization bypassed (fit_mode=evaluate_x0)",
                             nit=0, nfev=1)
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
            _attach_wald_stats(out, nll_func_fixed, res.x, bounds, param_names)

        if slice_analysis or profile_likelihood_analysis or fast_profile_likelihood_analysis:
            nll_at_optimum = nll_func_fixed(res.x)
            out["stats"]["nll_at_optimum"] = nll_at_optimum

            def likelihood_slice_func(param_idx, n_points=20, range_factor=2.0):
                return _run_likelihood_slice_single(
                    param_idx, nll_func_fixed, res.x, nll_at_optimum, param_names,
                    n_points=n_points, range_factor=range_factor, scales=scales,
                )

            if slice_analysis:
                out["stats"]["likelihood_slice"] = likelihood_slice_func

            if fast_profile_likelihood_analysis:
                def fast_profile_likelihood_func(param_idx, n_points=3, range_factor=1.1, variation_pct=0.10):
                    return _run_fast_profile_likelihood_single(
                        param_idx, nll_func_fixed, bounds, res.x, nll_at_optimum,
                        param_names, variation_pct=variation_pct,
                        method=profile_method, optimizer_kwargs=profile_opt_kwargs,
                        scales=scales, stats_out=out["stats"],
                    )

                out["stats"]["profile_likelihood"] = fast_profile_likelihood_func
            elif profile_likelihood_analysis:
                def true_profile_likelihood_func(param_idx, n_points=20, range_factor=2.0):
                    se_array = out["stats"].get("wald_se")
                    wald_se_val = se_array[param_idx] if se_array is not None else None
                    return _run_pypesto_profile_single(
                        param_idx, nll_func_fixed, bounds, res.x, nll_at_optimum,
                        param_names, n_points=n_points, range_factor=range_factor,
                        fallback_func=likelihood_slice_func, wald_se_val=wald_se_val,
                        method=profile_method, optimizer_kwargs=profile_opt_kwargs,
                        scales=scales,
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
    fast_profile_likelihood_analysis=False,
    sobol_analysis=False,
    sobol_kwargs=None,
    fast=False,
    maxiter=None,
    tol=None,
    optimization_spec=None,
    fit_mode=None,
    n_workers=None,
    profile_checkpoint=True,
):
    """
    Optimize shared parameters using ``experiment.opt_groups`` or ``optimization_spec``.

    fit_mode : "optimize" (default) runs the optimizer; "evaluate_x0" skips the
        fit and evaluates the starting point, so diagnostics can be run against
        stored parameters.  Supersedes the deprecated ``profile_without_opt``.
    n_workers : processes used for the diagnostics (Wald / slice / Sobol).
        None uses all cores but one; 1 forces serial evaluation.  The fit itself
        is serial regardless -- Nelder-Mead is inherently sequential.
    profile_checkpoint : write each completed profile point to
        results/<MODEL>/profiles/<run_id>/<param>.jsonl so a killed run resumes
        instead of restarting.  Set False to disable.
    """
    if (profile_likelihood_analysis or fast_profile_likelihood_analysis) and not wald_analysis:
        print("Note: profile likelihood requested — also enabling wald_analysis so the "
              "Hessian-based CI is available as an independent cross-check.")
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
        x0_lin = optimization_spec.x0
        bounds_lin = optimization_spec.bounds
        method = optimization_spec.method
        optimizer_kwargs = optimization_spec.optimizer_kwargs or {}
        selected_group_names = set(optimization_spec.groups.keys())
        groups_tag = "_".join(sorted(selected_group_names))

        # ── Parameter scaling ─────────────────────────────────────────────
        # The optimizer, Hessian, slice and profile all work in "opt space";
        # everything reported back to the caller is converted to linear units.
        scales = _resolve_scales(
            getattr(optimization_spec, "parameter_scale", None), param_names
        )
        x0 = _to_opt_space(x0_lin, scales)
        bounds = _bounds_to_opt_space(bounds_lin, scales)
        if _any_log(scales):
            logged = [p for p, s in zip(param_names, scales) if s == "log10"]
            print(f"[opt] Fitting {len(logged)}/{len(param_names)} parameter(s) on a "
                  f"log10 scale: {logged}")

        fit_mode = _resolve_fit_mode(fit_mode, optimizer_kwargs)
        profile_method, profile_opt_kwargs = _resolve_profile_optimizer(
            method, optimizer_kwargs
        )
        settings_checkpoint = profile_checkpoint

        # ── Which simulations are needed, and which only for plotting ─────
        # Active simulations contribute to the loss and must be integrated on
        # every evaluation. Passive ones exist solely so the plot function gets
        # complete curves, so they are integrated once, after the fit.
        active_sim_names = set()
        for g_name, g_config in optimization_spec.groups.items():
            for elem in g_config.get("loss_elements", []):
                is_composite = elem.get("type") == "composite" or "simulations" in elem
                if is_composite:
                    sub_sims = elem.get("simulations", [])
                    active_sim_names.update(sub_sims)
                    data_sim = elem.get("data_simulation") or (sub_sims[0] if sub_sims else None)
                    if data_sim:
                        active_sim_names.add(data_sim)
                else:
                    active_sim_names.add(elem.get("simulation"))
        active_sim_names.discard(None)
        passive_sim_names = set(optimization_spec.passive_simulations) - active_sim_names
        unique_sim_names = active_sim_names | passive_sim_names
        if passive_sim_names:
            print(f"[opt] {len(active_sim_names)} simulation(s) in the objective; "
                  f"{len(passive_sim_names)} passive simulation(s) deferred to "
                  f"after the fit: {sorted(passive_sim_names)}")

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

            # events_str is retained so pool workers rebuild the *same* model
            # rather than regenerating events from data themselves.
            models[sim_name] = {"r_ic": r_ic, "r": r, "df_dict": df_dict,
                                "events": events_str}
            replicates[sim_name] = replicate

        # Active replicates only: passive ones are simulated once after the fit.
        active_replicates = {
            name: rep for name, rep in replicates.items() if name in active_sim_names
        }

        _debug_calls = [0]
        _progress = {"best": float("inf"), "t0": None}

        def set_params(r, p):
            set_parameters_from_dict(r, p)

        def objective(x):
            call_n = _debug_calls[0]
            do_debug = call_n < 3
            if call_n == 0:
                _progress["t0"] = time.time()

            # x arrives in opt space; the model always gets linear values.
            x_lin = _to_linear(x, scales)
            if do_debug:
                print(f"\n[opt debug] call #{call_n + 1}  "
                      + "  ".join(f"{n}={v:.4g}" for n, v in zip(param_names, x_lin.tolist())))

            x_dict = dict(zip(param_names, x_lin.tolist()))
            events_dynamic = optimizer_kwargs.get("events_depend_on_opt_param", False) if optimizer_kwargs else False

            sim_results = {}
            for sim_name, replicate in active_replicates.items():
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
                            x_lin, sub_results, sim_results[data_sim]["data"],
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
                        param_names=param_names, param_values=x_lin,
                        model_name=paths.get("MODEL_NAME", ""),
                        experiment_id=groups_tag, method=method
                    )
            _debug_calls[0] += 1
            return total_loss

        opt_kw = _prepare_optimizer_kwargs(method, optimizer_kwargs, fast, maxiter, tol)
        if fit_mode == "evaluate_x0":
            from scipy.optimize import OptimizeResult
            print("[opt] fit_mode='evaluate_x0' — skipping the fit and evaluating x0 "
                  "so diagnostics run against the supplied parameters.")
            x0_arr = np.array(x0)
            res = OptimizeResult(x=x0_arr, fun=objective(x0_arr), success=True,
                                 message="Optimization bypassed (fit_mode=evaluate_x0)",
                                 nit=0, nfev=1)
        elif method.lower() in _GLOBAL_METHODS:
            res = _run_global_optimization(objective, x0, bounds, method, opt_kw)
        else:
            res = minimize(objective, x0, method=method, bounds=bounds or [], **opt_kw)

        if _debug_calls[0] > 3:
            elapsed = time.time() - _progress["t0"]
            print(f"\n  [opt] done — {_debug_calls[0]} evals in {elapsed:.0f}s"
                  f"  best={_progress['best']:.5g}")

        # res.x is in opt space; everything reported out is linear.
        x_lin_opt = _to_linear(res.x, scales)

        out = {
            "x": x_lin_opt, "fun": res.fun, "success": res.success,
            "message": res.message, "stats": {},
            "groups": sorted(selected_group_names),
            "nit":  getattr(res, "nit",  None),
            "nfev": getattr(res, "nfev", None),
            "timestamp": _progress_overlay_state.get("timestamp"),
            "parameter_scale": list(scales),
            "x0": list(np.asarray(x0_lin, dtype=float)),
            "fit_mode": fit_mode,
        }

        param_dict = dict(zip(param_names, x_lin_opt.tolist()))
        best_results = {}
        fixed_sigmas = {}
        total_n = 0
        k = len(param_names)

        # Simulate every replicate once at the optimum -- this is where passive
        # (plot-only) simulations get their curves, instead of on every eval.
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

                            try:
                                # Composite path: same shared resolver, for the same reason.
                                y_sim = evaluate_observable(obs_s, result, param_dict)
                            except Exception as exc:
                                print(f"  [sigma] skipping {_short_obs_label(obs_s)}: {exc}")
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
                        try:
                            # Same resolver the loss uses, so an expression observable gets
                            # evaluated here too instead of silently becoming zeros.
                            y_sim = evaluate_observable(obs, result, param_dict)
                        except Exception as exc:
                            print(f"  [sigma] skipping {_short_obs_label(obs)}: {exc}")
                            continue
                        y_pred = np.interp(t_data, t_sim, y_sim)
                        valid = np.isfinite(y_pred) & np.isfinite(y_data)
                        if not valid.any():
                            continue
                        residuals = y_data[valid] - y_pred[valid]
                        sigma = np.sqrt(np.sum(residuals**2) / max(1, len(residuals) - k/max(1, len(observables))))
                        fixed_sigmas[(sim, obs)] = max(sigma, 1e-6)
                        total_n += len(residuals)

        # A silent fixed_sigmas miss is the difference between profiling a
        # likelihood and profiling a normalized residual, so say so loudly.
        _n_active_obs = 0
        for _g in optimization_spec.groups.values():
            for _e in _g.get("loss_elements", []):
                _sim = _e.get("simulation")
                if _sim not in replicates:
                    continue
                _lcf = _e.get("loss_config")
                _lc = _lcf(replicates[_sim]) if callable(_lcf) else _lcf
                _n_active_obs += len((_lc or {}).get("observables", []))

        if not fixed_sigmas:
            print()
            print("*** WARNING: fixed_sigmas is EMPTY. Every diagnostic falls back to "
                  "the point-averaged heuristic loss, which is NOT a log-likelihood, "
                  "so dNLL cannot be compared against the 1.9207 threshold. ***")
            print()
        elif _n_active_obs and len(fixed_sigmas) < _n_active_obs:
            print()
            print(f"*** WARNING: fixed_sigmas covers {len(fixed_sigmas)} of "
                  f"{_n_active_obs} active observable(s). The unmatched ones fall back "
                  f"to the point-averaged heuristic loss, so dNLL mixes likelihood and "
                  f"non-likelihood terms. Use Engine.Optimize.describe_nll_terms to see "
                  f"which. ***")
            print()

        def nll_func_fixed(p):
            return evaluate_nll_fixed(
                p, models, active_replicates, param_names, scales,
                optimization_spec.groups, optimization_spec.group_normalization,
                fixed_sigmas, model_text=model_text, paths=paths,
                events_dynamic=_events_dynamic,
            )

        out["results_dict"] = best_results

        _needs_diagnostics = (wald_analysis or slice_analysis
                              or profile_likelihood_analysis
                              or fast_profile_likelihood_analysis or sobol_analysis)

        # ── Parallel evaluation pool ──────────────────────────────────────
        # Every diagnostic below is a large batch of independent nll_func_fixed
        # calls, so it runs on a worker pool when one can be built. Falls back
        # to serial silently-but-loudly: the reason is always printed.
        evaluator = None
        if _needs_diagnostics and n_workers != 1:
            evaluator = _try_build_evaluator(
                model_text, paths, models, active_replicates, param_names, scales,
                optimization_spec, fixed_sigmas, _events_dynamic, n_workers,
            )
        out["stats"]["parallel"] = evaluator is not None

        _pool_state = {"evaluator": evaluator}

        def nll_batch(xs, label=None):
            """Evaluate many parameter vectors, in parallel when available.

            If the pool breaks mid-run — a worker segfaulting in the integrator
            leaves ProcessPoolExecutor permanently broken — fall back to serial
            for this and every later batch rather than losing hours of work.
            """
            ev = _pool_state["evaluator"]
            if ev is not None:
                try:
                    return ev.evaluate_batch(xs, label=label)
                except Exception as exc:
                    print(f"[pool] batch failed ({type(exc).__name__}: {exc}); "
                          f"falling back to serial evaluation for the rest of "
                          f"this run.")
                    try:
                        ev.shutdown()
                    except Exception:
                        pass
                    _pool_state["evaluator"] = None
                    out["stats"]["parallel"] = False
            return [nll_func_fixed(np.asarray(x, dtype=float)) for x in xs]

        try:
            if _needs_diagnostics:
                nll_proper = nll_func_fixed(res.x)
                aic = 2 * k + 2 * nll_proper
                bic = k * np.log(max(total_n, 1)) + 2 * nll_proper
                out["stats"]["aic"] = aic
                out["stats"]["bic"] = bic
                out["stats"]["nll_proper"] = nll_proper

            if wald_analysis:
                _attach_wald_stats(out, nll_func_fixed, res.x, bounds, param_names,
                                   scales=scales, nll_batch=nll_batch)

            if slice_analysis or profile_likelihood_analysis or fast_profile_likelihood_analysis:
                nll_at_optimum = nll_func_fixed(res.x)
                out["stats"]["nll_at_optimum"] = nll_at_optimum
                out["stats"]["fixed_sigmas"]   = fixed_sigmas

                def likelihood_slice_func(param_idx, n_points=20, range_factor=2.0):
                    return _run_likelihood_slice_single(
                        param_idx, nll_func_fixed, res.x, nll_at_optimum, param_names,
                        n_points=n_points, range_factor=range_factor, scales=scales,
                        nll_batch=nll_batch,
                    )

                if slice_analysis:
                    out["stats"]["likelihood_slice"] = likelihood_slice_func
                    # All parameters at once is the whole point: k x n_points
                    # independent evaluations with no dependencies between them.
                    out["stats"]["likelihood_slice_all"] = (
                        lambda n_points=20, range_factor=2.0:
                        _run_likelihood_slice_all(
                            nll_batch, res.x, nll_at_optimum, param_names,
                            n_points=n_points, range_factor=range_factor, scales=scales,
                        )
                    )

                if fast_profile_likelihood_analysis:
                    def fast_profile_likelihood_func(param_idx, n_points=3, range_factor=1.1, variation_pct=0.10):
                        return _run_fast_profile_likelihood_single(
                            param_idx, nll_func_fixed, bounds, res.x, nll_at_optimum,
                            param_names, variation_pct=variation_pct,
                            method=profile_method, optimizer_kwargs=profile_opt_kwargs,
                            scales=scales, stats_out=out["stats"],
                        )

                    out["stats"]["profile_likelihood"] = fast_profile_likelihood_func
                elif profile_likelihood_analysis:
                    def true_profile_likelihood_func(param_idx, n_points=20, range_factor=2.0):
                        se_array = out["stats"].get("wald_se")
                        wald_se_val = se_array[param_idx] if se_array is not None else None
                        return _run_pypesto_profile_single(
                            param_idx, nll_func_fixed, bounds, res.x, nll_at_optimum,
                            param_names, n_points=n_points, range_factor=range_factor,
                            fallback_func=likelihood_slice_func, wald_se_val=wald_se_val,
                            method=profile_method, optimizer_kwargs=profile_opt_kwargs,
                            scales=scales,
                        )
                    out["stats"]["profile_likelihood"]  = true_profile_likelihood_func

                    # Parallel, checkpointed alternative to the sequential
                    # walker. Model_optimize prefers this when a pool exists.
                    if _pool_state["evaluator"] is not None:
                        out["stats"]["profile_likelihood_all"] = (
                            lambda n_grid=5, range_factor=2.0, se_span=4.0,
                            n_refine=2, run_id=None:
                            _run_parallel_profile_with_checkpoint(
                                _pool_state["evaluator"], res.x, nll_at_optimum,
                                param_names, bounds, scales,
                                groups=optimization_spec.groups,
                                model_text=model_text, paths=paths,
                                method=profile_method,
                                optimizer_kwargs=profile_opt_kwargs,
                                wald_se=out["stats"].get("wald_se"),
                                n_grid=n_grid, range_factor=range_factor,
                                se_span=se_span, n_refine=n_refine,
                                run_id=run_id or groups_tag,
                                checkpoint_enabled=settings_checkpoint,
                                fixed_sigmas=fixed_sigmas,
                            )
                        )

            if sobol_analysis:
                from Engine.Sensitivity_analysis import run_sobol_analysis
                skwargs = sobol_kwargs or {}
                out["stats"]["sobol"] = run_sobol_analysis(
                    nll_func_fixed, param_names, bounds, res.x,
                    nll_batch=nll_batch, **skwargs
                )
        except Exception as e:
            import traceback
            print(f"Warning: diagnostics failed: {e}")
            traceback.print_exc()
        finally:
            # The closures above are consumed by Model_optimize *after* this
            # function returns, so the pool cannot be torn down here. Hand it to
            # the caller and let it close the pool once plotting is done.
            if _pool_state["evaluator"] is not None:
                out["stats"]["_evaluator"] = _pool_state["evaluator"]

        out["r"] = list(models.values())[0]["r"] if models else None
        return out

    # =========================================================================
    # LEGACY / FLAT REPLICATE ROUTE
    # =========================================================================

    # Legacy dict-based route: linear-only (no parameter_scale field exists here).
    scales = ["lin"] * len(param_names)
    fit_mode = _resolve_fit_mode(fit_mode, optimizer_kwargs)
    profile_method, profile_opt_kwargs = _resolve_profile_optimizer(method, optimizer_kwargs)

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
    
    if fit_mode == "evaluate_x0":
        from scipy.optimize import OptimizeResult
        print("[opt] fit_mode='evaluate_x0' — skipping the fit and evaluating x0 "
              "so diagnostics run against the supplied parameters.")
        x0_arr = np.array(x0)
        res = OptimizeResult(x=x0_arr, fun=objective(x0_arr), success=True,
                             message="Optimization bypassed (fit_mode=evaluate_x0)",
                             nit=0, nfev=1)
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
                
                # Unweighted: replicate_weight shapes the fit, but the joint
                # log-likelihood is the plain sum. Weighting it would rescale
                # every dNLL and invalidate the 1.9207 threshold.
                total_nll += loss_function(
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
                # Unweighted, for the same reason as above.
                total_nll += loss_val

        return total_nll

    out["results_dict"] = best_results

    if wald_analysis or slice_analysis or profile_likelihood_analysis or fast_profile_likelihood_analysis or sobol_analysis:
        nll_proper = nll_func_fixed(res.x)
        aic = 2 * k + 2 * nll_proper
        bic = k * np.log(max(total_n, 1)) + 2 * nll_proper
        out["stats"]["aic"] = aic
        out["stats"]["bic"] = bic
        out["stats"]["nll_proper"] = nll_proper

    if wald_analysis:
        _attach_wald_stats(out, nll_func_fixed, res.x, bounds, param_names)

    if slice_analysis or profile_likelihood_analysis or fast_profile_likelihood_analysis:
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
                return _run_likelihood_slice_single(
                    param_idx, nll_func_fixed, res.x, nll_at_optimum, param_names,
                    n_points=n_points, range_factor=range_factor, scales=scales,
                )

            if slice_analysis:
                out["stats"]["likelihood_slice"] = likelihood_slice_func

            if fast_profile_likelihood_analysis:
                def fast_profile_likelihood_func(param_idx, n_points=3, range_factor=1.1, variation_pct=0.10):
                    return _run_fast_profile_likelihood_single(
                        param_idx, nll_func_fixed, bounds, res.x, nll_at_optimum,
                        param_names, variation_pct=variation_pct,
                        method=profile_method, optimizer_kwargs=profile_opt_kwargs,
                        scales=scales, stats_out=out["stats"],
                    )

                out["stats"]["profile_likelihood"] = fast_profile_likelihood_func
            elif profile_likelihood_analysis:
                def true_profile_likelihood_func(param_idx, n_points=20, range_factor=2.0):
                    se_array = out["stats"].get("wald_se")
                    wald_se_val = se_array[param_idx] if se_array is not None else None
                    return _run_pypesto_profile_single(
                        param_idx, nll_func_fixed, bounds, res.x, nll_at_optimum,
                        param_names, n_points=n_points, range_factor=range_factor,
                        fallback_func=likelihood_slice_func, wald_se_val=wald_se_val,
                        method=profile_method, optimizer_kwargs=profile_opt_kwargs,
                        scales=scales,
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

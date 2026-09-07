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
from Engine.Event_times import attach_event_times
from Engine.Simulate import simulate
from Engine.Profile_checkpoint import record_is_better as _profile_record_is_better
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


def _block_title_stats(tr, blocks):
    """The block's own term in the objective, for a panel title.

    *blocks* is the ``{(block_key, obs): [sse, n, known_sigma]}`` mapping the
    concentrated likelihood is summed over, and the trace carries which block
    it fed. Returns ``(n, sigma, term, tag)`` or None when the trace cannot be
    matched -- a legacy route with no blocks, for instance.
    """
    if not blocks or tr.get("block") is None:
        return None
    v = blocks.get(tuple(tr["block"]))
    if v is None:
        return None
    sse, n, ks = _unpack_block(v)
    if n <= 0:
        return None
    if ks is None:
        s2 = max(sse / n, _SIGMA2_FLOOR)
        return n, float(np.sqrt(s2)), 0.5 * n * np.log(s2), "sigma fitted"
    ks = max(float(ks), 1e-300)
    return n, ks, sse / (2.0 * ks * ks), "sigma fixed"


def _render_progress_overlay(trace_collector, total_loss, best_loss, eval_n,
                             plot_path, min_interval_s=2.0,
                             param_names=None, param_values=None,
                             model_name=None, experiment_id=None, method=None,
                             blocks=None):
    """Save a multi-panel overlay (model curve + data points) per (replicate,
    observable) traced during an optimization eval. Reuses one Figure and
    overwrites a single PNG so the file refreshes in place.

    With *blocks* each panel is titled with the block's actual term in the
    objective -- its n, its sigma and its (n/2)log(SSE/n) or SSE/(2 sigma^2)
    -- rather than the trace's legacy per-point chi-square, which is scaled
    by a heuristic sigma and is not what the fit minimizes. On the
    aggregation spec that legacy number showed the centiloid block at 6e-6
    and MFL42 at 0.07 while the centiloid block, at n=501, was the one
    pinning the fit.
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
        stats_line = _block_title_stats(tr, blocks)
        if stats_line is not None:
            n_b, sig_b, term_b, tag_b = stats_line
            detail = (f"n={n_b:.0f}  sigma={sig_b:.3g} ({tag_b})"
                      f"  nll={term_b:+.4g}")
        else:
            detail = f"contrib={tr['contrib']:.4g}"
        ax.set_title(f"{_short_obs_label(rep, 28)} · {_short_obs_label(obs, 28)}"
                     f"\n{detail}", fontsize=9)
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

def run_all(r, exp_num, experiment, df_dict, set_parameters=None, parameters=None,
            preequil_cache=None):
    """
    Run one simulation for *experiment* using the pre-built Tellurium model *r*
    and pre-loaded *df_dict*, optionally applying *parameters* first.

    With *preequil_cache* supplied, the leading untracked pre-dose block is
    resolved from the cache and the fitted parameters are applied *after* it
    rather than before. That reordering is what makes the block reusable, and it
    is sound only while those parameters cannot act before the first dose --
    which ``Engine.Preequil_cache.verify_invariance`` establishes at startup and
    without which the cache is never enabled. With no cache the original order
    is used unchanged.

    Returns a results dict keyed by treatment label.
    """
    r.reset()

    # Re-apply treatment-specific parameters which were wiped out by r.reset()
    update_params = experiment.get("Update_parameters")
    if update_params is not None:
        update_params(r, experiment)

    solver_settings  = experiment["Solver_settings"](experiment)
    observed_species = experiment["Observed_species"](r)
    label            = experiment.get("Label")

    # Must happen before the fitted parameters are applied: the cache key is the
    # pre-dose setup, which has to be free of them to be the same every
    # evaluation.
    if preequil_cache is not None:
        solver_settings = preequil_cache.apply(
            r, solver_settings, observed_species, label=label)

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
        update_opt(r, experiment, parameters)

    results = simulate(r, solver_settings, observed_species, label=label)

    return {label: {"results": results, "data": df_dict, "replicate": experiment}}


def set_parameters_from_dict(r, params):
    """Apply a name -> value dict to a Tellurium model."""
    for name, value in params.items():
        try:
            r[name] = value
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Concentrated (profile-out-sigma) Gaussian likelihood
# ---------------------------------------------------------------------------
#
# One objective serves both the fit and every diagnostic. For a block of n
# residuals with unknown sigma, the Gaussian NLL is
#
#     n*log(sigma) + SSE/(2 sigma^2) + (n/2) log(2 pi)
#
# which is minimized at the ML variance sigma^2 = SSE/n. Substituting that back
# eliminates sigma analytically and leaves
#
#     NLL = (n/2) * log(SSE/n)  +  (n/2) * (1 + log(2 pi))
#
# The second term depends only on n, so it cancels from every dNLL and shifts
# no minimum -- but it is a real part of the likelihood, so AIC/BIC and any
# cross-model comparison must add it back (``include_constant=True``).
#
# Why this is the right objective here:
#   * No chicken-and-egg. Older code fitted a weighted chi-square, estimated
#     sigma from the residuals at that optimum, then profiled a *different*
#     function -- so the reported optimum was not the minimizer of the curve
#     being profiled, and dNLL went negative.
#   * The log makes each block scale-free, which is the principled version of
#     "weight the experiments equally": a block measured in different units or
#     with a wider dynamic range can no longer dominate.
#   * dNLL is a genuine likelihood ratio, so the chi2(1)/2 = 1.9207 threshold
#     applies directly.
#
# Note the n_e/2 factor still means a 50-point block counts ten times a 5-point
# block. That is correct under a likelihood. Wanting them equal regardless of
# point count is a claim about correlated residuals within a block, and belongs
# in a correlated noise model -- not in a 1/n divide on a log-likelihood.

# Guards log(0) when a block fits exactly. Squared-residual units.
_SIGMA2_FLOOR = 1e-30

# Below this, a block's sigma is estimated from too little data to trust, and
# the (n/2)log(SSE/n) term is at risk of running away if the model can fit the
# block almost exactly. Warned about after the fit, not enforced.
_MIN_BLOCK_POINTS = 5


def _known_sigma_for(obs_cfg, obs_df=None, valid=None):
    """The sigma this observable *asserts*, or None to estimate it.

    Restores the explicit noise specifications as first-class known-sigma
    blocks. This is not a nicety: a block of one point cannot support a variance
    estimate at all -- its concentrated contribution is ``log|residual|``, which
    is unbounded below -- so a lone scalar constraint is only usable if its
    precision is asserted rather than inferred.

    Recognised, in priority order:
      * ``noise_column``  per-datapoint sigma from the data (PEtab style);
                          reduced to its RMS so the block keeps one sigma.
      * ``sigma_method="fixed"`` with ``sigma_value``.
    ``sigma_method="relative"`` is deliberately *not* handled here: it depends
    on the prediction, so it changes every evaluation and is not a fixed known
    sigma. Everything else estimates sigma.
    """
    col = obs_cfg.get("noise_column")
    if col and obs_df is not None and col in getattr(obs_df, "columns", []):
        s = np.asarray(obs_df[col], dtype=float)
        if valid is not None:
            s = s[valid]
        s = s[np.isfinite(s) & (s > 0)]
        if s.size:
            return float(np.sqrt(np.mean(s ** 2)))
    if obs_cfg.get("sigma_method") == "fixed":
        v = obs_cfg.get("sigma_value")
        if v is not None and float(v) > 0:
            return float(v)
    return None


def _record_block(blocks, key, obs, residuals, weights, known_sigma=None):
    """Accumulate weighted SSE and effective count for one observable block.

    Entries are ``[weighted_sse, n_eff, known_sigma_or_None]``. A block with a
    known sigma is scored with the plain Gaussian NLL and costs no parameter;
    one without has its sigma profiled out.
    """
    if blocks is None:
        return
    acc = blocks.setdefault((key, _short_obs_label(obs)), [0.0, 0.0, known_sigma])
    acc[0] += float(np.dot(weights, np.asarray(residuals, dtype=float) ** 2))
    acc[1] += float(np.sum(weights))
    # Pooled blocks must agree: if any contributor asserts a sigma, the pool is
    # known-sigma. Mixing asserted and estimated noise in one pool is a spec
    # error, so the first assertion wins and stays.
    if acc[2] is None and known_sigma is not None:
        acc[2] = known_sigma


def _unpack_block(v):
    """(sse, n, known_sigma) from a 2- or 3-element block entry."""
    if len(v) >= 3:
        return float(v[0]), float(v[1]), v[2]
    return float(v[0]), float(v[1]), None


def concentrated_nll(blocks, include_constant=False):
    """Gaussian NLL over *blocks*.

    Blocks without a declared sigma have theirs profiled out analytically
    (``(n/2)log(SSE/n)``); blocks with a known sigma get the plain Gaussian NLL
    (``SSE/(2*sigma^2)``). The two compose in one likelihood, so a spec may mix
    them freely.

    *blocks* maps a key to ``[weighted_sse, n_eff, known_sigma_or_None]``.
    Returns ``0.0`` for an empty mapping so a spec with no resolvable data does
    not silently produce ``nan``.
    """
    total = 0.0
    for v in blocks.values():
        sse, n, ks = _unpack_block(v)
        if n <= 0:
            continue
        if ks is None:
            total += 0.5 * n * np.log(max(sse / n, _SIGMA2_FLOOR))
            if include_constant:
                total += 0.5 * n * (1.0 + np.log(2.0 * np.pi))
        else:
            ks = max(float(ks), 1e-300)
            total += sse / (2.0 * ks * ks)
            if include_constant:
                total += n * np.log(ks) + 0.5 * n * np.log(2.0 * np.pi)
    return float(total)


def block_sigmas(blocks):
    """Sigma per block: the ML estimate, or the declared value where given.

    The ML estimate is deliberately not degrees-of-freedom corrected -- a
    corrected sigma substituted back into the concentrated form would no longer
    be the profile likelihood, and the 1.9207 threshold would stop being exact.
    """
    out = {}
    for key, v in blocks.items():
        sse, n, ks = _unpack_block(v)
        if n <= 0:
            continue
        out[key] = float(ks) if ks is not None else float(
            np.sqrt(max(sse / n, _SIGMA2_FLOOR)))
    return out


def known_sigma_blocks(blocks):
    """Keys whose sigma was declared rather than estimated."""
    return {k for k, v in blocks.items() if _unpack_block(v)[2] is not None}


def total_points(blocks):
    """Effective number of data points across every block."""
    return float(sum(_unpack_block(v)[1] for v in blocks.values()))


def effective_k(param_names, blocks):
    """Parameters an information criterion must charge for.

    The estimated parameters *plus* one sigma for every block whose sigma is
    *estimated*. Profiling sigma out analytically makes it invisible in the
    objective but does not make it free -- it is still fitted from the same
    data. A block with a declared sigma costs nothing, because nothing about it
    was estimated.
    """
    n_est = 0
    for v in blocks.values():
        sse, n, ks = _unpack_block(v)
        if n > 0 and ks is None:
            n_est += 1
    return len(param_names) + n_est


def collect_loss_blocks(
    sim_results, groups, replicates, param_names, p_lin, p_dict,
    trace_collector=None, loss_components=None, seen=None,
):
    """Walk every loss element in *groups* and return ``{block: [sse, n]}``.

    Mirrors :func:`accumulate_group_nll`'s traversal so the concentrated
    likelihood and the legacy weighted objective always see the same elements,
    but keys blocks by the element's own identity rather than by any display
    label -- the fit and the diagnostics must agree on block identity or their
    sigmas would not correspond.

    **Pooling.** A loss element may declare ``"sigma_block": "<name>"`` to share
    one estimated sigma with every other element carrying the same name. Default
    is one block per simulation, which is right when each simulation has its own
    noise process -- but wrong, and sometimes catastrophically so, when several
    arms of one trial were measured on one assay:

      * A block's contribution is ``(n/2)log(SSE/n)``, which runs to -inf as
        SSE -> 0. Blocks of 1-3 points are easy for a flexible model to fit
        almost exactly, so they can hijack the objective. Pooling them raises n
        and removes the cliff.
      * Every block costs a parameter in AIC/BIC. A spec with 12 two-point
        blocks spends 12 parameters on noise alone, which can approach the total
        number of data points.

    Pooling arms that genuinely share a noise process fixes both at once, and is
    the more defensible model besides.

    If *seen* is a set, the pre-pooling ``(element, observable)`` identity of
    every contributing block is added to it. Block count cannot measure data
    coverage once elements share a sigma_block, so coverage checks must count
    these instead.
    """
    blocks = {}
    for g_name, g_config in groups.items():
        for idx, elem in enumerate(g_config.get("loss_elements", [])):
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
                    key, elem, param_names, loss_config=lc,
                    trace_collector=trace_collector, blocks=blocks,
                    block_key=elem.get("sigma_block") or key, seen=seen,
                )
            else:
                sim = elem.get("simulation")
                if sim not in sim_results:
                    continue
                lc = lc_fn(replicates[sim]) if callable(lc_fn) else lc_fn
                key = sim
                loss_val = loss_function_evaluated(
                    p_dict, {f"{g_name} · {sim}": sim_results[sim]}, param_names,
                    loss_config=lc, trace_collector=trace_collector,
                    blocks=blocks, block_key=elem.get("sigma_block") or key,
                    seen=seen,
                )

            if loss_components is not None:
                loss_components[key] = loss_val

    return blocks


def loss_function_evaluated(
    param_dict,
    results_dict,
    param_names,
    loss_config=None,
    fixed_sigmas=None,
    debug=False,
    trace_collector=None,
    blocks=None,
    block_key=None,
    seen=None,
):
    """
    Evaluate the loss for already simulated results.

    If *blocks* is a dict, the weighted sum of squared residuals and the
    effective point count are recorded into it under
    ``(block_key or exp_id, observable_label)``.  That is everything
    :func:`concentrated_nll` needs, and it is captured before any sigma branch
    runs, so it does not depend on ``loss_type`` or on which sigma heuristic
    fired.  *block_key* lets the caller pin a stable identity for the element
    even when ``results_dict`` is keyed by a display label.
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

            _record_block(blocks, block_key if block_key is not None else exp_id,
                          obs, residuals, obs_weights_v,
                          known_sigma=_known_sigma_for(obs_cfg, obs_df, valid))
            if seen is not None:
                # Pre-pooling identity. Block count cannot measure data coverage
                # once several elements share a sigma_block, so record which
                # (element, observable) pairs actually contributed residuals.
                seen.add((str(exp_id), _short_obs_label(obs)))

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
                    # Which likelihood block this trace fed, keyed exactly as
                    # _record_block keys it, so the overlay can show the
                    # block's real term rather than the legacy contrib.
                    "block":   (block_key if block_key is not None else exp_id,
                                obs_label),
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
    blocks=None,
    block_key=None,
    seen=None,
):
    """
    Evaluate aggregated composite loss across multiple simulated results.

    *blocks* / *block_key* behave exactly as in :func:`loss_function_evaluated`.
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

        _record_block(blocks, block_key if block_key is not None else exp_id,
                      obs, residuals, obs_weights_v,
                      known_sigma=_known_sigma_for(obs_cfg, obs_df, valid))
        if seen is not None:
            seen.add((str(exp_id), _short_obs_label(obs)))

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
                "block":   (block_key if block_key is not None else exp_id,
                            obs_label),
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
                attach_event_times(replicate, r_to_use)
            else:
                m["r"].reset()
                r_to_use = m["r"]
            # Only the persistent per-arm model carries a cache. The dynamic
            # branch above compiles a fresh RoadRunner every evaluation, so
            # there is nothing to reuse and nothing to key on.
            run_res = run_all(r_to_use, sim_name, replicate, m["df_dict"],
                              set_parameters=set_parameters_from_dict,
                              parameters=p_dict,
                              preequil_cache=(None if events_dynamic
                                              else m.get("preequil_cache")))
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
    failure_value=1e10, for_inference=True, concentrated=True,
    include_constant=False,
):
    """Joint NLL at *p* (optimizer space).

    This is the function every diagnostic consumes -- Wald, slice, profile,
    Sobol, AIC/BIC -- and the one the parallel pool evaluates in workers.

    ``concentrated=True`` (the default) evaluates the concentrated Gaussian
    likelihood, with each block's sigma profiled out analytically. It is the
    *same* function the fit minimizes, which is what makes the reported optimum
    the minimizer of the curve being profiled and lets dNLL be compared with
    1.9207 directly. ``fixed_sigmas`` is unused on this path.

    ``concentrated=False`` reproduces the older frozen-sigma behaviour, where
    ``fixed_sigmas`` carries sigmas estimated once at the optimum. Retained for
    comparison against archived runs; it is not the inference path.
    """
    p_lin = _to_linear(p, scales)
    p_dict = dict(zip(param_names, p_lin.tolist()))
    sim_results = simulate_active_replicates(
        p_dict, models, replicates, param_names,
        model_text=model_text, paths=paths, events_dynamic=events_dynamic,
    )
    if sim_results is None:
        return failure_value

    if concentrated:
        blocks = collect_loss_blocks(
            sim_results, groups, replicates, param_names, p_lin, p_dict,
        )
        if not blocks:
            return failure_value
        return concentrated_nll(blocks, include_constant=include_constant)

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
                       group_normalization=None, fixed_sigmas=None):
    """Per-block breakdown of the concentrated NLL at *p_vec*.

    Answers the question that matters when a dNLL looks wrong: which block is
    driving it? Under the concentrated likelihood a block contributes exactly

        (n/2) * log(SSE/n)

    so its whole influence is summarised by its ML sigma and its point count.
    A block that fits badly shows a large sigma; a block with many points
    carries more weight for the same sigma.

    ``group_normalization`` and ``fixed_sigmas`` are accepted and ignored --
    neither enters the likelihood any more. They are kept in the signature so
    existing call sites do not break.

    Returns a list of dicts, one per (block, observable).
    """
    p_lin = _to_linear(p_vec, scales)
    p_dict = dict(zip(param_names, p_lin.tolist()))
    sim_results = simulate_active_replicates(p_dict, models, replicates, param_names)
    if sim_results is None:
        return []

    blocks = collect_loss_blocks(
        sim_results, groups, replicates, param_names, p_lin, p_dict,
    )
    total = concentrated_nll(blocks)

    rows = []
    for (block_key, obs_label), v in sorted(blocks.items()):
        sse, n, known = _unpack_block(v)
        if n <= 0:
            rows.append({"block": block_key, "obs": obs_label, "n": 0,
                         "status": "no-valid-points"})
            continue
        if known is None:
            sigma = float(np.sqrt(max(sse / n, _SIGMA2_FLOOR)))
            contrib = 0.5 * n * np.log(max(sse / n, _SIGMA2_FLOOR))
        else:
            sigma = float(known)
            contrib = sse / (2.0 * sigma * sigma)
        rows.append({
            "block": block_key,
            "obs": obs_label,
            "n": int(round(n)),
            "sigma": sigma,
            "sigma_source": "declared" if known is not None else "estimated",
            "sum_sq": float(sse),
            "mean_sq": float(sse / n),
            "rms": float(np.sqrt(sse / n)),
            "contrib": float(contrib),
            "share": float(contrib / total) if total else float("nan"),
            "status": "ok",
        })

    # Which observables the spec asked for but no block covered: a silent miss
    # here means a whole dataset is contributing nothing to the fit.
    expected = set()
    for g_cfg in groups.values():
        for elem in g_cfg.get("loss_elements", []):
            sim = elem.get("simulation")
            if sim is None or sim not in replicates:
                continue
            lc_fn = elem.get("loss_config")
            lc = lc_fn(replicates[sim]) if callable(lc_fn) else lc_fn
            for oc in (lc or {}).get("observables", []):
                expected.add((sim, _short_obs_label(oc["observed_variable"])))
    for key in sorted(expected - set(blocks)):
        rows.append({"block": key[0], "obs": key[1], "n": 0,
                     "status": "unresolved-data"})
    return rows


def print_nll_decomposition(rows, title="NLL decomposition"):
    """Human-readable table from :func:`describe_nll_terms`."""
    print("\n" + "=" * 112)
    print(title + "   [concentrated Gaussian likelihood]")
    print("=" * 112)
    print(f"{'block':<28} {'observable':<22} {'n':>5} {'sigma':>12} {'source':<10} "
          f"{'sum r^2':>12} {'contrib':>12} {'share':>7}")
    print("-" * 112)
    n_ok = 0
    unresolved = []
    total = sum(r.get("contrib", 0.0) for r in rows if r.get("status") == "ok")
    for r in rows:
        if r.get("status") != "ok":
            unresolved.append(r)
            continue
        n_ok += 1
        print(f"{str(r['block']):<28} {r['obs']:<22} {r['n']:>5} {r['sigma']:>12.4g} "
              f"{r.get('sigma_source', 'estimated'):<10} "
              f"{r['sum_sq']:>12.4g} {r['contrib']:>12.4g} "
              f"{100 * r['contrib'] / total if total else float('nan'):>6.1f}%")
    print("-" * 112)
    print(f"total NLL (constant omitted): {total:.6g}   across {n_ok} block(s)")
    if unresolved:
        print()
        print(f"*** WARNING: {len(unresolved)} observable(s) contributed NOTHING to "
              f"the likelihood — no data resolved. Check data_column / time_column / "
              f"data_dict_key for: ***")
        for r in unresolved:
            print(f"      {r['block']} · {r['obs']}  ({r.get('status')})")

# Hessian utilities
# ---------------------------------------------------------------------------

def compute_hessian_numdifftools(func, params):
    try:
        hessian_func = nd.Hessian(func, method='central', step=1e-5)
        return hessian_func(params)
    except Exception as e:
        warnings.warn(f"Error computing Hessian with numdifftools: {e}")
        return compute_hessian_manual(func, params)

# Central-difference step for a log10 coordinate, in decades. See
# _finite_difference_steps: 0.02 keeps the noise term 4*sigma/h^2 near 0.1 for
# objective noise anywhere from 1e-7 to 1e-4 nats, while the truncation term
# H*h^2/12 stays near 0.003 for curvatures of order 100. Two decades of margin
# either side, which is what a numerically noisy objective needs.
_LOG_FD_STEP = 0.02


def _finite_difference_steps(params, epsilon=1e-4, abs_floor=None, scales=None,
                             log_step=_LOG_FD_STEP):
    """Per-parameter step for central differences.

    For a **linear** coordinate the step is *relative*, not absolute. The
    obvious rule -- ``eps * max(|p|, 1.0)``, which compute_hessian_manual uses
    -- degenerates into a fixed 1e-5 step for every parameter below 1.0. PBPK
    rate constants routinely sit at 1e-8 or smaller (AGGREGATION_BOUNDS reaches
    1e-12), so that step is many orders of magnitude larger than the parameter:
    it pushes the value negative, trips the ``p <= 0`` guard, and the stencil
    comes back as a wall of 1e10 sentinels.

    For a **log10** coordinate the step is *absolute*, in decades, and a
    relative rule is actively wrong. In log space the coordinate is log10(p),
    so ``epsilon * |log10(p)|`` scales the step by the parameter's distance
    from 1.0, which means nothing: on silk_appfull it spread the steps 55-fold
    and handed the smallest one to IDE_Kcat_ISF_base purely because its value
    happens to sit near 1. A parameter at exactly 1.0 would get a step of zero.
    Log space exists to make coordinates comparable; the step should be too.

    ``log_step`` is also much larger than epsilon on purpose. The classic
    "h ~ macheps^(1/4)" balance behind epsilon=1e-4 assumes the only noise is
    machine epsilon. A central second difference divides by h^2, so an
    objective with numerical noise sigma carries an error of about 4*sigma/h^2
    into every Hessian diagonal -- and an adaptive ODE solve is noisy at around
    1e-5 nats, not 1e-16. At h=3e-4 that is an error near 450 against curvatures
    of order 100: the Hessian becomes noise, the matrix stops being positive
    definite, and sqrt(diag(inv(H))) returns NaN.
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
    steps = np.maximum(steps, abs_floor)

    if scales is not None:
        for i, s in enumerate(scales[:steps.size]):
            if s == "log10":
                steps[i] = float(log_step)
    return steps


def compute_hessian_batched(nll_batch, params, epsilon=1e-4, scales=None):
    """Central-difference Hessian evaluated as one batch.

    The stencil is fixed in advance -- 1 centre, 2k diagonal points and 4 points
    per off-diagonal pair -- so every evaluation can be submitted at once
    instead of trickling through numdifftools one call at a time. That is
    2k^2 + 1 evaluations with no dependencies, which is exactly what the pool
    is for.
    """
    params = np.atleast_1d(np.asarray(params, dtype=float))
    n = params.size
    steps = _finite_difference_steps(params, epsilon, scales=scales)

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
    """Correlation matrix from a covariance matrix.

    A negative variance on the diagonal is not a numerical curiosity to be
    silenced -- it says the Hessian was not positive definite at the anchor, so
    that direction is not a minimum. Left alone it surfaces as a bare
    ``RuntimeWarning: invalid value encountered in sqrt``, which names neither
    the cause nor the parameters. Say it plainly instead.
    """
    if cov_matrix is None:
        return None
    try:
        diag = np.diag(np.asarray(cov_matrix, dtype=float))
        bad = int(np.sum(~(diag > 0)))
        if bad:
            print(f"[Wald] {bad}/{diag.size} parameter(s) have a non-positive "
                  f"variance, so the Hessian is not positive definite here and "
                  f"their SE and correlations come back NaN. Two usual causes: "
                  f"the anchor is not actually a minimum (check the fit, or that "
                  f"--no-fit was given the fitted x0), or the finite-difference "
                  f"step sits below the objective's numerical noise floor, which "
                  f"makes the Hessian noise.")
        with np.errstate(invalid="ignore"):
            std_devs = np.sqrt(diag)
            return cov_matrix / np.outer(std_devs, std_devs)
    except Exception as e:
        print(f"Error computing correlation matrix: {e}")
        return None


def compute_wald_uncertainty(nll_func, x, bounds=None, loss_scale=1.0, alpha=0.05,
                             nll_batch=None, scales=None):
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
            hessian_raw = compute_hessian_batched(nll_batch, x, scales=scales)
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


def _make_anchor_cache(paths, tag, param_names, x_opt, groups, scales,
                       model_text, fixed_sigmas, bounds, enabled=True):
    """An :class:`AnchorCache` beside this run's profile checkpoints, or None.

    Keyed on exactly what the Hessian depends on, so it is shared with the
    profile's own checkpoint directory and invalidated by the same changes.
    Never raises: a cache that cannot be built just means the Hessian is
    recomputed, which is what happened before it existed.
    """
    try:
        from Engine.Profile_checkpoint import spec_fingerprint, default_run_id
        from Engine.Anchor_cache import AnchorCache, bounds_fingerprint

        root = paths.get("plot_path") if enabled else None
        if not root:
            return None
        model_hash, spec_hash = spec_fingerprint(
            param_names, x_opt, groups, scales, model_text,
            fixed_sigmas=fixed_sigmas,
        )
        return AnchorCache(
            root, default_run_id(tag, model_hash, spec_hash),
            model_hash, spec_hash, bounds_fingerprint(bounds),
            n_params=len(param_names), enabled=True,
        )
    except Exception as exc:
        print(f"[Wald] no Hessian cache ({exc}); it will be recomputed.")
        return None


def _attach_wald_stats(out, nll_func, x, bounds, param_names=None, scales=None,
                       nll_batch=None, cache=None):
    """Compute Wald statistics and store them in ``out["stats"]``.

    Shared by all optimization routes so they cannot drift apart again.  When
    *scales* is given the Hessian is taken in the optimizer's (possibly log10)
    space and the results are converted back to linear units for reporting —
    see :func:`_transform_wald_to_linear`.

    With a *cache*, the whole block is reloaded rather than recomputed when the
    model, spec, optimum, scaling and bounds are unchanged. On the SILK APP
    spec that is 513 evaluations at 116 s each -- about 25 minutes of a
    four-hour link, paid again by every link of a chain.
    """
    if cache is not None:
        cached = cache.load()
        if cached is not None:
            out["stats"].update(cached)
            print(f"[Wald] reusing the Hessian cached at {cache.path} "
                  f"(same model, spec, optimum, scaling and bounds).")
            return

    try:
        cov, se, ci = compute_wald_uncertainty(nll_func, x, bounds=bounds,
                                               nll_batch=nll_batch, scales=scales)
        corr = compute_parameter_correlations(cov) if cov is not None else None
        # Kept in the optimizer's own space, before the transform below. The
        # profile and slice walkers place their grids as ``p_opt +/- span*se``
        # with p_opt in opt space, so they need the SE in that space too. The
        # reported ``wald_se`` is in linear units, and handing that to a
        # log10-fitted parameter is a units error with a spectacular failure
        # mode: on PK_Aducanumab it made the grid ~1e4 times too narrow, so
        # every side "ran out of extension steps" a fraction of a percent from
        # the optimum and the interval came back open. It stayed hidden until
        # parameter_scale switched to "auto" on 2026-08-27, because before that
        # these parameters were linear and the two spaces coincided.
        out["stats"]["wald_se_opt"] = se
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
        if cache is not None:
            cache.save(out["stats"])
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
# "auto" is an input only; it is resolved per parameter before anything else
# sees the list, so every consumer still deals in _VALID_SCALES.
_SCALE_INPUTS = _VALID_SCALES + ("auto",)
_LN10 = np.log(10.0)

# A range spanning at least this many decades between two positive numbers is
# taken as multiplicative. Bounds written as (v/10, v*10) span exactly two.
_AUTO_MIN_DECADES = 1.0

_VALID_FIT_MODES = ("optimize", "evaluate_x0")

# Keys authors may place in optimizer_kwargs that configure the engine rather
# than scipy. They must be stripped before any call to scipy.optimize.
_ENGINE_ONLY_OPTIMIZER_KEYS = (
    "events_depend_on_opt_param",
    "profile_without_opt",      # deprecated, superseded by run_settings["fit_mode"]
    "profile_method",
    "profile_optimizer_kwargs",
    "profile_grid",             # read by Engine.Model_optimize._profile_kwargs
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


def _auto_scale(value, bound, search_decades=None,
                min_decades=_AUTO_MIN_DECADES):
    """Pick 'log10' or 'lin' for one parameter.

    ``search_decades`` -- the declared search radius, in decades either side of
    x0 -- is consulted first, and ``bounds`` only as a fallback. That ordering
    is the point: the radius states how the author thinks of the parameter,
    multiplicatively or not, while bounds are meant to carry physics. Reading
    the scale off bounds means dropping a physical limit silently reverts the
    fit to linear, which is the one outcome this is here to prevent.

    Fitting a multiplicative parameter linearly is what makes a single absolute
    ``xatol`` incomparable across a vector whose magnitudes differ by decades,
    and what drives the Hessian's dynamic range past what float64 can invert --
    which is why every Wald SE on the larger specs here comes back None.

    Anything that can reach zero or go negative stays linear: log10 has no
    meaning there, and _to_opt_space would raise.
    """
    if value is None or not np.isfinite(value) or value <= 0:
        return "lin"

    if search_decades is not None:
        try:
            d = float(search_decades)
        except (TypeError, ValueError):
            d = None
        if d is not None and np.isfinite(d) and d > 0:
            # A radius of d decades either side spans 2d in total.
            return "log10" if 2.0 * d >= min_decades else "lin"

    if bound is None:
        return "lin"
    lo, hi = bound
    if lo is None or hi is None or lo <= 0 or hi <= lo:
        return "lin"
    return "log10" if np.log10(hi / lo) >= min_decades else "lin"


def _resolve_scales(parameter_scale, param_names, bounds=None, x0=None,
                    search_decades=None):
    """Normalize a ``parameter_scale`` spec to a list of per-parameter scales.

    Accepts None (all linear), a single string applied to every parameter, a
    {name: scale} dict (unlisted names default to "lin"), or an explicit
    per-parameter sequence.

    The string "auto" -- alone, or per parameter in a dict or sequence -- defers
    to ``_auto_scale``, which reads the decision off that parameter's bounds.
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
    bad = [s for s in scales if s not in _SCALE_INPUTS]
    if bad:
        raise ValueError(
            f"Unknown parameter scale(s) {bad}; valid options are {_SCALE_INPUTS}"
        )
    if "auto" in scales:
        vals = list(x0) if x0 is not None else [None] * k
        bnds = list(bounds) if bounds is not None else [None] * k
        if len(vals) != k or len(bnds) != k:
            raise ValueError(
                "parameter_scale='auto' needs x0 and bounds aligned with "
                f"param_names ({k} entries); got {len(vals)} and {len(bnds)}"
            )
        scales = [
            _auto_scale(vals[i], bnds[i], search_decades) if s == "auto" else s
            for i, s in enumerate(scales)
        ]
    return scales


def _multistart_points(x0, bounds, scales, n_starts, search_decades=None,
                       seed=None, verbose=True):
    """Starting points for a multi-start fit, in opt space.

    Point 0 is always the declared x0, so a multi-start run can never come back
    worse than the single fit it replaces.  The rest are a Latin hypercube over
    a radius *around* x0 -- ``search_decades`` in opt units for a log-scaled
    parameter, which makes the perturbation multiplicative and therefore the
    right shape for a rate constant.

    Sampling a radius rather than the whole declared box is the part that pays.
    On the NfL fit, drawing log-uniformly across its (1e-9, 1) bound put three
    of ten starts nine decades below the answer and they converged to a basin
    11 nats worse; the same budget spent within two decades of x0 found the
    global optimum three times in ten.
    """
    x0 = np.asarray(x0, dtype=float)
    k = len(x0)
    try:
        n = max(1, int(n_starts))
    except (TypeError, ValueError):
        n = 1
    if n == 1 or k == 0:
        return [x0.copy()]

    lo = np.empty(k)
    hi = np.empty(k)
    for i in range(k):
        lb, ub = _param_bounds(bounds, i)
        if search_decades and scales[i] == "log10":
            # opt space is log10(p), so a radius in decades is a radius here.
            r = float(search_decades)
        elif search_decades and np.isfinite(x0[i]) and x0[i] != 0.0:
            # A linear parameter has no decades; spread it relatively instead.
            r = abs(x0[i]) * 0.5
        elif np.isfinite(lb) and np.isfinite(ub):
            r = (ub - lb) / 2.0
        else:
            r = max(abs(x0[i]), 1.0) * 0.5
        lo[i] = max(x0[i] - r, lb)
        hi[i] = min(x0[i] + r, ub)
        if not hi[i] > lo[i]:          # bound collapsed the range to a point
            lo[i] = hi[i] = x0[i]

    how = "Latin hypercube"
    try:
        from scipy.stats import qmc
        unit = qmc.LatinHypercube(d=k, seed=seed).random(n - 1)
    except Exception:
        how = "uniform random"
        unit = np.random.default_rng(seed).random((n - 1, k))

    pts = [x0.copy()] + [lo + row * (hi - lo) for row in unit]
    if verbose:
        print(f"[opt] multi-start: {n} starts ({how}); start 1 is the spec's x0")
    return pts


def _run_multistart(objective, starts, method, bounds, opt_kw, scales,
                    screen=True, failure_value=1e10, verbose=True):
    """Fit from every start, keep the best, and report the spread.

    The spread is as much the point as the best value.  If every start lands on
    the same NLL the objective is unimodal in that region and one start will do
    from here on.  If they scatter, the fit is start-dependent, and a profile or
    confidence interval anchored on any single one of them is measuring the
    wrong basin -- which the engine can otherwise only discover much later, via
    ``profile_anchor_gap`` finding a point better than the reported optimum.
    """
    from scipy.optimize import minimize

    starts = [np.asarray(s, dtype=float) for s in starts]

    # Screen with one evaluation each before committing to whole fits. Any
    # sampler will eventually propose a point where the model cannot be
    # integrated, and a Nelder-Mead run started there spends dozens of failed
    # integrations -- each dragging safe_simulate's retry ladder behind it --
    # to learn what a single evaluation already said. The screen still pays for
    # one such evaluation; if that is happening often, the search radius is too
    # wide for the model rather than the screen being at fault.
    #
    # Start 1 is never screened out. It is the spec's own x0, and if that
    # cannot be integrated the caller needs to see the failure, not a silent skip.
    if screen and len(starts) > 1:
        kept, dropped = [], 0
        for i, s in enumerate(starts):
            if i == 0:
                kept.append(s)
                continue
            try:
                v = float(objective(s))
            except Exception:
                v = float("inf")
            if np.isfinite(v) and v < failure_value:
                kept.append(s)
            else:
                dropped += 1
        if dropped and verbose:
            print(f"  [opt] {dropped}/{len(starts) - 1} sampled start(s) dropped: "
                  f"the model does not integrate there.")
        starts = kept

    records, best, best_i = [], None, -1
    for i, s in enumerate(starts):
        s = np.asarray(s, dtype=float)
        try:
            r = minimize(objective, s, method=method,
                         bounds=bounds or None, **opt_kw)
            fun = float(r.fun)
            ok, msg = bool(r.success), str(r.message)
        except Exception as exc:
            r, fun, ok = None, float("inf"), False
            msg = f"{type(exc).__name__}: {exc}"
        records.append({
            "start": _to_linear(s, scales).tolist(),
            "fun": fun if np.isfinite(fun) else None,
            "success": ok,
            "message": msg,
            "nfev": getattr(r, "nfev", None) if r is not None else None,
        })
        if r is not None and np.isfinite(fun) and (best is None or fun < best.fun):
            best, best_i = r, i
        if verbose:
            mark = "  <-- best so far" if i == best_i else ""
            print(f"  [opt] start {i + 1}/{len(starts)}: nll={fun:.6g}{mark}",
                  flush=True)

    if verbose:
        vals = [rec["fun"] for rec in records if rec["fun"] is not None]
        if len(vals) > 1:
            uniq = sorted({round(v, 3) for v in vals})
            print(f"\n[opt] multi-start: best {min(vals):.6g} from start "
                  f"{best_i + 1} of {len(starts)}; {len(uniq)} distinct "
                  f"optimum/optima {[f'{u:.5g}' for u in uniq[:8]]}")
            if len(uniq) > 1:
                print("[opt] this objective is multimodal here: a profile or CI "
                      "anchored on one start would describe whichever basin it "
                      "happened to reach.")
    return best, records, best_i


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
            {"method": "L-BFGS-B", "bounds": bounds or None},
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


def _profile_nuisance_defaults(method, n_nuisance=None):
    """Convergence options appropriate to *method* and to the problem size.

    scipy rejects unknown option keys per method, so "ftol" cannot simply be
    handed to Nelder-Mead.  Gradient-free methods also need a larger iteration
    budget to reach comparable accuracy.

    The budget scales with the number of nuisance parameters, matching scipy's
    own per-method defaults.  A flat cap is the wrong shape: a profile point on
    a 20-parameter fit is a 19-dimensional minimization, and the ``maxiter=200``
    this used to return stopped Nelder-Mead while its simplex was still barely
    contracted -- roughly a tenth of scipy's own ``N*200``.  The recorded
    "minimum over the nuisance parameters" was then too high, which lifts dNLL,
    which drags the 1.9207 crossing inward.  A truncated nuisance optimization
    does not make a profile noisy; it makes every confidence interval too
    narrow, in the same direction, every time.

    The cost of the larger cap falls only where it was doing harm.  A point
    whose nuisance optimization was already converging stops on
    ``fatol``/``xatol`` long before the cap and is unaffected; only the points
    that were being cut off run longer.
    """
    m = (method or "").lower()
    try:
        n = max(1, int(n_nuisance))
    except (TypeError, ValueError):
        n = 1

    if m in ("l-bfgs-b", "tnc", "slsqp"):
        # ftol here is *relative*: at 1e-4 the search stops once a step improves
        # an NLL of order 100 by ~1e-2 nats, which is a substantial fraction of
        # the 1.9207 threshold.  1e-8 puts the stopping criterion well below
        # anything that can move a CI bound.
        return {"maxiter": 500, "ftol": 1e-8}
    if m == "nelder-mead":
        return {"maxiter": 200 * n, "maxfev": 200 * n,
                "fatol": 1e-4, "xatol": 1e-4}
    if m == "powell":
        return {"maxiter": 200 * n, "maxfev": 1000 * n,
                "ftol": 1e-8, "xtol": 1e-6}
    return {"maxiter": 100 * n}


def nuisance_convergence(res):
    """Convergence diagnostics from one nuisance minimization, as plain types.

    scipy reports a hit iteration or evaluation cap as ``success=False``, and
    that is the failure mode that matters here: a truncated minimization returns
    a nuisance minimum that is too high, so its profile point sits above the
    true profile and any CI read through it is too narrow.  Nothing downstream
    could previously tell such a point from a converged one -- both arrived as a
    finite number -- so the bias was invisible.

    Returned as plain ints/bools/strs because these travel to a worker process
    and then into a JSONL checkpoint record.
    """
    def _int(name):
        v = getattr(res, name, None)
        try:
            return int(v)
        except (TypeError, ValueError):
            return -1

    return {
        "converged": bool(getattr(res, "success", True)),
        "nit": _int("nit"),
        "nfev": _int("nfev"),
        "opt_message": str(getattr(res, "message", ""))[:200],
    }


def nuisance_option_budget(method, n_nuisance, optimizer_kwargs=None):
    """The iteration and evaluation caps one nuisance minimization may spend.

    Split out of :func:`_minimize_nuisance` because a point that is stopped on
    the clock and resumed in a later job has to spend the *same total* budget
    across all of its launches. Handing each launch a fresh allowance would
    mean a point resumed often could never exhaust its cap, and so could never
    terminate -- the run would resume it forever, each time from a slightly
    better place, with nothing to say when it should stop.
    """
    options = dict(_profile_nuisance_defaults(method, n_nuisance))
    options.update((optimizer_kwargs or {}).get("options") or {})
    return {k: int(options[k]) for k in ("maxiter", "maxfev") if k in options}


def _minimize_nuisance(fun, x0, args, method, bounds, optimizer_kwargs=None,
                       extra_options=None):
    """Minimize over the nuisance parameters with the caller's chosen method.

    Falls back to Nelder-Mead if *method* cannot handle the problem (for
    example a gradient method that scipy refuses for these bounds), so a
    profile run degrades rather than dying.

    *extra_options* is merged last and is how a resumed point carries its
    optimizer state back in: the remaining slice of its iteration budget, and
    for Nelder-Mead the simplex it had reached when the clock stopped it.
    """
    import scipy.optimize as opt

    n_nuisance = len(np.atleast_1d(x0))
    kwargs = dict(optimizer_kwargs or {})
    options = dict(_profile_nuisance_defaults(method, n_nuisance))
    options.update(kwargs.pop("options", {}) or {})
    options.update(extra_options or {})
    for key in _ENGINE_ONLY_OPTIMIZER_KEYS:
        kwargs.pop(key, None)

    try:
        return opt.minimize(fun, x0, args=args, method=method,
                            bounds=bounds, options=options, **kwargs)
    except (ValueError, TypeError) as exc:
        print(f"    [profile] method {method!r} failed ({exc}); "
              f"retrying with Nelder-Mead.", flush=True)
        fallback = dict(_profile_nuisance_defaults("Nelder-Mead", n_nuisance))
        # The simplex belongs to the method that produced it; carrying it into
        # a different one would be meaningless, and scipy would reject it.
        fallback.update({k: v for k, v in (extra_options or {}).items()
                         if k != "initial_simplex"})
        return opt.minimize(
            fun, x0, args=args, method="Nelder-Mead", bounds=bounds,
            options=fallback,
        )


def _enable_preequil_cache(models, active_replicates, param_names, x0_lin,
                           bounds_lin, enabled=True, verbose=True):
    """Attach a pre-dose cache to each active model, if it is safe to do so.

    The saving is real -- on the twelve-arm microglia group the pre-dose block is
    about 28% of every objective evaluation, and it is identical every time --
    but it is only sound while no fitted parameter can act before the first dose.
    That is checked here rather than assumed, on one representative arm, and a
    failure disables the cache everywhere instead of quietly returning a stale
    state. Runs then continue exactly as they did before, only slower.

    Returns True if the cache was enabled.
    """
    if not enabled or not active_replicates:
        return False

    from Engine.Preequil_cache import PreequilCache, split_preequil_block, verify_invariance

    # Check on an arm that actually has a cacheable block; if none does, there
    # is nothing to enable and nothing to warn about.
    probe_name = None
    for name, rep in active_replicates.items():
        try:
            block, _rest = split_preequil_block(rep["Solver_settings"](rep))
        except Exception:
            continue
        if block is not None and name in models:
            probe_name = name
            break
    if probe_name is None:
        return False

    try:
        ok, _report = verify_invariance(
            models[probe_name]["r"], active_replicates[probe_name], param_names,
            x0_lin, bounds_lin, verbose=verbose,
        )
    except Exception as exc:
        print(f"[preequil] invariance check failed to run ({exc}); "
              f"cache left disabled.")
        return False

    if not ok:
        return False

    n = 0
    for name in active_replicates:
        m = models.get(name)
        if m is None:
            continue
        m["preequil_cache"] = PreequilCache(enabled=True)
        n += 1
    if verbose:
        print(f"[preequil] cache enabled for {n} arm(s); the pre-dose segment "
              f"is integrated once per arm and restored thereafter.")
    return True


def _try_build_evaluator(model_text, paths, models, active_replicates, param_names,
                         scales, optimization_spec, fixed_sigmas, events_dynamic,
                         n_workers, preequil_cache=False):
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
            preequil_cache=preequil_cache,
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
    """Parameter values to sample for one slice, in optimizer space.

    The optimum is always one of the sampled points, so the returned grid holds
    ``n_points`` or ``n_points + 1`` values. An even *n_points* spread
    symmetrically about the optimum straddles it without ever landing on it,
    which is why every slice curve stepped across the optimum instead of through
    it -- and the optimum is the one point on the curve whose height is known in
    advance, so its absence is what a reader notices first.

    It is evaluated like any other point rather than spliced in at 0.0, for the
    same reason the profile stopped splicing: a hardcoded zero would hide an
    evaluator that disagrees with the fit at the optimum, and would give a
    genuinely flat slice a non-zero range it did not earn.
    """
    is_log = scales[param_idx] == "log10"
    p_val = res_x[param_idx]
    if is_log:
        offset = np.log10(range_factor)
        grid = np.linspace(p_val - offset, p_val + offset, n_points)
    else:
        grid = np.linspace(p_val / range_factor, p_val * range_factor, n_points)
    if not np.any(np.isclose(grid, p_val, rtol=1e-12, atol=0.0)):
        grid = np.sort(np.append(grid, p_val))
    return grid, is_log


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

    # Grid lengths vary by one depending on whether the optimum had to be added,
    # so the chunks below are cut by each grid's own length rather than by
    # n_points.
    print(f"\n[slice] {len(param_names)} parameter(s) x ~{n_points} points "
          f"= {len(xs)} evaluations, submitted as one batch")
    vals = nll_batch(xs, label="slice")

    out = {}
    pos = 0
    for i, name in enumerate(param_names):
        chunk = np.asarray(vals[pos:pos + len(grids[i])], dtype=float)
        pos += len(grids[i])
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
    n_grid = len(param_vals)

    if nll_batch is not None:
        print(f"\n[slice] {pname}  ({n_grid} points, x{range_factor} range)")
        nll_vals = nll_batch(xs, label=f"slice:{pname}")
    else:
        width = len(str(n_grid))
        print(f"\n[slice] {pname}  ({n_grid} points, x{range_factor} range)")
        nll_vals = []
        for i, (val, x) in enumerate(zip(param_vals, xs)):
            nll = nll_func(x)
            shown = 10.0 ** val if is_log else val
            print(f"  [{i+1:{width}d}/{n_grid}]  {pname}={shown:.4g}  nll={nll:.6g}")
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
            # A capped nuisance minimization overstates the profile here exactly
            # as it does in the parallel grid, so mark it rather than printing a
            # number that looks like every other one.
            flag = "" if getattr(res, "success", True) else "  [CAPPED]"
            print(f"  [{tag}]  {pname}={_lin(x_target):.4g}  dNLL={nll_rel:.6g}{flag}",
                  flush=True)
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

    # Re-anchor on the profile's own minimum, matching the parallel grid. The
    # 0.0 above is dNLL at res_x by definition; it is the *reference* that can
    # be wrong. With one shared objective a negative minimum means the fit did
    # not converge, so say so rather than plotting a curve that dips below its
    # own optimum line.
    if all_nlls.size:
        gap = float(all_nlls.min())
        if gap < -1e-3:
            print(f"  *** WARNING: {pname} profile reached {abs(gap):.4g} nats below "
                  f"the reported optimum — the fit has not converged. Re-anchoring.",
                  flush=True)
        all_nlls = all_nlls - min(gap, 0.0)

    if is_log:
        all_vals = 10.0 ** all_vals

    return all_vals, all_nlls


def _param_bounds(bounds, param_idx):
    """(lower, upper) for one parameter in opt space, as finite-or-infinite floats."""
    if bounds is not None and param_idx < len(bounds) and bounds[param_idx] is not None:
        lb, ub = bounds[param_idx]
        return (-np.inf if lb is None else float(lb),
                np.inf if ub is None else float(ub))
    return -np.inf, np.inf


def _profile_grid_for(param_idx, res_x, bounds, scales, wald_se, n_grid,
                      range_factor, se_span):
    """Grid of fixed values for one parameter, both directions, in opt space.

    Seeded from the Wald standard error when one is available: the profile
    crossing sits near 1.96 SE, so spanning a few SE puts most points where the
    threshold actually is instead of spreading them over a range_factor window
    that may be far too wide or far too narrow. Falls back to the multiplicative
    range_factor when no SE exists (a flat or confounded direction).

    Either way this grid is only an opening bid. A missing SE means the Hessian
    was singular in this direction, which is precisely when the parameter is
    least likely to be pinned down inside a range_factor of 2 -- so the fallback
    is at its narrowest exactly where width is most needed. Rather than guess a
    wide span here and waste points in the flat middle of every well-determined
    parameter, ``_build_extension_jobs`` walks whichever side has not reached
    the threshold outward from this grid until it does.
    """
    p_opt = res_x[param_idx]
    is_log = scales[param_idx] == "log10"
    lb, ub = _param_bounds(bounds, param_idx)

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


def _at_bound(x, bound, sign):
    """Whether *x* already sits on the bound it is walking towards."""
    if bound is None or not np.isfinite(bound):
        return False
    tol = 1e-12 * max(abs(bound), 1.0)
    return x <= bound + tol if sign < 0 else x >= bound - tol


def _next_extension_value(p_opt, outer_x, lb, ub, sign, is_log, growth):
    """One geometric step further out than *outer_x*, clipped to the bounds.

    Returns None when this side can go no further -- it is already on the bound,
    or the clipped step would not move.

    The step multiplies the *distance* from the optimum, with distance measured
    multiplicatively wherever that is meaningful: a log10-scaled parameter is
    stored as its own logarithm, so doubling the offset there already doubles a
    ratio, and a positive linear parameter is taken into logarithms for the same
    reason. Doubling a linear offset instead fails in the direction that matters
    most. Walking down from 8.5e-5 towards a lower bound of 1e-9, the first
    doubled offset overshoots zero and clips straight onto the bound, collapsing
    five unexplored decades into a single useless bracket; doubling the log
    distance visits p_opt/4, /16, /256, /65536 and reaches the same bound in five
    informative steps.
    """
    bound = lb if sign < 0 else ub
    if _at_bound(outer_x, bound, sign):
        return None

    if not is_log and p_opt > 0 and outer_x > 0:
        u_opt, u_out = np.log10(p_opt), np.log10(outer_x)
        x_next = 10.0 ** (u_opt + growth * (u_out - u_opt))
    else:
        x_next = p_opt + growth * (outer_x - p_opt)

    if not np.isfinite(x_next):
        return None
    if bound is not None and np.isfinite(bound):
        x_next = max(x_next, bound) if sign < 0 else min(x_next, bound)
    # A clipped step that lands on (or inside) the point we came from carries no
    # information, so treat the side as finished rather than re-evaluating it.
    if (x_next >= outer_x) if sign < 0 else (x_next <= outer_x):
        return None
    return float(x_next)


def _ext_distance(p_opt, x, is_log):
    """Distance from the optimum in the same measure the stepping rule uses.

    ``_next_extension_value`` multiplies a *distance* by a growth factor, and
    which distance it means depends on the branch it takes: the opt-space offset
    for a log10-scaled parameter, the log10 ratio for a positive linear one, the
    plain offset otherwise. Anything that predicts a growth factor from measured
    dNLL has to measure distance the same way or the prediction lands somewhere
    else entirely.
    """
    if is_log:
        return abs(x - p_opt)
    if p_opt > 0 and x > 0:
        return abs(np.log10(x) - np.log10(p_opt))
    return abs(x - p_opt)


_EXT_GROWTH_MIN = 1.2
_EXT_GROWTH_MAX = 8.0
_EXT_OVERSHOOT = 1.15


def _extension_growth(side, p_opt, is_log, threshold, anchor, default_growth,
                      min_growth=_EXT_GROWTH_MIN, max_growth=_EXT_GROWTH_MAX,
                      overshoot=_EXT_OVERSHOOT):
    """How far out the next step should reach, read off the curve so far.

    The old rule doubled the distance from the optimum every step, which knows
    nothing about the curve it is walking. Doubling is far too slow on a flat
    side and overshoots a sharp one, and the cost is not symmetric: a side that
    runs out of ``max_extend`` steps before reaching dNLL 1.9207 reports no
    confidence bound at all. Ten of the sixteen sides in the last SILK/APP run
    ended that way, several of them with dNLL still below 0.01, which doubling
    would have needed a dozen more steps to lift.

    Instead, fit the curve that is already there and step to where it says the
    threshold is. Locally dNLL grows as a power of the distance, dNLL = c * d^p,
    with p = 2 wherever the quadratic approximation holds. Two points give both
    c and p, so the step that should land on the threshold is

        growth = (target / dNLL_outer) ** (1 / p)

    Estimating p rather than assuming 2 is what makes this work on the sides
    that need it. A direction that is flatter than quadratic -- the case where
    doubling stalls worst -- has p < 2 and gets a correspondingly longer step.

    Clamped at both ends. The lower clamp keeps a step from collapsing to
    nothing when the fit is poor; the upper one keeps a near-zero dNLL, which is
    as likely to be optimizer noise as signal, from launching a step of 10^6.
    Overshooting the threshold slightly is deliberate: the point of the step is
    to *bracket* the crossing so refinement has something to interpolate inside,
    and a step that lands exactly on it brackets nothing.
    """
    pts = [r for r in side
           if r.get("dnll") is not None and np.isfinite(r.get("dnll"))]
    if not pts:
        return default_growth
    outer = pts[-1]
    d_out = _ext_distance(p_opt, outer["x_fixed"], is_log)
    dnll_out = float(outer["dnll"]) - anchor
    if not (d_out > 0 and np.isfinite(dnll_out) and dnll_out > 0):
        return default_growth

    target = float(threshold) * float(overshoot)
    if dnll_out >= target:
        return default_growth

    exponent = 2.0
    if len(pts) >= 2:
        inner = pts[-2]
        d_in = _ext_distance(p_opt, inner["x_fixed"], is_log)
        dnll_in = float(inner["dnll"]) - anchor
        if (d_in > 0 and dnll_in > 0 and d_out > d_in * (1.0 + 1e-9)
                and dnll_out > dnll_in * (1.0 + 1e-9)):
            with np.errstate(divide="ignore", invalid="ignore"):
                cand = np.log(dnll_out / dnll_in) / np.log(d_out / d_in)
            if np.isfinite(cand) and cand > 0:
                exponent = float(cand)
    exponent = float(np.clip(exponent, 0.5, 6.0))

    growth = (target / dnll_out) ** (1.0 / exponent)
    if not np.isfinite(growth):
        return default_growth
    return float(np.clip(growth, min_growth, max_growth))


def _side_points(completed, name, p_opt, sign):
    """Completed points strictly on one side of the optimum, nearest first."""
    pts = [r for r in completed.get(name, {}).values()
           if (r["x_fixed"] < p_opt if sign < 0 else r["x_fixed"] > p_opt)]
    pts.sort(key=lambda r: abs(r["x_fixed"] - p_opt))
    return pts


def _side_has_crossed(side, threshold, anchor):
    """Whether any point on this side is above the threshold."""
    for r in side:
        d = r.get("dnll")
        if d is not None and np.isfinite(d) and (d - anchor) > threshold:
            return True
    return False


def _build_extension_jobs(param_names, completed, res_x, bounds, meta,
                          nuisance_bounds_for, make_job, threshold, max_extend,
                          growth=2.0, anchor=0.0):
    """One outward step per parameter/side that has not yet reached the threshold.

    This is the pass that answers the question the profile is actually for. The
    opening grid is placed from the Wald SE (or, when the Hessian gave none,
    from ``range_factor``), and neither is a prediction of where dNLL reaches
    1.9207 -- it is a guess. When the guess falls short the old code had no
    recourse: refinement only interpolates *inside* a bracket, so a side with no
    point above the threshold got no further work from any later pass, and the
    run ended reporting an open interval that a few more evaluations would have
    closed. Widening ``profile_se_span`` and re-running was the only remedy, and
    it widens every parameter at once, including the ones already answered.

    A side stops for one of three reasons, and they are not the same answer:
    it brackets the threshold (done -- refinement takes over), it reaches the
    declared parameter bound (done, and genuinely unidentifiable within those
    bounds), or it exhausts ``max_extend`` steps (not done -- the budget ran
    out). ``profile_reach_report`` keeps them distinct so the log can say which.

    Each step is warm-started from the point it steps off -- its solution and
    its simplex both -- which is the same continuation that makes pass 4 work
    and matters more here: these are the outermost, most correlated points on
    the curve, and an extension step is the longest jump on it, so it is the
    step whose simplex most needs to arrive already stretched along the valley.
    """
    jobs = []
    for i, name in enumerate(param_names):
        if not completed.get(name):
            continue
        is_log = meta[i]["is_log"]
        nb = nuisance_bounds_for(i)
        p_opt = res_x[i]
        lb, ub = _param_bounds(bounds, i)
        for sign in (-1, +1):
            side = _side_points(completed, name, p_opt, sign)
            if not side or _side_has_crossed(side, threshold, anchor):
                continue
            # Counted from the stored records, so the budget survives a resume
            # instead of granting a fresh allowance on every launch.
            n_ext = sum(1 for r in side
                        if int(r.get("phase", 1)) == 4
                        and int(r.get("direction", 0)) == sign)
            if n_ext >= max_extend:
                continue
            outer = side[-1]
            step_growth = _extension_growth(side, p_opt, is_log, threshold,
                                            anchor, growth)
            x_next = _next_extension_value(p_opt, outer["x_fixed"], lb, ub,
                                           sign, is_log, step_growth)
            if x_next is None:
                continue
            if ProfileCheckpointKey(x_next) in completed.get(name, {}):
                continue
            jobs.append(make_job(i, x_next, is_log, nb, phase=4, direction=sign,
                                 seed=outer))
    return jobs


def unfinished_profile_points(completed):
    """Stored points the clock stopped partway, which still need work.

    These are not failures and not merely imprecise: an unfinished point sits
    *above* the true profile by an unknown amount, so it is a valid upper bound
    but a misleading input to any decision that reads dNLL.
    """
    return [(name, rec)
            for name, pts in completed.items()
            for rec in pts.values()
            if rec.get("interrupted")]


def _seed_travel(rec):
    """How far the nuisance minimum moved from its own starting point.

    The best available prediction of how far the *next* point's minimum will
    have to move, because consecutive points on a chain are a similar distance
    apart and the nuisance path is smooth. Returns None when the record cannot
    say -- a cold point that never moved, or one missing either vector.
    """
    if not rec:
        return None
    nx, xs = rec.get("nuisance_x"), rec.get("x_start")
    if not nx or not xs:
        return None
    try:
        nx = np.asarray(nx, dtype=float)
        xs = np.asarray(xs, dtype=float)
    except (TypeError, ValueError):
        return None
    if nx.shape != xs.shape:
        return None
    d = float(np.linalg.norm(nx - xs))
    return d if np.isfinite(d) and d > 0 else None


def _finalize_simplex(sim_new, bounds=None):
    """Clip a candidate simplex into its bounds and reject a degenerate one.

    Shared by both constructors below. A rank-deficient simplex is the failure
    they exist to prevent: Nelder-Mead can never search a direction its starting
    simplex has no extent in, so it converges early and reports a nuisance
    minimum that is too high, which lifts dNLL and narrows the interval. Better
    to hand back nothing and let scipy build its own than to hand back a search
    that is blind in some direction.
    """
    sim_new = np.asarray(sim_new, dtype=float)
    n = sim_new.shape[1]
    if bounds is not None:
        lo = np.array([-np.inf if (b is None or b[0] is None) else b[0]
                       for b in bounds], dtype=float)
        hi = np.array([np.inf if (b is None or b[1] is None) else b[1]
                       for b in bounds], dtype=float)
        if lo.size == n and hi.size == n:
            sim_new = np.clip(sim_new, lo, hi)
    if not np.all(np.isfinite(sim_new)):
        return None
    if np.linalg.matrix_rank(sim_new[1:] - sim_new[0]) < n:
        return None
    return sim_new.tolist()


_COLD_SIMPLEX_SE = 1.0
_COLD_SIMPLEX_FALLBACK = 0.2


def _cold_simplex(x_start, se_nuisance=None, bounds=None,
                  mult=_COLD_SIMPLEX_SE, fallback=_COLD_SIMPLEX_FALLBACK):
    """A starting simplex for a point with no neighbour to inherit one from.

    The grid points of pass 1, and the first step of every continuation chain,
    start at the MLE with nothing to carry. Left alone scipy sizes each vertex
    at 5% of that coordinate's own value, which on log10-fitted parameters is
    not a step in the problem at all -- it is a step in the arbitrary offset of
    the log scale. Two parameters of identical physical uncertainty get vertex
    offsets differing by a factor of a thousand purely because one optimum is
    near 1e5 and the other near 1e0.

    The Wald standard error is the right scale instead: it is what the
    likelihood itself says the coordinate's uncertainty is, it is already in opt
    space, and the profile grid is placed from it too, so the simplex and the
    grid end up measured in the same units. Where the Hessian gave no SE for a
    coordinate -- singular in that direction -- *fallback* is used, a fixed step
    in opt space, which for a log10 parameter is a fixed factor.

    Measured on the 15-parameter ill-conditioned quadratic, mean excess over the
    exact profile for cold points, at equal evaluation cost:

        scipy default             4.1963
        diagonal, 1 x Wald SE     0.0024
        diagonal, 2 x Wald SE     0.0004
        uniform 0.05 (no SE)      0.1047
        uniform 0.20 (no SE)      0.0040

    The gain is four orders of magnitude and it is free: the evaluation counts
    differ by under 3%. It is also why the accuracy of a cold point is not a
    budget problem, which is what the old ``maxfev`` comment assumed -- a
    simplex that starts a thousand times too wide does not converge no matter
    how long it runs.

    CHOOSING *mult*. The sweep above walks a single chain with no budget limit,
    where anything from 1 to 8 measures the same. Run end to end against a cap,
    which is the regime a real profile is in, the picture separates. Three seeds
    of the 15-parameter problem at maxfev 400:

        mult    mean error   mean worst-case   worst seen
        0.5       -0.138%        -0.801%         -1.064%
        1.0       -0.136%        -0.736%         -1.034%
        2.0       -0.181%        -0.975%         -1.590%
        4.0       -0.193%        -0.878%         -1.005%

    So 1.0, not the 2.0 this started at. The margin is modest and it vanishes
    at the production budget, where every value reaches -0.00% -- this is
    insurance for a run whose budget is cut, not a headline gain. The direction
    is what matters and it is consistent: too wide costs more than too narrow
    here, because a vertex placed at twice the *marginal* SE is at dNLL 50 to
    126 rather than the 2 that an axis step is meant to cost. The marginal SE
    overstates how far one coordinate can move alone by the variance inflation
    factor, 2.5 to 7.9 across these coordinates.

    Using the conditional standard deviation instead, 1/sqrt(diag(H)), is the
    principled version of that correction and was measured: it is neutral to
    slightly worse at every budget, so it is not worth the plumbing. Narrowing
    the multiplier captures the same effect.
    """
    if x_start is None:
        return None
    try:
        x0 = np.asarray(x_start, dtype=float)
    except (TypeError, ValueError):
        return None
    n = x0.size
    if n == 0 or not np.all(np.isfinite(x0)):
        return None

    steps = np.full(n, float(fallback))
    if se_nuisance is not None:
        se = np.asarray(se_nuisance, dtype=float)
        if se.size == n:
            good = np.isfinite(se) & (se > 0)
            steps[good] = se[good] * float(mult)
    if not np.all(np.isfinite(steps)) or np.any(steps <= 0):
        return None

    return _finalize_simplex(x0 + np.vstack([np.zeros(n), np.diag(steps)]),
                             bounds)


# How far the predicted travel may be stretched or shrunk relative to the
# distance the seed itself moved. A guard against a wild sensitivity estimate,
# not a working limit: a legitimate extension step is a factor of a few.
_SEED_PREDICT_MAX = 50.0
_SEED_PREDICT_MIN = 0.1


def _seed_sensitivity(rec):
    """Nuisance distance travelled per unit of movement in the fixed parameter.

    A profile chain's nuisance minimum traces a smooth path, so the distance the
    minimum moves is proportional to the distance the *fixed* parameter moved to
    get there. Recording that ratio is what lets the next point size its simplex
    for the step it is about to take rather than the step before it.

    Returns None when the record cannot say -- an old checkpoint with no
    ``x_step``, or a point that never moved.
    """
    travel = _seed_travel(rec)
    if travel is None:
        return None
    try:
        step = float(rec.get("x_step"))
    except (TypeError, ValueError):
        return None
    if not np.isfinite(step) or step <= 0:
        return None
    return travel / step


def _predicted_travel(seed, x_step):
    """How far this point's nuisance minimum should have to move.

    The seed's own travel is the obvious estimate and it is what this used to
    use, but it is only right when consecutive steps are the same size. They are
    not. The extension pass deliberately lengthens its stride as it hunts for
    the threshold, so a chain can walk 8.75, 10, then jump to 25: the step
    quadruples and the distance the nuisance partner must follow quadruples with
    it, while the simplex, sized from the previous step, arrives six times too
    small. Measured on an exactly confounded pair in fourteen nuisance
    dimensions, that one undersized point was the whole failure -- it came back
    at dNLL 13.2 where its own extension-pass evaluation had reached 0.14, and
    it was enough to hand the parameter a confidence bound it does not have.

    So the seed's travel is converted to a rate and re-applied to the step
    actually being taken. Clamped at both ends, because a sensitivity estimated
    from a single pair of points can be wrong and the failure is one-sided: too
    small a simplex stops the search before it arrives and reports a nuisance
    minimum that is too high, which is the error that invents intervals.
    """
    travel = _seed_travel(seed)
    if travel is None:
        return None
    sens = _seed_sensitivity(seed)
    if sens is None or x_step is None:
        return travel
    try:
        step = float(x_step)
    except (TypeError, ValueError):
        return travel
    if not np.isfinite(step) or step <= 0:
        return travel
    predicted = sens * step
    if not np.isfinite(predicted) or predicted <= 0:
        return travel
    return float(np.clip(predicted, travel * _SEED_PREDICT_MIN,
                         travel * _SEED_PREDICT_MAX))


_WARM_SIMPLEX_SCALE = 2.0
_WARM_SIMPLEX_COND_FLOOR = 0.10


def _warm_simplex(simplex, x_start, travel=None, bounds=None,
                  scale=_WARM_SIMPLEX_SCALE,
                  cond_floor=_WARM_SIMPLEX_COND_FLOOR):
    """A starting simplex for a warm point, built from its neighbour's.

    Carrying the *point* forward and letting scipy build the simplex is what
    made warm starts underperform. scipy offsets each vertex by 5% of that
    coordinate's own value, and these parameters are fitted in log10, so the
    offset is set by where the log scale happens to put zero rather than by
    anything about the problem: a parameter at log10 = 5.43 gets a vertex 0.27
    away, a factor of 1.9 in the linear parameter, while one at log10 = 0.01
    gets 0.0005. In fifteen dimensions the optimizer then spends its whole
    budget contracting that back down and never reaches its tolerance, which is
    what the last SILK/APP run showed: 118 points, every one of them stopped by
    the evaluation cap rather than by convergence.

    So the neighbour's simplex is reused -- but only after two corrections,
    because reusing it as it stands is *worse* than letting scipy guess.

    SCALE is replaced, not inherited. A converged simplex is the size of the
    convergence tolerance, which is far smaller than the distance to the next
    point's minimum: measured on the Aducanumab chains the converged diameter
    sits near 1e-4 while the step to the next minimum runs from 1e-5 to 9e-2.
    It is therefore rescaled to ``scale`` times *travel*, the distance the seed
    point itself moved, which is the best available estimate of how far the next
    one must move. This is where the accuracy comes from, and it is insensitive
    to the multiplier: 1.5, 2 and 3 measure the same.

    SHAPE is inherited, but only after its conditioning is repaired. A converged
    Nelder-Mead simplex in a correlated valley is a needle. Measured on the
    engine's own 14-nuisance quadratic, its singular values span
    1.1e-2 down to 4.2e-8 -- a condition number of 2.7e5, with several axes
    collapsed to nothing. Restarted on that shape the optimizer cannot search
    the collapsed directions at all, so it converges quickly and reports a
    nuisance minimum that is far too high. That is not a small effect: carried
    raw, the mean excess over the exact profile was 21x *worse* than not
    carrying the simplex at all, while spending 57% fewer evaluations to get
    there. Flooring the singular values at ``cond_floor`` of the largest
    restores searchability in those directions while keeping the orientation
    that makes the rest of the step cheap.

    Note this is a high-dimension failure. At the two nuisance dimensions of the
    Aducanumab fit the same converged simplexes have a median condition number
    of 3.3, which is why the resume path has carried simplexes for that long
    without anyone seeing this.

    Measured over three seeds of the 15-parameter ill-conditioned quadratic,
    against the analytic profile:

        strategy                       excess NLL   evaluations
        simplex not carried              0.0764        322k
        carried raw, no floor            1.6016        138k
        rescaled, cond_floor 0.10        0.0197        205k

    Accuracy is flat for a floor anywhere from 0.30 to 0.08 and degrades below
    0.04, so the default sits mid-band with an order of magnitude of margin
    above the cliff at zero.

    Returns None when there is nothing usable to carry, in which case the caller
    leaves the simplex out of the job and scipy builds its own.
    """
    if simplex is None or x_start is None:
        return None
    try:
        sim = np.asarray(simplex, dtype=float)
        x0 = np.asarray(x_start, dtype=float)
    except (TypeError, ValueError):
        return None
    n = x0.size
    if n == 0 or sim.ndim != 2 or sim.shape != (n + 1, n):
        return None
    if not (np.all(np.isfinite(sim)) and np.all(np.isfinite(x0))):
        return None

    # Vertex 0 of a scipy final_simplex is the best one, so offsets are measured
    # from the point the seed actually converged to.
    offsets = sim[1:] - sim[0]
    try:
        U, S, Vt = np.linalg.svd(offsets)
    except np.linalg.LinAlgError:
        return None
    if not np.isfinite(S[0]) or S[0] <= 0:
        return None

    # Size is measured as the widest principal axis rather than the longest
    # vertex offset. For a needle those differ by a lot, and the principal axis
    # is the one that says how far the search can actually reach.
    #
    # With no travel to go on, the size is left exactly as it was found: the
    # normalization below is then the identity, so such a call repairs the
    # conditioning and changes nothing else.
    size = float(S[0])
    if travel is not None and np.isfinite(travel) and travel > 0:
        size = float(scale) * float(travel)

    S = np.maximum(S, S[0] * float(cond_floor))
    offsets = U @ np.diag(S / S[0] * size) @ Vt

    return _finalize_simplex(x0 + np.vstack([np.zeros(n), offsets]), bounds)


def _carry_resume_state(keep, res):
    """Move a resumed point's bookkeeping onto whichever record we are keeping.

    Needed because the value and the state can come from different places. If a
    resumed slice did not improve on the stored value, the stored value is what
    we keep -- but the *spend counters and the simplex* still have to come from
    the slice that just ran, or the point would be resumed forever with its
    allowance never advancing and its optimizer state never moving.
    """
    for field in ("interrupted", "converged", "nfev_total", "nit_total",
                  "nm_simplex", "opt_message"):
        if field in res:
            keep[field] = res[field]
    return keep


def _build_resume_jobs(param_names, completed, meta, nuisance_bounds_for,
                       make_job, spend_cap=None):
    """One job per point the clock stopped, continuing from where it stopped.

    Ordered outward from the optimum only for the log's benefit; these jobs are
    mutually independent and go out in a single batch like any other pass.

    *spend_cap* is the total evaluations a point may ever have, across every
    job and every launch. A point that has reached it is not re-queued: the
    worker would decline the work anyway, but the loop that calls this must be
    able to see that it is finished with a point without dispatching a job to
    find out. Termination of the whole run rests on this -- without it, a point
    that reports itself unfinished for ever is resumed for ever.
    """
    jobs = []
    for i, name in enumerate(param_names):
        nb = nuisance_bounds_for(i)
        is_log = meta[i]["is_log"]
        for rec in sorted(completed.get(name, {}).values(),
                          key=lambda r: abs(float(r["x_fixed"]))):
            if not rec.get("interrupted"):
                continue
            if spend_cap is not None and int(
                    rec.get("nfev_total") or 0) >= spend_cap:
                continue
            jobs.append(make_job(
                i, rec["x_fixed"], is_log, nb,
                phase=int(rec.get("phase", 1)),
                direction=int(rec.get("direction", 0)),
                x_start=rec.get("nuisance_x"),
                resume_from=rec,
            ))
    return jobs


def run_parallel_profile(
    nll_batch_profile, res_x, nll_at_optimum, param_names, bounds, scales,
    method="Nelder-Mead", optimizer_kwargs=None, wald_se=None,
    n_grid=5, range_factor=2.0, se_span=4.0, n_refine=4,
    checkpoint=None, threshold=_PROFILE_THRESHOLD, warm_passes=1,
    max_extend=8, extend_growth=2.0, bracket_rtol=0.05,
):
    """Profile likelihood for every parameter as parallel batches.

    Five passes:
      1. coarse grid, cold-started from the optimum, fully parallel;
      2. outward extension: any side whose points all sit below the threshold
         steps further out, multiplying its distance from the optimum, until it
         brackets the threshold, reaches the parameter's own bound, or spends
         ``max_extend`` steps;
      3. refinement only in the bracket that straddles the threshold, probing
         the predicted crossing via sqrt(dNLL) interpolation (which is linear in
         distance for a locally quadratic NLL) rather than bisecting blindly,
         and stopping once the crossing is located to ``bracket_rtol``;
      4. warm-started continuation over every point, which is where the accuracy
         actually comes from (see below);
      5. extension and refinement once more, because pass 4 lowers the curve: it
         can move the crossing outside the bracket pass 3 found, and can even
         drop a side back below the threshold it had reached.

    Pass 2 is what makes the run answer its own question. Passes 1 and 3 place
    points from the Wald SE, or from ``range_factor`` when the Hessian gave no
    SE, and neither is a prediction of where dNLL reaches the threshold. When
    that opening guess falls short, refinement cannot help -- it interpolates
    inside a bracket and there is no bracket -- so before pass 2 existed a side
    that ended below 1.9207 simply stayed there, and the run reported an open
    interval that a handful of further evaluations would have closed. The
    fallback was worst exactly where it was needed most: no Wald SE means the
    Hessian was singular in that direction, which is the least likely direction
    to be pinned down inside a factor of two.

    Passes 1 and 3 are wide and cold: every point starts its nuisance
    minimization at the MLE, so all ``2k * n_grid`` of them are independent and
    go out at once. That buys width at a real cost in accuracy. The nuisance
    minimum at a fixed value ``v`` sits at the end of a curved path through the
    nuisance space, and a cold optimizer has to travel that whole path from a
    standing start before it can refine; in a correlated likelihood it stalls
    partway and reports a value that is too *high*. Too high is the damaging
    direction: an inflated dNLL makes the curve appear to reach the threshold
    sooner than it does, so every interval comes out too narrow.

    Pass 3 fixes that by continuation. Points are ordered outward from the
    optimum and each one starts from its inner neighbour's solution, so the
    optimizer travels only the increment along the path rather than the whole of
    it. On a 15-parameter ill-conditioned quadratic this moves the mean CI error
    from about -34% to about -6% at essentially the same number of function
    evaluations -- accuracy that no affordable increase in the cold-start budget
    reaches, because the problem is where the search begins, not how long it
    runs.

    The dependency in pass 4 runs only *along* a chain (one parameter, one
    direction); the ``2k`` chains are mutually independent. So the chains are
    advanced in lockstep -- batch *j* holds the *j*-th step of every chain --
    which keeps each batch ``2k`` wide and needs no change to the worker or the
    pool.

    Returns ``({param_name: (param_vals_linear, dnll)}, anchor, where,
    diagnostics)``; the traces match the sequential walkers so plotting and CI
    extraction are unchanged.
    """
    from Engine.Deadline import DeadlineReached

    n_params = len(param_names)
    completed = checkpoint.load(param_names) if checkpoint is not None else \
        {n: {} for n in param_names}

    def _lin(v, is_log):
        return 10.0 ** v if is_log else v

    def make_job(i, x_fixed, is_log, nb, phase=1, direction=0, x_start=None,
                 resume_from=None, seed=None):
        # x_start defaults to the MLE (a cold start). Every continuation pass
        # overrides it with a neighbour's nuisance solution, which is the whole
        # mechanism by which continuation is cheaper than starting over.
        #
        # *seed* is that neighbour's whole record rather than just its solution
        # vector. The vector alone was what the passes used to hand over, and it
        # left the most valuable half of the optimizer's state behind: the
        # simplex. See _warm_simplex for why carrying the point without the
        # simplex recovers much less than it looks like it should.
        if seed is not None and x_start is None:
            x_start = seed.get("nuisance_x")
        # Distance in the fixed parameter from the point this job is seeded
        # from. For a cold job that is the optimum itself, which is genuinely
        # where its nuisance start comes from, so the rate is well defined for
        # every point on the curve.
        seed_x = res_x[i] if seed is None else seed.get("x_fixed", res_x[i])
        try:
            x_step = abs(float(x_fixed) - float(seed_x))
        except (TypeError, ValueError):
            x_step = None
        start = np.delete(res_x, i) if x_start is None else np.asarray(
            x_start, dtype=float)
        job = {
            "param_idx": i,
            "param_name": param_names[i],
            "x_fixed": float(x_fixed),
            "x_fixed_linear": float(_lin(x_fixed, is_log)),
            "x_start": np.asarray(start, dtype=float).tolist(),
            # Whether this point was born warm. A refinement probe seeded from
            # its bracket's inner point already has what the continuation pass
            # would give it, so it must not force a fresh sweep on every resume.
            "warm_seeded": x_start is not None,
            "nuisance_bounds": nb,
            "method": method,
            "optimizer_kwargs": optimizer_kwargs,
            # phase/direction are recorded so a resume can tell how much
            # refinement a side already has and stop, instead of adding another
            # probe on every launch.
            "phase": phase,
            "direction": direction,
            # Carried into the record so the *next* point on this chain can turn
            # this one's travel into a rate. See _predicted_travel.
            "x_step": x_step,
        }
        # Every job carries an explicit starting simplex, from its neighbour
        # where there is one and from the Wald SE otherwise. Leaving it to
        # scipy is what the two constructors exist to avoid: its 5%-of-value
        # rule sizes the search from where the log scale puts zero rather than
        # from the problem. Only a truly unusable simplex -- degenerate, or
        # flattened onto a bound -- falls through to scipy's own.
        warm_sim = None
        if seed is not None:
            warm_sim = _warm_simplex(seed.get("nm_simplex"), start,
                                     travel=_predicted_travel(seed, x_step),
                                     bounds=nb)
        if warm_sim is None:
            warm_sim = _cold_simplex(start, nuisance_se_for(i), bounds=nb)
        if warm_sim is not None:
            job["initial_simplex"] = warm_sim
        if resume_from is not None:
            # Continuing a point the clock stopped. The simplex is the real
            # payload: without it the next slice rebuilds from one vertex and
            # spends itself relearning what this record already knows. The
            # spend counters come along so the point's total allowance is
            # shared across launches rather than reset by each one.
            job.update({
                "nfev_used": int(resume_from.get("nfev_total") or 0),
                "nit_used": int(resume_from.get("nit_total") or 0),
                "initial_simplex": resume_from.get("nm_simplex"),
                # The value this point has already reached. Needed for the case
                # where the allowance turns out to be spent and no slice runs
                # at all: without it the job would report an infinite NLL,
                # which is a sentinel, which is never checkpointed -- so the
                # stored record would keep its interrupted flag and the point
                # would be resumed doing nothing on every link for ever.
                "nll_so_far": resume_from.get("nll"),
                # An interrupted grid point is still a cold point that pass 4
                # owes a warm sweep; finishing it must not silently consume
                # that debt.
                "warm_refined": int(resume_from.get("warm_refined") or 0),
                "warm_seeded": bool(resume_from.get("warm_seeded")),
                "resumed": True,
            })
        return job

    def nuisance_bounds_for(i):
        if bounds is None:
            return None
        return [list(b) if b is not None else None
                for b in (list(bounds[:i]) + list(bounds[i + 1:]))]

    def nuisance_se_for(i):
        """Wald SE of the nuisance parameters, in opt space, for parameter *i*.

        ``wald_se`` arrives in the optimizer's own space (the caller passes
        ``wald_se_opt``), which is the space the simplex is built in, so no
        transform is needed here. None when the Hessian gave nothing.
        """
        if wald_se is None:
            return None
        try:
            se = np.atleast_1d(np.asarray(wald_se, dtype=float))
        except (TypeError, ValueError):
            return None
        if se.size != n_params:
            return None
        return np.delete(se, i)

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
        # A warm-seeded probe counts as already continued; a cold grid point
        # does not, and pass 4 will pick it up.
        res.setdefault("warm_refined", 1 if res.get("warm_seeded") else 0)
        name = res["param_name"]
        key = ProfileCheckpointKey(res["x_fixed"])
        prev = completed.get(name, {}).get(key)
        keep = res
        # Two evaluations at the same fixed value are both upper bounds on the
        # profile, so the lower one is the better estimate. Nothing in passes 1,
        # 2 or 4 re-visits a point, but keeping the rule here means the store
        # can never be walked upward by a future caller that does.
        if prev is not None and not _profile_record_is_better(res, prev):
            # A resumed point is the one case where a non-improving result must
            # still be written: it carries how much of the point's allowance
            # has now been spent and whether the clock stopped it again.
            # Dropping it would resume the point forever against a budget that
            # never moves.
            if not (prev.get("interrupted") and res.get("status") == "ok"):
                return
            keep = _carry_resume_state(dict(prev), res)
        if checkpoint is not None and keep.get("status") == "ok":
            checkpoint.append(keep)
        completed.setdefault(name, {})[key] = keep

    # ── Extension rounds (used for pass 2 and again for pass 5) ───────────
    # One step per side per round, so each round is a single batch up to 2k
    # wide and the pool stays busy; rounds must be sequential because each step
    # begins where the previous one ended, both in position and in its warm
    # start. Sides drop out as they bracket, so a well-determined fit runs no
    # rounds at all and this pass costs nothing.
    #
    # *cap* is the total phase-4 steps allowed per side across all launches,
    # counted from the stored records so a resume continues the walk instead of
    # restarting it. Pass 5 raises it because pass 4 can lower a side back below
    # a threshold it had already reached.
    def extend_rounds(n_rounds, cap, tag, label):
        for _round in range(max(int(n_rounds), 0)):
            anchor_r, _where_r = profile_anchor_gap(completed)
            ext_jobs = _build_extension_jobs(
                param_names, completed, res_x, bounds, meta,
                nuisance_bounds_for, make_job, threshold, cap,
                growth=extend_growth, anchor=anchor_r,
            )
            if not ext_jobs:
                break
            print(f"\n[profile] {tag} (round {_round + 1}): stepping "
                  f"{len(ext_jobs)} side(s) further out to reach the threshold")
            nll_batch_profile(ext_jobs, on_result=record,
                              label=f"{label}{_round + 1}")

    # ── Refinement rounds (used for pass 3 and again for pass 5) ──────────
    # Each round is a single wide batch across every parameter and direction
    # that still needs work. Rounds are sequential because each probe depends on
    # the previous bracket, but every round is still ~2k wide, so the pool stays
    # busy. Doing all rounds here (rather than one per launch) means a completed
    # run resumes as a genuine no-op.
    #
    # *cap* is the total phase-2 probes allowed per side across all launches.
    # Pass 5 raises it so that lowering the curve in pass 4 can buy fresh probes
    # without the counter from pass 3 blocking them.
    #
    # ``n_refine`` can afford to be generous now that ``bracket_rtol`` gates
    # each probe: a side whose crossing is already located stops immediately and
    # spends nothing, so the allowance is only ever drawn on by the wide
    # brackets that need it -- typically the ones pass 2 opened up by stepping
    # several decades in a single jump. Before the gate existed, every extra
    # round was spent unconditionally, on sides that had nothing left to learn.
    def refine_rounds(n_rounds, cap, tag, label):
        for _round in range(max(int(n_rounds), 0)):
            # Re-anchor before bracketing. The threshold is a height above the
            # profile's own minimum, so the stepping rule and the CI reader must
            # measure from the same place -- when they did not, a -1.38 offset
            # made the walker bracket ~1.38 nats too early on one side and never
            # bracket at all on the other, which is how upper bounds came back
            # as None while the plotted curve had clearly crossed.
            anchor_r, _where_r = profile_anchor_gap(completed)
            refine_jobs = _build_refinement_jobs(
                param_names, completed, res_x, meta, nuisance_bounds_for,
                make_job, threshold, cap, anchor=anchor_r,
                bracket_rtol=bracket_rtol,
            )
            if not refine_jobs:
                break
            print(f"\n[profile] {tag} (round {_round + 1}): refining "
                  f"{len(refine_jobs)} threshold crossing(s)")
            nll_batch_profile(refine_jobs, on_result=record,
                              label=f"{label}{_round + 1}")

    # ── Passes 1-5 ────────────────────────────────────────────────────────
    # Run as one guarded unit because they share a single interruption: the
    # wall clock. Every point that landed is already in the store and in
    # ``completed`` by the time DeadlineReached unwinds, so being cut short
    # loses only the points that never started -- and the next launch rebuilds
    # this same frontier from the store and carries on from it. The reporting
    # below then runs either way, on whatever this launch did manage to reach.
    stopped_early = None
    warm_stats = {"n_attempted": 0, "n_improved": 0, "nats_recovered": 0.0,
                  "passes": 0}
    # ── Pass 0: drive unfinished points to completion ─────────────────────
    # Runs before any new grid point, and again after them. A point stopped
    # part-way is sunk cost that is worth nothing until it is finished, and its
    # value sits above the true profile by an unknown amount, so every pass
    # that reads dNLL is reading a wrong number until it is done. Depth before
    # breadth: finish what is started.
    #
    # This is a loop rather than a single batch because a job is capped at a
    # slice of wall clock -- so that a preemption costs one slice rather than
    # everything since the point began -- and one slice rarely finishes a
    # point. Each turn of the loop is one more slice for every point that still
    # needs one, checkpointed as it lands. The loop ends when every point is
    # finished, or when the batch runs out of clock and raises.
    def _drain_signature():
        """What is unfinished, and how much each has spent.

        The loop's progress measure. Normally the wall clock ends the loop by
        raising out of the batch, but that only happens when there *is* a
        clock: an unfinished point with no deadline -- a stub, a misreporting
        worker -- would otherwise spin here for ever, doing no work and
        producing no output.
        """
        return {(name, ProfileCheckpointKey(rec["x_fixed"])):
                int(rec.get("nfev_total") or 0)
                for name, rec in unfinished_profile_points(completed)}

    # The same total allowance the worker enforces, computed here so the loop
    # below can stop re-queueing a spent point instead of dispatching a job to
    # be told it is spent.
    spend_cap = nuisance_option_budget(
        method, max(1, n_params - 1), optimizer_kwargs).get("maxfev")

    def drain_unfinished():
        rounds = 0
        while True:
            resume_jobs = _build_resume_jobs(param_names, completed, meta,
                                             nuisance_bounds_for, make_job,
                                             spend_cap=spend_cap)
            if not resume_jobs:
                return
            before = _drain_signature()
            rounds += 1
            print(f"\n[profile] pass 0 (round {rounds}): continuing "
                  f"{len(resume_jobs)} unfinished point(s)")
            nll_batch_profile(resume_jobs, on_result=record,
                              label=f"profile-resume{rounds}")
            if _drain_signature() == before:
                # A round that changed nothing will change nothing next time.
                print(f"[profile] pass 0: a round of {len(resume_jobs)} "
                      f"point(s) made no progress and spent nothing; stopping "
                      f"rather than repeating it.")
                return

    try:
        drain_unfinished()

        # ── Pass 1: the coarse grid assembled above ───────────────────────
        if jobs:
            nll_batch_profile(jobs, on_result=record, label="profile-pass1")
            # Grid points are capped by the same slice, so most of them come
            # back unfinished on a long-point model. Carry them on rather than
            # leaving the rest of the link idle.
            drain_unfinished()

        # Passes 2-5 all decide *where to spend the next point* by reading
        # dNLL off the points already stored. While any point is unfinished its
        # dNLL is too high, and too high is the damaging direction: a side
        # looks like it has crossed the threshold when it has not, extension
        # stops early, refinement brackets in the wrong place, and every
        # interval comes out too narrow. Holding these passes costs a relaunch;
        # running them on provisional numbers costs the answer.
        unfinished = unfinished_profile_points(completed)
        if unfinished:
            print(f"\n[profile] holding passes 2-5: {len(unfinished)} point(s) "
                  f"are still unfinished, and their dNLL is an upper bound "
                  f"rather than a value.")
            print(f"    Extension and refinement would read those numbers to "
                  f"decide where to step next and would step the wrong way. "
                  f"Relaunch to keep going.")
        else:
            # ── Pass 2: walk sides that fell short of the threshold outward ─
            extend_rounds(max_extend, max_extend, "pass 2", "profile-extend")

            # ── Pass 3: refine inside whatever bracket now exists ─────────
            refine_rounds(n_refine, n_refine, "pass 3", "profile-refine")

            # ── Pass 4: warm-started continuation ─────────────────────────
            anchor_w, _where_w = profile_anchor_gap(completed)
            warm_stats = _run_warm_passes(
                nll_batch_profile, completed, param_names, res_x, meta,
                make_job, nuisance_bounds_for, nll_at_optimum, checkpoint,
                warm_passes, threshold=threshold, anchor=anchor_w,
            )

            # ── Pass 5: re-extend and re-bracket, pass 4 moved the curve ──
            if warm_stats.get("n_improved"):
                extend_rounds(max_extend, 2 * max_extend, "pass 5",
                              "profile-reextend")
                refine_rounds(n_refine, 2 * n_refine, "pass 5",
                              "profile-rebracket")

            # Extension, refinement and warm points are capped by the same
            # slice as everything else, so finish any they left part-way
            # rather than reporting a curve built partly from upper bounds.
            drain_unfinished()
    except DeadlineReached as exc:
        stopped_early = exc
        print(f"\n[profile] stopping early: {exc}.")
        print(f"    This is the expected way a link ends on a preemptible "
              f"queue, not a failure. Every point computed is checkpointed; "
              f"relaunch the same command to continue from here.")

    anchor, where = profile_anchor_gap(completed)
    if anchor < -1e-3:
        pname, xval, _nx = where if where else ("?", float("nan"), None)
        print()
        print(f"*** WARNING: the profile found a point {abs(anchor):.4g} nats better "
              f"than the reported optimum ({pname} = {xval:.6g}).")
        print(f"    With a single shared objective this means the FIT did not "
              f"converge — not that two objectives disagree.")
        print(f"    Traces below are re-anchored on that minimum so the 1.9207 "
              f"threshold stays meaningful, but the reported parameters are not "
              f"the MLE. Refit from the improving point before quoting them.")
        print()

    convergence = profile_convergence_report(param_names, completed)
    convergence["warm"] = warm_stats
    convergence["reach"] = profile_reach_report(
        param_names, completed, res_x, bounds, threshold, anchor=anchor)
    # Whether the traces below are the finished profile or a snapshot of one
    # still being built. Every consumer that quotes an interval needs to be
    # able to tell the difference: a side that has not reached the threshold
    # because the clock ran out looks identical to one that is genuinely
    # unbounded, and only this flag separates them.
    still_unfinished = unfinished_profile_points(completed)
    convergence["n_unfinished"] = len(still_unfinished)
    convergence["n_not_started"] = (
        stopped_early.n_remaining if stopped_early is not None else 0)
    convergence["incomplete"] = bool(still_unfinished) or stopped_early is not None
    # Rides into opt["stats"]["profile_convergence"] and from there into the
    # results snapshot, so the restart point is recorded rather than only
    # printed once and lost with the log.
    convergence["better_point"] = _better_point_record(
        param_names, res_x, meta, anchor, where, nll_at_optimum)
    _print_profile_convergence(convergence)
    _print_profile_reach(convergence["reach"])
    if convergence["incomplete"]:
        parts = []
        if still_unfinished:
            parts.append(f"{len(still_unfinished)} point(s) mid-optimization")
        if convergence["n_not_started"]:
            parts.append(f"{convergence['n_not_started']} point(s) not started")
        print(f"\n[profile] INCOMPLETE: {' and '.join(parts)}. Every interval "
              f"below is provisional and, where a side is still unfinished, "
              f"too narrow rather than too wide. Relaunch the same command to "
              f"continue; nothing computed is repeated.")
    else:
        # Deliberately greppable, and the counterpart of the INCOMPLETE line:
        # a chain of links needs one grep to answer "is it done yet".
        print(f"\n[profile] COMPLETE: every point converged or spent its "
              f"allowance; relaunching is a no-op.")

    traces = _assemble_profile_traces(param_names, completed, res_x, meta,
                                      anchor=anchor)
    return traces, anchor, where, convergence


def _nll_gain(prev, res):
    """How many nats *res* recovered relative to *prev*, or 0 if not comparable."""
    try:
        gain = float(prev.get("nll")) - float(res.get("nll"))
    except (TypeError, ValueError):
        return 0.0
    return gain if np.isfinite(gain) else 0.0


def _warm_chain(completed, name, p_opt, sign):
    """Points on one side of the optimum, ordered outward from it.

    Ordering by distance from the optimum is what makes continuation possible:
    consecutive points are close together, so each one's nuisance minimum is a
    short step from the previous one's. Refinement probes sit between grid
    points and extension steps sit beyond them; both are picked up here in their
    natural place, so the chain stays a single ordered walk outward.
    """
    return _side_points(completed, name, p_opt, sign)


_WARM_KEEP_FACTOR = 10.0


def _warm_chain_relevant(pts, threshold, anchor, keep_factor=_WARM_KEEP_FACTOR):
    """Trim a chain to the points that can still move the confidence bound.

    ``_extract_profile_ci`` reads the interval at the *first* point on a side
    whose dNLL exceeds the threshold. Everything beyond that sits in the tail
    and cannot change the answer: lowering a point from dNLL 213 to dNLL 180,
    which is the kind of number the outer extension steps reach, moves no bound
    and closes no interval. Re-running those costs exactly as much as re-running
    the points at the crossing, because the budget is per point.

    So the chain is cut at the first point that is both past an established
    crossing and above ``keep_factor`` times the threshold. The margin is
    deliberately generous -- ten times the threshold, so a point has to be more
    than 17 nats clear of the crossing to be dropped -- because the warm pass
    exists to lower points and a cut that could strand the crossing outside the
    refined region would defeat it. For scale, the largest total the warm pass
    has ever recovered across a whole 15-parameter run is 785 nats over 175
    points, and that was from cold starts with no explicit simplex.

    The cut is a prefix, not a filter: continuation needs a contiguous walk, so
    a point is never dropped from the middle of a chain.
    """
    limit = float(keep_factor) * float(threshold)
    crossed = False
    for idx, rec in enumerate(pts):
        d = rec.get("dnll")
        if d is None or not np.isfinite(d):
            continue
        d = float(d) - anchor
        if crossed and d > limit:
            return pts[:idx]
        if d > threshold:
            crossed = True
    return pts


def _run_warm_passes(nll_batch_profile, completed, param_names, res_x, meta,
                     make_job, nuisance_bounds_for, nll_at_optimum, checkpoint,
                     warm_passes, threshold=_PROFILE_THRESHOLD, anchor=0.0):
    """Re-run every profile point warm-started from its inner neighbour.

    This is where the profile's accuracy comes from. A cold nuisance
    minimization has to travel the whole curved path from the MLE to the
    nuisance minimum at the fixed value; a warm one travels only the gap to its
    neighbour. Same budget, a fraction of the distance.

    Each step inherits its neighbour's simplex as well as its solution. Without
    that the start point is warm but the search around it is not: scipy rebuilds
    an axis-aligned simplex scaled to 5% of each coordinate's value, which
    discards the valley orientation the neighbour had just finished learning and
    is, on these log10-fitted parameters, one to three orders of magnitude wider
    than the step being taken. See _warm_simplex.

    Safe by construction: both the cold and the warm evaluation are evaluations
    of the same function at the same fixed value, so both are upper bounds on
    the true profile and keeping the lower one can only lower the curve. A lower
    curve can only push the threshold crossing outward, so an interval can widen
    but never narrow. There is no configuration in which this pass makes the
    answer worse -- if it changes nothing, the cold starts were good enough.

    Chains (one parameter, one direction) are mutually independent and are
    advanced in lockstep, so each batch is ``2k`` wide and the existing pool and
    worker are used unchanged.
    """
    stats_out = {"n_attempted": 0, "n_improved": 0, "nats_recovered": 0.0,
                 "passes": 0}
    if warm_passes is None or int(warm_passes) <= 0:
        return stats_out

    for sweep in range(int(warm_passes)):
        chains = {}
        for i, name in enumerate(param_names):
            for sign in (-1, +1):
                pts = _warm_chain_relevant(
                    _warm_chain(completed, name, res_x[i], sign),
                    threshold, anchor)
                # Resume guard: a point already warm-refined in this sweep must
                # not be redone, or every relaunch would repeat the whole pass.
                if not pts or all(int(r.get("warm_refined", 0)) > sweep
                                  for r in pts):
                    continue
                chains[(i, sign)] = pts
        if not chains:
            break

        stats_out["passes"] += 1
        max_len = max(len(p) for p in chains.values())
        # Every chain starts at the MLE: its first point is the one nearest the
        # optimum, so the MLE genuinely is its inner neighbour. Held as a
        # record rather than a bare vector so each step can hand the next one
        # its simplex as well as its solution; the MLE has no simplex of its
        # own, so the first step of each chain is the one place scipy still
        # builds the initial simplex itself.
        seeds = {k: {"nuisance_x": np.delete(res_x, k[0]).tolist(),
                     "x_fixed": float(res_x[k[0]])}
                 for k in chains}

        print(f"\n[profile] pass 4 (sweep {sweep + 1}): warm-started "
              f"continuation over {len(chains)} chain(s), up to {max_len} "
              f"step(s) each")

        def record_warm(res):
            """Keep the lower of the warm and cold evaluations, and mark the
            point warm-refined either way so a resume does not redo it."""
            res["dnll"] = (float(res["nll"]) - nll_at_optimum
                           if res.get("nll") is not None else float("nan"))
            name = res["param_name"]
            key = ProfileCheckpointKey(res["x_fixed"])
            prev = completed.get(name, {}).get(key)

            stats_out["n_attempted"] += 1
            keep = res
            gain = 0.0
            if prev is not None:
                if _profile_record_is_better(res, prev):
                    gain = _nll_gain(prev, res)
                    if gain > 0:
                        stats_out["n_improved"] += 1
                        stats_out["nats_recovered"] += gain
                else:
                    # The warm slice may itself have been stopped by the clock,
                    # so its spend and its optimizer state travel onto the
                    # record even when its value lost.
                    keep = _carry_resume_state(dict(prev), res)
                # phase records *why* a point exists (grid point vs bracket
                # probe) and is counted by the refinement budget, so it must
                # survive a warm re-run regardless of which value won.
                keep["phase"] = prev.get("phase", res.get("phase", 1))
            keep["warm_refined"] = int(
                (prev or {}).get("warm_refined", 0)) + 1
            # How much re-running this point warm actually lowered it. This is
            # the only direct measurement of the bias in the point's earlier
            # value, and it is what the report should judge a capped point on:
            # hitting the evaluation cap says nothing on its own, since a point
            # can stop on the cap having long since reached the minimum. A
            # point that moved is one whose earlier dNLL was too high.
            keep["warm_gain"] = max(float((prev or {}).get("warm_gain", 0.0)),
                                    float(max(gain, 0.0)))

            if checkpoint is not None and keep.get("status") == "ok":
                checkpoint.append(keep)
            completed.setdefault(name, {})[key] = keep

            # Seed the next step from whichever record we kept, not blindly
            # from the warm one: if the cold evaluation was better, its
            # location -- and its simplex -- are the better thing to continue
            # from.
            k = (int(res["param_idx"]), int(res["direction"]))
            if k in seeds and keep.get("nuisance_x"):
                seeds[k] = keep

        for step in range(max_len):
            jobs = []
            for (i, sign), pts in chains.items():
                if step >= len(pts):
                    continue
                jobs.append(make_job(
                    i, pts[step]["x_fixed"], meta[i]["is_log"],
                    nuisance_bounds_for(i), phase=3, direction=sign,
                    seed=seeds[(i, sign)],
                ))
            if jobs:
                nll_batch_profile(jobs, on_result=record_warm,
                                  label=f"profile-warm{sweep + 1}.{step + 1}")

    if stats_out["n_attempted"]:
        print(f"\n[profile] pass 4: {stats_out['n_improved']} of "
              f"{stats_out['n_attempted']} point(s) improved, "
              f"{stats_out['nats_recovered']:.4g} nats recovered in total")
        if stats_out["n_improved"]:
            print(f"    Every improvement lowers the profile, so the intervals "
                  f"below are wider than the cold-started grid alone would "
                  f"give. That difference was bias, not noise.")
    return stats_out


def _bracket_is_tight(x_in, x_out, p_opt, is_log, rtol):
    """Whether the crossing is already located closely enough to stop probing.

    Expressed as a relative precision on the confidence bound itself, so the
    same number means the same thing on a log10-scaled parameter and a linear
    one.

    The profile exists to answer whether a parameter is identifiable and roughly
    where its bound lies, not to pin that bound to five significant figures. One
    Lecanemab run spent all four of its probes moving a crossing from 0.017920
    to 0.017911 -- 0.05%, on an interval whose own half-width is 25% -- while the
    upper side of the same parameter sat at dNLL 1.05 and never got a single
    evaluation. Stopping at a relative width is what frees that budget for the
    sides that have no answer yet.
    """
    if rtol is None or rtol <= 0:
        return False
    if is_log:
        return 10.0 ** abs(x_out - x_in) - 1.0 <= rtol
    if x_in > 0 and x_out > 0:
        return abs(x_out / x_in - 1.0) <= rtol
    scale = max(abs(x_in - p_opt), abs(x_out - p_opt))
    return scale > 0 and abs(x_out - x_in) / scale <= rtol


def _build_refinement_jobs(param_names, completed, res_x, meta,
                           nuisance_bounds_for, make_job, threshold, n_refine,
                           anchor=0.0, bracket_rtol=0.0):
    """Probes at the predicted threshold crossing for each parameter/direction.

    *anchor* is the common profile minimum; every stored dNLL is measured from
    it here so bracketing uses the same reference the CI extraction will.

    Each probe is warm-started from the inner end of its bracket -- the nearest
    completed point on the same side -- taking both its nuisance solution and
    its simplex. A probe sits
    exactly where the interval is read, so a cold-started one is the most
    damaging point on the whole curve: it stalls high, and because
    ``_extract_profile_ci`` stops at the *first* point above the threshold, a
    single inflated probe truncates the interval there regardless of how good
    the surrounding grid is. Seeding from the bracket's inner point costs
    nothing and keeps the curve monotone.
    """
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
            # inner_rec stays None while the bracket's inner end is the optimum
            # itself, which is exactly when a cold start (from the MLE) is the
            # right seed anyway.
            inner_x, inner_d, inner_rec = p_opt, 0.0 - anchor, None
            for r in side:
                d = r.get("dnll")
                if d is None or not np.isfinite(d):
                    continue
                d = d - anchor
                if d > threshold:
                    # Bracket found: probe the predicted crossing. One probe per
                    # launch keeps pass 2 a single wide batch -- iterating here
                    # would serialize what we just made parallel.
                    r_in, r_out = np.sqrt(max(inner_d, 0.0)), np.sqrt(d)
                    x_in, x_out = inner_x, r["x_fixed"]
                    if _bracket_is_tight(x_in, x_out, p_opt, is_log,
                                         bracket_rtol):
                        break
                    if r_out > r_in + 1e-12:
                        frac = (np.sqrt(threshold) - r_in) / (r_out - r_in)
                        frac = min(max(frac, 0.05), 0.95)
                    else:
                        frac = 0.5
                    x_probe = x_in + frac * (x_out - x_in)
                    if ProfileCheckpointKey(x_probe) not in completed.get(name, {}):
                        jobs.append(
                            make_job(i, x_probe, is_log, nb, phase=2,
                                     direction=sign, seed=inner_rec)
                        )
                    break
                inner_x, inner_d = r["x_fixed"], d
                if r.get("nuisance_x"):
                    inner_rec = r
    return jobs


def profile_reach_report(param_names, completed, res_x, bounds, threshold,
                         anchor=0.0):
    """Per parameter and side: did the profile cross, stop at a bound, or run out?

    "No confidence bound" used to be one undifferentiated outcome, and it hid
    the only distinction that tells the user what to do next. A side that walked
    all the way to its declared parameter bound without dNLL reaching 1.9207 has
    given a real answer -- the parameter is not identifiable anywhere it is
    allowed to go, and the fix is to widen the bound if wider values are
    physical, or to drop the parameter. A side that merely used up its extension
    budget has given no answer at all, and the fix is more budget. Reporting both
    as "open -- raise profile_se_span" sent the user to re-run for the first case,
    where re-running with a wider span changes nothing.
    """
    out = {}
    for i, name in enumerate(param_names):
        p_opt = res_x[i]
        lb, ub = _param_bounds(bounds, i)
        sides = {}
        for sign, key in ((-1, "lower"), (+1, "upper")):
            side = [r for r in _side_points(completed, name, p_opt, sign)
                    if r.get("dnll") is not None and np.isfinite(r["dnll"])]
            if not side:
                sides[key] = {"state": "empty", "reach": None, "max_dnll": None,
                              "at_bound": False}
                continue
            outer = side[-1]
            bound = lb if sign < 0 else ub
            at_bound = _at_bound(outer["x_fixed"], bound, sign)
            if _side_has_crossed(side, threshold, anchor):
                state = "crossed"
            else:
                state = "bound" if at_bound else "budget"
            sides[key] = {
                "state": state,
                "reach": float(outer.get("x_fixed_linear", outer["x_fixed"])),
                "max_dnll": float(max(r["dnll"] for r in side) - anchor),
                "at_bound": bool(at_bound),
            }
        out[name] = sides
    return out


def _print_profile_reach(reach):
    """Say which sides never reached the threshold, and why."""
    stalled = []
    for name, sides in reach.items():
        for key, d in sides.items():
            if d["state"] in ("budget", "bound"):
                stalled.append((name, key, d))
    if not stalled:
        return
    print(f"\n[profile] {len(stalled)} side(s) never reached the 1.9207 threshold:")
    for name, key, d in stalled[:20]:
        why = ("reached the parameter bound" if d["state"] == "bound"
               else "ran out of extension steps")
        print(f"    {name} ({key}): {why} at {d['reach']:.6g}, "
              f"highest dNLL {d['max_dnll']:.4g}")
    if len(stalled) > 20:
        print(f"    ... and {len(stalled) - 20} more")
    if any(d["state"] == "bound" for _n, _k, d in stalled):
        print("    A side stopped at its bound is an answer: the parameter is "
              "not identifiable anywhere it is allowed to go. Widen that "
              "parameter's bound only if wider values are physical.")
    if any(d["state"] == "budget" for _n, _k, d in stalled):
        print("    A side that ran out of steps is not an answer. Raise "
              "profile_max_extend (or profile_se_span) and re-run; the "
              "checkpoint keeps the points already computed.")


def profile_convergence_report(param_names, completed):
    """How many profile points stopped on the optimizer cap, in total and per
    parameter.

    Points restored from a checkpoint written before convergence was recorded
    carry no ``converged`` key. They are counted as *unknown*, not as converged:
    giving a clean bill of health to points whose convergence was never measured
    would defeat the point of measuring it.
    """
    per_param = {}
    n_total = n_bad = n_unknown = n_running = 0
    for name in param_names:
        total = bad = unknown = running = 0
        # The largest amount the warm pass lowered any point, and the largest
        # among the points that are near enough to the threshold to set the
        # bound. The second is the one that matters: a 40-nat drop out in the
        # tail moves nothing, a 0.5-nat drop at the crossing moves the bound.
        gain_max = gain_near = 0.0
        n_warm = 0
        for r in completed.get(name, {}).values():
            total += 1
            g = r.get("warm_gain")
            if g is not None and np.isfinite(g):
                n_warm += 1
                gain_max = max(gain_max, float(g))
                d = r.get("dnll")
                if (d is not None and np.isfinite(d)
                        and abs(float(d)) <= 2.0 * _PROFILE_THRESHOLD):
                    gain_near = max(gain_near, float(g))
            if r.get("interrupted"):
                # Stopped by the wall clock with its state saved, not by the
                # optimizer giving up. Counting these as capped would report
                # every point of a preempted link as an optimizer failure and
                # warn that every interval is too narrow, when the honest
                # answer is that the work is simply not finished yet.
                running += 1
            elif "converged" not in r:
                unknown += 1
            elif not r["converged"]:
                bad += 1
        per_param[name] = {"n": total, "n_not_converged": bad,
                           "n_unknown": unknown, "n_interrupted": running,
                           "n_warm_measured": n_warm,
                           "warm_gain_max": gain_max,
                           "warm_gain_near": gain_near}
        n_total += total
        n_bad += bad
        n_unknown += unknown
        n_running += running
    return {"n_points": n_total, "n_not_converged": n_bad,
            "n_unknown": n_unknown, "n_interrupted": n_running,
            "per_param": per_param}


def _print_profile_convergence(report):
    """Report capped nuisance optimizations, which bias CIs in one direction."""
    n, n_bad, n_unknown = (report["n_points"], report["n_not_converged"],
                           report["n_unknown"])
    if n_bad:
        print()
        print(f"*** WARNING: {n_bad} of {n} profile point(s) stopped on the "
              f"optimizer's iteration cap instead of converging.")
        print(f"    A capped nuisance minimization returns an upper bound on "
              f"the profile, not the profile, so every CI read through those "
              f"points is too narrow. Raise the budget with")
        print(f"    optimizer_kwargs['profile_optimizer_kwargs'] = "
              f"{{'options': {{'maxiter': ...}}}} and re-run into a fresh "
              f"checkpoint directory.")
        offenders = sorted(
            ((name, d) for name, d in report["per_param"].items()
             if d["n_not_converged"]),
            key=lambda kv: -kv[1]["n_not_converged"],
        )
        for name, d in offenders[:10]:
            print(f"      {name}: {d['n_not_converged']}/{d['n']} point(s)")
        if len(offenders) > 10:
            print(f"      ... and {len(offenders) - 10} more parameter(s)")
    if n_unknown:
        print(f"\n[profile] {n_unknown} point(s) were checkpointed before "
              f"convergence was recorded; their convergence is unknown. Delete "
              f"the checkpoint directory to re-measure them.")
    if n and not n_bad and not n_unknown:
        print(f"\n[profile] all {n} nuisance optimization(s) converged.")


def profile_anchor_gap(completed):
    """Lowest dNLL found anywhere in the profile, across all parameters.

    Every profile point evaluates the same likelihood, so a point below the
    reported optimum on *any* parameter is evidence about the global minimum.
    A materially negative value means the fit did not converge to the minimum
    of the function being profiled -- with one shared objective that is a
    convergence failure, not a definition mismatch, and it must be reported
    rather than silently absorbed into the anchor.
    """
    best = 0.0
    where = None
    for name, pts in completed.items():
        for r in pts.values():
            d = r.get("dnll")
            if d is None or not np.isfinite(d) or d >= best:
                continue
            best = float(d)
            where = (name, r.get("x_fixed_linear", r.get("x_fixed")),
                     r.get("nuisance_x"))
    return best, where


def _better_point_record(param_names, res_x, meta, anchor, where,
                         nll_at_optimum):
    """The lowest-NLL point the scan found, when it beats the reported optimum.

    Finding one is normal rather than alarming. A profile minimizes over every
    nuisance parameter at each fixed value, so it searches places the fit never
    visited -- that is the whole point of it, and on a hard likelihood it will
    sometimes land lower than the optimizer did.

    What makes the finding useful rather than merely worrying is having
    somewhere to go with it. A gap in nats tells the reader their optimum is
    wrong without telling them what to do; the full parameter vector is a
    refit's starting point. It is assembled here because this is the only place
    that holds all three pieces at once: the fixed value, the nuisance solution
    beside it, and the per-parameter scaling needed to put both back into
    linear units.

    Returns None when the fit already sits at the profile minimum.
    """
    if where is None or anchor >= -1e-3:
        return None
    name, value_lin, nuisance_x = where
    try:
        idx = list(param_names).index(name)
    except ValueError:
        return None

    record = {
        "parameter": name,
        "value": float(value_lin),
        "dnll": float(anchor),
        "nll": float(nll_at_optimum + anchor),
    }

    # The full vector, in linear units, ready to be pasted into a spec's x0.
    try:
        nx = np.asarray(nuisance_x, dtype=float)
        if nx.size != len(param_names) - 1:
            return record
        is_log = meta[idx]["is_log"]
        if is_log and not (value_lin > 0):
            return record
        x_fixed_opt = np.log10(value_lin) if is_log else float(value_lin)
        full_opt = np.insert(nx, idx, x_fixed_opt)
        record["x"] = [float(10.0 ** v) if meta[i]["is_log"] else float(v)
                       for i, v in enumerate(full_opt)]
        record["param_names"] = list(param_names)
    except (TypeError, ValueError, KeyError, IndexError):
        pass
    return record


def _assemble_profile_traces(param_names, completed, res_x, meta, anchor=0.0):
    """Turn completed points into {name: (param_vals_linear, dnll)} traces.

    *anchor* is subtracted from every dNLL so the whole set of traces is
    measured from one common minimum.

    The optimum is included as a point evaluated at ``res_x`` like any other,
    and it is *not* forced to zero. The old hardcoded 0.0 was arithmetically
    true at the anchor but produced a spike in an otherwise smooth curve
    whenever the anchor was not the minimum, and it also defeated the flatness
    detector downstream: a genuinely flat profile could never have zero range
    while an artificial 0.0 was spliced into it.
    """
    out = {}
    for i, name in enumerate(param_names):
        is_log = meta.get(i, {}).get("is_log", False)
        pts = [r for r in completed.get(name, {}).values()
               if r.get("dnll") is not None and np.isfinite(r["dnll"])]
        xs = [res_x[i]] + [r["x_fixed"] for r in pts]
        ys = [0.0 - anchor] + [float(r["dnll"]) - anchor for r in pts]
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
    fixed_sigmas=None, warm_passes=1, max_extend=8, extend_growth=2.0,
    bracket_rtol=0.05, screen_span_decades=None, screen_min_reach_decades=None,
    replicates=None,
):
    """Wire the pool, the checkpoint store, the wall budget and the profile."""
    from Engine.Profile_checkpoint import (
        ProfileCheckpoint, spec_fingerprint, default_run_id, solver_fingerprint,
    )
    from Engine.Deadline import RunBudget, resolve_deadline
    from Engine.Identifiability import screen_or_raise, screen_summary

    # How the model is integrated is part of the objective -- output density
    # feeds np.interp, tolerances decide what comes back -- and it lives in
    # functions on the replicates, where nothing else in the fingerprint can
    # see it. Passed in rather than derived because this is the only place that
    # has both the replicates and the checkpoint store.
    model_hash, spec_hash = spec_fingerprint(
        param_names, res_x, groups, scales, model_text,
        fixed_sigmas=fixed_sigmas,
        solver_hash=solver_fingerprint(replicates),
    )
    # run_id must be stable across launches or resuming can never happen.
    run_id = default_run_id(run_id, model_hash, spec_hash)
    root = paths.get("plot_path") if checkpoint_enabled else None
    ckpt = ProfileCheckpoint(root, run_id, model_hash, spec_hash,
                             enabled=bool(root))
    if ckpt.dir:
        print(f"\n[profile] checkpointing to {ckpt.dir}")

    # The timing history lives beside the points it describes, so it is scoped
    # to this exact model and spec: a run whose points got ten times slower
    # gets a new directory and starts measuring afresh rather than admitting
    # work on the previous version's costs.
    budget = RunBudget(
        deadline=resolve_deadline(),
        timing_path=os.path.join(ckpt.dir, "timing.json") if ckpt.dir else None,
    )
    print(f"[profile] {budget.describe()}")

    # How much of this launch went on getting to the point where profile points
    # could start. Everything before this line -- model generation, the fit or
    # x0 evaluation, the Hessian, compiling the models in every worker -- is
    # paid again by every link of a chain, so it is reported rather than left
    # to be guessed at.
    try:
        started = float(os.environ.get("PROFILE_PROCESS_START", ""))
    except ValueError:
        started = None
    if started:
        setup_s = time.time() - started
        share = (f", {100.0 * setup_s / (setup_s + budget.remaining()):.0f}% "
                 f"of this link" if budget.is_limited else "")
        print(f"[profile] {setup_s / 60.0:.1f} min spent getting here "
              f"(model build, fit/x0, Hessian, worker startup){share}")

    # Pass 0: the slice screen, before a single profile point is started.
    #
    # It is placed here rather than inside run_parallel_profile because it is
    # not a pass of the profile -- it decides whether there is a profile to
    # run. A parameter whose slice sits below the threshold decades from its
    # fitted value has a profile that sits no higher, so its interval is open,
    # and no amount of profiling will close it; the run stops and a human fixes
    # the parameter to a defensible value. Cheap in the direction that matters:
    # an unbounded parameter is what the extension pass spends the most on and
    # learns the least from.
    #
    # The screen is given the Wald SE, which by this point is computed and
    # cached, but only to place its ladder where a crossing is likely. The
    # verdict comes from the far end of the walk and does not depend on it, so
    # a singular Hessian weakens the resolution and not the finding.
    screen_kw = {}
    if screen_span_decades is not None:
        screen_kw["span_decades"] = float(screen_span_decades)
    if screen_min_reach_decades is not None:
        screen_kw["min_reach_decades"] = float(screen_min_reach_decades)
    screen = screen_or_raise(
        lambda xs, label=None: evaluator.evaluate_batch(xs, label=label),
        res_x, nll_at_optimum, param_names, bounds,
        scales=scales, wald_se=wald_se, ckpt_dir=ckpt.dir,
        threshold=_PROFILE_THRESHOLD, range_factor=range_factor, **screen_kw,
    )

    def batch(jobs, on_result=None, label=None):
        return evaluator.profile_batch(jobs, on_result=on_result, label=label,
                                       budget=budget)

    try:
        traces, anchor, where, convergence = run_parallel_profile(
            batch, res_x, nll_at_optimum, param_names,
            bounds, scales, method=method, optimizer_kwargs=optimizer_kwargs,
            wald_se=wald_se, n_grid=n_grid, range_factor=range_factor,
            se_span=se_span, n_refine=n_refine, checkpoint=ckpt,
            warm_passes=warm_passes, max_extend=max_extend,
            extend_growth=extend_growth, bracket_rtol=bracket_rtol,
        )
        if ckpt.n_skipped_stale:
            print(f"[profile] ignored {ckpt.n_skipped_stale} checkpoint record(s) "
                  f"from a different model or spec")
        # Rides into the results snapshot so a reader can tell a profile that
        # was screened from one that predates the screen entirely.
        convergence["screen"] = screen_summary(screen)
        return traces, anchor, where, convergence
    finally:
        budget.save()
        ckpt.close()


def _run_fast_profile_with_checkpoint(
    evaluator, res_x, nll_at_optimum, param_names, bounds, scales, groups,
    model_text, paths, method, optimizer_kwargs, wald_se, wald_cov, run_id,
    checkpoint_enabled=True, fixed_sigmas=None, replicates=None,
    round_evals=None, n_rounds=None, near_zero_frac=None,
    screen_span_decades=None, screen_min_reach_decades=None,
):
    """Wire the pool, the checkpoint store and the wall budget to the fast pass.

    Same directory, same fingerprint and same record format as the full
    profile, so every point this pass computes is already in the checkpoint
    when a full profile is run afterwards. The screen is run here without the
    halt the full profile applies: an open side is one of this pass's verdicts
    rather than a reason to stop, since the whole point is a table that says
    which sides are open, which are closed, and which are in between.
    """
    from Engine.Fast_profile import (
        DEFAULT_NEAR_ZERO_FRAC, DEFAULT_ROUND_EVALS, DEFAULT_ROUNDS,
        run_fast_profile, save_report,
    )
    from Engine.Identifiability import MIN_REACH_DECADES, SPAN_DECADES
    from Engine.Profile_checkpoint import (
        ProfileCheckpoint, spec_fingerprint, default_run_id, solver_fingerprint,
    )
    from Engine.Deadline import RunBudget, resolve_deadline

    model_hash, spec_hash = spec_fingerprint(
        param_names, res_x, groups, scales, model_text,
        fixed_sigmas=fixed_sigmas,
        solver_hash=solver_fingerprint(replicates),
    )
    run_id = default_run_id(run_id, model_hash, spec_hash)
    root = paths.get("plot_path") if checkpoint_enabled else None
    ckpt = ProfileCheckpoint(root, run_id, model_hash, spec_hash,
                             enabled=bool(root))
    if ckpt.dir:
        print(f"\n[fast profile] checkpointing to {ckpt.dir}")

    budget = RunBudget(
        deadline=resolve_deadline(),
        timing_path=os.path.join(ckpt.dir, "timing.json") if ckpt.dir else None,
    )
    print(f"[fast profile] {budget.describe()}")

    def batch(jobs, on_result=None, label=None):
        return evaluator.profile_batch(jobs, on_result=on_result, label=label,
                                       budget=budget)

    def nll_batch(xs, label=None):
        return evaluator.evaluate_batch(xs, label=label)

    try:
        report = run_fast_profile(
            batch, nll_batch, res_x, nll_at_optimum, param_names, bounds,
            scales, method=method, optimizer_kwargs=optimizer_kwargs,
            wald_se=wald_se, wald_cov=wald_cov, checkpoint=ckpt,
            ckpt_dir=ckpt.dir, threshold=_PROFILE_THRESHOLD,
            round_evals=(round_evals if round_evals is not None
                         else DEFAULT_ROUND_EVALS),
            n_rounds=n_rounds if n_rounds is not None else DEFAULT_ROUNDS,
            near_zero_frac=(near_zero_frac if near_zero_frac is not None
                            else DEFAULT_NEAR_ZERO_FRAC),
            span_decades=(float(screen_span_decades)
                          if screen_span_decades is not None else SPAN_DECADES),
            min_reach_decades=(float(screen_min_reach_decades)
                               if screen_min_reach_decades is not None
                               else MIN_REACH_DECADES),
        )
        path = save_report(report, ckpt.dir)
        if path:
            print(f"[fast profile] report written to {path}")
        return report
    finally:
        budget.save()
        ckpt.close()


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
        attach_event_times(experiment, r, verbose=True)

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
                # The events were just regenerated, so the previous attachment
                # describes a model that no longer exists.
                attach_event_times(experiment, r_new)
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
        res = minimize(objective, x0, method=method, bounds=bounds or None, **opt_kw)
        
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
                    attach_event_times(experiment, r_new)
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
                print("[opt] the fast profile runs only through the decoupled "
                      "spec route (run_optimization_from_groups with an "
                      "Optimization spec), where the worker pool and the "
                      "checkpoint live; nothing was profiled here.")
            elif profile_likelihood_analysis:
                def true_profile_likelihood_func(param_idx, n_points=20, range_factor=2.0):
                    se_array = out["stats"].get("wald_se_opt")
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
    preequil_cache=True,
    reuse_fit=True,
):
    """
    Optimize shared parameters using ``experiment.opt_groups`` or ``optimization_spec``.

    fit_mode : "optimize" (default) runs the optimizer; "evaluate_x0" skips the
        fit and evaluates the starting point, so diagnostics can be run against
        stored parameters.  Supersedes the deprecated ``profile_without_opt``.
    reuse_fit : under fit_mode="optimize", look for a finished fit of exactly
        this problem in results/<MODEL>/fits/ and take its optimum instead of
        refitting, so a relaunch (a requeue on a preemptible partition) reaches
        the profile in minutes and anchors it on the same optimum as before.
        A fit that was killed part-way resumes from its best point. False
        always fits afresh. See Engine.Fit_cache.
    n_workers : processes used for the diagnostics (Wald / slice / Sobol).
        None uses all cores but one; 1 forces serial evaluation.  The fit itself
        is serial regardless -- Nelder-Mead is inherently sequential.
    profile_checkpoint : write each completed profile point to
        results/<MODEL>/profiles/<run_id>/<param>.jsonl so a killed run resumes
        instead of restarting.  Set False to disable.
    preequil_cache : reuse the leading untracked pre-dose block across
        evaluations instead of re-integrating it every time.  Enabled only when
        an invariance check confirms the fitted parameters cannot act before the
        first dose, so setting this True is a request, not an assertion.
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
            getattr(optimization_spec, "parameter_scale", None), param_names,
            bounds=bounds_lin, x0=x0_lin,
            search_decades=getattr(optimization_spec, "search_decades", None),
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

            # Where this arm's discontinuities are, read off the compiled model.
            # Attached rather than computed, because a trigger built on a fitted
            # parameter has to be re-read as the fit moves; see
            # Engine.Event_times. A solver setting that does not ask for them is
            # unaffected.
            attach_event_times(replicate, r, verbose=True)

            # events_str is retained so pool workers rebuild the *same* model
            # rather than regenerating events from data themselves.
            models[sim_name] = {"r_ic": r_ic, "r": r, "df_dict": df_dict,
                                "events": events_str}
            replicates[sim_name] = replicate

        # Active replicates only: passive ones are simulated once after the fit.
        active_replicates = {
            name: rep for name, rep in replicates.items() if name in active_sim_names
        }

        preequil_ok = _enable_preequil_cache(
            models, active_replicates, param_names, x0_lin, bounds_lin,
            enabled=preequil_cache and not _events_dynamic,
        )

        _debug_calls = [0]
        _progress = {"best": float("inf"), "t0": None,
                     "best_x": None, "best_dirty": False}

        # The fit's own cache. Keyed on the problem rather than the answer, so
        # a relaunch of the same job finds the optimum it already paid for and
        # the profile that follows is anchored on the same point as before.
        # Only under fit_mode="optimize": --no-fit means "profile the spec's
        # x0", and that must stay literally true.
        fit_cache = None
        if fit_mode == "optimize" and reuse_fit:
            from Engine.Fit_cache import FitCache, fit_fingerprint, data_fingerprint
            from Engine.Profile_checkpoint import solver_fingerprint
            _fit_model_hash, _fit_hash = fit_fingerprint(
                param_names, x0_lin, bounds_lin, scales,
                optimization_spec.groups, model_text, method, optimizer_kwargs,
                n_starts=getattr(optimization_spec, "n_starts", 1),
                start_seed=getattr(optimization_spec, "start_seed", None),
                search_decades=getattr(optimization_spec, "search_decades", None),
                solver_hash=solver_fingerprint(active_replicates),
                data_hash=data_fingerprint(
                    {k: models[k] for k in active_replicates}),
            )
            fit_cache = FitCache(paths.get("plot_path"), groups_tag,
                                 _fit_model_hash, _fit_hash, len(param_names))

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

            trace_collector = {}

            # The concentrated likelihood *is* the objective. The group weights
            # and group_normalization in the spec are deliberately not applied
            # here: per-block sigma already does the "weight experiments
            # equally" job on a principled scale, and applying an ad-hoc
            # normalization on top would make the fitted optimum the minimizer
            # of a function no diagnostic evaluates -- which is exactly what put
            # the reported optimum 1.38 nats above the profile minimum before.
            blocks = collect_loss_blocks(
                sim_results, optimization_spec.groups, replicates, param_names,
                x_lin, x_dict, trace_collector=trace_collector,
            )
            if not blocks:
                return 1e10
            total_loss = concentrated_nll(blocks)
            if not np.isfinite(total_loss):
                return 1e10

            # Report per-block sigma rather than a raw loss share: with the
            # concentrated form a block's whole contribution is (n/2)log(sigma^2),
            # so sigma is the number that says which block fits badly.
            loss_components = {f"{key} [{obs}]": sig
                               for (key, obs), sig in block_sigmas(blocks).items()}

            # Best-tracking runs on every eval, the first three included. Those
            # debug calls are real evaluations -- call 1 is x0 itself -- so
            # leaving them out of the tally let a better point go unrecorded and
            # kept the progress JSON's parameter snapshot off the true
            # best-so-far. The overlay/JSON write is driven off new_best for the
            # same reason.
            n = call_n + 1
            new_best = total_loss < _progress["best"]
            if new_best:
                _progress["best"] = total_loss
                _progress["best_x"] = np.array(x_lin, dtype=float)
                _progress["best_dirty"] = True

            # The best point so far, written on a throttle so a fit that is
            # killed hours in restarts from where it got to. Checked on every
            # evaluation rather than only on an improvement, so a best that
            # arrived while the throttle was closed is still written by the
            # next evaluation after it opens.
            if fit_cache is not None and _progress["best_dirty"]:
                if fit_cache.save_partial(_progress["best_x"], _progress["best"],
                                          n_evals=n):
                    _progress["best_dirty"] = False

            if do_debug:
                print(f"  -> concentrated NLL = {total_loss:.6g}"
                      f"  best={_progress['best']:.6g}")
                if loss_components:
                    comp_str = "  ".join(f"{k}={v:.4g}" for k, v in loss_components.items())
                    print(f"    block sigma: {comp_str}")
                if new_best:
                    _render_progress_overlay(
                        trace_collector, total_loss, _progress["best"],
                        n, paths.get("plot_path"),
                        param_names=param_names, param_values=x_lin,
                        model_name=paths.get("MODEL_NAME", ""),
                        experiment_id=groups_tag, method=method,
                        blocks=blocks,
                    )
            else:
                if n % 10 == 0 or new_best:
                    elapsed = time.time() - _progress["t0"]
                    rate = n / elapsed if elapsed > 0 else 0.0
                    tag = "*" if new_best else " "
                    print(
                        f"  [opt]{tag}eval {n:5d}  nll={total_loss:.5g}"
                        f"  best={_progress['best']:.5g}"
                        f"  {elapsed:6.0f}s  ({rate:.1f} eval/s)",
                        flush=True,
                    )
                    if loss_components:
                        comp_str = "  ".join(f"{k}={v:.4g}" for k, v in loss_components.items())
                        print(f"         block sigma: {comp_str}", flush=True)
                    _render_progress_overlay(
                        trace_collector, total_loss, _progress["best"],
                        n, paths.get("plot_path"),
                        param_names=param_names, param_values=x_lin,
                        model_name=paths.get("MODEL_NAME", ""),
                        experiment_id=groups_tag, method=method,
                        blocks=blocks,
                    )
            _debug_calls[0] += 1
            return total_loss

        opt_kw = _prepare_optimizer_kwargs(method, optimizer_kwargs, fast, maxiter, tol)
        start_records = None
        # Where the optimum came from when it was not fitted here: the path of
        # the cached fit. None means the optimizer ran in this process.
        fit_source = None
        res = None
        if fit_mode == "evaluate_x0":
            from scipy.optimize import OptimizeResult
            print("[opt] fit_mode='evaluate_x0' — skipping the fit and evaluating x0 "
                  "so diagnostics run against the supplied parameters.")
            x0_arr = np.array(x0)
            res = OptimizeResult(x=x0_arr, fun=objective(x0_arr), success=True,
                                 message="Optimization bypassed (fit_mode=evaluate_x0)",
                                 nit=0, nfev=1)
        elif fit_cache is not None and fit_cache.load_complete() is not None:
            # A finished fit of exactly this problem. Re-evaluated here rather
            # than trusted: one evaluation is cheap, it proves the model still
            # integrates at that point, and it is the last line of defence
            # against a change the fingerprint does not see. A value that has
            # moved is treated as a miss and the fit runs.
            from scipy.optimize import OptimizeResult
            cached = fit_cache.load_complete()
            x_cached = _to_opt_space(cached["x_lin"], scales)
            fun_now = objective(np.array(x_cached))
            fun_then = cached.get("fun")
            drift = (abs(fun_now - fun_then)
                     if fun_then is not None and np.isfinite(fun_now) else float("inf"))
            if drift <= 1e-3 + 1e-6 * abs(fun_then or 0.0):
                print(f"[opt] reusing the fitted optimum from {cached['path']}\n"
                      f"      (saved {cached.get('saved')}, nll {fun_then:.6g}; "
                      f"re-evaluated here as {fun_now:.6g}). The profile is "
                      f"anchored on it. Pass --refit to fit again.")
                res = OptimizeResult(
                    x=np.array(x_cached), fun=fun_now, success=True,
                    message=f"Optimum reused from {cached['path']}",
                    nit=cached.get("nit", 0), nfev=cached.get("nfev", 1))
                fit_source = cached["path"]
            else:
                print(f"[opt] a cached fit exists at {cached['path']} but its NLL "
                      f"re-evaluates as {fun_now:.6g} against the stored "
                      f"{fun_then}; something outside the fingerprint has "
                      f"changed, so the fit runs afresh.")
        if res is not None:
            pass  # settled above: x0 evaluated, or a cached fit reused
        elif method.lower() in _GLOBAL_METHODS:
            # A global method searches the whole domain by construction, so
            # wrapping it in multi-start would only pay for the same thing twice.
            res = _run_global_optimization(objective, x0, bounds, method, opt_kw)
        else:
            starts = _multistart_points(
                x0, bounds, scales,
                getattr(optimization_spec, "n_starts", 1),
                search_decades=getattr(optimization_spec, "search_decades", None),
                seed=getattr(optimization_spec, "start_seed", None),
            )
            if len(starts) == 1:
                x_start = x0
                partial = fit_cache.load_partial() if fit_cache is not None else None
                if partial is not None:
                    # A previous launch of this same fit was killed before it
                    # returned. Its best point is a better start than x0 by
                    # exactly the work it did; the simplex is rebuilt around
                    # it, which costs n+1 evaluations rather than the hours.
                    x_start = _to_opt_space(partial["x_lin"], scales)
                    print(f"[opt] resuming an unfinished fit from its best point "
                          f"(nll {partial.get('fun')}, after "
                          f"{partial.get('n_evals')} evaluation(s), saved "
                          f"{partial.get('saved')}) rather than from x0.")
                res = minimize(objective, x_start, method=method,
                               bounds=bounds or None, **opt_kw)
            else:
                res, start_records, _best_start = _run_multistart(
                    objective, starts, method, bounds, opt_kw, scales
                )
                if res is None:
                    print("[opt] every multi-start fit failed; falling back to "
                          "a single fit from the spec's x0.")
                    res = minimize(objective, x0, method=method,
                                   bounds=bounds or None, **opt_kw)

        if _debug_calls[0] > 3:
            elapsed = time.time() - _progress["t0"]
            print(f"\n  [opt] done — {_debug_calls[0]} evals in {elapsed:.0f}s"
                  f"  best={_progress['best']:.5g}")

        # res.x is in opt space; everything reported out is linear.
        x_lin_opt = _to_linear(res.x, scales)

        # Written now, before any diagnostic starts, because the diagnostics are
        # where a job on a preemptible partition spends its life and where it
        # gets killed. The optimum has to be on disk before that can happen or
        # the next launch fits it all over again.
        if fit_cache is not None and fit_source is None:
            if fit_cache.save_complete(
                    x_lin_opt, res.fun, param_names=param_names,
                    nit=getattr(res, "nit", None), nfev=getattr(res, "nfev", None),
                    success=bool(res.success), message=str(res.message),
                    x0_lin=[float(v) for v in np.atleast_1d(x0_lin)]):
                print(f"[opt] fitted optimum saved to {fit_cache.path}; a relaunch "
                      f"of this same fit will reuse it instead of refitting.")

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
            # Path of the cached fit the optimum was taken from, or None when
            # the optimizer ran in this process.
            "fit_source": fit_source,
            # Every start and where it landed, so the spread that justified
            # (or did not justify) the extra fits is on the record.
            "starts": start_records,
        }

        param_dict = dict(zip(param_names, x_lin_opt.tolist()))
        best_results = {}
        # total_n and the effective parameter count are derived from the loss
        # blocks below, so that they and the objective can never disagree.

        # Simulate every replicate once at the optimum -- this is where passive
        # (plot-only) simulations get their curves, instead of on every eval.
        for sim_name, replicate in replicates.items():
            m = models[sim_name]
            res_dict = run_all(m["r"], sim_name, replicate, m["df_dict"],
                               set_parameters=set_params, parameters=param_dict)
            best_results.update(res_dict)

        # Blocks at the optimum. These are the same (SSE, n) the objective
        # accumulated, so the sigmas reported here are exactly the ones the
        # concentrated likelihood profiled out -- no second, differently-scaled
        # estimate that the diagnostics would then disagree with.
        #
        # This replaces a hand-rolled residual walk that estimated
        #     sigma = sqrt(SSE / (n_block - k/n_obs))
        # per block, which deducted the *whole* parameter count from *every*
        # block (4 arms x 4 parameters removed 16 df from 75 points, inflating
        # each sigma ~13% and widening every interval). The concentrated
        # likelihood wants the ML estimator sqrt(SSE/n) and no df correction --
        # a corrected sigma substituted back is no longer the profile
        # likelihood, and the 1.9207 threshold stops being exact.
        opt_contributions = set()
        opt_blocks = collect_loss_blocks(
            best_results, optimization_spec.groups, replicates, param_names,
            x_lin_opt, param_dict, seen=opt_contributions,
        )
        sigma_by_block = block_sigmas(opt_blocks)
        total_n = total_points(opt_blocks)
        k_eff = effective_k(param_names, opt_blocks)

        # Kept so the frozen-sigma path (evaluate_nll_fixed(concentrated=False))
        # and describe_nll_terms still resolve, and so archived runs remain
        # comparable. It no longer feeds the fit or any diagnostic.
        fixed_sigmas = {}
        for (block_key, _obs_label), sig in sigma_by_block.items():
            for _g in optimization_spec.groups.values():
                for _i, _e in enumerate(_g.get("loss_elements", [])):
                    if _e.get("simulation") == block_key:
                        _lcf = _e.get("loss_config")
                        _lc = _lcf(replicates[block_key]) if callable(_lcf) else _lcf
                        for _oc in (_lc or {}).get("observables", []):
                            if _short_obs_label(_oc["observed_variable"]) == _obs_label:
                                fixed_sigmas[(block_key, _oc["observed_variable"])] = sig

        _n_active_obs = 0
        for _g in optimization_spec.groups.values():
            for _e in _g.get("loss_elements", []):
                _sim = _e.get("simulation")
                if _sim not in replicates:
                    continue
                _lcf = _e.get("loss_config")
                _lc = _lcf(replicates[_sim]) if callable(_lcf) else _lcf
                _n_active_obs += len((_lc or {}).get("observables", []))

        if not opt_blocks:
            print()
            print("*** WARNING: no loss blocks resolved at the optimum. No observable "
                  "matched its data, so there is no likelihood to profile. ***")
            print()
        elif _n_active_obs and len(opt_contributions) < _n_active_obs:
            # Count CONTRIBUTIONS, not blocks. Comparing block count against
            # observable count reports a false miss the moment several elements
            # share a sigma_block -- twelve pooled arms are one block but twelve
            # contributions, and the earlier version called that "1 of 12
            # resolved".
            print()
            print(f"*** WARNING: {len(opt_contributions)} of {_n_active_obs} active "
                  f"observable(s) resolved to data. The rest contribute nothing to "
                  f"the likelihood — check their data_column / time_column mapping. ***")
            print()

        print(f"\n[fit] concentrated NLL at optimum: {res.fun:.6g}")
        _known = known_sigma_blocks(opt_blocks)
        print(f"[fit] {len(opt_blocks)} block(s), {total_n:.0f} points, "
              f"k={len(param_names)} parameters + "
              f"{len(opt_blocks) - len(_known)} estimated sigma(s) = {k_eff}"
              + (f"   ({len(_known)} declared sigma(s), not charged)"
                 if _known else ""))
        for (block_key, obs_label), sig in sorted(sigma_by_block.items()):
            _, n_b, _ks = _unpack_block(opt_blocks[(block_key, obs_label)])
            tag = "declared" if (block_key, obs_label) in _known else "estimated"
            print(f"    sigma[{block_key} · {obs_label}] = {sig:.4g}  "
                  f"(n={n_b:.0f}, {tag})")

        # Estimating a variance from very few points is the one failure mode the
        # concentrated form introduces: a block's contribution is
        # (n/2)log(SSE/n), which runs to -inf as SSE -> 0, so a small block that
        # the model *can* drive to a near-exact fit will dominate the objective.
        # Shared parameters normally prevent that (no single block can be zeroed
        # on its own), so this is a warning rather than a guard -- it bites when
        # a block has few points and a parameter that is effectively private
        # to it. A declared sigma is bounded by construction and so exempt.
        _degen = [(bk, ob, _unpack_block(v)[1]) for (bk, ob), v in opt_blocks.items()
                  if 0 < _unpack_block(v)[1] < 2 and (bk, ob) not in _known]
        _tiny = [(bk, ob, _unpack_block(v)[1]) for (bk, ob), v in opt_blocks.items()
                 if 2 <= _unpack_block(v)[1] < _MIN_BLOCK_POINTS
                 and (bk, ob) not in _known]
        if _degen:
            print()
            print(f"*** WARNING: {len(_degen)} block(s) have a single data point. A "
                  f"variance cannot be estimated from one observation: the block's "
                  f"contribution is log|residual|, which is UNBOUNDED BELOW, so the "
                  f"optimizer can drive the objective to -inf by fitting that one "
                  f"point exactly. ***")
            for bk, ob, n in _degen:
                print(f"      {bk} · {ob}  (n={n:.0f})")
            print('    Fix by pooling: give these elements a shared '
                  '"sigma_block": "<name>" in the spec so they estimate one sigma '
                  'together, or supply a known sigma for them.')
            print()
        if _tiny:
            print()
            print(f"*** NOTE: {len(_tiny)} block(s) have fewer than "
                  f"{_MIN_BLOCK_POINTS} points. Their sigma rests on very little "
                  f"data, and a block the model can fit almost exactly pulls the "
                  f"objective sharply negative: ***")
            for bk, ob, n in sorted(_tiny, key=lambda t: t[2]):
                print(f"      {bk} · {ob}  (n={n:.0f})")
            print('    Check that no parameter is private to one of these blocks. '
                  'Pooling arms that share a noise process via '
                  '"sigma_block": "<name>" is the usual fix, and it also stops each '
                  'arm costing a parameter in AIC/BIC.')
            print()


        def nll_func_fixed(p):
            # Identical to the fit objective by construction: same blocks, same
            # concentrated form. That identity is the whole point -- it is what
            # makes res.x the minimizer of the curve the profile walks.
            return evaluate_nll_fixed(
                p, models, active_replicates, param_names, scales,
                optimization_spec.groups, optimization_spec.group_normalization,
                fixed_sigmas, model_text=model_text, paths=paths,
                events_dynamic=_events_dynamic, concentrated=True,
            )

        out["results_dict"] = best_results

        # Recorded on EVERY run, not just diagnostic ones. The blocks are already
        # in hand, so nll_proper / AIC / BIC / sigma cost no extra simulation --
        # and _NO_DIAGNOSTICS is the default path, the one whose parameters get
        # copied into the registry. Leaving its output unlabelled is how a number
        # ends up in the registry with nothing recorded about what produced it.
        out["stats"]["nll_proper"] = concentrated_nll(opt_blocks, include_constant=True)
        out["stats"]["aic"] = 2 * k_eff + 2 * out["stats"]["nll_proper"]
        out["stats"]["bic"] = (k_eff * np.log(max(total_n, 1))
                               + 2 * out["stats"]["nll_proper"])
        out["stats"]["k_effective"] = k_eff
        out["stats"]["n_data_points"] = total_n
        out["stats"]["fixed_sigmas"] = fixed_sigmas
        out["stats"]["block_sigmas"] = {
            f"{bk} · {ob}": sg for (bk, ob), sg in sigma_by_block.items()
        }
        out["stats"]["block_n"] = {
            f"{bk} · {ob}": _unpack_block(v)[1] for (bk, ob), v in opt_blocks.items()
        }

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
                preequil_cache=preequil_ok,
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
            if wald_analysis:
                # Cached only when profile checkpointing is on. That is the
                # same switch, and the same directory, as the points the
                # Hessian is computed in order to place.
                _attach_wald_stats(
                    out, nll_func_fixed, res.x, bounds, param_names,
                    scales=scales, nll_batch=nll_batch,
                    # groups_tag, not the profile's run_id: the Hessian is
                    # computed here, before the profile closure exists to
                    # supply one. They agree in every normal run, since the
                    # profile also falls back to groups_tag; a caller that
                    # passes an explicit run_id just puts the cache in a
                    # different directory, which costs sharing, not
                    # correctness -- the key is the fingerprint either way.
                    cache=_make_anchor_cache(
                        paths, groups_tag, param_names, res.x,
                        optimization_spec.groups, scales, model_text,
                        fixed_sigmas, bounds, enabled=settings_checkpoint,
                    ),
                )

            if slice_analysis or profile_likelihood_analysis or fast_profile_likelihood_analysis:
                # Re-evaluated rather than reusing res.fun: this asserts that the
                # diagnostic objective and the fit objective really do agree at
                # the optimum. They are the same function now, so a mismatch here
                # means the two paths have drifted apart and every dNLL below
                # would be measured from the wrong place.
                nll_at_optimum = nll_func_fixed(res.x)
                drift = abs(nll_at_optimum - float(res.fun))
                if drift > 1e-6 * max(1.0, abs(float(res.fun))):
                    print()
                    print(f"*** WARNING: the diagnostic objective disagrees with the "
                          f"fit objective at the optimum by {drift:.4g} "
                          f"({nll_at_optimum:.8g} vs {float(res.fun):.8g}). They are "
                          f"supposed to be the same function — every dNLL below is "
                          f"anchored on a value the fit did not minimize. ***")
                    print()
                out["stats"]["nll_at_optimum"] = nll_at_optimum

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
                    # One capped profile point per side at the slice crossing,
                    # through the pool and into the same checkpoint the full
                    # profile reads. See Engine.Fast_profile.
                    if _pool_state["evaluator"] is None:
                        print("[opt] the fast profile needs the worker pool "
                              "(n_workers > 1); none was built, so nothing "
                              "was profiled.")
                    else:
                        out["stats"]["fast_profile_all"] = (
                            lambda round_evals=None, n_rounds=None,
                            near_zero_frac=None, run_id=None,
                            screen_span_decades=None,
                            screen_min_reach_decades=None:
                            _run_fast_profile_with_checkpoint(
                                _pool_state["evaluator"], res.x, nll_at_optimum,
                                param_names, bounds, scales,
                                groups=optimization_spec.groups,
                                model_text=model_text, paths=paths,
                                method=profile_method,
                                optimizer_kwargs=profile_opt_kwargs,
                                wald_se=out["stats"].get("wald_se_opt"),
                                wald_cov=out["stats"].get("wald_cov"),
                                run_id=run_id or groups_tag,
                                checkpoint_enabled=settings_checkpoint,
                                fixed_sigmas=fixed_sigmas,
                                replicates=active_replicates,
                                round_evals=round_evals, n_rounds=n_rounds,
                                near_zero_frac=near_zero_frac,
                                screen_span_decades=screen_span_decades,
                                screen_min_reach_decades=(
                                    screen_min_reach_decades),
                            )
                        )
                elif profile_likelihood_analysis:
                    def true_profile_likelihood_func(param_idx, n_points=20, range_factor=2.0):
                        se_array = out["stats"].get("wald_se_opt")
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
                            n_refine=4, run_id=None, warm_passes=1,
                            max_extend=8, extend_growth=2.0,
                            bracket_rtol=0.05, screen_span_decades=None,
                            screen_min_reach_decades=None:
                            _run_parallel_profile_with_checkpoint(
                                _pool_state["evaluator"], res.x, nll_at_optimum,
                                param_names, bounds, scales,
                                groups=optimization_spec.groups,
                                model_text=model_text, paths=paths,
                                method=profile_method,
                                optimizer_kwargs=profile_opt_kwargs,
                                wald_se=out["stats"].get("wald_se_opt"),
                                n_grid=n_grid, range_factor=range_factor,
                                se_span=se_span, n_refine=n_refine,
                                run_id=run_id or groups_tag,
                                checkpoint_enabled=settings_checkpoint,
                                fixed_sigmas=fixed_sigmas,
                                warm_passes=warm_passes,
                                max_extend=max_extend,
                                extend_growth=extend_growth,
                                bracket_rtol=bracket_rtol,
                                screen_span_decades=screen_span_decades,
                                screen_min_reach_decades=(
                                    screen_min_reach_decades),
                                replicates=active_replicates,
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
        # Best-tracking runs on every eval, the first three included. Those
        # debug calls are real evaluations -- call 1 is x0 itself -- so leaving
        # them out of the tally let a better point go unrecorded and kept the
        # progress JSON's parameter snapshot off the true best-so-far. The
        # overlay/JSON write is driven off new_best for the same reason.
        n        = call_n + 1
        new_best = total_loss < _progress["best"]
        if new_best:
            _progress["best"] = total_loss

        if do_debug:
            print(f"  -> total_loss = {total_loss:.6g}"
                  f"  best={_progress['best']:.6g}")
            if loss_components:
                comp_str = "  ".join(f"{k}={v:.4g}" for k, v in loss_components.items())
                print(f"    components: {comp_str}")
            if new_best:
                _render_progress_overlay(
                    trace_collector, total_loss, _progress["best"],
                    n, paths.get("plot_path"),
                    param_names=param_names, param_values=x,
                    model_name=paths.get("MODEL_NAME", ""),
                    experiment_id=groups_tag, method=method
                )
        else:
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
        res = minimize(objective, x0, method=method, bounds=bounds or None, **opt_kw)
        
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
                print("[opt] the fast profile runs only through the decoupled "
                      "spec route (run_optimization_from_groups with an "
                      "Optimization spec), where the worker pool and the "
                      "checkpoint live; nothing was profiled here.")
            elif profile_likelihood_analysis:
                def true_profile_likelihood_func(param_idx, n_points=20, range_factor=2.0):
                    se_array = out["stats"].get("wald_se_opt")
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
        model._scalar_abs_tol = 1e-8
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


def profile_ci_status(param_vals, nll_vals_rel, lo, hi, flat_tol=1e-3):
    """Why a bound is missing: 'ok', 'flat', 'open_lower', 'open_upper', 'open'.

    A bare nan told you nothing -- "this parameter is structurally
    non-identifiable" and "the grid was too narrow to reach the threshold" both
    came out as None, and they call for opposite responses (drop or fix the
    parameter, versus widen se_span and re-run).
    """
    y = np.asarray(nll_vals_rel, dtype=float)
    if y.size == 0:
        return "empty"
    if np.ptp(y) < flat_tol:
        return "flat"
    missing_lo = not np.isfinite(lo)
    missing_hi = not np.isfinite(hi)
    if missing_lo and missing_hi:
        return "open"
    if missing_lo:
        return "open_lower"
    if missing_hi:
        return "open_upper"
    return "ok"


def _extract_profile_ci(param_vals, nll_vals_rel, threshold=1.9207):
    """Interpolate lower/upper CI bounds where ΔNLL crosses threshold (default 95%).

    The threshold is a height above the profile's *own* minimum, so the search
    starts at ``argmin`` and walks outward. Callers must hand in traces that are
    already anchored on the shared profile minimum, or this and the stepping
    rule will disagree about where the crossing is.
    """
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

    opt_points = []
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

        # The fitted point is on the grid, so show where the curve crosses it
        # rather than leaving the reader to infer it from the vertical line.
        at_opt = np.isclose(param_vals_normalized, 1.0, rtol=1e-6, atol=0.0)
        if at_opt.any():
            opt_points.append(float(nll_vals_rel[np.argmax(at_opt)]))

    if opt_points:
        ax.plot([1.0] * len(opt_points), opt_points, linestyle='none', marker='x',
                color='black', markersize=8, markeredgewidth=1.5, zorder=5,
                label=f'Optimum (Δ NLL = {max(opt_points, key=abs):.3g})')
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

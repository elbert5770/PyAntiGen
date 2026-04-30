"""
Parameter optimization over multiple experiments: run each experiment with a
given parameter set, compare to data, return a scalar loss for the optimizer.
Also provides run_all for running experiments (with optional parameter injection).
"""

import numpy as np
import warnings
from scipy import stats
try:
    import numdifftools as nd
except ImportError:
    pass
from framework.TelluriumGen import TelluriumGen
from Engine.Simulate import simulate
from Modules.Loss_config import no_optimization


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

    if set_parameters is not None and parameters is not None:
        set_parameters(r, parameters)
        for hook in experiment.get("parameter_hooks", []):
            hook(r, parameters)

    solver_settings  = experiment["Solver_settings"](experiment)
    observed_species = experiment["Observed_species"](r)
    results          = simulate(r, solver_settings, observed_species)
    label            = experiment["Label"]

    return {label: {"results": results, "data": df_dict, "replicate": experiment}}


def set_parameters_from_dict(r, params):
    """Apply a name→value dict to a Tellurium model."""
    for name, value in params.items():
        try:
            r[name] = value
        except Exception:
            pass


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
):
    """
    Run *experiment* with the given parameters and return a scalar NLL loss.

    Parameters
    ----------
    params       : vector or dict of parameter values.
    r            : pre-built Tellurium model.
    exp_num      : experiment identifier (used for fixed_sigmas keying).
    experiment   : treatment dict.
    df_dict      : pre-loaded data dict (not re-read from disk).
    param_names  : list of parameter names matching the params vector.
    loss_config  : dict with "observables" list.
    fixed_sigmas : optional dict (exp_id, obs) -> sigma for Hessian use.
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
    except Exception:
        return 1e10

    total_loss = 0.0
    for label, item in results_dict.items():
        result   = item["results"]
        item_df  = item["data"]
        exp_id   = label

        local_dict = {"np": np, "time": np.asarray(result["time"])}
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
        t_sim = np.asarray(result["time"])

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
                elif isinstance(obs, str) and obs in cols:
                    y_sim = np.asarray(result[obs])
                elif isinstance(obs, str):
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
            else:
                if fixed_sigmas is not None and (exp_id, obs) in fixed_sigmas:
                    sigma = fixed_sigmas[(exp_id, obs)]
                else:
                    sigma_config = obs_cfg.get("noise_formula", None)
                    if sigma_config and sigma_config in local_dict:
                        sigma = float(local_dict[sigma_config])
                    else:
                        w_sum  = obs_weights_v.sum()
                        w_mean = np.dot(obs_weights_v, residuals) / w_sum
                        sigma2 = np.dot(obs_weights_v, (residuals - w_mean) ** 2) / w_sum
                        sigma  = np.sqrt(sigma2) if sigma2 > 1e-12 else 1e-6
                contrib = -np.sum(obs_weights_v * stats.norm.logpdf(y_data_v, loc=y_pred_v, scale=sigma)) / n_eff

            total_loss += contrib

            if debug:
                obs_label = getattr(obs, '__name__', str(obs))
                print(f"  [loss] rep={exp_id}  obs='{obs_label}'")
                print(f" Params: {params}")
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
                    print(f"    sigma={sigma:.4g}  nll={contrib:.4g}")

    return float(total_loss)


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
    method="Nelder-Mead",
    optimizer_kwargs=None,
    fast=False,
    maxiter=None,
    tol=None,
):
    try:
        from scipy.optimize import minimize
    except ImportError:
        raise ImportError("scipy is required for run_optimization") from None

    data_path = paths["data_path"]

    # Pre-build one Tellurium model per experiment and load its data once.
    models = {}
    for exp_num, experiment in experiments.items():
        df_dict    = experiment["Data"](experiment, data_path)
        events_str = experiment["Events"](experiment, df_dict)
        r          = TelluriumGen(model_text + "\n" + events_str, paths)
        experiment["Update_parameters"](r, experiment, mode="Optimizer")
        models[exp_num] = {"r": r, "df_dict": df_dict}

    def objective(x):
        total_loss = 0.0
        for exp_num, experiment in experiments.items():
            m        = models[exp_num]
            loss_val = loss_function(
                x, m["r"], exp_num, experiment, m["df_dict"],
                param_names, loss_config=loss_config,
            )
            if loss_val >= 1e10:
                return 1e10
            total_loss += loss_val
        return total_loss

    opt_kw = _prepare_optimizer_kwargs(method, optimizer_kwargs, fast, maxiter, tol)
    if method.lower() in _GLOBAL_METHODS:
        res = _run_global_optimization(objective, x0, bounds, method, opt_kw)
    else:
        res = minimize(objective, x0, method=method, bounds=bounds or [], **opt_kw)
    out = {"x": res.x, "fun": res.fun, "success": res.success,
           "message": res.message, "stats": {}}

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
                        sigma = (np.sqrt(np.sum(residuals**2) /
                                         max(1, n_block - k / max(1, len(observables_config))))
                                 if n_block > 1 else 1e-6)
                    fixed_sigmas[(exp_id, obs)] = sigma
                    total_n += len(residuals)

        def nll_func_fixed(p):
            total_nll = 0.0
            for exp_num, experiment in experiments.items():
                m = models[exp_num]
                total_nll += loss_function(
                    p, m["r"], exp_num, experiment, m["df_dict"],
                    param_names, loss_config, fixed_sigmas=fixed_sigmas,
                )
            return total_nll

        out["results_dict"] = best_results

        if wald_analysis or slice_analysis or profile_likelihood_analysis:
            aic = 2 * k + 2 * res.fun
            bic = k * np.log(total_n) + 2 * res.fun
            out["stats"]["aic"] = aic
            out["stats"]["bic"] = bic

        if wald_analysis:
            try:
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
                    try:
                        import pypesto
                        import pypesto.profile as pypesto_profile
                        import pypesto.optimize as pypesto_optimize
                        pname = param_names[param_idx]
                        print(f"\n[true profile] {pname}  (max {n_points} steps/side, re-optimizing nuisance params)")
                        objective = pypesto.Objective(fun=nll_func_fixed)
                        if bounds is not None:
                            lb = np.array([b[0] for b in bounds], dtype=float)
                            ub = np.array([b[1] for b in bounds], dtype=float)
                        else:
                            lb = res.x / (range_factor ** 2)
                            ub = res.x * (range_factor ** 2)
                        problem = pypesto.Problem(objective=objective, lb=lb, ub=ub, x_names=list(param_names))
                        pypesto_result = pypesto.Result(problem=problem)
                        pypesto_result.optimize_result.append(
                            pypesto.result.OptimizerResult(id='0', x=res.x.copy(), fval=float(nll_at_optimum)),
                            sort=True,
                        )
                        optimizer = pypesto_optimize.ScipyOptimizer(method='L-BFGS-B')
                        profile_options = pypesto_profile.ProfileOptions()
                        pypesto_profile.parameter_profile(
                            problem=problem, result=pypesto_result, optimizer=optimizer,
                            profile_index=np.array([param_idx]), result_index=0,
                            profile_options=profile_options,
                        )
                        profiler = pypesto_result.profile_result.list[0][param_idx]
                        param_vals = profiler.x_path[param_idx, :]
                        nll_vals_rel = profiler.fval_path - float(nll_at_optimum)
                        sort_idx = np.argsort(param_vals)
                        pv, nr = param_vals[sort_idx], nll_vals_rel[sort_idx]
                        width = len(str(len(pv)))
                        for i, (v, n) in enumerate(zip(pv, nr)):
                            print(f"  [{i+1:{width}d}/{len(pv)}]  {pname}={v:.4g}  Δnll={n:.6g}")
                        return pv, nr
                    except ImportError:
                        print("pypesto not installed — falling back to likelihood slice")
                        return likelihood_slice_func(param_idx, n_points=n_points, range_factor=range_factor)
                    except Exception as e:
                        print(f"[true profile] failed ({e}) — falling back to likelihood slice")
                        return likelihood_slice_func(param_idx, n_points=n_points, range_factor=range_factor)

                out["stats"]["profile_likelihood"] = true_profile_likelihood_func

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
    fast=False,
    maxiter=None,
    tol=None,
):
    """
    Optimize shared parameters using ``experiment.opt_groups``.

    Each replicate in ``experiment.replicates`` carries an ``Opt_group`` key
    that controls group membership, and a ``Loss_config`` key that controls
    whether it contributes to the NLL objective:
      - ``no_optimization()`` → passive: simulated at end for plotting only
      - ``{"observables": [...]}`` → active: contributes to the NLL objective

    Parameters
    ----------
    experiment      : Experiment with .replicates and .opt_groups property.
    group_names     : opt_group names to include; None = all groups.
    method          : local scipy method name OR one of _GLOBAL_METHODS.
    optimizer_kwargs: extra kwargs forwarded to the chosen optimizer.
    """
    try:
        from scipy.optimize import minimize
    except ImportError:
        raise ImportError("scipy is required for run_optimization_from_groups") from None

    data_path = paths["data_path"]

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

    def _effective_loss_cfg(treatment):
        lc = treatment.get("Loss_config", no_optimization)
        return lc(treatment)

    # Pre-build one Tellurium model per replicate in selected groups.
    models     = {}
    replicates = {}
    for key, replicate in experiment.replicates.items():
        if replicate.get("Opt_group") not in selected_group_names:
            continue
        df_dict    = replicate["Data"](replicate, data_path)
        events_str = replicate["Events"](replicate, df_dict)
        r          = TelluriumGen(model_text + "\n" + events_str, paths)
        replicate["Update_parameters"](r, replicate, mode="Optimizer")
        models[key]     = {"r": r, "df_dict": df_dict}
        replicates[key] = replicate

    if not models:
        raise ValueError("No valid replicates found across selected groups.")

    # Objective: sum NLL over active replicates only.
    # First 3 calls print diagnostics to help identify time-alignment issues.
    _debug_calls = [0]

    def objective(x):
        do_debug = _debug_calls[0] < 3
        if do_debug:
            print(f"\n[opt debug] call #{_debug_calls[0] + 1}  "
                  + "  ".join(f"{n}={v:.4g}" for n, v in zip(param_names, x.tolist())))
        total_loss = 0.0
        for key, replicate in replicates.items():
            effective_lc = _effective_loss_cfg(replicate)
            if not effective_lc or not effective_lc.get("observables"):
                continue
            loss_val = loss_function(
                x, models[key]["r"], key, replicate,
                models[key]["df_dict"], param_names,
                loss_config=effective_lc,
            )
            if loss_val >= 1e10:
                _debug_calls[0] += 1
                return 1e10
            total_loss += loss_val
        if do_debug:
            print(f"  → total_loss = {total_loss:.6g}")
        _debug_calls[0] += 1
        return total_loss

    opt_kw = _prepare_optimizer_kwargs(method, optimizer_kwargs, fast, maxiter, tol)
    if method.lower() in _GLOBAL_METHODS:
        res = _run_global_optimization(objective, x0, bounds, method, opt_kw)
    else:
        res = minimize(objective, x0, method=method, bounds=bounds or [], **opt_kw)

    out = {
        "x": res.x, "fun": res.fun, "success": res.success,
        "message": res.message, "stats": {},
        "groups": sorted(selected_group_names),
        "nit":  getattr(res, "nit",  None),
        "nfev": getattr(res, "nfev", None),
    }

    if not res.success:
        print(f"Warning: optimizer reported non-convergence — {res.message}")
        print("Proceeding with best-found parameters for results and profile likelihood.")

    param_dict = dict(zip(param_names, res.x.tolist()))

    def set_params(r, p):
        set_parameters_from_dict(r, p)

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
                    sigma = (np.sqrt(np.sum(residuals**2) /
                                     max(1, n_block - k / max(1, len(observables_config))))
                             if n_block > 1 else 1e-6)
                fixed_sigmas[(key, obs)] = sigma
                total_n += len(residuals)

    # Simulate replicates NOT in selected groups at optimal params so that
    # plot functions receive a complete results_dict.
    for key, replicate in experiment.replicates.items():
        if replicate.get("Opt_group") in selected_group_names:
            continue
        df_dict    = replicate["Data"](replicate, data_path)
        events_str = replicate["Events"](replicate, df_dict)
        r          = TelluriumGen(model_text + "\n" + events_str, paths)
        replicate["Update_parameters"](r, replicate, mode="Optimizer")
        res_dict   = run_all(r, key, replicate, df_dict,
                             set_parameters=set_params, parameters=param_dict)
        best_results.update(res_dict)

    def nll_func_fixed(p):
        total_nll = 0.0
        for key, replicate in replicates.items():
            effective_lc = _effective_loss_cfg(replicate)
            if not effective_lc or not effective_lc.get("observables"):
                continue
            total_nll += loss_function(
                p, models[key]["r"], key, replicate,
                models[key]["df_dict"], param_names,
                effective_lc, fixed_sigmas=fixed_sigmas,
            )
        return total_nll

    out["results_dict"] = best_results

    if wald_analysis or slice_analysis or profile_likelihood_analysis:
        aic = 2 * k + 2 * res.fun
        bic = k * np.log(max(total_n, 1)) + 2 * res.fun
        out["stats"]["aic"] = aic
        out["stats"]["bic"] = bic

    if wald_analysis:
        try:
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
                print(f"    {_pname}: {nll_at_optimum:.6g} → {_nll_test:.6g}  (delta={_nll_test - nll_at_optimum:+.6g})")

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
                    try:
                        import pypesto
                        import pypesto.profile as pypesto_profile
                        import pypesto.optimize as pypesto_optimize
                        pname = param_names[param_idx]
                        print(f"\n[true profile] {pname}  (max {n_points} steps/side, re-optimizing nuisance params)")
                        objective = pypesto.Objective(fun=nll_func_fixed)
                        if bounds is not None:
                            lb = np.array([b[0] for b in bounds], dtype=float)
                            ub = np.array([b[1] for b in bounds], dtype=float)
                        else:
                            lb = res.x / (range_factor ** 2)
                            ub = res.x * (range_factor ** 2)
                        problem = pypesto.Problem(objective=objective, lb=lb, ub=ub, x_names=list(param_names))
                        pypesto_result = pypesto.Result(problem=problem)
                        pypesto_result.optimize_result.append(
                            pypesto.result.OptimizerResult(id='0', x=res.x.copy(), fval=float(nll_at_optimum)),
                            sort=True,
                        )
                        optimizer = pypesto_optimize.ScipyOptimizer(method='L-BFGS-B')
                        profile_options = pypesto_profile.ProfileOptions()
                        pypesto_profile.parameter_profile(
                            problem=problem, result=pypesto_result, optimizer=optimizer,
                            profile_index=np.array([param_idx]), result_index=0,
                            profile_options=profile_options,
                        )
                        profiler = pypesto_result.profile_result.list[0][param_idx]
                        param_vals = profiler.x_path[param_idx, :]
                        nll_vals_rel = profiler.fval_path - float(nll_at_optimum)
                        sort_idx = np.argsort(param_vals)
                        pv, nr = param_vals[sort_idx], nll_vals_rel[sort_idx]
                        width = len(str(len(pv)))
                        for i, (v, n) in enumerate(zip(pv, nr)):
                            print(f"  [{i+1:{width}d}/{len(pv)}]  {pname}={v:.4g}  Δnll={n:.6g}")
                        return pv, nr
                    except ImportError:
                        print("pypesto not installed — falling back to likelihood slice")
                        return likelihood_slice_func(param_idx, n_points=n_points, range_factor=range_factor)
                    except Exception as e:
                        print(f"[true profile] failed ({e}) — falling back to likelihood slice")
                        return likelihood_slice_func(param_idx, n_points=n_points, range_factor=range_factor)

                out["stats"]["profile_likelihood"] = true_profile_likelihood_func
        except Exception as e:
            print(f"Warning: likelihood analysis setup failed: {e}")

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
            test_result        = model.simulate(0, 1, 2, observed_species)
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
            result = model.simulate(0.0, max_time, n_points, variables_to_simulate)
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
    result   = model.simulate(0, times[-1], len(times)*1000, ['time', observed_var])
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

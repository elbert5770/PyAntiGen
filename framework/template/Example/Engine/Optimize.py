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

def run_all(
    r,
    exp_num,
    experiment,
    data_path,
    set_parameters=None,
    parameters=None,
):
    """
    Run one Tellurium simulation per experiment (using pre-setup 'r'), optionally
    applying parameters before simulate. Returns a list of result dicts for plotting
    or loss computation.
    """
    r.reset()
    results_dict = {}

    if set_parameters is not None and parameters is not None:
        set_parameters(r, parameters)

    solver_settings = experiment["Solver_settings"](experiment)
    observed_species = experiment["Observed_species"](r)
    results = simulate(r, solver_settings, observed_species)
    df_dict = experiment["Data"](experiment, data_path)
    label = experiment["Label"]
  
    results_dict[label] = {
        "results": results,
        "data": df_dict,
    }
    return results_dict

def set_parameters_from_dict(r, params):
    """
    Apply parameter dict to Tellurium model r.
    params: dict mapping model parameter id -> value.
    Uses roadrunner form: r[name] = value.
    
    """
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
    data_path,
    param_names,
    loss_config=None,
    fixed_sigmas=None,
):
    """
    Run all experiments with the given parameters and return a scalar loss
    (Negative Log Likelihood vs. loaded data).

    Args:
        params: Vector (list/array) of parameter values in the same order as param_names,
                or dict name -> value. If dict, param_names can be None.
        r: Tellurium object (pre-setup).
        experiment: Current experiment.
        data_path: Data directory.
        param_names: List of parameter ids corresponding to params vector; ignored if params is dict.
        loss_config: Optional dict, e.g. {"observables": [{"observed_variable": "[B_Comp1]", "data_column": "B", "time_column": "time", "data_dict_key": "B"}]}.
        fixed_sigmas: Optional dict mapping (experiment_id, observable) to a fixed sigma (used for Hessian calculation).

    Returns:
        Scalar NLL loss (float). Lower = better fit.
    """
    # print(params)
    loss_config = loss_config or {}
    time_column = loss_config.get("time_column", "time")
    
    observables_config = loss_config.get("observables", [])
    if not observables_config:
        raise ValueError(
            "loss_config must provide an 'observables' list specifying how to map Tellurium output to data columns.\n"
        )

    if isinstance(params, dict):
        param_dict = params
    else:
        param_dict = dict(zip(param_names, np.atleast_1d(params).tolist()))

    for param in param_names:
        print(param, "=", param_dict[param])

    # Prevent invalid parameter values if they fall into negative values
    try:
        p_vals = np.asarray(list(param_dict.values()))
        if np.any(p_vals <= 0):
            return 1e10
    except:
        pass

    def set_params(r, p):
        set_parameters_from_dict(r, p)

    try:
        results_dict = run_all(
            r,
            exp_num,
            experiment,
            data_path,
            set_parameters=set_params,
            parameters=param_dict,
        )
    except Exception as e:
        return 1e10

    total_loss = 0.0
    for i, (label, item) in enumerate(results_dict.items()):
        result = item["results"]
        df_dict = item["data"]
        exp_id = i
        
        # Build local dictionary for evaluating mathematical formulas
        local_dict = {"np": np, "time": np.asarray(result["time"])}
        if hasattr(result, "colnames"):
            cols = result.colnames
        elif hasattr(result, "dtype") and result.dtype.names:
            cols = result.dtype.names
        else:
            cols = []
            
        for col in cols:
            local_dict[col] = np.asarray(result[col])
            # If it's a bracketed concentration, provide stripped version for easy eval
            if col.startswith('[') and col.endswith(']'):
                local_dict[col[1:-1]] = np.asarray(result[col])

        # Expose all optimization parameters to the formula evaluation
        local_dict.update(param_dict)

        t_sim = np.asarray(result["time"])

        for obs_cfg in observables_config:
            obs = obs_cfg["observed_variable"]
            d_col = obs_cfg["data_column"]
            t_col = obs_cfg["time_column"]

            if isinstance(df_dict, dict):
                data_key = obs_cfg.get("data_dict_key")
                if data_key is not None:
                    obs_df = df_dict.get(data_key)
                else:
                    obs_df = next((v for v in df_dict.values() if d_col in v.columns and t_col in v.columns), None)
                if obs_df is None:
                    continue
            else:
                obs_df = df

            if d_col not in obs_df.columns or t_col not in obs_df.columns:
                continue

            y_data = np.asarray(obs_df[d_col])
            t_data = np.asarray(obs_df[t_col])
            # Evaluate the observable (string column, string formula, or callable)
            try:
                if callable(obs):
                    y_sim = np.asarray(obs(result))
                elif isinstance(obs, str) and obs in cols:
                    y_sim = np.asarray(result[obs])
                elif isinstance(obs, str):
                    # Prevent brackets from being parsed as Python lists
                    eval_obs = str(obs)
                    for col in cols:
                        if col.startswith('[') and col.endswith(']'):
                            eval_obs = eval_obs.replace(col, col[1:-1])
                    y_sim = np.asarray(eval(eval_obs, {}, local_dict))
                else:
                    raise ValueError(f"Invalid observable type: {type(obs)}")
            except Exception as e:
                raise RuntimeError(f"Failed to evaluate observable '{obs}': {e}") from e

            # Clean NaNs and infs which often occur due to division by zero at t=0
            y_sim = np.nan_to_num(y_sim, nan=0.0, posinf=0.0, neginf=0.0)

            # Interpolate simulation at data timepoints
            y_pred = np.interp(t_data, t_sim, y_sim)
            residuals = y_data - y_pred

            # Determine sigma
            if fixed_sigmas is not None and (exp_id, obs) in fixed_sigmas:
                sigma = fixed_sigmas[(exp_id, obs)]
            else:
                sigma_config = obs_cfg.get("noise_formula", None)
                if sigma_config and sigma_config in local_dict:
                    # Parameterized sigma
                    sigma = float(local_dict[sigma_config])
                else:
                    # Dynamic empirical variance
                    sigma2 = np.var(residuals)
                    sigma = np.sqrt(sigma2) if sigma2 > 1e-12 else 1e-6
            
            # Use scipy.stats.norm for likelihood calculation
            nll = -np.sum(stats.norm.logpdf(y_data, loc=y_pred, scale=sigma))
            total_loss += nll
    print("total_loss=", total_loss)
    return float(total_loss)

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
        params_plus = np.array(params, copy=True)
        params_minus = np.array(params, copy=True)
        step = epsilon * max(abs(params[i]), 1.0)
        params_plus[i] += step
        params_minus[i] -= step
        f_plus = func(params_plus)
        f_minus = func(params_minus)
        hessian[i, i] = (f_plus - 2*f0 + f_minus) / (step**2)
    for i in range(n):
        for j in range(i+1, n):
            params_pp = np.array(params, copy=True)
            params_pm = np.array(params, copy=True)
            params_mp = np.array(params, copy=True)
            params_mm = np.array(params, copy=True)
            step_i = epsilon * max(abs(params[i]), 1.0)
            step_j = epsilon * max(abs(params[j]), 1.0)
            params_pp[i] += step_i; params_pp[j] += step_j
            params_pm[i] += step_i; params_pm[j] -= step_j
            params_mp[i] -= step_i; params_mp[j] += step_j
            params_mm[i] -= step_i; params_mm[j] -= step_j
            f_pp = func(params_pp)
            f_pm = func(params_pm)
            f_mp = func(params_mp)
            f_mm = func(params_mm)
            hessian[i, j] = (f_pp - f_pm - f_mp + f_mm) / (4 * step_i * step_j)
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

def run_optimization(
    model_text,
    paths,
    experiments,
    param_names,
    x0,
    bounds=None,
    loss_config=None,
    likelihood_analysis=False,
    method="Nelder-Mead",
):
    try:
        from scipy.optimize import minimize
    except ImportError:
        raise ImportError("scipy is required for run_optimization") from None

    data_path = paths["data_path"]
    
    # Pre-setup the tellurium problem for each experiment
    models = {}
    for exp_num, experiment in experiments.items():
        df_dict = experiment["Data"](experiment, data_path)
        events_str = experiment["Events"](experiment,df_dict)
        full_model_text = model_text + "\n" + events_str
        r = TelluriumGen(full_model_text, paths)
        models[exp_num] = r

    def objective(x):
        total_loss = 0.0
        for exp_num, experiment in experiments.items():
            r = models[exp_num]
            loss_val = loss_function(
                x,
                r,
                exp_num,
                experiment,
                data_path,
                param_names,
                loss_config=loss_config,
            )
            if loss_val >= 1e10:
                return 1e10
            total_loss += loss_val
        return total_loss

    res = minimize(objective, x0, method=method, bounds=bounds or [])
    out = {"x": res.x, "fun": res.fun, "success": res.success, "message": res.message, "stats": {}}
    
    if res.success:
        param_dict = dict(zip(param_names, res.x.tolist()))

        def set_params(r, p):
            set_parameters_from_dict(r, p)

        # Run at optimal params to get results and calculate empirical sigmas
        best_results = {}
        fixed_sigmas = {}
        total_n = 0
        k = len(param_names)
        
        loss_config_safe = loss_config or {}
        observables_config = loss_config_safe.get("observables", [])
        
        for exp_num, experiment in experiments.items():
            r = models[exp_num]
            res_dict = run_all(
                r,
                exp_num,
                experiment,
                data_path,
                set_parameters=set_params,
                parameters=param_dict,
            )
            best_results.update(res_dict)

            for i, (label, item) in enumerate(res_dict.items()):
                df_dict = item["data"]
                result = item["results"]
                exp_id = exp_num
                t_sim = np.asarray(result["time"])
                
                # Setup local dictionary for evaluating mathematical formulas
                local_dict = {"np": np, "time": t_sim}
                cols = result.colnames if hasattr(result, "colnames") else (result.dtype.names if hasattr(result, "dtype") else [])
                for c in cols:
                    local_dict[c] = np.asarray(result[c])
                    if c.startswith('[') and c.endswith(']'):
                        local_dict[c[1:-1]] = np.asarray(result[c])
                local_dict.update(param_dict)
                
                for obs_cfg in observables_config:
                    obs = obs_cfg["observed_variable"]
                    d_col = obs_cfg["data_column"]
                    t_col = obs_cfg["time_column"]

                    if isinstance(df_dict, dict):
                        data_key = obs_cfg.get("data_dict_key")
                        if data_key is not None:
                            obs_df = df_dict.get(data_key)
                        else:
                            obs_df = next((v for v in df_dict.values() if d_col in v.columns and t_col in v.columns), None)
                        if obs_df is None:
                            continue
                    else:
                        obs_df = df

                    if d_col not in obs_df.columns or t_col not in obs_df.columns:
                        continue
                    y_data = np.asarray(obs_df[d_col])
                    t_data = np.asarray(obs_df[t_col])
                    
                    # evaluate observable
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

                    # Clean NaNs and infs which often occur due to division by zero at t=0
                    y_sim = np.nan_to_num(y_sim, nan=0.0, posinf=0.0, neginf=0.0)
                    
                    y_pred = np.interp(t_data, t_sim, y_sim)
                    residuals = y_data - y_pred
                    sigma_config = obs_cfg.get("noise_formula", None)
                    if sigma_config and sigma_config in local_dict:
                        sigma = float(local_dict[sigma_config])
                    else:
                        n_block = len(residuals)
                        # use n-k approximation for standard error if possible, but safely fallback to simple var
                        sigma = np.sqrt(np.sum(residuals**2) / max(1, n_block - k/max(1, len(observables_config)))) if n_block > 1 else 1e-6
                    fixed_sigmas[(exp_id, obs)] = sigma
                    total_n += len(residuals)
                
        # Statistical functions
        def nll_func_fixed(p):
            total_nll = 0.0
            for exp_num, experiment in experiments.items():
                r = models[exp_num]
                total_nll += loss_function(
                    p, r, exp_num, experiment, data_path,
                    param_names, loss_config, fixed_sigmas=fixed_sigmas
                )
            return total_nll

        out["results_dict"] = best_results

        # 1. Hessian and Fisher Info
        if likelihood_analysis:
            try:
                hessian_raw = compute_hessian_numdifftools(nll_func_fixed, res.x)
                fisher_info = (hessian_raw + hessian_raw.T) / 2.0
                eigenvalues = np.linalg.eigvals(fisher_info)
                if np.any(eigenvalues <= 1e-10):
                    print("Warning: Hessian is NOT positive definite.")
                    cov_matrix = None
                    standard_errors = None
                else:
                    cov_matrix = np.linalg.inv(fisher_info)
                    diag_elements = np.diag(cov_matrix)
                    if np.any(diag_elements < 0):
                        standard_errors = None
                    else:
                        standard_errors = np.sqrt(diag_elements)
                out["stats"]["fisher_info"] = fisher_info
                out["stats"]["covariance_matrix"] = cov_matrix
                out["stats"]["standard_errors"] = standard_errors
                
                # Confidence Intervals and Correlations
                if standard_errors is not None:
                    alpha = 1 - 0.95
                    cv = stats.norm.ppf(1 - alpha/2)
                    ci = []
                    for p_val, se in zip(res.x, standard_errors):
                        ci.append((p_val - cv * se, p_val + cv * se))
                    out["stats"]["confidence_intervals"] = ci
                
                if cov_matrix is not None:
                    corr = compute_parameter_correlations(cov_matrix)
                    out["stats"]["correlation_matrix"] = corr
            except Exception as e:
                print(f"Error computing advanced statistics: {e}")
                
            # AIC/BIC
            aic = 2 * k + 2 * res.fun
            bic = k * np.log(total_n) + 2 * res.fun
            out["stats"]["aic"] = aic
            out["stats"]["bic"] = bic
            
            # Profile Likelihood capability (callable closure)
            def profile_likelihood_func(param_idx, n_points=20, range_factor=2.0):
                p_val = res.x[param_idx]
                param_vals = np.linspace(p_val / range_factor, p_val * range_factor, n_points)
                nll_vals = []
                for val in param_vals:
                    p_test = res.x.copy()
                    p_test[param_idx] = val
                    nll_vals.append(nll_func_fixed(p_test))
                return param_vals, np.array(nll_vals)
                
            out["stats"]["profile_likelihood"] = profile_likelihood_func

    out["r"] = list(models.values())[0] if models else None
    return out

def predict_concentrations(params, observable_data, model, param_names, observable_defs=None):
    """
    Simulate the model with given parameters and return predicted concentrations for multiple observables.
    
    Args:
        params: array of parameter values to estimate
        observable_data: dict mapping observable_id -> {'times': array, 'values': array (optional)}
        model: Tellurium model object (roadrunner instance)
        param_names: list of parameter names to set
        observable_defs: dict mapping observable_id -> observable definition dict (with 'observableFormula')
    
    Returns:
        dict mapping observable_id -> predicted values array
    """
    # Set parameters in the model
    model.reset()
    for name, value in zip(param_names, params):
        model[name] = value

    # Configure integrator to match Simulate.py settings (prevents CVODE failures)
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

    # Handle both new dict format and legacy format
    if isinstance(observable_data, dict):
        # New format: dict of observable_id -> {'times': array, ...}
        predictions = {}
        # Get all unique observables and their max times
        all_observables = list(observable_data.keys())
        all_times = [obs_data['times'] for obs_data in observable_data.values()]
        if not all_times or all([len(t) == 0 for t in all_times]):
            return {obs_id: np.array([]) for obs_id in all_observables}
        
        max_time = float(max([t.max() if len(t) > 0 else 0 for t in all_times]))
        min_time = 0.0 # Force start from 0 to capture events properly
        
        # Get available variables from the model (species, assignment rules, etc.)
        # The best way is to do a test simulation and see what columns are available
        try:
            observed_species = ['time'] + list(model.getFloatingSpeciesIds()) + list(model.getBoundarySpeciesIds()) + list(model.getAssignmentRuleIds())
            # for p in ['SF', 'PlasmaLeu']:
            #     if p in model.keys() and p not in observed_species:
            #         observed_species.append(p)
            test_result = model.simulate(0, 1, 2, observed_species)
            available_variables = set(test_result.colnames) - {'time'}
        except:
            # Fallback: try to get species IDs and assignment rules
            try:
                floating_species = list(model.getFloatingSpeciesIds())
                boundary_species = list(model.getBoundarySpeciesIds())
                assignment_rules = list(model.getAssignmentRuleIds())
                available_variables = set(floating_species + boundary_species + assignment_rules)
            except:
                available_variables = set()
                warnings.warn("Could not determine available variables from model")
        
        # Collect all variables needed for formulas
        variables_to_simulate = ['time']
        formulas_to_evaluate = {}
        parameters_in_formulas = set()  # Track parameters separately
        
        for obs_id in all_observables:
            if observable_defs and obs_id in observable_defs:
                formula = observable_defs[obs_id].get('observableFormula', obs_id)
                # Check if the observable ID itself exists as a variable in the model
                # (could be a species, assignment rule, etc.)
                if obs_id in available_variables:
                    # The observable is directly available - simulate it
                    if obs_id not in variables_to_simulate:
                        variables_to_simulate.append(obs_id)
                else:
                    # Observable not directly available - need to evaluate formula
                    formulas_to_evaluate[obs_id] = formula
                    # Try to extract variable names from formula (simple heuristic)
                   
                    import re
                    # Match identifiers (words that could be variables/parameters)
                    var_pattern = r'\b([A-Za-z_][A-Za-z0-9_]*)\b'
                    formula_vars = re.findall(var_pattern, formula)
                    for var in formula_vars:
                        # Skip Python keywords and common operators
                        if var not in ['time', 'and', 'or', 'not', 'if', 'else', 'True', 'False', 'None', 'np', 'numpy']:
                            if var in available_variables:
                                # It's an available variable - add to simulation
                                if var not in variables_to_simulate:
                                    variables_to_simulate.append(var)
                            else:
                                # Might be a parameter - track it for later access
                                parameters_in_formulas.add(var)
            else:
                # No formula defined - check if observable ID exists as variable
                if obs_id in available_variables:
                    if obs_id not in variables_to_simulate:
                        variables_to_simulate.append(obs_id)
                else:
                    # Not available - treat as formula (use ID as formula)
                    formulas_to_evaluate[obs_id] = obs_id
        
        # Simulate once for all needed species
        n_points = max(10000, sum(len(t) for t in all_times) * 100)  # Enough points for interpolation
        try:
            result = model.simulate(min_time, max_time, n_points, variables_to_simulate)
        except Exception as e:
            warnings.warn(f"Simulation error: {e}. Trying with just time.")
            try:
                res_time = model.simulate(min_time, max_time, n_points, ['time'])
                # Convert to dict since NamedArray is immutable
                result = {'time': res_time['time']}
                # Add species one by one
                for var in variables_to_simulate[1:]:
                    try:
                        result_var = model.simulate(min_time, max_time, n_points, ['time', var])
                        result[var] = result_var[var]
                    except Exception as var_err:
                        warnings.warn(f"Could not simulate {var}: {var_err}")
            except Exception as outer_err:
                # Complete failure, could not even simulate time
                warnings.warn(f"Complete simulation failure: {outer_err}. Returning zeros.")
                for obs_id, obs_data in observable_data.items():
                    predictions[obs_id] = np.zeros_like(obs_data['times'])
                return predictions
        
        # Interpolate for each observable at its specific time points
        time_sim = result['time']
        result_cols = set(getattr(result, 'colnames', result.keys() if hasattr(result, 'keys') else []))
        for obs_id, obs_data in observable_data.items():
            times = obs_data['times']

            # Check if we need to evaluate a formula
            if obs_id in formulas_to_evaluate:
                formula = formulas_to_evaluate[obs_id]
                try:
                    # Evaluate formula using simulation results
                    # Create a namespace with all simulation variables
                    cols = getattr(result, 'colnames', result.keys())
                    namespace = {col: result[col] for col in cols}
                    
                    # Add model parameters that might be in formula
                    # First, try parameters from param_names (estimated parameters)
                    for param_name in param_names:
                        if param_name not in namespace:
                            try:
                                namespace[param_name] = model[param_name]
                            except:
                                pass
                    
                    # Also add any other parameters found in formulas
                    for param_name in parameters_in_formulas:
                        if param_name not in namespace:
                            try:
                                namespace[param_name] = model[param_name]
                            except:
                                pass
                    
                    # Try to get any remaining parameters from the model
                    try:
                        global_params = model.getGlobalParameterIds()
                        for param_id in global_params:
                            if param_id not in namespace:
                                try:
                                    namespace[param_id] = model[param_id]
                                except:
                                    pass
                    except:
                        pass
                    
                    # Evaluate formula - handle both array and scalar operations
                    # Use numpy operations
                    namespace.update({'np': np, 'numpy': np})
                    
                    # Evaluate formula
                    obs_values = eval(formula, namespace, {})
                    
                    # If result is scalar, make it an array
                    if np.isscalar(obs_values):
                        obs_values = np.full_like(time_sim, obs_values)
                    
                    predictions[obs_id] = np.interp(times, time_sim, obs_values)
                except Exception as e:
                    warnings.warn(f"Could not evaluate formula for {obs_id}: {formula}. Error: {e}")
                    predictions[obs_id] = np.zeros_like(times)
            elif obs_id in result_cols:
                # Direct variable access
                conc_sim = result[obs_id]
                predictions[obs_id] = np.interp(times, time_sim, conc_sim)
            else:
                warnings.warn(f"Observable {obs_id} not found in simulation results and no formula provided")
                predictions[obs_id] = np.zeros_like(times)
        
        return predictions
    else:
        # Legacy format: single observable
        if isinstance(observable_data, (list, tuple)) and len(observable_data) == 2:
            observed_var, times = observable_data
        else:
            # Assume observable_data is just the variable name (backward compatibility)
            observed_var = observable_data
            times = None
            raise ValueError("Legacy format requires times to be provided separately")
        
        # Simulate - use more time points for smooth interpolation
        if times is None or len(times) == 0:
            return np.array([])
        result = model.simulate(0, times[-1], len(times)*1000, ['time', observed_var])
        
        # Interpolate at observed times using np.interp
        time_sim = result['time']
        conc_sim = result[observed_var]
        return np.interp(times, time_sim, conc_sim)

def neg_log_likelihood(params, observable_data, model, param_names, observable_defs=None):
    """
    Negative log-likelihood assuming normally distributed errors.
    Supports multiple observables, each with their own time points.
    
    Args:
        params: array of parameter values to estimate
        observable_data: dict mapping observable_id -> {'times': array, 'values': array}
        model: Tellurium model object (roadrunner instance)
        param_names: list of parameter names to set
        observable_defs: dict mapping observable_id -> observable definition dict (with 'observableFormula')
    
    Returns: -log L(params)
    """
    # Prevent invalid parameter values
    if np.any(params <= 0):
        return 1e10
    try:
        # Get predictions for all observables
        predictions = predict_concentrations(params, observable_data, model, param_names, observable_defs)
        
        # Sum likelihood over all observables
        total_nll = 0.0
        for obs_id, obs_data in observable_data.items():
            times = obs_data['times']
            values = obs_data['values']
            pred = predictions[obs_id]
            
            residuals = values - pred
            # Using scipy.stats for likelihood calculation
            # Assuming constant variance (can be extended to include sigma as parameter)
            sigma2 = np.var(residuals)
            if sigma2 <= 0:
                sigma2 = 1e-6
            # Use scipy.stats.norm for likelihood calculation
            nll = -np.sum(stats.norm.logpdf(values, loc=pred, scale=np.sqrt(sigma2)))
            total_nll += nll
        
        return total_nll
    except Exception as e:
        warnings.warn(f"Error in neg_log_likelihood: {e}")
        return 1e10

def neg_log_likelihood_fixed_sigma(params, observable_data, model, param_names, sigmas, observable_defs=None):
    """
    Negative log-likelihood with FIXED sigma for Hessian computation.
    This ensures the function is smooth for numerical differentiation.
    
    Args:
        params: array of parameter values to estimate
        observable_data: dict mapping observable_id -> {'times': array, 'values': array}
        model: Tellurium model object (roadrunner instance)
        param_names: list of parameter names to set
        sigmas: dict mapping observable_id -> sigma value, or single float for all observables
        observable_defs: dict mapping observable_id -> observable definition dict (with 'observableFormula')
    """
    # Prevent invalid parameter values
    if np.any(params <= 0):
        return 1e10
    try:
        # Get predictions for all observables
        predictions = predict_concentrations(params, observable_data, model, param_names, observable_defs)
        
        # Sum likelihood over all observables
        total_nll = 0.0
        for obs_id, obs_data in observable_data.items():
            values = obs_data['values']
            pred = predictions[obs_id]
            
            # Get sigma for this observable
            if isinstance(sigmas, dict):
                sigma = sigmas.get(obs_id, 1e-6)
            else:
                sigma = sigmas  # Single value for all
            
            # Use fixed sigma, not estimated from residuals
            nll = -np.sum(stats.norm.logpdf(values, loc=pred, scale=sigma))
            total_nll += nll
        
        return total_nll
    except Exception as e:
        warnings.warn(f"Error in neg_log_likelihood_fixed_sigma: {e}")
        return 1e10

def profile_likelihood(param_idx, params_estimated, observable_data, model, param_names, sigmas, observable_defs=None, n_points=20, range_factor=2.0):
    """
    Profile the likelihood for a single parameter to check identifiability.
    Uses FIXED sigma to match Hessian calculation.
    Supports multiple observables.
    Returns: parameter values and corresponding NLL values
    """
    param_values = np.linspace(
        params_estimated[param_idx] / range_factor,
        params_estimated[param_idx] * range_factor,
        n_points
    )
    
    nll_values = []
    for val in param_values:
        params_test = params_estimated.copy()
        params_test[param_idx] = val
        nll = neg_log_likelihood_fixed_sigma(params_test, observable_data, model, param_names, sigmas, observable_defs)
        nll_values.append(nll)
    
    return param_values, np.array(nll_values)

def _plot_profile_likelihood(ax, plot_config, opt, params_estimated, param_names):
    """Plot profile likelihood for parameters."""
    
    # Get parameters to profile
    params_to_profile = plot_config.get('parameters', param_names)
    n_points = plot_config.get('n_points', 30)
    range_factor = plot_config.get('range_factor', 3.0)
    
    colors = ['blue', 'green', 'red', 'orange', 'purple']
    markers = ['o', 's', '^', 'D', 'v']
    
    profile_func = opt.get("stats", {}).get("profile_likelihood")
    if not profile_func:
        print("Warning: no profile_likelihood closure found in opt['stats']")
        return

    result_fun = opt.get("fun", 0)

    for idx, param_name in enumerate(params_to_profile):
        if param_name not in param_names:
            continue
        
        param_idx = param_names.index(param_name)
        param_vals, nll_vals = profile_func(param_idx, n_points=n_points, range_factor=range_factor)
        
        # Normalize
        param_vals_normalized = param_vals / params_estimated[param_idx]
        nll_vals_rel = nll_vals - result_fun
        
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
        ax.plot(param_vals_normalized, nll_vals_rel, 
               marker=marker, linestyle='-', label=param_name, linewidth=2, color=color)
    
    # Add vertical line at optimal value
    ax.axvline(1.0, color='red', linestyle='--', alpha=0.5, linewidth=1.5, label='Optimal')
    
    ax.set_xlabel(plot_config.get('xlabel', 'Parameter Value (relative to optimal)'))
    ax.set_ylabel(plot_config.get('ylabel', 'Δ NLL (relative to minimum)'))
    ax.set_title(plot_config.get('title', 'Profile Likelihood (Identifiability Check)'))
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    # ax.set_ylim(bottom=0)
    
    # Set xlim if specified
    if 'xlim' in plot_config:
        ax.set_xlim(plot_config['xlim'])
    else:
        ax.set_xlim(0.2, 3.5)

"""
Parameter optimization over multiple experiments: run each experiment with a
given parameter set, compare to data, return a scalar loss for the optimizer.
Also provides run_all for running experiments (with optional parameter injection).
"""

import numpy as np
from .AntimonyGen import TelluriumGen
from .Simulate import simulate


def run_all(
    model_text,
    MODEL_NAME,
    repo_root,
    experiment_specs,
    data_path,
    set_parameters=None,
    parameters=None,
):
    """
    Run one Tellurium simulation per experiment (model_text + events), optionally
    applying parameters before simulate. Returns a list of result dicts for plotting
    or loss computation.

    Args:
        model_text: Base Antimony model string (no experiment-specific events).
        MODEL_NAME: Model name (e.g. folder name).
        repo_root: Repo root for TelluriumGen.
        experiment_specs: List of dicts with keys: id, event_func, load_data, label (optional).
        data_path: Path to data directory (passed to each load_data(spec)(data_path)).
        set_parameters: Optional callable(r, params). If given, called before each simulate with (r, parameters).
        parameters: Optional dict (or list) of parameter values; used only when set_parameters is provided.

    Returns:
        List of dicts: [{"id", "result", "data", "label"}, ...].
    """
    results = []

    for spec in experiment_specs:
        event_func = spec["event_func"]
        load_data = spec["load_data"]
        events = event_func()
        full_model_text = model_text + "\n" + events
        r = TelluriumGen(full_model_text, MODEL_NAME, repo_root)

        if set_parameters is not None and parameters is not None:
            set_parameters(r, parameters)

        result = simulate(r)
        df = load_data(data_path)

        results.append({
            "id": spec["id"],
            "result": result,
            "data": df,
            "label": spec.get("label", spec["id"]),
        })

    return results


def set_parameters_from_dict(r, params):
    """
    Apply parameter dict to Tellurium model r.
    params: dict mapping model parameter id -> value.
    Uses roadrunner form: r[name] = value.
    """
    for name, value in params.items():
        try:
            r[name] = value
        except Exception as e:
            raise RuntimeError(f"Failed to set parameter '{name}': {e}") from e


def loss_function(
    params,
    model_text,
    MODEL_NAME,
    repo_root,
    experiment_specs,
    data_path,
    param_names,
    loss_config=None,
):
    """
    Run all experiments with the given parameters and return a scalar loss
    (e.g. sum of squared errors vs. loaded data).

    Args:
        params: Vector (list/array) of parameter values in the same order as param_names,
                or dict name -> value. If dict, param_names can be None.
        model_text: Base Antimony model string.
        MODEL_NAME: Model name.
        repo_root: Repo root for TelluriumGen.
        experiment_specs: Same list as used in Experiment.run_all.
        data_path: Data directory.
        param_names: List of parameter ids corresponding to params vector; ignored if params is dict.
        loss_config: Optional dict, e.g. {"observable": "[B_Comp1]", "data_column": "B", "time_column": "time"}.

    Returns:
        Scalar loss (float). Lower = better fit.
    """
    loss_config = loss_config or {}
    observable = loss_config.get("observable", "[B_Comp1]")
    data_column = loss_config.get("data_column", "B")
    time_column = loss_config.get("time_column", "time")

    if isinstance(params, dict):
        param_dict = params
    else:
        param_dict = dict(zip(param_names, np.atleast_1d(params).tolist()))

    def set_params(r, p):
        set_parameters_from_dict(r, p)

    results = run_all(
        model_text,
        MODEL_NAME,
        repo_root,
        experiment_specs,
        data_path,
        set_parameters=set_params,
        parameters=param_dict,
    )

    total_loss = 0.0
    for item in results:
        result = item["result"]
        df = item["data"]
        if time_column not in df.columns or data_column not in df.columns:
            continue
        t_data = np.asarray(df[time_column])
        y_data = np.asarray(df[data_column])
        t_sim = np.asarray(result["time"])
        y_sim = np.asarray(result[observable])
        # Interpolate simulation at data timepoints
        y_pred = np.interp(t_data, t_sim, y_sim)
        total_loss += np.sum((y_data - y_pred) ** 2)

    return float(total_loss)


def run_optimization(
    model_text,
    MODEL_NAME,
    repo_root,
    experiment_specs,
    data_path,
    param_names,
    x0,
    bounds=None,
    loss_config=None,
    method="L-BFGS-B",
):
    """
    Run a local optimizer to minimize loss over parameters. Returns best params
    and optional full results for plotting.

    Args:
        model_text, MODEL_NAME, repo_root, experiment_specs, data_path: As in loss_function.
        param_names: List of parameter ids.
        x0: Initial parameter vector (same length as param_names).
        bounds: Optional list of (low, high) per parameter; required for L-BFGS-B if bounds exist.
        loss_config: Optional dict for loss_function.
        method: scipy.optimize.minimize method.

    Returns:
        dict with "x" (best params), "fun" (loss), "success", "message", and optionally "results"
        from Experiment.run_all at best params (if you re-run and attach).
    """
    try:
        from scipy.optimize import minimize
    except ImportError:
        raise ImportError("scipy is required for run_optimization") from None

    def objective(x):
        return loss_function(
            x,
            model_text,
            MODEL_NAME,
            repo_root,
            experiment_specs,
            data_path,
            param_names,
            loss_config=loss_config,
        )

    res = minimize(objective, x0, method=method, bounds=bounds or [])
    out = {"x": res.x, "fun": res.fun, "success": res.success, "message": res.message}
    # Optionally attach results at best params for plotting
    if res.success:
        param_dict = dict(zip(param_names, res.x.tolist()))

        def set_params(r, p):
            set_parameters_from_dict(r, p)

        out["results"] = run_all(
            model_text,
            MODEL_NAME,
            repo_root,
            experiment_specs,
            data_path,
            set_parameters=set_params,
            parameters=param_dict,
        )
    return out

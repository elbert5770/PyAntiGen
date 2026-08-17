import os
import sys
import numpy as np
import matplotlib.pyplot as plt

try:
    from SALib.sample import saltelli
    from SALib.analyze import sobol
    _SALIB_AVAILABLE = True
except ImportError:
    _SALIB_AVAILABLE = False


def run_sobol_analysis(nll_func, param_names, bounds, res_x, N=128, range_factor=2.0,
                       mode='loss', nll_batch=None):
    """
    Run Sobol Global Sensitivity Analysis on the Negative Log-Likelihood (NLL).
    This tests the identifiability of parameters against the experimental data.
    
    Parameters
    ----------
    nll_func : callable
        Function taking a parameter array and returning a scalar loss (NLL).
    param_names : list
        Names of the parameters being optimized.
    bounds : list of tuples
        Bounds for each parameter.
    res_x : array
        The nominal / optimized parameter values.
    N : int
        Number of Saltelli samples (generates N * (2D + 2) evaluations).
    range_factor : float
        If bounds are missing, use res_x / range_factor to res_x * range_factor.
    mode : str
        Currently supports 'loss' (identifiability). 'output' mode can be added for specific observables.
    """
    if not _SALIB_AVAILABLE:
        print("Warning: SALib not installed. Skipping Sobol analysis.")
        return None

    if mode != 'loss':
        raise NotImplementedError(f"Sobol mode '{mode}' is not yet implemented. Use 'loss'.")

    problem = {
        'num_vars': len(param_names),
        'names': param_names,
        'bounds': []
    }
    
    for i, p in enumerate(param_names):
        if bounds and bounds[i] is not None:
            problem['bounds'].append(list(bounds[i]))
        else:
            opt_val = res_x[i]
            lb = opt_val / range_factor
            ub = opt_val * range_factor
            if lb > ub: lb, ub = ub, lb
            if lb == ub: ub = lb + 1e-6
            problem['bounds'].append([lb, ub])
            
    print(f"\n[Sobol {mode.capitalize()}] Generating Saltelli samples (N={N}) for {len(param_names)} parameters...")
    param_values = saltelli.sample(problem, N, calc_second_order=True)
    num_samples = param_values.shape[0]
    
    print(f"[Sobol {mode.capitalize()}] Evaluating {num_samples} samples...")

    import time
    t0 = time.time()

    if nll_batch is not None:
        # Saltelli samples are independent by construction, so the whole design
        # goes out as one batch -- the single largest parallel win available here.
        Y = np.asarray(nll_batch(list(param_values), label="sobol"), dtype=float)
        print(f"[Sobol {mode.capitalize()}] {num_samples} samples in "
              f"{time.time() - t0:.1f}s")
    else:
        Y = np.zeros(num_samples)
        for i, sample in enumerate(param_values):
            try:
                Y[i] = nll_func(sample)
            except Exception:
                Y[i] = np.nan

            if (i+1) % 100 == 0 or (i+1) == num_samples:
                elapsed = time.time() - t0
                rate = (i+1) / elapsed if elapsed > 0 else 0
                print(f"  [{i+1}/{num_samples}]  {elapsed:.1f}s  ({rate:.1f} evals/s)", end='\r', flush=True)

    print(f"\n[Sobol {mode.capitalize()}] Done. Running SALib analysis...")
    
    valid_mask = np.isfinite(Y)
    if not np.all(valid_mask):
        n_invalid = np.sum(~valid_mask)
        print(f"  Warning: {n_invalid} invalid evaluations (NaN/Inf). Setting to max valid loss.")
        max_valid = np.nanmax(Y[valid_mask]) if np.any(valid_mask) else 1e10
        Y[~valid_mask] = max_valid
        
    Si = sobol.analyze(problem, Y, calc_second_order=True, print_to_console=False)
    
    results = {
        "S1": Si["S1"],
        "ST": Si["ST"],
        "S1_conf": Si["S1_conf"],
        "ST_conf": Si["ST_conf"],
        "names": param_names,
        "type": mode,
        "N": N,
        "num_samples": num_samples
    }
    
    return results


def save_sobol_plot(sobol_results, plot_path, model_name, tag="ALL"):
    """
    Generate horizontal bar charts for S1 and ST indices and save summary text.
    """
    if sobol_results is None:
        return
        
    os.makedirs(plot_path, exist_ok=True)
    
    names = sobol_results["names"]
    mode = sobol_results["type"]
    
    S1 = sobol_results["S1"]
    ST = sobol_results["ST"]
    S1_conf = sobol_results.get("S1_conf", np.zeros_like(S1))
    ST_conf = sobol_results.get("ST_conf", np.zeros_like(ST))
    
    # Sort by Total Sensitivity (ST)
    sort_idx = np.argsort(ST)
    sorted_names = [names[i] for i in sort_idx]
    sorted_S1 = S1[sort_idx]
    sorted_ST = ST[sort_idx]
    sorted_S1_conf = S1_conf[sort_idx]
    sorted_ST_conf = ST_conf[sort_idx]
    
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 5 + 0.3 * len(names)))
    y_pos = np.arange(len(names))
    
    ax.barh(y_pos, sorted_ST, xerr=sorted_ST_conf, color='skyblue', label='Total Order (ST)', alpha=0.8)
    ax.barh(y_pos, sorted_S1, xerr=sorted_S1_conf, color='orange', label='First Order (S1)', alpha=0.8)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_names)
    ax.set_xlabel('Sensitivity Index')
    title_mode = "Loss Identifiability" if mode == "loss" else "Model Output"
    ax.set_title(f'Sobol Parameter Sensitivity ({title_mode})')
    ax.legend()
    ax.grid(axis='x', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    out_file = os.path.join(plot_path, f"{model_name}_{tag}_sobol_{mode}.png")
    plt.savefig(out_file, dpi=300)
    plt.close(fig)
    
    # Text Summary
    txt_file = os.path.join(plot_path, f"{model_name}_{tag}_sobol_{mode}_summary.txt")
    with open(txt_file, "w") as f:
        f.write(f"Sobol Global Sensitivity Analysis ({title_mode})\n")
        f.write("========================================================\n")
        f.write(f"N (Samples): {sobol_results['N']}\n")
        f.write(f"Total Evaluations: {sobol_results['num_samples']}\n\n")
        f.write(f"{'Parameter':<35} | {'ST':<15} | {'S1':<15}\n")
        f.write("-" * 75 + "\n")
        
        identifiable = []
        unidentifiable = []
        
        # Reverse sort for text file (highest sensitivity first)
        for i in sort_idx[::-1]:
            st_val = ST[i]
            s1_val = S1[i]
            f.write(f"{names[i]:<35} | {st_val:<15.5f} | {s1_val:<15.5f}\n")
            if st_val > 0.01:
                identifiable.append(names[i])
            else:
                unidentifiable.append(names[i])
                
        f.write("\n\n=== IDENTIFIABILITY SUMMARY (ST > 0.01) ===\n")
        f.write(f"Highly Influential / Identifiable ({len(identifiable)}):\n")
        for p in identifiable: f.write(f"  - {p}\n")
        
        f.write(f"\nNon-Influential / Unidentifiable ({len(unidentifiable)}):\n")
        for p in unidentifiable: f.write(f"  - {p}\n")


def setup_sobol_analysis(settings, optimization_settings, experiment_dict):
    """
    Classical mode entry point for running Sobol Sensitivity on specific model outputs
    (e.g., specific species concentrations over time).
    Called standalone from *run.py similar to setup_optimization.
    """
    import AntiGen_paths
    from framework.AntimonyGen import AntimonyGen
    from framework.TelluriumGen import TelluriumGen
    import pandas as pd
    
    if not _SALIB_AVAILABLE:
        print("SALib is required for Sobol Analysis. Please pip install SALib.")
        return
        
    MODEL_NAME = settings.get("MODEL_NAME", AntiGen_paths.MODEL_NAME)
    model_text, paths = AntimonyGen(MODEL_NAME, repo_root=AntiGen_paths.REPO_ROOT)
    
    param_names = optimization_settings["param_names"]
    x0 = optimization_settings["x0"]
    bounds = optimization_settings.get("bounds")
    sobol_kwargs = optimization_settings.get("sobol_kwargs", {})
    N = sobol_kwargs.get("N", 128)
    
    observables = sobol_kwargs.get("observables", [])
    times = sobol_kwargs.get("times", [10, 50, 100])
    
    if not observables:
        print("Error: No 'observables' specified in sobol_kwargs for classical output mode.")
        return
        
    problem = {
        'num_vars': len(param_names),
        'names': param_names,
        'bounds': []
    }
    
    for i, p in enumerate(param_names):
        if bounds and bounds[i] is not None:
            problem['bounds'].append(list(bounds[i]))
        else:
            opt_val = x0[i]
            problem['bounds'].append([opt_val * 0.9, opt_val * 1.1])
            
    print(f"\n[Sobol Output] Generating samples (N={N}) for {len(param_names)} parameters...")
    param_values = saltelli.sample(problem, N, calc_second_order=True)
    num_samples = param_values.shape[0]
    
    experiment = experiment_dict.get("EXPERIMENT", experiment_dict.get("experiment"))
    # Use the first replicate/experiment model to test outputs
    if isinstance(experiment, dict) and "replicates" not in dir(experiment):
        first_key = list(experiment.keys())[0]
        rep = experiment[first_key]
    else:
        first_key = list(experiment.replicates.keys())[0]
        rep = experiment.replicates[first_key]
        
    df_dict = rep["Data"](rep, paths["data_path"])
    events_str = rep["Events"](rep, df_dict)
    r = TelluriumGen(model_text + "\n" + events_str, paths)
    rep["Update_parameters"](r, rep)
    
    # We will record Y of shape (num_samples, len(observables), len(times))
    Y = np.zeros((num_samples, len(observables), len(times)))
    
    import time
    t0 = time.time()
    for i, sample in enumerate(param_values):
        r.resetAll()
        for j, p_name in enumerate(param_names):
            try:
                r[p_name] = sample[j]
            except: pass
            
        try:
            from Engine.Simulate import safe_simulate
            block = {
                "start": 0.0,
                "end": max(times),
                "n_points": 500,
                "variable_step_size": True
            }
            res, _ = safe_simulate(r, block, ['time'] + observables)
            for obs_idx, obs_name in enumerate(observables):
                obs_data = res[obs_name]
                time_data = res['time']
                for t_idx, t_val in enumerate(times):
                    idx_t = np.argmin(np.abs(time_data - t_val))
                    Y[i, obs_idx, t_idx] = obs_data[idx_t]
        except Exception:
            Y[i, :, :] = np.nan
            
        if (i+1) % 100 == 0 or (i+1) == num_samples:
            elapsed = time.time() - t0
            rate = (i+1) / elapsed if elapsed > 0 else 0
            print(f"  [{i+1}/{num_samples}]  {elapsed:.1f}s  ({rate:.1f} evals/s)", end='\r', flush=True)
            
    print(f"\n[Sobol Output] Analysis complete. Generating reports...")
    plot_path = paths["plot_path"]
    os.makedirs(plot_path, exist_ok=True)
    
    for obs_idx, obs_name in enumerate(observables):
        for t_idx, t_val in enumerate(times):
            y_slice = Y[:, obs_idx, t_idx]
            valid_mask = np.isfinite(y_slice)
            if not np.all(valid_mask):
                y_slice[~valid_mask] = np.nanmedian(y_slice)
                
            Si = sobol.analyze(problem, y_slice, calc_second_order=True, print_to_console=False)
            
            res_dict = {
                "S1": Si["S1"], "ST": Si["ST"], "names": param_names,
                "type": "output", "N": N, "num_samples": num_samples
            }
            # Clean observable name for file saving
            safe_obs = obs_name.replace('[', '').replace(']', '')
            tag = f"{safe_obs}_t{t_val}"
            save_sobol_plot(res_dict, plot_path, MODEL_NAME, tag=tag)
            
    print(f"Classical Sobol Analysis outputs saved to {plot_path}")


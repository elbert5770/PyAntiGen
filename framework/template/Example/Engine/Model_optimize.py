"""
Example parameter optimization: same experiments as Example_run, with parameters
tuned to minimize loss vs. data. Uses Modules.Experiment and Modules.Optimize.
"""
import os
import sys
import pandas as pd
import AntiGen_paths

REPO_ROOT = AntiGen_paths.REPO_ROOT

from framework.AntimonyGen import AntimonyGen
from framework.TelluriumGen import TelluriumGen
from Modules.Experiment import *
from Engine.Optimize import run_all, run_optimization
from Modules.Plots import *
from Engine.Results import log_optimization_results


def main(settings, optimization_settings, experiment_dict):
    MODEL_NAME = AntiGen_paths.MODEL_NAME
    print(MODEL_NAME)
    model_text, paths = AntimonyGen(MODEL_NAME, repo_root=REPO_ROOT)
    print(paths["plot_path"])
    rss = TelluriumGen(model_text, paths)
    print("Steady state: ", rss.steadyState())
    print("getFloatingSpeciesIds: ", rss.getFloatingSpeciesIds())
    print("getBoundarySpeciesIds: ", rss.getBoundarySpeciesIds())
    print("getAssignmentRuleIds: ", rss.getAssignmentRuleIds())
    ic_path = os.path.join(paths["repo_root"], "antimony_models", MODEL_NAME, f"{MODEL_NAME}_InitialConditions.csv")
    if os.path.exists(ic_path):
        df_ic = pd.read_csv(ic_path)
        if 'Species' in df_ic.columns:
            max_val = 0.0
            vals = {}
            for idx, row in df_ic.iterrows():
                species = row['Species']
                try:
                    # using roadrunner's dict-like access which is robust
                    val = rss[species]
                    vals[idx] = val
                    if val > max_val:
                        max_val = val
                except RuntimeError:
                    pass
            for idx, val in vals.items():
                if val < 1e-10 * max_val:
                    val = 0.0
                df_ic.at[idx, 'InitialCondition'] = val
            df_ic.to_csv(ic_path, index=False)
            print(f"Updated InitialConditions in {ic_path}")

    Species = rss.getFloatingSpeciesIds()
    for s in Species:
        print(s, rss.getValue(s))

    model_text, paths = AntimonyGen(MODEL_NAME, repo_root=REPO_ROOT)


    param_names = optimization_settings["param_names"]
    x0 = optimization_settings["x0"]
    bounds = optimization_settings["bounds"]
    if not param_names:
        print("Error: No parameters to optimize. Set param_names, x0 (and optionally bounds) in *_optimize.py.")
        return

    experiments = experiment_dict['experiment']
    plot_function = experiment_dict["plot"]
    opt = run_optimization(
        model_text,
        paths,
        experiments,
        param_names=param_names,
        x0=x0,
        bounds=bounds,
        loss_config={"observables": optimization_settings["observables"]},
        likelihood_analysis=optimization_settings.get("likelihood_analysis", False),
        method="Nelder-Mead",
    )
    print("Optimization success:", opt["success"], "loss:", opt["fun"])
    import os as _os
    plot_path = paths["plot_path"]
    _csv_path = _os.path.join(plot_path, f"{MODEL_NAME}_optimization_results.csv")
    log_optimization_results(
        opt,
        param_names,
        _csv_path,
        model_name=MODEL_NAME,
        experiment_id="ALL",
        method="Nelder-Mead",
    )
    repo_root = paths["repo_root"]
    if opt.get("results_dict") is not None:
        plot_function(paths,opt["results_dict"])

    if optimization_settings.get("likelihood_analysis", False):
        import numpy as np
        import matplotlib.pyplot as plt
        from Engine.Optimize import _plot_profile_likelihood

        fig, ax = plt.subplots(figsize=(8, 6))
        plot_config = {
            "parameters": param_names,
            "n_points": 20,
            "range_factor": 2.0
        }
        _plot_profile_likelihood(ax, plot_config, opt, opt["x"], param_names)
        plt.tight_layout()
        plot_name2 = os.path.join(plot_path, MODEL_NAME + "_ALL_optimize_profile_likelihood.png")
        plt.savefig(plot_name2, bbox_inches="tight")
        print(f"Second plot (Profile Likelihood) saved to: {plot_name2}")
        # plt.show()

def setup_optimization(settings, optimization_settings, experiment_dict):
    """
    Setup and run optimization.
    """
    main(settings, optimization_settings, experiment_dict)

if __name__ == "__main__":
    settings = []
    optimization_settings = []
    experiment_dict = []
    main(settings, optimization_settings, experiment_dict)

"""
Example parameter optimization: same experiments as Example_run, with parameters
tuned to minimize loss vs. data. Uses Modules.Experiment and Modules.Optimize.
"""
import os
import sys

_project_dir = os.path.dirname(os.path.abspath(__file__))
if _project_dir not in sys.path:
    sys.path.insert(0, _project_dir)
MODEL_NAME = os.path.basename(_project_dir)

from framework.AntimonyGen import AntimonyGen
from Modules.Experiment import EXPERIMENTS
from Modules.Optimize import run_all, run_optimization
from Modules.Plots import plot_results


def main():
    if os.path.basename(_project_dir) == "scripts":
        repo_root = os.path.abspath(os.path.join(_project_dir, ".."))
    else:
        repo_root = os.path.abspath(os.path.join(_project_dir, "..", ".."))
    model_text, data_path, plot_path, repo_root = AntimonyGen(MODEL_NAME, repo_root=repo_root)

    # Define parameters to optimize (can be model parameters or formula variables like SF).
    # Replace with your model's parameter names and bounds.
    param_names = ["k_A_to_B", "SF"]  # e.g. ["k1", "k2", "SF"]
    x0 = [0.5, 1.0]          # e.g. [0.1, 0.05, 1.0]
    bounds = [(0.0, 1.0), (0.1, 10.0)]    # e.g. [(1e-6, 10), (1e-6, 10), (0.1, 10)]

    if not param_names:
        print("No parameters to optimize. Set param_names, x0 (and optionally bounds) in Example_optimize.py.")
        print("Running all experiments with default parameters and plotting.")
        results = run_all(model_text, MODEL_NAME, repo_root, EXPERIMENTS, data_path)
        plot_results(plot_path, MODEL_NAME + "_optimize", results)
        return

    opt = run_optimization(
        model_text,
        MODEL_NAME,
        repo_root,
        EXPERIMENTS,
        data_path,
        param_names=param_names,
        x0=x0,
        bounds=bounds,
        loss_config={
            "observables": [
                {"observable": "SF*[B_Comp1]", "data_column": "B"}
            ],
            "time_column": "time"
        },
        method="L-BFGS-B",
    )
    print("Optimization success:", opt["success"], "loss:", opt["fun"])
    if opt.get("results") is not None:
        plot_results(plot_path, MODEL_NAME + "_optimize", opt["results"])


if __name__ == "__main__":
    main()

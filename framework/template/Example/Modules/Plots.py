import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import numpy as np


def plot_results(paths, results_dict):
    """
    Plot simulation results for N experiments.

    Args:
        plot_path: Directory to save the plot.
        MODEL_NAME: Model name for title/filename.
        results: List of dicts from Experiment.run_all: each has "result", "data", "label".
    """
    
    plot_path = paths["plot_path"]
    MODEL_NAME = paths["MODEL_NAME"]
    repo_root = paths["repo_root"]
    n = max(len(results_dict), 1)
    color_A = ["blue","green"]
    color_B = ["red","orange"]
    fig = plt.figure(figsize=(10, 10))
    gs  = gridspec.GridSpec(1, 1, figure=fig, hspace=0.5, wspace=0.35)
    ax = fig.add_subplot(gs[0, 0])
    for i, (label, item) in enumerate(results_dict.items()):
        
        results = item["results"]
        data_dict = item["data"]
        data = data_dict["B data"]
        
        time_points = results["time"]
        
        ax.plot(time_points, results["predicted_A"], label=f"[A] {label}", color=color_A[i])
        ax.plot(time_points, results["predicted_B"], label=f"[B] {label}", color=color_B[i])
        if "time" in data.columns and "B" in data.columns:
            ax.scatter(data["time"], data["B"], color=color_B[i], s=30, zorder=5, label=f"Measured [B] {label}")
            
            # If the piecewise interpolation was simulated, plot it directly from the result
        if label == "Experiment_1":
            ax.plot(time_points, results["pw_interp1"], '--', color="cyan", alpha=0.7, label=f"Piecewise Linear", zorder=4)
            ax.plot(time_points, results["pw_interp2"], '--', color="purple", alpha=0.7, label=f"Piecewise Spline", zorder=4)
          
            
    ax.set_xlabel("Time")
    ax.set_ylabel("Concentration")
    ax.set_title("Simulation Results for " + MODEL_NAME)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    plt.subplots_adjust(right=0.65)
    plot_name = os.path.join(plot_path, MODEL_NAME + ".png")
    plt.savefig(plot_name, bbox_inches="tight")
    print(f"Plot saved to: {plot_name}")
    plt.show()


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
    color_A = ["blue","green","cyan","black"]
    color_B = ["red","orange","purple","brown"]
    fig = plt.figure(figsize=(10, 10))
    gs  = gridspec.GridSpec(1, 1, figure=fig, hspace=0.5, wspace=0.35)
    ax = fig.add_subplot(gs[0, 0])
    for i, (label, item) in enumerate(results_dict.items()):
        
        results = item["results"]
        data_dict = item["data"]
        
        data = data_dict[f"{label}"]
        
        time_points = results["time"]
        
        ax.plot(time_points, results["predicted_A"], label=f"[A] {label}", color=color_A[i])
        ax.plot(time_points, results["predicted_B"], label=f"[B] {label}", color=color_B[i], linestyle='--')
        if "time" in data.columns and "B" in data.columns:
            ax.scatter(data["time"], data["B"], color=color_B[i], s=30, zorder=5, label=f"Measured [B] {label}")
            

            
    ax.set_xlabel("Time")
    ax.set_ylabel("Concentration")
    ax.set_title("Simulation Results for " + MODEL_NAME)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    plt.subplots_adjust(right=0.65)
    plot_name = os.path.join(plot_path, MODEL_NAME + ".png")
    plt.savefig(plot_name, bbox_inches="tight")
    print(f"Plot saved to: {plot_name}")
    plt.show()



def plot_flipflop(paths, results_dict):
    """Plot the flip-flop example on a log scale (the scale the loss uses)."""
    plot_path = paths["plot_path"]
    MODEL_NAME = paths["MODEL_NAME"]
    colors = ["tab:blue", "tab:red", "tab:green", "tab:orange"]
    fig, ax = plt.subplots(figsize=(9, 6))
    for i, (label, item) in enumerate(results_dict.items()):
        results = item["results"]
        data_dict = item["data"]
        c = colors[i % len(colors)]
        ax.semilogy(results["time"], np.maximum(results["predicted_B"], 1e-12),
                    color=c, label=f"predicted_B {label}")
        ax.semilogy(results["time"], np.maximum(results["predicted_A"], 1e-12),
                    color=c, linestyle=":", label=f"predicted_A {label}")
        data_B = data_dict.get(label)
        if data_B is not None and "logB" in data_B.columns:
            ax.scatter(data_B["time"], 10.0 ** data_B["logB"], color=c, s=20,
                       zorder=5, label=f"logB data {label}")
        data_A = data_dict.get(f"{label}_A")
        if data_A is not None and "logA" in data_A.columns:
            ax.scatter(data_A["time"], 10.0 ** data_A["logA"], color=c, s=40,
                       marker="x", zorder=5, label=f"logA data {label}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Output (log scale)")
    ax.set_ylim(1e-3, 30)
    ax.set_title("Flip-flop example: " + MODEL_NAME)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    plt.subplots_adjust(right=0.62)
    plot_name = os.path.join(plot_path, MODEL_NAME + "_flipflop.png")
    plt.savefig(plot_name, bbox_inches="tight")
    print(f"Plot saved to: {plot_name}")
    plt.show()

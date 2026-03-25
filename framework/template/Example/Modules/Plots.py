import matplotlib.pyplot as plt
import os
import numpy as np



def plot_results(plot_path, MODEL_NAME, results):
    """
    Plot simulation results for N experiments.

    Args:
        plot_path: Directory to save the plot.
        MODEL_NAME: Model name for title/filename.
        results: List of dicts from Experiment.run_all: each has "result", "data", "label".
    """
    
    color_A = ["blue","green"]
    color_B = ["red","orange"]
    plt.figure(figsize=(10, 8))
    for i, item in enumerate(results):
        result = item["result"]
        df = item["data"]
        label = item["label"]
        time_points = result["time"]
        
        plt.plot(time_points, result["[A_Comp1]"], label=f"[A] {label}", color=color_A[i])
        plt.plot(time_points, result["[B_Comp1]"], label=f"[B] {label}", color=color_B[i])
        if "time" in df.columns and "B" in df.columns:
            plt.scatter(df["time"], df["B"], color=color_B[i], s=30, zorder=5, label=f"Measured [B] {label}")
            
            # If the piecewise interpolation was simulated, plot it directly from the result
            if label == "Experiment 1":
                if "pw_interp1" in result.colnames:
                    plt.plot(time_points, result["pw_interp1"], '--', color="cyan", alpha=0.7, label=f"Piecewise Linear", zorder=4)
                if "pw_interp2" in result.colnames:
                    plt.plot(time_points, result["pw_interp2"], '--', color="purple", alpha=0.7, label=f"Piecewise Spline", zorder=4)
    plt.xlabel("Time")
    plt.ylabel("Concentration")
    plt.title("Simulation Results for " + MODEL_NAME)
    plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    plt.subplots_adjust(right=0.7)
    plot_name = os.path.join(plot_path, MODEL_NAME + ".png")
    plt.savefig(plot_name, bbox_inches="tight")
    print(f"Plot saved to: {plot_name}")
    plt.show()


import os
import sys

import paths

REPO_ROOT = paths.REPO_ROOT

from framework.AntimonyGen import AntimonyGen, TelluriumGen
from framework.data_interpolation import generate_antimony_piecewise
from Modules.Experiment import EXPERIMENTS
from Modules.Plots import plot_results
from Modules.Simulate import simulate


def run_simulation(settings=[]):
    MODEL_NAME = paths.MODEL_NAME
    model_text, data_path, plot_path, repo_root = AntimonyGen(MODEL_NAME, repo_root=REPO_ROOT)
    results = []
    for spec in EXPERIMENTS:
        events = spec["event_func"]()
        df = spec["load_data"](data_path)
        
        if "Experiment 1" in spec["label"]:
            pw_str1 = generate_antimony_piecewise(df["time"], df["B"], data_name="pw_interp1", interpolation_type="linear")
            pw_str2 = generate_antimony_piecewise(df["time"], df["B"], data_name="pw_interp2", interpolation_type="spline")
        else:
            pw_str1 = "pw_interp1 := 0.0"
            pw_str2 = "pw_interp2 := 0.0"
            
        full_model_text = model_text + "\n" + events + "\n" + pw_str1 + "\n" + pw_str2
        r = TelluriumGen(full_model_text, MODEL_NAME, repo_root)
        result = simulate(r)
        results.append({
            "id": spec["id"],
            "result": result,
            "data": df,
            "label": spec.get("label", spec["id"]),
        })
    plot_results(plot_path, MODEL_NAME, results)
    
if __name__ == "__main__":
    settings = []
    run_simulation(settings=settings)

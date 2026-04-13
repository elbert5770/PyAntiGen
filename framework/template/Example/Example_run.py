
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
        df = spec["load_data"](data_path)
        events = spec["event_func"](df)
            
        full_model_text = model_text + "\n" + events
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


import os
import sys

# Use location: import Model_* from the same folder as this script (model folder)
_project_dir = os.path.dirname(os.path.abspath(__file__))
if _project_dir not in sys.path:
    sys.path.insert(0, _project_dir)
MODEL_NAME = os.path.basename(_project_dir)

from framework.AntimonyGen import AntimonyGen, TelluriumGen
from Modules.Experiment import EXPERIMENTS
from Modules.Plots import plot_results
from Modules.Simulate import simulate


def run_simulation():
    if os.path.basename(_project_dir) == "scripts":
        repo_root = os.path.abspath(os.path.join(_project_dir, ".."))
    else:
        repo_root = os.path.abspath(os.path.join(_project_dir, "..", ".."))
    model_text, data_path, plot_path, repo_root = AntimonyGen(MODEL_NAME, repo_root=repo_root)
    results = []
    for spec in EXPERIMENTS:
        events = spec["event_func"]()
        full_model_text = model_text + "\n" + events
        r = TelluriumGen(full_model_text, MODEL_NAME, repo_root)
        result = simulate(r)
        df = spec["load_data"](data_path)
        results.append({
            "id": spec["id"],
            "result": result,
            "data": df,
            "label": spec.get("label", spec["id"]),
        })
    plot_results(plot_path, MODEL_NAME, results)
    
if __name__ == "__main__":
    run_simulation()


import os
import sys

# Use location: import Model_* from the same folder as this script (model folder)
_project_dir = os.path.dirname(os.path.abspath(__file__))
if _project_dir not in sys.path:
    sys.path.insert(0, _project_dir)
MODEL_NAME = os.path.basename(_project_dir)

import os
# Add root of the repository to sys.path
_repo_root = os.path.abspath(os.path.join(_project_dir, '..', '..', '..'))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from framework.AntimonyGen import AntimonyGen, TelluriumGen
from framework.data_interpolation import generate_antimony_piecewise
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
    run_simulation()

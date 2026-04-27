
import os
import sys
import pandas as pd
import AntiGen_paths

REPO_ROOT = AntiGen_paths.REPO_ROOT

from framework.AntimonyGen import AntimonyGen
from framework.TelluriumGen import TelluriumGen

from Modules.Experiment import *
from Modules.Plots import *
from Engine.Simulate import simulate

def run_steady_state(model_text, paths, settings):
    
    rss = TelluriumGen(model_text, paths)
    if settings["Verbose"]:
        print("Steady state: ", rss.steadyState())
        print("getFloatingSpeciesIds: ", rss.getFloatingSpeciesIds())
        print("getBoundarySpeciesIds: ", rss.getBoundarySpeciesIds())
        print("getAssignmentRuleIds: ", rss.getAssignmentRuleIds())
    
    
    if os.path.exists(paths["models_path"]):
        df_ic = pd.read_csv(paths["models_path"])
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
            df_ic.to_csv(paths["models_path"], index=False)
            print(f"Updated InitialConditions in {paths['models_path']}")
        else:
            print(f"No 'Species' column found in {paths['models_path']}")
            return
    else:
        print(f"No InitialConditions file found in {paths['ic_path']}")
        return

    Species = rss.getFloatingSpeciesIds()
    for s in Species:
        print(s, rss.getValue(s))


def run_simulation(model_text, paths, settings, EXPERIMENT_dict):

    data_path = paths["data_path"]
    repo_root = paths["repo_root"]
    MODEL_NAME = paths["MODEL_NAME"]
    print("run_simulation", MODEL_NAME)
    save_path = os.path.join(repo_root, "generated", MODEL_NAME, MODEL_NAME + "_events.txt")

    results_dict = {}
    experiment = EXPERIMENT_dict['EXPERIMENT']
    for group in experiment.optimization_groups:
        print(group)
    
    for label, treatment in experiment.treatments.items():
        print("Label", label)
        df_dict = treatment["Data"](treatment, data_path)
        events = treatment["Events"](treatment,df_dict)

        with open(save_path, "w") as f:
            f.write(events)

        full_model_text = model_text + "\n" + events

        r = TelluriumGen(full_model_text, paths)
        # print("k_plaque1", r["k_plaque1"])
        # print("k_oligo1_AB42", r['k_oligo1_AB42'])
        # print("amyloid_positive", treatment["amyloid_positive"])
        print("Antibody_IV_Dose", r["Antibody_IV_Dose"])
        treatment["Update_parameters"](r, treatment)
        print("Antibody_IV_Dose", r["Antibody_IV_Dose"])
        # print("k_plaque1", r["k_plaque1"])
        # print("k_oligo1_AB42", r['k_oligo1_AB42'])
        # print("amyloid_positive", treatment["amyloid_positive"])
        solver_settings = treatment["Solver_settings"](treatment)
        observed_species = treatment["Observed_species"](r)
        results = simulate(r, solver_settings, observed_species)
        

        results_dict[label] = {
            "results": results,
            "treatment": treatment,
            "data": df_dict,
            "observed_species": observed_species,
            "solver_settings": solver_settings,
            "events": events
        }
    EXPERIMENT_dict["plot"](paths,results_dict)
    return results_dict

def setup_simulation(settings, EXPERIMENT_dict):
    if settings.get("MODEL_NAME"):
        MODEL_NAME = settings["MODEL_NAME"]
    else:
        MODEL_NAME = AntiGen_paths.MODEL_NAME
    print("setup_simulation", MODEL_NAME)

    model_text, paths = AntimonyGen(MODEL_NAME, repo_root=REPO_ROOT)

    if settings["run_steady_state_first"]:
        run_steady_state(model_text, paths, settings)

    results_dict = run_simulation(model_text, paths, settings, EXPERIMENT_dict)



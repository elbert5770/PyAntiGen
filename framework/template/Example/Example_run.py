import os
import sys

from Engine.Model_simulate import setup_simulation
from Engine.Model_optimize import setup_optimization_from_groups
from Modules.Plots import *
from Modules.Experiment import get_experiment
from Modules.Optimizer_settings import OPTIMIZATION_SETTINGS
from AntiGen_paths import MODEL_NAME


EXPERIMENT_dict = {
    "Example": {'EXPERIMENT': get_experiment('EXPERIMENT_Example'), 'plot': plot_results, 'opt_settings_key': 'Example'},
}

run_settings = {
    "run_steady_state_first" : True,
    "Verbose" : True,
}


if __name__ == "__main__":
    print("Run_settings: ", run_settings)
    
    run_optimization = True
    run_name = "Example"

    update_antimony_model()
    
    if run_optimization:
        opt_key      = EXPERIMENT_dict[run_name].get('opt_settings_key', run_name)
        experiment_arg = {
            "EXPERIMENT": EXPERIMENT_dict[run_name]['EXPERIMENT'],
            "plot":       EXPERIMENT_dict[run_name]['plot'],
        }
        setup_optimization_from_groups(run_settings, OPTIMIZATION_SETTINGS[opt_key], experiment_arg)
    else:
        setup_simulation(run_settings, EXPERIMENT_dict[run_name])


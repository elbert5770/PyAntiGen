import os
import sys

from Engine.Model_simulate import setup_simulation
from Engine.Model_optimize import setup_optimization
from Modules.Plots import plot_results
from Modules.Experiment import get_experiment

experiment_dict = {
    "Example": {'experiment': get_experiment('EXPERIMENTS_Example'), 'plot': plot_results, },
}

run_settings = {
    "run_steady_state_first" : True,
    "Verbose" : True,
}

optimization_settings = {
    "run_optimization" : True,
    "run_steady_state_first" : True,
    "likelihood_analysis" : False,
    "param_names" : ["k_A_to_B", "SF"],
    "x0" : [ 0.5, 2.0 ],
    "bounds" : [ (0.01, 10.0), (0.01, 10.0) ],
    "Verbose" : True,
    "observables": [
        {
            "observed_variable": "predicted_B",
            "data_dict_key": "B data",
            "data_column": "B", 
            "time_column": "time"
        },
    ],
}

if __name__ == "__main__":
    print("run_settings: ", run_settings)
    print("optimization_settings: ", optimization_settings)
    if optimization_settings["run_optimization"]:
        setup_optimization(run_settings, optimization_settings, experiment_dict["Example"])
    else:
        setup_simulation(run_settings, experiment_dict["Example"])

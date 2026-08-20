import os
import sys

from Engine.Model_simulate import setup_simulation
from Engine.Model_optimize import (
    setup_optimization_from_groups,
    _FULL_DIAGNOSTICS,
    _SLICE_ONLY,
    _PROFILE_ONLY,
    _SOBOL_ONLY,
    _FAST_PROFILE_ONLY,
    _NO_DIAGNOSTICS,
    DIAGNOSTICS_PRESETS,
)
from Modules.Plots import *
from Modules.Experiment import get_EXPERIMENT
from Modules.Optimizer_settings import get_OPTIMIZATION
from AntiGen_paths import MODEL_NAME, REPO_ROOT
from Model_generate import update_antimony_model

EXPERIMENT_dict = {
    "Example": {'EXPERIMENT': get_EXPERIMENT('EXPERIMENT_Example'), 'plot': plot_results, 'opt_settings_key': 'Example'},
}

diagnostics = _NO_DIAGNOSTICS

OPTIMIZATION_REGISTRY = {
    # One CLI selection can run multiple independent Optimization specs in
    # sequence (e.g. parameters split across specs for identifiability
    # reasons); opt_key may be a single name or a list of names.
    # Optional "diagnostics" dict is merged into run_settings for every spec
    # run under that selection (wald/slice/profile-likelihood/Sobol flags).
    "Example1": {
        'opt_key': ['OPTIMIZATION_Example1_ADpos', 'OPTIMIZATION_Example1_ADneg'],
        'experiment': 'EXPERIMENT_Example',
    },
    "Example2": {
        # Same fits as Example1 (split by identifiability), with full
        # diagnostics turned on to confirm both fits are well-identified.
        'opt_key': ['OPTIMIZATION_Example1_ADpos', 'OPTIMIZATION_Example1_ADneg'],
        'experiment': 'EXPERIMENT_Example',
        'diagnostics': _FULL_DIAGNOSTICS,
    },
    "Example3": {
        # Negative example: k_A_to_B, SF, and V_Comp1 fit jointly against the
        # ADpos data (the fit Example1 deliberately avoids). Diagnostics
        # expose SF/V_Comp1 as structurally unidentifiable together.
        'opt_key': ['OPTIMIZATION_Example3_joint'],
        'experiment': 'EXPERIMENT_Example',
        'diagnostics': _FULL_DIAGNOSTICS,
    },
}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Tellurium/RoadRunner Simulation and Optimization Runner")
    parser.add_argument("--optimize", type=str, choices=list(OPTIMIZATION_REGISTRY.keys()),
                        help="Optimization mode")
    parser.add_argument("--simulate", type=str, choices=list(EXPERIMENT_dict.keys()),
                        help="Simulation mode")
    parser.add_argument("--diagnostics", type=str, default="_NO_DIAGNOSTICS",
                        choices=list(DIAGNOSTICS_PRESETS.keys()),
                        help="Diagnostics settings preset to run (default: _NO_DIAGNOSTICS)")
    parser.add_argument("--no-fit", action="store_true", dest="no_fit",
                        help="Skip the optimizer and evaluate the spec's x0 instead, "
                             "then run the requested diagnostics against it.")

    args = parser.parse_args()

    if len(sys.argv) == 1:
        parser.error("No arguments provided.")

    update_antimony_model()


# ---  Optimization ------------------------------------
    if args.optimize:
        opt_info = OPTIMIZATION_REGISTRY[args.optimize]
        model_name_to_use = MODEL_NAME

        run_settings = {
            "run_steady_state_first": False,
            "Verbose": True,
            "save_SBML?": False,
            "MODEL_NAME": model_name_to_use,
            "slice_analysis": False,
            "fit_mode": "evaluate_x0" if args.no_fit else "optimize",
        }
        selected_diagnostics = DIAGNOSTICS_PRESETS.get(args.diagnostics, _NO_DIAGNOSTICS)
        run_settings.update(selected_diagnostics)
        if "diagnostics" in opt_info:
            run_settings.update(opt_info["diagnostics"])


        # Retrieve experiment object & optimization spec(s)
        exp_obj = get_EXPERIMENT(opt_info["experiment"])
        opt_keys = opt_info["opt_key"]
        if isinstance(opt_keys, str):
            opt_keys = [opt_keys]

        experiment_arg = {
            "EXPERIMENT": exp_obj,
            "plot": None,  # Bypassed during optimization run
        }

        for opt_key in opt_keys:
            opt_spec = get_OPTIMIZATION(opt_key)
            print(f"Starting domain optimization for: {args.optimize} [{opt_key}]")
            setup_optimization_from_groups(run_settings, opt_spec, experiment_arg)

# --- Simulation ---------------------------------------------
    elif args.simulate:
        run_name = args.simulate
        fig_config = EXPERIMENT_dict[run_name]
        model_name_to_use = MODEL_NAME


        run_settings = {
            "run_steady_state_first": False,
            "Verbose": True,
            "save_SBML?": False,
            "MODEL_NAME": model_name_to_use
        }

        setup_simulation(run_settings, fig_config)
    
    print("Run_settings: ", run_settings)
    


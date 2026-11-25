"""Experiment registry: one 'replicate' entry per simulation.

Which simulations are fitted together, and how their losses are weighted, is a property of the
optimization (Optimization.groups in Optimizer_settings.py), not of the experiment.
"""
from .Data import *
from .Events import *
from .Loss_config import *
from .Observed_species import *
from .Solver_settings import *
from .Update_parameters import *
from .Update_opt_parameters import *

from dataclasses import dataclass, field

@dataclass
class Experiment:
    replicates: dict = field(default_factory=dict)


def make_replicate(config, **meta):
    """
    Build a single replicate entry dict from an explicit config dict.

    config keys (all required unless noted):
      "Label"            : str
      "Events"           : callable(replicate, df_dict) -> str
      "Data"             : callable(replicate, data_path) -> Dict of DataFrame or None
      "Observed_species" : callable(RoadRunner_instance or None) -> list[str]
      "Solver_settings"  : callable(replicate) -> dict
      "Update_parameters": callable(RoadRunner_instance, replicate, mode) -> dict
      "Loss_config"      : callable(replicate) -> dict
      
    'callable' functions are contained in separate files with the same name as the key. 
    For example, "Events" is contained in the file "Events.py", "Data" is contained in
    the file "Data.py", etc. 
    
    'replicate' is the output of make_replicate. It contains the dictionary above plus
    any optional keyword arguments. 

    Any arguments required by the 'callable' functions should be passed as keyword arguments.
    
    Keyword arguments are stored in 'replicate' as additional keys.  Example arguments:
      Age = 70, Status = True, Population = "Amyloid_negative", Drug = "Lecanemab", 
      dose_nmol = 10, Schedule = "default", Type = "default", …  

    Example::

        make_replicate(
            {
                "Label":            "Example_1",
                "Events":           generate_no_events,
                "Data":             load_data_Example,
                "Observed_species": observed_Example,
                "Solver_settings":  solver_settings_Example,
                "Update_parameters": update_no_parameters,
                "Loss_config": Example_loss_config,
            },
        )
    """
    entry = dict(config)
    return {**entry, **meta}

# ***************************************************************************
# USER DEFINED OPTIMIZATION
# ***************************************************************************

# Build each replicate so that each element described in the
# make_replicate function has a value. 
# It can be helpful (but not required) to define experimental groups
# and treatments (conditions) as separate dictionaries, and then 
# build each replicate by combining elements from the experimental groups
# and treatments. "params" are used to pass arguments to the 'callable' functions.
# Which replicates are fitted together is decided by the optimization, not here: each
# group in Optimization.groups (Optimizer_settings.py) lists its simulations by "Label".
# The "Label" should be unique for each replicate, although not required.
# The "replicate" designation implies an experimental replicate, but it is rare
# to optimize at the replicate level. Typically, replicates are handled at
# the "Data" level, with all replicates' data fit simultaneously.



def _build_experiment():
    # Initialize Experiment class object
    exp = Experiment()

    # Treatments are typically linked to events, but other settings can also be linked.
    # In the example below, "loss_config" is also linked to the treatment because
    # the data loaded depends on the treatment.
    treatments = {
        "Early": {"params": {"dose": 10, "delay": 5}},
        "Late": {"params": {"dose": 5, "delay": 10}},
    }
    
    # Experimental groups are not required but are natural for some
    # experimental designs. For example, when studying the effect of a treatment
    # in different populations.
    exp_groups = {
        "ADneg": {"params": {"amyloid_positive": False}},
        "ADpos": {"params": {"amyloid_positive": True}}
    }
    
    for exp_group_name, exp_group_data in exp_groups.items():
        exp_params = exp_group_data.get("params", {})
        for treatment_name, treatment_data in treatments.items():
            treatment_params = treatment_data.get("params", {})

            key = f"{exp_group_name}_{treatment_name}"
            
            replicate = make_replicate(
                {
                    "Label":               key,
                    "Events":              Example_event,
                    "Data":                load_ad_data,
                    "Observed_species":    all_species,
                    "Solver_settings":     solver_settings_Example,
                    "Update_opt_parameters": update_opt_no_parameters,
                    "Update_parameters":   update_Example,
                },
                **exp_params,
                **treatment_params
            )
            
            # Store replicate in the flat registry and nested structure
            exp.replicates[key] = replicate

    return exp

EXPERIMENT_Example = _build_experiment()


def _build_flipflop_experiment():
    """Flip-flop identifiability experiment (Example4/Example5).

    Two treatments of the A -> B -> C chain, dosed by the same event style as
    the base example, observed through predicted_B on a log10 scale. The Early
    treatment additionally carries four very noisy predicted_A observations
    ("has_A_data") — without them the likelihood would have two *exactly*
    equal modes; with them the swapped mode sits at a known dNLL of ~2.4
    (printed by data/make_flipflop_data.py when regenerating the data).
    """
    exp = Experiment()
    treatments = {
        "Early": {"params": {"dose": 10, "delay": 5, "has_A_data": True}},
        "Late":  {"params": {"dose": 5, "delay": 10, "has_A_data": False}},
    }
    for treatment_name, treatment_data in treatments.items():
        key = f"Flipflop_{treatment_name}"
        exp.replicates[key] = make_replicate(
            {
                "Label":                 key,
                "Events":                Example_event,
                "Data":                  load_flipflop_data,
                "Observed_species":      all_species,
                "Solver_settings":       solver_settings_Example,
                "Update_opt_parameters": update_opt_no_parameters,
                "Update_parameters":     update_flipflop,
            },
            **treatment_data.get("params", {}),
        )
    return exp

EXPERIMENT_Flipflop = _build_flipflop_experiment()

# ***************************************************************************
# END USER DEFINED EXPERIMENTS
# ***************************************************************************

# ---------------------------------------------------------------------------
# Registry accessors
# ---------------------------------------------------------------------------

def get_EXPERIMENTS():
    """Returns a dict mapping experiment name -> Experiment."""
    return {k: v for k, v in globals().items() if k.startswith('EXPERIMENT_')}

def get_EXPERIMENT(name):
    """Returns a single Experiment by name."""
    return globals().get(name)

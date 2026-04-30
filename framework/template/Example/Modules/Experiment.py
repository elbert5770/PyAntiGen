"""Experiment registry: one 'replicate' entry per simulation,
simulations are optimized together if in the same opt_group. 
"""
from .Data import *
from .Events import *
from .Loss_config import *
from .Observed_species import *
from .Solver_settings import *
from .Update_parameters import *

from dataclasses import dataclass, field

@dataclass
class Experiment:
    replicates: dict = field(default_factory=dict)

    @property
    def opt_groups(self):
        """Return {opt_group: [replicate_key, ...]} by scanning replicates."""
        groups = {}
        for key, rep in self.replicates.items():
            og = rep.get("Opt_group")
            if og is not None:
                groups.setdefault(og, []).append(key)
        return groups


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
      "Opt_group"        : str
      
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
                "Opt_group": opt_group
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
# The "opt_group" is a string that defines which replicates are optimized
# together, each contributing to the total loss function value when their
# "Opt_group" valuse are the same. For example, if "Opt_group" is "ADneg" for
# "Early" and "Late", then the loss function will be calculated for both
# "Early" and "Late" replicates.
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
    # Treatment types may also comprise the 'opt_group' in some cases.
    treatments = {
        "Early": {"params": {"dose": 10, "delay": 5}},
        "Late": {"params": {"dose": 5, "delay": 10}},
    }
    
    # Experimental groups are not always necessary but are natural for some
    # experimental designs. For example, when studying the effect of a treatment
    # in different populations.
    exp_groups = {
        "ADneg": {"opt_group": "ADneg", "params": {"amyloid_positive": False}},
        "ADpos": {"opt_group": "ADpos", "params": {"amyloid_positive": True}}
    }
    
    for exp_group_name, exp_group_data in exp_groups.items():
        opt_group = exp_group_data.get("opt_group", exp_group_name)
        exp_params = exp_group_data.get("params", {})
        for treatment_name, treatment_data in treatments.items():
            treatment_params = treatment_data.get("params", {})

            key = f"{exp_group_name}__{treatment_name}"
            
            replicate = make_replicate(
                {
                    "Label":               key,
                    "Events":              Example_event,
                    "Data":                load_ad_data,
                    "Observed_species":    all_species,
                    "Solver_settings":     solver_settings_Example,
                    "Update_parameters":   update_Example,
                    "Loss_config":         Example_loss_config_AD,
                    "Opt_group":           opt_group,
                },
                **exp_params, 
                **treatment_params
            )
            
            # Store replicate in the flat registry and nested structure
            exp.replicates[key] = replicate

    return exp

EXPERIMENT_Example = _build_experiment()

# ***************************************************************************
# END USER DEFINED EXPERIMENTS
# ***************************************************************************

# ---------------------------------------------------------------------------
# Registry accessors
# ---------------------------------------------------------------------------

def get_experiments():
    """Returns a dict mapping experiment name -> experiment dict."""
    return {k: v for k, v in globals().items() if k.startswith('EXPERIMENT_')}

def get_experiment(name):
    """Returns an experiment dict by name."""
    return globals().get(name)

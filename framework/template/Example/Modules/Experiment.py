"""Experiment registry: one entry per experiment for run and optimize."""
from .Data import *
from .Events import *
from .Observed_species import *
from .Solver_settings import *

def get_experiments():
    """Returns a dict mapping experiment name -> experiment dict."""
    return {k: v for k, v in globals().items() if k.startswith('EXPERIMENT_')}

def get_experiment(name):
    """Returns a single experiment dict by name."""
    return globals().get(name)

def make_experiment(config, **meta):
    """
    Build a single experiment entry dict from an explicit config dict.
    """
    entry = dict(config)
    return {**entry, **meta}

EXPERIMENTS_Example = {
    "1": make_experiment(
        {
            "Label": "Experiment_1",
            "Events": event1,
            "Data": load_experiment1_data,
            "Observed_species": all_species,
            "Solver_settings": solver_settings_Example,
        }
    ),
    "2": make_experiment(
        {
            "Label": "Experiment_2",
            "Events": event2,
            "Data": load_experiment2_data,
            "Observed_species": all_species,
            "Solver_settings": solver_settings_Example,
        }
    )
}

"""Experiment registry: one entry per experiment for run and optimize."""
from .Data import load_experiment1_data, load_experiment2_data
from .Events import event1, event2
""" Tips: There is only one data file per experiment (intentionally). Use it this way:
    1) Only load the data file needed for optimization loss functions here, 
    2) Data files for forcing functions should be loaded in the run module if they are 
       different from the optimization data file, 
    3) Data files for plots should be loaded in the plotting module if they are different 
       from the optimization data file."""
EXPERIMENTS = [
    {"id": "1", "event_func": event1, "load_data": load_experiment1_data, "label": "Experiment 1"},
    {"id": "2", "event_func": event2, "load_data": load_experiment2_data, "label": "Experiment 2"},
]

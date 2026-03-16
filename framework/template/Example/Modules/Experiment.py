"""Experiment registry: one entry per experiment for run and optimize."""
from .Data import load_experiment1_data, load_experiment2_data
from .Events import event1, event2

EXPERIMENTS = [
    {"id": "1", "event_func": event1, "load_data": load_experiment1_data, "label": "Experiment 1"},
    {"id": "2", "event_func": event2, "load_data": load_experiment2_data, "label": "Experiment 2"},
]

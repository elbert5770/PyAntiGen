import pandas as pd
import os


def load_no_data(experiment, data_path):
    data_dict = {}  
    return data_dict


def load_experiment1_data(experiment, data_path):
    experiment_path = os.path.join(data_path, 'Example_experiment1.csv')
    df = pd.read_csv(experiment_path)
    data_dict = {
        "B data": df
    }
    return data_dict

def load_experiment2_data(experiment, data_path):
    experiment_path = os.path.join(data_path, 'Example_experiment2.csv')
    df = pd.read_csv(experiment_path)
    data_dict = {
        "B data": df
    }
    return data_dict


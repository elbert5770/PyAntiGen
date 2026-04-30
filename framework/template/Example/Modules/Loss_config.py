def no_optimization(replicate):
    return {}
    
def Example_loss_config_AD(replicate):
    label = replicate["Label"]
    return {"observables": [{
        "observed_variable": "predicted_B",
        "data_column":       "B",
        "time_column":       "time",
        "data_dict_key":     label,
    }]}


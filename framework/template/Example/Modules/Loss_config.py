def no_optimization(replicate):
    return {}
    
def Example1_loss_config(replicate):
    label = replicate["Label"]
    return {"observables": [{
        "observed_variable": "predicted_B",
        "data_column":       "B",
        "time_column":       "time",
        "data_dict_key":     label,
    }]}

# def Example2_loss_config(replicate):
#     label = replicate["Label"]
#     return {"observables": [{
#         "observed_variable": "predicted_A",
#         "data_column":       "A",
#         "time_column":       "time",
#         "data_dict_key":     label,
#     }]}


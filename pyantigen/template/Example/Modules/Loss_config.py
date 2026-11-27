# Noise in a loss_config entry (all optional; by default each observable estimates its own sigma):
#   "sigma_method"/"sigma_value"  a declared sigma ("fixed"), or a heuristic for the fit
#   "sigma_pool": "<name>"        observables with the same name share ONE estimated sigma
#                                 (PEtab's single noise parameter for several observables)
#   "noise_column": "<column>"    a known sigma on every datapoint, read from that column of the data (PEtab's numeric noise);
#                                 scored exactly, 0.5 sum w (r / sigma_i)^2 + sum w ln sigma_i. Every scored row needs a positive sigma.
#                                 It cannot share a sigma_block / sigma_pool with observables that have no such column.
# Not supported, and an error rather than ignored: "noise_formula" and "sigma_method": "relative" (a sigma that depends on the
# simulation).
# Pooling the same observable across simulations is "sigma_block" on the loss ELEMENT
# in Optimizer_settings.py; the two compose.
#   "reads_parameters": [names]   for a CALLABLE observed_variable only: the fitted parameters it reads by name.
#                                 Default: all of them (a callable could read anything), probed: one extra pass of the loss per
#                                 parameter in every AMICI gradient, and the four passes of its partial only for a parameter the
#                                 probe shows it reads (a parameter unread on three points is not probed again). A callable of the table alone reads none:
#                                 say "reads_parameters": [] and the gradient costs no extra passes. (An expression string
#                                 needs no declaration: its names are found in it.)

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

def Flipflop_loss_config(replicate):
    """Log10-objective loss for the flip-flop example (Example4/Example5).

    Both observables are fit in log10 space: the data columns already hold
    log10 values, and the observed_variable expressions log-transform the model
    output (with a floor, since predicted_B is exactly 0 before the dose event
    and log10(0) would otherwise poison the interpolation).

    The sigma_method/sigma_value entries only shape the *fitting* objective
    (they set the relative weight of the dense, precise logB data against the
    sparse, very noisy logA data). The dNLL diagnostics deliberately ignore
    them: profile/slice/Wald re-estimate each observable's sigma by MLE from
    the residuals at the optimum and freeze it (fixed_sigmas). Comparing the
    resulting profiles against Flipflop_reference.py checks that this
    re-scaling is done correctly — log10 objectives are where an incorrect
    sigma convention is most visible, because the heuristic sigma estimates
    (mean/std of the log data) are 10-20x larger than the actual residual
    scale in dex, which suppresses every dNLL by 2-3 orders of magnitude.
    """
    label = replicate["Label"]
    observables = [{
        "observed_variable": "np.log10(np.maximum(predicted_B, 1e-12))",
        "data_column":       "logB",
        "time_column":       "time",
        "data_dict_key":     label,
        "sigma_method":      "fixed",
        "sigma_value":       0.05,
    }]
    if replicate.get("has_A_data"):
        observables.append({
            "observed_variable": "np.log10(np.maximum(predicted_A, 1e-12))",
            "data_column":       "logA",
            "time_column":       "time",
            "data_dict_key":     f"{label}_A",
            "sigma_method":      "fixed",
            "sigma_value":       0.75,
        })
    return {"observables": observables}


# def Example2_loss_config(replicate):
#     label = replicate["Label"]
#     return {"observables": [{
#         "observed_variable": "predicted_A",
#         "data_column":       "A",
#         "time_column":       "time",
#         "data_dict_key":     label,
#     }]}


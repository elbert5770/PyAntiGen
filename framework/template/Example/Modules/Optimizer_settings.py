OPTIMIZATION_SETTINGS = {

    "Example": {
        "ADneg": {
            "param_names": ["k_A_to_B", "SF"],
            "x0": [0.5, 2.0],
            "bounds": [(0.01, 10.0), (0.01, 10.0)],
            "method": "Nelder-Mead",
            "optimizer_kwargs": {"options": {"maxiter": 500}},
            "wald_analysis": False,
            "slice_analysis": True,
            "profile_likelihood_analysis": False,
        },
        "ADpos": {
            "param_names": ["k_A_to_B", "SF"],
            "x0": [0.5, 2.0],
            "bounds": [(0.01, 10.0), (0.01, 10.0)],
            "method": "Nelder-Mead",
            "optimizer_kwargs": {"options": {"maxiter": 500}},
            "wald_analysis": False,
            "slice_analysis": False,
            "profile_likelihood_analysis": False,
        },
    }
}

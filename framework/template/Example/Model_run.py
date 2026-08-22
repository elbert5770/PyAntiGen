import os
import sys

from Engine.Model_simulate import setup_simulation
from Engine.Model_optimize import (
    setup_optimization_from_groups,
    _FULL_DIAGNOSTICS,
    _SLICE_ONLY,
    _PROFILE_ONLY,
    _SOBOL_ONLY,
    _FAST_PROFILE_ONLY,
    _NO_DIAGNOSTICS,
    DIAGNOSTICS_PRESETS,
)
from Modules.Plots import *
from Modules.Experiment import get_EXPERIMENT
from Modules.Optimizer_settings import get_OPTIMIZATION
from AntiGen_paths import MODEL_NAME, REPO_ROOT
from Model_generate import update_antimony_model

EXPERIMENT_dict = {
    "Example": {'EXPERIMENT': get_EXPERIMENT('EXPERIMENT_Example'), 'plot': plot_results, 'opt_settings_key': 'Example'},
    "Flipflop": {'EXPERIMENT': get_EXPERIMENT('EXPERIMENT_Flipflop'), 'plot': plot_flipflop, 'opt_settings_key': 'Flipflop'},
}

diagnostics = _NO_DIAGNOSTICS

OPTIMIZATION_REGISTRY = {
    # One CLI selection can run multiple independent Optimization specs in
    # sequence (e.g. parameters split across specs for identifiability
    # reasons); opt_key may be a single name or a list of names.
    # Optional "diagnostics" dict is merged into run_settings for every spec
    # run under that selection (wald/slice/profile-likelihood/Sobol flags).
    # 'title' / 'description' are printed before the run, 'interpretation'
    # after it, so the console output explains what the example demonstrates
    # and what to look for in the numbers and figures. Every figure and CSV
    # written by a run is prefixed with the example name (e.g.
    # Example_Example4_...) so the five examples never overwrite each other.
    "Example1": {
        'opt_key': ['OPTIMIZATION_Example1_ADpos', 'OPTIMIZATION_Example1_ADneg'],
        'experiment': 'EXPERIMENT_Example',
        'title': "Baseline well-posed fits, split by identifiability",
        'description': """\
Two independent optimizations run in sequence, no diagnostics:
  1. k_A_to_B and SF fit jointly against the ADpos replicates (they are
     jointly identifiable there).
  2. V_Comp1 fit alone against the ADneg replicates. It is kept OUT of the
     first fit because SF and V_Comp1 trade off against the same [B]
     trajectories -- fitting all three together lands on a ridge, not a
     minimum (Example3 demonstrates exactly that failure).""",
        'interpretation': """\
Both optimizations should report success=True with a small final loss.
Figures Example_Example1_ADpos.png and Example_Example1_ADneg.png show the
fitted [A]/[B] curves passing through the measured [B] points. Run Example2
for the same fits with full identifiability diagnostics.""",
    },
    "Example2": {
        'opt_key': ['OPTIMIZATION_Example1_ADpos', 'OPTIMIZATION_Example1_ADneg'],
        'experiment': 'EXPERIMENT_Example',
        'diagnostics': _FULL_DIAGNOSTICS,
        'title': "Example1 fits with full identifiability diagnostics",
        'description': """\
The same two fits as Example1, with every diagnostic enabled: Wald
statistics (Hessian-based SEs/CIs), likelihood slices (vary one parameter,
hold the rest), true profile likelihood (vary one parameter, RE-OPTIMIZE
the rest), and Sobol sensitivity indices. This is the positive control:
both fits are well-posed, so every diagnostic should agree.""",
        'interpretation': """\
What a well-identified fit looks like:
  * Wald SEs are finite; Wald and profile 95% CIs roughly agree.
  * Every profile-likelihood curve is a clean parabola-like bowl crossing
    the 95% threshold (dNLL = 1.92) on BOTH sides -> finite CIs, no [nan].
  * Slices and profiles nearly coincide (little parameter compensation).
  * No '*** FLAT ***' warnings.
Figures (prefix Example_Example2_<group>_): *_profile_likelihood.png,
*_likelihood_slice.png (+ _zoom variants), *_sobol.png; fit curves in
Example_Example2_ADpos.png / _ADneg.png. Contrast with Example3, where the
joint fit breaks these diagnostics in a recognizable way.""",
    },
    "Example4": {
        'opt_key': ['OPTIMIZATION_Example4_flipflop'],
        'experiment': 'EXPERIMENT_Flipflop',
        'diagnostics': _FULL_DIAGNOSTICS,
        'title': "Multimodal flip-flop kinetics, started in the TRUE basin",
        'description': """\
A -> B -> C with rates k_A_to_B, k_B_to_C and a fitted scale factor SF,
observed on a log10 scale. Swapping the two rate constants rescales B(t)
by a constant that SF absorbs exactly ('flip-flop'), so the likelihood has
a second local minimum at the swapped parameters. Four noisy predicted_A
points break the symmetry by a small, known amount. The fit starts inside
the true-mode basin (k_A_to_B > k_B_to_C). This example tests the ACCURACY
of profile-likelihood dNLL values, not just flat-direction detection.""",
        'interpretation': """\
What accurate diagnostics must show (reference values from
Flipflop_reference.py, which recomputes everything in closed form):
  * Likelihood slices: ONE sharp minimum. Slices cannot see the second
    mode -- reaching it requires the other two parameters to move.
  * Every true profile DIPS to dNLL ~ -2.12 next to the fit point. Not a
    bug: the fitting objective weights observables differently from the
    inference NLL the profiles are anchored to, so the fit optimum sits
    slightly off the NLL optimum. A profile that does not dip is wrong.
  * Each profile shows a SECOND minimum (the swapped mode) at dNLL ~ +0.8,
    BELOW the 1.92 threshold: the correct 95% confidence set is a union of
    two disjoint intervals even though the fit found the right mode. A
    profile walker that stops at the first threshold crossing misses it.
Compare figures Example_Example4_Flipflop_profile_likelihood*.png against
'python Flipflop_reference.py'; any disagreement is a bug in the profile
machinery, not the model. Then run Example5 for the wrong-basin version.""",
    },
    "Example5": {
        'opt_key': ['OPTIMIZATION_Example5_flipflop_swapped'],
        'experiment': 'EXPERIMENT_Flipflop',
        'diagnostics': _FULL_DIAGNOSTICS,
        'title': "Flip-flop kinetics, started in the WRONG (swapped) basin",
        'description': """\
Same problem as Example4 but started with k_A_to_B < k_B_to_C, so
Nelder-Mead converges to the swapped LOCAL minimum -- the situation
multimodal problems create in practice, where nobody tells you the
optimizer found the wrong mode. All sigmas are frozen by MLE at that local
optimum: the logA sigma absorbs the swapped mode's misfit (~1.73 dex
instead of ~0.92), which flattens every dNLL built on it.""",
        'interpretation': """\
The unambiguous signature that the fit missed the global optimum:
  * An accurate profile goes NEGATIVE, dipping to dNLL ~ -0.98 at the true
    mode. The analysis must report that dip, not clip or re-anchor it.
  * With the true mode at -0.98, BOTH modes lie below the 1.92 threshold:
    the correct 95% confidence set for every parameter is two disjoint
    intervals (e.g. k_A_to_B in [0.072, 0.076] U [0.284, 0.363]). A
    first-crossing CI extractor reports only the narrow interval around
    the WRONG mode and silently discards the one containing the truth.
Compare Example_Example5_Flipflop_profile_likelihood*.png against
'python Flipflop_reference.py --anchor swapped' (same wrong-mode anchor).""",
    },
    "Example3": {
        'opt_key': ['OPTIMIZATION_Example3_joint'],
        'experiment': 'EXPERIMENT_Example',
        'diagnostics': _FULL_DIAGNOSTICS,
        'title': "Negative example: structurally unidentifiable joint fit",
        'description': """\
k_A_to_B, SF, and V_Comp1 fit JOINTLY against the ADpos data -- the fit
Example1 deliberately avoids. The confound is exact: V_Comp1 cancels out
of the ODE entirely, and the output map predicted_B = SF*B_Comp1/V_Comp1
depends only on the ratio SF/V_Comp1, so any (SF, V_Comp1) pair with the
same ratio fits identically. Full diagnostics are on to show what
structural unidentifiability looks like.""",
        'interpretation': """\
The signature of a structurally unidentifiable pair:
  * Profile likelihood: k_A_to_B gets a tight, finite 95% CI, but SF and
    V_Comp1 come back [nan, nan] -- their profiles stay flat below the
    threshold because the optimizer trades one against the other to hold
    SF/V_Comp1 constant at identical loss.
  * Wald: SF/V_Comp1 correlation ~ +1 (or the Hessian is not positive
    definite), so their SEs are meaningless or absent.
  * CAVEAT -- Sobol says the OPPOSITE (large ST for SF and V_Comp1, ~0 for
    k_A_to_B), and that is expected, not a bug: Sobol perturbs parameters
    independently with no re-optimization, so it measures uncorrelated
    loss sensitivity, not identifiability. For a ridge these two questions
    have opposite answers. Treat Sobol as a sensitivity screen and profile
    likelihood as the identifiability check.
Figures carry the prefix Example_Example3_ADpos_.""",
    },
}


def _print_block(header, body):
    """Print a framed explanation block so it stands out in the run log."""
    bar = "=" * 78
    print(f"\n{bar}\n{header}\n{bar}")
    print(body)
    print(bar + "\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Tellurium/RoadRunner Simulation and Optimization Runner")
    parser.add_argument("--optimize", type=str, choices=list(OPTIMIZATION_REGISTRY.keys()),
                        help="Optimization mode")
    parser.add_argument("--simulate", type=str, choices=list(EXPERIMENT_dict.keys()),
                        help="Simulation mode")
    parser.add_argument("--diagnostics", type=str, default="_NO_DIAGNOSTICS",
                        choices=list(DIAGNOSTICS_PRESETS.keys()),
                        help="Diagnostics settings preset to run (default: _NO_DIAGNOSTICS)")
    parser.add_argument("--no-fit", action="store_true", dest="no_fit",
                        help="Skip the optimizer and evaluate the spec's x0 instead, "
                             "then run the requested diagnostics against it.")

    args = parser.parse_args()

    if len(sys.argv) == 1:
        parser.error("No arguments provided.")

    update_antimony_model()


# ---  Optimization ------------------------------------
    if args.optimize:
        opt_info = OPTIMIZATION_REGISTRY[args.optimize]
        model_name_to_use = MODEL_NAME

        run_settings = {
            "run_steady_state_first": False,
            "Verbose": True,
            "save_SBML?": False,
            "MODEL_NAME": model_name_to_use,
            "slice_analysis": False,
            "fit_mode": "evaluate_x0" if args.no_fit else "optimize",
            # Prefixes every figure/CSV name with the example name so runs
            # of different examples never overwrite each other's output.
            "run_label": args.optimize,
        }
        selected_diagnostics = DIAGNOSTICS_PRESETS.get(args.diagnostics, _NO_DIAGNOSTICS)
        run_settings.update(selected_diagnostics)
        if "diagnostics" in opt_info:
            run_settings.update(opt_info["diagnostics"])


        # Retrieve experiment object & optimization spec(s)
        exp_obj = get_EXPERIMENT(opt_info["experiment"])
        opt_keys = opt_info["opt_key"]
        if isinstance(opt_keys, str):
            opt_keys = [opt_keys]

        # Plot the fitted curves after each optimization with the experiment's
        # plot function (figures are tagged with the example/group names, e.g.
        # Example_Example1_ADpos.png).
        exp_short = opt_info["experiment"].replace("EXPERIMENT_", "")
        experiment_arg = {
            "EXPERIMENT": exp_obj,
            "plot": EXPERIMENT_dict.get(exp_short, {}).get("plot"),
        }

        if opt_info.get("title"):
            _print_block(f"{args.optimize}: {opt_info['title']}",
                         opt_info.get("description", ""))

        for opt_key in opt_keys:
            opt_spec = get_OPTIMIZATION(opt_key)
            print(f"Starting domain optimization for: {args.optimize} [{opt_key}]")
            setup_optimization_from_groups(run_settings, opt_spec, experiment_arg)

        if opt_info.get("interpretation"):
            _print_block(f"{args.optimize}: what to look for in the results",
                         opt_info["interpretation"])

# --- Simulation ---------------------------------------------
    elif args.simulate:
        run_name = args.simulate
        fig_config = EXPERIMENT_dict[run_name]
        model_name_to_use = MODEL_NAME


        run_settings = {
            "run_steady_state_first": False,
            "Verbose": True,
            "save_SBML?": False,
            "MODEL_NAME": model_name_to_use
        }

        setup_simulation(run_settings, fig_config)



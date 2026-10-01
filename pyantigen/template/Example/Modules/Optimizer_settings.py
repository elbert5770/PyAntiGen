from dataclasses import dataclass, field
from .Loss_config import *

@dataclass
class Optimization:
    param_names: list
    x0: list
    bounds: list = None
    method: str = "Nelder-Mead"
    optimizer_kwargs: dict = field(default_factory=dict)
    group_normalization: str = "mean_over_groups"  # "mean_over_groups" | "sum_over_groups"
    groups: dict = field(default_factory=dict)       # nested group/loss configuration
    passive_simulations: list = field(default_factory=list) # passive simulations to run for plotting
    # Per-parameter optimizer scale: None/"lin" keeps the parameter linear,
    # "log10" fits log10(p) instead. Accepts a single string for all parameters,
    # a {name: scale} dict, or a list aligned with param_names. x0 and bounds
    # stay in linear units; so does everything reported back.
    parameter_scale: object = None


def _build_example_opt1_ADpos():
    """
    "k_A_to_B" and "SF" are jointly identifiable from the ADpos data, so they
    are fit together in one optimization against the ADpos replicates.
    """
    return Optimization(
        param_names=["k_A_to_B", "SF"],
        x0=[0.5, 2.0],
        bounds=[(0.01, 10.0), (0.01, 10.0)],
        method="Nelder-Mead",
        optimizer_kwargs={"options": {"maxiter": 500}},
        group_normalization="mean_over_groups",
        groups={
            "ADpos": {
                "group_weight": 1.0,
                "loss_elements": [
                    {"simulation": "ADpos_Early", "loss_config": Example1_loss_config, "weight": 1.0},
                    {"simulation": "ADpos_Late",  "loss_config": Example1_loss_config, "weight": 1.0},
                ]
            }
        },
        passive_simulations=[],
    )


def _build_example_opt1_ADneg():
    """
    "V_Comp1" is fit separately, against the ADneg data. It is not folded
    into the ADpos optimization above because it is not jointly identifiable
    with "SF" there: both trade off against the same [B] trajectories, so a
    combined fit lands on a ridge rather than a minimum. Splitting the fit by
    group (rather than lumping all three parameters into one optimization)
    keeps each individual fit well-posed.
    """
    return Optimization(
        param_names=["V_Comp1"],
        x0=[0.5],
        bounds=[(0.01, 10.0)],
        method="Nelder-Mead",
        optimizer_kwargs={"options": {"maxiter": 500}},
        group_normalization="mean_over_groups",
        groups={
            "ADneg": {
                "group_weight": 1.0,
                "loss_elements": [
                    {"simulation": "ADneg_Early", "loss_config": Example1_loss_config, "weight": 1.0},
                    {"simulation": "ADneg_Late",  "loss_config": Example1_loss_config, "weight": 1.0},
                ]
            }
        },
        passive_simulations=[],
    )

OPTIMIZATION_Example1_ADpos = _build_example_opt1_ADpos()
OPTIMIZATION_Example1_ADneg = _build_example_opt1_ADneg()


def _build_example3_joint():
    """
    NEGATIVE EXAMPLE: fits "k_A_to_B", "SF", and "V_Comp1" jointly against the
    ADpos data, instead of splitting "V_Comp1" out as Example1 does. This is
    the fit Example1 deliberately avoids.

    The confound is exact, not approximate: the reaction rate law is
    "k_A_to_B * (A_Comp1/V_Comp1) * V_Comp1", so V_Comp1 cancels out of the
    ODE entirely (see antimony_models/Example/Example_reactions.txt) and
    B_Comp1(t) never depends on it. The only place V_Comp1 appears is the
    output map "predicted_B := SF * B_Comp1 / V_Comp1" (Example_manual.txt),
    so predicted_B depends only on the ratio SF/V_Comp1 -- any (SF, V_Comp1)
    pair with the same ratio fits identically. "SF" and "V_Comp1" are
    therefore structurally unidentifiable together; only "k_A_to_B" (which
    shapes the amount trajectory itself) is well-identified here.

    Run with Example2's diagnostics settings to see the signature of this in
    practice: profile likelihood gives k_A_to_B a tight, finite 95% CI while
    SF and V_Comp1 both come back [nan, nan] (their profiles never cross the
    threshold, because the optimizer can always trade one against the other
    to hold SF/V_Comp1 constant and recover the identical loss).

    CAVEAT -- Sobol says the opposite, and that is expected, not a bug:
    run_sobol_analysis samples k_A_to_B, SF, and V_Comp1 *independently* and
    uniformly across their bounds and measures variance of the raw loss, with
    no re-optimization. Since the model only depends on the ratio SF/V_Comp1,
    independently randomizing them over a 1000x range almost always lands far
    off the SF/V_Comp1 = const ridge that fits the data -- that scale mismatch
    dominates the loss variance, so SF and V_Comp1 get large ST (in a run of
    this example, ST(V_Comp1)=4.25, ST(SF)=0.64) while k_A_to_B's genuine but
    comparatively small effect on trajectory timing gets washed out
    (ST(k_A_to_B)=0.00000), flipping the "identifiable" (ST > 0.01) label onto
    exactly the two confounded parameters and off the one real one. Sobol-on-
    loss measures uncorrelated sensitivity ("does perturbing this parameter
    alone move the loss"), not identifiability ("can the other parameters
    compensate for it") -- for a ridge/confound like this, those two
    questions have opposite answers. Treat Sobol here as a sensitivity
    screen, and profile likelihood as the identifiability check.
    """
    return Optimization(
        param_names=["k_A_to_B", "SF", "V_Comp1"],
        x0=[0.5, 2.0, 0.5],
        bounds=[(0.01, 10.0), (0.01, 10.0), (0.01, 10.0)],
        method="Nelder-Mead",
        optimizer_kwargs={"options": {"maxiter": 500}},
        group_normalization="mean_over_groups",
        groups={
            "ADpos": {
                "group_weight": 1.0,
                "loss_elements": [
                    {"simulation": "ADpos_Early", "loss_config": Example1_loss_config, "weight": 1.0},
                    {"simulation": "ADpos_Late",  "loss_config": Example1_loss_config, "weight": 1.0},
                ]
            }
        },
        passive_simulations=[],
    )

OPTIMIZATION_Example3_joint = _build_example3_joint()


def _flipflop_groups():
    return {
        "Flipflop": {
            "group_weight": 1.0,
            "loss_elements": [
                {"simulation": "Flipflop_Early", "loss_config": Flipflop_loss_config, "weight": 1.0},
                {"simulation": "Flipflop_Late",  "loss_config": Flipflop_loss_config, "weight": 1.0},
            ],
        }
    }


def _build_example4_flipflop():
    """
    MULTIMODAL EXAMPLE: tests the *accuracy* of profile likelihood dNLL, not
    just its ability to flag a flat direction (which Example3 covers).

    The model is the chain A -> B -> C (rates k_A_to_B, k_B_to_C) observed
    through predicted_B := SF*B_Comp1/V_Comp1 on a log10 scale. Since

        B(t) = dose * k_A_to_B * (exp(-k_A_to_B*t) - exp(-k_B_to_C*t))
               / (k_B_to_C - k_A_to_B),

    swapping the two rate constants rescales B by a constant factor that the
    fitted SF absorbs exactly: (k1, k2, SF) and (k2, k1, SF*k1/k2) produce
    IDENTICAL predicted_B trajectories (pharmacokinetic "flip-flop"). Four
    deliberately noisy predicted_A points on the Early treatment break the
    symmetry by a small, known amount: relative to the NLL optimum the
    swapped mode is a genuine local minimum at dNLL ~ 2.4, just above the
    95% threshold of 1.9207 (printed by data/make_flipflop_data.py).

    What accurate diagnostics must show here (numbers from
    Flipflop_reference.py, which replicates the full pipeline):
      * likelihood slices: a single sharp minimum — slices cannot see the
        second mode at all, because reaching it requires the OTHER two
        parameters to move (k_B_to_C and SF swap along with k_A_to_B).
      * every true profile dips to dNLL ~ -2.12 next to the fit point. That
        is not an error: the FITTING objective averages each observable's
        chi-square over its own points, which upweights the 4 noisy logA
        points ~11x against 45 logB points, so the fit optimum (k_A_to_B ~
        0.363) sits measurably away from the NLL optimum (~0.324) that
        diagnostics are anchored to. A profile that does not dip is wrong.
      * true profile likelihood for each parameter: TWO minima. Because of
        that anchor offset, the swapped mode (k_A_to_B ~ 0.074, k_B_to_C ~
        0.32, SF ~ 7.2) reports at dNLL ~ +0.8 — BELOW the threshold — so
        the correct 95% confidence set is a union of two disjoint intervals
        even though the fit started in the right basin. A profile walker
        that stops at the first threshold crossing never discovers the
        second mode, and a first-crossing CI extractor cannot represent the
        disjoint set; both failures make the reported CI silently
        conditional on the wrong assumption of unimodality. And if the dNLL
        scale is off (the log10-objective sigma pitfalls described in
        Flipflop_loss_config), everything above shifts across the threshold
        and the CI changes shape entirely.

    Flipflop_reference.py recomputes the exact profiles from the closed-form
    solution with scipy (no RoadRunner, no framework loss code) under the same
    frozen-sigma convention — run it and overlay the two sets of curves. Any
    disagreement is an error in the profile machinery, not in the model.
    """
    return Optimization(
        param_names=["k_A_to_B", "k_B_to_C", "SF"],
        x0=[0.3, 0.08, 1.5],  # inside the true-mode basin (k_A_to_B > k_B_to_C)
        bounds=[(0.005, 5.0), (0.005, 5.0), (0.05, 50.0)],
        method="Nelder-Mead",
        optimizer_kwargs={"options": {"maxiter": 2000, "xatol": 1e-8, "fatol": 1e-10}},
        group_normalization="sum_over_groups",
        parameter_scale="log10",
        groups=_flipflop_groups(),
        passive_simulations=[],
    )


def _build_example5_flipflop_swapped():
    """
    Same problem as Example4 but started inside the WRONG basin
    (k_A_to_B < k_B_to_C), so Nelder-Mead converges to the swapped local
    minimum. This stresses the diagnostics in the way multimodal problems
    stress them in practice, where nobody tells you the optimizer found the
    wrong mode:

      * every sigma is frozen by MLE at the *local* optimum. The logA sigma
        absorbs the swapped mode's A-misfit (~1.73 dex instead of ~0.92), and
        that inflation deflates every dNLL built on it — the same
        log10-objective sigma sensitivity described in Flipflop_loss_config,
        arising here from mode choice rather than from a heuristic;
      * an accurate profile must go NEGATIVE — dropping to dNLL ~ -0.98 at
        the true mode (not the -2.4 the true-mode sigmas would give: the
        inflated sigma has already flattened the landscape). A negative
        profile minimum is the unambiguous signature that the fit missed the
        global optimum, and the analysis must report it rather than clip,
        re-anchor, or hide it;
      * with the true mode at -0.98, BOTH modes lie below the 1.9207
        threshold: the correct 95% confidence set for every parameter is a
        union of two disjoint intervals (e.g. k_A_to_B in [0.072, 0.076] U
        [0.284, 0.363]), which a first-threshold-crossing CI extractor
        cannot represent — it reports the narrow interval around the wrong
        mode and silently discards the one containing the truth.

    Compare against Flipflop_reference.py --anchor swapped, which computes
    the reference profiles anchored at the same wrong mode.
    """
    return Optimization(
        param_names=["k_A_to_B", "k_B_to_C", "SF"],
        x0=[0.06, 0.4, 7.0],  # inside the swapped-mode basin
        bounds=[(0.005, 5.0), (0.005, 5.0), (0.05, 50.0)],
        method="Nelder-Mead",
        optimizer_kwargs={"options": {"maxiter": 2000, "xatol": 1e-8, "fatol": 1e-10}},
        group_normalization="sum_over_groups",
        parameter_scale="log10",
        groups=_flipflop_groups(),
        passive_simulations=[],
    )

OPTIMIZATION_Example4_flipflop = _build_example4_flipflop()
OPTIMIZATION_Example5_flipflop_swapped = _build_example5_flipflop_swapped()

def get_OPTIMIZATION(name):
    """Returns a single Optimization configuration by name."""
    return globals().get(name)

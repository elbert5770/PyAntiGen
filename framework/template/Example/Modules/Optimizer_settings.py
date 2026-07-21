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

def get_OPTIMIZATION(name):
    """Returns a single Optimization configuration by name."""
    return globals().get(name)

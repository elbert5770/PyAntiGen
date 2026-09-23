"""The template Example as v2 Studies.

Two studies, each the Study-form of what Modules/Experiment.py and
Modules/Optimizer_settings.py define for 1.x:

  example   two cohorts (amyloid-negative and -positive) x two dosing
            treatments, [B] measured three times per time point.
  flipflop  one synthetic subject x two treatments, log10 [B] plus four
            noisy log10 [A] points on the Early treatment only.

Build them in Python -- loops and functions are allowed here -- and write
the design record with ``pyantigen.study.save(study, "example.json")``.
The estimations below reproduce the 1.x specs of the same names; see
tests/study/test_example_equivalence.py, which checks that each gives the
same objective as its 1.x original.
"""
from pyantigen.study import (Estimation, Measured, Noise, Obs, DataSource, Study)

from Modules.Events import Example_event
from Modules.Observed_species import all_species
from Modules.Solver_settings import solver_settings_Example


def build_example():
    s = Study("example")
    s.factor("status", {
        # The 1.x update_Example set k_A_to_B from amyloid_positive; here the
        # level that implies it says so.
        "ADneg": {"amyloid_positive": False, "params": {"k_A_to_B": 0.1}},
        "ADpos": {"amyloid_positive": True,  "params": {"k_A_to_B": 0.2}},
    })
    s.factor("treatment", {
        "Early": {"dose": 10, "delay": 5},
        "Late":  {"dose": 5,  "delay": 10},
    })
    for status in ("ADneg", "ADpos"):
        s.subject(status, kind="cohort", status=status)
    s.protocol("dose_at_delay", events=Example_event,
               solver=solver_settings_Example, observed=all_species)
    s.cross(["ADneg", "ADpos"], "dose_at_delay", treatment="*")

    s.assay("B", Measured(
        "B", Obs("predicted_B"),
        DataSource("{status}.csv", time="time", value=["B1", "B2", "B3"],
                   where={"Treatment": "{treatment}"})))
    s.measure("B")

    s.param("k_A_to_B", x0=0.5, bounds=(0.01, 10.0))
    s.param("SF", x0=2.0, bounds=(0.01, 10.0))
    s.param("V_Comp1", x0=0.5, bounds=(0.01, 10.0))
    return s


EXAMPLE_ESTIMATIONS = {
    # 1.x OPTIMIZATION_Example1_ADpos / _ADneg / Example3_joint
    "Example1_ADpos": Estimation("Example1_ADpos", params=["k_A_to_B", "SF"],
                                 on={"status": "ADpos"},
                                 optimizer_kwargs={"options": {"maxiter": 500}}),
    "Example1_ADneg": Estimation("Example1_ADneg", params=["V_Comp1"],
                                 on={"status": "ADneg"},
                                 optimizer_kwargs={"options": {"maxiter": 500}}),
    "Example3_joint": Estimation("Example3_joint", params=["k_A_to_B", "SF", "V_Comp1"],
                                 on={"status": "ADpos"},
                                 optimizer_kwargs={"options": {"maxiter": 500}}),
}


def build_flipflop():
    s = Study("flipflop")
    # The synthetic data's generating values, which --simulate should
    # reproduce. A fit overwrites them; resolve() applies fitted values after
    # level params.
    s.factor("generator", {"truth": {"params": {"k_A_to_B": 0.35, "k_B_to_C": 0.07, "SF": 1.6}}})
    s.factor("treatment", {
        "Early": {"dose": 10, "delay": 5,  "has_A_data": True},
        "Late":  {"dose": 5,  "delay": 10, "has_A_data": False},
    })
    s.subject("Flipflop", kind="cohort", generator="truth")
    s.protocol("dose_at_delay", events=Example_event,
               solver=solver_settings_Example, observed=all_species)
    s.cross(["Flipflop"], "dose_at_delay", treatment="*")

    s.assay("flipflop",
            Measured("logB", Obs.log10("predicted_B"),
                     DataSource("Flipflop.csv", time="time",
                                value=["logB1", "logB2", "logB3"],
                                where={"Treatment": "{treatment}"}),
                     Noise(sigma=0.05)),
            Measured("logA", Obs.log10("predicted_A"),
                     DataSource("Flipflop.csv", time="time", value="logA",
                                where={"Treatment": "{treatment}"}),
                     Noise(sigma=0.75),
                     only={"has_A_data": True}))
    s.measure("flipflop")

    for name, x0, b in (("k_A_to_B", 0.3, (0.005, 5.0)),
                        ("k_B_to_C", 0.08, (0.005, 5.0)),
                        ("SF", 1.5, (0.05, 50.0))):
        s.param(name, x0=x0, bounds=b, scale="log10")
    return s


FLIPFLOP_ESTIMATIONS = {
    "Example4_flipflop": Estimation(
        "Example4_flipflop",
        optimizer_kwargs={"options": {"maxiter": 2000, "xatol": 1e-8, "fatol": 1e-10}}),
}


if __name__ == "__main__":
    # Rewrite the design records beside this file. They are what a reviewer
    # (or an LLM) reads and diffs; tests check they match what this file builds.
    import os
    from pyantigen.study import save
    here = os.path.dirname(os.path.abspath(__file__))
    for st in (build_example(), build_flipflop()):
        print("wrote", save(st, os.path.join(here, f"{st.name}.json")))

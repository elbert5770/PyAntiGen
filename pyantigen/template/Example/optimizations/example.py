"""The template Example's fits, as Optimizations.

Each reproduces the 1.x spec of the same name in Modules/Optimizer_settings.py:
which datasets of which study are scored, and which parameters they fit.
The studies themselves (studies/example.py) make no fitting choices.

    from optimizations.example import build
    fit = build("Example1_ADpos")

Write the records with ``python -m optimizations.example``.
"""
from pyantigen.study import Optimization, Param

from studies.example import build_example, build_flipflop

NM_500 = {"options": {"maxiter": 500}}


def _example1_adpos(example, flipflop):
    # k_A_to_B and SF are jointly identifiable from the ADpos data.
    fit = Optimization("Example1_ADpos", params=[
        Param("k_A_to_B", x0=0.5, bounds=(0.01, 10.0)),
        Param("SF", x0=2.0, bounds=(0.01, 10.0))], optimizer_kwargs=NM_500)
    fit.use(example, assays=["B"], on={"status": "ADpos"})
    return fit


def _example1_adneg(example, flipflop):
    # V_Comp1 alone, against the ADneg data: with SF it would sit on a ridge.
    fit = Optimization("Example1_ADneg", params=[
        Param("V_Comp1", x0=0.5, bounds=(0.01, 10.0))], optimizer_kwargs=NM_500)
    fit.use(example, assays=["B"], on={"status": "ADneg"})
    return fit


def _example3_joint(example, flipflop):
    # The negative example: SF and V_Comp1 are exactly confounded here.
    fit = Optimization("Example3_joint", params=[
        Param("k_A_to_B", x0=0.5, bounds=(0.01, 10.0)),
        Param("SF", x0=2.0, bounds=(0.01, 10.0)),
        Param("V_Comp1", x0=0.5, bounds=(0.01, 10.0))], optimizer_kwargs=NM_500)
    fit.use(example, assays=["B"], on={"status": "ADpos"})
    return fit


def _example4_flipflop(example, flipflop):
    fit = Optimization("Example4_flipflop", params=[
        Param("k_A_to_B", x0=0.3, bounds=(0.005, 5.0), scale="log10"),
        Param("k_B_to_C", x0=0.08, bounds=(0.005, 5.0), scale="log10"),
        Param("SF", x0=1.5, bounds=(0.05, 50.0), scale="log10")],
        optimizer_kwargs={"options": {"maxiter": 2000, "xatol": 1e-8, "fatol": 1e-10}})
    fit.use(flipflop)
    return fit


BUILDERS = {"Example1_ADpos": _example1_adpos, "Example1_ADneg": _example1_adneg,
            "Example3_joint": _example3_joint, "Example4_flipflop": _example4_flipflop}


def studies():
    """{name: Study} for every study these optimizations use."""
    return {st.name: st for st in (build_example(), build_flipflop())}


def build(name, studies_by_name=None):
    st = studies_by_name or studies()
    return BUILDERS[name](st["example"], st["flipflop"])


if __name__ == "__main__":
    import os
    from pyantigen.study import save_optimization
    here = os.path.dirname(os.path.abspath(__file__))
    st = studies()
    for name in BUILDERS:
        print("wrote", save_optimization(build(name, st), os.path.join(here, f"{name}.json")))

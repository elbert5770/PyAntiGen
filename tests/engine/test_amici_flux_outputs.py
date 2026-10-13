"""A rule that reads a reaction's rate sees the unscaled flux in the scaled-time model (pyantigen.generate.scaled_time).

``scale_model`` multiplies every rate law by ``sched_Delta`` (the length of the segment). A reaction id in a
formula means the reaction's rate, so ``F := J1`` used to read ``Delta*J1``: right on a segment of length 1, wrong
by the segment's length on every other. Species and ratios of species never noticed; an output such as
BBB_transport (a sum of reaction ids) was off by a factor of 5e5 on a 100-year window.

The text checks run anywhere; the AMICI checks need AMICI and a compiler (WSL on Windows).
"""
import numpy as np
import pytest

tellurium = pytest.importorskip("tellurium")
libsbml = pytest.importorskip("libsbml")

from pyantigen.generate.input_schedule import InputSchedule  # noqa: E402
from pyantigen.generate.scaled_time import DELTA, scale_model  # noqa: E402

K, A0 = 0.3, 2.0
MODEL = f"""
A = {A0}; kdec = {K}
J1: A -> ; kdec*A
F := J1
G := 2*J1 + A
"""


def _scaled():
    sbml = tellurium.loada(MODEL).getSBML()
    return sbml, scale_model(sbml, InputSchedule({}))


def test_a_rule_that_reads_a_reaction_id_reads_it_divided_by_delta():
    _sbml, scaled = _scaled()
    model = libsbml.readSBMLFromString(scaled.sbml).getModel()
    for rule_var in ("F", "G"):
        text = libsbml.formulaToL3String(model.getRuleByVariable(rule_var).getMath())
        assert f"J1 / {DELTA}" in text or f"(J1/{DELTA})" in text.replace(" ", ""), text
    # the rate law itself is the one that carries Delta
    law = libsbml.formulaToL3String(model.getReaction("J1").getKineticLaw().getMath())
    assert DELTA in law


def test_a_model_whose_rules_read_no_reaction_is_left_alone():
    sbml = tellurium.loada("A = 1; kdec = 0.3\nJ1: A -> ; kdec*A\nH := 3*A").getSBML()
    scaled = scale_model(sbml, InputSchedule({}))
    model = libsbml.readSBMLFromString(scaled.sbml).getModel()
    assert DELTA not in libsbml.formulaToL3String(model.getRuleByVariable("H").getMath())


OBS = [0.5, 1.0, 3.0, 7.0]          # segments of length 0.5, 0.5, 2, 4: Delta is not 1 on any of them


@pytest.fixture(scope="module")
def runner(tmp_path_factory):
    pytest.importorskip("amici")
    from pyantigen.engine.Scaled_amici import AmiciSegmentRunner, compile_model, concentration_outputs
    sbml, scaled = _scaled()
    module = compile_model(scaled, "flux_toy", tmp_path_factory.mktemp("amici_flux"), free=["kdec"],
                           outputs=concentration_outputs(sbml, ["F", "G", "[A]"]))
    return AmiciSegmentRunner(scaled, InputSchedule({}), module, states=[])


def test_amici_outputs_that_read_fluxes_match_the_closed_form(runner):
    y, sy = runner.run_outputs({"kdec": K}, 0.0, 7.0, OBS, wrt=["kdec"])
    names = list(runner.output_names)
    t = np.array(OBS)
    a = A0 * np.exp(-K * t)
    flux = K * a
    np.testing.assert_allclose(y[:, names.index("[A]")], a, rtol=1e-7)
    np.testing.assert_allclose(y[:, names.index("F")], flux, rtol=1e-7)
    np.testing.assert_allclose(y[:, names.index("G")], 2 * flux + a, rtol=1e-7)
    # d/dk of F = k*A0*exp(-k t) is A0*exp(-k t)*(1 - k t)
    np.testing.assert_allclose(sy[:, names.index("F"), 0], A0 * np.exp(-K * t) * (1 - K * t), rtol=1e-6)

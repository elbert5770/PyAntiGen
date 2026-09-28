"""Q and Reagent: the protocol layer's value objects.

The governing property, checked first and hardest: declaring either one
cannot change what is simulated. Everything else here is ordinary unit
arithmetic.

Run from the repository root: python -m pytest tests/study
"""
import pytest

from pyantigen.study import (Potency, Q, Reagent, Study, describe, fingerprint,
                             from_dict, resolve, to_dict, validate)
from pyantigen.study.quantity import UnitError, dimension_name, parse_unit


def _events(replicate, df_dict, r_ic=None):
    return ""


def _solver(replicate):
    return {}


def _observed(r):
    return ["time"]


def _study_with_reagents():
    s = Study("paper", doi="10.0000/x")
    s.factor("dose", {"vehicle": {"mgkg": 0}, "high": {"mgkg": 240}})
    s.subject("cohort", kind="cohort", covariates={"Species": "Rhesus"})
    s.protocol("p", _events, _solver, _observed)
    s.simulate("cohort", "p", dose="vehicle")
    s.simulate("cohort", "p", dose="high")
    s.reagent("water", id="CHEBI:15377")
    s.reagent("drug", id="DrugBank:DB00000", label="MK-0752", vehicle="water",
              potencies=(Potency("IC50", Q(5, "nmol/L"), "SH-SY5Y CVCL_0019",
                                 source="Cook 2010 p. 6744"),))
    return s


# --- the property that matters ------------------------------------------

def test_declaring_reagents_changes_no_simulation():
    """A reagent is recorded, printed, and read by nothing that simulates."""
    plain = _study_with_reagents()
    plain.reagents.clear()
    rich = _study_with_reagents()

    assert [s.id for s in plain.simulations.values()] == \
           [s.id for s in rich.simulations.values()]
    for sid in plain.simulations:
        assert plain.attributes(sid) == rich.attributes(sid)
        assert resolve(plain, sid) == resolve(rich, sid)


def test_a_reagent_is_not_a_simulation_attribute():
    s = _study_with_reagents()
    assert "drug" not in s._attribute_names()
    for sim in s.simulations.values():
        assert "drug" not in s.attributes(sim)
    with pytest.raises(KeyError):
        s.select(drug="drug")


def test_reagents_change_the_fingerprint_because_they_are_recorded():
    """They are part of the design record, so they are part of its hash --
    the point being that they are recorded, not that they are inert."""
    plain = _study_with_reagents()
    plain.reagents.clear()
    assert fingerprint(plain) != fingerprint(_study_with_reagents())


def test_a_study_without_reagents_is_untouched_by_this_version():
    """The version number says what the FILE needs, so adding the feature
    did not re-stamp -- or re-hash -- designs that do not use it. Every
    Optimization pinning a study fingerprint stays valid."""
    s = _study_with_reagents()
    s.reagents.clear()
    assert to_dict(s)["pyantigen_design"] == 4
    assert to_dict(_study_with_reagents())["pyantigen_design"] == 5


# --- Q --------------------------------------------------------------------

def test_conversion_between_commensurable_units():
    assert Q(1, "L").in_("mL") == pytest.approx(1000.0)
    assert Q(1.5, "mL").in_("L") == pytest.approx(0.0015)
    assert Q(1, "h").in_("min") == pytest.approx(60.0)
    assert Q(240, "mg/kg").in_("g/kg") == pytest.approx(0.24)


def test_an_atom_beats_a_prefix_reading():
    """'h' is an hour, 'd' a day, 'min' a minute, 'a' a year -- not
    hecto-, deci-, milli- or atto-anything."""
    assert Q(1, "h").in_("s") == pytest.approx(3600.0)
    assert Q(1, "d").in_("h") == pytest.approx(24.0)
    assert Q(1, "min").in_("s") == pytest.approx(60.0)
    assert Q(1, "a").in_("d") == pytest.approx(365.25)


def test_the_drain_rate_derivation_runs():
    """The number that was aliased between two papers, derived instead.

    Dobrowolska 2014: 16 draws of 1.5 mL over a 144 h window.
    """
    rate = 16 * Q(1.5, "mL") / Q(144, "h")
    assert rate.in_("L/h") == pytest.approx(16 * 0.0015 / 144)
    # Cook 2010's own 48 h window is a different number, and cannot be this one
    cook = 15 * Q(1.0, "mL") / Q(48, "h")
    assert not rate.isclose(cook)


def test_equality_is_semantic_across_units():
    assert Q(1, "L") == Q(1000, "mL")
    assert Q(1, "L") != Q(1, "mL")
    assert hash(Q(1, "L")) == hash(Q(1000, "mL"))


def test_arithmetic_keeps_dimensions_and_refuses_nonsense():
    assert (Q(2, "mg") + Q(1000, "ug")).in_("mg") == pytest.approx(3.0)
    assert (Q(4, "mg/kg/h") * Q(12, "h")).in_("mg/kg") == pytest.approx(48.0)
    with pytest.raises(UnitError):
        Q(1, "mL") + Q(1, "h")
    with pytest.raises(UnitError):
        Q(1, "mL").in_("h")


def test_a_mole_mass_conversion_is_refused_and_says_why():
    with pytest.raises(UnitError, match="molar mass"):
        Q(5, "nmol/L").in_("ug/mL")


def test_percent_is_dimensionless_and_converts_to_a_fraction():
    assert Q(15, "%").in_("1") == pytest.approx(0.15)
    assert Q(15, "%").is_dimensionless()


def test_unknown_units_are_refused_rather_than_guessed():
    for bad in ("furlong", "mg/", "m--", "3mg"):
        with pytest.raises(UnitError):
            parse_unit(bad)


def test_text_round_trip():
    for text in ("1.5 mL", "4 mg/kg/h", "240 mg", "0.15"):
        assert str(Q.parse(text)) == text
        assert Q.from_json(Q.parse(text).to_json()) == Q.parse(text)


def test_dimension_names_are_readable():
    assert dimension_name(parse_unit("mL")[1]) == "L^3"
    assert dimension_name(parse_unit("1")[1]) == "dimensionless"


# --- Reagent --------------------------------------------------------------

def test_reagent_round_trips_through_json_with_its_vehicle_and_potency():
    s = _study_with_reagents()
    back = from_dict(to_dict(s))
    r = back.reagents["drug"]
    assert r.id == "DrugBank:DB00000"
    assert r.label == "MK-0752"
    assert r.vehicle_name == "water"
    assert r.potency("IC50")[0].value == Q(5, "nmol/L")
    assert r.potency("IC50")[0].system == "SH-SY5Y CVCL_0019"
    assert to_dict(back) == to_dict(s)


def test_a_nested_vehicle_is_a_reagent_and_round_trips():
    s = Study("p")
    s.reagent("stock", id="CHEBI:1", concentration=Q(7.5, "mg/mL"),
              vehicle=Reagent("saline", id="CHEBI:75958"))
    back = from_dict(to_dict(s))
    stock = back.reagents["stock"]
    assert stock.concentration == Q(7.5, "mg/mL")
    assert isinstance(stock.vehicle, Reagent)
    assert stock.vehicle.id == "CHEBI:75958"
    assert [r.name for r in stock.mixture()] == ["stock", "saline"]


def test_validate_warns_about_a_missing_id_and_errors_on_a_dangling_vehicle():
    s = _study_with_reagents()
    s.reagent("mystery")
    s.reagent("orphan", id="CHEBI:2", vehicle="nowhere")
    probs = [str(p) for p in validate(s)]
    assert any("reagent 'mystery' has no id" in p and p.startswith("[warning]")
               for p in probs)
    assert any("'nowhere' is not a declared reagent" in p and p.startswith("[error]")
               for p in probs)
    # the ones that are properly declared say nothing
    assert not any("'drug'" in p or "'water'" in p for p in probs)


def test_describe_lists_reagents_once_not_per_simulation():
    text = describe(_study_with_reagents())
    assert text.count("reagent drug:") == 1
    assert "IC50 5 nmol/L in SH-SY5Y CVCL_0019" in text


def test_a_format_4_design_still_loads_and_gains_no_reagents():
    s = _study_with_reagents()
    s.reagents.clear()
    old = to_dict(s)
    old["pyantigen_design"] = 4
    back = from_dict(old)
    assert back.reagents == {}
    assert [sim.id for sim in back.simulations.values()] == \
           [sim.id for sim in s.simulations.values()]


def test_a_reagent_needs_a_name():
    with pytest.raises(ValueError):
        Reagent("")

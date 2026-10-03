"""Reaction/species value objects and the small pure helpers in pyantigen.generate.

None of this needs tellurium or the filesystem.

Run from the repository root: python -m pytest tests/generate
"""
import pytest

from pyantigen.generate.isotopomer_tools import ensure_isotopes_format
from pyantigen.generate.models import (
    RATE_TYPES_TWO_CONSTANTS,
    VALID_RATE_TYPES,
    Reaction,
    Species,
    normalize_species_list,
    parse_rate_equation_list,
    parse_species_list,
    reaction_from_args,
)
from pyantigen.generate.module_base import PyAntiGenModule
from pyantigen.generate.rate_laws import (
    CustomLaw,
    MassActionLaw,
    VolumeTransportLaw,
    get_rate_law_info,
)
from pyantigen.generate.reaction_creation import reaction_creation


# --- species list parsing -------------------------------------------------

@pytest.mark.parametrize("text, expected", [
    ("[A, B, C]", ["A", "B", "C"]),
    ("[A] + [B]", ["A", "B"]),
    ("[A]+[B]", ["A", "B"]),
    ("A + B", ["A", "B"]),
    ("A", ["A"]),
    ("[A]", ["A"]),
    ("", []),
])
def test_parse_species_list(text, expected):
    assert parse_species_list(text) == expected


@pytest.mark.parametrize("value, expected", [
    (None, []),
    ("", []),
    ("0", []),
    ("[0]", []),
    (["A", " B ", "", None], ["A", "B"]),
    ("[A, B]", ["A", "B"]),
])
def test_normalize_species_list(value, expected):
    assert normalize_species_list(value) == expected


@pytest.mark.parametrize("proto, expected", [
    (None, []),
    ("", []),
    ("k", ["k"]),
    ("[kf, kr]", ["kf", "kr"]),
    (["kf", " kr "], ["kf", "kr"]),
])
def test_parse_rate_equation_list(proto, expected):
    assert parse_rate_equation_list(proto, "RMA") == expected


# --- Species ----------------------------------------------------------------

def test_species_compartment_is_last_underscore_segment():
    assert Species("AB40_O12_ISF").resolve_compartment() == "ISF"


def test_species_without_underscore_is_its_own_compartment():
    assert Species("Plasma").resolve_compartment() == "Plasma"


def test_species_explicit_compartment_wins():
    assert Species("AB40_ISF", compartment="CSF").resolve_compartment() == "CSF"


# --- Reaction ---------------------------------------------------------------

def _rxn(**over):
    kw = dict(Reaction_name="r", Reactants=["A_x"], Products=["B_x"],
              Rate_type="MA", Rate_eqtn_prototype="k")
    kw.update(over)
    return Reaction(**kw)


def test_reaction_name_loses_spaces():
    assert _rxn(Reaction_name="a b c").Reaction_name == "abc"


def test_reaction_rejects_unknown_rate_type():
    with pytest.raises(ValueError, match="Invalid Rate_type"):
        _rxn(Rate_type="nonsense")


def test_rma_needs_two_constants():
    with pytest.raises(ValueError, match="RMA"):
        _rxn(Rate_type="RMA", Rate_eqtn_prototype="kf")
    _rxn(Rate_type="RMA", Rate_eqtn_prototype="[kf, kr]")


def test_bdf_needs_a_constant():
    with pytest.raises(ValueError, match="BDF"):
        _rxn(Rate_type="BDF", Rate_eqtn_prototype=[])


def test_two_constant_types_are_valid_types():
    assert RATE_TYPES_TWO_CONSTANTS <= VALID_RATE_TYPES


def test_to_dict_uses_bracket_serialisation():
    d = _rxn(Reactants=["A_x", "C_x"], Rate_type="RMA",
             Rate_eqtn_prototype=["kf", "kr"]).to_dict()
    assert d == {
        "Reaction_name": "r",
        "Reactants": "[A_x, C_x]",
        "Products": "[B_x]",
        "Rate_type": "RMA",
        "Rate_eqtn_prototype": "[kf, kr]",
    }


def test_to_dict_empty_side_is_zero_and_compartments_only_when_set():
    d = _rxn(Reactants=[]).to_dict()
    assert d["Reactants"] == "[0]"
    assert "compartment" not in d and "compartment_reverse" not in d
    d = _rxn(compartment="x", compartment_reverse="y").to_dict()
    assert d["compartment"] == "x" and d["compartment_reverse"] == "y"


# --- reaction_from_args / reaction_creation --------------------------------------

def test_reaction_from_args_accepts_bracket_strings():
    r = reaction_from_args("my rxn", "[A_x] + [B_x]", "[C_x]", " MA ", "k")
    assert r.Reaction_name == "myrxn"
    assert r.Reactants == ["A_x", "B_x"]
    assert r.Products == ["C_x"]
    assert r.Rate_type == "MA"


def test_reaction_from_args_blank_name_becomes_NA():
    assert reaction_from_args("  ", "A_x", "B_x", "MA", "k").Reaction_name == "NA"


@pytest.mark.parametrize("rate_type, rate", [("", "k"), (None, "k"), ("MA", ""), ("MA", None), ("MA", "  ")])
def test_reaction_from_args_requires_rate_type_and_equation(rate_type, rate):
    with pytest.raises(ValueError):
        reaction_from_args("r", "A_x", "B_x", rate_type, rate)


def test_reaction_creation_counts_constants():
    rxns = []
    counter, rxns = reaction_creation(rxns, 0, "a", "A_x", "B_x", "MA", "k")
    assert counter == 1
    counter, rxns = reaction_creation(rxns, counter, "b", "A_x", "B_x", "RMA", "[kf, kr]")
    assert counter == 3
    counter, rxns = reaction_creation(rxns, counter, "c", "A_x", "B_x", "BDF", "q")
    assert counter == 5
    assert [r["Reaction_name"] for r in rxns] == ["a", "b", "c"]


def test_reaction_creation_wraps_validation_errors_and_leaves_state_alone():
    rxns = []
    with pytest.raises(ValueError, match="add_reaction validation failed"):
        reaction_creation(rxns, 0, "a", "A_x", "B_x", "bogus", "k")
    assert rxns == []


# --- isotopes ---------------------------------------------------------------

@pytest.mark.parametrize("given, expected", [
    (["L", "13C"], ["", "L", "13C"]),
    (["", "", "L"], ["", "L"]),
    (["L", "L"], ["", "L"]),
    ([], [""]),
    (("L",), ["", "L"]),
    ([1, 2], ["", "1", "2"]),
])
def test_ensure_isotopes_format(given, expected):
    assert ensure_isotopes_format(given) == expected


def test_ensure_isotopes_format_is_idempotent():
    once = ensure_isotopes_format(["L", "13C"])
    assert ensure_isotopes_format(once) == once


# --- rate laws --------------------------------------------------------------

def test_volume_scaling_by_rate_type():
    assert MassActionLaw().multiplies_by_volume is True
    assert VolumeTransportLaw().multiplies_by_volume is False
    assert CustomLaw("custom_conc_per_time").multiplies_by_volume is True
    assert CustomLaw("custom_amt_per_time").multiplies_by_volume is False
    assert CustomLaw("custom").multiplies_by_volume is False


def test_custom_law_explicit_flag_overrides_default():
    assert CustomLaw("custom", species_names_are_conc_per_time=True).multiplies_by_volume is True
    assert CustomLaw("custom_conc_per_time", species_names_are_conc_per_time=False).multiplies_by_volume is False


def test_rate_law_info_covers_every_valid_rate_type():
    for rate_type in VALID_RATE_TYPES:
        assert get_rate_law_info(rate_type) != "Unknown rate type", rate_type
    assert get_rate_law_info("nope") == "Unknown rate type"


# --- module_base ------------------------------------------------------------

class _FakeModel:
    def __init__(self):
        self.reactions, self.rules = [], []

    def add_reaction(self, *args, **kwargs):
        self.reactions.append((args, kwargs))

    def add_rule(self, rule):
        self.rules.append(rule)


def test_module_base_requires_build():
    with pytest.raises(NotImplementedError):
        PyAntiGenModule(_FakeModel())


def test_module_builds_on_init_and_forwards_to_model():
    class Mod(PyAntiGenModule):
        def build(self):
            self.add_reaction("r", "A_x", "B_x", "MA", "k", compartment="x")
            self.add_rule("V_x := 1")

    model = _FakeModel()
    mod = Mod(model, flag=True)
    assert mod.config == {"flag": True}
    assert model.reactions == [(("r", "A_x", "B_x", "MA", "k"),
                                {"compartment": "x", "compartment_reverse": None})]
    assert model.rules == ["V_x := 1"]

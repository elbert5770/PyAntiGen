"""Reaction dict -> Antimony text (pyantigen.generate.RxnDict_to_antimony).

Pure string-in / string-out, plus one file round trip. No tellurium: the
output is checked as text, which is the contract the rest of the pipeline reads.

Run from the repository root: python -m pytest tests/generate
"""
import pytest

from pyantigen.generate.RxnDict_to_antimony import (
    add_reaction_names_to_string,
    collect_unique_compartments_from_reactions,
    convert_species_to_concentrations,
    extract_species_and_parameters_from_reactions,
    generate_antimony_from_txt,
    generate_single_reaction_from_dict,
    generate_species_declarations,
    read_reactions_from_txt,
    write_list_to_file,
)


def _d(rate_type="MA", rate="k", reactants="[A_x]", products="[B_y]", **extra):
    return {"Reaction_name": "r", "Reactants": reactants, "Products": products,
            "Rate_type": rate_type, "Rate_eqtn_prototype": rate, **extra}


# --- generate_single_reaction_from_dict: one test per rate type -----------------

def test_ma_multiplies_by_reactants_and_source_volume():
    assert generate_single_reaction_from_dict(_d()) == "A_x -> B_y; k * A_x * V_x\n"


def test_ma_bimolecular_multiplies_every_reactant():
    out = generate_single_reaction_from_dict(_d(reactants="[A_x, C_x]"))
    assert out == "A_x + C_x -> B_y; k * A_x * C_x * V_x\n"


def test_ma_zero_order_uses_product_compartment_volume():
    out = generate_single_reaction_from_dict(_d(reactants="[0]"))
    assert out == " -> B_y; k * V_y\n"


def test_rma_emits_forward_and_reverse_each_in_own_compartment():
    out = generate_single_reaction_from_dict(_d("RMA", "[kf, kr]"))
    assert out == ("A_x -> B_y; kf * A_x * V_x\n"
                   "B_y -> A_x; kr * B_y * V_y\n")


def test_rma_with_one_constant_is_an_error():
    with pytest.raises(ValueError, match="two rate constants"):
        generate_single_reaction_from_dict(_d("RMA", "kf"))


def test_udf_has_no_volume_factor():
    assert generate_single_reaction_from_dict(_d("UDF", "q")) == "A_x -> B_y; q * A_x\n"


def test_bdf_reuses_one_constant_both_ways_without_volume():
    out = generate_single_reaction_from_dict(_d("BDF", "q"))
    assert out == "A_x -> B_y; q * A_x\nB_y -> A_x; q * B_y\n"


def test_custom_conc_per_time_is_scaled_but_amt_and_custom_are_not():
    conc = generate_single_reaction_from_dict(_d("custom_conc_per_time", "k*A_x"))
    amt = generate_single_reaction_from_dict(_d("custom_amt_per_time", "k*A_x"))
    raw = generate_single_reaction_from_dict(_d("custom", "k*A_x"))
    assert conc == "A_x -> B_y; k*A_x * V_x\n"
    assert amt == "A_x -> B_y; k*A_x\n"
    assert raw == "A_x -> B_y; k*A_x\n"


def test_bracketed_custom_expression_is_unwrapped():
    out = generate_single_reaction_from_dict(_d("custom", "[k*A_x]"))
    assert out == "A_x -> B_y; k*A_x\n"


def test_explicit_compartment_overrides_name_suffix():
    out = generate_single_reaction_from_dict(_d(compartment="CSF"))
    assert out == "A_x -> B_y; k * A_x * V_CSF\n"


def test_unknown_rate_type_emits_nothing():
    assert generate_single_reaction_from_dict(_d("mystery")) == ""


# --- compartments -----------------------------------------------------------

def test_compartments_inferred_from_suffix():
    comps, errors, mapping = collect_unique_compartments_from_reactions([_d()])
    assert comps == {"x", "y"}
    assert errors == []
    assert mapping == {"A_x": "x", "B_y": "y"}


def test_explicit_compartments_win_per_side():
    comps, _, mapping = collect_unique_compartments_from_reactions(
        [_d(compartment="P", compartment_reverse="Q")])
    assert comps == {"P", "Q"}
    assert mapping == {"A_x": "P", "B_y": "Q"}


def test_malformed_compartment_is_reported_not_added():
    _, errors, _ = collect_unique_compartments_from_reactions(
        [_d(compartment="[bad]")])
    assert errors and "Malformed compartment" in errors[0]


# --- declarations -----------------------------------------------------------

def test_species_declarations_infer_or_use_map():
    assert generate_species_declarations(["A_x", "B_y"]) == (
        "substanceOnly species A_x in x\nsubstanceOnly species B_y in y")
    assert generate_species_declarations(["A_x"], {"A_x": "CSF"}) == (
        "substanceOnly species A_x in CSF")


# --- extraction ---------------------------------------------------------------

def test_extract_species_and_parameters():
    text = "A_x + 2 C_x -> B_y; k * A_x * V_x\n"
    species, params, errors = extract_species_and_parameters_from_reactions(text)
    assert species == ["A_x", "B_y", "C_x"]
    assert params == ["V_x", "k"]
    assert errors == []


def test_extract_skips_declarations_and_blank_lines():
    text = ("compartment x := V_x\n\nsubstanceOnly species A_x in x\n"
            "A_x -> ; k * A_x\n")
    species, params, _ = extract_species_and_parameters_from_reactions(text)
    assert species == ["A_x"]
    assert params == ["k"]


def test_extract_flags_malformed_species():
    _, _, errors = extract_species_and_parameters_from_reactions("[A_x] -> B_y; k\n")
    assert errors and "Malformed species" in errors[0]


# --- concentrations -----------------------------------------------------------

def test_species_in_rate_become_concentrations():
    out = convert_species_to_concentrations("A_x -> B_y; k * A_x * V_x\n", ["A_x", "B_y"])
    assert out.splitlines()[0] == "A_x -> B_y; k * (A_x/V_x) * V_x"


def test_concentration_conversion_respects_word_boundaries():
    out = convert_species_to_concentrations("A_x -> ; k * A_xy\n", ["A_x"])
    assert "(A_x/V_x)" not in out


def test_concentration_conversion_prefers_explicit_map():
    out = convert_species_to_concentrations("A_x -> ; k * A_x\n", ["A_x"], {"A_x": "CSF"})
    assert "(A_x/V_CSF)" in out


def test_concentration_conversion_leaves_left_side_alone():
    out = convert_species_to_concentrations("A_x -> B_y; k * A_x\n", ["A_x", "B_y"])
    assert out.startswith("A_x -> B_y;")


# --- naming -----------------------------------------------------------------

def test_names_get_fwd_rev_suffixes_for_two_line_reactions():
    reactions = [_d("RMA", "[kf, kr]"), dict(_d(), Reaction_name="s")]
    body = "".join(generate_single_reaction_from_dict(r) for r in reactions)
    named = add_reaction_names_to_string(body, reactions).splitlines()
    assert [ln.split(" : ")[0] for ln in named] == ["r_fwd", "r_rev", "s"]


# --- file round trip ----------------------------------------------------------

def test_read_reactions_skips_blank_and_unparsable_lines(tmp_path, capsys):
    f = tmp_path / "rd.txt"
    f.write_text(str(_d()) + "\n\n{not a dict\n" + str(_d("UDF", "q")) + "\n")
    reactions = read_reactions_from_txt(str(f))
    assert [r["Rate_type"] for r in reactions] == ["MA", "UDF"]
    assert "Error parsing line 3" in capsys.readouterr().out


def test_write_list_to_file(tmp_path):
    f = tmp_path / "items.txt"
    write_list_to_file(["a", "b"], str(f))
    assert f.read_text() == "a\nb\n"


def test_generate_antimony_from_txt_full_script(tmp_path):
    f = tmp_path / "rd.txt"
    f.write_text(str(_d("RMA", "[kf, kr]")) + "\n")
    script, species, params, comps, errors = generate_antimony_from_txt(str(f), "m")
    assert species == ["A_x", "B_y"]
    assert params == ["V_x", "V_y", "kf", "kr"]
    assert comps == {"x", "y"}
    assert errors == []
    assert script == (
        "compartment x := V_x\n"
        "compartment y := V_y\n"
        "\n"
        "substanceOnly species A_x in x\n"
        "substanceOnly species B_y in y\n"
        "\n"
        "r_fwd : A_x -> B_y; kf * (A_x/V_x) * V_x\n"
        "r_rev : B_y -> A_x; kr * (B_y/V_y) * V_y\n"
    )


def test_compartments_are_assignment_rules_not_initial_assignments(tmp_path):
    """':=' keeps the compartment tracking V_<name>; '=' would freeze a copy at t=0."""
    f = tmp_path / "rd.txt"
    f.write_text(str(_d()) + "\n")
    script = generate_antimony_from_txt(str(f), "m")[0]
    compartment_lines = [ln for ln in script.splitlines() if ln.startswith("compartment ")]
    assert compartment_lines and all(":=" in ln for ln in compartment_lines)

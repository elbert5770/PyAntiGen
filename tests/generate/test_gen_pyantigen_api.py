"""The user-facing PyAntiGen builder and generate_model, end to end on disk.

Run from the repository root: python -m pytest tests/generate
"""
import ast
import os

import pytest

from pyantigen.generate.model_generation import generate_model
from pyantigen.generate.pyantigen import PyAntiGen


def _build(m):
    m.add_reaction("prod", "[0]", "[A_x]", "MA", "kprod", compartment_reverse="x")
    m.add_reaction("bind", "[A_x]", "[B_x]", "RMA", "[kf, kr]")
    m.add_rule("V_x := 1")


def test_defaults_and_isotopes_normalised():
    m = PyAntiGen("m")
    assert (m.name, m.isotopes, m.reactions, m.rules, m.counter) == ("m", [""], [], [], 0)
    assert PyAntiGen("m", isotopes=["L"]).isotopes == ["", "L"]


def test_add_reaction_tracks_counter_and_serialises():
    m = PyAntiGen("m")
    _build(m)
    assert m.counter == 3  # MA: 1, RMA: 2
    assert [r["Reaction_name"] for r in m.reactions] == ["prod", "bind"]
    assert m.reactions[0]["compartment_reverse"] == "x"
    assert m.rules == ["V_x := 1"]


def test_add_reaction_rejects_bad_input_without_side_effects():
    m = PyAntiGen("m")
    with pytest.raises(ValueError):
        m.add_reaction("bad", "A_x", "B_x", "RMA", "kf")
    assert m.reactions == [] and m.counter == 0


def _script_in(tmp_path, sub="scripts"):
    d = tmp_path / sub
    d.mkdir(parents=True, exist_ok=True)
    return str(d / "build.py")


def test_generate_writes_project_tree_next_to_scripts_dir(tmp_path):
    m = PyAntiGen("m")
    _build(m)
    m.generate(_script_in(tmp_path))

    gen = tmp_path / "generated" / "m"
    assert (tmp_path / "antimony_models" / "m" / "m_reactions.txt").is_file()
    assert (gen / "m_rules.txt").read_text() == "V_x := 1\n"
    lines = (gen / "m_reaction_dict.txt").read_text().splitlines()
    assert [ast.literal_eval(ln)["Reaction_name"] for ln in lines] == ["prod", "bind"]
    assert (gen / "conversion_errors_m.log").read_text() == ""


def test_generate_model_name_overrides_builder_name(tmp_path):
    m = PyAntiGen("m")
    _build(m)
    m.generate(_script_in(tmp_path), model_name="other")
    assert (tmp_path / "generated" / "other" / "other_reaction_dict.txt").is_file()
    assert not (tmp_path / "generated" / "m").exists()


def test_generate_from_nested_script_goes_two_levels_up(tmp_path):
    """scripts/Example/run.py -> project root is two levels above the script."""
    m = PyAntiGen("m")
    _build(m)
    script = _script_in(tmp_path, "scripts/Example")
    m.generate(script)
    assert (tmp_path / "generated" / "m" / "m_reaction_dict.txt").is_file()


def test_generate_is_repeatable_and_overwrites(tmp_path):
    m = PyAntiGen("m")
    _build(m)
    script = _script_in(tmp_path)
    m.generate(script)
    first = (tmp_path / "generated" / "m" / "m_reaction_dict.txt").read_text()
    m.generate(script)
    assert (tmp_path / "generated" / "m" / "m_reaction_dict.txt").read_text() == first


def test_generate_model_function_passes_normalised_isotopes(tmp_path):
    seen = {}

    def build(isotopes):
        seen["isotopes"] = isotopes
        return [], []

    generate_model(build, ["L", "L"], _script_in(tmp_path), "empty")
    assert seen["isotopes"] == ["", "L"]
    assert (tmp_path / "generated" / "empty" / "empty_reaction_dict.txt").read_text() == ""

"""File handling in pyantigen.generate.antimony_utils, in a temp project tree.

Run from the repository root: python -m pytest tests/generate
"""
import csv
import json
import os

from pyantigen.generate import antimony_utils as au

RXN = {"Reaction_name": "r", "Reactants": "[A_x]", "Products": "[B_x]",
       "Rate_type": "MA", "Rate_eqtn_prototype": "k"}


def _write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


# --- settings ---------------------------------------------------------------

def test_settings_default_when_file_missing(tmp_path):
    assert au.load_project_settings(tmp_path) == {"archive_with_timestamp": False}


def test_settings_read_known_key_and_ignore_unknown(tmp_path):
    _write(str(tmp_path / au.SETTINGS_FILENAME),
           json.dumps({"archive_with_timestamp": True, "other": 1}))
    assert au.load_project_settings(tmp_path) == {"archive_with_timestamp": True}


def test_settings_survive_invalid_json_and_non_dict(tmp_path):
    p = str(tmp_path / au.SETTINGS_FILENAME)
    _write(p, "{nope")
    assert au.load_project_settings(tmp_path)["archive_with_timestamp"] is False
    _write(p, "[1, 2]")
    assert au.load_project_settings(tmp_path)["archive_with_timestamp"] is False


# --- CSV -> Antimony --------------------------------------------------------

def test_parameters_csv_lines(tmp_path):
    p = str(tmp_path / "p.csv")
    _write(p, "Parameter,Value,Units,Comment\n"
              "k,0.5,1/h,rate\n"
              "empty,,,\n"
              "dyn,var,,\n"
              ",9,,\n")
    assert au._csv_to_antimony_parameters(p).splitlines() == [
        "k = 0.5 // [1/h] # rate",
        "empty = 0",
        "var dyn",
    ]


def test_initial_conditions_csv_lines(tmp_path):
    p = str(tmp_path / "ic.csv")
    _write(p, "Species,InitialCondition,Units,Comment\nA_x,2,nM,seed\nB_x,,,\n")
    first, second = au._csv_to_antimony_initial_conditions(p).splitlines()
    # The space before "#" is currently stripped ("A_x = 2# nM seed"); it is
    # still an Antimony comment, so only the value and the comment are pinned.
    assert first.startswith("A_x = 2") and first.endswith("# nM seed")
    assert second == "B_x = 0"


def test_missing_csv_gives_empty_string(tmp_path):
    assert au._csv_to_antimony_parameters(str(tmp_path / "none.csv")) == ""
    assert au._csv_to_antimony_initial_conditions(str(tmp_path / "none.csv")) == ""


def test_csv_with_bom_is_read(tmp_path):
    p = tmp_path / "p.csv"
    p.write_bytes("﻿Parameter,Value,Units,Comment\nk,1,,\n".encode("utf-8"))
    assert au._csv_to_antimony_parameters(str(p)) == "k = 1"


# --- load_antimony_files ----------------------------------------------------

def test_load_concatenates_in_documented_order(tmp_path):
    root = str(tmp_path)
    base = os.path.join(root, "antimony_models", "m")
    _write(os.path.join(base, "m_reactions.txt"), "R\n\n")
    _write(os.path.join(base, "m_parameters.csv"), "Parameter,Value,Units,Comment\nk,1,,\n")
    _write(os.path.join(base, "m_InitialConditions.csv"),
           "Species,InitialCondition,Units,Comment\nA_x,3,,\n")
    _write(os.path.join(base, "m_manual.txt"), "MANUAL")
    _write(os.path.join(base, "m_rules.txt"), "RULES")
    _write(os.path.join(base, "m_events.txt"), "EVENTS")
    assert au.load_antimony_files("m", root).splitlines() == [
        "R", "k = 1", "A_x = 3", "MANUAL", "RULES", "EVENTS"]


def test_load_copies_missing_files_from_generated(tmp_path):
    root = str(tmp_path)
    _write(os.path.join(root, "generated", "m", "m_reactions.txt"), "FROM_GENERATED")
    assert au.load_antimony_files("m", root) == "FROM_GENERATED"
    assert os.path.isfile(os.path.join(root, "antimony_models", "m", "m_reactions.txt"))


def test_load_never_overwrites_an_edited_antimony_models_file(tmp_path):
    root = str(tmp_path)
    _write(os.path.join(root, "antimony_models", "m", "m_reactions.txt"), "HAND_EDITED")
    _write(os.path.join(root, "generated", "m", "m_reactions.txt"), "GENERATED")
    assert au.load_antimony_files("m", root) == "HAND_EDITED"


def test_load_with_nothing_present_is_empty(tmp_path):
    assert au.load_antimony_files("m", str(tmp_path)) == ""


# --- archive ----------------------------------------------------------------

def test_archive_copies_with_suffix_and_writes_sbml(tmp_path):
    root = str(tmp_path)
    _write(os.path.join(root, "antimony_models", "m", "m_reactions.txt"), "R")
    _write(os.path.join(root, "antimony_models", "m", "m_parameters.csv"), "P")
    out = au.archive_antimony_snapshot("m", root, sbml_content="<sbml/>")
    assert out == os.path.join(root, "SBML_models", "m")
    assert sorted(os.listdir(out)) == [
        "m.xml", "m_parameters_archive.csv", "m_reactions_archive.txt"]
    assert open(os.path.join(out, "m.xml")).read() == "<sbml/>"


def test_archive_without_sbml_writes_no_xml(tmp_path):
    root = str(tmp_path)
    _write(os.path.join(root, "antimony_models", "m", "m_reactions.txt"), "R")
    out = au.archive_antimony_snapshot("m", root)
    assert "m.xml" not in os.listdir(out)


def test_archive_with_timestamp_setting_uses_subfolder(tmp_path):
    root = str(tmp_path)
    _write(os.path.join(root, au.SETTINGS_FILENAME),
           json.dumps({"archive_with_timestamp": True}))
    _write(os.path.join(root, "antimony_models", "m", "m_reactions.txt"), "R")
    out = au.archive_antimony_snapshot("m", root)
    assert os.path.dirname(out) == os.path.join(root, "SBML_models", "m")
    assert os.path.isfile(os.path.join(out, "m_reactions_archive.txt"))


# --- convert_to_antimony ----------------------------------------------------

def test_convert_to_antimony_writes_the_generated_tree(tmp_path):
    rd = str(tmp_path / "rd.txt")
    _write(rd, str(RXN) + "\n")
    rules = str(tmp_path / "rules.txt")
    _write(rules, "V_x := 1\n")
    out = str(tmp_path / "proj")
    au.convert_to_antimony(rd, "m", rules, output_dir=out)

    gen = os.path.join(out, "generated", "m")
    am = os.path.join(out, "antimony_models", "m")
    assert open(os.path.join(am, "m_reactions.txt")).read() == \
        open(os.path.join(gen, "m_reactions.txt")).read()
    assert open(os.path.join(am, "m_rules.txt")).read() == "V_x := 1\n"
    assert open(os.path.join(gen, "m_unique_species.txt")).read() == "A_x\nB_x\n"
    assert open(os.path.join(gen, "m_unique_compartments.txt")).read() == "x\n"
    assert open(os.path.join(gen, "m_events.txt")).read() == ""
    assert open(os.path.join(gen, "conversion_errors_m.log")).read() == ""

    with open(os.path.join(gen, "m_parameters.csv"), newline="") as f:
        assert [r["Parameter"] for r in csv.DictReader(f)] == ["V_x", "k"]
    with open(os.path.join(gen, "m_InitialConditions.csv"), newline="") as f:
        rows = list(csv.DictReader(f))
    assert [(r["Species"], r["InitialCondition"]) for r in rows] == [("A_x", "0"), ("B_x", "0")]


def test_convert_to_antimony_without_rules_file(tmp_path):
    rd = str(tmp_path / "rd.txt")
    _write(rd, str(RXN) + "\n")
    out = str(tmp_path / "proj")
    au.convert_to_antimony(rd, "m", None, output_dir=out)
    assert not os.path.exists(os.path.join(out, "antimony_models", "m", "m_rules.txt"))

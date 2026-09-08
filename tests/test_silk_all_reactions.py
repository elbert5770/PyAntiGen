"""
Tests that run Elbert_2022_SILK scripts and validate the generated *_reaction_dict files.

Target scripts (and their reaction output files, keyed by model name rather
than script name -- the two need not agree):
- scripts/Elbert_2022_1a.py  -> generated/Elbert_2022_1a/Elbert_2022_1a_reaction_dict.txt
- scripts/Bloomingdale_2021_1a.py -> generated/Bloomingdale_2021_1a/Bloomingdale_2021_1a_reaction_dict.txt

Requires the Elbert_2022_SILK project to be available (sibling directory or SILK_PROJECT_PATH).
"""
import ast
import os
import subprocess
import sys

import pytest


REQUIRED_KEYS = {"Reaction_name", "Reactants", "Products", "Rate_type", "Rate_eqtn_prototype"}
VALID_RATE_TYPES = {"MA", "RMA", "UDF", "BDF", "custom", "custom_conc_per_time", "custom_amt_per_time"}


def parse_reaction_line(line):
    """Parse a single line from *_reaction_dict.txt into a dict. Uses ast.literal_eval for safety."""
    line = line.strip()
    if not line:
        return None
    try:
        obj = ast.literal_eval(line)
        return obj if isinstance(obj, dict) else None
    except (ValueError, SyntaxError):
        return None


def load_reactions_file(path):
    """Load and parse a generated _reaction_dict.txt file. Returns list of reaction dicts and errors."""
    if not os.path.isfile(path):
        return [], [f"File not found: {path}"]
    reactions = []
    errors = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            d = parse_reaction_line(line)
            if d is None:
                errors.append(f"Line {i}: could not parse as reaction dict")
                continue
            missing = REQUIRED_KEYS - set(d.keys())
            if missing:
                errors.append(f"Line {i} ({d.get('Reaction_name', '?')}): missing keys {missing}")
                continue
            if d.get("Rate_type") and d["Rate_type"] not in VALID_RATE_TYPES:
                errors.append(f"Line {i} ({d.get('Reaction_name')}): invalid Rate_type {d.get('Rate_type')}")
            reactions.append(d)
    return reactions, errors


def run_script(silk_root, script_basename, pyantigen_root):
    """Run a script from the SILK project so it generates *_reaction_dict.txt."""
    script_path = os.path.join(silk_root, "scripts", script_basename)
    if not os.path.isfile(script_path):
        raise FileNotFoundError(f"Script not found: {script_path}")
    env = os.environ.copy()
    parts = [pyantigen_root, silk_root]
    existing_pp = env.get("PYTHONPATH", "").strip()
    if existing_pp:
        parts.extend(p.strip() for p in existing_pp.split(os.pathsep) if p.strip())
    env["PYTHONPATH"] = os.pathsep.join(parts)
    result = subprocess.run(
        [sys.executable, script_path],
        cwd=silk_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return result


# model_name is the name passed to PyAntiGen() inside the script, which is what
# sets the generated/<model_name>/ output folder. It is stated explicitly rather
# than derived from the script filename, because the two need not agree: a
# script whose model name differs would otherwise be checked against a path
# nothing writes, and fail as "File not found" even though it had succeeded.
@pytest.mark.parametrize("script_name,model_name,expected_min_reactions,expected_reaction_names", [
    (
        "Elbert_2022_1a.py",
        "Elbert_2022_1a",
        100,
        ["APPSynthesis_BrainISF", "FlowWithinTissue_AB38_CV_SAS", "BidirectionalFlowWithinTissue_AB38_SAS_BrainISF", "AB40Exchange_BrainISF"],
    ),
    (
        "Bloomingdale_2021_1a.py",
        "Bloomingdale_2021_1a",
        25,
        ["FlowWithinTissue_Antibody_BrainVascular_BrainISF", "BidirectionalFlowWithinTissue_Antibody_CSF_BrainISF", "BindingToFcRn_Antibody_BBB"],
    ),
])
def test_silk_script_generates_all_reactions(
    silk_project_path,
    pyantigen_root,
    script_name,
    model_name,
    expected_min_reactions,
    expected_reaction_names,
):
    """Run the SILK script and assert the generated _reaction_dict file is valid and has expected content."""
    reactions_file = os.path.join(
        silk_project_path, "generated", model_name, f"{model_name}_reaction_dict.txt"
    )

    # Run the script to (re)generate the file
    result = run_script(silk_project_path, script_name, pyantigen_root)
    assert result.returncode == 0, (
        f"Script failed: stdout={result.stdout!r} stderr={result.stderr!r}"
    )

    # Load and validate the reactions file
    reactions, errors = load_reactions_file(reactions_file)
    assert not errors, f"Parse errors in {reactions_file}: {errors}"
    assert len(reactions) >= expected_min_reactions, (
        f"Expected at least {expected_min_reactions} reactions, got {len(reactions)}"
    )

    names = {r["Reaction_name"] for r in reactions}
    for expected_name in expected_reaction_names:
        assert expected_name in names, f"Expected reaction name {expected_name!r} not found in {reactions_file}"


def test_silk_all_reactions_file_structure(silk_project_path):
    """
    If the SILK project exists and has pre-generated _reaction_dict files,
    validate their structure without re-running the scripts.
    """
    # Model names, not script names -- see the note on the parametrize above.
    #
    # Note the skip below fires inside the loop, so one missing file marks the
    # whole test skipped and discards the cases already checked. Keep this list
    # in step with the parametrize above, or a stale entry goes quiet instead of
    # red.
    cases = [
        ("Elbert_2022_1a", 100),
        ("Bloomingdale_2021_1a", 25),
    ]
    for name, min_reactions in cases:
        filename = f"{name}_reaction_dict.txt"
        path = os.path.join(silk_project_path, "generated", name, filename)
        if not os.path.isfile(path):
            pytest.skip(f"Pre-generated file not found: {path}")
        reactions, errors = load_reactions_file(path)
        assert not errors, f"Parse errors in {path}: {errors}"
        assert len(reactions) >= min_reactions, (
            f"{name}: expected at least {min_reactions} reactions, got {len(reactions)}"
        )
        for r in reactions:
            assert REQUIRED_KEYS.issubset(set(r.keys())), f"Missing keys in reaction {r.get('Reaction_name')}"
            assert r["Rate_type"] in VALID_RATE_TYPES, f"Invalid Rate_type in reaction {r.get('Reaction_name')}"

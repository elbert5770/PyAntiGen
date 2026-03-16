import os
import sys

from framework.antimony_utils import convert_to_antimony
from framework.isotopomer_tools import ensure_isotopes_format

def generate_model(build_reactions_func, Isotopes, calling_file_path):
    """
    Generates an Antimony model using the provided reaction building function.

    Args:
        build_reactions_func (callable): Function that takes Isotopes list and returns (all_reactions, rules).
        Isotopes (list): List of isotopes to include.
        calling_file_path (str): The __file__ path of the calling script, used to determine output filenames.
    """
    Isotopes = ensure_isotopes_format(Isotopes)
    all_reactions, rules = build_reactions_func(Isotopes)
    
    # Model name = project name: strip _generate from script name if present (e.g. MyNewModel_generate.py -> MyNewModel)
    script_basename = os.path.basename(calling_file_path).replace('.py', '')
    model_name = script_basename[:-9] if script_basename.endswith('_generate') else script_basename

    # Resolve to absolute path so output_dir does not depend on cwd (fixes VSCode "Run Python File" where cwd can be script dir)
    script_dir = os.path.abspath(os.path.dirname(os.path.normpath(calling_file_path)))
    # Project root: parent of "scripts", or two levels up when script is in scripts/Example/
    if os.path.basename(script_dir) == "scripts":
        output_dir = os.path.abspath(os.path.join(script_dir, ".."))
    else:
        output_dir = os.path.abspath(os.path.join(script_dir, "..", ".."))
    generated_dir = os.path.join(output_dir, "generated", model_name)
    os.makedirs(generated_dir, exist_ok=True)

    # Output rules and reaction_dict using model_name only (one write each)
    rules_path = os.path.join(generated_dir, f'{model_name}_rules.txt')
    with open(rules_path, "w", encoding="utf-8") as f:
        for rule in rules:
            f.write(str(rule) + "\n")
    print("Wrote to", rules_path)

    model_path = os.path.join(generated_dir, f'{model_name}_reaction_dict.txt')
    with open(model_path, "w", encoding="utf-8") as f:
        for reaction in all_reactions:
            f.write(str(reaction) + "\n")
    print("Wrote to", model_path)

    # Convert to Antimony format (use absolute paths so outputs go to project dir regardless of cwd)
    model_path_abs = os.path.abspath(model_path)
    rules_path_abs = os.path.abspath(rules_path)
    convert_to_antimony(model_path_abs, model_name, rules_path_abs, output_dir=output_dir)

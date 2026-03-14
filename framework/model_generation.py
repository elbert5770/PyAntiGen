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
    
    # Determine directories based on the calling file's location (assuming calling file is in scripts/)
    script_dir = os.path.dirname(calling_file_path) 
    
    # Output rules
    filename = os.path.basename(calling_file_path).replace('.py', '_rules.txt')
    rules_path = os.path.join(script_dir, '..', 'generated', filename)
    os.makedirs(os.path.dirname(rules_path), exist_ok=True)
    with open(rules_path, "w", encoding="utf-8") as f:
        for rule in rules:
            f.write(str(rule) + "\n")
    print("Wrote to", rules_path)
    
    # Output reactions
    filename = os.path.basename(calling_file_path).replace('.py', '_all_reactions.txt')
    model_path = os.path.join(script_dir, '..', 'generated', filename)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, "w", encoding="utf-8") as f:
        for reaction in all_reactions:
            f.write(str(reaction) + "\n")

    print("Wrote to", model_path)

    # Convert to Antimony format (use absolute paths so outputs go to project dir regardless of cwd)
    name = os.path.basename(calling_file_path).replace('.py', '')
    output_dir = os.path.abspath(os.path.join(script_dir, '..'))
    model_path_abs = os.path.abspath(model_path)
    rules_path_abs = os.path.abspath(rules_path)
    convert_to_antimony(model_path_abs, name, rules_path_abs, output_dir=output_dir)

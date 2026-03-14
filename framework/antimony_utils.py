import csv
import os
import shutil

from framework.RxnDict_to_antimony import generate_antimony_from_txt, write_list_to_file

# Filenames (without path) that load_antimony_files expects, in concatenation order
ANTIMONY_FILE_NAMES = [
    "_reactions.txt",
    "_parameters.csv",
    "_InitialConditions.csv",
    "_manual.txt",
    "_rules.txt",
    "_events.txt",
]


def _ensure_file_in_antimony_models(model_name, base_name, antimony_models_dir, generated_dir):
    """
    If the file does not exist in antimony_models, copy it from generated.
    Never overwrite an existing file in antimony_models.
    """
    dest = os.path.join(antimony_models_dir, f"{model_name}{base_name}")
    if os.path.isfile(dest):
        return
    src = os.path.join(generated_dir, f"{model_name}{base_name}")
    if not os.path.isfile(src):
        return
    os.makedirs(antimony_models_dir, exist_ok=True)
    shutil.copy2(src, dest)


def _csv_to_antimony_parameters(csv_path):
    """Read parameters CSV and return Antimony lines: Parameter = Value # Units Comment."""
    lines = []
    try:
        with open(csv_path, "r", newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                param = row.get("Parameter", "").strip()
                if not param:
                    continue
                val = row.get("Value", "").strip()
                units = row.get("Units", "").strip()
                comment = row.get("Comment", "").strip()
                try:
                    float(val) if val else 0
                    line = f"{param} = {val if val else '0'}"
                except ValueError:
                    line = f"{param} := {val}" if val else f"{param} := 0"
                if units or comment:
                    line += f" # {units} {comment}".strip()
                lines.append(line)
    except FileNotFoundError:
        pass
    return "\n".join(lines)


def _csv_to_antimony_initial_conditions(csv_path):
    """Read InitialConditions CSV and return Antimony lines: Species = InitialCondition # Units Comment."""
    lines = []
    try:
        with open(csv_path, "r", newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                species = row.get("Species", "").strip()
                if not species:
                    continue
                ic = row.get("InitialCondition", "").strip() or "0"
                units = row.get("Units", "").strip()
                comment = row.get("Comment", "").strip()
                line = f"{species} = {ic}"
                if units or comment:
                    line += f" # {units} {comment}".strip()
                lines.append(line)
    except FileNotFoundError:
        pass
    return "\n".join(lines)


def load_antimony_files(model_name, project_root):
    """
    Load and concatenate Antimony model files from antimony_models.

    For each of the standard files ({model_name}_reactions.txt, _parameters.csv,
    _InitialConditions.csv, _manual.txt, _rules.txt, _events.txt), look first in
    antimony_models. If a file is missing there, copy it from generated (without
    overwriting if it already exists in antimony_models). Then read from
    antimony_models and concatenate in order to form the full model text.

    Args:
        model_name (str): Base name of the model (may differ from the run script name).
        project_root (str): Project directory containing antimony_models and generated.

    Returns:
        str: Concatenated Antimony model text for Tellurium.
    """
    project_root = os.path.abspath(project_root)
    antimony_models_dir = os.path.join(project_root, "antimony_models")
    generated_dir = os.path.join(project_root, "generated")

    for base_name in ANTIMONY_FILE_NAMES:
        _ensure_file_in_antimony_models(model_name, base_name, antimony_models_dir, generated_dir)

    parts = []
    for base_name in ANTIMONY_FILE_NAMES:
        path = os.path.join(antimony_models_dir, f"{model_name}{base_name}")
        if not os.path.isfile(path):
            continue
        if base_name == "_parameters.csv":
            parts.append(_csv_to_antimony_parameters(path))
        elif base_name == "_InitialConditions.csv":
            parts.append(_csv_to_antimony_initial_conditions(path))
        else:
            with open(path, "r", encoding="utf-8") as f:
                parts.append(f.read().rstrip())

    return "\n".join(p for p in parts if p.strip())


def convert_to_antimony(model_path, name, rules_path, output_dir=None):
    """
    Convert the generated reaction file to Antimony format.

    Args:
        model_path (str): Path to the generated reaction text file
        name (str): Name to use for output files
        rules_path (str): Path to the rules file (optional)
        output_dir (str): Base directory for outputs (defaults to parent of this file)
    """
    if output_dir is None:
        output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    else:
        output_dir = os.path.abspath(output_dir)

    # Resolve model path so the reaction file is found regardless of cwd
    model_path = os.path.abspath(model_path)

    # Generate Antimony script from text file
    complete_script, species, parameters, unique_compartments, errors = generate_antimony_from_txt(model_path, name)
    
    generated_dir = os.path.join(output_dir, "generated")
    os.makedirs(generated_dir, exist_ok=True)
    antimony_models_dir = os.path.join(output_dir, "antimony_models")
    os.makedirs(antimony_models_dir, exist_ok=True)

    # Write _reactions.txt to both antimony_models and generated
    reactions_basename = f"{name}_reactions.txt"
    for folder in (antimony_models_dir, generated_dir):
        path = os.path.join(folder, reactions_basename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(complete_script)
    
    # Write unique compartments/species/parameters to generated (model_name before unique_*)
    compartments_filename = os.path.join(generated_dir, f'{name}_unique_compartments.txt')
    with open(compartments_filename, "w", encoding="utf-8") as f:
        for compartment in sorted(unique_compartments):
            f.write(f"{compartment}\n")
    
    species_filename = os.path.join(generated_dir, f'{name}_unique_species.txt')
    parameters_list_filename = os.path.join(generated_dir, f'{name}_unique_parameters.txt')
    write_list_to_file(species, species_filename)
    write_list_to_file(parameters, parameters_list_filename)
    
    # Write generated-only CSV and empty files (only in generated; user may copy to antimony_models)
    init_cond_path = os.path.join(generated_dir, f"{name}_InitialConditions.csv")
    with open(init_cond_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Species", "InitialCondition", "Units", "Comment"])
        writer.writeheader()
        for s in sorted(species):
            writer.writerow({"Species": s, "InitialCondition": "0", "Units": "", "Comment": ""})

    params_csv_path = os.path.join(generated_dir, f"{name}_parameters.csv")
    with open(params_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Parameter", "Value", "Units", "Comment"])
        writer.writeheader()
        for p in sorted(parameters):
            writer.writerow({"Parameter": p, "Value": "", "Units": "", "Comment": ""})

    for empty_basename in (f"{name}_events.txt", f"{name}_manual.txt"):
        empty_path = os.path.join(generated_dir, empty_basename)
        with open(empty_path, "w", encoding="utf-8") as f:
            pass
    
    # Write errors to file
    errors_filename = os.path.join(generated_dir, f'conversion_errors_{name}.log')
    with open(errors_filename, "w", encoding="utf-8") as f:
        for error in errors:
            f.write(f"{error}\n")

    # Read rules file and write to both antimony_models and generated
    if rules_path is not None and os.path.isfile(rules_path):
        with open(rules_path, "r", encoding="utf-8") as f:
            rules_content = f.read()
        rules_basename = f"{name}_rules.txt"
        for folder in (antimony_models_dir, generated_dir):
            rules_output_path = os.path.join(folder, rules_basename)
            with open(rules_output_path, "w", encoding="utf-8") as f:
                f.write(rules_content)
        print(f"  - Rules file written to antimony_models and generated")

    
    # Print summary
    print(f"\nAntimony conversion complete:")
    print(f"  - Found {len(unique_compartments)} unique compartments")
    print(f"  - Found {len(species)} unique species and {len(parameters)} unique parameters")
    print(f"  - Antimony reactions written to antimony_models and generated")
    if errors:
        print(f"  - Found {len(errors)} errors (written to '{errors_filename}')")
    else:
        print(f"  - No errors found during conversion")

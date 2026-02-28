import os
from framework.RxnDict_to_antimony import generate_antimony_from_txt, write_list_to_file

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
        output_dir = os.path.join(os.path.dirname(__file__), '..')

    # Generate Antimony script from text file
    complete_script, species, parameters, unique_compartments, errors = generate_antimony_from_txt(model_path, name)
    
    # Write complete script to file
    antimony_filename = os.path.join(output_dir, 'antimony_models', f'Antimony_{name}_all_reactions.txt')
    os.makedirs(os.path.dirname(antimony_filename), exist_ok=True)
    with open(antimony_filename, "w", encoding="utf-8") as f:
        f.write(complete_script)
    
    # Write unique compartments to file
    generated_dir = os.path.join(output_dir, 'generated')
    os.makedirs(generated_dir, exist_ok=True)
    
    compartments_filename = os.path.join(generated_dir, f'unique_compartments_{name}.txt')
    with open(compartments_filename, "w", encoding="utf-8") as f:
        for compartment in sorted(unique_compartments):
            f.write(f"{compartment}\n")
    
    # Write species and parameters to files
    species_filename = os.path.join(generated_dir, f'unique_species_{name}.txt')
    parameters_filename = os.path.join(generated_dir, f'unique_parameters_{name}.txt')
    write_list_to_file(species, species_filename)
    write_list_to_file(parameters, parameters_filename)
    
    # Write errors to file
    errors_filename = os.path.join(generated_dir, f'conversion_errors_{name}.log')
    with open(errors_filename, "w", encoding="utf-8") as f:
        for error in errors:
            f.write(f"{error}\n")

     # Read rules file and write to antimony_models folder
    if rules_path is not None:
        with open(rules_path, "r", encoding="utf-8") as f:
            rules_content = f.read()
            
        rules_output_path = os.path.join(output_dir, 'antimony_models', f'Antimony_{name}_rules.txt')
        with open(rules_output_path, "w", encoding="utf-8") as f:
            f.write(rules_content)
        print(f"  - Rules file written to '{rules_output_path}'")

    
    # Print summary
    print(f"\nAntimony conversion complete:")
    print(f"  - Found {len(unique_compartments)} unique compartments")
    print(f"  - Found {len(species)} unique species and {len(parameters)} unique parameters")
    print(f"  - Antimony script written to '{antimony_filename}'")
    if errors:
        print(f"  - Found {len(errors)} errors (written to '{errors_filename}')")
    else:
        print(f"  - No errors found during conversion")

import os
from framework.RxnDict_to_antimony import generate_antimony_from_txt, write_list_to_file

def convert_to_antimony(model_path, name, rules_path):
    """
    Convert the generated reaction file to Antimony format.
    
    Args:
        model_path (str): Path to the generated reaction text file
        name (str): Name to use for output files
        rules_path (str): Path to the rules file (optional)
    """
    # Generate Antimony script from text file
    complete_script, species, parameters, unique_compartments, errors = generate_antimony_from_txt(model_path, name)
    
    # Write complete script to file
    current_dir = os.path.dirname(__file__)
    antimony_filename = os.path.join(current_dir, '..', 'antimony_models', f'Antimony_{name}_all_reactions.txt')
    with open(antimony_filename, "w", encoding="utf-8") as f:
        f.write(complete_script)
    
    # Write unique compartments to file
    compartments_filename = os.path.join(current_dir, '..', 'generated', f'unique_compartments_{name}.txt')
    with open(compartments_filename, "w", encoding="utf-8") as f:
        for compartment in sorted(unique_compartments):
            f.write(f"{compartment}\n")
    
    # Write species and parameters to files
    species_filename = os.path.join(current_dir, '..', 'generated', f'unique_species_{name}.txt')
    parameters_filename = os.path.join(current_dir, '..', 'generated', f'unique_parameters_{name}.txt')
    write_list_to_file(species, species_filename)
    write_list_to_file(parameters, parameters_filename)
    
    # Write errors to file
    errors_filename = os.path.join(current_dir, '..', 'generated', f'conversion_errors_{name}.log')
    with open(errors_filename, "w", encoding="utf-8") as f:
        for error in errors:
            f.write(f"{error}\n")

     # Read rules file and write to antimony_models folder
    if rules_path is not None:
        with open(rules_path, "r", encoding="utf-8") as f:
            rules_content = f.read()
        with open(os.path.join(current_dir, '..', 'antimony_models', f'Antimony_{name}_rules.txt'), "w", encoding="utf-8") as f:
            f.write(rules_content)
        print(f"  - Rules file written to '{os.path.join(current_dir, '..', 'antimony_models', f'Antimony_{name}_rules.txt')}'")

    
    # Print summary
    print(f"\nAntimony conversion complete:")
    print(f"  - Found {len(unique_compartments)} unique compartments")
    print(f"  - Found {len(species)} unique species and {len(parameters)} unique parameters")
    print(f"  - Antimony script written to '{antimony_filename}'")
    if errors:
        print(f"  - Found {len(errors)} errors (written to '{errors_filename}')")
    else:
        print(f"  - No errors found during conversion")

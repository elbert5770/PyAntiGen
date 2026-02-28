import json
import re
import sys

def extract_species_and_parameters_from_reactions(reaction_string):
    """
    Extract unique species and parameters from reaction string (excluding compartment declarations).
    
    Args:
        reaction_string (str): The reaction string to analyze
        
    Returns:
        tuple: (species_list, parameters_list, errors_list)
    """
    # Sets to store unique species and parameters
    species = set()
    parameters = set()
    errors = []
    
    # Split into lines and process each line
    lines = reaction_string.split('\n')
    for line_num, line in enumerate(lines, 1):
        line = line.strip()
        if not line:  # Skip empty lines
            continue
        
        # Skip compartment declarations
        if line.startswith('compartment '):
            continue
        
        # Skip species declarations (they start with "substanceOnly species")
        if line.startswith('substanceOnly species'):
            continue
                
        # Split at semicolon
        parts = line.split(';')
        if len(parts) != 2:
            continue
                
        # Extract species from left side
        left_side = parts[0].strip()
        # Split by + and -> to get individual species
        species_parts = re.split(r'[+\->]+', left_side)
        for part in species_parts:
            part = part.strip()
            # Remove leading digits and whitespace (e.g., '2 AB40_O12_ISF' -> 'AB40_O12_ISF')
            part = re.sub(r'^\d+\s*', '', part)
            # Skip empty strings and pure numbers
            if part and not part.isdigit():
                # Only add if it contains at least one letter (to avoid lone numbers)
                if re.search(r'[A-Za-z]', part):
                    # If species contains a space, only take what is to the right of the space
                    if ' ' in part:
                        part = part.split(' ', 1)[1]  # Split on first space and take right part
                    
                    # Check for malformed species names
                    if part.startswith('[') or part.endswith(']') or "'" in part:
                        error_msg = f"ERROR: Malformed species name '{part}' in line {line_num}: {line}"
                        errors.append(error_msg)
                    else:
                        species.add(part)
            
        # Extract parameters from right side
        right_side = parts[1].strip()
        # Find all words that look like parameters (containing letters, numbers, and underscores)
        param_matches = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', right_side)
        parameters.update(param_matches)
    
    # Remove species from parameters
    parameters = parameters - species
    
    # # Clean up species list - remove any that look like parameters
    # species = {s for s in species if not re.match(r'^[a-z]', s)}
    
    return sorted(list(species)), sorted(list(parameters)), errors

def write_list_to_file(items, filename):
    """Write a list of items to a file, one per line."""
    with open(filename, 'w') as f:
        for item in items:
            f.write(f"{item}\n")

def generate_species_declarations(species_list):
    """
    Generate species declarations in the format 'species species_X in compartment X'
    where X is the text after the final '_' in the species name.
    
    Args:
        species_list (list): List of species names
        
    Returns:
        str: Formatted species declarations
    """
    declarations = []
    for species in species_list:
        # Find the last underscore and get the compartment
        if '_' in species:
            compartment = species.split('_')[-1]
            declarations.append(f"substanceOnly species {species} in {compartment}")
        else:
            # If no underscore, use the species name as compartment
            declarations.append(f"substanceOnly species {species} in {species}")
    
    return '\n'.join(declarations)

def collect_unique_compartments_from_reactions(reactions):
    """
    Collect all unique compartment names from the reactions.
    
    Args:
        reactions (list): List of reaction dictionaries
        
    Returns:
        set: A set containing all unique compartment names.
    """
    unique_compartments = set()
    errors = []
    
    for i, reaction in enumerate(reactions):
        reaction_name = reaction.get("Reaction_name", f"Reaction_{i}")
        # Extract compartments from reactants and products
        reactants_str = reaction.get("Reactants", "")
        products_str = reaction.get("Products", "")
        
        # Parse reactants
        if reactants_str and reactants_str != "[0]":
            reactants_str = reactants_str.strip('[]')
            if reactants_str:
                reactants = [item.strip() for item in reactants_str.split(',')]
                for reactant in reactants:
                    if '_' in reactant:
                        compartment = reactant.split('_')[-1]
                        # Check for malformed compartment names
                        if compartment.startswith('[') or compartment.endswith(']') or "'" in compartment:
                            error_msg = f"ERROR: Malformed compartment name '{compartment}' in reactant '{reactant}' in reaction '{reaction_name}'"
                            errors.append(error_msg)
                        else:
                            unique_compartments.add(compartment)
        
        # Parse products
        if products_str and products_str != "[0]":
            products_str = products_str.strip('[]')
            if products_str:
                products = [item.strip() for item in products_str.split(',')]
                for product in products:
                    if '_' in product:
                        compartment = product.split('_')[-1]
                        # Check for malformed compartment names
                        if compartment.startswith('[') or compartment.endswith(']') or "'" in compartment:
                            error_msg = f"ERROR: Malformed compartment name '{compartment}' in product '{product}' in reaction '{reaction_name}'"
                            errors.append(error_msg)
                        else:
                            unique_compartments.add(compartment)
    
    return unique_compartments, errors

def generate_single_reaction_from_dict(reaction_dict):
    """
    Generate a single reaction string from a reaction dictionary.
    
    Args:
        reaction_dict (dict): The reaction dictionary
        
    Returns:
        str: The generated reaction string
    """
    reactants_str = reaction_dict["Reactants"]
    products_str = reaction_dict["Products"]
    rate_proto_str = reaction_dict["Rate_eqtn_prototype"]
    rate_type = reaction_dict["Rate_type"]
    
    # Extract compartment from reactants or products
    compartment = None
    compartment_reverse = None
    if reactants_str and reactants_str != "[0]":
        reactants_str_clean = reactants_str.strip('[]')
        if reactants_str_clean:
            first_reactant = reactants_str_clean.split(',')[0].strip()
            if '_' in first_reactant:
                compartment = first_reactant.split('_')[-1]
                # Check for malformed compartment names
                if compartment.startswith('[') or compartment.endswith(']') or "'" in compartment:
                    compartment = None
    if products_str and products_str != "[0]":
        products_str_clean = products_str.strip('[]')
        if products_str_clean:
            first_product = products_str_clean.split(',')[0].strip()
            if '_' in first_product:
                compartment_reverse = first_product.split('_')[-1]
                # Check for malformed compartment names
                if compartment_reverse.startswith('[') or compartment_reverse.endswith(']') or "'" in compartment_reverse:
                    compartment_reverse = None
    
    if reactants_str == "0" or reactants_str == "[0]":
        reactants = []
    else:
        reactants_str = reactants_str.strip('[]')
        if reactants_str:
            reactants = [item.strip() for item in reactants_str.split(',')]
        else:
            reactants = []

    if products_str == "0" or products_str == "[0]":
        products = []
    else:
        # Products can be a single item or a list
        prod_cleaned = products_str.strip('[]')
        if prod_cleaned:
            products = [item.strip() for item in prod_cleaned.split(',')]
        else:
            products = []

    reaction_string = ""

    if rate_type == "RMA":
        # RMA (reversible mass action) needs two rate constants
        rate_constants = [item.strip() for item in rate_proto_str.strip('[]').split(',')]
        if len(rate_constants) < 2:
            raise ValueError(f"RMA reaction '{reaction_dict.get('Reaction_name', 'UNKNOWN')}' requires two rate constants in Rate_eqtn_prototype, got: '{reaction_dict['Rate_eqtn_prototype']}'")
        # Forward
        reactants_fwd_str = " + ".join(reactants)
        products_fwd_str = " + ".join(products)
        if reactants:
            rate_fwd = f"{rate_constants[0]} * {' * '.join(reactants)}"
        else:
            rate_fwd = rate_constants[0]  # Zero-order reaction
        # Multiply by compartment volume for MA/RMA/custom_conc_per_time
        if compartment:
            rate_fwd = f"{rate_fwd} * V_{compartment}"
        reaction_string += f"{reactants_fwd_str} -> {products_fwd_str}; {rate_fwd}\n"
        # Reverse
        reactants_rev_str = " + ".join(products)
        products_rev_str = " + ".join(reactants)
        if products:
            rate_rev = f"{rate_constants[1]} * {' * '.join(products)}"
        else:
            rate_rev = rate_constants[1]  # Zero-order reaction
        # Multiply by compartment volume for MA/RMA/custom_conc_per_time
        if compartment_reverse:
            rate_rev = f"{rate_rev} * V_{compartment_reverse}"
        reaction_string += f"{reactants_rev_str} -> {products_rev_str}; {rate_rev}\n"
    elif rate_type == "BDF":
        # BDF (bidirectional flow) uses the same rate constant for both directions
        rate_constants = [item.strip() for item in rate_proto_str.strip('[]').split(',')]
        if len(rate_constants) < 1:
            raise ValueError(f"BDF reaction '{reaction_dict.get('Reaction_name', 'UNKNOWN')}' requires at least one rate constant in Rate_eqtn_prototype, got: '{reaction_dict['Rate_eqtn_prototype']}'")
        # Forward
        reactants_fwd_str = " + ".join(reactants)
        products_fwd_str = " + ".join(products)
        if reactants:
            rate_fwd = f"{rate_constants[0]} * {' * '.join(reactants)}"
        else:
            rate_fwd = rate_constants[0]  # Zero-order reaction
        reaction_string += f"{reactants_fwd_str} -> {products_fwd_str}; {rate_fwd}\n"
        # Reverse (same rate constant)
        reactants_rev_str = " + ".join(products)
        products_rev_str = " + ".join(reactants)
        if products:
            rate_rev = f"{rate_constants[0]} * {' * '.join(products)}"
        else:
            rate_rev = rate_constants[0]  # Zero-order reaction
        reaction_string += f"{reactants_rev_str} -> {products_rev_str}; {rate_rev}\n"
    elif rate_type == "MA" or rate_type == "UDF":
        # UDF is treated the same as MA (unidirectional flow)
        rate_constants = [item.strip() for item in rate_proto_str.strip('[]').split(',')]
        # Forward
        reactants_fwd_str = " + ".join(reactants)
        products_fwd_str = " + ".join(products)
        if reactants:
            rate_fwd = f"{rate_constants[0]} * {' * '.join(reactants)}"
        else:
            rate_fwd = rate_constants[0]  # Zero-order reaction
        # Multiply by compartment volume for MA/RMA/custom_conc_per_time
        if rate_type == "MA" and compartment:
            rate_fwd = f"{rate_fwd} * V_{compartment}"
        if rate_type == "MA" and not reactants and compartment_reverse:
            rate_fwd = f"{rate_fwd} * V_{compartment_reverse}"
        reaction_string += f"{reactants_fwd_str} -> {products_fwd_str}; {rate_fwd}\n"
        
    elif rate_type == "custom_conc_per_time":
        reactants_side = " + ".join(reactants)
        products_side = " + ".join(products) if products else ""
        
        rate_eqtn = rate_proto_str
        # if it was a list-like string, take the content.
        if rate_eqtn.startswith('[') and rate_eqtn.endswith(']'):
            rate_eqtn = rate_eqtn.strip('[]')
        
        # Multiply by compartment volume for custom_conc_per_time
        if compartment:
            rate_eqtn = f"{rate_eqtn} * V_{compartment}"
        
        reaction_string += f"{reactants_side} -> {products_side}; {rate_eqtn}\n"
        
    elif rate_type == "custom_amt_per_time":
        reactants_side = " + ".join(reactants)
        products_side = " + ".join(products) if products else ""
        
        rate_eqtn = rate_proto_str
        # if it was a list-like string, take the content.
        if rate_eqtn.startswith('[') and rate_eqtn.endswith(']'):
            rate_eqtn = rate_eqtn.strip('[]')
        
        # No multiplication for custom_amt_per_time
        
        reaction_string += f"{reactants_side} -> {products_side}; {rate_eqtn}\n"
        
    elif rate_type == "custom":
        reactants_side = " + ".join(reactants)
        products_side = " + ".join(products) if products else ""
        
        rate_eqtn = rate_proto_str
        # if it was a list-like string, take the content.
        if rate_eqtn.startswith('[') and rate_eqtn.endswith(']'):
            rate_eqtn = rate_eqtn.strip('[]')
        
        # Use rate expression as-is, no multiplication by species or volume
        
        reaction_string += f"{reactants_side} -> {products_side}; {rate_eqtn}\n"
    
    return reaction_string

def convert_species_to_concentrations(reaction_string, species_list):
    """
    Convert species in rate equations from amounts to concentrations by dividing by compartment volumes.
    This is needed for antimony solver where species names refer to amounts but rate equations need concentrations.
    
    Args:
        reaction_string (str): The reaction string with species as amounts
        species_list (list): List of species names
        
    Returns:
        str: Modified reaction string with species converted to concentrations in rate equations
    """
    # Create a mapping of species to their compartments
    species_to_compartment = {}
    for species in species_list:
        if '_' in species:
            compartment = species.split('_')[-1]
            species_to_compartment[species] = compartment
        else:
            # If no underscore, use the species name as compartment
            species_to_compartment[species] = species
    
    # Process each line
    lines = reaction_string.split('\n')
    modified_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:  # Skip empty lines
            modified_lines.append(line)
            continue
        
        # Skip compartment declarations
        if line.startswith('compartment '):
            modified_lines.append(line)
            continue
        
        # Skip species declarations (they start with "substanceOnly species")
        if line.startswith('substanceOnly species'):
            modified_lines.append(line)
            continue
        
        # Split at semicolon to separate reactants/products from rate
        parts = line.split(';')
        if len(parts) != 2:
            modified_lines.append(line)
            continue
        
        reactants_products = parts[0].strip()
        rate_equation = parts[1].strip()
        
        # Modify the rate equation to convert species to concentrations
        modified_rate = rate_equation
        
        # Find all species in the rate equation and replace them with species/V_compartment
        for species in species_list:
            if species in modified_rate:
                compartment = species_to_compartment[species]
                # Use word boundaries to avoid partial matches
                pattern = r'\b' + re.escape(species) + r'\b'
                replacement = f"({species}/V_{compartment})"
                modified_rate = re.sub(pattern, replacement, modified_rate)
        
        # Reconstruct the line
        modified_line = f"{reactants_products}; {modified_rate}"
        modified_lines.append(modified_line)
    
    return '\n'.join(modified_lines)

def add_reaction_names_to_string(reaction_string, reactions):
    """
    Add reaction names to each reaction line in the format "{reaction_name} : ".
    
    Args:
        reaction_string (str): The reaction string without names
        reactions (list): List of reaction dictionaries with Reaction_name keys
        
    Returns:
        str: Reaction string with names prepended
    """
    lines = reaction_string.split('\n')
    modified_lines = []
    reaction_index = 0
    reaction_line_count = 0  # Track how many lines the current reaction has generated
    
    for line in lines:
        line = line.strip()
        if not line:  # Keep empty lines as-is
            modified_lines.append(line)
            continue
        
        # Skip compartment and species declarations
        if line.startswith('compartment ') or line.startswith('substanceOnly species'):
            modified_lines.append(line)
            continue
        
        # This is a reaction line - add reaction name
        if reaction_index < len(reactions):
            reaction_name = reactions[reaction_index].get("Reaction_name", "UNKNOWN")
            rate_type = reactions[reaction_index].get("Rate_type", "")
            
            # Add _fwd and _rev suffixes for BDF and RMA reactions
            if rate_type in ["RMA", "BDF"]:
                if reaction_line_count == 0:
                    # First line gets _fwd suffix
                    reaction_name_with_suffix = f"{reaction_name}_fwd"
                else:
                    # Second line gets _rev suffix
                    reaction_name_with_suffix = f"{reaction_name}_rev"
            else:
                reaction_name_with_suffix = reaction_name
            
            modified_lines.append(f"{reaction_name_with_suffix} : {line}")
            reaction_line_count += 1
            
            # Determine how many lines this reaction should generate
            if rate_type in ["RMA", "BDF"]:
                lines_per_reaction = 2
            else:
                lines_per_reaction = 1
            
            # Move to next reaction if we've processed all lines for this one
            if reaction_line_count >= lines_per_reaction:
                reaction_index += 1
                reaction_line_count = 0
        else:
            # Fallback if we run out of reactions
            modified_lines.append(line)
    
    return '\n'.join(modified_lines)

def read_reactions_from_txt(txt_file_path):
    """
    Read reactions from a text file where each line is a Python dictionary string.
    
    Args:
        txt_file_path (str): Path to the text file
        
    Returns:
        list: List of reaction dictionaries
    """
    reactions = []
    
    with open(txt_file_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:  # Skip empty lines
                continue
            
            try:
                # Convert string representation of dict to actual dict
                reaction_dict = eval(line)
                reactions.append(reaction_dict)
            except Exception as e:
                print(f"Error parsing line {line_num}: {e}")
                print(f"Line content: {line}")
                continue
    
    return reactions

def generate_antimony_from_txt(txt_file_path, name):
    """
    Generate Antimony script from a text file containing individual reactions.
    
    Args:
        txt_file_path (str): Path to the text file
        name (str): Name to use in place of 'Geerts' in filenames
        
    Returns:
        str: Generated Antimony script
    """
    # Read reactions from text file
    reactions = read_reactions_from_txt(txt_file_path)
    
    # Collect unique compartments and check for errors
    unique_compartments, compartment_errors = collect_unique_compartments_from_reactions(reactions)
    
    # Generate reactions
    reaction_string = ""
    for reaction in reactions:
        reaction_string += generate_single_reaction_from_dict(reaction)
    
    # Extract species and parameters from reactions
    species, parameters, species_errors = extract_species_and_parameters_from_reactions(reaction_string)
    
    # Add compartment volume parameters to the parameters list
    for compartment in unique_compartments:
        parameters.append(f"V_{compartment}")
    
    # Make parameters unique by converting to set and back to sorted list
    parameters = sorted(list(set(parameters)))
    
    # Convert species in rate equations to concentrations for antimony solver
    reactions_antimony = convert_species_to_concentrations(reaction_string, species)
    
    # Add reaction names to the reaction string (after all processing)
    reactions_antimony = add_reaction_names_to_string(reactions_antimony, reactions)
    
    # Generate the complete script with compartments, species, and reactions
    complete_script = ""
    
    # Add compartment declarations
    for compartment in sorted(unique_compartments):
        complete_script += f"compartment {compartment} = V_{compartment}\n"
    complete_script += "\n"  # Add blank line after compartments
    
    # Add species declarations
    species_declarations = generate_species_declarations(species)
    complete_script += species_declarations
    complete_script += "\n\n"  # Add blank lines after species
    
    # Add reactions (antimony version with species converted to concentrations in rate equations)
    complete_script += reactions_antimony
    
    # Combine all errors
    all_errors = compartment_errors + species_errors
    
    return complete_script, species, parameters, unique_compartments, all_errors

if __name__ == "__main__":
    # Check if name argument is provided
    if len(sys.argv) < 2:
        print("Usage: python txt_to_antimony.py <name>")
        print("Example: python txt_to_antimony.py Smith")
        print("Exiting...")
        exit()
    else:
        name = sys.argv[1]
    rxn_filename = f"generated/{name}_all_reactions.txt"
    print(rxn_filename)
    # Generate Antimony script from text file
    complete_script, species, parameters, unique_compartments, errors = generate_antimony_from_txt(rxn_filename, name)
    
    # Write complete script to file with name
    antimony_filename = f"antimony_models/Antimony_{name}_all_reactions.txt"
    with open(antimony_filename, "w") as f:
        f.write(complete_script)
    
    # Write unique compartments to file with name
    compartments_filename = f"generated/unique_compartments_{name}.txt"
    with open(compartments_filename, "w") as f:
        for compartment in sorted(unique_compartments):
            f.write(f"{compartment}\n")
    
    # Write species and parameters to files with name
    species_filename = f'generated/unique_species_{name}.txt'
    parameters_filename = f'generated/unique_parameters_{name}.txt'
    write_list_to_file(species, species_filename)
    write_list_to_file(parameters, parameters_filename)
    
    # Write errors to file with name
    errors_filename = f"generated/conversion_errors_{name}.log"
    with open(errors_filename, "w") as f:
        for error in errors:
            f.write(f"{error}\n")
    
    # Print summary
    print(f"Found {len(unique_compartments)} unique compartments:")
    for compartment in sorted(unique_compartments):
        print(f"  - {compartment}")
    print(f"Unique compartments written to '{compartments_filename}'")
    print()
    
    print(f"Found {len(species)} unique species and {len(parameters)} unique parameters")
    print(f"Species written to '{species_filename}'")
    print(f"Parameters written to '{parameters_filename}'")
    print()
    
    print(f"Antimony script written to '{antimony_filename}'")
    print()
    
    # Print errors
    if errors:
        print(f"Found {len(errors)} errors during conversion:")
        print("=" * 50)
        for error in errors:
            print(error)
        print("=" * 50)
        print(f"Errors also written to '{errors_filename}'")
        print()
    else:
        print("No errors found during conversion.")
        print()
    
    # print("Generated Antimony script:")
    # print("=" * 50)
    # print(complete_script) 
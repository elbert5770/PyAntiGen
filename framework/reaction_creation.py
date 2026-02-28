def reaction_creation(all_reactions, counter, Reaction_name, Reactants, Products, Rate_type, Rate_eqtn_prototype):
    """Helper function to create and add reactions to the all_reactions list"""
    
    
    # Valid Rate_type values from RxnDict_to_antimony.py
    valid_rate_types = {"RMA", "BDF", "MA", "UDF", "custom_conc_per_time", "custom_amt_per_time", "custom"}
    
    # Check for missing elements and warn
    missing_elements = []
    if Reaction_name is None or Reaction_name == "":
        missing_elements.append("Reaction_name")
        Reaction_name = "NA"
    if Reactants is None:
        missing_elements.append("Reactants")
    if Products is None:
        missing_elements.append("Products")
    if Rate_type is None or Rate_type == "":
        missing_elements.append("Rate_type")
    if Rate_eqtn_prototype is None or Rate_eqtn_prototype == "":
        missing_elements.append("Rate_eqtn_prototype")
    
    # Warn about missing elements
    if missing_elements:
        warnings.warn(f"Missing elements in reaction creation: {', '.join(missing_elements)}. Reaction_name set to 'NA' if it was missing.", UserWarning)
    
    # Check for valid Rate_type
    if Rate_type is not None and Rate_type != "" and Rate_type not in valid_rate_types:
        warnings.warn(f"Invalid Rate_type '{Rate_type}' provided. Valid types are: {', '.join(sorted(valid_rate_types))}", UserWarning)
    
    # Remove spaces from Reaction_name
    Reaction_name = Reaction_name.replace(" ", "")
    
    if Rate_type == "RMA" or Rate_type == "BDF":
        counter += 2
    else:
        counter += 1
    Reaction_dict = {"Reaction_name": Reaction_name,"Reactants": Reactants,"Products": Products,"Rate_type": Rate_type,"Rate_eqtn_prototype": Rate_eqtn_prototype,}
    
    all_reactions.append(Reaction_dict)
    

    return counter, all_reactions

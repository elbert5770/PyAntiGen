def all_species(r):
    observed_species = ['time'] + list(r.getFloatingSpeciesIds()) + list(r.getBoundarySpeciesIds()) + list(r.getAssignmentRuleIds()) + list(r.getGlobalParameterIds())
    return observed_species

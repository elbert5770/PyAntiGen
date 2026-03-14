from framework.models import (
    reaction_from_args,
    VALID_RATE_TYPES,
    RATE_TYPES_TWO_CONSTANTS,
)


def reaction_creation(
    all_reactions,
    counter,
    Reaction_name,
    Reactants,
    Products,
    Rate_type,
    Rate_eqtn_prototype,
    compartment=None,
    compartment_reverse=None,
):
    """
    Create and append a validated reaction to all_reactions.
    Accepts reactants/products as either list of strings or bracket string (e.g. "[A, B]").
    Validates required fields and rate type at call time; raises on invalid input.
    """
    try:
        reaction = reaction_from_args(
            name=Reaction_name,
            reactants=Reactants,
            products=Products,
            rate_type=Rate_type,
            rate_eqtn=Rate_eqtn_prototype,
            compartment=compartment,
            compartment_reverse=compartment_reverse,
        )
    except ValueError as e:
        raise ValueError(f"add_reaction validation failed: {e}") from e

    if reaction.Rate_type in RATE_TYPES_TWO_CONSTANTS:
        counter += 2
    else:
        counter += 1

    all_reactions.append(reaction.to_dict())
    return counter, all_reactions

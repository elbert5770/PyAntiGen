import os
from framework.isotopomer_tools import ensure_isotopes_format
from framework.model_generation import generate_model

class PyAntiGen:
    def __init__(self, name, isotopes=None):
        self.name = name
        self.isotopes = ensure_isotopes_format(isotopes or [''])
        self.reactions = []
        self.rules = []
        self.counter = 0

    def add_reaction(self, name, reactants, products, rate_type, rate_eqtn):
        from framework.reaction_creation import reaction_creation
        self.counter, self.reactions = reaction_creation(
            self.reactions, self.counter, name, reactants, products, rate_type, rate_eqtn
        )

    def add_rule(self, rule):
        self.rules.append(rule)

    def generate(self, calling_file_path):
        """
        Generates the Antimony model. This wraps the existing model_generation logic.
        """
        def build_reactions_wrapper(Isotopes):
            # The isotopes passed in here by generate_model are essentially self.isotopes
            # because we formatted them earlier. So we can just return our state.
            return self.reactions, self.rules
            
        generate_model(build_reactions_wrapper, self.isotopes, calling_file_path)

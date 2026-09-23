class PyAntiGenModule:
    """
    Base class for declarative PyAntiGen modules.
    Modules should inherit from this and implement the build() method.
    When instantiated, the module will automatically build and add its reactions 
    to the passed PyAntiGen model instance.
    """
    def __init__(self, model, **kwargs):
        self.model = model
        self.config = kwargs
        
        # Build reactions automatically on init
        self.build()

    def build(self):
        """Override this in subclasses to define reactions."""
        raise NotImplementedError("Modules must implement a build() method.")

    def add_reaction(
        self,
        name,
        reactants,
        products,
        rate_type,
        rate_eqtn,
        compartment=None,
        compartment_reverse=None,
    ):
        """Helper to add reactions directly into the parent model."""
        self.model.add_reaction(
            name,
            reactants,
            products,
            rate_type,
            rate_eqtn,
            compartment=compartment,
            compartment_reverse=compartment_reverse,
        )

    def add_rule(self, rule):
        """Helper to add rules directly into the parent model."""
        self.model.add_rule(rule)

"""
CNS Flows Module 1f

"""

# Import the reaction_creation function from the reaction_creation module
from framework.reaction_creation import reaction_creation
from framework.module_base import PyAntiGenModule

class Tissue_RxnsModule(PyAntiGenModule):
    """
    Create Tissue compartmental flow reactions for a given species.
    Uses self.config['Species'] and self.config['No_Isotope_SpeciesList'].
    """
    def build(self):
        Species = self.config.get('Species')
        No_Isotope_SpeciesList = self.config.get('No_Isotope_SpeciesList', [])
        
        Isotopes = [''] if Species in No_Isotope_SpeciesList else self.model.isotopes

        # Tissue flow (unidirectional) - Tissue circulation and drainage

        
        for Isotope in Isotopes:
            Isotope_str = f"_{Isotope}_" if Isotope else "_"
            for Comp in ['TissueEndosomal','BBB','BCSFB']:
                Reaction_name = f"DegradationWithinEndosome_{Species}{Isotope_str}{Comp}"
                Reactants = f"[{Species}{Isotope_str}{Comp}]"
                Products = f"[0]"
                Rate_type = "MA"
                Rate_eqtn_prototype = "Kdeg"
                self.add_reaction(Reaction_name, Reactants, Products, Rate_type, Rate_eqtn_prototype)

                Reaction_name = f"BindingToFcRn_{Species}{Isotope_str}{Comp}"
                Reactants = f"[{Species}{Isotope_str}{Comp},FcRn_{Comp}]"
                Products = f"[{Species}{Isotope_str}_FcRn_{Comp}]"
                Rate_type = "RMA"
                Rate_eqtn_prototype = "[Kon_FcRn,Koff_FcRn]"
                self.add_reaction(Reaction_name, Reactants, Products, Rate_type, Rate_eqtn_prototype)

"""
Abeta Production and Clearance Module (SILK-enabled)

This module creates Abeta production and clearance reactions:
- Abeta synthesis from C99 (gamma-secretase cleavage)
- Abeta degradation (IDE-mediated clearance)

SILK Support:
- When SILK=True: Creates both labeled (_L) and unlabeled reactions using linker system
- Rate equations account for competitive inhibition between labeled and unlabeled species
- Uses linker list ['_', '_L_'] to generate appropriate species names
"""

from framework.module_base import PyAntiGenModule

class AbetaProductionModule(PyAntiGenModule):
    """
    Create Abeta production and clearance reactions for a given species.
    Uses self.config['Species'] to pass in the species.
    """
    def build(self):
        Species = self.config.get('Species')

        for Comp in ['BrainISF']:
            Total_C83 = "(" + " + ".join([f"C83{f'_{iso}_' if iso else '_'}{Comp}" for iso in self.model.isotopes]) + ")"
            Total_C99 = "(" + " + ".join([f"C99{f'_{iso}_' if iso else '_'}{Comp}" for iso in self.model.isotopes]) + ")"
            
            for Isotope in self.model.isotopes:
                Isotope_str = f"_{Isotope}_" if Isotope else "_"
                
                # Abeta Synthesis
                Reaction_name = f"{Species}Synthesis{Isotope_str}{Comp}"
                Reactants = f"[C99{Isotope_str}{Comp}]"
                Products = f"[{Species}{Isotope_str}{Comp}]"
                Rate_type = "MA"
                Rate_eqtn_prototype = f"(k_gammasec_{Species}/Km_C99_Abeta)/(1+{Total_C83}/Km_C83_p3+{Total_C99}/Km_C99_Abeta)"
                self.add_reaction(Reaction_name, Reactants, Products, Rate_type, Rate_eqtn_prototype)
               
                # Abeta Degradation
                Reaction_name = f"{Species}Degradation{Isotope_str}{Comp}"
                Reactants = f"[{Species}{Isotope_str}{Comp}]"
                Products = f"[0]"
                Rate_type = "MA"
                Rate_eqtn_prototype = f"k_BPD_{Species}"
                self.add_reaction(Reaction_name, Reactants, Products, Rate_type, Rate_eqtn_prototype)
               
                # Abeta Exchange
                Reaction_name = f"{Species}Exchange{Isotope_str}{Comp}"
                Reactants = f"[{Species}{Isotope_str}{Comp}]"
                Products = f"[{Species}{Isotope_str}Exchange]"
                Rate_type = "RMA"
                Rate_eqtn_prototype = f"[k_ex_{Species},k_ret_{Species}]"
                self.add_reaction(Reaction_name, Reactants, Products, Rate_type, Rate_eqtn_prototype)

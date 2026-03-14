"""
APP Processing Module (SILK-enabled)

This module creates APP processing reactions:
- APP synthesis
- C83 synthesis (alpha-secretase cleavage)
- C99 synthesis (beta-secretase cleavage)
- APP internalization and recycling
- C83 internalization and p3 production

SILK Support:
- When SILK=True: Creates both labeled (_L) and unlabeled reactions using linker system
- Rate equations account for competitive inhibition between labeled and unlabeled species
- Uses linker list ['_', '_L_'] to generate appropriate species names
- The labeled designation can be modified by the Modifications list
"""

from framework.module_base import PyAntiGenModule

class APP_ReactionsModule(PyAntiGenModule):
    """
    Create C99-related reactions.
    """
    def build(self):
        for Comp in ['BrainISF']:
            Total_APP = "(" + " + ".join([f"APP{f'_{iso}_' if iso else '_'}{Comp}" for iso in self.model.isotopes]) + ")"
            Total_C99 = "(" + " + ".join([f"C99{f'_{iso}_' if iso else '_'}{Comp}" for iso in self.model.isotopes]) + ")"
            Total_C83 = "(" + " + ".join([f"C83{f'_{iso}_' if iso else '_'}{Comp}" for iso in self.model.isotopes]) + ")"
     
            labeled_isotopes = [iso for iso in self.model.isotopes if iso]
            if labeled_isotopes:
                sum_f_Isotope = "(" + " + ".join([f"f_{iso}" for iso in labeled_isotopes]) + ")"
            else:
                sum_f_Isotope = "0"
            
            for Isotope in self.model.isotopes:
                Isotope_str = f"_{Isotope}_" if Isotope else "_"
                
                # APP Synthesis
                Reaction_name = f"APPSynthesis{Isotope_str}{Comp}"
                Reactants = f"[0]"
                Products = f"[APP{Isotope_str}{Comp}]"
                Rate_type = "MA"
                if Isotope == '':
                    Rate_eqtn_prototype = f"k_APP*(1-{sum_f_Isotope})"
                else:
                    Rate_eqtn_prototype = f"k_APP*f_{Isotope}"
                self.add_reaction(Reaction_name, Reactants, Products, Rate_type, Rate_eqtn_prototype)
               
                # C83 Synthesis
                Reaction_name = f"C83Synthesis{Isotope_str}{Comp}"
                Reactants = f"[APP{Isotope_str}{Comp}]"
                Products = f"[C83{Isotope_str}{Comp},sAPPa{Isotope_str}{Comp}]"
                Rate_type = "MA"
                Rate_eqtn_prototype = f"(Vm_APP_C83/Km_APP_C83)/(1+{Total_APP}/Km_APP_C83+{Total_C99}/Km_C99_C83)"
                self.add_reaction(Reaction_name, Reactants, Products, Rate_type, Rate_eqtn_prototype)

                # C99 Synthesis
                Reaction_name = f"C99Synthesis{Isotope_str}{Comp}"
                Reactants = f"[APP{Isotope_str}{Comp}]"
                Products = f"[C99{Isotope_str}{Comp},sAPPb{Isotope_str}{Comp}]"
                Rate_type = "MA"
                Rate_eqtn_prototype = f"(Vm_APP_C99/Km_APP_C99)/(1+{Total_APP}/Km_APP_C99)"
                self.add_reaction(Reaction_name, Reactants, Products, Rate_type, Rate_eqtn_prototype)

                # C99 to C83
                Reaction_name = f"C99toC83{Isotope_str}{Comp}"
                Reactants = f"[C99{Isotope_str}{Comp}]"
                Products = f"[C83{Isotope_str}{Comp}]"
                Rate_type = "MA"
                Rate_eqtn_prototype = f"(Vm_C99_C83/Km_C99_C83)/(1+{Total_APP}/Km_APP_C83+{Total_C99}/Km_C99_C83)"
                self.add_reaction(Reaction_name, Reactants, Products, Rate_type, Rate_eqtn_prototype)
               
                # C83 to p3
                Reaction_name = f"C83toP3{Isotope_str}{Comp}"
                Reactants = f"[C83{Isotope_str}{Comp}]"
                Products = f"[p3{Isotope_str}{Comp}]"
                Rate_type = "MA"
                Rate_eqtn_prototype = f"(Vm_C83_p3/Km_C83_p3)/(1+{Total_C83}/Km_C83_p3+{Total_C99}/Km_C99_Abeta)"
                self.add_reaction(Reaction_name, Reactants, Products, Rate_type, Rate_eqtn_prototype)

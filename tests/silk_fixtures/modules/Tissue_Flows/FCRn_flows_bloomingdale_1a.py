

# Import the reaction_creation function from the reaction_creation module
from framework.reaction_creation import reaction_creation
from framework.module_base import PyAntiGenModule

class FcRn_FlowsModule(PyAntiGenModule):
    """
    Create Tissue compartmental flow reactions for a given species.
    Uses self.config['Species'] and self.config['No_Isotope_SpeciesList'].
    """
    def build(self):
        Species = self.config.get('Species')
        No_Isotope_SpeciesList = self.config.get('No_Isotope_SpeciesList', [])
        
        Isotopes = [''] if Species in No_Isotope_SpeciesList else self.model.isotopes

        # Tissue flow (unidirectional) - Tissue circulation and drainage
        fcrn_flow_rates = {
            # Tissue distribution to subarachnoid space and spinal compartments
            ('BBB', 'BrainISF'): "(1 - FR_B) * CLup_BBB",
            ('BBB', 'BrainVascular'): "FR_B * CLup_BBB",
            ('BCSFB', 'BrainISF'): "(1 - FR_B) * CLup_BCSFB",
            ('BCSFB', 'BrainVascular'): "FR_B * CLup_BCSFB",
            ('TissueEndosomal', 'TissueISF'): "(1 - FR) * CLup_Tissue",
            ('TissueEndosomal', 'TissueVascular'): "FR * CLup_Tissue",
        }
        
        for Isotope in Isotopes:
            Isotope_str = f"_{Isotope}_" if Isotope else "_"
            for Comp1, Comp2 in fcrn_flow_rates.keys():
                Reaction_name = f"Recycle_FcRn_{Species}{Isotope_str}{Comp1}_{Comp2}"
                Reactants = f"[{Species}{Isotope_str}_FcRn_{Comp1}]"
                Products = f"[{Species}{Isotope_str}{Comp2},FcRn_{Comp1}]"
                Rate_type = "UDF"
                Rate_eqtn_prototype = fcrn_flow_rates[(Comp1, Comp2)]
                self.add_reaction(Reaction_name, Reactants, Products, Rate_type, Rate_eqtn_prototype)




# Import the reaction_creation function from the reaction_creation module
from framework.reaction_creation import reaction_creation
from framework.module_base import PyAntiGenModule

class Tissue_FlowsModule(PyAntiGenModule):
    """
    Create Tissue compartmental flow reactions for a given species.
    Uses self.config['Species'] and self.config['No_Isotope_SpeciesList'].
    """
    def build(self):
        Species = self.config.get('Species')
        No_Isotope_SpeciesList = self.config.get('No_Isotope_SpeciesList', [])
        
        Isotopes = [''] if Species in No_Isotope_SpeciesList else self.model.isotopes

        # Tissue flow (unidirectional) - Tissue circulation and drainage
        tissue_flow_rates = {
            # Tissue distribution to subarachnoid space and spinal compartments
            ('TissueVascular', 'TissueISF'): "(1 - RC_TissueVascular) * LT",
            ('TissueISF', 'Lymph'): "(1 - RC_TissueLymph) * LT",
            ('Plasma', 'TissueVascular'): "QT",
            ('TissueVascular','Plasma'): "(QT - LT)",
            ('Lymph', 'Plasma'): "(LB + LT)",
            ('Plasma','BrainVascular'): "QB",
            ('BrainVascular','Plasma'): "(QB - LB)",
            ('TissueVascular','TissueEndosomal'): "CLup_Tissue",
            ('TissueISF','TissueEndosomal'): "CLup_Tissue",
        }
        
        for Isotope in Isotopes:
            Isotope_str = f"_{Isotope}_" if Isotope else "_"
            for Comp1, Comp2 in tissue_flow_rates.keys():
                Reaction_name = f"FlowWithinTissue_{Species}{Isotope_str}{Comp1}_{Comp2}"
                Reactants = f"[{Species}{Isotope_str}{Comp1}]"
                Products = f"[{Species}{Isotope_str}{Comp2}]"
                Rate_type = "UDF"
                Rate_eqtn_prototype = tissue_flow_rates[(Comp1, Comp2)]
                self.add_reaction(Reaction_name, Reactants, Products, Rate_type, Rate_eqtn_prototype)


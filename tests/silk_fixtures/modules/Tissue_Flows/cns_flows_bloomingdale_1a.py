"""
CNS Flows Module 1f

"""

# Import the reaction_creation function from the reaction_creation module
from framework.reaction_creation import reaction_creation
from framework.module_base import PyAntiGenModule

class CNS_FlowsModule(PyAntiGenModule):
    """
    Create CNS compartmental flow reactions for a given species.
    Uses self.config['Species'] and self.config['No_Isotope_SpeciesList'].
    """
    def build(self):
        Species = self.config.get('Species')
        No_Isotope_SpeciesList = self.config.get('No_Isotope_SpeciesList', [])
        
        Isotopes = [''] if Species in No_Isotope_SpeciesList else self.model.isotopes

        # CNS tissue flow (unidirectional) - CSF circulation and drainage
        cns_tissue_flow_rates = {
            # CSF distribution to subarachnoid space and spinal compartments
            ('BrainVascular', 'BrainISF'): "(1 - RC_BBB) * QB_ECF",
            ('BrainISF', 'Lymph'): "(1 - RC_BrainISF) * QB_ECF",
            ('BrainVascular', 'CSF'): "(1 - RC_BCSFB) * QB_CSF",
            ('CSF', 'Lymph'): "(1 - RC_CSF) * QB_ECF",
            ('BrainVascular','BBB'): "CLup_BBB",
            ('BrainISF','BBB'): "CLup_BBB",
            ('BrainVascular','BCSFB'): "CLup_BCSFB",
            ('CSF','BCSFB'): "CLup_BCSFB",
        }
        
        for Isotope in Isotopes:
            Isotope_str = f"_{Isotope}_" if Isotope else "_"
            for Comp1, Comp2 in cns_tissue_flow_rates.keys():
                Reaction_name = f"FlowWithinTissue_{Species}{Isotope_str}{Comp1}_{Comp2}"
                Reactants = f"[{Species}{Isotope_str}{Comp1}]"
                Products = f"[{Species}{Isotope_str}{Comp2}]"
                Rate_type = "UDF"
                Rate_eqtn_prototype = cns_tissue_flow_rates[(Comp1, Comp2)]
                self.add_reaction(Reaction_name, Reactants, Products, Rate_type, Rate_eqtn_prototype)

        # CNS bidirectional flow - glymphatic system and oscillatory flows
        cns_bidirectional_flow_rates = {
            # Glymphatic system flows (waste clearance)
            ('CSF', 'BrainISF'): "QB_ECF",
            
        }
        
        for Isotope in Isotopes:
            Isotope_str = f"_{Isotope}_" if Isotope else "_"
            for Comp1, Comp2 in cns_bidirectional_flow_rates.keys():
                Reaction_name = f"BidirectionalFlowWithinTissue_{Species}{Isotope_str}{Comp1}_{Comp2}"
                Reactants = f"[{Species}{Isotope_str}{Comp1}]"
                Products = f"[{Species}{Isotope_str}{Comp2}]"
                Rate_type = "BDF"
                Rate_eqtn_prototype = cns_bidirectional_flow_rates[(Comp1, Comp2)]
                self.add_reaction(Reaction_name, Reactants, Products, Rate_type, Rate_eqtn_prototype)

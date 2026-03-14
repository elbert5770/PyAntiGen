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
            ('CV', 'SAS'): "(Q_CSF - Q_SN - Q_LP)",
            ('SAS', 'Lymph'): "(Q_CSF - Q_SN - Q_LP)",
            ('CV', 'SP1'): "(Q_SN + Q_LP)",
            ('SP1','Lymph'): "(1/3) * Q_SN",
            ('SP1', 'SP2'): "((2/3) * Q_SN + Q_LP)",
            ('SP2','Lymph'): "(1/3) * Q_SN",
            ('SP2', 'SP3'): "((1/3) * Q_SN + Q_LP)",
            ('SP3','Lymph'): "(1/3) * Q_SN",
            ('SP3', 'LumbarDrain'): "(Q_LP + Q_refill)"
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
            ('SAS', 'BrainISF'): "Q_glymph",
            
            # Oscillatory flows (bidirectional CSF movement)
            ('CV', 'SAS'): "Q_osc",
            ('CV', 'SP1'): "Q_osc",
            # Oscillatory flows for all 30 spinal compartments
            ('SP1', 'SP2'): "Q_osc",
            ('SP2', 'SP3'): "Q_osc",
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

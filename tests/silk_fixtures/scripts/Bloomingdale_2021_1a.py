# Abeta Model Builder
# Model: Bloomingdale_2021_1a
# Recent Updates:
import sys
import os

# Add the project root to the Python path
current_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, project_root)

from framework.pyantigen import PyAntiGen
from modules.Tissue_Flows.cns_flows_bloomingdale_1a import CNS_FlowsModule
from modules.Tissue_Flows.Tissue_flows_bloomingdale_1a import Tissue_FlowsModule
from modules.Tissue_Flows.FCRn_flows_bloomingdale_1a import FcRn_FlowsModule
from modules.Tissue_Rxns.Tissue_rxns_bloomingdale_1a import Tissue_RxnsModule

if __name__ == "__main__":
    # Natural abundance plus exogenous leucine isotope label
    # Isotopes = ['','13C6Leu']
    # Natural abundance only
    Isotopes = ['']
    
    # Initialize the PyAntiGen model
    model = PyAntiGen(name="Bloomingdale_2021_1a", isotopes=Isotopes)
    
    SpeciesList = ['Antibody']

    
   
    # Brain and Flow Modules (Species Specific)
    for Species in SpeciesList:
        CNS_FlowsModule(model, Species=Species)
        Tissue_FlowsModule(model, Species=Species)
        FcRn_FlowsModule(model,Species=Species)
        Tissue_RxnsModule(model,Species=Species)

    print(f"Reactions generated: {model.counter}")
    print(f"Rules generated: {len(model.rules)}")
    
    # Write and Convert to Antimony
    model.generate(__file__)

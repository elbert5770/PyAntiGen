# Abeta Model Builder
# Model: Elbert_2022_1a
# Recent Updates:

"""
Model Builder: Elbert_2022_1a
Assemble Antimony reactions from modules
"""
import sys
import os

# Add the project root to the Python path
current_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, project_root)

from framework.pyantigen import PyAntiGen
from modules.Tissue_Flows.cns_flows_elbert_1a import CNS_FlowsModule
from modules.Abeta.Abeta_production_clearance_elbert_1a import AbetaProductionModule
from modules.Abeta.APP_reactions_elbert_1a import APP_ReactionsModule

if __name__ == "__main__":
    # Natural abundance plus exogenous leucine isotope label
    Isotopes = ['','13C6Leu']
    # Natural abundance only
    # Isotopes = ['']
    
    # Initialize the PyAntiGen model
    model = PyAntiGen(name="Elbert_2022_1a", isotopes=Isotopes)
    
    SpeciesList_Abeta_peptides = ['AB38','AB40','AB42']
    No_Isotope_SpeciesList = ['']

    # APP reactions module (1-liner automatic add)
    APP_ReactionsModule(model)
   
    # Brain and Flow Modules (Species Specific)
    for Species in SpeciesList_Abeta_peptides:
        CNS_FlowsModule(model, Species=Species, No_Isotope_SpeciesList=No_Isotope_SpeciesList)
        AbetaProductionModule(model, Species=Species)

    print(f"Reactions generated: {model.counter}")
    print(f"Rules generated: {len(model.rules)}")
    
    # Write and Convert to Antimony
    model.generate(__file__)

import sys
import os

from framework.module_base import PyAntiGenModule

class Antibody_PK_Lin2022Module(PyAntiGenModule):
    """
    Lin2022 Module for Antibody (Aducanumab) Pharmacokinetics.
    """
    def build(self):
        Species_param = self.config.get('Species')
        No_Isotope_SpeciesList = self.config.get('No_Isotope_SpeciesList', [])
        Isotopes = [''] if Species_param in No_Isotope_SpeciesList else self.model.isotopes

        kinfusion = "kinfusion" # Defined dynamically
        kclearmAb = "kclearmAb"
        k12mAb = "k12mAb"
        k21mAb = "k21mAb"
        
        k13mAb = "k13mAb"
        k31mAb = "k31mAb"
        k14mAb = "k14mAb"
        k41mAb = "k41mAb"
        k43mAb = "k43mAb"

        for Isotope in Isotopes:
            Isotope_str = f"_{Isotope}_" if Isotope else "_"

            mAb_IV = f"Antibody{Isotope_str}IV"
            mAb_Plasma = f"Antibody{Isotope_str}Plasma"
            mAb_Peripheral = f"Antibody{Isotope_str}Peripheral"
            mAb_CSF = f"Antibody{Isotope_str}CSF"
            mAb_BrainISF = f"Antibody{Isotope_str}BrainISF"

            self.add_reaction(name=f"Flow_IV_mAb{Isotope_str}", reactants=mAb_IV, products=mAb_Plasma, rate_type="MA", rate_eqtn=kinfusion)
            self.add_reaction(name=f"Clear_mAb{Isotope_str}Plasma", reactants=mAb_Plasma, products="", rate_type="MA", rate_eqtn=kclearmAb)
            self.add_reaction(name=f"Flow_12_mAb{Isotope_str}", reactants=mAb_Plasma, products=mAb_Peripheral, rate_type="MA", rate_eqtn=k12mAb)
            self.add_reaction(name=f"Flow_21_mAb{Isotope_str}", reactants=mAb_Peripheral, products=mAb_Plasma, rate_type="MA", rate_eqtn=k21mAb)

            self.add_reaction(name=f"Flow_13_mAb{Isotope_str}", reactants=mAb_Plasma, products=mAb_CSF, rate_type="MA", rate_eqtn=k13mAb)
            self.add_reaction(name=f"Flow_31_mAb{Isotope_str}", reactants=mAb_CSF, products=mAb_Plasma, rate_type="MA", rate_eqtn=k31mAb)
            self.add_reaction(name=f"Flow_14_mAb{Isotope_str}", reactants=mAb_Plasma, products=mAb_BrainISF, rate_type="MA", rate_eqtn=k14mAb)
            self.add_reaction(name=f"Flow_41_mAb{Isotope_str}", reactants=mAb_BrainISF, products=mAb_Plasma, rate_type="MA", rate_eqtn=k41mAb)
            self.add_reaction(name=f"Flow_43_mAb{Isotope_str}", reactants=mAb_BrainISF, products=mAb_CSF, rate_type="MA", rate_eqtn=k43mAb)

def Antibody_PK_Lin2022(model, Species, No_Isotope_SpeciesList=None):
    if No_Isotope_SpeciesList is None:
        No_Isotope_SpeciesList = ['Antibody']
    config = {
        'Species': Species,
        'No_Isotope_SpeciesList': No_Isotope_SpeciesList
    }
    module = Antibody_PK_Lin2022Module(model, **config)
    return model

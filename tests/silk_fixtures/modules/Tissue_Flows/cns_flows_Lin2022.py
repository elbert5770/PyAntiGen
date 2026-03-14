import sys
import os

from framework.module_base import PyAntiGenModule

class CNS_flows_Lin2022Module(PyAntiGenModule):
    """
    Lin2022 Module for transport flows across Plasma, CSF, and BrainISF.
    """
    def build(self):
        Species_param = self.config.get('Species')
        No_Isotope_SpeciesList = self.config.get('No_Isotope_SpeciesList', [])
        Isotopes = [''] if Species_param in No_Isotope_SpeciesList else self.model.isotopes

        # Flow rates from Table S2 based on Species/Compartments
        # k13: Plasma -> CSF
        # k31: CSF -> Plasma
        # k14: Plasma -> BrainISF
        # k41: BrainISF -> Plasma
        # k43: BrainISF -> CSF

        rates = {
            'Abeta': {
                'k13': "k13Abeta",
                'k31': "k31Abeta",
                'k14': "k14Abeta",
                'k41': "k41Abeta",
                'k43': "k43Abeta",
            },
            'Aolig': {
                'k13': "k13Aolig",
                'k31': "k31Aolig",
                'k14': "k14Aolig",
                'k41': "k41Aolig",
                'k43': "k43Aolig",
            },
            'BACE': { # Soluble BACE
                'k13': "k13BACE",
                'k31': "k31BACE",
                'k14': "k14BACE",
                'k41': "k41BACE",
                'k43': "k43BACE",
            }
        }

        for Isotope in Isotopes:
            Isotope_str = f"_{Isotope}_" if Isotope else "_"

            # Define Species variables for compartments
            Abeta_Plasma = f"{Species_param}{Isotope_str}Plasma"
            Abeta_CSF = f"{Species_param}{Isotope_str}CSF"
            Abeta_BrainISF = f"{Species_param}{Isotope_str}BrainISF"

            Aolig_Plasma = f"{Species_param}_Oligomer{Isotope_str}Plasma"
            Aolig_CSF = f"{Species_param}_Oligomer{Isotope_str}CSF"
            Aolig_BrainISF = f"{Species_param}_Oligomer{Isotope_str}BrainISF"

            BACEs_Plasma = f"SolubleBACE{Isotope_str}Plasma"
            BACEs_CSF = f"SolubleBACE{Isotope_str}CSF"
            BACEs_BrainISF = f"SolubleBACE{Isotope_str}BrainISF"

            # Abeta Transport
            self.add_reaction(name=f"Flow_13_Abeta{Isotope_str}", reactants=Abeta_Plasma, products=Abeta_CSF, rate_type="MA", rate_eqtn=rates['Abeta']['k13'])
            self.add_reaction(name=f"Flow_31_Abeta{Isotope_str}", reactants=Abeta_CSF, products=Abeta_Plasma, rate_type="MA", rate_eqtn=rates['Abeta']['k31'])
            self.add_reaction(name=f"Flow_14_Abeta{Isotope_str}", reactants=Abeta_Plasma, products=Abeta_BrainISF, rate_type="MA", rate_eqtn=rates['Abeta']['k14'])
            self.add_reaction(name=f"Flow_41_Abeta{Isotope_str}", reactants=Abeta_BrainISF, products=Abeta_Plasma, rate_type="MA", rate_eqtn=rates['Abeta']['k41'])
            self.add_reaction(name=f"Flow_43_Abeta{Isotope_str}", reactants=Abeta_BrainISF, products=Abeta_CSF, rate_type="MA", rate_eqtn=rates['Abeta']['k43'])

            # Aolig Transport
            self.add_reaction(name=f"Flow_13_Aolig{Isotope_str}", reactants=Aolig_Plasma, products=Aolig_CSF, rate_type="MA", rate_eqtn=rates['Aolig']['k13'])
            self.add_reaction(name=f"Flow_31_Aolig{Isotope_str}", reactants=Aolig_CSF, products=Aolig_Plasma, rate_type="MA", rate_eqtn=rates['Aolig']['k31'])
            self.add_reaction(name=f"Flow_14_Aolig{Isotope_str}", reactants=Aolig_Plasma, products=Aolig_BrainISF, rate_type="MA", rate_eqtn=rates['Aolig']['k14'])
            self.add_reaction(name=f"Flow_41_Aolig{Isotope_str}", reactants=Aolig_BrainISF, products=Aolig_Plasma, rate_type="MA", rate_eqtn=rates['Aolig']['k41'])
            self.add_reaction(name=f"Flow_43_Aolig{Isotope_str}", reactants=Aolig_BrainISF, products=Aolig_CSF, rate_type="MA", rate_eqtn=rates['Aolig']['k43'])

            # BACEs Transport
            self.add_reaction(name=f"Flow_13_BACEs{Isotope_str}", reactants=BACEs_Plasma, products=BACEs_CSF, rate_type="MA", rate_eqtn=rates['BACE']['k13'])
            self.add_reaction(name=f"Flow_31_BACEs{Isotope_str}", reactants=BACEs_CSF, products=BACEs_Plasma, rate_type="MA", rate_eqtn=rates['BACE']['k31'])
            self.add_reaction(name=f"Flow_14_BACEs{Isotope_str}", reactants=BACEs_Plasma, products=BACEs_BrainISF, rate_type="MA", rate_eqtn=rates['BACE']['k14'])
            self.add_reaction(name=f"Flow_41_BACEs{Isotope_str}", reactants=BACEs_BrainISF, products=BACEs_Plasma, rate_type="MA", rate_eqtn=rates['BACE']['k41'])
            self.add_reaction(name=f"Flow_43_BACEs{Isotope_str}", reactants=BACEs_BrainISF, products=BACEs_CSF, rate_type="MA", rate_eqtn=rates['BACE']['k43'])


def cns_flows_Lin2022(model, Species, No_Isotope_SpeciesList=None):
    if No_Isotope_SpeciesList is None:
        No_Isotope_SpeciesList = ['Antibody']
    config = {
        'Species': Species,
        'No_Isotope_SpeciesList': No_Isotope_SpeciesList
    }
    module = CNS_flows_Lin2022Module(model, **config)
    return model

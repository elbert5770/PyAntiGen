import sys
import os

from framework.module_base import PyAntiGenModule

class Abeta_aggregation_Lin2022Module(PyAntiGenModule):
    """
    Lin2022 Module for Abeta aggregation (monomer <-> oligomer <-> plaque).
    """
    def build(self):
        Species_param = self.config.get('Species')
        No_Isotope_SpeciesList = self.config.get('No_Isotope_SpeciesList', [])
        Isotopes = [''] if Species_param in No_Isotope_SpeciesList else self.model.isotopes

        kM2G = "kM2G"
        
        kG2M_params = {
            'Plasma': "kG2M_plasma",
            'CSF': "kG2M_csf",
            'BrainISF': "kG2M_bisf"
        }

        kG2P = "kG2P"
        kP2G_bisf = "kP2G_bisf"

        clearance_rates = {
            'Plasma': {
                'Abeta': "kclearAbeta_plasma",
                'Aolig': "kclearAolig_plasma"
            },
            'BrainISF': {
                'Abeta': "kclearAbeta_bisf",
                'Aolig': "kclearAolig_bisf",
                'Aplaq': "kclearAplaque_bisf"
            }
        }

        # Handle Plasma and CSF (Monomer <-> Oligomer)
        for comp in ['Plasma', 'CSF', 'BrainISF']:
            for Isotope in Isotopes:
                Isotope_str = f"_{Isotope}_" if Isotope else "_"

                Abeta = f"{Species_param}{Isotope_str}{comp}"
                Aolig = f"{Species_param}_Oligomer{Isotope_str}{comp}"

                # 1. Monomer <-> Oligomer
                self.add_reaction(
                    name=f"Agg_M2G_G2M_Abeta{Isotope_str}{comp}",
                    reactants=Abeta,
                    products=Aolig,
                    rate_type="RMA",
                    rate_eqtn=f"[{kM2G}, {kG2M_params[comp]}]"
                )

                # 2. Clearance of Monomer and Oligomer (Only in Plasma and BrainISF based on Table S2)
                if comp in clearance_rates:
                    self.add_reaction(
                        name=f"Clear_Abeta{Isotope_str}{comp}",
                        reactants=Abeta,
                        products="",
                        rate_type="MA",
                        rate_eqtn=clearance_rates[comp]['Abeta']
                    )
                    self.add_reaction(
                        name=f"Clear_Aolig{Isotope_str}{comp}",
                        reactants=Aolig,
                        products="",
                        rate_type="MA",
                        rate_eqtn=clearance_rates[comp]['Aolig']
                    )

        # Handle BrainISF specific reactions (Oligomer <-> Plaque)
        comp = 'BrainISF'
        for Isotope in Isotopes:
            Isotope_str = f"_{Isotope}_" if Isotope else "_"

            Aolig = f"{Species_param}_Oligomer{Isotope_str}{comp}"
            Aplaq = f"{Species_param}_Plaque{Isotope_str}{comp}"

            # 3. Oligomer <-> Plaque
            self.add_reaction(
                name=f"Agg_G2P_P2G_Abeta{Isotope_str}{comp}",
                reactants=Aolig,
                products=Aplaq,
                rate_type="RMA",
                rate_eqtn=f"[{kG2P}, {kP2G_bisf}]"
            )

            # 4. Clearance of Plaque
            self.add_reaction(
                name=f"Clear_Aplaq{Isotope_str}{comp}",
                reactants=Aplaq,
                products="",
                rate_type="MA",
                rate_eqtn=clearance_rates[comp]['Aplaq']
            )

def Abeta_aggregation_Lin2022(model, Species, No_Isotope_SpeciesList=None):
    if No_Isotope_SpeciesList is None:
        No_Isotope_SpeciesList = ['Antibody']
    config = {
        'Species': Species,
        'No_Isotope_SpeciesList': No_Isotope_SpeciesList
    }
    module = Abeta_aggregation_Lin2022Module(model, **config)
    return model

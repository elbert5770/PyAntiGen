import sys
import os

from framework.module_base import PyAntiGenModule

class APP_reactions_Lin2022Module(PyAntiGenModule):
    """
    Lin2022 Module for APP processing reactions across Plasma and BrainISF.
    """
    def build(self):
        Species_param = self.config.get('Species')
        No_Isotope_SpeciesList = self.config.get('No_Isotope_SpeciesList', [])
        Isotopes = [''] if Species_param in No_Isotope_SpeciesList else self.model.isotopes

        # These match the constants from Table S2
        konPP = "konPP"
        koffBACE = "koffBACE"
        kcatBACE = "kcatBACE"
        koffGamma = "koffGamma"
        kcatGamma = "kcatGamma"

        # Parameters that vary by compartment
        params = {
            'Plasma': {
                'ksynthAPP': "ksynthAPP_plasma",
                'kclearAPP': "kclearAPP",
                'ksynthBACE': "ksynthBACE_plasma",
                'kclearBACE': "kclearBACE",
                'ksynthGamma': "ksynthGamma_plasma",
                'kclearGamma': "kclearGamma",
                'kcleaveBACE': "kcleave",
            },
            'BrainISF': {
                'ksynthAPP': "ksynthAPP_bisf",
                'kclearAPP': "kclearAPP",       # Using plasma value as it was same in table
                'ksynthBACE': "ksynthBACE_bisf",
                'kclearBACE': "kclearBACE",      # Using plasma value as it was same in table
                'ksynthGamma': "ksynthGamma_bisf",
                'kclearGamma': "kclearGamma",     # Using plasma value as it was same in table
                'kcleaveBACE': "kcleave",     # Using plasma value as it was same in table
            }
        }

        for comp in ['Plasma', 'BrainISF']:
            for Isotope in Isotopes:
                Isotope_str = f"_{Isotope}_" if Isotope else "_"

                # Species references
                APP = f"APP{Isotope_str}{comp}"
                BACE = f"BACE{Isotope_str}{comp}"
                SolubleBACE = f"SolubleBACE{Isotope_str}{comp}"
                Gamma = f"GammaSecretase{Isotope_str}{comp}"
                APP_BACE = f"APP__BACE{Isotope_str}{comp}"
                CTFb = f"CTFbeta{Isotope_str}{comp}"
                CTFb_Gamma = f"CTFbeta__GammaSecretase{Isotope_str}{comp}"
                Abeta = f"{Species_param}{Isotope_str}{comp}"

                # 1. APP Synthesis & Degradation
                self.add_reaction(
                    name=f"Synth_APP{Isotope_str}{comp}",
                    reactants="",
                    products=APP,
                    rate_type="custom",
                    rate_eqtn=params[comp]['ksynthAPP']
                )
                self.add_reaction(
                    name=f"Clear_APP{Isotope_str}{comp}",
                    reactants=APP,
                    products="",
                    rate_type="MA",
                    rate_eqtn=params[comp]['kclearAPP']
                )

                # 2. BACE Synthesis & Degradation
                self.add_reaction(
                    name=f"Synth_BACE{Isotope_str}{comp}",
                    reactants="",
                    products=BACE,
                    rate_type="custom",
                    rate_eqtn=params[comp]['ksynthBACE']
                )
                self.add_reaction(
                    name=f"Clear_BACE{Isotope_str}{comp}",
                    reactants=BACE,
                    products="",
                    rate_type="MA",
                    rate_eqtn=params[comp]['kclearBACE']
                )

                # 3. GammaSecretase Synthesis & Degradation
                self.add_reaction(
                    name=f"Synth_GammaSecretase{Isotope_str}{comp}",
                    reactants="",
                    products=Gamma,
                    rate_type="custom",
                    rate_eqtn=params[comp]['ksynthGamma']
                )
                self.add_reaction(
                    name=f"Clear_GammaSecretase{Isotope_str}{comp}",
                    reactants=Gamma,
                    products="",
                    rate_type="MA",
                    rate_eqtn=params[comp]['kclearGamma']
                )

                # 4. Cleavage of BACE to SolubleBACE
                self.add_reaction(
                    name=f"Cleave_BACE{Isotope_str}{comp}",
                    reactants=BACE,
                    products=SolubleBACE,
                    rate_type="MA",
                    rate_eqtn=params[comp]['kcleaveBACE']
                )

                # 5. APP + BACE <-> APP_BACE
                self.add_reaction(
                    name=f"Bind_APP_BACE{Isotope_str}{comp}",
                    reactants=f"[{APP}, {BACE}]",
                    products=APP_BACE,
                    rate_type="RMA",
                    rate_eqtn=f"[{konPP}, {koffBACE}]"
                )

                # 6. Cleavage of APP_BACE to CTFbeta
                self.add_reaction(
                    name=f"Cat_APP_BACE{Isotope_str}{comp}",
                    reactants=APP_BACE,
                    products=f"[{CTFb}, {BACE}]",
                    rate_type="MA",
                    rate_eqtn=kcatBACE
                )

                # 7. CTFbeta + GammaSecretase <-> CTFb_Gamma
                self.add_reaction(
                    name=f"Bind_CTFb_Gamma{Isotope_str}{comp}",
                    reactants=f"[{CTFb}, {Gamma}]",
                    products=CTFb_Gamma,
                    rate_type="RMA",
                    rate_eqtn=f"[{konPP}, {koffGamma}]"
                )

                # 8. Catalysis of CTFb_Gamma to Abeta
                self.add_reaction(
                    name=f"Cat_CTFb_Gamma{Isotope_str}{comp}",
                    reactants=CTFb_Gamma,
                    products=f"[{Abeta}, {Gamma}]",
                    rate_type="MA",
                    rate_eqtn=kcatGamma
                )

def APP_reactions_Lin2022(model, Species, No_Isotope_SpeciesList=None):
    if No_Isotope_SpeciesList is None:
        No_Isotope_SpeciesList = ['Antibody']
    config = {
        'Species': Species,
        'No_Isotope_SpeciesList': No_Isotope_SpeciesList
    }
    module = APP_reactions_Lin2022Module(model, **config)
    return model

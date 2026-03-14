import sys
import os

from framework.module_base import PyAntiGenModule

class Abeta_ADCP_clearance_Lin2022Module(PyAntiGenModule):
    """
    Lin2022 Module for ADCP clearance via FcR binding.
    """
    def build(self):
        Species_param = self.config.get('Species')
        No_Isotope_SpeciesList = self.config.get('No_Isotope_SpeciesList', [])
        Isotopes = [''] if Species_param in No_Isotope_SpeciesList else self.model.isotopes

        # Parameters
        kclearFcR = "kclearFcR"
        ksynthFcR_bisf = "ksynthFcR_bisf"
        konPF = "konPF"
        koffPF = "koffPF"
        kcatADCP = "kcatADCP"

        comp = 'BrainISF'
        # Assume FcR is in No_Isotope_SpeciesList
        FcR = f"FcR_{comp}" 
        
        # Setup FcR synthesis and degradation outside the isotopes loop assuming it's unlabelled
        self.add_reaction(name=f"Synth_FcR_{comp}", reactants="", products=f"FcR_{comp}", rate_type="custom", rate_eqtn=ksynthFcR_bisf)
        self.add_reaction(name=f"Clear_FcR_{comp}", reactants=f"FcR_{comp}", products="", rate_type="MA", rate_eqtn=kclearFcR)


        for Isotope in Isotopes:
            Isotope_str = f"_{Isotope}_" if Isotope else "_"

            mAb_Aolig = f"{Species_param}_Oligomer__Antibody{Isotope_str}{comp}"
            mAb_Aplaq = f"{Species_param}_Plaque__Antibody{Isotope_str}{comp}"

            # FcR assumes no explicit isotope inside its brackets here to avoid complex combination arrays.
            # Using {Species_param}_Oligomer__Antibody__FcR as convention, assuming FcR stays unlabelled.
            mAb_Aolig_FcR = f"{Species_param}_Oligomer__Antibody__FcR{Isotope_str}{comp}"
            mAb_Aplaq_FcR = f"{Species_param}_Plaque__Antibody__FcR{Isotope_str}{comp}"

            self.add_reaction(name=f"Bind_mAb_Aolig_FcR{Isotope_str}{comp}", reactants=f"[{mAb_Aolig}, FcR_{comp}]", products=mAb_Aolig_FcR, rate_type="RMA", rate_eqtn=f"[{konPF}, {koffPF}]")
            self.add_reaction(name=f"Catalysis_ADCP_Aolig{Isotope_str}{comp}", reactants=mAb_Aolig_FcR, products=f"FcR_{comp}", rate_type="MA", rate_eqtn=kcatADCP)

            self.add_reaction(name=f"Bind_mAb_Aplaq_FcR{Isotope_str}{comp}", reactants=f"[{mAb_Aplaq}, FcR_{comp}]", products=mAb_Aplaq_FcR, rate_type="RMA", rate_eqtn=f"[{konPF}, {koffPF}]")
            self.add_reaction(name=f"Catalysis_ADCP_Aplaq{Isotope_str}{comp}", reactants=mAb_Aplaq_FcR, products=f"FcR_{comp}", rate_type="MA", rate_eqtn=kcatADCP)


def Abeta_ADCP_clearance_Lin2022(model, Species, No_Isotope_SpeciesList=None):
    if No_Isotope_SpeciesList is None:
        No_Isotope_SpeciesList = ['Antibody', 'FcR']
    config = {
        'Species': Species,
        'No_Isotope_SpeciesList': No_Isotope_SpeciesList
    }
    module = Abeta_ADCP_clearance_Lin2022Module(model, **config)
    return model

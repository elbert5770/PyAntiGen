import sys
import os

from framework.module_base import PyAntiGenModule

class Abeta_antibody_binding_Lin2022Module(PyAntiGenModule):
    """
    Lin2022 Module for Antibody binding to Abeta species.
    """
    def build(self):
        Species_param = self.config.get('Species')
        No_Isotope_SpeciesList = self.config.get('No_Isotope_SpeciesList', [])
        
        Isotopes_mAb = [''] if 'Antibody' in No_Isotope_SpeciesList else self.model.isotopes
        Isotopes = [''] if Species_param in No_Isotope_SpeciesList else self.model.isotopes

        konPP_monomer = "konPP"
        kmAb0 = "kmAb0"

        konPP_oligomer = "konPP"
        kmAb1 = "kmAb1"

        konPD = "konPD"
        kmAb2 = "kmAb2"

        kclearmAb = "kclearmAb"

        k13mAb = "k13mAb"
        k31mAb = "k31mAb"
        k14mAb = "k14mAb"
        k14mix = "k14mix"
        k41mAb = "k41mAb"
        k41mix = "k41mix"
        k43mAb = "k43mAb"
        k43mix = "k43mix"
        
        # NOTE: Using a single isotope loop for simplicity. If Antibody and Abeta isotopes are meant to be decoupled,
        # one would need to specify how combinatorial labeling is handled. Based on SKILL.md rules and typical cases,
        # we will use the Abeta Isotope list as primary, matching the Antibody isotope correctly.

        for Isotope in Isotopes:
            Isotope_str = f"_{Isotope}_" if Isotope else "_"

            for comp in ['Plasma', 'CSF', 'BrainISF']:
                mAb = f"Antibody{Isotope_str}{comp}"
                Abeta = f"{Species_param}{Isotope_str}{comp}"
                Aolig = f"{Species_param}_Oligomer{Isotope_str}{comp}"

                mAb_Abeta = f"{Species_param}__Antibody{Isotope_str}{comp}"
                mAb_Aolig = f"{Species_param}_Oligomer__Antibody{Isotope_str}{comp}"

                self.add_reaction(name=f"Bind_mAb_Abeta{Isotope_str}{comp}", reactants=f"[{mAb}, {Abeta}]", products=mAb_Abeta, rate_type="RMA", rate_eqtn=f"[{konPP_monomer}, {kmAb0}]")
                self.add_reaction(name=f"Bind_mAb_Aolig{Isotope_str}{comp}", reactants=f"[{mAb}, {Aolig}]", products=mAb_Aolig, rate_type="RMA", rate_eqtn=f"[{konPP_oligomer}, {kmAb1}]")

                # MAb-Abeta Transport and Clearance mimicking Antibody PK
                if comp == 'Plasma':
                    self.add_reaction(name=f"Clear_mAb_Abeta{Isotope_str}Plasma", reactants=mAb_Abeta, products="", rate_type="MA", rate_eqtn=kclearmAb)
                    self.add_reaction(name=f"Flow_13_mAb_Abeta{Isotope_str}", reactants=mAb_Abeta, products=f"{Species_param}__Antibody{Isotope_str}CSF", rate_type="MA", rate_eqtn=k13mAb)
                    self.add_reaction(name=f"Flow_14_mAb_Abeta{Isotope_str}", reactants=mAb_Abeta, products=f"{Species_param}__Antibody{Isotope_str}BrainISF", rate_type="MA", rate_eqtn=k14mAb)

                    self.add_reaction(name=f"Clear_mAb_Aolig{Isotope_str}Plasma", reactants=mAb_Aolig, products="", rate_type="MA", rate_eqtn=kclearmAb)
                    self.add_reaction(name=f"Flow_13_mAb_Aolig{Isotope_str}", reactants=mAb_Aolig, products=f"{Species_param}_Oligomer__Antibody{Isotope_str}CSF", rate_type="MA", rate_eqtn=k13mAb)
                    self.add_reaction(name=f"Flow_14_mAb_Aolig{Isotope_str}", reactants=mAb_Aolig, products=f"{Species_param}_Oligomer__Antibody{Isotope_str}BrainISF", rate_type="MA", rate_eqtn=k14mix)

                if comp == 'CSF':
                    self.add_reaction(name=f"Flow_31_mAb_Abeta{Isotope_str}", reactants=mAb_Abeta, products=f"{Species_param}__Antibody{Isotope_str}Plasma", rate_type="MA", rate_eqtn=k31mAb)
                    self.add_reaction(name=f"Flow_31_mAb_Aolig{Isotope_str}", reactants=mAb_Aolig, products=f"{Species_param}_Oligomer__Antibody{Isotope_str}Plasma", rate_type="MA", rate_eqtn=k31mAb)

                if comp == 'BrainISF':
                    self.add_reaction(name=f"Flow_41_mAb_Abeta{Isotope_str}", reactants=mAb_Abeta, products=f"{Species_param}__Antibody{Isotope_str}Plasma", rate_type="MA", rate_eqtn=k41mAb)
                    self.add_reaction(name=f"Flow_43_mAb_Abeta{Isotope_str}", reactants=mAb_Abeta, products=f"{Species_param}__Antibody{Isotope_str}CSF", rate_type="MA", rate_eqtn=k43mAb)
                    self.add_reaction(name=f"Flow_41_mAb_Aolig{Isotope_str}", reactants=mAb_Aolig, products=f"{Species_param}_Oligomer__Antibody{Isotope_str}Plasma", rate_type="MA", rate_eqtn=k41mix)
                    self.add_reaction(name=f"Flow_43_mAb_Aolig{Isotope_str}", reactants=mAb_Aolig, products=f"{Species_param}_Oligomer__Antibody{Isotope_str}CSF", rate_type="MA", rate_eqtn=k43mix)


            # Abeta Plaque binding (BrainISF specific)
            comp = 'BrainISF'
            mAb = f"Antibody{Isotope_str}{comp}"
            Aplaq = f"{Species_param}_Plaque{Isotope_str}{comp}"
            mAb_Aplaq = f"{Species_param}_Plaque__Antibody{Isotope_str}{comp}"
            self.add_reaction(name=f"Bind_mAb_Aplaq{Isotope_str}{comp}", reactants=f"[{mAb}, {Aplaq}]", products=mAb_Aplaq, rate_type="RMA", rate_eqtn=f"[{konPD}, {kmAb2}]")

def Abeta_antibody_binding_Lin2022(model, Species, No_Isotope_SpeciesList=None):
    if No_Isotope_SpeciesList is None:
        No_Isotope_SpeciesList = ['Antibody']
    config = {
        'Species': Species,
        'No_Isotope_SpeciesList': No_Isotope_SpeciesList
    }
    module = Abeta_antibody_binding_Lin2022Module(model, **config)
    return model

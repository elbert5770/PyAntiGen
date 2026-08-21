"""
Model builder. Outputs go to antimony_models/{MODEL_NAME}/ and generated/{MODEL_NAME}/.
"""
import os
import sys

import AntiGen_paths


from framework.pyantigen import PyAntiGen
from antimony_modules.Basic.ma_reaction import BasicMAReaction, BasicChainReaction

def generate_antimony_model(Isotopes=['']):
    MODEL_NAME = AntiGen_paths.MODEL_NAME
    model = PyAntiGen(name=MODEL_NAME, isotopes=Isotopes)
    BasicMAReaction(model)
    # Second chain step B -> C. Inert for Example1-3 (k_B_to_C defaults to 0);
    # fit by Example4/Example5 to create flip-flop bimodality.
    BasicChainReaction(model)

    print(f"Reactions generated: {model.counter}")
    print(f"Rules generated: {len(model.rules)}")

    model.generate(__file__, model_name=MODEL_NAME)
    
    print("\nModel generated successfully.")
    print("Next steps:")
    print(f"  1. Optionally edit parameters in antimony_models/{MODEL_NAME}/{MODEL_NAME}_parameters.csv")
    print(f"  2. From Projects/{MODEL_NAME}/, run: python Model_run.py")


def update_antimony_model():
    Isotopes = ['']
    generate_antimony_model(Isotopes)

if __name__ == "__main__":
    update_antimony_model()

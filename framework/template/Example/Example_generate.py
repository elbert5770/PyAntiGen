"""
Example model builder. Outputs go to antimony_models/Example/ and generated/Example/.
"""
import os
import sys

import paths


from framework.pyantigen import PyAntiGen
from modules.Basic.ma_reaction import BasicMAReaction

def generate_antimony_model(Isotopes=['']):
    MODEL_NAME = paths.MODEL_NAME
    model = PyAntiGen(name=MODEL_NAME, isotopes=Isotopes)
    BasicMAReaction(model)

    print(f"Reactions generated: {model.counter}")
    print(f"Rules generated: {len(model.rules)}")

    model.generate(__file__, model_name=MODEL_NAME)
    
    print("\nModel generated successfully.")
    print("Next steps:")
    print(f"  1. Optionally edit parameters in antimony_models/{MODEL_NAME}/{MODEL_NAME}_parameters.csv")
    print(f"  2. From scripts/{MODEL_NAME}/, run: python {MODEL_NAME}_run.py")

if __name__ == "__main__":
    Isotopes = ['']
    generate_antimony_model(Isotopes)

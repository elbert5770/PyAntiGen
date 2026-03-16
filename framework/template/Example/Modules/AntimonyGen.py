import os
import tellurium as te

# from Model_Modules.Model_Events import generate_silk_events_from_data

from framework.antimony_utils import load_antimony_files, archive_antimony_snapshot

def AntimonyGen(MODEL_NAME):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # From Model_Modules: ../../.. = repo root (contains data/, results/, generated/, antimony_models/)
    repo_root = os.path.normpath(os.path.join(current_dir, "..", "..", ".."))
    data_path = os.path.join(repo_root, "data")
    plot_path = os.path.normpath(os.path.join(repo_root, "results", MODEL_NAME))


    
    
    event_block = ''
    

    model_text = load_antimony_files(MODEL_NAME, repo_root)
 
    if not model_text.strip():
        raise RuntimeError(
            f"No model content loaded. Generate the model first: python {MODEL_NAME}_generate.py"
        )

    
    
    return model_text, data_path, plot_path, repo_root

def TelluriumGen(model_text, MODEL_NAME, repo_root):
    print("Loading model into Tellurium...")
    try:
        r = te.loada(model_text)
    except Exception as e:
        raise RuntimeError(f"Error loading model: {e}") from e

    sbml_content = r.getSBML()
    archive_dir = archive_antimony_snapshot(MODEL_NAME, repo_root, sbml_content=sbml_content)
    print(f"Archive and SBML written to: {archive_dir}")
    return r

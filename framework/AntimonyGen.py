import os


# from Model_Modules.Model_Events import generate_silk_events_from_data

from framework.antimony_utils import load_antimony_files

def AntimonyGen(MODEL_NAME, repo_root=None):
    if repo_root is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Fallback to current directory if not provided
        repo_root = current_dir
    data_path = os.path.join(repo_root, "data")
    plot_path = os.path.normpath(os.path.join(repo_root, "results", MODEL_NAME))
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)

    
    
    event_block = ''
    

    model_text = load_antimony_files(MODEL_NAME, repo_root)
 
    if not model_text.strip():
        raise RuntimeError(
            f"No model content loaded. Generate the model first: python {MODEL_NAME}_generate.py"
        )

    events_path = os.path.join(repo_root, "generated", MODEL_NAME, MODEL_NAME + "_events.txt")
    
    models_path = os.path.join(repo_root, "antimony_models", MODEL_NAME, f"{MODEL_NAME}_InitialConditions.csv")

    project_root = os.path.join(repo_root, "Projects", MODEL_NAME)
    
    paths = {
        "MODEL_NAME": MODEL_NAME,
        "data_path": data_path,
        "plot_path": plot_path,
        "repo_root": repo_root,
        "events_path": events_path,
        "models_path": models_path,
        "project_root": project_root
    }
    
    return model_text,paths



import tellurium as te
from framework.antimony_utils import archive_antimony_snapshot

def TelluriumGen(model_text, paths, settings=None ):
    print("Loading model into Tellurium...")
    try:
        r = te.loada(model_text)
    except Exception as e:
        raise RuntimeError(f"Error loading model: {e}") from e

    if settings is not None:
        if settings["save_SBML?"]:
            sbml_content = r.getSBML()
            archive_dir = archive_antimony_snapshot(paths["MODEL_NAME"], paths["repo_root"], sbml_content=sbml_content)
            print(f"Archive and SBML written to: {archive_dir}")
    return r
import sys
from pathlib import Path

# Use location: import Modules from the same folder as this script (model folder)
_project_dir = Path(__file__).resolve().parent
if str(_project_dir) not in sys.path:
    sys.path.insert(0, str(_project_dir))

MODEL_NAME = _project_dir.name

# Add PyAntiGen root to sys.path if running from within the framework template
if _project_dir.parent.name == "template":
    _pyantigen_root = _project_dir.parents[2]
    if str(_pyantigen_root) not in sys.path:
        sys.path.insert(0, str(_pyantigen_root))

if _project_dir.name == "scripts":
    REPO_ROOT = str(_project_dir.parent)
else:
    REPO_ROOT = str(_project_dir.parents[1])

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

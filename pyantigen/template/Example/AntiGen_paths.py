"""Project paths, and a check that PyAntiGen 2 is the one being used.

MODEL_NAME is this folder's name. REPO_ROOT is two levels up and holds
antimony_models/, data/, generated/ and results/.

The Engine is not part of a project. It is ``pyantigen.engine`` from the
installed PyAntiGen 2 package -- one copy, shared by every project -- so this
module refuses to go on if:

  * ``pyantigen.engine`` cannot be imported (PyAntiGen 1.x installs only the
    ``framework`` package, or the wrong environment is active), or
  * an ``Engine/`` folder has appeared beside this file, which would put a
    second, diverging copy of the Engine on the import path.
"""
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

if (_project_dir / "Engine").exists():
    raise RuntimeError(
        f"{_project_dir / 'Engine'} exists. Projects use pyantigen.engine from "
        "the installed PyAntiGen 2 package; delete the local Engine folder.")

try:
    import pyantigen
    import pyantigen.engine  # noqa: F401
except ImportError as exc:
    raise ImportError(
        "PyAntiGen 2 (the 'pyantigen' package with pyantigen.engine) is not "
        "installed in this Python. Activate the environment you installed it "
        "into, or run 'pip install \"pyantigen>=2\"'. See the README.") from exc

PYANTIGEN_VERSION = getattr(pyantigen, "__version__", "unknown")

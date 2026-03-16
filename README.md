# PyAntiGen

PyAntiGen is a declarative, object-oriented framework for generating compartmental biological models in Antimony format. It is designed to abstract away the repetitive boilerplate of defining reactions and compartments manually, allowing researchers to build complex, scalable models using clean Python syntax.

## Features

- **Object-Oriented Modules:** Encapsulate tissues, flows, synthesis, and excretion into reusable Python classes.
- **Dynamic Registration:** Automatically binds reactions, compartments, and species to the global model state when a module is instantiated.
- **Project Scaffolding:** Includes a CLI command to instantly spin up new modeling projects with all necessary directories.
- **Isotope Tracking:** Natively supports tracking labeled isotopes and generating corresponding parallel reactions.

## Installation

You can install PyAntiGen globally into your preferred Python environment by cloning this repository and running pip:

```bash
git clone https://github.com/elbert5770/PyAntiGen.git
cd PyAntiGen
pip install -e .
```

## Quick Start: Creating a New Model

Because PyAntiGen is installed as a system-level Python package, you don't need a copy of the framework files in your working directory. 

To start a brand new modeling workspace, just open a terminal and navigate to a folder where you want it to live (not PyAntiGen) and run:

```bash
pyantigen-create MyNewModel
```

This will automatically scaffold the following project directory:
```text
MyNewModel/
├── .agents/
│   └── skills/          (agent skills, e.g. module generation, ODE conversion)
├── scripts/
│   ├── Example/         (full example: generate, run, optimize + Modules/)
│   │   ├── Example_generate.py
│   │   ├── Example_run.py
│   │   ├── Example_optimize.py
│   │   └── Modules/     (Data, AntimonyGen, Plots, Simulate, Optimize, Experiment, Events)
│   └── MyNewModel/      (same structure, Modules/ left empty for your code)
│       ├── MyNewModel_generate.py
│       ├── MyNewModel_run.py
│       ├── MyNewModel_optimize.py
│       └── Modules/     (Data, AntimonyGen, Plots, Simulate, Optimize, Experiment, Events)
├── modules/
│   └── __init__.py      (plus Basic/ for the example)
├── data/                (Example experiment CSVs copied for the example)
├── antimony_models/
│   └── Example/         (Example_parameters.csv, Example_InitialConditions.csv, etc.)
├── generated/
│   └── Example/         (reaction dict, rules, etc. after generate)
├── results/
│   └── Example/         (plots from Example_run.py)
├── SBML_models/
└── pyantigen_settings.json   (e.g. archive_with_timestamp: false)
```

All model-specific files are grouped under folders named by `MODEL_NAME` (e.g. `Example` or your project name). From `MyNewModel/scripts/Example/` run:

```bash
python Example_generate.py
```

This generates the model and writes outputs to `antimony_models/Example/` and `generated/Example/`. Edit parameters if desired in `antimony_models/Example/Example_parameters.csv`, then run:

```bash
python Example_run.py
```

For parameter fitting against data, run:

```bash
python Example_optimize.py
```

Your own model lives under `scripts/MyNewModel/` with the same files as the Example. Modify the code for your model in `scripts/MyNewModel/Modules/` and `scripts/MyNewModel/MyNewModel_generate.py`, `scripts/MyNewModel/MyNewModel_run.py`, and `scripts/MyNewModel/MyNewModel_optimize.py'. Your problem will also require new modules in folder 'modules'.

Keeping the generation and simulation steps separate gives you time to adjust parameters and inspect the generated files before running.

### Running from an IDE (Cursor / VS Code)

The **Play** button uses whichever Python interpreter is currently selected. If your environment (conda/venv) isn’t loaded, the run may fail with import or path errors.

1. **Select the correct interpreter**: `Ctrl+Shift+P` (or `Cmd+Shift+P` on macOS) → **Python: Select Interpreter** → choose the environment where you ran `pip install -e .` (e.g. your conda or venv).
2. **Run from project root**: Open the *project* folder (e.g. `MyNewModel`) as the workspace. Use **Run and Debug** (or Play on `scripts/Example/Example_run.py`); the project root is resolved from the script location so `antimony_models/Example/`, `generated/Example/`, and `results/Example/` resolve correctly.

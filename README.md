# PyAntiGen

PyAntiGen is a declarative, object-oriented framework for generating compartmental biological models in Antimony format. It is designed to abstract away the repetitive boilerplate of defining reactions and compartments manually, allowing researchers to build complex, scalable models using clean Python syntax.

## Features

- **Object-Oriented Modules:** Encapsulate tissues, flows, synthesis, and excretion into reusable Python classes.
- **Dynamic Registration:** Automatically binds reactions, compartments, and species to the global model state when a module is instantiated.
- **Project Scaffolding:** Includes a CLI command to instantly spin up new modeling projects with all necessary directories.
- **Isotope Tracking:** Natively supports tracking labeled isotopes and generating corresponding parallel reactions.

## Installation
You can install PyAntiGen into your Python environment with:
```bash
pip install pyantigen
```

You can also install PyAntiGen globally into your preferred Python environment by cloning this repository and running pip:

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
├── Projects/
│   ├── Example/         (full example: generate, run + Modules/, Engine/)
│   │   ├── Model_generate.py
│   │   ├── Model_run.py
│   │   └── Modules/     (Data, AntimonyGen, Plots, Simulate, Optimize, Experiment, Events)
│   └── MyNewModel/      (same structure, Modules/ pre-populated from Example)
│       ├── Model_generate.py
│       ├── Model_run.py
│       └── Modules/     (Data, AntimonyGen, Plots, Simulate, Optimize, Experiment, Events)
├── antimony_modules/
│   └── __init__.py      (plus Basic/ for the example)
├── data/                (Example experiment CSVs copied for the example)
├── antimony_models/
│   └── Example/         (Example_parameters.csv, Example_InitialConditions.csv, etc.)
├── generated/
│   └── Example/         (reaction dict, rules, etc. after generate)
├── results/
│   └── Example/         (plots from Model_run.py)
├── SBML_models/
└── pyantigen_settings.json   (e.g. archive_with_timestamp: false)
```

Every project folder under `Projects/` uses the same two entry-point filenames, `Model_generate.py` and `Model_run.py`. `MODEL_NAME` is derived automatically from the enclosing folder name (see `AntiGen_paths.py`), so being in the right project folder is all that's needed to generate/run the correct project. From `MyNewModel/Projects/Example/` run:

```bash
python Model_generate.py
```

This generates the model and writes outputs to `antimony_models/Example/` and `generated/Example/`. Edit parameters if desired in `antimony_models/Example/Example_parameters.csv`, then run:

```bash
python Model_run.py
```

Your own model lives under `Projects/MyNewModel/` with the same file names as the Example. Modify the code for your model in `Projects/MyNewModel/Modules/`, `Projects/MyNewModel/Model_generate.py`, and `Projects/MyNewModel/Model_run.py`. Your problem will also require new modules in `antimony_modules/`.

Keeping the generation and simulation steps separate gives you time to adjust parameters and inspect the generated files before running.

### Running from an IDE (Cursor / VS Code)

The **Play** button uses whichever Python interpreter is currently selected. If your environment (conda/venv) isn’t loaded, the run may fail with import or path errors.

1. **Select the correct interpreter**: `Ctrl+Shift+P` (or `Cmd+Shift+P` on macOS) → **Python: Select Interpreter** → choose the environment where you ran `pip install -e .` (e.g. your conda or venv).
2. **Run from project root**: Open the *project* folder (e.g. `MyNewModel`) as the workspace. Use **Run and Debug** (or Play on `Projects/Example/Model_run.py`); the project root is resolved from the script location so `antimony_models/Example/`, `generated/Example/`, and `results/Example/` resolve correctly.

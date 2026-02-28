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
git clone https://github.com/<your-username>/PyAntiGen.git
cd PyAntiGen
pip install -e .
```

## Quick Start: Creating a New Model

Because PyAntiGen is installed as a system-level Python package, you don't need a copy of the framework files in your working directory. 

To start a brand new modeling workspace, just open a terminal where you want it to live and run:

```bash
pyantigen-create MyNewModel
```

This will automatically scaffold the following project directory:
```text
MyNewModel/
├── scripts/
│   └── MyNewModel_main.py
├── modules/
│   └── __init__.py
├── data/
├── antimony_models/
├── generated/
├── results/
└── SBML_models/
```

Navigate to `MyNewModel/scripts/` and run `python MyNewModel_main.py`—your model will automatically generate an Antimony script and safely log it to your `antimony_models` directory!

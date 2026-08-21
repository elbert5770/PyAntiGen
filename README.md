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

To start a brand new modeling workspace, just open a terminal and navigate to a folder where you want your project to live (be careful not to build within the PyAntiGen folder itself) and run:

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

Every project folder under `Projects/` has a script called `Model_generate.py` that is run to generate the model. `MODEL_NAME` is derived automatically from the enclosing folder name (see `AntiGen_paths.py`), so being in the right project folder is all that's needed to generate/run the correct project. From `MyNewModel/Projects/Example/` run:

```bash
python Model_generate.py
```

This generates the model from the specified files in 'antimony_modules' and writes outputs to `antimony_models/Example/` and `generated/Example/`. This is entirely optional as Model_run.py also calls the constructor functions in `Model_generate.py`.

To run a simulation or optimization, you may edit the parameters in `antimony_models/Example/Example_parameters.csv`, derived parameters in `antimony_models/Example/Example_manual.txt`, and initial conditions in `antimony_models/Example/Example_InitialConditions.csv`.

Then, run `Model_run.py` to simulate the Example in Tellurium/RoadRunner:

```bash
python Model_run.py --simulate Example
```

To run optimization examples use the --optimize flag:

```bash
python Model_run.py --optimize Example1
python Model_run.py --optimize Example2
python Model_run.py --optimize Example3
python Model_run.py --optimize Example4
python Model_run.py --optimize Example5
```

Examples 1–3 demonstrate structural identifiability: Example1/2 split the fit so each sub-problem is well-posed, while Example3 deliberately fits two exactly confounded parameters (`SF`/`V_Comp1`) jointly and shows how profile likelihood flags the ridge that likelihood slices and Sobol indices miss.

Examples 4–5 go further and test the *accuracy* of the profile likelihood ΔNLL itself, on a genuinely multimodal problem with a log10 objective. The chain A → B → C observed through `SF*B_Comp1/V_Comp1` has the classic pharmacokinetic "flip-flop" ambiguity — swapping the two rate constants and rescaling `SF` reproduces the data exactly — so the likelihood has two modes separated by a known ΔNLL gap (~2.4 at the NLL optimum, set by a few deliberately noisy observations of A and printed by `data/make_flipflop_data.py`). Example4 starts in the correct basin. Accurate profiles must (a) dip below zero by a known amount (~−2.1), because the fitting objective's per-observable averaging places the fit away from the inference NLL optimum that ΔNLL is anchored to, and (b) show the second mode at ~+0.8 — below the 95% threshold, so the correct confidence set is a union of two disjoint intervals; a walker that stops at the first threshold crossing never finds it, and a first-crossing CI extractor cannot represent it. Example5 starts in the wrong basin: sigmas are frozen at the wrong mode (inflating σ for the A data and deflating every ΔNLL), and the profile must dip to ~−0.98 at the true mode. `Projects/Example/Flipflop_reference.py` recomputes the exact reference profiles from the closed-form solution with scipy (independent of RoadRunner and of the framework's loss code), replicating the pipeline's conventions — fit objective for the anchor, MLE-frozen sigmas, summed NLL — and its `--compare results/Example/<run>.json` mode scores the framework's stored profile traces against the reference automatically.

Your own model lives under `Projects/MyNewModel/`. Modify the code for your model in `Projects/MyNewModel/Modules/`, `Projects/MyNewModel/Model_generate.py`, and `Projects/MyNewModel/Model_run.py`. Your problem will also require new modules in `antimony_modules/` to define the model.


### Running from an IDE (Cursor / VS Code)

The **Play** button uses whichever Python interpreter is currently selected. If your environment (conda/venv) isn’t loaded, the run may fail with import or path errors.

1. **Select the correct interpreter**: `Ctrl+Shift+P` (or `Cmd+Shift+P` on macOS) → **Python: Select Interpreter** → choose the environment where you ran `pip install -e .` (e.g. your conda or venv).
2. **Run from project root**: Open the *project* folder (e.g. `MyNewModel`) as the workspace. Use **Run and Debug** (or Play on `Projects/Example/Model_run.py`); the project root is resolved from the script location so `antimony_models/Example/`, `generated/Example/`, and `results/Example/` resolve correctly.

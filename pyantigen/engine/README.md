# Engine/ — the Engine ⇄ Modules contract

`Engine/` is the portable fitting/simulation/profiling framework: it is meant
to be reused unchanged across model projects (see `ENGINE_PORTABILITY_HANDOFF.md`
in this project for the audit that produced this file). It never contains
project-specific model, data, drug, species, or figure code — instead it
imports a small, fixed set of names from the project's own `Modules/`
package, and reads a few dict keys off objects the project supplies.

This file is the complete list of what a new project's `Modules/` must
provide for `Engine/*.py` to import and run cleanly. It exists because
without it, a mistemplated project's failure mode is a late, confusing
`ModuleNotFoundError` or `KeyError` several calls deep, instead of a clear
"you forgot X" at startup. If `Engine/` ever comes to need something else
from `Modules/`, add it here in the same edit.

## Modules/ files Engine imports directly

- **`Modules/Experiment.py`** — wildcard-imported by `Model_optimize.py` and
  `Model_simulate.py`. Must make available whatever `EXPERIMENT_dict` the
  project passes into `setup_optimization_from_groups` / `run_simulation`;
  Engine itself never references a specific experiment name, only the dict
  shape below.
- **`Modules/Plots.py`** — wildcard-imported by the same two files. Can be a
  near-empty stub (no figures needed yet) as long as it exists; Engine's own
  code never calls a specific plot function by name.
- **`Modules/Optimizer_settings.py`** — must define an `Optimization` class
  (a dataclass) with at minimum `param_names`, `x0`, `bounds`. Engine imports
  only the *class*; every project supplies its own instances. See that
  module's own docstring for the full field list.

`Modules/Loss_config.py`, `Modules/Data.py`, `Modules/Events.py`,
`Modules/Update_parameters.py`, `Modules/Observed_species.py`,
`Modules/Solver_settings.py` are **not** imported by `Engine/` by name — a
project can name/organize that implementation however it likes — but Engine
does require the callables they produce to be reachable through the
replicate dict contract below.

## `EXPERIMENT_dict` contract

Passed by the project into `Model_optimize.setup_optimization_from_groups`
and `Model_simulate.run_simulation`:

- `EXPERIMENT_dict["EXPERIMENT"]` — an object with a `.replicates` attribute:
  a `dict[label, replicate]`, where each `replicate` is a plain dict (see
  below). `Evaluator.py`'s `spec.replicates` (an `Optimization` spec) has the
  same shape.
- `EXPERIMENT_dict["plot"]` — `callable(paths, results_dict)`.

## Replicate dict contract

`Engine/Model_simulate.py`, `Model_optimize.py`, `Optimize.py`,
`Evaluator.py`, `Preequil_cache.py`, `Sensitivity_analysis.py`,
`Petab_export.py`, and `Event_times.py` all treat a replicate as a plain
dict. Keys Engine reads:

**Required:**
- `replicate["Data"](replicate, data_path)` → `df_dict`
- `replicate["Events"](replicate, df_dict[, r_ic=...])` → events (Antimony)
  text. Engine tries the `r_ic=` keyword first and falls back to the 2-arg
  call on `TypeError` — accept either signature.
- `replicate["Update_parameters"](r, replicate)` → sets model parameters on
  a compiled RoadRunner instance
- `replicate["Solver_settings"](replicate)` → solver settings
- `replicate["Observed_species"](r)` → species to record

**Optional** (Engine checks with `.get()`/`in` and has a defined fallback):
- `replicate["Loss_config"]` — observable/loss spec callable. Absent or
  `None` ⇒ the replicate is simulated but not fit (Engine passes it through
  to plotting with an empty loss).
- `replicate["Label"]` — falls back to `"?"` if absent.
- `replicate["parameter_hooks"]` — list of `callable(r, p_vec)` run in
  addition to `Update_parameters`.
- `replicate["Update_opt_parameters"]` — `callable(r, p_vec)` for
  optimizer-specific parameter updates.

**Engine-owned** (Engine sets this itself — do not use the name for project
content):
- `replicate["_event_times_fn"]` — set by `Engine/Event_times.py:attach_event_times`.

## What Engine explicitly does not require

There is no ABC/protocol validation today — the contract above is duck-typed
(dict keys and module-level names checked only at the point of use). Engine
never imports a project's figure functions, drug names, species names, or
file paths by name; those live entirely in `Modules/`.

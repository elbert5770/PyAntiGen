
import os
import sys
import pandas as pd

from pyantigen.generate.AntimonyGen import AntimonyGen
from pyantigen.generate.TelluriumGen import TelluriumGen

from pyantigen.engine.Event_times import attach_event_times
from pyantigen.engine.Simulate import simulate


def _project(settings):
    """MODEL_NAME and REPO_ROOT for this run, taken from settings.

    1.x read both from a project-level AntiGen_paths module imported here,
    which tied the Engine to one project's folder layout. The project's
    runner now passes them in; nothing in the Engine imports project code.
    """
    try:
        return settings["MODEL_NAME"], settings["REPO_ROOT"]
    except KeyError as exc:
        raise KeyError(
            f"settings is missing {exc.args[0]!r}; the project runner must pass "
            "MODEL_NAME and REPO_ROOT (see template Model_run.py)") from None

def run_steady_state(model_text, paths, settings):
    
    rss = TelluriumGen(model_text, paths, settings)
    if settings["Verbose"]:
        print("Steady state: ", rss.steadyState())
        print("getFloatingSpeciesIds: ", rss.getFloatingSpeciesIds())
        print("getBoundarySpeciesIds: ", rss.getBoundarySpeciesIds())
        print("getAssignmentRuleIds: ", rss.getAssignmentRuleIds())
    
    
    if os.path.exists(paths["models_path"]):
        df_ic = pd.read_csv(paths["models_path"])
        if 'Species' in df_ic.columns:
            max_val = 0.0
            vals = {}
            for idx, row in df_ic.iterrows():
                species = row['Species']
                try:
                    # using roadrunner's dict-like access which is robust
                    val = rss[species]
                    vals[idx] = val
                    if val > max_val:
                        max_val = val
                except RuntimeError:
                    pass
            for idx, val in vals.items():
                if val < 1e-10 * max_val:
                    val = 0.0
                df_ic.at[idx, 'InitialCondition'] = val
            df_ic.to_csv(paths["models_path"], index=False)
            print(f"Updated InitialConditions in {paths['models_path']}")
        else:
            print(f"No 'Species' column found in {paths['models_path']}")
            return
    else:
        print(f"No InitialConditions file found in {paths['ic_path']}")
        return

    Species = rss.getFloatingSpeciesIds()
    for s in Species:
        print(s, rss.getValue(s))


def _print_comparison(label, rows, show=8):
    """The AMICI-versus-RoadRunner disagreement of one replicate, worst columns first."""
    print(f"[compare] {label}: AMICI vs RoadRunner on {len(rows)} shared column(s); "
          f"max relative difference {rows[0][2]:.3e} ({rows[0][0]})" if rows else
          f"[compare] {label}: no shared column")
    for name, diff, rel, at, a, b in rows[:show]:
        print(f"    {name:<48s} max|diff| {diff:.3e}   relative {rel:.3e}   at {at:.6g} h: AMICI {a:.6g}, RoadRunner {b:.6g}")


def run_simulation(model_text, paths, settings, EXPERIMENT_dict, parameter_overrides=None):

    data_path = paths["data_path"]
    repo_root = paths["repo_root"]
    MODEL_NAME = paths["MODEL_NAME"]
    print("run_simulation", MODEL_NAME)
    save_path = os.path.join(repo_root, "generated", MODEL_NAME, MODEL_NAME + "_events.txt")

    use_amici = str(settings.get("backend") or "roadrunner").lower() == "amici"
    if use_amici:
        if parameter_overrides:
            raise ValueError("parameter_overrides are not supported with the AMICI backend")
        from pyantigen.engine.Backend import default_cache_dir, explicit_cache_dir
        cache_dir = explicit_cache_dir(settings.get("amici_cache_dir")) or default_cache_dir()
        # Figures, parity CSVs and caches go to results/<MODEL>_amici, so an AMICI run never overwrites the
        # RoadRunner figures it is being compared with (and a figure that reads another figure's CSV reads the AMICI one).
        paths = dict(paths, plot_path=os.path.normpath(paths["plot_path"]) + "_amici", backend="amici")
        os.makedirs(paths["plot_path"], exist_ok=True)
        print(f"[amici] figures go to {paths['plot_path']}")

    results_dict = {}
    left_out = {}
    experiment = EXPERIMENT_dict['EXPERIMENT']
    for label, replicate in experiment.replicates.items():
        print("Label", label)
        df_dict = replicate["Data"](replicate, data_path)
        events = replicate["Events"](replicate,df_dict)

        with open(save_path, "w") as f:
            f.write(events)

        full_model_text = model_text + "\n" + events

        r = TelluriumGen(full_model_text, paths, settings)
        if use_amici:
            # the replay AMICI's setup runs restores this baseline before Update_parameters, as an optimization does
            from pyantigen.engine.Optimize import remember_parameter_baseline
            remember_parameter_baseline(r)

        replicate["Update_parameters"](r, replicate)
        if parameter_overrides:
            for p_name, p_val in parameter_overrides.items():
                try:
                    r[p_name] = p_val
                except Exception:
                    pass

        # After the overrides: a trigger built on an overridden parameter has to
        # resolve against the value this run will actually integrate with.
        attach_event_times(replicate, r, verbose=True)

        solver_settings = replicate["Solver_settings"](replicate)
        observed_species = replicate["Observed_species"](r)
        if use_amici:
            from pyantigen.engine.Simulate_amici import compare_results, simulate_amici
            roadrunner = simulate(r, solver_settings, observed_species) if settings.get("compare_backends") else None
            results, dropped = simulate_amici(
                replicate, r, events, model_text, paths, replicate["Label"], cache_dir, solver_settings,
                verbose=settings.get("Verbose", True), seed=settings.get("amici_seed"))
            if dropped:
                left_out[replicate["Label"]] = dropped
                print(f"[amici] {replicate['Label']}: not in the SBML, left out: {dropped}")
            if roadrunner is not None:
                _print_comparison(replicate["Label"], compare_results(results, roadrunner))
        else:
            results = simulate(r, solver_settings, observed_species)


        results_dict[replicate["Label"]] = {
            "results": results,
            "replicate": replicate,
            "data": df_dict,
            "observed_species": observed_species,
            "solver_settings": solver_settings,
            "events": events
        }
    try:
        EXPERIMENT_dict["plot"](paths,results_dict)
    except KeyError as e:
        if not left_out:
            raise
        # AMICI does not output reaction rates, so a plot that reads one cannot be drawn from its result.
        need = str(e.args[0] if e.args else e).split(". Available")[0]
        print(f"[amici] plot skipped: {need}, a column AMICI does not provide "
              f"(left out above). Run without --backend amici to draw the figure.")
    return results_dict

def setup_simulation(settings, EXPERIMENT_dict, parameter_overrides=None):
    MODEL_NAME, REPO_ROOT = _project(settings)
    print("setup_simulation", MODEL_NAME)

    model_text, paths = AntimonyGen(MODEL_NAME, repo_root=REPO_ROOT)

    if settings["run_steady_state_first"]:
        run_steady_state(model_text, paths, settings)

    results_dict = run_simulation(model_text, paths, settings, EXPERIMENT_dict, parameter_overrides=parameter_overrides)



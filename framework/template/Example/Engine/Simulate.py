import time
import numpy as np

class StackedResult(np.ndarray):
    def __new__(cls, input_array, colnames):
        obj = np.asarray(input_array).view(cls)
        obj.colnames = list(colnames)
        return obj

    def __getitem__(self, key):
        if isinstance(key, str):
            if hasattr(self, 'colnames') and key in self.colnames:
                return super().__getitem__((slice(None), self.colnames.index(key)))
            raise KeyError(f"Column '{key}' not in simulation results. "
                           f"Available: {getattr(self, 'colnames', [])}")
        return super().__getitem__(key)

def safe_simulate(r, settings):
    abs_tols = [1e-8, 1e-9, 1e-10, 1e-11, 1e-12, 1e-13, 1e-14, 1e-15]
    rel_tols = [1e-8, 1e-9, 1e-10, 1e-11, 1e-12, 1e-13, 1e-14, 1e-15]
    max_steps_list = [20000, 200000, 400000]
    
    last_exception = None
    variable_step_size = settings["variable_step_size"]
    start = settings["start_time"]
    end = settings["end_time"]
    points = settings["eval_points"]
    for max_steps in max_steps_list:
        r.integrator.maximum_num_steps = max_steps
        for a_tol in abs_tols:
            for r_tol in rel_tols:
                try:
                    # Reset tolerances
                    # r.reset()
                    r.integrator.absolute_tolerance = a_tol
                    r.integrator.relative_tolerance = r_tol
                    r.integrator.setValue('variable_step_size', variable_step_size)
                    # Reset model time to start of segment in case a previous attempt moved it
                    r.time = start
                    print(f"      Simulating from {start} to {end} with abs_tol={a_tol}, rel_tol={r_tol}, max_steps={max_steps}")
                    print(r.integrator)
                    res = r.simulate(start, end, points, cols)
                    metadata = {
                        "start": start,
                        "end": end,
                        "abs_tol": a_tol,
                        "rel_tol": r_tol,
                        "max_steps": max_steps,
                        "initial_time_step": r.integrator.initial_time_step,
                        "variable_step_size": variable_step_size
                    }
                    return res, metadata
                except Exception as e:
                    last_exception = e
                    continue
    
    print(f"      CRITICAL: All simulation attempts failed for interval [{start}, {end}]")
    try:
        print(f"      Final Time in model: {r.getValue('time')}")
        # Check for non-finite values in species
        sn = r.model.getFloatingSpeciesIds()
        sv = [r.getValue(s) for s in sn]
        nan_indices = [i for i, v in enumerate(sv) if not np.isfinite(v)]
        if len(nan_indices) > 0:
            print(f"      NaNs detected in: {[sn[i] for i in nan_indices]}")
    except:
        pass
    raise last_exception

def simulate(r, solver_settings, observed_species):
    
    solver_settings["integrator"] = solver_settings.get("integrator", "cvode")
    solver_settings["absolute_tolerance"] = solver_settings.get("absolute_tolerance", 1e-8)
    solver_settings["relative_tolerance"] = solver_settings.get("relative_tolerance", 1e-8)
    solver_settings["stiff"] = solver_settings.get("stiff", True)
    solver_settings["variable_step_size"] = solver_settings.get("variable_step_size", True)
    # solver_settings["initial_time_step"] = solver_settings.get("initial_time_step", 1e-6)
    # solver_settings["maximum_num_steps"] = solver_settings.get("maximum_num_steps", 20000)
    # print("solver_settings", solver_settings)
    r.setIntegrator(solver_settings["integrator"])
    r.integrator.absolute_tolerance = solver_settings["absolute_tolerance"]
    r.integrator.relative_tolerance = solver_settings["relative_tolerance"]
    r.integrator.setValue('stiff', solver_settings["stiff"])
    r.integrator.variable_step_size = solver_settings["variable_step_size"]
    # r.integrator.setValue('initial_time_step', solver_settings["initial_time_step"])
    # r.integrator.setValue('maximum_num_steps', solver_settings["maximum_num_steps"])
    # print(r.integrator)

    # te.noticesOff()
    # te.r.printVersionInfo()
    # te.roadrunner.Logger.setLevel(te.roadrunner.Logger.LOG_ERROR)

    t0 = time.perf_counter()

    blocks = solver_settings['simulation_blocks']
    if isinstance(blocks, dict):
        block_list = list(blocks.values())
    else:
        block_list = blocks

    # Drop any requested symbols that don't exist in this model variant.
    _ids = (set(r.getFloatingSpeciesIds()) | set(r.getBoundarySpeciesIds())
            | set(r.getAssignmentRuleIds()) | set(r.getGlobalParameterIds())
            | set(r.getReactionIds()))
    _available = {'time'} | _ids | {'[' + s + ']' for s in _ids}
    observed_species = [s for s in observed_species if s in _available]

    all_results = None

    # Find index of the last tracked block so we can skip trailing untracked ones.
    last_tracked_idx = -1
    for i, blk in enumerate(block_list):
        if blk.get('tracked', True):
            last_tracked_idx = i

    for i, block in enumerate(block_list):
        if i > last_tracked_idx:
            break
        tracked = block.get('tracked', True)

        if 'maximum_num_steps' in block:
            r.integrator.maximum_num_steps = block['maximum_num_steps']

        try:
            res = r.simulate(block['start'], block['end'], block['n_points'], observed_species)
        except Exception as exc:
            raise RuntimeError(
                f"Integration failed in block {i} "
                f"[{block['start']:.4g}, {block['end']:.4g}]: {exc}"
            ) from exc

        if not tracked:
            continue
        if all_results is None:
            all_results = res
        else:
            colnames = all_results.colnames
            stacked = np.vstack((all_results, res))
            all_results = StackedResult(stacked, colnames)
    return all_results

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

def save_model_state(r):
    # Get all assignment rules to exclude them from being set/restored
    try:
        assignment_rules = set(r.getAssignmentRuleIds())
    except Exception:
        assignment_rules = set()

    return {
        "time": r.getValue('time'),
        "floating_species": {s: r.getValue(s) for s in r.getFloatingSpeciesIds() if s not in assignment_rules},
        "boundary_species": {s: r.getValue(s) for s in r.getBoundarySpeciesIds() if s not in assignment_rules},
        "global_parameters": {s: r.getValue(s) for s in r.getGlobalParameterIds() if s not in assignment_rules}
    }

def restore_model_state(r, state):
    try:
        r.setValue('time', state["time"])
    except Exception:
        pass
    for s, val in state["floating_species"].items():
        try:
            r.setValue(s, val)
        except Exception:
            pass
    for s, val in state["boundary_species"].items():
        try:
            r.setValue(s, val)
        except Exception:
            pass
    for s, val in state["global_parameters"].items():
        try:
            r.setValue(s, val)
        except Exception:
            pass

def _adjust_settings_for_error(err_msg, cur_abs_tol, cur_rel_tol, cur_max_steps, cur_initial_step):
    """Return updated (abs_tol, rel_tol, max_steps, initial_step) based on the CVODE error message."""
    if any(k in err_msg for k in ["mxstep", "too many steps", "too much work", "mxstep steps taken"]):
        cur_max_steps = min(cur_max_steps * 2, 400000)
        cur_abs_tol = min(cur_abs_tol * 2, 1e-4)
        cur_rel_tol = min(cur_rel_tol * 2, 1e-4)
    elif any(k in err_msg for k in ["error test", "hmin", "step size"]):
        cur_abs_tol = min(cur_abs_tol * 10, 1e-4)
        cur_rel_tol = min(cur_rel_tol * 10, 1e-4)
        if cur_initial_step is None or cur_initial_step > 1e-9:
            cur_initial_step = 1e-9
        else:
            cur_initial_step = cur_initial_step / 10
    elif any(k in err_msg for k in ["corrector", "convergence", "mxncf"]):
        cur_abs_tol = min(cur_abs_tol * 10, 1e-4)
        cur_rel_tol = min(cur_rel_tol * 10, 1e-4)
        cur_max_steps = min(cur_max_steps * 2, 400000)
        if cur_initial_step is None or cur_initial_step > 1e-9:
            cur_initial_step = 1e-9
        else:
            cur_initial_step = cur_initial_step / 10
    else:
        cur_abs_tol = min(cur_abs_tol * 10, 1e-4)
        cur_rel_tol = min(cur_rel_tol * 10, 1e-4)
        cur_max_steps = min(cur_max_steps * 2, 400000)
    return cur_abs_tol, cur_rel_tol, cur_max_steps, cur_initial_step


def safe_simulate(r, solver_settings, observed_species, depth=0):
    # Extract start, end, points, variable_step_size
    start = solver_settings.get("start")
    if start is None:
        start = solver_settings.get("start_time")
    end = solver_settings.get("end")
    if end is None:
        end = solver_settings.get("end_time")
    points = solver_settings.get("n_points")
    if points is None:
        points = solver_settings.get("eval_points")
    variable_step_size = solver_settings.get("variable_step_size", True)

    # Ensure proper types
    start = float(start)
    end = float(end)
    points = int(points)

    # Fast path: try once with whatever settings the model already has. On
    # success we skip the integrator-attr churn, the time-set, and the full
    # model-state snapshot — all of which scale with species count and are
    # the dominant per-call overhead during optimization.
    try:
        res = r.simulate(start, end, points, observed_species)
        return res, {
            "start": start, "end": end,
            "variable_step_size": variable_step_size,
            "attempts": 1, "subdivided": False,
        }
    except Exception as e:
        last_exception = e
        first_err_msg = str(e).lower()
        first_err_str = str(e).strip()

    # Slow path: simulation failed. Now pay the cost to snapshot state, clamp
    # small negatives that may have caused a CVODE ill-defined error, and
    # enter the adaptive retry loop.
    orig_abs_tol = r.integrator.absolute_tolerance
    orig_rel_tol = r.integrator.relative_tolerance
    orig_max_steps = r.integrator.maximum_num_steps
    orig_initial_step = r.integrator.initial_time_step

    # Reset model time and clamp small negative initial values which can cause
    # CVODE ill-defined errors. Done before save_model_state so the clamped
    # values become the restore target for subsequent retries.
    r.setValue('time', start)
    for s_id in r.model.getFloatingSpeciesIds():
        if r[s_id] < 1e-12:
            r[s_id] = 0.0

    saved_state = save_model_state(r)

    # Pre-adjust settings using the first error so attempt 2 starts smarter
    # than just retrying with the same configuration.
    cur_abs_tol, cur_rel_tol, cur_max_steps, cur_initial_step = _adjust_settings_for_error(
        first_err_msg, orig_abs_tol, orig_rel_tol,
        min(max(orig_max_steps, 20000), 400000), orig_initial_step,
    )

    print(f"      [safe_simulate] (Depth {depth}) Initial attempt failed: {first_err_str}")

    max_attempts = 10
    attempt = 1  # one attempt already burned on the fast path

    while attempt < max_attempts:
        attempt += 1
        r.integrator.absolute_tolerance = cur_abs_tol
        r.integrator.relative_tolerance = cur_rel_tol
        r.integrator.maximum_num_steps = int(cur_max_steps)
        r.integrator.setValue('variable_step_size', variable_step_size)
        if cur_initial_step is not None:
            try:
                r.integrator.initial_time_step = cur_initial_step
            except Exception:
                pass

        # Restore state to try again
        restore_model_state(r, saved_state)

        try:
            print(f"      [safe_simulate] (Depth {depth}) Attempt {attempt}/{max_attempts}: Simulating [{start:.4g} to {end:.4g}] "
                  f"points={points} with abs_tol={cur_abs_tol:.2e}, rel_tol={cur_rel_tol:.2e}, max_steps={cur_max_steps}")
            res = r.simulate(start, end, points, observed_species)

            # Restore original integrator settings on success
            r.integrator.absolute_tolerance = orig_abs_tol
            r.integrator.relative_tolerance = orig_rel_tol
            r.integrator.maximum_num_steps = orig_max_steps
            try:
                r.integrator.initial_time_step = orig_initial_step
            except Exception:
                pass

            metadata = {
                "start": start,
                "end": end,
                "abs_tol": cur_abs_tol,
                "rel_tol": cur_rel_tol,
                "max_steps": cur_max_steps,
                "initial_time_step": cur_initial_step,
                "variable_step_size": variable_step_size,
                "attempts": attempt,
                "subdivided": False
            }
            return res, metadata

        except Exception as e:
            last_exception = e
            err_msg = str(e).lower()
            print(f"      [safe_simulate] (Depth {depth}) Attempt {attempt} failed: {str(e).strip()}")
            cur_abs_tol, cur_rel_tol, cur_max_steps, cur_initial_step = _adjust_settings_for_error(
                err_msg, cur_abs_tol, cur_rel_tol, cur_max_steps, cur_initial_step,
            )

    # Subdivide as last resort
    if depth < 4 and points >= 2:
        mid = (start + end) / 2.0
        p1 = max(2, points // 2)
        p2 = max(2, points - p1)
        print(f"      [safe_simulate] (Depth {depth}) ALL solver adjustment attempts failed. Subdividing [{start:.4g}, {end:.4g}] "
              f"into [{start:.4g}, {mid:.4g}] (points={p1}) and [{mid:.4g}, {end:.4g}] (points={p2})")
        
        # Restore to initial state for the first half
        restore_model_state(r, saved_state)
        
        # Simulate first half
        block1 = {
            "start": start,
            "end": mid,
            "n_points": p1,
            "variable_step_size": variable_step_size
        }
        res1, meta1 = safe_simulate(r, block1, observed_species, depth=depth+1)
        
        # Simulate second half starting from current state (which is at mid)
        block2 = {
            "start": mid,
            "end": end,
            "n_points": p2,
            "variable_step_size": variable_step_size
        }
        res2, meta2 = safe_simulate(r, block2, observed_species, depth=depth+1)
        
        # Stack results
        colnames = getattr(res1, "colnames", observed_species)
        if len(res2) > 1:
            stacked = np.vstack((res1, res2[1:]))
        else:
            stacked = res1
        res = StackedResult(stacked, colnames)
        
        # Restore original integrator settings before returning
        r.integrator.absolute_tolerance = orig_abs_tol
        r.integrator.relative_tolerance = orig_rel_tol
        r.integrator.maximum_num_steps = orig_max_steps
        try:
            r.integrator.initial_time_step = orig_initial_step
        except Exception:
            pass
            
        metadata = {
            "start": start,
            "end": end,
            "subdivided": True,
            "meta1": meta1,
            "meta2": meta2
        }
        return res, metadata

    # If all else fails, restore original settings and raise exception
    r.integrator.absolute_tolerance = orig_abs_tol
    r.integrator.relative_tolerance = orig_rel_tol
    r.integrator.maximum_num_steps = orig_max_steps
    try:
        r.integrator.initial_time_step = orig_initial_step
    except Exception:
        pass
        
    print(f"      [safe_simulate] (Depth {depth}) CRITICAL: All simulation attempts and subdivision failed for interval [{start:.4g}, {end:.4g}]")
    try:
        print(f"      Final Time in model: {r.getValue('time')}")
        sn = r.model.getFloatingSpeciesIds()
        sv = [r.getValue(s) for s in sn]
        nan_indices = [i for i, v in enumerate(sv) if not np.isfinite(v)]
        if len(nan_indices) > 0:
            print(f"      NaNs detected in: {[sn[i] for i in nan_indices]}")
    except Exception:
        pass
    raise last_exception

def simulate(r, solver_settings, observed_species):
    
    solver_settings["integrator"] = solver_settings.get("integrator", "cvode")
    solver_settings["absolute_tolerance"] = solver_settings.get("absolute_tolerance", 1e-8)
    solver_settings["relative_tolerance"] = solver_settings.get("relative_tolerance", 1e-8)
    solver_settings["stiff"] = solver_settings.get("stiff", True)
    solver_settings["variable_step_size"] = solver_settings.get("variable_step_size", True)
    # solver_settings["initial_time_step"] = solver_settings.get("initial_time_step", 1e-6)
    solver_settings["maximum_num_steps"] = solver_settings.get("maximum_num_steps", 20000)
    # print("solver_settings", solver_settings)
    r.setIntegrator(solver_settings["integrator"])
    r.integrator.absolute_tolerance = solver_settings["absolute_tolerance"]
    r.integrator.relative_tolerance = solver_settings["relative_tolerance"]
    r.integrator.setValue('stiff', solver_settings["stiff"])
    r.integrator.variable_step_size = solver_settings["variable_step_size"]
    # r.integrator.setValue('initial_time_step', solver_settings["initial_time_step"])
    r.integrator.setValue('maximum_num_steps', solver_settings["maximum_num_steps"])
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
    # The available-symbol set never changes after the model is compiled, so
    # cache it on the RoadRunner instance to avoid 5 SWIG round-trips per
    # block per optimization step. The negative-clamp that used to live here
    # has moved into safe_simulate's failure-recovery path.
    try:
        _available = r._available_symbols
    except AttributeError:
        _ids = (set(r.getFloatingSpeciesIds()) | set(r.getBoundarySpeciesIds())
                | set(r.getAssignmentRuleIds()) | set(r.getGlobalParameterIds())
                | set(r.getReactionIds()))
        _available = {'time'} | _ids | {'[' + s + ']' for s in _ids}
        try:
            r._available_symbols = _available
        except Exception:
            pass
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
            res, _ = safe_simulate(r, block, observed_species)
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

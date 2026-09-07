import time
import numpy as np

# Magnitude below which a species value is numerical dust rather than a
# quantity.
#
# Seventy years of pre-equilibration does not leave the compartments that hold
# no material at exactly zero. It leaves them at things like -2.4e-107, and on
# Lecanemab_10mgkg 94 of 269 species end the pre-dose block negative with 109
# of them below 1e-30. Nothing is wrong with that trajectory -- but when the
# dose then fires, those species acquire derivatives of order 1e+05 while
# sitting at 1e-107, so the integrator has to carry them up through roughly a
# hundred orders of magnitude. It does, by taking enormous numbers of tiny
# steps: the first 13.6 hours after the dose accounted for 98.5% of a 123 s
# block. Zeroing the dust first costs nothing and takes that block to 0.39 s,
# with no material change to any result.
#
# The threshold is deliberately far below anything physical. The real
# pre-dose value of these species is exactly zero, while genuinely small
# quantities elsewhere in the model reach 1e-15 (AB40_DensePlaqueNumber_BrainISF
# on the antibody-trial arms). A 1e-12 cut would delete those; 1e-25 cannot
# reach them and still removes dust that is eighty orders of magnitude smaller.
_DUST_THRESHOLD = 1e-25


def clamp_state_dust(r, threshold=_DUST_THRESHOLD):
    """Zero every floating species whose magnitude is below *threshold*.

    Returns the number of species zeroed. Negative dust and positive dust are
    treated alike: both are noise around a true zero, and the negative half is
    what trips CVODE's error-weight check on the next step.
    """
    n = 0
    for s_id in r.model.getFloatingSpeciesIds():
        try:
            v = r[s_id]
        except Exception:
            continue
        if v != 0.0 and -threshold < v < threshold:
            r[s_id] = 0.0
            n += 1
    return n

# Smallest per-species absolute tolerance CVODE is allowed to be given.
#
# RoadRunner does not hand CVODE the scalar `absolute_tolerance` set on the
# integrator. It derives a per-species vector from it, scaled by each species'
# current amount:
#
#     atol_i = absolute_tolerance * amount_i          (amount = conc * volume)
#     atol_i = absolute_tolerance * volume_i          when amount_i is exactly 0
#
# There is no lower bound on that product, so a species decaying toward zero
# drags its own tolerance down with it -- and the tolerance underflows long
# before the value does. Tracing the vector's floor through one GRADUATE arm:
#
#     after load or reset()                6.6e-13
#     after the 72-year pre-equilibration  1.48e-24
#     by the third dosing piece            4.25e-234
#     by the eighth                        9.23e-301
#     by the twelfth                       exactly 0
#
# At zero, CVODE refuses the call outright: CV_ILL_INPUT from cvInitialSetup,
# "Initial ewt has component(s) equal to zero (illegal)". That is a
# start-of-call check, so every r.simulate() is another chance to trip it, and
# the arms that trip it are the ones that integrate longest. Across the SILK
# arms the floor tracks follow-up length exactly -- 1.53e-178 at 20 years,
# 4.33e-230 at 27.8, 3.45e-270 at 35, and 2.07e-308 at 60, which is already
# subnormal. SILK_young_neg is named for when its labeling starts, not for how
# long it runs; it runs the longest and it is the one that fails.
#
# Re-setting the scalar does not repair the vector, and neither does
# setIntegrator; only reset() does, by restoring the initial state.
# setIndividualTolerance does, per species, and survives both simulate() and
# setIntegrator -- so the vector is floored here before each block instead.
#
# The value is bounded above by the smallest quantity that carries a result --
# 1e-15 in this model -- and below by the point where it stops reaching the
# tolerances that actually fail. 1e-21 is six orders below that smallest real
# quantity, so an absolute tolerance there still resolves it to six figures.
#
# This sits *above* _DUST_THRESHOLD, which an earlier value deliberately
# avoided. That constraint guards the wrong thing: everything below the dust
# threshold is zeroed by clamp_state_dust before the integrator sees it, so a
# floor above it cannot loosen anything that survives to be integrated.
# Requiring the floor to sit under the dust threshold pushed it down into a
# region where it no longer engages, which is how 1e-30 came to be a floor that
# almost never fires on the arms that need it.
#
# Measured on the current model, one evaluation of each spec at x0. An earlier
# table here recorded 0/18 retries at 1e-30; that no longer reproduces.
#
#              microglia_clearance        silk_appfull
#   floor      blocks   retries        blocks   retries   real obs
#   1e-30          66        11            85         0        --
#   1e-27          56         2            85         0   4.8e-09
#   1e-24          56         2            85         0   3.6e-09
#   1e-21          56         0            85         0   6.6e-09
#   1e-18          56         0            85         0   6.6e-09
#
# At 1e-30 six arms fall into the retry ladder -- all four Aducanumab doses,
# GRAD_1_Drug and Lecanemab 10 mg/kg -- and the ten extra blocks are the
# subdivisions that follow. Their tolerance minima sit at 1e-24 to 1e-29:
# above the old floor, so nothing was raised, and below what CVODE can work
# with. A retry loosens abs_tol to 1e-8 and rel_tol to 1e-10, so that run is
# not the accurate baseline it looks like -- it is the one that gave up
# accuracy to finish at all.
#
# At 1e-21 no arm retries or subdivides. Against the 1e-30 run the joint NLL
# moves by 1.1e-06 nats, and every state value that differs is a residue around
# a true zero -- the same population the earlier analysis identified. The
# largest mover of any size is Antibody_SubCutComp at 3.1e-07, the
# subcutaneous depot after the drug has cleared; the largest quantity that
# shifts at all is 4.2e-04, by 0.12%. Nothing that carries a result moves.
# On silk_appfull nothing changes at any floor: no retries anywhere, and
# observables agree to eight or nine significant figures.
#
# 1e-18 also clears the retries but leaves only three figures on a 1e-15
# quantity, so 1e-21 is the point that fixes the failures with headroom left.
_MIN_ABSOLUTE_TOLERANCE = 1e-21


def _scalar_tolerance(value, fallback=1e-9):
    """Read a tolerance back from the integrator as a single number.

    Once ``setIndividualTolerance`` has been used, RoadRunner's
    ``absolute_tolerance`` *getter* stops returning the scalar that was set and
    returns the whole per-species vector as a list instead. The retry ladder
    then does ``min(cur_abs_tol * 10, 1e-4)`` on a list, where ``* 10`` is list
    repetition and the comparison raises

        '<' not supported between instances of 'float' and 'list'

    which surfaces as an integration failure with a nonsense message. The
    tightest entry is the right scalar to carry forward: it is the accuracy the
    run was actually being held to, and loosening from there is what the ladder
    is for.
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        pass
    try:
        values = [float(v) for v in value]
    except (TypeError, ValueError):
        return fallback
    return min(values) if values else fallback


def state_vector_ids(r):
    """Ids for CVODE's state vector, in the integrator's own order.

    ``getAbsoluteToleranceVector`` returns one entry per *state variable*, and
    the floating species are only the first part of that: a model with rate
    rules carries one further entry per rate-rule variable, appended after the
    species. Verified against RoadRunner by setting a distinct tolerance per id
    and reading the vector back -- the order is exactly species then rate rules,
    and ``setIndividualTolerance`` accepts a rate-rule id like any other.

    This matters because the obvious thing -- zipping the vector against
    ``getFloatingSpeciesIds()`` -- is silently short. ``zip`` stops at the
    shorter sequence, so every rate-rule entry falls off the end and is never
    floored, on a model where nothing about the call looks wrong. Those
    variables decay like any other state and CVODE's error test covers them, so
    an unfloored one is enough to fail a block on its own.
    """
    ids = []
    try:
        ids.extend(r.model.getFloatingSpeciesIds())
    except Exception:
        return ids
    for accessor in ("getRateRuleIds", "getRateRuleSymbols"):
        for holder in (r, getattr(r, "model", None)):
            if holder is None:
                continue
            try:
                extra = list(getattr(holder, accessor)())
            except Exception:
                continue
            if extra:
                return ids + extra
    return ids


def floor_tolerance_vector(r, floor=_MIN_ABSOLUTE_TOLERANCE):
    """Raise every per-state absolute tolerance below *floor* up to it.

    Returns the number raised. Cheap and idempotent: on a freshly reset model
    nothing is below the floor and nothing is touched.

    Must run *after* the scalar tolerance is applied, because setting the scalar
    re-derives the whole vector and discards individual settings. It therefore
    belongs before each block rather than once per run -- the vector is healthy
    at t=0 and degrades as the trajectory does, so flooring once at setup
    achieves precisely nothing.

    Covers the whole state vector, not just the floating species: see
    :func:`state_vector_ids` for why that distinction is not academic.
    """
    try:
        current = list(r.integrator.getAbsoluteToleranceVector())
    except Exception:
        return 0
    ids = state_vector_ids(r)
    n = 0
    # Still a zip, but now against the full state vector. If a future
    # RoadRunner adds another class of state variable the tail is skipped
    # rather than misassigned -- the ordering is a prefix, so what is named is
    # named correctly.
    for s_id, tol in zip(ids, current):
        if tol < floor:
            try:
                r.integrator.setIndividualTolerance(s_id, floor)
                n += 1
            except Exception:
                pass
    return n


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
    # Cache the state keys on the RoadRunner instance to minimize overhead
    try:
        keys = r._cached_state_keys
    except AttributeError:
        try:
            assignment_rules = set(r.getAssignmentRuleIds())
        except Exception:
            assignment_rules = set()
        keys = ['time']
        keys += [s for s in r.getFloatingSpeciesIds() if s not in assignment_rules]
        keys += [s for s in r.getBoundarySpeciesIds() if s not in assignment_rules]
        keys += [s for s in r.getGlobalParameterIds() if s not in assignment_rules]
        r._cached_state_keys = keys

    prev_selections = r.selections
    r.selections = keys
    values = list(r.getSelectedValues())
    r.selections = prev_selections
    
    return {
        "keys": keys,
        "values": values
    }

def restore_model_state(r, state):
    r.setValues(state["keys"], state["values"])

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


def safe_simulate(r, solver_settings, observed_species, depth=0, label=None):
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

    # Save original state before any simulation is attempted.
    # This is critical because if the fast path fails, it leaves the model in a
    # corrupted/advanced state that cannot be reverted without this snapshot.
    saved_state = save_model_state(r)

    # Fast path: try once with whatever settings the model already has.
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

    # Slow path: simulation failed. Now pay the cost to restore state, clamp
    # small negatives that may have caused a CVODE ill-defined error, and
    # enter the adaptive retry loop.
    #
    # The configured scalar cannot be read back from the integrator once
    # floor_tolerance_vector has run: the getter then returns the per-species
    # vector, whose minimum is the artificial 1e-30 floor -- and a ladder
    # seeded there loosens to 2e-30, fails every attempt, and subdivides
    # forever. configure_integrator caches the real scalar for exactly this
    # read; the getter path remains only for callers that never went through
    # configure_integrator.
    orig_abs_tol = getattr(r, "_scalar_abs_tol", None)
    if orig_abs_tol is None:
        orig_abs_tol = _scalar_tolerance(r.integrator.absolute_tolerance)
    orig_rel_tol = _scalar_tolerance(r.integrator.relative_tolerance)
    orig_max_steps = r.integrator.maximum_num_steps
    orig_initial_step = r.integrator.initial_time_step

    # Restore the model to the starting state to revert any changes made by the failed attempt
    restore_model_state(r, saved_state)

    # Clamp dust which can cause CVODE ill-defined errors. Done before re-saving
    # so the clamped values become the restore target for subsequent retries.
    #
    # This used to zero everything below 1e-12, which on the antibody-trial arms
    # also destroyed genuine quantities: AB40_DensePlaqueNumber_BrainISF sits at
    # about 1e-15 there, and those arms do reach this path. The threshold now
    # only reaches values that cannot be anything but noise.
    clamp_state_dust(r)

    # Save state again with the clamped values
    saved_state = save_model_state(r)

    # Pre-adjust settings using the first error so attempt 2 starts smarter
    # than just retrying with the same configuration.
    cur_abs_tol, cur_rel_tol, cur_max_steps, cur_initial_step = _adjust_settings_for_error(
        first_err_msg, orig_abs_tol, orig_rel_tol,
        min(max(orig_max_steps, 20000), 400000), orig_initial_step,
    )

    label_prefix = f" [{label}]" if label else ""
    print(f"      [safe_simulate]{label_prefix} (Depth {depth}) Initial attempt failed: {first_err_str}")

    max_attempts = 10
    attempt = 1  # one attempt already burned on the fast path

    while attempt < max_attempts:
        attempt += 1
        r.integrator.absolute_tolerance = cur_abs_tol
        r.integrator.relative_tolerance = cur_rel_tol
        r.integrator.maximum_num_steps = int(cur_max_steps)
        r.integrator.setValue('variable_step_size', variable_step_size)
        # Setting the scalar re-derives the per-species vector from the current
        # state and discards any individual settings, so the floor has to be
        # re-applied here or every retry would run without it -- including the
        # retries provoked by its absence.
        floor_tolerance_vector(r)
        if cur_initial_step is not None:
            try:
                r.integrator.initial_time_step = cur_initial_step
            except Exception:
                pass

        # Restore state to try again
        restore_model_state(r, saved_state)

        try:
            print(f"      [safe_simulate]{label_prefix} (Depth {depth}) Attempt {attempt}/{max_attempts}: Simulating [{start:.4g} to {end:.4g}] "
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
            print(f"      [safe_simulate]{label_prefix} (Depth {depth}) Attempt {attempt} failed: {str(e).strip()}")
            cur_abs_tol, cur_rel_tol, cur_max_steps, cur_initial_step = _adjust_settings_for_error(
                err_msg, cur_abs_tol, cur_rel_tol, cur_max_steps, cur_initial_step,
            )

    # Subdivide as last resort
    if depth < 4 and points >= 2:
        mid = (start + end) / 2.0
        p1 = max(2, points // 2)
        p2 = max(2, points - p1)
        print(f"      [safe_simulate]{label_prefix} (Depth {depth}) ALL solver adjustment attempts failed. Subdividing [{start:.4g}, {end:.4g}] "
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
        res1, meta1 = safe_simulate(r, block1, observed_species, depth=depth+1, label=label)
        
        # Simulate second half starting from current state (which is at mid)
        block2 = {
            "start": mid,
            "end": end,
            "n_points": p2,
            "variable_step_size": variable_step_size
        }
        res2, meta2 = safe_simulate(r, block2, observed_species, depth=depth+1, label=label)
        
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
        
    print(f"      [safe_simulate]{label_prefix} (Depth {depth}) CRITICAL: All simulation attempts and subdivision failed for interval [{start:.4g}, {end:.4g}]")
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

def configure_integrator(r, solver_settings):
    """Fill in solver defaults and apply them to *r*.

    Split out of :func:`simulate` so that anything running a single block on its
    own -- the pre-dose cache in particular -- integrates under exactly the same
    solver configuration the block would have got inside the normal loop. Two
    copies of this setup that drifted apart would produce a cached state subtly
    different from the one it replaces.

    Mutates *solver_settings* with the defaults, as the original inline version
    did, so callers that read the values back afterwards still see them.
    """
    solver_settings["integrator"] = solver_settings.get("integrator", "cvode")
    solver_settings["absolute_tolerance"] = solver_settings.get("absolute_tolerance", 1e-8)
    solver_settings["relative_tolerance"] = solver_settings.get("relative_tolerance", 1e-8)
    solver_settings["stiff"] = solver_settings.get("stiff", True)
    solver_settings["variable_step_size"] = solver_settings.get("variable_step_size", True)
    # solver_settings["initial_time_step"] = solver_settings.get("initial_time_step", 1e-6)
    solver_settings["maximum_num_steps"] = solver_settings.get("maximum_num_steps", 20000)
    r.setIntegrator(solver_settings["integrator"])
    r.integrator.absolute_tolerance = solver_settings["absolute_tolerance"]
    # Cache the scalar where safe_simulate can recover it. The integrator's
    # own getter is unreliable for this: after floor_tolerance_vector uses
    # setIndividualTolerance it returns the per-species vector, whose min is
    # the 1e-30 floor rather than the configured accuracy.
    try:
        r._scalar_abs_tol = float(solver_settings["absolute_tolerance"])
    except Exception:
        pass
    r.integrator.relative_tolerance = solver_settings["relative_tolerance"]
    r.integrator.setValue('stiff', solver_settings["stiff"])
    r.integrator.variable_step_size = solver_settings["variable_step_size"]
    # r.integrator.setValue('initial_time_step', solver_settings["initial_time_step"])
    r.integrator.setValue('maximum_num_steps', solver_settings["maximum_num_steps"])
    return solver_settings


def simulate(r, solver_settings, observed_species, label=None):

    configure_integrator(r, solver_settings)

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

        # Before every block, not once per run: RoadRunner's per-species
        # tolerance vector is derived from the current state, so it is healthy
        # here at t=0 and underflows as the trajectory decays. See
        # _MIN_ABSOLUTE_TOLERANCE.
        floor_tolerance_vector(r)

        try:
            res, _ = safe_simulate(r, block, observed_species, label=label)
        except Exception as exc:
            raise RuntimeError(
                f"Integration failed in block {i} "
                f"[{block['start']:.4g}, {block['end']:.4g}]: {exc}"
            ) from exc

        if not tracked:
            # An untracked block exists only to leave the model in a state, and
            # the dust it leaves in the empty compartments is what makes the
            # first dosed block expensive. Clear it here, where the block's own
            # output is being discarded anyway.
            clamp_state_dust(r)
            continue
        if all_results is None:
            all_results = res
        else:
            colnames = all_results.colnames
            stacked = np.vstack((all_results, res))
            all_results = StackedResult(stacked, colnames)
    return all_results

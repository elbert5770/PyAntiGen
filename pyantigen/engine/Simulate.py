import time
import numpy as np

from pyantigen.engine import Model_cache as _model_cache

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
            # Look the method up on the class, not the instance. RoadRunner's
            # __getattr__ fallback treats any missed instance attribute as a
            # possible model symbol and pays for a full _getIds() rebuild of
            # the symbol table (seconds, not microseconds) before giving up --
            # which is invisible when a model has rate rules (the first probe
            # succeeds and short-circuits the loop) and becomes the dominant
            # cost of every retry once a model has none, since a plain
            # getattr(holder, accessor) miss on "getRateRuleSymbols" would
            # trigger that fallback on every single call. Reading from
            # type(holder) resolves via the normal class MRO and never
            # touches the instance's __getattr__.
            method = getattr(type(holder), accessor, None)
            if method is None:
                continue
            try:
                extra = list(method(holder))
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


# ---------------------------------------------------------------------------
# Pinning the vector, not only flooring it
# ---------------------------------------------------------------------------
# floor_tolerance_vector raises what has fallen below the floor and leaves
# everything else as RoadRunner has it. That is enough for a vector that has
# drifted DOWN, which is what the floor was written for. It does nothing about
# two other things:
#
#   * The retry ladder sets the scalar before every attempt, and setting the
#     scalar re-derives the whole vector from the amounts the model holds
#     *at that moment* -- the failing state, not the initial one. Measured on
#     Aducanumab 60 mg/kg: between the segment-start vector and the vector of
#     attempt 2, 176 entries were more than 10x larger and 62 more than 10x
#     smaller (ratios 1.7e-11 .. 2.9e+07). The floor raised 3 of them. So the
#     loosened retries do not run at "the same tolerances, looser"; they run at
#     tolerances derived from wherever the integration died.
#
#   * RoadRunner's initial-derived scale is atol_i = scalar * amount_i, or
#     scalar * volume_i for a species that starts at 0 (see the note above
#     _MIN_ABSOLUTE_TOLERANCE). Every species that starts empty therefore gets
#     one number, scalar * its compartment volume -- 6.7e-11 at scalar 1e-9 for
#     the 0.067-volume endosome -- however small it later runs. After the FcRn
#     saturation front of a 60 mg/kg dose the endosomal antibody species sit at
#     1.6e-9 .. 1.3e-8, so that budget is 0.5-4% of their value, and CVODE fails
#     (CV_ILL_INPUT / CV_TOO_MUCH_ACC) ~200 h after the dose. Lowering the scalar
#     10x, pinning those five species at 1e-15, or capping every entry at 1e-12
#     each took the displaced-parameter failures from 6/40 to 0/40.
#
# Pinning keeps ONE reference -- the per-state scale RoadRunner derives at
# configure_integrator when the model is at its initial time -- and from it
# sets every entry explicitly before each block, each segment and each ladder
# attempt:
#
#     atol_i = clip( scalar * scale_i , floor , cap_ratio * scalar )
#
# *scalar* is the configured one, or the ladder's current (loosened) one, so a
# retry scales the same vector instead of re-deriving it. The cap is relative
# to the scalar for the same reason: an absolute cap would undo the ladder's
# loosening. cap_ratio is optional; without it pinning only restores the
# initial-derived vector.
#
# Opt in per replicate through the solver settings:
#
#     "tolerance_pinning": True,        # default TOLERANCE_PINNING (False)
#     "tolerance_cap_ratio": 1e-3,      # default TOLERANCE_CAP_RATIO (None)
#
# Both are plain values in the settings dict, so Profile_checkpoint's solver
# fingerprint already covers them: switching them on starts a new checkpoint
# directory rather than mixing two objectives. Left off, apply_tolerance_vector
# is floor_tolerance_vector and nothing changes.
TOLERANCE_PINNING = False
TOLERANCE_CAP_RATIO = None


def tolerance_policy(solver_settings):
    """``{"pin": bool, "cap_ratio": float | None}`` from *solver_settings*.

    ``tolerance_pinning`` must be a real bool: ``bool("false")`` is True, so a
    string from a config file would switch pinning on while reading as off.
    """
    pin = solver_settings.get("tolerance_pinning", TOLERANCE_PINNING)
    if not isinstance(pin, (bool, np.bool_)):
        raise TypeError(f"tolerance_pinning must be True or False, got {pin!r}")
    pin = bool(pin)
    cap = solver_settings.get("tolerance_cap_ratio", TOLERANCE_CAP_RATIO)
    if cap is not None:
        if isinstance(cap, (bool, np.bool_, str)):
            raise TypeError(f"tolerance_cap_ratio must be a number, got {cap!r}")
        cap = float(cap)
        if not cap > 0:
            raise ValueError(f"tolerance_cap_ratio must be positive, got {cap}")
        if not pin:
            raise ValueError("tolerance_cap_ratio needs tolerance_pinning: a cap is "
                             "applied to the pinned vector")
    return {"pin": pin, "cap_ratio": cap}


def _remember_tolerance_policy(r, solver_settings, scalar):
    """Cache the policy on *r* and, at the model's initial time, the reference scale.

    Called by configure_integrator right after the scalar is set, when RoadRunner
    has just derived the vector. The scale is only taken at time 0: a second
    configure on a state restored from the pre-dose cache would derive it from
    amounts that have already decayed, which is the drift this exists to undo.
    """
    policy = tolerance_policy(solver_settings)
    if not policy["pin"]:
        _model_cache.put(r, "_tolerance_policy", None)
        return
    # The model clock is the test for "still at the initial state": reset() and
    # resetAll() return it to exactly 0.0 whatever was simulated before (a model
    # has no other start time), and save/restore_model_state carry 'time' with
    # the amounts, so a restored state brings its clock back with it. A clock
    # that is not 0 means the model has been integrated, whatever its start.
    try:
        at_start = float(r.getCurrentTime()) == 0.0
    except Exception:
        at_start = False
    if at_start:
        # At the initial time the reference is re-derived every time, and a
        # failure to do so drops the old one: it belongs to an earlier parameter
        # vector, and pinning to it would silently use the wrong scales. (Away
        # from t=0 the stored reference is kept on purpose -- see above.)
        reference = None
        try:
            vec = np.asarray(r.integrator.getAbsoluteToleranceVector(), float)
            ids = state_vector_ids(r)
            n = min(len(ids), len(vec))
            if float(scalar) > 0 and n:
                reference = (list(ids[:n]), vec[:n] / float(scalar))
        except Exception:
            pass
        _model_cache.put(r, "_atol_reference", reference)
    if _model_cache.get(r, "_atol_reference") is None:
        # Never saw the initial state, so there is no initial scale to pin to.
        import warnings
        warnings.warn("tolerance_pinning requested but the model was first configured "
                      "away from its initial time; falling back to the floor",
                      RuntimeWarning, stacklevel=3)
        _model_cache.put(r, "_tolerance_policy", None)
        return
    _model_cache.put(r, "_tolerance_policy", policy)


def apply_tolerance_vector(r, scalar=None, floor=_MIN_ABSOLUTE_TOLERANCE):
    """Set the per-state absolute tolerances for the next integration.

    Pinning off (the default): exactly :func:`floor_tolerance_vector`. Pinning
    on: every entry is set to ``clip(scalar * scale_i, floor, cap_ratio * scalar)``
    from the reference scale. *scalar* defaults to the configured absolute
    tolerance; the retry ladder passes its current, loosened one. Returns the
    number of entries set.
    """
    policy = _model_cache.get(r, "_tolerance_policy")
    ref = _model_cache.get(r, "_atol_reference") if policy else None
    if ref is None:
        return floor_tolerance_vector(r, floor)
    if scalar is None:
        scalar = _model_cache.get(r, "_scalar_abs_tol")
    if scalar is None:
        return floor_tolerance_vector(r, floor)
    ids, scale = ref
    try:
        current = np.asarray(r.integrator.getAbsoluteToleranceVector(), float)
    except Exception:
        # A vector that cannot be read cannot be floored either (the floor reads
        # it first), so there is nothing to fall back to.
        return 0
    if len(current) < len(ids):
        return floor_tolerance_vector(r, floor)
    scalar = float(scalar)
    want = scalar * scale
    if policy["cap_ratio"] is not None:
        want = np.minimum(want, policy["cap_ratio"] * scalar)
    want = np.maximum(want, floor)
    n = 0
    failed = []
    for i in np.flatnonzero(~np.isclose(current[:len(ids)], want, rtol=1e-9, atol=0.0)):
        try:
            r.integrator.setIndividualTolerance(ids[i], float(want[i]))
            n += 1
        except Exception:
            failed.append(ids[i])
    if failed and not _model_cache.get(r, "_tolerance_set_warned"):
        # Once per model: a state left at its drifted tolerance is what pinning
        # exists to prevent, so it must not be silent, but this runs before
        # every block and would otherwise repeat for the whole fit.
        import warnings
        _model_cache.put(r, "_tolerance_set_warned", True)
        warnings.warn(f"tolerance_pinning could not set the absolute tolerance of "
                      f"{len(failed)} state(s) (first: {failed[0]!r}); they keep "
                      f"RoadRunner's derived value", RuntimeWarning, stacklevel=2)
    return n


class StackedResult(np.ndarray):
    def __new__(cls, input_array, colnames):
        obj = np.asarray(input_array).view(cls)
        obj.colnames = list(colnames)
        return obj

    def __contains__(self, key):
        return hasattr(self, 'colnames') and key in self.colnames

    def __getitem__(self, key):
        if isinstance(key, str):
            if hasattr(self, 'colnames') and key in self.colnames:
                return super().__getitem__((slice(None), self.colnames.index(key)))
            raise KeyError(f"Column '{key}' not in simulation results. "
                           f"Available: {getattr(self, 'colnames', [])}")
        return super().__getitem__(key)

def save_model_state(r):
    # Cache the state keys on the RoadRunner instance to minimize overhead.
    # Read through Model_cache, not r._cached_state_keys: a missed attribute on
    # a RoadRunner builds every selection id the model has (see Model_cache).
    keys = _model_cache.get(r, "_cached_state_keys")
    if keys is None:
        try:
            assignment_rules = set(r.getAssignmentRuleIds())
        except Exception:
            assignment_rules = set()
        keys = ['time']
        keys += [s for s in r.getFloatingSpeciesIds() if s not in assignment_rules]
        keys += [s for s in r.getBoundarySpeciesIds() if s not in assignment_rules]
        keys += [s for s in r.getGlobalParameterIds() if s not in assignment_rules]
        _model_cache.put(r, "_cached_state_keys", keys)

    values = _state_values(r, keys)
    if values is None:
        prev_selections = r.selections
        r.selections = keys
        values = list(r.getSelectedValues())
        r.selections = prev_selections

    return {
        "keys": keys,
        "values": values
    }


def _state_values(r, keys):
    """The values of *keys* read through the model's vector getters, or None.

    Reassigning ``r.selections`` recompiles every id on each call (twice per
    snapshot), which cost more than the integration of a short segment. The
    vectors need no selection. Which positions of them the keys occupy is worked
    out once per model; if a vector does not line up with its ids (a model with
    conserved moieties, say) the layout is cached as unusable and the caller
    falls back to the selection route.
    """
    layout = _model_cache.get(r, "_cached_state_layout")
    if layout is None:
        layout = False
        try:
            m = r.model
            sources = (("fl", r.getFloatingSpeciesIds(), m.getFloatingSpeciesAmounts),
                       ("bd", r.getBoundarySpeciesIds(), m.getBoundarySpeciesAmounts),
                       ("gp", r.getGlobalParameterIds(), m.getGlobalParameterValues))
            where, ok = {}, True
            for tag, ids, getter in sources:
                ids = list(ids)
                if len(getter()) != len(ids):
                    ok = False
                    break
                where.update({i: (tag, n) for n, i in enumerate(ids)})
            if ok and all(k == "time" or k in where for k in keys):
                layout = tuple(None if k == "time" else where[k] for k in keys)
        except Exception:
            layout = False
        _model_cache.put(r, "_cached_state_layout", layout)
    if layout is False:
        return None
    m = r.model
    vec = {"fl": m.getFloatingSpeciesAmounts(), "bd": m.getBoundarySpeciesAmounts(),
           "gp": m.getGlobalParameterValues()}
    time_now = float(m.getTime())
    return [time_now if w is None else float(vec[w[0]][w[1]]) for w in layout]

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


# ---------------------------------------------------------------------------
# Search mode: a bounded price for a bad parameter vector
# ---------------------------------------------------------------------------
#
# safe_simulate is built to rescue a run: when CVODE gives up it loosens the
# tolerances, doubles the step allowance up to 400,000 and, if that fails,
# halves the time span recursively to depth 4 with ten more attempts on every
# piece. That is the right behaviour for a vector someone cares about and the
# wrong one for a global search, which throws thousands of vectors at the model
# and expects most of them to be nonsense. Measured on aggregation_silk (the
# 6-decade box, 16 parameters): a healthy simulation takes 0.03-0.3 s and needs
# between 1,500 and 2,200 CVODE steps, while one box vertex failed its first
# attempt at the 200,000-step limit and was still climbing the ladder more than
# eight minutes later. Differential evolution samples exactly such points, so
# that one evaluation was the whole run.
#
# In search mode a simulation is one attempt with the step allowance capped, and
# a failure raises at once. The caller's existing handling turns that into the
# 1e10 failure value, so the search treats the vector as infeasible and moves on:
# 6.9 s for the vertex above at a 10,000-step cap, against minutes.
#
# It only ever REMOVES candidates. A value returned in search mode is the same
# fast-path value a normal run would have returned; nothing is computed with
# looser tolerances. The answer is polished in normal mode afterwards.
_SEARCH_MAX_STEPS = None


def set_search_mode(max_steps):
    """Cap every simulation at *max_steps* CVODE steps and never retry.

    ``None`` switches search mode off. Returns the previous setting so a caller
    can restore it. Process-global, like the integrator settings it governs:
    each pool worker sets its own (see pyantigen.engine.Evaluator._init_worker).
    """
    global _SEARCH_MAX_STEPS
    previous = _SEARCH_MAX_STEPS
    _SEARCH_MAX_STEPS = None if max_steps is None else int(max_steps)
    return previous


class search_mode:
    """``with search_mode(10000): ...`` -- search mode for the block."""

    def __init__(self, max_steps):
        self.max_steps = max_steps

    def __enter__(self):
        self._previous = set_search_mode(self.max_steps)
        return self

    def __exit__(self, *exc):
        set_search_mode(self._previous)
        return False


# ---------------------------------------------------------------------------
# Retry limits: a shorter ladder, but still a ladder
# ---------------------------------------------------------------------------
#
# Search mode (above) removes the ladder altogether. The multi-start stage wants
# something in between: most vectors it samples integrate on the first attempt,
# and a handful need a looser tolerance, but none is worth the full ladder of ten
# attempts and four levels of time-span subdivision that a vector somebody cares
# about gets. Limits cap both. With max_attempts=4 and max_depth=0 a vector gets
# the fast path plus three loosened retries, is never subdivided, and a failure
# raises -- which the objective turns into its failure value.
#
# Unlike search mode this does not cap the step allowance, so a vector that
# integrates slowly still integrates. Process-global, like search mode; the
# default (None) is the original behaviour, 10 attempts and depth 4.
_DEFAULT_MAX_ATTEMPTS = 10
_DEFAULT_MAX_DEPTH = 4
_RETRY_LIMITS = None


def set_retry_limits(max_attempts=None, max_depth=None):
    """Cap safe_simulate's ladder. ``None`` for both restores the defaults.

    max_attempts counts the first (fast-path) attempt, as the log's
    "Attempt n/N" does. Returns the previous setting so a caller can restore it.
    """
    global _RETRY_LIMITS
    previous = _RETRY_LIMITS
    if max_attempts is None and max_depth is None:
        _RETRY_LIMITS = None
        return previous
    attempts = _DEFAULT_MAX_ATTEMPTS if max_attempts is None else int(max_attempts)
    depth = _DEFAULT_MAX_DEPTH if max_depth is None else int(max_depth)
    if attempts < 1 or depth < 0:
        raise ValueError(f"retry limits need max_attempts >= 1 and max_depth >= 0, "
                         f"got {attempts} and {depth}")
    _RETRY_LIMITS = (attempts, depth)
    return previous


class retry_limits:
    """``with retry_limits(4, 0): ...`` -- the limits for the block."""

    def __init__(self, max_attempts=None, max_depth=None):
        self.max_attempts, self.max_depth = max_attempts, max_depth

    def __enter__(self):
        self._previous = set_retry_limits(self.max_attempts, self.max_depth)
        return self

    def __exit__(self, *exc):
        global _RETRY_LIMITS
        _RETRY_LIMITS = self._previous
        return False


# ---------------------------------------------------------------------------
# Relaxed tolerances: a cheap, coarse objective
# ---------------------------------------------------------------------------
#
# The multi-start triage scores every candidate once, to tell the mountains from
# the valleys, not to locate a minimum. For that the integrator can be run
# looser. The relaxation is applied where tolerances are applied -- in
# configure_integrator, which simulate() calls before every run and which
# re-derives them from each replicate's solver_settings -- so setting the
# integrator directly would be overwritten on the next call.
#
# Both tolerances are multiplied by *factor* and capped (the retry ladder's own
# ceiling is 1e-4), and it never TIGHTENS: a configured tolerance already above
# the cap is left alone. Process-global, like search mode and retry limits; the
# default (None) changes nothing.
#
# A state integrated under relaxed tolerances must never be mistaken for one
# integrated under the configured ones. The pre-dose cache keys on the
# tolerances, so it reads effective_tolerances() rather than the settings dict.
_TOL_RELAX = None


def set_tolerance_relaxation(factor=None, abs_cap=1e-4, rel_cap=1e-4):
    """Loosen the integrator tolerances by *factor*. ``None`` switches it off.
    Returns the previous setting so a caller can restore it."""
    global _TOL_RELAX
    previous = _TOL_RELAX
    if factor is None:
        _TOL_RELAX = None
        return previous
    factor = float(factor)
    if not factor >= 1.0:
        raise ValueError(f"tolerance relaxation factor must be >= 1, got {factor}")
    _TOL_RELAX = (factor, float(abs_cap), float(rel_cap))
    return previous


class relaxed_tolerances:
    """``with relaxed_tolerances(100): ...`` -- relaxed tolerances for the block."""

    def __init__(self, factor, abs_cap=1e-4, rel_cap=1e-4):
        self.args = (factor, abs_cap, rel_cap)

    def __enter__(self):
        self._previous = set_tolerance_relaxation(*self.args)
        return self

    def __exit__(self, *exc):
        global _TOL_RELAX
        _TOL_RELAX = self._previous
        return False


def effective_tolerances(solver_settings):
    """(absolute, relative) tolerance a run under *solver_settings* is held to,
    with any relaxation applied."""
    atol = float(solver_settings.get("absolute_tolerance", 1e-8))
    rtol = float(solver_settings.get("relative_tolerance", 1e-8))
    if _TOL_RELAX is not None:
        factor, abs_cap, rel_cap = _TOL_RELAX
        atol = max(atol, min(atol * factor, abs_cap))
        rtol = max(rtol, min(rtol * factor, rel_cap))
    return atol, rtol


def _run(r, start, end, points, times, observed_species):
    """One ``r.simulate`` call: a uniform grid, or exactly the *times* asked for.

    *times* (ascending, first entry the start) is what an event-free segment
    needs: output on the caller's grid inside an interval that was cut at a
    switch. It requires fixed-step output, which the segment runner sets.
    """
    if times is not None:
        return r.simulate(times=times, selections=observed_species)
    return r.simulate(start, end, points, observed_species)


def _simulate_search(r, start, end, points, observed_species, times=None):
    """One capped attempt. Raises on failure; leaves the step allowance as found."""
    original = r.integrator.maximum_num_steps
    # Never raise a model's own, lower limit: the cap is a ceiling.
    r.integrator.maximum_num_steps = int(min(original, _SEARCH_MAX_STEPS))
    try:
        res = _run(r, start, end, points, times, observed_species)
    finally:
        r.integrator.maximum_num_steps = original
    return res, {"start": start, "end": end, "attempts": 1, "subdivided": False,
                 "search_mode": True}


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

    # An explicit output grid (see _run). It fixes start, end and the number of
    # points, so the three values above are not needed in that case.
    times = solver_settings.get("times")
    if times is not None:
        times = [float(t) for t in times]
        if len(times) < 2:
            raise ValueError("'times' needs at least a start and an end")
        start, end, points = times[0], times[-1], len(times)

    # Ensure proper types
    start = float(start)
    end = float(end)
    points = int(points)

    if _SEARCH_MAX_STEPS is not None:
        return _simulate_search(r, start, end, points, observed_species, times=times)

    # Save original state before any simulation is attempted.
    # This is critical because if the fast path fails, it leaves the model in a
    # corrupted/advanced state that cannot be reverted without this snapshot.
    saved_state = save_model_state(r)

    # A sub-call of a subdivision begins wherever the previous one left the
    # vector: re-derived from the decayed state after a restore, or loosened by
    # a ladder. The top-level call is pinned by its caller (_run_blocks,
    # SegmentRunner); the halves are not, so pin them here, from the configured
    # scalar. Each half then tries the configured accuracy and runs its own
    # ladder if that fails.
    if depth > 0:
        apply_tolerance_vector(r)

    # Fast path: try once with whatever settings the model already has.
    try:
        res = _run(r, start, end, points, times, observed_species)
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
    orig_abs_tol = _model_cache.get(r, "_scalar_abs_tol")
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

    max_attempts, max_depth = _RETRY_LIMITS or (_DEFAULT_MAX_ATTEMPTS, _DEFAULT_MAX_DEPTH)
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
        apply_tolerance_vector(r, scalar=cur_abs_tol)
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
            res = _run(r, start, end, points, times, observed_species)

            # Restore original integrator settings on success
            r.integrator.absolute_tolerance = orig_abs_tol
            apply_tolerance_vector(r, scalar=orig_abs_tol)
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
    if depth < max_depth and points >= 2:
        mid = (start + end) / 2.0
        p1 = max(2, points // 2)
        p2 = max(2, points - p1)
        if times is not None:
            # Split the grid itself: the halves share one point, and a two-point
            # interval gets a midpoint so there is something to split at.
            if len(times) == 2:
                times = [times[0], mid, times[1]]
            cut = len(times) // 2
            mid = times[cut]
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
        if times is not None:
            block1["times"] = times[:cut + 1]
        res1, meta1 = safe_simulate(r, block1, observed_species, depth=depth+1, label=label)
        
        # Simulate second half starting from current state (which is at mid)
        block2 = {
            "start": mid,
            "end": end,
            "n_points": p2,
            "variable_step_size": variable_step_size
        }
        if times is not None:
            block2["times"] = times[cut:]
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
        apply_tolerance_vector(r, scalar=orig_abs_tol)
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
    apply_tolerance_vector(r, scalar=orig_abs_tol)
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
    # Before anything is touched: a bad pinning setting is a configuration
    # error, and should not leave the integrator half set up.
    tolerance_policy(solver_settings)
    solver_settings["integrator"] = solver_settings.get("integrator", "cvode")
    solver_settings["absolute_tolerance"] = solver_settings.get("absolute_tolerance", 1e-8)
    solver_settings["relative_tolerance"] = solver_settings.get("relative_tolerance", 1e-8)
    solver_settings["stiff"] = solver_settings.get("stiff", True)
    solver_settings["variable_step_size"] = solver_settings.get("variable_step_size", True)
    # solver_settings["initial_time_step"] = solver_settings.get("initial_time_step", 1e-6)
    solver_settings["maximum_num_steps"] = solver_settings.get("maximum_num_steps", 20000)
    r.setIntegrator(solver_settings["integrator"])
    # The settings dict keeps the CONFIGURED tolerances; what the integrator is
    # given is the effective pair, which differs only under relaxed_tolerances.
    atol, rtol = effective_tolerances(solver_settings)
    r.integrator.absolute_tolerance = atol
    # Cache the scalar where safe_simulate can recover it. The integrator's
    # own getter is unreliable for this: after floor_tolerance_vector uses
    # setIndividualTolerance it returns the per-species vector, whose min is
    # the 1e-30 floor rather than the configured accuracy.
    try:
        _model_cache.put(r, "_scalar_abs_tol", float(atol))
    except Exception:
        pass
    r.integrator.relative_tolerance = rtol
    r.integrator.setValue('stiff', solver_settings["stiff"])
    r.integrator.variable_step_size = solver_settings["variable_step_size"]
    # r.integrator.setValue('initial_time_step', solver_settings["initial_time_step"])
    r.integrator.setValue('maximum_num_steps', solver_settings["maximum_num_steps"])
    _remember_tolerance_policy(r, solver_settings, atol)
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
    _available = _model_cache.get(r, "_available_symbols")
    if _available is None:
        _ids = (set(r.getFloatingSpeciesIds()) | set(r.getBoundarySpeciesIds())
                | set(r.getAssignmentRuleIds()) | set(r.getGlobalParameterIds())
                | set(r.getReactionIds()))
        _available = {'time'} | _ids | {'[' + s + ']' for s in _ids}
        _model_cache.put(r, "_available_symbols", _available)
    observed_species = [s for s in observed_species if s in _available]

    # An InputSchedule replaces the model's events: the runner applies each step
    # at its switch time and integrates the stretches between them.
    runner = None
    schedule = solver_settings.get("schedule")
    if schedule is not None:
        from pyantigen.engine.Segmented import SegmentRunner
        runner = SegmentRunner(r, schedule)

    # Find index of the last tracked block so we can skip trailing untracked ones.
    last_tracked_idx = -1
    for i, blk in enumerate(block_list):
        if blk.get('tracked', True):
            last_tracked_idx = i

    try:
        return _run_blocks(r, block_list, last_tracked_idx, observed_species,
                           label, runner)
    finally:
        if runner is not None:
            runner.close()


def _run_blocks(r, block_list, last_tracked_idx, observed_species, label, runner):
    all_results = None
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
        apply_tolerance_vector(r)

        try:
            if runner is not None:
                res = runner.run_block(block, observed_species, label=label)
            else:
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

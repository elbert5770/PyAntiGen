"""Pinning the per-species absolute tolerances, as opposed to flooring them.

Run from the repository root:
    python -m pytest tests/engine/test_tolerance_pinning.py

floor_tolerance_vector only raises entries that have fallen below the floor. Two
things it leaves alone are what these tests pin down:

* setting the scalar (which the retry ladder does before every attempt) re-derives
  the whole vector from the amounts the model holds at that moment, so a loosened
  retry runs at tolerances taken from the failing state rather than the initial one;
* RoadRunner's initial-derived scale gives every species that starts at 0 the same
  number, scalar * its compartment's volume, however small it later runs.

apply_tolerance_vector, with ``tolerance_pinning`` on, sets every entry from the
scale RoadRunner derived at the model's initial time. Off, it is the floor.
"""
import os
import sys
import warnings

import numpy as np
import pytest

sys.path.insert(0, os.getcwd())

tellurium = pytest.importorskip("tellurium")
te = tellurium

import pyantigen.engine.Simulate as S                                    # noqa: E402
from pyantigen.engine import Model_cache as cache                        # noqa: E402
from pyantigen.engine.Simulate import (                                  # noqa: E402
    _MIN_ABSOLUTE_TOLERANCE,
    apply_tolerance_vector,
    configure_integrator,
    simulate,
    tolerance_policy,
)

VOLUME = 3.0


def model():
    """A decaying chain with one species starting empty (E) and one tiny (D)."""
    r = te.loada(f"""model m()
  compartment C = {VOLUME};
  species A in C = 1.0, B in C = 1e-3, D in C = 1e-9, E in C = 0.0;
  k = 1e-1;
  J1: A -> B; k*A;
  J2: B -> D; k*B;
end""")
    r.setIntegrator('cvode')
    return r


def settings(**extra):
    s = {"integrator": "cvode", "absolute_tolerance": 1e-9, "relative_tolerance": 1e-9,
         "stiff": True, "variable_step_size": True, "maximum_num_steps": 20000,
         "simulation_blocks": {"b1": {"start": 0, "end": 40, "n_points": 20, "tracked": True}}}
    s.update(extra)
    return s


def atol(r):
    return np.asarray(r.integrator.getAbsoluteToleranceVector(), float)


def initial_scale(r):
    """scalar-free scale RoadRunner's rule gives at t=0: amount, or volume where the amount is 0."""
    out = []
    for sid in r.model.getFloatingSpeciesIds():
        amount = float(r[f"[{sid}]"]) * VOLUME
        out.append(amount if amount > 0 else VOLUME)
    return np.array(out)


# ---------------------------------------------------------------------------
# policy
# ---------------------------------------------------------------------------

def test_policy_defaults_to_off():
    assert tolerance_policy({}) == {"pin": False, "cap_ratio": None}


def test_policy_reads_the_settings_keys():
    assert tolerance_policy({"tolerance_pinning": True, "tolerance_cap_ratio": 1e-3}) \
        == {"pin": True, "cap_ratio": 1e-3}


def test_a_cap_without_pinning_is_refused():
    with pytest.raises(ValueError, match="needs tolerance_pinning"):
        tolerance_policy({"tolerance_cap_ratio": 1e-3})


@pytest.mark.parametrize("value", ["false", "False", "0", 0, 1, None])
def test_pinning_must_be_a_real_bool(value):
    """bool("false") is True: a string from a config file must not switch pinning on."""
    with pytest.raises(TypeError, match="tolerance_pinning"):
        tolerance_policy({"tolerance_pinning": value})


@pytest.mark.parametrize("value", ["1e-3", True])
def test_the_cap_must_be_a_number(value):
    with pytest.raises(TypeError, match="tolerance_cap_ratio"):
        tolerance_policy({"tolerance_pinning": True, "tolerance_cap_ratio": value})


def test_numpy_bools_are_accepted():
    assert tolerance_policy({"tolerance_pinning": np.True_})["pin"] is True


def test_a_bad_setting_fails_before_the_integrator_is_touched():
    r = model()
    r.integrator.absolute_tolerance = 1e-3
    with pytest.raises(ValueError, match="needs tolerance_pinning"):
        configure_integrator(r, settings(tolerance_cap_ratio=1e-3))
    assert r.integrator.absolute_tolerance == pytest.approx(1e-3)


def test_a_non_positive_cap_is_refused():
    with pytest.raises(ValueError, match="positive"):
        tolerance_policy({"tolerance_pinning": True, "tolerance_cap_ratio": 0.0})


# ---------------------------------------------------------------------------
# off: exactly the floor
# ---------------------------------------------------------------------------

def test_off_is_the_floor_and_stores_nothing():
    r = model()
    configure_integrator(r, settings())
    assert cache.get(r, "_tolerance_policy") is None
    before = atol(r).copy()
    r.integrator.setIndividualTolerance("E", 1e-40)          # below the production floor
    n = apply_tolerance_vector(r)
    after = atol(r)
    assert n == 1
    ids = list(r.model.getFloatingSpeciesIds())
    assert after[ids.index("E")] == _MIN_ABSOLUTE_TOLERANCE
    keep = [i for i, s in enumerate(ids) if s != "E"]
    assert np.array_equal(after[keep], before[keep]), "the floor must touch only what is below it"


# ---------------------------------------------------------------------------
# on: the reference, and re-pinning it
# ---------------------------------------------------------------------------

def test_reference_is_roadrunners_initial_derivation():
    r = model()
    configure_integrator(r, settings(tolerance_pinning=True))
    ids, scale = cache.get(r, "_atol_reference")
    assert ids == list(r.model.getFloatingSpeciesIds())
    assert np.allclose(scale, initial_scale(r), rtol=1e-9)
    # E starts empty: it gets the volume, the same number as any other empty species.
    assert scale[ids.index("E")] == pytest.approx(VOLUME)


def test_apply_restores_a_drifted_vector():
    r = model()
    configure_integrator(r, settings(tolerance_pinning=True))
    want = atol(r).copy()
    ids = list(r.model.getFloatingSpeciesIds())
    r.integrator.setIndividualTolerance("A", 1e-40)          # drifted down
    r.integrator.setIndividualTolerance("D", 5.0)            # drifted up
    n = apply_tolerance_vector(r)
    assert n == 2
    assert np.allclose(atol(r), want, rtol=1e-9)
    assert atol(r)[ids.index("D")] == pytest.approx(want[ids.index("D")])


def test_a_loosened_scalar_scales_the_reference_not_the_failing_state():
    """The ladder's situation: the scalar is set, which re-derives from the current state."""
    r = model()
    configure_integrator(r, settings(tolerance_pinning=True))
    ids = list(r.model.getFloatingSpeciesIds())
    scale0 = initial_scale(r)
    r.simulate(0, 40, 10)                                    # amounts move well away from t=0
    loosened = 1e-7
    r.integrator.absolute_tolerance = loosened               # what the ladder does
    derived = atol(r).copy()
    assert not np.allclose(derived, loosened * scale0, rtol=1e-3), \
        "the re-derived vector should differ from the initial one, or this test proves nothing"
    apply_tolerance_vector(r, scalar=loosened)
    assert np.allclose(atol(r), loosened * scale0, rtol=1e-9)
    # a species that has decayed far from its start keeps its initial-derived tolerance
    assert atol(r)[ids.index("A")] == pytest.approx(loosened * 1.0 * VOLUME)


def test_scalar_defaults_to_the_configured_one():
    r = model()
    configure_integrator(r, settings(tolerance_pinning=True, absolute_tolerance=1e-8))
    r.integrator.absolute_tolerance = 1e-12                  # something else sets it
    apply_tolerance_vector(r)
    assert np.allclose(atol(r), 1e-8 * initial_scale(r), rtol=1e-9)


# ---------------------------------------------------------------------------
# the cap
# ---------------------------------------------------------------------------

def test_cap_lowers_large_entries_and_leaves_small_ones():
    r = model()
    configure_integrator(r, settings(tolerance_pinning=True, tolerance_cap_ratio=1e-1))
    apply_tolerance_vector(r)
    scalar, cap = 1e-9, 1e-1
    want = np.maximum(np.minimum(scalar * initial_scale(r), cap * scalar), _MIN_ABSOLUTE_TOLERANCE)
    assert np.allclose(atol(r), want, rtol=1e-9)
    ids = list(r.model.getFloatingSpeciesIds())
    # D starts at 1e-9 * 3 = 3e-9, so scale 3e-9 < cap: untouched by the cap.
    assert atol(r)[ids.index("D")] == pytest.approx(scalar * 3e-9)
    # A (scale 3) and E (scale = volume 3) are capped.
    assert atol(r)[ids.index("A")] == pytest.approx(cap * scalar)
    assert atol(r)[ids.index("E")] == pytest.approx(cap * scalar)


def test_the_cap_follows_the_scalar_so_the_ladder_can_still_loosen():
    r = model()
    configure_integrator(r, settings(tolerance_pinning=True, tolerance_cap_ratio=1e-1))
    apply_tolerance_vector(r, scalar=1e-6)
    ids = list(r.model.getFloatingSpeciesIds())
    assert atol(r)[ids.index("A")] == pytest.approx(1e-1 * 1e-6)


# ---------------------------------------------------------------------------
# when there is no initial state to pin to
# ---------------------------------------------------------------------------

def test_first_configured_away_from_t0_falls_back_to_the_floor():
    r = model()
    r.simulate(0, 5, 3)                                      # the model is no longer at its initial time
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        configure_integrator(r, settings(tolerance_pinning=True))
    assert any("initial time" in str(x.message) for x in w)
    assert cache.get(r, "_tolerance_policy") is None
    assert cache.get(r, "_atol_reference") is None


def test_a_later_configure_away_from_t0_keeps_the_reference():
    """The pre-dose cache restores a decayed state and configures again."""
    r = model()
    configure_integrator(r, settings(tolerance_pinning=True))
    ref = cache.get(r, "_atol_reference")
    r.simulate(0, 40, 10)
    configure_integrator(r, settings(tolerance_pinning=True))
    assert cache.get(r, "_atol_reference") is ref
    assert cache.get(r, "_tolerance_policy") is not None


def test_a_reference_that_cannot_be_rederived_at_t0_is_dropped():
    """A stale scale from an earlier parameter vector must not survive a failed refresh."""
    r = model()
    configure_integrator(r, settings(tolerance_pinning=True))
    assert cache.get(r, "_atol_reference") is not None
    r.reset()                                                # back at t=0
    real = S.state_vector_ids
    S.state_vector_ids = lambda rr: (_ for _ in ()).throw(RuntimeError("no ids"))
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            configure_integrator(r, settings(tolerance_pinning=True))
    finally:
        S.state_vector_ids = real
    assert cache.get(r, "_atol_reference") is None
    assert cache.get(r, "_tolerance_policy") is None
    assert any("initial time" in str(x.message) for x in w)


def test_the_ladder_leaves_the_vector_pinned_after_restoring_the_scalar():
    """Otherwise the next sub-call's fast path runs on a vector re-derived from the decayed state."""
    r = model()
    configure_integrator(r, settings(tolerance_pinning=True))
    want = atol(r).copy()
    real = S._run
    state = {"n": 0}

    def flaky(*a, **k):
        state["n"] += 1
        if state["n"] == 1:
            raise RuntimeError("CVODE Error: CV_ERR_FAILURE")
        return real(*a, **k)

    S._run = flaky
    try:
        S.safe_simulate(r, {"start": 0, "end": 40, "n_points": 10, "variable_step_size": True},
                        ["time", "[A]"])
    finally:
        S._run = real
    assert state["n"] >= 2, "the ladder should have run"
    assert np.allclose(atol(r), want, rtol=1e-9)


def test_the_second_half_of_a_subdivision_starts_on_the_pinned_vector():
    """The first half restores the scalar on success; the second half's fast path must not see the re-derived vector."""
    r = model()
    configure_integrator(r, settings(tolerance_pinning=True))
    want = atol(r).copy()
    real = S._run
    seen = {}

    def flaky(rr, start, end, *a, **k):
        if (start, end) == (0, 40):
            raise RuntimeError("CVODE Error: CV_ERR_FAILURE")    # every ladder attempt fails -> subdivide
        if start > 0:
            seen[start] = atol(rr).copy()                        # what the second half's first attempt runs on
        return real(rr, start, end, *a, **k)

    S._run = flaky
    try:
        _, meta = S.safe_simulate(r, {"start": 0, "end": 40, "n_points": 10, "variable_step_size": True},
                                  ["time", "[A]"])
    finally:
        S._run = real
    assert meta["subdivided"]
    assert list(seen) == [20.0]
    assert np.allclose(seen[20.0], want, rtol=1e-9)
    assert np.allclose(atol(r), want, rtol=1e-9)


def test_the_standalone_predict_helper_drops_a_pinning_policy():
    """It takes no solver settings, so it cannot ask for pinning; a leftover policy would use a stale reference."""
    from pyantigen.engine.Optimize import predict_concentrations
    r = model()
    configure_integrator(r, settings(tolerance_pinning=True))
    assert cache.get(r, "_tolerance_policy") is not None
    predict_concentrations([0.2], {"A": {"times": np.array([])}}, r, ["k"])
    assert cache.get(r, "_tolerance_policy") is None
    # and the floor path is what runs now
    r.integrator.setIndividualTolerance("E", 1e-40)
    assert apply_tolerance_vector(r) == 1


def test_a_state_that_cannot_be_pinned_warns_once_per_model():
    r = model()
    configure_integrator(r, settings(tolerance_pinning=True))
    ids, scale = cache.get(r, "_atol_reference")
    bad = list(ids)
    bad[ids.index("A")] = "not_a_state"                      # setIndividualTolerance will refuse it
    cache.put(r, "_atol_reference", (bad, scale))
    r.integrator.setIndividualTolerance("A", 7.0)            # drifted, so the bad id is attempted
    r.integrator.setIndividualTolerance("B", 5.0)            # drifted, and settable
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        n = apply_tolerance_vector(r)
        assert n >= 1, "the other states are still pinned"
        r.integrator.setIndividualTolerance("D", 5.0)
        apply_tolerance_vector(r)
    msgs = [str(x.message) for x in w if "could not set" in str(x.message)]
    assert len(msgs) == 1 and "not_a_state" in msgs[0]


def test_turning_it_off_again_clears_the_policy():
    r = model()
    configure_integrator(r, settings(tolerance_pinning=True))
    configure_integrator(r, settings())
    assert cache.get(r, "_tolerance_policy") is None


# ---------------------------------------------------------------------------
# wiring through simulate() and the ladder
# ---------------------------------------------------------------------------

def test_simulate_pins_once_per_block():
    r = model()
    calls = []
    real = S.apply_tolerance_vector
    S.apply_tolerance_vector = lambda rr, *a, **k: (calls.append((a, k)), real(rr, *a, **k))[1]
    try:
        s = settings(tolerance_pinning=True)
        s["simulation_blocks"] = {f"b{i}": {"start": 5 * i, "end": 5 * (i + 1), "n_points": 5, "tracked": True}
                                  for i in range(3)}
        res = simulate(r, s, ["time", "A"])
    finally:
        S.apply_tolerance_vector = real
    assert len(calls) == 3 and res is not None and len(res) > 0


def test_the_ladder_pins_at_its_own_loosened_scalar():
    r = model()
    scalars = []
    real = S.apply_tolerance_vector

    def spy(rr, *a, **k):
        scalars.append(k.get("scalar"))
        return real(rr, *a, **k)
    S.apply_tolerance_vector = spy
    try:
        s = settings(tolerance_pinning=True, maximum_num_steps=1)   # the first attempt cannot finish
        s["simulation_blocks"] = {"b1": {"start": 0, "end": 1000, "n_points": 50, "tracked": True}}
        try:
            simulate(r, s, ["time", "A"])
        except Exception:
            pass                                             # whether the ladder finishes is not the point
    finally:
        S.apply_tolerance_vector = real
    retries = [x for x in scalars if x is not None]
    assert retries, "the ladder never re-applied the vector"
    assert all(x >= 1e-9 for x in retries) and max(retries) > 1e-9, f"{retries}"

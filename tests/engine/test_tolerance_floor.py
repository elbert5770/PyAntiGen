"""Checks for the per-species absolute-tolerance floor.

Run from the repository root:
    python tests/engine/test_tolerance_floor.py

RoadRunner scales each species' absolute tolerance by that species' current
amount, with no lower bound, so a species decaying toward zero drags its own
tolerance below DBL_MIN and CVODE then refuses the call outright. These run
against small models where the amounts are known, so a failure is the floor and
not the biology.

The first test pins the scaling rule itself. If a future RoadRunner changes it,
that test fails and says so, rather than the floor quietly becoming a no-op.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.getcwd())

import tellurium as te                                            # noqa: E402

from pyantigen.engine.Simulate import (                                     # noqa: E402
    _DUST_THRESHOLD,
    _MIN_ABSOLUTE_TOLERANCE,
    _scalar_tolerance,
    floor_tolerance_vector,
    safe_simulate,
    simulate,
    state_vector_ids,
)

failures = []


def check(name, cond, detail=""):
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}  {detail}")
        failures.append(name)


def model(volume=3.0):
    r = te.loada(f"""model m()
  compartment C = {volume};
  species A in C = 1.0, B in C = 1e-6, D in C = 1e-18, E in C = 0.0;
  k = 1e-3;
  J1: A -> B; k*A;
  J2: B -> D; k*B;
end""")
    r.setIntegrator('cvode')
    return r


def atol(r):
    return np.asarray(r.integrator.getAbsoluteToleranceVector(), float)


# ---------------------------------------------------------------------------

def rate_rule_model():
    """Two species and three rate-rule variables.

    Rate-rule variables are state variables: CVODE integrates them and its
    error test covers them, so their tolerances collapse toward zero exactly as
    a decaying species' does. They are *not* floating species, which is what
    made them easy to miss.
    """
    r = te.loada("""model m()
  compartment C = 3.0;
  species A in C = 1.0, B in C = 1e-6;
  k = 1e-3;
  V = 0.0;  W = 0.0;  Z = 0.0;
  V' = k;   W' = 2*k;  Z' = 3*k;
  J1: A -> B; k*A;
end""")
    r.setIntegrator('cvode')
    return r


def test_state_vector_is_more_than_the_species():
    """The tolerance vector is longer than getFloatingSpeciesIds().

    This is the whole reason state_vector_ids exists. Zipping the vector
    against the species list stops at the shorter of the two, so every
    rate-rule entry falls off the end and is never floored -- silently, on a
    model where nothing about the call looks wrong.
    """
    print("\nState vector vs floating species:")
    r = rate_rule_model()
    tols = list(r.integrator.getAbsoluteToleranceVector())
    species = list(r.model.getFloatingSpeciesIds())
    ids = state_vector_ids(r)

    check("the vector is longer than the species list",
          len(tols) > len(species), f"{len(tols)} vs {len(species)}")
    check("state_vector_ids covers the whole vector",
          len(ids) == len(tols), f"{len(ids)} ids vs {len(tols)} tolerances")
    check("the species come first, in their own order",
          ids[:len(species)] == species, str(ids))

    # Order is not an assumption: set a distinct tolerance per id and read the
    # vector back. If RoadRunner ever reorders its state vector this fails and
    # says so, rather than the floor quietly landing on the wrong variables.
    for i, sid in enumerate(ids):
        r.integrator.setIndividualTolerance(sid, 10.0 ** -(20 + i))
    got = [f"{v:.0e}" for v in r.integrator.getAbsoluteToleranceVector()]
    want = [f"{10.0 ** -(20 + i):.0e}" for i in range(len(ids))]
    check("each id maps to the vector slot state_vector_ids implies",
          got == want, f"{got} vs {want}")


def test_floor_reaches_rate_rule_variables():
    """A rate-rule entry below the floor must be raised like any other."""
    print("\nFlooring a rate-rule variable:")
    r = rate_rule_model()
    ids = state_vector_ids(r)
    rate_rule_ids = [i for i in ids if i not in
                     set(r.model.getFloatingSpeciesIds())]
    check("the model has rate-rule state to test", bool(rate_rule_ids),
          str(ids))

    # Drive one under the floor the way a long decay would.
    r.integrator.setIndividualTolerance(rate_rule_ids[0], 1e-40)
    before = list(r.integrator.getAbsoluteToleranceVector())
    check("it starts below the floor",
          min(before) < _MIN_ABSOLUTE_TOLERANCE, f"{min(before):.2e}")

    n = floor_tolerance_vector(r)
    after = list(r.integrator.getAbsoluteToleranceVector())
    check("the floor raises it", n >= 1, f"raised {n}")
    check("and nothing is left below the floor",
          min(after) >= _MIN_ABSOLUTE_TOLERANCE, f"{min(after):.2e}")
    check("the species entries are untouched",
          before[:len(r.model.getFloatingSpeciesIds())]
          == after[:len(r.model.getFloatingSpeciesIds())],
          "flooring must only raise what is below the floor")


def test_scaling_rule():
    """atol_i = scalar * amount_i, and scalar * volume where the amount is 0.

    This is the behaviour the floor exists to correct. Pinning it means a
    RoadRunner that changes the rule breaks a test here instead of silently
    making the floor pointless.
    """
    print("\nRoadRunner's scaling rule:")
    V = 3.0
    for scalar in (1e-6, 1e-9, 1e-12):
        r = model(V)
        r.integrator.absolute_tolerance = scalar
        ids = r.model.getFloatingSpeciesIds()
        got = dict(zip(ids, atol(r)))
        ok = True
        for sid in ids:
            amount = float(r[f"[{sid}]"]) * V
            want = scalar * (amount if amount > 0 else V)
            if not np.isclose(got[sid], want, rtol=1e-9):
                ok = False
                print(f"        {sid}: got {got[sid]:.4g}, expected {want:.4g}")
        check(f"scalar={scalar:g}: atol tracks amount * scalar", ok, "")


def test_floor_raises_only_what_is_below_it():
    print("\nWhat the floor touches:")
    r = model()
    r.integrator.absolute_tolerance = 1e-9
    before = atol(r).copy()
    ids = r.model.getFloatingSpeciesIds()
    floor = 1e-20

    n = floor_tolerance_vector(r, floor)
    after = atol(r)

    expected = int((before < floor).sum())
    check("it raises exactly the entries below the floor", n == expected,
          f"raised {n}, expected {expected}")
    check("nothing ends up below the floor", (after >= floor).all(),
          f"min {after.min():.3g}")
    for i, sid in enumerate(ids):
        if before[i] >= floor:
            check(f"{sid} above the floor is untouched",
                  after[i] == before[i], f"{before[i]:.3g} -> {after[i]:.3g}")


def test_floor_is_idempotent_and_free_on_a_fresh_model():
    """Flooring once at setup achieves nothing -- that is the point.

    The vector is healthy at t=0 and degrades as the trajectory does, which is
    why the floor is applied before every block rather than once per run.
    """
    print("\nOn a freshly reset model:")
    # Deliberately not the shared model(): that one carries D at 1e-18 to give
    # the other tests something small to floor, and at scalar 1e-9 its
    # tolerance starts at 3e-27 -- below the production floor before the
    # trajectory has moved at all. That is a property of the fixture, not of a
    # real model: on the SILK and antibody models the vector's minimum after
    # reset() is 6.6e-13, nine orders above the floor. Using species that are
    # all comfortably above it tests the claim this is here to make.
    r = te.loada("""model m()
  compartment C = 3.0;
  species A in C = 1.0, B in C = 1e-6, E in C = 0.0;
  k = 1e-3;
  J1: A -> B; k*A;
end""")
    r.setIntegrator('cvode')
    r.integrator.absolute_tolerance = 1e-9
    n = floor_tolerance_vector(r)
    check("nothing is below the production floor at t=0", n == 0, f"{n} raised")
    before = atol(r).copy()
    floor_tolerance_vector(r)
    check("a second pass changes nothing",
          np.array_equal(before, atol(r)), "")


def test_floor_survives_simulate_and_setintegrator():
    """The two operations it has to outlive, and the one it does not.

    The probe floor is deliberately absurd -- 1e-3, far above anything the
    scaling rule would ever produce here -- so that "still floored" and "back to
    scalar * amount" cannot be confused. A probe near the scaled values cannot
    tell them apart, because the rescaled minimum may land above it by chance.
    """
    print("\nPersistence:")
    PROBE = 1e-3
    r = model()
    r.integrator.absolute_tolerance = 1e-9
    floor_tolerance_vector(r, PROBE)
    check("the probe floor applies", (atol(r) >= PROBE).all(),
          f"min {atol(r).min():.3g}")

    r.simulate(0, 10, 5)
    check("survives simulate()", (atol(r) >= PROBE).all(),
          f"min {atol(r).min():.3g}")

    r.setIntegrator('cvode')
    check("survives setIntegrator()", (atol(r) >= PROBE).all(),
          f"min {atol(r).min():.3g}")

    r.integrator.absolute_tolerance = 1e-9
    check("is wiped by re-setting the scalar, as documented",
          atol(r).min() < PROBE,
          f"min {atol(r).min():.3g} -- if this now survives, the re-flooring "
          f"in safe_simulate's retry loop is redundant")


def test_floor_sits_below_anything_physical():
    """It must not be able to loosen the tolerance on a real quantity.

    This test used to require the floor to sit *below* ``_DUST_THRESHOLD``, and
    that was guarding the wrong thing. What can be harmed by a floor that is too
    high is a state value carrying a result, and the smallest of those in this
    model is 1e-15 (AB40_DensePlaqueNumber_BrainISF, per the note on
    _DUST_THRESHOLD). Values below the dust threshold are zeroed by
    ``clamp_state_dust`` before the integrator ever sees them, so a floor above
    that threshold cannot loosen anything that survives to be integrated.

    Keeping the old requirement forced the floor down to a region where it stops
    doing its job: measured on the antibody arms, 1e-30 leaves the tolerance
    minima at 1e-24 to 1e-29 untouched -- above the floor, below what CVODE can
    work with -- and eleven blocks per evaluation fall into the retry ladder.
    The invariant that matters is headroom below a real quantity, which is what
    is checked now.
    """
    print("\nHeadroom:")
    # Six orders below the smallest genuine quantity: at 1e-15 the absolute
    # tolerance still resolves six significant figures, so nothing that carries
    # a result is being coarsened.
    check("the floor is at least six orders below the smallest real "
          "quantity (1e-15)",
          _MIN_ABSOLUTE_TOLERANCE <= 1e-15 * 1e-6,
          f"{_MIN_ABSOLUTE_TOLERANCE:g}")
    check("but comfortably above subnormal doubles",
          _MIN_ABSOLUTE_TOLERANCE > 1e-300, "")
    # Not a hazard, but worth stating so the relationship is visible rather
    # than assumed either way: dust is zeroed before integration.
    print(f"       floor {_MIN_ABSOLUTE_TOLERANCE:g}, dust threshold "
          f"{_DUST_THRESHOLD:g} -- values below the latter are zeroed by "
          f"clamp_state_dust before the integrator sees them")


def test_simulate_applies_it_per_block():
    """The wiring: a multi-block run must floor before each block."""
    print("\nWiring into simulate():")
    r = model()
    calls = []
    real = floor_tolerance_vector

    import pyantigen.engine.Simulate as S
    def spy(rr, floor=_MIN_ABSOLUTE_TOLERANCE):
        calls.append(floor)
        return real(rr, floor)
    S.floor_tolerance_vector = spy
    try:
        settings = {
            "integrator": "cvode", "absolute_tolerance": 1e-9,
            "relative_tolerance": 1e-9, "stiff": True,
            "variable_step_size": True, "maximum_num_steps": 20000,
            "simulation_blocks": {
                "b1": {"start": 0, "end": 5, "n_points": 10, "tracked": True},
                "b2": {"start": 5, "end": 10, "n_points": 10, "tracked": True},
                "b3": {"start": 10, "end": 15, "n_points": 10, "tracked": True},
            },
        }
        res = simulate(r, settings, ["time", "A", "B"])
    finally:
        S.floor_tolerance_vector = real
    check("called once per block", len(calls) == 3, f"{len(calls)} call(s)")
    check("with the production floor",
          all(c == _MIN_ABSOLUTE_TOLERANCE for c in calls), f"{calls}")
    check("and the run still produces results", res is not None and len(res) > 0,
          "")


def test_getter_returns_a_list_after_flooring():
    """The side effect that has to be defended against, pinned.

    Once setIndividualTolerance has been used, RoadRunner's absolute_tolerance
    GETTER stops returning the scalar and returns the whole vector as a list.
    The retry ladder then does min(cur_abs_tol * 10, 1e-4) on a list -- where
    `* 10` is list repetition -- and raises "'<' not supported between
    instances of 'float' and 'list'", which surfaces as an integration failure
    with a nonsense message. That is a real crash this cost a run to find.
    """
    print("\nThe getter side effect:")
    r = model()
    r.integrator.absolute_tolerance = 1e-9
    check("the getter returns a scalar before flooring",
          isinstance(r.integrator.absolute_tolerance, float), "")

    # 1e-20 is above this model's smallest entry (D sits at 3e-27), so the
    # override actually fires. The production floor of 1e-30 would not touch
    # anything here, and a test that never calls setIndividualTolerance would
    # not be testing the side effect at all.
    n = floor_tolerance_vector(r, 1e-20)
    check("the probe floor actually overrode something", n > 0, f"{n} raised")
    got = r.integrator.absolute_tolerance
    if isinstance(got, float):
        print("        (this RoadRunner keeps the getter scalar; the coercion "
              "below is then belt-and-braces rather than load-bearing)")
    else:
        check("it returns a sequence after flooring",
              not isinstance(got, (float, int)), f"{type(got).__name__}")
        raised = False
        try:
            min(got * 10, 1e-4)
        except TypeError:
            raised = True
        check("raw arithmetic on it raises, as the ladder used to", raised, "")

    check("_scalar_tolerance coerces it to the tightest entry",
          _scalar_tolerance(got) == min(got) if not isinstance(got, float)
          else _scalar_tolerance(got) == got, f"{_scalar_tolerance(got)}")
    check("_scalar_tolerance passes a plain float through",
          _scalar_tolerance(1e-9) == 1e-9, "")
    check("and falls back on something it cannot read",
          _scalar_tolerance(object(), fallback=1e-7) == 1e-7, "")


def test_retry_ladder_survives_a_floored_model():
    """End to end: a failing block on a floored model must not crash.

    Forces the fast path to fail by asking for an impossible step count, so the
    ladder runs with individual tolerances already set.
    """
    print("\nThe retry ladder on a floored model:")
    r = model()
    r.integrator.absolute_tolerance = 1e-9
    r.integrator.relative_tolerance = 1e-9
    r.integrator.setValue('maximum_num_steps', 1)   # guarantees a failure
    # 1e-20 so the override fires on this model; see the note above.
    floor_tolerance_vector(r, 1e-20)
    blk = {"start": 0, "end": 1000, "n_points": 50, "variable_step_size": True}
    try:
        res, meta = safe_simulate(r, blk, ["time", "A"], label=None)
        ok, detail = res is not None, ""
    except TypeError as exc:
        ok, detail = False, f"TypeError leaked from the ladder: {exc}"
    except Exception as exc:
        # Any other failure is the model's business, not the coercion's.
        ok, detail = True, f"(non-TypeError: {type(exc).__name__})"
    check("no TypeError escapes the ladder", ok, detail)


def main():
    print("=" * 72)
    print("Per-species absolute tolerance floor")
    print("=" * 72)
    test_scaling_rule()
    test_state_vector_is_more_than_the_species()
    test_floor_reaches_rate_rule_variables()
    test_floor_raises_only_what_is_below_it()
    test_floor_is_idempotent_and_free_on_a_fresh_model()
    test_floor_survives_simulate_and_setintegrator()
    test_floor_sits_below_anything_physical()
    test_simulate_applies_it_per_block()
    test_getter_returns_a_list_after_flooring()
    test_retry_ladder_survives_a_floored_model()

    print("\n" + "=" * 72)
    if failures:
        print(f"FAILED ({len(failures)}):")
        for f in failures:
            print("  -", f)
        return 1
    print("ALL TOLERANCE FLOOR CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())

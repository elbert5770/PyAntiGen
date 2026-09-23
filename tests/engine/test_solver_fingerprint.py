"""Solver settings are part of the objective, so they belong in the fingerprint.

Run from the repository root:
    python tests/engine/test_solver_fingerprint.py

``n_points`` is output rows rather than integration accuracy -- variable_step_size
is on, so CVODE picks its own steps -- but the loss reads those rows through
np.interp onto the data times, so the density still moves the NLL. Measured on
the real SILK model, taking the labelling window from 200,000 rows to 10,000
shifted it by about 1.5e-3 nats: negligible against the 1.9207 threshold, and
not zero.

That is the situation a fingerprint exists for. Points computed under two
different settings are samples of two different curves, and a checkpoint
directory that mixes them is quietly wrong. Nothing else in spec_fingerprint can
see these numbers, because they live in functions on the replicates rather than
in the model text or the optimization spec.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.getcwd())


from pyantigen.engine.Profile_checkpoint import (                          # noqa: E402
    solver_fingerprint,
    spec_fingerprint,
)

failures = []


def check(name, cond, detail=""):
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}  {detail}")
        failures.append(name)


def replicates_with(n_points=200000, abs_tol=1e-10, end=10.0):
    def settings(rep):
        return {
            "simulation_blocks": {
                "block1": {"start": 0, "end": end, "n_points": n_points},
            },
            "abs_tol": abs_tol, "rel_tol": 1e-10, "max_steps": 200000,
            "variable_step_size": True,
        }
    return {"arm": {"Solver_settings": settings}}


def test_it_tracks_what_changes_the_objective():
    print("\nWhat the solver fingerprint sees:")
    base = solver_fingerprint(replicates_with())

    check("it is stable across calls",
          base == solver_fingerprint(replicates_with()))
    check("output density moves it",
          base != solver_fingerprint(replicates_with(n_points=10000)))
    check("tolerances move it",
          base != solver_fingerprint(replicates_with(abs_tol=1e-8)))
    check("block boundaries move it",
          base != solver_fingerprint(replicates_with(end=20.0)))

    # Float noise in a computed boundary -- an age in hours, say -- must not
    # invalidate a directory on its own, or resuming becomes impossible.
    check("float noise below nine significant figures does not",
          base == solver_fingerprint(replicates_with(end=10.0 + 1e-12)))


def test_it_never_takes_a_run_down():
    print("\nRobustness:")
    check("no replicates means no hash", solver_fingerprint(None) is None)
    check("an empty mapping means no hash", solver_fingerprint({}) is None)

    def boom(rep):
        raise RuntimeError("this settings function does not run here")

    broken = solver_fingerprint({"arm": {"Solver_settings": boom}})
    check("a settings function that raises still yields a hash",
          isinstance(broken, str) and broken, str(broken))
    # "Unreadable" has to be its own stable state, not an alias for anything
    # else, or two different failures would share a directory.
    check("and it differs from a readable one",
          broken != solver_fingerprint(replicates_with()))
    check("a replicate with no settings yields a hash too",
          isinstance(solver_fingerprint({"arm": {}}), str))


def test_it_covers_engine_level_integration_constants():
    """The tolerance floor decides what the integrator returns, too.

    ``_MIN_ABSOLUTE_TOLERANCE`` and ``_DUST_THRESHOLD`` live in pyantigen.engine.Simulate
    rather than in any settings dict, so nothing else in the fingerprint can
    see them -- but a run with a different floor is integrating a different
    problem. Raising the floor from 1e-30 to 1e-21 on the antibody arms moved
    the NLL by about 1e-6 nats, which is small and not zero.
    """
    print("\nEngine constants:")
    import pyantigen.engine.Simulate as sim

    base = solver_fingerprint(replicates_with())
    original = sim._MIN_ABSOLUTE_TOLERANCE
    try:
        # Any value other than the production one; derived from it so this
        # cannot silently become a no-op when the constant is next retuned.
        sim._MIN_ABSOLUTE_TOLERANCE = original * 1e-3
        moved = solver_fingerprint(replicates_with())
    finally:
        sim._MIN_ABSOLUTE_TOLERANCE = original

    check("the tolerance floor is part of the hash", base != moved)
    check("restoring it restores the hash",
          base == solver_fingerprint(replicates_with()))


def test_the_spec_hash_carries_it():
    print("\nThe spec hash:")
    args = (["p"], np.zeros(1), {}, ["lin"], "model text")
    dense = solver_fingerprint(replicates_with(n_points=200000))
    sparse = solver_fingerprint(replicates_with(n_points=10000))

    bare = spec_fingerprint(*args)
    check("omitting the solver hash leaves the old spec hash untouched",
          bare == spec_fingerprint(*args, solver_hash=None),
          "callers with no replicates must keep resuming their own points")
    check("supplying one changes the spec hash",
          bare[1] != spec_fingerprint(*args, solver_hash=dense)[1],
          f"{bare[1]} unchanged")
    check("two settings give two spec hashes",
          spec_fingerprint(*args, solver_hash=dense)[1]
          != spec_fingerprint(*args, solver_hash=sparse)[1])
    check("the model hash is unaffected by the solver settings",
          spec_fingerprint(*args, solver_hash=dense)[0]
          == spec_fingerprint(*args, solver_hash=sparse)[0])


if __name__ == "__main__":
    test_it_tracks_what_changes_the_objective()
    test_it_covers_engine_level_integration_constants()
    test_it_never_takes_a_run_down()
    test_the_spec_hash_carries_it()

    print("\n" + "=" * 72)
    if failures:
        print(f"{len(failures)} CHECK(S) FAILED: {', '.join(failures)}")
        sys.exit(1)
    print("ALL SOLVER FINGERPRINT CHECKS PASSED")

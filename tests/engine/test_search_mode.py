"""Search mode in pyantigen.engine.Simulate: a bounded price for a bad parameter vector.

Run from the repository root:
    python tests/engine/test_search_mode.py

The normal ``safe_simulate`` retries a failed integration up to ten times with
looser tolerances and a doubling step allowance, then halves the time span
recursively -- correct for a vector someone cares about, and what made
differential evolution never finish on a 16-parameter aggregation fit, where one box vertex was
still on the ladder eight minutes after failing its first attempt. Search mode is
one capped attempt that raises on failure. These checks pin the three things that
matter: it does not retry, it only ever removes candidates (a value it returns is
the value normal mode returns), and it leaves the integrator as it found it.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.getcwd())

import pyantigen.engine.Simulate as S                                          # noqa: E402

failures = []


def check(name, cond, detail=""):
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}  {detail}")
        failures.append(name)


# A predator-prey oscillator: cheap, but over a long window it needs thousands
# of CVODE steps, so a small step allowance fails and the ladder can rescue it.
_MODEL = """
model osc
  x = 10; y = 5
  a = 1.0; b = 0.1; c = 1.5; d = 0.075
  J1: -> x; a*x
  J2: x -> ; b*x*y
  J3: -> y; d*x*y
  J4: y -> ; c*y
end
"""
# Fixed output steps, so the allowance is per output interval (8 time units
# here) -- which is how the fast path behaves on the real specs.
_WINDOW = {"start": 0.0, "end": 400.0, "n_points": 50,
           "variable_step_size": False}


class _Counting:
    """Forwards everything to a RoadRunner and counts simulate() calls."""

    def __init__(self, r):
        object.__setattr__(self, "_r", r)
        object.__setattr__(self, "calls", 0)

    def simulate(self, *a, **k):
        object.__setattr__(self, "calls", self.calls + 1)
        return self._r.simulate(*a, **k)

    def __getattr__(self, name):
        return getattr(self._r, name)

    def __setattr__(self, name, value):
        setattr(self._r, name, value)


def _model(max_steps=None):
    import tellurium as te
    r = te.loada(_MODEL)
    r.integrator = "cvode"
    r.integrator.setValue("variable_step_size", False)
    if max_steps is not None:
        r.integrator.maximum_num_steps = int(max_steps)
    return r


def _steps_needed():
    """The smallest allowance in a coarse ladder at which the model integrates."""
    for n in (10, 20, 40, 80, 160, 320, 640, 1280, 2560, 5120):
        r = _model(n)
        try:
            r.simulate(0.0, 400.0, 50, ["time", "x", "y"])
            return n
        except Exception:
            continue
    return None


def test_search_mode_is_off_by_default():
    print("\nDefault:")
    check("no cap is set", S._SEARCH_MAX_STEPS is None)


def test_the_ladder_rescues_what_search_mode_rejects():
    print("\nThe same failure, two ways:")
    needed = _steps_needed()
    check("(setup) the model needs more than a tiny allowance", needed and needed > 40,
          str(needed))
    tiny = 40

    # Normal mode: the first attempt fails at 50 steps and the ladder rescues it.
    rc = _Counting(_model(tiny))
    res, meta = S.safe_simulate(rc, _WINDOW, ["time", "x", "y"], label="ladder")
    check("normal mode retries and succeeds", meta.get("attempts", 1) > 1
          and rc.calls > 1, f"{rc.calls} call(s), meta={meta}")

    # Search mode: one attempt, then it raises.
    rc = _Counting(_model(tiny))
    with S.search_mode(tiny):
        try:
            S.safe_simulate(rc, _WINDOW, ["time", "x", "y"], label="search")
            raised = False
        except Exception:
            raised = True
    check("search mode raises rather than climbing the ladder", raised)
    check("and it made exactly one attempt", rc.calls == 1, f"{rc.calls} calls")
    check("the integrator's own allowance was restored after the failure",
          rc.integrator.maximum_num_steps == tiny,
          str(rc.integrator.maximum_num_steps))


def test_search_mode_only_removes_candidates():
    """A value it returns is the value normal mode returns."""
    print("\nSame answer when it succeeds:")
    needed = _steps_needed()
    normal, _ = S.safe_simulate(_model(needed * 4), _WINDOW, ["time", "x", "y"])
    with S.search_mode(needed * 4):
        capped, meta = S.safe_simulate(_model(needed * 4), _WINDOW,
                                       ["time", "x", "y"])
    check("bit-identical to normal mode", np.array_equal(np.asarray(normal),
                                                        np.asarray(capped)))
    check("it reports that it was a single search-mode attempt",
          meta.get("search_mode") is True and meta.get("attempts") == 1)


def test_the_cap_is_a_ceiling_not_a_floor():
    print("\nThe cap never raises a model's own lower limit:")
    rc = _Counting(_model(30))
    with S.search_mode(1_000_000):
        try:
            S.safe_simulate(rc, _WINDOW, ["time", "x", "y"])
            ok = False
        except Exception:
            ok = True
    check("a model limited to 30 steps still fails under a huge cap", ok)
    check("and its limit is still 30 afterwards",
          rc.integrator.maximum_num_steps == 30,
          str(rc.integrator.maximum_num_steps))


def test_the_context_manager_restores():
    print("\nScoping:")
    S.set_search_mode(None)
    with S.search_mode(123):
        check("inside the block the cap is set", S._SEARCH_MAX_STEPS == 123)
        with S.search_mode(456):
            check("nested blocks stack", S._SEARCH_MAX_STEPS == 456)
        check("and unwind", S._SEARCH_MAX_STEPS == 123)
    check("outside the block it is off again", S._SEARCH_MAX_STEPS is None)
    try:
        with S.search_mode(789):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    check("an exception inside still restores it", S._SEARCH_MAX_STEPS is None)


def test_zz_every_check_passed():
    """``check`` records failures rather than raising, so pytest needs this."""
    assert not failures, "failed checks: " + ", ".join(failures)


if __name__ == "__main__":
    test_search_mode_is_off_by_default()
    test_the_ladder_rescues_what_search_mode_rejects()
    test_search_mode_only_removes_candidates()
    test_the_cap_is_a_ceiling_not_a_floor()
    test_the_context_manager_restores()
    print()
    if failures:
        print(f"{len(failures)} FAILED: {failures}")
        sys.exit(1)
    print("all checks passed")

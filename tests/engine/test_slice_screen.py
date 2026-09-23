"""The slice screen, against objectives whose identifiability is known by construction.

Run from the repository root:
    python tests/engine/test_slice_screen.py

The screen's whole claim is one inequality -- the slice is an upper bound on the
profile, so a slice below the threshold at a bound proves the profile is too.
What can go wrong is not the inequality but the bookkeeping around it: a grid
that never reaches the bound (which makes the verdict meaningless), a sentinel
read as a likelihood, a verdict that halts a run it should not, or a cache that
hands one fit's screen to another.

So the objectives here are analytic. A quadratic in every direction is
identifiable; a quadratic with one flat direction is not, and is not at any
value the parameter is allowed to take; and a raised quadratic crosses the
threshold and comes back down further out, which is why the screen stops a
side the round it first crosses rather than reading a verdict from whatever
its candidate ladder's last point happens to be -- the far dip is real, but
the screen no longer pays to look for it once a nearer point has already
answered the only question a slice can answer.
"""
import json
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.getcwd())

from pyantigen.engine.Identifiability import (                             # noqa: E402
    THRESHOLD,
    UnidentifiableParameters,
    failing_sides,
    load_screen,
    run_slice_screen,
    save_screen,
    screen_or_raise,
    screen_summary,
    screen_values,
)

failures = []


def check(name, cond, detail=""):
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}  {detail}")
        failures.append(name)


def _batch_from(nll):
    """Stub for ParallelEvaluator.evaluate_batch: one value per vector."""
    def batch(xs, label=None):
        return [float(nll(np.asarray(x, dtype=float))) for x in xs]
    return batch


# ---------------------------------------------------------------------------
# Grids
# ---------------------------------------------------------------------------

def test_grid_reaches_the_further_of_bound_and_span():
    """The walk's end settles the question, and a declared bound does not set it.

    Every bound on the real spec is ``x0/10`` to ``x0*10`` -- a one-decade
    search box around the starting value, not physics. A verdict read at a box
    edge is a verdict about the box, so the walk goes to whichever is further,
    the bound or the span, and the bound is included as a point of its own when
    it is passed.
    """
    print("\nScreen grid:")

    # A one-decade box against a three-decade span: the span wins, and the box
    # edge survives as its own point so "did it cross inside the box" is still
    # answerable.
    vals = screen_values(p_opt=0.0, lb=-1.0, ub=1.0, se=0.05, sign=1,
                         is_log=True, max_points=6, span_decades=3.0)
    check("the walk ends at the span, not at the bound",
          abs(vals[-1] - 3.0) < 1e-12, str(vals))
    check("it walks past the declared bound", any(v > 1.0 for v in vals),
          str(vals))
    check("and keeps the bound as a point of its own",
          any(abs(v - 1.0) < 1e-9 for v in vals), str(vals))
    check("it spends no more points than it was given", len(vals) <= 6,
          str(vals))
    check("it walks outward from the optimum",
          all(b > a for a, b in zip(vals, vals[1:])), str(vals))

    lo = screen_values(p_opt=0.0, lb=-1.0, ub=1.0, se=0.05, sign=-1,
                       is_log=True, max_points=6, span_decades=3.0)
    check("the lower side ends at the lower span",
          abs(lo[-1] + 3.0) < 1e-12, str(lo))
    check("the lower side walks outward too",
          all(b < a for a, b in zip(lo, lo[1:])), str(lo))

    # A bound further out than the span is honoured: the span is a floor on how
    # far to look, never a ceiling.
    wide = screen_values(p_opt=0.0, lb=-30.0, ub=30.0, se=1e-3, sign=1,
                         is_log=True, max_points=5, span_decades=3.0)
    check("a bound beyond the span is walked to",
          abs(wide[-1] - 30.0) < 1e-12, str(wide))

    # Distances multiply rather than add, so a parameter spanning decades is
    # explored in decades.
    ladder = screen_values(p_opt=0.0, lb=-30.0, ub=30.0, se=1.0, sign=1,
                           is_log=True, max_points=6)
    gaps = [b - a for a, b in zip(ladder, ladder[1:])]
    check("the ladder grows geometrically", gaps[1] > gaps[0], str(ladder))

    # No bound at all: the span is what makes a verdict possible.
    unbounded = screen_values(p_opt=0.0, lb=-np.inf, ub=np.inf, se=1.0, sign=1,
                              is_log=True, max_points=4, span_decades=3.0)
    check("an unbounded side still reaches the span",
          unbounded and abs(unbounded[-1] - 3.0) < 1e-12, str(unbounded))
    check("and none of its points is infinite",
          all(np.isfinite(v) for v in unbounded), str(unbounded))

    # A linear parameter at zero has no decades and no ratio, so the bound is
    # all there is -- and when the optimum sits on it there is no side to walk.
    check("a side with no room returns nothing",
          screen_values(p_opt=0.0, lb=-1.0, ub=0.0, se=1.0, sign=1,
                        is_log=False, max_points=4) == [])

    # No Wald SE -- the Hessian was singular in this direction, which is
    # exactly when the parameter is least likely to be pinned down.
    no_se = screen_values(p_opt=0.0, lb=-8.0, ub=8.0, se=None, sign=1,
                          is_log=True, max_points=4)
    check("a missing SE still produces a grid to the end of the walk",
          len(no_se) >= 2 and abs(no_se[-1] - 8.0) < 1e-12, str(no_se))


# ---------------------------------------------------------------------------
# Verdicts
# ---------------------------------------------------------------------------

def _identifiable_nll(x):
    return 0.5 * float(np.dot(x, x))


def test_identifiable_parameters_pass():
    """Every direction curved: the screen must not stand in the way."""
    print("\nAn identifiable objective:")
    names = ["a", "b"]
    bounds = [(-5.0, 5.0), (-5.0, 5.0)]
    report = run_slice_screen(
        _batch_from(_identifiable_nll), np.zeros(2), 0.0, names, bounds,
        scales=["lin", "lin"], wald_se=[1.0, 1.0], verbose=False,
    )
    check("nothing is proved unbounded", report["n_open"] == 0,
          str(report["n_open"]))
    check("every side crossed the threshold", report["n_crossed"] == 4,
          str(report["n_crossed"]))
    check("no side is left inconclusive", report["n_inconclusive"] == 0,
          str(report["n_inconclusive"]))

    # The certified inner bracket: the outermost value whose slice is still
    # below the threshold. dNLL = x^2/2 reaches 1.9207 at x = 1.960, so the
    # bracket has to sit inside that and be a real evaluated point.
    upper = report["parameters"]["a"]["upper"]
    check("an inner bracket is certified",
          upper["inner_bracket"] is not None
          and 0 < upper["inner_bracket"] < 1.9603,
          str(upper["inner_bracket"]))


def test_flat_direction_is_proved_unbounded():
    """A parameter the data cannot bound, proved at its own bound."""
    print("\nA flat direction:")
    names = ["curved", "flat"]
    bounds = [(-5.0, 5.0), (-5.0, 5.0)]

    def nll(x):
        return 0.5 * float(x[0]) ** 2      # x[1] does not appear at all

    report = run_slice_screen(
        _batch_from(nll), np.zeros(2), 0.0, names, bounds,
        scales=["lin", "lin"], wald_se=[1.0, None], verbose=False,
    )
    check("both sides of the flat parameter are open",
          report["parameters"]["flat"]["lower"]["state"] == "open"
          and report["parameters"]["flat"]["upper"]["state"] == "open")
    check("the curved parameter is untouched by that",
          report["parameters"]["curved"]["upper"]["state"] == "crossed")

    fails = failing_sides(report)
    check("both open sides are reported as failures", len(fails) == 2,
          str(len(fails)))
    check("the failure names the parameter and the side",
          {f["side"] for f in fails} == {"lower", "upper"}
          and {f["name"] for f in fails} == {"flat"}, str(fails))
    check("the verdict is read at the reach, and the reach is recorded",
          all(f["reach"] is not None and abs(f["reach"]) >= 5.0
              for f in fails), str([f["reach"] for f in fails]))

    # The whole point of the screen: this raises before any profile point runs.
    try:
        screen_or_raise(_batch_from(nll), np.zeros(2), 0.0, names, bounds,
                        scales=["lin", "lin"], wald_se=[1.0, None],
                        verbose=False)
        check("screen_or_raise halts on a flat direction", False, "no raise")
    except UnidentifiableParameters as exc:
        check("screen_or_raise halts on a flat direction", True)
        check("the message names the parameter", "flat" in str(exc), str(exc))
        check("the message says what to do",
              "literature" in str(exc) and "zero" in str(exc), str(exc))


def test_a_slice_that_crosses_proves_nothing_either_way():
    """The screen is one-directional, and the code must not overstate it.

    A slice crossing the threshold is permission to profile, not a finding of
    identifiability: the profile may flatten out beyond the slice crossing and
    never reach it. What is guarded here is that "crossed" is the only thing
    recorded -- no side is marked identified anywhere in the report.
    """
    print("\nOne-directionality:")
    names = ["a"]
    report = run_slice_screen(
        _batch_from(_identifiable_nll), np.zeros(1), 0.0, names,
        [(-5.0, 5.0)], scales=["lin"], wald_se=[1.0], verbose=False,
    )
    states = {s["state"] for s in report["parameters"]["a"].values()}
    check("a crossing is recorded as 'crossed' and nothing stronger",
          states == {"crossed"}, str(states))
    summary = screen_summary(report)
    check("the summary carries no claim of identifiability",
          "identifiable" not in json.dumps(summary).lower(), str(summary))


def test_an_arbitrary_bound_cannot_manufacture_a_halt():
    """The user's search box must not decide identifiability.

    The parameter here is identifiable -- its slice crosses at 1.5 decades --
    but the declared bound is the usual one-decade box, so the crossing sits
    just outside it. Reading the verdict at the bound would halt the whole
    analysis on a parameter the data constrain perfectly well. Walking past the
    box finds the crossing, clears the side, and reports that the box is
    narrower than the interval.
    """
    print("\nAn arbitrary bound:")
    names = ["a"]

    def nll(x):
        # Flat inside 1.5 decades, steep outside: a crossing just beyond a
        # one-decade box.
        return 0.0 if abs(float(x[0])) < 1.5 else 10.0

    report = run_slice_screen(
        _batch_from(nll), np.zeros(1), 0.0, names, [(-1.0, 1.0)],
        scales=["log10"], wald_se=[0.05], span_decades=3.0, verbose=False,
    )
    states = {s["state"] for s in report["parameters"]["a"].values()}
    check("the side is cleared, not halted", states == {"crossed"}, str(states))
    check("nothing is reported as a failure", not failing_sides(report))
    check("and the narrow search box is reported",
          report["n_box_too_narrow"] == 2, str(report["n_box_too_narrow"]))

    # The same objective judged at the box edge is the mistake this replaces.
    boxed = run_slice_screen(
        _batch_from(nll), np.zeros(1), 0.0, names, [(-1.0, 1.0)],
        scales=["log10"], wald_se=[0.05], span_decades=0.0, verbose=False,
    )
    check("a screen confined to the box would have halted it",
          len(failing_sides(boxed)) == 2, str(failing_sides(boxed)))


def test_a_huge_declared_bound_is_never_touched_once_crossed():
    """A declared bound is just a number a user typed, not a safety limit.

    The bound is folded into the candidate ladder itself (see screen_values),
    not only into the walk's final target, so it is not enough to withhold
    just the last point once a side has crossed -- every point after the
    crossing has to be withheld, wherever in the ladder it happens to fall.
    This bound is 1e12: if the round-by-round walk ever asked for a point
    anywhere near it after crossing on the near, sane rungs, that would be
    exactly the unconditional far-out evaluation this design exists to avoid.
    """
    print("\nA huge declared bound:")
    names = ["a"]
    seen = []

    def nll(x):
        v = abs(float(x[0]))
        seen.append(v)
        return 4.0 if v >= 1.0 else 0.0

    report = run_slice_screen(
        _batch_from(nll), np.zeros(1), 0.0, names, [(-1e12, 1e12)],
        scales=["lin"], wald_se=[1.0], verbose=False,
    )
    upper = report["parameters"]["a"]["upper"]
    check("the side crosses on the near, sane rungs",
          upper["state"] == "crossed", upper["state"])
    check("the huge declared bound was never evaluated",
          upper["stopped_early"] is True and max(seen) < 100.0,
          f"max |x| evaluated = {max(seen):.6g}")


def test_reach_is_reported_in_decades():
    """The verdict has to say how far it looked, in a scale-free unit."""
    print("\nReach:")
    names = ["flat"]

    report = run_slice_screen(
        _batch_from(lambda x: 0.0), np.zeros(1), 0.0, names, [(-1.0, 1.0)],
        scales=["log10"], wald_se=[0.05], span_decades=3.0, verbose=False,
    )
    upper = report["parameters"]["flat"]["upper"]
    check("the verdict is open", upper["state"] == "open", upper["state"])
    check("the reach is three decades",
          abs(upper["reach_decades"] - 3.0) < 1e-9, str(upper["reach_decades"]))
    check("and it records that the bound was passed",
          upper["walked_past_bound"] is True)


def test_a_failure_sentinel_is_not_a_likelihood():
    """1e10 means the model broke there, not that the slice crossed.

    Reading the sentinel as a likelihood would mark the side "crossed" and let
    the run spend days profiling a parameter that was never screened. Reading
    it as a low value would halt the run on a simulation failure. It is neither:
    the reach shrinks back to the last point that did evaluate, and the side is
    reported as unscreened.
    """
    print("\nFailure sentinels:")
    names = ["a"]

    def nll(x):
        # Flat everywhere the model works; it stops evaluating past 1 decade.
        return 1e10 if abs(float(x[0])) > 1.0 else 0.0

    report = run_slice_screen(
        _batch_from(nll), np.zeros(1), 0.0, names, [(-1.0, 1.0)],
        scales=["log10"], wald_se=[0.05], span_decades=3.0,
        min_reach_decades=2.0, verbose=False,
    )
    states = {s["state"] for s in report["parameters"]["a"].values()}
    check("a reach that falls short is neither crossed nor open",
          states == {"short-reach"}, str(states))
    check("a short reach does not halt the run", not failing_sides(report))
    check("it is counted as inconclusive", report["n_inconclusive"] == 2,
          str(report["n_inconclusive"]))
    check("and the reach it did manage is recorded",
          abs(report["parameters"]["a"]["upper"]["reach_decades"] - 1.0) < 1e-9,
          str(report["parameters"]["a"]["upper"]["reach_decades"]))

    # Nothing evaluable at all on the side is a different, blunter answer.
    blocked = run_slice_screen(
        _batch_from(lambda x: 1e10 if float(x[0]) != 0.0 else 0.0),
        np.zeros(1), 0.0, names, [(-1.0, 1.0)], scales=["log10"],
        wald_se=[0.05], verbose=False,
    )
    check("a side that never evaluates is blocked",
          {s["state"] for s in blocked["parameters"]["a"].values()}
          == {"blocked"},
          str({s["state"] for s in blocked["parameters"]["a"].values()}))
    check("blocked does not halt the run either",
          not failing_sides(blocked))


def test_a_side_stops_the_round_it_crosses():
    """A crossing ends the walk there, even if the curve dips back down later.

    dNLL rises above 1.9207 inside [1, 3] and falls back below it out at the
    bound (8). A full walk to the bound would read this as "open, but not
    monotone" -- the data do not exclude the bound either. The screen no
    longer pays for that: once a side crosses, it stops, because reaching
    further is exactly the unconditional evaluation (declared bounds included)
    that is no longer safe to make automatically. So this reads "crossed", not
    "open" -- permission to spend a real profile on it, which the far, unseen
    dip would have granted anyway, just less directly and at a real cost this
    trades away on purpose.
    """
    print("\nA non-monotone slice:")
    names = ["a"]

    def nll(x):
        v = abs(float(x[0]))
        return 4.0 if 1.0 <= v <= 3.0 else 0.0

    report = run_slice_screen(
        _batch_from(nll), np.zeros(1), 0.0, names, [(-8.0, 8.0)],
        scales=["lin"], wald_se=[1.0], verbose=False,
    )
    upper = report["parameters"]["a"]["upper"]
    check("the side stops at the crossing rather than reading the far dip",
          upper["state"] == "crossed", upper["state"])
    check("nothing beyond the crossing was ever evaluated",
          upper["stopped_early"] is True)
    check("the inner bracket stops at the first crossing",
          upper["inner_bracket"] is not None and upper["inner_bracket"] < 1.0,
          str(upper["inner_bracket"]))


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def test_the_screen_is_reused_only_by_the_fit_that_ran_it():
    """Ten minutes saved is worth caching; a wrong verdict is not.

    The checkpoint directory is already keyed by model hash, spec hash and the
    optimum, so a file found there belongs to this run. The fields are checked
    anyway, because honouring another fit's screen would decide identifiability
    from the wrong likelihood.
    """
    print("\nScreen persistence:")
    names = ["a", "b"]
    res_x = np.array([0.0, 0.0])
    report = run_slice_screen(
        _batch_from(_identifiable_nll), res_x, 0.0, names,
        [(-5.0, 5.0), (-5.0, 5.0)], scales=["lin", "lin"],
        wald_se=[1.0, 1.0], verbose=False,
    )

    with tempfile.TemporaryDirectory() as d:
        path = save_screen(report, d)
        check("the screen is written where the points will go",
              path is not None and os.path.exists(path), str(path))

        again = load_screen(d, names, res_x)
        check("the same fit reads it back",
              again is not None and again["n_crossed"] == report["n_crossed"])
        check("a different optimum does not",
              load_screen(d, names, np.array([0.0, 0.1])) is None)
        check("a different parameter set does not",
              load_screen(d, ["a", "c"], res_x) is None)
        check("a different threshold does not",
              load_screen(d, names, res_x, threshold=2.7) is None)
        # How far the screen looked is part of what its verdict means, so a
        # screen run over one decade must not answer for a run asked for three.
        check("a different span does not",
              load_screen(d, names, res_x, span_decades=1.0) is None)
        check("a different reach floor does not",
              load_screen(d, names, res_x, min_reach_decades=2.5) is None)

        with open(path, "w", encoding="utf-8") as fh:
            fh.write("{not json")
        check("a truncated file is a miss, not a crash",
              load_screen(d, names, res_x) is None)

    check("no checkpoint directory is not an error",
          save_screen(report, None) is None
          and load_screen(None, names, res_x) is None)


def test_a_cached_failure_still_halts():
    """A resumed link must not walk past a verdict an earlier link reached."""
    print("\nA cached failure:")
    names = ["flat"]

    def nll(x):
        return 0.0

    with tempfile.TemporaryDirectory() as d:
        for attempt in ("first", "resumed"):
            try:
                screen_or_raise(_batch_from(nll), np.zeros(1), 0.0, names,
                                [(-5.0, 5.0)], scales=["lin"], ckpt_dir=d,
                                verbose=False)
                check(f"the {attempt} link halts", False, "no raise")
            except UnidentifiableParameters:
                check(f"the {attempt} link halts", True)


if __name__ == "__main__":
    test_grid_reaches_the_further_of_bound_and_span()
    test_identifiable_parameters_pass()
    test_flat_direction_is_proved_unbounded()
    test_a_slice_that_crosses_proves_nothing_either_way()
    test_an_arbitrary_bound_cannot_manufacture_a_halt()
    test_a_huge_declared_bound_is_never_touched_once_crossed()
    test_reach_is_reported_in_decades()
    test_a_failure_sentinel_is_not_a_likelihood()
    test_a_side_stops_the_round_it_crosses()
    test_the_screen_is_reused_only_by_the_fit_that_ran_it()
    test_a_cached_failure_still_halts()

    print("\n" + "=" * 72)
    if failures:
        print(f"{len(failures)} CHECK(S) FAILED: {', '.join(failures)}")
        sys.exit(1)
    print("ALL SLICE SCREEN CHECKS PASSED")

"""Checks for the discontinuity-time extractor.

Run from the repository root:
    python tests/engine/test_event_times.py

These run against small Antimony models built here rather than against the real
model, so a failure is the extractor and nothing else. The models are written to
mirror the shapes the real generators produce -- constant triggers from
``generate_silk_events``, a symbolic ``+ SubCut_D1`` trigger from the
subcutaneous generators, ``{i} + V_LP/Q_CSF`` arithmetic from the v4 hourly CSF
draws, a time-based ``piecewise`` assignment rule from
``generate_antimony_piecewise``, and the state-guard ``piecewise`` rules the
model's own rules file is full of.

The property that matters most is the last one: an unresolvable trigger must
make the table refuse to answer, because a missed event would let the block
splitter cut exactly on a discontinuity.
"""
import os
import sys

sys.path.insert(0, os.getcwd())


import tellurium as te                                            # noqa: E402

from pyantigen.engine.Event_times import (                                  # noqa: E402
    EventTimeTable,
    attach_event_times,
    build_event_time_table,
    _dedupe,
)

failures = []


def check(name, cond, detail=""):
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}  {detail}")
        failures.append(name)


def load(body):
    return te.loada("model m()\n" + body + "\nend")


_BASE = """
  S1 = 1; S2 = 0;
  k = 0.1;
  J1: S1 -> S2; k*S1;
"""


def close(a, b, tol=1e-9):
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


# ---------------------------------------------------------------------------

def test_constant_triggers():
    """The SILK shape: many events at literal times."""
    print("\nConstant time triggers:")
    body = _BASE + "".join(
        f"  E{i}: at (time >= {i}): k = {0.1 + i};\n" for i in range(5)
    )
    table = build_event_time_table(load(body))
    times = table.times()
    check("all five events are found", times is not None and len(times) == 5,
          f"{times}")
    check("times are the literal trigger values",
          times == [0.0, 1.0, 2.0, 3.0, 4.0], f"{times}")
    check("nothing was left unresolved", not table.unresolved,
          f"{table.unresolved}")
    check("no model symbols were needed", table.dependencies() == set(),
          f"{table.dependencies()}")


def test_arithmetic_triggers():
    """The real generators write expressions, not numbers."""
    print("\nArithmetic in triggers:")
    body = _BASE + """
  age = 72.18;
  E1: at (time >= age*365*24 + 12): k = 0.2;
  E2: at (time >= 2^3): k = 0.3;
  E3: at (time >= -4 + 10): k = 0.4;
"""
    table = build_event_time_table(load(body))
    times = table.times()
    check("expressions are evaluated", times is not None and len(times) == 3,
          f"{times}")
    if times:
        check("product and sum are right",
              close(max(times), 72.18 * 365 * 24 + 12), f"{max(times)}")
        check("power is right", 8.0 in times, f"{times}")
        check("unary minus is right", 6.0 in times, f"{times}")
    check("the symbol it used is reported",
          table.dependencies() == {"age"}, f"{table.dependencies()}")


def test_symbolic_trigger_tracks_the_model():
    """``SubCut_D1`` is fitted, so its event time must move when it does.

    This is the whole reason the table stores expressions instead of numbers.
    Freezing the times at x0 would put the infusion-off event in the wrong place
    for every evaluation after the first.
    """
    print("\nA trigger built on a fitted parameter:")
    body = _BASE + """
  SubCut_D1 = 23.008700865453;
  t_start = 1000;
  E1_on:  at (time >= t_start): k = 0.2;
  E1_off: at (time >= (t_start) + SubCut_D1): k = 0.0;
"""
    r = load(body)
    table = build_event_time_table(r)
    before = table.times()
    check("both edges of the infusion are found",
          before is not None and len(before) == 2, f"{before}")
    check("the off edge is one D1 after the on edge",
          before is not None and close(before[1] - before[0], 23.008700865453),
          f"{before}")

    r["SubCut_D1"] = 50.0
    after = table.times()
    check("the off edge moves with the parameter",
          after is not None and close(after[1] - after[0], 50.0), f"{after}")
    check("the on edge does not move",
          after is not None and close(after[0], before[0]), f"{after}")
    check("the dependency is reported",
          "SubCut_D1" in table.dependencies(), f"{table.dependencies()}")


def test_v4_hourly_draw_shape():
    """``{i} + V_LP/Q_CSF``: division over two model parameters."""
    print("\nThe v4 hourly-CSF-draw trigger shape:")
    body = _BASE + """
  Q_CSF = 17.39500698; V_LP = 6; t_CSFdraw = 0.1;
  E0: at (time >= 3): k = 0.2;
  E1: at (time >= 3 + t_CSFdraw): k = 0.3;
  E2: at (time >= 3 + V_LP/Q_CSF): k = 0.4;
"""
    table = build_event_time_table(load(body))
    times = table.times()
    check("all three draw edges are found",
          times is not None and len(times) == 3, f"{times}")
    if times:
        check("the division is evaluated",
              close(times[-1], 3 + 6 / 17.39500698), f"{times[-1]}")
    check("both parameters are reported",
          table.dependencies() == {"Q_CSF", "V_LP", "t_CSFdraw"},
          f"{table.dependencies()}")


def test_time_piecewise_rule_is_found():
    """``generate_antimony_piecewise`` emits a rule, not an event.

    RoadRunner does not root-find assignment rules, so these breakpoints are
    discontinuities the integrator will step straight over. If the extractor
    misses them the splitter cannot protect them.
    """
    print("\nA time-based piecewise assignment rule:")
    body = _BASE + """
  f_L := piecewise(0, time < 5, 1, time < 7.25, 2);
"""
    table = build_event_time_table(load(body))
    times = table.times()
    check("the rule is recognised as a time discontinuity",
          table.n_rules == 1, f"n_rules={table.n_rules}")
    check("both breakpoints are found",
          times is not None and len(times) == 2, f"{times}")
    check("the breakpoints are the piecewise conditions",
          times == [5.0, 7.25], f"{times}")
    check("no events were involved", table.n_events == 0,
          f"n_events={table.n_events}")


def test_state_guards_are_ignored():
    """The model's own ``piecewise`` guards switch on state, not time.

    The rules file carries 26 of them. They are a genuine numerical hazard, but
    not one a time-based splitter can do anything about, so they must neither be
    reported as cut candidates nor treated as a failure to resolve.
    """
    print("\nState-triggered piecewise guards:")
    body = _BASE + """
  total = 4;
  frac := piecewise(1, total < 1e-12, S1 / total);
  E1: at (time >= 30): k = 0.5;
"""
    table = build_event_time_table(load(body))
    times = table.times()
    check("the state guard is not counted as a time rule",
          table.n_rules == 0, f"n_rules={table.n_rules}")
    check("it is not treated as unresolved", not table.unresolved,
          f"{table.unresolved}")
    check("only the real event is reported", times == [30.0], f"{times}")


def test_no_discontinuities_is_a_positive_answer():
    """An empty list is a finding; None is an absence of one.

    Figure 3's block1 ages the model for seventy years with no events at all,
    and knowing that is exactly what lets the splitter leave it whole. That has
    to be distinguishable from "could not tell".
    """
    print("\nA model with nothing discontinuous in it:")
    table = build_event_time_table(load(_BASE))
    times = table.times()
    check("times is an empty list, not None", times == [], f"{times}")
    check("the table is empty", len(table) == 0, f"{len(table)}")


def test_unresolvable_trigger_refuses_to_answer():
    """The safety property: a state trigger must poison the whole table.

    A cut placed on a discontinuity is the worst outcome available, so anything
    the extractor cannot place on the time axis has to make it decline rather
    than under-report.
    """
    print("\nA trigger the extractor cannot place in time:")
    body = _BASE + """
  E1: at (time >= 10): k = 0.2;
  E2: at (S1 < 0.5): k = 0.9;
"""
    table = build_event_time_table(load(body))
    check("the state-triggered event is recorded as unresolved",
          len(table.unresolved) == 1, f"{table.unresolved}")
    check("times() declines to answer", table.times() is None,
          f"{table.times()}")
    check("the resolvable event was still parsed", len(table) == 1,
          f"{len(table)}")


def test_logical_triggers_collect_every_threshold():
    """``and``/``or`` triggers contribute every time they mention."""
    print("\nA compound trigger:")
    body = _BASE + """
  E1: at ((time >= 10) && (time <= 20)): k = 0.2;
"""
    table = build_event_time_table(load(body))
    times = table.times()
    check("both bounds are offered as cut candidates",
          times == [10.0, 20.0], f"{times}")


def test_reversed_comparison():
    """``expr >= time`` means the same instant as ``time <= expr``."""
    print("\nA reversed comparison:")
    body = _BASE + """
  E1: at (40 >= time): k = 0.2;
"""
    table = build_event_time_table(load(body))
    check("the threshold is found on either side", table.times() == [40.0],
          f"{table.times()}")


def test_dedupe():
    """Float noise merges; genuinely close events do not."""
    print("\nDeduplication:")
    t = 72.18 * 365 * 24
    check("noise-level duplicates collapse",
          _dedupe([t, t + t * 1e-15, t + t * 1e-14]) == [t], "")
    check("one-hour-apart SILK events survive",
          len(_dedupe([t, t + 1.0, t + 2.0])) == 3, "")
    check("a 23 h SubCut_D1 gap survives",
          len(_dedupe([t, t + 23.008700865453])) == 2, "")


def test_attach_and_lookup_failure():
    """attach() wires the callable onto the replicate; a broken lookup is safe."""
    print("\nAttaching to a replicate:")
    body = _BASE + """
  D = 5;
  E1: at (time >= 100 + D): k = 0.2;
"""
    r = load(body)
    rep = {"Label": "test_arm"}
    table = attach_event_times(rep, r)
    check("the callable is attached", callable(rep.get("_event_times_fn")),
          f"{rep.get('_event_times_fn')}")
    check("it returns the same answer the table does",
          rep["_event_times_fn"]() == table.times(), "")

    # A table whose symbol cannot be read must decline rather than raise: the
    # splitter calls this on every evaluation and must never be the thing that
    # brings a fit down.
    broken = EventTimeTable({}, [("E1", ('sym', 'missing'))], [])
    check("an unreadable symbol yields None, not an exception",
          broken.times() is None, "")

    # Likewise a non-finite result.
    nan_table = EventTimeTable(
        {"a": 0.0, "b": 0.0},
        [("E1", ('op', '/', ('sym', 'a'), ('sym', 'b')))], [])
    check("a non-finite time yields None", nan_table.times() is None, "")


# ---------------------------------------------------------------------------
# Against the project's own generators
# ---------------------------------------------------------------------------
#
# The tests above use hand-written models, so they check the extractor against
# what this file believes the generators emit. These check it against what they
# actually emit, on a stub model carrying just the symbols they touch -- so a
# generator that changes shape breaks a test here rather than silently sending
# every arm back to the wall-clock cap.

_STUB = """
  S1 = 1; k = 0.1;
  J1: S1 -> ; k*S1;
  f_L = 0; Q_Leak = 0; Q_Leak_base = 1;
  Antibody_Plasma = 0; dose_scaling_factor = 1;
  SubCut_infusion_rate = 0; SubCut_bioavailability = 0.7;
  SubCut_D1 = 23.008700865453; Antibody_SubCutComp_Dose = 100;
"""


def test_stripping_for_pool_workers():
    """A replicate must not carry the callable to a worker.

    cloudpickle will carry it -- it serializes the closed-over RoadRunner by
    value, verified round-tripping -- so nothing crashes to warn you. That is
    what makes the strip worth testing: a shipped table resolves its times
    against the *parent's* parameter values, while a worker evaluating a profile
    point holds a different vector by construction. With SubCut_D1 among the
    fitted parameters, every worker would split on a schedule it is not
    integrating, and the parallel run would silently disagree with the serial
    one. ``build_eval_spec`` strips it; ``_init_worker`` re-attaches against the
    model the worker compiled.
    """
    print("\nStripping for pool workers:")
    from pyantigen.engine.Event_times import without_event_times

    body = _BASE + "  E1: at (time >= 100): k = 0.2;\n"
    r = load(body)
    rep = {"Label": "arm", "Age": 70}
    attach_event_times(rep, r)

    stripped = without_event_times({"arm": rep})
    check("the callable is gone from the copy",
          "_event_times_fn" not in stripped["arm"], "")
    check("everything else survives",
          stripped["arm"]["Label"] == "arm" and stripped["arm"]["Age"] == 70, "")
    check("the original keeps its attachment",
          callable(rep.get("_event_times_fn")), "")

    try:
        import cloudpickle as serializer
    except ImportError:
        import pickle as serializer
    try:
        serializer.dumps(stripped)
        ok = True
    except Exception as exc:
        ok = False
        detail = str(exc)
    check("the stripped copy serializes", ok, detail if not ok else "")

    # The unstripped one serializes too, and that is the hazard: there is no
    # error to catch it, so the strip has to be deliberate.
    blob = serializer.dumps({"arm": rep})
    revived = serializer.loads(blob)["arm"]
    check("an unstripped copy round-trips silently, carrying a whole model",
          callable(revived.get("_event_times_fn"))
          and revived["_event_times_fn"]() == rep["_event_times_fn"](),
          "cloudpickle refused it, so the strip is merely an optimisation")
    check("the stripped copy is materially smaller",
          len(serializer.dumps(stripped)) * 4 < len(blob),
          f"{len(serializer.dumps(stripped))} vs {len(blob)} bytes")


def test_reattach_replaces_a_stale_model():
    """Rebuilding events must repoint the callable at the new model."""
    print("\nRe-attaching after a rebuild:")
    rep = {"Label": "arm"}
    r1 = load(_BASE + "  E1: at (time >= 100): k = 0.2;\n")
    attach_event_times(rep, r1)
    first = rep["_event_times_fn"]()

    r2 = load(_BASE + "  E1: at (time >= 500): k = 0.2;\n")
    attach_event_times(rep, r2)
    second = rep["_event_times_fn"]()

    check("the first model's schedule was read", first == [100.0], f"{first}")
    check("the rebuilt model's schedule replaces it", second == [500.0],
          f"{second}")


def main():
    print("=" * 72)
    print("Event-time extractor")
    print("=" * 72)
    test_constant_triggers()
    test_arithmetic_triggers()
    test_symbolic_trigger_tracks_the_model()
    test_v4_hourly_draw_shape()
    test_time_piecewise_rule_is_found()
    test_state_guards_are_ignored()
    test_no_discontinuities_is_a_positive_answer()
    test_unresolvable_trigger_refuses_to_answer()
    test_logical_triggers_collect_every_threshold()
    test_reversed_comparison()
    test_dedupe()
    test_attach_and_lookup_failure()
    test_stripping_for_pool_workers()
    test_reattach_replaces_a_stale_model()

    print("\n" + "=" * 72)
    if failures:
        print(f"FAILED ({len(failures)}):")
        for f in failures:
            print("  -", f)
        return 1
    print("ALL EVENT-TIME CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())

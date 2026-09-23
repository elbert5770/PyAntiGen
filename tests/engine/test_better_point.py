"""What the profile records when it finds a point below the reported optimum.

A profile minimizes over every nuisance parameter at each fixed value, so it
searches places the fit never visited. Landing lower than the optimizer did is
a normal outcome of that search, not a malfunction -- but a bare "0.08 nats
below" tells the reader their optimum is wrong without telling them where to go
instead. So the scan records which parameter's scan found it, the value that
parameter held there, the resulting NLL, and the whole parameter vector in
linear units, ready to restart a fit from.

The vector is the part worth testing. It is assembled from a fixed value and a
nuisance solution that live in the optimizer's space, which for these specs is
log10, and a wrong reassembly would be a plausible-looking set of numbers that
silently is not the point that was evaluated.

Run from the repository root:
    python tests/engine/test_better_point.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.getcwd())

from pyantigen.engine.Optimize import _better_point_record                   # noqa: E402

failures = []


def check(name, cond, detail=""):
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}  {detail}")
        failures.append(name)


NAMES = ["a", "b", "c", "d"]
NLL_OPT = -3482.747


def _meta(is_log):
    return {i: {"is_log": is_log[i]} for i in range(len(is_log))}


def test_records_what_was_found():
    print("\nWhat is recorded:")
    meta = _meta([False] * 4)
    res_x = np.array([1.0, 2.0, 3.0, 4.0])
    where = ("c", 3.5, [1.1, 2.2, 4.4])          # nuisance = a, b, d
    rec = _better_point_record(NAMES, res_x, meta, -0.08112, where, NLL_OPT)

    check("a record is produced", rec is not None)
    check("it names the parameter whose scan found it",
          rec["parameter"] == "c", str(rec.get("parameter")))
    check("it gives that parameter's value there",
          rec["value"] == 3.5, str(rec.get("value")))
    check("it gives the improvement in nats",
          abs(rec["dnll"] + 0.08112) < 1e-12, str(rec.get("dnll")))
    # The absolute NLL is what a reader compares against the fit's own reported
    # loss, so it is recorded rather than left to be worked out from the gap.
    check("it gives the absolute NLL, not just the gap",
          abs(rec["nll"] - (NLL_OPT - 0.08112)) < 1e-9, str(rec.get("nll")))


def test_the_vector_is_reassembled_in_the_right_order():
    print("\nVector reassembly (linear spec):")
    meta = _meta([False] * 4)
    res_x = np.array([1.0, 2.0, 3.0, 4.0])
    where = ("c", 3.5, [1.1, 2.2, 4.4])
    rec = _better_point_record(NAMES, res_x, meta, -0.5, where, NLL_OPT)

    check("the vector has one entry per parameter",
          rec.get("x") is not None and len(rec["x"]) == 4,
          str(rec.get("x")))
    # The fixed value must land at its own index, with the nuisance solution
    # filling in around it -- inserting at the wrong position would produce a
    # vector that looks reasonable and is not the point that was evaluated.
    check("the fixed value sits at its own index",
          np.allclose(rec["x"], [1.1, 2.2, 3.5, 4.4]), str(rec["x"]))
    check("the names are recorded alongside",
          rec.get("param_names") == NAMES, str(rec.get("param_names")))


def test_log_scaled_parameters_come_back_linear():
    print("\nVector reassembly (log10 spec):")
    # Every parameter log10-fitted, as PK and SILK specs now are.
    meta = _meta([True] * 4)
    lin = np.array([3.0e-2, 1.4e-3, 6.3e-5, 2.7e-1])
    res_x = np.log10(lin)
    # The scan fixed "c" at 7.0e-5 and solved the rest, in opt space.
    nuisance_opt = np.log10([3.1e-2, 1.5e-3, 2.6e-1]).tolist()
    where = ("c", 7.0e-5, nuisance_opt)
    rec = _better_point_record(NAMES, res_x, meta, -0.5, where, NLL_OPT)

    check("a vector is produced", rec.get("x") is not None)
    # A missing 10** here would report log10 values as if they were the
    # parameters, i.e. negative rate constants.
    check("log-scaled values are returned in linear units",
          np.allclose(rec["x"], [3.1e-2, 1.5e-3, 7.0e-5, 2.6e-1]),
          str(rec.get("x")))
    check("and every one is positive",
          all(v > 0 for v in rec["x"]), str(rec.get("x")))


def test_mixed_scales():
    print("\nMixed scales:")
    meta = _meta([True, False, True, False])
    res_x = np.array([np.log10(3.0e-2), 2.0, np.log10(6.3e-5), 4.0])
    where = ("c", 7.0e-5, [np.log10(3.1e-2), 2.5, 4.5])
    rec = _better_point_record(NAMES, res_x, meta, -0.5, where, NLL_OPT)
    check("each parameter is converted by its own scale",
          np.allclose(rec["x"], [3.1e-2, 2.5, 7.0e-5, 4.5]),
          str(rec.get("x")))


def test_nothing_is_recorded_when_the_fit_is_the_minimum():
    print("\nNo finding:")
    meta = _meta([False] * 4)
    res_x = np.array([1.0, 2.0, 3.0, 4.0])
    where = ("c", 3.5, [1.1, 2.2, 4.4])
    check("an anchor at zero records nothing",
          _better_point_record(NAMES, res_x, meta, 0.0, where, NLL_OPT) is None)
    # Float noise around the optimum is not a finding; the threshold matches
    # the one the report warns on.
    check("a negligible gap records nothing",
          _better_point_record(NAMES, res_x, meta, -1e-9, where,
                               NLL_OPT) is None)
    check("no located point records nothing",
          _better_point_record(NAMES, res_x, meta, -0.5, None,
                               NLL_OPT) is None)


def test_degenerate_inputs_still_report_what_they_can():
    print("\nDegenerate inputs:")
    meta = _meta([False] * 4)
    res_x = np.array([1.0, 2.0, 3.0, 4.0])

    # A nuisance vector of the wrong length cannot be reassembled, but the
    # parameter, its value and the NLL are still worth reporting.
    rec = _better_point_record(NAMES, res_x, meta, -0.5, ("c", 3.5, [1.1]),
                               NLL_OPT)
    check("a short nuisance vector still yields the headline facts",
          rec is not None and rec["parameter"] == "c" and "x" not in rec,
          str(rec))

    rec = _better_point_record(NAMES, res_x, meta, -0.5, ("c", 3.5, None),
                               NLL_OPT)
    check("a missing nuisance vector is survivable",
          rec is not None and "x" not in rec, str(rec))

    rec = _better_point_record(NAMES, res_x, meta, -0.5,
                               ("nosuch", 3.5, [1.1, 2.2, 4.4]), NLL_OPT)
    check("an unknown parameter name records nothing", rec is None, str(rec))

    # A non-positive value cannot be put back into log space.
    rec = _better_point_record(NAMES, res_x, _meta([True] * 4), -0.5,
                               ("c", 0.0, [0.1, 0.2, 0.3]), NLL_OPT)
    check("a non-positive value on a log parameter is survivable",
          rec is not None and "x" not in rec, str(rec))


def test_zz_every_check_passed():
    assert not failures, "failed checks: " + ", ".join(failures)


def main():
    test_records_what_was_found()
    test_the_vector_is_reassembled_in_the_right_order()
    test_log_scaled_parameters_come_back_linear()
    test_mixed_scales()
    test_nothing_is_recorded_when_the_fit_is_the_minimum()
    test_degenerate_inputs_still_report_what_they_can()

    print("\n" + "=" * 72)
    if failures:
        print(f"FAILED ({len(failures)}):")
        for f in failures:
            print("  -", f)
        return 1
    print("ALL BETTER-POINT CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())

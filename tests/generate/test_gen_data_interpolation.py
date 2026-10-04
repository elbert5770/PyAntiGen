"""Antimony piecewise generators in pyantigen.generate.data_interpolation.

The generated text is checked two ways: literally for the structural bits, and
by evaluating the piecewise it describes in Python, which is what shows the
interpolation is right without needing tellurium.

Run from the repository root: python -m pytest tests/generate
"""
import re

import numpy as np
import pytest

from pyantigen.generate.data_interpolation import (
    format_number,
    generate_antimony_piecewise,
)


def _eval_piecewise(text, t):
    """Evaluate ``name := piecewise(v1, c1, v2, c2, ..., [default])`` at time t.

    Splits on top-level commas only (parentheses nest) and evaluates each
    expression with ``time`` bound. Antimony's && is Python's and.
    """
    body = text[text.index("piecewise") + len("piecewise"):].strip()
    assert body[0] == "(" and body[-1] == ")"
    body = body[1:-1].replace("\\", "").replace("&&", " and ")
    args, depth, cur = [], 0, ""
    for ch in body:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            args.append(cur.strip())
            cur = ""
        else:
            cur += ch
    args.append(cur.strip())
    env = {"time": t}
    for i in range(0, len(args) - 1, 2):
        if eval(args[i + 1], {}, env):
            return eval(args[i], {}, env)
    return eval(args[-1], {}, env) if len(args) % 2 else np.nan


# --- format_number ----------------------------------------------------------

@pytest.mark.parametrize("num, expected", [
    (1.0, "1"), (0.5, "0.5"), (-2.0, "-2"), (100.0, "100"), (1e-7, "1e-07"),
])
def test_format_number(num, expected):
    assert format_number(num) == expected


def test_format_number_precision_limits_digits():
    assert format_number(1 / 3, precision=3) == "0.333"


# --- linear -----------------------------------------------------------------

def test_linear_text_structure():
    out = generate_antimony_piecewise([0, 1, 2], [0, 1, 1])
    assert out.startswith("data := piecewise(")
    assert "time < 0" in out and "time >= 2" in out


def test_linear_interpolates_between_points():
    out = generate_antimony_piecewise([0, 10, 20], [0, 10, 0])
    assert _eval_piecewise(out, 5) == pytest.approx(5)
    assert _eval_piecewise(out, 10) == pytest.approx(10)
    assert _eval_piecewise(out, 15) == pytest.approx(5)


def test_linear_hits_every_knot_exactly():
    times, vals = [0, 1, 3, 7], [2.0, -1.0, 4.0, 4.0]
    out = generate_antimony_piecewise(times, vals)
    for t, v in zip(times, vals):
        assert _eval_piecewise(out, t) == pytest.approx(v)


def test_linear_defaults_before_zero_and_after_last_value():
    out = generate_antimony_piecewise([1, 2], [5, 9])
    assert _eval_piecewise(out, 0) == 0
    assert _eval_piecewise(out, 99) == 9


def test_linear_default_overrides():
    out = generate_antimony_piecewise([1, 2], [5, 9], default_before=-1, default_after="z")
    assert out.startswith("data := piecewise(-1, time < 1")
    assert ", z, time >= 2" in out


def test_unsorted_input_is_sorted():
    a = generate_antimony_piecewise([2, 0, 1], [4, 0, 1])
    b = generate_antimony_piecewise([0, 1, 2], [0, 1, 4])
    assert a == b


def test_repeated_times_are_averaged_within_tolerance():
    out = generate_antimony_piecewise([0, 0.005, 10], [2, 4, 8], time_tol=1e-2)
    assert _eval_piecewise(out, 0.0025) == pytest.approx(3.0, abs=0.05)
    assert "time < 0.0025" in out


def test_names_and_time_variable_are_configurable():
    out = generate_antimony_piecewise([0, 1], [0, 1], data_name="dose", time_var="t")
    assert out.startswith("dose := piecewise(")
    assert "time" not in out and "t < 0" in out


def test_flat_segment_is_a_constant():
    out = generate_antimony_piecewise([0, 1, 2], [3, 3, 5])
    assert _eval_piecewise(out, 0.5) == 3


def test_linear_needs_two_points():
    with pytest.raises(ValueError, match="At least 2"):
        generate_antimony_piecewise([1], [1])


def test_unknown_interpolation_type():
    with pytest.raises(ValueError, match="Unknown interpolation_type"):
        generate_antimony_piecewise([0, 1], [0, 1], interpolation_type="cubic-ish")


def test_interpolation_type_is_case_insensitive():
    assert generate_antimony_piecewise([0, 1], [0, 1], interpolation_type="LINEAR") == \
        generate_antimony_piecewise([0, 1], [0, 1])


# --- symbolic ---------------------------------------------------------------

def test_symbolic_keeps_names_verbatim():
    out = generate_antimony_piecewise(["t0", "t1"], ["a", "b"], symbolic=True)
    assert out == ("data := piecewise(0.0, time < t0, "
                   "(a + (b - (a)) * (time - t0) / (t1 - t0)), time < t1, "
                   "b, time > t1)")


def test_symbolic_evaluates_with_bound_names():
    out = generate_antimony_piecewise(["t0", "t1"], ["a", "b"], symbolic=True)
    env = {"time": 5, "t0": 0, "t1": 10, "a": 2, "b": 6}
    expr = out.split("piecewise(")[1]
    # second value expression: the interpolation
    interp = expr.split(", ")[2]
    assert eval(interp, {}, env) == pytest.approx(4.0)


def test_symbolic_needs_matching_lengths():
    with pytest.raises(ValueError, match="same length"):
        generate_antimony_piecewise(["t0", "t1"], ["a"], symbolic=True)


# --- spline -----------------------------------------------------------------

def test_spline_passes_through_the_data():
    times, vals = [0, 1, 2, 3], [0.0, 1.0, 0.0, 2.0]
    out = generate_antimony_piecewise(times, vals, interpolation_type="spline")
    for t, v in zip(times, vals):
        assert _eval_piecewise(out, t) == pytest.approx(v, abs=1e-5)


def test_spline_is_continuous_across_a_knot():
    out = generate_antimony_piecewise([0, 1, 2, 3], [0, 3, 1, 4], interpolation_type="spline")
    left = _eval_piecewise(out, 1 - 1e-6)
    right = _eval_piecewise(out, 1 + 1e-6)
    assert left == pytest.approx(right, abs=1e-4)


def test_monotone_spline_does_not_overshoot():
    times, vals = [0, 1, 2, 3, 4], [0, 0, 0, 10, 10]
    out = generate_antimony_piecewise(times, vals, interpolation_type="spline",
                                      monotone=True)
    samples = [_eval_piecewise(out, t) for t in np.linspace(0, 4, 81)]
    assert min(samples) >= -1e-5 and max(samples) <= 10 + 1e-5


def test_natural_spline_does_overshoot_where_monotone_does_not():
    times, vals = [0, 1, 2, 3, 4], [0, 0, 0, 10, 10]
    out = generate_antimony_piecewise(times, vals, interpolation_type="spline")
    samples = [_eval_piecewise(out, t) for t in np.linspace(0, 4, 81)]
    assert min(samples) < -1e-3 or max(samples) > 10 + 1e-3


def test_spline_needs_three_points():
    with pytest.raises(ValueError, match="At least 3"):
        generate_antimony_piecewise([0, 1], [0, 1], interpolation_type="spline")


def test_spline_line_continuation_is_on_by_default_and_can_be_disabled():
    default = generate_antimony_piecewise([0, 1, 2], [0, 1, 4], interpolation_type="spline")
    off = generate_antimony_piecewise([0, 1, 2], [0, 1, 4], interpolation_type="spline",
                                      antimony_continuation=False)
    assert re.search(r",\\ \n", default)
    assert "\\" not in off

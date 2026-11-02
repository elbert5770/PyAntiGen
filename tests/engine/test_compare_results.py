"""compare_results: RoadRunner repeats a time stamp at a block boundary (state before the switch, then after it);
AMICI reads a time once, after the switches. Only the post-switch row of a repeat is compared."""
import numpy as np

from pyantigen.engine.Simulate import StackedResult
from pyantigen.engine.Simulate_amici import compare_results


def _result(rows):
    return StackedResult(np.array(rows, dtype=float), ["time", "Q_Leak"])


def test_the_pre_switch_row_of_a_repeated_time_is_not_a_disagreement():
    # RoadRunner: row 1 is the block end (before the switch), row 2 the next block's start (after it)
    roadrunner = _result([[0.0, 0.0], [5.0, 0.0], [5.0, 0.02], [6.0, 0.02]])
    amici = _result([[0.0, 0.0], [5.0, 0.02], [5.0, 0.02], [6.0, 0.02]])      # the post-switch value in both rows
    assert compare_results(amici, roadrunner)[0][1] == 0.0


def test_a_real_disagreement_on_the_post_switch_row_is_still_reported():
    roadrunner = _result([[0.0, 0.0], [5.0, 0.0], [5.0, 0.02], [6.0, 0.02]])
    amici = _result([[0.0, 0.0], [5.0, 0.03], [5.0, 0.03], [6.0, 0.02]])
    (name, diff, _rel, at, a, b), = compare_results(amici, roadrunner)
    assert name == "Q_Leak" and at == 5.0 and abs(diff - 0.01) < 1e-12 and (a, b) == (0.03, 0.02)


def test_times_a_rounding_error_apart_count_as_a_repeat():
    roadrunner = _result([[0.0, 0.0], [5.0, 0.0], [5.0 + 1e-11, 0.02], [6.0, 0.02]])
    amici = _result([[0.0, 0.0], [5.0, 0.02], [5.0, 0.02], [6.0, 0.02]])
    assert compare_results(amici, roadrunner)[0][1] == 0.0


def test_a_time_that_is_not_repeated_is_compared_as_before():
    roadrunner = _result([[0.0, 0.0], [5.0, 0.0], [6.0, 0.02]])
    amici = _result([[0.0, 0.0], [5.0, 0.02], [6.0, 0.02]])
    assert compare_results(amici, roadrunner)[0][1] == 0.02

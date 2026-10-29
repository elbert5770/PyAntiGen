"""Profile.propose: where fill puts its next point in a bracket of the threshold."""
from types import SimpleNamespace as P

import pytest

from pyantigen.engine.Profile import propose

LEVEL = 1.9207


def pts(*xy):
    return [P(x_fixed=x, dnll=y) for x, y in xy]


def test_an_ordinary_bracket_is_interpolated():
    q = pts((0.0, 0.0), (1.0, 4.0))
    assert propose(q, 0, 1, LEVEL) == pytest.approx(LEVEL / 4.0)


def test_the_proposal_stays_inside_the_middle_of_the_bracket():
    q = pts((0.0, 0.0), (1.0, 40.0))                 # linear would say 0.048: pulled up to 10 %
    assert propose(q, 0, 1, LEVEL) == pytest.approx(0.1)
    q = pts((0.0, 1.9), (1.0, 2.0))                  # linear would say 0.3 -> inside, unchanged
    assert 0.1 <= propose(q, 0, 1, LEVEL) <= 0.9


def test_a_far_end_far_above_the_level_is_bisected():
    # a flat profile that ends in a cliff: the far point is 3e5 above, linear interpolation moved 7e-6 of the way
    q = pts((-2.87, 0.0003), (-7.0, 283582.0))
    assert propose(q, 0, 1, LEVEL) == pytest.approx(-2.87 + 0.5 * (-7.0 + 2.87))


def test_a_descending_direction_works_too():
    q = pts((0.0, 0.0), (-4.0, 1e5))
    assert propose(q, 0, 1, LEVEL) == pytest.approx(-2.0)

import numpy as np
import pytest

from pyantigen.engine.Optimize import _multistart_points

X0 = np.array([0.0, 1.0, -2.0])
BOUNDS = [(-9.0, 9.0)] * 3
SCALES = ["log10"] * 3


def _pts(sampler, n=9, seed=7):
    return _multistart_points(X0, BOUNDS, SCALES, n, search_decades=0.5,
                              seed=seed, verbose=False, sampler=sampler)


@pytest.mark.parametrize("sampler", ["lhs", "sobol"])
def test_start_one_is_x0_and_rest_inside_radius(sampler):
    pts = np.array(_pts(sampler))
    assert pts.shape == (9, 3)
    assert np.array_equal(pts[0], X0)
    assert np.all(np.abs(pts[1:] - X0) <= 0.5 + 1e-12)


@pytest.mark.parametrize("sampler", ["lhs", "sobol"])
def test_seeded_reproducible(sampler):
    assert np.array_equal(np.array(_pts(sampler)), np.array(_pts(sampler)))


def test_sobol_differs_from_lhs():
    assert not np.allclose(np.array(_pts("sobol"))[1:], np.array(_pts("lhs"))[1:])


def test_single_start_ignores_sampler():
    assert np.array_equal(_pts("sobol", n=1)[0], X0)


def test_unknown_sampler_rejected():
    with pytest.raises(ValueError, match="start_sampler"):
        _pts("halton")

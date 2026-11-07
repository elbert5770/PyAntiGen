"""The one-sided log10 priors of the AMICI residual provider (Analytic_gradient.AmiciGradient priors)."""
import numpy as np
import pytest

from pyantigen.engine.Analytic_gradient import AmiciGradient, AnalyticGradientError

NAMES = ["a", "dwell", "c"]


def provider(priors):
    g = AmiciGradient.__new__(AmiciGradient)          # the prior terms need only the names and the parsed priors
    g.param_names = NAMES
    g.priors = AmiciGradient._parse_priors(priors, NAMES)
    return g


def test_below_penalizes_under_the_reference_only():
    g = provider([{"param": "dwell", "ref": 168.0, "sigma": 0.5, "side": "below"}])
    r, J = g.prior_terms([1.0, 1680.0, 1.0])
    assert r[0] == 0.0 and not J.any() and g.last_prior_nll == 0.0
    r, J = g.prior_terms([1.0, 16.8, 1.0])                                      # a decade under: (1 / 0.5)^2 / 2 = 2 nats
    assert np.isclose(r[0], 2.0) and np.isclose(g.last_prior_nll, 2.0)
    assert J[0, 1] < 0 and not J[0, [0, 2]].any()                               # a larger p lowers the penalty


def test_the_jacobian_is_the_derivative_of_the_residual():
    for side in ("below", "above", "both"):
        g = provider([{"param": "dwell", "ref": 100.0, "sigma": 0.3, "side": side}])
        for p in (30.0, 400.0) + ((100.0,) if side == "both" else ()):          # the one-sided residual has a corner at the reference (its loss does not)
            r, J = g.prior_terms([1.0, p, 1.0])
            h = 1e-6 * p
            rp, _ = g.prior_terms([1.0, p + h, 1.0])
            rm, _ = g.prior_terms([1.0, p - h, 1.0])
            assert np.isclose(J[0, 1], (rp[0] - rm[0]) / (2 * h), rtol=1e-4, atol=1e-9), (side, p)


def test_the_loss_is_continuous_at_the_reference():
    g = provider([{"param": "dwell", "ref": 168.0, "sigma": 0.5, "side": "below"}])
    g.prior_terms([1.0, 168.0 * (1 - 1e-9), 1.0])
    assert g.last_prior_nll < 1e-15


def test_a_prior_on_a_parameter_that_is_not_fitted_is_refused():
    with pytest.raises(AnalyticGradientError):
        provider([{"param": "zzz", "ref": 1.0, "sigma": 1.0}])
    with pytest.raises(AnalyticGradientError):
        provider([{"param": "a", "ref": 1.0, "sigma": 1.0, "side": "sideways"}])

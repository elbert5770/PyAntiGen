"""The LM fit (Fit_lm): converges, respects the box, treats a failed integration as a rejected step, stops at a deadline."""
import time

import numpy as np
import pytest

from pyantigen.engine.Analytic_gradient import AnalyticGradientError
from pyantigen.engine.Fit_lm import is_lm, lm_fit, settings_from

SCALES = ["log10", "log10", "lin"]
TARGET = np.array([2.0, 0.5, -1.0])
CONST = -10.0


class Provider:
    """r = w * (p - target), so f = CONST + 0.5 r.r, minimal at the target (linear parameters, J in linear units)."""
    W = np.array([1.0, 3.0, 2.0])

    def __init__(self, fail_above=None):
        self.fail_above, self.calls = fail_above, 0

    def linear_residual_jacobian(self, p):
        self.calls += 1
        if self.fail_above is not None and p[0] > self.fail_above:
            raise AnalyticGradientError("CVODES failed")
        return self.W * (np.asarray(p) - TARGET), np.diag(self.W), CONST


def box():
    return [(-2.0, 2.0), (-2.0, 2.0), (-3.0, 3.0)]


def x_of(p):
    return np.array([np.log10(p[0]), np.log10(p[1]), p[2]])


def test_is_lm_names():
    assert is_lm("LM") and is_lm("Levenberg-Marquardt") and not is_lm("L-BFGS-B")


def test_converges_to_the_optimum_inside_the_box():
    res = lm_fit(Provider(), x_of([0.1, 5.0, 2.0]), box(), SCALES, {"maxfev": 100}, log=lambda *_: None)
    assert res.success and res.verdict == "converged"
    assert res.fun == pytest.approx(CONST, abs=1e-6)
    assert 10 ** res.x[0] == pytest.approx(2.0, rel=1e-3) and 10 ** res.x[1] == pytest.approx(0.5, rel=1e-3)
    assert res.x[2] == pytest.approx(-1.0, abs=1e-3)


def test_stays_in_the_box_when_the_optimum_is_outside():
    tight = [(-2.0, np.log10(1.5)), (-2.0, 2.0), (-3.0, 3.0)]
    res = lm_fit(Provider(), x_of([0.1, 5.0, 2.0]), tight, SCALES, {"maxfev": 100}, log=lambda *_: None)
    assert res.x[0] <= np.log10(1.5) + 1e-12
    assert 10 ** res.x[0] == pytest.approx(1.5, rel=1e-6)
    assert res.verdict == "converged"           # the projected gradient is zero on an active bound


def test_a_failed_integration_is_a_rejected_step_not_an_error():
    res = lm_fit(Provider(fail_above=3.0), x_of([0.1, 5.0, 2.0]), box(), SCALES, {"maxfev": 100},
                 log=lambda *_: None)
    assert res.success
    assert 10 ** res.x[0] == pytest.approx(2.0, rel=1e-3)


def test_the_deadline_returns_the_best_point_unsuccessfully():
    start = x_of([0.1, 5.0, 2.0])
    res = lm_fit(Provider(), start, box(), SCALES, {"maxfev": 100}, deadline=time.time() - 1, log=lambda *_: None)
    assert not res.success and res.message == "deadline reached"
    assert np.allclose(res.x, start)


def test_on_best_sees_only_improvements():
    seen = []
    lm_fit(Provider(), x_of([0.1, 5.0, 2.0]), box(), SCALES, {"maxfev": 100},
           on_best=lambda x, f, n: seen.append(f), log=lambda *_: None)
    assert seen and all(b < a for a, b in zip(seen, seen[1:]))


def test_a_provider_without_the_residual_jacobian_is_refused():
    with pytest.raises(RuntimeError, match="residual Jacobian"):
        lm_fit(object(), np.zeros(3), box(), SCALES)


def test_the_budget_comes_from_maxfev():
    assert settings_from({"maxfev": 7, "pg_tol": 0.5}).max_evals == 7
    assert settings_from(None).max_evals == 120
    res = lm_fit(Provider(), x_of([0.1, 5.0, 2.0]), box(), SCALES, {"maxfev": 2}, log=lambda *_: None)
    assert res.nfev <= 2 and not res.success

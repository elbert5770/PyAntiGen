"""Log10 finite-difference steps must fit each parameter's own curvature.

A single 0.02-decade step is a compromise that fails in the tight direction.
On the Example spec, two parameters with SEs near 0.01 decades were probed 2
SEs from the centre, outside their quadratic region, and the Wald SEs came back
12% and 18% off, with a correlation of -0.25 instead of -0.55.
compute_hessian_batched now calibrates each log10 step from a pilot pass so one
probe raises the NLL by about _FD_TARGET_RISE nats (_calibrate_log_steps).

Every objective here has the shape of the concentrated likelihood,
(N/2) log(1 + d'Ad/N) with d the displacement in *linear* units. Its Hessian at
the optimum is exactly A, so the true linear SEs and correlation are known in
closed form, and the delta method that maps the log10 answer back is exact
there. Away from the optimum it bends below the quadratic the way (n/2)log(SSE)
does -- which is what made the fixed step fail on the real model; a plain
quadratic in linear units is still nearly quadratic over a 0.02-decade step and
would not expose it. Any disagreement is the step.

Run from the repository root:
    python -m pytest tests/engine/test_hessian_steps.py
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.getcwd())

from pyantigen.engine.Optimize import (                                      # noqa: E402
    _FD_TARGET_RISE,
    _LOG_FD_STEP,
    _LOG_FD_STEP_MIN,
    _calibrate_log_steps,
    _transform_wald_to_linear,
    compute_hessian_batched,
    compute_wald_uncertainty,
)

P0 = np.array([0.204, 0.998])


N_POINTS = 8      # few data points: strongly non-quadratic past ~1 SE


def _quadratic(se, corr):
    """Concentrated-likelihood-shaped NLL with the given SEs and correlation at
    its optimum P0, as a function of the log10 coordinates."""
    se = np.asarray(se, dtype=float)
    cov = np.array([[se[0] ** 2, corr * se[0] * se[1]],
                    [corr * se[0] * se[1], se[1] ** 2]])
    A = np.linalg.inv(cov)

    def nll(q):
        d = 10.0 ** np.asarray(q, dtype=float) - P0
        return float(0.5 * N_POINTS * np.log1p(d @ A @ d / N_POINTS))
    return nll


def _counting_batch(f):
    calls = {"n": 0}

    def batch(xs, label=None):
        calls["n"] += len(xs)
        return [f(x) for x in xs]
    return batch, calls


def _linear_wald(nll, scales=("log10", "log10")):
    q0 = np.log10(P0)
    batch, calls = _counting_batch(nll)
    cov, se_q, _ci = compute_wald_uncertainty(nll, q0, nll_batch=batch,
                                              scales=list(scales))
    se_lin, _ = _transform_wald_to_linear(se_q, None, q0, list(scales))
    corr = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    return np.asarray(se_lin), corr, calls["n"]


def test_tight_parameters_get_the_right_se_and_correlation():
    # The Example's own numbers: SEs of ~0.01 decades, correlation -0.55.
    se, corr = [0.00548, 0.004878], -0.5527
    got_se, got_corr, _n = _linear_wald(_quadratic(se, corr))
    # 2% and 0.03: with only N_POINTS=8 this likelihood is far less quadratic
    # than a real block's, and the calibrated probes -- the cross terms, which
    # move two parameters at once, most -- still see some of that. The fixed
    # 0.02-decade step was 7% and 99% off on the SEs here, with a correlation
    # of -0.08. (On the Example model itself: within 0.3%, and -0.548 against
    # an exact -0.553.)
    assert np.allclose(got_se, se, rtol=0.02), got_se
    assert abs(got_corr - corr) < 0.03, got_corr


def test_loose_parameters_keep_the_default_step_at_no_extra_cost():
    # SEs of ~0.4 decades: the default probe rises far less than the target,
    # so the step must stay put and every pilot value be reused -- the whole
    # computation costs exactly the 2k^2 + 1 points it always did.
    se, corr = [0.2, 0.9], 0.3
    nll = _quadratic(se, corr)
    q0 = np.log10(P0)
    steps, known = _calibrate_log_steps(
        _counting_batch(nll)[0], q0, np.full(2, _LOG_FD_STEP),
        ["log10", "log10"], 1e10)
    assert np.all(steps == _LOG_FD_STEP)
    assert len(known) == 1 + 2 * 2          # centre + both diagonal pairs

    got_se, got_corr, n_evals = _linear_wald(nll)
    k = 2
    assert n_evals == 2 * k * k + 1
    assert np.allclose(got_se, se, rtol=0.01), got_se
    assert abs(got_corr - corr) < 0.01


def test_a_calibrated_probe_rises_by_about_the_target():
    nll = _quadratic([0.00548, 0.004878], -0.5527)
    q0 = np.log10(P0)
    steps, _known = _calibrate_log_steps(
        _counting_batch(nll)[0], q0, np.full(2, _LOG_FD_STEP),
        ["log10", "log10"], 1e10)
    f0 = nll(q0)
    for i in range(2):
        assert steps[i] < _LOG_FD_STEP
        e = np.zeros(2); e[i] = steps[i]
        rise = 0.5 * (nll(q0 + e) + nll(q0 - e)) - f0
        # Accepted once the rise is at most twice the target. Each rescaling
        # comes from a probe that sat too far out, where this likelihood has
        # flattened, so it undershoots the curvature and lands at or above the
        # target rather than below it.
        assert 0.5 * _FD_TARGET_RISE < rise <= 2.0 * _FD_TARGET_RISE, rise


def test_step_never_drops_below_the_floor():
    # An absurdly tight parameter would ask for a step far below the floor.
    nll = _quadratic([1e-7, 0.9], 0.0)
    q0 = np.log10(P0)
    steps, _ = _calibrate_log_steps(
        _counting_batch(nll)[0], q0, np.full(2, _LOG_FD_STEP),
        ["log10", "log10"], 1e10)
    assert steps[0] == _LOG_FD_STEP_MIN
    assert steps[1] == _LOG_FD_STEP


def test_a_flat_or_noisy_direction_keeps_the_default():
    # A rise that is not positive says nothing about curvature.
    def nll(q):
        return -abs(float(q[0] - np.log10(P0[0]))) + (q[1] - np.log10(P0[1])) ** 2
    steps, _ = _calibrate_log_steps(
        _counting_batch(nll)[0], np.log10(P0), np.full(2, _LOG_FD_STEP),
        ["log10", "log10"], 1e10)
    assert steps[0] == _LOG_FD_STEP


def test_linear_parameters_are_not_calibrated():
    nll = _quadratic([0.00548, 0.004878], -0.5527)
    batch, calls = _counting_batch(lambda p: nll(np.log10(p)))
    compute_hessian_batched(batch, P0, scales=["lin", "lin"])
    assert calls["n"] == 2 * 2 * 2 + 1       # no pilot pass at all


def test_a_failed_pilot_point_raises():
    nll = _quadratic([0.00548, 0.004878], -0.5527)
    q0 = np.log10(P0)

    def batch(xs, label=None):
        return [1e10 if x[0] > q0[0] else nll(x) for x in xs]
    with pytest.raises(ValueError, match="step-calibration"):
        compute_hessian_batched(batch, q0, scales=["log10", "log10"])

"""A sigma on every datapoint (``noise_column``) is scored as given, and noise the likelihood cannot honour is refused.

Run from the repository root: python -m pytest tests/engine/test_point_sigma.py

Two silent hazards of the block objective are covered:

* ``noise_column`` used to be reduced to its RMS: a block with sigma_i from 0.5 to 3 was scored as one sigma, so the objective was not the
  per-point Gaussian NLL, and the LM residual vector (sqrt(w) r / sigma) was wrong point by point (by up to 74 % in the case that found it).
  The block is now ``0.5 sum w (r / sigma_i)^2 + sum w ln sigma_i`` exactly, and its residual form and gradient follow.
* ``noise_formula`` and ``sigma_method="relative"`` were accepted and left the objective every fit minimises unchanged. They now raise.
"""
import numpy as np
import pandas as pd
import pytest

import pyantigen.engine.Optimize as O
from pyantigen.engine.Analytic_gradient import LossGradient, SimTable

T = np.linspace(0.1, 8.0, 40)
NAMES = ["a", "k", "b"]
TRUE = np.array([2.0, 0.7, 0.3])
COLUMNS = ["time", "pred"]
RNG = np.random.default_rng(5)
SD = np.linspace(0.05, 0.6, len(T))                      # a different sigma on every point, a factor 12 apart


def table(theta, t):
    a, k, b = theta
    e = np.exp(-k * t)
    data = np.stack([t, a * e + b], axis=1)
    D = np.zeros((len(t), 2, 3))
    D[:, 1, :] = np.stack([e, -a * t * e, np.ones_like(t)], axis=-1)
    return SimTable(COLUMNS, data), D


Y = table(TRUE, T)[0].data[:, 1] + RNG.normal(0, SD)


def make(extra=None, weights=None):
    obs = {"observed_variable": "pred", "time_column": "time", "data_column": "y", "noise_column": "sd"}
    obs.update(extra or {})
    replicate = {"Label": "s"}
    groups = {"g": {"loss_elements": [{"simulation": "s", "loss_config": {"observables": [obs]}}]}}
    loss = LossGradient(groups, {"s": replicate}, NAMES)
    data = pd.DataFrame({"time": T, "y": Y, "sd": SD})
    if weights is not None:
        data["w"] = weights
    df = {"s": data}

    def items(theta):
        tab, D = table(theta, T)
        return {"s": {"results": tab, "data": df, "replicate": replicate}}, {"s": D}
    return loss, items


def petab_nll(theta, w=None):
    """Gaussian NLL with the per-point sigmas, without the n/2 ln 2 pi constant."""
    w = np.ones(len(T)) if w is None else w
    r = Y - table(theta, T)[0].data[:, 1]
    return float(np.sum(w * (np.log(SD) + 0.5 * (r / SD) ** 2)))


THETA = np.array([2.2, 0.6, 0.35])


def test_a_noise_column_block_is_the_per_point_gaussian_nll():
    loss, items = make()
    it, _ = items(THETA)
    assert loss.value(it, THETA) == pytest.approx(petab_nll(THETA), rel=1e-12)
    # and it is not what one RMS sigma gives
    rms = float(np.sqrt(np.mean(SD ** 2)))
    r = Y - table(THETA, T)[0].data[:, 1]
    assert abs(petab_nll(THETA) - (len(T) * np.log(rms) + 0.5 * np.sum(r ** 2) / rms ** 2)) > 1.0


def test_the_block_counts_every_point_once_whatever_its_sigma():
    loss, items = make()
    it, _ = items(THETA)
    blocks = O.collect_loss_blocks(it, loss.groups, loss.replicates, NAMES, THETA, dict(zip(NAMES, THETA)))
    (v,) = blocks.values()
    assert v[1] == pytest.approx(len(T))                 # n for AIC/BIC, not the sum of 1 / sigma^2
    assert O.total_points(blocks) == pytest.approx(len(T))
    assert O.effective_k(NAMES, blocks) == len(NAMES)    # the sigmas are given: no extra parameter


def test_data_weights_multiply_the_per_point_term():
    w = np.linspace(0.5, 2.0, len(T))
    loss, items = make(extra={"weight_column": "w"}, weights=w)
    it, _ = items(THETA)
    assert loss.value(it, THETA) == pytest.approx(petab_nll(THETA, w), rel=1e-12)


def test_the_residual_form_is_the_loss_and_its_jacobian_is_the_gradient():
    loss, items = make()
    it, dirs = items(THETA)
    f = loss.value(it, THETA)
    r, J, const = loss.residual_jacobian(it, dirs, THETA)
    assert f == pytest.approx(const + 0.5 * float(r @ r), rel=1e-12, abs=1e-9)
    np.testing.assert_allclose(r, (Y - table(THETA, T)[0].data[:, 1]) / SD, rtol=1e-12)      # per point, not one RMS
    f2, g = loss.gradient(it, dirs, THETA)
    assert f2 == pytest.approx(f)
    np.testing.assert_allclose(J.T @ r, g, rtol=1e-6, atol=1e-6 * np.abs(g).max())
    ref = np.zeros_like(J)
    for k in range(3):
        h = 1e-6 * THETA[k]
        up, dn = THETA.copy(), THETA.copy()
        up[k] += h
        dn[k] -= h
        ref[:, k] = (loss.residual_vector(items(up)[0], up)[0] - loss.residual_vector(items(dn)[0], dn)[0]) / (2 * h)
    np.testing.assert_allclose(J, ref, rtol=1e-5, atol=1e-6 * np.abs(ref).max())


def test_the_gradient_is_the_gradient_of_the_value():
    loss, items = make()
    it, dirs = items(THETA)
    _f, g = loss.gradient(it, dirs, THETA)
    fd = np.zeros(3)
    for k in range(3):
        h = 1e-6 * THETA[k]
        up, dn = THETA.copy(), THETA.copy()
        up[k] += h
        dn[k] -= h
        fd[k] = (loss.value(items(up)[0], up) - loss.value(items(dn)[0], dn)) / (2 * h)
    np.testing.assert_allclose(g, fd, rtol=1e-5, atol=1e-6 * np.abs(fd).max())


def test_a_constant_noise_column_equals_a_declared_sigma():
    """Same sigma on every point: the per-point block and the declared-sigma block are the same likelihood."""
    sd = np.full(len(T), 0.25)
    obs_col = {"observed_variable": "pred", "time_column": "time", "data_column": "y", "noise_column": "sd"}
    obs_fix = {"observed_variable": "pred", "time_column": "time", "data_column": "y", "sigma_method": "fixed", "sigma_value": 0.25}
    out = []
    for obs in (obs_col, obs_fix):
        replicate = {"Label": "s"}
        groups = {"g": {"loss_elements": [{"simulation": "s", "loss_config": {"observables": [obs]}}]}}
        loss = LossGradient(groups, {"s": replicate}, NAMES)
        tab, _ = table(THETA, T)
        it = {"s": {"results": tab, "data": {"s": pd.DataFrame({"time": T, "y": Y, "sd": sd})}, "replicate": replicate}}
        out.append(loss.value(it, THETA))
    assert out[0] == pytest.approx(out[1], rel=1e-12)


# --- refusals ------------------------------------------------------------------------------

def run_blocks(obs_extra, data_extra=None, blocks=True):
    obs = {"observed_variable": "y", "data_column": "y", "time_column": "time"}
    obs.update(obs_extra)
    t = np.linspace(0, 10, 11)
    res = np.zeros(len(t), dtype=[("time", float), ("y", float)])
    res["time"], res["y"] = t, 1.0 + t
    df = pd.DataFrame({"time": t, "y": 1.0 + t + 0.1, "sd": np.full(len(t), 0.2)})
    for k, v in (data_extra or {}).items():
        df[k] = v
    out = {} if blocks else None
    O.loss_function_evaluated({}, {"sim": {"results": res, "data": df}}, [], loss_config={"observables": [obs]},
                              blocks=out, block_key="sim")
    return out


def test_noise_formula_is_refused_instead_of_ignored():
    with pytest.raises(ValueError, match="noise_formula"):
        run_blocks({"noise_formula": "sig"})


def test_relative_sigma_is_refused_instead_of_ignored():
    with pytest.raises(ValueError, match="relative"):
        run_blocks({"sigma_method": "relative", "sigma_value": 0.05})


def test_the_legacy_value_without_blocks_is_left_alone():
    run_blocks({"sigma_method": "relative", "sigma_value": 0.05}, blocks=False)


def test_a_missing_noise_column_is_an_error_not_an_estimated_sigma():
    with pytest.raises(ValueError, match="noise_column 'nope'"):
        run_blocks({"noise_column": "nope"})


@pytest.mark.parametrize("bad", [0.0, -1.0, np.nan])
def test_a_scored_point_without_a_usable_sigma_is_an_error(bad):
    sd = np.full(11, 0.2)
    sd[3] = bad
    with pytest.raises(ValueError, match="positive finite sigma"):
        run_blocks({"noise_column": "sd"}, {"sd": sd})


def test_a_per_point_block_cannot_be_pooled_with_one_that_has_none():
    blocks = {}
    O._record_block(blocks, "s", "y", [1.0], [1.0], known_sigma=1.0, count_weights=[1.0], log_sigma=0.0)
    with pytest.raises(ValueError, match="per-point sigma"):
        O._record_block(blocks, "s", "y", [1.0], [1.0], known_sigma=0.5)
    other = {}
    O._record_block(other, "s", "y", [1.0], [1.0], known_sigma=0.5)
    with pytest.raises(ValueError, match="per-point sigma"):
        O._record_block(other, "s", "y", [1.0], [1.0], known_sigma=1.0, count_weights=[1.0], log_sigma=0.0)

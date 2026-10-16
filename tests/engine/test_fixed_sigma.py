"""Fixed-sigma mode (``sigma_floor="fixed"``) and the residual form of the loss it makes possible.

Run from the repository root: python -m pytest tests/engine/test_fixed_sigma.py

With every sigma known the concentrated loss is ``const + 0.5 r.r`` in the scaled residuals r, so the residual
vector and its Jacobian give the loss, its gradient (J^T r) and a Gauss-Newton Hessian (J^T J). These tests give
``LossGradient`` simulations with known derivatives (a sum of exponentials, as in test_analytic_gradient.py) and
check: the default mode is the old one; fixed mode scores a floored block with its floor; the residual form is the
loss; J^T r is the gradient; the fixed objective does not depend on the order of evaluation once the floors are
seeded; the mode is in the fit and profile fingerprints and is rejected when the spec and the environment disagree.
"""
import os

import numpy as np
import pandas as pd
import pytest

from pyantigen.engine import Noise_floor
from pyantigen.engine.Analytic_gradient import AnalyticGradientError, LossGradient, SimTable
from pyantigen.engine.Fit_cache import fit_fingerprint
from pyantigen.engine.Optimize import (
    _ENGINE_ONLY_OPTIMIZER_KEYS, _freeze_floor, apply_sigma_floor_setting, sigma_floor_mode)
from pyantigen.engine.Profile_checkpoint import spec_fingerprint

T = np.linspace(0.1, 8.0, 48)
NAMES = ["a", "k", "b"]
TRUE = np.array([2.0, 0.7, 0.3])
COLUMNS = ["time", "pred"]


def table(theta, t):
    a, k, b = theta
    e = np.exp(-k * t)
    data = np.stack([t, a * e + b], axis=1)
    D = np.zeros((len(t), 2, 3))
    D[:, 1, :] = np.stack([e, -a * t * e, np.ones_like(t)], axis=-1)
    return SimTable(COLUMNS, data), D


def make(floored=True, declared=None):
    rng = np.random.default_rng(11)
    y = table(TRUE, T)[0].data[:, 1] * (1 + 0.04 * rng.standard_normal(len(T)))
    obs = {"observed_variable": "pred", "time_column": "time", "data_column": "y"}
    if floored:
        obs["sigma_floor_from_data"] = True
    if declared is not None:
        obs.update({"sigma_method": "fixed", "sigma_value": declared})
    replicate = {"Label": "s"}
    groups = {"g": {"loss_elements": [{"simulation": "s", "loss_config": {"observables": [obs]}}]}}
    loss = LossGradient(groups, {"s": replicate}, NAMES)
    df = {"s": pd.DataFrame({"time": T, "y": y})}

    def items(theta):
        tab, D = table(theta, T)
        return {"s": {"results": tab, "data": df, "replicate": replicate}}, {"s": D}
    return loss, items


@pytest.fixture(autouse=True)
def fresh(monkeypatch):
    monkeypatch.delenv("PYANTIGEN_SIGMA_FLOOR", raising=False)
    Noise_floor.clear_cache()
    yield
    Noise_floor.clear_cache()


def theta_at(scale):
    return TRUE * np.array(scale)


# --- the rule -------------------------------------------------------------------------------

def test_the_default_mode_is_cap_and_freeze_floor_is_unchanged(monkeypatch):
    assert sigma_floor_mode() == "cap"
    assert _freeze_floor(None, 0.1, None) == (None, 0.1)
    assert _freeze_floor(None, 0.1, 0.05) == (0.05, None)         # the profile's own pin at the fit's sigma_used
    assert _freeze_floor(0.2, None, None) == (0.2, None)
    monkeypatch.setenv("PYANTIGEN_SIGMA_FLOOR", "nonsense")
    assert sigma_floor_mode() == "cap"


def test_fixed_mode_scores_a_floored_block_with_its_floor_and_ignores_a_frozen_value(monkeypatch):
    monkeypatch.setenv("PYANTIGEN_SIGMA_FLOOR", "fixed")
    assert _freeze_floor(None, 0.1, None) == (0.1, None)
    assert _freeze_floor(None, 0.1, 0.05) == (0.1, None)
    assert _freeze_floor(0.2, 0.1, None) == (0.2, 0.1)             # a declared sigma is never overridden
    assert _freeze_floor(None, None, None) == (None, None)         # an unfloored block still profiles its sigma


# --- the residual form ----------------------------------------------------------------------

def test_the_residual_form_is_the_loss_and_its_jacobian_is_the_gradient(monkeypatch):
    monkeypatch.setenv("PYANTIGEN_SIGMA_FLOOR", "fixed")
    loss, items = make()
    theta = theta_at([1.05, 0.95, 1.2])
    it, dirs = items(theta)
    f = loss.value(it, theta)
    r, J, const = loss.residual_jacobian(it, dirs, theta)
    assert f == pytest.approx(const + 0.5 * float(r @ r), rel=1e-12, abs=1e-9)
    f2, g = loss.gradient(it, dirs, theta)
    assert f2 == pytest.approx(f)
    np.testing.assert_allclose(J.T @ r, g, rtol=1e-6, atol=1e-6 * np.abs(g).max())
    # J against central differences of the residual vector itself
    ref = np.zeros_like(J)
    for k in range(3):
        h = 1e-6 * theta[k]
        up, dn = theta.copy(), theta.copy()
        up[k] += h
        dn[k] -= h
        ref[:, k] = (loss.residual_vector(items(up)[0], up)[0] - loss.residual_vector(items(dn)[0], dn)[0]) / (2 * h)
    np.testing.assert_allclose(J, ref, rtol=1e-5, atol=1e-6 * np.abs(ref).max())


def test_a_declared_sigma_block_has_a_residual_form_without_the_mode(monkeypatch):
    loss, items = make(floored=False, declared=0.08)
    theta = theta_at([1.1, 0.9, 1.0])
    it, _dirs = items(theta)
    r, const = loss.residual_vector(it, theta)
    assert loss.value(it, theta) == pytest.approx(const + 0.5 * float(r @ r), rel=1e-12, abs=1e-9)


def test_the_residual_form_refuses_a_block_that_still_profiles_its_sigma(monkeypatch):
    loss, items = make(floored=True)                                # cap mode: the floor only bounds the profiled sigma
    theta = theta_at([1.1, 0.9, 1.0])
    with pytest.raises(AnalyticGradientError, match="profiles its sigma"):
        loss.residual_vector(items(theta)[0], theta)
    loss, items = make(floored=False)                               # no floor and no declared sigma at all
    monkeypatch.setenv("PYANTIGEN_SIGMA_FLOOR", "fixed")
    with pytest.raises(AnalyticGradientError, match="profiles its sigma"):
        loss.residual_vector(items(theta)[0], theta)


def test_the_residual_sink_is_released_even_when_the_collection_fails(monkeypatch):
    import pyantigen.engine.Optimize as O
    loss, items = make(floored=True)
    with pytest.raises(AnalyticGradientError):
        loss.residual_vector(items(theta_at([1.1, 0.9, 1.0]))[0], theta_at([1.1, 0.9, 1.0]))
    assert O._RESIDUAL_SINK is None


# --- one function whatever the order of evaluation --------------------------------------------

def test_with_the_floors_seeded_the_fixed_objective_does_not_depend_on_the_order_of_evaluation(monkeypatch):
    monkeypatch.setenv("PYANTIGEN_SIGMA_FLOOR", "fixed")
    loss, items = make()
    a, b = theta_at([1.0, 1.0, 1.0]), theta_at([1.3, 0.8, 1.5])
    fa = loss.value(items(a)[0], a)                                 # calibrates at a
    fb = loss.value(items(b)[0], b)
    snap = Noise_floor.export_cache()
    Noise_floor.clear_cache()
    Noise_floor.seed_cache(snap)
    assert loss.value(items(b)[0], b) == pytest.approx(fb, rel=1e-13)
    assert loss.value(items(a)[0], a) == pytest.approx(fa, rel=1e-13)


# --- the setting, and what it must not let through -----------------------------------------------

def test_the_spec_key_sets_the_mode_and_is_an_engine_key():
    from unittest import mock
    with mock.patch.dict(os.environ):                              # apply_sigma_floor_setting writes os.environ directly
        os.environ.pop("PYANTIGEN_SIGMA_FLOOR", None)
        assert "sigma_floor" in _ENGINE_ONLY_OPTIMIZER_KEYS
        assert apply_sigma_floor_setting({}) == "cap"
        assert apply_sigma_floor_setting({"sigma_floor": "fixed"}) == "fixed"
        assert sigma_floor_mode() == "fixed"                       # and so is every spawned worker's
        assert apply_sigma_floor_setting({"sigma_floor": "fixed"}) == "fixed"
        with pytest.raises(ValueError, match="agree"):
            apply_sigma_floor_setting({"sigma_floor": "cap"})
        os.environ.pop("PYANTIGEN_SIGMA_FLOOR")
        with pytest.raises(ValueError, match="must be one of"):
            apply_sigma_floor_setting({"sigma_floor": "loose"})
    assert "PYANTIGEN_SIGMA_FLOOR" not in os.environ


def _fit_hash(optimizer_kwargs=None):
    return fit_fingerprint(NAMES, TRUE, None, ["log10"] * 3, {"g": {}}, "model", "L-BFGS-B", optimizer_kwargs)[1]


def _spec_hash():
    return spec_fingerprint(NAMES, TRUE, {"g": {}}, ["log10"] * 3, "model", fixed_sigmas={("g", "pred"): 0.1})[1]


def test_the_mode_is_in_the_fit_and_profile_fingerprints_and_the_default_keeps_the_old_hash(monkeypatch):
    base_fit, base_spec = _fit_hash(), _spec_hash()
    assert _fit_hash({"sigma_floor": "cap"}) == base_fit            # an explicit default is not a new fit
    monkeypatch.setenv("PYANTIGEN_SIGMA_FLOOR", "fixed")
    assert _fit_hash() != base_fit
    assert _spec_hash() != base_spec
    assert _fit_hash({"sigma_floor": "fixed"}) == _fit_hash()       # the key and the variable name the same fit

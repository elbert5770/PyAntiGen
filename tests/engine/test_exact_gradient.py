"""The exact chain-rule gradient, and the progress-figure throttle.

Run from the repository root: python -m pytest tests/engine/test_exact_gradient.py

``LossGradient.gradient`` used to difference the whole Python loss along every sensitivity direction. When every scored
observable is a bare result column the chain rule does it in closed form, from the engine's own traversal (masks, weights,
pooling, sigma rules). These tests check that the two routes agree wherever the shortcut applies -- declared, profiled and
pooled sigmas, masked data, data outside the simulated times, weights -- that an expression observable still takes the old
route, that the shortcut evaluates the loss once, and that ``concentrated_nll_dsse`` is the derivative of
``concentrated_nll`` in each of its branches. The last tests are the throttle on the progress figure.
"""
import os
import time

import numpy as np
import pandas as pd
import pytest

from pyantigen.engine import Optimize as O
from pyantigen.engine.Analytic_gradient import LossGradient, SimTable

T_SIM = np.linspace(0.0, 6.0, 25)
NAMES = ["a", "k"]
THETA = np.array([2.0, 0.7])


def table(theta, t=T_SIM):
    """p1 = a e^{-kt}, p2 = 0.3 p1, and a column the loss never reads, with a huge sensitivity."""
    a, k = theta
    e = np.exp(-k * t)
    data = np.stack([t, a * e, 0.3 * a * e, 1e3 + a * e], axis=1)
    D = np.zeros((len(t), 4, 2))
    D[:, 1, :] = np.stack([e, -a * t * e], axis=-1)
    D[:, 2, :] = 0.3 * D[:, 1, :]
    D[:, 3, :] = 1e12 * D[:, 1, :]
    return SimTable(["time", "p1", "p2", "junk"], data), D


T_DATA = np.array([-0.5, 0.0, 0.3, 1.1, 2.4, 2.4, 3.9, 5.5, 6.0, 7.2])      # outside [0, 6] at both ends, a repeat


def frame(seed=3, nan_at=()):
    rng = np.random.default_rng(seed)
    p = table(THETA, T_DATA)[0]
    y1 = 2.0 * np.exp(-0.7 * T_DATA) * (1 + 0.05 * rng.standard_normal(len(T_DATA)))
    y2 = 0.6 * np.exp(-0.7 * T_DATA) + 0.01 * rng.standard_normal(len(T_DATA))
    for i in nan_at:
        y1[i] = np.nan
    return pd.DataFrame({"time": T_DATA, "y1": y1, "y2": y2, "w": np.linspace(0.5, 1.5, len(T_DATA))})


def make(observables, sims=("s",), element_block=None, nan_at=()):
    replicates = {s: {"Label": s} for s in sims}
    lc = {"observables": observables}
    elements = [{"simulation": s, "loss_config": lc, **({"sigma_block": element_block} if element_block else {})}
                for s in sims]
    groups = {"g": {"loss_elements": elements}}
    frames = {s: frame(seed=3 + i, nan_at=nan_at) for i, s in enumerate(sims)}
    loss = LossGradient(groups, replicates, NAMES)

    def items(theta):
        tab, D = table(theta)
        return ({s: {"results": tab, "data": frames[s], "replicate": replicates[s]} for s in sims},
                {s: D for s in sims})
    return loss, items


def obs(column="p1", data="y1", **extra):
    return {"observed_variable": column, "data_column": data, "time_column": "time", **extra}


def both(loss, items, theta=THETA * [1.07, 0.93]):
    it, dirs = items(theta)
    exact = loss._exact_gradient(it, dirs, theta)
    assert exact is not None, "the shortcut should have applied"
    loss.exact = False
    try:
        fd = loss.gradient(it, dirs, theta)
    finally:
        loss.exact = True
    return exact, fd


def agree(exact, fd):
    assert exact[0] == pytest.approx(fd[0])
    np.testing.assert_allclose(exact[1], fd[1], rtol=1e-6, atol=1e-6 * np.abs(fd[1]).max())


# --- agreement with the finite-difference route ----------------------------------------------------------

def test_a_profiled_sigma_block():
    agree(*both(*make([obs()])))


def test_two_observables_each_with_its_own_sigma():
    agree(*both(*make([obs(), obs("p2", "y2")])))


def test_two_observables_sharing_one_sigma():
    agree(*both(*make([obs(sigma_pool="s"), obs("p2", "y2", sigma_pool="s")])))


def test_a_declared_sigma():
    agree(*both(*make([obs(sigma_method="fixed", sigma_value=0.1), obs("p2", "y2")])))


def test_several_simulations_pooled_by_sigma_block():
    agree(*both(*make([obs(), obs("p2", "y2")], sims=("s1", "s2"), element_block="assay")))


def test_masked_data_points_and_data_outside_the_simulated_times():
    """NaN data drop out of the sums; times before the first and after the last simulated point clamp, as np.interp does."""
    agree(*both(*make([obs(), obs("p2", "y2")], nan_at=(2, 6))))


def test_weights():
    agree(*both(*make([obs(weight_column="w"), obs("p2", "y2", weight_column="w")])))


def test_a_column_with_a_huge_sensitivity_that_is_not_read_does_not_matter():
    """The old route needed a step per column for this; the chain rule never touches the column."""
    agree(*both(*make([obs()])))
    loss, items = make([obs()])
    it, dirs = items(THETA)
    assert np.abs(dirs["s"][:, 3, :]).max() > 1e11
    assert np.all(np.isfinite(loss._exact_gradient(it, dirs, THETA)[1]))


# --- when it does not apply ------------------------------------------------------------------------------

def test_an_expression_observable_takes_the_finite_difference_route():
    loss, items = make([obs("2*p1", "y1")])
    it, dirs = items(THETA)
    assert loss._exact_gradient(it, dirs, THETA) is None
    f, g = loss.gradient(it, dirs, THETA)                          # still works, by finite differences
    loss.exact = False
    f2, g2 = loss.gradient(it, dirs, THETA)
    loss.exact = True
    assert f == pytest.approx(f2) and np.allclose(g, g2)


def test_a_callable_observable_takes_the_finite_difference_route():
    loss, items = make([obs(lambda r: r["p1"], "y1")])
    it, dirs = items(THETA)
    assert loss._exact_gradient(it, dirs, THETA) is None


def test_the_shortcut_evaluates_the_loss_once_and_the_old_route_hundreds_of_times():
    loss, items = make([obs(), obs("p2", "y2")])
    calls = {"n": 0}
    real = loss.value

    def counted(*a, **kw):
        calls["n"] += 1
        return real(*a, **kw)
    loss.value = counted
    it, dirs = items(THETA)
    loss.gradient(it, dirs, THETA)
    assert calls["n"] == 0                                         # one traversal inside collect_loss_blocks, no value()
    loss.exact = False
    loss.gradient(it, dirs, THETA)
    assert calls["n"] >= 8


# --- the residual Jacobian (Levenberg-Marquardt) -----------------------------------------------------------

def known(column="p1", data="y1", sigma=0.08, **extra):
    return obs(column, data, sigma_method="fixed", sigma_value=sigma, **extra)


def both_j(loss, items, theta=THETA * [1.07, 0.93]):
    it, dirs = items(theta)
    exact = loss._exact_residual_jacobian(it, dirs, theta)
    assert exact is not None, "the shortcut should have applied"
    loss.exact = False
    try:
        fd = loss.residual_jacobian(it, dirs, theta)
    finally:
        loss.exact = True
    return exact, fd


def agree_j(exact, fd):
    np.testing.assert_allclose(exact[0], fd[0])
    assert exact[2] == pytest.approx(fd[2])
    np.testing.assert_allclose(exact[1], fd[1], rtol=1e-5, atol=1e-6 * np.abs(fd[1]).max())


def test_the_exact_jacobian_agrees_with_central_differences():
    agree_j(*both_j(*make([known(), known("p2", "y2", 0.02)])))


def test_the_exact_jacobian_with_masked_points_weights_and_clamped_times():
    agree_j(*both_j(*make([known(weight_column="w"), known("p2", "y2", 0.02, weight_column="w")], nan_at=(2, 6))))


def test_the_exact_jacobian_over_several_simulations():
    agree_j(*both_j(*make([known(), known("p2", "y2", 0.02)], sims=("s1", "s2"))))


def test_jacobian_transpose_residual_is_the_gradient_of_the_loss():
    """With every sigma known the loss is const + 0.5 r.r, so J^T r is dNLL/dtheta: the two exact routes must agree."""
    loss, items = make([known(), known("p2", "y2", 0.02)])
    theta = THETA * [1.07, 0.93]
    it, dirs = items(theta)
    r, J, const = loss.residual_jacobian(it, dirs, theta)
    f, g = loss.gradient(it, dirs, theta)
    np.testing.assert_allclose(J.T @ r, g, rtol=1e-6)
    assert f == pytest.approx(const + 0.5 * float(r @ r))


def test_the_jacobian_of_an_expression_observable_takes_the_finite_difference_route():
    loss, items = make([known("2*p1", "y1")])
    it, dirs = items(THETA)
    assert loss._exact_residual_jacobian(it, dirs, THETA) is None
    r, J, _ = loss.residual_jacobian(it, dirs, THETA)
    assert J.shape == (r.size, 2) and np.all(np.isfinite(J))


def test_a_profiled_sigma_still_cannot_be_a_least_squares_problem():
    from pyantigen.engine.Analytic_gradient import AnalyticGradientError
    loss, items = make([obs()])
    it, dirs = items(THETA)
    with pytest.raises(AnalyticGradientError, match="profiles its sigma"):
        loss.residual_jacobian(it, dirs, THETA)


# --- dNLL/dSSE -------------------------------------------------------------------------------------------

@pytest.mark.parametrize("block", [
    [3.0, 20.0, None, None],                 # profiled
    [3.0, 20.0, 0.05, None],                 # declared
    [3.0, 20.0, None, 10.0],                 # floored, cap not binding (sigma_hat 0.39 < 10)
    [3.0, 20.0, None, 0.1],                  # floored, cap binding
    [0.0, 20.0, None, None],                 # on the _SIGMA2_FLOOR plateau
])
def test_dnll_dsse_is_the_derivative_of_the_concentrated_nll(block):
    sse, n, ks, fl = block
    key = ("b", "o")
    d = O.concentrated_nll_dsse({key: list(block)})[key]
    h = max(1e-6 * sse, 1e-9)
    f = lambda x: O.concentrated_nll({key: [x, n, ks, fl]})
    ref = (f(sse + h) - f(max(sse - h, 0.0))) / (sse + h - max(sse - h, 0.0))
    if sse == 0.0:
        assert d == 0.0
    else:
        assert d == pytest.approx(ref, rel=1e-5)


# --- the progress figure ---------------------------------------------------------------------------------

def trace():
    return {("sim", "obs"): {"t_data": np.array([0.0, 1.0]), "y_data": np.array([1.0, 2.0]),
                             "t_sim": np.array([0.0, 1.0]), "y_sim": np.array([1.1, 1.9]), "contrib": 1.0,
                             "block": ("sim", "obs"), "floor": None}}


def fresh_state(monkeypatch, **extra):
    state = O._progress_overlay_state
    for k, v in {"timestamp": "20260101_000000", "last_render": 0.0, "last_cost": 0.0, "fig": None, "keys": None,
                 "pending": None, "figure": "final", **extra}.items():
        monkeypatch.setitem(state, k, v)
    return state


def render(tmp_path, eval_n=1, loss=1.0, **kw):
    O._render_progress_overlay(trace(), loss, loss, eval_n, str(tmp_path), param_names=["a"],
                               param_values=np.array([2.0]), model_name="M", experiment_id="T", **kw)


def pngs(tmp_path):
    return [f for f in os.listdir(tmp_path) if f.endswith(".png")]


def test_during_a_fit_only_the_json_is_written_and_the_figure_waits_for_the_end(tmp_path, monkeypatch):
    state = fresh_state(monkeypatch)
    for n in range(1, 6):                                          # five improvements, as a gradient method makes
        render(tmp_path, eval_n=n, loss=1.0 - 0.1 * n)
    assert pngs(tmp_path) == []
    assert "M_T_optimization_progress_20260101_000000.json" in os.listdir(tmp_path)
    assert state["pending"]["eval_n"] == 5                         # the latest improvement is the one that will be drawn


def test_flush_draws_one_figure_of_the_best_point_and_only_once(tmp_path, monkeypatch):
    state = fresh_state(monkeypatch)
    for n in range(1, 4):
        render(tmp_path, eval_n=n, loss=1.0 - 0.1 * n)
    O.flush_progress_overlay()
    assert pngs(tmp_path) == ["M_T_optimization_progress_20260101_000000.png"]
    assert state["pending"] is None and state["last_cost"] > 0.0
    os.remove(tmp_path / pngs(tmp_path)[0])
    O.flush_progress_overlay()                                     # nothing pending: nothing drawn
    assert pngs(tmp_path) == []


def test_a_point_that_did_not_improve_is_not_remembered(tmp_path, monkeypatch):
    state = fresh_state(monkeypatch, last_render=time.time())      # inside the 2 s window: a non-improvement returns early
    O._render_progress_overlay(trace(), 2.0, 1.0, 9, str(tmp_path), param_names=["a"], param_values=np.array([2.0]),
                               model_name="M", experiment_id="T")
    assert state["pending"] is None and pngs(tmp_path) == []


def test_a_suspended_overlay_remembers_nothing(tmp_path, monkeypatch):
    state = fresh_state(monkeypatch)
    with O._overlay_suspended():
        render(tmp_path)
    assert state["pending"] is None
    O.flush_progress_overlay()
    assert os.listdir(tmp_path) == []


def test_live_mode_redraws_during_the_fit_within_its_budget(tmp_path, monkeypatch):
    state = fresh_state(monkeypatch, figure="live")
    render(tmp_path, eval_n=1)                                     # nothing drawn yet: drawn at once
    assert len(pngs(tmp_path)) == 1 and state["last_cost"] > 0.0
    os.remove(tmp_path / pngs(tmp_path)[0])
    monkeypatch.setitem(state, "last_render", time.time())
    monkeypatch.setitem(state, "last_cost", 1000.0)                # the last figure was very expensive: not yet
    render(tmp_path, eval_n=2, loss=0.5)
    assert pngs(tmp_path) == []

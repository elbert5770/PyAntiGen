"""The fit's analytic gradient: the loss differentiated along sensitivity tables, and its plumbing.

Run from the repository root: python -m pytest tests/engine/test_analytic_gradient.py

Nothing here needs AMICI. ``LossGradient`` turns d(simulation)/d(theta) tables into
d(NLL)/d(theta) by re-evaluating the engine's own loss, so the checks give it simulations
with known derivatives (a sum of exponentials) and compare with finite differences of the
loss itself: through log10 observables, fixed and pooled sigmas, and parameters the loss names
(``SF*pred``). The rest is the optimizer side: ``AnalyticObjective``, the multi-start and single
fits taking a provider, and the settings. The AMICI half is ``test_analytic_gradient_amici.py``.
"""
import numpy as np
import pandas as pd
import pytest
from scipy.optimize import minimize

from pyantigen.engine import Multistart as M
from pyantigen.engine.Analytic_gradient import (
    AmiciGradient, AnalyticGradientError, LossGradient, SimTable, build_gradient)
from pyantigen.engine.Optimize import (
    _ENGINE_ONLY_OPTIMIZER_KEYS, _local_minimize, _multistart_points, _prepare_optimizer_kwargs,
    _resolve_multistart_limits, _run_multistart, _run_multistart_stage, _stage_signature)

T_DATA = np.array([0.0, 0.3, 0.7, 1.2, 1.9, 2.6, 3.4, 4.5, 5.2, 6.0])
SIMS = {"s1": T_DATA, "s2": T_DATA[1:-1] + 0.05}
NAMES = ["a", "k", "b", "SF"]
TRUE = np.array([2.0, 0.7, 0.3, 1.5])


def truth(theta, t):
    """pred(t) and its derivatives with respect to (a, k, b, SF); SF scales only what the loss names."""
    a, k, b = theta[0], theta[1], theta[2]
    e = np.exp(-k * t)
    y = a * e + b
    return y, np.stack([e, -a * t * e, np.ones_like(t), np.zeros_like(t)], axis=-1)


COLUMNS = ["time", "pred", "[X]"]


def table(theta, t):
    y, dy = truth(theta, t)
    data = np.stack([t, y, 0.5 * y], axis=1)
    D = np.zeros((len(t), 3, 4))
    D[:, 1, :] = dy
    D[:, 2, :] = 0.5 * dy
    return SimTable(COLUMNS, data), D


def observed_data(t, observed_log=False, scale=1.0):
    rng = np.random.default_rng(3)
    y = truth(TRUE, t)[0] * scale * (1 + 0.03 * rng.standard_normal(len(t)))
    return np.log10(y) if observed_log else y


def problem(observables, sims=("s1",), groups=None):
    replicates = {s: {"Label": s} for s in sims}
    text = str(observables[0]["observed_variable"])
    log, scale = "log10" in text, (1.5 if "SF" in text else 1.0)
    df = {s: {s: pd.DataFrame({"time": SIMS[s], "y": observed_data(SIMS[s], observed_log=log, scale=scale)})}
          for s in sims}
    lc = {"observables": [{"time_column": "time", "data_column": "y", **o} for o in observables]}
    groups = groups or {"g": {"loss_elements": [
        {"simulation": s, "loss_config": lc, "sigma_block": "pool" if len(sims) > 1 else None} for s in sims]}}
    loss = LossGradient(groups, replicates, NAMES)

    def items(theta):
        out, dirs = {}, {}
        for s in sims:
            tab, D = table(theta, SIMS[s])
            out[s] = {"results": tab, "data": df[s], "replicate": replicates[s]}
            dirs[s] = D
        return out, dirs
    return loss, items


def fd_gradient(loss, items, theta, rel=1e-4):
    g = np.zeros(len(theta))
    for k in range(len(theta)):
        h = rel * max(abs(theta[k]), 1e-12)

        def f(x):
            return loss.value(items(x)[0], x)
        up, dn, up2, dn2 = (theta.copy() for _ in range(4))
        up[k] += h; dn[k] -= h; up2[k] += h / 2; dn2[k] -= h / 2
        c1, c2 = (f(up) - f(dn)) / (2 * h), (f(up2) - f(dn2)) / h
        g[k] = (4 * c2 - c1) / 3
    return g


def check(loss, items, theta=None, rtol=1e-7):
    theta = TRUE * np.array([1.1, 0.9, 1.2, 1.0]) if theta is None else theta
    it, dirs = items(theta)
    f, g = loss.gradient(it, dirs, theta)
    assert f == pytest.approx(loss.value(it, theta))
    ref = fd_gradient(loss, items, theta)
    np.testing.assert_allclose(g, ref, rtol=rtol, atol=rtol * np.abs(ref).max())
    return g


# --- the loss differentiated along the tables -------------------------------------------------

def test_plain_observable_with_a_profiled_sigma():
    loss, items = problem([{"observed_variable": "pred"}])
    g = check(loss, items)
    assert np.abs(g[:3]).min() > 1e-3                   # the check is not vacuous
    assert g[3] == 0.0                                   # SF is nowhere in this loss


def test_a_log10_observable_with_a_floor():
    loss, items = problem([{"observed_variable": "np.log10(np.maximum(pred, 1e-12))"}])
    check(loss, items)


def test_a_species_column_by_its_bracketed_name():
    loss, items = problem([{"observed_variable": "[X]"}])
    check(loss, items)


def test_a_parameter_the_loss_names_gets_its_explicit_partial_derivative():
    loss, items = problem([{"observed_variable": "SF*pred"}])
    assert loss.explicit == {NAMES.index("SF")}
    g = check(loss, items)
    assert abs(g[NAMES.index("SF")]) > 1e-3              # SF moves the loss only through the expression


def test_a_declared_sigma_and_pooled_blocks_across_two_simulations():
    loss, items = problem([{"observed_variable": "pred", "sigma_method": "fixed", "sigma_value": 0.07}],
                          sims=("s1", "s2"))
    check(loss, items)
    loss, items = problem([{"observed_variable": "pred"}], sims=("s1", "s2"))
    check(loss, items)


def test_frozen_sigmas_change_the_loss_and_its_gradient_the_way_a_profile_point_scores_it():
    """A profile point pins every floored block's sigma (frozen_sigmas). The gradient of that loss is not the
    gradient of the free one, and it matches differences of the frozen loss."""
    from pyantigen.engine.Noise_floor import clear_cache
    from pyantigen.engine.Optimize import block_sigmas, collect_loss_blocks
    clear_cache()
    try:
        loss, items = problem([{"observed_variable": "pred", "sigma_floor_from_data": True}], sims=("s1", "s2"))
        theta = TRUE * np.array([1.1, 0.9, 1.2, 1.0])
        it, dirs = items(theta)
        blocks = collect_loss_blocks(it, loss.groups, loss.replicates, loss.param_names, theta,
                                     dict(zip(NAMES, theta.tolist())))
        sigmas = {k: 0.5 * v for k, v in block_sigmas(blocks).items()}
        assert sigmas and any(v[3] is not None for v in blocks.values()), "the blocks must be floored"
        free = loss.gradient(it, dirs, theta)[1]

        free_value = loss.value(it, theta)
        loss.frozen_sigmas = sigmas
        assert loss.value(it, theta) != pytest.approx(free_value)       # the frozen loss is a different loss
        frozen = check(loss, items)                     # matches finite differences of the frozen loss
        assert np.abs(frozen[:3] - free[:3]).max() > 1e-2 * np.abs(free[:3]).max()
        loss.frozen_sigmas = None
        np.testing.assert_allclose(loss.gradient(it, dirs, theta)[1], free)          # released: back to the free one
    finally:
        clear_cache()


def test_amici_gradient_freezes_and_releases_sigmas_on_its_loss():
    provider = AmiciGradient.__new__(AmiciGradient)
    provider.loss = LossGradient({}, {}, NAMES)
    provider.freeze_sigmas({("pool", "y"): 0.1})
    assert provider.loss.frozen_sigmas == {("pool", "y"): 0.1}
    provider.freeze_sigmas(None)
    assert provider.loss.frozen_sigmas is None


def test_a_callable_observable_makes_every_parameter_explicit():
    loss, items = problem([{"observed_variable": lambda res: np.asarray(res["pred"]) * 1.0}])
    assert loss.explicit == set(range(len(NAMES)))
    check(loss, items)


def test_a_direction_of_zeros_gives_a_zero_gradient():
    loss, items = problem([{"observed_variable": "pred"}])
    it, dirs = items(TRUE)
    f, g = loss.gradient(it, {s: np.zeros_like(D) for s, D in dirs.items()}, TRUE)
    assert np.all(g == 0.0)


def test_a_loss_with_no_data_is_an_error_not_a_zero():
    loss, items = problem([{"observed_variable": "pred"}])
    it, dirs = items(TRUE)
    for s in it:
        it[s]["data"] = {s: pd.DataFrame({"time": [], "y": []})}
    with pytest.raises(AnalyticGradientError):
        loss.gradient(it, dirs, TRUE)


def test_the_step_is_chosen_relative_to_the_smallest_entry_so_a_log_loss_stays_linear():
    # y spans 8 decades: a step set by the column maximum would move the small entries by orders of magnitude
    t = np.linspace(0, 12, 25)
    theta = np.array([3.0, 1.5, 1e-9, 1.0])
    df = {"s1": pd.DataFrame({"time": t, "y": np.log10(truth(theta, t)[0] * 1.02)})}
    lc = {"observables": [{"observed_variable": "np.log10(np.maximum(pred, 1e-12))", "time_column": "time",
                           "data_column": "y"}]}
    groups = {"g": {"loss_elements": [{"simulation": "s1", "loss_config": lc}]}}
    reps = {"s1": {"Label": "s1"}}
    loss = LossGradient(groups, reps, NAMES)

    def items(x):
        tab, D = table(x, t)
        return {"s1": {"results": tab, "data": df, "replicate": reps["s1"]}}, {"s1": D}
    check(loss, items, theta=np.array([3.3, 1.4, 2e-9, 1.0]), rtol=1e-6)


# --- the optimizer side -------------------------------------------------------------------------

def _quadratic():
    center = np.array([1.0, -2.0, 0.5])
    return (lambda x: float(np.sum((np.asarray(x) - center) ** 2) + 1.0),
            lambda x: 2.0 * (np.asarray(x) - center), center)


def test_analytic_objective_returns_the_engine_value_and_the_provider_gradient():
    f, g, c = _quadratic()
    ao = M.AnalyticObjective(f, g, None, ["lin"] * 3)
    x = np.array([0.0, 0.0, 0.0])
    val, grad = ao(x)
    assert val == f(x)
    np.testing.assert_allclose(grad, g(x))
    assert (ao.n_calls, ao.n_evals, ao.n_fallback) == (1, 1, 0)


def test_a_failed_objective_is_a_plateau_with_a_zero_gradient():
    ao = M.AnalyticObjective(lambda x: 1e10, lambda x: np.ones(2), None, ["lin"] * 2, penalty=55.0)
    val, grad = ao(np.zeros(2))
    assert val == 55.0 and np.all(grad == 0.0)
    ao = M.AnalyticObjective(lambda x: float("nan"), lambda x: np.ones(2), None, ["lin"] * 2)
    assert ao(np.zeros(2))[0] == 1e10


def test_a_provider_that_fails_at_one_point_is_replaced_there_by_finite_differences():
    f, g, c = _quadratic()
    calls = []

    def flaky(x):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("AMICI gave up")
        return g(x)

    ao = M.AnalyticObjective(f, flaky, [(-5, 5)] * 3, ["lin"] * 3)
    x1, x2 = np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0])
    ao(x1)
    _v, grad = ao(x2)
    np.testing.assert_allclose(grad, g(x2), rtol=2e-3, atol=2e-3)       # forward differences at 1e-3
    assert ao.n_fallback == 1 and "AMICI gave up" in ao.last_error
    assert ao.n_evals > 2 + 3                                            # the finite differences cost evaluations


def test_a_non_finite_gradient_counts_as_a_failure():
    f, g, c = _quadratic()
    ao = M.AnalyticObjective(f, lambda x: np.array([np.nan, 0.0, 0.0]), [(-5, 5)] * 3, ["lin"] * 3)
    _v, grad = ao(np.zeros(3))
    assert np.all(np.isfinite(grad)) and ao.n_fallback == 1


def test_a_setup_error_is_never_hidden_by_a_fallback():
    def bad(x):
        raise M.GradientSetupError("no schedule")
    ao = M.AnalyticObjective(lambda x: 1.0, bad, None, ["lin"])
    with pytest.raises(M.GradientSetupError):
        ao(np.zeros(1))


def test_scipy_converges_with_the_analytic_objective_in_few_evaluations():
    f, g, c = _quadratic()
    ao = M.AnalyticObjective(f, g, [(-5, 5)] * 3, ["lin"] * 3)
    res = minimize(ao, np.zeros(3), jac=True, method="L-BFGS-B", bounds=[(-5, 5)] * 3)
    np.testing.assert_allclose(res.x, c, atol=1e-6)
    assert ao.n_evals == ao.n_calls <= 8


def test_local_minimize_uses_the_provider_for_a_gradient_method_and_not_otherwise():
    f, g, c = _quadratic()
    counted = []

    def counting(x):
        counted.append(1)
        return f(x)

    res = _local_minimize(counting, np.zeros(3), "L-BFGS-B", [(-5, 5)] * 3, {}, g, ["lin"] * 3)
    np.testing.assert_allclose(res.x, c, atol=1e-6)
    assert res.gradient["scheme"] == "analytic" and res.nfev == len(counted)
    # Nelder-Mead takes no gradient: the provider is never asked
    asked = []
    res = _local_minimize(f, np.zeros(3), "Nelder-Mead", None, {"options": {"xatol": 1e-6, "fatol": 1e-10}},
                          lambda x: asked.append(1), ["lin"] * 3)
    assert not asked and not hasattr(res, "gradient")


def test_the_multistart_fits_take_the_provider_and_record_it():
    f, g, c = _quadratic()
    summary = {}
    best, records, _i = _run_multistart(
        f, [np.array([3.0, 3.0, 3.0]), np.array([-3.0, 1.0, -2.0])], "L-BFGS-B", [(-5, 5)] * 3, {},
        ["lin"] * 3, screen=False, verbose=False, summary=summary, analytic=g)
    np.testing.assert_allclose(best.x, c, atol=1e-6)
    for rec in records:
        assert rec["gradient"]["scheme"] == "analytic" and rec["gradient"]["n_fallback"] == 0
        assert rec["nfev"] == rec["gradient"]["n_gradients"]            # one evaluation per gradient, no more


def test_a_gradient_free_method_ignores_the_provider_in_the_multistart():
    f, g, c = _quadratic()
    asked = []
    best, records, _i = _run_multistart(
        f, [np.array([3.0, 3.0, 3.0])], "Nelder-Mead", [(-5, 5)] * 3,
        {"options": {"xatol": 1e-6, "fatol": 1e-10, "maxiter": 500}}, ["lin"] * 3, screen=False, verbose=False,
        analytic=lambda x: asked.append(1))
    assert not asked and "gradient" not in records[0]


# --- settings -----------------------------------------------------------------------------------

def test_the_analytic_gradient_key_is_engine_only_and_stripped_before_scipy():
    assert "analytic_gradient" in _ENGINE_ONLY_OPTIMIZER_KEYS
    kw = _prepare_optimizer_kwargs("L-BFGS-B", {"analytic_gradient": {"rtol": 1e-9}, "options": {}}, False,
                                   None, None)
    assert "analytic_gradient" not in kw


def test_settings_are_validated_and_absent_means_none():
    assert AmiciGradient.resolve({}) is None
    assert AmiciGradient.resolve({"analytic_gradient": None}) is None
    on = AmiciGradient.resolve({"analytic_gradient": True})
    assert on["provider"] == "amici" and on["rtol"] == 1e-10
    assert AmiciGradient.resolve({"analytic_gradient": {"rtol": 1e-8}})["rtol"] == 1e-8
    for bad in ({"rtl": 1e-8}, {"provider": "jax"}):
        with pytest.raises(ValueError):
            AmiciGradient.resolve({"analytic_gradient": bad})
    assert build_gradient({}) is None


def test_a_stage_with_an_analytic_gradient_has_its_own_checkpoint_signature():
    args = ([np.zeros(2)], {"n_candidates": 1, "keep_fraction": 1, "cluster_radius": 0.1,
                            "tolerance_factor": 2, "parallel": False}, {}, "L-BFGS-B", {}, 1)
    plain = _stage_signature(*args, gradient=M.resolve_gradient({}))
    assert _stage_signature(*args, gradient=M.resolve_gradient({}), analytic=False) == plain
    assert _stage_signature(*args, gradient=M.resolve_gradient({}), analytic=True) != plain


def test_the_multistart_stage_hands_the_provider_to_its_local_fits_and_builds_no_pool_for_them():
    scales, bounds = ["log10", "log10"], [(-6.0, 6.0)] * 2

    def valley(x):
        return float(np.sum((np.asarray(x, float) - 1.0) ** 2))

    def grad(x):
        return 2.0 * (np.asarray(x, float) - 1.0)

    x0 = np.array([-2.0, -2.0])
    cands = _multistart_points(x0, bounds, scales, 17, search_decades=4.0, seed=3, verbose=False,
                               sampler="sobol")
    tri = {**M.TRIAGE_DEFAULTS, "n_candidates": 16, "keep_fraction": 0.5, "cluster_radius": 0.3,
           "parallel": False}
    built = []
    start, records, report = _run_multistart_stage(
        valley, cands, x0, bounds, scales, "L-BFGS-B", {"options": {"maxiter": 25}}, tri,
        _resolve_multistart_limits({}), n_fits=3, method="Nelder-Mead", verbose=False,
        pool_builder=lambda: built.append(1), gradient=M.resolve_gradient({"multistart_gradient":
                                                                           {"parallel": True}}),
        analytic=grad)
    assert built == []                                        # the gradients need no pool
    assert report["gradient"]["scheme"] == "analytic"
    assert all(r["gradient"]["scheme"] == "analytic" and r["gradient"]["n_fallback"] == 0 for r in records)
    np.testing.assert_allclose(start, [1.0, 1.0], atol=1e-4)


# ---- freezing initial assignments ------------------------------------------------------------------

FREEZE_MODEL = """
model freeze
  species S in C; C = 1; S = 1;
  J: S -> ; k_decay*S;
  k_decay = 0.5;
  k_fit = 3;
  k_const = 9.56e-7*70;
  V_base = 2; V = V_base;
  Q_in = 4; Q_g = Q_in*0.5;
end
"""


def _freeze_inputs():
    import libsbml
    import tellurium as te
    from pyantigen.engine.Analytic_gradient import freeze_stale_assignments
    r = te.loada(FREEZE_MODEL)
    r["k_const"] = 0.25           # what a project's Update_parameters does to such a parameter
    sbml = r.getSBML()
    return libsbml, freeze_stale_assignments, r, sbml


def _initial_assigned(libsbml, sbml):
    model = libsbml.readSBMLFromString(sbml).getModel()
    return {model.getInitialAssignment(i).getSymbol() for i in range(model.getNumInitialAssignments())}


def test_freezing_everything_makes_every_initial_assigned_parameter_plain_at_the_engines_value():
    libsbml, freeze, r, sbml = _freeze_inputs()
    before = _initial_assigned(libsbml, sbml)
    assert {"k_const", "V", "Q_g"} <= before
    out, frozen = freeze(sbml, r, everything=True)
    assert set(frozen) >= {"k_const", "V", "Q_g"}
    assert not _initial_assigned(libsbml, out) & {"k_const", "V", "Q_g"}
    model = libsbml.readSBMLFromString(out).getModel()
    assert model.getParameter("k_const").getValue() == pytest.approx(0.25), "the value RoadRunner holds, not the SBML's"
    assert model.getParameter("V").getValue() == pytest.approx(2.0)


def test_without_everything_only_the_parameters_that_read_a_varying_one_are_frozen():
    libsbml, freeze, r, sbml = _freeze_inputs()
    out, frozen = freeze(sbml, r, varying=["Q_in"])
    assert frozen == ["Q_g"]
    assert {"k_const", "V"} <= _initial_assigned(libsbml, out)


def test_a_protected_variable_and_what_it_is_built_from_are_never_frozen():
    libsbml, freeze, r, sbml = _freeze_inputs()
    out, frozen = freeze(sbml, r, protected=["Q_g"], everything=True)
    assert "Q_g" not in frozen and "Q_g" in _initial_assigned(libsbml, out)


def test_the_seed_setting_is_accepted_and_defaults_to_none():
    got = AmiciGradient.resolve({"analytic_gradient": {"seed": {"AB40_BrainISF": 1e-12}}})
    assert got["seed"] == {"AB40_BrainISF": 1e-12}
    assert AmiciGradient.resolve({"analytic_gradient": True})["seed"] is None
    with pytest.raises(ValueError, match="unknown analytic_gradient key"):
        AmiciGradient.resolve({"analytic_gradient": {"seeds": {}}})

"""The tangent predictor of the profile walk and the stalled classification of a profile point.

The walk used to start its first hop with the optimum's own nuisance vector (no slope to extrapolate along yet), so on a ridge the hop landed far from the valley, spent its whole
evaluation budget on a worse branch, and every later step was chained from it. ``tangent_matrix`` gives the direction the nuisance parameters move along the profile from the
Gauss-Newton Hessian at the optimum; ``predict_start`` uses it for the first hop and interpolates between usable neighbours after that.
"""
import math

import numpy as np

from pyantigen.engine import Profile as PR
from pyantigen.engine import Profile_lm as L
from pyantigen.engine import Profile_record as P
from pyantigen.engine.Optimize import profile_convergence_report

NAMES = ["a", "b", "c"]


def evaluate(x):
    """r = (a - 1, b - a^2, c - 3): b follows a on a curve (db/da = 2a = 2 at the optimum), c sits at 3."""
    r = np.array([x[0] - 1.0, x[1] - x[0] ** 2, x[2] - 3.0])
    J = np.array([[1.0, 0, 0], [-2.0 * x[0], 1.0, 0], [0, 0, 1.0]])
    return 0.5 * float(r @ r), r, J


BEST = np.array([1.0, 1.0, 3.0])


class Recording(L.LmSolver):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.requests = []

    def solve(self, req):
        self.requests.append(req)
        return super().solve(req)


def problem(tangent=True, lb=(-9.0, -99.0, -9.0), ub=(9.0, 99.0, 9.0)):
    solver = Recording(evaluate, NAMES, ["lin"] * 3, lb, ub)
    T = PR.tangent_matrix(evaluate(BEST)[2], BEST, lb, ub) if tangent else None
    return PR.ProfileProblem(names=NAMES, lb=np.array(lb), ub=np.array(ub), best_x=BEST.copy(), anchor_nll=0.0, solver=solver,
                             best_value=lambda k: float(BEST[k]), sd_first=np.full(3, 0.5), tangent=T)


def test_tangent_matrix_of_a_curved_valley():
    T = PR.tangent_matrix(evaluate(BEST)[2], BEST, np.full(3, -9.0), np.full(3, 9.0))
    np.testing.assert_allclose(T[:, 0], [1.0, 2.0, 0.0], atol=1e-6)         # b follows a with slope 2a, c does not move
    np.testing.assert_allclose(T[:, 1], [0.4, 1.0, 0.0], atol=1e-6)         # hand value: H_NN = diag(5, 1), H_Nk = (-2, 0)
    np.testing.assert_allclose(T[:, 2], [0.0, 0.0, 1.0], atol=1e-6)


def test_a_ridge_that_involves_the_profiled_parameter_is_recovered():
    # a flat direction v through parameters 1, 2 and 3 (the microglia fit: two clearance rates and a gain that act as products): H v = 0
    rng = np.random.default_rng(3)
    v = np.array([0.0, 1.0, 1.0, -1.0, 0.0])
    P_ = np.eye(5) - np.outer(v, v) / (v @ v)
    J = rng.normal(size=(12, 5)) @ P_
    x = np.zeros(5)
    T = PR.tangent_matrix(J, x, np.full(5, -9.0), np.full(5, 9.0))
    for k in (1, 2, 3):
        np.testing.assert_allclose(T[:, k], v / v[k], atol=1e-5)               # profiling any of the three moves the other two 1:1 along v
    H = J.T @ J
    np.testing.assert_allclose((H @ T[:, 0])[1:], 0.0, atol=1e-6)            # parameter 0 couples to the others: its column is stationary in them (the constrained minimum)
    assert T[0, 0] == 1.0


def test_a_nuisance_on_a_bound_is_held_there():
    lb, ub = np.array([-9.0, -99.0, 3.0]), np.array([9.0, 99.0, 9.0])      # c sits on its lower bound at the optimum
    T = PR.tangent_matrix(evaluate(BEST)[2], BEST, lb, ub)
    assert T[2, 0] == 0.0 and T[2, 1] == 0.0 and T[2, 2] == 1.0


def _rec(x_fixed, x, verdict="converged", pg=0.005, source="walk", reason=""):
    return P.PointRecord(param="a", x_fixed=x_fixed, nll=0.0, dnll=0.1, verdict=verdict, param_idx=0, x=list(x), pg=pg, reason=reason, source=source)


def test_predict_start_interpolates_extrapolates_and_follows_the_tangent():
    pr, opt = problem(), _rec(1.0, [1.0, 1.0, 3.0], source="optimum")
    p1 = _rec(1.4, [1.4, 1.8, 3.0])
    p2 = _rec(1.8, [1.8, 2.6, 3.0])
    # between two usable points: linear interpolation
    np.testing.assert_allclose(PR.predict_start(pr, 0, 1.6, [opt, p1, p2], +1), [1.6, 2.2, 3.0])
    # a point at the target is its own start
    np.testing.assert_allclose(PR.predict_start(pr, 0, 1.4, [opt, p1, p2], +1), [1.4, 1.8, 3.0])
    # beyond the last two: their secant
    np.testing.assert_allclose(PR.predict_start(pr, 0, 2.2, [opt, p1, p2], +1), [2.2, 3.4, 3.0], atol=1e-12)
    # only the optimum: the tangent of the profile (db/da = 2), damped by half by default and whole on request; without a tangent the old behaviour, the optimum's vector
    np.testing.assert_allclose(PR.predict_start(pr, 0, 1.5, [opt], +1), [1.5, 1.5, 3.0], atol=1e-6)
    np.testing.assert_allclose(PR.predict_start(pr, 0, 1.5, [opt], +1, PR.WalkSettings(tangent_damping=1.0)), [1.5, 2.0, 3.0], atol=1e-6)
    np.testing.assert_allclose(PR.predict_start(problem(tangent=False), 0, 1.5, [opt], +1), [1.5, 1.0, 3.0])
    np.testing.assert_allclose(PR.predict_start(pr, 0, 1.5, [opt], +1, PR.WalkSettings(use_tangent=False)), [1.5, 1.0, 3.0])


def test_a_point_far_from_stationary_is_not_a_start_but_a_stalled_one_is():
    pr, opt = problem(), _rec(1.0, [1.0, 1.0, 3.0], source="optimum")
    wrong = _rec(1.4, [1.4, 9.0, 3.0], verdict="not_converged", pg=53.0, reason="evaluation budget spent")
    far = _rec(1.8, [1.8, 2.6, 3.0])
    np.testing.assert_allclose(PR.predict_start(pr, 0, 1.6, [opt, wrong, far], +1), [1.6, 2.2, 3.0])       # interpolated between the optimum and the far point, the wrong one ignored
    stalled = _rec(1.4, [1.4, 1.5, 3.0], verdict="not_converged", pg=0.03, reason="no model decrease available")
    np.testing.assert_allclose(PR.predict_start(pr, 0, 1.6, [opt, stalled, far], +1), [1.6, 2.05, 3.0])    # the stalled point seeds its neighbours


def test_the_first_hop_starts_along_the_tangent_and_needs_fewer_evaluations():
    with_t, without = problem(tangent=True), problem(tangent=False)
    pts_t, _ = PR.walk(with_t, 0, +1)
    pts_n, _ = PR.walk(without, 0, +1)
    first_t, first_n = with_t.solver.requests[0], without.solver.requests[0]
    h = first_t.x_fixed - 1.0
    np.testing.assert_allclose(first_t.x_start[1], 1.0 + 0.5 * 2.0 * h, atol=1e-6)      # along db/da = 2, damped by the default half
    np.testing.assert_allclose(first_n.x_start[1], 1.0)                            # the optimum's own vector
    assert pts_t[1].evals <= pts_n[1].evals
    assert all(p.verdict == "converged" for p in pts_t)
    for p in pts_t[1:]:                                                            # the profile itself is unchanged: dnll = 0.5 (a - 1)^2
        assert math.isclose(p.dnll, 0.5 * (p.x_fixed - 1.0) ** 2, abs_tol=2e-3)


def test_stalled_classification():
    ok = dict(param="a", x_fixed=1.0, nll=0.0, dnll=0.2, param_idx=0)
    assert P.PointRecord(**ok, verdict="not_converged", pg=0.03, reason="no model decrease available (pred 2e-12)").stalled()
    assert not P.PointRecord(**ok, verdict="not_converged", pg=0.5, reason="no model decrease available").stalled()          # not near stationary
    assert not P.PointRecord(**ok, verdict="not_converged", pg=0.03, reason="evaluation budget spent").stalled()              # a budget problem
    assert not P.PointRecord(**ok, verdict="converged", pg=0.009, reason="projected gradient 0.009").stalled()
    assert not P.PointRecord(param="a", x_fixed=1.0, verdict="failed").stalled()
    assert P.PointRecord(**ok, verdict="not_converged", pg=0.3, reason="no model decrease available").stalled(pg_max=0.5)


def test_the_convergence_report_keeps_stalled_points_apart_from_capped_ones(capsys):
    completed = {"a": {1: {"converged": True, "dnll": 0.1, "x_fixed": 1.1},
                       2: {"converged": False, "stalled": True, "dnll": 0.2, "x_fixed": 1.2},
                       3: {"converged": False, "stalled": False, "dnll": 0.3, "x_fixed": 1.3},
                       4: {"converged": False, "dnll": 0.4, "x_fixed": 1.4}}}
    rep = profile_convergence_report(["a"], completed)
    assert rep["n_points"] == 4 and rep["n_not_converged"] == 2 and rep["n_stalled"] == 1
    assert rep["per_param"]["a"]["n_stalled"] == 1


def test_the_tangent_displacement_is_clipped_per_nuisance():
    pr, opt = problem(), _rec(1.0, [1.0, 1.0, 3.0], source="optimum")
    unit = PR._unit(pr, 1)                             # a linear parameter's "decade" is its declared span (198 here)
    loose = PR.WalkSettings(tangent_damping=1.0, tangent_clip=0.1)
    np.testing.assert_allclose(PR.predict_start(pr, 0, 1.5, [opt], +1, loose)[1], 1.0 + 1.0, atol=1e-6)          # db = 2 x 0.5 is under the cap of 19.8
    tight = PR.WalkSettings(tangent_damping=1.0, tangent_clip=0.2 / unit)                                        # a cap of 0.2
    np.testing.assert_allclose(PR.predict_start(pr, 0, 1.5, [opt], +1, tight)[1], 1.0 + 0.2, atol=1e-6)
    np.testing.assert_allclose(PR.predict_start(pr, 0, 0.5, [opt], -1, tight)[1], 1.0 - 0.2, atol=1e-6)          # and symmetric below the optimum


def test_the_first_hop_can_be_capped():
    opt = [_rec(1.0, [1.0, 1.0, 3.0], source="optimum")]
    assert math.isclose(PR.plan_step(opt, 1.0, 0.5, PR.WalkSettings()), 0.5)                                    # 4 x the Wald sd (2), held to h_max, as before
    assert math.isclose(PR.plan_step(opt, 1.0, 0.5, PR.WalkSettings(first_hop_max=0.05), unit=1.0), 0.05)       # capped, in decades
    assert math.isclose(PR.plan_step(opt, 1.0, 0.5, PR.WalkSettings(first_hop_max=0.05), unit=2.0), 0.1)        # in units of the parameter's scale
    assert math.isclose(PR.plan_step(opt, 1.0, 0.01, PR.WalkSettings(first_hop_max=0.05), unit=1.0), 0.04)      # a smaller natural hop is left alone


def test_the_walk_settings_and_store_tag_come_from_the_environment(monkeypatch):
    from pyantigen.engine import Profile_amici as PA
    from pyantigen.engine.Profile_checkpoint import default_run_id
    mode, st = PA.settings_from_env({})
    assert st.first_hop_max is None and st.point_max_evals == 40
    mode, st = PA.settings_from_env({"PYANTIGEN_PROFILE_FIRST_HOP": "0.15", "PYANTIGEN_PROFILE_POINT_EVALS": "100"})
    assert st.first_hop_max == 0.15 and st.point_max_evals == 100
    monkeypatch.delenv("PYANTIGEN_PROFILE_STORE_TAG", raising=False)
    assert default_run_id("grp", "aaaaaaaabbbb", "ccccccccdddd") == "grp_aaaaaaaa_cccccccc"
    monkeypatch.setenv("PYANTIGEN_PROFILE_STORE_TAG", "tangent1")
    assert default_run_id("grp", "aaaaaaaabbbb", "ccccccccdddd") == "grp_aaaaaaaa_cccccccc_tangent1"

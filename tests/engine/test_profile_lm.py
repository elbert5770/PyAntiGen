"""Bounded LM and the LM nuisance solver, on residual problems with a known profile."""
import math

import numpy as np
import pytest

from pyantigen.engine import Profile_lm as L
from pyantigen.engine import Profile_record as P


def evaluate(x):
    """r = (x0 - 1, x1 - 2 x0, x2 - 3): profile of x0 at v is 0.5 (v - 1)^2 (x1 follows, x2 sits at 3)."""
    r = np.array([x[0] - 1.0, x[1] - 2.0 * x[0], x[2] - 3.0])
    J = np.array([[1.0, 0, 0], [-2.0, 1.0, 0], [0, 0, 1.0]])
    return 0.5 * float(r @ r), r, J


NAMES = ["a", "b", "c"]
LB, UB = np.array([-5.0, -5.0, -5.0]), np.array([5.0, 5.0, 5.0])


def test_minimize_converges_to_the_optimum():
    out = L.minimize(evaluate, [0.0, 0.0, 0.0], LB, UB, [0, 1, 2])
    assert out.verdict == "converged" and out.f < 1e-12
    assert np.allclose(out.x, [1, 2, 3], atol=1e-6)


def test_minimize_respects_bounds_and_judges_the_projected_gradient():
    ub = np.array([0.5, 5.0, 5.0])
    out = L.minimize(evaluate, [0.0, 0.0, 0.0], LB, ub, [0, 1, 2])
    assert out.x[0] <= 0.5 + 1e-12 and out.verdict == "converged"      # at the bound, the gradient points out


def test_minimize_reports_a_spent_budget_and_a_failed_start():
    st = L.LmSettings(max_evals=1)
    assert L.minimize(evaluate, [0, 0, 0], LB, UB, [0, 1, 2], st).verdict == "not_converged"
    assert L.minimize(lambda x: None, [0, 0, 0], LB, UB, [0, 1, 2]).verdict == "failed"


def test_deadline_raises_between_evaluations():
    with pytest.raises(L.OutOfTime):
        L.minimize(evaluate, [0, 0, 0], LB, UB, [0, 1, 2], deadline=0.0)


def req(v, **kw):
    return P.PointRequest(param="a", param_idx=0, x_fixed=v, x_start=(1.0, 2.0, 3.0), anchor_nll=0.0, **kw)


def test_solver_profile_point_matches_the_known_profile():
    s = L.LmSolver(evaluate, NAMES, ["lin"] * 3, LB, UB)
    assert isinstance(s, P.NuisanceSolver)
    rec = s.solve(req(2.0, best_value=1.0, direction=1))
    assert rec.verdict == "converged" and math.isclose(rec.dnll, 0.5, abs_tol=1e-9)
    assert math.isclose(rec.rel, 2.0) and not P.validate_result(req(2.0, best_value=1.0), rec)


def test_solver_flags_a_nuisance_on_its_bound():
    ub = np.array([5.0, 3.0, 5.0])          # x1 = 2 v wants 4 at v = 2; the bound stops it at 3
    s = L.LmSolver(evaluate, NAMES, ["lin"] * 3, LB, ub)
    rec = s.solve(req(2.0))
    assert rec.bound_active == ["b"] and rec.box_limited and rec.dnll > 0.5


def test_solver_pin_box_keeps_the_parameter_free():
    s = L.LmSolver(evaluate, NAMES, ["lin"] * 3, LB, UB)
    rec = s.solve(req(2.0, pin_rel=0.001))
    assert abs(rec.x[0] - 2.0) <= 0.002 + 1e-9 and rec.verdict == "converged"

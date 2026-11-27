"""The profile walk that does not extend from points that did not converge, and the gradient noise floor.

Run from the repository root: python -m pytest tests/engine/test_profile_adaptive.py

From the Elbert_Esguerra_20261121 microglia profile (PYANTIGEN_REQUESTS.md, sections 1 and 3): on a ridge the first hop of 0.5 decades landed at
dnll 2.6 without converging, and every later point was chained from it. The walk now goes back to the last settled point and halves the hop; a
point over the level that did not converge never brackets a crossing (verdict ``unbracketed``); a point under the gradient's measured error floor
counts as converged within noise; a stall that reproduces itself is not retried.
"""
import contextlib
import math

import numpy as np
import pytest

from pyantigen.engine import Profile as PR
from pyantigen.engine import Profile_amici as PA
from pyantigen.engine import Profile_lm as L
from pyantigen.engine import Profile_record as P

NAMES = ["a", "b", "c"]
TRUE_UP = 1.0 + math.sqrt(2 * P.DNLL_95)


def evaluate(x):
    """r = (a - 1, b - a^2, c - 3): the profile of a at v is 0.5 (v - 1)^2 (as in test_profile.py)."""
    r = np.array([x[0] - 1.0, x[1] - x[0] ** 2, x[2] - 3.0])
    J = np.array([[1.0, 0, 0], [-2.0 * x[0], 1.0, 0], [0, 0, 1.0]])
    return 0.5 * float(r @ r), r, J


class Wrapped:
    """An LM solver whose answers are altered by ``alter(request, record)`` (return the record); every request is kept."""
    name, backend, supports_pin_box = "lm", "test", True

    def __init__(self, alter, lb=(-9.0, -99.0, -9.0), ub=(9.0, 99.0, 9.0)):
        self.inner = L.LmSolver(evaluate, NAMES, ["lin"] * 3, lb, ub)
        self.alter, self.requests = alter, []

    def solve(self, req):
        self.requests.append(req)
        return self.alter(req, self.inner.solve(req))

    def with_bounds(self, lb, ub):
        out = Wrapped(self.alter, lb, ub)
        out.requests = self.requests
        return out


def problem(alter, sd=0.5):
    solver = Wrapped(alter)
    return PR.ProfileProblem(names=NAMES, lb=np.array([-9.0, -99.0, -9.0]), ub=np.array([9.0, 99.0, 9.0]), best_x=np.array([1.0, 1.0, 3.0]),
                             anchor_nll=0.0, solver=solver, best_value=lambda k: 1.0, sd_first=np.full(3, sd)), solver


def stall(rec, pg=20.0, why="evaluation budget spent", scale=1.5):
    rec.verdict, rec.pg, rec.reason, rec.dnll = "not_converged", pg, why, rec.dnll * scale
    rec.nll = rec.dnll
    rec.x = [rec.x[0], rec.x[1] + 0.5, rec.x[2]]               # a different vector from the settled fit at that value, so an anchor on it is recognisable
    return rec


# --- the walk -----------------------------------------------------------------------------

def test_a_hop_that_does_not_converge_is_retried_shorter_and_never_extended_from():
    def alter(req, rec):                                   # a hop longer than 0.3 decades from the last settled point stalls far from stationary
        return stall(rec) if req.x_fixed - req.anchor_x[0] > 0.3 else rec
    pr, solver = problem(alter)
    pts, why = PR.walk(pr, 0, +1)
    assert "threshold crossed" in why
    bad = [p for p in pts if p.verdict != "converged"]
    assert bad and all(p.status() == "budget" for p in bad)                    # the tries are kept as evidence
    anchors = {tuple(round(v, 9) for v in q.anchor_x) for q in solver.requests}
    starts = {tuple(round(v, 9) for v in q.x_start) for q in solver.requests}
    assert not any(tuple(round(v, 9) for v in p.x) in anchors | starts for p in bad)         # no hop was anchored on, or started from, one
    first = solver.requests[0]
    assert first.x_fixed - 1.0 == pytest.approx(0.5)                           # first hop: h_max
    second = solver.requests[1]
    assert second.anchor_x[0] == pytest.approx(1.0) and second.x_fixed - 1.0 == pytest.approx(0.25)   # back to the optimum, half the hop
    assert P.side_verdicts(pts, 1.0)[1] == "crossed"


def test_without_adaptive_hop_the_old_walk_is_unchanged():
    def alter(req, rec):
        return stall(rec) if req.x_fixed - req.anchor_x[0] > 0.3 else rec
    pr, solver = problem(alter)
    pts, why = PR.walk(pr, 0, +1, PR.WalkSettings(adaptive_hop=False))
    assert solver.requests[1].anchor_x[0] == pytest.approx(solver.requests[0].x_fixed)     # chained from the unconverged point, as before


def test_a_walk_with_no_converged_point_ends_without_a_crossing_and_says_so():
    store = PR.MemoryStore()
    pr, solver = problem(lambda req, rec: stall(rec, scale=1.0) if req.x_fixed > 1.0 else rec)
    pts, why = PR.walk(pr, 0, +1, PR.WalkSettings(), store)
    assert "no converged point within 5 halving(s)" in why and "not bracketed by converged points" in why
    assert len(solver.requests) == 6                                                       # the hop and five halvings, then it stops
    assert all(r.anchor_x[0] == pytest.approx(1.0) for r in solver.requests)               # always from the optimum
    assert P.side_verdicts(pts, 1.0, noise_pg=None)[1] in ("open", "unbracketed")


def test_an_unconverged_point_over_the_level_is_unbracketed_and_fill_says_so():
    store = PR.MemoryStore()
    pr, solver = problem(lambda req, rec: stall(rec, scale=1.0) if abs(req.x_fixed - 1.0) > 0.45 else rec)
    st = PR.WalkSettings(max_halvings=0)
    # force an over-level unconverged record into the walk's store by hand: the walk itself would have halved it away
    opt = PR._optimum_record(pr, 0)
    over = P.PointRecord(param="a", x_fixed=2.5, nll=3.0, dnll=3.0, verdict="not_converged", pg=12.0, reason="evaluation budget spent", x=[2.5, 6.0, 3.0], param_idx=0)
    store.append("profile", opt)
    store.append("profile", over)
    assert P.side_verdicts([opt, over], 1.0)[1] == "unbracketed"
    assert PR.fill(pr, 0, +1, st, store) is None
    assert store.done("fill") == "crossing not bracketed by converged points"
    assert not solver.requests                                                              # no insert was started from an unconverged point


def test_a_point_within_the_noise_floor_settles_the_walk_and_a_smaller_floor_does_not():
    def alter(req, rec):
        return stall(rec, pg=0.03, why="no model decrease available", scale=1.0)
    pr, _ = problem(alter)
    pts, why = PR.walk(pr, 0, +1, PR.WalkSettings(noise_floor=0.05))
    assert "threshold crossed" in why and all(p.status(0.05) in ("converged", "within_noise") for p in pts)
    pr, _ = problem(alter)
    pts, why = PR.walk(pr, 0, +1, PR.WalkSettings(noise_floor=0.01))
    assert "no converged point" in why and pts[-1].status(0.01) == "capped"


def test_fill_brackets_on_settled_points_only():
    st, store = PR.WalkSettings(), PR.MemoryStore()
    pr, _ = problem(lambda req, rec: rec)
    PR.walk(pr, 0, +1, st, store)
    # a stalled record sitting between the real bracket ends must not become one
    junk = P.PointRecord(param="a", x_fixed=TRUE_UP + 0.001, nll=1.0, dnll=1.0, verdict="not_converged", pg=9.0, reason="evaluation budget spent",
                         x=[TRUE_UP, 1.0, 3.0], param_idx=0)
    store.append("profile", junk)
    cross = PR.fill(pr, 0, +1, st, store)
    assert cross == pytest.approx(TRUE_UP, rel=1e-3)


def test_a_stall_that_reproduces_itself_is_not_retried_by_relax():
    calls = []

    class Same:
        name, backend, supports_pin_box = "same", "test", True

        def solve(self, req):
            calls.append(req.source)
            return P.PointRecord(param=req.param, x_fixed=req.x_fixed, nll=req.anchor_nll + 2.0, dnll=2.0, verdict="not_converged", pg=0.04,
                                 reason="no model decrease available", x=list(req.x_start), param_idx=req.param_idx, source=req.source)

    pr, _ = problem(lambda req, rec: rec)
    pr.solver.relaxed = lambda decades: Same()
    store = PR.MemoryStore()
    boxed = P.PointRecord(param="a", x_fixed=1.5, nll=2.0, dnll=2.0, verdict="converged", x=[1.5, 2.2, 3.0], param_idx=0, bound_active=["b"], source="walk:1")
    store.append("profile", PR._optimum_record(pr, 0))
    store.append("profile", boxed)
    PR.relax(pr, 0, +1, PR.WalkSettings(), store)
    assert calls == ["relax:1"]                                                            # one retry, the same dnll: settled
    assert store.read("relax")[0].reproduced


# --- records ------------------------------------------------------------------------------

def rec(x, d, verdict="converged", pg=None, reason=None, bound=()):
    return P.PointRecord(param="k", x_fixed=x, dnll=d, nll=d, verdict=verdict, pg=pg, reason=reason, bound_active=list(bound))


def test_status_separates_converged_noise_capped_budget_and_failed():
    assert rec(1, 1.0).status() == "converged"
    assert rec(1, 1.0, reason="projected gradient 0.03 within the gradient noise floor 0.05").status() == "within_noise"
    assert rec(1, 1.0, "not_converged", pg=0.03, reason="no model decrease available").status(0.05) == "within_noise"
    assert rec(1, 1.0, "not_converged", pg=0.03, reason="no model decrease available").status(0.01) == "capped"
    assert rec(1, 1.0, "not_converged", pg=12.0, reason="evaluation budget spent").status(0.05) == "budget"
    assert rec(1, 1.0, "not_converged", pg=12.0, reason="lambda above 1e12").status(0.05) == "capped"
    assert P.PointRecord(param="k", x_fixed=1.0).status() == "failed"
    assert rec(1, 1.0, "not_converged", pg=0.04).settled(0.05) and not rec(1, 1.0, "not_converged", pg=0.04).settled(0.01)


def test_reproduces_is_three_digits():
    assert P.reproduces(0.173, 0.1734) and not P.reproduces(0.173, 0.19)
    assert not P.reproduces(None, 0.1) and not P.reproduces(float("nan"), 0.1)


def test_side_verdicts_unbracketed_and_settled_crossings():
    pts = [rec(0.5, 0.5), rec(1.0, 1.5), rec(1.5, 3.0, "not_converged", pg=9.0, reason="evaluation budget spent")]
    assert P.side_verdicts(pts, 0.0) == ("open", "unbracketed")
    pts.append(rec(2.0, 3.5))                                                              # a settled point past it brackets the crossing
    assert P.side_verdicts(pts, 0.0)[1] == "crossed"
    lo_all, hi_all = P.crossings(pts, 0.0)
    _, hi_settled = P.crossings(pts, 0.0, settled_only=True)
    assert hi_all < hi_settled                                                             # across the upper bound the crossing was too narrow
    pts[2].bound_active = ["Vm"]
    assert P.side_verdicts(pts, 0.0)[1] == "crossed"                                       # the bound is on a point that is not an end of the bracket
    pts[3].bound_active = ["Vm"]
    assert P.side_verdicts(pts, 0.0)[1] == "box-limited"


# --- the LM and the noise floor --------------------------------------------------------------

def noisy(x):
    """r = 0.03 whatever x is, but J claims a sensitivity of 1: the gradient J'r = 0.03 is all error."""
    return 0.5 * 0.03 ** 2, np.array([0.03]), np.array([[1.0]])


def test_lm_calls_a_point_under_the_noise_floor_converged_and_says_why():
    bare = L.minimize(noisy, [0.0], [-5.0], [5.0], [0], L.LmSettings())
    assert bare.verdict == "not_converged" and bare.pg == pytest.approx(0.03)
    held = L.minimize(noisy, [0.0], [-5.0], [5.0], [0], L.LmSettings(noise_pg=0.05))
    assert held.verdict == "converged" and "noise floor" in held.reason and held.evals == 1
    under = L.minimize(noisy, [0.0], [-5.0], [5.0], [0], L.LmSettings(noise_pg=0.01))
    assert under.verdict == "not_converged"                                                # a floor below the gradient does not excuse it


def test_polish_with_tighter_sensitivities_converges_a_point_the_standard_ones_could_not():
    def standard(x):                      # the second residual's sensitivity is wrong: the standard J'r keeps pointing at 0.03 that is not there
        r = np.array([x[1] - 2.0, 0.03])
        return 0.5 * float(r @ r), r, np.array([[0.0, 1.0], [0.0, 1.0]])

    def tight(x):
        r = np.array([x[1] - 2.0, 0.03])
        return 0.5 * float(r @ r), r, np.array([[0.0, 1.0], [0.0, 0.0]])
    req = P.PointRequest(param="a", param_idx=0, x_fixed=1.5, x_start=(1.5, 2.0), anchor_nll=0.0, max_evals=40)
    bare = L.LmSolver(standard, ["a", "b"], ["lin", "lin"], [-9.0, -9.0], [9.0, 9.0]).solve(req)
    assert bare.verdict == "not_converged" and bare.pg == pytest.approx(0.03)
    pol = L.LmSolver(standard, ["a", "b"], ["lin", "lin"], [-9.0, -9.0], [9.0, 9.0], settings=L.LmSettings(polish_evals=10),
                     evaluate_tight=tight).solve(req)
    assert pol.verdict == "converged" and "polished" in pol.reason and pol.evals > bare.evals - 1
    off = L.LmSolver(standard, ["a", "b"], ["lin", "lin"], [-9.0, -9.0], [9.0, 9.0], settings=L.LmSettings(polish_evals=0),
                     evaluate_tight=tight).solve(req)
    assert off.verdict == "not_converged"


def test_measure_noise_floor_is_the_difference_of_the_two_gradients():
    def standard(x):
        return 0.0, np.array([1.0, 2.0]), np.array([[1.0, 0.0], [0.0, 1.0]]) + np.array([[0.0, 0.0], [0.01, 0.0]])

    def tight(x):
        return 0.0, np.array([1.0, 2.0]), np.eye(2)
    m = PA.measure_noise_floor(standard, tight, [0.0, 0.0])
    assert m["per_param"] == pytest.approx([0.02, 0.0])
    assert PA.side_noise_floor(m["per_param"], 0) == 0.0 and PA.side_noise_floor(m["per_param"], 1) == pytest.approx(0.02)
    assert PA.measure_noise_floor(lambda x: None, tight, [0.0]) is None


def test_weak_directions_finds_the_parameters_on_a_ridge():
    rng = np.random.default_rng(0)
    base = rng.normal(size=(30, 1))
    J = np.hstack([base, 2.0 * base + 1e-8 * rng.normal(size=(30, 1)), rng.normal(size=(30, 1))])     # a and b are a product's two factors
    assert PA.weak_directions(J) == {0, 1}
    assert PA.weak_directions(rng.normal(size=(30, 3))) == set()


def test_settings_from_the_spec_and_the_environment():
    st = PA.settings_from_spec({"first_hop_max": 0.15, "plateau_window": 20, "no_widen": ["k"]})
    assert st.first_hop_max == 0.15 and st.plateau_window == 20 and st.no_widen == ("k",)
    with pytest.raises(ValueError, match="unknown setting"):
        PA.settings_from_spec({"first_hop": 0.1})
    mode, st = PA.settings_from_env({PA.NOISE_FLOOR_ENV: "0.04", PA.POLISH_ENV: "8", "PYANTIGEN_PROFILE_FIRST_HOP": "0.2"}, declared={"first_hop_max": 0.15})
    assert st.noise_floor == 0.04 and st.polish_evals == 8 and st.first_hop_max == 0.2          # the environment wins over the spec
    assert PA.settings_from_env({PA.NOISE_FLOOR_ENV: "off"})[1].noise_floor_mode == "off"
    assert PA.settings_from_env({})[1].noise_floor_mode == "measure"


def test_side_settings_give_a_weak_side_a_short_first_hop_and_the_plateau_stop_unless_set():
    st = PA.side_settings(PR.WalkSettings(), 0, nf_per=[0.5, 0.03, 0.04], weak={0})
    assert st.first_hop_max == 0.15 and st.plateau_window == 20 and st.noise_floor == 0.04       # the largest among the nuisance parameters
    st = PA.side_settings(PR.WalkSettings(first_hop_max=0.4, plateau_window=7), 0, weak={0})
    assert st.first_hop_max == 0.4 and st.plateau_window == 7 and st.noise_floor is None
    st = PA.side_settings(PR.WalkSettings(), 1, nf_per=[0.5, 0.03, 0.04], weak={0})
    assert st.first_hop_max is None and st.plateau_window == 0


def test_point_counts_and_their_line():
    recs = [P.PointRecord(param="k", x_fixed=0.0, dnll=0.0, nll=0.0, verdict="converged", source="optimum"),
            rec(1, 0.5), rec(2, 1.0, "not_converged", pg=0.03, reason="no model decrease available"),
            rec(3, 1.5, "not_converged", pg=9.0, reason="lambda above 1e12"), rec(4, 2.0, "not_converged", pg=9.0, reason="evaluation budget spent"),
            P.PointRecord(param="k", x_fixed=5.0)]
    c = PA.point_counts(recs, 0.05)
    assert c == {"converged": 1, "within_noise": 1, "capped": 1, "budget": 1, "failed": 1}
    line = PA.format_point_counts(c, 0.05)
    assert line.startswith("4 points: 1 converged, 1 within the noise floor (<= 0.05 nats), 1 capped far from stationarity, 1 out of budget")


# --- the stage -------------------------------------------------------------------------------

class ToleranceProvider:
    """r = L p - y whose Jacobian carries a small error unless the tolerances are scaled down, as an integrator's sensitivities do."""

    def __init__(self):
        rng = np.random.default_rng(3)
        self.L = rng.normal(size=(12, 3))
        self.y = self.L @ np.array([1.0, -0.5, 2.0]) + 0.05 * rng.normal(size=12)
        self.err = 1e-3 * rng.normal(size=self.L.shape)
        self.scale = None

    @contextlib.contextmanager
    def tolerances_scaled(self, factor):
        self.scale = factor
        try:
            yield
        finally:
            self.scale = None

    def linear_residual_jacobian(self, p):
        J = self.L.copy() if self.scale is not None else self.L + self.err
        return self.L @ np.asarray(p, float) - self.y, J, 0.0


def test_the_stage_measures_the_noise_floor_once_and_every_side_uses_it(tmp_path):
    p = ToleranceProvider()
    xs, *_ = np.linalg.lstsq(p.L, p.y, rcond=None)
    lines = []
    PA.run(p, NAMES, [(-20.0, 20.0)] * 3, ["lin"] * 3, xs, str(tmp_path), side=(0, 1), log=lines.append)
    saved = PA.read_noise_floor(str(tmp_path))
    assert saved and len(saved["per_param"]) == 3 and max(saved["per_param"]) > 0
    assert any("gradient noise floor" in m for m in lines)
    assert any(m.startswith("[profile] 0_up: ") and "points:" in m for m in lines)         # the per-side count line
    again = []
    PA.run(p, NAMES, [(-20.0, 20.0)] * 3, ["lin"] * 3, xs, str(tmp_path), side=(1, 1), log=again.append)
    assert PA.read_noise_floor(str(tmp_path)) == saved                                      # measured by the first side, read by the next


def test_the_stage_without_a_tolerance_switch_measures_nothing(tmp_path):
    class Plain:
        def linear_residual_jacobian(self, p):
            return np.array([p[0] - 1.0]), np.array([[1.0]]), 0.0
    lines = []
    PA.run(Plain(), ["a"], [(-5.0, 5.0)], ["lin"], np.array([1.0]), str(tmp_path), side=(0, 1), log=lines.append)
    assert PA.read_noise_floor(str(tmp_path)) is None and not any("noise floor" in m and "gradient" in m for m in lines)


def test_side_report_names_an_unbracketed_side(tmp_path):
    opt = P.PointRecord(param="a", x_fixed=1.0, nll=0.0, dnll=0.0, verdict="converged", source="optimum", x=[1.0], param_idx=0)
    over = P.PointRecord(param="a", x_fixed=3.0, nll=3.0, dnll=3.0, verdict="not_converged", pg=9.0, reason="evaluation budget spent", x=[3.0], param_idx=0)
    rep = PA.side_report({"a": [opt, over]}, ["a"], [(-9.0, 9.0)])
    assert rep["a"]["above"]["verdict"] == "unbracketed" and rep["a"]["above"]["counts"]["budget"] == 1
    assert "a above: NOT BRACKETED" in "\n".join(PA.format_side_report(rep))

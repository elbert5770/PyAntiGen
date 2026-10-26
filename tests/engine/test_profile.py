"""The stepping protocol on a residual problem whose profile is known: dnll(v) = 0.5 (v - 1)^2 for the first parameter."""
import math

import numpy as np

from pyantigen.engine import Profile as PR
from pyantigen.engine import Profile_lm as L
from pyantigen.engine import Profile_record as P

NAMES = ["a", "b", "c"]


def evaluate(x):
    """r = (a - 1, b - a^2, c - 3); b follows a on a curve, c sits at 3. Profile of a at v: 0.5 (v - 1)^2."""
    r = np.array([x[0] - 1.0, x[1] - x[0] ** 2, x[2] - 3.0])
    J = np.array([[1.0, 0, 0], [-2.0 * x[0], 1.0, 0], [0, 0, 1.0]])
    return 0.5 * float(r @ r), r, J


def problem(lb=(-9.0, -99.0, -9.0), ub=(9.0, 99.0, 9.0), sd=0.5):
    solver = L.LmSolver(evaluate, NAMES, ["lin"] * 3, lb, ub)
    return PR.ProfileProblem(names=NAMES, lb=np.array(lb), ub=np.array(ub), best_x=np.array([1.0, 1.0, 3.0]),
                             anchor_nll=0.0, solver=solver, best_value=lambda k: 1.0, sd_first=np.full(3, sd))


TRUE_UP = 1.0 + math.sqrt(2 * P.DNLL_95)


def test_walk_follows_the_profile_and_stops_past_the_threshold():
    pts, why = PR.walk(problem(), 0, +1)
    assert "threshold crossed" in why and pts[-1].dnll > 2.5 and pts[0].reason == "optimum"
    assert all(p.verdict == "converged" and not p.box_limited for p in pts)
    for p in pts[1:]:
        assert math.isclose(p.dnll, 0.5 * (p.x_fixed - 1.0) ** 2, abs_tol=2e-3)
    assert 4 <= len(pts) <= 12                      # curvature-driven: a few points, not a fixed grid


def test_fill_places_a_point_at_the_crossing():
    st, store, pr = PR.WalkSettings(), PR.MemoryStore(), problem()
    PR.walk(pr, 0, +1, st, store)
    cross = PR.fill(pr, 0, +1, st, store)
    assert math.isclose(cross, TRUE_UP, rel_tol=1e-3)
    inserted = store.read("fill")
    assert inserted and min(abs(p.dnll - st.level) for p in inserted) <= st.fill_tol
    below, above = P.crossings(store.read("profile") + inserted, 1.0)
    assert math.isclose(above, TRUE_UP, rel_tol=1e-3) and below is None
    assert P.side_verdicts(store.read("profile") + inserted, 1.0) == ("open", "crossed")


def test_walk_resumes_from_its_checkpoint(tmp_path):
    store = PR.JsonlStore(str(tmp_path), "0_up")
    pr = problem()
    pts, why = PR.walk(pr, 0, +1, PR.WalkSettings(), store, deadline=0.0)       # out of time at once
    assert why is None and len(pts) == 1
    pts, why = PR.walk(pr, 0, +1, PR.WalkSettings(), PR.JsonlStore(str(tmp_path), "0_up"))
    assert why and len(pts) > 3
    n = len(pts)
    assert len(PR.JsonlStore(str(tmp_path), "0_up").read("profile")) == n
    assert PR.walk(pr, 0, +1, PR.WalkSettings(), store)[1] == why               # finished: nothing more is run


def test_extension_limit_and_failure_end_the_walk():
    pr = problem(lb=(-9.0, -99.0, -9.0), ub=(1.3, 99.0, 9.0))                    # box ends 0.3 above the optimum
    st = PR.WalkSettings(extend=0.2, df_stop=99.0)
    pts, why = PR.walk(pr, 0, +1, st)
    assert "extension limit" in why and math.isclose(pts[-1].x_fixed, 1.5)
    pr.solver.evaluate = lambda x: None
    pts, why = PR.walk(pr, 0, +1)
    assert why == "point could not be evaluated"


def test_box_limited_crossing_is_not_called_crossed():
    pr = problem(ub=(9.0, 3.0, 9.0))                 # b = a^2 reaches 3 at a = 1.73; beyond it b sits on its bound
    pts, _ = PR.walk(pr, 0, +1)
    assert any(p.box_limited for p in pts) and pts[-1].dnll > 0.5 * (pts[-1].x_fixed - 1) ** 2
    assert P.side_verdicts(pts, 1.0)[1] == "box-limited"


def test_recheck_keeps_the_lowest_and_is_resumable(tmp_path):
    store = PR.JsonlStore(str(tmp_path), "0_up")
    pr = problem()
    PR.walk(pr, 0, +1, PR.WalkSettings(), store)
    out = PR.recheck(pr, 0, +1, PR.WalkSettings(), store)
    n = len(store.read("profile")) - 1
    assert len(out) == 3 * n - 1 and store.done("recheck")
    assert PR.recheck(pr, 0, +1, PR.WalkSettings(), store) == out


def test_relax_resolves_box_limited_points_with_wider_bounds(tmp_path):
    pr = problem(ub=(9.0, 3.0, 9.0))                 # b = a^2 hits its bound at a = 1.73
    store = PR.JsonlStore(str(tmp_path), "0_up")
    PR.walk(pr, 0, +1, PR.WalkSettings(), store)
    flagged = [p for p in store.read("profile") if p.box_limited]
    assert flagged
    got = PR.relax(pr, 0, +1, PR.WalkSettings(), store)
    assert len(got) == len(flagged) and store.done("relax")
    assert all(not r.box_limited and r.dnll < p.dnll for r, p in zip(got, flagged))
    assert PR.relax(pr, 0, +1, PR.WalkSettings(), store) == got                  # finished: nothing re-run
    PR.fill(pr, 0, +1, PR.WalkSettings(), store)
    allrec = store.read("profile") + store.read("relax") + store.read("fill")
    assert P.side_verdicts(allrec, 1.0)[1] == "crossed"


def test_relax_off_and_without_box_limits_do_nothing():
    pr = problem(ub=(9.0, 3.0, 9.0))
    store = PR.MemoryStore()
    PR.walk(pr, 0, +1, PR.WalkSettings(), store)
    assert PR.relax(pr, 0, +1, PR.WalkSettings(relax=0.0), store) == [] and store.done("relax") is None
    clean = PR.MemoryStore()
    PR.walk(problem(), 0, +1, PR.WalkSettings(), clean)
    assert PR.relax(problem(), 0, +1, PR.WalkSettings(), clean) == [] and clean.done("relax")

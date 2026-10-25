"""The common profile-point record, its readers for the stored formats, and the solver contract."""
import math

import pytest

from pyantigen.engine import Profile_record as P

# Shaped like real stored records (RoadRunner checkpoint line; AMICI walk line; AMICI pinned start file).
RR = {"param_idx": 1, "param_name": "k", "x_fixed": -2.0, "x_fixed_linear": 0.01, "x_start": [0.1, 0.2],
      "direction": -1, "pass_label": "pass1-grid", "nll": -100.0, "status": "ok", "n_evals": 90, "wall_s": 12.5,
      "converged": True, "interrupted": False, "stop_reason": "tolerance", "nuisance_x": [0.1, 0.2],
      "nfev_total": 90, "dnll": 1.5}
WALK = {"n": 2, "val": 1.4, "f": -3006.0, "df": 1.4, "x": [0.0, 1.4, 2.0], "verdict": "converged", "pg": 0.004,
        "evals": 4, "seconds": 300.0, "reason": "newton", "floors_sha256": "abc"}
START = {"start_id": 7, "names": ["a", "b", "c"], "scales": ["lin", "log10", "log10"], "x": [1.0, -1.0, 0.5],
         "fval": 50.0, "verdict": "not_converged", "pg_inf": 0.3, "evaluations_total": 120, "reason": "budget",
         "floors_sha256": "abc"}


def test_roadrunner_record_maps_to_the_common_shape():
    r = P.from_roadrunner(RR, ["lin", "log10", "log10"], best_value=0.1)
    assert (r.param, r.x_fixed, r.nll, r.dnll, r.verdict) == ("k", -2.0, -100.0, 1.5, "converged")
    assert r.x == [0.1, -2.0, 0.2] and r.value == 0.01 and math.isclose(r.rel, 0.1)
    assert (r.evals, r.direction, r.solver, r.backend) == (90, -1, "nelder-mead", "roadrunner")
    assert not r.check()


def test_roadrunner_verdicts():
    assert P.from_roadrunner({**RR, "converged": False}).verdict == "not_converged"
    assert P.from_roadrunner({**RR, "interrupted": True}).verdict == "not_converged"
    failed = P.from_roadrunner({**RR, "status": "worker-not-initialized", "nll": 1e10})
    assert failed.verdict == "failed"
    assert P.from_roadrunner({**RR, "nll": None, "dnll": float("nan")}).verdict == "failed"


def test_amici_walk_and_start_records():
    w = P.from_amici_walk(WALK, "b", 1, "log10", best_value=10.0, direction=1)
    assert (w.nll, w.dnll, w.verdict, w.pg, w.solver, w.backend) == (-3006.0, 1.4, "converged", 0.004, "lm", "amici")
    assert math.isclose(w.value, 10 ** 1.4) and w.source == "walk"
    assert P.from_amici_walk({**WALK, "n": 0}, "b").source == "optimum"
    s = P.from_amici_start(START, "b", 1, "log10", f_opt=48.0, pin_value=0.1, best_value=0.2, pin_id=7)
    assert (s.x_fixed, s.dnll, s.verdict, s.id, s.pg) == (-1.0, 2.0, "not_converged", 7, 0.3)
    assert math.isclose(s.rel, 0.5)
    with pytest.raises(ValueError):
        P.from_amici_start(START, "zzz", 1, "log10", 48.0)


def test_lower_flag():
    assert P.from_amici_walk({**WALK, "df": -0.2}, "b").lower
    assert not P.from_amici_walk({**WALK, "df": -0.01}, "b").lower
    assert P.from_amici_walk({**WALK, "df": 0.3, "LOWER": True}, "b").lower


def test_dict_roundtrip_and_check():
    r = P.from_roadrunner(RR)
    assert P.PointRecord.from_dict({**r.to_dict(), "unknown": 1}) == r
    bad = P.PointRecord("k", 0.0, verdict="converged")
    assert any("finite dnll" in p for p in bad.check())
    assert any("verdict" in p for p in P.PointRecord("k", 0.0, dnll=0.0, verdict="maybe").check())


def _pts(pairs, best=0.0):
    return [P.PointRecord("k", x, nll=d, dnll=d, verdict="converged") for x, d in pairs]


def test_curve_keeps_the_lowest_of_coincident_points_and_skips_failures():
    recs = _pts([(1.0, 3.0), (1.0, 2.5), (2.0, 5.0)]) + [P.PointRecord("k", 3.0)]
    c = P.curve(recs)
    assert [(r.x_fixed, r.dnll) for r in c] == [(1.0, 2.5), (2.0, 5.0)]


def test_crossings_interpolate_between_the_last_under_and_the_first_over():
    level = P.DNLL_95
    recs = _pts([(1.0, 0.5), (2.0, 1.0), (3.0, 3.0), (-1.0, 0.5), (-2.0, 1.0), (-3.0, 4.0)])
    lo, hi = P.crossings(recs, 0.0, level)
    assert math.isclose(hi, 2.0 + (level - 1.0) / 2.0)
    assert math.isclose(lo, -(2.0 + (level - 1.0) / 3.0))
    assert P.crossings(_pts([(1.0, 0.5), (2.0, 1.0)]), 0.0)[1] is None         # open: never reached the level
    # a first step that overshoots interpolates from the optimum itself
    assert math.isclose(P.crossings(_pts([(4.0, 3.8414)]), 0.0)[1], 4.0 * level / 3.8414)


class _Fake:
    name, backend, supports_pin_box = "fake", "amici", True

    def solve(self, req):
        return P.PointRecord(req.param, req.x_fixed, nll=req.anchor_nll + 1.0, dnll=1.0, verdict="converged")


def test_solver_protocol_and_contract_check():
    req = P.PointRequest("k", 1, -2.0, (0.1, 0.0, 0.2), anchor_nll=-101.0)
    assert isinstance(_Fake(), P.NuisanceSolver)
    assert P.validate_result(req, _Fake().solve(req)) == []
    bad = P.PointRecord("k", -2.5, nll=-100.0, dnll=0.3, verdict="converged")
    problems = P.validate_result(req, bad)
    assert any("differs" in p for p in problems) and any("dnll != nll" in p for p in problems)
    assert any("asked" in p for p in P.validate_result(req, P.PointRecord("other", -2.0, nll=-100.0, dnll=1.0,
                                                                      verdict="converged")))


def test_roadrunner_solver_builds_the_job_and_reads_the_result():
    seen = {}

    def run_job(job):
        seen.update(job)
        return {**{k: v for k, v in RR.items() if k != "dnll"}, "nll": -99.0, "x_fixed": job["x_fixed"]}

    s = P.RoadRunnerSolver(run_job, ["lin", "log10", "log10"], bounds=[(0, 1), (-3, 0), (-4, 4)])
    assert isinstance(s, P.NuisanceSolver) and not s.supports_pin_box
    req = P.PointRequest("k", 1, -2.0, (0.1, -1.0, 0.2), anchor_nll=-101.0, direction=-1, id=5, max_evals=300,
                         source="pins", deadline=123.0)
    rec = s.solve(req)
    assert seen["x_start"] == [0.1, 0.2] and seen["nuisance_bounds"] == [(0, 1), (-4, 4)]
    assert seen["param_idx"] == 1 and seen["x_fixed_linear"] == 0.01 and seen["deadline"] == 123.0
    assert seen["optimizer_kwargs"]["options"]["maxfev"] == 300
    assert rec.dnll == 2.0 and rec.id == 5 and rec.source == "pins" and P.validate_result(req, rec) == []
    with pytest.raises(ValueError, match="pin_rel"):
        s.solve(P.PointRequest("k", 1, -2.0, (0.1, -1.0, 0.2), -101.0, pin_rel=1e-3))

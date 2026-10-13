"""The adaptive, streamed first pass of the profile.

Run from the repository root:
    python tests/engine/test_profile_stream.py

Two things are under test. The planner (``Profile_plan.plan_next_points``) is a
pure function of the stored records, so it is checked against hand-built
records. The stream (``run_parallel_profile(adaptive_pass1=True)`` through a
pool that obeys the ``refill`` contract of ``ParallelEvaluator.profile_batch``)
is checked on an analytic profile whose slice is far steeper than its profile,
which is the shape that made a slice-placed grid useless: it must keep exactly
``n_workers`` points running while there is work, never more, and end with every
side bracketed.
"""
import contextlib
import heapq
import io
import os
import sys

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from pyantigen.engine.Optimize import run_parallel_profile                   # noqa: E402
from pyantigen.engine.Profile_plan import plan_next_points                   # noqa: E402

failures = []
THR = 1.9207


def check(name, cond, detail=""):
    if cond:
        print(f"  ok    {name}")
    else:
        print(f"  FAIL  {name} {detail}")
        failures.append(name)


# ---------------------------------------------------------------- planner ---

def rec(x, dnll, converged=True, nuis=None, warm=True, **extra):
    r = {"x_fixed": x, "dnll": dnll, "converged": converged,
         "interrupted": False, "nuisance_x": nuis or [0.0, 0.0],
         "warm_seeded": warm}
    r.update(extra)
    return r


def plan(completed=None, in_flight=None, n_free=8, names=("a",), res_x=(0.0,),
         scales=("log10",), se=(0.3,), screen=None, bounds=((-6.0, 6.0),),
         attempted=None, **kw):
    comp = {n: {round(r["x_fixed"], 12): r for r in (completed or {}).get(n, [])}
            for n in names}
    args = dict(param_names=list(names), completed=comp,
                in_flight=in_flight or [], res_x=np.array(res_x, float),
                bounds=[list(b) for b in bounds], scales=list(scales),
                wald_se=None if se is None else np.array(se, float),
                n_grid=3, range_factor=2.0, open_decades=None, screen=screen,
                threshold=THR, anchor=0.0, max_extend=8, extend_growth=2.0,
                bracket_rtol=0.05, max_step_decades=0.7, attempted=attempted)
    args.update(kw)
    return plan_next_points(n_free, **args)


print("a first probe sits at 1.96 SE, one per side, before anything else")
out = plan()
firsts = [c for c in out if c["kind"] == "first probe"]
check("one first probe per side", len(firsts) == 2, [c["kind"] for c in out])
check("at 1.96 SE", {round(c["x"], 3) for c in firsts} == {round(1.96 * 0.3, 3),
                                                            round(-1.96 * 0.3, 3)})
check("first probes outrank the ladder",
      [c["tier"] for c in out][:2] == [0, 0], [c["tier"] for c in out])

print("the slice crossing is a floor under the Wald guess")
scr = {"parameters": {"a": {"upper": {"crossing": {"x": 2.0}},
                            "lower": {"crossing": {"x": -0.1}}}}}
out = plan(screen=scr, n_free=2)
xs = sorted(c["x"] for c in out)
check("a slice crossing further out than Wald wins", abs(xs[1] - 2.0) < 1e-9, xs)
check("one nearer than Wald does not pull the probe in",
      abs(xs[0] + 1.96 * 0.3) < 1e-9, xs)

print("a huge Wald SE is capped by open_decades")
out = plan(se=(5.0,), open_decades=1.0, n_free=2)
check("both first probes within 1 decade",
      all(abs(c["x"]) <= 1.0 + 1e-9 for c in out), [c["x"] for c in out])

print("never more than n_free, never on top of a running point")
out = plan(n_free=3)
check("n_free respected", len(out) == 3)
fl = [{"param_idx": 0, "x_fixed": 1.96 * 0.3}]
out = plan(in_flight=fl, n_free=8)
check("no first probe is planned where one is running",
      not any(abs(c["x"] - 1.96 * 0.3) < 1e-9 for c in out))

print("first_probe='slice_ladder': a ladder from the slice crossing to the cap")
scr2 = {"parameters": {"a": {"upper": {"crossing": {"x": 0.03}},
                             "lower": {"crossing": {"x": -0.03}}}}}
out = plan(screen=scr2, first_probe="slice_ladder", open_decades=1.0, n_free=8)
up = sorted(c["x"] for c in out if c["sign"] > 0)
check("three rungs on the upper side", len(up) == 3, up)
check("log-spaced from the slice crossing to the cap",
      np.allclose(up, [0.03, np.sqrt(0.03), 1.0]), up)
lead = [c for c in out if c["tier"] == 0]
check("the middle rung leads, one per side",
      sorted(round(abs(c["x"]), 4) for c in lead) == [round(np.sqrt(0.03), 4)] * 2,
      [(c["x"], c["tier"]) for c in out])
check("the rest are spare", all(c["tier"] == 3 for c in out if c not in lead))
out2 = plan(screen=scr2, first_probe="slice_ladder", open_decades=1.0, n_free=2)
check("with two free slots each side gets its middle rung",
      sorted(c["x"] for c in out2) == sorted([-np.sqrt(0.03), np.sqrt(0.03)]),
      [c["x"] for c in out2])
a = plan(screen=scr2, first_probe="slice_ladder", open_decades=1.0, se=(0.01,))
b = plan(screen=scr2, first_probe="slice_ladder", open_decades=1.0, se=(5.0,))
c0 = plan(screen=scr2, first_probe="slice_ladder", open_decades=1.0, se=None)
check("the Wald SE is not read at all",
      [x["x"] for x in a] == [x["x"] for x in b] == [x["x"] for x in c0])
check("with no cap the ladder runs to 100x the slice distance",
      abs(max(x["x"] for x in plan(screen=scr2, first_probe="slice_ladder",
                                   n_free=8) if x["sign"] > 0) - 3.0) < 1e-9)

print("first_probe='slice': one probe, at the slice crossing")
out = plan(screen=scr2, first_probe="slice", n_free=8)
check("one probe per side at the crossing",
      sorted(c["x"] for c in out) == [-0.03, 0.03], [c["x"] for c in out])

print("a side the screen did not cross falls back to Wald")
scr3 = {"parameters": {"a": {"upper": {"crossing": {"x": 0.03}},
                             "lower": {"crossing": None}}}}
out = plan(screen=scr3, first_probe="slice_ladder", open_decades=1.0, n_free=8)
lo = [c for c in out if c["sign"] < 0]
check("lower side gets the Wald first probe",
      len(lo) >= 1 and lo[0]["kind"] == "first probe"
      and abs(lo[0]["x"] + 1.96 * 0.3) < 1e-9, [(c["kind"], c["x"]) for c in lo])
check("upper side still uses the ladder",
      any(c["kind"] == "slice ladder" for c in out if c["sign"] > 0))

print("an unknown first_probe is refused")
try:
    plan(first_probe="wald")
    check("ValueError raised", False, "no exception")
except ValueError:
    check("ValueError raised", True)

print("below the threshold everywhere: step out, capped per step")
pts = {"a": [rec(0.2, 0.05), rec(0.4, 0.2)]}
out = plan(completed=pts, n_free=4)
up = [c for c in out if c["sign"] > 0]
check("the upper side extends", up and up[0]["kind"] == "extend", up)
check("by no more than max_step_decades",
      up[0]["x"] - 0.4 <= 0.7 + 1e-9, up[0]["x"])
check("warm-started from its outer point", up[0]["seed"]["x_fixed"] == 0.4)
out = plan(completed=pts, n_free=2)
check("with no spare worker the side gets one rung",
      len([c for c in out if c["sign"] > 0 and c["tier"] == 1]) == 1)

print("an open side sends several rungs out together when workers are idle")
many = sorted([c for c in plan(completed=pts, n_free=12) if c["sign"] > 0],
              key=lambda c: c["x"])
xs = [0.4] + [c["x"] for c in many]
check("the next rung and three beyond it", len(many) == 4, xs)
check("each rung is further out than the last, none more than max_step beyond",
      all(b > a and b - a <= 0.7 + 1e-9 for a, b in zip(xs, xs[1:])), xs)
check("the nearest is the primary and the rest rank below every primary",
      many[0]["tier"] == 1 and all(c["tier"] >= 4 for c in many[1:]))
check("all seeded from the outer measured point",
      all(c["seed"]["x_fixed"] == 0.4 for c in many))
fl = [{"param_idx": 0, "x_fixed": many[0]["x"]},
      {"param_idx": 0, "x_fixed": many[1]["x"]}]
more = sorted([c for c in plan(completed=pts, in_flight=fl, n_free=12)
               if c["sign"] > 0], key=lambda c: c["x"])
check("rungs already running count toward the four, and the ladder goes on "
      "beyond the farthest of them",
      len(more) == 2 and more[0]["x"] > many[1]["x"], [c["x"] for c in more])
fl = [{"param_idx": 0, "x_fixed": c["x"]} for c in many]
check("a full ladder in flight plans nothing more on that side",
      not [c for c in plan(completed=pts, in_flight=fl, n_free=12)
           if c["sign"] > 0])
near_bound = plan(completed={"a": [rec(0.2, 0.05), rec(5.5, 0.2)]}, n_free=12,
                  bounds=((-6.0, 6.0),))
check("the ladder stops at the parameter's bound",
      all(c["x"] <= 6.0 + 1e-9 for c in near_bound if c["sign"] > 0)
      and len([c for c in near_bound if c["sign"] > 0]) <= 1,
      [c["x"] for c in near_bound if c["sign"] > 0])

print("crossed only on an unconverged point: narrow the bracket, warm")
pts = {"a": [rec(0.3, 0.8), rec(0.9, 4.0, converged=False)]}
out = plan(completed=pts, n_free=4)
nr = [c for c in out if c["kind"] == "narrow" and c["sign"] > 0]
check("a narrowing probe is planned, and any restart of the far point is seeded",
      len(nr) == 1 and all(c["seed"] is not None for c in out
                           if c["kind"] == "rerun" and c["sign"] > 0),
      [(c["kind"], c["x"]) for c in out])
check("it lies strictly inside the bracket", nr and 0.3 < nr[0]["x"] < 0.9, nr)
check("seeded from the inner point", nr and nr[0]["seed"]["x_fixed"] == 0.3)
check("and it is not a duplicate of an existing point",
      nr and all(abs(nr[0]["x"] - r["x_fixed"]) > 1e-9 for r in pts["a"]))

print("an unconverged outer point is narrowed from the optimum if nothing is inner")
pts = {"a": [rec(0.9, 4.0, converged=False)]}
out = plan(completed=pts, n_free=4)
nr = [c for c in out if c["kind"] == "narrow" and c["sign"] > 0]
check("a cold narrowing probe lies between the optimum and the failed point",
      len(nr) == 1 and 0.0 < nr[0]["x"] < 0.9 and nr[0]["seed"] is None, nr)

print("the Hyak case: a far cold point failed while the near rung still runs")
# k_plus upper: 0.035 decades unconverged at dNLL 7.6, the 0.0012 rung in flight.
pts = {"a": [rec(0.035, 7.6, converged=False), rec(1.0, 830.0, converged=False)]}
fl = [{"param_idx": 0, "x_fixed": 0.0012}]
out = plan(completed=pts, in_flight=fl, n_free=4)
up = [c for c in out if c["sign"] > 0]
check("the bracket is narrowed without waiting for the inner rung",
      any(c["kind"] == "narrow" and c["tier"] == 1 for c in up),
      [(c["kind"], c["x"]) for c in out])
pts["a"].append(rec(0.0012, 0.007))
out = plan(completed=pts, n_free=4)
up = [c for c in out if c["sign"] > 0]
nr = [c for c in up if c["kind"] == "narrow"]
check("once it lands, the probe goes between it and the failed point, seeded from it",
      nr and nr[0]["tier"] == 1
      and 0.0012 < nr[0]["x"] < 0.035 and nr[0]["seed"]["x_fixed"] == 0.0012,
      [(c["kind"], c["x"]) for c in up])
rr = [c for c in up if c["kind"] == "rerun"]
check("and the failed point is restarted from it, at once, not last",
      len(rr) == 1 and rr[0]["x"] == 0.035 and rr[0]["tier"] == 1
      and rr[0]["seed"]["x_fixed"] == 0.0012, rr)
check("never a re-run of an unseeded point",
      not any(c["kind"] == "rerun" and c["seed"] is None for c in out))

print("a tight bracket whose outer end is unconverged: re-run it once, warm")
pts = {"a": [rec(0.90, 1.5), rec(0.92, 2.5, converged=False)]}
att = set()
out = plan(completed=pts, attempted=att, n_free=4)
rr = [c for c in out if c["kind"] == "rerun" and c["sign"] > 0]
check("a re-run is planned", len(rr) == 1, [(c["kind"], c["x"]) for c in out])
check("seeded from the inner point", rr and rr[0]["seed"]["x_fixed"] == 0.90)
att.add(rr[0]["attempt_key"])
out = plan(completed=pts, attempted=att, n_free=4)
check("and not planned twice",
      not any(c["kind"] == "rerun" and c["sign"] > 0 for c in out))
out = plan(completed={"a": [rec(1e-4, 2.5, converged=False)]}, n_free=4)
rr = [c for c in out if c["kind"] == "rerun"]
check("a cold original with no inner neighbour is carried on from its own end "
      "state, never restarted from the same start",
      len(rr) == 1 and rr[0]["seed"] is not None
      and rr[0]["seed"]["x_fixed"] == 1e-4, [(c["kind"], c["x"]) for c in out])

print("bracketed: narrow it, then fill, then stop")
wide = {"a": [rec(0.3, 0.8), rec(1.2, 6.0)]}
out = plan(completed=wide, n_free=4)
br = [c for c in out if c["kind"] == "bracket" and c["sign"] > 0]
check("a wide bracket gets an interpolated probe",
      len(br) == 1 and 0.3 < br[0]["x"] < 1.2, br)
tight = {"a": [rec(0.3, 0.8), rec(0.6, 1.5), rec(0.90, 1.90), rec(0.92, 2.0)]}
out = plan(completed=tight, n_free=4)
check("a tight bracket with n_grid points plans nothing on its side",
      not any(c["sign"] > 0 for c in out), [c for c in out if c["sign"] > 0])
sparse = {"a": [rec(0.90, 1.90), rec(0.92, 2.0)]}
out = plan(completed=sparse, n_free=4)
check("a tight bracket short of n_grid is filled",
      any(c["kind"] == "fill" and c["sign"] > 0 for c in out))


print("idle workers get extra probes inside an open bracket")
wide = {"a": [rec(0.3, 0.8), rec(1.2, 6.0)]}
one = [c for c in plan(completed=wide, n_free=2) if c["sign"] > 0]
many = [c for c in plan(completed=wide, n_free=12) if c["sign"] > 0]
check("with no spare worker the side gets the interpolated guess only",
      len(one) == 1 and one[0]["tier"] == 1, one)
check("spare workers get several probes in the same bracket",
      len(many) >= 4 and all(0.3 < c["x"] < 1.2 for c in many),
      [(c["kind"], round(c["x"], 3), c["tier"]) for c in many])
check("the guess comes first and the extras rank below every primary",
      many[0]["tier"] == 1 and all(c["tier"] >= 4 for c in many[1:]))
check("the extras are distinct and all seeded from the inner point",
      len({round(c["x"], 9) for c in many}) == len(many)
      and all(c["seed"]["x_fixed"] == 0.3 for c in many))
two = {"a": [rec(0.3, 0.8), rec(1.2, 6.0)], "b": [rec(0.3, 0.8), rec(1.2, 6.0)]}
mixed = plan(completed=two, names=("a", "b"), res_x=(0.0, 0.0), se=(0.3, 0.3),
             bounds=((-6.0, 6.0), (-6.0, 6.0)), scales=("log10", "log10"),
             n_free=4)
ups = [c for c in mixed if c["sign"] > 0]
check("every side gets its own guess before any side gets a second probe",
      sorted(c["name"] for c in ups if c["tier"] == 1) == ["a", "b"], ups)
fl = [{"param_idx": 0, "x_fixed": float(c["x"])} for c in many]
again = [c for c in plan(completed=wide, in_flight=fl, n_free=12) if c["sign"] > 0]
check("running probes are not planned again, and the rest of the bracket is not "
      "re-planned on top of them",
      not any(abs(c["x"] - f["x_fixed"]) < 1e-9 for c in again for f in fl), again)
located = {"a": [rec(0.3, 0.8), rec(0.90, 1.92), rec(1.2, 6.0)]}
out = plan(completed=located, n_free=12)
check("a crossing already located by a measured point gets no extras",
      not any(c["kind"] in ("bracket", "narrow") and c["sign"] > 0 for c in out),
      [(c["kind"], c["x"]) for c in out if c["sign"] > 0])
unconv = {"a": [rec(0.3, 0.8), rec(1.2, 6.0, converged=False)]}
many = [c for c in plan(completed=unconv, n_free=12) if c["sign"] > 0]
many = [c for c in many if c["kind"] == "narrow"]
check("an unconverged outer end is narrowed with spares too",
      len(many) >= 4 and many[0]["tier"] == 1, many)

print("a point past the threshold is provisional until it has been warm-restarted")
base = [rec(0.3, 0.8)]
def planned(pts_, **kw):
    return plan(completed={"a": pts_}, n_free=12, **kw)
def reruns(out):
    return [c for c in out if c["kind"] == "rerun" and c["sign"] > 0]
out = planned(base + [rec(1.2, 6.0, converged=False)])
rr = reruns(out)
check("an unconverged nearest-above point is restarted at once, from the inner point",
      len(rr) == 1 and rr[0]["tier"] == 1 and rr[0]["seed"]["x_fixed"] == 0.3, rr)
check("the narrowing probes are planned in the same round",
      any(c["kind"] == "narrow" and c["sign"] > 0 for c in out))
out = planned(base + [rec(1.2, 6.0, warm=False)])
check("a converged but cold point is provisional too", len(reruns(out)) == 1,
      [(c["kind"], c["x"]) for c in out])
out = planned(base + [rec(1.2, 6.0)])
check("a converged warm point is firm: no restart", not reruns(out))
out = planned(base + [rec(1.2, 6.0, converged=False, pass_label="pass1-adaptive:rerun")])
check("a restarted point is firm even unconverged: no second restart, and it "
      "ends the bracket", not reruns(out)
      and any(c["kind"] == "bracket" and c["sign"] > 0 for c in out),
      [(c["kind"], c["x"]) for c in out])
out = planned(base + [rec(1.2, 6.0, converged=False, reruns_tried=1)])
check("a restart that failed to lower the point is remembered in the record",
      not reruns(out))
out = planned(base + [rec(0.9, 3.0), rec(1.5, 40.0, warm=False)])
check("a provisional point beyond a firm end cannot move the crossing: left alone",
      not reruns(out), reruns(out))
out = planned(base + [rec(0.8, 2.5, converged=False), rec(1.2, 40.0, warm=False)])
rr = sorted(reruns(out), key=lambda c: c["x"])
check("the nearest two provisional points are restarted, the nearer first",
      [c["x"] for c in rr] == [0.8, 1.2] and rr[0]["tier"] == 1
      and rr[1]["tier"] > 1, [(c["x"], c["tier"]) for c in rr])
out = plan(completed={"a": base + [rec(1.2, 6.0, converged=False)]},
           in_flight=[{"param_idx": 0, "x_fixed": 1.2}], n_free=12)
check("a restart already running is not planned again", not reruns(out))
unc_same = base + [rec(1.2, 6.0, converged=False, x_start=[0.0, 0.0])]
rr = reruns(planned(unc_same))
check("an unconverged point whose start matches the inner point's is carried on "
      "from its own end state", len(rr) == 1 and rr[0]["seed"]["x_fixed"] == 1.2, rr)
conv_same = base + [rec(1.2, 6.0, warm=False, x_start=[0.0, 0.0])]
check("a converged one with the same start has nothing new to try",
      not reruns(planned(conv_same)))

print("a re-run is not repeated by a relaunch that forgot it")
rr = rec(0.92, 2.5, converged=False)
rr["pass_label"] = "pass1-adaptive:rerun"
out = plan(completed={"a": [rec(0.90, 1.5), rr]}, attempted=set(), n_free=4)
check("a point that is itself a re-run is not re-run again",
      not any(c["kind"] == "rerun" and c["sign"] > 0 for c in out),
      [(c["kind"], c["x"]) for c in out])

print("a point over the threshold with one under it further out is not a crossing")
# The k_oligo2 case: 2.90 between 0.41 and 0.62, and under the threshold beyond.
stuck = [rec(0.10, 0.05, nuis=[0.1, 0.0]), rec(0.20, 0.41, nuis=[0.2, 0.0]),
         rec(0.21, 2.90, x_start=[0.2, 0.0], nuis=[0.9, 0.9]),
         rec(0.30, 0.62, nuis=[0.3, 0.0]), rec(0.50, 1.31, nuis=[0.5, 0.0]),
         rec(0.80, 5.09, nuis=[0.8, 0.0])]
out = plan(completed={"a": stuck}, n_free=12)
up = [c for c in out if c["sign"] > 0]
rr = [c for c in up if c["kind"] == "rerun"]
check("the stuck point is restarted, from the neighbour it was not started from",
      len(rr) == 1 and rr[0]["x"] == 0.21 and rr[0]["seed"]["x_fixed"] == 0.30, rr)
br = [c for c in up if c["kind"] in ("bracket", "narrow")]
check("and the bracket is sought beyond the furthest point that is under",
      br and all(0.50 < c["x"] < 0.80 for c in br), [(c["kind"], c["x"]) for c in up])
rr0 = dict(stuck[2], reruns_tried=1)
out = plan(completed={"a": stuck[:2] + [rr0] + stuck[3:]}, n_free=12)
check("a restart that did not lower it is not repeated",
      not [c for c in out if c["kind"] == "rerun" and c["sign"] > 0])
only_in = [rec(0.10, 0.05), rec(0.20, 2.90, x_start=[0.0, 0.0], nuis=[0.9, 0.9]),
           rec(0.40, 6.0)]
out = plan(completed={"a": only_in}, n_free=12)
check("an over-threshold point with nothing under beyond it is still the bracket",
      any(c["kind"] == "bracket" and 0.10 < c["x"] < 0.20 for c in out if c["sign"] > 0),
      [(c["kind"], c["x"]) for c in out if c["sign"] > 0])
check("and a side's under-threshold outer point leaves it open, not bracketed",
      any(c["kind"] == "extend" for c in plan(
          completed={"a": [rec(0.10, 0.05), rec(0.20, 2.90), rec(0.40, 0.8)]},
          n_free=12) if c["sign"] > 0))

print("spare-core extras do not use up a side's allowance")
many_pts = [rec(0.05 * k, 0.1 * k) for k in range(1, 8)]
spare = [rec(0.4 + 0.01 * k, 1.0 + 0.01 * k, plan_tier=5) for k in range(10)]
kept = [rec(0.4 + 0.01 * k, 1.0 + 0.01 * k) for k in range(10)]
out_spare = plan(completed={"a": many_pts + spare + [rec(2.0, 5.0)]}, n_free=12)
out_kept = plan(completed={"a": many_pts + kept + [rec(2.0, 5.0)]}, n_free=12)
check("a side full of extras is still worked on",
      any(c["sign"] > 0 for c in out_spare), out_spare)
check("the same points as primaries exhaust the allowance",
      not any(c["sign"] > 0 for c in out_kept), out_kept)
check("a hard cap at twice the allowance still ends a side",
      not any(c["sign"] > 0 for c in plan(
          completed={"a": many_pts + spare + [rec(2.0 + 0.01 * k, 5.0, plan_tier=5)
                                             for k in range(20)]}, n_free=12)))

print("two firm ends within the optimizer's scatter are a located crossing")
wide_flat = {"a": [rec(0.30, 1.80), rec(0.50, 2.00)]}
out = plan(completed=wide_flat, n_free=12)
check("a bracket 0.2 decades wide but 0.2 nats high is not narrowed further",
      not [c for c in out if c["kind"] in ("bracket", "narrow") and c["sign"] > 0],
      [(c["kind"], c["x"]) for c in out if c["sign"] > 0])
wide_steep = {"a": [rec(0.30, 1.0), rec(0.50, 4.0)]}
check("the same width with a 3-nat rise is narrowed",
      any(c["kind"] == "bracket" for c in plan(completed=wide_steep, n_free=12)
          if c["sign"] > 0))
loose = {"a": [rec(0.30, 1.80), rec(0.50, 2.00, converged=False)]}
check("an unconverged end does not qualify",
      any(c["kind"] == "narrow" for c in plan(completed=loose, n_free=12)
          if c["sign"] > 0))

# ------------------------------------------------------------------ stream ---

K = 3
A = np.array([[100.0, 99.0, 0.0], [99.0, 100.0, 0.0], [0.0, 0.0, 4.0]])
# Profile curvature of p0 and p1: 100 - 99^2/100 = 1.99 (SE 0.709); the slice's
# own curvature is 100 (SE 0.1). p2 is separable, SE 0.5.
SE = np.array([1.0 / np.sqrt(1.99), 1.0 / np.sqrt(1.99), 0.5])


def nll(x):
    x = np.asarray(x, dtype=float)
    return float(0.5 * x @ A @ x)


class SimPool:
    """Obeys the ``refill`` contract of ParallelEvaluator.profile_batch."""

    def __init__(self, n_workers):
        self.n_workers = n_workers
        self.max_in_flight = 0
        self.in_flight_after_admit = []
        self.refill_calls = 0
        self.n_points = 0

    def run(self, job):
        i, xf = job["param_idx"], job["x_fixed"]

        def f(z, i=i, xf=xf):
            return nll(np.insert(z, i, xf))

        opts = {"maxfev": 2000, "xatol": 1e-7, "fatol": 1e-10}
        sim = job.get("initial_simplex")
        if sim is not None:
            opts["initial_simplex"] = np.asarray(sim, dtype=float)
        r = minimize(f, np.asarray(job["x_start"], dtype=float),
                     method="Nelder-Mead", options=opts)
        out = dict(job)
        out.pop("initial_simplex", None)
        out.update({"nll": float(r.fun), "nuisance_x": r.x.tolist(),
                    "nm_simplex": np.asarray(r.final_simplex[0]).tolist(),
                    "status": "ok", "converged": True, "nit": int(r.nit),
                    "nfev": int(r.nfev), "nit_total": int(r.nit),
                    "nfev_total": int(r.nfev)})
        return out

    def batch(self, jobs, on_result=None, label=None, budget=None,
              frozen_sigmas=None, state_dir=None, refill=None):
        W = self.n_workers
        backlog = list(jobs)
        pending = []          # heap of (finish, seq, job, result)
        now, seq = 0.0, 0

        def top_up():
            self.refill_calls += 1
            n_free = W - len(pending) - len(backlog)
            if refill is None or n_free <= 0:
                return
            new = refill(n_free, [p[2] for p in pending]) or []
            assert len(new) <= n_free, "planner returned more than n_free"
            backlog.extend(new)

        def admit():
            nonlocal seq
            while backlog and len(pending) < W:
                job = backlog.pop(0)
                seq += 1
                self.n_points += 1
                heapq.heappush(pending, (now + 1.0 + (seq % 3), seq, job,
                                         self.run(job)))
            self.max_in_flight = max(self.max_in_flight, len(pending))
            self.in_flight_after_admit.append((len(pending), len(backlog)))

        top_up()
        admit()
        while pending:
            finish, _s, _job, res = heapq.heappop(pending)
            now = finish
            on_result(res)
            top_up()
            admit()


def run_stream(workers, **kw):
    pool = SimPool(workers)
    names = [f"p{i}" for i in range(K)]
    buf = io.StringIO()
    comp = {}
    from pyantigen.engine.Profile_checkpoint import ProfileCheckpoint
    import tempfile
    with tempfile.TemporaryDirectory() as root:
        ck = ProfileCheckpoint(root, "s", "m", "s", enabled=True)
        with contextlib.redirect_stdout(buf):
            traces, anchor, where, conv = run_parallel_profile(
                pool.batch, np.zeros(K), 0.0, names, [(-50.0, 50.0)] * K,
                ["lin"] * K, method="Nelder-Mead", wald_se=SE, n_grid=3,
                n_refine=2, bracket_rtol=0.05, checkpoint=ck,
                adaptive_pass1=True, n_workers=workers, **kw)
        ck.close()
        import json
        for n in names:
            with open(ck.path_for(n), encoding="utf-8") as fh:
                comp[n] = [json.loads(line) for line in fh if line.strip()]
    return pool, comp, buf.getvalue()


print("the stream keeps the pool full and never over-full")
for W in (4, 6, 9):
    pool, comp, _log = run_stream(W)
    check(f"W={W}: never more than n_workers in flight",
          pool.max_in_flight <= W, pool.max_in_flight)
    first_wave = pool.in_flight_after_admit[0][0]
    check(f"W={W}: the first wave fills the pool",
          first_wave == W, first_wave)
    # A slot may sit empty only when the planner has nothing to give it.
    idle_with_backlog = [x for x in pool.in_flight_after_admit
                         if x[0] < W and x[1] > 0]
    check(f"W={W}: no slot is left empty while work is waiting",
          not idle_with_backlog, idle_with_backlog[:3])

print("every side ends bracketed, from a first probe near the Wald guess")
pool, comp, log = run_stream(6)
for n in ("p0", "p1", "p2"):
    for sign, nm in ((-1, "lower"), (1, "upper")):
        side = [r for r in comp[n] if r["x_fixed"] * sign > 0]
        above = [r for r in side if r["nll"] > THR]
        below = [r for r in side if r["nll"] <= THR]
        check(f"{n} {nm}: has points on both sides of the threshold",
              bool(above) and bool(below), (len(above), len(below)))
p0 = [r for r in comp["p0"] if r["pass_label"].endswith("first probe")]
check("the first probes on p0 are at 1.96 SE of the profile, not of the slice",
      all(abs(abs(r["x_fixed"]) - 1.96 * SE[0]) < 1e-6 for r in p0),
      [r["x_fixed"] for r in p0])
check("every record carries the side and an adaptive label",
      all(r["direction"] == (-1 if r["x_fixed"] < 0 else 1)
          and r["pass_label"].startswith("pass1-adaptive")
          for n in comp for r in comp[n]
          if r["pass_label"].startswith("pass1")))
check("the log says which settings were planned",
      "pass 1 (adaptive)" in log)

print("slice-ladder stream, no Wald SE: every side still ends bracketed")
curv = {"p0": 100.0, "p1": 100.0, "p2": 4.0}
scr_e = {"parameters": {n: {"upper": {"crossing": {"x": float(np.sqrt(2 * 4.4 / c))}},
                            "lower": {"crossing": {"x": -float(np.sqrt(2 * 4.4 / c))}}}
                        for n, c in curv.items()}}
pool = SimPool(6)
comp = {}
import json
import tempfile
from pyantigen.engine.Profile_checkpoint import ProfileCheckpoint
with tempfile.TemporaryDirectory() as root:
    ck = ProfileCheckpoint(root, "e", "m", "s", enabled=True)
    with contextlib.redirect_stdout(io.StringIO()):
        run_parallel_profile(
            pool.batch, np.zeros(K), 0.0, [f"p{i}" for i in range(K)],
            [(-50.0, 50.0)] * K, ["lin"] * K, method="Nelder-Mead",
            wald_se=None, n_grid=3, n_refine=2, bracket_rtol=0.05,
            checkpoint=ck, adaptive_pass1=True, n_workers=6, screen=scr_e,
            first_probe="slice_ladder")
    ck.close()
    for n in ("p0", "p1", "p2"):
        with open(ck.path_for(n), encoding="utf-8") as fh:
            comp[n] = [json.loads(line) for line in fh if line.strip()]
check("never more than n_workers in flight", pool.max_in_flight <= 6,
      pool.max_in_flight)
for n in ("p0", "p1", "p2"):
    for sign, nm in ((-1, "lower"), (1, "upper")):
        side = [r for r in comp[n] if r["x_fixed"] * sign > 0]
        check(f"{n} {nm} bracketed",
              any(r["nll"] > THR for r in side)
              and any(r["nll"] <= THR for r in side))
check("the first wave was slice ladder probes",
      any("slice ladder" in r["pass_label"] for n in comp for r in comp[n]))

print("far cold starts that do not converge: narrowed, never repeated from the same start")


class FarColdFailsPool(SimPool):
    """A cold start more than FAR from the optimum stalls high and unconverged,
    as the SILK points a decade out did; a warm start (seeded from a neighbour)
    or a near one converges. Deterministic, like the real optimizer."""

    FAR = 0.3

    def run(self, job):
        out = super().run(job)
        cold = not job.get("warm_seeded")
        if cold and abs(job["x_fixed"]) > self.FAR:
            out["nll"] = out["nll"] + 50.0
            out["converged"] = False
            out["nfev_total"] = 2000
        return out


pool = FarColdFailsPool(6)
comp = {}
starts = {}
with tempfile.TemporaryDirectory() as root:
    ck = ProfileCheckpoint(root, "f", "m", "s", enabled=True)
    with contextlib.redirect_stdout(io.StringIO()):
        run_parallel_profile(
            pool.batch, np.zeros(K), 0.0, [f"p{i}" for i in range(K)],
            [(-50.0, 50.0)] * K, ["lin"] * K, method="Nelder-Mead",
            wald_se=None, n_grid=3, n_refine=2, bracket_rtol=0.05,
            checkpoint=ck, adaptive_pass1=True, n_workers=6, screen=scr_e,
            first_probe="slice_ladder", open_decades=1.0)
    ck.close()
    for n in ("p0", "p1", "p2"):
        with open(ck.path_for(n), encoding="utf-8") as fh:
            comp[n] = [json.loads(line) for line in fh if line.strip()]
for n in ("p0", "p1", "p2"):
    for sign, nm in ((-1, "lower"), (1, "upper")):
        side = [r for r in comp[n] if r["x_fixed"] * sign > 0]
        conv_above = [r for r in side if r["nll"] > THR and r["converged"]]
        conv_below = [r for r in side if r["nll"] <= THR and r["converged"]]
        check(f"{n} {nm}: a converged point on each side of the threshold",
              bool(conv_above) and bool(conv_below),
              (len(conv_above), len(conv_below)))
        check(f"{n} {nm}: narrowing probes were used",
              any(r["pass_label"].endswith("narrow") for r in side))
dups = [(n, r["x_fixed"]) for n in comp for r in comp[n]
        if r["pass_label"].endswith("rerun") and not r.get("warm_seeded")]
check("no unseeded re-run of a point", not dups, dups)

print("a warm restart that lowers a point past the threshold, and one that cannot")


class ExcessPool(SimPool):
    """Every first run farther out than FAR comes back inflated and unconverged
    (an upper bound with excess). A restart returns the true value if
    ``restart_helps``, otherwise the same inflated one."""

    FAR = 0.3

    def __init__(self, n, restart_helps):
        super().__init__(n)
        self.restart_helps = restart_helps
        self.restarts = []

    def run(self, job):
        out = super().run(job)
        is_restart = str(job.get("pass_label", "")).endswith("rerun")
        if is_restart:
            self.restarts.append((job["param_name"], job["x_fixed"]))
        if abs(job["x_fixed"]) > self.FAR and not (is_restart and self.restart_helps):
            out["nll"] = out["nll"] + 0.6
            out["converged"] = False
            out["nfev_total"] = 1500
        return out


def launch(pool, root, n_refine=0):
    ck = ProfileCheckpoint(root, "f", "m", "s", enabled=True)
    with contextlib.redirect_stdout(io.StringIO()):
        run_parallel_profile(
            pool.batch, np.zeros(K), 0.0, [f"p{i}" for i in range(K)],
            [(-50.0, 50.0)] * K, ["lin"] * K, method="Nelder-Mead",
            wald_se=None, n_grid=3, n_refine=n_refine, bracket_rtol=0.05,
            checkpoint=ck, adaptive_pass1=True, n_workers=6, screen=scr_e,
            first_probe="slice_ladder", open_decades=1.0, warm_passes=0)
    ck.close()
    recs = {}
    for n in ("p0", "p1", "p2"):
        with open(ck.path_for(n), encoding="utf-8") as fh:
            recs[n] = [json.loads(line) for line in fh if line.strip()]
    return recs


with tempfile.TemporaryDirectory() as root:
    pool = ExcessPool(6, restart_helps=True)
    recs = launch(pool, root)
    check("restarts happened", len(pool.restarts) > 0)
    check("no point is restarted twice in one run",
          len(pool.restarts) == len(set(pool.restarts)), pool.restarts)
    ends = []
    for n in recs:
        best = {}
        for r in recs[n]:
            if r["x_fixed"] not in best or r["nll"] <= best[r["x_fixed"]]["nll"]:
                best[r["x_fixed"]] = r
        for sign in (-1, 1):
            side = sorted([r for r in best.values() if r["x_fixed"] * sign > 0],
                          key=lambda r: abs(r["x_fixed"]))
            first_above = next((r for r in side if r["nll"] > THR), None)
            ends.append(first_above is not None and (
                first_above["converged"] or first_above.get("reruns_tried")))
    check("every side's first point past the threshold is firm: converged, or "
          "restarted", all(ends), ends)

with tempfile.TemporaryDirectory() as root:
    pool = ExcessPool(6, restart_helps=False)
    recs = launch(pool, root)
    done = [r for n in recs for r in recs[n] if r.get("reruns_tried")]
    check("a restart that did not lower a point is recorded on it",
          len(done) > 0 and all(r["reruns_tried"] == 1 for r in done),
          [(r["x_fixed"], r.get("reruns_tried")) for r in done])
    pool2 = ExcessPool(6, restart_helps=False)
    launch(pool2, root)
    check("a relaunch over the same checkpoint restarts none of them again",
          not set(pool2.restarts) & set(pool.restarts),
          sorted(set(pool2.restarts) & set(pool.restarts)))

print("the grid pass still works when the stream is off")
pool = SimPool(6)
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    run_parallel_profile(
        pool.batch, np.zeros(K), 0.0, [f"p{i}" for i in range(K)],
        [(-50.0, 50.0)] * K, ["lin"] * K, method="Nelder-Mead", wald_se=SE,
        n_grid=3, n_refine=1, checkpoint=None)
check("grid points ran", pool.n_points > 0)

if failures:
    print(f"\n{len(failures)} failure(s): {failures}")
    sys.exit(1)
print("\nall passed")

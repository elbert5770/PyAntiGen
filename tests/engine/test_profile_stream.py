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

def rec(x, dnll, converged=True, nuis=None):
    return {"x_fixed": x, "dnll": dnll, "converged": converged,
            "interrupted": False, "nuisance_x": nuis or [0.0, 0.0]}


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
check("the upper side extends", len(up) == 1 and up[0]["kind"] == "extend", up)
check("by no more than max_step_decades",
      up[0]["x"] - 0.4 <= 0.7 + 1e-9, up[0]["x"])
check("warm-started from its outer point", up[0]["seed"]["x_fixed"] == 0.4)

print("crossed only on an unconverged point: re-run it once, warm")
pts = {"a": [rec(0.3, 0.8), rec(0.9, 4.0, converged=False)]}
att = set()
out = plan(completed=pts, attempted=att, n_free=4)
rr = [c for c in out if c["kind"] == "rerun" and c["sign"] > 0]
check("a re-run is planned", len(rr) == 1)
check("seeded from the inner point", rr and rr[0]["seed"]["x_fixed"] == 0.3)
att.add(rr[0]["attempt_key"])
out = plan(completed=pts, attempted=att, n_free=4)
check("and not planned twice",
      not any(c["kind"] == "rerun" and c["sign"] > 0 for c in out))

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

"""The shadow stop rules: they watch a point and never change it.

Run from the repository root:
    python tests/engine/test_stop_rules.py
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from pyantigen.engine.Stop_rules import (                                   # noqa: E402
    PointMonitor, format_shadow_summary, shadow_summary)

failures = []
THR = 1.9207
ANCHOR = -300.0


def check(name, cond, detail=""):
    if cond:
        print(f"  ok    {name}")
    else:
        print(f"  FAIL  {name} {detail}")
        failures.append(name)


def feed(mon, us, start=1):
    """Feed a list of U values as consecutive evaluations; return the monitor."""
    best = math.inf
    for k, u in enumerate(us):
        best = min(best, ANCHOR + u)
        mon.update(start + k, best)
    return mon


def monitor(max_fev=1500, n=8, **kw):
    return PointMonitor(ANCHOR, max_fev, n, cfg=kw or None)


print("the trace is sparse and ends on the last evaluation")
m = feed(monitor(), [50.0] * 130)
ns = [p[0] for p in m.trace]
check("first sample is evaluation 1", ns[0] == 1, ns[:3])
check("then every 25", ns[1:] == [25, 50, 75, 100, 125], ns)
out = m.finish(130, ANCHOR + 50.0)
check("finish appends the final evaluation", out["trace"][-1][0] == 130)
check("best values are stored as nll, not dNLL", out["trace"][0][1] == ANCHOR + 50.0)

print("below: a proof, fires at once and needs no minimum spend")
m = feed(monitor(), [900.0, 400.0, 0.3])
check("fires on the evaluation that gets under 0.25*thr",
      m.fired.get("below", {}).get("nfev") == 3, m.fired)
m = feed(monitor(), [900.0, 400.0, 0.6])
check("does not fire above 0.25*thr", "below" not in m.fired, m.fired)

print("the window rules wait for the initial simplex and a window")
m = feed(monitor(), [40.0] * 100)
check("a flat U=40 at eval 100 has fired nothing yet", not m.fired, m.fired)
m = feed(monitor(), [40.0] * 200)
check("by eval 200 it is settled", "settled" in m.fired, m.fired)
check("and hopeless: flat and above u_trust",
      "hopeless" in m.fired, m.fired)

print("hopeless: only if even the current rate would not reach u_trust by the cap")
creeping = [1000.0 * math.exp(-0.001 * k) for k in range(300)]   # 1000 -> 741
m = feed(monitor(), creeping)
check("a slow creep from 1000 is hopeless", "hopeless" in m.fired, m.fired)
fast = [1000.0 * math.exp(-0.03 * k) for k in range(300)]         # to ~0.1
m = feed(monitor(), fast)
check("a fast descent is not", "hopeless" not in m.fired, m.fired)
m = feed(monitor(), [15.0] * 300)
check("U under u_trust is never hopeless", "hopeless" not in m.fired, m.fired)
m = feed(PointMonitor(ANCHOR, None, 8), [500.0] * 300)
check("with no cap there is nothing to extrapolate to",
      "hopeless" not in m.fired, m.fired)

print("settled: the tolerance scales with the distance from the decision")
m = feed(monitor(), [0.04] * 200)
check("U=0.04 flat is settled", "settled" in m.fired, m.fired)
slow_low = [2.05 - 0.0001 * k for k in range(200)]    # gain 0.01/100 evals
m = feed(monitor(), slow_low)
# |U - thr| is ~0.13 at the end, so the tolerance is ~0.0066 < the 0.01 gained
check("U near thr that is still gaining more than 5% of its distance is not settled",
      "settled" not in m.fired, m.fired)
plateau = [20.0 - 0.0001 * k for k in range(200)]
m = feed(monitor(), plateau)
check("U=20 that gained 0.01 in a window is settled (needs 0.9)",
      "settled" in m.fired, m.fired)
big_gain = [30.0 - 0.1 * k for k in range(200)]       # gains 10 per window
m = feed(monitor(), big_gain)
check("U still falling by 10 per window is not settled",
      "settled" not in m.fired, m.fired)

print("a resumed point keeps one history")
a = feed(monitor(), [40.0] * 120)
part = a.finish(120, ANCHOR + 40.0)
b = PointMonitor(ANCHOR, 1500, 8, trace=part["trace"], fired=part["shadow_stops"])
feed(b, [40.0] * 100, start=121)
ns = [p[0] for p in b.trace]
check("numbering continues", ns == sorted(set(ns)) and ns[-1] >= 200, ns)
check("window rules use the earlier launch's samples", "settled" in b.fired,
      b.fired)
feed(b, [0.1] * 5, start=221)
check("below can still fire after the resume", "below" in b.fired, b.fired)
once = dict(b.fired["settled"])
feed(b, [40.0] * 100, start=300)
check("a rule fires once", b.fired["settled"] == once)

print("bad input never raises")
m = monitor()
m.update(1, None)
m.update(2, float("inf"))
m.update(3, float("nan"))
check("non-finite values are ignored", not m.trace and not m.fired)

print("summary: savings and the class changes that are the real cost")
recs = [
    {"param_name": "a", "x_fixed": 1.0, "nll": ANCHOR + 0.1, "nfev_total": 900,
     "shadow_stops": {"below": {"nfev": 80, "U": 0.4}}},
    {"param_name": "b", "x_fixed": 2.0, "nll": ANCHOR + 1.0, "nfev_total": 1500,
     "shadow_stops": {"settled": {"nfev": 400, "U": 2.5}}},
    {"param_name": "c", "x_fixed": 3.0, "nll": ANCHOR + 600.0, "nfev_total": 1500,
     "shadow_stops": {"hopeless": {"nfev": 300, "U": 900.0}}},
    {"param_name": "d", "x_fixed": 4.0, "nll": ANCHOR + 3.0, "nfev_total": 700},
]
s = shadow_summary(recs, ANCHOR)
check("points without a trace are skipped", s["n_points"] == 3, s["n_points"])
check("below saved 820", s["rules"]["below"]["saved"] == 820)
check("settled saved 1100", s["rules"]["settled"]["saved"] == 1100)
check("settled changed class: mid at 2.5 -> below at 1.0",
      s["rules"]["settled"]["class_changed"] == 1)
check("hopeless kept its class (high -> high)",
      s["rules"]["hopeless"]["class_changed"] == 0)
check("below kept its class", s["rules"]["below"]["class_changed"] == 0)
text = format_shadow_summary(s)
check("the log names the class change", "changed: b" in text, text)
check("an empty run prints nothing", format_shadow_summary(
    shadow_summary([], ANCHOR)) == "")


# ----------------------------------------------------------- the worker ---

print("through the real worker: watching changes nothing")
from pyantigen.engine.Evaluator import EvalSpec, _profile_task               # noqa: E402
import pyantigen.engine.Evaluator as ev_mod                                    # noqa: E402


def install(fn, k=4):
    ev_mod._WORKER["spec"] = EvalSpec(
        model_text="x", paths={}, events={}, replicates={"sim": {}},
        param_names=[f"p{i}" for i in range(k)], scales=["lin"] * k, groups={},
        group_normalization=None, fixed_sigmas={})
    ev_mod._WORKER["models"] = {}
    ev_mod._worker_nll_original = getattr(
        ev_mod, "_worker_nll_original", ev_mod._worker_nll)
    ev_mod._worker_nll = lambda x, frozen_sigmas=None: fn(x)


def restore():
    ev_mod._worker_nll = ev_mod._worker_nll_original
    ev_mod._WORKER["spec"] = None


def bowl(x):
    x = np.asarray(x, float)
    return float(ANCHOR + 400.0 + np.sum((x - 1.3) ** 2) * 50.0)


def job(**kw):
    j = {"param_idx": 0, "param_name": "p0", "x_fixed": 0.0, "x_fixed_linear": 0.0,
         "x_start": [0.0, 0.0, 0.0], "nuisance_bounds": [(-5.0, 5.0)] * 3,
         "method": "Nelder-Mead", "optimizer_kwargs": None, "phase": 1,
         "direction": 0}
    j.update(kw)
    return j


install(bowl)
try:
    plain = _profile_task(job())
    watched = _profile_task(job(stop_rules={"anchor": ANCHOR, "threshold": THR}))
    check("same value", plain["nll"] == watched["nll"], (plain["nll"], watched["nll"]))
    check("same evaluations", plain["nfev_total"] == watched["nfev_total"])
    check("same nuisance vector", plain["nuisance_x"] == watched["nuisance_x"])
    check("an unwatched record has no trace", "trace" not in plain)
    check("a watched record has one", len(watched.get("trace") or []) >= 2)
    check("the trace ends at the point's total",
          watched["trace"][-1][0] == watched["nfev_total"])
    check("the trace is non-increasing in best value",
          all(b[1] <= a[1] + 1e-12 for a, b in zip(watched["trace"],
                                                   watched["trace"][1:])))
    check("it fired something on a descent from U=400",
          bool(watched["shadow_stops"]), watched["shadow_stops"])
    check("the settings are an input, not part of the record",
          "stop_rules" not in watched)
    check("the point says why it stopped",
          plain["stop_reason"] == "tolerance", plain.get("stop_reason"))

    capped = _profile_task(job(optimizer_kwargs={"options": {
        "maxfev": 40, "maxiter": 40, "fatol": 1e-4, "xatol": 1e-4}}))
    check("a capped point says so", capped["stop_reason"] == "maxfev",
          capped.get("stop_reason"))

    bad = _profile_task(job(stop_rules={"anchor": "not a number"}))
    check("a malformed watcher setting does not fail the point",
          bad["status"] == "ok" and bad["nll"] == plain["nll"],
          (bad["status"], bad.get("nll")))
    check("and the point simply has no trace", "trace" not in bad)
finally:
    restore()

if failures:
    print(f"\n{len(failures)} failure(s): {failures}")
    sys.exit(1)
print("\nall passed")

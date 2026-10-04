"""The slice screen's crossing window: walk a first crossing back in to dNLL in (1.92, 10].

Run from the repository root:
    python tests/engine/test_slice_window.py

On a model whose other parameters compensate, the first ladder point above the
threshold is far above it (a slice dNLL of 1e4 against a profile of a few). With
``window_hi`` set the screen walks such a side back inward until a point lands
inside the window. These objectives are analytic, so where the crossing is is
known: a stiff quadratic crosses at sqrt(2 * 1.9207) * sigma from the optimum.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from pyantigen.engine.Identifiability import (                               # noqa: E402
    THRESHOLD, load_screen, run_slice_screen, save_screen,
)
from pyantigen.engine.Evaluator import FAILURE_VALUE                         # noqa: E402

failures = []


def check(name, cond, detail=""):
    if cond:
        print(f"  ok    {name}")
    else:
        print(f"  FAIL  {name} {detail}")
        failures.append(name)


NAMES = ["stiff", "soft", "flat"]
RES_X = np.array([0.0, 0.0, 0.0])
BOUNDS = [(-3.0, 3.0)] * 3
SIGMA = np.array([0.002, 0.2, 1e9])     # decades; the last is flat


def make_nll(fail_inside=None):
    calls = []

    def nll_batch(xs, label=None):
        calls.append(label)
        out = []
        for x in xs:
            d = (np.asarray(x) - RES_X) / SIGMA
            if fail_inside is not None and abs(x[0]) < fail_inside \
                    and abs(x[0]) > 0:
                out.append(FAILURE_VALUE)
            else:
                out.append(float(0.5 * np.sum(d * d)))
        return out
    return nll_batch, calls


def run(window_hi, **kw):
    nll_batch, calls = make_nll(**kw)
    rep = run_slice_screen(nll_batch, RES_X, 0.0, NAMES, BOUNDS,
                           scales=["log10"] * 3, wald_se=np.array([0.4] * 3),
                           threshold=THRESHOLD, window_hi=window_hi,
                           verbose=False)
    return rep, calls


print("without a window: first crossing is wherever the ladder lands")
rep, calls = run(None)
c = rep["parameters"]["stiff"]["upper"]
check("stiff side crossed", c["state"] == "crossed")
check("and its crossing is far above the window", c["points"][-1]["dnll"] > 10,
      c["points"][-1]["dnll"])
check("report records no window", rep["window_hi"] is None)

print("with a window: stiff side is walked back in")
rep, calls = run(10.0)
for side in ("lower", "upper"):
    c = rep["parameters"]["stiff"][side]
    cr = c["crossing"]
    check(f"stiff {side} has a crossing in the window",
          cr is not None and cr["in_window"]
          and THRESHOLD < cr["dnll"] <= 10.0, cr)
    xs = [abs(p["x"]) for p in c["points"]]
    check(f"stiff {side} points are ordered outward", xs == sorted(xs), xs)
    check(f"stiff {side} landed within the round cap", len(c["points"]) <= 7,
          len(c["points"]))
exact = np.sqrt(2 * THRESHOLD) * SIGMA[0]
check("crossing distance is of the right order",
      0.5 * exact < rep["parameters"]["stiff"]["upper"]["crossing"]["decades"]
      < 3.0 * exact,
      rep["parameters"]["stiff"]["upper"]["crossing"]["decades"])

print("a side that already crossed inside the window is not refined")
c = rep["parameters"]["soft"]["upper"]
check("soft side settled on its first crossing",
      len(c["points"]) <= 3 and c["crossing"]["in_window"], c["points"])

print("the open verdict is untouched")
c = rep["parameters"]["flat"]["upper"]
check("flat side is still open", c["state"] == "open", c["state"])
check("and has no crossing", c["crossing"] is None)

print("an unevaluable point inside the first crossing counts as too high")
rep, _ = run(10.0, fail_inside=0.01)
c = rep["parameters"]["stiff"]["upper"]
check("screen still returns, side crossed", c["state"] == "crossed", c["state"])
check("no point reports an unusable dNLL as the crossing",
      c["crossing"] is None or c["crossing"]["dnll"] < FAILURE_VALUE)

print("the cache key includes the window")
import tempfile
with tempfile.TemporaryDirectory() as d:
    rep, _ = run(10.0)
    save_screen(rep, d)
    check("same window is reused",
          load_screen(d, NAMES, RES_X, window_hi=10.0) is not None)
    check("no window does not reuse a windowed screen",
          load_screen(d, NAMES, RES_X, window_hi=None) is None)
    check("a different window does not either",
          load_screen(d, NAMES, RES_X, window_hi=5.0) is None)

if failures:
    print(f"\n{len(failures)} failure(s): {failures}")
    sys.exit(1)
print("\nall passed")

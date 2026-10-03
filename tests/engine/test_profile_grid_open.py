"""The opt-in opening-grid controls of the parallel profile.

Run from the repository root:
    python tests/engine/test_profile_grid_open.py

Three controls, all inert at their defaults: ``screen`` ends a side's grid at the
slice screen's first crossing, ``open_decades`` caps the Wald opening span for a
side with no crossing, and ``grid_spacing="geometric"`` packs points toward the
optimum. What matters is that the defaults place the historical grid exactly and
that a huge Wald SE no longer sets the step once a crossing is known.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from pyantigen.engine.Optimize import _profile_grid_for                     # noqa: E402

failures = []


def check(name, cond, detail=""):
    if cond:
        print(f"  ok    {name}")
    else:
        print(f"  FAIL  {name} {detail}")
        failures.append(name)


P_OPT = 0.0                      # log10 space: x0 = 1
BOUNDS = [(-3.0, 3.0)]
HUGE_SE = 1.5                    # decades; 4 SE = 6 decades, clipped to the box


def grid(se=HUGE_SE, **kw):
    return _profile_grid_for(0, np.array([P_OPT]), BOUNDS, ["log10"],
                             np.array([se]), 3, 2.0, 4.0,
                             param_name="k", **kw)


def screen(lower=None, upper=None, thr=1.9207):
    def side(x):
        if x is None:
            return {"state": "open", "points": []}
        return {"state": "crossed",
                "points": [{"x": x / 2, "dnll": 0.1}, {"x": x, "dnll": thr + 5}]}
    return {"threshold": thr,
            "parameters": {"k": {"lower": side(lower), "upper": side(upper)}}}


print("defaults are the historical grid")
left, right, _b, _l = grid()
check("left is linspace to the bound",
      np.array_equal(left, np.linspace(0, -3, 4)[1:]), left)
check("right is linspace to the bound",
      np.array_equal(right, np.linspace(0, 3, 4)[1:]), right)

print("open_decades caps a Wald span")
left, right, _b, _l = grid(open_decades=0.6)
check("right ends at 0.6 decade", np.isclose(right[-1], 0.6), right)
check("left ends at -0.6 decade", np.isclose(left[-1], -0.6), left)

print("slice crossing ends the grid there, whatever the SE says")
left, right, _b, _l = grid(screen=screen(lower=-0.3, upper=0.45))
check("right ends at the upper crossing", np.isclose(right[-1], 0.45), right)
check("left ends at the lower crossing", np.isclose(left[-1], -0.3), left)
check("crossing wins over open_decades",
      np.isclose(grid(screen=screen(upper=0.9), open_decades=0.2)[1][-1], 0.9))

print("a side with no crossing falls back, capped")
left, right, _b, _l = grid(screen=screen(lower=-0.3, upper=None),
                           open_decades=0.5)
check("open upper side uses the capped Wald span",
      np.isclose(right[-1], 0.5), right)
check("crossed lower side still uses its crossing",
      np.isclose(left[-1], -0.3), left)

print("a crossing past the bound is clipped to the bound")
check("clipped", np.isclose(grid(screen=screen(upper=5.0))[1][-1], 3.0))

print("geometric spacing")
_l, right, _b, _x = grid(screen=screen(upper=0.8), grid_spacing="geometric")
check("halving toward the optimum", np.allclose(right, [0.2, 0.4, 0.8]), right)

print("a missing or malformed screen is ignored")
check("None screen", np.isclose(grid(screen=None)[1][-1], 3.0))
check("empty screen", np.isclose(grid(screen={})[1][-1], 3.0))

if failures:
    print(f"\n{len(failures)} failure(s): {failures}")
    sys.exit(1)
print("\nall passed")

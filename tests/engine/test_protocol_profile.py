"""run_protocol_profile: the walk/fill protocol on the Nelder-Mead worker's job and result dicts (a stand-in worker)."""
import numpy as np

from pyantigen.engine import Optimize as O

NAMES = ["a", "b"]
RES_X = np.array([0.0, 0.0])          # both log10 parameters at 1
CURV = {"a": 2.0, "b": 8.0}          # dnll = CURV * v^2 on the optimizer axis


def fake_batch(jobs, **_):
    out = []
    for j in jobs:
        v = j["x_fixed"] - RES_X[j["param_idx"]]
        res = dict(j)
        res.update({"nll": CURV[j["param_name"]] * v * v + 10.0, "status": "ok", "converged": True, "interrupted": False,
                    "n_evals": 30, "wall_s": 1.0, "nuisance_x": list(j["x_start"]), "nfev": 30})
        out.append(res)
    return out


def test_protocol_profile_returns_traces_and_finds_both_crossings():
    traces, anchor, where, conv = O.run_protocol_profile(
        fake_batch, RES_X, 10.0, NAMES, [(-3.0, 3.0), (-3.0, 3.0)], ["log10", "log10"], "Nelder-Mead",
        {"options": {"maxfev": 200}}, np.array([0.1, 0.05]), None, 1.9207, n_workers=4)
    assert set(traces) == set(NAMES) and anchor == 0.0 and where is None
    assert conv["protocol"] == "walk" and not conv["incomplete"]
    for n in NAMES:
        xs, ys = traces[n]
        assert ys.min() == 0.0 and ys.max() > 1.9207 and (xs > 0).all()       # linear values
        assert conv["reach"][n]["lower"]["state"] == "crossed" and conv["reach"][n]["upper"]["state"] == "crossed"

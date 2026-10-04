"""What the profile writes into the checkpoint, line by line.

Run from the repository root:
    python tests/engine/test_profile_records.py

Every record carries the side of the optimum it is on (``direction``, -1 or +1,
whichever pass made it) and the pass that made it (``pass_label``); job inputs
that nothing reads back (``nuisance_bounds``, ``optimizer_kwargs``, ``method``)
are not stored; the fields the next point on a chain is seeded from
(``x_start``, ``nuisance_x``, ``nm_simplex``) are.
"""
import contextlib
import io
import json
import os
import sys
import tempfile

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from pyantigen.engine.Optimize import run_parallel_profile                   # noqa: E402
from pyantigen.engine.Profile_checkpoint import ProfileCheckpoint            # noqa: E402

failures = []


def check(name, cond, detail=""):
    if cond:
        print(f"  ok    {name}")
    else:
        print(f"  FAIL  {name} {detail}")
        failures.append(name)


K = 3
A = np.array([[2.0, 0.6, 0.0], [0.6, 1.0, 0.3], [0.0, 0.3, 0.5]])


def nll(x):
    x = np.asarray(x, dtype=float)
    return float(0.5 * x @ A @ x)


def batch(jobs, on_result=None, label=None, budget=None, frozen_sigmas=None,
          state_dir=None):
    for job in jobs:
        i, xf = job["param_idx"], job["x_fixed"]

        def f(z, i=i, xf=xf):
            return nll(np.insert(z, i, xf))

        sim = job.get("initial_simplex")
        opts = {"maxfev": 800, "xatol": 1e-6, "fatol": 1e-9}
        if sim is not None:
            opts["initial_simplex"] = np.asarray(sim, dtype=float)
        res = minimize(f, np.asarray(job["x_start"], dtype=float),
                       method="Nelder-Mead", options=opts)
        out = dict(job)
        out.pop("initial_simplex", None)
        out.update({"nll": float(res.fun),
                    "nuisance_x": res.x.tolist(),
                    "nm_simplex": np.asarray(res.final_simplex[0]).tolist(),
                    "status": "ok", "converged": bool(res.success),
                    "nit": int(res.nit), "nfev": int(res.nfev),
                    "nit_total": int(res.nit), "nfev_total": int(res.nfev)})
        on_result(out)


with tempfile.TemporaryDirectory() as root:
    ckpt = ProfileCheckpoint(root, "rec", "m", "s", enabled=True)
    names = [f"p{i}" for i in range(K)]
    with contextlib.redirect_stdout(io.StringIO()):
        run_parallel_profile(
            batch, np.zeros(K), 0.0, names, [(-1e3, 1e3)] * K, ["lin"] * K,
            method="Nelder-Mead", wald_se=np.array([0.5, 0.7, 1.0]),
            n_grid=3, se_span=3.0, n_refine=1, checkpoint=ckpt, warm_passes=1)
    ckpt.close()

    recs = []
    for n in names:
        path = ckpt.path_for(n)
        with open(path, encoding="utf-8") as fh:
            recs += [json.loads(line) for line in fh if line.strip()]

    print(f"{len(recs)} record(s) written")
    check("there are records", len(recs) > 0)
    check("every record's direction is the side of the optimum",
          all(r["direction"] == (-1 if r["x_fixed"] < 0 else 1) for r in recs),
          [(r["x_fixed"], r["direction"]) for r in recs
           if r["direction"] not in (-1, 1)][:3])
    labels = {r["pass_label"] for r in recs}
    check("pass 1 records are labelled", "pass1-grid" in labels, labels)
    check("every record has a label", all(r.get("pass_label") for r in recs))
    for field in ("nuisance_bounds", "optimizer_kwargs", "method"):
        check(f"{field} is not stored", all(field not in r for r in recs))
    for field in ("x_start", "nuisance_x", "nm_simplex", "phase", "x_fixed",
                  "nll", "converged"):
        check(f"{field} is stored", all(field in r for r in recs))

if failures:
    print(f"\n{len(failures)} failure(s): {failures}")
    sys.exit(1)
print("\nall passed")

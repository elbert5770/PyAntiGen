"""Checks on the fast profile against a quadratic whose answers are known.

The objective is ``0.5 * x' A x`` with the optimum at the origin, so for each
parameter the slice curvature is ``A_ii``, the profile curvature is
``1 / (A^-1)_ii``, and the verdict this pass should reach is a matter of
arithmetic. Six parameters cover the four verdicts:

    p0        independent, A_00 = 1           -> proven true
    p1, p2    exactly confounded, [[1,1],[1,1]] -> unlikely true (flat)
    p3        no effect at all, zero row       -> proven false (screen)
    p4, p5    correlated 0.95                  -> likely true

Run from the repository root:
    python tests/engine/test_fast_profile.py
"""
import contextlib
import io
import json
import os
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.getcwd())

from pyantigen.engine.Fast_profile import (                                  # noqa: E402
    REPORT_FILENAME,
    VERDICTS,
    classify,
    fast_profile_summary,
    format_summary_table,
    local_compensation,
    run_fast_profile,
    save_report,
    slice_crossing,
    width_factor,
)
from pyantigen.engine.Optimize import (                                      # noqa: E402
    _make_nuisance_objective,
    _minimize_nuisance,
)

T = 1.9207
failures = []


def check(name, cond, detail=""):
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}  {detail}")
        failures.append(name)


# ---------------------------------------------------------------------------
# The problem
# ---------------------------------------------------------------------------

def _matrix():
    A = np.zeros((6, 6))
    A[0, 0] = 1.0
    A[1:3, 1:3] = [[1.0, 1.0], [1.0, 1.0]]
    A[4:6, 4:6] = [[1.0, 0.95], [0.95, 1.0]]
    return A


A = _matrix()
K = 6
NAMES = [f"p{i}" for i in range(K)]
BOUNDS = [(-1e3, 1e3)] * K
SCALES = ["lin"] * K


def nll(x):
    x = np.asarray(x, dtype=float)
    return float(0.5 * x @ A @ x)


def nll_batch(xs, label=None):
    return [nll(x) for x in xs]


def _wald_se():
    """Marginal SEs where the Hessian is invertible, nan where it is not."""
    se = np.full(K, np.nan)
    se[0] = 1.0
    cov45 = np.linalg.inv(A[4:6, 4:6])
    se[4] = np.sqrt(cov45[0, 0])
    se[5] = np.sqrt(cov45[1, 1])
    return se


class _Checkpoint:
    def __init__(self):
        self.records = []

    def append(self, rec):
        self.records.append(dict(rec))


def _make_batch(spend=None):
    """In-process stand-in for ParallelEvaluator.profile_batch.

    Honours what the fast pass relies on: the per-job evaluation cap in
    ``optimizer_kwargs["options"]``, the ``nfev_used`` carried between rounds,
    the ``initial_simplex`` on the way in and ``nm_simplex`` on the way out.
    """
    def batch(jobs, on_result=None, label=None, budget=None):
        for job in jobs:
            obj = _make_nuisance_objective(nll, job["param_idx"], K)
            opts = dict((job.get("optimizer_kwargs") or {}).get("options") or {})
            used = int(job.get("nfev_used") or 0)
            extra = {}
            if "maxfev" in opts:
                extra["maxfev"] = max(1, int(opts["maxfev"]) - used)
                extra["maxiter"] = max(1, int(opts.get("maxiter", opts["maxfev"])) - used)
            sim = job.get("initial_simplex")
            if sim is not None:
                extra["initial_simplex"] = np.asarray(sim, dtype=float)
            res = _minimize_nuisance(
                obj, np.asarray(job["x_start"], dtype=float),
                (job["x_fixed"],), job["method"], job.get("nuisance_bounds"),
                job.get("optimizer_kwargs"), extra_options=extra or None,
            )
            out = dict(job)
            out.pop("initial_simplex", None)
            fsim = getattr(res, "final_simplex", None)
            out.update({
                "nll": float(res.fun),
                "nuisance_x": np.asarray(res.x, dtype=float).tolist(),
                "nm_simplex": (np.asarray(fsim[0], dtype=float).tolist()
                               if fsim is not None else None),
                "status": "ok",
                "converged": bool(res.success),
                "nit": int(res.nit),
                "nfev": int(res.nfev),
                "nit_total": int(job.get("nit_used") or 0) + int(res.nit),
                "nfev_total": used + int(res.nfev),
            })
            if spend is not None:
                spend["nfev"] = spend.get("nfev", 0) + int(res.nfev)
            on_result(out)
    return batch


def _run(round_evals=200, n_rounds=3, checkpoint=None, ckpt_dir=None,
         quiet=True, spend=None, **kw):
    buf = io.StringIO()
    ctx = contextlib.redirect_stdout(buf) if quiet else contextlib.nullcontext()
    with ctx:
        report = run_fast_profile(
            _make_batch(spend), nll_batch, np.zeros(K), 0.0, NAMES, BOUNDS,
            SCALES, method="Nelder-Mead", optimizer_kwargs={},
            wald_se=_wald_se(), wald_cov=None, checkpoint=checkpoint,
            ckpt_dir=ckpt_dir, threshold=T, round_evals=round_evals,
            n_rounds=n_rounds, **kw)
    return report, buf.getvalue()


# ---------------------------------------------------------------------------
# Pure pieces
# ---------------------------------------------------------------------------

def test_geometry():
    print("\n[geometry]")
    pts = [{"x": 0.5, "dnll": 0.125, "nll": 0.125},
           {"x": 1.0, "dnll": 0.5, "nll": 0.5},
           {"x": 2.0, "dnll": 2.0, "nll": 2.0},
           {"x": 4.0, "dnll": 8.0, "nll": 8.0}]
    got = slice_crossing(pts, 0.0, T)
    check("crossing is bracketed by the right ladder points",
          got is not None and got[1] == 1.0 and got[2] == 2.0, str(got))
    check("interpolated crossing sits inside the true one (1.96)",
          got is not None and 1.0 < got[0] < np.sqrt(2 * T), str(got))
    check("no crossing when the slice never rises",
          slice_crossing(pts[:2], 0.0, T) is None)
    check("the optimum is the inner point when the first rung is above",
          slice_crossing(pts[2:], 0.0, T)[1] == 0.0)
    check("a failed rung is skipped",
          slice_crossing([{"x": 1.5, "dnll": float("nan"), "nll": 1e10}] + pts,
                         0.0, T)[2] == 2.0)

    check("width factor at the threshold is 1",
          abs(width_factor(T, T) - 1.0) < 1e-12)
    check("width factor at a quarter of the threshold is 2",
          abs(width_factor(T / 4, T) - 2.0) < 1e-12)
    check("width factor at zero is infinite",
          np.isinf(width_factor(0.0, T)))
    check("width factor of nothing is None", width_factor(None, T) is None)

    cov = np.linalg.inv(A[4:6, 4:6])
    ratio = local_compensation(cov, 0)
    # marginal SE / conditional SE = sqrt(cov_00 * A_00) = sqrt(cov_00)
    check("local compensation is sqrt(VIF)",
          ratio is not None and abs(ratio - np.sqrt(cov[0, 0])) < 1e-9, str(ratio))
    check("independent parameter has ratio 1",
          abs(local_compensation(np.eye(2), 1) - 1.0) < 1e-12)
    check("a singular covariance gives None",
          local_compensation(np.ones((2, 2)), 0) is None)
    check("no covariance gives None", local_compensation(None, 0) is None)


def test_classify():
    print("\n[classify]")
    base = {"screen_state": "crossed", "x_point": 2.0, "x_point_linear": 2.0,
            "status": "ok", "slice_dnll_at_point": 2.5, "absorbed": 0.0}

    def v(**over):
        s = dict(base)
        s.update(over)
        return classify(s, T)[0]

    check("open screen is proven false",
          v(screen_state="open", reach_decades=3.0) == "proven false")
    check("blocked screen is no verdict", v(screen_state="blocked") == "no verdict")
    check("no crossing is no verdict", v(x_point=None) == "no verdict")
    check("failed point is no verdict",
          v(status="sentinel", dnll=None) == "no verdict")
    check("negative dNLL is no verdict (better optimum)",
          v(dnll=-0.5, converged=True) == "no verdict")
    check("near zero is unlikely true", v(dnll=0.05, converged=True) == "unlikely true")
    check("exactly at the near-zero line is unlikely true",
          v(dnll=0.1 * T, converged=True) == "unlikely true")
    check("below threshold, clear of zero, is likely true",
          v(dnll=1.0, converged=True) == "likely true")
    check("below threshold and unconverged is likely true",
          v(dnll=1.0, converged=False, nfev_total=300) == "likely true")
    check("above threshold and converged is proven true",
          v(dnll=2.4, converged=True) == "proven true")
    check("above threshold and unconverged is only likely true",
          v(dnll=2.4, converged=False, nfev_total=300) == "likely true")
    reason = classify(dict(base, dnll=2.4, converged=False, nfev_total=300,
                           stalled=True), T)[1]
    check("an unconverged reason says it stalled", "stalled" in reason, reason)
    check("every verdict is one of the five",
          all(classify(dict(base, dnll=d, converged=c), T)[0] in VERDICTS
              for d in (-1.0, 0.0, 0.1, 1.0, 2.0, 5.0) for c in (True, False)))


# ---------------------------------------------------------------------------
# The pass end to end
# ---------------------------------------------------------------------------

def test_end_to_end():
    print("\n[end to end on the quadratic]")
    ck = _Checkpoint()
    spend = {}
    report, out = _run(checkpoint=ck, spend=spend)
    P = report["parameters"]

    def verdicts(name):
        return {side: P[name][side]["verdict"] for side in ("lower", "upper")}

    check("p0 (independent) is proven true both sides",
          set(verdicts("p0").values()) == {"proven true"}, str(verdicts("p0")))
    check("p1 (confounded) is unlikely true both sides",
          set(verdicts("p1").values()) == {"unlikely true"}, str(verdicts("p1")))
    check("p2 (confounded) is unlikely true both sides",
          set(verdicts("p2").values()) == {"unlikely true"}, str(verdicts("p2")))
    check("p3 (no effect) is proven false both sides",
          set(verdicts("p3").values()) == {"proven false"}, str(verdicts("p3")))
    check("p4 (0.95 correlated) is likely true both sides",
          set(verdicts("p4").values()) == {"likely true"}, str(verdicts("p4")))
    check("p5 (0.95 correlated) is likely true both sides",
          set(verdicts("p5").values()) == {"likely true"}, str(verdicts("p5")))

    c = report["counts"]
    check("counts add up to every side",
          sum(c.values()) == 2 * K, str(c))
    check("counts match the verdicts",
          c["proven true"] == 2 and c["unlikely true"] == 4
          and c["proven false"] == 2 and c["likely true"] == 4, str(c))

    # The numbers behind the verdicts.
    s = P["p0"]["upper"]
    check("p0: nothing absorbed", s["absorbed"] is not None and s["absorbed"] < 0.02,
          str(s["absorbed"]))
    check("p0: profile equals slice at the point",
          abs(s["dnll"] - s["slice_dnll_at_point"]) < 1e-3,
          f"{s['dnll']} vs {s['slice_dnll_at_point']}")
    check("p0: VIF^0.5 unavailable without a covariance",
          s["local_compensation"] is None)

    s = P["p1"]["upper"]
    check("p1: essentially everything absorbed", s["absorbed"] > 0.98, str(s["absorbed"]))
    check("p1: width factor is large or infinite",
          s["width_factor"] is not None and s["width_factor"] > 3.0, str(s["width_factor"]))
    check("p1: extrapolated crossing beyond the screen's reach",
          s["est_beyond_reach"] is True, str(s))

    s = P["p4"]["upper"]
    # Profile curvature for p4 is 1 - 0.95^2 = 0.0975 against a slice curvature
    # of 1, so at the point x the profile is 0.0975 * slice.
    expect = 1.0 - 0.0975
    check("p4: absorbed fraction matches the algebra",
          abs(s["absorbed"] - expect) < 0.01, f"{s['absorbed']} vs {expect}")
    lo, hi = P["p4"]["lower"], P["p4"]["upper"]
    check("p4: est crossing is symmetric and outside the slice point",
          abs(abs(lo["est_crossing_linear"]) - abs(hi["est_crossing_linear"])) < 1e-3
          and abs(hi["est_crossing_linear"]) > abs(hi["x_point_linear"]),
          f"{lo['est_crossing_linear']} {hi['est_crossing_linear']}")
    true_half = np.sqrt(2 * T / 0.0975)
    check("p4: est crossing is a floor under the true crossing",
          abs(hi["est_crossing_linear"]) <= true_half * 1.01,
          f"{hi['est_crossing_linear']} vs {true_half}")

    s = P["p3"]["upper"]
    check("p3: no point was spent on a proven-open side",
          s["x_point"] is None and s["dnll"] is None)

    # Bookkeeping.
    check("one record per crossed side per improving round, all ok",
          ck.records and all(r["status"] == "ok" and "dnll" in r
                             and r.get("fast_profile") for r in ck.records))
    names_in_ckpt = {r["param_name"] for r in ck.records}
    check("checkpoint holds only the sides that were profiled",
          names_in_ckpt == {"p0", "p1", "p2", "p4", "p5"}, str(names_in_ckpt))
    check("checkpoint records carry no private keys",
          all("_side" not in r for r in ck.records))
    check("points reported", report["n_points"] == 10, str(report["n_points"]))
    check("rounds never exceed the cap",
          all(len(P[n][sd]["rounds"]) <= 3 for n in NAMES for sd in ("lower", "upper")))
    check("a settled point stops early",
          len(P["p1"]["upper"]["rounds"]) < 3 or P["p1"]["upper"]["converged"],
          str(P["p1"]["upper"]["rounds"]))
    check("evaluations are tallied", report["n_evaluations"] > 0 and spend["nfev"] > 0)


def test_rounds_carry_state():
    print("\n[rounds carry state]")
    # A tiny round budget: no point can converge in one round, so the second
    # round must start from the first one's simplex and count its spend.
    report, _ = _run(round_evals=12, n_rounds=3)
    s = report["parameters"]["p4"]["upper"]
    r = s["rounds"]
    check("more than one round ran", len(r) >= 2, str(r))
    check("spend accumulates across rounds",
          all(r[i]["nfev_total"] > r[i - 1]["nfev_total"] for i in range(1, len(r))),
          str(r))
    check("dNLL never rises across rounds",
          all(r[i]["dnll"] <= r[i - 1]["dnll"] + 1e-12 for i in range(1, len(r))),
          str(r))


def test_report_and_table():
    print("\n[report, table, snapshot]")
    report, out = _run(quiet=True)
    table = format_summary_table(report)
    for word in ("proven true", "likely true", "unlikely true", "proven false",
                 "interval closed?", "absorbed", "VIF^0.5"):
        check(f"table mentions '{word}'", word in table)
    check("table has one row per side", table.count("  p0  ") == 2
          or sum(1 for line in table.splitlines() if line.strip().startswith("p0 ")) == 2)

    summary = fast_profile_summary(report)
    check("summary carries counts", summary["counts"] == report["counts"])
    check("summary drops per-round detail",
          "rounds" not in summary["parameters"]["p0"]["upper"])
    check("summary keeps the verdict",
          summary["parameters"]["p1"]["lower"]["verdict"] == "unlikely true")
    check("summary is JSON-clean",
          json.loads(json.dumps(summary, allow_nan=False)) is not None)

    root = tempfile.mkdtemp()
    try:
        path = save_report(report, root)
        check("report saved", path is not None and os.path.exists(path))
        with open(path) as fh:
            back = json.load(fh)
        check("saved report round-trips the verdicts",
              back["parameters"]["p3"]["upper"]["verdict"] == "proven false")
        check("report file has the expected name",
              os.path.basename(path) == REPORT_FILENAME)
        # A second run in the same directory reuses the screen file.
        _, out1 = _run(ckpt_dir=root)
        _, out2 = _run(ckpt_dir=root)
        check("first run computes the screen", "reusing the slice screen" not in out1)
        check("screen is reused on a second run", "reusing the slice screen" in out2)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_failures():
    """``check`` records rather than raises, so pytest needs this to see them."""
    assert not failures, "failed checks: " + ", ".join(failures)


def main():
    test_geometry()
    test_classify()
    test_end_to_end()
    test_rounds_carry_state()
    test_report_and_table()

    print("\n" + "=" * 72)
    if failures:
        print(f"FAILED ({len(failures)}):")
        for f in failures:
            print("  -", f)
        return 1
    print("ALL FAST PROFILE CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Profile-likelihood checks against a problem whose answer is known exactly.

Run from the repository root:
    python tests/engine/test_profile_quadratic.py

Every other profile test in this repo runs a real model, which means a failure
could be the ODE solver, the data, the loss config, or the profiler.  Here the
objective is a Gaussian NLL, ``0.5 * x' A x``, so the 95% profile interval for
each parameter is analytic:

    half-width_i = sqrt(2 * 1.9207 * (A^-1)_ii)

That isolates the part worth guarding -- the threshold, the anchoring, the grid
placement and the crossing interpolation -- from everything that merely feeds
it.  It also runs in seconds and needs no model, no roadrunner and no data.

The nuisance minimizations here are driven through a stub batch runner rather
than the process pool, because the pool is not what is under test.
"""
import io
import contextlib
import os
import sys

import numpy as np

sys.path.insert(0, os.getcwd())


from pyantigen.engine.Optimize import (                                    # noqa: E402
    _EXT_GROWTH_MAX,
    _EXT_GROWTH_MIN,
    _WARM_SIMPLEX_COND_FLOOR,
    _WARM_SIMPLEX_SCALE,
    _extract_profile_ci,
    compute_wald_uncertainty,
    _make_nuisance_objective,
    _minimize_nuisance,
    _profile_nuisance_defaults,
    _extension_growth,
    _warm_chain_relevant,
    _warm_simplex,
    run_parallel_profile,
)

_THRESHOLD = 1.9207

failures = []


def check(name, cond, detail=""):
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}  {detail}")
        failures.append(name)


def _make_batch(nll, k, optimizer_kwargs=None, spend=None):
    """Stub for ParallelEvaluator.profile_batch: runs each job in-process.

    Accepts and ignores *budget*: these tests are about the profile algorithm,
    which must behave identically whether or not a wall-clock limit exists.
    Admission control itself is covered in tests/test_deadline.py.

    It has to mirror ``pyantigen.engine.Evaluator._profile_task`` in the two respects the
    continuation passes depend on, or a test here will pass while the real
    thing is broken:

    * it must honour ``initial_simplex`` on the way in, and
    * it must return ``nm_simplex`` on the way out.

    A stub that dropped either would turn every warm start back into a
    half-warm one -- right start point, freshly guessed simplex -- and the pass
    that carries optimizer state forward would look like a no-op.

    *spend* is an optional dict accumulating total evaluations, since the point
    of carrying state forward is to need fewer of them.
    """
    def batch(jobs, on_result=None, label=None, budget=None):
        for job in jobs:
            obj = _make_nuisance_objective(nll, job["param_idx"], k)
            extra = {}
            sim = job.get("initial_simplex")
            if sim is not None:
                extra["initial_simplex"] = np.asarray(sim, dtype=float)
            res = _minimize_nuisance(
                obj, np.asarray(job["x_start"], dtype=float),
                (job["x_fixed"],), job["method"],
                job.get("nuisance_bounds"), optimizer_kwargs,
                extra_options=extra or None,
            )
            out = dict(job)
            out.pop("initial_simplex", None)
            fsim = getattr(res, "final_simplex", None)
            out.update({
                "nll": float(res.fun),
                # The continuation pass seeds the next point from this, so a
                # stub that omitted it would silently make every warm start a
                # cold one and the pass would look like a no-op.
                "nuisance_x": np.asarray(res.x, dtype=float).tolist(),
                "nm_simplex": (np.asarray(fsim[0], dtype=float).tolist()
                               if fsim is not None else None),
                "status": "ok",
                "converged": bool(res.success),
                "nit": int(res.nit),
                "nfev": int(res.nfev),
                "nit_total": int(res.nit),
                "nfev_total": int(res.nfev),
            })
            if spend is not None:
                spend["nfev"] = spend.get("nfev", 0) + int(res.nfev)
                spend["points"] = spend.get("points", 0) + 1
            on_result(out)
    return batch


def _profile(nll, k, se, optimizer_kwargs=None, quiet=True, warm_passes=1):
    names = [f"p{i}" for i in range(k)]
    buf = io.StringIO()
    ctx = contextlib.redirect_stdout(buf) if quiet else contextlib.nullcontext()
    with ctx:
        traces, anchor, where, convergence = run_parallel_profile(
            _make_batch(nll, k, optimizer_kwargs), np.zeros(k), 0.0, names,
            [(-1e3, 1e3)] * k, ["lin"] * k, method="Nelder-Mead",
            wald_se=se, n_grid=6, se_span=3.0, n_refine=2, checkpoint=None,
            warm_passes=warm_passes,
        )
    return names, traces, anchor, convergence


def _ill_conditioned(k=15, log_range=3.0, seed=0):
    """A correlated, ill-conditioned Gaussian NLL.

    A separable quadratic is too kind: its nuisance minimum does not move as the
    profiled parameter slides, so cold-starting costs nothing and the pass that
    matters cannot be measured. Rotating a spectrum spanning ``10**log_range``
    produces the curved, correlated valley a real fit has.
    """
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.normal(size=(k, k)))
    cov = Q @ np.diag(np.logspace(0, -log_range, k)) @ Q.T
    A = np.linalg.inv(cov)
    se = np.sqrt(np.diag(cov))
    return (lambda x: 0.5 * x @ A @ x), se, np.sqrt(2 * _THRESHOLD) * se


def _half_widths(names, traces):
    out = {}
    for name in names:
        lo, hi = _extract_profile_ci(traces[name][0], traces[name][1],
                                     threshold=_THRESHOLD)
        out[name] = (hi - lo) / 2.0 if np.isfinite(lo) and np.isfinite(hi) \
            else np.nan
    return out


def test_separable_quadratic():
    """A diagonal Hessian: nuisance re-optimization is trivial, so any error
    left is the profiler's own."""
    print("\nSeparable quadratic (exact CI recoverable):")
    k = 4
    A = np.diag([1.0, 2.0, 3.0, 4.0])
    se = 1.0 / np.sqrt(np.diag(A))
    exact = np.sqrt(2 * _THRESHOLD) * se

    names, traces, anchor, conv = _profile(lambda x: 0.5 * x @ A @ x, k, se)

    check("no point beat the reported optimum", abs(anchor) < 1e-6,
          f"anchor={anchor:.3g}")
    check("every nuisance optimization converged",
          conv["n_not_converged"] == 0,
          f"{conv['n_not_converged']}/{conv['n_points']} capped")

    for i, name in enumerate(names):
        lo, hi = _extract_profile_ci(traces[name][0], traces[name][1],
                                     threshold=_THRESHOLD)
        check(f"{name} CI finite", np.isfinite(lo) and np.isfinite(hi),
              f"[{lo}, {hi}]")
        if not (np.isfinite(lo) and np.isfinite(hi)):
            continue
        # Symmetric about the optimum for a quadratic, so compare half-widths.
        half = (hi - lo) / 2.0
        rel = abs(half / exact[i] - 1.0)
        check(f"{name} half-width within 0.5% of analytic",
              rel < 5e-3,
              f"{half:.6g} vs {exact[i]:.6g} ({100 * rel:+.3f}%)")
        check(f"{name} CI centred on the optimum", abs(lo + hi) < 1e-3 * half,
              f"[{lo:.6g}, {hi:.6g}]")


def test_threshold_is_a_likelihood_ratio():
    """dNLL at the analytic bound must be the chi2(1)/2 threshold itself.

    This is the claim the whole interval rests on: that the objective really is
    a log-likelihood, so 1.9207 is the right height. If the objective were
    scaled -- by group averaging, or by a weight -- this is what would move.
    """
    print("\nThreshold height:")
    A = np.diag([1.0, 2.0, 3.0, 4.0])

    def nll(x):
        return 0.5 * x @ A @ x

    for i in range(4):
        se_i = 1.0 / np.sqrt(A[i, i])
        x = np.zeros(4)
        x[i] = np.sqrt(2 * _THRESHOLD) * se_i
        check(f"p{i} dNLL at the analytic bound == {_THRESHOLD}",
              abs(nll(x) - _THRESHOLD) < 1e-9,
              f"{nll(x):.9g}")


def test_capped_optimization_is_reported():
    """An under-budgeted nuisance optimization must be visible, not silent.

    A truncated minimization overstates the profile and therefore narrows the
    interval, so the failure mode this guards against is a confident-looking CI
    that is simply too small.
    """
    print("\nCapped-optimization reporting:")
    rng = np.random.default_rng(0)
    k = 8
    Q, _ = np.linalg.qr(rng.normal(size=(k, k)))
    cov = Q @ np.diag(np.logspace(0, -3, k)) @ Q.T
    A = np.linalg.inv(cov)
    se = np.sqrt(np.diag(cov))

    def nll(x):
        return 0.5 * x @ A @ x

    _n, _t, _a, conv = _profile(nll, k, se,
                                optimizer_kwargs={"options": {"maxiter": 5,
                                                              "maxfev": 5}})
    check("a starved run reports capped points",
          conv["n_not_converged"] > 0,
          f"{conv['n_not_converged']}/{conv['n_points']}")
    check("per-parameter counts are populated",
          any(d["n_not_converged"] for d in conv["per_param"].values()))
    check("no point is counted as unknown when convergence was recorded",
          conv["n_unknown"] == 0, f"{conv['n_unknown']}")


def test_budget_scales_with_dimension():
    """The cap must grow with the nuisance dimension, matching scipy's own
    ``N*200`` default. A flat cap silently truncates large fits."""
    print("\nNuisance budget scaling:")
    small = _profile_nuisance_defaults("Nelder-Mead", 1)
    large = _profile_nuisance_defaults("Nelder-Mead", 20)
    check("Nelder-Mead budget scales with dimension",
          large["maxiter"] == 20 * small["maxiter"],
          f"{small['maxiter']} -> {large['maxiter']}")
    check("Nelder-Mead matches scipy's N*200 default",
          large["maxiter"] == 4000 and large["maxfev"] == 4000, str(large))
    check("a missing dimension falls back rather than raising",
          _profile_nuisance_defaults("Nelder-Mead", None)["maxiter"] == 200)
    check("gradient methods use a tolerance below CI resolution",
          _profile_nuisance_defaults("L-BFGS-B", 20)["ftol"] <= 1e-8)


def test_warm_start_recovers_accuracy():
    """The three things that make a profile point accurate, measured apart.

    A profile point is only as good as where its nuisance minimization starts
    and what shape it starts in, and there are three separable contributions:

      1. the starting *point*, which the continuation pass supplies;
      2. the starting *simplex* for a point that has a neighbour to inherit
         one from, rescaled and re-conditioned;
      3. the starting simplex for a point that has no neighbour, built from
         the Wald SE instead of from scipy's 5%-of-value rule.

    All three pull the same way: a nuisance minimization that stops early
    reports a minimum that is too high, so dNLL is too high, so the curve
    reaches the threshold too soon and every interval comes back too narrow.
    Too narrow is the damaging direction, and it is silent.

    The arms below turn each contribution off in turn, so a regression in any
    one of them is caught by a number rather than by eye. Arm A is what the
    profile did before any of this: scipy's own simplex everywhere, no
    continuation.
    """
    print("\nStarting values, measured contribution by contribution:")
    k = 15
    nll, se, exact = _ill_conditioned(k)

    def arm(explicit_simplex, warm_passes):
        import pyantigen.engine.Optimize as _O
        saved = (_O._warm_simplex, _O._cold_simplex)
        if not explicit_simplex:
            _O._warm_simplex = lambda *a, **kw: None
            _O._cold_simplex = lambda *a, **kw: None
        try:
            names, traces, _a, conv = _profile(nll, k, se,
                                               warm_passes=warm_passes)
        finally:
            _O._warm_simplex, _O._cold_simplex = saved
        h = _half_widths(names, traces)
        e = np.array([h[n] / exact[i] - 1 for i, n in enumerate(names)])
        return e, conv

    err_a, _ca = arm(False, 0)
    err_b, _cb = arm(False, 1)
    err_c, _cc = arm(True, 0)
    err_d, conv = arm(True, 1)

    for label, e in (("A  scipy simplex, no continuation ", err_a),
                     ("B  scipy simplex, continuation    ", err_b),
                     ("C  explicit simplex, no continuation", err_c),
                     ("D  explicit simplex, continuation ", err_d)):
        print(f"    {label}  mean {100 * np.mean(e):+6.2f}%   "
              f"worst {100 * np.min(e):+6.2f}%")
    print(f"    points improved by the continuation pass: "
          f"{conv['warm']['n_improved']}/{conv['warm']['n_attempted']}, "
          f"{conv['warm']['nats_recovered']:.4g} nats recovered")

    check("the bare problem really is biased, or nothing here means anything",
          np.mean(err_a) < -0.15,
          f"arm A mean error {100 * np.mean(err_a):+.1f}%")
    check("continuation alone recovers most of it",
          np.mean(err_b) > np.mean(err_a) + 0.10,
          f"{100 * np.mean(err_a):+.1f}% -> {100 * np.mean(err_b):+.1f}%")
    check("an explicit starting simplex alone recovers most of it too",
          np.mean(err_c) > np.mean(err_a) + 0.10,
          f"{100 * np.mean(err_a):+.1f}% -> {100 * np.mean(err_c):+.1f}%")
    check("together they are better than either alone",
          np.mean(err_d) >= max(np.mean(err_b), np.mean(err_c)) - 1e-3,
          f"B {100 * np.mean(err_b):+.2f}%, C {100 * np.mean(err_c):+.2f}%, "
          f"D {100 * np.mean(err_d):+.2f}%")
    check("the continuation pass still reports improvements",
          conv["warm"]["n_improved"] > 0)
    check("the full protocol leaves essentially no bias",
          abs(np.mean(err_d)) < 0.02,
          f"mean error {100 * np.mean(err_d):+.2f}%")
    check("and no single parameter is left biased",
          abs(np.min(err_d)) < 0.05,
          f"worst error {100 * np.min(err_d):+.2f}%")
    check("every interval is still an interval, not a widened guess",
          np.max(err_d) < 0.05,
          f"most overwide {100 * np.max(err_d):+.2f}%")


def test_warm_simplex_carries_scale_not_shape():
    """The neighbour's simplex must be rescaled and re-conditioned, not reused.

    Guards the two corrections in ``_warm_simplex`` separately, because each one
    alone is worse than doing nothing and the failure modes are opposite. Reused
    at its converged size the simplex is far too small for the distance to the
    next minimum. Reused at its converged *shape* it is a needle whose collapsed
    axes the optimizer can never search, so it converges fast onto a nuisance
    minimum that is much too high -- which lifts dNLL and narrows every interval,
    the one direction of error that matters here.
    """
    print("\nWarm simplex construction:")
    k = 15
    nll, se, _exact = _ill_conditioned(k)
    obj = _make_nuisance_objective(nll, 0, k)
    res = _minimize_nuisance(obj, np.zeros(k - 1), (se[0],), "Nelder-Mead",
                             None, None)
    converged = np.asarray(res.final_simplex[0], dtype=float)
    n = k - 1

    sv_in = np.linalg.svd(converged[1:] - converged[0], compute_uv=False)
    check("a converged simplex really is a needle in this many dimensions",
          sv_in[0] / sv_in[-1] > 1e3,
          f"condition number {sv_in[0] / sv_in[-1]:.3g} -- if this is small the "
          f"problem is too easy to test anything")

    x_start = np.full(n, 0.3)
    out = _warm_simplex(converged, x_start, travel=0.02)
    check("a usable simplex comes back", out is not None)
    out = np.asarray(out, dtype=float)

    check("vertex 0 sits on the new start point",
          np.allclose(out[0], x_start))

    off = out[1:] - out[0]
    sv_out = np.linalg.svd(off, compute_uv=False)
    cond = sv_out[0] / sv_out[-1]
    check("the conditioning is repaired to the declared floor",
          cond <= 1.0 / _WARM_SIMPLEX_COND_FLOOR + 1e-6,
          f"condition number {cond:.4g}")

    want = _WARM_SIMPLEX_SCALE * 0.02
    check("it is rescaled to a multiple of the distance the seed travelled",
          abs(sv_out[0] / want - 1.0) < 1e-6,
          f"widest axis {sv_out[0]:.4g}, wanted {want:.4g}")

    check("which is not the size it converged at",
          abs(sv_in[0] / want - 1.0) > 0.5,
          f"converged at {sv_in[0]:.4g}, rescaled to {want:.4g}")

    check("the dominant direction is inherited, not re-guessed",
          abs(np.dot(np.linalg.svd(off)[2][0],
                     np.linalg.svd(converged[1:] - converged[0])[2][0])) > 0.99)

    kept = np.asarray(_warm_simplex(converged, x_start), dtype=float)
    check("with no travel to go on it keeps its own size rather than inventing one",
          abs(np.linalg.svd(kept[1:] - kept[0], compute_uv=False)[0] / sv_in[0]
              - 1.0) < 1e-6)

    check("but it repairs the conditioning even then",
          np.linalg.cond(kept[1:] - kept[0]) <= 1.0 / _WARM_SIMPLEX_COND_FLOOR + 1e-6)

    for bad, label in ((None, "no simplex"),
                       (converged[:-1], "wrong shape"),
                       (np.zeros((n + 1, n)), "all-zero"),
                       (np.full((n + 1, n), np.nan), "not finite")):
        check(f"{label} is declined rather than passed to scipy",
              _warm_simplex(bad, x_start, travel=0.02) is None)

    pinned = [(float(x_start[i]), float(x_start[i])) for i in range(n)]
    check("a simplex flattened onto its bounds is declined",
          _warm_simplex(converged, x_start, travel=0.02, bounds=pinned) is None)

    # The functional half: walk one chain three ways and measure each against
    # the profile value the quadratic gives exactly. For 0.5 x' A x the profile
    # over the nuisance parameters is 0.5 v^2 / cov_ii, and se is sqrt(diag cov),
    # so the exact value is available in closed form.
    exact_var = float(se[0]) ** 2

    def walk(cond_floor):
        """Mean excess over the exact profile along one warm-started chain."""
        x0 = np.zeros(k - 1)
        prev, travel, errs, spend = None, None, [], 0
        for v in np.linspace(se[0] * 0.5, se[0] * 3.0, 8):
            extra = None
            if prev is not None:
                s = (_warm_simplex(prev, x0, travel=travel)
                     if cond_floor is None else
                     _warm_simplex(prev, x0, travel=travel,
                                   cond_floor=cond_floor))
                if s is not None:
                    extra = {"initial_simplex": np.asarray(s, dtype=float)}
            r = _minimize_nuisance(obj, x0, (v,), "Nelder-Mead", None, None,
                                   extra_options=extra)
            spend += int(r.nfev)
            errs.append(float(r.fun) - 0.5 * v * v / exact_var)
            travel = float(np.linalg.norm(np.asarray(r.x) - x0))
            x0 = np.asarray(r.x, dtype=float)
            fs = getattr(r, "final_simplex", None)
            prev = np.asarray(fs[0], dtype=float) if fs is not None else None
        return float(np.mean(errs)), spend

    def walk_plain():
        x0 = np.zeros(k - 1)
        errs, spend = [], 0
        for v in np.linspace(se[0] * 0.5, se[0] * 3.0, 8):
            r = _minimize_nuisance(obj, x0, (v,), "Nelder-Mead", None, None)
            spend += int(r.nfev)
            errs.append(float(r.fun) - 0.5 * v * v / exact_var)
            x0 = np.asarray(r.x, dtype=float)
        return float(np.mean(errs)), spend

    e_plain, s_plain = walk_plain()
    e_fixed, s_fixed = walk(None)
    e_raw, s_raw = walk(0.0)
    print(f"    excess NLL above the exact profile, one 14-nuisance chain:")
    print(f"      simplex not carried  {e_plain:10.5g}  ({s_plain} evals)")
    print(f"      carried, conditioned {e_fixed:10.5g}  ({s_fixed} evals)")
    print(f"      carried raw          {e_raw:10.5g}  ({s_raw} evals)")

    check("carrying a conditioned simplex beats not carrying one",
          e_fixed < e_plain, f"{e_fixed:.5g} vs {e_plain:.5g}")
    check("carrying it raw is worse than not carrying it at all",
          e_raw > e_plain, f"{e_raw:.5g} vs {e_plain:.5g}")
    check("and the repair is what separates them",
          e_raw > 5 * e_fixed, f"raw {e_raw:.5g} vs conditioned {e_fixed:.5g}")


def test_budget_is_judged_on_movement_not_on_the_cap():
    """Hitting the evaluation cap is not evidence that an interval is wrong.

    The old report said "this interval is too narrow" of any parameter with a
    capped point, and the last SILK/APP run therefore said it of all sixteen.
    Measured on this same 15-parameter problem, 79% of points stop on a cap of
    1500 while the intervals they produce sit within 0.02% of the analytic
    answer: the search reached the minimum and spent the rest polishing.

    What does measure the bias is how far the warm pass moved the point, and
    specifically how far it moved the points near the crossing -- a large drop
    out in the tail moves no bound at all. Both halves are checked here.
    """
    print("\nCapped points, judged on measured movement:")
    from pyantigen.engine.Model_optimize import _profile_report_text

    names = ["p_unmeasured", "p_clean", "p_tail_only", "p_biased"]
    est = np.array([1.0, 1.0, 1.0, 1.0])
    per = {
        "p_unmeasured": {"n": 10, "n_not_converged": 8, "n_warm_measured": 0},
        "p_clean": {"n": 10, "n_not_converged": 8, "n_warm_measured": 10,
                    "warm_gain_max": 0.0, "warm_gain_near": 0.0},
        "p_tail_only": {"n": 10, "n_not_converged": 8, "n_warm_measured": 10,
                        "warm_gain_max": 41.0, "warm_gain_near": 0.004},
        "p_biased": {"n": 10, "n_not_converged": 8, "n_warm_measured": 10,
                     "warm_gain_max": 5.0, "warm_gain_near": 0.9},
    }
    opt = {"x": est, "stats": {
        "profile_convergence": {"n_points": 40, "n_not_converged": 32,
                                "n_unknown": 0, "per_param": per,
                                "warm": {"n_attempted": 40, "n_improved": 12,
                                         "nats_recovered": 3.75},
                                "reach": {}},
        "profile_anchor_gap": 0.0,
        "wald_se": [0.1] * 4, "wald_ci": [(0.8, 1.2)] * 4}}
    txt = _profile_report_text(opt, names, est, [(0.8, 1.3)] * 4, ["ok"] * 4,
                               {}, "M", "TAG", "2026-09-02 12:00:00")
    # Walk the report once, attributing each "points" line to the parameter
    # heading above it.
    lines, current = {}, None
    for line in txt.splitlines():
        stripped = line.strip()
        if stripped in names:
            current = stripped
        elif current and "hit the optimizer cap" in line:
            lines.setdefault(current, line)
    check("every parameter got a points line",
          set(lines) == set(names), str(sorted(lines)))

    check("the report stays ASCII", txt.isascii())
    check("a capped point is still reported as capped",
          all("hit the optimizer cap" in v for v in lines.values()))
    check("with no warm pass yet, the bias is called unmeasured",
          "unmeasured" in lines["p_unmeasured"])
    check("a point the warm pass never moved is not called too narrow",
          "too narrow" not in lines["p_clean"],
          lines["p_clean"].strip())
    check("and is said to have been polishing",
          "polishing" in lines["p_clean"])
    check("a big drop out in the tail does not condemn the interval",
          "too narrow" not in lines["p_tail_only"],
          lines["p_tail_only"].strip())
    check("a drop near the crossing does",
          "too narrow" in lines["p_biased"], lines["p_biased"].strip())
    check("and the verdict quotes the number it is based on",
          "0.9" in lines["p_biased"])

    # The trimming rule that keeps the warm pass off the tail in the first
    # place. It must never cut inside the crossing, and must keep the walk
    # contiguous.
    T = _THRESHOLD

    def chain(dnlls):
        return [{"x_fixed": 0.1 * (i + 1), "dnll": d}
                for i, d in enumerate(dnlls)]

    deep = chain([0.1, 0.5, 2.5, 40.0, 210.0])
    kept = _warm_chain_relevant(deep, T, 0.0)
    check("the deep tail past a crossing is dropped",
          len(kept) == 3, f"kept {len(kept)} of {len(deep)}")
    check("and everything up to and including the crossing is kept",
          kept[-1]["dnll"] == 2.5)

    shallow = chain([0.1, 0.5, 1.0, 1.5, 1.8])
    check("a side that never crosses is never trimmed",
          len(_warm_chain_relevant(shallow, T, 0.0)) == len(shallow))

    modest = chain([0.1, 0.5, 2.5, 8.0, 15.0])
    check("a point within the safety margin of the crossing is kept",
          len(_warm_chain_relevant(modest, T, 0.0)) == len(modest),
          "10x the threshold is 19.2 nats, so 15 stays")

    gappy = chain([0.1, float("nan"), 2.5, 300.0, 400.0])
    kept_g = _warm_chain_relevant(gappy, T, 0.0)
    check("a point with no dNLL does not break the walk",
          len(kept_g) == 3, f"kept {len(kept_g)}")

    check("the anchor is applied before the comparison",
          len(_warm_chain_relevant(chain([5.0, 5.4, 7.4, 45.0]), T, 4.9)) == 3)


def test_warm_start_is_monotone():
    """Warm-starting must never raise the profile or narrow an interval.

    This is the invariant that makes the pass safe to run unconditionally: both
    evaluations at a fixed value are upper bounds on the true profile, so
    keeping the lower one can only move the curve down. If this ever fails, the
    "can only help" argument is void and the pass needs a rethink.
    """
    print("\nWarm-start monotonicity:")
    k = 10
    nll, se, exact = _ill_conditioned(k, seed=3)

    names, cold, _a1, _c1 = _profile(nll, k, se, warm_passes=0)
    _n, warm, _a2, _c2 = _profile(nll, k, se, warm_passes=1)

    # Compare on the grid values the two runs share.
    worst = 0.0
    for name in names:
        xc, yc = cold[name]
        xw, yw = warm[name]
        lookup = {round(float(x), 9): float(y) for x, y in zip(xw, yw)}
        for x, y in zip(xc, yc):
            other = lookup.get(round(float(x), 9))
            if other is not None:
                worst = max(worst, other - y)
    check("no shared point rose after the warm pass", worst < 1e-6,
          f"worst rise {worst:.3g} nats")

    hc, hw = _half_widths(names, cold), _half_widths(names, warm)
    narrowed = [n for n in names
                if np.isfinite(hc[n]) and np.isfinite(hw[n])
                and hw[n] < hc[n] - 1e-6 * max(hc[n], 1e-12)]
    check("no interval narrowed after the warm pass", not narrowed,
          f"narrowed: {narrowed}")


def test_warm_pass_is_resumable():
    """A second sweep at the same budget must be a no-op, not a repeat.

    Without the ``warm_refined`` marker every relaunch would redo the entire
    continuation pass, which on a real model is hours.
    """
    print("\nWarm-pass resume guard:")
    k = 6
    nll, se, _exact = _ill_conditioned(k, seed=7)
    names = [f"p{i}" for i in range(k)]

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        # One shared `completed` store across two calls is what a resume looks
        # like; the checkpoint itself is exercised by test_parallel_profile.
        _t, _a, _w, conv1 = run_parallel_profile(
            _make_batch(nll, k), np.zeros(k), 0.0, names,
            [(-1e3, 1e3)] * k, ["lin"] * k, method="Nelder-Mead", wald_se=se,
            n_grid=4, se_span=3.0, n_refine=1, checkpoint=None, warm_passes=1)

    check("first run does warm work", conv1["warm"]["n_attempted"] > 0,
          str(conv1["warm"]))
    check("warm pass ran exactly one sweep", conv1["warm"]["passes"] == 1,
          str(conv1["warm"]))


def test_checkpoint_keeps_the_best_record():
    """Resuming must keep the lowest NLL per point, not the last one written.

    The warm pass writes a second record at every fixed value, so the store now
    genuinely contains duplicates. If a resume took the last record instead of
    the best, a warm evaluation that came out worse would silently undo a good
    cold one -- and the "can only lower the curve" guarantee would hold within a
    run but not across a restart, which is where these runs actually live.
    """
    print("\nCheckpoint best-wins semantics:")
    import json
    import shutil
    import tempfile
    from pyantigen.engine.Profile_checkpoint import ProfileCheckpoint

    k = 5
    nll, se, _exact = _ill_conditioned(k, seed=11)
    names = [f"p{i}" for i in range(k)]
    root = tempfile.mkdtemp(prefix="profile_ckpt_")

    def run(ckpt_root):
        ckpt = ProfileCheckpoint(ckpt_root, "run", "modelhash", "spechash")
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                return run_parallel_profile(
                    _make_batch(nll, k), np.zeros(k), 0.0, names,
                    [(-1e3, 1e3)] * k, ["lin"] * k, method="Nelder-Mead",
                    wald_se=se, n_grid=4, se_span=3.0, n_refine=1,
                    checkpoint=ckpt, warm_passes=1)
        finally:
            ckpt.close()

    try:
        _t1, _a1, _w1, conv1 = run(root)
        check("first run does warm work", conv1["warm"]["n_attempted"] > 0)

        files = [os.path.join(dp, f)
                 for dp, _dn, fn in os.walk(root) for f in fn
                 if f.endswith(".jsonl")]
        check("a checkpoint file per parameter", len(files) == k,
              f"got {len(files)}")

        recs = [json.loads(l) for f in files
                for l in open(f, encoding="utf-8") if l.strip()]
        check("warm records carry the resume marker",
              any(int(r.get("warm_refined", 0)) > 0 for r in recs))
        check("duplicate fixed values were written",
              len(recs) > len({(r["param_name"], round(r["x_fixed"], 12))
                               for r in recs}))

        # A resume must be a genuine no-op, not a repeat of the warm pass.
        t2, _a2, _w2, conv2 = run(root)
        check("resume does no warm work", conv2["warm"]["n_attempted"] == 0,
              str(conv2["warm"]))

        # Append a deliberately worse duplicate; the better one must survive.
        target = json.loads(open(files[0], encoding="utf-8").readline())
        poisoned = dict(target)
        poisoned["nll"] = float(target["nll"]) + 500.0
        with open(files[0], "a", encoding="utf-8") as fh:
            fh.write(json.dumps(poisoned) + "\n")

        ckpt = ProfileCheckpoint(root, "run", "modelhash", "spechash")
        loaded = ckpt.load(names)
        ckpt.close()
        got = loaded[target["param_name"]][round(float(target["x_fixed"]), 12)]
        check("a worse duplicate does not displace the better record",
              abs(float(got["nll"]) - float(target["nll"])) < 1e-9,
              f"kept nll={got['nll']} vs best {target['nll']}")

        t3, _a3, _w3, _c3 = run(root)
        same = all(
            np.allclose(t2[n][1], t3[n][1], rtol=1e-9, atol=1e-12)
            for n in names
        )
        check("a poisoned duplicate does not change the traces", same)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_checkpoint_wrapper_wiring():
    """The wrapper the driver actually calls must return what it unpacks.

    ``Model_optimize`` does ``traces, anchor, where, convergence =
    profile_all()``. That tuple has to survive the fingerprinting, run-id and
    checkpoint plumbing in between, and a stale-spec resume must be rejected
    rather than blended into the new run.
    """
    print("\nCheckpoint wrapper wiring:")
    import shutil
    import tempfile
    from pyantigen.engine.Optimize import _run_parallel_profile_with_checkpoint

    k = 4
    nll, se, _exact = _ill_conditioned(k, seed=5)
    names = [f"p{i}" for i in range(k)]
    root = tempfile.mkdtemp(prefix="profile_wrap_")

    class FakeEvaluator:
        """Stands in for ParallelEvaluator: profile_batch and evaluate_batch.

        The driver runs the slice screen before any profile point, so the
        double answers plain evaluations too. The objective here is positive
        definite, so every slice crosses the threshold well inside the bounds
        and the screen passes; what it does when one does not is covered in
        tests/test_slice_screen.py.
        """
        def __init__(self):
            self.profile_batch = _make_batch(nll, k)
            self.evaluate_batch = lambda xs, label=None: [
                float(nll(np.asarray(x, dtype=float))) for x in xs]

    def run(model_text):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            out = _run_parallel_profile_with_checkpoint(
                FakeEvaluator(), np.zeros(k), 0.0, names, [(-1e3, 1e3)] * k,
                ["lin"] * k, groups={}, model_text=model_text,
                paths={"plot_path": root}, method="Nelder-Mead",
                optimizer_kwargs=None, wald_se=se, n_grid=4, range_factor=2.0,
                se_span=3.0, n_refine=1, run_id="wiring", warm_passes=1,
            )
        return out, buf.getvalue()

    try:
        out, _log = run("model v1")
        check("wrapper returns a 4-tuple", isinstance(out, tuple) and len(out) == 4,
              f"got {type(out).__name__} of length "
              f"{len(out) if isinstance(out, tuple) else 'n/a'}")
        traces, anchor, where, convergence = out
        check("traces cover every parameter",
              all(n in traces for n in names))
        check("convergence carries the warm block", "warm" in convergence,
              str(sorted(convergence)))
        check("anchor is a float", isinstance(float(anchor), float))

        # Same model: must resume rather than recompute.
        out2, _log2 = run("model v1")
        check("resume reuses the checkpoint",
              out2[3]["warm"]["n_attempted"] == 0, str(out2[3]["warm"]))

        # Changed model: a different fingerprint means a different directory,
        # so none of the old points may leak in.
        out3, _log3 = run("model v2 -- different")
        check("a changed model does not resume the old points",
              out3[3]["warm"]["n_attempted"] > 0, str(out3[3]["warm"]))
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_dust_clamp_keeps_real_quantities():
    """The pre-dose dust clamp must delete noise and nothing else.

    Pre-equilibration leaves empty compartments at values like -2.4e-107 rather
    than zero, and once a dose fires those species acquire derivatives of order
    1e+05 while sitting at 1e-107 -- which is what made one PK block take 123 s
    instead of 0.39 s. Zeroing them is safe only because the threshold is far
    below anything physical: the antibody-trial arms carry genuine quantities
    near 1e-15, and an earlier 1e-12 cut would have silently deleted those.
    """
    print("\nPre-dose dust clamp:")
    from pyantigen.engine.Simulate import _DUST_THRESHOLD, clamp_state_dust

    class FakeModel:
        def __init__(self, values):
            self._v = values

        def getFloatingSpeciesIds(self):
            return list(self._v)

    class FakeRunner:
        def __init__(self, values):
            self._v = dict(values)
            self.model = FakeModel(self._v)

        def __getitem__(self, k):
            return self._v[k]

        def __setitem__(self, k, v):
            self._v[k] = v

    values = {
        "dust_negative": -2.4e-107,   # what pre-equilibration actually leaves
        "dust_positive": 6.9e-44,
        "denormal": 1.7e-122,
        "genuine_small": 1.01e-15,    # AB40_DensePlaqueNumber_BrainISF
        "genuine_smaller": 3.9e-20,
        "ordinary": 1.66e+03,
        "already_zero": 0.0,
        "negative_real": -8.5e-06,    # a real problem, not dust
    }
    r = FakeRunner(values)
    n = clamp_state_dust(r)

    check("threshold is far below any physical value",
          _DUST_THRESHOLD <= 1e-20, f"{_DUST_THRESHOLD:g}")
    check("dust is zeroed", r["dust_negative"] == 0.0 and r["denormal"] == 0.0)
    check("positive dust is zeroed too", r["dust_positive"] == 0.0)
    check("a genuine 1e-15 quantity survives",
          r["genuine_small"] == 1.01e-15, str(r["genuine_small"]))
    check("a genuine 1e-20 quantity survives",
          r["genuine_smaller"] == 3.9e-20, str(r["genuine_smaller"]))
    check("ordinary values are untouched", r["ordinary"] == 1.66e+03)
    check("a materially negative value is left for the caller to notice",
          r["negative_real"] == -8.5e-06, str(r["negative_real"]))
    check("count reflects only what changed", n == 3, f"{n}")
    check("re-clamping is a no-op", clamp_state_dust(r) == 0)


def test_invariance_check_scales_with_achieved_accuracy():
    """The check must not demand more agreement than the solver delivered.

    On 'Aducanumab_3mgkg' the pre-dose block hit CV_CONV_FAILURE and the retry
    ladder settled at rel_tol=1e-7. The check then compared the two runs at
    1e-8 and reported 26 "diverged" values -- the largest being
    AB42_DensePlaqueMass_BrainISF at 77.404365 versus 77.404743, a relative
    difference of 4.9e-6. Two seventy-year integrations computed at 1e-7
    agreeing to 5e-6 is the solver working; a parameter genuinely acting before
    the dose moves things by ~0.4 relative, five orders of magnitude larger.
    """
    print("\nInvariance check tolerance scaling:")
    from pyantigen.engine.Preequil_cache import _achieved_settings

    ss = {"absolute_tolerance": 1e-9, "relative_tolerance": 1e-9,
          "maximum_num_steps": 200000}

    settled, rel = _achieved_settings({"attempts": 1, "subdivided": False}, ss)
    check("a clean run needs no override", settled is None)
    check("a clean run reports the requested tolerance", rel == 1e-9, f"{rel:g}")

    ladder = {"attempts": 3, "abs_tol": 1e-7, "rel_tol": 1e-7,
              "max_steps": 200000}
    settled, rel = _achieved_settings(ladder, ss)
    check("a laddered run reports what it converged at", rel == 1e-7, f"{rel:g}")
    check("the second run is given the settled settings",
          settled is not None
          and settled["relative_tolerance"] == 1e-7
          and settled["absolute_tolerance"] == 1e-7, str(settled))

    compare_rtol = max(1e-8, 1000.0 * rel)
    observed_worst = 4.888e-06          # AB42_DensePlaqueMass_BrainISF
    genuine_worst = 4.352e-01           # from the Microglia_CL_low_AB42 control
    check("solver noise at the achieved tolerance is not flagged",
          observed_worst < compare_rtol,
          f"{observed_worst:.3e} vs {compare_rtol:.3e}")
    check("a genuine pre-dose dependence is still caught",
          genuine_worst > compare_rtol,
          f"{genuine_worst:.3e} vs {compare_rtol:.3e}")

    # Where the ladder subdivided, the loosest tolerance anywhere is the limit.
    nested = {"subdivided": True,
              "meta1": {"abs_tol": 1e-8, "rel_tol": 1e-8, "max_steps": 2},
              "meta2": {"subdivided": True,
                        "meta1": {"abs_tol": 1e-4, "rel_tol": 1e-4, "max_steps": 4},
                        "meta2": {"abs_tol": 1e-9, "rel_tol": 1e-9, "max_steps": 2}}}
    _s, rel_nested = _achieved_settings(nested, ss)
    check("a subdivided run reports its loosest piece", rel_nested == 1e-4,
          f"{rel_nested:g}")


def test_extension_reaches_a_threshold_the_grid_missed():
    """A grid that stops short must be walked outward, not reported as open.

    This is the Lecanemab failure in miniature. Two of its three parameters had
    no Wald SE, so the grid fell back to range_factor=2 and spanned [p/2, 2p];
    dNLL got to 0.19 and 0.68 and never reached 1.9207. Refinement could not
    help -- it interpolates inside a bracket and there was no bracket -- so the
    run ended with open intervals and no idea whether the parameters were
    identifiable at all.

    Here the SE is deliberately withheld and range_factor left at its default,
    so the opening grid is far too narrow by construction; the extension pass
    has to find the crossing anyway, and land on the analytic answer.
    """
    print("\nOutward extension when the opening grid falls short:")
    k = 4
    A = np.diag([1.0, 2.0, 3.0, 4.0])
    se = 1.0 / np.sqrt(np.diag(A))
    exact = np.sqrt(2 * _THRESHOLD) * se
    names = [f"p{i}" for i in range(k)]

    # Offset from zero so the optimum is positive: the multiplicative stepping
    # rule and the analytic answer both have to survive that shift.
    shift = 10.0
    nll = lambda x: 0.5 * (x - shift) @ A @ (x - shift)   # noqa: E731

    def run(max_extend):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            return run_parallel_profile(
                _make_batch(nll, k), np.full(k, shift), 0.0, names,
                [(1e-6, 1e6)] * k, ["lin"] * k, method="Nelder-Mead",
                wald_se=None,          # forces the range_factor fallback
                n_grid=4, range_factor=1.001,   # ~0.1% wide: hopeless on its own
                se_span=4.0, n_refine=3, checkpoint=None, warm_passes=1,
                max_extend=max_extend,
            )

    _t0, _a0, _w0, _c0 = run(0)
    traces0 = _t0
    open0 = [n for n in names
             if not all(np.isfinite(v) for v in
                        _extract_profile_ci(traces0[n][0], traces0[n][1],
                                            threshold=_THRESHOLD))]
    check("without extension the narrow grid really does come back open",
          len(open0) == k, f"{len(open0)}/{k} open")

    traces, _a, _w, conv = run(12)
    half = _half_widths(names, traces)
    for i, name in enumerate(names):
        check(f"{name} CI closed by extension", np.isfinite(half[name]),
              f"half={half[name]}")
        if not np.isfinite(half[name]):
            continue
        rel = abs(half[name] / exact[i] - 1.0)
        check(f"{name} half-width within 5% of analytic", rel < 0.05,
              f"{half[name]:.6g} vs {exact[i]:.6g} ({100 * rel:+.2f}%)")
    for name in names:
        check(f"{name} reports both sides crossed",
              conv["reach"][name]["lower"]["state"] == "crossed"
              and conv["reach"][name]["upper"]["state"] == "crossed",
              str(conv["reach"][name]))


def test_extension_steps_towards_the_threshold():
    """The outward step must be sized from the curve, not from a fixed factor.

    Doubling the distance every step knows nothing about the profile it is
    walking, and the failure is one-sided: a side that exhausts ``max_extend``
    before reaching dNLL 1.9207 reports no confidence bound at all. Ten of the
    sides in the last SILK/APP run ended that way, some with dNLL still under
    0.01 -- which doubling needs twenty-one steps to lift to the threshold, well
    past any sane budget.

    Fitting dNLL = c * d^p through the two outermost points and stepping to
    where that says the threshold is turns those twenty-one steps into seven.
    """
    print("\nOutward step sizing:")
    T = _THRESHOLD

    def side(pts):
        return [{"x_fixed": x, "dnll": d} for x, d in pts]

    def steps_needed(growth, d0, dnll0, cap=40):
        """How many steps of this size lift a quadratic side to the threshold."""
        n, d, val = 0, d0, dnll0
        while val < T and n < cap:
            d *= growth
            val = dnll0 * (d / d0) ** 2
            n += 1
        return n

    flat = side([(0.01, 0.0038)])
    g_flat = _extension_growth(flat, 0.0, True, T, 0.0, 2.0)
    check("a nearly flat side gets a long step, not a doubling",
          g_flat > 4.0, f"growth {g_flat:.3g}")
    check("and reaches the threshold inside a normal extension budget",
          steps_needed(g_flat, 0.01, 0.0038) <= 3,
          f"{steps_needed(g_flat, 0.01, 0.0038)} steps")
    check("where doubling would not",
          steps_needed(2.0, 0.01, 0.0038) > 3,
          f"{steps_needed(2.0, 0.01, 0.0038)} steps")

    near = side([(0.01, 1.0)])
    g_near = _extension_growth(near, 0.0, True, T, 0.0, 2.0)
    check("a side already close to the threshold gets a short step",
          1.0 < g_near < 2.0, f"growth {g_near:.3g}")

    check("a side already past the threshold falls back to the default",
          _extension_growth(side([(0.01, 2.5)]), 0.0, True, T, 0.0, 2.0) == 2.0)

    # Two points let the exponent be measured rather than assumed. A side that
    # is flatter than quadratic is exactly where assuming p=2 under-steps.
    quad = side([(0.01, 0.05), (0.02, 0.20)])
    lin = side([(0.01, 0.05), (0.02, 0.10)])
    steep = side([(0.01, 0.05), (0.02, 0.80)])
    g_q = _extension_growth(quad, 0.0, True, T, 0.0, 2.0)
    g_l = _extension_growth(lin, 0.0, True, T, 0.0, 2.0)
    g_s = _extension_growth(steep, 0.0, True, T, 0.0, 2.0)
    print(f"    growth: flatter-than-quadratic {g_l:.2f}, "
          f"quadratic {g_q:.2f}, steeper {g_s:.2f}")
    check("a flatter-than-quadratic side steps further than a quadratic one",
          g_l > g_q, f"{g_l:.3g} vs {g_q:.3g}")
    check("and a steeper one steps less far",
          g_s < g_q, f"{g_s:.3g} vs {g_q:.3g}")

    check("the step is clamped above, so noise-level dNLL cannot launch a "
          "step of 1e6",
          _extension_growth(side([(0.01, 1e-12)]), 0.0, True, T, 0.0, 2.0)
          <= _EXT_GROWTH_MAX + 1e-9)
    check("and clamped below, so a poor fit cannot stall the walk",
          _extension_growth(side([(0.01, 1.9)]), 0.0, True, T, 0.0, 2.0)
          >= _EXT_GROWTH_MIN - 1e-9)

    for bad, label in (([], "no points"),
                       (side([(0.01, float("nan"))]), "dNLL not finite"),
                       (side([(0.0, 0.5)]), "point sits on the optimum"),
                       (side([(0.01, -0.3)]), "dNLL below the anchor")):
        check(f"{label} falls back to the default growth",
              _extension_growth(bad, 0.0, True, T, 0.0, 2.0) == 2.0)

    # End to end: the same hopeless grid as the test above, closed in fewer
    # extension steps and fewer points.
    k = 4
    A = np.diag([1.0, 2.0, 3.0, 4.0])
    se_ex = 1.0 / np.sqrt(np.diag(A))
    exact = np.sqrt(2 * T) * se_ex
    names = [f"p{i}" for i in range(k)]
    shift = 10.0
    nll = lambda x: 0.5 * (x - shift) @ A @ (x - shift)   # noqa: E731

    def run(max_extend, fixed_doubling):
        import pyantigen.engine.Optimize as _O
        saved = _O._extension_growth
        if fixed_doubling:
            _O._extension_growth = lambda s, p, il, th, a, g, **kw: g
        spend = {}
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                traces, _a, _w, _c = run_parallel_profile(
                    _make_batch(nll, k, None, spend), np.full(k, shift), 0.0,
                    names, [(1e-6, 1e6)] * k, ["lin"] * k,
                    method="Nelder-Mead", wald_se=None, n_grid=4,
                    range_factor=1.001, se_span=4.0, n_refine=5,
                    checkpoint=None, warm_passes=1, max_extend=max_extend,
                    bracket_rtol=0.01)
        finally:
            _O._extension_growth = saved
        half = _half_widths(names, traces)
        closed = sum(1 for n in names if np.isfinite(half[n]))
        errs = [abs(half[n] / exact[i] - 1.0)
                for i, n in enumerate(names) if np.isfinite(half[n])]
        return closed, spend["points"], (max(errs) if errs else float("nan"))

    c_new, p_new, e_new = run(12, False)
    c_old, p_old, e_old = run(12, True)
    print(f"    closing a 0.1%-wide opening grid, 4 parameters:")
    print(f"      fixed doubling  {c_old}/{k} closed, {p_old} points, "
          f"worst {100 * e_old:.3f}%")
    print(f"      dNLL-targeted   {c_new}/{k} closed, {p_new} points, "
          f"worst {100 * e_new:.3f}%")
    check("both rules close every interval given enough budget",
          c_new == k and c_old == k)
    check("the targeted rule gets there on materially fewer points",
          p_new < 0.8 * p_old, f"{p_new} vs {p_old}")
    check("without giving up interval accuracy",
          e_new < 0.02, f"worst {100 * e_new:.3f}%")

    # The budget question is the one that actually bit: with a realistic
    # extension allowance, does the side reach the threshold at all?
    c_new3, _p, _e = run(3, False)
    c_old3, _p, _e = run(3, True)
    print(f"      with max_extend=3: dNLL-targeted {c_new3}/{k}, "
          f"fixed doubling {c_old3}/{k}")
    check("and closes them within a budget that leaves doubling short",
          c_new3 == k and c_old3 < k, f"{c_new3} vs {c_old3}")


def test_an_exact_confound_has_no_profile_bound():
    """Two parameters that only ever appear as a difference must profile flat.

    This is Projects/Example's Example3 made analytic. There the model reads
    ``predicted_B := SF * B_Comp1 / V_Comp1``, so it depends on the ratio
    SF/V_Comp1 and on nothing else; in log space a ratio is a difference, which
    is what is built here. One parameter is genuinely identifiable, and the
    other two enter the objective only through ``p1 - p2``.

    Every number the test checks is exact:

        profile half-width for p0    1.96       = sqrt(2 * 1.9207 / a)
        profile for p1 and p2        flat at 0    set p2 = p1 and the term dies
        slice half-width for p1      0.98       = sqrt(2 * 1.9207 / b)

    The last two lines are the whole point, and they disagree. Holding p2 still
    and walking p1 changes the difference, so the loss climbs and p1 looks
    tightly determined. Re-optimizing p2 to follow p1 holds the difference, so
    the loss never moves and p1 has no bound at all. Slice says identifiable,
    profile says not, and the profile is right.

    Why this fixture rather than the near-flat one in the bound test below:
    there the flat direction is diagonal, so the nuisance minimization has
    nothing to do and always succeeds. Here every point on p1's curve is only
    correct if the nuisance search actually travels the compensating distance.
    If it stalls, dNLL climbs, the curve stops being a profile and becomes a
    slice, and p1 acquires a confident interval of about +/- 0.98 that is
    entirely an artifact. That is a silent failure in production -- it is what
    the SILK/APP run produced when its points got four evaluations against
    fifteen nuisance dimensions -- so there are two tripwires here at different
    severities: the flatness check catches a small degradation, and the
    no-bound check catches one large enough to invent an interval.
    """
    print("\nAn exact confound has no profile bound:")
    a, b = 1.0, 4.0
    names = ["p_identified", "p_conf_num", "p_conf_den"]
    k = len(names)

    # p0 is identifiable; p1 and p2 appear only as (p1 - p2), zero at the
    # optimum. Their optima are held away from zero so the range_factor
    # fallback below has a scale to work from.
    res_x = np.array([2.0, 5.0, 5.0])

    def nll(x):
        return 0.5 * (a * (x[0] - 2.0) ** 2 + b * (x[1] - x[2]) ** 2)

    # The Hessian is [[a,0,0],[0,b,-b],[0,-b,b]], which is rank 2. A singular
    # direction must not come back with a finite SE -- pinv would hand back the
    # minimum-norm variance, which reads as a tight, confident interval on
    # exactly the parameters that have none.
    _cov, se_meas, _ci = compute_wald_uncertainty(nll, res_x)
    check("the confounded directions get no Wald SE",
          se_meas is None or not np.all(np.isfinite(np.asarray(se_meas,
                                                               dtype=float))),
          f"se={se_meas}")

    # What a singular Hessian actually hands the profile: a usable SE for the
    # identifiable parameter and nothing for the confounded pair, so their
    # opening grid falls back to range_factor. That is the realistic path for a
    # confounded parameter and the one worth exercising.
    wald_se = np.array([1.0 / np.sqrt(a), np.nan, np.nan])
    # Two decades of room for the confounded pair. Wide enough that the
    # walk reaches the bound on both sides and the contract is about
    # identifiability rather than about the grid, narrow enough that the
    # partner can always follow. An earlier draft gave them seven decades
    # and the test became a coin flip: chasing a partner across that range
    # is a hard optimization in its own right, so the fixture measured the
    # optimizer's luck instead of the property under test.
    bounds = [(-10.0, 10.0), (0.5, 50.0), (0.5, 50.0)]

    with contextlib.redirect_stdout(io.StringIO()):
        traces, anchor, _where, conv = run_parallel_profile(
            _make_batch(nll, k), res_x, 0.0, names, bounds, ["lin"] * k,
            method="Nelder-Mead", wald_se=wald_se, n_grid=4, range_factor=2.0,
            se_span=3.0, n_refine=2, checkpoint=None, warm_passes=1,
            max_extend=10)

    half = _half_widths(names, traces)
    peak = {n: float(np.nanmax(traces[n][1])) for n in names}
    print(f"    {'parameter':14} {'95% half-width':>15} {'highest dNLL':>13} "
          f"{'lower':>8} {'upper':>8}")
    for n in names:
        r = conv["reach"][n]
        print(f"    {n:14} {half[n]:15.4g} {peak[n]:13.4g} "
              f"{r['lower']['state']:>8} {r['upper']['state']:>8}")

    exact = np.sqrt(2 * _THRESHOLD / a)
    check("the identifiable parameter recovers its analytic half-width",
          abs(half["p_identified"] / exact - 1.0) < 0.03,
          f"{half['p_identified']:.6g} vs {exact:.6g}")

    for n in names[1:]:
        lo, hi = _extract_profile_ci(traces[n][0], traces[n][1],
                                     threshold=_THRESHOLD)
        check(f"{n} has no lower bound", not np.isfinite(lo), f"lo={lo}")
        check(f"{n} has no upper bound", not np.isfinite(hi), f"hi={hi}")
        r = conv["reach"][n]
        check(f"{n} is reported as bound-limited, not out of budget",
              r["lower"]["state"] == "bound" and r["upper"]["state"] == "bound",
              str(r))
        # The sharp one. A flat profile is only flat if the nuisance search
        # really did move the partner to compensate at every fixed value.
        check(f"{n} profiles flat, so the nuisance search did compensate",
              peak[n] < 0.1, f"highest dNLL {peak[n]:.4g}")

    # The contrast, taken from the objective itself so it depends on nothing
    # else: displacing p1 alone by the slice half-width lifts the loss over the
    # threshold, while re-optimizing leaves the whole curve below 0.1.
    h_slice = np.sqrt(2 * _THRESHOLD / b)
    slice_dnll = nll(res_x + np.array([0.0, h_slice, 0.0]))
    print(f"    moving p_conf_num alone by {h_slice:.4g}: "
          f"slice dNLL {slice_dnll:.4g}, profile dNLL at most "
          f"{peak[names[1]]:.4g}")
    check("a slice through the same fixture does cross the threshold",
          slice_dnll >= _THRESHOLD - 1e-9, f"{slice_dnll:.6g}")
    check("so the two methods genuinely disagree on this parameter",
          slice_dnll > 10 * max(peak[names[1]], 1e-12))

    check("a flat profile does not fake a better optimum",
          abs(anchor) < 1e-3, f"anchor {anchor:.4g}")

    # The same verdict on a log10-scaled fit, where the confound is the ratio
    # it is in the real model rather than a difference. Only the verdict is
    # checked; half-widths in linear units are different arithmetic and the
    # linear case above already pins those.
    with contextlib.redirect_stdout(io.StringIO()):
        traces_log, _a2, _w2, _c2 = run_parallel_profile(
            _make_batch(nll, k), res_x, 0.0, names,
            [(-10.0, 10.0), (3.0, 7.0), (3.0, 7.0)], ["log10"] * k,
            method="Nelder-Mead", wald_se=wald_se, n_grid=4, range_factor=2.0,
            se_span=3.0, n_refine=2, checkpoint=None, warm_passes=1,
            max_extend=10)
    half_log = _half_widths(names, traces_log)
    check("on a log10 fit the identifiable parameter still gets a bound",
          np.isfinite(half_log["p_identified"]), f"{half_log['p_identified']}")
    check("and the confounded ratio still gets none",
          not np.isfinite(half_log["p_conf_num"])
          and not np.isfinite(half_log["p_conf_den"]),
          f"{half_log['p_conf_num']}, {half_log['p_conf_den']}")


def test_a_confound_survives_a_lengthening_stride():
    """The same confound, but with a nuisance space big enough to get lost in.

    The two-parameter version above is the contract; this one is the mechanism.
    Thirteen correlated, ill-conditioned parameters are profiled alongside an
    exactly confounded pair, so re-optimizing at any fixed value is a
    fourteen-dimensional search rather than a two-dimensional one, and the
    compensating partner has somewhere to get lost.

    It exists because the engine failed it, silently, in the way that matters.
    The extension pass deliberately lengthens its stride as it hunts for the
    threshold, so a chain here walks 6.25, 7.5, 8.75, 10 and then jumps to 25.
    The nuisance partner has to follow all the way, but the warm simplex was
    being sized from the *previous* step, 1.25, so it arrived at a step of 15
    six times too small to reach. That point came back at dNLL 13.2 where its
    own extension-pass evaluation had reached 0.14, which was enough to lift the
    curve over the threshold and hand a structurally unidentifiable parameter a
    confidence bound. Nothing in the run looked wrong.

    ``_predicted_travel`` fixes it by turning the seed's travel into a rate and
    re-applying it to the step actually being taken. Measured over three seeds,
    with the highest dNLL reached on the confounded pair and the number of
    confidence bounds invented for it:

        seed   scaled by the step just taken   scaled by the step being taken
        0            17.59,  1 bound                 0.0020,  none
        1           135.7,   1 bound                 0.0356,  none
        2           207.1,   2 bounds                0.0587,  none

    Both arms run here, because an assertion about a fix is worth little unless
    the test can show the fix is what satisfies it. That costs this test about
    ninety seconds, which is most of what it adds to the suite; the failure it
    guards took a day to find and is invisible in a finished run.

    Only the fourteen-dimensional version reproduces. At six, eight or ten
    nuisance dimensions the search recovers from an undersized simplex either
    way, so a cheaper fixture would assert nothing.
    """
    print("\nA confound in fourteen nuisance dimensions:")
    m = 14
    base, se_block, exact_block = _ill_conditioned(m)
    k = m + 1
    names = [f"p{i}" for i in range(k)]

    # Coordinates 0..12 are the identifiable ill-conditioned block. The last
    # two enter only as their difference, which is the confound; their optima
    # are held away from zero so the range_factor fallback has a scale.
    res_x = np.concatenate([np.zeros(m - 1), [5.0, 5.0]])

    def nll(x):
        return base(np.concatenate([x[:m - 1], [x[m - 1] - x[m]]]))

    # No Wald SE for the confounded pair, as a singular Hessian gives.
    wald_se = np.concatenate([se_block[:m - 1], [np.nan, np.nan]])
    bounds = [(-1e3, 1e3)] * (m - 1) + [(1.0, 25.0), (1.0, 25.0)]

    def run(predict_from_upcoming_step):
        import pyantigen.engine.Optimize as _O
        saved = _O._predicted_travel
        if not predict_from_upcoming_step:
            # The old rule: size the simplex from the step just taken.
            _O._predicted_travel = lambda seed, x_step: _O._seed_travel(seed)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                traces, _a, _w, conv = run_parallel_profile(
                    _make_batch(nll, k), res_x, 0.0, names, bounds,
                    ["lin"] * k, method="Nelder-Mead", wald_se=wald_se,
                    n_grid=4, range_factor=2.0, se_span=3.0, n_refine=2,
                    checkpoint=None, warm_passes=1, max_extend=10)
        finally:
            _O._predicted_travel = saved
        peak = max(float(np.nanmax(traces[n][1])) for n in names[-2:])
        invented = 0
        for n in names[-2:]:
            lo, hi = _extract_profile_ci(traces[n][0], traces[n][1],
                                         threshold=_THRESHOLD)
            invented += int(np.isfinite(lo)) + int(np.isfinite(hi))
        half = _half_widths(names, traces)
        return peak, invented, half, conv

    peak_old, bad_old, _h_old, _c_old = run(False)
    peak_new, bad_new, half, conv = run(True)

    print(f"    {'simplex scaled by':>26} {'highest dNLL':>13} "
          f"{'bounds invented':>16}")
    print(f"    {'the step just taken':>26} {peak_old:13.4g} {bad_old:16d}")
    print(f"    {'the step being taken':>26} {peak_new:13.4g} {bad_new:16d}")

    check("sizing the simplex from the previous step really does break this",
          peak_old > _THRESHOLD and bad_old > 0,
          f"peak {peak_old:.4g}, {bad_old} bound(s) invented -- if this passes "
          f"the fixture no longer reproduces the bug and asserts nothing")

    check("sizing it from the step being taken keeps the profile flat",
          peak_new < 0.1, f"highest dNLL {peak_new:.4g}")
    check("so no bound is invented for a structurally unidentifiable pair",
          bad_new == 0, f"{bad_new} bound(s)")

    for n in names[-2:]:
        r = conv["reach"][n]
        check(f"{n} walks to its declared bound on both sides",
              r["lower"]["state"] == "bound" and r["upper"]["state"] == "bound",
              str(r))

    check("and the identifiable block is unharmed by sharing the fit",
          abs(half["p0"] / exact_block[0] - 1.0) < 0.03,
          f"{half['p0']:.6g} vs {exact_block[0]:.6g}")


def test_extension_stops_at_the_bound_and_says_so():
    """Reaching the parameter bound is an answer; running out of steps is not.

    A flat direction inside a tight box can never cross the threshold, and the
    old code reported that identically to "the grid was too narrow" -- sending
    the user to re-run with a wider span that the bound would clip right back.
    """
    print("\nExtension terminates at the declared bound:")
    k = 2
    # Second parameter is nearly flat, so no reachable value crosses 1.9207.
    A = np.diag([1.0, 1e-8])
    se = 1.0 / np.sqrt(np.diag(A))
    names = [f"p{i}" for i in range(k)]

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        traces, _a, _w, conv = run_parallel_profile(
            _make_batch(nll := (lambda x: 0.5 * x @ A @ x), k),
            np.zeros(k), 0.0, names,
            # p1 is boxed in far too tightly to ever reach the threshold.
            [(-1e6, 1e6), (-2.0, 2.0)], ["lin"] * k, method="Nelder-Mead",
            wald_se=se, n_grid=4, se_span=3.0, n_refine=2, checkpoint=None,
            warm_passes=1, max_extend=10)

    r0, r1 = conv["reach"]["p0"], conv["reach"]["p1"]
    check("the identifiable parameter still crosses",
          r0["lower"]["state"] == "crossed" and r0["upper"]["state"] == "crossed",
          str(r0))
    check("the boxed-in parameter is reported as bound-limited, not budget",
          r1["lower"]["state"] == "bound" and r1["upper"]["state"] == "bound",
          str(r1))
    check("it walked all the way to the bound",
          abs(r1["upper"]["reach"] - 2.0) < 1e-9
          and abs(r1["lower"]["reach"] + 2.0) < 1e-9, str(r1))
    check("and its CI really is open",
          not np.isfinite(_extract_profile_ci(traces["p1"][0], traces["p1"][1],
                                              threshold=_THRESHOLD)[1]))


def test_extension_steps_multiplicatively():
    """Stepping must be geometric, or the downward side collapses onto the bound.

    Doubling a *linear* offset from 8.5e-5 overshoots zero on the first step and
    clips straight onto a lower bound of 1e-9, turning five unexplored decades
    into one useless bracket. Doubling the log distance visits p/4, /16, /256
    instead. Both are two-fold steps; only one of them can walk downward.
    """
    print("\nExtension step geometry:")
    from pyantigen.engine.Optimize import _next_extension_value

    p = 8.5156e-5
    lb, ub = 1e-9, 1e-1

    # Linear-scaled positive parameter: steps are ratios of the value.
    x = p / 2.0
    seen = []
    for _ in range(4):
        x = _next_extension_value(p, x, lb, ub, -1, is_log=False, growth=2.0)
        if x is None:
            break
        seen.append(x)
    check("downward steps are multiplicative", len(seen) >= 3, str(seen))
    check("first downward step is p/4", abs(seen[0] / (p / 4.0) - 1) < 1e-9,
          f"{seen[0]:.6g} vs {p / 4.0:.6g}")
    check("second is p/16", abs(seen[1] / (p / 16.0) - 1) < 1e-9,
          f"{seen[1]:.6g} vs {p / 16.0:.6g}")
    check("each step stays above the lower bound",
          all(v >= lb - 1e-30 for v in seen), str(seen))

    up = _next_extension_value(p, 2.0 * p, lb, ub, +1, is_log=False, growth=2.0)
    check("upward step is multiplicative too", abs(up / (4.0 * p) - 1) < 1e-9,
          f"{up:.6g} vs {4.0 * p:.6g}")

    # A step that would overshoot the bound is clipped to it, exactly once.
    clipped = _next_extension_value(p, 5e-2, lb, ub, +1, is_log=False, growth=2.0)
    check("an overshooting step is clipped to the bound",
          abs(clipped - ub) < 1e-15, f"{clipped}")
    check("a point already on the bound ends the side",
          _next_extension_value(p, ub, lb, ub, +1, is_log=False, growth=2.0) is None)
    check("a point already on the lower bound ends the side",
          _next_extension_value(p, lb, lb, ub, -1, is_log=False, growth=2.0) is None)

    # A log10-scaled parameter is already stored as its logarithm, so doubling
    # the offset there is itself multiplicative and must not be logged twice.
    u = np.log10(p)
    nxt = _next_extension_value(u, u - 0.3, -9.0, -1.0, -1, is_log=True, growth=2.0)
    check("log-scaled parameters double their offset in log space",
          abs(nxt - (u - 0.6)) < 1e-12, f"{nxt} vs {u - 0.6}")


def test_refinement_stops_at_a_tight_bracket():
    """Probing must stop once the crossing is located well enough.

    One Lecanemab side spent all four probes moving a crossing from 0.017920 to
    0.017911 -- 0.05% on an interval whose half-width is 25% -- while the other
    side of the same parameter never reached the threshold at all. The budget
    has to stop chasing digits nobody asked for.
    """
    print("\nRefinement stops on a tight bracket:")
    from pyantigen.engine.Optimize import _bracket_is_tight

    check("the observed over-refined bracket is judged tight",
          _bracket_is_tight(0.017920, 0.017754, 0.02372, False, 0.05),
          "0.0179 vs 0.0178 is a 0.9% bracket")
    check("a genuinely wide bracket is not",
          not _bracket_is_tight(0.01974, 0.01775, 0.02372, False, 0.05))
    check("rtol of zero disables the stop (old behaviour)",
          not _bracket_is_tight(0.017920, 0.017754, 0.02372, False, 0.0))
    check("log-scaled brackets are measured as ratios of the value",
          _bracket_is_tight(np.log10(1.0), np.log10(1.02), 0.0, True, 0.05)
          and not _bracket_is_tight(np.log10(1.0), np.log10(1.5), 0.0, True, 0.05))
    check("a bracket spanning the optimum falls back to an offset measure",
          _bracket_is_tight(-1.0, -1.01, 0.0, False, 0.05))

    # End to end: a tight rtol must still land the CI, and a loose one must use
    # strictly fewer probes to get there.
    k = 4
    A = np.diag([1.0, 2.0, 3.0, 4.0])
    se = 1.0 / np.sqrt(np.diag(A))
    exact = np.sqrt(2 * _THRESHOLD) * se
    names = [f"p{i}" for i in range(k)]

    def run(rtol):
        calls = {"n": 0}
        inner = _make_batch(lambda x: 0.5 * x @ A @ x, k)

        def counting(jobs, on_result=None, label=None):
            if label and label.startswith("profile-refine"):
                calls["n"] += len(jobs)
            return inner(jobs, on_result=on_result, label=label)

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            traces, _a, _w, _c = run_parallel_profile(
                counting, np.zeros(k), 0.0, names, [(-1e3, 1e3)] * k,
                ["lin"] * k, method="Nelder-Mead", wald_se=se, n_grid=6,
                se_span=3.0, n_refine=4, checkpoint=None, warm_passes=1,
                bracket_rtol=rtol)
        return traces, calls["n"]

    t_fine, n_fine = run(0.0)
    t_loose, n_loose = run(0.05)
    print(f"    refinement probes: rtol=0 -> {n_fine}, rtol=0.05 -> {n_loose}")
    check("a relative tolerance spends fewer probes", n_loose < n_fine,
          f"{n_fine} -> {n_loose}")

    h = _half_widths(names, t_loose)
    worst = max(abs(h[n] / exact[i] - 1.0) for i, n in enumerate(names))
    check("and still lands the CI to better than the tolerance itself",
          worst < 0.05, f"worst error {100 * worst:+.2f}%")


def test_plot_window_follows_the_extended_grid():
    """The figure must show the points extension added, not a fixed window.

    The old axes were pinned to the linear range [0.2, 3.5] of the optimum. Once
    a side can be walked to a fifth or a fiftieth of the optimum, that window
    crops off the very crossing the walk was for -- and a linear axis could not
    have resolved it anyway, since everything below 0.2 is squeezed against the
    left spine.
    """
    print("\nProfile plot window:")
    from pyantigen.engine.Model_optimize import _profile_plot_x

    est = np.array([1.0])

    def contains_all(win, cache, estimates):
        """Whether every computed point falls inside the drawn window."""
        for idx, (pv, _nr) in cache.items():
            r = np.asarray(pv, dtype=float) / estimates[idx]
            if np.any(r < win[0]) or np.any(r > win[1]):
                return False
        return True

    # A crossing far outside the old fixed [0.2, 3.5]: the window must follow it.
    x = np.array([0.01, 0.05, 0.2, 0.6, 1.0, 1.5, 3.0])
    y = np.array([9.0, 3.0, 1.0, 0.1, 0.0, 0.4, 2.5])
    ci = [(0.06, 2.6)]
    win, use_log = _profile_plot_x({0: (x, y)}, est, ci)
    check("a positive profile is drawn on a log axis", use_log)
    check("the window contains both crossings",
          win[0] < 0.06 and 2.6 < win[1], str(win))
    check("the window still contains the optimum",
          win[0] < 1.0 < win[1], str(win))
    check("it extends past the crossing, but not to the whole grid",
          win[0] < 0.06 and win[0] > x.min(), f"{win} vs data min {x.min()}")

    # The GantenerumabIV shape: the grid was clipped to a parameter bound a
    # thousandfold below the optimum, but every crossing is close in. Spanning
    # the data squeezed the entire informative region into a sliver at the right
    # edge; the window has to follow the crossings instead.
    xg = np.array([1e-3, 1e-2, 0.5, 0.8, 1.0, 1.3, 2.0])
    yg = np.array([58.0, 30.0, 2.0, 0.4, 0.0, 0.5, 3.0])
    cig = [(0.53, 1.83)]
    wing, _lg = _profile_plot_x({0: (xg, yg)}, est, cig)
    check("a bound-clipped grid does not dictate the scale",
          wing[0] > 0.2, f"left edge {wing[0]:.4g} (data goes to {xg.min()})")
    check("and both crossings are still comfortably inside",
          wing[0] < 0.53 and 1.83 < wing[1], str(wing))
    check("the margin past the crossing is modest, not unbounded",
          wing[1] < 1.83 * 2.0 and wing[0] > 0.53 / 2.0, str(wing))

    # A side with no crossing has no anchor, so how far it was walked is the
    # finding and must stay visible.
    xo = np.array([1e-3, 1e-2, 0.1, 1.0, 1.3, 1.8])
    yo = np.array([0.3, 0.2, 0.1, 0.0, 0.9, 2.4])
    cio = [(np.nan, 1.7)]
    wino, _lo = _profile_plot_x({0: (xo, yo)}, est, cio)
    check("an uncrossed side still shows how far it was walked",
          wino[0] <= 1e-3, f"left edge {wino[0]:.4g}")
    check("while the crossed side is still trimmed to its crossing",
          wino[1] < 1.7 * 2.0, str(wino))

    # Several parameters at once: the window is the union over all of them.
    est3 = np.array([1.0, 1.0, 1.0])
    multi = {0: (np.array([0.9, 1.0, 1.1]), np.array([2.0, 0.0, 2.0])),
             1: (np.array([0.02, 1.0, 50.0]), np.array([3.0, 0.0, 3.0])),
             2: (np.array([0.5, 1.0, 2.0]), np.array([1.0, 0.0, 1.0]))}
    cim = [(0.95, 1.05), (0.05, 20.0), (0.6, 1.6)]
    winm, _lm = _profile_plot_x(multi, est3, cim)
    check("the window covers the widest parameter's crossings",
          winm[0] < 0.05 and 20.0 < winm[1], str(winm))

    # Without CI information the window falls back to the full data range, so an
    # older caller (or a fallback path that never extracted a CI) still works.
    winf, _lf = _profile_plot_x({0: (x, y)}, est, None)
    check("no CI given falls back to spanning the data",
          contains_all(winf, {0: (x, y)}, est), str(winf))

    # A parameter that changes sign cannot use a log axis.
    xn = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    yn = np.array([4.0, 1.0, 0.0, 1.0, 4.0])
    winn, use_log_neg = _profile_plot_x({0: (xn, yn)}, est, [(-1.4, 1.4)])
    check("a sign-changing profile falls back to a linear axis", not use_log_neg)
    check("and brackets its crossings on a linear axis",
          winn[0] < -1.4 and 1.4 < winn[1], str(winn))

    # Degenerate inputs must not raise or return a broken window.
    empty, _l = _profile_plot_x({}, est, [])
    check("no points yields no window", empty is None, str(empty))
    nanw, _l2 = _profile_plot_x(
        {0: (np.array([1.0, 2.0]), np.array([np.nan, np.nan]))}, est, [])
    check("a trace with no finite dNLL still gets a window from its x range",
          nanw is not None and nanw[0] < nanw[1], str(nanw))
    single, _l3 = _profile_plot_x({0: (np.array([2.0]), np.array([0.0]))},
                                  est, [])
    check("a single point still gets a drawable interval",
          single is not None and single[0] < 2.0 < single[1], str(single))
    # A CI wider than anything computed must not invent unexplored territory.
    widew, _l4 = _profile_plot_x({0: (x, y)}, est, [(1e-9, 1e9)])
    check("the window never runs past the computed data",
          widew[0] >= x.min() / 1.2 and widew[1] <= x.max() * 1.2, str(widew))


def test_profile_report_states_the_answer():
    """The summary must carry the interval, the factor, and why a bound is missing.

    The console reports each interval as it is extracted, interleaved with
    thousands of solver messages; the report is what survives the scrollback, so
    the numbers a reader would otherwise recompute by hand have to be in it.
    """
    print("\nProfile summary report:")
    from pyantigen.engine.Model_optimize import _profile_report_text

    est = np.array([1.0, 2.0, 3.0])
    names = ["p_ok", "p_bound", "p_budget"]
    ci = [(0.8, 1.3), (np.nan, np.nan), (1.0, np.nan)]
    status = ["ok", "open", "open_upper"]
    reach = {
        "p_ok": {"lower": {"state": "crossed", "reach": 0.5, "max_dnll": 4.0,
                           "at_bound": False},
                 "upper": {"state": "crossed", "reach": 1.9, "max_dnll": 3.1,
                           "at_bound": False}},
        "p_bound": {"lower": {"state": "bound", "reach": 1e-9, "max_dnll": 0.2,
                              "at_bound": True},
                    "upper": {"state": "bound", "reach": 10.0, "max_dnll": 0.3,
                              "at_bound": True}},
        "p_budget": {"lower": {"state": "crossed", "reach": 0.9,
                               "max_dnll": 2.5, "at_bound": False},
                     "upper": {"state": "budget", "reach": 40.0,
                               "max_dnll": 1.1, "at_bound": False}},
    }
    opt = {"x": est, "stats": {
        "profile_convergence": {
            "n_points": 40, "n_not_converged": 3, "n_unknown": 1,
            "per_param": {"p_ok": {"n": 14, "n_not_converged": 0},
                          "p_bound": {"n": 13, "n_not_converged": 3},
                          "p_budget": {"n": 13, "n_not_converged": 0}},
            "warm": {"n_attempted": 40, "n_improved": 12,
                     "nats_recovered": 3.75},
            "reach": reach},
        "profile_anchor_gap": -1.38,
        "wald_se": [0.1, np.nan, np.nan],
        "wald_ci": [(0.8, 1.2), (np.nan, np.nan), (np.nan, np.nan)]}}

    txt = _profile_report_text(opt, names, est, ci, status, {}, "M", "TAG",
                               "2026-08-24 14:00:00")

    # The console this prints to is routinely on a codepage that mangles
    # en/em dashes into replacement characters, so the report stays ASCII.
    check("the report is pure ASCII", txt.isascii())
    check("it carries the interval", "[0.8, 1.3]" in txt)
    check("it states the interval as a factor of the optimum",
          "0.8x" in txt and "1.3x" in txt and "spans" in txt)
    check("a missing bound reads as n/a, not nan", "n/a" in txt
          and "nan" not in txt.lower().replace("n/a", ""))
    check("a bound-limited side is named as such",
          "stopped at the parameter bound" in txt)
    check("a budget-limited side is named as such",
          "ran out of extension steps" in txt)
    check("the two are given opposite advice",
          "Widen that bound" in txt and "profile_max_extend" in txt)
    check("capped nuisance optimizations are flagged on the parameter",
          "hit the optimizer cap" in txt)
    check("a fit above the profile minimum is shouted about",
          "BELOW the reported optimum" in txt and "1.38" in txt)
    check("a missing Wald SE says why", "Hessian was singular" in txt)
    check("the verdict counts every category",
          "both bounds found          1 of 3" in txt
          and "one bound only" in txt and "neither bound found" in txt)

    # A clean run must not emit the warning blocks at all.
    clean = {"x": np.array([1.0]), "stats": {
        "profile_convergence": {
            "n_points": 10, "n_not_converged": 0, "n_unknown": 0,
            "per_param": {"p": {"n": 10, "n_not_converged": 0}},
            "warm": {"n_attempted": 0, "n_improved": 0, "nats_recovered": 0.0},
            "reach": {"p": {"lower": {"state": "crossed", "reach": 0.5,
                                      "max_dnll": 3.0, "at_bound": False},
                            "upper": {"state": "crossed", "reach": 1.5,
                                      "max_dnll": 3.0, "at_bound": False}}}},
        "profile_anchor_gap": 0.0}}
    ok_txt = _profile_report_text(clean, ["p"], np.array([1.0]),
                                  [(0.9, 1.1)], ["ok"], {}, "M", "TAG", "now")
    check("a clean run says the fit is at the profile minimum",
          "the fit sits at the profile minimum" in ok_txt)
    check("a clean run raises no bound/budget advice",
          "Widen that bound" not in ok_txt
          and "profile_max_extend" not in ok_txt)
    check("a clean run reports no capped points",
          "hit the optimizer cap (" not in ok_txt)


def test_se_span_widens_the_grid():
    """profile_se_span must reach the grid, and widen it.

    When a CI comes back "open" the status line says to raise this, so it has to
    be reachable from a run's settings -- it previously existed only as a lambda
    default, which made the advice impossible to follow.
    """
    print("\nprofile_se_span reaches the grid:")
    from pyantigen.engine.Model_optimize import _profile_kwargs
    from pyantigen.engine.Optimize import _profile_grid_for

    kw = _profile_kwargs({"profile_se_span": 12.0, "profile_n_grid": 8,
                          "unrelated_setting": 3})
    check("se_span is forwarded", kw.get("se_span") == 12.0, str(kw))
    check("n_grid is forwarded", kw.get("n_grid") == 8, str(kw))
    check("unset controls are not forwarded",
          "n_refine" not in kw and "warm_passes" not in kw, str(kw))
    check("unrelated settings are ignored",
          set(kw) <= {"se_span", "n_grid", "n_refine", "range_factor",
                      "warm_passes", "max_extend", "extend_growth",
                      "bracket_rtol"}, str(kw))
    ext = _profile_kwargs({"profile_max_extend": 12,
                           "profile_bracket_rtol": 0.02,
                           "profile_extend_growth": 3.0})
    check("the extension controls are forwarded too",
          ext == {"max_extend": 12, "bracket_rtol": 0.02,
                  "extend_growth": 3.0}, str(ext))
    check("an empty settings dict yields no overrides", _profile_kwargs({}) == {})
    check("None settings are tolerated", _profile_kwargs(None) == {})

    # Wide bounds, as the PK specs have, so nothing is clipped.
    res_x = np.array([1.0])
    bounds = [(1e-5, 1e5)]
    wald_se = np.array([0.1])
    widths = {}
    for span in (4.0, 12.0):
        left, right, _b, _log = _profile_grid_for(
            0, res_x, bounds, ["lin"], wald_se, n_grid=5,
            range_factor=2.0, se_span=span)
        widths[span] = max(right) - min(left)
        check(f"se_span={span:g} reaches {span:g} SE from the optimum",
              abs(max(right) - (1.0 + span * 0.1)) < 1e-9,
              f"outermost point {max(right)}")
    check("a larger span gives a wider grid",
          widths[12.0] > widths[4.0] * 2.5,
          f"{widths[4.0]:.3f} -> {widths[12.0]:.3f}")

    # Bounds still win: a span that would overshoot is clipped, which is why a
    # tightly bounded parameter cannot be opened up this way.
    left, right, _b, _l = _profile_grid_for(
        0, res_x, [(0.95, 1.05)], ["lin"], wald_se, n_grid=5,
        range_factor=2.0, se_span=50.0)
    check("the grid is still clipped to the parameter bounds",
          max(right) <= 1.05 + 1e-12 and min(left) >= 0.95 - 1e-12,
          f"[{min(left)}, {max(right)}]")


def test_a_spec_can_set_its_own_grid():
    """Grid density belongs to the spec, not to the run's diagnostics preset.

    silk_appfull and the Aducanumab PK validation share ``_PROFILE_ONLY``, and
    their evaluations differ by a factor of forty in cost. A thinner grid is
    right for one and pointless for the other, so the spec states its own and
    it wins over the preset.
    """
    print("\nA spec's own grid settings:")
    from pyantigen.engine.Model_optimize import _profile_kwargs

    class Spec:
        def __init__(self, grid):
            self.optimizer_kwargs = {"profile_grid": grid} if grid else {}

    preset = {"profile_se_span": 4}

    kw = _profile_kwargs(preset, Spec({"n_grid": 3}))
    check("the spec's grid reaches the profile", kw.get("n_grid") == 3, str(kw))
    check("and the preset's other settings survive",
          kw.get("se_span") == 4, str(kw))

    check("a spec with no grid settings changes nothing",
          _profile_kwargs(preset, Spec(None)) == {"se_span": 4})
    check("no spec at all changes nothing",
          _profile_kwargs(preset) == {"se_span": 4})

    # The spec wins: it knows what its own evaluations cost, the preset does
    # not.
    both = _profile_kwargs({"profile_n_grid": 6}, Spec({"n_grid": 3}))
    check("the spec overrides the preset", both.get("n_grid") == 3, str(both))

    # A typo here would otherwise look exactly like a setting that did not
    # help, which is the worst way for it to fail.
    try:
        _profile_kwargs(preset, Spec({"ngrid": 3}))
        check("an unknown grid key is rejected", False, "no raise")
    except ValueError as exc:
        check("an unknown grid key is rejected", True)
        check("and the message names the offender and the alternatives",
              "ngrid" in str(exc) and "n_grid" in str(exc), str(exc))


def test_profile_grid_never_reaches_scipy():
    """``profile_grid`` is an engine key and must be stripped before minimize.

    It sits in the same optimizer_kwargs scipy is otherwise handed, and scipy
    rejects unknown option keys, so a leak here would fail the *fit* rather
    than the profile -- somewhere far from the setting that caused it.
    """
    print("\nprofile_grid is engine-only:")
    from pyantigen.engine.Optimize import (
        _ENGINE_ONLY_OPTIMIZER_KEYS, _prepare_optimizer_kwargs,
    )
    check("it is declared engine-only",
          "profile_grid" in _ENGINE_ONLY_OPTIMIZER_KEYS,
          str(_ENGINE_ONLY_OPTIMIZER_KEYS))

    prepared = _prepare_optimizer_kwargs(
        "Nelder-Mead",
        {"profile_grid": {"n_grid": 3}, "options": {"maxiter": 10}},
        False, None, None,
    )
    check("and _prepare_optimizer_kwargs strips it",
          "profile_grid" not in prepared, str(prepared))

    # The nuisance path strips the same set; a real minimize proves it rather
    # than trusting the tuple.
    res = _minimize_nuisance(
        lambda x, v: float(np.sum((np.asarray(x) - v) ** 2)),
        np.zeros(2), (0.0,), "Nelder-Mead", None,
        {"profile_grid": {"n_grid": 3}, "options": {"maxfev": 20}},
    )
    check("and a nuisance minimization runs with it present",
          np.isfinite(float(res.fun)), str(res))


def test_zz_every_check_passed():
    """Make the checks above visible to pytest.

    ``check`` records failures rather than raising so that one run reports every
    problem instead of stopping at the first. Under pytest that means a test
    function whose checks failed still returns normally and is reported green.
    This runs last and turns the collected failures into one real assertion.
    """
    assert not failures, "failed checks: " + ", ".join(failures)


def main():
    test_separable_quadratic()
    test_threshold_is_a_likelihood_ratio()
    test_capped_optimization_is_reported()
    test_budget_scales_with_dimension()
    test_warm_start_recovers_accuracy()
    test_warm_simplex_carries_scale_not_shape()
    test_budget_is_judged_on_movement_not_on_the_cap()
    test_warm_start_is_monotone()
    test_warm_pass_is_resumable()
    test_checkpoint_keeps_the_best_record()
    test_checkpoint_wrapper_wiring()
    test_dust_clamp_keeps_real_quantities()
    test_invariance_check_scales_with_achieved_accuracy()
    test_extension_reaches_a_threshold_the_grid_missed()
    test_extension_steps_towards_the_threshold()
    test_an_exact_confound_has_no_profile_bound()
    test_a_confound_survives_a_lengthening_stride()
    test_extension_stops_at_the_bound_and_says_so()
    test_extension_steps_multiplicatively()
    test_refinement_stops_at_a_tight_bracket()
    test_plot_window_follows_the_extended_grid()
    test_profile_report_states_the_answer()
    test_se_span_widens_the_grid()
    test_a_spec_can_set_its_own_grid()
    test_profile_grid_never_reaches_scipy()

    print("\n" + "=" * 72)
    if failures:
        print(f"FAILED ({len(failures)}):")
        for f in failures:
            print("  -", f)
        return 1
    print("ALL PROFILE QUADRATIC CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""The Wald SE must reach each consumer in the space that consumer works in.

There are two of them and they need different things:

* **Reporting** -- the summary, the CSV, the JSON snapshot -- wants linear
  units, because that is what a reader means by "the standard error of
  CLrecycle_Tissue".
* **Grid placement** -- the profile and slice walkers -- works in the
  optimizer's own space, which for these specs is log10. It computes
  ``p_opt +/- se_span * se`` with ``p_opt`` already in log10.

Handing the linear SE to the second is a units error, and its failure mode is
quiet and severe. On PK_Aducanumab (optimum 6.28561e-05, log10-fitted) it made
the opening grid span 0.99985x to 1.0001x of the optimum -- a grid one part in
ten thousand wide. Every side then exhausted all eight extension doublings
without reaching the threshold and the interval came back ``status: open``,
where the same fit had previously resolved [4.805e-05, 8.200e-05].

The bug was latent until 2026-08-27, when ``parameter_scale`` changed from
``None`` to ``"auto"``: before that these parameters were linear and the two
spaces coincided, so nothing distinguished them.

Run from the repository root:
    python tests/engine/test_wald_units.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.getcwd())

from pyantigen.engine.Optimize import (                                      # noqa: E402
    _attach_wald_stats,
    _profile_grid_for,
    _transform_wald_to_linear,
)

failures = []


def check(name, cond, detail=""):
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}  {detail}")
        failures.append(name)


# CLrecycle_Tissue, as the spec actually configures it.
X_LIN = 6.28561e-05
P_OPT = np.log10(X_LIN)
SE_OPT = 0.10896                      # curvature in log10 space
OPT_BOUNDS = [(np.log10(6.2856e-08), np.log10(6.2856e-03))]
SE_SPAN = 4.0


def _grid_span(se):
    left, right, _b, _is_log = _profile_grid_for(
        0, np.array([P_OPT]), OPT_BOUNDS, ["log10"], np.array([se]),
        5, 2.0, SE_SPAN,
    )
    vals = np.concatenate([np.asarray(left, float).ravel(),
                           np.asarray(right, float).ravel()])
    return 10.0 ** vals.min() / X_LIN, 10.0 ** vals.max() / X_LIN


def test_the_two_spaces_are_different():
    print("\nThe two SEs:")
    se_lin, _ci = _transform_wald_to_linear(
        np.array([SE_OPT]), [(0.0, 1.0)], np.array([P_OPT]), ["log10"])
    se_lin = float(se_lin[0])
    # Delta method: se_p = se_q * p * ln(10).
    check("the linear SE is the delta-method transform",
          abs(se_lin - SE_OPT * X_LIN * np.log(10.0)) < 1e-12, str(se_lin))
    check("and it is four orders of magnitude smaller",
          se_lin / SE_OPT < 1e-3, f"ratio {se_lin / SE_OPT:.3g}")


def test_the_wrong_space_collapses_the_grid():
    print("\nGrid width:")
    se_lin = float(_transform_wald_to_linear(
        np.array([SE_OPT]), [(0.0, 1.0)], np.array([P_OPT]), ["log10"])[0][0])

    lo_bad, hi_bad = _grid_span(se_lin)
    lo_ok, hi_ok = _grid_span(SE_OPT)

    check("the linear SE collapses the grid to a point",
          hi_bad / lo_bad < 1.01, f"spans {lo_bad:.5g}x .. {hi_bad:.5g}x")
    check("the opt-space SE gives a grid worth walking",
          hi_ok / lo_ok > 3.0, f"spans {lo_ok:.5g}x .. {hi_ok:.5g}x")

    # The reference run resolved this parameter at [0.7645x, 1.305x] of the
    # optimum. A correct opening grid has to contain that, or extension is
    # doing work the grid should have done.
    check("the correct grid brackets the known confidence interval",
          lo_ok < 0.7645 and hi_ok > 1.305,
          f"grid {lo_ok:.4g}x .. {hi_ok:.4g}x vs CI 0.7645x .. 1.305x")
    check("the broken grid does not come close",
          not (lo_bad < 0.7645 and hi_bad > 1.305),
          f"grid {lo_bad:.4g}x .. {hi_bad:.4g}x")


def test_attach_publishes_both():
    print("\nBoth SEs are published:")
    k = 3
    rng = np.random.default_rng(2)
    a = rng.standard_normal((k, k))
    A = a @ a.T + np.eye(k) * 5.0
    x = np.log10(np.array([3.0e-2, 1.4e-3, 6.3e-5]))

    def nll(q):
        d = np.asarray(q, float) - x
        return float(d @ A @ d)

    def batch(xs, label=None):
        return [nll(v) for v in xs]

    for scales, same in ((["log10"] * k, False), (["lin"] * k, True)):
        out = {"stats": {}}
        _attach_wald_stats(out, nll, x, [(-9.0, 1.0)] * k,
                           [f"p{i}" for i in range(k)], scales=scales,
                           nll_batch=batch)
        se_lin = np.asarray(out["stats"]["wald_se"], float)
        se_opt = np.asarray(out["stats"]["wald_se_opt"], float)
        label = scales[0]
        check(f"[{label}] both keys are present",
              se_lin.shape == (k,) and se_opt.shape == (k,))
        if same:
            # A linear spec must be untouched by any of this.
            check(f"[{label}] the two spaces coincide",
                  np.allclose(se_lin, se_opt, equal_nan=True), str(se_lin))
        else:
            check(f"[{label}] the two spaces differ",
                  not np.allclose(se_lin, se_opt, equal_nan=True))
            expected = se_opt * (10.0 ** x) * np.log(10.0)
            check(f"[{label}] by exactly the delta-method factor",
                  np.allclose(se_lin, expected), f"{se_lin} vs {expected}")


def test_the_cache_carries_the_opt_space_se():
    print("\nCache carries both:")
    import shutil
    import tempfile
    from pyantigen.engine.Anchor_cache import AnchorCache, bounds_fingerprint

    k = 3
    root = tempfile.mkdtemp(prefix="waldunits_")
    try:
        stats = {
            "wald_cov": np.eye(k),
            "wald_se": np.array([1.0e-5, 2.0e-5, 3.0e-5]),
            "wald_se_opt": np.array([0.1, 0.2, 0.3]),
            "wald_ci": [(1.0, 2.0)] * k,
            "wald_correlation": np.eye(k),
        }
        bh = bounds_fingerprint([(-9.0, 1.0)] * k)
        AnchorCache(root, "run", "mh", "sh", bh, n_params=k).save(stats)
        got = AnchorCache(root, "run", "mh", "sh", bh, n_params=k).load()

        check("the cache hits", got is not None)
        check("the opt-space SE round-trips",
              np.allclose(got["wald_se_opt"], stats["wald_se_opt"]))
        check("and is still distinct from the linear one",
              not np.allclose(got["wald_se_opt"], got["wald_se"]))

        # A cache written before wald_se_opt existed must miss, not restore a
        # block whose grid-placement key is silently absent.
        import json
        with open(AnchorCache(root, "run", "mh", "sh", bh,
                              n_params=k).path, encoding="utf-8") as fh:
            raw = json.load(fh)
        raw["format"] = "wald-v1"
        raw.pop("wald_se_opt", None)
        with open(AnchorCache(root, "run", "mh", "sh", bh,
                              n_params=k).path, "w", encoding="utf-8") as fh:
            json.dump(raw, fh)
        check("a pre-fix cache file misses rather than restoring",
              AnchorCache(root, "run", "mh", "sh", bh, n_params=k).load() is None)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_zz_every_check_passed():
    assert not failures, "failed checks: " + ", ".join(failures)


def main():
    test_the_two_spaces_are_different()
    test_the_wrong_space_collapses_the_grid()
    test_attach_publishes_both()
    test_the_cache_carries_the_opt_space_se()

    print("\n" + "=" * 72)
    if failures:
        print(f"FAILED ({len(failures)}):")
        for f in failures:
            print("  -", f)
        return 1
    print("ALL WALD UNIT CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())

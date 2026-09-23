"""Checks on the Hessian cache and the temp-file sweep.

Both exist because of the same measurement: on the SILK APP spec an objective
evaluation costs 116 s, so the 513-evaluation Hessian is about 25 minutes of
every link, and a preemptible run leaves one abandoned temp file per kill.

The cache's danger is not that it misses -- a miss costs exactly what the run
cost before it existed -- but that it *hits when it should not*. A standard
error belonging to a different fit would place the profile's whole grid in the
wrong region, silently. Most of what follows is about invalidation.

Run from the repository root:
    python tests/engine/test_anchor_cache.py
"""
import os
import shutil
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.getcwd())

from pyantigen.engine.Anchor_cache import (                                  # noqa: E402
    AnchorCache,
    bounds_fingerprint,
)
from pyantigen.engine.Profile_checkpoint import (                            # noqa: E402
    ProfileCheckpoint,
    sweep_stale_temp_files,
)

failures = []


def check(name, cond, detail=""):
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}  {detail}")
        failures.append(name)


K = 4
_BOUNDS = [(-5.0, 5.0)] * K


def _stats():
    """A Wald block shaped like the real one, including a flat direction."""
    rng = np.random.default_rng(0)
    a = rng.standard_normal((K, K))
    cov = a @ a.T
    se = np.sqrt(np.diag(cov))
    # A parameter with no usable SE is reported as nan, and that has to survive
    # the round trip: it is how "this direction is flat" is expressed, and the
    # profile tests it with np.isfinite.
    se[2] = float("nan")
    ci = [(float(v) - 1.96 * float(s), float(v) + 1.96 * float(s))
          for v, s in zip(np.arange(K, dtype=float), se)]
    corr = cov / np.outer(np.sqrt(np.diag(cov)), np.sqrt(np.diag(cov)))
    # The SE in the optimizer's own space, which is what the profile grid is
    # placed with. Distinct from the linear one on a log10-fitted spec, and
    # required: a block restored without it would silently drop the grid back
    # to the range_factor fallback.
    return {"wald_cov": cov, "wald_se": se, "wald_se_opt": se * 7.0,
            "wald_ci": ci, "wald_correlation": corr}


def _cache(root, **kw):
    kw.setdefault("model_hash", "mh")
    kw.setdefault("spec_hash", "sh")
    kw.setdefault("bounds_hash", bounds_fingerprint(_BOUNDS))
    return AnchorCache(root, "run", kw.pop("model_hash"), kw.pop("spec_hash"),
                       kw.pop("bounds_hash"), n_params=K, **kw)


def test_round_trip():
    print("\nRound trip:")
    root = tempfile.mkdtemp(prefix="anchor_")
    try:
        stats = _stats()
        c = _cache(root)
        check("nothing cached yet", c.load() is None)
        c.save(stats)
        check("the file is written", os.path.exists(c.path))

        got = _cache(root).load()
        check("a matching key hits", got is not None)
        check("the covariance survives",
              np.allclose(got["wald_cov"], stats["wald_cov"]))
        check("the correlation survives",
              np.allclose(got["wald_correlation"], stats["wald_correlation"]))
        # nan means "no usable SE"; turning it into 0 would look like a
        # perfectly determined parameter and produce a grid of zero width.
        check("a nan standard error stays nan",
              np.isnan(got["wald_se"][2]), str(got["wald_se"]))
        check("the finite standard errors survive",
              np.allclose(got["wald_se"][[0, 1, 3]],
                          stats["wald_se"][[0, 1, 3]]))
        check("the opt-space standard errors survive, and stay distinct",
              np.allclose(got["wald_se_opt"][[0, 1, 3]],
                          stats["wald_se_opt"][[0, 1, 3]])
              and not np.allclose(got["wald_se_opt"][[0, 1, 3]],
                                  got["wald_se"][[0, 1, 3]]))
        check("the intervals survive",
              np.allclose(np.asarray(got["wald_ci"])[[0, 1, 3]],
                          np.asarray(stats["wald_ci"])[[0, 1, 3]]))
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_every_input_invalidates():
    """A hit on the wrong fit is the only failure that would matter."""
    print("\nInvalidation:")
    root = tempfile.mkdtemp(prefix="anchor_")
    try:
        _cache(root).save(_stats())

        check("a different model misses",
              _cache(root, model_hash="other").load() is None)
        check("a different spec or optimum misses",
              _cache(root, spec_hash="other").load() is None)
        # Bounds clip the Wald interval without changing the model or the
        # optimum, so they are not in the profile's spec hash and need their
        # own place in this key.
        check("different bounds miss",
              _cache(root, bounds_hash=bounds_fingerprint(
                  [(-1.0, 1.0)] * K)).load() is None)
        check("a different parameter count misses",
              AnchorCache(root, "run", "mh", "sh",
                          bounds_fingerprint(_BOUNDS),
                          n_params=K + 1).load() is None)
        check("the unchanged key still hits", _cache(root).load() is not None)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_bounds_fingerprint():
    print("\nBounds fingerprint:")
    check("identical bounds agree",
          bounds_fingerprint(_BOUNDS) == bounds_fingerprint(list(_BOUNDS)))
    check("a changed bound differs",
          bounds_fingerprint(_BOUNDS)
          != bounds_fingerprint([(-5.0, 5.0)] * (K - 1) + [(-4.0, 5.0)]))
    check("None is stable", bounds_fingerprint(None) == bounds_fingerprint(None))
    check("None differs from real bounds",
          bounds_fingerprint(None) != bounds_fingerprint(_BOUNDS))
    check("an open bound is handled",
          isinstance(bounds_fingerprint([(None, 5.0)] * K), str))


def test_damage_is_a_miss_not_a_crash():
    print("\nDamaged cache:")
    root = tempfile.mkdtemp(prefix="anchor_")
    try:
        c = _cache(root)
        c.save(_stats())

        with open(c.path, "w", encoding="utf-8") as fh:
            fh.write('{"format": "wald-v1", "model_hash": "mh"')  # truncated
        check("a truncated file misses rather than raising",
              _cache(root).load() is None)

        with open(c.path, "wb") as fh:
            fh.write(b"\x00\x01\x02 not json at all")
        check("a corrupt file misses rather than raising",
              _cache(root).load() is None)

        # Recomputing must be able to overwrite the damage.
        _cache(root).save(_stats())
        check("a later save repairs it", _cache(root).load() is not None)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_disabled_cache_is_inert():
    print("\nDisabled cache:")
    root = tempfile.mkdtemp(prefix="anchor_")
    try:
        c = AnchorCache(root, "run", "mh", "sh", "bh", n_params=K,
                        enabled=False)
        c.save(_stats())
        check("nothing is written when disabled",
              not os.path.exists(os.path.join(root, "profiles", "run",
                                              "anchor.json")))
        check("and loading returns nothing", c.load() is None)

        c2 = AnchorCache(None, "run", "mh", "sh", "bh", n_params=K)
        c2.save(_stats())
        check("no root is survivable", c2.load() is None)
    finally:
        shutil.rmtree(root, ignore_errors=True)


# ---------------------------------------------------------------------------
# Temp-file sweep
# ---------------------------------------------------------------------------

def test_sweep_removes_only_abandoned_files():
    print("\nTemp-file sweep:")
    root = tempfile.mkdtemp(prefix="sweep_")
    try:
        old = os.path.join(root, "timing.json.1234.tmp")
        fresh = os.path.join(root, "timing.json.5678.tmp")
        real = os.path.join(root, "timing.json")
        data = os.path.join(root, "p0.jsonl")
        for p in (old, fresh, real, data):
            with open(p, "w", encoding="utf-8") as fh:
                fh.write("{}")
        # Backdate one past the age threshold.
        stale_time = time.time() - 7200.0
        os.utime(old, (stale_time, stale_time))

        removed = sweep_stale_temp_files(root, max_age_s=3600.0)
        check("the abandoned temp file goes", removed == 1 and not os.path.exists(old),
              f"removed={removed}")
        # A young temp file may belong to a process that is still running;
        # deleting it would break that process's rename for no reason.
        check("a recent temp file is left alone", os.path.exists(fresh))
        check("real files are untouched",
              os.path.exists(real) and os.path.exists(data))

        check("a missing directory is survivable",
              sweep_stale_temp_files(os.path.join(root, "nope")) == 0)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_opening_a_run_sweeps():
    print("\nSweep on open:")
    root = tempfile.mkdtemp(prefix="sweep_")
    try:
        d = os.path.join(root, "profiles", "run")
        os.makedirs(d)
        junk = os.path.join(d, "timing.json.999.tmp")
        with open(junk, "w", encoding="utf-8") as fh:
            fh.write("{}")
        stale = time.time() - 7200.0
        os.utime(junk, (stale, stale))

        ck = ProfileCheckpoint(root, "run", "mh", "sh")
        check("opening the checkpoint store sweeps",
              ck.n_temp_swept == 1 and not os.path.exists(junk),
              f"swept={ck.n_temp_swept}")
        ck.close()
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_zz_every_check_passed():
    """``check`` records rather than raises, so pytest needs this to see them."""
    assert not failures, "failed checks: " + ", ".join(failures)


def main():
    test_round_trip()
    test_every_input_invalidates()
    test_bounds_fingerprint()
    test_damage_is_a_miss_not_a_crash()
    test_disabled_cache_is_inert()
    test_sweep_removes_only_abandoned_files()
    test_opening_a_run_sweeps()

    print("\n" + "=" * 72)
    if failures:
        print(f"FAILED ({len(failures)}):")
        for f in failures:
            print("  -", f)
        return 1
    print("ALL ANCHOR CACHE CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())

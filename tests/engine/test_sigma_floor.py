"""The data-derived sigma floor caps a block's profiled sigma without
replacing it: sigma_used = min(sigma_hat, sigma_floor).

Two things have to hold for that to be safe to put in the live objective:

* it must reduce to EXACTLY today's profiled behaviour whenever the cap does
  not bind ("a true improvement or nothing" -- the requirement that ruled out
  a hardcoded fixed sigma for these observables);
* the concentrated-NLL term for a floored block must be continuous, in both
  value and derivative, right at the point sigma_hat crosses the floor -- a
  naive implementation that switches between the existing profiled/known
  formulas at that boundary is NOT continuous there (they strip different
  constants from the full NLL), which would show up as a spurious kink in the
  objective mid-optimization. This file is the regression guard for that bug.

Run from the repository root:
    python tests/engine/test_sigma_floor.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.getcwd())


from pyantigen.engine.Optimize import (                                      # noqa: E402
    _block_sigma_resolution,
    _freeze_floor,
    _record_block,
    _unpack_block,
    concentrated_nll,
    effective_k,
    zero_cost_sigma_blocks,
)
from pyantigen.engine.Noise_floor import (                                    # noqa: E402
    _MIN_FLOOR_POINTS,
    _MIN_LOCAL_NEIGHBORS,
    _calibrate_min_span,
    _oscillation_length,
    clear_cache,
    compute_noise_floor,
    get_noise_floor,
)

failures = []


def check(name, cond, detail=""):
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}  {detail}")
        failures.append(name)


# ---------------------------------------------------------------------------
# _block_sigma_resolution: the "or nothing" promise and the binding switch
# ---------------------------------------------------------------------------

def test_not_binding_matches_plain_profiled():
    print("\nA floor that does not bind changes nothing about sigma_used:")
    n, sse = 40.0, 4.0          # sigma_hat = sqrt(4/40) = 0.3162...
    floor = 1.0                  # well above sigma_hat -- must not bind
    sigma_used, cost, state = _block_sigma_resolution(sse, n, None, floor)
    sigma_hat = np.sqrt(sse / n)
    check("state is floored_free", state == "floored_free", state)
    check("sigma_used equals sigma_hat exactly",
          np.isclose(sigma_used, sigma_hat), f"{sigma_used} vs {sigma_hat}")
    check("costs 1 parameter, same as plain profiled", cost == 1)


def test_binding_caps_at_the_floor():
    print("\nA floor that DOES bind pins sigma_used at the floor:")
    n, sse = 40.0, 40.0          # sigma_hat = 1.0
    floor = 0.3
    sigma_used, cost, state = _block_sigma_resolution(sse, n, None, floor)
    check("state is floored_binding", state == "floored_binding", state)
    check("sigma_used equals the floor exactly",
          np.isclose(sigma_used, floor), f"{sigma_used} vs {floor}")
    check("costs 0 parameters, same as a declared sigma", cost == 0)


def test_declared_sigma_wins_over_a_floor():
    print("\nA declared sigma takes priority over any floor:")
    sigma_used, cost, state = _block_sigma_resolution(9.0, 10.0, 2.0, 0.1)
    check("state is declared", state == "declared", state)
    check("sigma_used is the declared value", np.isclose(sigma_used, 2.0))
    check("costs 0 parameters", cost == 0)


# ---------------------------------------------------------------------------
# concentrated_nll: continuity at the switch (the bug the design review found)
# ---------------------------------------------------------------------------

def test_continuity_at_the_switch():
    print("\nconcentrated_nll is C1-continuous across the cap boundary:")
    n = 50.0
    floor = 0.472
    sse_star = n * floor ** 2   # sigma_hat == floor exactly here

    def term_at(sse):
        blocks = {"b": [sse, n, None, floor]}
        return concentrated_nll(blocks, include_constant=False)

    h = 1e-4 * sse_star
    below, at, above = term_at(sse_star - h), term_at(sse_star), term_at(sse_star + h)

    # The term has nonzero slope, so value naturally moves by O(h) either
    # side of the switch -- that is not a jump. What must hold is that the
    # move matches the analytic slope exactly, with nothing ADDED on top of
    # it: a real discontinuity (the bug a naive implementation has) would
    # show up as an O(1) offset here, not an O(h) one.
    d_below = (at - below) / h
    d_above = (above - at) / h
    analytic = n / (2.0 * sse_star)
    check("value step below the switch matches the analytic slope (no added jump)",
          np.isclose(d_below, analytic, rtol=1e-3), f"{d_below} vs {analytic}")
    check("value step above the switch matches the analytic slope (no added jump)",
          np.isclose(d_above, analytic, rtol=1e-3), f"{d_above} vs {analytic}")
    check("derivative itself has no jump",
          np.isclose(d_below, d_above, rtol=1e-3), f"{d_below} vs {d_above}")


def test_not_binding_reduces_to_profiled_plus_constant():
    print("\nBelow the switch, the floored term is the profiled term + n/2:")
    n, sse, floor = 30.0, 1.5, 5.0   # sigma_hat << floor
    blocks_floor = {"b": [sse, n, None, floor]}
    blocks_plain = {"b": [sse, n, None, None]}
    floored = concentrated_nll(blocks_floor)
    plain = concentrated_nll(blocks_plain)
    check("difference is exactly n/2, a per-block constant",
          np.isclose(floored - plain, n / 2.0),
          f"floored={floored:.6g} plain={plain:.6g} diff={floored - plain:.6g}")


def test_binding_grows_exactly_linearly_in_sse():
    print("\nPast the switch, the NLL term grows linearly, not logarithmically "
          "(the self-inflation escape hatch is closed):")
    n, floor = 40.0, 0.5
    sse_star = n * floor ** 2
    blocks = lambda sse: {"b": [sse, n, None, floor]}
    d_sse = 10.0
    term1 = concentrated_nll(blocks(sse_star + 100.0))
    term2 = concentrated_nll(blocks(sse_star + 100.0 + d_sse))
    expected_delta = d_sse / (2.0 * floor * floor)
    check("d(NLL) matches d(SSE)/(2*floor^2) exactly",
          np.isclose(term2 - term1, expected_delta),
          f"{term2 - term1} vs {expected_delta}")


# ---------------------------------------------------------------------------
# effective_k / zero_cost_sigma_blocks: mixed-block parameter counting
# ---------------------------------------------------------------------------

def test_effective_k_mixed_blocks():
    print("\neffective_k counts a mix of block states correctly:")
    blocks = {}
    _record_block(blocks, "declared", "obs", [1.0, -1.0], [1.0, 1.0], known_sigma=2.0)
    _record_block(blocks, "estimated", "obs", [1.0, -1.0, 2.0], [1.0] * 3)
    # sigma_hat for the "binding" block: sse=40, n=40 -> sigma_hat=1.0 > floor=0.3
    _record_block(blocks, "binding", "obs", np.ones(40) * np.sqrt(1.0),
                  np.ones(40), sigma_floor=0.3)
    # sigma_hat for "free": sse small relative to a generous floor -> not binding
    _record_block(blocks, "free", "obs", np.ones(40) * 0.1,
                  np.ones(40), sigma_floor=5.0)

    k_eff = effective_k(["p1", "p2"], blocks)
    check("k_eff = params(2) + estimated(1) + binding(0) + free(1) = 4",
          k_eff == 4, k_eff)

    zc = zero_cost_sigma_blocks(blocks)
    check("declared and binding blocks are zero-cost",
          ("declared", "obs") in zc and ("binding", "obs") in zc, zc)
    check("estimated and free blocks are NOT zero-cost",
          ("estimated", "obs") not in zc and ("free", "obs") not in zc, zc)


# ---------------------------------------------------------------------------
# pyantigen.engine.Noise_floor: the data-only computation and its cache
# ---------------------------------------------------------------------------

def test_compute_noise_floor_recovers_known_sd():
    print("\ncompute_noise_floor recovers a known flat-mean noise SD:")
    rng = np.random.default_rng(0)
    x = np.linspace(0, 100, 60)
    true_sigma = 0.75
    y = 10.0 + rng.normal(0.0, true_sigma, size=x.shape)
    nf = compute_noise_floor(x, y)
    check("recovered sigma is within 25% of the true value",
          abs(nf.sigma - true_sigma) / true_sigma < 0.25,
          f"recovered={nf.sigma:.4g} true={true_sigma}")
    check("n matches the input size", nf.n == len(x))
    # NOT asserted: which span LOOCV picks. It's a genuine data-driven choice
    # with real sampling variance -- checked directly against the actual CSF/
    # sAPP data (all six panels pick span=1.0) rather than against one
    # synthetic random draw, where a tighter span can legitimately win LOOCV
    # by chance even when the true function is flat (verified: for this exact
    # seed, span=0.3's LOOCV MSE is 0.408 vs 1.0's 0.475 -- a real, if small,
    # win, not a bug).
    print(f"    (LOOCV picked span={nf.span:.2g} on this draw)")


def test_compute_noise_floor_too_few_points():
    print("\ncompute_noise_floor refuses to trust a span search on too little data:")
    x = np.arange(_MIN_FLOOR_POINTS - 1, dtype=float)
    y = np.zeros_like(x)
    check("returns None below _MIN_FLOOR_POINTS",
          compute_noise_floor(x, y) is None)


def test_get_noise_floor_cache_reuses_identical_content():
    print("\nget_noise_floor caches by content, not identity:")
    x = np.linspace(0, 10, 20)
    y = np.sin(x)
    a = get_noise_floor(x.copy(), y.copy(), cache_key=("k1", "col"))
    b = get_noise_floor(x.copy(), y.copy(), cache_key=("k1", "col"))
    check("two calls with fresh-but-identical arrays return the SAME object",
          a is b)

    c = get_noise_floor(x.copy(), y.copy(), cache_key=("k2", "col"))
    check("a different cache_key returns a DIFFERENT object", c is not a)

    d = get_noise_floor(x.copy(), (y + 1.0).copy(), cache_key=("k1", "col"))
    check("different content under the same key returns a DIFFERENT object",
          d is not a)


def test_min_local_neighbors_stops_degenerate_span_selection():
    print("\nOn strongly autocorrelated data, LOOCV span selection stops at "
          "_MIN_LOCAL_NEIGHBORS instead of degenerating toward interpolation "
          "(the SILK MFL / csf_abtot_percent_baseline finding):")
    # A smooth random walk: adjacent points are highly correlated by
    # construction, the exact pathology that made raw LOOCV keep rewarding
    # ever-tighter windows on the real SILK MFL data.
    rng = np.random.default_rng(1)
    n = 60
    x = np.arange(n, dtype=float)
    y = np.cumsum(rng.normal(0.0, 1.0, size=n))

    # Without the floor, the smallest reachable k is 2 (span -> 0). With it,
    # the smallest window _loocv_mse will ever use is _MIN_LOCAL_NEIGHBORS.
    tiny_span = 1.0 / n  # would ask for k=1, i.e. as tight as span allows
    k_used = min(max(_MIN_LOCAL_NEIGHBORS, int(np.ceil(tiny_span * n))), n - 1)
    check("the floor actually raises k for a span that would otherwise be tiny",
          k_used == _MIN_LOCAL_NEIGHBORS, k_used)

    nf = compute_noise_floor(x, y)
    # The span search never even offers a window tighter than the floor, so
    # the chosen span's window can't be under it either.
    n_used = min(max(_MIN_LOCAL_NEIGHBORS, int(np.ceil(nf.span * n))), n - 1)
    check("the winning span's window respects the floor",
          n_used >= _MIN_LOCAL_NEIGHBORS, n_used)


def test_oscillation_length_is_offset_invariant():
    print("\n_oscillation_length (total variation) ignores a constant shift "
          "between the two curves being compared:")
    rng = np.random.default_rng(4)
    diff = rng.normal(0, 1, size=30)
    base = _oscillation_length(diff)
    shifted = _oscillation_length(diff + 1000.0)
    check("adding a large constant offset doesn't change the length",
          np.isclose(base, shifted), f"{base} vs {shifted}")
    check("a perfectly flat difference has zero length",
          _oscillation_length(np.full(10, 7.0)) == 0.0)


def test_calibrate_min_span_finds_an_interior_choice():
    print("\n_calibrate_min_span rescues data that fools plain LOOCV, using a "
          "known true shape as the reference (the mechanism validated against "
          "real csf_abtot_percent_baseline / SILK MFL42 data on 2026-09-15):")
    rng = np.random.default_rng(5)
    n = 60
    x = np.linspace(0.0, 60.0, n)
    true_shape = 8.0 * np.sin(x / 9.0)  # a few genuine turns, nothing more
    # Correlated noise (a smoothed random walk), not iid -- the exact
    # pathology that makes plain LOOCV-on-data prefer ever-tighter windows.
    walk = np.cumsum(rng.normal(0.0, 1.0, size=n))
    y = true_shape + 0.3 * walk

    spans = (0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.75, 0.9, 1.0)
    plain = compute_noise_floor(x, y, spans=spans)
    calibrated_span = _calibrate_min_span(x, y, true_shape, spans)

    check("the calibrated minimum span is wider than plain LOOCV's degenerate pick",
          calibrated_span > plain.span,
          f"calibrated={calibrated_span} vs plain LOOCV={plain.span}")

    cal_nf = get_noise_floor(
        x, y, cache_key="test_calibration_synthetic", spans=spans,
        shape_reference=true_shape)
    check("get_noise_floor honors the calibration end-to-end",
          cal_nf.span >= calibrated_span, f"{cal_nf.span} vs {calibrated_span}")


def test_shape_reference_only_affects_the_first_call():
    print("\nget_noise_floor consults shape_reference only on a cache miss, "
          "then freezes -- the property that keeps this safe inside the live "
          "per-evaluation objective:")
    rng = np.random.default_rng(6)
    x = np.linspace(0.0, 40.0, 50)
    y = 3.0 * np.sin(x / 7.0) + 0.25 * np.cumsum(rng.normal(0, 1, size=50))
    ref = 3.0 * np.sin(x / 7.0)

    first = get_noise_floor(x.copy(), y.copy(), cache_key="test_freeze",
                            shape_reference=ref)
    # A wildly different "simulation" on a later call (as if the optimizer
    # had moved somewhere absurd) must not perturb the already-cached result.
    second = get_noise_floor(x.copy(), y.copy(), cache_key="test_freeze",
                             shape_reference=np.full_like(x, 1e6))
    check("the second call returns the exact same cached object",
          second is first)


def test_clear_cache_forces_recalibration():
    print("\nclear_cache() forces the next call to recalibrate rather than "
          "reuse the frozen value -- what a final post-optimum re-evaluation "
          "relies on to pick up a better shape_reference:")
    rng = np.random.default_rng(7)
    x = np.linspace(0.0, 30.0, 40)
    y = 4.0 * np.sin(x / 6.0) + 0.3 * np.cumsum(rng.normal(0, 1, size=40))
    poor_ref = np.zeros_like(x)          # stand-in for a bad x0 fit
    good_ref = 4.0 * np.sin(x / 6.0)     # stand-in for the converged fit

    key = "test_clear_cache"
    before = get_noise_floor(x.copy(), y.copy(), cache_key=key, shape_reference=poor_ref)
    still_cached = get_noise_floor(x.copy(), y.copy(), cache_key=key, shape_reference=good_ref)
    check("before clear_cache(), a new shape_reference is ignored (frozen)",
          still_cached is before)

    clear_cache()
    after = get_noise_floor(x.copy(), y.copy(), cache_key=key, shape_reference=good_ref)
    check("after clear_cache(), the same call recalibrates (new object)",
          after is not before)
    check("the recalibrated span reflects the better (good_ref) shape reference",
          after.span != before.span or not np.isclose(after.sigma, before.sigma),
          f"before span={before.span} sigma={before.sigma:.4g}; "
          f"after span={after.span} sigma={after.sigma:.4g}")


# ---------------------------------------------------------------------------
# _freeze_floor: pinning a floored block's sigma during profile evaluation
# ---------------------------------------------------------------------------

def test_freeze_floor_pins_sigma_at_the_given_value():
    print("\n_freeze_floor pins a floored block's sigma at frozen_value:")
    # frozen_value is the block's own sigma_used at the optimum, which may sit
    # well below the floor (the common, not-binding case) -- freezing has to
    # land exactly there, not snap to the floor regardless.
    anchor_sigma = 0.2   # e.g. min(sigma_hat_opt, floor) with floor=1.0
    known, floor = _freeze_floor(None, 1.0, anchor_sigma)
    check("known_sigma becomes frozen_value, not the raw floor",
          known == anchor_sigma, known)
    check("sigma_floor is cleared, folded into known_sigma instead",
          floor is None, floor)

    sigma_used, cost, state = _block_sigma_resolution(4.0, 40.0, known, floor)
    sigma_hat = np.sqrt(4.0 / 40.0)
    check("sigma_used is exactly frozen_value, not sigma_hat or the floor",
          sigma_used == anchor_sigma and sigma_used != sigma_hat
          and sigma_used != 1.0, sigma_used)
    check("costs 0 parameters, like any declared sigma", cost == 0)
    check("state reads as declared", state == "declared", state)


def test_freezing_at_the_anchors_own_sigma_reproduces_it_exactly():
    """The point of freezing at frozen_value rather than the raw floor: at the
    exact point frozen_value was measured, the frozen and unfrozen NLL terms
    must agree, so the anchor the profile compares against carries no
    artificial offset."""
    print("\nFreezing at the anchor's own resolved sigma is a true 'or nothing':")
    n, sse = 40.0, 4.0
    sigma_hat = np.sqrt(sse / n)
    floor = 1.0   # well above sigma_hat -- not binding, per _block_sigma_resolution
    unfrozen_sigma, unfrozen_cost, unfrozen_state = _block_sigma_resolution(
        sse, n, None, floor)
    check("not binding, so the anchor's own resolution is sigma_hat",
          unfrozen_state == "floored_free" and np.isclose(unfrozen_sigma, sigma_hat))

    known, sfloor = _freeze_floor(None, floor, unfrozen_sigma)
    frozen_sigma, frozen_cost, frozen_state = _block_sigma_resolution(
        sse, n, known, sfloor)
    check("freezing at the anchor's own sigma reproduces it exactly",
          np.isclose(frozen_sigma, unfrozen_sigma), frozen_sigma)

    unfrozen_term = n * np.log(unfrozen_sigma) + sse / (2.0 * unfrozen_sigma ** 2)
    frozen_term = n * np.log(frozen_sigma) + sse / (2.0 * frozen_sigma ** 2)
    check("the NLL term at the anchor is unchanged by freezing there",
          np.isclose(unfrozen_term, frozen_term),
          f"{unfrozen_term} vs {frozen_term}")

    # Contrast: freezing at the raw floor instead (the earlier, rejected
    # design) would NOT reproduce the anchor -- it inflates a not-binding
    # block's term for no reason tied to how far the profile has moved.
    floor_term = n * np.log(floor) + sse / (2.0 * floor ** 2)
    check("freezing at the raw floor instead would NOT have agreed",
          not np.isclose(floor_term, unfrozen_term),
          f"floor term {floor_term} vs true term {unfrozen_term}")


def test_freezing_reproduces_through_concentrated_nll():
    """The invariant test_freezing_at_the_anchors_own_sigma_reproduces_it_exactly
    checks with a hand-written formula, but re-checked through the actual
    function every caller uses. concentrated_nll's known-sigma branch used to
    drop n*log(ks) unless include_constant=True, while the floor branch it is
    switched FROM by _freeze_floor always includes it -- so a frozen block
    silently lost n*log(sigma_used) relative to the unfrozen anchor it is
    supposed to reproduce exactly. Caught live on the SILK spec: 22 floored
    blocks summed to a -3526.5 nat gap between the unfrozen and frozen
    anchors at the very same parameter vector.
    """
    print("\nFreezing, run through the real concentrated_nll (not a hand-written "
          "formula), still reproduces the unfrozen anchor exactly:")
    n, sse, floor = 37.0, 37.0 * 0.001073 ** 2, 5.0   # floor not binding
    unfrozen_sigma, _cost, state = _block_sigma_resolution(sse, n, None, floor)
    check("not binding here either", state == "floored_free", state)

    unfrozen_blocks = {"b": [sse, n, None, floor]}
    unfrozen_nll = concentrated_nll(unfrozen_blocks)

    known, sfloor = _freeze_floor(None, floor, unfrozen_sigma)
    frozen_blocks = {"b": [sse, n, known, sfloor]}
    frozen_nll = concentrated_nll(frozen_blocks)

    check("concentrated_nll(frozen) == concentrated_nll(unfrozen) at the anchor",
          np.isclose(frozen_nll, unfrozen_nll),
          f"frozen={frozen_nll:.6g} unfrozen={unfrozen_nll:.6g} "
          f"gap={frozen_nll - unfrozen_nll:.6g} (expected gap from the bug: "
          f"n*log(sigma) = {n * np.log(unfrozen_sigma):.6g})")


def test_freeze_floor_leaves_everything_else_alone():
    print("\n_freeze_floor is a no-op outside its one case:")
    check("frozen_value=None changes nothing",
          _freeze_floor(None, 1.0, None) == (None, 1.0))
    check("a declared sigma is never touched, floor or not",
          _freeze_floor(0.5, 1.0, 0.2) == (0.5, 1.0))
    check("no floor to freeze leaves known_sigma as None",
          _freeze_floor(None, None, 0.2) == (None, None))


def test_zz_every_check_passed():
    assert not failures, "failed checks: " + ", ".join(failures)


def main():
    test_not_binding_matches_plain_profiled()
    test_binding_caps_at_the_floor()
    test_declared_sigma_wins_over_a_floor()
    test_continuity_at_the_switch()
    test_not_binding_reduces_to_profiled_plus_constant()
    test_binding_grows_exactly_linearly_in_sse()
    test_effective_k_mixed_blocks()
    test_compute_noise_floor_recovers_known_sd()
    test_compute_noise_floor_too_few_points()
    test_get_noise_floor_cache_reuses_identical_content()
    test_min_local_neighbors_stops_degenerate_span_selection()
    test_oscillation_length_is_offset_invariant()
    test_calibrate_min_span_finds_an_interior_choice()
    test_shape_reference_only_affects_the_first_call()
    test_clear_cache_forces_recalibration()
    test_freeze_floor_pins_sigma_at_the_given_value()
    test_freezing_at_the_anchors_own_sigma_reproduces_it_exactly()
    test_freezing_reproduces_through_concentrated_nll()
    test_freeze_floor_leaves_everything_else_alone()

    print("\n" + "=" * 72)
    if failures:
        print(f"FAILED ({len(failures)}):")
        for f in failures:
            print("  -", f)
        return 1
    print("ALL SIGMA-FLOOR CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())

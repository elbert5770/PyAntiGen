"""The panel title's per-block term is the block's term in the objective (concentrated_nll), for every sigma state."""
import numpy as np
import pytest

from pyantigen.engine.Optimize import _block_title_stats, concentrated_nll

KEY = ("sim", "obs")
N, SSE = 54.0, 54.0 * 0.116 ** 2 * 0.98


def term(known=None, floor=None):
    blocks = {KEY: [SSE, N, known, floor]}
    n, sigma, t, tag = _block_title_stats({"block": KEY}, blocks)
    return t, tag, concentrated_nll(blocks)


def test_fixed_sigma_keeps_the_log_term():
    t, tag, total = term(known=0.116)
    assert tag == "sigma fixed"
    assert t == pytest.approx(N * np.log(0.116) + SSE / (2 * 0.116 ** 2))
    assert t == pytest.approx(total)


def test_floored_not_binding_matches_the_objective():
    t, tag, total = term(floor=0.2)
    assert tag == "sigma floor (not binding)"
    assert t == pytest.approx(total)


def test_floored_binding_matches_the_objective():
    t, tag, total = term(floor=0.05)
    assert tag == "sigma floor (capped)"
    assert t == pytest.approx(total)


def test_fixed_and_unbinding_floor_differ_only_by_the_sigma_used():
    # the same data scored with sigma 0.116 fixed and with the fitted sigma_hat differ by the sigma terms, not by a dropped n*log(sigma)
    fixed, _, _ = term(known=0.116)
    free, _, _ = term(floor=0.2)
    sigma_hat = np.sqrt(SSE / N)
    assert free - fixed == pytest.approx(N * np.log(sigma_hat) + N / 2 - N * np.log(0.116) - SSE / (2 * 0.116 ** 2))

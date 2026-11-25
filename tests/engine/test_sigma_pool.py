"""``"sigma_pool"``: several observables of one simulation share ONE estimated sigma.

Run from the repository root: python -m pytest tests/engine/test_sigma_pool.py

By default every observable is its own likelihood block with its own profiled sigma. An observable entry in a
loss_config may carry ``"sigma_pool": "<name>"``; entries with the same name fall into one block, whose sigma is
estimated from all their residuals together (PEtab's single noise parameter shared by every observable). The checks:
the block structure, the likelihood against its closed form, the analytic gradient against finite differences, the
composition with the element-level ``sigma_block``, and that the coverage report still sees every observable.
"""
import numpy as np
import pandas as pd
import pytest

from pyantigen.engine.Analytic_gradient import LossGradient, SimTable
from pyantigen.engine.Optimize import (
    block_sigmas, collect_loss_blocks, concentrated_nll, total_points)

T = np.linspace(0.2, 6.0, 12)
NAMES = ["a", "k"]
TRUE = np.array([2.0, 0.7])
COLUMNS = ["time", "p1", "p2"]


def table(theta, t):
    """p1 = a e^{-kt}, p2 = 0.3 a e^{-kt} (a second observable on a very different scale) and d/d(a, k)."""
    a, k = theta
    e = np.exp(-k * t)
    data = np.stack([t, a * e, 0.3 * a * e], axis=1)
    D = np.zeros((len(t), 3, 2))
    D[:, 1, :] = np.stack([e, -a * t * e], axis=-1)
    D[:, 2, :] = 0.3 * D[:, 1, :]
    return SimTable(COLUMNS, data), D


def data(seed=5):
    rng = np.random.default_rng(seed)
    tab = table(TRUE, T)[0].data
    return pd.DataFrame({"time": T,
                         "y1": tab[:, 1] * (1 + 0.05 * rng.standard_normal(len(T))),
                         "y2": tab[:, 2] + 0.01 * rng.standard_normal(len(T))})


def observables(pool1=None, pool2=None):
    out = []
    for col, y, pool in (("p1", "y1", pool1), ("p2", "y2", pool2)):
        o = {"observed_variable": col, "data_column": y, "time_column": "time"}
        if pool is not None:
            o["sigma_pool"] = pool
        out.append(o)
    return out


def setup(obs, sims=("s",), element_block=None):
    replicates = {s: {"Label": s} for s in sims}
    lc = {"observables": obs}
    elements = [{"simulation": s, "loss_config": lc, **({"sigma_block": element_block} if element_block else {})}
                for s in sims]
    groups = {"g": {"loss_elements": elements}}
    frames = {s: data(seed=5 + i) for i, s in enumerate(sims)}

    def items(theta):
        tab, D = table(theta, T)
        return ({s: {"results": tab, "data": frames[s], "replicate": replicates[s]} for s in sims},
                {s: D for s in sims})
    return groups, replicates, items, frames


def blocks_at(obs, theta=TRUE * [1.05, 0.95], **kw):
    groups, replicates, items, frames = setup(obs, **kw)
    it, _ = items(theta)
    p = dict(zip(NAMES, theta))
    return collect_loss_blocks(it, groups, replicates, NAMES, np.asarray(theta), p), frames, theta


# --- the block structure -------------------------------------------------------------------------

def test_without_a_pool_every_observable_has_its_own_block_and_sigma():
    blocks, _, _ = blocks_at(observables())
    assert sorted(blocks) == [("s", "p1"), ("s", "p2")]
    assert len(block_sigmas(blocks)) == 2


def test_a_shared_pool_is_one_block_with_all_the_residuals():
    blocks, frames, theta = blocks_at(observables("noise", "noise"))
    assert list(blocks) == [("s", "noise")]
    sse, n = blocks[("s", "noise")][:2]
    tab = table(theta, T)[0].data
    y = frames["s"]
    expected = ((y["y1"] - tab[:, 1]) ** 2).sum() + ((y["y2"] - tab[:, 2]) ** 2).sum()
    assert n == 2 * len(T)
    assert sse == pytest.approx(expected)
    assert total_points(blocks) == 2 * len(T)
    assert len(block_sigmas(blocks)) == 1


def test_different_pool_names_stay_separate_and_an_unpooled_observable_keeps_its_own_block():
    blocks, _, _ = blocks_at(observables("a", "b"))
    assert sorted(blocks) == [("s", "a"), ("s", "b")]
    blocks, _, _ = blocks_at(observables("only-p1", None))
    assert sorted(blocks) == [("s", "only-p1"), ("s", "p2")]


def test_a_long_pool_name_is_one_consistent_key():
    name = "one shared noise parameter for every observable of the assay"
    blocks, _, _ = blocks_at(observables(name, name))
    assert len(blocks) == 1


# --- the likelihood ------------------------------------------------------------------------------

def test_the_pooled_nll_is_the_closed_form_with_one_sigma():
    blocks, _, _ = blocks_at(observables("noise", "noise"))
    (sse, n), = [v[:2] for v in blocks.values()]
    assert concentrated_nll(blocks) == pytest.approx(0.5 * n * np.log(sse / n))
    assert block_sigmas(blocks)[("s", "noise")] == pytest.approx(np.sqrt(sse / n))


def test_pooling_costs_one_sigma_not_two():
    pooled, _, _ = blocks_at(observables("noise", "noise"))
    separate, _, _ = blocks_at(observables())
    # same residuals, so the pooled objective differs only in how the variance is estimated
    assert concentrated_nll(pooled) != pytest.approx(concentrated_nll(separate))
    assert len(pooled) == 1 and len(separate) == 2


def test_a_pool_composes_with_the_element_level_sigma_block():
    """sigma_block pools one observable across simulations; sigma_pool pools observables: together, everything."""
    blocks, _, _ = blocks_at(observables("noise", "noise"), sims=("s1", "s2"), element_block="assay")
    assert list(blocks) == [("assay", "noise")]
    assert blocks[("assay", "noise")][1] == 2 * 2 * len(T)


# --- the gradient --------------------------------------------------------------------------------

def test_the_analytic_gradient_of_a_pooled_loss_matches_finite_differences():
    groups, replicates, items, _ = setup(observables("noise", "noise"), sims=("s1", "s2"), element_block="assay")
    loss = LossGradient(groups, replicates, NAMES)
    theta = TRUE * [1.1, 0.9]
    it, dirs = items(theta)
    f, g = loss.gradient(it, dirs, theta)
    assert f == pytest.approx(loss.value(it, theta))
    ref = np.zeros(2)
    for k in range(2):
        h = 1e-6 * theta[k]
        up, dn = theta.copy(), theta.copy()
        up[k] += h
        dn[k] -= h
        ref[k] = (loss.value(items(up)[0], up) - loss.value(items(dn)[0], dn)) / (2 * h)
    np.testing.assert_allclose(g, ref, rtol=1e-6)


# --- reporting -----------------------------------------------------------------------------------

def test_every_pooled_observable_counts_as_resolved():
    """The coverage check counts (simulation, observable) pairs, not blocks, so a pool is not a lost observable."""
    groups, replicates, items, _ = setup(observables("noise", "noise"))
    it, _ = items(TRUE)
    seen = set()
    collect_loss_blocks(it, groups, replicates, NAMES, TRUE, dict(zip(NAMES, TRUE)), seen=seen)
    assert len(seen) == 2 and {label for _, label in seen} == {"p1", "p2"}

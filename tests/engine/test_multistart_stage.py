import numpy as np
import pytest

from pyantigen.engine.Fit_cache import fit_fingerprint
from pyantigen.engine import Simulate
from pyantigen.engine.Optimize import (
    _ENGINE_ONLY_OPTIMIZER_KEYS,
    _prepare_optimizer_kwargs,
    _resolve_multistart_limits,
    _resolve_multistart_stage,
    _run_multistart,
)

STAGE = {"multistart_method": "L-BFGS-B",
         "multistart_optimizer_kwargs": {"options": {"maxiter": 20}}}


def test_absent_means_no_stage():
    assert _resolve_multistart_stage("Nelder-Mead", {}) == (None, {})
    assert _resolve_multistart_stage("Nelder-Mead", None) == (None, {})
    assert _resolve_multistart_stage("Nelder-Mead", {"options": {"maxiter": 5}}) == (None, {})


def test_stage_resolved_without_aliasing_the_spec():
    m, kw = _resolve_multistart_stage("Nelder-Mead", STAGE)
    assert m == "L-BFGS-B" and kw == {"options": {"maxiter": 20}}
    kw["extra"] = 1
    assert "extra" not in STAGE["multistart_optimizer_kwargs"]


def test_global_method_rejected():
    with pytest.raises(ValueError, match="global method"):
        _resolve_multistart_stage("Nelder-Mead", {"multistart_method": "differential_evolution"})


def test_stage_keys_never_reach_scipy():
    kw = _prepare_optimizer_kwargs("Nelder-Mead", {**STAGE, "options": {"maxiter": 9}},
                                   False, None, None)
    assert not set(kw) & set(_ENGINE_ONLY_OPTIMIZER_KEYS)
    assert kw["options"]["maxiter"] == 9


# --- fit fingerprint: a default stage must not invalidate cached fits --------

NAMES = ["a", "b"]
KW = {"options": {"maxiter": 500}}


def _fp(kw, n_starts):
    return fit_fingerprint(NAMES, [1.0, 2.0], [(0.1, 10.0)] * 2, ["log10"] * 2,
                           {"A": {}}, "model", "Nelder-Mead", kw,
                           n_starts=n_starts, solver_hash="s", data_hash="d")


def test_stage_ignored_in_fingerprint_at_one_start():
    assert _fp(KW, 1) == _fp({**KW, **STAGE}, 1)


def test_stage_changes_fingerprint_with_several_starts():
    assert _fp(KW, 5) != _fp({**KW, **STAGE}, 5)
    other = {**KW, "multistart_method": "L-BFGS-B",
             "multistart_optimizer_kwargs": {"options": {"maxiter": 50}}}
    assert _fp({**KW, **STAGE}, 5) != _fp(other, 5)


# --- the stage itself ---------------------------------------------------------

def _two_basins(x):
    """Global minimum 0 at (3, 3); a shallower basin of depth 1 at (-3, -3)."""
    x = np.asarray(x, float)
    return min(np.sum((x - 3.0) ** 2), 1.0 + np.sum((x + 3.0) ** 2))


def test_lbfgsb_stage_respects_bounds_and_picks_best_start():
    starts = [np.array([-3.5, -3.5]), np.array([2.0, 2.0]), np.array([-2.5, -3.2])]
    bounds = [(-5.0, 5.0)] * 2
    kw = _prepare_optimizer_kwargs("L-BFGS-B", STAGE["multistart_optimizer_kwargs"],
                                   False, None, None)
    best, records, best_i = _run_multistart(_two_basins, starts, "L-BFGS-B", bounds,
                                            kw, ["lin"] * 2, screen=False, verbose=False)
    assert best_i == 1
    assert best.fun < 1e-6
    assert np.allclose(best.x, [3.0, 3.0], atol=1e-3)
    assert np.all(best.x >= -5.0) and np.all(best.x <= 5.0)
    assert len(records) == 3


# --- evaluation limits and the failure penalty --------------------------------

def test_limit_defaults_are_depth_zero_four_attempts():
    assert _resolve_multistart_limits({}) == {"max_attempts": 4, "max_depth": 0,
                                              "failure_dnll": 1000.0}


def test_limits_overlay_and_validate():
    out = _resolve_multistart_limits({"multistart_limits": {"max_attempts": 2}})
    assert out["max_attempts"] == 2 and out["max_depth"] == 0
    with pytest.raises(ValueError, match="unknown"):
        _resolve_multistart_limits({"multistart_limits": {"max_attempt": 2}})
    with pytest.raises(ValueError, match="positive"):
        _resolve_multistart_limits({"multistart_limits": {"failure_dnll": -1}})
    assert _resolve_multistart_limits(
        {"multistart_limits": {"failure_dnll": None}})["failure_dnll"] is None


def test_limits_are_engine_only_and_dropped_from_a_one_start_fingerprint():
    assert "multistart_limits" in _ENGINE_ONLY_OPTIMIZER_KEYS
    lim = {"multistart_limits": {"max_attempts": 2}}
    assert _fp({**KW, **STAGE}, 1) == _fp({**KW, **STAGE, **lim}, 1)
    assert _fp({**KW, **STAGE}, 5) != _fp({**KW, **STAGE, **lim}, 5)


class _Integrator:
    absolute_tolerance = 1e-12
    relative_tolerance = 1e-6
    maximum_num_steps = 20000
    initial_time_step = 0.0

    def setValue(self, *a):
        pass


class _FailingRR:
    """A roadrunner stand-in whose simulate always raises, counting the calls."""
    _scalar_abs_tol = 1e-12

    def __init__(self):
        self.integrator = _Integrator()
        self.calls = 0

    def simulate(self, *a):
        self.calls += 1
        raise RuntimeError("CVODE failure: too much work")

    def getValue(self, name):
        return 0.0


@pytest.fixture
def bare_ladder(monkeypatch):
    for name in ("save_model_state", "restore_model_state", "clamp_state_dust",
                 "floor_tolerance_vector"):
        monkeypatch.setattr(Simulate, name, lambda *a, **k: None)
    monkeypatch.setattr(Simulate, "_scalar_tolerance", lambda v: 1e-12)


def _attempts(**limits):
    rr = _FailingRR()
    block = {"start": 0.0, "end": 10.0, "n_points": 8}
    with Simulate.retry_limits(**limits):
        with pytest.raises(Exception):
            Simulate.safe_simulate(rr, block, ["time"])
    return rr.calls


def test_default_ladder_is_unchanged(bare_ladder):
    # 10 attempts, then subdivide; the first half fails the same way and its
    # exception ends the whole call, so 10 attempts at each of depths 0..4.
    assert _attempts() == 50


def test_four_attempts_depth_zero(bare_ladder):
    assert _attempts(max_attempts=4, max_depth=0) == 4


def test_limits_restore_on_exit(bare_ladder):
    assert Simulate._RETRY_LIMITS is None
    with Simulate.retry_limits(4, 0):
        assert Simulate._RETRY_LIMITS == (4, 0)
        with Simulate.retry_limits(2, 0):
            assert Simulate._RETRY_LIMITS == (2, 0)
        assert Simulate._RETRY_LIMITS == (4, 0)
    assert Simulate._RETRY_LIMITS is None


def test_invalid_limits_rejected():
    with pytest.raises(ValueError):
        Simulate.set_retry_limits(0, 0)
    with pytest.raises(ValueError):
        Simulate.set_retry_limits(4, -1)


def _cliff(x):
    """(x - 1)^2 on x <= 2; the model 'does not integrate' beyond that."""
    x = float(np.asarray(x)[0])
    return (x - 1.0) ** 2 if x <= 2.0 else 1e10


def test_failure_penalty_is_finite_and_the_fit_still_finds_the_minimum():
    seen = []

    def spy(x):
        v = _cliff(x)
        seen.append(v)
        return v

    # Start 3 ends up on the plateau only if it is not screened out.
    best, records, best_i = _run_multistart(
        spy, [np.array([0.0]), np.array([1.9]), np.array([3.0])], "L-BFGS-B",
        [(-5.0, 5.0)], {"options": {"maxiter": 30}}, ["lin"], screen=True,
        verbose=False, failure_dnll=1000.0)
    assert best.fun < 1e-6 and np.allclose(best.x, [1.0], atol=1e-3)
    assert len(records) == 2               # the failing start was screened out


def test_scipy_never_sees_the_1e10_cliff():
    given_to_scipy = []
    import scipy.optimize as so
    real = so.minimize

    def spy_minimize(fun, x0, **kw):
        def wrapped(x):
            v = fun(x)
            given_to_scipy.append(v)
            return v
        return real(wrapped, x0, **kw)

    def edge(x):
        """The minimum sits right at the edge of the integrable region, so the
        first line-search steps from the left overshoot it."""
        x = float(np.asarray(x)[0])
        return (x - 1.0) ** 2 if x <= 1.01 else 1e10

    import unittest.mock as mock
    with mock.patch("scipy.optimize.minimize", spy_minimize):
        best, _, _ = _run_multistart(
            edge, [np.array([0.0]), np.array([0.5])], "L-BFGS-B",
            [(-5.0, 5.0)], {"options": {"maxiter": 30}}, ["lin"],
            screen=True, verbose=False, failure_dnll=1000.0)
    assert given_to_scipy and max(given_to_scipy) < 1e5    # never 1e10
    assert any(v > 900 for v in given_to_scipy)            # but the wall was hit
    assert best.fun < 1e-3 and abs(best.x[0] - 1.0) < 0.05


def test_a_start_that_ends_on_the_plateau_is_a_failure_not_an_optimum():
    best, records, best_i = _run_multistart(
        _cliff, [np.array([1.5]), np.array([4.0])], "L-BFGS-B",
        [(-5.0, 5.0)], {"options": {"maxiter": 30}}, ["lin"], screen=False,
        verbose=False, failure_dnll=1000.0)
    assert best_i == 0
    assert records[1]["fun"] is None and records[1]["success"] is False


def test_without_failure_dnll_the_old_behaviour_is_untouched():
    best, records, _ = _run_multistart(
        _cliff, [np.array([0.0]), np.array([1.9])], "L-BFGS-B", [(-5.0, 5.0)],
        {"options": {"maxiter": 30}}, ["lin"], screen=True, verbose=False)
    assert best.fun < 1e-6 and len(records) == 2

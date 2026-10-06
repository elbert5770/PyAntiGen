"""The engine's per-model caches must never enumerate the model's ids.

RoadRunner's ``__getattr__`` answers every missed attribute by building the list
of all selection ids the model has (``_getIds(SelectionRecord_ALL)``). Reading a
cache slot with ``try: r._x`` / ``getattr(r, "_x", None)`` therefore costs
nothing on a small model and about 18 minutes and 16 GB on one with ~9,000 ODEs,
once per slot per RoadRunner. See pyantigen.engine.Model_cache.

These tests count ``_getIds`` calls instead of timing anything, so they catch a
reintroduced probe on a two-species model.

Run from the repository root: python -m pytest tests/engine/test_model_cache.py
"""
import pathlib
import re

import numpy as np
import pytest

te = pytest.importorskip("tellurium")

from pyantigen.engine import Model_cache  # noqa: E402
from pyantigen.engine.Optimize import (OptRoadRunnerProxy,  # noqa: E402
                                       remember_parameter_baseline,
                                       restore_parameter_baseline, run_all,
                                       set_parameters_from_dict)
from pyantigen.engine.Simulate import (safe_simulate,  # noqa: E402
                                       save_model_state, simulate)

MODEL = """
model m
  species A = 1, B = 0
  A' = -k*A
  B' = k*A
  k = 0.5
end
"""


def _replicate():
    return {
        "Label": "arm",
        "Update_parameters": lambda r, rep: None,
        "Observed_species": lambda r: ["time", "[A]", "[B]"],
        "Solver_settings": lambda rep: {
            "integrator": "cvode", "absolute_tolerance": 1e-10,
            "relative_tolerance": 1e-10, "maximum_num_steps": 20000,
            "stiff": True, "variable_step_size": True,
            "simulation_blocks": {"b": {"start": 0, "end": 5, "n_points": 11}}},
    }


@pytest.fixture
def id_calls(monkeypatch):
    """Record every ``_getIds`` call made on any RoadRunner."""
    base = next(c for c in type(te.loada(MODEL)).__mro__
                if "_getIds" in c.__dict__)
    original = base._getIds
    calls = []

    def counting(self, types):
        calls.append(types)
        return original(self, types)

    monkeypatch.setattr(base, "_getIds", counting)
    return calls


def test_the_counter_sees_a_missed_attribute(id_calls):
    """Control: without this, a counter that never fires would pass everything."""
    r = te.loada(MODEL)
    id_calls.clear()
    assert getattr(r, "_no_such_attribute", None) is None
    assert len(id_calls) == 1


def test_a_missed_cache_read_does_not_enumerate_ids(id_calls):
    r = te.loada(MODEL)
    id_calls.clear()
    assert Model_cache.get(r, "_no_such_attribute") is None
    assert Model_cache.get(r, "_no_such_attribute", 7) == 7
    assert id_calls == []


def test_first_run_all_on_a_fresh_model_enumerates_nothing(id_calls):
    """The path a caller takes when it never recorded a baseline: every cache
    slot is a miss on the first call."""
    r = te.loada(MODEL)
    id_calls.clear()
    res = run_all(r, "arm", _replicate(), {},
                  set_parameters=set_parameters_from_dict, parameters={"k": 0.5})
    assert np.all(np.isfinite(np.asarray(res["arm"]["results"]["[A]"], dtype=float)))
    assert id_calls == []


def test_each_cache_site_enumerates_nothing_on_its_own(id_calls):
    r = te.loada(MODEL)
    id_calls.clear()
    restore_parameter_baseline(r)       # _pyantigen_param_baseline: miss, then write
    save_model_state(r)                 # _cached_state_keys
    block = {"b": {"start": 0, "end": 5, "n_points": 11}}
    simulate(r, {"integrator": "cvode", "simulation_blocks": block},
             ["time", "[A]"])           # _available_symbols and _scalar_abs_tol
    assert id_calls == []


def test_safe_simulate_without_configure_integrator_enumerates_nothing(id_calls):
    """_scalar_abs_tol is normally written by configure_integrator before the
    retry path reads it; a caller that skips that step must not pay for it."""
    r = te.loada(MODEL)
    id_calls.clear()
    res, _meta = safe_simulate(r, {"start": 0, "end": 5, "n_points": 11}, ["time", "[A]"])
    assert len(res) == 11
    assert Model_cache.get(r, "_scalar_abs_tol") is None
    assert id_calls == []


def test_the_caches_are_filled_once_and_reused(id_calls):
    r = te.loada(MODEL)
    run_all(r, "arm", _replicate(), {}, set_parameters=set_parameters_from_dict,
            parameters={"k": 0.5})
    keys = Model_cache.get(r, "_cached_state_keys")
    symbols = Model_cache.get(r, "_available_symbols")
    baseline = Model_cache.get(r, "_pyantigen_param_baseline")
    assert keys and symbols and baseline is not None
    assert Model_cache.get(r, "_scalar_abs_tol") == pytest.approx(1e-10)
    run_all(r, "arm", _replicate(), {}, set_parameters=set_parameters_from_dict,
            parameters={"k": 0.5})
    assert Model_cache.get(r, "_cached_state_keys") is keys
    assert Model_cache.get(r, "_available_symbols") is symbols
    assert Model_cache.get(r, "_pyantigen_param_baseline") is baseline


def test_a_cache_written_through_the_proxy_is_read_back_either_way(id_calls):
    r = te.loada(MODEL)
    proxy = OptRoadRunnerProxy(r, ["k"])
    id_calls.clear()
    assert Model_cache.get(proxy, "_x") is None
    assert Model_cache.put(proxy, "_x", 3)
    assert Model_cache.get(proxy, "_x") == 3
    assert Model_cache.get(r, "_x") == 3        # it lives on the RoadRunner
    assert "_x" not in vars(proxy)
    assert id_calls == []


def test_remember_through_the_proxy_is_seen_by_restore_on_the_model(id_calls):
    r = te.loada(MODEL)
    remember_parameter_baseline(r)
    r["k"] = 9.0
    assert restore_parameter_baseline(OptRoadRunnerProxy(r, [])) == ["k"]
    assert r["k"] == 0.5


def test_objects_without_an_instance_dict_are_tolerated():
    assert Model_cache.get(3, "_x", "d") == "d"
    assert Model_cache.put(3, "_x", 1) is False


def test_no_engine_module_probes_a_private_attribute_by_name():
    """A cache slot read as ``getattr(r, "_x", ...)`` or ``hasattr(r, "_x")``
    reaches RoadRunner's __getattr__ on a miss. Use Model_cache.get instead."""
    pattern = re.compile(r"""(?:getattr|hasattr)\([^,()]+,\s*["']_[A-Za-z]""")
    engine = pathlib.Path(__file__).resolve().parents[2] / "pyantigen" / "engine"
    offenders = []
    for path in sorted(engine.glob("*.py")):
        if path.name == "Model_cache.py":
            continue                    # the module's own docstring quotes the pattern
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if pattern.search(line):
                offenders.append(f"{path.name}:{n}: {line.strip()}")
    assert offenders == [], "\n".join(offenders)

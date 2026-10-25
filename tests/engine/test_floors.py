"""Saved noise floors (pyantigen.engine.Floors) and the fixed-sigma rule at the end of a fit."""
import math

import numpy as np
import pytest

from pyantigen.engine import Floors, Noise_floor
from pyantigen.engine import Optimize as O


@pytest.fixture(autouse=True)
def _clean_cache():
    Noise_floor.clear_cache()
    yield
    Noise_floor.clear_cache()


def _data():
    x = np.linspace(0, 10, 60)
    return x, np.sin(x) + 0.2 * np.random.default_rng(3).standard_normal(60)


def _calibrate(reference="shape", key="obs"):
    """The same data, calibrated against a different model curve: the case a recalibration at the optimum is."""
    x, y = _data()
    ref = np.sin(x) if reference == "shape" else y
    return Noise_floor.get_noise_floor(x, y, key, shape_reference=ref)


def test_calibrate_snapshots_only_what_the_evaluation_created():
    Noise_floor.get_noise_floor(np.arange(20.0), np.arange(20.0) ** 0.5, "stale")
    snap = Floors.calibrate(lambda: _calibrate())
    assert len(snap) == 1 and next(iter(snap))[0] == "obs"


def test_save_load_roundtrip_and_stamp(tmp_path):
    snap = Floors.calibrate(lambda: _calibrate())
    path = str(tmp_path / "sub" / "floors.pkl")
    stamp = Floors.save(snap, path)
    assert stamp == Floors.file_sha(path) and len(stamp) == 16
    Noise_floor.clear_cache()
    loaded, stamp2 = Floors.load(path, expect_sha=stamp)
    assert stamp2 == stamp and Noise_floor.export_cache().keys() == snap.keys()
    assert next(iter(loaded.values())).sigma == next(iter(snap.values())).sigma
    with pytest.raises(ValueError, match="expected"):
        Floors.load(path, expect_sha="0" * 16)


def test_seeded_floor_is_used_not_recomputed(tmp_path):
    snap = Floors.calibrate(lambda: _calibrate())
    path = str(tmp_path / "f.pkl")
    Floors.save(snap, path)
    Noise_floor.clear_cache()
    Floors.load(path)
    x, y = _data()
    got = Noise_floor.get_noise_floor(x, y, "obs", shape_reference=y)       # a different reference is ignored: a hit
    assert got.sigma == next(iter(snap.values())).sigma


def _snap(**sigmas):
    return {(k, b"x", b"y", (0.1,)): Noise_floor.NoiseFloor(sigma=v, span=0.25, n=30, x_fit=None, y_fit=None)
            for k, v in sigmas.items()}


def test_compare_orders_by_size_of_change_and_stable_applies_a_tolerance():
    a, b = _snap(p=1.0, q=2.0, r=3.0), _snap(p=1.01, q=3.0, r=3.0)
    rows = Floors.compare(a, b)
    assert [r["key"] for r in rows] == ["q", "p", "r"]
    assert math.isclose(rows[0]["log_ratio"], math.log(1.5))
    assert not Floors.stable(rows) and Floors.stable(rows, tol=0.6)
    assert Floors.stable(Floors.compare(a, a))
    assert Floors.stable(Floors.compare({}, {}))


def test_a_floor_that_appears_or_vanishes_is_not_stable():
    a = _snap(p=1.0)
    assert not Floors.stable(Floors.compare(a, {}))
    assert not Floors.stable(Floors.compare({}, a))
    missing = {next(iter(a)): None}                                    # too few points: no floor on either side
    assert Floors.stable(Floors.compare(missing, missing))


def test_fixed_mode_keeps_its_floors_at_the_end_of_a_fit(monkeypatch):
    Noise_floor.get_noise_floor(np.arange(20.0), np.arange(20.0) ** 0.5, "k")
    monkeypatch.setenv("PYANTIGEN_SIGMA_FLOOR", "fixed")
    O._recalibrate_floors_at_optimum()
    assert len(Noise_floor.export_cache()) == 1
    monkeypatch.setenv("PYANTIGEN_SIGMA_FLOOR", "cap")
    O._recalibrate_floors_at_optimum()
    assert Noise_floor.export_cache() == {}

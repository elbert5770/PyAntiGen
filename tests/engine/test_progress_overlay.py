"""The progress overlay: named like the profile plots, and silent for an evaluation that is not part of a fit."""
import os

import numpy as np

from pyantigen.engine import Optimize as O


def test_the_progress_files_are_named_like_the_profile_plots():
    assert O._progress_stem("Model", "Tag") == "Model_Tag_optimization_progress"
    assert O._progress_stem("", "") == "optimization_progress"
    assert O._progress_stem("Model", None) == "Model_optimization_progress"


def test_a_suspended_overlay_writes_nothing_and_comes_back(tmp_path):
    state = O._progress_overlay_state
    assert state["enabled"]
    with O._overlay_suspended():
        assert not state["enabled"]
        O._render_progress_overlay({("sim", "obs"): object()}, 1.0, 1.0, 1, str(tmp_path),
                                   param_names=["a"], param_values=np.array([1.0]), model_name="M", experiment_id="T")
        with O._overlay_suspended():                      # nested: still off, and the outer one restores it
            assert not state["enabled"]
        assert not state["enabled"]
    assert state["enabled"]
    assert os.listdir(tmp_path) == []                     # no JSON, no figure


def test_the_overlay_json_carries_the_profile_plot_naming(tmp_path, monkeypatch):
    # an empty trace gives no panels, but the progress JSON of an improved point is written first, under the new name
    monkeypatch.setitem(O._progress_overlay_state, "timestamp", "20260101_000000")
    try:
        O._render_progress_overlay({("sim", "obs"): [(0.0, 1.0, 1.0)]}, 1.0, 1.0, 1, str(tmp_path),
                                   param_names=["a"], param_values=np.array([2.0]), model_name="M", experiment_id="T")
    except Exception:
        pass                                              # drawing a fake trace may fail; the JSON is written before it
    assert "M_T_optimization_progress_20260101_000000.json" in os.listdir(tmp_path)

"""A saved profile-point state must never be resumed into a different problem.

``param_idx`` is a position, not an identity: a reparametrization that keeps the
parameter counts but changes what occupies each slot, or a bounds-only change
(which :func:`spec_fingerprint` does not hash), must change the identity so the
old state is declined rather than resumed.
"""
import os
import tempfile

from pyantigen.engine.Profile_checkpoint import (
    load_point_state, point_identity, point_state_path, save_point_state)

BASE = dict(param_idx=2, x_fixed=0.5, n_nuisance=3, method="Nelder-Mead",
            param_name="k", bounds=[(0.0, 1.0), (0.0, 2.0), (0.0, 3.0), (0.0, 4.0)])


def test_identity_is_stable_across_harmless_representation_changes():
    a = point_identity(**BASE)
    b = point_identity(**{**BASE, "bounds": [[0, 1], [0, 2], [0, 3], [0, 4]],
                          "method": "nelder-mead"})
    assert a == b


def test_identity_changes_with_parameter_name():
    assert point_identity(**BASE) != point_identity(**{**BASE, "param_name": "other"})


def test_identity_changes_with_bounds_only():
    tighter = [(0.0, 1.0), (0.0, 2.0), (0.0, 1.5), (0.0, 4.0)]
    assert point_identity(**BASE) != point_identity(**{**BASE, "bounds": tighter})


def test_identity_accepts_one_sided_bounds():
    assert point_identity(**{**BASE, "bounds": [(0.0, None), None, (None, 3.0), (0, 4)]})


def test_state_saved_for_one_problem_is_not_loaded_for_another():
    with tempfile.TemporaryDirectory() as d:
        path = point_state_path(d, "k", 0.5)
        assert save_point_state(path, point_identity(**BASE), {"x": [1.0]})
        assert load_point_state(path, point_identity(**BASE)) is not None
        changed = point_identity(**{**BASE, "param_name": "other"})
        assert load_point_state(path, changed) is None
        assert os.path.exists(path)

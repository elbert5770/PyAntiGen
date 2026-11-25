"""The progress JSON of an AMICI LM fit, and the profile side tasks that no longer log a results snapshot."""
import json

import numpy as np

from pyantigen.engine import Optimize as OPT
from pyantigen.engine.Fit_lm import lm_fit
from pyantigen.engine.Model_optimize import _is_profile_side_run


def test_write_progress_json_name_and_content(tmp_path):
    OPT._write_progress_json(str(tmp_path), "M_APP_optimization_progress", "20260101_000000", ["a", "b"],
                             np.array([1.5, np.nan]), -12.5, 7, model_name="M", experiment_id="APP", method="LM")
    d = json.loads((tmp_path / "M_APP_optimization_progress_20260101_000000.json").read_text())
    assert d["parameters"] == {"a": 1.5, "b": None}
    assert d["metadata"]["total_loss"] == -12.5 and d["metadata"]["n_fevals"] == 7
    assert d["metadata"]["message"] == "Optimization in progress" and d["metadata"]["method"] == "LM"


def test_on_best_fires_for_each_improvement():
    class Provider:
        L = np.array([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]])
        y = np.array([1.0, -1.0, 0.5])

        def linear_residual_jacobian(self, p):
            return self.L @ np.asarray(p, float) - self.y, self.L.copy(), 0.0

    seen = []
    lm_fit(Provider(), np.array([5.0, 5.0]), [(-20, 20)] * 2, scales=["linear", "linear"],
           on_best=lambda x, f, n: seen.append((f, n)), log=lambda m: None)
    assert seen and all(b[0] < a[0] for a, b in zip(seen, seen[1:]))


def test_profile_side_run_is_not_logged(monkeypatch):
    reused = {"fit_source": "/x/fits/APP.json"}
    monkeypatch.delenv("PYANTIGEN_PROFILE_SIDE", raising=False)
    monkeypatch.delenv("PYANTIGEN_PROFILE_MODE", raising=False)
    assert not _is_profile_side_run(reused)                       # a plain run on a reused fit still logs
    monkeypatch.setenv("PYANTIGEN_PROFILE_SIDE", "3")
    assert _is_profile_side_run(reused)                           # a side task does not
    assert not _is_profile_side_run({"fit_source": None})         # a side task that fitted in its own process does
    monkeypatch.setenv("PYANTIGEN_PROFILE_MODE", "collect")
    assert not _is_profile_side_run(reused)


def _opt_with_profile(calls):
    import numpy as np
    pv = np.array([0.5, 0.75, 1.0, 1.25, 1.5])
    nr = 8.0 * (pv - 1.0) ** 2

    def profile_all(**kw):
        calls.append(kw)
        return {"a": (pv, nr)}, 0.0, None, {"per_param": {"a": {"n": 5, "n_not_converged": 0}}, "points": {}, "scales": {}}
    return {"x": np.array([1.0]), "stats": {"profile_likelihood": object(), "profile_likelihood_all": profile_all}}


def test_a_profile_side_still_runs_the_profile_but_writes_no_files(tmp_path):
    """The profile is computed inside _save_profile_likelihood_plot, so a side task must reach it and only skip the files."""
    from pyantigen.engine.Model_optimize import _save_profile_likelihood_plot
    calls = []
    opt = _opt_with_profile(calls)
    _save_profile_likelihood_plot(opt, ["a"], str(tmp_path), "M", tag="T", write_files=False)
    assert len(calls) == 1                                          # the walk ran
    assert "profile_ci" in opt["stats"] and "profile_traces" in opt["stats"]
    assert list(tmp_path.iterdir()) == []                           # and nothing was written


def test_the_collect_writes_the_plots_and_the_summary(tmp_path):
    from pyantigen.engine.Model_optimize import _save_profile_likelihood_plot
    calls = []
    _save_profile_likelihood_plot(_opt_with_profile(calls), ["a"], str(tmp_path), "M", tag="T")
    names = sorted(p.name for p in tmp_path.iterdir())
    assert any("profile_likelihood_full" in n for n in names) and any("profile_summary" in n for n in names), names

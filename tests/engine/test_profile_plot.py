"""Profile_plot: the per-parameter profile figures, their reader for the AMICI stage's files, and their use by the engine."""
import json
import os

import numpy as np
import pytest

from pyantigen.engine import Profile_plot as PP

THR = PP.THRESHOLD


def parabola(c, w, xs, conv=None):
    return [(float(x), float(0.5 * ((x - c) / w) ** 2), True if conv is None else conv(x)) for x in xs]


def test_interval_reach_and_open_sides():
    pts = parabola(0.0, 1.0, [-3, -2, -1, 1, 2]) + [(3.0, 1.0, True)]       # the high end stays inside the cut: open
    iv = PP.interval(pts, 0.0, THR)
    assert iv["low"]["reach"] == -1 and not iv["low"]["open"]               # -2 is at 2.0, past the cut of 1.92; -1 is at 0.5
    assert iv["high"]["open"] and iv["high"]["reach"] == 3.0


def test_a_hollow_point_does_not_count_toward_reach():
    pts = [(-1.0, 0.5, True), (-2.0, 1.0, False), (1.0, 0.5, True)]
    assert PP.interval(pts, 0.0)["low"]["reach"] == -1.0


def test_figures_suffix_goes_after_the_figure_tag(tmp_path):
    prof = {"a": parabola(0.0, 1.0, [-2, -1, 1, 2])}
    paths = PP.figures(str(tmp_path / "base"), "run", ["a"], {"a": 0.0}, prof, {"a": "log10"}, suffix="_T1", log=lambda m: None)
    assert [os.path.basename(p) for p in paths] == ["base_full_T1.png", "base_zoom_T1.png"]


def test_figures_write_full_and_zoom(tmp_path):
    prof = {"a": parabola(0.0, 1.0, [-3, -2, -1, 1, 2, 3]), "b": parabola(1.0, 0.5, [0, 0.5, 1.5, 2], lambda x: x != 2)}
    lines = []
    paths = PP.figures(str(tmp_path / "p"), "run", ["a", "b"], {"a": 0.0, "b": 1.0}, prof, {"a": "log10", "b": "lin"},
                       log=lines.append)
    assert [os.path.basename(p) for p in paths] == ["p_full.png", "p_zoom.png"]
    assert all(os.path.getsize(p) > 5000 for p in paths)
    assert any(l.startswith("a ") for l in lines) and lines[-1].startswith("* =")


def write(path, rows):
    with open(path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def rec(param, x, nll, verdict="converged", evals=5):
    return {"param": param, "x_fixed": x, "nll": nll, "dnll": nll + 10.0, "verdict": verdict, "evals": evals}


def test_load_points_keeps_the_lowest_nll_per_point_and_skips_the_optimum_copy(tmp_path):
    write(tmp_path / "profile_0_up.jsonl", [rec("a", 0.0, -10.0, evals=0), rec("a", 1.0, -8.0, "not_converged"), rec("a", 2.0, -5.0)])
    write(tmp_path / "relax_0_up.jsonl", [rec("a", 1.0, -9.0)])                       # re-solved with wider bounds: lower, converged
    write(tmp_path / "fill_0_up.jsonl", [rec("a", 1.5, -7.0)])
    prof = PP.load_points(str(tmp_path), ["a", "b"])
    assert sorted(prof["a"]) == [(1.0, 1.0, True), (1.5, 3.0, True), (2.0, 5.0, True)]
    assert prof["b"] == []


def test_load_meta_from_meta_json_then_fim_npz(tmp_path):
    json.dump({"names": ["a"], "scales": ["log10"], "x": [0.5]}, open(tmp_path / "meta.json", "w"))
    assert PP.load_meta(str(tmp_path)) == (["a"], ["log10"], [0.5])
    os.remove(tmp_path / "meta.json")
    os.makedirs(tmp_path / "fim")
    np.savez(tmp_path / "fim" / "fim.npz", names=np.array(["b"]), scales=np.array(["lin"]), x=np.array([2.0]))
    assert PP.load_meta(str(tmp_path)) == (["b"], ["lin"], [2.0])
    with pytest.raises(FileNotFoundError):
        PP.load_meta(str(tmp_path / "fim"))


def test_main_reads_a_stage_folder_and_writes_the_figures(tmp_path):
    d = tmp_path / "amici"
    os.makedirs(d)
    json.dump({"names": ["a", "b"], "scales": ["log10", "lin"], "x": [0.0, 1.0]}, open(d / "meta.json", "w"))
    write(d / "profile_0_up.jsonl", [rec("a", 0.0, -10.0, evals=0), rec("a", 1.0, -9.0)])
    write(d / "profile_0_down.jsonl", [rec("a", 0.0, -10.0, evals=0), rec("a", -1.0, -9.5)])
    write(d / "profile_1_up.jsonl", [rec("b", 1.0, -10.0, evals=0), rec("b", 2.0, -9.0)])
    assert PP.main([str(d), "--out-prefix", str(tmp_path / "fig"), "--no-timestamp"]) == 0
    assert (tmp_path / "fig_full.png").exists() and (tmp_path / "fig_zoom.png").exists()


def test_main_names_the_files_like_the_engine_and_stamps_them(tmp_path):
    d = tmp_path / "results" / "MODEL_X" / "profiles" / "RUN_1" / "amici"
    os.makedirs(d)
    json.dump({"names": ["a"], "scales": ["log10"], "x": [0.0]}, open(d / "meta.json", "w"))
    write(d / "profile_0_up.jsonl", [rec("a", 0.0, -10.0, evals=0), rec("a", 1.0, -9.0)])
    assert PP.locate(str(d)) == ("MODEL_X", "RUN_1", str(tmp_path / "results" / "MODEL_X"))     # the engine writes into <results>/<MODEL>
    assert PP.main([str(d)]) == 0
    names = sorted(os.listdir(tmp_path / "results" / "MODEL_X"))
    full = [n for n in names if n.startswith("MODEL_X_RUN_1_profile_likelihood_full_") and n.endswith(".png")]
    zoom = [n for n in names if n.startswith("MODEL_X_RUN_1_profile_likelihood_zoom_") and n.endswith(".png")]
    assert len(full) == 1 and len(zoom) == 1 and full[0].split("_full_")[1] == zoom[0].split("_zoom_")[1]


def test_locate_outside_the_engine_layout_uses_the_current_directory(tmp_path):
    assert PP.locate(str(tmp_path / "somewhere"))[2] == os.getcwd()


def test_slice_figures_use_the_same_panels(tmp_path):
    prof = {"a": parabola(0.0, 1.0, [-2, -1, 1, 2]), "b": parabola(0.5, 0.2, [0.1, 0.3, 0.7, 0.9])}
    paths = PP.figures(str(tmp_path / "s"), "run", ["a", "b"], {"a": 0.0, "b": 0.5}, prof, {"a": "log10", "b": "lin"},
                       suffix="_T", kind="slice", log=lambda m: pytest.fail("a slice prints no profile table"))
    assert [os.path.basename(p) for p in paths] == ["s_full_T.png", "s_zoom_T.png"] and all(os.path.getsize(p) > 5000 for p in paths)


def _fake_opt(with_points):
    est = [2.0, 0.5]                                    # linear values: a is log10, b is linear
    xs_a = np.log10(est[0]) + np.array([-0.6, -0.3, 0.3, 0.6])
    pv_a, nr_a = np.r_[10 ** xs_a, est[0]], np.r_[0.5 * (xs_a / 0.3) ** 2, 0.0]
    pv_b, nr_b = np.array([0.1, 0.3, 0.5, 0.8, 1.2]), 0.5 * ((np.array([0.1, 0.3, 0.5, 0.8, 1.2]) - 0.5) / 0.2) ** 2
    order_a, order_b = np.argsort(pv_a), np.argsort(pv_b)
    traces = {"a": (pv_a[order_a], nr_a[order_a]), "b": (pv_b[order_b], nr_b[order_b])}
    conv = {"reach": {}, "per_param": {}}
    if with_points:
        conv["scales"] = ["log10", "lin"]
        conv["points"] = {"a": [[float(x), float(y), bool(i != 0)] for i, (x, y) in enumerate(zip(np.log10(traces["a"][0]), traces["a"][1]))],
                          "b": [[float(x), float(y), True] for x, y in zip(*traces["b"])]}
    stats = {"profile_likelihood": lambda i: traces[("a", "b")[i]],
             "profile_likelihood_all": lambda **kw: (traces, 0.0, None, conv)}
    return {"x": np.array(est), "fun": -1.0, "stats": stats}


@pytest.mark.parametrize("with_points", [True, False])
def test_the_engine_writes_the_new_figures(tmp_path, capsys, with_points):
    from pyantigen.engine.Model_optimize import _save_profile_likelihood_plot
    _save_profile_likelihood_plot(_fake_opt(with_points), ["a", "b"], str(tmp_path), "M", "T")
    out = capsys.readouterr().out
    names = sorted(os.listdir(tmp_path))
    full = [n for n in names if n.startswith("M_T_profile_likelihood_full_") and n.endswith(".png")]
    zoom = [n for n in names if n.startswith("M_T_profile_likelihood_zoom_") and n.endswith(".png")]
    assert len(full) == 1 and len(zoom) == 1 and not [n for n in names if "zoom2" in n]
    assert full[0].split("_full_")[1] == zoom[0].split("_zoom_")[1]            # one timestamp for both, as before
    assert "Profile likelihood saved to" in out and "(zoomed) saved to" in out
    assert any(n.endswith(".txt") for n in names)                       # the summary report is still written


def test_the_engine_writes_the_slice_figures_with_the_same_naming(tmp_path, capsys):
    from pyantigen.engine.Model_optimize import _save_likelihood_slice_plot
    est = np.array([2.0, 0.5])
    grid = {"a": np.array([0.5, 1.0, 2.0, 4.0, 8.0]), "b": np.array([0.1, 0.3, 0.5, 0.8, 1.2])}
    d = {"a": 0.5 * (np.log10(grid["a"] / 2.0) / 0.3) ** 2, "b": 0.5 * ((grid["b"] - 0.5) / 0.2) ** 2}
    opt = {"x": est, "stats": {"likelihood_slice": lambda i, n_points=20, range_factor=2.0: (grid[("a", "b")[i]], d[("a", "b")[i]])}}
    _save_likelihood_slice_plot(opt, ["a", "b"], str(tmp_path), "M", "T")
    names = sorted(os.listdir(tmp_path))
    assert [n for n in names if n.startswith("M_T_likelihood_slice_full_")] and [n for n in names if n.startswith("M_T_likelihood_slice_zoom_")]
    assert not [n for n in names if "zoom2" in n]
    assert "Likelihood slice saved to" in capsys.readouterr().out

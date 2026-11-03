"""The AMICI multi-start's waterfall: rows sorted best to worst, drawn by the collect and by the module on its own."""
import json
import os

import pytest

from pyantigen.engine import Multistart_amici as MA
from pyantigen.engine import Multistart_plot as MP


def _write_starts(directory, fs, converged=None):
    os.makedirs(directory, exist_ok=True)
    for i, f in enumerate(fs):
        ok = True if converged is None else converged[i]
        rec = {"start": i, "x_start": [0.0, 0.0], "x": [0.1 * i, 0.0], "x_lin": [10 ** (0.1 * i), 1.0], "f": f,
               "converged": ok, "verdict": "ok", "pg": 0.0, "evals": 5, "saved": "2026-11-02 12:00:00"}
        with open(os.path.join(directory, f"start_{i}.json"), "w") as fh:
            json.dump(rec, fh)


def test_rows_are_sorted_with_delta_nll_from_the_best(tmp_path):
    _write_starts(str(tmp_path), [-100.0, -103.0, -100.5, -50.0])
    rows, missing = MP.waterfall_rows(MP.read_starts(str(tmp_path)), n_starts=5)
    assert [r["start"] for r in rows] == [1, 2, 0, 3]
    assert [r["rank"] for r in rows] == [0, 1, 2, 3]
    assert rows[0]["dnll"] == 0.0 and rows[-1]["dnll"] == pytest.approx(53.0)
    assert missing == [4]                      # start 4 has no result


def test_non_finite_and_unreadable_starts_are_left_out(tmp_path):
    _write_starts(str(tmp_path), [-10.0, float("nan"), -9.0])
    (tmp_path / "start_9.json").write_text("{not json")
    rows, _ = MP.waterfall_rows(MP.read_starts(str(tmp_path)), n_starts=3)
    assert [r["start"] for r in rows] == [0, 2]


def test_plot_directory_writes_a_png(tmp_path):
    _write_starts(str(tmp_path), [-100.0, -100.2, -80.0], converged=[True, False, True])
    out = MP.plot_directory(str(tmp_path), n_starts=4)
    assert out == str(tmp_path / "waterfall.png")
    assert os.path.getsize(out) > 1000


def test_draw_refuses_an_empty_run(tmp_path):
    with pytest.raises(ValueError):
        MP.draw(str(tmp_path / "w.png"), [])


def test_collect_draws_the_waterfall_and_survives_a_plot_failure(tmp_path, monkeypatch):
    _write_starts(str(tmp_path), [-100.0, -101.0, -99.0])
    logs = []
    MA.collect(str(tmp_path), ["a", "b"], 3, log=logs.append)
    assert (tmp_path / "waterfall.png").exists()
    (tmp_path / "waterfall.png").unlink()

    def boom(*a, **k):
        raise RuntimeError("no display")
    monkeypatch.setattr(MP, "plot_directory", boom)
    logs.clear()
    report = MA.collect(str(tmp_path), ["a", "b"], 3, log=logs.append)
    assert report["best"] == 1                                   # the collect still finished and wrote its files
    assert (tmp_path / "best.json").exists()
    assert any("waterfall plot not drawn" in m for m in logs)

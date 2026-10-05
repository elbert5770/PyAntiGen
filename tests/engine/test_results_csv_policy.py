"""The results CSV holds single-start fits only; a multi-start run goes to its snapshot."""
import glob
import json
import os

import numpy as np
import pandas as pd

from pyantigen.engine.Results import log_optimization_results


def _log(folder, **opt_extra):
    os.makedirs(folder, exist_ok=True)
    opt = {"x": np.array([1.0, 2.0]), "fun": 5.0, "success": True, "message": "ok",
           "stats": {}, "x0": [1.0, 2.0], **opt_extra}
    csv = os.path.join(folder, "res.csv")
    log_optimization_results(opt, ["a", "b"], csv, model_name="m", experiment_id="e",
                             method="Nelder-Mead")
    return csv


def _snapshots(folder):
    return [json.load(open(p, encoding="utf-8")) for p in sorted(glob.glob(os.path.join(folder, "res_*.json")))]


def test_a_single_start_run_appends_its_row_as_before(tmp_path):
    csv = _log(str(tmp_path / "one"))
    assert os.path.exists(csv) and len(pd.read_csv(csv)) == 1
    csv = _log(str(tmp_path / "one"), n_starts=1)
    assert len(pd.read_csv(csv)) == 2                       # appended, not replaced
    (snap, *_) = _snapshots(str(tmp_path / "one"))
    assert snap["metadata"]["n_starts"] == 1


def test_a_multistart_run_writes_no_csv_row_but_writes_its_snapshot(tmp_path):
    folder = str(tmp_path / "multi")
    csv = _log(folder, n_starts=5,
               multistart={"mode": "triage", "verdict": "unimodal", "n_candidates": 64},
               starts=[{"fun": 5.0, "x": [1.0, 2.0], "candidate": 0}])
    assert not os.path.exists(csv)
    (snap,) = _snapshots(folder)
    assert snap["metadata"]["n_starts"] == 5
    assert snap["multistart"]["verdict"] == "unimodal" and snap["parameters"] == {"a": 1.0, "b": 2.0}


def test_n_starts_alone_marks_a_multistart_run(tmp_path):
    # A reused fit whose cache holds no report: the run is still a multi-start one.
    folder = str(tmp_path / "reused")
    csv = _log(folder, n_starts=3, multistart=None, starts=None)
    assert not os.path.exists(csv)
    (snap,) = _snapshots(folder)
    assert snap["metadata"]["n_starts"] == 3 and "multistart" not in snap


def test_a_report_alone_marks_a_multistart_run(tmp_path):
    folder = str(tmp_path / "report_only")
    csv = _log(folder, multistart={"mode": "all-starts", "verdict": "unimodal"})
    assert not os.path.exists(csv)


def test_a_multistart_run_leaves_an_existing_csv_untouched(tmp_path):
    folder = str(tmp_path / "mixed")
    csv = _log(folder)                                      # a single-start row first
    before = open(csv, "rb").read()
    _log(folder, n_starts=4, multistart={"mode": "triage", "verdict": "unimodal"})
    assert open(csv, "rb").read() == before                 # byte for byte

"""``Model_run.py --predict ensemble`` end to end, on a project scaffolded from the template (its Example model): every stage through the real command line.

The profile store is written by hand (``meta.json`` and the profile files the AMICI profile stage writes), so no AMICI and no fit is needed; the members are simulated
on RoadRunner, which is what the stage does by design.
"""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

pytest.importorskip("tellurium")
pytest.importorskip("matplotlib")

import pyantigen  # noqa: E402
from pyantigen.engine import Prediction_ensemble as PE  # noqa: E402
from pyantigen.engine.Profile import JsonlStore  # noqa: E402
from pyantigen.engine.Profile_record import PointRecord  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(pyantigen.__file__)))
NAMES, SCALES = ["k_A_to_B", "SF"], ["log10", "log10"]
OPT = [float(np.log10(0.5)), float(np.log10(2.0))]


@pytest.fixture(scope="module")
def project(tmp_path_factory):
    base = tmp_path_factory.mktemp("proj")
    env = {**os.environ, "PYTHONPATH": ROOT, "MPLBACKEND": "Agg", "OMP_NUM_THREADS": "1"}
    code = "import sys; sys.argv = ['new', 'proj']; from pyantigen.cli import create_project; create_project()"
    out = subprocess.run([sys.executable, "-c", code], cwd=base, capture_output=True, text=True, env=env)
    assert out.returncode == 0, out.stderr
    folder = base / "proj" / "Projects" / "Example"
    store = base / "store"
    (store / "fim").mkdir(parents=True)
    json.dump({"names": NAMES, "scales": SCALES, "x": OPT}, open(store / "meta.json", "w"))
    for k, n in enumerate(NAMES):
        for lab, sgn in (("up", 1), ("down", -1)):
            st = JsonlStore(str(store), f"{k}_{lab}")
            for dx, dn in ((0.02, 0.3), (0.05, 1.2), (0.08, 2.5)):           # the last is outside the level
                x = list(OPT)
                x[k] += sgn * dx
                st.append("profile", PointRecord(param=n, x_fixed=x[k], nll=dn, dnll=dn, verdict="converged", param_idx=k, x=x, source="walk"))
    np.savez(store / "fim" / "fim.npz", cov=np.diag([0.01, 0.02]), names=np.array(NAMES))
    return {"dir": folder, "store": store, "work": base, "env": {**env, "PYTHONPATH": f"{ROOT}:{folder}"}}


def model_run(project, *argv, check=True):
    out = subprocess.run([sys.executable, "Model_run.py", *argv], cwd=project["dir"], capture_output=True, text=True, env=project["env"])
    if check:
        assert out.returncode == 0, out.stdout[-2000:] + out.stderr[-2000:]
    return out


def line(files, member, label):
    with np.load(files[member], allow_pickle=False) as z:
        key = next(k[:-6] for k in z.files if k.endswith(":label") and str(z[k]) == label)
        return z[f"{key}:y"]


def test_the_stages_run_and_a_member_means_what_the_fit_meant(project):
    w = project["work"]
    model_run(project, "--predict", "ensemble", "--predict-stage", "members", "--predict-store", str(project["store"]), "--predict-out", str(w / "members.json"))
    data = json.load(open(w / "members.json"))
    assert len(data["members"]) == 9 and data["members"][0]["param"] == "optimum"
    for task in (0, 1):                                                             # two tasks share the members; a re-run skips what exists
        model_run(project, "--simulate", "Example", "--predict", "ensemble", "--predict-stage", "run", "--predict-members", str(w / "members.json"),
                  "--predict-out", str(w / "m"), "--predict-task", str(task), "--predict-ntasks", "2")
    files = PE.member_files(str(w / "m"))
    assert sorted(files) == list(range(9))
    again = model_run(project, "--simulate", "Example", "--predict", "ensemble", "--predict-stage", "run", "--predict-members", str(w / "members.json"),
                      "--predict-out", str(w / "m"))
    assert "0 member(s) run, 9 already present" in again.stdout
    by = {(m["param"], m["side"], round(m["x_fixed"] - OPT[0 if m["param"] == "k_A_to_B" else 1], 6)): m["id"] for m in data["members"][1:]}
    sf_up, sf_dn = by[("SF", "up", 0.05)], by[("SF", "down", -0.05)]
    b_up, b_dn, b_0 = (line(files, i, "[B] ADpos_Early") for i in (sf_up, sf_dn, 0))
    assert b_up[-1] / b_dn[-1] == pytest.approx(10 ** 0.1, rel=1e-9)               # predicted_B = SF * B / V: the fitted SF reached the model
    assert np.allclose(line(files, sf_up, "[A] ADpos_Early"), line(files, 0, "[A] ADpos_Early"))      # [A] does not depend on SF
    k_up = by[("k_A_to_B", "up", 0.05)]
    assert np.max(np.abs(line(files, k_up, "[A] ADpos_Early") - line(files, 0, "[A] ADpos_Early"))) > 1e-3      # and does depend on k_A_to_B
    assert b_0[-1] > 0

    model_run(project, "--predict", "ensemble", "--predict-stage", "band", "--predict-out", str(w / "m"), "--predict-band", str(w / "band.npz"))
    band = PE.read_band(str(w / "band.npz"))
    assert len(band) == 8 and all(np.all(b["hi"] >= b["lo"]) for b in band.values())
    wide = [k for k, b in band.items() if np.max(b["hi"] - b["lo"]) > 0.05 * np.ptp(b["opt"])]
    assert len(wide) == 8

    out = model_run(project, "--predict", "ensemble", "--predict-stage", "opensides", "--predict-members", str(w / "members.json"),
                    "--predict-out", str(w / "m"), "--predict-band", str(w / "band.npz"))
    assert "open-side plateau check" in out.stdout

    model_run(project, "--predict", "ensemble", "--predict-stage", "fdmembers", "--predict-store", str(project["store"]), "--predict-out", str(w / "fd_members.json"))
    model_run(project, "--simulate", "Example", "--predict", "ensemble", "--predict-stage", "run", "--predict-members", str(w / "fd_members.json"), "--predict-out", str(w / "fd"))
    out = model_run(project, "--predict", "ensemble", "--predict-stage", "delta", "--predict-fd-dir", str(w / "fd"), "--predict-fim", str(project["store"] / "fim" / "fim.npz"),
                    "--predict-band", str(w / "band.npz"), "--predict-out", str(w / "delta.npz"))
    assert (w / "delta.npz").exists() and "delta-method band" in out.stdout

    out = model_run(project, "--simulate", "Example", "--predict", "ensemble", "--predict-stage", "plot", "--predict-band", str(w / "band.npz"), "--predict-tree", "plot1")
    assert "8 band(s) drawn" in out.stdout
    assert "WARNING" in out.stdout                    # the Example's own parameters are not this store's optimum, so the band is not about the drawn lines
    assert list(w.glob("proj/results/predict/plot1/results/Example/*.png"))


def test_a_misspelt_parameter_is_an_error_not_a_member_equal_to_the_optimum(project):
    w = project["work"]
    json.dump({"names": ["k_A_to_B", "kk"], "scales": ["lin", "lin"],
               "members": [{"id": 0, "param": "optimum", "side": "", "x_fixed": None, "dnll": 0.0, "verdict": "converged", "values": {"k_A_to_B": 0.5, "kk": 1.0}}]},
              open(w / "bad.json", "w"))
    out = model_run(project, "--simulate", "Example", "--predict", "ensemble", "--predict-stage", "run", "--predict-members", str(w / "bad.json"),
                    "--predict-out", str(w / "bad"), check=False)
    assert out.returncode != 0 and "change nothing in the model" in out.stderr and "kk" in out.stderr
    assert not list((w / "bad").glob("member_*.npz"))


def test_predict_is_off_by_default_and_refuses_what_it_cannot_do(project):
    out = model_run(project, "--simulate", "Example", "--predict", "ensemble", check=False)
    assert out.returncode == 2 and "--predict-stage" in out.stderr
    out = model_run(project, "--simulate", "Example", "--backend", "amici", "--predict", "ensemble", "--predict-stage", "plot", "--predict-band", "x.npz", check=False)
    assert out.returncode == 2 and "RoadRunner" in out.stderr
    assert "none" in model_run(project, "--help").stdout


def test_the_projects_hooks_run_in_order_and_what_they_return_is_used(project):
    w = project["work"]
    model_run(project, "--predict", "ensemble", "--predict-stage", "members", "--predict-store", str(project["store"]), "--predict-out", str(w / "members_h.json"))
    hooks = project["dir"] / "Modules" / "Predict_hooks.py"
    hooks.write_text(
        "import os\nimport numpy as np\n\n"
        "def prepare_tree(root, stage):\n    open(os.path.join(root, 'prepared_' + stage), 'w').close()\n\n"
        "def member_values(member, label, replicate):\n    v = dict(member['values'])\n    if member['id'] == 3:\n        v['SF'] = 2.0\n    return v\n\n"
        "def before_member(member):\n    return {'extra:v': np.array([float(member['id']), 42.0])}\n\n"
        "def after_band(files, band_path):\n    open(band_path + '.after', 'w').write(str(sorted(files)))\n\n"
        "def before_plot(band_path):\n    open(band_path + '.before_plot', 'w').write('x')\n")
    try:
        model_run(project, "--simulate", "Example", "--predict", "ensemble", "--predict-stage", "run", "--predict-members", str(w / "members_h.json"),
                  "--predict-out", str(w / "hooked"), "--predict-only", "0:3", "--predict-tree", "hooked")
        model_run(project, "--predict", "ensemble", "--predict-stage", "band", "--predict-out", str(w / "hooked"), "--predict-band", str(w / "hooked.npz"))
        model_run(project, "--simulate", "Example", "--predict", "ensemble", "--predict-stage", "plot", "--predict-band", str(w / "hooked.npz"), "--predict-tree", "hookedplot")
    finally:
        hooks.unlink()
    files = PE.member_files(str(w / "hooked"))
    assert sorted(files) == [0, 3]
    with np.load(files[3], allow_pickle=False) as z:
        assert list(z["extra:v"]) == [3.0, 42.0]
    assert (w / "proj" / "results" / "predict" / "hooked_0" / "prepared_run").exists()
    assert (w / "proj" / "results" / "predict" / "hookedplot" / "prepared_plot").exists()
    assert open(w / "hooked.npz.after").read() == "[0, 3]" and (w / "hooked.npz.before_plot").exists()
    # member 3 differs from the optimum only in SF; member_values put SF back to the optimum's 2.0, so its curve is the optimum's
    assert line(files, 3, "[B] ADpos_Early")[-1] == pytest.approx(line(files, 0, "[B] ADpos_Early")[-1], rel=1e-9)


def test_a_replicate_that_pins_a_fitted_parameter_can_say_so_per_replicate(project):
    w = project["work"]
    model_run(project, "--predict", "ensemble", "--predict-stage", "members", "--predict-store", str(project["store"]), "--predict-out", str(w / "members_p.json"))
    data = json.load(open(w / "members_p.json"))
    sf_up = next(m["id"] for m in data["members"][1:] if m["param"] == "SF" and m["side"] == "up")
    hooks = project["dir"] / "Modules" / "Predict_hooks.py"
    hooks.write_text("def member_values(member, label, replicate):\n    v = dict(member['values'])\n    if label.startswith('ADneg'):\n        v.pop('SF')          # the neg arms keep their own SF\n    return v\n")
    try:
        model_run(project, "--simulate", "Example", "--predict", "ensemble", "--predict-stage", "run", "--predict-members", str(w / "members_p.json"),
                  "--predict-out", str(w / "pinned"), "--predict-only", f"0:{sf_up}", "--predict-tree", "pinned")
    finally:
        hooks.unlink()
    files = PE.member_files(str(w / "pinned"))
    # SF moves only the arms that were not pinned: ADneg lines equal the optimum's, ADpos lines do not
    assert np.array_equal(line(files, sf_up, "[B] ADneg_Early"), line(files, 0, "[B] ADneg_Early"))
    assert not np.allclose(line(files, sf_up, "[B] ADpos_Early"), line(files, 0, "[B] ADpos_Early"))

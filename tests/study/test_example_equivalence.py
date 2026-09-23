"""The template Example, written as Studies, scores exactly as its 1.x specs.

Three objectives must agree exactly at x0 for every 1.x spec: the 1.x spec
itself, the Study built in Python, and the Study loaded back from the
committed design JSON (which must also equal what the Python builds).

Scaffolds a fresh project with pyantigen-create, then in a subprocess (so the
project's ``Modules`` package cannot leak between tests) evaluates each 1.x
Optimization spec and the lowered Study at x0 through the same Engine call
and compares the concentrated NLLs for exact equality.
"""
import json
import os
import subprocess
import sys
import textwrap

import pytest

pytest.importorskip("tellurium")

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SCRIPT = textwrap.dedent('''
    import contextlib, io, json
    import AntiGen_paths
    from AntiGen_paths import MODEL_NAME, REPO_ROOT
    from pyantigen.generate.AntimonyGen import AntimonyGen
    from pyantigen.engine.Optimize import run_optimization_from_groups
    from Modules.Experiment import get_EXPERIMENT
    from Modules.Optimizer_settings import get_OPTIMIZATION
    from pyantigen.study import lower, validate, load, to_dict
    from studies.example import (build_example, build_flipflop,
                                 EXAMPLE_ESTIMATIONS, FLIPFLOP_ESTIMATIONS)

    def nll(model_text, paths, exp, spec):
        with contextlib.redirect_stdout(io.StringIO()):
            out = run_optimization_from_groups(
                model_text, paths, exp, param_names=spec.param_names, x0=spec.x0,
                bounds=spec.bounds, method=spec.method,
                optimizer_kwargs=spec.optimizer_kwargs, optimization_spec=spec,
                fit_mode="evaluate_x0", n_workers=1, profile_checkpoint=False,
                preequil_cache=False, reuse_fit=False)
        return out["fun"]

    model_text, paths = AntimonyGen(MODEL_NAME, repo_root=REPO_ROOT)
    rows, problems = [], []
    for build, ests, v1exp in ((build_example, EXAMPLE_ESTIMATIONS, "EXPERIMENT_Example"),
                               (build_flipflop, FLIPFLOP_ESTIMATIONS, "EXPERIMENT_Flipflop")):
        st = build()
        problems += [str(p) for p in validate(st, data_path=paths["data_path"])]
        from_json = load("studies/" + st.name + ".json")
        if to_dict(from_json) != to_dict(st):
            problems.append(st.name + ".json does not match what studies/example.py builds")
        for name, est in ests.items():
            v1 = nll(model_text, paths, get_EXPERIMENT(v1exp),
                     get_OPTIMIZATION("OPTIMIZATION_" + name))
            exp2, spec2 = lower(st, est)
            exp3, spec3 = lower(from_json, est)
            rows.append([name, v1, nll(model_text, paths, exp2, spec2),
                         nll(model_text, paths, exp3, spec3)])
    print("@@" + json.dumps({"rows": rows, "problems": problems}))
''')


def test_example_studies_match_v1_objective(tmp_path):
    env = dict(os.environ, PYTHONPATH=REPO + os.pathsep + os.environ.get("PYTHONPATH", ""))
    subprocess.run([sys.executable, "-c",
                    "import sys; sys.argv=['pyantigen-create','proj']; "
                    "from pyantigen.cli import create_project; create_project()"],
                   cwd=tmp_path, env=env, check=True, capture_output=True)
    proj = tmp_path / "proj" / "Projects" / "Example"
    assert (proj / "studies" / "example.py").exists()
    run = subprocess.run([sys.executable, "-c", SCRIPT], cwd=proj, env=env,
                         capture_output=True, text=True, timeout=900)
    assert run.returncode == 0, run.stderr[-3000:]
    line = [l for l in run.stdout.splitlines() if l.startswith("@@")][-1]
    out = json.loads(line[2:])
    assert out["problems"] == []
    assert [r[0] for r in out["rows"]] == ["Example1_ADpos", "Example1_ADneg",
                                          "Example3_joint", "Example4_flipflop"]
    for name, v1, v2, v3 in out["rows"]:
        assert v1 == v2 == v3, (name, v1, v2, v3)

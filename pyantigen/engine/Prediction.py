"""``Model_run.py --predict ensemble``: the stages of the profile-ensemble prediction band (method and caveats in ``Prediction_ensemble``).

The stages, in the order they are run. Options carry the prefix ``--predict-`` so they cannot collide with a project's own:

    members     --predict-store <profile store> --predict-out members.json [--predict-level 1.9207]
                the parameter vectors of the profile points inside the level, the optimum first
    run         --simulate KEY --predict-members members.json --predict-out <dir> [--predict-task I --predict-ntasks N] [--predict-only 0:1]
                simulate KEY once per member (this task's share: members I, I+N, ...), record the lines of its plot function as <dir>/member_<id>.npz;
                a member whose file exists is skipped, so a requeued task continues
    band        --predict-out <dir> --predict-band band.npz
                the envelope over the members' lines, with the growth of the band as members are added; also <band>.notes.json (annotation ranges)
    plot        --simulate KEY --predict-band band.npz [--predict-note TEXT] [--predict-edges M0:M1|all] [--predict-tree NAME]
                simulate KEY once and draw the bands behind the lines of its plot function (in a private results tree)
    fdmembers   --predict-store <profile store> --predict-out fd_members.json [--predict-delta 0.01]
    delta       --predict-fd-dir <dir of the fd members' files> --predict-fim <store>/fim/fim.npz --predict-band band.npz [--predict-out delta.npz]
                the delta-method band (Fisher information and central differences) and its comparison with the ensemble band
    opensides   --predict-members members.json --predict-out <dir> --predict-band band.npz
                do the curves still move toward the end of each open profile side?

A member is simulated on RoadRunner, the solver the figures are drawn with, and its parameter vector is applied as the fit applies it
(``Model_simulate.run_simulation(member=...)``). The vectors come from an AMICI profile; the difference between the solvers is not part of the band.

An optional ``Modules/Predict_hooks.py`` of the project may define, every name optional:

    prepare_tree(root, stage)        called with the private results tree before anything is simulated in it (stage is 'run' or 'plot')
    member_values(member, label, replicate) -> dict
                                     the parameter vector applied to that replicate, when it is not ``member['values']``: leave out a parameter the replicate pins
                                     on purpose (an arm that is not fitted) or that the project consumes itself; an empty dict leaves the replicate as built
    before_member(member) -> dict    run before a member is simulated; the arrays returned are saved into its file
    after_band(files, band_path)     after the band is written; *files* maps member id to file
    before_plot(band_path)           before the plot stage simulates

Lists in options use ':' or ';' because a ',' is split by the job systems that pass options through an environment list.
"""
import importlib
import json
import os
import re

import numpy as np

from pyantigen.engine import Prediction_ensemble as PE
from pyantigen.engine import Prediction_record as PRec

STAGES = ("members", "run", "band", "plot", "fdmembers", "delta", "opensides")
OFFCENTER_TOL = 1e-6     # a drawn line further than this (as a share of the optimum curve's range) from the band's optimum is reported
TREE_LINKS = ("antimony_models", "antimony_modules", "data", "generated", "pyantigen_settings.json")     # what a private results tree shares with the project's
TREE_DIRS = ("results", "SBML_models", "compiled_models")                                                 # what it has of its own


def add_arguments(parser):
    """Add ``--predict`` and its options to a project's ``Model_run.py`` parser."""
    parser.add_argument("--predict", choices=["none", "ensemble"], default="none",
                        help="Prediction bands from a stored profile (default none). 'ensemble': the profile ensemble, see --predict-stage. "
                             "(An exact prediction profile likelihood, 'exact', is not implemented.)")
    g = parser.add_argument_group("--predict ensemble")
    g.add_argument("--predict-stage", choices=STAGES, help="which stage of the ensemble (required with --predict ensemble)")
    g.add_argument("--predict-store", help="members, fdmembers: the profile store folder (holds meta.json)")
    g.add_argument("--predict-members", help="run, opensides: the members file written by 'members'")
    g.add_argument("--predict-out", help="members, fdmembers: the file written; run, band, opensides: the folder of member files; delta: the delta band file")
    g.add_argument("--predict-band", help="band: the band file written; plot, delta, opensides: the ensemble band file read")
    g.add_argument("--predict-task", type=int, default=0, help="run: this task's index (members I, I+N, ...)")
    g.add_argument("--predict-ntasks", type=int, default=1, help="run: the number of tasks")
    g.add_argument("--predict-only", default="", help="run: only these member ids, ':' separated (a trial)")
    g.add_argument("--predict-level", type=float, default=PE.LEVEL, help="members: the dnll level (default chi2(1, 0.95)/2 = 1.9207)")
    g.add_argument("--predict-tree", default="predict", help="run, plot: name of the ensemble's folder under <results>/predict/; the run tasks use <name>/task_<N>, the plot <name>/plot (a name containing / is used as given for the plot)")
    g.add_argument("--predict-note", default="", help="plot: the footnote under the figure")
    g.add_argument("--predict-edges", default="", help="plot: axes prefixes (M0:M1) whose bands also get two very thin edge lines (white on a dark curve, black on a light one); 'all' gives every band two white edge lines")
    g.add_argument("--predict-delta", type=float, default=0.01, help="fdmembers, delta: the finite-difference step in optimizer space")
    g.add_argument("--predict-fd-dir", help="delta: the folder of the finite-difference members' files")
    g.add_argument("--predict-fim", help="delta: <store>/fim/fim.npz")
    return parser


def check_arguments(args, parser, simulate_keys=None):
    """Argument errors for ``--predict ensemble`` as ``parser.error`` messages (the caller passes its parser)."""
    if args.predict != "ensemble":
        return
    if not args.predict_stage:
        parser.error("--predict ensemble needs --predict-stage " + "|".join(STAGES))
    need = {"members": ("predict_store", "predict_out"), "run": ("simulate", "predict_members", "predict_out"), "band": ("predict_out", "predict_band"),
            "plot": ("simulate", "predict_band"), "fdmembers": ("predict_store", "predict_out"), "delta": ("predict_fd_dir", "predict_fim", "predict_band"),
            "opensides": ("predict_members", "predict_out", "predict_band")}[args.predict_stage]
    missing = ["--" + n.replace("_", "-") for n in need if not getattr(args, n, None)]
    if missing:
        parser.error(f"--predict-stage {args.predict_stage} needs " + ", ".join(missing))
    if args.optimize:
        parser.error("--predict ensemble does not run with --optimize")
    if getattr(args, "backend", None) == "amici":
        parser.error("--predict ensemble simulates on RoadRunner, the solver the figures are drawn with; --backend amici does not apply")


def project_hooks():
    """The project's ``Modules.Predict_hooks`` if it has one, else None. Only that module's absence is tolerated: an error inside it is raised."""
    try:
        return importlib.import_module("Modules.Predict_hooks")
    except ModuleNotFoundError as exc:
        if exc.name in ("Modules.Predict_hooks", "Modules"):
            return None
        raise


def _call(hooks, name, *args):
    """The project hook *name* called with *args* (its result), or None when the project has no hooks module or the hook."""
    fn = getattr(hooks, name, None) if hooks is not None else None
    return fn(*args) if fn is not None else None


def _values_for(hooks, member):
    """The *member* argument of ``setup_simulation`` for one member: its vector for every replicate, or the project's per-replicate choice (``member_values``)."""
    fn = getattr(hooks, "member_values", None) if hooks is not None else None
    if fn is None:
        return member["values"]
    return lambda label, replicate: fn(member, label, replicate)


def tree_name(tree, stage, task=0):
    """The folder under ``results/predict`` of one stage's private tree, so that everything of one ensemble is in one folder: the run tasks are ``<tree>/task_<N>`` and the
    plot is ``<tree>/plot``. A plot tree whose name has its own ``/`` is taken as given (``silk/plot_edges`` for a second drawing of the same band)."""
    if stage == "run":
        return f"{tree}/task_{task}"
    return tree if "/" in tree else f"{tree}/plot"


def private_tree(repo_root, name):
    """A results tree of its own under ``<repo_root>/results/predict/<name>`` that shares the project's models and data by links, so a plot function's side files
    (csv, caches) do not touch the project's results. Returns its root. The ``generated`` folder is shared, so tasks run at the same time write the same events text to it."""
    root = os.path.join(repo_root, "results", "predict", name)
    os.makedirs(root, exist_ok=True)
    for item in TREE_LINKS:
        src, dst = os.path.join(repo_root, item), os.path.join(root, item)
        if os.path.exists(src) and not os.path.lexists(dst):
            os.symlink(src, dst)
    for d in TREE_DIRS:
        os.makedirs(os.path.join(root, d), exist_ok=True)
    return root


def parse_ids(text):
    return {int(v) for v in re.split(r"[,:;]", text or "") if v.strip()}


# ---------------------------------------------------------------------------------------------- stages
def stage_members(a):
    names, scales, optimum, by_side = PE.read_store(a.predict_store)
    members = PE.extract_members(by_side, names, scales, optimum, a.predict_level)
    json.dump({"store": os.path.abspath(a.predict_store), "level": a.predict_level, "names": names, "scales": scales, "members": members}, open(a.predict_out, "w"), indent=1)
    from collections import Counter
    print(f"{len(members)} members (the optimum and {len(members) - 1} profile points with dnll <= {a.predict_level:g}) -> {a.predict_out}")
    print("  by parameter:", dict(Counter(m["param"] for m in members)))
    print(f"  unconverged: {sum(1 for m in members if m['verdict'] != 'converged')}")


def stage_fdmembers(a):
    names, scales, optimum, _ = PE.read_store(a.predict_store)
    members = PE.fd_members(names, scales, optimum, a.predict_delta)
    json.dump({"store": os.path.abspath(a.predict_store), "delta": a.predict_delta, "names": names, "scales": scales, "members": members}, open(a.predict_out, "w"), indent=1)
    print(f"{len(members)} finite-difference members (the optimum and +-{a.predict_delta:g} on each of {len(names)} parameters) -> {a.predict_out}")


def stage_run(a, experiment_dict, run_settings, hooks):
    from pyantigen.engine.Model_simulate import setup_simulation
    data = json.load(open(a.predict_members))
    only = parse_ids(a.predict_only)
    chosen = [m for m in data["members"] if (not only or m["id"] in only)]
    mine = chosen[a.predict_task::a.predict_ntasks]
    os.makedirs(a.predict_out, exist_ok=True)
    root = private_tree(run_settings["REPO_ROOT"], tree_name(a.predict_tree, "run", a.predict_task))
    _call(hooks, "prepare_tree", root, "run")
    settings = dict(run_settings, REPO_ROOT=root, backend="roadrunner", Verbose=False)
    fig_config = experiment_dict[a.simulate]
    done = 0
    for m in mine:
        if os.path.exists(os.path.join(a.predict_out, f"member_{m['id']}.npz")):
            continue
        print(f"[ensemble] member {m['id']} ({m['param']} {m['side']} dnll {m['dnll']:.3f})", flush=True)
        extra = _call(hooks, "before_member", m)
        recorder = PRec.record_member(a.predict_out, m, fig_config["plot"], extra)
        setup_simulation(settings, dict(fig_config, plot=recorder), member=_values_for(hooks, m))
        done += 1
    print(f"[ensemble] task {a.predict_task}: {done} member(s) run, {len(mine) - done} already present", flush=True)


def stage_band(a, hooks=None):
    by_id = PE.member_files(a.predict_out)
    files = list(by_id.values())
    if not files:
        raise SystemExit(f"no member files in {a.predict_out}")
    band = PE.build_band(files)
    PE.write_band(band, a.predict_band)
    rows = PE.band_table(band)
    print(f"{len(files)} member file(s), {len(band)} line(s) recorded, {len(rows)} with a band -> {a.predict_band}")
    print(f"  {'key':8s} {'label':34s} {'max width':>11s} {'line range':>11s} {'relative':>9s}  mean width at half / all members")
    for key, label, w, span, rel, half, full in rows:
        print(f"  {key:8s} {label[:34]:34s} {w:11.4g} {span:11.4g} {100 * rel:8.3f}%  {half:.4g} / {full:.4g}")
    notes = PE.annotation_ranges(files)
    if notes:
        print(f"{len(notes)} annotation(s) (numbers the plot function writes on the figure), range over the members:")
        for key, v in sorted(notes.items(), key=lambda kv: PE.curve_order(kv[0])):
            print(f"  {key:10s} text {v['label']:>8s}   optimum {v['x0']:9.3f}   range {v['lo']:9.3f} to {v['hi']:9.3f}   ({v['n']} members)")
        json.dump(notes, open(a.predict_band + ".notes.json", "w"), indent=1)
    _call(hooks, "after_band", by_id, a.predict_band)


def stage_plot(a, experiment_dict, run_settings, hooks):
    from pyantigen.engine.Model_simulate import setup_simulation
    band = PE.read_band(a.predict_band)
    note = a.predict_note or "Shaded: pointwise 95% band from the profile likelihood (parameter sets of the stored profile with dnll <= 1.92; bands narrower than the line are hidden by it)"
    root = private_tree(run_settings["REPO_ROOT"], tree_name(a.predict_tree, "plot"))
    _call(hooks, "prepare_tree", root, "plot")
    _call(hooks, "before_plot", a.predict_band)
    settings = dict(run_settings, REPO_ROOT=root, backend="roadrunner")
    edges = [v for v in re.split(r"[,:;]", a.predict_edges) if v.strip()]
    with PRec.BandDrawer(band, note=note, edges=edges) as drawer:
        setup_simulation(settings, experiment_dict[a.simulate])
    print(f"[ensemble] {drawer.drawn} band(s) drawn, {drawer.edged} of them with thin edge lines; figures under {os.path.join(root, 'results')}")
    off = [(k, d) for k, d in drawer.offcenter if d > OFFCENTER_TOL]
    if off:
        worst = max(off, key=lambda kd: kd[1])
        print(f"[ensemble] WARNING: {len(off)} of {len(drawer.offcenter)} drawn line(s) differ from the optimum curve the band was built around by more than "
              f"{OFFCENTER_TOL:g} of its range (largest: {worst[0]}, {worst[1]:.3g}). The band is about the stored profile's optimum, so the project's own parameters "
              f"are not that optimum, the plot function draws something that changed since the members were run, or an arm pins a fitted parameter on purpose "
              f"and member_values does not say so (see docs/PREDICTION.md).")


def stage_delta(a):
    fim = np.load(a.predict_fim, allow_pickle=False)
    cov, names = np.asarray(fim["cov"], float), [str(n) for n in fim["names"]]
    files = PE.member_files(a.predict_fd_dir)
    need = set(range(1 + 2 * len(names)))
    if not need <= set(files):
        raise SystemExit(f"{len(need - set(files))} finite-difference member file(s) missing in {a.predict_fd_dir}")
    dlt = PE.delta_band(files, len(names), a.predict_delta, cov)
    ens = PE.read_band(a.predict_band)
    with np.load(files[0], allow_pickle=False) as z:
        labels = {k.rsplit(":", 1)[0]: str(z[k]) for k in z.files if k.endswith(":label")}
    for k, v in ens.items():
        v["label"] = labels.get(k, "")
    rows = PE.compare_bands(ens, dlt)
    print(f"delta-method band (central differences +-{a.predict_delta:g}, 95% = 1.96 sd) against the ensemble band, {len(rows)} line(s) with a band; ratio = delta width / ensemble width (max over time)")
    print(f"  {'key':8s} {'label':30s} {'ensemble':>10s} {'delta':>10s} {'ratio':>7s}  {'ens inside delta':>16s}  ensemble width / line range")
    for key, label, we, wd, ratio, inside, rel in rows:
        print(f"  {key:8s} {label[:30]:30s} {we:10.4g} {wd:10.4g} {ratio:7.2f}  {100 * inside:15.0f}%  {100 * rel:.3f}%")
    if a.predict_out:
        PE.write_band({k: {"t": v["t"], "lo": v["lo"], "hi": v["hi"], "opt": v["opt"]} for k, v in dlt.items()}, a.predict_out)
        print(f"delta band written to {a.predict_out}")


def stage_opensides(a):
    data = json.load(open(a.predict_members))
    files = PE.member_files(a.predict_out)
    rows = PE.open_side_check(data, files, PE.read_band(a.predict_band))
    print("open-side plateau check: largest change of a curve between the farthest member of a side and the member at a fraction of the way out, as a share of the line's band width")
    print("  (a share near 0: the curve has stopped moving; near 1: where the box ends still sets the band)")
    print(f"  {'parameter':24s} {'side':5s} {'decades':>8s} {'frac':>5s} {'key':8s} {'label':26s} {'change':>10s} {'band':>10s} {'share':>6s}")
    worst = {}
    for param, side, reach, key, label, frac, change, width, ratio in rows:
        k = (param, side, frac)
        if ratio > worst.get(k, (0,))[0]:
            worst[k] = (ratio, reach, key, label, change, width)
    for (param, side, frac), (ratio, reach, key, label, change, width) in sorted(worst.items(), key=lambda kv: -kv[1][0]):
        print(f"  {param:24s} {side:5s} {reach:8.2f} {frac:5.2f} {key:8s} {label[:26]:26s} {change:10.4g} {width:10.4g} {ratio:6.2f}")
    print("  (worst line of each side and fraction)")


def run_predict(args, experiment_dict, run_settings):
    """Run the stage ``args.predict_stage``. *experiment_dict* is the project's ``EXPERIMENT_dict``; *run_settings* holds at least ``MODEL_NAME`` and ``REPO_ROOT``."""
    hooks = project_hooks()
    stage = args.predict_stage
    if stage == "members":
        stage_members(args)
    elif stage == "fdmembers":
        stage_fdmembers(args)
    elif stage == "run":
        stage_run(args, experiment_dict, run_settings, hooks)
    elif stage == "band":
        stage_band(args, hooks)
    elif stage == "plot":
        stage_plot(args, experiment_dict, run_settings, hooks)
    elif stage == "delta":
        stage_delta(args)
    elif stage == "opensides":
        stage_opensides(args)
    else:
        raise ValueError(f"unknown stage {stage!r}")

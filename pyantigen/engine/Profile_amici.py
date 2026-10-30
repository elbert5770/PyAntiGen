"""The profile stage on AMICI: the walk / fill / recheck protocol with the Levenberg-Marquardt solver.

One *side* is a parameter and a direction. A side is the unit of work and of resumption, so the stage runs either way:

* ``side=None``  every side in turn, in this process (the default; right when there is one job slot);
* ``side=task``  one side, ``task = 2*k + (0 for up | 1 for down)``, so a Slurm array of ``2k`` tasks profiles in parallel.
  ``PYANTIGEN_PROFILE_SIDE`` carries the task id into ``Model_run.py`` (``--profile-side``).

Each side writes ``profile_<k>_<dir>.jsonl`` (and ``fill_``, ``recheck_``) into *store_dir* and a ``.done`` file when a pass
finishes, so a killed or timed-out task is relaunched with the same command and continues. The floors the points were scored
under are written once to ``floors.pkl`` in *store_dir*; every task loads that file, so all sides see one objective and every
record carries the file's stamp. Run again with ``side=None`` after an array: finished sides are skipped, and the traces,
the convergence report and the verdicts are assembled from every side's files.

AMICI solver objects are not thread-safe, which is why this is one side per process and not one thread per side as the
RoadRunner route does.
"""
import json
import math
import os

import numpy as np

from pyantigen.engine import Profile as PR
from pyantigen.engine import Profile_lm as LM
from pyantigen.engine.Profile_record import curve

SIDE_ENV = "PYANTIGEN_PROFILE_SIDE"
KINDS = ("profile", "pins", "relax", "confirm", "fill", "dense", "recheck", "backfill")
MODE_ENV = "PYANTIGEN_PROFILE_MODE"          # walk (default) | pins | fim | collect (walks nothing: reports what the sides left)
ANCHOR_TOL = 1e-4                            # optimizer-space distance at which a refit is a different anchor
BACKFILL_ENV = "PYANTIGEN_PROFILE_BACKFILL"  # converged points wanted inside the level on each side; 0 turns the last pass off
PINS_ENV = "PYANTIGEN_PROFILE_PINS"          # comma-separated relative pins, e.g. 0.1,0.5,2,10
FAILURE_VALUE = 1e10


def side_from_env(n_params, env=None):
    """``(k, direction)`` of the side named in ``PYANTIGEN_PROFILE_SIDE``, or None for every side."""
    raw = (os.environ if env is None else env).get(SIDE_ENV)
    if raw is None or str(raw).strip() == "":
        return None
    task = int(raw)
    if not 0 <= task < 2 * n_params:
        raise ValueError(f"{SIDE_ENV}={task}: this problem has {n_params} parameters, so sides are 0..{2 * n_params - 1} "
                         f"(2*k for the upper side of parameter k, 2*k+1 for the lower)")
    return task // 2, (1 if task % 2 == 0 else -1)


def settings_from_env(env=None):
    """``(mode, WalkSettings)`` from ``PYANTIGEN_PROFILE_MODE`` and ``PYANTIGEN_PROFILE_PINS``."""
    env = os.environ if env is None else env
    mode = (env.get(MODE_ENV) or "walk").strip().lower()
    if mode not in ("walk", "pins", "fim", "collect"):
        raise ValueError(f"{MODE_ENV}={mode!r}: expected 'walk', 'pins', 'fim' or 'collect'")
    st = PR.WalkSettings()
    if env.get(PINS_ENV):
        st.pins = tuple(float(v) for v in env[PINS_ENV].split(",") if v.strip())
    if env.get(BACKFILL_ENV, "").strip():
        st.backfill_points = int(env[BACKFILL_ENV])
    return mode, st


def all_sides(n_params):
    return [(k, d) for k in range(n_params) for d in (1, -1)]


def task_of(k, direction):
    """The array task (``PYANTIGEN_PROFILE_SIDE``) of a side: ``2*k`` for the upper side, ``2*k + 1`` for the lower."""
    return 2 * k + (0 if direction > 0 else 1)


def unfinished_tasks(finished, n_params):
    """Array task ids of the sides *finished* (one flag per side, in :func:`all_sides` order) says are not done."""
    return [task_of(k, d) for (k, d), ok in zip(all_sides(n_params), finished) if not ok]


def write_unfinished(store_dir, tasks):
    """``unfinished.txt``: the tasks still to run as a comma list, ready for ``sbatch --array=``; empty when none are."""
    from pyantigen.engine import Unfinished
    Unfinished.write(store_dir, tasks)


def read_meta(store_dir):
    p = os.path.join(store_dir, "meta.json")
    return json.load(open(p)) if os.path.exists(p) else None


def check_anchor(store_dir, x, tol=ANCHOR_TOL, fit_method=None):
    """Refuse to continue a profile that was anchored at a different optimum than *x* (optimizer space).

    The profile directory is named for the problem, not the optimum, so a refit lands in the same place; every stored
    dnll is measured from the optimum the first side saw (``meta.json``), and mixing two would be silent. ``meta.json``
    names the fit method that found that optimum, so the error says when the two differ (two methods stop at slightly
    different points of the same minimum) and a refit by the same method is told apart from a different optimum.
    """
    meta = read_meta(store_dir)
    if meta is None:
        return
    saved = np.asarray(meta["x"], float)
    xa = np.asarray(x, float)
    if saved.shape != xa.shape or float(np.max(np.abs(saved - xa))) > tol:
        far = "different length" if saved.shape != xa.shape else f"max |dx| {np.max(np.abs(saved - xa)):.3g}"
        was = meta.get("fit_method")
        who = (f" It was anchored by a {was} fit and this one is {fit_method}: the methods stop at slightly different points "
               "of the same minimum." if was and fit_method and was != fit_method else
               f" Both fits are {fit_method}: the optimum itself moved." if was and was == fit_method else "")
        raise RuntimeError(f"{store_dir} holds a profile anchored at a different optimum ({far} > {tol:g}, optimizer space)."
                           f"{who} Move or delete the directory to profile the new optimum from scratch.")


def record_anchor(store_dir, f0):
    """Add the objective at the optimum to ``meta.json`` once, so a collect run needs no evaluation to know the anchor."""
    meta = read_meta(store_dir)
    if meta is None or "f" in meta:
        return
    tmp = os.path.join(store_dir, f"meta.json.{os.getpid()}.tmp")
    with open(tmp, "w") as fh:
        json.dump({**meta, "f": float(f0)}, fh)
    os.replace(tmp, os.path.join(store_dir, "meta.json"))


def tag_of(k, direction, mode="walk"):
    """File tag of a side; pins mode has its own files, so a walk's fill and relax points never leak into a pins run."""
    return f"{'p' if mode == 'pins' else ''}{k}_{'up' if direction > 0 else 'down'}"


def use_floors(store_dir, log=print):
    """Make every task of this profile see the same floors; returns their stamp.

    The first task writes the calibrated floors it holds to ``floors.pkl``; a later one loads that file instead of
    trusting its own calibration (which was made against whatever point it evaluated first).
    """
    from pyantigen.engine import Floors, Noise_floor
    os.makedirs(store_dir, exist_ok=True)
    path = os.path.join(store_dir, "floors.pkl")
    if not os.path.exists(path):
        tmp = f"{path}.{os.getpid()}.tmp"
        Floors.save(Noise_floor.export_cache(), tmp)
        try:
            os.link(tmp, path)               # fails if another task got there first; theirs then wins
        except FileExistsError:
            pass
        finally:
            os.remove(tmp)
    _snapshot, stamp = Floors.load(path)
    log(f"[profile] floors {stamp} ({path})")
    return stamp


def optimum_state(evaluate, x, lb, ub, failures=()):
    """``(f, sd, worst_gain)`` at the optimum *x*: the objective, the Wald lower bound on each sd (optimizer space) from
    ``J'J``, and the largest expected gain, in nats, from the projected gradient (about 0 at a stationary point)."""
    out = evaluate(np.asarray(x, float))
    if out is None:
        why = f": {failures[-1]}" if failures else ""
        raise RuntimeError(f"AMICI could not evaluate the objective at the optimum{why}")
    f, r, J = out
    cov = np.linalg.pinv(J.T @ J)
    sd = np.sqrt(np.maximum(np.diag(cov), 0.0))
    g = J.T @ r
    xa = np.asarray(x, float)
    g = np.where((xa <= lb + 1e-12) & (g > 0), 0.0, np.where((xa >= ub - 1e-12) & (g < 0), 0.0, g))
    return float(f), sd, float(np.max(np.abs(g) * sd))


def record_to_raw(rec, names, scales):
    """A :class:`PointRecord` as the raw point dict the reporting functions of ``Optimize`` read."""
    k = rec.param_idx if rec.param_idx is not None else names.index(rec.param)
    nuis = None if rec.x is None else [v for j, v in enumerate(rec.x) if j != k]
    value = rec.value if rec.value is not None else (10.0 ** rec.x_fixed if scales[k] == "log10" else rec.x_fixed)
    return {"param_name": rec.param, "param_idx": k, "x_fixed": float(rec.x_fixed), "x_fixed_linear": float(value),
            "nll": rec.nll, "dnll": rec.dnll, "status": "ok", "converged": rec.verdict == "converged",
            "interrupted": bool(rec.interrupted), "nuisance_x": nuis, "n_evals": rec.evals or 0,
            "wall_s": rec.seconds or 0.0, "direction": rec.direction, "pass_label": rec.source,
            "backend": "amici", "solver": rec.solver, "bound_active": list(rec.bound_active),
            "pin_outside": bool(rec.pin_outside)}


def read_sides(store_dir, names, sides=None, mode="walk"):
    """Every stored record, ``{param: [PointRecord]}``, and one flag per side: all passes of the side finished."""
    by_param = {n: [] for n in names}
    for k, d in (sides or all_sides(len(names))):
        store = PR.JsonlStore(store_dir, tag_of(k, d, mode))
        for kind in KINDS:
            by_param[names[k]].extend(store.read(kind))
    return by_param


def sides_finished(store_dir, names, recheck=False, sides=None, mode="walk", backfill_points=None):
    """One flag per side: every pass is done. With a ``backfill_points`` target a side counts only once its backfill ran for
    that target; with none (or 0) the backfill is not asked for and a side is finished without it."""
    flags = []
    for k, d in (sides or all_sides(len(names))):
        store = PR.JsonlStore(store_dir, tag_of(k, d, mode))
        filled = store.done("backfill")
        ok = not backfill_points or (filled is not None and filled.startswith(f"target {int(backfill_points)}:"))
        flags.append(bool(store.done("pins" if mode == "pins" else "profile")) and bool(store.done("fill")) and bool(store.done("dense"))
                     and (not recheck or bool(store.done("recheck"))) and ok)
    return flags


def completed_from_store(store_dir, names, scales, key, mode="walk"):
    """The legacy ``{param: {key(x_fixed): raw}}`` dict from every side's files; one point per value, the lowest dnll."""
    completed = {n: {} for n in names}
    for name, recs in read_sides(store_dir, names, mode=mode).items():
        for rec in curve(recs):
            completed[name][key(rec.x_fixed)] = record_to_raw(rec, names, scales)
    return completed


def side_report(by_param, names, bounds, level=None):
    """Per parameter and side: the verdict, the 95% limit, and what the declared box did to the profile.

    ``{param: {"below"|"above": {"verdict", "limit", "bound_active", "needs_wider": {nuisance: (lo, hi)}}}}``
    ``bound_active`` names the nuisances on a bound at the points that fix the crossing (a box-limited side is only an upper
    bound on the true limit). ``needs_wider`` lists nuisances whose relaxed fit left the declared bounds, with the range it
    used, so the box can be widened on purpose. Records of the other side and the optimum itself are ignored.
    """
    from pyantigen.engine.Profile_record import DNLL_95, crossings, side_verdicts
    level = DNLL_95 if level is None else level
    lb = [(-math.inf if b is None or b[0] is None else b[0]) for b in bounds]
    ub = [(math.inf if b is None or b[1] is None else b[1]) for b in bounds]
    out = {}
    for k, name in enumerate(names):
        recs = by_param.get(name, [])
        opt = [r for r in recs if r.source == "optimum"]
        if not opt:
            continue
        best = opt[0].x_fixed
        verdicts = side_verdicts(recs, best, level)
        limits = crossings(recs, best, level)
        out[name] = {}
        for label, verdict, limit, side in zip(("below", "above"), verdicts, limits, (-1, 1)):
            pts = sorted((r for r in curve(recs) if (r.x_fixed - best) * side > 0), key=lambda r: abs(r.x_fixed - best))
            over = next((r for r in pts if r.dnll > level), None)
            under = [r for r in pts if r.dnll <= level][-1:] 
            active = sorted({n for r in ([over] if over else []) + under for n in r.bound_active})
            wider = {}
            for r in recs:
                if r.source and r.source.startswith("relax") and r.x is not None and (r.x_fixed - best) * side > 0:
                    for i, xi in enumerate(r.x):
                        if i != k and (xi < lb[i] - 1e-9 or xi > ub[i] + 1e-9):
                            lo, hi = wider.get(names[i], (xi, xi))
                            wider[names[i]] = (min(lo, xi), max(hi, xi))
            out[name][label] = {"verdict": verdict, "limit": limit, "bound_active": active, "needs_wider": wider}
    return out


def format_side_report(report):
    """The lines :func:`side_report` prints."""
    lines = []
    for name, sides in report.items():
        for label, d in sides.items():
            if d["verdict"] == "crossed" and not d["needs_wider"]:
                continue
            what = {"box-limited": "BOX-LIMITED (the limit is only an upper bound on the interval)",
                    "open": "OPEN (no point past the threshold)", "crossed": "crossed"}[d["verdict"]]
            lines.append(f"  {name} {label}: {what}")
            if d["bound_active"]:
                lines.append(f"      nuisance on a bound at the crossing: {', '.join(d['bound_active'])}")
            for n, (lo, hi) in d["needs_wider"].items():
                lines.append(f"      relaxed fit left the declared box: {n} in [{lo:.4g}, {hi:.4g}] (optimizer space): widen its bound")
    return lines


def run(provider, names, bounds, scales, res_x, store_dir, side=None, recheck=False, budget=None, settings=None, log=print, mode="walk",
        fit_method=None):
    """Walk, fill and (with *recheck*) recheck the chosen sides. Returns ``{"anchor": f, "floors": stamp, "ran": [...]}``.

    *provider* has ``linear_residual_jacobian`` (an ``Analytic_gradient.AmiciGradient``); *bounds* are ``(lo, hi)`` in the
    optimizer's space, None for none. Sides that run out of the clock are left resumable, and the rest are not started.
    """
    if mode == "collect":
        # Nothing is evaluated and nothing walked: the anchor and floors are what the sides recorded.
        meta = read_meta(store_dir)
        if meta is None or "f" not in meta:
            raise RuntimeError(f"collect: {store_dir} has no anchor yet; no side has started (run the walk first)")
        check_anchor(store_dir, res_x, fit_method=fit_method)
        return {"anchor": float(meta["f"]), "floors": meta["floors"], "ran": []}
    n = len(names)
    lb = np.array([(-math.inf if b is None or b[0] is None else b[0]) for b in bounds], float)
    ub = np.array([(math.inf if b is None or b[1] is None else b[1]) for b in bounds], float)
    xb = np.asarray(res_x, float)
    stamp = use_floors(store_dir, log)
    meta = {"names": list(names), "scales": list(scales), "x": [float(v) for v in xb], "lb": [None if not math.isfinite(v) else float(v) for v in lb],
            "ub": [None if not math.isfinite(v) else float(v) for v in ub], "floors": stamp, "fit_method": fit_method}
    check_anchor(store_dir, xb, fit_method=fit_method)
    mp = os.path.join(store_dir, "meta.json")
    if not os.path.exists(mp):                      # what Profile_plot reads to draw this folder
        tmp = f"{mp}.{os.getpid()}.tmp"
        with open(tmp, "w") as fh:
            json.dump(meta, fh)
        os.replace(tmp, mp)
    failures = []
    # amici_evaluate returns None on a failed integration and says why through its log; keep the reasons, drop the
    # per-evaluation chatter (the walk reports each point).
    evaluate = LM.amici_evaluate(provider, scales, log=lambda m: failures.append(m) if "failed" in m else None)
    if mode == "fim":
        # Only the Fisher information at the optimum: one evaluation, no walk.
        from pyantigen.engine import Fim
        out = evaluate(xb)
        if out is None:
            raise RuntimeError("AMICI could not evaluate the objective at the optimum"
                               + (f": {failures[-1]}" if failures else ""))
        rep = Fim.fim_report(out[2], names, scales, xb)
        for line in Fim.format_fim(rep, out[0]):
            log(line)
        path = Fim.save_fim(rep, os.path.join(store_dir, "fim"), "fim", out[0])
        log(f"[fim] written to {path} (and fim.npz)")
        return {"anchor": float(out[0]), "floors": stamp, "ran": [], "fim": path}
    f0, sd, worst = optimum_state(evaluate, xb, lb, ub, failures)
    record_anchor(store_dir, f0)
    log(f"[profile] AMICI objective at the optimum: f {f0:.6f}; largest projected-gradient gain {worst:.3g} nats")
    if worst > 0.05:
        log("[profile] *** the optimum is not stationary under these floors; a LOWER point is likely and the profile "
            "is measured from a point that is not the minimum ***")
    solver = LM.LmSolver(evaluate, names, scales, lb, ub, floors_sha256=stamp, log=None)
    to_lin = lambda i: float(10.0 ** xb[i]) if scales[i] == "log10" else float(xb[i])
    pr = PR.ProfileProblem(names=list(names), lb=lb, ub=ub, best_x=xb, anchor_nll=f0, solver=solver, best_value=to_lin,
                           sd_first=sd, floors_sha256=stamp)
    st = settings or PR.WalkSettings()
    deadline = budget.work_deadline() if budget is not None else None
    todo = all_sides(n) if side is None else [side]
    ran = []
    for k, d in todo:
        tag = tag_of(k, d, mode)
        store = PR.JsonlStore(store_dir, tag)
        say = lambda m, tag=tag: log(f"  [walk {tag}] {m}")
        kind = "pins" if mode == "pins" else "profile"
        _pts, why = (PR.pins(pr, k, d, st, store, deadline, say) if mode == "pins"
                     else PR.walk(pr, k, d, st, store, deadline, say))
        if why is None:
            log(f"[profile] out of time in {tag}; relaunch to continue")
            break
        if mode == "pins":                           # the walk widens bounds as it goes; a pin grid cannot
            PR.relax(pr, k, d, st, store, deadline, say, walk_kind=kind)
            if store.done("relax") is None and st.relax > 0:
                log(f"[profile] out of time relaxing {tag}; relaunch to continue")
                break
        if PR.fill(pr, k, d, st, store, deadline, say, walk_kind=kind) is None and store.done("fill") is None:
            log(f"[profile] out of time filling {tag}; relaunch to continue")
            break
        PR.densify(pr, k, d, st, store, deadline, say, walk_kind=kind)
        if store.done("dense") is None:
            log(f"[profile] out of time densifying {tag}; relaunch to continue")
            break
        if recheck:
            PR.recheck(pr, k, d, st, store, deadline, say, walk_kind=kind)
        PR.backfill(pr, k, d, st, store, deadline, say, walk_kind=kind)
        if st.backfill_points > 0 and store.done("backfill") is None:
            log(f"[profile] out of time backfilling {tag}; relaunch to continue")
            break
        saved = store.state("bounds")
        for i, n_i in enumerate(names):
            if saved and (saved["lb"][i] != (None if not math.isfinite(lb[i]) else lb[i])
                          or saved["ub"][i] != (None if not math.isfinite(ub[i]) else ub[i])):
                log(f"[profile] {tag}: bound of {n_i} widened to [{saved['lb'][i]}, {saved['ub'][i]}] "
                    f"(declared [{lb[i]}, {ub[i]}], optimizer space)")
        ran.append(tag)
    return {"anchor": f0, "floors": stamp, "ran": ran}

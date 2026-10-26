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
import math
import os

import numpy as np

from pyantigen.engine import Profile as PR
from pyantigen.engine import Profile_lm as LM
from pyantigen.engine.Profile_record import curve

SIDE_ENV = "PYANTIGEN_PROFILE_SIDE"
KINDS = ("profile", "fill", "recheck")
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


def all_sides(n_params):
    return [(k, d) for k in range(n_params) for d in (1, -1)]


def tag_of(k, direction):
    return f"{k}_{'up' if direction > 0 else 'down'}"


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


def read_sides(store_dir, names, sides=None):
    """Every stored record, ``{param: [PointRecord]}``, and one flag per side: all passes of the side finished."""
    by_param = {n: [] for n in names}
    for k, d in (sides or all_sides(len(names))):
        store = PR.JsonlStore(store_dir, tag_of(k, d))
        for kind in KINDS:
            by_param[names[k]].extend(store.read(kind))
    return by_param


def sides_finished(store_dir, names, recheck=False, sides=None):
    flags = []
    for k, d in (sides or all_sides(len(names))):
        store = PR.JsonlStore(store_dir, tag_of(k, d))
        flags.append(bool(store.done("profile")) and bool(store.done("fill")) and (not recheck or bool(store.done("recheck"))))
    return flags


def completed_from_store(store_dir, names, scales, key):
    """The legacy ``{param: {key(x_fixed): raw}}`` dict from every side's files; one point per value, the lowest dnll."""
    completed = {n: {} for n in names}
    for name, recs in read_sides(store_dir, names).items():
        for rec in curve(recs):
            completed[name][key(rec.x_fixed)] = record_to_raw(rec, names, scales)
    return completed


def run(provider, names, bounds, scales, res_x, store_dir, side=None, recheck=False, budget=None, settings=None, log=print):
    """Walk, fill and (with *recheck*) recheck the chosen sides. Returns ``{"anchor": f, "floors": stamp, "ran": [...]}``.

    *provider* has ``linear_residual_jacobian`` (an ``Analytic_gradient.AmiciGradient``); *bounds* are ``(lo, hi)`` in the
    optimizer's space, None for none. Sides that run out of the clock are left resumable, and the rest are not started.
    """
    n = len(names)
    lb = np.array([(-math.inf if b is None or b[0] is None else b[0]) for b in bounds], float)
    ub = np.array([(math.inf if b is None or b[1] is None else b[1]) for b in bounds], float)
    xb = np.asarray(res_x, float)
    stamp = use_floors(store_dir, log)
    failures = []
    # amici_evaluate returns None on a failed integration and says why through its log; keep the reasons, drop the
    # per-evaluation chatter (the walk reports each point).
    evaluate = LM.amici_evaluate(provider, scales, log=lambda m: failures.append(m) if "failed" in m else None)
    f0, sd, worst = optimum_state(evaluate, xb, lb, ub, failures)
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
        tag = tag_of(k, d)
        store = PR.JsonlStore(store_dir, tag)
        say = lambda m, tag=tag: log(f"  [walk {tag}] {m}")
        _pts, why = PR.walk(pr, k, d, st, store, deadline, say)
        if why is None:
            log(f"[profile] out of time in {tag}; relaunch to continue")
            break
        if PR.fill(pr, k, d, st, store, deadline, say) is None and store.done("fill") is None:
            log(f"[profile] out of time filling {tag}; relaunch to continue")
            break
        if recheck:
            PR.recheck(pr, k, d, st, store, deadline, say)
        ran.append(tag)
    return {"anchor": f0, "floors": stamp, "ran": ran}

"""Multi-start on AMICI: one bounded Levenberg-Marquardt fit per start, one array task each, then a collect.

The Nelder-Mead multi-start (``Optimize._run_multistart``) runs every start in one process through the worker pool. An AMICI
fit is one process per fit (the solver objects are not thread-safe, and a fit takes hours), so the starts are spread over a
Slurm array the way the profile sides are. The spec says what the starts are (``n_starts``, ``search_decades``,
``start_seed``, ``start_sampler``; see ``docs/MULTISTART.md``); start 0 is always the spec's own x0.

``PYANTIGEN_MULTISTART`` picks what a run does, in the fit stage of a spec whose method is LM:

* ``<i>``       fit from start ``i`` and write ``start_<i>.json``. A relaunch skips a converged start and continues an
                unconverged one from where it stopped, so a killed or timed-out task is run again with the same command;
* ``collect``   fit nothing: read every start, say which are missing or unconverged (``unfinished.txt``, the task ids for
                ``sbatch --array=``), report the basins, and write ``best.json`` in the shape of a saved fit
                (``{"complete": {"parameters", "fun", ...}}``), so the winner is read like any other fit, and draw ``waterfall.png`` (the
                starts' final ΔNLL, best to worst; ``Multistart_plot`` redraws it from the directory).

The files live in ``<plot_path>/multistart/<fit id>/``, named for the fit problem like the fit cache, and the fit cache
itself is never written: a start that is not x0 must not become the cached fit of the problem.

One objective for every start. With ``PYANTIGEN_SIGMA_FLOOR=fixed`` the noise floors are the sigmas, and each is calibrated
against the model's shape at the first point a process evaluates. Left alone every task would calibrate at its own random start,
and the starts' f values would be values of different functions. So the first task calibrates at start 0 (the spec's x0) and
writes ``floors.pkl`` (the way a profile does, ``Profile_amici.use_floors``); every task loads that file before its first
fit evaluation; each start's record carries the file's stamp; and ``collect`` ranks only the starts that carry it.
"""
import json
import math
import os
import time

import numpy as np

ENV = "PYANTIGEN_MULTISTART"
BETTER_NATS = 0.05          # a winner this much below start 0 is a better optimum, not optimizer scatter


def requested(env=None):
    """True when ``PYANTIGEN_MULTISTART`` names a task (or ``collect``): the run is a multi-start stage, whatever the fit cache holds."""
    return bool(str((os.environ if env is None else env).get(ENV) or "").strip())


def task_from_env(n_starts, env=None):
    """``None`` (not a multi-start run), ``"collect"``, or the start index named in ``PYANTIGEN_MULTISTART``."""
    raw = (os.environ if env is None else env).get(ENV)
    if raw is None or str(raw).strip() == "":
        return None
    raw = str(raw).strip().lower()
    if raw == "collect":
        return raw
    i = int(raw)
    if not 0 <= i < int(n_starts):
        raise ValueError(f"{ENV}={i}: the spec has {n_starts} start(s), so tasks are 0..{int(n_starts) - 1}")
    return i


def start_dir(fit_cache):
    """Directory of this fit problem's starts, beside the fit cache, or an error if there is no cache to name it."""
    if fit_cache is None or not fit_cache.dir:
        raise RuntimeError(f"{ENV} needs the fit cache (a plot_path and a fit that reuses its results) to name the "
                           "directory its starts are kept in")
    return os.path.join(os.path.dirname(fit_cache.dir), "multistart", os.path.splitext(fit_cache._name)[0])


def _path(directory, i):
    return os.path.join(directory, f"start_{i}.json")


def _read(directory, i):
    try:
        with open(_path(directory, i)) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def _write(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh)
    os.replace(tmp, path)


FLOORS_FILE = "floors.pkl"


def floors_stamp(directory):
    """Stamp of the floors file the starts share, or None when no task has written it."""
    from pyantigen.engine import Floors
    path = os.path.join(directory, FLOORS_FILE)
    return Floors.file_sha(path) if os.path.exists(path) else None


def use_floors(directory, x0, provider, scales, log=print):
    """Score this process under the floors every start shares; returns their stamp.

    The file is written once, by whichever task gets there first, from a calibration at *x0* (start 0, the spec's own x0), the
    point the floors would have in a single fit. Any later task, and a relaunch, loads the file and evaluates nothing here. When
    two tasks race, ``os.link`` lets one file win and both load that one.
    """
    from pyantigen.engine import Floors
    from pyantigen.engine.Profile_lm import amici_evaluate

    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, FLOORS_FILE)
    if not os.path.exists(path):
        log(f"[multistart] no shared floors yet: calibrating them at start 0 (one evaluation) -> {path}")

        def calibrate_here():
            if amici_evaluate(provider, scales)(np.asarray(x0, float)) is None:
                raise RuntimeError("the objective failed at start 0 (the spec's x0), so the noise floors cannot be calibrated; "
                                   "every start would be scored under a different function. Fix x0 first.")

        snapshot = Floors.calibrate(calibrate_here)
        tmp = f"{path}.{os.getpid()}.tmp"
        Floors.save(snapshot, tmp)
        try:
            os.link(tmp, path)                  # fails if another task got there first; theirs then wins
        except FileExistsError:
            pass
        finally:
            os.remove(tmp)
    _snapshot, stamp = Floors.load(path)           # also seeds this process's cache with the shared floors
    log(f"[multistart] floors {stamp} ({path})")
    return stamp


def run_task(i, points, provider, bounds, scales, directory, options=None, deadline=None, log=print):
    """Fit from start *i* (``points[i]``, optimizer space) and write ``start_<i>.json``; returns the record.

    A record written under other floors (or none: a run from before the floors were shared) is not a result of this
    objective. Such a start is fitted again from its own start point.
    """
    from pyantigen.engine.Fit_lm import lm_fit
    from pyantigen.engine.Optimize import _to_linear

    prev = _read(directory, i)
    shared = floors_stamp(directory)
    if prev is not None and prev.get("converged") and shared is not None and prev.get("floors") == shared:
        log(f"[multistart] start {i} already converged (f {prev['f']:.6f}); nothing to run")
        return prev
    stamp = use_floors(directory, points[0], provider, scales, log=log)
    if prev is not None and prev.get("floors") != stamp:
        log(f"[multistart] start {i}: the saved fit (f {prev['f']:.6f}) was scored under other floors "
            f"({prev.get('floors')}); fitting again from its start point")
        prev = None
    x_start = np.asarray(prev["x"] if prev else points[i], float)
    if prev:
        log(f"[multistart] start {i}: continuing an unconverged fit from f {prev['f']:.6f}")
    res = lm_fit(provider, x_start, bounds, scales, options=options, deadline=deadline, log=log)
    rec = {"start": int(i), "x_start": [float(v) for v in points[i]], "x": [float(v) for v in res.x],
           "x_lin": [float(v) for v in _to_linear(np.asarray(res.x, float), scales)], "f": float(res.fun),
           "converged": bool(res.success), "verdict": res.verdict, "pg": None if res.pg is None else float(res.pg),
           "evals": int(res.nfev) + int(prev["evals"] if prev else 0), "floors": stamp,
           "saved": time.strftime("%Y-%m-%d %H:%M:%S")}
    _write(_path(directory, i), rec)
    log(f"[multistart] start {i}: {res.verdict}, f {res.fun:.6f}")
    return rec


def collect(directory, param_names, n_starts, cluster_radius=0.15, report_dnll=1.0, log=print):
    """Read every start, report, and write ``best.json`` and ``multistart_report.json``. Returns the report dict.

    Only converged starts compete; an unconverged or missing one is listed in ``unfinished.txt`` for relaunch. The report
    says whether the converged ends are one basin, a degenerate valley, or separate basins (``Multistart.basins``), and
    whether the best of them beats start 0, the spec's x0, by more than optimizer scatter.
    """
    from pyantigen.engine.Multistart import basins

    recs = [_read(directory, i) for i in range(int(n_starts))]
    shared = floors_stamp(directory)
    stale = [i for i, r in enumerate(recs) if r is not None and r.get("floors") != shared]
    if stale:
        log(f"[multistart] start(s) {stale} were scored under other noise floors than the shared {shared}: their f is a value of "
            "another objective and is not ranked. Relaunch them.")
    recs = [None if i in stale else r for i, r in enumerate(recs)]
    todo = [i for i, r in enumerate(recs) if r is None or not r.get("converged")]
    from pyantigen.engine import Unfinished
    Unfinished.write(directory, todo)
    done = [r for r in recs if r is not None and r.get("converged") and math.isfinite(r["f"])]
    if todo:
        log(f"[multistart] {len(todo)} of {n_starts} start(s) not converged or missing; relaunch with "
            f"--array={','.join(map(str, todo))}")
    if not done:
        log("[multistart] no start has converged yet; nothing to report")
        return {"n_starts": int(n_starts), "unfinished": todo, "best": None}

    log(f"[multistart] {'start':>5} {'f':>14} {'dist from start 0':>18}")
    ref = np.asarray(done[0]["x"], float) if done[0]["start"] == 0 else None
    for r in sorted(done, key=lambda r: r["f"]):
        dist = "" if ref is None else f"{float(np.max(np.abs(np.asarray(r['x']) - ref))):.3g}"
        log(f"[multistart] {r['start']:>5} {r['f']:>14.6f} {dist:>18}")
    groups, verdict, text = basins([r["x_lin"] for r in done], [r["f"] for r in done], cluster_radius, report_dnll)
    log(f"[multistart] {text}")

    best = min(done, key=lambda r: r["f"])
    zero = next((r for r in done if r["start"] == 0), None)
    gain = None if zero is None else zero["f"] - best["f"]
    better = best["start"] != 0 and gain is not None and gain > BETTER_NATS
    if zero is None:
        log("[multistart] start 0 (the spec's x0) has not converged, so the winner cannot be compared with it")
    elif better:
        log(f"[multistart] start {best['start']} beats start 0 by {gain:.4g}: a better optimum than the registry's. "
            f"Its parameters are in {os.path.join(directory, 'best.json')}: promote them into the spec's x0 or your registry.")
    else:
        log("[multistart] start 0 stands: no start found a lower optimum.")
    fit = {"complete": {"parameters": dict(zip(param_names, best["x_lin"])), "fun": best["f"], "saved": best["saved"],
                        "success": True, "start": best["start"], "floors": shared}}
    _write(os.path.join(directory, "best.json"), fit)
    report = {"n_starts": int(n_starts), "unfinished": todo, "verdict": verdict, "best": best["start"], "f_best": best["f"],
              "gain_over_start_0": gain, "better_than_start_0": bool(better), "floors": shared,
              "starts": [{"start": r["start"], "f": r["f"]} for r in sorted(done, key=lambda r: r["f"])]}
    _write(os.path.join(directory, "multistart_report.json"), report)
    try:
        from pyantigen.engine.Multistart_plot import plot_directory
        log(f"[multistart] waterfall plot: {plot_directory(directory, n_starts=n_starts, dnll=report_dnll)}")
    except Exception as exc:       # the plot is a convenience: a missing matplotlib must not fail the collect
        log(f"[multistart] waterfall plot not drawn ({type(exc).__name__}: {exc}); "
            f"draw it later with python -m pyantigen.engine.Multistart_plot {directory}")
    return report

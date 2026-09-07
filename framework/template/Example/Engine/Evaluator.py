"""Parallel evaluation service for the proper joint NLL.

Every expensive diagnostic in this Engine -- the Wald Hessian, likelihood
slices, Sobol sampling, and profile likelihood -- is a large batch of
independent ``nll_func_fixed`` evaluations. This module turns that into one
primitive::

    with ParallelEvaluator(spec, n_workers=16) as ev:
        losses = ev.evaluate_batch([x1, x2, x3, ...])

and every consumer above it becomes embarrassingly parallel.

Design notes
------------

**Windows-first, not fork.** The previous parallel path forked, so it did
nothing on Windows. This uses a ``spawn`` context, which behaves identically on
Windows, Linux and macOS. Spawn cannot inherit memory, so each worker rebuilds
what it needs in an initializer -- which is the right shape anyway.

**Compile once per worker, not once per task.** Compiling the model costs
1.7-4.9 s while a single evaluation costs ~1 s, so a naive "one task = one
process" pool would spend all its time in ``te.loada``. Workers are persistent
and compile every needed RoadRunner exactly once at startup.

**cloudpickle for the spec, plain pickle for the tasks.** Replicate dicts hold
callables, and some ``loss_config`` entries are closures produced by factories
(e.g. ``figure5_loss_config_factory``), which ``pickle`` cannot serialize.
``cloudpickle`` can. We serialize the spec once, pass it to the initializer as
*bytes* (which pickle handles fine), and thereafter send only parameter vectors.
That keeps per-task IPC tiny.

**Failure is data.** A worker never raises across the boundary; it returns a
status and the failure sentinel, so one bad integration cannot abort a batch of
several hundred. Callers get a count of what failed rather than silence.

**No plotting in workers.** The progress overlay writes a fixed PNG/JSON path
from module-level mutable state; with N workers that becomes N processes
fighting over one file. Workers only ever compute.

Calling scripts must guard their entry point
--------------------------------------------

``spawn`` re-imports the ``__main__`` module inside every worker, so any script
that reaches this code must do its work under a guard::

    if __name__ == "__main__":
        main()

``Model_run.py`` already does. An ad-hoc analysis script that calls
``setup_optimization_from_groups`` at module level will have each worker re-run
the whole analysis, and multiprocessing raises "an attempt has been made to start
a new process before the current process has finished its bootstrapping phase".
``evaluate_batch`` rewrites that message to name the actual cause; the caller
then falls back to serial rather than losing the run.

Note that even with the guard, module-level work in the main script (imports,
building EXPERIMENT registries) is repeated in every worker at startup, so
keeping that work light pays off directly in pool start-up time.
"""

import os
import sys
import time
from dataclasses import dataclass, field

import numpy as np

try:
    import cloudpickle as _serializer
    _SERIALIZER_NAME = "cloudpickle"
except ImportError:  # pragma: no cover - cloudpickle is a declared dependency
    import pickle as _serializer
    _SERIALIZER_NAME = "pickle"


# Failure sentinel, matching Engine/Optimize.py.
FAILURE_VALUE = 1e10

# Windows caps ProcessPoolExecutor at 61 (WaitForMultipleObjects); leave headroom.
_MAX_WORKERS_WINDOWS = 60

# How often to say something while a batch is running but nothing has landed.
# A profile point is a whole nuisance minimization and can take hours, so with
# every worker busy on its first point the run is silent from the moment the
# models finish compiling until the first result -- which on a cluster is
# indistinguishable from a hang, for hours at a time.
_HEARTBEAT_SECONDS = 300


def _fmt_dur(seconds):
    """Compact duration: '3h07m', '12m40s', '45s'."""
    s = int(max(0.0, seconds))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{sec:02d}s"
    return f"{sec}s"


@dataclass
class EvalSpec:
    """Everything a worker needs to rebuild the objective from scratch.

    Must be serializable by cloudpickle. It deliberately carries the *event
    strings* rather than a way to regenerate them, so workers never re-run
    data-dependent event generation and cannot disagree with the parent.
    """
    model_text: str
    paths: dict
    events: dict                  # sim_name -> antimony event block
    replicates: dict              # sim_name -> replicate dict
    param_names: list
    scales: list
    groups: dict
    group_normalization: str
    fixed_sigmas: dict
    events_dynamic: bool = False
    data_path: str = None
    # Workers must compute exactly what the parent computes. True means the
    # joint log-likelihood (plain sum over loss elements, unit weights); False
    # reproduces the objective's own normalization and weighting.
    for_inference: bool = True
    # The concentrated likelihood is the single objective shared by the fit and
    # every diagnostic; workers must use it too or the profile would be anchored
    # on a function the parent never minimized.
    concentrated: bool = True
    # Whether workers may reuse the pre-dose block across evaluations. Decided
    # in the parent, which runs the invariance check once; a worker must never
    # make that call on its own, or 40 of them would each re-derive it.
    preequil_cache: bool = False
    # Reserved for future use by the profile grid (Stage 2).
    meta: dict = field(default_factory=dict)


# --------------------------------------------------------------------------
# Worker side
# --------------------------------------------------------------------------

_WORKER = {"spec": None, "models": None, "n_evals": 0}


def _init_worker(spec_blob):
    """Compile every model this worker needs, exactly once."""
    # Import here: with spawn, the worker re-imports the module tree anyway, and
    # keeping these out of module scope avoids paying for them in the parent.
    from framework.TelluriumGen import TelluriumGen
    from Engine.Event_times import attach_event_times
    from Engine.Optimize import OptRoadRunnerProxy

    spec = _serializer.loads(spec_blob)
    models = {}
    t0 = time.time()
    for sim_name, replicate in spec.replicates.items():
        df_dict = replicate["Data"](replicate, spec.data_path or spec.paths["data_path"])
        events_str = spec.events.get(sim_name, "")

        r_ic = None
        if spec.events_dynamic:
            r_ic = TelluriumGen(spec.model_text, spec.paths)
            replicate["Update_parameters"](
                OptRoadRunnerProxy(r_ic, spec.param_names), replicate
            )

        r = TelluriumGen(spec.model_text + "\n" + events_str, spec.paths)
        replicate["Update_parameters"](
            OptRoadRunnerProxy(r, spec.param_names), replicate
        )
        # The parent's attachment closes over the parent's RoadRunner and could
        # not be shipped here, so it was stripped from the spec. A worker must
        # attach against the model it will integrate anyway -- reusing the
        # parent's times would be reading one model's schedule off another's.
        # Silent: forty workers each printing the same summary is noise.
        attach_event_times(replicate, r)
        entry = {"r": r, "r_ic": r_ic, "df_dict": df_dict}
        # The parent already verified that the pre-dose block does not depend on
        # the fitted parameters; workers only act on that verdict.
        if getattr(spec, "preequil_cache", False) and not spec.events_dynamic:
            from Engine.Preequil_cache import PreequilCache
            entry["preequil_cache"] = PreequilCache(enabled=True)
        models[sim_name] = entry

    _WORKER["spec"] = spec
    _WORKER["models"] = models
    _WORKER["n_evals"] = 0
    print(f"  [worker {os.getpid()}] compiled {len(models)} model(s) in "
          f"{time.time() - t0:.1f}s", flush=True)


def _worker_nll(x):
    """The joint NLL, evaluated with this worker's own compiled models."""
    from Engine.Optimize import evaluate_nll_fixed

    spec = _WORKER["spec"]
    return evaluate_nll_fixed(
        np.asarray(x, dtype=float),
        _WORKER["models"], spec.replicates, spec.param_names, spec.scales,
        spec.groups, spec.group_normalization, spec.fixed_sigmas,
        model_text=spec.model_text, paths=spec.paths,
        events_dynamic=spec.events_dynamic, failure_value=FAILURE_VALUE,
        for_inference=getattr(spec, "for_inference", True),
        concentrated=getattr(spec, "concentrated", True),
    )


def _eval_task(x):
    """Evaluate one parameter vector. Never raises across the pool boundary."""
    if _WORKER["spec"] is None:
        return (FAILURE_VALUE, "worker-not-initialized", 0.0)

    t0 = time.time()
    try:
        val = _worker_nll(x)
        _WORKER["n_evals"] += 1
        status = "ok" if np.isfinite(val) and val < FAILURE_VALUE else "sentinel"
        return (float(val), status, time.time() - t0)
    except Exception as exc:
        return (FAILURE_VALUE, f"error: {type(exc).__name__}: {exc}", time.time() - t0)


class _TimeUp(Exception):
    """Raised inside the objective when a point's wall slice has run out."""


# How much wall clock one inner minimization slice aims to cover. Every slice
# boundary is a chance to capture the optimizer's state, so this sets how much
# progress a hard kill can cost; the cost of more slices is near zero because
# the vertices they re-evaluate come from the cache below.
_SLICE_TARGET_S = 600.0

# Evaluations in the opening slice when nothing is known about their cost yet.
# Enough to measure a rate, few enough to be cheap if each one is slow.
_SLICE_PROBE_EVALS = 24

# Fraction of the remaining time a slice is allowed to plan for. The margin
# absorbs variance in per-evaluation cost, which on an ODE model is
# substantial: a stiff parameter vector can take several times the median. A
# slice that overruns is stopped by the backstop and hands back no simplex, so
# finishing early is worth much more than the evaluations it gives up.
_SLICE_SAFETY = 0.8

# Objective values kept for reuse. A slice boundary makes scipy re-evaluate
# every simplex vertex, which is exactly what was computed just before, so a
# handful of entries turns the restart cost into nothing.
_EVAL_CACHE_SIZE = 128


def _simplex_including(sim, fsim, x, f):
    """The simplex with *x* substituted for its worst vertex, if *x* beats it.

    Used when a slice is killed by the clock rather than returning normally: the
    last complete simplex is the optimizer state worth keeping, but the best
    point found during the killed slice would otherwise be thrown away. Swapping
    it in for the worst vertex keeps both, and yields a simplex that is still
    a valid starting shape.
    """
    if sim is None or fsim is None or x is None or not np.isfinite(f):
        return sim
    sim = np.asarray(sim, dtype=float).copy()
    fsim = np.asarray(fsim, dtype=float)
    if sim.shape[0] != fsim.shape[0]:
        return sim
    worst = int(np.argmax(fsim))
    if f < fsim[worst]:
        sim[worst] = np.asarray(x, dtype=float)
    return sim


def _minimize_in_slices(objective, x0, args, method, bounds, optimizer_kwargs,
                        fev_left, it_left, simplex, deadline, sec_per_eval,
                        eval_count):
    """Minimize in slices so the optimizer's state is never out of reach.

    The problem this solves: a profile point can want far more wall clock than
    the queue will give it, so it has to be stoppable and resumable. Stopping is
    easy -- raise out of the objective. Resuming is the hard half, because for
    Nelder-Mead the optimizer's entire state is its simplex, and an exception
    thrown through ``scipy.optimize.minimize`` takes that simplex with it. A
    resume from the best point alone re-derives the simplex from scratch and
    spends most of its next slice relearning what it already knew: measured on a
    4-parameter quadratic, thirty further evaluations from the bare best point
    moved the objective from 6.71 to 6.63, while the same thirty with the
    simplex restored reached 4.99.

    So the minimization is run as a sequence of bounded slices. Each one returns
    normally, which means each one hands back ``final_simplex``, so there is
    always a complete and current state to write down. Slices are sized from the
    measured cost of an evaluation to land near ``_SLICE_TARGET_S``, and the
    last one is sized to finish just before the deadline.

    Slicing is close to free because of the evaluation cache: the vertices scipy
    re-evaluates when handed an ``initial_simplex`` are precisely the points the
    previous slice just computed.

    Returns ``(res, simplex, outcome)`` where *outcome* is one of ``"done"``
    (the optimizer stopped on its own terms), ``"capped"`` (the point's total
    allowance across all launches is spent) or ``"interrupted"`` (the clock).
    """
    from Engine.Optimize import _minimize_nuisance

    supports_simplex = str(method).lower() == "nelder-mead"
    n = max(1, len(np.atleast_1d(x0)))
    res = None
    x_cur = np.asarray(x0, dtype=float)

    # Below n+1 evaluations scipy cannot even establish a simplex, so a slice
    # that short returns no state at all and the point resumes cold. Above it,
    # a slice returns a usable simplex whether or not it also made progress.
    state_floor = n + 1
    useful = 2 * (n + 1)

    # One measured evaluation before committing to a slice length. Without a
    # rate the first slice has to be guessed at, and guessing high on a slow
    # objective is how a point ends up killed by the backstop with nothing
    # saved -- which is exactly what happened on the first cluster run: five
    # evaluations of a 116 s objective inside an eight-minute slice, when
    # sixteen were needed before any state existed. The probe is close to free
    # because its result is cached, so the slice that follows re-uses it
    # instead of recomputing it.
    if (deadline is not None and not sec_per_eval and eval_count["n"] == 0
            and time.time() < deadline):
        try:
            objective(x_cur, *args)
        except _TimeUp:
            return None, simplex, "interrupted"

    while True:
        if fev_left <= 0 or it_left <= 0:
            return res, simplex, "capped"

        slice_fev = fev_left
        if deadline is not None:
            t_left = deadline - time.time()
            if t_left <= 0:
                return res, simplex, "interrupted"
            rate = sec_per_eval or (eval_count["seconds"] / eval_count["n"]
                                    if eval_count["n"] else None)
            if rate and rate > 0:
                # Deliberately short of what the clock allows. A slice sized to
                # consume every remaining second finishes only if the rate
                # estimate is perfect; any variance trips the backstop, and the
                # backstop is the one exit that returns no simplex. Aiming to
                # land early is what makes the state reliably saveable.
                affordable = int(_SLICE_SAFETY * t_left / rate)
                if affordable < state_floor:
                    if res is not None:
                        # Not even enough left to re-establish a simplex. Stop
                        # while the state from the last slice is intact rather
                        # than spending the remainder and losing it.
                        return res, simplex, "interrupted"
                    # No slice here can reach a state worth saving: scipy needs
                    # n+1 evaluations before a simplex exists, and there is not
                    # time for them. Attempting it anyway spends the whole
                    # slice and hands back nothing.
                    #
                    # That is not hypothetical. One SILK link ran with the
                    # 30-minute default cap and no timing history, against a
                    # 15-nuisance-parameter model at 113 s per evaluation: a
                    # simplex costs 30.2 minutes, so every point burned its
                    # entire slice on exactly n+1 evaluations, took zero
                    # Nelder-Mead iterations, and came back with its nuisance
                    # vector still equal to its starting point and no simplex.
                    # Across 39 workers that is hours of wall clock for
                    # nothing.
                    #
                    # Stopping now costs one evaluation instead of sixteen, and
                    # -- the point of it -- that evaluation measures the rate,
                    # which is written to timing.json and lets the next round
                    # size its cap correctly.
                    if eval_count["n"] == 0:
                        try:
                            objective(x_cur, *args)
                        except _TimeUp:
                            pass
                    return None, simplex, "interrupted"
                else:
                    # Ask for what actually fits, not for what would be ideal.
                    # Requesting more than the clock allows guarantees the
                    # backstop fires, and the backstop is the one exit that
                    # returns no simplex.
                    slice_fev = min(fev_left, affordable,
                                    max(useful, int(_SLICE_TARGET_S / rate)))
                    slice_fev = max(slice_fev, min(fev_left, state_floor))
            else:
                slice_fev = min(slice_fev, max(state_floor, _SLICE_PROBE_EVALS))

        extra = {"maxfev": int(slice_fev),
                 "maxiter": int(min(it_left, slice_fev))}
        if simplex is not None and supports_simplex:
            extra["initial_simplex"] = np.asarray(simplex, dtype=float)

        before = eval_count["n"]
        try:
            res = _minimize_nuisance(objective, x_cur, args, method, bounds,
                                     optimizer_kwargs, extra_options=extra)
        except _TimeUp:
            # The rate estimate was too optimistic -- one evaluation took far
            # longer than the others. The previous slice's simplex is still the
            # best state available, so keep it rather than losing everything.
            return res, simplex, "interrupted"

        final = getattr(res, "final_simplex", None)
        if final is not None:
            simplex = np.asarray(final[0], dtype=float)
        x_cur = np.asarray(res.x, dtype=float)

        # The point's allowance is spent in *evaluations of the model*, so it
        # is charged what actually ran. scipy's own nfev counts the vertex
        # re-evaluations at each slice boundary, which the cache serves for
        # free; charging those would make a heavily-sliced point exhaust its
        # budget without doing the work the budget was meant to buy.
        fev_left -= max(0, eval_count["n"] - before)
        it_left -= max(1, int(getattr(res, "nit", 0) or 0))

        # Stopping short of the slice cap means the optimizer stopped for its
        # own reasons -- converged, or on xatol/fatol -- and there is nothing
        # more to do. Filling the slice means it was cut off by us, not by the
        # problem, so there is more to do if there is time to do it.
        if int(getattr(res, "nfev", 0) or 0) < slice_fev:
            return res, simplex, "done"


def _profile_task(job):
    """Run one profile-likelihood point: minimize over the nuisance parameters
    with parameter ``param_idx`` pinned at ``x_fixed``.

    A profile point is a whole optimization, not a single evaluation, so the
    scipy call runs *inside* the worker against its local models. That is what
    makes the profile parallel: 2k x n_grid independent optimizations in flight,
    instead of one adaptive walk stepping sequentially.

    **A point does not have to fit in one job.** On a four-hour queue a single
    nuisance minimization can easily want forty, so the point carries a
    ``deadline`` and stops itself when it arrives, reporting where it had got
    to. That is sound rather than merely convenient: every evaluation of this
    objective is an upper bound on the profile, so a half-finished point is a
    real point that happens to sit too high, and the store keeps the lowest
    value seen at each fixed value. Resuming can therefore only lower the
    curve, never raise it -- the same invariant the warm-continuation pass
    already relies on.

    Two things are carried across the interruption, and the second is what
    makes it worth doing:

    * ``nuisance_x`` -- the best nuisance vector reached so far, which becomes
      the next job's starting point.
    * ``nm_simplex`` -- for Nelder-Mead, the whole simplex. Without it a resume
      restarts the simplex from a single point and spends most of its next
      slice rebuilding what it already knew; measured on a 4-parameter
      quadratic, 30 further evaluations from the bare best point recovered
      almost nothing while the same 30 with the simplex restored made normal
      progress. The optimizer's state *is* the simplex, so saving it is the
      difference between resuming and starting over.

    ``job`` is a plain dict so it pickles cheaply. Returns a result dict that is
    written straight to the checkpoint file.
    """
    from Engine.Optimize import (
        _make_nuisance_objective, _minimize_nuisance, nuisance_convergence,
        nuisance_option_budget,
    )

    t0 = time.time()
    out = dict(job)
    # An input, not a result, and a bulky one: a 15-nuisance simplex is 240
    # floats, which every record would otherwise carry into the checkpoint
    # alongside the nm_simplex it actually needs to store. The resume path
    # reads nm_simplex, never this.
    out.pop("initial_simplex", None)
    out.update({"nll": None, "status": "ok", "n_evals": 0, "wall_s": 0.0,
                "worker": os.getpid(), "converged": True, "nit": -1,
                "nfev": -1, "opt_message": "", "interrupted": False})

    if _WORKER["spec"] is None:
        out.update({"status": "worker-not-initialized", "nll": FAILURE_VALUE})
        return out

    try:
        spec = _WORKER["spec"]
        n_params = len(spec.param_names)
        param_idx = int(job["param_idx"])
        x_fixed = float(job["x_fixed"])
        x_start = np.asarray(job["x_start"], dtype=float)
        method = job.get("method", "Nelder-Mead")
        deadline = job.get("deadline")

        # What earlier jobs on this same point already spent. The caps are a
        # total across every launch, so a point that keeps being interrupted
        # still terminates instead of being resumed forever.
        nfev_used = int(job.get("nfev_used") or 0)
        nit_used = int(job.get("nit_used") or 0)

        # Real evaluations and the time they cost, which is what sizes the next
        # slice. Cache hits are excluded from both: they are neither work done
        # nor budget spent.
        eval_count = {"n": 0, "seconds": 0.0}
        # The best point seen. Tracked here rather than read off an
        # OptimizeResult because when the clock stops a slice there is no
        # OptimizeResult to read it from.
        best = {"f": float("inf"), "x": x_start}
        cache = {}
        cache_order = []

        raw_objective = _make_nuisance_objective(_worker_nll, param_idx, n_params)

        def nuisance_objective(x_nuisance, fixed_val):
            x_arr = np.asarray(x_nuisance, dtype=float)
            key = x_arr.tobytes()
            hit = cache.get(key)
            if hit is not None:
                return hit

            t_eval = time.time()
            v = raw_objective(x_arr, fixed_val)
            eval_count["n"] += 1
            eval_count["seconds"] += time.time() - t_eval

            cache[key] = v
            cache_order.append(key)
            if len(cache_order) > _EVAL_CACHE_SIZE:
                cache.pop(cache_order.pop(0), None)

            if np.isfinite(v) and v < best["f"]:
                best["f"] = float(v)
                best["x"] = x_arr.copy()
            # Checked after recording, so the value just computed is never lost
            # to the interruption that follows it.
            if deadline is not None and time.time() >= deadline:
                raise _TimeUp()
            return v

        bounds = job.get("nuisance_bounds")
        if bounds is not None:
            bounds = [tuple(b) if b is not None else None for b in bounds]

        if x_start.size == 0:
            # Single-parameter fit: nothing to re-optimize, so the profile value
            # is just the objective at the fixed value -- exact by definition.
            nll = raw_objective(x_start, x_fixed)
            x_opt = x_start
            out.update({"converged": True, "nit": 0, "nfev": 1})
            eval_count["n"] = 1
        else:
            caps = nuisance_option_budget(method, x_start.size,
                                          job.get("optimizer_kwargs"))
            fev_left = caps.get("maxfev", 10 ** 9) - nfev_used
            it_left = caps.get("maxiter", 10 ** 9) - nit_used

            simplex = job.get("initial_simplex")
            res, simplex, outcome = _minimize_in_slices(
                nuisance_objective, x_start, (x_fixed,), method, bounds,
                job.get("optimizer_kwargs"), fev_left, it_left, simplex,
                deadline, job.get("sec_per_eval"), eval_count,
            )

            if res is not None:
                out.update(nuisance_convergence(res))

            if outcome == "interrupted":
                # Not a failure: the slice ended. Report where the search had
                # reached and mark the point so a later launch continues it.
                nll = best["f"] if np.isfinite(best["f"]) else (
                    float(res.fun) if res is not None else FAILURE_VALUE)
                x_opt = np.asarray(best["x"], dtype=float)
                fsim = (getattr(res, "final_simplex", (None, None))[1]
                        if res is not None else None)
                simplex = _simplex_including(simplex, fsim, best["x"], best["f"])
                out.update({
                    "interrupted": True,
                    "converged": False,
                    "opt_message": "stopped on the wall clock; resumable",
                })
            elif res is None:
                # Capped before a single slice could run: the point has spent
                # its whole allowance across earlier launches. It reports the
                # value it had already reached, so the record stays usable and
                # -- crucially -- checkpointable. A sentinel here would never be
                # written, leaving the stored record marked interrupted and the
                # point resumed on every future link for no work at all.
                prior = job.get("nll_so_far")
                nll = float(prior) if prior is not None else best["f"]
                x_opt = np.asarray(x_start, dtype=float)
                out.update({
                    "converged": False, "nit": nit_used, "nfev": nfev_used,
                    "opt_message": "evaluation budget exhausted across launches",
                })
            else:
                nll = float(res.fun)
                x_opt = np.asarray(res.x, dtype=float)
                if outcome == "capped":
                    out.update({
                        "converged": False,
                        "opt_message": "evaluation budget exhausted across launches",
                    })

            if simplex is not None:
                out["nm_simplex"] = np.asarray(simplex, dtype=float).tolist()

        out.update({
            "nll": float(nll),
            "nuisance_x": np.asarray(x_opt, dtype=float).tolist(),
            "n_evals": eval_count["n"],
            # Totals across every launch this point has had, so the next one
            # knows how much of the allowance is left and the point terminates
            # instead of being resumed forever.
            "nfev_total": nfev_used + eval_count["n"],
            "nit_total": nit_used + max(0, int(out.get("nit") or 0)),
            "status": "ok" if np.isfinite(nll) and nll < FAILURE_VALUE else "sentinel",
        })
    except Exception as exc:
        out.update({"status": f"error: {type(exc).__name__}: {exc}",
                    "nll": FAILURE_VALUE, "interrupted": False})

    out["wall_s"] = time.time() - t0
    return out


# --------------------------------------------------------------------------
# Parent side
# --------------------------------------------------------------------------

def available_cpus():
    """CPUs this process may actually use -- not what the machine has.

    On a scheduler-managed node ``os.cpu_count()`` reports the whole machine.
    A job allocated 8 cores of a 40-core node would then start 39 workers
    inside an 8-core cgroup: roughly a fivefold slowdown from oversubscription,
    while taking cores from whoever else is sharing the node.

    ``SLURM_CPUS_PER_TASK`` is what the allocation asked for;
    ``sched_getaffinity`` is the cgroup or taskset the kernel will actually
    honour. ``os.cpu_count()`` is the last resort, and is right on a laptop.
    """
    for var in ("SLURM_CPUS_PER_TASK", "SLURM_JOB_CPUS_PER_NODE"):
        raw = os.environ.get(var)
        if not raw:
            continue
        # SLURM_JOB_CPUS_PER_NODE can read "8", "8(x2)" or "8,4".
        head = raw.split("(")[0].split(",")[0].strip()
        try:
            n = int(head)
        except ValueError:
            continue
        if n > 0:
            return n
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:      # not Linux
        pass
    return max(1, os.cpu_count() or 2)


def available_memory_gb():
    """Memory this process may actually use, in GB, or None if unknown.

    The counterpart to :func:`available_cpus`, and needed for the same reason:
    a scheduler hands out cores and memory separately, and for this workload it
    is memory that runs out first. Each worker compiles every simulation in the
    spec, so the pool's footprint is ``n_workers x n_simulations x per-model``
    while the core count grows only in the first factor.

    ``SLURM_MEM_PER_NODE`` is what the allocation asked for, in MB.
    ``SLURM_MEM_PER_CPU`` is the same budget expressed per core and has to be
    multiplied back up. Off-cluster the machine's own total is the answer.

    None means "no idea", which callers must treat as "do not restrict" -- an
    unknown limit has to fail open, or a laptop with an unreadable meminfo
    would silently drop to a single worker.
    """
    raw = os.environ.get("SLURM_MEM_PER_NODE")
    if raw:
        try:
            return float(raw) / 1024.0
        except ValueError:
            pass

    raw = os.environ.get("SLURM_MEM_PER_CPU")
    if raw:
        try:
            return float(raw) * available_cpus() / 1024.0
        except ValueError:
            pass

    try:
        import psutil
        return psutil.virtual_memory().total / 1e9
    except Exception:
        pass

    try:
        return (os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")) / 1e9
    except (AttributeError, ValueError, OSError):
        return None


def default_worker_count(n_workers=None):
    """Workers to use: explicit value, else all available cores but one.

    Cores only. The memory ceiling is applied by :class:`ParallelEvaluator`,
    which is the first place that knows how large the models are.
    """
    if n_workers is None:
        n_workers = max(1, available_cpus() - 1)
    n_workers = max(1, int(n_workers))
    if sys.platform == "win32":
        n_workers = min(n_workers, _MAX_WORKERS_WINDOWS)
    return n_workers


class ParallelEvaluator:
    """A pool of persistent workers evaluating the joint NLL.

    Use as a context manager so the pool is always shut down::

        with ParallelEvaluator(spec, n_workers=16) as ev:
            losses = ev.evaluate_batch(xs)

    ``evaluate_batch`` preserves input order. Failures come back as
    ``FAILURE_VALUE`` and are counted in ``ev.n_failures`` rather than raised.
    """

    def __init__(self, spec, n_workers=None, chunk_size=None, verbose=True,
                 memory_limit_gb=None):
        self.spec = spec
        self.chunk_size = chunk_size
        self.verbose = verbose
        self._pool = None
        self._blob = None
        self.n_evals = 0
        self.n_failures = 0
        self.total_worker_seconds = 0.0

        self.n_workers_requested = default_worker_count(n_workers)
        self.n_workers = self._fit_to_memory(self.n_workers_requested,
                                             memory_limit_gb)

    # -- lifecycle ---------------------------------------------------------

    # Calibration point: the ~370-species / 978-reaction SILK variant, whose
    # antimony source is ~124k characters, costs ~0.35 GB per compiled model.
    _REF_MODEL_CHARS = 124_000
    _REF_MODEL_GB = 0.35

    # Fraction of the allocation the pool is allowed to plan for. The estimate
    # below is a proxy rather than a measurement, and the parent process, the
    # data and the plotting all want memory the workers are not accounted for,
    # so the pool aims well short of the limit.
    _MEM_HEADROOM = 0.8

    def per_worker_memory_gb(self):
        """Rough resident cost of one worker, in GB.

        Per-model cost is scaled from the antimony source length against a
        measured reference. That is a crude proxy -- it tracks model size, not
        RoadRunner's exact allocation -- so treat it as an order of magnitude.
        """
        chars = max(len(self.spec.model_text or ""), 1)
        per_model_gb = self._REF_MODEL_GB * (chars / self._REF_MODEL_CHARS)
        overhead_gb = 0.15
        n_models = max(len(self.spec.replicates), 1)
        return n_models * per_model_gb + overhead_gb

    def memory_estimate_gb(self):
        """Rough resident-memory estimate for the whole pool, in GB.

        Each worker compiles every simulation in the spec, so the footprint
        scales as ``n_workers x n_simulations x per-model``. That product, not
        the core count, is what limits how wide this can run: a 12-simulation
        spec at 40 workers wants well over 100 GB.
        """
        return self.n_workers * self.per_worker_memory_gb()

    def _fit_to_memory(self, n_workers, memory_limit_gb=None):
        """Lower *n_workers* until the pool is expected to fit in memory.

        This used to be a warning, and a warning was the wrong response: a pool
        that overcommits memory is not slower, it is OOM-killed partway through
        a batch, and on a preemptible queue that looks exactly like the
        eviction it is not. Being scheduled 40 cores does not mean 40 workers
        fit -- for a spec with a dozen simulations it usually means the
        opposite -- so cores propose and memory disposes.

        The cap applies even to an explicitly requested worker count, because
        the failure it prevents is not a matter of taste. ``PROFILE_MEM_LIMIT_GB``
        overrides the detected limit, and setting it to 0 disables the cap for
        anyone who knows better than the estimate.
        """
        limit_gb = memory_limit_gb
        if limit_gb is None:
            raw = os.environ.get("PROFILE_MEM_LIMIT_GB")
            if raw:
                try:
                    limit_gb = float(raw)
                except ValueError:
                    limit_gb = None
        if limit_gb is None:
            limit_gb = available_memory_gb()

        if not limit_gb or limit_gb <= 0:
            return n_workers

        per_worker = self.per_worker_memory_gb()
        if per_worker <= 0:
            return n_workers

        affordable = int((limit_gb * self._MEM_HEADROOM) // per_worker)
        # At least one worker regardless: a spec too large for even a single
        # worker is a real problem, but refusing to start is not how to report
        # it -- let it run and fail with RoadRunner's own message.
        affordable = max(1, affordable)
        if affordable >= n_workers:
            return n_workers

        if self.verbose:
            print(f"[pool] {n_workers} core(s) available but only {affordable} "
                  f"worker(s) fit in {limit_gb:.0f} GB at ~{per_worker:.1f} GB "
                  f"each ({len(self.spec.replicates)} model(s) per worker); "
                  f"memory is the binding constraint.", flush=True)
        return affordable

    def start(self):
        if self._pool is not None:
            return self
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor

        self._blob = _serializer.dumps(self.spec)
        n_models = len(self.spec.replicates)
        est_gb = self.memory_estimate_gb()
        if self.verbose:
            capped = (f", capped from {self.n_workers_requested}"
                      if self.n_workers < self.n_workers_requested else "")
            print(f"[pool] starting {self.n_workers} worker(s){capped} "
                  f"({_SERIALIZER_NAME} spec: {len(self._blob) / 1e6:.1f} MB, "
                  f"{n_models} model(s) each, ~{est_gb:.0f} GB estimated)",
                  flush=True)
        self._pool = ProcessPoolExecutor(
            max_workers=self.n_workers,
            mp_context=mp.get_context("spawn"),
            initializer=_init_worker,
            initargs=(self._blob,),
        )
        return self

    def shutdown(self):
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc, tb):
        self.shutdown()
        return False

    # -- evaluation --------------------------------------------------------

    def evaluate_batch(self, xs, label=None):
        """Evaluate every parameter vector in *xs*; return losses in input order."""
        xs = [np.asarray(x, dtype=float) for x in xs]
        if not xs:
            return []
        if self._pool is None:
            self.start()

        chunk = self.chunk_size
        if chunk is None:
            # Enough chunks to keep every worker fed, few enough to avoid
            # per-task overhead dominating.
            chunk = max(1, len(xs) // (self.n_workers * 4) or 1)

        t0 = time.time()
        if self.verbose:
            # map() returns nothing until the whole batch is done, so this line
            # is the only warning the caller gets that the next stretch of
            # silence is expected. Individual points are reported by
            # profile_batch; this path deliberately trades that for chunking.
            tag = f" [{label}]" if label else ""
            print(f"[pool]{tag} {len(xs)} evaluation(s) submitted to "
                  f"{self.n_workers} worker(s) in chunks of {chunk}; "
                  f"no output until the batch completes.", flush=True)
        try:
            out = list(self._pool.map(_eval_task, xs, chunksize=chunk))
        except RuntimeError as exc:
            if "bootstrapping phase" in str(exc):
                # spawn re-imports the __main__ module in every worker. If the
                # caller's script runs its work at module level, each worker
                # re-runs the whole analysis and multiprocessing refuses. The
                # stock message never mentions the caller's script, so say it.
                raise RuntimeError(
                    "The parallel evaluator needs the calling script to guard "
                    "its entry point:\n\n"
                    "    if __name__ == '__main__':\n"
                    "        main()\n\n"
                    "Worker processes are started with 'spawn', which re-imports "
                    "the main module; without the guard each worker would re-run "
                    "your analysis from the top. Model_run.py already does this — "
                    "ad-hoc analysis scripts need it too. "
                    f"(original error: {exc})"
                ) from exc
            raise

        losses = []
        failures = []
        for i, (val, status, secs) in enumerate(out):
            losses.append(val)
            self.total_worker_seconds += secs
            if status != "ok":
                failures.append((i, status))

        self.n_evals += len(xs)
        self.n_failures += len(failures)

        if self.verbose:
            elapsed = max(time.time() - t0, 1e-9)
            # Worker-seconds is the serial cost of the same work; the ratio to
            # wall time is the speedup actually realized. On the first batch it
            # includes worker startup, so it understates steady-state throughput
            # -- report both numbers rather than one flattering one.
            work = sum(o[2] for o in out)
            tag = f" [{label}]" if label else ""
            print(f"[pool]{tag} {len(xs)} evals in {elapsed:.1f}s wall "
                  f"({work:.1f}s of work, {work / elapsed:.1f}x, "
                  f"{len(xs) / elapsed:.1f} eval/s)", flush=True)
            if failures:
                shown = "; ".join(f"#{i}: {s}" for i, s in failures[:3])
                more = f" (+{len(failures) - 3} more)" if len(failures) > 3 else ""
                print(f"[pool]{tag} {len(failures)} failed — {shown}{more}", flush=True)

        return losses

    def profile_batch(self, jobs, on_result=None, label=None,
                      heartbeat_s=_HEARTBEAT_SECONDS, budget=None):
        """Run profile-likelihood points in parallel, within a wall budget.

        Unlike ``evaluate_batch`` this uses submit/wait rather than map, because
        each job is minutes to hours long and results must be checkpointed *as
        they land* -- the whole point of checkpointing is that killing the run
        halfway keeps the half that finished.

        Points are admitted a poolful at a time rather than submitted all at
        once. Two things follow from that, and the second is the reason for it:

        * Only ``n_workers`` points are ever committed, so an eviction that
          arrives anyway destroys at most that many, not the whole batch.
        * A point is admitted against the clock as it is about to *start*, not
          when the batch was assembled. Submitting the whole list up front and
          checking the deadline there would clear a point that will not begin
          for another three hours, which is precisely the case the check exists
          to catch.

        When *budget* runs out the already-running points are waited for --
        they were admitted in good faith and are usually the expensive ones --
        and then :class:`~Engine.Deadline.DeadlineReached` is raised naming how
        many never started. Everything that landed has already been through
        *on_result*, so nothing computed is lost by the raise.

        A heartbeat is printed every ``heartbeat_s`` while nothing is landing.
        Waiting on completions alone means that with as many workers as points
        in flight, the run says nothing between the models compiling and the
        first result -- hours of silence that reads exactly like a hang.

        *on_result* is called with each result dict the moment it arrives.
        Returns results in completion order; callers key off the job fields.
        """
        if not jobs:
            return []
        if self._pool is None:
            self.start()
        from concurrent.futures import wait, FIRST_COMPLETED
        from Engine.Deadline import DeadlineReached

        t0 = time.time()
        results = []
        backlog = list(jobs)
        futures = {}
        pending = set()
        n_jobs = len(jobs)
        done = 0
        halted = False
        tag = f" [{label}]" if label else ""

        sec_per_eval = budget.seconds_per_eval() if budget is not None else None

        # What one slice has to be allowed, so that it ends with optimizer
        # state worth resuming from rather than being cut off before any
        # exists. Nelder-Mead needs n+1 evaluations before it has a simplex at
        # all, and twice that before it has also made progress.
        n_nuisance = max(1, len(jobs[0].get("x_start") or ()))
        state_slice_s = (2 * (n_nuisance + 1) * sec_per_eval
                         if sec_per_eval else None)
        if (self.verbose and budget is not None and budget.is_limited
                and state_slice_s):
            room = budget.remaining() - budget.margin_s
            if state_slice_s > room:
                print(f"[pool]{tag} WARNING: at {sec_per_eval:.0f}s per "
                      f"evaluation a {n_nuisance + 1}-vertex simplex needs "
                      f"~{state_slice_s / 60.0:.0f} min, but only "
                      f"{room / 60.0:.0f} min remain. Points will be stopped "
                      f"before their optimizer state can be saved and will "
                      f"resume cold — give the link more wall clock.",
                      flush=True)

        def admit():
            """Start points until the pool is full or the clock says stop."""
            nonlocal halted
            while backlog and len(pending) < self.n_workers:
                if budget is not None and not budget.admits():
                    halted = True
                    return
                job = backlog.pop(0)
                # Stamped here rather than where the job was built, because
                # this is the only place that knows both the clock and the
                # moment the point actually starts -- and the slice cap is
                # measured from that moment, so it has to be resolved per job
                # rather than once for the batch.
                job = dict(job,
                           deadline=(budget.job_deadline(state_slice_s)
                                     if budget is not None else None),
                           sec_per_eval=sec_per_eval)
                fut = self._pool.submit(_profile_task, job)
                futures[fut] = job
                pending.add(fut)

        if self.verbose:
            print(f"[pool]{tag} {n_jobs} profile point(s) for "
                  f"{self.n_workers} worker(s); progress every "
                  f"{_fmt_dur(heartbeat_s)} until results start landing.",
                  flush=True)
            if budget is not None and budget.is_limited:
                print(f"[pool]{tag} {budget.describe(state_slice_s)}",
                      flush=True)

        admit()

        while pending:
            finished, pending = wait(pending, timeout=heartbeat_s,
                                     return_when=FIRST_COMPLETED)

            if not finished:
                if self.verbose:
                    now = time.time()
                    msg = (f"  [profile{tag}] {done}/{n_jobs} done, "
                           f"{len(pending)} in flight, "
                           f"{len(backlog)} not started, "
                           f"{_fmt_dur(now - t0)} elapsed")
                    if done:
                        rate = done / max(now - t0, 1e-9)
                        msg += f", ~{_fmt_dur((n_jobs - done) / rate)} remaining"
                    else:
                        msg += " (no point has finished yet, so no estimate)"
                    if halted:
                        msg += "; admitting no more work before the deadline"
                    print(msg, flush=True)
                continue

            for fut in finished:
                try:
                    res = fut.result()
                except Exception as exc:
                    job = futures[fut]
                    res = dict(job)
                    res.update({"nll": FAILURE_VALUE, "n_evals": 0, "wall_s": 0.0,
                                "status": f"error: {type(exc).__name__}: {exc}"})
                results.append(res)
                done += 1
                self.n_evals += int(res.get("n_evals") or 0)
                self.total_worker_seconds += float(res.get("wall_s") or 0.0)
                if res.get("status") != "ok":
                    self.n_failures += 1
                # Only a real point teaches anything about what a point costs;
                # a job that died on the way out returns wall_s 0 and would
                # drag the estimate toward zero, which is the direction that
                # admits work there is no time for.
                if budget is not None and res.get("status") == "ok":
                    budget.record(res.get("wall_s"), res.get("n_evals"))
                if on_result is not None:
                    on_result(res)
                if self.verbose:
                    elapsed = time.time() - t0
                    # An interrupted point is progress, not a problem, and the
                    # log has to say so or every link will read as a run of
                    # failures.
                    state = " CONTINUES" if res.get("interrupted") else ""
                    print(f"  [profile {done}/{n_jobs}]{state} "
                          f"{res.get('param_name')} "
                          f"= {res.get('x_fixed_linear', res.get('x_fixed')):.4g}  "
                          f"dNLL={res.get('dnll', float('nan')):.4g}  "
                          f"({res.get('n_evals')} evals, {res.get('wall_s', 0):.0f}s)"
                          f"  [{_fmt_dur(elapsed)} elapsed]", flush=True)

            admit()

        if self.verbose:
            elapsed = max(time.time() - t0, 1e-9)
            work = sum(float(r.get("wall_s") or 0.0) for r in results)
            print(f"[pool]{tag} {done} profile points in {elapsed:.0f}s wall "
                  f"({work:.0f}s of work, {work / elapsed:.1f}x)", flush=True)

        if backlog:
            if budget is not None:
                budget.stopped_early = True
                budget.save()
            raise DeadlineReached(len(backlog), label)
        return results

    def as_scalar_func(self):
        """A serial-looking ``f(x) -> float`` backed by the pool.

        Convenience for code that cannot batch. It gives no speedup on its own
        (one vector at a time) -- prefer ``evaluate_batch``.
        """
        def _f(x):
            return self.evaluate_batch([x])[0]
        return _f


def build_eval_spec(
    model_text, paths, events, replicates, param_names, scales, groups,
    group_normalization, fixed_sigmas, events_dynamic=False, data_path=None,
    for_inference=True, concentrated=True, preequil_cache=False,
):
    """Convenience constructor mirroring the spec-route local variables."""
    from Engine.Event_times import without_event_times

    return EvalSpec(
        model_text=model_text,
        paths=paths,
        events=dict(events),
        # The parent's event-time callable closes over the parent's RoadRunner.
        # cloudpickle will happily carry it -- that is the problem. Shipped, it
        # would resolve each worker's event times against the *parent's*
        # parameter values, and during a profile the worker is by definition
        # holding a different vector. With SubCut_D1 among the fitted
        # parameters, the Gantenerumab infusion-off edges would land in the
        # wrong place in every worker, silently. It also drags a duplicate
        # RoadRunner (~13 kB per arm) through the spec for nothing.
        # _init_worker re-attaches against the model the worker compiled.
        replicates=without_event_times(replicates),
        param_names=list(param_names),
        scales=list(scales),
        groups=groups,
        group_normalization=group_normalization,
        fixed_sigmas=dict(fixed_sigmas or {}),
        events_dynamic=bool(events_dynamic),
        data_path=data_path or paths.get("data_path"),
        for_inference=bool(for_inference),
        concentrated=bool(concentrated),
        preequil_cache=bool(preequil_cache),
    )


def check_spec_serializable(spec, verbose=True):
    """Round-trip the spec so serialization problems surface in the parent.

    Without this, an unpicklable ``loss_config`` closure fails inside worker
    startup, where the traceback is far less legible.
    """
    try:
        blob = _serializer.dumps(spec)
        _serializer.loads(blob)
        if verbose:
            print(f"[pool] spec serializes cleanly via {_SERIALIZER_NAME} "
                  f"({len(blob) / 1e6:.1f} MB)")
        return True, len(blob), None
    except Exception as exc:
        msg = (f"{type(exc).__name__}: {exc}. The parallel path needs every "
               f"replicate callable and loss_config to be serializable; a "
               f"closure that captures a RoadRunner or an open file will fail "
               f"here. Falling back to serial evaluation.")
        if verbose:
            print(f"[pool] spec is NOT serializable — {msg}")
        return False, 0, msg

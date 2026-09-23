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
    # Engine.Noise_floor.export_cache() snapshot, taken in the parent
    # AFTER its own calibration (see Engine.Optimize.run_optimization_from_groups,
    # clear_cache() + the post-optimum re-evaluation). Workers seed their own
    # (otherwise empty, since spawn shares no memory) floor cache from this in
    # _init_worker, so every worker scores every floored observable against
    # the SAME calibrated sigma the parent settled on, rather than each one
    # independently calibrating against whatever parameter vector it happens
    # to be handed first -- an arbitrary profile-grid point or Sobol sample,
    # not the converged optimum. Same reasoning as fixed_sigmas above, one
    # mechanism down: compute once where it's meaningful, ship the answer.
    floor_cache: dict = field(default_factory=dict)
    # When set, every simulation in this worker is one attempt capped at this many
    # CVODE steps and a failure is scored as the failure value at once, instead
    # of climbing safe_simulate's retry ladder. For a global search, which throws
    # vectors at the model that are mostly nonsense; see Engine.Simulate's
    # "Search mode". None (the default) leaves normal behaviour untouched.
    search_max_steps: int = None
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
    from Engine.Noise_floor import seed_cache

    spec = _serializer.loads(spec_blob)
    # Before any task runs: this worker's own Engine.Noise_floor
    # module was just re-imported fresh (spawn shares no memory with the
    # parent), so its floor cache starts empty. Seed it from the parent's
    # already-calibrated snapshot so every worker agrees with the parent --
    # and with each other -- on every floored observable's sigma, instead of
    # each recalibrating independently against whichever task it draws first.
    seed_cache(spec.floor_cache)
    # Set here, once per worker: the setting is process-global, like the
    # integrator state it governs, and a spawned worker starts with it off.
    from Engine.Simulate import set_search_mode
    set_search_mode(getattr(spec, "search_max_steps", None))
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


def _worker_nll(x, frozen_sigmas=None):
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
        frozen_sigmas=frozen_sigmas,
    )


def _eval_task(x, frozen_sigmas=None):
    """Evaluate one parameter vector. Never raises across the pool boundary."""
    if _WORKER["spec"] is None:
        return (FAILURE_VALUE, "worker-not-initialized", 0.0)

    t0 = time.time()
    try:
        val = _worker_nll(x, frozen_sigmas=frozen_sigmas)
        _WORKER["n_evals"] += 1
        status = "ok" if np.isfinite(val) and val < FAILURE_VALUE else "sentinel"
        return (float(val), status, time.time() - t0)
    except Exception as exc:
        return (FAILURE_VALUE, f"error: {type(exc).__name__}: {exc}", time.time() - t0)


class _Stopped(Exception):
    """Raised inside a non-Nelder-Mead objective when the point's time is up."""


def _profile_task(job):
    """Run one profile-likelihood point: minimize over the nuisance parameters
    with parameter ``param_idx`` pinned at ``x_fixed``.

    A profile point is a whole optimization, not a single evaluation, so the
    optimizer runs *inside* the worker against its local models. That is what
    makes the profile parallel: 2k x n_grid independent optimizations in flight,
    instead of one adaptive walk stepping sequentially.

    **A point survives being killed.** A point can take hours, and on a
    preemptible partition the process is stopped without notice. For
    Nelder-Mead -- the method every spec here uses -- the worker runs
    :mod:`Engine.Nelder_mead`, whose entire state is a small dict, and rewrites
    that state to ``job["state_path"]`` every few seconds. A later job for the
    same point loads it and continues: no evaluation is repeated, and the
    result is identical to an uninterrupted run. A kill loses the evaluation in
    flight and nothing more, which is why nothing on this path needs to know
    when the job will end.

    ``deadline`` is the one place the clock is consulted, and it is a plain "stop
    here": when it passes the point saves its state, returns marked
    ``interrupted`` and reports where it got to. That is sound rather than
    merely convenient: every evaluation of this objective is an upper bound on
    the profile, so a half-finished point is a real point that happens to sit too
    high, and the store keeps the lowest value seen at each fixed value.
    Resuming can therefore only lower the curve, never raise it -- the same
    invariant the warm-continuation pass already relies on.

    Two things are carried across an interruption:

    * ``nuisance_x`` -- the best nuisance vector reached so far, which becomes
      the next job's starting point.
    * ``nm_simplex`` -- for Nelder-Mead, the whole simplex, so a job started
      from a *record* rather than from the state file (a continuation round, a
      point stopped on the clock and resumed by a later launch) restarts from
      the simplex instead of one vertex. The state file is the exact resume and
      wins when it exists; the simplex is what the record path has.

    ``job`` is a plain dict so it pickles cheaply. Returns a result dict that is
    written straight to the checkpoint file.
    """
    from Engine.Nelder_mead import (
        STATUS_STOPPED, best_of, bounds_arrays, can_run, minimize_nelder_mead,
        new_state, state_from_json, state_to_json,
    )
    from Engine.Optimize import (
        _make_nuisance_objective, _minimize_nuisance, nuisance_convergence,
        nuisance_options,
    )
    from Engine.Profile_checkpoint import (
        POINT_STATE_INTERVAL_S, clear_point_state, load_point_state,
        point_identity, save_point_state,
    )

    t0 = time.time()
    out = dict(job)
    # An input, not a result, and a bulky one: a 15-nuisance simplex is 240
    # floats, which every record would otherwise carry into the checkpoint
    # alongside the nm_simplex it actually needs to store. The resume path
    # reads nm_simplex, never this.
    out.pop("initial_simplex", None)
    # Also an input, not a result -- reconstructed identically from
    # sigma_by_block on every launch, so a resume needs nothing from here.
    # Worse than bulky: its keys are (block_key, obs_label) tuples (see
    # Evaluator.profile_batch), which json.dump rejects outright, so leaving
    # it in `out` fails every checkpoint write for the whole profile.
    out.pop("frozen_sigmas", None)
    # A location on this machine's disk, meaningless in a stored record.
    out.pop("state_path", None)
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
        state_path = job.get("state_path")

        # What earlier jobs on this same point already spent. The caps are a
        # total across every launch, so a point that keeps being interrupted
        # still terminates instead of being resumed forever.
        nfev_used = int(job.get("nfev_used") or 0)
        nit_used = int(job.get("nit_used") or 0)

        # Real evaluations this job made; the point's running totals live in
        # the optimizer state and are reported as nfev_total/nit_total.
        eval_count = {"n": 0}
        # The best point seen. Tracked here as well as in the optimizer state
        # because a point stopped before its first simplex is complete has no
        # sorted vertex to read it from.
        best = {"f": float("inf"), "x": x_start}

        # Every profile point pins each floored block at its own sigma_used
        # from the fit rather than letting it re-concentrate (see
        # Engine.Optimize._freeze_floor) -- stamped onto the job by
        # profile_batch's frozen_sigmas, not decided here, so the caller
        # controls it per batch.
        frozen = job.get("frozen_sigmas")

        def _pinned_nll(x_full):
            return _worker_nll(x_full, frozen_sigmas=frozen)

        raw_objective = _make_nuisance_objective(_pinned_nll, param_idx, n_params)

        def nuisance_objective(x_nuisance, fixed_val):
            x_arr = np.asarray(x_nuisance, dtype=float)
            v = raw_objective(x_arr, fixed_val)
            eval_count["n"] += 1
            if np.isfinite(v) and v < best["f"]:
                best["f"] = float(v)
                best["x"] = x_arr.copy()
            return v

        def time_is_up():
            return deadline is not None and time.time() >= deadline

        bounds = job.get("nuisance_bounds")
        if bounds is not None:
            bounds = [tuple(b) if b is not None else None for b in bounds]

        nm_state = None
        if x_start.size == 0:
            # Single-parameter fit: nothing to re-optimize, so the profile value
            # is just the objective at the fixed value -- exact by definition.
            nll = raw_objective(x_start, x_fixed)
            x_opt = x_start
            out.update({"converged": True, "nit": 0, "nfev": 1})
            eval_count["n"] = 1
        else:
            options, extra_kwargs = nuisance_options(
                method, x_start.size, job.get("optimizer_kwargs"))
            max_fev = options.get("maxfev", float("inf"))
            max_it = options.get("maxiter", float("inf"))
            resumable = str(method).lower() == "nelder-mead" and can_run(
                options, extra_kwargs)

            res = None
            outcome = "done"
            if resumable:
                identity = point_identity(param_idx, x_fixed, x_start.size, method)
                nm_state = state_from_json(
                    load_point_state(state_path, identity), x_start.size)
                if nm_state is not None:
                    out["resumed_from_state"] = True
                else:
                    sim = job.get("initial_simplex")
                    lb, ub = bounds_arrays(bounds, x_start.size)
                    try:
                        nm_state = new_state(x_start, lb, ub, initial_simplex=sim,
                                             nfev=nfev_used,
                                             nit=max(1, nit_used))
                    except ValueError:
                        # A simplex of the wrong shape is a hint that did not
                        # fit, not a reason to lose the point.
                        nm_state = new_state(x_start, lb, ub, nfev=nfev_used,
                                             nit=max(1, nit_used))

                if nm_state["nfev"] >= max_fev or nm_state["nit"] >= max_it:
                    # The allowance was spent before this job began; nothing to
                    # run, and nothing to save either.
                    outcome = "capped"
                else:
                    interval = float(job.get("state_interval_s",
                                             POINT_STATE_INTERVAL_S))
                    last_save = [0.0]

                    def on_step(st):
                        now = time.time()
                        if state_path and now - last_save[0] >= interval:
                            save_point_state(state_path, identity,
                                             state_to_json(st))
                            last_save[0] = now

                    res, nm_state = minimize_nelder_mead(
                        nuisance_objective, x_start, args=(x_fixed,),
                        bounds=bounds, options=options, state=nm_state,
                        on_step=on_step, should_stop=time_is_up)
                    if res.status == STATUS_STOPPED:
                        outcome = "interrupted"
                        # Written unthrottled: this is the state the next job
                        # resumes from, and it is the last chance to save it.
                        save_point_state(state_path, identity,
                                         state_to_json(nm_state))
                    elif res.status in (1, 2):
                        outcome = "capped"
                    if outcome != "interrupted":
                        clear_point_state(state_path)
            else:
                # Some other method, or Nelder-Mead options the resumable
                # implementation does not cover: one ordinary scipy call, stopped
                # by raising out of the objective. It has no state to save, so
                # an interrupted point of this kind resumes from its best vector.
                if nfev_used >= max_fev or nit_used >= max_it:
                    outcome = "capped"
                else:
                    def guarded(x_nuisance, fixed_val):
                        v = nuisance_objective(x_nuisance, fixed_val)
                        if time_is_up():
                            raise _Stopped()
                        return v
                    try:
                        res = _minimize_nuisance(
                            guarded, x_start, (x_fixed,), method, bounds,
                            job.get("optimizer_kwargs"))
                        if not getattr(res, "success", True) and int(
                                getattr(res, "nfev", 0) or 0) >= max_fev:
                            outcome = "capped"
                    except _Stopped:
                        outcome = "interrupted"

            if res is not None:
                out.update(nuisance_convergence(res))

            if outcome == "interrupted":
                # Not a failure: the point stopped. Report where the search had
                # reached and mark the point so a later launch continues it.
                x_state, f_state = (best_of(nm_state) if nm_state is not None
                                    else (None, float("inf")))
                if f_state < best["f"]:
                    best["f"], best["x"] = f_state, x_state
                nll = best["f"] if np.isfinite(best["f"]) else FAILURE_VALUE
                x_opt = np.asarray(best["x"], dtype=float)
                out.update({
                    "interrupted": True,
                    "converged": False,
                    "opt_message": "stopped on the wall clock; resumable",
                })
            elif res is None or not np.isfinite(getattr(res, "fun", np.inf)):
                # Capped before anything could run: the point has spent its
                # whole allowance across earlier launches. It reports the value
                # it had already reached, so the record stays usable and --
                # crucially -- checkpointable. A sentinel here would never be
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

            if nm_state is not None:
                out["nm_simplex"] = np.asarray(nm_state["sim"],
                                               dtype=float).tolist()

        if nm_state is not None:
            nfev_total, nit_total = nm_state["nfev"], nm_state["nit"]
        else:
            nfev_total = nfev_used + eval_count["n"]
            nit_total = nit_used + max(0, int(out.get("nit") or 0))
        out.update({
            "nll": float(nll),
            "nuisance_x": np.asarray(x_opt, dtype=float).tolist(),
            "n_evals": eval_count["n"],
            # Totals across every launch this point has had, so the next one
            # knows how much of the allowance is left and the point terminates
            # instead of being resumed forever.
            "nfev_total": nfev_total,
            "nit_total": nit_total,
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

    def evaluate_batch(self, xs, label=None, heartbeat_s=_HEARTBEAT_SECONDS,
                       frozen_sigmas=None):
        """Evaluate every parameter vector in *xs*; return losses in input order.

        Uses submit/wait, not map -- see profile_batch's docstring for the
        general reasoning. map() (the previous implementation here) returns
        nothing until the WHOLE batch is done, so one slow straggler among
        many fast points -- a slice-screen point far from the optimum landing
        in a stiff numerical regime, say -- makes the entire batch silent for
        as long as that one point takes, indistinguishable from a hang. A
        heartbeat every heartbeat_s while nothing has landed answers that
        directly: it says how many are done, how many are still in flight, and
        an ETA once at least one has finished.

        chunk_size no longer applies to this method: submitting one task per
        vector is what makes the heartbeat and per-point completion visibility
        possible at all, and no caller in this codebase sets chunk_size to
        anything but the default anyway.

        Still returns losses in INPUT order, not completion order -- unlike
        profile_batch, whose callers key off fields in each job/result dict,
        callers here (the slice screen especially) index into the return value
        positionally.
        """
        xs = [np.asarray(x, dtype=float) for x in xs]
        n = len(xs)
        if n == 0:
            return []
        if self._pool is None:
            self.start()

        from concurrent.futures import wait, FIRST_COMPLETED

        t0 = time.time()
        tag = f" [{label}]" if label else ""
        if self.verbose:
            print(f"[pool]{tag} {n} evaluation(s) submitted to "
                  f"{self.n_workers} worker(s); progress every "
                  f"{_fmt_dur(heartbeat_s)} until results start landing.",
                  flush=True)

        try:
            futures = {self._pool.submit(_eval_task, x, frozen_sigmas): i
                      for i, x in enumerate(xs)}
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

        pending = set(futures)
        out = [None] * n
        work = 0.0
        done = 0
        failures = []

        while pending:
            finished, pending = wait(pending, timeout=heartbeat_s,
                                     return_when=FIRST_COMPLETED)

            if not finished:
                if self.verbose:
                    now = time.time()
                    msg = (f"  [pool{tag}] {done}/{n} done, "
                           f"{len(pending)} in flight, "
                           f"{_fmt_dur(now - t0)} elapsed")
                    if done:
                        rate = done / max(now - t0, 1e-9)
                        msg += f", ~{_fmt_dur((n - done) / rate)} remaining"
                    else:
                        msg += " (no point has finished yet, so no estimate)"
                    print(msg, flush=True)
                continue

            for fut in finished:
                i = futures[fut]
                try:
                    val, status, secs = fut.result()
                except Exception as exc:
                    val = FAILURE_VALUE
                    status = f"error: {type(exc).__name__}: {exc}"
                    secs = 0.0
                out[i] = val
                work += secs
                self.total_worker_seconds += secs
                if status != "ok":
                    failures.append((i, status))
                done += 1

        self.n_evals += n
        self.n_failures += len(failures)

        if self.verbose:
            elapsed = max(time.time() - t0, 1e-9)
            # Worker-seconds is the serial cost of the same work; the ratio to
            # wall time is the speedup actually realized. On the first batch it
            # includes worker startup, so it understates steady-state throughput
            # -- report both numbers rather than one flattering one.
            print(f"[pool]{tag} {n} evals in {elapsed:.1f}s wall "
                  f"({work:.1f}s of work, {work / elapsed:.1f}x, "
                  f"{n / elapsed:.1f} eval/s)", flush=True)
            if failures:
                shown = "; ".join(f"#{i}: {s}" for i, s in failures[:3])
                more = f" (+{len(failures) - 3} more)" if len(failures) > 3 else ""
                print(f"[pool]{tag} {len(failures)} failed — {shown}{more}", flush=True)

        return out

    def profile_batch(self, jobs, on_result=None, label=None,
                      heartbeat_s=_HEARTBEAT_SECONDS, budget=None,
                      frozen_sigmas=None, state_dir=None):
        """Run profile-likelihood points in parallel, within a wall budget.

        ``state_dir`` is where each running point keeps its own resumable
        optimizer state (see ``Engine.Profile_checkpoint.point_state_path``).
        Without it points still run, they just cannot be resumed mid-way after
        a kill.

        ``frozen_sigmas``, stamped onto every job here rather than left to
        each caller's job-building code, is a ``{(block_key_or_exp_id,
        obs_label): sigma_used}`` lookup (see ``Engine.Optimize.
        block_sigmas``) pinning each data-floored block found in it at its own
        resolved sigma for the point's whole nuisance re-optimization, instead
        of letting it re-concentrate as the nuisance vector moves -- see
        ``Engine.Optimize._freeze_floor``. Every profile pass submitted
        through one ``batch()`` closure gets it uniformly this way, with no
        change needed at the individual job-building sites.

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

        When *budget* runs out, running points are told to stop at their next
        evaluation, save their state and return, and are waited for; then
        :class:`~Engine.Deadline.DeadlineReached` is raised naming how many never
        started. Everything that landed has already been through *on_result*, so
        nothing computed is lost by the raise. With no budget a point runs until
        it converges or spends its allowance, and a kill is survived by the
        state file rather than by anything here.

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
        from Engine.Profile_checkpoint import (
            point_state_path, sweep_stale_temp_files,
        )

        t0 = time.time()
        results = []
        # Stamped once here, not per admitted job: unlike the deadline this does
        # not depend on the clock, so every job in the batch gets it up front.
        backlog = [dict(j, frozen_sigmas=frozen_sigmas) for j in jobs]
        futures = {}
        pending = set()
        n_jobs = len(jobs)
        done = 0
        halted = False
        tag = f" [{label}]" if label else ""

        if state_dir:
            # Litter from a process killed between writing a state file and
            # renaming it into place; one per kill, so a preempted run collects
            # them.
            sweep_stale_temp_files(state_dir)

        def admit():
            """Start points until the pool is full or the clock says stop."""
            nonlocal halted
            while backlog and len(pending) < self.n_workers:
                if budget is not None and not budget.admits():
                    halted = True
                    return
                job = backlog.pop(0)
                # Stamped here rather than where the job was built, because
                # this is the only place that knows the clock. The stop time is
                # the run's own, the same for every point: nothing is
                # predicted about how long a point will take, because a point
                # stopped early is resumed from its saved state and loses
                # nothing.
                job = dict(job,
                           deadline=(budget.work_deadline()
                                     if budget is not None else None),
                           state_path=point_state_path(
                               state_dir, job.get("param_name"),
                               job.get("x_fixed")))
                fut = self._pool.submit(_profile_task, job)
                futures[fut] = job
                pending.add(fut)

        if self.verbose:
            print(f"[pool]{tag} {n_jobs} profile point(s) for "
                  f"{self.n_workers} worker(s); progress every "
                  f"{_fmt_dur(heartbeat_s)} until results start landing.",
                  flush=True)
            if budget is not None and budget.is_limited:
                print(f"[pool]{tag} {budget.describe()}",
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
                    # Same non-JSON-safe input as in _profile_task's success
                    # path -- strip it here too, or a worker crash makes the
                    # checkpoint write fail instead of just recording the error.
                    res.pop("initial_simplex", None)
                    res.pop("frozen_sigmas", None)
                    res.update({"nll": FAILURE_VALUE, "n_evals": 0, "wall_s": 0.0,
                                "status": f"error: {type(exc).__name__}: {exc}"})
                results.append(res)
                done += 1
                self.n_evals += int(res.get("n_evals") or 0)
                self.total_worker_seconds += float(res.get("wall_s") or 0.0)
                if res.get("status") != "ok":
                    self.n_failures += 1
                if on_result is not None:
                    on_result(res)
                if self.verbose:
                    elapsed = time.time() - t0
                    # An interrupted point is progress, not a problem, and the
                    # log has to say so or every link will read as a run of
                    # failures. RESUMED marks a point that picked up its saved
                    # state after an earlier process was killed on it -- the
                    # line to count when checking that a preemptible run is
                    # really surviving evictions.
                    state = (" CONTINUES" if res.get("interrupted")
                             else " RESUMED" if res.get("resumed_from_state")
                             else "")
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
    search_max_steps=None,
):
    """Convenience constructor mirroring the spec-route local variables."""
    from Engine.Event_times import without_event_times
    from Engine.Noise_floor import export_cache

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
        search_max_steps=None if search_max_steps is None else int(search_max_steps),
        # Captured HERE, at spec-build time -- called in the parent after its
        # own clear_cache()-and-recalibrate pass (see run_optimization_from_
        # groups), so this snapshot is the same calibration the parent's own
        # subsequent diagnostics use, not whatever was cached earlier in the
        # run (e.g. during the live optimize()).
        floor_cache=export_cache(),
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

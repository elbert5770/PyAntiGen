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
        models[sim_name] = {"r": r, "r_ic": r_ic, "df_dict": df_dict}

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


def _profile_task(job):
    """Run one profile-likelihood point: minimize over the nuisance parameters
    with parameter ``param_idx`` pinned at ``x_fixed``.

    A profile point is a whole optimization, not a single evaluation, so the
    scipy call runs *inside* the worker against its local models. That is what
    makes the profile parallel: 2k x n_grid independent optimizations in flight,
    instead of one adaptive walk stepping sequentially.

    ``job`` is a plain dict so it pickles cheaply. Returns a result dict that is
    written straight to the checkpoint file.
    """
    from Engine.Optimize import _make_nuisance_objective, _minimize_nuisance

    t0 = time.time()
    out = dict(job)
    out.update({"nll": None, "status": "ok", "n_evals": 0, "wall_s": 0.0,
                "worker": os.getpid()})

    if _WORKER["spec"] is None:
        out.update({"status": "worker-not-initialized", "nll": FAILURE_VALUE})
        return out

    try:
        spec = _WORKER["spec"]
        n_params = len(spec.param_names)
        param_idx = int(job["param_idx"])
        x_fixed = float(job["x_fixed"])
        x_start = np.asarray(job["x_start"], dtype=float)

        counter = {"n": 0}

        def counted_nll(x):
            counter["n"] += 1
            return _worker_nll(x)

        nuisance_objective = _make_nuisance_objective(counted_nll, param_idx, n_params)

        bounds = job.get("nuisance_bounds")
        if bounds is not None:
            bounds = [tuple(b) if b is not None else None for b in bounds]

        if x_start.size == 0:
            # Single-parameter fit: nothing to re-optimize, so the profile value
            # is just the objective at the fixed value.
            nll = nuisance_objective(x_start, x_fixed)
            x_opt = x_start
        else:
            res = _minimize_nuisance(
                nuisance_objective, x_start, (x_fixed,),
                job.get("method", "Nelder-Mead"), bounds,
                job.get("optimizer_kwargs"),
            )
            nll = float(res.fun)
            x_opt = np.asarray(res.x, dtype=float)

        out.update({
            "nll": float(nll),
            "nuisance_x": np.asarray(x_opt, dtype=float).tolist(),
            "n_evals": counter["n"],
            "status": "ok" if np.isfinite(nll) and nll < FAILURE_VALUE else "sentinel",
        })
    except Exception as exc:
        out.update({"status": f"error: {type(exc).__name__}: {exc}",
                    "nll": FAILURE_VALUE})

    out["wall_s"] = time.time() - t0
    return out


# --------------------------------------------------------------------------
# Parent side
# --------------------------------------------------------------------------

def default_worker_count(n_workers=None):
    """Workers to use: explicit value, else all cores but one, within limits."""
    if n_workers is None:
        n_workers = max(1, (os.cpu_count() or 2) - 1)
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

    def __init__(self, spec, n_workers=None, chunk_size=None, verbose=True):
        self.spec = spec
        self.n_workers = default_worker_count(n_workers)
        self.chunk_size = chunk_size
        self.verbose = verbose
        self._pool = None
        self._blob = None
        self.n_evals = 0
        self.n_failures = 0
        self.total_worker_seconds = 0.0

    # -- lifecycle ---------------------------------------------------------

    # Calibration point: the ~370-species / 978-reaction SILK variant, whose
    # antimony source is ~124k characters, costs ~0.35 GB per compiled model.
    _REF_MODEL_CHARS = 124_000
    _REF_MODEL_GB = 0.35

    def memory_estimate_gb(self):
        """Rough resident-memory estimate for the pool, in GB.

        Each worker compiles every simulation in the spec, so the footprint
        scales as ``n_workers x n_simulations x per-model``. That product, not
        the core count, is what usually limits how wide this can run: a
        12-simulation spec at 40 workers would want well over 100 GB.

        Per-model cost is scaled from the antimony source length against a
        measured reference. That is a crude proxy -- it tracks model size, not
        RoadRunner's exact allocation -- so treat the number as an order of
        magnitude for choosing n_workers, not a budget.
        """
        chars = max(len(self.spec.model_text or ""), 1)
        per_model_gb = self._REF_MODEL_GB * (chars / self._REF_MODEL_CHARS)
        overhead_gb = 0.15
        n_models = max(len(self.spec.replicates), 1)
        return self.n_workers * (n_models * per_model_gb + overhead_gb)

    def start(self):
        if self._pool is not None:
            return self
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor

        self._blob = _serializer.dumps(self.spec)
        n_models = len(self.spec.replicates)
        est_gb = self.memory_estimate_gb()
        if self.verbose:
            print(f"[pool] starting {self.n_workers} worker(s) "
                  f"({_SERIALIZER_NAME} spec: {len(self._blob) / 1e6:.1f} MB, "
                  f"{n_models} model(s) each, ~{est_gb:.0f} GB estimated)",
                  flush=True)
        if est_gb > 32:
            print(f"[pool] WARNING: {self.n_workers} workers x {n_models} models "
                  f"is an estimated ~{est_gb:.0f} GB. Memory, not core count, is "
                  f"usually the binding constraint for specs with many "
                  f"simulations — lower n_workers if the machine starts "
                  f"swapping.", flush=True)
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

    def profile_batch(self, jobs, on_result=None, label=None):
        """Run profile-likelihood points in parallel.

        Unlike ``evaluate_batch`` this uses submit/as_completed rather than map,
        because each job is minutes long and results must be checkpointed *as
        they land* -- the whole point of checkpointing is that killing the run
        halfway keeps the half that finished.

        *on_result* is called with each result dict the moment it arrives.
        Returns results in completion order; callers key off the job fields.
        """
        if not jobs:
            return []
        if self._pool is None:
            self.start()
        from concurrent.futures import as_completed

        t0 = time.time()
        results = []
        futures = {self._pool.submit(_profile_task, j): j for j in jobs}
        done = 0
        for fut in as_completed(futures):
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
            if on_result is not None:
                on_result(res)
            if self.verbose:
                elapsed = time.time() - t0
                print(f"  [profile {done}/{len(jobs)}] {res.get('param_name')} "
                      f"= {res.get('x_fixed_linear', res.get('x_fixed')):.4g}  "
                      f"dNLL={res.get('dnll', float('nan')):.4g}  "
                      f"({res.get('n_evals')} evals, {res.get('wall_s', 0):.0f}s)"
                      f"  [{elapsed:.0f}s elapsed]", flush=True)

        if self.verbose:
            elapsed = max(time.time() - t0, 1e-9)
            work = sum(float(r.get("wall_s") or 0.0) for r in results)
            tag = f" [{label}]" if label else ""
            print(f"[pool]{tag} {len(jobs)} profile points in {elapsed:.0f}s wall "
                  f"({work:.0f}s of work, {work / elapsed:.1f}x)", flush=True)
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
    for_inference=True,
):
    """Convenience constructor mirroring the spec-route local variables."""
    return EvalSpec(
        model_text=model_text,
        paths=paths,
        events=dict(events),
        replicates=dict(replicates),
        param_names=list(param_names),
        scales=list(scales),
        groups=groups,
        group_normalization=group_normalization,
        fixed_sigmas=dict(fixed_sigmas or {}),
        events_dynamic=bool(events_dynamic),
        data_path=data_path or paths.get("data_path"),
        for_inference=bool(for_inference),
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

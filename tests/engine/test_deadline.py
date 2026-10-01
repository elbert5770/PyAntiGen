"""Checks on stopping and resuming: the optional stop time, and a point that
survives being killed.

Run from the repository root:
    python tests/engine/test_deadline.py

Two separate things are under test, because they are separate things. The first
is the optional wall-clock stop time (--wall-time, or a partition that runs to
its limit): a run must stop handing out profile points at it and report what it
did rather than dying with its results half-written. The second is what makes a
kill on a preemptible partition survivable, which has nothing to do with the
clock: a running point writes its own optimizer state as it goes, and a later
job for the same point continues from it, repeating no evaluation.

None of this needs a process pool. ``profile_batch``'s admission loop is driven
through a fake executor that completes tasks inline, which is what makes the
timing deterministic -- a test that raced a real deadline would be flaky in
exactly the way the feature exists to prevent.
"""
import contextlib
import io
import os
import sys
import tempfile
import time
from concurrent.futures import Future

import numpy as np

sys.path.insert(0, os.getcwd())

from pyantigen.engine.Deadline import (                                      # noqa: E402
    DeadlineReached,
    RunBudget,
    parse_duration,
    resolve_deadline,
)
from pyantigen.engine.Evaluator import EvalSpec, ParallelEvaluator           # noqa: E402
from pyantigen.engine.Optimize import run_parallel_profile                   # noqa: E402

failures = []


def check(name, cond, detail=""):
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}  {detail}")
        failures.append(name)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _screen_batch(xs, label=None, frozen_sigmas=None):
    """``evaluate_batch`` for the doubles below: a bowl centred on the optimum.

    The profile driver now runs the slice screen before any profile point, so a
    stand-in for ParallelEvaluator has to answer plain evaluations too. Every
    optimum in this file is the origin and every bound is several units out, so
    a sum of squares crosses the threshold on every side and the screen passes
    -- which is what these tests want, since what they are about is the wall
    clock, not identifiability. The screen itself is covered in
    tests/test_slice_screen.py. Also stands in for the driver's own frozen-
    anchor lookup (evaluate_batch([res_x], frozen_sigmas=sigma_by_block)),
    which a bowl function has no sigma blocks to freeze in the first place.
    """
    return [float(np.dot(np.asarray(x, dtype=float),
                         np.asarray(x, dtype=float))) for x in xs]


class _InlinePool:
    """Stands in for ProcessPoolExecutor: runs the job at submit time.

    Records what it was asked to run, which is the whole question for admission
    control -- not what came back, but what was allowed to start.
    """

    def __init__(self, run):
        self.run = run
        self.submitted = []

    def submit(self, _task, job, *args):
        """Dispatches on what *job* actually is, not on *_task*'s identity:
        ``evaluate_batch`` now submits one plain parameter vector per task
        (submit/wait, not map -- see Evaluator.evaluate_batch), while
        ``profile_batch`` submits a job dict. A vector gets the same bowl-
        centred-on-the-origin answer ``map`` below already gave it before that
        change; a dict keeps going through ``self.run``, which is what the
        admission-control tests actually inspect. ``*args`` absorbs
        ``evaluate_batch``'s trailing ``frozen_sigmas`` positional, which this
        stub's synthetic bowl function has no use for.
        """
        if not isinstance(job, dict):
            fut = Future()
            x = np.asarray(job, dtype=float)
            fut.set_result((float(np.dot(x, x)), "ok", 0.0))
            return fut
        self.submitted.append(job)
        fut = Future()
        fut.set_result(self.run(job))
        return fut

    def map(self, _task, xs, chunksize=None):
        """Plain evaluations, in the ``(value, status, seconds)`` shape.

        Kept for any caller still using ``pool.map`` directly; ``evaluate_batch``
        itself now goes through ``submit`` (see above), like ``profile_batch``
        already did. A bowl centred on the origin, same as ``submit``'s
        vector path -- every optimum in this file sits there.
        """
        return [(float(np.dot(np.asarray(x, dtype=float),
                              np.asarray(x, dtype=float))), "ok", 0.0)
                for x in xs]


def _evaluator(run, n_workers=2):
    """A ParallelEvaluator wired to an inline pool instead of real workers."""
    spec = EvalSpec(
        model_text="x", paths={}, events={}, replicates={"sim": {}},
        param_names=["p"], scales=["lin"], groups={},
        group_normalization=None, fixed_sigmas={},
    )
    ev = ParallelEvaluator(spec, n_workers=n_workers, verbose=False)
    ev._pool = _InlinePool(run)
    return ev


def _jobs(n):
    return [{"param_idx": 0, "param_name": "p", "x_fixed": float(i),
             "x_fixed_linear": float(i), "x_start": [0.0],
             "nuisance_bounds": None, "method": "Nelder-Mead",
             "optimizer_kwargs": None, "phase": 1, "direction": 0}
            for i in range(n)]


def _ok(job, wall_s=1.0):
    out = dict(job)
    out.update({"nll": 0.0, "dnll": 0.0, "status": "ok", "n_evals": 1,
                "wall_s": wall_s, "nuisance_x": [0.0], "converged": True})
    return out


# ---------------------------------------------------------------------------
# Duration parsing and deadline discovery
# ---------------------------------------------------------------------------

def _env(**kw):
    """Context manager: set (or with None, remove) environment variables."""
    class _Ctx:
        def __enter__(self):
            self.saved = {k: os.environ.get(k) for k in kw}
            for k, v in kw.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

        def __exit__(self, *exc):
            for k, v in self.saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
    return _Ctx()


_NO_SCHEDULER = dict(PROFILE_WALL_TIME=None, SLURM_JOB_END_TIME=None,
                     SLURM_JOB_PARTITION=None, SLURM_JOB_ID=None,
                     SLURM_JOBID=None)


def test_parse_duration():
    print("\nDuration parsing:")
    cases = {
        "4h": 14400.0, "3.5h": 12600.0, "90m": 5400.0, "45s": 45.0,
        "2d": 172800.0, "4:00:00": 14400.0, "3:30:00": 12600.0,
        "12:30": 750.0, "1-12:00:00": 129600.0, "3600": 3600.0,
    }
    for text, want in cases.items():
        got = parse_duration(text)
        check(f"{text!r} -> {want:g}s", got == want, f"got {got}")

    for bad in (None, "", "later", "4x", "1:2:3:4", "-5"):
        check(f"{bad!r} is rejected", parse_duration(bad) is None,
              f"got {parse_duration(bad)}")


# ---------------------------------------------------------------------------
# The stop time
# ---------------------------------------------------------------------------

def test_resolve_deadline():
    print("\nStop time discovery:")
    now = time.time()
    with _env(**_NO_SCHEDULER):
        check("no scheduler and no flag means no stop time",
              resolve_deadline(verbose=False) is None)

        got = resolve_deadline(wall_time="30m", now=now, verbose=False)
        check("an explicit wall time is honoured",
              got is not None and abs(got - (now + 1800.0)) < 1.0, str(got))

        os.environ["PROFILE_WALL_TIME"] = "10m"
        got = resolve_deadline(now=now, verbose=False)
        check("PROFILE_WALL_TIME is honoured",
              got is not None and abs(got - (now + 600.0)) < 1.0, str(got))

        # A typo must not take down a run that would otherwise finish.
        os.environ["PROFILE_WALL_TIME"] = "nonsense"
        check("an unparseable wall time is no stop time, not an error",
              resolve_deadline(now=now, verbose=False) is None)
        os.environ.pop("PROFILE_WALL_TIME")

        # On a partition that runs to its limit, the scheduler's end time is
        # the real end.
        os.environ["SLURM_JOB_PARTITION"] = "compute-hugemem"
        os.environ["SLURM_JOB_END_TIME"] = str(now + 7200.0)
        got = resolve_deadline(now=now, verbose=False)
        check("SLURM_JOB_END_TIME is the stop time on a non-preempted partition",
              got is not None and abs(got - (now + 7200.0)) < 1.0, str(got))

        got = resolve_deadline(wall_time="30m", now=now, verbose=False)
        check("an explicit wall time beats the scheduler",
              got is not None and abs(got - (now + 1800.0)) < 1.0, str(got))

        # An end time already past is stale, not a reason to refuse to start.
        os.environ["SLURM_JOB_END_TIME"] = str(now - 60.0)
        check("an end time in the past is ignored",
              resolve_deadline(now=now, verbose=False) is None)
        os.environ["SLURM_JOB_END_TIME"] = "garbage"
        check("an unreadable end time is ignored",
              resolve_deadline(now=now, verbose=False) is None)


def test_a_preemptible_partition_gets_no_stop_time():
    """Nothing on ckpt can say when the job will die, so nothing pretends to.

    SLURM_JOB_END_TIME there is the time *limit*, days away, and on a requeued
    job it has been seen carrying a value left over from an earlier attempt. A
    plausible-looking wrong end time is the one thing that could make a healthy
    run stop early, and there is nothing to gain from reading it: the optimizers
    save their state as they go.
    """
    print("\nPreemptible partitions:")
    now = time.time()
    for part in ("ckpt", "ckpt-g2", "ckpt-all"):
        with _env(**{**_NO_SCHEDULER, "SLURM_JOB_PARTITION": part,
                     "SLURM_JOB_END_TIME": str(now + 5 * 86400.0)}):
            check(f"{part}: the reported time limit is not used",
                  resolve_deadline(now=now, verbose=False) is None)
            got = resolve_deadline(wall_time="2h", now=now, verbose=False)
            check(f"{part}: an explicit wall time still is",
                  got is not None and abs(got - (now + 7200.0)) < 1.0, str(got))

    # The stale-value case that used to need a dedicated defence: a reading a
    # few seconds from now on a requeued ckpt job.
    with _env(**{**_NO_SCHEDULER, "SLURM_JOB_PARTITION": "ckpt",
                 "SLURM_JOB_END_TIME": str(now + 5.0)}):
        check("a stale near-now end time on ckpt cannot end the run",
              resolve_deadline(now=now, verbose=False) is None)


# ---------------------------------------------------------------------------
# The budget itself
# ---------------------------------------------------------------------------

def test_budget_admits_until_the_margin():
    print("\nBudget admission:")
    check("an unlimited budget always admits", RunBudget(deadline=None).admits())

    now = time.time()
    check("plenty of time admits",
          RunBudget(deadline=now + 2400.0, margin_s=600.0).admits())
    check("nothing is admitted inside the margin",
          not RunBudget(deadline=now + 300.0, margin_s=600.0).admits())
    check("nothing is admitted once the deadline has passed",
          not RunBudget(deadline=now - 5.0, margin_s=600.0).admits())
    # The point of dropping the minimum-useful-slice rule: a point that is
    # stopped saves its state and is resumed later, so starting one with a few
    # minutes to spare costs nothing and refusing it on a guess loses time.
    check("a point may start with only a little time before the margin",
          RunBudget(deadline=now + 700.0, margin_s=600.0).admits())
    check("an unlimited budget stamps no stop time on running points",
          RunBudget(deadline=None).work_deadline() is None)

def test_work_deadline_leaves_the_margin():
    print("\nWork deadline:")
    check("no deadline means points run to convergence",
          RunBudget(deadline=None).work_deadline() is None)

    now = time.time()
    b = RunBudget(deadline=now + 3600.0, margin_s=600.0)
    wd = b.work_deadline()
    # Points must stop early enough that the parent can still assemble traces,
    # plot and write results before the scheduler kills the job.
    check("points stop one margin before the job ends",
          abs(wd - (now + 3000.0)) < 1.0, str(wd))


def test_the_link_keeps_resuming_instead_of_idling():
    """Points that come back unfinished are carried on within the same run.

    A point stopped on the clock is a record marked interrupted. The driver's
    pass 0 turns those back into jobs and keeps doing so until every point has
    finished or spent its allowance, rather than leaving the rest of the run
    idle after one round.
    """
    print("\nResuming within one run:")
    k = 2
    names = [f"p{i}" for i in range(k)]
    # Each point needs three jobs before it converges.
    slices_needed = {}
    labels = []

    def batch(jobs, on_result=None, label=None, budget=None,
             frozen_sigmas=None, state_dir=None):
        labels.append(label)
        for job in jobs:
            key = (job["param_name"], round(float(job["x_fixed"]), 9))
            n = slices_needed.get(key, 0) + 1
            slices_needed[key] = n
            done = n >= 3
            out = dict(job)
            out.update({
                "nll": float(job["x_fixed"]) ** 2 + (0.0 if done else 3.0 / n),
                "status": "ok", "n_evals": 4, "wall_s": 1.0,
                "nuisance_x": list(job["x_start"]),
                "nm_simplex": [[1.0] * (k - 1)] * k,
                "nfev_total": int(job.get("nfev_used") or 0) + 4,
                "nit_total": int(job.get("nit_used") or 0) + 4,
                "interrupted": not done, "converged": done})
            on_result(out)

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        _t, _a, _w, convergence = run_parallel_profile(
            batch, np.zeros(k), 0.0, names, [(-10.0, 10.0)] * k, ["lin"] * k,
            n_grid=2, wald_se=np.ones(k), n_refine=1, warm_passes=0)
    log = buf.getvalue()

    resume_rounds = [l for l in labels if l and l.startswith("profile-resume")]
    check("the link resumes repeatedly rather than once",
          len(resume_rounds) >= 2, str(labels))
    check("every point reached convergence inside this one link",
          convergence.get("n_unfinished") == 0,
          str(convergence.get("n_unfinished")))
    check("the run reports itself complete",
          convergence.get("incomplete") is False, log[-300:])
    check("and the decision passes then ran",
          any(l and ("extend" in l or "refine" in l) for l in labels),
          str(labels))


# ---------------------------------------------------------------------------
# Admission control inside profile_batch
# ---------------------------------------------------------------------------

def test_batch_runs_everything_when_time_allows():
    print("\nBatch with time to spare:")
    ev = _evaluator(_ok)
    got = []
    res = ev.profile_batch(_jobs(7), on_result=got.append, label="t",
                           budget=RunBudget(deadline=time.time() + 86400.0))
    check("every point ran", len(ev._pool.submitted) == 7,
          str(len(ev._pool.submitted)))
    check("every result was delivered", len(got) == 7, str(len(got)))
    check("results are returned", len(res) == 7, str(len(res)))


def test_batch_without_a_budget_is_unchanged():
    print("\nBatch with no budget at all:")
    ev = _evaluator(_ok)
    res = ev.profile_batch(_jobs(5), label="t")
    check("a budget-free batch runs everything", len(res) == 5, str(len(res)))


def test_batch_starts_nothing_past_the_deadline():
    print("\nBatch with the deadline already gone:")
    ev = _evaluator(_ok)
    got = []
    try:
        ev.profile_batch(_jobs(6), on_result=got.append, label="t",
                         budget=RunBudget(deadline=time.time() - 1.0))
    except DeadlineReached as exc:
        check("DeadlineReached is raised", True)
        check("it names every point as not started", exc.n_remaining == 6,
              str(exc.n_remaining))
    else:
        check("DeadlineReached is raised", False, "no exception")
    check("nothing was submitted", not ev._pool.submitted,
          str(len(ev._pool.submitted)))


def test_batch_keeps_what_it_finished():
    print("\nBatch cut short partway:")

    class _StopsAfter(RunBudget):
        """Admits a fixed number of points, then refuses -- deterministically."""

        def __init__(self, n):
            super().__init__(deadline=time.time() + 3600.0)
            self.left = n

        def admits(self, now=None):
            if self.left <= 0:
                return False
            self.left -= 1
            return True

    ev = _evaluator(_ok)
    got = []
    budget = _StopsAfter(4)
    try:
        ev.profile_batch(_jobs(10), on_result=got.append, label="t",
                         budget=budget)
    except DeadlineReached as exc:
        check("the batch stops rather than running on", exc.n_remaining == 6,
              str(exc.n_remaining))
    else:
        check("the batch stops rather than running on", False, "no exception")

    check("the admitted points all ran", len(ev._pool.submitted) == 4,
          str(len(ev._pool.submitted)))
    # The reason a raise is safe: everything computed already went out through
    # on_result, which is what writes it to the checkpoint store.
    check("their results were delivered before the raise", len(got) == 4,
          str(len(got)))
    check("the run is marked as stopped early", budget.stopped_early)


def test_batch_commits_at_most_a_poolful():
    print("\nBatch commitment:")

    seen = []

    class _Watcher(RunBudget):
        def admits(self, now=None):
            seen.append(1)
            return True

    ev = _evaluator(_ok, n_workers=3)
    ev.profile_batch(_jobs(9), label="t", budget=_Watcher(deadline=None))
    # With an inline pool each job finishes before the next is admitted, so the
    # observable claim is that admission is asked per job rather than once for
    # the whole list -- which is what makes the check happen at start time.
    check("the clock is consulted for every point, not once per batch",
          len(seen) == 9, str(len(seen)))


def test_batch_stamps_the_stop_time_and_a_state_path():
    print("\nWhat a job is told:")
    from pyantigen.engine.Profile_checkpoint import point_state_path

    seen = []

    def run(job):
        seen.append(job)
        return _ok(job)

    now = time.time()
    budget = RunBudget(deadline=now + 3600.0, margin_s=600.0)
    with tempfile.TemporaryDirectory(prefix="stamp_") as root:
        state_dir = os.path.join(root, "state")
        _evaluator(run).profile_batch(_jobs(3), label="t", budget=budget,
                                      state_dir=state_dir)
        check("every job is told the run's stop time",
              all(abs(j["deadline"] - budget.work_deadline()) < 1.0
                  for j in seen), str([j["deadline"] for j in seen]))
        paths = [j["state_path"] for j in seen]
        check("every job is given a state path under the state directory",
              all(p and p.startswith(state_dir) for p in paths), str(paths))
        check("and each point gets its own",
              len(set(paths)) == 3, str(paths))
        check("the path is a function of the point alone, so a later launch "
              "finds the same one",
              paths[1] == point_state_path(state_dir, "p", 1.0), paths[1])

    seen.clear()
    _evaluator(run).profile_batch(_jobs(2), label="t")
    check("with no budget there is no stop time",
          all(j["deadline"] is None for j in seen))
    check("with no state directory there is no state path",
          all(j["state_path"] is None for j in seen))


# ---------------------------------------------------------------------------
# End to end through the profile driver
# ---------------------------------------------------------------------------

def test_profile_reports_that_it_was_cut_short():
    print("\nProfile driver under a deadline:")

    k = 3
    names = [f"p{i}" for i in range(k)]
    state = {"n": 0}

    def batch(jobs, on_result=None, label=None, budget=None,
             frozen_sigmas=None, state_dir=None):
        """Runs four points, then behaves as the real batch does when time runs out."""
        for i, job in enumerate(jobs):
            if state["n"] >= 4:
                raise DeadlineReached(len(jobs) - i, label)
            state["n"] += 1
            out = dict(job)
            out.update({"nll": float(job["x_fixed"]) ** 2, "status": "ok",
                        "n_evals": 1, "wall_s": 1.0,
                        "nuisance_x": list(job["x_start"]), "converged": True})
            on_result(out)

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        traces, anchor, where, convergence = run_parallel_profile(
            batch, np.zeros(k), 0.0, names, [(-10.0, 10.0)] * k, ["lin"] * k,
            n_grid=4, wald_se=np.ones(k), n_refine=1, warm_passes=1,
        )
    log = buf.getvalue()

    check("the driver still returns its 4-tuple",
          isinstance(traces, dict) and isinstance(convergence, dict))
    check("the report says the run is incomplete",
          convergence.get("incomplete") is True, str(convergence.get("incomplete")))
    check("it says how much was left",
          convergence.get("n_not_started", 0) > 0,
          str(convergence.get("n_not_started")))
    check("the log calls it INCOMPLETE rather than a failure",
          "INCOMPLETE" in log and "not a failure" in log)
    check("the points that landed are still in the traces",
          any(len(t[0]) for t in traces.values()))
    check("only the admitted points ran", state["n"] == 4, str(state["n"]))


def test_profile_without_a_deadline_is_untouched():
    print("\nProfile driver with no deadline:")

    k = 3
    names = [f"p{i}" for i in range(k)]

    def batch(jobs, on_result=None, label=None, budget=None,
             frozen_sigmas=None, state_dir=None):
        for job in jobs:
            out = dict(job)
            out.update({"nll": float(job["x_fixed"]) ** 2, "status": "ok",
                        "n_evals": 1, "wall_s": 1.0,
                        "nuisance_x": list(job["x_start"]), "converged": True})
            on_result(out)

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        _traces, _anchor, _where, convergence = run_parallel_profile(
            batch, np.zeros(k), 0.0, names, [(-10.0, 10.0)] * k, ["lin"] * k,
            n_grid=4, wald_se=np.ones(k), n_refine=1, warm_passes=1,
        )
    check("a run that finishes is not marked incomplete",
          convergence.get("incomplete") is False, str(convergence.get("incomplete")))
    check("nothing is reported as unstarted",
          convergence.get("n_not_started") == 0,
          str(convergence.get("n_not_started")))


def test_wall_time_env_reaches_the_pool():
    """--wall-time must survive the whole path down to admission control.

    The flag is read at the bottom of the call chain rather than threaded
    through it, so the wiring between the environment variable and the pool is
    exactly the part that can silently stop working. This drives the real
    wrapper and the real ``profile_batch`` -- only the worker processes are
    stood in for.
    """
    print("\nWall time from the environment:")
    from pyantigen.engine.Optimize import _run_parallel_profile_with_checkpoint

    k = 3
    names = [f"p{i}" for i in range(k)]

    def run(job):
        out = dict(job)
        out.update({"nll": float(job["x_fixed"]) ** 2, "status": "ok",
                    "n_evals": 1, "wall_s": 2.0,
                    "nuisance_x": list(job["x_start"]), "converged": True})
        return out

    saved = os.environ.get("PROFILE_WALL_TIME")
    saved_margin = os.environ.get("PROFILE_DEADLINE_MARGIN")
    try:
        # Already spent: the run should reach the profile stage and decline to
        # start anything, rather than running to completion.
        os.environ["PROFILE_WALL_TIME"] = "1s"
        os.environ["PROFILE_DEADLINE_MARGIN"] = "60s"

        with tempfile.TemporaryDirectory(prefix="walltime_") as root:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                out = _run_parallel_profile_with_checkpoint(
                    _evaluator(run), np.zeros(k), 0.0, names,
                    [(-10.0, 10.0)] * k, ["lin"] * k, groups={},
                    model_text="m", paths={"plot_path": root},
                    method="Nelder-Mead", optimizer_kwargs=None,
                    wald_se=np.ones(k), n_grid=4, range_factor=2.0,
                    se_span=3.0, n_refine=1, run_id="walltime",
                )
            log = buf.getvalue()

            check("the wrapper still returns its 4-tuple", len(out) == 4)
            check("an exhausted wall time stops the run",
                  out[3].get("incomplete") is True, str(out[3].get("incomplete")))
            check("the budget is reported in the log",
                  "reserved for the write-up" in log
                  and "save their state and stop" in log, log[-400:])

        # Plenty of time: the same call must complete normally.
        os.environ["PROFILE_WALL_TIME"] = "12h"
        with tempfile.TemporaryDirectory(prefix="walltime_") as root:
            with contextlib.redirect_stdout(io.StringIO()):
                out = _run_parallel_profile_with_checkpoint(
                    _evaluator(run), np.zeros(k), 0.0, names,
                    [(-10.0, 10.0)] * k, ["lin"] * k, groups={},
                    model_text="m", paths={"plot_path": root},
                    method="Nelder-Mead", optimizer_kwargs=None,
                    wald_se=np.ones(k), n_grid=4, range_factor=2.0,
                    se_span=3.0, n_refine=1, run_id="walltime",
                )
            check("an ample wall time runs to completion",
                  out[3].get("incomplete") is False,
                  str(out[3].get("incomplete")))
    finally:
        for key, val in (("PROFILE_WALL_TIME", saved),
                         ("PROFILE_DEADLINE_MARGIN", saved_margin)):
            if val is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = val


# ---------------------------------------------------------------------------
# Interrupting and resuming a single point
# ---------------------------------------------------------------------------

def _install_worker_objective(fn, k=4):
    """Point the worker-side objective at *fn* without starting a pool.

    *k* is the number of fitted parameters and must match what *fn* expects:
    the worker rebuilds a full-length vector from the nuisance vector plus the
    pinned value, so a mismatch surfaces as a broadcast error inside the
    objective and both runs come back as the failure sentinel -- which compares
    equal, and would quietly pass an equality test.
    """
    import pyantigen.engine.Evaluator as ev_mod
    ev_mod._WORKER["spec"] = EvalSpec(
        model_text="x", paths={}, events={}, replicates={"sim": {}},
        param_names=[f"p{i}" for i in range(k)], scales=["lin"] * k, groups={},
        group_normalization=None, fixed_sigmas={},
    )
    ev_mod._WORKER["models"] = {}
    ev_mod._worker_nll_original = getattr(
        ev_mod, "_worker_nll_original", ev_mod._worker_nll)

    # _profile_task always calls _worker_nll(x, frozen_sigmas=...); the
    # stub objectives below only take x, so accept and discard it here rather
    # than touching every one of them.
    def _worker_nll_stub(x, frozen_sigmas=None):
        return fn(x)

    ev_mod._worker_nll = _worker_nll_stub
    return ev_mod


def _usable(res, label):
    """Guard against comparing two identical failure sentinels."""
    ok = (res.get("status") == "ok" and res.get("nll") is not None
          and np.isfinite(res["nll"]) and res["nll"] < 1e9)
    check(f"{label} produced a real value", ok,
          f"status={res.get('status')} nll={res.get('nll')}")
    return ok


def _restore_worker_objective(ev_mod):
    ev_mod._worker_nll = ev_mod._worker_nll_original
    ev_mod._WORKER["spec"] = None


def test_a_point_stops_on_the_clock_and_says_where_it_got_to():
    print("\nInterrupting one point:")
    from pyantigen.engine.Evaluator import _profile_task

    calls = {"n": 0}

    def slow_nll(x):
        calls["n"] += 1
        time.sleep(0.002)
        return float(np.sum((np.asarray(x, float) - 1.3) ** 2))

    ev_mod = _install_worker_objective(slow_nll)
    try:
        job = {"param_idx": 0, "param_name": "p0", "x_fixed": 0.0,
               "x_fixed_linear": 0.0, "x_start": [0.0, 0.0, 0.0],
               "nuisance_bounds": [(-5.0, 5.0)] * 3, "method": "Nelder-Mead",
               "optimizer_kwargs": None, "phase": 1, "direction": 0,
               "deadline": time.time() + 0.25}
        res = _profile_task(job)

        check("the point returns rather than raising",
              res.get("status") == "ok", str(res.get("status")))
        check("it is marked interrupted", res.get("interrupted") is True,
              str(res.get("interrupted")))
        check("it reports a finite value it actually reached",
              res.get("nll") is not None and np.isfinite(res["nll"]),
              str(res.get("nll")))
        check("it carries a nuisance vector to restart from",
              len(res.get("nuisance_x") or []) == 3,
              str(res.get("nuisance_x")))
        # The whole point of slicing: an interrupted point still hands back the
        # optimizer's state, which an exception thrown through scipy would not.
        check("it carries the simplex, not just the best point",
              res.get("nm_simplex") is not None
              and len(res["nm_simplex"]) == 4,
              str(np.shape(res.get("nm_simplex"))))
        check("it reports what it spent", res.get("nfev_total", 0) > 0,
              str(res.get("nfev_total")))
        check("it did not run to convergence",
              res.get("converged") is False, str(res.get("converged")))
    finally:
        _restore_worker_objective(ev_mod)


def _quad(n, seed=0, sleep=0.0, counter=None):
    """An ill-conditioned quadratic in *n* variables, counting its evaluations."""
    rng = np.random.default_rng(seed)
    Q = np.linalg.qr(rng.standard_normal((n, n)))[0]
    A = Q @ np.diag(np.geomspace(1.0, 300.0, n)) @ Q.T
    target = np.full(n, 1.3)

    def nll(x):
        if counter is not None:
            counter["n"] += 1
        if sleep:
            time.sleep(sleep)
        d = np.asarray(x, dtype=float) - target
        return float(d @ A @ d)
    return nll


def _base_job(n, **extra):
    job = {"param_idx": 0, "param_name": "p0", "x_fixed": 0.0,
           "x_fixed_linear": 0.0, "x_start": [0.0] * (n - 1),
           "nuisance_bounds": [(-5.0, 5.0)] * (n - 1),
           "method": "Nelder-Mead", "phase": 1, "direction": 0,
           "optimizer_kwargs": {"options": {"maxfev": 400, "maxiter": 400}}}
    job.update(extra)
    return job


def test_a_point_stopped_and_resumed_is_the_uninterrupted_point():
    """The property that matters: being stopped must not move the result.

    A profile value is a scientific result, so it must not depend on how many
    times the point happened to be stopped. It used to hold only approximately
    -- a slice boundary truncated an iteration and the next slice began a fresh
    one, leaving ~3e-5 nats -- because scipy's state could only be recovered at
    the boundary. The optimizer's state is now saved and restored whole, so the
    result is *identical*: the same value, the same nuisance vector, the same
    number of evaluations, none of them repeated.
    """
    print("\nStopped and resumed:")
    import shutil
    from pyantigen.engine.Evaluator import _profile_task

    n = 6
    counter = {"n": 0}
    ev_mod = _install_worker_objective(_quad(n, sleep=0.001, counter=counter),
                                       k=n)
    root = tempfile.mkdtemp(prefix="resume_")
    try:
        counter["n"] = 0
        whole = _profile_task(_base_job(n))
        evals_whole = counter["n"]

        state_path = os.path.join(root, "state", "p0__0.0.json")
        counter["n"] = 0
        launches = 0
        while True:
            launches += 1
            res = _profile_task(_base_job(
                n, state_path=state_path, state_interval_s=0.0,
                deadline=time.time() + 0.03))
            if not res.get("interrupted") or launches > 500:
                break
        evals_stopped = counter["n"]

        _usable(whole, "the uninterrupted run")
        _usable(res, "the run stopped and resumed")
        check("the point really was stopped several times", launches >= 3,
              f"{launches} launch(es)")
        check("the value is exactly the uninterrupted value",
              res["nll"] == whole["nll"],
              f"{res['nll']!r} vs {whole['nll']!r}")
        check("and so is the nuisance vector",
              res["nuisance_x"] == whole["nuisance_x"])
        check("no evaluation was repeated across the stops",
              evals_stopped == evals_whole,
              f"{evals_stopped} vs {evals_whole}")
        check("the spend carried across launches is the whole point's spend",
              res["nfev_total"] == whole["nfev_total"],
              f"{res['nfev_total']} vs {whole['nfev_total']}")
        check("a finished point leaves no state behind",
              not os.path.exists(state_path))
        check("the state path is not smuggled into the stored record",
              "state_path" not in res)
    finally:
        _restore_worker_objective(ev_mod)
        shutil.rmtree(root, ignore_errors=True)


def test_a_point_resumed_from_a_record_continues():
    """The other resume path: a record, with no state file.

    A continuation round or a later launch builds a job from a stored record,
    which carries the simplex and the spend but not the function values. That
    costs the n+1 evaluations to re-measure the simplex, and must still continue
    rather than restart.
    """
    print("\nResumed from a record:")
    from pyantigen.engine.Evaluator import _profile_task

    n = 6
    ev_mod = _install_worker_objective(_quad(n), k=n)
    try:
        short = _profile_task(_base_job(
            n, optimizer_kwargs={"options": {"maxfev": 120, "maxiter": 120}}))
        resumed = _profile_task(_base_job(
            n, x_start=short["nuisance_x"], initial_simplex=short["nm_simplex"],
            nfev_used=short["nfev_total"], nit_used=short["nit_total"]))
        check("a resumed point improves on where it was stopped",
              resumed["nll"] < short["nll"],
              f"{resumed['nll']:.6g} vs {short['nll']:.6g}")
        check("a resumed point spends only what is left of its allowance",
              resumed["nfev_total"] <= 400, str(resumed["nfev_total"]))
        check("its spend includes what came before",
              resumed["nfev_total"] > short["nfev_total"],
              f"{resumed['nfev_total']} vs {short['nfev_total']}")
    finally:
        _restore_worker_objective(ev_mod)


def test_a_killed_point_resumes_from_its_state_file():
    """A kill mid-point loses the evaluation in flight and nothing else.

    ``Killed`` derives from BaseException so it goes straight through the
    worker's ``except Exception`` the way a SIGKILL goes through everything:
    whatever is on disk at that instant is all the next job has.
    """
    print("\nKilled mid-point:")
    import shutil
    from pyantigen.engine.Evaluator import _profile_task

    class Killed(BaseException):
        pass

    n = 6
    counter = {"n": 0}
    kill_at = {"n": None}
    base_nll = _quad(n, counter=counter)

    def nll(x):
        if kill_at["n"] is not None and counter["n"] + 1 == kill_at["n"]:
            counter["n"] += 1
            raise Killed
        return base_nll(x)

    ev_mod = _install_worker_objective(nll, k=n)
    root = tempfile.mkdtemp(prefix="killed_")
    try:
        counter["n"] = 0
        whole = _profile_task(_base_job(n))
        evals_whole = counter["n"]

        for kill in (2, 9, 25, 61):
            state_path = os.path.join(root, f"k{kill}", "p0__0.0.json")
            job = _base_job(n, state_path=state_path, state_interval_s=0.0)
            counter["n"] = 0
            kill_at["n"] = kill
            try:
                _profile_task(dict(job))
                died = False
            except Killed:
                died = True
            calls_before_death = counter["n"]
            kill_at["n"] = None
            check(f"kill at evaluation {kill}: the process really died", died)
            check(f"kill at evaluation {kill}: a state file was left behind",
                  os.path.exists(state_path))

            counter["n"] = 0
            res = _profile_task(dict(job))
            check(f"kill at evaluation {kill}: the next job resumed from it",
                  res.get("resumed_from_state") is True)
            check(f"kill at evaluation {kill}: the answer is the uninterrupted "
                  f"one", res["nll"] == whole["nll"]
                  and res["nuisance_x"] == whole["nuisance_x"],
                  f"{res['nll']!r} vs {whole['nll']!r}")
            repeated = calls_before_death + counter["n"] - evals_whole
            check(f"kill at evaluation {kill}: at most one evaluation was "
                  f"lost to it", 0 <= repeated <= 2, f"{repeated} repeated")
            check(f"kill at evaluation {kill}: the state is cleared on "
                  f"completion", not os.path.exists(state_path))
    finally:
        _restore_worker_objective(ev_mod)
        shutil.rmtree(root, ignore_errors=True)


def test_a_damaged_or_foreign_state_file_is_ignored():
    print("\nUnusable state files:")
    import shutil
    from pyantigen.engine.Evaluator import _profile_task

    class Killed(BaseException):
        pass

    n = 4
    counter = {"n": 0}
    kill_at = {"n": None}
    base_nll = _quad(n, counter=counter)

    def nll(x):
        if kill_at["n"] is not None and counter["n"] + 1 == kill_at["n"]:
            counter["n"] += 1
            raise Killed
        return base_nll(x)

    ev_mod = _install_worker_objective(nll, k=n)
    root = tempfile.mkdtemp(prefix="foreign_")
    try:
        whole = _profile_task(_base_job(n))

        garbage = os.path.join(root, "garbage.json")
        with open(garbage, "w", encoding="utf-8") as fh:
            fh.write("{this is not json")
        res = _profile_task(_base_job(n, state_path=garbage))
        check("a damaged file costs the resume, not the point",
              res.get("status") == "ok" and res["nll"] == whole["nll"]
              and not res.get("resumed_from_state"), str(res.get("status")))

        # A real state, left by a job for a *different* fixed value, at the path
        # this job will look at.
        other = os.path.join(root, "other.json")
        kill_at["n"], counter["n"] = 6, 0
        try:
            _profile_task(_base_job(n, x_fixed=0.5, x_fixed_linear=0.5,
                                    state_path=other, state_interval_s=0.0))
        except Killed:
            pass
        kill_at["n"] = None
        check("(setup) the other point left a state file", os.path.exists(other))
        res = _profile_task(_base_job(n, state_path=other))
        check("a state that belongs to another point is not resumed",
              res.get("status") == "ok" and res["nll"] == whole["nll"]
              and not res.get("resumed_from_state"), str(res.get("nll")))
    finally:
        _restore_worker_objective(ev_mod)
        shutil.rmtree(root, ignore_errors=True)


def test_a_method_without_resumable_state_still_stops_on_the_clock():
    """Only Nelder-Mead has a state to save; anything else must still stop."""
    print("\nOther methods:")
    from pyantigen.engine.Evaluator import _profile_task

    n = 4
    ev_mod = _install_worker_objective(_quad(n, sleep=0.002), k=n)
    try:
        done = _profile_task(_base_job(n, method="Powell",
                                       optimizer_kwargs=None))
        _usable(done, "Powell run to the end")
        check("it is not marked interrupted",
              done.get("interrupted") is False)

        stopped = _profile_task(_base_job(n, method="Powell",
                                          optimizer_kwargs=None,
                                          deadline=time.time() + 0.05))
        check("with a deadline it stops and says so",
              stopped.get("interrupted") is True, str(stopped.get("status")))
        check("it still reports a real value and a vector to restart from",
              np.isfinite(stopped["nll"]) and stopped["nll"] < 1e9
              and len(stopped["nuisance_x"]) == n - 1, str(stopped.get("nll")))
        check("it carries no simplex, because it has none",
              stopped.get("nm_simplex") is None)
    finally:
        _restore_worker_objective(ev_mod)


def test_a_points_allowance_is_shared_across_launches():
    print("\nBudget across launches:")
    from pyantigen.engine.Evaluator import _profile_task

    def nll(x):
        return float(np.sum((np.asarray(x, float) - 1.3) ** 2))

    ev_mod = _install_worker_objective(nll)
    try:
        base = {"param_idx": 0, "param_name": "p0", "x_fixed": 0.0,
                "x_fixed_linear": 0.0, "x_start": [0.0, 0.0, 0.0],
                "nuisance_bounds": [(-5.0, 5.0)] * 3, "method": "Nelder-Mead",
                "optimizer_kwargs": {"options": {"maxfev": 40, "maxiter": 40}},
                "phase": 1, "direction": 0}

        first = _profile_task(dict(base))
        spent = first["nfev_total"]
        check("the first launch reports its spend", spent > 0, str(spent))

        second = _profile_task(dict(base, nfev_used=spent, nit_used=spent,
                                    x_start=first["nuisance_x"],
                                    initial_simplex=first["nm_simplex"],
                                    nll_so_far=first["nll"]))
        check("a launch that starts with the allowance spent runs nothing",
              second["n_evals"] == 0, str(second["n_evals"]))
        # Without this a point that is interrupted often would be resumed for
        # ever, its budget reset by every launch, and would never terminate.
        check("and it is not marked for another resume",
              second.get("interrupted") is False,
              str(second.get("interrupted")))
        check("it says why it stopped",
              "exhausted" in (second.get("opt_message") or ""),
              str(second.get("opt_message")))
        # The bug this guards: reporting an infinite NLL here makes the record
        # a sentinel, sentinels are never checkpointed, so the stored record
        # keeps its interrupted flag and the point is resumed doing nothing on
        # every link for the rest of the chain.
        check("it reports the value already reached, not a sentinel",
              second["status"] == "ok"
              and abs(second["nll"] - first["nll"]) < 1e-12,
              f"status={second['status']} nll={second['nll']}")
    finally:
        _restore_worker_objective(ev_mod)


def test_an_uninterrupted_point_is_unchanged():
    print("\nNo deadline, no change:")
    from pyantigen.engine.Evaluator import _profile_task

    def nll(x):
        return float(np.sum((np.asarray(x, float) - 1.3) ** 2))

    ev_mod = _install_worker_objective(nll)
    try:
        res = _profile_task({
            "param_idx": 0, "param_name": "p0", "x_fixed": 0.0,
            "x_fixed_linear": 0.0, "x_start": [0.0, 0.0, 0.0],
            "nuisance_bounds": [(-5.0, 5.0)] * 3, "method": "Nelder-Mead",
            "optimizer_kwargs": None, "phase": 1, "direction": 0})
        check("it converges", res.get("converged") is True,
              str(res.get("opt_message")))
        check("it is not marked interrupted",
              res.get("interrupted") is False, str(res.get("interrupted")))
        # The pinned parameter sits at 0 while its target is 1.3, so the
        # minimum over the rest is that one term: (0 - 1.3)**2.
        check("it finds the constrained minimum",
              abs(res["nll"] - 1.69) < 1e-5, str(res["nll"]))
    finally:
        _restore_worker_objective(ev_mod)


# ---------------------------------------------------------------------------
# The driver's handling of unfinished points
# ---------------------------------------------------------------------------

def test_unfinished_points_hold_back_the_decision_passes():
    """Passes 2-5 must not read dNLL off a point that is still mid-optimization.

    An unfinished point sits above the true profile, so a side containing one
    looks like it has already crossed the threshold. Extending or refining from
    that reading steps the wrong way and narrows the interval -- the one
    direction of error that matters here.
    """
    print("\nHolding the decision passes:")

    k = 3
    names = [f"p{i}" for i in range(k)]
    labels = []

    def batch(jobs, on_result=None, label=None, budget=None,
             frozen_sigmas=None, state_dir=None):
        labels.append(label)
        for job in jobs:
            out = dict(job)
            out.update({"nll": float(job["x_fixed"]) ** 2, "status": "ok",
                        "n_evals": 5, "wall_s": 1.0,
                        "nuisance_x": list(job["x_start"]),
                        "nm_simplex": [[0.0] * (k - 1)] * k,
                        "nfev_total": 5, "nit_total": 5,
                        # Every point comes back unfinished.
                        "interrupted": True, "converged": False})
            on_result(out)

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        _t, _a, _w, convergence = run_parallel_profile(
            batch, np.zeros(k), 0.0, names, [(-10.0, 10.0)] * k, ["lin"] * k,
            n_grid=4, wald_se=np.ones(k), n_refine=1, warm_passes=1)
    log = buf.getvalue()

    check("pass 1 ran", any(l and "pass1" in l for l in labels), str(labels))
    check("no extension pass ran",
          not any(l and "extend" in l for l in labels), str(labels))
    check("no refinement pass ran",
          not any(l and "refine" in l for l in labels), str(labels))
    check("no warm pass ran",
          not any(l and "warm" in l for l in labels), str(labels))
    check("the log explains the hold", "holding passes 2-5" in log)
    check("the run reports itself incomplete",
          convergence.get("incomplete") is True)
    check("it counts the unfinished points",
          convergence.get("n_unfinished", 0) > 0,
          str(convergence.get("n_unfinished")))


def test_a_later_launch_resumes_unfinished_points_first():
    print("\nResuming across launches:")
    import shutil
    from pyantigen.engine.Optimize import _run_parallel_profile_with_checkpoint

    k = 3
    names = [f"p{i}" for i in range(k)]
    root = tempfile.mkdtemp(prefix="resume_")
    state = {"interrupt": True, "labels": [], "resume_jobs": []}

    def run(evaluator_batch):
        class Fake:
            def __init__(self):
                self.profile_batch = evaluator_batch
                self.evaluate_batch = _screen_batch
        with contextlib.redirect_stdout(io.StringIO()):
            return _run_parallel_profile_with_checkpoint(
                Fake(), np.zeros(k), 0.0, names, [(-10.0, 10.0)] * k,
                ["lin"] * k, groups={}, model_text="m",
                paths={"plot_path": root}, method="Nelder-Mead",
                optimizer_kwargs=None, wald_se=np.ones(k), n_grid=4,
                range_factor=2.0, se_span=3.0, n_refine=1, run_id="resume")

    def batch(jobs, on_result=None, label=None, budget=None,
             frozen_sigmas=None, state_dir=None):
        state["labels"].append(label)
        if label and label.startswith("profile-resume"):
            state["resume_jobs"] = [dict(j) for j in jobs]
        # The link ends after a couple of batches, which is how unfinished
        # points come to be carried into the next launch at all: within a
        # launch the drain loop keeps working on them.
        if state["interrupt"] and len(state["labels"]) > 2:
            raise DeadlineReached(len(jobs), label)
        for job in jobs:
            out = dict(job)
            nll = float(job["x_fixed"]) ** 2
            if state["interrupt"]:
                nll += 5.0          # unfinished points sit above the truth
            out.update({"nll": nll, "status": "ok", "n_evals": 5,
                        "wall_s": 1.0, "nuisance_x": list(job["x_start"]),
                        "nm_simplex": [[1.0] * (k - 1)] * k,
                        "nfev_total": int(job.get("nfev_used") or 0) + 5,
                        "nit_total": int(job.get("nit_used") or 0) + 5,
                        "interrupted": state["interrupt"],
                        "converged": not state["interrupt"]})
            on_result(out)

    try:
        run(batch)          # first launch: everything comes back unfinished
        n_first = len([l for l in state["labels"] if l])

        state["interrupt"] = False
        state["labels"] = []
        _t, _a, _w, convergence = run(batch)   # second launch: they finish

        check("the second launch starts with a resume pass",
              state["labels"]
              and state["labels"][0].startswith("profile-resume"),
              str(state["labels"][:3]))
        check("it resumes every unfinished point",
              len(state["resume_jobs"]) > 0, str(len(state["resume_jobs"])))
        check("resumed jobs carry the saved simplex",
              all(j.get("initial_simplex") is not None
                  for j in state["resume_jobs"]),
              "some job had no simplex")
        # The first launch drained twice before its clock ran out, so the spend
        # carried across the launch boundary is what both rounds accumulated.
        check("resumed jobs carry the spend so far",
              all(int(j.get("nfev_used") or 0) >= 5
                  for j in state["resume_jobs"]),
              str([j.get("nfev_used") for j in state["resume_jobs"][:3]]))
        check("resumed jobs restart from the stored nuisance vector",
              all(j.get("resumed") for j in state["resume_jobs"]))
        check("once finished, the decision passes run",
              any(l and ("extend" in l or "refine" in l or "warm" in l)
                  for l in state["labels"]), str(state["labels"]))
        check("the run is no longer incomplete",
              convergence.get("incomplete") is False,
              str(convergence.get("incomplete")))
        check("nothing is left unfinished",
              convergence.get("n_unfinished") == 0,
              str(convergence.get("n_unfinished")))
        check("the first launch did some work", n_first > 0, str(n_first))
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_a_resumed_point_that_gains_nothing_still_records_its_spend():
    """Otherwise a stubborn point is resumed for ever against a budget that
    never moves, and the run can never terminate."""
    print("\nBookkeeping on a non-improving resume:")
    import shutil
    from pyantigen.engine.Optimize import _run_parallel_profile_with_checkpoint

    k = 2
    names = [f"p{i}" for i in range(k)]
    root = tempfile.mkdtemp(prefix="stall_")
    state = {"rounds": 0, "seen_used": []}

    def batch(jobs, on_result=None, label=None, budget=None,
             frozen_sigmas=None, state_dir=None):
        if label and label.startswith("profile-resume"):
            state["rounds"] += 1
            state["seen_used"].append(
                sorted(int(j.get("nfev_used") or 0) for j in jobs))
            # End the link after a few rounds so the test terminates on the
            # clock, the way a real link does.
            if state["rounds"] > 3:
                raise DeadlineReached(len(jobs), label)
        for job in jobs:
            out = dict(job)
            out.update({
                # The same value every time: no progress at all.
                "nll": float(job["x_fixed"]) ** 2, "status": "ok",
                "n_evals": 7, "wall_s": 1.0,
                "nuisance_x": list(job["x_start"]),
                "nm_simplex": [[1.0] * (k - 1)] * k,
                "nfev_total": int(job.get("nfev_used") or 0) + 7,
                "nit_total": int(job.get("nit_used") or 0) + 7,
                "interrupted": True, "converged": False})
            on_result(out)

    def run():
        class Fake:
            def __init__(self):
                self.profile_batch = batch
                self.evaluate_batch = _screen_batch
        with contextlib.redirect_stdout(io.StringIO()):
            return _run_parallel_profile_with_checkpoint(
                Fake(), np.zeros(k), 0.0, names, [(-10.0, 10.0)] * k,
                ["lin"] * k, groups={}, model_text="m",
                paths={"plot_path": root}, method="Nelder-Mead",
                optimizer_kwargs=None, wald_se=np.ones(k), n_grid=4,
                range_factor=2.0, se_span=3.0, n_refine=1, run_id="stall")

    try:
        run()
        rounds = state["seen_used"]
        check("the link resumed the point several times",
              len(rounds) >= 3, str(rounds))
        # The point never improves, so if the non-improving merge dropped the
        # bookkeeping every round would resume from the same spend and the run
        # could never terminate. Each round must start from more than the last.
        first_of = [r[0] for r in rounds]
        check("the spend accumulates even though nothing improves",
              all(b > a for a, b in zip(first_of, first_of[1:])),
              str(first_of))
        check("and it accumulates by what each round actually spent",
              first_of[1] - first_of[0] == 7, str(first_of))

        # It must also survive the launch boundary, which is the checkpoint
        # round-trip rather than the in-memory one.
        state["rounds"] = 0
        state["seen_used"] = []
        run()
        # The last round of the previous launch raised before doing any work,
        # so the stored total is what the round before it left. The point is
        # that the new launch carries that forward rather than starting over.
        check("a later launch picks up the accumulated spend",
              state["seen_used"]
              and state["seen_used"][0][0] >= first_of[-1]
              and state["seen_used"][0][0] > first_of[0],
              f"{state['seen_used'][:1]} vs launch-1 rounds {first_of}")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_a_stubborn_point_eventually_stops_being_resumed():
    """The chain has to terminate even if a point never converges.

    A point that is interrupted on every link must still run out of allowance
    and be reported as capped. Otherwise every remaining link of the chain
    spends its whole four hours resuming a point that can no longer do
    anything, and the run never ends.
    """
    print("\nTermination of a stubborn point:")
    import shutil
    from pyantigen.engine.Optimize import _run_parallel_profile_with_checkpoint
    from pyantigen.engine.Evaluator import _profile_task

    # Enough nuisance parameters that Nelder-Mead cannot finish in one job,
    # and an objective slow enough that the deadline really bites.
    k = 6
    names = [f"p{i}" for i in range(k)]
    root = tempfile.mkdtemp(prefix="stubborn_")

    allowance = 150          # total evaluations a point may ever spend
    per_eval = 0.002         # seconds
    slice_s = 0.06           # so roughly 30 evaluations fit in each job

    rng = np.random.default_rng(3)
    Q = np.linalg.qr(rng.standard_normal((k, k)))[0]
    A = Q @ np.diag(np.geomspace(1.0, 1e4, k)) @ Q.T

    def nll(x):
        time.sleep(per_eval)
        d = np.asarray(x, dtype=float) - 1.3
        return float(d @ A @ d)

    ev_mod = _install_worker_objective(nll, k=k)

    def batch(jobs, on_result=None, label=None, budget=None,
             frozen_sigmas=None, state_dir=None):
        for job in jobs:
            out = _profile_task(dict(
                job,
                optimizer_kwargs={"options": {"maxfev": allowance,
                                              "maxiter": allowance}},
                deadline=time.time() + slice_s))
            out["dnll"] = out["nll"]
            on_result(out)

    def run():
        class Fake:
            def __init__(self):
                self.profile_batch = batch
                self.evaluate_batch = _screen_batch
        with contextlib.redirect_stdout(io.StringIO()) as buf:
            out = _run_parallel_profile_with_checkpoint(
                Fake(), np.zeros(k), 0.0, names, [(-5.0, 5.0)] * k,
                ["lin"] * k, groups={}, model_text="m",
                paths={"plot_path": root}, method="Nelder-Mead",
                optimizer_kwargs=None, wald_se=np.ones(k), n_grid=2,
                range_factor=2.0, se_span=3.0, n_refine=1, run_id="stubborn")
        return out, buf.getvalue()

    try:
        out, log = run()
        rounds = log.count("[profile] pass 0 (round")
        check("points really were interrupted and resumed repeatedly",
              rounds >= 2, f"{rounds} resume round(s)")
        # The guarantee. Every point either converges or spends its shared
        # allowance, so the loop ends on its own -- no wall clock needed.
        check("the run stops resuming rather than looping for ever",
              out[3].get("n_unfinished") == 0,
              f"{out[3].get('n_unfinished')} still unfinished after "
              f"{rounds} rounds")
        check("and it reports itself complete",
              out[3].get("incomplete") is False, str(out[3].get("incomplete")))
    finally:
        _restore_worker_objective(ev_mod)
        shutil.rmtree(root, ignore_errors=True)


# ---------------------------------------------------------------------------
# Memory-bound worker sizing
# ---------------------------------------------------------------------------

def test_workers_are_capped_by_memory():
    print("\nWorker sizing:")
    spec = EvalSpec(
        model_text="x" * 124_000, paths={}, events={},
        replicates={f"sim{i}": {} for i in range(12)},
        param_names=["p"], scales=["lin"], groups={},
        group_normalization=None, fixed_sigmas={},
    )
    # The calibration point: a 124k-character model costs ~0.35 GB compiled,
    # and each worker compiles all twelve simulations.
    per = ParallelEvaluator(spec, n_workers=1, verbose=False,
                            memory_limit_gb=0).per_worker_memory_gb()
    check("per-worker cost tracks models x model size", 4.0 < per < 4.6,
          f"{per:.2f} GB")

    # 40 workers of this spec is ~174 GB, which genuinely does fit in a 240 GB
    # allocation. The cap is not there to second-guess a pool that fits.
    fits = ParallelEvaluator(spec, n_workers=40, verbose=False,
                             memory_limit_gb=240.0)
    check("a pool that fits its allocation is left alone",
          fits.n_workers == 40, str(fits.n_workers))

    # Halve the allocation and the same 40 cores no longer fit. Before this,
    # the pool would have started all 40 and been OOM-killed mid-batch.
    ev = ParallelEvaluator(spec, n_workers=40, verbose=False,
                           memory_limit_gb=120.0)
    check("40 cores do not become 40 workers at 120 GB",
          ev.n_workers < 40, str(ev.n_workers))
    check("the pool plans to fit inside the allocation",
          ev.memory_estimate_gb() <= 120.0, f"{ev.memory_estimate_gb():.0f} GB")
    check("it leaves headroom rather than filling the allocation",
          ev.memory_estimate_gb() <= 0.8 * 120.0,
          f"{ev.memory_estimate_gb():.0f} GB")
    check("the requested count is remembered for reporting",
          ev.n_workers_requested == 40, str(ev.n_workers_requested))

    # An unknown limit has to fail open, or a machine with unreadable memory
    # info would silently drop to one worker.
    unknown = ParallelEvaluator(spec, n_workers=8, verbose=False,
                                memory_limit_gb=0)
    check("an unknown memory limit does not restrict", unknown.n_workers == 8,
          str(unknown.n_workers))

    tiny = ParallelEvaluator(spec, n_workers=8, verbose=False,
                             memory_limit_gb=1.0)
    check("at least one worker survives an impossible budget",
          tiny.n_workers == 1, str(tiny.n_workers))


def test_zz_every_check_passed():
    """Make the checks above visible to pytest.

    ``check`` records failures rather than raising, so that one run reports
    every problem instead of stopping at the first. Under pytest that means a
    test function with failing checks still returns normally and is reported
    green -- which is how a broken assertion in this file went unnoticed. This
    runs last and turns the collected failures into one real assertion.
    """
    assert not failures, "failed checks: " + ", ".join(failures)


def main():
    test_parse_duration()
    test_resolve_deadline()
    test_a_preemptible_partition_gets_no_stop_time()
    test_budget_admits_until_the_margin()
    test_work_deadline_leaves_the_margin()
    test_the_link_keeps_resuming_instead_of_idling()
    test_batch_runs_everything_when_time_allows()
    test_batch_without_a_budget_is_unchanged()
    test_batch_starts_nothing_past_the_deadline()
    test_batch_keeps_what_it_finished()
    test_batch_commits_at_most_a_poolful()
    test_batch_stamps_the_stop_time_and_a_state_path()
    test_a_point_stops_on_the_clock_and_says_where_it_got_to()
    test_a_point_stopped_and_resumed_is_the_uninterrupted_point()
    test_a_point_resumed_from_a_record_continues()
    test_a_killed_point_resumes_from_its_state_file()
    test_a_damaged_or_foreign_state_file_is_ignored()
    test_a_method_without_resumable_state_still_stops_on_the_clock()
    test_a_points_allowance_is_shared_across_launches()
    test_an_uninterrupted_point_is_unchanged()
    test_unfinished_points_hold_back_the_decision_passes()
    test_a_later_launch_resumes_unfinished_points_first()
    test_a_resumed_point_that_gains_nothing_still_records_its_spend()
    test_a_stubborn_point_eventually_stops_being_resumed()
    test_profile_reports_that_it_was_cut_short()
    test_profile_without_a_deadline_is_untouched()
    test_wall_time_env_reaches_the_pool()
    test_workers_are_capped_by_memory()

    print("\n" + "=" * 72)
    if failures:
        print(f"FAILED ({len(failures)}):")
        for f in failures:
            print("  -", f)
        return 1
    print("ALL DEADLINE CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""An optional wall-clock stop time, and nothing that tries to predict the future.

What this is *not* for. It is not what keeps a run alive on a preemptible
partition. Hyak's ``ckpt`` partitions stop a job without notice and with no
grace period, and the time limit Slurm reports (``SLURM_JOB_END_TIME``) is not
when that happens, so no amount of watching the clock can say when the job will
die. What makes a kill survivable is that the optimizers write their state as
they go -- see :mod:`pyantigen.engine.Nelder_mead` -- so at any instant the copy on disk
is at most a couple of evaluations old. Given that, there is nothing left for a
deadline to protect against on ckpt, and this module deliberately does nothing
there.

What it is for. Two cases where the end *is* known and stopping tidily is worth
a little:

* ``--wall-time`` (or ``PROFILE_WALL_TIME``), an explicit budget, which is how
  one asks for a shorter run than the allocation on a laptop or in a test.
* A partition that runs to its ``--time`` limit and is never preempted, where
  ``SLURM_JOB_END_TIME`` really is the end.

In both, the point of stopping a little early is the write-up: the trace
assembly, the plots and the results file take real time, and a run that is
killed in the middle of them has computed everything and reported nothing.
So the run stops handing out work ``margin_s`` before the end, lets every point
in flight save its state at its next evaluation and return, and reports what it
has as ``INCOMPLETE``.

A wrong answer here is cheap by construction. A stop time that comes out too
early makes the run end early and say so; one that comes out too late means the
job is killed while working, which the state on disk already survives. Neither
loses a computed point, which is why none of the source-ranking, staleness
detection or fallback logic this file used to carry is needed any more.

Off a scheduler with no flag, :func:`resolve_deadline` returns ``None``, every
check passes, and the run behaves exactly as it always did.
"""

import os
import time

# How long before the end the run stops handing out work. It has to cover what
# happens *after* the last point lands: the pool shutdown, trace assembly,
# plotting and the results write, none of which are instant on a large spec.
DEFAULT_MARGIN_S = 600.0

# Set by Model_run.py from --wall-time. An environment variable rather than a
# threaded argument because the only consumer sits at the bottom of a call chain
# that would otherwise need the flag added to a dozen signatures for it.
WALL_TIME_ENV = "PROFILE_WALL_TIME"
MARGIN_ENV = "PROFILE_DEADLINE_MARGIN"


class DeadlineReached(RuntimeError):
    """Raised when the wall budget stopped a batch before its jobs were done.

    Carries how much work was left so the caller can say so plainly. This is a
    normal, successful outcome of a run with a stop time -- not a failure -- and
    callers are expected to catch it, finish reporting on what they have, and
    exit 0.
    """

    def __init__(self, n_remaining, label=None):
        self.n_remaining = int(n_remaining)
        self.label = label
        where = f" in {label}" if label else ""
        super().__init__(
            f"wall-clock budget reached with {self.n_remaining} point(s) "
            f"not started{where}"
        )


# ---------------------------------------------------------------------------
# Parsing and discovery
# ---------------------------------------------------------------------------

def parse_duration(text):
    """Seconds from ``4h``, ``3.5h``, ``90m``, ``45s``, ``4:00:00`` or ``1-12:00:00``.

    A bare number is seconds. That differs from ``sbatch --time``, where a bare
    number is minutes, so prefer an explicit suffix when writing one by hand.
    Returns None for anything unparseable, which the callers treat as "no
    budget" rather than as an error -- a malformed value must not take down a
    run that would otherwise have completed.
    """
    if text is None:
        return None
    if isinstance(text, (int, float)):
        return float(text) if text > 0 else None

    s = str(text).strip().lower()
    if not s:
        return None

    days = 0.0
    if "-" in s:
        head, _, s = s.partition("-")
        try:
            days = float(head)
        except ValueError:
            return None

    try:
        if ":" in s:
            parts = [float(p) for p in s.split(":")]
            if len(parts) > 3:
                return None
            while len(parts) < 3:
                parts.insert(0, 0.0)
            h, m, sec = parts
            total = h * 3600.0 + m * 60.0 + sec
        elif s[-1] in "smhd":
            mult = {"s": 1.0, "m": 60.0, "h": 3600.0, "d": 86400.0}[s[-1]]
            total = float(s[:-1]) * mult
        else:
            total = float(s)
    except (ValueError, IndexError):
        return None

    total += days * 86400.0
    return total if total > 0 else None


def partition_is_preemptible():
    """Whether this job can be evicted before its time limit.

    On Hyak the checkpoint partitions -- ``ckpt``, ``ckpt-g2``, ``ckpt-all`` --
    run on other groups' idle nodes and are stopped without notice when an owner
    reclaims them. Every other partition runs to its ``--time`` and is not
    preempted. Read from the partition name because that is the only thing Slurm
    exposes that distinguishes the two; off a scheduler there is nothing to be
    preempted by, so the answer is False.
    """
    partition = (os.environ.get("SLURM_JOB_PARTITION") or "").lower()
    return "ckpt" in partition


def resolve_deadline(wall_time=None, now=None, verbose=True):
    """Absolute epoch time this process should be finished by, or None.

    In order:

    1. *wall_time*, else ``PROFILE_WALL_TIME`` -- an explicit budget measured
       from now. It wins over the scheduler because it is how someone asks for a
       shorter run than the allocation allows, and because it is the only source
       that exists on a laptop.
    2. ``SLURM_JOB_END_TIME``, but only on a partition that is not preempted.
       On ckpt the job's real end is an eviction that this cannot see, the
       reported limit is days away, and on a requeued job the variable has been
       seen to carry a value left over from an earlier attempt. Nothing is lost
       by ignoring it there -- state is written as the run goes -- and a
       plausible-looking wrong end time is the one thing here that could make
       a healthy run stop early.

    None means no stop time, which turns every downstream check into a no-op.
    """
    now = time.time() if now is None else now

    budget = parse_duration(wall_time)
    label = None
    if budget is None:
        raw_env = os.environ.get(WALL_TIME_ENV)
        budget = parse_duration(raw_env)
        if raw_env:
            label = f"{WALL_TIME_ENV}={raw_env!r}"
    elif wall_time:
        label = f"--wall-time {wall_time!r}"
    if budget is not None:
        end = now + budget
        if verbose:
            print(f"[deadline] stop time from {label}: {_fmt_epoch(end)} "
                  f"({budget / 60.0:.0f} min from now)", flush=True)
        return end

    if partition_is_preemptible():
        if verbose:
            print("[deadline] preemptible partition: no stop time. Nothing "
                  "here can predict an eviction, and every optimizer saves "
                  "its state as it goes, so a kill costs a couple of "
                  "evaluations either way.", flush=True)
        return None

    raw = os.environ.get("SLURM_JOB_END_TIME")
    try:
        end = float(raw) if raw else None
    except ValueError:
        end = None
    if end is not None and end > now:
        if verbose:
            print(f"[deadline] stop time from SLURM_JOB_END_TIME: "
                  f"{_fmt_epoch(end)} ({(end - now) / 60.0:.0f} min from now)",
                  flush=True)
        return end
    return None


def _fmt_epoch(t):
    """Local-time rendering of an epoch float, for a human reading a log."""
    try:
        return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t))
    except (OSError, OverflowError, ValueError):
        return repr(t)


def _default_margin():
    return parse_duration(os.environ.get(MARGIN_ENV)) or DEFAULT_MARGIN_S


class RunBudget:
    """How much of this run's wall clock is left before it should stop.

    Deliberately small: a stop time and a margin, and the two questions worth
    asking of them -- "may another point start?" and "when must a running point
    stop?". An unset deadline makes both permissive, so the same object can be
    constructed unconditionally and passed everywhere.

    There is no estimate of what a point costs. The old admission rule refused a
    point that would not fit, which is exactly what made it wrong: a point can
    be resumed from wherever it is stopped, so starting one with ten minutes to
    spare is not a loss, and refusing it on a guess is.
    """

    def __init__(self, deadline=None, margin_s=None):
        self.deadline = deadline
        self.margin_s = _default_margin() if margin_s is None else float(margin_s)
        # Set once a batch has been cut short, so the run can report honestly
        # that it stopped for time rather than because the work was done.
        self.stopped_early = False

    @property
    def is_limited(self):
        return self.deadline is not None

    def remaining(self, now=None):
        """Seconds left before the deadline; ``inf`` when there is none."""
        if self.deadline is None:
            return float("inf")
        return self.deadline - (time.time() if now is None else now)

    def work_deadline(self):
        """The last moment this run may still be computing, or None.

        Leaves the margin for trace assembly, plotting and the results write.
        Running points are told to stop here; None means run to convergence.
        """
        if self.deadline is None:
            return None
        return self.deadline - self.margin_s

    def admits(self, now=None):
        """Whether another point may be started."""
        if self.deadline is None:
            return True
        return self.remaining(now) > self.margin_s

    def describe(self):
        """One line for the log, saying what the budget will actually do."""
        if self.deadline is None:
            return "no wall-clock stop time; running until the work is done"
        return (f"{self.remaining() / 60.0:.0f} min left, "
                f"{self.margin_s / 60.0:.0f} min of it reserved for the "
                f"write-up; no new point starts after that, and running points "
                f"save their state and stop at it")

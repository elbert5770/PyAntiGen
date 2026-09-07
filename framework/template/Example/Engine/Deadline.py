"""Wall-clock budget for a run that expects to be killed.

On a preemptible partition the job does not end when the work ends; it ends when
the scheduler says so. The four-hour cap on Hyak's ``ckpt`` partition is not by
itself the problem -- the profile store is per-point and a relaunch resumes --
but a run that does not know its own deadline keeps handing out new profile
points right up to the moment it is shot, and every point in flight at that
moment is lost. With forty workers each holding a point that has been running
for hours, an eviction at 3h55m throws away most of a link's compute. That, and
not the length of the work, is why four hours was not enough.

The fix is admission control: before starting another point, ask whether it can
plausibly finish. A point that cannot is simply not started, the link ends early
and cleanly, and everything already computed is on disk. What made this
affordable is that the profile store is append-only and a resume is a no-op, so
"stop early" costs nothing but the points not yet begun.

How long a point takes is measured rather than guessed. The first link of a
chain has nothing to go on and admits work until its own first result lands;
every link after it starts from the durations the earlier ones wrote down.

Nothing here is Slurm-specific at the call site. Off-cluster
:func:`resolve_deadline` returns ``None``, every check passes, and the run
behaves exactly as it did before this module existed.
"""

import json
import math
import os
import subprocess
import time

# How close to the deadline a link is willing to start nothing new. It has to
# cover what happens *after* the last point lands: the pool shutdown, trace
# assembly, plotting and the results write, none of which are instant on a
# large spec.
DEFAULT_MARGIN_S = 600.0

# Set by Model_run.py from --wall-time. An environment variable rather than a
# threaded argument because the only consumer sits at the bottom of a call chain
# that would otherwise need the flag added to a dozen signatures for it, and
# because on a cluster the deadline genuinely does arrive through the
# environment.
WALL_TIME_ENV = "PROFILE_WALL_TIME"
MARGIN_ENV = "PROFILE_DEADLINE_MARGIN"
MIN_SLICE_ENV = "PROFILE_MIN_SLICE"
SLICE_CAP_ENV = "PROFILE_SLICE_CAP"

# How long a single job may run before it must hand its state back, whether or
# not the link is anywhere near its deadline.
#
# The deadline protects against the *time limit*. On a preemptible partition
# that is not what usually stops the job: preemption arrives unannounced, and
# `SLURM_JOB_END_TIME` reports the limit rather than the eviction. A worker only
# returns its result -- and the parent only checkpoints it -- when its job ends,
# so without this cap a preemption three hours into a four-hour point discards
# all three hours.
#
# Capping the job turns that unbounded loss into a bounded one: at most one
# slice per point in flight. The cost is the n+1 evaluations each new job spends
# re-establishing its simplex, which at 30 minutes is a few percent.
DEFAULT_SLICE_CAP_S = 1800.0

# The shortest slice of wall clock worth starting a point in. A resumed point
# re-evaluates its whole simplex before it improves on anything, so a slice
# below this is spent on overhead and the work is better left to the next link.
DEFAULT_MIN_SLICE_S = 900.0

# Durations older than this many points are dropped. Keeps the file small and
# lets the estimate track a spec whose cost has changed rather than averaging
# over every run the directory has ever seen.
_KEEP_DURATIONS = 200

# Writing timing.json on every result would be one fsync per point for data
# that is only advisory; writing it only at the end would lose it to the very
# eviction it exists to survive.
_SAVE_EVERY = 10


class DeadlineReached(RuntimeError):
    """Raised when the wall budget stopped a batch before its jobs were done.

    Carries how much work was left so the caller can say so plainly. This is a
    normal, successful outcome of a link on a preemptible queue -- not a
    failure -- and callers are expected to catch it, finish reporting on what
    they have, and exit 0.
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


def _deadline_from_slurm_env():
    """``SLURM_JOB_END_TIME``: the scheduler's own answer, in epoch seconds."""
    raw = os.environ.get("SLURM_JOB_END_TIME")
    if not raw:
        return None
    try:
        end = float(raw)
    except ValueError:
        return None
    return end if end > 0 else None


def _deadline_from_scontrol():
    """The same answer via ``scontrol``, for builds that export no end time.

    Deliberately best-effort: a missing binary, a slow controller or an
    unparseable field all return None and leave the run unlimited, which is the
    behaviour that existed before any of this.
    """
    job_id = os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOBID")
    if not job_id:
        return None
    try:
        out = subprocess.run(
            ["scontrol", "show", "job", "-o", str(job_id)],
            capture_output=True, text=True, timeout=10, check=False,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None

    for field in out.split():
        if not field.startswith("EndTime="):
            continue
        value = field[len("EndTime="):]
        if value in ("Unknown", "None", ""):
            return None
        try:
            # Slurm renders local time as YYYY-MM-DDTHH:MM:SS.
            return time.mktime(time.strptime(value, "%Y-%m-%dT%H:%M:%S"))
        except ValueError:
            return None
    return None


def resolve_deadline(wall_time=None, now=None):
    """Absolute epoch time this process should be finished by, or None.

    Sources, in order of trust:

    1. *wall_time*, else ``PROFILE_WALL_TIME`` -- an explicit budget measured
       from now. It wins over the scheduler because it is how someone asks for
       a shorter link than the allocation allows, and because it is the only
       source that exists on a laptop.
    2. ``SLURM_JOB_END_TIME``.
    3. ``scontrol show job``.

    None means no deadline, which turns every downstream check into a no-op.
    """
    now = time.time() if now is None else now

    budget = parse_duration(wall_time)
    if budget is None:
        budget = parse_duration(os.environ.get(WALL_TIME_ENV))
    if budget is not None:
        return now + budget

    for source in (_deadline_from_slurm_env, _deadline_from_scontrol):
        end = source()
        if end is not None and end > now:
            return end
    return None


def _default_margin():
    return parse_duration(os.environ.get(MARGIN_ENV)) or DEFAULT_MARGIN_S


def _default_min_slice():
    return parse_duration(os.environ.get(MIN_SLICE_ENV)) or DEFAULT_MIN_SLICE_S


def partition_is_preemptible():
    """Whether this job can be evicted before its time limit.

    On Hyak the checkpoint partitions -- ``ckpt``, ``ckpt-g2``, ``ckpt-all`` --
    run on other groups' idle nodes and are stopped without notice when an
    owner reclaims them. Every other partition runs to its ``--time`` and is
    not preempted.

    Read from the partition name because that is the only thing Slurm exposes
    that actually distinguishes the two. Off a scheduler entirely there is
    nothing to be preempted by, so the answer is False.
    """
    partition = (os.environ.get("SLURM_JOB_PARTITION") or "").lower()
    return "ckpt" in partition


def _default_slice_cap():
    """How often a job must hand its optimizer state back.

    Slicing exists to bound what an eviction destroys, and it is not free: a
    job boundary empties the evaluation cache, so the next job spends n+1 real
    evaluations re-establishing its simplex before it can improve on anything.
    On the SILK spec that is 16 evaluations at 113 s -- half an hour per
    resume. Paying that against a preemption that cannot happen is a large,
    silent waste, which is why this is not simply always on.

    So: cap on a preemptible partition, and run to the link's own deadline
    anywhere else. ``PROFILE_SLICE_CAP`` overrides in both directions, with
    ``off`` (or ``0``) meaning "never slice".
    """
    raw = os.environ.get(SLICE_CAP_ENV)
    if raw is not None and str(raw).strip():
        text = str(raw).strip().lower()
        if text in ("0", "off", "no", "none", "never", "inf", "infinite"):
            return float("inf")
        parsed = parse_duration(raw)
        if parsed:
            return parsed
    return DEFAULT_SLICE_CAP_S if partition_is_preemptible() else float("inf")


# ---------------------------------------------------------------------------
# Observed point durations
# ---------------------------------------------------------------------------

def _quantile(values, q):
    """Linear-interpolated quantile of an unsorted list, or None if empty."""
    vals = sorted(v for v in values if v is not None and math.isfinite(v))
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    pos = q * (len(vals) - 1)
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return vals[int(lo)]
    return vals[int(lo)] + (vals[int(hi)] - vals[int(lo)]) * (pos - lo)


class RunBudget:
    """What is left of this launch's wall clock, and what a point costs.

    Both halves are needed for the one question worth asking -- "is there time
    to start another point?" -- and neither is useful alone: a deadline with no
    cost model cannot tell a five-second point from a five-hour one, and a cost
    model with no deadline has nothing to compare against.

    An unset deadline makes every method permissive, so the same object can be
    constructed unconditionally and passed everywhere.
    """

    def __init__(self, deadline=None, margin_s=None, timing_path=None,
                 quantile=0.9, min_slice_s=None, slice_cap_s=None):
        self.deadline = deadline
        self.margin_s = _default_margin() if margin_s is None else float(margin_s)
        self.min_slice_s = (_default_min_slice() if min_slice_s is None
                            else float(min_slice_s))
        self.slice_cap_s = (_default_slice_cap() if slice_cap_s is None
                            else float(slice_cap_s))
        self.timing_path = timing_path
        self.quantile = float(quantile)
        self.durations, self.per_eval = self._load()
        self._n_since_save = 0
        # Set once a batch has been cut short, so the run can report honestly
        # that it stopped for time rather than because the work was done.
        self.stopped_early = False

    # -- persistence -------------------------------------------------------

    def _load(self):
        if not self.timing_path or not os.path.exists(self.timing_path):
            return [], []
        try:
            with open(self.timing_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            vals = [float(v) for v in data.get("durations", [])]
            per = [float(v) for v in data.get("per_eval", [])]
        except (OSError, ValueError, TypeError, AttributeError):
            # A truncated or hand-edited timing file is not worth failing a
            # multi-hour run over; an empty history just means this link
            # calibrates itself the way the first one did.
            return [], []

        def _clean(xs):
            return [v for v in xs
                    if math.isfinite(v) and v > 0][-_KEEP_DURATIONS:]

        return _clean(vals), _clean(per)

    def save(self):
        """Write the duration history, atomically.

        Written via a temporary file and ``os.replace`` because several array
        tasks may share the directory: last writer wins, which is fine for
        advisory data, but a half-written file read by the next link is not.
        """
        if not self.timing_path:
            return
        payload = {
            "durations": self.durations[-_KEEP_DURATIONS:],
            "per_eval": self.per_eval[-_KEEP_DURATIONS:],
            "quantile": self.quantile,
            "updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        tmp = f"{self.timing_path}.{os.getpid()}.tmp"
        try:
            os.makedirs(os.path.dirname(self.timing_path), exist_ok=True)
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(payload, fh)
            os.replace(tmp, self.timing_path)
        except OSError:
            try:
                os.unlink(tmp)
            except OSError:
                pass
        else:
            self._n_since_save = 0

    def record(self, seconds, n_evals=None):
        """Note how long one point took, and how many evaluations it bought."""
        try:
            v = float(seconds)
        except (TypeError, ValueError):
            return
        if not math.isfinite(v) or v <= 0:
            return
        self.durations.append(v)
        del self.durations[:-_KEEP_DURATIONS]
        try:
            n = int(n_evals)
        except (TypeError, ValueError):
            n = 0
        if n > 0:
            self.per_eval.append(v / n)
            del self.per_eval[:-_KEEP_DURATIONS]
        self._n_since_save += 1
        if self._n_since_save >= _SAVE_EVERY:
            self.save()

    def seconds_per_eval(self):
        """Measured cost of one objective evaluation, or None.

        Sent to the worker so the very first slice of a point can be sized to
        the time available instead of guessed at. It is the median rather than
        an upper quantile: this one is used to *size* work, not to decide
        whether to start it, and over-estimating here would cut every slice
        short and pay the restart overhead more often than necessary.
        """
        vals = sorted(v for v in self.per_eval if math.isfinite(v) and v > 0)
        if not vals:
            return None
        return vals[len(vals) // 2]

    # -- the actual question -----------------------------------------------

    @property
    def is_limited(self):
        return self.deadline is not None

    def remaining(self, now=None):
        """Seconds left before the deadline; ``inf`` when there is none."""
        if self.deadline is None:
            return float("inf")
        return self.deadline - (time.time() if now is None else now)

    def work_deadline(self):
        """The last moment this link may still be computing.

        Leaves the margin for trace assembly, plotting and the results write.
        None means no limit, and points run to convergence.
        """
        if self.deadline is None:
            return None
        return self.deadline - self.margin_s

    def job_deadline(self, min_needed_s=None, now=None):
        """When the job being started now must hand its state back.

        The earlier of "the link is ending" and "this job has had its slice".
        The second bound is what makes preemption survivable: a job that ends
        every ``slice_cap_s`` has been checkpointed that recently, so an
        eviction costs at most one slice per point in flight instead of
        everything since the point began.

        *min_needed_s* is how long the optimizer needs to reach a state it can
        hand back at all. The cap is raised to meet it, because a cap below it
        is worse than no cap: the job is stopped before any state exists, so it
        resumes from scratch and the slice is spent for nothing. A cap only
        helps if the work inside it can finish. The observed case was a
        30-minute default against a 16-parameter model at 116 s per evaluation,
        which needs about half an hour just to establish its simplex.

        The cap does not depend on the link's own deadline being known, and
        that is deliberate. ``SLURM_JOB_END_TIME`` is occasionally absent and
        ``scontrol`` occasionally does not answer, and when both fail
        :func:`resolve_deadline` returns None. Tying the cap to the deadline
        meant a job that could not read its end time also stopped slicing: no
        point ever handed its state back, nothing was checkpointed, and the
        first eviction discarded every hour of it. The two facts are separate
        -- "how long until the allocation ends" and "how often must state be
        saved" -- so they are now read separately.

        Off a scheduler ``slice_cap_s`` is already infinite, so a laptop run
        with no deadline still returns None and points run to convergence
        exactly as they always did.
        """
        end = self.work_deadline()
        cap = self.slice_cap_s
        if min_needed_s and min_needed_s > cap:
            cap = float(min_needed_s)
        if math.isinf(cap):
            return end
        now = time.time() if now is None else now
        sliced = now + cap
        return sliced if end is None else min(end, sliced)

    def estimate(self):
        """How long a point has been taking, or None.

        The upper quantile rather than the mean, because it is read as "how
        much longer might this run", not "what is typical". Used for the log
        and the progress estimate; since points can now be interrupted and
        resumed it no longer gates admission.
        """
        return _quantile(self.durations, self.quantile)

    def admits(self, now=None):
        """Whether there is room to start one more point.

        The rule here changed when points became interruptible, and the old one
        would now be actively harmful. It compared the time left against how
        long a point takes, and refused anything that would not fit -- correct
        when an unfinished point was a total loss, but catastrophic once a
        point can take forty hours: every four-hour link would look at a
        forty-hour estimate, admit nothing at all, and the profile would never
        advance.

        What matters instead is whether a slice is long enough to be worth
        starting. A resumed point re-evaluates its simplex before it makes any
        progress, so a slice shorter than that is spent entirely on overhead;
        ``min_slice_s`` is the floor below which the link stops handing out
        work and lets the margin do its job.
        """
        if self.deadline is None:
            return True
        return self.remaining(now) >= self.margin_s + self.min_slice_s

    def describe(self, effective_slice_s=None):
        """One line for the log, saying what the budget will actually do.

        Three separate numbers govern three separate things, and an earlier
        version of this line conflated them -- it reported ``min_slice_s`` as
        "points interrupted at N min remaining", which is not what that
        threshold does and left the slice cap, the mechanism that actually
        bounds preemption damage, unmentioned.

        *effective_slice_s* is the cap after being raised to fit a simplex,
        known only to the caller that has the jobs in hand.
        """
        if self.deadline is None:
            return "no wall-clock deadline; running until the work is done"

        est = self.estimate()
        est_txt = (f"points have been taking up to {est / 60.0:.0f} min"
                   if est is not None else "no timing history yet")

        if math.isinf(self.slice_cap_s):
            # Not a preemptible partition, or slicing switched off. Points run
            # to convergence and are interrupted only by the link's own
            # deadline, which saves the n+1 evaluations a resume would spend
            # rebuilding its simplex.
            hand_back = ("running points are interrupted only at the deadline "
                         "(no slice cap: nothing here preempts)")
        else:
            cap = max(self.slice_cap_s, effective_slice_s or 0.0)
            raised = (" (raised to fit a simplex)"
                      if cap > self.slice_cap_s else "")
            hand_back = (f"running points hand their state back every "
                         f"{cap / 60.0:.0f} min{raised}")

        return (f"{self.remaining() / 60.0:.0f} min left, "
                f"{self.margin_s / 60.0:.0f} min of it reserved for the "
                f"write-up; no new point starts with under "
                f"{self.min_slice_s / 60.0:.0f} min to go; {hand_back}; "
                f"{est_txt}")

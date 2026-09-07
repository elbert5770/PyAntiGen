"""Append-only checkpoint store for profile-likelihood runs.

A full profile on a QSP model is hours of compute. Without checkpointing, a
crash, a reboot, or a decision to move the job to a bigger machine throws all of
it away -- which is why "massively parallel" and "checkpointed" are the same
requirement, not two.

Format: one JSON object per line, one file per parameter, under

    results/<MODEL>/profiles/<run_id>/<param>.jsonl

Append-only JSONL is chosen deliberately over a single JSON document or a
database:

* A half-written line is detectable and discardable; a half-written JSON
  document is not, so a kill during a write cannot corrupt earlier results.
* One file per parameter means the 2k concurrent writers never contend.
* Concatenating directories from two machines merges two partial runs, so a job
  can be split across boxes and reassembled.

Each record carries the hashes of the spec and the model, so a resume against a
changed model or a changed optimization spec is detected rather than silently
mixing incompatible points.
"""

import hashlib
import json
import os
import re
import time
from datetime import datetime

import numpy as np


def _safe_name(s):
    """Filesystem-safe version of a parameter name."""
    out = re.sub(r"[^A-Za-z0-9_.-]", "_", str(s))
    return out or "param"


# Files written as "<name>.<pid>.tmp" and renamed into place. A process killed
# between the write and the rename leaves one behind, and on a preemptible
# queue that happens routinely.
_TEMP_SUFFIX = ".tmp"

# Only temp files older than this are swept. A younger one may belong to a
# process that is still running -- another array task, or this one -- and
# deleting it would make that process's rename fail for no reason.
_TEMP_MAX_AGE_S = 3600.0


def sweep_stale_temp_files(directory, max_age_s=_TEMP_MAX_AGE_S):
    """Delete abandoned temp files in *directory*; return how many went.

    These are litter rather than corruption: the atomic-rename pattern that
    creates them is exactly what stops a half-written file from ever being
    read. But one accumulates per kill, and a run that is preempted a hundred
    times leaves a hundred of them beside the results.

    Never raises. A sweep that cannot run is not a reason to fail a run that
    would otherwise work.
    """
    removed = 0
    try:
        names = os.listdir(directory)
    except OSError:
        return 0

    now = time.time()
    for name in names:
        if not name.endswith(_TEMP_SUFFIX):
            continue
        path = os.path.join(directory, name)
        try:
            if now - os.path.getmtime(path) < max_age_s:
                continue
            os.unlink(path)
            removed += 1
        except OSError:
            continue
    return removed


def record_is_better(rec, prev):
    """Whether *rec* should replace *prev* as the value stored at a grid point.

    Lower NLL wins. Every profile point is an *upper bound* on the true profile
    -- it is a real evaluation of the nuisance minimum, just not necessarily a
    converged one -- so when two records exist for the same fixed value both are
    valid and the lower one is strictly closer to the truth. Taking the minimum
    is what makes the warm-started continuation pass safe to run on top of an
    existing checkpoint: it can only lower the curve, never raise it.

    Ties go to the later record so that a re-run which only updates bookkeeping
    (a warm pass that did not improve a point, but must still mark it as visited
    so a resume does not redo it) is not discarded.

    A record whose NLL is missing or non-finite -- a failed simulation, a
    sentinel -- never displaces a usable one. Getting this backwards would let a
    single failed re-evaluation erase a good point that cost minutes to compute.
    """
    def _nll(r):
        try:
            v = float(r.get("nll"))
        except (TypeError, ValueError):
            return None
        return v if np.isfinite(v) else None

    a, b = _nll(rec), _nll(prev)
    if a is None:
        # Junk never wins over a real value; two junk records tie to the later.
        return b is None
    if b is None:
        return True
    return a <= b


def _round_sig(x, sig=6):
    """Round to *sig* significant figures (0 and non-finite pass through)."""
    x = float(x)
    if x == 0.0 or not np.isfinite(x):
        return 0.0 if x == 0.0 else None
    return round(x, sig - 1 - int(np.floor(np.log10(abs(x)))))


def solver_fingerprint(replicates):
    """A hash of how every replicate will actually be integrated, or None.

    The solver settings are part of the objective, not decoration. Output
    density feeds ``np.interp`` onto the data times, and the tolerances and step
    budget decide what the integrator returns, so changing any of them moves the
    NLL -- on the SILK spec, dropping the labelling window from 200,000 output
    points to 10,000 shifts it by about 1.5e-3 nats.

    That is small against the 1.9207 threshold and large against nothing at all,
    which is exactly the situation a fingerprint is for: points computed under
    two different settings are two different curves, and a checkpoint directory
    that mixes them is quietly wrong. Nothing else in the fingerprint sees these
    numbers -- they live in functions on the replicates, not in the model text
    or the spec -- so before this they could change under a resumed run with no
    signal at all.

    Best-effort by design. A settings function that will not run here would
    otherwise take down a profile that was going to work, so the failure is
    recorded as an unknown marker for that replicate rather than raised. The
    marker still participates in the hash, so "we could not read this" is itself
    a stable, distinguishable state.
    """
    if not replicates:
        return None

    def _norm(value):
        if isinstance(value, dict):
            return {str(k): _norm(v) for k, v in sorted(value.items())}
        if isinstance(value, (list, tuple)):
            return [_norm(v) for v in value]
        if isinstance(value, (int, float, np.floating, np.integer)):
            # Rounded so float noise in a computed block boundary -- an age in
            # hours, say -- cannot invalidate a directory on its own.
            return _round_sig(float(value), 9)
        if isinstance(value, (str, bool)) or value is None:
            return value
        return f"<{type(value).__name__}>"

    entries = {}

    # Engine-level constants that decide what the integrator returns. They are
    # not in any settings dict -- they live in Engine.Simulate -- but a run with
    # a different tolerance floor or dust threshold is integrating a different
    # problem, and its points do not belong in the same directory. Read lazily
    # so this module stays importable without the simulation stack.
    try:
        from Engine.Simulate import _DUST_THRESHOLD, _MIN_ABSOLUTE_TOLERANCE
        entries["__engine__"] = {
            "min_absolute_tolerance": float(_MIN_ABSOLUTE_TOLERANCE),
            "dust_threshold": float(_DUST_THRESHOLD),
        }
    except Exception:
        entries["__engine__"] = "<unreadable>"

    for name, rep in sorted(replicates.items()):
        fn = rep.get("Solver_settings") if hasattr(rep, "get") else None
        if fn is None:
            entries[str(name)] = "<no-solver-settings>"
            continue
        try:
            settings = fn(rep)
            entries[str(name)] = _norm(
                {k: v for k, v in settings.items() if k != "event_times"}
            )
        except Exception as exc:      # noqa: BLE001 - see the docstring
            entries[str(name)] = f"<unreadable:{type(exc).__name__}>"

    blob = json.dumps(entries, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def spec_fingerprint(param_names, x_opt, groups, scales, model_text,
                     fixed_sigmas=None, solver_hash=None):
    """Stable hashes identifying what a set of profile points belongs to.

    Points computed against a different model, parameter set, optimum, or set of
    solver settings must not be silently reused, so all of them are hashed and
    stored on every record.

    The optimum is rounded to six significant figures first. Profile points are
    centred on it and their dNLL is measured relative to it, so a genuinely
    different optimum must invalidate them -- but a re-fit that lands 1e-9 away
    is the *same* optimum, and letting float noise discard hours of completed
    work would make resuming useless in practice.

    *solver_hash* comes from :func:`solver_fingerprint` and is passed by the
    profile driver, which is the only caller with the replicates in hand. It is
    optional so that callers without them -- tests, and anything profiling a
    bare function -- keep working; when it is absent the hash is the same as it
    was before solver settings were tracked.
    """
    model_hash = hashlib.sha256((model_text or "").encode("utf-8")).hexdigest()[:16]
    spec_blob = json.dumps(
        {
            "param_names": list(param_names),
            "x_opt": [_round_sig(v) for v in np.atleast_1d(x_opt)],
            "scales": list(scales),
            "groups": sorted(groups.keys()) if hasattr(groups, "keys") else None,
            # Bumped when the meaning of a stored dNLL changes.
            #   v1 carried the objective's group averaging and weights.
            #   v2 was the summed unweighted NLL with sigmas frozen at the fit
            #      optimum -- a different function from the one the fit
            #      minimized, which is what let dNLL go negative.
            #   v3 is the concentrated Gaussian likelihood, with each block's
            #      sigma profiled out analytically. It is the same function the
            #      fit minimizes, so its dNLL is on a different scale again and
            #      v1/v2 points must not be resumed into a v3 run.
            "likelihood_convention": "v3-concentrated-gaussian",
            # Under v3 sigma is profiled out per evaluation rather than frozen,
            # so these no longer enter dNLL. They are still hashed because they
            # are a compact fingerprint of the residuals at the optimum, which
            # does change whenever the fit lands somewhere else.
            "sigmas": sorted(
                (str(k), round(float(v), 12))
                for k, v in (fixed_sigmas or {}).items()
            ),
            # Absent for callers with no replicates to read, which keeps their
            # hash identical to what it was before solver settings were
            # tracked; present, it makes any change to how the model is
            # integrated start a new directory. See solver_fingerprint.
            **({"solver": str(solver_hash)} if solver_hash else {}),
        },
        sort_keys=True,
    )
    spec_hash = hashlib.sha256(spec_blob.encode("utf-8")).hexdigest()[:16]
    return model_hash, spec_hash


def default_run_id(tag, model_hash, spec_hash):
    """Directory name for a run's checkpoints.

    Deliberately excludes any timestamp: a run_id that changes every launch can
    never resume, which defeats the entire mechanism. Identity comes from *what
    is being profiled*, so relaunching the same problem finds its own results
    and a different problem gets a different directory.
    """
    return f"{_safe_name(tag)}_{model_hash[:8]}_{spec_hash[:8]}"


class ProfileCheckpoint:
    """Reads and appends profile points for one run."""

    def __init__(self, root, run_id, model_hash, spec_hash, enabled=True):
        self.enabled = bool(enabled)
        self.run_id = run_id
        self.model_hash = model_hash
        self.spec_hash = spec_hash
        self.dir = os.path.join(root, "profiles", run_id) if root else None
        self._handles = {}
        self.n_loaded = 0
        self.n_skipped_stale = 0
        self.n_temp_swept = 0
        if self.enabled and self.dir:
            os.makedirs(self.dir, exist_ok=True)
            self.n_temp_swept = sweep_stale_temp_files(self.dir)

    # -- paths -------------------------------------------------------------

    def path_for(self, param_name):
        return os.path.join(self.dir, f"{_safe_name(param_name)}.jsonl")

    # -- reading -----------------------------------------------------------

    def load(self, param_names):
        """Return {param_name: {rounded_x_fixed: record}} for completed points.

        Records whose model or spec hash does not match the current run are
        counted and ignored -- resuming onto a changed model must not silently
        blend old points with new ones.

        Where a fixed value appears more than once -- which is what the
        warm-started pass produces -- the *lowest* NLL is kept rather than the
        last one written. Relying on write order would mean a later, worse
        evaluation silently replaced a better earlier one on the next resume.
        """
        found = {name: {} for name in param_names}
        if not (self.enabled and self.dir and os.path.isdir(self.dir)):
            return found

        for name in param_names:
            path = self.path_for(name)
            if not os.path.exists(path):
                continue
            with open(path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        # Truncated final line from a killed run: expected, skip.
                        continue
                    if (rec.get("model_hash") != self.model_hash
                            or rec.get("spec_hash") != self.spec_hash):
                        self.n_skipped_stale += 1
                        continue
                    if rec.get("status") != "ok":
                        continue
                    key = self._key(rec.get("x_fixed"))
                    if key is None:
                        continue
                    prev = found[name].get(key)
                    if prev is not None and not record_is_better(rec, prev):
                        continue
                    found[name][key] = rec

        self.n_loaded = sum(len(v) for v in found.values())
        return found

    @staticmethod
    def _key(x):
        """Grid points are matched on value, rounded so float noise cannot
        create a near-duplicate that gets recomputed every resume."""
        try:
            return round(float(x), 12)
        except (TypeError, ValueError):
            return None

    # -- writing -----------------------------------------------------------

    def append(self, record):
        if not (self.enabled and self.dir):
            return
        name = record.get("param_name", "param")
        rec = dict(record)
        rec.setdefault("timestamp", datetime.now().isoformat(timespec="seconds"))
        rec["model_hash"] = self.model_hash
        rec["spec_hash"] = self.spec_hash
        fh = self._handles.get(name)
        if fh is None:
            fh = open(self.path_for(name), "a", encoding="utf-8")
            self._handles[name] = fh
        json.dump(_jsonable(rec), fh)
        fh.write("\n")
        # Flush per record: the value of a checkpoint is entirely in surviving
        # an abrupt kill, which buffering would defeat.
        fh.flush()
        os.fsync(fh.fileno())

    def close(self):
        for fh in self._handles.values():
            try:
                fh.close()
            except Exception:
                pass
        self._handles.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False


def _jsonable(obj):
    """Convert numpy scalars/arrays so json.dump accepts them."""
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return v if np.isfinite(v) else None
    if isinstance(obj, float):
        return obj if np.isfinite(obj) else None
    return obj

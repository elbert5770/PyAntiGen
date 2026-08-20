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
from datetime import datetime

import numpy as np


def _safe_name(s):
    """Filesystem-safe version of a parameter name."""
    out = re.sub(r"[^A-Za-z0-9_.-]", "_", str(s))
    return out or "param"


def _round_sig(x, sig=6):
    """Round to *sig* significant figures (0 and non-finite pass through)."""
    x = float(x)
    if x == 0.0 or not np.isfinite(x):
        return 0.0 if x == 0.0 else None
    return round(x, sig - 1 - int(np.floor(np.log10(abs(x)))))


def spec_fingerprint(param_names, x_opt, groups, scales, model_text,
                     fixed_sigmas=None):
    """Stable hashes identifying what a set of profile points belongs to.

    Points computed against a different model, parameter set, or optimum must
    not be silently reused, so all three are hashed and stored on every record.

    The optimum is rounded to six significant figures first. Profile points are
    centred on it and their dNLL is measured relative to it, so a genuinely
    different optimum must invalidate them -- but a re-fit that lands 1e-9 away
    is the *same* optimum, and letting float noise discard hours of completed
    work would make resuming useless in practice.
    """
    model_hash = hashlib.sha256((model_text or "").encode("utf-8")).hexdigest()[:16]
    spec_blob = json.dumps(
        {
            "param_names": list(param_names),
            "x_opt": [_round_sig(v) for v in np.atleast_1d(x_opt)],
            "scales": list(scales),
            "groups": sorted(groups.keys()) if hasattr(groups, "keys") else None,
            # Bumped when the meaning of a stored dNLL changes. v2 = the joint
            # log-likelihood (plain sum, unit weights); v1 carried the
            # objective's group averaging and weights, so its dNLL values are on
            # a different scale and must not be resumed into a v2 run.
            "likelihood_convention": "v2-summed-unweighted",
            # dNLL is computed against these sigmas, so a point stored under
            # different ones is not comparable. Including them means a corrected
            # sigma estimate invalidates stale checkpoints automatically instead
            # of silently resuming numbers from a previous convention.
            "sigmas": sorted(
                (str(k), round(float(v), 12))
                for k, v in (fixed_sigmas or {}).items()
            ),
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
        if self.enabled and self.dir:
            os.makedirs(self.dir, exist_ok=True)

    # -- paths -------------------------------------------------------------

    def path_for(self, param_name):
        return os.path.join(self.dir, f"{_safe_name(param_name)}.jsonl")

    # -- reading -----------------------------------------------------------

    def load(self, param_names):
        """Return {param_name: {rounded_x_fixed: record}} for completed points.

        Records whose model or spec hash does not match the current run are
        counted and ignored -- resuming onto a changed model must not silently
        blend old points with new ones.
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
                    if key is not None:
                        found[name][key] = rec
                        self.n_loaded += 1
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

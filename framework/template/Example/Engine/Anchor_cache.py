"""Hessian-derived quantities, cached against the fit they belong to.

The Wald statistics cost ``1 + 2k + 2k(k-1)`` objective evaluations -- 513 on
the 16-parameter SILK APP spec -- and they are recomputed from scratch on every
launch. That was tolerable when an evaluation was assumed to cost seconds. It
is not: the measured cost on that spec is 116 s, so across 39 workers the
Hessian alone is about 25 minutes, and it is charged again on every link of a
chain that may run to a hundred links.

On a preemptible partition the number matters for a second and sharper reason.
Nothing is written until a profile point finishes, so a link only makes
progress if the node survives setup *plus* one slice. Cutting 25 minutes off
setup lowers that threshold directly, which is the difference between a
short-lived node contributing something and contributing nothing at all.

Caching is safe here because the Hessian is a pure function of things the run
already fingerprints: the model, the optimization spec, the parameter scaling
and the optimum it is taken at. A change in any of them produces a different
key and a miss, so a stale Hessian cannot be silently reused -- the failure
mode that would matter, since an SE that does not belong to this fit would set
the profile's whole grid in the wrong place.
"""

import hashlib
import json
import os
from datetime import datetime

import numpy as np

from Engine.Profile_checkpoint import sweep_stale_temp_files

# Everything _attach_wald_stats puts in out["stats"]. Cached and restored as a
# set: a partial restore would leave the CI from one fit beside the SE of
# another.
WALD_FIELDS = ("wald_cov", "wald_se", "wald_se_opt", "wald_ci",
               "wald_correlation")

# Bumped when the set or meaning of the cached fields changes, so old files
# miss rather than being misread.
#   v1 held only the linear "wald_se".
#   v2 adds "wald_se_opt", the SE in the optimizer's own space, which is what
#      the profile and slice grids are placed with. A v1 file restored into a
#      v2 run would leave that key absent and silently drop every grid back to
#      the range_factor fallback.
_FORMAT = "wald-v2"

# Without this the block is not worth restoring: the profile grid is placed
# from it, and "no SE at all" triggers a different, deliberate fallback than
# "an SE in the wrong units".
_REQUIRED = ("wald_se", "wald_se_opt")


def _encode(obj):
    """Arrays to nested lists, with non-finite values as null.

    Non-finite entries are meaningful here -- an SE of nan is how "this
    direction is flat, there is no usable standard error" is reported -- but
    they are not portable JSON. They come back as nan, which is what every
    consumer tests for with ``np.isfinite``.
    """
    if obj is None:
        return None
    arr = np.asarray(obj, dtype=float)
    out = arr.tolist()

    def _clean(v):
        if isinstance(v, list):
            return [_clean(x) for x in v]
        return v if np.isfinite(v) else None

    return _clean(out)


def _decode(obj):
    """The inverse: nulls back to nan, lists back to arrays."""
    if obj is None:
        return None

    def _fill(v):
        if isinstance(v, list):
            return [_fill(x) for x in v]
        return float("nan") if v is None else float(v)

    return np.asarray(_fill(obj), dtype=float)


def bounds_fingerprint(bounds):
    """Hash of the declared bounds.

    Separate from the profile's own spec hash on purpose. Bounds change the
    Wald *interval* (it is clipped to them) without changing the model or the
    optimum, so they belong in this cache's key -- but adding them to
    ``spec_fingerprint`` would change every existing profile directory name and
    orphan work already done.
    """
    if bounds is None:
        return "none"
    try:
        blob = json.dumps(
            [None if b is None else [None if v is None else round(float(v), 12)
                                     for v in b]
             for b in bounds],
            sort_keys=True,
        )
    except (TypeError, ValueError):
        return "unhashable"
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


class AnchorCache:
    """Reads and writes the Wald block for one fit."""

    def __init__(self, root, run_id, model_hash, spec_hash, bounds_hash,
                 n_params, enabled=True):
        self.enabled = bool(enabled and root)
        self.model_hash = model_hash
        self.spec_hash = spec_hash
        self.bounds_hash = bounds_hash
        self.n_params = int(n_params)
        self.dir = os.path.join(root, "profiles", run_id) if root else None
        if self.enabled and self.dir:
            try:
                os.makedirs(self.dir, exist_ok=True)
                # A kill between writing a temp file and renaming it leaves the
                # temp behind; this directory is where they collect.
                sweep_stale_temp_files(self.dir)
            except OSError:
                self.enabled = False

    @property
    def path(self):
        return os.path.join(self.dir, "anchor.json") if self.dir else None

    def load(self):
        """The cached Wald block, or None on any miss.

        Every failure is a miss rather than an error: a corrupt, truncated or
        stale file must cost the 25 minutes of recomputation it was meant to
        save, never the correctness of the run.
        """
        if not (self.enabled and self.path and os.path.exists(self.path)):
            return None
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, ValueError):
            return None

        if (data.get("format") != _FORMAT
                or data.get("model_hash") != self.model_hash
                or data.get("spec_hash") != self.spec_hash
                or data.get("bounds_hash") != self.bounds_hash
                or int(data.get("n_params") or -1) != self.n_params):
            return None

        stats = {}
        for field in WALD_FIELDS:
            if field in data:
                stats[field] = _decode(data[field])

        for field in _REQUIRED:
            arr = stats.get(field)
            if arr is None or np.asarray(arr).shape != (self.n_params,):
                return None
        return stats

    def save(self, stats):
        """Write the Wald block, atomically."""
        if not (self.enabled and self.path):
            return
        payload = {
            "format": _FORMAT,
            "model_hash": self.model_hash,
            "spec_hash": self.spec_hash,
            "bounds_hash": self.bounds_hash,
            "n_params": self.n_params,
            "saved": datetime.now().isoformat(timespec="seconds"),
        }
        for field in WALD_FIELDS:
            if stats.get(field) is not None:
                payload[field] = _encode(stats[field])

        tmp = f"{self.path}.{os.getpid()}.tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(payload, fh)
            os.replace(tmp, self.path)
        except (OSError, TypeError, ValueError):
            try:
                os.unlink(tmp)
            except OSError:
                pass

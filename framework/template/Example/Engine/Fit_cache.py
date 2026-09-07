"""The fitted optimum, saved the moment the fit ends and reused on relaunch.

The profile is anchored on the fit that precedes it: every point is centred on
``res.x`` and every dNLL is measured against ``nll(res.x)``. Inside one process
that anchor is simply handed from the fit to the profile. The problem is the
process does not survive. On a preemptible partition the whole script is
restarted every 4-5 hours, and a restart with ``fit_mode="optimize"`` used to
re-run the fit from the spec's x0 -- hours of serial Nelder-Mead charged again
on every link, before a single profile point could start.

That refit is also what put the profile's checkpoint at risk. The checkpoint
directory is named from a hash that includes the optimum, so a refit that lands
even six significant figures away from the last one opens a *new* directory and
the profile starts over. Nelder-Mead from an identical start is deterministic,
so in practice the refit usually lands on the same point; but "usually" is not
a property a five-day run should rest on.

So the fit is cached. The key is the *problem* -- model text, parameter names,
the spec's x0, bounds, scaling, groups, method, the optimizer's own settings,
the multi-start configuration, the solver settings and the data -- and the
value is where the fit landed. A relaunch that poses the same problem gets the
same answer without paying for it; any change in the problem is a miss, and
the fit runs as it always has.

Two records live under one key:

* ``complete`` -- a fit that returned. Reused as the optimum outright.
* ``partial`` -- the best point seen so far by a fit that has not returned.
  Written on a throttle from inside the objective, so that a fit killed at
  hour three restarts from hour three's best point rather than from x0. The
  simplex itself is not saved; Nelder-Mead rebuilds one around the point,
  which costs n+1 evaluations rather than the hours already spent.

Layout::

    results/<MODEL>/fits/<tag>_<model_hash>_<fit_hash>.json

beside the ``profiles/`` tree the checkpoint uses, and following the same
write-temp-then-rename discipline: a kill mid-write leaves a temp file, never
a truncated record.
"""

import hashlib
import json
import os
import time
from datetime import datetime

import numpy as np

from Engine.Profile_checkpoint import _safe_name, sweep_stale_temp_files

# Bumped when the meaning of a stored record changes, so old files miss.
_FORMAT = "fit-v1"

# How often the partial record may be rewritten. An evaluation costs tens of
# seconds on the specs this exists for, so once a minute is at most one write
# per few evaluations and negligible against them; on a toy problem that runs
# thousands of evaluations a second it stops the objective becoming a disk
# benchmark.
PARTIAL_SAVE_INTERVAL_S = 60.0


def _json_default(obj):
    """Make optimizer settings hashable: arrays to lists, the rest to repr."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return repr(obj)


def data_fingerprint(models):
    """Hash of every data table the objective reads.

    Nothing else in the fingerprint sees the data: the model text and the spec
    say which files are read, not what is in them. Editing a data file changes
    the objective without changing anything a hash of the code would notice,
    and a cached optimum for the *old* data would then anchor a profile of the
    new one. The tables are already loaded on every replicate by the time the
    fit could be looked up, so hashing their contents is cheap.

    Every value that cannot be hashed contributes its type name only. That is a
    weaker key, not a broken one, and it is deterministic, which is what makes
    a stale record a miss rather than a wrong hit.
    """
    h = hashlib.sha256()

    def _feed(v):
        if v is None:
            h.update(b"None")
        elif hasattr(v, "to_csv"):
            # pandas: the CSV text is a stable, complete rendering.
            h.update(v.to_csv(index=True).encode("utf-8"))
        elif isinstance(v, np.ndarray):
            h.update(repr(v.shape).encode("utf-8"))
            h.update(np.ascontiguousarray(v).tobytes())
        elif isinstance(v, dict):
            for k in sorted(v, key=str):
                h.update(str(k).encode("utf-8"))
                _feed(v[k])
        elif isinstance(v, (list, tuple)):
            for item in v:
                _feed(item)
        elif isinstance(v, (str, bytes, int, float, bool, np.generic)):
            h.update(repr(v).encode("utf-8"))
        else:
            h.update(type(v).__name__.encode("utf-8"))

    for sim_name in sorted(models or {}, key=str):
        h.update(str(sim_name).encode("utf-8"))
        try:
            _feed((models[sim_name] or {}).get("df_dict"))
        except Exception:
            h.update(b"unhashable")
    return h.hexdigest()[:16]


def fit_fingerprint(param_names, x0_lin, bounds_lin, scales, groups, model_text,
                    method, optimizer_kwargs, n_starts=1, start_seed=None,
                    search_decades=None, solver_hash=None, data_hash=None):
    """Hashes identifying a fit *problem*, before it has been solved.

    Deliberately distinct from :func:`Engine.Profile_checkpoint.spec_fingerprint`,
    which hashes the *answer* (the optimum) and so cannot be used to look the
    answer up. Everything that decides where Nelder-Mead lands from here is
    included; nothing that only decides what is done with the result (the
    profile's own settings, the diagnostics requested) is.

    x0 is in the key on purpose. Two fits from different starting points are
    two different fits -- Nelder-Mead is local -- and copying fitted values
    into the registry, which is how this project moves a fit forward, changes
    x0 and so asks for a fresh fit from there. That is the right behaviour;
    the cache is for the *same* launch repeated, not for skipping fits.
    """
    model_hash = hashlib.sha256((model_text or "").encode("utf-8")).hexdigest()[:16]

    def _r(v):
        try:
            v = float(v)
        except (TypeError, ValueError):
            return None
        return round(v, 12) if np.isfinite(v) else None

    kwargs = dict(optimizer_kwargs or {})
    # Profile-only keys configure what happens after the fit, not the fit.
    for k in ("profile_method", "profile_optimizer_kwargs", "profile_grid",
              "profile_without_opt"):
        kwargs.pop(k, None)

    blob = json.dumps(
        {
            "format": _FORMAT,
            "param_names": list(param_names),
            "x0": [_r(v) for v in np.atleast_1d(x0_lin)],
            "bounds": None if bounds_lin is None else [
                None if b is None else [_r(v) for v in b] for b in bounds_lin
            ],
            "scales": list(scales),
            "groups": sorted(groups.keys()) if hasattr(groups, "keys") else None,
            "method": str(method).lower(),
            "optimizer_kwargs": kwargs,
            "n_starts": int(n_starts or 1),
            "start_seed": start_seed,
            "search_decades": search_decades,
            "solver": solver_hash,
            "data": data_hash,
        },
        sort_keys=True, default=_json_default,
    )
    fit_hash = hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]
    return model_hash, fit_hash


class FitCache:
    """Reads and writes the fitted optimum for one fit problem."""

    def __init__(self, root, tag, model_hash, fit_hash, n_params, enabled=True):
        self.enabled = bool(enabled and root)
        self.model_hash = model_hash
        self.fit_hash = fit_hash
        self.n_params = int(n_params)
        self.dir = os.path.join(root, "fits") if root else None
        self._name = f"{_safe_name(tag)}_{model_hash[:8]}_{fit_hash[:8]}.json"
        self._last_partial_save = 0.0
        if self.enabled and self.dir:
            try:
                os.makedirs(self.dir, exist_ok=True)
                sweep_stale_temp_files(self.dir)
            except OSError:
                self.enabled = False

    @property
    def path(self):
        return os.path.join(self.dir, self._name) if self.dir else None

    # -- reading -----------------------------------------------------------

    def _read(self):
        """The record on disk, or None on any miss.

        Every failure is a miss rather than an error: a corrupt or stale file
        must cost the refit it was meant to save, never the run.
        """
        if not (self.enabled and self.path and os.path.exists(self.path)):
            return None
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, ValueError):
            return None
        if (not isinstance(data, dict)
                or data.get("format") != _FORMAT
                or data.get("model_hash") != self.model_hash
                or data.get("fit_hash") != self.fit_hash
                or int(data.get("n_params") or -1) != self.n_params):
            return None
        return data

    def _vector(self, block):
        if not isinstance(block, dict):
            return None
        try:
            x = np.asarray(block.get("x_lin"), dtype=float)
        except (TypeError, ValueError):
            return None
        if x.shape != (self.n_params,) or not np.all(np.isfinite(x)):
            return None
        return x

    def load_complete(self):
        """A finished fit for this problem: ``{"x_lin", "fun", ...}`` or None."""
        data = self._read()
        if data is None:
            return None
        block = data.get("complete")
        x = self._vector(block)
        if x is None:
            return None
        out = dict(block)
        out["x_lin"] = x
        out["path"] = self.path
        return out

    def load_partial(self):
        """The best point of an unfinished fit, or None."""
        data = self._read()
        if data is None:
            return None
        block = data.get("partial")
        x = self._vector(block)
        if x is None:
            return None
        out = dict(block)
        out["x_lin"] = x
        out["path"] = self.path
        return out

    # -- writing -----------------------------------------------------------

    def _write(self, data):
        if not (self.enabled and self.path):
            return False
        data = dict(data)
        data.update({
            "format": _FORMAT,
            "model_hash": self.model_hash,
            "fit_hash": self.fit_hash,
            "n_params": self.n_params,
        })
        tmp = f"{self.path}.{os.getpid()}.tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2)
            os.replace(tmp, self.path)
            return True
        except (OSError, TypeError, ValueError):
            try:
                os.unlink(tmp)
            except OSError:
                pass
            return False

    @staticmethod
    def _block(x_lin, fun, **extra):
        block = {
            "x_lin": [float(v) for v in np.atleast_1d(x_lin)],
            "fun": None if fun is None or not np.isfinite(fun) else float(fun),
            "saved": datetime.now().isoformat(timespec="seconds"),
        }
        block.update(extra)
        return block

    def save_complete(self, x_lin, fun, param_names=None, **extra):
        """Record a finished fit. Clears any partial record: it is superseded."""
        if not self.enabled:
            return False
        block = self._block(x_lin, fun, **extra)
        if param_names is not None:
            # Registry-shaped, so the file is readable by a person too.
            block["parameters"] = {
                str(n): float(v) for n, v in zip(param_names, np.atleast_1d(x_lin))
            }
        data = self._read() or {}
        data.pop("partial", None)
        data["complete"] = block
        return self._write(data)

    def save_partial(self, x_lin, fun, n_evals=None, force=False):
        """Record the best point so far, at most once per interval.

        Never overwrites a ``complete`` record: a finished fit is strictly
        better information than any point along the way to it, and a partial
        from a *later* process (a relaunch that found no complete record yet,
        then raced one that did) must not demote it.
        """
        if not self.enabled:
            return False
        now = time.time()
        if not force and now - self._last_partial_save < PARTIAL_SAVE_INTERVAL_S:
            return False
        data = self._read() or {}
        if data.get("complete") is not None:
            return False
        data["partial"] = self._block(x_lin, fun, n_evals=n_evals)
        ok = self._write(data)
        if ok:
            self._last_partial_save = now
        return ok

"""Saving, loading and comparing the calibrated noise floors, the sigmas of a fixed-sigma run.

In ``sigma_floor="fixed"`` mode every data-floored block is scored with its LOESS floor as a known sigma
(:mod:`pyantigen.engine.Noise_floor`). The floor is calibrated once, against the model's shape at the first
point evaluated, and then does not move, so the objective is one function: ``f = const + 0.5 r.r``. This
module makes that calibration a file you can name:

* :func:`save` writes the floors and returns a short sha256 of the file. Stamp it on every result, so two
  values of f are compared only if they were scored with the same sigmas.
* :func:`load` seeds this process's cache from a file (workers are seeded by the evaluator) and refuses a
  file whose hash is not the one expected.
* :func:`compare` says how far one calibration is from another. The floors come from the model's shape at
  one parameter vector, so after a fit moves far from x0 they can be recalibrated at the optimum and the
  fit repeated; ``compare`` is the test of whether that mattered (:func:`stable`).

The file is a pickle of ``Noise_floor.export_cache()``, the format ``tools/pypesto_make_floors.py`` already
writes, so existing floors files and their hashes are unchanged. Load only files you wrote.
"""
import hashlib
import math
import os
import pickle

from pyantigen.engine import Noise_floor


def file_sha(path):
    """First 16 hex digits of the sha256 of the file's bytes (the stamp results carry)."""
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()[:16]


def calibrate(evaluate):
    """Forget every floor, call ``evaluate()`` (one evaluation at the parameter vector to calibrate at), and
    return the snapshot of the floors that evaluation created."""
    Noise_floor.clear_cache()
    evaluate()
    return Noise_floor.export_cache()


def save(snapshot, path):
    """Write *snapshot* to *path* and return its stamp (see :func:`file_sha`)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(snapshot, fh)
    return file_sha(path)


def load(path, expect_sha=None):
    """Seed this process's floor cache from *path*; returns ``(snapshot, stamp)``.

    Raises ``ValueError`` if *expect_sha* is given and differs: a result scored with other floors is a result
    of a different function.
    """
    stamp = file_sha(path)
    if expect_sha is not None and stamp != expect_sha:
        raise ValueError(f"floors file {path} has stamp {stamp}, expected {expect_sha}")
    with open(path, "rb") as fh:
        snapshot = pickle.load(fh)
    Noise_floor.seed_cache(snapshot)
    return snapshot, stamp


def compare(old, new):
    """How each floor moved between two snapshots: a list of dicts, largest change first.

    Each row has ``key`` (the observable's cache key), ``sigma_old``, ``sigma_new``, ``log_ratio``
    (natural log of new/old), and ``span_old``/``span_new``. Floors present in only one snapshot, or missing
    in either (too few points), are listed with ``log_ratio`` None.
    """
    rows = []
    for k in sorted(set(old) | set(new), key=repr):
        a, b = old.get(k), new.get(k)
        row = {"key": k[0] if isinstance(k, tuple) else k,
               "sigma_old": getattr(a, "sigma", None), "sigma_new": getattr(b, "sigma", None),
               "span_old": getattr(a, "span", None), "span_new": getattr(b, "span", None), "log_ratio": None}
        if a is not None and b is not None and a.sigma > 0 and b.sigma > 0:
            row["log_ratio"] = math.log(b.sigma / a.sigma)
        rows.append(row)
    rows.sort(key=lambda r: -abs(r["log_ratio"]) if r["log_ratio"] is not None else 1.0)
    return rows


def stable(rows, tol=0.05):
    """True if no floor moved by more than *tol* (a fraction: 0.05 is 5%) and none appeared or vanished."""
    for r in rows:
        if r["log_ratio"] is None:
            if r["sigma_old"] is not None or r["sigma_new"] is not None:
                return False
            continue
        if abs(math.expm1(r["log_ratio"])) > tol:
            return False
    return True

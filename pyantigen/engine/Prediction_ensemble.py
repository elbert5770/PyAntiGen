"""Pointwise 95% prediction bands from the parameter vectors of a stored profile: the profile ensemble.

METHOD. For a model output z = g(theta) at one time, the pointwise 95% confidence interval is the range of g over the set of parameter vectors whose negative
log-likelihood is within chi2(1, 0.95) / 2 = 1.9207 of its minimum, the interval of the prediction profile likelihood (Kreutz, Raue and Timmer, BMC Systems Biology
2012; check the reference before use). Every point of every profile stored by the AMICI profile stage carries the full parameter vector of that point, fitted with the
profiled parameter held; those with dnll <= 1.9207 are the *members* of the set. The band is the minimum and maximum, at each time, of the model curve over the members
(the optimum included).

WHAT IT IS NOT. The band is pointwise, not simultaneous; it is a band for the model curve conditional on the declared noise floors and the model structure, not a
prediction interval for new observations; and it is only as independent as the residuals the likelihood assumed. It is an INNER APPROXIMATION of the exact interval: the
members lie on the one-dimensional profile paths, not throughout the set, so the exact band can be wider where a prediction is extreme in a direction between them.
:func:`build_band` reports how the band grows as members are added, which shows whether the profile paths are exhausted (not that the set is covered). The delta-method
band (:func:`delta_band`) is the cross-check: symmetric, quadratic, built from the Fisher information of the fit.

This module has no project imports and no plotting. ``Prediction_record`` records and draws lines, ``Prediction`` drives the stages from ``Model_run.py --predict ensemble``.
"""
import glob
import json
import math
import os
import re

import numpy as np

from pyantigen.engine.Profile_record import DNLL_95 as LEVEL

KINDS = ("profile", "pins", "relax", "confirm", "fill", "dense", "recheck", "backfill")     # the files of a profile side that hold points


# ---------------------------------------------------------------------------------------------- members
def to_values(names, scales, x):
    """Linear parameter values from an optimizer-space vector: 10**x for the log10 parameters, x itself for the others."""
    return {n: (10.0 ** float(v) if sc == "log10" else float(v)) for n, sc, v in zip(names, scales, x)}


def extract_members(records_by_side, names, scales, optimum, level=LEVEL):
    """The members of the sublevel set from stored profile records.

    *records_by_side* is ``{(param, side): [PointRecord, ...]}`` (every kind of the side); the lowest dnll at each x is taken as the profile does (``curve``),
    points with dnll <= *level* and a full vector are kept, duplicates (the same vector) dropped. The optimum is member 0.
    """
    from pyantigen.engine.Profile_record import curve
    members = [{"id": 0, "param": "optimum", "side": "", "x_fixed": None, "dnll": 0.0, "verdict": "converged", "values": to_values(names, scales, optimum)}]
    seen = {tuple(np.round(np.asarray(optimum, float), 9))}
    for (param, side), records in sorted(records_by_side.items()):
        for q in curve(records):
            if q.dnll > level or q.x is None or len(q.x) != len(names):
                continue
            key = tuple(np.round(np.asarray(q.x, float), 9))
            if key in seen:
                continue
            seen.add(key)
            members.append({"id": len(members), "param": param, "side": side, "x_fixed": float(q.x_fixed), "dnll": float(q.dnll), "verdict": q.verdict,
                            "values": to_values(names, scales, q.x)})
    return members


def read_store(store_dir):
    """``(names, scales, optimum, records_by_side)`` of an AMICI profile store folder (``meta.json`` and the ``<kind>_<k>_<up|down>`` files)."""
    from pyantigen.engine import Profile as PR
    meta = json.load(open(os.path.join(store_dir, "meta.json")))
    names, scales = meta["names"], meta["scales"]
    by_side = {}
    for k, n in enumerate(names):
        for lab in ("up", "down"):
            st = PR.JsonlStore(store_dir, f"{k}_{lab}")
            by_side[(n, lab)] = [r for kind in KINDS for r in st.read(kind)]
    return names, scales, meta["x"], by_side


def fd_members(names, scales, optimum, delta=0.01):
    """The members for central finite differences: the optimum (id 0) and every parameter moved by +delta (id 1 + 2j) and -delta (id 2 + 2j) in optimizer space."""
    nan = float("nan")
    members = [{"id": 0, "param": "optimum", "side": "", "x_fixed": None, "dnll": 0.0, "verdict": "converged", "values": to_values(names, scales, optimum)}]
    for j, n in enumerate(names):
        for sign, lab in ((1.0, "+"), (-1.0, "-")):
            x = np.array(optimum, float)
            x[j] += sign * delta
            members.append({"id": len(members), "param": n, "side": lab, "x_fixed": float(x[j]), "dnll": nan, "verdict": "fd", "values": to_values(names, scales, x)})
    return members


def member_id(path):
    return int(os.path.basename(path).split("_")[1].split(".")[0])


def member_files(directory):
    """``{member id: path}`` of the finished ``member_<id>.npz`` files in *directory* (a ``.tmp.`` file is a member still being written)."""
    return {member_id(f): f for f in glob.glob(os.path.join(directory, "member_*.npz")) if ".tmp." not in f}


# ---------------------------------------------------------------------------------------------- band
def curve_order(k):
    m = re.match(r"^([A-Z]+)(\d*)_?(\d+)\.([fa]?)(\d+)$", k)
    if not m:
        return (k, 0, 0, 0)
    fig, n_fig, ax, kind, line = m.groups()
    return (fig, int(n_fig or 0), int(ax), int(line) + {"": 0, "f": 10**6, "a": 2 * 10**6}[kind])


def curve_keys(files):
    """The line keys found in the member files, from the first one (a key ``M0.3`` has the arrays ``M0.3:t``, ``M0.3:y`` and ``M0.3:label``), in figure order."""
    with np.load(files[0], allow_pickle=False) as z:
        return sorted({k.rsplit(":", 1)[0] for k in z.files if k.endswith(":t")}, key=curve_order)


def build_band(files, keys=None):
    """``{key: dict(t, lo, hi, opt, n, peak_lo, peak_hi, ...)}`` from member files (``member_0.npz`` is the optimum); members are on the first file's t grid
    (interpolated onto it if they are not). Also ``growth``: the band's mean width after the first 1, 2, 4, 8, ... members, in file order."""
    files = sorted(files, key=member_id)
    keys = keys or curve_keys(files)
    out = {}
    for key in keys:
        with np.load(files[0], allow_pickle=False) as z0:
            if len(z0[f"{key}:t"]) < 20:                                     # a reference mark or a two-point segment, not a curve of the model
                continue
        t0, ys = None, []
        for f in files:
            with np.load(f, allow_pickle=False) as z:
                t, y = z[f"{key}:t"], z[f"{key}:y"]
            if t0 is None:
                t0 = t
            ys.append(y if (len(t) == len(t0) and np.allclose(t, t0)) else np.interp(t0, t, y))
        Y = np.array(ys)
        lo, hi = Y.min(axis=0), Y.max(axis=0)
        ip = Y.argmax(axis=1)
        growth, n = [], 1
        while True:
            m = min(n, len(Y))
            growth.append((m, float(np.mean(Y[:m].max(axis=0) - Y[:m].min(axis=0)))))
            if m == len(Y):
                break
            n *= 2
        with np.load(files[0], allow_pickle=False) as z:
            label = str(z[f"{key}:label"]) if f"{key}:label" in z.files else ""
        out[key] = {"label": label, "t": t0, "lo": lo, "hi": hi, "opt": Y[0], "n": len(Y), "peak_opt": float(Y[0].max()), "peak_lo": float(Y.max(axis=1).min()),
                    "peak_hi": float(Y.max(axis=1).max()), "tpeak_opt": float(t0[Y[0].argmax()]), "tpeak_lo": float(t0[ip].min()),
                    "tpeak_hi": float(t0[ip].max()), "growth": growth}
    return out


def write_band(band, path):
    arrays = {}
    for key, b in band.items():
        for name in ("t", "lo", "hi", "opt"):
            arrays[f"{key}:{name}"] = b[name]
    np.savez_compressed(path, **arrays)


def read_band(path):
    """The band file written by :func:`write_band` as ``{key: dict(t, lo, hi, opt)}``."""
    out = {}
    with np.load(path, allow_pickle=False) as z:
        for k in z.files:
            key, name = k.rsplit(":", 1)
            out.setdefault(key, {})[name] = z[k]
    return out


def band_table(band, min_rel=1e-6):
    """Rows ``(key, label, max width, range of the optimum's line, relative width, growth after half the members, growth at all)`` for the lines whose band is not zero, widest first."""
    rows = []
    for key, b in band.items():
        w = float(np.max(b["hi"] - b["lo"]))
        span = float(np.ptp(b["opt"])) or 1.0
        if w / span < min_rel:
            continue
        g = b["growth"]
        half = next((v for n, v in g if n >= max(b["n"] // 2, 1)), g[-1][1])
        rows.append((key, b.get("label", ""), w, span, w / span, half, g[-1][1]))
    return sorted(rows, key=lambda r: -r[4])


def annotation_ranges(files):
    """``{key: dict(label, x0, lo, hi, n, ids)}`` for the annotations (``<key>.a<n>:x``) recorded in the member files: the position of each one in the optimum (member 0)
    and its range over the members that have it (a plot function that writes a derived number under an axis, such as the age at which a curve reaches a level)."""
    vals = {}
    for f in sorted(files, key=member_id):
        mid = member_id(f)
        with np.load(f, allow_pickle=False) as z:
            for k in z.files:
                if k.endswith(":x"):
                    key = k[:-2]
                    vals.setdefault(key, []).append((mid, float(z[k][0]), str(z[f"{key}:label"])))
    out = {}
    for key, v in vals.items():
        xs = np.array([x for _, x, _ in v])
        zero = next((x for mid, x, _ in v if mid == 0), float("nan"))
        out[key] = {"label": v[0][2], "x0": zero, "lo": float(xs.min()), "hi": float(xs.max()), "n": len(v), "ids": [mid for mid, _, _ in v]}
    return out


# ---------------------------------------------------------------------------------------------- delta method (the cross-check)
def delta_band(files_by_id, n_params, delta, cov, z=1.96):
    """``{key: dict(t, opt, lo, hi, G)}``: the delta-method band ``opt +- z * sqrt(g' cov g)`` per line, with the central-difference sensitivity g of the line to the
    parameters (columns in the order of *cov*, optimizer space). *files_by_id* maps the member ids of :func:`fd_members` to their files."""
    keys = curve_keys([files_by_id[0]])
    out = {}
    for key in keys:
        def load(i):
            with np.load(files_by_id[i], allow_pickle=False) as zz:
                return zz[f"{key}:t"], zz[f"{key}:y"]
        t0, y0 = load(0)
        G = np.zeros((len(t0), n_params))
        for j in range(n_params):
            (tp, yp), (tm, ym) = load(1 + 2 * j), load(2 + 2 * j)
            yp = yp if len(tp) == len(t0) and np.allclose(tp, t0) else np.interp(t0, tp, yp)
            ym = ym if len(tm) == len(t0) and np.allclose(tm, t0) else np.interp(t0, tm, ym)
            G[:, j] = (yp - ym) / (2.0 * delta)
        half = z * np.sqrt(np.maximum(np.einsum("tj,jk,tk->t", G, np.asarray(cov, float), G), 0.0))
        out[key] = {"t": t0, "opt": y0, "lo": y0 - half, "hi": y0 + half, "G": G}
    return out


def compare_bands(ens, dlt, min_rel=1e-6):
    """Rows ``(key, label, ensemble max width, delta max width, ratio delta/ensemble, share of times the ensemble band lies inside the delta band, ensemble width / line range)``
    for lines whose ensemble or delta band is not zero, by decreasing ratio."""
    rows = []
    for key, e in ens.items():
        d = dlt.get(key)
        if d is None:
            continue
        we, wd = float(np.max(e["hi"] - e["lo"])), float(np.max(d["hi"] - d["lo"]))
        span = float(np.ptp(e["opt"])) or 1.0
        if max(we, wd) / span < min_rel:
            continue
        tol = 1e-9 * span
        inside = float(np.mean((e["lo"] >= d["lo"] - tol) & (e["hi"] <= d["hi"] + tol)))
        rows.append((key, e.get("label", ""), we, wd, wd / we if we > 0 else float("inf"), inside, we / span))
    return sorted(rows, key=lambda r: -r[4])


# ---------------------------------------------------------------------------------------------- open sides
def open_side_check(data, member_files_by_id, band, fractions=(0.5, 0.75), min_rel=1e-6):
    """For each profile side (parameter, up or down) that has members, how much do the curves still move toward the farthest member of the side? On a side that is
    open (its last point is still inside the level, at the declared box) this is the plateau test. Returns
    ``[(param, side, decades, key, label, fraction, change, band width, change / width)]``: *change* is the largest difference over time between the curve of the
    farthest member and the curve of the member nearest to *fraction* of the way out (in optimizer units), *band width* the largest width of the whole band of that line.
    A ratio near 0 means the curve has stopped moving (a plateau, the band does not depend on where the box ends); a ratio near 1 means the end of the side still
    sets the band. *data* is the members file (``names``, ``scales``, ``members``)."""
    names, scales = data["names"], data["scales"]
    opt = data["members"][0]["values"]
    out = []
    by_side = {}
    for m in data["members"][1:]:
        by_side.setdefault((m["param"], m["side"]), []).append(m)

    def pos(m):
        return float(m["x_fixed"])                                  # the profiled parameter's value in optimizer space
    for (param, side), ms in sorted(by_side.items()):
        i = names.index(param)
        x_opt = math.log10(opt[param]) if scales[i] == "log10" else opt[param]
        ms = sorted(ms, key=lambda m: abs(pos(m) - x_opt))
        far = ms[-1]
        reach = abs(pos(far) - x_opt)
        for frac in fractions:
            near = min(ms, key=lambda m: abs(abs(pos(m) - x_opt) - frac * reach))
            if near is far:
                continue
            with np.load(member_files_by_id[far["id"]], allow_pickle=False) as zf, np.load(member_files_by_id[near["id"]], allow_pickle=False) as zn:
                for key, b in band.items():
                    width = float(np.max(b["hi"] - b["lo"]))
                    if width <= min_rel * (float(np.ptp(b["opt"])) or 1.0) or width < 1e-4 * float(np.max(np.abs(b["opt"]))):
                        continue                                      # no band, or a band that is rounding noise on a line that is itself tiny
                    t0 = zf[f"{key}:t"]
                    yf = zf[f"{key}:y"]
                    yn = np.interp(t0, zn[f"{key}:t"], zn[f"{key}:y"])
                    change = float(np.max(np.abs(yf - yn)))
                    label = str(zf[f"{key}:label"]) if f"{key}:label" in zf.files else ""
                    out.append((param, side, reach, key, label, frac, change, width, change / width if width > 0 else 0.0))
    return out

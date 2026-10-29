"""Profile-likelihood figures: one panel per parameter, drawn the same way for every route.

This is the standard of the project's ``tools/profile_plot_pypesto.py`` moved into the engine. Two figures with identical data:

    <prefix>_full.png   each panel's y-axis spans every point, whatever its dNLL
    <prefix>_zoom.png   y-limit ``zoom`` (default 5), the whole series still drawn, so a curve that leaves the frame is cut by the
                        axis, not dropped

A panel's x-axis is the optimizer scale: log10 of the fixed value for a log10 parameter, the value itself otherwise. The optimum
(dNLL 0) is part of every series and marked; a filled marker is a converged nuisance fit, a hollow one did not converge (its dNLL
is then an upper bound); the dashed line is the chi-square(1)/2 cut. There are no slice points.

``prof`` is ``{name: [(x, dnll, converged), ...]}`` and ``mle`` is ``{name: x}``, both on the optimizer scale.

As a program it reads the files the AMICI profile stage keeps (``profile_*.jsonl``, ``relax_*.jsonl``, ``fill_*.jsonl``) and
``meta.json`` (names, scales, optimum), which the stage writes next to them:

    python -m pyantigen.engine.Profile_plot PROFILE_DIR [--meta META.json|fim.npz] [--zoom 5] [--out-prefix P] [--no-timestamp]

The files are named as the engine names them, <results>/<MODEL>_<run>_profile_likelihood_full_<timestamp>.png and
..._zoom_<timestamp>.png, when PROFILE_DIR is <results>/<MODEL>/profiles/<run>[/amici]; a likelihood slice is drawn the same way
(figures(..., kind="slice")) and named ..._likelihood_slice_full_<timestamp>.png.

Read-only on the profile folder; no AMICI, safe on a login node.
"""
import argparse
import glob
import json
import os
import re

import numpy as np

THRESHOLD = 1.9207
BLUE, RED, INK, GREY = "#2a78d6", "#c23b2e", "#0b0b0b", "#8a8a85"


def interval(pts, mle, thr=THRESHOLD):
    """Converged in-threshold extent on each side and whether the outermost tested point is still inside the cut."""
    out = {}
    for side, sel in (("low", lambda x: x < mle), ("high", lambda x: x > mle)):
        S = sorted([q for q in pts if sel(q[0])], key=lambda q: q[0], reverse=(side == "low"))
        inside = [q for q in S if q[2] and q[1] <= thr]
        edge = S[-1] if S else None          # outermost tested point on this side
        out[side] = dict(reach=(inside[-1][0] if inside else None), open=bool(edge and edge[1] <= thr), edge=edge)
    return out


def summary_lines(names, mle, prof, thr=THRESHOLD):
    """The table printed with the figures: points, converged points, and how far each side reached."""
    lines = [f"{'parameter':40s} {'MLE':>8s} {'n':>3s} {'conv':>4s} {'low reach':>10s} {'high reach':>10s}   edges (x, dNLL)"]
    for p in names:
        Pn = sorted(prof[p])
        iv = interval(Pn, mle[p], thr)
        edges = []
        for side in ("low", "high"):
            e = iv[side]["edge"]
            edges.append(f"{e[0]:.2f}, {e[1]:.1f}{'*' if iv[side]['open'] else ''}" if e else "none")
        fmt = lambda v: "   -" if v is None else f"{v:8.2f}"
        lines.append(f"{p:40s} {mle[p]:8.2f} {len(Pn):3d} {sum(q[2] for q in Pn):4d} "
                     f"{fmt(iv['low']['reach']):>10s} {fmt(iv['high']['reach']):>10s}   {edges[0]} | {edges[1]}")
    lines.append("* = outermost tested point on that side is still inside the cut (open)")
    return lines


def draw(path, run, names, thr, mle, prof, ymax, scales=None, note=None, kind="profile"):
    """One figure; ymax=None lets each panel's y-axis span all of its points. *kind* is ``"profile"`` or ``"slice"`` (the other
    parameters held at the optimum: grey diamonds, no verdicts)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    scales = scales or {}
    ncol = 3
    nrow = max(1, int(np.ceil(len(names) / ncol)))
    fig, ax = plt.subplots(nrow, ncol, figsize=(4.3 * ncol, 3.1 * nrow))
    ax = np.atleast_1d(ax).ravel()
    for k, p in enumerate(names):
        a_ = ax[k]
        P = sorted(list(prof[p]) + [(mle[p], 0.0, True)])           # the optimum is part of the profile series
        px = np.array([q[0] for q in P])
        py = np.array([q[1] for q in P])
        pc = np.array([q[2] for q in P], dtype=bool)
        a_.axhline(thr, color=RED, ls="--", lw=1)
        if kind == "slice":
            a_.plot(px, py, "-", color=GREY, lw=1.0, zorder=2)
            a_.plot(px, py, "D", color=GREY, ms=4, mec="white", mew=0.5, zorder=3, label="slice (others at the optimum)")
        else:
            a_.plot(px, py, "-", color=BLUE, lw=1.2, zorder=2)
            a_.plot(px[pc], py[pc], "o", color=BLUE, ms=4.5, mec="white", mew=0.6, zorder=3, label="profile, converged")
            a_.plot(px[~pc], py[~pc], "o", mfc="white", mec=BLUE, ms=4.5, mew=1.1, zorder=3, label="profile, not converged")
        a_.plot([mle[p]], [0], "o", color=INK, ms=5, zorder=4, label="MLE")
        a_.axvline(mle[p], color=INK, lw=0.6, alpha=0.35)
        pad = 0.05 * max(np.ptp(px), 1e-3)
        a_.set_xlim(px.min() - pad, px.max() + pad)
        top = float(py.max()) * 1.05 if ymax is None else ymax
        top = max(top, 1e-6)
        a_.set_ylim(-0.02 * top, top)
        a_.set_title(p, loc="left", fontsize=10)
        a_.set_xlabel("log10 value (fixed)" if scales.get(p, "log10") == "log10" else "value (fixed)", fontsize=8)
        a_.set_ylabel("ΔNLL", fontsize=8)
        a_.tick_params(labelsize=8)
        for sp in ("top", "right"):
            a_.spines[sp].set_visible(False)
    for a_ in ax[len(names):]:
        a_.axis("off")
    h, l = ax[0].get_legend_handles_labels()
    h.append(plt.Line2D([], [], color=RED, ls="--"))
    l.append(f"cut {thr:.2f}")
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=9)
    what = "full range" if ymax is None else f"y-limit ΔNLL {ymax:g}, series drawn in full"
    import textwrap       # a run named after its groups and hashes can be longer than the figure
    head = "\n".join(textwrap.wrap(f"{run}: {'likelihood slice' if kind == 'slice' else 'profile likelihood'}, all parameters  ({what})", 120))
    fig.suptitle(head + (f"\n{note}" if note else ""), x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0.05, 1, 0.94 if note else 0.96))
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def figures(prefix, run, names, mle, prof, scales=None, zoom=5.0, thr=THRESHOLD, note=None, log=print, suffix="", kind="profile"):
    """Write ``<prefix>_full<suffix>.png`` and ``<prefix>_zoom<suffix>.png``; returns their paths."""
    names = [n for n in names if prof.get(n)]
    paths = []
    for tag, ymax in (("_full", None), ("_zoom", zoom)):
        paths.append(draw(prefix + tag + suffix + ".png", run, names, thr, mle, prof, ymax, scales, note, kind))
    if kind == "profile":                       # a slice has no verdicts and no reach: the engine prints its ranges
        for line in summary_lines(names, mle, prof, thr):
            log(line)
    return paths


# --- reading the files of the AMICI profile stage -------------------------------------------------------------------

def _read_jsonl(path):
    out = []
    if os.path.exists(path):
        for line in open(path):
            try:
                out.append(json.loads(line))
            except ValueError:
                pass
    return out


def load_meta(pdir, meta=None):
    """``(names, scales, x)`` of the optimum: from ``meta.json`` in *pdir*, else *meta* (a json or the FIM ``.npz``)."""
    cands = [os.path.join(pdir, "meta.json")] + ([meta] if meta else []) + [os.path.join(pdir, "fim", "fim.npz")]
    for c in cands:
        if c and os.path.exists(c):
            if c.endswith(".npz"):
                z = np.load(c)
                return list(z["names"]), list(z["scales"]), [float(v) for v in z["x"]]
            d = json.load(open(c))
            return list(d["names"]), list(d["scales"]), [float(v) for v in d["x"]]
    raise FileNotFoundError(f"no meta.json in {pdir}: pass --meta (a meta.json or a fim.npz of this fit)")


def load_points(pdir, names):
    """``{name: [(x, dnll, converged)]}``: walk, relax and fill records; a point keeps its lowest nll over the files."""
    prof = {n: [] for n in names}
    best = {}
    for kind in ("profile", "relax", "fill"):
        for f in sorted(glob.glob(os.path.join(pdir, f"{kind}_*.jsonl"))):
            for r in _read_jsonl(f):
                p, x, nll = r.get("param"), r.get("x_fixed"), r.get("nll")
                if p not in prof or x is None or nll is None or r.get("dnll") is None:
                    continue
                if abs(r["dnll"]) < 1e-12 and not r.get("evals"):      # the walk's own copy of the optimum
                    continue
                key = (p, round(float(x), 9))
                if key not in best or nll < best[key][0]:
                    best[key] = (nll, float(x), float(r["dnll"]), r.get("verdict") == "converged")
    for (p, _), (_nll, x, dnll, conv) in best.items():
        prof[p].append((x, dnll, conv))
    return prof


def locate(pdir):
    """``(model, run, results_dir)`` of a stage folder ``<results>/<MODEL>/profiles/<run>[/amici]``, as the engine lays them out.

    Outside that layout the model is empty and the figures go to the current directory.
    """
    parts = os.path.abspath(pdir).rstrip("/").split(os.sep)
    if parts and parts[-1] == "amici":
        parts = parts[:-1]
    if len(parts) >= 3 and parts[-2] == "profiles":
        return parts[-3], parts[-1], os.sep.join(parts[:-2]) or os.sep
    return "", parts[-1], os.getcwd()


def main(argv=None):
    from datetime import datetime
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("profile_dir")
    ap.add_argument("--meta", default=None)
    ap.add_argument("--zoom", type=float, default=5.0)
    ap.add_argument("--out-prefix", default=None, help="default: <results>/<MODEL>_<run>_profile_likelihood, as the engine names them")
    ap.add_argument("--no-timestamp", action="store_true", help="name the files <prefix>_full.png and <prefix>_zoom.png")
    a = ap.parse_args(argv)
    pdir = a.profile_dir.rstrip("/")
    names, scales, x = load_meta(pdir, a.meta)
    prof = load_points(pdir, names)
    mle = dict(zip(names, x))
    model, run, results = locate(pdir)
    prefix = a.out_prefix or os.path.join(results, "_".join(t for t in (model, run, "profile_likelihood") if t))
    suffix = "" if a.no_timestamp else "_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    for p in figures(prefix, f"{model} {run}".strip(), names, mle, prof, dict(zip(names, scales)), a.zoom, suffix=suffix):
        print("wrote", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

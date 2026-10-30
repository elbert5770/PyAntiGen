"""The Fisher information figure: how well the optimum is determined, and where the quadratic picture of it breaks.

Three panels, one parameter order (clustered by correlation, so the strongly coupled parameters sit together):

    A  Wald against profile    each parameter's profile-likelihood interval in units of its Wald half-width, so the Fisher information's
                               95 % interval is the shaded band at -1 .. +1 for every parameter and the distance of a blue end from the
                               band is how far the likelihood departs from the quadratic. An arrow is a side that never crossed the
                               cut. The Wald half-width itself (decades for a log10 parameter) is written beside each row.
    B  correlation             the Cramer-Rao correlation matrix, diverging blue/red about a neutral zero; values shown where
                               |rho| >= ``show``.
    C  weakest directions      the eigenvectors of the smallest eigenvalues (rows, weakest first) as loadings on the parameters, so
                               the combination the data barely determine is read off directly. Its columns are panel B's.

A hollow marker is a crossing set by a point that did not converge (the interval is at least that wide).
A parameter's Wald interval is ``x +- sqrt(2 * thr) * se`` in the optimizer space (decades for a log10 parameter), so it is symmetric
there; ``thr`` is the profile cut (chi-square(1)/2, 1.9207), which makes the two intervals the same statistic.
Panel A divides by that half-width because the intervals of one problem span several orders of magnitude: on a common axis in decades
the tight parameters are dots.

As a program, from the stage's FIM and profile files:

    python -m pyantigen.engine.Fim_plot PROFILE_DIR [--fim FIM.npz] [--weak 3] [--out PATH] [--no-timestamp]

PROFILE_DIR is ``<results>/<MODEL>/profiles/<run>[/amici]``; ``--fim`` defaults to ``PROFILE_DIR/fim/fim.npz``. Without profile points
the figure is panels B and C and the Wald half of A. Read-only on the folders; no AMICI, safe on a login node.
"""
import argparse
import math
import os

import numpy as np

THRESHOLD = 1.9207
# The reference palette (light surface): categorical slots 1 and 2, the diverging pair with its gray midpoint, the ink and chrome.
BLUE, ORANGE, RED = "#2a78d6", "#eb6834", "#e34948"
NEUTRAL, SURFACE = "#f0efec", "#fcfcfb"
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"


def profile_extent(pts, mle, thr=THRESHOLD):
    """``{"low": (x, state), "high": (x, state)}``: where the profile reaches the cut on each side of *mle*, linearly interpolated.

    *pts* is ``[(x, dnll, converged), ...]`` on the optimizer scale. *state* is ``"closed"`` when the first point over the cut
    converged (the crossing is bracketed), ``"bound"`` when it did not (its dnll is an upper bound on the profile, so the true
    crossing is at least this far out: the interval is at least this wide), and ``"open"`` when no point is over the cut (*x* is the
    outermost point, or None for a side without points).
    """
    out = {}
    for side, sel, rev in (("low", lambda x: x < mle, True), ("high", lambda x: x > mle, False)):
        S = sorted([q for q in pts if sel(q[0])], key=lambda q: q[0], reverse=rev)
        prev, hit = (mle, 0.0), None
        for q in S:
            if q[1] > thr:
                hit = q
                break
            prev = (q[0], q[1])
        if hit is not None:
            t = (thr - prev[1]) / (hit[1] - prev[1]) if hit[1] != prev[1] else 1.0
            out[side] = (prev[0] + t * (hit[0] - prev[0]), "closed" if hit[2] else "bound")
        else:
            out[side] = (prev[0] if S else None, "open")
    return out


def cluster_order(corr):
    """Parameter order that puts strongly correlated parameters side by side (average linkage on 1 - |rho|)."""
    n = len(corr)
    if n < 3:
        return list(range(n))
    from scipy.cluster.hierarchy import leaves_list, linkage, optimal_leaf_ordering
    from scipy.spatial.distance import squareform
    d = 1.0 - np.abs(np.asarray(corr, float))
    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, 0.0)
    z = linkage(squareform(np.clip(d, 0.0, None), checks=False), method="average")
    return [int(i) for i in leaves_list(optimal_leaf_ordering(z, squareform(np.clip(d, 0.0, None), checks=False)))]


def weakest_directions(F, k=3):
    """``(eigenvalues, loadings)`` of the *k* smallest eigenvalues, weakest first; each loading vector's largest entry is positive."""
    F = np.asarray(F, float)
    w, V = np.linalg.eigh(0.5 * (F + F.T))
    k = min(k, len(w))
    vecs = []
    for i in range(k):
        v = V[:, i].copy()
        if v[int(np.argmax(np.abs(v)))] < 0:
            v = -v
        vecs.append(v)
    return w[:k], np.array(vecs)


def _cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("blue_neutral_red", [BLUE, NEUTRAL, RED])


def _ink_on(rgba):
    r, g, b = rgba[:3]
    return INK if 0.2126 * r + 0.7152 * g + 0.0722 * b > 0.45 else "#ffffff"


def draw(path, title, names, scales, x, F, prof=None, k=3, thr=THRESHOLD, show=0.8, load_show=0.3, dpi=130):
    """Write the three-panel figure to *path*; *prof* is ``{name: [(x, dnll, converged)]}`` or None. Returns *path*."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    names, x = list(names), np.asarray(x, float)
    n = len(names)
    cov = np.linalg.pinv(np.asarray(F, float))
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    corr = cov / np.outer(np.maximum(se, 1e-300), np.maximum(se, 1e-300))
    order = cluster_order(corr)
    lam, load = weakest_directions(F, k)
    half = math.sqrt(2.0 * thr)
    w = np.linalg.eigvalsh(0.5 * (F + F.T))
    cond = w[-1] / w[0] if w[0] > 0 else math.inf

    rows = []                                             # per parameter, in plot order: (Wald half-width, {side: (units, state)} | None)
    for j in order:
        hw = max(half * se[j], 1e-12)
        pe = None
        if prof is not None and prof.get(names[j]):
            ext = profile_extent(prof[names[j]], x[j], thr)
            pe = {sd: (None if v is None else (v - x[j]) / hw, state) for sd, (v, state) in ext.items()}
        rows.append((hw, pe))
    seen = [abs(v) for _, pe in rows if pe for v, _st in pe.values() if v is not None and math.isfinite(v)]
    lim = min(6.0, max(2.5, math.ceil(1.15 * max(seen, default=2.5))))

    plt.rcParams.update({"font.family": "sans-serif", "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": INK2,
                         "ytick.color": INK2, "text.color": INK})
    height = max(4.6, 0.33 * n + 2.8)
    fig = plt.figure(figsize=(16.5 if n > 6 else 12.5, height), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 2, width_ratios=[0.95, 1.1], height_ratios=[max(n, 4), max(k, 1) + 1.6], wspace=0.5, hspace=0.12, top=0.84, bottom=0.14)
    axA = fig.add_subplot(gs[:, 0], facecolor=SURFACE)
    axB = fig.add_subplot(gs[0, 1], facecolor=SURFACE)
    axC = fig.add_subplot(gs[1, 1], facecolor=SURFACE, sharex=axB)

    # A: the profile interval in units of the Wald half-width; the Wald interval is the band at -1 .. +1
    axA.axvspan(-1.0, 1.0, color=GRID, alpha=0.9, lw=0, zorder=0)
    for i, (hw, pe) in enumerate(rows):
        if pe:
            ends = []
            for side in ("low", "high"):
                v, state = pe[side]
                clipped = v is not None and abs(v) > lim
                v = 0.0 if v is None else float(np.clip(v, -lim, lim))
                ends.append((v, "open" if clipped else state, -1 if side == "low" else 1))
            axA.plot([ends[0][0], ends[1][0]], [i, i], color=BLUE, lw=2.4, solid_capstyle="round", zorder=2)
            for v, state, sg in ends:
                if pe["low" if sg < 0 else "high"][0] is None:
                    continue
                if state == "open":
                    axA.plot([v], [i], "<" if sg < 0 else ">", color=BLUE, ms=7.5, mec=SURFACE, mew=1.2, zorder=3)
                else:
                    axA.plot([v], [i], "o", mfc=BLUE if state == "closed" else SURFACE, mec=BLUE, ms=6 if state == "closed" else 6.5,
                             mew=1.2 if state == "closed" else 1.8, zorder=3)
        unit = " dec" if scales[order[i]] == "log10" else ""
        axA.text(1.012, i, f"\u00b1{rows[i][0]:.2g}{unit}", transform=axA.get_yaxis_transform(), va="center", ha="left",
                 fontsize=8, color=MUTED)
    axA.axvline(0.0, color=AXIS, lw=1.0, zorder=1)
    axA.set_ylim(n - 0.5, -0.5)
    axA.set_xlim(-lim, lim)
    axA.set_yticks(range(n))
    axA.set_yticklabels([names[j] for j in order], fontsize=9)
    axA.set_xlabel("profile limit from the optimum, in Wald half-widths  (Wald 95 % = shaded band)", fontsize=9.5)
    axA.grid(axis="x", color=GRID, lw=0.8)
    axA.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        axA.spines[sp].set_visible(False)
    axA.tick_params(axis="y", length=0)
    handles = [Line2D([0], [0], color=BLUE, lw=2.4, marker="o", ms=6, label="profile 95 %"),
               Line2D([0], [0], color="none", marker="o", mfc=SURFACE, mec=BLUE, mew=1.8, ms=6.5, label="set by an unconverged point"),
               Line2D([0], [0], color="none", marker=">", mfc=BLUE, mec=BLUE, ms=7.5, label="open: no crossing"),
               Patch(facecolor=GRID, edgecolor="none", label="Wald 95 % (FIM)")]
    if prof is None:
        handles = handles[3:]
    axA.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, frameon=False, fontsize=8.5, handlelength=1.8)
    axA.set_title("A  Profile interval against the quadratic (Wald) interval", loc="left", fontsize=11, pad=44, color=INK)

    # B: correlation
    cm = _cmap()
    Cb = corr[np.ix_(order, order)].copy()
    np.fill_diagonal(Cb, 0.0)                          # a parameter against itself says nothing
    mesh = axB.pcolormesh(np.arange(n + 1), np.arange(n + 1), Cb, cmap=cm, vmin=-1, vmax=1, edgecolors=SURFACE, linewidth=1.0)
    for i in range(n):
        for j in range(n):
            if abs(Cb[i, j]) >= show and i != j:
                axB.text(j + 0.5, i + 0.5, f"{Cb[i, j]:.2f}", ha="center", va="center", fontsize=6.8,
                         color=_ink_on(cm((Cb[i, j] + 1) / 2)))
    axB.set_ylim(n, 0)
    axB.set_yticks(np.arange(n) + 0.5)
    axB.set_yticklabels([names[j] for j in order], fontsize=8)
    axB.yaxis.tick_right()
    axB.tick_params(axis="both", length=0, labelbottom=False)
    axB.set_title(f"B  Correlation of the estimates (values shown at |rho| >= {show:g})", loc="left", fontsize=11, pad=8, color=INK)

    # C: weakest directions
    L = load[:, order]
    axC.pcolormesh(np.arange(n + 1), np.arange(len(L) + 1), L, cmap=cm, vmin=-1, vmax=1, edgecolors=SURFACE, linewidth=1.0)
    for i in range(L.shape[0]):
        for j in range(n):
            if abs(L[i, j]) >= load_show:
                axC.text(j + 0.5, i + 0.5, f"{L[i, j]:.2f}", ha="center", va="center", fontsize=6.8, color=_ink_on(cm((L[i, j] + 1) / 2)))
    axC.set_ylim(L.shape[0], 0)
    axC.set_yticks(np.arange(L.shape[0]) + 0.5)
    axC.set_yticklabels([f"{'weakest' if i == 0 else 'next'}  λ = {v:.3g}" for i, v in enumerate(lam)], fontsize=8.5)
    axC.set_xticks(np.arange(n) + 0.5)
    axC.set_xticklabels([names[j] for j in order], rotation=60, ha="right", fontsize=8)
    axC.tick_params(axis="both", length=0)
    axC.set_title(f"C  Weakest directions (loadings of the {len(L)} smallest eigenvalues)", loc="left", fontsize=11, pad=6, color=INK)
    for ax in (axB, axC):
        for sp in ax.spines.values():
            sp.set_visible(False)

    cax = axB.inset_axes([1.5, 0.55, 0.035, 0.42])
    cb = fig.colorbar(mesh, cax=cax)
    cb.set_label("rho  /  loading", fontsize=8.5, color=INK2)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=8, length=0)

    cond_txt = "singular" if not math.isfinite(cond) else f"{cond:.3g}"
    fig.suptitle(f"{title}   |   {n} parameters, condition number {cond_txt}", x=0.01, ha="left", fontsize=12.5, color=INK, y=0.985)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=dpi, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return path


def main(argv=None):
    from datetime import datetime

    from pyantigen.engine import Profile_plot as PP
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("profile_dir")
    ap.add_argument("--fim", default=None, help="the FIM file (default: PROFILE_DIR/fim/fim.npz)")
    ap.add_argument("--weak", type=int, default=3, help="how many of the weakest directions panel C shows")
    ap.add_argument("--out", default=None, help="the PNG (default: <results>/<MODEL>_<run>_fim[_<timestamp>].png)")
    ap.add_argument("--no-timestamp", action="store_true")
    a = ap.parse_args(argv)
    pdir = a.profile_dir.rstrip("/")
    fim = a.fim or os.path.join(pdir, "fim", "fim.npz")
    if not os.path.exists(fim):
        raise SystemExit(f"no FIM at {fim}: run the fim stage first (PYANTIGEN_PROFILE_MODE=fim)")
    z = np.load(fim, allow_pickle=True)
    names, scales, x, F = [str(v) for v in z["names"]], [str(v) for v in z["scales"]], np.asarray(z["x"], float), np.asarray(z["F"], float)
    prof = PP.load_points(pdir, names)
    prof = prof if any(prof.values()) else None
    model, run, results = PP.locate(pdir)
    stem = "_".join(t for t in (model, run, "fim") if t)
    out = a.out or os.path.join(results, stem + ("" if a.no_timestamp else "_" + datetime.now().strftime("%Y%m%d_%H%M%S")) + ".png")
    print("wrote", draw(out, f"{model} {run}".strip() or "Fisher information", names, scales, x, F, prof, k=a.weak))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

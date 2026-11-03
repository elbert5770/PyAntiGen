"""Waterfall plot of an AMICI multi-start: the final objective of every start, best to worst.

    python -m pyantigen.engine.Multistart_plot <multistart directory> [--out PATH] [--dnll 1.0]

*directory* is the one ``Multistart_amici`` keeps the starts in (``<plot_path>/multistart/<fit id>/``, holding
``start_<i>.json``). The collect stage draws it for you, as ``waterfall.png`` in that directory; this module is also the
way to redraw it, or to draw one for a directory whose collect predates the plot.

Reading the plot: the points are the starts' final fits sorted best to worst, so a flat run of points at the left is one
basin that several starts reached, and a step up is a start that stopped somewhere worse. Height is the NLL above the
best start (ΔNLL), with a symmetric log axis so that both optimizer scatter (≪ 1) and a different basin (≫ 1) show. The
dashed line is *dnll* (default 1.0): starts below it are within the usual "same optimum" tolerance of the best.
A start that has not converged is drawn open, and a start that was never run or failed to write is listed in the title.
"""
import argparse
import json
import math
import os

import numpy as np

from pyantigen.engine.Profile_plot import BLUE, GREY, INK, RED   # the profile plots' colours

ORANGE = "#d98a1f"


def read_starts(directory):
    """Every ``start_<i>.json`` in *directory*, as a list of records ordered by start index."""
    out = []
    for name in os.listdir(directory):
        if name.startswith("start_") and name.endswith(".json"):
            try:
                with open(os.path.join(directory, name)) as fh:
                    rec = json.load(fh)
            except (OSError, ValueError):
                continue
            if "f" in rec and "start" in rec:
                out.append(rec)
    return sorted(out, key=lambda r: r["start"])


def waterfall_rows(records, n_starts=None):
    """``(rows, missing)``: *rows* sorted best to worst, each ``{"start", "f", "dnll", "converged", "rank"}``; *missing* the
    start indices (of ``range(n_starts)`` when given) with no finite result."""
    finite = [r for r in records if r.get("f") is not None and math.isfinite(r["f"])]
    finite.sort(key=lambda r: r["f"])
    best = finite[0]["f"] if finite else float("nan")
    rows = [{"start": int(r["start"]), "f": float(r["f"]), "dnll": float(r["f"]) - best,
             "converged": bool(r.get("converged")), "rank": k} for k, r in enumerate(finite)]
    have = {r["start"] for r in rows}
    missing = [] if n_starts is None else [i for i in range(int(n_starts)) if i not in have]
    return rows, missing


def draw(path, rows, missing=(), dnll=1.0, title=None):
    """Write the waterfall of *rows* (:func:`waterfall_rows`) to *path* (png/pdf/svg by extension)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if not rows:
        raise ValueError("no start with a finite objective to draw")
    x = np.arange(len(rows))
    y = np.array([r["dnll"] for r in rows])
    done = np.array([r["converged"] for r in rows])
    fig, ax = plt.subplots(figsize=(max(4.5, 0.55 * len(rows) + 2.5), 3.6))
    ax.axhline(dnll, color=RED, ls="--", lw=1, label=f"ΔNLL = {dnll:g}")
    ax.plot(x, y, "-", color=GREY, lw=0.8, zorder=1)
    ax.plot(x[done], y[done], "o", color=BLUE, ms=7, mec="white", mew=0.7, zorder=3, label="converged")
    ax.plot(x[~done], y[~done], "o", mfc="white", mec=BLUE, ms=7, mew=1.3, zorder=3, label="not converged")
    for k, r in enumerate(rows):
        if r["start"] == 0:
            ax.plot([k], [y[k]], "o", mfc="none", mec=ORANGE, ms=12, mew=1.8, zorder=2, label="start 0 (x0)")
        ax.annotate(str(r["start"]), (k, y[k]), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=8, color=INK)
    ax.set_yscale("symlog", linthresh=max(1e-3, dnll * 1e-2))
    ax.set_ylim(bottom=0)
    ax.set_xlim(-0.6, len(rows) - 0.4)
    ax.set_xticks(x)
    ax.set_xlabel("rank (best to worst); labels are start indices", fontsize=9)
    ax.set_ylabel("ΔNLL from the best start", fontsize=9)
    ax.tick_params(labelsize=8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    note = f"; no result for start(s) {', '.join(map(str, missing))}" if missing else ""
    ax.set_title((title or "Multi-start waterfall") + note, loc="left", fontsize=10)
    # the legend repeats "start 0" once per draw call; keep the first of each label
    h, l = ax.get_legend_handles_labels()
    seen = dict(zip(l[::-1], h[::-1]))
    ax.legend(list(seen.values())[::-1], list(seen.keys())[::-1], frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_directory(directory, out=None, n_starts=None, dnll=1.0, title=None):
    """Read the starts in *directory*, draw the waterfall to *out* (default ``<directory>/waterfall.png``), return the path."""
    rows, missing = waterfall_rows(read_starts(directory), n_starts)
    return draw(out or os.path.join(directory, "waterfall.png"), rows, missing, dnll, title)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("directory", help="the multistart directory holding start_<i>.json")
    ap.add_argument("--out", default=None, help="output file (default: <directory>/waterfall.png)")
    ap.add_argument("--dnll", type=float, default=1.0, help="same-optimum tolerance drawn as a dashed line")
    ap.add_argument("--n-starts", type=int, default=None, help="the spec's n_starts, to name starts with no result")
    a = ap.parse_args(argv)
    print(f"waterfall written to {plot_directory(a.directory, a.out, a.n_starts, a.dnll)}")


if __name__ == "__main__":
    main()

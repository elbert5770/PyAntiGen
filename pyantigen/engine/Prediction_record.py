"""Record the lines a plot function draws, and draw a band behind them: the figure side of the ensemble prediction band.

A project's plot function (``EXPERIMENT_dict["plot"]``) draws curves the project defines: concentrations, fluxes, ratios, a
quantity derived from several columns. Rather than ask the project to describe each, :class:`LineRecorder` wraps ``Axes.plot``,
``Axes.fill_between`` and ``Axes.annotate`` while the plot function runs and keeps what was drawn, keyed by the figure, the axes and
the line. :class:`BandDrawer` does the reverse: it adds a ``fill_between`` behind each line that has a band, so no plot function knows
about bands.

Keys: ``<fig><axes>.<line>`` with *fig* ``M`` for the first figure that drew a line, ``S`` for the second and ``F2_``, ``F3_`` ... after
(:func:`fig_prefix`); *axes* is the position of the axes in ``fig.axes`` when it first drew; *line* counts the lines of that axes. A stacked
area's upper boundary is ``f<n>`` and an annotation ``a<n>``. The key depends on the order a plot function draws, so a change to the plot
function changes the keys and invalidates band files made before it.

Matplotlib is imported only when a recorder or drawer is entered.
"""
import os
import textwrap

import numpy as np

MAXPTS = 4000          # a recorded line with more points is thinned to this many, evenly spaced by index, the ends kept


def thin(x, y, maxpts=MAXPTS):
    if len(x) <= maxpts:
        return x, y
    idx = np.unique(np.linspace(0, len(x) - 1, maxpts).astype(int))
    return x[idx], y[idx]


def fig_prefix(index):
    """Key prefix of the *index*-th figure that drew a line: ``M`` (the first), ``S`` (the second), ``F2_``, ``F3_`` ...; keys are ``M0.1``, ``S3.0``, ``F2_0.4``."""
    return "MS"[index] if index < 2 else f"F{index}_"


class LineRecorder:
    """Records every line drawn by ``Axes.plot`` while active, the upper boundary of every ``Axes.fill_between`` (a stacked area) and the x position of every
    ``Axes.annotate``; ``Figure.savefig`` becomes a no-op (a member's figure is not needed).

    ``arrays()`` gives ``{key: x, key: y, ...}`` with the keys ``<key>:t`` / ``:y`` / ``:label`` for lines and fill boundaries, ``:x`` / ``:label`` for an
    annotation. Lines of more than ``MAXPTS`` points are thinned; fills of fewer than 20 points are not recorded.
    """

    def __enter__(self):
        import matplotlib.axes as mpl_axes
        import matplotlib.figure as mpl_figure
        self.calls, self.fills, self.notes = [], [], []
        self._plot, self._save = mpl_axes.Axes.plot, mpl_figure.Figure.savefig
        self._fill, self._annotate = mpl_axes.Axes.fill_between, mpl_axes.Axes.annotate
        rec = self

        def plot(ax, *args, **kw):
            lines = rec._plot(ax, *args, **kw)
            for ln in lines:
                try:
                    rec.calls.append((ax, np.asarray(ln.get_xdata(), float), np.asarray(ln.get_ydata(), float), str(ln.get_label())))
                except (TypeError, ValueError):                              # a line of dates or strings: not a model curve
                    pass
            return lines

        def fill_between(ax, x, y1, y2=0, *args, **kw):
            out = rec._fill(ax, x, y1, y2, *args, **kw)
            try:
                xs, top = np.asarray(x, float), np.asarray(y2, float)
                if xs.ndim == 1 and top.shape == xs.shape and len(xs) >= 20:
                    rec.fills.append((ax, xs, top, str(kw.get("label", ""))))
            except (TypeError, ValueError):
                pass
            return out

        def annotate(ax, text, xy, *args, **kw):
            out = rec._annotate(ax, text, xy, *args, **kw)
            try:
                rec.notes.append((ax, float(xy[0]), str(text)))
            except (TypeError, ValueError):
                pass
            return out
        mpl_axes.Axes.plot, mpl_axes.Axes.fill_between, mpl_axes.Axes.annotate = plot, fill_between, annotate
        mpl_figure.Figure.savefig = lambda fig, *a, **k: None
        return self

    def __exit__(self, *exc):
        import matplotlib.axes as mpl_axes
        import matplotlib.figure as mpl_figure
        mpl_axes.Axes.plot, mpl_axes.Axes.fill_between, mpl_axes.Axes.annotate = self._plot, self._fill, self._annotate
        mpl_figure.Figure.savefig = self._save
        return False

    def arrays(self):
        out, figs, axes_idx, count = {}, [], {}, {}

        def place(ax, kind):
            fig = ax.figure
            if id(fig) not in [id(f) for f in figs]:
                figs.append(fig)
            fi = [id(f) for f in figs].index(id(fig))
            ai = axes_idx.setdefault(id(ax), fig.axes.index(ax))
            n = count.get((fi, ai, kind), 0)
            count[(fi, ai, kind)] = n + 1
            return f"{fig_prefix(fi)}{ai}.{kind}{n}"
        for ax, x, y, label in self.calls:
            key = place(ax, "")
            x, y = thin(x, y)
            out[f"{key}:t"], out[f"{key}:y"], out[f"{key}:label"] = x, y, np.array(label)
        for ax, x, y, label in self.fills:
            key = place(ax, "f")
            x, y = thin(x, y)
            out[f"{key}:t"], out[f"{key}:y"], out[f"{key}:label"] = x, y, np.array(label)
        for ax, x, text in self.notes:
            key = place(ax, "a")
            out[f"{key}:x"], out[f"{key}:label"] = np.array([x]), np.array(text)
        return out


class BandDrawer:
    """While active, draws a ``fill_between`` of the band behind every line that has one, and writes a footnote under the first figure when it is saved.

    *band* is ``{key: dict(t, lo, hi, ...)}`` as ``Prediction_ensemble.build_band`` returns it (or ``read_band``); lines whose band is narrower than *tiny* times
    the range of the line (data, fixed quantities) are left alone. *edges*: axes prefixes (``"M0"`` is the first axes of the first figure) whose bands are also
    outlined by two very thin lines on top of the curve (white on a dark curve, black on a light one), for panels where the band is narrower than the curve's own
    width and the shading is hidden by it. ``offcenter`` lists, per banded line, how far the line being drawn lies from the optimum curve the band was built around.
    """

    def __init__(self, band, note=None, tiny=1e-9, edges=()):
        self.band, self.note, self.tiny, self.edges = band, note, tiny, set(edges)
        self.drawn, self.edged = 0, 0
        self.offcenter = []      # (key, largest |drawn line - the band's optimum curve| / the optimum's range): the band is about the optimum, not the line drawn

    def __enter__(self):
        import matplotlib.axes as mpl_axes
        import matplotlib.figure as mpl_figure
        self._plot, self._save = mpl_axes.Axes.plot, mpl_figure.Figure.savefig
        self._fill, self._annotate = mpl_axes.Axes.fill_between, mpl_axes.Axes.annotate
        self.figs, self.axes_idx, self.count = [], {}, {}
        me = self

        def place(ax, kind):
            fig = ax.figure
            if id(fig) not in [id(f) for f in me.figs]:
                me.figs.append(fig)
            fi = [id(f) for f in me.figs].index(id(fig))
            ai = me.axes_idx.setdefault(id(ax), fig.axes.index(ax))
            n = me.count.get((fi, ai, kind), 0)
            me.count[(fi, ai, kind)] = n + 1
            return f"{fig_prefix(fi)}{ai}.{kind}{n}"

        def varies(b):
            span = float(np.ptp(b["opt"])) or 1.0
            return float(np.max(b["hi"] - b["lo"])) > me.tiny * span

        def plot(ax, *args, **kw):
            lines = me._plot(ax, *args, **kw)
            for ln in lines:
                key = place(ax, "")
                b = me.band.get(key)
                if b is None or not varies(b):
                    continue
                me._fill(ax, b["t"], b["lo"], b["hi"], color=ln.get_color(), alpha=0.25, lw=0, zorder=max(ln.get_zorder() - 1, 0))
                me.drawn += 1
                try:
                    dev = float(np.max(np.abs(np.interp(b["t"], np.asarray(ln.get_xdata(), float), np.asarray(ln.get_ydata(), float)) - b["opt"])))
                    me.offcenter.append((key, dev / (float(np.ptp(b["opt"])) or 1.0)))
                except (TypeError, ValueError):
                    pass
                if key.split(".")[0] in me.edges:
                    from matplotlib.colors import to_rgb
                    r, g, bl = to_rgb(ln.get_color())
                    edge = "white" if 0.299 * r + 0.587 * g + 0.114 * bl < 0.55 else "black"
                    for edge_y in (b["lo"], b["hi"]):
                        me._plot(ax, b["t"], edge_y, color=edge, lw=0.5, zorder=ln.get_zorder() + 0.5, solid_capstyle="butt")
                    me.edged += 1
            return lines

        def fill_between(ax, x, y1, y2=0, *args, **kw):
            out = me._fill(ax, x, y1, y2, *args, **kw)
            try:
                xs, top = np.asarray(x, float), np.asarray(y2, float)
                recorded = xs.ndim == 1 and top.shape == xs.shape and len(xs) >= 20
            except (TypeError, ValueError):
                recorded = False
            if recorded:
                b = me.band.get(place(ax, "f"))
                if b is not None and varies(b):                      # a stacked area: the band of its upper boundary, hatched over the area
                    me._fill(ax, b["t"], b["lo"], b["hi"], facecolor="none", edgecolor="0.1", hatch="////", lw=0.0, alpha=0.7, zorder=out.get_zorder() + 1)
                    me.drawn += 1
            return out

        def annotate(ax, text, xy, *args, **kw):
            place(ax, "a")                                           # keeps the keys of the later annotations in step with the recorder's
            return me._annotate(ax, text, xy, *args, **kw)

        def savefig(fig, *a, **k):
            if me.note and me.figs and fig is me.figs[0]:
                fig.text(0.5, 0.004, "\n".join(textwrap.wrap(me.note, 110)), ha="center", va="bottom", fontsize=11, color="0.25")
            return me._save(fig, *a, **k)
        mpl_axes.Axes.plot, mpl_axes.Axes.fill_between, mpl_axes.Axes.annotate, mpl_figure.Figure.savefig = plot, fill_between, annotate, savefig
        return self

    def __exit__(self, *exc):
        import matplotlib.axes as mpl_axes
        import matplotlib.figure as mpl_figure
        mpl_axes.Axes.plot, mpl_axes.Axes.fill_between, mpl_axes.Axes.annotate, mpl_figure.Figure.savefig = self._plot, self._fill, self._annotate, self._save
        return False


def record_member(out_dir, member, original, extra=None):
    """A replacement for a plot function that runs *original* under a :class:`LineRecorder` and saves the member's lines (and the arrays of *extra*) as
    ``member_<id>.npz`` in *out_dir*, atomically."""
    def recorder(paths, results_dict, *a, **k):
        with LineRecorder() as rec:
            original(paths, results_dict, *a, **k)
        tmp = os.path.join(out_dir, f"member_{member['id']}.tmp.npz")
        np.savez_compressed(tmp, **rec.arrays(), **(extra or {}))
        os.replace(tmp, os.path.join(out_dir, f"member_{member['id']}.npz"))
    return recorder

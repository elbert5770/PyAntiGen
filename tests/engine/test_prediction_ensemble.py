"""Prediction ensemble: members from stored profile records, the envelope band, the delta-method cross-check, and the line recorder and band drawer.

``pyantigen.engine.Prediction_ensemble`` (pure numpy) and ``pyantigen.engine.Prediction_record`` (matplotlib wrappers). No model is simulated here.
"""
import numpy as np
import pytest

pytest.importorskip("pyantigen.engine.Profile")
from pyantigen.engine import Prediction_ensemble as PE  # noqa: E402
from pyantigen.engine import Prediction_record as PR  # noqa: E402
from pyantigen.engine.Profile_record import PointRecord  # noqa: E402

NAMES = ["a", "b", "c"]
SCALES = ["log10", "lin", "log10"]

def rec(param, xk, dnll, x, verdict="converged"):
    return PointRecord(param=param, x_fixed=xk, nll=dnll, dnll=dnll, verdict=verdict, param_idx=NAMES.index(param), x=list(x), source="walk")


def test_values_are_linear_for_log10_parameters_and_raw_otherwise():
    v = PE.to_values(NAMES, SCALES, [2.0, -1.5, -3.0])
    assert v == {"a": pytest.approx(100.0), "b": -1.5, "c": pytest.approx(1e-3)}


def test_members_are_the_points_inside_the_level_with_the_optimum_first_and_duplicates_dropped():
    by_side = {
        ("a", "up"): [rec("a", 0.0, 0.0, [0.0, 0.0, 0.0]),                      # the optimum again: a duplicate
                      rec("a", 0.1, 0.5, [0.1, 0.1, 0.0]),
                      rec("a", 0.2, 1.8, [0.2, 0.2, 0.1], verdict="not_converged"),   # unconverged but inside: kept, flagged
                      rec("a", 0.3, 2.5, [0.3, 0.3, 0.2])],                     # outside the level
        ("b", "down"): [rec("b", -0.1, 1.0, [0.1, -0.1, 0.0]),                  # the same vector as a's 0.1 point? no: differs in b
                        rec("b", -0.1, 1.2, [0.1, -0.1, 0.0]),                  # same x_fixed: the lower dnll is kept, one member
                        rec("b", -0.5, 1.9, [0.4, -0.5, 0.3])],
    }
    members = PE.extract_members(by_side, NAMES, SCALES, [0.0, 0.0, 0.0], level=1.9207)
    assert members[0]["id"] == 0 and members[0]["param"] == "optimum"
    keyed = [(m["param"], m["x_fixed"], m["dnll"]) for m in members[1:]]
    assert ("a", 0.1, 0.5) in keyed and ("a", 0.2, 1.8) in keyed and ("b", -0.1, 1.0) in keyed and ("b", -0.5, 1.9) in keyed
    assert not any(m["dnll"] > 1.9207 for m in members) and len(members) == 5
    assert next(m for m in members if m["x_fixed"] == 0.2)["verdict"] == "not_converged"
    assert [m["id"] for m in members] == list(range(len(members)))


def write_member(tmp_path, i, a, t=None):
    t = np.linspace(0, 48, 49) if t is None else t
    np.savez_compressed(tmp_path / f"member_{i}.npz", **{"a:40:t": t, "a:40:y": a * np.sin(t / 10.0) ** 2 + 0.01 * i})


def test_the_band_is_the_envelope_over_members_with_the_optimum_kept(tmp_path):
    for i, a in enumerate([1.0, 0.8, 1.3, 1.1]):
        write_member(tmp_path, i, a)
    files = [str(tmp_path / f"member_{i}.npz") for i in range(4)]
    b = PE.build_band(files)["a:40"]
    t = b["t"]
    Y = np.array([a * np.sin(t / 10.0) ** 2 + 0.01 * i for i, a in enumerate([1.0, 0.8, 1.3, 1.1])])
    assert b["n"] == 4 and np.allclose(b["lo"], Y.min(axis=0)) and np.allclose(b["hi"], Y.max(axis=0)) and np.allclose(b["opt"], Y[0])
    assert b["peak_lo"] == pytest.approx(Y.max(axis=1).min()) and b["peak_hi"] == pytest.approx(Y.max(axis=1).max())
    assert [n for n, _ in b["growth"]] == [1, 2, 4] and b["growth"][0][1] == 0.0
    assert all(g2 >= g1 - 1e-12 for (_, g1), (_, g2) in zip(b["growth"], b["growth"][1:]))     # the band only widens as members are added


def test_a_member_on_another_time_grid_is_interpolated_onto_the_first(tmp_path):
    write_member(tmp_path, 0, 1.0)
    write_member(tmp_path, 1, 1.2, t=np.linspace(0, 48, 97))
    b = PE.build_band([str(tmp_path / "member_0.npz"), str(tmp_path / "member_1.npz")])["a:40"]
    assert len(b["t"]) == 49 and np.all(b["hi"] >= b["lo"])


# --- generic line recording and band drawing ------------------------------------------------------------------------------

def figure_with_two_axes(scale=1.0, wobble=0.0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = np.linspace(0, 10, 21)
    fig, (a, b) = plt.subplots(1, 2)
    a.plot(t, scale * np.sin(t), label="model")                    # changes with the parameters
    a.plot(t, np.cos(t), label="fixed")                            # does not
    b.plot(t, scale * t + wobble, label="line b")
    return fig


def test_recorded_lines_are_keyed_by_figure_axes_and_line_and_savefig_is_silenced(tmp_path):
    import matplotlib.pyplot as plt
    with PR.LineRecorder() as rec:
        fig = figure_with_two_axes(1.0)
        fig.savefig(str(tmp_path / "never.png"))
        plt.close(fig)
    arrays = rec.arrays()
    assert not (tmp_path / "never.png").exists()                                       # a member's figure is not written
    keys = sorted({k.rsplit(":", 1)[0] for k in arrays})
    assert keys == ["M0.0", "M0.1", "M1.0"]
    assert str(arrays["M0.0:label"]) == "model" and arrays["M1.0:t"].shape == (21,)
    import matplotlib.axes as mpl_axes
    assert mpl_axes.Axes.plot.__name__ == "plot" and "LineRecorder" not in repr(mpl_axes.Axes.plot)   # the patch is removed on exit


def test_a_band_is_drawn_behind_the_lines_that_vary_and_not_behind_the_others(tmp_path):
    import matplotlib.pyplot as plt
    files = []
    for i, (scale, wob) in enumerate([(1.0, 0.0), (1.2, 0.3), (0.9, -0.2)]):
        with PR.LineRecorder() as rec:
            fig = figure_with_two_axes(scale, wob)
            plt.close(fig)
        np.savez_compressed(tmp_path / f"member_{i}.npz", **rec.arrays())
        files.append(str(tmp_path / f"member_{i}.npz"))
    band = PE.build_band(files)
    assert set(band) == {"M0.0", "M0.1", "M1.0"} and band["M0.0"]["label"] == "model"
    rows = PE.band_table(band)
    assert [r[0] for r in rows] == sorted([r[0] for r in rows], key=lambda k: -dict((r[0], r[4]) for r in rows)[k]) and "M0.1" not in [r[0] for r in rows]
    PE.write_band(band, str(tmp_path / "band.npz"))
    back = PE.read_band(str(tmp_path / "band.npz"))
    with PR.BandDrawer(back, note="footnote") as drawer:
        fig = figure_with_two_axes(1.0)
        n_a, n_b = len(fig.axes[0].collections), len(fig.axes[1].collections)
        fig.savefig(str(tmp_path / "with_band.png"))
        plt.close(fig)
    assert drawer.drawn == 2 and n_a == 1 and n_b == 1                                # "model" and "line b" have bands, "fixed" does not
    assert (tmp_path / "with_band.png").exists()


def test_lines_of_dates_or_text_are_skipped_not_fatal():
    import matplotlib.pyplot as plt
    with PR.LineRecorder() as rec:
        fig, ax = plt.subplots()
        ax.plot(["a", "b", "c"], [1, 2, 3])
        ax.plot([0, 1, 2], [1, 2, 3])
        plt.close(fig)
    assert len({k.rsplit(":", 1)[0] for k in rec.arrays()}) >= 1


# --- the delta-method cross-check -----------------------------------------------------------------------------------------

def test_the_finite_difference_members_move_each_parameter_by_plus_and_minus_delta():
    ms = PE.fd_members(NAMES, SCALES, [1.0, -2.0, 0.5], delta=0.01)
    assert len(ms) == 1 + 2 * 3 and ms[0]["param"] == "optimum"
    assert ms[1]["values"]["a"] == pytest.approx(10 ** 1.01) and ms[2]["values"]["a"] == pytest.approx(10 ** 0.99)     # log10 parameter
    assert ms[3]["values"]["b"] == pytest.approx(-1.99) and ms[4]["values"]["b"] == pytest.approx(-2.01)              # linear parameter
    assert ms[1]["values"]["b"] == pytest.approx(-2.0) and [m["id"] for m in ms] == list(range(7))


def test_the_delta_band_is_z_times_the_standard_deviation_of_the_line_and_compares_with_the_ensemble(tmp_path):
    t = np.linspace(0, 10, 11)
    g_true = np.array([[1.0, 0.0, 0.5]] * len(t)) * t[:, None]                  # the line is y = g(t) . theta, so central differences are exact
    delta = 0.01
    files = {0: tmp_path / "member_0.npz"}
    np.savez_compressed(files[0], **{"M0.0:t": t, "M0.0:y": np.zeros_like(t)})
    for j in range(3):
        for k, sign in ((1 + 2 * j, +1.0), (2 + 2 * j, -1.0)):
            files[k] = tmp_path / f"member_{k}.npz"
            np.savez_compressed(files[k], **{"M0.0:t": t, "M0.0:y": sign * delta * g_true[:, j]})
    cov = np.diag([0.04, 0.01, 0.09])
    d = PE.delta_band({k: str(v) for k, v in files.items()}, 3, delta, cov)["M0.0"]
    sd = np.sqrt((g_true ** 2) @ np.diag(cov))
    assert np.allclose(d["hi"] - d["opt"], 1.96 * sd) and np.allclose(d["opt"] - d["lo"], 1.96 * sd)
    ens = {"M0.0": {"label": "x", "t": t, "opt": np.zeros_like(t), "lo": -0.5 * 1.96 * sd, "hi": 0.5 * 1.96 * sd}}
    row = PE.compare_bands(ens, {"M0.0": d})[0]
    assert row[4] == pytest.approx(2.0) and row[5] == 1.0                      # the delta band is twice as wide, and contains the ensemble band


def test_the_third_and_later_figures_get_their_own_key_prefix_and_keys_sort_in_figure_order(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figs = [plt.figure() for _ in range(4)]
    with PR.LineRecorder() as rec:
        for f in figs:
            f.add_subplot(1, 2, 1).plot([0, 1], [0, 1])
        # the 12th axes of the last figure: "F3_11.0" must not be read as figure F31, axes 1
        axs = [figs[3].add_subplot(4, 4, i + 3) for i in range(11)]
        axs[10].plot([0, 1], [1, 2])
    keys = sorted({k.rsplit(":", 1)[0] for k in rec.arrays()})
    assert keys == ["F2_0.0", "F3_0.0", "F3_11.0", "M0.0", "S0.0"], keys
    f = tmp_path / "m.npz"
    np.savez(f, **rec.arrays())
    assert PE.curve_keys([str(f)]) == ["F2_0.0", "F3_0.0", "F3_11.0", "M0.0", "S0.0"]
    for f_ in figs:
        plt.close(f_)


def _stack_figure(scale, age):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    t = np.linspace(0, 10, 100)
    ax.fill_between(t, 0, scale * t, label="lower")
    ax.fill_between(t, scale * t, scale * t + 1.0 * t, label="upper")           # a stacked area: the upper boundary of each is recorded
    ax.plot(t, scale * t + 5, color="k")
    ax.plot([age, age], [0, 3], color="r")                                      # a two-point reference line
    ax.annotate(f"{age:.1f}", xy=(age, 0), xycoords="data", xytext=(0, -9), textcoords="offset points")
    return fig


def test_fills_annotations_and_long_lines_are_recorded_with_their_own_keys_and_thinned(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with PR.LineRecorder() as rec:
        fig = _stack_figure(2.0, 4.0)
        ax = fig.axes[0]
        ax.plot(np.linspace(0, 1, 100000), np.linspace(0, 1, 100000) ** 2)
    arr = rec.arrays()
    keys = {k.rsplit(":", 1)[0] for k in arr}
    assert keys == {"M0.0", "M0.1", "M0.2", "M0.f0", "M0.f1", "M0.a0"}, keys      # lines 0 (data), 1 (two points), 2 (the 100,000 point curve)
    assert len(arr["M0.2:t"]) <= PR.MAXPTS and arr["M0.2:t"][0] == 0 and arr["M0.2:t"][-1] == 1
    assert arr["M0.f1:y"][-1] == pytest.approx(3.0 * 10)                           # the upper boundary of the second area
    assert float(arr["M0.a0:x"][0]) == 4.0 and str(arr["M0.a0:label"]) == "4.0"
    plt.close(fig)


def test_bands_skip_two_point_lines_and_cover_fill_boundaries_and_annotation_ranges_are_reported(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for i, (scale, age) in enumerate([(2.0, 4.0), (2.5, 4.6), (1.8, 3.9)]):
        with PR.LineRecorder() as rec:
            fig = _stack_figure(scale, age)
        np.savez(tmp_path / f"member_{i}.npz", **rec.arrays())
        plt.close(fig)
    files = [str(tmp_path / f"member_{i}.npz") for i in range(3)]
    band = PE.build_band(files)
    assert "M0.1" not in band and "M0.0" in band and "M0.f1" in band               # the two-point line has no band
    t = band["M0.f1"]["t"]
    assert band["M0.f1"]["hi"][-1] == pytest.approx(3.5 * 10) and band["M0.f1"]["lo"][-1] == pytest.approx(2.8 * 10)
    notes = PE.annotation_ranges(files)
    assert notes["M0.a0"]["x0"] == 4.0 and notes["M0.a0"]["lo"] == 3.9 and notes["M0.a0"]["hi"] == 4.6 and notes["M0.a0"]["n"] == 3


def test_the_drawer_hatches_a_band_over_a_stacked_area_and_keeps_its_own_fills_out_of_the_numbering(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    members = []
    for i, (scale, age) in enumerate([(2.0, 4.0), (2.5, 4.6)]):
        with PR.LineRecorder() as rec:
            fig = _stack_figure(scale, age)
        np.savez(tmp_path / f"member_{i}.npz", **rec.arrays())
        plt.close(fig)
    band = PE.build_band([str(tmp_path / "member_0.npz"), str(tmp_path / "member_1.npz")])
    with PR.BandDrawer(band) as drawer:
        fig = _stack_figure(2.0, 4.0)
    # bands behind the curve M0.0 and the two stacked boundaries M0.f0 and M0.f1
    assert drawer.drawn == 3
    plt.close(fig)


def test_edge_lines_are_drawn_only_for_the_chosen_axes_and_their_colour_contrasts_with_the_curve(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def figure():
        fig, axes = plt.subplots(1, 2)
        t = np.linspace(0, 1, 50)
        axes[0].plot(t, t, color="darkred", lw=4)
        axes[1].plot(t, t, color="gold", lw=4)
        return fig, axes
    members = []
    for i, s in enumerate((1.0, 1.1)):
        with PR.LineRecorder() as rec:
            fig, _ = figure()
            for ax in fig.axes:                                          # varying curves in both axes
                for ln in list(ax.lines):
                    ln.set_ydata(ln.get_ydata() * s)
        t = np.linspace(0, 1, 50)
        np.savez(tmp_path / f"member_{i}.npz", **{"M0.0:t": t, "M0.0:y": t * s, "M1.0:t": t, "M1.0:y": t * s})
        plt.close(fig)
    band = PE.build_band([str(tmp_path / "member_0.npz"), str(tmp_path / "member_1.npz")])
    with PR.BandDrawer(band, edges=["M0", "M1"]) as d:
        fig, axes = figure()
    assert d.drawn == 2 and d.edged == 2
    colours = [[ln.get_color() for ln in ax.lines] for ax in axes]
    assert colours[0].count("white") == 2 and colours[1].count("black") == 2      # white on dark red, black on gold
    plt.close(fig)
    with PR.BandDrawer(band, edges=["M0"]) as d:
        fig, axes = figure()
    assert d.drawn == 2 and d.edged == 1 and len(axes[1].lines) == 1
    plt.close(fig)


def test_the_drawer_reports_a_line_that_is_not_the_optimum_the_band_was_built_around(tmp_path):
    import matplotlib.pyplot as plt
    files = []
    for i, scale in enumerate((1.0, 1.2, 0.9)):
        with PR.LineRecorder() as rec:
            fig = figure_with_two_axes(scale)
            plt.close(fig)
        np.savez_compressed(tmp_path / f"member_{i}.npz", **rec.arrays())
        files.append(str(tmp_path / f"member_{i}.npz"))
    band = PE.build_band(files)
    with PR.BandDrawer(band) as same:                        # member 0's own curve: nothing to report
        plt.close(figure_with_two_axes(1.0))
    assert same.drawn == 2 and max(d for _, d in same.offcenter) < 1e-12
    with PR.BandDrawer(band) as other:                       # a line at other parameter values
        plt.close(figure_with_two_axes(1.5))
    assert max(d for k, d in other.offcenter if k == "M0.0") > 0.1

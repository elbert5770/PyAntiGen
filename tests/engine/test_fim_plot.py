"""Fim_plot: the interval read from profile points, the parameter order, the weakest directions, and a figure that gets written."""
import json
import os

import numpy as np
import pytest

from pyantigen.engine import Fim_plot as FP

pytest.importorskip("matplotlib")


def _quad(mle, curv, xs):
    return [(x, 0.5 * curv * (x - mle) ** 2, True) for x in xs]


def test_a_crossed_side_is_interpolated_at_the_cut_and_an_open_side_ends_at_its_last_point():
    pts = _quad(1.0, 2.0, [0.0, 0.5, 1.5, 2.0, 3.0])                # dnll = (x - 1)^2
    ext = FP.profile_extent(pts, 1.0, thr=1.9207)
    assert ext["high"][1] is False and abs(ext["high"][0] - (2.0 + (1.9207 - 1.0) / (4.0 - 1.0))) < 1e-9     # between 2.0 and 3.0
    assert ext["low"] == (0.0, True)                                  # 0.0 is at dnll 1.0: never over the cut
    short = [q for q in pts if q[0] <= 1.5]
    assert FP.profile_extent(short, 1.0)["high"] == (1.5, True)


def test_unconverged_points_do_not_count_and_a_side_with_none_is_open_with_no_extent():
    pts = [(1.5, 0.3, True), (2.0, 5.0, False)]                  # the point over the cut did not converge: it is only an upper bound
    ext = FP.profile_extent(pts, 1.0)
    assert ext["high"] == (1.5, True) and ext["low"] == (None, True)


def test_cluster_order_is_a_permutation_that_keeps_correlated_parameters_together():
    corr = np.eye(5)
    corr[0, 3] = corr[3, 0] = 0.95
    corr[1, 4] = corr[4, 1] = -0.9
    order = FP.cluster_order(corr)
    assert sorted(order) == list(range(5))
    pos = {p: i for i, p in enumerate(order)}
    assert abs(pos[0] - pos[3]) == 1 and abs(pos[1] - pos[4]) == 1


def test_weakest_directions_are_the_smallest_eigenvalues_with_a_positive_largest_loading():
    F = np.diag([100.0, 1.0, 10.0])
    F[0, 1] = F[1, 0] = 0.5
    lam, V = FP.weakest_directions(F, k=2)
    assert lam[0] < lam[1] < 10.0 + 1e-6 and V.shape == (2, 3)
    assert all(v[int(np.argmax(np.abs(v)))] > 0 for v in V) and np.allclose(np.linalg.norm(V, axis=1), 1.0)


def _fixture(tmp_path):
    names, scales, x = ["a", "b", "c"], ["log10", "log10", "lin"], np.array([0.0, 1.0, 2.0])
    F = np.array([[50.0, 40.0, 0.0], [40.0, 50.0, 0.0], [0.0, 0.0, 9.0]])
    pdir = tmp_path / "SILK" / "profiles" / "run" / "amici"
    (pdir / "fim").mkdir(parents=True)
    np.savez(pdir / "fim" / "fim.npz", names=names, scales=scales, x=x, F=F)
    for k, n in enumerate(names):
        pts = _quad(x[k], 8.0 if n != "c" else 2.0, [x[k] + d for d in (-1.0, -0.5, -0.2, 0.2, 0.5, 1.0)])
        with open(pdir / f"profile_{k}_up.jsonl", "w") as fh:
            for xx, d, _c in pts:
                fh.write(json.dumps({"param": n, "x_fixed": xx, "nll": d, "dnll": d, "evals": 5, "verdict": "converged"}) + "\n")
    return pdir, names, scales, x, F


def test_the_figure_is_written_with_and_without_profile_points(tmp_path):
    pdir, names, scales, x, F = _fixture(tmp_path)
    out = FP.draw(str(tmp_path / "a.png"), "t", names, scales, x, F, {n: [(1.0, 0.0, True)] for n in names})
    bare = FP.draw(str(tmp_path / "b.png"), "t", names, scales, x, F, None)
    assert os.path.getsize(out) > 5000 and os.path.getsize(bare) > 5000


def test_the_program_reads_the_folders_and_names_the_file(tmp_path, capsys):
    pdir, *_ = _fixture(tmp_path)
    assert FP.main([str(pdir), "--no-timestamp"]) == 0
    assert (tmp_path / "SILK" / "SILK_run_fim.png").stat().st_size > 5000
    with pytest.raises(SystemExit):
        FP.main([str(tmp_path)])                                  # no fim.npz there

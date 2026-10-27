"""Fim: the Fisher information of a linear residual model, and the AMICI stage's fim mode."""
import numpy as np
import pytest

from pyantigen.engine import Fim
from pyantigen.engine import Profile_amici as PA


def test_fisher_information_of_a_linear_model():
    rng = np.random.default_rng(1)
    J = rng.normal(size=(40, 3))
    x = np.array([-1.0, 0.5, 2.0])
    rep = Fim.fim_report(J, ["a", "b", "c"], ["log10", "log10", "lin"], x)
    assert np.allclose(rep["F"], J.T @ J)
    assert rep["se"] == pytest.approx(np.sqrt(np.diag(np.linalg.inv(J.T @ J))))
    assert rep["value"][0] == pytest.approx(0.1) and rep["value"][2] == 2.0
    assert rep["ci_lo"][0] == pytest.approx(10 ** (-1.0 - Fim.Z95 * rep["se"][0]))   # a log10 interval is multiplicative
    assert rep["ci_hi"][2] == pytest.approx(2.0 + Fim.Z95 * rep["se"][2])
    assert rep["rank"] == 3 and rep["n_not_identifiable"] == 0


def test_a_degenerate_direction_is_flagged_and_loaded():
    rng = np.random.default_rng(2)
    base = rng.normal(size=(30, 2))
    J = np.column_stack([base[:, 0], base[:, 1], base[:, 0] + 1e-6 * rng.normal(size=30)])   # c ~ a
    rep = Fim.fim_report(J, ["a", "b", "c"], ["lin"] * 3, np.zeros(3))
    assert rep["n_not_identifiable"] == 1
    top = {n for n, _ in rep["weak"][0]["loadings"][:2]}
    assert top == {"a", "c"}


def test_fim_mode_writes_the_report_and_walks_nothing(tmp_path):
    class Provider:
        L = np.array([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]])

        def linear_residual_jacobian(self, p):
            return self.L @ np.asarray(p, float) - 1.0, self.L.copy(), 0.0

    store = str(tmp_path / "amici")
    lines = []
    out = PA.run(Provider(), ["a", "b"], [(-5.0, 5.0)] * 2, ["lin", "lin"], [0.1, 0.2], store, mode="fim", log=lines.append)
    assert out["ran"] == [] and out["fim"].endswith("fim.txt")
    assert (tmp_path / "amici" / "fim" / "fim.npz").exists()
    assert any(l.startswith("[fim] Fisher information") for l in lines)
    assert not [f for f in (tmp_path / "amici").iterdir() if f.name.endswith(".jsonl")]


def test_the_mode_is_accepted_from_the_environment():
    assert PA.settings_from_env({PA.MODE_ENV: "fim"})[0] == "fim"

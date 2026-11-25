"""The plateau stop of the nuisance LM of a profile point: a point that has bought almost nothing in the last window of evaluations while the
projected gradient is already small is stalled, and is classified so after the window instead of after the whole evaluation budget.

On the microglia profile (2026-11-21) the points on the ridge used their whole budget (100 evaluations, 2 to 3 hours) crawling at a projected
gradient of 0.02 to 0.05 nats, and three attempts reproduced the same dnll to three digits.
"""
import numpy as np

from pyantigen.engine import Profile as PR
from pyantigen.engine import Profile_lm as L
from pyantigen.engine import Profile_amici as PA
from pyantigen.engine import Profile_record as P


def test_plateau_reached_cases():
    hist = [(0, 1.0), (4, 0.9995), (9, 0.9992), (14, 0.99915)]
    reached, gained = L.plateau_reached(hist, evals=24, window=20, tol=0.005, pg=0.03, pg_max=0.05)
    assert reached and abs(gained - (0.9995 - 0.99915)) < 1e-12                            # the entry current at evals 4 is the reference
    assert L.plateau_reached(hist, evals=24, window=20, tol=0.0002, pg=0.03, pg_max=0.05) == (False, 0.9995 - 0.99915)      # still gaining more than the tolerance
    assert L.plateau_reached(hist, evals=24, window=20, tol=0.005, pg=0.2, pg_max=0.05)[0] is False                         # not near a stationary point: the budget ends it
    assert L.plateau_reached(hist, evals=15, window=20, tol=0.005, pg=0.03, pg_max=0.05) == (False, None)                   # no full window yet
    assert L.plateau_reached(hist, evals=24, window=0, tol=0.005, pg=0.03, pg_max=0.05) == (False, None)                    # off
    assert L.plateau_reached(hist, evals=24, window=20, tol=0.005, pg=None, pg_max=0.05) == (False, None)


def _crawl(counter):
    """A point whose objective gains 1e-6 per evaluation while the gradient the Jacobian reports stays at 0.03: it will never pass the 0.01 test."""
    def evaluate(x):
        counter[0] += 1
        return 1.0 - 1e-6 * counter[0], np.array([0.03]), np.array([[1.0]])
    return evaluate


def test_a_crawling_point_stops_after_the_window_as_stalled():
    n_off, n_on = [0], [0]
    off = L.minimize(_crawl(n_off), [0.0], [-9.0], [9.0], [0], L.LmSettings(max_evals=80))
    on = L.minimize(_crawl(n_on), [0.0], [-9.0], [9.0], [0], L.LmSettings(max_evals=80, plateau_window=10, plateau_tol=0.005, plateau_pg=0.05))
    assert on.verdict == "not_converged" and "no model decrease" in on.reason and "plateau" in on.reason
    assert on.evals <= 14 and off.evals > on.evals                                      # the window, not the budget
    rec = P.PointRecord(param="a", x_fixed=1.0, nll=1.0, dnll=0.2, verdict=on.verdict, pg=on.pg, reason=on.reason, param_idx=0)
    assert rec.stalled()                                                                # the classification of the engine's reports takes it as stalled


def test_a_point_that_is_still_gaining_is_not_stopped():
    cnt = [0]

    def evaluate(x):                                                                    # a real descent: dnll falls by 0.05 per evaluation, pg stays above the stall gradient
        cnt[0] += 1
        return 5.0 - 0.05 * cnt[0], np.array([0.5]), np.array([[1.0]])
    res = L.minimize(evaluate, [0.0], [-9.0], [9.0], [0], L.LmSettings(max_evals=30, plateau_window=10, plateau_tol=0.005, plateau_pg=0.05))
    assert "plateau" not in res.reason


def test_plateau_settings_come_from_the_environment():
    mode, st = PA.settings_from_env({})
    assert st.plateau_window == 0
    mode, st = PA.settings_from_env({"PYANTIGEN_PROFILE_PLATEAU": "20", "PYANTIGEN_PROFILE_PLATEAU_TOL": "0.01", "PYANTIGEN_PROFILE_PLATEAU_PG": "0.08"})
    assert (st.plateau_window, st.plateau_tol, st.plateau_pg) == (20, 0.01, 0.08)

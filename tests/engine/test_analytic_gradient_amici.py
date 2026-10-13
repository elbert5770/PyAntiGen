"""The engine's fit gradient from AMICI, end to end on a small model with moving switches.

Run from the repository root: python -m pytest tests/engine/test_analytic_gradient_amici.py
Skipped unless AMICI >= 1.1 and tellurium import; on Windows run it under WSL (there is no AMICI
wheel and no compiler there).

The model is the stand-in for the v4 SILK CSF draw of ``test_segmented.py`` with two more
parameters: ``k_y2`` (set by a hook from the fitted ``k_y``) and ``SF`` (named by the loss).
The replicate is built the way ``run_optimization_from_groups`` builds one -- a RoadRunner model,
its data, its solver settings with the schedule -- and the objective is the engine's own
(``evaluate_nll_fixed``: ``run_all`` per simulation, ``collect_loss_blocks``, ``concentrated_nll``).
The gradient AMICI delivers is compared with Richardson finite differences of that objective,
in optimizer space (log10 for two of the parameters), with the loss reading a log10 observable
and ``SF*z``, a hook between the fitted and the model parameters, and an input
(``Q_SN = Q_CSF*f_SN``) whose initial assignment RoadRunner does not re-derive.
"""
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("amici")
te = pytest.importorskip("tellurium")

from pyantigen.engine import Multistart as M  # noqa: E402
from pyantigen.engine.Analytic_gradient import AmiciGradient, build_gradient  # noqa: E402
from pyantigen.engine.Optimize import (  # noqa: E402
    _local_minimize, evaluate_nll_fixed, remember_parameter_baseline)
from pyantigen.generate.input_schedule import InputSchedule  # noqa: E402

PARAMS = {"Q_CSF": 17.39500698, "V_LP": 6.0, "t_CSFdraw": 0.1, "f_SN": 0.1}
NAMES = ["Q_CSF", "V_LP", "t_CSFdraw", "k_y", "SF"]
SCALES = ["log10", "lin", "lin", "log10", "lin"]
TRUTH = np.array([17.39500698, 6.0, 0.04, 0.3, 2.0])
MODEL = """
Q_CSF = 17.39500698; f_SN = 0.1; V_LP = 6; t_CSFdraw = 0.1; k_y = 0.3; k_y2 = 0.6; SF = 2
Q_g = Q_CSF
Q_Leak = 0; Q_LP = 0; Q_refill = 0; Q_SN = Q_CSF*f_SN
x_leak = 0; x_sn = 0; x_lp = 0; x_refill = 0; y = 1; z = 0; w = 0
x_leak' = Q_Leak; x_sn' = Q_SN; x_lp' = Q_LP; x_refill' = Q_refill
y' = 0.05*Q_LP - k_y2*y*(1 + Q_SN)
z' = Q_refill*y/(1 + z*z) - 0.2*z - 0.004*Q_g*z
w' = f_L - 0.4*w
"""
OBSERVED = ["time", "x_leak", "x_sn", "x_lp", "x_refill", "y", "z", "w", "Q_Leak", "Q_SN", "Q_LP", "Q_refill",
            "f_L"]
DATA_T = np.round(np.arange(0.25, 8.0, 0.25), 2)
# a box the schedule can be integrated in everywhere: V_LP/Q_CSF stays in (0.3, 0.65), above t_CSFdraw
BOX = [(10.0, 50.0), (3.0, 6.5), (0.02, 0.055), (0.05, 2.0), (0.5, 5.0)]
BOX_OPT = [(np.log10(lo), np.log10(hi)) if sc == "log10" else (lo, hi) for (lo, hi), sc in zip(BOX, SCALES)]


def schedule():
    s = InputSchedule(PARAMS)
    t_draw = s.moving("t_draw", "t_CSFdraw")
    t_refill = s.moving("t_refill", "V_LP/Q_CSF")
    s.step(0, {"Q_Leak": "15.0"})
    s.step(3, {"Q_Leak": "0.0"})
    s.step(5, {"Q_Leak": "Q_CSF - f_SN*Q_CSF"})
    s.periodic(0, 1, 8, [
        (0, {"Q_SN": "0.0", "Q_LP": "Q_CSF", "Q_refill": "Q_CSF - V_LP/t_CSFdraw"}),
        (t_draw, {"Q_refill": "Q_CSF"}),
        (t_refill, {"Q_SN": "f_SN*Q_CSF", "Q_LP": "Q_Leak", "Q_refill": "0"}),
    ])
    s.linear_input("f_L", [0, 1, 2.5, 4, 6.5], [0.1, 0.3, 0.2, 0.5, 0.1], default_before=0, default_after=0)
    return s


def _hook(r, experiment, parameters):
    """Update_opt_parameters: the model's decay constant is twice the fitted one."""
    if "k_y" in parameters:
        r["k_y2"] = 2.0 * parameters["k_y"]


LOSS = {"observables": [
    {"observed_variable": "np.log10(np.maximum(y, 1e-9))", "data_column": "ly", "time_column": "time",
     "data_dict_key": "toy"},
    {"observed_variable": "SF*z", "data_column": "zz", "time_column": "time", "data_dict_key": "toy"},
]}
GROUPS = {"fit": {"loss_elements": [{"simulation": "toy", "loss_config": lambda rep: LOSS, "weight": 1.0}]}}


PRE_RUN = {"pre": {"start": -5, "end": 0, "n_points": 51, "tracked": False},
           "run": {"start": 0, "end": 8, "n_points": 801, "tracked": True}}
RUN_ONLY = {"run": {"start": 0, "end": 8, "n_points": 801, "tracked": True}}


def make_problem(backend, data=None, blocks=PRE_RUN):
    """(models, replicates) as the engine would build them for one replicate."""
    sched = schedule()
    replicate = {"Label": "toy", "Update_parameters": lambda r, rep: None, "Update_opt_parameters": _hook,
                 "Observed_species": lambda r: list(OBSERVED)}

    def settings(rep):
        out = {"integrator": "cvode", "absolute_tolerance": 1e-12, "relative_tolerance": 1e-12,
               "stiff": True, "variable_step_size": True,
               "simulation_blocks": blocks}
        if rep.get("Schedule") is not None:
            out["schedule"] = rep["Schedule"]
        return out

    replicate["Solver_settings"] = settings
    if backend == "schedule":
        replicate["Schedule"] = sched
    events = sched.to_antimony(events=(backend == "events"))
    r = te.loada(MODEL + events)
    remember_parameter_baseline(r)
    df = data if data is not None else make_data()
    return ({"toy": {"r": r, "df_dict": {"toy": df}, "events": events}}, {"toy": replicate}, MODEL)


def make_data():
    """The loss's data: the model at TRUTH on the output grid, 2% noise."""
    models, reps, _ = make_problem_nodata()
    from pyantigen.engine.Optimize import run_all, set_parameters_from_dict
    p = dict(zip(NAMES, TRUTH.tolist()))
    out = run_all(models["toy"]["r"], "toy", reps["toy"], {}, set_parameters=set_parameters_from_dict,
                  parameters=p)["toy"]["results"]
    t = np.asarray(out["time"])
    rows = [int(np.argmin(np.abs(t - d))) for d in DATA_T]
    rng = np.random.default_rng(11)
    ly = np.log10(np.asarray(out["y"])[rows]) + 0.01 * rng.standard_normal(len(rows))
    zz = 2.0 * np.asarray(out["z"])[rows] * (1 + 0.02 * rng.standard_normal(len(rows)))
    return pd.DataFrame({"time": DATA_T, "ly": ly, "zz": zz})


def make_problem_nodata():
    sched = schedule()
    replicate = {"Label": "toy", "Update_parameters": lambda r, rep: None, "Update_opt_parameters": _hook,
                 "Observed_species": lambda r: list(OBSERVED), "Schedule": sched,
                 "Solver_settings": lambda rep: {
                     "integrator": "cvode", "absolute_tolerance": 1e-12, "relative_tolerance": 1e-12,
                     "stiff": True, "variable_step_size": True, "schedule": rep["Schedule"],
                     "simulation_blocks": {"pre": {"start": -5, "end": 0, "n_points": 51, "tracked": False},
                                           "run": {"start": 0, "end": 8, "n_points": 801, "tracked": True}}}}
    r = te.loada(MODEL + sched.to_antimony(events=False))
    remember_parameter_baseline(r)
    return {"toy": {"r": r, "df_dict": {}, "events": ""}}, {"toy": replicate}, MODEL


def engine_objective(models, reps):
    """The engine's own concentrated NLL as a function of the optimizer-space vector."""
    def f(x):
        return evaluate_nll_fixed(np.asarray(x, float), models, reps, NAMES, SCALES, GROUPS, "mean_over_groups",
                                  None, concentrated=True, for_inference=False)
    return f


def richardson(f, x, rel=2e-4):
    g = np.zeros(len(x))
    for k in range(len(x)):
        h = rel * max(abs(x[k]), 1.0)

        def c(s):
            up, dn = x.copy(), x.copy()
            up[k] += s
            dn[k] -= s
            return (f(up) - f(dn)) / (2 * s)
        g[k] = (4 * c(h / 2) - c(h)) / 3
    return g


def to_opt(p_lin):
    return np.array([np.log10(v) if s == "log10" else v for v, s in zip(p_lin, SCALES)])


@pytest.fixture(scope="module")
def schedule_problem():
    return make_problem("schedule", make_data())


@pytest.fixture(scope="module")
def provider(schedule_problem, tmp_path_factory):
    models, reps, text = schedule_problem
    bounds = BOX
    return build_gradient(
        {"analytic_gradient": {"cache_dir": str(tmp_path_factory.mktemp("amici_models")), "verbose": False}},
        models=models, replicates=reps, groups=GROUPS, param_names=NAMES, scales=SCALES,
        x0_lin=TRUTH * np.array([1.2, 1.0, 1.0, 1.3, 1.1]), bounds_lin=bounds, model_text=text, paths={})


@pytest.mark.parametrize("scale", [
    np.array([1.2, 1.0, 1.0, 1.3, 1.1]),            # near the truth
    np.array([0.7, 0.9, 1.6, 0.8, 0.7]),            # elsewhere in the box, switch times moved
    np.array([2.5, 1.1, 0.7, 1.0, 1.0]),            # Q_CSF large: the refill close to the draw
])
def test_the_gradient_matches_finite_differences_of_the_engine_objective(schedule_problem, provider, scale):
    models, reps, _ = schedule_problem
    f = engine_objective(models, reps)
    x = to_opt(TRUTH * scale)
    g = provider(x)
    ref = richardson(f, x)
    assert np.all(np.isfinite(ref)) and np.abs(ref).max() > 1e-3
    np.testing.assert_allclose(g, ref, rtol=2e-4, atol=2e-4 * np.abs(ref).max())


def test_every_parameter_contributes(schedule_problem, provider):
    # the moving switches (Q_CSF, V_LP, t_CSFdraw), the hooked decay constant and the named SF all matter
    x = to_opt(TRUTH * np.array([1.2, 1.1, 1.3, 1.3, 1.1]))
    g = provider(x)
    assert np.all(np.abs(g) > 1e-3 * np.abs(g).max())


def test_the_value_on_amicis_simulation_is_the_engines_value(schedule_problem, provider):
    models, reps, _ = schedule_problem
    x = to_opt(TRUTH * np.array([1.2, 1.0, 1.0, 1.3, 1.1]))
    f_amici, _g = provider.linear_gradient(np.array([10 ** v if s == "log10" else v for v, s in zip(x, SCALES)]))
    assert f_amici == pytest.approx(engine_objective(models, reps)(x), rel=1e-6, abs=1e-6)


def test_events_in_the_replicate_are_read_as_a_schedule_and_give_the_same_gradient(
        schedule_problem, provider, tmp_path):
    ev_models, ev_reps, text = make_problem("events", schedule_problem[0]["toy"]["df_dict"]["toy"])
    bounds = BOX
    ev = build_gradient({"analytic_gradient": {"cache_dir": str(tmp_path), "verbose": False}},
                        models=ev_models, replicates=ev_reps, groups=GROUPS, param_names=NAMES, scales=SCALES,
                        x0_lin=TRUTH * np.array([1.2, 1.0, 1.0, 1.3, 1.1]), bounds_lin=bounds, model_text=MODEL,
                        paths={})
    x = to_opt(TRUTH * np.array([1.2, 1.0, 1.0, 1.3, 1.1]))
    np.testing.assert_allclose(ev(x), provider(x), rtol=1e-6, atol=1e-8)


def test_a_fit_with_the_analytic_gradient_reaches_the_optimum_in_few_evaluations(schedule_problem, provider):
    models, reps, _ = schedule_problem
    f = engine_objective(models, reps)
    start = to_opt(TRUTH * np.array([1.3, 1.1, 1.2, 1.4, 0.8]))
    bounds = BOX_OPT
    counted = []

    def counting(x):
        counted.append(1)
        return f(x)

    res = _local_minimize(counting, start, "L-BFGS-B", bounds, {"options": {"maxiter": 60}}, provider, SCALES)
    assert res.fun < f(start) - 1.0
    assert f(to_opt(TRUTH)) - 10.0 < res.fun < f(to_opt(TRUTH)) + 1.0      # at the generating point's level
    assert res.gradient["n_fallback"] == 0
    assert len(counted) <= 60 + 1                                           # one evaluation per iteration, not k + 1
    assert len(counted) < (len(NAMES) + 1) * res.nit                        # what forward differences would have cost


def test_when_amici_fails_the_fit_falls_back_to_finite_differences(schedule_problem, provider):
    models, reps, _ = schedule_problem
    f = engine_objective(models, reps)

    def broken(x):
        raise RuntimeError("AMICI: TOO_MUCH_WORK")

    start = to_opt(TRUTH * np.array([1.1, 1.0, 1.1, 1.2, 0.9]))
    bounds = BOX_OPT
    res = _local_minimize(f, start, "L-BFGS-B", bounds, {"options": {"maxiter": 8}}, broken, SCALES)
    assert res.fun < f(start)
    assert res.gradient["n_fallback"] >= 1 and "TOO_MUCH_WORK" in res.gradient["last_error"]


def test_a_fitted_parameter_that_reaches_the_model_nowhere_is_refused_up_front(schedule_problem, tmp_path):
    models, reps, text = schedule_problem
    with pytest.raises(M.GradientSetupError, match="reach no plain parameter"):
        build_gradient({"analytic_gradient": {"cache_dir": str(tmp_path), "verbose": False}},
                       models=models, replicates=reps, groups=GROUPS, param_names=NAMES + ["not_in_the_model"],
                       scales=SCALES + ["lin"], x0_lin=list(TRUTH) + [1.0], bounds_lin=None, model_text=text,
                       paths={})


def test_a_replicate_with_neither_schedule_nor_events_is_refused(schedule_problem, tmp_path):
    models, reps, text = schedule_problem
    rep = dict(reps["toy"])
    rep["Schedule"] = None
    bare = {"toy": {**models["toy"], "events": ""}}
    with pytest.raises(M.GradientSetupError, match="InputSchedule"):
        build_gradient({"analytic_gradient": {"cache_dir": str(tmp_path), "verbose": False}},
                       models=bare, replicates={"toy": rep}, groups=GROUPS, param_names=NAMES, scales=SCALES,
                       x0_lin=TRUTH, bounds_lin=None, model_text=text, paths={})


def test_an_input_whose_initial_assignment_roadrunner_does_not_rederive_follows_the_engine(provider):
    # Q_SN = Q_CSF*f_SN is a parameter initial assignment: reset() leaves it at the value it had when
    # the model was built, so with Q_CSF moved the engine integrates from the old Q_SN. The gradient
    # follows the engine (a number, no sensitivity) there, and the model's formula where they agree.
    sim = provider.sims["toy"]
    nominal = sim.replay.values(TRUTH)
    assert sim._engine_initial_inputs(nominal)["Q_SN"] == "Q_CSF * f_SN"
    moved = sim.replay.values(TRUTH * np.array([1.5, 1.0, 1.0, 1.0, 1.0]))
    stale = sim._engine_initial_inputs(moved)["Q_SN"]
    assert isinstance(stale, float) and stale == pytest.approx(TRUTH[0] * 0.1)


def test_a_parameter_defined_from_a_fitted_one_is_frozen_as_the_engine_holds_it(schedule_problem, provider):
    # Q_g = Q_CSF is a parameter initial assignment used by the dynamics: RoadRunner leaves it at the
    # value it had when the model was built, so the engine's objective does not move with it when Q_CSF
    # moves. AMICI is given the same model (Q_g made a plain parameter at the engine's value).
    sim = provider.sims["toy"]
    assert sim.frozen == ["Q_g"]
    assert "Q_g" in sim.replay.plain and "Q_g" not in sim.free


def test_without_freezing_the_gradient_is_that_of_a_different_model(schedule_problem, tmp_path, monkeypatch):
    # the negative control: leave Q_g = Q_CSF in the AMICI model and the gradient with respect to Q_CSF
    # gains a term (d Q_g/d Q_CSF) the engine's objective does not have
    import pyantigen.engine.Analytic_gradient as ag
    monkeypatch.setattr(ag, "freeze_stale_assignments", lambda sbml, r, protected=(), varying=(), everything=False: (sbml, []))
    models, reps, text = schedule_problem
    loose = build_gradient({"analytic_gradient": {"cache_dir": str(tmp_path), "verbose": False}},
                           models=models, replicates=reps, groups=GROUPS, param_names=NAMES, scales=SCALES,
                           x0_lin=TRUTH, bounds_lin=BOX, model_text=text, paths={})
    x = to_opt(TRUTH * np.array([1.4, 1.0, 1.0, 1.0, 1.0]))
    ref = richardson(engine_objective(models, reps), x)
    err = np.abs(loose(x) - ref) / np.abs(ref).max()
    assert err[0] > 1e-2                         # d/d log10 Q_CSF is wrong by percents, not 1e-4


@pytest.fixture(scope="module")
def run_only_problem(schedule_problem):
    """The same problem with no pre-equilibration block: the first tracked row is the start of the simulation."""
    return make_problem("schedule", schedule_problem[0]["toy"]["df_dict"]["toy"], blocks=RUN_ONLY)


def _build(problem, tmp_path, **settings):
    models, reps, text = problem
    return build_gradient({"analytic_gradient": {"cache_dir": str(tmp_path), "verbose": False, **settings}},
                          models=models, replicates=reps, groups=GROUPS, param_names=NAMES, scales=SCALES,
                          x0_lin=TRUTH * np.array([1.2, 1.0, 1.0, 1.3, 1.1]), bounds_lin=BOX, model_text=text,
                          paths={})


def test_a_window_that_starts_at_the_first_tracked_row_matches_finite_differences(run_only_problem, tmp_path):
    # AMICI reports the end of each segment, so the row at the start of the window cannot be read: the table
    # must not ask for it (it used to, and the table and the outputs differed by one row)
    models, reps, _ = run_only_problem
    g = _build(run_only_problem, tmp_path)
    x = to_opt(TRUTH * np.array([0.9, 1.0, 1.2, 0.8, 1.1]))
    ref = richardson(engine_objective(models, reps), x)
    assert np.abs(ref).max() > 1e-3
    np.testing.assert_allclose(g(x), ref, rtol=2e-4, atol=2e-4 * np.abs(ref).max())


def test_a_data_time_at_the_start_of_the_simulation_is_refused_with_a_message(run_only_problem, tmp_path):
    from pyantigen.engine.Analytic_gradient import AnalyticGradientError
    models, reps, text = run_only_problem
    df = models["toy"]["df_dict"]["toy"]
    at_start = pd.concat([pd.DataFrame({"time": [0.0], "ly": [0.0], "zz": [0.0]}), df], ignore_index=True)
    problem = ({"toy": {**models["toy"], "df_dict": {"toy": at_start}}}, reps, text)
    g = _build(problem, tmp_path)
    with pytest.raises(AnalyticGradientError, match="at or before the start"):
        g(to_opt(TRUTH))


def test_extra_times_add_rows_the_loss_may_read_without_changing_the_gradient(
        schedule_problem, provider, tmp_path):
    g = _build(schedule_problem, tmp_path, extra_times={"toy": [2.1, 7.9, -9.0, 99.0]})
    p_lin = TRUTH * np.array([1.2, 1.0, 1.0, 1.3, 1.1])
    table, _D = g.sims["toy"].tables(p_lin, GROUPS, schedule_problem[1], schedule_problem[0])
    t = table["time"]
    assert 2.1 in t and 7.9 in t                       # asked for, inside the window
    assert -9.0 not in t and 99.0 not in t            # outside (start, end]: ignored
    x = to_opt(p_lin)
    np.testing.assert_allclose(g(x), provider(x), rtol=1e-6, atol=1e-8)

"""Boluses: schedule steps that assign a *state*, with sensitivities carried through the jump.

Run from the repository root: python -m pytest tests/engine/test_segment_jumps.py

The toy is a one-compartment depot -> central model with a nonlinear effect. The
schedule has an additive bolus ``depot = depot + dose``, a state-dependent
reset ``central = frac*central``, a state-independent set ``eff = 0.2*dose``, and
an infusion whose end moves with the fitted parameters (``2 + D_tot/R_inf``).

A jump x+ = g(x-, p) at a switch time t_s(p) changes the sensitivity S = dx/dp
in the two formulations as

* time-scaled (what AMICI integrates): ``S+ = g_x S- + g_p``, nothing about t_s;
* original time: ``S+ = g_x (S- + f- dt_s/dp) + g_p - f+ dt_s/dp``.

They must agree with each other and with finite differences. Dropping ``g_x``
(the sensitivity the state carried in) must be wrong, which shows the toy
exercises it.
"""
import numpy as np
import pytest
from scipy.integrate import solve_ivp

from pyantigen.generate.input_schedule import InputSchedule

THETA = ["k_a", "k_e", "dose", "R_inf", "D_tot", "frac"]
NOMINAL = {"k_a": 1.2, "k_e": 0.3, "dose": 5.0, "R_inf": 4.0, "D_tot": 6.0, "frac": 0.4}
STATES = ["depot", "central", "eff"]
STATE_NAMES = STATES
OBS = [1.0, 2.5, 3.2, 3.7, 6.5, 8.0, 9.5, 11.0, 12.5, 14.0]
N, M = len(STATES), len(THETA)

MODEL = """
k_a = {k_a!r}; k_e = {k_e!r}; dose = {dose!r}; R_inf = {R_inf!r}; D_tot = {D_tot!r}; frac = {frac!r}
Q_in = 0
depot = 0; central = 0; eff = 0
depot' = -k_a*depot
central' = k_a*depot - k_e*central + Q_in
eff' = central/(1 + central) - 0.5*eff
"""


def schedule():
    s = InputSchedule(NOMINAL)
    dur = s.moving("t_inf", "D_tot/R_inf")
    s.periodic(0, 6, 3, [(0, {"depot": "depot + dose"})])        # additive bolus at 0, 6, 12
    s.step(3, {"central": "frac*central"})                       # state-dependent reset
    s.step(9, {"eff": "0.2*dose"})                               # state-independent set
    for start in (2, 10):                                        # infusions that end at start + D_tot/R_inf
        s.step(start, {"Q_in": "R_inf"})
        s.step(start + dur, {"Q_in": "0"})
    return s


def at(**kw):
    return dict(NOMINAL, **kw)


# --- right-hand side and Jacobians by hand ---------------------------------------------

def f_rhs(x, q, ka, ke):
    d, c, e = x
    return np.array([-ka * d, ka * d - ke * c + q, c / (1 + c) - 0.5 * e])


def f_jac(x, ka, ke):
    d, c, _e = x
    fx = np.array([[-ka, 0, 0], [ka, -ke, 0], [0, 1 / (1 + c) ** 2, -0.5]])
    fq = np.array([0.0, 1.0, 0.0])
    fp = np.zeros((N, M))                       # explicit dependence on the parameters
    fp[:, THETA.index("k_a")] = [-d, d, 0]
    fp[:, THETA.index("k_e")] = [0, -c, 0]
    return fx, fq, fp


def plan(sched, theta, obs=OBS):
    return sched.plan(0, 14, obs, params=theta, wrt=THETA, states=STATES)


def apply_jumps(seg, x, S):
    """The new state and sensitivity after the seg's bolus, all right-hand sides on the old state."""
    if not seg.jumps:
        return x, S
    live = {n: (x[i], tuple(S[i])) for i, n in enumerate(STATES)}
    new = {v: seg.evaluate_jump(t, live) for v, t in seg.jumps.items()}
    x, S = x.copy(), S.copy()
    for v, (val, grad) in new.items():
        x[STATES.index(v)] = val
        S[STATES.index(v)] = grad
    return x, S


def forward(sched, theta, obs=OBS):
    ka, ke = theta["k_a"], theta["k_e"]
    x, S, out = np.zeros(N), np.zeros((N, M)), []
    for seg in plan(sched, theta, obs):
        x, S = apply_jumps(seg, x, S)
        q = seg.values["Q_in"]
        if seg.t1 > seg.t0:
            sol = solve_ivp(lambda t, y: f_rhs(y, q, ka, ke), (seg.t0, seg.t1), x,
                            method="DOP853", rtol=1e-11, atol=1e-13)
            x = sol.y[:, -1]
        if seg.observed:
            out.append(x.copy())
    return np.array(out)


def sens_scaled(sched, theta, obs=OBS, keep_gx=True):
    """dX_obs/dtheta, time-scaled: the jump carries S through g_x; no term for the switch time."""
    ka, ke = theta["k_a"], theta["k_e"]
    x, S, out = np.zeros(N), np.zeros((N, M)), []
    for seg in plan(sched, theta, obs):
        if seg.jumps:
            if keep_gx:
                x, S = apply_jumps(seg, x, S)
            else:                                   # negative control: forget the carried sensitivity
                x2, S2 = apply_jumps(seg, x, np.zeros_like(S))
                x, S = x2, S2
        q, dq = seg.values["Q_in"], np.array(seg.grads["Q_in"])
        dt0, dt1 = np.array(seg.dt0), np.array(seg.dt1)
        D, dD = seg.t1 - seg.t0, dt1 - dt0

        def aug(sc, z, q=q, dq=dq, D=D, dD=dD):
            xx, SS = z[:N], z[N:].reshape(N, M)
            f = f_rhs(xx, q, ka, ke)
            fx, fq, fp = f_jac(xx, ka, ke)
            return np.concatenate([D * f, (D * (fx @ SS + fp + np.outer(fq, dq)) + np.outer(f, dD)).ravel()])

        sol = solve_ivp(aug, (0.0, 1.0), np.concatenate([x, S.ravel()]), method="DOP853",
                        rtol=1e-11, atol=1e-13)
        x, S = sol.y[:N, -1], sol.y[N:, -1].reshape(N, M)
        if seg.observed:
            out.append(S.copy())
    return np.array(out)


# the jumps written out by hand: new value, d/dx (as a row), d/dp (as a vector), per right-hand side
def _hand_jump(var, x, theta):
    th = {n: i for i, n in enumerate(THETA)}
    gx, gp = np.zeros(N), np.zeros(M)
    if var == "depot":                              # depot + dose
        gx[STATES.index("depot")] = 1.0
        gp[th["dose"]] = 1.0
        return x[0] + theta["dose"], gx, gp
    if var == "central":                            # frac*central
        gx[STATES.index("central")] = theta["frac"]
        gp[th["frac"]] = x[1]
        return theta["frac"] * x[1], gx, gp
    if var == "eff":                                # 0.2*dose
        gp[th["dose"]] = 0.2
        return 0.2 * theta["dose"], gx, gp
    raise KeyError(var)


def sens_original(sched, theta, obs=OBS):
    """The same in original time, with the jump terms of the two formulas written out by hand."""
    ka, ke = theta["k_a"], theta["k_e"]
    x, S, out = np.zeros(N), np.zeros((N, M)), []
    for seg in plan(sched, theta, obs):
        dt = np.array(seg.dt0)
        f_before = f_rhs(x, seg.pre_values["Q_in"], ka, ke)
        if seg.jumps or dt.any() or seg.step:
            x_new, S_new = x.copy(), S.copy()
            if seg.jumps:
                J = np.eye(N)
                gp = np.zeros((N, M))
                for v in seg.jumps:
                    val, gx, gpv = _hand_jump(v, x, theta)
                    i = STATES.index(v)
                    x_new[i], J[i], gp[i] = val, gx, gpv
                S_new = J @ (S + np.outer(f_before, dt)) + gp - 0.0     # total derivative of x+ at t_s
            else:
                S_new = S + np.outer(f_before, dt)
            f_after = f_rhs(x_new, seg.values["Q_in"], ka, ke)
            S = S_new - np.outer(f_after, dt)
            x = x_new
        q, dq = seg.values["Q_in"], np.array(seg.grads["Q_in"])

        def aug(t, z, q=q, dq=dq):
            xx, SS = z[:N], z[N:].reshape(N, M)
            fx, fq, fp = f_jac(xx, ka, ke)
            return np.concatenate([f_rhs(xx, q, ka, ke), (fx @ SS + fp + np.outer(fq, dq)).ravel()])

        if seg.t1 > seg.t0:
            sol = solve_ivp(aug, (seg.t0, seg.t1), np.concatenate([x, S.ravel()]), method="DOP853",
                            rtol=1e-11, atol=1e-13)
            x, S = sol.y[:N, -1], sol.y[N:, -1].reshape(N, M)
        if seg.observed:
            out.append(S.copy())
    return np.array(out)


def richardson_fd(fun, theta, h_rel=2e-3):
    cols = []
    for name in THETA:
        h = h_rel * max(1.0, abs(theta[name]))

        def central(h):
            up, dn = dict(theta, **{name: theta[name] + h}), dict(theta, **{name: theta[name] - h})
            return (fun(up) - fun(dn)) / (2 * h)

        cols.append((4 * central(h / 2) - central(h)) / 3)
    return np.stack(cols, axis=-1)


def close(a, b, rtol=1e-6):
    scale = float(np.max(np.abs(b))) + 1e-12
    np.testing.assert_allclose(a, b, rtol=rtol, atol=rtol * scale)


# --- the formulas ---------------------------------------------------------------------------

@pytest.mark.parametrize("variant", [{}, {"R_inf": 3.2}, {"R_inf": 4.5, "frac": 0.7}, {"k_a": 0.5, "D_tot": 7.0}])
def test_the_time_scaled_jump_gradient_matches_finite_differences(variant):
    sched, theta = schedule(), at(**variant)
    close(sens_scaled(sched, theta), richardson_fd(lambda th: forward(sched, th), theta))


@pytest.mark.parametrize("variant", [{}, {"R_inf": 3.2}])
def test_original_time_jump_terms_give_the_same_numbers(variant):
    sched, theta = schedule(), at(**variant)
    close(sens_original(sched, theta), sens_scaled(sched, theta), rtol=1e-7)


def test_dropping_the_carried_sensitivity_at_a_jump_is_wrong():
    sched, theta = schedule(), at()
    fd = richardson_fd(lambda th: forward(sched, th), theta)
    wrong = sens_scaled(sched, theta, keep_gx=False)
    assert np.abs(wrong - fd).max() > 1e-2 * np.abs(fd).max()


def test_the_gradient_through_the_moving_infusion_end_and_the_boluses_is_real():
    s = sens_scaled(schedule(), at())
    for name in ("R_inf", "D_tot", "dose", "frac"):
        assert np.abs(s[..., THETA.index(name)]).max() > 1e-2


def test_a_bolus_just_after_the_moving_end_of_an_infusion():
    # the first infusion ends at 2 + D_tot/R_inf = 3.01, 0.01 h after the reset central = frac*central
    sched, theta = schedule(), at(D_tot=4.04, R_inf=4.0)
    assert any(abs(seg.t0 - 3.0) < 1e-12 and seg.jumps for seg in plan(sched, theta))
    assert any(abs(seg.t0 - 3.01) < 1e-12 and not seg.jumps for seg in plan(sched, theta))
    fd = richardson_fd(lambda th: forward(sched, th, [2.5, 3.5, 5.0]), theta)
    close(sens_scaled(sched, theta, [2.5, 3.5, 5.0]), fd)
    close(sens_original(sched, theta, [2.5, 3.5, 5.0]), fd)


# --- plan() and the emitter ---------------------------------------------------------------------

def test_plan_separates_states_from_inputs():
    seg = next(g for g in plan(schedule(), at()) if g.jumps and "central" in g.jumps)
    assert seg.jumps == {"central": "frac*central"}
    assert "Q_in" in seg.values and "central" not in seg.values and "depot" not in seg.values


def test_an_input_that_reads_a_state_is_refused():
    from pyantigen.generate.input_schedule import ScheduleError
    s = InputSchedule({"k": 1.0})
    s.step(1, {"Q_in": "k*depot"})
    with pytest.raises(ScheduleError, match="state"):
        s.plan(0, 2, params={"k": 1.0}, states=["depot"])


def test_evaluate_jump_uses_the_values_from_before_the_switch():
    s = InputSchedule({"k": 2.0})
    s.step(1, {"a": "b + k", "b": "a"})                      # a swap-like pair, both states
    seg = s.plan(0, 2, params={"k": 2.0}, wrt=["k"], states=["a", "b"])[1]
    state = {"a": (10.0, (0.0,)), "b": (3.0, (1.0,))}
    assert seg.evaluate_jump("b + k", state) == (5.0, (2.0,))
    assert seg.evaluate_jump("a", state) == (10.0, (0.0,))


# --- the real runners ------------------------------------------------------------------------------

te = pytest.importorskip("tellurium")
from pyantigen.engine import Simulate as sim  # noqa: E402
from pyantigen.generate.scaled_time import scale_model  # noqa: E402

OBSERVED = ["time"] + STATES + ["Q_in"]


def settings(sched=None):
    out = {"integrator": "cvode", "absolute_tolerance": 1e-12, "relative_tolerance": 1e-12,
           "stiff": True, "variable_step_size": True,
           "simulation_blocks": {
               "pre": {"start": -1, "end": 0, "n_points": 11, "tracked": False},
               "run": {"start": 0, "end": 14, "n_points": 1401, "tracked": True}}}
    if sched is not None:
        out["schedule"] = sched
    return out


def test_the_segment_runner_applies_boluses_like_events():
    for variant in ({}, {"R_inf": 3.2}):
        theta = at(**variant)
        legacy = np.asarray(sim.simulate(
            te.loada(MODEL.format(**theta) + schedule().to_antimony(events=True)), settings(), OBSERVED))
        r = te.loada(MODEL.format(**theta) + schedule().to_antimony(events=False))
        new = np.asarray(sim.simulate(r, settings(schedule()), OBSERVED))
        assert new.shape == legacy.shape
        np.testing.assert_allclose(new, legacy, rtol=1e-6, atol=1e-8)
        # states are left where the run ended, inputs put back
        assert r["Q_in"] == 0.0


def test_the_event_text_a_generator_writes_runs_the_same():
    # the same schedule, read back from its own Antimony events
    theta = at()
    parsed = InputSchedule.from_antimony_events(schedule().to_antimony(events=True), theta)
    r = te.loada(MODEL.format(**theta) + parsed.to_antimony(events=False))
    new = np.asarray(sim.simulate(r, settings(parsed), OBSERVED))
    legacy = np.asarray(sim.simulate(
        te.loada(MODEL.format(**theta) + schedule().to_antimony(events=True)), settings(), OBSERVED))
    np.testing.assert_allclose(new, legacy, rtol=1e-6, atol=1e-8)


def test_an_infusion_ending_exactly_on_a_bolus_is_refused_not_guessed():
    from pyantigen.generate.input_schedule import ScheduleError
    with pytest.raises(ScheduleError):
        plan(schedule(), at(D_tot=4.0, R_inf=4.0))             # ends at 3.0, the time of the reset


def test_a_species_in_concentration_units_is_refused_by_the_runner():
    from pyantigen.generate.input_schedule import ScheduleError
    from pyantigen.engine.Segmented import SegmentRunner
    r = te.loada("compartment C = 2; species A in C; A = 1; k = 0.1\nJ: A -> ; k*A\n")
    s = InputSchedule({"k": 0.1})
    s.step(1, {"A": "A + 1"})
    with pytest.raises(ScheduleError, match="concentration"):
        SegmentRunner(r, s)


def test_the_scaled_model_classifies_states_and_inputs():
    sched = schedule()
    sm = scale_model(te.antimonyToSBML(MODEL.format(**NOMINAL) + sched.to_antimony(events=False)), sched)
    assert sm.inputs == ["Q_in"] and sorted(sm.jump_states) == ["central", "depot", "eff"]


def test_the_scaled_model_with_boluses_reproduces_the_events():
    theta = at(R_inf=3.5)
    sched = schedule()
    sm = scale_model(te.antimonyToSBML(MODEL.format(**theta) + sched.to_antimony(events=False)), sched)
    r = te.loadSBMLModel(sm.sbml)
    r.integrator.absolute_tolerance = r.integrator.relative_tolerance = 1e-12
    x, S = np.zeros(N), np.zeros((N, M))
    got = []
    for seg in sched.plan(0, 14, OBS, params=theta, wrt=THETA, states=sm.jump_states):
        x, S = apply_jumps(seg, x, S)
        r.reset()
        for k, v in sm.segment_constants(seg).items():
            r[k] = v
        r["Q_in"] = seg.values["Q_in"]
        for n, v in zip(STATES, x):
            r[n] = v
        x = np.asarray(r.simulate(0, 1, 2, STATES))[-1]
        if seg.observed:
            got.append(x.copy())
    legacy = te.loada(MODEL.format(**theta) + sched.to_antimony(events=True))
    legacy.integrator.absolute_tolerance = legacy.integrator.relative_tolerance = 1e-12
    res = legacy.simulate(-1, 14, 1501, ["time"] + STATES)
    rows = [int(np.argmin(np.abs(res[:, 0] - t))) for t in OBS]
    np.testing.assert_allclose(np.array(got), res[rows, 1:], rtol=1e-6, atol=1e-8)

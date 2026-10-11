"""Forward sensitivities through a segmented run: the formulas Phase 4 will emit, checked in scipy.

Run from the repository root: python -m pytest tests/engine/test_segment_sensitivity.py

The question is whether the gradient of an observation with respect to a parameter
that *moves a switch time* (Q_CSF moves the refill at V_LP/Q_CSF) can be had from a
plain forward-sensitivity integration of an event-free model. Two formulations are
written out here, independently, and compared with each other, with central finite
differences of the forward solution, and with finite differences of the real
RoadRunner segment runner.

Time-scaled (what an AMICI model without events would integrate). Inside a segment
``t = t_a(p) + D(p) s, s in [0, 1]``, ``dX/ds = D f``, and with S = dX/dp::

    dS/ds = D (f_X S + f_p + f_u du/dp + f_c c'(t) dt/dp) + f dD/dp,   dt/dp = dt_a/dp + s dD/dp

S is *carried over unchanged* from one segment to the next: the dependence of the
switch time lives in ``f dD/dp`` (and, for an input that is a function of the clock,
in the ``c'(t) dt/dp`` term).

Original time (what an event-based forward sensitivity does). ``dS/dt = f_X S + f_p
+ f_u du/dp`` inside a segment, and at a moving switch ``S+ = S- + (f- - f+) dt_s/dp``.

Both must give the same numbers. Two negative controls drop one term each and must
be wrong, so the agreement is not an accident of the test model.
"""
import numpy as np
import pytest
from scipy.integrate import solve_ivp

from pyantigen.generate.input_schedule import InputSchedule, ScheduleError

THETA = ["Q_CSF", "V_LP", "t_CSFdraw", "f_SN", "k_y"]
NOMINAL = {"Q_CSF": 17.39500698, "V_LP": 6.0, "t_CSFdraw": 0.1, "f_SN": 0.1, "k_y": 0.3}
INPUTS = ["Q_Leak", "Q_SN", "Q_LP", "Q_refill"]           # piecewise constant, set by steps
STATES = ["x_leak", "x_sn", "x_lp", "x_refill", "y", "z", "w"]
OBS = [0.3, 0.45, 1.0, 2.2, 3.7, 5.25, 7.0, 8.0]
N, M = len(STATES), len(THETA)
K_Y = THETA.index("k_y")

MODEL = """
Q_CSF = {Q_CSF!r}; V_LP = {V_LP!r}; t_CSFdraw = {t_CSFdraw!r}; f_SN = {f_SN!r}; k_y = {k_y!r}
Q_Leak = 0; Q_LP = 0; Q_refill = 0; Q_SN = Q_CSF*f_SN
x_leak = 0; x_sn = 0; x_lp = 0; x_refill = 0; y = 1; z = 0; w = 0
x_leak' = Q_Leak; x_sn' = Q_SN; x_lp' = Q_LP; x_refill' = Q_refill
y' = 0.05*Q_LP - k_y*y*(1 + Q_SN)
z' = Q_refill*y/(1 + z*z) - 0.2*z
w' = f_L - 0.4*w
"""


def schedule():
    s = InputSchedule(NOMINAL)
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
    s.linear_input("f_L", [0, 1, 2.5, 4, 6.5], [0.1, 0.3, 0.2, 0.5, 0.1],
                   default_before=0, default_after=0)
    return s


# --- the toy right-hand side and its derivatives, written out by hand ---------------

def f_rhs(x, u, fl, k):
    y, z, w = x[4], x[5], x[6]
    q_leak, q_sn, q_lp, q_refill = u
    return np.array([q_leak, q_sn, q_lp, q_refill,
                     0.05 * q_lp - k * y * (1 + q_sn),
                     q_refill * y / (1 + z * z) - 0.2 * z,
                     fl - 0.4 * w])


def f_jac(x, u, k):
    """(f_X, f_u, f_c, f_k): the Jacobians with respect to state, inputs, curve, k_y."""
    y, z = x[4], x[5]
    q_sn, q_refill = u[1], u[3]
    fx = np.zeros((N, N))
    fx[4, 4] = -k * (1 + q_sn)
    fx[5, 4] = q_refill / (1 + z * z)
    fx[5, 5] = -2 * z * q_refill * y / (1 + z * z) ** 2 - 0.2
    fx[6, 6] = -0.4
    fu = np.zeros((N, 4))
    fu[0, 0] = fu[1, 1] = fu[2, 2] = fu[3, 3] = 1.0
    fu[4, 2] = 0.05
    fu[4, 1] = -k * y
    fu[5, 3] = y / (1 + z * z)
    fc = np.zeros(N)
    fc[6] = 1.0
    fk = np.zeros(N)
    fk[4] = -y * (1 + q_sn)
    return fx, fu, fc, fk


# --- the schedule, flattened into segments with exact derivatives -------------------

class Seg:
    """One stretch of the clock with everything constant on it (arrays over THETA / INPUTS)."""


def build_plan(sched, theta, obs=OBS, t_start=0.0, t_end=8.0):
    """Segments in time order, each with its inputs and the derivatives of all of it."""
    out = []
    for p in sched.plan(t_start, t_end, obs, params=theta, wrt=THETA):
        s = Seg()
        s.t0, s.t1, s.observed = p.t0, p.t1, p.observed
        s.dt0, s.dt1 = np.array(p.dt0), np.array(p.dt1)
        s.u = np.array([p.values[v] for v in INPUTS])
        s.du = np.array([p.grads[v] for v in INPUTS])
        s.fl, s.slope, s.mid = p.curves["f_L"]
        out.append(s)
    return out


def _curve_at(s, t):
    return s.fl + s.slope * (t - s.mid)


def _solve(fun, span, z0):
    sol = solve_ivp(fun, span, z0, method="DOP853", rtol=1e-10, atol=1e-12)
    assert sol.success, sol.message
    return sol.y[:, -1]


X0 = np.array([0, 0, 0, 0, 1.0, 0, 0])


def forward(sched, theta, obs=OBS):
    """States at the observation times, no sensitivities."""
    k = theta["k_y"]
    x, out = X0.copy(), []
    for s in build_plan(sched, theta, obs):
        if s.t1 > s.t0:
            x = _solve(lambda t, x: f_rhs(x, s.u, _curve_at(s, t), k), (s.t0, s.t1), x)
        if s.observed:
            out.append(x.copy())
    return np.array(out)


def sens_scaled(sched, theta, obs=OBS, source=True, time_term=True):
    """dX_obs/dtheta from the time-scaled equations; S is carried across segments untouched."""
    k = theta["k_y"]
    x, S, out = X0.copy(), np.zeros((N, M)), []
    for s in build_plan(sched, theta, obs):
        D, dD = s.t1 - s.t0, s.dt1 - s.dt0

        def aug(sc, z, s=s, D=D, dD=dD):
            x, S = z[:N], z[N:].reshape(N, M)
            fl = _curve_at(s, s.t0 + D * sc)
            f = f_rhs(x, s.u, fl, k)
            fx, fu, fc, fk = f_jac(x, s.u, k)
            ftheta = fu @ s.du
            ftheta[:, K_Y] += fk
            inner = fx @ S + ftheta
            if time_term:
                inner = inner + np.outer(fc * s.slope, s.dt0 + sc * dD)
            dS = D * inner
            if source:
                dS = dS + np.outer(f, dD)
            return np.concatenate([D * f, dS.ravel()])

        z = _solve(aug, (0.0, 1.0), np.concatenate([x, S.ravel()]))
        x, S = z[:N], z[N:].reshape(N, M)
        if s.observed:
            out.append(S.copy())
    return np.array(out)


def sens_original(sched, theta, obs=OBS):
    """The same, in original time: integrate each segment and add the jump at a moving switch."""
    k = theta["k_y"]
    x, S, out = X0.copy(), np.zeros((N, M)), []
    plan = build_plan(sched, theta, obs)
    for n, s in enumerate(plan):
        def aug(t, z, s=s):
            x, S = z[:N], z[N:].reshape(N, M)
            f = f_rhs(x, s.u, _curve_at(s, t), k)
            fx, fu, _fc, fk = f_jac(x, s.u, k)
            ftheta = fu @ s.du
            ftheta[:, K_Y] += fk
            return np.concatenate([f, (fx @ S + ftheta).ravel()])

        if s.t1 > s.t0:
            z = _solve(aug, (s.t0, s.t1), np.concatenate([x, S.ravel()]))
            x, S = z[:N], z[N:].reshape(N, M)
        if s.observed:
            out.append(S.copy())
        if n + 1 < len(plan):
            nxt = plan[n + 1]
            jump = f_rhs(x, s.u, _curve_at(s, s.t1), k) - f_rhs(x, nxt.u, _curve_at(nxt, nxt.t0), k)
            S = S + np.outer(jump, s.dt1)
    return np.array(out)


def richardson_fd(fun, theta, h_rel=2e-3):
    """Central differences of fun(theta) (an array), Richardson-extrapolated to O(h^4)."""
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


def at(q):
    return dict(NOMINAL, Q_CSF=q)


# --- the formulas -------------------------------------------------------------------

@pytest.mark.parametrize("q", [17.39500698, 12.0, 9.0, 40.0])
def test_time_scaled_sensitivities_match_finite_differences(q):
    sched, theta = schedule(), at(q)
    fd = richardson_fd(lambda th: forward(sched, th), theta)
    close(sens_scaled(sched, theta), fd)


@pytest.mark.parametrize("q", [17.39500698, 25.0, 12.0])
def test_original_time_jumps_give_the_same_numbers(q):
    sched, theta = schedule(), at(q)
    close(sens_original(sched, theta), sens_scaled(sched, theta), rtol=1e-8)


def test_the_parameter_that_moves_the_switches_actually_matters():
    # the gradient with respect to Q_CSF and V_LP is not small next to the others
    sched, theta = schedule(), at(NOMINAL["Q_CSF"])
    s = sens_scaled(sched, theta)
    assert np.abs(s[..., THETA.index("V_LP")]).max() > 0.1
    assert np.abs(s[..., THETA.index("Q_CSF")]).max() > 0.1


@pytest.mark.parametrize("q", [17.39500698, 25.0])
def test_without_the_source_term_the_gradient_is_wrong(q):
    sched, theta = schedule(), at(q)
    fd = richardson_fd(lambda th: forward(sched, th), theta)
    wrong = sens_scaled(sched, theta, source=False)
    err = np.abs(wrong - fd)[..., [THETA.index(n) for n in ("Q_CSF", "V_LP", "t_CSFdraw")]].max()
    assert err > 1e-2
    # parameters that move no switch time do not need it
    free = [THETA.index("f_SN"), THETA.index("k_y")]
    np.testing.assert_allclose(wrong[..., free], fd[..., free], rtol=1e-6,
                               atol=1e-6 * np.abs(fd[..., free]).max())


def test_without_the_clock_term_a_moving_switch_inside_a_curve_interval_is_wrong():
    # the refill at V_LP/Q_CSF sits inside the 0-1 h and 4-6.5 h curve intervals, where
    # the measured input is a function of the clock the switch time moves
    sched, theta = schedule(), at(NOMINAL["Q_CSF"])
    fd = richardson_fd(lambda th: forward(sched, th), theta)
    wrong = sens_scaled(sched, theta, time_term=False)
    w = STATES.index("w")
    assert np.abs(wrong[:, w] - fd[:, w]).max() > 1e-4


# --- the awkward cases ----------------------------------------------------------------

def test_a_switch_landing_on_a_curve_knot_has_a_zero_length_segment_and_the_right_gradient():
    # V_LP/Q_CSF = 0.5: the hour-2 refill is exactly at the 2.5 h knot
    sched, theta = schedule(), at(12.0)
    plan = build_plan(sched, theta)
    assert any(s.t1 == s.t0 and np.any(s.dt1 != s.dt0) for s in plan)
    fd = richardson_fd(lambda th: forward(sched, th), theta)
    close(sens_scaled(sched, theta), fd)
    close(sens_original(sched, theta), fd)


def test_an_observation_exactly_on_a_moving_switch_has_a_one_sided_gradient():
    # the state is continuous but its Q_CSF-derivative is not: it is S- before the switch
    # and S+ = S- + (f- - f+) dt/dQ after it, and the observation sits on the kink
    sched, theta = schedule(), at(12.0)
    obs = [2.5]
    both = np.array([sens_scaled(sched, theta, obs=obs)[0], sens_original(sched, theta, obs=obs)[0]])
    assert np.allclose(both[0], both[1], rtol=1e-8, atol=1e-8)       # the two formulations agree

    q = THETA.index("Q_CSF")
    h = 1e-3

    def one_sided(sign):
        f = [forward(sched, at(12.0 + sign * j * h), obs)[0] for j in range(3)]
        return sign * (-3 * f[0] + 4 * f[1] - f[2]) / (2 * h)

    right, left = one_sided(+1), one_sided(-1)       # right: the refill is earlier than the knot
    assert not np.allclose(right, left, rtol=1e-3, atol=1e-6)        # a real kink
    mine = both[0][:, q]
    scale = np.abs(right).max()
    near_right = np.allclose(mine, right, rtol=1e-4, atol=1e-6 * scale)
    near_left = np.allclose(mine, left, rtol=1e-4, atol=1e-6 * scale)
    assert near_right or near_left       # one of the two one-sided derivatives, never a blend


def test_infeasible_parameters_are_refused_before_any_integration():
    for q in (5.0, 61.0):
        with pytest.raises(ScheduleError):
            build_plan(schedule(), at(q))


# --- the real segment runner ------------------------------------------------------------

te = pytest.importorskip("tellurium")
from pyantigen.engine import Simulate as sim  # noqa: E402


def runner_states(sched, theta, obs=OBS):
    """The same observations from the RoadRunner segment runner (no events in the model)."""
    r = te.loada(MODEL.format(**theta) + sched.to_antimony(events=False))
    ss = {"integrator": "cvode", "absolute_tolerance": 1e-12, "relative_tolerance": 1e-12,
          "stiff": True, "variable_step_size": True,
          "simulation_blocks": {"run": {"start": 0, "end": 8, "n_points": 801, "tracked": True}},
          "schedule": sched}
    res = np.asarray(sim.simulate(r, ss, ["time"] + STATES))
    rows = [int(np.argmin(np.abs(res[:, 0] - t))) for t in obs]
    assert np.allclose(res[rows, 0], obs, atol=1e-9)
    return res[rows, 1:]


@pytest.mark.parametrize("q", [17.39500698, 25.0, 12.0])
def test_the_scipy_forward_solution_is_the_runners(q):
    sched, theta = schedule(), at(q)
    np.testing.assert_allclose(runner_states(sched, theta), forward(sched, theta),
                               rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("q", [17.39500698, 12.0])
def test_time_scaled_sensitivities_match_finite_differences_of_the_runner(q):
    sched, theta = schedule(), at(q)
    fd = richardson_fd(lambda th: runner_states(sched, th), theta, h_rel=1e-2)
    close(sens_scaled(sched, theta), fd, rtol=1e-4)

"""Bounded Levenberg-Marquardt for a profile point, and the :class:`NuisanceSolver` built on it.

With fixed sigmas the loss is ``const + 0.5 r.r``, so one evaluation gives the residuals ``r`` and their Jacobian ``J``;
the gradient is ``J^T r`` and ``J^T J`` is the Gauss-Newton Hessian. A step solves ``(J^T J + lam D) s = -J^T r`` with
``D = diag(J^T J)`` (Marquardt scaling), in the optimizer space, with the bounds handled by an active set and each
``|s_i|`` capped. ``rho`` = actual / predicted decrease; ``rho > 1e-3`` accepts the step and ``lam`` follows Nielsen's
rule; anything else (a failed integration is just a rejected step) multiplies ``lam`` and shrinks the cap.

The verdict never trusts an optimizer exit flag. A point is ``converged`` when the projected gradient over the free
parameters is <= ``pg_tol``, or when the Newton decrement (what a full Gauss-Newton step would still gain) is below
``newton_tol`` with the projected gradient under ``pg_loose``. Anything else is ``not_converged``, an upper bound on the
profile there, with the reason.

This is the stepping core of ``tools/pypesto_silk_appfull_lm.py`` / ``_profile.py`` moved into the engine, with the
problem (AMICI, a test function) behind one callable, ``evaluate(x) -> (f, r, J) | None``, where ``J`` is the Jacobian of
the residuals with respect to the optimizer-space parameters.
"""
import math
import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from pyantigen.engine.Profile_record import PointRecord, PointRequest, flag_bounds, _flag_lower


@dataclass
class LmSettings:
    lam0: float = 1e-3
    lam_min: float = 1e-10          # below this the solve is the raw Gauss-Newton step, ill-conditioned
    step0: float = 0.05             # initial cap on |s_i| (decades for a log parameter)
    step_max: float = 0.3
    pg_tol: float = 1e-2
    newton_tol: float = 1e-4        # model decrease of a full step: the remaining error in the profile value
    pg_loose: float = 0.2           # the projected gradient allowed alongside the Newton test
    max_evals: int = 40
    strict_newton_tol: float = 1e-6
    strict_pg_loose: float = 0.05


def judge(x, g, lb, ub, pg_tol):
    """``(verdict, projected-gradient inf-norm)``: converged, not_converged or unverified (no finite gradient)."""
    if g is None or not np.all(np.isfinite(g)):
        return "unverified", None
    x = np.asarray(x, float)
    pg = np.where((x <= lb + 1e-12) & (g > 0), 0.0, np.where((x >= ub - 1e-12) & (g < 0), 0.0, g))
    m = float(np.max(np.abs(pg))) if pg.size else 0.0
    return ("converged" if m <= pg_tol else "not_converged"), m


def lm_step(H, g, lam, x, lb, ub, cap):
    """Bounded Marquardt step and its predicted decrease under the quadratic model."""
    n = len(x)
    d = np.maximum(np.diag(H), 1e-10 * np.diag(H).max())
    tiny = 1e-12
    fixed = ((x <= lb + tiny) & (g > 0)) | ((x >= ub - tiny) & (g < 0))
    s = np.zeros(n)
    for _ in range(n + 1):
        fr = np.where(~fixed)[0]
        if fr.size == 0:
            break
        fx = np.where(fixed)[0]
        rhs = -(g[fr] + H[np.ix_(fr, fx)] @ s[fx])
        A = H[np.ix_(fr, fr)] + lam * np.diag(d[fr])
        try:
            s[fr] = np.linalg.solve(A, rhs)
        except np.linalg.LinAlgError:
            s[fr] = np.linalg.lstsq(A, rhs, rcond=None)[0]
        xn = x + s
        viol = (~fixed) & ((xn < lb) | (xn > ub))
        if not viol.any():
            break
        s[viol] = np.clip(xn[viol], lb[viol], ub[viol]) - x[viol]
        fixed |= viol
    m = float(np.abs(s).max()) if n else 0.0
    if m > cap:
        s *= cap / m
    return s, float(-(g @ s + 0.5 * s @ H @ s))


@dataclass
class LmResult:
    x: np.ndarray
    f: float
    verdict: str
    pg: Optional[float]
    reason: str
    evals: int


class OutOfTime(Exception):
    """The deadline passed between evaluations."""


def minimize(evaluate: Callable, x0, lb, ub, free, settings: LmSettings = None, strict=False, deadline=None,
             anchor=None, prior_evals=0, log=None) -> LmResult:
    """Bounded LM over the *free* indices of ``x0`` (the rest are held). ``evaluate(x)`` -> ``(f, r, J)`` or None.

    *anchor* is where an unevaluable start is pulled back toward (up to six times, halfway each). *prior_evals* counts
    evaluations already spent on this point in an earlier process, against ``max_evals``. Raises :class:`OutOfTime`
    when *deadline* (epoch seconds) passes between evaluations.
    """
    st = settings or LmSettings()
    newton_tol = st.strict_newton_tol if strict else st.newton_tol
    pg_loose = st.strict_pg_loose if strict else st.pg_loose
    say = log or (lambda *_: None)
    lb, ub = np.asarray(lb, float), np.asarray(ub, float)
    free = np.asarray(free, int)
    x = np.array(x0, float)
    evals = prior_evals

    def ev(xx):
        nonlocal evals
        evals += 1
        return evaluate(xx)

    def late():
        return deadline is not None and time.time() > deadline

    cur = None
    for _ in range(6):
        if late():
            raise OutOfTime
        cur = ev(x)
        if cur is not None:
            break
        if anchor is None:
            break
        xs = x.copy()
        x[free] = 0.5 * (x[free] + np.asarray(anchor, float)[free])
        say(f"point could not be evaluated; retrying nearer the anchor (dist {np.abs(x - xs).max():.3g})")
    if cur is None:
        return LmResult(x, math.inf, "failed", None, "no evaluable start", evals)
    f, r, J = cur
    Jf = J[:, free]
    g = Jf.T @ r
    lam, nu, cap = st.lam0, 2.0, st.step0
    verdict, pg, reason = "not_converged", None, "evaluation budget spent"
    while True:
        verdict, pg = judge(x[free], g, lb[free], ub[free], st.pg_tol)
        if verdict == "converged":
            reason = f"projected gradient {pg:.3g}"
            break
        nd = lm_step(Jf.T @ Jf, g, st.lam_min, x[free], lb[free], ub[free], 1e9)[1]
        if pg is not None and pg <= pg_loose and nd < newton_tol:
            verdict, reason = "converged", f"newton decrement {nd:.2g} < {newton_tol} (projected gradient {pg:.3g})"
            break
        if evals >= st.max_evals:
            break
        if late():
            raise OutOfTime
        H = Jf.T @ Jf
        s, pred = lm_step(H, g, lam, x[free], lb[free], ub[free], cap)
        for _ in range(8):
            if pred > 0:
                break
            lam *= 10.0
            s, pred = lm_step(H, g, lam, x[free], lb[free], ub[free], cap)
        if not pred > 0 or np.abs(s).max() < 1e-10:
            reason = f"no model decrease available (pred {pred:.3g}, lam {lam:.3g})"
            break
        xn = x.copy()
        xn[free] += s
        en = ev(xn)
        rho = float("nan") if en is None else (f - en[0]) / pred
        ok = en is not None and rho > 1e-3
        say(f"f {f:.6f} pg {pg:.4g} lam {lam:.2e} cap {cap:.3g} pred {pred:.4g} "
            f"{'FAILED' if en is None else f'rho {rho:.3g}'} -> {'accept' if ok else 'reject'}")
        if ok:
            x, (f, r, J) = xn, en
            Jf = J[:, free]
            g = Jf.T @ r
            lam = max(lam * max(1 / 3, 1 - (2 * rho - 1) ** 3), st.lam_min)
            nu = 2.0
            cap = min(cap * 1.5, st.step_max) if rho > 0.5 else cap
        else:
            lam, nu = lam * nu, nu * 2
            cap = max(cap * 0.7, 1e-6)
        if lam > 1e12:
            reason = "lambda above 1e12"
            break
    return LmResult(x, f, verdict, pg, reason, evals)


class LmSolver:
    """A :class:`NuisanceSolver`: bounded LM on ``evaluate``, with the names, scales and bounds of the problem.

    ``supports_pin_box`` is True: for a request with ``pin_rel`` the profiled parameter stays in the free set with its
    bounds replaced by a +-``pin_rel`` (relative, linear) box around the pin, so a compiled model with that parameter
    as a sensitivity parameter is reused unchanged.
    """
    name = "lm"
    supports_pin_box = True

    def __init__(self, evaluate, names, scales, lb, ub, settings=None, backend="amici", floors_sha256=None, log=None,
                 on_eval=None):
        self.evaluate, self.names, self.scales = evaluate, list(names), list(scales)
        self.lb, self.ub = np.asarray(lb, float), np.asarray(ub, float)
        self.settings = settings or LmSettings()
        self.backend, self.floors_sha256, self.log = backend, floors_sha256, log
        self.on_eval = on_eval

    def _pin_box(self, req):
        k = req.param_idx
        lin = 10.0 ** req.x_fixed if self.scales[k] == "log10" else req.x_fixed
        lo, hi = lin * (1 - req.pin_rel), lin * (1 + req.pin_rel)
        if self.scales[k] == "log10":
            lo, hi = math.log10(max(lo, 1e-300)), math.log10(hi)
        return lo, hi

    def solve(self, req: PointRequest) -> PointRecord:
        t0 = time.time()
        k = req.param_idx
        lb, ub = self.lb.copy(), self.ub.copy()
        x0 = np.array(req.x_start, float)
        if req.pin_rel is None:
            free = np.array([i for i in range(len(x0)) if i != k])
            x0[k] = req.x_fixed
            lb[k] = ub[k] = req.x_fixed
        else:
            free = np.arange(len(x0))
            lb[k], ub[k] = self._pin_box(req)
            x0[k] = min(max(req.x_fixed, lb[k]), ub[k])
        x0 = np.clip(x0, lb, ub)
        st = self.settings
        if req.max_evals or req.pg_tol != st.pg_tol:
            st = LmSettings(**{**st.__dict__, "max_evals": req.max_evals or st.max_evals, "pg_tol": req.pg_tol})
        anchor = np.array(req.anchor_x if getattr(req, "anchor_x", None) is not None else x0, float)

        def ev(x):
            out = self.evaluate(x)
            if self.on_eval is not None:
                self.on_eval(x, None if out is None else out[0])
            return out

        res = minimize(ev, x0, lb, ub, free, st, strict=getattr(req, "strict", False), deadline=req.deadline,
                       anchor=anchor, log=self.log)
        ok = math.isfinite(res.f)
        rec = PointRecord(
            param=req.param, x_fixed=float(req.x_fixed if req.pin_rel is None else res.x[k]),
            nll=float(res.f) if ok else None, dnll=float(res.f - req.anchor_nll) if ok else None,
            verdict=res.verdict if ok and res.verdict != "unverified" else ("failed" if not ok else "not_converged"),
            value=None, param_idx=k, direction=req.direction, x=[float(v) for v in res.x], evals=res.evals,
            seconds=round(time.time() - t0, 3), pg=res.pg, reason=res.reason, solver="lm", backend=self.backend,
            floors_sha256=self.floors_sha256, id=req.id, source=req.source)
        if req.best_value:
            lin = 10.0 ** rec.x_fixed if self.scales[k] == "log10" else rec.x_fixed
            rec.value, rec.rel = lin, lin / req.best_value
        else:
            rec.value = 10.0 ** rec.x_fixed if self.scales[k] == "log10" else rec.x_fixed
        flag_bounds(rec, self.names, self.lb, self.ub)
        return _flag_lower(rec)


def amici_evaluate(provider, scales, log=None):
    """``evaluate(x) -> (f, r, J) | None`` on an :class:`AmiciGradient`-like provider with ``linear_residual_jacobian``.

    ``x`` is in the optimizer space; ``J`` is returned with respect to it (the chain factor ``ln10 * p`` for a log10
    parameter). A failed integration, or a non-finite value, returns None: it is a rejected step, not an error.
    """
    from pyantigen.engine.Analytic_gradient import AnalyticGradientError
    from pyantigen.engine.Optimize import _to_linear

    def evaluate(x):
        p_lin = np.asarray(_to_linear(np.asarray(x, float), scales), float)
        t0 = time.time()
        try:
            r, J, const = provider.linear_residual_jacobian(p_lin)
        except (AnalyticGradientError, RuntimeError, FloatingPointError) as exc:
            if log:
                log(f"evaluation failed: {type(exc).__name__}: {str(exc)[:160]}")
            return None
        chain = np.array([math.log(10.0) * p if s == "log10" else 1.0 for p, s in zip(p_lin, scales)])
        J = np.asarray(J, float) * chain[None, :]
        r = np.asarray(r, float)
        f = float(const + 0.5 * r @ r)
        if not (math.isfinite(f) and np.all(np.isfinite(J))):
            return None
        if log:
            log(f"evaluated in {time.time() - t0:.0f} s: f {f:.6f}")
        return f, r, J

    return evaluate

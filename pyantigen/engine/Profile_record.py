"""One record for a profile point, one interface for the solver that makes it.

A profile point fixes one parameter at a value and minimizes the loss over the others. Two routes compute
it today and they write different files:

* RoadRunner (``Optimize`` / ``Evaluator._profile_task``): Nelder-Mead, one JSONL per parameter under
  ``results/<MODEL>/profiles/<run>/``, keys ``x_fixed``, ``nll``, ``dnll``, ``nuisance_x``, ``converged``,
  ``interrupted``, ``status``.
* AMICI (``tools/pypesto_silk_appfull_profile.py`` and the pins runs): Levenberg-Marquardt on the residual
  Jacobian, ``profile_<k>_<up|down>.jsonl`` (``val``, ``f``, ``df``, ``x``, ``verdict``, ``pg``, ``evals``)
  and ``start_<id>.json`` for a pinned point (``x``, ``fval``, ``verdict``, ``pg_inf``,
  ``evaluations_total``).

:class:`PointRecord` is the common shape. The readers (:func:`from_roadrunner`, :func:`from_amici_walk`,
:func:`from_amici_start`) turn the stored records of either route into it **without touching the files**, so
any report, crossing finder or comparison is written once. :class:`NuisanceSolver` is what the stepping
protocol will call to get a point; :class:`RoadRunnerSolver` wraps the existing Nelder-Mead worker as one
(the worker itself is unchanged), and the LM solver on AMICI will be the other.

Conventions:

* ``x_fixed`` and ``x`` are in the optimizer's space (log10 for a log parameter), ``value`` is linear.
* ``nll`` is the loss the point minimized, ``dnll = nll - anchor`` with the anchor the profile optimum. AMICI's
  ``f`` (``const + 0.5 r.r``) and RoadRunner's ``nll`` are the same quantity up to a constant, so ``dnll``
  agrees between routes; ``nll`` does not and is only compared within a route.
* ``verdict`` is ``converged``, ``not_converged`` (an upper bound on the profile there) or ``failed``.
  ``lower`` flags a point more than ``LOWER_TOL`` below the optimum: the optimum was not global.
* ``bound_active`` lists the free parameters sitting within ``BOUND_TOL`` of a declared bound, and the pinned
  parameter itself if the pin lies beyond its bound. Such a point is only an upper bound on the profile (the box,
  not the data, stopped the nuisance parameter), so a crossing it produces is *box-limited*, not an interval
  limit; see :func:`flag_bounds` and :func:`side_verdicts`.
"""
import json
import math
import os
from dataclasses import asdict, dataclass, field, fields
from typing import Optional, Protocol, runtime_checkable

VERDICTS = ("converged", "not_converged", "failed")
BOUND_TOL = 0.002         # optimizer-space (decades for a log parameter) distance that counts as on a bound
LOWER_TOL = 0.05          # a point this far below the optimum (in nll) is flagged ``lower``
DNLL_95 = 1.9207          # chi2(1, 0.95) / 2: the one-parameter 95% level


@dataclass
class PointRecord:
    param: str
    x_fixed: float                          # optimizer space
    nll: Optional[float] = None
    dnll: Optional[float] = None
    verdict: str = "failed"
    value: Optional[float] = None           # linear
    rel: Optional[float] = None             # value / best-fit value, linear
    param_idx: Optional[int] = None
    direction: int = 0                      # +1 above the optimum, -1 below, 0 unknown
    x: Optional[list] = None                # full parameter vector, optimizer space
    evals: Optional[int] = None
    seconds: Optional[float] = None
    pg: Optional[float] = None              # projected gradient, where the solver has one
    reason: Optional[str] = None
    lower: bool = False
    solver: Optional[str] = None            # "nelder-mead" | "lm"
    backend: Optional[str] = None           # "roadrunner" | "amici"
    floors_sha256: Optional[str] = None
    id: Optional[int] = None                # pin-table id
    source: Optional[str] = None            # where the point or its start came from
    interrupted: bool = False
    bound_active: list = field(default_factory=list)

    @property
    def box_limited(self):
        return bool(self.bound_active)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})

    @property
    def usable(self):
        """Has a finite dnll: it is an upper bound on the profile whether or not it converged."""
        return self.dnll is not None and math.isfinite(self.dnll)

    def check(self):
        """Problems with the record, as strings; empty if it is well formed."""
        out = []
        if self.verdict not in VERDICTS:
            out.append(f"verdict {self.verdict!r} not in {VERDICTS}")
        if self.verdict != "failed" and not self.usable:
            out.append("a non-failed record needs a finite dnll")
        if self.nll is not None and self.dnll is not None and not (math.isfinite(self.nll) and math.isfinite(self.dnll)):
            out.append("nll/dnll not finite")
        if not math.isfinite(float(self.x_fixed)):
            out.append("x_fixed not finite")
        return out


def _finite(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _lin(x, scale):
    return 10.0 ** x if scale == "log10" else x


def _flag_lower(rec):
    rec.lower = bool(rec.dnll is not None and math.isfinite(rec.dnll) and rec.dnll < -LOWER_TOL)
    return rec


def from_roadrunner(raw, scales=None, best_value=None):
    """A stored RoadRunner checkpoint record as a :class:`PointRecord`.

    *scales* (per parameter, ``"log10"``/``"lin"``) is only needed to rebuild the full vector's linear
    meaning; the record carries ``x_fixed_linear`` already. *best_value* is the linear best-fit value of the
    profiled parameter, for ``rel``.
    """
    nll, dnll = _finite(raw.get("nll")), _finite(raw.get("dnll"))
    ok = raw.get("status") == "ok" and nll is not None
    if not ok:
        verdict = "failed"
    elif raw.get("interrupted") or not raw.get("converged", True):
        verdict = "not_converged"
    else:
        verdict = "converged"
    idx = raw.get("param_idx")
    nuis = raw.get("nuisance_x")
    full = None
    if nuis is not None and idx is not None:
        full = list(nuis[:idx]) + [raw["x_fixed"]] + list(nuis[idx:])
    value = _finite(raw.get("x_fixed_linear"))
    rec = PointRecord(
        param=raw.get("param_name", "param"), x_fixed=float(raw["x_fixed"]), nll=nll, dnll=dnll, verdict=verdict,
        value=value, rel=(value / best_value if value is not None and best_value else None), param_idx=idx,
        direction=int(raw.get("direction") or 0), x=full, evals=raw.get("nfev_total", raw.get("n_evals")),
        seconds=_finite(raw.get("wall_s")), reason=raw.get("stop_reason") or raw.get("opt_message") or None,
        solver="nelder-mead", backend="roadrunner", source=raw.get("pass_label"),
        interrupted=bool(raw.get("interrupted")))
    return _flag_lower(rec)


def from_amici_walk(raw, param, param_idx=None, scale="log10", best_value=None, direction=0):
    """A line of ``profile_<k>_<up|down>.jsonl``. Point ``n == 0`` is the optimum itself."""
    v = raw["val"]
    value = _lin(v, scale)
    verdict = raw.get("verdict", "failed")
    verdict = verdict if verdict in VERDICTS else "failed"
    rec = PointRecord(
        param=param, x_fixed=float(v), nll=_finite(raw.get("f")), dnll=_finite(raw.get("df")), verdict=verdict,
        value=value, rel=(value / best_value if best_value else None), param_idx=param_idx, direction=direction,
        x=raw.get("x"), evals=raw.get("evals"), seconds=_finite(raw.get("seconds")), pg=_finite(raw.get("pg")),
        reason=raw.get("reason"), solver="lm", backend="amici", floors_sha256=raw.get("floors_sha256"),
        source="walk" if raw.get("n") else "optimum")
    out = _flag_lower(rec)
    if raw.get("LOWER"):
        out.lower = True
    return out


def from_amici_start(raw, param, param_idx, scale, f_opt, pin_value=None, best_value=None, pin_id=None,
                     floors_ok=None):
    """A pinned-point ``start_<id>.json``. *pin_value* is the linear pin; *f_opt* the profile optimum's f."""
    names = list(raw["names"])
    x = list(raw["x"])
    xk = x[param_idx]
    nll = _finite(raw.get("fval"))
    verdict = raw.get("verdict", "failed")
    verdict = verdict if verdict in VERDICTS else "failed"
    value = pin_value if pin_value is not None else _lin(xk, scale)
    rec = PointRecord(
        param=param, x_fixed=float(xk), nll=nll, dnll=(nll - f_opt if nll is not None else None), verdict=verdict,
        value=value, rel=(value / best_value if best_value else None), param_idx=param_idx, x=x,
        evals=raw.get("evaluations_total"), seconds=_finite(raw.get("seconds_this_process")),
        pg=_finite(raw.get("pg_inf")), reason=raw.get("reason"), solver="lm", backend="amici",
        floors_sha256=raw.get("floors_sha256"), id=pin_id if pin_id is not None else raw.get("start_id"),
        source=f"pin:{raw.get('start_id')}")
    if names[param_idx] != param:
        raise ValueError(f"start file parameter {names[param_idx]!r} at index {param_idx}, expected {param!r}")
    return _flag_lower(rec)


def read_jsonl(path):
    """Lines of a JSONL file; a truncated last line (a killed writer) is skipped."""
    out = []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                try:
                    out.append(json.loads(line))
                except ValueError:
                    pass
    return out


def flag_bounds(rec, names, lower, upper, tol=BOUND_TOL):
    """Fill ``rec.bound_active`` from the point's full vector. *lower*/*upper* are in the optimizer's space.

    A free parameter within *tol* of a bound is named. The pinned parameter (``rec.param_idx``) is named with a
    ``" (pin)"`` suffix if its pinned value lies beyond a bound (not merely near it: a pin at an edge is allowed).
    Returns the record; one without a vector is left unflagged.
    """
    if rec.x is None:
        return rec
    out = []
    for i, (xi, lo, hi) in enumerate(zip(rec.x, lower, upper)):
        if xi is None or not math.isfinite(xi):
            continue
        if i == rec.param_idx:
            if xi < lo - tol or xi > hi + tol:
                out.append(f"{names[i]} (pin)")
        elif xi - lo <= tol or hi - xi <= tol:
            out.append(names[i])
    rec.bound_active = out
    return rec


def side_verdicts(records, best_x, level=DNLL_95):
    """``(below, above)``, each ``"crossed"``, ``"box-limited"`` or ``"open"``.

    The side's crossing is the pair of points :func:`crossings` interpolates between. It is ``crossed`` only if
    the first point over *level* converged with no active bound; ``box-limited`` if that point (or the last one
    under the level) has an active bound or did not converge, since the true profile there may be lower and the
    limit further out; ``open`` if no point on the side is over *level*.
    """
    pts = curve(records)
    out = []
    for side in (-1, 1):
        walk = sorted((r for r in pts if (r.x_fixed - best_x) * side > 0), key=lambda r: abs(r.x_fixed - best_x))
        under = None
        verdict = "open"
        for r in walk:
            if r.dnll > level:
                bad = r.box_limited or r.verdict != "converged" or (under is not None and under.box_limited)
                verdict = "box-limited" if bad else "crossed"
                break
            under = r
        out.append(verdict)
    return tuple(out)


def curve(records):
    """Usable records of one parameter sorted by ``x_fixed``, the lowest dnll kept where values coincide.

    Every record is an upper bound on the profile, so the lower of two at one value is the better estimate
    (the same rule ``Profile_checkpoint.record_is_better`` applies).
    """
    best = {}
    for r in records:
        if not r.usable:
            continue
        k = round(r.x_fixed, 12)
        if k not in best or r.dnll <= best[k].dnll:
            best[k] = r
    return [best[k] for k in sorted(best)]


def crossings(records, best_x, level=DNLL_95):
    """``(below, above)``: where the curve first reaches *level* walking out from *best_x* each way.

    Each is the linear interpolation in ``x_fixed`` between the last point at or under the level and the first
    over it, or None if no point on that side is over it (the interval is open on that side as far as the
    points show). Interpolating in x is the plain rule; ``sqrt(dnll)`` interpolation, which is exact for a
    quadratic profile, is what the stepping protocol uses to place points.
    """
    pts = curve(records)
    out = []
    for side in (-1, 1):
        walk = [r for r in pts if (r.x_fixed - best_x) * side > 0]
        walk.sort(key=lambda r: abs(r.x_fixed - best_x))
        prev = (best_x, 0.0)
        cross = None
        for r in walk:
            if r.dnll > level:
                x0, d0 = prev
                cross = x0 + (r.x_fixed - x0) * (level - d0) / (r.dnll - d0)
                break
            prev = (r.x_fixed, r.dnll)
        out.append(cross)
    return tuple(out)


@dataclass(frozen=True)
class PointRequest:
    """What a solver is asked for: minimize over everything but ``param_idx`` with it held at ``x_fixed``."""
    param: str
    param_idx: int
    x_fixed: float                           # optimizer space
    x_start: tuple                           # full vector, optimizer space; entry param_idx is overwritten
    anchor_nll: float                        # the profile optimum, so dnll = nll - anchor_nll
    best_value: Optional[float] = None       # linear best-fit value of the parameter, for ``rel``
    direction: int = 0
    id: Optional[int] = None
    max_evals: Optional[int] = None
    deadline: Optional[float] = None         # epoch seconds; a solver stops between evaluations after it
    pg_tol: float = 1e-2
    pin_rel: Optional[float] = None          # None: held exactly. Else the parameter stays free in a +-pin_rel box
    source: Optional[str] = None
    label: Optional[str] = None


@runtime_checkable
class NuisanceSolver(Protocol):
    """Anything that can produce a profile point. ``solve`` returns a :class:`PointRecord`, never raises for a
    point that merely fails to converge (that is a verdict); it raises for a request it cannot honour."""
    name: str
    backend: str
    supports_pin_box: bool

    def solve(self, request: PointRequest) -> PointRecord: ...


def validate_result(request, record):
    """Problems with a solver's answer to a request, as strings; empty if it honours the contract."""
    out = list(record.check())
    if record.param != request.param:
        out.append(f"answered for {record.param!r}, asked {request.param!r}")
    if abs(record.x_fixed - request.x_fixed) > 1e-9 * max(1.0, abs(request.x_fixed)) and request.pin_rel is None:
        out.append(f"x_fixed {record.x_fixed} differs from the request {request.x_fixed}")
    if record.usable and record.nll is not None:
        if abs((record.nll - request.anchor_nll) - record.dnll) > 1e-9 * max(1.0, abs(record.nll)):
            out.append("dnll != nll - anchor_nll")
    if record.verdict == "converged" and not record.usable:
        out.append("converged without a value")
    return out


class RoadRunnerSolver:
    """The existing Nelder-Mead profile worker as a :class:`NuisanceSolver`; the worker is not changed.

    *run_job* takes the job dict ``Optimize`` builds for a point (``make_job``) and returns the raw result
    dict the worker writes to the checkpoint: ``Evaluator._profile_task`` in a worker process, or a pool's
    ``profile_batch`` for one job. *bounds* is the full list of ``(lo, hi)`` in optimizer space, *scales* the
    per-parameter ``"log10"``/``"lin"``. A pin box is not supported (Nelder-Mead holds the parameter fixed).
    """
    name = "nelder-mead"
    backend = "roadrunner"
    supports_pin_box = False

    def __init__(self, run_job, scales, bounds=None, method="Nelder-Mead", optimizer_kwargs=None):
        self.run_job = run_job
        self.scales = list(scales)
        self.bounds = list(bounds) if bounds is not None else None
        self.method = method
        self.optimizer_kwargs = dict(optimizer_kwargs or {})

    def job_for(self, req):
        if req.pin_rel is not None:
            raise ValueError("the RoadRunner solver holds the profiled parameter fixed; pin_rel is not supported")
        i = req.param_idx
        start = [v for j, v in enumerate(req.x_start) if j != i]
        nb = None if self.bounds is None else [tuple(b) if b is not None else None
                                               for j, b in enumerate(self.bounds) if j != i]
        kw = dict(self.optimizer_kwargs)
        if req.max_evals is not None:
            opts = dict(kw.get("options") or {})
            opts["maxfev"] = int(req.max_evals)
            kw["options"] = opts
        return {"param_idx": i, "param_name": req.param, "x_fixed": float(req.x_fixed),
                "x_fixed_linear": float(_lin(req.x_fixed, self.scales[i])), "x_start": [float(v) for v in start],
                "warm_seeded": True, "nuisance_bounds": nb, "method": self.method, "optimizer_kwargs": kw,
                "phase": 1, "direction": int(req.direction), "pass_label": req.label or req.source or "solver",
                "x_step": None, "deadline": req.deadline}

    def solve(self, req):
        raw = self.run_job(self.job_for(req))
        if _finite(raw.get("nll")) is not None:
            raw = dict(raw, dnll=float(raw["nll"]) - req.anchor_nll)
        rec = from_roadrunner(raw, self.scales, req.best_value)
        rec.id, rec.source = req.id, req.source or rec.source
        if rec.direction == 0:
            rec.direction = req.direction
        return rec


def interp_dnll(records, x):
    """The profile's dnll at *x* by linear interpolation between the neighbouring usable points, or None outside them."""
    pts = curve(records)
    for a, b in zip(pts, pts[1:]):
        if a.x_fixed <= x <= b.x_fixed:
            if b.x_fixed == a.x_fixed:
                return min(a.dnll, b.dnll)
            t = (x - a.x_fixed) / (b.x_fixed - a.x_fixed)
            return a.dnll + t * (b.dnll - a.dnll)
    return None


def compare_curves(a, b, best_a, best_b, upto=3.0, level=DNLL_95):
    """How two profiles of one parameter (two routes) agree.

    Returns a dict: ``rows`` (each point of *a* that lies inside *b*'s range and has ``dnll <= upto``: x, dnll_a,
    dnll_b interpolated, difference), ``max_abs_diff`` over those rows, the ``crossings`` of each, the number of
    points and not-converged points of each, and the summed ``evals`` and ``seconds``. The two routes anchor at
    their own optima (*best_a*, *best_b*, in ``x_fixed`` units), so dnll is each route's rise above its own.
    Interpolating a curve that is only piecewise sampled adds an error of its own: compare where *b* is dense.
    """
    rows = []
    for r in curve(a):
        if r.dnll > upto:
            continue
        y = interp_dnll(b, r.x_fixed)
        if y is not None:
            rows.append({"x": r.x_fixed, "dnll_a": r.dnll, "dnll_b": y, "diff": r.dnll - y})

    def stats(recs):
        return {"points": len(recs), "not_converged": sum(r.verdict == "not_converged" for r in recs),
                "failed": sum(r.verdict == "failed" for r in recs), "lower": sum(r.lower for r in recs),
                "evals": sum(r.evals or 0 for r in recs), "seconds": sum(r.seconds or 0.0 for r in recs)}

    return {"rows": rows, "max_abs_diff": max((abs(r["diff"]) for r in rows), default=None),
            "crossings_a": crossings(a, best_a, level), "crossings_b": crossings(b, best_b, level),
            "a": stats(a), "b": stats(b)}

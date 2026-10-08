"""Shadow stop rules for profile points: record what they would have done.

A profile point is a nuisance minimization, and its only use to the planner
(:mod:`pyantigen.engine.Profile_plan`) is a verdict on the value it reaches U, an
upper bound on the profile there, against the threshold. Most of the 1500
evaluations a point is allowed buy digits that do not change the verdict:

* U already below the threshold is a proof that the profile is, and more
  evaluations only lower it;
* U that is creeping down from far above the threshold will not get there
  within the cap, and its magnitude is not trusted either way;
* U that has stopped improving by more than a fraction of its distance from the
  threshold has said everything it is going to say.

None of this has been calibrated against real trajectories, and loosening the
optimizer's tolerance blind has cost accuracy here before (see the TOLERANCES
note in Optimizer_settings). So this module only *watches*. For each point it
keeps a sparse trace of the best value reached, and the first evaluation at which
each rule would have fired. Nothing is ever stopped. The numbers are for reading
off a finished run (:func:`shadow_summary`) and tuning the constants below
before any of them is allowed to end a point.

The rules, with U = best nll reached - anchor and thr the profile threshold:

``below``      U < ``below_frac`` * thr. A proof, needs no minimum spend.
``hopeless``   U > ``u_trust`` and, extrapolating the log-rate of the last
               window to the evaluation cap, U would still be above ``u_trust``.
               The extrapolation is optimistic -- Nelder-Mead's descent usually
               slows -- so a point it fires on is very unlikely to have got
               anywhere useful.
``settled``    U improved by less than ``kappa`` * |U - thr| over the last
               window: the tolerance scales with the distance from the decision,
               so a U of 7.6 needs ~0.4 and a U of 0.04 needs almost nothing.
               Measured on progress over a window, not on the simplex spread,
               which fires on a collapsed simplex in a curved valley.

The two window rules wait for ``max(MIN_EVALS, MIN_EVALS_PER_DIM * (n + 1))``
evaluations plus one window, so the initial simplex has been built and the
descent has started.
"""
import math

DEFAULTS = {
    "threshold": 1.9207,
    "below_frac": 0.25,
    "u_trust": 20.0,
    "kappa": 0.05,
    "window": 100,
    "sample_every": 25,
}
MIN_EVALS = 60
MIN_EVALS_PER_DIM = 5

RULES = ("below", "hopeless", "settled")


class PointMonitor:
    """Watches one point's best-so-far; never alters it.

    *trace* and *fired* continue those of an earlier launch of the same point,
    so a point resumed after the clock stopped keeps one history.
    """

    def __init__(self, anchor, max_fev, n_nuis, cfg=None, trace=None, fired=None):
        c = dict(DEFAULTS)
        c.update({k: v for k, v in (cfg or {}).items() if k in DEFAULTS})
        self.cfg = c
        self.anchor = float(anchor)
        self.max_fev = (float(max_fev) if max_fev is not None
                        and math.isfinite(float(max_fev)) else None)
        self.min_evals = max(MIN_EVALS, MIN_EVALS_PER_DIM * (int(n_nuis) + 1))
        self.trace = [list(map(float, p)) for p in (trace or [])]
        self.fired = {k: dict(v) for k, v in (fired or {}).items()}
        self._last_n = int(self.trace[-1][0]) if self.trace else 0
        self._best = self.trace[-1][1] if self.trace else math.inf

    # -- feeding -----------------------------------------------------------
    def update(self, nfev, best_nll):
        """Call after every evaluation with the running total and best so far."""
        if best_nll is None or not math.isfinite(best_nll):
            return
        nfev = int(nfev)
        if nfev <= self._last_n and self.trace:
            return
        self._best = float(best_nll)
        self._last_n = nfev
        u = self._best - self.anchor
        if "below" not in self.fired and u < self.cfg["below_frac"] * self.cfg["threshold"]:
            self.fired["below"] = {"nfev": nfev, "U": u}
        if nfev == 1 or nfev % self.cfg["sample_every"] == 0:
            self.trace.append([nfev, self._best])
            self._window_rules(nfev, u)

    def _reference(self, nfev):
        """The last sample at least one window back, or None."""
        want = nfev - self.cfg["window"]
        ref = None
        for n, b in self.trace:
            if n <= want:
                ref = (n, b)
            else:
                break
        return ref

    def _window_rules(self, nfev, u):
        if nfev < self.min_evals + self.cfg["window"]:
            return
        ref = self._reference(nfev)
        if ref is None:
            return
        n_ref, b_ref = ref
        u_ref = b_ref - self.anchor
        thr, u_trust = self.cfg["threshold"], self.cfg["u_trust"]
        gain = u_ref - u

        if "settled" not in self.fired and gain < self.cfg["kappa"] * abs(u - thr):
            self.fired["settled"] = {"nfev": nfev, "U": u, "gain": gain}

        if ("hopeless" not in self.fired and u > u_trust and self.max_fev is not None
                and u_ref > 0 and u > 0):
            span = nfev - n_ref
            rate = math.log(u_ref / u) / span if span > 0 else 0.0
            u_cap = (u * math.exp(-rate * max(self.max_fev - nfev, 0.0))
                     if rate > 0 else u)
            if u_cap > u_trust:
                self.fired["hopeless"] = {"nfev": nfev, "U": u, "U_cap": u_cap}

    # -- reporting ---------------------------------------------------------
    def finish(self, nfev, best_nll):
        """The fields to merge into the point's record."""
        if best_nll is not None and math.isfinite(best_nll):
            if not self.trace or int(self.trace[-1][0]) != int(nfev):
                self.trace.append([int(nfev), float(best_nll)])
        return {"trace": self.trace, "shadow_stops": self.fired}


# ---------------------------------------------------------------------------
# Reading a finished run
# ---------------------------------------------------------------------------

def _verdict_class(u, thr, u_trust):
    if u < thr:
        return "below"
    return "mid" if u < u_trust else "high"


def shadow_summary(records, anchor, cfg=None):
    """What the rules would have saved, and what they would have got wrong.

    *records* is any iterable of point records carrying ``shadow_stops``. For
    each rule: how many points it fired on, the evaluations it would have
    saved (the point's ``nfev_total`` minus the evaluation it fired at), and how
    many of those points ended in a different class -- below the threshold,
    between it and ``u_trust``, or above -- than the value the rule fired on.
    A class change is the only thing that matters to the planner, so that count
    is the cost of the rule, in the only currency that counts.
    """
    c = dict(DEFAULTS)
    c.update({k: v for k, v in (cfg or {}).items() if k in DEFAULTS})
    thr, u_trust = c["threshold"], c["u_trust"]
    out = {r: {"fired": 0, "saved": 0, "class_changed": 0, "examples": []}
           for r in RULES}
    n_points, spent = 0, 0
    for rec in records:
        stops = rec.get("shadow_stops")
        nll, total = rec.get("nll"), rec.get("nfev_total")
        if stops is None or nll is None or total is None:
            continue
        n_points += 1
        spent += int(total)
        u_final = float(nll) - float(anchor)
        for rule, hit in stops.items():
            if rule not in out:
                continue
            o = out[rule]
            o["fired"] += 1
            o["saved"] += max(int(total) - int(hit["nfev"]), 0)
            if (_verdict_class(hit["U"], thr, u_trust)
                    != _verdict_class(u_final, thr, u_trust)):
                o["class_changed"] += 1
                if len(o["examples"]) < 5:
                    o["examples"].append({
                        "param": rec.get("param_name"), "x": rec.get("x_fixed"),
                        "nfev": hit["nfev"], "U_at_fire": hit["U"],
                        "U_final": u_final})
    return {"n_points": n_points, "evals_spent": spent, "rules": out}


def format_shadow_summary(summary):
    """The summary as log lines."""
    s = summary
    if not s["n_points"]:
        return ""
    lines = [f"[profile] shadow stop rules (nothing was stopped): "
             f"{s['n_points']} point(s), {s['evals_spent']} evaluation(s) spent"]
    for rule in RULES:
        o = s["rules"][rule]
        lines.append(f"    {rule:<9} fired on {o['fired']:>3}, would have saved "
                     f"{o['saved']:>6} eval(s), class changed on "
                     f"{o['class_changed']}")
        for e in o["examples"]:
            lines.append(f"        changed: {e['param']} at {e['x']:.6g}: "
                         f"U {e['U_at_fire']:.3g} at eval {e['nfev']} -> "
                         f"{e['U_final']:.3g} final")
    return "\n".join(lines)

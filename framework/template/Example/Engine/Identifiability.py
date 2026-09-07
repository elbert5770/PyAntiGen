"""The slice screen: proving a parameter unbounded before profiling it.

A *slice* holds the nuisance parameters at their fitted values and varies one
parameter alone. A *profile* re-minimizes over the nuisance parameters at every
fixed value. The fitted nuisance vector is one feasible point of that
minimization, so for every value of every parameter

    dNLL_profile(x)  <=  dNLL_slice(x)

and the slice is an upper bound on the profile. That inequality is the whole of
this module. The codebase already relies on it elsewhere -- ``_profile_task``
keeps a half-finished point because "a half-finished point is a real point that
happens to sit too high" -- and here it is used in the one direction that turns
a cheap evaluation into a proof:

**If the slice far from the fitted value sits below the 1.9207 threshold, the
profile there does too, so the data do not exclude that value and this side of
the confidence interval is open.** One evaluation per side settles it, with no
nuisance optimization at all.

How far is "far", and why not the declared bound
------------------------------------------------
An earlier version of this screen read the verdict at the parameter's declared
bound. That is the wrong ruler. On this spec every bound is ``x0/10`` to
``x0*10`` -- a uniform one-decade search box around the starting value, written
to keep the optimizer in a sensible region, carrying no claim about physics. A
verdict read there says "the slice did not cross inside the box someone drew",
which halts the analysis on a convention when the crossing may sit just outside
it.

So the screen walks to whichever is *further*, the declared bound or
``span_decades`` from the fitted value, and states its reach in decades. Going
past a bound is safe in both directions and is the point of doing it. A bound is
a prior, not data: a slice that stays flat beyond it makes the finding stronger,
not weaker, and a slice that crosses just beyond it prevents exactly the false
halt an arbitrary box would have caused. The verdict then rests on a scale-free
statement -- "a thousandfold change in this parameter costs nothing" -- rather
than on where the box was drawn.

Where the bound still matters it is reported rather than assumed: a side whose
slice does not cross inside the declared bound but does cross outside it means
the fit's own search box is narrower than the confidence interval, which is
worth knowing and is not a reason to stop.

The screen is one-directional and must be read that way. It can prove a
parameter unbounded; it can never prove one identifiable. A slice that crosses
the threshold says nothing about whether the profile ever will -- the profile
may flatten out beyond the slice crossing and never reach it. So a pass here is
permission to spend profile evaluations, not an answer.

Why it is worth the evaluations
-------------------------------
An unbounded parameter is the most expensive object in the profile run. Every
one of ``max_extend`` extension steps fires on it, each a full nuisance
minimization at increasingly extreme values -- where the integrator is slowest
and ``safe_simulate``'s retry ladder is most likely to fire -- and the answer at
the end is still "open interval". At the measured cost of this spec the screen
is about ten minutes across the pool and the run it can avoid is days.

Why it halts rather than skipping
---------------------------------
A parameter the data cannot bound has to be fixed to a defensible value, from
the literature or to zero, and removed from the fit. That is a modelling
decision resting on evidence outside this run, and it is not one an automated
pass may take on its own or defer. So the screen raises
:class:`UnidentifiableParameters` and the analysis stops before spending any
profile compute. There is deliberately no override flag: an override is exactly
the on-the-fly decision this exists to prevent.
"""

import json
import os

import numpy as np

from Engine.Evaluator import FAILURE_VALUE

# chi2(df=1, p=0.95) / 2 -- the same threshold the profile crosses, and it has
# to stay the same or the screen would rule out intervals the profile would
# have drawn.
THRESHOLD = 1.9207

# Written into the checkpoint directory, which is already keyed by model hash,
# spec hash and the optimum, so a file found there belongs to this run.
SCREEN_FILENAME = "slice_screen.json"

# How far from the fitted value the screen walks, when the declared bound does
# not already reach further. Three decades either way: if the data cannot tell a
# parameter from a thousandth or a thousand times itself, nothing a profile does
# will bound it. Wider costs nothing when the model survives out there and
# yields no verdict when it does not, which is the safe direction.
SPAN_DECADES = 3.0

# The reach a side must actually achieve before "the slice never crossed" is
# allowed to mean anything. A slice that stays flat over a factor of ten is a
# statement; one that stays flat over a factor of 1.1 is not, and would halt the
# run on a parameter nobody had looked at properly. This bites when the model
# stops evaluating close in and the walk cannot get out to the span.
MIN_REACH_DECADES = 1.0

# One side's verdict. Only "open" halts the run.
#   crossed      the slice rose above the threshold: profiling may proceed
#   open         the slice is still below the threshold decades out: PROVEN open
#   blocked      nothing on this side could be evaluated: no verdict
#   short-reach  the furthest evaluable point is too close in to conclude from
#   empty        there is no side to walk
_FAILING_STATES = ("open",)
_INCONCLUSIVE_STATES = ("blocked", "short-reach")


class UnidentifiableParameters(RuntimeError):
    """The slice screen proved one or more sides cannot be bounded.

    Carries the full report so the caller can print it rather than reconstruct
    it, and so the message is self-contained wherever it surfaces -- on a
    cluster this is read out of a Slurm log hours later, with nothing else to
    hand.
    """

    def __init__(self, failures, path=None, report=None):
        self.failures = list(failures)
        self.path = path
        self.report = report
        named = ", ".join(f"{f['name']} ({f['side']})" for f in self.failures[:6])
        more = (f" and {len(self.failures) - 6} more"
                if len(self.failures) > 6 else "")
        where = f"; see {path}" if path else ""
        super().__init__(
            f"the slice screen proved {len(self.failures)} side(s) unbounded: "
            f"{named}{more}. Fix each parameter to a literature-supported "
            f"value, or to zero, and drop it from the fit before "
            f"profiling{where}"
        )


def _se_for(wald_se, param_idx):
    """The Wald SE for one parameter in optimizer space, or None.

    None is a real answer, not a missing one: it means the Hessian was singular
    in this direction. The grid below falls back to a multiplicative offset,
    the same fallback ``_profile_grid_for`` uses.
    """
    if wald_se is None:
        return None
    try:
        cand = float(np.atleast_1d(wald_se)[param_idx])
    except (IndexError, TypeError, ValueError):
        return None
    return cand if np.isfinite(cand) and cand > 0 else None


def decades_from(p_opt, x, is_log):
    """How many decades *x* sits from the fitted value, or nan.

    The screen's own ruler, and the one the report quotes. A log-scaled
    parameter is stored as its own logarithm, so the distance is already in
    decades; a positive linear one is taken into logarithms for the same
    reason. A linear parameter at or below zero has no meaningful ratio and
    returns nan, which the verdict falls back from rather than guesses at.
    """
    if is_log:
        return abs(float(x) - float(p_opt))
    if p_opt > 0 and x > 0:
        return abs(float(np.log10(x)) - float(np.log10(p_opt)))
    return float("nan")


def _span_target(p_opt, sign, is_log, span_decades):
    """The value *span_decades* out from the optimum, or None if undefined."""
    if is_log:
        return float(p_opt) + sign * float(span_decades)
    if p_opt > 0:
        return float(p_opt) * float(10.0 ** (sign * span_decades))
    return None


def screen_target(p_opt, lb, ub, sign, is_log, span_decades=SPAN_DECADES):
    """How far this side walks: the declared bound or the span, whichever is further.

    Deliberately past the bound when the span reaches further. The bounds on
    this spec are a one-decade search box around the starting value, and a
    verdict read at a box edge is a verdict about the box. Evaluating outside it
    is safe -- there is no optimizer here, only an evaluation, and a value the
    model cannot take comes back as the failure sentinel and yields no verdict
    -- and it is what keeps an arbitrary bound from either manufacturing a halt
    or hiding one.
    """
    bound = lb if sign < 0 else ub
    span = _span_target(p_opt, sign, is_log, span_decades)
    if bound is None or not np.isfinite(bound):
        return span
    if span is None:
        return float(bound)
    # Whichever lies further out in the direction of travel.
    return float(max(bound, span) if sign > 0 else min(bound, span))


def screen_values(p_opt, lb, ub, se, sign, is_log, max_points=6, growth=2.0,
                  range_factor=2.0, span_decades=SPAN_DECADES):
    """Values to evaluate on one side, innermost first, ending at the target.

    Two jobs, and the second is what makes this a screen rather than a scan.

    Resolution near the crossing: the ladder starts at half a Wald SE and
    doubles its distance from the optimum, so it straddles the 1.96 SE where a
    crossing is expected. The slice crossing sits *inside* the profile crossing
    -- the slice is the steeper curve -- so a ladder placed for the profile
    brackets the slice comfortably.

    Half an SE rather than a whole one, because the innermost point is the one
    that certifies the inner bracket. A ladder starting at 1 SE puts its first
    point outside the slice crossing of any parameter tighter than the Hessian
    predicted, and then every evaluated point is above the threshold and
    nothing is certified. Starting inside costs one evaluation per side and is
    what makes the bracket usable.

    Reach: the last value is always :func:`screen_target` -- the declared bound
    or ``span_decades`` out, whichever is further -- even when the ladder would
    have stopped short of it. Without that point the screen concludes nothing. A
    slice that fails to cross within four standard errors is not evidence of
    anything; only one that fails to cross across decades is.

    The declared bound is included as a point of its own when the walk goes past
    it, because "does this cross inside the search box" is a separate and useful
    question from "does this cross at all" -- it is how a fit running against
    its own walls is detected.

    Distances are multiplied rather than added, via the same
    ``_next_extension_value`` the profile's own extension pass uses, so a
    log-scaled parameter walks in decades and a positive linear one walks in
    ratios. Adding a fixed offset instead collapses the interesting range on
    exactly the parameters that span decades.
    """
    from Engine.Optimize import _at_bound, _next_extension_value

    bound = lb if sign < 0 else ub
    has_bound = bound is not None and np.isfinite(bound)
    target = screen_target(p_opt, lb, ub, sign, is_log, span_decades)
    if target is None or not np.isfinite(target):
        return []

    # Nowhere to walk: the optimum already sits at the furthest point this side
    # would reach. Reporting "unbounded" from here would be nonsense.
    if _at_bound(p_opt, target, sign):
        return []

    # The walk is clipped to the target rather than to the bound, so a span
    # reaching past the bound is followed rather than truncated.
    walk_lb = target if sign < 0 else -np.inf
    walk_ub = target if sign > 0 else np.inf

    if se is not None:
        first = p_opt + sign * 0.5 * float(se)
    elif is_log:
        first = p_opt + sign * float(np.log10(range_factor))
    elif p_opt > 0:
        first = p_opt * range_factor if sign > 0 else p_opt / range_factor
    else:
        # A linear parameter at or below zero has no meaningful ratio, so the
        # step is absolute and scaled to the parameter's own size.
        first = p_opt + sign * max(abs(p_opt), 1.0)

    first = max(first, target) if sign < 0 else min(first, target)
    if not np.isfinite(first) or first == p_opt:
        return [float(target)]

    # Two slots are reserved: the target, and the declared bound when the walk
    # passes it. Both are appended below if the ladder has not landed on them.
    # ``_at_bound`` asks "has this reached or passed the bound", which every
    # point beyond it also satisfies, so the test here is a strict comparison:
    # what matters is whether the bound lies *inside* the walk.
    tol = 1e-12 * max(abs(bound), 1.0) if has_bound else 0.0
    beyond = has_bound and (target < bound - tol if sign < 0
                            else target > bound + tol)
    n_ladder = max(1, int(max_points) - (2 if beyond else 1))
    vals = [float(first)]
    while len(vals) < n_ladder:
        nxt = _next_extension_value(p_opt, vals[-1], walk_lb, walk_ub, sign,
                                    is_log, growth)
        if nxt is None:
            break
        vals.append(float(nxt))

    if beyond and not any(abs(v - bound) <= tol for v in vals):
        vals.append(float(bound))
    if not _at_bound(vals[-1], target, sign):
        vals.append(float(target))

    vals = sorted(set(vals), reverse=(sign < 0))
    return [v for v in vals if (v < p_opt if sign < 0 else v > p_opt)]


def _verdict(points, bound, threshold, p_opt, sign, is_log,
             min_reach_decades=MIN_REACH_DECADES):
    """One side's state, its reach, and the certified inner bracket.

    The verdict is read at the furthest point the model could actually be
    evaluated at, and it counts only if that point is at least
    ``min_reach_decades`` from the fitted value. Two things follow, and both are
    the point of reading it this way rather than at the declared bound.

    A bound narrower than the span no longer decides anything: the walk goes
    past it, so a slice that crosses just outside an arbitrary box is seen to
    cross and the side is cleared. And a model that stops evaluating part way
    out no longer yields a verdict by default: the reach shrinks to wherever the
    last finite value was, and if that is too close in the side is reported as
    unscreened rather than declared open.

    ``inner_bracket`` is the outermost value whose slice sits at or below the
    threshold with nothing above it in between -- so every point from the
    optimum out to it is certified to lie *inside* the confidence interval, by
    the same inequality the module relies on. No profile evaluation ever needs
    to be spent there. It is recorded rather than consumed: wiring it into the
    profile's opening grid is a separate change.
    """
    def _usable(p):
        return np.isfinite(p["dnll"]) and p["nll"] < FAILURE_VALUE

    finite = [p for p in points if _usable(p)]

    inner = None
    for p in points:                       # points are ordered outward
        if not _usable(p) or p["dnll"] > threshold:
            break
        inner = p["x_linear"]

    crossed_any = any(p["dnll"] > threshold for p in finite)
    has_bound = bound is not None and np.isfinite(bound)

    # Inside the declared search box, treated separately from the verdict: a
    # side that does not cross in the box but does cross outside it says the
    # fit's own walls are narrower than the interval.
    in_box = [p for p in finite
              if not has_bound
              or (p["x"] >= bound - 1e-12 if sign < 0
                  else p["x"] <= bound + 1e-12)]
    crossed_in_box = any(p["dnll"] > threshold for p in in_box)

    outer = finite[-1] if finite else None
    reach = (decades_from(p_opt, outer["x"], is_log)
             if outer is not None else float("nan"))
    # A linear parameter straddling zero has no decades. Fall back to the older
    # question -- did the walk get all the way out -- rather than guessing.
    reached_far = (reach >= min_reach_decades if np.isfinite(reach)
                   else (outer is not None and points
                         and outer["x"] == points[-1]["x"]))

    if not points:
        state = "empty"
    elif outer is None:
        # Nothing on this side could be evaluated at all. Not proof of
        # anything: a model may legitimately break away from its fitted region.
        state = "blocked"
    elif outer["dnll"] <= threshold:
        state = "open" if reached_far else "short-reach"
    elif crossed_any:
        state = "crossed"
    else:
        state = "short-reach"

    return {
        "state": state,
        "inner_bracket": inner,
        "max_dnll": max((p["dnll"] for p in finite), default=None),
        "reach": (float(outer["x_linear"]) if outer is not None else None),
        "reach_decades": (float(reach) if np.isfinite(reach) else None),
        "dnll_at_reach": (float(outer["dnll"]) if outer is not None else None),
        "bound": float(bound) if has_bound else None,
        "walked_past_bound": bool(
            has_bound and outer is not None
            and (outer["x"] < bound - 1e-12 if sign < 0
                 else outer["x"] > bound + 1e-12)),
        # The search box is narrower than the interval: nothing inside the
        # declared bound crossed, but something outside it did. Not a reason to
        # stop, and worth saying -- it means the fit was working against its
        # own walls.
        "box_too_narrow": bool(crossed_any and not crossed_in_box and has_bound),
        # The slice crossed on the way out and fell back below the threshold
        # further on. The proof stands -- the data do not exclude the far value
        # -- but the curve is not monotone and the report has to say so,
        # because "open" alone would misdescribe it.
        "non_monotone": bool(crossed_any and state == "open"),
        "points": points,
    }


def run_slice_screen(nll_batch, res_x, nll_at_optimum, param_names, bounds,
                     scales=None, wald_se=None, threshold=THRESHOLD,
                     max_points=6, growth=2.0, range_factor=2.0,
                     span_decades=SPAN_DECADES,
                     min_reach_decades=MIN_REACH_DECADES, verbose=True):
    """Evaluate every parameter's slice out across decades, as one batch.

    A slice point has no dependency on any other, so all of them go out
    together: this is the cheapest possible use of the pool, one evaluation per
    point with no optimizer wrapped around it.

    That shape has a second use. If evaluations far from the fitted values are
    pathologically slow -- the region where the integrator struggles and
    ``safe_simulate`` enters its retry ladder -- this finds out in minutes, and
    says so plainly, instead of the run discovering it hours into a profile
    batch where every stuck evaluation is buried inside a nuisance
    minimization.
    """
    from Engine.Optimize import _param_bounds

    res_x = np.asarray(res_x, dtype=float)
    scales = list(scales) if scales is not None else ["lin"] * len(param_names)

    plan, xs = [], []
    for i, name in enumerate(param_names):
        is_log = scales[i] == "log10"
        lb, ub = _param_bounds(bounds, i)
        se = _se_for(wald_se, i)
        for sign, side in ((-1, "lower"), (1, "upper")):
            vals = screen_values(res_x[i], lb, ub, se, sign, is_log,
                                 max_points=max_points, growth=growth,
                                 range_factor=range_factor,
                                 span_decades=span_decades)
            plan.append({"index": i, "name": name, "side": side, "sign": sign,
                         "is_log": is_log, "values": vals,
                         "p_opt": float(res_x[i]),
                         "bound": (lb if sign < 0 else ub)})
            for v in vals:
                x = res_x.copy()
                x[i] = v
                xs.append(x)

    if verbose:
        print(f"\n[screen] slice screen: {len(param_names)} parameter(s) x 2 "
              f"side(s) = {len(xs)} evaluation(s), submitted as one batch. "
              f"No nuisance optimization: each point is an upper bound on the "
              f"profile, which is all the screen needs.", flush=True)

    values = list(nll_batch(xs, label="slice-screen")) if xs else []

    report = {"threshold": float(threshold),
              "anchor": float(nll_at_optimum),
              "res_x": [float(v) for v in res_x],
              "param_names": list(param_names),
              "span_decades": float(span_decades),
              "min_reach_decades": float(min_reach_decades),
              "n_evaluations": len(xs),
              "parameters": {}}

    pos = 0
    for entry in plan:
        pts = []
        for v in entry["values"]:
            nll = float(values[pos])
            pos += 1
            pts.append({
                "x": float(v),
                "x_linear": float(10.0 ** v if entry["is_log"] else v),
                "nll": nll,
                "dnll": float(nll - nll_at_optimum),
            })
        side = _verdict(pts, entry["bound"], threshold, entry["p_opt"],
                        entry["sign"], entry["is_log"],
                        min_reach_decades=min_reach_decades)
        side["is_log"] = entry["is_log"]
        report["parameters"].setdefault(entry["name"], {})[entry["side"]] = side

    states = [s["state"] for sides in report["parameters"].values()
              for s in sides.values()]
    report["n_open"] = states.count("open")
    report["n_inconclusive"] = sum(states.count(s) for s in _INCONCLUSIVE_STATES)
    report["n_crossed"] = states.count("crossed")
    report["n_box_too_narrow"] = sum(
        1 for sides in report["parameters"].values()
        for s in sides.values() if s["box_too_narrow"])
    return report


def failing_sides(report):
    """Every side the screen proved unbounded, flattened for reporting."""
    out = []
    for name, sides in report.get("parameters", {}).items():
        for side, rec in sides.items():
            if rec["state"] in _FAILING_STATES:
                out.append(dict(rec, name=name, side=side))
    return out


def screen_summary(report):
    """The screen without its raw points, small enough to ride in the results.

    The evaluated points stay in ``slice_screen.json``; what a consumer of the
    results snapshot needs is the verdict per side and the certified inner
    bracket, so that a reader months later can tell a profile that was allowed
    to run from one that was never screened.
    """
    if not report:
        return None
    return {
        "threshold": report.get("threshold"),
        "n_evaluations": report.get("n_evaluations"),
        "n_open": report.get("n_open"),
        "n_inconclusive": report.get("n_inconclusive"),
        "n_crossed": report.get("n_crossed"),
        "n_box_too_narrow": report.get("n_box_too_narrow"),
        "parameters": {
            name: {side: {"state": rec["state"],
                          "inner_bracket": rec["inner_bracket"],
                          "reach": rec["reach"],
                          "reach_decades": rec["reach_decades"],
                          "dnll_at_reach": rec["dnll_at_reach"],
                          "walked_past_bound": rec["walked_past_bound"],
                          "box_too_narrow": rec["box_too_narrow"],
                          "bound": rec["bound"]}
                   for side, rec in sides.items()}
            for name, sides in report.get("parameters", {}).items()
        },
    }


def _decades_text(rec):
    d = rec.get("reach_decades")
    return f"{d:.2g} decade(s) out" if d is not None else "at its bound"


def print_screen_report(report):
    """The screen's findings, in the order a reader needs them."""
    failures = failing_sides(report)
    thr = report["threshold"]

    if failures:
        print(f"\n[screen] {len(failures)} side(s) are PROVEN unbounded:",
              flush=True)
        for f in failures:
            notes = []
            if f["walked_past_bound"]:
                notes.append("the walk went past the declared bound of "
                             f"{f['bound']:.6g}, so this verdict does not rest "
                             f"on it")
            if f["non_monotone"]:
                notes.append("the slice rose above the threshold further in "
                             "and came back down, so this curve is not "
                             "monotone and is worth looking at directly")
            tail = ("; " + "; ".join(notes)) if notes else ""
            print(f"    {f['name']} ({f['side']}): slice dNLL "
                  f"{f['dnll_at_reach']:.4g} at {f['reach']:.6g}, "
                  f"{_decades_text(f)}, still below {thr}. The profile there "
                  f"is no higher, so the data do not exclude it{tail}.")

    narrow = [(name, side, rec)
              for name, sides in report["parameters"].items()
              for side, rec in sides.items() if rec["box_too_narrow"]]
    if narrow:
        print(f"\n[screen] {len(narrow)} side(s) cross the threshold only "
              f"*outside* the declared bound, so the fit's own search box is "
              f"narrower than the interval (this does not halt the run):",
              flush=True)
        for name, side, rec in narrow[:20]:
            print(f"    {name} ({side}): nothing inside the bound "
                  f"{rec['bound']:.6g} reached {thr}; widen it if wider values "
                  f"are physical, or the fit is working against its own walls")
        if len(narrow) > 20:
            print(f"    ... and {len(narrow) - 20} more")

    inconclusive = [(name, side, rec)
                    for name, sides in report["parameters"].items()
                    for side, rec in sides.items()
                    if rec["state"] in _INCONCLUSIVE_STATES]
    if inconclusive:
        print(f"\n[screen] {len(inconclusive)} side(s) could not be screened "
              f"(this does not halt the run, and does not clear them either):",
              flush=True)
        for name, side, rec in inconclusive[:20]:
            if rec["state"] == "blocked":
                why = "nothing on this side could be evaluated at all"
            else:
                why = (f"the furthest evaluable point is only "
                       f"{_decades_text(rec)}, too close in to conclude from")
            print(f"    {name} ({side}): {why}")
        if len(inconclusive) > 20:
            print(f"    ... and {len(inconclusive) - 20} more")

    if not failures:
        print(f"\n[screen] every screened side rose above {thr}. That is "
              f"permission to profile, not a finding: the slice is an upper "
              f"bound, so it can prove a parameter unbounded but never prove "
              f"one identifiable.", flush=True)


def save_screen(report, ckpt_dir):
    """Write the screen beside the points it will govern. Returns the path.

    Atomically, and never fatally: several links may share the directory, and a
    screen that cannot be cached is a re-run of ten minutes, not a reason to
    lose the run.
    """
    if not ckpt_dir:
        return None
    path = os.path.join(ckpt_dir, SCREEN_FILENAME)
    tmp = f"{path}.{os.getpid()}.tmp"
    try:
        os.makedirs(ckpt_dir, exist_ok=True)
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=1)
        os.replace(tmp, path)
    except OSError:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        return None
    return path


def load_screen(ckpt_dir, param_names, res_x, threshold=THRESHOLD,
                span_decades=SPAN_DECADES,
                min_reach_decades=MIN_REACH_DECADES):
    """A screen already run for this exact fit, or None.

    The checkpoint directory is keyed by model hash, spec hash and the optimum,
    so a file found in it already belongs to this run; the fields are checked
    anyway because the cost of re-running the screen is ten minutes and the
    cost of honouring someone else's is a wrong verdict on identifiability.
    """
    if not ckpt_dir:
        return None
    path = os.path.join(ckpt_dir, SCREEN_FILENAME)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            report = json.load(fh)
    except (OSError, ValueError):
        return None

    if list(report.get("param_names") or []) != list(param_names):
        return None
    # Every number that can change a verdict is part of the key. A screen run
    # over one decade must not answer for a run asked to look over three.
    for key, want in (("threshold", threshold),
                      ("span_decades", span_decades),
                      ("min_reach_decades", min_reach_decades)):
        try:
            if abs(float(report.get(key)) - float(want)) > 1e-12:
                return None
        except (TypeError, ValueError):
            return None
    stored = np.asarray(report.get("res_x") or [], dtype=float)
    current = np.asarray(res_x, dtype=float)
    if stored.shape != current.shape or not np.allclose(stored, current,
                                                        rtol=0, atol=1e-12):
        return None
    return report


def screen_or_raise(nll_batch, res_x, nll_at_optimum, param_names, bounds,
                    scales=None, wald_se=None, ckpt_dir=None,
                    threshold=THRESHOLD, max_points=6, growth=2.0,
                    range_factor=2.0, span_decades=SPAN_DECADES,
                    min_reach_decades=MIN_REACH_DECADES, verbose=True):
    """Run the screen (or reuse one), report it, and stop the run if it failed.

    Raises :class:`UnidentifiableParameters` when any side is proven unbounded.
    Returns the report otherwise, whose ``inner_bracket`` values are certified
    to lie inside the confidence interval.
    """
    report = load_screen(ckpt_dir, param_names, res_x, threshold,
                         span_decades, min_reach_decades)
    if report is not None:
        if verbose:
            print(f"\n[screen] reusing the slice screen already run for this "
                  f"fit ({report.get('n_evaluations', 0)} evaluation(s)); "
                  f"nothing is recomputed.", flush=True)
        path = os.path.join(ckpt_dir, SCREEN_FILENAME)
    else:
        report = run_slice_screen(
            nll_batch, res_x, nll_at_optimum, param_names, bounds,
            scales=scales, wald_se=wald_se, threshold=threshold,
            max_points=max_points, growth=growth, range_factor=range_factor,
            span_decades=span_decades, min_reach_decades=min_reach_decades,
            verbose=verbose,
        )
        path = save_screen(report, ckpt_dir)

    if verbose:
        print_screen_report(report)

    failures = failing_sides(report)
    if failures:
        if verbose:
            print(f"\n[screen] HALTED before any profile point was started. "
                  f"A parameter the data cannot bound has to be fixed to a "
                  f"defensible value -- from the literature, or to zero -- and "
                  f"dropped from the fit. That rests on evidence outside this "
                  f"run, so it is not a decision to take here or to defer, and "
                  f"there is deliberately no flag to skip this check. Profiling "
                  f"these parameters would spend days to report the open "
                  f"intervals the screen has just proved in minutes.",
                  flush=True)
        raise UnidentifiableParameters(failures, path=path, report=report)

    return report

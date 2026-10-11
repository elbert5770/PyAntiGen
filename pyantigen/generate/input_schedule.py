"""One exact description of everything a model does on a clock.

A model's time-driven inputs -- hourly CSF draws, dosing ladders, a measured
plasma curve -- used to be written as Antimony text by several generators, each
with its own copy of the switch times. Those copies drift apart:

* ``{i} + V_LP/Q_CSF`` in an event and ``0.344926564668731`` in a piecewise knot
  are the same instant to 15 digits and different doubles, so a root finder sees
  two crossings 1e-15 apart;
* a knot computed from a hard-coded ``Q_CSF = 17.395`` stays put when the fitted
  ``Q_CSF`` moves, while the events move with it;
* two events at the same instant are two root functions at the same instant.

``InputSchedule`` is the single source. It stores every switch time exactly, as

    anchor + sum(coefficient * moving_quantity)

with ``anchor`` and the coefficients rational (``fractions.Fraction``) and each
*moving quantity* a named arithmetic expression in model parameters
(``t_refill = V_LP/Q_CSF``). Two switch times are the same switch if and only if
their ``(anchor, terms)`` are equal, so de-duplication is symbolic and never
depends on a tolerance. From the one description the schedule can

* evaluate every switch time for a parameter set (``times``) and its exact
  gradient with respect to the parameters (``gradient``, forward-mode, no
  finite differences, no sympy);
* check that the order of the switches is the one the schedule was designed for
  and that no two distinct switches are closer than the solver can tell apart
  (``validate``) -- the check an optimizer's bounds must satisfy;
* cut a run into segments whose end points are switches, observation times
  included (``segments``), which is what an event-free backend integrates;
* write the equivalent Antimony ``at (time >= ...)`` events (``to_antimony``),
  with simultaneous switches merged into one event, so the existing RoadRunner
  path keeps working from the same numbers.

Only switches that depend on the clock and on parameters are covered. A trigger
that depends on a *state* (``at (S < 1e-6)``) has no place here.

Simultaneous assignments
------------------------
Everything that happens at one switch time becomes one event. SBML evaluates all
the assignments of an event, and of simultaneous events, with the values the
model had at the trigger time (``useValuesFromTriggerTime="true"``, which is
what Antimony writes), so ``Q_Leak = Q_CSF - Q_SN`` reads the ``Q_SN`` from
*before* a coincident ``Q_SN = 0``. That is well defined, but it is easy to read
the other way, so :meth:`InputSchedule.validate` reports every such read, and
two assignments to the same variable at one switch time are refused.
"""
import ast
import math
from dataclasses import dataclass
from fractions import Fraction


class ScheduleError(ValueError):
    """The schedule is inconsistent for the parameters it was asked about."""


# ---------------------------------------------------------------------------
# Exact numbers
# ---------------------------------------------------------------------------

def _frac(x):
    """*x* as a Fraction, reading a float by its shortest decimal form.

    ``0.1`` means one tenth, not the binary double nearest to it; the double is
    recovered, correctly rounded, by ``float(Fraction(1, 10))``.
    """
    if isinstance(x, Fraction):
        return x
    if isinstance(x, bool):
        raise TypeError("a bool is not a time")
    if isinstance(x, int):
        return Fraction(x)
    x = float(x)
    if not math.isfinite(x):
        raise ScheduleError(f"non-finite time {x!r}")
    return Fraction(repr(x))


def _num_text(a):
    """Antimony text for a Fraction that parses back to ``float(a)``."""
    if a.denominator == 1:
        return str(a.numerator)
    return repr(float(a))


# ---------------------------------------------------------------------------
# Forward-mode derivatives of the moving quantities
# ---------------------------------------------------------------------------

class _Dual:
    """A value and its partial derivatives with respect to named parameters."""
    __slots__ = ("v", "g")

    def __init__(self, v, g=None):
        self.v = float(v)
        self.g = g or {}

    @staticmethod
    def _lift(o):
        return o if isinstance(o, _Dual) else _Dual(o)

    @staticmethod
    def _comb(a, b, fa, fb):
        out = {}
        for k in set(a.g) | set(b.g):
            out[k] = fa * a.g.get(k, 0.0) + fb * b.g.get(k, 0.0)
        return out

    def __add__(self, o):
        o = self._lift(o)
        return _Dual(self.v + o.v, self._comb(self, o, 1.0, 1.0))
    __radd__ = __add__

    def __sub__(self, o):
        o = self._lift(o)
        return _Dual(self.v - o.v, self._comb(self, o, 1.0, -1.0))

    def __rsub__(self, o):
        return self._lift(o).__sub__(self)

    def __mul__(self, o):
        o = self._lift(o)
        return _Dual(self.v * o.v, self._comb(self, o, o.v, self.v))
    __rmul__ = __mul__

    def __truediv__(self, o):
        o = self._lift(o)
        q = self.v / o.v
        return _Dual(q, self._comb(self, o, 1.0 / o.v, -q / o.v))

    def __rtruediv__(self, o):
        return self._lift(o).__truediv__(self)

    def __neg__(self):
        return _Dual(-self.v, {k: -x for k, x in self.g.items()})

    def __pos__(self):
        return self

    def __pow__(self, o):
        o = self._lift(o)
        p = self.v ** o.v
        ga = o.v * self.v ** (o.v - 1.0) if self.v != 0.0 or o.v != 0.0 else 0.0
        gb = p * math.log(self.v) if o.g else 0.0
        return _Dual(p, self._comb(self, o, ga, gb))

    def __rpow__(self, o):
        return self._lift(o).__pow__(self)


_BINOPS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.Pow: lambda a, b: a ** b,
}


def _parse_expression(text):
    """Parse *text* and refuse anything that is not plain arithmetic."""
    try:
        tree = ast.parse(str(text), mode="eval")
    except SyntaxError as exc:
        raise ScheduleError(f"cannot parse {text!r}: {exc.msg}") from None
    for node in ast.walk(tree):
        if isinstance(node, (ast.Expression, ast.Name, ast.Load, ast.BinOp,
                             ast.UnaryOp, ast.UAdd, ast.USub, *_BINOPS)):
            continue
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) \
                and not isinstance(node.value, bool):
            continue
        raise ScheduleError(
            f"{text!r}: only numbers, parameter names and + - * / ** are allowed")
    return tree


def _symbols(tree):
    return {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}


def _eval_dual(node, lookup):
    if isinstance(node, ast.Expression):
        return _eval_dual(node.body, lookup)
    if isinstance(node, ast.Constant):
        return _Dual(node.value)
    if isinstance(node, ast.Name):
        return lookup(node.id)
    if isinstance(node, ast.UnaryOp):
        v = _eval_dual(node.operand, lookup)
        return -v if isinstance(node.op, ast.USub) else v
    a = _eval_dual(node.left, lookup)
    b = _eval_dual(node.right, lookup)
    return _BINOPS[type(node.op)](a, b)


# ---------------------------------------------------------------------------
# Switch times
# ---------------------------------------------------------------------------

class Breakpoint:
    """``anchor + sum(coeff * moving[name])``, immutable and exactly comparable.

    Equal breakpoints are the same switch. Which of two *different* breakpoints
    comes first depends on the parameters, so ordering lives in
    :class:`InputSchedule`, not here.
    """
    __slots__ = ("anchor", "terms")

    def __init__(self, anchor=0, terms=()):
        merged = {}
        for name, c in (terms.items() if isinstance(terms, dict) else terms):
            merged[name] = merged.get(name, Fraction(0)) + _frac(c)
        self.anchor = _frac(anchor)
        self.terms = tuple(sorted((n, c) for n, c in merged.items() if c != 0))

    @property
    def fixed(self):
        """True if this switch does not depend on any parameter."""
        return not self.terms

    def _other(self, o):
        return o if isinstance(o, Breakpoint) else Breakpoint(o)

    def __add__(self, o):
        o = self._other(o)
        return Breakpoint(self.anchor + o.anchor, self.terms + o.terms)
    __radd__ = __add__

    def __neg__(self):
        return Breakpoint(-self.anchor, tuple((n, -c) for n, c in self.terms))

    def __sub__(self, o):
        return self + (-self._other(o))

    def __rsub__(self, o):
        return self._other(o) - self

    def __mul__(self, k):
        k = _frac(k)
        return Breakpoint(self.anchor * k, tuple((n, c * k) for n, c in self.terms))
    __rmul__ = __mul__

    def __eq__(self, o):
        return isinstance(o, Breakpoint) and (self.anchor, self.terms) == (o.anchor, o.terms)

    def __hash__(self):
        return hash((self.anchor, self.terms))

    def __repr__(self):
        parts = [_num_text(self.anchor)]
        parts += [f"{'+' if c > 0 else '-'} {_num_text(abs(c))}*{n}" for n, c in self.terms]
        return "Breakpoint(" + " ".join(parts) + ")"


@dataclass(frozen=True)
class Segment:
    """The stretch of the clock between two consecutive switches."""
    index: int
    start: Breakpoint
    end: Breakpoint

    @property
    def duration(self):
        """``end - start`` as a Breakpoint (its value and gradient are the duration's)."""
        return self.end - self.start


class InputSchedule:
    """Every clock-driven change of a model, described once and exactly.

    Parameters
    ----------
    params : dict
        Nominal values of every model parameter that a moving quantity or an
        assignment mentions. They fix the intended order of the switches;
        every query takes an optional ``params`` to ask about other values.

    Example
    -------
    The v4 SILK CSF draw: three switches an hour, two of which move with
    ``Q_CSF`` and ``V_LP``::

        s = InputSchedule({"Q_CSF": 17.395, "V_LP": 6.0, "t_CSFdraw": 0.1, "f_SN": 0.1})
        t_draw = s.moving("t_draw", "t_CSFdraw")
        t_refill = s.moving("t_refill", "V_LP/Q_CSF")
        s.periodic(first=0, period=1, count=48, phases=[
            (0,        {"Q_SN": "0.0", "Q_LP": "Q_CSF", "Q_refill": "Q_CSF - V_LP/t_CSFdraw"}),
            (t_draw,   {"Q_refill": "Q_CSF"}),
            (t_refill, {"Q_SN": "f_SN*Q_CSF", "Q_LP": "Q_Leak", "Q_refill": "0"}),
        ])
    """

    def __init__(self, params):
        self.params = {k: float(v) for k, v in params.items()}
        self._moving = {}                 # name -> (text, parsed tree)
        self._steps = []                  # (Breakpoint, {var: text})
        self._inputs = []                 # (name, times, values, before, after)
        self._reference_order = None

    # -- construction ------------------------------------------------------

    def moving(self, name, expression):
        """Declare a quantity that moves with the parameters and return its Breakpoint.

        The returned object is ``0 + 1*name``; add it to numbers and to other
        breakpoints to build switch times.
        """
        if not name.isidentifier():
            raise ScheduleError(f"{name!r} is not an identifier")
        if name in self._moving:
            raise ScheduleError(f"moving quantity {name!r} already declared")
        tree = _parse_expression(expression)
        missing = _symbols(tree) - set(self.params)
        if missing:
            raise ScheduleError(
                f"{name} = {expression!r} uses {sorted(missing)}, which have no nominal value")
        self._moving[name] = (str(expression).strip(), tree)
        self._reference_order = None
        return Breakpoint(0, {name: 1})

    def step(self, when, assignments):
        """At the switch *when*, set each variable in *assignments* (name -> expression text).

        *when* is a Breakpoint or a number. Expressions are kept as text and
        written to the model unchanged, so they may use anything Antimony can.
        """
        when = when if isinstance(when, Breakpoint) else Breakpoint(when)
        self._check_names(when)
        self._steps.append((when, {str(k): str(v) for k, v in assignments.items()}))
        self._reference_order = None

    def periodic(self, first, period, count, phases):
        """Repeat *phases* ``count`` times, ``period`` apart, starting at *first*.

        *phases* is a list of ``(offset, assignments)``; the offset is a number
        or a Breakpoint (``0`` for the start of the period).
        """
        first, period = _frac(first), _frac(period)
        for k in range(int(count)):
            base = Breakpoint(first + k * period)
            for offset, assignments in phases:
                self.step(base + offset, assignments)

    def linear_input(self, name, times, values, default_before=0.0, default_after=None):
        """A measured curve, linear between its points, as a function of the clock.

        Its knots are switches (the integrator must not step across a kink), but
        fixed ones. ``times`` must be strictly increasing and further apart than
        0.01 h: ``generate_antimony_piecewise`` averages closer points, and a
        knot the schedule does not know about would defeat the purpose.
        """
        times = [float(t) for t in times]
        values = [float(v) for v in values]
        if len(times) != len(values) or len(times) < 2:
            raise ScheduleError("times and values must have the same length, at least 2")
        gaps = [b - a for a, b in zip(times, times[1:])]
        if any(g <= 0.01 for g in gaps):
            raise ScheduleError(
                f"{name}: times must be strictly increasing with gaps above 0.01 h "
                "(merge or drop closer points first)")
        self._inputs.append((name, times, values, default_before, default_after))
        self._reference_order = None

    def _check_names(self, bp):
        unknown = [n for n, _ in bp.terms if n not in self._moving]
        if unknown:
            raise ScheduleError(f"unknown moving quantity {unknown}: declare it with moving() first")

    # -- exact switch times ------------------------------------------------

    def _lookup(self, params, wrt):
        def lookup(name):
            if name not in params:
                raise ScheduleError(f"no value for parameter {name!r}")
            v = float(params[name])
            return _Dual(v, {name: 1.0} if name in wrt else {})
        return lookup

    def _moving_values(self, params, wrt=()):
        lookup = self._lookup(params, set(wrt))
        return {n: _eval_dual(tree, lookup) for n, (_t, tree) in self._moving.items()}

    def _value(self, bp, mv):
        """Value of *bp*; same left-to-right arithmetic as the text :meth:`to_antimony` writes."""
        self._check_names(bp)
        t = float(bp.anchor)
        for n, c in bp.terms:
            t = t + (mv[n].v if c == 1 else (-mv[n].v if c == -1 else float(c) * mv[n].v))
        return t

    def time(self, bp, params=None):
        """The switch time of *bp* for *params* (default: the nominal parameters)."""
        mv = self._moving_values(self.params if params is None else params)
        t = self._value(bp, mv)
        if not math.isfinite(t):
            raise ScheduleError(f"{bp} is not finite for these parameters")
        return t

    def gradient(self, bp, params=None, wrt=None):
        """``{parameter: d(switch time)/d(parameter)}``, exact.

        *wrt* defaults to every parameter some moving quantity mentions.
        """
        params = self.params if params is None else params
        if wrt is None:
            wrt = sorted({s for _t, tree in self._moving.values() for s in _symbols(tree)})
        mv = self._moving_values(params, wrt)
        self._check_names(bp)
        out = {p: 0.0 for p in wrt}
        for n, c in bp.terms:
            for p, d in mv[n].g.items():
                out[p] += float(c) * d
        return out

    # -- the set of switches ----------------------------------------------

    def events(self):
        """The assignments grouped by switch: ``[(Breakpoint, {variable: text}), ...]``.

        Steps at the same switch are merged. A variable assigned twice at one
        switch is an error rather than a coin toss.
        """
        merged = {}
        for bp, assignments in self._steps:
            slot = merged.setdefault(bp, {})
            for var, text in assignments.items():
                if var in slot and slot[var] != text:
                    raise ScheduleError(
                        f"{var} is assigned both {slot[var]!r} and {text!r} at {bp}")
                slot[var] = text
        return list(merged.items())

    def breakpoints(self):
        """Every distinct switch -- event times and measured-curve knots -- unordered."""
        found = {bp for bp, _a in self.events()}
        for _name, times, _v, _b, _a in self._inputs:
            found.update(Breakpoint(t) for t in times)
        return found

    def _ordered(self, bps, params):
        mv = self._moving_values(params)
        # Coincident switches: a moving one first, so a fixed time at the same instant (an
        # observation, a curve knot) comes after it and sees the state after the switch.
        # That is what the runner reports, and it makes the gradient of such an
        # observation the right-hand one on purpose, not by how the breakpoints are spelled.
        timed = sorted(((self._value(bp, mv), bp) for bp in bps),
                       key=lambda p: (p[0], p[1].fixed, repr(p[1])))
        return timed

    def order(self, params=None, extra=()):
        """The distinct switches, earliest first, for *params*."""
        params = self.params if params is None else params
        return [bp for _t, bp in self._ordered(set(self.breakpoints()) | set(extra), params)]

    def times(self, params=None):
        """Sorted switch times in hours. Distinct switches never merge here.

        Use :meth:`validate` to find out whether two of them are too close to
        be told apart.
        """
        params = self.params if params is None else params
        return [t for t, _bp in self._ordered(self.breakpoints(), params)]

    def ordered_events(self, params=None):
        """``[(time, {variable: text}), ...]`` in time order for *params*.

        The steps an event-free backend applies, with the same doubles as
        :meth:`times` and the Antimony triggers. Switches that are only
        measured-curve knots carry no assignments and are not listed.
        """
        params = self.params if params is None else params
        by_bp = dict(self.events())
        return [(t, by_bp[bp]) for t, bp in self._ordered(set(by_bp), params)]

    @staticmethod
    def evaluate(text, lookup):
        """Value of one assignment's right-hand side.

        *lookup* maps a name to its current value. Only numbers, names and
        ``+ - * / **`` are understood; free-form Antimony raises ScheduleError
        rather than being guessed at.
        """
        tree = _parse_expression(text)
        return _eval_dual(tree, lambda name: _Dual(float(lookup(name)))).v

    # -- checks ------------------------------------------------------------

    def validate(self, params=None, min_gap=1e-9):
        """Raise ScheduleError if the schedule cannot be integrated at *params*.

        Checks, for the given parameters:

        * every moving quantity and switch time is finite;
        * the order of the *steps* (switches that assign something) equals the
          order at the nominal parameters: a step that has crossed another has
          changed the structure of the model, and the gradient there is not the
          one being optimized;
        * no two distinct steps, and no two distinct fixed times, are within
          ``max(min_gap, 1e-12 * |t|)``.

        A measured-curve knot or an observation time is only a place the
        integrator must stop. A moving step may cross one, or land on it, at any
        parameters: nothing about the model changes, and ``times``/``segments``
        merge or keep the coincident cut points as the caller needs.

        Returns a list of warnings (reads of a variable that a coincident
        assignment overwrites). It is the check to wrap around an optimizer's
        bounds: the feasible region of a fitted parameter is the set where this
        does not raise.
        """
        params = self.params if params is None else params
        steps = {bp for bp, _a in self.events()}
        reference = [bp for bp in self.order(self.params) if bp in steps]
        mv = self._moving_values(params)
        for n, d in mv.items():
            if not math.isfinite(d.v):
                raise ScheduleError(f"moving quantity {n} is not finite")
        timed = self._ordered(set(self.breakpoints()), params)
        now = [bp for _t, bp in timed if bp in steps]
        if now != reference:
            moved = [repr(a) for a, b in zip(now, reference) if a != b][:2]
            raise ScheduleError(
                "the order of the switches differs from the nominal order "
                f"(first differences: {moved}); parameters are outside the feasible region")
        self._check_gaps(timed, steps, min_gap)
        return self._read_after_write_warnings()

    @staticmethod
    def _check_gaps(timed, steps, min_gap):
        """Refuse distinct steps, or distinct fixed times, too close to separate.

        A moving step against a fixed knot is allowed to be arbitrarily close.
        """
        for group in ([(t, bp) for t, bp in timed if bp in steps],
                      [(t, bp) for t, bp in timed if bp.fixed]):
            for (t0, b0), (t1, b1) in zip(group, group[1:]):
                if t1 - t0 <= max(min_gap, 1e-12 * max(abs(t0), abs(t1))):
                    raise ScheduleError(
                        f"{b0} and {b1} are {t1 - t0:.3g} h apart "
                        f"(t = {t0:.12g}); the solver cannot tell them apart")

    def _read_after_write_warnings(self):
        out = []
        for bp, assignments in self.events():
            written = set(assignments)
            for var, text in assignments.items():
                try:
                    reads = _symbols(_parse_expression(text))
                except ScheduleError:
                    continue            # free-form Antimony: nothing to say
                clash = (reads & written) - {var}
                if clash:
                    out.append(
                        f"at {bp}: {var} = {text} reads {sorted(clash)}, assigned at the same "
                        "switch; it sees the value from before the switch")
        return out

    # -- segments ----------------------------------------------------------

    def segments(self, t_start, t_end, observation_times=(), params=None):
        """Cut [t_start, t_end] at every switch and every observation time.

        Observation times become fixed switches, so a segment boundary is
        either a fixed clock time or a moving one, and outputs are only ever
        read at fixed ones. The segments are ordered for *params*, which is
        validated first.
        """
        t_start, t_end = _frac(t_start), _frac(t_end)
        if not t_end > t_start:
            raise ScheduleError("t_end must be after t_start")
        params = self.params if params is None else params
        self.validate(params)
        extra = {Breakpoint(t_start), Breakpoint(t_end)} | {Breakpoint(t) for t in observation_times}
        timed = self._ordered(set(self.breakpoints()) | extra, params)
        lo, hi = float(t_start), float(t_end)
        # a switch that sits exactly on an end point is that end point
        timed = [(t, bp) for t, bp in timed if lo <= t <= hi]
        self._check_gaps(timed, {bp for bp, _a in self.events()}, 1e-9)
        bps = [bp for _t, bp in timed]
        return [Segment(i, a, b) for i, (a, b) in enumerate(zip(bps, bps[1:]))]

    # -- Antimony ----------------------------------------------------------

    def _trigger_text(self, bp):
        self._check_names(bp)
        text = _num_text(bp.anchor)
        for n, c in bp.terms:
            expr = self._moving[n][0]
            if c == 1:
                text += f" + ({expr})"
            elif c == -1:
                text += f" - ({expr})"
            else:
                text += f" + {_num_text(c)}*({expr})"
        return text

    def to_antimony(self, events=True):
        """The equivalent Antimony: one event per switch, then the measured curves.

        Events come out in time order for the nominal parameters, and trigger
        times are the very expressions :meth:`time` evaluates. With
        ``events=False`` only the measured curves are written, which is what a
        model needs when an event-free backend (``pyantigen.engine.Segmented``)
        applies the steps itself and cuts the run at every curve knot.
        """
        from pyantigen.generate.data_interpolation import generate_antimony_piecewise
        lines = []
        by_bp = dict(self.events())
        for bp in self.order():
            if not events or bp not in by_bp:
                continue
            body = ", ".join(f"{v} = {e}" for v, e in by_bp[bp].items())
            lines.append(f"at (time >= {self._trigger_text(bp)}): {body}")
        text = "\n".join(lines) + ("\n" if lines else "")
        for name, times, values, before, after in self._inputs:
            kwargs = {"default_before": before}
            if after is not None:
                kwargs["default_after"] = after
            text += generate_antimony_piecewise(
                times, values, data_name=name, time_tol=0.0, **kwargs) + "\n"
        return text

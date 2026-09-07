"""When, in absolute simulation hours, the right-hand side stops being smooth.

The block splitter in ``Modules.Solver_settings`` cuts long simulations into
pieces so no single ``r.simulate()`` call has to cross too many discontinuities.
It currently cuts on a wall-clock grid, which is a proxy for "how many doses are
in this interval" that only holds for evenly-spaced chronic dosing. This module
supplies the quantity the splitter actually wants: the times at which something
discontinuous happens.

Four things produce a time discontinuity in these models, and only the first is
what a naive scan would find.

* **Events with a constant time trigger.** ``generate_silk_events`` writes 49
  hourly ``f_L`` steps plus two tail events; the antibody-trial generators write
  ``at (time >= {age}*365*24 + {week}*7*24)``.

* **Events whose trigger contains model symbols.** The subcutaneous generators
  end an infusion at ``({t_start}) + SubCut_D1``, and ``SubCut_D1`` is a *fitted*
  parameter (``PK_GANTENERUMAB_names``). RoadRunner evaluates the trigger against
  the current parameter value at simulation time, which is why those arms can run
  with ``events_depend_on_opt_param`` False -- and why a list of event times
  computed once at x0 goes stale as the fit moves. Times are therefore resolved
  lazily, on every call, against the model's current values. Evaluating a hundred
  two-node expressions costs microseconds against a multi-second simulation, so
  there is nothing to gain by caching the numbers and a correctness bug to lose.

* **Arithmetic over several symbols.** The v4 SILK model triggers CSF draws at
  ``{i} + t_CSFdraw`` and ``{i} + V_LP/Q_CSF``, so the evaluator has to handle
  expressions rather than constants.

* **Time-based piecewise assignment rules.** ``generate_antimony_piecewise``
  emits ``X := piecewise(...)``, which is an assignment rule and *not* an event.
  RoadRunner does not root-find those, so CVODE steps straight over the
  breakpoints and fits a high-order polynomial across a kink -- numerically worse
  than an event, and invisible to anything that only enumerates events. The v4
  model drives ``V_SP3`` through 144 such breakpoints.

Not every ``piecewise`` is a time discontinuity: the model's own rules file
carries 26 guards of the form ``piecewise(1, AB40Total_BrainISF < 1e-12, ...)``,
which switch on *state*. Those are a real numerical hazard too, but no
time-based splitter can help with them, so they are filtered out here rather
than reported as cut candidates.

The times come from the compiled model via libSBML rather than from the
generated Antimony text. That catches every event regardless of which generator
wrote it, survives any syntax the generators use, and describes the model that
actually runs.

Fail open, never silently: if any trigger cannot be resolved to a number,
:meth:`EventTimeTable.times` returns None and the splitter falls back to its
wall-clock cap. A missed event would let a cut land exactly on a discontinuity,
which is the one place a cut must never go.
"""

import math


# ---------------------------------------------------------------------------
# A compiled arithmetic expression
# ---------------------------------------------------------------------------
#
# libSBML AST nodes are compiled into plain tuples rather than being kept as
# handed over. Two reasons, both learned the hard way:
#
#   * The nodes are owned by the SBMLDocument. Holding them past the document's
#     lifetime is a dangling pointer, and the table outlives the parse.
#   * ``node.getType()`` returns an opaque SWIG pointer in this libSBML build,
#     so the AST_* integer constants do not compare equal to it and any code
#     written against them silently matches nothing. The ``is*`` predicates are
#     the portable interface.
#
# Node forms: ('num', float) | ('sym', name) | ('op', char, left, right)
#             | ('neg', child) | ('fn', name, *children)


class _Unevaluable(Exception):
    """A trigger this module declines to guess at."""


_BINARY = {
    '+': lambda a, b: a + b,
    '-': lambda a, b: a - b,
    '*': lambda a, b: a * b,
    '/': lambda a, b: a / b,
    '^': lambda a, b: a ** b,
}

# Functions that are pure, scalar and unambiguous in arity. Antimony turns even
# ``2^3`` into a ``power`` function node rather than an operator, so without at
# least this much a perfectly ordinary trigger would send the whole table to
# None. ``log`` is left out on purpose: its arity differs between MathML and
# SBML L3, and guessing wrong would put an event at the wrong instant, which is
# worse than declining to place it at all.
_FUNCTIONS = {
    'power':   (2, lambda a, b: a ** b),
    'root':    (2, lambda degree, x: x ** (1.0 / degree)),
    'sqrt':    (1, math.sqrt),
    'abs':     (1, abs),
    'ceiling': (1, math.ceil),
    'floor':   (1, math.floor),
    'exp':     (1, math.exp),
    'ln':      (1, math.log),
    'log10':   (1, math.log10),
    'min':     (2, min),
    'max':     (2, max),
}


def _compile(node):
    """libSBML AST -> tuple tree, or raise :class:`_Unevaluable`."""
    if node is None:
        raise _Unevaluable("empty node")

    if node.isNumber():
        return ('num', float(node.getValue()))

    if node.isName():
        name = node.getName()
        if not name:
            raise _Unevaluable("unnamed symbol")
        return ('sym', name)

    if node.isUMinus():
        if node.getNumChildren() != 1:
            raise _Unevaluable("malformed unary minus")
        return ('neg', _compile(node.getChild(0)))

    if node.isUPlus():
        if node.getNumChildren() != 1:
            raise _Unevaluable("malformed unary plus")
        return _compile(node.getChild(0))

    if node.isOperator():
        char = node.getCharacter()
        if char not in _BINARY:
            raise _Unevaluable(f"operator {char!r}")
        if node.getNumChildren() != 2:
            raise _Unevaluable(f"operator {char!r} with "
                               f"{node.getNumChildren()} children")
        return ('op', char, _compile(node.getChild(0)),
                _compile(node.getChild(1)))

    if node.isFunction():
        name = node.getName()
        spec = _FUNCTIONS.get(name)
        if spec is None:
            raise _Unevaluable(f"function {name!r}")
        arity, _fn = spec
        if node.getNumChildren() != arity:
            raise _Unevaluable(f"function {name!r} with "
                               f"{node.getNumChildren()} argument(s)")
        return ('fn', name) + tuple(_compile(node.getChild(i))
                                    for i in range(arity))

    # Constants (pi, exponentiale, avogadro) evaluate fine, but nothing in these
    # models uses one in a time trigger, so treating them as unknown costs
    # nothing and keeps the evaluator honest about what it has actually seen.
    raise _Unevaluable("unsupported node")


def _evaluate(node, lookup):
    kind = node[0]
    if kind == 'num':
        return node[1]
    if kind == 'sym':
        return lookup(node[1])
    if kind == 'neg':
        return -_evaluate(node[1], lookup)
    if kind == 'fn':
        _arity, fn = _FUNCTIONS[node[1]]
        return fn(*(_evaluate(arg, lookup) for arg in node[2:]))
    return _BINARY[node[1]](_evaluate(node[2], lookup),
                            _evaluate(node[3], lookup))


def _symbols(node, out=None):
    out = set() if out is None else out
    kind = node[0]
    if kind == 'sym':
        out.add(node[1])
    elif kind == 'neg':
        _symbols(node[1], out)
    elif kind == 'op':
        _symbols(node[2], out)
        _symbols(node[3], out)
    elif kind == 'fn':
        for arg in node[2:]:
            _symbols(arg, out)
    return out


# ---------------------------------------------------------------------------
# Finding the thresholds
# ---------------------------------------------------------------------------

def _is_time(node):
    """Is this AST node the simulation-time symbol?

    ``time`` arrives as a csymbol whose ``isName`` is true and whose name is
    'time'. The AST type code would say so more directly if it were comparable
    (see the note above), so the name is what gets checked.
    """
    return bool(node is not None and node.isName() and node.getName() == 'time')


def _threshold_from_relational(node):
    """``time >= expr`` (either way round) -> compiled *expr*, else None.

    Which comparison operator it is does not matter. Any relation between time
    and an expression marks the instant the relation flips, and that instant is
    the value of the expression -- so the operator carries no information the
    splitter needs, and not reading it avoids depending on an accessor libSBML
    does not offer.
    """
    if node.getNumChildren() != 2:
        return None
    left, right = node.getChild(0), node.getChild(1)
    if _is_time(left) and not _is_time(right):
        other = right
    elif _is_time(right) and not _is_time(left):
        other = left
    else:
        return None
    return _compile(other)


def _collect(node, found):
    """Walk *node*, appending every compiled time threshold to *found*.

    Raises :class:`_Unevaluable` if a relation involving time cannot be reduced
    to an expression, which is what makes the whole table refuse to answer.
    Relations that do not involve time at all -- the state guards in the model's
    own rules -- are simply not collected; they are not failures.
    """
    if node is None:
        return

    if node.isRelational():
        if _is_time(node.getChild(0)) or _is_time(node.getChild(1)):
            expr = _threshold_from_relational(node)
            if expr is None:
                raise _Unevaluable("time compared against time")
            found.append(expr)
        return

    # Logical connectives, piecewise conditions and anything else: recurse.
    # Over-collecting is safe. An extra candidate only offers the splitter one
    # more place it may not cut; a missed one lets it cut on a discontinuity.
    for i in range(node.getNumChildren()):
        _collect(node.getChild(i), found)


# ---------------------------------------------------------------------------
# The table
# ---------------------------------------------------------------------------

class EventTimeTable:
    """Lazily resolved time discontinuities for one compiled model.

    Holds compiled expressions, not numbers: see the module docstring on
    ``SubCut_D1``. Call :meth:`times` as often as you like.
    """

    def __init__(self, r, entries, unresolved, n_events=0, n_rules=0):
        self._r = r
        self._entries = entries            # list of (source_label, node)
        self.unresolved = list(unresolved)  # list of (source_label, reason)
        self.n_events = n_events
        self.n_rules = n_rules

    def __len__(self):
        return len(self._entries)

    def dependencies(self):
        """Model symbols the thresholds are computed from."""
        out = set()
        for _label, node in self._entries:
            _symbols(node, out)
        return out

    def times(self):
        """Sorted discontinuity times in absolute hours, or None if unknown.

        None means "something here could not be resolved", and callers must
        treat it as no information at all rather than as an empty schedule. An
        empty *list*, by contrast, is a positive finding: this model has no time
        discontinuities, which is what lets the splitter leave a seventy-year
        pre-aging block whole.
        """
        if self.unresolved:
            return None

        # One read per distinct symbol, not one per trigger that mentions it.
        # The 50 Gantenerumab infusion-off edges all reference SubCut_D1, and a
        # model lookup is a SWIG round-trip; memoizing takes this call from
        # 231 us to 147 us on that arm, the remainder being the 100 expression
        # evaluations themselves. The memo lives for one call only, so a
        # parameter that moves between calls is still picked up.
        cache = {}

        def lookup(name):
            if name in cache:
                return cache[name]
            try:
                value = float(self._r[name])
            except Exception as exc:
                raise _Unevaluable(f"{name}: {exc}") from exc
            cache[name] = value
            return value

        out = []
        for _label, node in self._entries:
            try:
                t = _evaluate(node, lookup)
            except _Unevaluable:
                return None
            except (ArithmeticError, TypeError, ValueError):
                return None
            if not math.isfinite(t):
                return None
            out.append(float(t))

        out.sort()
        return _dedupe(out)


def _dedupe(times, rel_tol=1e-12):
    """Drop values that differ only by floating-point noise.

    Two generators can compute the same instant along different routes --
    ``{age}*365*24 + 48`` and ``{age}*365*24 + 48.00`` -- and at t of order 1e6
    those need not land on the same float. Genuinely distinct events here are
    never closer than the 23 h of ``SubCut_D1`` or the 1 h of the SILK ladder,
    so a relative tolerance of 1e-12 cannot merge two real ones.
    """
    out = []
    for t in times:
        if out and abs(t - out[-1]) <= rel_tol * max(1.0, abs(t), abs(out[-1])):
            continue
        out.append(t)
    return out


def build_event_time_table(r, verbose=False, label=None):
    """Read *r*'s events and time-based piecewise rules into an EventTimeTable.

    Returns None if the model's SBML cannot be read at all, which callers should
    treat exactly like an unresolved trigger: no information.
    """
    try:
        import libsbml
    except ImportError:
        if verbose:
            print("[events] libsbml is unavailable; block splitting will fall "
                  "back to the wall-clock cap.")
        return None

    try:
        doc = libsbml.readSBMLFromString(r.getSBML())
        model = doc.getModel()
    except Exception as exc:
        if verbose:
            print(f"[events] could not read the model's SBML ({exc}); block "
                  f"splitting will fall back to the wall-clock cap.")
        return None
    if model is None:
        return None

    entries, unresolved = [], []

    n_events = model.getNumEvents()
    for i in range(n_events):
        event = model.getEvent(i)
        name = event.getId() or f"event[{i}]"
        trigger = event.getTrigger()
        if trigger is None:
            continue
        found = []
        try:
            _collect(trigger.getMath(), found)
        except _Unevaluable as exc:
            unresolved.append((name, str(exc)))
            continue
        if not found:
            # A trigger with no time in it: a state-triggered event. It is a
            # genuine discontinuity that this module cannot place on the time
            # axis, so it has to count as unresolved -- pretending the model is
            # event-free would be worse than declining to answer.
            unresolved.append((name, "trigger does not depend on time"))
            continue
        entries.extend((name, node) for node in found)

    n_rules = 0
    for i in range(model.getNumRules()):
        rule = model.getRule(i)
        rule_math = rule.getMath()
        if rule_math is None or not _contains_piecewise(rule_math):
            continue
        name = rule.getVariable() or f"rule[{i}]"
        found = []
        try:
            _collect(rule_math, found)
        except _Unevaluable as exc:
            unresolved.append((name, str(exc)))
            continue
        if found:
            # Only rules that actually switch on time are counted; the model's
            # state guards (``AB40Total_BrainISF < 1e-12``) contribute nothing
            # and are not a failure.
            n_rules += 1
            entries.extend((name, node) for node in found)

    table = EventTimeTable(r, entries, unresolved, n_events, n_rules)

    if verbose:
        _describe(table, label)
    return table


def _contains_piecewise(node):
    if node is None:
        return False
    if node.isPiecewise():
        return True
    return any(_contains_piecewise(node.getChild(i))
               for i in range(node.getNumChildren()))


def _describe(table, label):
    who = f"'{label}'" if label else "model"
    if table.unresolved:
        print(f"[events] {who}: {len(table.unresolved)} trigger(s) could not be "
              f"resolved, so block splitting falls back to the wall-clock cap:")
        for name, reason in table.unresolved[:5]:
            print(f"           {name}: {reason}")
        if len(table.unresolved) > 5:
            print(f"           ... and {len(table.unresolved) - 5} more")
        return
    deps = sorted(table.dependencies())
    times = table.times()
    n = len(times) if times is not None else 0
    span = (f"{times[0]:.4g} to {times[-1]:.4g} h" if n else "none")
    print(f"[events] {who}: {n} time discontinuity(ies) from "
          f"{table.n_events} event(s) and {table.n_rules} time-based "
          f"piecewise rule(s); {span}.")
    if deps:
        print(f"           resolved against: {', '.join(deps)}")


def attach_event_times(replicate, r, verbose=False):
    """Give *replicate* the means to report its own discontinuity times.

    ``Solver_settings`` callables are handed only the replicate, and they live
    in ``Modules`` where nothing else imports from ``Engine``. Passing a bound
    method keeps that layering intact and keeps the resolution lazy, so a
    trigger built on a fitted parameter is re-read every evaluation instead of
    being frozen at x0.

    Returns the table, or None if one could not be built.
    """
    table = build_event_time_table(r, verbose=verbose,
                                   label=replicate.get("Label"))
    if table is None:
        replicate.pop("_event_times_fn", None)
        return None
    replicate["_event_times_fn"] = table.times
    return table


def without_event_times(replicates):
    """Shallow copies of *replicates* with the attached callable removed.

    For sending a replicate to a pool worker. cloudpickle *will* carry the
    callable -- it serializes the closed-over RoadRunner by value -- which makes
    this a correctness fix rather than a serialization one: a shipped table
    resolves its times against the parent's parameter values, and a worker
    running a profile point is holding a different vector by construction. With
    a fitted parameter in a trigger the worker would then integrate one
    schedule while splitting on another.

    Copies rather than mutating: the parent still needs its own attachment.
    """
    out = {}
    for name, rep in replicates.items():
        try:
            trimmed = {k: v for k, v in rep.items() if k != "_event_times_fn"}
        except AttributeError:
            out[name] = rep
            continue
        out[name] = trimmed
    return out

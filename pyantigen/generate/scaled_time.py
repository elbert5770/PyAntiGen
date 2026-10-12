"""Rewrite an event-free SBML model so that each schedule segment is integrated on s in [0, 1].

Why
---
A forward-sensitivity solver (CVODES, AMICI) differentiates the right-hand side
symbolically but cannot differentiate through the *time* at which an event
fires. With the events replaced by an :class:`InputSchedule`, the clock of one
segment is written ``t = t_a(p) + D(p) s`` with ``s`` in [0, 1] and the model is
integrated in ``s``::

    dx/ds = D(p) f(x, u, c(t), p)

The solver then produces the right sensitivity, the extra ``f dD/dp`` term
included, with no event handling at all, and the sensitivities simply carry over
from one segment to the next (see ``tests/engine/test_segment_sensitivity.py``).

What this module does to the model
----------------------------------
* ``sched_clock = sched_ta + sched_Delta*time``, with ``time`` now meaning ``s``;
  every other use of ``time`` in the model is replaced by ``sched_clock``.
* ``sched_Delta`` and ``sched_ta`` are built from the schedule's *moving
  quantities* (assignment rules in the model, symbolic in the parameters) and
  from per-segment constants ``sched_a0/a1`` and ``sched_c0_<m>/c1_<m>`` (the
  anchors and coefficients of the segment's two end points). The solver sees
  ``D`` as a function of the parameters.
* Every reaction rate and every rate rule is multiplied by ``sched_Delta``.
* A measured curve ``c`` (an assignment rule in the model) becomes
  ``sched_ci_c + sched_cs_c*sched_clock``: on one segment it is a straight line.
* Each parameter the schedule assigns (``Q_Leak``, ...) becomes a state with rate
  zero. The runner gives it its value at the start of each segment, and its
  sensitivities, from :meth:`InputSchedule.plan`.
* A variable that is already a state -- a rate-rule parameter or compartment, or an
  amount-valued (``substanceOnly``) species -- stays as it is. Assigning it is a
  jump (``ScaledModel.jump_states``): the runner evaluates the right-hand side on
  the live state and carries its sensitivities through, ``S+ = g_x S- + g_p``.
  Nothing about the switch time enters a jump in scaled time.

Per segment the runner sets :meth:`ScaledModel.segment_constants`, sets the input
states, integrates ``s`` from 0 to 1 and hands the final state and sensitivities
to the next segment.

Not covered: models with events (use the schedule for them), algebraic rules,
and a ``time`` that appears inside a function definition.
"""
import re
import ast
from dataclasses import dataclass, field

import libsbml

from pyantigen.generate.input_schedule import ScheduleError

PREFIX = "sched_"
CLOCK = PREFIX + "clock"
DELTA = PREFIX + "Delta"
T_A = PREFIX + "ta"


class ScaledTimeError(ValueError):
    """The model cannot be rewritten for scaled-time integration."""


@dataclass
class ScaledModel:
    sbml: str
    moving: list = field(default_factory=list)          # names of the moving quantities
    curves: list = field(default_factory=list)          # names of the measured inputs
    inputs: list = field(default_factory=list)          # assigned variables, now zero-rate states
    initial_inputs: dict = field(default_factory=dict)  # their values before the first step (number or expression)
    jump_states: list = field(default_factory=list)     # model states a step assigns (boluses): see PlanSegment.jumps
    constants: list = field(default_factory=list)       # the per-segment constants, in order
    parameter_values: dict = field(default_factory=dict)  # every plain parameter's value in the source model

    def segment_constants(self, seg):
        """``{constant: value}`` for one :class:`PlanSegment` of the schedule that built this model."""
        out = {PREFIX + "a0": float(seg.start.anchor), PREFIX + "a1": float(seg.end.anchor)}
        for m in self.moving:
            out[f"{PREFIX}c0_{m}"] = 0.0
            out[f"{PREFIX}c1_{m}"] = 0.0
        for which, bp in (("c0", seg.start), ("c1", seg.end)):
            for name, coeff in bp.terms:
                out[f"{PREFIX}{which}_{name}"] = float(coeff)
        for c in self.curves:
            value, slope, mid = seg.curves[c]
            out[f"{PREFIX}ci_{c}"] = value - slope * mid
            out[f"{PREFIX}cs_{c}"] = slope
        return out


# --- libsbml helpers -----------------------------------------------------------------

def _name_node(name):
    n = libsbml.ASTNode(libsbml.AST_NAME)
    n.setName(name)
    return n


def _times(a, b):
    n = libsbml.ASTNode(libsbml.AST_TIMES)
    n.addChild(a)
    n.addChild(b)
    return n


def _parse(text):
    node = libsbml.parseL3Formula(text)
    if node is None:
        raise ScaledTimeError(f"cannot parse {text!r}: {libsbml.getLastParseL3Error()}")
    return node


_TIME_TOKEN = re.compile(r"(?<![A-Za-z0-9_.])time(?![A-Za-z0-9_])")


def _reads_time(node):
    # isName()/getName(), not getType() == AST_NAME_TIME: see Event_times._is_time
    if node.isName() and node.getName() == "time":
        return True
    return any(_reads_time(node.getChild(i)) for i in range(node.getNumChildren()))


def _with_clock(node):
    """*node*'s mathematics with ``time`` rewritten as the segment clock, as a new tree; None if it reads no time.

    Done on the formula text, not by editing the node. Once AMICI has been imported in a process, python-libsbml
    hands out nodes wrapped by libsedml (AMICI's own copy of the SWIG types) and ``node.setType(libsbml.AST_NAME)``
    raises TypeError, while comparing ``getType()`` with a constant is simply False for every node, which would
    leave the rule reading ``time``. The text route needs neither, and drops the ``definitionURL`` of the time
    symbol that AMICI's importer refuses on a plain <ci>. Numbers are printed to 15 significant digits.
    """
    if not _reads_time(node):
        return None
    return _parse(_TIME_TOKEN.sub(CLOCK, libsbml.formulaToL3String(node)))


def _expand_initial(model, text, depth=0):
    """*text* with every parameter that has an initial assignment replaced by that assignment.

    ``Q_SN`` starts as ``Q_SN_0``, which starts as ``Q_CSF*f_SN``: the input's
    sensitivity to Q_CSF before the first switch lives two levels down. Done on the
    formula text with Python's ``ast`` -- the schedule parses these expressions
    that way anyway -- so no libsbml tree is edited in place (its Python bindings
    make that easy to get wrong).
    """
    if depth > 32:
        raise ScaledTimeError("initial assignments refer to one another in a cycle")

    class Substitute(ast.NodeTransformer):
        def visit_Name(self, node):
            ia = model.getInitialAssignmentBySymbol(node.id)
            if (model.getParameter(node.id) is not None and ia is not None
                    and model.getRuleByVariable(node.id) is None):
                inner = _expand_initial(model, libsbml.formulaToL3String(ia.getMath()), depth + 1)
                return ast.parse("(" + inner + ")", mode="eval").body
            return node

    tree = ast.parse(text, mode="eval")
    return ast.unparse(Substitute().visit(tree).body)


def _add_parameter(model, pid, value=0.0, constant=True):
    if model.getElementBySId(pid) is not None:
        raise ScaledTimeError(f"the model already has an element named {pid!r}")
    p = model.createParameter()
    p.setId(pid)
    p.setValue(float(value))
    p.setConstant(constant)
    return p


def _assignment_rule(model, variable, formula):
    r = model.createAssignmentRule()
    r.setVariable(variable)
    r.setMath(_parse(formula))


# --- the rewrite ---------------------------------------------------------------------

def scale_model(sbml, schedule):
    """Return a :class:`ScaledModel` for *sbml* (an event-free model) and *schedule*."""
    doc = libsbml.readSBMLFromString(sbml)
    model = doc.getModel()
    if model is None or doc.getNumErrors(libsbml.LIBSBML_SEV_FATAL):
        raise ScaledTimeError("cannot read the SBML model")
    if model.getNumEvents():
        raise ScaledTimeError(
            f"the model still has {model.getNumEvents()} events: build it with "
            "schedule.to_antimony(events=False) so the schedule alone drives it")
    if any(model.getRule(i).isAlgebraic() for i in range(model.getNumRules())):
        raise ScaledTimeError("algebraic rules are not supported")

    parameter_values = {p.getId(): p.getValue() for p in model.getListOfParameters()
                        if p.getConstant() and model.getRuleByVariable(p.getId()) is None}
    moving = schedule.moving_quantities()
    curves = {c[0] for c in schedule.inputs()}
    assigned = schedule.assigned_variables()

    # 1. the clock inside the model's own mathematics
    for i in range(model.getNumRules()):
        rule = model.getRule(i)
        new = _with_clock(rule.getMath())
        if new is not None:
            rule.setMath(new)
    for i in range(model.getNumReactions()):
        kl = model.getReaction(i).getKineticLaw()
        new = _with_clock(kl.getMath()) if kl is not None and kl.isSetMath() else None
        if new is not None:
            kl.setMath(new)
    for i in range(model.getNumInitialAssignments()):
        assignment = model.getInitialAssignment(i)
        new = _with_clock(assignment.getMath())
        if new is not None:
            assignment.setMath(new)

    # 2. measured curves become straight lines of the clock, constants per segment
    for c in sorted(curves):
        rule = model.getRuleByVariable(c)
        if rule is None or not rule.isAssignment():
            raise ScaledTimeError(f"the model has no assignment rule for the measured input {c!r}")
        _add_parameter(model, f"{PREFIX}ci_{c}")
        _add_parameter(model, f"{PREFIX}cs_{c}")
        rule.setMath(_parse(f"{PREFIX}ci_{c} + {PREFIX}cs_{c}*{CLOCK}"))

    # 3. rates times Delta (before the new rules, which must not be scaled)
    delta = lambda: _name_node(DELTA)  # noqa: E731
    for i in range(model.getNumReactions()):
        kl = model.getReaction(i).getKineticLaw()
        if kl is None or not kl.isSetMath():
            raise ScaledTimeError(f"reaction {model.getReaction(i).getId()!r} has no rate law")
        kl.setMath(_times(delta(), kl.getMath().deepCopy()))
    for i in range(model.getNumRules()):
        rule = model.getRule(i)
        if rule.isRate():
            rule.setMath(_times(delta(), rule.getMath().deepCopy()))

    # 4. the assigned variables. A parameter nothing drives becomes a state that does not move
    #    within a segment (an input). A variable that is already a state -- a rate-rule
    #    parameter or compartment, or an amount-valued species -- stays one: assigning it is
    #    a jump the runner applies with its sensitivities (a bolus).
    initial_inputs, jump_states, inputs = {}, [], []
    for v in assigned:
        rule = model.getRuleByVariable(v)
        sp = model.getSpecies(v)
        if rule is not None and rule.isRate() and (model.getParameter(v) or model.getCompartment(v)):
            jump_states.append(v)
            continue
        if sp is not None and rule is None:
            if not sp.getHasOnlySubstanceUnits():
                raise ScaledTimeError(
                    f"{v!r} is a species in concentration units: an event assigns its concentration, "
                    "which the amount-valued solver state cannot take directly (declare it with "
                    "substanceOnly, or assign through a parameter)")
            jump_states.append(v)
            continue
        p = model.getParameter(v)
        if p is None:
            raise ScaledTimeError(f"{v!r} is assigned by the schedule but is not a parameter, species or "
                                  "rate-rule variable of the model")
        if rule is not None:
            raise ScaledTimeError(f"{v!r} is assigned by the schedule and also driven by a rule")
        inputs.append(v)
        ia = model.getInitialAssignmentBySymbol(v)
        if ia is not None:
            initial_inputs[v] = _expand_initial(model, libsbml.formulaToL3String(ia.getMath()))
            model.removeInitialAssignment(v)
            p.setValue(0.0)                 # the runner sets the value at the start of each segment
        else:
            initial_inputs[v] = p.getValue()
        p.setConstant(False)
        rule = model.createRateRule()
        rule.setVariable(v)
        rule.setMath(_parse("0"))

    # 5. the moving quantities, Delta, t_a and the clock
    for name, text in moving.items():
        _add_parameter(model, name, constant=False)
        _assignment_rule(model, name, text)
    names = list(moving)
    const = [PREFIX + "a0", PREFIX + "a1"]
    _add_parameter(model, PREFIX + "a0")
    _add_parameter(model, PREFIX + "a1")
    d = f"({PREFIX}a1 - {PREFIX}a0)"
    t = f"{PREFIX}a0"
    for m in names:
        for which in ("c0", "c1"):
            _add_parameter(model, f"{PREFIX}{which}_{m}")
            const.append(f"{PREFIX}{which}_{m}")
        d += f" + ({PREFIX}c1_{m} - {PREFIX}c0_{m})*{m}"
        t += f" + {PREFIX}c0_{m}*{m}"
    for c in sorted(curves):
        const += [f"{PREFIX}ci_{c}", f"{PREFIX}cs_{c}"]
    for pid, formula in ((DELTA, d), (T_A, t)):
        _add_parameter(model, pid, constant=False)
        _assignment_rule(model, pid, formula)
    _add_parameter(model, CLOCK, constant=False)
    _assignment_rule(model, CLOCK, f"{T_A} + {DELTA}*time")

    return ScaledModel(libsbml.writeSBMLToString(doc), names, sorted(curves), inputs,
                       initial_inputs, jump_states, const, parameter_values)

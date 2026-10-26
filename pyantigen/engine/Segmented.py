"""Integrate a model whose clock-driven inputs are an InputSchedule, with no events.

The usual route writes every switch as an SBML event and lets CVODE root-find
it. That works until two switches fall within a few ulps of each other, and it
makes the switch times invisible to anything that wants to differentiate
through them. Here the model is compiled *without* those events. The runner
reads the switch times from the schedule (``InputSchedule.times``, the same
doubles the Antimony triggers evaluate), stops the integrator exactly at each
one, applies the step's assignments, and carries on from the same state.

What a segment is
-----------------
Between two consecutive switches the right-hand side is smooth, so each stretch
is one ``r.simulate`` call. CVODE restarts at a switch, which is what an event
costs anyway, and never has to find a root. Output on the block's uniform grid
is read from inside the stretch (``times=``), so a dense grid costs no extra
restarts. A grid point that coincides with a switch (to 1e-9 h) is reported
with the value after the switch, the same as an event that fires at its
trigger time.

Assignments
-----------
All the assignments of one switch are evaluated first and then applied, so each
sees the values from before the switch -- SBML's ``useValuesFromTriggerTime``,
which is what the events this replaces carry. An assignment may use numbers,
parameter names and ``+ - * / **``; anything else is refused by
``InputSchedule.evaluate`` rather than guessed at.

A step may assign a state as well as an input (a bolus ``A = A + dose``): that is
an ordinary assignment here, and the state is left where the run ends. A species
must be amount-valued (``substanceOnly``): assigning a concentration is refused.

The input variables the schedule assigns are put back to the values they had when
the run began, so a second simulation does not start from the last switch of the
first. (The optimizer also restores parameter baselines; this keeps the runner
correct on its own.)

Not covered
-----------
A switch strictly before the first block's start is applied at that start, in
order. A switch exactly at a block's end is applied when the next block starts,
or not at all if there is none. A pre-equilibration block that the optimizer
caches separately must not contain a switch: ``switches_before`` says whether it does.
"""
import numpy as np

from pyantigen.engine import Model_cache as _model_cache

from pyantigen.engine.Simulate import StackedResult, apply_tolerance_vector, safe_simulate
from pyantigen.generate.input_schedule import ScheduleError


def _eps(t):
    """How close two times must be to count as one -- the schedule's own gap."""
    return max(1e-9, 1e-12 * abs(t))


def _merged(times):
    """Sorted times with each cluster of near-coincident ones reduced to its first.

    A moving step can land on a fixed measured-curve knot (V_LP/Q_CSF = 0.5 puts
    the refill at 2.5 h). They are one place the integrator stops, not two.
    """
    out = []
    for t in times:
        if out and t - out[-1] <= _eps(t):
            continue
        out.append(t)
    return out


class SegmentRunner:
    """Runs the blocks of one ``simulate()`` call from an InputSchedule.

    Construct it after the model's parameters are set: the switch times and the
    right-hand sides are taken from the model's values at that moment, and the
    schedule is validated against them (``ScheduleError`` if the order of the
    switches has changed or two are too close to tell apart).
    """

    def __init__(self, r, schedule):
        self.r = r
        self.schedule = schedule
        self.params = {p: float(r[p]) for p in schedule.params}
        schedule.validate(self.params)
        self.steps = schedule.ordered_events(self.params)
        self.cuts = _merged(schedule.times(self.params))
        self._next = 0
        self._before = {}
        self._variable_step_size = r.integrator.variable_step_size
        # a state a step assigns (a bolus) is not put back afterwards: only inputs are
        self._states = set(r.getFloatingSpeciesIds()) | set(r.getRateRuleIds())
        self._check_species_units(r, schedule)

    # -- applying the steps -------------------------------------------------

    def _apply_due(self, t):
        """Apply every step at or before *t* that has not been applied yet."""
        r, evaluate = self.r, self.schedule.evaluate
        while self._next < len(self.steps) and self.steps[self._next][0] <= t + _eps(t):
            _when, assignments = self.steps[self._next]
            values = {name: evaluate(text, lambda n: float(r[n]))
                      for name, text in assignments.items()}
            for name, value in values.items():
                self._before.setdefault(name, float(r[name]))
                r[name] = value
            self._next += 1

    @staticmethod
    def _check_species_units(r, schedule):
        """Refuse to assign a species whose events would set a concentration, not an amount."""
        species = set(r.getFloatingSpeciesIds()) & set(schedule.assigned_variables())
        # The units are a property of the compiled model, so each species is checked once per model. Re-reading the
        # SBML on every evaluation cost a serialisation and a parse of the whole model each time, and the memory of
        # it was never returned (about 0.07 GB per evaluation on the PK model).
        checked = _model_cache.get(r, "_species_units_checked") or frozenset()
        species -= checked
        if not species:
            return
        import libsbml
        model = libsbml.readSBMLFromString(r.getCurrentSBML()).getModel()
        for name in sorted(species):
            sp = model.getSpecies(name)
            if sp is not None and not sp.getHasOnlySubstanceUnits():
                raise ScheduleError(
                    f"{name!r} is a species in concentration units: assigning it sets a concentration, "
                    "but the runner sets amounts (declare it substanceOnly)")
        _model_cache.put(r, "_species_units_checked", checked | frozenset(species))

    def close(self):
        """Undo what the run changed: the assigned inputs and the output mode."""
        for name, value in self._before.items():
            if name not in self._states:
                self.r[name] = value
        self._before.clear()
        self._next = 0
        self.r.integrator.variable_step_size = self._variable_step_size

    # -- one block ----------------------------------------------------------

    def run_block(self, block, observed_species, label=None):
        """Integrate one block of ``simulation_blocks``; same result as ``safe_simulate``'s."""
        start, end = float(block["start"]), float(block["end"])
        n_points = block.get("n_points", block.get("eval_points"))
        if not end > start:
            raise ValueError(f"block end {end} is not after its start {start}")
        grid = np.linspace(start, end, int(n_points))

        inside = [t for t in self.cuts if start + _eps(start) < t < end - _eps(end)]
        bounds = [start] + inside + [end]

        # Output at chosen times (`times=`) needs fixed-step output mode.
        self.r.integrator.variable_step_size = False
        rows, colnames = [], None
        for k, (u, v) in enumerate(zip(bounds, bounds[1:])):
            last = k == len(bounds) - 2
            self._apply_due(u)

            eu, ev = _eps(u), _eps(v)
            mine = grid[(grid >= u - eu) & ((grid <= v + ev) if last else (grid < v - ev))]
            has_start = bool(len(mine)) and mine[0] <= u + eu
            between = mine[(mine > u + eu) & (mine < v - ev)]
            times = [u] + between.tolist() + [v]

            apply_tolerance_vector(self.r)
            res, _meta = safe_simulate(
                self.r, {"times": times, "variable_step_size": False},
                observed_species, label=label)
            if colnames is None:
                colnames = list(getattr(res, "colnames", observed_species))

            keep = ([0] if has_start else []) + list(range(1, 1 + len(between)))
            if last:
                keep.append(len(times) - 1)
            if keep:
                rows.append(np.asarray(res)[keep])

        return StackedResult(np.vstack(rows), colnames)


def switches_before(r, schedule, until):
    """True if a switch falls before *until*, so a block ending there is not cacheable.

    The pre-equilibration cache integrates the leading untracked block on its
    own and replays its end state. That block would run without the schedule's
    steps, so a schedule that switches inside it must not be cached.
    """
    params = {p: float(r[p]) for p in schedule.params}
    return any(t < until - _eps(until) for t in schedule.times(params))

"""Forward sensitivities through an InputSchedule with AMICI, one segment at a time.

The model is rewritten by :func:`pyantigen.generate.scaled_time.scale_model` so
that every segment is integrated on ``s`` in [0, 1] and AMICI differentiates
the segment's duration and start time as functions of the parameters. This
module compiles that model, then for each segment

* sets the segment's constants (``ScaledModel.segment_constants``),
* gives the input states the values and derivatives ``InputSchedule.plan``
  worked out, keeps every other state and sensitivity from the previous segment,
* integrates ``s`` from 0 to 1 with forward sensitivities.

AMICI never sees an event. Observation times are segment ends, so they are fixed
points of ``s``.

AMICI is optional and, on Windows, needs WSL (no wheel, no compiler); this
module imports it only when a runner is built.
"""
import json
import os

import numpy as np

_OUTPUT_PREFIX = "out_"


def _amici():
    try:
        import amici
        import amici.sim.sundials as sundials
        from amici.importers.sbml import SbmlImporter
    except ImportError as exc:                 # pragma: no cover - environment dependent
        raise ImportError(
            "AMICI >= 1.1 is required for the scaled-time backend "
            "(pip install amici; on Windows run it under WSL)") from exc
    return amici, sundials, SbmlImporter


def _plain_parameters(sbml):
    """Ids of the parameters that are constants no rule or initial assignment defines: the ones a fit can move."""
    import libsbml
    model = libsbml.readSBMLFromString(sbml).getModel()
    return [p.getId() for p in model.getListOfParameters()
            if p.getConstant() and model.getRuleByVariable(p.getId()) is None
            and model.getInitialAssignmentBySymbol(p.getId()) is None]


def concentration_outputs(sbml, columns):
    """``{column: formula}`` for the result columns of a RoadRunner run, to compile as AMICI outputs.

    ``[S]`` is the concentration of species S: ``S / compartment`` when the species is
    in amount units (``hasOnlySubstanceUnits``, which is what AMICI's state holds),
    ``S`` otherwise. Any other column (an assigned variable, a parameter, a state)
    is its own formula. ``time`` is not an output; it is the observation time.
    """
    import libsbml
    model = libsbml.readSBMLFromString(sbml).getModel()
    out = {}
    for col in columns:
        if col == "time":
            continue
        if col.startswith("[") and col.endswith("]"):
            sp = model.getSpecies(col[1:-1])
            if sp is None:
                raise KeyError(f"{col}: not a species of the model")
            out[col] = (f"{sp.getId()} / {sp.getCompartment()}" if sp.getHasOnlySubstanceUnits()
                        else sp.getId())
        else:
            out[col] = col
    return out


def compile_model(scaled, name, output_dir, free=None, outputs=None, verbose=False):
    """Compile *scaled* (a ScaledModel) with AMICI and return the model module.

    Conservation laws are switched off so the state vector is the model's own;
    the schedule's constants are AMICI fixed parameters. *free* lists the
    parameters to differentiate with respect to: every other plain parameter is
    compiled as a fixed one, which keeps the generated sensitivity code (and the
    compile time) small for a large model. Default: all of them stay free.

    *outputs* is ``{name: formula}``: quantities AMICI evaluates, and differentiates
    itself, at the end of every segment (see :func:`concentration_outputs`). A
    runner returns them from :meth:`AmiciSegmentRunner.run_outputs`; their names
    are written beside the model (``outputs.json``) so the runner knows the order.
    """
    amici, _sundials, SbmlImporter = _amici()
    fixed = list(scaled.constants)
    if free is not None:
        free = set(free)
        unknown = free - set(_plain_parameters(scaled.sbml))
        if unknown:
            raise KeyError(f"not parameters of the model: {sorted(unknown)}")
        fixed += [p for p in _plain_parameters(scaled.sbml) if p not in free and p not in fixed]
    outputs = dict(outputs or {})
    channels = [amici.MeasurementChannel(id_="o_clock", formula="sched_clock")]
    channels += [amici.MeasurementChannel(id_=f"{_OUTPUT_PREFIX}{i}", formula=f)
                 for i, f in enumerate(outputs.values())]
    importer = SbmlImporter(scaled.sbml, from_file=False)
    importer.sbml2amici(
        name, str(output_dir), fixed_parameters=fixed,
        compute_conservation_laws=False, verbose=verbose, observation_model=channels)
    with open(os.path.join(str(output_dir), "outputs.json"), "w") as fh:
        json.dump(list(outputs), fh)
    return amici.import_model_module(name, str(output_dir))


class AmiciSegmentRunner:
    """Run a compiled scaled model over an InputSchedule and return states and sensitivities.

    Parameters
    ----------
    scaled, schedule : the ScaledModel and the InputSchedule it was built from.
    module : the compiled model module (:func:`compile_model`).
    states : names of the model's states to report, in the order wanted. A name
        AMICI had to rename (``y`` -> ``amici_y``) is found either way.
    """

    def __init__(self, scaled, schedule, module, states, rtol=1e-12, atol=1e-14, max_steps=100000):
        amici, sundials, _ = _amici()
        self._sundials = sundials
        self.scaled, self.schedule, self.module = scaled, schedule, module
        self.model = module.get_model()
        self.solver = self.model.create_solver()
        self.solver.set_sensitivity_order(sundials.SensitivityOrder.first)
        self.solver.set_sensitivity_method(sundials.SensitivityMethod.forward)
        self.solver.set_relative_tolerance(rtol)
        self.solver.set_absolute_tolerance(atol)
        self.solver.set_relative_tolerance_fsa(rtol)
        self.solver.set_absolute_tolerance_fsa(atol)
        self.solver.set_max_steps(max_steps)
        self.state_ids = list(self.model.get_state_ids())
        self.free_ids = list(self.model.get_free_parameter_ids())
        self.fixed_ids = list(self.model.get_fixed_parameter_ids())
        self._fixed_defaults = list(self.model.get_fixed_parameters())
        self.states = list(states)
        self._index = {n: self._state_index(n) for n in set(self.states) | set(scaled.inputs)
                       | set(scaled.jump_states)}
        # every state by the name the schedule's expressions use (AMICI renames y -> amici_y)
        self._named = {(n[len("amici_"):] if n.startswith("amici_") and n[len("amici_"):] not in self.state_ids
                        else n): i for i, n in enumerate(self.state_ids)}
        # the outputs the model was compiled with (compile_model(outputs=...)), in order
        self.output_names = self._read_output_names(module)
        observables = list(self.model.get_observable_ids())
        self._output_index = [observables.index(f"{_OUTPUT_PREFIX}{i}") for i in range(len(self.output_names))]

    @staticmethod
    def _read_output_names(module):
        directory = os.path.dirname(os.path.dirname(os.path.abspath(module.__file__)))
        try:
            with open(os.path.join(directory, "outputs.json")) as fh:
                return list(json.load(fh))
        except OSError:
            return []

    def _state_index(self, name):
        for candidate in (name, "amici_" + name):
            if candidate in self.state_ids:
                return self.state_ids.index(candidate)
        raise KeyError(f"{name!r} is not a state of the compiled model ({self.state_ids})")

    def run(self, theta, t_start, t_end, observation_times, wrt=None, initial=None, initial_inputs=None):
        """States and sensitivities at the observation times.

        theta : every parameter value the schedule or the model needs (name -> value).
        wrt : parameters to differentiate with respect to (default: the schedule's).
        initial : state values at *t_start* (name -> value); default: the model's own.
        initial_inputs : what the assigned variables hold before their first
            assignment (default: the model's own, expressions included).

        Returns ``(X, S)``: X[k, i] the i-th requested state at the k-th
        observation time, S[k, i, j] its derivative with respect to ``wrt[j]``.
        """
        x, s, _y, _sy = self._run(theta, t_start, t_end, observation_times, wrt, initial, initial_inputs,
                                  outputs=False)
        return x, s

    def run_outputs(self, theta, t_start, t_end, observation_times, wrt=None, initial=None,
                    initial_inputs=None):
        """The compiled outputs (``compile_model(outputs=...)``) and their derivatives at the observation times.

        Same arguments as :meth:`run`. Returns ``(Y, SY)``: Y[k, i] the i-th output
        (``self.output_names[i]``) at the k-th observation time, SY[k, i, j] its
        derivative with respect to ``wrt[j]``, which AMICI takes through the output's own formula.
        """
        if not self.output_names:
            raise ValueError("the model was compiled without outputs (compile_model(outputs=...))")
        _x, _s, y, sy = self._run(theta, t_start, t_end, observation_times, wrt, initial, initial_inputs,
                                  outputs=True)
        return y, sy

    def _run(self, theta, t_start, t_end, observation_times, wrt, initial, initial_inputs, outputs):
        wrt = sorted(self.schedule.params) if wrt is None else list(wrt)
        model, solver = self.model, self.solver
        for name in wrt:
            if name not in self.free_ids:
                raise KeyError(f"cannot differentiate with respect to {name!r}: not a free parameter "
                               f"of the compiled model ({self.free_ids})")
        # every name an initial expression may mention, with the model's own value unless given
        # a parameter the SBML gives no value (it has an initial assignment) is NaN in parameter_values:
        # leave it to AMICI, which evaluates the assignment, rather than overwrite it with NaN
        theta = {**{k: v for k, v in self.scaled.parameter_values.items() if np.isfinite(v)},
                 **{k: v for k, v in theta.items()}}
        values = list(model.get_free_parameters())
        for name, v in theta.items():
            if name in self.free_ids:
                values[self.free_ids.index(name)] = float(v)
        model.set_free_parameters(values)
        nx, npar = len(self.state_ids), len(wrt)
        # with nothing to differentiate the run is a plain forward integration
        solver.set_sensitivity_order(
            self._sundials.SensitivityOrder.first if npar else self._sundials.SensitivityOrder.none)
        if npar:
            model.set_parameter_list([self.free_ids.index(n) for n in wrt])

        # A custom initial state set by an earlier run would be returned here; an empty
        # vector hands the initial state back to the model (its initial assignments).
        model.set_initial_state([])
        model.set_initial_state_sensitivities([])
        x = np.array(model.get_initial_state(), dtype=float)
        sx = (np.array(model.get_initial_state_sensitivities(), dtype=float).reshape(npar, nx)
              if npar else np.zeros((0, nx)))
        for name, v in (initial or {}).items():
            x[self._state_index(name)] = float(v)
            sx[:, self._state_index(name)] = 0.0

        model.set_t0(0.0)
        model.set_timepoints([1.0])
        out_x, out_s, out_y, out_sy = [], [], [], []
        if initial_inputs is None:
            initial_inputs = self.scaled.initial_inputs
        plan = self.schedule.plan(t_start, t_end, observation_times, params=theta, wrt=wrt,
                                  initial=initial_inputs, states=self.scaled.jump_states)
        for seg in plan:
            constants = self.scaled.segment_constants(seg)
            fixed = dict(zip(self.fixed_ids, self._fixed_defaults))
            fixed.update({k: float(v) for k, v in theta.items() if k in fixed})
            fixed.update(constants)
            model.set_fixed_parameters([fixed[i] for i in self.fixed_ids])
            if seg.jumps:                  # boluses: every right-hand side sees the states from before
                live = {n: (x[i], tuple(sx[:, i])) for n, i in self._named.items()}
                new = {v: seg.evaluate_jump(text, live) for v, text in seg.jumps.items()}
                for v, (value, grad) in new.items():
                    i = self._index[v]
                    x[i] = value
                    sx[:, i] = grad
            for v in self.scaled.inputs:
                i = self._index[v]
                x[i] = seg.values[v]
                sx[:, i] = seg.grads[v]
            model.set_initial_state(list(x))
            if npar:
                model.set_initial_state_sensitivities(list(sx.ravel()))
            rdata = self._sundials.run_simulation(model, solver)
            if int(rdata.status) != self._sundials.AMICI_SUCCESS:
                raise RuntimeError(f"AMICI failed on the segment [{seg.t0:.6g}, {seg.t1:.6g}] h: "
                                   f"status {int(rdata.status)}; {list(rdata.messages)[:2]}")
            x = np.array(rdata.x)[-1].copy()
            if npar:
                sx = np.array(rdata.sx)[-1].reshape(npar, nx).copy()
            if seg.observed:
                cols = [self._index[n] for n in self.states]
                out_x.append(x[cols])
                out_s.append(sx[:, cols].T)
                if outputs:
                    out_y.append(np.array(rdata.y)[-1][self._output_index])
                    out_sy.append(np.array(rdata.sy)[-1].reshape(npar, -1)[:, self._output_index].T
                                  if npar else np.zeros((len(self._output_index), 0)))
        return np.array(out_x), np.array(out_s), np.array(out_y), np.array(out_sy)

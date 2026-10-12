"""The gradient of the fit objective from AMICI's forward sensitivities.

What the optimizer minimizes is the concentrated Gaussian NLL of
``pyantigen.engine.Optimize`` -- groups, pooled sigma blocks, sigma floors, data
weights, observable expressions such as ``np.log10(np.maximum(predicted_B, 1e-12))``.
Writing its derivative out by hand would duplicate all of that, and the copy would
drift. This module does not. It splits the derivative the way the chain rule does:

    dNLL/dtheta_k = sum over simulated values y_ic of  dNLL/dy_ic * dy_ic/dtheta_k

* ``dy/dtheta`` is a table the same shape as the result table: AMICI's sensitivities
  of every observed column, mapped from the model parameters to the fitted ones
  (:class:`ParameterReplay`).
* The contraction with ``dNLL/dy`` is the *directional derivative of the engine's own
  loss* along that table: ``(L(R + h D_k) - L(R - h D_k)) / 2h``, with the loss
  evaluated by ``collect_loss_blocks`` and ``concentrated_nll`` themselves
  (:class:`LossGradient`). It re-evaluates arithmetic on tables already in memory --
  no integration, no noise, so ``h`` can be small and a Richardson step makes the
  result accurate to rounding. Anything the loss does keeps working as it is.

The simulations come from AMICI on the event-free model of
``pyantigen.generate.scaled_time`` (the schedule route), at the observed data times
and the ends of the tracked window; between them the engine interpolates linearly,
and the gradient is that of the loss at the data times.

Where the engine's objective is not a function of the parameters alone -- a
parameter with an initial assignment that RoadRunner's ``reset()`` does not
re-derive (``Q_SN = Q_CSF*f_SN``), a species set by a hook -- the gradient follows
the engine, not the model's own formulas: the value RoadRunner holds after the
engine's parameter updates is what AMICI is given.

AMICI is optional; on Windows it needs WSL. Nothing here imports it until a
gradient is built.
"""
import hashlib
import json
import os
import re
import tempfile

import numpy as np

from pyantigen.engine.Multistart import GradientSetupError
from pyantigen.generate.input_schedule import InputSchedule, ScheduleError


class AnalyticGradientError(RuntimeError):
    """The analytic gradient cannot be built or evaluated for this problem."""


# ---------------------------------------------------------------------------------------
# A result table the loss can read
# ---------------------------------------------------------------------------------------

class SimTable:
    """What the loss reads of a RoadRunner result: ``table["col"]`` and ``table.colnames``."""

    def __init__(self, colnames, data):
        self.colnames = list(colnames)
        self.data = np.asarray(data, dtype=float)
        if self.data.ndim != 2 or self.data.shape[1] != len(self.colnames):
            raise ValueError("a SimTable is (rows, len(colnames))")
        self._index = {c: i for i, c in enumerate(self.colnames)}

    def __getitem__(self, key):
        return self.data[:, self._index[key]]

    def __len__(self):
        return self.data.shape[0]

    def shifted(self, direction, h):
        return SimTable(self.colnames, self.data + h * direction)


# ---------------------------------------------------------------------------------------
# dNLL/dtheta from the loss itself
# ---------------------------------------------------------------------------------------

_NAME = re.compile(r"[A-Za-z_][A-Za-z_0-9]*")


class LossGradient:
    """The gradient of the concentrated NLL along tables of ``d(result)/d(theta)``.

    ``groups``, ``replicates`` and ``param_names`` are the engine's: the same objects
    ``Optimize.collect_loss_blocks`` is called with.
    """

    def __init__(self, groups, replicates, param_names, rel_step=1e-5):
        self.groups, self.replicates = groups, replicates
        self.param_names = list(param_names)
        self.rel_step = float(rel_step)
        self.explicit = self._explicit_parameters()

    # -- which parameters the loss reads by name ------------------------------------
    def _explicit_parameters(self):
        """Indices of the fitted parameters the loss names itself (``SF*predicted_B``).

        Those enter the loss a second way, besides through the simulation, and need
        their own partial derivative. A callable observable could read anything: then
        every parameter counts.
        """
        found, names = set(), set(self.param_names)
        for g in self.groups.values():
            for elem in g.get("loss_elements", []):
                sims = list(elem.get("simulations", [])) or [elem.get("simulation")]
                lc_fn = elem.get("loss_config")
                for sim in sims:
                    if sim not in self.replicates:
                        continue
                    lc = lc_fn(self.replicates[sim]) if callable(lc_fn) else lc_fn
                    for obs in (lc or {}).get("observables", []):
                        ov = obs.get("observed_variable")
                        if callable(ov):
                            return set(range(len(self.param_names)))
                        texts = [str(ov), str(obs.get("noise_formula") or "")]
                        for t in texts:
                            found |= set(_NAME.findall(t)) & names
        return {self.param_names.index(n) for n in found}

    # -- the loss -------------------------------------------------------------------
    def value(self, items, p_lin):
        from pyantigen.engine.Optimize import collect_loss_blocks, concentrated_nll
        p_dict = dict(zip(self.param_names, np.asarray(p_lin, dtype=float).tolist()))
        blocks = collect_loss_blocks(items, self.groups, self.replicates, self.param_names,
                                     np.asarray(p_lin, dtype=float), p_dict)
        if not blocks:
            raise AnalyticGradientError("the loss has no data to score")
        v = concentrated_nll(blocks)
        if not np.isfinite(v):
            raise AnalyticGradientError("the loss is not finite")
        return v

    @staticmethod
    def _shift(items, directions, k, h):
        out = {}
        for sim, item in items.items():
            D = directions.get(sim)
            table = item["results"] if D is None else item["results"].shifted(D[:, :, k], h)
            out[sim] = {**item, "results": table}
        return out

    def _step(self, items, directions, k):
        """The step along direction k that moves the smallest-scale entry by ``rel_step`` of itself."""
        ratio = 0.0
        for sim, item in items.items():
            D = directions.get(sim)
            if D is None:
                continue
            R = item["results"].data
            colmax = np.nanmax(np.abs(R), axis=0) if R.size else np.zeros(R.shape[1])
            scale = np.maximum(np.abs(R), 1e-6 * colmax[None, :] + 1e-300)
            ratio = max(ratio, float(np.nanmax(np.abs(D[:, :, k]) / scale)) if D.size else 0.0)
        return self.rel_step / ratio if ratio > 0 else None

    def gradient(self, items, directions, p_lin):
        """``(nll, grad)`` at *p_lin*.

        items : ``{sim: {"results": SimTable, "data": df_dict, "replicate": ...}}``.
        directions : ``{sim: array (rows, columns, K)}``, d(table)/d(theta_k) in linear units.
        """
        p_lin = np.asarray(p_lin, dtype=float)
        K = len(self.param_names)
        f0 = self.value(items, p_lin)
        g = np.zeros(K)
        for k in range(K):
            h = self._step(items, directions, k)
            if h is not None:
                def central(s, k=k):
                    return (self.value(self._shift(items, directions, k, s), p_lin)
                            - self.value(self._shift(items, directions, k, -s), p_lin)) / (2.0 * s)
                g[k] += (4.0 * central(h / 2.0) - central(h)) / 3.0
            if k in self.explicit:                      # the loss names this parameter itself
                d = 1e-6 * max(abs(p_lin[k]), 1e-8)

                def partial(s, k=k):
                    up, dn = p_lin.copy(), p_lin.copy()
                    up[k] += s
                    dn[k] -= s
                    return (self.value(items, up) - self.value(items, dn)) / (2.0 * s)
                g[k] += (4.0 * partial(d / 2.0) - partial(d)) / 3.0
        return f0, g


# ---------------------------------------------------------------------------------------
# theta -> model parameters, the way the engine applies them
# ---------------------------------------------------------------------------------------

class ParameterReplay:
    """Apply fitted parameters to a RoadRunner model exactly as ``Optimize.run_all`` does, and read the result.

    The engine resets the model, restores the parameter baseline, runs the replicate's
    ``Update_parameters``, sets the fitted values, then the ``parameter_hooks`` and
    ``Update_opt_parameters``. Replaying that and reading every plain parameter back
    gives the model parameters AMICI needs, and (by differencing the replay) the
    Jacobian of the model parameters with respect to the fitted ones -- whatever the
    hooks do in between.
    """

    def __init__(self, r, replicate, param_names, plain):
        self.r, self.replicate = r, replicate
        self.param_names, self.plain = list(param_names), list(plain)
        self.species = list(r.model.getFloatingSpeciesIds())

    def apply(self, p_lin):
        from pyantigen.engine.Optimize import restore_parameter_baseline, set_parameters_from_dict
        r, rep = self.r, self.replicate
        params = dict(zip(self.param_names, np.asarray(p_lin, dtype=float).tolist()))
        r.reset()
        restore_parameter_baseline(r)
        if rep.get("Update_parameters") is not None:
            rep["Update_parameters"](r, rep)
        set_parameters_from_dict(r, params)
        for hook in rep.get("parameter_hooks", []):
            hook(r, params)
        if rep.get("Update_opt_parameters") is not None:
            rep["Update_opt_parameters"](r, rep, params)

    def read(self, names):
        return np.array([float(self.r[n]) for n in names])

    def values(self, p_lin):
        """``{plain parameter: value}`` after the engine's updates."""
        self.apply(p_lin)
        return dict(zip(self.plain, self.read(self.plain)))

    def jacobian(self, p_lin, h_rel=1e-6):
        """``(values, J)``: J[m, k] = d(plain parameter m)/d(theta_k), central differences of the replay."""
        p_lin = np.asarray(p_lin, dtype=float)
        base = self.values(p_lin)
        J = np.zeros((len(self.plain), len(p_lin)))
        for k in range(len(p_lin)):
            h = h_rel * max(abs(p_lin[k]), 1e-8)
            up, dn = p_lin.copy(), p_lin.copy()
            up[k] += h
            dn[k] -= h
            vu = np.array(list(self.values(up).values()))
            vd = np.array(list(self.values(dn).values()))
            J[:, k] = (vu - vd) / (2.0 * h)
        self.apply(p_lin)
        return base, J

    def direct_species(self, p_lin, rtol=1e-9):
        """Species a hook set directly: ``{name: amount}`` where the model differs from what ``reset()`` derives.

        ``reset()`` re-evaluates a species' initial assignment from the current parameters,
        so those need no help; a value a hook wrote over that does, because AMICI never sees the hook.
        """
        self.apply(p_lin)
        mine = np.array(self.r.model.getFloatingSpeciesAmounts(), dtype=float)
        self.r.reset()
        derived = np.array(self.r.model.getFloatingSpeciesAmounts(), dtype=float)
        tol = rtol * np.maximum(np.maximum(np.abs(mine), np.abs(derived)), 1e-300)
        off = np.flatnonzero(np.abs(mine - derived) > tol)
        self.apply(p_lin)
        return {self.species[i]: float(mine[i]) for i in off}


# ---------------------------------------------------------------------------------------
# AMICI
# ---------------------------------------------------------------------------------------

def _tracked_window(settings):
    blocks = settings["simulation_blocks"]
    blocks = list(blocks.values()) if isinstance(blocks, dict) else list(blocks)
    tracked = [b for b in blocks if b.get("tracked", True)]
    if not tracked:
        raise AnalyticGradientError("no tracked simulation block")
    last = max(i for i, b in enumerate(blocks) if b.get("tracked", True))
    return (float(blocks[0]["start"]), float(tracked[0]["start"]), float(blocks[last]["end"]))


def _data_times(groups, replicates, models, sim, t_lo, t_hi):
    """Every finite data time the loss reads for *sim*, inside [t_lo, t_hi]."""
    from pyantigen.engine.Optimize import _resolve_obs_df
    times = set()
    for g in groups.values():
        for elem in g.get("loss_elements", []):
            sims = list(elem.get("simulations", [])) or [elem.get("simulation")]
            if sim not in sims:
                continue
            data_sim = elem.get("data_simulation") or sims[0]
            df_dict = models[data_sim]["df_dict"]
            lc_fn = elem.get("loss_config")
            lc = lc_fn(replicates[data_sim]) if callable(lc_fn) else lc_fn
            for obs in (lc or {}).get("observables", []):
                df = _resolve_obs_df(df_dict, obs)
                if df is None or obs["time_column"] not in df.columns:
                    continue
                t = np.asarray(df[obs["time_column"]], dtype=float)
                times.update(float(v) for v in t[np.isfinite(t) & (t >= t_lo) & (t <= t_hi)])
    return times


def freeze_stale_assignments(sbml, r, protected=(), varying=()):
    """The SBML with the parameters RoadRunner leaves stale made plain, at the values RoadRunner holds.

    A parameter with an initial assignment that (through other symbols) reads one of *varying* -- the
    model parameters the fitted ones move -- keeps, in the engine's in-place simulation, the value it was
    given when the model was built; RoadRunner's ``reset()`` re-derives species but not parameters. The
    returned SBML has that initial assignment removed and the parameter set to ``r``'s value, so it is a
    plain parameter whose number comes from the engine's replay like any other.

    *protected*: the variables the schedule assigns, and whatever their initial values are built from, are
    left alone (:meth:`AmiciGradient`'s inputs handle those). Returns ``(sbml, [frozen names])``.
    """
    import libsbml
    doc = libsbml.readSBMLFromString(sbml)
    model = doc.getModel()
    formulas = {}
    for i in range(model.getNumInitialAssignments()):
        a = model.getInitialAssignment(i)
        formulas[a.getSymbol()] = libsbml.formulaToL3String(a.getMath())
    for i in range(model.getNumRules()):
        rule = model.getRule(i)
        if rule.isAssignment() or rule.isRate():
            formulas.setdefault("rule:" + rule.getVariable(), libsbml.formulaToL3String(rule.getMath()))

    def tokens(sym, seen):
        for src in (formulas.get(sym), formulas.get("rule:" + sym)):
            for tok in set(re.findall(r"[A-Za-z_][A-Za-z_0-9]*", src or "")):
                if tok not in seen:
                    seen.add(tok)
                    tokens(tok, seen)
        return seen

    keep = set()
    for v in protected:
        keep |= tokens(v, {v})
    frozen = []
    for sym in list(formulas):
        if sym.startswith("rule:") or sym in keep:
            continue
        p = model.getParameter(sym)
        if p is None or model.getRuleByVariable(sym) is not None:
            continue
        if tokens(sym, set()) & set(varying):
            model.removeInitialAssignment(sym)
            p.setValue(float(r[sym]))
            p.setConstant(True)
            frozen.append(sym)
    return libsbml.writeSBMLToString(doc), frozen


class _SimGradient:
    """AMICI's view of one simulation: the compiled event-free model, the schedule, the parameter replay."""

    def __init__(self, name, replicate, model, models, groups, replicates, param_names, probes, *,
                 model_text, paths, cache_dir, rtol, atol, max_steps, verbose):
        from pyantigen.engine.Optimize import OptRoadRunnerProxy, remember_parameter_baseline
        from pyantigen.engine.Scaled_amici import (AmiciSegmentRunner, _plain_parameters,
                                                   compile_model, concentration_outputs)
        from pyantigen.generate.scaled_time import scale_model
        from pyantigen.generate.TelluriumGen import TelluriumGen

        self.name, self.replicate = name, replicate
        settings = replicate["Solver_settings"](replicate)
        schedule = settings.get("schedule")
        r = model["r"]
        if schedule is None:
            schedule, r = self._schedule_from_events(model, replicate, param_names, model_text,
                                                     paths, OptRoadRunnerProxy, remember_parameter_baseline,
                                                     TelluriumGen)
        self.schedule = schedule
        self.r = r
        sbml = r.getSBML()
        # Parameters defined by an initial assignment from a fitted parameter (Q_glymph = Q_CSF) are not
        # re-derived by RoadRunner's reset(): the engine integrates with the value they had when the
        # model was built. Freeze them, so AMICI integrates the same model the objective does.
        plain0 = _plain_parameters(sbml)
        varying = set()
        for point in probes:
            _v, J0 = ParameterReplay(r, replicate, param_names, plain0).jacobian(point)
            varying |= {p for p, hit in zip(plain0, np.abs(J0).max(axis=1) > 0) if hit}
        sbml, self.frozen = freeze_stale_assignments(
            sbml, r, protected=schedule.assigned_variables(), varying=varying)
        plain = _plain_parameters(sbml)
        self.replay = ParameterReplay(r, replicate, param_names, plain)

        # which model parameters the fitted ones reach: the union over a few points of the box
        reach = np.zeros((len(plain), len(param_names)), dtype=bool)
        direct = None
        for point in probes:
            _values, J = self.replay.jacobian(point)
            reach |= np.abs(J) > 0
            found = self.replay.direct_species(point)
            if direct is not None and found != direct:
                raise AnalyticGradientError(
                    f"{name}: a hook sets species initial values that depend on the fitted parameters "
                    f"({sorted(set(found) | set(direct))}); their sensitivities are not available")
            direct = found
        lost = [p for p, hit in zip(param_names, reach.any(axis=0)) if not hit]
        if lost:
            raise AnalyticGradientError(
                f"{name}: fitted parameter(s) {lost} reach no plain parameter of the model (a species "
                "initial value, or a parameter with an initial assignment?): the analytic gradient "
                "would be zero for them")
        self.free = [p for p, used in zip(plain, reach.any(axis=1)) if used]

        columns = [c for c in replicate["Observed_species"](r) if c != "time"]
        columns = list(dict.fromkeys(columns))
        usable, dropped = self._usable_columns(r, sbml, columns)
        self.columns = ["time"] + usable
        self.dropped = dropped

        scaled = scale_model(sbml, schedule)
        self.scaled = scaled
        outputs = concentration_outputs(sbml, usable)
        tag = hashlib.sha256(json.dumps([scaled.sbml, sorted(self.free), list(outputs.items())],
                                        sort_keys=True).encode()).hexdigest()[:16]
        directory = os.path.join(cache_dir, f"{name}_{tag}")
        module_name = re.sub(r"\W", "_", f"pag_{name}_{tag}")
        import amici
        if os.path.exists(os.path.join(directory, module_name, "__init__.py")):
            module = amici.import_model_module(module_name, directory)
        else:
            if verbose:
                print(f"  [amici] compiling {name}: {len(self.free)} sensitivity parameter(s), "
                      f"{len(usable)} output(s) -> {directory}", flush=True)
            module = compile_model(scaled, module_name, directory, free=self.free, outputs=outputs)
        self.runner = AmiciSegmentRunner(scaled, schedule, module, states=[], rtol=rtol, atol=atol,
                                         max_steps=max_steps)
        self.t_start, self.t_first, self.t_last = _tracked_window(settings)

    @staticmethod
    def _usable_columns(r, sbml, columns):
        import libsbml
        model = libsbml.readSBMLFromString(sbml).getModel()
        ids = ({s.getId() for s in model.getListOfSpecies()} | {p.getId() for p in model.getListOfParameters()}
               | {c.getId() for c in model.getListOfCompartments()})
        usable, dropped = [], []
        for c in columns:
            base = c[1:-1] if c.startswith("[") and c.endswith("]") else c
            (usable if base in ids else dropped).append(c)
        return usable, dropped

    @staticmethod
    def _schedule_from_events(model, replicate, param_names, model_text, paths, Proxy, remember,
                              TelluriumGen):
        """No schedule on the replicate: read it from the event text, and drop those events from the model."""
        events = model.get("events") or ""
        if "at (" not in events and "at(" not in events:
            raise AnalyticGradientError(
                "the replicate has no schedule and no events to read one from; "
                "an analytic gradient needs the inputs as an InputSchedule")
        r = model["r"]
        params = {p: float(r[p]) for p in r.getGlobalParameterIds()}
        try:
            schedule = InputSchedule.from_antimony_events(events, params, ignore_other=True)
        except ScheduleError as exc:
            raise AnalyticGradientError(f"cannot read the replicate's events as a schedule: {exc}") from exc
        rest = "\n".join(schedule.unparsed)
        r_free = TelluriumGen(model_text + "\n" + rest, paths)
        Proxy_r = Proxy(r_free, param_names)
        replicate["Update_parameters"](Proxy_r, replicate)
        remember(r_free)
        return schedule, r_free

    def tables(self, p_lin, groups, replicates, models):
        """``(SimTable, d/d theta array (rows, columns, K))`` at the data times, or raises."""
        p_lin = np.asarray(p_lin, dtype=float)
        values, J = self.replay.jacobian(p_lin)
        direct = self.replay.direct_species(p_lin)
        theta = dict(values)
        inputs = self._engine_initial_inputs(theta)
        times = _data_times(groups, replicates, models, self.name, self.t_first, self.t_last)
        T = np.array(sorted(times | {self.t_first, self.t_last}))
        wrt = self.free
        Y, SY = self.runner.run_outputs(theta, self.t_start, self.t_last, list(T), wrt=wrt,
                                        initial=direct or None, initial_inputs=inputs)
        # d(output)/d(fitted) = d(output)/d(model parameter) . d(model parameter)/d(fitted)
        Jfree = J[[self.replay.plain.index(p) for p in wrt], :]
        data = np.zeros((len(T), len(self.columns)))
        data[:, 0] = T
        data[:, 1:] = Y
        D = np.zeros((len(T), len(self.columns), J.shape[1]))
        D[:, 1:, :] = np.einsum("tcm,mk->tck", SY, Jfree)
        return SimTable(self.columns, data), D

    def _engine_initial_inputs(self, theta):
        """What each assigned input holds before its first step, as the engine's model holds it.

        RoadRunner's ``reset()`` re-derives a species' initial assignment but not a parameter's, so an
        input such as ``Q_SN = Q_CSF*f_SN`` keeps the value it had at the start. Where that is the
        model's own expression evaluated now, the expression is used (and differentiated); where it is
        not, the engine's number is, with no sensitivity.
        """
        out = {}
        for v, expr in self.scaled.initial_inputs.items():
            engine = float(self.r[v])
            if isinstance(expr, str):
                try:
                    model_value = self.schedule.evaluate(expr, lambda n: float(theta.get(n, self.r[n])))
                except Exception:
                    model_value = None
                if model_value is not None and np.isclose(model_value, engine, rtol=1e-9, atol=1e-12):
                    out[v] = expr
                    continue
            out[v] = engine
        return out


class AmiciGradient:
    """``x (opt space) -> d NLL / d x`` for the engine's concentrated NLL, from AMICI sensitivities.

    Parameters are the objects ``run_optimization_from_groups`` builds: ``models`` (sim -> dict with the
    RoadRunner model ``"r"``, its ``"df_dict"`` and ``"events"``), ``replicates``, the spec's ``groups``,
    ``param_names``, ``scales`` ("log10" or linear, per parameter), and ``x0_lin`` / ``bounds_lin`` (the
    points at which the parameter reach is probed). *settings* is the ``analytic_gradient`` dict of the
    spec's ``optimizer_kwargs``: ``rtol``, ``atol``, ``max_steps`` (AMICI), ``rel_step`` (the loss
    differentiation), ``cache_dir`` (compiled models).
    """

    DEFAULTS = {"provider": "amici", "rtol": 1e-10, "atol": 1e-12, "max_steps": 2_000_000,
                "rel_step": 1e-5, "cache_dir": None, "verbose": True}

    @classmethod
    def resolve(cls, optimizer_kwargs):
        given = (optimizer_kwargs or {}).get("analytic_gradient")
        if not given:
            return None
        if given is True:
            given = {}
        unknown = sorted(set(given) - set(cls.DEFAULTS))
        if unknown:
            raise ValueError(f"unknown analytic_gradient key(s) {unknown}; expected a subset of "
                             f"{sorted(cls.DEFAULTS)}")
        out = {**cls.DEFAULTS, **given}
        if out["provider"] != "amici":
            raise ValueError("analytic_gradient['provider'] must be 'amici'")
        return out

    def __init__(self, settings, *, models, replicates, groups, param_names, scales, x0_lin, bounds_lin,
                 model_text, paths):
        self.settings = settings
        self.models, self.replicates, self.groups = models, replicates, groups
        self.param_names, self.scales = list(param_names), list(scales)
        self.loss = LossGradient(groups, replicates, param_names, rel_step=settings["rel_step"])
        cache_dir = settings["cache_dir"] or os.path.join(
            (paths or {}).get("plot_path") or tempfile.gettempdir(), "amici_models")
        os.makedirs(cache_dir, exist_ok=True)
        probes = self._probe_points(x0_lin, bounds_lin)
        self.sims = {}
        for name, rep in replicates.items():
            self.sims[name] = _SimGradient(
                name, rep, models[name], models, groups, replicates, param_names, probes,
                model_text=model_text, paths=paths, cache_dir=cache_dir, rtol=settings["rtol"],
                atol=settings["atol"], max_steps=settings["max_steps"], verbose=settings["verbose"])
        self.n_calls = 0

    @staticmethod
    def _probe_points(x0_lin, bounds_lin):
        """x0 and the two corners of the box: where the fitted parameters' reach is read off."""
        x0 = np.asarray(x0_lin, dtype=float)
        points = [x0]
        if bounds_lin:
            lo = np.array([x0[i] if b is None or b[0] is None else b[0] for i, b in enumerate(bounds_lin)], float)
            hi = np.array([x0[i] if b is None or b[1] is None else b[1] for i, b in enumerate(bounds_lin)], float)
            points += [p for p in (lo, hi) if np.all(np.isfinite(p)) and not np.array_equal(p, x0)]
        return points

    def linear_gradient(self, p_lin):
        """``(nll, d nll / d p_lin)`` evaluated on AMICI's own simulation."""
        items, dirs = {}, {}
        for name, sim in self.sims.items():
            table, D = sim.tables(p_lin, self.groups, self.replicates, self.models)
            items[name] = {"results": table, "data": self.models[name]["df_dict"],
                           "replicate": self.replicates[name]}
            dirs[name] = D
        self.n_calls += 1
        return self.loss.gradient(items, dirs, p_lin)

    def __call__(self, x_opt):
        """The gradient in optimizer space (a log10 parameter's derivative is taken with respect to log10 p)."""
        from pyantigen.engine.Optimize import _to_linear
        x_opt = np.asarray(x_opt, dtype=float)
        p_lin = _to_linear(x_opt, self.scales)
        _f, g_lin = self.linear_gradient(p_lin)
        chain = np.array([np.log(10.0) * p if sc == "log10" else 1.0
                          for p, sc in zip(np.asarray(p_lin, dtype=float), self.scales)])
        return g_lin * chain


def build_gradient(optimizer_kwargs, **context):
    """The analytic gradient asked for in *optimizer_kwargs*, or None.

    A problem it cannot handle raises ``Multistart.GradientSetupError`` here, before any fit.
    """
    settings = AmiciGradient.resolve(optimizer_kwargs)
    if settings is None:
        return None
    try:
        return AmiciGradient(settings, **context)
    except GradientSetupError:
        raise
    except (AnalyticGradientError, ScheduleError, ImportError, KeyError) as exc:
        raise GradientSetupError(f"analytic_gradient cannot be set up: {type(exc).__name__}: {exc}") from exc

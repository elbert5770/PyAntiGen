"""Lower a Study + Estimation to the Engine's current inputs.

The Engine still consumes the 1.x shapes: an object with ``.replicates``
(one dict per simulation, carrying Events / Data / Solver_settings /
Observed_species / Update_parameters / Update_opt_parameters hooks) and an
Optimization-shaped spec whose ``groups`` list loss elements. This module
produces exactly those from a Study, so a Study runs on today's Engine with
no Engine change, and its objective can be checked against the 1.x spec it
replaces, value for value.

Every hook is an instance of a module-level class carrying only its own
occasion's payload, so replicates pickle to pool workers without dragging
the whole study along.

Lowering rules
--------------
* one replicate per occasion, labelled with the occasion id; every attribute
  of the occasion is also a key of the replicate, so 1.x event and solver
  functions that read ``replicate["dose"]`` keep working;
* one loss element per (occasion, measured observable) -- or per
  (numerator, denominator) pair for a contrast, as a composite element;
* ``Noise.pool`` becomes the element's ``sigma_block``;
* ``resolve()`` is applied twice: its fixed part in Update_parameters (before
  the pre-dose block), and in full, fitted values included, in
  Update_opt_parameters (after the Engine's set_parameters). Rules therefore
  always land last.
"""
from dataclasses import dataclass, field

from .assay import CONTRAST_OPS
from .design import _matches
from .params import by_name, resolve
from .refs import from_ref


@dataclass
class Estimation:
    """What to fit, against which measurements, and how.

    params  names of Study.params to estimate (default: every estimated one)
    on      Study.select filter restricting which measured occasions are
            scored (default: all). A contrast is kept when its numerator is.
    passive occasion ids simulated but not scored, e.g. for plots; default is
            none, as in 1.x.
    """
    name: str
    params: list = None
    on: dict = None
    method: str = "Nelder-Mead"
    optimizer_kwargs: dict = field(default_factory=dict)
    n_starts: int = 1
    start_seed: int = None
    search_decades: float = None
    passive: list = field(default_factory=list)


@dataclass
class LoweredSpec:
    """The Optimization shape the Engine reads (recognised by its fields)."""
    param_names: list
    x0: dict
    bounds: dict
    method: str = "Nelder-Mead"
    optimizer_kwargs: dict = field(default_factory=dict)
    group_normalization: str = "mean_over_groups"
    groups: dict = field(default_factory=dict)
    passive_simulations: list = field(default_factory=list)
    parameter_scale: object = "auto"
    search_decades: float = None
    n_starts: int = 1
    start_seed: int = None


class LoweredExperiment:
    def __init__(self, replicates):
        self.replicates = replicates


# --- hooks (module level, picklable) ----------------------------------------

class ApplyFixed:
    """Update_parameters: the fixed part of resolve(), before the pre-dose block,
    then the protocol's own update_parameters hook if it has one."""

    def __init__(self, values, project_hook=None):
        self.values = dict(values)
        self.project_hook = project_hook

    def __call__(self, r, replicate, mode=None):
        for k, v in self.values.items():
            r[k] = v
        if self.project_hook is not None:
            self.project_hook(r, replicate)


class ApplyResolved:
    """Update_opt_parameters: resolve() in full, after the fitted values land.

    fixed    what factor levels and fixed Params set
    fitted   {optimizer name: model parameter} for this occasion's by-level
             Params (global Params map to themselves)
    rules    [(target, value)] that apply to this occasion, in order
    """

    def __init__(self, fixed, fitted, rules, project_hook=None):
        self.fixed = dict(fixed)
        self.fitted = dict(fitted)
        self.rules = list(rules)
        self.project_hook = project_hook

    def values(self, parameters):
        out = dict(self.fixed)
        for opt_name, model_name in self.fitted.items():
            if parameters and opt_name in parameters:
                out[model_name] = parameters[opt_name]
        for target, value in self.rules:
            out[target] = value
        return out

    def __call__(self, r, replicate, parameters):
        for k, v in self.values(parameters).items():
            r[k] = v
        if self.project_hook is not None:
            self.project_hook(r, replicate, parameters)


class LoadData:
    """Data: the protocol's inputs, plus the rows of every DataSource scored
    on this occasion.

    protocol_data, if given, is called first with the replicate; its tables
    are passed through unchanged (the event builder reads them) and are what
    DataSource(input=...) selects from. sources is
    [(key, attrs, DataSource, column_name)]; each is read with the occasion's
    (or, for a contrast, the numerator's) attributes and stored under its
    key with its value column renamed to column_name.
    """

    def __init__(self, sources, protocol_data=None):
        self.sources = list(sources)
        self.protocol_data = protocol_data

    def __call__(self, replicate, data_path):
        inputs = (self.protocol_data(replicate, data_path)
                  if self.protocol_data is not None else {})
        cache, out = {}, dict(inputs)
        for key, attrs, src, col in self.sources:
            df = src.rows_for(attrs, data_path, _cache=cache, inputs=inputs)
            out[key] = df.rename(columns={"value": col})
        return out


# --- lowering ---------------------------------------------------------------

def _measured_key(assay, measured):
    return f"{assay}.{measured.name}"


def _obs_cfg(assay_name, m, attrs):
    cfg = {"observed_variable": m.model.engine_form(attrs),
           "data_column": m.name,
           "time_column": "time",
           "data_dict_key": _measured_key(assay_name, m)}
    cfg.update(m.noise.engine_keys())
    return cfg


def _sigma_block(study, assay_name, m, occ):
    pool = m.noise.pool
    if pool is None:
        return None
    if pool == "all":
        return f"{assay_name}.{m.name}"
    pooled = set(pool)
    unknown = pooled - set(study.factors) - {"subject"}
    if unknown:
        raise KeyError(f"assay {assay_name!r}/{m.name}: pool names unknown factor(s) {sorted(unknown)}")
    keep = [("subject", occ.subject)] if "subject" not in pooled else []
    keep += [(f, lev) for f, lev in occ.levels if f not in pooled]
    return f"{assay_name}.{m.name}|" + "|".join(f"{k}={v}" for k, v in keep)


def lower(study, est):
    """Return (LoweredExperiment, LoweredSpec) for *est* on *study*."""
    # parameters
    names = est.params if est.params is not None else [
        n for n, p in study.params.items() if p.estimate]
    param_names, x0, bounds, scale = [], {}, {}, {}
    for n in names:
        p = study.params[n]
        if not p.estimate:
            raise ValueError(f"estimation {est.name!r}: param {n!r} is fixed (estimate=False)")
        if p.by is None:
            items = [(n, None)]
        else:
            items = [(by_name(n, p.by, lev), lev) for lev in p._used_levels(study)]
        for opt_name, lev in items:
            param_names.append(opt_name)
            x0[opt_name] = p.x0_for(lev)
            b = p.bounds_for(lev)
            bounds[opt_name] = tuple(b) if b is not None else None
            scale[opt_name] = p.scale
    if any(b is None for b in bounds.values()):
        if not all(b is None for b in bounds.values()):
            raise ValueError(f"estimation {est.name!r}: give bounds for every parameter or none")
        bounds = None

    scored = None
    if est.on:
        scored = {o.id for o in study.select(**est.on)}

    # loss elements and the data each occasion needs
    groups = {}
    data_needs = {oid: [] for oid in study.occasions}
    for meas in study.measurements:
        assay = study.assays[meas.assay]
        elems = groups.setdefault(meas.assay if not meas.contrast else f"{meas.assay}:{meas.contrast}",
                                  {"loss_elements": []})["loss_elements"]
        if meas.contrast:
            c = study.contrasts[meas.contrast]
            op = CONTRAST_OPS[c.op]
            for num, den in c.pairs(study):
                if scored is not None and num.id not in scored:
                    continue
                attrs = study.attributes(num)
                for m in assay.observables:
                    if not all(_matches(attrs.get(k), v) for k, v in m.only.items()):
                        continue
                    key = _measured_key(meas.assay, m)
                    data_needs[num.id].append((key, attrs, m.data, m.name))
                    e = {"type": "composite", "simulations": [num.id, den.id],
                         "data_simulation": num.id, "aggregation": op,
                         "loss_config": {"observables": [_obs_cfg(meas.assay, m, attrs)]}}
                    blk = _sigma_block(study, meas.assay, m, num)
                    if blk:
                        e["sigma_block"] = blk
                    elems.append(e)
        else:
            occs = study.select(**meas.on) if meas.on else list(study.occasions.values())
            for occ in occs:
                if scored is not None and occ.id not in scored:
                    continue
                attrs = study.attributes(occ)
                for m in assay.observables:
                    if not all(_matches(attrs.get(k), v) for k, v in m.only.items()):
                        continue
                    key = _measured_key(meas.assay, m)
                    data_needs[occ.id].append((key, attrs, m.data, m.name))
                    e = {"simulation": occ.id,
                         "loss_config": {"observables": [_obs_cfg(meas.assay, m, attrs)]}}
                    blk = _sigma_block(study, meas.assay, m, occ)
                    if blk:
                        e["sigma_block"] = blk
                    elems.append(e)
    groups = {g: v for g, v in groups.items() if v["loss_elements"]}
    if not groups:
        raise ValueError(f"estimation {est.name!r} scores nothing")

    # replicates
    fitted_set = set(param_names)
    replicates = {}
    for occ in study.occasions.values():
        proto = study.protocols[occ.protocol]
        events, solver, observed = proto.resolved()
        p_data, p_update, p_update_opt = proto.hooks()
        attrs = study.attributes(occ)
        fixed = resolve(study, occ, None)
        lv = occ.level_dict
        fitted = {}
        for p in study.params.values():
            if not p.estimate:
                continue
            opt_name = p.name if p.by is None else by_name(p.name, p.by, lv.get(p.by))
            if opt_name in fitted_set:
                fitted[opt_name] = p.name
        rules = [(r.target, r.value) for r in study.rules if r.applies(attrs)]
        rep = dict(attrs)
        rep.update({
            "Label": occ.id,
            "Events": events,
            "Data": LoadData(_dedupe(data_needs[occ.id]), p_data),
            "Observed_species": observed,
            "Solver_settings": solver,
            "Update_parameters": ApplyFixed(fixed, p_update),
            "Update_opt_parameters": ApplyResolved(fixed, fitted, rules, p_update_opt),
        })
        replicates[occ.id] = rep

    spec = LoweredSpec(
        param_names=param_names, x0=x0, bounds=bounds, method=est.method,
        optimizer_kwargs=dict(est.optimizer_kwargs), groups=groups,
        passive_simulations=list(est.passive),
        parameter_scale=scale, search_decades=est.search_decades,
        n_starts=est.n_starts, start_seed=est.start_seed)
    return LoweredExperiment(replicates), spec


def _dedupe(needs):
    seen, out = set(), []
    for key, attrs, src, col in needs:
        if key not in seen:
            seen.add(key)
            out.append((key, attrs, src, col))
    return out

"""Lower an Optimization (and the Studies it uses) to the Engine's current inputs.

The Engine still consumes the 1.x shapes: an object with ``.replicates``
(one dict per simulation, carrying Events / Data / Solver_settings /
Observed_species / Update_parameters / Update_opt_parameters hooks) and an
Optimization-shaped spec whose ``groups`` list loss elements. This module
produces exactly those from a Study, so a Study runs on today's Engine with
no Engine change, and its objective can be checked against the 1.x spec it
replaces, value for value.

Every hook is an instance of a module-level class carrying only its own
simulation's payload, so replicates pickle to pool workers without dragging
the whole study along.

Lowering rules
--------------
* one replicate per simulation, labelled with the simulation id; every attribute
  of the simulation is also a key of the replicate, so 1.x event and solver
  functions that read ``replicate["dose"]`` keep working;
* one loss element per (simulation, measured observable) -- or per
  (numerator, denominator) pair for a contrast, as a composite element;
* ``Noise.pool`` becomes the element's ``sigma_block``;
* ``resolve()`` is applied twice: its fixed part in Update_parameters (before
  the pre-dose block), and in full, fitted values included, in
  Update_opt_parameters (after the Engine's set_parameters). Rules therefore
  always land last.
"""
from dataclasses import dataclass, field

from .assay import CONTRAST_OPS, peak_normalize
from .design import _matches
from .params import by_name, resolve
from .refs import from_ref


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
    fitted   {optimizer name: model parameter} for this simulation's by-level
             Params (global Params map to themselves)
    rules    [(target, value)] that apply to this simulation, in order
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
    on this simulation.

    protocol_data, if given, is called first with the replicate; its tables
    are passed through unchanged (the event builder reads them) and are what
    DataSource(input=...) selects from.

    sources is [(key, attrs, DataSource, column_name, loader)]. Each is read
    with ``attrs`` -- this simulation's, or, for either side of a contrast
    pair, the NUMERATOR's -- and stored under ``key`` with its value column
    renamed. ``loader`` is the protocol loader of the simulation ``attrs``
    describe, needed when that is not this simulation (a contrast's
    denominator reading its numerator's table).
    """

    def __init__(self, sources, protocol_data=None):
        self.sources = list(sources)
        self.protocol_data = protocol_data

    def __call__(self, replicate, data_path):
        inputs = (self.protocol_data(replicate, data_path)
                  if self.protocol_data is not None else {})
        own = replicate.get("simulation")
        cache, out, other_inputs = {}, dict(inputs), {}
        for key, attrs, src, col, loader in self.sources:
            src_inputs = inputs
            if src.input is not None and attrs.get("simulation") != own:
                sid = attrs.get("simulation")
                if sid not in other_inputs:
                    other_inputs[sid] = loader(dict(attrs, Label=sid), data_path)
                src_inputs = other_inputs[sid]
            df = src.rows_for(attrs, data_path, _cache=cache, inputs=src_inputs)
            out[key] = df.rename(columns={"value": col})
        return out


# --- lowering ---------------------------------------------------------------

def _measured_key(assay, measured, pair=None):
    """Data-table key. A contrast pair gets its own key, because the Engine
    evaluates EACH simulation of a composite at the times of that
    simulation's own copy of the table: both sides must hold the numerator's
    rows, and one denominator can serve several numerators."""
    key = f"{assay}.{measured.name}"
    return key if pair is None else f"{key}|{pair}"


def _obs_cfg(assay_name, m, attrs, pair=None):
    cfg = {"observed_variable": m.model.engine_form(attrs),
           "data_column": m.name,
           "time_column": "time",
           "data_dict_key": _measured_key(assay_name, m, pair)}
    cfg.update(m.noise.engine_keys())
    return cfg


def _sigma_block(study, assay_name, m, sim):
    pool = m.noise.pool
    if pool is None:
        return None
    if pool == "all":
        return f"{assay_name}.{m.name}"
    pooled = set(pool)
    unknown = pooled - set(study.factors) - {"subject"}
    if unknown:
        raise KeyError(f"assay {assay_name!r}/{m.name}: pool names unknown factor(s) {sorted(unknown)}")
    keep = [("subject", sim.subject)] if "subject" not in pooled else []
    keep += [(f, lev) for f, lev in sim.levels if f not in pooled]
    return f"{assay_name}.{m.name}|" + "|".join(f"{k}={v}" for k, v in keep)


def lower(opt):
    """Return (LoweredExperiment, LoweredSpec) for Optimization *opt*.

    Simulations from every study it uses become replicates; their ids must be
    unique across those studies (they are the Engine's labels, and some 1.x
    event builders still key off them). Loss elements are emitted use by use,
    then simulation by simulation, then assay by assay -- the order the
    Engine sums blocks in. A contrast's element belongs to its numerator.
    """
    studies = opt.studies
    owner = {}
    for st in studies:
        for sid in st.simulations:
            if sid in owner and owner[sid] is not st:
                raise ValueError(f"optimization {opt.name!r}: simulation id {sid!r} is in "
                                 f"both {owner[sid].name!r} and {st.name!r}")
            owner[sid] = st

    # parameters
    param_names, x0, bounds, scale = [], {}, {}, {}
    for p in opt.params:
        levels = [None] if p.by is None else p.levels(studies)
        if p.by is not None and not levels:
            raise ValueError(f"optimization {opt.name!r}: Param {p.name!r} by={p.by!r} "
                             "matches no factor level in the studies it uses")
        for lev in levels:
            opt_name = p.name if lev is None else by_name(p.name, p.by, lev)
            param_names.append(opt_name)
            x0[opt_name] = p.x0_for(lev)
            b = p.bounds_for(lev)
            bounds[opt_name] = tuple(b) if b is not None else None
            scale[opt_name] = p.scale
    if any(b is None for b in bounds.values()):
        if not all(b is None for b in bounds.values()):
            raise ValueError(f"optimization {opt.name!r}: give bounds for every parameter or none")
        bounds = None

    # loss elements and the data each simulation needs
    elems = []
    data_needs = {sid: [] for sid in owner}
    full = {id(st): st.scoring() for st in studies}
    passive = []
    for u in opt.uses:
        st = u.study
        if not u.score:
            passive += [s.id for s in u.simulations() if s.id not in passive]
            continue
        scoring = u.scoring(full[id(st)])
        for sim in st.simulations.values():
            attrs = st.attributes(sim)
            for assay, partner, only_obs in scoring[sim.id]:
                for m in assay.observables:
                    if only_obs is not None and m.name not in only_obs:
                        continue
                    if not all(_matches(attrs.get(k), v) for k, v in m.only.items()):
                        continue
                    if partner is not None:
                        if m.normalize:
                            raise ValueError(
                                f"assay {assay.name!r}/{m.name}: normalize={m.normalize!r} is "
                                "not supported on a contrast; normalize the result instead")
                        pair = f"{sim.id}/{partner.id}"
                        key = _measured_key(assay.name, m, pair)
                        loader = st.protocols[sim.protocol].hooks()[0]
                        data_needs[sim.id].append((key, attrs, m.data, m.name, loader))
                        data_needs[partner.id].append((key, attrs, m.data, m.name, loader))
                        e = {"type": "composite", "simulations": [sim.id, partner.id],
                             "data_simulation": sim.id,
                             "aggregation": CONTRAST_OPS[assay.contrast.op],
                             "loss_config": {"observables": [_obs_cfg(assay.name, m, attrs, pair)]}}
                    else:
                        key = _measured_key(assay.name, m)
                        data_needs[sim.id].append((key, attrs, m.data, m.name, None))
                        if m.normalize == "peak":
                            # The Engine's one hook that sees predictions
                            # already at the data times is a composite's
                            # aggregation.
                            e = {"type": "composite", "simulations": [sim.id],
                                 "data_simulation": sim.id, "aggregation": peak_normalize,
                                 "loss_config": {"observables": [_obs_cfg(assay.name, m, attrs)]}}
                        else:
                            e = {"simulation": sim.id,
                                 "loss_config": {"observables": [_obs_cfg(assay.name, m, attrs)]}}
                    blk = _sigma_block(st, assay.name, m, sim)
                    if blk:
                        e["sigma_block"] = f"{st.name}:{blk}"
                    elems.append(e)
    if not elems:
        raise ValueError(f"optimization {opt.name!r} scores nothing")
    groups = {opt.name: {"loss_elements": elems}}

    # replicates, for every simulation of every study used
    fitted_set = set(param_names)
    replicates = {}
    for st in studies:
        for sim in st.simulations.values():
            proto = st.protocols[sim.protocol]
            events, solver, observed = proto.resolved()
            p_data, p_update, p_update_opt = proto.hooks()
            attrs = st.attributes(sim)
            fixed = resolve(st, sim, None)
            lv = sim.level_dict
            fitted = {}
            for p in opt.params:
                if p.by is not None and p.by not in lv:
                    continue
                opt_name = p.name if p.by is None else by_name(p.name, p.by, lv[p.by])
                if opt_name in fitted_set:
                    fitted[opt_name] = p.name
            rules = [(r.target, r.value) for r in st.rules if r.applies(attrs)]
            rep = dict(attrs)
            rep.update({
                "Label": sim.id,
                "Events": events,
                "Data": LoadData(_dedupe(data_needs[sim.id]), p_data),
                "Observed_species": observed,
                "Solver_settings": solver,
                "Update_parameters": ApplyFixed(fixed, p_update),
                "Update_opt_parameters": ApplyResolved(fixed, fitted, rules, p_update_opt),
            })
            replicates[sim.id] = rep

    spec = LoweredSpec(
        param_names=param_names, x0=x0, bounds=bounds, method=opt.method,
        optimizer_kwargs=dict(opt.optimizer_kwargs), groups=groups,
        passive_simulations=passive,
        parameter_scale=scale, search_decades=opt.search_decades,
        n_starts=opt.n_starts, start_seed=opt.start_seed)
    return LoweredExperiment(replicates), spec


def _dedupe(needs):
    seen, out = set(), []
    for need in needs:
        if need[0] not in seen:
            seen.add(need[0])
            out.append(need)
    return out
